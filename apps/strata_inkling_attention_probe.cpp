#include "strata/cuda_backend.hpp"
#include "strata/numerics.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::uint32_t rows = 512U;
    std::uint32_t repeats = 20U;
    int device = 0;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--rows" && index + 1 < argc) {
            rows = static_cast<std::uint32_t>(std::stoul(argv[++index]));
        } else if (argument == "--repeats" && index + 1 < argc) {
            repeats = static_cast<std::uint32_t>(std::stoul(argv[++index]));
        } else if (argument == "--device" && index + 1 < argc) {
            device = std::stoi(argv[++index]);
        } else {
            std::fprintf(stderr,
                         "usage: strata-inkling-attention-probe "
                         "[--rows N] [--repeats N] [--device N]\n");
            return 2;
        }
    }
    if (rows == 0U || repeats == 0U) return 2;

    constexpr std::uint32_t query_heads = 32U;
    constexpr std::uint32_t kv_heads = 8U;
    constexpr std::uint32_t head_dim = 128U;
    constexpr std::uint32_t bias_extent = 1024U;
    constexpr std::uint32_t query_elements = query_heads * head_dim;
    constexpr std::uint32_t kv_elements = kv_heads * head_dim;
    constexpr float scale = 1.0F / 128.0F;

    std::vector<float> query(query_elements);
    std::vector<std::uint16_t> keys(static_cast<std::size_t>(rows) * kv_elements);
    std::vector<std::uint16_t> values(keys.size());
    std::vector<float> decoded_keys(keys.size());
    std::vector<float> decoded_values(values.size());
    std::vector<float> bias(query_heads * bias_extent);
    for (std::size_t index = 0U; index < query.size(); ++index) {
        query[index] = static_cast<float>(static_cast<int>(index % 31U) - 15) /
                       64.0F;
    }
    for (std::size_t index = 0U; index < keys.size(); ++index) {
        const float key = static_cast<float>(static_cast<int>(index % 29U) - 14) /
                          64.0F;
        const float value =
            static_cast<float>(static_cast<int>(index % 23U) - 11) / 32.0F;
        keys[index] = strata::bf16_encode(key);
        values[index] = strata::bf16_encode(value);
        decoded_keys[index] =
            std::bit_cast<float>(static_cast<std::uint32_t>(keys[index]) << 16U);
        decoded_values[index] = std::bit_cast<float>(
            static_cast<std::uint32_t>(values[index]) << 16U);
    }
    for (std::size_t index = 0U; index < bias.size(); ++index) {
        bias[index] = static_cast<float>(static_cast<int>(index % 17U) - 8) /
                      256.0F;
    }

    std::vector<float> expected(query_elements);
    std::vector<float> scores(rows);
    const auto cpu_started = std::chrono::steady_clock::now();
    const auto heads_per_kv = query_heads / kv_heads;
    for (std::uint32_t head = 0U; head < query_heads; ++head) {
        const auto kv_head = head / heads_per_kv;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::uint32_t row = 0U; row < rows; ++row) {
            float dot = 0.0F;
            for (std::uint32_t element = 0U; element < head_dim; ++element) {
                dot += query[static_cast<std::size_t>(head) * head_dim + element] *
                       decoded_keys[(static_cast<std::size_t>(row) * kv_heads +
                                     kv_head) * head_dim + element];
            }
            float score = dot * scale;
            const auto distance = rows - 1U - row;
            if (distance < bias_extent) {
                score += bias[static_cast<std::size_t>(head) * bias_extent +
                              distance];
            }
            scores[row] = score;
            maximum = std::max(maximum, score);
        }
        float denominator = 0.0F;
        for (auto& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        for (auto& score : scores) score /= denominator;
        for (std::uint32_t element = 0U; element < head_dim; ++element) {
            float accumulator = 0.0F;
            for (std::uint32_t row = 0U; row < rows; ++row) {
                accumulator +=
                    scores[row] *
                    decoded_values[(static_cast<std::size_t>(row) * kv_heads +
                                    kv_head) * head_dim + element];
            }
            expected[static_cast<std::size_t>(head) * head_dim + element] =
                accumulator;
        }
    }
    const double cpu_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - cpu_started).count();

    strata::CudaBackend backend;
    const std::vector<int> devices{device};
    auto status = backend.initialize(devices, true);
    if (!status.ok()) return 1;
    strata::CudaBuffer cache;
    status = backend.allocate_buffer(
        device, 2ULL * rows * kv_elements * sizeof(std::uint16_t), cache);
    if (!status.ok()) return 1;
    status = backend.upload_gemma4_kv(
        cache, keys, values, 0U, rows, kv_elements);
    if (!status.ok()) return 1;

    strata::CudaBf16KvAttentionRequest request;
    request.cache = &cache;
    request.queries = query;
    request.next_keys = std::span<const std::uint16_t>(keys).last(kv_elements);
    request.next_values = std::span<const std::uint16_t>(values).last(kv_elements);
    request.relative_bias = bias;
    request.query_heads = query_heads;
    request.key_value_heads = kv_heads;
    request.head_dim = head_dim;
    request.capacity_rows = rows;
    request.cache_start = 0U;
    request.cached_rows = rows;
    request.position = rows - 1U;
    request.relative_bias_extent = bias_extent;
    request.scale = scale;
    std::vector<float> actual(query_elements);
    status = backend.bf16_kv_attention(device, request, actual);
    if (!status.ok()) return 1;
    float maximum_error = 0.0F;
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        maximum_error =
            std::max(maximum_error, std::fabs(actual[index] - expected[index]));
    }

    const auto before = backend.stats().flash_attention_nanoseconds;
    for (std::uint32_t repetition = 0U; repetition < repeats; ++repetition) {
        status = backend.bf16_kv_attention(device, request, actual);
        if (!status.ok()) return 1;
    }
    const auto after = backend.stats().flash_attention_nanoseconds;
    std::printf("rows=%u cpu_ms=%.3f gpu_ms=%.3f speedup=%.3f "
                "max_abs_error=%.9g kv_mib=%.3f\n",
                rows, cpu_ms,
                static_cast<double>(after - before) / 1.0e6 / repeats,
                cpu_ms / (static_cast<double>(after - before) / 1.0e6 / repeats),
                maximum_error,
                static_cast<double>(cache.device_bytes()) / (1024.0 * 1024.0));
    return maximum_error <= 1.0e-6F ? 0 : 1;
}
