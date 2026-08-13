// Times the physical-format Lightning Indexer at the candidate widths a DSV4
// decode step actually produces, so the indexer term in
//     tau = max_r W_r/B_r + Sigma_serial
// is measured rather than assumed.
//
// The width that matters is context/4: the learned-index compressor runs at
// ratio 4, so a 1,048,576-token context presents 262,144 candidates per layer
// and 43 layers of that per decoded token. Reporting is per layer and per
// token, because the budget is stated per token.
//
// Usage: strata-dsv4-index-probe [device] [repetitions]

#include "strata/cuda_backend.hpp"
#include "strata/deepseek_kv_cache.hpp"
#include "strata/deepseek_ops.hpp"
#include "strata/dsv4_attention_kv.hpp"
#include "strata/model_adapter.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint32_t kHeads =
    strata::kDeepSeekV4ExecutionContract.index_heads;
constexpr std::uint32_t kHeadDim =
    strata::kDeepSeekV4ExecutionContract.index_head_dim;
constexpr std::uint32_t kTopK =
    strata::kDeepSeekV4ExecutionContract.index_topk;
constexpr std::uint32_t kBlockRows = strata::kDsv4PhysicalKvBlockRows;

// The sparse indexer engages only where the compressor runs at ratio 4
// (see the ratio == 4 guard in the runtime); the ratio-128 layers attend
// their compressed history densely and never reach the indexer. Counting all
// 43 layers overstates the per-token term by a factor of two.
constexpr std::uint32_t indexer_layers() {
    std::uint32_t count = 0U;
    for (std::uint32_t layer = 0U;
         layer < strata::kDeepSeekV4ExecutionContract.layer_count; ++layer) {
        if (strata::kDeepSeekV4ExecutionContract.compression_ratios[layer] ==
            4U) {
            ++count;
        }
    }
    return count;
}

constexpr std::uint32_t kLayers = indexer_layers();

std::uint32_t parse(std::string_view text, std::uint32_t fallback) {
    std::uint32_t value = 0U;
    const auto* end = text.data() + text.size();
    const auto parsed = std::from_chars(text.data(), end, value);
    return parsed.ec == std::errc{} && parsed.ptr == end ? value : fallback;
}

double median(std::vector<double>& samples) {
    std::sort(samples.begin(), samples.end());
    const auto middle = samples.size() / 2U;
    return samples.size() % 2U == 1U
        ? samples[middle]
        : (samples[middle - 1U] + samples[middle]) / 2.0;
}

}  // namespace

int main(int argc, char** argv) {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) {
        std::cerr << "no CUDA device available\n";
        return 1;
    }
    const int device = argc > 1
        ? static_cast<int>(parse(argv[1], static_cast<std::uint32_t>(
                                              devices.front())))
        : devices.front();
    const auto repetitions = argc > 2 ? parse(argv[2], 5U) : 5U;

    strata::CudaBackend backend;
    const std::array<int, 1> selected{device};
    if (auto result = backend.initialize(selected, true); !result.ok()) {
        for (const auto& error : result.errors) std::cerr << error << '\n';
        return 1;
    }
    if (auto result = backend.validate_lightning_index_device(device);
        !result.ok()) {
        for (const auto& error : result.errors) std::cerr << error << '\n';
        return 1;
    }

    const auto layout = strata::dsv4_physical_kv_layout(
        strata::Dsv4KvBlockKind::LearnedIndex, kBlockRows);
    if (!layout.ok()) {
        std::cerr << "physical learned-index layout unavailable\n";
        return 1;
    }

    std::vector<float> queries(
        static_cast<std::size_t>(kHeads) * kHeadDim);
    for (std::size_t index = 0U; index < queries.size(); ++index) {
        queries[index] = static_cast<float>((index * 37U) % 23U) / 8.0F - 1.5F;
    }
    if (auto result = strata::dsv4_physical_quantize_query_e4m3_f32(queries);
        !result.ok()) {
        std::cerr << "query quantization failed\n";
        return 1;
    }
    std::vector<float> weights(kHeads);
    for (std::uint32_t head = 0U; head < kHeads; ++head) {
        weights[head] = static_cast<float>(static_cast<int>(head % 7U) - 3) /
                        8.0F;
    }

    std::cout << "physical Lightning Indexer, device " << device << ", "
              << repetitions << " interleaved repetitions\n"
              << "heads " << kHeads << "  head_dim " << kHeadDim
              << "  top_k " << kTopK << "  ratio-4 (indexed) layers "
              << kLayers << " of "
              << strata::kDeepSeekV4ExecutionContract.layer_count << "\n\n"
              << std::setw(12) << "context" << std::setw(12) << "candidates"
              << std::setw(14) << "ms/layer" << std::setw(14) << "ms/token"
              << std::setw(16) << "GB/s effective" << std::setw(14)
              << "kernel ms" << std::setw(14) << "outside ms" << '\n';

    for (const std::uint64_t context :
         {std::uint64_t{4'096}, std::uint64_t{65'536},
          std::uint64_t{262'144}, std::uint64_t{1'048'576}}) {
        const auto candidates = static_cast<std::uint32_t>(context / 4U);
        const auto pages = (candidates + kBlockRows - 1U) / kBlockRows;
        const auto tail = candidates - (pages - 1U) * kBlockRows;

        std::vector<std::byte> storage(
            static_cast<std::size_t>(pages) * layout.value.block_bytes);
        std::vector<float> key(kHeadDim);
        for (std::uint32_t row = 0U; row < candidates; ++row) {
            for (std::uint32_t column = 0U; column < kHeadDim; ++column) {
                key[column] = static_cast<float>(
                    static_cast<int>((row * 31U + column * 11U) % 17U) - 8) /
                    4.0F;
            }
            auto page = std::span<std::byte>(storage).subspan(
                static_cast<std::size_t>(row / kBlockRows) *
                    layout.value.block_bytes,
                static_cast<std::size_t>(layout.value.block_bytes));
            if (auto result = strata::dsv4_physical_encode_kv_row(
                    strata::Dsv4KvBlockKind::LearnedIndex, key,
                    row % kBlockRows, page);
                !result.ok()) {
                std::cerr << "encode failed at row " << row << '\n';
                return 1;
            }
        }

        strata::CudaBuffer resident;
        if (auto result = backend.upload_buffer(device, storage, resident);
            !result.ok()) {
            for (const auto& error : result.errors) std::cerr << error << '\n';
            return 1;
        }
        std::vector<strata::CudaDsv4PhysicalIndexPage> descriptors;
        descriptors.reserve(pages);
        for (std::uint32_t page = 0U; page < pages; ++page) {
            descriptors.push_back(strata::CudaDsv4PhysicalIndexPage{
                &resident,
                static_cast<std::uint64_t>(page) * layout.value.block_bytes,
                kBlockRows, page + 1U == pages ? tail : kBlockRows});
        }
        strata::CudaDsv4PhysicalIndexRequest request;
        request.queries = queries;
        request.weights = weights;
        request.pages = descriptors;
        request.heads = kHeads;
        request.head_dim = kHeadDim;
        request.top_k = std::min(kTopK, candidates);
        std::vector<std::uint32_t> output(request.top_k);

        // One untimed call absorbs workspace allocation, so the reported
        // figure is steady-state rather than first-call cost.
        if (auto result = backend.dsv4_physical_lightning_index(
                device, request, output);
            !result.ok()) {
            for (const auto& error : result.errors) std::cerr << error << '\n';
            return 1;
        }
        const auto before = backend.stats();
        std::vector<double> samples;
        samples.reserve(repetitions);
        for (std::uint32_t repetition = 0U; repetition < repetitions;
             ++repetition) {
            const auto started = std::chrono::steady_clock::now();
            if (auto result = backend.dsv4_physical_lightning_index(
                    device, request, output);
                !result.ok()) {
                for (const auto& error : result.errors) {
                    std::cerr << error << '\n';
                }
                return 1;
            }
            samples.push_back(
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count());
        }
        const auto after = backend.stats();
        // Splitting measured wall time into device kernel time and everything
        // else separates a volume problem from a launch/serialization one.
        const auto kernel_ms =
            static_cast<double>(after.lightning_index_kernel_nanoseconds -
                                before.lightning_index_kernel_nanoseconds) /
            1.0e6 / static_cast<double>(repetitions);
        const auto per_layer = median(samples);
        const auto row_bytes = static_cast<double>(kHeadDim) + sizeof(float);
        const auto bytes = static_cast<double>(candidates) * row_bytes;
        std::cout << std::setw(12) << context << std::setw(12) << candidates
                  << std::setw(14) << std::fixed << std::setprecision(3)
                  << per_layer << std::setw(14) << per_layer * kLayers
                  << std::setw(16) << std::setprecision(1)
                  << bytes / (per_layer * 1.0e-3) / 1.0e9
                  << std::setw(14) << std::setprecision(3) << kernel_ms
                  << std::setw(14) << per_layer - kernel_ms << '\n';
    }
    return 0;
}
