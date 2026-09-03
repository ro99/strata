#include "strata/models/glm53/glm53_runtime.hpp"
#include "strata/models/glm53/glm53_sequence.hpp"

#include "../common/cuda_stats_delta.hpp"

#include "strata/engine/runtime_support.hpp"
#include "strata/engine/route_predictor.hpp"
#include "strata/models/common/tokenizer.hpp"
#include "strata/models/deepseek/deepseek_ops.hpp"
#include "strata/models/glm53/glm53_checkpoint.hpp"
#include "strata/models/glm53/glm53_expert_profile.hpp"
#include "strata/models/kimi_k3/kimi_k3_ops.hpp"
#include "strata/platform/diagnostics.hpp"
#include "strata/platform/hardware_profile.hpp"
#include "strata/platform/numerics.hpp"
#include "strata/platform/worker_pool.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <fstream>
#include <iostream>
#include <iterator>
#include <list>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <numeric>
#include <random>
#include <sstream>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define STRATA_GLM53_HOST_AVX2 1
#else
#define STRATA_GLM53_HOST_AVX2 0
#endif

namespace strata {

std::vector<std::size_t> glm53_projection_slots(
    std::span<const std::string_view> keys,
    std::span<const std::uint64_t> costs,
    std::span<const std::uint64_t> capacities,
    std::size_t preferred_slot) {
    if (keys.empty() || keys.size() != costs.size() || capacities.empty() ||
        preferred_slot >= capacities.size() ||
        std::any_of(capacities.begin(), capacities.end(),
                    [](std::uint64_t value) { return value == 0U; })) {
        return {};
    }
    std::vector<std::size_t> order(keys.size());
    std::iota(order.begin(), order.end(), 0U);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t left,
                                                      std::size_t right) {
        if (costs[left] != costs[right]) return costs[left] > costs[right];
        if (keys[left] != keys[right]) return keys[left] < keys[right];
        return left < right;
    });
    std::vector<long double> loads(capacities.size(), 0.0L);
    std::vector<std::size_t> slots(keys.size());
    for (const auto index : order) {
        std::size_t best = 0U;
        for (std::size_t slot = 1U; slot < capacities.size(); ++slot) {
            const auto candidate = loads[slot] /
                static_cast<long double>(capacities[slot]);
            const auto incumbent = loads[best] /
                static_cast<long double>(capacities[best]);
            const auto candidate_distance =
                (slot + capacities.size() - preferred_slot) % capacities.size();
            const auto incumbent_distance =
                (best + capacities.size() - preferred_slot) % capacities.size();
            if (candidate < incumbent ||
                (candidate == incumbent &&
                 candidate_distance < incumbent_distance)) {
                best = slot;
            }
        }
        slots[index] = best;
        loads[best] += static_cast<long double>(std::max<std::uint64_t>(
            costs[index], 1U));
    }
    return slots;
}

namespace {

[[nodiscard]] std::uint64_t elapsed_nanoseconds(
    std::chrono::steady_clock::time_point started) noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count());
}

[[nodiscard]] bool phase_profile_environment_enabled() noexcept {
    const char* value = std::getenv("STRATA_GLM53_PHASE_PROFILE");
    return value != nullptr && std::string_view(value) != "0" &&
           std::string_view(value) != "false" &&
           std::string_view(value) != "off";
}

[[nodiscard]] std::uint32_t prefill_page_tokens_from_environment(
    std::uint32_t fallback) noexcept {
    const char* value = std::getenv("STRATA_GLM53_PREFILL_PAGE_TOKENS");
    if (value == nullptr || *value == '\0') return fallback;
    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0U ||
        parsed > std::numeric_limits<std::uint32_t>::max()) {
        return 0U;
    }
    return static_cast<std::uint32_t>(parsed);
}

[[nodiscard]] Glm53CacheMetrics cache_delta(
    const Glm53CacheMetrics& after,
    const Glm53CacheMetrics& before) noexcept {
    return {
        after.hits - before.hits,
        after.misses - before.misses,
        after.evictions - before.evictions,
        after.prefetches - before.prefetches,
        after.useful_prefetches - before.useful_prefetches,
        after.failed_prefetches - before.failed_prefetches};
}

[[nodiscard]] Glm53HostExpertMetrics host_expert_delta(
    const Glm53HostExpertMetrics& after,
    const Glm53HostExpertMetrics& before) noexcept {
    return {
        after.calls - before.calls,
        after.rows - before.rows,
        after.gate_up_weight_bytes - before.gate_up_weight_bytes,
        after.down_weight_bytes - before.down_weight_bytes,
        after.view_resolution_nanoseconds - before.view_resolution_nanoseconds,
        after.input_quantization_nanoseconds -
            before.input_quantization_nanoseconds,
        after.gate_up_nanoseconds - before.gate_up_nanoseconds,
        after.activation_nanoseconds - before.activation_nanoseconds,
        after.down_nanoseconds - before.down_nanoseconds,
        after.reduction_nanoseconds - before.reduction_nanoseconds,
        after.service_nanoseconds - before.service_nanoseconds,
        after.temporary_allocation_calls - before.temporary_allocation_calls};
}

[[nodiscard]] Glm53GraphMetrics graph_delta(
    const Glm53GraphMetrics& after,
    const Glm53GraphMetrics& before) noexcept {
    return {
        after.forward_calls - before.forward_calls,
        after.forward_rows - before.forward_rows,
        after.embedding_nanoseconds - before.embedding_nanoseconds,
        after.layer_nanoseconds - before.layer_nanoseconds,
        after.attention_block_nanoseconds -
            before.attention_block_nanoseconds,
        after.kda_nanoseconds - before.kda_nanoseconds,
        after.mla_nanoseconds - before.mla_nanoseconds,
        after.feedforward_block_nanoseconds -
            before.feedforward_block_nanoseconds,
        after.output_head_nanoseconds - before.output_head_nanoseconds,
        after.sampling_nanoseconds - before.sampling_nanoseconds};
}

[[nodiscard]] Glm53SparseMlaMetrics sparse_mla_delta(
    const Glm53SparseMlaMetrics& after,
    const Glm53SparseMlaMetrics& before) noexcept {
    return {
        after.calls - before.calls,
        after.input_download_nanoseconds - before.input_download_nanoseconds,
        after.indexer_projection_nanoseconds -
            before.indexer_projection_nanoseconds,
        after.indexer_state_nanoseconds - before.indexer_state_nanoseconds,
        after.query_rank_projection_nanoseconds -
            before.query_rank_projection_nanoseconds,
        after.pool_scoring_nanoseconds - before.pool_scoring_nanoseconds,
        after.topk_sort_nanoseconds - before.topk_sort_nanoseconds,
        after.arena_bookkeeping_nanoseconds -
            before.arena_bookkeeping_nanoseconds,
        after.index_upload_nanoseconds - before.index_upload_nanoseconds,
        after.device_scores_wait_nanoseconds -
            before.device_scores_wait_nanoseconds,
        after.host_softmax_nanoseconds - before.host_softmax_nanoseconds,
        after.coefficient_upload_nanoseconds -
            before.coefficient_upload_nanoseconds};
}

void print_token_ids(std::ostream& output,
                     std::span<const std::uint32_t> token_ids) {
    output << '[';
    for (std::size_t index = 0U; index < token_ids.size(); ++index) {
        if (index != 0U) output << ',';
        output << token_ids[index];
    }
    output << ']';
}

void print_phase_metrics(std::ostream& output,
                         const Glm53PhaseMetrics& phase) {
    output << "{\"cuda\":{\"weight_h2d_bytes\":"
           << phase.cuda.weight_upload_bytes
           << ",\"activation_h2d_bytes\":"
           << phase.cuda.activation_h2d_bytes
           << ",\"activation_d2h_bytes\":"
           << phase.cuda.activation_d2h_bytes
           << ",\"matmul_calls\":" << phase.cuda.matmul_calls
           << ",\"weight_allocation_calls\":"
           << phase.cuda.weight_allocation_calls
           << ",\"workspace_allocation_calls\":"
           << phase.cuda.workspace_allocation_calls
           << ",\"synchronization_calls\":"
           << phase.cuda.synchronization_calls
           << ",\"synchronization_nanoseconds\":"
           << phase.cuda.synchronization_nanoseconds
           << ",\"kernel_nanoseconds\":"
           << phase.cuda.kernel_nanoseconds
           << ",\"glm53_kda_kernel_nanoseconds\":"
           << phase.cuda.glm53_kda_kernel_nanoseconds
           << ",\"glm53_mla_kernel_nanoseconds\":"
           << phase.cuda.glm53_mla_kernel_nanoseconds
           << ",\"glm53_expert_kernel_nanoseconds\":"
           << phase.cuda.glm53_expert_kernel_nanoseconds
           << ",\"glm53_other_kernel_nanoseconds\":"
           << phase.cuda.glm53_other_kernel_nanoseconds
           << "},\"cache\":{\"hits\":" << phase.cache.hits
           << ",\"misses\":" << phase.cache.misses
           << ",\"evictions\":" << phase.cache.evictions
           << ",\"prefetches\":" << phase.cache.prefetches
           << ",\"useful_prefetches\":"
           << phase.cache.useful_prefetches
           << ",\"failed_prefetches\":" << phase.cache.failed_prefetches
           << "},\"host_experts\":{\"calls\":"
           << phase.host_experts.calls
           << ",\"rows\":" << phase.host_experts.rows
           << ",\"gate_up_weight_bytes\":"
           << phase.host_experts.gate_up_weight_bytes
           << ",\"down_weight_bytes\":"
           << phase.host_experts.down_weight_bytes
           << ",\"view_resolution_nanoseconds\":"
           << phase.host_experts.view_resolution_nanoseconds
           << ",\"input_quantization_nanoseconds\":"
           << phase.host_experts.input_quantization_nanoseconds
           << ",\"gate_up_nanoseconds\":"
           << phase.host_experts.gate_up_nanoseconds
           << ",\"activation_nanoseconds\":"
           << phase.host_experts.activation_nanoseconds
           << ",\"down_nanoseconds\":"
           << phase.host_experts.down_nanoseconds
           << ",\"reduction_nanoseconds\":"
           << phase.host_experts.reduction_nanoseconds
           << ",\"service_nanoseconds\":"
           << phase.host_experts.service_nanoseconds
           << ",\"temporary_allocation_calls\":"
           << phase.host_experts.temporary_allocation_calls
           << "},\"graph\":{\"forward_calls\":"
           << phase.graph.forward_calls
           << ",\"forward_rows\":" << phase.graph.forward_rows
           << ",\"embedding_nanoseconds\":"
           << phase.graph.embedding_nanoseconds
           << ",\"layer_nanoseconds\":" << phase.graph.layer_nanoseconds
           << ",\"attention_block_nanoseconds\":"
           << phase.graph.attention_block_nanoseconds
           << ",\"kda_nanoseconds\":" << phase.graph.kda_nanoseconds
           << ",\"mla_nanoseconds\":" << phase.graph.mla_nanoseconds
           << ",\"feedforward_block_nanoseconds\":"
           << phase.graph.feedforward_block_nanoseconds
           << ",\"output_head_nanoseconds\":"
           << phase.graph.output_head_nanoseconds
           << ",\"sampling_nanoseconds\":"
           << phase.graph.sampling_nanoseconds
           << "},\"sparse_mla\":{\"calls\":"
           << phase.sparse_mla.calls
           << ",\"input_download_nanoseconds\":"
           << phase.sparse_mla.input_download_nanoseconds
           << ",\"indexer_projection_nanoseconds\":"
           << phase.sparse_mla.indexer_projection_nanoseconds
           << ",\"indexer_state_nanoseconds\":"
           << phase.sparse_mla.indexer_state_nanoseconds
           << ",\"query_rank_projection_nanoseconds\":"
           << phase.sparse_mla.query_rank_projection_nanoseconds
           << ",\"pool_scoring_nanoseconds\":"
           << phase.sparse_mla.pool_scoring_nanoseconds
           << ",\"topk_sort_nanoseconds\":"
           << phase.sparse_mla.topk_sort_nanoseconds
           << ",\"arena_bookkeeping_nanoseconds\":"
           << phase.sparse_mla.arena_bookkeeping_nanoseconds
           << ",\"index_upload_nanoseconds\":"
           << phase.sparse_mla.index_upload_nanoseconds
           << ",\"device_scores_wait_nanoseconds\":"
           << phase.sparse_mla.device_scores_wait_nanoseconds
           << ",\"host_softmax_nanoseconds\":"
           << phase.sparse_mla.host_softmax_nanoseconds
           << ",\"coefficient_upload_nanoseconds\":"
           << phase.sparse_mla.coefficient_upload_nanoseconds << "}}";
}

constexpr std::uint32_t kHidden = 4096U;
constexpr std::uint32_t kLayers = 45U;
constexpr std::uint32_t kMtpLayer = 45U;
// Consecutive projection rows claimed per worker in the host expert loops.
// 64 rows is 256 KiB of contiguous FP8 weight per claim -- wide enough for the
// hardware prefetcher and for a coalesced device request, narrow enough that 28
// workers still balance across a 2048-row expert at 32 claims each. Single-index
// dispatch instead interleaves 28 workers across one matrix, so each walks it
// with a 28-row stride; experiment 0198 measured that shape at 0.69 GB/s against
// 1.96 GB/s blocked on identical cold bytes.
constexpr std::size_t kExpertDispatchBlock = 64U;

// STRATA_GLM53_EXPERT_DISPATCH_BLOCK overrides it for the M2 A/B. 1 reproduces
// the previous single-index dispatch exactly, so both arms of the comparison
// run the same binary and the build cannot be a confound. Production default
// stays 64.
[[nodiscard]] std::size_t expert_dispatch_block() noexcept {
    static const std::size_t block = [] {
        const char* value = std::getenv("STRATA_GLM53_EXPERT_DISPATCH_BLOCK");
        if (value == nullptr) return kExpertDispatchBlock;
        const std::size_t parsed = std::strtoul(value, nullptr, 10);
        return parsed == 0U ? kExpertDispatchBlock : parsed;
    }();
    return block;
}
constexpr std::uint32_t kHeads = 64U;
constexpr std::uint32_t kLinearHead = 128U;
constexpr std::uint32_t kLinearWidth = kHeads * kLinearHead;

// The three storage formats a GLM-5.3 routed or shared expert can arrive in.
// The FP8 release uses only the first; the MXFP4 release uses the second for
// routed experts in the 39 quantized layers and the third everywhere else,
// including the shared expert and the routed experts of layers 3, 5 and 6 that
// the publisher's mixed-precision correction left in BF16.
enum class Glm53ExpertEncoding : std::uint8_t {
    Fp8E4m3Block128F32,
    Fp4E2m1Group32E8m0,
    Bf16,
};

struct Glm53HostExpertLinear {
    std::span<const std::byte> weights;
    // FP8: F32 inverse scales, one per 128x128 block, row-major over blocks.
    // MXFP4: E8M0 bytes, one per 32 columns of each row. BF16: empty.
    std::span<const std::byte> scales;
    std::uint32_t rows{};
    std::uint32_t columns{};
    Glm53ExpertEncoding encoding{Glm53ExpertEncoding::Fp8E4m3Block128F32};
};

// One output row of one projection, resolved once outside the inner loop --
// the page path reuses a row across every token routed to that expert, so the
// row arithmetic must not sit inside the assignment loop.
struct Glm53HostExpertRow {
    const std::byte* weights{};
    const void* scales{};
    Glm53ExpertEncoding encoding{Glm53ExpertEncoding::Fp8E4m3Block128F32};
};

// Reusable host-MoE scratch. The three buffers are fully overwritten every
// call -- gate/up writes every activation slot, down writes every output slot --
// so reuse without clearing is exact, and the byte-identical output gate is what
// verifies it rather than the argument. Thread-local so reuse carries no
// assumption about which thread drives the MoE.
//
// `grow` returns true when it actually allocated, so the profiler's allocation
// counter reports what happened instead of a constant: M0 counted 30,208 timed
// allocations per 128 decode tokens, and after warm-up this should be zero.
[[nodiscard]] bool glm53_grow(std::vector<float>& buffer, std::size_t size) {
    if (buffer.size() >= size) return false;
    const auto before = buffer.capacity();
    buffer.resize(size);
    return buffer.capacity() != before;
}

[[nodiscard]] float glm53_quantize_e4m3(float value) noexcept {
    const float magnitude = std::min(std::abs(value), 448.0F);
    float quantized = 0.0F;
    if (magnitude < 0.015625F) {
        quantized = std::rint(std::ldexp(magnitude, 9)) *
                    std::ldexp(1.0F, -9);
    } else {
        int exponent = 0;
        static_cast<void>(std::frexp(magnitude, &exponent));
        exponent = std::clamp(exponent - 1, -6, 8);
        const float step = std::ldexp(1.0F, exponent - 3);
        quantized = std::min(std::rint(magnitude / step) * step, 448.0F);
    }
    return std::copysign(quantized, value);
}

void glm53_quantize_activation(std::span<float> values) noexcept {
    constexpr std::size_t block = 128U;
    for (std::size_t begin = 0U; begin < values.size(); begin += block) {
        const auto end = std::min(begin + block, values.size());
        float maximum = 0.0F;
        for (auto index = begin; index < end; ++index) {
            maximum = std::max(maximum, std::abs(values[index]));
        }
        const float scale = maximum > 0.0F ? maximum / 448.0F : 1.0F;
        for (auto index = begin; index < end; ++index) {
            values[index] = glm53_quantize_e4m3(values[index] / scale) * scale;
        }
    }
}

[[nodiscard]] const std::array<float, 256U>& glm53_fp8_values() noexcept {
    static const auto values = [] {
        std::array<float, 256U> result{};
        for (std::size_t index = 0U; index < result.size(); ++index) {
            result[index] = fp8_e4m3_f32(static_cast<std::uint8_t>(index));
        }
        return result;
    }();
    return values;
}

[[nodiscard]] float glm53_host_fp8_dot_scalar(
    const std::byte* weights, const float* scales,
    std::span<const float> input) noexcept {
    const auto& values = glm53_fp8_values();
    float sum = 0.0F;
    for (std::size_t column = 0U; column < input.size(); ++column) {
        const auto code = std::to_integer<std::uint8_t>(weights[column]);
        sum = std::fma(input[column] * values[code], scales[column / 128U], sum);
    }
    return sum;
}

#if STRATA_GLM53_HOST_AVX2
__attribute__((target("avx2,fma")))
[[nodiscard]] float glm53_host_fp8_dot_avx2(
    const std::byte* weights, const float* scales,
    std::span<const float> input) noexcept {
    const auto& values = glm53_fp8_values();
    __m256 accumulators[8]{
        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps()};
    std::size_t column = 0U;
    for (; column + 64U <= input.size(); column += 64U) {
        const auto scale = _mm256_set1_ps(scales[column / 128U]);
        for (std::size_t group = 0U; group < 8U; ++group) {
            const auto offset = column + group * 8U;
            const auto bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
                weights + offset));
            const auto indices = _mm256_cvtepu8_epi32(bytes);
            const auto decoded = _mm256_i32gather_ps(values.data(), indices, 4);
            const auto activation = _mm256_loadu_ps(input.data() + offset);
            accumulators[group] = _mm256_fmadd_ps(
                _mm256_mul_ps(decoded, scale), activation,
                accumulators[group]);
        }
    }
    // Keep eight independent dependency chains through the matrix and combine
    // only once at the end.  This is the host analogue of DeepSeek's tiled
    // executor: the checkpoint byte is decoded in-register and never expanded
    // into a second resident copy.
    for (std::size_t width = 4U; width != 0U; width >>= 1U) {
        for (std::size_t index = 0U; index < width; ++index) {
            accumulators[index] = _mm256_add_ps(
                accumulators[index], accumulators[index + width]);
        }
    }
    const __m128 low = _mm256_castps256_ps128(accumulators[0]);
    const __m128 high = _mm256_extractf128_ps(accumulators[0], 1);
    __m128 total = _mm_add_ps(low, high);
    total = _mm_hadd_ps(total, total);
    total = _mm_hadd_ps(total, total);
    float sum = _mm_cvtss_f32(total);
    for (; column < input.size(); ++column) {
        const auto code = std::to_integer<std::uint8_t>(weights[column]);
        sum = std::fma(input[column] * values[code], scales[column / 128U], sum);
    }
    return sum;
}
#endif

[[nodiscard]] float glm53_host_fp8_dot(
    const std::byte* weights, const float* scales,
    std::span<const float> input) noexcept {
#if STRATA_GLM53_HOST_AVX2
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        return glm53_host_fp8_dot_avx2(weights, scales, input);
    }
#endif
    return glm53_host_fp8_dot_scalar(weights, scales, input);
}

// E8M0 is a bare binary exponent biased by 127, so the whole format is 255
// powers of two plus one NaN code. Decoding it through a table keeps the inner
// loop free of ldexp and lets the AVX2 path broadcast a plain float.
[[nodiscard]] const std::array<float, 256U>& glm53_e8m0_values() noexcept {
    static const auto values = [] {
        std::array<float, 256U> result{};
        for (std::size_t index = 0U; index < 255U; ++index) {
            result[index] = std::ldexp(1.0F, static_cast<int>(index) - 127);
        }
        result[255U] = std::numeric_limits<float>::quiet_NaN();
        return result;
    }();
    return values;
}

[[nodiscard]] const std::array<float, 16U>& glm53_fp4_values() noexcept {
    static const auto values = [] {
        std::array<float, 16U> result{};
        for (std::size_t index = 0U; index < result.size(); ++index) {
            result[index] = fp4_e2m1_f32(static_cast<std::uint8_t>(index));
        }
        return result;
    }();
    return values;
}

// MXFP4: `packed` holds two E2M1 nibbles per byte, column 2b in the low nibble
// of byte b and column 2b+1 in the high nibble; `scales` holds one E8M0 byte
// per 32 columns of this row. The accumulation order mirrors the FP8 dot so
// the two formats differ only in how a weight is decoded.
[[nodiscard]] float glm53_host_fp4_dot_scalar(
    const std::byte* packed, const std::uint8_t* scales,
    std::span<const float> input) noexcept {
    const auto& values = glm53_fp4_values();
    const auto& exponents = glm53_e8m0_values();
    float sum = 0.0F;
    for (std::size_t column = 0U; column < input.size(); ++column) {
        const auto byte = std::to_integer<std::uint8_t>(packed[column / 2U]);
        const auto nibble = static_cast<std::uint8_t>(
            (column % 2U == 0U) ? (byte & 0x0FU) : (byte >> 4U));
        sum = std::fma(input[column] * values[nibble],
                       exponents[scales[column / 32U]], sum);
    }
    return sum;
}

#if STRATA_GLM53_HOST_AVX2
__attribute__((target("avx2,fma")))
[[nodiscard]] float glm53_host_fp4_dot_avx2(
    const std::byte* packed, const std::uint8_t* scales,
    std::span<const float> input) noexcept {
    const auto& values = glm53_fp4_values();
    const auto& exponents = glm53_e8m0_values();
    // Eight consecutive columns live in four consecutive bytes, so one 32-bit
    // load plus a variable right shift places each nibble in its own lane, in
    // column order and without a shuffle.
    const auto shifts = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    const auto mask = _mm256_set1_epi32(0x0F);
    // E2M1's sixteen values are eight magnitudes and a sign bit, so a single
    // in-register permute plus an XOR decodes them. That matters on this host:
    // `vgatherdps` is microcoded on Broadwell and the FP8 dot has no choice
    // but to pay it for a 256-entry table, while FP4 does. The result is bit-
    // identical to indexing `glm53_fp4_values`, sign of zero included.
    const auto magnitudes =
        _mm256_setr_ps(0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F);
    const auto magnitude_mask = _mm256_set1_epi32(0x07);
    const auto sign_mask = _mm256_set1_epi32(0x08);
    __m256 accumulators[8]{
        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps()};
    std::size_t column = 0U;
    for (; column + 64U <= input.size(); column += 64U) {
        // Sixty-four columns span exactly two group-32 scales, which keeps the
        // eight independent accumulator chains of the FP8 dot intact.
        const auto low_scale = _mm256_set1_ps(exponents[scales[column / 32U]]);
        const auto high_scale =
            _mm256_set1_ps(exponents[scales[column / 32U + 1U]]);
        for (std::size_t group = 0U; group < 8U; ++group) {
            const auto offset = column + group * 8U;
            std::uint32_t word = 0U;
            std::memcpy(&word, packed + offset / 2U, sizeof(word));
            const auto nibbles = _mm256_and_si256(
                _mm256_srlv_epi32(_mm256_set1_epi32(
                                      static_cast<int>(word)), shifts), mask);
            const auto decoded = _mm256_xor_ps(
                _mm256_permutevar8x32_ps(
                    magnitudes, _mm256_and_si256(nibbles, magnitude_mask)),
                _mm256_castsi256_ps(_mm256_slli_epi32(
                    _mm256_and_si256(nibbles, sign_mask), 28)));
            const auto activation = _mm256_loadu_ps(input.data() + offset);
            accumulators[group] = _mm256_fmadd_ps(
                _mm256_mul_ps(decoded, group < 4U ? low_scale : high_scale),
                activation, accumulators[group]);
        }
    }
    for (std::size_t width = 4U; width != 0U; width >>= 1U) {
        for (std::size_t index = 0U; index < width; ++index) {
            accumulators[index] = _mm256_add_ps(
                accumulators[index], accumulators[index + width]);
        }
    }
    const __m128 low = _mm256_castps256_ps128(accumulators[0]);
    const __m128 high = _mm256_extractf128_ps(accumulators[0], 1);
    __m128 total = _mm_add_ps(low, high);
    total = _mm_hadd_ps(total, total);
    total = _mm_hadd_ps(total, total);
    float sum = _mm_cvtss_f32(total);
    for (; column < input.size(); ++column) {
        const auto byte = std::to_integer<std::uint8_t>(packed[column / 2U]);
        const auto nibble = static_cast<std::uint8_t>(
            (column % 2U == 0U) ? (byte & 0x0FU) : (byte >> 4U));
        sum = std::fma(input[column] * values[nibble],
                       exponents[scales[column / 32U]], sum);
    }
    return sum;
}
#endif

[[nodiscard]] float glm53_host_fp4_dot(
    const std::byte* packed, const std::uint8_t* scales,
    std::span<const float> input) noexcept {
#if STRATA_GLM53_HOST_AVX2
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        return glm53_host_fp4_dot_avx2(packed, scales, input);
    }
#endif
    return glm53_host_fp4_dot_scalar(packed, scales, input);
}

// BF16 rows carry no scale. The MXFP4 release leaves the shared expert and the
// routed experts of layers 3, 5 and 6 in this form, so the host MoE meets it on
// every one of those layers, not as an exceptional case.
[[nodiscard]] float glm53_host_bf16_dot_scalar(
    const std::byte* weights, std::span<const float> input) noexcept {
    float sum = 0.0F;
    for (std::size_t column = 0U; column < input.size(); ++column) {
        std::uint16_t encoded = 0U;
        std::memcpy(&encoded, weights + column * 2U, sizeof(encoded));
        const auto value = std::bit_cast<float>(
            static_cast<std::uint32_t>(encoded) << 16U);
        sum = std::fma(input[column], value, sum);
    }
    return sum;
}

#if STRATA_GLM53_HOST_AVX2
__attribute__((target("avx2,fma")))
[[nodiscard]] float glm53_host_bf16_dot_avx2(
    const std::byte* weights, std::span<const float> input) noexcept {
    __m256 accumulators[8]{
        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps()};
    std::size_t column = 0U;
    for (; column + 64U <= input.size(); column += 64U) {
        for (std::size_t group = 0U; group < 8U; ++group) {
            const auto offset = column + group * 8U;
            const auto encoded = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(weights + offset * 2U));
            const auto widened = _mm256_slli_epi32(
                _mm256_cvtepu16_epi32(encoded), 16);
            const auto decoded = _mm256_castsi256_ps(widened);
            const auto activation = _mm256_loadu_ps(input.data() + offset);
            accumulators[group] = _mm256_fmadd_ps(decoded, activation,
                                                  accumulators[group]);
        }
    }
    for (std::size_t width = 4U; width != 0U; width >>= 1U) {
        for (std::size_t index = 0U; index < width; ++index) {
            accumulators[index] = _mm256_add_ps(
                accumulators[index], accumulators[index + width]);
        }
    }
    const __m128 low = _mm256_castps256_ps128(accumulators[0]);
    const __m128 high = _mm256_extractf128_ps(accumulators[0], 1);
    __m128 total = _mm_add_ps(low, high);
    total = _mm_hadd_ps(total, total);
    total = _mm_hadd_ps(total, total);
    float sum = _mm_cvtss_f32(total);
    for (; column < input.size(); ++column) {
        std::uint16_t encoded = 0U;
        std::memcpy(&encoded, weights + column * 2U, sizeof(encoded));
        sum = std::fma(input[column],
                       std::bit_cast<float>(
                           static_cast<std::uint32_t>(encoded) << 16U),
                       sum);
    }
    return sum;
}
#endif

[[nodiscard]] float glm53_host_bf16_dot(
    const std::byte* weights, std::span<const float> input) noexcept {
#if STRATA_GLM53_HOST_AVX2
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        return glm53_host_bf16_dot_avx2(weights, input);
    }
#endif
    return glm53_host_bf16_dot_scalar(weights, input);
}

[[nodiscard]] Glm53HostExpertRow glm53_host_expert_row(
    const Glm53HostExpertLinear& linear, std::size_t row) noexcept {
    Glm53HostExpertRow view;
    view.encoding = linear.encoding;
    switch (linear.encoding) {
        case Glm53ExpertEncoding::Fp8E4m3Block128F32:
            view.weights = linear.weights.data() + row * linear.columns;
            view.scales = reinterpret_cast<const float*>(linear.scales.data()) +
                          (row / 128U) * (linear.columns / 128U);
            break;
        case Glm53ExpertEncoding::Fp4E2m1Group32E8m0:
            view.weights = linear.weights.data() + row * (linear.columns / 2U);
            view.scales =
                reinterpret_cast<const std::uint8_t*>(linear.scales.data()) +
                row * (linear.columns / 32U);
            break;
        case Glm53ExpertEncoding::Bf16:
            view.weights = linear.weights.data() + row * linear.columns * 2U;
            break;
    }
    return view;
}

[[nodiscard]] float glm53_host_expert_dot(
    const Glm53HostExpertRow& row, std::span<const float> input) noexcept {
    switch (row.encoding) {
        case Glm53ExpertEncoding::Fp4E2m1Group32E8m0:
            return glm53_host_fp4_dot(
                row.weights, static_cast<const std::uint8_t*>(row.scales),
                input);
        case Glm53ExpertEncoding::Bf16:
            return glm53_host_bf16_dot(row.weights, input);
        case Glm53ExpertEncoding::Fp8E4m3Block128F32:
            break;
    }
    return glm53_host_fp8_dot(
        row.weights, static_cast<const float*>(row.scales), input);
}
constexpr std::uint32_t kMlaHead = 256U;
constexpr std::uint32_t kMlaWidth = kHeads * kMlaHead;
constexpr std::uint32_t kQueryRank = 1536U;
constexpr std::uint32_t kKvRank = 512U;
constexpr std::uint32_t kMhc = 4U;
constexpr std::uint32_t kVocabulary = 154880U;
// With the k-pool indexer the MLA workspace no longer grows with history, so
// the bound is host sequence state: about 33.5 KB per token across the eleven
// sparse layers (512-wide latent plus the 256-wide indexer row). 262,144
// tokens is roughly 8.8 GiB, which this host holds comfortably. The checkpoint
// itself declares 1,048,576; that is not offered until it has been measured.
constexpr std::uint32_t kMaximumSupportedContext = 262144U;
// GLM-5.3 k-pool sparse indexer (record 0237). The checkpoint ships 84 indexer
// tensors that this adapter loaded and validated but never used, attending
// densely instead. That is exact only while history <= kIndexTopK, because
// selecting the top kIndexTopK of at most kIndexTopK candidates is the
// identity -- which is why the dense path passed every exactness gate and why
// the context was capped at 2,048.
constexpr std::uint32_t kIndexHeads = 32U;
constexpr std::uint32_t kIndexHeadDim = 128U;
constexpr std::uint32_t kIndexTopK = 2048U;
constexpr std::uint32_t kIndexPool = 4U;
// Selected pools expand to kIndexTopK tokens; the always-selected tail adds at
// most one incomplete pool, i.e. kIndexPool - 1 further raw positions.
constexpr std::uint32_t kIndexSelectionWidth = kIndexTopK + kIndexPool - 1U;
// The saturated selection owns 512 live pools. The measured maximum churn was
// 89 pools per token, so retain 128 additional recently-used pools as bounded
// headroom; a pool that leaves and promptly re-enters need not be expanded
// again. Three dedicated rows hold the always-selected incomplete tail.
constexpr std::uint32_t kIndexArenaPools =
    kIndexTopK / kIndexPool + 128U;
constexpr std::uint32_t kIndexArenaRows =
    kIndexArenaPools * kIndexPool + kIndexPool - 1U;

// Whether a sequence runs the k-pool indexer at all. Only the host attention
// path implements it, so this decides where a sequence's MLA attention lives.
//
// It is deliberately a property of the admitted *context*, not of the current
// position. Switching at the crossing looks cheaper -- run the resident device
// chain until history reaches `kIndexTopK`, then move -- but the device chain
// keeps its own latent cache and computes no indexer state at all, so the host
// MLA and indexer caches would be empty for every token before the crossing and
// the first host step would find no history to pool. Below the threshold the
// selection is the identity, so a sequence that cannot cross it keeps the
// resident device path and is unaffected.
[[nodiscard]] constexpr bool sparse_indexer_active(
    std::uint32_t maximum_context_tokens) noexcept {
    return maximum_context_tokens > kIndexTopK;
}

// Chooses which history positions one decode query attends to. Returns the
// selected positions in ascending order.
//
// Two details are taken from the reference rather than inferred, and both are
// silent at history <= kIndexTopK:
//   * the softmax scale is applied INSIDE the ReLU (DeepSeek-V4's otherwise
//     similar indexer applies it outside);
//   * a pool is a candidate only when every one of its kIndexPool members is a
//     real, visible token, so the trailing incomplete group is never pooled --
//     it is appended raw as the tail.
// The indexer's key norm is nn.LayerNorm(head_dim, eps=1e-6) -- mean
// subtracting, with a bias -- and NOT the RMSNorm this model uses everywhere
// else, including the attention k_norm. Taken from the reference; getting it
// wrong is silent below index_topk and wrong above it.
// `index_kpool_compress_gate` is [head_dim, hidden] applied as F.linear.
void glm53_indexer_gate(std::span<float> output, std::span<const float> input,
                        std::span<const float> weight) noexcept {
    for (std::size_t row = 0U; row < output.size(); ++row) {
        const auto* w = weight.data() + row * input.size();
        float sum = 0.0F;
        for (std::size_t column = 0U; column < input.size(); ++column) {
            sum = std::fma(w[column], input[column], sum);
        }
        output[row] = sum;
    }
}

void glm53_indexer_layer_norm(std::span<float> values,
                              std::span<const float> weight,
                              std::span<const float> bias) noexcept {
    double sum = 0.0;
    for (const auto value : values) sum += value;
    const auto mean = sum / static_cast<double>(values.size());
    double variance = 0.0;
    for (const auto value : values) {
        const auto centered = static_cast<double>(value) - mean;
        variance += centered * centered;
    }
    variance /= static_cast<double>(values.size());
    const auto inverse = 1.0 / std::sqrt(variance + 1.0e-6);
    for (std::size_t index = 0U; index < values.size(); ++index) {
        values[index] = static_cast<float>(
            (static_cast<double>(values[index]) - mean) * inverse) *
                weight[index] + bias[index];
    }
}

// One complete pool's learned key: a per-channel softmax average over the
// pool's four members, `logits = gate + index_kpool_compress_ape`.
//
// This depends on the pool's members alone -- not on the query -- so it is
// computed once, when the pool completes, and cached on the sequence. Rebuilding
// it per query is the indexer's dominant cost: it is `index_kpool x head_dim`
// exponentials per pool, and a 64-row prefill page was paying about 190 million
// of them to rebuild keys that are identical across all 64 rows.
//
// `key_at(token)` and `gate_at(token)` return one position's 128-wide
// normalized indexer key and k-pool gate. They are accessors rather than
// contiguous spans because the caller's history is a page table: materializing
// it would copy `history x 256 x 4 B` per query.
template <typename KeyAt, typename GateAt>
void glm53_index_pool_key(std::span<float> pool_key, std::uint32_t pool,
                          KeyAt&& key_at, GateAt&& gate_at,
                          std::span<const float> pool_ape) {
    const auto base = pool * kIndexPool;
    std::array<const float*, kIndexPool> member_key{};
    std::array<const float*, kIndexPool> member_gate{};
    for (std::uint32_t member = 0U; member < kIndexPool; ++member) {
        member_key[member] = key_at(base + member);
        member_gate[member] = gate_at(base + member);
    }
    std::array<float, kIndexPool> probability{};
    for (std::uint32_t channel = 0U; channel < kIndexHeadDim; ++channel) {
        float highest = -std::numeric_limits<float>::infinity();
        for (std::uint32_t member = 0U; member < kIndexPool; ++member) {
            const auto logit =
                member_gate[member][channel] +
                pool_ape[static_cast<std::size_t>(member) * kIndexHeadDim +
                         channel];
            probability[member] = logit;
            highest = std::max(highest, logit);
        }
        float total = 0.0F;
        for (auto& value : probability) {
            value = std::exp(value - highest);
            total += value;
        }
        float mixed = 0.0F;
        for (std::uint32_t member = 0U; member < kIndexPool; ++member) {
            mixed += (probability[member] / total) * member_key[member][channel];
        }
        pool_key[channel] = mixed;
    }
}

// `pool_key_at(pool)` returns that complete pool's cached 128-wide key.
template <typename PoolKeyAt>
[[nodiscard]] std::size_t glm53_sparse_index_select(
    std::span<std::uint32_t> selected, std::span<const float> indexer_query,
    PoolKeyAt&& pool_key_at, std::span<const float> head_weights,
    std::uint32_t history, Glm53SparseMlaMetrics* timing = nullptr) {
    if (history <= kIndexTopK) {
        // Selection is the identity here. Returning the dense range keeps the
        // sparse and dense paths bit-identical below the threshold, which is
        // the regression test for everything above it.
        for (std::uint32_t token = 0U; token < history; ++token) {
            selected[token] = token;
        }
        return history;
    }
    const auto pools = history / kIndexPool;          // complete pools only
    const auto tail_count = history - pools * kIndexPool;
    const auto scale = 1.0F / std::sqrt(static_cast<float>(kIndexHeadDim));
    const auto head_scale = 1.0F / std::sqrt(static_cast<float>(kIndexHeads));

    std::vector<std::pair<float, std::uint32_t>> ranked;
    ranked.reserve(pools);

    const auto scoring_started = timing != nullptr
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    for (std::uint32_t pool = 0U; pool < pools; ++pool) {
        const auto* pool_key = pool_key_at(pool);
        float score = 0.0F;
        for (std::uint32_t head = 0U; head < kIndexHeads; ++head) {
            const auto* q = indexer_query.data() +
                            static_cast<std::size_t>(head) * kIndexHeadDim;
            float dot = 0.0F;
            for (std::uint32_t channel = 0U; channel < kIndexHeadDim; ++channel) {
                dot += q[channel] * pool_key[channel];
            }
            // Scale inside the ReLU.
            score += head_weights[head] * head_scale *
                     std::max(0.0F, dot * scale);
        }
        ranked.emplace_back(score, pool);
    }
    if (timing != nullptr) {
        timing->pool_scoring_nanoseconds +=
            elapsed_nanoseconds(scoring_started);
    }

    const auto keep = std::min<std::size_t>(kIndexTopK / kIndexPool, pools);
    const auto sorting_started = timing != nullptr
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    std::partial_sort(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(keep),
                      ranked.end(),
                      [](const auto& left, const auto& right) {
                          if (left.first != right.first) return left.first > right.first;
                          return left.second < right.second;  // stable on ties
                      });
    if (timing != nullptr) {
        timing->topk_sort_nanoseconds +=
            elapsed_nanoseconds(sorting_started);
    }
    std::vector<std::uint32_t> chosen;
    chosen.reserve(keep);
    for (std::size_t index = 0U; index < keep; ++index) {
        chosen.push_back(ranked[index].second);
    }
    std::sort(chosen.begin(), chosen.end());

    std::size_t count = 0U;
    for (const auto pool : chosen) {
        for (std::uint32_t member = 0U; member < kIndexPool; ++member) {
            selected[count++] = pool * kIndexPool + member;
        }
    }
    // Always-selected tail: the current incomplete pool, as raw positions.
    for (std::uint32_t offset = 0U; offset < tail_count; ++offset) {
        selected[count++] = pools * kIndexPool + offset;
    }
    return count;
}

constexpr std::uint64_t kKdaWorkspaceFloats =
    2ULL * kHidden + 6ULL * kLinearWidth + 2ULL * kLinearHead + kHeads;
constexpr std::uint64_t kDeviceWorkspaceReserve = 2ULL << 30U;
constexpr std::uint64_t kMinimumDeviceBudget = 2ULL << 30U;

[[nodiscard]] std::uint64_t host_sequence_state_bytes(
    std::uint32_t maximum_context_tokens) noexcept {
    const auto kda_state = 34ULL * kHeads * kLinearHead * kLinearHead *
                           sizeof(float);
    const auto convolution_state =
        34ULL * 3ULL * kLinearWidth * 3ULL * sizeof(float);
    const auto mla_state = 11ULL * maximum_context_tokens * kKvRank *
                           sizeof(float);
    const auto hidden_history =
        static_cast<std::uint64_t>(maximum_context_tokens) * kHidden *
        sizeof(float);
    const auto logits = static_cast<std::uint64_t>(kVocabulary) *
                        sizeof(float);
    const auto tokens = static_cast<std::uint64_t>(maximum_context_tokens) *
                        sizeof(std::uint32_t);
    return std::max<std::uint64_t>(
        kda_state + convolution_state + mla_state + hidden_history + logits +
            tokens,
        1U);
}

[[nodiscard]] std::size_t host_sequence_capacity(
    std::uint32_t maximum_context_tokens) noexcept {
    const auto state_bytes = host_sequence_state_bytes(maximum_context_tokens);
    const auto budget = host_hardware_profile().host_usable_bytes(0.05);
    if (budget == 0U) return 1U;
    return std::clamp<std::size_t>(
        static_cast<std::size_t>(budget / state_bytes), 1U, 64U);
}

[[nodiscard]] bool batched_projections_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_GLM53_BATCHED_PROJECTIONS");
        return value == nullptr ||
               (std::string_view(value) != "0" &&
                std::string_view(value) != "false" &&
                std::string_view(value) != "off");
    }();
    return enabled;
}

[[nodiscard]] bool cross_gpu_projections_enabled(
    std::span<const int> devices) noexcept {
    static const int policy = [] {
        const char* value = std::getenv("STRATA_GLM53_CROSS_GPU_PROJECTIONS");
        if (value == nullptr) return -1;
        return std::string_view(value) != "0" &&
                       std::string_view(value) != "false" &&
                       std::string_view(value) != "off"
                   ? 1
                   : 0;
    }();
    if (policy >= 0) return policy != 0;
    for (std::size_t source = 0U; source < devices.size(); ++source) {
        for (std::size_t destination = source + 1U;
             destination < devices.size(); ++destination) {
            if (!CudaBackend::high_speed_peer_access_supported(
                    devices[source], devices[destination]) ||
                !CudaBackend::high_speed_peer_access_supported(
                    devices[destination], devices[source])) {
                return false;
            }
        }
    }
    return devices.size() > 1U;
}

[[nodiscard]] bool tensor_parallel_head_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_GLM53_TENSOR_PARALLEL_HEAD");
        return value == nullptr ||
               (std::string_view(value) != "0" &&
                std::string_view(value) != "false" &&
                std::string_view(value) != "off");
    }();
    return enabled;
}

[[nodiscard]] bool full_tensor_parallel_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_GLM53_FULL_TP");
        return value == nullptr ||
               (std::string_view(value) != "0" &&
                std::string_view(value) != "false" &&
                std::string_view(value) != "off");
    }();
    return enabled;
}

[[nodiscard]] bool replay_ssm_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_GLM53_REPLAY_SSM");
        return value == nullptr ||
               (std::string_view(value) != "0" &&
                std::string_view(value) != "false" &&
                std::string_view(value) != "off");
    }();
    return enabled;
}

[[nodiscard]] bool phase_scheduler_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_GLM53_PHASE_SCHEDULER");
        return value == nullptr ||
               (std::string_view(value) != "0" &&
                std::string_view(value) != "false" &&
                std::string_view(value) != "off");
    }();
    return enabled;
}

[[nodiscard]] bool fused_kda_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_GLM53_FUSED_KDA");
        return value == nullptr ||
               (std::string_view(value) != "0" &&
                std::string_view(value) != "false" &&
                std::string_view(value) != "off");
    }();
    return enabled;
}

// Device prefill builds MLA state on the device and back-fills the host cache
// from it. There is no device-side indexer state to back-fill, because the
// device chain does not compute the k-pool keys and gates, so a sequence that
// can cross kIndexTopK must prefill on the host or its indexer history is
// missing exactly where it is first needed.
[[nodiscard]] bool device_prefill_for_context(
    std::uint32_t maximum_context_tokens) noexcept;

[[nodiscard]] bool device_prefill_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_GLM53_DEVICE_PREFILL");
        return value != nullptr && std::string_view(value) != "0" &&
               std::string_view(value) != "false" &&
               std::string_view(value) != "off";
    }();
    return enabled;
}

bool device_prefill_for_context(
    std::uint32_t maximum_context_tokens) noexcept {
    return device_prefill_enabled() &&
           !sparse_indexer_active(maximum_context_tokens);
}

[[nodiscard]] bool device_page_mla_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_GLM53_DEVICE_PAGE_MLA");
        return value == nullptr ||
               (std::string_view(value) != "0" &&
                std::string_view(value) != "false" &&
                std::string_view(value) != "off");
    }();
    return enabled;
}

[[nodiscard]] bool resident_mla_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_GLM53_RESIDENT_MLA");
        // On by default since 2026-08-30. Its exactness gate is closed: under a
        // control that holds the mHC fixed and varies only where the attention
        // runs, three-and-three alternating arms are byte-identical, and decode
        // wall falls 24.0% with MLA down 78.9% and activation D2H down 98.3%
        // (record 0214).
        //
        // Landing it also repairs a latent defect the campaign had not noticed:
        // the model computed mHC two different ways depending on layer type,
        // device for the 34 KDA layers and host for the 11 MLA fallback layers,
        // and those two are not the same function -- they disagree on every
        // layer by up to 0.16 absolute. All 45 layers now use one. The owner
        // ruled this a repair rather than a regression and authorized the
        // re-baseline; figures anchored to `b3deffc5d0f0` are superseded by
        // `fe74dc4ec7ab`.
        return value == nullptr ||
               (std::string_view(value) != "0" &&
                std::string_view(value) != "false" &&
                std::string_view(value) != "off");
    }();
    return enabled;
}

// Runs the MLA attention on the accepted host fallback while the layer keeps
// the *device* mHC. This is the control the resident MLA candidate actually
// needs, and it is not the production fallback.
//
// Host `mhc_pre` and the device mHC are not the same function: they disagree on
// every layer, and flipping all 45 layers to the host path changes the
// generated text (`bf2016cd3e39` against the reference `b3deffc5d0f0`). The
// reference encodes a mixture -- device mHC on the 34 KDA layers, host mHC on
// the 11 MLA fallback layers -- so enabling resident MLA moves those 11 layers
// between two different mHC implementations and its output must change however
// exact its attention is. Comparing against the reference therefore measures
// the mHC swap, not the attention.
//
// With this flag the control keeps device mHC on all 45 layers and differs
// from the candidate only in where the MLA attention runs, which is the
// variable under test.
[[nodiscard]] bool resident_mla_host_attention() noexcept {
    static const bool enabled = [] {
        const char* value =
            std::getenv("STRATA_GLM53_RESIDENT_MLA_HOST_ATTENTION");
        return value != nullptr && std::string_view(value) != "0" &&
               std::string_view(value) != "false" &&
               std::string_view(value) != "off";
    }();
    return enabled;
}

[[nodiscard]] bool resident_mla_compare_enabled() noexcept {
    const char* value = std::getenv("STRATA_GLM53_RESIDENT_MLA_COMPARE");
    return value != nullptr && std::string_view(value) != "0" &&
           std::string_view(value) != "false" &&
           std::string_view(value) != "off";
}

[[nodiscard]] bool profiler_capture_enabled() noexcept {
    const char* value = std::getenv("STRATA_GLM53_NSYS_CAPTURE");
    return value != nullptr && std::string_view(value) != "0" &&
           std::string_view(value) != "false" &&
           std::string_view(value) != "off";
}

// -1 selects from the discovered CPU width and the admitted CUDA residency;
// 0/1 are explicit campaign overrides.
[[nodiscard]] int host_moe_override() noexcept {
    const char* value = std::getenv("STRATA_GLM53_HOST_MOE");
    if (value == nullptr || std::string_view(value) == "auto") return -1;
    return std::string_view(value) != "0" &&
                   std::string_view(value) != "false" &&
                   std::string_view(value) != "off"
               ? 1 : 0;
}

// The shared expert is the ninth of the nine every MoE layer runs and the only
// one the router does not choose, so it can be computed on the GPU that owns
// the layer while the host works through the eight routed ones. The device dot
// associates its sum exactly as the host AVX2 dot does and every rounding stays
// on the host, so the tier is bit-exact -- 129-token decode is byte-identical
// across five alternating arms.
//
// On by default. It was off from 0213 until 0221, on a measurement that is now
// stale rather than wrong: three device arms then measured +1.71% median decode
// wall and only -0.21% host expert service, so an 11.1% cut in host expert
// bytes bought nothing. That audit predates the resident MLA chain (0214),
// which took activation D2H from 2.265 to 0.002 GB per token and CUDA
// synchronizations down 49.5%. That dependency change shortened the measured
// collect wait enough for the host work removed by the tier to dominate. The
// later 0224 audit found the old aggregate CUDA timer incomplete, so no idle-
// device claim is carried forward; the protected wall-time and collect-wait
// measurements below remain direct observations.
//
// Re-measured in 0221 as 3+3 alternating pairs with an ordering repeat, twice,
// because a shipped default must not rest on a benchmark-only memory policy:
//
//   interleaved     decode 50.130 -> 46.650 s  -6.94%, service -7.77%
//   default policy  decode 50.360 -> 46.890 s  -6.89%, service -7.50%
//
// Decomposed: 4.1-4.3 s of host dispatch work removed against 0.76 s of collect
// wait added. Ranges do not overlap in either pair, ordering repeats drift
// +/-0.20%, and all sixteen outputs hash `fe74dc4ec7ab`. `0` opts out.
//
// -1 is the default admission and 1 an explicit request; they differ only in
// how a checkpoint that cannot fit the whole tier is treated. The descriptor
// selects the checkpoint-native FP8 or BF16 dot, so both published releases
// use the same overlap without changing their stored representation.
[[nodiscard]] int shared_expert_device_override() noexcept {
    const char* value = std::getenv("STRATA_GLM53_SHARED_EXPERT_DEVICE");
    if (value == nullptr || std::string_view(value) == "auto") return -1;
    return std::string_view(value) != "0" &&
                   std::string_view(value) != "false" &&
                   std::string_view(value) != "off"
               ? 1 : 0;
}

// Where to write the M4 route census, or empty for no census. Section 11's
// first gate item needs the sequence, not just frequencies: byte-weighted
// coverage answers "would a resident tier of size K hold this token's experts"
// and reuse distance answers "would it still hold them next token", and only
// the second needs order.
[[nodiscard]] const std::string& route_census_path() {
    static const std::string path = [] {
        const char* value = std::getenv("STRATA_GLM53_ROUTE_CENSUS");
        return value == nullptr ? std::string{} : std::string(value);
    }();
    return path;
}

// The accepted static routed tier is capability-driven and default-on. `0`
// retains a same-binary control; explicit census paths remain an experimental
// retraining hook and replace the built-in representative ranking.
[[nodiscard]] int static_expert_override() noexcept {
    const char* value = std::getenv("STRATA_GLM53_STATIC_EXPERT");
    if (value == nullptr || std::string_view(value) == "auto") return -1;
    return std::string_view(value) != "0" &&
                   std::string_view(value) != "false" &&
                   std::string_view(value) != "off"
               ? 1 : 0;
}

// Multiple already-captured route census TSVs are separated by semicolons.
[[nodiscard]] const std::string& static_expert_census_paths() {
    static const std::string paths = [] {
        const char* value = std::getenv("STRATA_GLM53_STATIC_EXPERT_CENSUS");
        return value == nullptr ? std::string{} : std::string(value);
    }();
    return paths;
}

[[nodiscard]] bool mtp_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_GLM53_MTP");
        // The checkpoint MTP layer is fully wired, but verification only pays
        // when its measured acceptance rate amortizes the extra draft pass.
        // Keep the production latency route deterministic and opt in to an
        // MTP campaign explicitly until that gate has been established for a
        // workload.
        return value != nullptr && std::string_view(value) != "0" &&
                std::string_view(value) != "false" &&
                std::string_view(value) != "off";
    }();
    return enabled;
}

struct Glm53RowRange {
    std::uint64_t begin{};
    std::uint64_t count{};
};

[[nodiscard]] std::vector<Glm53RowRange> weighted_row_ranges(
    std::uint64_t rows, std::span<const std::uint64_t> capacities,
    std::uint64_t alignment) {
    if (rows == 0U || capacities.empty() || alignment == 0U ||
        std::any_of(capacities.begin(), capacities.end(),
                    [](std::uint64_t value) { return value == 0U; })) {
        return {};
    }
    if (rows < capacities.size() * alignment) {
        alignment = 1U;
    }
    long double total_capacity = 0.0L;
    for (const auto capacity : capacities) {
        total_capacity += static_cast<long double>(capacity);
    }
    std::vector<Glm53RowRange> ranges;
    ranges.reserve(capacities.size());
    std::uint64_t begin = 0U;
    long double cumulative = 0.0L;
    for (std::size_t slot = 0U; slot < capacities.size(); ++slot) {
        std::uint64_t end = rows;
        if (slot + 1U != capacities.size()) {
            cumulative += static_cast<long double>(capacities[slot]);
            const auto target = static_cast<std::uint64_t>(
                static_cast<long double>(rows) * cumulative / total_capacity);
            end = target - target % alignment;
            const auto minimum = begin + alignment;
            const auto remaining = static_cast<std::uint64_t>(
                capacities.size() - slot - 1U) * alignment;
            end = std::clamp(end, minimum, rows - remaining);
        }
        ranges.push_back({begin, end - begin});
        begin = end;
    }
    return ranges;
}

[[nodiscard]] std::vector<std::size_t> contiguous_layer_schedule(
    std::uint32_t layers, std::span<const std::uint64_t> capacities) {
    const auto ranges = weighted_row_ranges(layers, capacities, 1U);
    if (ranges.size() != capacities.size()) return {};
    std::vector<std::size_t> schedule(layers);
    for (std::size_t slot = 0U; slot < ranges.size(); ++slot) {
        const auto range = ranges[slot];
        for (std::uint64_t layer = range.begin;
             layer < range.begin + range.count; ++layer) {
            schedule[static_cast<std::size_t>(layer)] = slot;
        }
    }
    return schedule;
}

[[nodiscard]] std::vector<int> projection_worker_cpus(
    std::span<const int> devices) {
    const auto& hardware = host_hardware_profile();
    std::vector<int> chosen;
    chosen.reserve(devices.size());
    const auto usable = [&](int cpu) {
        return std::find(hardware.usable_cpu_ids.begin(),
                         hardware.usable_cpu_ids.end(), cpu) !=
               hardware.usable_cpu_ids.end();
    };
    const auto available = [&](int cpu) {
        return usable(cpu) &&
               std::find(chosen.begin(), chosen.end(), cpu) == chosen.end();
    };
    for (const int device : devices) {
        const int node = CudaBackend::device_numa_node(device);
        const std::vector<int>* local = nullptr;
        if (node >= 0 && static_cast<std::size_t>(node) <
                             hardware.numa.node_primary_cpus.size() &&
            !hardware.numa.node_primary_cpus[static_cast<std::size_t>(node)]
                 .empty()) {
            local = &hardware.numa.node_primary_cpus[
                static_cast<std::size_t>(node)];
        } else if (node >= 0 && static_cast<std::size_t>(node) <
                                    hardware.numa.node_cpus.size()) {
            local = &hardware.numa.node_cpus[static_cast<std::size_t>(node)];
        }
        auto selected = hardware.usable_cpu_ids.end();
        if (local != nullptr) {
            const auto candidate = std::find_if(
                local->begin(), local->end(), available);
            if (candidate != local->end()) {
                selected = std::find(hardware.usable_cpu_ids.begin(),
                                     hardware.usable_cpu_ids.end(), *candidate);
            }
        }
        if (selected == hardware.usable_cpu_ids.end()) {
            selected = std::find_if(hardware.usable_cpu_ids.begin(),
                                    hardware.usable_cpu_ids.end(), available);
        }
        if (selected == hardware.usable_cpu_ids.end()) return {};
        chosen.push_back(*selected);
    }
    return chosen;
}

[[nodiscard]] std::vector<int> compute_worker_cpus() {
    const auto& hardware = host_hardware_profile();
    std::vector<int> cpus;
    const auto usable = [&](int cpu) {
        return std::find(hardware.usable_cpu_ids.begin(),
                         hardware.usable_cpu_ids.end(), cpu) !=
               hardware.usable_cpu_ids.end();
    };
    for (const auto& node : hardware.numa.node_primary_cpus) {
        for (const int cpu : node) {
            if (usable(cpu)) cpus.push_back(cpu);
        }
    }
    if (cpus.empty()) cpus = hardware.usable_cpu_ids;
    return cpus;
}

double now_seconds() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

float sigmoid(float value) noexcept {
    return value >= 0.0F ? 1.0F / (1.0F + std::exp(-value))
                         : std::exp(value) / (1.0F + std::exp(value));
}

void round_bf16(std::span<float> values) noexcept {
    for (auto& value : values) value = bf16_round_f32(value);
}

void append(std::vector<std::string>& destination,
            std::vector<std::string> source) {
    for (auto& error : source) destination.push_back(std::move(error));
}

[[nodiscard]] std::string projection_group_key(
    std::string_view base, Glm53TensorRole role) {
    const auto separator = base.find_last_of('.');
    const auto prefix = base.substr(0U, separator + 1U);
    const auto leaf = base.substr(separator + 1U);
    if (role == Glm53TensorRole::KdaAttention) {
        if (leaf == "q_proj" || leaf == "k_proj" || leaf == "v_proj" ||
            leaf == "f_a_proj" || leaf == "b_proj" || leaf == "g_a_proj") {
            return std::string(prefix) + "#kda-input";
        }
        if (leaf == "f_b_proj" || leaf == "g_b_proj") {
            return std::string(prefix) + "#kda-low-rank";
        }
    } else if (role == Glm53TensorRole::SparseAttention) {
        if (leaf == "q_a_proj" || leaf == "kv_a_proj_with_mqa") {
            return std::string(prefix) + "#mla-input";
        }
        if (leaf == "q_b_proj" || leaf == "kv_b_proj") {
            return std::string(prefix) + "#mla-expanded";
        }
    } else if (role == Glm53TensorRole::DenseMlp &&
               (leaf == "gate_proj" || leaf == "up_proj")) {
        return std::string(prefix) + "#dense-gate-up";
    }
    return std::string(base);
}

class Glm53WeightCache {
    struct Entry {
        CudaWeight weight;
        bool pinned{};
        bool prefetched{};
        std::uint32_t leases{};
        std::list<std::string>::iterator recency;
    };

    struct State {
        std::mutex mutex;
        std::unordered_map<std::string, Entry> entries;
        std::list<std::string> recency;
        std::uint64_t capacity{};
        std::uint64_t used{};
        std::uint64_t pinned{};
        std::uint64_t hits{};
        std::uint64_t misses{};
        std::uint64_t evictions{};
        std::uint64_t prefetches{};
        std::uint64_t useful_prefetches{};
        std::uint64_t failed_prefetches{};
    };

public:
    struct LinearRequest {
        std::string_view base;
        std::uint64_t output_columns{};
        std::uint64_t input_columns{};
        std::span<const float> input;
        std::uint32_t rows{};
        std::span<float> output;
        bool bf16_output{};
        std::uint64_t weight_rows{};
        std::uint64_t weight_row_begin{};
    };

    struct Stats {
        std::vector<std::uint64_t> capacity;
        std::vector<std::uint64_t> used;
        std::vector<std::uint64_t> pinned;
        std::uint64_t hits{};
        std::uint64_t misses{};
        std::uint64_t evictions{};
        std::uint64_t prefetches{};
        std::uint64_t useful_prefetches{};
        std::uint64_t failed_prefetches{};
    };

    Glm53WeightCache(Glm53CheckpointReader& checkpoint, CudaBackend& backend,
                     std::vector<int> devices,
                     std::vector<std::uint64_t> capacities)
        : checkpoint_(checkpoint), backend_(backend),
          devices_(std::move(devices)) {
        std::uint64_t largest_linear = 0U;
        for (const auto& tensor : checkpoint_.manifest().tensors) {
            if ((tensor.role != Glm53TensorRole::RoutedExpert &&
                 tensor.role != Glm53TensorRole::SharedExpert) ||
                !tensor.name.ends_with(".weight") ||
                tensor.source_shape.size() != 2U) {
                continue;
            }
            largest_linear = std::max(
                largest_linear,
                checkpoint_.cuda_linear_storage_bytes(
                    tensor.name.substr(0U, tensor.name.size() - 7U)));
        }
        const auto fragmentation_reserve =
            largest_linear <= std::numeric_limits<std::uint64_t>::max() / 2U
                ? 2U * largest_linear
                : largest_linear;
        states_.reserve(capacities.size());
        for (const auto capacity : capacities) {
            auto state = std::make_unique<State>();
            state->capacity = capacity > fragmentation_reserve
                ? capacity - fragmentation_reserve : capacity;
            states_.push_back(std::move(state));
        }
    }

    [[nodiscard]] ValidationResult preload(
        std::size_t slot, std::string_view base, std::uint64_t rows,
        std::uint64_t columns, bool& admitted) {
        admitted = false;
        const auto bytes = checkpoint_.cuda_linear_storage_bytes(base);
        if (slot >= states_.size() || bytes == 0U) {
            return {{"GLM-5.3 preload references an invalid CUDA linear"}};
        }
        auto& state = *states_[slot];
        std::scoped_lock lock(state.mutex);
        auto found = state.entries.find(std::string(base));
        if (found != state.entries.end()) {
            if (!found->second.pinned) {
                found->second.pinned = true;
                state.pinned += found->second.weight.device_bytes();
                state.recency.erase(found->second.recency);
            }
            admitted = true;
            ++state.hits;
            return {};
        }
        // A smaller or busier GPU may not fit its complete share of the
        // resident spine. Skipping residency changes only performance: the
        // exact weight is admitted through the demand/LRU path when needed.
        if (bytes > state.capacity - state.used) return {};
        Entry entry;
        auto loaded = checkpoint_.load_cuda_linear(
            base, rows, columns, devices_[slot], backend_, entry.weight);
        if (!loaded.ok()) return loaded;
        entry.pinned = true;
        const auto actual = entry.weight.device_bytes();
        if (actual > state.capacity - state.used) {
            return {{"GLM-5.3 resident linear exceeded its admitted CUDA cache"}};
        }
        state.used += actual;
        state.pinned += actual;
        state.entries.emplace(std::string(base), std::move(entry));
        admitted = true;
        ++state.misses;
        return {};
    }

    [[nodiscard]] ValidationResult preload_slice(
        std::size_t slot, std::string_view base, std::uint64_t total_rows,
        std::uint64_t columns, std::uint64_t row_begin,
        std::uint64_t row_count, bool& admitted) {
        admitted = false;
        const auto key = slice_key(base, row_begin, row_count);
        const auto bytes = checkpoint_.cuda_linear_slice_storage_bytes(
            base, row_begin, row_count);
        if (slot >= states_.size() || bytes == 0U) {
            return {{"GLM-5.3 preload references an invalid CUDA slice"}};
        }
        auto& state = *states_[slot];
        std::scoped_lock lock(state.mutex);
        auto found = state.entries.find(key);
        if (found != state.entries.end()) {
            admitted = true;
            ++state.hits;
            return {};
        }
        if (bytes > state.capacity - state.used) return {};
        Entry entry;
        auto loaded = checkpoint_.load_cuda_linear_slice(
            base, total_rows, columns, row_begin, row_count, devices_[slot],
            backend_, entry.weight);
        if (!loaded.ok()) return loaded;
        entry.pinned = true;
        const auto actual = entry.weight.device_bytes();
        if (actual > state.capacity - state.used) {
            return {{"GLM-5.3 resident slice exceeded its admitted CUDA cache"}};
        }
        state.used += actual;
        state.pinned += actual;
        state.entries.emplace(key, std::move(entry));
        admitted = true;
        ++state.misses;
        return {};
    }

    // Pin one complete routed expert atomically inside the already-admitted
    // weight arena. Static residency consumes the arena's measured reusable
    // tail; it never allocates a second representation outside the ledger.
    [[nodiscard]] ValidationResult pin_expert(
        std::size_t slot, std::uint32_t layer, std::uint32_t expert,
        CudaGlm53Expert& descriptor, bool& admitted) {
        admitted = false;
        if (slot >= states_.size() || layer >= kLayers || expert >= 288U ||
            !glm53_moe_layer(layer)) {
            return {{"GLM-5.3 static expert references an invalid target"}};
        }
        const auto prefix = "model.language_model.layers." +
                            std::to_string(layer) + ".mlp.experts." +
                            std::to_string(expert) + ".";
        struct Projection {
            std::string key;
            std::uint64_t rows{};
            std::uint64_t columns{};
        };
        const std::array<Projection, 3U> projections{{
            {prefix + "gate_proj", 2048U, kHidden},
            {prefix + "up_proj", 2048U, kHidden},
            {prefix + "down_proj", kHidden, 2048U}}};
        auto& state = *states_[slot];
        std::scoped_lock lock(state.mutex);
        std::uint64_t required = 0U;
        for (const auto& projection : projections) {
            if (!state.entries.contains(projection.key)) {
                const auto bytes =
                    checkpoint_.cuda_linear_storage_bytes(projection.key);
                if (bytes == 0U) {
                    return {{"GLM-5.3 static expert projection is absent: " +
                             projection.key}};
                }
                required += bytes;
            }
        }
        if (required > state.capacity - state.used) return {};
        std::array<Entry*, 3U> entries{};
        for (std::size_t index = 0U; index < projections.size(); ++index) {
            const auto& projection = projections[index];
            auto found = state.entries.find(projection.key);
            if (found == state.entries.end()) {
                Entry entry;
                auto loaded = checkpoint_.load_cuda_linear(
                    projection.key, projection.rows, projection.columns,
                    devices_[slot], backend_, entry.weight, false, true);
                if (!loaded.ok()) return loaded;
                entry.pinned = true;
                const auto actual = entry.weight.device_bytes();
                if (actual > state.capacity - state.used) {
                    return {{"GLM-5.3 static expert exceeded its admitted "
                             "CUDA arena"}};
                }
                state.used += actual;
                state.pinned += actual;
                found = state.entries.emplace(projection.key,
                                              std::move(entry)).first;
                ++state.misses;
            } else if (!found->second.pinned) {
                found->second.pinned = true;
                state.pinned += found->second.weight.device_bytes();
                state.recency.erase(found->second.recency);
                ++state.hits;
            }
            entries[index] = &found->second;
        }
        const auto* weight = checkpoint_.find(projections[0].key + ".weight");
        if (weight == nullptr) {
            return {{"GLM-5.3 static expert has no source descriptor"}};
        }
        CudaGlm53ExpertEncoding encoding{};
        if (weight->source_dtype == SafetensorsDtype::F8E4M3) {
            encoding = CudaGlm53ExpertEncoding::Fp8E4m3Block128F32;
        } else if (weight->source_dtype == SafetensorsDtype::U8) {
            encoding = CudaGlm53ExpertEncoding::Fp4E2m1Group32E8m0;
        } else if (weight->source_dtype == SafetensorsDtype::Bf16) {
            encoding = CudaGlm53ExpertEncoding::Bf16;
        } else {
            return {{"GLM-5.3 static expert encoding is unsupported"}};
        }
        descriptor = {};
        descriptor.hidden = kHidden;
        descriptor.intermediate = 2048U;
        descriptor.encoding = encoding;
        descriptor.gate = &entries[0]->weight;
        descriptor.up = &entries[1]->weight;
        descriptor.down = &entries[2]->weight;
        admitted = true;
        return {};
    }

    [[nodiscard]] ValidationResult matmul(
        std::size_t slot, std::string_view base, std::uint64_t output_columns,
        std::uint64_t input_columns, std::span<const float> input,
        std::uint32_t rows, std::span<float> output, bool bf16_output) {
        const LinearRequest request{base, output_columns, input_columns, input,
                                    rows, output, bf16_output, 0U, 0U};
        return matmul_batch(slot, std::span<const LinearRequest>(&request, 1U));
    }

    [[nodiscard]] ValidationResult matmul_batch(
        std::size_t slot, std::span<const LinearRequest> requests) {
        if (slot >= states_.size()) {
            return {{"GLM-5.3 linear targets an invalid CUDA cache slot"}};
        }
        if (requests.empty()) {
            return {{"GLM-5.3 linear batch is empty"}};
        }
        auto& state = *states_[slot];
        std::scoped_lock lock(state.mutex);
        struct BatchLeases {
            State& state;
            std::vector<std::string> keys;
            ~BatchLeases() {
                for (const auto& key : keys) {
                    const auto found = state.entries.find(key);
                    if (found != state.entries.end() &&
                        found->second.leases != 0U) {
                        --found->second.leases;
                    }
                }
            }
        } leases{state, {}};
        leases.keys.reserve(requests.size());
        std::vector<CudaMatmulBatchItem> batch;
        batch.reserve(requests.size());
        for (const auto& request : requests) {
            const bool sliced = request.weight_rows != 0U;
            const std::string key = sliced
                ? slice_key(request.base, request.weight_row_begin,
                            request.output_columns)
                : std::string(request.base);
            auto found = state.entries.find(key);
            if (found == state.entries.end()) {
                const auto bytes = sliced
                    ? checkpoint_.cuda_linear_slice_storage_bytes(
                          request.base, request.weight_row_begin,
                          request.output_columns)
                    : checkpoint_.cuda_linear_storage_bytes(request.base);
                if (bytes == 0U || bytes > state.capacity) {
                    return {{"GLM-5.3 linear is absent or exceeds its CUDA cache: " +
                             key}};
                }
                while (state.used + bytes > state.capacity) {
                    auto victim_position = state.recency.end();
                    for (auto candidate = state.recency.begin();
                         candidate != state.recency.end(); ++candidate) {
                        const auto entry = state.entries.find(*candidate);
                        if (entry != state.entries.end() &&
                            entry->second.leases == 0U) {
                            victim_position = candidate;
                            break;
                        }
                    }
                    if (victim_position == state.recency.end()) {
                        return {{"GLM-5.3 pinned spine leaves insufficient CUDA "
                                 "cache for an exact demand weight"}};
                    }
                    const auto victim_key = *victim_position;
                    state.recency.erase(victim_position);
                    auto victim = state.entries.find(victim_key);
                    if (victim == state.entries.end() || victim->second.pinned) {
                        return {{"GLM-5.3 CUDA cache recency bookkeeping is invalid"}};
                    }
                    state.used -= victim->second.weight.device_bytes();
                    state.entries.erase(victim);
                    ++state.evictions;
                }
                Entry entry;
                const auto load = [&] {
                    return sliced
                        ? checkpoint_.load_cuda_linear_slice(
                              request.base, request.weight_rows,
                              request.input_columns, request.weight_row_begin,
                              request.output_columns, devices_[slot], backend_,
                              entry.weight)
                        : checkpoint_.load_cuda_linear(
                              request.base, request.output_columns,
                              request.input_columns, devices_[slot], backend_,
                              entry.weight);
                };
                auto loaded = load();
                while (!loaded.ok() && arena_exhausted(loaded) &&
                       evict_one(state)) {
                    entry.weight = CudaWeight{};
                    loaded = load();
                }
                if (!loaded.ok()) return loaded;
                const auto actual = entry.weight.device_bytes();
                if (actual > state.capacity - state.used) {
                    return {{"GLM-5.3 demand linear exceeded its admitted CUDA cache"}};
                }
                state.recency.push_back(key);
                entry.recency = std::prev(state.recency.end());
                state.used += actual;
                found = state.entries.emplace(key, std::move(entry)).first;
                ++state.misses;
            } else {
                ++state.hits;
                if (found->second.prefetched) {
                    found->second.prefetched = false;
                    ++state.useful_prefetches;
                }
                if (!found->second.pinned) {
                    state.recency.splice(state.recency.end(), state.recency,
                                         found->second.recency);
                }
            }
            ++found->second.leases;
            leases.keys.push_back(key);
            batch.push_back({&found->second.weight, request.input, request.rows,
                             request.output, request.bf16_output,
                             request.rows > 1U});
        }
        // One device-side event orders all deferred cache-miss uploads before
        // the consumer. This never blocks the host and is a no-op on a hit-only
        // path; matmul's output completion still protects the LRU entry.
        if (auto ordered = backend_.synchronize_uploads(devices_[slot]);
            !ordered.ok()) {
            return ordered;
        }
        return backend_.matmul_batch(batch);
    }

    // Admit one routed expert without consuming it. The worker holds the
    // device cache lock through the copy-stream completion, so a demand can
    // never observe a half-uploaded entry and an eviction cannot recycle its
    // arena storage early. Storage faults and H2D copies happen on the worker,
    // concurrently with the preceding layer's compute stream.
    [[nodiscard]] ValidationResult prefetch_expert(
        std::size_t slot, std::uint32_t layer, std::uint32_t expert) {
        if (slot >= states_.size() || layer >= kLayers || expert >= 288U ||
            !glm53_moe_layer(layer)) {
            return { {"GLM-5.3 expert prefetch has an invalid target"} };
        }
        const auto prefix = "model.language_model.layers." +
                            std::to_string(layer) + ".mlp.experts." +
                            std::to_string(expert) + ".";
        struct Projection {
            std::string key;
            std::uint64_t rows{};
            std::uint64_t columns{};
        };
        const std::array<Projection, 3U> projections{{
            {prefix + "gate_proj", 2048U, kHidden},
            {prefix + "up_proj", 2048U, kHidden},
            {prefix + "down_proj", kHidden, 2048U}}};
        auto& state = *states_[slot];
        std::scoped_lock lock(state.mutex);
        bool admitted = false;
        for (const auto& projection : projections) {
            if (state.entries.contains(projection.key)) continue;
            const auto bytes = checkpoint_.cuda_linear_storage_bytes(
                projection.key);
            if (bytes == 0U || bytes > state.capacity) {
                ++state.failed_prefetches;
                return {};
            }
            while (state.used + bytes > state.capacity) {
                auto victim = state.recency.end();
                for (auto candidate = state.recency.begin();
                     candidate != state.recency.end(); ++candidate) {
                    const auto found = state.entries.find(*candidate);
                    if (found != state.entries.end() &&
                        !found->second.pinned && found->second.leases == 0U) {
                        victim = candidate;
                        break;
                    }
                }
                if (victim == state.recency.end()) {
                    ++state.failed_prefetches;
                    return {};
                }
                auto found = state.entries.find(*victim);
                state.used -= found->second.weight.device_bytes();
                state.entries.erase(found);
                state.recency.erase(victim);
                ++state.evictions;
            }
            Entry entry;
            const auto load = [&] {
                return checkpoint_.load_cuda_linear(
                    projection.key, projection.rows, projection.columns,
                    devices_[slot], backend_, entry.weight, true);
            };
            auto loaded = load();
            while (!loaded.ok() && arena_exhausted(loaded) &&
                   evict_one(state)) {
                entry.weight = CudaWeight{};
                loaded = load();
            }
            if (!loaded.ok()) {
                ++state.failed_prefetches;
                return loaded;
            }
            const auto actual = entry.weight.device_bytes();
            if (actual > state.capacity - state.used) {
                ++state.failed_prefetches;
                return {};
            }
            state.recency.push_back(projection.key);
            entry.recency = std::prev(state.recency.end());
            entry.prefetched = true;
            state.used += actual;
            state.entries.emplace(projection.key, std::move(entry));
            admitted = true;
        }
        if (!admitted) return {};
        auto ordered = backend_.synchronize_uploads(devices_[slot]);
        if (!ordered.ok()) {
            ++state.failed_prefetches;
            return ordered;
        }
        ++state.prefetches;
        return {};
    }

    [[nodiscard]] bool contains_expert(
        std::size_t slot, std::uint32_t layer, std::uint32_t expert) const {
        if (slot >= states_.size()) return false;
        const auto prefix = "model.language_model.layers." +
                            std::to_string(layer) + ".mlp.experts." +
                            std::to_string(expert) + ".";
        auto& state = *states_[slot];
        std::scoped_lock lock(state.mutex);
        return state.entries.contains(prefix + "gate_proj") &&
               state.entries.contains(prefix + "up_proj") &&
               state.entries.contains(prefix + "down_proj");
    }

    [[nodiscard]] ValidationResult kda_decode(
        std::size_t slot, std::string_view attention,
        CudaGlm53KdaRequest request, std::span<float> output) {
        if (slot >= states_.size() || request.state == nullptr) {
            return {{"GLM-5.3 fused KDA targets an invalid CUDA cache slot"}};
        }
        auto& state = *states_[slot];
        std::scoped_lock lock(state.mutex);
        const std::array<std::string, 9U> keys{
            std::string(attention) + "q_proj",
            std::string(attention) + "k_proj",
            std::string(attention) + "v_proj",
            std::string(attention) + "f_a_proj",
            std::string(attention) + "b_proj",
            std::string(attention) + "g_a_proj",
            std::string(attention) + "f_b_proj",
            std::string(attention) + "g_b_proj",
            std::string(attention) + "o_proj"};
        std::array<Entry*, 9U> entries{};
        const auto first = request.input.empty() &&
                                   !request.mhc_source_destination
                               ? keys.size() - 1U
                               : 0U;
        for (std::size_t index = first; index < keys.size(); ++index) {
            const auto found = state.entries.find(keys[index]);
            if (found == state.entries.end() ||
                found->second.weight.device() != request.state->device()) {
                return {{"GLM-5.3 fused KDA projection was not admitted on "
                         "its layer device: " + keys[index]}};
            }
            entries[index] = &found->second;
        }
        if (entries.back() == nullptr) {
            return {{"GLM-5.3 fused KDA output projection was not admitted "
                     "on its layer device"}};
        }
        struct Lease {
            std::span<Entry* const> entries;
            ~Lease() {
                for (auto* entry : entries) {
                    if (entry != nullptr) --entry->leases;
                }
            }
        } lease{entries};
        for (auto* entry : entries) {
            if (entry != nullptr) ++entry->leases;
        }
        request.query_projection = entries[0] == nullptr
            ? nullptr : &entries[0]->weight;
        request.key_projection = entries[1] == nullptr
            ? nullptr : &entries[1]->weight;
        request.value_projection = entries[2] == nullptr
            ? nullptr : &entries[2]->weight;
        request.forget_a_projection = entries[3] == nullptr
            ? nullptr : &entries[3]->weight;
        request.beta_projection = entries[4] == nullptr
            ? nullptr : &entries[4]->weight;
        request.gate_a_projection = entries[5] == nullptr
            ? nullptr : &entries[5]->weight;
        request.forget_b_projection = entries[6] == nullptr
            ? nullptr : &entries[6]->weight;
        request.gate_b_projection = entries[7] == nullptr
            ? nullptr : &entries[7]->weight;
        request.output_projection = &entries[8]->weight;
        return backend_.glm53_kda_decode(request, output);
    }

    [[nodiscard]] ValidationResult router_mhc(
        std::size_t slot, std::string_view key, std::span<float> logits) {
        if (slot >= states_.size()) {
            return {{"GLM-5.3 resident router targets an invalid cache slot"}};
        }
        auto& state = *states_[slot];
        std::scoped_lock lock(state.mutex);
        const auto found = state.entries.find(std::string(key));
        if (found == state.entries.end() ||
            found->second.weight.device() != devices_[slot]) {
            return {{"GLM-5.3 resident router was not admitted on its layer "
                     "device: " + std::string(key)}};
        }
        ++found->second.leases;
        struct Lease {
            Entry& entry;
            ~Lease() { --entry.leases; }
        } lease{found->second};
        return backend_.glm53_mhc_router(
            devices_[slot], found->second.weight, logits);
    }

    [[nodiscard]] ValidationResult mla_decode_mhc(
        std::size_t slot, std::string_view attention,
        CudaGlm53MlaRequest request, std::span<float> scores,
        const std::function<void(std::span<float>, std::uint32_t,
                                 std::uint32_t)>& softmax) {
        if (slot >= states_.size() || request.state == nullptr) {
            return {{"GLM-5.3 resident MLA targets an invalid cache slot"}};
        }
        auto& state = *states_[slot];
        std::scoped_lock lock(state.mutex);
        const std::array<std::string, 5U> keys{
            std::string(attention) + "q_a_proj",
            std::string(attention) + "kv_a_proj_with_mqa",
            std::string(attention) + "q_b_proj",
            std::string(attention) + "kv_b_proj",
            std::string(attention) + "o_proj"};
        std::array<Entry*, 5U> entries{};
        for (std::size_t index = 0U; index < keys.size(); ++index) {
            const auto found = state.entries.find(keys[index]);
            if (found == state.entries.end() ||
                found->second.weight.device() != request.state->device()) {
                return {{"GLM-5.3 resident MLA projection was not admitted: " +
                         keys[index]}};
            }
            entries[index] = &found->second;
        }
        for (auto* entry : entries) ++entry->leases;
        struct Lease {
            std::array<Entry*, 5U>& entries;
            ~Lease() {
                for (auto* entry : entries) --entry->leases;
            }
        } lease{entries};
        request.query_a = &entries[0]->weight;
        request.key_value_a = &entries[1]->weight;
        request.query_b = &entries[2]->weight;
        request.key_value_b = &entries[3]->weight;
        request.output = &entries[4]->weight;
        // Two phases with the softmax between them, on the host. The device
        // returns raw scores; `softmax` turns them into the BF16 coefficients
        // the accepted fallback would have produced, using that fallback's own
        // arithmetic; the device then finishes the layer. Leases are held
        // across both, which is why this is one call from the caller's side.
        auto scored = backend_.glm53_mla_decode_to_mhc(request, scores);
        if (!scored.ok()) return scored;
        softmax(scores, request.heads,
                static_cast<std::uint32_t>(request.position) + 1U);
        return backend_.glm53_mla_decode_finish(request, scores);
    }

    [[nodiscard]] ValidationResult sparse_mla_decode_mhc(
        std::size_t slot, std::string_view attention,
        CudaGlm53MlaRequest request, std::span<float> scores,
        const std::function<void(std::span<float>, std::uint32_t,
                                 std::uint32_t)>& softmax) {
        if (slot >= states_.size() || request.state == nullptr) {
            return {{"GLM-5.3 sparse resident MLA targets an invalid cache "
                     "slot"}};
        }
        auto& state = *states_[slot];
        std::scoped_lock lock(state.mutex);
        const std::array<std::string, 5U> keys{
            std::string(attention) + "q_a_proj",
            std::string(attention) + "kv_a_proj_with_mqa",
            std::string(attention) + "q_b_proj",
            std::string(attention) + "kv_b_proj",
            std::string(attention) + "o_proj"};
        std::array<Entry*, 5U> entries{};
        for (std::size_t index = 0U; index < keys.size(); ++index) {
            const auto found = state.entries.find(keys[index]);
            if (found == state.entries.end() ||
                found->second.weight.device() != request.state->device()) {
                return {{"GLM-5.3 sparse resident MLA projection was not "
                         "admitted: " + keys[index]}};
            }
            entries[index] = &found->second;
        }
        for (auto* entry : entries) ++entry->leases;
        struct Lease {
            std::array<Entry*, 5U>& entries;
            ~Lease() {
                for (auto* entry : entries) --entry->leases;
            }
        } lease{entries};
        request.query_a = &entries[0]->weight;
        request.key_value_a = &entries[1]->weight;
        request.query_b = &entries[2]->weight;
        request.key_value_b = &entries[3]->weight;
        request.output = &entries[4]->weight;
        auto scored = backend_.glm53_sparse_mla_decode_to_mhc(
            request, scores);
        if (!scored.ok()) return scored;
        const auto attended_rows = request.selected_positions.empty()
            ? static_cast<std::uint32_t>(request.position) + 1U
            : static_cast<std::uint32_t>(
                  request.selected_positions.size());
        const auto softmax_started = request.host_timing != nullptr
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        softmax(scores, request.heads, attended_rows);
        if (request.host_timing != nullptr) {
            request.host_timing->host_softmax_nanoseconds +=
                elapsed_nanoseconds(softmax_started);
        }
        return backend_.glm53_sparse_mla_decode_finish(request, scores);
    }

    [[nodiscard]] bool mla_kv_b_is_bf16(
        std::string_view attention) const noexcept {
        const auto* tensor = checkpoint_.find(
            std::string(attention) + "kv_b_proj.weight");
        return tensor != nullptr &&
               tensor->source_dtype == SafetensorsDtype::Bf16;
    }

    [[nodiscard]] ValidationResult prepare_mla_history(
        std::size_t slot, std::string_view attention,
        CudaGlm53MlaRequest request, std::uint32_t history) {
        if (slot >= states_.size() || request.state == nullptr) {
            return {{"GLM-5.3 MLA history targets an invalid cache slot"}};
        }
        auto& state = *states_[slot];
        std::scoped_lock lock(state.mutex);
        const auto key = std::string(attention) + "kv_b_proj";
        const auto found = state.entries.find(key);
        if (found == state.entries.end() ||
            found->second.weight.device() != request.state->device()) {
            return {{"GLM-5.3 MLA KV-B projection was not admitted: " + key}};
        }
        ++found->second.leases;
        struct Lease {
            Entry& entry;
            ~Lease() { --entry.leases; }
        } lease{found->second};
        request.key_value_b = &found->second.weight;
        return backend_.glm53_mla_prepare_history(request, history);
    }

    [[nodiscard]] ValidationResult swiglu_mhc(
        std::size_t slot, std::string_view prefix,
        std::uint32_t intermediate) {
        if (slot >= states_.size()) {
            return {{"GLM-5.3 resident SwiGLU targets an invalid cache slot"}};
        }
        auto& state = *states_[slot];
        std::scoped_lock lock(state.mutex);
        const std::array<std::string, 3U> keys{
            std::string(prefix) + "gate_proj",
            std::string(prefix) + "up_proj",
            std::string(prefix) + "down_proj"};
        std::array<Entry*, 3U> entries{};
        for (std::size_t index = 0U; index < keys.size(); ++index) {
            const auto found = state.entries.find(keys[index]);
            if (found == state.entries.end() ||
                found->second.weight.device() != devices_[slot]) {
                return {{"GLM-5.3 resident SwiGLU projection was not "
                         "admitted: " + keys[index]}};
            }
            entries[index] = &found->second;
        }
        for (auto* entry : entries) ++entry->leases;
        struct Lease {
            std::array<Entry*, 3U>& entries;
            ~Lease() {
                for (auto* entry : entries) --entry->leases;
            }
        } lease{entries};
        return backend_.glm53_mhc_swiglu(
            devices_[slot], entries[0]->weight, entries[1]->weight,
            entries[2]->weight, intermediate);
    }

    [[nodiscard]] ValidationResult moe(
        std::size_t slot, std::string_view prefix,
        std::span<const KimiRoutedExpert> routed,
        std::span<const float> input, std::span<float> output,
        bool mhc_source_destination = false) {
        ValidationResult result;
        if (slot >= states_.size() || routed.size() != 8U ||
            (mhc_source_destination
                 ? (!input.empty() || !output.empty())
                 : (input.size() != kHidden || output.size() != kHidden))) {
            result.errors.emplace_back("GLM-5.3 MoE command has an invalid shape");
            return result;
        }
        struct Projection {
            std::string key;
            std::uint64_t rows{};
            std::uint64_t columns{};
        };
        const auto make_modules = [](const std::string& base) {
            return std::array<Projection, 3U>{
                Projection{base + "gate_proj", 2048U, kHidden},
                Projection{base + "up_proj", 2048U, kHidden},
                Projection{base + "down_proj", kHidden, 2048U}};
        };
        struct DeviceGroup {
            std::vector<std::size_t> routes;
            bool has_shared{};
            bool enqueued{};
            std::vector<std::string> leased;
            std::vector<CudaMoeExpert> descriptors;
            CudaMoeExpert shared_descriptor;
            std::vector<float> routed_output;
            std::vector<float> shared_output;
        };
        std::vector<DeviceGroup> groups(states_.size());
        groups[slot].has_shared = true;

        // A best-rank peer fabric makes expert parallelism profitable: split
        // the eight independent routes capacity-proportionally and join their
        // exact host-visible outputs in original router order. PHB/PCIe keeps
        // every route with the layer owner, avoiding duplicate cache traffic.
        if (!mhc_source_destination && devices_.size() == 2U &&
            full_tensor_parallel_enabled() &&
            cross_gpu_projections_enabled(devices_)) {
            std::vector<std::uint64_t> capacities;
            capacities.reserve(states_.size());
            for (const auto& state : states_) {
                capacities.push_back(state->capacity);
            }
            const auto ranges = weighted_row_ranges(
                routed.size(), capacities, 1U);
            if (ranges.size() != groups.size()) {
                return {{"GLM-5.3 expert-parallel assignment is invalid"}};
            }
            for (std::size_t group_slot = 0U; group_slot < ranges.size();
                 ++group_slot) {
                for (std::uint64_t route = ranges[group_slot].begin;
                     route < ranges[group_slot].begin +
                                 ranges[group_slot].count;
                     ++route) {
                    groups[group_slot].routes.push_back(
                        static_cast<std::size_t>(route));
                }
            }
        } else {
            for (std::size_t route_index = 0U; route_index < routed.size();
                 ++route_index) {
                groups[slot].routes.push_back(route_index);
            }
        }

        const auto release = [&](std::size_t group_slot) {
            auto& state = *states_[group_slot];
            std::scoped_lock lock(state.mutex);
            for (const auto& key : groups[group_slot].leased) {
                const auto found = state.entries.find(key);
                if (found != state.entries.end() && found->second.leases != 0U) {
                    --found->second.leases;
                }
            }
        };
        const auto ensure = [&](State& state, std::size_t target_slot,
                                const Projection& projection,
                                std::vector<std::string>& leased)
            -> ValidationResult {
            auto found = state.entries.find(projection.key);
            if (found == state.entries.end()) {
                const auto bytes =
                    checkpoint_.cuda_linear_storage_bytes(projection.key);
                if (bytes == 0U || bytes > state.capacity) {
                    return {{"GLM-5.3 MoE projection is absent or exceeds its "
                             "CUDA cache: " + projection.key}};
                }
                while (state.used + bytes > state.capacity) {
                    auto victim = state.recency.end();
                    for (auto candidate = state.recency.begin();
                         candidate != state.recency.end(); ++candidate) {
                        const auto entry = state.entries.find(*candidate);
                        if (entry != state.entries.end() &&
                            !entry->second.pinned && entry->second.leases == 0U) {
                            victim = candidate;
                            break;
                        }
                    }
                    if (victim == state.recency.end()) {
                        return {{"GLM-5.3 exact MoE expert set exceeds the "
                                 "available CUDA cache"}};
                    }
                    auto entry = state.entries.find(*victim);
                    state.used -= entry->second.weight.device_bytes();
                    state.entries.erase(entry);
                    state.recency.erase(victim);
                    ++state.evictions;
                }
                Entry entry;
                const auto load = [&] {
                    return checkpoint_.load_cuda_linear(
                        projection.key, projection.rows, projection.columns,
                        devices_[target_slot], backend_, entry.weight);
                };
                auto loaded = load();
                while (!loaded.ok() && arena_exhausted(loaded) &&
                       evict_one(state)) {
                    entry.weight = CudaWeight{};
                    loaded = load();
                }
                if (!loaded.ok()) return loaded;
                const auto actual = entry.weight.device_bytes();
                if (actual > state.capacity - state.used) {
                    return {{"GLM-5.3 MoE projection exceeded its admitted "
                             "CUDA cache"}};
                }
                state.recency.push_back(projection.key);
                entry.recency = std::prev(state.recency.end());
                state.used += actual;
                found = state.entries.emplace(projection.key,
                                              std::move(entry)).first;
                ++state.misses;
            } else {
                ++state.hits;
                if (found->second.prefetched) {
                    found->second.prefetched = false;
                    ++state.useful_prefetches;
                }
                if (!found->second.pinned) {
                    state.recency.splice(state.recency.end(), state.recency,
                                         found->second.recency);
                }
            }
            ++found->second.leases;
            leased.push_back(projection.key);
            return {};
        };

        std::vector<float> routed_output(routed.size() * kHidden);
        std::vector<float> shared_output(kHidden);
        for (std::size_t group_slot = 0U; group_slot < groups.size();
             ++group_slot) {
            auto& group = groups[group_slot];
            if (group.routes.empty() && !group.has_shared) continue;
            auto& state = *states_[group_slot];
            std::scoped_lock lock(state.mutex);
            std::vector<std::array<Projection, 3U>> modules;
            modules.reserve(group.routes.size());
            for (const auto route_index : group.routes) {
                modules.push_back(make_modules(
                    std::string(prefix) + "experts." +
                    std::to_string(routed[route_index].expert) + "."));
            }
            const auto shared_modules = make_modules(
                std::string(prefix) + "shared_experts.");
            group.leased.reserve(
                (modules.size() + (group.has_shared ? 1U : 0U)) * 3U);
            for (const auto& expert : modules) {
                for (const auto& projection : expert) {
                    auto loaded = ensure(state, group_slot, projection,
                                         group.leased);
                    if (!loaded.ok()) {
                        append(result.errors, std::move(loaded.errors));
                        break;
                    }
                }
                if (!result.ok()) break;
            }
            if (result.ok() && group.has_shared) {
                for (const auto& projection : shared_modules) {
                    auto loaded = ensure(state, group_slot, projection,
                                         group.leased);
                    if (!loaded.ok()) {
                        append(result.errors, std::move(loaded.errors));
                        break;
                    }
                }
            }
            if (!result.ok()) break;
            group.descriptors.resize(modules.size());
            for (std::size_t index = 0U; index < modules.size(); ++index) {
                const auto& expert = modules[index];
                group.descriptors[index] = {
                    &state.entries.at(expert[0].key).weight,
                    &state.entries.at(expert[1].key).weight,
                    &state.entries.at(expert[2].key).weight, 1.0F};
            }
            if (group.has_shared) {
                group.shared_descriptor = {
                    &state.entries.at(shared_modules[0].key).weight,
                    &state.entries.at(shared_modules[1].key).weight,
                    &state.entries.at(shared_modules[2].key).weight, 1.0F};
                group.shared_output.resize(kHidden);
            }
            // The whole routed-plus-shared set was admitted with deferred
            // copies. Order it once instead of synchronizing all 27 projection
            // uploads independently.
            auto ordered = backend_.synchronize_uploads(devices_[group_slot]);
            if (!ordered.ok()) {
                append(result.errors, std::move(ordered.errors));
                break;
            }
            group.routed_output.resize(group.routes.size() * kHidden);
            ValidationResult enqueued;
            if (mhc_source_destination) {
                std::vector<float> coefficients;
                coefficients.reserve(group.routes.size());
                for (const auto route : group.routes) {
                    coefficients.push_back(routed[route].weight);
                }
                enqueued = backend_.enqueue_glm53_moe_from_mhc(
                    devices_[group_slot], group.descriptors,
                    group.shared_descriptor, coefficients, 10.0F);
            } else {
                enqueued = backend_.enqueue_moe(
                    devices_[group_slot], input, 1U, group.descriptors,
                    group.has_shared ? &group.shared_descriptor : nullptr,
                    10.0F);
            }
            if (!enqueued.ok()) {
                append(result.errors, std::move(enqueued.errors));
                break;
            }
            group.enqueued = true;
        }

        // Every active device has been enqueued before the first completion
        // boundary, so their expert projections and transfers overlap.
        for (std::size_t group_slot = 0U; group_slot < groups.size();
             ++group_slot) {
            auto& group = groups[group_slot];
            if (group.enqueued) {
                auto collected = mhc_source_destination
                    ? backend_.finish_deepseek_moe_chain(devices_[group_slot])
                    : backend_.collect_moe(
                          devices_[group_slot], group.routed_output,
                          group.has_shared
                              ? std::span<float>(group.shared_output)
                              : std::span<float>{});
                if (!collected.ok()) {
                    append(result.errors, std::move(collected.errors));
                } else if (!mhc_source_destination) {
                    for (std::size_t local = 0U; local < group.routes.size();
                         ++local) {
                        std::copy_n(
                            group.routed_output.begin() +
                                static_cast<std::ptrdiff_t>(local * kHidden),
                            kHidden,
                            routed_output.begin() + static_cast<std::ptrdiff_t>(
                                group.routes[local] * kHidden));
                    }
                    if (group.has_shared) {
                        std::copy(group.shared_output.begin(),
                                  group.shared_output.end(),
                                  shared_output.begin());
                    }
                }
            }
            if (!group.leased.empty()) release(group_slot);
        }
        if (!result.ok()) return result;
        if (mhc_source_destination) return result;
        std::copy(shared_output.begin(), shared_output.end(), output.begin());
        for (std::size_t expert = 0U; expert < routed.size(); ++expert) {
            const auto begin = expert * kHidden;
            for (std::size_t column = 0U; column < kHidden; ++column) {
                output[column] = bf16_round_f32(
                    output[column] + bf16_round_f32(
                        routed[expert].weight *
                        routed_output[begin + column]));
            }
        }
        return result;
    }

    [[nodiscard]] Stats stats() const {
        Stats result;
        for (const auto& state_ptr : states_) {
            auto& state = *state_ptr;
            std::scoped_lock lock(state.mutex);
            result.capacity.push_back(state.capacity);
            result.used.push_back(state.used);
            result.pinned.push_back(state.pinned);
            result.hits += state.hits;
            result.misses += state.misses;
            result.evictions += state.evictions;
            result.prefetches += state.prefetches;
            result.useful_prefetches += state.useful_prefetches;
            result.failed_prefetches += state.failed_prefetches;
        }
        return result;
    }

private:
    [[nodiscard]] static bool arena_exhausted(
        const ValidationResult& result) noexcept {
        return std::any_of(
            result.errors.begin(), result.errors.end(),
            [](const std::string& error) {
                return error.starts_with("CUDA weight arena is exhausted");
            });
    }

    [[nodiscard]] static bool evict_one(State& state) {
        for (auto candidate = state.recency.begin();
             candidate != state.recency.end(); ++candidate) {
            auto found = state.entries.find(*candidate);
            if (found == state.entries.end() || found->second.pinned ||
                found->second.leases != 0U) {
                continue;
            }
            state.used -= found->second.weight.device_bytes();
            state.entries.erase(found);
            state.recency.erase(candidate);
            ++state.evictions;
            return true;
        }
        return false;
    }

    [[nodiscard]] static std::string slice_key(
        std::string_view base, std::uint64_t row_begin,
        std::uint64_t row_count) {
        return std::string(base) + "#rows=" + std::to_string(row_begin) + "+" +
               std::to_string(row_count);
    }

    Glm53CheckpointReader& checkpoint_;
    CudaBackend& backend_;
    std::vector<int> devices_;
    std::vector<std::unique_ptr<State>> states_;
};

}  // namespace

struct Glm53Runtime::Impl {
    struct PrefetchJob {
        ExpertKey key;
        std::size_t slot{};
    };

    struct DeviceSequenceState {
        struct SparseMlaArena {
            std::vector<std::uint32_t> slot_pools;
            std::vector<std::uint64_t> slot_recency;
            std::uint64_t clock{};
            std::uint32_t identity_rows{};
            bool seeded{};
            bool seed_validation_pending{};
        };
        std::array<CudaBuffer, kLayers> kda;
        std::array<CudaBuffer, kLayers> mla;
        std::array<CudaBuffer, kLayers> sparse_mla_expanded;
        std::array<SparseMlaArena, kLayers> sparse_mla_arenas;
        bool ready{};
    };

    struct ResidentLayerWeights {
        CudaDsv4MhcWeights attention;
        CudaDsv4MhcWeights feedforward;
    };

    struct PrefixEntry {
        // This cache is owned by one immutable runtime instance: checkpoint,
        // tokenizer, adapter set, state layout and numerical-mode controls are
        // fixed before the first entry exists. Tokens are therefore the only
        // varying component of the full prefix key; entries can never cross a
        // model or configuration boundary.
        std::vector<std::uint32_t> tokens;
        Glm53SequenceState state;
        std::vector<float> logits;
        std::vector<float> base_hidden;
        std::uint64_t recency{};
    };

    struct ProfileSnapshot {
        CudaBackendStats cuda;
        Glm53CacheMetrics cache;
        Glm53HostExpertMetrics host_experts;
        Glm53GraphMetrics graph;
        Glm53SparseMlaMetrics sparse_mla;
    };

    struct ScheduledRequest {
        std::vector<std::uint32_t> prompt;
        std::uint32_t maximum_new_tokens{};
        SamplingOptions sampling;
        std::vector<std::string> stop;
        TokenStreamCallback on_token;
        Glm53GenerationResult result;
        Glm53SequenceState sequence;
        DeviceSequenceState device_sequence;
        std::vector<float> logits;
        std::vector<float> base_hidden;
        std::vector<std::uint32_t> counts;
        std::vector<std::uint32_t> sampled;
        Glm53WeightCache::Stats decode_cache_start;
        std::uint64_t decode_static_hits_start{};
        std::uint64_t decode_static_misses_start{};
        std::mt19937_64 generator;
        std::unique_ptr<StopSequenceBuffer> streamed;
        std::size_t prefill_cursor{};
        double prefill_started{};
        ProfileSnapshot profile_started;
        ProfileSnapshot profile_after_prefill;
        ProfileSnapshot profile_decode_started;
        std::uint32_t position{};
        std::uint32_t iteration{};
        double decode_started{};
        std::mutex completion_mutex;
        std::condition_variable completion;
        bool prepared{};
        bool decoding{};
        bool mtp_ready{};
        bool done{};
    };

    Glm53RuntimeConfig config;
    std::unique_ptr<Glm53CheckpointReader> checkpoint;
    ModelTokenizer tokenizer;
    CudaBackend cuda;
    std::vector<int> devices;
    std::vector<std::size_t> device_schedule;
    std::vector<std::uint64_t> device_budgets;
    std::vector<std::uint64_t> weight_capacities;
    std::vector<std::uint64_t> resident_reserve_bytes;
    std::vector<Glm53RowRange> lm_head_ranges;
    std::unique_ptr<Glm53WeightCache> weights;
    std::array<ResidentLayerWeights, kLayers> resident_layers;
    bool resident_execution_active{};
    RoutePredictor route_predictor;
    std::mutex prefetch_mutex;
    std::condition_variable prefetch_ready;
    std::deque<PrefetchJob> prefetch_queue;
    std::unordered_set<ExpertKey, ExpertKeyHash> pending_prefetch;
    std::vector<std::thread> prefetch_threads;
    std::size_t prefetch_queue_limit{};
    std::size_t prefetch_prediction_limit{};
    double prefetch_minimum_confidence{1.0};
    bool prefetch_stopping{};
    std::atomic<std::uint64_t> prefetch_requests{};
    std::atomic<std::uint64_t> prefetch_completed{};
    std::atomic<std::uint64_t> prefetch_dropped{};
    std::atomic<std::uint64_t> prefetch_errors{};
    std::unique_ptr<HostWorkerPool> projection_workers;
    std::unique_ptr<HostWorkerPool> host_moe_workers;
    // Host-MoE scratch, reused across calls instead of reallocated per call.
    //
    // These are Impl members rather than function-local `thread_local`: the
    // buffers are filled by the worker pool, and a thread_local resolves to the
    // *worker's* copy inside the dispatch lambda, not the caller's, so every
    // worker indexed into an empty vector. They are plain members for the same
    // reason `host_moe_workers` is: one runtime drives one MoE call at a time.
    //
    // Every element is overwritten each call -- gate/up writes every activation
    // slot and down every output slot -- so reuse without clearing is exact,
    // and the byte-identical output gate is what verifies that.
    std::vector<float> host_moe_quantized_input;
    std::vector<float> host_moe_activations;
    std::vector<float> host_moe_expert_outputs;
    // All 42 shared experts, resident on the GPU that owns their layer.
    //
    // Every MoE layer runs nine experts and the router chooses only eight of
    // them: the ninth, the shared expert, is known before routing and never
    // competes for the routed cache. Moving it across costs 25.17 MB per FP8
    // layer or 50.33 MB per BF16 layer -- 1.06 or 2.11 GB for the complete
    // tier -- and the device dot associates its sum exactly as the matching
    // host AVX2 dot does, so the output is unchanged.
    struct SharedExpertTier {
        // Six slots per admitted expert. FP8 uses weight/scale pairs; BF16
        // uses the three weight slots and leaves each scale slot invalid. They
        // own the device memory the descriptors point into.
        std::vector<CudaBuffer> storage;
        std::array<CudaGlm53Expert, kLayers + 1U> experts{};
        // The device holding each layer's tier, or -1 where it was not
        // admitted. The MTP layer keeps -1: it is opt-in and its expert runs
        // on the host path.
        std::array<int, kLayers + 1U> devices{};
        std::vector<std::uint64_t> bytes_by_slot;
        std::uint64_t bytes{};
        bool active{};
    };
    SharedExpertTier shared_experts;
    struct StaticExpertTier {
        std::array<std::array<CudaGlm53Expert, 288U>, kLayers> experts{};
        std::array<std::array<std::uint8_t, 288U>, kLayers> active{};
        std::vector<std::uint64_t> bytes_by_slot;
        std::uint64_t bytes{};
        std::uint64_t experts_admitted{};
        std::atomic<std::uint64_t> route_hits{};
        std::atomic<std::uint64_t> route_misses{};
        bool active_tier{};
    };
    StaticExpertTier static_experts;
    // One bit per layer, set when a MoE layer serviced its shared expert on
    // the host. With the tier admitted this should stay zero; anything else
    // names exactly which layers the dispatch missed.
    // One row per routed MoE layer visit: the eight experts the router chose,
    // in order, with enough context to separate phases and tokens. 42 layers
    // times 8 experts is 336 uint16 per decode token, so a 128-token census is
    // 86 KB -- small enough to hold and write whole rather than stream.
    struct RouteCensusRow {
        std::uint32_t layer{};
        std::uint32_t position{};
        std::uint64_t request{};
        bool prefill{};
        std::array<std::uint16_t, 8U> experts{};
    };
    std::mutex route_census_mutex;
    std::vector<RouteCensusRow> route_census;
    std::atomic<std::uint64_t> shared_expert_host_layers{};
    std::atomic<std::uint64_t> shared_expert_host_calls{};
    std::atomic<std::uint64_t> shared_expert_device_calls{};
    // Resident MLA softmax scratch, `kHeads * history` floats. Grown with the
    // context and reused, so no allocation happens inside a timed step.
    std::vector<float> mla_softmax_scores;
    std::vector<float> sparse_index_input;
    std::vector<std::uint32_t> mla_selected_positions;
    std::vector<std::uint32_t> mla_arena_rows;
    std::vector<std::uint32_t> mla_expansion_source_positions;
    std::vector<std::uint32_t> mla_expansion_destination_rows;
    // Expansion scratch for the host MLA paths. These are the largest buffers
    // in the model's steady state -- `attended_rows x 32,768 x 4 B`, 268 MiB at
    // a saturated selection -- and allocating them per layer per token made
    // decode fault in and zero about 1.5 GB of fresh anonymous pages every
    // token, then take the D2H into pageable memory. Measured at 0.86 s per GB,
    // roughly ten times what the link costs. Decode, dense prefill and sparse
    // prefill never run concurrently, so one set serves all three.
    std::vector<float> mla_expanded_scratch;
    std::vector<float> mla_gathered_scratch;
    std::vector<float> mla_head_score_scratch;
    std::vector<float> shared_expert_gate;
    std::vector<float> shared_expert_up;
    std::vector<float> shared_expert_output;
    struct Glm53ExpertViews {
        Glm53HostExpertLinear gate;
        Glm53HostExpertLinear up;
        Glm53HostExpertLinear down;
    };
    // Paged-primitive scratch. Group metadata and the flat assignment table
    // retain their capacity between layers, so expert-major dispatch performs
    // no allocation after admission.
    struct PageAssignment {
        std::size_t input_row{};
        std::size_t output_slot{};
    };
    struct PageGroup {
        std::uint32_t expert{};
        bool shared{};
        Glm53ExpertViews module;
        std::size_t assignment_begin{};
        std::size_t assignment_count{};
    };
    std::vector<PageGroup> page_groups;
    std::vector<PageAssignment> page_assignments;
    std::vector<CudaGlm53Expert> page_device_experts;
    std::vector<std::size_t> page_device_output_slots;
    std::vector<float> page_device_inputs;
    std::vector<float> page_quantized_input;
    std::vector<float> page_activations;
    std::vector<float> page_expert_outputs;
    // Process-lifetime cache of resolved expert weight spans, indexed by
    // (layer, expert slot) with slot 288 the shared expert.
    //
    // `Glm53CheckpointReader::view` computes a span into the mapping and
    // touches no pages -- `read` is the one that moves bytes -- so caching a
    // view changes nothing about what is read. What it removes is per-call
    // work that M0 counted every step: three module-name strings built and
    // three manifest lookups per expert, 27 per MoE call and 5,376 calls per
    // 128 decode tokens.
    //
    // Guarded rather than lock-free: the MoE call site is the scheduler thread
    // today, but nothing in the type system says so, and the lock is taken once
    // per call around the whole group rather than once per expert.
    static constexpr std::uint32_t kExpertSlots = 289U;  // 288 routed + shared
    mutable std::mutex expert_view_mutex;
    mutable std::vector<Glm53ExpertViews> expert_view_cache;
    mutable std::vector<std::uint8_t> expert_view_ready;
    bool host_moe_active{};
    std::atomic<std::uint64_t> host_moe_calls{};
    std::atomic<std::uint64_t> host_moe_rows{};
    std::atomic<std::uint64_t> host_moe_nanoseconds{};
    std::atomic<std::uint64_t> host_moe_gate_up_weight_bytes{};
    std::atomic<std::uint64_t> host_moe_down_weight_bytes{};
    std::atomic<std::uint64_t> host_moe_view_nanoseconds{};
    std::atomic<std::uint64_t> host_moe_input_quantization_nanoseconds{};
    std::atomic<std::uint64_t> host_moe_gate_up_nanoseconds{};
    std::atomic<std::uint64_t> host_moe_activation_nanoseconds{};
    std::atomic<std::uint64_t> host_moe_down_nanoseconds{};
    std::atomic<std::uint64_t> host_moe_reduction_nanoseconds{};
    std::atomic<std::uint64_t> host_moe_temporary_allocation_calls{};
    std::atomic<std::uint64_t> graph_forward_calls{};
    std::atomic<std::uint64_t> graph_forward_rows{};
    std::atomic<std::uint64_t> graph_embedding_nanoseconds{};
    std::atomic<std::uint64_t> graph_layer_nanoseconds{};
    std::atomic<std::uint64_t> graph_attention_block_nanoseconds{};
    std::atomic<std::uint64_t> graph_kda_nanoseconds{};
    std::atomic<std::uint64_t> graph_mla_nanoseconds{};
    std::atomic<std::uint64_t> graph_feedforward_block_nanoseconds{};
    std::atomic<std::uint64_t> graph_output_head_nanoseconds{};
    std::atomic<std::uint64_t> graph_sampling_nanoseconds{};
    std::atomic<std::uint64_t> sparse_mla_calls{};
    std::atomic<std::uint64_t> sparse_mla_input_download_nanoseconds{};
    std::atomic<std::uint64_t> sparse_mla_indexer_projection_nanoseconds{};
    std::atomic<std::uint64_t> sparse_mla_indexer_state_nanoseconds{};
    std::atomic<std::uint64_t> sparse_mla_query_rank_projection_nanoseconds{};
    std::atomic<std::uint64_t> sparse_mla_pool_scoring_nanoseconds{};
    std::atomic<std::uint64_t> sparse_mla_topk_sort_nanoseconds{};
    std::atomic<std::uint64_t> sparse_mla_arena_bookkeeping_nanoseconds{};
    std::atomic<std::uint64_t> sparse_mla_index_upload_nanoseconds{};
    std::atomic<std::uint64_t> sparse_mla_device_scores_wait_nanoseconds{};
    std::atomic<std::uint64_t> sparse_mla_host_softmax_nanoseconds{};
    std::atomic<std::uint64_t> sparse_mla_coefficient_upload_nanoseconds{};
    bool full_tensor_parallel_active{};
    std::unique_ptr<HostWorkerPool> kda_workers;
    std::atomic<std::uint64_t> parallel_projection_batches{};
    std::atomic<std::uint64_t> parallel_projection_requests{};
    std::atomic<std::uint64_t> tensor_parallel_head_batches{};
    std::atomic<std::uint64_t> parallel_encode_pages{};
    std::atomic<std::uint64_t> prefix_cache_hits{};
    std::atomic<std::uint64_t> prefix_cache_tokens{};
    std::atomic<std::uint64_t> prefix_cache_bytes{};
    std::atomic<std::uint64_t> prefix_cache_entries{};
    std::atomic<std::uint64_t> prefix_cache_evictions{};
    std::atomic<std::uint64_t> prefix_cache_evicted_bytes{};
    std::mutex prefix_mutex;
    std::vector<PrefixEntry> prefix_cache;
    std::size_t prefix_cache_limit{1U};
    std::uint64_t prefix_clock{};
    std::mutex host_tensor_mutex;
    std::unordered_map<std::string,
                       std::shared_ptr<const std::vector<float>>> host_tensors;
    ValidationResult warmup_result;
    bool ready{};
    std::thread warmup_thread;
    std::mutex warmup_mutex;
    std::mutex scheduler_mutex;
    std::condition_variable scheduler_ready;
    std::deque<std::shared_ptr<ScheduledRequest>> pending_requests;
    std::vector<std::shared_ptr<ScheduledRequest>> active_requests;
    std::thread scheduler_thread;
    std::size_t scheduler_capacity{1U};
    std::size_t host_state_capacity{1U};
    std::vector<std::uint64_t> sequence_device_bytes;
    std::vector<std::uint64_t> sequence_device_free_bytes;
    std::vector<std::uint64_t> sequence_device_safety_bytes;
    std::vector<std::uint64_t> sequence_device_fragmented_bytes;
    std::vector<std::size_t> sequence_device_capacities;
    std::uint64_t host_state_fragmented_bytes{};
    std::atomic<std::uint64_t> sequence_slots_live{};
    std::atomic<std::uint64_t> sequence_slots_peak{};
    bool scheduler_stopping{};
    std::atomic<std::uint64_t> scheduler_iterations{};
    std::atomic<std::uint64_t> scheduler_batched_iterations{};
    std::atomic<std::uint64_t> scheduler_single_forward_nanoseconds{};
    std::atomic<std::uint64_t> scheduler_single_forward_tokens{};
    std::atomic<std::uint64_t> scheduler_batch_forward_nanoseconds{};
    std::atomic<std::uint64_t> scheduler_batch_forward_tokens{};
    std::array<std::atomic<std::uint64_t>, 33U>
        scheduler_width_iterations{};
    std::array<std::atomic<std::uint64_t>, 33U>
        scheduler_width_forward_nanoseconds{};
    std::array<std::atomic<std::uint64_t>, 33U>
        scheduler_width_tokens{};
    std::atomic<std::uint64_t> mtp_drafts{};
    std::atomic<std::uint64_t> mtp_accepted{};
    std::atomic<bool> profiler_captured{};

    [[nodiscard]] Glm53CacheMetrics cache_metrics() const {
        if (weights == nullptr) return {};
        const auto stats = weights->stats();
        return {stats.hits, stats.misses, stats.evictions, stats.prefetches,
                stats.useful_prefetches, stats.failed_prefetches};
    }

    [[nodiscard]] Glm53HostExpertMetrics host_expert_metrics() const noexcept {
        return {
            host_moe_calls.load(std::memory_order_relaxed),
            host_moe_rows.load(std::memory_order_relaxed),
            host_moe_gate_up_weight_bytes.load(std::memory_order_relaxed),
            host_moe_down_weight_bytes.load(std::memory_order_relaxed),
            host_moe_view_nanoseconds.load(std::memory_order_relaxed),
            host_moe_input_quantization_nanoseconds.load(
                std::memory_order_relaxed),
            host_moe_gate_up_nanoseconds.load(std::memory_order_relaxed),
            host_moe_activation_nanoseconds.load(std::memory_order_relaxed),
            host_moe_down_nanoseconds.load(std::memory_order_relaxed),
            host_moe_reduction_nanoseconds.load(std::memory_order_relaxed),
            host_moe_nanoseconds.load(std::memory_order_relaxed),
            host_moe_temporary_allocation_calls.load(
                std::memory_order_relaxed)};
    }

    [[nodiscard]] Glm53GraphMetrics graph_metrics() const noexcept {
        return {
            graph_forward_calls.load(std::memory_order_relaxed),
            graph_forward_rows.load(std::memory_order_relaxed),
            graph_embedding_nanoseconds.load(std::memory_order_relaxed),
            graph_layer_nanoseconds.load(std::memory_order_relaxed),
            graph_attention_block_nanoseconds.load(std::memory_order_relaxed),
            graph_kda_nanoseconds.load(std::memory_order_relaxed),
            graph_mla_nanoseconds.load(std::memory_order_relaxed),
            graph_feedforward_block_nanoseconds.load(
                std::memory_order_relaxed),
            graph_output_head_nanoseconds.load(std::memory_order_relaxed),
            graph_sampling_nanoseconds.load(std::memory_order_relaxed)};
    }

    [[nodiscard]] Glm53SparseMlaMetrics sparse_mla_metrics() const noexcept {
        return {
            sparse_mla_calls.load(std::memory_order_relaxed),
            sparse_mla_input_download_nanoseconds.load(
                std::memory_order_relaxed),
            sparse_mla_indexer_projection_nanoseconds.load(
                std::memory_order_relaxed),
            sparse_mla_indexer_state_nanoseconds.load(
                std::memory_order_relaxed),
            sparse_mla_query_rank_projection_nanoseconds.load(
                std::memory_order_relaxed),
            sparse_mla_pool_scoring_nanoseconds.load(
                std::memory_order_relaxed),
            sparse_mla_topk_sort_nanoseconds.load(
                std::memory_order_relaxed),
            sparse_mla_arena_bookkeeping_nanoseconds.load(
                std::memory_order_relaxed),
            sparse_mla_index_upload_nanoseconds.load(
                std::memory_order_relaxed),
            sparse_mla_device_scores_wait_nanoseconds.load(
                std::memory_order_relaxed),
            sparse_mla_host_softmax_nanoseconds.load(
                std::memory_order_relaxed),
            sparse_mla_coefficient_upload_nanoseconds.load(
                std::memory_order_relaxed)};
    }

    void add_sparse_mla_metrics(const Glm53SparseMlaMetrics& timing) noexcept {
        sparse_mla_calls.fetch_add(timing.calls, std::memory_order_relaxed);
        sparse_mla_input_download_nanoseconds.fetch_add(
            timing.input_download_nanoseconds, std::memory_order_relaxed);
        sparse_mla_indexer_projection_nanoseconds.fetch_add(
            timing.indexer_projection_nanoseconds, std::memory_order_relaxed);
        sparse_mla_indexer_state_nanoseconds.fetch_add(
            timing.indexer_state_nanoseconds, std::memory_order_relaxed);
        sparse_mla_query_rank_projection_nanoseconds.fetch_add(
            timing.query_rank_projection_nanoseconds,
            std::memory_order_relaxed);
        sparse_mla_pool_scoring_nanoseconds.fetch_add(
            timing.pool_scoring_nanoseconds, std::memory_order_relaxed);
        sparse_mla_topk_sort_nanoseconds.fetch_add(
            timing.topk_sort_nanoseconds, std::memory_order_relaxed);
        sparse_mla_arena_bookkeeping_nanoseconds.fetch_add(
            timing.arena_bookkeeping_nanoseconds, std::memory_order_relaxed);
        sparse_mla_index_upload_nanoseconds.fetch_add(
            timing.index_upload_nanoseconds, std::memory_order_relaxed);
        sparse_mla_device_scores_wait_nanoseconds.fetch_add(
            timing.device_scores_wait_nanoseconds, std::memory_order_relaxed);
        sparse_mla_host_softmax_nanoseconds.fetch_add(
            timing.host_softmax_nanoseconds, std::memory_order_relaxed);
        sparse_mla_coefficient_upload_nanoseconds.fetch_add(
            timing.coefficient_upload_nanoseconds, std::memory_order_relaxed);
    }

    [[nodiscard]] ProfileSnapshot profile_snapshot() const {
        if (!config.phase_profile) return {};
        return {cuda.stats(), cache_metrics(), host_expert_metrics(),
                graph_metrics(), sparse_mla_metrics()};
    }

    [[nodiscard]] static Glm53PhaseMetrics phase_delta(
        const ProfileSnapshot& after, const ProfileSnapshot& before) {
        return {detail::cuda_delta(after.cuda, before.cuda),
                cache_delta(after.cache, before.cache),
                host_expert_delta(after.host_experts, before.host_experts),
                graph_delta(after.graph, before.graph),
                sparse_mla_delta(after.sparse_mla, before.sparse_mla)};
    }

    ~Impl() {
        {
            std::scoped_lock lock(scheduler_mutex);
            scheduler_stopping = true;
        }
        scheduler_ready.notify_all();
        if (scheduler_thread.joinable()) scheduler_thread.join();
        if (warmup_thread.joinable()) warmup_thread.join();
        {
            std::scoped_lock lock(prefetch_mutex);
            prefetch_stopping = true;
        }
        prefetch_ready.notify_all();
        for (auto& worker : prefetch_threads) {
            if (worker.joinable()) worker.join();
        }
    }

    void prefetch_loop() {
        for (;;) {
            PrefetchJob job;
            {
                std::unique_lock lock(prefetch_mutex);
                prefetch_ready.wait(lock, [&] {
                    return prefetch_stopping || !prefetch_queue.empty();
                });
                if (prefetch_stopping && prefetch_queue.empty()) return;
                job = prefetch_queue.front();
                prefetch_queue.pop_front();
            }
            auto status = weights->prefetch_expert(
                job.slot, job.key.layer, job.key.expert);
            if (status.ok()) {
                prefetch_completed.fetch_add(1U, std::memory_order_relaxed);
            } else if (prefetch_errors.fetch_add(
                           1U, std::memory_order_relaxed) == 0U) {
                std::cerr << "[glm53-residency] first_prefetch_error="
                          << status.errors.front() << '\n';
            }
            {
                std::scoped_lock lock(prefetch_mutex);
                pending_prefetch.erase(job.key);
            }
        }
    }

    void request_prefetch(const RoutePrediction& prediction) {
        // Nsight measured the fused resident chain at 12 useful predictions
        // out of 90, with speculative uploads adding 2.3 GB/token and holding
        // the demand cache mutex for 2.55 seconds.  Resident decode already
        // overlaps a layer's admitted set as one command, so cache pollution
        // is more expensive than the predictor's occasional hit.  Keep the
        // predictor available to the host-bound path where it was originally
        // validated, but never let it contend with the fused demand chain.
        if (resident_execution_active || host_moe_active ||
            prefetch_queue_limit == 0U ||
            prediction.key.layer >= kLayers ||
            !glm53_moe_layer(prediction.key.layer)) {
            return;
        }
        const auto slot = slot_for(prediction.key.layer);
        if (weights->contains_expert(slot, prediction.key.layer,
                                     prediction.key.expert)) {
            return;
        }
        prefetch_requests.fetch_add(1U, std::memory_order_relaxed);
        std::scoped_lock lock(prefetch_mutex);
        if (prefetch_stopping || pending_prefetch.contains(prediction.key)) {
            return;
        }
        if (prefetch_queue.size() >= prefetch_queue_limit) {
            prefetch_dropped.fetch_add(1U, std::memory_order_relaxed);
            return;
        }
        pending_prefetch.insert(prediction.key);
        prefetch_queue.push_back({prediction.key, slot});
        prefetch_ready.notify_one();
    }

    [[nodiscard]] static std::uint64_t route_request_key(
        const Glm53SequenceState* sequence, std::uint32_t position) noexcept {
        // Each logical token owns one transition chain. Prompt execution is
        // layer-major, so using only the sequence address would connect rows
        // in execution order instead of connecting adjacent layers.
        auto value = static_cast<std::uint64_t>(
            reinterpret_cast<std::uintptr_t>(sequence));
        value ^= static_cast<std::uint64_t>(position) +
                 0x9e3779b97f4a7c15ULL + (value << 6U) + (value >> 2U);
        return value == 0U ? 1U : value;
    }

    void observe_route(std::uint32_t layer,
                       std::span<const KimiRoutedExpert> selected,
                       std::uint64_t request, std::uint32_t position,
                       bool schedule_prefetch, bool prefill = false) {
        // Recorded before the prefetch guard: the census has to see every
        // routed layer visit, and prefetch is a separate opt-in that is off
        // whenever the host expert path owns the experts.
        if (!route_census_path().empty() && layer < kLayers &&
            selected.size() == 8U) {
            RouteCensusRow row;
            row.layer = layer;
            row.position = position;
            row.request = request;
            row.prefill = prefill;
            for (std::size_t index = 0U; index < selected.size(); ++index) {
                row.experts[index] =
                    static_cast<std::uint16_t>(selected[index].expert);
            }
            std::scoped_lock guard(route_census_mutex);
            route_census.push_back(row);
        }
        if (prefetch_prediction_limit == 0U || request == 0U ||
            layer >= kLayers) {
            return;
        }
        RouteEvent event;
        event.request = request;
        event.token_position = position;
        event.layer = layer;
        event.phase = RoutePhase::Decode;
        event.experts.reserve(selected.size());
        event.coefficients.reserve(selected.size());
        for (const auto& route : selected) {
            event.experts.push_back(route.expert);
            event.coefficients.push_back(route.weight);
        }
        route_predictor.observe(event);
        if (!schedule_prefetch) return;
        for (const auto& prediction : route_predictor.predict(
                 event, prefetch_prediction_limit,
                 prefetch_minimum_confidence)) {
            // One-layer lookahead is the useful overlap window: farther
            // predictions consume cache before their demand and are more
            // likely to evict a nearer expert.
            if (prediction.key.layer == layer + 1U) {
                request_prefetch(prediction);
            }
        }
    }

    [[nodiscard]] std::size_t slot_for(std::uint32_t layer) const noexcept {
        const auto target_layer = std::min(layer, kLayers - 1U);
        return device_schedule[target_layer % device_schedule.size()];
    }

    [[nodiscard]] int device_for(std::uint32_t layer) const noexcept {
        return devices[slot_for(layer)];
    }

    [[nodiscard]] std::vector<std::uint64_t>
    expected_sequence_device_bytes() const {
        std::vector<std::uint64_t> bytes(devices.size(), 0U);
        const auto kda_floats =
            static_cast<std::uint64_t>(kHeads) * kLinearHead * kLinearHead +
            3ULL * kLinearWidth * 3ULL +
            3ULL * kLinearWidth * 4ULL + kHeads + kLinearWidth +
            kLinearHead + kKdaWorkspaceFloats;
        const auto mla_floats =
            static_cast<std::uint64_t>(config.maximum_context_tokens) *
                kKvRank +
            kQueryRank + kKvRank;
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            const auto slot = slot_for(layer);
            if (glm53_kda_layer(layer)) {
                bytes[slot] += kda_floats * sizeof(float);
                continue;
            }
            // Sparse MLA keeps only the compressed latent history and its two
            // norm vectors resident. It must be charged here exactly as
            // `prepare_device_sequence` allocates it; unlike the dense arm it
            // never reserves the maximum-context expanded KV cache.
            if (sparse_indexer_active(config.maximum_context_tokens)) {
                const auto expanded_mla_bytes =
                    static_cast<std::uint64_t>(kIndexArenaRows) *
                    kHeads * 2ULL * kMlaHead * sizeof(std::uint16_t);
                bytes[slot] += mla_floats * sizeof(float) +
                               expanded_mla_bytes;
                continue;
            }
            const auto attention = "model.language_model.layers." +
                std::to_string(layer) + ".self_attn.";
            // Both current releases store MLA KV-B in BF16. Key this ledger
            // from the actual tensor dtype, not the checkpoint's routed-expert
            // quantization label: otherwise FP8 is undercharged by the full
            // persistent expanded-history allocation and over-admitted.
            const bool persistent_mla_expansion = weights != nullptr &&
                weights->mla_kv_b_is_bf16(attention);
            const auto expanded_mla_bytes = persistent_mla_expansion
                ? static_cast<std::uint64_t>(
                      config.maximum_context_tokens) * kHeads * 2ULL *
                      kMlaHead * sizeof(std::uint16_t)
                : 0U;
            bytes[slot] += mla_floats * sizeof(float) + expanded_mla_bytes;
        }
        return bytes;
    }

    [[nodiscard]] std::vector<std::uint64_t> actual_sequence_device_bytes(
        const DeviceSequenceState& sequence) const {
        std::vector<std::uint64_t> bytes(devices.size(), 0U);
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            const auto& buffer = glm53_kda_layer(layer)
                ? sequence.kda[layer] : sequence.mla[layer];
            if (!buffer.valid()) continue;
            const auto found = std::find(devices.begin(), devices.end(),
                                         buffer.device());
            if (found == devices.end()) continue;
            bytes[static_cast<std::size_t>(found - devices.begin())] +=
                buffer.device_bytes();
        }
        for (const auto& buffer : sequence.sparse_mla_expanded) {
            if (!buffer.valid()) continue;
            const auto found = std::find(devices.begin(), devices.end(),
                                         buffer.device());
            if (found == devices.end()) continue;
            bytes[static_cast<std::size_t>(found - devices.begin())] +=
                buffer.device_bytes();
        }
        return bytes;
    }

    [[nodiscard]] ValidationResult configure_sequence_admission() {
        sequence_device_bytes = expected_sequence_device_bytes();
        sequence_device_free_bytes.assign(devices.size(), 0U);
        sequence_device_safety_bytes.assign(devices.size(), 0U);
        sequence_device_fragmented_bytes.assign(devices.size(), 0U);
        sequence_device_capacities.assign(devices.size(), 0U);
        const auto host_state_size =
            host_sequence_state_bytes(config.maximum_context_tokens);
        const auto host_budget =
            host_hardware_profile().host_usable_bytes(0.05);
        host_state_fragmented_bytes = host_budget >
                host_state_capacity * host_state_size
            ? host_budget - host_state_capacity * host_state_size : 0U;
        std::size_t capacity = std::min<std::size_t>(32U,
                                                     host_state_capacity);
        for (std::size_t slot = 0U; slot < devices.size(); ++slot) {
            auto memory = CudaBackend::device_memory(devices[slot]);
            if (!memory.ok()) return {std::move(memory.errors)};
            const auto free = memory.value.free_bytes;
            // Keep five percent of the post-warm free pool outside admission
            // for driver bookkeeping and transient CUDA launch state. This is
            // discovered from the loaded process, not a card-size constant.
            const auto safety = free / 20U;
            const auto usable = free - safety;
            const auto required = sequence_device_bytes[slot];
            const auto slots = required == 0U
                ? std::size_t{32U}
                : static_cast<std::size_t>(usable / required);
            sequence_device_free_bytes[slot] = free;
            sequence_device_safety_bytes[slot] = safety;
            sequence_device_fragmented_bytes[slot] = required == 0U
                ? 0U : usable - static_cast<std::uint64_t>(slots) * required;
            sequence_device_capacities[slot] = slots;
            capacity = std::min(capacity, slots);
        }
        if (capacity == 0U) {
            return {{"GLM-5.3 has insufficient post-warm CUDA capacity for "
                     "one complete sequence state"}};
        }
        // Prefix snapshots own host-only COW state. Reserve one host block for
        // that required M7 capability before publishing live-request
        // capacity; if the host can hold only one state, keep serving exact
        // requests and disable prefix retention instead of overcommitting.
        scheduler_capacity = host_state_capacity > 1U
            ? std::min(capacity, host_state_capacity - 1U) : capacity;
        prefix_cache_limit = host_state_capacity > scheduler_capacity
            ? host_state_capacity - scheduler_capacity : 0U;
        std::cerr << "[glm53-sequence-admission] host_state_bytes="
                  << host_sequence_state_bytes(config.maximum_context_tokens)
                  << " host_capacity=" << host_state_capacity;
        for (std::size_t slot = 0U; slot < devices.size(); ++slot) {
            std::cerr << " cuda" << devices[slot] << "_state_bytes="
                      << sequence_device_bytes[slot]
                      << " cuda" << devices[slot] << "_free_bytes="
                      << sequence_device_free_bytes[slot]
                      << " cuda" << devices[slot] << "_safety_bytes="
                      << sequence_device_safety_bytes[slot]
                      << " cuda" << devices[slot] << "_capacity="
                      << sequence_device_capacities[slot];
        }
        std::cerr << " scheduler_capacity=" << scheduler_capacity
                  << " prefix_capacity=" << prefix_cache_limit << '\n';
        return {};
    }

    [[nodiscard]] ParseResult<std::shared_ptr<const std::vector<float>>>
    host_tensor(std::string_view name, std::uint64_t elements) {
        ParseResult<std::shared_ptr<const std::vector<float>>> result;
        const std::string key(name);
        {
            std::scoped_lock lock(host_tensor_mutex);
            const auto found = host_tensors.find(key);
            if (found != host_tensors.end()) {
                if (found->second->size() != elements) {
                    result.errors.push_back(
                        "GLM-5.3 cached host tensor has an invalid extent: " + key);
                } else {
                    result.value = found->second;
                }
                return result;
            }
        }
        auto loaded = checkpoint->read_f32(name, elements);
        if (!loaded.ok()) {
            result.errors = std::move(loaded.errors);
            return result;
        }
        auto value = std::make_shared<const std::vector<float>>(
            std::move(loaded.value));
        {
            std::scoped_lock lock(host_tensor_mutex);
            const auto [found, inserted] = host_tensors.emplace(key, value);
            result.value = inserted ? std::move(value) : found->second;
        }
        return result;
    }

    // Resolves one expert's three matrices, caching them. `expert_slot` is the
    // routed expert id, or 288 for the shared expert.
    // The FP8 release's host MoE quantizes its own activations to E4M3 so the
    // host reproduces what `enqueue_glm53_expert_*` does on the device. The
    // MXFP4 release has no device counterpart and no E4M3 anywhere in its
    // expert path, so quantizing there would discard precision the format
    // never asked to lose: it is W4A16, weights only.
    [[nodiscard]] bool fp8_expert_checkpoint() const noexcept {
        return checkpoint->manifest().quantization ==
               Glm53Quantization::Fp8E4m3Block128;
    }

    [[nodiscard]] ValidationResult expert_views(
        std::uint32_t layer, std::uint32_t expert_slot, std::string_view prefix,
        Glm53ExpertViews& out) const {
        ValidationResult result;
        if (layer >= kLayers || expert_slot >= kExpertSlots) {
            result.errors.emplace_back("GLM-5.3 expert view index is invalid");
            return result;
        }
        const std::size_t index =
            static_cast<std::size_t>(layer) * kExpertSlots + expert_slot;
        std::scoped_lock guard(expert_view_mutex);
        if (expert_view_cache.empty()) {
            expert_view_cache.resize(static_cast<std::size_t>(kLayers) *
                                     kExpertSlots);
            expert_view_ready.assign(expert_view_cache.size(), 0U);
        }
        if (expert_view_ready[index] == 0U) {
            const auto module =
                expert_slot + 1U == kExpertSlots
                    ? std::string(prefix) + "shared_experts."
                    : std::string(prefix) + "experts." +
                          std::to_string(expert_slot) + ".";
            auto gate = host_expert_linear(module + "gate_proj", 2048U, kHidden);
            auto up = host_expert_linear(module + "up_proj", 2048U, kHidden);
            auto down = host_expert_linear(module + "down_proj", kHidden, 2048U);
            if (!gate.ok() || !up.ok() || !down.ok()) {
                if (!gate.ok()) append(result.errors, std::move(gate.errors));
                if (!up.ok()) append(result.errors, std::move(up.errors));
                if (!down.ok()) append(result.errors, std::move(down.errors));
                return result;
            }
            expert_view_cache[index] = {gate.value, up.value, down.value};
            expert_view_ready[index] = 1U;
        }
        out = expert_view_cache[index];
        return result;
    }

    // Resolves one expert projection into a mapped view, from whichever of the
    // three storage formats the open checkpoint actually holds. The shapes are
    // the discriminator, not a configuration flag: an FP8 checkpoint can only
    // present E4M3 rows and an MXFP4 checkpoint only packed nibbles or BF16, so
    // a mismatch here is a corrupt or unsupported checkpoint rather than a
    // wrongly selected path.
    [[nodiscard]] ParseResult<Glm53HostExpertLinear> host_expert_linear(
        std::string_view base, std::uint32_t rows,
        std::uint32_t columns) const {
        ParseResult<Glm53HostExpertLinear> result;
        const auto weight_name = std::string(base) + ".weight";
        const auto* descriptor = checkpoint->find(weight_name);
        if (descriptor == nullptr) {
            result.errors.push_back(
                "GLM-5.3 host expert is missing a weight: " + weight_name);
            return result;
        }
        const auto invalid = [&](std::string_view what) {
            result.errors.push_back("GLM-5.3 host expert has an invalid " +
                                    std::string(what) + ": " +
                                    std::string(base));
        };
        std::string scale_name;
        std::uint64_t expected_weight_bytes = 0U;
        std::uint64_t expected_scale_bytes = 0U;
        Glm53ExpertEncoding encoding{};
        if (descriptor->source_dtype == SafetensorsDtype::F8E4M3) {
            const auto scale_rows = (rows + 127U) / 128U;
            const auto scale_columns = (columns + 127U) / 128U;
            scale_name = std::string(base) + ".weight_scale_inv";
            const auto* scale = checkpoint->find(scale_name);
            if (descriptor->source_shape !=
                    std::vector<std::uint64_t>{rows, columns} ||
                scale == nullptr ||
                scale->source_dtype != SafetensorsDtype::F32 ||
                scale->source_shape !=
                    std::vector<std::uint64_t>{scale_rows, scale_columns}) {
                invalid("FP8 linear");
                return result;
            }
            encoding = Glm53ExpertEncoding::Fp8E4m3Block128F32;
            expected_weight_bytes =
                static_cast<std::uint64_t>(rows) * columns;
            expected_scale_bytes = static_cast<std::uint64_t>(scale_rows) *
                                   scale_columns * sizeof(float);
        } else if (descriptor->source_dtype == SafetensorsDtype::U8) {
            scale_name = std::string(base) + ".weight_scale";
            const auto* scale = checkpoint->find(scale_name);
            if (columns % 32U != 0U ||
                descriptor->source_shape !=
                    std::vector<std::uint64_t>{rows, columns / 2U} ||
                scale == nullptr ||
                scale->source_dtype != SafetensorsDtype::U8 ||
                scale->source_shape !=
                    std::vector<std::uint64_t>{rows, columns / 32U}) {
                invalid("MXFP4 linear");
                return result;
            }
            encoding = Glm53ExpertEncoding::Fp4E2m1Group32E8m0;
            expected_weight_bytes =
                static_cast<std::uint64_t>(rows) * (columns / 2U);
            expected_scale_bytes =
                static_cast<std::uint64_t>(rows) * (columns / 32U);
        } else if (descriptor->source_dtype == SafetensorsDtype::Bf16) {
            if (descriptor->source_shape !=
                    std::vector<std::uint64_t>{rows, columns}) {
                invalid("BF16 linear");
                return result;
            }
            encoding = Glm53ExpertEncoding::Bf16;
            expected_weight_bytes =
                static_cast<std::uint64_t>(rows) * columns * 2U;
        } else {
            invalid("expert dtype");
            return result;
        }
        auto weight_payload = checkpoint->view(weight_name);
        if (!weight_payload.ok()) {
            result.errors = std::move(weight_payload.errors);
            return result;
        }
        std::span<const std::byte> scale_payload;
        if (!scale_name.empty()) {
            auto scales = checkpoint->view(scale_name);
            if (!scales.ok()) {
                result.errors = std::move(scales.errors);
                return result;
            }
            scale_payload = scales.value;
            if (encoding == Glm53ExpertEncoding::Fp8E4m3Block128F32 &&
                reinterpret_cast<std::uintptr_t>(scale_payload.data()) %
                        alignof(float) != 0U) {
                result.errors.push_back(
                    "GLM-5.3 host expert mapped payload is mis-aligned");
                return result;
            }
        }
        if (weight_payload.value.size_bytes() != expected_weight_bytes ||
            scale_payload.size_bytes() != expected_scale_bytes) {
            result.errors.push_back(
                "GLM-5.3 host expert mapped payload is mis-sized");
            return result;
        }
        result.value = {weight_payload.value, scale_payload, rows, columns,
                        encoding};
        return result;
    }

    // Admits every shared expert to the GPU that owns its layer.
    //
    // The tier is admitted whole or not at all. A partly admitted tier would
    // make host expert service depend on which layer ran, which would make
    // every subsequent A/B a comparison of two different workloads.
    [[nodiscard]] ValidationResult admit_shared_experts() {
        shared_experts.devices.fill(-1);
        shared_experts.active = false;
        shared_experts.bytes = 0U;
        shared_experts.bytes_by_slot.assign(devices.size(), 0U);
        const auto override = shared_expert_device_override();
        if (override == 0 || !host_moe_active || devices.empty()) return {};
        constexpr std::uint32_t intermediate = 2048U;
        const bool fp8 = fp8_expert_checkpoint();
        const auto encoding = fp8
            ? CudaGlm53ExpertEncoding::Fp8E4m3Block128F32
            : CudaGlm53ExpertEncoding::Bf16;
        // Gate and up are intermediate x hidden and down is hidden x
        // intermediate. FP8 carries one F32 scale per 128x128 tile; BF16 is
        // two checkpoint-native bytes per value and has no scale payload.
        const std::uint64_t projection_bytes =
            static_cast<std::uint64_t>(intermediate) * kHidden *
            (fp8 ? 1U : 2U);
        const std::uint64_t scale_bytes = fp8 ?
            static_cast<std::uint64_t>(intermediate / 128U) *
                (kHidden / 128U) * sizeof(float) : 0U;
        const std::uint64_t layer_bytes = 3U * projection_bytes +
                                          3U * scale_bytes;
        std::vector<std::uint64_t> required(devices.size(), 0U);
        std::vector<std::uint32_t> admitted_layers;
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            if (!glm53_moe_layer(layer)) continue;
            required[slot_for(layer)] += layer_bytes;
            admitted_layers.push_back(layer);
        }
        if (admitted_layers.empty()) return {};
        // The tier holds its own device memory rather than living in the
        // weight cache's arena, so its bytes come out of that arena before the
        // cache is built with them -- the same reservation the resident mHC
        // weights make, for the same reason. Reading free VRAM here would
        // prove nothing: the cache fills lazily during prefill, so at this
        // point almost all of the capacity it has been promised is still
        // unclaimed and still free.
        for (std::size_t slot = 0U; slot < devices.size(); ++slot) {
            if (required[slot] == 0U) continue;
            if (weight_capacities[slot] <=
                required[slot] + kMinimumDeviceBudget) {
                // An explicit opt-in that cannot fit is an error; the default
                // admission simply declines and leaves the host path intact.
                if (override > 0) {
                    return {{"GLM-5.3 shared expert tier does not fit the "
                             "admitted CUDA capacity on device " +
                             std::to_string(devices[slot])}};
                }
                return {};
            }
        }
        for (std::size_t slot = 0U; slot < devices.size(); ++slot) {
            weight_capacities[slot] -= required[slot];
        }

        shared_experts.storage.resize(admitted_layers.size() * 6U);
        std::size_t cursor = 0U;
        for (const auto layer : admitted_layers) {
            const auto slot = slot_for(layer);
            const auto device = devices[slot];
            const auto module = "model.language_model.layers." +
                std::to_string(layer) + ".mlp.shared_experts.";
            struct Projection {
                std::string name;
                std::uint32_t rows;
                std::uint32_t columns;
            };
            const std::array<Projection, 3U> projections{
                Projection{module + "gate_proj", intermediate, kHidden},
                Projection{module + "up_proj", intermediate, kHidden},
                Projection{module + "down_proj", kHidden, intermediate}};
            std::array<CudaBuffer*, 6U> uploaded{};
            for (std::size_t index = 0U; index < projections.size(); ++index) {
                auto linear = host_expert_linear(projections[index].name,
                                                 projections[index].rows,
                                                 projections[index].columns);
                if (!linear.ok()) return {std::move(linear.errors)};
                auto& weight_buffer = shared_experts.storage[cursor++];
                auto& scale_buffer = shared_experts.storage[cursor++];
                auto weight_upload = cuda.upload_buffer(
                    device, linear.value.weights, weight_buffer);
                if (!weight_upload.ok()) return weight_upload;
                if (linear.value.encoding != (fp8
                        ? Glm53ExpertEncoding::Fp8E4m3Block128F32
                        : Glm53ExpertEncoding::Bf16)) {
                    return {{"GLM-5.3 shared expert tier mixes checkpoint "
                             "encodings"}};
                }
                uploaded[index * 2U] = &weight_buffer;
                if (fp8) {
                    auto scale_upload = cuda.upload_buffer(
                        device, std::as_bytes(linear.value.scales),
                        scale_buffer);
                    if (!scale_upload.ok()) return scale_upload;
                    uploaded[index * 2U + 1U] = &scale_buffer;
                }
                shared_experts.bytes += linear.value.weights.size_bytes() +
                                        linear.value.scales.size_bytes();
                shared_experts.bytes_by_slot[slot] +=
                    linear.value.weights.size_bytes() +
                    linear.value.scales.size_bytes();
            }
            shared_experts.experts[layer] = {
                uploaded[0], uploaded[1], uploaded[2],
                uploaded[3], uploaded[4], uploaded[5],
                kHidden, intermediate, encoding};
            shared_experts.devices[layer] = device;
        }
        shared_experts.active = true;
        return {};
    }

    [[nodiscard]] ValidationResult admit_static_experts() {
        static_experts.bytes_by_slot.assign(devices.size(), 0U);
        static_experts.bytes = 0U;
        static_experts.experts_admitted = 0U;
        static_experts.active_tier = false;
        const auto override = static_expert_override();
        if (override == 0) return {};
        const auto& paths = static_expert_census_paths();
        if (!host_moe_active || weights == nullptr || devices.empty()) {
            return override > 0
                ? ValidationResult{{"GLM-5.3 static expert tier requires host "
                                    "MoE and CUDA"}}
                : ValidationResult{};
        }
        struct Candidate {
            std::uint64_t frequency{};
            std::uint32_t layer{};
            std::uint32_t expert{};
        };
        std::array<std::array<std::uint64_t, 288U>, kLayers> counts{};
        std::size_t files = 0U;
        std::vector<Candidate> candidates;
        if (!paths.empty()) {
            std::size_t begin = 0U;
            while (begin <= paths.size()) {
                const auto end = paths.find(';', begin);
                const auto path = paths.substr(
                    begin, end == std::string::npos ? std::string::npos
                                                    : end - begin);
                if (!path.empty()) {
                    std::ifstream input(path);
                    if (!input) {
                        return {{"GLM-5.3 static expert census cannot be opened: " +
                                 path}};
                    }
                    ++files;
                    std::string line;
                    while (std::getline(input, line)) {
                        std::array<std::string, 5U> fields;
                        std::size_t cursor = 0U;
                        bool valid = true;
                        for (std::size_t index = 0U; index < fields.size();
                             ++index) {
                            const auto tab = line.find('\t', cursor);
                            if (index + 1U != fields.size() &&
                                tab == std::string::npos) {
                                valid = false;
                                break;
                            }
                            fields[index] = line.substr(
                                cursor, tab == std::string::npos
                                            ? std::string::npos : tab - cursor);
                            cursor = tab == std::string::npos ? line.size()
                                                              : tab + 1U;
                        }
                        if (!valid || fields[0] != "decode") continue;
                        const auto layer = std::strtoul(
                            fields[3].c_str(), nullptr, 10);
                        if (layer >= kLayers || !glm53_moe_layer(
                                static_cast<std::uint32_t>(layer))) continue;
                        std::stringstream experts(fields[4]);
                        std::string expert;
                        while (std::getline(experts, expert, ',')) {
                            const auto parsed = std::strtoul(
                                expert.c_str(), nullptr, 10);
                            if (parsed < 288U) ++counts[layer][parsed];
                        }
                    }
                }
                if (end == std::string::npos) break;
                begin = end + 1U;
            }
            for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
                for (std::uint32_t expert = 0U; expert < 288U; ++expert) {
                    if (counts[layer][expert] != 0U) {
                        candidates.push_back(
                            {counts[layer][expert], layer, expert});
                    }
                }
            }
            std::sort(candidates.begin(), candidates.end(),
                      [](const Candidate& left, const Candidate& right) {
                          if (left.frequency != right.frequency) {
                              return left.frequency > right.frequency;
                          }
                          if (left.layer != right.layer) {
                              return left.layer < right.layer;
                          }
                          return left.expert < right.expert;
                      });
        } else {
            std::array<std::array<bool, 288U>, kLayers> ranked{};
            const auto profile = glm53_default_expert_ranking();
            candidates.reserve(static_cast<std::size_t>(kLayers) * 288U);
            for (std::size_t index = 0U; index < profile.size(); ++index) {
                const auto layer = static_cast<std::uint32_t>(profile[index] >> 9U);
                const auto expert = static_cast<std::uint32_t>(profile[index] & 0x1ffU);
                if (layer >= kLayers || expert >= 288U ||
                    !glm53_moe_layer(layer) || ranked[layer][expert]) continue;
                ranked[layer][expert] = true;
                candidates.push_back({profile.size() - index, layer, expert});
            }
            // A larger accelerator can use more than the measured host's hot
            // prefix. Fill its remaining admitted arena deterministically;
            // never strand capacity merely because the profile was bounded.
            for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
                if (!glm53_moe_layer(layer)) continue;
                for (std::uint32_t expert = 0U; expert < 288U; ++expert) {
                    if (!ranked[layer][expert]) {
                        candidates.push_back({0U, layer, expert});
                    }
                }
            }
        }
        for (const auto& candidate : candidates) {
            const auto slot = slot_for(candidate.layer);
            CudaGlm53Expert descriptor;
            bool admitted = false;
            auto status = weights->pin_expert(
                slot, candidate.layer, candidate.expert, descriptor, admitted);
            if (!status.ok()) return status;
            if (!admitted) continue;
            const auto bytes = descriptor.gate->device_bytes() +
                               descriptor.up->device_bytes() +
                               descriptor.down->device_bytes();
            static_experts.experts[candidate.layer][candidate.expert] =
                descriptor;
            static_experts.active[candidate.layer][candidate.expert] = 1U;
            static_experts.bytes_by_slot[slot] += bytes;
            static_experts.bytes += bytes;
            ++static_experts.experts_admitted;
        }
        if (static_experts.experts_admitted == 0U) {
            return override > 0
                ? ValidationResult{{"GLM-5.3 static expert tier admitted no experts"}}
                : ValidationResult{};
        }
        static_experts.active_tier = true;
        // Size every device-expert scratch buffer before the timed path. A
        // layer can dispatch the eight routed experts plus its shared expert
        // together, and glm53_grow must therefore remain a no-op in decode.
        shared_expert_gate.resize(9U * 2048U);
        shared_expert_up.resize(9U * 2048U);
        shared_expert_output.resize(9U * kHidden);
        std::cerr << "[glm53-static-tier] profile="
                  << (paths.empty() ? "builtin" : "census")
                  << " census_files=" << files
                  << " experts=" << static_experts.experts_admitted
                  << " bytes=" << static_experts.bytes;
        for (std::size_t slot = 0U; slot < devices.size(); ++slot) {
            std::cerr << " cuda" << devices[slot] << "_bytes="
                      << static_experts.bytes_by_slot[slot];
        }
        std::cerr << '\n';
        return {};
    }

    [[nodiscard]] ValidationResult host_moe(
        std::uint32_t layer, std::string_view prefix,
        std::span<const KimiRoutedExpert> routed,
        std::span<const float> input, std::span<float> output) {
        ValidationResult result;
        constexpr std::uint32_t intermediate = 2048U;
        constexpr std::size_t missing = std::numeric_limits<std::size_t>::max();
        if (!host_moe_active || host_moe_workers == nullptr ||
            routed.size() != 8U || input.size() != kHidden ||
            output.size() != kHidden) {
            return {{"GLM-5.3 host MoE command has an invalid shape"}};
        }
        const auto started = std::chrono::steady_clock::now();
        constexpr std::size_t route_limit = 8U;
        constexpr bool include_shared = true;
        const int layer_device = device_for(layer);
        const int shared_device = include_shared && shared_experts.active &&
                                  layer < shared_experts.devices.size()
            ? shared_experts.devices[layer] : -1;

        std::array<std::size_t, 9U> host_index;
        std::array<std::size_t, 9U> device_index;
        host_index.fill(missing);
        device_index.fill(missing);
        std::array<std::size_t, 9U> host_positions{};
        std::array<CudaGlm53Expert, 9U> device_experts{};
        std::size_t host_count = 0U;
        std::size_t device_count = 0U;
        for (std::size_t route = 0U; route < route_limit; ++route) {
            const auto expert = static_cast<std::uint32_t>(routed[route].expert);
            const bool resident = static_experts.active_tier &&
                static_experts.active[layer][expert] != 0U;
            if (resident) {
                device_index[route] = device_count;
                device_experts[device_count++] =
                    static_experts.experts[layer][expert];
                static_experts.route_hits.fetch_add(
                    1U, std::memory_order_relaxed);
            } else {
                host_index[route] = host_count;
                host_positions[host_count++] = route;
                if (static_experts.active_tier) {
                    static_experts.route_misses.fetch_add(
                        1U, std::memory_order_relaxed);
                }
            }
        }
        if (include_shared) {
            if (shared_device >= 0) {
                if (shared_device != layer_device) {
                    return {{"GLM-5.3 expert tiers disagree on layer owner"}};
                }
                device_index[8U] = device_count;
                device_experts[device_count++] = shared_experts.experts[layer];
                shared_expert_device_calls.fetch_add(
                    1U, std::memory_order_relaxed);
            } else {
                host_index[8U] = host_count;
                host_positions[host_count++] = 8U;
                if (shared_experts.active) {
                    shared_expert_host_calls.fetch_add(
                        1U, std::memory_order_relaxed);
                    if (layer < 64U) {
                        shared_expert_host_layers.fetch_or(
                            std::uint64_t{1U} << layer,
                            std::memory_order_relaxed);
                    }
                }
            }
        }

        const auto view_started = std::chrono::steady_clock::now();
        std::array<Glm53ExpertViews, 9U> host_experts{};
        for (std::size_t index = 0U; index < host_count; ++index) {
            const auto position = host_positions[index];
            const auto slot = position < 8U
                ? static_cast<std::uint32_t>(routed[position].expert)
                : kExpertSlots - 1U;
            result = expert_views(layer, slot, prefix, host_experts[index]);
            if (!result.ok()) return result;
        }
        std::uint64_t allocations = 0U;
        if (config.phase_profile) {
            std::uint64_t gate_up_bytes = 0U;
            std::uint64_t down_bytes = 0U;
            for (std::size_t index = 0U; index < host_count; ++index) {
                const auto& expert = host_experts[index];
                gate_up_bytes += expert.gate.weights.size_bytes() +
                                 expert.gate.scales.size_bytes() +
                                 expert.up.weights.size_bytes() +
                                 expert.up.scales.size_bytes();
                down_bytes += expert.down.weights.size_bytes() +
                              expert.down.scales.size_bytes();
            }
            host_moe_gate_up_weight_bytes.fetch_add(
                gate_up_bytes, std::memory_order_relaxed);
            host_moe_down_weight_bytes.fetch_add(
                down_bytes, std::memory_order_relaxed);
            host_moe_view_nanoseconds.fetch_add(
                elapsed_nanoseconds(view_started), std::memory_order_relaxed);
        }

        const auto input_quantization_started =
            std::chrono::steady_clock::now();
        auto& quantized_input = host_moe_quantized_input;
        if (glm53_grow(quantized_input, input.size())) ++allocations;
        std::copy(input.begin(), input.end(), quantized_input.begin());
        if (fp8_expert_checkpoint()) {
            glm53_quantize_activation(
                std::span<float>(quantized_input).first(input.size()));
        }
        if (config.phase_profile) {
            host_moe_input_quantization_nanoseconds.fetch_add(
                elapsed_nanoseconds(input_quantization_started),
                std::memory_order_relaxed);
        }
        const auto device_expert_span =
            std::span<const CudaGlm53Expert>(device_experts).first(device_count);
        if (device_count != 0U) {
            result = cuda.enqueue_glm53_expert_gate_up(
                layer_device, device_expert_span,
                std::span<const float>(quantized_input).first(kHidden));
            if (!result.ok()) return result;
        }

        auto& activations = host_moe_activations;
        if (glm53_grow(activations, host_count * intermediate)) ++allocations;
        const auto gate_up_started = std::chrono::steady_clock::now();
        if (host_count != 0U) {
            result = host_moe_workers->parallel_for_blocked(
                host_count * intermediate, expert_dispatch_block(),
                [&](std::size_t task) {
                    const auto expert = task / intermediate;
                    const auto row = task % intermediate;
                    const auto gate_row = glm53_host_expert_row(
                        host_experts[expert].gate, row);
                    const auto up_row = glm53_host_expert_row(
                        host_experts[expert].up, row);
                    auto gate = bf16_round_f32(glm53_host_expert_dot(
                        gate_row, quantized_input));
                    auto up = bf16_round_f32(glm53_host_expert_dot(
                        up_row, quantized_input));
                    gate = std::min(gate, 10.0F);
                    up = std::clamp(up, -10.0F, 10.0F);
                    activations[expert * intermediate + row] =
                        bf16_round_f32(gate * sigmoid(gate) * up);
                });
            if (!result.ok()) return result;
        }
        if (config.phase_profile) {
            host_moe_gate_up_nanoseconds.fetch_add(
                elapsed_nanoseconds(gate_up_started),
                std::memory_order_relaxed);
        }

        const auto activation_started = std::chrono::steady_clock::now();
        if (fp8_expert_checkpoint()) {
            for (std::size_t expert = 0U; expert < host_count; ++expert) {
                glm53_quantize_activation(std::span<float>(activations).subspan(
                    expert * intermediate, intermediate));
            }
        }
        if (device_count != 0U) {
            const auto device_floats = device_count * intermediate;
            if (glm53_grow(shared_expert_gate, device_floats)) ++allocations;
            if (glm53_grow(shared_expert_up, device_floats)) ++allocations;
            result = cuda.collect_glm53_expert_gate_up(
                layer_device,
                std::span<float>(shared_expert_gate).first(device_floats),
                std::span<float>(shared_expert_up).first(device_floats));
            if (!result.ok()) return result;
            for (std::size_t index = 0U; index < device_floats; ++index) {
                auto gate = bf16_round_f32(shared_expert_gate[index]);
                auto up = bf16_round_f32(shared_expert_up[index]);
                gate = std::min(gate, 10.0F);
                up = std::clamp(up, -10.0F, 10.0F);
                shared_expert_gate[index] =
                    bf16_round_f32(gate * sigmoid(gate) * up);
            }
            if (fp8_expert_checkpoint()) {
                for (std::size_t expert = 0U; expert < device_count; ++expert) {
                    glm53_quantize_activation(
                        std::span<float>(shared_expert_gate).subspan(
                            expert * intermediate, intermediate));
                }
            }
            result = cuda.enqueue_glm53_expert_down(
                layer_device, device_expert_span,
                std::span<const float>(shared_expert_gate).first(
                    device_floats));
            if (!result.ok()) return result;
        }
        if (config.phase_profile) {
            host_moe_activation_nanoseconds.fetch_add(
                elapsed_nanoseconds(activation_started),
                std::memory_order_relaxed);
        }

        auto& host_outputs = host_moe_expert_outputs;
        if (glm53_grow(host_outputs, host_count * kHidden)) ++allocations;
        if (config.phase_profile) {
            host_moe_temporary_allocation_calls.fetch_add(
                allocations, std::memory_order_relaxed);
        }
        const auto down_started = std::chrono::steady_clock::now();
        if (host_count != 0U) {
            result = host_moe_workers->parallel_for_blocked(
                host_count * kHidden, expert_dispatch_block(),
                [&](std::size_t task) {
                    const auto expert = task / kHidden;
                    const auto row = task % kHidden;
                    const auto weight_row = glm53_host_expert_row(
                        host_experts[expert].down, row);
                    host_outputs[expert * kHidden + row] = bf16_round_f32(
                        glm53_host_expert_dot(
                            weight_row,
                            std::span<const float>(activations).subspan(
                                expert * intermediate, intermediate)));
                });
            if (!result.ok()) return result;
        }
        if (config.phase_profile) {
            host_moe_down_nanoseconds.fetch_add(
                elapsed_nanoseconds(down_started), std::memory_order_relaxed);
        }

        const auto reduction_started = std::chrono::steady_clock::now();
        if (device_count != 0U) {
            const auto device_floats = device_count * kHidden;
            if (glm53_grow(shared_expert_output, device_floats)) ++allocations;
            result = cuda.collect_glm53_expert_down(
                layer_device,
                std::span<float>(shared_expert_output).first(device_floats));
            if (!result.ok()) return result;
        }
        const auto term = [&](std::size_t position, std::size_t column) {
            if (host_index[position] != missing) {
                return host_outputs[host_index[position] * kHidden + column];
            }
            if (device_index[position] != missing) {
                return bf16_round_f32(
                    shared_expert_output[
                        device_index[position] * kHidden + column]);
            }
            return 0.0F;
        };
        for (std::size_t column = 0U; column < kHidden; ++column) {
            output[column] = include_shared ? term(8U, column) : 0.0F;
        }
        for (std::size_t route = 0U; route < route_limit; ++route) {
            for (std::size_t column = 0U; column < kHidden; ++column) {
                output[column] = bf16_round_f32(
                    output[column] + bf16_round_f32(
                        routed[route].weight * term(route, column)));
            }
        }
        if (config.phase_profile) {
            host_moe_reduction_nanoseconds.fetch_add(
                elapsed_nanoseconds(reduction_started),
                std::memory_order_relaxed);
        }
        host_moe_calls.fetch_add(1U, std::memory_order_relaxed);
        host_moe_rows.fetch_add(1U, std::memory_order_relaxed);
        host_moe_nanoseconds.fetch_add(
            elapsed_nanoseconds(started), std::memory_order_relaxed);
        return result;
    }

    [[nodiscard]] ValidationResult host_moe_page(
        std::uint32_t layer, std::string_view prefix,
        std::span<const std::array<KimiRoutedExpert, 8U>> routes,
        std::span<const float> input, std::span<float> output,
        bool allow_device_tiers) {
        ValidationResult result;
        constexpr std::uint32_t intermediate = 2048U;
        constexpr std::size_t routes_per_row = 8U;
        constexpr std::size_t outputs_per_row = routes_per_row + 1U;
        const auto rows = routes.size();
        if (!host_moe_active || host_moe_workers == nullptr || rows == 0U ||
            input.size() != rows * kHidden ||
            output.size() != rows * kHidden) {
            return {{"GLM-5.3 host page MoE command has an invalid shape"}};
        }
        const auto started = std::chrono::steady_clock::now();
        const auto view_started = started;
        const int layer_device = device_for(layer);
        const int shared_device = allow_device_tiers &&
                                  shared_experts.active &&
                                  layer < shared_experts.devices.size()
            ? shared_experts.devices[layer] : -1;
        if (shared_device >= 0 && shared_device != layer_device) {
            return {{"GLM-5.3 expert tiers disagree on layer owner"}};
        }
        std::array<std::size_t, 288U> group_for_expert;
        group_for_expert.fill(std::numeric_limits<std::size_t>::max());
        auto& groups = page_groups;
        groups.clear();
        auto& device_experts = page_device_experts;
        auto& device_output_slots = page_device_output_slots;
        device_experts.clear();
        device_output_slots.clear();
        for (std::size_t row = 0U; row < rows; ++row) {
            for (std::size_t route = 0U; route < routes_per_row; ++route) {
                const auto expert = routes[row][route].expert;
                if (expert >= group_for_expert.size()) {
                    return {{"GLM-5.3 host page route is out of range"}};
                }
                const bool resident = allow_device_tiers &&
                    static_experts.active_tier &&
                    static_experts.active[layer][expert] != 0U;
                if (resident) {
                    device_experts.push_back(
                        static_experts.experts[layer][expert]);
                    device_output_slots.push_back(
                        row * outputs_per_row + route);
                    static_experts.route_hits.fetch_add(
                        1U, std::memory_order_relaxed);
                    continue;
                }
                if (allow_device_tiers && static_experts.active_tier) {
                    static_experts.route_misses.fetch_add(
                        1U, std::memory_order_relaxed);
                }
                auto& group_index = group_for_expert[expert];
                if (group_index == std::numeric_limits<std::size_t>::max()) {
                    group_index = groups.size();
                    groups.push_back({expert, false, {}, 0U, 0U});
                }
                ++groups[group_index].assignment_count;
            }
            if (shared_device >= 0) {
                device_experts.push_back(shared_experts.experts[layer]);
                device_output_slots.push_back(
                    row * outputs_per_row + routes_per_row);
                shared_expert_device_calls.fetch_add(
                    1U, std::memory_order_relaxed);
            }
        }
        if (shared_device < 0) {
            groups.push_back({0U, true, {}, 0U, rows});
            if (allow_device_tiers && shared_experts.active) {
                shared_expert_host_calls.fetch_add(
                    rows, std::memory_order_relaxed);
                if (layer < 64U) {
                    shared_expert_host_layers.fetch_or(
                        std::uint64_t{1U} << layer,
                        std::memory_order_relaxed);
                }
            }
        }
        // Group identical resident matrices contiguously. The CUDA primitive
        // then evaluates up to four independent activations while streaming a
        // weight row once. Insertion sort is allocation-free and the admitted
        // cohort is bounded to 32 * 9 assignments.
        const auto device_key = [&](std::size_t output_slot) {
            const auto row = output_slot / outputs_per_row;
            const auto route = output_slot % outputs_per_row;
            return route < routes_per_row
                ? static_cast<std::uint32_t>(routes[row][route].expert)
                : 288U;
        };
        for (std::size_t index = 1U; index < device_output_slots.size();
             ++index) {
            auto slot = device_output_slots[index];
            auto descriptor = device_experts[index];
            const auto key = device_key(slot);
            auto insertion = index;
            while (insertion != 0U &&
                   device_key(device_output_slots[insertion - 1U]) > key) {
                device_output_slots[insertion] =
                    device_output_slots[insertion - 1U];
                device_experts[insertion] = device_experts[insertion - 1U];
                --insertion;
            }
            device_output_slots[insertion] = slot;
            device_experts[insertion] = descriptor;
        }
        std::size_t assignment_begin = 0U;
        for (auto& group : groups) {
            group.assignment_begin = assignment_begin;
            assignment_begin += group.assignment_count;
        }
        if (assignment_begin + device_experts.size() !=
                rows * outputs_per_row ||
            device_experts.size() != device_output_slots.size() ||
            page_assignments.size() < assignment_begin) {
            return {{"GLM-5.3 host page assignment scratch is too small"}};
        }
        std::array<std::size_t, 289U> cursors{};
        for (std::size_t group = 0U; group < groups.size(); ++group) {
            cursors[group] = groups[group].assignment_begin;
        }
        for (std::size_t row = 0U; row < rows; ++row) {
            for (std::size_t route = 0U; route < routes_per_row; ++route) {
                const auto group = group_for_expert[routes[row][route].expert];
                if (group == std::numeric_limits<std::size_t>::max()) {
                    continue;
                }
                page_assignments[cursors[group]++] = {
                    row, row * outputs_per_row + route};
            }
            if (shared_device < 0) {
                const auto shared_group = groups.size() - 1U;
                page_assignments[cursors[shared_group]++] = {
                    row, row * outputs_per_row + routes_per_row};
            }
        }

        for (auto& group : groups) {
            const auto slot = group.shared ? kExpertSlots - 1U : group.expert;
            Glm53ExpertViews views;
            result = expert_views(layer, slot, prefix, views);
            if (!result.ok()) return result;
            group.module = {views.gate, views.up, views.down};
        }

        if (config.phase_profile) {
            std::uint64_t gate_up_bytes = 0U;
            std::uint64_t down_bytes = 0U;
            for (const auto& group : groups) {
                // The production page primitive traverses each expert weight
                // row once and reuses it across every assigned prompt row.
                gate_up_bytes += group.module.gate.weights.size_bytes() +
                                 group.module.gate.scales.size_bytes() +
                                 group.module.up.weights.size_bytes() +
                                 group.module.up.scales.size_bytes();
                down_bytes += group.module.down.weights.size_bytes() +
                              group.module.down.scales.size_bytes();
            }
            host_moe_gate_up_weight_bytes.fetch_add(
                gate_up_bytes, std::memory_order_relaxed);
            host_moe_down_weight_bytes.fetch_add(
                down_bytes, std::memory_order_relaxed);
            host_moe_view_nanoseconds.fetch_add(
                elapsed_nanoseconds(view_started), std::memory_order_relaxed);
        }

        const auto input_quantization_started =
            std::chrono::steady_clock::now();
        auto& quantized_input = page_quantized_input;
        if (quantized_input.size() < input.size()) {
            return {{"GLM-5.3 host page input scratch is too small"}};
        }
        std::copy(input.begin(), input.end(), quantized_input.begin());
        if (fp8_expert_checkpoint()) {
            result = host_moe_workers->parallel_for(
                rows, [&](std::size_t row) {
                    glm53_quantize_activation(
                        std::span<float>(quantized_input)
                            .subspan(row * kHidden, kHidden));
                });
            if (!result.ok()) return result;
        }
        if (config.phase_profile) {
            host_moe_input_quantization_nanoseconds.fetch_add(
                elapsed_nanoseconds(input_quantization_started),
                std::memory_order_relaxed);
        }

        const auto device_count = device_experts.size();
        struct DeviceCommand {
            std::size_t begin{};
            std::size_t count{};
        };
        std::array<DeviceCommand, 2U> device_commands{};
        std::size_t device_command_count = 0U;
        for (std::size_t begin = 0U; begin < device_count;) {
            auto end = begin + 1U;
            while (end < device_count &&
                   device_experts[end].encoding ==
                       device_experts[begin].encoding) {
                ++end;
            }
            if (device_command_count == device_commands.size()) {
                return {{"GLM-5.3 device page requires too many expert "
                         "encoding commands"}};
            }
            device_commands[device_command_count++] = {begin, end - begin};
            begin = end;
        }
        auto& device_inputs = page_device_inputs;
        if (device_inputs.size() < device_count * kHidden) {
            return {{"GLM-5.3 device page input scratch is too small"}};
        }
        for (std::size_t index = 0U; index < device_count; ++index) {
            const auto input_row =
                device_output_slots[index] / outputs_per_row;
            std::copy_n(quantized_input.begin() +
                            static_cast<std::ptrdiff_t>(input_row * kHidden),
                        kHidden,
                        device_inputs.begin() +
                            static_cast<std::ptrdiff_t>(index * kHidden));
        }
        if (device_command_count != 0U) {
            const auto& command = device_commands.front();
            result = cuda.enqueue_glm53_expert_gate_up(
                layer_device,
                std::span<const CudaGlm53Expert>(device_experts)
                    .subspan(command.begin, command.count),
                std::span<const float>(device_inputs)
                    .subspan(command.begin * kHidden,
                             command.count * kHidden));
            if (!result.ok()) return result;
        }

        const auto output_slots = rows * outputs_per_row;
        auto& activations = page_activations;
        if (activations.size() < output_slots * intermediate) {
            return {{"GLM-5.3 host page activation scratch is too small"}};
        }
        const auto gate_up_started = std::chrono::steady_clock::now();
        result = host_moe_workers->parallel_for_blocked(
            groups.size() * intermediate, expert_dispatch_block(),
            [&](std::size_t task) {
                const auto group_index = task / intermediate;
                const auto projection_row = task % intermediate;
                const auto& group = groups[group_index];
                const auto gate_row =
                    glm53_host_expert_row(group.module.gate, projection_row);
                const auto up_row =
                    glm53_host_expert_row(group.module.up, projection_row);
                const auto assignments = std::span<const PageAssignment>(
                    page_assignments)
                    .subspan(group.assignment_begin,
                             group.assignment_count);
                for (const auto& assignment : assignments) {
                    const auto source = std::span<const float>(quantized_input)
                        .subspan(assignment.input_row * kHidden, kHidden);
                    auto gate = bf16_round_f32(
                        glm53_host_expert_dot(gate_row, source));
                    auto up = bf16_round_f32(
                        glm53_host_expert_dot(up_row, source));
                    gate = std::min(gate, 10.0F);
                    up = std::clamp(up, -10.0F, 10.0F);
                    activations[assignment.output_slot * intermediate +
                                projection_row] =
                        bf16_round_f32(gate * sigmoid(gate) * up);
                }
            });
        if (!result.ok()) return result;
        if (config.phase_profile) {
            host_moe_gate_up_nanoseconds.fetch_add(
                elapsed_nanoseconds(gate_up_started),
                std::memory_order_relaxed);
        }
        const auto activation_started = std::chrono::steady_clock::now();
        if (fp8_expert_checkpoint()) {
            result = host_moe_workers->parallel_for(
                assignment_begin, [&](std::size_t assignment) {
                    const auto slot = page_assignments[assignment].output_slot;
                    glm53_quantize_activation(std::span<float>(activations)
                                                  .subspan(slot * intermediate,
                                                           intermediate));
                });
            if (!result.ok()) return result;
        }
        const auto activate_device_command =
            [&](const DeviceCommand& command) -> ValidationResult {
            const auto begin = command.begin * intermediate;
            const auto count = command.count * intermediate;
            auto gate = std::span<float>(shared_expert_gate)
                .subspan(begin, count);
            auto up = std::span<float>(shared_expert_up)
                .subspan(begin, count);
            auto status = cuda.collect_glm53_expert_gate_up(
                layer_device, gate, up);
            if (!status.ok()) return status;
            for (std::size_t index = 0U; index < count; ++index) {
                auto gate_value = bf16_round_f32(gate[index]);
                auto up_value = bf16_round_f32(up[index]);
                gate_value = std::min(gate_value, 10.0F);
                up_value = std::clamp(up_value, -10.0F, 10.0F);
                gate[index] = bf16_round_f32(
                    gate_value * sigmoid(gate_value) * up_value);
            }
            if (fp8_expert_checkpoint()) {
                for (std::size_t expert = 0U; expert < command.count;
                     ++expert) {
                    glm53_quantize_activation(
                        gate.subspan(expert * intermediate, intermediate));
                }
            }
            return cuda.enqueue_glm53_expert_down(
                layer_device,
                std::span<const CudaGlm53Expert>(device_experts)
                    .subspan(command.begin, command.count),
                std::span<const float>(gate));
        };
        if (device_command_count != 0U) {
            result = activate_device_command(device_commands.front());
            if (!result.ok()) return result;
        }
        if (config.phase_profile) {
            host_moe_activation_nanoseconds.fetch_add(
                elapsed_nanoseconds(activation_started),
                std::memory_order_relaxed);
        }

        auto& expert_outputs = page_expert_outputs;
        if (expert_outputs.size() < output_slots * kHidden) {
            return {{"GLM-5.3 host page output scratch is too small"}};
        }
        const auto down_started = std::chrono::steady_clock::now();
        result = host_moe_workers->parallel_for_blocked(
            groups.size() * kHidden, expert_dispatch_block(),
            [&](std::size_t task) {
                const auto group_index = task / kHidden;
                const auto projection_row = task % kHidden;
                const auto& group = groups[group_index];
                const auto projection_view = glm53_host_expert_row(
                    group.module.down, projection_row);
                const auto assignments = std::span<const PageAssignment>(
                    page_assignments)
                    .subspan(group.assignment_begin,
                             group.assignment_count);
                for (const auto& assignment : assignments) {
                    expert_outputs[assignment.output_slot * kHidden +
                                   projection_row] = bf16_round_f32(
                        glm53_host_expert_dot(
                            projection_view,
                            std::span<const float>(activations).subspan(
                                assignment.output_slot * intermediate,
                                intermediate)));
                }
            });
        if (!result.ok()) return result;
        if (config.phase_profile) {
            host_moe_down_nanoseconds.fetch_add(
                elapsed_nanoseconds(down_started), std::memory_order_relaxed);
        }
        const auto collect_device_command =
            [&](const DeviceCommand& command) -> ValidationResult {
            auto device_output = std::span<float>(shared_expert_output)
                .subspan(command.begin * kHidden, command.count * kHidden);
            auto status = cuda.collect_glm53_expert_down(
                layer_device, device_output);
            if (!status.ok()) return status;
            for (std::size_t offset = 0U; offset < command.count; ++offset) {
                const auto index = command.begin + offset;
                const auto source = device_output.subspan(
                    offset * kHidden, kHidden);
                auto destination = std::span<float>(expert_outputs).subspan(
                    device_output_slots[index] * kHidden, kHidden);
                for (std::size_t column = 0U; column < kHidden; ++column) {
                    // Match host_moe::term: CUDA dots are raw, and every
                    // device expert output crosses the same BF16 boundary
                    // before the shared/routed reduction.
                    destination[column] = bf16_round_f32(source[column]);
                }
            }
            return {};
        };
        if (device_command_count != 0U) {
            result = collect_device_command(device_commands.front());
            if (!result.ok()) return result;
        }
        for (std::size_t command_index = 1U;
             command_index < device_command_count; ++command_index) {
            const auto& command = device_commands[command_index];
            result = cuda.enqueue_glm53_expert_gate_up(
                layer_device,
                std::span<const CudaGlm53Expert>(device_experts)
                    .subspan(command.begin, command.count),
                std::span<const float>(device_inputs)
                    .subspan(command.begin * kHidden,
                             command.count * kHidden));
            if (!result.ok()) return result;
            result = activate_device_command(command);
            if (!result.ok()) return result;
            result = collect_device_command(command);
            if (!result.ok()) return result;
        }
        const auto reduction_started = std::chrono::steady_clock::now();
        result = host_moe_workers->parallel_for(
            rows * kHidden, [&](std::size_t task) {
                const auto row = task / kHidden;
                const auto column = task % kHidden;
                auto value = expert_outputs[
                    (row * outputs_per_row + routes_per_row) * kHidden +
                    column];
                for (std::size_t route = 0U; route < routes_per_row; ++route) {
                    value = bf16_round_f32(
                        value + bf16_round_f32(
                            routes[row][route].weight *
                            expert_outputs[
                                (row * outputs_per_row + route) * kHidden +
                                column]));
                }
                output[row * kHidden + column] = value;
            });
        if (!result.ok()) return result;
        if (config.phase_profile) {
            host_moe_reduction_nanoseconds.fetch_add(
                elapsed_nanoseconds(reduction_started),
                std::memory_order_relaxed);
        }
        host_moe_calls.fetch_add(1U, std::memory_order_relaxed);
        host_moe_rows.fetch_add(rows, std::memory_order_relaxed);
        host_moe_nanoseconds.fetch_add(
            static_cast<std::uint64_t>(std::chrono::duration_cast<
                std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                          started).count()),
            std::memory_order_relaxed);
        return result;
    }

    [[nodiscard]] ValidationResult wait_for_warmup() {
        std::scoped_lock lock(warmup_mutex);
        if (warmup_thread.joinable()) warmup_thread.join();
        return warmup_result;
    }

    [[nodiscard]] std::size_t restore_prefix(
        std::span<const std::uint32_t> tokens, Glm53SequenceState& state,
        std::span<float> logits, std::vector<float>& base_hidden) {
        std::scoped_lock lock(prefix_mutex);
        PrefixEntry* best = nullptr;
        for (auto& entry : prefix_cache) {
            if (entry.tokens.size() > tokens.size() ||
                entry.logits.size() != logits.size() ||
                (best != nullptr &&
                 entry.tokens.size() <= best->tokens.size()) ||
                !std::equal(entry.tokens.begin(), entry.tokens.end(),
                            tokens.begin())) {
                continue;
            }
            best = &entry;
        }
        if (best == nullptr) return 0U;
        state = best->state;
        std::copy(best->logits.begin(), best->logits.end(), logits.begin());
        base_hidden = best->base_hidden;
        best->recency = ++prefix_clock;
        prefix_cache_hits.fetch_add(1U, std::memory_order_relaxed);
        prefix_cache_tokens.fetch_add(best->tokens.size(),
                                      std::memory_order_relaxed);
        return best->tokens.size();
    }

    [[nodiscard]] static std::uint64_t prefix_entry_bytes(
        const PrefixEntry& entry) noexcept {
        return entry.state.private_bytes() +
            static_cast<std::uint64_t>(entry.tokens.capacity()) *
                sizeof(std::uint32_t) +
            static_cast<std::uint64_t>(entry.logits.capacity()) *
                sizeof(float) +
            static_cast<std::uint64_t>(entry.base_hidden.capacity()) *
                sizeof(float);
    }

    void store_prefix(std::span<const std::uint32_t> tokens,
                      const Glm53SequenceState& state,
                      std::span<const float> logits,
                      std::span<const float> base_hidden) {
        if (prefix_cache_limit == 0U || tokens.empty() ||
            state.token_count() != tokens.size()) return;
        std::scoped_lock lock(prefix_mutex);
        for (auto& entry : prefix_cache) {
            if (entry.tokens.size() == tokens.size() &&
                std::equal(entry.tokens.begin(), entry.tokens.end(),
                           tokens.begin())) {
                const auto old_bytes = prefix_entry_bytes(entry);
                entry.state = state;
                entry.logits.assign(logits.begin(), logits.end());
                entry.base_hidden.assign(base_hidden.begin(), base_hidden.end());
                entry.recency = ++prefix_clock;
                const auto current = prefix_cache_bytes.load(
                    std::memory_order_relaxed);
                prefix_cache_bytes.store(
                    current - old_bytes + prefix_entry_bytes(entry),
                    std::memory_order_relaxed);
                return;
            }
        }
        if (prefix_cache.size() >= prefix_cache_limit) {
            const auto victim = std::min_element(
                prefix_cache.begin(), prefix_cache.end(),
                [](const PrefixEntry& left, const PrefixEntry& right) {
                    return left.recency < right.recency;
                });
            if (victim != prefix_cache.end()) {
                const auto bytes = prefix_entry_bytes(*victim);
                prefix_cache_bytes.fetch_sub(bytes,
                                             std::memory_order_relaxed);
                prefix_cache_evictions.fetch_add(1U,
                                                 std::memory_order_relaxed);
                prefix_cache_evicted_bytes.fetch_add(
                    bytes, std::memory_order_relaxed);
                prefix_cache.erase(victim);
                prefix_cache_entries.store(prefix_cache.size(),
                                           std::memory_order_relaxed);
            }
        }
        PrefixEntry entry;
        entry.tokens.assign(tokens.begin(), tokens.end());
        entry.state = state;
        entry.logits.assign(logits.begin(), logits.end());
        entry.base_hidden.assign(base_hidden.begin(), base_hidden.end());
        entry.recency = ++prefix_clock;
        prefix_cache_bytes.fetch_add(prefix_entry_bytes(entry),
                                     std::memory_order_relaxed);
        prefix_cache.push_back(std::move(entry));
        prefix_cache_entries.store(prefix_cache.size(),
                                   std::memory_order_relaxed);
    }

    [[nodiscard]] ValidationResult warmup() {
        struct LinearTask {
            std::string base;
            std::string group;
            std::uint64_t rows{};
            std::uint64_t columns{};
            std::uint32_t layer{};
            std::uint64_t weight_rows{};
            std::uint64_t weight_row_begin{};
        };
        struct HostTask {
            std::string name;
            std::uint64_t elements{};
        };
        ValidationResult result;
        std::vector<LinearTask> linear_tasks;
        std::vector<std::vector<LinearTask>> device_tasks(devices.size());
        std::vector<HostTask> host_tasks;
        for (const auto& tensor : checkpoint->manifest().tensors) {
            if (tensor.role == Glm53TensorRole::Vision ||
                tensor.role == Glm53TensorRole::AttentionIndexer ||
                tensor.role == Glm53TensorRole::RoutedExpert ||
                tensor.role == Glm53TensorRole::Embedding ||
                tensor.name.find(".layers.45.mlp.experts.") !=
                    std::string::npos ||
                tensor.component == Glm53TensorComponent::Scale) {
                continue;
            }
            const bool linear = tensor.name.ends_with(".weight") &&
                                tensor.source_shape.size() == 2U;
            if (linear) {
                const auto layer = tensor.layer >= 0
                    ? static_cast<std::uint32_t>(tensor.layer)
                    : kLayers - 1U;
                const auto base = tensor.name.substr(
                    0U, tensor.name.size() - 7U);
                if (base == "lm_head" && lm_head_ranges.size() > 1U) {
                    for (std::size_t slot = 0U;
                         slot < lm_head_ranges.size(); ++slot) {
                        const auto range = lm_head_ranges[slot];
                        device_tasks[slot].push_back({
                            base, base, range.count, tensor.source_shape[1],
                            layer, tensor.source_shape[0], range.begin});
                    }
                    continue;
                }
                if (full_tensor_parallel_active) {
                    const auto ranges = weighted_row_ranges(
                        tensor.source_shape[0], weight_capacities, 128U);
                    if (ranges.size() == devices.size() &&
                        std::all_of(ranges.begin(), ranges.end(),
                                    [](const auto& range) {
                                        return range.count != 0U &&
                                               range.begin % 128U == 0U;
                                    })) {
                        for (std::size_t slot = 0U; slot < ranges.size();
                             ++slot) {
                            device_tasks[slot].push_back({
                                base, base, ranges[slot].count,
                                tensor.source_shape[1], layer,
                                tensor.source_shape[0], ranges[slot].begin});
                        }
                        continue;
                    }
                }
                linear_tasks.push_back({
                    base, projection_group_key(base, tensor.role),
                    tensor.source_shape[0], tensor.source_shape[1], layer});
                continue;
            }
            if (tensor.source_dtype != SafetensorsDtype::Bf16 &&
                tensor.source_dtype != SafetensorsDtype::F16 &&
                tensor.source_dtype != SafetensorsDtype::F32) {
                continue;
            }
            std::uint64_t elements = 1U;
            bool valid = !tensor.source_shape.empty();
            for (const auto dimension : tensor.source_shape) {
                if (dimension == 0U ||
                    elements > std::numeric_limits<std::uint64_t>::max() /
                                   dimension) {
                    valid = false;
                    break;
                }
                elements *= dimension;
            }
            if (valid) host_tasks.push_back({tensor.name, elements});
        }

        std::map<std::string, std::vector<LinearTask>> linear_groups;
        for (auto& task : linear_tasks) {
            linear_groups[task.group].push_back(std::move(task));
        }
        const bool parallel = projection_workers != nullptr &&
                              batched_projections_enabled() &&
                              cross_gpu_projections_enabled(devices) &&
                              devices.size() > 1U;
        for (auto& [group, tasks] : linear_groups) {
            static_cast<void>(group);
            if (!parallel || tasks.size() == 1U) {
                for (auto& task : tasks) {
                    device_tasks[slot_for(task.layer)].push_back(std::move(task));
                }
                continue;
            }
            std::vector<std::string_view> keys;
            std::vector<std::uint64_t> costs;
            keys.reserve(tasks.size());
            costs.reserve(tasks.size());
            for (const auto& task : tasks) {
                keys.push_back(task.base);
                costs.push_back(checkpoint->cuda_linear_storage_bytes(task.base));
            }
            const auto slots = glm53_projection_slots(
                keys, costs, weight_capacities, slot_for(tasks.front().layer));
            if (slots.size() != tasks.size()) {
                return {{"GLM-5.3 projection warmup assignment is invalid"}};
            }
            for (std::size_t index = 0U; index < tasks.size(); ++index) {
                device_tasks[slots[index]].push_back(std::move(tasks[index]));
            }
        }

        std::vector<ValidationResult> device_results(devices.size());
        std::vector<std::uint64_t> admitted(devices.size());
        std::vector<std::uint64_t> skipped(devices.size());
        std::atomic<std::size_t> next_slot{};
        const auto load_devices = [&] {
            for (;;) {
                const auto slot = next_slot.fetch_add(1U,
                                                       std::memory_order_relaxed);
                if (slot >= devices.size()) return;
                for (const auto& task : device_tasks[slot]) {
                    bool kept = false;
                    auto loaded = task.weight_rows == 0U
                        ? weights->preload(slot, task.base, task.rows,
                                           task.columns, kept)
                        : weights->preload_slice(
                              slot, task.base, task.weight_rows, task.columns,
                              task.weight_row_begin, task.rows, kept);
                    if (!loaded.ok()) {
                        append(device_results[slot].errors,
                               std::move(loaded.errors));
                        break;
                    }
                    kept ? ++admitted[slot] : ++skipped[slot];
                }
                if (device_results[slot].ok()) {
                    auto ordered = cuda.synchronize_uploads(devices[slot]);
                    if (!ordered.ok()) {
                        append(device_results[slot].errors,
                               std::move(ordered.errors));
                    }
                }
            }
        };
        const auto workers = std::min<std::size_t>(
            devices.size(), host_hardware_profile().worker_threads(0.1));
        std::vector<std::thread> loaders;
        loaders.reserve(workers);
        for (std::size_t worker = 0U; worker < workers; ++worker) {
            loaders.emplace_back(load_devices);
        }
        // Host-resident norms, convolution taps and mHC projections are only
        // 67 MiB for this checkpoint. Load them while independent PCIe links
        // receive their layer-split spine weights.
        for (const auto& task : host_tasks) {
            auto loaded = host_tensor(task.name, task.elements);
            if (!loaded.ok()) {
                append(result.errors, std::move(loaded.errors));
                break;
            }
        }
        for (auto& loader : loaders) loader.join();
        for (auto& device_result : device_results) {
            append(result.errors, std::move(device_result.errors));
        }
        if (result.ok() && resident_execution_active) {
            for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
                const auto prefix = "model.language_model.layers." +
                                    std::to_string(layer) + ".";
                const auto upload_mhc = [&](const std::string& mhc,
                                            const std::string& norm,
                                            CudaDsv4MhcWeights& destination)
                    -> ValidationResult {
                    auto projection = host_tensor(mhc + "_fn", 24U * 16384U);
                    auto base = host_tensor(mhc + "_base", 24U);
                    auto scale = host_tensor(mhc + "_scale", 3U);
                    auto norm_weight = host_tensor(norm, kHidden);
                    ValidationResult status;
                    if (!projection.ok() || !base.ok() || !scale.ok() ||
                        !norm_weight.ok()) {
                        append(status.errors, std::move(projection.errors));
                        append(status.errors, std::move(base.errors));
                        append(status.errors, std::move(scale.errors));
                        append(status.errors, std::move(norm_weight.errors));
                        return status;
                    }
                    return cuda.upload_dsv4_mhc_weights(
                        device_for(layer), *projection.value, *scale.value,
                        *base.value, *norm_weight.value, destination);
                };
                auto status = upload_mhc(
                    prefix + "hc_attn", prefix + "input_layernorm.weight",
                    resident_layers[layer].attention);
                if (!status.ok()) return status;
                status = upload_mhc(
                    prefix + "hc_ffn",
                    prefix + "post_attention_layernorm.weight",
                    resident_layers[layer].feedforward);
                if (!status.ok()) return status;
            }
        }
        if (result.ok()) {
            auto static_status = admit_static_experts();
            if (!static_status.ok()) return static_status;
            // The page and independent-sequence schedulers share the fused
            // mHC slot arena. Reserve its maximum admitted row count before
            // sampling free VRAM; previously the first request allocated it
            // after admission and made the reported concurrency optimistic.
            for (const auto device : devices) {
                auto slots = cuda.dsv4_mhc_reserve_slots(
                    device, config.prefill_page_tokens);
                if (!slots.ok()) return slots;
            }
            auto sequence_status = configure_sequence_admission();
            if (!sequence_status.ok()) return sequence_status;
        }
        if (config.verbose) {
            const auto stats = weights->stats();
            for (std::size_t slot = 0U; slot < devices.size(); ++slot) {
                std::cerr << "[glm53-load] cuda=" << devices[slot]
                          << " resident_linears=" << admitted[slot]
                          << " streamed_linears=" << skipped[slot]
                          << " pinned_bytes=" << stats.pinned[slot]
                          << " cache_capacity_bytes=" << stats.capacity[slot]
                          << '\n';
            }
        }
        if (config.phase_profile) {
            const auto stats = weights->stats();
            for (std::size_t slot = 0U; slot < devices.size(); ++slot) {
                std::size_t moe_layers = 0U;
                for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
                    if (glm53_moe_layer(layer) && slot_for(layer) == slot) {
                        ++moe_layers;
                    }
                }
                std::cerr << "[glm53-capacity] cuda=" << devices[slot]
                          << " moe_layers=" << moe_layers
                          << " admitted_budget_bytes=" << device_budgets[slot]
                          << " workspace_reserve_bytes="
                          << kDeviceWorkspaceReserve
                          << " resident_reserve_bytes="
                          << resident_reserve_bytes[slot]
                          << " shared_expert_bytes="
                          << shared_experts.bytes_by_slot[slot]
                          << " static_expert_bytes="
                          << static_experts.bytes_by_slot[slot]
                          << " arena_capacity_bytes=" << stats.capacity[slot]
                          << " arena_used_bytes=" << stats.used[slot]
                          << " arena_pinned_bytes=" << stats.pinned[slot]
                          << " arena_free_bytes="
                          << (stats.capacity[slot] - stats.used[slot]) << '\n';
            }
        }
        return result;
    }

    [[nodiscard]] ValidationResult reset_sequence(
        Glm53SequenceState& sequence) const {
        return sequence.reset(config.maximum_context_tokens, 64U);
    }

    [[nodiscard]] ValidationResult linear(
        std::string_view base, std::span<const float> input,
        std::uint32_t rows, std::uint32_t columns,
        std::span<float> output, std::uint32_t layer,
        bool bf16_output = true) {
        ValidationResult result;
        if (input.size() != static_cast<std::size_t>(columns) * rows ||
            output.empty()) {
            result.errors.push_back("GLM-5.3 linear activation shape is invalid for " +
                                    std::string(base));
            return result;
        }
        const auto output_columns = output.size() / rows;
        if (output_columns * rows != output.size()) {
            result.errors.push_back("GLM-5.3 linear output shape is invalid for " +
                                    std::string(base));
            return result;
        }
        // PyTorch returns every published BF16/FP8 linear at the model's BF16
        // activation dtype. Keep that boundary even though the host-facing
        // CUDA API transports activations as float.
        const Glm53WeightCache::LinearRequest request{
            base, output_columns, columns, input, rows, output, bf16_output};
        return linear_batch(
            std::span<const Glm53WeightCache::LinearRequest>(&request, 1U),
            layer);
    }

    [[nodiscard]] ValidationResult linear_batch(
        std::span<const Glm53WeightCache::LinearRequest> requests,
        std::uint32_t layer) {
        if (requests.empty()) {
            return {{"GLM-5.3 linear projection batch is empty"}};
        }
        for (const auto& request : requests) {
            if (request.rows == 0U ||
                request.input.size() !=
                    static_cast<std::size_t>(request.input_columns) *
                        request.rows ||
                request.output.size() !=
                    static_cast<std::size_t>(request.output_columns) *
                        request.rows) {
                return {{"GLM-5.3 linear projection batch has an invalid shape"}};
            }
        }
        if (full_tensor_parallel_active) {
            std::vector<std::vector<Glm53RowRange>> ranges(requests.size());
            bool eligible = true;
            for (std::size_t index = 0U; index < requests.size(); ++index) {
                const auto& request = requests[index];
                ranges[index] = weighted_row_ranges(
                    request.output_columns, weight_capacities, 128U);
                if (request.weight_rows != 0U ||
                    ranges[index].size() != devices.size()) {
                    eligible = false;
                    break;
                }
                for (const auto range : ranges[index]) {
                    if (range.count == 0U || range.begin % 128U != 0U ||
                        checkpoint->cuda_linear_slice_storage_bytes(
                            request.base, range.begin, range.count) == 0U) {
                        eligible = false;
                        break;
                    }
                }
                if (!eligible) break;
            }
            if (eligible) {
                std::vector<std::vector<std::vector<float>>> shards(
                    requests.size(),
                    std::vector<std::vector<float>>(devices.size()));
                std::vector<std::vector<Glm53WeightCache::LinearRequest>>
                    groups(devices.size());
                for (std::size_t index = 0U; index < requests.size(); ++index) {
                    for (std::size_t slot = 0U; slot < devices.size(); ++slot) {
                        const auto range = ranges[index][slot];
                        shards[index][slot].resize(
                            static_cast<std::size_t>(requests[index].rows) *
                            range.count);
                        groups[slot].push_back({
                            requests[index].base, range.count,
                            requests[index].input_columns,
                            requests[index].input, requests[index].rows,
                            shards[index][slot], requests[index].bf16_output,
                            requests[index].output_columns, range.begin});
                    }
                }
                std::vector<ValidationResult> device_results(devices.size());
                auto dispatched = projection_workers->parallel_for_addressed(
                    devices.size(), [&](std::size_t slot) {
                        device_results[slot] =
                            weights->matmul_batch(slot, groups[slot]);
                    });
                if (!dispatched.ok()) return dispatched;
                for (auto& device_result : device_results) {
                    if (!device_result.ok()) return device_result;
                }
                for (std::size_t index = 0U; index < requests.size(); ++index) {
                    for (std::size_t slot = 0U; slot < devices.size(); ++slot) {
                        const auto range = ranges[index][slot];
                        for (std::uint32_t row = 0U;
                             row < requests[index].rows; ++row) {
                            std::copy_n(
                                shards[index][slot].begin() +
                                    static_cast<std::ptrdiff_t>(
                                        static_cast<std::size_t>(row) *
                                        range.count),
                                range.count,
                                requests[index].output.begin() +
                                    static_cast<std::ptrdiff_t>(
                                        static_cast<std::size_t>(row) *
                                            requests[index].output_columns +
                                        range.begin));
                        }
                    }
                }
                parallel_projection_batches.fetch_add(
                    1U, std::memory_order_relaxed);
                parallel_projection_requests.fetch_add(
                    requests.size(), std::memory_order_relaxed);
                return {};
            }
        }
        if (!batched_projections_enabled()) {
            for (const auto& request : requests) {
                auto projected = weights->matmul(
                    slot_for(layer), request.base, request.output_columns,
                    request.input_columns, request.input, request.rows,
                    request.output, request.bf16_output);
                if (!projected.ok()) return projected;
            }
            return {};
        }
        if (cross_gpu_projections_enabled(devices) &&
            projection_workers != nullptr &&
            devices.size() > 1U && requests.size() > 1U) {
            std::vector<std::string_view> keys;
            std::vector<std::uint64_t> costs;
            keys.reserve(requests.size());
            costs.reserve(requests.size());
            for (const auto& request : requests) {
                keys.push_back(request.base);
                costs.push_back(
                    checkpoint->cuda_linear_storage_bytes(request.base));
            }
            const auto slots = glm53_projection_slots(
                keys, costs, weight_capacities, slot_for(layer));
            if (slots.size() != requests.size()) {
                return {{"GLM-5.3 parallel projection assignment is invalid"}};
            }
            std::vector<std::vector<Glm53WeightCache::LinearRequest>> groups(
                devices.size());
            for (std::size_t index = 0U; index < requests.size(); ++index) {
                groups[slots[index]].push_back(requests[index]);
            }
            std::vector<ValidationResult> device_results(devices.size());
            auto dispatched = projection_workers->parallel_for_addressed(
                devices.size(), [&](std::size_t slot) {
                    if (!groups[slot].empty()) {
                        device_results[slot] =
                            weights->matmul_batch(slot, groups[slot]);
                    }
                });
            if (!dispatched.ok()) return dispatched;
            ValidationResult joined;
            std::uint64_t active_slots = 0U;
            for (std::size_t slot = 0U; slot < device_results.size(); ++slot) {
                if (!groups[slot].empty()) ++active_slots;
                append(joined.errors, std::move(device_results[slot].errors));
            }
            if (joined.ok() && active_slots > 1U) {
                parallel_projection_batches.fetch_add(
                    1U, std::memory_order_relaxed);
                parallel_projection_requests.fetch_add(
                    static_cast<std::uint64_t>(requests.size()),
                    std::memory_order_relaxed);
            }
            return joined;
        }
        return weights->matmul_batch(slot_for(layer), requests);
    }

    [[nodiscard]] ValidationResult norm(
        std::span<float> output, std::span<const float> input,
        std::string_view weight_name) {
        auto weight = host_tensor(weight_name, input.size());
        if (!weight.ok()) return {std::move(weight.errors)};
        auto result = kimi_rms_norm(output, input, *weight.value, 1.0e-5F);
        if (result.ok()) round_bf16(output);
        return result;
    }

    [[nodiscard]] ValidationResult norm_rows(
        std::span<float> output, std::span<const float> input,
        std::uint32_t rows, std::uint32_t columns,
        std::string_view weight_name) {
        ValidationResult result;
        if (rows == 0U || input.size() != output.size() ||
            input.size() != static_cast<std::size_t>(rows) * columns) {
            result.errors.emplace_back("GLM-5.3 RMSNorm page shape is invalid");
            return result;
        }
        auto weight = host_tensor(weight_name, columns);
        if (!weight.ok()) return {std::move(weight.errors)};
        for (std::uint32_t row = 0U; row < rows; ++row) {
            const auto begin = static_cast<std::size_t>(row) * columns;
            auto normalized = kimi_rms_norm(
                output.subspan(begin, columns), input.subspan(begin, columns),
                *weight.value, 1.0e-5F);
            if (!normalized.ok()) return normalized;
            round_bf16(output.subspan(begin, columns));
        }
        return result;
    }

    [[nodiscard]] ValidationResult mhc_pre(
        std::span<float> collapsed, Dsv4MhcMix& mix,
        std::span<const float> streams, const std::string& prefix) {
        ValidationResult result;
        auto projection = host_tensor(prefix + "_fn", 24U * 16384U);
        auto base = host_tensor(prefix + "_base", 24U);
        auto scale = host_tensor(prefix + "_scale", 3U);
        if (!projection.ok() || !base.ok() || !scale.ok()) {
            append(result.errors, std::move(projection.errors));
            append(result.errors, std::move(base.errors));
            append(result.errors, std::move(scale.errors));
            return result;
        }
        double square_sum = 0.0;
        for (const auto value : streams) square_sum += static_cast<double>(value) * value;
        const auto reciprocal = 1.0F / std::sqrt(
            static_cast<float>(square_sum /
                               static_cast<double>(streams.size())) + 1.0e-5F);
        std::vector<float> projected(24U, 0.0F);
        for (std::size_t row = 0U; row < projected.size(); ++row) {
            double sum = 0.0;
            for (std::size_t column = 0U; column < streams.size(); ++column) {
                sum += static_cast<double>((*projection.value)[row * streams.size() + column]) *
                       streams[column];
            }
            projected[row] = static_cast<float>(sum) * reciprocal;
        }
        auto split = dsv4_mhc_split_sinkhorn_f32(
            projected, *scale.value, *base.value, kMhc, 20U, 1.0e-6F);
        if (!split.ok()) return {std::move(split.errors)};
        mix = std::move(split.value);
        round_bf16(mix.post);
        round_bf16(mix.combination);
        std::fill(collapsed.begin(), collapsed.end(), 0.0F);
        for (std::size_t stream = 0U; stream < kMhc; ++stream) {
            for (std::size_t column = 0U; column < kHidden; ++column) {
                collapsed[column] += mix.pre[stream] *
                    streams[stream * kHidden + column];
            }
        }
        round_bf16(collapsed);
        return result;
    }

    [[nodiscard]] ValidationResult attention_kda(
        std::span<float> output, std::span<const float> input,
        std::uint32_t layer, const std::string& attention,
        Glm53SequenceState& sequence, CudaBuffer* device_state = nullptr) {
        ValidationResult result;
        if (device_state != nullptr && fused_kda_enabled()) {
            CudaGlm53KdaRequest request;
            request.state = device_state;
            request.input = input;
            request.heads = kHeads;
            request.head_dim = kLinearHead;
            request.convolution_kernel = 4U;
            return weights->kda_decode(
                slot_for(layer), attention, request, output);
        }
        std::vector<float> query(kLinearWidth), key(kLinearWidth),
            value(kLinearWidth), low(kLinearHead), beta(kHeads),
            gate_low(kLinearHead);
        const std::array<std::string, 6U> first_bases{
            attention + "q_proj", attention + "k_proj", attention + "v_proj",
            attention + "f_a_proj", attention + "b_proj",
            attention + "g_a_proj"};
        const std::array<Glm53WeightCache::LinearRequest, 6U> first{
            {{first_bases[0], kLinearWidth, kHidden, input, 1U, query, true},
             {first_bases[1], kLinearWidth, kHidden, input, 1U, key, true},
             {first_bases[2], kLinearWidth, kHidden, input, 1U, value, true},
             {first_bases[3], kLinearHead, kHidden, input, 1U, low, true},
             {first_bases[4], kHeads, kHidden, input, 1U, beta, true},
             {first_bases[5], kLinearHead, kHidden, input, 1U, gate_low, true}}};
        result = linear_batch(first, layer);
        if (!result.ok()) return result;
        if (device_state == nullptr) {
          for (std::uint32_t projection = 0U; projection < 3U; ++projection) {
            auto taps = host_tensor(
                attention + (projection == 0U ? "q_conv1d.weight"
                              : projection == 1U ? "k_conv1d.weight"
                                                 : "v_conv1d.weight"),
                static_cast<std::uint64_t>(kLinearWidth) * 4U);
            if (!taps.ok()) return {std::move(taps.errors)};
            auto& values = projection == 0U ? query : projection == 1U ? key : value;
            auto convolved = values;
            result = kimi_short_conv_step(
                convolved, values, *taps.value,
                sequence.convolution(layer, projection), 4U);
            if (!result.ok()) return result;
            values = std::move(convolved);
            round_bf16(values);
          }
        }
        std::vector<float> forget(kLinearWidth), gate(kLinearWidth);
        const std::array<std::string, 2U> second_bases{
            attention + "f_b_proj", attention + "g_b_proj"};
        const std::array<Glm53WeightCache::LinearRequest, 2U> second{
            {{second_bases[0], kLinearWidth, kLinearHead, low, 1U, forget, true},
             {second_bases[1], kLinearWidth, kLinearHead, gate_low, 1U, gate,
              true}}};
        result = linear_batch(second, layer);
        if (!result.ok()) return result;
        for (auto& element : beta) {
            element = bf16_round_f32(sigmoid(element));
        }
        auto a_log = host_tensor(attention + "A_log", kHeads);
        auto dt_bias = host_tensor(attention + "dt_bias", kLinearWidth);
        auto o_norm = host_tensor(attention + "o_norm.weight", kLinearHead);
        if (!a_log.ok() || !dt_bias.ok() || !o_norm.ok()) {
            append(result.errors, std::move(a_log.errors));
            append(result.errors, std::move(dt_bias.errors));
            append(result.errors, std::move(o_norm.errors));
            return result;
        }
        if (device_state != nullptr) {
            CudaGlm53KdaRequest request;
            request.state = device_state;
            request.query = query;
            request.key = key;
            request.value = value;
            request.forget = forget;
            request.beta = beta;
            request.gate = gate;
            request.heads = kHeads;
            request.head_dim = kLinearHead;
            request.convolution_kernel = 4U;
            return weights->kda_decode(slot_for(layer), attention, request,
                                       output);
        }
        std::vector<float> heads_out(kLinearWidth);
        const auto query_scale = 1.0F / std::sqrt(static_cast<float>(kLinearHead));
        for (std::uint32_t head = 0U; head < kHeads; ++head) {
            const auto begin = static_cast<std::size_t>(head) * kLinearHead;
            auto q = std::span<float>(query).subspan(begin, kLinearHead);
            auto k = std::span<float>(key).subspan(begin, kLinearHead);
            result = kimi_l2_normalize(q, 1.0e-6F);
            if (!result.ok()) return result;
            result = kimi_l2_normalize(k, 1.0e-6F);
            if (!result.ok()) return result;
            for (auto& element : q) element *= query_scale;
            std::vector<float> decay(kLinearHead);
            result = kimi_kda_log_decay(
                decay, std::span<const float>(forget).subspan(begin, kLinearHead),
                std::span<const float>(*dt_bias.value).subspan(begin, kLinearHead),
                (*a_log.value)[head], -5.0F);
            if (!result.ok()) return result;
            for (auto& element : decay) element = std::exp(element);
            std::vector<float> raw(kLinearHead);
            auto state = sequence.recurrent(layer).subspan(
                static_cast<std::size_t>(head) * kLinearHead * kLinearHead,
                static_cast<std::size_t>(kLinearHead) * kLinearHead);
            result = kimi_kda_step(
                raw, state, q, k,
                std::span<const float>(value).subspan(begin, kLinearHead),
                decay, beta[head], kLinearHead, kLinearHead);
            if (!result.ok()) return result;
            round_bf16(raw);
            result = kimi_kda_output_norm(
                std::span<float>(heads_out).subspan(begin, kLinearHead), raw,
                std::span<const float>(gate).subspan(begin, kLinearHead),
                *o_norm.value, 1.0e-5F);
            if (!result.ok()) return result;
            round_bf16(std::span<float>(heads_out).subspan(begin, kLinearHead));
        }
        return linear(attention + "o_proj", heads_out, 1U, kLinearWidth,
                      output, layer);
    }

    [[nodiscard]] ValidationResult attention_kda_page(
        std::span<float> output, std::span<const float> input,
        std::uint32_t rows, std::uint32_t layer,
        const std::string& attention, Glm53SequenceState& sequence) {
        ValidationResult result;
        const auto wide_elements = static_cast<std::size_t>(rows) * kLinearWidth;
        std::vector<float> query(wide_elements), key(wide_elements),
            value(wide_elements),
            low(static_cast<std::size_t>(rows) * kLinearHead),
            beta(static_cast<std::size_t>(rows) * kHeads),
            gate_low(static_cast<std::size_t>(rows) * kLinearHead);
        const std::array<std::string, 6U> first_bases{
            attention + "q_proj", attention + "k_proj", attention + "v_proj",
            attention + "f_a_proj", attention + "b_proj",
            attention + "g_a_proj"};
        const std::array<Glm53WeightCache::LinearRequest, 6U> first{
            {{first_bases[0], kLinearWidth, kHidden, input, rows, query, true},
             {first_bases[1], kLinearWidth, kHidden, input, rows, key, true},
             {first_bases[2], kLinearWidth, kHidden, input, rows, value, true},
             {first_bases[3], kLinearHead, kHidden, input, rows, low, true},
             {first_bases[4], kHeads, kHidden, input, rows, beta, true},
             {first_bases[5], kLinearHead, kHidden, input, rows, gate_low, true}}};
        result = linear_batch(first, layer);
        if (!result.ok()) return result;
        for (std::uint32_t projection = 0U; projection < 3U; ++projection) {
            auto taps = host_tensor(
                attention + (projection == 0U ? "q_conv1d.weight"
                              : projection == 1U ? "k_conv1d.weight"
                                                 : "v_conv1d.weight"),
                static_cast<std::uint64_t>(kLinearWidth) * 4U);
            if (!taps.ok()) return {std::move(taps.errors)};
            auto& values = projection == 0U ? query : projection == 1U ? key : value;
            for (std::uint32_t row = 0U; row < rows; ++row) {
                const auto begin = static_cast<std::size_t>(row) * kLinearWidth;
                std::vector<float> convolved(kLinearWidth);
                result = kimi_short_conv_step(
                    convolved,
                    std::span<const float>(values).subspan(begin, kLinearWidth),
                    *taps.value, sequence.convolution(layer, projection), 4U);
                if (!result.ok()) return result;
                round_bf16(convolved);
                std::copy(convolved.begin(), convolved.end(),
                          values.begin() + static_cast<std::ptrdiff_t>(begin));
            }
        }
        std::vector<float> forget(wide_elements);
        std::vector<float> gate(wide_elements);
        const std::array<std::string, 2U> second_bases{
            attention + "f_b_proj", attention + "g_b_proj"};
        const std::array<Glm53WeightCache::LinearRequest, 2U> second{
            {{second_bases[0], kLinearWidth, kLinearHead, low, rows, forget, true},
             {second_bases[1], kLinearWidth, kLinearHead, gate_low, rows, gate,
              true}}};
        result = linear_batch(second, layer);
        if (!result.ok()) return result;
        for (auto& element : beta) element = bf16_round_f32(sigmoid(element));
        auto a_log = host_tensor(attention + "A_log", kHeads);
        auto dt_bias = host_tensor(attention + "dt_bias", kLinearWidth);
        auto o_norm = host_tensor(attention + "o_norm.weight", kLinearHead);
        if (!a_log.ok() || !dt_bias.ok() || !o_norm.ok()) {
            append(result.errors, std::move(a_log.errors));
            append(result.errors, std::move(dt_bias.errors));
            append(result.errors, std::move(o_norm.errors));
            return result;
        }
        std::vector<float> heads_out(wide_elements);
        const auto query_scale = 1.0F / std::sqrt(static_cast<float>(kLinearHead));
        // Resolve the copy-on-write buffer before dispatch. Calling recurrent()
        // for the first time from every head worker races the shared_ptr's lazy
        // allocation; workers below own disjoint head matrices once this span
        // is stable.
        auto recurrent = sequence.recurrent(layer);
        if (recurrent.size() != static_cast<std::size_t>(kHeads) *
                                    kLinearHead * kLinearHead) {
            return {{"GLM-5.3 KDA recurrent state has an invalid shape"}};
        }
        // A prompt page is a scheduling boundary, not a numerical one. Replay
        // the declared per-token recurrence for every page width and expose
        // only independent heads to the worker pool. The former chunk path
        // changed its FP32 association with both page size and host core count.
        const auto replay_head = [&](std::size_t head) -> ValidationResult {
            std::vector<float> decay(kLinearHead), raw(kLinearHead);
            auto state = recurrent.subspan(
                head * kLinearHead * kLinearHead,
                static_cast<std::size_t>(kLinearHead) * kLinearHead);
            for (std::uint32_t row = 0U; row < rows; ++row) {
                const auto begin = static_cast<std::size_t>(row) *
                                       kLinearWidth +
                                   head * kLinearHead;
                auto q = std::span<float>(query).subspan(begin, kLinearHead);
                auto k = std::span<float>(key).subspan(begin, kLinearHead);
                auto status = kimi_l2_normalize(q, 1.0e-6F);
                if (!status.ok()) return status;
                status = kimi_l2_normalize(k, 1.0e-6F);
                if (!status.ok()) return status;
                for (auto& element : q) element *= query_scale;
                status = kimi_kda_log_decay(
                    decay,
                    std::span<const float>(forget).subspan(begin, kLinearHead),
                    std::span<const float>(*dt_bias.value).subspan(
                        head * kLinearHead, kLinearHead),
                    (*a_log.value)[head], -5.0F);
                if (!status.ok()) return status;
                for (auto& element : decay) element = std::exp(element);
                status = kimi_kda_step(
                    raw, state, q, k,
                    std::span<const float>(value).subspan(begin, kLinearHead),
                    decay,
                    beta[static_cast<std::size_t>(row) * kHeads + head],
                    kLinearHead, kLinearHead);
                if (!status.ok()) return status;
                round_bf16(raw);
                status = kimi_kda_output_norm(
                    std::span<float>(heads_out).subspan(begin, kLinearHead),
                    raw,
                    std::span<const float>(gate).subspan(begin, kLinearHead),
                    *o_norm.value, 1.0e-5F);
                if (!status.ok()) return status;
                round_bf16(std::span<float>(heads_out).subspan(
                    begin, kLinearHead));
            }
            return {};
        };
        if (kda_workers != nullptr) {
            std::vector<ValidationResult> failures(kHeads);
            auto replayed = kda_workers->parallel_for(
                kHeads, [&](std::size_t head) {
                    failures[head] = replay_head(head);
                });
            if (!replayed.ok()) return replayed;
            for (auto& failure : failures) {
                if (!failure.ok()) return failure;
            }
        } else {
            for (std::size_t head = 0U; head < kHeads; ++head) {
                result = replay_head(head);
                if (!result.ok()) return result;
            }
        }
        return linear(attention + "o_proj", heads_out, rows, kLinearWidth,
                      output, layer);
    }

    [[nodiscard]] ValidationResult attention_mla(
        std::span<float> output, std::span<const float> input,
        std::uint32_t layer, std::uint32_t position,
        const std::string& attention, Glm53SequenceState& sequence) {
        ValidationResult result;
        std::vector<float> q_rank(kQueryRank), query(kMlaWidth), latent(kKvRank);
        const std::array<std::string, 2U> first_bases{
            attention + "q_a_proj", attention + "kv_a_proj_with_mqa"};
        const std::array<Glm53WeightCache::LinearRequest, 2U> first{
            {{first_bases[0], kQueryRank, kHidden, input, 1U, q_rank, true},
             {first_bases[1], kKvRank, kHidden, input, 1U, latent, true}}};
        result = linear_batch(first, layer);
        if (!result.ok()) return result;
        result = norm(q_rank, q_rank, attention + "q_a_layernorm.weight");
        if (!result.ok()) return result;
        result = norm(latent, latent, attention + "kv_a_layernorm.weight");
        if (!result.ok()) return result;
        auto& cache = sequence.mla(layer);
        if (cache.rows() != position) {
            return {{"GLM-5.3 physical MLA position is not contiguous"}};
        }
        result = cache.append(latent);
        if (!result.ok()) return result;
        const auto history = position + 1U;

        // k-pool sparse indexer (record 0237). Selects which history positions
        // this query attends to, so attention stops growing with context. At
        // history <= kIndexTopK the selection is the whole history, so this is
        // bit-identical to the dense path there -- which is the regression gate
        // for the sparse path above it.
        std::vector<std::uint32_t> selected(kIndexSelectionWidth);
        std::size_t selected_count = history;
        auto& index_cache = sequence.indexer(layer);
        // A sequence whose whole context fits under the threshold can never
        // reach a query where the selection is anything but the identity, so
        // it pays nothing for the indexer at all.
        const bool sparse_reachable =
            sparse_indexer_active(sequence.maximum_context_tokens());
        if (sparse_reachable) {
            std::vector<float> index_key(kIndexHeadDim);
            std::vector<float> index_gate(kIndexHeadDim);
            std::vector<float> index_query(
                static_cast<std::size_t>(kIndexHeads) * kIndexHeadDim);
            std::vector<float> head_weights(kIndexHeads);
            // The key and gate must be recorded for every token, because a
            // later query beyond the threshold will pool them. The query-side
            // projections are only needed when the selection is not the
            // identity, and `wq_b` alone is six times the cost of both cache
            // projections, so they are deferred.
            // LinearRequest::base is a string_view, so these names must
            // outlive the batch. Temporaries here dangle and surface later as a
            // corrupted tensor name.
            const std::array<std::string, 4U> indexer_bases{
                attention + "indexer.wk.weight",
                attention + "indexer.index_kpool_compress_gate",
                attention + "indexer.wq_b.weight",
                attention + "indexer.weights_proj.weight"};
            // The indexer's own weights stay on the host. They are about
            // 152 MB across the eleven sparse layers, they are only touched
            // above kIndexTopK, and demanding them into the CUDA cache
            // competes with the pinned spine and the static expert tier for
            // VRAM that is already fully committed.
            {
                auto wk = host_tensor(indexer_bases[0],
                                      static_cast<std::uint64_t>(kIndexHeadDim) *
                                          kHidden);
                if (!wk.ok()) return {std::move(wk.errors)};
                glm53_indexer_gate(index_key, input, *wk.value);
            }
            // `index_kpool_compress_gate` is a bare nn.Parameter, not a Linear
            // module, so it has no ".weight" suffix and cannot be keyed into
            // the CUDA linear cache. Applied directly instead.
            {
                auto gate = host_tensor(indexer_bases[1],
                                        static_cast<std::uint64_t>(kIndexHeadDim) *
                                            kHidden);
                if (!gate.ok()) return {std::move(gate.errors)};
                glm53_indexer_gate(index_gate, input, *gate.value);
            }
            if (history > kIndexTopK) {
                auto wq = host_tensor(
                    indexer_bases[2],
                    static_cast<std::uint64_t>(kIndexHeads) * kIndexHeadDim *
                        kQueryRank);
                if (!wq.ok()) return {std::move(wq.errors)};
                glm53_indexer_gate(index_query, q_rank, *wq.value);
                auto wp = host_tensor(
                    indexer_bases[3],
                    static_cast<std::uint64_t>(kIndexHeads) * kHidden);
                if (!wp.ok()) return {std::move(wp.errors)};
                glm53_indexer_gate(head_weights, input, *wp.value);
            }
            auto norm_weight =
                host_tensor(attention + "indexer.k_norm.weight", kIndexHeadDim);
            if (!norm_weight.ok()) return {std::move(norm_weight.errors)};
            auto norm_bias =
                host_tensor(attention + "indexer.k_norm.bias", kIndexHeadDim);
            if (!norm_bias.ok()) return {std::move(norm_bias.errors)};
            glm53_indexer_layer_norm(index_key, *norm_weight.value,
                                     *norm_bias.value);

            std::vector<float> packed(2U * kIndexHeadDim);
            std::copy(index_key.begin(), index_key.end(), packed.begin());
            std::copy(index_gate.begin(), index_gate.end(),
                      packed.begin() + kIndexHeadDim);
            if (index_cache.rows() != position) {
                return {{"GLM-5.3 indexer position is not contiguous"}};
            }
            result = index_cache.append(packed);
            if (!result.ok()) return result;

            result = complete_index_pools(sequence, layer, attention);
            if (!result.ok()) return result;
            if (history > kIndexTopK) {
                // Read the cached pool keys in place. Materializing the indexer
                // history here would copy `history x 256 x 4 B` per layer per
                // token, which grows with context exactly as the dense path did.
                const auto& pool_cache = sequence.index_pool(layer);
                selected_count = glm53_sparse_index_select(
                    selected, index_query,
                    [&](std::uint32_t pool) {
                        return pool_cache.row(pool).data();
                    },
                    head_weights, history);
            }
        }
        if (history <= kIndexTopK) {
            for (std::uint32_t token = 0U; token < history; ++token) {
                selected[token] = token;
            }
            selected_count = history;
        }

        // Expand only the selected latents. This is what bounds both the MLA
        // workspace and the decode cost at any context.
        const auto attended_rows = static_cast<std::uint32_t>(selected_count);
        static_cast<void>(glm53_grow(
            mla_expanded_scratch,
            static_cast<std::size_t>(attended_rows) * kHeads * 2U * kMlaHead));
        const auto expanded = std::span<float>(mla_expanded_scratch)
            .first(static_cast<std::size_t>(attended_rows) * kHeads * 2U *
                   kMlaHead);
        const std::array<std::string, 2U> second_bases{
            attention + "q_b_proj", attention + "kv_b_proj"};
        // Gather straight out of the page table. The selection is bounded by
        // kIndexSelectionWidth, so this copy is bounded too; materializing the
        // whole latent history first would not be.
        static_cast<void>(glm53_grow(
            mla_gathered_scratch,
            static_cast<std::size_t>(attended_rows) * kKvRank));
        const auto gathered = std::span<float>(mla_gathered_scratch)
            .first(static_cast<std::size_t>(attended_rows) * kKvRank);
        for (std::uint32_t row = 0U; row < attended_rows; ++row) {
            const auto source = cache.row(selected[row]);
            std::copy(source.begin(), source.end(),
                      gathered.begin() +
                          static_cast<std::ptrdiff_t>(
                              static_cast<std::size_t>(row) * kKvRank));
        }
        const auto latent_history = std::span<const float>(gathered);
        const std::array<Glm53WeightCache::LinearRequest, 2U> second{
            {{second_bases[0], kMlaWidth, kQueryRank, q_rank, 1U, query, true},
             {second_bases[1], kHeads * 2U * kMlaHead, kKvRank, latent_history,
              attended_rows, expanded, true}}};
        result = linear_batch(second, layer);
        if (!result.ok()) return result;
        std::vector<float> attended(kMlaWidth, 0.0F);
        const auto score_scale = 1.0F / std::sqrt(static_cast<float>(kMlaHead));
        // The 64 heads are independent: each reads the shared expansion and
        // writes its own slice of `attended`, and the accumulation order within
        // a head is untouched, so spreading them across the pool is exact. This
        // loop is 93 ms per layer at a 2,051-position selection -- 1.02 s per
        // token across the eleven sparse layers -- and it was serial and scalar.
        static_cast<void>(glm53_grow(
            mla_head_score_scratch,
            static_cast<std::size_t>(kHeads) * attended_rows));
        auto* const head_scores = mla_head_score_scratch.data();
        const auto attend_head = [&](std::size_t head) {
            auto* scores = head_scores +
                           head * static_cast<std::size_t>(attended_rows);
            const auto* q = query.data() + head * kMlaHead;
            float highest = -std::numeric_limits<float>::infinity();
            for (std::uint32_t token = 0U; token < attended_rows; ++token) {
                const auto* kv = expanded.data() +
                    (static_cast<std::size_t>(token) * kHeads + head) *
                        (2U * kMlaHead);
                float score = 0.0F;
                for (std::uint32_t column = 0U; column < kMlaHead; ++column) {
                    score += q[column] * kv[column];
                }
                scores[token] = score * score_scale;
                highest = std::max(highest, scores[token]);
            }
            float total = 0.0F;
            for (std::uint32_t token = 0U; token < attended_rows; ++token) {
                scores[token] = std::exp(scores[token] - highest);
                total += scores[token];
            }
            auto* destination = attended.data() + head * kMlaHead;
            for (std::uint32_t token = 0U; token < attended_rows; ++token) {
                const auto* values = expanded.data() +
                    (static_cast<std::size_t>(token) * kHeads + head) *
                        (2U * kMlaHead) + kMlaHead;
                const auto coefficient = bf16_round_f32(scores[token] / total);
                for (std::uint32_t column = 0U; column < kMlaHead; ++column) {
                    destination[column] += coefficient * values[column];
                }
            }
        };
        if (kda_workers != nullptr) {
            auto attended_all = kda_workers->parallel_for(kHeads, attend_head);
            if (!attended_all.ok()) return attended_all;
        } else {
            for (std::size_t head = 0U; head < kHeads; ++head) attend_head(head);
        }
        round_bf16(attended);
        return linear(attention + "o_proj", attended, 1U, kMlaWidth,
                      output, layer);
    }

    // Bring this layer's pool-key store up to date with its indexer store.
    // Every group of `kIndexPool` consecutive positions that has just become
    // complete gets its key computed once, here, and never again.
    [[nodiscard]] ValidationResult complete_index_pools(
        Glm53SequenceState& sequence, std::uint32_t layer,
        const std::string& attention) {
        const auto& index_cache = sequence.indexer(layer);
        auto& pool_cache = sequence.index_pool(layer);
        const auto complete = index_cache.rows() / kIndexPool;
        if (pool_cache.rows() >= complete) return {};
        auto ape = host_tensor(attention + "indexer.index_kpool_compress_ape",
                               kIndexPool * kIndexHeadDim);
        if (!ape.ok()) return {std::move(ape.errors)};
        std::vector<float> pool_key(kIndexHeadDim);
        for (auto pool = pool_cache.rows(); pool < complete; ++pool) {
            glm53_index_pool_key(
                pool_key, pool,
                [&](std::uint32_t token) {
                    return index_cache.row(token).data();
                },
                [&](std::uint32_t token) {
                    return index_cache.row(token).data() + kIndexHeadDim;
                },
                *ape.value);
            auto appended = pool_cache.append(pool_key);
            if (!appended.ok()) return appended;
        }
        return {};
    }

    // Maintain the exact host indexer while sparse MLA itself remains in the
    // resident CUDA chain. Selection above index_topk is delegated to the
    // already-oracled host implementation and only its ascending positions
    // cross to the device. Below the threshold this records future pool state
    // but returns an empty selection, leaving the independent identity MLA
    // path untouched.
    [[nodiscard]] ValidationResult prepare_sparse_device_selection(
        std::vector<std::uint32_t>& selected,
        std::vector<std::uint32_t>& arena_rows,
        std::vector<std::uint32_t>& expansion_sources,
        std::vector<std::uint32_t>& expansion_destinations,
        std::span<const float> input, std::uint32_t layer,
        std::uint32_t position, const std::string& attention,
        Glm53SequenceState& sequence, DeviceSequenceState& device_sequence,
        Glm53SparseMlaMetrics* timing) {
        selected.clear();
        arena_rows.clear();
        expansion_sources.clear();
        expansion_destinations.clear();
        const auto history = position + 1U;
        std::vector<float> index_key(kIndexHeadDim);
        std::vector<float> index_gate(kIndexHeadDim);
        const std::array<std::string, 4U> indexer_bases{
            attention + "indexer.wk.weight",
            attention + "indexer.index_kpool_compress_gate",
            attention + "indexer.wq_b.weight",
            attention + "indexer.weights_proj.weight"};
        const auto indexer_projection_started = timing != nullptr
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        {
            auto wk = host_tensor(
                indexer_bases[0],
                static_cast<std::uint64_t>(kIndexHeadDim) * kHidden);
            if (!wk.ok()) return {std::move(wk.errors)};
            glm53_indexer_gate(index_key, input, *wk.value);
        }
        {
            auto gate = host_tensor(
                indexer_bases[1],
                static_cast<std::uint64_t>(kIndexHeadDim) * kHidden);
            if (!gate.ok()) return {std::move(gate.errors)};
            glm53_indexer_gate(index_gate, input, *gate.value);
        }
        if (timing != nullptr) {
            timing->indexer_projection_nanoseconds +=
                elapsed_nanoseconds(indexer_projection_started);
        }
        const auto indexer_state_started = timing != nullptr
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        auto norm_weight =
            host_tensor(attention + "indexer.k_norm.weight", kIndexHeadDim);
        if (!norm_weight.ok()) return {std::move(norm_weight.errors)};
        auto norm_bias =
            host_tensor(attention + "indexer.k_norm.bias", kIndexHeadDim);
        if (!norm_bias.ok()) return {std::move(norm_bias.errors)};
        glm53_indexer_layer_norm(index_key, *norm_weight.value,
                                 *norm_bias.value);

        auto& index_cache = sequence.indexer(layer);
        if (index_cache.rows() != position) {
            return {{"GLM-5.3 resident indexer position is not contiguous"}};
        }
        std::vector<float> packed(2U * kIndexHeadDim);
        std::copy(index_key.begin(), index_key.end(), packed.begin());
        std::copy(index_gate.begin(), index_gate.end(),
                  packed.begin() + kIndexHeadDim);
        auto appended = index_cache.append(packed);
        if (!appended.ok()) return appended;
        auto completed = complete_index_pools(sequence, layer, attention);
        if (!completed.ok()) return completed;
        if (timing != nullptr) {
            timing->indexer_state_nanoseconds +=
                elapsed_nanoseconds(indexer_state_started);
        }
        auto& arena = device_sequence.sparse_mla_arenas[layer];
        if (history <= kIndexTopK) {
            arena.identity_rows = history;
            return {};
        }

        std::vector<float> q_rank(kQueryRank);
        const auto query_rank_started = timing != nullptr
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        auto projected = linear(attention + "q_a_proj", input, 1U, kHidden,
                                q_rank, layer);
        if (!projected.ok()) return projected;
        projected = norm(q_rank, q_rank,
                         attention + "q_a_layernorm.weight");
        if (!projected.ok()) return projected;
        if (timing != nullptr) {
            timing->query_rank_projection_nanoseconds +=
                elapsed_nanoseconds(query_rank_started);
        }
        std::vector<float> index_query(
            static_cast<std::size_t>(kIndexHeads) * kIndexHeadDim);
        std::vector<float> head_weights(kIndexHeads);
        const auto query_projection_started = timing != nullptr
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        auto wq = host_tensor(
            indexer_bases[2],
            static_cast<std::uint64_t>(kIndexHeads) * kIndexHeadDim *
                kQueryRank);
        if (!wq.ok()) return {std::move(wq.errors)};
        glm53_indexer_gate(index_query, q_rank, *wq.value);
        auto wp = host_tensor(
            indexer_bases[3],
            static_cast<std::uint64_t>(kIndexHeads) * kHidden);
        if (!wp.ok()) return {std::move(wp.errors)};
        glm53_indexer_gate(head_weights, input, *wp.value);
        if (timing != nullptr) {
            timing->indexer_projection_nanoseconds +=
                elapsed_nanoseconds(query_projection_started);
        }

        selected.resize(kIndexSelectionWidth);
        const auto& pool_cache = sequence.index_pool(layer);
        const auto selected_count = glm53_sparse_index_select(
            selected, index_query,
            [&](std::uint32_t pool) {
                return pool_cache.row(pool).data();
            },
            head_weights, history, timing);
        selected.resize(selected_count);

        const auto arena_started = timing != nullptr
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        std::vector<std::uint32_t> selected_pools;
        selected_pools.reserve(kIndexTopK / kIndexPool);
        const auto complete_tokens = history / kIndexPool * kIndexPool;
        for (const auto token : selected) {
            if (token >= complete_tokens) continue;
            const auto pool = token / kIndexPool;
            if (selected_pools.empty() || selected_pools.back() != pool) {
                selected_pools.push_back(pool);
            }
        }

        if (!arena.seeded) {
            // If this sequence decoded through the entire identity region,
            // physical rows [0, 2048) already contain pools [0, 512). A long
            // host-prefilled prompt has no such device contents and starts
            // with an empty arena instead.
            if (arena.identity_rows == kIndexTopK) {
                for (std::uint32_t pool = 0U;
                     pool < kIndexTopK / kIndexPool; ++pool) {
                    arena.slot_pools[pool] = pool;
                }
            }
            arena.seeded = true;
            arena.seed_validation_pending = true;
        }
        ++arena.clock;
        const auto find_slot = [&](std::uint32_t pool) {
            const auto found = std::find(arena.slot_pools.begin(),
                                         arena.slot_pools.end(), pool);
            return found == arena.slot_pools.end()
                ? kIndexArenaPools
                : static_cast<std::uint32_t>(
                      found - arena.slot_pools.begin());
        };
        std::vector<std::uint32_t> selected_pool_slots;
        selected_pool_slots.reserve(selected_pools.size());
        for (const auto pool : selected_pools) {
            auto slot = find_slot(pool);
            if (slot == kIndexArenaPools) {
                for (std::uint32_t candidate = 0U;
                     candidate < kIndexArenaPools; ++candidate) {
                    if (arena.slot_pools[candidate] ==
                        std::numeric_limits<std::uint32_t>::max()) {
                        slot = candidate;
                        break;
                    }
                }
            }
            if (slot == kIndexArenaPools) {
                std::uint64_t oldest =
                    std::numeric_limits<std::uint64_t>::max();
                for (std::uint32_t candidate = 0U;
                     candidate < kIndexArenaPools; ++candidate) {
                    if (std::binary_search(
                            selected_pools.begin(), selected_pools.end(),
                            arena.slot_pools[candidate])) {
                        continue;
                    }
                    if (arena.slot_recency[candidate] < oldest) {
                        oldest = arena.slot_recency[candidate];
                        slot = candidate;
                    }
                }
            }
            if (slot == kIndexArenaPools) {
                return {{"GLM-5.3 sparse MLA arena has no evictable pool"}};
            }
            if (arena.slot_pools[slot] != pool) {
                arena.slot_pools[slot] = pool;
                for (std::uint32_t member = 0U; member < kIndexPool;
                     ++member) {
                    expansion_sources.push_back(pool * kIndexPool + member);
                    expansion_destinations.push_back(
                        slot * kIndexPool + member);
                }
            }
            arena.slot_recency[slot] = arena.clock;
            selected_pool_slots.push_back(slot);
        }

        const auto tail_begin = history / kIndexPool * kIndexPool;
        const auto tail_arena_begin = kIndexArenaPools * kIndexPool;
        arena_rows.reserve(selected.size());
        for (std::size_t index = 0U; index < selected_pool_slots.size();
             ++index) {
            const auto slot = selected_pool_slots[index];
            if (slot >= arena.slot_pools.size() ||
                arena.slot_pools[slot] != selected_pools[index]) {
                return {{"GLM-5.3 sparse MLA consumed pool is not cached"}};
            }
            for (std::uint32_t member = 0U; member < kIndexPool; ++member) {
                arena_rows.push_back(slot * kIndexPool + member);
            }
        }
        const auto tail_rows = history % kIndexPool;
        for (std::uint32_t offset = 0U; offset < tail_rows; ++offset) {
            const auto token = tail_begin + offset;
            const auto tail_row = tail_arena_begin + offset;
            arena_rows.push_back(tail_row);
            expansion_sources.push_back(token);
            expansion_destinations.push_back(tail_row);
        }
        if (arena_rows.size() != selected.size()) {
            return {{"GLM-5.3 sparse MLA arena view disagrees with its "
                     "selection"}};
        }
        if (timing != nullptr) {
            timing->arena_bookkeeping_nanoseconds +=
                elapsed_nanoseconds(arena_started);
        }
        return {};
    }

    // Bounded prefill attention for histories past the selection threshold.
    //
    // The dense page path expands the whole history once and shares it across
    // the page's rows, which is what makes prefill cheap per token; its problem
    // is only that `history x 32,768 x 4 B` grows without bound. Expanding each
    // row's own selection instead fixes the bound and destroys the sharing:
    // every row would re-expand about `kIndexTopK` latents, 64 times the work of
    // one dense expansion for a 64-row page.
    //
    // So expand the *union* of the group's selections once. Consecutive prompt
    // rows score nearly the same history, so their selections overlap heavily
    // and the union stays close to one row's width. The group is closed as soon
    // as the union would exceed `kMaxExpandedRows`, which caps the working set
    // whatever the context length while keeping the dense path's sharing.
    [[nodiscard]] ValidationResult attention_mla_page_sparse(
        std::span<float> output, std::span<const float> input,
        std::uint32_t rows, std::uint32_t layer, const std::string& attention,
        Glm53SequenceState& sequence, std::span<const float> q_rank,
        const Glm53PagedRows& latent_rows, std::uint32_t history_begin) {
        // 4,096 expanded rows is 512 MiB, the same order as the dense path's
        // expansion at the context lengths it could still serve, and at least
        // twice the widest single selection so a row can never fail to fit.
        constexpr std::uint32_t kMaxExpandedRows = 4096U;
        static_assert(kMaxExpandedRows >= kIndexSelectionWidth);
        ValidationResult result = complete_index_pools(sequence, layer,
                                                       attention);
        if (!result.ok()) return result;
        const auto& pool_cache = sequence.index_pool(layer);

        std::vector<float> index_query(
            static_cast<std::size_t>(rows) * kIndexHeads * kIndexHeadDim);
        std::vector<float> head_weights(
            static_cast<std::size_t>(rows) * kIndexHeads);
        std::vector<float> query(static_cast<std::size_t>(rows) * kMlaWidth);
        const std::array<std::string, 5U> bases{
            attention + "q_b_proj", attention + "indexer.wq_b.weight",
            attention + "indexer.weights_proj.weight", attention + "kv_b_proj",
            attention + "o_proj"};
        const std::array<Glm53WeightCache::LinearRequest, 1U> projections{
            {{bases[0], kMlaWidth, kQueryRank, q_rank, rows, query, true}}};
        result = linear_batch(projections, layer);
        if (!result.ok()) return result;
        {
            auto wq = host_tensor(bases[1],
                                  static_cast<std::uint64_t>(kIndexHeads) *
                                      kIndexHeadDim * kQueryRank);
            if (!wq.ok()) return {std::move(wq.errors)};
            auto wp = host_tensor(bases[2],
                                  static_cast<std::uint64_t>(kIndexHeads) * kHidden);
            if (!wp.ok()) return {std::move(wp.errors)};
            for (std::uint32_t row = 0U; row < rows; ++row) {
                glm53_indexer_gate(
                    std::span<float>(index_query).subspan(
                        static_cast<std::size_t>(row) * kIndexHeads * kIndexHeadDim,
                        static_cast<std::size_t>(kIndexHeads) * kIndexHeadDim),
                    q_rank.subspan(static_cast<std::size_t>(row) * kQueryRank,
                                   kQueryRank),
                    *wq.value);
                glm53_indexer_gate(
                    std::span<float>(head_weights).subspan(
                        static_cast<std::size_t>(row) * kIndexHeads, kIndexHeads),
                    input.subspan(static_cast<std::size_t>(row) * kHidden, kHidden),
                    *wp.value);
            }
        }

        // Every row's selection first, so the group boundaries can be chosen
        // from the unions they actually produce rather than guessed.
        std::vector<std::vector<std::uint32_t>> selection(rows);
        {
            std::vector<std::uint32_t> selected(kIndexSelectionWidth);
            for (std::uint32_t row = 0U; row < rows; ++row) {
                const auto visible = history_begin + row + 1U;
                const auto count = glm53_sparse_index_select(
                    selected,
                    std::span<const float>(index_query).subspan(
                        static_cast<std::size_t>(row) * kIndexHeads * kIndexHeadDim,
                        static_cast<std::size_t>(kIndexHeads) * kIndexHeadDim),
                    [&](std::uint32_t pool) {
                        return pool_cache.row(pool).data();
                    },
                    std::span<const float>(head_weights).subspan(
                        static_cast<std::size_t>(row) * kIndexHeads, kIndexHeads),
                    visible);
                selection[row].assign(selected.begin(),
                                      selected.begin() +
                                          static_cast<std::ptrdiff_t>(count));
            }
        }

        const auto score_scale = 1.0F / std::sqrt(static_cast<float>(kMlaHead));
        // One row of `attended` per prompt row, so `o_proj` runs once for the
        // page exactly as it does on the dense path.
        std::vector<float> attended(
            static_cast<std::size_t>(rows) * kMlaWidth, 0.0F);
        std::vector<std::uint32_t> group_union, merged;
        std::vector<float> scores;
        for (std::uint32_t group_begin = 0U; group_begin < rows;) {
            group_union.clear();
            auto group_end = group_begin;
            while (group_end < rows) {
                merged.clear();
                std::set_union(group_union.begin(), group_union.end(),
                               selection[group_end].begin(),
                               selection[group_end].end(),
                               std::back_inserter(merged));
                if (group_end != group_begin &&
                    merged.size() > kMaxExpandedRows) {
                    break;
                }
                group_union.swap(merged);
                ++group_end;
            }
            const auto expanded_rows =
                static_cast<std::uint32_t>(group_union.size());
            static_cast<void>(glm53_grow(
                mla_gathered_scratch,
                static_cast<std::size_t>(expanded_rows) * kKvRank));
            const auto gathered = std::span<float>(mla_gathered_scratch)
                .first(static_cast<std::size_t>(expanded_rows) * kKvRank);
            for (std::uint32_t index = 0U; index < expanded_rows; ++index) {
                const auto source = latent_rows.row(group_union[index]);
                std::copy(source.begin(), source.end(),
                          gathered.begin() +
                              static_cast<std::ptrdiff_t>(
                                  static_cast<std::size_t>(index) * kKvRank));
            }
            static_cast<void>(glm53_grow(
                mla_expanded_scratch,
                static_cast<std::size_t>(expanded_rows) * kHeads * 2U *
                    kMlaHead));
            const auto expanded = std::span<float>(mla_expanded_scratch)
                .first(static_cast<std::size_t>(expanded_rows) * kHeads * 2U *
                       kMlaHead);
            const std::array<Glm53WeightCache::LinearRequest, 1U> expand{
                {{bases[3], kHeads * 2U * kMlaHead, kKvRank,
                  gathered, expanded_rows, expanded, true}}};
            result = linear_batch(expand, layer);
            if (!result.ok()) return result;

            for (auto row = group_begin; row < group_end; ++row) {
                // The group's union is sorted and so is each selection, so the
                // mapped positions stay ascending and the accumulation order is
                // the dense path's.
                const auto& chosen = selection[row];
                const auto attended_rows =
                    static_cast<std::uint32_t>(chosen.size());
                std::vector<std::uint32_t> local(attended_rows);
                for (std::uint32_t index = 0U; index < attended_rows; ++index) {
                    local[index] = static_cast<std::uint32_t>(
                        std::lower_bound(group_union.begin(), group_union.end(),
                                         chosen[index]) -
                        group_union.begin());
                }
                scores.assign(attended_rows, 0.0F);
                for (std::uint32_t head = 0U; head < kHeads; ++head) {
                    const auto* q = query.data() +
                        (static_cast<std::size_t>(row) * kHeads + head) * kMlaHead;
                    float highest = -std::numeric_limits<float>::infinity();
                    for (std::uint32_t token = 0U; token < attended_rows; ++token) {
                        const auto* kv = expanded.data() +
                            (static_cast<std::size_t>(local[token]) * kHeads +
                             head) * (2U * kMlaHead);
                        float score = 0.0F;
                        for (std::uint32_t column = 0U; column < kMlaHead; ++column) {
                            score += q[column] * kv[column];
                        }
                        scores[token] = score * score_scale;
                        highest = std::max(highest, scores[token]);
                    }
                    float total = 0.0F;
                    for (auto& score : scores) {
                        score = std::exp(score - highest);
                        total += score;
                    }
                    auto* out = attended.data() +
                        (static_cast<std::size_t>(row) * kHeads + head) * kMlaHead;
                    for (std::uint32_t token = 0U; token < attended_rows; ++token) {
                        // The accepted softmax arithmetic, element for element
                        // with the dense page path: the coefficient is rounded
                        // to BF16 before it ever multiplies a value.
                        const auto coefficient =
                            bf16_round_f32(scores[token] / total);
                        const auto* value = expanded.data() +
                            (static_cast<std::size_t>(local[token]) * kHeads +
                             head) * (2U * kMlaHead) + kMlaHead;
                        for (std::uint32_t column = 0U; column < kMlaHead; ++column) {
                            out[column] += coefficient * value[column];
                        }
                    }
                }
            }
            group_begin = group_end;
        }
        round_bf16(attended);
        return linear(attention + "o_proj", attended, rows, kMlaWidth,
                      output, layer);
    }

    [[nodiscard]] ValidationResult attention_mla_page(
        std::span<float> output, std::span<const float> input,
        std::uint32_t rows, std::uint32_t layer,
        const std::string& attention, Glm53SequenceState& sequence) {
        ValidationResult result;
        std::vector<float> q_rank(static_cast<std::size_t>(rows) * kQueryRank);
        std::vector<float> query(static_cast<std::size_t>(rows) * kMlaWidth);
        std::vector<float> latent(static_cast<std::size_t>(rows) * kKvRank);
        const std::array<std::string, 2U> first_bases{
            attention + "q_a_proj", attention + "kv_a_proj_with_mqa"};
        const std::array<Glm53WeightCache::LinearRequest, 2U> first{
            {{first_bases[0], kQueryRank, kHidden, input, rows, q_rank, true},
             {first_bases[1], kKvRank, kHidden, input, rows, latent, true}}};
        result = linear_batch(first, layer);
        if (!result.ok()) return result;
        result = norm_rows(q_rank, q_rank, rows, kQueryRank,
                           attention + "q_a_layernorm.weight");
        if (!result.ok()) return result;
        result = norm_rows(latent, latent, rows, kKvRank,
                           attention + "kv_a_layernorm.weight");
        if (!result.ok()) return result;
        auto& cache = sequence.mla(layer);
        const auto history_begin = cache.rows();
        result = cache.append_rows(latent, rows);
        if (!result.ok()) return result;
        const auto history_rows = cache.rows();

        // Record this page's indexer keys and gates so later queries -- in this
        // page or a later one -- can pool them. Only needed when the sequence
        // can actually reach the selection threshold.
        auto& index_cache = sequence.indexer(layer);
        const bool sparse_reachable =
            sparse_indexer_active(sequence.maximum_context_tokens());
        if (sparse_reachable) {
            std::vector<float> page_keys(
                static_cast<std::size_t>(rows) * kIndexHeadDim);
            std::vector<float> page_gates(page_keys.size());
            const std::array<std::string, 2U> page_indexer_bases{
                attention + "indexer.wk.weight",
                attention + "indexer.index_kpool_compress_gate"};
            {
                auto wk = host_tensor(page_indexer_bases[0],
                                      static_cast<std::uint64_t>(kIndexHeadDim) *
                                          kHidden);
                if (!wk.ok()) return {std::move(wk.errors)};
                for (std::uint32_t row = 0U; row < rows; ++row) {
                    glm53_indexer_gate(
                        std::span<float>(page_keys).subspan(
                            static_cast<std::size_t>(row) * kIndexHeadDim,
                            kIndexHeadDim),
                        input.subspan(static_cast<std::size_t>(row) * kHidden,
                                      kHidden),
                        *wk.value);
                }
            }
            {
                auto gate = host_tensor(page_indexer_bases[1],
                                        static_cast<std::uint64_t>(kIndexHeadDim) *
                                            kHidden);
                if (!gate.ok()) return {std::move(gate.errors)};
                for (std::uint32_t row = 0U; row < rows; ++row) {
                    glm53_indexer_gate(
                        std::span<float>(page_gates).subspan(
                            static_cast<std::size_t>(row) * kIndexHeadDim,
                            kIndexHeadDim),
                        input.subspan(static_cast<std::size_t>(row) * kHidden,
                                      kHidden),
                        *gate.value);
                }
            }
            auto norm_weight =
                host_tensor(attention + "indexer.k_norm.weight", kIndexHeadDim);
            if (!norm_weight.ok()) return {std::move(norm_weight.errors)};
            auto norm_bias =
                host_tensor(attention + "indexer.k_norm.bias", kIndexHeadDim);
            if (!norm_bias.ok()) return {std::move(norm_bias.errors)};
            std::vector<float> packed(
                static_cast<std::size_t>(rows) * 2U * kIndexHeadDim);
            for (std::uint32_t row = 0U; row < rows; ++row) {
                auto key = std::span<float>(page_keys).subspan(
                    static_cast<std::size_t>(row) * kIndexHeadDim,
                    kIndexHeadDim);
                glm53_indexer_layer_norm(key, *norm_weight.value,
                                         *norm_bias.value);
                auto* destination =
                    packed.data() +
                    static_cast<std::size_t>(row) * 2U * kIndexHeadDim;
                std::copy_n(key.data(), kIndexHeadDim, destination);
                std::copy_n(page_gates.data() +
                                static_cast<std::size_t>(row) * kIndexHeadDim,
                            kIndexHeadDim, destination + kIndexHeadDim);
            }
            result = index_cache.append_rows(packed, rows);
            if (!result.ok()) return result;
        }

        // Above the selection threshold a shared dense expansion of the whole
        // history is what makes long prompts impossible -- it is
        // `history x 32,768 x 4 B`, 4.3 GiB per layer at 32k. Each row then
        // expands only its own bounded selection instead. Below the threshold
        // the shared expansion is kept exactly as it was, which keeps that path
        // bit-identical.
        if (sparse_reachable && history_rows > kIndexTopK) {
            return attention_mla_page_sparse(
                output, input, rows, layer, attention, sequence, q_rank,
                cache, history_begin);
        }

        const auto latent_history = cache.materialize();
        static_cast<void>(glm53_grow(
            mla_expanded_scratch,
            static_cast<std::size_t>(history_rows) * kHeads * 2U * kMlaHead));
        const auto expanded = std::span<float>(mla_expanded_scratch)
            .first(static_cast<std::size_t>(history_rows) * kHeads * 2U *
                   kMlaHead);
        const std::array<std::string, 2U> second_bases{
            attention + "q_b_proj", attention + "kv_b_proj"};
        const std::array<Glm53WeightCache::LinearRequest, 2U> second{
            {{second_bases[0], kMlaWidth, kQueryRank, q_rank, rows, query, true},
             {second_bases[1], kHeads * 2U * kMlaHead, kKvRank,
              latent_history, history_rows, expanded, true}}};
        result = linear_batch(second, layer);
        if (!result.ok()) return result;
        std::vector<float> attended(
            static_cast<std::size_t>(rows) * kMlaWidth, 0.0F);
        const auto score_scale = 1.0F / std::sqrt(static_cast<float>(kMlaHead));
        // Prefill rows are independent given the shared expansion: each writes
        // its own slice of `attended` and its own scores. This loop is what a
        // dense page spends its time in -- about 46 GMAC per page across the
        // eleven layers at a 2,000-token history -- and it was serial.
        const auto attend_row = [&](std::size_t row) {
            const auto visible =
                history_begin + static_cast<std::uint32_t>(row) + 1U;
            std::vector<float> scores(visible);
            for (std::uint32_t head = 0U; head < kHeads; ++head) {
                const auto* q = query.data() +
                    (static_cast<std::size_t>(row) * kHeads + head) * kMlaHead;
                float highest = -std::numeric_limits<float>::infinity();
                for (std::uint32_t token = 0U; token < visible; ++token) {
                    const auto* kv = expanded.data() +
                        (static_cast<std::size_t>(token) * kHeads + head) *
                            (2U * kMlaHead);
                    float score = 0.0F;
                    for (std::uint32_t column = 0U; column < kMlaHead; ++column) {
                        score += q[column] * kv[column];
                    }
                    scores[token] = score * score_scale;
                    highest = std::max(highest, scores[token]);
                }
                float total = 0.0F;
                for (auto& score : scores) {
                    score = std::exp(score - highest);
                    total += score;
                }
                auto* destination = attended.data() +
                    (row * kHeads + head) * kMlaHead;
                for (std::uint32_t token = 0U; token < visible; ++token) {
                    const auto* values = expanded.data() +
                        (static_cast<std::size_t>(token) * kHeads + head) *
                            (2U * kMlaHead) + kMlaHead;
                    const auto coefficient =
                        bf16_round_f32(scores[token] / total);
                    for (std::uint32_t column = 0U; column < kMlaHead; ++column) {
                        destination[column] += coefficient * values[column];
                    }
                }
            }
        };
        if (kda_workers != nullptr) {
            auto attended_all = kda_workers->parallel_for(rows, attend_row);
            if (!attended_all.ok()) return attended_all;
        } else {
            for (std::size_t row = 0U; row < rows; ++row) attend_row(row);
        }
        round_bf16(attended);
        return linear(attention + "o_proj", attended, rows, kMlaWidth,
                      output, layer);
    }

    [[nodiscard]] ValidationResult swiglu_block(
        std::span<float> output, std::span<const float> input,
        const std::string& prefix, std::uint32_t inner,
        std::uint32_t layer) {
        ValidationResult result;
        std::vector<float> gate(inner), up(inner), activated(inner);
        const std::array<std::string, 2U> bases{
            prefix + "gate_proj", prefix + "up_proj"};
        const std::array<Glm53WeightCache::LinearRequest, 2U> projections{
            {{bases[0], inner, kHidden, input, 1U, gate, true},
             {bases[1], inner, kHidden, input, 1U, up, true}}};
        result = linear_batch(projections, layer);
        if (!result.ok()) return result;
        for (std::size_t index = 0U; index < inner; ++index) {
            const auto g = std::min(gate[index], 10.0F);
            const auto u = std::clamp(up[index], -10.0F, 10.0F);
            activated[index] = g * sigmoid(g) * u;
        }
        round_bf16(activated);
        return linear(prefix + "down_proj", activated, 1U, inner,
                      output, layer);
    }

    [[nodiscard]] ValidationResult swiglu_block_page(
        std::span<float> output, std::span<const float> input,
        const std::string& prefix, std::uint32_t rows,
        std::uint32_t inner, std::uint32_t layer) {
        ValidationResult result;
        std::vector<float> gate(static_cast<std::size_t>(rows) * inner);
        std::vector<float> up(gate.size()), activated(gate.size());
        const std::array<std::string, 2U> bases{
            prefix + "gate_proj", prefix + "up_proj"};
        const std::array<Glm53WeightCache::LinearRequest, 2U> projections{
            {{bases[0], inner, kHidden, input, rows, gate, true},
             {bases[1], inner, kHidden, input, rows, up, true}}};
        result = linear_batch(projections, layer);
        if (!result.ok()) return result;
        for (std::size_t index = 0U; index < gate.size(); ++index) {
            const auto g = std::min(gate[index], 10.0F);
            const auto u = std::clamp(up[index], -10.0F, 10.0F);
            activated[index] = g * sigmoid(g) * u;
        }
        round_bf16(activated);
        return linear(prefix + "down_proj", activated, rows, inner,
                      output, layer);
    }

    [[nodiscard]] ValidationResult feedforward(
        std::span<float> output, std::span<const float> input,
        std::uint32_t layer, const std::string& prefix,
        std::uint64_t route_request = 0U,
        std::uint32_t route_position = 0U,
        bool schedule_prefetch = false) {
        if (layer != kMtpLayer && !glm53_moe_layer(layer)) {
            return swiglu_block(output, input, prefix + "mlp.", 12288U, layer);
        }
        ValidationResult result;
        std::vector<float> logits(288U);
        // The reference router explicitly promotes both operands to F32.
        result = linear(prefix + "mlp.gate", input, 1U, kHidden, logits,
                        layer, false);
        if (!result.ok()) return result;
        auto bias = host_tensor(
            prefix + "mlp.gate.e_score_correction_bias", 288U);
        if (!bias.ok()) return {std::move(bias.errors)};
        std::array<KimiRoutedExpert, 8U> selected{};
        result = kimi_route_topk(selected, logits, *bias.value, 2.5F);
        if (!result.ok()) return result;
        observe_route(layer, selected, route_request, route_position,
                      schedule_prefetch);
        if (host_moe_active) {
            return host_moe(layer, prefix + "mlp.", selected, input, output);
        }
        return weights->moe(slot_for(layer), prefix + "mlp.", selected,
                            input, output);
    }

    [[nodiscard]] ValidationResult feedforward_page(
        std::span<float> output, std::span<const float> input,
        std::uint32_t rows, std::uint32_t layer, const std::string& prefix,
        std::span<const std::uint64_t> route_requests = {},
        std::span<const std::uint32_t> route_positions = {},
        bool schedule_prefetch = false, bool route_prefill = false,
        bool allow_device_tiers = false) {
        if ((!route_requests.empty() || !route_positions.empty()) &&
            (route_requests.size() != rows || route_positions.size() != rows)) {
            return {{"GLM-5.3 route-observation page has an invalid shape"}};
        }
        if (layer != kMtpLayer && !glm53_moe_layer(layer)) {
            if (allow_device_tiers) {
                for (std::uint32_t row = 0U; row < rows; ++row) {
                    auto result = swiglu_block(
                        output.subspan(static_cast<std::size_t>(row) * kHidden,
                                       kHidden),
                        input.subspan(static_cast<std::size_t>(row) * kHidden,
                                      kHidden),
                        prefix + "mlp.", 12288U, layer);
                    if (!result.ok()) return result;
                }
                return {};
            }
            return swiglu_block_page(output, input, prefix + "mlp.", rows,
                                     12288U, layer);
        }
        ValidationResult result;
        std::vector<float> logits(static_cast<std::size_t>(rows) * 288U);
        if (allow_device_tiers) {
            // Independent decode sequences share expert weights, not dense
            // projection arithmetic. Preserve the accepted batch-1 router
            // association for each row so a close route decision cannot move
            // merely because another request joined the scheduler cohort.
            for (std::uint32_t row = 0U; row < rows; ++row) {
                result = linear(
                    prefix + "mlp.gate",
                    input.subspan(static_cast<std::size_t>(row) * kHidden,
                                  kHidden),
                    1U, kHidden,
                    std::span<float>(logits).subspan(
                        static_cast<std::size_t>(row) * 288U, 288U),
                    layer, false);
                if (!result.ok()) return result;
            }
        } else {
            result = linear(prefix + "mlp.gate", input, rows, kHidden, logits,
                            layer, false);
            if (!result.ok()) return result;
        }
        auto bias = host_tensor(
            prefix + "mlp.gate.e_score_correction_bias", 288U);
        if (!bias.ok()) return {std::move(bias.errors)};
        std::vector<std::array<KimiRoutedExpert, 8U>> selected_rows(rows);
        for (std::uint32_t row = 0U; row < rows; ++row) {
            auto& selected = selected_rows[row];
            result = kimi_route_topk(
                selected,
                std::span<const float>(logits).subspan(
                    static_cast<std::size_t>(row) * 288U, 288U),
                *bias.value, 2.5F);
            if (!result.ok()) return result;
            if (!route_requests.empty()) {
                // Prompt pages and independent decode cohorts both have
                // `rows > 1`; only the caller knows which one this is. Using
                // width as phase mislabeled concurrent server decode as
                // prefill and silently dropped those tokens from M4 policy
                // analysis.
                observe_route(layer, selected, route_requests[row],
                              route_positions[row], schedule_prefetch,
                              route_prefill);
            }
        }
        if (host_moe_active) {
            return host_moe_page(layer, prefix + "mlp.", selected_rows, input,
                                 output, allow_device_tiers);
        }
        for (std::uint32_t row = 0U; row < rows; ++row) {
            result = weights->moe(
                slot_for(layer), prefix + "mlp.", selected_rows[row],
                input.subspan(static_cast<std::size_t>(row) * kHidden, kHidden),
                output.subspan(static_cast<std::size_t>(row) * kHidden, kHidden));
            if (!result.ok()) return result;
        }
        return result;
    }

    [[nodiscard]] ValidationResult initialize_streams(
        std::uint32_t token, std::span<float> streams) {
        ValidationResult result;
        if (streams.size() != static_cast<std::size_t>(kMhc) * kHidden) {
            result.errors.emplace_back(
                "GLM-5.3 token streams have an invalid shape");
            return result;
        }
        auto embedding = checkpoint->read_f32_row(
            "model.language_model.embed_tokens.weight", token);
        if (!embedding.ok()) return {std::move(embedding.errors)};
        for (std::uint32_t stream = 0U; stream < kMhc; ++stream) {
            std::copy(embedding.value.begin(), embedding.value.end(),
                      streams.begin() + static_cast<std::ptrdiff_t>(
                          stream * kHidden));
        }
        return result;
    }

    [[nodiscard]] ValidationResult forward_layer(
        std::span<float> streams, std::uint32_t layer,
        std::uint32_t position, Glm53SequenceState& sequence) {
        ValidationResult result;
        if (streams.size() != static_cast<std::size_t>(kMhc) * kHidden ||
            layer >= kLayers) {
            result.errors.emplace_back(
                "GLM-5.3 layer command has an invalid shape");
            return result;
        }
        std::vector<float> collapsed(kHidden), normalized(kHidden), branch(kHidden);
        const auto prefix = "model.language_model.layers." +
                            std::to_string(layer) + ".";
        Dsv4MhcMix mix;
        result = mhc_pre(collapsed, mix, streams, prefix + "hc_attn");
        if (!result.ok()) return result;
        result = norm(normalized, collapsed, prefix + "input_layernorm.weight");
        if (!result.ok()) return result;
        const auto attention = prefix + "self_attn.";
        const auto attention_started = std::chrono::steady_clock::now();
        result = glm53_kda_layer(layer)
            ? attention_kda(branch, normalized, layer, attention, sequence)
            : attention_mla(branch, normalized, layer, position, attention,
                            sequence);
        if (!result.ok()) return result;
        if (config.phase_profile) {
            const auto elapsed = elapsed_nanoseconds(attention_started);
            graph_attention_block_nanoseconds.fetch_add(
                elapsed, std::memory_order_relaxed);
            (glm53_kda_layer(layer) ? graph_kda_nanoseconds
                                    : graph_mla_nanoseconds)
                .fetch_add(elapsed, std::memory_order_relaxed);
        }
        std::vector<float> transitioned(streams.size());
        result = dsv4_mhc_post_f32(transitioned, branch, streams, mix, kMhc);
        if (!result.ok()) return result;
        round_bf16(transitioned);
        std::copy(transitioned.begin(), transitioned.end(), streams.begin());

        result = mhc_pre(collapsed, mix, streams, prefix + "hc_ffn");
        if (!result.ok()) return result;
        result = norm(normalized, collapsed,
                      prefix + "post_attention_layernorm.weight");
        if (!result.ok()) return result;
        const auto feedforward_started = std::chrono::steady_clock::now();
        result = feedforward(
            branch, normalized, layer, prefix,
            route_request_key(&sequence, position), position, true);
        if (!result.ok()) return result;
        if (config.phase_profile) {
            graph_feedforward_block_nanoseconds.fetch_add(
                elapsed_nanoseconds(feedforward_started),
                std::memory_order_relaxed);
        }
        std::fill(transitioned.begin(), transitioned.end(), 0.0F);
        result = dsv4_mhc_post_f32(transitioned, branch, streams, mix, kMhc);
        if (!result.ok()) return result;
        round_bf16(transitioned);
        std::copy(transitioned.begin(), transitioned.end(), streams.begin());
        return result;
    }

    [[nodiscard]] ValidationResult forward_layer_resident(
        std::span<float> streams, std::uint32_t layer,
        std::uint32_t position, Glm53SequenceState& sequence,
        DeviceSequenceState& device_sequence) {
        if (!resident_execution_active || !device_sequence.ready ||
            streams.size() != static_cast<std::size_t>(kMhc) * kHidden) {
            return {{"GLM-5.3 resident layer command is not admissible"}};
        }
        const auto device = device_for(layer);
        auto result = cuda.dsv4_mhc_begin_device(
            device, resident_layers[layer].attention, streams);
        if (!result.ok()) return result;
        const auto prefix = "model.language_model.layers." +
                            std::to_string(layer) + ".";
        const auto attention = prefix + "self_attn.";
        // The layer input, before attention touches it.
        //
        // Record 0214's attention oracle downloads this vector and feeds it to
        // the host fallback, so it compares attention given identical input and
        // is blind to whatever produced the input. `dsv4_mhc_begin_device` runs
        // the mHC pre projection and the input layer norm on the device; the
        // fallback runs both on the host. Compare them here, on every resident
        // layer rather than only the MLA ones, because the KDA layers take the
        // device path in both arms and a difference there would mean the
        // control arm is itself a mixture.
        if (resident_mla_compare_enabled()) {
            std::vector<float> device_normalized(kHidden),
                collapsed(kHidden), host_normalized(kHidden);
            result = cuda.dsv4_mhc_download_layer_input(
                device, device_normalized);
            if (!result.ok()) return result;
            // `dsv4_mhc_begin_impl` BF16-encodes the incoming streams before
            // it uploads them, so model that here: any difference that
            // survives is a difference between the two mHC implementations
            // rather than between their inputs.
            std::vector<float> rounded(streams.begin(), streams.end());
            round_bf16(rounded);
            Dsv4MhcMix host_mix;
            result = mhc_pre(collapsed, host_mix, rounded,
                             prefix + "hc_attn");
            if (!result.ok()) return result;
            result = norm(host_normalized, collapsed,
                          prefix + "input_layernorm.weight");
            if (!result.ok()) return result;
            std::size_t differing = 0U;
            float worst = 0.0F;
            for (std::size_t index = 0U; index < host_normalized.size();
                 ++index) {
                if (host_normalized[index] == device_normalized[index]) continue;
                ++differing;
                worst = std::max(worst, std::fabs(host_normalized[index] -
                                                  device_normalized[index]));
            }
            if (differing != 0U && position == 36U) {
                std::cerr << "[glm53-mhc-compare] layer " << layer << " ("
                          << (glm53_kda_layer(layer) ? "KDA" : "MLA")
                          << ") differing " << differing << "/"
                          << host_normalized.size() << " worst " << worst
                          << '\n';
            }
        }
        const auto attention_started = std::chrono::steady_clock::now();
        if (glm53_kda_layer(layer)) {
            CudaGlm53KdaRequest request;
            request.state = &device_sequence.kda[layer];
            request.heads = kHeads;
            request.head_dim = kLinearHead;
            request.convolution_kernel = 4U;
            request.mhc_source_destination = true;
            result = weights->kda_decode(
                slot_for(layer), attention, request, {});
        } else if (resident_mla_host_attention()) {
            // Explicit diagnostic control: move only attention to the host.
            // Sparse contexts have their own independent device entry point
            // below and never share the <=2,048 resident MLA kernels.
            std::vector<float> normalized(kHidden), branch(kHidden);
            result = cuda.dsv4_mhc_download_layer_input(device, normalized);
            if (!result.ok()) return result;
            result = attention_mla(branch, normalized, layer, position,
                                   attention, sequence);
            if (!result.ok()) return result;
            result = cuda.dsv4_mhc_publish_branch(device, branch);
        } else if (sparse_indexer_active(config.maximum_context_tokens)) {
            Glm53SparseMlaMetrics sparse_timing;
            CudaGlm53MlaRequest::HostTiming cuda_host_timing;
            auto* const sparse_timing_ptr = config.phase_profile
                ? &sparse_timing
                : nullptr;
            if (sparse_timing_ptr != nullptr) sparse_timing.calls = 1U;
            CudaGlm53MlaRequest request;
            request.state = &device_sequence.mla[layer];
            request.sparse_expanded =
                &device_sequence.sparse_mla_expanded[layer];
            request.position = position;
            request.maximum_context = config.maximum_context_tokens;
            request.heads = kHeads;
            request.head_dim = kMlaHead;
            request.query_rank = kQueryRank;
            request.key_value_rank = kKvRank;
            request.host_timing = config.phase_profile
                ? &cuda_host_timing
                : nullptr;
            static_cast<void>(glm53_grow(sparse_index_input, kHidden));
            auto normalized = std::span<float>(sparse_index_input)
                .first(kHidden);
            const auto input_download_started = config.phase_profile
                ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point{};
            result = cuda.dsv4_mhc_download_layer_input(device, normalized);
            if (!result.ok()) return result;
            if (sparse_timing_ptr != nullptr) {
                sparse_timing.input_download_nanoseconds +=
                    elapsed_nanoseconds(input_download_started);
            }
            result = prepare_sparse_device_selection(
                mla_selected_positions, mla_arena_rows,
                mla_expansion_source_positions,
                mla_expansion_destination_rows, normalized, layer,
                position, attention, sequence, device_sequence,
                sparse_timing_ptr);
            if (!result.ok()) return result;
            auto& sparse_arena =
                device_sequence.sparse_mla_arenas[layer];
            request.validate_sparse_identity_rows =
                sparse_arena.seed_validation_pending;
            request.sparse_identity_rows = sparse_arena.identity_rows;
            request.selected_positions = mla_selected_positions;
            request.sparse_arena_rows = mla_arena_rows;
            request.sparse_expansion_sources =
                mla_expansion_source_positions;
            request.sparse_expansion_destinations =
                mla_expansion_destination_rows;
            auto& scores = mla_softmax_scores;
            const auto history = position + 1U;
            const auto attended_rows = request.selected_positions.empty()
                ? static_cast<std::size_t>(history)
                : request.selected_positions.size();
            static_cast<void>(glm53_grow(
                scores, static_cast<std::size_t>(kHeads) * attended_rows));
            result = weights->sparse_mla_decode_mhc(
                slot_for(layer), attention, request,
                std::span<float>(scores).first(
                    static_cast<std::size_t>(kHeads) * attended_rows),
                [](std::span<float> values, std::uint32_t heads,
                   std::uint32_t tokens) {
                    for (std::uint32_t head = 0U; head < heads; ++head) {
                        auto* row = values.data() +
                            static_cast<std::size_t>(head) * tokens;
                        float highest = -std::numeric_limits<float>::infinity();
                        for (std::uint32_t token = 0U; token < tokens;
                             ++token) {
                            highest = std::max(highest, row[token]);
                        }
                        float total = 0.0F;
                        for (std::uint32_t token = 0U; token < tokens;
                             ++token) {
                            row[token] = std::exp(row[token] - highest);
                            total += row[token];
                        }
                        for (std::uint32_t token = 0U; token < tokens;
                             ++token) {
                            row[token] = bf16_round_f32(row[token] / total);
                        }
                    }
                });
            if (result.ok() && request.validate_sparse_identity_rows) {
                sparse_arena.seed_validation_pending = false;
            }
            if (sparse_timing_ptr != nullptr) {
                sparse_timing.index_upload_nanoseconds +=
                    cuda_host_timing.index_upload_nanoseconds;
                sparse_timing.device_scores_wait_nanoseconds +=
                    cuda_host_timing.device_scores_wait_nanoseconds;
                sparse_timing.host_softmax_nanoseconds +=
                    cuda_host_timing.host_softmax_nanoseconds;
                sparse_timing.coefficient_upload_nanoseconds +=
                    cuda_host_timing.coefficient_upload_nanoseconds;
                add_sparse_mla_metrics(sparse_timing);
            }
        } else {
            CudaGlm53MlaRequest request;
            request.state = &device_sequence.mla[layer];
            request.position = position;
            request.maximum_context = config.maximum_context_tokens;
            request.heads = kHeads;
            request.head_dim = kMlaHead;
            request.query_rank = kQueryRank;
            request.key_value_rank = kKvRank;
            // The softmax the device cannot run and stay exact. This is the
            // accepted fallback's arithmetic, element for element: the maximum
            // is subtracted, `std::exp` is glibc's, the sum is accumulated in
            // token order, and the coefficient is rounded to BF16 before it
            // ever multiplies a value.
            auto& scores = mla_softmax_scores;
            const auto history = position + 1U;
            if (glm53_grow(scores,
                           static_cast<std::size_t>(kHeads) * history)) {
                // Grown once per context length, never inside steady state.
            }
            result = weights->mla_decode_mhc(
                slot_for(layer), attention, request,
                std::span<float>(scores).first(
                    static_cast<std::size_t>(kHeads) * history),
                [](std::span<float> values, std::uint32_t heads,
                   std::uint32_t tokens) {
                    for (std::uint32_t head = 0U; head < heads; ++head) {
                        auto* row = values.data() +
                            static_cast<std::size_t>(head) * tokens;
                        float highest = -std::numeric_limits<float>::infinity();
                        for (std::uint32_t token = 0U; token < tokens; ++token) {
                            highest = std::max(highest, row[token]);
                        }
                        float total = 0.0F;
                        for (std::uint32_t token = 0U; token < tokens; ++token) {
                            row[token] = std::exp(row[token] - highest);
                            total += row[token];
                        }
                        for (std::uint32_t token = 0U; token < tokens; ++token) {
                            row[token] = bf16_round_f32(row[token] / total);
                        }
                    }
                });
        }
        if (!result.ok()) return result;
        if (!glm53_kda_layer(layer) && resident_mla_compare_enabled()) {
            std::vector<float> normalized(kHidden), actual(kHidden),
                expected(kHidden);
            result = cuda.dsv4_mhc_download_layer_input(device, normalized);
            if (!result.ok()) return result;
            result = cuda.dsv4_mhc_download_branch(device, actual);
            if (!result.ok()) return result;
            result = attention_mla(expected, normalized, layer, position,
                                   attention, sequence);
            if (!result.ok()) return result;
            // The branch is what the layer publishes, but it is not the only
            // thing the resident path writes: it also appends this position's
            // latent to its device cache, and every later step attends over
            // that history. A branch that matches today and a latent that is
            // one ULP out is exactly the failure that survives a short oracle
            // and diverges a long one, so compare the latent first.
            //
            // `attention_mla` above has already appended this position's row
            // to the host cache, so both sides now hold row `position`.
            auto& host_cache = sequence.mla(layer);
            if (host_cache.rows() == position + 1U) {
                const auto host_rows = host_cache.materialize();
                std::vector<float> device_row(kKvRank);
                const auto latent_offset =
                    static_cast<std::uint64_t>(position) * kKvRank *
                    sizeof(float);
                auto downloaded = cuda.download_buffer(
                    device_sequence.mla[layer], latent_offset,
                    std::as_writable_bytes(std::span<float>(device_row)));
                if (!downloaded.ok()) return downloaded;
                const auto* host_row =
                    host_rows.data() +
                    static_cast<std::size_t>(position) * kKvRank;
                for (std::size_t index = 0U; index < device_row.size();
                     ++index) {
                    if (host_row[index] == device_row[index]) continue;
                    return {{"GLM-5.3 resident MLA latent mismatch at layer " +
                             std::to_string(layer) + ", position " +
                             std::to_string(position) + ", element " +
                             std::to_string(index) + ": expected " +
                             std::to_string(host_row[index]) + ", actual " +
                             std::to_string(device_row[index])}};
                }
            }
            std::size_t first = expected.size();
            float maximum = 0.0F;
            for (std::size_t index = 0U; index < expected.size(); ++index) {
                maximum = std::max(maximum,
                                   std::fabs(expected[index] - actual[index]));
                if (first == expected.size() &&
                    expected[index] != actual[index]) first = index;
            }
            if (first != expected.size()) {
                return {{"GLM-5.3 resident MLA exactness mismatch at layer " +
                         std::to_string(layer) + ", position " +
                         std::to_string(position) + ", element " +
                         std::to_string(first) + ": expected " +
                         std::to_string(expected[first]) + ", actual " +
                         std::to_string(actual[first]) + ", maximum " +
                         std::to_string(maximum)}};
            }
        }
        if (config.phase_profile) {
            const auto elapsed = elapsed_nanoseconds(attention_started);
            graph_attention_block_nanoseconds.fetch_add(
                elapsed, std::memory_order_relaxed);
            if (glm53_kda_layer(layer)) {
                graph_kda_nanoseconds.fetch_add(elapsed,
                                                std::memory_order_relaxed);
            } else {
                graph_mla_nanoseconds.fetch_add(elapsed,
                                                std::memory_order_relaxed);
            }
        }
        result = cuda.dsv4_mhc_transition_next_device(
            device, resident_layers[layer].feedforward);
        if (!result.ok()) return result;
        const auto feedforward_started = std::chrono::steady_clock::now();
        if (host_moe_active) {
            std::vector<float> normalized(kHidden), branch(kHidden);
            result = cuda.dsv4_mhc_download_layer_input(device, normalized);
            if (!result.ok()) return result;
            if (glm53_moe_layer(layer)) {
                std::vector<float> logits(288U);
                result = weights->router_mhc(
                    slot_for(layer), prefix + "mlp.gate", logits);
                if (!result.ok()) return result;
                auto bias = host_tensor(
                    prefix + "mlp.gate.e_score_correction_bias", 288U);
                if (!bias.ok()) return {std::move(bias.errors)};
                std::array<KimiRoutedExpert, 8U> selected{};
                result = kimi_route_topk(selected, logits, *bias.value, 2.5F);
                if (!result.ok()) return result;
                observe_route(
                    layer, selected, route_request_key(&sequence, position),
                    position, false);
                result = host_moe(layer, prefix + "mlp.", selected, normalized,
                                  branch);
            } else {
                result = swiglu_block(branch, normalized, prefix + "mlp.",
                                      12288U, layer);
            }
            if (!result.ok()) return result;
            if (config.phase_profile) {
                graph_feedforward_block_nanoseconds.fetch_add(
                    elapsed_nanoseconds(feedforward_started),
                    std::memory_order_relaxed);
            }
            return cuda.dsv4_mhc_finish(device, branch, streams);
        }
        if (glm53_moe_layer(layer)) {
            std::vector<float> logits(288U);
            result = weights->router_mhc(
                slot_for(layer), prefix + "mlp.gate", logits);
            if (!result.ok()) return result;
            auto bias = host_tensor(
                prefix + "mlp.gate.e_score_correction_bias", 288U);
            if (!bias.ok()) return {std::move(bias.errors)};
            std::array<KimiRoutedExpert, 8U> selected{};
            result = kimi_route_topk(selected, logits, *bias.value, 2.5F);
            if (!result.ok()) return result;
            observe_route(
                layer, selected, route_request_key(&sequence, position),
                position, true);
            result = weights->moe(
                slot_for(layer), prefix + "mlp.", selected, {}, {}, true);
        } else {
            result = weights->swiglu_mhc(
                slot_for(layer), prefix + "mlp.", 12288U);
        }
        if (!result.ok()) return result;
        if (config.phase_profile) {
            graph_feedforward_block_nanoseconds.fetch_add(
                elapsed_nanoseconds(feedforward_started),
                std::memory_order_relaxed);
        }
        return cuda.dsv4_mhc_finish_device(device, streams);
    }

    [[nodiscard]] ValidationResult forward_layer_page_device(
        std::span<float> streams, std::uint32_t rows, std::uint32_t layer,
        Glm53SequenceState& sequence, DeviceSequenceState& device_sequence) {
        const auto stream_columns = static_cast<std::size_t>(kMhc) * kHidden;
        if (!resident_execution_active || !device_sequence.ready || rows == 0U ||
            layer >= kLayers ||
            streams.size() != static_cast<std::size_t>(rows) * stream_columns) {
            return {{"GLM-5.3 device page command is not admissible"}};
        }
        const auto device = device_for(layer);
        const auto prefix = "model.language_model.layers." +
                            std::to_string(layer) + ".";
        const auto attention = prefix + "self_attn.";
        const bool device_mla = !glm53_kda_layer(layer) &&
            device_page_mla_enabled() &&
            weights->mla_kv_b_is_bf16(attention);
        std::vector<float> normalized(
            static_cast<std::size_t>(rows) * kHidden);
        std::vector<float> branch(normalized.size());
        ValidationResult result;
        for (std::uint32_t row = 0U; row < rows; ++row) {
            result = cuda.dsv4_mhc_select_slot(device, row);
            if (!result.ok()) return result;
            result = cuda.dsv4_mhc_begin_device(
                device, resident_layers[layer].attention,
                std::span<const float>(streams).subspan(
                    static_cast<std::size_t>(row) * stream_columns,
                    stream_columns));
            if (!result.ok()) return result;
            if (!glm53_kda_layer(layer) && !device_mla) {
                result = cuda.dsv4_mhc_download_layer_input(
                    device, std::span<float>(normalized).subspan(
                                static_cast<std::size_t>(row) * kHidden,
                                kHidden));
                if (!result.ok()) return result;
            }
        }
        const auto attention_started = std::chrono::steady_clock::now();
        if (glm53_kda_layer(layer)) {
            CudaGlm53KdaRequest request;
            request.state = &device_sequence.kda[layer];
            request.heads = kHeads;
            request.head_dim = kLinearHead;
            request.convolution_kernel = 4U;
            request.mhc_source_destination = true;
            request.page_rows = rows;
            result = weights->kda_decode(
                slot_for(layer), attention, request, {});
        } else if (device_mla) {
            // Causal prompt rows share one MLA state, so advance it in token
            // order. Each selected mHC slot retains that row's device input;
            // the accepted resident primitive performs all linear algebra on
            // the device and returns only raw scores for the exact GLIBC
            // softmax. This removes the page-wide hidden-state download and
            // branch upload while preserving every per-token association.
            const auto position_base = sequence.token_count();
            for (std::uint32_t row = 0U; row < rows; ++row) {
                result = cuda.dsv4_mhc_select_slot(device, row);
                if (!result.ok()) break;
                CudaGlm53MlaRequest request;
                request.state = &device_sequence.mla[layer];
                request.position = position_base + row;
                request.maximum_context = config.maximum_context_tokens;
                request.heads = kHeads;
                request.head_dim = kMlaHead;
                request.query_rank = kQueryRank;
                request.key_value_rank = kKvRank;
                const auto history = request.position + 1U;
                auto scores = std::span<float>(mla_softmax_scores).first(
                    static_cast<std::size_t>(kHeads) * history);
                result = weights->mla_decode_mhc(
                    slot_for(layer), attention, request, scores,
                    [](std::span<float> values, std::uint32_t heads,
                       std::uint32_t tokens) {
                        for (std::uint32_t head = 0U; head < heads; ++head) {
                            auto* scores_row = values.data() +
                                static_cast<std::size_t>(head) * tokens;
                            float highest =
                                -std::numeric_limits<float>::infinity();
                            for (std::uint32_t token = 0U; token < tokens;
                                 ++token) {
                                highest = std::max(highest, scores_row[token]);
                            }
                            float total = 0.0F;
                            for (std::uint32_t token = 0U; token < tokens;
                                 ++token) {
                                scores_row[token] =
                                    std::exp(scores_row[token] - highest);
                                total += scores_row[token];
                            }
                            for (std::uint32_t token = 0U; token < tokens;
                                 ++token) {
                                scores_row[token] = bf16_round_f32(
                                    scores_row[token] / total);
                            }
                        }
                    });
                if (!result.ok()) break;
            }
        } else {
            result = attention_mla_page(
                branch, normalized, rows, layer, attention, sequence);
            if (result.ok()) {
                for (std::uint32_t row = 0U; row < rows; ++row) {
                    result = cuda.dsv4_mhc_select_slot(device, row);
                    if (!result.ok()) break;
                    result = cuda.dsv4_mhc_publish_branch(
                        device, std::span<const float>(branch).subspan(
                                    static_cast<std::size_t>(row) * kHidden,
                                    kHidden));
                    if (!result.ok()) break;
                }
            }
        }
        if (!result.ok()) return result;
        if (config.phase_profile) {
            const auto elapsed = elapsed_nanoseconds(attention_started);
            graph_attention_block_nanoseconds.fetch_add(
                elapsed, std::memory_order_relaxed);
            (glm53_kda_layer(layer) ? graph_kda_nanoseconds
                                    : graph_mla_nanoseconds)
                .fetch_add(elapsed, std::memory_order_relaxed);
        }
        for (std::uint32_t row = 0U; row < rows; ++row) {
            result = cuda.dsv4_mhc_select_slot(device, row);
            if (!result.ok()) return result;
            result = cuda.dsv4_mhc_transition_next_device(
                device, resident_layers[layer].feedforward);
            if (!result.ok()) return result;
            result = cuda.dsv4_mhc_download_layer_input(
                device, std::span<float>(normalized).subspan(
                            static_cast<std::size_t>(row) * kHidden, kHidden));
            if (!result.ok()) return result;
        }
        std::vector<std::uint64_t> route_requests(rows);
        std::vector<std::uint32_t> route_positions(rows);
        const auto position_base = sequence.token_count();
        for (std::uint32_t row = 0U; row < rows; ++row) {
            route_positions[row] = position_base + row;
            route_requests[row] = route_request_key(
                &sequence, route_positions[row]);
        }
        const auto feedforward_started = std::chrono::steady_clock::now();
        result = feedforward_page(
            branch, normalized, rows, layer, prefix, route_requests,
            route_positions, false, true);
        if (!result.ok()) return result;
        if (config.phase_profile) {
            graph_feedforward_block_nanoseconds.fetch_add(
                elapsed_nanoseconds(feedforward_started),
                std::memory_order_relaxed);
        }
        for (std::uint32_t row = 0U; row < rows; ++row) {
            result = cuda.dsv4_mhc_select_slot(device, row);
            if (!result.ok()) return result;
            result = cuda.dsv4_mhc_finish(
                device,
                std::span<const float>(branch).subspan(
                    static_cast<std::size_t>(row) * kHidden, kHidden),
                streams.subspan(
                    static_cast<std::size_t>(row) * stream_columns,
                    stream_columns));
            if (!result.ok()) return result;
        }
        return cuda.dsv4_mhc_select_slot(device, 0U);
    }

    [[nodiscard]] ValidationResult forward_layer_page(
        std::span<float> streams, std::uint32_t rows, std::uint32_t layer,
        Glm53SequenceState& sequence) {
        ValidationResult result;
        const auto stream_columns = static_cast<std::size_t>(kMhc) * kHidden;
        if (rows == 0U || layer >= kLayers ||
            streams.size() != static_cast<std::size_t>(rows) * stream_columns) {
            result.errors.emplace_back(
                "GLM-5.3 layer page has an invalid shape");
            return result;
        }
        const auto hidden_elements = static_cast<std::size_t>(rows) * kHidden;
        std::vector<float> collapsed(hidden_elements), normalized(hidden_elements),
            branch(hidden_elements);
        std::vector<Dsv4MhcMix> mixes(rows);
        const auto prefix = "model.language_model.layers." +
                            std::to_string(layer) + ".";
        for (std::uint32_t row = 0U; row < rows; ++row) {
            result = mhc_pre(
                std::span<float>(collapsed).subspan(
                    static_cast<std::size_t>(row) * kHidden, kHidden),
                mixes[row],
                std::span<const float>(streams).subspan(
                    static_cast<std::size_t>(row) * stream_columns,
                    stream_columns),
                prefix + "hc_attn");
            if (!result.ok()) return result;
        }
        result = norm_rows(normalized, collapsed, rows, kHidden,
                           prefix + "input_layernorm.weight");
        if (!result.ok()) return result;
        const auto attention = prefix + "self_attn.";
        const auto attention_started = std::chrono::steady_clock::now();
        result = glm53_kda_layer(layer)
            ? attention_kda_page(branch, normalized, rows, layer, attention,
                                 sequence)
            : attention_mla_page(branch, normalized, rows, layer, attention,
                                 sequence);
        if (!result.ok()) return result;
        if (config.phase_profile) {
            const auto elapsed = elapsed_nanoseconds(attention_started);
            graph_attention_block_nanoseconds.fetch_add(
                elapsed, std::memory_order_relaxed);
            if (glm53_kda_layer(layer)) {
                graph_kda_nanoseconds.fetch_add(elapsed,
                                                std::memory_order_relaxed);
            } else {
                graph_mla_nanoseconds.fetch_add(elapsed,
                                                std::memory_order_relaxed);
            }
        }
        for (std::uint32_t row = 0U; row < rows; ++row) {
            const auto stream_begin =
                static_cast<std::size_t>(row) * stream_columns;
            std::vector<float> transitioned(stream_columns);
            result = dsv4_mhc_post_f32(
                transitioned,
                std::span<const float>(branch).subspan(
                    static_cast<std::size_t>(row) * kHidden, kHidden),
                std::span<const float>(streams).subspan(
                    stream_begin, stream_columns),
                mixes[row], kMhc);
            if (!result.ok()) return result;
            round_bf16(transitioned);
            std::copy(transitioned.begin(), transitioned.end(),
                      streams.begin() + static_cast<std::ptrdiff_t>(stream_begin));
        }
        for (std::uint32_t row = 0U; row < rows; ++row) {
            result = mhc_pre(
                std::span<float>(collapsed).subspan(
                    static_cast<std::size_t>(row) * kHidden, kHidden),
                mixes[row],
                std::span<const float>(streams).subspan(
                    static_cast<std::size_t>(row) * stream_columns,
                    stream_columns),
                prefix + "hc_ffn");
            if (!result.ok()) return result;
        }
        result = norm_rows(normalized, collapsed, rows, kHidden,
                           prefix + "post_attention_layernorm.weight");
        if (!result.ok()) return result;
        std::vector<std::uint64_t> route_requests(rows);
        std::vector<std::uint32_t> route_positions(rows);
        const auto position_base = sequence.token_count();
        for (std::uint32_t row = 0U; row < rows; ++row) {
            route_positions[row] = position_base + row;
            route_requests[row] = route_request_key(
                &sequence, route_positions[row]);
        }
        const auto feedforward_started = std::chrono::steady_clock::now();
        result = feedforward_page(branch, normalized, rows, layer, prefix,
                                  route_requests, route_positions, false, true);
        if (!result.ok()) return result;
        if (config.phase_profile) {
            graph_feedforward_block_nanoseconds.fetch_add(
                elapsed_nanoseconds(feedforward_started),
                std::memory_order_relaxed);
        }
        for (std::uint32_t row = 0U; row < rows; ++row) {
            const auto stream_begin =
                static_cast<std::size_t>(row) * stream_columns;
            std::vector<float> transitioned(stream_columns);
            result = dsv4_mhc_post_f32(
                transitioned,
                std::span<const float>(branch).subspan(
                    static_cast<std::size_t>(row) * kHidden, kHidden),
                std::span<const float>(streams).subspan(
                    stream_begin, stream_columns),
                mixes[row], kMhc);
            if (!result.ok()) return result;
            round_bf16(transitioned);
            std::copy(transitioned.begin(), transitioned.end(),
                      streams.begin() + static_cast<std::ptrdiff_t>(stream_begin));
        }
        return result;
    }

    // Independent sequence rows share the layer's resident weights while
    // retaining disjoint recurrent/MLA state. This is the decode batch shape:
    // unlike prompt pages, rows are not causally related to one another.
    [[nodiscard]] ValidationResult forward_layer_sequences_resident(
        std::span<float> streams, std::uint32_t layer,
        std::span<const std::uint32_t> positions,
        std::span<Glm53SequenceState* const> sequences,
        std::span<DeviceSequenceState* const> device_sequences) {
        const auto rows = static_cast<std::uint32_t>(sequences.size());
        const auto stream_columns = static_cast<std::size_t>(kMhc) * kHidden;
        if (!resident_execution_active || rows == 0U || layer >= kLayers ||
            positions.size() != rows || device_sequences.size() != rows ||
            streams.size() != static_cast<std::size_t>(rows) * stream_columns ||
            std::any_of(device_sequences.begin(), device_sequences.end(),
                        [](const auto* state) {
                            return state == nullptr || !state->ready;
                        })) {
            return {{"GLM-5.3 resident sequence batch is not admissible"}};
        }
        const auto device = device_for(layer);
        const auto prefix = "model.language_model.layers." +
                            std::to_string(layer) + ".";
        const auto attention = prefix + "self_attn.";
        ValidationResult result;
        for (std::uint32_t row = 0U; row < rows; ++row) {
            result = cuda.dsv4_mhc_select_slot(device, row);
            if (!result.ok()) return result;
            result = cuda.dsv4_mhc_begin_device(
                device, resident_layers[layer].attention,
                streams.subspan(static_cast<std::size_t>(row) * stream_columns,
                                stream_columns));
            if (!result.ok()) return result;
        }
        const auto attention_started = std::chrono::steady_clock::now();
        for (std::uint32_t row = 0U; row < rows; ++row) {
            result = cuda.dsv4_mhc_select_slot(device, row);
            if (!result.ok()) return result;
            if (glm53_kda_layer(layer)) {
                CudaGlm53KdaRequest request;
                request.state = &device_sequences[row]->kda[layer];
                request.heads = kHeads;
                request.head_dim = kLinearHead;
                request.convolution_kernel = 4U;
                request.mhc_source_destination = true;
                result = weights->kda_decode(
                    slot_for(layer), attention, request, {});
            } else if (resident_mla_host_attention()) {
                std::vector<float> normalized(kHidden), attended(kHidden);
                result = cuda.dsv4_mhc_download_layer_input(device, normalized);
                if (!result.ok()) return result;
                result = attention_mla(attended, normalized, layer,
                                       positions[row], attention,
                                       *sequences[row]);
                if (!result.ok()) return result;
                result = cuda.dsv4_mhc_publish_branch(device, attended);
            } else if (sparse_indexer_active(
                           config.maximum_context_tokens)) {
                Glm53SparseMlaMetrics sparse_timing;
                CudaGlm53MlaRequest::HostTiming cuda_host_timing;
                auto* const sparse_timing_ptr = config.phase_profile
                    ? &sparse_timing
                    : nullptr;
                if (sparse_timing_ptr != nullptr) sparse_timing.calls = 1U;
                CudaGlm53MlaRequest request;
                request.state = &device_sequences[row]->mla[layer];
                request.sparse_expanded =
                    &device_sequences[row]->sparse_mla_expanded[layer];
                request.position = positions[row];
                request.maximum_context = config.maximum_context_tokens;
                request.heads = kHeads;
                request.head_dim = kMlaHead;
                request.query_rank = kQueryRank;
                request.key_value_rank = kKvRank;
                request.host_timing = config.phase_profile
                    ? &cuda_host_timing
                    : nullptr;
                static_cast<void>(glm53_grow(sparse_index_input, kHidden));
                auto normalized = std::span<float>(sparse_index_input)
                    .first(kHidden);
                const auto input_download_started = config.phase_profile
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};
                result = cuda.dsv4_mhc_download_layer_input(
                    device, normalized);
                if (!result.ok()) return result;
                if (sparse_timing_ptr != nullptr) {
                    sparse_timing.input_download_nanoseconds +=
                        elapsed_nanoseconds(input_download_started);
                }
                result = prepare_sparse_device_selection(
                    mla_selected_positions, mla_arena_rows,
                    mla_expansion_source_positions,
                    mla_expansion_destination_rows, normalized, layer,
                    positions[row], attention, *sequences[row],
                    *device_sequences[row], sparse_timing_ptr);
                if (!result.ok()) return result;
                auto& sparse_arena =
                    device_sequences[row]->sparse_mla_arenas[layer];
                request.validate_sparse_identity_rows =
                    sparse_arena.seed_validation_pending;
                request.sparse_identity_rows = sparse_arena.identity_rows;
                request.selected_positions = mla_selected_positions;
                request.sparse_arena_rows = mla_arena_rows;
                request.sparse_expansion_sources =
                    mla_expansion_source_positions;
                request.sparse_expansion_destinations =
                    mla_expansion_destination_rows;
                const auto history = positions[row] + 1U;
                const auto attended_rows = request.selected_positions.empty()
                    ? static_cast<std::size_t>(history)
                    : request.selected_positions.size();
                static_cast<void>(glm53_grow(
                    mla_softmax_scores,
                    static_cast<std::size_t>(kHeads) * attended_rows));
                auto scores = std::span<float>(mla_softmax_scores).first(
                    static_cast<std::size_t>(kHeads) * attended_rows);
                result = weights->sparse_mla_decode_mhc(
                    slot_for(layer), attention, request, scores,
                    [](std::span<float> values, std::uint32_t heads,
                       std::uint32_t tokens) {
                        for (std::uint32_t head = 0U; head < heads; ++head) {
                            auto* row_scores = values.data() +
                                static_cast<std::size_t>(head) * tokens;
                            float highest =
                                -std::numeric_limits<float>::infinity();
                            for (std::uint32_t token = 0U; token < tokens;
                                 ++token) {
                                highest = std::max(highest,
                                                   row_scores[token]);
                            }
                            float total = 0.0F;
                            for (std::uint32_t token = 0U; token < tokens;
                                 ++token) {
                                row_scores[token] =
                                    std::exp(row_scores[token] - highest);
                                total += row_scores[token];
                            }
                            for (std::uint32_t token = 0U; token < tokens;
                                 ++token) {
                                row_scores[token] = bf16_round_f32(
                                    row_scores[token] / total);
                            }
                        }
                    });
                if (result.ok() && request.validate_sparse_identity_rows) {
                    sparse_arena.seed_validation_pending = false;
                }
                if (sparse_timing_ptr != nullptr) {
                    sparse_timing.index_upload_nanoseconds +=
                        cuda_host_timing.index_upload_nanoseconds;
                    sparse_timing.device_scores_wait_nanoseconds +=
                        cuda_host_timing.device_scores_wait_nanoseconds;
                    sparse_timing.host_softmax_nanoseconds +=
                        cuda_host_timing.host_softmax_nanoseconds;
                    sparse_timing.coefficient_upload_nanoseconds +=
                        cuda_host_timing.coefficient_upload_nanoseconds;
                    add_sparse_mla_metrics(sparse_timing);
                }
            } else {
                CudaGlm53MlaRequest request;
                request.state = &device_sequences[row]->mla[layer];
                request.position = positions[row];
                request.maximum_context = config.maximum_context_tokens;
                request.heads = kHeads;
                request.head_dim = kMlaHead;
                request.query_rank = kQueryRank;
                request.key_value_rank = kKvRank;
                const auto history = positions[row] + 1U;
                auto scores = std::span<float>(mla_softmax_scores).first(
                    static_cast<std::size_t>(kHeads) * history);
                result = weights->mla_decode_mhc(
                    slot_for(layer), attention, request, scores,
                    [](std::span<float> values, std::uint32_t heads,
                       std::uint32_t tokens) {
                        for (std::uint32_t head = 0U; head < heads; ++head) {
                            auto* row_scores = values.data() +
                                static_cast<std::size_t>(head) * tokens;
                            float highest =
                                -std::numeric_limits<float>::infinity();
                            for (std::uint32_t token = 0U; token < tokens;
                                 ++token) {
                                highest = std::max(highest,
                                                   row_scores[token]);
                            }
                            float total = 0.0F;
                            for (std::uint32_t token = 0U; token < tokens;
                                 ++token) {
                                row_scores[token] =
                                    std::exp(row_scores[token] - highest);
                                total += row_scores[token];
                            }
                            for (std::uint32_t token = 0U; token < tokens;
                                 ++token) {
                                row_scores[token] = bf16_round_f32(
                                    row_scores[token] / total);
                            }
                        }
                    });
            }
            if (!result.ok()) return result;
        }
        if (config.phase_profile) {
            const auto elapsed = elapsed_nanoseconds(attention_started);
            graph_attention_block_nanoseconds.fetch_add(
                elapsed, std::memory_order_relaxed);
            (glm53_kda_layer(layer) ? graph_kda_nanoseconds
                                    : graph_mla_nanoseconds)
                .fetch_add(elapsed, std::memory_order_relaxed);
        }
        std::vector<float> normalized(
            static_cast<std::size_t>(rows) * kHidden);
        std::vector<float> branch(normalized.size());
        for (std::uint32_t row = 0U; row < rows; ++row) {
            result = cuda.dsv4_mhc_select_slot(device, row);
            if (!result.ok()) return result;
            result = cuda.dsv4_mhc_transition_next_device(
                device, resident_layers[layer].feedforward);
            if (!result.ok()) return result;
            result = cuda.dsv4_mhc_download_layer_input(
                device, std::span<float>(normalized).subspan(
                            static_cast<std::size_t>(row) * kHidden, kHidden));
            if (!result.ok()) return result;
        }
        std::vector<std::uint64_t> route_requests(rows);
        for (std::uint32_t row = 0U; row < rows; ++row) {
            route_requests[row] = route_request_key(sequences[row],
                                                    positions[row]);
        }
        const auto feedforward_started = std::chrono::steady_clock::now();
        result = feedforward_page(branch, normalized, rows, layer, prefix,
                                  route_requests, positions, false, false,
                                  true);
        if (!result.ok()) return result;
        if (config.phase_profile) {
            graph_feedforward_block_nanoseconds.fetch_add(
                elapsed_nanoseconds(feedforward_started),
                std::memory_order_relaxed);
        }
        for (std::uint32_t row = 0U; row < rows; ++row) {
            result = cuda.dsv4_mhc_select_slot(device, row);
            if (!result.ok()) return result;
            result = cuda.dsv4_mhc_finish(
                device,
                std::span<const float>(branch).subspan(
                    static_cast<std::size_t>(row) * kHidden, kHidden),
                streams.subspan(static_cast<std::size_t>(row) * stream_columns,
                                stream_columns));
            if (!result.ok()) return result;
        }
        return cuda.dsv4_mhc_select_slot(device, 0U);
    }

    [[nodiscard]] ValidationResult forward_layer_sequences(
        std::span<float> streams, std::uint32_t layer,
        std::span<const std::uint32_t> positions,
        std::span<Glm53SequenceState* const> sequences,
        std::span<DeviceSequenceState* const> device_sequences) {
        ValidationResult result;
        const auto rows = static_cast<std::uint32_t>(sequences.size());
        const auto stream_columns = static_cast<std::size_t>(kMhc) * kHidden;
        if (rows == 0U || positions.size() != rows ||
            device_sequences.size() != rows ||
            streams.size() != static_cast<std::size_t>(rows) * stream_columns) {
            return {{"GLM-5.3 independent layer batch has an invalid shape"}};
        }
        if (rows > 1U && resident_execution_active &&
            std::all_of(device_sequences.begin(), device_sequences.end(),
                        [](const auto* state) {
                            return state != nullptr && state->ready;
                        })) {
            return forward_layer_sequences_resident(
                streams, layer, positions, sequences, device_sequences);
        }
        if (rows == 1U && device_sequences.front() != nullptr &&
            resident_execution_active &&
            (glm53_kda_layer(layer) || resident_mla_enabled()) &&
            (glm53_moe_layer(layer) || host_moe_active)) {
            return forward_layer_resident(
                streams, layer, positions.front(), *sequences.front(),
                *device_sequences.front());
        }
        const auto hidden_elements = static_cast<std::size_t>(rows) * kHidden;
        std::vector<float> collapsed(hidden_elements), normalized(hidden_elements),
            branch(hidden_elements);
        std::vector<Dsv4MhcMix> mixes(rows);
        const auto prefix = "model.language_model.layers." +
                            std::to_string(layer) + ".";
        for (std::uint32_t row = 0U; row < rows; ++row) {
            result = mhc_pre(
                std::span<float>(collapsed).subspan(
                    static_cast<std::size_t>(row) * kHidden, kHidden),
                mixes[row],
                std::span<const float>(streams).subspan(
                    static_cast<std::size_t>(row) * stream_columns,
                    stream_columns),
                prefix + "hc_attn");
            if (!result.ok()) return result;
        }
        result = norm_rows(normalized, collapsed, rows, kHidden,
                           prefix + "input_layernorm.weight");
        if (!result.ok()) return result;
        const auto attention = prefix + "self_attn.";
        const auto attention_started = std::chrono::steady_clock::now();
        for (std::uint32_t row = 0U; row < rows; ++row) {
            auto destination = std::span<float>(branch).subspan(
                static_cast<std::size_t>(row) * kHidden, kHidden);
            const auto input = std::span<const float>(normalized).subspan(
                static_cast<std::size_t>(row) * kHidden, kHidden);
            result = glm53_kda_layer(layer)
                ? attention_kda(destination, input, layer, attention,
                                *sequences[row],
                                device_sequences[row] == nullptr
                                    ? nullptr
                                    : &device_sequences[row]->kda[layer])
                : attention_mla(destination, input, layer, positions[row],
                                attention, *sequences[row]);
            if (!result.ok()) return result;
        }
        if (config.phase_profile) {
            const auto elapsed = elapsed_nanoseconds(attention_started);
            graph_attention_block_nanoseconds.fetch_add(
                elapsed, std::memory_order_relaxed);
            if (glm53_kda_layer(layer)) {
                graph_kda_nanoseconds.fetch_add(elapsed,
                                                std::memory_order_relaxed);
            } else {
                graph_mla_nanoseconds.fetch_add(elapsed,
                                                std::memory_order_relaxed);
            }
        }
        for (std::uint32_t row = 0U; row < rows; ++row) {
            const auto stream_begin =
                static_cast<std::size_t>(row) * stream_columns;
            std::vector<float> transitioned(stream_columns);
            result = dsv4_mhc_post_f32(
                transitioned,
                std::span<const float>(branch).subspan(
                    static_cast<std::size_t>(row) * kHidden, kHidden),
                std::span<const float>(streams).subspan(
                    stream_begin, stream_columns),
                mixes[row], kMhc);
            if (!result.ok()) return result;
            round_bf16(transitioned);
            std::copy(transitioned.begin(), transitioned.end(),
                      streams.begin() + static_cast<std::ptrdiff_t>(stream_begin));
        }
        for (std::uint32_t row = 0U; row < rows; ++row) {
            result = mhc_pre(
                std::span<float>(collapsed).subspan(
                    static_cast<std::size_t>(row) * kHidden, kHidden),
                mixes[row],
                std::span<const float>(streams).subspan(
                    static_cast<std::size_t>(row) * stream_columns,
                    stream_columns),
                prefix + "hc_ffn");
            if (!result.ok()) return result;
        }
        result = norm_rows(normalized, collapsed, rows, kHidden,
                           prefix + "post_attention_layernorm.weight");
        if (!result.ok()) return result;
        std::vector<std::uint64_t> route_requests(rows);
        for (std::uint32_t row = 0U; row < rows; ++row) {
            route_requests[row] = route_request_key(sequences[row],
                                                    positions[row]);
        }
        const auto feedforward_started = std::chrono::steady_clock::now();
        result = feedforward_page(branch, normalized, rows, layer, prefix,
                                  route_requests, positions, true, false);
        if (!result.ok()) return result;
        if (config.phase_profile) {
            graph_feedforward_block_nanoseconds.fetch_add(
                elapsed_nanoseconds(feedforward_started),
                std::memory_order_relaxed);
        }
        for (std::uint32_t row = 0U; row < rows; ++row) {
            const auto stream_begin =
                static_cast<std::size_t>(row) * stream_columns;
            std::vector<float> transitioned(stream_columns);
            result = dsv4_mhc_post_f32(
                transitioned,
                std::span<const float>(branch).subspan(
                    static_cast<std::size_t>(row) * kHidden, kHidden),
                std::span<const float>(streams).subspan(
                    stream_begin, stream_columns),
                mixes[row], kMhc);
            if (!result.ok()) return result;
            round_bf16(transitioned);
            std::copy(transitioned.begin(), transitioned.end(),
                      streams.begin() + static_cast<std::ptrdiff_t>(stream_begin));
        }
        return result;
    }

    [[nodiscard]] ValidationResult collapse_streams_page(
        std::span<const float> streams, std::uint32_t rows,
        std::span<float> collapsed) const {
        const auto stream_columns = static_cast<std::size_t>(kMhc) * kHidden;
        if (rows == 0U ||
            streams.size() != static_cast<std::size_t>(rows) * stream_columns ||
            collapsed.size() != static_cast<std::size_t>(rows) * kHidden) {
            return {{"GLM-5.3 residual collapse has an invalid shape"}};
        }
        for (std::uint32_t row = 0U; row < rows; ++row) {
            const auto stream_base =
                static_cast<std::size_t>(row) * stream_columns;
            const auto hidden_base = static_cast<std::size_t>(row) * kHidden;
            for (std::size_t column = 0U; column < kHidden; ++column) {
                collapsed[hidden_base + column] = 0.25F *
                    (streams[stream_base + column] +
                     streams[stream_base + kHidden + column] +
                     streams[stream_base + 2U * kHidden + column] +
                     streams[stream_base + 3U * kHidden + column]);
            }
        }
        round_bf16(collapsed);
        return {};
    }

    [[nodiscard]] ValidationResult finish_streams(
        std::span<const float> streams, std::span<float> logits) {
        ValidationResult result;
        if (streams.size() != static_cast<std::size_t>(kMhc) * kHidden ||
            logits.empty()) {
            result.errors.emplace_back(
                "GLM-5.3 final text state has an invalid shape");
            return result;
        }
        std::vector<float> collapsed(kHidden), normalized(kHidden);
        result = collapse_streams_page(streams, 1U, collapsed);
        if (!result.ok()) return result;
        result = norm(normalized, collapsed, "model.language_model.norm.weight");
        if (!result.ok()) return result;
        if (lm_head_ranges.size() > 1U && projection_workers != nullptr &&
            lm_head_ranges.size() == devices.size() &&
            logits.size() == kVocabulary) {
            std::vector<ValidationResult> shard_results(devices.size());
            auto dispatched = projection_workers->parallel_for_addressed(
                devices.size(), [&](std::size_t slot) {
                    const auto range = lm_head_ranges[slot];
                    const Glm53WeightCache::LinearRequest request{
                        "lm_head", range.count, kHidden, normalized, 1U,
                        logits.subspan(static_cast<std::size_t>(range.begin),
                                       static_cast<std::size_t>(range.count)),
                        true, kVocabulary, range.begin};
                    shard_results[slot] = weights->matmul_batch(
                        slot, std::span<const Glm53WeightCache::LinearRequest>(
                                  &request, 1U));
                });
            if (!dispatched.ok()) return dispatched;
            for (auto& shard_result : shard_results) {
                append(result.errors, std::move(shard_result.errors));
            }
            if (result.ok()) {
                tensor_parallel_head_batches.fetch_add(
                    1U, std::memory_order_relaxed);
            }
            return result;
        }
        return linear("lm_head", normalized, 1U, kHidden, logits, kLayers - 1U);
    }

    [[nodiscard]] ValidationResult project_lm_head_page(
        std::span<const float> normalized, std::uint32_t rows,
        std::span<float> logits) {
        if (rows == 0U || normalized.size() !=
                static_cast<std::size_t>(rows) * kHidden ||
            logits.size() != static_cast<std::size_t>(rows) * kVocabulary) {
            return {{"GLM-5.3 LM-head page has an invalid shape"}};
        }
        if (lm_head_ranges.size() > 1U && projection_workers != nullptr &&
            lm_head_ranges.size() == devices.size()) {
            std::vector<std::vector<float>> shards(devices.size());
            std::vector<ValidationResult> shard_results(devices.size());
            auto dispatched = projection_workers->parallel_for_addressed(
                devices.size(), [&](std::size_t slot) {
                    const auto range = lm_head_ranges[slot];
                    shards[slot].resize(static_cast<std::size_t>(rows) *
                                        range.count);
                    const Glm53WeightCache::LinearRequest request{
                        "lm_head", range.count, kHidden, normalized, rows,
                        shards[slot], true, kVocabulary, range.begin};
                    shard_results[slot] = weights->matmul_batch(
                        slot, std::span<const Glm53WeightCache::LinearRequest>(
                                  &request, 1U));
                });
            if (!dispatched.ok()) return dispatched;
            for (std::size_t slot = 0U; slot < devices.size(); ++slot) {
                if (!shard_results[slot].ok()) return shard_results[slot];
                const auto range = lm_head_ranges[slot];
                for (std::uint32_t row = 0U; row < rows; ++row) {
                    std::copy_n(
                        shards[slot].begin() + static_cast<std::ptrdiff_t>(
                            static_cast<std::size_t>(row) * range.count),
                        range.count,
                        logits.begin() + static_cast<std::ptrdiff_t>(
                            static_cast<std::size_t>(row) * kVocabulary +
                            range.begin));
                }
            }
            tensor_parallel_head_batches.fetch_add(1U,
                                                    std::memory_order_relaxed);
            return {};
        }
        return linear("lm_head", normalized, rows, kHidden, logits,
                      kLayers - 1U);
    }

    [[nodiscard]] ValidationResult finish_streams_page(
        std::span<const float> streams, std::uint32_t rows,
        std::span<float> logits) {
        const auto stream_columns = static_cast<std::size_t>(kMhc) * kHidden;
        if (rows == 0U ||
            streams.size() != static_cast<std::size_t>(rows) * stream_columns ||
            logits.size() != static_cast<std::size_t>(rows) * kVocabulary) {
            return {{"GLM-5.3 final sequence batch has an invalid shape"}};
        }
        std::vector<float> collapsed(static_cast<std::size_t>(rows) * kHidden);
        auto result = collapse_streams_page(streams, rows, collapsed);
        if (!result.ok()) return result;
        std::vector<float> normalized(collapsed.size());
        result = norm_rows(normalized, collapsed, rows, kHidden,
                           "model.language_model.norm.weight");
        if (!result.ok()) return result;
        return project_lm_head_page(normalized, rows, logits);
    }

    [[nodiscard]] ValidationResult forward_token_batch(
        std::span<const std::uint32_t> tokens,
        std::span<const std::uint32_t> positions,
        std::span<Glm53SequenceState* const> sequences,
        std::span<DeviceSequenceState* const> device_sequences,
        std::span<float> logits, std::span<float> base_hidden = {}) {
        const auto rows = static_cast<std::uint32_t>(tokens.size());
        if (rows == 0U || positions.size() != rows ||
            sequences.size() != rows || device_sequences.size() != rows ||
            logits.size() != static_cast<std::size_t>(rows) * kVocabulary ||
            (!base_hidden.empty() &&
             base_hidden.size() != static_cast<std::size_t>(rows) * kHidden)) {
            return {{"GLM-5.3 decode batch has an invalid shape"}};
        }
        // FP8's exact device-expert composition has not cleared the
        // independent-row parity gate. Preserve its accepted batch-1 path for
        // every request rather than silently changing tokens or replacing the
        // resident tiers with a slower all-host page. MXFP4 retains the
        // measured expert-major cohort path below.
        if (rows > 1U && fp8_expert_checkpoint()) {
            for (std::uint32_t row = 0U; row < rows; ++row) {
                const auto hidden = base_hidden.empty()
                    ? std::span<float>{}
                    : base_hidden.subspan(
                          static_cast<std::size_t>(row) * kHidden, kHidden);
                auto result = forward_token_batch(
                    tokens.subspan(row, 1U), positions.subspan(row, 1U),
                    sequences.subspan(row, 1U),
                    device_sequences.subspan(row, 1U),
                    logits.subspan(
                        static_cast<std::size_t>(row) * kVocabulary,
                        kVocabulary),
                    hidden);
                if (!result.ok()) return result;
            }
            return {};
        }
        const auto stream_columns = static_cast<std::size_t>(kMhc) * kHidden;
        std::vector<float> streams(static_cast<std::size_t>(rows) *
                                   stream_columns);
        if (config.phase_profile) {
            graph_forward_calls.fetch_add(1U, std::memory_order_relaxed);
            graph_forward_rows.fetch_add(rows, std::memory_order_relaxed);
        }
        const auto embedding_started = std::chrono::steady_clock::now();
        for (std::uint32_t row = 0U; row < rows; ++row) {
            auto result = initialize_streams(
                tokens[row], std::span<float>(streams).subspan(
                    static_cast<std::size_t>(row) * stream_columns,
                    stream_columns));
            if (!result.ok()) return result;
        }
        if (config.phase_profile) {
            graph_embedding_nanoseconds.fetch_add(
                elapsed_nanoseconds(embedding_started),
                std::memory_order_relaxed);
        }
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            const auto layer_started = std::chrono::steady_clock::now();
            auto result = forward_layer_sequences(
                streams, layer, positions, sequences, device_sequences);
            if (!result.ok()) return result;
            if (config.phase_profile) {
                graph_layer_nanoseconds.fetch_add(
                    elapsed_nanoseconds(layer_started),
                    std::memory_order_relaxed);
            }
        }
        if (!base_hidden.empty()) {
            auto collapsed = collapse_streams_page(streams, rows, base_hidden);
            if (!collapsed.ok()) return collapsed;
        }
        const auto output_head_started = std::chrono::steady_clock::now();
        ValidationResult result;
        if (rows > 1U) {
            // As with the router, the output head is not the M7 reuse target.
            // Execute the exact accepted one-row primitive independently so
            // batch membership cannot perturb greedy logits.
            for (std::uint32_t row = 0U; row < rows; ++row) {
                result = finish_streams_page(
                    std::span<float>(streams).subspan(
                        static_cast<std::size_t>(row) * stream_columns,
                        stream_columns),
                    1U,
                    logits.subspan(static_cast<std::size_t>(row) * kVocabulary,
                                   kVocabulary));
                if (!result.ok()) break;
            }
        } else {
            result = finish_streams_page(streams, rows, logits);
        }
        if (config.phase_profile) {
            graph_output_head_nanoseconds.fetch_add(
                elapsed_nanoseconds(output_head_started),
                std::memory_order_relaxed);
        }
        if (result.ok()) {
            for (std::uint32_t row = 0U; row < rows; ++row) {
                sequences[row]->set_token_count(positions[row] + 1U);
            }
        }
        return result;
    }

    [[nodiscard]] ValidationResult forward_mtp(
        std::uint32_t next_token, std::span<const float> previous_hidden,
        std::uint32_t position, Glm53SequenceState& sequence,
        std::span<float> logits, std::span<float> feedback_hidden) {
        if (previous_hidden.size() != kHidden || logits.size() != kVocabulary ||
            feedback_hidden.size() != kHidden ||
            sequence.mla(kMtpLayer).rows() != position) {
            return {{"GLM-5.3 MTP command has an invalid sequence shape"}};
        }
        const std::string prefix = "model.language_model.layers.45.";
        auto embedding = checkpoint->read_f32_row(
            "model.language_model.embed_tokens.weight", next_token);
        if (!embedding.ok()) return {std::move(embedding.errors)};
        if (position == 0U) {
            std::fill(embedding.value.begin(), embedding.value.end(), 0.0F);
        }
        std::vector<float> normalized_embedding(kHidden);
        std::vector<float> normalized_hidden(kHidden);
        auto result = norm(normalized_embedding, embedding.value,
                           prefix + "enorm.weight");
        if (!result.ok()) return result;
        result = norm(normalized_hidden, previous_hidden,
                      prefix + "hnorm.weight");
        if (!result.ok()) return result;
        std::vector<float> fused(static_cast<std::size_t>(2U) * kHidden);
        std::copy(normalized_embedding.begin(), normalized_embedding.end(),
                  fused.begin());
        std::copy(normalized_hidden.begin(), normalized_hidden.end(),
                  fused.begin() + kHidden);
        std::vector<float> hidden(kHidden);
        result = linear(prefix + "eh_proj", fused, 1U, 2U * kHidden,
                        hidden, kMtpLayer);
        if (!result.ok()) return result;

        // The standalone MTP block deliberately disables mHC and follows the
        // ordinary BF16 residual contract: residual <- input, attention,
        // add+norm, MoE, final add. Its shared head then applies its own norm
        // before reusing the target LM head.
        std::vector<float> residual = hidden;
        std::vector<float> normalized(kHidden), branch(kHidden);
        result = norm(normalized, hidden, prefix + "input_layernorm.weight");
        if (!result.ok()) return result;
        result = attention_mla(branch, normalized, kMtpLayer, position,
                               prefix + "self_attn.", sequence);
        if (!result.ok()) return result;
        for (std::size_t index = 0U; index < kHidden; ++index) {
            residual[index] = bf16_round_f32(residual[index] + branch[index]);
        }
        result = norm(normalized, residual,
                      prefix + "post_attention_layernorm.weight");
        if (!result.ok()) return result;
        result = feedforward(branch, normalized, kMtpLayer, prefix);
        if (!result.ok()) return result;
        for (std::size_t index = 0U; index < kHidden; ++index) {
            feedback_hidden[index] =
                bf16_round_f32(residual[index] + branch[index]);
        }
        result = norm(normalized, feedback_hidden,
                      prefix + "shared_head.norm.weight");
        if (!result.ok()) return result;
        result = project_lm_head_page(normalized, 1U, logits);
        return result;
    }

    [[nodiscard]] ValidationResult prepare_mtp_prompt(
        std::span<const std::uint32_t> prompt,
        std::span<const float> base_hidden, Glm53SequenceState& sequence) {
        if (base_hidden.size() != prompt.size() * kHidden) {
            return {{"GLM-5.3 MTP prefill hidden-state extent is invalid"}};
        }
        auto& cache = sequence.mla(kMtpLayer);
        const auto required_rows = prompt.empty() ? 0U :
            static_cast<std::uint32_t>(prompt.size() - 1U);
        if (cache.rows() > required_rows) {
            return {{"GLM-5.3 MTP prefix state is ahead of the prompt"}};
        }
        std::vector<float> ignored_logits(kVocabulary);
        std::vector<float> feedback(kHidden);
        for (std::uint32_t position = cache.rows(); position < required_rows;
             ++position) {
            auto result = forward_mtp(
                prompt[position + 1U],
                base_hidden.subspan(static_cast<std::size_t>(position) * kHidden,
                                    kHidden),
                position, sequence, ignored_logits, feedback);
            if (!result.ok()) return result;
        }
        return {};
    }

    [[nodiscard]] ValidationResult forward_token(
        std::uint32_t token, std::uint32_t position,
        std::span<float> logits, Glm53SequenceState& sequence) {
        ValidationResult result;
        std::vector<float> streams(static_cast<std::size_t>(kMhc) * kHidden);
        result = initialize_streams(token, streams);
        if (!result.ok()) return result;
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            result = forward_layer(streams, layer, position, sequence);
            if (!result.ok()) return result;
            if (config.load_progress) {
                std::cerr << "\r[glm53] layer " << (layer + 1U) << '/' << kLayers
                          << std::flush;
            }
        }
        if (config.load_progress) {
            std::cerr << '\r' << std::string(32U, ' ') << '\r';
        }
        result = finish_streams(streams, logits);
        if (result.ok()) sequence.set_token_count(position + 1U);
        return result;
    }

    [[nodiscard]] ValidationResult forward_prompt(
        std::span<const std::uint32_t> tokens, std::span<float> logits,
        Glm53SequenceState& sequence,
        std::vector<float>* base_hidden_rows = nullptr,
        bool all_row_logits = false,
        DeviceSequenceState* device_sequence = nullptr) {
        ValidationResult result;
        if (tokens.empty() ||
            logits.size() != (all_row_logits
                ? tokens.size() * kVocabulary : kVocabulary)) {
            result.errors.emplace_back("GLM-5.3 prefill has an invalid shape");
            return result;
        }
        const auto position_base = sequence.token_count();
        const auto stream_columns = static_cast<std::size_t>(kMhc) * kHidden;
        std::vector<float> streams(tokens.size() * stream_columns);
        std::vector<ValidationResult> encode_results(tokens.size());
        if (config.phase_profile) {
            graph_forward_calls.fetch_add(1U, std::memory_order_relaxed);
            graph_forward_rows.fetch_add(tokens.size(),
                                         std::memory_order_relaxed);
        }
        const auto embedding_started = std::chrono::steady_clock::now();
        const auto encode = [&](std::size_t position) {
            encode_results[position] = initialize_streams(
                tokens[position],
                std::span<float>(streams).subspan(
                    position * stream_columns, stream_columns));
        };
        if (phase_scheduler_enabled() && kda_workers != nullptr &&
            tokens.size() >= kda_workers->size()) {
            result = kda_workers->parallel_for(tokens.size(), encode);
            if (!result.ok()) return result;
            parallel_encode_pages.fetch_add(1U, std::memory_order_relaxed);
        } else {
            for (std::size_t position = 0U; position < tokens.size();
                 ++position) {
                encode(position);
            }
        }
        for (auto& encoded : encode_results) {
            if (!encoded.ok()) return encoded;
        }
        if (config.phase_profile) {
            graph_embedding_nanoseconds.fetch_add(
                elapsed_nanoseconds(embedding_started),
                std::memory_order_relaxed);
        }
        // Prompt rows are page/layer-major. Recurrent KDA and causal MLA state
        // still advance in token order inside each layer, while the active
        // layer's routed experts remain reusable in the bounded CUDA cache.
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            const auto layer_started = std::chrono::steady_clock::now();
            result = device_sequence == nullptr
                ? forward_layer_page(
                      streams, static_cast<std::uint32_t>(tokens.size()),
                      layer, sequence)
                : forward_layer_page_device(
                      streams, static_cast<std::uint32_t>(tokens.size()),
                      layer, sequence, *device_sequence);
            if (!result.ok()) return result;
            if (config.phase_profile) {
                graph_layer_nanoseconds.fetch_add(
                    elapsed_nanoseconds(layer_started),
                    std::memory_order_relaxed);
            }
            if (config.load_progress) {
                std::cerr << "\r[glm53-prefill] layer " << (layer + 1U) << '/'
                          << kLayers << " rows " << tokens.size() << std::flush;
            }
        }
        if (config.load_progress) {
            std::cerr << '\r' << std::string(48U, ' ') << '\r';
        }
        if (base_hidden_rows != nullptr) {
            const auto old_size = base_hidden_rows->size();
            base_hidden_rows->resize(old_size + tokens.size() * kHidden);
            result = collapse_streams_page(
                streams, static_cast<std::uint32_t>(tokens.size()),
                std::span<float>(*base_hidden_rows).subspan(old_size));
            if (!result.ok()) return result;
        }
        const auto output_head_started = std::chrono::steady_clock::now();
        result = all_row_logits
            ? finish_streams_page(
                  streams, static_cast<std::uint32_t>(tokens.size()), logits)
            : finish_streams(
                  std::span<const float>(streams).last(stream_columns), logits);
        if (config.phase_profile) {
            graph_output_head_nanoseconds.fetch_add(
                elapsed_nanoseconds(output_head_started),
                std::memory_order_relaxed);
        }
        if (result.ok()) {
            sequence.set_token_count(
                position_base + static_cast<std::uint32_t>(tokens.size()));
        }
        return result;
    }

    void complete_request(const std::shared_ptr<ScheduledRequest>& request) {
        if (request->decoding) {
            request->result.metrics.decode_seconds =
                now_seconds() - request->decode_started;
            if (request->streamed != nullptr) {
                request->streamed->finish(request->on_token);
                request->result.text = request->streamed->text();
                request->result.stopped = request->result.stopped ||
                                          request->streamed->stopped();
            }
            const auto cache = weights->stats();
            std::cerr << "[glm53-decode-cache] misses="
                      << (cache.misses - request->decode_cache_start.misses)
                      << " evictions="
                      << (cache.evictions -
                          request->decode_cache_start.evictions)
                      << " useful_prefetches="
                      << (cache.useful_prefetches -
                          request->decode_cache_start.useful_prefetches)
                      << '\n';
            if (!route_census_path().empty()) {
                std::scoped_lock guard(route_census_mutex);
                std::ofstream census(route_census_path(), std::ios::trunc);
                if (census) {
                    census << "phase\trequest\tposition\tlayer\texperts\n";
                    for (const auto& row : route_census) {
                        census << (row.prefill ? "prefill" : "decode") << '\t'
                               << row.request << '\t' << row.position << '\t'
                               << row.layer << '\t';
                        for (std::size_t index = 0U;
                             index < row.experts.size(); ++index) {
                            if (index != 0U) census << ',';
                            census << row.experts[index];
                        }
                        census << '\n';
                    }
                }
                std::cerr << "[glm53-route-census] rows="
                          << route_census.size() << " path="
                          << route_census_path()
                          << (census ? " written" : " FAILED") << '\n';
            }
            if (shared_experts.active) {
                const auto missed = shared_expert_host_layers.load(
                    std::memory_order_relaxed);
                std::cerr << "[glm53-shared-expert] device_layers="
                          << (shared_experts.storage.size() / 6U)
                          << " device_calls="
                          << shared_expert_device_calls.load(
                                 std::memory_order_relaxed)
                          << " host_calls="
                          << shared_expert_host_calls.load(
                                 std::memory_order_relaxed)
                          << " host_path_layers=";
                if (missed == 0U) {
                    std::cerr << "none";
                } else {
                    const char* separator = "";
                    for (std::uint32_t layer = 0U; layer < 64U; ++layer) {
                        if ((missed >> layer & 1U) == 0U) continue;
                        std::cerr << separator << layer;
                        separator = ",";
                    }
                }
                std::cerr << '\n';
            }
            if (static_experts.active_tier) {
                const auto hits = static_experts.route_hits.load(
                    std::memory_order_relaxed) -
                    request->decode_static_hits_start;
                const auto misses = static_experts.route_misses.load(
                    std::memory_order_relaxed) -
                    request->decode_static_misses_start;
                const auto routes = hits + misses;
                std::cerr << "[glm53-static-tier] decode_route_hits=" << hits
                          << " decode_route_misses=" << misses
                          << " decode_route_coverage="
                          << (routes == 0U ? 0.0
                                          : static_cast<double>(hits) /
                                                static_cast<double>(routes))
                          << '\n';
            }
        }
        if (config.phase_profile && request->prepared) {
            const auto profile_after_decode = profile_snapshot();
            request->result.metrics.decode = phase_delta(
                profile_after_decode, request->profile_decode_started);
        }
        request->result.metrics.rss_bytes = process_resident_set_bytes();
        request->result.metrics.device_vram_used_bytes =
            device_vram_used_bytes(devices);
        request->result.metrics.phase_profile = config.phase_profile;
        // Release the scheduler's scarce persistent CUDA slot before waking
        // the caller. Otherwise a completed request remains alive in the
        // waiting API thread briefly after it leaves `active_requests`, and a
        // replacement can transiently exceed the capacity ledger by one full
        // sequence.
        if (request->device_sequence.ready) {
            sequence_slots_live.fetch_sub(1U, std::memory_order_relaxed);
        }
        request->device_sequence = DeviceSequenceState{};
        {
            std::scoped_lock lock(request->completion_mutex);
            request->done = true;
        }
        request->completion.notify_all();
    }

    [[nodiscard]] bool prepare_request(
        const std::shared_ptr<ScheduledRequest>& request) {
        auto warm = wait_for_warmup();
        if (!warm.ok()) {
            request->result.errors = std::move(warm.errors);
            complete_request(request);
            return false;
        }
        auto reset = reset_sequence(request->sequence);
        if (!reset.ok()) {
            request->result.errors = std::move(reset.errors);
            complete_request(request);
            return false;
        }
        request->result.prompt_token_ids = request->prompt;
        request->result.metrics.prompt_tokens = request->prompt.size();
        request->logits.resize(kVocabulary);
        const auto reused = restore_prefix(
            request->prompt, request->sequence, request->logits,
            request->base_hidden);
        request->result.metrics.reused_prompt_tokens = reused;
        request->prefill_cursor = reused;
        if (device_prefill_for_context(config.maximum_context_tokens)) {
            if (!resident_execution_active) {
                request->result.errors.emplace_back(
                    "GLM-5.3 device prefill requires the resident exact path");
                complete_request(request);
                return false;
            }
            auto prepared = prepare_device_sequence(
                request->sequence, request->device_sequence);
            if (!prepared.ok()) {
                request->result.errors = std::move(prepared.errors);
                complete_request(request);
                return false;
            }
            for (const auto device : devices) {
                prepared = cuda.dsv4_mhc_reserve_slots(
                    device, config.prefill_page_tokens);
                if (!prepared.ok()) {
                    request->result.errors = std::move(prepared.errors);
                    complete_request(request);
                    return false;
                }
            }
        }
        request->prefill_started = now_seconds();
        request->profile_started = profile_snapshot();
        request->prepared = true;
        return true;
    }

    [[nodiscard]] ValidationResult prepare_device_mla_layer(
        std::uint32_t layer, Glm53SequenceState& sequence,
        CudaBuffer& buffer) {
        const auto attention = "model.language_model.layers." +
            std::to_string(layer) + ".self_attn.";
        const auto cache_floats =
            static_cast<std::size_t>(config.maximum_context_tokens) * kKvRank;
        std::vector<float> packed(
            cache_floats + kQueryRank + kKvRank, 0.0F);
        const auto latent = sequence.mla(layer).materialize();
        if (latent.size() > cache_floats || latent.size() % kKvRank != 0U) {
            return {{"GLM-5.3 resident MLA cache exceeds its admitted "
                     "context"}};
        }
        std::copy(latent.begin(), latent.end(), packed.begin());
        auto q_norm = host_tensor(
            attention + "q_a_layernorm.weight", kQueryRank);
        auto kv_norm = host_tensor(
            attention + "kv_a_layernorm.weight", kKvRank);
        if (!q_norm.ok() || !kv_norm.ok()) {
            ValidationResult result;
            append(result.errors, std::move(q_norm.errors));
            append(result.errors, std::move(kv_norm.errors));
            return result;
        }
        std::copy(q_norm.value->begin(), q_norm.value->end(),
                  packed.begin() + static_cast<std::ptrdiff_t>(cache_floats));
        std::copy(kv_norm.value->begin(), kv_norm.value->end(),
                  packed.begin() + static_cast<std::ptrdiff_t>(
                                       cache_floats + kQueryRank));

        const auto history = static_cast<std::uint32_t>(
            latent.size() / kKvRank);
        // An empty prompt still needs the admitted BF16 expansion extent:
        // device page MLA appends its first latent after this preparation.
        const bool persistent_bf16 =
            weights->mla_kv_b_is_bf16(attention);
        const auto packed_bytes =
            std::as_bytes(std::span<const float>(packed));
        if (!persistent_bf16) {
            return cuda.upload_buffer(device_for(layer), packed_bytes, buffer);
        }

        constexpr std::uint64_t expanded_width =
            static_cast<std::uint64_t>(kHeads) * 2U * kMlaHead;
        const auto expanded_bytes =
            static_cast<std::uint64_t>(config.maximum_context_tokens) *
            expanded_width * sizeof(std::uint16_t);
        auto result = cuda.allocate_buffer(
            device_for(layer), packed_bytes.size() + expanded_bytes, buffer);
        if (!result.ok()) return result;
        const CudaBufferPatch patch{0U, packed_bytes};
        result = cuda.update_buffer(buffer, std::span(&patch, 1U));
        if (!result.ok()) return result;
        CudaGlm53MlaRequest request;
        request.state = &buffer;
        request.maximum_context = config.maximum_context_tokens;
        request.heads = kHeads;
        request.head_dim = kMlaHead;
        request.query_rank = kQueryRank;
        request.key_value_rank = kKvRank;
        return weights->prepare_mla_history(
            slot_for(layer), attention, request, history);
    }

    [[nodiscard]] ValidationResult prepare_device_sparse_mla_layer(
        std::uint32_t layer, Glm53SequenceState& sequence,
        CudaBuffer& buffer) {
        const auto attention = "model.language_model.layers." +
            std::to_string(layer) + ".self_attn.";
        const auto cache_floats =
            static_cast<std::size_t>(config.maximum_context_tokens) * kKvRank;
        std::vector<float> packed(
            cache_floats + kQueryRank + kKvRank, 0.0F);
        const auto latent = sequence.mla(layer).materialize();
        if (latent.size() > cache_floats || latent.size() % kKvRank != 0U) {
            return {{"GLM-5.3 sparse resident MLA cache exceeds its admitted "
                     "context"}};
        }
        std::copy(latent.begin(), latent.end(), packed.begin());
        auto q_norm = host_tensor(
            attention + "q_a_layernorm.weight", kQueryRank);
        auto kv_norm = host_tensor(
            attention + "kv_a_layernorm.weight", kKvRank);
        if (!q_norm.ok() || !kv_norm.ok()) {
            ValidationResult result;
            append(result.errors, std::move(q_norm.errors));
            append(result.errors, std::move(kv_norm.errors));
            return result;
        }
        std::copy(q_norm.value->begin(), q_norm.value->end(),
                  packed.begin() + static_cast<std::ptrdiff_t>(cache_floats));
        std::copy(kv_norm.value->begin(), kv_norm.value->end(),
                  packed.begin() + static_cast<std::ptrdiff_t>(
                                       cache_floats + kQueryRank));
        return cuda.upload_buffer(
            device_for(layer), std::as_bytes(std::span<const float>(packed)),
            buffer);
    }

    [[nodiscard]] ValidationResult prepare_device_sequence(
        Glm53SequenceState& sequence, DeviceSequenceState& device_sequence) {
        if (device_sequence.ready) return {};
        ValidationResult result;
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            const auto attention = "model.language_model.layers." +
                std::to_string(layer) + ".self_attn.";
            if (!glm53_kda_layer(layer)) {
                // Sparse contexts reserve only their latent cache. Dense MLA
                // keeps its accepted expanded-BF16 history allocation and its
                // preparation path unchanged as the identity control arm.
                if (sparse_indexer_active(config.maximum_context_tokens)) {
                    const auto scratch_bytes =
                        static_cast<std::uint64_t>(kIndexArenaRows) *
                        kHeads * 2ULL * kMlaHead * sizeof(std::uint16_t);
                    result = cuda.allocate_buffer(
                        device_for(layer), scratch_bytes,
                        device_sequence.sparse_mla_expanded[layer]);
                    if (!result.ok()) return result;
                    auto& arena =
                        device_sequence.sparse_mla_arenas[layer];
                    arena.slot_pools.assign(
                        kIndexArenaPools,
                        std::numeric_limits<std::uint32_t>::max());
                    arena.slot_recency.assign(kIndexArenaPools, 0U);
                    result = prepare_device_sparse_mla_layer(
                        layer, sequence, device_sequence.mla[layer]);
                    if (!result.ok()) return result;
                    continue;
                }
                result = prepare_device_mla_layer(
                    layer, sequence, device_sequence.mla[layer]);
                if (!result.ok()) return result;
                continue;
            }
            const std::array<std::string, 3U> tap_names{
                attention + "q_conv1d.weight",
                attention + "k_conv1d.weight",
                attention + "v_conv1d.weight"};
            std::array<std::shared_ptr<const std::vector<float>>, 3U> taps;
            for (std::size_t projection = 0U; projection < taps.size();
                 ++projection) {
                auto loaded = host_tensor(
                    tap_names[projection],
                    static_cast<std::uint64_t>(kLinearWidth) * 4U);
                if (!loaded.ok()) return {std::move(loaded.errors)};
                taps[projection] = std::move(loaded.value);
            }
            auto a_log = host_tensor(attention + "A_log", kHeads);
            auto dt_bias = host_tensor(attention + "dt_bias", kLinearWidth);
            auto o_norm = host_tensor(attention + "o_norm.weight", kLinearHead);
            if (!a_log.ok() || !dt_bias.ok() || !o_norm.ok()) {
                append(result.errors, std::move(a_log.errors));
                append(result.errors, std::move(dt_bias.errors));
                append(result.errors, std::move(o_norm.errors));
                return result;
            }
            const auto recurrent = sequence.recurrent(layer);
            const auto convolution_elements =
                static_cast<std::size_t>(3U) * kLinearWidth * 3U;
            const auto tap_elements =
                static_cast<std::size_t>(3U) * kLinearWidth * 4U;
            std::vector<float> packed(
                recurrent.size() + convolution_elements + tap_elements +
                kHeads + kLinearWidth + kLinearHead + kKdaWorkspaceFloats);
            auto destination = packed.begin();
            destination = std::copy(recurrent.begin(), recurrent.end(),
                                    destination);
            for (std::uint32_t projection = 0U; projection < 3U;
                 ++projection) {
                const auto history = sequence.convolution(layer, projection);
                destination = std::copy(history.begin(), history.end(),
                                        destination);
            }
            for (const auto& tap : taps) {
                destination = std::copy(tap->begin(), tap->end(), destination);
            }
            destination = std::copy(a_log.value->begin(), a_log.value->end(),
                                    destination);
            destination = std::copy(dt_bias.value->begin(), dt_bias.value->end(),
                                    destination);
            static_cast<void>(std::copy(o_norm.value->begin(),
                                        o_norm.value->end(), destination));
            result = cuda.upload_buffer(
                device_for(layer), std::as_bytes(std::span<const float>(packed)),
                device_sequence.kda[layer]);
            if (!result.ok()) return result;
        }
        const auto actual = actual_sequence_device_bytes(device_sequence);
        if (actual != sequence_device_bytes) {
            return {{"GLM-5.3 sequence CUDA allocation disagrees with its "
                     "admission ledger"}};
        }
        if (config.phase_profile) {
            std::cerr << "[glm53-sequence-state] host_private_bytes="
                      << sequence.private_bytes();
            for (std::size_t slot = 0U; slot < devices.size(); ++slot) {
                std::cerr << " cuda" << devices[slot] << "_bytes="
                          << actual[slot];
            }
            std::cerr << '\n';
        }
        device_sequence.ready = true;
        const auto live = sequence_slots_live.fetch_add(
                              1U, std::memory_order_relaxed) + 1U;
        auto peak = sequence_slots_peak.load(std::memory_order_relaxed);
        while (peak < live && !sequence_slots_peak.compare_exchange_weak(
                   peak, live, std::memory_order_relaxed,
                   std::memory_order_relaxed)) {}
        return result;
    }

    [[nodiscard]] ValidationResult synchronize_kda_sequence_from_device(
        Glm53SequenceState& sequence,
        const DeviceSequenceState& device_sequence) {
        if (!device_sequence.ready) {
            return {{"GLM-5.3 device prefill state is not ready"}};
        }
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            if (!glm53_kda_layer(layer)) continue;
            auto recurrent = sequence.recurrent(layer);
            std::array<std::span<float>, 3U> convolution{
                sequence.convolution(layer, 0U),
                sequence.convolution(layer, 1U),
                sequence.convolution(layer, 2U)};
            const auto convolution_elements =
                convolution[0].size() + convolution[1].size() +
                convolution[2].size();
            std::vector<float> packed(
                recurrent.size() + convolution_elements);
            auto downloaded = cuda.download_buffer(
                device_sequence.kda[layer], 0U,
                std::as_writable_bytes(std::span<float>(packed)));
            if (!downloaded.ok()) return downloaded;
            auto source = packed.begin();
            std::copy_n(source, recurrent.size(), recurrent.begin());
            source += static_cast<std::ptrdiff_t>(recurrent.size());
            for (auto destination : convolution) {
                std::copy_n(source, destination.size(), destination.begin());
                source += static_cast<std::ptrdiff_t>(destination.size());
            }
        }
        return {};
    }

    [[nodiscard]] ValidationResult synchronize_mla_layer_from_device(
        Glm53SequenceState& sequence,
        const DeviceSequenceState& device_sequence, std::uint32_t layer) {
        if (!device_sequence.ready) {
            return {{"GLM-5.3 device prefill state is not ready"}};
        }
        if (layer >= kLayers || glm53_kda_layer(layer)) {
            return {{"GLM-5.3 device MLA synchronization layer is invalid"}};
        }
        const auto required_rows = sequence.token_count();
        auto& cache = sequence.mla(layer);
        const auto existing_rows = cache.rows();
        if (existing_rows > required_rows) {
            return {{"GLM-5.3 host MLA state is ahead of device prefill"}};
        }
        const auto missing_rows = required_rows - existing_rows;
        if (missing_rows == 0U) return {};
        std::vector<float> latent(
            static_cast<std::size_t>(missing_rows) * kKvRank);
        const auto offset = static_cast<std::uint64_t>(existing_rows) *
                            kKvRank * sizeof(float);
        auto downloaded = cuda.download_buffer(
            device_sequence.mla[layer], offset,
            std::as_writable_bytes(std::span<float>(latent)));
        if (!downloaded.ok()) return downloaded;
        return cache.append_rows(latent, missing_rows);
    }

    [[nodiscard]] std::uint64_t kda_state_hash(
        const Glm53SequenceState& sequence) const noexcept {
        auto hash = kDiagnosticFnvOffset;
        const auto fold = [&](std::span<const float> values,
                              std::uint64_t& destination) {
            for (const auto value : values) {
                destination = diagnostic_hash_u32(
                    destination, std::bit_cast<std::uint32_t>(value));
            }
        };
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            if (!glm53_kda_layer(layer)) continue;
            hash = diagnostic_hash_u32(hash, layer);
            fold(sequence.recurrent(layer), hash);
            for (std::uint32_t projection = 0U; projection < 3U;
                 ++projection) {
                fold(sequence.convolution(layer, projection), hash);
            }
        }
        hash = diagnostic_hash_u32(hash, sequence.token_count());
        return hash;
    }

    void print_kda_layer_hashes(const Glm53SequenceState& sequence) const {
        const auto span_hash = [](std::span<const float> values) {
            auto hash = kDiagnosticFnvOffset;
            for (const auto value : values) {
                hash = diagnostic_hash_u32(
                    hash, std::bit_cast<std::uint32_t>(value));
            }
            return hash;
        };
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            if (!glm53_kda_layer(layer)) continue;
            auto convolution_hash = kDiagnosticFnvOffset;
            for (std::uint32_t projection = 0U; projection < 3U;
                 ++projection) {
                for (const auto value :
                     sequence.convolution(layer, projection)) {
                    convolution_hash = diagnostic_hash_u32(
                        convolution_hash,
                        std::bit_cast<std::uint32_t>(value));
                }
            }
            std::cerr << "[glm53-prefill-kda-layer] layer=" << layer
                      << " recurrent=" << std::hex
                      << span_hash(sequence.recurrent(layer))
                      << " convolution=" << convolution_hash << std::dec
                      << '\n';
        }
    }

    void print_mla_layer_hashes(const Glm53SequenceState& sequence) const {
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            if (glm53_kda_layer(layer)) continue;
            auto hash = kDiagnosticFnvOffset;
            for (const auto value : sequence.mla(layer).materialize()) {
                hash = diagnostic_hash_u32(
                    hash, std::bit_cast<std::uint32_t>(value));
            }
            std::cerr << "[glm53-prefill-mla-layer] layer=" << layer
                      << " latent=" << std::hex << hash << std::dec << '\n';
        }
    }

    void finish_prefill(const std::shared_ptr<ScheduledRequest>& request) {
        if (device_prefill_for_context(config.maximum_context_tokens)) {
            auto synchronized = synchronize_kda_sequence_from_device(
                request->sequence, request->device_sequence);
            if (!synchronized.ok()) {
                request->result.errors = std::move(synchronized.errors);
                complete_request(request);
                return;
            }
            for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
                if (glm53_kda_layer(layer)) continue;
                const auto attention =
                    "model.language_model.layers." +
                    std::to_string(layer) + ".self_attn.";
                if (device_page_mla_enabled() &&
                    weights->mla_kv_b_is_bf16(attention)) {
                    // Device page MLA owns the exact latent and expanded
                    // histories. Preserve the small latent state in the host
                    // prefix snapshot without replacing the live buffer.
                    synchronized = synchronize_mla_layer_from_device(
                        request->sequence, request->device_sequence, layer);
                } else {
                    // Control and FP8: page MLA ran on the host, so admit its
                    // completed latent history for resident decode now.
                    synchronized = prepare_device_mla_layer(
                        layer, request->sequence,
                        request->device_sequence.mla[layer]);
                }
                if (!synchronized.ok()) {
                    request->result.errors = std::move(synchronized.errors);
                    complete_request(request);
                    return;
                }
            }
        }
        request->result.metrics.prefill_tokens =
            request->prompt.size() -
            request->result.metrics.reused_prompt_tokens;
        request->result.metrics.prefill_seconds =
            now_seconds() - request->prefill_started;
        request->profile_after_prefill = profile_snapshot();
        if (config.phase_profile) {
            request->result.metrics.prefill = phase_delta(
                request->profile_after_prefill, request->profile_started);
            std::cerr << "[glm53-prefill-state] kda_raw_f32_hash="
                      << std::hex << kda_state_hash(request->sequence)
                      << std::dec << '\n';
            print_kda_layer_hashes(request->sequence);
            print_mla_layer_hashes(request->sequence);
        }
        const auto decode_prepare_started = std::chrono::steady_clock::now();
        if (mtp_enabled() && request->sampling.temperature == 0.0 &&
            request->maximum_new_tokens > 1U) {
            auto mtp = prepare_mtp_prompt(
                request->prompt, request->base_hidden, request->sequence);
            if (!mtp.ok()) {
                request->result.errors = std::move(mtp.errors);
                complete_request(request);
                return;
            }
            request->mtp_ready = true;
        }
        store_prefix(request->prompt, request->sequence, request->logits,
                     request->base_hidden);
        // The prompt cache remains host/COW F32. Decode state is admitted once
        // after that immutable snapshot, then never read back per token.
        if (request->maximum_new_tokens > 1U && fused_kda_enabled() &&
            !request->device_sequence.ready) {
            auto prepared = prepare_device_sequence(
                request->sequence, request->device_sequence);
            if (!prepared.ok()) {
                request->result.errors = std::move(prepared.errors);
                complete_request(request);
                return;
            }
        }
        request->counts.assign(kVocabulary, 0U);
        request->generator.seed(request->sampling.seed);
        request->streamed =
            std::make_unique<StopSequenceBuffer>(request->stop);
        request->position = static_cast<std::uint32_t>(request->prompt.size());
        request->decode_cache_start = weights->stats();
        request->decode_static_hits_start = static_experts.route_hits.load(
            std::memory_order_relaxed);
        request->decode_static_misses_start = static_experts.route_misses.load(
            std::memory_order_relaxed);
        request->result.metrics.decode_prepare_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          decode_prepare_started).count();
        request->profile_decode_started = profile_snapshot();
        request->decode_started = now_seconds();
        request->decoding = true;
        if (request->maximum_new_tokens == 0U) {
            complete_request(request);
        }
    }

    void advance_prefill(const std::shared_ptr<ScheduledRequest>& request,
                         std::size_t maximum_rows) {
        if (request->done || request->decoding) return;
        if (request->prefill_cursor == request->prompt.size()) {
            finish_prefill(request);
            return;
        }
        const auto count = std::min(
            maximum_rows, request->prompt.size() - request->prefill_cursor);
        auto prefill = forward_prompt(
            std::span<const std::uint32_t>(request->prompt).subspan(
                request->prefill_cursor, count),
            request->logits, request->sequence, &request->base_hidden, false,
            device_prefill_for_context(config.maximum_context_tokens)
                ? &request->device_sequence : nullptr);
        if (!prefill.ok()) {
            request->result.errors = std::move(prefill.errors);
            complete_request(request);
            return;
        }
        request->prefill_cursor += count;
        if (request->prefill_cursor == request->prompt.size()) {
            finish_prefill(request);
        }
    }

    [[nodiscard]] bool publish_draw(
        const std::shared_ptr<ScheduledRequest>& request,
        const TokenLogprob& drawn, std::uint32_t& forward_token_id) {
        if (drawn.token == 154820U || drawn.token == 154827U ||
            drawn.token == 154829U) {
            request->result.stopped = true;
            complete_request(request);
            return false;
        }
        request->result.generated_token_ids.push_back(drawn.token);
        request->result.logprobs.push_back(drawn);
        request->sampled.push_back(drawn.token);
        ++request->counts[drawn.token];
        auto piece = tokenizer.decode_token(drawn.token);
        if (!piece.ok()) {
            request->result.errors = std::move(piece.errors);
            complete_request(request);
            return false;
        }
        request->streamed->append(drawn.token, piece.value,
                                  request->on_token);
        if (request->streamed->stopped() ||
            request->streamed->cancelled() ||
            request->result.generated_token_ids.size() ==
                request->maximum_new_tokens) {
            complete_request(request);
            return false;
        }
        forward_token_id = drawn.token;
        return true;
    }

    [[nodiscard]] bool sample_request(
        const std::shared_ptr<ScheduledRequest>& request,
        std::uint32_t& forward_token_id) {
        const auto sampling_started = std::chrono::steady_clock::now();
        auto drawn = sample_logits(
            request->logits, request->sampling,
            SamplingHistory{request->counts, request->sampled},
            request->generator);
        if (!drawn.ok()) {
            request->result.errors = std::move(drawn.errors);
            complete_request(request);
            return false;
        }
        if (config.phase_profile) {
            graph_sampling_nanoseconds.fetch_add(
                elapsed_nanoseconds(sampling_started),
                std::memory_order_relaxed);
        }
        return publish_draw(request, drawn, forward_token_id);
    }

    [[nodiscard]] bool try_mtp_step(
        const std::shared_ptr<ScheduledRequest>& request,
        std::uint32_t first_token) {
        if (!request->mtp_ready || request->sampling.temperature != 0.0 ||
            request->sampling.xtc_probability != 0.0 ||
            request->sampling.future_entropy_candidates != 0U ||
            request->base_hidden.size() < kHidden || request->done) {
            return false;
        }
        Glm53SequenceState mtp_after_first = request->sequence;
        std::vector<float> draft_logits(kVocabulary), draft_feedback(kHidden);
        const auto mtp_position =
            static_cast<std::uint32_t>(mtp_after_first.mla(kMtpLayer).rows());
        auto status = forward_mtp(
            first_token,
            std::span<const float>(request->base_hidden).last(kHidden),
            mtp_position, mtp_after_first, draft_logits, draft_feedback);
        if (!status.ok()) {
            request->result.errors = std::move(status.errors);
            complete_request(request);
            return true;
        }
        ++mtp_drafts;
        auto draft_generator = request->generator;
        auto draft = sample_logits(
            draft_logits, request->sampling,
            SamplingHistory{request->counts, request->sampled},
            draft_generator);
        if (!draft.ok()) {
            request->result.errors = std::move(draft.errors);
            complete_request(request);
            return true;
        }

        Glm53SequenceState verified = request->sequence;
        const std::array<std::uint32_t, 2U> candidates{
            first_token, draft.token};
        std::vector<float> verification_logits(
            static_cast<std::size_t>(2U) * kVocabulary);
        std::vector<float> verification_hidden;
        status = forward_prompt(candidates, verification_logits, verified,
                                &verification_hidden, true);
        if (!status.ok()) {
            request->result.errors = std::move(status.errors);
            complete_request(request);
            return true;
        }
        auto target_generator = request->generator;
        auto target = sample_logits(
            std::span<const float>(verification_logits).first(kVocabulary),
            request->sampling,
            SamplingHistory{request->counts, request->sampled},
            target_generator);
        if (!target.ok()) {
            request->result.errors = std::move(target.errors);
            complete_request(request);
            return true;
        }
        if (target.token == draft.token) {
            verified.copy_mla_from(kMtpLayer, mtp_after_first);
            request->sequence = std::move(verified);
            request->generator = std::move(target_generator);
            request->base_hidden.insert(
                request->base_hidden.end(), verification_hidden.begin(),
                verification_hidden.end());
            std::copy_n(verification_logits.begin() + kVocabulary,
                        kVocabulary, request->logits.begin());
            request->position += 2U;
            request->result.metrics.decode_tokens += 2U;
            request->iteration += 2U;
            ++mtp_accepted;
            std::uint32_t ignored = 0U;
            static_cast<void>(publish_draw(request, target, ignored));
            if (!request->done) {
                // Keep the draft cache aligned through the accepted token.
                // Its next proposal is deliberately discarded; the next
                // target sample remains the sole source of published tokens.
                std::vector<float> ignored_logits(kVocabulary);
                std::vector<float> ignored_feedback(kHidden);
                status = forward_mtp(
                    target.token,
                    std::span<const float>(verification_hidden).subspan(
                        0U, kHidden),
                    mtp_position + 1U, request->sequence, ignored_logits,
                    ignored_feedback);
                if (!status.ok()) {
                    request->result.errors = std::move(status.errors);
                    complete_request(request);
                }
            }
            return true;
        }

        // A rejected second token must leave the target exactly after the
        // first token. COW makes the retry cheap in state memory; execution is
        // intentionally repeated rather than trying to extract a mutable
        // intermediate snapshot from the two-row verification page.
        std::vector<float> first_hidden;
        status = forward_prompt(
            std::span<const std::uint32_t>(&first_token, 1U), request->logits,
            request->sequence, &first_hidden);
        if (!status.ok()) {
            request->result.errors = std::move(status.errors);
            complete_request(request);
            return true;
        }
        request->sequence.copy_mla_from(kMtpLayer, mtp_after_first);
        request->base_hidden.insert(request->base_hidden.end(),
                                    first_hidden.begin(), first_hidden.end());
        ++request->position;
        ++request->result.metrics.decode_tokens;
        ++request->iteration;
        return true;
    }

    void scheduler_loop() {
        for (;;) {
            {
                std::unique_lock lock(scheduler_mutex);
                scheduler_ready.wait(lock, [&] {
                    return scheduler_stopping || !pending_requests.empty() ||
                           !active_requests.empty();
                });
                if (scheduler_stopping && pending_requests.empty() &&
                    active_requests.empty()) {
                    return;
                }
                // A fresh queue gets a tiny admission window so requests that
                // arrived together become one prefill/decode cohort. Once a
                // cohort is active there is no delay: iteration admission is
                // immediate. Two milliseconds is below network jitter while
                // avoiding a model- or hardware-specific batching timeout.
                if (active_requests.empty() && pending_requests.size() == 1U &&
                    !scheduler_stopping) {
                    static_cast<void>(scheduler_ready.wait_for(
                        lock, std::chrono::milliseconds(2), [&] {
                            return scheduler_stopping ||
                                   pending_requests.size() > 1U;
                        }));
                }
                while (!pending_requests.empty() &&
                       active_requests.size() < scheduler_capacity) {
                    active_requests.push_back(pending_requests.front());
                    pending_requests.pop_front();
                }
            }
            for (auto& request : active_requests) {
                if (!request->prepared && !request->done) {
                    static_cast<void>(prepare_request(request));
                }
            }
            std::size_t live = 0U;
            std::size_t decoding = 0U;
            for (const auto& request : active_requests) {
                if (!request->done) ++live;
                if (!request->done && request->decoding) ++decoding;
            }
            if (live > 0U) {
                scheduler_iterations.fetch_add(1U, std::memory_order_relaxed);
                if (live > 1U) {
                    scheduler_batched_iterations.fetch_add(
                        1U, std::memory_order_relaxed);
                }
                std::vector<std::shared_ptr<ScheduledRequest>> step_requests;
                std::vector<std::uint32_t> step_tokens;
                std::vector<std::uint32_t> step_positions;
                std::vector<Glm53SequenceState*> step_sequences;
                std::vector<DeviceSequenceState*> step_device_sequences;
                for (auto& request : active_requests) {
                    if (request->done || !request->decoding) continue;
                    std::uint32_t token = 0U;
                    if (sample_request(request, token)) {
                        step_requests.push_back(request);
                        step_tokens.push_back(token);
                        step_positions.push_back(request->position);
                        step_sequences.push_back(&request->sequence);
                        step_device_sequences.push_back(
                            request->device_sequence.ready
                                ? &request->device_sequence : nullptr);
                    }
                }
                if (!step_requests.empty()) {
                    const bool mtp_handled = step_requests.size() == 1U &&
                        try_mtp_step(step_requests.front(),
                                     step_tokens.front());
                    if (!mtp_handled) {
                        if (step_requests.size() > 1U) {
                            for (auto& request : step_requests) {
                                request->mtp_ready = false;
                            }
                        }
                        std::vector<float> step_logits(
                            step_requests.size() * kVocabulary);
                        std::vector<float> step_hidden(
                            step_requests.size() * kHidden);
                        const bool capture = profiler_capture_enabled() &&
                            !profiler_captured.exchange(
                                true, std::memory_order_relaxed);
                        if (capture) {
                            const auto started = cuda.profiler_start();
                            if (!started.ok()) {
                                std::cerr << "[glm53-profile] "
                                          << started.errors.front() << '\n';
                            }
                        }
                        const auto forward_started =
                            std::chrono::steady_clock::now();
                        auto step = forward_token_batch(
                            step_tokens, step_positions, step_sequences,
                            step_device_sequences, step_logits, step_hidden);
                        const auto forward_nanoseconds =
                            elapsed_nanoseconds(forward_started);
                        const auto width = std::min<std::size_t>(
                            step_requests.size(),
                            scheduler_width_iterations.size() - 1U);
                        scheduler_width_iterations[width].fetch_add(
                            1U, std::memory_order_relaxed);
                        scheduler_width_forward_nanoseconds[width].fetch_add(
                            forward_nanoseconds, std::memory_order_relaxed);
                        scheduler_width_tokens[width].fetch_add(
                            step_requests.size(), std::memory_order_relaxed);
                        if (step_requests.size() == 1U) {
                            scheduler_single_forward_nanoseconds.fetch_add(
                                forward_nanoseconds,
                                std::memory_order_relaxed);
                            scheduler_single_forward_tokens.fetch_add(
                                1U, std::memory_order_relaxed);
                        } else {
                            scheduler_batch_forward_nanoseconds.fetch_add(
                                forward_nanoseconds,
                                std::memory_order_relaxed);
                            scheduler_batch_forward_tokens.fetch_add(
                                step_requests.size(),
                                std::memory_order_relaxed);
                        }
                        if (capture) {
                            const auto stopped = cuda.profiler_stop();
                            if (!stopped.ok()) {
                                std::cerr << "[glm53-profile] "
                                          << stopped.errors.front() << '\n';
                            }
                        }
                        if (!step.ok()) {
                            for (auto& request : step_requests) {
                                request->result.errors = step.errors;
                                complete_request(request);
                            }
                        } else {
                            for (std::size_t row = 0U;
                                 row < step_requests.size(); ++row) {
                                auto& request = step_requests[row];
                                std::copy_n(
                                    step_logits.begin() +
                                        static_cast<std::ptrdiff_t>(
                                            row * kVocabulary),
                                    kVocabulary, request->logits.begin());
                                request->base_hidden.insert(
                                    request->base_hidden.end(),
                                    step_hidden.begin() +
                                        static_cast<std::ptrdiff_t>(row * kHidden),
                                    step_hidden.begin() +
                                        static_cast<std::ptrdiff_t>(
                                            (row + 1U) * kHidden));
                                ++request->position;
                                ++request->result.metrics.decode_tokens;
                                ++request->iteration;
                            }
                        }
                    }
                }
                // Decode has latency priority. A newly admitted prompt gets a
                // single-token chunk while decoders are live; with no decode
                // work, a page-sized chunk retains the wide prefill route.
                const std::size_t prefill_rows =
                    decoding == 0U ? config.prefill_page_tokens : 1U;
                for (auto& request : active_requests) {
                    if (!request->done && !request->decoding) {
                        advance_prefill(request, prefill_rows);
                    }
                }
            }
            active_requests.erase(
                std::remove_if(active_requests.begin(), active_requests.end(),
                    [](const auto& request) { return request->done; }),
                active_requests.end());
        }
    }

    [[nodiscard]] Glm53GenerationResult schedule(
        std::vector<std::uint32_t> prompt,
        std::uint32_t maximum_new_tokens, const SamplingOptions& sampling,
        std::span<const std::string> stop,
        const TokenStreamCallback& on_token) {
        auto request = std::make_shared<ScheduledRequest>();
        request->prompt = std::move(prompt);
        request->maximum_new_tokens = maximum_new_tokens;
        request->sampling = sampling;
        request->stop.assign(stop.begin(), stop.end());
        request->on_token = on_token;
        {
            std::scoped_lock lock(scheduler_mutex);
            if (scheduler_stopping) {
                request->result.errors.emplace_back(
                    "GLM-5.3 iteration scheduler is stopping");
                return std::move(request->result);
            }
            pending_requests.push_back(request);
        }
        scheduler_ready.notify_one();
        std::unique_lock lock(request->completion_mutex);
        request->completion.wait(lock, [&] { return request->done; });
        return std::move(request->result);
    }
};

Glm53Runtime::Glm53Runtime() : impl_(std::make_unique<Impl>()) {}
Glm53Runtime::~Glm53Runtime() = default;
Glm53Runtime::Glm53Runtime(Glm53Runtime&&) noexcept = default;
Glm53Runtime& Glm53Runtime::operator=(Glm53Runtime&&) noexcept = default;

ValidationResult Glm53Runtime::initialize(
    const std::string& model_directory, const Glm53RuntimeConfig& config) {
    ValidationResult result;
    if (impl_->ready) {
        result.errors.emplace_back("GLM-5.3 runtime is already initialized");
        return result;
    }
    if (config.maximum_context_tokens == 0U ||
        config.maximum_context_tokens > kMaximumSupportedContext) {
        result.errors.push_back(
            "GLM-5.3 text context must be within [1, " +
            std::to_string(kMaximumSupportedContext) + "]");
        return result;
    }
    impl_->config = config;
    impl_->config.phase_profile =
        config.phase_profile || phase_profile_environment_enabled();
    impl_->config.prefill_page_tokens =
        prefill_page_tokens_from_environment(config.prefill_page_tokens);
    if (impl_->config.prefill_page_tokens == 0U ||
        impl_->config.prefill_page_tokens > config.maximum_context_tokens) {
        result.errors.emplace_back(
            "GLM-5.3 prefill page tokens must be within the configured "
            "context window");
        return result;
    }
    impl_->host_state_capacity =
        host_sequence_capacity(config.maximum_context_tokens);
    // Warmup resolves the binding post-load CUDA capacity before the first
    // request is prepared. Until then admit one request, never the former
    // host-only estimate of as many as 32.
    impl_->scheduler_capacity = 1U;
    impl_->prefix_cache_limit = impl_->host_state_capacity;
    impl_->devices = resolve_runtime_devices(config.devices);
    result = validate_common_runtime_config(
        impl_->devices, config.vram_cache_fraction,
        config.sampling_temperature, "GLM-5.3");
    if (!result.ok()) return result;
    auto device_plan = plan_runtime_devices(
        impl_->devices, config.vram_cache_fraction, kDeviceWorkspaceReserve,
        kMinimumDeviceBudget, "GLM-5.3");
    if (!device_plan.ok()) return {std::move(device_plan.errors)};
    auto tokenizer = ModelTokenizer::load(model_directory + "/tokenizer.json");
    if (!tokenizer.ok()) return {std::move(tokenizer.errors)};
    // The tokenizer has 154,820 base pieces plus 36 added special tokens.
    // The checkpoint pads its embedding and output matrices to 154,880 rows;
    // those 24 padding rows are deliberately not tokenizable.
    if (tokenizer.value.vocabulary_size() != 154856U) {
        result.errors.emplace_back(
            "GLM-5.3 tokenizer must expose 154856 usable token ids");
        return result;
    }
    auto checkpoint = Glm53CheckpointReader::open(model_directory);
    if (!checkpoint.ok()) return {std::move(checkpoint.errors)};
    result = impl_->cuda.initialize(impl_->devices,
                                    impl_->config.phase_profile);
    if (!result.ok()) return result;
    const auto admitted_page_rows = static_cast<std::uint64_t>(
        impl_->config.prefill_page_tokens);
    const auto context_rows = static_cast<std::uint64_t>(
        impl_->config.maximum_context_tokens);
    const auto maximum_matmul_input_bytes = sizeof(float) * std::max(
        admitted_page_rows * std::uint64_t{12288},
        context_rows * static_cast<std::uint64_t>(kKvRank));
    const auto maximum_matmul_output_bytes = sizeof(float) * std::max(
        admitted_page_rows * std::uint64_t{12288},
        context_rows * static_cast<std::uint64_t>(kHeads) * 2U *
            static_cast<std::uint64_t>(kMlaHead));
    for (const auto device : impl_->devices) {
        result = impl_->cuda.reserve_matmul_workspace(
            device, maximum_matmul_input_bytes,
            maximum_matmul_output_bytes);
        if (!result.ok()) return result;
    }
    // Both decode and causal device-page MLA return one raw score per
    // (head, visible-token) to the exact host softmax. Admit its maximum span
    // before either timed phase so a longer page never grows it in flight.
    impl_->mla_softmax_scores.resize(
        static_cast<std::size_t>(kHeads) *
        impl_->config.maximum_context_tokens);
    impl_->mla_selected_positions.reserve(kIndexSelectionWidth);
    impl_->mla_arena_rows.reserve(kIndexSelectionWidth);
    impl_->mla_expansion_source_positions.reserve(kIndexSelectionWidth);
    impl_->mla_expansion_destination_rows.reserve(kIndexSelectionWidth);
    impl_->tokenizer = std::move(tokenizer.value);
    impl_->checkpoint = std::move(checkpoint.value);
    impl_->device_budgets = device_plan.value.budgets;
    impl_->weight_capacities = device_plan.value.weight_capacities;
    impl_->resident_reserve_bytes.assign(impl_->devices.size(), 0U);
    impl_->shared_experts.bytes_by_slot.assign(impl_->devices.size(), 0U);
    if (impl_->devices.size() > 1U &&
        !cross_gpu_projections_enabled(impl_->devices)) {
        // PCIe/PHB systems pay a full activation bridge for every owner
        // change. Use capacity-weighted contiguous pipeline stages so a token
        // crosses once. Best-rank P2P (NVLink/NVSwitch) keeps the fine-grained
        // schedule, which the TP executor can consume without redistributing
        // layer ownership when that topology is available.
        impl_->device_schedule = contiguous_layer_schedule(
            kLayers, impl_->weight_capacities);
    } else {
        impl_->device_schedule = std::move(
            device_plan.value.weighted_schedule);
    }
    if (impl_->device_schedule.empty()) {
        return {{"GLM-5.3 could not derive a topology-aware layer schedule"}};
    }
    impl_->resident_execution_active = fused_kda_enabled();
    if (impl_->resident_execution_active) {
        for (const auto device : impl_->devices) {
            if (!impl_->cuda.validate_dsv4_mhc_device(device).ok()) {
                impl_->resident_execution_active = false;
                break;
            }
        }
    }
    if (impl_->resident_execution_active) {
        // mHC weights use the same arena as cached linears. Reserve their
        // exact order of magnitude per discovered layer owner before the
        // cache fills the arena; no device-count or VRAM-size assumption is
        // embedded here.
        constexpr std::uint64_t per_layer_mhc_reserve = 4ULL << 20U;
        std::vector<std::uint64_t> resident_reserve(impl_->devices.size());
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            resident_reserve[impl_->slot_for(layer)] +=
                per_layer_mhc_reserve;
        }
        for (std::size_t slot = 0U; slot < impl_->devices.size(); ++slot) {
            if (impl_->weight_capacities[slot] <= resident_reserve[slot] +
                                                    kMinimumDeviceBudget) {
                impl_->resident_execution_active = false;
                break;
            }
        }
        if (impl_->resident_execution_active) {
            impl_->resident_reserve_bytes = resident_reserve;
            for (std::size_t slot = 0U; slot < impl_->devices.size(); ++slot) {
                impl_->weight_capacities[slot] -= resident_reserve[slot];
            }
        }
    }
    // Layer 3 is the first MoE layer, but the MXFP4 release leaves its routed
    // experts BF16 under the publisher's mixed-precision correction. Sampling
    // layer 4 there measures the format 39 of the 42 MoE layers actually use.
    const std::string expert_prefix =
        "model.language_model.layers." +
        std::string(impl_->checkpoint->manifest().quantization ==
                            Glm53Quantization::Mxfp4Group32
                        ? "4" : "3") +
        ".mlp.experts.0.";
    const auto expert_bytes =
        impl_->checkpoint->cuda_linear_storage_bytes(
            expert_prefix + "gate_proj") +
        impl_->checkpoint->cuda_linear_storage_bytes(
            expert_prefix + "up_proj") +
        impl_->checkpoint->cuda_linear_storage_bytes(
            expert_prefix + "down_proj");
    const auto initial_cache_bytes = std::accumulate(
        impl_->weight_capacities.begin(), impl_->weight_capacities.end(),
        std::uint64_t{0U});
    const auto routed_bytes = std::accumulate(
        impl_->checkpoint->manifest().tensors.begin(),
        impl_->checkpoint->manifest().tensors.end(), std::uint64_t{0U},
        [](std::uint64_t total, const Glm53ManifestTensor& tensor) {
            return tensor.role == Glm53TensorRole::RoutedExpert
                ? total + tensor.source_bytes : total;
        });
    const auto& hardware = host_hardware_profile();
    const auto host_width = std::min<std::size_t>(
        hardware.worker_threads(0.5), hardware.usable_cpu_ids.size());
    const auto model_parallel_width = static_cast<std::size_t>(
        impl_->checkpoint->config().experts_per_token) * 2U;
    const auto override = host_moe_override();
    const bool host_instruction_support =
#if STRATA_GLM53_HOST_AVX2
        __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
        false;
#endif
    const bool host_moe_admitted = override > 0 ||
        (override < 0 && host_instruction_support &&
         host_width >= model_parallel_width &&
         initial_cache_bytes != 0U && routed_bytes > 2U * initial_cache_bytes);
    if (host_moe_admitted && host_width != 0U) {
        std::vector<int> cpus(hardware.usable_cpu_ids.begin(),
                              hardware.usable_cpu_ids.begin() +
                                  static_cast<std::ptrdiff_t>(host_width));
        impl_->host_moe_workers = std::make_unique<HostWorkerPool>(
            std::move(cpus), std::chrono::milliseconds(1));
        impl_->host_moe_active =
            impl_->host_moe_workers->size() == host_width;
    }
    if (impl_->host_moe_active) {
        // Decode's batch-1 expert primitive uses separate scratch from the
        // page-major prefill primitive. Reserve its fixed maximum extents
        // before profiling begins so the first generated token cannot grow a
        // vector inside the measured interval.
        impl_->host_moe_quantized_input.resize(kHidden);
        impl_->host_moe_activations.resize(9U * 2048U);
        impl_->host_moe_expert_outputs.resize(9U * kHidden);
        const auto page_rows = static_cast<std::size_t>(
            impl_->config.prefill_page_tokens);
        constexpr std::size_t outputs_per_page_row = 9U;
        constexpr std::size_t maximum_cohort_rows = 32U;
        constexpr std::size_t maximum_device_assignments =
            maximum_cohort_rows * outputs_per_page_row;
        impl_->page_groups.reserve(Impl::kExpertSlots);
        impl_->page_assignments.resize(page_rows * outputs_per_page_row);
        impl_->page_device_experts.reserve(maximum_device_assignments);
        impl_->page_device_output_slots.reserve(maximum_device_assignments);
        impl_->page_device_inputs.resize(
            maximum_device_assignments * kHidden);
        impl_->page_quantized_input.resize(page_rows * kHidden);
        impl_->page_activations.resize(
            page_rows * outputs_per_page_row * 2048U);
        impl_->page_expert_outputs.resize(
            page_rows * outputs_per_page_row * kHidden);
        auto admitted = impl_->admit_shared_experts();
        if (!admitted.ok()) return admitted;
        // The same buffers carry either one decode row's resident tier or all
        // device assignments in an admitted independent-sequence cohort.
        impl_->shared_expert_gate.resize(
            maximum_device_assignments * 2048U);
        impl_->shared_expert_up.resize(
            maximum_device_assignments * 2048U);
        impl_->shared_expert_output.resize(
            maximum_device_assignments * kHidden);
    }
    // Shared experts own independent cudaMalloc buffers and have already
    // reduced this capacity. Resident mHC weights are different: CudaWeight
    // uploads suballocate from this same arena, while the cache must not spend
    // their reserved bytes. Give the physical arena cache+resident capacity
    // and expose only the cache part to Glm53WeightCache. Otherwise the direct
    // mHC uploads consume invisible arena space and an apparently admissible
    // final cached expert fails partway through its triplet.
    for (std::size_t slot = 0U; slot < impl_->devices.size(); ++slot) {
        result = impl_->cuda.reserve_weight_arena(
            impl_->devices[slot], impl_->weight_capacities[slot] +
                                      impl_->resident_reserve_bytes[slot]);
        if (!result.ok()) return result;
    }
    impl_->weights = std::make_unique<Glm53WeightCache>(
        *impl_->checkpoint, impl_->cuda, impl_->devices,
        impl_->weight_capacities);
    const auto cache_bytes = std::accumulate(
        impl_->weight_capacities.begin(), impl_->weight_capacities.end(),
        std::uint64_t{0U});
    if (expert_bytes != 0U) {
        const auto cache_experts = static_cast<std::size_t>(
            cache_bytes / expert_bytes);
        const auto host_window = static_cast<std::size_t>(
            host_hardware_profile().worker_threads(0.25)) *
            impl_->checkpoint->config().experts_per_token;
        impl_->prefetch_queue_limit = std::max<std::size_t>(
            1U, std::min(cache_experts, host_window));
        impl_->prefetch_prediction_limit = std::min<std::size_t>(
            impl_->checkpoint->config().experts_per_token,
            impl_->prefetch_queue_limit);
        impl_->prefetch_minimum_confidence =
            1.0 - static_cast<double>(
                      impl_->checkpoint->config().experts_per_token) /
                      static_cast<double>(
                          impl_->checkpoint->config().routed_experts);
    }
    if (impl_->devices.size() > 1U) {
        auto worker_cpus = projection_worker_cpus(impl_->devices);
        if (worker_cpus.size() == impl_->devices.size()) {
            impl_->projection_workers = std::make_unique<HostWorkerPool>(
                std::move(worker_cpus));
        }
    }
    impl_->full_tensor_parallel_active =
        full_tensor_parallel_enabled() && impl_->devices.size() == 2U &&
        impl_->projection_workers != nullptr &&
        cross_gpu_projections_enabled(impl_->devices);
    if ((tensor_parallel_head_enabled() ||
         impl_->full_tensor_parallel_active) &&
        impl_->projection_workers != nullptr) {
        impl_->lm_head_ranges = weighted_row_ranges(
            kVocabulary, impl_->weight_capacities, 128U);
    }
    if (impl_->devices.size() > 1U || config.verbose) {
        std::uint32_t hops = 0U;
        for (std::uint32_t layer = 1U; layer < kLayers; ++layer) {
            if (impl_->slot_for(layer) != impl_->slot_for(layer - 1U)) ++hops;
        }
        std::cerr << "[glm53-topology] mode="
                  << (impl_->full_tensor_parallel_active
                          ? "high-speed-peer-tp2"
                          : (cross_gpu_projections_enabled(impl_->devices)
                                 ? "high-speed-peer"
                                 : "contiguous-pipeline"))
                  << " activation_hops=" << hops << " layers=" << kLayers
                  << '\n';
        std::cerr << "[glm53-resident] mode="
                  << (impl_->resident_execution_active
                          ? "fused-layer"
                          : "host-boundary-fallback")
                  << '\n';
        std::cerr << "[glm53-expert-tier] mode="
                  << (impl_->host_moe_active
                          ? (impl_->checkpoint->manifest().quantization ==
                                     Glm53Quantization::Mxfp4Group32
                                 ? "host-mxfp4" : "host-fp8")
                          : "cuda-lru")
                  << " workers="
                  << (impl_->host_moe_workers == nullptr
                          ? 0U : impl_->host_moe_workers->size())
                  << " routed_gib="
                  << static_cast<double>(routed_bytes) /
                         static_cast<double>(1ULL << 30U)
                  << " cuda_cache_gib="
                  << static_cast<double>(cache_bytes) /
                         static_cast<double>(1ULL << 30U)
                  << " shared_expert="
                  << (impl_->shared_experts.active ? "device" : "host")
                  << " shared_expert_gib="
                  << static_cast<double>(impl_->shared_experts.bytes) /
                         static_cast<double>(1ULL << 30U)
                  << '\n';
    }
    if (replay_ssm_enabled() || phase_scheduler_enabled()) {
        auto worker_cpus = compute_worker_cpus();
        if (!worker_cpus.empty()) {
            impl_->kda_workers = std::make_unique<HostWorkerPool>(
                std::move(worker_cpus), std::chrono::milliseconds(1));
        }
    }
    impl_->ready = true;
    try {
        // Keep API/server startup lazy-fast while warming independent device
        // spines in the background. The first generation joins this work; an
        // idle server usually reaches full residency before its first request.
        impl_->warmup_thread = std::thread([state = impl_.get()] {
            state->warmup_result = state->warmup();
        });
        impl_->prefetch_threads.reserve(impl_->devices.size());
        for (std::size_t worker = 0U; worker < impl_->devices.size(); ++worker) {
            impl_->prefetch_threads.emplace_back([state = impl_.get()] {
                state->prefetch_loop();
            });
        }
        impl_->scheduler_thread = std::thread([state = impl_.get()] {
            state->scheduler_loop();
        });
    } catch (const std::system_error& error) {
        impl_->ready = false;
        result.errors.push_back(
            "GLM-5.3 could not start background spine warmup: " +
            std::string(error.what()));
    }
    return result;
}

Glm53GenerationResult Glm53Runtime::generate_chat_stream(
    std::span<const ChatMessage> messages, std::uint32_t maximum_new_tokens,
    const SamplingOptions& sampling, std::span<const std::string> stop,
    const TokenStreamCallback& on_token) {
    Glm53GenerationResult result;
    if (!impl_->ready) {
        result.errors.emplace_back("GLM-5.3 runtime is not initialized");
        return result;
    }
    std::string error;
    if (!validate_sampling_options(sampling, error)) {
        result.errors.push_back(std::move(error));
        return result;
    }
    if (!validate_chat_messages(messages, error)) {
        result.errors.push_back(std::move(error));
        return result;
    }
    for (const auto& message : messages) {
        for (const auto& part : message.parts) {
            if (part.kind != ChatContentKind::Text) {
                result.errors.emplace_back(
                    "GLM-5.3 vision is not implemented; this runtime supports text-only messages");
                return result;
            }
        }
    }
    auto encoded = impl_->tokenizer.encode(
        render_glm53_chat_prompt(messages, "max", true));
    if (!encoded.ok()) {
        result.errors = std::move(encoded.errors);
        return result;
    }
    if (encoded.value.empty() || encoded.value.size() + maximum_new_tokens >
            impl_->config.maximum_context_tokens) {
        result.errors.emplace_back(
            "GLM-5.3 prompt and requested generation exceed the admitted text context");
        return result;
    }
    const auto mtp_drafts_before =
        impl_->mtp_drafts.load(std::memory_order_relaxed);
    const auto mtp_accepted_before =
        impl_->mtp_accepted.load(std::memory_order_relaxed);
    const auto prefetch_requests_before =
        impl_->prefetch_requests.load(std::memory_order_relaxed);
    const auto cache_before = impl_->weights->stats();
    result = impl_->schedule(std::move(encoded.value), maximum_new_tokens,
                             sampling, stop, on_token);
    const auto request_mtp_drafts =
        impl_->mtp_drafts.load(std::memory_order_relaxed) - mtp_drafts_before;
    const auto request_mtp_accepted =
        impl_->mtp_accepted.load(std::memory_order_relaxed) -
        mtp_accepted_before;
    if (request_mtp_drafts != 0U) {
        std::cerr << "[glm53-mtp] drafts=" << request_mtp_drafts
                  << " accepted=" << request_mtp_accepted
                  << " acceptance="
                  << (100.0 * static_cast<double>(request_mtp_accepted) /
                      static_cast<double>(request_mtp_drafts))
                  << "%\n";
    }
    const auto request_prefetches =
        impl_->prefetch_requests.load(std::memory_order_relaxed) -
        prefetch_requests_before;
    if (request_prefetches != 0U) {
        const auto cache_after = impl_->weights->stats();
        std::cerr << "[glm53-residency] predictions=" << request_prefetches
                  << " completed="
                  << impl_->prefetch_completed.load(std::memory_order_relaxed)
                  << " dropped="
                  << impl_->prefetch_dropped.load(std::memory_order_relaxed)
                  << " errors="
                  << impl_->prefetch_errors.load(std::memory_order_relaxed)
                  << " useful="
                  << (cache_after.useful_prefetches -
                      cache_before.useful_prefetches)
                  << " demand_misses="
                  << (cache_after.misses - cache_before.misses)
                  << " evictions="
                  << (cache_after.evictions - cache_before.evictions)
                  << '\n';
    }
    std::cerr << "[glm53-scheduler] iterations="
              << impl_->scheduler_iterations.load(std::memory_order_relaxed)
              << " batched_iterations="
              << impl_->scheduler_batched_iterations.load(
                     std::memory_order_relaxed)
              << " single_forward_ns="
              << impl_->scheduler_single_forward_nanoseconds.load(
                     std::memory_order_relaxed)
              << " single_forward_tokens="
              << impl_->scheduler_single_forward_tokens.load(
                     std::memory_order_relaxed)
              << " batch_forward_ns="
              << impl_->scheduler_batch_forward_nanoseconds.load(
                     std::memory_order_relaxed)
              << " batch_forward_tokens="
              << impl_->scheduler_batch_forward_tokens.load(
                     std::memory_order_relaxed)
              << '\n';
    const auto live_slots = impl_->sequence_slots_live.load(
        std::memory_order_relaxed);
    std::cerr << "[glm53-state-pool] live_slots=" << live_slots
              << " peak_slots="
              << impl_->sequence_slots_peak.load(std::memory_order_relaxed)
              << " free_slots="
              << (impl_->scheduler_capacity > live_slots
                      ? impl_->scheduler_capacity - live_slots : 0U)
              << " host_block_bytes="
              << host_sequence_state_bytes(
                     impl_->config.maximum_context_tokens)
              << " host_fragmented_bytes="
              << impl_->host_state_fragmented_bytes
              << " shared_prefix_entries="
              << impl_->prefix_cache_entries.load(std::memory_order_relaxed)
              << " shared_prefix_bytes="
              << impl_->prefix_cache_bytes.load(std::memory_order_relaxed)
              << " evicted_prefix_entries="
              << impl_->prefix_cache_evictions.load(std::memory_order_relaxed)
              << " evicted_prefix_bytes="
              << impl_->prefix_cache_evicted_bytes.load(
                     std::memory_order_relaxed);
    for (std::size_t slot = 0U; slot < impl_->devices.size(); ++slot) {
        const auto capacity = impl_->sequence_device_capacities[slot];
        std::cerr << " cuda" << impl_->devices[slot] << "_live_bytes="
                  << live_slots * impl_->sequence_device_bytes[slot]
                  << " cuda" << impl_->devices[slot] << "_free_block_bytes="
                  << (capacity > live_slots
                          ? (capacity - live_slots) *
                                impl_->sequence_device_bytes[slot] : 0U)
                  << " cuda" << impl_->devices[slot]
                  << "_fragmented_bytes="
                  << impl_->sequence_device_fragmented_bytes[slot];
    }
    std::cerr << '\n';
    // Build the cohort counter line before publishing it. Concurrent request
    // completions otherwise interleave individual stream insertions and make
    // the already-existing sharing counters impossible to attribute.
    std::ostringstream batch_composition;
    batch_composition << "[glm53-batch-composition]";
    for (std::size_t width = 1U;
         width < impl_->scheduler_width_iterations.size(); ++width) {
        const auto iterations =
            impl_->scheduler_width_iterations[width].load(
                std::memory_order_relaxed);
        if (iterations == 0U) continue;
        batch_composition
            << " width" << width << "_iterations=" << iterations
            << " width" << width << "_tokens="
            << impl_->scheduler_width_tokens[width].load(
                   std::memory_order_relaxed)
            << " width" << width << "_forward_ns="
            << impl_->scheduler_width_forward_nanoseconds[width].load(
                   std::memory_order_relaxed);
    }
    batch_composition
        << " host_expert_calls="
        << impl_->host_moe_calls.load(std::memory_order_relaxed)
        << " host_expert_rows="
        << impl_->host_moe_rows.load(std::memory_order_relaxed)
        << " host_expert_weight_bytes="
        << (impl_->host_moe_gate_up_weight_bytes.load(
                std::memory_order_relaxed) +
            impl_->host_moe_down_weight_bytes.load(
                std::memory_order_relaxed));
    std::cerr << batch_composition.str() << '\n';
    if (impl_->config.verbose) {
        std::cerr << "[glm53-projection] parallel_batches="
                  << impl_->parallel_projection_batches.load(
                         std::memory_order_relaxed)
                  << " parallel_requests="
                  << impl_->parallel_projection_requests.load(
                         std::memory_order_relaxed)
                  << " tensor_parallel_head_batches="
                  << impl_->tensor_parallel_head_batches.load(
                         std::memory_order_relaxed)
                  << " parallel_encode_pages="
                  << impl_->parallel_encode_pages.load(
                         std::memory_order_relaxed)
                  << " prefix_cache_hits="
                  << impl_->prefix_cache_hits.load(std::memory_order_relaxed)
                  << " prefix_cache_tokens="
                  << impl_->prefix_cache_tokens.load(std::memory_order_relaxed)
                  << " scheduler_iterations="
                  << impl_->scheduler_iterations.load(std::memory_order_relaxed)
                  << " scheduler_batched_iterations="
                  << impl_->scheduler_batched_iterations.load(
                         std::memory_order_relaxed)
                  << " scheduler_single_forward_ns="
                  << impl_->scheduler_single_forward_nanoseconds.load(
                         std::memory_order_relaxed)
                  << " scheduler_single_forward_tokens="
                  << impl_->scheduler_single_forward_tokens.load(
                         std::memory_order_relaxed)
                  << " scheduler_batch_forward_ns="
                  << impl_->scheduler_batch_forward_nanoseconds.load(
                         std::memory_order_relaxed)
                  << " scheduler_batch_forward_tokens="
                  << impl_->scheduler_batch_forward_tokens.load(
                         std::memory_order_relaxed)
                  << " mtp_drafts="
                  << impl_->mtp_drafts.load(std::memory_order_relaxed)
                  << " mtp_accepted="
                  << impl_->mtp_accepted.load(std::memory_order_relaxed)
                  << " prefetch_requests="
                  << impl_->prefetch_requests.load(std::memory_order_relaxed)
                  << " prefetch_completed="
                  << impl_->prefetch_completed.load(std::memory_order_relaxed)
                  << " prefetch_dropped="
                  << impl_->prefetch_dropped.load(std::memory_order_relaxed)
                  << " prefetch_errors="
                  << impl_->prefetch_errors.load(std::memory_order_relaxed)
                  << " prefetch_queue_limit="
                  << impl_->prefetch_queue_limit
                  << " host_moe_calls="
                  << impl_->host_moe_calls.load(std::memory_order_relaxed)
                  << " host_moe_ms="
                  << static_cast<double>(impl_->host_moe_nanoseconds.load(
                         std::memory_order_relaxed)) / 1.0e6
                  << '\n';
        const auto cache = impl_->weights->stats();
        std::cerr << "[glm53-cache] hits=" << cache.hits
                  << " misses=" << cache.misses
                  << " evictions=" << cache.evictions
                  << " prefetches=" << cache.prefetches
                  << " useful_prefetches=" << cache.useful_prefetches
                  << " failed_prefetches=" << cache.failed_prefetches
                  << '\n';
    }
    if (impl_->config.phase_profile) {
        std::cerr << "[glm53-phase-profile] {\"prefill_page_tokens\":"
                  << impl_->config.prefill_page_tokens
                  << ",\"prompt_token_ids\":";
        print_token_ids(std::cerr, result.prompt_token_ids);
        std::cerr << ",\"generated_token_ids\":";
        print_token_ids(std::cerr, result.generated_token_ids);
        std::cerr << ",\"prompt_tokens\":" << result.metrics.prompt_tokens
                  << ",\"prefill_tokens\":" << result.metrics.prefill_tokens
                  << ",\"decode_tokens\":" << result.metrics.decode_tokens
                  << ",\"prefill_seconds\":" << result.metrics.prefill_seconds
                  << ",\"decode_prepare_seconds\":"
                  << result.metrics.decode_prepare_seconds
                  << ",\"decode_seconds\":" << result.metrics.decode_seconds
                  << ",\"rss_bytes\":" << result.metrics.rss_bytes
                  << ",\"device_vram_used_bytes\":[";
        for (std::size_t index = 0U;
             index < result.metrics.device_vram_used_bytes.size(); ++index) {
            if (index != 0U) std::cerr << ',';
            std::cerr << result.metrics.device_vram_used_bytes[index];
        }
        std::cerr << "],\"prefill\":";
        print_phase_metrics(std::cerr, result.metrics.prefill);
        std::cerr << ",\"decode\":";
        print_phase_metrics(std::cerr, result.metrics.decode);
        std::cerr << "}\n";
    }
    return result;
}

float glm53_host_fp4_group32_row_dot(
    std::span<const std::byte> packed, std::span<const std::byte> scales,
    std::span<const float> input, bool use_avx2) noexcept {
    const auto* codes =
        reinterpret_cast<const std::uint8_t*>(scales.data());
#if STRATA_GLM53_HOST_AVX2
    if (use_avx2) return glm53_host_fp4_dot_avx2(packed.data(), codes, input);
#else
    static_cast<void>(use_avx2);
#endif
    return glm53_host_fp4_dot_scalar(packed.data(), codes, input);
}

float glm53_host_fp8_block128_row_dot(
    std::span<const std::byte> weights, std::span<const float> scales,
    std::span<const float> input, bool use_avx2) noexcept {
#if STRATA_GLM53_HOST_AVX2
    if (use_avx2) {
        return glm53_host_fp8_dot_avx2(weights.data(), scales.data(), input);
    }
#else
    static_cast<void>(use_avx2);
#endif
    return glm53_host_fp8_dot_scalar(weights.data(), scales.data(), input);
}

float glm53_host_bf16_row_dot(
    std::span<const std::byte> weights, std::span<const float> input,
    bool use_avx2) noexcept {
#if STRATA_GLM53_HOST_AVX2
    if (use_avx2) return glm53_host_bf16_dot_avx2(weights.data(), input);
#else
    static_cast<void>(use_avx2);
#endif
    return glm53_host_bf16_dot_scalar(weights.data(), input);
}

std::size_t glm53_sparse_index_select_for_test(
    std::span<std::uint32_t> selected, std::span<const float> indexer_query,
    std::span<const float> indexer_keys, std::span<const float> gate_scores,
    std::span<const float> pool_ape, std::span<const float> head_weights,
    std::uint32_t history) {
    constexpr auto width = Glm53SparseIndexParameters::head_dim;
    const auto pools = history / Glm53SparseIndexParameters::pool;
    std::vector<float> pool_keys(static_cast<std::size_t>(pools) * width);
    for (std::uint32_t pool = 0U; pool < pools; ++pool) {
        glm53_index_pool_key(
            std::span<float>(pool_keys).subspan(
                static_cast<std::size_t>(pool) * width, width),
            pool,
            [&](std::uint32_t token) {
                return indexer_keys.data() +
                       static_cast<std::size_t>(token) * width;
            },
            [&](std::uint32_t token) {
                return gate_scores.data() +
                       static_cast<std::size_t>(token) * width;
            },
            pool_ape);
    }
    return glm53_sparse_index_select(
        selected, indexer_query,
        [&](std::uint32_t pool) {
            return pool_keys.data() + static_cast<std::size_t>(pool) * width;
        },
        head_weights, history);
}

void glm53_indexer_gate_for_test(std::span<float> output,
                                 std::span<const float> input,
                                 std::span<const float> weight) noexcept {
    glm53_indexer_gate(output, input, weight);
}

void glm53_indexer_layer_norm_for_test(std::span<float> values,
                                       std::span<const float> weight,
                                       std::span<const float> bias) noexcept {
    glm53_indexer_layer_norm(values, weight, bias);
}

}  // namespace strata
