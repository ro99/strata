#include "strata/models/glm53/glm53_runtime.hpp"
#include "strata/models/glm53/glm53_sequence.hpp"

#include "../common/cuda_stats_delta.hpp"

#include "strata/engine/runtime_support.hpp"
#include "strata/engine/route_predictor.hpp"
#include "strata/models/common/tokenizer.hpp"
#include "strata/models/deepseek/deepseek_ops.hpp"
#include "strata/models/glm53/glm53_checkpoint.hpp"
#include "strata/models/kimi_k3/kimi_k3_ops.hpp"
#include "strata/platform/hardware_profile.hpp"
#include "strata/platform/numerics.hpp"
#include "strata/platform/worker_pool.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <functional>
#include <fstream>
#include <iostream>
#include <list>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <numeric>
#include <random>
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
           << phase.graph.sampling_nanoseconds << "}}";
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

struct Glm53HostFp8Linear {
    std::span<const std::byte> weights;
    std::span<const float> scales;
    std::uint32_t rows{};
    std::uint32_t columns{};
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
constexpr std::uint32_t kMlaHead = 256U;
constexpr std::uint32_t kMlaWidth = kHeads * kMlaHead;
constexpr std::uint32_t kQueryRank = 1536U;
constexpr std::uint32_t kKvRank = 512U;
constexpr std::uint32_t kMhc = 4U;
constexpr std::uint32_t kVocabulary = 154880U;
constexpr std::uint32_t kExactSparseContext = 2048U;
constexpr std::uint64_t kKdaWorkspaceFloats =
    2ULL * kHidden + 6ULL * kLinearWidth + 2ULL * kLinearHead + kHeads;
constexpr std::uint64_t kDeviceWorkspaceReserve = 2ULL << 30U;
constexpr std::uint64_t kMinimumDeviceBudget = 2ULL << 30U;

[[nodiscard]] std::size_t prefix_cache_entries(
    std::uint32_t maximum_context_tokens) noexcept {
    const auto kda_state = 34ULL * kHeads * kLinearHead * kLinearHead *
                           sizeof(float);
    const auto convolution_state =
        34ULL * 3ULL * kLinearWidth * 3ULL * sizeof(float);
    const auto mla_state = 11ULL * maximum_context_tokens * kKvRank *
                           sizeof(float);
    const auto state_bytes = std::max<std::uint64_t>(
        kda_state + convolution_state + mla_state, 1U);
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

[[nodiscard]] bool resident_mla_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_GLM53_RESIDENT_MLA");
        // The absorbed resident MLA route remains a profiling candidate until
        // its layer-by-layer exactness gate is closed.  Never make an
        // experimental arithmetic path the production default.
        return value != nullptr && std::string_view(value) != "0" &&
               std::string_view(value) != "false" &&
               std::string_view(value) != "off";
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
// Off by default after the protected six-arm rerun. The coalesced kernel is
// exact and fast, but three device arms measured +1.71% median decode wall and
// only -0.21% host expert service against three host arms, both inside the
// observed ranges. Prefill was neutral (+0.22%); initialization consistently
// paid about 0.24 s and the tier consumes 1.06 GB. The exact primitive remains
// for an explicitly selected resident policy. `1` opts in.
[[nodiscard]] int shared_expert_device_override() noexcept {
    const char* value = std::getenv("STRATA_GLM53_SHARED_EXPERT_DEVICE");
    if (value == nullptr || std::string_view(value) == "auto") return 0;
    return std::string_view(value) != "0" &&
                   std::string_view(value) != "false" &&
                   std::string_view(value) != "off"
               ? 1 : 0;
}

// Measurement-only bound: service at most this many of the nine experts a MoE
// layer needs. Output is WRONG by construction -- the dropped experts simply do
// not contribute -- so this is never a candidate configuration. It exists to
// price the marginal cost of an expert in the host dispatch, which is the
// number both M3's shared-expert lever and M4's hot-tier plan are betting on.
// 0 or unset leaves the production path untouched.
[[nodiscard]] std::size_t host_expert_bound() noexcept {
    static const std::size_t bound = [] {
        const char* value = std::getenv("STRATA_GLM53_EXPERT_BOUND");
        if (value == nullptr) return std::size_t{0U};
        const auto parsed = std::strtoul(value, nullptr, 10);
        return static_cast<std::size_t>(parsed);
    }();
    return bound;
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
        std::array<CudaBuffer, kLayers> kda;
        std::array<CudaBuffer, kLayers> mla;
        bool ready{};
    };

    struct ResidentLayerWeights {
        CudaDsv4MhcWeights attention;
        CudaDsv4MhcWeights feedforward;
    };

    struct PrefixEntry {
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
    // competes for the routed cache. It is also exactly one ninth of host
    // expert service, which M0 attributed at 67.7% of a decode step, while the
    // GPUs sit idle for 94% of that step. Moving it across costs 25.17 MB of
    // VRAM per layer -- 1.06 GB for the tier -- and the device dot associates
    // its sum exactly as the host AVX2 dot does, so the output is unchanged.
    struct SharedExpertTier {
        // Six buffers per admitted expert, in checkpoint-native FP8 with F32
        // block scales. They own the device memory the descriptors point into.
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
    std::vector<float> shared_expert_gate;
    std::vector<float> shared_expert_up;
    std::vector<float> shared_expert_output;
    // Paged-primitive scratch, same ownership reasoning. `page_groups` keeps
    // its per-group assignment vectors alive between calls so their capacity is
    // reused; only `clear()` is called on them, never destruction.
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
    struct Glm53ExpertViews {
        Glm53HostFp8Linear gate;
        Glm53HostFp8Linear up;
        Glm53HostFp8Linear down;
    };
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
    bool full_tensor_parallel_active{};
    std::unique_ptr<HostWorkerPool> kda_workers;
    std::atomic<std::uint64_t> parallel_projection_batches{};
    std::atomic<std::uint64_t> parallel_projection_requests{};
    std::atomic<std::uint64_t> tensor_parallel_head_batches{};
    std::atomic<std::uint64_t> parallel_encode_pages{};
    std::atomic<std::uint64_t> prefix_cache_hits{};
    std::atomic<std::uint64_t> prefix_cache_tokens{};
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
    bool scheduler_stopping{};
    std::atomic<std::uint64_t> scheduler_iterations{};
    std::atomic<std::uint64_t> scheduler_batched_iterations{};
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

    [[nodiscard]] ProfileSnapshot profile_snapshot() const {
        if (!config.phase_profile) return {};
        return {cuda.stats(), cache_metrics(), host_expert_metrics(),
                graph_metrics()};
    }

    [[nodiscard]] static Glm53PhaseMetrics phase_delta(
        const ProfileSnapshot& after, const ProfileSnapshot& before) {
        return {detail::cuda_delta(after.cuda, before.cuda),
                cache_delta(after.cache, before.cache),
                host_expert_delta(after.host_experts, before.host_experts),
                graph_delta(after.graph, before.graph)};
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
            auto gate = host_fp8_linear(module + "gate_proj", 2048U, kHidden);
            auto up = host_fp8_linear(module + "up_proj", 2048U, kHidden);
            auto down = host_fp8_linear(module + "down_proj", kHidden, 2048U);
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

    [[nodiscard]] ParseResult<Glm53HostFp8Linear> host_fp8_linear(
        std::string_view base, std::uint32_t rows,
        std::uint32_t columns) const {
        ParseResult<Glm53HostFp8Linear> result;
        const auto weight_name = std::string(base) + ".weight";
        const auto scale_name = std::string(base) + ".weight_scale_inv";
        const auto* descriptor = checkpoint->find(weight_name);
        const auto* scale_descriptor = checkpoint->find(scale_name);
        const auto scale_rows = (rows + 127U) / 128U;
        const auto scale_columns = (columns + 127U) / 128U;
        if (descriptor == nullptr || scale_descriptor == nullptr ||
            descriptor->source_dtype != SafetensorsDtype::F8E4M3 ||
            descriptor->source_shape !=
                std::vector<std::uint64_t>{rows, columns} ||
            scale_descriptor->source_dtype != SafetensorsDtype::F32 ||
            scale_descriptor->source_shape !=
                std::vector<std::uint64_t>{scale_rows, scale_columns}) {
            result.errors.push_back(
                "GLM-5.3 host expert has an invalid FP8 linear: " +
                std::string(base));
            return result;
        }
        auto weight_payload = checkpoint->view(weight_name);
        auto scales = checkpoint->view(scale_name);
        if (!weight_payload.ok()) {
            result.errors = std::move(weight_payload.errors);
            return result;
        }
        if (!scales.ok()) {
            result.errors = std::move(scales.errors);
            return result;
        }
        if (weight_payload.value.size_bytes() !=
                static_cast<std::size_t>(rows) * columns ||
            scales.value.size_bytes() !=
                static_cast<std::size_t>(scale_rows) * scale_columns *
                    sizeof(float) ||
            reinterpret_cast<std::uintptr_t>(scales.value.data()) %
                    alignof(float) != 0U) {
            result.errors.push_back(
                "GLM-5.3 host expert mapped payload is mis-sized");
            return result;
        }
        result.value = {
            weight_payload.value,
            std::span<const float>(
                reinterpret_cast<const float*>(scales.value.data()),
                static_cast<std::size_t>(scale_rows) * scale_columns),
            rows, columns};
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
        // gate and up are intermediate x hidden, down is hidden x
        // intermediate, and each carries one F32 scale per 128x128 tile.
        const std::uint64_t projection_bytes =
            static_cast<std::uint64_t>(intermediate) * kHidden;
        const std::uint64_t scale_bytes =
            static_cast<std::uint64_t>(intermediate / 128U) *
            (kHidden / 128U) * sizeof(float);
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
                auto linear = host_fp8_linear(projections[index].name,
                                              projections[index].rows,
                                              projections[index].columns);
                if (!linear.ok()) return {std::move(linear.errors)};
                auto& weight_buffer = shared_experts.storage[cursor++];
                auto& scale_buffer = shared_experts.storage[cursor++];
                auto weight_upload = cuda.upload_buffer(
                    device, linear.value.weights, weight_buffer);
                if (!weight_upload.ok()) return weight_upload;
                auto scale_upload = cuda.upload_buffer(
                    device,
                    std::as_bytes(linear.value.scales), scale_buffer);
                if (!scale_upload.ok()) return scale_upload;
                uploaded[index * 2U] = &weight_buffer;
                uploaded[index * 2U + 1U] = &scale_buffer;
                shared_experts.bytes += linear.value.weights.size_bytes() +
                                        linear.value.scales.size_bytes();
                shared_experts.bytes_by_slot[slot] +=
                    linear.value.weights.size_bytes() +
                    linear.value.scales.size_bytes();
            }
            shared_experts.experts[layer] = {
                uploaded[0], uploaded[1], uploaded[2],
                uploaded[3], uploaded[4], uploaded[5],
                kHidden, intermediate};
            shared_experts.devices[layer] = device;
        }
        shared_experts.active = true;
        return {};
    }

    [[nodiscard]] ValidationResult host_moe(
        std::uint32_t layer, std::string_view prefix,
        std::span<const KimiRoutedExpert> routed,
        std::span<const float> input, std::span<float> output) {
        ValidationResult result;
        constexpr std::uint32_t intermediate = 2048U;
        // The shared expert is the ninth of the nine this layer runs and the
        // only one the router does not choose, so it can be computed on the
        // idle GPU while the host works through the eight routed ones. When
        // its tier is admitted the host runs eight; otherwise it runs all
        // nine, exactly as before.
        const int shared_device = shared_experts.active &&
                                  layer < shared_experts.devices.size()
            ? shared_experts.devices[layer] : -1;
        std::size_t expert_count = shared_device >= 0 ? 8U : 9U;
        const auto bound = host_expert_bound();
        if (bound != 0U) expert_count = std::min(expert_count, bound);
        if (shared_experts.active) {
            if (shared_device < 0) {
                shared_expert_host_calls.fetch_add(
                    1U, std::memory_order_relaxed);
                if (layer < 64U) {
                    shared_expert_host_layers.fetch_or(
                        std::uint64_t{1U} << layer, std::memory_order_relaxed);
                }
            } else {
                shared_expert_device_calls.fetch_add(
                    1U, std::memory_order_relaxed);
            }
        }
        if (!host_moe_active || host_moe_workers == nullptr ||
            routed.size() != 8U || input.size() != kHidden ||
            output.size() != kHidden) {
            return {{"GLM-5.3 host MoE command has an invalid shape"}};
        }
        const auto started = std::chrono::steady_clock::now();
        const auto view_started = started;
        struct Expert {
            Glm53HostFp8Linear gate;
            Glm53HostFp8Linear up;
            Glm53HostFp8Linear down;
        };
        std::array<Expert, 9U> experts;
        for (std::size_t index = 0U; index < expert_count; ++index) {
            const auto slot = index < routed.size()
                ? static_cast<std::uint32_t>(routed[index].expert)
                : kExpertSlots - 1U;
            Glm53ExpertViews views;
            result = expert_views(layer, slot, prefix, views);
            if (!result.ok()) return result;
            experts[index] = {views.gate, views.up, views.down};
        }
        std::uint64_t allocations = 0U;
        if (config.phase_profile) {
            std::uint64_t gate_up_bytes = 0U;
            std::uint64_t down_bytes = 0U;
            for (std::size_t index = 0U; index < expert_count; ++index) {
                const auto& expert = experts[index];
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
        glm53_quantize_activation(
            std::span<float>(quantized_input).first(input.size()));
        if (config.phase_profile) {
            host_moe_input_quantization_nanoseconds.fetch_add(
                elapsed_nanoseconds(input_quantization_started),
                std::memory_order_relaxed);
        }
        // Enqueued before the host dispatch, collected after it: the device
        // gate and up run in the shadow of the eight routed experts.
        if (shared_device >= 0) {
            result = cuda.enqueue_glm53_expert_gate_up(
                shared_device,
                std::span<const CudaGlm53Expert>(
                    &shared_experts.experts[layer], 1U),
                std::span<const float>(quantized_input).first(kHidden));
            if (!result.ok()) return result;
        }
        auto& activations = host_moe_activations;
        if (glm53_grow(activations, expert_count * intermediate)) ++allocations;
        const auto gate_up_started = std::chrono::steady_clock::now();
        const auto gate_up = host_moe_workers->parallel_for_blocked(
            expert_count * intermediate, expert_dispatch_block(),
            [&](std::size_t task) {
                const auto expert = task / intermediate;
                const auto row = task % intermediate;
                const auto& module = experts[expert];
                const auto scale_columns = kHidden / 128U;
                const auto* gate_weights = module.gate.weights.data() +
                    row * kHidden;
                const auto* up_weights = module.up.weights.data() +
                    row * kHidden;
                const auto* gate_scales = module.gate.scales.data() +
                    (row / 128U) * scale_columns;
                const auto* up_scales = module.up.scales.data() +
                    (row / 128U) * scale_columns;
                auto gate = bf16_round_f32(glm53_host_fp8_dot(
                    gate_weights, gate_scales, quantized_input));
                auto up = bf16_round_f32(glm53_host_fp8_dot(
                    up_weights, up_scales, quantized_input));
                gate = std::min(gate, 10.0F);
                up = std::clamp(up, -10.0F, 10.0F);
                activations[expert * intermediate + row] =
                    bf16_round_f32(gate * sigmoid(gate) * up);
            });
        if (!gate_up.ok()) return gate_up;
        if (config.phase_profile) {
            host_moe_gate_up_nanoseconds.fetch_add(
                elapsed_nanoseconds(gate_up_started),
                std::memory_order_relaxed);
        }
        const auto activation_started = std::chrono::steady_clock::now();
        for (std::size_t expert = 0U; expert < expert_count; ++expert) {
            glm53_quantize_activation(std::span<float>(activations).subspan(
                expert * intermediate, intermediate));
        }
        // The device returns the two raw dots. Every rounding, the clamp and
        // the SwiGLU stay here, on the same code the host path runs for its
        // own eight experts, so no device libm function -- expf above all --
        // enters the result.
        if (shared_device >= 0) {
            if (glm53_grow(shared_expert_gate, intermediate)) ++allocations;
            if (glm53_grow(shared_expert_up, intermediate)) ++allocations;
            result = cuda.collect_glm53_expert_gate_up(
                shared_device,
                std::span<float>(shared_expert_gate).first(intermediate),
                std::span<float>(shared_expert_up).first(intermediate));
            if (!result.ok()) return result;
            for (std::size_t row = 0U; row < intermediate; ++row) {
                auto gate = bf16_round_f32(shared_expert_gate[row]);
                auto up = bf16_round_f32(shared_expert_up[row]);
                gate = std::min(gate, 10.0F);
                up = std::clamp(up, -10.0F, 10.0F);
                shared_expert_gate[row] =
                    bf16_round_f32(gate * sigmoid(gate) * up);
            }
            glm53_quantize_activation(
                std::span<float>(shared_expert_gate).first(intermediate));
            result = cuda.enqueue_glm53_expert_down(
                shared_device,
                std::span<const CudaGlm53Expert>(
                    &shared_experts.experts[layer], 1U),
                std::span<const float>(shared_expert_gate).first(intermediate));
            if (!result.ok()) return result;
        }
        if (config.phase_profile) {
            host_moe_activation_nanoseconds.fetch_add(
                elapsed_nanoseconds(activation_started),
                std::memory_order_relaxed);
        }
        auto& expert_outputs = host_moe_expert_outputs;
        if (glm53_grow(expert_outputs, expert_count * kHidden)) ++allocations;
        if (config.phase_profile) {
            host_moe_temporary_allocation_calls.fetch_add(
                allocations, std::memory_order_relaxed);
        }
        const auto down_started = std::chrono::steady_clock::now();
        const auto down = host_moe_workers->parallel_for_blocked(
            expert_count * kHidden, expert_dispatch_block(),
            [&](std::size_t task) {
                const auto expert = task / kHidden;
                const auto row = task % kHidden;
                const auto& module = experts[expert].down;
                const auto scale_columns = intermediate / 128U;
                const auto* weight_row = module.weights.data() +
                    row * intermediate;
                const auto* scales = module.scales.data() +
                    (row / 128U) * scale_columns;
                expert_outputs[expert * kHidden + row] = bf16_round_f32(
                    glm53_host_fp8_dot(
                        weight_row, scales,
                        std::span<const float>(activations).subspan(
                            expert * intermediate, intermediate)));
            });
        if (!down.ok()) return down;
        if (config.phase_profile) {
            host_moe_down_nanoseconds.fetch_add(
                elapsed_nanoseconds(down_started), std::memory_order_relaxed);
        }
        const auto reduction_started = std::chrono::steady_clock::now();
        if (shared_device >= 0) {
            if (glm53_grow(shared_expert_output, kHidden)) ++allocations;
            result = cuda.collect_glm53_expert_down(
                shared_device,
                std::span<float>(shared_expert_output).first(kHidden));
            if (!result.ok()) return result;
            // The host down dispatch rounds its own dot to BF16; round the
            // device's the same way and the shared term is bit-identical.
            for (std::size_t column = 0U; column < kHidden; ++column) {
                output[column] = bf16_round_f32(shared_expert_output[column]);
            }
        } else if (expert_count == 9U) {
            std::copy_n(expert_outputs.begin() + 8U * kHidden, kHidden,
                        output.begin());
        } else {
            // Only reachable under the measurement bound, where the shared
            // expert was not computed and `expert_outputs` has no ninth slot.
            std::fill(output.begin(), output.end(), 0.0F);
        }
        // `expert_count` is `routed.size()` in every production configuration;
        // it is smaller only under the measurement bound, where summing an
        // expert that was never computed would read stale scratch.
        const auto reduced = std::min(routed.size(), expert_count);
        for (std::size_t expert = 0U; expert < reduced; ++expert) {
            for (std::size_t column = 0U; column < kHidden; ++column) {
                output[column] = bf16_round_f32(
                    output[column] + bf16_round_f32(
                        routed[expert].weight *
                        expert_outputs[expert * kHidden + column]));
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
            static_cast<std::uint64_t>(std::chrono::duration_cast<
                std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                          started).count()),
            std::memory_order_relaxed);
        return result;
    }

    [[nodiscard]] ValidationResult host_moe_page(
        std::uint32_t layer, std::string_view prefix,
        std::span<const std::array<KimiRoutedExpert, 8U>> routes,
        std::span<const float> input, std::span<float> output) {
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
        struct Expert {
            Glm53HostFp8Linear gate;
            Glm53HostFp8Linear up;
            Glm53HostFp8Linear down;
        };
        struct Assignment {
            std::size_t input_row{};
            std::size_t output_slot{};
        };
        struct Group {
            std::uint32_t expert{};
            bool shared{};
            Expert module;
            std::vector<Assignment> assignments;
        };

        std::array<std::size_t, 288U> group_for_expert;
        group_for_expert.fill(std::numeric_limits<std::size_t>::max());
        std::vector<Group> groups;
        groups.reserve(std::min<std::size_t>(288U, rows * routes_per_row) + 1U);
        for (std::size_t row = 0U; row < rows; ++row) {
            for (std::size_t route = 0U; route < routes_per_row; ++route) {
                const auto expert = routes[row][route].expert;
                if (expert >= group_for_expert.size()) {
                    return {{"GLM-5.3 host page route is out of range"}};
                }
                auto& group_index = group_for_expert[expert];
                if (group_index == std::numeric_limits<std::size_t>::max()) {
                    group_index = groups.size();
                    groups.push_back({expert, false, {}, {}});
                }
                groups[group_index].assignments.push_back(
                    {row, row * outputs_per_row + route});
            }
        }
        std::uint64_t allocations = 0U;
        groups.push_back({0U, true, {}, {}});
        auto& shared = groups.back();
        shared.assignments.reserve(rows);
        for (std::size_t row = 0U; row < rows; ++row) {
            shared.assignments.push_back(
                {row, row * outputs_per_row + routes_per_row});
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
            host_moe_temporary_allocation_calls.fetch_add(
                allocations, std::memory_order_relaxed);
        }

        const auto input_quantization_started =
            std::chrono::steady_clock::now();
        auto& quantized_input = page_quantized_input;
        if (glm53_grow(quantized_input, input.size())) ++allocations;
        std::copy(input.begin(), input.end(), quantized_input.begin());
        result = host_moe_workers->parallel_for(rows, [&](std::size_t row) {
            glm53_quantize_activation(
                std::span<float>(quantized_input)
                    .subspan(row * kHidden, kHidden));
        });
        if (!result.ok()) return result;
        if (config.phase_profile) {
            host_moe_input_quantization_nanoseconds.fetch_add(
                elapsed_nanoseconds(input_quantization_started),
                std::memory_order_relaxed);
        }

        const auto output_slots = rows * outputs_per_row;
        auto& activations = page_activations;
        if (glm53_grow(activations, output_slots * intermediate)) ++allocations;
        const auto gate_up_started = std::chrono::steady_clock::now();
        result = host_moe_workers->parallel_for_blocked(
            groups.size() * intermediate, expert_dispatch_block(),
            [&](std::size_t task) {
                const auto group_index = task / intermediate;
                const auto projection_row = task % intermediate;
                const auto& group = groups[group_index];
                const auto scale_columns = kHidden / 128U;
                const auto* gate_weights = group.module.gate.weights.data() +
                    projection_row * kHidden;
                const auto* up_weights = group.module.up.weights.data() +
                    projection_row * kHidden;
                const auto* gate_scales = group.module.gate.scales.data() +
                    (projection_row / 128U) * scale_columns;
                const auto* up_scales = group.module.up.scales.data() +
                    (projection_row / 128U) * scale_columns;
                for (const auto& assignment : group.assignments) {
                    const auto source = std::span<const float>(quantized_input)
                        .subspan(assignment.input_row * kHidden, kHidden);
                    auto gate = bf16_round_f32(glm53_host_fp8_dot(
                        gate_weights, gate_scales, source));
                    auto up = bf16_round_f32(glm53_host_fp8_dot(
                        up_weights, up_scales, source));
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
        result = host_moe_workers->parallel_for(
            output_slots, [&](std::size_t slot) {
                glm53_quantize_activation(
                    std::span<float>(activations)
                        .subspan(slot * intermediate, intermediate));
            });
        if (!result.ok()) return result;
        if (config.phase_profile) {
            host_moe_activation_nanoseconds.fetch_add(
                elapsed_nanoseconds(activation_started),
                std::memory_order_relaxed);
        }

        auto& expert_outputs = page_expert_outputs;
        if (glm53_grow(expert_outputs, output_slots * kHidden)) ++allocations;
        const auto down_started = std::chrono::steady_clock::now();
        result = host_moe_workers->parallel_for_blocked(
            groups.size() * kHidden, expert_dispatch_block(),
            [&](std::size_t task) {
                const auto group_index = task / kHidden;
                const auto projection_row = task % kHidden;
                const auto& group = groups[group_index];
                const auto& projection = group.module.down;
                const auto scale_columns = intermediate / 128U;
                const auto* projection_weights = projection.weights.data() +
                    projection_row * intermediate;
                const auto* projection_scales = projection.scales.data() +
                    (projection_row / 128U) * scale_columns;
                for (const auto& assignment : group.assignments) {
                    expert_outputs[assignment.output_slot * kHidden +
                                   projection_row] = bf16_round_f32(
                        glm53_host_fp8_dot(
                            projection_weights, projection_scales,
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

    void store_prefix(std::span<const std::uint32_t> tokens,
                      const Glm53SequenceState& state,
                      std::span<const float> logits,
                      std::span<const float> base_hidden) {
        if (tokens.empty() || state.token_count() != tokens.size()) return;
        std::scoped_lock lock(prefix_mutex);
        for (auto& entry : prefix_cache) {
            if (entry.tokens.size() == tokens.size() &&
                std::equal(entry.tokens.begin(), entry.tokens.end(),
                           tokens.begin())) {
                entry.state = state;
                entry.logits.assign(logits.begin(), logits.end());
                entry.base_hidden.assign(base_hidden.begin(), base_hidden.end());
                entry.recency = ++prefix_clock;
                return;
            }
        }
        if (prefix_cache.size() >= prefix_cache_limit) {
            const auto victim = std::min_element(
                prefix_cache.begin(), prefix_cache.end(),
                [](const PrefixEntry& left, const PrefixEntry& right) {
                    return left.recency < right.recency;
                });
            if (victim != prefix_cache.end()) prefix_cache.erase(victim);
        }
        PrefixEntry entry;
        entry.tokens.assign(tokens.begin(), tokens.end());
        entry.state = state;
        entry.logits.assign(logits.begin(), logits.end());
        entry.base_hidden.assign(base_hidden.begin(), base_hidden.end());
        entry.recency = ++prefix_clock;
        prefix_cache.push_back(std::move(entry));
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
        // The chunk form exposes heads to the physical-core pool, but a page
        // narrower than the runner count cannot amortize waking that pool.
        // Derive the crossover from the discovered pool width rather than a
        // token constant measured on one host.
        if (kda_workers != nullptr &&
            rows >= std::min<std::size_t>(kHeads, kda_workers->size())) {
            std::vector<ValidationResult> failures(kHeads);
            auto replayed = kda_workers->parallel_for(
                kHeads, [&](std::size_t head) {
                    std::vector<float> q(static_cast<std::size_t>(rows) *
                                         kLinearHead);
                    std::vector<float> k(q.size()), v(q.size()), decay(q.size());
                    std::vector<float> head_beta(rows);
                    for (std::uint32_t row = 0U; row < rows; ++row) {
                        const auto source = static_cast<std::size_t>(row) *
                                                kLinearWidth +
                                            head * kLinearHead;
                        const auto target = static_cast<std::size_t>(row) *
                                            kLinearHead;
                        auto q_row = std::span<float>(q).subspan(
                            target, kLinearHead);
                        auto k_row = std::span<float>(k).subspan(
                            target, kLinearHead);
                        std::copy_n(query.begin() +
                                        static_cast<std::ptrdiff_t>(source),
                                    kLinearHead, q_row.begin());
                        std::copy_n(key.begin() +
                                        static_cast<std::ptrdiff_t>(source),
                                    kLinearHead, k_row.begin());
                        std::copy_n(value.begin() +
                                        static_cast<std::ptrdiff_t>(source),
                                    kLinearHead,
                                    v.begin() + static_cast<std::ptrdiff_t>(target));
                        auto status = kimi_l2_normalize(q_row, 1.0e-6F);
                        if (!status.ok()) {
                            failures[head] = std::move(status);
                            return;
                        }
                        status = kimi_l2_normalize(k_row, 1.0e-6F);
                        if (!status.ok()) {
                            failures[head] = std::move(status);
                            return;
                        }
                        for (auto& element : q_row) element *= query_scale;
                        status = kimi_kda_log_decay(
                            std::span<float>(decay).subspan(target, kLinearHead),
                            std::span<const float>(forget).subspan(
                                source, kLinearHead),
                            std::span<const float>(*dt_bias.value).subspan(
                                head * kLinearHead, kLinearHead),
                            (*a_log.value)[head], -5.0F);
                        if (!status.ok()) {
                            failures[head] = std::move(status);
                            return;
                        }
                        head_beta[row] = beta[
                            static_cast<std::size_t>(row) * kHeads + head];
                    }
                    std::vector<float> raw(q.size());
                    auto state = recurrent.subspan(
                        head * kLinearHead * kLinearHead,
                        static_cast<std::size_t>(kLinearHead) * kLinearHead);
                    auto status = kimi_kda_chunk(
                        raw, state, q, k, v, decay, head_beta, rows,
                        kLinearHead, kLinearHead);
                    if (!status.ok()) {
                        failures[head] = std::move(status);
                        return;
                    }
                    for (std::uint32_t row = 0U; row < rows; ++row) {
                        const auto source = static_cast<std::size_t>(row) *
                                            kLinearHead;
                        const auto target = static_cast<std::size_t>(row) *
                                                kLinearWidth +
                                            head * kLinearHead;
                        auto raw_row = std::span<float>(raw).subspan(
                            source, kLinearHead);
                        round_bf16(raw_row);
                        status = kimi_kda_output_norm(
                            std::span<float>(heads_out).subspan(
                                target, kLinearHead),
                            raw_row,
                            std::span<const float>(gate).subspan(
                                target, kLinearHead),
                            *o_norm.value, 1.0e-5F);
                        if (!status.ok()) {
                            failures[head] = std::move(status);
                            return;
                        }
                        round_bf16(std::span<float>(heads_out).subspan(
                            target, kLinearHead));
                    }
                });
            if (!replayed.ok()) return replayed;
            for (auto& failure : failures) {
                if (!failure.ok()) return failure;
            }
        } else {
            for (std::uint32_t row = 0U; row < rows; ++row) {
                const auto row_begin = static_cast<std::size_t>(row) *
                                       kLinearWidth;
                for (std::uint32_t head = 0U; head < kHeads; ++head) {
                    const auto begin = row_begin +
                        static_cast<std::size_t>(head) * kLinearHead;
                    auto q = std::span<float>(query).subspan(begin, kLinearHead);
                    auto k = std::span<float>(key).subspan(begin, kLinearHead);
                    result = kimi_l2_normalize(q, 1.0e-6F);
                    if (!result.ok()) return result;
                    result = kimi_l2_normalize(k, 1.0e-6F);
                    if (!result.ok()) return result;
                    for (auto& element : q) element *= query_scale;
                    std::vector<float> decay(kLinearHead);
                    result = kimi_kda_log_decay(
                        decay,
                        std::span<const float>(forget).subspan(
                            begin, kLinearHead),
                        std::span<const float>(*dt_bias.value).subspan(
                            static_cast<std::size_t>(head) * kLinearHead,
                            kLinearHead),
                        (*a_log.value)[head], -5.0F);
                    if (!result.ok()) return result;
                    for (auto& element : decay) element = std::exp(element);
                    std::vector<float> raw(kLinearHead);
                    auto state = recurrent.subspan(
                        static_cast<std::size_t>(head) * kLinearHead *
                            kLinearHead,
                        static_cast<std::size_t>(kLinearHead) * kLinearHead);
                    result = kimi_kda_step(
                        raw, state, q, k,
                        std::span<const float>(value).subspan(
                            begin, kLinearHead),
                        decay,
                        beta[static_cast<std::size_t>(row) * kHeads + head],
                        kLinearHead, kLinearHead);
                    if (!result.ok()) return result;
                    round_bf16(raw);
                    result = kimi_kda_output_norm(
                        std::span<float>(heads_out).subspan(
                            begin, kLinearHead),
                        raw,
                        std::span<const float>(gate).subspan(
                            begin, kLinearHead),
                        *o_norm.value, 1.0e-5F);
                    if (!result.ok()) return result;
                    round_bf16(std::span<float>(heads_out).subspan(
                        begin, kLinearHead));
                }
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
        std::vector<float> expanded(
            static_cast<std::size_t>(history) * kHeads * 2U * kMlaHead);
        const std::array<std::string, 2U> second_bases{
            attention + "q_b_proj", attention + "kv_b_proj"};
        const auto latent_storage = cache.materialize();
        const auto latent_history = std::span<const float>(latent_storage);
        const std::array<Glm53WeightCache::LinearRequest, 2U> second{
            {{second_bases[0], kMlaWidth, kQueryRank, q_rank, 1U, query, true},
             {second_bases[1], kHeads * 2U * kMlaHead, kKvRank, latent_history,
              history, expanded, true}}};
        result = linear_batch(second, layer);
        if (!result.ok()) return result;
        std::vector<float> attended(kMlaWidth, 0.0F);
        const auto score_scale = 1.0F / std::sqrt(static_cast<float>(kMlaHead));
        std::vector<float> scores(history);
        for (std::uint32_t head = 0U; head < kHeads; ++head) {
            const auto* q = query.data() + static_cast<std::size_t>(head) * kMlaHead;
            float highest = -std::numeric_limits<float>::infinity();
            for (std::uint32_t token = 0U; token < history; ++token) {
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
                                static_cast<std::size_t>(head) * kMlaHead;
            for (std::uint32_t token = 0U; token < history; ++token) {
                const auto* values = expanded.data() +
                    (static_cast<std::size_t>(token) * kHeads + head) *
                        (2U * kMlaHead) + kMlaHead;
                const auto coefficient = bf16_round_f32(scores[token] / total);
                for (std::uint32_t column = 0U; column < kMlaHead; ++column) {
                    destination[column] += coefficient * values[column];
                }
            }
        }
        round_bf16(attended);
        return linear(attention + "o_proj", attended, 1U, kMlaWidth,
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
        const auto latent_history = cache.materialize();
        std::vector<float> expanded(
            static_cast<std::size_t>(history_rows) * kHeads * 2U * kMlaHead);
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
        for (std::uint32_t row = 0U; row < rows; ++row) {
            const auto visible = history_begin + row + 1U;
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
                    (static_cast<std::size_t>(row) * kHeads + head) * kMlaHead;
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
        bool schedule_prefetch = false, bool route_prefill = false) {
        if ((!route_requests.empty() || !route_positions.empty()) &&
            (route_requests.size() != rows || route_positions.size() != rows)) {
            return {{"GLM-5.3 route-observation page has an invalid shape"}};
        }
        if (layer != kMtpLayer && !glm53_moe_layer(layer)) {
            return swiglu_block_page(output, input, prefix + "mlp.", rows,
                                     12288U, layer);
        }
        ValidationResult result;
        std::vector<float> logits(static_cast<std::size_t>(rows) * 288U);
        result = linear(prefix + "mlp.gate", input, rows, kHidden, logits,
                        layer, false);
        if (!result.ok()) return result;
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
                                 output);
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
            // The control arm: device mHC produced this layer's input, and the
            // accepted host fallback consumes it. Only the attention moves.
            std::vector<float> normalized(kHidden), branch(kHidden);
            result = cuda.dsv4_mhc_download_layer_input(device, normalized);
            if (!result.ok()) return result;
            result = attention_mla(branch, normalized, layer, position,
                                   attention, sequence);
            if (!result.ok()) return result;
            result = cuda.dsv4_mhc_publish_branch(device, branch);
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
        auto result = finish_streams_page(streams, rows, logits);
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
        bool all_row_logits = false) {
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
            result = forward_layer_page(
                streams, static_cast<std::uint32_t>(tokens.size()), layer,
                sequence);
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
        request->prefill_started = now_seconds();
        request->profile_started = profile_snapshot();
        request->prepared = true;
        return true;
    }

    [[nodiscard]] ValidationResult prepare_device_sequence(
        Glm53SequenceState& sequence, DeviceSequenceState& device_sequence) {
        ValidationResult result;
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            const auto attention = "model.language_model.layers." +
                std::to_string(layer) + ".self_attn.";
            if (!glm53_kda_layer(layer)) {
                const auto cache_floats =
                    static_cast<std::size_t>(config.maximum_context_tokens) *
                    kKvRank;
                std::vector<float> packed(
                    cache_floats + kQueryRank + kKvRank, 0.0F);
                const auto latent = sequence.mla(layer).materialize();
                if (latent.size() > cache_floats) {
                    return {{"GLM-5.3 resident MLA cache exceeds its admitted "
                             "context"}};
                }
                std::copy(latent.begin(), latent.end(), packed.begin());
                auto q_norm = host_tensor(
                    attention + "q_a_layernorm.weight", kQueryRank);
                auto kv_norm = host_tensor(
                    attention + "kv_a_layernorm.weight", kKvRank);
                if (!q_norm.ok() || !kv_norm.ok()) {
                    append(result.errors, std::move(q_norm.errors));
                    append(result.errors, std::move(kv_norm.errors));
                    return result;
                }
                std::copy(q_norm.value->begin(), q_norm.value->end(),
                          packed.begin() +
                              static_cast<std::ptrdiff_t>(cache_floats));
                std::copy(kv_norm.value->begin(), kv_norm.value->end(),
                          packed.begin() + static_cast<std::ptrdiff_t>(
                                               cache_floats + kQueryRank));
                result = cuda.upload_buffer(
                    device_for(layer),
                    std::as_bytes(std::span<const float>(packed)),
                    device_sequence.mla[layer]);
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
        device_sequence.ready = true;
        return result;
    }

    void finish_prefill(const std::shared_ptr<ScheduledRequest>& request) {
        request->result.metrics.prefill_tokens =
            request->prompt.size() -
            request->result.metrics.reused_prompt_tokens;
        request->result.metrics.prefill_seconds =
            now_seconds() - request->prefill_started;
        request->profile_after_prefill = profile_snapshot();
        if (config.phase_profile) {
            request->result.metrics.prefill = phase_delta(
                request->profile_after_prefill, request->profile_started);
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
        if (request->maximum_new_tokens > 1U && fused_kda_enabled()) {
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
            request->logits, request->sequence, &request->base_hidden);
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
                        auto step = forward_token_batch(
                            step_tokens, step_positions, step_sequences,
                            step_device_sequences, step_logits, step_hidden);
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
        config.maximum_context_tokens > kExactSparseContext) {
        result.errors.push_back(
            "GLM-5.3 text context must be within [1, 2048]; above 2048 the "
            "checkpoint's exact k-pool sparse indexer is required");
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
    impl_->prefix_cache_limit =
        prefix_cache_entries(config.maximum_context_tokens);
    impl_->scheduler_capacity = std::max<std::size_t>(
        1U, std::min<std::size_t>(32U, impl_->prefix_cache_limit));
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
    const std::string expert_prefix =
        "model.language_model.layers.3.mlp.experts.0.";
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
        auto admitted = impl_->admit_shared_experts();
        if (!admitted.ok()) return admitted;
    }
    // Allocate the arena only after every independently allocated resident
    // tier has reduced its capacity. This makes the physical allocation match
    // the admission ledger; reducing only the cache's logical capacity after
    // cudaMalloc would spend the workspace/safety reserve a second time.
    for (std::size_t slot = 0U; slot < impl_->devices.size(); ++slot) {
        result = impl_->cuda.reserve_weight_arena(
            impl_->devices[slot], impl_->weight_capacities[slot]);
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
                  << (impl_->host_moe_active ? "host-fp8" : "cuda-lru")
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

}  // namespace strata
