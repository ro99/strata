#include "strata/models/deepseek/deepseek_runtime.hpp"

#include "../common/cuda_stats_delta.hpp"

#include "strata/models/deepseek/deepseek_ops.hpp"
#include "strata/models/deepseek/deepseek_host_expert.hpp"
#include "strata/models/deepseek/deepseek_attention_kv.hpp"
#include "strata/models/deepseek/deepseek_rank_local_kv.hpp"
#include "strata/models/deepseek/deepseek_rank_local_weights.hpp"
// The two-rank executor is a CUDA translation unit compiled only when NCCL is
// available. Rank-local decode already requires NCCL at config validation, so
// the session it owns is guarded on the same condition rather than declared
// and left unlinkable.
#if defined(STRATA_HAS_NCCL)
#include "strata/models/deepseek/deepseek_rank_local_layer_executor.hpp"
#endif
#include "strata/models/common/model_adapter.hpp"
#include "strata/platform/hardware_profile.hpp"
#include "strata/platform/numa_topology.hpp"
#include "strata/platform/numerics.hpp"
#include "strata/engine/route_predictor.hpp"
#include "strata/engine/sampling.hpp"
#include "strata/engine/runtime_support.hpp"
#include "strata/models/common/tokenizer.hpp"
#include "strata/platform/trace.hpp"
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
#include <list>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <numbers>
#include <numeric>
#include <optional>
#include <span>
#include <random>
#include <sched.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace strata {

namespace {

constexpr std::uint32_t kHidden = kDeepSeekV4ExecutionContract.hidden_size;
// Intermediate-dimension TP shards the resident arena transforms each routed
// expert into. The device reads the same shards.
constexpr std::uint32_t kResidentExpertShards = 2U;
constexpr std::uint32_t kLayers = kDeepSeekV4ExecutionContract.layer_count;
constexpr std::uint32_t kHeads = kDeepSeekV4ExecutionContract.attention_heads;
constexpr std::uint32_t kHeadDim = kDeepSeekV4ExecutionContract.head_dim;
constexpr std::uint32_t kRopeDim = kDeepSeekV4ExecutionContract.rope_head_dim;
constexpr std::uint32_t kQueryRank = kDeepSeekV4ExecutionContract.query_lora_rank;
constexpr std::uint32_t kOutputRank = kDeepSeekV4ExecutionContract.output_lora_rank;
constexpr std::uint32_t kOutputGroups = kDeepSeekV4ExecutionContract.output_groups;
constexpr std::uint32_t kWindow = kDeepSeekV4ExecutionContract.sliding_window;
constexpr std::uint32_t kIndexHeads = kDeepSeekV4ExecutionContract.index_heads;
constexpr std::uint32_t kIndexHeadDim = kDeepSeekV4ExecutionContract.index_head_dim;
constexpr std::uint32_t kIndexTopK = kDeepSeekV4ExecutionContract.index_topk;
// Applied to every index weight after its projection rounds to BF16, and the
// product rounds again.
constexpr float kIndexQueryScale =
    1.0F / std::sqrt(static_cast<float>(kIndexHeadDim * kIndexHeads));
constexpr std::uint32_t kPhysicalPagedHeads = 32U;
constexpr std::uint32_t kExperts = kDeepSeekV4ExecutionContract.routed_experts;
constexpr std::uint32_t kTopK = kDeepSeekV4ExecutionContract.experts_per_token;
constexpr std::uint32_t kExpertIntermediate =
    kDeepSeekV4ExecutionContract.expert_intermediate_size;
constexpr std::uint32_t kVocabulary = kDeepSeekV4ExecutionContract.vocabulary_size;
constexpr std::uint32_t kMhc = kDeepSeekV4ExecutionContract.mhc_multiplier;
constexpr std::uint32_t kMix = kDeepSeekV4ExecutionContract.mix_width;
constexpr std::uint64_t kDeviceWorkspaceReserve = 256ULL << 20U;
// A page is what a routed expert's upload is amortised over, so the bound is
// the working set a page needs, not a dispatch limit. At 8,192 rows that is
// 537 MB of mHC state plus 402 MB of layer input, branch and MoE rows, all
// host-side and inside the admitted ceiling.
constexpr std::uint32_t kMaximumPrefillPageTokens = 8192U;
constexpr float kRmsEpsilon = kDeepSeekV4ExecutionContract.rms_epsilon;
constexpr float kAttentionScale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
static_assert(kHeads == 2U * kPhysicalPagedHeads);

// The rank-local per-device VRAM ceiling, as a fraction of what the card
// actually reports rather than the byte count measured on one 24 GiB card.
// Zero when the device cannot be queried, which admission treats as "no
// headroom" and rejects, never as "unlimited".
// Fills in every config field whose zero means "ask the hardware". Applied
// once at initialize so the rest of the runtime sees concrete numbers and no
// later code has to know which defaults were probed.
void resolve_hardware_defaults(Dsv4RuntimeConfig& config) noexcept {
    const auto& profile = host_hardware_profile();
    if (config.host_memory_limit_bytes == 0U) {
        config.host_memory_limit_bytes = profile.host_usable_bytes();
    }
    if (config.host_attention_threads == 0U) {
        config.host_attention_threads = profile.worker_threads(0.5);
    }
    if (config.resident_read_workers == 0U) {
        // Storage staging saturates well below core count; more readers past
        // this contend on the same queue rather than adding bandwidth.
        config.resident_read_workers =
            std::min<std::uint32_t>(profile.worker_threads(0.15), 8U);
    }
    if (config.spine_warmup_workers == 0U) {
        config.spine_warmup_workers =
            std::min<std::uint32_t>(profile.worker_threads(0.05), 3U);
    }
}

[[nodiscard]] std::uint64_t rank_local_vram_ceiling(int device) noexcept {
    const auto memory = CudaBackend::device_memory(device);
    if (!memory.ok()) return 0U;
    return dsv4_rank_local_vram_ceiling(memory.value.total_bytes);
}

[[nodiscard]] std::string layer_prefix(std::uint32_t layer) {
    return "layers." + std::to_string(layer) + ".";
}

void append_errors(ValidationResult& result, std::vector<std::string> errors,
                   std::string_view context = {}) {
    for (auto& error : errors) {
        if (!context.empty()) error = std::string(context) + ": " + error;
        result.errors.push_back(std::move(error));
    }
}

[[nodiscard]] float round_bf16(float value) noexcept {
    auto bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7F80'0000U) == 0x7F80'0000U) return value;
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return std::bit_cast<float>(bits & 0xFFFF'0000U);
}

void round_bf16(std::span<float> values) noexcept {
    for (auto& value : values) value = round_bf16(value);
}

[[nodiscard]] float quantize_e4m3(float value) noexcept {
    const float magnitude = std::min(std::abs(value), 448.0F);
    float quantized = 0.0F;
    if (magnitude < 0.015625F) {
        quantized = std::nearbyint(std::ldexp(magnitude, 9)) * std::ldexp(1.0F, -9);
    } else {
        int exponent = 0;
        static_cast<void>(std::frexp(magnitude, &exponent));
        exponent = std::clamp(exponent - 1, -6, 8);
        const float step = std::ldexp(1.0F, exponent - 3);
        quantized = std::min(std::nearbyint(magnitude / step) * step, 448.0F);
    }
    return std::copysign(quantized, value);
}

void quantize_activation_in_place(std::span<float> values,
                                  std::uint32_t group_size) {
    for (std::size_t begin = 0U; begin < values.size(); begin += group_size) {
        const auto count = std::min<std::size_t>(group_size, values.size() - begin);
        float maximum = 0.0F;
        for (std::size_t index = 0U; index < count; ++index) {
            maximum = std::max(maximum, std::abs(values[begin + index]));
        }
        float scale = 1.0F;
        if (maximum > 0.0F) {
            scale = std::exp2(std::ceil(std::log2(maximum / 448.0F)));
        }
        for (std::size_t index = 0U; index < count; ++index) {
            values[begin + index] = round_bf16(
                quantize_e4m3(values[begin + index] / scale) * scale);
        }
    }
}

[[nodiscard]] std::vector<float> rope_frequencies(std::uint32_t compression_ratio) {
    const float base = compression_ratio == 0U ? 10'000.0F : 160'000.0F;
    std::vector<float> result(kRopeDim / 2U);
    for (std::uint32_t index = 0U; index < result.size(); ++index) {
        result[index] = 1.0F /
            std::pow(base, static_cast<float>(2U * index) / static_cast<float>(kRopeDim));
    }
    if (compression_ratio == 0U) return result;
    constexpr float original = 65'536.0F;
    const auto correction = [base](float rotations) {
        return static_cast<float>(kRopeDim) *
               std::log(original / (rotations * 2.0F * std::numbers::pi_v<float>)) /
               (2.0F * std::log(base));
    };
    const float low = std::max(0.0F, std::floor(correction(32.0F)));
    const float high = std::min(static_cast<float>(kRopeDim - 1U),
                                std::ceil(correction(1.0F)));
    for (std::uint32_t index = 0U; index < result.size(); ++index) {
        const float ramp = std::clamp((static_cast<float>(index) - low) /
                                          std::max(0.001F, high - low),
                                      0.0F, 1.0F);
        const float smooth = 1.0F - ramp;
        result[index] = result[index] / 16.0F * (1.0F - smooth) +
                        result[index] * smooth;
    }
    return result;
}

void apply_rope(std::span<float> values, std::uint64_t position,
                std::span<const float> frequencies, bool inverse = false) {
    for (std::size_t index = 0U; index < frequencies.size(); ++index) {
        const float angle = static_cast<float>(position) * frequencies[index] *
                            (inverse ? -1.0F : 1.0F);
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        const float first = values[index * 2U];
        const float second = values[index * 2U + 1U];
        values[index * 2U] = first * cosine - second * sine;
        values[index * 2U + 1U] = second * cosine + first * sine;
    }
}

[[nodiscard]] Dsv4CheckpointReadStats read_delta(
    const Dsv4CheckpointReadStats& after,
    const Dsv4CheckpointReadStats& before) noexcept {
    return {after.calls - before.calls, after.bytes - before.bytes,
            after.nanoseconds - before.nanoseconds};
}

[[nodiscard]] CudaBackendStats cuda_delta(const CudaBackendStats& after,
                                           const CudaBackendStats& before) {
    return detail::cuda_delta(after, before);
}

[[nodiscard]] Dsv4CacheStats cache_delta(const Dsv4CacheStats& after,
                                         const Dsv4CacheStats& before) {
    Dsv4CacheStats result = after;
    result.hits -= before.hits;
    result.misses -= before.misses;
    result.evictions -= before.evictions;
    result.lease_acquires -= before.lease_acquires;
    result.lease_releases -= before.lease_releases;
    result.demand_h2d_bytes -= before.demand_h2d_bytes;
    result.demand_wait_nanoseconds -= before.demand_wait_nanoseconds;
    result.prefetch_requests -= before.prefetch_requests;
    result.prefetch_h2d_bytes -= before.prefetch_h2d_bytes;
    result.useful_prefetch_bytes -= before.useful_prefetch_bytes;
    result.late_prefetch_bytes -= before.late_prefetch_bytes;
    result.duplicate_prefetch_bytes -= before.duplicate_prefetch_bytes;
    result.evicted_prefetch_bytes -= before.evicted_prefetch_bytes;
    result.wasted_prefetch_bytes -= before.wasted_prefetch_bytes;
    result.cancelled_prefetch_bytes -= before.cancelled_prefetch_bytes;
    result.prefetch_lease_acquires -= before.prefetch_lease_acquires;
    result.prefetch_lease_releases -= before.prefetch_lease_releases;
    result.prefetch_queue_peak = after.prefetch_queue_peak;
    return result;
}

[[nodiscard]] Dsv4KvCacheStats kv_cache_delta(
    const Dsv4KvCacheStats& after,
    const Dsv4KvCacheStats& before) {
    Dsv4KvCacheStats result = after;
    result.allocated_blocks -= before.allocated_blocks;
    result.allocation_calls -= before.allocation_calls;
    result.allocation_nanoseconds -= before.allocation_nanoseconds;
    result.hits -= before.hits;
    result.misses -= before.misses;
    result.evictions -= before.evictions;
    result.promotions -= before.promotions;
    result.promotion_nanoseconds -= before.promotion_nanoseconds;
    result.host_to_device_bytes -= before.host_to_device_bytes;
    result.device_to_host_bytes -= before.device_to_host_bytes;
    result.host_write_bytes -= before.host_write_bytes;
    result.gather_bytes -= before.gather_bytes;
    result.copy_on_write_blocks -= before.copy_on_write_blocks;
    result.sequence_creations -= before.sequence_creations;
    result.sequence_resets -= before.sequence_resets;
    result.sequence_releases -= before.sequence_releases;
    result.sequence_truncations -= before.sequence_truncations;
    return result;
}

[[nodiscard]] Dsv4DeviceMoeStats device_moe_delta(
    const Dsv4DeviceMoeStats& after,
    const Dsv4DeviceMoeStats& before) noexcept {
    Dsv4DeviceMoeStats result;
    result.batches = after.batches - before.batches;
    result.host_callback_batches = after.host_callback_batches -
                                   before.host_callback_batches;
    result.host_callback_failures = after.host_callback_failures -
                                    before.host_callback_failures;
    result.device_join_batches = after.device_join_batches -
                                 before.device_join_batches;
    result.device_commands = after.device_commands - before.device_commands;
    result.routed_experts = after.routed_experts - before.routed_experts;
    result.shared_experts = after.shared_experts - before.shared_experts;
    result.routed_gate_up_nanoseconds = after.routed_gate_up_nanoseconds -
                                        before.routed_gate_up_nanoseconds;
    result.routed_down_nanoseconds = after.routed_down_nanoseconds -
                                     before.routed_down_nanoseconds;
    result.routed_reduce_nanoseconds = after.routed_reduce_nanoseconds -
                                       before.routed_reduce_nanoseconds;
    result.routed_cpu_nanoseconds = after.routed_cpu_nanoseconds -
                                    before.routed_cpu_nanoseconds;
    result.shared_collect_nanoseconds = after.shared_collect_nanoseconds -
                                        before.shared_collect_nanoseconds;
    result.combine_nanoseconds = after.combine_nanoseconds -
                                 before.combine_nanoseconds;
    result.nanoseconds = after.nanoseconds - before.nanoseconds;
    return result;
}

[[nodiscard]] Dsv4GraphStats graph_delta(
    const Dsv4GraphStats& after, const Dsv4GraphStats& before) noexcept {
    return {
        after.forward_tokens - before.forward_tokens,
        after.prefill_pages - before.prefill_pages,
        after.prefill_pages == before.prefill_pages
            ? 0U : after.prefill_max_page_tokens,
        after.prefill_pages == before.prefill_pages
            ? 0U : after.prefill_max_workspace_bytes,
        after.embedding_nanoseconds - before.embedding_nanoseconds,
        after.mhc_pre_nanoseconds - before.mhc_pre_nanoseconds,
        after.mhc_prepacked_calls - before.mhc_prepacked_calls,
        after.branch_norm_nanoseconds - before.branch_norm_nanoseconds,
        after.attention_nanoseconds - before.attention_nanoseconds,
        after.attention_query_nanoseconds - before.attention_query_nanoseconds,
        after.attention_kv_nanoseconds - before.attention_kv_nanoseconds,
        after.attention_query_allocation_nanoseconds -
            before.attention_query_allocation_nanoseconds,
        after.attention_query_weight_acquisition_nanoseconds -
            before.attention_query_weight_acquisition_nanoseconds,
        after.attention_query_matmul_issue_nanoseconds -
            before.attention_query_matmul_issue_nanoseconds,
        after.attention_query_matmul_finish_nanoseconds -
            before.attention_query_matmul_finish_nanoseconds,
        after.attention_query_matmul_sync_nanoseconds -
            before.attention_query_matmul_sync_nanoseconds,
        after.attention_query_matmul_h2d_nanoseconds -
            before.attention_query_matmul_h2d_nanoseconds,
        after.attention_query_matmul_kernel_nanoseconds -
            before.attention_query_matmul_kernel_nanoseconds,
        after.attention_query_matmul_d2h_nanoseconds -
            before.attention_query_matmul_d2h_nanoseconds,
        after.attention_query_rank_norm_nanoseconds -
            before.attention_query_rank_norm_nanoseconds,
        after.attention_query_finish_nanoseconds -
            before.attention_query_finish_nanoseconds,
        after.attention_query_rms_cpu_nanoseconds -
            before.attention_query_rms_cpu_nanoseconds,
        after.attention_query_rope_cpu_nanoseconds -
            before.attention_query_rope_cpu_nanoseconds,
        after.attention_kv_allocation_nanoseconds -
            before.attention_kv_allocation_nanoseconds,
        after.attention_kv_weight_acquisition_nanoseconds -
            before.attention_kv_weight_acquisition_nanoseconds,
        after.attention_kv_matmul_issue_nanoseconds -
            before.attention_kv_matmul_issue_nanoseconds,
        after.attention_kv_matmul_finish_nanoseconds -
            before.attention_kv_matmul_finish_nanoseconds,
        after.attention_kv_matmul_sync_nanoseconds -
            before.attention_kv_matmul_sync_nanoseconds,
        after.attention_kv_matmul_h2d_nanoseconds -
            before.attention_kv_matmul_h2d_nanoseconds,
        after.attention_kv_matmul_kernel_nanoseconds -
            before.attention_kv_matmul_kernel_nanoseconds,
        after.attention_kv_matmul_d2h_nanoseconds -
            before.attention_kv_matmul_d2h_nanoseconds,
        after.attention_kv_norm_nanoseconds -
            before.attention_kv_norm_nanoseconds,
        after.attention_kv_rope_nanoseconds -
            before.attention_kv_rope_nanoseconds,
        after.attention_projection_matmul_calls -
            before.attention_projection_matmul_calls,
        after.attention_projection_matmul_rows -
            before.attention_projection_matmul_rows,
        after.attention_index_nanoseconds - before.attention_index_nanoseconds,
        after.attention_index_queries - before.attention_index_queries,
        after.attention_index_candidates - before.attention_index_candidates,
        after.attention_index_selected - before.attention_index_selected,
        after.attention_index_cuda_dispatches -
            before.attention_index_cuda_dispatches,
        after.attention_index_scalar_dispatches -
            before.attention_index_scalar_dispatches,
        after.attention_cuda_dispatches - before.attention_cuda_dispatches,
        after.attention_scalar_dispatches - before.attention_scalar_dispatches,
        after.attention_page_set_builds - before.attention_page_set_builds,
        after.attention_page_set_pages - before.attention_page_set_pages,
        after.attention_page_set_build_nanoseconds -
            before.attention_page_set_build_nanoseconds,
        after.attention_candidate_resolutions -
            before.attention_candidate_resolutions,
        after.attention_candidate_resolution_nanoseconds -
            before.attention_candidate_resolution_nanoseconds,
        after.attention_page_index_selection_nanoseconds -
            before.attention_page_index_selection_nanoseconds,
        after.attention_page_weight_acquire_nanoseconds -
            before.attention_page_weight_acquire_nanoseconds,
        after.attention_score_nanoseconds - before.attention_score_nanoseconds,
        after.attention_output_nanoseconds - before.attention_output_nanoseconds,
        after.moe_nanoseconds - before.moe_nanoseconds,
        after.moe_router_nanoseconds - before.moe_router_nanoseconds,
        after.moe_prepare_nanoseconds - before.moe_prepare_nanoseconds,
        after.mhc_post_nanoseconds - before.mhc_post_nanoseconds,
        after.output_head_nanoseconds - before.output_head_nanoseconds,
        after.rank_local_layer_nanoseconds -
            before.rank_local_layer_nanoseconds,
        after.rank_local_device_nanoseconds -
            before.rank_local_device_nanoseconds,
        after.rank_local_kv_nanoseconds - before.rank_local_kv_nanoseconds,
        after.rank_local_candidate_nanoseconds -
            before.rank_local_candidate_nanoseconds,
        after.rank_local_boundary_nanoseconds -
            before.rank_local_boundary_nanoseconds,
        after.rank_local_collective_nanoseconds -
            before.rank_local_collective_nanoseconds,
        after.rank_local_transition_nanoseconds -
            before.rank_local_transition_nanoseconds,
        after.rank_local_shared_nanoseconds -
            before.rank_local_shared_nanoseconds,
    };
}

[[nodiscard]] std::uint64_t elapsed_nanoseconds(
    std::chrono::steady_clock::time_point started) noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count());
}

[[nodiscard]] std::uint64_t linear_bytes(const Dsv4CheckpointReader& checkpoint,
                                         std::string_view base) {
    const auto* weight = checkpoint.find(std::string(base) + ".weight");
    if (weight == nullptr) return 0U;
    if (base.ends_with(".attn.wo_a") &&
        weight->encoding == Dsv4TensorEncoding::Fp8E4m3Block128) {
        if (weight->source_bytes >
            std::numeric_limits<std::uint64_t>::max() / 2U) return 0U;
        return CudaBackend::weight_storage_bytes(weight->source_bytes * 2U, 0U);
    }
    std::uint64_t scale_bytes = 0U;
    if (weight->encoding != Dsv4TensorEncoding::Plain) {
        const auto* scale = checkpoint.find(std::string(base) + ".scale");
        if (scale == nullptr) return 0U;
        scale_bytes = scale->source_bytes;
    }
    return CudaBackend::weight_storage_bytes(weight->source_bytes, scale_bytes);
}

class Dsv4WeightCache {
    struct Entry {
        CudaWeight weight;
        std::uint64_t last_use{};
        std::uint64_t prefetch_lease_until{};
        // Position of this entry's key in the recency list it belongs to.
        // Front is least recently used, so eviction reads the front instead of
        // ranking every entry. A full device holds about 5,300 entries and
        // decode evicts ~127 times a step, so the scan this replaces walked
        // roughly 670,000 hash nodes a step -- measured at 14.3 ms/step, the
        // gap between moe_prepare and the demand wait it contains.
        std::list<std::string>::iterator recency{};
        std::uint32_t leases{};
        bool linked{};
        bool pinned{};
        bool prefetched{};
    };
    struct State {
        std::unordered_map<std::string, Entry> entries;
        // Ordered by last_use ascending, exactly the key the ranking scan used.
        // Prefetched entries live in their own list because the scan preferred
        // them over any demand entry regardless of recency.
        std::list<std::string> recency;
        std::list<std::string> prefetched_recency;
        std::uint64_t capacity{};
        std::uint64_t used{};
        std::uint64_t pinned{};
        std::uint64_t clock{};
    };

    struct PrefetchLinear {
        std::string name;
        std::uint64_t rows{};
        std::uint64_t columns{};
        std::uint64_t bytes{};
    };

    struct PrefetchJob {
        ExpertKey key;
        std::size_t slot{};
        std::vector<PrefetchLinear> linears;
        std::uint64_t bytes{};
    };

    enum class LoadKind : std::uint8_t {
        Preload,
        Demand,
        Prefetch,
    };

public:
    class DemandGuard {
    public:
        DemandGuard() = default;
        ~DemandGuard() { reset(); }
        DemandGuard(const DemandGuard&) = delete;
        DemandGuard& operator=(const DemandGuard&) = delete;

        DemandGuard(DemandGuard&& other) noexcept : owner_(other.owner_) {
            other.owner_ = nullptr;
        }

        DemandGuard& operator=(DemandGuard&& other) noexcept {
            if (this == &other) return *this;
            reset();
            owner_ = other.owner_;
            other.owner_ = nullptr;
            return *this;
        }

    private:
        friend class Dsv4WeightCache;

        explicit DemandGuard(Dsv4WeightCache* owner) : owner_(owner) {}

        void reset() noexcept {
            if (owner_ != nullptr) owner_->end_demand();
            owner_ = nullptr;
        }

        Dsv4WeightCache* owner_{};
    };

    class Lease {
    public:
        Lease() = default;
        ~Lease() { reset(); }
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;

        Lease(Lease&& other) noexcept
            : owner_(other.owner_), entry_(other.entry_) {
            other.owner_ = nullptr;
            other.entry_ = nullptr;
        }

        Lease& operator=(Lease&& other) noexcept {
            if (this == &other) return *this;
            reset();
            owner_ = other.owner_;
            entry_ = other.entry_;
            other.owner_ = nullptr;
            other.entry_ = nullptr;
            return *this;
        }

        [[nodiscard]] const CudaWeight& weight() const noexcept {
            return entry_->weight;
        }

    private:
        friend class Dsv4WeightCache;

        Lease(Dsv4WeightCache* owner, Entry* entry) noexcept
            : owner_(owner), entry_(entry) {}

        void reset() noexcept {
            if (owner_ != nullptr && entry_ != nullptr) owner_->release(entry_);
            owner_ = nullptr;
            entry_ = nullptr;
        }

        Dsv4WeightCache* owner_{};
        Entry* entry_{};
    };

    // A layer's routed experts are spread over every device, and each device
    // owns an independent PCIe link. Waiting out each upload where it is issued
    // drives those links one at a time: measured, the per-device demand wait
    // sums to 99.1 ms/step against a per-device maximum of 40.2 ms/step, so
    // about 59 ms of a 245 ms step is serialization rather than transfer.
    // Inside a batch the copies are left in flight and waited out once per
    // device at close() -- the same bytes over concurrent links. Priced in
    // isolation at the production byte mix and slice size: 105.4 ms serial
    // against 55.4 ms overlapped, 1.90x.
    class UploadBatch {
    public:
        UploadBatch() = default;
        // Best-effort. A batch must never leave copies in flight past the scope
        // that owns their host source; callers that need the error close()
        // explicitly.
        ~UploadBatch() { static_cast<void>(close()); }
        UploadBatch(const UploadBatch&) = delete;
        UploadBatch& operator=(const UploadBatch&) = delete;

        UploadBatch(UploadBatch&& other) noexcept
            : owner_(other.owner_), demand_(std::move(other.demand_)) {
            other.owner_ = nullptr;
        }

        UploadBatch& operator=(UploadBatch&& other) noexcept {
            if (this == &other) return *this;
            static_cast<void>(close());
            owner_ = other.owner_;
            demand_ = std::move(other.demand_);
            other.owner_ = nullptr;
            return *this;
        }

        ValidationResult close() {
            ValidationResult result;
            if (owner_ == nullptr) return result;
            auto* owner = owner_;
            owner_ = nullptr;
            result = owner->finish_upload_batch();
            // Released only after the wait: a prefetch admitted while copies
            // are in flight would issue on the same stream.
            demand_ = {};
            return result;
        }

    private:
        friend class Dsv4WeightCache;

        UploadBatch(Dsv4WeightCache* owner, DemandGuard demand)
            : owner_(owner), demand_(std::move(demand)) {}

        Dsv4WeightCache* owner_{};
        DemandGuard demand_;
    };

    Dsv4WeightCache(Dsv4CheckpointReader& checkpoint,
                    Dsv4ResidentWeightStore& resident, CudaBackend& backend,
                    std::vector<int> devices,
                    std::vector<std::uint64_t> capacities,
                    std::uint64_t prefetch_byte_budget,
                    std::size_t prefetch_queue_depth,
                    std::uint64_t prefetch_lease_ticks)
        : checkpoint_(checkpoint), resident_(resident), backend_(backend),
          devices_(std::move(devices)),
          prefetch_byte_budget_(prefetch_byte_budget),
          prefetch_queue_depth_(prefetch_queue_depth),
          prefetch_lease_ticks_(prefetch_lease_ticks) {
        for (const auto capacity : capacities) {
            states_.emplace_back();
            states_.back().capacity = capacity;
        }
        deferred_upload_slots_.assign(states_.size(), 0U);
        if (prefetch_byte_budget_ != 0U && prefetch_queue_depth_ != 0U) {
            prefetch_worker_ = std::thread([this] { prefetch_loop(); });
        }
    }

    ~Dsv4WeightCache() {
        finish_prefetch();
        {
            std::scoped_lock lock(activity_mutex_);
            stop_prefetch_ = true;
        }
        activity_changed_.notify_all();
        if (prefetch_worker_.joinable()) prefetch_worker_.join();
    }

    [[nodiscard]] DemandGuard demand(
        std::span<const ExpertKey> keys = {}) {
        if (!prefetch_worker_.joinable()) return {};
        begin_demand(keys);
        return DemandGuard(this);
    }

    // Opens a scope in which demand loads may leave their H2D copies in flight
    // so transfers to different devices overlap. Only one may be open, and only
    // on the thread that opened it: deferral is decided by a plain member flag,
    // and the prefetch worker's loads are excluded by kind rather than by it.
    [[nodiscard]] UploadBatch begin_upload_batch(
        std::span<const ExpertKey> keys = {}) {
        if (upload_batch_open_) return {};
        auto guard = demand(keys);
        upload_batch_open_ = true;
        return UploadBatch(this, std::move(guard));
    }

    ValidationResult preload(std::size_t slot, std::string_view base,
                             std::uint64_t rows, std::uint64_t columns) {
        auto demand_guard = demand();
        Entry* entry = nullptr;
        return ensure(slot, base, rows, columns, LoadKind::Preload, entry);
    }

    ValidationResult acquire(std::size_t slot, std::string_view base,
                             std::uint64_t rows, std::uint64_t columns,
                             Lease& output) {
        auto demand_guard = demand();
        output.reset();
        Entry* entry = nullptr;
        auto result = ensure(slot, base, rows, columns, LoadKind::Demand, entry);
        if (!result.ok()) return result;
        ++entry->leases;
        ++lease_acquires_;
        output = Lease(this, entry);
        return result;
    }

    // Acquires one routed expert's transformed shard straight out of the
    // resident arena. The canonical triplet would have to be re-read from the
    // checkpoint, because host memory holds only the transformed copy.
    ValidationResult acquire_tiled_expert(std::size_t slot, std::uint32_t layer,
                                          std::uint32_t expert,
                                          std::uint32_t shard, Lease& output) {
        auto demand_guard = demand();
        output.reset();
        const auto key = "layers." + std::to_string(layer) +
                         ".ffn.experts." + std::to_string(expert) +
                         ".shard." + std::to_string(shard);
        const TiledShardKey tiled{layer, expert, shard};
        Entry* entry = nullptr;
        auto result = ensure(slot, key, kHidden,
                             kExpertIntermediate / kResidentExpertShards,
                             LoadKind::Demand, entry, &tiled);
        if (!result.ok()) return result;
        ++entry->leases;
        ++lease_acquires_;
        output = Lease(this, entry);
        return result;
    }

    ValidationResult validate_atomic_expert_capacity(
        std::uint64_t required_bytes) const {
        ValidationResult result;
        for (std::size_t slot = 0U; slot < states_.size(); ++slot) {
            const auto& state = states_[slot];
            const auto available = state.capacity >= state.pinned
                                       ? state.capacity - state.pinned
                                       : 0U;
            if (required_bytes > available) {
                result.errors.emplace_back(
                    "DeepSeek device " + std::to_string(devices_[slot]) +
                    " cannot lease the worst-case exact top-k expert set");
            }
        }
        return result;
    }

    // Applies the rank-local admission result to the cache that owns the
    // centralized prefill weights. `expert_bytes` excludes the pinned spine;
    // the resulting capacity includes it. This runs only during setup, after
    // warm-up and before a session can become active.
    ValidationResult cap_expert_capacity(
        std::span<const std::uint64_t> expert_bytes) {
        ValidationResult result;
        finish_prefetch();
        if (expert_bytes.size() != states_.size()) {
            result.errors.emplace_back(
                "DeepSeek rank-local cache cap count does not match devices");
            return result;
        }
        std::vector<std::uint64_t> targets(states_.size());
        for (std::size_t slot = 0U; slot < states_.size(); ++slot) {
            auto& state = states_[slot];
            if (expert_bytes[slot] >
                std::numeric_limits<std::uint64_t>::max() - state.pinned) {
                result.errors.emplace_back(
                    "DeepSeek rank-local cache cap overflows");
                return result;
            }
            targets[slot] = state.pinned + expert_bytes[slot];
            const auto leased = std::any_of(
                state.entries.begin(), state.entries.end(),
                [](const auto& entry) { return entry.second.leases != 0U; });
            if (targets[slot] < state.pinned || leased) {
                result.errors.emplace_back(
                    "DeepSeek rank-local cache cannot be capped with pinned "
                    "or leased weights in excess of admission");
                return result;
            }
        }
        for (std::size_t slot = 0U; slot < states_.size(); ++slot) {
            auto& state = states_[slot];
            while (state.used > targets[slot]) {
                auto victim = select_victim(state);
                if (victim == state.entries.end()) {
                    result.errors.emplace_back(
                        "DeepSeek rank-local cache cap cannot evict enough "
                        "centralized prefill weights");
                    return result;
                }
                state.used -= victim->second.weight.device_bytes();
                discard_prefetched(state, victim->second, true);
                unlink_recency(state, victim->second);
                state.entries.erase(victim);
                ++evictions_;
            }
            state.capacity = targets[slot];
        }
        return result;
    }

    ValidationResult matmul(std::size_t slot, std::string_view base,
                            std::uint64_t output_columns,
                            std::uint64_t input_columns,
                            std::span<const float> input, std::uint32_t rows,
                            std::span<float> output, bool bf16_output = true,
                            CudaMatmulProfile* profile = nullptr,
                            bool dsv4_fp8_tensor_page = false) {
        const auto acquisition_started = std::chrono::steady_clock::now();
        auto demand_guard = demand();
        Entry* entry = nullptr;
        auto result = ensure(slot, base, output_columns, input_columns,
                             LoadKind::Demand, entry);
        if (!result.ok()) return result;
        const auto acquisition_nanoseconds =
            elapsed_nanoseconds(acquisition_started);
        result = backend_.matmul(entry->weight, input, rows, output,
                                 bf16_output, profile,
                                 dsv4_fp8_tensor_page);
        if (profile != nullptr) {
            profile->weight_acquisition_nanoseconds =
                acquisition_nanoseconds;
        }
        return result;
    }

    ValidationResult grouped(std::size_t slot, std::string_view base,
                             std::uint64_t output_columns,
                             std::uint64_t input_columns,
                             std::span<const float> input,
                             std::uint32_t groups,
                             std::uint64_t rows_per_group,
                             std::span<float> output) {
        return grouped_rows(slot, base, output_columns, input_columns,
                            input, 1U, groups, rows_per_group, output);
    }

    ValidationResult grouped_rows(std::size_t slot, std::string_view base,
                                  std::uint64_t output_columns,
                                  std::uint64_t input_columns,
                                  std::span<const float> input,
                                  std::uint32_t rows,
                                  std::uint32_t groups,
                                  std::uint64_t rows_per_group,
                                  std::span<float> output) {
        auto demand_guard = demand();
        Entry* entry = nullptr;
        auto result = ensure(slot, base, output_columns, input_columns,
                             LoadKind::Demand, entry);
        if (!result.ok()) return result;
        result = backend_.matmul_grouped_rows(
            entry->weight, input, rows, groups, rows_per_group, output);
        if (result.ok()) round_bf16(output);
        return result;
    }

    [[nodiscard]] Dsv4CacheStats stats() const {
        Dsv4CacheStats result;
        result.hits = hits_.load(std::memory_order_relaxed);
        result.misses = misses_.load(std::memory_order_relaxed);
        result.evictions = evictions_.load(std::memory_order_relaxed);
        result.lease_acquires = lease_acquires_.load(std::memory_order_relaxed);
        result.lease_releases = lease_releases_.load(std::memory_order_relaxed);
        result.demand_h2d_bytes = demand_h2d_bytes_.load(std::memory_order_relaxed);
        result.demand_wait_nanoseconds =
            demand_wait_nanoseconds_.load(std::memory_order_relaxed);
        result.prefetch_requests = prefetch_requests_.load(std::memory_order_relaxed);
        result.prefetch_h2d_bytes = prefetch_h2d_bytes_.load(std::memory_order_relaxed);
        result.useful_prefetch_bytes =
            useful_prefetch_bytes_.load(std::memory_order_relaxed);
        result.late_prefetch_bytes =
            late_prefetch_bytes_.load(std::memory_order_relaxed);
        result.duplicate_prefetch_bytes =
            duplicate_prefetch_bytes_.load(std::memory_order_relaxed);
        result.evicted_prefetch_bytes =
            evicted_prefetch_bytes_.load(std::memory_order_relaxed);
        result.wasted_prefetch_bytes =
            wasted_prefetch_bytes_.load(std::memory_order_relaxed);
        result.cancelled_prefetch_bytes =
            cancelled_prefetch_bytes_.load(std::memory_order_relaxed);
        result.prefetch_lease_acquires =
            prefetch_lease_acquires_.load(std::memory_order_relaxed);
        result.prefetch_lease_releases =
            prefetch_lease_releases_.load(std::memory_order_relaxed);
        result.active_prefetch_leases =
            active_prefetch_leases_.load(std::memory_order_relaxed);
        result.prefetch_queue_peak =
            prefetch_queue_peak_.load(std::memory_order_relaxed);
        for (const auto& state : states_) {
            result.used_bytes.push_back(state.used);
            result.capacity_bytes.push_back(state.capacity);
            result.pinned_bytes.push_back(state.pinned);
            std::uint64_t leased_bytes = 0U;
            std::uint64_t active_leases = 0U;
            for (const auto& [name, entry] : state.entries) {
                static_cast<void>(name);
                if (entry.leases == 0U) continue;
                leased_bytes += entry.weight.device_bytes();
                active_leases += entry.leases;
            }
            result.leased_bytes.push_back(leased_bytes);
            result.active_leases.push_back(active_leases);
        }
        return result;
    }

    void request_prefetch(ExpertKey key, std::size_t slot) {
        ++prefetch_requests_;
        if (!prefetch_worker_.joinable() || slot >= states_.size()) return;

        const auto prefix = layer_prefix(key.layer) + "ffn.experts." +
                            std::to_string(key.expert) + ".";
        const std::array dimensions{
            std::pair{std::uint64_t{kExpertIntermediate}, std::uint64_t{kHidden}},
            std::pair{std::uint64_t{kExpertIntermediate}, std::uint64_t{kHidden}},
            std::pair{std::uint64_t{kHidden}, std::uint64_t{kExpertIntermediate}},
        };
        constexpr std::array<std::string_view, 3U> operations{"w1", "w3", "w2"};

        PrefetchJob job;
        job.key = key;
        job.slot = slot;
        std::uint64_t expert_bytes = 0U;
        for (std::size_t index = 0U; index < operations.size(); ++index) {
            PrefetchLinear linear;
            linear.name = prefix + std::string(operations[index]);
            linear.rows = dimensions[index].first;
            linear.columns = dimensions[index].second;
            linear.bytes = linear_bytes(checkpoint_, linear.name);
            if (linear.bytes == 0U ||
                expert_bytes > std::numeric_limits<std::uint64_t>::max() -
                                   linear.bytes) {
                cancelled_prefetch_bytes_ += expert_bytes;
                return;
            }
            expert_bytes += linear.bytes;
            job.linears.push_back(std::move(linear));
        }

        std::scoped_lock lock(activity_mutex_);
        if (pending_prefetch_.contains(key)) {
            duplicate_prefetch_bytes_ += expert_bytes;
            return;
        }
        auto& state = states_[slot];
        for (auto current = job.linears.begin(); current != job.linears.end();) {
            const auto found = state.entries.find(current->name);
            if (found == state.entries.end()) {
                job.bytes += current->bytes;
                ++current;
            } else {
                duplicate_prefetch_bytes_ += current->bytes;
                current = job.linears.erase(current);
            }
        }
        if (job.linears.empty()) return;
        if (job.bytes > prefetch_byte_budget_ ||
            prefetch_queue_.size() >= prefetch_queue_depth_) {
            cancelled_prefetch_bytes_ += job.bytes;
            return;
        }
        pending_prefetch_.insert(key);
        prefetch_queue_.push_back(std::move(job));
        const auto depth = static_cast<std::uint64_t>(prefetch_queue_.size());
        auto peak = prefetch_queue_peak_.load(std::memory_order_relaxed);
        while (peak < depth && !prefetch_queue_peak_.compare_exchange_weak(
                   peak, depth, std::memory_order_relaxed)) {}
        activity_changed_.notify_all();
    }

    void drain_prefetch() {
        if (!prefetch_worker_.joinable()) return;
        std::unique_lock lock(activity_mutex_);
        activity_changed_.wait(lock, [this] {
            return prefetch_queue_.empty() && !prefetch_active_;
        });
    }

    void finish_prefetch() {
        auto demand_guard = demand();
        {
            std::scoped_lock lock(activity_mutex_);
            for (const auto& job : prefetch_queue_) {
                cancelled_prefetch_bytes_ += job.bytes;
                pending_prefetch_.erase(job.key);
            }
            prefetch_queue_.clear();
        }
        for (auto& state : states_) {
            for (auto entry = state.entries.begin(); entry != state.entries.end();) {
                if (!entry->second.prefetched) {
                    ++entry;
                    continue;
                }
                state.used -= entry->second.weight.device_bytes();
                discard_prefetched(state, entry->second, false);
                unlink_recency(state, entry->second);
                entry = state.entries.erase(entry);
            }
        }
    }

private:
    // Recency bookkeeping. The list an entry sits in is decided by
    // `prefetched`, and its position inside that list by `last_use`; both
    // mirror the keys the ranking scan used, so the victim these pick is the
    // victim the scan picked.
    static std::list<std::string>& recency_list(State& state,
                                                bool prefetched) noexcept {
        return prefetched ? state.prefetched_recency : state.recency;
    }

    // Pinned entries are never candidates, so they are kept out of the lists
    // entirely rather than stepped over on every walk. The resident spine is
    // 500-900 entries a device.
    static void link_recency(State& state, const std::string& key,
                             Entry& entry) {
        if (entry.pinned) return;
        auto& list = recency_list(state, entry.prefetched);
        list.push_back(key);
        entry.recency = std::prev(list.end());
        entry.linked = true;
    }

    static void touch_recency(State& state, Entry& entry) noexcept {
        if (!entry.linked) return;
        auto& list = recency_list(state, entry.prefetched);
        list.splice(list.end(), list, entry.recency);
    }

    // Moves an entry between the two lists without disturbing its position
    // relative to the entries already there, which is what a prefetched entry
    // promoted by a demand hit needs: the hit has already bumped `last_use`.
    static void relink_recency(State& state, Entry& entry, bool was_prefetched) {
        if (!entry.linked) return;
        auto& source = recency_list(state, was_prefetched);
        auto& destination = recency_list(state, entry.prefetched);
        if (&source == &destination) return;
        destination.splice(destination.end(), source, entry.recency);
    }

    static void unlink_recency(State& state, Entry& entry) noexcept {
        if (!entry.linked) return;
        recency_list(state, entry.prefetched).erase(entry.recency);
        entry.linked = false;
    }

    // Least recently used first, prefetched entries ahead of demand ones.
    // Pinned and leased entries stay listed and are stepped over: both are
    // touched on the step that holds them, so they sit at the back and the
    // walk does not reach them.
    [[nodiscard]] std::unordered_map<std::string, Entry>::iterator select_victim(
        State& state) {
        for (auto* list : {&state.prefetched_recency, &state.recency}) {
            for (const auto& key : *list) {
                auto candidate = state.entries.find(key);
                if (candidate == state.entries.end()) continue;
                if (candidate->second.pinned || candidate->second.leases != 0U) {
                    continue;
                }
                return candidate;
            }
        }
        return state.entries.end();
    }

    // One routed expert's transformed TP shard, addressed the way the resident
    // arena holds it. This is the only copy of the routed experts on the host,
    // so prefill uploads it as it stands rather than reconstructing a second
    // layout that would not fit beside it.
    struct TiledShardKey {
        std::uint32_t layer{};
        std::uint32_t expert{};
        std::uint32_t shard{};
    };

    ValidationResult ensure(std::size_t slot, std::string_view base,
                            std::uint64_t rows, std::uint64_t columns,
                            LoadKind kind, Entry*& output,
                            const TiledShardKey* tiled = nullptr) {
        ValidationResult result;
        if (slot >= states_.size()) {
            result.errors.emplace_back("DeepSeek linear targets an invalid CUDA slot");
            return result;
        }
        auto& state = states_[slot];
        ++state.clock;
        const bool pin = kind == LoadKind::Preload;
        const std::string key(base);
        auto found = state.entries.find(key);
        if (found != state.entries.end()) {
            found->second.last_use = state.clock;
            touch_recency(state, found->second);
            if (pin && !found->second.pinned) {
                found->second.pinned = true;
                state.pinned += found->second.weight.device_bytes();
                unlink_recency(state, found->second);
            }
            if (kind == LoadKind::Demand) {
                ++hits_;
                mark_prefetch_useful(state, found->second);
            } else if (kind == LoadKind::Preload) {
                ++hits_;
            } else {
                duplicate_prefetch_bytes_ += found->second.weight.device_bytes();
            }
            output = &found->second;
            return result;
        }
        const auto shard_payload_bytes = tiled == nullptr
            ? 0U
            : dsv4_tiled_expert_shard_bytes(kHidden, kExpertIntermediate,
                                            kResidentExpertShards);
        const auto bytes = tiled == nullptr
            ? linear_bytes(checkpoint_, base)
            : CudaBackend::weight_storage_bytes(shard_payload_bytes, 0U);
        if (bytes == 0U || bytes > state.capacity) {
            result.errors.emplace_back("DeepSeek linear is absent or exceeds device cache: " +
                                       key);
            return result;
        }
        while (kind != LoadKind::Prefetch && state.used + bytes > state.capacity) {
            auto victim = select_victim(state);
            if (victim == state.entries.end()) {
                const bool in_flight = std::any_of(
                    state.entries.begin(), state.entries.end(),
                    [](const auto& candidate) {
                        return candidate.second.leases != 0U;
                    });
                result.errors.emplace_back(in_flight
                    ? "DeepSeek atomic in-flight expert set exceeds a device VRAM budget"
                    : "DeepSeek pinned resident spine exceeds a device VRAM budget");
                return result;
            }
            state.used -= victim->second.weight.device_bytes();
            discard_prefetched(state, victim->second, true);
            unlink_recency(state, victim->second);
            state.entries.erase(victim);
            ++evictions_;
        }
        if (state.used + bytes > state.capacity) {
            result.errors.emplace_back("DeepSeek prefetch cannot displace demand weights");
            return result;
        }
        Entry entry;
        // Only demand loads inside an open batch may defer. Prefetch runs on
        // its own thread and must not leave copies in flight that no one on
        // this thread will wait for; preload reads the checkpoint into a
        // temporary, which the loader refuses to defer anyway.
        const bool defer = upload_batch_open_ && kind == LoadKind::Demand;
        const auto load_started = std::chrono::steady_clock::now();
        if (tiled != nullptr) {
            const auto storage = resident_.find_tiled_expert(
                tiled->layer, tiled->expert, tiled->shard);
            if (storage.size() != shard_payload_bytes) {
                result.errors.emplace_back(
                    "DeepSeek transformed expert shard is not resident: " + key);
                return result;
            }
            CudaWeightDescriptor descriptor;
            descriptor.encoding = CudaWeightEncoding::Fp4E2m1Tiled32;
            descriptor.dtype = SafetensorsDtype::I8;
            descriptor.rows = kHidden;
            descriptor.columns =
                kExpertIntermediate / kResidentExpertShards;
            descriptor.group_size = 32U;
            result = backend_.upload(
                devices_[slot], descriptor, storage, {}, entry.weight,
                defer ? CudaBackend::UploadCompletion::Deferred
                      : CudaBackend::UploadCompletion::Synchronous);
        } else {
            result = load_dsv4_cuda_linear(
                checkpoint_, kind == LoadKind::Preload ? nullptr : &resident_,
                base, rows, columns, devices_[slot], backend_, entry.weight,
                defer);
        }
        if (!result.ok()) return result;
        if (defer) deferred_upload_slots_[slot] = 1U;
        entry.last_use = state.clock;
        entry.pinned = pin;
        entry.prefetched = kind == LoadKind::Prefetch;
        entry.prefetch_lease_until = entry.prefetched
            ? state.clock + std::min(
                prefetch_lease_ticks_,
                std::numeric_limits<std::uint64_t>::max() - state.clock)
            : 0U;
        state.used += entry.weight.device_bytes();
        if (pin) state.pinned += entry.weight.device_bytes();
        found = state.entries.emplace(key, std::move(entry)).first;
        link_recency(state, found->first, found->second);
        if (kind == LoadKind::Prefetch) {
            const auto loaded_bytes = found->second.weight.device_bytes();
            prefetched_bytes_ += loaded_bytes;
            prefetch_h2d_bytes_ += loaded_bytes;
            ++prefetch_lease_acquires_;
            ++active_prefetch_leases_;
        } else {
            ++misses_;
            if (kind == LoadKind::Demand) {
                demand_h2d_bytes_ += found->second.weight.device_bytes();
                demand_wait_nanoseconds_ += elapsed_nanoseconds(load_started);
            }
        }
        output = &found->second;
        return result;
    }

    // Waits out every device the batch deferred to. The wait the individual
    // loads no longer pay is charged here instead, so demand_wait_seconds
    // stays the cost of getting the weights onto the devices and the phase
    // timers keep summing to the step.
    ValidationResult finish_upload_batch() {
        ValidationResult result;
        upload_batch_open_ = false;
        const auto wait_started = std::chrono::steady_clock::now();
        bool waited = false;
        for (std::size_t slot = 0U; slot < deferred_upload_slots_.size(); ++slot) {
            if (deferred_upload_slots_[slot] == 0U) continue;
            deferred_upload_slots_[slot] = 0U;
            waited = true;
            auto synchronized = backend_.synchronize_uploads(devices_[slot]);
            if (!synchronized.ok()) {
                append_errors(result, std::move(synchronized.errors));
            }
        }
        if (waited) demand_wait_nanoseconds_ += elapsed_nanoseconds(wait_started);
        return result;
    }

    void mark_prefetch_useful(State& state, Entry& entry) {
        if (!entry.prefetched) return;
        const auto bytes = entry.weight.device_bytes();
        useful_prefetch_bytes_ += bytes;
        prefetched_bytes_ -= bytes;
        entry.prefetched = false;
        entry.prefetch_lease_until = 0U;
        relink_recency(state, entry, true);
        ++prefetch_lease_releases_;
        --active_prefetch_leases_;
    }

    void discard_prefetched(State& state, Entry& entry, bool evicted) {
        if (!entry.prefetched) return;
        const auto bytes = entry.weight.device_bytes();
        prefetched_bytes_ -= bytes;
        wasted_prefetch_bytes_ += bytes;
        if (evicted) evicted_prefetch_bytes_ += bytes;
        entry.prefetched = false;
        relink_recency(state, entry, true);
        ++prefetch_lease_releases_;
        --active_prefetch_leases_;
    }

    [[nodiscard]] bool evict_prefetched(std::optional<std::size_t> only_slot) {
        std::size_t selected_slot = states_.size();
        auto selected = states_.front().entries.end();
        for (std::size_t slot = 0U; slot < states_.size(); ++slot) {
            if (only_slot && slot != *only_slot) continue;
            auto& state = states_[slot];
            for (auto candidate = state.entries.begin();
                 candidate != state.entries.end(); ++candidate) {
                const auto& entry = candidate->second;
                if (!entry.prefetched || entry.leases != 0U ||
                    entry.prefetch_lease_until > state.clock) {
                    continue;
                }
                if (selected_slot == states_.size() ||
                    entry.last_use < selected->second.last_use) {
                    selected_slot = slot;
                    selected = candidate;
                }
            }
        }
        if (selected_slot == states_.size()) return false;
        auto& state = states_[selected_slot];
        state.used -= selected->second.weight.device_bytes();
        discard_prefetched(state, selected->second, true);
        unlink_recency(state, selected->second);
        state.entries.erase(selected);
        ++evictions_;
        return true;
    }

    [[nodiscard]] bool prepare_prefetch(const PrefetchJob& job) {
        auto& state = states_[job.slot];
        if (job.bytes > state.capacity || job.bytes > prefetch_byte_budget_) {
            return false;
        }
        while (prefetched_bytes_ + job.bytes > prefetch_byte_budget_) {
            if (!evict_prefetched(std::nullopt)) return false;
        }
        while (state.used + job.bytes > state.capacity) {
            if (!evict_prefetched(job.slot)) return false;
        }
        return true;
    }

    void run_prefetch(const PrefetchJob& job) {
        if (!prepare_prefetch(job)) {
            cancelled_prefetch_bytes_ += job.bytes;
            return;
        }
        std::vector<std::string_view> loaded;
        loaded.reserve(job.linears.size());
        for (const auto& linear : job.linears) {
            Entry* entry = nullptr;
            auto result = ensure(job.slot, linear.name, linear.rows,
                                 linear.columns, LoadKind::Prefetch, entry);
            if (result.ok()) {
                loaded.push_back(linear.name);
                continue;
            }
            cancelled_prefetch_bytes_ += job.bytes;
            auto& state = states_[job.slot];
            for (const auto name : loaded) {
                const auto found = state.entries.find(std::string(name));
                if (found == state.entries.end() || !found->second.prefetched) continue;
                state.used -= found->second.weight.device_bytes();
                discard_prefetched(state, found->second, false);
                unlink_recency(state, found->second);
                state.entries.erase(found);
            }
            return;
        }
    }

    [[nodiscard]] static bool contains_key(
        std::span<const ExpertKey> keys, ExpertKey key) noexcept {
        return std::find(keys.begin(), keys.end(), key) != keys.end();
    }

    void begin_demand(std::span<const ExpertKey> keys) {
        const auto started = std::chrono::steady_clock::now();
        std::unique_lock lock(activity_mutex_);
        ++waiting_demands_;
        for (auto job = prefetch_queue_.begin(); job != prefetch_queue_.end();) {
            if (!contains_key(keys, job->key)) {
                ++job;
                continue;
            }
            late_prefetch_bytes_ += job->bytes;
            cancelled_prefetch_bytes_ += job->bytes;
            pending_prefetch_.erase(job->key);
            job = prefetch_queue_.erase(job);
        }
        if (active_prefetch_key_ && contains_key(keys, *active_prefetch_key_) &&
            !active_prefetch_late_) {
            late_prefetch_bytes_ += active_prefetch_bytes_;
            active_prefetch_late_ = true;
        }
        const bool waited = prefetch_active_;
        activity_changed_.wait(lock, [this] { return !prefetch_active_; });
        --waiting_demands_;
        ++active_demands_;
        lock.unlock();
        if (waited) demand_wait_nanoseconds_ += elapsed_nanoseconds(started);
    }

    void end_demand() noexcept {
        {
            std::scoped_lock lock(activity_mutex_);
            if (active_demands_ != 0U) --active_demands_;
        }
        activity_changed_.notify_all();
    }

    void prefetch_loop() {
        for (;;) {
            PrefetchJob job;
            {
                std::unique_lock lock(activity_mutex_);
                activity_changed_.wait(lock, [this] {
                    return stop_prefetch_ ||
                           (!prefetch_queue_.empty() && active_demands_ == 0U &&
                            waiting_demands_ == 0U);
                });
                if (stop_prefetch_) return;
                job = std::move(prefetch_queue_.front());
                prefetch_queue_.pop_front();
                active_prefetch_key_ = job.key;
                active_prefetch_bytes_ = job.bytes;
                active_prefetch_late_ = false;
                prefetch_active_ = true;
            }
            try {
                run_prefetch(job);
            } catch (...) {
                // Prediction is advisory; background allocation failure must
                // never prevent the exact demand path from running.
                cancelled_prefetch_bytes_ += job.bytes;
            }
            {
                std::scoped_lock lock(activity_mutex_);
                pending_prefetch_.erase(job.key);
                active_prefetch_key_.reset();
                active_prefetch_bytes_ = 0U;
                active_prefetch_late_ = false;
                prefetch_active_ = false;
            }
            activity_changed_.notify_all();
        }
    }

    void release(Entry* entry) noexcept {
        if (entry != nullptr && entry->leases != 0U) {
            --entry->leases;
            ++lease_releases_;
        }
    }

    Dsv4CheckpointReader& checkpoint_;
    Dsv4ResidentWeightStore& resident_;
    CudaBackend& backend_;
    std::vector<int> devices_;
    std::vector<State> states_;
    std::uint64_t prefetch_byte_budget_{};
    std::size_t prefetch_queue_depth_{};
    std::uint64_t prefetch_lease_ticks_{};
    std::uint64_t prefetched_bytes_{};
    std::mutex activity_mutex_;
    std::condition_variable activity_changed_;
    std::deque<PrefetchJob> prefetch_queue_;
    std::unordered_set<ExpertKey, ExpertKeyHash> pending_prefetch_;
    std::optional<ExpertKey> active_prefetch_key_;
    std::uint64_t active_prefetch_bytes_{};
    std::thread prefetch_worker_;
    std::size_t active_demands_{};
    std::size_t waiting_demands_{};
    // Touched only by the thread that opened the batch. std::uint8_t rather
    // than bool because this is a per-slot flag array, not a bitset.
    std::vector<std::uint8_t> deferred_upload_slots_;
    bool upload_batch_open_{};
    bool prefetch_active_{};
    bool active_prefetch_late_{};
    bool stop_prefetch_{};
    std::atomic<std::uint64_t> hits_{};
    std::atomic<std::uint64_t> misses_{};
    std::atomic<std::uint64_t> evictions_{};
    std::atomic<std::uint64_t> lease_acquires_{};
    std::atomic<std::uint64_t> lease_releases_{};
    std::atomic<std::uint64_t> demand_h2d_bytes_{};
    std::atomic<std::uint64_t> demand_wait_nanoseconds_{};
    std::atomic<std::uint64_t> prefetch_requests_{};
    std::atomic<std::uint64_t> prefetch_h2d_bytes_{};
    std::atomic<std::uint64_t> useful_prefetch_bytes_{};
    std::atomic<std::uint64_t> late_prefetch_bytes_{};
    std::atomic<std::uint64_t> duplicate_prefetch_bytes_{};
    std::atomic<std::uint64_t> evicted_prefetch_bytes_{};
    std::atomic<std::uint64_t> wasted_prefetch_bytes_{};
    std::atomic<std::uint64_t> cancelled_prefetch_bytes_{};
    std::atomic<std::uint64_t> prefetch_lease_acquires_{};
    std::atomic<std::uint64_t> prefetch_lease_releases_{};
    std::atomic<std::uint64_t> active_prefetch_leases_{};
    std::atomic<std::uint64_t> prefetch_queue_peak_{};
};

class PagedFloatRows {
public:
    [[nodiscard]] bool configure(std::size_t rows, std::size_t columns) {
        rows_ = rows;
        columns_ = columns;
        constexpr std::size_t target_page_bytes = 256U << 10U;
        page_rows_ = std::max<std::size_t>(
            1U, target_page_bytes / std::max<std::size_t>(sizeof(float),
                                                          columns_ * sizeof(float)));
        try {
            pages_.clear();
            pages_.resize((rows_ + page_rows_ - 1U) / page_rows_);
        } catch (const std::bad_alloc&) {
            pages_.clear();
            rows_ = 0U;
            columns_ = 0U;
            page_rows_ = 0U;
            return false;
        }
        return true;
    }

    [[nodiscard]] std::span<float> writable_row(std::size_t row) {
        if (row >= rows_ || columns_ == 0U || page_rows_ == 0U) return {};
        const auto page = row / page_rows_;
        if (pages_[page] == nullptr) {
            const auto rows_in_page = std::min(
                page_rows_, rows_ - page * page_rows_);
            pages_[page].reset(
                new (std::nothrow) float[rows_in_page * columns_]);
            if (pages_[page] == nullptr) return {};
        }
        return {pages_[page].get() + (row % page_rows_) * columns_, columns_};
    }

    [[nodiscard]] std::span<const float> row(std::size_t row) const noexcept {
        if (row >= rows_ || columns_ == 0U || page_rows_ == 0U) return {};
        const auto page = row / page_rows_;
        if (pages_[page] == nullptr) return {};
        return {pages_[page].get() + (row % page_rows_) * columns_, columns_};
    }

private:
    std::size_t rows_{};
    std::size_t columns_{};
    std::size_t page_rows_{};
    std::vector<std::unique_ptr<float[]>> pages_;
};

struct CompressorState {
    std::uint32_t ratio{};
    std::uint32_t coefficient{};
    std::uint32_t head_dim{};
    bool rotate_fp4{};
    Dsv4KvBlockKind kind{Dsv4KvBlockKind::Csa};
    std::vector<float> values;
    std::vector<float> scores;
    PagedFloatRows compressed;
};

struct AttentionState {
    std::vector<float> sliding;
    CompressorState compressor;
    CompressorState indexer_compressor;
    std::vector<float> frequencies;
};

// Everything a one-token forward pass mutates and a rollback must put back.
// The compressor accumulators are fixed-size scratch that a pass overwrites in
// place, so they are copied whole; the sliding and compressed caches are
// append- or ring-addressed, so only the single row a pass can reach is kept.
struct CompressorSnapshot {
    std::vector<float> values;
    std::vector<float> scores;
    std::vector<float> compressed_row;
    std::size_t compressed_index{};
    bool compressed_saved{};
};

struct LayerSnapshot {
    CompressorSnapshot compressor;
    CompressorSnapshot indexer;
    std::vector<float> sliding_row;
    std::size_t sliding_index{};
    bool sliding_saved{};
};

struct SpeculativeState {
    std::vector<LayerSnapshot> layers;
    std::uint64_t tokens{};
};

// Finds the block owning `logical_row` in a per-(sequence, kind, layer)
// physical block table.
//
// This replaces a linear `std::find_if` that ran once per attended candidate.
// At the declared 1,048,576-token context a compressed stream's table holds
// 4,096 blocks and a ratio-128 layer attends 8,192 candidates, so the scan
// measured 2,889.637 ms per decoded token across the 43 layers -- about
// twenty-two times the entire decode budget.
//
// Blocks partition the row space and are appended in increasing logical order,
// and every block holds `capacity_rows` rows with only the last possibly
// short, so the owning block is normally computable directly from the first
// block's geometry. Ordered and exhaustive fallbacks follow, so a table that
// has been reordered by eviction or reuse still resolves correctly.
//
// Every path validates the same predicate before returning, and blocks do not
// overlap within one table, so whichever path succeeds returns exactly the
// block the original scan would have found.
[[nodiscard]] std::size_t locate_physical_kv_block(
    const std::vector<Dsv4KvBlockInfo>& table, std::uint64_t logical_row) {
    const auto owns = [&](std::size_t index) {
        const auto& block = table[index];
        if (block.compression_ratio == 0U) return false;
        const auto begin = block.logical_begin / block.compression_ratio;
        return logical_row >= begin &&
               logical_row < begin + block.used_rows;
    };
    if (table.empty()) return table.size();

    const auto& first = table.front();
    if (first.compression_ratio != 0U && first.capacity_rows != 0U) {
        const auto base = first.logical_begin / first.compression_ratio;
        if (logical_row >= base) {
            const auto guess = static_cast<std::size_t>(
                (logical_row - base) / first.capacity_rows);
            if (guess < table.size() && owns(guess)) return guess;
        }
    }

    std::size_t low = 0U;
    std::size_t high = table.size();
    while (low < high) {
        const auto middle = low + (high - low) / 2U;
        const auto& block = table[middle];
        if (block.compression_ratio == 0U) break;
        const auto begin = block.logical_begin / block.compression_ratio;
        if (logical_row < begin) {
            high = middle;
        } else if (logical_row >= begin + block.used_rows) {
            low = middle + 1U;
        } else {
            return middle;
        }
    }

    for (std::size_t index = 0U; index < table.size(); ++index) {
        if (owns(index)) return index;
    }
    return table.size();
}

}  // namespace

struct DeepSeekV4Runtime::Impl {
    // Page projections overwrite every output element before returning. Keep
    // their largest admitted host extents for the runtime lifetime and avoid
    // std::vector(size), whose value-initialization zeroed hundreds of MiB per
    // layer before CUDA immediately overwrote it.
    struct UninitializedFloatScratch {
        std::span<float> acquire(std::size_t elements) {
            if (elements > capacity) {
                data = std::make_unique_for_overwrite<float[]>(elements);
                capacity = elements;
            }
            return {data.get(), elements};
        }

        std::unique_ptr<float[]> data;
        std::size_t capacity{};
    };

    // Physical page metadata is shared by every row in one prefill page.
    // Device leases are moved into pending_attention_leases after each
    // successful command, so the buffers remain resident until the matching
    // MoE collect while this context keeps only the stable page numbering.
    struct PhysicalAttentionPageSet {
        std::vector<Dsv4KvDeviceLease> leases;
        std::vector<CudaDsv4PhysicalPage> pages;
        std::unordered_map<std::uint64_t, std::uint32_t> page_indices;
    };

    struct PhysicalAttentionContext {
        Impl* owner{};
        std::uint32_t layer{};
        std::uint32_t position{};
        ValidationResult result;
        bool invoked{};
        std::vector<float> query_rank;
        std::vector<float> key_value;
        std::vector<float> compressor_values;
        std::vector<float> compressor_scores;
        std::vector<float> index_compressor_values;
        std::vector<float> index_compressor_scores;
        std::optional<Dsv4KvPhysicalAppend> sliding_append;
        std::optional<Dsv4KvPhysicalAppend> compressed_append;
        std::optional<Dsv4KvPhysicalAppend> index_append;
    };

    struct HostMoeContext {
        Impl* owner{};
        std::uint32_t layer{};
        std::uint32_t token{};
        std::uint32_t position{};
        ValidationResult result;
        bool invoked{};
        bool accepted{};
        std::chrono::steady_clock::time_point execution_started{};
        std::chrono::steady_clock::time_point callback_finished{};
        std::vector<float> input;
        Dsv4Route route;
        std::array<std::array<Dsv4TiledExpertWeights, kTopK>, 2U> tiled{};
        std::array<Dsv4WeightCache::Lease, 3U> shared_leases;
        CudaDeepSeekMoeExpert shared;
    };

    struct DeviceHeadContext {
        Impl* owner{};
        ValidationResult result;
        bool invoked{};
        std::array<float, static_cast<std::size_t>(kMhc) * kHidden> hidden{};
    };

    Dsv4RuntimeConfig config;
    Dsv4MemoryPlan memory;
    Dsv4GenerationMetrics initialization_metrics;
    std::unique_ptr<Dsv4CheckpointReader> checkpoint;
    Dsv4ResidentWeightStore resident;
    ModelTokenizer tokenizer;
    CudaBackend cuda;
    std::unique_ptr<Dsv4WeightCache> weights;
    std::unique_ptr<Dsv4KvCache> kv_cache;
    // Rank-local decode session. Present only under the explicit opt-in, and
    // only once admission has passed against measured byte accounts; the
    // centralized path never consults it.
    std::unique_ptr<Dsv4RankLocalWeightStore> rank_local_weights;
#if defined(STRATA_HAS_NCCL)
    std::unique_ptr<Dsv4RankLocalLayerExecutor> rank_local_executor;
    // Outlives the executor that points at it, so it is declared before the
    // executor is torn down and destroyed after.
    std::vector<std::unique_ptr<Dsv4StaticExpertTier>> static_expert_tiers;
#endif
    Dsv4RankLocalAdmission rank_local_admission;
    std::vector<std::uint64_t> rank_local_initial_device_vram_bytes;
    std::vector<std::uint64_t> rank_local_actual_device_vram_bytes;
    bool rank_local_active{};
    // One rank-local layer's borrowed views. Queued mode submits all 43 calls
    // before finish_chain, so every layer owns a distinct stable record.
    struct RankLocalLayerScratch {
        std::vector<Dsv4KvDeviceLease> leases;
        std::array<std::vector<CudaDsv4PhysicalPage>, kDsv4RankLocalWorld> pages;
        std::vector<CudaDsv4AttentionCandidate> candidates;
        std::array<float, kRopeDim / 2U> cosines{};
        std::array<float, kRopeDim / 2U> inverse_sines{};
        std::vector<float> query_rank;
        std::vector<float> key_value;
        std::vector<float> compressor_values;
        std::vector<float> compressor_scores;
        std::vector<float> index_compressor_values;
        std::vector<float> index_compressor_scores;
        std::vector<float> compressed_row;
        std::vector<float> index_row;
        std::vector<std::uint32_t> indexed_positions;
        std::array<std::vector<std::byte>, kDsv4RankLocalWorld> patches;
        std::span<std::byte> replica_patch;
        std::array<std::vector<CudaDsv4AttentionPageWrite>,
                   kDsv4RankLocalWorld> page_writes;
        // Which compressed blocks already hold a lease this token. Positional
        // page numbering fixes a block's index up front, but the lease is still
        // taken on first touch, so an unselected block is never leased: 512 of
        // 4,096 blocks at the declared context rather than all of them.
        //
        // First touch is only available while the host knows the selection.
        // An in-chain layer leases every attendable block instead, because the
        // page a device-selected row will name is not knowable here.
        std::vector<std::uint8_t> compressed_block_leased;
        // In-chain selection state for an indexed layer. The leases are held
        // separately from the attention page leases only so their lifetimes
        // read clearly; both are released together at the next token.
        std::vector<Dsv4KvDeviceLease> index_leases;
        std::array<std::vector<CudaDsv4PhysicalIndexPage>, kDsv4RankLocalWorld>
            index_pages;
        std::vector<CudaDsv4KvBlockDescriptor> blocks;
        std::array<float, kRopeDim / 2U> index_cosines{};
        std::array<float, kRopeDim / 2U> index_sines{};
        std::array<Dsv4WeightCache::Lease, kDsv4RankLocalWorld>
            index_query_projection;
        std::array<Dsv4WeightCache::Lease, kDsv4RankLocalWorld>
            index_weight_projection;
    };
    struct RankLocalPageContext {
        Impl* owner{};
        Dsv4RankLocalKvTransaction* transaction{};
        RankLocalLayerScratch* scratch{};
        std::uint32_t layer{};
        std::uint32_t position{};
        std::size_t rank{};
        ValidationResult result;
        bool invoked{};
        std::uint64_t elapsed_nanoseconds{};
    };
    std::array<RankLocalLayerScratch, kLayers> rank_local_scratch{};
    std::array<Dsv4RankLocalLayerCall, kLayers> rank_local_calls{};
    std::array<std::array<RankLocalPageContext, kDsv4RankLocalWorld>, kLayers>
        rank_local_page_contexts{};
    std::array<DeviceHeadContext, kDsv4RankLocalWorld> rank_local_head{};
    std::array<std::vector<float>, kDsv4RankLocalWorld> rank_local_local_logits;
    std::array<std::vector<std::uint16_t>, kDsv4RankLocalWorld>
        rank_local_published_logits;
    // The attention input of the layer about to run, carried out of the
    // previous layer's reduction. Layers below three route by table and never
    // reach the indexer, which is the only consumer.
    std::vector<float> rank_local_attention_input;
    Dsv4SequenceHandle active_sequence{};
    // Token-major execution reaches one of these per layer within a token;
    // page-major prompt execution reaches one per row within a layer. Either
    // way the pending set is what the collect boundary drains. A raw pointer
    // to an element is handed to a CUDA host callback, so growing the table
    // must not move the elements that are already in flight.
    std::vector<std::unique_ptr<PhysicalAttentionContext>>
        physical_attention_contexts;
    // Reused block tables for candidate resolution. Both topologies rebuild
    // these once per kind per layer per decoded token; at the declared context
    // a compressed table is 4,096 blocks, so returning them by value allocated
    // on the timed path and cost about 8.1 ms/token. Only one layer's tables
    // are live at a time, so a single pair of buffers suffices.
    std::vector<Dsv4KvBlockInfo> physical_sliding_blocks;
    std::vector<Dsv4KvBlockInfo> physical_compressed_blocks;
    // Learned-index table for sparse selection. Held separately because it is
    // resolved in a different phase from the attention candidate tables.
    std::vector<Dsv4KvBlockInfo> physical_index_blocks;
    // Host-evaluated rope rotation for the index query, uploaded rather than
    // recomputed on the device so the two agree bit for bit.
    std::vector<float> index_rope_cosines;
    std::vector<float> index_rope_sines;
    UninitializedFloatScratch attention_page_query_rank_scratch;
    UninitializedFloatScratch attention_page_query_scratch;
    UninitializedFloatScratch attention_page_kv_scratch;
    std::vector<std::unique_ptr<HostMoeContext>> host_moe_contexts;
    std::uint32_t host_moe_pending{};
    // Page-major prompt execution enqueues the CPU-MoE chain once per row
    // inside one layer rather than once per layer inside one token. The chain
    // index is what keeps the callback contexts distinct and keeps the order
    // guard meaningful; token-major execution leaves this empty and indexes by
    // layer exactly as before.
    std::optional<std::uint32_t> host_moe_chain_row;
    std::uint64_t host_moe_routed_cpu_before{};
    DeviceHeadContext device_head_context{};
    std::vector<Dsv4KvDeviceLease> pending_attention_leases;
    std::vector<Dsv4WeightCache::Lease> pending_attention_weights;
    std::unique_ptr<HostWorkerPool> attention_workers;
    std::unique_ptr<HostWorkerPool> expert_workers;
    std::vector<int> expert_lane_nodes;
    std::vector<std::size_t> expert_lane_positions;
    std::array<std::vector<std::size_t>, 2U> expert_node_lanes;
    std::vector<float> tiled_activation;
    std::vector<float> tiled_routed;
    std::array<std::atomic<std::uint64_t>, 48U> tiled_lane_next{};
    std::array<std::uint64_t, 48U> tiled_lane_end{};
    DiagnosticTrace diagnostics;
    Dsv4DeviceMoeStats device_moe_stats;
    Dsv4GraphStats graph_stats;
    std::vector<int> devices;
    std::vector<std::uint64_t> capacities;
    std::vector<std::size_t> schedule;
    std::size_t mhc_slot{};
    std::unordered_map<std::string, std::vector<float>> host_tensors;
    std::unordered_map<std::string, std::vector<float>> prepacked_mhc;
    std::array<std::array<CudaDsv4MhcWeights, 2U>, kLayers>
        device_mhc_weights;
    bool pending_mhc_attention_transition{};
    std::array<float, kHidden> combined_attention_mhc_input{};
    bool completed_attention_mhc_transition{};
    std::array<float, kExperts> combined_router_logits{};
    bool completed_router_projection{};
    bool deferred_attention_moe_input{};
    std::unordered_map<std::string, std::vector<std::byte>> host_raw;
    std::array<AttentionState, kLayers> attention_state;
    std::vector<std::uint32_t> cached_token_ids;
    RouteTraceWriter route_trace;
    RoutePredictor route_predictor;
    std::vector<RoutePrediction> pending_prefetch_predictions;
    std::vector<RouteEvent> deferred_route_events;
    bool defer_prefill_observability{};
    // Set only while a future-entropy lookahead is in flight. Suppresses the
    // observability that describes the emitted sequence; the graph itself is
    // unchanged, so the logits a lookahead reads are the real ones.
    bool speculative_pass{};
    std::mt19937_64 sampler;
    SamplingOptions active_sampling;
    std::vector<std::uint32_t> sampled_token_counts;
    std::vector<std::uint32_t> sampled_token_ids;
    TokenLogprob last_sample;
    bool initialized{};
    bool reusable_sequence{};
    std::uint64_t active_request_id{};
    std::uint64_t generated_requests{};
    std::uint32_t active_prompt_tokens{};

    [[nodiscard]] std::size_t layer_device(std::uint32_t layer) const {
        return schedule[layer % schedule.size()];
    }

    [[nodiscard]] std::size_t expert_device(std::uint32_t expert) const {
        return schedule[expert % schedule.size()];
    }

    ParseResult<const std::vector<float>*> host_tensor(std::string name,
                                                       std::uint64_t ceiling) {
        ParseResult<const std::vector<float>*> result;
        const auto found = host_tensors.find(name);
        if (found != host_tensors.end()) {
            result.value = &found->second;
            return result;
        }
        auto loaded = checkpoint->read_f32(name, ceiling);
        if (!loaded.ok()) {
            result.errors = std::move(loaded.errors);
            return result;
        }
        auto inserted = host_tensors.emplace(std::move(name), std::move(loaded.value));
        result.value = &inserted.first->second;
        return result;
    }

    ParseResult<const std::vector<std::byte>*> raw_tensor(std::string name,
                                                          std::uint64_t ceiling) {
        ParseResult<const std::vector<std::byte>*> result;
        const auto found = host_raw.find(name);
        if (found != host_raw.end()) {
            result.value = &found->second;
            return result;
        }
        auto loaded = checkpoint->read(name, ceiling);
        if (!loaded.ok()) {
            result.errors = std::move(loaded.errors);
            return result;
        }
        auto inserted = host_raw.emplace(std::move(name), std::move(loaded.value));
        result.value = &inserted.first->second;
        return result;
    }

    ValidationResult linear(std::size_t slot, std::string_view base,
                            std::uint64_t outputs, std::uint64_t inputs,
                            std::span<const float> input,
                            std::span<float> output,
                            bool bf16_output = true) {
        return weights->matmul(slot, base, outputs, inputs, input, 1U,
                               output, bf16_output);
    }

    ValidationResult linear_rows(std::size_t slot, std::string_view base,
                                 std::uint64_t outputs, std::uint64_t inputs,
                                 std::span<const float> input,
                                 std::uint32_t rows,
                                 std::span<float> output,
                                 bool bf16_output = true,
                                 CudaMatmulProfile* profile = nullptr,
                                 bool dsv4_fp8_tensor_page = false) {
        return weights->matmul(slot, base, outputs, inputs, input, rows,
                               output, bf16_output, profile,
                               dsv4_fp8_tensor_page);
    }

    ValidationResult norm(std::span<float> output, std::span<const float> input,
                          std::string name) {
        ValidationResult result;
        auto weight = host_tensor(std::move(name), input.size());
        if (!weight.ok()) {
            append_errors(result, std::move(weight.errors));
            return result;
        }
        result = rms_norm_f32(output, input, *weight.value, kRmsEpsilon);
        if (result.ok()) round_bf16(output);
        return result;
    }

    ValidationResult norm_rows(std::span<float> output,
                               std::span<const float> input,
                               std::uint32_t rows,
                               std::uint32_t columns,
                               std::string name) {
        ValidationResult result;
        if (rows == 0U || columns == 0U ||
            input.size() != static_cast<std::size_t>(rows) * columns ||
            output.size() != input.size()) {
            result.errors.emplace_back(
                "DeepSeek batched normalization spans have incompatible sizes");
            return result;
        }
        auto weight = host_tensor(std::move(name), columns);
        if (!weight.ok()) {
            append_errors(result, std::move(weight.errors));
            return result;
        }
        std::vector<ValidationResult> row_results(rows);
        const auto normalize_row = [&](std::size_t row) {
            auto output_row = output.subspan(row * columns, columns);
            const auto input_row = input.subspan(row * columns, columns);
            row_results[row] = rms_norm_f32(output_row, input_row,
                                             *weight.value, kRmsEpsilon);
            if (row_results[row].ok()) round_bf16(output_row);
        };
        if (attention_workers != nullptr && rows > 1U) {
            result = attention_workers->parallel_for(rows, normalize_row);
            if (!result.ok()) return result;
        } else {
            for (std::uint32_t row = 0U; row < rows; ++row) {
                normalize_row(row);
            }
        }
        for (auto& row_result : row_results) {
            if (!row_result.ok()) {
                append_errors(result, std::move(row_result.errors));
            }
        }
        return result;
    }

    ValidationResult warmup();
    // `parallel_projection` is for the one-row decode path only. A prefill
    // page calls this once per row, where a per-row pool dispatch would cost
    // more than the projection it splits; rows are the parallel axis there.
    ValidationResult mhc_pre(std::span<float> reduced, Dsv4MhcMix& mix,
                             std::span<const float> hidden,
                             const std::string& projection_name,
                             std::span<const float> projection,
                             std::span<const float> scale,
                             std::span<const float> base,
                             bool parallel_projection = false);
    ValidationResult reset_sequence(std::uint32_t active_context_tokens);
    // Drops every KV device lease this runtime still holds from a finished
    // generation. Page leases are deliberately kept until the next token's
    // same-layer prepare, so when generation ends they are still open on the
    // last blocks. An append cannot mutate a leased block, so a continuation
    // that skips reset_sequence must drop them before it prefills.
    void release_retained_kv_leases() noexcept;
    // Admits rank-local decode against measured byte accounts, then loads the
    // rank-sharded weights and initializes the two-rank executor. Fail-closed:
    // any rejection leaves rank_local_active false and the centralized path
    // untouched, so a refused opt-in degrades to a reported error rather than
    // to a silently different decode.
    ValidationResult admit_rank_local();
    // One rank-local decode token across the two-rank executor. Reached only
    // when admission passed and the position is past the prompt.
    ValidationResult rank_local_forward_hidden(
        std::uint32_t token, std::uint32_t position, std::span<float> hidden,
        std::vector<float>* fused_logits);
    // Everything one rank-local layer needs before the executor runs it:
    // the replicated KV rows for this position written into both ranks'
    // device pages, the candidate set, and the borrowed call views.
    ValidationResult rank_local_prepare_layer(
        std::uint32_t layer, std::uint32_t token, std::uint32_t position,
        Dsv4RankLocalKvTransaction& transaction,
        RankLocalLayerScratch& scratch, Dsv4RankLocalLayerCall& call);
    static bool rank_local_page_patch_callback(
        void* opaque, const CudaDsv4AttentionPrepareHostView& view);
    bool complete_rank_local_page_patch(
        RankLocalPageContext& context,
        const CudaDsv4AttentionPrepareHostView& view);
    // Writes one layer's committed rows into every rank's device page. The
    // bytes were encoded once by the transaction; this is the transport.
    ValidationResult rank_local_patch_pages(
        const RankLocalLayerScratch& scratch);
    // Pages and candidates for both ranks over one shared logical row order.
    // `in_chain` selects the form an indexed layer takes inside the queued
    // chain: every attendable compressed block is leased and described up
    // front, and the compressed candidate region is left for the device to
    // resolve. `indexed_positions` is then unused and must be empty.
    ValidationResult rank_local_candidates(
        std::uint32_t layer, std::uint32_t position,
        std::span<const std::uint32_t> indexed_positions,
        RankLocalLayerScratch& scratch, bool in_chain = false);
    // Per-rank index weights and learned-index pages for one in-chain layer.
    // Both ranks select over the same replicated rows on their own device.
    ValidationResult rank_local_index_selection(
        std::uint32_t layer, std::uint32_t position,
        RankLocalLayerScratch& scratch);
    // Makes both ranks' copies of every indexed layer's projection weights
    // resident before decode begins. No-op outside an admitted rank-local
    // session with an admitted indexer.
    ValidationResult rank_local_warm_index_projections();
    ParseResult<std::vector<float>> kv_row(
        std::uint32_t layer, Dsv4KvBlockKind kind,
        std::uint64_t logical_row);
    void reset_diagnostics();
    void record_layer_hash(std::uint32_t position, std::uint32_t token,
                           std::uint32_t layer, std::span<const float> hidden);
    void record_operation_hash(std::uint32_t position, std::uint32_t token,
                               std::uint32_t layer, std::string_view operation,
                               std::span<const float> values);
    void record_logits(std::uint32_t position, std::uint32_t token,
                       std::uint32_t selected, std::span<const float> logits);
    ValidationResult embed(std::uint32_t token, std::span<float> output);
    ValidationResult compressor(std::uint32_t layer, std::span<const float> input,
                                std::uint32_t position,
                                std::span<const float> prepared_values = {},
                                std::span<const float> prepared_scores = {});
    ValidationResult compress_state(std::uint32_t layer,
                                    CompressorState& state,
                                    const std::string& prefix,
                                    std::span<const float> input,
                                    std::uint32_t position,
                                    std::span<const float> frequencies,
                                    std::span<const float> prepared_values = {},
                                    std::span<const float> prepared_scores = {},
                                    Dsv4KvPhysicalAppend* prepared_append = nullptr,
                                    std::span<std::byte> prepared_patch = {},
                                    // When set, the pooled row is returned
                                    // instead of being appended, and stays
                                    // empty away from a block boundary. The
                                    // rank-local path needs the row itself
                                    // because it writes one logical row into
                                    // two devices' pages.
                                    std::vector<float>* pooled_row = nullptr);
    static bool physical_attention_prepare_callback(
        void* opaque, const CudaDsv4AttentionPrepareHostView& view);
    bool complete_physical_attention_prepare(
        PhysicalAttentionContext& context,
        const CudaDsv4AttentionPrepareHostView& view);
    ValidationResult index_positions(std::uint32_t layer,
                                     std::span<const float> input,
                                     std::span<const float> query_rank,
                                     std::uint32_t position,
                                     std::vector<std::uint32_t>& selected,
                                     std::span<const float> prepared_values = {},
                                     std::span<const float> prepared_scores = {},
                                     bool device_prepared_source = false);
    // Selection only, against an indexer compressor whose row for this
    // position has already been advanced. The queued page-update path commits
    // that row in stream order from the preparation callback, so it must not
    // be appended a second time here.
    //
    // `device_prepared_source` asserts that this layer's preparation command
    // ran on this layer's device immediately before, so the index projections
    // may consume its device activations instead of `input` and `query_rank`.
    // Only a caller that issued that preparation may set it: the batched
    // prefill page projects many rows at once and leaves no such state, so it
    // passes false and the host projections run.
    ValidationResult index_select(std::uint32_t layer,
                                  std::span<const float> input,
                                  std::span<const float> query_rank,
                                  std::uint32_t position,
                                  std::vector<std::uint32_t>& selected,
                                  bool device_prepared_source = false);
    ValidationResult attention(std::uint32_t layer, std::span<const float> input,
                               std::uint32_t position, std::span<float> output);
    ValidationResult physical_paged_attention(
        std::uint32_t layer, std::span<const float> queries,
        std::span<const float> sinks, std::uint32_t position,
        std::span<const std::uint32_t> indexed_positions,
        std::span<float> diagnostic_branch,
        PhysicalAttentionPageSet* page_set = nullptr);
    ValidationResult physical_paged_attention_page(
        std::uint32_t layer, std::span<const float> input,
        std::span<const float> query_rank, std::span<const float> queries,
        std::span<const float> sinks, std::uint32_t position_base,
        std::span<const std::uint32_t> row_slots,
        std::span<float> diagnostic_branches);
    ValidationResult attention_append_prepared(
        std::uint32_t layer, std::span<const float> input,
        std::span<const float> kv, std::uint32_t position,
        std::span<const float> compressor_values = {},
        std::span<const float> compressor_scores = {},
        std::span<const float> index_compressor_values = {},
        std::span<const float> index_compressor_scores = {},
        bool append_index_compressor = false,
        std::uint64_t sliding_retention_floor =
            std::numeric_limits<std::uint64_t>::max());
    ValidationResult attention_attend_prepared(
        std::uint32_t layer, std::span<const float> input,
        std::span<const float> query_rank, std::span<const float> queries,
        std::uint32_t position, std::span<float> output,
        bool index_compressor_prepared = false,
        bool device_prepared_source = false,
        PhysicalAttentionPageSet* page_set = nullptr);
    ValidationResult attention_prepared(
        std::uint32_t layer, std::span<const float> input,
        std::span<const float> query_rank, std::span<const float> queries,
        std::span<const float> kv, std::uint32_t position,
        std::span<float> output,
        std::span<const float> compressor_values = {},
        std::span<const float> compressor_scores = {},
        std::span<const float> index_compressor_values = {},
        std::span<const float> index_compressor_scores = {},
        bool device_prepared_source = false);
    // `row_slots`, when given, names the device mHC slot each row belongs to.
    // The physical attention command publishes its branch into the selected
    // slot's workspace, so a page must point it at the right row before every
    // per-row attention. It must be empty on the token-major path, which owns
    // exactly one slot.
    ValidationResult attention_page(std::uint32_t layer,
                                    std::span<const float> input,
                                    std::uint32_t position_base,
                                    std::span<float> output,
                                    std::span<const std::uint32_t> row_slots = {});
    ValidationResult expert(std::uint32_t layer, std::uint32_t expert_id,
                            float routed_coefficient,
                            std::span<const float> input, std::span<float> output);
    ValidationResult device_moe(std::uint32_t layer,
                                const Dsv4Route& route,
                                std::span<const float> input,
                                std::span<float> output);
    ValidationResult host_routed_moe(std::uint32_t layer,
                                    const Dsv4Route& route,
                                    std::span<const float> input,
                                    std::span<float> output);
    ValidationResult host_routed_moe_from_device_input(
        std::uint32_t layer, std::uint32_t token, std::uint32_t position,
        std::span<float> output);
    ValidationResult enqueue_host_routed_moe(
        std::uint32_t layer, std::uint32_t token,
        std::uint32_t position);
    static bool host_routed_moe_callback(
        void* opaque, std::span<const std::uint16_t> encoded_hidden,
        std::span<const float> router_logits,
        std::span<float> rank_partials);
    bool execute_host_routed_moe_callback(
        HostMoeContext& context,
        std::span<const std::uint16_t> encoded_hidden,
        std::span<const float> router_logits,
        std::span<float> rank_partials);
    ValidationResult collect_host_routed_moe_chain();
    // Grow-on-demand accessors for the two pending-callback tables. Both are
    // addressed by layer in token-major execution and by page row in
    // page-major prompt execution, so neither bound is known at construction.
    PhysicalAttentionContext& physical_attention_context(std::size_t index);
    HostMoeContext& host_moe_context(std::size_t index);
    // Dispatch `tasks` shard-local tasks across the NUMA-addressed expert
    // pool. Every lane runs the whole range for its own shard; `steal` lets a
    // finished lane claim work from another lane on the same shard.
    template <typename Operation>
    ValidationResult run_expert_ranges(
        std::uint64_t tasks, Operation&& operation, bool steal) {
        constexpr std::size_t shards = 2U;
        const auto lanes = expert_workers->size();
        if (steal) {
            for (std::size_t lane = 0U; lane < lanes; ++lane) {
                const auto shard = static_cast<std::size_t>(
                    expert_lane_nodes[lane]);
                const auto position = expert_lane_positions[lane];
                const auto workers = expert_node_lanes[shard].size();
                tiled_lane_next[lane].store(
                    tasks * position / workers, std::memory_order_relaxed);
                tiled_lane_end[lane] = tasks * (position + 1U) / workers;
            }
        }
        auto dispatched = expert_workers->parallel_for_addressed(
            lanes, [&](std::size_t lane) {
                const auto shard = static_cast<std::size_t>(
                    expert_lane_nodes[lane]);
                if (shard >= shards || expert_node_lanes[shard].empty()) return;
                const auto position = expert_lane_positions[lane];
                const auto& node_lanes = expert_node_lanes[shard];
                if (!steal) {
                    const auto begin = tasks * position / node_lanes.size();
                    const auto end = tasks * (position + 1U) /
                                     node_lanes.size();
                    for (auto task = begin; task < end; ++task) {
                        operation(shard, task);
                    }
                    return;
                }
                const auto claim = [&](std::size_t owner) {
                    const auto task = tiled_lane_next[owner].fetch_add(
                        1U, std::memory_order_relaxed);
                    if (task >= tiled_lane_end[owner]) return false;
                    operation(shard, task);
                    return true;
                };
                while (claim(lane)) {}
                for (;;) {
                    bool stole = false;
                    for (std::size_t offset = 1U;
                         offset < node_lanes.size(); ++offset) {
                        const auto owner = node_lanes[
                            (position + offset) % node_lanes.size()];
                        if (claim(owner)) {
                            stole = true;
                            break;
                        }
                    }
                    if (!stole) break;
                }
            });
        if (dispatched.ok() && steal) {
            for (std::size_t lane = 0U; lane < lanes; ++lane) {
                if (tiled_lane_next[lane].load(std::memory_order_relaxed) <
                    tiled_lane_end[lane]) {
                    dispatched.errors.emplace_back(
                        "DeepSeek host-routed MoE left a lane range unfinished");
                    break;
                }
            }
        }
        return dispatched;
    }
    ValidationResult host_routed_moe_impl(
        std::uint32_t layer, const Dsv4Route* route,
        std::span<const float> input, std::span<float> output,
        std::uint32_t token, std::uint32_t token_position);
    ValidationResult moe(std::uint32_t layer, std::uint32_t token,
                         std::span<const float> input, std::span<float> output,
                         std::uint32_t position);
    ValidationResult route_moe(std::uint32_t layer, std::uint32_t token,
                               std::span<const float> logits,
                               std::uint32_t position, Dsv4Route& route);
    ValidationResult execute_moe_page(std::uint32_t layer,
                                      std::span<const Dsv4Route> routes,
                                      std::span<const float> input,
                                      std::span<float> output);
    ValidationResult execute_moe(std::uint32_t layer, const Dsv4Route& route,
                                 std::span<const float> input,
                                 std::span<float> output);
    ValidationResult moe_page(std::uint32_t layer,
                              std::span<const std::uint32_t> tokens,
                              std::span<const float> input,
                              std::span<float> output,
                              std::uint32_t position_base);
    ValidationResult block(std::uint32_t layer, std::uint32_t token,
                           std::span<float> hidden, std::uint32_t position);
    ValidationResult block_page(std::uint32_t layer,
                                std::span<const std::uint32_t> tokens,
                                std::span<float> hidden,
                                std::uint32_t position_base);
    ValidationResult forward_hidden(std::uint32_t token, std::uint32_t position,
                                    std::span<float> hidden,
                                    std::vector<float>* fused_logits = nullptr);
    ValidationResult device_mhc_forward_hidden(
        std::uint32_t token, std::uint32_t position,
        std::span<float> hidden, std::vector<float>* fused_logits);
    // Same fused device graph as device_mhc_forward_hidden, visited
    // branch-outermost over a page of prompt rows instead of token by token.
    // Each row owns an mHC slot, so its command sequence is unchanged; only
    // the order in which rows reach each command differs. The caller has
    // already embedded every row and must exclude any row that needs logits.
    ValidationResult device_mhc_forward_prefill_page(
        std::span<const std::uint32_t> tokens, std::uint32_t position_base,
        std::span<float> hidden);
    ValidationResult device_mhc_forward_prefill_page_impl(
        std::span<const std::uint32_t> tokens, std::uint32_t position_base,
        std::span<float> hidden);
    static bool device_head_callback(
        void* opaque, std::span<const std::uint16_t> encoded_hidden,
        std::span<float> reduced);
    bool execute_device_head_callback(
        DeviceHeadContext& context,
        std::span<const std::uint16_t> encoded_hidden,
        std::span<float> reduced);
    ValidationResult head_logits(std::span<const float> hidden,
                                 std::vector<float>& logits);
    ParseResult<std::uint32_t> sample_hidden(std::uint32_t token,
                                             std::uint32_t position,
                                             std::span<const float> hidden,
                                             const std::vector<float>*
                                                 prepared_logits = nullptr);
    ParseResult<std::uint32_t> forward_prefill(
        std::span<const std::uint32_t> tokens,
        std::uint32_t position_base);
    ValidationResult flush_deferred_routes();
    ValidationResult flush_prefill_observability();
    ParseResult<std::uint32_t> forward_token(std::uint32_t token,
                                             std::uint32_t position,
                                             bool logits);
    SpeculativeState capture_speculative_state(std::uint32_t position) const;
    ValidationResult restore_speculative_state(const SpeculativeState& saved);
    ValidationResult future_entropy(std::span<const std::uint32_t> candidates,
                                    std::uint32_t top_n, std::uint32_t position,
                                    std::span<double> normalized_entropy);
};

// Private implementation fragments are included into this one owning
// translation unit. They divide responsibility without changing linkage,
// initialization order, or the runtime's numerical boundaries.
#include "detail/runtime_loading.inc.cpp"
#include "detail/runtime_attention.inc.cpp"
#include "detail/runtime_moe.inc.cpp"
#include "detail/runtime_rank_local.inc.cpp"
#include "detail/runtime_generation.inc.cpp"
#include "detail/runtime_public.inc.cpp"
