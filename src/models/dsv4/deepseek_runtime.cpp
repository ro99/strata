#include "strata/deepseek_runtime.hpp"

#include "../common/cuda_stats_delta.hpp"

#include "strata/deepseek_ops.hpp"
#include "strata/deepseek_host_expert.hpp"
#include "strata/dsv4_attention_kv.hpp"
#include "strata/dsv4_rank_local_kv.hpp"
#include "strata/dsv4_rank_local_weights.hpp"
// The two-rank executor is a CUDA translation unit compiled only when NCCL is
// available. Rank-local decode already requires NCCL at config validation, so
// the session it owns is guarded on the same condition rather than declared
// and left unlinkable.
#if defined(STRATA_HAS_NCCL)
#include "strata/dsv4_rank_local_layer_executor.hpp"
#endif
#include "strata/model_adapter.hpp"
#include "strata/hardware_profile.hpp"
#include "strata/numa_topology.hpp"
#include "strata/numerics.hpp"
#include "strata/route_predictor.hpp"
#include "strata/sampling.hpp"
#include "strata/runtime_support.hpp"
#include "strata/tokenizer.hpp"
#include "strata/trace.hpp"
#include "strata/worker_pool.hpp"

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

void DeepSeekV4Runtime::Impl::reset_diagnostics() {
    diagnostics = {};
    diagnostics.logit_trace_enabled = config.enable_logit_trace;
    diagnostics.layer_hash_trace_enabled = config.enable_layer_hash_trace;
    diagnostics.logit_top_k = config.logit_trace_top_k;
    diagnostics.index_selection_trace_hash = kDiagnosticFnvOffset;
    graph_stats.prefill_pages = 0U;
    graph_stats.prefill_max_page_tokens = 0U;
    graph_stats.prefill_max_workspace_bytes = 0U;
    pending_prefetch_predictions.clear();
    deferred_route_events.clear();
    defer_prefill_observability = false;
    if (config.enable_logit_trace) {
        diagnostics.logit_aggregate.trace_hash = diagnostic_hash_u32(
            kDiagnosticFnvOffset, config.logit_trace_top_k);
        diagnostics.logits.reserve(config.maximum_context_tokens);
    }
    if (config.enable_layer_hash_trace) {
        diagnostics.layer_hash_trace_hash = diagnostic_hash_u32(
            kDiagnosticFnvOffset, kLayers);
        diagnostics.layer_hashes.reserve(
            static_cast<std::size_t>(config.maximum_context_tokens) * kLayers);
        diagnostics.operation_hash_trace_hash = diagnostic_hash_u32(
            kDiagnosticFnvOffset, kLayers);
    }
}

void DeepSeekV4Runtime::Impl::record_layer_hash(
    std::uint32_t position, std::uint32_t token, std::uint32_t layer,
    std::span<const float> hidden) {
    const auto hash = stable_bf16_hash(hidden);
    diagnostics.layer_hashes.push_back({position, token, layer, hash});
    auto aggregate = diagnostics.layer_hash_trace_hash;
    aggregate = diagnostic_hash_u32(aggregate, position);
    aggregate = diagnostic_hash_u32(aggregate, token);
    aggregate = diagnostic_hash_u32(aggregate, layer);
    diagnostics.layer_hash_trace_hash = diagnostic_hash_u64(aggregate, hash);
}

void DeepSeekV4Runtime::Impl::record_operation_hash(
    std::uint32_t position, std::uint32_t token,
    std::uint32_t layer, std::string_view operation,
    std::span<const float> values) {
    const auto hash = stable_bf16_hash(values);
    diagnostics.operation_hashes.push_back(
        {position, token, layer, std::string(operation), hash});
    auto aggregate = diagnostics.operation_hash_trace_hash;
    aggregate = diagnostic_hash_u32(aggregate, position);
    aggregate = diagnostic_hash_u32(aggregate, token);
    aggregate = diagnostic_hash_u32(aggregate, layer);
    for (const char ch : operation) {
        aggregate = diagnostic_hash_byte(aggregate, static_cast<std::uint8_t>(ch));
    }
    diagnostics.operation_hash_trace_hash = diagnostic_hash_u64(aggregate, hash);
}

void DeepSeekV4Runtime::Impl::record_logits(
    std::uint32_t position, std::uint32_t token, std::uint32_t selected,
    std::span<const float> logits) {
    auto analysis = analyze_logits(logits, config.logit_trace_top_k);
    const auto& summary = analysis.summary;
    auto& aggregate = diagnostics.logit_aggregate;
    ++aggregate.forward_count;
    aggregate.value_count += summary.value_count;
    aggregate.finite_count += summary.finite_count;
    aggregate.non_finite_count += summary.non_finite_count;
    aggregate.sum += summary.sum;
    aggregate.absolute_sum += summary.absolute_sum;
    aggregate.square_sum += summary.square_sum;
    if (summary.has_finite) {
        if (!aggregate.has_finite) {
            aggregate.minimum = summary.minimum;
            aggregate.maximum = summary.maximum;
            aggregate.has_finite = true;
        } else {
            aggregate.minimum = std::min(aggregate.minimum, summary.minimum);
            aggregate.maximum = std::max(aggregate.maximum, summary.maximum);
        }
    }
    auto hash = aggregate.trace_hash;
    hash = diagnostic_hash_u32(hash, position);
    hash = diagnostic_hash_u32(hash, token);
    hash = diagnostic_hash_u32(hash, selected);
    aggregate.trace_hash = diagnostic_hash_u64(hash, summary.raw_f32_hash);
    diagnostics.logits.push_back(
        {position, token, selected, summary, std::move(analysis.top)});
}

ValidationResult DeepSeekV4Runtime::Impl::warmup() {
    ValidationResult result;
    const auto preload = [this](ValidationResult& target, std::size_t slot,
                                const std::string& base, std::uint64_t rows,
                                std::uint64_t columns) {
        if (target.ok()) target = weights->preload(slot, base, rows, columns);
    };
    const auto load_host = [this](ValidationResult& target, const std::string& name,
                                  std::uint64_t ceiling) {
        if (!target.ok()) return;
        auto loaded = host_tensor(name, ceiling);
        if (!loaded.ok()) append_errors(target, std::move(loaded.errors));
    };
    const auto load_raw = [this](ValidationResult& target, const std::string& name,
                                 std::uint64_t ceiling) {
        if (!target.ok()) return;
        auto loaded = raw_tensor(name, ceiling);
        if (!loaded.ok()) append_errors(target, std::move(loaded.errors));
    };
    const auto load_mhc = [this](ValidationResult& target,
                                 const std::string& name) {
        if (!target.ok()) return;
        auto loaded = host_tensor(name, kMix * kMhc * kHidden);
        if (!loaded.ok()) {
            append_errors(target, std::move(loaded.errors));
            return;
        }
        if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice ||
            !config.prepack_mhc_projection) {
            return;
        }
        try {
            std::vector<float> packed(loaded.value->size());
            auto status = dsv4_pack_mhc_projection_f32(
                packed, *loaded.value, kMix, kMhc * kHidden);
            if (!status.ok()) {
                append_errors(target, std::move(status.errors));
                return;
            }
            prepacked_mhc.emplace(name, std::move(packed));
        } catch (const std::bad_alloc&) {
            target.errors.emplace_back(
                "cannot allocate DeepSeek prepacked mHC projection");
        }
    };

    const auto ratios = deepseek_v4_flash_0731_spec().deepseek_v4.compression_ratios;
    const auto preload_layer = [this, &preload, &ratios](
        ValidationResult& target, std::uint32_t layer) {
        const auto slot = layer_device(layer);
        const auto prefix = layer_prefix(layer);
        const auto attention = prefix + "attn.";
        preload(target, slot, attention + "wq_a", kQueryRank, kHidden);
        preload(target, slot, attention + "wq_b", kHeads * kHeadDim, kQueryRank);
        preload(target, slot, attention + "wkv", kHeadDim, kHidden);
        preload(target, slot, attention + "wo_a", kOutputGroups * kOutputRank,
                kHeads * kHeadDim / kOutputGroups);
        preload(target, slot, attention + "wo_b", kHidden,
                kOutputGroups * kOutputRank);
        preload(target, slot, prefix + "ffn.gate", kExperts, kHidden);
        for (const auto* operation : {"w1", "w3"}) {
            preload(target, slot, prefix + "ffn.shared_experts." + operation,
                    kExpertIntermediate, kHidden);
        }
        preload(target, slot, prefix + "ffn.shared_experts.w2", kHidden,
                kExpertIntermediate);
        const auto ratio = ratios[layer];
        if (ratio != 0U) {
            const auto coefficient = ratio == 4U ? 2U : 1U;
            const auto dimensions = coefficient * kHeadDim;
            preload(target, slot, attention + "compressor.wkv", dimensions, kHidden);
            preload(target, slot, attention + "compressor.wgate", dimensions, kHidden);
            if (ratio == 4U) {
                preload(target, slot, attention + "indexer.wq_b", 64U * 128U,
                        kQueryRank);
                preload(target, slot, attention + "indexer.weights_proj", 64U, kHidden);
                preload(target, slot, attention + "indexer.compressor.wkv", 2U * 128U,
                        kHidden);
                preload(target, slot, attention + "indexer.compressor.wgate", 2U * 128U,
                        kHidden);
            }
        }
    };
    const auto load_host_layer = [this, &load_host, &load_raw, &load_mhc, &ratios](
        ValidationResult& target, std::uint32_t layer) {
        const auto prefix = layer_prefix(layer);
        const auto attention = prefix + "attn.";
        load_host(target, attention + "q_norm.weight", kQueryRank);
        load_host(target, attention + "kv_norm.weight", kHeadDim);
        load_host(target, attention + "attn_sink", kHeads);
        load_host(target, prefix + "attn_norm.weight", kHidden);
        load_host(target, prefix + "ffn_norm.weight", kHidden);
        constexpr std::array<const char*, 2U> branches{"attn", "ffn"};
        for (std::size_t branch_index = 0U;
             branch_index < branches.size(); ++branch_index) {
            const std::string branch(branches[branch_index]);
            const auto projection_name = prefix + "hc_" + branch + "_fn";
            const auto scale_name = prefix + "hc_" + branch + "_scale";
            const auto base_name = prefix + "hc_" + branch + "_base";
            const auto norm_name = prefix + branch + "_norm.weight";
            load_mhc(target, projection_name);
            load_host(target, scale_name, 3U);
            load_host(target, base_name, kMix);
            if (!target.ok() ||
                config.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice) {
                continue;
            }
            auto projection = host_tensor(
                projection_name, kMix * kMhc * kHidden);
            auto scale = host_tensor(scale_name, 3U);
            auto base = host_tensor(base_name, kMix);
            auto norm_weight = host_tensor(norm_name, kHidden);
            if (!projection.ok()) {
                append_errors(target, std::move(projection.errors));
            }
            if (!scale.ok()) append_errors(target, std::move(scale.errors));
            if (!base.ok()) append_errors(target, std::move(base.errors));
            if (!norm_weight.ok()) {
                append_errors(target, std::move(norm_weight.errors));
            }
            if (!target.ok()) continue;
            target = cuda.upload_dsv4_mhc_weights(
                devices[mhc_slot], *projection.value, *scale.value,
                *base.value, *norm_weight.value,
                device_mhc_weights[layer][branch_index]);
            if (target.ok()) host_tensors.erase(projection_name);
        }
        if (layer < 3U) {
            load_raw(target, prefix + "ffn.gate.tid2eid",
                     8ULL * kVocabulary * kTopK);
        } else {
            load_host(target, prefix + "ffn.gate.bias", kExperts);
        }
        const auto ratio = ratios[layer];
        if (ratio != 0U) {
            const auto coefficient = ratio == 4U ? 2U : 1U;
            const auto dimensions = coefficient * kHeadDim;
            load_host(target, attention + "compressor.ape",
                      static_cast<std::uint64_t>(ratio) * dimensions);
            load_host(target, attention + "compressor.norm.weight", kHeadDim);
            if (ratio == 4U) {
                load_host(target, attention + "indexer.compressor.ape",
                          4U * 2U * 128U);
                load_host(target, attention + "indexer.compressor.norm.weight", 128U);
            }
        }
    };
    const auto preload_head = [this, &preload](ValidationResult& target) {
        preload(target, layer_device(kLayers - 1U), "head", kVocabulary, kHidden);
    };
    const auto load_host_head = [this, &load_host](ValidationResult& target) {
        load_host(target, "norm.weight", kHidden);
        load_host(target, "hc_head_fn", kMhc * kMhc * kHidden);
        load_host(target, "hc_head_scale", 1U);
        load_host(target, "hc_head_base", kMhc);
    };

    if (config.spine_warmup_workers == 1U) {
        for (std::uint32_t layer = 0U; layer < kLayers && result.ok(); ++layer) {
            preload_layer(result, layer);
            load_host_layer(result, layer);
            if (config.verbose) {
                std::cerr << "[deepseek-load] resident spine layer " << layer + 1U
                          << '/' << kLayers << '\n';
            }
        }
        if (result.ok()) {
            preload_head(result);
            load_host_head(result);
            if (result.ok() &&
                config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
                result = cuda.reserve_dsv4_mhc_head(
                    devices[mhc_slot], kVocabulary);
            }
        }
        return result;
    }

    std::vector<ValidationResult> device_results(devices.size());
    std::atomic<std::size_t> next_slot{};
    const auto worker = [&] {
        for (;;) {
            const auto slot = next_slot.fetch_add(1U, std::memory_order_relaxed);
            if (slot >= devices.size()) return;
            auto& target = device_results[slot];
            for (std::uint32_t layer = 0U; layer < kLayers && target.ok(); ++layer) {
                if (layer_device(layer) == slot) preload_layer(target, layer);
            }
            if (target.ok() && layer_device(kLayers - 1U) == slot) {
                preload_head(target);
            }
        }
    };
    const auto active_workers = std::min<std::size_t>(
        config.spine_warmup_workers, devices.size());
    std::vector<std::thread> workers;
    workers.reserve(active_workers);
    for (std::size_t index = 0U; index < active_workers; ++index) {
        workers.emplace_back(worker);
    }
    for (auto& thread : workers) thread.join();
    for (auto& device_result : device_results) {
        if (!device_result.ok()) append_errors(result, std::move(device_result.errors));
    }
    for (std::uint32_t layer = 0U; layer < kLayers && result.ok(); ++layer) {
        load_host_layer(result, layer);
        if (config.verbose) {
            std::cerr << "[deepseek-load] resident spine layer " << layer + 1U
                      << '/' << kLayers << '\n';
        }
    }
    if (result.ok()) load_host_head(result);
    if (result.ok() &&
        config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
        result = cuda.reserve_dsv4_mhc_head(devices[mhc_slot], kVocabulary);
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::mhc_pre(
    std::span<float> reduced, Dsv4MhcMix& mix,
    std::span<const float> hidden, const std::string& projection_name,
    std::span<const float> projection, std::span<const float> scale,
    std::span<const float> base, bool parallel_projection) {
    if (!config.prepack_mhc_projection) {
        return dsv4_mhc_pre_f32(
            reduced, mix, hidden, projection, scale, base);
    }
    const auto found = prepacked_mhc.find(projection_name);
    if (found == prepacked_mhc.end()) {
        ValidationResult result;
        result.errors.emplace_back(
            "DeepSeek prepacked mHC projection is unavailable: " +
            projection_name);
        return result;
    }
    ++graph_stats.mhc_prepacked_calls;
    Dsv4ParallelFor lanes;
    if (parallel_projection && attention_workers != nullptr &&
        attention_workers->size() > 1U) {
        lanes = [this](std::size_t tasks,
                       const std::function<void(std::size_t)>& body) {
            return attention_workers->parallel_for(tasks, body);
        };
    }
    return dsv4_mhc_prepacked_f32(
        reduced, mix, hidden, found->second, scale, base,
        kDsv4MhcMultiplier, kDsv4MhcSinkhornIterations, kDsv4NormEpsilon,
        lanes);
}

void DeepSeekV4Runtime::Impl::release_retained_kv_leases() noexcept {
    for (auto& scratch : rank_local_scratch) {
        scratch.compressed_block_leased.clear();
        scratch.leases.clear();
        scratch.index_leases.clear();
        for (auto& pages : scratch.pages) pages.clear();
        for (auto& pages : scratch.index_pages) pages.clear();
    }
    // The centralized path parks its leases here until the matching MoE
    // collect, which an ended generation never reaches.
    pending_attention_leases.clear();
}

ValidationResult DeepSeekV4Runtime::Impl::reset_sequence(
    std::uint32_t active_context_tokens) {
    ValidationResult result;
    reusable_sequence = false;
    cached_token_ids.clear();
    // Page slots and leases describe blocks of the sequence being discarded,
    // where the same indices will refer to different pages in the next one.
    release_retained_kv_leases();
    if (kv_cache != nullptr) {
        result = kv_cache->reset_sequence(active_sequence);
        if (!result.ok()) return result;
    }
    const auto ratios = deepseek_v4_flash_0731_spec().deepseek_v4.compression_ratios;
    for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
        auto& state = attention_state[layer];
        if (kv_cache == nullptr) {
            state.sliding.assign(static_cast<std::size_t>(kWindow) * kHeadDim,
                                 0.0F);
        } else {
            state.sliding.clear();
        }
        state.frequencies = rope_frequencies(ratios[layer]);
        auto& compressor_state = state.compressor;
        compressor_state = {};
        compressor_state.ratio = ratios[layer];
        if (compressor_state.ratio == 0U) continue;
        compressor_state.kind = compressor_state.ratio == 4U
            ? Dsv4KvBlockKind::Csa : Dsv4KvBlockKind::Hca;
        compressor_state.coefficient = compressor_state.ratio == 4U ? 2U : 1U;
        compressor_state.head_dim = kHeadDim;
        const auto rows = static_cast<std::size_t>(compressor_state.coefficient) *
                          compressor_state.ratio;
        const auto dimensions = static_cast<std::size_t>(
            compressor_state.coefficient) * compressor_state.head_dim;
        compressor_state.values.assign(rows * dimensions, 0.0F);
        compressor_state.scores.assign(
            rows * dimensions, -std::numeric_limits<float>::infinity());
        const auto compressed_rows =
            (static_cast<std::size_t>(config.maximum_context_tokens) +
             compressor_state.ratio - 1U) / compressor_state.ratio;
        if (kv_cache == nullptr &&
            !compressor_state.compressed.configure(compressed_rows,
                                                   compressor_state.head_dim)) {
            result.errors.emplace_back(
                "cannot reserve paged DeepSeek compressed KV metadata");
            return result;
        }
        state.indexer_compressor = {};
        if (compressor_state.ratio == 4U &&
            active_context_tokens > kIndexTopK * compressor_state.ratio) {
            auto& indexer = state.indexer_compressor;
            indexer.ratio = compressor_state.ratio;
            indexer.kind = Dsv4KvBlockKind::LearnedIndex;
            indexer.coefficient = 2U;
            indexer.head_dim = kIndexHeadDim;
            indexer.rotate_fp4 =
                config.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice;
            const auto indexer_rows = static_cast<std::size_t>(indexer.coefficient) *
                                      indexer.ratio;
            const auto indexer_dimensions =
                static_cast<std::size_t>(indexer.coefficient) * indexer.head_dim;
            indexer.values.assign(indexer_rows * indexer_dimensions, 0.0F);
            indexer.scores.assign(
                indexer_rows * indexer_dimensions,
                -std::numeric_limits<float>::infinity());
            if (kv_cache == nullptr &&
                !indexer.compressed.configure(compressed_rows,
                                              indexer.head_dim)) {
                result.errors.emplace_back(
                    "cannot reserve paged DeepSeek sparse-index metadata");
                return result;
            }
        }
    }
    return result;
}

ParseResult<std::vector<float>> DeepSeekV4Runtime::Impl::kv_row(
    std::uint32_t layer, Dsv4KvBlockKind kind,
    std::uint64_t logical_row) {
    if (kv_cache != nullptr) {
        return kv_cache->row(active_sequence, kind, layer, logical_row);
    }
    ParseResult<std::vector<float>> result;
    const auto& state = attention_state[layer];
    if (kind == Dsv4KvBlockKind::Sliding) {
        const auto values = std::span<const float>(state.sliding).subspan(
            static_cast<std::size_t>(logical_row % kWindow) * kHeadDim,
            kHeadDim);
        result.value.assign(values.begin(), values.end());
        return result;
    }
    const auto& compressed = kind == Dsv4KvBlockKind::LearnedIndex
        ? state.indexer_compressor.compressed : state.compressor.compressed;
    const auto values = compressed.row(logical_row);
    if (values.empty()) {
        result.errors.emplace_back("DeepSeek scalar KV row is unavailable");
    } else {
        result.value.assign(values.begin(), values.end());
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::embed(std::uint32_t token,
                                                 std::span<float> output) {
    ValidationResult result;
    if (token >= kVocabulary || output.size() != static_cast<std::size_t>(kMhc) * kHidden) {
        result.errors.emplace_back("DeepSeek embedding token or output shape is invalid");
        return result;
    }
    const auto embedding = resident.find("embed.weight");
    const auto row_bytes = static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t);
    const auto offset = static_cast<std::size_t>(token) * row_bytes;
    if (embedding.size() < offset + row_bytes) {
        result.errors.emplace_back("DeepSeek resident embedding extent is incomplete");
        return result;
    }
    for (std::uint32_t column = 0U; column < kHidden; ++column) {
        std::uint16_t encoded = 0U;
        std::memcpy(&encoded, embedding.data() + offset + column * sizeof(encoded),
                    sizeof(encoded));
        const float value = std::bit_cast<float>(
            static_cast<std::uint32_t>(encoded) << 16U);
        for (std::uint32_t copy = 0U; copy < kMhc; ++copy) {
            output[static_cast<std::size_t>(copy) * kHidden + column] = value;
        }
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::compressor(
    std::uint32_t layer, std::span<const float> input, std::uint32_t position,
    std::span<const float> prepared_values,
    std::span<const float> prepared_scores) {
    return compress_state(layer, attention_state[layer].compressor,
                          layer_prefix(layer) + "attn.compressor.", input,
                          position, attention_state[layer].frequencies,
                          prepared_values, prepared_scores);
}

bool DeepSeekV4Runtime::Impl::physical_attention_prepare_callback(
    void* opaque, const CudaDsv4AttentionPrepareHostView& view) {
    if (opaque == nullptr) return false;
    auto& context = *static_cast<PhysicalAttentionContext*>(opaque);
    if (context.owner == nullptr) return false;
    return context.owner->complete_physical_attention_prepare(context, view);
}

bool DeepSeekV4Runtime::Impl::rank_local_page_patch_callback(
    void* opaque, const CudaDsv4AttentionPrepareHostView& view) {
    if (opaque == nullptr) return false;
    auto& context = *static_cast<RankLocalPageContext*>(opaque);
    if (context.owner == nullptr) return false;
    return context.owner->complete_rank_local_page_patch(context, view);
}

bool DeepSeekV4Runtime::Impl::complete_rank_local_page_patch(
    RankLocalPageContext& context,
    const CudaDsv4AttentionPrepareHostView& view) {
    const auto started = std::chrono::steady_clock::now();
    context.invoked = true;
    context.result = {};
    auto* scratch = context.scratch;
    const auto complete = [&](bool success) {
        context.elapsed_nanoseconds = ::strata::elapsed_nanoseconds(started);
        return success;
    };
    const auto fail = [&](std::string message) {
        context.result.errors.push_back(std::move(message));
        return complete(false);
    };
    if (context.transaction == nullptr || scratch == nullptr ||
        context.layer >= kLayers || context.rank >= kDsv4RankLocalWorld ||
        view.query_rank.size() != kQueryRank ||
        view.key_value.size() != kHeadDim) {
        return fail("rank-local deferred page-patch shape is invalid");
    }

    const auto& state = attention_state[context.layer];
    const auto compressor_elements = static_cast<std::size_t>(
        state.compressor.coefficient) * state.compressor.head_dim;
    const auto index_elements = static_cast<std::size_t>(
        state.indexer_compressor.coefficient) *
        state.indexer_compressor.head_dim;
    if (view.compressor_values.size() != compressor_elements ||
        view.compressor_scores.size() != compressor_elements ||
        view.index_compressor_values.size() != index_elements ||
        view.index_compressor_scores.size() != index_elements ||
        view.page_patches.size() != scratch->replica_patch.size()) {
        return fail("rank-local canonical page-patch payload is invalid");
    }

    scratch->key_value.resize(kHeadDim);
    for (std::size_t index = 0U; index < kHeadDim; ++index) {
        scratch->key_value[index] = std::bit_cast<float>(
            static_cast<std::uint32_t>(view.key_value[index]) << 16U);
    }

    const auto prefix = layer_prefix(context.layer) + "attn.";
    context.result = compress_state(
        context.layer, attention_state[context.layer].compressor,
        prefix + "compressor.", {}, context.position,
        attention_state[context.layer].frequencies,
        view.compressor_values, view.compressor_scores, nullptr, {},
        &scratch->compressed_row);
    if (context.result.ok()) {
        context.result = compress_state(
            context.layer, attention_state[context.layer].indexer_compressor,
            prefix + "indexer.compressor.", {}, context.position,
            attention_state[context.layer].frequencies,
            view.index_compressor_values, view.index_compressor_scores,
            nullptr, {}, &scratch->index_row);
    }
    if (context.result.ok()) {
        std::array<std::span<std::byte>, kDsv4RankLocalWorld> patches{
            view.page_patches, scratch->replica_patch};
        context.result = context.transaction->commit_layer(
            context.layer, scratch->key_value, scratch->compressed_row,
            patches, scratch->index_row);
    }
    return complete(context.result.ok());
}

bool DeepSeekV4Runtime::Impl::complete_physical_attention_prepare(
    PhysicalAttentionContext& context,
    const CudaDsv4AttentionPrepareHostView& view) {
    context.invoked = true;
    context.result = {};
    if (context.layer >= kLayers || view.key_value.size() != kHeadDim ||
        !context.sliding_append.has_value()) {
        context.result.errors.emplace_back(
            "DeepSeek deferred attention preparation shape is invalid");
        return false;
    }
    const auto decode = [](std::span<const std::uint16_t> source,
                           std::span<float> destination) {
        for (std::size_t index = 0U; index < source.size(); ++index) {
            destination[index] = std::bit_cast<float>(
                static_cast<std::uint32_t>(source[index]) << 16U);
        }
    };
    decode(view.key_value, context.key_value);
    std::size_t patch_cursor = 0U;
    const auto commit = [&](Dsv4KvPhysicalAppend& append,
                            std::span<const float> values) {
        const auto bytes = static_cast<std::size_t>(append.patch_bytes());
        if (bytes > view.page_patches.size() - patch_cursor) {
            context.result.errors.emplace_back(
                "DeepSeek deferred page-patch staging is truncated");
            return false;
        }
        auto committed = append.commit(
            values, view.page_patches.subspan(patch_cursor, bytes));
        patch_cursor += bytes;
        if (!committed.ok()) {
            append_errors(context.result, std::move(committed.errors));
            return false;
        }
        return true;
    };
    if (!commit(*context.sliding_append, context.key_value)) return false;

    auto& layer_state = attention_state[context.layer];
    auto* compressed_append = context.compressed_append.has_value()
        ? &*context.compressed_append : nullptr;
    std::span<std::byte> compressed_patch;
    if (compressed_append != nullptr) {
        const auto bytes = static_cast<std::size_t>(
            compressed_append->patch_bytes());
        if (bytes > view.page_patches.size() - patch_cursor) {
            context.result.errors.emplace_back(
                "DeepSeek deferred compressed-page staging is truncated");
            return false;
        }
        compressed_patch = view.page_patches.subspan(patch_cursor, bytes);
        patch_cursor += bytes;
    }
    context.result = compress_state(
        context.layer, layer_state.compressor,
        layer_prefix(context.layer) + "attn.compressor.", {},
        context.position, layer_state.frequencies,
        view.compressor_values, view.compressor_scores,
        compressed_append, compressed_patch);
    if (!context.result.ok()) return false;
    if (layer_state.indexer_compressor.ratio != 0U) {
        // The learned-index row is committed here, in the same stream order as
        // the sliding and compressed rows, so index_select() later performs
        // selection only and must not append it a second time.
        auto* index_append = context.index_append.has_value()
            ? &*context.index_append : nullptr;
        std::span<std::byte> index_patch;
        if (index_append != nullptr) {
            const auto bytes = static_cast<std::size_t>(
                index_append->patch_bytes());
            if (bytes > view.page_patches.size() - patch_cursor) {
                context.result.errors.emplace_back(
                    "DeepSeek deferred learned-index staging is truncated");
                return false;
            }
            index_patch = view.page_patches.subspan(patch_cursor, bytes);
            patch_cursor += bytes;
        }
        context.result = compress_state(
            context.layer, layer_state.indexer_compressor,
            layer_prefix(context.layer) + "attn.indexer.compressor.", {},
            context.position, layer_state.frequencies,
            view.index_compressor_values, view.index_compressor_scores,
            index_append, index_patch);
        if (!context.result.ok()) return false;
    } else if (!view.index_compressor_values.empty() ||
               !view.index_compressor_scores.empty() ||
               context.index_append.has_value()) {
        context.result.errors.emplace_back(
            "DeepSeek deferred sparse-index preparation was supplied for a "
            "layer whose indexer is not admitted");
        return false;
    }
    if (patch_cursor != view.page_patches.size()) {
        context.result.errors.emplace_back(
            "DeepSeek deferred page-patch staging has unused bytes");
        return false;
    }
    return true;
}

ValidationResult DeepSeekV4Runtime::Impl::compress_state(
    std::uint32_t layer, CompressorState& state, const std::string& prefix,
    std::span<const float> input, std::uint32_t position,
    std::span<const float> frequencies,
    std::span<const float> prepared_values,
    std::span<const float> prepared_scores,
    Dsv4KvPhysicalAppend* prepared_append,
    std::span<std::byte> prepared_patch,
    std::vector<float>* pooled_row) {
    ValidationResult result;
    if (pooled_row != nullptr) pooled_row->clear();
    if (state.ratio == 0U) return result;
    const auto dimensions = static_cast<std::size_t>(state.coefficient) *
                            state.head_dim;
    std::vector<float> values(dimensions);
    std::vector<float> scores(dimensions);
    if (!prepared_values.empty() || !prepared_scores.empty()) {
        if (prepared_values.size() != dimensions ||
            prepared_scores.size() != dimensions) {
            result.errors.emplace_back(
                "DeepSeek prepared compressor spans have incompatible sizes");
            return result;
        }
        std::copy(prepared_values.begin(), prepared_values.end(), values.begin());
        std::copy(prepared_scores.begin(), prepared_scores.end(), scores.begin());
    } else {
        const auto slot = layer_device(layer);
        result = linear(slot, prefix + "wkv", dimensions, kHidden, input,
                        values, false);
        if (!result.ok()) return result;
        result = linear(slot, prefix + "wgate", dimensions, kHidden, input,
                        scores, false);
        if (!result.ok()) return result;
    }
    auto ape = host_tensor(prefix + "ape",
                           static_cast<std::uint64_t>(state.ratio) * dimensions);
    if (!ape.ok()) {
        append_errors(result, std::move(ape.errors));
        return result;
    }
    const auto phase = position % state.ratio;
    const auto row = state.coefficient == 2U ? state.ratio + phase : phase;
    const auto row_offset = static_cast<std::size_t>(row) * dimensions;
    const auto ape_offset = static_cast<std::size_t>(phase) * dimensions;
    for (std::size_t dimension = 0U; dimension < dimensions; ++dimension) {
        state.values[row_offset + dimension] = values[dimension];
        state.scores[row_offset + dimension] =
            scores[dimension] + (*ape.value)[ape_offset + dimension];
    }
    if ((position + 1U) % state.ratio != 0U) return result;

    std::vector<float> pooled(state.head_dim, 0.0F);
    for (std::uint32_t dimension = 0U; dimension < state.head_dim; ++dimension) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::uint32_t candidate = 0U;
             candidate < state.coefficient * state.ratio; ++candidate) {
            std::size_t index = 0U;
            if (state.coefficient == 2U) {
                const auto source_row = candidate < state.ratio ? candidate : candidate;
                const auto source_dimension = candidate < state.ratio ? dimension :
                    static_cast<std::uint32_t>(state.head_dim + dimension);
                index = static_cast<std::size_t>(source_row) * dimensions +
                        source_dimension;
            } else {
                index = static_cast<std::size_t>(candidate) * dimensions + dimension;
            }
            maximum = std::max(maximum, state.scores[index]);
        }
        double denominator = 0.0;
        double numerator = 0.0;
        for (std::uint32_t candidate = 0U;
             candidate < state.coefficient * state.ratio; ++candidate) {
            std::size_t index = 0U;
            if (state.coefficient == 2U) {
                const auto source_dimension = candidate < state.ratio ? dimension :
                    static_cast<std::uint32_t>(state.head_dim + dimension);
                index = static_cast<std::size_t>(candidate) * dimensions +
                        source_dimension;
            } else {
                index = static_cast<std::size_t>(candidate) * dimensions + dimension;
            }
            const double weight = std::exp(
                static_cast<double>(state.scores[index] - maximum));
            denominator += weight;
            numerator += weight * static_cast<double>(state.values[index]);
        }
        pooled[dimension] = static_cast<float>(numerator / denominator);
    }
    if (state.coefficient == 2U) {
        const auto block_bytes = static_cast<std::size_t>(state.ratio) * dimensions;
        std::copy_n(state.values.begin() + block_bytes, block_bytes,
                    state.values.begin());
        std::copy_n(state.scores.begin() + block_bytes, block_bytes,
                    state.scores.begin());
    }
    result = norm(pooled, pooled, prefix + "norm.weight");
    if (!result.ok()) return result;
    apply_rope(std::span<float>(pooled).last(kRopeDim),
               position + 1U - state.ratio, frequencies);
    round_bf16(pooled);
    if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
        // The live physical page encoder owns the accepted BF16-boundary,
        // power-of-two scale, and half-up E4M3 conversion.
    } else if (state.rotate_fp4) {
        result = dsv4_hadamard_rotate_f32(pooled);
        if (!result.ok()) return result;
        result = dsv4_fp4_e2m1_simulate_f32(pooled, 32U);
        if (!result.ok()) return result;
    } else {
        quantize_activation_in_place(
            std::span<float>(pooled).first(state.head_dim - kRopeDim), 64U);
    }
    const auto compressed_row = position / state.ratio;
    if (pooled_row != nullptr) {
        // The caller owns publication. Returning here keeps the accumulator
        // advance and the row encoding in one place while letting the
        // rank-local path lay the same bytes into both ranks' pages.
        *pooled_row = std::move(pooled);
        return result;
    }
    if (prepared_append != nullptr) {
        result = prepared_append->commit(pooled, prepared_patch);
        if (!result.ok()) return result;
    } else if (kv_cache != nullptr) {
        result = kv_cache->append(active_sequence, state.kind, layer,
                                  state.ratio, compressed_row, pooled);
        if (!result.ok()) return result;
    } else {
        auto destination = state.compressed.writable_row(compressed_row);
        if (destination.size() != pooled.size()) {
            result.errors.emplace_back("DeepSeek compressed cache allocation failed");
            return result;
        }
        std::copy(pooled.begin(), pooled.end(), destination.begin());
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::index_positions(
    std::uint32_t layer, std::span<const float> input,
    std::span<const float> query_rank, std::uint32_t position,
    std::vector<std::uint32_t>& selected,
    std::span<const float> prepared_values,
    std::span<const float> prepared_scores, bool device_prepared_source) {
    ValidationResult result;
    selected.clear();
    auto& compressor_state = attention_state[layer].indexer_compressor;
    if (compressor_state.ratio == 0U) {
        result.errors.emplace_back(
            "DeepSeek sparse indexer was not admitted for a long context");
        return result;
    }
    result = compress_state(
        layer, compressor_state,
        layer_prefix(layer) + "attn.indexer.compressor.", input, position,
        attention_state[layer].frequencies, prepared_values, prepared_scores);
    if (!result.ok()) return result;
    return index_select(layer, input, query_rank, position, selected,
                        device_prepared_source);
}

ValidationResult DeepSeekV4Runtime::Impl::index_select(
    std::uint32_t layer, std::span<const float> input,
    std::span<const float> query_rank, std::uint32_t position,
    std::vector<std::uint32_t>& selected, bool device_prepared_source) {
    ValidationResult result;
    selected.clear();
    const auto record_selection = [&] {
        auto hash = diagnostics.index_selection_trace_hash;
        hash = diagnostic_hash_u32(hash, layer);
        hash = diagnostic_hash_u32(hash, position);
        hash = diagnostic_hash_u32(
            hash, static_cast<std::uint32_t>(selected.size()));
        for (const auto selected_position : selected) {
            hash = diagnostic_hash_u32(hash, selected_position);
        }
        diagnostics.index_selection_trace_hash = hash;
        ++diagnostics.index_selection_count;
    };
    const auto& state = attention_state[layer].indexer_compressor;
    if (state.ratio == 0U) {
        result.errors.emplace_back(
            "DeepSeek sparse indexer was not admitted for a long context");
        return result;
    }
    const auto prefix = layer_prefix(layer) + "attn.indexer.";

    const auto compressed_count = (position + 1U) / state.ratio;
    if (compressed_count <= kIndexTopK) {
        selected.resize(compressed_count);
        std::iota(selected.begin(), selected.end(), 0U);
        record_selection();
        return result;
    }
    ++graph_stats.attention_index_queries;
    graph_stats.attention_index_candidates += compressed_count;
    graph_stats.attention_index_selected += kIndexTopK;

    const auto slot = layer_device(layer);
    std::vector<float> queries(
        static_cast<std::size_t>(kIndexHeads) * kIndexHeadDim);
    std::vector<float> index_weights(kIndexHeads);
    constexpr float index_scale = kIndexQueryScale;
    const auto& frequencies = attention_state[layer].frequencies;
    if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice &&
        device_prepared_source) {
        // Both projections run on the device against the preparation command's
        // own activations, so neither the query rank nor the layer input is
        // sent back across the bus. The rotation angles are still evaluated
        // here rather than in the kernel because host libm and device
        // trigonometry differ in the last ulp, and selection is a hard top-k
        // where that becomes a different candidate set. They depend only on the
        // position and the layer frequencies, so they are knowable before the
        // call -- which is also what will let this run inside the queued chain.
        if (frequencies.size() < kRopeDim / 2U) {
            result.errors.emplace_back(
                "DeepSeek index rope frequencies are too short");
            return result;
        }
        index_rope_cosines.resize(kRopeDim / 2U);
        index_rope_sines.resize(kRopeDim / 2U);
        for (std::size_t pair = 0U; pair < index_rope_cosines.size(); ++pair) {
            const float angle =
                static_cast<float>(position) * frequencies[pair];
            index_rope_cosines[pair] = std::cos(angle);
            index_rope_sines[pair] = std::sin(angle);
        }
        auto cuda_demand = weights->demand();
        Dsv4WeightCache::Lease query_projection;
        Dsv4WeightCache::Lease weight_projection;
        result = weights->acquire(slot, prefix + "wq_b",
                                  kIndexHeads * kIndexHeadDim, kQueryRank,
                                  query_projection);
        if (!result.ok()) return result;
        result = weights->acquire(slot, prefix + "weights_proj", kIndexHeads,
                                  kHidden, weight_projection);
        if (!result.ok()) return result;
        CudaDsv4IndexProjectionRequest projection;
        projection.query_projection = &query_projection.weight();
        projection.weight_projection = &weight_projection.weight();
        projection.rope_cosines = index_rope_cosines;
        projection.rope_sines = index_rope_sines;
        projection.heads = kIndexHeads;
        projection.head_dim = kIndexHeadDim;
        projection.rope_dim = kRopeDim;
        projection.weight_scale = index_scale;
        result = cuda.dsv4_index_projections(devices[slot], projection, queries,
                                             index_weights);
        if (!result.ok()) return result;
    } else {
        result = linear(slot, prefix + "wq_b", kIndexHeads * kIndexHeadDim,
                        kQueryRank, query_rank, queries);
        if (!result.ok()) return result;
        if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
            // The batched prefill page projects many rows before any of them
            // reaches selection, so it leaves no per-row preparation state on
            // the device. Its query still crosses the same rotation,
            // rounding and half-up quantization, one uploaded row at a time.
            if (frequencies.size() < kRopeDim / 2U) {
                result.errors.emplace_back(
                    "DeepSeek index rope frequencies are too short");
                return result;
            }
            index_rope_cosines.resize(kRopeDim / 2U);
            index_rope_sines.resize(kRopeDim / 2U);
            for (std::size_t pair = 0U; pair < index_rope_cosines.size();
                 ++pair) {
                const float angle =
                    static_cast<float>(position) * frequencies[pair];
                index_rope_cosines[pair] = std::cos(angle);
                index_rope_sines[pair] = std::sin(angle);
            }
            result = cuda.dsv4_index_query_rope_quantize(
                devices[slot], queries, index_rope_cosines, index_rope_sines,
                kIndexHeads, kIndexHeadDim, kRopeDim, true);
            if (!result.ok()) return result;
        } else {
            for (std::uint32_t head = 0U; head < kIndexHeads; ++head) {
                auto query = std::span<float>(queries).subspan(
                    static_cast<std::size_t>(head) * kIndexHeadDim,
                    kIndexHeadDim);
                apply_rope(query.last(kRopeDim), position, frequencies);
                round_bf16(query.last(kRopeDim));
                if (!config.enable_gpu_lightning_indexer) {
                    result = dsv4_hadamard_rotate_f32(query);
                    if (!result.ok()) return result;
                    result = dsv4_fp4_e2m1_simulate_f32(query, 32U);
                    if (!result.ok()) return result;
                }
            }
        }
        result = linear(slot, prefix + "weights_proj", kIndexHeads, kHidden,
                        input, index_weights);
        if (!result.ok()) return result;
        for (auto& weight : index_weights) {
            weight = round_bf16(weight * index_scale);
        }
    }

    if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
        // Physical KV keeps the learned index as E4M3 rows with one f32 scale
        // each, which the FP4 Lightning Indexer above cannot read. The scalar
        // path below is exact but scores every candidate on the host: at the
        // declared 1,048,576-token context that is 262,144 candidates over 64
        // heads of 128 dimensions per layer, 1.85e11 FLOP per token across the
        // 21 ratio-4 layers, which no CPU budget reaches. Device selection is
        // therefore the only viable path here, not an optimization of one.
        auto cuda_demand = weights->demand();
        auto& blocks = physical_index_blocks;
        result = kv_cache->block_table_into(
            active_sequence, Dsv4KvBlockKind::LearnedIndex, layer, blocks);
        if (!result.ok()) return result;
        std::vector<Dsv4KvDeviceLease> leases;
        std::vector<CudaDsv4PhysicalIndexPage> pages;
        try {
            leases.reserve(blocks.size());
            pages.reserve(blocks.size());
        } catch (const std::bad_alloc&) {
            result.errors.emplace_back(
                "cannot allocate physical Lightning Indexer page metadata");
            return result;
        }
        std::uint32_t remaining = compressed_count;
        for (const auto& block : blocks) {
            if (remaining == 0U) break;
            const auto logical_row = block.logical_begin /
                                     block.compression_ratio;
            auto lease = kv_cache->acquire_device(
                active_sequence, Dsv4KvBlockKind::LearnedIndex, layer,
                logical_row, slot);
            if (!lease.ok()) {
                append_errors(
                    result, std::move(lease.errors),
                    "DeepSeek physical index page lease layer " +
                        std::to_string(layer) + " logical row " +
                        std::to_string(logical_row));
                return result;
            }
            const auto rows = std::min(remaining, block.used_rows);
            leases.push_back(std::move(lease.value));
            // A physical device lease holds the block-major payload alone;
            // acquire_device strips the header on upload.
            pages.push_back(CudaDsv4PhysicalIndexPage{
                leases.back().buffer(), 0U, block.capacity_rows, rows});
            remaining -= rows;
        }
        if (remaining != 0U) {
            result.errors.emplace_back(
                "physical Lightning Indexer device history is incomplete");
            return result;
        }
        selected.resize(kIndexTopK);
        CudaDsv4PhysicalIndexRequest request;
        request.queries = queries;
        request.weights = index_weights;
        request.pages = pages;
        request.heads = kIndexHeads;
        request.head_dim = kIndexHeadDim;
        request.top_k = kIndexTopK;
        result = cuda.dsv4_physical_lightning_index(
            devices[slot], request, selected);
        if (!result.ok()) {
            selected.clear();
            return result;
        }
        ++graph_stats.attention_index_cuda_dispatches;
        record_selection();
        return result;
    }

    if (config.enable_gpu_lightning_indexer) {
        auto cuda_demand = weights->demand();
        std::vector<CudaLightningIndexSegment> segments;
        std::vector<Dsv4KvDeviceLease> leases;
        const bool device_resident =
            !config.device_kv_cache_bytes.empty() &&
            config.device_kv_cache_bytes[slot] != 0U;
        if (device_resident) {
            auto& blocks = physical_index_blocks;
            result = kv_cache->block_table_into(
                active_sequence, Dsv4KvBlockKind::LearnedIndex, layer,
                blocks);
            if (!result.ok()) return result;
            try {
                segments.reserve(blocks.size());
                leases.reserve(blocks.size());
            } catch (const std::bad_alloc&) {
                result.errors.emplace_back(
                    "cannot allocate Lightning Indexer block metadata");
                return result;
            }
            std::uint32_t remaining = compressed_count;
            for (const auto& block : blocks) {
                if (remaining == 0U) break;
                const auto logical_row = block.logical_begin /
                                         block.compression_ratio;
                auto lease = kv_cache->acquire_device(
                    active_sequence, Dsv4KvBlockKind::LearnedIndex,
                    layer, logical_row, slot);
                if (!lease.ok()) {
                    append_errors(result, std::move(lease.errors));
                    return result;
                }
                const auto rows = std::min(remaining, block.used_rows);
                leases.push_back(std::move(lease.value));
                segments.push_back(CudaLightningIndexSegment{
                    leases.back().buffer(), {}, kDsv4KvBlockHeaderBytes, rows});
                remaining -= rows;
            }
            if (remaining != 0U) {
                result.errors.emplace_back(
                    "Lightning Indexer device history is incomplete");
                return result;
            }
        } else {
            auto compact = kv_cache->learned_index_segments(
                active_sequence, layer, compressed_count);
            if (!compact.ok()) {
                append_errors(result, std::move(compact.errors));
                return result;
            }
            segments = std::move(compact.value);
        }
        selected.resize(kIndexTopK);
        CudaLightningIndexRequest request;
        request.queries = queries;
        request.weights = index_weights;
        request.segments = segments;
        request.heads = kIndexHeads;
        request.head_dim = kIndexHeadDim;
        request.top_k = kIndexTopK;
        result = cuda.lightning_index(devices[slot], request, selected);
        if (!result.ok()) {
            selected.clear();
            return result;
        }
        ++graph_stats.attention_index_cuda_dispatches;
        record_selection();
        return result;
    }

    std::vector<float> scores(compressed_count);
    const auto score_key = [&](std::size_t row, std::span<const float> key) {
        if (key.size() != kIndexHeadDim) {
            scores[row] = -std::numeric_limits<float>::infinity();
            return;
        }
        auto destination = std::span<float>(scores).subspan(row, 1U);
        const auto scored = dsv4_index_scores_f32(
            destination, queries, key, index_weights, kIndexHeads,
            kIndexHeadDim);
        if (!scored.ok()) scores[row] = -std::numeric_limits<float>::infinity();
    };
    const auto score_row = [&](std::size_t row) {
        score_key(row, state.compressed.row(row));
    };
    if (kv_cache != nullptr) {
        for (std::uint32_t row = 0U; row < compressed_count; ++row) {
            auto key = kv_row(layer, Dsv4KvBlockKind::LearnedIndex, row);
            if (!key.ok()) {
                append_errors(result, std::move(key.errors));
                return result;
            }
            score_key(row, key.value);
        }
    } else if (attention_workers != nullptr && attention_workers->size() > 1U) {
        const auto workers = std::min<std::size_t>(
            attention_workers->size(), compressed_count);
        result = attention_workers->parallel_for(
            workers, [&](std::size_t worker) {
                const auto begin = compressed_count * worker / workers;
                const auto end = compressed_count * (worker + 1U) / workers;
                for (std::size_t row = begin; row < end; ++row) {
                    score_row(row);
                }
            });
        if (!result.ok()) return result;
    } else {
        for (std::size_t row = 0U; row < compressed_count; ++row) {
            score_row(row);
        }
    }

    auto topk = dsv4_index_topk_f32(scores, kIndexTopK);
    if (!topk.ok()) {
        append_errors(result, std::move(topk.errors));
        return result;
    }
    selected = std::move(topk.positions);
    ++graph_stats.attention_index_scalar_dispatches;
    record_selection();
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::attention(
    std::uint32_t layer, std::span<const float> input, std::uint32_t position,
    std::span<float> output) {
    ValidationResult result;
    if (input.size() != kHidden ||
        (output.size() != kHidden &&
         !(config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice &&
           output.empty()))) {
        result.errors.emplace_back("DeepSeek attention spans have incompatible sizes");
        return result;
    }
    const auto slot = layer_device(layer);
    const auto prefix = layer_prefix(layer) + "attn.";
    if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
        auto cuda_demand = weights->demand();
        // The learned-index *row* can now be reserved and committed in stream
        // order with the sliding and compressed rows (see the index_append
        // handling below and in complete_physical_attention_prepare). What
        // still excludes the sparse indexer from the queued path is
        // *selection*, not the append: index_select() projects index queries
        // through wq_b and weights_proj, and linear() dispatches those to
        // CUDA. Selection can only run once query_rank exists host-side, which
        // in the queued path is inside the CUDA host callback -- where calling
        // a CUDA API is not permitted.
        //
        // Until index query projection, scoring and top-k are moved onto the
        // device inside the queued chain, this path must stay closed: enabling
        // it would reach physical_paged_attention with sparse == true and an
        // empty indexed_positions, which silently attends zero compressed
        // rows. Fail closed instead.
        const bool deferred_page_update =
            !config.enable_layer_hash_trace && kv_cache != nullptr &&
            attention_state[layer].indexer_compressor.ratio == 0U;
        Dsv4WeightCache::Lease query_a;
        Dsv4WeightCache::Lease query_b;
        Dsv4WeightCache::Lease key_value_weight;
        Dsv4WeightCache::Lease compressor_value_weight;
        Dsv4WeightCache::Lease compressor_gate_weight;
        Dsv4WeightCache::Lease index_compressor_value_weight;
        Dsv4WeightCache::Lease index_compressor_gate_weight;
        result = weights->acquire(
            slot, prefix + "wq_a", kQueryRank, kHidden, query_a);
        if (!result.ok()) return result;
        result = weights->acquire(
            slot, prefix + "wq_b", kHeads * kHeadDim, kQueryRank,
            query_b);
        if (!result.ok()) return result;
        result = weights->acquire(
            slot, prefix + "wkv", kHeadDim, kHidden, key_value_weight);
        if (!result.ok()) return result;
        const auto compressor_dimensions =
            static_cast<std::size_t>(
                attention_state[layer].compressor.coefficient) *
            attention_state[layer].compressor.head_dim;
        std::vector<float> compressor_values;
        std::vector<float> compressor_scores;
        if (attention_state[layer].compressor.ratio != 0U) {
            result = weights->acquire(
                slot, prefix + "compressor.wkv", compressor_dimensions,
                kHidden, compressor_value_weight);
            if (!result.ok()) return result;
            result = weights->acquire(
                slot, prefix + "compressor.wgate", compressor_dimensions,
                kHidden, compressor_gate_weight);
            if (!result.ok()) return result;
            compressor_values.resize(compressor_dimensions);
            compressor_scores.resize(compressor_dimensions);
        }
        const auto index_compressor_dimensions =
            static_cast<std::size_t>(
                attention_state[layer].indexer_compressor.coefficient) *
            attention_state[layer].indexer_compressor.head_dim;
        std::vector<float> index_compressor_values;
        std::vector<float> index_compressor_scores;
        if (attention_state[layer].indexer_compressor.ratio != 0U) {
            result = weights->acquire(
                slot, prefix + "indexer.compressor.wkv",
                index_compressor_dimensions, kHidden,
                index_compressor_value_weight);
            if (!result.ok()) return result;
            result = weights->acquire(
                slot, prefix + "indexer.compressor.wgate",
                index_compressor_dimensions, kHidden,
                index_compressor_gate_weight);
            if (!result.ok()) return result;
            index_compressor_values.resize(index_compressor_dimensions);
            index_compressor_scores.resize(index_compressor_dimensions);
        }
        auto query_norm = host_tensor(prefix + "q_norm.weight", kQueryRank);
        if (!query_norm.ok()) {
            append_errors(result, std::move(query_norm.errors));
            return result;
        }
        auto key_value_norm = host_tensor(
            prefix + "kv_norm.weight", kHeadDim);
        if (!key_value_norm.ok()) {
            append_errors(result, std::move(key_value_norm.errors));
            return result;
        }
        std::array<float, kRopeDim / 2U> cosines{};
        std::array<float, kRopeDim / 2U> sines{};
        for (std::size_t index = 0U; index < cosines.size(); ++index) {
            const float angle = static_cast<float>(position) *
                                attention_state[layer].frequencies[index];
            cosines[index] = std::cos(angle);
            sines[index] = std::sin(angle);
        }
        std::vector<float> query_rank(kQueryRank);
        std::vector<float> kv(kHeadDim);
        auto& deferred_context = physical_attention_context(
            host_moe_chain_row.value_or(layer));
        std::vector<CudaDsv4AttentionPageWrite> page_writes;
        if (deferred_page_update) {
            deferred_context.owner = this;
            deferred_context.layer = layer;
            deferred_context.position = position;
            deferred_context.result = {};
            deferred_context.invoked = false;
            deferred_context.key_value.resize(kHeadDim);
            deferred_context.sliding_append.reset();
            deferred_context.compressed_append.reset();
            deferred_context.index_append.reset();
            auto sliding = kv_cache->reserve_physical_append(
                active_sequence, Dsv4KvBlockKind::Sliding, layer, 1U,
                position, slot);
            if (!sliding.ok()) {
                append_errors(result, std::move(sliding.errors));
                return result;
            }
            deferred_context.sliding_append.emplace(
                std::move(sliding.value));
            const auto ratio = attention_state[layer].compressor.ratio;
            if (ratio != 0U && (position + 1U) % ratio == 0U) {
                auto compressed = kv_cache->reserve_physical_append(
                    active_sequence,
                    attention_state[layer].compressor.kind, layer, ratio,
                    position / ratio, slot);
                if (!compressed.ok()) {
                    append_errors(result, std::move(compressed.errors));
                    return result;
                }
                deferred_context.compressed_append.emplace(
                    std::move(compressed.value));
            }
            const auto add_write = [&](const Dsv4KvPhysicalAppend& append) {
                page_writes.push_back(CudaDsv4AttentionPageWrite{
                    append.buffer(), append.data_offset(),
                    append.scale_offset(), append.data_bytes(),
                    append.scale_bytes()});
            };
            const auto index_ratio =
                attention_state[layer].indexer_compressor.ratio;
            if (index_ratio != 0U && (position + 1U) % index_ratio == 0U) {
                auto index = kv_cache->reserve_physical_append(
                    active_sequence,
                    attention_state[layer].indexer_compressor.kind, layer,
                    index_ratio, position / index_ratio, slot);
                if (!index.ok()) {
                    append_errors(result, std::move(index.errors));
                    return result;
                }
                deferred_context.index_append.emplace(std::move(index.value));
            }
            add_write(*deferred_context.sliding_append);
            if (deferred_context.compressed_append.has_value()) {
                add_write(*deferred_context.compressed_append);
            }
            if (deferred_context.index_append.has_value()) {
                add_write(*deferred_context.index_append);
            }
        }
        CudaDsv4AttentionPrepareRequest request;
        request.query_a = &query_a.weight();
        request.query_b = &query_b.weight();
        request.key_value = &key_value_weight.weight();
        if (pending_mhc_attention_transition) {
            request.mhc_transition = &device_mhc_weights[layer][0U];
        }
        if (!compressor_values.empty()) {
            request.compressor_value = &compressor_value_weight.weight();
            request.compressor_gate = &compressor_gate_weight.weight();
        }
        if (!index_compressor_values.empty()) {
            request.index_compressor_value =
                &index_compressor_value_weight.weight();
            request.index_compressor_gate =
                &index_compressor_gate_weight.weight();
        }
        request.query_norm = *query_norm.value;
        request.key_value_norm = *key_value_norm.value;
        request.rope_cosines = cosines;
        request.rope_sines = sines;
        request.mhc_device = devices[mhc_slot];
        if (devices[slot] != devices[mhc_slot] &&
            !pending_mhc_attention_transition &&
            (layer != 0U || config.enable_layer_hash_trace)) {
            request.cross_device_input = input;
        }
        if (deferred_page_update) {
            request.host_callback = physical_attention_prepare_callback;
            request.host_callback_context = &deferred_context;
            request.page_writes = page_writes;
        }
        request.maximum_workspace_bytes = 1ULL << 20U;
        auto subphase_started = std::chrono::steady_clock::now();
        const auto prepared_compressor_matmuls =
            static_cast<std::uint64_t>(!compressor_values.empty()) * 2U +
            static_cast<std::uint64_t>(!index_compressor_values.empty()) * 2U;
        graph_stats.attention_projection_matmul_calls +=
            3U + prepared_compressor_matmuls;
        graph_stats.attention_projection_matmul_rows +=
            3U + prepared_compressor_matmuls;
        result = cuda.dsv4_prepare_attention(
            devices[slot], request, query_rank, kv,
            compressor_values, compressor_scores,
            index_compressor_values, index_compressor_scores);
        graph_stats.attention_query_nanoseconds +=
            elapsed_nanoseconds(subphase_started);
        if (!result.ok()) return result;
        pending_mhc_attention_transition = false;
        if (deferred_page_update) {
            pending_attention_weights.push_back(std::move(query_a));
            pending_attention_weights.push_back(std::move(query_b));
            pending_attention_weights.push_back(
                std::move(key_value_weight));
            if (!compressor_values.empty()) {
                pending_attention_weights.push_back(
                    std::move(compressor_value_weight));
                pending_attention_weights.push_back(
                    std::move(compressor_gate_weight));
            }
            if (!index_compressor_values.empty()) {
                pending_attention_weights.push_back(
                    std::move(index_compressor_value_weight));
                pending_attention_weights.push_back(
                    std::move(index_compressor_gate_weight));
            }
            auto sink = host_tensor(prefix + "attn_sink", kHeads);
            if (!sink.ok()) {
                append_errors(result, std::move(sink.errors));
                return result;
            }
            return physical_paged_attention(
                layer, {}, *sink.value, position, {}, output);
        }
        std::span<const float> resident_queries;
        // The preparation command above ran on this layer's device and its
        // activations are still the most recent there, so selection may
        // project from them.
        return attention_prepared(
            layer, input, query_rank, resident_queries, kv, position, output,
            compressor_values, compressor_scores,
            index_compressor_values, index_compressor_scores, true);
    }
    auto subphase_started = std::chrono::steady_clock::now();
    std::vector<float> query_rank(kQueryRank);
    ++graph_stats.attention_projection_matmul_calls;
    ++graph_stats.attention_projection_matmul_rows;
    result = linear(slot, prefix + "wq_a", kQueryRank, kHidden, input, query_rank);
    if (!result.ok()) return result;
    result = norm(query_rank, query_rank, prefix + "q_norm.weight");
    if (!result.ok()) return result;
    std::vector<float> queries(static_cast<std::size_t>(kHeads) * kHeadDim);
    ++graph_stats.attention_projection_matmul_calls;
    ++graph_stats.attention_projection_matmul_rows;
    result = linear(slot, prefix + "wq_b", kHeads * kHeadDim, kQueryRank,
                    query_rank, queries);
    if (!result.ok()) return result;
    const auto normalize_query = [&](std::uint32_t head) {
        auto query = std::span<float>(queries).subspan(
            static_cast<std::size_t>(head) * kHeadDim, kHeadDim);
        double square_sum = 0.0;
        for (const float value : query) {
            square_sum += static_cast<double>(value) * value;
        }
        const float reciprocal = 1.0F / std::sqrt(
            static_cast<float>(square_sum / kHeadDim) + kRmsEpsilon);
        for (auto& value : query) value = round_bf16(value * reciprocal);
        apply_rope(query.last(kRopeDim), position,
                   attention_state[layer].frequencies);
        round_bf16(query.last(kRopeDim));
    };
    if (attention_workers != nullptr) {
        result = attention_workers->parallel_for(
            kHeads, [&](std::size_t head) {
                normalize_query(static_cast<std::uint32_t>(head));
            });
        if (!result.ok()) return result;
    } else {
        for (std::uint32_t head = 0U; head < kHeads; ++head) {
            normalize_query(head);
        }
    }
    graph_stats.attention_query_nanoseconds += elapsed_nanoseconds(subphase_started);

    subphase_started = std::chrono::steady_clock::now();
    std::vector<float> kv(kHeadDim);
    ++graph_stats.attention_projection_matmul_calls;
    ++graph_stats.attention_projection_matmul_rows;
    result = linear(slot, prefix + "wkv", kHeadDim, kHidden, input, kv);
    if (!result.ok()) return result;
    result = norm(kv, kv, prefix + "kv_norm.weight");
    if (!result.ok()) return result;
    apply_rope(std::span<float>(kv).last(kRopeDim), position,
               attention_state[layer].frequencies);
    round_bf16(std::span<float>(kv).last(kRopeDim));
    if (config.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice) {
        quantize_activation_in_place(
            std::span<float>(kv).first(kHeadDim - kRopeDim), 64U);
    }
    graph_stats.attention_kv_nanoseconds += elapsed_nanoseconds(subphase_started);
    return attention_prepared(layer, input, query_rank, queries, kv, position,
                              output);
}

ValidationResult DeepSeekV4Runtime::Impl::physical_paged_attention(
    std::uint32_t layer, std::span<const float> queries,
    std::span<const float> sinks, std::uint32_t position,
    std::span<const std::uint32_t> indexed_positions,
    std::span<float> diagnostic_branch,
    PhysicalAttentionPageSet* page_set) {
    ValidationResult result;
    if (kv_cache == nullptr ||
        config.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice) {
        result.errors.emplace_back(
            "DeepSeek physical paged attention requires its physical cache");
        return result;
    }
    const auto slot = layer_device(layer);
    // Process-lifetime buffers: these tables reach 4,096 blocks at the declared
    // context and are rebuilt once per kind per layer per token, so they must
    // not allocate on the timed path.
    auto& sliding = physical_sliding_blocks;
    auto& compressed = physical_compressed_blocks;
    result = kv_cache->block_table_into(
        active_sequence, Dsv4KvBlockKind::Sliding, layer, sliding);
    if (!result.ok()) return result;
    const auto ratio = attention_state[layer].compressor.ratio;
    compressed.clear();
    if (ratio != 0U) {
        result = kv_cache->block_table_into(
            active_sequence, attention_state[layer].compressor.kind, layer,
            compressed);
        if (!result.ok()) return result;
    }

    std::uint64_t page_set_build_nanoseconds = 0U;
    const bool builds_page_set =
        page_set == nullptr || page_set->pages.empty();
    if (builds_page_set) ++graph_stats.attention_page_set_builds;
    const auto page_set_setup_started = std::chrono::steady_clock::now();
    std::vector<Dsv4KvDeviceLease> local_leases;
    std::vector<CudaDsv4PhysicalPage> local_pages;
    std::unordered_map<std::uint64_t, std::uint32_t> local_page_indices;
    auto& leases = page_set != nullptr ? page_set->leases : local_leases;
    auto& pages = page_set != nullptr ? page_set->pages : local_pages;
    auto& page_indices = page_set != nullptr
        ? page_set->page_indices : local_page_indices;
    if (page_set != nullptr && page_set->pages.empty()) {
        const auto reserve = sliding.size() + compressed.size();
        page_set->leases.reserve(reserve);
        page_set->pages.reserve(reserve);
        page_set->page_indices.reserve(reserve);
    }
    if (builds_page_set) {
        page_set_build_nanoseconds +=
            elapsed_nanoseconds(page_set_setup_started);
    }
    const auto locate = [&](Dsv4KvBlockKind kind,
                            const std::vector<Dsv4KvBlockInfo>& table,
                            std::uint32_t logical_row,
                            CudaDsv4AttentionCandidate& candidate) {
        const auto located = locate_physical_kv_block(table, logical_row);
        if (located == table.size()) {
            result.errors.emplace_back(
                "DeepSeek physical attention candidate page is unavailable");
            return;
        }
        const auto found = table.begin() + static_cast<std::ptrdiff_t>(located);
        const auto begin = found->logical_begin / found->compression_ratio;
        auto page = page_indices.find(found->id);
        if (page == page_indices.end()) {
            const auto page_build_started = std::chrono::steady_clock::now();
            auto lease = kv_cache->acquire_device(
                active_sequence, kind, layer, logical_row, slot);
            if (!lease.ok()) {
                append_errors(
                    result, std::move(lease.errors),
                    "DeepSeek physical attention page lease layer " +
                        std::to_string(layer) + " kind " +
                        std::to_string(static_cast<unsigned>(kind)) +
                        " logical row " + std::to_string(logical_row));
                return;
            }
            const auto index = static_cast<std::uint32_t>(pages.size());
            leases.push_back(std::move(lease.value));
            pages.push_back({leases.back().buffer(), found->capacity_rows});
            page = page_indices.emplace(found->id, index).first;
            ++graph_stats.attention_page_set_pages;
            page_set_build_nanoseconds +=
                elapsed_nanoseconds(page_build_started);
        }
        candidate.page = page->second;
        candidate.row = static_cast<std::uint32_t>(logical_row - begin);
        candidate.valid = true;
    };

    const auto compressed_count = ratio == 0U
        ? 0U : (position + 1U) / ratio;
    const bool sparse = ratio == 4U &&
        attention_state[layer].indexer_compressor.ratio == 4U;
    const auto compressed_width = ratio == 0U ? 0U : ratio == 4U
        ? kIndexTopK
        : ((std::max(1U, compressed_count) + 127U) / 128U) * 128U;
    constexpr std::uint32_t sliding_width = kWindow;
    std::vector<CudaDsv4AttentionCandidate> candidates(
        static_cast<std::size_t>(compressed_width) + sliding_width);
    const auto attended_compressed = sparse
        ? static_cast<std::uint32_t>(indexed_positions.size())
        : compressed_count;
    if (attended_compressed > compressed_width) {
        result.errors.emplace_back(
            "DeepSeek physical attention compressed candidates exceed their fixed region");
        return result;
    }
    const auto candidate_resolution_started = std::chrono::steady_clock::now();
    graph_stats.attention_candidate_resolutions +=
        static_cast<std::uint64_t>(attended_compressed);
    for (std::uint32_t item = 0U; item < attended_compressed; ++item) {
        const auto logical_row = sparse ? indexed_positions[item] : item;
        locate(attention_state[layer].compressor.kind, compressed,
               logical_row, candidates[item]);
        if (!result.ok()) return result;
    }
    const auto window_count = std::min(position + 1U, kWindow);
    graph_stats.attention_candidate_resolutions += window_count;
    for (std::uint32_t item = 0U; item < window_count; ++item) {
        const auto logical_row = position + 1U - window_count + item;
        locate(Dsv4KvBlockKind::Sliding, sliding, logical_row,
               candidates[static_cast<std::size_t>(compressed_width) + item]);
        if (!result.ok()) return result;
    }
    const auto candidate_resolution_nanoseconds =
        elapsed_nanoseconds(candidate_resolution_started);
    graph_stats.attention_page_set_build_nanoseconds +=
        page_set_build_nanoseconds;
    graph_stats.attention_candidate_resolution_nanoseconds +=
        candidate_resolution_nanoseconds > page_set_build_nanoseconds
            ? candidate_resolution_nanoseconds - page_set_build_nanoseconds
            : 0U;

    const auto prefix = layer_prefix(layer) + "attn.";
    const auto weight_started = std::chrono::steady_clock::now();
    Dsv4WeightCache::Lease output_a;
    Dsv4WeightCache::Lease output_b;
    Dsv4WeightCache::Lease router;
    result = weights->acquire(
        slot, prefix + "wo_a", kOutputGroups * kOutputRank,
        kHeads * kHeadDim / kOutputGroups, output_a);
    if (!result.ok()) return result;
    result = weights->acquire(
        slot, prefix + "wo_b", kHidden,
        kOutputGroups * kOutputRank, output_b);
    if (!result.ok()) return result;
    graph_stats.attention_page_weight_acquire_nanoseconds +=
        elapsed_nanoseconds(weight_started);
    std::array<float, kRopeDim / 2U> cosines{};
    std::array<float, kRopeDim / 2U> sines{};
    for (std::size_t index = 0U; index < cosines.size(); ++index) {
        const float angle = static_cast<float>(position) *
                            attention_state[layer].frequencies[index] * -1.0F;
        cosines[index] = std::cos(angle);
        sines[index] = std::sin(angle);
    }
    CudaDsv4PagedAttentionMhcRequest request;
    request.attention.queries = queries;
    request.attention.head_sinks = sinks;
    request.attention.pages = pages;
    request.attention.candidates = candidates;
    request.attention.scale = kAttentionScale;
    request.attention.maximum_workspace_bytes = 4ULL << 20U;
    request.inverse_rope_cosines = cosines;
    request.inverse_rope_sines = sines;
    request.output_a = &output_a.weight();
    request.output_b = &output_b.weight();
    request.mhc_device = devices[mhc_slot];
    const bool combine_mhc_transition = diagnostic_branch.empty();
    if (combine_mhc_transition) {
        result = weights->acquire(
            mhc_slot, layer_prefix(layer) + "ffn.gate", kExperts, kHidden,
            router);
        if (!result.ok()) return result;
        request.mhc_transition = &device_mhc_weights[layer][1U];
        request.router = &router.weight();
        request.defer_host_moe_input = true;
    }
    result = cuda.dsv4_paged_attention_to_mhc(
        devices[slot], request, diagnostic_branch);
    if (result.ok()) {
        completed_attention_mhc_transition = combine_mhc_transition;
        completed_router_projection = combine_mhc_transition;
        deferred_attention_moe_input = combine_mhc_transition;
        if (combine_mhc_transition) {
            pending_attention_leases.insert(
                pending_attention_leases.end(),
                std::make_move_iterator(leases.begin()),
                std::make_move_iterator(leases.end()));
            pending_attention_weights.push_back(std::move(output_a));
            pending_attention_weights.push_back(std::move(output_b));
            pending_attention_weights.push_back(std::move(router));
        }
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::physical_paged_attention_page(
    std::uint32_t layer, std::span<const float> input,
    std::span<const float> query_rank, std::span<const float> queries,
    std::span<const float> sinks, std::uint32_t position_base,
    std::span<const std::uint32_t> row_slots,
    std::span<float> diagnostic_branches) {
    ValidationResult result;
    const auto rows64 = input.size() / kHidden;
    const auto query_stride = static_cast<std::size_t>(kHeads) * kHeadDim;
    if (kv_cache == nullptr ||
        config.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice ||
        rows64 < 2U || rows64 > std::numeric_limits<std::uint32_t>::max() ||
        input.size() != rows64 * kHidden ||
        query_rank.size() != rows64 * kQueryRank ||
        queries.size() != rows64 * query_stride ||
        sinks.size() != kHeads || row_slots.size() != rows64 ||
        diagnostic_branches.size() != rows64 * kHidden) {
        result.errors.emplace_back(
            "DeepSeek physical attention page spans are incompatible");
        return result;
    }
    const auto rows = static_cast<std::uint32_t>(rows64);
    const auto slot = layer_device(layer);
    auto& layer_state = attention_state[layer];
    const auto ratio = layer_state.compressor.ratio;
    const auto last_position = position_base + rows - 1U;

    auto& sliding = physical_sliding_blocks;
    auto& compressed = physical_compressed_blocks;
    result = kv_cache->block_table_into(
        active_sequence, Dsv4KvBlockKind::Sliding, layer, sliding);
    if (!result.ok()) return result;
    compressed.clear();
    if (ratio != 0U) {
        result = kv_cache->block_table_into(
            active_sequence, layer_state.compressor.kind, layer, compressed);
        if (!result.ok()) return result;
    }

    ++graph_stats.attention_page_set_builds;
    const auto page_build_started = std::chrono::steady_clock::now();
    std::vector<Dsv4KvDeviceLease> leases;
    std::vector<CudaDsv4PhysicalPage> pages;
    std::unordered_map<std::uint64_t, std::uint32_t> page_indices;
    const auto reserve = compressed.size() + sliding.size();
    leases.reserve(reserve);
    pages.reserve(reserve);
    page_indices.reserve(reserve);
    const auto lease_table = [&](Dsv4KvBlockKind kind,
                                 const std::vector<Dsv4KvBlockInfo>& table) {
        for (const auto& block : table) {
            if (block.compression_ratio == 0U) {
                result.errors.emplace_back(
                    "DeepSeek physical attention page has a zero-ratio block");
                return;
            }
            // The lease names a block, not a row: every row inside the block
            // resolves to the same device buffer. Blocks are retired whole,
            // 256 rows at a time, while the sliding window retires rows one at
            // a time, so a block that is still live can already have its first
            // row outside the retained window -- which is every prefill page
            // based at or beyond kWindow. Name the block's most recent row
            // instead; it is retained for exactly as long as the block is.
            const auto first_row =
                block.logical_begin / block.compression_ratio;
            const auto logical_row = static_cast<std::uint32_t>(
                block.used_rows == 0U
                    ? first_row
                    : first_row + block.used_rows - 1U);
            auto lease = kv_cache->acquire_device(
                active_sequence, kind, layer, logical_row, slot);
            if (!lease.ok()) {
                append_errors(result, std::move(lease.errors));
                return;
            }
            const auto page = static_cast<std::uint32_t>(pages.size());
            leases.push_back(std::move(lease.value));
            pages.push_back({leases.back().buffer(), block.capacity_rows});
            page_indices.emplace(block.id, page);
            ++graph_stats.attention_page_set_pages;
        }
    };
    if (ratio != 0U) lease_table(layer_state.compressor.kind, compressed);
    if (!result.ok()) return result;
    lease_table(Dsv4KvBlockKind::Sliding, sliding);
    if (!result.ok()) return result;
    graph_stats.attention_page_set_build_nanoseconds +=
        elapsed_nanoseconds(page_build_started);

    const auto compressed_count = ratio == 0U
        ? 0U : (last_position + 1U) / ratio;
    const bool sparse = ratio == 4U &&
        layer_state.indexer_compressor.ratio == 4U;
    const auto compressed_width = ratio == 0U ? 0U : ratio == 4U
        ? kIndexTopK
        : ((std::max(1U, compressed_count) + 127U) / 128U) * 128U;
    constexpr std::uint32_t sliding_width = kWindow;
    const auto candidate_width = compressed_width + sliding_width;
    std::vector<CudaDsv4AttentionCandidate> candidates(
        static_cast<std::size_t>(rows) * candidate_width);

    const auto locate = [&](Dsv4KvBlockKind kind,
                            const std::vector<Dsv4KvBlockInfo>& table,
                            std::uint32_t logical_row,
                            CudaDsv4AttentionCandidate& candidate) {
        const auto located = locate_physical_kv_block(table, logical_row);
        if (located == table.size()) {
            result.errors.emplace_back(
                "DeepSeek physical attention page candidate is unavailable");
            return;
        }
        const auto& block = table[located];
        const auto found = page_indices.find(block.id);
        if (found == page_indices.end()) {
            result.errors.emplace_back(
                "DeepSeek physical attention page lease is unavailable");
            return;
        }
        const auto begin = block.logical_begin / block.compression_ratio;
        candidate.page = found->second;
        candidate.row = static_cast<std::uint32_t>(logical_row - begin);
        candidate.valid = true;
        static_cast<void>(kind);
    };

    const auto candidate_started = std::chrono::steady_clock::now();
    std::uint64_t selection_nanoseconds = 0U;
    std::vector<std::uint32_t> selected;
    for (std::uint32_t row = 0U; row < rows; ++row) {
        const auto position = position_base + row;
        const auto row_compressed_count = ratio == 0U
            ? 0U : (position + 1U) / ratio;
        if (sparse) {
            const auto selection_started = std::chrono::steady_clock::now();
            result = index_select(
                layer,
                input.subspan(static_cast<std::size_t>(row) * kHidden,
                              kHidden),
                query_rank.subspan(
                    static_cast<std::size_t>(row) * kQueryRank, kQueryRank),
                position, selected, false);
            selection_nanoseconds += elapsed_nanoseconds(selection_started);
            if (!result.ok()) return result;
        } else {
            selected.resize(row_compressed_count);
            std::iota(selected.begin(), selected.end(), 0U);
        }
        if (selected.size() > compressed_width) {
            result.errors.emplace_back(
                "DeepSeek physical attention page selection exceeds its region");
            return result;
        }
        auto row_candidates = std::span<CudaDsv4AttentionCandidate>(candidates)
            .subspan(static_cast<std::size_t>(row) * candidate_width,
                     candidate_width);
        graph_stats.attention_candidate_resolutions += selected.size();
        for (std::size_t item = 0U; item < selected.size(); ++item) {
            locate(layer_state.compressor.kind, compressed, selected[item],
                   row_candidates[item]);
            if (!result.ok()) return result;
        }
        const auto window_count = std::min(position + 1U, kWindow);
        graph_stats.attention_candidate_resolutions += window_count;
        for (std::uint32_t item = 0U; item < window_count; ++item) {
            const auto logical_row = position + 1U - window_count + item;
            locate(Dsv4KvBlockKind::Sliding, sliding, logical_row,
                   row_candidates[compressed_width + item]);
            if (!result.ok()) return result;
        }
    }
    graph_stats.attention_page_index_selection_nanoseconds +=
        selection_nanoseconds;
    graph_stats.attention_candidate_resolution_nanoseconds +=
        elapsed_nanoseconds(candidate_started) - selection_nanoseconds;

    const auto prefix = layer_prefix(layer) + "attn.";
    const auto weight_started = std::chrono::steady_clock::now();
    Dsv4WeightCache::Lease output_a;
    Dsv4WeightCache::Lease output_b;
    result = weights->acquire(
        slot, prefix + "wo_a", kOutputGroups * kOutputRank,
        kHeads * kHeadDim / kOutputGroups, output_a);
    if (!result.ok()) return result;
    result = weights->acquire(
        slot, prefix + "wo_b", kHidden,
        kOutputGroups * kOutputRank, output_b);
    if (!result.ok()) return result;
    graph_stats.attention_page_weight_acquire_nanoseconds +=
        elapsed_nanoseconds(weight_started);

    std::vector<float> cosines(static_cast<std::size_t>(rows) * kRopeDim / 2U);
    std::vector<float> sines(cosines.size());
    for (std::uint32_t row = 0U; row < rows; ++row) {
        const auto position = position_base + row;
        for (std::size_t index = 0U; index < kRopeDim / 2U; ++index) {
            const float angle = static_cast<float>(position) *
                                layer_state.frequencies[index] * -1.0F;
            cosines[static_cast<std::size_t>(row) * kRopeDim / 2U + index] =
                std::cos(angle);
            sines[static_cast<std::size_t>(row) * kRopeDim / 2U + index] =
                std::sin(angle);
        }
    }

    // A scheduling page may be wider than the bounded CUDA workspace. Derive
    // its row admission from the backend's exact allocation layout, then slice
    // only the query dimension. Pages and weights remain shared by every
    // slice, matching the reference stack's budgeted prefill chunks.
    constexpr std::uint64_t maximum_workspace_bytes = 384ULL << 20U;
    auto admitted = cuda.dsv4_paged_attention_to_mhc_page_maximum_rows(
        pages, rows, candidate_width, maximum_workspace_bytes);
    if (!admitted.ok()) return {std::move(admitted.errors)};
    const auto maximum_rows = admitted.value;
    if (const char* trace = std::getenv("STRATA_TRACE_ATTENTION_LAYOUT");
        trace != nullptr && *trace == '1') {
        static std::atomic<int> emitted{0};
        if (emitted.fetch_add(1) < 2) {
            auto at = [&](std::uint32_t probe) -> std::uint64_t {
                auto bytes = cuda
                    .dsv4_paged_attention_to_mhc_page_workspace_bytes(
                        pages, probe, candidate_width);
                return bytes.ok() ? bytes.value : 0U;
            };
            std::fprintf(stderr,
                "page admission: requested=%u admitted=%u chunks=%u "
                "ws(admitted)=%.2f MB ws(requested)=%.2f MB cap=%.2f MB\n",
                rows, maximum_rows,
                (rows + maximum_rows - 1U) / maximum_rows,
                at(maximum_rows) / 1048576.0, at(rows) / 1048576.0,
                maximum_workspace_bytes / 1048576.0);
        }
    }

    const auto rope_stride = static_cast<std::size_t>(kRopeDim) / 2U;
    std::uint32_t begin = 0U;
    while (begin < rows) {
        const auto remaining = rows - begin;
        auto chunk_rows = std::min(maximum_rows, remaining);
        // The page command uses the single-row path when rows == 1, which has
        // a different mHC state contract. Rebalance the preceding chunk so
        // every page slice retains at least two rows.
        if (remaining - chunk_rows == 1U) --chunk_rows;
        if (chunk_rows < 2U) {
            result.errors.emplace_back(
                "DeepSeek attention page workspace split produced a singleton slice");
            return result;
        }

        CudaDsv4PagedAttentionMhcRequest request;
        request.attention.queries = queries.subspan(
            static_cast<std::size_t>(begin) * query_stride,
            static_cast<std::size_t>(chunk_rows) * query_stride);
        request.attention.head_sinks = sinks;
        request.attention.pages = pages;
        request.attention.candidates = std::span<const CudaDsv4AttentionCandidate>(
            candidates).subspan(
                static_cast<std::size_t>(begin) * candidate_width,
                static_cast<std::size_t>(chunk_rows) * candidate_width);
        request.attention.rows = chunk_rows;
        request.attention.candidate_width = candidate_width;
        request.attention.scale = kAttentionScale;
        request.attention.maximum_workspace_bytes = maximum_workspace_bytes;
        request.mhc_slots = row_slots.subspan(begin, chunk_rows);
        request.inverse_rope_cosines = std::span<const float>(cosines).subspan(
            static_cast<std::size_t>(begin) * rope_stride,
            static_cast<std::size_t>(chunk_rows) * rope_stride);
        request.inverse_rope_sines = std::span<const float>(sines).subspan(
            static_cast<std::size_t>(begin) * rope_stride,
            static_cast<std::size_t>(chunk_rows) * rope_stride);
        request.output_a = &output_a.weight();
        request.output_b = &output_b.weight();
        request.mhc_device = devices[mhc_slot];

        result = cuda.dsv4_paged_attention_to_mhc(
            devices[slot], request, diagnostic_branches.subspan(
                static_cast<std::size_t>(begin) * kHidden,
                static_cast<std::size_t>(chunk_rows) * kHidden));
        if (!result.ok()) return result;
        begin += chunk_rows;
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::attention_append_prepared(
    std::uint32_t layer, std::span<const float> input,
    std::span<const float> kv, std::uint32_t position,
    std::span<const float> compressor_values,
    std::span<const float> compressor_scores,
    std::span<const float> index_compressor_values,
    std::span<const float> index_compressor_scores,
    bool append_index_compressor,
    std::uint64_t sliding_retention_floor) {
    ValidationResult result;
    if (input.size() != kHidden || kv.size() != kHeadDim) {
        result.errors.emplace_back("DeepSeek prepared attention append spans have incompatible sizes");
        return result;
    }
    const auto prefix = layer_prefix(layer) + "attn.";
    auto subphase_started = std::chrono::steady_clock::now();
    auto& layer_state = attention_state[layer];
    if (kv_cache != nullptr) {
        result = kv_cache->append(active_sequence,
                                  Dsv4KvBlockKind::Sliding, layer,
                                  1U, position, kv,
                                  sliding_retention_floor);
        if (!result.ok()) return result;
    } else {
        std::copy(kv.begin(), kv.end(),
                  layer_state.sliding.begin() +
                      static_cast<std::size_t>(position % kWindow) * kHeadDim);
    }
    result = compressor(layer, input, position,
                        compressor_values, compressor_scores);
    if (!result.ok()) return result;
    graph_stats.attention_kv_nanoseconds += elapsed_nanoseconds(subphase_started);
    if (append_index_compressor && layer_state.indexer_compressor.ratio != 0U) {
        subphase_started = std::chrono::steady_clock::now();
        result = compress_state(
            layer, layer_state.indexer_compressor,
            layer_prefix(layer) + "attn.indexer.compressor.", input, position,
            layer_state.frequencies, index_compressor_values,
            index_compressor_scores);
        if (!result.ok()) return result;
        graph_stats.attention_index_nanoseconds +=
            elapsed_nanoseconds(subphase_started);
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::attention_attend_prepared(
    std::uint32_t layer, std::span<const float> input,
    std::span<const float> query_rank, std::span<const float> queries,
    std::uint32_t position, std::span<float> output,
    bool index_compressor_prepared, bool device_prepared_source,
    PhysicalAttentionPageSet* page_set) {
    ValidationResult result;
    if (input.size() != kHidden || query_rank.size() != kQueryRank ||
        (queries.size() != static_cast<std::size_t>(kHeads) * kHeadDim &&
         !(config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice &&
           queries.empty())) ||
        (output.size() != kHidden &&
         !(config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice &&
           output.empty()))) {
        result.errors.emplace_back(
            "DeepSeek prepared attention spans have incompatible sizes");
        return result;
    }
    const auto slot = layer_device(layer);
    const auto prefix = layer_prefix(layer) + "attn.";
    auto& layer_state = attention_state[layer];
    std::vector<std::uint32_t> indexed_positions;
    const bool use_sparse_indexer =
        layer_state.compressor.ratio == 4U &&
        layer_state.indexer_compressor.ratio == 4U;
    if (use_sparse_indexer) {
        const auto subphase_started = std::chrono::steady_clock::now();
        if (index_compressor_prepared) {
            result = index_select(layer, input, query_rank, position,
                                  indexed_positions, device_prepared_source);
        } else {
            result = index_positions(layer, input, query_rank, position,
                                     indexed_positions, {}, {},
                                     device_prepared_source);
        }
        if (!result.ok()) return result;
        const auto selection_nanoseconds = elapsed_nanoseconds(subphase_started);
        graph_stats.attention_index_nanoseconds += selection_nanoseconds;
        if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
            graph_stats.attention_page_index_selection_nanoseconds +=
                selection_nanoseconds;
        }
    }

    auto sink = host_tensor(prefix + "attn_sink", kHeads);
    if (!sink.ok()) {
        append_errors(result, std::move(sink.errors));
        return result;
    }
    std::vector<float> attended(
        config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice
            ? 0U : static_cast<std::size_t>(kHeads) * kHeadDim,
        0.0F);
    const auto window_count = std::min<std::uint32_t>(position + 1U, kWindow);
    const auto ratio = layer_state.compressor.ratio;
    const auto compressed_count = ratio == 0U ? 0U : (position + 1U) / ratio;
    const auto attended_compressed_count = use_sparse_indexer
        ? static_cast<std::uint32_t>(indexed_positions.size())
        : compressed_count;
    const auto score_stride = static_cast<std::size_t>(window_count) +
                              attended_compressed_count;
    std::vector<std::vector<float>> block_rows;
    if (kv_cache != nullptr &&
        config.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice) {
        block_rows.reserve(score_stride);
        for (std::uint32_t item = 0U; item < window_count; ++item) {
            const auto absolute = position + 1U - window_count + item;
            auto row = kv_row(layer, Dsv4KvBlockKind::Sliding, absolute);
            if (!row.ok()) {
                append_errors(result, std::move(row.errors));
                return result;
            }
            block_rows.push_back(std::move(row.value));
        }
        for (std::uint32_t item = 0U; item < attended_compressed_count; ++item) {
            const auto cache_row = use_sparse_indexer
                ? indexed_positions[item] : item;
            auto row = kv_row(layer, layer_state.compressor.kind, cache_row);
            if (!row.ok()) {
                append_errors(result, std::move(row.errors));
                return result;
            }
            block_rows.push_back(std::move(row.value));
        }
    }
    const bool use_cuda_attention = should_dispatch_flash_attention_cuda(
        config.enable_flash_attention, score_stride,
        config.flash_attention_minimum_rows);
    auto subphase_started = std::chrono::steady_clock::now();
    if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
        auto cuda_demand = weights->demand();
        ++graph_stats.attention_cuda_dispatches;
        result = physical_paged_attention(
            layer, queries, *sink.value, position, indexed_positions,
            output, page_set);
        if (!result.ok()) return result;
    } else if (use_cuda_attention) {
        auto cuda_demand = weights->demand();
        ++graph_stats.attention_cuda_dispatches;
        std::vector<std::uint32_t> sliding_rows(window_count);
        for (std::uint32_t item = 0U; item < window_count; ++item) {
            const auto absolute = position + 1U - window_count + item;
            sliding_rows[item] = absolute % kWindow;
        }
        std::vector<FlashAttentionSegment> segments;
        if (kv_cache != nullptr) {
            segments.reserve(block_rows.size());
            for (const auto& row : block_rows) {
                segments.push_back({std::span<const float>(row), {}, {}});
            }
        } else {
            segments.reserve(
                static_cast<std::size_t>(attended_compressed_count) + 1U);
            if (window_count != 0U) {
                segments.push_back({layer_state.sliding, {}, sliding_rows});
            }
            for (std::uint32_t item = 0U;
                 item < attended_compressed_count; ++item) {
                const auto cache_row = use_sparse_indexer
                    ? indexed_positions[item] : item;
                const auto row = layer_state.compressor.compressed.row(cache_row);
                if (row.size() != kHeadDim) {
                    result.errors.emplace_back(
                        "DeepSeek FlashAttention compressed row is unavailable");
                    return result;
                }
                segments.push_back({row, {}, {}});
            }
        }
        FlashAttentionRequest request;
        request.queries = queries;
        request.segments = segments;
        request.head_sinks = *sink.value;
        request.query_rows = 1U;
        request.query_heads = kHeads;
        request.key_value_heads = 1U;
        request.query_key_dim = kHeadDim;
        request.value_dim = kHeadDim;
        request.scale = kAttentionScale;
        request.numerics =
            FlashAttentionNumerics::f64_dot_f32_score_f32_accum;
        request.maximum_workspace_bytes = kDeviceWorkspaceReserve;
        result = cuda.flash_attention(devices[slot], request, attended);
        if (!result.ok()) return result;
        const auto finish_head = [&](std::uint32_t head) {
            auto destination = std::span<float>(attended).subspan(
                static_cast<std::size_t>(head) * kHeadDim, kHeadDim);
            round_bf16(destination);
            apply_rope(destination.last(kRopeDim), position,
                       layer_state.frequencies, true);
            round_bf16(destination.last(kRopeDim));
        };
        if (attention_workers != nullptr) {
            result = attention_workers->parallel_for(
                kHeads, [&](std::size_t head) {
                    finish_head(static_cast<std::uint32_t>(head));
                });
            if (!result.ok()) return result;
        } else {
            for (std::uint32_t head = 0U; head < kHeads; ++head) {
                finish_head(head);
            }
        }
    } else {
        ++graph_stats.attention_scalar_dispatches;
    const auto attend_head = [&](std::uint32_t head,
                                 std::span<float> scores) {
        const auto query = std::span<const float>(queries).subspan(
            static_cast<std::size_t>(head) * kHeadDim, kHeadDim);
        std::size_t next_score = 0U;
        float maximum = (*sink.value)[head];
        for (std::uint32_t item = 0U; item < window_count; ++item) {
            const auto absolute = position + 1U - window_count + item;
            const auto key = kv_cache == nullptr
                ? std::span<const float>(layer_state.sliding).subspan(
                      static_cast<std::size_t>(absolute % kWindow) * kHeadDim,
                      kHeadDim)
                : std::span<const float>(block_rows[item]);
            double dot = 0.0;
            for (std::uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
                dot += static_cast<double>(query[dimension]) * key[dimension];
            }
            const float score = static_cast<float>(dot) * kAttentionScale;
            scores[next_score++] = score;
            maximum = std::max(maximum, score);
        }
        for (std::uint32_t item = 0U; item < attended_compressed_count; ++item) {
            const auto cache_row = use_sparse_indexer
                ? indexed_positions[item]
                : item;
            const auto key = kv_cache == nullptr
                ? layer_state.compressor.compressed.row(cache_row)
                : std::span<const float>(
                      block_rows[static_cast<std::size_t>(window_count) + item]);
            double dot = 0.0;
            for (std::uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
                dot += static_cast<double>(query[dimension]) * key[dimension];
            }
            const float score = static_cast<float>(dot) * kAttentionScale;
            scores[next_score++] = score;
            maximum = std::max(maximum, score);
        }
        double denominator = std::exp(
            static_cast<double>((*sink.value)[head] - maximum));
        for (const float score : scores) {
            denominator += std::exp(static_cast<double>(score - maximum));
        }
        auto destination = std::span<float>(attended).subspan(
            static_cast<std::size_t>(head) * kHeadDim, kHeadDim);
        std::size_t score_index = 0U;
        for (std::uint32_t item = 0U; item < window_count; ++item) {
            const auto absolute = position + 1U - window_count + item;
            const auto value = kv_cache == nullptr
                ? std::span<const float>(layer_state.sliding).subspan(
                      static_cast<std::size_t>(absolute % kWindow) * kHeadDim,
                      kHeadDim)
                : std::span<const float>(block_rows[item]);
            const float probability = static_cast<float>(
                std::exp(static_cast<double>(scores[score_index++] - maximum)) /
                denominator);
            for (std::uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
                destination[dimension] += probability * value[dimension];
            }
        }
        for (std::uint32_t item = 0U; item < attended_compressed_count; ++item) {
            const auto cache_row = use_sparse_indexer
                ? indexed_positions[item]
                : item;
            const auto value = kv_cache == nullptr
                ? layer_state.compressor.compressed.row(cache_row)
                : std::span<const float>(
                      block_rows[static_cast<std::size_t>(window_count) + item]);
            const float probability = static_cast<float>(
                std::exp(static_cast<double>(scores[score_index++] - maximum)) /
                denominator);
            for (std::uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
                destination[dimension] += probability * value[dimension];
            }
        }
        round_bf16(destination);
        apply_rope(destination.last(kRopeDim), position,
                   layer_state.frequencies, true);
        round_bf16(destination.last(kRopeDim));
    };
    if (attention_workers != nullptr) {
        std::vector<float> parallel_scores(
            static_cast<std::size_t>(kHeads) * score_stride);
        result = attention_workers->parallel_for(
            kHeads, [&](std::size_t head) {
                attend_head(
                    static_cast<std::uint32_t>(head),
                    std::span<float>(parallel_scores).subspan(
                        head * score_stride, score_stride));
            });
        if (!result.ok()) return result;
    } else {
        std::vector<float> scores(score_stride);
        for (std::uint32_t head = 0U; head < kHeads; ++head) {
            attend_head(head, scores);
        }
    }
    }
    graph_stats.attention_score_nanoseconds += elapsed_nanoseconds(subphase_started);

    if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
        return result;
    }

    subphase_started = std::chrono::steady_clock::now();
    std::vector<float> output_rank(static_cast<std::size_t>(kOutputGroups) *
                                   kOutputRank);
    result = weights->grouped(slot, prefix + "wo_a", kOutputGroups * kOutputRank,
                              kHeads * kHeadDim / kOutputGroups, attended,
                              kOutputGroups, kOutputRank, output_rank);
    if (!result.ok()) return result;
    result = linear(slot, prefix + "wo_b", kHidden,
                    kOutputGroups * kOutputRank, output_rank, output);
    graph_stats.attention_output_nanoseconds += elapsed_nanoseconds(subphase_started);
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::attention_prepared(
    std::uint32_t layer, std::span<const float> input,
    std::span<const float> query_rank, std::span<const float> queries,
    std::span<const float> kv, std::uint32_t position,
    std::span<float> output,
    std::span<const float> compressor_values,
    std::span<const float> compressor_scores,
    std::span<const float> index_compressor_values,
    std::span<const float> index_compressor_scores,
    bool device_prepared_source) {
    auto result = attention_append_prepared(
        layer, input, kv, position, compressor_values, compressor_scores,
        index_compressor_values, index_compressor_scores, true);
    if (!result.ok()) return result;
    return attention_attend_prepared(
        layer, input, query_rank, queries, position, output, true,
        device_prepared_source);
}

ValidationResult DeepSeekV4Runtime::Impl::attention_page(
    std::uint32_t layer, std::span<const float> input,
    std::uint32_t position_base, std::span<float> output,
    std::span<const std::uint32_t> row_slots) {
    ValidationResult result;
    if (input.empty() || input.size() % kHidden != 0U ||
        output.size() != input.size() ||
        (!row_slots.empty() && row_slots.size() != input.size() / kHidden)) {
        result.errors.emplace_back(
            "DeepSeek attention page spans have incompatible sizes");
        return result;
    }
    const auto row_count = input.size() / kHidden;
    if (row_count > std::numeric_limits<std::uint32_t>::max()) {
        result.errors.emplace_back("DeepSeek attention page row count overflows");
        return result;
    }
    const auto rows = static_cast<std::uint32_t>(row_count);
    if (rows > config.prefill_page_tokens ||
        rows > config.maximum_context_tokens ||
        position_base > config.maximum_context_tokens - rows) {
        result.errors.emplace_back(
            "DeepSeek attention page exceeds the admitted context bounds");
        return result;
    }

    const auto slot = layer_device(layer);
    const auto prefix = layer_prefix(layer) + "attn.";
    const auto add_matmul_profile = [](const CudaMatmulProfile& profile,
                                       std::uint64_t& weight_acquisition,
                                       std::uint64_t& issue,
                                       std::uint64_t& finish,
                                       std::uint64_t& synchronization,
                                       std::uint64_t& h2d,
                                       std::uint64_t& kernel,
                                       std::uint64_t& d2h) {
        weight_acquisition += profile.weight_acquisition_nanoseconds;
        issue += profile.issue_nanoseconds;
        finish += profile.finish_nanoseconds;
        synchronization += profile.synchronization_nanoseconds;
        h2d += profile.h2d_nanoseconds;
        kernel += profile.kernel_nanoseconds;
        d2h += profile.d2h_nanoseconds;
    };
    auto subphase_started = std::chrono::steady_clock::now();
    auto allocation_started = std::chrono::steady_clock::now();
    auto query_rank = attention_page_query_rank_scratch.acquire(
        row_count * kQueryRank);
    graph_stats.attention_query_allocation_nanoseconds +=
        elapsed_nanoseconds(allocation_started);
    ++graph_stats.attention_projection_matmul_calls;
    graph_stats.attention_projection_matmul_rows += rows;
    CudaMatmulProfile query_a_profile;
    result = linear_rows(slot, prefix + "wq_a", kQueryRank, kHidden, input,
                         rows, query_rank, true, &query_a_profile,
                         config.enable_dsv4_fp8_tensor_page);
    if (!result.ok()) return result;
    add_matmul_profile(
        query_a_profile,
        graph_stats.attention_query_weight_acquisition_nanoseconds,
        graph_stats.attention_query_matmul_issue_nanoseconds,
        graph_stats.attention_query_matmul_finish_nanoseconds,
        graph_stats.attention_query_matmul_sync_nanoseconds,
        graph_stats.attention_query_matmul_h2d_nanoseconds,
        graph_stats.attention_query_matmul_kernel_nanoseconds,
        graph_stats.attention_query_matmul_d2h_nanoseconds);
    const auto query_rank_norm_started = std::chrono::steady_clock::now();
    result = norm_rows(query_rank, query_rank, rows, kQueryRank,
                       prefix + "q_norm.weight");
    if (!result.ok()) return result;
    graph_stats.attention_query_rank_norm_nanoseconds +=
        elapsed_nanoseconds(query_rank_norm_started);

    const auto query_stride = static_cast<std::size_t>(kHeads) * kHeadDim;
    allocation_started = std::chrono::steady_clock::now();
    auto queries = attention_page_query_scratch.acquire(
        row_count * query_stride);
    graph_stats.attention_query_allocation_nanoseconds +=
        elapsed_nanoseconds(allocation_started);
    ++graph_stats.attention_projection_matmul_calls;
    graph_stats.attention_projection_matmul_rows += rows;
    CudaMatmulProfile query_b_profile;
    result = linear_rows(slot, prefix + "wq_b", query_stride, kQueryRank,
                         query_rank, rows, queries, true, &query_b_profile,
                         config.enable_dsv4_fp8_tensor_page);
    if (!result.ok()) return result;
    add_matmul_profile(
        query_b_profile,
        graph_stats.attention_query_weight_acquisition_nanoseconds,
        graph_stats.attention_query_matmul_issue_nanoseconds,
        graph_stats.attention_query_matmul_finish_nanoseconds,
        graph_stats.attention_query_matmul_sync_nanoseconds,
        graph_stats.attention_query_matmul_h2d_nanoseconds,
        graph_stats.attention_query_matmul_kernel_nanoseconds,
        graph_stats.attention_query_matmul_d2h_nanoseconds);
    // One task per row rather than per (row, head). A head is about 1,500
    // flops, so at 64 heads a page of 677 rows dispatched 1.86 million tasks
    // across 43 layers and spent 11.2 s in pool overhead for work that is
    // nowhere near that size.
    std::atomic<std::uint64_t> query_rms_cpu_nanoseconds{0U};
    std::atomic<std::uint64_t> query_rope_cpu_nanoseconds{0U};
    const auto normalize_query_row = [&](std::size_t row) {
        auto cpu_started = std::chrono::steady_clock::now();
        for (std::uint32_t head = 0U; head < kHeads; ++head) {
            auto query = queries.subspan(
                row * query_stride + head * kHeadDim, kHeadDim);
            double square_sum = 0.0;
            for (const float value : query) {
                square_sum += static_cast<double>(value) * value;
            }
            const float reciprocal = 1.0F / std::sqrt(
                static_cast<float>(square_sum / kHeadDim) + kRmsEpsilon);
            for (auto& value : query) value = round_bf16(value * reciprocal);
        }
        query_rms_cpu_nanoseconds.fetch_add(
            elapsed_nanoseconds(cpu_started), std::memory_order_relaxed);
        cpu_started = std::chrono::steady_clock::now();
        for (std::uint32_t head = 0U; head < kHeads; ++head) {
            auto query = queries.subspan(
                row * query_stride + head * kHeadDim, kHeadDim);
            apply_rope(query.last(kRopeDim),
                       position_base + static_cast<std::uint32_t>(row),
                       attention_state[layer].frequencies);
            round_bf16(query.last(kRopeDim));
        }
        query_rope_cpu_nanoseconds.fetch_add(
            elapsed_nanoseconds(cpu_started), std::memory_order_relaxed);
    };
    const auto query_finish_started = std::chrono::steady_clock::now();
    if (attention_workers != nullptr && row_count > 1U) {
        result = attention_workers->parallel_for(row_count,
                                                 normalize_query_row);
        if (!result.ok()) return result;
    } else {
        for (std::size_t row = 0U; row < row_count; ++row) {
            normalize_query_row(row);
        }
    }
    graph_stats.attention_query_finish_nanoseconds +=
        elapsed_nanoseconds(query_finish_started);
    graph_stats.attention_query_rms_cpu_nanoseconds +=
        query_rms_cpu_nanoseconds.load(std::memory_order_relaxed);
    graph_stats.attention_query_rope_cpu_nanoseconds +=
        query_rope_cpu_nanoseconds.load(std::memory_order_relaxed);
    graph_stats.attention_query_nanoseconds += elapsed_nanoseconds(subphase_started);

    subphase_started = std::chrono::steady_clock::now();
    allocation_started = std::chrono::steady_clock::now();
    auto kv = attention_page_kv_scratch.acquire(row_count * kHeadDim);
    graph_stats.attention_kv_allocation_nanoseconds +=
        elapsed_nanoseconds(allocation_started);
    ++graph_stats.attention_projection_matmul_calls;
    graph_stats.attention_projection_matmul_rows += rows;
    CudaMatmulProfile kv_profile;
    result = linear_rows(slot, prefix + "wkv", kHeadDim, kHidden, input, rows,
                         kv, true, &kv_profile,
                         config.enable_dsv4_fp8_tensor_page);
    if (!result.ok()) return result;
    add_matmul_profile(
        kv_profile, graph_stats.attention_kv_weight_acquisition_nanoseconds,
        graph_stats.attention_kv_matmul_issue_nanoseconds,
        graph_stats.attention_kv_matmul_finish_nanoseconds,
        graph_stats.attention_kv_matmul_sync_nanoseconds,
        graph_stats.attention_kv_matmul_h2d_nanoseconds,
        graph_stats.attention_kv_matmul_kernel_nanoseconds,
        graph_stats.attention_kv_matmul_d2h_nanoseconds);
    const auto kv_norm_started = std::chrono::steady_clock::now();
    result = norm_rows(kv, kv, rows, kHeadDim, prefix + "kv_norm.weight");
    if (!result.ok()) return result;
    graph_stats.attention_kv_norm_nanoseconds +=
        elapsed_nanoseconds(kv_norm_started);
    const auto finish_kv = [&](std::size_t row) {
        auto kv_row = std::span<float>(kv).subspan(row * kHeadDim, kHeadDim);
        apply_rope(kv_row.last(kRopeDim),
                   position_base + static_cast<std::uint32_t>(row),
                   attention_state[layer].frequencies);
        round_bf16(kv_row.last(kRopeDim));
        if (config.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice) {
            quantize_activation_in_place(
                kv_row.first(kHeadDim - kRopeDim), 64U);
        }
    };
    const auto kv_rope_started = std::chrono::steady_clock::now();
    if (attention_workers != nullptr && rows > 1U) {
        result = attention_workers->parallel_for(rows, finish_kv);
        if (!result.ok()) return result;
    } else {
        for (std::uint32_t row = 0U; row < rows; ++row) finish_kv(row);
    }
    graph_stats.attention_kv_rope_nanoseconds +=
        elapsed_nanoseconds(kv_rope_started);
    graph_stats.attention_kv_nanoseconds += elapsed_nanoseconds(subphase_started);

    auto& layer_state = attention_state[layer];
    const auto last_position = position_base + rows - 1U;
    const auto ratio = layer_state.compressor.ratio;
    const bool use_sparse_indexer =
        ratio == 4U && layer_state.indexer_compressor.ratio == 4U;
    const auto maximum_score_rows =
        std::min(last_position + 1U, kWindow) +
        (ratio == 0U ? 0U : (last_position + 1U) / ratio);
    if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice &&
        config.enable_dsv4_batched_page_attention) {
        // A physical KV block rejects mutation while any device lease is
        // outstanding. Append every row (including both compressor states)
        // before resolving or leasing a page, then attend the rows in order
        // against one page-set/map. The candidates remain row-local, so
        // causality and candidate order are unchanged.
        PhysicalAttentionPageSet page_set;
        const auto sliding_retention_floor =
            position_base + 1U > kWindow
                ? static_cast<std::uint64_t>(position_base + 1U - kWindow)
                : 0U;
        for (std::uint32_t row = 0U; row < rows; ++row) {
            const auto input_row = input.subspan(
                static_cast<std::size_t>(row) * kHidden, kHidden);
            const auto kv_row_values = std::span<const float>(kv).subspan(
                static_cast<std::size_t>(row) * kHeadDim, kHeadDim);
            result = attention_append_prepared(
                layer, input_row, kv_row_values,
                position_base + row, {}, {}, {}, {}, true,
                sliding_retention_floor);
            if (!result.ok()) return result;
        }
        if (rows > 1U && !row_slots.empty()) {
            auto sink = host_tensor(prefix + "attn_sink", kHeads);
            if (!sink.ok()) {
                append_errors(result, std::move(sink.errors));
                return result;
            }
            subphase_started = std::chrono::steady_clock::now();
            result = physical_paged_attention_page(
                layer, input, query_rank, queries, *sink.value,
                position_base, row_slots, output);
            graph_stats.attention_score_nanoseconds +=
                elapsed_nanoseconds(subphase_started);
            return result;
        }
        for (std::uint32_t row = 0U; row < rows; ++row) {
            if (!row_slots.empty()) {
                result = cuda.dsv4_mhc_select_slot(
                    devices[mhc_slot], row_slots[row]);
                if (!result.ok()) return result;
            }
            const auto input_row = input.subspan(
                static_cast<std::size_t>(row) * kHidden, kHidden);
            const auto query_rank_row = std::span<const float>(query_rank)
                .subspan(static_cast<std::size_t>(row) * kQueryRank, kQueryRank);
            const auto queries_row = std::span<const float>(queries).subspan(
                static_cast<std::size_t>(row) * query_stride, query_stride);
            auto output_row = output.subspan(
                static_cast<std::size_t>(row) * kHidden, kHidden);
            result = attention_attend_prepared(
                layer, input_row, query_rank_row, queries_row,
                position_base + row, output_row, true, false, &page_set);
            if (!result.ok()) return result;
        }
        return result;
    }
    const bool batch_cuda =
        config.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice &&
        rows > 1U && kv_cache != nullptr &&
        !use_sparse_indexer && should_dispatch_flash_attention_cuda(
            config.enable_flash_attention, maximum_score_rows,
            config.flash_attention_minimum_rows);
    if (batch_cuda) {
        const auto sliding_begin = position_base + 1U > kWindow
            ? position_base + 1U - kWindow : 0U;
        const auto sliding_end = last_position + 1U;
        const auto sliding_rows = sliding_end - sliding_begin;
        const auto historical_sliding_rows = position_base - sliding_begin;
        const auto compressed_rows = ratio == 0U
            ? 0U : (last_position + 1U) / ratio;
        const auto key_rows = sliding_rows + compressed_rows;
        std::vector<float> gathered(
            static_cast<std::size_t>(key_rows) * kHeadDim);
        for (std::uint32_t row = 0U; row < historical_sliding_rows; ++row) {
            auto source = kv_row(layer, Dsv4KvBlockKind::Sliding,
                                 sliding_begin + row);
            if (!source.ok()) {
                append_errors(result, std::move(source.errors));
                return result;
            }
            std::copy(source.value.begin(), source.value.end(),
                      gathered.begin() +
                          static_cast<std::ptrdiff_t>(row) * kHeadDim);
        }

        subphase_started = std::chrono::steady_clock::now();
        for (std::uint32_t row = 0U; row < rows; ++row) {
            const auto position = position_base + row;
            const auto input_row = input.subspan(
                static_cast<std::size_t>(row) * kHidden, kHidden);
            const auto kv_row_values = std::span<const float>(kv).subspan(
                static_cast<std::size_t>(row) * kHeadDim, kHeadDim);
            result = kv_cache->append(active_sequence,
                                      Dsv4KvBlockKind::Sliding, layer,
                                      1U, position, kv_row_values);
            if (!result.ok()) return result;
            result = compressor(layer, input_row, position);
            if (!result.ok()) return result;
            std::copy(kv_row_values.begin(), kv_row_values.end(),
                      gathered.begin() + static_cast<std::ptrdiff_t>(
                          historical_sliding_rows + row) * kHeadDim);
        }
        graph_stats.attention_kv_nanoseconds +=
            elapsed_nanoseconds(subphase_started);

        auto sink = host_tensor(prefix + "attn_sink", kHeads);
        if (!sink.ok()) {
            append_errors(result, std::move(sink.errors));
            return result;
        }
        for (std::uint32_t row = 0U; row < compressed_rows; ++row) {
            auto source = kv_row(layer, layer_state.compressor.kind, row);
            if (!source.ok()) {
                append_errors(result, std::move(source.errors));
                return result;
            }
            std::copy(source.value.begin(), source.value.end(),
                      gathered.begin() + static_cast<std::ptrdiff_t>(
                          sliding_rows + row) * kHeadDim);
        }

        std::vector<std::uint8_t> mask(
            static_cast<std::size_t>(rows) * key_rows, 0U);
        for (std::uint32_t row = 0U; row < rows; ++row) {
            const auto position = position_base + row;
            const auto window_count = std::min(position + 1U, kWindow);
            const auto window_begin = position + 1U - window_count;
            auto row_mask = std::span<std::uint8_t>(mask).subspan(
                static_cast<std::size_t>(row) * key_rows, key_rows);
            std::fill(row_mask.begin() +
                          static_cast<std::ptrdiff_t>(window_begin - sliding_begin),
                      row_mask.begin() + static_cast<std::ptrdiff_t>(
                          position + 1U - sliding_begin),
                      1U);
            const auto visible_compressed = ratio == 0U
                ? 0U : (position + 1U) / ratio;
            std::fill_n(row_mask.begin() + sliding_rows,
                        visible_compressed, 1U);
        }

        subphase_started = std::chrono::steady_clock::now();
        graph_stats.attention_cuda_dispatches += rows;
        const std::array<FlashAttentionSegment, 1> segments{{
            {gathered, {}, {}}}};
        std::vector<float> attended(
            row_count * static_cast<std::size_t>(kHeads) * kHeadDim);
        FlashAttentionRequest request;
        request.queries = queries;
        request.segments = segments;
        request.head_sinks = *sink.value;
        request.query_key_mask = mask;
        request.query_rows = rows;
        request.query_heads = kHeads;
        request.key_value_heads = 1U;
        request.query_key_dim = kHeadDim;
        request.value_dim = kHeadDim;
        request.scale = kAttentionScale;
        request.numerics =
            FlashAttentionNumerics::f64_dot_f32_score_f32_accum;
        request.maximum_workspace_bytes = kDeviceWorkspaceReserve;
        {
            auto cuda_demand = weights->demand();
            result = cuda.flash_attention(devices[slot], request, attended);
        }
        if (!result.ok()) return result;
        const auto finish_head = [&](std::size_t task) {
            const auto row = task / kHeads;
            auto destination = std::span<float>(attended).subspan(
                task * kHeadDim, kHeadDim);
            round_bf16(destination);
            apply_rope(destination.last(kRopeDim),
                       position_base + static_cast<std::uint32_t>(row),
                       layer_state.frequencies, true);
            round_bf16(destination.last(kRopeDim));
        };
        if (attention_workers != nullptr) {
            result = attention_workers->parallel_for(
                row_count * kHeads, finish_head);
            if (!result.ok()) return result;
        } else {
            for (std::size_t task = 0U; task < row_count * kHeads; ++task) {
                finish_head(task);
            }
        }
        graph_stats.attention_score_nanoseconds +=
            elapsed_nanoseconds(subphase_started);

        subphase_started = std::chrono::steady_clock::now();
        std::vector<float> output_rank(
            row_count * static_cast<std::size_t>(kOutputGroups) * kOutputRank);
        result = weights->grouped_rows(
            slot, prefix + "wo_a", kOutputGroups * kOutputRank,
            kHeads * kHeadDim / kOutputGroups, attended, rows,
            kOutputGroups, kOutputRank, output_rank);
        if (!result.ok()) return result;
        result = linear_rows(slot, prefix + "wo_b", kHidden,
                             kOutputGroups * kOutputRank, output_rank,
                             rows, output, true, nullptr,
                             config.enable_dsv4_fp8_tensor_page);
        graph_stats.attention_output_nanoseconds +=
            elapsed_nanoseconds(subphase_started);
        return result;
    }

    for (std::uint32_t row = 0U; row < rows; ++row) {
        if (!row_slots.empty()) {
            result = cuda.dsv4_mhc_select_slot(
                devices[mhc_slot], row_slots[row]);
            if (!result.ok()) return result;
        }
        const auto input_row = input.subspan(
            static_cast<std::size_t>(row) * kHidden, kHidden);
        const auto query_rank_row = std::span<const float>(query_rank).subspan(
            static_cast<std::size_t>(row) * kQueryRank, kQueryRank);
        const auto queries_row = std::span<const float>(queries).subspan(
            static_cast<std::size_t>(row) * query_stride, query_stride);
        const auto kv_row = std::span<const float>(kv).subspan(
            static_cast<std::size_t>(row) * kHeadDim, kHeadDim);
        auto output_row = output.subspan(
            static_cast<std::size_t>(row) * kHidden, kHidden);
        result = attention_prepared(layer, input_row, query_rank_row,
                                    queries_row, kv_row, position_base + row,
                                    output_row);
        if (!result.ok()) return result;
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::expert(
    std::uint32_t layer, std::uint32_t expert_id,
    float routed_coefficient,
    std::span<const float> input, std::span<float> output) {
    ValidationResult result;
    if (expert_id >= kExperts || input.size() != kHidden || output.size() != kHidden) {
        result.errors.emplace_back("DeepSeek expert id or span shape is invalid");
        return result;
    }
    const auto slot = expert_device(expert_id);
    const auto prefix = layer_prefix(layer) + "ffn.experts." +
                        std::to_string(expert_id) + ".";
    std::vector<float> gate(kExpertIntermediate);
    std::vector<float> up(kExpertIntermediate);
    std::vector<float> activated(kExpertIntermediate);
    result = linear(slot, prefix + "w1", kExpertIntermediate, kHidden, input, gate);
    if (!result.ok()) return result;
    result = linear(slot, prefix + "w3", kExpertIntermediate, kHidden, input, up);
    if (!result.ok()) return result;
    result = dsv4_swiglu_f32(activated, gate, up, 10.0F);
    if (!result.ok()) return result;
    for (auto& value : activated) {
        value *= routed_coefficient;
    }
    round_bf16(activated);
    return linear(slot, prefix + "w2", kHidden, kExpertIntermediate,
                  activated, output);
}

ValidationResult DeepSeekV4Runtime::Impl::host_routed_moe(
    std::uint32_t layer, const Dsv4Route& route,
    std::span<const float> input, std::span<float> output) {
    return host_routed_moe_impl(layer, &route, input, output, 0U, 0U);
}

ValidationResult DeepSeekV4Runtime::Impl::host_routed_moe_from_device_input(
    std::uint32_t layer, std::uint32_t token, std::uint32_t position,
    std::span<float> output) {
    if (!output.empty()) {
        return {{"DeepSeek deferred CPU-MoE output must remain device-owned"}};
    }
    return enqueue_host_routed_moe(layer, token, position);
}

ValidationResult DeepSeekV4Runtime::Impl::enqueue_host_routed_moe(
    std::uint32_t layer, std::uint32_t token, std::uint32_t position) {
    ValidationResult result;
    const auto chain = host_moe_chain_row.value_or(layer);
    if (layer >= kLayers || host_moe_pending != chain ||
        expert_workers == nullptr) {
        result.errors.emplace_back(
            "DeepSeek fixed CPU-MoE command order is invalid");
        return result;
    }
    if (host_moe_pending == 0U) {
        host_moe_routed_cpu_before =
            device_moe_stats.routed_cpu_nanoseconds;
    }
    auto& context = host_moe_context(chain);
    context.owner = this;
    context.layer = layer;
    context.token = token;
    context.position = position;
    context.result = {};
    context.invoked = false;
    context.accepted = false;
    context.callback_finished = {};
    context.execution_started = std::chrono::steady_clock::now();
    if (context.input.size() != kHidden) context.input.resize(kHidden);

    if (context.shared.w1 == nullptr) {
        const auto shared_slot = mhc_slot;
        context.shared.coefficient = 1.0F;
        const auto prefix = layer_prefix(layer) + "ffn.shared_experts.";
        const std::array<std::string, 3U> names{
            prefix + "w1", prefix + "w3", prefix + "w2"};
        constexpr std::array<std::pair<std::uint64_t, std::uint64_t>, 3U>
            shapes{{{kExpertIntermediate, kHidden},
                    {kExpertIntermediate, kHidden},
                    {kHidden, kExpertIntermediate}}};
        const CudaWeight** outputs[]{
            &context.shared.w1, &context.shared.w3, &context.shared.w2};
        for (std::size_t index = 0U; index < names.size(); ++index) {
            auto acquired = weights->acquire(
                shared_slot, names[index], shapes[index].first,
                shapes[index].second, context.shared_leases[index]);
            if (!acquired.ok()) {
                append_errors(result, std::move(acquired.errors),
                              names[index]);
                return result;
            }
            *outputs[index] = &context.shared_leases[index].weight();
        }
    }
    graph_stats.moe_prepare_nanoseconds += elapsed_nanoseconds(
        context.execution_started);
    auto enqueued = cuda.enqueue_dsv4_host_moe_from_device_input(
        devices[mhc_slot], context.shared,
        kDeepSeekV4ExecutionContract.swiglu_limit,
        host_routed_moe_callback, &context);
    if (!enqueued.ok()) {
        append_errors(result, std::move(enqueued.errors),
                      "DeepSeek fixed CPU/shared MoE enqueue");
        return result;
    }
    ++host_moe_pending;
    return result;
}

bool DeepSeekV4Runtime::Impl::host_routed_moe_callback(
    void* opaque, std::span<const std::uint16_t> encoded_hidden,
    std::span<const float> router_logits,
    std::span<float> rank_partials) {
    if (opaque == nullptr) return false;
    auto& context = *static_cast<HostMoeContext*>(opaque);
    if (context.owner == nullptr) return false;
    return context.owner->execute_host_routed_moe_callback(
        context, encoded_hidden, router_logits, rank_partials);
}

bool DeepSeekV4Runtime::Impl::execute_host_routed_moe_callback(
    HostMoeContext& context,
    std::span<const std::uint16_t> encoded_hidden,
    std::span<const float> router_logits,
    std::span<float> rank_partials) {
    constexpr std::size_t shards = 2U;
    constexpr std::size_t shard_intermediate =
        kExpertIntermediate / shards;
    context.invoked = true;
    context.result = {};
    const auto routed_started = std::chrono::steady_clock::now();
    if (encoded_hidden.size() != kHidden ||
        router_logits.size() != kExperts ||
        rank_partials.size() != shards * kHidden) {
        context.result.errors.emplace_back(
            "DeepSeek fixed CPU-MoE callback shape is invalid");
        context.callback_finished = std::chrono::steady_clock::now();
        return false;
    }
    for (std::size_t index = 0U; index < encoded_hidden.size(); ++index) {
        context.input[index] = std::bit_cast<float>(
            static_cast<std::uint32_t>(encoded_hidden[index]) << 16U);
        if (!std::isfinite(context.input[index])) {
            context.result.errors.emplace_back(
                "DeepSeek fixed CPU-MoE hidden row is non-finite");
            context.callback_finished = std::chrono::steady_clock::now();
            return false;
        }
    }
    const auto route_started = std::chrono::steady_clock::now();
    context.result = route_moe(
        context.layer, context.token, router_logits, context.position,
        context.route);
    graph_stats.moe_router_nanoseconds += elapsed_nanoseconds(route_started);
    if (!context.result.ok()) {
        context.callback_finished = std::chrono::steady_clock::now();
        return false;
    }
    for (std::size_t shard = 0U; shard < shards; ++shard) {
        for (std::size_t rank = 0U; rank < kTopK; ++rank) {
            auto viewed = dsv4_tiled_expert_weights(
                resident.find_tiled_expert(
                    context.layer, context.route.experts[rank],
                    static_cast<std::uint32_t>(shard)),
                kHidden, kExpertIntermediate, shards);
            if (!viewed.ok()) {
                append_errors(context.result, std::move(viewed.errors));
                context.callback_finished = std::chrono::steady_clock::now();
                return false;
            }
            context.tiled[shard][rank] = viewed.value;
        }
    }


    constexpr std::size_t block = 32U;
    constexpr auto intermediate_blocks = shard_intermediate / block;
    auto phase_started = std::chrono::steady_clock::now();
    context.result = run_expert_ranges(
        kTopK * intermediate_blocks,
        [&](std::size_t shard, std::uint64_t task) {
            const auto rank = static_cast<std::size_t>(
                task / intermediate_blocks);
            const auto offset = (task % intermediate_blocks) * block;
            std::array<float, block> gate{};
            std::array<float, block> up{};
            for (std::size_t half = 0U; half < 2U; ++half) {
                dsv4_tiled_expert_matvec16(
                    std::span<float, 16U>(
                        gate.data() + half * 16U, 16U),
                    context.input,
                    context.tiled[shard][rank].w13_packed,
                    context.tiled[shard][rank].w13_scales,
                    2U * shard_intermediate,
                    offset + half * 16U);
                dsv4_tiled_expert_matvec16(
                    std::span<float, 16U>(
                        up.data() + half * 16U, 16U),
                    context.input,
                    context.tiled[shard][rank].w13_packed,
                    context.tiled[shard][rank].w13_scales,
                    2U * shard_intermediate,
                    shard_intermediate + offset + half * 16U);
            }
            auto* destination = tiled_activation.data() +
                (shard * kTopK + rank) * shard_intermediate + offset;
            for (std::size_t index = 0U; index < block; ++index) {
                destination[index] = gate[index] /
                    (1.0F + std::exp(-gate[index])) * up[index];
            }
        }, true);
    device_moe_stats.routed_gate_up_nanoseconds +=
        elapsed_nanoseconds(phase_started);

    phase_started = std::chrono::steady_clock::now();
    if (context.result.ok()) {
        constexpr auto hidden_blocks = kHidden / block;
        context.result = run_expert_ranges(
            kTopK * hidden_blocks,
            [&](std::size_t shard, std::uint64_t task) {
                const auto rank = static_cast<std::size_t>(
                    task / hidden_blocks);
                const auto offset = (task % hidden_blocks) * block;
                const auto source = std::span<const float>(tiled_activation)
                    .subspan((shard * kTopK + rank) * shard_intermediate,
                             shard_intermediate);
                for (std::size_t half = 0U; half < 2U; ++half) {
                    auto destination = std::span<float, 16U>(
                        tiled_routed.data() +
                            (shard * kTopK + rank) * kHidden +
                            offset + half * 16U,
                        16U);
                    dsv4_tiled_expert_matvec16(
                        destination, source,
                        context.tiled[shard][rank].w2_packed,
                        context.tiled[shard][rank].w2_scales,
                        kHidden, offset + half * 16U);
                }
            }, true);
    }
    device_moe_stats.routed_down_nanoseconds +=
        elapsed_nanoseconds(phase_started);

    phase_started = std::chrono::steady_clock::now();
    if (context.result.ok()) {
        constexpr auto hidden_blocks = kHidden / block;
        context.result = run_expert_ranges(
            hidden_blocks,
            [&](std::size_t shard, std::uint64_t task) {
                const auto offset = task * block;
                auto* destination =
                    rank_partials.data() + shard * kHidden + offset;
                for (std::size_t index = 0U; index < block; ++index) {
                    destination[index] = tiled_routed[
                        (shard * kTopK) * kHidden + offset + index] *
                        context.route.weights[0];
                }
                for (std::size_t rank = 1U; rank < kTopK; ++rank) {
                    const auto* source = tiled_routed.data() +
                        (shard * kTopK + rank) * kHidden + offset;
                    for (std::size_t index = 0U; index < block; ++index) {
                        destination[index] = std::fma(
                            source[index], context.route.weights[rank],
                            destination[index]);
                    }
                }
            }, false);
    }
    device_moe_stats.routed_reduce_nanoseconds +=
        elapsed_nanoseconds(phase_started);
    device_moe_stats.routed_cpu_nanoseconds +=
        elapsed_nanoseconds(routed_started);
    context.callback_finished = std::chrono::steady_clock::now();
    context.accepted = context.result.ok();
    return context.accepted;
}

DeepSeekV4Runtime::Impl::PhysicalAttentionContext&
DeepSeekV4Runtime::Impl::physical_attention_context(std::size_t index) {
    while (physical_attention_contexts.size() <= index) {
        physical_attention_contexts.push_back(
            std::make_unique<PhysicalAttentionContext>());
    }
    return *physical_attention_contexts[index];
}

DeepSeekV4Runtime::Impl::HostMoeContext&
DeepSeekV4Runtime::Impl::host_moe_context(std::size_t index) {
    while (host_moe_contexts.size() <= index) {
        host_moe_contexts.push_back(std::make_unique<HostMoeContext>());
    }
    return *host_moe_contexts[index];
}

ValidationResult DeepSeekV4Runtime::Impl::collect_host_routed_moe_chain() {
    ValidationResult result;
    if (host_moe_pending == 0U) return result;
    const auto collect_started = std::chrono::steady_clock::now();
    auto collected = cuda.collect_deepseek_moe(
        devices[mhc_slot], {}, {});
    if (!collected.ok()) {
        append_errors(result, std::move(collected.errors),
                      "DeepSeek fixed CPU/shared MoE collect");
    }
    for (std::uint32_t pending = 0U; pending < host_moe_pending; ++pending) {
        auto& context = host_moe_context(pending);
        if (context.invoked) ++device_moe_stats.host_callback_batches;
        if (!context.invoked || !context.accepted) {
            ++device_moe_stats.host_callback_failures;
        }
        if (!context.result.ok()) {
            append_errors(result, std::move(context.result.errors));
        }
        ++device_moe_stats.batches;
        ++device_moe_stats.device_join_batches;
        ++device_moe_stats.device_commands;
        device_moe_stats.routed_experts += kTopK;
        ++device_moe_stats.shared_experts;
    }
    device_moe_stats.shared_collect_nanoseconds +=
        elapsed_nanoseconds(collect_started);
    device_moe_stats.nanoseconds += elapsed_nanoseconds(collect_started) +
        (device_moe_stats.routed_cpu_nanoseconds -
         host_moe_routed_cpu_before);
    host_moe_pending = 0U;
    host_moe_routed_cpu_before = 0U;
    pending_attention_leases.clear();
    pending_attention_weights.clear();
    for (auto& entry : host_moe_contexts) {
        auto& context = *entry;
        context.shared = {};
        for (auto& lease : context.shared_leases) {
            lease = Dsv4WeightCache::Lease{};
        }
    }
    for (auto& entry : physical_attention_contexts) {
        auto& context = *entry;
        if (!context.result.ok()) {
            append_errors(
                result, std::move(context.result.errors),
                "DeepSeek physical attention callback layer " +
                    std::to_string(context.layer));
        }
        if ((context.sliding_append.has_value() ||
             context.compressed_append.has_value() ||
             context.index_append.has_value()) &&
            !context.invoked) {
            result.errors.emplace_back(
                "DeepSeek physical attention callback was not invoked for layer " +
                std::to_string(context.layer));
        }
        const auto account = [&](auto& append) {
            if (!append.has_value()) return;
            auto accounted = append->account();
            if (!accounted.ok()) {
                append_errors(
                    result, std::move(accounted.errors),
                    "DeepSeek physical attention account layer " +
                        std::to_string(context.layer));
            }
        };
        account(context.sliding_append);
        account(context.compressed_append);
        account(context.index_append);
        context.sliding_append.reset();
        context.compressed_append.reset();
        context.index_append.reset();
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::host_routed_moe_impl(
    std::uint32_t layer, const Dsv4Route* route,
    std::span<const float> input, std::span<float> output,
    std::uint32_t token, std::uint32_t token_position) {
    ValidationResult result;
    constexpr std::size_t shards = 2U;
    constexpr std::size_t shard_intermediate = kExpertIntermediate / shards;
    const bool persistent_device_branch =
        config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice;
    const bool device_input = route == nullptr;
    if (layer >= kLayers ||
        (device_input ? !input.empty() : input.size() != kHidden) ||
        (output.size() != kHidden &&
         !(persistent_device_branch && output.empty())) ||
        (!device_input &&
         (route->experts.size() != kTopK ||
          route->weights.size() != kTopK)) ||
        (device_input && !persistent_device_branch) ||
        expert_workers == nullptr || expert_lane_nodes.size() != expert_workers->size()) {
        result.errors.emplace_back("DeepSeek host-routed MoE state is invalid");
        return result;
    }

    const auto prepare_started = std::chrono::steady_clock::now();
    std::array<std::array<Dsv4TiledExpertWeights, kTopK>, shards> tiled{};
    const auto resolve_tiled = [&](const Dsv4Route& active_route) {
        for (std::size_t shard = 0U; shard < shards; ++shard) {
            for (std::size_t rank = 0U; rank < kTopK; ++rank) {
                if (active_route.experts[rank] >= kExperts ||
                    !std::isfinite(active_route.weights[rank])) {
                    result.errors.emplace_back(
                        "DeepSeek host-routed MoE route is invalid");
                    return false;
                }
                auto viewed = dsv4_tiled_expert_weights(
                    resident.find_tiled_expert(
                        layer, active_route.experts[rank],
                        static_cast<std::uint32_t>(shard)),
                    kHidden, kExpertIntermediate, shards);
                if (!viewed.ok()) {
                    append_errors(result, std::move(viewed.errors));
                    return false;
                }
                tiled[shard][rank] = viewed.value;
            }
        }
        return true;
    };
    if (!device_input && !resolve_tiled(*route)) {
        if (result.ok()) {
                result.errors.emplace_back(
                    "DeepSeek host-routed MoE route is invalid");
        }
        return result;
    }

    const auto shared_slot = persistent_device_branch
        ? mhc_slot : layer_device(layer);
    std::array<Dsv4WeightCache::Lease, 3U> shared_leases;
    CudaDeepSeekMoeExpert shared;
    shared.coefficient = 1.0F;
    const auto shared_prefix = layer_prefix(layer) + "ffn.shared_experts.";
    const std::array<std::string, 3U> names{
        shared_prefix + "w1", shared_prefix + "w3", shared_prefix + "w2"};
    constexpr std::array<std::pair<std::uint64_t, std::uint64_t>, 3U> shapes{{
        {kExpertIntermediate, kHidden}, {kExpertIntermediate, kHidden},
        {kHidden, kExpertIntermediate}}};
    const CudaWeight** shared_weights[]{&shared.w1, &shared.w3, &shared.w2};
    for (std::size_t index = 0U; index < shared_leases.size(); ++index) {
        auto acquired = weights->acquire(shared_slot, names[index],
                                         shapes[index].first,
                                         shapes[index].second,
                                         shared_leases[index]);
        if (!acquired.ok()) {
            append_errors(result, std::move(acquired.errors), names[index]);
            return result;
        }
        *shared_weights[index] = &shared_leases[index].weight();
    }
    graph_stats.moe_prepare_nanoseconds += elapsed_nanoseconds(prepare_started);
    const auto execution_started = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point callback_finished{};
    bool callback_invoked = false;
    bool callback_accepted = false;
    std::vector<float> staged_input(device_input ? kHidden : 0U);
    Dsv4Route staged_route;
    auto active_input = input;
    const Dsv4Route* active_route = route;
    auto routed_callback = [&](std::span<float> rank_partials) {
    callback_invoked = true;
    const auto routed_started = std::chrono::steady_clock::now();
    if (rank_partials.size() != shards * kHidden) {
        result.errors.emplace_back(
            "DeepSeek CPU-MoE callback rank-partial shape is invalid");
        callback_finished = std::chrono::steady_clock::now();
        return false;
    }
    constexpr std::size_t block = 32U;
    constexpr auto intermediate_blocks = shard_intermediate / block;
    auto routed_phase_started = std::chrono::steady_clock::now();
    result = run_expert_ranges(kTopK * intermediate_blocks,
                        [&](std::size_t shard, std::uint64_t task) {
        const auto rank = static_cast<std::size_t>(task / intermediate_blocks);
        const auto offset = (task % intermediate_blocks) * block;
        std::array<float, block> gate{};
        std::array<float, block> up{};
        for (std::size_t half = 0U; half < 2U; ++half) {
            dsv4_tiled_expert_matvec16(
                std::span<float, 16U>(gate.data() + half * 16U, 16U),
                active_input,
                tiled[shard][rank].w13_packed,
                tiled[shard][rank].w13_scales, 2U * shard_intermediate,
                offset + half * 16U);
            dsv4_tiled_expert_matvec16(
                std::span<float, 16U>(up.data() + half * 16U, 16U),
                active_input,
                tiled[shard][rank].w13_packed,
                tiled[shard][rank].w13_scales, 2U * shard_intermediate,
                shard_intermediate + offset + half * 16U);
        }
        auto* destination = tiled_activation.data() +
            (shard * kTopK + rank) * shard_intermediate + offset;
        for (std::size_t index = 0U; index < block; ++index) {
            destination[index] = gate[index] /
                (1.0F + std::exp(-gate[index])) * up[index];
        }
    }, true);
    device_moe_stats.routed_gate_up_nanoseconds +=
        elapsed_nanoseconds(routed_phase_started);
    routed_phase_started = std::chrono::steady_clock::now();
    if (result.ok()) {
        constexpr auto hidden_blocks = kHidden / block;
        result = run_expert_ranges(kTopK * hidden_blocks,
                            [&](std::size_t shard, std::uint64_t task) {
            const auto rank = static_cast<std::size_t>(task / hidden_blocks);
            const auto offset = (task % hidden_blocks) * block;
            const auto source = std::span<const float>(tiled_activation)
                .subspan((shard * kTopK + rank) * shard_intermediate,
                         shard_intermediate);
            for (std::size_t half = 0U; half < 2U; ++half) {
                auto destination = std::span<float, 16U>(
                    tiled_routed.data() + (shard * kTopK + rank) * kHidden +
                        offset + half * 16U,
                    16U);
                dsv4_tiled_expert_matvec16(
                    destination, source, tiled[shard][rank].w2_packed,
                    tiled[shard][rank].w2_scales, kHidden,
                    offset + half * 16U);
            }
        }, true);
    }
    device_moe_stats.routed_down_nanoseconds +=
        elapsed_nanoseconds(routed_phase_started);
    routed_phase_started = std::chrono::steady_clock::now();
    if (result.ok()) {
        constexpr auto hidden_blocks = kHidden / block;
        result = run_expert_ranges(hidden_blocks,
                            [&](std::size_t shard, std::uint64_t task) {
            const auto offset = task * block;
            auto* destination = rank_partials.data() + shard * kHidden + offset;
            for (std::size_t index = 0U; index < block; ++index) {
                destination[index] = tiled_routed[
                    (shard * kTopK) * kHidden + offset + index] *
                    active_route->weights[0];
            }
            for (std::size_t rank = 1U; rank < kTopK; ++rank) {
                const auto* source = tiled_routed.data() +
                    (shard * kTopK + rank) * kHidden + offset;
                for (std::size_t index = 0U; index < block; ++index) {
                    destination[index] = std::fma(
                        source[index], active_route->weights[rank],
                        destination[index]);
                }
            }
        }, false);
    }
    device_moe_stats.routed_reduce_nanoseconds +=
        elapsed_nanoseconds(routed_phase_started);
    device_moe_stats.routed_cpu_nanoseconds +=
        elapsed_nanoseconds(routed_started);
    callback_finished = std::chrono::steady_clock::now();
    callback_accepted = result.ok();
    return callback_accepted;
    };

    using RoutedCallback = decltype(routed_callback);
    const auto invoke_routed = +[](void* opaque,
                                   std::span<float> rank_partials) {
        return (*static_cast<RoutedCallback*>(opaque))(rank_partials);
    };
    auto device_input_callback = [&nobreak = callback_invoked,
                                  &accepted = callback_accepted,
                                  &finished = callback_finished,
                                  &result, &staged_input, &staged_route,
                                  &active_input, &active_route, &resolve_tiled,
                                  &routed_callback, this, layer, token,
                                  token_position](
        std::span<const std::uint16_t> encoded_hidden,
        std::span<const float> router_logits,
        std::span<float> rank_partials) {
        nobreak = true;
        if (encoded_hidden.size() != kHidden ||
            router_logits.size() != kExperts) {
            result.errors.emplace_back(
                "DeepSeek device-input CPU-MoE callback shape is invalid");
            finished = std::chrono::steady_clock::now();
            accepted = false;
            return false;
        }
        for (std::size_t index = 0U; index < encoded_hidden.size(); ++index) {
            staged_input[index] = std::bit_cast<float>(
                static_cast<std::uint32_t>(encoded_hidden[index]) << 16U);
            if (!std::isfinite(staged_input[index])) {
                result.errors.emplace_back(
                    "DeepSeek device-input CPU-MoE hidden row is non-finite");
                finished = std::chrono::steady_clock::now();
                accepted = false;
                return false;
            }
        }
        const auto route_started = std::chrono::steady_clock::now();
        result = route_moe(
            layer, token, router_logits, token_position, staged_route);
        graph_stats.moe_router_nanoseconds += elapsed_nanoseconds(route_started);
        if (!result.ok() || !resolve_tiled(staged_route)) {
            finished = std::chrono::steady_clock::now();
            accepted = false;
            return false;
        }
        active_input = staged_input;
        active_route = &staged_route;
        return routed_callback(rank_partials);
    };
    using DeviceInputCallback = decltype(device_input_callback);
    const auto invoke_device_input = +[](
        void* opaque, std::span<const std::uint16_t> encoded_hidden,
        std::span<const float> router_logits,
        std::span<float> rank_partials) {
        return (*static_cast<DeviceInputCallback*>(opaque))(
            encoded_hidden, router_logits, rank_partials);
    };
    ValidationResult enqueued;
    if (device_input) {
        enqueued = cuda.enqueue_dsv4_host_moe_from_device_input(
            devices[shared_slot], shared,
            kDeepSeekV4ExecutionContract.swiglu_limit,
            invoke_device_input, &device_input_callback);
    } else if (persistent_device_branch) {
        enqueued = cuda.enqueue_dsv4_host_moe_from_mhc(
            devices[shared_slot], shared,
            kDeepSeekV4ExecutionContract.swiglu_limit,
            invoke_routed, &routed_callback);
    } else {
        enqueued = cuda.enqueue_dsv4_host_moe(
            devices[shared_slot], input, shared,
            kDeepSeekV4ExecutionContract.swiglu_limit,
            invoke_routed, &routed_callback);
    }
    if (!enqueued.ok()) {
        append_errors(result, std::move(enqueued.errors),
                      "DeepSeek CPU/shared MoE enqueue");
        return result;
    }

    std::vector<float> ignored_routed;
    const auto shared_started = std::chrono::steady_clock::now();
    auto device_output = persistent_device_branch &&
                         !config.enable_layer_hash_trace
        ? std::span<float>{} : output;
    auto collected = cuda.collect_deepseek_moe(
        devices[shared_slot], ignored_routed, device_output);
    const auto collected_finished = std::chrono::steady_clock::now();
    if (callback_finished != std::chrono::steady_clock::time_point{} &&
        collected_finished >= callback_finished) {
        device_moe_stats.shared_collect_nanoseconds +=
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    collected_finished - callback_finished).count());
    } else {
        device_moe_stats.shared_collect_nanoseconds +=
            elapsed_nanoseconds(shared_started);
    }
    if (callback_invoked) ++device_moe_stats.host_callback_batches;
    if (!callback_accepted) ++device_moe_stats.host_callback_failures;
    if (!collected.ok()) {
        append_errors(result, std::move(collected.errors),
                      "DeepSeek CPU/shared MoE collect");
    }
    if (!result.ok()) return result;

    ++device_moe_stats.batches;
    ++device_moe_stats.device_join_batches;
    ++device_moe_stats.device_commands;
    device_moe_stats.routed_experts += kTopK;
    ++device_moe_stats.shared_experts;
    device_moe_stats.nanoseconds += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - execution_started).count());
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::device_moe(
    std::uint32_t layer, const Dsv4Route& route,
    std::span<const float> input, std::span<float> output) {
    ValidationResult result;
    if (layer >= kLayers || input.size() != kHidden || output.size() != kHidden ||
        route.experts.size() != kTopK || route.weights.size() != kTopK) {
        result.errors.emplace_back("DeepSeek device MoE input or route shape is invalid");
        return result;
    }

    const auto prepare_started = std::chrono::steady_clock::now();
    struct RoutePlacement {
        std::size_t slot{};
        std::size_t local_rank{};
    };
    struct PendingDevice {
        std::vector<Dsv4WeightCache::Lease> leases;
        std::vector<CudaDeepSeekMoeExpert> routed;
        CudaDeepSeekMoeExpert shared;
        std::vector<float> routed_output;
        std::vector<float> shared_output;
        bool has_shared{};
        bool enqueued{};
    };

    std::vector<PendingDevice> pending(devices.size());
    for (auto& device : pending) {
        device.leases.reserve((kTopK + 1U) * 3U);
        device.routed.reserve(kTopK);
    }
    std::array<RoutePlacement, kTopK> placements{};

    const auto acquire_triplet = [this, &result](
        std::size_t slot, std::string_view prefix, float coefficient,
        PendingDevice& pending_device, CudaDeepSeekMoeExpert& descriptor) {
        descriptor.coefficient = coefficient;
        const auto acquire = [this, &result, slot, &pending_device](
            std::string name, std::uint64_t rows, std::uint64_t columns,
            const CudaWeight*& weight) {
            pending_device.leases.emplace_back();
            auto loaded = weights->acquire(slot, name, rows, columns,
                                           pending_device.leases.back());
            if (!loaded.ok()) {
                append_errors(result, std::move(loaded.errors), name);
                pending_device.leases.pop_back();
                return false;
            }
            weight = &pending_device.leases.back().weight();
            return true;
        };
        return acquire(std::string(prefix) + "w1", kExpertIntermediate,
                       kHidden, descriptor.w1) &&
               acquire(std::string(prefix) + "w3", kExpertIntermediate,
                       kHidden, descriptor.w3) &&
               acquire(std::string(prefix) + "w2", kHidden,
                       kExpertIntermediate, descriptor.w2);
    };

    // Every acquire below is a candidate demand transfer, and this layer's
    // experts are spread over all three devices. Batching lets those copies
    // run on their links concurrently instead of one at a time; the batch is
    // closed before the first MoE command is enqueued.
    auto upload_batch = config.serial_expert_upload
        ? Dsv4WeightCache::UploadBatch{}
        : weights->begin_upload_batch();

    const auto routed_prefix = layer_prefix(layer) + "ffn.experts.";
    for (std::size_t rank = 0U; rank < kTopK; ++rank) {
        const auto expert_id = route.experts[rank];
        if (expert_id >= kExperts || !std::isfinite(route.weights[rank])) {
            result.errors.emplace_back(
                "DeepSeek device MoE expert id or coefficient is invalid");
            return result;
        }
        const auto slot = expert_device(expert_id);
        auto& pending_device = pending[slot];
        placements[rank] = {slot, pending_device.routed.size()};
        CudaDeepSeekMoeExpert descriptor;
        const auto prefix = routed_prefix + std::to_string(expert_id) + ".";
        if (!acquire_triplet(slot, prefix, route.weights[rank],
                             pending_device, descriptor)) {
            return result;
        }
        pending_device.routed.push_back(descriptor);
    }

    const auto shared_slot = layer_device(layer);
    auto& shared_device = pending[shared_slot];
    const auto shared_prefix = layer_prefix(layer) + "ffn.shared_experts.";
    if (!acquire_triplet(shared_slot, shared_prefix, 1.0F,
                         shared_device, shared_device.shared)) {
        return result;
    }
    shared_device.has_shared = true;

    // Waits out the deferred copies, so every weight below is on its device
    // before any command reads it, and the wait is inside moe_prepare where
    // the serial version paid it.
    if (auto closed = upload_batch.close(); !closed.ok()) {
        append_errors(result, std::move(closed.errors),
                      "DeepSeek routed expert upload");
        return result;
    }

    graph_stats.moe_prepare_nanoseconds += elapsed_nanoseconds(prepare_started);
    const auto execution_started = std::chrono::steady_clock::now();
    const auto device_commands = static_cast<std::uint64_t>(std::count_if(
        pending.begin(), pending.end(), [](const auto& pending_device) {
            return !pending_device.routed.empty() || pending_device.has_shared;
        }));

    for (std::size_t slot = 0U; slot < pending.size(); ++slot) {
        auto& pending_device = pending[slot];
        if (pending_device.routed.empty() && !pending_device.has_shared) continue;
        pending_device.routed_output.resize(
            pending_device.routed.size() * kHidden);
        if (pending_device.has_shared) {
            pending_device.shared_output.resize(kHidden);
        }
        auto enqueued = cuda.enqueue_deepseek_moe(
            devices[slot], input, pending_device.routed,
            pending_device.has_shared ? &pending_device.shared : nullptr, 10.0F);
        if (!enqueued.ok()) {
            append_errors(result, std::move(enqueued.errors),
                          "DeepSeek device MoE enqueue");
            break;
        }
        pending_device.enqueued = true;
    }

    // Every accepted command must be observed before its cache leases leave
    // scope, including commands submitted before a later-device enqueue error.
    for (std::size_t slot = 0U; slot < pending.size(); ++slot) {
        auto& pending_device = pending[slot];
        if (!pending_device.enqueued) continue;
        auto collected = cuda.collect_deepseek_moe(
            devices[slot], pending_device.routed_output,
            pending_device.shared_output);
        pending_device.enqueued = false;
        if (!collected.ok()) {
            append_errors(result, std::move(collected.errors),
                          "DeepSeek device MoE collect");
        }
    }
    if (!result.ok()) return result;

    for (auto& pending_device : pending) {
        round_bf16(pending_device.routed_output);
        round_bf16(pending_device.shared_output);
    }
    std::fill(output.begin(), output.end(), 0.0F);
    for (std::size_t rank = 0U; rank < kTopK; ++rank) {
        const auto placement = placements[rank];
        const auto routed = std::span<const float>(
            pending[placement.slot].routed_output)
            .subspan(placement.local_rank * kHidden, kHidden);
        for (std::uint32_t column = 0U; column < kHidden; ++column) {
            output[column] += routed[column];
        }
    }
    for (std::uint32_t column = 0U; column < kHidden; ++column) {
        output[column] = round_bf16(
            output[column] + shared_device.shared_output[column]);
    }
    ++device_moe_stats.batches;
    device_moe_stats.device_commands += device_commands;
    device_moe_stats.routed_experts += kTopK;
    ++device_moe_stats.shared_experts;
    device_moe_stats.nanoseconds += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - execution_started).count());
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::route_moe(
    std::uint32_t layer, std::uint32_t token, std::span<const float> logits,
    std::uint32_t position, Dsv4Route& output) {
    ValidationResult result;
    const auto prefix = layer_prefix(layer) + "ffn.";
    const auto& router = deepseek_v4_flash_0731_spec().router;
    Dsv4RouteResult route;
    if (layer < 3U) {
        const auto name = prefix + "gate.tid2eid";
        const auto found = host_raw.find(name);
        if (found == host_raw.end()) {
            result.errors.emplace_back("DeepSeek resident hash-routing table is absent");
            return result;
        }
        const auto row_bytes = static_cast<std::size_t>(kTopK) * sizeof(std::int64_t);
        const auto offset = static_cast<std::size_t>(token) * row_bytes;
        if (token >= kVocabulary || found->second.size() < offset + row_bytes) {
            result.errors.emplace_back("DeepSeek hash-routing token row is out of range");
            return result;
        }
        std::array<std::uint32_t, kTopK> selected{};
        for (std::uint32_t rank = 0U; rank < kTopK; ++rank) {
            std::int64_t encoded = 0;
            std::memcpy(&encoded,
                        found->second.data() + offset + rank * sizeof(encoded),
                        sizeof(encoded));
            if (encoded < 0 || encoded >= static_cast<std::int64_t>(kExperts)) {
                result.errors.emplace_back("DeepSeek hash-routing expert is invalid");
                return result;
            }
            selected[rank] = static_cast<std::uint32_t>(encoded);
        }
        route = dsv4_route_hash_sqrtsoftplus_f32(logits, selected, router);
    } else {
        auto bias = host_tensor(prefix + "gate.bias", kExperts);
        if (!bias.ok()) {
            append_errors(result, std::move(bias.errors));
            return result;
        }
        route = dsv4_route_sqrtsoftplus_f32(logits, *bias.value, router);
    }
    if (!route.ok()) {
        append_errors(result, std::move(route.errors));
        return result;
    }
    if (config.enable_layer_hash_trace) {
        record_operation_hash(position, token, layer, "ffn_router_weights", route.value.weights);
    }
    const bool prefetch_enabled = config.expert_prefetch_predictions != 0U;
    // A speculative pass is rolled back, so its routes are not part of the
    // sequence. Recording them would put tokens that were never emitted into
    // the trace and teach the predictor a history that did not happen.
    if (!speculative_pass && (route_trace.is_open() || prefetch_enabled)) {
        RouteEvent event;
        event.request = active_request_id;
        event.token_position = position;
        event.layer = layer;
        event.experts = route.value.experts;
        event.coefficients = route.value.weights;
        event.phase = position < active_prompt_tokens
                          ? RoutePhase::Prefill : RoutePhase::Decode;
        if (defer_prefill_observability && event.phase == RoutePhase::Prefill) {
            deferred_route_events.push_back(std::move(event));
        } else {
            if (prefetch_enabled) {
                route_predictor.observe(event);
                if (event.phase == RoutePhase::Decode) {
                    pending_prefetch_predictions = route_predictor.predict(
                        event, config.expert_prefetch_predictions,
                        config.expert_prefetch_minimum_confidence);
                }
            }
            if (route_trace.is_open()) {
                auto written = route_trace.write(event);
                if (!written.ok()) return written;
            }
        }
    }
    output = std::move(route.value);
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::execute_moe(
    std::uint32_t layer, const Dsv4Route& route,
    std::span<const float> input, std::span<float> output) {
    ValidationResult result;
    if (config.enable_host_routed_moe) {
        return host_routed_moe(layer, route, input, output);
    }
    const auto prefix = layer_prefix(layer) + "ffn.";
    std::vector<ExpertKey> demand_keys;
    demand_keys.reserve(route.experts.size());
    for (const auto expert_id : route.experts) {
        demand_keys.push_back(ExpertKey{layer, expert_id});
    }
    auto demand_guard = weights->demand(demand_keys);
    const auto schedule_prefetch = [this] {
        for (const auto& prediction : pending_prefetch_predictions) {
            weights->request_prefetch(
                prediction.key, expert_device(prediction.key.expert));
        }
        pending_prefetch_predictions.clear();
    };

    if (config.enable_device_moe) {
        result = device_moe(layer, route, input, output);
        if (result.ok()) schedule_prefetch();
        return result;
    }

    std::fill(output.begin(), output.end(), 0.0F);
    std::vector<float> routed(kHidden);
    for (std::size_t rank = 0U; rank < route.experts.size(); ++rank) {
        result = expert(layer, route.experts[rank], route.weights[rank], input,
                        routed);
        if (!result.ok()) return result;
        for (std::uint32_t column = 0U; column < kHidden; ++column) {
            output[column] += routed[column];
        }
    }

    const auto slot = layer_device(layer);
    std::vector<float> shared_gate(kExpertIntermediate);
    std::vector<float> shared_up(kExpertIntermediate);
    std::vector<float> shared_activated(kExpertIntermediate);
    std::vector<float> shared_output(kHidden);
    result = linear(slot, prefix + "shared_experts.w1", kExpertIntermediate,
                    kHidden, input, shared_gate);
    if (!result.ok()) return result;
    result = linear(slot, prefix + "shared_experts.w3", kExpertIntermediate,
                    kHidden, input, shared_up);
    if (!result.ok()) return result;
    result = dsv4_swiglu_f32(shared_activated, shared_gate, shared_up, 10.0F);
    if (!result.ok()) return result;
    round_bf16(shared_activated);
    result = linear(slot, prefix + "shared_experts.w2", kHidden,
                    kExpertIntermediate, shared_activated, shared_output);
    if (!result.ok()) return result;
    for (std::uint32_t column = 0U; column < kHidden; ++column) {
        output[column] = round_bf16(output[column] + shared_output[column]);
    }
    schedule_prefetch();
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::moe(
    std::uint32_t layer, std::uint32_t token, std::span<const float> input,
    std::span<float> output, std::uint32_t position) {
    ValidationResult result;
    const auto router_started = std::chrono::steady_clock::now();
    if (completed_router_projection && deferred_attention_moe_input) {
        completed_router_projection = false;
        deferred_attention_moe_input = false;
        return host_routed_moe_from_device_input(
            layer, token, position, output);
    }
    std::vector<float> logits(kExperts);
    if (completed_router_projection) {
        std::copy(combined_router_logits.begin(), combined_router_logits.end(),
                  logits.begin());
        completed_router_projection = false;
    } else {
        result = linear(layer_device(layer), layer_prefix(layer) + "ffn.gate",
                        kExperts, kHidden, input, logits, false);
        if (!result.ok()) return result;
    }
    Dsv4Route route;
    result = route_moe(layer, token, logits, position, route);
    graph_stats.moe_router_nanoseconds += elapsed_nanoseconds(router_started);
    if (!result.ok()) return result;
    return execute_moe(layer, route, input, output);
}

ValidationResult DeepSeekV4Runtime::Impl::moe_page(
    std::uint32_t layer, std::span<const std::uint32_t> tokens,
    std::span<const float> input, std::span<float> output,
    std::uint32_t position_base) {
    ValidationResult result;
    const auto rows = static_cast<std::uint32_t>(tokens.size());
    if (rows == 0U || input.size() != static_cast<std::size_t>(rows) * kHidden ||
        output.size() != input.size()) {
        result.errors.emplace_back("DeepSeek MoE page has incompatible dimensions");
        return result;
    }
    const auto router_started = std::chrono::steady_clock::now();
    std::vector<float> logits(static_cast<std::size_t>(rows) * kExperts);
    result = linear_rows(layer_device(layer), layer_prefix(layer) + "ffn.gate",
                         kExperts, kHidden, input, rows, logits, false,
                         nullptr, config.enable_dsv4_fp8_tensor_page);
    if (!result.ok()) return result;
    std::vector<Dsv4Route> routes(rows);
    for (std::uint32_t row = 0U; row < rows; ++row) {
        result = route_moe(
            layer, tokens[row],
            std::span<const float>(logits).subspan(
                static_cast<std::size_t>(row) * kExperts, kExperts),
            position_base + row, routes[row]);
        if (!result.ok()) return result;
    }
    graph_stats.moe_router_nanoseconds += elapsed_nanoseconds(router_started);
    if (rows == 1U || config.row_major_moe_page) {
        for (std::uint32_t row = 0U; row < rows; ++row) {
            result = execute_moe(
                layer, routes[row],
                input.subspan(static_cast<std::size_t>(row) * kHidden, kHidden),
                output.subspan(static_cast<std::size_t>(row) * kHidden, kHidden));
            if (!result.ok()) return result;
        }
        return result;
    }
    return execute_moe_page(layer, routes, input, output);
}

// Row-grouped MoE for a prefill page. The single-row path reads a 13.37 MB
// expert triplet from HBM to serve one row, so a page of R rows reads a hot
// expert once for every row that chose it. Here each distinct expert is
// acquired and read once and applied to all its rows at once, which is the
// only structural difference between this engine's prefill and a CPU-hybrid
// stack that batches the same page 30x faster (experiment 0055).
//
// Per-row arithmetic is unchanged: every row still visits its own six experts
// with its own coefficients, and the final accumulation still runs in rank
// order, so output is bit-identical to looping execute_moe over the rows.
ValidationResult DeepSeekV4Runtime::Impl::execute_moe_page(
    std::uint32_t layer, std::span<const Dsv4Route> routes,
    std::span<const float> input, std::span<float> output) {
    ValidationResult result;
    const auto rows = static_cast<std::uint32_t>(routes.size());
    if (rows == 0U || input.size() != static_cast<std::size_t>(rows) * kHidden ||
        output.size() != input.size()) {
        result.errors.emplace_back("DeepSeek MoE page shape is invalid");
        return result;
    }

    const auto prepare_started = std::chrono::steady_clock::now();
    struct PendingDevice {
        std::vector<Dsv4WeightCache::Lease> leases;
        std::vector<CudaDeepSeekMoeRowGroup> groups;
        std::deque<std::vector<std::uint32_t>> group_rows;
        std::deque<std::vector<float>> group_coefficients;
        std::vector<std::uint32_t> group_offsets;
        std::unordered_map<std::uint32_t, std::size_t> expert_slot;
        CudaDeepSeekMoeExpert shared;
        std::vector<std::uint32_t> shared_rows;
        std::vector<float> routed_output;
        std::vector<float> shared_output;
        std::uint32_t work_count{};
        bool has_shared{};
        bool enqueued{};
    };
    struct Placement {
        std::uint32_t slot{};
        std::uint32_t group{};
        std::uint32_t position{};
    };

    std::vector<PendingDevice> pending(devices.size());
    std::vector<Placement> placements(
        static_cast<std::size_t>(rows) * kTopK);

    const auto acquire_triplet = [this, &result](
        std::size_t slot, std::string_view prefix,
        PendingDevice& pending_device, CudaDeepSeekMoeExpert& descriptor) {
        descriptor.coefficient = 1.0F;
        const auto acquire = [this, &result, slot, &pending_device](
            std::string name, std::uint64_t weight_rows,
            std::uint64_t weight_columns, const CudaWeight*& weight) {
            pending_device.leases.emplace_back();
            auto loaded = weights->acquire(slot, name, weight_rows,
                                           weight_columns,
                                           pending_device.leases.back());
            if (!loaded.ok()) {
                append_errors(result, std::move(loaded.errors), name);
                pending_device.leases.pop_back();
                return false;
            }
            weight = &pending_device.leases.back().weight();
            return true;
        };
        return acquire(std::string(prefix) + "w1", kExpertIntermediate,
                       kHidden, descriptor.w1) &&
               acquire(std::string(prefix) + "w3", kExpertIntermediate,
                       kHidden, descriptor.w3) &&
               acquire(std::string(prefix) + "w2", kHidden,
                       kExpertIntermediate, descriptor.w2);
    };

    auto upload_batch = config.serial_expert_upload
        ? Dsv4WeightCache::UploadBatch{}
        : weights->begin_upload_batch();

    const auto routed_prefix = layer_prefix(layer) + "ffn.experts.";
    for (std::uint32_t row = 0U; row < rows; ++row) {
        const auto& route = routes[row];
        if (route.experts.size() != kTopK || route.weights.size() != kTopK) {
            result.errors.emplace_back("DeepSeek MoE page route shape is invalid");
            return result;
        }
        for (std::size_t rank = 0U; rank < kTopK; ++rank) {
            const auto expert_id = route.experts[rank];
            if (expert_id >= kExperts || !std::isfinite(route.weights[rank])) {
                result.errors.emplace_back(
                    "DeepSeek MoE page expert id or coefficient is invalid");
                return result;
            }
            const auto slot = expert_device(expert_id);
            auto& pending_device = pending[slot];
            auto found = pending_device.expert_slot.find(expert_id);
            if (found == pending_device.expert_slot.end()) {
                CudaDeepSeekMoeRowGroup group;
                if (resident.tiled_experts()) {
                    // The transformed shards are the resident copy, so this
                    // uploads them as they stand. Reading the canonical
                    // triplet instead means re-reading the checkpoint for
                    // weights that are already in host memory.
                    for (std::uint32_t shard = 0U;
                         shard < kResidentExpertShards; ++shard) {
                        pending_device.leases.emplace_back();
                        auto acquired = weights->acquire_tiled_expert(
                            slot, layer, expert_id, shard,
                            pending_device.leases.back());
                        if (!acquired.ok()) {
                            append_errors(result, std::move(acquired.errors),
                                          "DeepSeek transformed expert shard");
                            pending_device.leases.pop_back();
                            return result;
                        }
                        group.tiled_shards[shard] =
                            &pending_device.leases.back().weight();
                    }
                } else {
                    CudaDeepSeekMoeExpert descriptor;
                    const auto prefix =
                        routed_prefix + std::to_string(expert_id) + ".";
                    if (!acquire_triplet(slot, prefix, pending_device,
                                         descriptor)) {
                        return result;
                    }
                    group.w1 = descriptor.w1;
                    group.w3 = descriptor.w3;
                    group.w2 = descriptor.w2;
                }
                const auto group_index = pending_device.groups.size();
                pending_device.group_rows.emplace_back();
                pending_device.group_coefficients.emplace_back();
                pending_device.groups.push_back(group);
                found = pending_device.expert_slot
                            .emplace(expert_id, group_index).first;
            }
            auto& group_rows = pending_device.group_rows[found->second];
            auto& group_coefficients =
                pending_device.group_coefficients[found->second];
            placements[static_cast<std::size_t>(row) * kTopK + rank] = {
                static_cast<std::uint32_t>(slot),
                static_cast<std::uint32_t>(found->second),
                static_cast<std::uint32_t>(group_rows.size())};
            group_rows.push_back(row);
            group_coefficients.push_back(route.weights[rank]);
        }
    }

    const auto shared_slot = layer_device(layer);
    auto& shared_device = pending[shared_slot];
    const auto shared_prefix = layer_prefix(layer) + "ffn.shared_experts.";
    if (!acquire_triplet(shared_slot, shared_prefix, shared_device,
                         shared_device.shared)) {
        return result;
    }
    shared_device.has_shared = true;
    shared_device.shared_rows.resize(rows);
    std::iota(shared_device.shared_rows.begin(),
              shared_device.shared_rows.end(), 0U);

    if (auto closed = upload_batch.close(); !closed.ok()) {
        append_errors(result, std::move(closed.errors),
                      "DeepSeek routed expert page upload");
        return result;
    }

    graph_stats.moe_prepare_nanoseconds += elapsed_nanoseconds(prepare_started);
    const auto execution_started = std::chrono::steady_clock::now();
    std::uint64_t device_commands = 0U;
    for (auto& pending_device : pending) {
        if (pending_device.groups.empty() && !pending_device.has_shared) continue;
        std::uint32_t offset = 0U;
        pending_device.group_offsets.reserve(pending_device.groups.size());
        for (std::size_t index = 0U; index < pending_device.groups.size();
             ++index) {
            pending_device.group_offsets.push_back(offset);
            pending_device.groups[index].rows =
                pending_device.group_rows[index];
            pending_device.groups[index].coefficients =
                pending_device.group_coefficients[index];
            offset += static_cast<std::uint32_t>(
                pending_device.group_rows[index].size());
        }
        pending_device.work_count = offset;
        pending_device.routed_output.resize(
            static_cast<std::size_t>(offset) * kHidden);
        if (pending_device.has_shared) {
            pending_device.shared_output.resize(
                static_cast<std::size_t>(rows) * kHidden);
        }
        ++device_commands;
    }

    for (std::size_t slot = 0U; slot < pending.size(); ++slot) {
        auto& pending_device = pending[slot];
        if (pending_device.groups.empty() && !pending_device.has_shared) continue;
        auto enqueued = cuda.enqueue_deepseek_moe_rows(
            devices[slot], input, rows, pending_device.groups,
            pending_device.has_shared ? &pending_device.shared : nullptr,
            pending_device.shared_rows, 10.0F);
        if (!enqueued.ok()) {
            append_errors(result, std::move(enqueued.errors),
                          "DeepSeek device MoE page enqueue");
            break;
        }
        pending_device.enqueued = true;
    }

    for (std::size_t slot = 0U; slot < pending.size(); ++slot) {
        auto& pending_device = pending[slot];
        if (!pending_device.enqueued) continue;
        auto collected = cuda.collect_deepseek_moe_rows(
            devices[slot], pending_device.routed_output,
            pending_device.shared_output);
        pending_device.enqueued = false;
        if (!collected.ok()) {
            append_errors(result, std::move(collected.errors),
                          "DeepSeek device MoE page collect");
        }
    }
    if (!result.ok()) return result;

    for (auto& pending_device : pending) {
        round_bf16(pending_device.routed_output);
        round_bf16(pending_device.shared_output);
    }
    std::fill(output.begin(), output.end(), 0.0F);
    for (std::uint32_t row = 0U; row < rows; ++row) {
        auto output_row = output.subspan(
            static_cast<std::size_t>(row) * kHidden, kHidden);
        for (std::size_t rank = 0U; rank < kTopK; ++rank) {
            const auto placement =
                placements[static_cast<std::size_t>(row) * kTopK + rank];
            const auto& pending_device = pending[placement.slot];
            const auto work_index =
                pending_device.group_offsets[placement.group] +
                placement.position;
            const auto routed = std::span<const float>(
                pending_device.routed_output)
                .subspan(static_cast<std::size_t>(work_index) * kHidden, kHidden);
            for (std::uint32_t column = 0U; column < kHidden; ++column) {
                output_row[column] += routed[column];
            }
        }
        const auto shared = std::span<const float>(shared_device.shared_output)
            .subspan(static_cast<std::size_t>(row) * kHidden, kHidden);
        for (std::uint32_t column = 0U; column < kHidden; ++column) {
            output_row[column] = round_bf16(output_row[column] + shared[column]);
        }
    }

    ++device_moe_stats.batches;
    device_moe_stats.device_commands += device_commands;
    device_moe_stats.routed_experts +=
        static_cast<std::uint64_t>(rows) * kTopK;
    device_moe_stats.shared_experts += rows;
    device_moe_stats.nanoseconds += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - execution_started).count());
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::block(
    std::uint32_t layer, std::uint32_t token, std::span<float> hidden,
    std::uint32_t position) {
    ValidationResult result;
    if (hidden.size() != static_cast<std::size_t>(kMhc) * kHidden) {
        result.errors.emplace_back("DeepSeek mHC hidden state has the wrong shape");
        return result;
    }
    const auto prefix = layer_prefix(layer);
    for (const auto* branch_name : {"attn", "ffn"}) {
        const std::string branch(branch_name);
        const auto projection_name = prefix + "hc_" + branch + "_fn";
        auto projection = host_tensor(projection_name,
                                      kMix * kMhc * kHidden);
        auto scale = host_tensor(prefix + "hc_" + branch + "_scale", 3U);
        auto base = host_tensor(prefix + "hc_" + branch + "_base", kMix);
        if (!projection.ok()) append_errors(result, std::move(projection.errors));
        if (!scale.ok()) append_errors(result, std::move(scale.errors));
        if (!base.ok()) append_errors(result, std::move(base.errors));
        if (!result.ok()) return result;
        const std::vector<float> residual(hidden.begin(), hidden.end());
        std::vector<float> reduced(kHidden);
        Dsv4MhcMix mix;
        auto phase_started = std::chrono::steady_clock::now();
        result = mhc_pre(reduced, mix, residual, projection_name,
                         *projection.value, *scale.value, *base.value, true);
        graph_stats.mhc_pre_nanoseconds += elapsed_nanoseconds(phase_started);
        if (!result.ok()) return result;
        round_bf16(reduced);
        if (config.enable_layer_hash_trace) {
            record_operation_hash(position, token, layer, branch + "_mhc_pre", reduced);
        }
        phase_started = std::chrono::steady_clock::now();
        result = norm(reduced, reduced, prefix + branch + "_norm.weight");
        graph_stats.branch_norm_nanoseconds += elapsed_nanoseconds(phase_started);
        if (!result.ok()) return result;
        if (config.enable_layer_hash_trace) {
            record_operation_hash(position, token, layer, branch + "_norm", reduced);
        }
        std::vector<float> branch_output(kHidden);
        phase_started = std::chrono::steady_clock::now();
        if (branch == "attn") {
            result = attention(layer, reduced, position, branch_output);
            graph_stats.attention_nanoseconds += elapsed_nanoseconds(phase_started);
        } else {
            result = moe(layer, token, reduced, branch_output, position);
            graph_stats.moe_nanoseconds += elapsed_nanoseconds(phase_started);
        }
        if (!result.ok()) return result;
        if (config.enable_layer_hash_trace) {
            record_operation_hash(position, token, layer, branch + "_output", branch_output);
        }
        phase_started = std::chrono::steady_clock::now();
        result = dsv4_mhc_post_f32(hidden, branch_output, residual, mix);
        graph_stats.mhc_post_nanoseconds += elapsed_nanoseconds(phase_started);
        if (!result.ok()) return result;
        round_bf16(hidden);
        if (config.enable_layer_hash_trace) {
            record_operation_hash(position, token, layer, branch + "_mhc_post", hidden);
        }
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::block_page(
    std::uint32_t layer, std::span<const std::uint32_t> tokens,
    std::span<float> hidden, std::uint32_t position_base) {
    ValidationResult result;
    const auto rows = tokens.size();
    const auto hidden_stride = static_cast<std::size_t>(kMhc) * kHidden;
    if (rows == 0U || rows > config.maximum_context_tokens ||
        hidden.size() != rows * hidden_stride ||
        position_base > config.maximum_context_tokens - rows) {
        result.errors.emplace_back(
            "DeepSeek prefill page has incompatible dimensions");
        return result;
    }
    const auto prefix = layer_prefix(layer);
    for (const auto* branch_name : {"attn", "ffn"}) {
        const std::string branch(branch_name);
        const auto projection_name = prefix + "hc_" + branch + "_fn";
        auto projection = host_tensor(projection_name,
                                      kMix * kMhc * kHidden);
        auto scale = host_tensor(prefix + "hc_" + branch + "_scale", 3U);
        auto base = host_tensor(prefix + "hc_" + branch + "_base", kMix);
        if (!projection.ok()) append_errors(result, std::move(projection.errors));
        if (!scale.ok()) append_errors(result, std::move(scale.errors));
        if (!base.ok()) append_errors(result, std::move(base.errors));
        if (!result.ok()) return result;

        const std::vector<float> residual(hidden.begin(), hidden.end());
        std::vector<float> reduced(rows * kHidden);
        std::vector<Dsv4MhcMix> mixes(rows);
        for (std::size_t row = 0U; row < rows; ++row) {
            const auto position = position_base + static_cast<std::uint32_t>(row);
            auto reduced_row = std::span<float>(reduced).subspan(row * kHidden,
                                                                 kHidden);
            const auto residual_row = std::span<const float>(residual).subspan(
                row * hidden_stride, hidden_stride);
            auto phase_started = std::chrono::steady_clock::now();
            result = mhc_pre(reduced_row, mixes[row], residual_row,
                             projection_name, *projection.value,
                             *scale.value, *base.value);
            graph_stats.mhc_pre_nanoseconds +=
                elapsed_nanoseconds(phase_started);
            if (!result.ok()) return result;
            round_bf16(reduced_row);
            if (config.enable_layer_hash_trace) {
                record_operation_hash(position, tokens[row], layer,
                                      branch + "_mhc_pre", reduced_row);
            }
            phase_started = std::chrono::steady_clock::now();
            result = norm(reduced_row, reduced_row,
                          prefix + branch + "_norm.weight");
            graph_stats.branch_norm_nanoseconds +=
                elapsed_nanoseconds(phase_started);
            if (!result.ok()) return result;
            if (config.enable_layer_hash_trace) {
                record_operation_hash(position, tokens[row], layer,
                                      branch + "_norm", reduced_row);
            }
        }

        std::vector<float> branch_output(rows * kHidden);
        if (branch == "attn") {
            const auto phase_started = std::chrono::steady_clock::now();
            result = attention_page(layer, reduced, position_base,
                                    branch_output);
            graph_stats.attention_nanoseconds +=
                elapsed_nanoseconds(phase_started);
            if (!result.ok()) return result;
        } else {
            const auto phase_started = std::chrono::steady_clock::now();
            result = moe_page(layer, tokens, reduced, branch_output,
                              position_base);
            graph_stats.moe_nanoseconds += elapsed_nanoseconds(phase_started);
            if (!result.ok()) return result;
        }
        if (config.enable_layer_hash_trace) {
            for (std::size_t row = 0U; row < rows; ++row) {
                const auto position = position_base +
                                      static_cast<std::uint32_t>(row);
                const auto output_row = std::span<const float>(branch_output)
                    .subspan(row * kHidden, kHidden);
                record_operation_hash(position, tokens[row], layer,
                                      branch + "_output", output_row);
            }
        }

        for (std::size_t row = 0U; row < rows; ++row) {
            const auto position = position_base + static_cast<std::uint32_t>(row);
            auto hidden_row = hidden.subspan(row * hidden_stride, hidden_stride);
            const auto output_row = std::span<const float>(branch_output).subspan(
                row * kHidden, kHidden);
            const auto residual_row = std::span<const float>(residual).subspan(
                row * hidden_stride, hidden_stride);
            const auto phase_started = std::chrono::steady_clock::now();
            result = dsv4_mhc_post_f32(hidden_row, output_row, residual_row,
                                       mixes[row]);
            graph_stats.mhc_post_nanoseconds +=
                elapsed_nanoseconds(phase_started);
            if (!result.ok()) return result;
            round_bf16(hidden_row);
            if (config.enable_layer_hash_trace) {
                record_operation_hash(position, tokens[row], layer,
                                      branch + "_mhc_post", hidden_row);
            }
        }
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::forward_hidden(
    std::uint32_t token, std::uint32_t position, std::span<float> hidden,
    std::vector<float>* fused_logits) {
    const auto embedding_started = std::chrono::steady_clock::now();
    auto result = embed(token, hidden);
    graph_stats.embedding_nanoseconds += elapsed_nanoseconds(embedding_started);
    if (!result.ok()) return result;
    if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
        if (rank_local_active && position >= active_prompt_tokens) {
            // Decode, and rank-local was admitted. Prefill stays centralized:
            // the rank-local set is a decode-shaped ownership of the weights,
            // and admission accounts for both being resident.
            return rank_local_forward_hidden(
                token, position, hidden, fused_logits);
        }
        return device_mhc_forward_hidden(
            token, position, hidden, fused_logits);
    }
    for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
        result = block(layer, token, hidden, position);
        if (!result.ok()) return result;
        if (config.enable_layer_hash_trace) {
            record_layer_hash(position, token, layer, hidden);
        }
    }
    return result;
}

bool DeepSeekV4Runtime::Impl::device_head_callback(
    void* opaque, std::span<const std::uint16_t> encoded_hidden,
    std::span<float> reduced) {
    if (opaque == nullptr) return false;
    auto& context = *static_cast<DeviceHeadContext*>(opaque);
    if (context.owner == nullptr) return false;
    return context.owner->execute_device_head_callback(
        context, encoded_hidden, reduced);
}

bool DeepSeekV4Runtime::Impl::execute_device_head_callback(
    DeviceHeadContext& context,
    std::span<const std::uint16_t> encoded_hidden,
    std::span<float> reduced) {
    context.invoked = true;
    context.result = {};
    const auto projection = host_tensors.find("hc_head_fn");
    const auto scale = host_tensors.find("hc_head_scale");
    const auto base = host_tensors.find("hc_head_base");
    const auto norm_weight = host_tensors.find("norm.weight");
    if (encoded_hidden.size() != context.hidden.size() ||
        reduced.size() != kHidden || projection == host_tensors.end() ||
        scale == host_tensors.end() || base == host_tensors.end() ||
        norm_weight == host_tensors.end()) {
        context.result.errors.emplace_back(
            "DeepSeek deferred output-head inputs are unavailable");
        return false;
    }
    for (std::size_t index = 0U; index < context.hidden.size(); ++index) {
        context.hidden[index] = std::bit_cast<float>(
            static_cast<std::uint32_t>(encoded_hidden[index]) << 16U);
        if (!std::isfinite(context.hidden[index])) {
            context.result.errors.emplace_back(
                "DeepSeek deferred output-head hidden state is non-finite");
            return false;
        }
    }
    double square_sum = 0.0;
    for (const float value : context.hidden) {
        square_sum += static_cast<double>(value) * value;
    }
    const float reciprocal = 1.0F / std::sqrt(
        static_cast<float>(square_sum /
                           static_cast<double>(context.hidden.size())) +
        kRmsEpsilon);
    std::fill(reduced.begin(), reduced.end(), 0.0F);
    for (std::uint32_t copy = 0U; copy < kMhc; ++copy) {
        double projected = 0.0;
        const auto row = static_cast<std::size_t>(copy) *
                         context.hidden.size();
        for (std::size_t column = 0U; column < context.hidden.size(); ++column) {
            projected += static_cast<double>(
                projection->second[row + column]) * context.hidden[column];
        }
        const float coefficient = sigmoid_f32(
            static_cast<float>(projected) * reciprocal * scale->second[0] +
            base->second[copy]) + kRmsEpsilon;
        for (std::uint32_t column = 0U; column < kHidden; ++column) {
            reduced[column] += coefficient * context.hidden[
                static_cast<std::size_t>(copy) * kHidden + column];
        }
    }
    round_bf16(reduced);
    context.result = rms_norm_f32(
        reduced, reduced, norm_weight->second, kRmsEpsilon);
    if (context.result.ok()) round_bf16(reduced);
    return context.result.ok();
}

ValidationResult DeepSeekV4Runtime::Impl::admit_rank_local() {
    ValidationResult result;
    rank_local_active = false;
    if (checkpoint == nullptr || weights == nullptr || kv_cache == nullptr) {
        result.errors.emplace_back(
            "rank-local decode requires a loaded checkpoint, weight arena and "
            "physical KV cache");
        return result;
    }
    std::array<int, kDsv4RankLocalWorld> rank_devices{};
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        rank_devices[rank] = devices[rank];
    }

    // Load first: admission needs the sharded set's measured size, and a
    // rejection after loading is still fail-closed because the store is
    // cleared before returning.
    auto store = std::make_unique<Dsv4RankLocalWeightStore>();
    const auto checkpoint_before = checkpoint->stats();
    const auto cuda_before = cuda.stats();
    result = store->load(*checkpoint, cuda, rank_devices, kLayers);
    if (!result.ok()) return result;
    if (config.verbose) {
        const auto stats = store->stats();
        const auto checkpoint_after = checkpoint->stats();
        const auto cuda_after = cuda.stats();
        std::uint64_t cuda_copy_nanoseconds = 0U;
        std::uint64_t cuda_wait_nanoseconds = 0U;
        std::uint64_t cuda_allocation_nanoseconds = 0U;
        for (std::size_t slot = 0U; slot < cuda_after.devices.size(); ++slot) {
            const auto& after = cuda_after.devices[slot];
            const auto before = slot < cuda_before.devices.size()
                ? cuda_before.devices[slot] : CudaBackendStats::Device{};
            cuda_copy_nanoseconds += after.weight_copy_nanoseconds -
                                     before.weight_copy_nanoseconds;
            cuda_wait_nanoseconds += after.upload_wait_nanoseconds -
                                     before.upload_wait_nanoseconds;
            cuda_allocation_nanoseconds += after.weight_allocation_nanoseconds -
                                           before.weight_allocation_nanoseconds;
        }
        const auto read_nanoseconds = checkpoint_after.nanoseconds -
                                      checkpoint_before.nanoseconds;
        const auto accounted_seconds = static_cast<double>(
            read_nanoseconds + cuda_copy_nanoseconds + cuda_wait_nanoseconds +
            cuda_allocation_nanoseconds) / 1.0e9;
        const auto cpu_other_seconds = std::max(
            0.0, stats.seconds - accounted_seconds);
        const auto read_gib_s = stats.seconds == 0.0
            ? 0.0
            : static_cast<double>(stats.checkpoint_read_bytes) /
                  stats.seconds / static_cast<double>(1ULL << 30U);
        std::cerr << "[deepseek-load] phase=rank_local_weights elapsed_ms="
                  << stats.seconds * 1000.0
                  << " checkpoint_read_calls=" << stats.checkpoint_read_calls
                  << " checkpoint_read_bytes=" << stats.checkpoint_read_bytes
                  << " checkpoint_read_gib_s=" << read_gib_s
                  << " checkpoint_read_ms="
                  << static_cast<double>(read_nanoseconds) / 1.0e6
                  << " cuda_copy_ms="
                  << static_cast<double>(cuda_copy_nanoseconds) / 1.0e6
                  << " cuda_wait_ms="
                  << static_cast<double>(cuda_wait_nanoseconds) / 1.0e6
                  << " cuda_allocation_ms="
                  << static_cast<double>(cuda_allocation_nanoseconds) / 1.0e6
                  << " cpu_other_ms=" << cpu_other_seconds * 1000.0
                  << " rank0_device_bytes=" << stats.device_weight_bytes[0]
                  << " rank1_device_bytes=" << stats.device_weight_bytes[1]
                  << '\n';
    }
    const auto sharded = store->device_bytes();

    Dsv4RankLocalAdmissionRequest request;
    request.devices.assign(rank_devices.begin(), rank_devices.end());
    request.kv_cache_mode = config.kv_cache_mode;
    request.supported_checkpoint = true;
    request.fp4_routed_experts = config.enable_device_moe;
    request.layer_count = kLayers;
    // Admission is a setup-time check, so it uses the configured maximum for
    // both: the prompt length is not known until generate() runs, and a plan
    // that only fits the current prompt is not a plan.
    request.active_context_tokens = config.maximum_context_tokens;
    request.maximum_context_tokens = config.maximum_context_tokens;
#if defined(STRATA_HAS_NCCL)
    request.nccl_available = true;
#else
    request.nccl_available = false;
#endif
    // Dsv4MemoryPlan's spine and expert-cache figures are aggregates across
    // every device -- admission compares their sum against the aggregate VRAM
    // budget -- while the rank-local ceiling is per device. Charging each rank
    // the aggregate would double-count both on a two-device topology and
    // reject a configuration that fits. The weight cache measures the real
    // per-slot split, so take it from there rather than dividing by the
    // device count: 43 layers over two slots is 22 and 21, not 21.5.
    const auto cache = weights->stats();
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        auto& device = request.device[rank];
        device.initial_device_usage_bytes =
            rank_local_initial_device_vram_bytes[rank];
        device.rank_local_weight_bytes = sharded[rank];
        device.centralized_spine_bytes =
            rank < cache.pinned_bytes.size() ? cache.pinned_bytes[rank] : 0U;
        if (rank == mhc_slot) {
            device.centralized_spine_bytes += memory.mhc_device_bytes;
        }
        device.workspace_bytes = kDeviceWorkspaceReserve;
        device.kv_capacity_bytes =
            rank < memory.per_device_kv_cache_bytes.size()
                ? memory.per_device_kv_cache_bytes[rank] : 0U;
        device.nccl_buffer_bytes = 64ULL << 20U;
        device.head_buffer_bytes = 16ULL << 20U;
        // The rank-local store and centralized cache share one fixed CUDA
        // weight arena. Admission must cap the cache against the suballocation
        // space left after the store, as well as against the overall program
        // ceiling; otherwise the logical cache capacity can promise bytes the
        // arena can never allocate.
        const auto cache_pinned =
            rank < cache.pinned_bytes.size() ? cache.pinned_bytes[rank] : 0U;
        if (rank >= capacities.size() ||
            sharded[rank] > capacities[rank] ||
            cache_pinned > capacities[rank] - sharded[rank]) {
            result.errors.emplace_back(
                "rank-local CUDA device " + std::to_string(devices[rank]) +
                " weight arena cannot retain the rank-local store beside "
                "the centralized prefill spine");
            continue;
        }
        // The routed-expert tier is permanent and suballocates from this same
        // arena, so its bytes are reserved before the prefill cache is sized.
        // Without this the cache is promised space the tier already holds and
        // fails an acquire mid-prefill rather than simply being smaller.
        const auto tier_reserved =
            config.static_expert_plan_path.empty()
                ? 0U : config.static_expert_tier_bytes;
        const auto arena_after_tier =
            capacities[rank] > sharded[rank] + cache_pinned + tier_reserved
                ? capacities[rank] - sharded[rank] - cache_pinned - tier_reserved
                : 0U;
        const auto arena_expert_bytes = arena_after_tier;
        const auto cache_expert_bytes =
            rank < cache.capacity_bytes.size() &&
                    cache.capacity_bytes[rank] > cache_pinned
                ? cache.capacity_bytes[rank] - cache_pinned
                : 0U;
        device.expert_cache_bytes =
            std::min(arena_expert_bytes, cache_expert_bytes);
    }
    if (!result.ok()) {
        store->clear();
        return result;
    }
    request.host.routed_cpu_storage_bytes = memory.routed_expert_host_bytes;
    request.host.host_parameter_bytes = memory.host_parameter_bytes;
    request.host.kv_state_bytes = memory.kv_state_bytes;
    request.host.host_workspace_bytes = memory.host_workspace_bytes;

    // The cards' real capacities, so the ceiling is a fraction of this
    // machine rather than of the one the constant was measured on.
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        const auto memory = CudaBackend::device_memory(devices[rank]);
        if (memory.ok()) {
            request.device[rank].device_total_bytes = memory.value.total_bytes;
        }
    }
    auto admitted = admit_dsv4_rank_local(request, NumaTopology::detect());
    if (!admitted.ok()) {
        store->clear();
        result.errors = std::move(admitted.errors);
        return result;
    }
    result = weights->cap_expert_capacity(
        admitted.expert_cache_capacity_bytes);
    if (!result.ok()) {
        store->clear();
        return result;
    }

#if defined(STRATA_HAS_NCCL)
    auto executor = std::make_unique<Dsv4RankLocalLayerExecutor>(cuda);
    Dsv4RankLocalLayerOptions options;
    options.devices = rank_devices;
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        options.rank_cpus[rank] = admitted.rank_cpus[rank];
    }
    options.resident = &resident;

    // Routed-expert tiers on the rank devices themselves. The layer is already
    // executing there, so the experts cost 0.128 ms each against the host
    // path's 0.282, and nothing crosses a device boundary. Each rank's tier
    // takes a disjoint slice of one ranking, so the two cards split the hottest
    // experts rather than both holding the same ones.
    if (!config.static_expert_plan_path.empty()) {
        auto plan = Dsv4ExpertResidencyPlan::load(
            config.static_expert_plan_path, kLayers, kExperts);
        if (!plan.ok()) {
            store->clear();
            append_errors(result, std::move(plan.errors),
                          "expert residency plan");
            return result;
        }
        static_expert_tiers.clear();
        for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
            auto tier = std::make_unique<Dsv4StaticExpertTier>();
            auto copy = plan.value;
            auto prepared = tier->initialize(
                rank_devices[rank], cuda, *checkpoint, std::move(copy),
                config.static_expert_tier_bytes, rank, kDsv4RankLocalWorld);
            if (!prepared.ok()) {
                store->clear();
                static_expert_tiers.clear();
                append_errors(result, std::move(prepared.errors),
                              "expert tier rank " + std::to_string(rank));
                return result;
            }
            options.static_expert_tiers[rank] = tier.get();
            static_expert_tiers.push_back(std::move(tier));
        }
    }
    result = executor->initialize(options);
    if (!result.ok()) {
        store->clear();
        return result;
    }
    rank_local_actual_device_vram_bytes =
        device_vram_used_bytes(devices);
    if (rank_local_actual_device_vram_bytes.size() !=
        kDsv4RankLocalWorld) {
        store->clear();
        result.errors.emplace_back(
            "rank-local actual VRAM ledger does not match devices");
        return result;
    }
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        const auto ceiling = rank_local_vram_ceiling(devices[rank]);
        if (rank_local_actual_device_vram_bytes[rank] > ceiling) {
            store->clear();
            result.errors.emplace_back(
                "rank-local CUDA device " +
                std::to_string(devices[rank]) + " uses " +
                std::to_string(rank_local_actual_device_vram_bytes[rank]) +
                " B after setup, above the " +
                std::to_string(ceiling) + " B program ceiling");
        }
    }
    if (!result.ok()) return result;
    rank_local_executor = std::move(executor);
#else
    store->clear();
    result.errors.emplace_back(
        "rank-local decode requires an NCCL-enabled build");
    return result;
#endif

    rank_local_weights = std::move(store);
    rank_local_admission = std::move(admitted);
    rank_local_active = true;
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::rank_local_forward_hidden(
    std::uint32_t token, std::uint32_t position, std::span<float> hidden,
    std::vector<float>* fused_logits) {
    ValidationResult result;
    static_cast<void>(token);
    static_cast<void>(position);
    static_cast<void>(hidden);
    static_cast<void>(fused_logits);
#if defined(STRATA_HAS_NCCL)
    const bool session = rank_local_active &&
                         rank_local_executor != nullptr &&
                         rank_local_weights != nullptr;
#else
    const bool session = false;
#endif
    if (!session) {
        result.errors.emplace_back(
            "rank-local decode was entered without an admitted session");
        return result;
    }
    if (hidden.size() != static_cast<std::size_t>(kMhc) * kHidden) {
        result.errors.emplace_back(
            "rank-local decode hidden state has the wrong shape");
        return result;
    }
    if (kv_cache == nullptr || devices.size() < kDsv4RankLocalWorld) {
        result.errors.emplace_back(
            "rank-local decode requires a physical KV cache on two devices");
        return result;
    }
#if defined(STRATA_HAS_NCCL)
    // The token is a transaction over the sequence: every layer's KV rows are
    // reserved and encoded here, and none of them is accounted until the
    // terminal head has produced output on both ranks. The destructor aborts,
    // so an early return truncates the sequence back to `position`.
    Dsv4RankLocalKvTransaction transaction(
        *kv_cache, active_sequence, {0U, 1U}, position);

    // Seed the mHC state on both ranks before any layer runs. The executor
    // takes dsv4_mhc_device_view per rank and refuses to execute against a
    // closed state, so this is a precondition, not an optimization.
    //
    // Three properties, all taken from seed_m3_layer0 at
    // a31ac58:apps/strata_dsv4_rank_local_layer.cu:2875 rather than inferred:
    //   - once per rank, each on its own device
    //   - always layer 0's *attention* mHC; later layers arrive through the
    //     per-layer call's next_attention_mhc, and the terminal layer passes
    //     nullptr for it
    //   - mHC weights are replicated per rank, never sharded, because
    //     dsv4_mhc_begin rejects a weight whose device is not the target
    //
    // It is once per token rather than once per session: the terminal
    // finish_chain closes the state machine, and any layer failure aborts the
    // branch, so each token re-seeds.
    const auto seed = rank_local_weights->layer_view(0U, token);
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        const auto* attention_mhc = seed.rank[rank].attention_mhc;
        if (attention_mhc == nullptr) {
            result.errors.emplace_back(
                "rank-local layer 0 attention mHC weights are unavailable for "
                "rank " + std::to_string(rank));
            return result;
        }
        result = cuda.dsv4_mhc_begin_device(
            devices[rank], *attention_mhc, hidden);
        if (!result.ok()) return result;
    }
    // From here every exit must close the mHC branches it opened and roll the
    // token back. The transaction's destructor truncates the sequence, so an
    // early return can never leave a half-appended position visible.
    const auto close_branches = [&] {
        for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
            static_cast<void>(cuda.dsv4_mhc_abort_branch(devices[rank]));
        }
    };
    rank_local_attention_input.clear();

    // An indexed layer used to force the whole token onto the sequential
    // driver, because selection needed the query rank on the host and the only
    // host node available is inside a CUDA callback, where a CUDA call is not
    // permitted. With projection, scoring, selection and candidate resolution
    // all enqueued in the layer's own command sequence, that is no longer true
    // of an indexed layer that actually selects.
    //
    // One narrow band remains sequential: an admitted indexer whose compressed
    // history has not yet passed its own top-k, where every compressed row is
    // attended and no selection runs at all. It is reachable only between an
    // admitted context above 2,048 tokens and a decode position below 2,052,
    // and it is left on the path that has been measured rather than moved onto
    // one that has not.
    const bool queued = std::none_of(
        attention_state.begin(), attention_state.end(),
        [position](const AttentionState& state) {
            const auto ratio = state.indexer_compressor.ratio;
            return ratio != 0U && (position + 1U) / ratio <= kIndexTopK;
        });
    if (queued) {
        // Finish every host-side reservation and cache lookup before a CUDA host
        // node can start. Besides keeping all borrowed views stable, this avoids
        // concurrent access to the host-tensor map from submission and callback
        // threads.
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            result = rank_local_prepare_layer(
                layer, token, position, transaction,
                rank_local_scratch[layer], rank_local_calls[layer]);
            if (!result.ok()) {
                close_branches();
                return result;
            }
        }
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            const auto enqueue_started = std::chrono::steady_clock::now();
            auto queued = rank_local_executor->enqueue_chain_layer(
                rank_local_calls[layer]);
            graph_stats.rank_local_layer_nanoseconds +=
                elapsed_nanoseconds(enqueue_started);
            if (!queued.ok()) {
                append_errors(result, std::move(queued.errors));
                static_cast<void>(rank_local_executor->abort_chain());
                close_branches();
                return result;
            }
        }
    }
    if (!queued) {
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            auto& scratch = rank_local_scratch[layer];
            auto& call = rank_local_calls[layer];
            result = rank_local_prepare_layer(
                layer, token, position, transaction, scratch, call);
            if (!result.ok()) {
                static_cast<void>(rank_local_executor->abort_chain());
                close_branches();
                return result;
            }
            if (layer + 1U < kLayers) {
                Dsv4RankLocalLayerResult layer_result;
                const auto layer_started = std::chrono::steady_clock::now();
                auto ran = rank_local_executor->run(
                    call, Dsv4RankLocalFailure::None, layer_result);
            graph_stats.rank_local_layer_nanoseconds +=
                elapsed_nanoseconds(layer_started);
            graph_stats.rank_local_device_nanoseconds +=
                static_cast<std::uint64_t>(layer_result.timing.total_ms * 1.0e6);
            // total_ms is wall around the whole layer and already contains the
            // boundary, so this is a component of it, not a separate term.
            graph_stats.rank_local_boundary_nanoseconds +=
                static_cast<std::uint64_t>(
                    layer_result.timing.diagnostic_boundary_ms * 1.0e6);
            graph_stats.rank_local_collective_nanoseconds +=
                static_cast<std::uint64_t>(
                    (layer_result.timing.attention_collective_ms +
                     layer_result.timing.attention_publication_ms +
                     layer_result.timing.moe_publication_ms) * 1.0e6);
            graph_stats.rank_local_transition_nanoseconds +=
                static_cast<std::uint64_t>(
                    (layer_result.timing.transition_router_ms +
                     layer_result.timing.final_transition_ms) * 1.0e6);
            graph_stats.rank_local_shared_nanoseconds +=
                static_cast<std::uint64_t>(
                    std::max(layer_result.timing.shared_gpu_rank0_ms,
                             layer_result.timing.shared_gpu_rank1_ms) * 1.0e6);
            if (!ran.ok() || !layer_result.success ||
                layer_result.global_attention_status != 0U ||
                layer_result.global_moe_status != 0U) {
                append_errors(result, std::move(
                    ran.errors.empty() ? layer_result.errors : ran.errors));
                result.errors.emplace_back(
                    "rank-local decode failed at layer " +
                    std::to_string(layer));
                close_branches();
                return result;
            }
            // Both ranks publish the same reduction. A divergence here is a
            // replica fault, not a rounding difference, and the next layer's
            // selection would silently use one rank's state for both.
            if (!std::equal(layer_result.next_attention_input[0].begin(),
                            layer_result.next_attention_input[0].end(),
                            layer_result.next_attention_input[1].begin())) {
                result.errors.emplace_back(
                    "rank-local attention input diverged between ranks at "
                    "layer " + std::to_string(layer));
                close_branches();
                return result;
            }
            rank_local_attention_input.assign(
                layer_result.next_attention_input[0].begin(),
                layer_result.next_attention_input[0].end());
            graph_stats.attention_nanoseconds += static_cast<std::uint64_t>(
                layer_result.timing.attention_ms * 1.0e6);
            graph_stats.moe_nanoseconds += static_cast<std::uint64_t>(
                (std::max(layer_result.timing.cpu_routed_rank0_ms,
                          layer_result.timing.cpu_routed_rank1_ms) +
                 layer_result.timing.moe_collective_ms) * 1.0e6);
                continue;
            }
            // Terminal layer. The executor refuses to run() it: the output
            // head consumes the mHC state, so the last layer is queued and
            // drained by finish_chain, exactly as run_m3_sequential does at
            // a31ac58:apps/strata_dsv4_rank_local_layer.cu:3038.
            const auto terminal_started = std::chrono::steady_clock::now();
            auto queued = rank_local_executor->enqueue_chain_layer(call);
            graph_stats.rank_local_layer_nanoseconds +=
                elapsed_nanoseconds(terminal_started);
            if (!queued.ok()) {
                append_errors(result, std::move(queued.errors));
                static_cast<void>(rank_local_executor->abort_chain());
                close_branches();
                return result;
            }
        }
    }

    Dsv4RankLocalHeadRequest head_request;
    const bool fuse_head = fused_logits != nullptr;
    if (fuse_head) {
        static_assert(kDsv4RankLocalVocabulary == kVocabulary,
                      "rank-local head vocabulary must match the contract");
        for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
            rank_local_head[rank] = {};
            rank_local_head[rank].owner = this;
            rank_local_local_logits[rank].assign(
                kDsv4RankLocalVocabularyShard, 0.0F);
            rank_local_published_logits[rank].assign(
                kDsv4RankLocalVocabulary, 0U);
            head_request.heads[rank] =
                &rank_local_weights->head().weights[rank];
            head_request.callbacks[rank] = device_head_callback;
            head_request.callback_contexts[rank] = &rank_local_head[rank];
            head_request.local_logits[rank] = rank_local_local_logits[rank];
            head_request.published_logits[rank] =
                rank_local_published_logits[rank];
        }
    }
    Dsv4RankLocalLayerChainResult chain;
    const auto finish_started = std::chrono::steady_clock::now();
    auto finished = rank_local_executor->finish_chain(
        &chain, fuse_head ? &head_request : nullptr);
    graph_stats.rank_local_layer_nanoseconds +=
        elapsed_nanoseconds(finish_started);
    const auto expected_chain_count = queued
        ? static_cast<std::size_t>(kLayers) : 1U;
    if (!finished.ok() || chain.chain_count != expected_chain_count ||
        !chain.terminal) {
        append_errors(result, std::move(finished.errors));
        if (queued) {
            for (const auto& contexts : rank_local_page_contexts) {
                for (const auto& context : contexts) {
                    append_errors(result, context.result.errors,
                                  "rank-local page callback layer " +
                                      std::to_string(context.layer) +
                                      " rank " +
                                      std::to_string(context.rank));
                }
            }
        }
        if (result.ok()) {
            result.errors.emplace_back(
                "rank-local terminal layer did not complete");
        }
        close_branches();
        return result;
    }
    if (queued) {
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            for (std::size_t rank = 0U; rank < 1U; ++rank) {
                const auto& context = rank_local_page_contexts[layer][rank];
                if (!context.invoked || !context.result.ok()) {
                    append_errors(result, context.result.errors,
                                  "rank-local page callback layer " +
                                      std::to_string(context.layer) +
                                      " rank " +
                                      std::to_string(context.rank));
                    if (context.result.ok()) {
                        result.errors.emplace_back(
                            "rank-local page callback was not invoked at layer " +
                            std::to_string(layer) + " rank " +
                            std::to_string(rank));
                    }
                }
            }
        }
        if (!result.ok()) {
            close_branches();
            return result;
        }
        std::uint64_t page_callback_nanoseconds = 0U;
        for (const auto& contexts : rank_local_page_contexts) {
            page_callback_nanoseconds += contexts[0].elapsed_nanoseconds;
        }
        graph_stats.rank_local_kv_nanoseconds += page_callback_nanoseconds;
    }
    {
        // Attribute the phases on the same rank the MoE term is taken from,
        // so the phase sum and the total describe one rank's critical path
        // rather than a mixture of both.
        const auto slower =
            chain.cpu_moe_phases[0].total_nanoseconds >=
                    chain.cpu_moe_phases[1].total_nanoseconds
                ? 0U : 1U;
        const auto& phases = chain.cpu_moe_phases[slower];
        graph_stats.moe_nanoseconds += phases.total_nanoseconds;
        graph_stats.rank_local_moe_gate_up_nanoseconds +=
            phases.gate_up_nanoseconds;
        graph_stats.rank_local_moe_down_nanoseconds += phases.down_nanoseconds;
        graph_stats.rank_local_moe_reduce_nanoseconds +=
            phases.reduce_nanoseconds;
    }
    if (fuse_head && (!rank_local_head[0].invoked ||
                      !rank_local_head[1].invoked)) {
        result.errors.emplace_back(
            "rank-local output-head host callback was not invoked on both "
            "ranks");
        return result;
    }
    if (!std::equal(chain.final_hidden[0].begin(), chain.final_hidden[0].end(),
                    chain.final_hidden[1].begin())) {
        result.errors.emplace_back(
            "rank-local terminal hidden state diverged between ranks");
        return result;
    }
    std::copy(chain.final_hidden[0].begin(), chain.final_hidden[0].end(),
              hidden.begin());
    if (fuse_head) {
        fused_logits->assign(kVocabulary, 0.0F);
        for (std::size_t index = 0U; index < kVocabulary; ++index) {
            (*fused_logits)[index] = std::bit_cast<float>(
                static_cast<std::uint32_t>(
                    rank_local_published_logits[0][index]) << 16U);
        }
        graph_stats.output_head_nanoseconds += static_cast<std::uint64_t>(
            chain.terminal_head_ms * 1.0e6);
    }
    // The queued chain has drained, so nothing still reads the index
    // projections. They are re-acquired next token as cache hits; holding them
    // across the whole generation would leave the lease account open at its
    // end, which the caller checks.
    for (auto& scratch : rank_local_scratch) {
        for (auto& lease : scratch.index_query_projection) lease = {};
        for (auto& lease : scratch.index_weight_projection) lease = {};
    }
    // Every layer ran, both ranks agreed, and the head produced output. Only
    // now is the token's KV allowed to become visible.
    result = transaction.commit();
    if (config.enable_layer_hash_trace) {
        record_layer_hash(position, token, kLayers - 1U, hidden);
    }
    return result;
#else
    result.errors.emplace_back(
        "rank-local decode requires an NCCL-enabled build");
    return result;
#endif
}

ValidationResult DeepSeekV4Runtime::Impl::rank_local_patch_pages(
    const RankLocalLayerScratch& scratch) {
    ValidationResult result;
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        const auto bytes = std::span<const std::byte>(scratch.patches[rank]);
        std::size_t cursor = 0U;
        for (const auto& write : scratch.page_writes[rank]) {
            const auto extent = static_cast<std::size_t>(write.data_bytes) +
                                write.scale_bytes;
            if (write.buffer == nullptr || cursor + extent > bytes.size()) {
                result.errors.emplace_back(
                    "rank-local page patch is truncated for rank " +
                    std::to_string(rank));
                return result;
            }
            const std::array<CudaBufferPatch, 2U> patches{{
                {write.data_offset, bytes.subspan(cursor, write.data_bytes)},
                {write.scale_offset,
                 bytes.subspan(cursor + write.data_bytes, write.scale_bytes)},
            }};
            result = cuda.update_buffer(*write.buffer, patches);
            if (!result.ok()) return result;
            cursor += extent;
        }
        if (cursor != bytes.size()) {
            result.errors.emplace_back(
                "rank-local page patch has unused bytes for rank " +
                std::to_string(rank));
            return result;
        }
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::rank_local_candidates(
    std::uint32_t layer, std::uint32_t position,
    std::span<const std::uint32_t> indexed_positions,
    RankLocalLayerScratch& scratch, bool in_chain) {
    ValidationResult result;
    auto& sliding = physical_sliding_blocks;
    auto& compressed = physical_compressed_blocks;
    result = kv_cache->block_table_into(
        active_sequence, Dsv4KvBlockKind::Sliding, layer, sliding);
    if (!result.ok()) return result;
    const auto ratio = attention_state[layer].compressor.ratio;
    compressed.clear();
    if (ratio != 0U) {
        result = kv_cache->block_table_into(
            active_sequence, attention_state[layer].compressor.kind, layer,
            compressed);
        if (!result.ok()) return result;
    }

    // Leases are released only after the executor has consumed the pages, so
    // they are cleared here rather than at the end of the previous layer.
    scratch.leases.clear();
    for (auto& pages : scratch.pages) pages.clear();
    scratch.compressed_block_leased.clear();
    std::unordered_map<std::uint64_t, std::uint32_t> page_indices;
    // One logical row order, two device page lists. Both ranks index the same
    // candidate array, so a page must occupy the same index in both lists.
    const auto locate = [&](Dsv4KvBlockKind kind,
                            const std::vector<Dsv4KvBlockInfo>& table,
                            std::uint32_t logical_row,
                            CudaDsv4AttentionCandidate& candidate) {
        const auto located = locate_physical_kv_block(table, logical_row);
        if (located == table.size()) {
            result.errors.emplace_back(
                "rank-local attention candidate page is unavailable");
            return;
        }
        const auto found = table.begin() + static_cast<std::ptrdiff_t>(located);
        const auto begin = found->logical_begin / found->compression_ratio;
        // A compressed block owns the page slot matching its block-table index;
        // a sliding block is appended after that reserved region, keeping the
        // lazy numbering it has always had.
        const bool positional =
            located < scratch.compressed_block_leased.size() &&
            kind != Dsv4KvBlockKind::Sliding;
        std::uint32_t page_index = 0U;
        if (positional) {
            page_index = static_cast<std::uint32_t>(located);
            if (scratch.compressed_block_leased[located] == 0U) {
                for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld;
                     ++rank) {
                    auto lease = kv_cache->acquire_device(
                        active_sequence, kind, layer, logical_row, rank);
                    if (!lease.ok()) {
                        append_errors(result, std::move(lease.errors));
                        return;
                    }
                    scratch.leases.push_back(std::move(lease.value));
                    scratch.pages[rank][page_index] = {
                        scratch.leases.back().buffer(), found->capacity_rows};
                }
                scratch.compressed_block_leased[located] = 1U;
            }
        } else {
            auto page = page_indices.find(found->id);
            if (page == page_indices.end()) {
                const auto index =
                    static_cast<std::uint32_t>(scratch.pages[0].size());
                for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld;
                     ++rank) {
                    auto lease = kv_cache->acquire_device(
                        active_sequence, kind, layer, logical_row, rank);
                    if (!lease.ok()) {
                        append_errors(result, std::move(lease.errors));
                        return;
                    }
                    scratch.leases.push_back(std::move(lease.value));
                    scratch.pages[rank].push_back(
                        {scratch.leases.back().buffer(),
                         found->capacity_rows});
                }
                page = page_indices.emplace(found->id, index).first;
            }
            page_index = page->second;
        }
        candidate.page = page_index;
        candidate.row = static_cast<std::uint32_t>(logical_row - begin);
        candidate.valid = true;
    };

    const auto compressed_count = ratio == 0U ? 0U : (position + 1U) / ratio;
    const bool sparse = ratio == 4U &&
        attention_state[layer].indexer_compressor.ratio == 4U;
    if (sparse) {
        // Positional page indexing for the compressed stream: a block's page
        // index is its block-table index, fixed before any candidate is
        // examined.
        //
        // The lazy first-touch numbering below cannot survive device-side
        // selection, which does not know which blocks a candidate set will
        // touch, let alone in what order. Making the mapping positional is what
        // lets a device kernel resolve a selected row to a page without the
        // host having seen the selection.
        //
        // Only the *numbering* is positional. The lease is still taken on first
        // touch, so an unselected block is never leased -- 512 of 4,096 blocks
        // at the declared context rather than all of them -- and the page slot
        // of an untouched block stays empty because no candidate can reference
        // it.
        std::uint32_t attendable_blocks = 0U;
        for (const auto& block : compressed) {
            const auto first_row = block.compression_ratio == 0U
                ? 0U : block.logical_begin / block.compression_ratio;
            if (first_row >= compressed_count) break;
            ++attendable_blocks;
        }
        for (auto& pages : scratch.pages) {
            pages.assign(attendable_blocks, CudaDsv4PhysicalPage{});
        }
        scratch.compressed_block_leased.assign(attendable_blocks, 0U);
        // In-chain selection has no first touch to lease on, so every
        // attendable block must carry a live page before the layer is queued.
        // Its block descriptor travels with it: the device resolves a selected
        // row against this table, and the page it names is this table's index.
        if (in_chain) {
            scratch.blocks.clear();
            for (std::uint32_t index = 0U; index < attendable_blocks; ++index) {
                const auto& block = compressed[index];
                const auto logical_row =
                    block.logical_begin / block.compression_ratio;
                for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld;
                     ++rank) {
                    auto lease = kv_cache->acquire_device(
                        active_sequence, attention_state[layer].compressor.kind,
                        layer, static_cast<std::uint32_t>(logical_row), rank);
                    if (!lease.ok()) {
                        append_errors(result, std::move(lease.errors));
                        return result;
                    }
                    scratch.leases.push_back(std::move(lease.value));
                    scratch.pages[rank][index] = {
                        scratch.leases.back().buffer(), block.capacity_rows};
                }
                scratch.compressed_block_leased[index] = 1U;
                scratch.blocks.push_back(CudaDsv4KvBlockDescriptor{
                    block.logical_begin, block.used_rows,
                    block.compression_ratio});
            }
        }
    }
    const auto compressed_width = ratio == 0U ? 0U : ratio == 4U
        ? kIndexTopK
        : ((std::max(1U, compressed_count) + 127U) / 128U) * 128U;
    constexpr std::uint32_t sliding_width = kWindow;
    scratch.candidates.assign(
        static_cast<std::size_t>(compressed_width) + sliding_width, {});
    const auto attended_compressed = sparse
        ? static_cast<std::uint32_t>(indexed_positions.size())
        : compressed_count;
    if (attended_compressed > compressed_width) {
        result.errors.emplace_back(
            "rank-local attention compressed candidates exceed their fixed "
            "region");
        return result;
    }
    // The in-chain compressed region is filled by the resolution kernel from a
    // selection this function never sees.
    if (!in_chain) {
        for (std::uint32_t item = 0U; item < attended_compressed; ++item) {
            const auto logical_row = sparse ? indexed_positions[item] : item;
            locate(attention_state[layer].compressor.kind, compressed,
                   logical_row, scratch.candidates[item]);
            if (!result.ok()) return result;
        }
    }
    const auto window_count = std::min(position + 1U, kWindow);
    for (std::uint32_t item = 0U; item < window_count; ++item) {
        const auto logical_row = position + 1U - window_count + item;
        locate(Dsv4KvBlockKind::Sliding, sliding, logical_row,
               scratch.candidates[
                   static_cast<std::size_t>(compressed_width) + item]);
        if (!result.ok()) return result;
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::rank_local_warm_index_projections() {
    ValidationResult result;
    if (!rank_local_active || weights == nullptr) return result;
    auto cuda_demand = weights->demand();
    for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
        if (attention_state[layer].indexer_compressor.ratio == 0U) continue;
        const auto prefix = layer_prefix(layer) + "attn.indexer.";
        for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
            // Acquired and released: this only has to put the pair in the
            // cache on this rank's device. The per-token acquire that follows
            // is then a hit, and the lease account stays balanced.
            Dsv4WeightCache::Lease query_projection;
            Dsv4WeightCache::Lease weight_projection;
            result = weights->acquire(rank, prefix + "wq_b",
                                      kIndexHeads * kIndexHeadDim, kQueryRank,
                                      query_projection);
            if (!result.ok()) return result;
            result = weights->acquire(rank, prefix + "weights_proj",
                                      kIndexHeads, kHidden, weight_projection);
            if (!result.ok()) return result;
        }
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::rank_local_index_selection(
    std::uint32_t layer, std::uint32_t position,
    RankLocalLayerScratch& scratch) {
    ValidationResult result;
    const auto& state = attention_state[layer].indexer_compressor;
    const auto prefix = layer_prefix(layer) + "attn.indexer.";
    const auto compressed_count = (position + 1U) / state.ratio;
    auto& blocks = physical_index_blocks;
    result = kv_cache->block_table_into(
        active_sequence, Dsv4KvBlockKind::LearnedIndex, layer, blocks);
    if (!result.ok()) return result;

    // Both ranks score the same replicated history on their own device, so
    // each needs its own leases and its own page list over the same rows.
    auto cuda_demand = weights->demand();
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        result = weights->acquire(rank, prefix + "wq_b",
                                  kIndexHeads * kIndexHeadDim, kQueryRank,
                                  scratch.index_query_projection[rank]);
        if (!result.ok()) return result;
        result = weights->acquire(rank, prefix + "weights_proj", kIndexHeads,
                                  kHidden,
                                  scratch.index_weight_projection[rank]);
        if (!result.ok()) return result;
        std::uint32_t remaining = compressed_count;
        for (const auto& block : blocks) {
            if (remaining == 0U) break;
            const auto logical_row = block.logical_begin /
                                     block.compression_ratio;
            auto lease = kv_cache->acquire_device(
                active_sequence, Dsv4KvBlockKind::LearnedIndex, layer,
                static_cast<std::uint32_t>(logical_row), rank);
            if (!lease.ok()) {
                append_errors(result, std::move(lease.errors));
                return result;
            }
            const auto rows = std::min(remaining, block.used_rows);
            scratch.index_leases.push_back(std::move(lease.value));
            // A physical device lease holds the block-major payload alone;
            // acquire_device strips the header on upload.
            scratch.index_pages[rank].push_back(CudaDsv4PhysicalIndexPage{
                scratch.index_leases.back().buffer(), 0U, block.capacity_rows,
                rows});
            remaining -= rows;
        }
        if (remaining != 0U) {
            result.errors.emplace_back(
                "rank-local in-chain index history is incomplete at layer " +
                std::to_string(layer));
            return result;
        }
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::rank_local_prepare_layer(
    std::uint32_t layer, std::uint32_t token, std::uint32_t position,
    Dsv4RankLocalKvTransaction& transaction,
    RankLocalLayerScratch& scratch, Dsv4RankLocalLayerCall& call) {
    ValidationResult result;
    auto& state = attention_state[layer];
    const auto ratio = state.compressor.ratio;
    const auto index_ratio = state.indexer_compressor.ratio;
    const auto slot = layer_device(layer);
    const auto prefix = layer_prefix(layer) + "attn.";

    // Every queued call has completed before the next token enters this
    // function. Release that layer's previous-token candidate leases before
    // reserving its next row; otherwise the cache correctly refuses to mutate
    // a block that still appears in flight.
    scratch.leases.clear();
    scratch.index_leases.clear();
    for (auto& pages : scratch.pages) pages.clear();
    for (auto& pages : scratch.index_pages) pages.clear();
    result = transaction.reserve_layer(
        layer, position, ratio, state.compressor.kind, index_ratio);
    if (!result.ok()) return result;

    // An indexed layer only stays in the chain once its selection runs on the
    // device. Below the threshold there is nothing to select: every compressed
    // row is attended, so the candidate list is determined by position alone.
    const auto index_compressed_count = index_ratio == 0U
        ? 0U : (position + 1U) / index_ratio;
    const bool in_chain_selection =
        index_ratio != 0U && index_compressed_count > kIndexTopK;

    // Every candidate is determined entirely by position and the reserved
    // block table, or resolved on the device from a selection queued in the
    // same command sequence. Build the complete borrowed call now; the live
    // Q/KV and compressor rows are produced later by the executor's
    // stream-ordered callback, so all 43 layers can be submitted before the one
    // completion boundary.
    if (index_ratio == 0U || in_chain_selection) {
        auto sink = host_tensor(prefix + "attn_sink", kHeads);
        if (!sink.ok()) {
            append_errors(result, std::move(sink.errors));
            return result;
        }
        // The compressor state advance runs inside the page callback, on a
        // CUDA host node where a missing tensor could not be reported cleanly.
        // Fault it into the host-tensor map here instead.
        const auto residency = [&](const CompressorState& compressor,
                                   const std::string& compressor_prefix) {
            if (compressor.ratio == 0U) return true;
            const auto dimensions = static_cast<std::size_t>(
                compressor.coefficient) * compressor.head_dim;
            auto ape = host_tensor(
                compressor_prefix + "ape",
                static_cast<std::uint64_t>(compressor.ratio) * dimensions);
            auto norm_weight = host_tensor(compressor_prefix + "norm.weight",
                                           compressor.head_dim);
            if (!ape.ok() || !norm_weight.ok()) {
                append_errors(result, ape.ok() ? std::move(norm_weight.errors)
                                               : std::move(ape.errors));
                return false;
            }
            return true;
        };
        if (!residency(state.compressor, prefix + "compressor.") ||
            !residency(state.indexer_compressor,
                       prefix + "indexer.compressor.")) {
            return result;
        }
        for (std::size_t index = 0U; index < scratch.cosines.size(); ++index) {
            const float angle = static_cast<float>(position) *
                                state.frequencies[index];
            scratch.cosines[index] = std::cos(angle);
            scratch.inverse_sines[index] = -std::sin(angle);
            // The index query rotates with the same angles but keeps the
            // forward sine, which is the sign convention index_select() uses.
            scratch.index_cosines[index] = scratch.cosines[index];
            scratch.index_sines[index] = -scratch.inverse_sines[index];
        }

        scratch.patches[0].clear();
        scratch.patches[1].clear();
        // The executor only exists in an NCCL build. Rank-local decode already
        // fails closed at initialization without it, so this branch is
        // unreachable there; it still reports rather than falling through, and
        // the guard is what keeps a default STRATA_ENABLE_NCCL=OFF build
        // compiling.
#if defined(STRATA_HAS_NCCL)
        result = rank_local_executor->replica_page_patch_staging(
            layer, static_cast<std::size_t>(transaction.patch_bytes(layer)),
            scratch.replica_patch);
#else
        result.errors.emplace_back(
            "rank-local replica page staging requires an NCCL build");
#endif
        if (!result.ok()) return result;
        std::fill(scratch.replica_patch.begin(), scratch.replica_patch.end(),
                  std::byte{});
        scratch.compressed_row.clear();
        scratch.index_row.clear();
        scratch.indexed_positions.clear();
        for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
            result = transaction.page_writes(
                layer, rank, scratch.page_writes[rank]);
            if (!result.ok()) return result;
        }
        const auto candidate_started = std::chrono::steady_clock::now();
        result = rank_local_candidates(layer, position, {}, scratch,
                                       in_chain_selection);
        graph_stats.rank_local_candidate_nanoseconds +=
            elapsed_nanoseconds(candidate_started);
        if (!result.ok()) return result;
        if (in_chain_selection) {
            result = rank_local_index_selection(layer, position, scratch);
            if (!result.ok()) return result;
        }

        call = {};
        call.layer = layer;
        call.position = position;
        call.weights = rank_local_weights->layer_view(layer, token);
        if (!in_chain_selection) {
            // The sparse-index compressor exists in the resident store for the
            // 1M operating point but is not part of this request's active
            // state.
            for (auto& rank_weights : call.weights.rank) {
                rank_weights.index_compressor_value = nullptr;
                rank_weights.index_compressor_gate = nullptr;
                rank_weights.index_compressor_elements = 0U;
            }
        }
        call.head_sinks = *sink.value;
        for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
            call.pages[rank] = scratch.pages[rank];
            auto& context = rank_local_page_contexts[layer][rank];
            context.owner = this;
            context.transaction = &transaction;
            context.scratch = &scratch;
            context.layer = layer;
            context.position = position;
            context.rank = rank;
            context.result = {};
            context.invoked = false;
            context.elapsed_nanoseconds = 0U;
            call.page_patches[rank].callback = rank == 0U
                ? rank_local_page_patch_callback : nullptr;
            call.page_patches[rank].context = rank == 0U ? &context : nullptr;
            call.page_patches[rank].ready_patch = rank == 1U
                ? std::span<const std::byte>(scratch.replica_patch)
                : std::span<const std::byte>{};
            call.page_patches[rank].writes = scratch.page_writes[rank];
        }
        call.candidates = scratch.candidates;
        call.inverse_rope_cosines = scratch.cosines;
        call.inverse_rope_sines = scratch.inverse_sines;
        if (in_chain_selection) {
            call.selection.active = true;
            for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
                call.selection.query_projection[rank] =
                    &scratch.index_query_projection[rank].weight();
                call.selection.weight_projection[rank] =
                    &scratch.index_weight_projection[rank].weight();
                call.selection.index_pages[rank] = scratch.index_pages[rank];
            }
            call.selection.blocks = scratch.blocks;
            call.selection.rope_cosines = scratch.index_cosines;
            call.selection.rope_sines = scratch.index_sines;
            call.selection.heads = kIndexHeads;
            call.selection.head_dim = kIndexHeadDim;
            call.selection.rope_dim = kRopeDim;
            call.selection.top_k = kIndexTopK;
            call.selection.compressed_width = kIndexTopK;
            call.selection.weight_scale = kIndexQueryScale;
            ++graph_stats.attention_index_queries;
            graph_stats.attention_index_candidates += index_compressed_count;
            graph_stats.attention_index_selected += kIndexTopK;
            graph_stats.attention_index_cuda_dispatches +=
                kDsv4RankLocalWorld;
        }
        call.ordered_page_patches = true;
        call.terminal = layer + 1U == kLayers;
        return result;
    }

    // One host-visible preparation per layer, on the slot that owns this
    // layer's centralized compressor weights. The executor's own preparation
    // is device-only and computes no compressor projection, so the pooled
    // rows and the index query have to come from here. It is one extra
    // projection pass per layer, not per rank: the result is replicated.
    //
    // The leases are scope-local because this preparation is synchronous:
    // with a host-visible query and no page-patch callback the backend copies
    // its diagnostics and synchronizes before returning, so nothing queued
    // still reads the weights after this function exits.
    Dsv4WeightCache::Lease query_a;
    Dsv4WeightCache::Lease query_b;
    Dsv4WeightCache::Lease key_value_weight;
    Dsv4WeightCache::Lease compressor_value;
    Dsv4WeightCache::Lease compressor_gate;
    Dsv4WeightCache::Lease index_value;
    Dsv4WeightCache::Lease index_gate;
    result = weights->acquire(
        slot, prefix + "wq_a", kQueryRank, kHidden, query_a);
    if (!result.ok()) return result;
    result = weights->acquire(
        slot, prefix + "wq_b", kHeads * kHeadDim, kQueryRank, query_b);
    if (!result.ok()) return result;
    result = weights->acquire(
        slot, prefix + "wkv", kHeadDim, kHidden, key_value_weight);
    if (!result.ok()) return result;
    scratch.compressor_values.clear();
    scratch.compressor_scores.clear();
    if (ratio != 0U) {
        const auto dimensions =
            static_cast<std::size_t>(state.compressor.coefficient) *
            state.compressor.head_dim;
        result = weights->acquire(slot, prefix + "compressor.wkv", dimensions,
                                  kHidden, compressor_value);
        if (!result.ok()) return result;
        result = weights->acquire(slot, prefix + "compressor.wgate", dimensions,
                                  kHidden, compressor_gate);
        if (!result.ok()) return result;
        scratch.compressor_values.assign(dimensions, 0.0F);
        scratch.compressor_scores.assign(dimensions, 0.0F);
    }
    scratch.index_compressor_values.clear();
    scratch.index_compressor_scores.clear();
    if (index_ratio != 0U) {
        const auto dimensions =
            static_cast<std::size_t>(state.indexer_compressor.coefficient) *
            state.indexer_compressor.head_dim;
        result = weights->acquire(slot, prefix + "indexer.compressor.wkv",
                                  dimensions, kHidden, index_value);
        if (!result.ok()) return result;
        result = weights->acquire(slot, prefix + "indexer.compressor.wgate",
                                  dimensions, kHidden, index_gate);
        if (!result.ok()) return result;
        scratch.index_compressor_values.assign(dimensions, 0.0F);
        scratch.index_compressor_scores.assign(dimensions, 0.0F);
    }
    auto query_norm = host_tensor(prefix + "q_norm.weight", kQueryRank);
    if (!query_norm.ok()) {
        append_errors(result, std::move(query_norm.errors));
        return result;
    }
    auto key_value_norm = host_tensor(prefix + "kv_norm.weight", kHeadDim);
    if (!key_value_norm.ok()) {
        append_errors(result, std::move(key_value_norm.errors));
        return result;
    }
    auto sink = host_tensor(prefix + "attn_sink", kHeads);
    if (!sink.ok()) {
        append_errors(result, std::move(sink.errors));
        return result;
    }

    for (std::size_t index = 0U; index < scratch.cosines.size(); ++index) {
        const float angle = static_cast<float>(position) *
                            state.frequencies[index];
        scratch.cosines[index] = std::cos(angle);
        // The executor recovers the forward sine as its negation, so only the
        // inverse pair travels in the call.
        scratch.inverse_sines[index] = -std::sin(angle);
    }

    CudaDsv4AttentionPrepareRequest request;
    request.query_a = &query_a.weight();
    request.query_b = &query_b.weight();
    request.key_value = &key_value_weight.weight();
    if (!scratch.compressor_values.empty()) {
        request.compressor_value = &compressor_value.weight();
        request.compressor_gate = &compressor_gate.weight();
    }
    if (!scratch.index_compressor_values.empty()) {
        request.index_compressor_value = &index_value.weight();
        request.index_compressor_gate = &index_gate.weight();
    }
    request.query_norm = *query_norm.value;
    request.key_value_norm = *key_value_norm.value;
    // Forward RoPE for the key/value row; the query carries the same angles.
    std::array<float, kRopeDim / 2U> sines{};
    for (std::size_t index = 0U; index < sines.size(); ++index) {
        sines[index] = -scratch.inverse_sines[index];
    }
    request.rope_cosines = scratch.cosines;
    request.rope_sines = sines;
    request.mhc_device = devices[slot];
    request.maximum_workspace_bytes = 1ULL << 20U;
    // This preparation exists to produce host-visible projections, not to
    // stage a command. The executor prepares again per rank to stage the one
    // its attention consumes, and a published query left here would make that
    // second preparation out of order.
    request.host_only = true;
    scratch.query_rank.assign(kQueryRank, 0.0F);
    scratch.key_value.assign(kHeadDim, 0.0F);
    auto prepare_started = std::chrono::steady_clock::now();
    result = cuda.dsv4_prepare_attention(
        devices[slot], request, scratch.query_rank, scratch.key_value,
        scratch.compressor_values, scratch.compressor_scores,
        scratch.index_compressor_values, scratch.index_compressor_scores);
    graph_stats.attention_query_nanoseconds +=
        elapsed_nanoseconds(prepare_started);
    if (!result.ok()) return result;

    // Pool the compressor rows without publishing them: one logical row has to
    // reach two devices' pages, so the transaction owns the encode.
    result = compress_state(layer, state.compressor, prefix + "compressor.",
                            {}, position, state.frequencies,
                            scratch.compressor_values,
                            scratch.compressor_scores, nullptr, {},
                            &scratch.compressed_row);
    if (!result.ok()) return result;
    result = compress_state(layer, state.indexer_compressor,
                            prefix + "indexer.compressor.", {}, position,
                            state.frequencies,
                            scratch.index_compressor_values,
                            scratch.index_compressor_scores, nullptr, {},
                            &scratch.index_row);
    if (!result.ok()) return result;

    const auto kv_started = std::chrono::steady_clock::now();
    const auto patch_bytes =
        static_cast<std::size_t>(transaction.patch_bytes(layer));
    std::array<std::span<std::byte>, kDsv4RankLocalWorld> patches{};
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        scratch.patches[rank].assign(patch_bytes, std::byte{});
        patches[rank] = scratch.patches[rank];
        result = transaction.page_writes(layer, rank,
                                         scratch.page_writes[rank]);
        if (!result.ok()) return result;
    }
    result = transaction.commit_layer(layer, scratch.key_value,
                                      scratch.compressed_row, patches,
                                      scratch.index_row);
    if (!result.ok()) return result;
    result = rank_local_patch_pages(scratch);
    graph_stats.rank_local_kv_nanoseconds += elapsed_nanoseconds(kv_started);
    if (!result.ok()) return result;

    scratch.indexed_positions.clear();
    if (index_ratio != 0U) {
        const auto compressed_count = (position + 1U) / index_ratio;
        if (compressed_count > kIndexTopK &&
            rank_local_attention_input.size() != kHidden) {
            result.errors.emplace_back(
                "rank-local sparse selection has no attention input at layer " +
                std::to_string(layer));
            return result;
        }
        auto select_started = std::chrono::steady_clock::now();
        // The host-visible preparation above is this device's most recent, so
        // the index projections read its activations rather than sending the
        // query rank and layer input back across the bus.
        result = index_select(layer, rank_local_attention_input,
                              scratch.query_rank, position,
                              scratch.indexed_positions, true);
        graph_stats.attention_index_nanoseconds +=
            elapsed_nanoseconds(select_started);
        if (!result.ok()) return result;
    }
    const auto candidate_started = std::chrono::steady_clock::now();
    result = rank_local_candidates(layer, position, scratch.indexed_positions,
                                   scratch);
    graph_stats.rank_local_candidate_nanoseconds +=
        elapsed_nanoseconds(candidate_started);
    if (!result.ok()) return result;

    call = {};
    call.layer = layer;
    call.position = position;
    call.weights = rank_local_weights->layer_view(layer, token);
    // Indexed-context preparation remains the explicit sequential arm until
    // Step 4 moves selection inside the device command. Its separate
    // host-visible preparation already computed these projections.
    for (auto& rank_weights : call.weights.rank) {
        rank_weights.compressor_value = nullptr;
        rank_weights.compressor_gate = nullptr;
        rank_weights.index_compressor_value = nullptr;
        rank_weights.index_compressor_gate = nullptr;
        rank_weights.compressor_elements = 0U;
        rank_weights.index_compressor_elements = 0U;
    }
    call.head_sinks = *sink.value;
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        call.pages[rank] = scratch.pages[rank];
    }
    call.candidates = scratch.candidates;
    call.inverse_rope_cosines = scratch.cosines;
    call.inverse_rope_sines = scratch.inverse_sines;
    // No page patch: both ranks' pages already hold this position's rows, so
    // the executor's preparation stays device-only and costs no host sync.
    call.terminal = layer + 1U == kLayers;
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::device_mhc_forward_hidden(
    std::uint32_t token, std::uint32_t position,
    std::span<float> hidden, std::vector<float>* fused_logits) {
    ValidationResult result;
    if (hidden.size() != static_cast<std::size_t>(kMhc) * kHidden) {
        result.errors.emplace_back(
            "DeepSeek device mHC hidden state has the wrong shape");
        return result;
    }
    const auto device = devices[mhc_slot];
    std::vector<float> weighted(
        config.enable_layer_hash_trace ? kHidden : 0U);
    std::vector<float> layer_input(kHidden);
    Dsv4WeightCache::Lease head_lease;
    const bool fuse_head = fused_logits != nullptr &&
        !config.enable_layer_hash_trace &&
        layer_device(kLayers - 1U) == mhc_slot;
    if (fused_logits != nullptr) fused_logits->clear();
    if (fuse_head) {
        result = weights->acquire(
            layer_device(kLayers - 1U), "head", kVocabulary, kHidden,
            head_lease);
        if (!result.ok()) return result;
        fused_logits->assign(kVocabulary, 0.0F);
    }
    auto phase_started = std::chrono::steady_clock::now();
    const bool device_only_begin = !config.enable_layer_hash_trace;
    if (!device_only_begin) {
        result = cuda.dsv4_mhc_begin(
            device, device_mhc_weights[0U][0U], hidden, weighted,
            layer_input);
    } else {
        result = cuda.dsv4_mhc_begin_device(
            device, device_mhc_weights[0U][0U], hidden);
    }
    graph_stats.mhc_pre_nanoseconds += elapsed_nanoseconds(phase_started);
    if (!result.ok()) return result;
    pending_mhc_attention_transition = false;
    completed_attention_mhc_transition = false;
    completed_router_projection = false;
    deferred_attention_moe_input = false;

    constexpr std::uint32_t branch_count = 2U * kLayers;
    for (std::uint32_t flat = 0U; flat < branch_count; ++flat) {
        const auto layer = flat / 2U;
        const auto branch_index = flat % 2U;
        const std::string branch = branch_index == 0U ? "attn" : "ffn";
        if (config.enable_layer_hash_trace) {
            record_operation_hash(position, token, layer,
                                  branch + "_mhc_pre", weighted);
            record_operation_hash(position, token, layer,
                                  branch + "_norm", layer_input);
        }

        std::vector<float> branch_output(
            config.enable_layer_hash_trace ? kHidden : 0U);
        phase_started = std::chrono::steady_clock::now();
        if (branch_index == 0U) {
            result = attention(
                layer, layer_input, position, branch_output);
            graph_stats.attention_nanoseconds +=
                elapsed_nanoseconds(phase_started);
        } else {
            result = moe(
                layer, token, layer_input, branch_output, position);
            graph_stats.moe_nanoseconds += elapsed_nanoseconds(phase_started);
        }
        if (!result.ok()) return result;
        if (config.enable_layer_hash_trace) {
            record_operation_hash(position, token, layer,
                                  branch + "_output", branch_output);
        }

        phase_started = std::chrono::steady_clock::now();
        if (completed_attention_mhc_transition) {
            if (branch_index != 0U || config.enable_layer_hash_trace) {
                result.errors.emplace_back(
                    "DeepSeek combined attention transition is out of order");
            } else {
                if (!deferred_attention_moe_input) {
                    std::copy(combined_attention_mhc_input.begin(),
                              combined_attention_mhc_input.end(),
                              layer_input.begin());
                }
                completed_attention_mhc_transition = false;
            }
        } else if (flat + 1U < branch_count) {
            const auto next = flat + 1U;
            const auto next_layer = next / 2U;
            const auto next_branch = next % 2U;
            const bool combine_with_attention =
                branch_index == 1U && next_branch == 0U &&
                !config.enable_layer_hash_trace &&
                attention_state[next_layer].indexer_compressor.ratio == 0U;
            if (combine_with_attention) {
                if (pending_mhc_attention_transition) {
                    result.errors.emplace_back(
                        "DeepSeek mHC attention transition is already pending");
                } else {
                    pending_mhc_attention_transition = true;
                }
            } else {
                auto post_output = config.enable_layer_hash_trace
                    ? hidden : std::span<float>{};
                result = cuda.dsv4_mhc_transition_device(
                    device, device_mhc_weights[next_layer][next_branch],
                    weighted, layer_input, post_output);
            }
        } else if (fuse_head) {
            device_head_context = {};
            device_head_context.owner = this;
            result = cuda.enqueue_dsv4_mhc_finish_head_device(
                device, head_lease.weight(), device_head_callback,
                &device_head_context);
        } else {
            result = cuda.dsv4_mhc_finish_device(device, hidden);
        }
        graph_stats.mhc_post_nanoseconds +=
            elapsed_nanoseconds(phase_started);
        if (!result.ok()) return result;
        if (config.enable_layer_hash_trace) {
            record_operation_hash(position, token, layer,
                                  branch + "_mhc_post", hidden);
            if (branch_index == 1U) {
                record_layer_hash(position, token, layer, hidden);
            }
        }
    }
    auto moe_collected = collect_host_routed_moe_chain();
    if (!moe_collected.ok()) {
        append_errors(result, std::move(moe_collected.errors));
    }
    if (fuse_head && result.ok()) {
        auto completed = cuda.complete_dsv4_mhc_head_device(
            device, *fused_logits);
        if (!completed.ok()) {
            append_errors(result, std::move(completed.errors));
        }
        if (!device_head_context.result.ok()) {
            append_errors(result,
                          std::move(device_head_context.result.errors));
        }
        if (!device_head_context.invoked) {
            result.errors.emplace_back(
                "DeepSeek output-head host callback was not invoked");
        }
    }
    if (pending_mhc_attention_transition) {
        result.errors.emplace_back(
            "DeepSeek mHC attention transition remained pending");
    }
    if (result.ok() && fused_logits != nullptr && !fuse_head) {
        result = head_logits(hidden, *fused_logits);
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::device_mhc_forward_prefill_page(
    std::span<const std::uint32_t> tokens, std::uint32_t position_base,
    std::span<float> hidden) {
    auto result = device_mhc_forward_prefill_page_impl(
        tokens, position_base, hidden);
    // A failure can leave the chain index set; the token-major path must
    // find it empty.
    host_moe_chain_row.reset();
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::device_mhc_forward_prefill_page_impl(
    std::span<const std::uint32_t> tokens, std::uint32_t position_base,
    std::span<float> hidden) {
    ValidationResult result;
    const auto rows = tokens.size();
    const auto stride = static_cast<std::size_t>(kMhc) * kHidden;
    if (rows == 0U || rows > config.prefill_page_tokens ||
        hidden.size() != rows * stride ||
        position_base > config.maximum_context_tokens - rows) {
        result.errors.emplace_back(
            "DeepSeek device mHC prefill page has incompatible dimensions");
        return result;
    }
    const auto device = devices[mhc_slot];
    result = cuda.dsv4_mhc_reserve_slots(
        device, static_cast<std::uint32_t>(rows));
    if (!result.ok()) return result;

    const bool trace = config.enable_layer_hash_trace;
    // A non-empty attention output selects the unfused physical attention
    // command. The fused one cannot be used here: it defers the MoE input and
    // holds the layer's KV device leases until the collect, and a block
    // refuses to be appended to while any lease is outstanding, so the second
    // row of any page would fail. The branch itself still reaches the mHC
    // workspace on the device either way.
    std::vector<float> layer_inputs(rows * kHidden);
    std::vector<float> branch_outputs(rows * kHidden);
    std::vector<std::uint32_t> row_slots(rows);
    for (std::size_t row = 0U; row < rows; ++row) {
        row_slots[row] = static_cast<std::uint32_t>(row);
    }
    std::vector<float> router_logits(rows * kExperts);
    std::vector<Dsv4Route> routes(rows);
    // Prefill puts the routed experts on the GPU once a page is wide enough to
    // pay for uploading each distinct expert: one upload then serves every row
    // that chose it, instead of every row reading the weights out of host DRAM
    // again. Decode never reaches here and keeps the CPU shards.
    const bool page_moe_on_device =
        config.prefill_device_moe_minimum_rows != 0U &&
        rows >= config.prefill_device_moe_minimum_rows;
    std::vector<float> moe_outputs(page_moe_on_device ? rows * kHidden : 0U);
    std::vector<std::vector<float>> weighted(
        rows, std::vector<float>(trace ? kHidden : 0U));
    const auto layer_input_row = [&](std::size_t row) {
        return std::span<float>(layer_inputs).subspan(row * kHidden, kHidden);
    };

    const auto select = [&](std::size_t row) {
        return cuda.dsv4_mhc_select_slot(
            device, static_cast<std::uint32_t>(row));
    };
    // Neither branch here is the fused command that carries state across a
    // row's attention or MoE call, so none of those flags may be set. They
    // would otherwise leak from one row into the next.
    const auto fused_state_is_clear = [this]() {
        return !pending_mhc_attention_transition &&
               !completed_attention_mhc_transition &&
               !completed_router_projection && !deferred_attention_moe_input;
    };

    for (std::size_t row = 0U; row < rows; ++row) {
        result = select(row);
        if (!result.ok()) return result;
        const auto hidden_row = hidden.subspan(row * stride, stride);
        const auto phase_started = std::chrono::steady_clock::now();
        result = cuda.dsv4_mhc_begin(
            device, device_mhc_weights[0U][0U], hidden_row, weighted[row],
            layer_input_row(row));
        graph_stats.mhc_pre_nanoseconds += elapsed_nanoseconds(phase_started);
        if (!result.ok()) return result;
    }

    constexpr std::uint32_t branch_count = 2U * kLayers;
    for (std::uint32_t flat = 0U; flat < branch_count; ++flat) {
        const auto layer = flat / 2U;
        const auto branch_index = flat % 2U;
        const std::string branch = branch_index == 0U ? "attn" : "ffn";
        if (branch_index == 0U) {
            // One call attends the whole page: the query, key/value and output
            // projections become three row-batched matmuls instead of three
            // per row, and each row still appends and attends in position
            // order behind them.
            const auto phase_started = std::chrono::steady_clock::now();
            result = attention_page(layer, layer_inputs, position_base,
                                    branch_outputs, row_slots);
            graph_stats.attention_nanoseconds +=
                elapsed_nanoseconds(phase_started);
            if (!result.ok()) return result;
        } else {
            // Route the page in one batched projection, then decode each
            // selected expert's weight tile once for every row that chose it.
            // The whole page must be routed before any of it is executed, so
            // each row's mHC-pre and norm records are emitted here to keep a
            // row's operation order the same as the token-major path's.
            // The router stays per row. Its logits are not rounded to BF16,
            // and the row-batched projection reassociates the accumulation:
            // the selection is unchanged but the coefficients move by a ULP,
            // which is a different model.
            const auto router_started = std::chrono::steady_clock::now();
            for (std::size_t row = 0U; row < rows; ++row) {
                const auto position =
                    position_base + static_cast<std::uint32_t>(row);
                auto logits = std::span<float>(router_logits)
                    .subspan(row * kExperts, kExperts);
                result = linear(layer_device(layer),
                                layer_prefix(layer) + "ffn.gate", kExperts,
                                kHidden, layer_input_row(row), logits, false);
                if (!result.ok()) return result;
                if (trace) {
                    record_operation_hash(position, tokens[row], layer,
                                          "ffn_mhc_pre", weighted[row]);
                    record_operation_hash(position, tokens[row], layer,
                                          "ffn_norm", layer_input_row(row));
                }
                result = route_moe(
                    layer, tokens[row],
                    std::span<const float>(router_logits)
                        .subspan(row * kExperts, kExperts),
                    position, routes[row]);
                if (!result.ok()) return result;
            }
            graph_stats.moe_router_nanoseconds +=
                elapsed_nanoseconds(router_started);
            if (page_moe_on_device) {
                // Prefill places the routed experts on the GPU: each distinct
                // expert of the page is uploaded once and applied to all its
                // rows as a matmul. Decode keeps them in the NUMA-local CPU
                // shards, where the weights already live and a step has six
                // experts and one row.
                const auto phase_started = std::chrono::steady_clock::now();
                result = execute_moe_page(layer, routes, layer_inputs,
                                          moe_outputs);
                graph_stats.moe_nanoseconds +=
                    elapsed_nanoseconds(phase_started);
                if (!result.ok()) return result;
            }
        }
        for (std::size_t row = 0U; row < rows; ++row) {
            result = select(row);
            if (!result.ok()) return result;
            // Both pending-callback tables are addressed by row for as long as
            // one layer is being swept across the page.
            host_moe_chain_row = static_cast<std::uint32_t>(row);
            const auto position =
                position_base + static_cast<std::uint32_t>(row);
            const auto hidden_row = hidden.subspan(row * stride, stride);
            if (trace && branch_index == 0U) {
                record_operation_hash(position, tokens[row], layer,
                                      "attn_mhc_pre", weighted[row]);
                record_operation_hash(position, tokens[row], layer,
                                      "attn_norm", layer_input_row(row));
            }

            // The attention branch left its result in this row's device
            // workspace. The routed experts, when they run on the GPU, produce
            // a host row instead, which the transition below uploads.
            const auto attended = std::span<const float>(branch_outputs)
                .subspan(row * kHidden, kHidden);
            const auto page_moe_row = std::span<const float>(moe_outputs)
                .subspan(row * kHidden, kHidden);
            std::vector<float> branch_output(
                trace && branch_index == 1U && !page_moe_on_device ? kHidden
                                                                   : 0U);
            auto phase_started = std::chrono::steady_clock::now();
            if (branch_index == 1U && !page_moe_on_device) {
                // The page was routed above, so this joins the row's shared
                // expert with the routed partial the precompute produced.
                result = execute_moe(layer, routes[row], layer_input_row(row),
                                     branch_output);
                graph_stats.moe_nanoseconds +=
                    elapsed_nanoseconds(phase_started);
                if (!result.ok()) return result;
            }
            if (trace) {
                record_operation_hash(
                    position, tokens[row], layer, branch + "_output",
                    branch_index == 0U ? attended
                    : page_moe_on_device
                        ? page_moe_row
                        : std::span<const float>(branch_output));
            }

            phase_started = std::chrono::steady_clock::now();
            const auto post_output = trace ? hidden_row : std::span<float>{};
            const bool host_branch = branch_index == 1U && page_moe_on_device;
            if (flat + 1U < branch_count) {
                const auto next = flat + 1U;
                const auto& next_weights =
                    device_mhc_weights[next / 2U][next % 2U];
                result = host_branch
                    ? cuda.dsv4_mhc_transition(
                          device, next_weights, page_moe_row, weighted[row],
                          layer_input_row(row), post_output)
                    : cuda.dsv4_mhc_transition_device(
                          device, next_weights, weighted[row],
                          layer_input_row(row), post_output);
            } else {
                result = host_branch
                    ? cuda.dsv4_mhc_finish(device, page_moe_row, hidden_row)
                    : cuda.dsv4_mhc_finish_device(device, hidden_row);
            }
            graph_stats.mhc_post_nanoseconds +=
                elapsed_nanoseconds(phase_started);
            if (!result.ok()) return result;
            if (trace) {
                record_operation_hash(position, tokens[row], layer,
                                      branch + "_mhc_post", hidden_row);
                if (branch_index == 1U) {
                    record_layer_hash(position, tokens[row], layer, hidden_row);
                }
            }
            if (!fused_state_is_clear()) {
                result.errors.emplace_back(
                    "DeepSeek page-major branch left fused state behind");
                return result;
            }
        }
        host_moe_chain_row.reset();
        if (branch_index == 1U) {
            // One collect per layer drains the page's routed-MoE callbacks.
            auto collected = collect_host_routed_moe_chain();
            if (!collected.ok()) {
                append_errors(result, std::move(collected.errors));
                return result;
            }
        }
    }

    result = select(0U);
    if (!result.ok()) return result;
    graph_stats.forward_tokens += rows;
    return result;
}

ParseResult<std::uint32_t> DeepSeekV4Runtime::Impl::forward_token(
    std::uint32_t token, std::uint32_t position, bool logits_required) {
    ParseResult<std::uint32_t> result;
    result.value = token;
    std::vector<float> hidden(static_cast<std::size_t>(kMhc) * kHidden);
    std::vector<float> fused_logits;
    auto validation = forward_hidden(
        token, position, hidden,
        logits_required ? &fused_logits : nullptr);
    if (!validation.ok()) {
        result.errors = std::move(validation.errors);
        return result;
    }
    if (!logits_required) {
        ++graph_stats.forward_tokens;
        result.value = token;
        return result;
    }

    return sample_hidden(
        token, position, hidden,
        fused_logits.empty() ? nullptr : &fused_logits);
}

// A one-token pass writes the compressor accumulators in place, may shift them
// when a block closes, and appends at most one row to each of the sliding and
// compressed caches. `position` is the position that pass will occupy, which
// fixes which single row of each cache it can reach.
SpeculativeState DeepSeekV4Runtime::Impl::capture_speculative_state(
    std::uint32_t position) const {
    SpeculativeState saved;
    saved.layers.resize(kLayers);
    saved.tokens = position;
    for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
        const auto& state = attention_state[layer];
        auto& snapshot = saved.layers[layer];
        const auto capture = [&](const CompressorState& source,
                                 CompressorSnapshot& destination) {
            if (source.ratio == 0U) return;
            destination.values = source.values;
            destination.scores = source.scores;
            // With a block cache the compressed rows live in the KV cache and
            // are undone by truncating the sequence instead.
            if (kv_cache != nullptr) return;
            destination.compressed_index = position / source.ratio;
            const auto row = source.compressed.row(destination.compressed_index);
            if (row.empty()) return;
            destination.compressed_row.assign(row.begin(), row.end());
            destination.compressed_saved = true;
        };
        capture(state.compressor, snapshot.compressor);
        capture(state.indexer_compressor, snapshot.indexer);
        if (kv_cache == nullptr && !state.sliding.empty()) {
            snapshot.sliding_index =
                static_cast<std::size_t>(position % kWindow) * kHeadDim;
            snapshot.sliding_row.assign(
                state.sliding.begin() +
                    static_cast<std::ptrdiff_t>(snapshot.sliding_index),
                state.sliding.begin() +
                    static_cast<std::ptrdiff_t>(snapshot.sliding_index + kHeadDim));
            snapshot.sliding_saved = true;
        }
    }
    return saved;
}

ValidationResult DeepSeekV4Runtime::Impl::restore_speculative_state(
    const SpeculativeState& saved) {
    ValidationResult result;
    if (kv_cache != nullptr) {
        result = kv_cache->truncate_sequence(active_sequence, saved.tokens);
        if (!result.ok()) return result;
    }
    for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
        auto& state = attention_state[layer];
        const auto& snapshot = saved.layers[layer];
        const auto restore = [&](CompressorState& target,
                                 const CompressorSnapshot& source) {
            if (target.ratio == 0U) return;
            target.values = source.values;
            target.scores = source.scores;
            if (kv_cache != nullptr) return;
            if (source.compressed_saved) {
                auto row = target.compressed.writable_row(source.compressed_index);
                if (row.size() == source.compressed_row.size()) {
                    std::copy(source.compressed_row.begin(),
                              source.compressed_row.end(), row.begin());
                }
                return;
            }
            // The row was unallocated before the pass. Nothing ever reads past
            // the accepted compressed count, so zeroing an allocation the pass
            // made is observably the state it started in, and it cannot leave
            // a speculative row behind for a later read to find.
            if (target.compressed.row(source.compressed_index).empty()) return;
            auto row = target.compressed.writable_row(source.compressed_index);
            std::fill(row.begin(), row.end(), 0.0F);
        };
        restore(state.compressor, snapshot.compressor);
        restore(state.indexer_compressor, snapshot.indexer);
        if (snapshot.sliding_saved) {
            std::copy(snapshot.sliding_row.begin(), snapshot.sliding_row.end(),
                      state.sliding.begin() +
                          static_cast<std::ptrdiff_t>(snapshot.sliding_index));
        }
    }
    return result;
}

// One lookahead step per candidate: decode `c + w`, measure how open the
// resulting distribution is, then put the runtime back. Sequential rather than
// batched because the graph carries a single sequence, so each candidate costs
// a full decode step on top of the one that produced these logits.
ValidationResult DeepSeekV4Runtime::Impl::future_entropy(
    std::span<const std::uint32_t> candidates, std::uint32_t top_n,
    std::uint32_t position, std::span<double> normalized_entropy) {
    ValidationResult result;
    const auto speculative_position = position + 1U;
    if (speculative_position >= config.maximum_context_tokens) {
        result.errors.emplace_back(
            "future-entropy lookahead exceeds the configured context ceiling");
        return result;
    }
    const auto lookahead_started = std::chrono::steady_clock::now();
    const auto saved = capture_speculative_state(speculative_position);
    speculative_pass = true;
    std::vector<float> hidden(static_cast<std::size_t>(kMhc) * kHidden);
    std::vector<float> logits;
    for (std::size_t index = 0U; index < candidates.size(); ++index) {
        ++graph_stats.future_entropy_passes;
        result = forward_hidden(
            candidates[index], speculative_position, hidden,
            config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice
                ? &logits : nullptr);
        if (result.ok() &&
            config.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice) {
            result = head_logits(hidden, logits);
        }
        // Restore whether or not the pass succeeded: a half-applied pass would
        // desynchronize the caches from the accepted sequence.
        auto restored = restore_speculative_state(saved);
        if (!result.ok()) break;
        if (!restored.ok()) {
            result = std::move(restored);
            break;
        }
        normalized_entropy[index] = normalized_top_n_entropy(logits, top_n);
    }
    speculative_pass = false;
    graph_stats.future_entropy_nanoseconds +=
        elapsed_nanoseconds(lookahead_started);
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::head_logits(
    std::span<const float> hidden, std::vector<float>& logits) {
    ValidationResult validation;
    if (hidden.size() != static_cast<std::size_t>(kMhc) * kHidden) {
        validation.errors.emplace_back(
            "DeepSeek output head received an invalid hidden-state shape");
        return validation;
    }

    auto head_projection = host_tensor("hc_head_fn", kMhc * kMhc * kHidden);
    auto head_scale = host_tensor("hc_head_scale", 1U);
    auto head_base = host_tensor("hc_head_base", kMhc);
    if (!head_projection.ok()) append_errors(validation, std::move(head_projection.errors));
    if (!head_scale.ok()) append_errors(validation, std::move(head_scale.errors));
    if (!head_base.ok()) append_errors(validation, std::move(head_base.errors));
    if (!validation.ok()) return validation;
    double square_sum = 0.0;
    for (const float value : hidden) square_sum += static_cast<double>(value) * value;
    const float reciprocal = 1.0F / std::sqrt(
        static_cast<float>(square_sum / static_cast<double>(hidden.size())) +
        kRmsEpsilon);
    std::vector<float> reduced(kHidden, 0.0F);
    for (std::uint32_t copy = 0U; copy < kMhc; ++copy) {
        double projected = 0.0;
        const auto row = static_cast<std::size_t>(copy) * hidden.size();
        for (std::size_t column = 0U; column < hidden.size(); ++column) {
            projected += static_cast<double>((*head_projection.value)[row + column]) *
                         hidden[column];
        }
        const float coefficient = sigmoid_f32(
            static_cast<float>(projected) * reciprocal * (*head_scale.value)[0] +
            (*head_base.value)[copy]) + kRmsEpsilon;
        for (std::uint32_t column = 0U; column < kHidden; ++column) {
            reduced[column] += coefficient *
                hidden[static_cast<std::size_t>(copy) * kHidden + column];
        }
    }
    round_bf16(reduced);
    validation = norm(reduced, reduced, "norm.weight");
    if (!validation.ok()) return validation;
    logits.assign(kVocabulary, 0.0F);
    return linear(layer_device(kLayers - 1U), "head", kVocabulary,
                  kHidden, reduced, logits, false);
}

ParseResult<std::uint32_t> DeepSeekV4Runtime::Impl::sample_hidden(
    std::uint32_t token, std::uint32_t position,
    std::span<const float> hidden,
    const std::vector<float>* prepared_logits) {
    ParseResult<std::uint32_t> result;
    const auto head_started = std::chrono::steady_clock::now();
    std::vector<float> logits;
    ValidationResult validation;
    if (prepared_logits == nullptr) {
        validation = head_logits(hidden, logits);
    }
    // Stop the head timer before sampling: with a lookahead in the pipeline
    // the draw is whole forward passes, and folding those into the output head
    // would report the graph's cheapest phase as its most expensive one.
    graph_stats.output_head_nanoseconds += elapsed_nanoseconds(head_started);
    if (!validation.ok()) {
        result.errors = std::move(validation.errors);
        return result;
    }
    FutureEntropyEvaluator lookahead;
    if (active_sampling.future_entropy_candidates != 0U) {
        lookahead = [this, position](std::span<const std::uint32_t> candidates,
                                     std::uint32_t top_n,
                                     std::span<double> normalized_entropy) {
            return future_entropy(candidates, top_n, position,
                                  normalized_entropy);
        };
    }
    const auto& active_logits = prepared_logits == nullptr
        ? logits : *prepared_logits;
    last_sample = sample_logits(
        active_logits, active_sampling,
        SamplingHistory{sampled_token_counts, sampled_token_ids}, sampler,
        lookahead);
    if (!last_sample.ok()) {
        result.errors = last_sample.errors;
        return result;
    }
    result.value = last_sample.token;
    ++sampled_token_counts[result.value];
    sampled_token_ids.push_back(result.value);
    if (config.enable_logit_trace) {
        record_logits(position, token, result.value, active_logits);
    }
    ++graph_stats.forward_tokens;
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::flush_deferred_routes() {
    ValidationResult result;
    std::stable_sort(
        deferred_route_events.begin(), deferred_route_events.end(),
        [](const RouteEvent& left, const RouteEvent& right) {
            if (left.token_position != right.token_position) {
                return left.token_position < right.token_position;
            }
            return left.layer < right.layer;
        });
    for (const auto& event : deferred_route_events) {
        if (config.expert_prefetch_predictions != 0U) {
            route_predictor.observe(event);
        }
        if (route_trace.is_open()) {
            auto written = route_trace.write(event);
            if (!written.ok()) {
                append_errors(result, std::move(written.errors));
                break;
            }
        }
    }
    deferred_route_events.clear();
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::flush_prefill_observability() {
    ValidationResult result;
    const auto position_layer_less = [](const auto& left, const auto& right) {
        if (left.position != right.position) return left.position < right.position;
        return left.layer < right.layer;
    };
    if (config.enable_layer_hash_trace) {
        std::stable_sort(diagnostics.layer_hashes.begin(),
                         diagnostics.layer_hashes.end(), position_layer_less);
        diagnostics.layer_hash_trace_hash = diagnostic_hash_u32(
            kDiagnosticFnvOffset, kLayers);
        for (const auto& record : diagnostics.layer_hashes) {
            auto aggregate = diagnostics.layer_hash_trace_hash;
            aggregate = diagnostic_hash_u32(aggregate, record.position);
            aggregate = diagnostic_hash_u32(aggregate, record.input_token);
            aggregate = diagnostic_hash_u32(aggregate, record.layer);
            diagnostics.layer_hash_trace_hash = diagnostic_hash_u64(
                aggregate, record.bf16_hash);
        }
        std::stable_sort(diagnostics.operation_hashes.begin(),
                         diagnostics.operation_hashes.end(),
                         position_layer_less);
    }
    return flush_deferred_routes();
}

ParseResult<std::uint32_t> DeepSeekV4Runtime::Impl::forward_prefill(
    std::span<const std::uint32_t> tokens, std::uint32_t position_base) {
    ParseResult<std::uint32_t> result;
    if (tokens.empty() || tokens.size() > config.maximum_context_tokens ||
        position_base > config.maximum_context_tokens - tokens.size()) {
        result.errors.emplace_back(
            "DeepSeek prefill requires a non-empty in-context token range");
        return result;
    }
    const auto hidden_stride = static_cast<std::size_t>(kMhc) * kHidden;
    const bool physical =
        config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice;
    if (physical && config.prefill_page_tokens > 1U && tokens.size() > 1U) {
        // Page-major over every prompt row but the last. The last row keeps
        // the token-major path because it is the only one that may need
        // logits, and the fused output head is defined only there.
        defer_prefill_observability = true;
        const auto paged_tokens = tokens.size() - 1U;
        const auto fail = [&](ValidationResult&& status) {
            defer_prefill_observability = false;
            result.errors = std::move(status.errors);
            return result;
        };
        for (std::size_t page_begin = 0U; page_begin < paged_tokens;
             page_begin += config.prefill_page_tokens) {
            const auto page_rows = std::min<std::size_t>(
                config.prefill_page_tokens, paged_tokens - page_begin);
            std::vector<float> hidden(page_rows * hidden_stride);
            for (std::size_t row = 0U; row < page_rows; ++row) {
                const auto embedding_started = std::chrono::steady_clock::now();
                auto embedded = embed(
                    tokens[page_begin + row],
                    std::span<float>(hidden).subspan(
                        row * hidden_stride, hidden_stride));
                graph_stats.embedding_nanoseconds +=
                    elapsed_nanoseconds(embedding_started);
                if (!embedded.ok()) return fail(std::move(embedded));
            }
            ++graph_stats.prefill_pages;
            graph_stats.prefill_max_page_tokens = std::max<std::uint64_t>(
                graph_stats.prefill_max_page_tokens, page_rows);
            graph_stats.prefill_max_workspace_bytes =
                std::max<std::uint64_t>(
                    graph_stats.prefill_max_workspace_bytes,
                    static_cast<std::uint64_t>(page_rows) * hidden_stride *
                        sizeof(float));
            auto executed = device_mhc_forward_prefill_page(
                tokens.subspan(page_begin, page_rows),
                position_base + static_cast<std::uint32_t>(page_begin),
                hidden);
            if (!executed.ok()) return fail(std::move(executed));
        }
        defer_prefill_observability = false;
        auto flushed = flush_prefill_observability();
        if (!flushed.ok()) {
            result.errors = std::move(flushed.errors);
            return result;
        }
        ++graph_stats.prefill_pages;
        graph_stats.prefill_max_page_tokens = std::max<std::uint64_t>(
            graph_stats.prefill_max_page_tokens, 1U);
        return forward_token(
            tokens.back(),
            position_base + static_cast<std::uint32_t>(tokens.size() - 1U),
            true);
    }
    if (config.prefill_page_tokens == 1U || physical) {
        for (std::size_t position = 0U; position < tokens.size(); ++position) {
            ++graph_stats.prefill_pages;
            graph_stats.prefill_max_page_tokens = 1U;
            graph_stats.prefill_max_workspace_bytes = std::max<std::uint64_t>(
                graph_stats.prefill_max_workspace_bytes,
                static_cast<std::uint64_t>(hidden_stride) * sizeof(float));
            result = forward_token(
                tokens[position],
                position_base + static_cast<std::uint32_t>(position),
                position + 1U == tokens.size());
            if (!result.ok()) return result;
        }
        return result;
    }
    defer_prefill_observability = true;
    // Expert residency, not activation memory, is what bounds prefill. One
    // layer's 256 routed experts are 3.4 GB and fit the VRAM cache; all 43
    // layers together are 147 GB and do not. Sweeping every layer inside the
    // page loop therefore evicts the cache once per page: a 3,565-token
    // prefill moved 3,367 GB of demand H2D, 22.9x the 147 GB it must move,
    // with 745,172 evictions and 59% of the phase spent in demand wait.
    // Visiting layers outermost over a tile of pages touches each layer's
    // experts once per tile instead of once per page. A tile equal to
    // prefill_page_tokens reproduces the page-major nest exactly.
    const auto tile_tokens = config.prefill_layer_tile_tokens == 0U
        ? tokens.size()
        : std::min<std::size_t>(config.prefill_layer_tile_tokens, tokens.size());
    for (std::size_t tile_begin = 0U; tile_begin < tokens.size();
         tile_begin += tile_tokens) {
        const auto tile_rows =
            std::min<std::size_t>(tile_tokens, tokens.size() - tile_begin);
        std::vector<float> hidden(tile_rows * hidden_stride);

        for (std::size_t row = 0U; row < tile_rows; ++row) {
            const auto embedding_started = std::chrono::steady_clock::now();
            auto status = embed(tokens[tile_begin + row],
                                std::span<float>(hidden).subspan(
                                    row * hidden_stride, hidden_stride));
            graph_stats.embedding_nanoseconds +=
                elapsed_nanoseconds(embedding_started);
            if (!status.ok()) {
                defer_prefill_observability = false;
                result.errors = std::move(status.errors);
                return result;
            }
        }

        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            for (std::size_t page_begin = 0U; page_begin < tile_rows;
                 page_begin += config.prefill_page_tokens) {
                const auto page_rows = static_cast<std::uint32_t>(
                    std::min<std::size_t>(config.prefill_page_tokens,
                                          tile_rows - page_begin));
                if (layer == 0U) {
                    ++graph_stats.prefill_pages;
                    graph_stats.prefill_max_page_tokens =
                        std::max<std::uint64_t>(
                            graph_stats.prefill_max_page_tokens, page_rows);
                    graph_stats.prefill_max_workspace_bytes =
                        std::max<std::uint64_t>(
                            graph_stats.prefill_max_workspace_bytes,
                            static_cast<std::uint64_t>(tile_rows) *
                                    hidden_stride * sizeof(float) +
                                static_cast<std::uint64_t>(page_rows) *
                                    (2U * hidden_stride + 2U * kHidden +
                                     kQueryRank +
                                     static_cast<std::size_t>(kHeads) *
                                         kHeadDim +
                                     kHeadDim) *
                                    sizeof(float));
                }
                const auto absolute_page_begin =
                    position_base +
                    static_cast<std::uint32_t>(tile_begin + page_begin);
                auto status = block_page(
                    layer, tokens.subspan(tile_begin + page_begin, page_rows),
                    std::span<float>(hidden).subspan(
                        page_begin * hidden_stride, page_rows * hidden_stride),
                    absolute_page_begin);
                if (!status.ok()) {
                    defer_prefill_observability = false;
                    result.errors = std::move(status.errors);
                    return result;
                }
                for (std::uint32_t row = 0U; row < page_rows; ++row) {
                    const auto token_index = tile_begin + page_begin + row;
                    if (config.enable_layer_hash_trace) {
                        record_layer_hash(
                            position_base +
                                static_cast<std::uint32_t>(token_index),
                            tokens[token_index], layer,
                            std::span<const float>(hidden).subspan(
                                (page_begin + row) * hidden_stride,
                                hidden_stride));
                    }
                }
            }
        }
        auto routes_flushed = flush_deferred_routes();
        if (!routes_flushed.ok()) {
            defer_prefill_observability = false;
            result.errors = std::move(routes_flushed.errors);
            return result;
        }
        graph_stats.forward_tokens += tile_rows;
        if (tile_begin + tile_rows == tokens.size()) {
            --graph_stats.forward_tokens;
            const auto last_row = std::span<const float>(hidden).last(hidden_stride);
            result = sample_hidden(tokens.back(),
                                   position_base + static_cast<std::uint32_t>(
                                       tokens.size() - 1U),
                                   last_row);
            if (!result.ok()) {
                defer_prefill_observability = false;
                return result;
            }
        }
    }
    defer_prefill_observability = false;
    auto flushed = flush_prefill_observability();
    if (!flushed.ok()) result.errors = std::move(flushed.errors);
    return result;
}

DeepSeekV4Runtime::DeepSeekV4Runtime() : impl_(std::make_unique<Impl>()) {}
DeepSeekV4Runtime::~DeepSeekV4Runtime() = default;
DeepSeekV4Runtime::DeepSeekV4Runtime(DeepSeekV4Runtime&&) noexcept = default;
DeepSeekV4Runtime& DeepSeekV4Runtime::operator=(DeepSeekV4Runtime&&) noexcept = default;

ValidationResult DeepSeekV4Runtime::initialize(
    const std::string& model_directory, const Dsv4RuntimeConfig& caller_config) {
    // Every zero-means-probe field is filled in here, once, so nothing below
    // this line has to know which numbers came from the caller and which from
    // the machine.
    Dsv4RuntimeConfig config(caller_config);
    resolve_hardware_defaults(config);
    ValidationResult result;
    const auto initialization_started = std::chrono::steady_clock::now();
    if (impl_->initialized) {
        result.errors.emplace_back("DeepSeek runtime is already initialized");
        return result;
    }
    result = validate_common_runtime_config(
        config.devices, config.vram_cache_fraction,
        config.sampling_temperature, "DeepSeek");
    if (!result.ok()) return result;
    if (config.decode_topology == Dsv4DecodeTopology::RankLocalTp2) {
        // Fail closed before the checkpoint is opened or any weight is
        // resident. Conditions that need the manifest are re-checked at
        // admission; these are the ones knowable from the build and the
        // request alone, and they must not cost a model load to discover.
#if !defined(STRATA_HAS_NCCL)
        result.errors.emplace_back(
            "rank-local decode was requested but this build has no NCCL "
            "support; rebuild with -DSTRATA_ENABLE_NCCL=ON");
#endif
        if (config.devices.size() != kDsv4RankLocalWorld) {
            result.errors.emplace_back(
                "rank-local decode requires exactly " +
                std::to_string(kDsv4RankLocalWorld) + " CUDA devices, got " +
                std::to_string(config.devices.size()));
        }
        if (config.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice) {
            result.errors.emplace_back(
                "rank-local decode requires the physical-device DSV4 KV mode");
        }
        std::array<std::vector<int>, kDsv4RankLocalWorld> rank_cpus;
        auto cpu_plan = plan_dsv4_rank_local_cpus(
            NumaTopology::detect(), kDsv4RankLocalMinimumCpusPerRank,
            rank_cpus);
        if (!cpu_plan.ok()) {
            result.errors.insert(result.errors.end(), cpu_plan.errors.begin(),
                                 cpu_plan.errors.end());
        }
        if (!result.ok()) return result;
    }
    const auto model_context =
        deepseek_v4_flash_0731_spec().max_context_tokens;
    if (config.maximum_context_tokens == 0U ||
        config.maximum_context_tokens > model_context) {
        result.errors.emplace_back(
            "DeepSeek runtime context must be within the model limit [1, " +
            std::to_string(model_context) + "] tokens");
        return result;
    }
    if (config.kv_cache_mode != Dsv4KvCacheMode::ScalarOracle &&
        config.kv_block_rows == 0U) {
        result.errors.emplace_back(
            "DeepSeek KV block row count must be positive");
        return result;
    }
    if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice &&
        config.kv_block_rows != kDsv4PhysicalKvBlockRows) {
        result.errors.emplace_back(
            "DeepSeek physical KV requires 256-source-token blocks");
        return result;
    }
    if (config.enable_gpu_lightning_indexer &&
        config.kv_cache_mode != Dsv4KvCacheMode::Block) {
        result.errors.emplace_back(
            "GPU Lightning Indexer requires the exact compact block KV cache");
        return result;
    }
    if (!config.device_kv_cache_bytes.empty() &&
        config.device_kv_cache_bytes.size() != config.devices.size()) {
        result.errors.emplace_back(
            "DeepSeek KV device budget count must match the device count");
        return result;
    }
    if (config.prefill_page_tokens == 0U ||
        config.prefill_page_tokens > kMaximumPrefillPageTokens) {
        result.errors.emplace_back(
            "DeepSeek prefill page must be within [1, 8192] tokens");
        return result;
    }
    if (config.prefill_layer_tile_tokens != 0U &&
        (config.prefill_layer_tile_tokens < config.prefill_page_tokens ||
         config.prefill_layer_tile_tokens > config.maximum_context_tokens)) {
        result.errors.emplace_back(
            "DeepSeek prefill layer tile must be zero or within "
            "[prefill page tokens, maximum context tokens]");
        return result;
    }
    if (config.enable_logit_trace &&
        (config.logit_trace_top_k == 0U ||
         config.logit_trace_top_k > kVocabulary)) {
        result.errors.emplace_back(
            "DeepSeek logit trace top-K must be within [1, 129280]");
        return result;
    }
    if (config.host_attention_threads > kHeads) {
        result.errors.emplace_back(
            "DeepSeek host attention worker count must not exceed 64");
        return result;
    }
    if (config.resident_read_workers == 0U ||
        config.resident_read_workers > 64U) {
        result.errors.emplace_back(
            "DeepSeek resident read worker count must be within [1, 64]");
        return result;
    }
    if (config.spine_warmup_workers == 0U ||
        config.spine_warmup_workers > 64U) {
        result.errors.emplace_back(
            "DeepSeek spine warmup worker count must be within [1, 64]");
        return result;
    }
    if (config.expert_prefetch_predictions > kExperts ||
        !std::isfinite(config.expert_prefetch_minimum_confidence) ||
        config.expert_prefetch_minimum_confidence < 0.0 ||
        config.expert_prefetch_minimum_confidence > 1.0 ||
        (config.expert_prefetch_predictions != 0U &&
         (config.expert_prefetch_queue_depth == 0U ||
          config.expert_prefetch_queue_depth > 1024U ||
          config.expert_prefetch_byte_budget == 0U ||
          config.expert_prefetch_lease_ticks == 0U))) {
        result.errors.emplace_back(
            "DeepSeek expert prefetch requires bounded predictions, bytes, queue, "
            "lease, and confidence");
        return result;
    }
    if (config.enable_host_routed_moe &&
        config.expert_prefetch_predictions != 0U) {
        result.errors.emplace_back(
            "DeepSeek host-routed MoE replaces routed GPU prefetch");
        return result;
    }
    if (config.enable_dspark) {
        result.errors.emplace_back(
            "DSpark tensors are verified, but speculative execution is not enabled in "
            "the base-model executor; refusing a silent approximation");
        return result;
    }
    if (config.prepack_mhc_projection &&
        config.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice &&
        !dsv4_mhc_prepacked_supported()) {
        result.errors.emplace_back(
            "DeepSeek prepacked mHC requires x86 AVX2");
        return result;
    }
    impl_ = std::make_unique<Impl>();
    auto checkpoint = Dsv4CheckpointReader::open(model_directory);
    if (!checkpoint.ok()) {
        result.errors = std::move(checkpoint.errors);
        return result;
    }
    auto tokenizer = ModelTokenizer::load(
        (std::filesystem::path(model_directory) / "tokenizer.json").string());
    if (!tokenizer.ok()) {
        result.errors = std::move(tokenizer.errors);
        return result;
    }
    // The static expert tier lives on a device outside the execution set, so
    // it needs its own context and weight arena. It is included here rather
    // than in config.devices because it must not join the layer schedule, the
    // KV placement or the rank pair -- it only holds weights and computes the
    // experts it holds.
    std::vector<int> context_devices(config.devices.begin(), config.devices.end());
    const bool tier_requested = !config.static_expert_plan_path.empty() &&
                                config.static_expert_tier_device >= 0;
    if (tier_requested) {
        const auto tier_device = config.static_expert_tier_device;
        if (std::find(context_devices.begin(), context_devices.end(),
                      tier_device) != context_devices.end()) {
            result.errors.emplace_back(
                "static expert tier device " + std::to_string(tier_device) +
                " is already an execution device; the tier must be separate");
            return result;
        }
        context_devices.push_back(tier_device);
    }
    result = impl_->cuda.initialize(context_devices, config.detailed_timing);
    if (!result.ok()) return result;
    if (tier_requested && config.static_expert_tier_bytes != 0U) {
        // Reserve up front: the tier allocates thousands of small weights and
        // the arena refuses a per-weight fallback once enabled.
        result = impl_->cuda.reserve_weight_arena(
            config.static_expert_tier_device,
            config.static_expert_tier_bytes);
        if (!result.ok()) return result;
    }
    if (config.enable_flash_attention) {
        for (const int device : config.devices) {
            result = impl_->cuda.validate_flash_attention_device(device);
            if (!result.ok()) return result;
        }
    }
    if (config.enable_gpu_lightning_indexer) {
        for (const int device : config.devices) {
            result = impl_->cuda.validate_lightning_index_device(device);
            if (!result.ok()) return result;
        }
    }

    auto effective_explicit_vram_budget = config.explicit_vram_budget_bytes;
    if (config.decode_topology == Dsv4DecodeTopology::RankLocalTp2) {
        impl_->rank_local_initial_device_vram_bytes =
            device_vram_used_bytes(config.devices);
        if (impl_->rank_local_initial_device_vram_bytes.size() !=
            config.devices.size()) {
            result.errors.emplace_back(
                "rank-local initial VRAM ledger does not match devices");
            return result;
        }
        for (std::size_t slot = 0U;
             slot < impl_->rank_local_initial_device_vram_bytes.size();
             ++slot) {
            const auto initial =
                impl_->rank_local_initial_device_vram_bytes[slot];
            const auto ceiling = rank_local_vram_ceiling(config.devices[slot]);
            if (initial >= ceiling) {
                result.errors.emplace_back(
                    "rank-local CUDA device " +
                    std::to_string(config.devices[slot]) +
                    " already uses " + std::to_string(initial) +
                    " B, which leaves no room below the " +
                    std::to_string(ceiling) + " B program ceiling");
                return result;
            }
            const auto available = ceiling - initial;
            effective_explicit_vram_budget =
                effective_explicit_vram_budget == 0U
                    ? available
                    : std::min(effective_explicit_vram_budget, available);
        }
    }
    auto device_plan = plan_runtime_devices(
        config.devices, config.vram_cache_fraction, kDeviceWorkspaceReserve,
        2ULL << 30U, "DeepSeek", effective_explicit_vram_budget);
    if (!device_plan.ok()) {
        result.errors = std::move(device_plan.errors);
        return result;
    }
    auto capacities = std::move(device_plan.value.budgets);
    auto weight_capacities = std::move(device_plan.value.weight_capacities);
    auto kv_device_capacities = config.device_kv_cache_bytes;
    if (kv_device_capacities.empty()) {
        kv_device_capacities.resize(config.devices.size());
        if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
            if (config.decode_topology == Dsv4DecodeTopology::RankLocalTp2) {
                auto physical = dsv4_physical_kv_admission(
                    config.maximum_context_tokens);
                if (!physical.ok()) {
                    result.errors = std::move(physical.errors);
                    return result;
                }
                std::fill(kv_device_capacities.begin(),
                          kv_device_capacities.end(),
                          physical.value.payload_bytes);
            } else {
                const auto add_pages = [&](std::size_t slot,
                                           Dsv4KvBlockKind kind,
                                           std::uint32_t ratio,
                                           std::uint64_t rows) {
                    const auto capacity_rows = dsv4_kv_block_rows(
                        kind, ratio, true);
                    const auto format = dsv4_kv_format(kind, false, true);
                    const auto page_bytes = dsv4_kv_row_bytes(kind, format) *
                                            capacity_rows;
                    const auto pages = (rows + capacity_rows - 1U) /
                                       capacity_rows;
                    if (capacity_rows == 0U || page_bytes == 0U ||
                        pages > std::numeric_limits<std::uint64_t>::max() /
                                    page_bytes ||
                        pages * page_bytes >
                            std::numeric_limits<std::uint64_t>::max() -
                                kv_device_capacities[slot]) {
                        return false;
                    }
                    kv_device_capacities[slot] += pages * page_bytes;
                    return true;
                };
                const auto& ratios =
                    kDeepSeekV4ExecutionContract.compression_ratios;
                for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
                    const auto slot = device_plan.value.weighted_schedule[
                        layer % device_plan.value.weighted_schedule.size()];
                    const auto sliding_rows = std::min<std::uint64_t>(
                        config.maximum_context_tokens,
                        kWindow + kDsv4PhysicalKvBlockRows - 1U);
                    const auto ratio = ratios[layer];
                    bool admitted = add_pages(
                        slot, Dsv4KvBlockKind::Sliding, 1U, sliding_rows);
                    if (admitted && ratio != 0U) {
                        const auto compressed_rows =
                            (static_cast<std::uint64_t>(
                                 config.maximum_context_tokens) + ratio - 1U) /
                            ratio;
                        admitted = add_pages(
                            slot, ratio == 4U ? Dsv4KvBlockKind::Csa
                                             : Dsv4KvBlockKind::Hca,
                            ratio, compressed_rows);
                        if (admitted && ratio == 4U &&
                            config.maximum_context_tokens >
                                kIndexTopK * ratio) {
                            admitted = add_pages(
                                slot, Dsv4KvBlockKind::LearnedIndex, ratio,
                                compressed_rows);
                        }
                    }
                    if (!admitted) {
                        result.errors.emplace_back(
                            "DeepSeek physical KV capacity overflows");
                        return result;
                    }
                }
            }
        }
    }
    if (config.decode_topology == Dsv4DecodeTopology::RankLocalTp2 &&
        config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
        auto physical = dsv4_physical_kv_admission(
            config.maximum_context_tokens);
        if (!physical.ok()) {
            result.errors = std::move(physical.errors);
            return result;
        }
        for (std::size_t slot = 0U; slot < kv_device_capacities.size();
             ++slot) {
            if (kv_device_capacities[slot] < physical.value.payload_bytes) {
                result.errors.emplace_back(
                    "rank-local CUDA device " +
                    std::to_string(config.devices[slot]) +
                    " KV capacity " +
                    std::to_string(kv_device_capacities[slot]) +
                    " B is below the replicated full-context requirement " +
                    std::to_string(physical.value.payload_bytes) + " B");
                return result;
            }
        }
    }
    for (std::size_t slot = 0U; slot < weight_capacities.size(); ++slot) {
        if (kv_device_capacities[slot] >= weight_capacities[slot]) {
            result.errors.emplace_back(
                "DeepSeek KV device budget leaves no weight-cache capacity");
            return result;
        }
        weight_capacities[slot] -= kv_device_capacities[slot];
    }
    auto arena_capacities = weight_capacities;
    auto cache_weight_capacities = arena_capacities;
    // The routed-expert tier suballocates from the same weight arena as the
    // centralized prefill cache, and the tier is permanent while the cache is
    // a prefill performance term. Reserve the tier's bytes out of the cache's
    // logical capacity so the cache stops short of the arena instead of
    // failing an acquire mid-prefill, which is what an unreserved tier caused.
    if (!config.static_expert_plan_path.empty() &&
        config.static_expert_tier_bytes != 0U) {
        for (auto& capacity : cache_weight_capacities) {
            capacity = capacity > config.static_expert_tier_bytes
                ? capacity - config.static_expert_tier_bytes : 0U;
        }
    }
    const auto mhc_slot = static_cast<std::size_t>(std::distance(
        arena_capacities.begin(),
        std::max_element(arena_capacities.begin(), arena_capacities.end())));
    std::uint64_t mhc_projection_bytes = 0U;
    std::uint64_t mhc_auxiliary_bytes = 0U;
    if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
        constexpr std::uint64_t projection_payload_bytes =
            static_cast<std::uint64_t>(kMix) * kMhc * kHidden *
            sizeof(float);
        const auto projection_storage_bytes =
            CudaBackend::weight_storage_bytes(projection_payload_bytes, 0U);
        constexpr std::uint64_t boundary_count = 2U * kLayers;
        constexpr std::uint64_t auxiliary_bytes_per_boundary =
            112U + static_cast<std::uint64_t>(kHidden) *
                       sizeof(std::uint16_t);
        if (projection_storage_bytes == 0U ||
            projection_storage_bytes >
                std::numeric_limits<std::uint64_t>::max() / boundary_count) {
            result.errors.emplace_back(
                "DeepSeek device mHC projection capacity overflows");
            return result;
        }
        mhc_projection_bytes = projection_storage_bytes * boundary_count;
        mhc_auxiliary_bytes =
            auxiliary_bytes_per_boundary * boundary_count;
        if (mhc_auxiliary_bytes >= arena_capacities[mhc_slot]) {
            result.errors.emplace_back(
                "DeepSeek device mHC auxiliaries exceed device capacity");
            return result;
        }
        arena_capacities[mhc_slot] -= mhc_auxiliary_bytes;
        cache_weight_capacities = arena_capacities;
        if (mhc_projection_bytes >= cache_weight_capacities[mhc_slot]) {
            result.errors.emplace_back(
                "DeepSeek device mHC projections leave no weight cache");
            return result;
        }
        cache_weight_capacities[mhc_slot] -= mhc_projection_bytes;
        result = impl_->cuda.validate_dsv4_mhc_device(
            config.devices[mhc_slot]);
        if (!result.ok()) return result;
    }
    const auto admission_started = std::chrono::steady_clock::now();
    Dsv4AdmissionConfig admission_config;
    admission_config.host_memory_ceiling_bytes = config.host_memory_limit_bytes;
    admission_config.vram_weight_budgets = capacities;
    admission_config.host_kv_cache_bytes = config.host_kv_cache_bytes;
    admission_config.device_kv_cache_bytes = kv_device_capacities;
    admission_config.maximum_context_tokens = config.maximum_context_tokens;
    admission_config.prefill_page_tokens = config.prefill_page_tokens;
    admission_config.enable_dspark = config.enable_dspark;
    admission_config.enable_mhc_prepack =
        config.prepack_mhc_projection &&
        config.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice;
    admission_config.host_routed_experts = config.enable_host_routed_moe;
    admission_config.compact_kv_cache =
        config.kv_cache_mode != Dsv4KvCacheMode::ScalarOracle;
    admission_config.physical_kv_cache =
        config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice;
    admission_config.device_resident_mhc =
        config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice;
    admission_config.require_zero_nvme_decode = config.require_zero_nvme_decode;
    auto admission = plan_dsv4_resident_topology(checkpoint.value->manifest(),
                                                  admission_config);
    const double admission_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - admission_started).count();
    if (!admission.ok()) {
        result.errors = std::move(admission.errors);
        return result;
    }
    admission.plan.fractional_vram_budget_bytes =
        device_plan.value.fractional_budgets;
    admission.plan.explicit_vram_budget_bytes =
        device_plan.value.explicit_budgets;
    admission.plan.applied_vram_budget_bytes = capacities;
    admission.plan.vram_budget_bound.reserve(capacities.size());
    for (std::size_t slot = 0U; slot < capacities.size(); ++slot) {
        admission.plan.vram_budget_bound.emplace_back(
            runtime_budget_bound_name(
                device_plan.value.fractional_budgets[slot],
                device_plan.value.explicit_budgets[slot]));
    }

    impl_->config = config;
    impl_->sampler.seed(config.sampling_seed);
    impl_->memory = admission.plan;
    impl_->checkpoint = std::move(checkpoint.value);
    impl_->tokenizer = std::move(tokenizer.value);
    impl_->devices = config.devices;
    impl_->capacities = cache_weight_capacities;
    impl_->schedule = std::move(device_plan.value.weighted_schedule);
    impl_->mhc_slot = mhc_slot;
    if (!config.route_trace_path.empty()) {
        result = impl_->route_trace.open(config.route_trace_path);
        if (!result.ok()) return result;
    }
    if (config.verbose) {
        for (std::size_t slot = 0U; slot < impl_->devices.size(); ++slot) {
            std::cerr << "[hardware] cuda=" << impl_->devices[slot]
                      << " vram_budget_bytes=" << capacities[slot]
                      << " weight_cache_bytes="
                      << cache_weight_capacities[slot]
                      << " weight_arena_bytes=" << arena_capacities[slot]
                      << " kv_cache_bytes=" << kv_device_capacities[slot]
                      << " workspace_reserve_bytes=" << kDeviceWorkspaceReserve
                      << '\n';
        }
        if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
            std::cerr << "[hardware] deepseek_device_mhc_cuda="
                      << impl_->devices[impl_->mhc_slot]
                      << " projection_bytes=" << mhc_projection_bytes
                      << " auxiliary_bytes=" << mhc_auxiliary_bytes
                      << " cross_layer_device_state=true\n";
        }
    }
    for (std::size_t slot = 0U; slot < impl_->devices.size(); ++slot) {
        result = impl_->cuda.reserve_weight_arena(
            impl_->devices[slot], arena_capacities[slot]);
        if (!result.ok()) return result;
    }
    impl_->weights = std::make_unique<Dsv4WeightCache>(
        *impl_->checkpoint, impl_->resident, impl_->cuda,
        impl_->devices, cache_weight_capacities,
        config.expert_prefetch_predictions == 0U
            ? 0U : config.expert_prefetch_byte_budget,
        config.expert_prefetch_queue_depth,
        config.expert_prefetch_lease_ticks);
    if (config.kv_cache_mode != Dsv4KvCacheMode::ScalarOracle) {
        Dsv4KvCacheConfig kv_config;
        kv_config.block_rows = config.kv_block_rows;
        kv_config.sliding_window_rows = kWindow;
        kv_config.host_capacity_bytes = impl_->memory.host_kv_cache_bytes;
        kv_config.devices = config.devices;
        kv_config.device_capacity_bytes = kv_device_capacities;
        kv_config.physical_layout =
            config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice;
        impl_->kv_cache = std::make_unique<Dsv4KvCache>(
            std::move(kv_config), &impl_->cuda);
        result = impl_->kv_cache->validate();
        if (!result.ok()) return result;
        auto sequence = impl_->kv_cache->create_sequence();
        if (!sequence.ok()) {
            result.errors = std::move(sequence.errors);
            return result;
        }
        impl_->active_sequence = sequence.value;
    }
    if (config.host_attention_threads != 0U) {
        impl_->attention_workers = std::make_unique<HostWorkerPool>(
            config.host_attention_threads);
    } else {
        impl_->attention_workers.reset();
    }
    if (config.enable_host_routed_moe) {
        constexpr std::size_t workers = 48U;
        impl_->expert_workers = std::make_unique<HostWorkerPool>(
            workers, std::chrono::milliseconds(1));
        impl_->expert_lane_nodes.resize(workers);
        impl_->expert_lane_positions.resize(workers);
        const auto topology = NumaTopology::detect();
        result = impl_->expert_workers->parallel_for_addressed(
            workers, [&](std::size_t lane) {
                impl_->expert_lane_nodes[lane] =
                    topology.node_of_cpu(sched_getcpu());
            });
        if (!result.ok()) return result;
        for (std::size_t lane = 0U; lane < workers; ++lane) {
            const auto node = impl_->expert_lane_nodes[lane];
            if (node < 0 || node >= 2) {
                result.errors.emplace_back(
                    "DeepSeek host-routed MoE worker is outside its two NUMA nodes");
                return result;
            }
            auto& node_lanes = impl_->expert_node_lanes[
                static_cast<std::size_t>(node)];
            impl_->expert_lane_positions[lane] = node_lanes.size();
            node_lanes.push_back(lane);
        }
        if (impl_->expert_node_lanes[0].size() != workers / 2U ||
            impl_->expert_node_lanes[1].size() != workers / 2U) {
            result.errors.emplace_back(
                "DeepSeek host-routed MoE needs its workers split evenly across "
                "the NUMA nodes");
            return result;
        }
        constexpr std::size_t shards = 2U;
        constexpr std::size_t shard_intermediate =
            kExpertIntermediate / shards;
        impl_->tiled_activation.resize(
            shards * kTopK * shard_intermediate);
        impl_->tiled_routed.resize(shards * kTopK * kHidden);
    }

    ValidationResult staging_result;
    ValidationResult warmup_result;
    double staging_seconds = 0.0;
    double warmup_seconds = 0.0;
    std::atomic<bool> staging_finished{false};
    const auto stage_resident = [&] {
        const auto started = std::chrono::steady_clock::now();
        staging_result = impl_->resident.stage(*impl_->checkpoint,
                                               config.host_memory_limit_bytes,
                                               config.resident_read_workers,
                                               config.enable_dspark,
                                               config.enable_host_routed_moe,
                                               config.hugepage_expert_arena);
        staging_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        staging_finished.store(true, std::memory_order_release);
    };
    const auto warm_spine = [&] {
        const auto started = std::chrono::steady_clock::now();
        warmup_result = impl_->warmup();
        warmup_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
    };
    if (config.overlap_resident_warmup) {
        std::thread staging_thread(stage_resident);
        warm_spine();
        if (config.verbose) {
            std::cerr << "[deepseek-load] phase=spine_warmup_complete elapsed_ms="
                      << warmup_seconds * 1000.0 << '\n';
            if (!staging_finished.load(std::memory_order_acquire)) {
                std::cerr << "[deepseek-load] phase=resident_stage_wait_start\n";
            }
        }
        staging_thread.join();
    } else {
        stage_resident();
        if (staging_result.ok()) warm_spine();
        if (config.verbose) {
            std::cerr << "[deepseek-load] phase=spine_warmup_complete elapsed_ms="
                      << warmup_seconds * 1000.0 << '\n';
        }
    }
    if (config.verbose) {
        const auto stage_stats = impl_->resident.stats();
        std::cerr << "[deepseek-load] phase=resident_stage_complete elapsed_ms="
                  << staging_seconds * 1000.0
                  << " bytes=" << stage_stats.bytes
                  << " workers=" << stage_stats.workers << '\n';
    }
    if (!staging_result.ok()) {
        append_errors(result, std::move(staging_result.errors));
    }
    if (!warmup_result.ok()) {
        append_errors(result, std::move(warmup_result.errors));
    }
    if (!result.ok()) return result;
    // Page-lock after staging and warm-up have both finished, so registration
    // never races an upload reading out of the same mapping. This is a pure
    // transfer-rate optimization: if the kernel refuses to lock the pages the
    // run continues unpinned and says so, because no output byte depends on it.
    double pin_seconds = 0.0;
    if (config.pin_resident_arena) {
        const auto pin_started = std::chrono::steady_clock::now();
        auto pinned = impl_->resident.pin(impl_->cuda);
        pin_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - pin_started).count();
        if (!pinned.ok()) {
            // Always say so. A silent failure here leaves the run at the
            // pageable transfer rate while every metric claims otherwise.
            std::cerr << "[deepseek-load] resident arena not pinned: "
                      << pinned.errors.front() << '\n';
        }
    }
    if (config.enable_device_moe) {
        // Each exact expert is three projections. Account conservatively for
        // the arena's per-projection pointer and block alignment padding.
        constexpr std::uint64_t kMaximumExpertArenaPadding =
            3U * (15U + 255U);
        if (impl_->memory.maximum_expert_bytes >
            std::numeric_limits<std::uint64_t>::max() -
                kMaximumExpertArenaPadding ||
            impl_->memory.maximum_expert_bytes + kMaximumExpertArenaPadding >
                std::numeric_limits<std::uint64_t>::max() / kTopK) {
            result.errors.emplace_back(
                "DeepSeek exact top-k expert lease size overflows");
            return result;
        }
        result = impl_->weights->validate_atomic_expert_capacity(
            (impl_->memory.maximum_expert_bytes + kMaximumExpertArenaPadding) *
            kTopK);
        if (!result.ok()) return result;
    }
    // Rank-local decode is admitted last, after every centralized component
    // has reported its real size. Admission needs measured byte accounts, not
    // estimates, and those only exist once the arena and KV cache are built.
    if (config.decode_topology == Dsv4DecodeTopology::RankLocalTp2) {
        if (config.verbose) {
            std::cerr << "[deepseek-load] phase=rank_local_setup_start\n";
        }
        const auto rank_local_started = std::chrono::steady_clock::now();
        result = impl_->admit_rank_local();
        const auto rank_local_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - rank_local_started).count();
        if (config.verbose) {
            std::cerr << "[deepseek-load] phase=rank_local_setup elapsed_ms="
                      << rank_local_seconds * 1000.0 << '\n';
        }
        if (!result.ok()) return result;
    }
    const auto reset_started = std::chrono::steady_clock::now();
    result = impl_->reset_sequence(1U);
    const auto reset_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - reset_started).count();
    if (config.verbose) {
        std::cerr << "[deepseek-load] phase=sequence_reset elapsed_ms="
                  << reset_seconds * 1000.0 << '\n';
    }
    if (!result.ok()) return result;
    impl_->initialized = true;
    impl_->initialization_metrics.initialization_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      initialization_started).count();
    impl_->initialization_metrics.admission_seconds = admission_seconds;
    impl_->initialization_metrics.resident_staging_seconds = staging_seconds;
    impl_->initialization_metrics.resident_warmup_seconds = warmup_seconds;
    impl_->initialization_metrics.memory = impl_->memory;
    impl_->initialization_metrics.resident_stage = impl_->resident.stats();
    impl_->initialization_metrics.cuda = impl_->cuda.stats();
    impl_->initialization_metrics.cache = impl_->weights->stats();
    if (impl_->rank_local_active && impl_->rank_local_weights != nullptr) {
        const auto rank_weights = impl_->rank_local_weights->device_bytes();
        impl_->initialization_metrics.rank_local_initial_device_vram_bytes =
            impl_->rank_local_initial_device_vram_bytes;
        impl_->initialization_metrics.rank_local_weight_bytes.assign(
            rank_weights.begin(), rank_weights.end());
        impl_->initialization_metrics
            .rank_local_expert_cache_capacity_bytes.assign(
                impl_->rank_local_admission.expert_cache_capacity_bytes.begin(),
                impl_->rank_local_admission.expert_cache_capacity_bytes.end());
        impl_->initialization_metrics.rank_local_admitted_device_bytes.assign(
            impl_->rank_local_admission.device_total_bytes.begin(),
            impl_->rank_local_admission.device_total_bytes.end());
        impl_->initialization_metrics.rank_local_admitted_host_bytes =
            impl_->rank_local_admission.host_total_bytes;
    }
    if (impl_->kv_cache != nullptr) {
        impl_->initialization_metrics.kv_cache = impl_->kv_cache->stats();
    }
    impl_->initialization_metrics.rss_bytes = process_resident_set_bytes();
    impl_->initialization_metrics.device_vram_used_bytes =
        device_vram_used_bytes(impl_->devices);
    impl_->initialization_metrics.detailed_timing = config.detailed_timing;
    impl_->initialization_metrics.dspark_enabled = false;
    impl_->initialization_metrics.device_moe_enabled = config.enable_device_moe;
    impl_->initialization_metrics.host_routed_moe_enabled =
        config.enable_host_routed_moe;
    impl_->initialization_metrics.resident_warmup_overlapped =
        config.overlap_resident_warmup;
    impl_->initialization_metrics.block_kv_cache_enabled =
        config.kv_cache_mode != Dsv4KvCacheMode::ScalarOracle;
    impl_->initialization_metrics.kv_block_rows = config.kv_block_rows;
    impl_->initialization_metrics.host_attention_threads =
        config.host_attention_threads;
    impl_->initialization_metrics.prefill_page_tokens =
        config.prefill_page_tokens;
    impl_->initialization_metrics.prefill_layer_tile_tokens =
        config.prefill_layer_tile_tokens;
    impl_->initialization_metrics.flash_attention_enabled =
        config.enable_flash_attention;
    impl_->initialization_metrics.gpu_lightning_indexer_enabled =
        config.enable_gpu_lightning_indexer;
    impl_->initialization_metrics.flash_attention_minimum_rows =
        config.flash_attention_minimum_rows;
    impl_->initialization_metrics.resident_read_workers =
        impl_->resident.stats().workers;
    impl_->initialization_metrics.resident_pin_seconds = pin_seconds;
    impl_->initialization_metrics.resident_arena_pinned = impl_->resident.pinned();
    impl_->initialization_metrics.spine_warmup_workers =
        static_cast<std::uint32_t>(std::min<std::size_t>(
            config.spine_warmup_workers, impl_->devices.size()));
    impl_->initialization_metrics.expert_prefetch_predictions =
        config.expert_prefetch_predictions;
    impl_->initialization_metrics.expert_prefetch_queue_depth =
        config.expert_prefetch_queue_depth;
    impl_->initialization_metrics.expert_prefetch_byte_budget =
        config.expert_prefetch_byte_budget;
    impl_->initialization_metrics.expert_prefetch_lease_ticks =
        config.expert_prefetch_lease_ticks;
    impl_->initialization_metrics.expert_prefetch_minimum_confidence =
        config.expert_prefetch_minimum_confidence;
    if (config.verbose) {
        const auto& metrics = impl_->initialization_metrics;
        const auto overlapped_seconds = std::max(
            metrics.resident_staging_seconds,
            metrics.resident_warmup_seconds);
        const auto serial_seconds = std::max(
            0.0, metrics.initialization_seconds - overlapped_seconds);
        const auto stage_gib_s = metrics.resident_staging_seconds == 0.0
            ? 0.0
            : static_cast<double>(metrics.resident_stage.bytes) /
                  metrics.resident_staging_seconds / static_cast<double>(1ULL << 30U);
        std::cerr << "[deepseek-load] phase=summary total_ms="
                  << metrics.initialization_seconds * 1000.0
                  << " admission_ms=" << metrics.admission_seconds * 1000.0
                  << " resident_stage_ms="
                  << metrics.resident_staging_seconds * 1000.0
                  << " resident_stage_bytes=" << metrics.resident_stage.bytes
                  << " resident_stage_gib_s=" << stage_gib_s
                  << " spine_warmup_ms="
                  << metrics.resident_warmup_seconds * 1000.0
                  << " weight_upload_bytes=" << metrics.cuda.weight_upload_bytes
                  << " weight_copy_ms="
                  << static_cast<double>(metrics.cuda.weight_copy_nanoseconds) /
                         1.0e6
                  << " weight_allocation_ms="
                  << static_cast<double>(
                         metrics.cuda.weight_allocation_nanoseconds) / 1.0e6
                  << " overlap_window_ms=" << overlapped_seconds * 1000.0
                  << " other_serial_ms=" << serial_seconds * 1000.0
                  << " rss_bytes=" << metrics.rss_bytes << '\n';
    }
    return result;
}

Dsv4GenerationResult DeepSeekV4Runtime::generate_stream(
    std::string_view prompt, std::uint32_t maximum_new_tokens,
    const TokenStreamCallback& on_token) {
    const std::array messages{ChatMessage{ChatRole::User,
                                          std::string(prompt)}};
    return generate_chat_stream(messages, maximum_new_tokens, on_token);
}

Dsv4GenerationResult DeepSeekV4Runtime::generate_chat_stream(
    std::span<const ChatMessage> messages,
    std::uint32_t maximum_new_tokens,
    const TokenStreamCallback& on_token) {
    SamplingOptions sampling;
    sampling.temperature = impl_->config.sampling_temperature;
    sampling.seed = impl_->config.sampling_seed;
    return generate_chat_stream(messages, maximum_new_tokens, sampling, {}, on_token);
}

Dsv4GenerationResult DeepSeekV4Runtime::generate_chat_stream(
    std::span<const ChatMessage> messages,
    std::uint32_t maximum_new_tokens,
    const SamplingOptions& sampling,
    std::span<const std::string> stop,
    const TokenStreamCallback& on_token) {
    Dsv4GenerationResult result;
    if (!impl_->initialized) {
        result.errors.emplace_back("DeepSeek runtime is not initialized");
        return result;
    }
    if (maximum_new_tokens == 0U) {
        result.errors.emplace_back("maximum_new_tokens must be positive");
        return result;
    }
    std::string sampling_error;
    if (!validate_sampling_options(sampling, sampling_error)) {
        result.errors.emplace_back("invalid sampling option: " + sampling_error);
        return result;
    }
    impl_->active_sampling = sampling;
    impl_->sampled_token_counts.assign(kVocabulary, 0U);
    impl_->sampled_token_ids.clear();
    impl_->sampler.seed(sampling.seed);
    std::string validation_error;
    if (!validate_chat_messages(messages, validation_error)) {
        result.errors.push_back(std::move(validation_error));
        return result;
    }
    impl_->weights->finish_prefetch();
    ChatPromptRequest prompt_request;
    prompt_request.messages = messages;
    prompt_request.maximum_new_tokens = maximum_new_tokens;
    prompt_request.maximum_context_tokens = impl_->config.maximum_context_tokens;
    prompt_request.render = [](std::span<const ChatMessage> active) {
        return render_deepseek_v4_chat_prompt(active);
    };
    prompt_request.encode = [&](const std::string& text) {
        return impl_->tokenizer.encode(text);
    };
    auto prompt = prepare_chat_prompt(prompt_request);
    if (!prompt.ok()) {
        result.errors = std::move(prompt.errors);
        return result;
    }
    result.prompt_token_ids = std::move(prompt.token_ids);
    impl_->active_request_id = impl_->generated_requests++;
    impl_->active_prompt_tokens =
        static_cast<std::uint32_t>(result.prompt_token_ids.size());
    impl_->reset_diagnostics();
    const auto active_context_tokens = static_cast<std::uint32_t>(
        result.prompt_token_ids.size() + maximum_new_tokens);
    std::size_t prefill_offset = impl_->config.enable_incremental_kv_continuation &&
        impl_->reusable_sequence
        ? incremental_kv_prefix_tokens(impl_->cached_token_ids,
                                       result.prompt_token_ids)
        : 0U;
    if (prefill_offset != 0U &&
        !std::all_of(impl_->attention_state.begin(),
                     impl_->attention_state.end(),
                     [active_context_tokens](const AttentionState& state) {
                         return state.compressor.ratio != 4U ||
                             active_context_tokens <=
                                 kIndexTopK * state.compressor.ratio ||
                             state.indexer_compressor.ratio ==
                                 state.compressor.ratio;
                     })) {
        prefill_offset = 0U;
    }
    impl_->reusable_sequence = false;
    if (prefill_offset == 0U) {
        auto reset = impl_->reset_sequence(active_context_tokens);
        if (!reset.ok()) {
            result.errors = std::move(reset.errors);
            return result;
        }
    } else {
        // A continuation keeps the sequence, so reset_sequence does not run and
        // the leases the previous generation left open are still held. The
        // prefill below appends the new turn into the last block of that same
        // sequence, and an append refuses to mutate a leased block.
        impl_->release_retained_kv_leases();
    }
    const auto reads_before = impl_->checkpoint->stats();
    const auto cuda_before = impl_->cuda.stats();
    const auto cache_before = impl_->weights->stats();
    const auto kv_cache_before = impl_->kv_cache == nullptr
        ? Dsv4KvCacheStats{} : impl_->kv_cache->stats();
    const auto device_moe_before = impl_->device_moe_stats;
    const auto graph_before = impl_->graph_stats;
    const auto prefill_started = std::chrono::steady_clock::now();
    auto prefill_tokens = std::span<const std::uint32_t>(
        result.prompt_token_ids).subspan(prefill_offset);
    auto next = impl_->forward_prefill(
        prefill_tokens, static_cast<std::uint32_t>(prefill_offset));
    if (!next.ok()) {
        result.errors = std::move(next.errors);
        return result;
    }
    impl_->weights->drain_prefetch();
    // In-chain selection projects the index query on both ranks, so both
    // devices need the index weights; prefill leaves each layer's pair on one
    // device only. Faulting the other copy in on the first decoded token would
    // be a checkpoint read inside the window the zero-NVMe contract covers, so
    // it happens here, before the decode boundary is sampled.
    auto warmed = impl_->rank_local_warm_index_projections();
    if (!warmed.ok()) {
        result.errors = std::move(warmed.errors);
        return result;
    }
    impl_->cached_token_ids = result.prompt_token_ids;
    result.metrics.prefill_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - prefill_started).count();
    result.metrics.prompt_tokens = result.prompt_token_ids.size();
    result.metrics.prefill_tokens = prefill_tokens.size();
    result.metrics.reused_prompt_tokens = prefill_offset;
    result.metrics.incremental_kv_continuation = prefill_offset != 0U;
    const auto reads_after_prefill = impl_->checkpoint->stats();
    const auto cuda_after_prefill = impl_->cuda.stats();
    const auto cache_after_prefill = impl_->weights->stats();
    const auto kv_cache_after_prefill = impl_->kv_cache == nullptr
        ? Dsv4KvCacheStats{} : impl_->kv_cache->stats();
    const auto device_moe_after_prefill = impl_->device_moe_stats;
    const auto graph_after_prefill = impl_->graph_stats;
    constexpr std::uint32_t stop_token = 1U;
    StopSequenceBuffer output(stop);
    if (next.value != stop_token) {
        result.generated_token_ids.push_back(next.value);
        result.logprobs.push_back(impl_->last_sample);
        const auto piece = impl_->tokenizer.decode_token(next.value);
        if (!piece.ok()) {
            result.errors = std::move(piece.errors);
            return result;
        }
        output.append(next.value, piece.value, on_token);
    }
    std::uint32_t position = static_cast<std::uint32_t>(
        result.prompt_token_ids.size());
    std::uint64_t decode_steps = 0U;
    result.metrics.decode_step_seconds.reserve(maximum_new_tokens);
    const auto decode_started = std::chrono::steady_clock::now();
    while (next.value != stop_token && !output.stopped() && !output.cancelled() &&
           result.generated_token_ids.size() < maximum_new_tokens) {
        const auto input_token = next.value;
        const auto step_started = std::chrono::steady_clock::now();
        next = impl_->forward_token(input_token, position++, true);
        result.metrics.decode_step_seconds.push_back(
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - step_started).count());
        if (!next.ok()) {
            result.errors = std::move(next.errors);
            return result;
        }
        impl_->cached_token_ids.push_back(input_token);
        ++decode_steps;
        if (next.value != stop_token) {
            result.generated_token_ids.push_back(next.value);
            result.logprobs.push_back(impl_->last_sample);
            const auto piece = impl_->tokenizer.decode_token(next.value);
            if (!piece.ok()) {
                result.errors = std::move(piece.errors);
                return result;
            }
            output.append(next.value, piece.value, on_token);
        }
    }
    output.finish(on_token);
    result.stopped = output.stopped();
    impl_->weights->finish_prefetch();
    result.metrics.decode_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - decode_started).count();
    result.metrics.decode_tokens = decode_steps;
    result.text = output.text();
    const auto reads_after_decode = impl_->checkpoint->stats();
    const auto cuda_after_decode = impl_->cuda.stats();
    const auto cache_after_decode = impl_->weights->stats();
    const auto kv_cache_after_decode = impl_->kv_cache == nullptr
        ? Dsv4KvCacheStats{} : impl_->kv_cache->stats();
    const auto device_moe_after_decode = impl_->device_moe_stats;
    const auto graph_after_decode = impl_->graph_stats;
    const double prefill_seconds = result.metrics.prefill_seconds;
    const double decode_seconds = result.metrics.decode_seconds;
    auto decode_step_seconds = std::move(result.metrics.decode_step_seconds);
    result.metrics = impl_->initialization_metrics;
    result.metrics.prefill_seconds = prefill_seconds;
    result.metrics.decode_seconds = decode_seconds;
    result.metrics.decode_step_seconds = std::move(decode_step_seconds);
    result.metrics.prompt_tokens = result.prompt_token_ids.size();
    result.metrics.prefill_tokens = prefill_tokens.size();
    result.metrics.reused_prompt_tokens = prefill_offset;
    result.metrics.decode_tokens = decode_steps;
    result.metrics.incremental_kv_continuation = prefill_offset != 0U;
    result.metrics.rss_bytes = process_resident_set_bytes();
    result.metrics.device_vram_used_bytes =
        device_vram_used_bytes(impl_->devices);
    result.metrics.generation_checkpoint_reads = read_delta(reads_after_decode,
                                                             reads_before);
    result.metrics.decode_checkpoint_reads = read_delta(reads_after_decode,
                                                         reads_after_prefill);
    result.metrics.cuda = impl_->cuda.stats();
    result.metrics.cache = impl_->weights->stats();
    result.metrics.kv_cache = kv_cache_after_decode;
    result.metrics.device_moe = device_moe_delta(
        device_moe_after_decode, device_moe_before);
    result.metrics.graph = graph_delta(graph_after_decode, graph_before);
    result.metrics.prefill.checkpoint_reads = read_delta(reads_after_prefill,
                                                         reads_before);
    result.metrics.prefill.cuda = cuda_delta(cuda_after_prefill, cuda_before);
    result.metrics.prefill.cache = cache_delta(cache_after_prefill, cache_before);
    result.metrics.prefill.kv_cache = kv_cache_delta(
        kv_cache_after_prefill, kv_cache_before);
    result.metrics.prefill.device_moe = device_moe_delta(
        device_moe_after_prefill, device_moe_before);
    result.metrics.prefill.graph = graph_delta(graph_after_prefill, graph_before);
    result.metrics.decode.checkpoint_reads = read_delta(reads_after_decode,
                                                        reads_after_prefill);
    result.metrics.decode.cuda = cuda_delta(cuda_after_decode, cuda_after_prefill);
    result.metrics.decode.cache = cache_delta(cache_after_decode, cache_after_prefill);
    result.metrics.decode.kv_cache = kv_cache_delta(
        kv_cache_after_decode, kv_cache_after_prefill);
    result.metrics.decode.device_moe = device_moe_delta(
        device_moe_after_decode, device_moe_after_prefill);
    result.metrics.decode.graph = graph_delta(
        graph_after_decode, graph_after_prefill);
    result.diagnostics = std::move(impl_->diagnostics);
    if (result.metrics.cache.lease_acquires !=
            result.metrics.cache.lease_releases ||
        result.metrics.cache.prefetch_lease_acquires !=
            result.metrics.cache.prefetch_lease_releases ||
        result.metrics.cache.active_prefetch_leases != 0U ||
        std::any_of(result.metrics.cache.active_leases.begin(),
                    result.metrics.cache.active_leases.end(),
                    [](std::uint64_t count) { return count != 0U; })) {
        result.errors.emplace_back(
            "DeepSeek generation completed with outstanding CUDA weight leases");
    }
    if (impl_->config.require_zero_nvme_decode &&
        (result.metrics.decode_checkpoint_reads.calls != 0U ||
         result.metrics.decode_checkpoint_reads.bytes != 0U)) {
        result.errors.emplace_back(
            "DeepSeek zero-NVMe decode contract was violated by checkpoint reads");
    }
    if (impl_->route_trace.is_open()) {
        auto flushed = impl_->route_trace.flush();
        result.errors.insert(result.errors.end(),
                             std::make_move_iterator(flushed.errors.begin()),
                             std::make_move_iterator(flushed.errors.end()));
    }
    impl_->reusable_sequence =
        impl_->config.enable_incremental_kv_continuation && result.ok();
    return result;
}

Dsv4GenerationResult DeepSeekV4Runtime::generate(
    std::string_view prompt, std::uint32_t maximum_new_tokens) {
    return generate_stream(prompt, maximum_new_tokens, {});
}

const Dsv4MemoryPlan& DeepSeekV4Runtime::memory_plan() const noexcept {
    return impl_->memory;
}

}  // namespace strata
