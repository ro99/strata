#include "strata/dsv4_rank_local_layer_executor.hpp"

#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace strata {
namespace {

constexpr std::size_t kWorld = 2U;
constexpr std::size_t kChainSlots = 43U;
static_assert(kChainSlots == 43U,
              "queued rank-local callback capacity is fixed at 43 layers");
constexpr std::size_t kHidden = 4096U;
constexpr std::size_t kQueryRank = 1024U;
constexpr std::size_t kKeyValue = 512U;
constexpr std::size_t kMaximumCompressorElements = 1024U;
constexpr std::size_t kMaximumIndexCompressorElements = 256U;
constexpr std::size_t kReplicaPagePatchSlotBytes = 2048U;
constexpr std::size_t kMhc = 4U * kHidden;
constexpr std::size_t kRouter = 256U;
constexpr std::size_t kVocabulary = kDsv4RankLocalVocabulary;
constexpr std::size_t kVocabularyShard = kDsv4RankLocalVocabularyShard;
constexpr std::size_t kTopK =
    kDeepSeekV4ExecutionContract.experts_per_token;
static_assert(kTopK == kDsv4RankLocalTopK,
              "rank-local observations must retain the declared top-k");
constexpr float kAttentionScale = 0.04419417382415922F;
constexpr float kSwiGluLimit = 10.0F;

__device__ std::uint16_t float_to_bf16(float value) {
    const auto bits = __float_as_uint(value);
    const auto rounding = 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

__global__ void publish_head_shard(const float* input, std::uint16_t* output) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < kVocabularyShard) output[index] = float_to_bf16(input[index]);
}

[[nodiscard]] bool cuda_ok(cudaError_t status, const char* operation,
                           std::vector<std::string>& errors) {
    if (status == cudaSuccess) return true;
    errors.emplace_back(std::string(operation) + ": " +
                        cudaGetErrorString(status));
    return false;
}

[[nodiscard]] bool nccl_ok(ncclResult_t status, const char* operation,
                           std::vector<std::string>& errors) {
    if (status == ncclSuccess) return true;
    errors.emplace_back(std::string(operation) + ": " +
                        ncclGetErrorString(status));
    return false;
}

[[nodiscard]] int failure_rank(Dsv4RankLocalFailure failure) noexcept {
    switch (failure) {
    case Dsv4RankLocalFailure::AttentionPreRank0:
    case Dsv4RankLocalFailure::AttentionPostRank0:
    case Dsv4RankLocalFailure::MoePreRank0:
    case Dsv4RankLocalFailure::MoePostRank0:
        return 0;
    case Dsv4RankLocalFailure::AttentionPreRank1:
    case Dsv4RankLocalFailure::AttentionPostRank1:
    case Dsv4RankLocalFailure::MoePreRank1:
    case Dsv4RankLocalFailure::MoePostRank1:
    case Dsv4RankLocalFailure::MoeBeforeEnqueueRank1:
        return 1;
    case Dsv4RankLocalFailure::None:
        return -1;
    }
    return -1;
}

[[nodiscard]] bool is_attention_failure(Dsv4RankLocalFailure failure) noexcept {
    return failure == Dsv4RankLocalFailure::AttentionPreRank0 ||
           failure == Dsv4RankLocalFailure::AttentionPreRank1 ||
           failure == Dsv4RankLocalFailure::AttentionPostRank0 ||
           failure == Dsv4RankLocalFailure::AttentionPostRank1;
}

[[nodiscard]] bool is_moe_failure(Dsv4RankLocalFailure failure) noexcept {
    return failure == Dsv4RankLocalFailure::MoePreRank0 ||
           failure == Dsv4RankLocalFailure::MoePreRank1 ||
           failure == Dsv4RankLocalFailure::MoePostRank0 ||
           failure == Dsv4RankLocalFailure::MoePostRank1;
}

[[nodiscard]] bool is_pre_failure(Dsv4RankLocalFailure failure) noexcept {
    return failure == Dsv4RankLocalFailure::AttentionPreRank0 ||
           failure == Dsv4RankLocalFailure::AttentionPreRank1 ||
           failure == Dsv4RankLocalFailure::MoePreRank0 ||
           failure == Dsv4RankLocalFailure::MoePreRank1;
}

[[nodiscard]] float bf16_to_float(std::uint16_t value) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

[[nodiscard]] std::uint16_t host_float_to_bf16(float value) noexcept {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] std::uint64_t hash_bf16(
    std::span<const std::uint16_t> values) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto value : values) {
        for (std::size_t byte = 0U; byte < sizeof(value); ++byte) {
            hash ^= static_cast<std::uint8_t>(value >> (8U * byte));
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

[[nodiscard]] bool valid_head_request(
    const Dsv4RankLocalHeadRequest& request,
    const std::array<int, kWorld>& devices) noexcept {
    for (std::size_t rank = 0U; rank < kWorld; ++rank) {
        if (request.heads[rank] == nullptr ||
            !request.heads[rank]->valid() ||
            request.heads[rank]->device() != devices[rank] ||
            request.callbacks[rank] == nullptr ||
            request.callback_contexts[rank] == nullptr ||
            request.local_logits[rank].size() != kVocabularyShard ||
            request.published_logits[rank].size() != kVocabulary) return false;
    }
    return true;
}

struct PhaseEvents {
    cudaEvent_t start{};
    cudaEvent_t attention{};
    cudaEvent_t attention_collective{};
    cudaEvent_t attention_publication{};
    cudaEvent_t router{};
    cudaEvent_t moe{};
    cudaEvent_t moe_collective{};
    cudaEvent_t moe_publication{};
    cudaEvent_t final{};
};

void destroy_events(PhaseEvents& events) noexcept {
    for (auto* event : {events.start, events.attention,
                        events.attention_collective,
                        events.attention_publication, events.router, events.moe,
                        events.moe_collective, events.moe_publication,
                        events.final}) {
        if (event != nullptr) static_cast<void>(cudaEventDestroy(event));
    }
    events = {};
}

[[nodiscard]] bool create_events(PhaseEvents& events,
                                 std::vector<std::string>& errors) {
    for (auto** event : {&events.start, &events.attention,
                         &events.attention_collective,
                         &events.attention_publication, &events.router,
                         &events.moe, &events.moe_collective,
                         &events.moe_publication, &events.final}) {
        if (!cuda_ok(cudaEventCreate(event), "create layer phase event", errors)) {
            destroy_events(events);
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool elapsed(cudaEvent_t begin, cudaEvent_t end, double& output,
                           std::vector<std::string>& errors) {
    float milliseconds = 0.0F;
    if (!cuda_ok(cudaEventElapsedTime(&milliseconds, begin, end),
                 "measure layer phase event", errors)) return false;
    output = static_cast<double>(milliseconds);
    return true;
}

}  // namespace

struct Dsv4RankLocalLayerExecutor::Impl {
    struct CallbackSlot {
        bool active{};
        std::uint32_t layer{};
        std::span<const float> router_bias;
        std::span<const std::uint32_t> router_token_experts;
        Dsv4RankLocalLayerResult* result{};
        Dsv4RankLocalLayerObservation* observation{};
        Dsv4RankLocalFailure failure{Dsv4RankLocalFailure::None};
    };

    struct CallbackContext {
        Impl* owner{};
        std::size_t rank{};
        std::size_t slot{};
    };

    struct RankState {
        std::unique_ptr<Dsv4HostMoeExecutor> cpu;
        ncclComm_t communicator{};
        unsigned int* moe_status{};
        PhaseEvents events;
        float* attention_reduction{};
        float* attention_snapshot{};
        std::uint16_t* attention_publication{};
        std::array<float, kQueryRank> prepared_query{};
        std::array<float, kKeyValue> prepared_key_value{};
        std::array<float, kMaximumCompressorElements>
            prepared_compressor_values{};
        std::array<float, kMaximumCompressorElements>
            prepared_compressor_scores{};
        std::array<float, kMaximumIndexCompressorElements>
            prepared_index_compressor_values{};
        std::array<float, kMaximumIndexCompressorElements>
            prepared_index_compressor_scores{};
        Dsv4HostMoePhaseTimings chain_cpu_moe_phases{};
        float* moe_snapshot{};
        std::uint16_t* moe_publication{};
        std::uint16_t* head_send{};
        std::uint16_t* head_full{};
        std::vector<float> route_scratch;
        Dsv4Route route;
        // Left by the route half for the host share that follows it. Per rank
        // rather than per slot: both halves of a slot are host nodes on the one
        // stream, so they run back to back and the next slot cannot overtake.
        std::array<float, kHidden> route_input{};
        std::array<bool, kTopK> route_skip{};
        bool route_valid{};
        bool route_any_skipped{};
        std::array<CallbackSlot, kChainSlots> callback_slots{};
        std::array<CallbackContext, kChainSlots> callback_contexts{};
    };

    CudaBackend* backend{};
    Dsv4RankLocalLayerOptions options;
    // Set once at initialize from whether any tier is active. Splitting the
    // boundary costs one extra host node per layer, which is only worth paying
    // when there is a tier to release early.
    bool split_route_callback{};
    std::array<RankState, kWorld> ranks;
    bool execution_active{};
    bool chain_active{};
    bool chain_mode{};
    std::size_t chain_count{};
    bool terminal_active{};
    cudaEvent_t rank0_page_patch_ready{};
    std::byte* replica_page_patch_staging{};
    RouterSpec router_spec;
    bool initialized{};
};

namespace {

struct ActiveCallGuard {
    Dsv4RankLocalLayerExecutor::Impl& owner;
    ~ActiveCallGuard() { owner.execution_active = false; }
};

[[nodiscard]] bool valid_rank_weights(
    const Dsv4RankLocalLayerRankWeights& weights, int device,
    bool require_next_attention_mhc) noexcept {
    const auto valid_weight = [device](const CudaWeight* weight) {
        return weight != nullptr && weight->valid() && weight->device() == device;
    };
    const auto valid_mhc = [device](const CudaDsv4MhcWeights* weight) {
        return weight != nullptr && weight->valid() && weight->device() == device;
    };
    const auto valid_expert = [device, &valid_weight](
                                  const CudaDeepSeekMoeExpert* expert) {
        return expert != nullptr && valid_weight(expert->w1) &&
               valid_weight(expert->w3) && valid_weight(expert->w2) &&
               expert->coefficient == 1.0F;
    };
    const bool next_mhc_valid = require_next_attention_mhc
        ? valid_mhc(weights.next_attention_mhc)
        : weights.next_attention_mhc == nullptr;
    const auto valid_optional_projection = [&](const CudaWeight* value,
                                               const CudaWeight* gate,
                                               std::uint32_t elements,
                                               std::size_t maximum) {
        return elements == 0U
            ? value == nullptr && gate == nullptr
            : elements <= maximum && valid_weight(value) && valid_weight(gate);
    };
    return valid_weight(weights.query_a) && valid_weight(weights.query_b) &&
           valid_weight(weights.key_value) && valid_weight(weights.output_a) &&
           valid_weight(weights.output_b) && valid_weight(weights.router) &&
           valid_expert(weights.shared) && valid_mhc(weights.attention_mhc) &&
           valid_mhc(weights.ffn_mhc) && next_mhc_valid &&
           valid_optional_projection(
               weights.compressor_value, weights.compressor_gate,
               weights.compressor_elements, kMaximumCompressorElements) &&
           valid_optional_projection(
               weights.index_compressor_value,
               weights.index_compressor_gate,
               weights.index_compressor_elements,
               kMaximumIndexCompressorElements) &&
           weights.query_norm.size() == 1024U &&
           weights.key_value_norm.size() == 512U &&
           std::all_of(weights.query_norm.begin(), weights.query_norm.end(),
                       [](float value) { return std::isfinite(value); }) &&
           std::all_of(weights.key_value_norm.begin(), weights.key_value_norm.end(),
                       [](float value) { return std::isfinite(value); });
}

[[nodiscard]] bool valid_call(const Dsv4RankLocalLayerCall& call,
                              const std::array<int, kWorld>& devices) noexcept {
    if (call.layer >= kDeepSeekV4ExecutionContract.layer_count ||
        (call.terminal && call.layer !=
             kDeepSeekV4ExecutionContract.layer_count - 1U) ||
        (!call.terminal && call.layer ==
             kDeepSeekV4ExecutionContract.layer_count - 1U) ||
        call.head_sinks.size() != 64U || call.candidates.empty() ||
        call.candidates.size() > 640U || call.candidates.size() % 64U != 0U ||
        call.inverse_rope_cosines.size() != 32U ||
        call.inverse_rope_sines.size() != 32U ||
        ((call.layer < 3U &&
          (call.weights.router_token_experts.size() != kTopK ||
           !call.weights.router_bias.empty())) ||
         (call.layer >= 3U &&
          (call.weights.router_bias.size() != kRouter ||
           !call.weights.router_token_experts.empty()))) ||
        !std::all_of(call.head_sinks.begin(), call.head_sinks.end(),
                     [](float value) { return std::isfinite(value); }) ||
        !std::all_of(call.inverse_rope_cosines.begin(), call.inverse_rope_cosines.end(),
                     [](float value) { return std::isfinite(value); }) ||
        !std::all_of(call.inverse_rope_sines.begin(), call.inverse_rope_sines.end(),
                     [](float value) { return std::isfinite(value); }) ||
        !std::all_of(call.weights.router_bias.begin(), call.weights.router_bias.end(),
                     [](float value) { return std::isfinite(value); })) return false;
    if (std::any_of(call.weights.router_token_experts.begin(),
                    call.weights.router_token_experts.end(),
                    [](std::uint32_t expert) { return expert >= kRouter; })) {
        return false;
    }
    bool patches_enabled = false;
    for (std::size_t rank = 0U; rank < kWorld; ++rank) {
        if (!valid_rank_weights(call.weights.rank[rank], devices[rank],
                                !call.terminal) ||
            call.pages[rank].empty()) return false;
        for (const auto& page : call.pages[rank]) {
            if (page.buffer == nullptr || !page.buffer->valid() ||
                page.buffer->device() != devices[rank] ||
                (page.rows != 2U && page.rows != 64U && page.rows != 256U) ||
                page.buffer->device_bytes() != static_cast<std::uint64_t>(page.rows) * 584U)
                return false;
        }
        const auto& patch = call.page_patches[rank];
        const bool enabled = patch.callback != nullptr ||
                              patch.context != nullptr ||
                              !patch.ready_patch.empty() ||
                              !patch.writes.empty();
        if (rank == 0U) patches_enabled = enabled;
        const bool host_patch = patch.callback != nullptr &&
            patch.context != nullptr && patch.ready_patch.empty() &&
            !patch.writes.empty();
        const bool replica_patch = call.ordered_page_patches && rank == 1U &&
            patch.callback == nullptr && patch.context == nullptr &&
            !patch.ready_patch.empty() && !patch.writes.empty();
        if (enabled != patches_enabled ||
            (enabled && !(host_patch || replica_patch)) ||
            (!enabled && (patch.callback != nullptr || patch.context != nullptr ||
                          !patch.ready_patch.empty() ||
                          !patch.writes.empty()))) return false;
        if (enabled) {
            for (const auto& write : patch.writes) {
                if (write.buffer == nullptr || !write.buffer->valid() ||
                    write.buffer->device() != devices[rank] ||
                    write.data_bytes == 0U || write.scale_bytes == 0U ||
                    write.data_offset > write.buffer->device_bytes() ||
                    write.data_bytes >
                        write.buffer->device_bytes() - write.data_offset ||
                    write.scale_offset > write.buffer->device_bytes() ||
                    write.scale_bytes >
                        write.buffer->device_bytes() - write.scale_offset) {
                    return false;
                }
            }
        }
    }
    if (call.ordered_page_patches && !patches_enabled) return false;
    const auto& selection = call.selection;
    if (selection.active) {
        if (selection.blocks.empty() || selection.heads == 0U ||
            selection.head_dim == 0U || selection.rope_dim == 0U ||
            selection.top_k == 0U ||
            selection.compressed_width < selection.top_k ||
            selection.compressed_width > call.candidates.size() ||
            selection.rope_cosines.size() != selection.rope_dim / 2U ||
            selection.rope_sines.size() != selection.rope_dim / 2U ||
            !std::isfinite(selection.weight_scale) ||
            !std::all_of(selection.rope_cosines.begin(),
                         selection.rope_cosines.end(),
                         [](float value) { return std::isfinite(value); }) ||
            !std::all_of(selection.rope_sines.begin(),
                         selection.rope_sines.end(),
                         [](float value) { return std::isfinite(value); })) {
            return false;
        }
        for (std::size_t rank = 0U; rank < kWorld; ++rank) {
            if (selection.query_projection[rank] == nullptr ||
                selection.weight_projection[rank] == nullptr ||
                !selection.query_projection[rank]->valid() ||
                !selection.weight_projection[rank]->valid() ||
                selection.query_projection[rank]->device() != devices[rank] ||
                selection.weight_projection[rank]->device() != devices[rank] ||
                selection.index_pages[rank].empty() ||
                // Every attendable compressed block must own a live attention
                // page: a selection the host never sees has no first touch to
                // lease on.
                selection.blocks.size() > call.pages[rank].size()) {
                return false;
            }
            for (const auto& page : selection.index_pages[rank]) {
                if (page.buffer == nullptr || !page.buffer->valid() ||
                    page.buffer->device() != devices[rank] ||
                    page.rows == 0U || page.rows > page.block_rows) {
                    return false;
                }
            }
        }
    }
    const auto host_candidate_begin = selection.active
        ? selection.compressed_width : 0U;
    for (std::size_t rank = 0U; rank < kWorld; ++rank) {
        for (std::size_t index = host_candidate_begin;
             index < call.candidates.size(); ++index) {
            const auto& candidate = call.candidates[index];
            if (candidate.valid &&
                (candidate.page >= call.pages[rank].size() ||
                 candidate.row >= call.pages[rank][candidate.page].rows)) return false;
        }
    }
    return true;
}

// True when this executor has any routed-expert tier to serve. It decides
// whether the CPU-MoE boundary is split into a route half and a host-share
// half: with no tier there is nothing to overlap, so the original single
// callback is kept and the no-tier decode path stays byte for byte what it was.
[[nodiscard]] bool any_tier_active(
    const Dsv4RankLocalLayerExecutor::Impl& owner) noexcept {
    for (const auto* candidate : owner.options.static_expert_tiers) {
        if (candidate != nullptr && candidate->active()) return true;
    }
    return false;
}

// Decides the route and publishes the tier selection, leaving both in the
// rank's scratch for the host share to consume. Shared by the split and
// unsplit entry points so that there is exactly one router: two would be two
// things to keep bit-identical, and the charter does not allow router
// semantics to drift silently.
[[nodiscard]] bool prepare_route(
    Dsv4RankLocalLayerExecutor::Impl::CallbackContext& context,
    std::span<const std::uint16_t> encoded,
    std::span<const float> router_logits) {
    auto& owner = *context.owner;
    const auto rank = context.rank;
    if (rank >= kWorld || context.slot >= kChainSlots ||
        owner.options.resident == nullptr ||
        encoded.size() != kHidden || router_logits.size() != kRouter) {
        return false;
    }
    auto& state = owner.ranks[rank];
    state.route_valid = false;
    state.route_any_skipped = false;
    auto& slot = state.callback_slots[context.slot];
    if (!slot.active) return false;
    auto* observation = slot.observation;
    Dsv4RankLocalLayerRankObservation* rank_observation = nullptr;
    if (observation != nullptr) {
        rank_observation = &observation->rank[rank];
        rank_observation->invoked = true;
        rank_observation->completed = false;
        rank_observation->success = false;
        rank_observation->failure = false;
        rank_observation->encoded_ffn_input.fill(0U);
        rank_observation->router_logits.fill(0.0F);
        rank_observation->routed_experts.fill(0U);
        rank_observation->routed_coefficients.fill(0.0F);
        rank_observation->cpu_rank_partial.fill(0.0F);
        rank_observation->cpu_moe_phases = {};
    }
    const auto observation_failure = [&]() noexcept {
        if (rank_observation != nullptr) {
            rank_observation->failure = true;
            rank_observation->completed = true;
        }
    };
    const auto pre_failure = is_pre_failure(slot.failure) &&
        failure_rank(slot.failure) == static_cast<int>(rank) &&
        is_moe_failure(slot.failure);
    if (pre_failure) {
        observation_failure();
        return false;
    }

    // This callback is intentionally allocation-free. The fixed stack decode
    // is the production BF16 callback boundary, and all executor scratch was
    // reserved by initialize().
    auto& input = state.route_input;
    for (std::size_t index = 0U; index < kHidden; ++index) {
        const auto bits = static_cast<std::uint32_t>(encoded[index]) << 16U;
        input[index] = std::bit_cast<float>(bits);
        if (rank_observation != nullptr)
            rank_observation->encoded_ffn_input[index] = encoded[index];
        if (!std::isfinite(input[index])) {
            observation_failure();
            return false;
        }
    }
    for (std::size_t index = 0U; index < router_logits.size(); ++index) {
        if (rank_observation != nullptr)
            rank_observation->router_logits[index] = router_logits[index];
        if (!std::isfinite(router_logits[index])) {
            observation_failure();
            return false;
        }
    }
    // Same-rank callbacks are enqueued on one CUDA stream. Host nodes execute
    // in stream order, so this fixed route scratch may be reused by the next
    // slot only after the preceding node has completed. The split keeps that
    // property: both halves of a slot are host nodes on the same stream, so
    // they run back to back before the next slot's route half.
    const auto routed = slot.router_token_experts.empty()
        ? dsv4_route_sqrtsoftplus_f32_into(
              router_logits, slot.router_bias, owner.router_spec,
              state.route_scratch, state.route.experts, state.route.weights)
        : dsv4_route_hash_sqrtsoftplus_f32_into(
              router_logits, slot.router_token_experts, owner.router_spec,
              state.route.experts, state.route.weights);
    if (!routed.ok()) {
        observation_failure();
        return false;
    }
    if (rank_observation != nullptr) {
        std::copy(state.route.experts.begin(), state.route.experts.end(),
                  rank_observation->routed_experts.begin());
        std::copy(state.route.weights.begin(), state.route.weights.end(),
                  rank_observation->routed_coefficients.begin());
    }
    if (slot.result != nullptr) {
        std::copy(state.route.experts.begin(), state.route.experts.end(),
                  slot.result->routed_experts[rank].begin());
        std::copy(state.route.weights.begin(), state.route.weights.end(),
                  slot.result->routed_coefficients[rank].begin());
    }

    // Split the route between the device tiers and this rank's host pool.
    //
    // Writing the selection is a plain store into pinned memory, which is legal
    // from a host function where a CUDA call is not. The tier kernels observe
    // it either behind this node on the same stream, or -- when the boundary is
    // split -- on their own stream released by an event recorded immediately
    // after this node returns, so they run while the host share below is still
    // computing rather than after it.
    //
    // A tier computes the *whole* expert, not this rank's shard of it, so the
    // skip mask is the union over every tier while the selection this rank
    // writes names only what its own device holds. Skipping only the local
    // tier would leave the other rank's shard of a remotely-held expert
    // computed twice.
    auto& skip = state.route_skip;
    skip.fill(false);
    std::size_t served = 0U;
    for (std::size_t index = 0U; index < kTopK; ++index) {
        const auto expert = state.route.experts[index];
        for (const auto* candidate : owner.options.static_expert_tiers) {
            if (candidate != nullptr && candidate->active() &&
                candidate->resident(slot.layer, expert)) {
                skip[index] = true;
                break;
            }
        }
    }
    auto* tier = owner.options.static_expert_tiers[rank];
    if (tier != nullptr && tier->active()) {
        auto* selection = owner.backend->dsv4_tier_selection(tier->device());
        if (selection != nullptr) {
            selection->layer = slot.layer;
            for (std::size_t index = 0U; index < kTopK; ++index) {
                const auto expert = state.route.experts[index];
                const bool mine = tier->resident(slot.layer, expert);
                selection->experts[index] = mine ? expert : kDsv4TierAbsent;
                selection->coefficients[index] =
                    mine ? state.route.weights[index] : 0.0F;
                if (mine) ++served;
            }
            selection->count = static_cast<std::uint32_t>(served);
        }
    }
    state.route_any_skipped =
        std::any_of(skip.begin(), skip.end(), [](bool value) { return value; });
    state.route_valid = true;
    return true;
}

// First host node of the split boundary. Nothing but the route belongs here.
bool route_callback(void* opaque, std::span<const std::uint16_t> encoded,
                    std::span<const float> router_logits) {
    auto& context = *static_cast<Dsv4RankLocalLayerExecutor::Impl::CallbackContext*>(
        opaque);
    return prepare_route(context, encoded, router_logits);
}

bool host_moe_callback(void* opaque, std::span<const std::uint16_t> encoded,
                       std::span<const float> router_logits,
                       std::span<float> rank_partials) {
    auto& context = *static_cast<Dsv4RankLocalLayerExecutor::Impl::CallbackContext*>(
        opaque);
    auto& owner = *context.owner;
    const auto rank = context.rank;
    if (rank >= kWorld || context.slot >= kChainSlots ||
        owner.options.resident == nullptr ||
        encoded.size() != kHidden || router_logits.size() != kRouter ||
        rank_partials.size() != 2U * kHidden) {
        std::fill(rank_partials.begin(), rank_partials.end(), 0.0F);
        return false;
    }
    auto& state = owner.ranks[rank];
    auto& slot = state.callback_slots[context.slot];
    if (!slot.active) {
        std::fill(rank_partials.begin(), rank_partials.end(), 0.0F);
        return false;
    }
    auto* observation = slot.observation;
    Dsv4RankLocalLayerRankObservation* rank_observation =
        observation == nullptr ? nullptr : &observation->rank[rank];
    const auto observation_failure = [&]() noexcept {
        if (rank_observation != nullptr) {
            rank_observation->failure = true;
            rank_observation->completed = true;
        }
    };
    std::fill(rank_partials.begin(), rank_partials.end(), 0.0F);

    // With a tier the route was decided by the node before this one, so that
    // the tier could start against it. Without one there is no such node and
    // the route is decided here, exactly as it always was.
    if (owner.split_route_callback) {
        if (!state.route_valid) {
            observation_failure();
            return false;
        }
    } else if (!prepare_route(context, encoded, router_logits)) {
        return false;
    }

    auto destination = rank_partials.subspan(rank * kHidden, kHidden);
    Dsv4HostMoePhaseTimings callback_phases{};
    const auto status = state.cpu->run(
        slot.layer, state.route, state.route_input,
        *owner.options.resident,
        destination, &callback_phases,
        state.route_any_skipped ? std::span<const bool>(state.route_skip)
                                : std::span<const bool>{});
    state.chain_cpu_moe_phases.gate_up_nanoseconds +=
        callback_phases.gate_up_nanoseconds;
    state.chain_cpu_moe_phases.down_nanoseconds +=
        callback_phases.down_nanoseconds;
    state.chain_cpu_moe_phases.reduce_nanoseconds +=
        callback_phases.reduce_nanoseconds;
    state.chain_cpu_moe_phases.total_nanoseconds +=
        callback_phases.total_nanoseconds;
    if (slot.result != nullptr)
        slot.result->cpu_moe_phases[rank] = callback_phases;
    if (slot.result != nullptr && status.ok()) {
        std::copy(destination.begin(), destination.end(),
                  slot.result->cpu_rank_partials[rank].begin());
    }
    if (rank_observation != nullptr) {
        std::copy(destination.begin(), destination.end(),
                  rank_observation->cpu_rank_partial.begin());
        rank_observation->cpu_moe_phases = callback_phases;
        rank_observation->success = status.ok();
        rank_observation->failure = !status.ok();
        rank_observation->completed = true;
    }
    return status.ok();
}

[[nodiscard]] bool collective_data(
    Dsv4RankLocalLayerExecutor::Impl& owner,
    const std::array<CudaDsv4HostMoeDeviceView, kWorld>& views,
    std::vector<std::string>& errors) {
    if (!nccl_ok(ncclGroupStart(), "start FP32 data collective", errors)) return false;
    std::array<ncclResult_t, kWorld> statuses{};
    for (std::size_t rank = 0U; rank < kWorld; ++rank) {
        if (!cuda_ok(cudaSetDevice(owner.options.devices[rank]),
                     "select data collective device", errors)) continue;
        statuses[rank] = ncclAllReduce(
            views[rank].output, views[rank].output, kHidden, ncclFloat,
            ncclSum, owner.ranks[rank].communicator,
            static_cast<cudaStream_t>(views[rank].stream));
    }
    const auto group_status = ncclGroupEnd();
    bool valid = nccl_ok(group_status, "finish FP32 data collective", errors);
    for (const auto status : statuses) {
        valid = nccl_ok(status, "FP32 data all-reduce", errors) && valid;
    }
    return valid;
}

[[nodiscard]] bool collective_status(
    Dsv4RankLocalLayerExecutor::Impl& owner,
    const std::array<unsigned int*, kWorld>& statuses,
    const std::array<cudaStream_t, kWorld>& streams,
    std::vector<std::string>& errors) {
    if (!nccl_ok(ncclGroupStart(), "start U32 status collective", errors)) return false;
    std::array<ncclResult_t, kWorld> results{};
    for (std::size_t rank = 0U; rank < kWorld; ++rank) {
        if (!cuda_ok(cudaSetDevice(owner.options.devices[rank]),
                     "select status collective device", errors)) continue;
        results[rank] = ncclAllReduce(
            statuses[rank], statuses[rank], 1U, ncclUint32, ncclMax,
            owner.ranks[rank].communicator, streams[rank]);
    }
    const auto group_status = ncclGroupEnd();
    bool valid = nccl_ok(group_status, "finish U32 status collective", errors);
    for (const auto status : results) {
        valid = nccl_ok(status, "U32 status all-reduce", errors) && valid;
    }
    return valid;
}

[[nodiscard]] bool copy_device_to_device(
    int device, void* destination, const void* source, std::size_t bytes,
    cudaStream_t stream, const char* operation, std::vector<std::string>& errors) {
    if (!cuda_ok(cudaSetDevice(device), "select snapshot device", errors)) return false;
    return cuda_ok(cudaMemcpyAsync(destination, source, bytes,
                                   cudaMemcpyDeviceToDevice, stream),
                   operation, errors);
}

[[nodiscard]] bool zero_device(
    int device, void* destination, std::size_t bytes, cudaStream_t stream,
    const char* operation, std::vector<std::string>& errors) {
    if (!cuda_ok(cudaSetDevice(device), "select zero device", errors)) return false;
    return cuda_ok(cudaMemsetAsync(destination, 0, bytes, stream), operation,
                   errors);
}

void clear_failed_payload(Dsv4RankLocalLayerResult& output) noexcept {
    const auto total_ms = output.timing.total_ms;
    for (std::size_t rank = 0U; rank < kWorld; ++rank) {
        output.attention_reduction[rank].fill(0.0F);
        output.moe_reduction[rank].fill(0.0F);
        output.cpu_rank_partials[rank].fill(0.0F);
        output.attention_publication[rank].fill(0U);
        output.moe_publication[rank].fill(0U);
        output.next_attention_input[rank].fill(0.0F);
        output.outgoing_residual[rank].fill(0U);
        output.routed_experts[rank].fill(0U);
        output.routed_coefficients[rank].fill(0.0F);
        output.cpu_moe_phases[rank] = {};
    }
    output.timing = {};
    output.timing.total_ms = total_ms;
}

void clear_callback_slots(Dsv4RankLocalLayerExecutor::Impl& impl) noexcept {
    for (auto& rank : impl.ranks) {
        for (auto& slot : rank.callback_slots) slot = {};
    }
    impl.chain_active = false;
    impl.chain_count = 0U;
    impl.terminal_active = false;
}

}  // namespace

Dsv4RankLocalLayerExecutor::Dsv4RankLocalLayerExecutor(CudaBackend& backend)
    : backend_(backend), impl_(std::make_unique<Impl>()) {
    impl_->backend = &backend;
}

Dsv4RankLocalLayerExecutor::~Dsv4RankLocalLayerExecutor() {
    if (impl_ == nullptr) return;
    if (impl_->rank0_page_patch_ready != nullptr) {
        static_cast<void>(cudaSetDevice(impl_->options.devices[0]));
        static_cast<void>(cudaEventDestroy(impl_->rank0_page_patch_ready));
        impl_->rank0_page_patch_ready = nullptr;
    }
    if (impl_->replica_page_patch_staging != nullptr) {
        static_cast<void>(cudaFreeHost(impl_->replica_page_patch_staging));
        impl_->replica_page_patch_staging = nullptr;
    }
    for (auto& rank : impl_->ranks) {
        if (rank.communicator != nullptr) {
            static_cast<void>(ncclCommDestroy(rank.communicator));
            rank.communicator = nullptr;
        }
        destroy_events(rank.events);
        if (rank.attention_reduction != nullptr ||
            rank.attention_snapshot != nullptr ||
            rank.attention_publication != nullptr || rank.moe_snapshot != nullptr ||
            rank.moe_publication != nullptr || rank.head_send != nullptr ||
            rank.head_full != nullptr) {
            static_cast<void>(cudaSetDevice(
                impl_->options.devices[&rank - impl_->ranks.data()]));
            if (rank.attention_reduction != nullptr)
                static_cast<void>(cudaFree(rank.attention_reduction));
            if (rank.attention_snapshot != nullptr)
                static_cast<void>(cudaFree(rank.attention_snapshot));
            if (rank.attention_publication != nullptr)
                static_cast<void>(cudaFree(rank.attention_publication));
            if (rank.moe_snapshot != nullptr)
                static_cast<void>(cudaFree(rank.moe_snapshot));
            if (rank.moe_publication != nullptr)
                static_cast<void>(cudaFree(rank.moe_publication));
            if (rank.head_send != nullptr)
                static_cast<void>(cudaFree(rank.head_send));
            if (rank.head_full != nullptr)
                static_cast<void>(cudaFree(rank.head_full));
        }
    }
}

ValidationResult Dsv4RankLocalLayerExecutor::initialize(
    const Dsv4RankLocalLayerOptions& options) {
    ValidationResult result;
    if (impl_->initialized || options.devices[0] == options.devices[1] ||
        options.resident == nullptr) {
        result.errors.emplace_back("rank-local layer executor options are invalid");
        return result;
    }
    impl_->options = options;
    impl_->split_route_callback = any_tier_active(*impl_);
    impl_->router_spec = deepseek_v4_flash_0731_spec().router;
    devices_ = options.devices;
    for (std::size_t rank = 0U; rank < kWorld; ++rank) {
        for (std::size_t slot = 0U; slot < kChainSlots; ++slot) {
            impl_->ranks[rank].callback_contexts[slot] =
                {impl_.get(), rank, slot};
        }
        impl_->ranks[rank].cpu = std::make_unique<Dsv4HostMoeExecutor>(
            rank, false, options.rank_cpus[rank]);
        auto ready = impl_->ranks[rank].cpu->initialize();
        if (!ready.ok()) {
            result.errors.insert(result.errors.end(), ready.errors.begin(), ready.errors.end());
            return result;
        }
        impl_->ranks[rank].route_scratch.resize(kRouter);
        impl_->ranks[rank].route.experts.resize(kTopK);
        impl_->ranks[rank].route.weights.resize(kTopK);
        if (!cuda_ok(cudaSetDevice(options.devices[rank]),
                     "select rank-local executor device", result.errors) ||
            !cuda_ok(cudaMalloc(&impl_->ranks[rank].attention_reduction,
                                kHidden * sizeof(float)),
                     "allocate attention reduction", result.errors) ||
            !cuda_ok(cudaMalloc(&impl_->ranks[rank].attention_snapshot,
                                kHidden * sizeof(float)),
                     "allocate attention snapshot", result.errors) ||
            !cuda_ok(cudaMalloc(&impl_->ranks[rank].attention_publication,
                                kHidden * sizeof(std::uint16_t)),
                     "allocate attention publication", result.errors) ||
            !cuda_ok(cudaMalloc(&impl_->ranks[rank].moe_snapshot,
                                kHidden * sizeof(float)),
                     "allocate MoE snapshot", result.errors) ||
            !cuda_ok(cudaMalloc(&impl_->ranks[rank].moe_publication,
                                kHidden * sizeof(std::uint16_t)),
                     "allocate MoE publication", result.errors) ||
            !cuda_ok(cudaMalloc(&impl_->ranks[rank].head_send,
                                kVocabularyShard * sizeof(std::uint16_t)),
                     "allocate output-head shard publication", result.errors) ||
            !cuda_ok(cudaMalloc(&impl_->ranks[rank].head_full,
                                kVocabulary * sizeof(std::uint16_t)),
                     "allocate output-head full publication", result.errors) ||
            !create_events(impl_->ranks[rank].events, result.errors)) {
            return result;
        }
        auto reserved = backend_.reserve_dsv4_mhc_head(
            options.devices[rank], kVocabularyShard);
        if (!reserved.ok()) {
            result.errors.insert(result.errors.end(), reserved.errors.begin(),
                                 reserved.errors.end());
            return result;
        }
    }
    if (!cuda_ok(cudaSetDevice(options.devices[0]),
                 "select rank-0 page-patch event device", result.errors)) {
        return result;
    }
    void* replica_staging = nullptr;
    if (!cuda_ok(cudaMallocHost(
                     &replica_staging,
                     kChainSlots * kReplicaPagePatchSlotBytes),
                 "allocate fixed replica page-patch staging", result.errors)) {
        return result;
    }
    impl_->replica_page_patch_staging =
        static_cast<std::byte*>(replica_staging);
    if (!cuda_ok(cudaEventCreateWithFlags(&impl_->rank0_page_patch_ready,
                                           cudaEventDisableTiming),
                 "create rank-0 page-patch event", result.errors)) {
        return result;
    }
    std::array<ncclComm_t, kWorld> communicators{};
    if (!nccl_ok(ncclCommInitAll(
                     communicators.data(), static_cast<int>(kWorld),
                     options.devices.data()),
                 "initialize rank-local NCCL communicators", result.errors)) {
        return result;
    }
    for (std::size_t rank = 0U; rank < kWorld; ++rank) {
        impl_->ranks[rank].communicator = communicators[rank];
    }
    impl_->initialized = true;
    return result;
}

ValidationResult Dsv4RankLocalLayerExecutor::replica_page_patch_staging(
    std::size_t slot, std::size_t bytes, std::span<std::byte>& output) {
    ValidationResult result;
    output = {};
    if (!impl_->initialized || impl_->replica_page_patch_staging == nullptr ||
        slot >= kChainSlots || bytes == 0U ||
        bytes > kReplicaPagePatchSlotBytes) {
        result.errors.emplace_back(
            "rank-local replica page-patch staging request is invalid");
        return result;
    }
    output = std::span<std::byte>(
        impl_->replica_page_patch_staging +
            static_cast<std::ptrdiff_t>(slot * kReplicaPagePatchSlotBytes),
        bytes);
    return result;
}

ValidationResult Dsv4RankLocalLayerExecutor::run(
    const Dsv4RankLocalLayerCall& call, Dsv4RankLocalFailure failure,
    Dsv4RankLocalLayerResult& output) {
    ValidationResult result;
    if (!impl_->initialized || !valid_call(call, devices_)) {
        result.errors.emplace_back("rank-local layer executor call view is invalid");
        return result;
    }
    if (impl_->execution_active || (impl_->chain_active && !impl_->chain_mode)) {
        result.errors.emplace_back("rank-local layer executor call is already active");
        return result;
    }
    if (call.terminal && !impl_->chain_mode) {
        result.errors.emplace_back("terminal layer must be queued before finish_chain");
        return result;
    }
    const std::size_t slot_index = impl_->chain_mode ? impl_->chain_count : 0U;
    if (slot_index >= kChainSlots) {
        result.errors.emplace_back("rank-local layer callback slot capacity exceeded");
        return result;
    }
    output.errors.clear();
    output.success = false;
    output.global_attention_status = 0U;
    output.global_moe_status = 0U;
    output.attention_status.fill(0U);
    output.moe_status.fill(0U);
    output.timing = {};
    for (std::size_t rank = 0U; rank < kWorld; ++rank) {
        output.attention_reduction[rank].fill(0.0F);
        output.moe_reduction[rank].fill(0.0F);
        output.cpu_rank_partials[rank].fill(0.0F);
        output.attention_publication[rank].fill(0U);
        output.moe_publication[rank].fill(0U);
        output.next_attention_input[rank].fill(0.0F);
        output.outgoing_residual[rank].fill(0U);
        output.routed_experts[rank].fill(0U);
        output.routed_coefficients[rank].fill(0.0F);
        output.cpu_moe_phases[rank] = {};
        impl_->ranks[rank].moe_status = nullptr;
    }
    for (std::size_t rank = 0U; rank < kWorld; ++rank) {
        auto& slot = impl_->ranks[rank].callback_slots[slot_index];
        slot.active = true;
        slot.layer = call.layer;
        slot.router_bias = call.weights.router_bias;
        slot.router_token_experts = call.weights.router_token_experts;
        // Queued callbacks outlive this synchronous submission call. They
        // deliberately have no diagnostic result sink; otherwise the next
        // enqueue could clear a shared payload while an earlier host node is
        // still writing it. Diagnostic run() retains its caller-owned result.
        slot.result = impl_->chain_mode ? nullptr : &output;
        slot.observation = call.observation;
        slot.failure = failure;
    }
    impl_->execution_active = true;
    if (impl_->chain_mode) {
        impl_->chain_active = true;
        ++impl_->chain_count;
        if (call.terminal) impl_->terminal_active = true;
    }
    ActiveCallGuard active_call{*impl_};
    const auto started = std::chrono::steady_clock::now();
    std::array<CudaDsv4MhcDeviceView, kWorld> mhc_views{};
    std::array<bool, kWorld> moe_enqueued{};
    const auto finalize_error = [&]() -> ValidationResult {
        for (std::size_t rank = 0U; rank < kWorld; ++rank) {
            if (!moe_enqueued[rank]) continue;
            auto finished = backend_.finish_deepseek_moe_chain(devices_[rank]);
            if (!finished.ok()) {
                output.errors.insert(output.errors.end(), finished.errors.begin(),
                                     finished.errors.end());
            }
            moe_enqueued[rank] = false;
        }
        for (std::size_t rank = 0U; rank < kWorld; ++rank) {
            auto reset = backend_.dsv4_mhc_abort_branch(devices_[rank]);
            if (!reset.ok()) {
                output.errors.insert(output.errors.end(), reset.errors.begin(),
                                     reset.errors.end());
            }
            impl_->ranks[rank].moe_status = nullptr;
            if (mhc_views[rank].stream != nullptr &&
                !cuda_ok(cudaStreamSynchronize(
                             static_cast<cudaStream_t>(mhc_views[rank].stream)),
                         "synchronize exceptional layer cleanup", output.errors)) {
                continue;
            }
        }
        output.success = false;
        output.timing.total_ms = std::chrono::duration_cast<
            std::chrono::duration<double>>(std::chrono::steady_clock::now() - started)
                                      .count() * 1000.0;
        clear_failed_payload(output);
        if (impl_->chain_mode) {
            clear_callback_slots(*impl_);
        } else {
            for (auto& rank : impl_->ranks) rank.callback_slots[slot_index] = {};
        }
        result.errors = output.errors;
        return result;
    };
    for (std::size_t rank = 0U; rank < kWorld; ++rank) {
        auto& state = impl_->ranks[rank];
        auto viewed = backend_.dsv4_mhc_device_view(devices_[rank], mhc_views[rank]);
        const auto stream = viewed.ok()
            ? static_cast<cudaStream_t>(mhc_views[rank].stream) : nullptr;
        if (!viewed.ok() || stream == nullptr ||
            (!impl_->chain_mode &&
             !cuda_ok(cudaEventRecord(state.events.start, stream),
                      "record layer start", output.errors))) {
            output.errors.emplace_back("rank-local layer requires a live mHC state");
            break;
        }
    }
    if (!output.errors.empty()) {
        return finalize_error();
    }

    std::array<float, 32U> forward_sines{};
    for (std::size_t index = 0U; index < forward_sines.size(); ++index) {
        forward_sines[index] = -call.inverse_rope_sines[index];
    }
    for (std::size_t rank = 0U; rank < kWorld; ++rank) {
        const auto& weights = call.weights.rank[rank];
        CudaDsv4AttentionPrepareRequest prepare;
        prepare.query_a = weights.query_a;
        prepare.query_b = weights.query_b;
        prepare.key_value = weights.key_value;
        prepare.compressor_value = weights.compressor_value;
        prepare.compressor_gate = weights.compressor_gate;
        prepare.index_compressor_value = weights.index_compressor_value;
        prepare.index_compressor_gate = weights.index_compressor_gate;
        prepare.mhc_device = devices_[rank];
        prepare.query_norm = weights.query_norm;
        prepare.key_value_norm = weights.key_value_norm;
        prepare.rope_cosines = call.inverse_rope_cosines;
        prepare.rope_sines = forward_sines;
        prepare.publish_index_source = call.selection.active;
        const auto& page_patch = call.page_patches[rank];
        const bool host_page_patch = page_patch.callback != nullptr;
        prepare.host_callback = page_patch.callback;
        prepare.host_callback_context = page_patch.context;
        prepare.ready_page_patches = page_patch.ready_patch;
        prepare.page_patch_ready_event = !page_patch.ready_patch.empty()
            ? static_cast<void*>(impl_->rank0_page_patch_ready) : nullptr;
        prepare.page_writes = page_patch.writes;
        prepare.device_only = !host_page_patch;
        auto prepared = backend_.dsv4_prepare_attention(
            devices_[rank], prepare,
            host_page_patch
                ? std::span<float>(impl_->ranks[rank].prepared_query)
                : std::span<float>{},
            host_page_patch
                ? std::span<float>(impl_->ranks[rank].prepared_key_value)
                : std::span<float>{},
            std::span<float>(impl_->ranks[rank].prepared_compressor_values)
                .first(weights.compressor_elements),
            std::span<float>(impl_->ranks[rank].prepared_compressor_scores)
                .first(weights.compressor_elements),
            std::span<float>(
                impl_->ranks[rank].prepared_index_compressor_values)
                .first(weights.index_compressor_elements),
            std::span<float>(
                impl_->ranks[rank].prepared_index_compressor_scores)
                .first(weights.index_compressor_elements));
        if (!prepared.ok()) {
            output.errors.insert(output.errors.end(), prepared.errors.begin(),
                                 prepared.errors.end());
            break;
        }
        if (rank == 0U && call.ordered_page_patches &&
            (!cuda_ok(cudaSetDevice(devices_[rank]),
                      "select canonical page-patch event device",
                      output.errors) ||
             !cuda_ok(cudaEventRecord(
                          impl_->rank0_page_patch_ready,
                          static_cast<cudaStream_t>(mhc_views[rank].stream)),
                      "record canonical rank-0 page patch",
                      output.errors))) {
            break;
        }
        // Selection is enqueued between this layer's preparation and its
        // attention on the same stream. That ordering is what makes it safe to
        // reuse one projection and one selection workspace per device across
        // 43 queued layers: layer N's resolve reads the winners before layer
        // N+1's selection overwrites them.
        CudaDsv4DeviceCandidateResolution resolution;
        if (call.selection.active) {
            CudaDsv4IndexProjectionRequest projection;
            projection.query_projection = call.selection.query_projection[rank];
            projection.weight_projection =
                call.selection.weight_projection[rank];
            projection.rope_cosines = call.selection.rope_cosines;
            projection.rope_sines = call.selection.rope_sines;
            projection.heads = call.selection.heads;
            projection.head_dim = call.selection.head_dim;
            projection.rope_dim = call.selection.rope_dim;
            projection.weight_scale = call.selection.weight_scale;
            auto projected = backend_.dsv4_index_projections(
                devices_[rank], projection, {}, {});
            if (!projected.ok()) {
                output.errors.insert(output.errors.end(),
                                     projected.errors.begin(),
                                     projected.errors.end());
                break;
            }
            CudaDsv4PhysicalIndexRequest index;
            index.pages = call.selection.index_pages[rank];
            index.heads = call.selection.heads;
            index.head_dim = call.selection.head_dim;
            index.top_k = call.selection.top_k;
            index.device_projected = true;
            auto selected = backend_.dsv4_physical_lightning_index(
                devices_[rank], index, {}, &resolution.selection);
            if (!selected.ok()) {
                output.errors.insert(output.errors.end(),
                                     selected.errors.begin(),
                                     selected.errors.end());
                break;
            }
            resolution.blocks = call.selection.blocks;
            resolution.compressed_width = call.selection.compressed_width;
        }
        CudaDsv4PagedAttentionMhcRequest request;
        request.attention.head_sinks = call.head_sinks
            .subspan(rank * 32U, 32U);
        request.attention.pages = call.pages[rank];
        request.attention.candidates = call.candidates;
        if (call.selection.active) request.attention.resolution = &resolution;
        request.attention.scale = kAttentionScale;
        request.attention.maximum_workspace_bytes = 8ULL << 20U;
        request.inverse_rope_cosines = call.inverse_rope_cosines;
        request.inverse_rope_sines = call.inverse_rope_sines;
        request.output_a = weights.output_a;
        request.output_b = weights.output_b;
        request.mhc_device = devices_[rank];
        request.rank_local = true;
        request.head_offset = static_cast<std::uint32_t>(rank * 32U);
        request.rank_local_raw_fp32_reduction =
            impl_->ranks[rank].attention_reduction;
        auto attended = backend_.dsv4_paged_attention_to_mhc(
            devices_[rank], request);
        if (!attended.ok()) {
            output.errors.insert(output.errors.end(), attended.errors.begin(),
                                 attended.errors.end());
            break;
        }
        auto viewed = backend_.dsv4_mhc_device_view(devices_[rank], mhc_views[rank]);
        if (!viewed.ok()) {
            output.errors.emplace_back("rank-local attention device view failed");
            break;
        }
        const auto stream = static_cast<cudaStream_t>(mhc_views[rank].stream);
        if (!impl_->chain_mode &&
            !cuda_ok(cudaEventRecord(impl_->ranks[rank].events.attention, stream),
                     "record attention completion", output.errors)) break;
        if ((failure == Dsv4RankLocalFailure::AttentionPreRank0 && rank == 0U) ||
            (failure == Dsv4RankLocalFailure::AttentionPreRank1 && rank == 1U)) {
            if (!cuda_ok(cudaMemsetAsync(mhc_views[rank].status, 1,
                                         sizeof(unsigned int), stream),
                         "inject attention pre-collective failure", output.errors)) break;
        }
    }
    if (!output.errors.empty()) {
        return finalize_error();
    }
    if (!collective_data(*impl_,
                         {CudaDsv4HostMoeDeviceView{mhc_views[0].stream,
                                                      impl_->ranks[0].attention_reduction,
                                                      mhc_views[0].status},
                          CudaDsv4HostMoeDeviceView{mhc_views[1].stream,
                                                      impl_->ranks[1].attention_reduction,
                                                      mhc_views[1].status}},
                         output.errors)) {
        return finalize_error();
    }
    if ((failure == Dsv4RankLocalFailure::AttentionPostRank0) ||
        (failure == Dsv4RankLocalFailure::AttentionPostRank1)) {
        const auto rank = failure_rank(failure);
        if (!cuda_ok(cudaSetDevice(devices_[static_cast<std::size_t>(rank)]),
                     "select attention post-failure device", output.errors) ||
            !cuda_ok(cudaMemsetAsync(
                         mhc_views[static_cast<std::size_t>(rank)].status, 1,
                         sizeof(unsigned int),
                         static_cast<cudaStream_t>(
                             mhc_views[static_cast<std::size_t>(rank)].stream)),
                     "inject attention post-collective failure", output.errors)) {
            return finalize_error();
        }
    }
    std::array<unsigned int*, kWorld> attention_status{
        mhc_views[0].status, mhc_views[1].status};
    std::array<cudaStream_t, kWorld> attention_streams{
        static_cast<cudaStream_t>(mhc_views[0].stream),
        static_cast<cudaStream_t>(mhc_views[1].stream)};
    if (!collective_status(*impl_, attention_status, attention_streams,
                           output.errors)) {
        return finalize_error();
    }
    if (!impl_->chain_mode) {
        for (std::size_t rank = 0U; rank < kWorld; ++rank) {
            if (!cuda_ok(cudaEventRecord(impl_->ranks[rank].events.attention_collective,
                                         attention_streams[rank]),
                         "record attention collective completion", output.errors)) break;
        }
    }
    const auto attention_failed = is_attention_failure(failure);
    if (attention_failed) {
        for (std::size_t rank = 0U; rank < kWorld; ++rank) {
            const auto stream = attention_streams[rank];
            if (!zero_device(devices_[rank], impl_->ranks[rank].attention_snapshot,
                             kHidden * sizeof(float), stream,
                             "clear failed attention snapshot", output.errors) ||
                !zero_device(devices_[rank], impl_->ranks[rank].attention_publication,
                             kHidden * sizeof(std::uint16_t), stream,
                             "clear failed attention publication", output.errors) ||
                !zero_device(devices_[rank], impl_->ranks[rank].moe_snapshot,
                             kHidden * sizeof(float), stream,
                             "clear withheld MoE snapshot", output.errors) ||
                !zero_device(devices_[rank], impl_->ranks[rank].moe_publication,
                             kHidden * sizeof(std::uint16_t), stream,
                             "clear withheld MoE publication", output.errors)) break;
        }
        for (std::size_t rank = 0U; rank < kWorld; ++rank) {
            if (!cuda_ok(cudaStreamSynchronize(attention_streams[rank]),
                         "close failed attention layer", output.errors)) break;
        }
        output.success = false;
    } else {
        for (std::size_t rank = 0U; rank < kWorld; ++rank) {
            auto committed = backend_.dsv4_mhc_commit_reduced_branch(
                devices_[rank], impl_->ranks[rank].attention_reduction);
            if (!committed.ok()) {
                output.errors.insert(output.errors.end(), committed.errors.begin(),
                                     committed.errors.end());
                break;
            }
            if (!impl_->chain_mode &&
                (!copy_device_to_device(
                     devices_[rank], impl_->ranks[rank].attention_snapshot,
                     impl_->ranks[rank].attention_reduction,
                     kHidden * sizeof(float), attention_streams[rank],
                     "snapshot attention reduction", output.errors) ||
                 !copy_device_to_device(
                     devices_[rank], impl_->ranks[rank].attention_publication,
                     mhc_views[rank].branch, kHidden * sizeof(std::uint16_t),
                     attention_streams[rank], "snapshot attention publication",
                     output.errors) ||
                 !cuda_ok(cudaEventRecord(
                              impl_->ranks[rank].events.attention_publication,
                              attention_streams[rank]),
                          "record attention publication", output.errors))) break;
        }
        if (!output.errors.empty()) {
            return finalize_error();
        }
        for (std::size_t rank = 0U; rank < kWorld; ++rank) {
            const auto& weights = call.weights.rank[rank];
            auto transitioned = backend_.dsv4_mhc_transition_router_device(
                devices_[rank], *weights.ffn_mhc, *weights.router);
            if (!transitioned.ok()) {
                output.errors.insert(output.errors.end(), transitioned.errors.begin(),
                                     transitioned.errors.end());
                break;
            }
            auto viewed = backend_.dsv4_mhc_device_view(devices_[rank], mhc_views[rank]);
            if (!viewed.ok()) {
                output.errors.emplace_back("router transition device view failed");
                break;
            }
            if (!impl_->chain_mode &&
                !cuda_ok(cudaEventRecord(
                    impl_->ranks[rank].events.router,
                    static_cast<cudaStream_t>(mhc_views[rank].stream)),
                    "record router transition", output.errors)) break;
        }
        if (!output.errors.empty()) {
            return finalize_error();
        }
        std::array<CudaDsv4HostMoeDeviceView, kWorld> moe_views{};
        for (std::size_t rank = 0U; rank < kWorld; ++rank) {
            if (failure == Dsv4RankLocalFailure::MoeBeforeEnqueueRank1 &&
                rank == 1U) {
                output.errors.emplace_back(
                    "injected MoeBeforeEnqueueRank1 before rank 1 enqueue");
                break;
            }
            const auto& weights = call.weights.rank[rank];
            auto enqueued = backend_.enqueue_dsv4_host_moe_from_device_input_device_view(
                devices_[rank], *weights.shared, kSwiGluLimit,
                host_moe_callback,
                &impl_->ranks[rank].callback_contexts[slot_index], moe_views[rank],
                impl_->split_route_callback ? route_callback : nullptr);
            if (!enqueued.ok()) {
                output.errors.insert(output.errors.end(), enqueued.errors.begin(),
                                     enqueued.errors.end());
                break;
            }
            moe_enqueued[rank] = true;
            impl_->ranks[rank].moe_status = moe_views[rank].status;
            if ((failure == Dsv4RankLocalFailure::MoePreRank0 && rank == 0U) ||
                (failure == Dsv4RankLocalFailure::MoePreRank1 && rank == 1U)) {
                if (!cuda_ok(cudaMemsetAsync(
                                 moe_views[rank].status, 1, sizeof(unsigned int),
                                 static_cast<cudaStream_t>(moe_views[rank].stream)),
                             "inject MoE pre-collective failure", output.errors)) break;
            }
            if (!impl_->chain_mode &&
                !cuda_ok(cudaEventRecord(
                             impl_->ranks[rank].events.moe,
                             static_cast<cudaStream_t>(moe_views[rank].stream)),
                         "record host MoE completion", output.errors)) break;
        }
        if (!output.errors.empty()) {
            return finalize_error();
        }
        if (!collective_data(*impl_, moe_views, output.errors)) {
            return finalize_error();
        }
        if (failure == Dsv4RankLocalFailure::MoePostRank0 ||
            failure == Dsv4RankLocalFailure::MoePostRank1) {
            const auto rank = failure_rank(failure);
            if (!cuda_ok(cudaSetDevice(devices_[static_cast<std::size_t>(rank)]),
                         "select MoE post-failure device", output.errors) ||
                !cuda_ok(cudaMemsetAsync(
                             moe_views[static_cast<std::size_t>(rank)].status, 1,
                             sizeof(unsigned int),
                             static_cast<cudaStream_t>(
                                 moe_views[static_cast<std::size_t>(rank)].stream)),
                         "inject MoE post-collective failure", output.errors)) {
                return finalize_error();
            }
        }
        std::array<unsigned int*, kWorld> moe_status{
            moe_views[0].status, moe_views[1].status};
        std::array<cudaStream_t, kWorld> moe_streams{
            static_cast<cudaStream_t>(moe_views[0].stream),
            static_cast<cudaStream_t>(moe_views[1].stream)};
        if (!collective_status(*impl_, moe_status, moe_streams,
                               output.errors)) {
            return finalize_error();
        }
        if (!impl_->chain_mode) {
            for (std::size_t rank = 0U; rank < kWorld; ++rank) {
                if (!cuda_ok(cudaEventRecord(impl_->ranks[rank].events.moe_collective,
                                             moe_streams[rank]),
                             "record MoE collective completion", output.errors)) break;
            }
        }
        const auto moe_failed = is_moe_failure(failure);
        if (moe_failed) {
            for (std::size_t rank = 0U; rank < kWorld; ++rank) {
                if (!zero_device(devices_[rank], impl_->ranks[rank].moe_snapshot,
                                 kHidden * sizeof(float), moe_streams[rank],
                                 "clear failed MoE snapshot", output.errors) ||
                    !zero_device(devices_[rank], impl_->ranks[rank].moe_publication,
                                 kHidden * sizeof(std::uint16_t), moe_streams[rank],
                                 "clear failed MoE publication", output.errors)) break;
            }
            if (!impl_->chain_mode) {
                for (std::size_t rank = 0U; rank < kWorld; ++rank) {
                    auto finished = backend_.finish_deepseek_moe_chain(
                        devices_[rank]);
                    if (finished.ok()) {
                        output.errors.emplace_back(
                            "failed MoE chain unexpectedly completed");
                    }
                    moe_enqueued[rank] = false;
                }
                for (std::size_t rank = 0U; rank < kWorld; ++rank) {
                    if (!cuda_ok(cudaStreamSynchronize(moe_streams[rank]),
                                 "close failed MoE layer", output.errors)) break;
                }
            }
            output.success = false;
        } else {
            for (std::size_t rank = 0U; rank < kWorld; ++rank) {
                auto committed = backend_.dsv4_mhc_commit_reduced_branch(
                    devices_[rank], moe_views[rank].output);
                if (!committed.ok()) {
                    output.errors.insert(output.errors.end(), committed.errors.begin(),
                                         committed.errors.end());
                    break;
                }
                if (!impl_->chain_mode &&
                    (!copy_device_to_device(
                         devices_[rank], impl_->ranks[rank].moe_snapshot,
                         moe_views[rank].output, kHidden * sizeof(float),
                         moe_streams[rank], "snapshot MoE reduction", output.errors) ||
                     !copy_device_to_device(
                         devices_[rank], impl_->ranks[rank].moe_publication,
                         mhc_views[rank].branch, kHidden * sizeof(std::uint16_t),
                         moe_streams[rank], "snapshot MoE publication", output.errors) ||
                     !cuda_ok(cudaEventRecord(
                                  impl_->ranks[rank].events.moe_publication,
                                  moe_streams[rank]),
                              "record MoE publication", output.errors))) break;
            }
            if (!output.errors.empty()) {
                return finalize_error();
            }
            if (!call.terminal) {
                for (std::size_t rank = 0U; rank < kWorld; ++rank) {
                    const auto& weights = call.weights.rank[rank];
                    auto final = backend_.dsv4_mhc_transition_next_device(
                        devices_[rank], *weights.next_attention_mhc);
                    if (!final.ok()) {
                        output.errors.insert(output.errors.end(), final.errors.begin(),
                                             final.errors.end());
                        break;
                    }
                    auto viewed = backend_.dsv4_mhc_device_view(devices_[rank], mhc_views[rank]);
                    if (!viewed.ok() ||
                        (!impl_->chain_mode &&
                         !cuda_ok(cudaEventRecord(
                                      impl_->ranks[rank].events.final,
                                      static_cast<cudaStream_t>(mhc_views[rank].stream)),
                                  "record final mHC transition", output.errors))) break;
                }
            }
            if (!output.errors.empty()) {
                return finalize_error();
            }
            if (!impl_->chain_mode) {
                for (std::size_t rank = 0U; rank < kWorld; ++rank) {
                    auto finished = backend_.finish_deepseek_moe_chain(devices_[rank]);
                    if (!finished.ok()) {
                        output.errors.insert(output.errors.end(), finished.errors.begin(),
                                             finished.errors.end());
                        break;
                    }
                    moe_enqueued[rank] = false;
                }
                if (!output.errors.empty()) {
                    return finalize_error();
                }
            }
            output.success = true;
        }
    }

    // This is the one diagnostic boundary for the layer. It is intentionally
    // after all final transitions and callback drains; no host continuation is
    // used to make a decision in the timed sequence. Chain mode defers this
    // entire boundary until finish_chain().
    const auto boundary_started = std::chrono::steady_clock::now();
    if (!impl_->chain_mode) {
    for (std::size_t rank = 0U; rank < kWorld; ++rank) {
        const auto stream = attention_streams[rank];
        if (!cuda_ok(cudaStreamSynchronize(stream), "synchronize layer diagnostic boundary",
                     output.errors)) continue;
        auto view_status = backend_.dsv4_mhc_device_view(devices_[rank], mhc_views[rank]);
        if (!view_status.ok()) {
            output.errors.insert(output.errors.end(), view_status.errors.begin(),
                                 view_status.errors.end());
            continue;
        }
        if (!cuda_ok(cudaMemcpy(output.attention_reduction[rank].data(),
                                impl_->ranks[rank].attention_snapshot,
                                kHidden * sizeof(float), cudaMemcpyDeviceToHost),
                     "copy attention diagnostic", output.errors) ||
            !cuda_ok(cudaMemcpy(output.attention_publication[rank].data(),
                                impl_->ranks[rank].attention_publication,
                                kHidden * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
                     "copy attention publication diagnostic", output.errors) ||
            !cuda_ok(cudaMemcpy(output.moe_reduction[rank].data(),
                                impl_->ranks[rank].moe_snapshot,
                                kHidden * sizeof(float), cudaMemcpyDeviceToHost),
                     "copy MoE diagnostic", output.errors) ||
            !cuda_ok(cudaMemcpy(output.moe_publication[rank].data(),
                                impl_->ranks[rank].moe_publication,
                                kHidden * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
                     "copy MoE publication diagnostic", output.errors)) {
            continue;
        }
        std::array<std::uint16_t, kHidden> next_input{};
        if (!cuda_ok(cudaMemcpy(next_input.data(), mhc_views[rank].layer_input,
                                kHidden * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
                     "copy next-attention state", output.errors) ||
            !cuda_ok(cudaMemcpy(output.outgoing_residual[rank].data(),
                                mhc_views[rank].residual,
                                kMhc * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
                     "copy outgoing residual", output.errors)) continue;
        for (std::size_t index = 0U; index < kHidden; ++index) {
            const auto bits = static_cast<std::uint32_t>(next_input[index]) << 16U;
            output.next_attention_input[rank][index] = std::bit_cast<float>(bits);
        }
        std::uint32_t attention_status = 0U;
        if (!cuda_ok(cudaMemcpy(&attention_status, mhc_views[rank].status,
                                sizeof(attention_status), cudaMemcpyDeviceToHost),
                     "copy attention status", output.errors)) continue;
        output.attention_status[rank] = attention_status;
        output.global_attention_status = std::max(output.global_attention_status,
                                                  attention_status);
        if (impl_->ranks[rank].moe_status != nullptr) {
            std::uint32_t moe_status = 0U;
            if (!cuda_ok(cudaMemcpy(&moe_status, impl_->ranks[rank].moe_status,
                                    sizeof(moe_status), cudaMemcpyDeviceToHost),
                         "copy MoE status", output.errors)) continue;
            output.moe_status[rank] = moe_status;
            output.global_moe_status = std::max(output.global_moe_status,
                                                moe_status);
        }
    }
    output.timing.diagnostic_boundary_ms =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - boundary_started).count() *
        1000.0;
    const auto elapsed_wall = std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::steady_clock::now() - started);
    output.timing.total_ms = elapsed_wall.count() * 1000.0;
    // Event timing is best-effort diagnostics; the one host wall interval is
    // still retained so an event failure cannot turn an exact run into a false
    // success.
    if (output.success && !impl_->chain_mode) {
        for (std::size_t rank = 0U; rank < kWorld; ++rank) {
            auto& timing = output.timing;
            const auto& events = impl_->ranks[rank].events;
            double phase = 0.0;
            if (!elapsed(events.start, events.attention, phase, output.errors)) continue;
            timing.attention_ms = std::max(timing.attention_ms, phase);
            if (!elapsed(events.attention, events.attention_collective, phase,
                         output.errors)) continue;
            timing.attention_collective_ms =
                std::max(timing.attention_collective_ms, phase);
            if (!elapsed(events.attention_collective, events.attention_publication,
                         phase, output.errors)) continue;
            timing.attention_publication_ms =
                std::max(timing.attention_publication_ms, phase);
            if (!elapsed(events.attention_publication, events.router, phase,
                         output.errors)) continue;
            timing.transition_router_ms = std::max(timing.transition_router_ms, phase);
            if (!elapsed(events.router, events.moe, phase, output.errors)) continue;
            timing.shared_gpu_rank0_ms = std::max(timing.shared_gpu_rank0_ms,
                                                  rank == 0U ? phase : 0.0);
            timing.shared_gpu_rank1_ms = std::max(timing.shared_gpu_rank1_ms,
                                                  rank == 1U ? phase : 0.0);
            if (!elapsed(events.moe, events.moe_collective, phase, output.errors)) continue;
            timing.moe_collective_ms = std::max(timing.moe_collective_ms, phase);
            if (!elapsed(events.moe_collective, events.moe_publication, phase,
                         output.errors)) continue;
            timing.moe_publication_ms = std::max(timing.moe_publication_ms, phase);
            if (elapsed(events.moe_publication, events.final, phase,
                        output.errors)) {
                timing.final_transition_ms = std::max(timing.final_transition_ms, phase);
            }
        }
    }
    if (!output.success && !impl_->chain_mode) {
        for (std::size_t rank = 0U; rank < kWorld; ++rank) {
            if (mhc_views[rank].stream == nullptr) continue;
            auto reset = backend_.dsv4_mhc_abort_branch(devices_[rank]);
            if (!reset.ok()) {
                output.errors.insert(output.errors.end(), reset.errors.begin(),
                                     reset.errors.end());
                continue;
            }
            if (!cuda_ok(cudaStreamSynchronize(
                             static_cast<cudaStream_t>(mhc_views[rank].stream)),
                         "reset failed layer state", output.errors)) continue;
        }
        clear_failed_payload(output);
    }
    }
    if (output.success && !output.errors.empty()) return finalize_error();
    if (!impl_->chain_mode) {
        for (auto& rank : impl_->ranks) rank.callback_slots[slot_index] = {};
    }
    result.errors = output.errors;
    if (!output.errors.empty()) return result;
    return result;
}

ValidationResult Dsv4RankLocalLayerExecutor::enqueue_chain_layer(
    const Dsv4RankLocalLayerCall& call, Dsv4RankLocalFailure failure) {
    ValidationResult result;
    if (!impl_->initialized || impl_->execution_active || impl_->chain_mode) {
        result.errors.emplace_back("rank-local chain is already active");
        return result;
    }
    if (impl_->terminal_active) {
        result.errors.emplace_back("rank-local chain is already terminal");
        return result;
    }
    if (impl_->chain_count >= kChainSlots) {
        result.errors.emplace_back("rank-local chain accepts at most 43 layers");
        return result;
    }
    if (impl_->chain_count == 0U) {
        for (auto& rank : impl_->ranks) {
            rank.chain_cpu_moe_phases = {};
        }
    }
    if (failure != Dsv4RankLocalFailure::None &&
        (!call.terminal || failure == Dsv4RankLocalFailure::MoeBeforeEnqueueRank1)) {
        result.errors.emplace_back(
            "rank-local chain failure injection requires a logical terminal-layer arm");
        return result;
    }
    impl_->chain_mode = true;
    Dsv4RankLocalLayerResult enqueue_output;
    result = run(call, failure, enqueue_output);
    impl_->chain_mode = false;
    return result;
}

ValidationResult Dsv4RankLocalLayerExecutor::finish_chain(
    Dsv4RankLocalLayerChainResult* output,
    const Dsv4RankLocalHeadRequest* head) {
    ValidationResult result;
    if (!impl_->initialized || impl_->execution_active || impl_->chain_mode) {
        result.errors.emplace_back("rank-local chain finish is not available now");
        return result;
    }
    if (!impl_->chain_active || impl_->chain_count == 0U) {
        result.errors.emplace_back("rank-local chain has no queued layers");
        return result;
    }
    if (head != nullptr &&
        (!impl_->terminal_active || !valid_head_request(*head, devices_))) {
        result.errors.emplace_back("rank-local terminal output-head request is invalid");
        return result;
    }
    Dsv4RankLocalLayerChainResult discarded;
    auto& chain_result = output == nullptr ? discarded : *output;
    chain_result = {};
    chain_result.chain_count = impl_->chain_count;
    chain_result.terminal = impl_->terminal_active;
    for (std::size_t rank = 0U; rank < kWorld; ++rank) {
        if (impl_->ranks[rank].moe_status == nullptr) continue;
        auto finished = backend_.finish_deepseek_moe_chain(devices_[rank]);
        result.errors.insert(result.errors.end(), finished.errors.begin(),
                             finished.errors.end());
    }
    // The CPU callbacks update these records asynchronously. Read them only
    // after both rank streams have drained above.
    for (std::size_t rank = 0U; rank < kWorld; ++rank) {
        chain_result.cpu_moe_phases[rank] =
            impl_->ranks[rank].chain_cpu_moe_phases;
    }
    for (std::size_t rank = 0U; rank < kWorld; ++rank) {
        CudaDsv4MhcDeviceView view{};
        auto viewed = backend_.dsv4_mhc_device_view(devices_[rank], view);
        result.errors.insert(result.errors.end(), viewed.errors.begin(),
                             viewed.errors.end());
        if (!viewed.ok() || view.status == nullptr) continue;
        static_cast<void>(cudaSetDevice(devices_[rank]));
        std::uint32_t attention_status = 0U;
        if (cudaMemcpy(&attention_status, view.status,
                       sizeof(attention_status), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
            result.errors.emplace_back(
                "copy rank-local chain attention status failed");
        }
        chain_result.attention_status[rank] = attention_status;
        chain_result.global_attention_status = std::max(
            chain_result.global_attention_status, attention_status);
        if (attention_status != 0U) {
            // The status is the attention command's own word, carried
            // verbatim. Bits 1 to 3 are raised by an in-chain candidate
            // resolution and say which of its three causes fired; bit 0 is the
            // plain failure any DeepSeek attention kernel reports.
            result.errors.emplace_back(
                "rank-local chain attention device status failed on rank " +
                std::to_string(rank) + " with status " +
                std::to_string(attention_status));
        }
        if (impl_->ranks[rank].moe_status != nullptr) {
            std::uint32_t moe_status = 0U;
            if (cudaMemcpy(&moe_status, impl_->ranks[rank].moe_status,
                           sizeof(moe_status), cudaMemcpyDeviceToHost) !=
                cudaSuccess) {
                result.errors.emplace_back(
                    "copy rank-local chain MoE status failed");
            }
            chain_result.moe_status[rank] = moe_status;
            chain_result.global_moe_status = std::max(
                chain_result.global_moe_status, moe_status);
            if (moe_status != 0U)
                result.errors.emplace_back(
                    "rank-local chain MoE device status failed");
        }
        if (chain_result.terminal && result.ok()) {
            if (cudaMemcpy(chain_result.terminal_weighted[rank].data(),
                           view.weighted, kHidden * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost) != cudaSuccess ||
                cudaMemcpy(chain_result.terminal_input[rank].data(),
                           view.layer_input, kHidden * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                result.errors.emplace_back(
                    "copy terminal mHC weighted/input failed");
            }
        }
    }
    std::array<bool, kWorld> head_consumed{};
    if (result.ok() && chain_result.terminal && head == nullptr) {
        for (std::size_t rank = 0U; rank < kWorld; ++rank) {
            auto finished = backend_.dsv4_mhc_finish_device(
                devices_[rank], chain_result.final_hidden[rank]);
            result.errors.insert(result.errors.end(), finished.errors.begin(),
                                 finished.errors.end());
        }
    }
    const auto terminal_head_started = std::chrono::steady_clock::now();
    if (result.ok() && chain_result.terminal && head != nullptr) {
        std::array<CudaDsv4MhcHeadDeviceView, kWorld> views{};
        for (std::size_t rank = 0U; rank < kWorld; ++rank) {
            auto enqueued = backend_.enqueue_dsv4_mhc_finish_head_device(
                devices_[rank], *head->heads[rank], head->callbacks[rank],
                head->callback_contexts[rank], &views[rank]);
            result.errors.insert(result.errors.end(), enqueued.errors.begin(),
                                 enqueued.errors.end());
            if (enqueued.ok()) head_consumed[rank] = true;
        }
        if (result.ok()) {
            for (const auto& view : views) {
                if (view.stream == nullptr || view.logits == nullptr ||
                    view.final_hidden == nullptr ||
                    view.count != kVocabularyShard) {
                    result.errors.emplace_back(
                        "rank-local output-head device view is invalid");
                    break;
                }
            }
        }
        if (result.ok()) {
            constexpr unsigned int threads = 256U;
            constexpr unsigned int blocks = static_cast<unsigned int>(
                (kVocabularyShard + threads - 1U) / threads);
            for (std::size_t rank = 0U; rank < kWorld; ++rank) {
                static_cast<void>(cudaSetDevice(devices_[rank]));
                publish_head_shard<<<blocks, threads, 0U,
                    static_cast<cudaStream_t>(views[rank].stream)>>>(
                        views[rank].logits, impl_->ranks[rank].head_send);
                if (!cuda_ok(cudaGetLastError(),
                             "publish rank-local output-head shard",
                             result.errors)) break;
            }
        }
        if (result.ok() &&
            nccl_ok(ncclGroupStart(), "start BF16 output-head all-gather",
                    result.errors)) {
            std::array<ncclResult_t, kWorld> gathered{};
            for (std::size_t rank = 0U; rank < kWorld; ++rank) {
                static_cast<void>(cudaSetDevice(devices_[rank]));
                gathered[rank] = ncclAllGather(
                    impl_->ranks[rank].head_send,
                    impl_->ranks[rank].head_full, kVocabularyShard,
                    ncclBfloat16, impl_->ranks[rank].communicator,
                    static_cast<cudaStream_t>(views[rank].stream));
            }
            const auto group = ncclGroupEnd();
            static_cast<void>(nccl_ok(
                group, "finish BF16 output-head all-gather", result.errors));
            for (const auto status : gathered) {
                static_cast<void>(nccl_ok(
                    status, "BF16 output-head all-gather", result.errors));
            }
        }
        for (std::size_t rank = 0U; rank < kWorld; ++rank) {
            if (!head_consumed[rank]) continue;
            static_cast<void>(cudaSetDevice(devices_[rank]));
            static_cast<void>(cuda_ok(
                cudaStreamSynchronize(
                    static_cast<cudaStream_t>(views[rank].stream)),
                "synchronize terminal output head", result.errors));
            auto completed = backend_.complete_dsv4_mhc_head_device(
                devices_[rank], head->local_logits[rank]);
            result.errors.insert(result.errors.end(), completed.errors.begin(),
                                 completed.errors.end());
            std::array<std::uint16_t, kMhc> encoded_hidden{};
            if (!cuda_ok(cudaMemcpy(
                    encoded_hidden.data(), views[rank].final_hidden,
                    encoded_hidden.size() * sizeof(std::uint16_t),
                    cudaMemcpyDeviceToHost),
                    "copy terminal output-head hidden", result.errors) ||
                !cuda_ok(cudaMemcpy(
                    head->published_logits[rank].data(),
                    impl_->ranks[rank].head_full,
                    kVocabulary * sizeof(std::uint16_t),
                    cudaMemcpyDeviceToHost),
                    "copy terminal gathered logits", result.errors)) continue;
            for (std::size_t index = 0U; index < kMhc; ++index) {
                chain_result.final_hidden[rank][index] =
                    bf16_to_float(encoded_hidden[index]);
            }
        }
        if (result.ok()) {
            if (!std::equal(head->published_logits[0].begin(),
                            head->published_logits[0].end(),
                            head->published_logits[1].begin())) {
                result.errors.emplace_back(
                    "terminal gathered logits differ between ranks");
            }
            for (std::size_t rank = 0U; rank < kWorld && result.ok(); ++rank) {
                const auto offset = rank * kVocabularyShard;
                for (std::size_t index = 0U; index < kVocabularyShard; ++index) {
                    const auto published =
                        head->published_logits[rank][offset + index];
                    if (published !=
                            host_float_to_bf16(head->local_logits[rank][index]) ||
                        !std::isfinite(bf16_to_float(published))) {
                        result.errors.emplace_back(
                            "terminal local/gathered logits association failed");
                        break;
                    }
                }
            }
        }
        if (result.ok()) {
            const auto logits = std::span<const std::uint16_t>(
                head->published_logits[0]);
            chain_result.terminal_logits_hash = hash_bf16(logits);
            float maximum = -std::numeric_limits<float>::infinity();
            std::uint32_t token = 0U;
            for (std::size_t index = 0U; index < logits.size(); ++index) {
                const auto value = bf16_to_float(logits[index]);
                if (value > maximum) {
                    maximum = value;
                    token = static_cast<std::uint32_t>(index);
                }
            }
            chain_result.next_token = token;
        }
    }
    if (chain_result.terminal && head != nullptr) {
        chain_result.terminal_head_ms = std::chrono::duration_cast<
            std::chrono::duration<double>>(
                std::chrono::steady_clock::now() - terminal_head_started)
                                            .count() * 1000.0;
    }
    if (!result.ok()) {
        for (std::size_t rank = 0U; rank < kWorld; ++rank) {
            if (head != nullptr) {
                std::fill(head->local_logits[rank].begin(),
                          head->local_logits[rank].end(), 0.0F);
                std::fill(head->published_logits[rank].begin(),
                          head->published_logits[rank].end(), 0U);
            }
            if (head_consumed[rank]) continue;
            auto reset = backend_.dsv4_mhc_abort_branch(devices_[rank]);
            result.errors.insert(result.errors.end(), reset.errors.begin(),
                                 reset.errors.end());
            if (reset.ok()) {
                CudaDsv4MhcDeviceView view;
                if (backend_.dsv4_mhc_device_view(devices_[rank], view).ok() &&
                    view.stream != nullptr) {
                    static_cast<void>(cudaSetDevice(devices_[rank]));
                    static_cast<void>(cudaStreamSynchronize(
                        static_cast<cudaStream_t>(view.stream)));
                }
            }
        }
        for (auto& values : chain_result.terminal_weighted) values.fill(0U);
        for (auto& values : chain_result.terminal_input) values.fill(0U);
        for (auto& values : chain_result.final_hidden) values.fill(0.0F);
        chain_result.terminal_logits_hash = 0U;
        chain_result.next_token = 0U;
    }
    clear_callback_slots(*impl_);
    return result;
}

ValidationResult Dsv4RankLocalLayerExecutor::abort_chain() {
    ValidationResult result;
    if (!impl_->initialized || impl_->execution_active || impl_->chain_mode) {
        result.errors.emplace_back("rank-local chain abort is not available now");
        return result;
    }
    if (!impl_->chain_active || impl_->chain_count == 0U) {
        result.errors.emplace_back("rank-local chain has no queued layers");
        return result;
    }
    for (std::size_t rank = 0U; rank < kWorld; ++rank) {
        auto finished = backend_.finish_deepseek_moe_chain(devices_[rank]);
        result.errors.insert(result.errors.end(), finished.errors.begin(),
                             finished.errors.end());
        auto reset = backend_.dsv4_mhc_abort_branch(devices_[rank]);
        result.errors.insert(result.errors.end(), reset.errors.begin(),
                             reset.errors.end());
        if (reset.ok()) {
            CudaDsv4MhcDeviceView view;
            auto viewed = backend_.dsv4_mhc_device_view(devices_[rank], view);
            if (viewed.ok() && view.stream != nullptr) {
                static_cast<void>(cudaSetDevice(devices_[rank]));
                if (cudaStreamSynchronize(static_cast<cudaStream_t>(view.stream)) !=
                    cudaSuccess) {
                    result.errors.emplace_back("synchronize rank-local chain abort");
                }
            }
        }
    }
    clear_callback_slots(*impl_);
    return result;
}

}  // namespace strata
