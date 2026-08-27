#pragma once

#include "strata/device/cuda_backend.hpp"
#include "strata/models/deepseek/deepseek_checkpoint.hpp"
#include "strata/models/deepseek/deepseek_host_moe_executor.hpp"
#include "strata/models/deepseek/deepseek_static_expert_tier.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace strata {

inline constexpr std::size_t kDsv4RankLocalHidden = 4096U;
inline constexpr std::size_t kDsv4RankLocalMhc = 4U * kDsv4RankLocalHidden;
inline constexpr std::size_t kDsv4RankLocalRouter = 256U;
inline constexpr std::size_t kDsv4RankLocalTopK = 6U;
inline constexpr std::size_t kDsv4RankLocalFinalHidden = kDsv4RankLocalMhc;
inline constexpr std::size_t kDsv4RankLocalVocabulary = 129280U;
inline constexpr std::size_t kDsv4RankLocalVocabularyShard =
    kDsv4RankLocalVocabulary / 2U;

struct Dsv4RankLocalLayerRankObservation {
    bool invoked{};
    bool completed{};
    bool success{};
    bool failure{};
    std::array<std::uint16_t, kDsv4RankLocalHidden> encoded_ffn_input{};
    std::array<float, kDsv4RankLocalRouter> router_logits{};
    std::array<std::uint32_t, kDsv4RankLocalTopK> routed_experts{};
    std::array<float, kDsv4RankLocalTopK> routed_coefficients{};
    std::array<float, kDsv4RankLocalHidden> cpu_rank_partial{};
    Dsv4HostMoePhaseTimings cpu_moe_phases{};
};

// Caller-owned fixed storage for one layer. Rank callbacks write only their
// own disjoint field; the executor never allocates or owns this object.
struct Dsv4RankLocalLayerObservation {
    std::array<Dsv4RankLocalLayerRankObservation, 2U> rank{};
};
struct Dsv4RankLocalLayerRankWeights {
    const CudaWeight* query_a{};
    const CudaWeight* query_b{};
    const CudaWeight* key_value{};
    // Optional unsharded compressor projections. A single rank may own these:
    // its ordered page callback advances the canonical compressor state and
    // publishes the identical encoded row to both device pages.
    const CudaWeight* compressor_value{};
    const CudaWeight* compressor_gate{};
    const CudaWeight* index_compressor_value{};
    const CudaWeight* index_compressor_gate{};
    std::uint32_t compressor_elements{};
    std::uint32_t index_compressor_elements{};
    const CudaWeight* output_a{};
    const CudaWeight* output_b{};
    const CudaWeight* router{};
    const CudaDeepSeekMoeExpert* shared{};
    const CudaDsv4MhcWeights* attention_mhc{};
    const CudaDsv4MhcWeights* ffn_mhc{};
    const CudaDsv4MhcWeights* next_attention_mhc{};
    std::span<const float> query_norm;
    std::span<const float> key_value_norm;
};

struct Dsv4RankLocalLayerWeights {
    std::array<Dsv4RankLocalLayerRankWeights, 2U> rank{};
    // Non-owning checkpoint storage; the executor never copies or owns it.
    std::span<const float> router_bias;
    // Layers 0..2 use the checkpoint token-to-expert row instead of a
    // selection bias. Exactly one routing membership span is populated.
    std::span<const std::uint32_t> router_token_experts;
};

struct Dsv4RankLocalLayerPagePatch {
    CudaDsv4AttentionPrepareHostCallback callback{};
    void* context{};
    std::span<const std::byte> ready_patch;
    std::span<const CudaDsv4AttentionPageWrite> writes;
};

// One indexed layer's in-chain sparse selection. When this is active the
// executor projects the index query, scores and selects, and resolves the
// compressed candidate region on each rank's own device between preparation
// and attention. Nothing crosses to the host, which is what lets an indexed
// layer stay in the queued chain.
//
// Both ranks select independently over their own replicated index pages rather
// than one rank selecting and broadcasting. The inputs are identical and the
// kernels are deterministic, so the two agree by construction, the duplicated
// work is concurrent on two devices, and no exchange enters the critical path.
struct Dsv4RankLocalLayerSelection {
    bool active{};
    std::array<const CudaWeight*, 2U> query_projection{};
    std::array<const CudaWeight*, 2U> weight_projection{};
    std::array<std::span<const CudaDsv4PhysicalIndexPage>, 2U> index_pages{};
    // Positional compressed block table; identical on both ranks because the
    // physical rows are replicated.
    std::span<const CudaDsv4KvBlockDescriptor> blocks;
    std::span<const float> rope_cosines;
    std::span<const float> rope_sines;
    std::uint32_t heads{};
    std::uint32_t head_dim{};
    std::uint32_t rope_dim{};
    std::uint32_t top_k{};
    std::uint32_t compressed_width{};
    float weight_scale{};
};

struct Dsv4RankLocalLayerCall {
    std::uint32_t layer{};
    std::uint32_t position{};
    Dsv4RankLocalLayerWeights weights{};
    std::span<const float> head_sinks;
    std::array<std::span<const CudaDsv4PhysicalPage>, 2U> pages{};
    std::span<const CudaDsv4AttentionCandidate> candidates;
    std::span<const float> inverse_rope_cosines;
    std::span<const float> inverse_rope_sines;
    Dsv4RankLocalLayerSelection selection{};
    std::array<Dsv4RankLocalLayerPagePatch, 2U> page_patches{};
    // The rank-1 preparation waits for rank 0's page callback and patch copies.
    // This is used only when rank 0 owns the canonical compressor/KV update and
    // rank 1 copies its already encoded bytes into the replica page.
    bool ordered_page_patches{};
    Dsv4RankLocalLayerObservation* observation{};
    bool terminal{};
};

enum class Dsv4RankLocalFailure : std::uint8_t {
    None,
    AttentionPreRank0,
    AttentionPreRank1,
    AttentionPostRank0,
    AttentionPostRank1,
    MoePreRank0,
    MoePreRank1,
    MoePostRank0,
    MoePostRank1,
    // Diagnostic-only exceptional path: rank 0 has been enqueued, while
    // rank 1's device-input metadata is still pending and must be cancelled.
    MoeBeforeEnqueueRank1,
};

struct Dsv4RankLocalLayerTiming {
    double total_ms{};
    double attention_ms{};
    double attention_collective_ms{};
    double attention_publication_ms{};
    double transition_router_ms{};
    double cpu_routed_rank0_ms{};
    double cpu_routed_rank1_ms{};
    double shared_gpu_rank0_ms{};
    double shared_gpu_rank1_ms{};
    double moe_collective_ms{};
    double moe_publication_ms{};
    double final_transition_ms{};
    // Wall time of the per-layer diagnostic boundary: one stream
    // synchronization per rank plus the state copies that follow it. Chain
    // mode defers this entire boundary to finish_chain(), so this is the cost
    // of driving the executor one layer at a time rather than queuing it.
    double diagnostic_boundary_ms{};
};

struct Dsv4RankLocalLayerResult {
    bool success{};
    std::uint32_t global_attention_status{};
    std::uint32_t global_moe_status{};
    std::array<std::uint32_t, 2U> attention_status{};
    std::array<std::uint32_t, 2U> moe_status{};
    std::array<std::array<float, kDsv4RankLocalHidden>, 2U> attention_reduction{};
    std::array<std::array<float, kDsv4RankLocalHidden>, 2U> moe_reduction{};
    std::array<std::array<float, kDsv4RankLocalHidden>, 2U> cpu_rank_partials{};
    std::array<std::array<std::uint16_t, kDsv4RankLocalHidden>, 2U> attention_publication{};
    std::array<std::array<std::uint16_t, kDsv4RankLocalHidden>, 2U> moe_publication{};
    std::array<std::array<float, kDsv4RankLocalHidden>, 2U> next_attention_input{};
    std::array<std::array<std::uint16_t, kDsv4RankLocalMhc>, 2U> outgoing_residual{};
    std::array<std::array<std::uint32_t, 6U>, 2U> routed_experts{};
    std::array<std::array<float, 6U>, 2U> routed_coefficients{};
    std::array<Dsv4HostMoePhaseTimings, 2U> cpu_moe_phases{};
    Dsv4RankLocalLayerTiming timing{};
    std::vector<std::string> errors;
};

struct Dsv4RankLocalLayerChainResult {
    std::size_t chain_count{};
    bool terminal{};
    std::uint32_t global_attention_status{};
    std::uint32_t global_moe_status{};
    std::array<std::uint32_t, 2U> attention_status{};
    std::array<std::uint32_t, 2U> moe_status{};
    std::array<std::array<std::uint16_t, kDsv4RankLocalHidden>, 2U>
        terminal_weighted{};
    std::array<std::array<std::uint16_t, kDsv4RankLocalHidden>, 2U>
        terminal_input{};
    std::array<std::array<float, kDsv4RankLocalFinalHidden>, 2U> final_hidden{};
    std::uint64_t terminal_logits_hash{};
    std::uint32_t next_token{};
    double terminal_head_ms{};
    // Sum of the exact routed CPU body over every queued layer, per rank.
    // The slower rank is the chain's routed-CPU critical term.
    std::array<Dsv4HostMoePhaseTimings, 2U> cpu_moe_phases{};
};

struct Dsv4RankLocalHeadRequest {
    std::array<const CudaWeight*, 2U> heads{};
    std::array<CudaDsv4MhcHeadCallback, 2U> callbacks{};
    std::array<void*, 2U> callback_contexts{};
    std::array<std::span<float>, 2U> local_logits{};
    std::array<std::span<std::uint16_t>, 2U> published_logits{};
};

struct Dsv4RankLocalLayerOptions {
    std::array<int, 2U> devices{};
    std::array<std::vector<int>, 2U> rank_cpus;
    const Dsv4ResidentWeightStore* resident{};
    // Optional static routed-expert tier on a device outside this pair. When
    // present, both ranks skip the triplets it holds -- it computes the whole
    // expert, not a shard -- and rank 0 alone adds its contribution in, so the
    // sum over the six slots is unchanged and nothing is counted twice.
    std::array<Dsv4StaticExpertTier*, 2U> static_expert_tiers{};
    std::uint32_t warmup_count{1U};
};

class Dsv4RankLocalLayerExecutor {
public:
    explicit Dsv4RankLocalLayerExecutor(CudaBackend& backend);
    ~Dsv4RankLocalLayerExecutor();

    Dsv4RankLocalLayerExecutor(const Dsv4RankLocalLayerExecutor&) = delete;
    Dsv4RankLocalLayerExecutor& operator=(const Dsv4RankLocalLayerExecutor&) = delete;

    [[nodiscard]] ValidationResult initialize(
        const Dsv4RankLocalLayerOptions& options);
    // One fixed pinned slot per queued layer. Rank 0 fills it in its canonical
    // callback and rank 1 uploads it after the executor's page-ready event.
    [[nodiscard]] ValidationResult replica_page_patch_staging(
        std::size_t slot, std::size_t bytes, std::span<std::byte>& output);

    // One complete layer. All device work is queued before the single final
    // diagnostic boundary. Failure cases still enter both data and U32 MAX
    // collectives and close the state machine with zeroed publication/state.
    [[nodiscard]] ValidationResult run(
        const Dsv4RankLocalLayerCall& call,
        Dsv4RankLocalFailure failure,
        Dsv4RankLocalLayerResult& result);

    // Queue one layer against the already-live mHC state. The optional
    // failure is diagnostic-only and is accepted only for the terminal layer.
    // borrowed call views must remain valid until finish_chain() or
    // abort_chain() returns.
    [[nodiscard]] ValidationResult enqueue_chain_layer(
        const Dsv4RankLocalLayerCall& call,
        Dsv4RankLocalFailure failure = Dsv4RankLocalFailure::None);
    // Drain all queued host callbacks and device work once. Without a head
    // request the terminal mHC state is returned directly; a head request
    // consumes it into rank-ordered published logits.
    [[nodiscard]] ValidationResult finish_chain(
        Dsv4RankLocalLayerChainResult* result = nullptr,
        const Dsv4RankLocalHeadRequest* head = nullptr);
    // Fail-closed cleanup for a partially queued chain; the executor may be
    // reseeded and reused after this returns.
    [[nodiscard]] ValidationResult abort_chain();

    [[nodiscard]] std::array<int, 2U> devices() const noexcept { return devices_; }

    // Public only so the CUDA host-node trampoline can name the opaque state;
    // callers never construct or inspect this implementation type.
    struct Impl;

private:
    CudaBackend& backend_;
    std::unique_ptr<Impl> impl_;
    std::array<int, 2U> devices_{};
};

}  // namespace strata
