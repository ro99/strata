#pragma once

#include "strata/device/cuda_backend.hpp"
#include "strata/engine/chat_protocol.hpp"
#include "strata/engine/sampling.hpp"
#include "strata/platform/result.hpp"
#include "strata/platform/types.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strata {

// Deterministic weighted assignment used by both warmup and execution. The
// same projection key therefore has exactly one CUDA home, while independent
// projections spread according to discovered cache capacity.
[[nodiscard]] std::vector<std::size_t> glm53_projection_slots(
    std::span<const std::string_view> keys,
    std::span<const std::uint64_t> costs,
    std::span<const std::uint64_t> capacities,
    std::size_t preferred_slot);

// Host decoders for the expert storage formats the two FP4 releases add.
// Exposed so the vectorized and scalar decoders can be compared against each
// other and against a reference dequantization; the host MoE calls exactly
// these functions. `use_avx2` false forces the scalar reference.
//
// MXFP4: `packed` holds one row's E2M1 nibbles, column 2b in the low nibble of
// byte b, and `scales` one E8M0 byte per 32 columns of that row.
[[nodiscard]] float glm53_host_fp4_group32_row_dot(
    std::span<const std::byte> packed, std::span<const std::byte> scales,
    std::span<const float> input, bool use_avx2) noexcept;
// NVFP4: the same nibble packing, but `scales` holds one E4M3 byte per 16
// columns and every group scale is divided by `global_scale` before it is
// applied. The division is part of the weight, not of the sum, and it comes
// first: that is the order the reference dequantization and the device kernel
// both use.
[[nodiscard]] float glm53_host_nvfp4_group16_row_dot(
    std::span<const std::byte> packed, std::span<const std::byte> scales,
    float global_scale, std::span<const float> input, bool use_avx2) noexcept;
// FP8: `weights` holds one row's E4M3 codes and `scales` the F32 inverse
// scales for that row's 128-column blocks. Exposed alongside the other two so
// the three formats can be priced against each other on the same input.
[[nodiscard]] float glm53_host_fp8_block128_row_dot(
    std::span<const std::byte> weights, std::span<const float> scales,
    std::span<const float> input, bool use_avx2) noexcept;
// BF16 rows carry no scale.
[[nodiscard]] float glm53_host_bf16_row_dot(
    std::span<const std::byte> weights, std::span<const float> input,
    bool use_avx2) noexcept;

// GLM-5.3 k-pool sparse indexer selection (record 0237). Chooses which history
// positions one decode query attends to, so attention cost stops growing with
// context. Exposed for test: below `index_topk` the selection is the identity,
// which is the regression gate for everything above it.
struct Glm53SparseIndexParameters {
    static constexpr std::uint32_t heads = 32U;
    static constexpr std::uint32_t head_dim = 128U;
    static constexpr std::uint32_t top_k = 2048U;
    static constexpr std::uint32_t pool = 4U;
    static constexpr std::uint32_t selection_width = top_k + pool - 1U;
};

[[nodiscard]] std::size_t glm53_sparse_index_select_for_test(
    std::span<std::uint32_t> selected, std::span<const float> indexer_query,
    std::span<const float> indexer_keys, std::span<const float> gate_scores,
    std::span<const float> pool_ape, std::span<const float> head_weights,
    std::uint32_t history);

// The two projections the selection is fed from, exposed so the reference
// oracle can drive the whole indexer chain rather than only its ranking. The
// key norm is `nn.LayerNorm(head_dim, eps=1e-6)` -- mean subtracting, with a
// bias -- and not the RMSNorm this model uses everywhere else, which is a
// difference no gate at or below `index_topk` can see.
void glm53_indexer_gate_for_test(std::span<float> output,
                                 std::span<const float> input,
                                 std::span<const float> weight) noexcept;
void glm53_indexer_layer_norm_for_test(std::span<float> values,
                                       std::span<const float> weight,
                                       std::span<const float> bias) noexcept;

// The per-device MLA activation workspace the runtime reserves for an admitted
// context, in bytes. The dense path expands the whole visible history, so its
// widest call scales with the context; the k-pool indexer expands a bounded
// selection instead, so above `top_k` the reservation is flat. Reserving the
// dense extent for a sparse context is 4 GiB per device at 32,768 and 32 GiB at
// the admitted ceiling, which is memory the weight arena, the sequence state
// and the mHC workspace then have to do without.
struct Glm53MlaWorkspaceBytes {
    std::uint64_t input{};
    std::uint64_t output{};

    [[nodiscard]] std::uint64_t total() const noexcept {
        return input + output;
    }
};

[[nodiscard]] Glm53MlaWorkspaceBytes glm53_mla_workspace_bytes(
    std::uint32_t maximum_context_tokens,
    std::uint32_t prefill_page_tokens) noexcept;

struct Glm53RuntimeConfig {
    std::vector<int> devices;
    double vram_cache_fraction{0.85};
    std::uint32_t maximum_context_tokens{2048U};
    double sampling_temperature{};
    std::uint64_t sampling_seed{33'377'335U};
    // Upper bound for a prefill scheduler page. The default preserves the
    // production path; experiments may override it to measure weight reuse.
    std::uint32_t prefill_page_tokens{64U};
    // Build prompt MLA and KDA state on the device and back-fill the host
    // cache from it, instead of prefilling on the host. Worth about 1.38x at
    // 619 tokens, almost entirely by moving KDA off the host (records 0240,
    // 0242). It is silently unavailable above kIndexTopK: the device chain
    // computes no indexer k-pool state, so a sequence that can cross that
    // threshold must prefill on the host. `STRATA_GLM53_DEVICE_PREFILL` also
    // sets it, and the two are OR-ed at initialization.
    bool device_prefill{};
    bool verbose{};
    bool load_progress{};
    // Opt-in request attribution. CUDA event timing is enabled only when this
    // is true; ordinary production execution retains its existing timing path.
    bool phase_profile{};
};

struct Glm53CacheMetrics {
    std::uint64_t hits{};
    std::uint64_t misses{};
    std::uint64_t evictions{};
    std::uint64_t prefetches{};
    std::uint64_t useful_prefetches{};
    std::uint64_t failed_prefetches{};
};

struct Glm53HostExpertMetrics {
    std::uint64_t calls{};
    std::uint64_t rows{};
    std::uint64_t group_windows{};
    std::uint64_t dispatch_nanoseconds{};
    std::uint64_t staging_nanoseconds{};
    // backend.upload() time inside load_cuda_linear, phased by delta. In
    // prefill this is the staging transfer path (memcpy + enqueue + ring
    // waits + arena alloc); staging_nanoseconds minus this is host-side
    // bookkeeping (validation, maps, eviction, describe, leases).
    std::uint64_t staging_upload_nanoseconds{};
    std::uint64_t gate_up_weight_bytes{};
    std::uint64_t down_weight_bytes{};
    std::uint64_t view_resolution_nanoseconds{};
    std::uint64_t input_quantization_nanoseconds{};
    std::uint64_t gate_up_nanoseconds{};
    std::uint64_t activation_nanoseconds{};
    std::uint64_t down_nanoseconds{};
    std::uint64_t reduction_nanoseconds{};
    std::uint64_t service_nanoseconds{};
    std::uint64_t temporary_allocation_calls{};
};

struct Glm53GraphMetrics {
    std::uint64_t forward_calls{};
    std::uint64_t forward_rows{};
    std::uint64_t embedding_nanoseconds{};
    std::uint64_t layer_nanoseconds{};
    std::uint64_t attention_block_nanoseconds{};
    std::uint64_t kda_nanoseconds{};
    std::uint64_t mla_nanoseconds{};
    std::uint64_t feedforward_block_nanoseconds{};
    std::uint64_t output_head_nanoseconds{};
    std::uint64_t sampling_nanoseconds{};
};

// Wall-clock attribution inside sparse resident MLA. CUDA kernel event time
// remains in CudaBackendStats; the device-scores wait deliberately includes
// queued device work because it measures the serialization boundary seen by
// the host thread.
struct Glm53SparseMlaMetrics {
    std::uint64_t calls{};
    std::uint64_t input_download_nanoseconds{};
    std::uint64_t indexer_projection_nanoseconds{};
    std::uint64_t indexer_state_nanoseconds{};
    std::uint64_t query_rank_projection_nanoseconds{};
    std::uint64_t pool_scoring_nanoseconds{};
    std::uint64_t topk_sort_nanoseconds{};
    std::uint64_t arena_bookkeeping_nanoseconds{};
    std::uint64_t index_upload_nanoseconds{};
    std::uint64_t device_scores_wait_nanoseconds{};
    std::uint64_t host_softmax_nanoseconds{};
    std::uint64_t coefficient_upload_nanoseconds{};
};

struct Glm53PhaseMetrics {
    CudaBackendStats cuda;
    Glm53CacheMetrics cache;
    Glm53HostExpertMetrics host_experts;
    Glm53GraphMetrics graph;
    Glm53SparseMlaMetrics sparse_mla;
};

struct Glm53RunMetrics {
    std::uint64_t prompt_tokens{};
    std::uint64_t prefill_tokens{};
    std::uint64_t reused_prompt_tokens{};
    std::uint64_t decode_tokens{};
    double prefill_seconds{};
    double decode_seconds{};
    double decode_prepare_seconds{};
    Glm53PhaseMetrics prefill;
    Glm53PhaseMetrics decode;
    std::uint64_t rss_bytes{};
    std::vector<std::uint64_t> device_vram_used_bytes;
    bool phase_profile{};
};

struct Glm53GenerationResult {
    std::string text;
    std::vector<std::uint32_t> prompt_token_ids;
    std::vector<std::uint32_t> generated_token_ids;
    std::vector<TokenLogprob> logprobs;
    Glm53RunMetrics metrics;
    std::vector<std::string> errors;
    bool stopped{};

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

// One layer-page of sparse MLA attention at an arbitrary history, on
// fabricated cache state (record 0269). A full 8,192-token prefill is twenty
// minutes and 32,768 is hours, so the regime this campaign exists to improve
// has never once been measured directly -- every arm read the shape it could
// afford rather than the shape the objective names. This reaches any history
// in seconds because it synthesizes the latent and indexer caches instead of
// computing them. The weights and the control flow are the production ones,
// so the timing split and the group structure are real; the activations are
// LCG noise, so the attention OUTPUT is meaningless and `checksum` is only a
// same-input/same-output gate between two builds of the same shape.
//
// Selection content is data-dependent, and noise selects more uniformly than
// a trained indexer does -- real fragmentation is layer-dependent (0268:
// layers 3/7/11 never split, 19/23 shatter). So the bench brackets the
// fragmented case rather than reproducing a given layer's, and its group
// counts must be calibrated against a real run at a shape both can reach
// before its numbers at 32,768 are believed.
struct Glm53AttentionBenchRequest {
    std::uint32_t layer{};
    // Tokens already resident when the page runs; `history_begin` in the
    // attention path. history + rows must exceed the indexer threshold or
    // the dense path runs instead and the bench measures nothing.
    std::uint32_t history{};
    std::uint32_t rows{};
    // Passes over the same fabricated state. The FIRST pass also builds the
    // layer's k-pool keys, which every later pass then finds cached, so a
    // repeated run reports a mean over one cold pass and n-1 warm ones. Use 1
    // to measure a page as production sees it mid-prompt.
    std::uint32_t repeats{1U};
    std::uint64_t seed{0x9E3779B97F4A7C15ULL};
    // How strongly a token's fabricated indexer key follows the previous
    // token's, in [0, 1]. 0 is independent noise, which fragments far worse
    // than any real layer; raise it until the reported group count matches a
    // shape a real arm has measured, and only then read the sweep.
    float smoothing{};
};

// Per-pass means. The names match the `[glm53-mla-split]` fields so a bench
// line and an arm line can be read side by side.
struct Glm53AttentionBenchResult {
    std::uint64_t wall_nanoseconds{};
    std::uint64_t prelude_nanoseconds{};
    std::uint64_t groups_nanoseconds{};
    std::uint64_t index_nanoseconds{};
    std::uint64_t device_nanoseconds{};
    std::uint64_t expand_nanoseconds{};
    std::uint64_t qk_nanoseconds{};
    std::uint64_t softmax_nanoseconds{};
    std::uint64_t av_nanoseconds{};
    std::uint64_t checksum{};
};

class Glm53Runtime {
public:
    Glm53Runtime();
    ~Glm53Runtime();
    Glm53Runtime(Glm53Runtime&&) noexcept;
    Glm53Runtime& operator=(Glm53Runtime&&) noexcept;
    Glm53Runtime(const Glm53Runtime&) = delete;
    Glm53Runtime& operator=(const Glm53Runtime&) = delete;

    [[nodiscard]] ValidationResult initialize(
        const std::string& model_directory,
        const Glm53RuntimeConfig& config = {});
    // `reasoning_effort` is the budget the checkpoint's chat template accepts:
    // "low", "high", or "max". The template silently falls back to "max" for
    // anything else, so this rejects an unrecognized value rather than running
    // the most verbose setting under a name that asked for the least. There is
    // no value that turns reasoning off: GLM-5.3's template opens a <think>
    // block unconditionally and ships no enable_thinking toggle.
    // See `Glm53AttentionBenchRequest`. Requires an initialized runtime; it
    // builds its own sequence state and leaves the runtime's alone.
    [[nodiscard]] ValidationResult attention_bench(
        const Glm53AttentionBenchRequest& request,
        Glm53AttentionBenchResult& result);
    [[nodiscard]] Glm53GenerationResult generate_chat_stream(
        std::span<const ChatMessage> messages,
        std::uint32_t maximum_new_tokens, const SamplingOptions& sampling,
        std::span<const std::string> stop,
        std::string_view reasoning_effort = "max",
        const TokenStreamCallback& on_token = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata
