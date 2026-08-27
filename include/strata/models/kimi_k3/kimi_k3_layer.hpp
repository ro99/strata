#pragma once

#include "strata/models/kimi_k3/kimi_k3_kv_cache.hpp"
#include "strata/models/kimi_k3/kimi_k3_ops.hpp"
#include "strata/models/common/model_adapter.hpp"
#include "strata/platform/result.hpp"
#include "strata/platform/worker_pool.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strata {

// One decoder layer of Kimi-K3, executed on the host over weights that stay in
// their checkpoint encoding.
//
// Weights are held as BF16 exactly as the shards store them and are widened
// during the multiply. That is not a numerical choice, it is a capacity one:
// the dense spine is 106.55 GiB in BF16 and 213 GiB in F32, and the host budget
// is 238 GiB with a 1.3 TiB routed set to cache in whatever is left.
//
// Activations are F32 throughout. The reference rounds them to BF16 at every op
// boundary, so this path is strictly more accurate, and the layer fixtures gate
// on relative error rather than on equality for that reason.

// A row-major `[rows, columns]` BF16 tensor borrowed from resident memory.
struct KimiBf16Matrix {
    std::span<const std::uint16_t> values;
    std::uint32_t rows{};
    std::uint32_t columns{};

    [[nodiscard]] bool valid() const noexcept {
        return values.size() ==
               static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns);
    }
};

// `output[t, r] = sum_c input[t, c] * weight[r, c]`, accumulated in F32.
// `pool` may be null, in which case the multiply runs on the calling thread.
[[nodiscard]] ValidationResult kimi_bf16_matmul(
    std::span<float> output, std::span<const float> input,
    const KimiBf16Matrix& weight, std::uint32_t tokens,
    HostWorkerPool* pool = nullptr);

struct KimiKdaWeights {
    KimiBf16Matrix q_proj;   // [heads * head_dim, hidden]
    KimiBf16Matrix k_proj;
    KimiBf16Matrix v_proj;
    KimiBf16Matrix g_proj;   // full-rank output gate
    KimiBf16Matrix o_proj;   // [hidden, heads * head_dim]
    KimiBf16Matrix f_a_proj;  // [head_dim, hidden]
    KimiBf16Matrix f_b_proj;  // [heads * head_dim, head_dim]
    KimiBf16Matrix b_proj;    // [heads, hidden]
    std::span<const float> q_conv;   // [heads * head_dim, kernel]
    std::span<const float> k_conv;
    std::span<const float> v_conv;
    // Per-head decay amplitude. The checkpoint zero-pads it to head_dim; only
    // the first `attention_heads` entries are live and this span holds those.
    std::span<const float> a_log;
    std::span<const float> dt_bias;  // [heads * head_dim]
    std::span<const float> o_norm;   // [head_dim]
};

struct KimiMlaWeights {
    KimiBf16Matrix q_a_proj;    // [query_lora_rank, hidden]
    KimiBf16Matrix q_b_proj;    // [heads * (nope + rope), query_lora_rank]
    KimiBf16Matrix kv_a_proj;   // [kv_lora_rank + rope, hidden]
    KimiBf16Matrix kv_b_proj;   // [heads * (nope + value), kv_lora_rank]
    KimiBf16Matrix o_proj;      // [hidden, heads * value_head_dim]
    KimiBf16Matrix g_proj;      // full-rank output gate
    std::span<const float> q_a_norm;
    std::span<const float> kv_a_norm;
};

struct KimiMoeWeights {
    KimiBf16Matrix router;            // [experts, hidden]
    std::span<const float> router_bias;
    KimiBf16Matrix latent_down;       // [routed_expert_hidden, hidden]
    KimiBf16Matrix latent_up;         // [hidden, routed_expert_hidden]
    std::span<const float> latent_norm;
    KimiBf16Matrix shared_gate;       // [shared * expert_intermediate, hidden]
    KimiBf16Matrix shared_up;
    KimiBf16Matrix shared_down;
};

struct KimiDenseMlpWeights {
    KimiBf16Matrix gate;
    KimiBf16Matrix up;
    KimiBf16Matrix down;
};

struct KimiLayerWeights {
    std::span<const float> input_norm;
    std::span<const float> post_attention_norm;
    std::span<const float> attention_res_norm;
    std::span<const float> attention_res_proj;
    std::span<const float> mlp_res_norm;
    std::span<const float> mlp_res_proj;
    KimiKdaWeights kda;
    KimiMlaWeights mla;
    KimiMoeWeights moe;
    KimiDenseMlpWeights dense;
};

// One routed expert module in its MXFP4 checkpoint encoding: two E2M1 elements
// per packed byte, low nibble first, with one E8M0 scale per group of 32.
//
// The multiply reads this form directly. Dequantizing an expert into F32 first
// would materialize 132 MiB per expert, and at sixteen experts across 92 MoE
// layers that is 194 GiB of extra memory traffic per token — the same order as
// the storage term the whole design is bounded by. Decoding a nibble inside the
// inner loop costs a shift and a table lookup and moves nothing.
struct KimiExpertModuleView {
    std::span<const std::uint8_t> packed;  // [rows, columns / 2]
    std::span<const std::uint8_t> scales;  // [rows, columns / 32]
    std::uint32_t rows{};
    std::uint32_t columns{};

    [[nodiscard]] bool valid() const noexcept;
};

// One routed expert's three modules in the order the LatentMoE block uses them.
struct KimiExpertWeights {
    KimiExpertModuleView gate;  // [expert_intermediate, routed_expert_hidden]
    KimiExpertModuleView up;
    KimiExpertModuleView down;  // [routed_expert_hidden, expert_intermediate]
};

// `output[r] = sum_c input[c] * dequantize(module)[r, c]`, decoded in place.
[[nodiscard]] ValidationResult kimi_mxfp4_matvec(
    std::span<float> output, std::span<const float> input,
    const KimiExpertModuleView& module, HostWorkerPool* pool = nullptr);

// Supplies the experts one MoE block needs. The runtime implements this over
// the arena and the reader; a fixture implements it over a small map. Selection
// happens before any expert is requested, which is what lets the runtime issue
// all sixteen reads at queue depth rather than one at a time.
class KimiExpertSource {
public:
    virtual ~KimiExpertSource() = default;
    // Reports the deduplicated routed set for one layer. `prepare` already
    // receives exactly that set, so the default does nothing and only an
    // implementation that wants to observe routing overrides it.
    virtual void observe(std::uint32_t /*layer*/,
                         std::span<const std::uint32_t> /*experts*/) {}
    // Called once per MoE block with every expert the block will read, so the
    // implementation can stage them together.
    [[nodiscard]] virtual ValidationResult prepare(
        std::uint32_t layer, std::span<const std::uint32_t> experts) = 0;
    [[nodiscard]] virtual ValidationResult fetch(std::uint32_t layer,
                                                 std::uint32_t expert,
                                                 KimiExpertWeights& weights) = 0;
};

// The per-token attention-residual state for a whole page of tokens.
//
// Kimi-3 opens a block every `attention_residual_block_size` layers and selects
// over the completed blocks plus the running prefix instead of adding into one
// residual stream. Over 93 layers that is 8 blocks and one prefix: nine sources
// per token, independent of depth and of context length.
class KimiResidualStream {
public:
    [[nodiscard]] ValidationResult reset(std::uint32_t tokens,
                                         std::uint32_t hidden_size,
                                         std::uint32_t block_size);
    // Seeds the prefix with the token embeddings.
    [[nodiscard]] ValidationResult begin(std::span<const float> embeddings);

    // Mixes each token over its completed blocks and its prefix. With no
    // completed block the prefix passes through, which the reference does by
    // skipping the call at the attention site of layer 0.
    [[nodiscard]] ValidationResult mix(std::span<float> output,
                                       std::span<const float> query_weight,
                                       std::span<const float> norm_weight,
                                       float epsilon,
                                       HostWorkerPool* pool = nullptr) const;
    // Closes the current prefix into a block and restarts the prefix at zero.
    [[nodiscard]] ValidationResult open_block();
    // Folds a sublayer output into the running prefix.
    [[nodiscard]] ValidationResult add(std::span<const float> delta);

    [[nodiscard]] std::span<const float> prefix() const noexcept { return prefix_; }
    [[nodiscard]] std::uint32_t completed_blocks() const noexcept {
        return completed_blocks_;
    }
    [[nodiscard]] std::uint32_t tokens() const noexcept { return tokens_; }

private:
    std::uint32_t tokens_{};
    std::uint32_t hidden_size_{};
    std::uint32_t block_size_{};
    std::uint32_t completed_blocks_{};
    std::vector<float> blocks_;   // [block][token][hidden]
    std::vector<float> prefix_;   // [token][hidden]
};

// Scratch buffers a layer needs, kept across layers and tokens so a 93-layer
// step does not allocate 93 times.
struct KimiLayerScratch {
    std::vector<float> normalized;
    std::vector<float> attention;
    std::vector<float> feedforward;
    std::vector<float> projection_a;
    std::vector<float> projection_b;
    std::vector<float> projection_c;
    std::vector<float> projection_d;
    std::vector<float> heads;
    std::vector<float> latent;
    std::vector<float> latent_mix;
    std::vector<float> expert_gate;
    std::vector<float> expert_up;
    std::vector<float> expert_out;
    std::vector<float> router_logits;
    std::vector<KimiRoutedExpert> selection;
    // One gate/up/activated/output block per worker, so the experts a worker
    // owns do not share scratch with any other worker's.
    std::vector<float> worker_workspace;
    // One `[tokens, routed_expert_hidden]` accumulator per worker. Routed
    // experts are summed into the same latent mixture, so each worker
    // accumulates its own share and the shares are reduced in worker order
    // afterwards -- which keeps the result independent of how the pool
    // happened to schedule them.
    std::vector<float> worker_mixture;

    // Per-phase nanoseconds, accumulated across layers and pages.
    //
    // The charter's first step is to emit the per-phase breakdown of a step and
    // name `argmax_r`. That was done for the routed and dense matvecs in
    // isolation and skipped for the step itself, and the two disagreed by 7.5x:
    // a measured 38.6 s step against a 5.2 s sum of the two components anyone
    // had profiled. These say where the rest is.
    std::uint64_t residual_mix_ns{};
    std::uint64_t attention_ns{};
    std::uint64_t feedforward_ns{};
};

// Kimi Delta Attention over `tokens` positions starting at `position`.
// Advances the layer's recurrent state and convolution history in `cache`.
[[nodiscard]] ValidationResult kimi_kda_layer(
    std::span<float> output, std::span<const float> input,
    const KimiKdaWeights& weights, KimiStateCache& cache, std::uint32_t layer,
    std::uint32_t tokens, KimiLayerScratch& scratch,
    HostWorkerPool* pool = nullptr);

// Gated MLA with NoPE. Appends this page's latents to `cache` and attends over
// every committed row plus this page.
[[nodiscard]] ValidationResult kimi_mla_layer(
    std::span<float> output, std::span<const float> input,
    const KimiMlaWeights& weights, KimiStateCache& cache, std::uint32_t layer,
    std::uint32_t position, std::uint32_t tokens, KimiLayerScratch& scratch,
    HostWorkerPool* pool = nullptr);

// Stable LatentMoE: route on the raw hidden state, project into the 3584-wide
// latent, run the selected experts there, normalize the mixture, project back,
// then add the shared experts computed on the raw hidden state.
[[nodiscard]] ValidationResult kimi_latent_moe_layer(
    std::span<float> output, std::span<const float> input,
    const KimiMoeWeights& weights, KimiExpertSource& experts,
    std::uint32_t layer, std::uint32_t tokens, KimiLayerScratch& scratch,
    HostWorkerPool* pool = nullptr);

// The dense SiTU-GLU MLP that replaces the MoE block on the prefix layers.
[[nodiscard]] ValidationResult kimi_dense_mlp_layer(
    std::span<float> output, std::span<const float> input,
    const KimiDenseMlpWeights& weights, std::uint32_t tokens,
    KimiLayerScratch& scratch, HostWorkerPool* pool = nullptr);

// The whole layer, in the reference's block-residual order.
[[nodiscard]] ValidationResult kimi_decoder_layer(
    KimiResidualStream& stream, const KimiLayerWeights& weights,
    KimiStateCache& cache, KimiExpertSource& experts, std::uint32_t layer,
    std::uint32_t position, std::uint32_t tokens, KimiLayerScratch& scratch,
    HostWorkerPool* pool = nullptr);

}  // namespace strata
