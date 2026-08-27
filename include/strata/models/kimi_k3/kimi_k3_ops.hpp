#pragma once

#include "strata/models/common/model_adapter.hpp"
#include "strata/platform/result.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strata {

// Every operation below computes in F32 and matches the checkpoint's own
// reference implementation, which upcasts to F32 for the same reasons. The
// reductions are written in the reference's association order so a fixture can
// compare exactly rather than within a tolerance chosen to hide a reordering.

[[nodiscard]] ValidationResult kimi_rms_norm(
    std::span<float> output, std::span<const float> input,
    std::span<const float> weight,
    float epsilon = kKimiK3ExecutionContract.rms_epsilon);

// SiTU-GLU: [b1 tanh(g/b1) sigmoid(g)] * [b2 tanh(u/b2)].
// Both factors of the first bracket read the gate projection. This is not
// SwiGLU with different constants, and substituting one for the other would
// change expert semantics silently.
[[nodiscard]] ValidationResult kimi_situ_glu(
    std::span<float> output, std::span<const float> gate,
    std::span<const float> up,
    float gate_beta = kKimiK3ExecutionContract.situ_gate_beta,
    float linear_beta = kKimiK3ExecutionContract.situ_linear_beta);

// L2 normalization with the epsilon inside the square root, as the KDA kernel
// applies it: x / sqrt(sum(x^2) + eps).
[[nodiscard]] ValidationResult kimi_l2_normalize(std::span<float> values,
                                                 float epsilon = 1.0e-6F);

// Causal depthwise convolution of width `kernel` followed by SiLU, over one
// channel block. `history` holds the `kernel - 1` previous inputs per channel
// and is advanced in place, so decode and prefill share one definition.
[[nodiscard]] ValidationResult kimi_short_conv_step(
    std::span<float> output, std::span<const float> input,
    std::span<const float> weight, std::span<float> history,
    std::uint32_t kernel = kKimiK3ExecutionContract.short_conv_kernel);

// One KDA head's log decay for one token:
//   g_k = lower_bound * sigmoid(exp(A_log) * (z_k + dt_bias_k)),  alpha = exp(g)
// Kimi-3's mapping, not Kimi-Linear's negative softplus. Bounding g to
// [lower_bound, 0) is what keeps the chunkwise form's causal tile finite.
// `a_log` is the head's scalar; the checkpoint stores the per-head vector
// zero-padded to head_dim and only the first `attention_heads` entries are live.
[[nodiscard]] ValidationResult kimi_kda_log_decay(
    std::span<float> logarithm, std::span<const float> logits,
    std::span<const float> dt_bias, float a_log,
    float lower_bound = kKimiK3ExecutionContract.kda_gate_lower_bound);

// One head's delta-rule step, in the transposed [value, key] state layout the
// reference kernels use:
//   S <- S diag(alpha);  v' <- beta (v - S k);  S <- S + v' k^T;  o <- S q
// `state` is `value_dim * key_dim` and is updated in place. `query` is expected
// already L2-normalized and scaled; `key` already L2-normalized.
[[nodiscard]] ValidationResult kimi_kda_step(
    std::span<float> output, std::span<float> state, std::span<const float> query,
    std::span<const float> key, std::span<const float> value,
    std::span<const float> decay, float beta, std::uint32_t key_dim,
    std::uint32_t value_dim);

// Head-wise gated RMSNorm on a KDA head's output:
//   y = (o * rsqrt(mean(o^2) + eps) * weight) * sigmoid(g)
// The gate multiplies after the norm and its weight, which is what
// FusedRMSNormGated(activation="sigmoid") computes.
[[nodiscard]] ValidationResult kimi_kda_output_norm(
    std::span<float> output, std::span<const float> input,
    std::span<const float> gate, std::span<const float> weight,
    float epsilon = kKimiK3ExecutionContract.rms_epsilon);

// Router: s = sigmoid(logits); select top-k of (s + bias); renormalize the
// selected s over their own sum. The frozen Quantile-Balancing bias steers
// selection only and is excluded from the weights. Ties break toward the lower
// expert index so a run is reproducible.
struct KimiRoutedExpert {
    std::uint32_t expert{};
    float weight{};
};

[[nodiscard]] ValidationResult kimi_route_topk(
    std::span<KimiRoutedExpert> selected, std::span<const float> logits,
    std::span<const float> selection_bias,
    float routed_scale = kKimiK3ExecutionContract.routed_scale);

// Attention residuals. Each site selects over prior depth instead of adding to
// one residual stream:
//   score_i = sum_c RMSNorm(v_i)_c * (norm_weight_c * query_weight_c)
//   h       = sum_i softmax(score)_i * v_i
// The scores are computed on the normalized sources; the mixture is over the
// raw ones. The two weight vectors multiply into one, which is how the
// reference forms `score_weight`.
//
// `sources` is `count` contiguous vectors of `hidden_size` values.
[[nodiscard]] ValidationResult kimi_attention_residual_mix(
    std::span<float> output, std::span<const float> sources,
    std::span<const float> query_weight, std::span<const float> norm_weight,
    std::uint32_t hidden_size, std::uint32_t count,
    float epsilon = kKimiK3ExecutionContract.rms_epsilon);

// The inference-time state the block form needs: one vector per completed
// block plus the running prefix sum of the block in progress. Kimi-3 opens a
// block every `attn_res_block_size` layers, so 93 layers leave 8 block vectors
// and one prefix — nine sources of 7168 values, independent of depth and of
// context length.
class KimiAttentionResidualState {
public:
    KimiAttentionResidualState() = default;
    [[nodiscard]] ValidationResult reset(std::uint32_t hidden_size,
                                         std::uint32_t block_size);

    // Seeds the prefix with the token embedding. Layer 0 immediately opens a
    // block from it, which is why source 0 is the embedding.
    [[nodiscard]] ValidationResult begin(std::span<const float> embedding);

    // Mixes over the completed blocks and the current prefix. With no completed
    // block — only true at the attention site of layer 0 — there is nothing to
    // select over and the prefix passes through, as the reference does by
    // skipping the call.
    [[nodiscard]] ValidationResult mix(std::span<float> output,
                                       std::span<const float> query_weight,
                                       std::span<const float> norm_weight,
                                       float epsilon) const;
    // Closes the current prefix into a block. The next `add` starts a new one.
    [[nodiscard]] ValidationResult open_block();
    // Folds a sublayer output into the running prefix.
    [[nodiscard]] ValidationResult add(std::span<const float> delta);

    [[nodiscard]] std::span<const float> prefix() const noexcept;
    [[nodiscard]] std::uint32_t completed_blocks() const noexcept {
        return completed_blocks_;
    }
    [[nodiscard]] std::uint32_t source_count() const noexcept {
        return completed_blocks_ + 1U;
    }

private:
    std::uint32_t hidden_size_{};
    std::uint32_t block_size_{};
    std::uint32_t completed_blocks_{};
    bool has_prefix_{};
    // Completed blocks followed by the prefix, contiguous so `mix` reads one
    // span and the mixture is a single pass.
    std::vector<float> sources_;
};

// Chunkwise KDA over one head, the prefill form of `kimi_kda_step`. Consumes
// `tokens` rows of already-normalized q and k, advances `state` once for the
// whole chunk, and writes `tokens` outputs.
//
// The intra-chunk terms are formed from differences of the cumulative log
// decay rather than from a ratio of cumulative products: for j <= t the
// exponent is non-positive by construction, so the dense causal tile stays
// finite where `k / Gamma` would overflow within a few tokens at this decay
// bound.
[[nodiscard]] ValidationResult kimi_kda_chunk(
    std::span<float> output, std::span<float> state, std::span<const float> query,
    std::span<const float> key, std::span<const float> value,
    std::span<const float> decay_logarithm, std::span<const float> beta,
    std::uint32_t tokens, std::uint32_t key_dim, std::uint32_t value_dim);

}  // namespace strata
