#pragma once

#include <array>
#include <cstdint>

namespace strata {

struct Glm52ExecutionContract {
    std::uint32_t hidden_size;
    std::uint32_t layer_count;
    std::uint32_t attention_heads;
    std::uint32_t query_lora_rank;
    std::uint32_t kv_lora_rank;
    std::uint32_t nope_head_dim;
    std::uint32_t rope_head_dim;
    std::uint32_t value_head_dim;
    std::uint32_t dense_intermediate_size;
    std::uint32_t expert_intermediate_size;
    std::uint32_t routed_experts;
    std::uint32_t experts_per_token;
    std::uint32_t vocabulary_size;
    std::uint32_t dense_prefix_layers;
    std::uint32_t sparse_attention_topk;
    std::uint32_t maximum_context_tokens;
    float attention_scale;
    float routed_scale;
};

inline constexpr Glm52ExecutionContract kGlm52ExecutionContract{
    6144U, 78U, 64U, 2048U, 512U, 192U, 64U, 256U,
    12288U, 2048U, 256U, 8U, 154880U, 3U, 2048U, 1'048'576U,
    1.0F / 16.0F, 2.5F};

struct DeepSeekV4ExecutionContract {
    std::uint32_t hidden_size;
    std::uint32_t layer_count;
    std::uint32_t attention_heads;
    std::uint32_t key_value_heads;
    std::uint32_t head_dim;
    std::uint32_t rope_head_dim;
    std::uint32_t query_lora_rank;
    std::uint32_t output_lora_rank;
    std::uint32_t output_groups;
    std::uint32_t sliding_window;
    std::uint32_t index_heads;
    std::uint32_t index_head_dim;
    std::uint32_t index_topk;
    std::uint32_t routed_experts;
    std::uint32_t experts_per_token;
    std::uint32_t expert_intermediate_size;
    std::uint32_t vocabulary_size;
    std::uint32_t mhc_multiplier;
    std::uint32_t mhc_sinkhorn_iterations;
    std::uint32_t mix_width;
    std::uint32_t maximum_context_tokens;
    float rms_epsilon;
    float routed_scale;
    float swiglu_limit;
    std::array<std::uint32_t, 46> compression_ratios;
};

inline constexpr DeepSeekV4ExecutionContract kDeepSeekV4ExecutionContract{
    4096U, 43U, 64U, 1U, 512U, 64U, 1024U, 1024U, 8U, 128U,
    64U, 128U, 512U, 256U, 6U, 2048U, 129280U, 4U, 20U, 24U,
    1'048'576U, 1.0e-6F, 1.5F, 10.0F,
    {0U, 0U, 4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U,
     4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U,
     4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U,
     4U, 128U, 4U, 128U, 4U, 128U, 4U, 0U, 0U, 0U}};

struct Gemma4ExecutionContract {
    std::uint32_t hidden_size;
    std::uint32_t layer_count;
    std::uint32_t attention_heads;
    std::uint32_t local_key_value_heads;
    std::uint32_t global_key_value_heads;
    std::uint32_t local_head_dim;
    std::uint32_t global_head_dim;
    std::uint32_t intermediate_size;
    std::uint32_t vocabulary_size;
    std::uint32_t sliding_window;
    std::uint32_t maximum_context_tokens;
    std::uint32_t vision_hidden_size;
    std::uint32_t vision_layer_count;
    std::uint32_t vision_attention_heads;
    std::uint32_t vision_head_dim;
    std::uint32_t vision_intermediate_size;
    std::uint32_t vision_patch_size;
    std::uint32_t vision_position_embeddings;
    std::uint32_t vision_pooling_kernel;
    std::uint32_t default_image_tokens;
    float rms_epsilon;
    float local_rope_theta;
    float global_rope_theta;
    float global_rope_proportion;
    float vision_rope_theta;
    float final_logit_softcap;
};

inline constexpr Gemma4ExecutionContract kGemma4ExecutionContract{
    5376U, 60U, 32U, 16U, 4U, 256U, 512U, 21504U, 262144U,
    1024U, 262144U, 1152U, 27U, 16U, 72U, 4304U, 16U, 10240U,
    3U, 280U, 1.0e-6F, 10000.0F, 1000000.0F, 0.25F, 100.0F, 30.0F};

[[nodiscard]] constexpr bool gemma4_global_attention_layer(
    std::uint32_t layer) noexcept {
    return layer < kGemma4ExecutionContract.layer_count && (layer + 1U) % 6U == 0U;
}

// Laguna S 2.1-NVFP4 (poolside). 48 layers in a 1:3 global/sliding pattern with
// per-layer head counts, softplus per-head output gating, QK RMSNorm, and a
// sigmoid-routed 256-expert MoE with one shared expert. Layer 0 is dense.
// Routed experts are NVFP4 (E2M1 pairs, group-16 E4M3 scales, per-tensor FP32
// global scale) for layers 1..39 and plain BF16 for layers 40..47; every other
// module in the checkpoint is BF16.
struct LagunaExecutionContract {
    std::uint32_t hidden_size;
    std::uint32_t layer_count;
    std::uint32_t global_attention_stride;
    std::uint32_t global_attention_heads;
    std::uint32_t sliding_attention_heads;
    std::uint32_t key_value_heads;
    std::uint32_t head_dim;
    std::uint32_t sliding_window;
    std::uint32_t dense_intermediate_size;
    std::uint32_t expert_intermediate_size;
    std::uint32_t shared_expert_intermediate_size;
    std::uint32_t routed_experts;
    std::uint32_t experts_per_token;
    std::uint32_t dense_prefix_layers;
    std::uint32_t quantized_expert_layers;
    std::uint32_t vocabulary_size;
    std::uint32_t maximum_context_tokens;
    std::uint32_t nvfp4_group_size;
    float rms_epsilon;
    float routed_scale;
    float router_logit_softcapping;
    // Full-attention layers: YaRN over the first half of each head.
    float global_rope_theta;
    float global_rope_factor;
    float global_rope_attention_factor;
    float global_rope_beta_fast;
    float global_rope_beta_slow;
    float global_rope_partial;
    std::uint32_t global_rope_original_context;
    // Sliding layers: default RoPE over the whole head.
    float sliding_rope_theta;
    float sliding_rope_partial;
};

inline constexpr LagunaExecutionContract kLagunaExecutionContract{
    3072U, 48U, 4U, 48U, 72U, 8U, 128U, 512U,
    12288U, 1024U, 1024U, 256U, 10U, 1U, 40U, 100352U,
    1'048'576U, 16U, 1.0e-6F, 2.5F, 0.0F,
    500000.0F, 128.0F, 1.4852030263919618F, 32.0F, 1.0F, 0.5F, 8192U,
    10000.0F, 1.0F};

// Layer 0 and every fourth layer thereafter use full causal attention with 48
// heads; the remaining three of every four use a 512-token sliding window with
// 72 heads.
[[nodiscard]] constexpr bool laguna_global_attention_layer(
    std::uint32_t layer) noexcept {
    return layer < kLagunaExecutionContract.layer_count &&
           layer % kLagunaExecutionContract.global_attention_stride == 0U;
}

[[nodiscard]] constexpr std::uint32_t laguna_attention_heads(
    std::uint32_t layer) noexcept {
    return laguna_global_attention_layer(layer)
        ? kLagunaExecutionContract.global_attention_heads
        : kLagunaExecutionContract.sliding_attention_heads;
}

// Routed experts below this layer are NVFP4; layers at or above it ship plain
// BF16 experts because the checkpoint's quantization config ignores them.
[[nodiscard]] constexpr bool laguna_quantized_expert_layer(
    std::uint32_t layer) noexcept {
    return layer < kLagunaExecutionContract.quantized_expert_layers;
}

[[nodiscard]] constexpr bool laguna_sparse_layer(std::uint32_t layer) noexcept {
    return layer >= kLagunaExecutionContract.dense_prefix_layers &&
           layer < kLagunaExecutionContract.layer_count;
}

// Inkling-Small-NVFP4 (Thinking Machines). A 42-layer decoder with no rotary
// embedding at all: position enters through a learned per-head relative bias
// added to the pre-softmax logits. Every layer carries four depthwise causal
// short convolutions (K, V, attention output, MLP output). Layers 0 and 1 are
// dense; layers 2..41 are a 256-expert top-6 MoE with two shared "sink"
// experts that score in the router but are never selection candidates. Routed
// experts are NVFP4 (E2M1 pairs, group-16 E4M3 scales, per-expert FP32 global
// scale) for layers 3..41; layer 2 ships plain BF16 experts and every other
// module in the checkpoint is BF16.
struct InklingExecutionContract {
    std::uint32_t hidden_size;
    std::uint32_t layer_count;
    std::uint32_t attention_heads;
    std::uint32_t key_value_heads;
    std::uint32_t head_dim;
    // Width of the per-head relative branch `r`, and the number of distance
    // buckets its projection spans. Local layers span exactly their window.
    std::uint32_t relative_dim;
    std::uint32_t global_relative_extent;
    std::uint32_t local_relative_extent;
    std::uint32_t sliding_window;
    std::uint32_t global_attention_period;
    std::uint32_t short_conv_kernel;
    std::uint32_t dense_intermediate_size;
    std::uint32_t expert_intermediate_size;
    std::uint32_t routed_experts;
    std::uint32_t experts_per_token;
    std::uint32_t shared_experts;
    std::uint32_t dense_prefix_layers;
    // Routed experts below this layer ship plain BF16; from it on they are
    // NVFP4. Only layer 2 is unquantized.
    std::uint32_t quantized_expert_start_layer;
    std::uint32_t padded_vocabulary_size;
    // Logits are computed over the padded table and truncated to this width.
    std::uint32_t vocabulary_size;
    std::uint32_t maximum_context_tokens;
    std::uint32_t nvfp4_group_size;
    std::uint32_t mtp_layers;
    // Long-context temperature: tau = 1 + alpha*log(max(1, (pos+1)/n_floor)),
    // applied to normalized queries and relative logits on global layers only.
    std::uint32_t log_scaling_position_floor;
    float log_scaling_alpha;
    float rms_epsilon;
    float routed_scale;
    // Q and K are per-head RMS-normed to unit norm, so the reference divides
    // logits by head_dim rather than its square root.
    float attention_scale;
    // Final logits are divided by this muP width multiplier.
    float logits_width_multiplier;
};

inline constexpr InklingExecutionContract kInklingExecutionContract{
    4096U, 42U, 32U, 8U, 128U,
    16U, 1024U, 512U, 512U, 6U, 4U,
    16384U, 2048U, 256U, 6U, 2U, 2U, 3U,
    201024U, 200058U, 1'048'576U, 16U, 8U,
    128000U, 0.1F, 1.0e-6F, 8.0F, 1.0F / 128.0F, 16.0F};

// Layers 5, 11, 17, 23, 29, 35 and 41 use full causal attention; the other 35
// use a 512-token sliding window. Head counts are identical either way, so the
// pattern only changes masking and relative extent.
[[nodiscard]] constexpr bool inkling_global_attention_layer(
    std::uint32_t layer) noexcept {
    return layer < kInklingExecutionContract.layer_count &&
           (layer + 1U) % kInklingExecutionContract.global_attention_period == 0U;
}

[[nodiscard]] constexpr std::uint32_t inkling_relative_extent(
    std::uint32_t layer) noexcept {
    return inkling_global_attention_layer(layer)
        ? kInklingExecutionContract.global_relative_extent
        : kInklingExecutionContract.local_relative_extent;
}

[[nodiscard]] constexpr bool inkling_sparse_layer(std::uint32_t layer) noexcept {
    return layer >= kInklingExecutionContract.dense_prefix_layers &&
           layer < kInklingExecutionContract.layer_count;
}

[[nodiscard]] constexpr bool inkling_quantized_expert_layer(
    std::uint32_t layer) noexcept {
    return layer >= kInklingExecutionContract.quantized_expert_start_layer &&
           layer < kInklingExecutionContract.layer_count;
}

// MTP depths 1 and 3 are full-attention; depths 0, 2, 4, 5, 6 and 7 slide.
// The checkpoint declares this as mtp_config.local_layer_ids, which is not a
// stride, so it is pinned as a mask.
[[nodiscard]] constexpr bool inkling_mtp_global_attention_depth(
    std::uint32_t depth) noexcept {
    return depth < kInklingExecutionContract.mtp_layers &&
           (depth == 1U || depth == 3U);
}

[[nodiscard]] constexpr std::uint32_t inkling_mtp_relative_extent(
    std::uint32_t depth) noexcept {
    return inkling_mtp_global_attention_depth(depth)
        ? kInklingExecutionContract.global_relative_extent
        : kInklingExecutionContract.local_relative_extent;
}

// Kimi-K3. Hybrid backbone: three Kimi Delta Attention layers then one gated
// NoPE MLA layer, repeating, plus a final MLA layer at the end of the backbone.
// Only the routed experts are quantized (MXFP4); the whole dense spine is BF16.
struct KimiK3ExecutionContract {
    std::uint32_t hidden_size;
    std::uint32_t layer_count;
    std::uint32_t attention_heads;
    std::uint32_t key_value_heads;
    // Gated MLA.
    std::uint32_t query_lora_rank;
    std::uint32_t kv_lora_rank;
    std::uint32_t nope_head_dim;
    std::uint32_t rope_head_dim;
    std::uint32_t value_head_dim;
    // KDA.
    std::uint32_t linear_attention_heads;
    std::uint32_t linear_head_dim;
    std::uint32_t short_conv_kernel;
    std::uint32_t decay_rank;
    // Feed-forward and MoE.
    std::uint32_t dense_intermediate_size;
    std::uint32_t expert_intermediate_size;
    std::uint32_t routed_expert_hidden_size;
    std::uint32_t routed_experts;
    std::uint32_t experts_per_token;
    std::uint32_t expert_groups;
    std::uint32_t selected_expert_groups;
    std::uint32_t shared_experts;
    std::uint32_t dense_prefix_layers;
    // Attention residuals.
    std::uint32_t attention_residual_block_size;
    std::uint32_t vocabulary_size;
    std::uint32_t maximum_context_tokens;
    std::uint32_t image_token_id;
    // Vision (MoonViT-V2).
    std::uint32_t vision_hidden_size;
    std::uint32_t vision_layer_count;
    std::uint32_t vision_attention_heads;
    std::uint32_t vision_qkv_hidden_size;
    std::uint32_t vision_intermediate_size;
    std::uint32_t vision_patch_size;
    std::uint32_t vision_position_embedding_extent;
    std::uint32_t vision_merge_kernel;
    std::uint32_t vision_projector_hidden_size;
    float rms_epsilon;
    float routed_scale;
    // SiTU-GLU: [b1 tanh(g/b1) sigmoid(g)] * [b2 tanh(u/b2)].
    float situ_gate_beta;
    float situ_linear_beta;
    // KDA decay: alpha = exp(gate_lower_bound * sigmoid(exp(A_log) * z)).
    float kda_gate_lower_bound;
};

inline constexpr KimiK3ExecutionContract kKimiK3ExecutionContract{
    7168U, 93U, 96U, 96U,
    1536U, 512U, 128U, 64U, 128U,
    96U, 128U, 4U, 128U,
    33792U, 3072U, 3584U, 896U, 16U, 1U, 1U, 2U, 1U,
    12U, 163840U, 1'048'576U, 163605U,
    1024U, 27U, 12U, 1536U, 4096U, 14U, 64U, 2U, 4096U,
    1.0e-5F, 1.0F, 4.0F, 25.0F, -5.0F};

// `linear_attn_config.full_attn_layers` is 1-based in the checkpoint and lists
// every fourth layer plus a final one: {4, 8, ..., 88, 92, 93}. Expressed here
// on Strata's 0-based layer index, which is what every caller uses.
[[nodiscard]] constexpr bool kimi_k3_full_attention_layer(
    std::uint32_t layer) noexcept {
    return layer < kKimiK3ExecutionContract.layer_count &&
           ((layer + 1U) % 4U == 0U ||
            layer + 1U == kKimiK3ExecutionContract.layer_count);
}

[[nodiscard]] constexpr bool kimi_k3_kda_layer(std::uint32_t layer) noexcept {
    return layer < kKimiK3ExecutionContract.layer_count &&
           !kimi_k3_full_attention_layer(layer);
}

// Layer 0 is the dense MLP prefix (`first_k_dense_replace = 1`); every later
// layer carries a LatentMoE block.
[[nodiscard]] constexpr bool kimi_k3_moe_layer(std::uint32_t layer) noexcept {
    return layer >= kKimiK3ExecutionContract.dense_prefix_layers &&
           layer < kKimiK3ExecutionContract.layer_count;
}


}  // namespace strata
