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

}  // namespace strata
