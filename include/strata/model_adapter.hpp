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
