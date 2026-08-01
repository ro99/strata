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

}  // namespace strata
