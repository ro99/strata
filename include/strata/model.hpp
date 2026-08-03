#pragma once

#include "strata/result.hpp"
#include "strata/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata {

enum class ArchitectureKind : std::uint8_t {
    Dense,
    StandardMoe,
    DeepSeek,
    GlmMoeDsa,
    Gemma4,
    Laguna,
};

enum class AttentionKind : std::uint8_t {
    Mha,
    Gqa,
    Mqa,
    Mla,
    HybridCompressedSparse,
    HybridLocalGlobal,
};

enum class RouterSelectionKind : std::uint8_t {
    None,
    TopK,
    GroupLimitedTopK,
    NoAuxTc,
};

enum class RouterScoreKind : std::uint8_t {
    None,
    Softmax,
    Sigmoid,
    SqrtSoftplus,
};

struct RouterSpec {
    RouterSelectionKind selection{RouterSelectionKind::None};
    RouterScoreKind scoring{RouterScoreKind::None};
    std::uint32_t routed_experts{};
    std::uint32_t experts_per_token{};
    std::uint32_t groups{1};
    std::uint32_t selected_groups{1};
    bool normalize_topk{true};
    bool selection_bias{false};
    float routed_scale{1.0F};
};

enum class QuantizationKind : std::uint8_t {
    Uniform,
    CompressedTensorsW4A16,
    CompressedTensorsW8A16,
    CompressedTensorsNvfp4W4A16,
    NativeFp4Fp8,
};

enum class QuantizationGranularity : std::uint8_t {
    Group,
    Channel,
};

struct QuantizedWeightSpec {
    std::uint32_t bits{kMinimumQuantBits};
    QuantizationGranularity granularity{QuantizationGranularity::Group};
    std::uint32_t group_size{};
    bool symmetric{true};
};

struct MixedQuantizationSpec {
    QuantizationKind kind{QuantizationKind::Uniform};
    std::uint32_t activation_bits{16};
    std::uint32_t quantized_linear_start_layer{};
    std::uint32_t quantized_expert_start_layer{};
    std::uint32_t mtp_layer_index{};
    QuantizedWeightSpec routed_experts;
    QuantizedWeightSpec linears;
    QuantizedWeightSpec mtp;
};

struct NativeFp4Fp8Spec {
    std::uint32_t activation_bits{8};
    std::uint32_t activation_group_size{128};
    std::uint32_t fp8_weight_bits{8};
    std::uint32_t fp8_block_rows{128};
    std::uint32_t fp8_block_columns{128};
    std::uint32_t fp4_weight_bits{4};
    std::uint32_t fp4_group_size{32};
    bool power_of_two_scales{true};
};

struct SourceManifestSpec {
    std::string repository;
    std::string revision;
    std::string index_sha256;
    std::uint64_t tensor_count{};
    std::uint64_t indexed_tensor_bytes{};
    std::uint64_t shard_file_bytes{};
    std::uint32_t main_shards{};
    std::uint32_t mtp_shards{};
};

struct GlmMoeDsaSpec {
    std::uint32_t sparse_attention_topk{};
    std::uint32_t index_share_frequency{};
    std::uint32_t mtp_layers{};
    bool index_share_for_mtp{};
};

struct DeepSeekV4Spec {
    std::uint32_t attention_heads{};
    std::uint32_t key_value_heads{};
    std::uint32_t head_dim{};
    std::uint32_t rope_head_dim{};
    std::uint32_t query_lora_rank{};
    std::uint32_t output_lora_rank{};
    std::uint32_t output_groups{};
    std::uint32_t sliding_window{};
    std::uint32_t index_heads{};
    std::uint32_t index_head_dim{};
    std::uint32_t index_topk{};
    std::uint32_t hash_layers{};
    std::uint32_t mhc_multiplier{};
    std::uint32_t mhc_sinkhorn_iterations{};
    std::uint32_t dspark_layers{};
    std::uint32_t dspark_block_size{};
    std::uint32_t dspark_noise_token_id{};
    std::uint32_t dspark_markov_rank{};
    float swiglu_limit{};
    std::vector<std::uint32_t> compression_ratios;
    std::vector<std::uint32_t> dspark_target_layers;
};

struct Gemma4Spec {
    std::uint32_t attention_heads{};
    std::uint32_t local_key_value_heads{};
    std::uint32_t global_key_value_heads{};
    std::uint32_t local_head_dim{};
    std::uint32_t global_head_dim{};
    std::uint32_t intermediate_size{};
    std::uint32_t sliding_window{};
    std::uint32_t global_attention_stride{};
    std::uint32_t vision_hidden_size{};
    std::uint32_t vision_layers{};
    std::uint32_t vision_attention_heads{};
    std::uint32_t vision_head_dim{};
    std::uint32_t vision_intermediate_size{};
    std::uint32_t vision_patch_size{};
    std::uint32_t vision_position_embeddings{};
    std::uint32_t vision_pooling_kernel{};
    std::uint32_t image_token_id{};
    std::uint32_t default_image_tokens{};
    float rms_epsilon{};
    float local_rope_theta{};
    float global_rope_theta{};
    float global_rope_proportion{};
    float vision_rope_theta{};
    float final_logit_softcap{};
    bool global_key_equals_value{};
    bool vision_bidirectional{};
};

struct LagunaSpec {
    std::uint32_t global_attention_heads{};
    std::uint32_t sliding_attention_heads{};
    std::uint32_t key_value_heads{};
    std::uint32_t head_dim{};
    std::uint32_t sliding_window{};
    std::uint32_t global_attention_stride{};
    std::uint32_t dense_intermediate_size{};
    std::uint32_t shared_expert_intermediate_size{};
    // Routed experts below this layer are NVFP4; the rest ship plain BF16.
    std::uint32_t quantized_expert_layers{};
    float rms_epsilon{};
    float router_logit_softcapping{};
    float global_rope_theta{};
    float global_rope_factor{};
    float global_rope_attention_factor{};
    float global_rope_beta_fast{};
    float global_rope_beta_slow{};
    float global_rope_partial{};
    std::uint32_t global_rope_original_context{};
    float sliding_rope_theta{};
    float sliding_rope_partial{};
    // softplus(g_proj(x)) broadcast across head_dim, applied before o_proj.
    bool per_head_output_gating{};
    // QK RMSNorm applied per head after reshape and before RoPE.
    bool query_key_norm{};
};

struct ModelSpec {
    std::string name;
    ArchitectureKind architecture{ArchitectureKind::Dense};
    AttentionKind attention{AttentionKind::Mha};
    RouterSpec router;
    MixedQuantizationSpec mixed_quantization;
    NativeFp4Fp8Spec native_fp4_fp8;
    SourceManifestSpec source;
    GlmMoeDsaSpec glm_moe_dsa;
    DeepSeekV4Spec deepseek_v4;
    Gemma4Spec gemma4;
    LagunaSpec laguna;
    std::uint32_t quant_bits{kMinimumQuantBits};
    std::uint32_t hidden_size{};
    std::uint32_t layer_count{};
    std::uint32_t max_context_tokens{};
    std::uint32_t dense_prefix_layers{};
    std::uint32_t shared_experts{};
    std::uint32_t expert_intermediate_size{};
};

[[nodiscard]] ValidationResult validate_model(const ModelSpec& spec);
[[nodiscard]] ModelSpec glm52_w4a16_spec();
[[nodiscard]] ValidationResult validate_glm52_w4a16(
    const ModelSpec& spec);
[[nodiscard]] ModelSpec deepseek_v4_flash_0731_spec();
[[nodiscard]] ValidationResult validate_deepseek_v4_flash_0731(
    const ModelSpec& spec);
[[nodiscard]] ModelSpec gemma4_31b_it_w8a16_spec();
[[nodiscard]] ValidationResult validate_gemma4_31b_it_w8a16(
    const ModelSpec& spec);
[[nodiscard]] ModelSpec laguna_s21_nvfp4_spec();
[[nodiscard]] ValidationResult validate_laguna_s21_nvfp4(
    const ModelSpec& spec);

}  // namespace strata
