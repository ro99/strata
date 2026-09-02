#pragma once

#include "strata/platform/result.hpp"
#include "strata/platform/safetensors.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace strata {

// The two GLM-5.3-Flash releases Strata executes. Both quantize only the
// routed experts; every other module is BF16 in the MXFP4 release and FP8 in
// the FP8 release. The distinction is a property of the checkpoint, resolved
// once at open and carried on the manifest, never a build-time choice.
enum class Glm53Quantization : std::uint8_t {
    // GLM-5.3-Flash FP8: E4M3 payload with one F32 inverse scale per 128x128
    // weight block.
    Fp8E4m3Block128,
    // GLM-5.3-Flash MXFP4 (quark "quark" export): E2M1 nibble pairs packed two
    // per byte along the input dimension, with one E8M0 scale byte per 32
    // columns of each row. Routed experts in layers 3, 5, 6 and the MTP layer
    // stay BF16 by the publisher's mixed-precision correction.
    Mxfp4Group32,
    Unsupported,
};

struct Glm53TextConfig {
    std::uint32_t hidden_size{};
    std::uint32_t layer_count{};
    std::uint32_t attention_heads{};
    std::uint32_t key_value_heads{};
    std::uint32_t query_lora_rank{};
    std::uint32_t kv_lora_rank{};
    std::uint32_t nope_head_dim{};
    std::uint32_t rope_head_dim{};
    std::uint32_t value_head_dim{};
    std::uint32_t linear_attention_heads{};
    std::uint32_t linear_head_dim{};
    std::uint32_t short_conv_kernel{};
    std::uint32_t dense_intermediate_size{};
    std::uint32_t expert_intermediate_size{};
    std::uint32_t routed_experts{};
    std::uint32_t experts_per_token{};
    std::uint32_t expert_groups{};
    std::uint32_t selected_expert_groups{};
    std::uint32_t shared_experts{};
    std::uint32_t dense_prefix_layers{};
    std::uint32_t vocabulary_size{};
    std::uint32_t maximum_context_tokens{};
    std::uint32_t mhc_multiplier{};
    std::uint32_t mhc_sinkhorn_iterations{};
    std::uint32_t index_heads{};
    std::uint32_t index_head_dim{};
    std::uint32_t index_topk{};
    std::uint32_t index_pool{};
    std::uint32_t fp8_block_rows{};
    std::uint32_t fp8_block_columns{};
    float rms_epsilon{};
    float mhc_epsilon{};
    float routed_scale{};
    float swiglu_limit{};
    float kda_gate_lower_bound{};
    bool normalize_topk{};
    bool mhc{};
    bool mla_use_nope{};
    bool index_pool_compress{};
    bool index_pool_select_tail{};
    bool tie_word_embeddings{};
    std::string architecture;
    std::string model_type;
    std::string hidden_activation;
    std::string router_scoring;
    std::string topk_method;
    std::uint32_t quantization_group_size{};
    std::string quantization_method;
    std::string quantization_format;
    std::string quantization_weight_dtype;
    std::string quantization_scale_format;
    std::vector<std::string> attention_layer_types;
    std::vector<std::string> mlp_layer_types;
    std::vector<std::uint32_t> full_attention_layers;
    std::vector<std::uint32_t> kda_layers;
};

struct Glm53ConfigResult {
    Glm53TextConfig value;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

enum class Glm53TensorRole : std::uint8_t {
    Embedding,
    OutputHead,
    Norm,
    Mhc,
    KdaAttention,
    SparseAttention,
    AttentionIndexer,
    DenseMlp,
    Router,
    SharedExpert,
    RoutedExpert,
    Mtp,
    Vision,
    Count,
};

enum class Glm53TensorComponent : std::uint8_t {
    Weight,
    Scale,
    Bias,
    State,
};

enum class Glm53TensorEncoding : std::uint8_t {
    Plain,
    Fp8E4m3Block128F32,
    Fp4E2m1Group32E8m0,
};

struct Glm53ManifestTensor {
    std::string name;
    std::string shard;
    Glm53TensorRole role{Glm53TensorRole::Norm};
    Glm53TensorComponent component{Glm53TensorComponent::State};
    Glm53TensorEncoding encoding{Glm53TensorEncoding::Plain};
    std::int32_t layer{-1};
    std::int32_t expert{-1};
    SafetensorsDtype source_dtype{SafetensorsDtype::Other};
    std::vector<std::uint64_t> source_shape;
    std::uint64_t source_offset{};
    std::uint64_t source_bytes{};
};

struct Glm53IndexManifest {
    Glm53Quantization quantization{Glm53Quantization::Fp8E4m3Block128};
    std::uint64_t indexed_tensor_bytes{};
    std::vector<std::string> shards;
    std::vector<Glm53ManifestTensor> tensors;
    std::array<std::uint64_t, static_cast<std::size_t>(Glm53TensorRole::Count)>
        role_counts{};
    std::uint64_t fp8_modules{};
    std::uint64_t dense_spine_bytes{};
    std::uint64_t routed_expert_bytes{};
    std::uint64_t vision_bytes{};
    std::uint64_t scanned_shards{};
    std::uint64_t shard_file_bytes{};
    std::uint64_t tensor_payload_bytes{};
};

struct Glm53ManifestResult {
    Glm53IndexManifest manifest;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

struct Glm53CheckpointOptions {
    bool require_all_shards{true};
    std::size_t maximum_errors{64U};
};

[[nodiscard]] constexpr bool glm53_full_attention_layer(
    std::uint32_t layer) noexcept {
    return layer < 45U && (layer + 1U) % 4U == 0U;
}

[[nodiscard]] constexpr bool glm53_kda_layer(std::uint32_t layer) noexcept {
    return layer < 45U && !glm53_full_attention_layer(layer);
}

[[nodiscard]] constexpr bool glm53_moe_layer(std::uint32_t layer) noexcept {
    return layer >= 3U && layer < 45U;
}

[[nodiscard]] Glm53ConfigResult parse_glm53_config(std::string_view json);
[[nodiscard]] ValidationResult validate_glm53_config(
    const Glm53TextConfig& config);
// Names the release a parsed config describes, or Unsupported when it matches
// neither. Validation reports the diagnostic; this is the selector every other
// layer branches on.
[[nodiscard]] Glm53Quantization glm53_config_quantization(
    const Glm53TextConfig& config) noexcept;
[[nodiscard]] Glm53TensorRole classify_glm53_tensor(
    std::string_view name, std::int32_t& layer, std::int32_t& expert) noexcept;
[[nodiscard]] Glm53ManifestResult build_glm53_index_manifest(
    SafetensorsIndex index);
[[nodiscard]] Glm53ManifestResult validate_glm53_checkpoint(
    const std::string& model_directory, Glm53IndexManifest manifest,
    const Glm53CheckpointOptions& options = {});
[[nodiscard]] std::string_view to_string(Glm53TensorRole role) noexcept;
[[nodiscard]] std::string_view to_string(Glm53TensorEncoding encoding) noexcept;
[[nodiscard]] std::string_view to_string(Glm53Quantization quantization) noexcept;

}  // namespace strata
