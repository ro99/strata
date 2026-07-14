#pragma once

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
};

enum class AttentionKind : std::uint8_t {
    Mha,
    Gqa,
    Mqa,
    Mla,
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
    CompressedTensorsInt4Int8Mix,
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

struct ModelSpec {
    std::string name;
    ArchitectureKind architecture{ArchitectureKind::Dense};
    AttentionKind attention{AttentionKind::Mha};
    RouterSpec router;
    MixedQuantizationSpec mixed_quantization;
    SourceManifestSpec source;
    GlmMoeDsaSpec glm_moe_dsa;
    std::uint32_t quant_bits{kMinimumQuantBits};
    std::uint32_t hidden_size{};
    std::uint32_t layer_count{};
    std::uint32_t max_context_tokens{};
    std::uint32_t dense_prefix_layers{};
    std::uint32_t shared_experts{};
    std::uint32_t expert_intermediate_size{};
};

struct ValidationResult {
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

[[nodiscard]] ValidationResult validate_model(const ModelSpec& spec);
[[nodiscard]] ModelSpec quanttrio_glm52_int4_int8_mix_spec();
[[nodiscard]] ValidationResult validate_quanttrio_glm52_int4_int8_mix(
    const ModelSpec& spec);

}  // namespace strata
