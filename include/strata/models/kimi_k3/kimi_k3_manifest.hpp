#pragma once

#include "strata/models/common/model.hpp"
#include "strata/platform/result.hpp"
#include "strata/platform/safetensors.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace strata {

// What a Kimi-K3 tensor is for. Placement, the checkpoint reader, and the
// runtime all classify by this rather than by re-parsing names.
enum class KimiTensorRole : std::uint8_t {
    Embedding,
    OutputHead,
    Norm,
    // Gated MLA projections: q_a, q_b, kv_a_proj_with_mqa, kv_b, o, g.
    MlaAttention,
    // KDA projections and state: q, k, v, b, f_a, f_b, g, o, conv1d, A_log,
    // dt_bias, o_norm.
    KdaAttention,
    // Per-layer attention-residual pseudo-query and its norm, at the attention
    // site, the MLP site, and the backbone output.
    AttentionResidual,
    // The layer-0 dense MLP.
    DenseMlp,
    Router,
    SharedExpert,
    // W-down into the 3584-wide latent space, W-up back out, and the norm
    // between aggregation and up-projection.
    LatentMoeProjection,
    RoutedExpert,
    Vision,
    VisionProjector,
    Count,
};

enum class KimiTensorComponent : std::uint8_t {
    Weight,
    Bias,
    PackedWeight,
    Scale,
};

enum class KimiTensorEncoding : std::uint8_t {
    // The dense spine is not quantized at all.
    PlainBf16,
    PlainF32,
    Mxfp4Group32,
};

struct KimiManifestTensor {
    std::string name;
    std::string shard;
    KimiTensorRole role{KimiTensorRole::Norm};
    KimiTensorComponent component{KimiTensorComponent::Weight};
    KimiTensorEncoding encoding{KimiTensorEncoding::PlainBf16};
    std::int32_t layer{-1};
    std::int32_t expert{-1};
    SafetensorsDtype source_dtype{SafetensorsDtype::Other};
    std::vector<std::uint64_t> source_shape;
    std::uint64_t source_offset{};
    std::uint64_t source_bytes{};
};

struct KimiIndexManifest {
    std::uint64_t indexed_tensor_bytes{};
    std::vector<std::string> shards;
    std::vector<KimiManifestTensor> tensors;
    std::array<std::uint64_t, static_cast<std::size_t>(KimiTensorRole::Count)>
        role_counts{};
    // Split the byte total the way the placement problem splits: the routed
    // experts are 92.7% of the checkpoint and the only quantized part of it.
    std::uint64_t routed_expert_bytes{};
    std::uint64_t dense_spine_bytes{};
    std::uint64_t routed_expert_modules{};
    std::uint64_t kda_layers{};
    std::uint64_t full_attention_layers{};
    std::uint64_t moe_layers{};
    std::uint64_t scanned_shards{};
    std::uint64_t shard_file_bytes{};
    std::uint64_t tensor_payload_bytes{};
};

// The subset of `config.json` the runtime is allowed to depend on. Parsed and
// then checked field by field against `kKimiK3ExecutionContract`; a checkpoint
// that disagrees is rejected at load rather than silently reinterpreted.
struct KimiK3Config {
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
    std::uint32_t routed_expert_hidden_size{};
    std::uint32_t routed_experts{};
    std::uint32_t experts_per_token{};
    std::uint32_t expert_groups{};
    std::uint32_t selected_expert_groups{};
    std::uint32_t shared_experts{};
    std::uint32_t dense_prefix_layers{};
    std::uint32_t attention_residual_block_size{};
    std::uint32_t vocabulary_size{};
    std::uint32_t maximum_context_tokens{};
    std::uint32_t image_token_id{};
    std::uint32_t vision_hidden_size{};
    std::uint32_t vision_layer_count{};
    std::uint32_t vision_attention_heads{};
    std::uint32_t vision_qkv_hidden_size{};
    std::uint32_t vision_intermediate_size{};
    std::uint32_t vision_patch_size{};
    std::uint32_t vision_merge_kernel{};
    std::uint32_t quantization_group_size{};
    std::uint32_t quantization_bits{};
    float rms_epsilon{};
    float routed_scale{};
    float situ_gate_beta{};
    float situ_linear_beta{};
    float kda_gate_lower_bound{};
    bool mla_use_nope{};
    bool mla_use_output_gate{};
    bool latent_moe_use_norm{};
    bool moe_renormalize{};
    bool use_full_rank_gate{};
    bool tie_word_embeddings{};
    std::string hidden_activation;
    std::string router_activation;
    std::string topk_method;
    std::string quantization_format;
    // 0-based, sorted. The checkpoint stores these 1-based.
    std::vector<std::uint32_t> full_attention_layers;
    std::vector<std::uint32_t> kda_layers;
};

struct KimiConfigResult {
    KimiK3Config value;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

struct KimiManifestResult {
    KimiIndexManifest manifest;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

struct KimiCheckpointOptions {
    bool require_all_shards{true};
    std::size_t maximum_errors{64};
};

[[nodiscard]] ModelSpec kimi_k3_mxfp4_spec();
[[nodiscard]] ValidationResult validate_kimi_k3_mxfp4(const ModelSpec& spec);

[[nodiscard]] KimiConfigResult parse_kimi_k3_config(std::string_view json);
// Rejects any deviation from `kKimiK3ExecutionContract`. Precision, router
// semantics, expert count, and top-k may not change silently, so every one of
// them is a named error here rather than a tolerance.
[[nodiscard]] ValidationResult validate_kimi_k3_config(
    const KimiK3Config& config);

[[nodiscard]] KimiTensorRole classify_kimi_tensor(std::string_view name,
                                                  std::int32_t& layer,
                                                  std::int32_t& expert) noexcept;

[[nodiscard]] KimiManifestResult build_kimi_k3_index_manifest(
    SafetensorsIndex index);
[[nodiscard]] KimiManifestResult validate_kimi_k3_checkpoint(
    const std::string& model_directory, KimiIndexManifest manifest,
    const KimiCheckpointOptions& options = {});

[[nodiscard]] std::string_view to_string(KimiTensorRole role) noexcept;
[[nodiscard]] std::string_view to_string(KimiTensorEncoding encoding) noexcept;

}  // namespace strata
