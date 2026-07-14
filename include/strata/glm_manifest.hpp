#pragma once

#include "strata/model.hpp"
#include "strata/safetensors.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace strata {

enum class GlmTensorRole : std::uint8_t {
    Embedding,
    OutputHead,
    Norm,
    Attention,
    AttentionIndexer,
    DenseMlp,
    Router,
    SharedExpert,
    RoutedExpert,
    MtpState,
    Count,
};

enum class GlmTensorComponent : std::uint8_t {
    Weight,
    Bias,
    PackedWeight,
    Scale,
    LogicalShape,
};

enum class GlmTensorEncoding : std::uint8_t {
    Plain,
    Int4Group128,
    Int8Group128,
    Int8Channel,
};

struct GlmManifestTensor {
    std::string name;
    std::string shard;
    GlmTensorRole role{GlmTensorRole::Norm};
    GlmTensorComponent component{GlmTensorComponent::Weight};
    GlmTensorEncoding encoding{GlmTensorEncoding::Plain};
    std::int32_t layer{-1};
    std::int32_t expert{-1};
    bool mtp{};
    SafetensorsDtype source_dtype{SafetensorsDtype::Other};
    std::vector<std::uint64_t> source_shape;
    std::uint64_t source_offset{};
    std::uint64_t source_bytes{};
};

struct GlmIndexManifest {
    std::uint64_t indexed_tensor_bytes{};
    std::vector<std::string> shards;
    std::vector<GlmManifestTensor> tensors;
    std::array<std::uint64_t, static_cast<std::size_t>(GlmTensorRole::Count)> role_counts{};
    std::uint64_t quantized_modules{};
    std::uint64_t int4_modules{};
    std::uint64_t int8_group_modules{};
    std::uint64_t int8_channel_modules{};
    std::uint64_t resolved_tensors{};
    std::uint64_t validated_layouts{};
    std::uint64_t scanned_shards{};
    std::uint64_t shard_file_bytes{};
    std::uint64_t tensor_payload_bytes{};
};

struct GlmCheckpointOptions {
    bool require_all_shards{true};
    bool require_read_only{false};
    bool validate_logical_shapes{true};
    std::size_t maximum_errors{64};
};

struct GlmManifestResult {
    GlmIndexManifest manifest;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

[[nodiscard]] GlmManifestResult build_quanttrio_glm52_index_manifest(
    SafetensorsIndex index);
[[nodiscard]] GlmManifestResult validate_quanttrio_glm52_checkpoint(
    const std::string& model_directory, GlmIndexManifest manifest,
    const GlmCheckpointOptions& options = {});
[[nodiscard]] std::string_view to_string(GlmTensorRole role) noexcept;
[[nodiscard]] std::string_view to_string(GlmTensorEncoding encoding) noexcept;

}  // namespace strata
