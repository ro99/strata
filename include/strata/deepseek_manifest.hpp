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

enum class Dsv4TensorRole : std::uint8_t {
    Embedding,
    OutputHead,
    Norm,
    Attention,
    AttentionCompressor,
    AttentionIndexer,
    Router,
    SharedExpert,
    RoutedExpert,
    Mhc,
    DsparkState,
    Count,
};

enum class Dsv4TensorComponent : std::uint8_t {
    Weight,
    Scale,
    Bias,
    State,
};

enum class Dsv4TensorEncoding : std::uint8_t {
    Plain,
    Fp8E4m3Block128,
    Fp4E2m1Group32,
};

struct Dsv4ManifestTensor {
    std::string name;
    std::string shard;
    Dsv4TensorRole role{Dsv4TensorRole::Norm};
    Dsv4TensorComponent component{Dsv4TensorComponent::State};
    Dsv4TensorEncoding encoding{Dsv4TensorEncoding::Plain};
    std::int32_t layer{-1};
    std::int32_t expert{-1};
    bool dspark{};
    SafetensorsDtype source_dtype{SafetensorsDtype::Other};
    std::vector<std::uint64_t> source_shape;
    std::uint64_t source_offset{};
    std::uint64_t source_bytes{};
};

struct Dsv4IndexManifest {
    std::uint64_t indexed_tensor_bytes{};
    std::vector<std::string> shards;
    std::vector<Dsv4ManifestTensor> tensors;
    std::array<std::uint64_t, static_cast<std::size_t>(Dsv4TensorRole::Count)>
        role_counts{};
    std::uint64_t quantized_modules{};
    std::uint64_t fp8_modules{};
    std::uint64_t fp4_modules{};
    std::uint64_t resolved_tensors{};
    std::uint64_t validated_layouts{};
    std::uint64_t scanned_shards{};
    std::uint64_t shard_file_bytes{};
    std::uint64_t tensor_payload_bytes{};
};

struct Dsv4CheckpointOptions {
    bool require_all_shards{true};
    bool require_read_only{false};
    bool validate_quantized_layouts{true};
    std::size_t maximum_errors{64};
};

struct Dsv4ManifestResult {
    Dsv4IndexManifest manifest;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

[[nodiscard]] Dsv4ManifestResult build_deepseek_v4_flash_dspark_index_manifest(
    SafetensorsIndex index);
[[nodiscard]] Dsv4ManifestResult validate_deepseek_v4_flash_dspark_checkpoint(
    const std::string& model_directory, Dsv4IndexManifest manifest,
    const Dsv4CheckpointOptions& options = {});
[[nodiscard]] std::string_view to_string(Dsv4TensorRole role) noexcept;
[[nodiscard]] std::string_view to_string(Dsv4TensorEncoding encoding) noexcept;

}  // namespace strata
