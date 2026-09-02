#pragma once

#include "strata/device/cuda_backend.hpp"
#include "strata/models/glm53/glm53_manifest.hpp"
#include "strata/platform/checkpoint_io.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace strata {

class Glm53CheckpointReader;

struct Glm53CheckpointOpenResult {
    std::unique_ptr<Glm53CheckpointReader> value;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept {
        return errors.empty() && value != nullptr;
    }
};

class Glm53CheckpointReader {
public:
    Glm53CheckpointReader(const Glm53CheckpointReader&) = delete;
    Glm53CheckpointReader& operator=(const Glm53CheckpointReader&) = delete;
    Glm53CheckpointReader(Glm53CheckpointReader&&) = delete;
    Glm53CheckpointReader& operator=(Glm53CheckpointReader&&) = delete;
    ~Glm53CheckpointReader();

    [[nodiscard]] static Glm53CheckpointOpenResult open(
        std::string model_directory);
    [[nodiscard]] const Glm53ManifestTensor* find(
        std::string_view name) const noexcept;
    [[nodiscard]] ParseResult<std::vector<std::byte>> read(
        std::string_view name, std::uint64_t maximum_bytes) const;
    [[nodiscard]] ParseResult<std::span<const std::byte>> view(
        std::string_view name) const;
    [[nodiscard]] ParseResult<std::vector<float>> read_f32(
        std::string_view name, std::uint64_t maximum_elements) const;
    [[nodiscard]] ParseResult<std::vector<float>> read_f32_row(
        std::string_view name, std::uint64_t row) const;
    [[nodiscard]] std::uint64_t cuda_linear_storage_bytes(
        std::string_view base_name) const;
    [[nodiscard]] std::uint64_t cuda_linear_slice_storage_bytes(
        std::string_view base_name, std::uint64_t row_begin,
        std::uint64_t row_count) const;
    [[nodiscard]] ValidationResult load_cuda_linear(
        std::string_view base_name, std::uint64_t rows,
        std::uint64_t columns, int device, CudaBackend& backend,
        CudaWeight& output, bool concurrent_prefetch = false,
        bool canonical_layout = false) const;
    [[nodiscard]] ValidationResult load_cuda_linear_slice(
        std::string_view base_name, std::uint64_t total_rows,
        std::uint64_t columns, std::uint64_t row_begin,
        std::uint64_t row_count, int device, CudaBackend& backend,
        CudaWeight& output) const;

    [[nodiscard]] const Glm53TextConfig& config() const noexcept {
        return config_;
    }
    [[nodiscard]] const Glm53IndexManifest& manifest() const noexcept {
        return manifest_;
    }

private:
    Glm53CheckpointReader() = default;
    [[nodiscard]] ParseResult<std::vector<std::byte>> read_slice(
        const Glm53ManifestTensor& tensor, std::uint64_t offset,
        std::uint64_t bytes) const;

    std::string model_directory_;
    Glm53TextConfig config_;
    Glm53IndexManifest manifest_;
    std::unordered_map<std::string_view, std::size_t> by_name_;
    CheckpointShardSet shards_;
    struct ShardMapping {
        std::byte* address{};
        std::uint64_t bytes{};
    };
    mutable std::mutex mapping_mutex_;
    mutable std::unordered_map<std::string, ShardMapping> mappings_;
};

}  // namespace strata
