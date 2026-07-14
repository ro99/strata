#pragma once

#include "strata/cuda_backend.hpp"
#include "strata/glm_manifest.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace strata {

struct CheckpointReadStats {
    std::uint64_t calls{};
    std::uint64_t bytes{};
    std::uint64_t nanoseconds{};
};

class GlmCheckpointReader;

struct GlmCheckpointOpenResult {
    std::unique_ptr<GlmCheckpointReader> value;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty() && value != nullptr; }
};

class GlmCheckpointReader {
public:
    ~GlmCheckpointReader();
    GlmCheckpointReader(GlmCheckpointReader&&) = delete;
    GlmCheckpointReader& operator=(GlmCheckpointReader&&) = delete;
    GlmCheckpointReader(const GlmCheckpointReader&) = delete;
    GlmCheckpointReader& operator=(const GlmCheckpointReader&) = delete;

    [[nodiscard]] static GlmCheckpointOpenResult open(
        std::string model_directory, bool require_read_only = false);

    [[nodiscard]] const GlmManifestTensor* find(std::string_view name) const noexcept;
    [[nodiscard]] ParseResult<std::vector<std::byte>> read(
        std::string_view name, std::uint64_t maximum_bytes) const;
    [[nodiscard]] ParseResult<std::vector<std::byte>> read_slice(
        std::string_view name, std::uint64_t relative_offset,
        std::uint64_t bytes) const;
    [[nodiscard]] ParseResult<std::vector<float>> read_f32(
        std::string_view name, std::uint64_t maximum_elements) const;
    [[nodiscard]] ParseResult<std::vector<float>> read_f32_row(
        std::string_view name, std::uint64_t row) const;

    [[nodiscard]] const GlmIndexManifest& manifest() const noexcept { return manifest_; }
    [[nodiscard]] const std::string& model_directory() const noexcept { return model_directory_; }
    [[nodiscard]] CheckpointReadStats stats() const noexcept;

private:
    GlmCheckpointReader() = default;
    [[nodiscard]] ParseResult<std::vector<std::byte>> pread_tensor(
        const GlmManifestTensor& tensor, std::uint64_t relative_offset,
        std::uint64_t bytes) const;

    std::string model_directory_;
    GlmIndexManifest manifest_;
    std::unordered_map<std::string_view, std::size_t> tensors_;
    std::unordered_map<std::string, int> shard_fds_;
    mutable std::atomic<std::uint64_t> read_calls_{};
    mutable std::atomic<std::uint64_t> read_bytes_{};
    mutable std::atomic<std::uint64_t> read_nanoseconds_{};
};

// Loads one checkpoint linear directly into a selected GPU. `base_name` omits
// the final .weight or compressed-tensors component suffix.
[[nodiscard]] ValidationResult load_glm_cuda_linear(
    const GlmCheckpointReader& checkpoint, std::string_view base_name,
    std::uint64_t expected_rows, std::uint64_t expected_columns,
    int device, CudaBackend& backend, CudaWeight& output);

}  // namespace strata
