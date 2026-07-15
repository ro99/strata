#pragma once

#include "strata/cuda_backend.hpp"
#include "strata/deepseek_manifest.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace strata {

struct Dsv4CheckpointReadStats {
    std::uint64_t calls{};
    std::uint64_t bytes{};
    std::uint64_t nanoseconds{};
};

class Dsv4CheckpointReader;

struct Dsv4CheckpointOpenResult {
    std::unique_ptr<Dsv4CheckpointReader> value;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty() && value != nullptr; }
};

class Dsv4CheckpointReader {
public:
    ~Dsv4CheckpointReader();
    Dsv4CheckpointReader(Dsv4CheckpointReader&&) = delete;
    Dsv4CheckpointReader& operator=(Dsv4CheckpointReader&&) = delete;
    Dsv4CheckpointReader(const Dsv4CheckpointReader&) = delete;
    Dsv4CheckpointReader& operator=(const Dsv4CheckpointReader&) = delete;

    [[nodiscard]] static Dsv4CheckpointOpenResult open(
        std::string model_directory, bool require_read_only = false);

    [[nodiscard]] const Dsv4ManifestTensor* find(std::string_view name) const noexcept;
    [[nodiscard]] ParseResult<std::vector<std::byte>> read(
        std::string_view name, std::uint64_t maximum_bytes) const;
    [[nodiscard]] ParseResult<std::vector<std::byte>> read_slice(
        std::string_view name, std::uint64_t relative_offset,
        std::uint64_t bytes) const;
    [[nodiscard]] ParseResult<std::vector<float>> read_f32(
        std::string_view name, std::uint64_t maximum_elements) const;
    [[nodiscard]] ParseResult<std::vector<float>> read_f32_row(
        std::string_view name, std::uint64_t row) const;
    [[nodiscard]] ParseResult<std::vector<std::uint32_t>> read_u32_row_from_i64(
        std::string_view name, std::uint64_t row) const;

    [[nodiscard]] const Dsv4IndexManifest& manifest() const noexcept { return manifest_; }
    [[nodiscard]] const std::string& model_directory() const noexcept {
        return model_directory_;
    }
    [[nodiscard]] Dsv4CheckpointReadStats stats() const noexcept;

private:
    Dsv4CheckpointReader() = default;
    [[nodiscard]] ParseResult<std::vector<std::byte>> pread_tensor(
        const Dsv4ManifestTensor& tensor, std::uint64_t relative_offset,
        std::uint64_t bytes) const;

    std::string model_directory_;
    Dsv4IndexManifest manifest_;
    std::unordered_map<std::string_view, std::size_t> tensors_;
    std::unordered_map<std::string, int> shard_fds_;
    mutable std::atomic<std::uint64_t> read_calls_{};
    mutable std::atomic<std::uint64_t> read_bytes_{};
    mutable std::atomic<std::uint64_t> read_nanoseconds_{};
};

struct Dsv4ResidentStageStats {
    std::uint64_t tensors{};
    std::uint64_t bytes{};
    double seconds{};
};

// Anonymous-memory canonical tier for routed experts and the embedding table.
// Once stage() succeeds, decode-time expert uploads and embedding gathers make
// no checkpoint reads.
class Dsv4ResidentWeightStore {
public:
    Dsv4ResidentWeightStore() = default;
    ~Dsv4ResidentWeightStore();
    Dsv4ResidentWeightStore(Dsv4ResidentWeightStore&&) = delete;
    Dsv4ResidentWeightStore& operator=(Dsv4ResidentWeightStore&&) = delete;
    Dsv4ResidentWeightStore(const Dsv4ResidentWeightStore&) = delete;
    Dsv4ResidentWeightStore& operator=(const Dsv4ResidentWeightStore&) = delete;

    [[nodiscard]] ValidationResult stage(
        const Dsv4CheckpointReader& checkpoint,
        std::uint64_t host_memory_ceiling_bytes,
        bool include_dspark = false);
    [[nodiscard]] std::span<const std::byte> find(std::string_view name) const noexcept;
    [[nodiscard]] bool complete() const noexcept { return complete_; }
    [[nodiscard]] Dsv4ResidentStageStats stats() const noexcept { return stats_; }

private:
    struct Extent {
        std::uint64_t offset{};
        std::uint64_t bytes{};
    };
    std::byte* arena_{};
    std::uint64_t arena_bytes_{};
    std::unordered_map<std::string_view, Extent> extents_;
    Dsv4ResidentStageStats stats_;
    bool complete_{};
};

[[nodiscard]] ValidationResult load_dsv4_cuda_linear(
    const Dsv4CheckpointReader& checkpoint,
    const Dsv4ResidentWeightStore* resident_weights,
    std::string_view base_name, std::uint64_t expected_rows,
    std::uint64_t expected_columns, int device, CudaBackend& backend,
    CudaWeight& output);

}  // namespace strata
