#pragma once

#include "strata/checkpoint.hpp"
#include "strata/checkpoint_io.hpp"
#include "strata/cuda_backend.hpp"
#include "strata/laguna_ops.hpp"
#include "strata/safetensors.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace strata {

enum class LagunaTensorEncoding : std::uint8_t {
    Plain,
    Nvfp4Group16,
    Mxfp4Group32,
};

enum class LagunaCheckpointFormat : std::uint8_t {
    Nvfp4Mixed,
    Mxfp4Group32,
};

struct LagunaTensor {
    std::string name;
    std::string shard;
    SafetensorsDtype dtype{SafetensorsDtype::Other};
    std::vector<std::uint64_t> shape;
    std::uint64_t offset{};
    std::uint64_t bytes{};
};

// One resolved linear module. `encoding` decides which component tensors are
// populated: Plain uses `weight` alone, Nvfp4Group16 uses the packed/scale/
// global-scale triplet, and Mxfp4Group32 uses packed/scale without a global
// scale.
struct LagunaLinear {
    LagunaTensorEncoding encoding{LagunaTensorEncoding::Plain};
    std::uint64_t rows{};
    std::uint64_t columns{};
    const LagunaTensor* weight{};
    const LagunaTensor* packed{};
    const LagunaTensor* scale{};
    const LagunaTensor* global_scale{};

    [[nodiscard]] std::uint64_t source_bytes() const noexcept {
        if (encoding == LagunaTensorEncoding::Plain) {
            return weight == nullptr ? 0U : weight->bytes;
        }
        return (packed == nullptr ? 0U : packed->bytes) +
               (scale == nullptr ? 0U : scale->bytes);
    }
};

class LagunaCheckpointReader;

struct LagunaCheckpointOpenResult {
    std::unique_ptr<LagunaCheckpointReader> value;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept {
        return errors.empty() && value != nullptr;
    }
};

class LagunaCheckpointReader {
public:
    ~LagunaCheckpointReader();
    LagunaCheckpointReader(LagunaCheckpointReader&&) = delete;
    LagunaCheckpointReader& operator=(LagunaCheckpointReader&&) = delete;
    LagunaCheckpointReader(const LagunaCheckpointReader&) = delete;
    LagunaCheckpointReader& operator=(const LagunaCheckpointReader&) = delete;

    // Opens and validates either pinned Laguna S 2.1 compressed checkpoint.
    // Every declared module must be present with the format's exact shape and
    // encoding; there is no partial-load mode or precision fallback.
    [[nodiscard]] static LagunaCheckpointOpenResult open(
        std::string model_directory);

    [[nodiscard]] const LagunaTensor* find(std::string_view name) const noexcept;
    // Resolves `base_name` (no component suffix) to a linear of the expected
    // shape, failing when the stored encoding does not match the contract.
    [[nodiscard]] ParseResult<LagunaLinear> linear(
        std::string_view base_name, std::uint64_t rows,
        std::uint64_t columns) const;
    [[nodiscard]] ParseResult<std::vector<std::byte>> read(
        std::string_view name, std::uint64_t maximum_bytes) const;
    [[nodiscard]] ParseResult<std::vector<float>> read_f32(
        std::string_view name, std::uint64_t maximum_elements) const;
    [[nodiscard]] ParseResult<std::vector<float>> read_f32_row(
        std::string_view name, std::uint64_t row) const;
    // Stable read-only view into a lazily mapped shard. The reader owns the
    // mapping; release_mapped_views drops this process's pages while leaving
    // the kernel page cache free to retain them.
    [[nodiscard]] ParseResult<std::span<const std::byte>> view(
        std::string_view name) const;
    void release_mapped_views() const noexcept;
    // Builds the host-side NVFP4 view of one routed expert projection.
    [[nodiscard]] ParseResult<LagunaNvfp4MatrixView> nvfp4_view(
        const LagunaLinear& module) const;

    [[nodiscard]] std::span<const LagunaTensor> tensors() const noexcept {
        return tensors_;
    }
    [[nodiscard]] std::uint64_t shard_file_bytes() const noexcept {
        return shard_file_bytes_;
    }
    [[nodiscard]] LagunaCheckpointFormat format() const noexcept {
        return format_;
    }
    [[nodiscard]] CheckpointReadStats stats() const noexcept;

private:
    LagunaCheckpointReader() = default;

    std::string model_directory_;
    LagunaCheckpointFormat format_{LagunaCheckpointFormat::Nvfp4Mixed};
    std::vector<LagunaTensor> tensors_;
    std::unordered_map<std::string_view, std::size_t> by_name_;
    CheckpointShardSet shards_;
    std::uint64_t shard_file_bytes_{};
    mutable std::atomic<std::uint64_t> read_calls_{};
    mutable std::atomic<std::uint64_t> read_bytes_{};
    mutable std::atomic<std::uint64_t> read_nanoseconds_{};
    struct ShardMapping {
        std::byte* address{};
        std::uint64_t bytes{};
    };
    mutable std::mutex mapping_mutex_;
    mutable std::unordered_map<std::string, ShardMapping> mappings_;
};

// Uploads one resolved linear straight to a device. Plain modules go up as
// BF16; NVFP4 modules keep their packed nibbles and E4M3 group scales, and the
// per-tensor global scale rides in the descriptor.
[[nodiscard]] ValidationResult load_laguna_cuda_linear(
    const LagunaCheckpointReader& checkpoint, const LagunaLinear& module,
    int device, CudaBackend& backend, CudaWeight& output);

}  // namespace strata
