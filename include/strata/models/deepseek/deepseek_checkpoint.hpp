#pragma once

#include "strata/device/cuda_backend.hpp"
#include "strata/platform/checkpoint_io.hpp"
#include "strata/models/deepseek/deepseek_manifest.hpp"

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
    [[nodiscard]] ValidationResult read_into(
        std::string_view name, std::span<std::byte> destination,
        Dsv4CheckpointReadStats* local_stats = nullptr) const;
    [[nodiscard]] ValidationResult read_slice_into(
        std::string_view name, std::uint64_t relative_offset,
        std::span<std::byte> destination,
        Dsv4CheckpointReadStats* local_stats = nullptr) const;
    [[nodiscard]] ParseResult<std::vector<std::byte>> read_slice(
        std::string_view name, std::uint64_t relative_offset,
        std::uint64_t bytes) const;
    [[nodiscard]] ParseResult<std::vector<float>> read_f32(
        std::string_view name, std::uint64_t maximum_elements) const;

    [[nodiscard]] const Dsv4IndexManifest& manifest() const noexcept { return manifest_; }
    [[nodiscard]] Dsv4CheckpointReadStats stats() const noexcept;

private:
    Dsv4CheckpointReader() = default;
    [[nodiscard]] ParseResult<std::vector<std::byte>> pread_tensor(
        const Dsv4ManifestTensor& tensor, std::uint64_t relative_offset,
        std::uint64_t bytes) const;
    [[nodiscard]] ValidationResult pread_tensor_into(
        const Dsv4ManifestTensor& tensor, std::uint64_t relative_offset,
        std::span<std::byte> destination,
        Dsv4CheckpointReadStats* local_stats = nullptr) const;

    std::string model_directory_;
    Dsv4IndexManifest manifest_;
    std::unordered_map<std::string_view, std::size_t> tensors_;
    CheckpointShardSet shards_;
    mutable std::atomic<std::uint64_t> read_calls_{};
    mutable std::atomic<std::uint64_t> read_bytes_{};
    mutable std::atomic<std::uint64_t> read_nanoseconds_{};
};

struct Dsv4ResidentStageStats {
    std::uint64_t tensors{};
    std::uint64_t bytes{};
    std::uint32_t workers{};
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

    // `hugepage_arena` asks the kernel to back the tiled expert arena with
    // transparent hugepages. It is advisory in the strict sense -- the kernel
    // may decline, and the bytes, layout and arithmetic are identical either
    // way -- so a refusal is recorded, not an error. It exists as a parameter
    // rather than an unconditional call so one binary can measure both arms.
    [[nodiscard]] ValidationResult stage(
        const Dsv4CheckpointReader& checkpoint,
        std::uint64_t host_memory_ceiling_bytes,
        std::uint32_t read_workers = 1U,
        bool include_dspark = false,
        bool tiled_experts = false,
        bool hugepage_arena = true);
    // Whether the tiled arena was advised for hugepages and the kernel
    // accepted the advice. Diagnostic only.
    [[nodiscard]] bool arena_hugepages_requested() const noexcept {
        return hugepage_requested_;
    }
    [[nodiscard]] bool arena_hugepages_accepted() const noexcept {
        return hugepage_accepted_;
    }
    [[nodiscard]] std::span<const std::byte> find(std::string_view name) const noexcept;
    [[nodiscard]] std::span<const std::byte> find_tiled_expert(
        std::uint32_t layer, std::uint32_t expert,
        std::uint32_t shard) const noexcept;
    // Page-locks the staged arena so demand uploads DMA out of it directly.
    // Purely a transfer-rate optimization: it changes no bytes and no shape, so
    // a failure is reported and left unpinned rather than aborting the run.
    [[nodiscard]] ValidationResult pin(CudaBackend& backend);
    [[nodiscard]] bool pinned() const noexcept { return pinned_; }
    // True when the routed experts are staged in the host expert's decode
    // layout, which is the only layout they exist in when it is.
    [[nodiscard]] bool tiled_experts() const noexcept { return tiled_experts_; }
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
    CudaBackend* pinned_backend_{};
    bool pinned_{};
    bool tiled_experts_{};
    bool hugepage_requested_{};
    bool hugepage_accepted_{};
    bool complete_{};
};

// `allow_deferred_upload` only permits deferral; whether the upload actually
// defers is decided here, by where the payload came from. A tensor served out
// of the resident arena is copied straight from storage that outlives the
// call, so its transfer may stay in flight. One decoded into a local buffer --
// a checkpoint read, or the wo_a FP8-to-BF16 conversion -- must not, because
// that buffer dies at return. The caller owes
// CudaBackend::synchronize_uploads() on every device it deferred to.
[[nodiscard]] ValidationResult load_dsv4_cuda_linear(
    const Dsv4CheckpointReader& checkpoint,
    const Dsv4ResidentWeightStore* resident_weights,
    std::string_view base_name, std::uint64_t expected_rows,
    std::uint64_t expected_columns, int device, CudaBackend& backend,
    CudaWeight& output, bool allow_deferred_upload = false);

}  // namespace strata
