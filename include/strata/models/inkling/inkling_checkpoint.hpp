#pragma once

#include "strata/models/common/checkpoint.hpp"
#include "strata/platform/checkpoint_io.hpp"
#include "strata/device/cuda_backend.hpp"
#include "strata/models/inkling/inkling_ops.hpp"
#include "strata/platform/safetensors.hpp"

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

enum class InklingTensorEncoding : std::uint8_t {
    Plain,
    Nvfp4Group16,
    Mxfp4Group32,
};

enum class InklingCheckpointFormat : std::uint8_t {
    Nvfp4Mixed,
    Mxfp4Group32,
};

struct InklingTensor {
    std::string name;
    std::string shard;
    SafetensorsDtype dtype{SafetensorsDtype::Other};
    std::vector<std::uint64_t> shape;
    std::uint64_t offset{};
    std::uint64_t bytes{};
};

// One resolved linear. The original NVFP4 checkpoint uses Plain for every
// non-routed module. The MLX MXFP4 checkpoint stores every matrix except the
// router gate as U32-packed E2M1 plus per-32 U8 E8M0 scales.
struct InklingLinear {
    InklingTensorEncoding encoding{InklingTensorEncoding::Plain};
    std::uint64_t rows{};
    std::uint64_t columns{};
    const InklingTensor* weight{};
    const InklingTensor* packed{};
    const InklingTensor* scale{};

    [[nodiscard]] std::uint64_t source_bytes() const noexcept {
        if (encoding == InklingTensorEncoding::Plain) {
            return weight == nullptr ? 0U : weight->bytes;
        }
        return (packed == nullptr ? 0U : packed->bytes) +
               (scale == nullptr ? 0U : scale->bytes);
    }
};

// One layer's routed experts. Inkling stacks all 256 experts of a projection
// into a single tensor rather than shipping one tensor per expert, so a layer
// is three tensors instead of 768. `encoding` decides which members are
// populated: Plain uses `weight`; NVFP4 and MXFP4 use `packed` and `scale`,
// with only NVFP4 carrying `global_scale`.
struct InklingExpertStack {
    InklingTensorEncoding encoding{InklingTensorEncoding::Plain};
    std::uint64_t experts{};
    std::uint64_t rows{};
    std::uint64_t columns{};
    const InklingTensor* weight{};
    const InklingTensor* packed{};
    const InklingTensor* scale{};
    // Per-expert FP32 multipliers, one per expert in the NVFP4 stack.
    const InklingTensor* global_scale{};

    [[nodiscard]] std::uint64_t source_bytes() const noexcept {
        if (encoding == InklingTensorEncoding::Plain) {
            return weight == nullptr ? 0U : weight->bytes;
        }
        return (packed == nullptr ? 0U : packed->bytes) +
               (scale == nullptr ? 0U : scale->bytes);
    }
    // Bytes for one expert's slice, which is the granularity the runtime
    // stages and the residency layer accounts for.
    [[nodiscard]] std::uint64_t expert_bytes() const noexcept {
        return experts == 0U ? 0U : source_bytes() / experts;
    }
};

class InklingCheckpointReader;

struct InklingCheckpointOpenResult {
    std::unique_ptr<InklingCheckpointReader> value;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept {
        return errors.empty() && value != nullptr;
    }
};

class InklingCheckpointReader {
public:
    ~InklingCheckpointReader();
    InklingCheckpointReader(InklingCheckpointReader&&) = delete;
    InklingCheckpointReader& operator=(InklingCheckpointReader&&) = delete;
    InklingCheckpointReader(const InklingCheckpointReader&) = delete;
    InklingCheckpointReader& operator=(const InklingCheckpointReader&) = delete;

    // Opens and validates the checkpoint against the pinned Inkling-Small
    // contract. Every declared module must be present with the declared shape
    // and encoding; there is no partial-load mode.
    [[nodiscard]] static InklingCheckpointOpenResult open(
        std::string model_directory);

    [[nodiscard]] const InklingTensor* find(std::string_view name) const noexcept;
    [[nodiscard]] ParseResult<InklingLinear> linear(
        std::string_view name, std::uint64_t rows, std::uint64_t columns) const;
    // Resolves a stacked routed-expert projection, failing when the stored
    // encoding does not match what the contract declares for that layer.
    [[nodiscard]] ParseResult<InklingExpertStack> expert_stack(
        std::string_view base_name, std::uint32_t layer, std::uint64_t experts,
        std::uint64_t rows, std::uint64_t columns) const;
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
    // Host-side NVFP4 view of one expert inside a stacked projection.
    [[nodiscard]] ParseResult<InklingNvfp4MatrixView> nvfp4_expert_view(
        const InklingExpertStack& stack, std::uint64_t expert) const;
    [[nodiscard]] ParseResult<InklingMxfp4MatrixView> mxfp4_view(
        const InklingLinear& module) const;
    [[nodiscard]] ParseResult<InklingMxfp4MatrixView> mxfp4_expert_view(
        const InklingExpertStack& stack, std::uint64_t expert) const;

    [[nodiscard]] std::span<const InklingTensor> tensors() const noexcept {
        return tensors_;
    }
    [[nodiscard]] std::uint64_t shard_file_bytes() const noexcept {
        return shard_file_bytes_;
    }
    [[nodiscard]] InklingCheckpointFormat format() const noexcept {
        return format_;
    }
    [[nodiscard]] CheckpointReadStats stats() const noexcept;

private:
    InklingCheckpointReader() = default;

    std::string model_directory_;
    InklingCheckpointFormat format_{InklingCheckpointFormat::Nvfp4Mixed};
    std::vector<InklingTensor> tensors_;
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

// Names of the pinned modules, shared by the reader and the runtime so a
// rename cannot drift between validation and loading.
[[nodiscard]] std::string inkling_layer_prefix(std::uint32_t layer);
[[nodiscard]] std::string inkling_mtp_prefix(std::uint32_t depth);

}  // namespace strata
