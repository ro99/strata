#pragma once

#include "strata/checkpoint_io.hpp"
#include "strata/compressed_tensors.hpp"
#include "strata/kimi_k3_manifest.hpp"
#include "strata/safetensors.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace strata {

class KimiCheckpointReader;

struct KimiCheckpointOpenResult {
    std::unique_ptr<KimiCheckpointReader> value;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept {
        return errors.empty() && value != nullptr;
    }
};

// Where one routed expert's three MXFP4 modules live. The runtime stages
// experts by this descriptor rather than by name lookup: at 896 experts over 92
// layers there are 247,296 modules and a per-read hash lookup is not free.
struct KimiExpertLocation {
    std::string_view shard;
    std::uint64_t packed_offset{};
    std::uint64_t packed_bytes{};
    std::uint64_t scale_offset{};
    std::uint64_t scale_bytes{};
};

struct KimiExpertModules {
    // Gate (w1), up (w3), and down (w2), in the order the LatentMoE block uses
    // them.
    KimiExpertLocation gate;
    KimiExpertLocation up;
    KimiExpertLocation down;
};

class KimiCheckpointReader {
public:
    [[nodiscard]] static KimiCheckpointOpenResult open(
        std::string model_directory);

    [[nodiscard]] const KimiManifestTensor* find(
        std::string_view name) const noexcept;
    [[nodiscard]] const KimiIndexManifest& manifest() const noexcept {
        return manifest_;
    }
    [[nodiscard]] const KimiK3Config& config() const noexcept { return config_; }

    // Bounded positional reads. Nothing here mmaps and nothing writes: bytes go
    // shard -> caller buffer, and the page cache pages a read produces are
    // clean, so no model byte can reach a swap or spill file by this path.
    [[nodiscard]] ParseResult<std::vector<std::byte>> read(
        std::string_view name, std::uint64_t maximum_bytes) const;
    [[nodiscard]] ValidationResult read_into(std::string_view name,
                                             std::span<std::byte> destination) const;
    // Decodes a plain BF16 or F32 tensor to F32.
    [[nodiscard]] ParseResult<std::vector<float>> read_f32(
        std::string_view name, std::uint64_t maximum_elements) const;

    // Locates one routed expert's three modules. Returns false when the layer
    // is dense or either index is out of range.
    [[nodiscard]] bool expert_modules(std::uint32_t layer, std::uint32_t expert,
                                      KimiExpertModules& modules) const noexcept;
    // Reads one routed expert module and dequantizes it into `output`, which
    // must hold `rows * columns` values.
    [[nodiscard]] ValidationResult read_expert_module_f32(
        const KimiExpertLocation& location, std::uint64_t rows,
        std::uint64_t columns, std::span<float> output) const;

    [[nodiscard]] std::span<const KimiManifestTensor> tensors() const noexcept {
        return manifest_.tensors;
    }
    [[nodiscard]] const std::string& model_directory() const noexcept {
        return model_directory_;
    }
    // Bytes one expert costs in the source: three packed payloads plus three
    // scale payloads. This is the unit the storage tier reads and the unit the
    // cost model counts.
    [[nodiscard]] static constexpr std::uint64_t expert_source_bytes() noexcept {
        return kExpertSourceBytes;
    }

private:
    KimiCheckpointReader() = default;
    [[nodiscard]] const KimiManifestTensor* expert_tensor(
        std::uint32_t layer, std::uint32_t expert,
        std::string_view module, bool scale) const noexcept;

    // 3072*1792 + 3072*1792 + 3584*1536 packed, plus 3072*112 + 3072*112 +
    // 3584*96 of E8M0 scales.
    static constexpr std::uint64_t kExpertSourceBytes = 17'547'264ULL;

    std::string model_directory_;
    KimiK3Config config_;
    KimiIndexManifest manifest_;
    std::unordered_map<std::string_view, std::size_t> by_name_;
    // Dense index over (layer, expert, module) so expert staging costs an
    // arithmetic offset rather than three string builds and three hash lookups.
    std::vector<std::size_t> expert_index_;
    CheckpointShardSet shards_;
};

[[nodiscard]] CompressedTensorLayout kimi_expert_layout(
    std::uint64_t rows, std::uint64_t columns) noexcept;
[[nodiscard]] QuantizedWeightSpec kimi_expert_quantization() noexcept;

}  // namespace strata
