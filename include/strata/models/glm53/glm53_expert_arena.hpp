#pragma once

#include "strata/models/glm53/glm53_manifest.hpp"
#include "strata/platform/result.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace strata {

class Glm53CheckpointReader;

struct Glm53ExpertArenaStats {
    std::uint64_t logical_bytes{};
    std::uint64_t mapped_bytes{};
    std::uint64_t anonymous_huge_bytes{};
    std::uint64_t tensors{};
    std::uint64_t shards{};
    std::vector<std::uint64_t> numa_bytes;
    double stage_seconds{};
    bool hugepage_accepted{};
    bool hugepage_prefaulted{};
    bool numa_balanced{};
};

// Process-lifetime replacement for the file-mapped GLM routed and shared
// expert payloads, including the optional MTP block's experts when MTP is
// enabled at load. The bytes are copied without re-encoding into one anonymous
// mapping, so transparent huge pages can cover them. Before first touch the
// arena is striped evenly across the available NUMA nodes because every host
// worker reads every selected expert; no page has a single NUMA owner under
// that access pattern.
class Glm53ExpertArena {
public:
    Glm53ExpertArena() = default;
    ~Glm53ExpertArena();
    Glm53ExpertArena(const Glm53ExpertArena&) = delete;
    Glm53ExpertArena& operator=(const Glm53ExpertArena&) = delete;
    Glm53ExpertArena(Glm53ExpertArena&&) = delete;
    Glm53ExpertArena& operator=(Glm53ExpertArena&&) = delete;

    // Includes alignment and the final huge-page-sized mapping tail. Zero
    // means the manifest contains no host expert payloads or its size
    // overflows. This is the exact capacity figure used before mmap.
    [[nodiscard]] static std::uint64_t required_bytes(
        const Glm53IndexManifest& manifest,
        bool include_mtp_experts = false) noexcept;

    [[nodiscard]] ValidationResult stage(
        const Glm53CheckpointReader& checkpoint,
        std::uint64_t host_memory_ceiling_bytes,
        bool include_mtp_experts = false);

    [[nodiscard]] std::span<const std::byte> find(
        std::string_view name) const noexcept;
    [[nodiscard]] const Glm53ExpertArenaStats& stats() const noexcept {
        return stats_;
    }
    [[nodiscard]] bool complete() const noexcept { return complete_; }

private:
    struct Extent {
        std::uint64_t offset{};
        std::uint64_t bytes{};
    };

    void release() noexcept;

    std::byte* base_{};
    std::uint64_t mapped_bytes_{};
    std::unordered_map<std::string, Extent> extents_;
    Glm53ExpertArenaStats stats_;
    bool complete_{};
};

}  // namespace strata
