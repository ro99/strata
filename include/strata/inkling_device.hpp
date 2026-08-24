#pragma once

#include "strata/cuda_backend.hpp"
#include "strata/inkling_checkpoint.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace strata {

// One routed expert staged on a device. The NVFP4 checkpoint stores gate and
// up interleaved in one w13 tensor, so staging splits alternating rows. The
// MXFP4 checkpoint already stores three separate projections and bypasses that
// transform.
struct InklingDeviceExpert {
    CudaWeight gate;
    CudaWeight up;
    CudaWeight down;

    [[nodiscard]] std::uint64_t device_bytes() const noexcept {
        return gate.device_bytes() + up.device_bytes() + down.device_bytes();
    }
};

struct InklingCacheStats {
    std::uint64_t hits{};
    std::uint64_t misses{};
    std::uint64_t evictions{};
    std::vector<std::uint64_t> used_bytes;
    std::vector<std::uint64_t> capacity_bytes;
    std::vector<std::uint64_t> peak_bytes;
    std::uint64_t stage_nanoseconds{};
    std::uint64_t staged_bytes{};
};

// A per-device LRU of routed experts backed by the host mapping. Capacity is
// what the resident spine leaves behind; the routed set cannot fit in the
// supported device budget, so the cache converts repeat routing into avoided
// H2D.
class InklingExpertCache {
public:
    InklingExpertCache(const InklingCheckpointReader& checkpoint,
                       CudaBackend& backend, std::vector<int> devices,
                       std::vector<std::uint64_t> capacities,
                       bool direct_mapped_mxfp4,
                       bool defer_mapped_mxfp4_uploads);
    ~InklingExpertCache();
    InklingExpertCache(InklingExpertCache&&) = delete;
    InklingExpertCache& operator=(InklingExpertCache&&) = delete;
    InklingExpertCache(const InklingExpertCache&) = delete;
    InklingExpertCache& operator=(const InklingExpertCache&) = delete;

    // Stages one routed expert and leases it. The returned pointers stay valid
    // until the matching release, which is what keeps an eviction from pulling
    // a weight out from under an in-flight device command.
    [[nodiscard]] ParseResult<const InklingDeviceExpert*> acquire(
        std::size_t device_slot, std::uint32_t layer, std::uint32_t expert,
        const InklingExpertStack& gate, const InklingExpertStack& up,
        const InklingExpertStack& down);
    void release(std::size_t device_slot, std::uint32_t layer,
                 std::uint32_t expert) noexcept;

    [[nodiscard]] InklingCacheStats stats() const;

private:
    struct Entry {
        InklingDeviceExpert expert;
        std::uint64_t last_use{};
        std::uint64_t leases{};
        std::uint64_t bytes{};
    };
    // Per-device page-locked staging. NVFP4 needs it to de-interleave gate/up.
    // MXFP4 is already split into canonical stacks and uploads directly from
    // its resident mapping: an interleaved real-model A/B found that the extra
    // memcpy into this scratch slowed miss service by 1.38x.
    struct Scratch {
        std::vector<std::byte> weights;
        std::vector<std::byte> scales;
        bool registered{};
    };
    struct State {
        mutable std::mutex mutex;
        Scratch scratch;
        std::unordered_map<std::uint64_t, Entry> entries;
        std::uint64_t capacity{};
        std::uint64_t used{};
        std::uint64_t peak{};
        std::uint64_t clock{};
        std::uint64_t hits{};
        std::uint64_t misses{};
        std::uint64_t evictions{};
        std::uint64_t stage_nanoseconds{};
        std::uint64_t staged_bytes{};
    };

    [[nodiscard]] static std::uint64_t key_of(std::uint32_t layer,
                                              std::uint32_t expert) noexcept {
        return (static_cast<std::uint64_t>(layer) << 32U) | expert;
    }
    // Frees least-recently-used unleased entries until `bytes` fits.
    [[nodiscard]] bool evict_locked(State& state, std::uint64_t bytes);

    const InklingCheckpointReader& checkpoint_;
    CudaBackend& backend_;
    std::vector<int> devices_;
    std::vector<std::unique_ptr<State>> states_;
    bool direct_mapped_mxfp4_{};
    bool defer_mapped_mxfp4_uploads_{};
};

// Uploads one BF16 linear straight to a device.
[[nodiscard]] ValidationResult load_inkling_cuda_linear(
    const InklingCheckpointReader& checkpoint, const InklingLinear& module,
    int device, CudaBackend& backend, CudaWeight& output);

// Uploads the gate or up half of an interleaved BF16 w13 tensor. `slice` is a
// row offset in units of whole [rows, columns] matrices, used to pick one of
// the stacked shared experts.
[[nodiscard]] ValidationResult load_inkling_cuda_interleaved_half(
    const InklingCheckpointReader& checkpoint, const std::string& name,
    std::uint64_t slice, std::uint64_t rows, std::uint64_t columns, bool up,
    int device, CudaBackend& backend, CudaWeight& output);

}  // namespace strata
