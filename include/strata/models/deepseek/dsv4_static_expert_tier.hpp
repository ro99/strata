#pragma once

#include "strata/device/cuda_backend.hpp"
#include "strata/models/deepseek/deepseek_checkpoint.hpp"
#include "strata/models/deepseek/dsv4_expert_residency.hpp"
#include "strata/platform/result.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace strata {

// Routed experts held in one device's VRAM and computed there.
//
// Decode reads 3.449 GB of routed-expert weight per token from host DRAM at
// 0.282 ms per expert. The same expert computed from VRAM on a 3090 costs
// 0.128 ms, so every expert that fits is worth moving.
//
// The obstacle was ordering, not arithmetic. Routing is decided inside a
// cudaLaunchHostFunc callback where CUDA calls are illegal, and the first
// implementation blocked that callback waiting on a worker thread, which
// stalled the whole CUDA context (experiment 0124).
//
// This version never blocks. Host functions are stream-ordered, so kernels
// enqueued behind the callback observe its writes. The callback writes the
// experts it is *not* going to compute into a pinned selection slot -- a plain
// store, legal from a host function -- and the tier kernels behind it look
// those up in a table built once at load and accumulate into the same rank
// partial the existing join already consumes.
//
// This class owns only the load-time side: choosing what to hold, loading it,
// and installing it into the backend's table. The decode path talks to the
// backend directly.
class Dsv4StaticExpertTier {
public:
    Dsv4StaticExpertTier() = default;

    Dsv4StaticExpertTier(Dsv4StaticExpertTier&&) = delete;
    Dsv4StaticExpertTier& operator=(Dsv4StaticExpertTier&&) = delete;
    Dsv4StaticExpertTier(const Dsv4StaticExpertTier&) = delete;
    Dsv4StaticExpertTier& operator=(const Dsv4StaticExpertTier&) = delete;

    // Loads the plan's triplets onto `device` and installs them in the
    // backend's tier table. `slice_offset`/`slice_stride` take a disjoint share
    // of one ranking so several devices split the hottest experts rather than
    // duplicating them. A tier that admits nothing is legal and never serves.
    [[nodiscard]] ValidationResult initialize(
        int device, CudaBackend& backend,
        const Dsv4CheckpointReader& checkpoint,
        Dsv4ExpertResidencyPlan plan, std::uint64_t vram_budget_bytes,
        std::size_t slice_offset = 0U, std::size_t slice_stride = 1U);

    [[nodiscard]] bool active() const noexcept { return active_; }
    [[nodiscard]] int device() const noexcept { return device_; }
    [[nodiscard]] std::size_t admitted() const noexcept { return admitted_; }
    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] bool resident(std::uint32_t layer,
                                std::uint32_t expert) const noexcept {
        return active_ && plan_.resident(layer, expert);
    }

private:
    int device_{-1};
    bool active_{};
    std::size_t admitted_{};
    std::uint64_t bytes_{};
    Dsv4ExpertResidencyPlan plan_;
    // The tier owns its weights for the life of the process; the backend table
    // holds raw pointers into them.
    std::vector<CudaWeight> weights_;
};

}  // namespace strata
