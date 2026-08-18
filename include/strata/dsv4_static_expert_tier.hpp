#pragma once

#include "strata/cuda_backend.hpp"
#include "strata/deepseek_checkpoint.hpp"
#include "strata/dsv4_expert_residency.hpp"
#include "strata/result.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <unordered_map>
#include <vector>

namespace strata {

// A routed-expert tier held permanently in one device's VRAM.
//
// Rank-local decode reads all 3.449 GB of a token's routed experts from host
// DRAM, which is 63% of the step and runs at 47.4 GB/s against a 76.3 GB/s
// ceiling. This tier moves the hottest triplets off DRAM entirely, onto a
// device that is otherwise idle because it is too small and too slow to be a
// tensor-parallel peer -- and neither of those matters here. The card does not
// need bandwidth parity; it needs to not be DRAM.
//
// The set is chosen offline (experiment 0124) and never changes, which is what
// makes it implementable at all: routing is decided inside a
// `cudaLaunchHostFunc` callback where CUDA calls are forbidden, so a tier whose
// membership could change would have to publish it to the device on every
// change.
//
// Because the callback cannot call CUDA, this class owns a worker thread that
// does. `submit` hands the resident subset over and returns immediately;
// `collect` waits and adds the result in. The worker drives only this device,
// on its own stream, with no dependency on the rank streams, so a callback
// blocked in `collect` cannot deadlock against work queued behind it.
//
// Fail-closed: any error leaves the tier disabled for the rest of the process
// and is reported. There is no silent fallback to the host path, because a
// tier that sometimes serves an expert and sometimes does not would double it
// or drop it.
class Dsv4StaticExpertTier {
public:
    Dsv4StaticExpertTier() = default;
    ~Dsv4StaticExpertTier();

    Dsv4StaticExpertTier(Dsv4StaticExpertTier&&) = delete;
    Dsv4StaticExpertTier& operator=(Dsv4StaticExpertTier&&) = delete;
    Dsv4StaticExpertTier(const Dsv4StaticExpertTier&) = delete;
    Dsv4StaticExpertTier& operator=(const Dsv4StaticExpertTier&) = delete;

    // Loads the plan's triplets onto `device`, truncating to what actually
    // fits after the device's own context and workspace. A tier that admits
    // zero triplets is legal and simply never serves.
    [[nodiscard]] ValidationResult initialize(
        int device, CudaBackend& backend,
        const Dsv4CheckpointReader& checkpoint,
        Dsv4ExpertResidencyPlan plan, std::uint64_t vram_budget_bytes,
        float swiglu_limit);

    [[nodiscard]] bool active() const noexcept { return active_; }
    [[nodiscard]] std::size_t admitted() const noexcept { return admitted_; }
    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] bool resident(std::uint32_t layer,
                                std::uint32_t expert) const noexcept {
        return active_ && plan_.resident(layer, expert);
    }

    // Fills `serve` with the slots this tier owns for `layer` and returns how
    // many. The caller passes the complement to the host executor as its skip
    // mask, so exactly one engine owes each expert.
    [[nodiscard]] std::size_t select(
        std::uint32_t layer, std::span<const std::uint32_t> experts,
        std::span<bool> serve) const noexcept;

    // Hands the selected slots to the worker. Returns immediately so the
    // caller can run the host share concurrently. One submit per collect.
    [[nodiscard]] ValidationResult submit(
        std::uint32_t layer, std::span<const std::uint32_t> experts,
        std::span<const float> weights, std::span<const bool> serve,
        std::span<const float> input);

    // Waits for the submitted work and adds it into `destination`. Adding
    // rather than assigning is deliberate: the host executor has already
    // written its own share there.
    [[nodiscard]] ValidationResult collect(std::span<float> destination);

    struct Stats {
        std::uint64_t submissions{};
        std::uint64_t experts_served{};
        std::uint64_t device_nanoseconds{};
        std::uint64_t wait_nanoseconds{};
    };
    [[nodiscard]] Stats stats() const noexcept;

private:
    struct Triplet {
        CudaWeight w1;
        CudaWeight w3;
        CudaWeight w2;
    };

    void worker_loop();
    void shutdown();

    int device_{-1};
    CudaBackend* backend_{};
    float swiglu_limit_{};
    bool active_{};
    std::size_t admitted_{};
    std::uint64_t bytes_{};
    Dsv4ExpertResidencyPlan plan_;
    std::unordered_map<std::uint64_t, Triplet> triplets_;

    // Worker handshake. `pending_` is set by submit and cleared by collect;
    // the worker owns the device between them.
    std::thread worker_;
    mutable std::mutex mutex_;
    std::condition_variable request_;
    std::condition_variable response_;
    bool stop_{};
    bool pending_{};
    bool complete_{};
    std::vector<CudaDeepSeekMoeExpert> request_experts_;
    std::vector<float> request_weights_;
    std::vector<float> request_input_;
    std::vector<float> response_output_;
    std::vector<std::string> worker_errors_;

    std::atomic<std::uint64_t> submissions_{};
    std::atomic<std::uint64_t> experts_served_{};
    std::atomic<std::uint64_t> device_nanoseconds_{};
    std::atomic<std::uint64_t> wait_nanoseconds_{};
};

}  // namespace strata
