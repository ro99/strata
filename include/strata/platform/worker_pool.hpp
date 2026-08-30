#pragma once

#include "strata/platform/result.hpp"

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace strata {

class HostWorkerPool {
public:
    explicit HostWorkerPool(
        std::size_t workers,
        std::chrono::microseconds idle_spin = std::chrono::microseconds::zero());
    explicit HostWorkerPool(
        std::vector<int> cpus,
        std::chrono::microseconds idle_spin = std::chrono::microseconds::zero());
    ~HostWorkerPool();
    HostWorkerPool(HostWorkerPool&&) noexcept;
    HostWorkerPool& operator=(HostWorkerPool&&) noexcept;
    HostWorkerPool(const HostWorkerPool&) = delete;
    HostWorkerPool& operator=(const HostWorkerPool&) = delete;

    [[nodiscard]] std::size_t size() const noexcept;
    // Runs task i on worker i, always. `parallel_for` lets any runner steal
    // any task, which is what a placement-agnostic dispatch wants; when the
    // caller has already decided which core a task must run on -- because the
    // memory it reads is bound to that core's NUMA node -- stealing is exactly
    // the thing that must not happen. `tasks` must not exceed size().
    [[nodiscard]] ValidationResult parallel_for_addressed(
        std::size_t tasks, const std::function<void(std::size_t)>& operation);

    [[nodiscard]] ValidationResult parallel_for(
        std::size_t tasks, const std::function<void(std::size_t)>& operation);

    // Same dispatch, but each runner claims `block` consecutive indices at a
    // time instead of one. For work whose index maps to a memory offset, the
    // difference is not scheduling overhead but access pattern: claiming single
    // indices from a shared counter interleaves N runners across one array, so
    // each walks it with an N-element stride and the hardware prefetcher sees N
    // strided streams instead of N contiguous ones.
    //
    // Measured on cold checkpoint reads (experiment 0198): the same 28 workers
    // over the same bytes moved 0.69 GB/s taking single expert rows and
    // 1.96 GB/s taking contiguous blocks, with device request size rising from
    // 41.2 KiB to 73.0 KiB. Reordering independent tasks changes no arithmetic;
    // callers whose tasks accumulate into shared state must not use this.
    [[nodiscard]] ValidationResult parallel_for_blocked(
        std::size_t tasks, std::size_t block,
        const std::function<void(std::size_t)>& operation);

    // Runs each partition only on workers pinned to its owning NUMA node.
    // Work is divided into one contiguous range per same-node worker and is
    // never stolen across nodes. Pools without explicit, discoverable CPU
    // affinity reject non-empty owned dispatches instead of silently losing
    // locality.
    [[nodiscard]] ValidationResult parallel_for_owned(
        std::span<const std::size_t> partition_tasks,
        std::span<const int> partition_nodes,
        const std::function<void(std::size_t, std::size_t)>& operation);

    [[nodiscard]] std::span<const int> worker_nodes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata
