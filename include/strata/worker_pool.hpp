#pragma once

#include "strata/result.hpp"

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
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

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata
