#pragma once

#include "strata/model.hpp"

#include <cstddef>
#include <functional>
#include <memory>

namespace strata {

class HostWorkerPool {
public:
    explicit HostWorkerPool(std::size_t workers);
    ~HostWorkerPool();
    HostWorkerPool(HostWorkerPool&&) noexcept;
    HostWorkerPool& operator=(HostWorkerPool&&) noexcept;
    HostWorkerPool(const HostWorkerPool&) = delete;
    HostWorkerPool& operator=(const HostWorkerPool&) = delete;

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] ValidationResult parallel_for(
        std::size_t tasks, const std::function<void(std::size_t)>& operation);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata
