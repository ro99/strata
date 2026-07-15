#include "strata/worker_pool.hpp"

#include <atomic>
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace strata {

struct HostWorkerPool::Impl {
    struct Completion {
        std::mutex mutex;
        std::condition_variable ready;
        std::size_t remaining{};
        std::atomic<std::size_t> next{};
        std::size_t tasks{};
        std::exception_ptr error;
    };

    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::function<void()>> queue;
    std::vector<std::thread> workers;
    bool stopping{};

    explicit Impl(std::size_t count) {
        workers.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            workers.emplace_back([this, index] {
                // This host exposes 28 physical Broadwell cores as CPUs 0-27,
                // then socket-0 siblings at 28-41 and socket-1 siblings at
                // 42-55. Keep the physical pass contiguous, but interleave the
                // optional SMT pass so extra workers consume both NUMA memory
                // controllers rather than piling onto socket 0.
                std::size_t cpu = index;
                if (index >= 28U && index < 56U) {
                    const auto sibling = index - 28U;
                    cpu = (sibling & 1U) == 0U
                        ? 28U + sibling / 2U
                        : 42U + sibling / 2U;
                }
                cpu_set_t affinity;
                CPU_ZERO(&affinity);
                CPU_SET(static_cast<int>(cpu), &affinity);
                static_cast<void>(pthread_setaffinity_np(
                    pthread_self(), sizeof(affinity), &affinity));
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock lock(mutex);
                        ready.wait(lock, [this] { return stopping || !queue.empty(); });
                        if (stopping && queue.empty()) return;
                        task = std::move(queue.front());
                        queue.pop_front();
                    }
                    task();
                }
            });
        }
    }

    ~Impl() {
        {
            std::scoped_lock lock(mutex);
            stopping = true;
        }
        ready.notify_all();
        for (auto& worker : workers) worker.join();
    }
};

HostWorkerPool::HostWorkerPool(std::size_t workers)
    : impl_(std::make_unique<Impl>(workers)) {}
HostWorkerPool::~HostWorkerPool() = default;
HostWorkerPool::HostWorkerPool(HostWorkerPool&&) noexcept = default;
HostWorkerPool& HostWorkerPool::operator=(HostWorkerPool&&) noexcept = default;

std::size_t HostWorkerPool::size() const noexcept {
    return impl_ == nullptr ? 0U : impl_->workers.size();
}

ValidationResult HostWorkerPool::parallel_for(
    std::size_t tasks, const std::function<void(std::size_t)>& operation) {
    ValidationResult result;
    if (tasks == 0U) return result;
    if (impl_ == nullptr || impl_->workers.empty() || !operation) {
        result.errors.emplace_back("host worker pool is not available");
        return result;
    }
    auto completion = std::make_shared<Impl::Completion>();
    const auto runners = std::min(tasks, impl_->workers.size());
    completion->remaining = runners;
    completion->tasks = tasks;
    {
        std::scoped_lock queue_lock(impl_->mutex);
        if (impl_->stopping) {
            result.errors.emplace_back("host worker pool is stopping");
            return result;
        }
        for (std::size_t runner = 0; runner < runners; ++runner) {
            static_cast<void>(runner);
            impl_->queue.emplace_back([completion, &operation] {
                try {
                    for (;;) {
                        const auto index = completion->next.fetch_add(
                            1U, std::memory_order_relaxed);
                        if (index >= completion->tasks) break;
                        operation(index);
                    }
                } catch (...) {
                    std::scoped_lock error_lock(completion->mutex);
                    if (completion->error == nullptr) {
                        completion->error = std::current_exception();
                    }
                }
                {
                    std::scoped_lock completion_lock(completion->mutex);
                    --completion->remaining;
                }
                completion->ready.notify_one();
            });
        }
    }
    impl_->ready.notify_all();
    {
        std::unique_lock lock(completion->mutex);
        completion->ready.wait(lock, [&completion] {
            return completion->remaining == 0U;
        });
    }
    if (completion->error != nullptr) {
        result.errors.emplace_back("host worker task raised an exception");
    }
    return result;
}

}  // namespace strata
