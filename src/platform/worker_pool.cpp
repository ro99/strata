#include "strata/platform/worker_pool.hpp"

#include <atomic>
#include <algorithm>
#include <chrono>
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
    // A shared queue, plus one addressed queue per worker. Which one a dispatch
    // uses depends on its width, because the two shapes want opposite things.
    //
    // A dispatch *narrower* than the pool is a free-for-all on a shared queue:
    // any thread can win a task. Workers 0-27 are pinned to this host's 28
    // physical cores and 28-55 to their SMT siblings, so a sixteen-lane
    // dispatch onto a fifty-six thread pool could land two lanes on one
    // physical core while another sat idle. Measured on the Kimi-K3 MoE block,
    // that cost exactly 2.0x -- 2.89 s/token against 1.43 -- which is what
    // sixteen lanes sharing eight cores predicts. Addressing task i to worker i
    // makes a narrow dispatch use the low-indexed workers, the physical ones.
    //
    // A *full-width* dispatch gets no placement benefit from that -- every
    // worker is used either way -- and addressing it is strictly worse. On a
    // shared queue a fast worker can retire two closures while a slow one never
    // wakes, so completion waits on the fastest N wake-ups; addressed, it waits
    // on all N. Measured on an empty `parallel_for`, 28 tasks on 28 workers:
    // 61.35 us shared against 105.58 us addressed. DeepSeek pays that barrier
    // 86 times a step, so addressing it unconditionally would cost ~3.8 ms of a
    // 209 ms step (experiments 0051, 0054) to buy nothing.
    std::deque<std::function<void()>> queue;
    std::vector<std::deque<std::function<void()>>> queues;
    std::vector<std::thread> workers;
    std::atomic<std::uint64_t> dispatches{};
    std::chrono::microseconds idle_spin;
    std::vector<int> affinity_cpus;
    bool stopping{};

    explicit Impl(std::size_t count, std::chrono::microseconds spin,
                  std::vector<int> affinity_cpu_list = {})
        : idle_spin(spin), affinity_cpus(std::move(affinity_cpu_list)) {
        if (!this->affinity_cpus.empty() && this->affinity_cpus.size() != count) {
            this->affinity_cpus.clear();
        }
        queues.resize(count);
        workers.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            workers.emplace_back([this, index] {
                // This host exposes 28 physical Broadwell cores as CPUs 0-27,
                // then socket-0 siblings at 28-41 and socket-1 siblings at
                // 42-55. Keep the physical pass contiguous, but interleave the
                // optional SMT pass so extra workers consume both NUMA memory
                // controllers rather than piling onto socket 0.
                std::size_t cpu = this->affinity_cpus.empty()
                    ? index : static_cast<std::size_t>(this->affinity_cpus[index]);
                if (this->affinity_cpus.empty() && index >= 28U && index < 56U) {
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
                    std::uint64_t task_dispatch{};
                    {
                        std::unique_lock lock(mutex);
                        ready.wait(lock, [this, index] {
                            return stopping || !queues[index].empty() ||
                                   !queue.empty();
                        });
                        // The addressed queue first: when a narrow dispatch
                        // named this worker, that is the task it must run.
                        auto& own = queues[index];
                        if (!own.empty()) {
                            task = std::move(own.front());
                            own.pop_front();
                        } else if (!queue.empty()) {
                            task = std::move(queue.front());
                            queue.pop_front();
                        } else {
                            return;  // stopping, both queues drained
                        }
                        task_dispatch = dispatches.load(std::memory_order_acquire);
                    }
                    task();
                    if (idle_spin > std::chrono::microseconds::zero()) {
                        const auto deadline =
                            std::chrono::steady_clock::now() + idle_spin;
                        std::uint32_t spins = 0U;
                        while (dispatches.load(std::memory_order_acquire) ==
                               task_dispatch) {
#if defined(__x86_64__) || defined(__i386__)
                            __builtin_ia32_pause();
#else
                            std::this_thread::yield();
#endif
                            if ((++spins & 255U) == 0U &&
                                std::chrono::steady_clock::now() >= deadline) {
                                break;
                            }
                        }
                    }
                }
            });
        }
    }

    ~Impl() {
        {
            std::scoped_lock lock(mutex);
            stopping = true;
            dispatches.fetch_add(1U, std::memory_order_release);
        }
        ready.notify_all();
        for (auto& worker : workers) worker.join();
    }
};

HostWorkerPool::HostWorkerPool(std::size_t workers,
                               std::chrono::microseconds idle_spin)
    : impl_(std::make_unique<Impl>(workers, idle_spin)) {}
HostWorkerPool::HostWorkerPool(std::vector<int> affinity_cpu_list,
                               std::chrono::microseconds idle_spin)
    : impl_(std::make_unique<Impl>(affinity_cpu_list.size(), idle_spin,
                                   std::move(affinity_cpu_list))) {}
HostWorkerPool::~HostWorkerPool() = default;
HostWorkerPool::HostWorkerPool(HostWorkerPool&&) noexcept = default;
HostWorkerPool& HostWorkerPool::operator=(HostWorkerPool&&) noexcept = default;

std::size_t HostWorkerPool::size() const noexcept {
    return impl_ == nullptr ? 0U : impl_->workers.size();
}

ValidationResult HostWorkerPool::parallel_for_addressed(
    std::size_t tasks, const std::function<void(std::size_t)>& operation) {
    ValidationResult result;
    if (tasks == 0U) return result;
    if (impl_ == nullptr || impl_->workers.empty() || !operation ||
        tasks > impl_->workers.size()) {
        result.errors.emplace_back(
            "host worker pool cannot address this dispatch");
        return result;
    }
    auto completion = std::make_shared<Impl::Completion>();
    completion->remaining = tasks;
    completion->tasks = tasks;
    {
        std::scoped_lock queue_lock(impl_->mutex);
        if (impl_->stopping) {
            result.errors.emplace_back("host worker pool is stopping");
            return result;
        }
        for (std::size_t task = 0; task < tasks; ++task) {
            impl_->queues[task].emplace_back([completion, &operation, task] {
                try {
                    operation(task);
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
        impl_->dispatches.fetch_add(1U, std::memory_order_release);
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
        // Address the runners only when the dispatch is narrower than the pool,
        // which is the only shape where core placement is in question. See the
        // two-queue note on `Impl`.
        const bool addressed = runners < impl_->workers.size();
        for (std::size_t runner = 0; runner < runners; ++runner) {
            auto& target = addressed ? impl_->queues[runner] : impl_->queue;
            target.emplace_back([completion, &operation] {
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
        impl_->dispatches.fetch_add(1U, std::memory_order_release);
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
