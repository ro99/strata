#include "test.hpp"

#include "strata/platform/worker_pool.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

// `parallel_for` dispatches through one of two queues depending on whether the
// runner count reaches the pool width, so every case here is run at both a
// full-width and a narrower-than-the-pool shape. See the note on `Impl` in
// src/worker_pool.cpp for why the two paths exist.

namespace {

constexpr std::size_t kWorkers = 8;

}  // namespace

TEST_CASE("worker pool runs every index exactly once at both dispatch widths") {
    strata::HostWorkerPool pool(kWorkers);
    const std::array<std::size_t, 4> widths{kWorkers * 4U, kWorkers,
                                            kWorkers - 3U, std::size_t{1}};
    for (const std::size_t tasks : widths) {
        std::vector<std::atomic<int>> visits(tasks);
        for (auto& visit : visits) visit.store(0);
        const auto result = pool.parallel_for(tasks, [&visits](std::size_t index) {
            visits[index].fetch_add(1);
        });
        REQUIRE(result.ok());
        for (std::size_t index = 0; index < tasks; ++index) {
            REQUIRE(visits[index].load() == 1);
        }
    }
}

TEST_CASE("worker pool leaves no task queued between dispatches") {
    // A closure left in a queue by one dispatch would be run by the next, so
    // repeated alternation between the addressed and shared paths is what
    // catches a queue that is not drained.
    strata::HostWorkerPool pool(kWorkers);
    std::atomic<int> total{0};
    for (int round = 0; round < 32; ++round) {
        const std::size_t tasks = (round % 2 == 0) ? kWorkers : kWorkers - 3U;
        const auto result = pool.parallel_for(tasks, [&total](std::size_t) {
            total.fetch_add(1);
        });
        REQUIRE(result.ok());
    }
    REQUIRE(total.load() == 16 * static_cast<int>(kWorkers) +
                            16 * static_cast<int>(kWorkers - 3U));
}

TEST_CASE("worker pool remains reusable while workers idle-spin") {
    strata::HostWorkerPool pool(kWorkers, std::chrono::milliseconds(1));
    std::atomic<int> total{0};
    for (int round = 0; round < 32; ++round) {
        const auto result = pool.parallel_for_addressed(
            kWorkers, [&total](std::size_t) { total.fetch_add(1); });
        REQUIRE(result.ok());
    }
    REQUIRE(total.load() == 32 * static_cast<int>(kWorkers));
}

TEST_CASE("worker pool spreads a full-width dispatch across distinct threads") {
    // The point of the pool: a full-width dispatch must not serialize onto one
    // worker. Each task parks until every other task has arrived, so the barrier
    // can only clear if the runners really are concurrent.
    strata::HostWorkerPool pool(kWorkers);
    std::atomic<std::size_t> arrived{0};
    const auto result = pool.parallel_for(kWorkers, [&arrived](std::size_t) {
        arrived.fetch_add(1);
        while (arrived.load() < kWorkers) std::this_thread::yield();
    });
    REQUIRE(result.ok());
    REQUIRE(arrived.load() == kWorkers);
}

TEST_CASE("worker pool reports a task exception without losing the barrier") {
    strata::HostWorkerPool pool(kWorkers);
    const std::array<std::size_t, 2> widths{kWorkers, kWorkers - 3U};
    for (const std::size_t tasks : widths) {
        const auto result = pool.parallel_for(tasks, [](std::size_t index) {
            if (index == 0U) throw std::runtime_error("task failed");
        });
        REQUIRE(!result.ok());
    }
    // The pool stays usable: a throwing dispatch must not leave the barrier or
    // either queue in a state that strands the next one.
    std::atomic<int> total{0};
    const auto after = pool.parallel_for(kWorkers, [&total](std::size_t) {
        total.fetch_add(1);
    });
    REQUIRE(after.ok());
    REQUIRE(total.load() == static_cast<int>(kWorkers));
}

TEST_CASE("worker pool accepts an empty dispatch and rejects a null operation") {
    strata::HostWorkerPool pool(kWorkers);
    REQUIRE(pool.size() == kWorkers);
    REQUIRE(pool.parallel_for(0U, [](std::size_t) {}).ok());
    REQUIRE(!pool.parallel_for(4U, std::function<void(std::size_t)>{}).ok());
}
