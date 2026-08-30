#include "test.hpp"

#include "strata/platform/worker_pool.hpp"
#include "strata/platform/hardware_profile.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <numeric>
#include <sched.h>
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

TEST_CASE("blocked dispatch visits every index exactly once") {
    // Reordering is the whole point, so the contract is coverage, not order:
    // every index runs, none runs twice, whatever the block size.
    strata::HostWorkerPool pool(4U);
    for (const std::size_t block : {1U, 3U, 64U, 1000U}) {
        constexpr std::size_t tasks = 500U;
        std::vector<std::atomic<int>> seen(tasks);
        for (auto& count : seen) count.store(0);
        const auto ran = pool.parallel_for_blocked(
            tasks, block, [&](std::size_t index) { seen[index].fetch_add(1); });
        REQUIRE(ran.ok());
        for (std::size_t index = 0U; index < tasks; ++index) {
            REQUIRE(seen[index].load() == 1);
        }
    }
}

TEST_CASE("blocked dispatch hands out contiguous runs to a single runner") {
    // The reason this method exists: a runner must walk memory contiguously
    // rather than with a stride of the pool width. With one runner the whole
    // range is one run; the assertion is that indices arrive in order within a
    // claim, which is what the prefetcher sees.
    strata::HostWorkerPool pool(1U);
    std::vector<std::size_t> order;
    const auto ran = pool.parallel_for_blocked(
        16U, 4U, [&](std::size_t index) { order.push_back(index); });
    REQUIRE(ran.ok());
    REQUIRE(order.size() == 16U);
    for (std::size_t index = 0U; index < order.size(); ++index) {
        REQUIRE(order[index] == index);
    }
}

TEST_CASE("a zero block is treated as one rather than spinning forever") {
    strata::HostWorkerPool pool(2U);
    std::atomic<int> ran{0};
    const auto status = pool.parallel_for_blocked(
        8U, 0U, [&](std::size_t) { ran.fetch_add(1); });
    REQUIRE(status.ok());
    REQUIRE(ran.load() == 8);
}

TEST_CASE("owned dispatch keeps partitions on their declared NUMA nodes") {
    const auto& hardware = strata::host_hardware_profile();
    std::vector<int> cpus;
    std::vector<int> nodes;
    for (const auto cpu : hardware.usable_cpu_ids) {
        const auto node = hardware.numa.node_of_cpu(cpu);
        if (std::find(nodes.begin(), nodes.end(), node) != nodes.end()) continue;
        cpus.push_back(cpu);
        nodes.push_back(node);
    }
    REQUIRE(!cpus.empty());
    strata::HostWorkerPool pool(cpus);
    REQUIRE(pool.worker_nodes().size() == cpus.size());
    std::vector<std::atomic<int>> visits(nodes.size() * 17U);
    for (auto& visit : visits) visit.store(0);
    std::vector<std::size_t> tasks(nodes.size(), 17U);
    const auto ran = pool.parallel_for_owned(
        tasks, nodes, [&](std::size_t partition, std::size_t index) {
            REQUIRE(hardware.numa.node_of_cpu(sched_getcpu()) ==
                    nodes[partition]);
            visits[partition * 17U + index].fetch_add(1);
        });
    REQUIRE(ran.ok());
    for (const auto& visit : visits) REQUIRE(visit.load() == 1);
}

TEST_CASE("owned dispatch rejects an incomplete partition map") {
    strata::HostWorkerPool pool(1U);
    const std::array<std::size_t, 2U> tasks{1U, 1U};
    const std::array<int, 1U> nodes{0};
    REQUIRE(!pool.parallel_for_owned(
        tasks, nodes, [](std::size_t, std::size_t) {}).ok());
}
