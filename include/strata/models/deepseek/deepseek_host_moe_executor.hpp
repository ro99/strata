#pragma once

#include "strata/models/deepseek/deepseek_checkpoint.hpp"
#include "strata/models/deepseek/deepseek_host_expert.hpp"
#include "strata/models/deepseek/deepseek_ops.hpp"
#include "strata/models/common/model_adapter.hpp"
#include "strata/platform/worker_pool.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace strata {

struct Dsv4HostMoePhaseTimings {
    std::uint64_t gate_up_nanoseconds{};
    std::uint64_t down_nanoseconds{};
    std::uint64_t reduce_nanoseconds{};
    std::uint64_t total_nanoseconds{};
};

// Exact production tiled CPU-MoE executor. The default constructor shape is
// the live runtime's 48-worker, one-millisecond hot-worker pool over the two
// NUMA nodes. A rank-local instance may provide one node's 24 CPU IDs and a
// shard index; it then runs the same addressed dispatch and same-node stealing
// policy over that one transformed intermediate shard.
class Dsv4HostMoeExecutor {
public:
    // `shard_index` is 0 or 1 for a rank-local executor and is ignored when
    // `both_shards` is true. `cpus` is empty for the production pool's
    // established host affinity mapping.
    explicit Dsv4HostMoeExecutor(
        std::size_t shard_index = 0U, bool both_shards = true,
        std::vector<int> cpus = {});
    ~Dsv4HostMoeExecutor();

    Dsv4HostMoeExecutor(Dsv4HostMoeExecutor&&) = delete;
    Dsv4HostMoeExecutor& operator=(Dsv4HostMoeExecutor&&) = delete;
    Dsv4HostMoeExecutor(const Dsv4HostMoeExecutor&) = delete;
    Dsv4HostMoeExecutor& operator=(const Dsv4HostMoeExecutor&) = delete;

    [[nodiscard]] ValidationResult initialize();
    [[nodiscard]] std::size_t output_shards() const noexcept {
        return both_shards_ ? 2U : 1U;
    }
    [[nodiscard]] std::size_t workers() const noexcept;

    // `route` is already selected using the target router contract. The
    // executor changes no route, coefficient, precision, or arithmetic order.
    // `rank_partials` is H or 2*H floats according to output_shards().
    //
    // `skip`, when non-empty, is one flag per top-k slot: a set flag means
    // that expert is served by another tier and this executor must contribute
    // nothing for it -- not read its weights, not accumulate its term. The
    // caller owes the skipped terms; the sum over all six is unchanged, only
    // its grouping. Skipping is the whole point of a residency tier, because
    // the cost this executor pays is the DRAM read, not the arithmetic.
    [[nodiscard]] ValidationResult run(
        std::uint32_t layer, const Dsv4Route& route,
        std::span<const float> input,
        const Dsv4ResidentWeightStore& resident,
        std::span<float> rank_partials,
        Dsv4HostMoePhaseTimings* timings = nullptr,
        std::span<const bool> skip = {});

private:
    [[nodiscard]] ValidationResult dispatch_ranges(
        std::uint64_t tasks, bool steal,
        const std::function<void(std::size_t, std::uint64_t)>& operation);

    std::size_t shard_index_{};
    bool both_shards_{};
    std::vector<int> cpus_;
    std::unique_ptr<HostWorkerPool> workers_;
    std::vector<int> lane_nodes_;
    std::vector<std::size_t> lane_positions_;
    std::array<std::vector<std::size_t>, 2U> node_lanes_;
    std::vector<float> tiled_activation_;
    std::vector<float> tiled_routed_;
    std::array<std::atomic<std::uint64_t>, 48U> lane_next_{};
    std::array<std::uint64_t, 48U> lane_end_{};
    bool initialized_{};
};

}  // namespace strata
