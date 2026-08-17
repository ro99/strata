#include "strata/dsv4_host_moe_executor.hpp"

#include "strata/numa_topology.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sched.h>
#include <string>
#include <utility>

namespace strata {
namespace {

constexpr std::size_t kHidden = kDeepSeekV4ExecutionContract.hidden_size;
constexpr std::size_t kIntermediate =
    kDeepSeekV4ExecutionContract.expert_intermediate_size;
constexpr std::size_t kExperts = kDeepSeekV4ExecutionContract.routed_experts;
constexpr std::size_t kTopK = kDeepSeekV4ExecutionContract.experts_per_token;
constexpr std::size_t kShards = 2U;
constexpr std::size_t kBlock = 32U;
constexpr std::size_t kDefaultWorkers = 48U;
constexpr std::size_t kLocalIntermediate = kIntermediate / kShards;

[[nodiscard]] std::uint64_t elapsed_nanoseconds(
    std::chrono::steady_clock::time_point begin) noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin).count());
}

}  // namespace

Dsv4HostMoeExecutor::Dsv4HostMoeExecutor(
    std::size_t shard_index, bool both_shards, std::vector<int> cpus)
    : shard_index_(shard_index), both_shards_(both_shards),
      cpus_(std::move(cpus)) {
    if (!both_shards_ && shard_index_ >= kShards) shard_index_ = 0U;
    const auto worker_count = both_shards_ ? kDefaultWorkers : cpus_.size();
    if (worker_count != 0U) {
        if (cpus_.empty()) {
            workers_ = std::make_unique<HostWorkerPool>(
                worker_count, std::chrono::milliseconds(1));
        } else {
            workers_ = std::make_unique<HostWorkerPool>(
                cpus_, std::chrono::milliseconds(1));
        }
    }
}

Dsv4HostMoeExecutor::~Dsv4HostMoeExecutor() = default;

std::size_t Dsv4HostMoeExecutor::workers() const noexcept {
    return workers_ == nullptr ? 0U : workers_->size();
}

ValidationResult Dsv4HostMoeExecutor::initialize() {
    ValidationResult result;
    if (workers_ == nullptr || workers_->size() == 0U ||
        workers_->size() > lane_next_.size()) {
        result.errors.emplace_back("DeepSeek host-MoE executor has no valid workers");
        return result;
    }
    const auto topology = NumaTopology::detect();
    lane_nodes_.assign(workers_->size(), -1);
    lane_positions_.assign(workers_->size(), 0U);
    for (auto& lanes : node_lanes_) lanes.clear();
    result = workers_->parallel_for_addressed(
        workers_->size(), [&](std::size_t lane) {
            if (both_shards_) {
                lane_nodes_[lane] = topology.node_of_cpu(sched_getcpu());
            } else {
                lane_nodes_[lane] = static_cast<int>(shard_index_);
            }
        });
    if (!result.ok()) return result;
    for (std::size_t lane = 0U; lane < workers_->size(); ++lane) {
        const auto node = lane_nodes_[lane];
        if (node < 0 || node >= static_cast<int>(kShards)) {
            result.errors.emplace_back(
                "DeepSeek host-MoE worker is outside the two NUMA nodes");
            return result;
        }
        auto& node_lanes = node_lanes_[static_cast<std::size_t>(node)];
        lane_positions_[lane] = node_lanes.size();
        node_lanes.push_back(lane);
    }
    if (both_shards_ &&
        (node_lanes_[0].size() != workers_->size() / kShards ||
         node_lanes_[1].size() != workers_->size() / kShards)) {
        result.errors.emplace_back(
            "DeepSeek host-MoE production executor requires 24 workers per NUMA node");
        return result;
    }
    if (!both_shards_ && node_lanes_[shard_index_].size() != workers_->size()) {
        result.errors.emplace_back(
            "DeepSeek rank-local executor workers are not NUMA-local");
        return result;
    }
    const auto shards = output_shards();
    tiled_activation_.assign(shards * kTopK * kLocalIntermediate, 0.0F);
    tiled_routed_.assign(shards * kTopK * kHidden, 0.0F);
    initialized_ = true;
    return result;
}

ValidationResult Dsv4HostMoeExecutor::dispatch_ranges(
    std::uint64_t tasks, bool steal,
    const std::function<void(std::size_t, std::uint64_t)>& operation) {
    ValidationResult result;
    if (!initialized_ || workers_ == nullptr || !operation || tasks == 0U) {
        result.errors.emplace_back("DeepSeek host-MoE executor dispatch is invalid");
        return result;
    }
    const auto lanes = workers_->size();
    const auto shards = output_shards();
    const auto shard_for_lane = [&](std::size_t lane) {
        return both_shards_
            ? static_cast<std::size_t>(lane_nodes_[lane])
            : std::size_t{0U};
    };
    if (steal) {
        for (std::size_t lane = 0U; lane < lanes; ++lane) {
            const auto shard = shard_for_lane(lane);
            const auto& node_lanes = node_lanes_[
                static_cast<std::size_t>(lane_nodes_[lane])];
            const auto position = lane_positions_[lane];
            const auto workers = node_lanes.size();
            lane_next_[lane].store(tasks * position / workers,
                                   std::memory_order_relaxed);
            lane_end_[lane] = tasks * (position + 1U) / workers;
            if (shard >= shards) {
                result.errors.emplace_back("DeepSeek executor lane shard is invalid");
                return result;
            }
        }
    }
    result = workers_->parallel_for_addressed(
        lanes, [&](std::size_t lane) {
            const auto shard = shard_for_lane(lane);
            if (shard >= shards) return;
            const auto node = static_cast<std::size_t>(lane_nodes_[lane]);
            const auto& node_lanes = node_lanes_[node];
            const auto position = lane_positions_[lane];
            if (!steal) {
                const auto begin = tasks * position / node_lanes.size();
                const auto end = tasks * (position + 1U) / node_lanes.size();
                for (auto task = begin; task < end; ++task) {
                    operation(shard, task);
                }
                return;
            }
            const auto claim = [&](std::size_t owner) {
                const auto task = lane_next_[owner].fetch_add(
                    1U, std::memory_order_relaxed);
                if (task >= lane_end_[owner]) return false;
                operation(shard, task);
                return true;
            };
            while (claim(lane)) {}
            for (;;) {
                bool stole = false;
                for (std::size_t offset = 1U;
                     offset < node_lanes.size(); ++offset) {
                    const auto owner = node_lanes[
                        (position + offset) % node_lanes.size()];
                    if (claim(owner)) {
                        stole = true;
                        break;
                    }
                }
                if (!stole) break;
            }
        });
    if (result.ok() && steal) {
        for (std::size_t lane = 0U; lane < lanes; ++lane) {
            if (lane_next_[lane].load(std::memory_order_relaxed) <
                lane_end_[lane]) {
                result.errors.emplace_back(
                    "DeepSeek host-MoE left a lane range unfinished");
                break;
            }
        }
    }
    return result;
}

ValidationResult Dsv4HostMoeExecutor::run(
    std::uint32_t layer, const Dsv4Route& route,
    std::span<const float> input, const Dsv4ResidentWeightStore& resident,
    std::span<float> rank_partials, Dsv4HostMoePhaseTimings* timings) {
    ValidationResult result;
    if (!initialized_ || layer >= kDeepSeekV4ExecutionContract.layer_count ||
        input.size() != kHidden || route.experts.size() != kTopK ||
        route.weights.size() != kTopK ||
        rank_partials.size() != output_shards() * kHidden) {
        result.errors.emplace_back("DeepSeek host-MoE executor shapes are invalid");
        return result;
    }
    for (std::size_t rank = 0U; rank < kTopK; ++rank) {
        if (route.experts[rank] >= kExperts ||
            !std::isfinite(route.weights[rank])) {
            result.errors.emplace_back("DeepSeek host-MoE route is invalid");
            return result;
        }
    }
    std::array<std::array<Dsv4TiledExpertWeights, kTopK>, kShards> tiled{};
    for (std::size_t local_shard = 0U; local_shard < output_shards();
         ++local_shard) {
        const auto source_shard = shard_index_ + local_shard;
        if (source_shard >= kShards) {
            result.errors.emplace_back("DeepSeek host-MoE source shard is invalid");
            return result;
        }
        for (std::size_t rank = 0U; rank < kTopK; ++rank) {
            auto viewed = dsv4_tiled_expert_weights(
                resident.find_tiled_expert(layer, route.experts[rank],
                                            static_cast<std::uint32_t>(source_shard)),
                kHidden, kIntermediate, kShards);
            if (!viewed.ok()) {
                result.errors.insert(result.errors.end(), viewed.errors.begin(),
                                     viewed.errors.end());
                return result;
            }
            tiled[local_shard][rank] = viewed.value;
        }
    }
    const auto started = std::chrono::steady_clock::now();
    const auto intermediate_blocks = kLocalIntermediate / kBlock;
    auto phase = std::chrono::steady_clock::now();
    result = dispatch_ranges(
        kTopK * intermediate_blocks, true,
        [&](std::size_t shard, std::uint64_t task) {
            const auto rank = static_cast<std::size_t>(task / intermediate_blocks);
            const auto offset = (task % intermediate_blocks) * kBlock;
            std::array<float, kBlock> gate{};
            std::array<float, kBlock> up{};
            for (std::size_t half = 0U; half < 2U; ++half) {
                dsv4_tiled_expert_matvec16(
                    std::span<float, 16U>(gate.data() + half * 16U, 16U), input,
                    tiled[shard][rank].w13_packed,
                    tiled[shard][rank].w13_scales, 2U * kLocalIntermediate,
                    offset + half * 16U);
                dsv4_tiled_expert_matvec16(
                    std::span<float, 16U>(up.data() + half * 16U, 16U), input,
                    tiled[shard][rank].w13_packed,
                    tiled[shard][rank].w13_scales, 2U * kLocalIntermediate,
                    kLocalIntermediate + offset + half * 16U);
            }
            auto* destination = tiled_activation_.data() +
                (shard * kTopK + rank) * kLocalIntermediate + offset;
            for (std::size_t index = 0U; index < kBlock; ++index) {
                destination[index] = gate[index] /
                    (1.0F + std::exp(-gate[index])) * up[index];
            }
        });
    if (timings != nullptr) {
        timings->gate_up_nanoseconds = elapsed_nanoseconds(phase);
    }
    phase = std::chrono::steady_clock::now();
    if (result.ok()) {
        const auto hidden_blocks = kHidden / kBlock;
        result = dispatch_ranges(
            kTopK * hidden_blocks, true,
            [&](std::size_t shard, std::uint64_t task) {
                const auto rank = static_cast<std::size_t>(task / hidden_blocks);
                const auto offset = (task % hidden_blocks) * kBlock;
                const auto source = std::span<const float>(tiled_activation_)
                    .subspan((shard * kTopK + rank) * kLocalIntermediate,
                             kLocalIntermediate);
                for (std::size_t half = 0U; half < 2U; ++half) {
                    dsv4_tiled_expert_matvec16(
                        std::span<float, 16U>(
                            tiled_routed_.data() +
                                (shard * kTopK + rank) * kHidden + offset + half * 16U,
                            16U), source, tiled[shard][rank].w2_packed,
                        tiled[shard][rank].w2_scales, kHidden,
                        offset + half * 16U);
                }
            });
    }
    if (timings != nullptr) timings->down_nanoseconds = elapsed_nanoseconds(phase);
    phase = std::chrono::steady_clock::now();
    if (result.ok()) {
        const auto hidden_blocks = kHidden / kBlock;
        result = dispatch_ranges(
            hidden_blocks, false,
            [&](std::size_t shard, std::uint64_t task) {
                const auto offset = task * kBlock;
                auto* destination = rank_partials.data() + shard * kHidden + offset;
                for (std::size_t index = 0U; index < kBlock; ++index) {
                    destination[index] = tiled_routed_[
                        (shard * kTopK) * kHidden + offset + index] *
                        route.weights[0];
                }
                for (std::size_t rank = 1U; rank < kTopK; ++rank) {
                    const auto* source = tiled_routed_.data() +
                        (shard * kTopK + rank) * kHidden + offset;
                    for (std::size_t index = 0U; index < kBlock; ++index) {
                        destination[index] = std::fma(
                            source[index], route.weights[rank], destination[index]);
                    }
                }
            });
    }
    if (timings != nullptr) {
        timings->reduce_nanoseconds = elapsed_nanoseconds(phase);
        timings->total_nanoseconds = elapsed_nanoseconds(started);
    }
    return result;
}

}  // namespace strata
