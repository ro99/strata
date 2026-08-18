#include "strata/dsv4_rank_local_topology.hpp"

#include "strata/hardware_profile.hpp"

#include <algorithm>
#include <iterator>
#include <limits>
#include <string>

namespace strata {
namespace {

// Saturating so a malformed component cannot wrap the total into a passing
// value. Admission must fail loudly on nonsense, not silently admit it.
[[nodiscard]] std::uint64_t add_saturating(
    std::uint64_t left, std::uint64_t right) noexcept {
    const auto limit = std::numeric_limits<std::uint64_t>::max();
    if (left > limit - right) return limit;
    return left + right;
}

}  // namespace

const char* dsv4_decode_topology_name(Dsv4DecodeTopology topology) noexcept {
    switch (topology) {
        case Dsv4DecodeTopology::Centralized: return "centralized";
        case Dsv4DecodeTopology::RankLocalTp2: return "rank_local_tp2";
    }
    return "unknown";
}

std::uint64_t Dsv4RankLocalDeviceAccount::fixed_total() const noexcept {
    std::uint64_t total = 0U;
    total = add_saturating(total, initial_device_usage_bytes);
    total = add_saturating(total, rank_local_weight_bytes);
    total = add_saturating(total, centralized_spine_bytes);
    total = add_saturating(total, workspace_bytes);
    total = add_saturating(total, kv_capacity_bytes);
    total = add_saturating(total, nccl_buffer_bytes);
    total = add_saturating(total, head_buffer_bytes);
    return total;
}

std::uint64_t Dsv4RankLocalDeviceAccount::total() const noexcept {
    return add_saturating(fixed_total(), expert_cache_bytes);
}

std::uint64_t Dsv4RankLocalHostAccount::total() const noexcept {
    std::uint64_t total = 0U;
    total = add_saturating(total, routed_cpu_storage_bytes);
    total = add_saturating(total, host_parameter_bytes);
    total = add_saturating(total, kv_state_bytes);
    total = add_saturating(total, host_workspace_bytes);
    return total;
}

ValidationResult plan_dsv4_rank_local_cpus(
    const NumaTopology& topology, std::size_t minimum_cpus_per_rank,
    std::array<std::vector<int>, kDsv4RankLocalWorld>& rank_cpus) {
    ValidationResult result;
    for (auto& cpus : rank_cpus) cpus.clear();

    if (topology.nodes < static_cast<int>(kDsv4RankLocalWorld)) {
        result.errors.emplace_back(
            "rank-local decode requires at least " +
            std::to_string(kDsv4RankLocalWorld) + " NUMA nodes, found " +
            std::to_string(topology.nodes));
        return result;
    }
    if (topology.node_cpus.size() < kDsv4RankLocalWorld) {
        result.errors.emplace_back(
            "NUMA topology reports " + std::to_string(topology.nodes) +
            " nodes but enumerates only " +
            std::to_string(topology.node_cpus.size()) + " CPU lists");
        return result;
    }

    // Node order, so the rank-to-node mapping is stable across runs.
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        const auto& cpus = topology.node_cpus[rank];
        if (cpus.size() < minimum_cpus_per_rank) {
            result.errors.emplace_back(
                "NUMA node " + std::to_string(rank) + " assigns " +
                std::to_string(cpus.size()) + " CPUs to rank " +
                std::to_string(rank) + ", below the required " +
                std::to_string(minimum_cpus_per_rank));
            continue;
        }
        // The accepted M3 operating point uses exactly 24 workers per rank.
        // Passing every logical CPU from a larger NUMA node changes that
        // measured resource shape (and can add SMT contention) while still
        // appearing to satisfy the minimum. Keep admission tied to the
        // calibrated pool width.
        rank_cpus[rank].assign(
            cpus.begin(),
            cpus.begin() + static_cast<std::ptrdiff_t>(minimum_cpus_per_rank));
    }
    if (!result.ok()) {
        for (auto& cpus : rank_cpus) cpus.clear();
        return result;
    }

    // Two ranks sharing a CPU would make the two pools contend for the same
    // cores, which silently destroys the routed-CPU throughput the topology
    // exists to gain.
    std::vector<int> overlap;
    std::set_intersection(rank_cpus[0].begin(), rank_cpus[0].end(),
                          rank_cpus[1].begin(), rank_cpus[1].end(),
                          std::back_inserter(overlap));
    if (!overlap.empty()) {
        result.errors.emplace_back(
            "rank CPU sets overlap on " + std::to_string(overlap.size()) +
            " CPUs; each rank must own a disjoint NUMA node");
        for (auto& cpus : rank_cpus) cpus.clear();
    }
    return result;
}

Dsv4RankLocalAdmission admit_dsv4_rank_local(
    const Dsv4RankLocalAdmissionRequest& request,
    const NumaTopology& topology) {
    Dsv4RankLocalAdmission result;

    // Build capability first: requesting the topology without NCCL must fail
    // here, before any model loading, rather than after weights are resident.
    if (!request.nccl_available) {
        result.errors.emplace_back(
            "rank-local decode was requested but this build has no NCCL "
            "support; rebuild with -DSTRATA_ENABLE_NCCL=ON");
    }

    if (request.devices.size() != kDsv4RankLocalWorld) {
        result.errors.emplace_back(
            "rank-local decode requires exactly " +
            std::to_string(kDsv4RankLocalWorld) + " CUDA devices, got " +
            std::to_string(request.devices.size()));
    } else if (request.devices[0] == request.devices[1]) {
        result.errors.emplace_back(
            "rank-local decode requires two distinct CUDA devices, both are " +
            std::to_string(request.devices[0]));
    }

    if (request.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice) {
        result.errors.emplace_back(
            "rank-local decode requires the physical-device DSV4 KV mode");
    }
    if (!request.supported_checkpoint) {
        result.errors.emplace_back(
            "checkpoint is not a supported DSV4 index for rank-local decode");
    }
    if (!request.fp4_routed_experts) {
        result.errors.emplace_back(
            "rank-local decode requires FP4 routed experts");
    }
    // Rank-local decode serves the model's full declared context, including
    // the sparse-indexer regime above kDsv4RankLocalSparseIndexerThreshold.
    // Only a request beyond the model's own maximum is rejected.
    if (request.maximum_context_tokens != 0U &&
        request.active_context_tokens > request.maximum_context_tokens) {
        result.errors.emplace_back(
            "rank-local decode was asked for " +
            std::to_string(request.active_context_tokens) +
            " active context tokens, above the model maximum " +
            std::to_string(request.maximum_context_tokens));
    }
    if (request.layer_count != kDsv4RankLocalLayerCount) {
        result.errors.emplace_back(
            "rank-local decode requires " +
            std::to_string(kDsv4RankLocalLayerCount) + " layers, checkpoint "
            "declares " + std::to_string(request.layer_count));
    }

    // Width comes from the machine, not from a constant. The old
    // kDsv4RankLocalMinimumCpusPerRank was doing two jobs at once -- the floor
    // below which a rank cannot work, and the pool width to actually assign --
    // and 24 was only ever correct for both on the box it was measured on.
    // The width is now the smallest node's CPU count (24 there, unchanged),
    // and the constant is only the floor.
    std::size_t width = 0U;
    for (const auto& cpus : topology.node_cpus) {
        if (cpus.empty()) continue;
        if (width == 0U || cpus.size() < width) width = cpus.size();
    }
    width = std::max(width, kDsv4RankLocalMinimumCpusPerRank);
    auto cpu_plan =
        plan_dsv4_rank_local_cpus(topology, width, result.rank_cpus);
    if (!cpu_plan.ok()) {
        result.errors.insert(result.errors.end(), cpu_plan.errors.begin(),
                             cpu_plan.errors.end());
    }

    // Per-device residency. The fixed components must fit outright; whatever
    // remains becomes the capped centralized prefill expert cache. A request
    // asking for more cache than remains is capped, not rejected -- the cache
    // is a prefill performance term, and prefill is outside the measured
    // decode window. Every other component overrunning is a hard rejection.
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        const auto& account = request.device[rank];
        const auto fixed = account.fixed_total();
        // Zero means "derive from this device". A caller that supplies a
        // figure is stating an explicit operator ceiling and gets it verbatim.
        const auto ceiling =
            request.per_device_vram_ceiling_bytes != 0U
                ? request.per_device_vram_ceiling_bytes
                : dsv4_rank_local_vram_ceiling(account.device_total_bytes);
        if (fixed > ceiling) {
            result.errors.emplace_back(
                "rank " + std::to_string(rank) + " fixed residency " +
                std::to_string(fixed) + " B exceeds the per-GPU ceiling " +
                std::to_string(ceiling) + " B by " +
                std::to_string(fixed - ceiling) + " B (rank-local weights " +
                std::to_string(account.rank_local_weight_bytes) +
                " B, initial device usage " +
                std::to_string(account.initial_device_usage_bytes) +
                " B, centralized spine " +
                std::to_string(account.centralized_spine_bytes) +
                " B, workspace " + std::to_string(account.workspace_bytes) +
                " B, KV " + std::to_string(account.kv_capacity_bytes) +
                " B, NCCL " + std::to_string(account.nccl_buffer_bytes) +
                " B, head " + std::to_string(account.head_buffer_bytes) + " B)");
            continue;
        }
        const auto remaining = ceiling - fixed;
        result.expert_cache_capacity_bytes[rank] =
            std::min(account.expert_cache_bytes, remaining);
        result.device_total_bytes[rank] =
            fixed + result.expert_cache_capacity_bytes[rank];
    }

    result.host_total_bytes = request.host.total();
    const auto host_ceiling = request.host_rss_ceiling_bytes != 0U
        ? request.host_rss_ceiling_bytes
        : dsv4_rank_local_host_rss_ceiling();
    if (result.host_total_bytes > host_ceiling) {
        result.errors.emplace_back(
            "host residency " + std::to_string(result.host_total_bytes) +
            " B exceeds the host ceiling " +
            std::to_string(host_ceiling) + " B by " +
            std::to_string(result.host_total_bytes -
                           host_ceiling) + " B");
    }

    if (!result.ok()) {
        // Fail closed: no caller may act on a partial plan.
        result.expert_cache_capacity_bytes = {};
        result.device_total_bytes = {};
        for (auto& cpus : result.rank_cpus) cpus.clear();
    }
    return result;
}


std::uint64_t dsv4_rank_local_host_rss_ceiling() noexcept {
    return host_hardware_profile().host_usable_bytes();
}

}  // namespace strata
