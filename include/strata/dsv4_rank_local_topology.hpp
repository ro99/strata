#pragma once

#include "strata/deepseek_kv_cache.hpp"
#include "strata/numa_topology.hpp"
#include "strata/result.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace strata {

// Decode execution topology. Centralized is the default and is unchanged by
// this feature; RankLocalTp2 is explicit opt-in and fail-closed.
enum class Dsv4DecodeTopology : std::uint8_t {
    Centralized,
    RankLocalTp2,
};

[[nodiscard]] const char* dsv4_decode_topology_name(
    Dsv4DecodeTopology topology) noexcept;

inline constexpr std::size_t kDsv4RankLocalWorld = 2U;
inline constexpr std::uint32_t kDsv4RankLocalLayerCount = 43U;
// Minimum CPUs a NUMA node must contribute to own one rank's routed-expert
// pool. The measured production shape is one node's 24 CPUs per rank.
inline constexpr std::size_t kDsv4RankLocalMinimumCpusPerRank = 24U;
// Program ceilings. These are the declared operating-point limits, not
// hardware capacities; admission rejects rather than exceeding them.
inline constexpr std::uint64_t kDsv4RankLocalPerDeviceVramCeiling =
    21'287'272'448ULL;
inline constexpr std::uint64_t kDsv4RankLocalHostRssCeiling =
    231'928'233'984ULL;

// Per-device residency accounted by component. These are deliberately separate
// fields rather than one total: the landing must be able to say which component
// pushed a device over the ceiling, and the centralized prefill cache is the
// only one that may be capped to make room.
struct Dsv4RankLocalDeviceAccount {
    // Rank-local sharded attention, router, shared-expert and mHC weights.
    std::uint64_t rank_local_weight_bytes{};
    // Centralized resident spine retained so prefill can still run.
    std::uint64_t centralized_spine_bytes{};
    std::uint64_t workspace_bytes{};
    std::uint64_t kv_capacity_bytes{};
    std::uint64_t nccl_buffer_bytes{};
    std::uint64_t head_buffer_bytes{};
    // Centralized prefill expert cache. Capped by admission; never grown.
    std::uint64_t expert_cache_bytes{};

    [[nodiscard]] std::uint64_t total() const noexcept;
    // Everything except the expert cache, which is the only capped component.
    [[nodiscard]] std::uint64_t fixed_total() const noexcept;
};

struct Dsv4RankLocalHostAccount {
    // Routed experts in the transformed tiled NUMA arena. Shared with the
    // centralized path rather than duplicated.
    std::uint64_t routed_cpu_storage_bytes{};
    std::uint64_t host_parameter_bytes{};
    std::uint64_t host_workspace_bytes{};

    [[nodiscard]] std::uint64_t total() const noexcept;
};

struct Dsv4RankLocalAdmissionRequest {
    std::vector<int> devices;
    Dsv4KvCacheMode kv_cache_mode{Dsv4KvCacheMode::ScalarOracle};
    // Whether this build linked NCCL. False must reject before model loading.
    bool nccl_available{};
    // Checkpoint contract: a supported DSV4 index with FP4 routed experts.
    bool supported_checkpoint{};
    bool fp4_routed_experts{};
    std::uint32_t layer_count{};
    std::array<Dsv4RankLocalDeviceAccount, kDsv4RankLocalWorld> device{};
    Dsv4RankLocalHostAccount host{};
    std::uint64_t per_device_vram_ceiling_bytes{
        kDsv4RankLocalPerDeviceVramCeiling};
    std::uint64_t host_rss_ceiling_bytes{kDsv4RankLocalHostRssCeiling};
};

struct Dsv4RankLocalAdmission {
    // Per-device expert-cache capacity after the fixed components are
    // reserved. Zero is legal: prefill then runs entirely on demand uploads.
    std::array<std::uint64_t, kDsv4RankLocalWorld> expert_cache_capacity_bytes{};
    std::array<std::uint64_t, kDsv4RankLocalWorld> device_total_bytes{};
    std::uint64_t host_total_bytes{};
    std::array<std::vector<int>, kDsv4RankLocalWorld> rank_cpus;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

// Assigns one NUMA node's CPUs to each rank. Requires at least two nodes, each
// contributing at least `minimum_cpus_per_rank`. Ranks are assigned in node
// order so the mapping is stable across runs.
[[nodiscard]] ValidationResult plan_dsv4_rank_local_cpus(
    const NumaTopology& topology, std::size_t minimum_cpus_per_rank,
    std::array<std::vector<int>, kDsv4RankLocalWorld>& rank_cpus);

// Complete fail-closed admission for the rank-local topology. Every failed
// condition is reported, not just the first, so one rejection names every
// unmet requirement. Never returns a partially usable plan.
[[nodiscard]] Dsv4RankLocalAdmission admit_dsv4_rank_local(
    const Dsv4RankLocalAdmissionRequest& request, const NumaTopology& topology);

}  // namespace strata
