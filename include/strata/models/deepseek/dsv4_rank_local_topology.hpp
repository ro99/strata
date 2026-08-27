#pragma once

#include "strata/models/deepseek/deepseek_kv_cache.hpp"
#include "strata/platform/numa_topology.hpp"
#include "strata/platform/result.hpp"

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
// Was 24, measured on the development box and asserted as a requirement --
// so the fastest path in the engine hard-failed on any other machine. It is
// now a floor below which one rank's expert pool cannot do useful work;
// admission takes the real figure from the hardware profile's smallest node.
inline constexpr std::size_t kDsv4RankLocalMinimumCpusPerRank = 4U;
// Program ceilings. These are the declared operating-point limits, not
// hardware capacities; admission rejects rather than exceeding them.
//
// 21 GiB per device, raised from experiment 0082's 21,287,272,448 B. That
// figure was the *observed peak* of the centralized baseline, recorded as a
// regression tripwire; it was never a hardware bound. Rank-local decode is
// expected to exceed it, because it holds a second weight set the baseline
// never had: the centralized prefill spine (9,204,991,520 B) alongside the
// rank-local decode set (12,893,962,880 B) is 22,098,954,400 B, which the old
// gate refused by about 0.76 GiB.
//
// The card is 24 GiB and measures 23.561 GiB free with nothing resident, so
// this leaves roughly 2.6 GiB for the CUDA context, cuBLAS workspaces, NCCL
// internals and allocator fragmentation. What 0082 actually validated at its
// budget was zero decode weight and workspace allocations; that property, not
// the byte count, is the one a higher ceiling puts at risk, and Stage 6 must
// re-assert it rather than assume it survives.
//
// This is headroom for integration, not the resolution. Releasing the
// centralized spine after prefill returns 9.2 GB instead of the 0.76 GiB this
// buys, and remains the correct fix.
// Both ceilings were byte counts measured on one machine. They are now
// fractions of what this machine reports, chosen to reproduce those exact
// figures on the box they were validated on:
//
//   22,548,578,304 / 24 GiB      = 0.8750 exactly
//   231,928,233,984 / 251.8 GiB  = 0.8577, so the profile's 0.85 default
//                                  lands at 214.1 GiB, within 1%
//
// The property experiment 0082 validated was zero decode weight and workspace
// allocations at the admitted budget -- not the byte count -- so expressing it
// as a fraction preserves what was actually gated while letting a different
// card or a different amount of RAM get the equivalent headroom.
inline constexpr double kDsv4RankLocalVramFraction = 0.875;

[[nodiscard]] inline std::uint64_t dsv4_rank_local_vram_ceiling(
    std::uint64_t device_total_bytes) noexcept {
    return static_cast<std::uint64_t>(
        static_cast<double>(device_total_bytes) * kDsv4RankLocalVramFraction);
}

[[nodiscard]] std::uint64_t dsv4_rank_local_host_rss_ceiling() noexcept;
// Rank-local decode supports the model's full declared context. Above
// `index_topk * compression_ratio` (2048 at the DSV4 contract) active tokens
// the sparse Lightning Indexer engages; the queued page-update path admits
// sparse-index preparation, so no separate context ceiling applies here.
// Admission still checks the request against the model's declared maximum.
inline constexpr std::uint32_t kDsv4RankLocalSparseIndexerThreshold = 2048U;

// Per-device residency accounted by component. These are deliberately separate
// fields rather than one total: the landing must be able to say which component
// pushed a device over the ceiling, and the centralized prefill cache is the
// only one that may be capped to make room.
struct Dsv4RankLocalDeviceAccount {
    // Device usage already present before the runtime allocation plan. This
    // includes the CUDA context and is conservatively measured as total device
    // usage, so unrelated processes can only reduce admission headroom.
    std::uint64_t initial_device_usage_bytes{};
    // Total VRAM the card reports. The ceiling is a fraction of this, so the
    // same admission arithmetic holds on a 16 GiB card and a 48 GiB one.
    // Zero leaves per_device_vram_ceiling_bytes as the only source.
    std::uint64_t device_total_bytes{};
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
    // Full admitted host KV/index/compressor state, including block metadata
    // and alignment. Physical pages are lazily committed, but their declared
    // capacity still belongs in the fail-closed host residency contract.
    std::uint64_t kv_state_bytes{};
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
    // Active context tokens the session will reach, and the model's declared
    // maximum. Rank-local decode serves the full declared context; the
    // request is rejected only if it exceeds what the model itself allows.
    std::uint32_t active_context_tokens{};
    std::uint32_t maximum_context_tokens{};
    std::array<Dsv4RankLocalDeviceAccount, kDsv4RankLocalWorld> device{};
    Dsv4RankLocalHostAccount host{};
    std::uint64_t per_device_vram_ceiling_bytes{
        0U};  // zero: derived per device at admission
    std::uint64_t host_rss_ceiling_bytes{};  // zero: derived from the profile
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
