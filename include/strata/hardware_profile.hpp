#pragma once

// What this machine actually is, measured once.
//
// Before this header, one box's dimensions were compile-time constants in nine
// places: 216 GiB of host RAM appeared three times (as
// Dsv4RuntimeConfig::host_memory_limit_bytes, as placement_model.cpp's
// kDeepSeekHostMemoryLimit, and again inside the rank-local RSS ceiling), a
// 24 GiB card's usable VRAM once, 24 CPUs per NUMA node once, and thread pool
// widths of 28 / 36 / 8 / 3 four more times. None of them were wrong -- they
// were honest measurements of the development machine, with careful comments
// explaining where each came from. They were just facts about hardware living
// in the type system, so every one of them was also a silent mis-size on any
// other machine, and the three copies of 216 GiB could drift apart.
//
// The rule this header establishes: a measurement belongs in a profile, and a
// *policy* -- a fraction, a reserve, a minimum -- belongs in the code that
// applies it. `host_usable_bytes` is a measurement scaled by a stated policy.
// `kDsv4RankLocalMinimumCpusPerRank` was neither: it was a measurement of one
// box asserted as a requirement, which is why it hard-failed elsewhere.
//
// Probing is lazy and cached: sysfs and /proc reads are cheap but not free,
// and admission calls this on every path.

#include "strata/numa_topology.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata {

struct HardwareProfile {
    // Physical host memory, from /proc/meminfo MemTotal. Zero if unreadable,
    // which callers must treat as "unknown", never as "none".
    std::uint64_t host_memory_bytes{};
    // Logical CPUs the process may run on, from its affinity mask rather than
    // the machine's total -- a cgroup or a taskset makes those differ, and the
    // affinity mask is the one that governs how many threads are useful.
    std::size_t usable_cpus{};
    NumaTopology numa;
    // CPUs on the smallest online node. The relevant figure for anything that
    // assigns one worker pool per node, because the smallest node is what
    // bounds a symmetric assignment.
    std::size_t minimum_cpus_per_node{};

    [[nodiscard]] bool multi_node() const noexcept { return numa.nodes > 1; }

    // Host memory a long-lived resident arena may claim, after leaving room
    // for page cache, the CUDA driver's pinned staging and everything else on
    // the box. The fraction is the policy; the total is the measurement.
    //
    // 0.85 reproduces the previous hardcoded ceiling closely on the machine
    // those constants were measured on -- 216 GiB against 251 GiB physical --
    // so this is not a behaviour change there, only a portable statement of
    // the same intent.
    [[nodiscard]] std::uint64_t host_usable_bytes(
        double fraction = 0.85) const noexcept;

    // Sensible width for a compute-bound host pool, bounded by what the
    // process may actually run on. Never zero: a caller that gets zero threads
    // does no work at all, which is a worse failure than a bad width.
    [[nodiscard]] std::uint32_t worker_threads(
        double fraction = 1.0) const noexcept;
};

// Probed once per process and cached. Never fails: an unreadable field is left
// zero and documented as unknown rather than guessed.
[[nodiscard]] const HardwareProfile& host_hardware_profile();

}  // namespace strata
