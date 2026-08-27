#pragma once

#include "strata/platform/result.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace strata {

// Host NUMA topology, read from sysfs so the runtime carries no libnuma
// dependency. Everything here is discovered rather than assumed: a one-node
// box, a two-socket box and a four-socket box all get the placement their
// hardware wants without a build flag or a tuning constant.
//
// Why the runtime cares: a routed expert read costs local DRAM bandwidth or
// remote DRAM bandwidth depending on which node its bytes landed on, and on a
// multi-node box those differ by more than any kernel change is worth. An
// interleaved arena makes roughly (nodes-1)/nodes of every thread's reads
// remote. Binding each expert's bytes to one node, and computing that expert
// only on that node's cores, makes every read local and puts all of the
// machine's memory controllers to work on disjoint data at once.
struct NumaTopology {
    // Number of online nodes. One means there is nothing to place.
    int nodes{1};
    // node_cpus[n] lists the logical CPUs of node n, in ascending order.
    std::vector<std::vector<int>> node_cpus;
    // cpu_node[c] is the node of logical CPU c, or -1 when unknown.
    std::vector<int> cpu_node;

    // node_primary_cpus[n] lists only the first logical CPU of each physical
    // core on node n -- one entry per core, SMT siblings excluded. Empty when
    // sysfs does not expose thread siblings, in which case callers fall back
    // to node_cpus.
    std::vector<std::vector<int>> node_primary_cpus;

    [[nodiscard]] static NumaTopology detect();
    // One worker per physical core on the smallest node. A compute-bound pool
    // gains nothing from SMT siblings on this workload and loses the runnable
    // headroom the rest of the process needs; see experiment 0123.
    [[nodiscard]] std::size_t smallest_node_cores() const noexcept;
    [[nodiscard]] bool multi_node() const noexcept { return nodes > 1; }
    // The node a logical CPU belongs to, or 0 when the topology is unknown.
    [[nodiscard]] int node_of_cpu(int cpu) const noexcept;
};

// Binds [base, base + bytes) to `node` with MPOL_BIND. Must be called before
// the range is first touched: without MPOL_MF_MOVE the policy only governs
// pages that have not been faulted in yet, which is exactly the staging order
// this is used in. Advisory -- the range is page-aligned inwards and a failure
// leaves the default policy, because placement changes no bytes.
[[nodiscard]] bool numa_bind_range(void* base, std::uint64_t bytes,
                                   int node) noexcept;

// Interleaves [base, base + bytes) across every node. The fallback when a
// range has no single owner.
[[nodiscard]] bool numa_interleave_range(void* base, std::uint64_t bytes,
                                         const NumaTopology& topology) noexcept;

// Parses the routed-expert index out of a DeepSeek tensor name of the form
// `layers.<layer>.ffn.experts.<expert>.<op>`. Returns false for any other
// tensor, which the caller places by interleaving instead.
[[nodiscard]] bool parse_routed_expert(std::string_view name,
                                       std::uint32_t& expert) noexcept;

}  // namespace strata
