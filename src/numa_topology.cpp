#include "strata/numa_topology.hpp"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <fstream>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>

namespace strata {

namespace {

// From linux/mempolicy.h, spelled here rather than taking a libnuma
// dependency for two syscalls.
constexpr int kMpolBind = 2;
constexpr int kMpolInterleave = 3;

constexpr std::uint64_t kPageBytes = 4096U;

// "0-13,28-41" -> {0..13, 28..41}
void parse_cpu_list(std::string_view text, std::vector<int>& cpus) {
    while (!text.empty()) {
        const auto comma = text.find(',');
        auto item = text.substr(0U, comma);
        while (!item.empty() && (item.back() == '\n' || item.back() == ' ')) {
            item.remove_suffix(1U);
        }
        const auto dash = item.find('-');
        int first = 0;
        int last = 0;
        const auto to_int = [](std::string_view value, int& out) {
            return std::from_chars(value.data(), value.data() + value.size(),
                                   out).ec == std::errc{};
        };
        if (dash == std::string_view::npos) {
            if (to_int(item, first)) cpus.push_back(first);
        } else if (to_int(item.substr(0U, dash), first) &&
                   to_int(item.substr(dash + 1U), last)) {
            for (int cpu = first; cpu <= last; ++cpu) cpus.push_back(cpu);
        }
        if (comma == std::string_view::npos) return;
        text.remove_prefix(comma + 1U);
    }
}

[[nodiscard]] std::vector<unsigned long> node_mask(int nodes,
                                                   int only_node) {
    const auto words = static_cast<std::size_t>((nodes + 63) / 64);
    std::vector<unsigned long> mask(words, 0UL);
    if (only_node >= 0) {
        mask[static_cast<std::size_t>(only_node) / 64U] |=
            1UL << (static_cast<std::size_t>(only_node) % 64U);
        return mask;
    }
    for (int node = 0; node < nodes; ++node) {
        mask[static_cast<std::size_t>(node) / 64U] |=
            1UL << (static_cast<std::size_t>(node) % 64U);
    }
    return mask;
}

// mbind() requires a page-aligned base. Trimming inwards leaves at most one
// page at each end under the default policy, which for a 13 MB expert is
// noise; growing outwards would instead steal pages from the neighbouring
// tensor, which may belong to another node.
[[nodiscard]] bool bind_range(void* base, std::uint64_t bytes, int mode,
                              const std::vector<unsigned long>& mask,
                              int nodes) noexcept {
    if (base == nullptr || bytes == 0U) return false;
    auto address = reinterpret_cast<std::uintptr_t>(base);
    const auto aligned = (address + kPageBytes - 1U) & ~(kPageBytes - 1U);
    const auto skipped = aligned - address;
    if (skipped >= bytes) return false;
    const auto length = (bytes - skipped) & ~(kPageBytes - 1U);
    if (length == 0U) return false;
    return syscall(SYS_mbind, reinterpret_cast<void*>(aligned),
                   static_cast<unsigned long>(length), mode, mask.data(),
                   static_cast<unsigned long>(nodes + 1), 0U) == 0;
}

}  // namespace

NumaTopology NumaTopology::detect() {
    NumaTopology result;
    std::vector<std::vector<int>> discovered;
    for (int node = 0; node < 1024; ++node) {
        const auto path = "/sys/devices/system/node/node" +
                          std::to_string(node) + "/cpulist";
        std::ifstream stream(path);
        if (!stream.is_open()) break;
        std::string text;
        std::getline(stream, text);
        discovered.emplace_back();
        parse_cpu_list(text, discovered.back());
    }
    if (discovered.empty()) return result;
    result.nodes = static_cast<int>(discovered.size());
    result.node_cpus = std::move(discovered);
    int highest = -1;
    for (const auto& cpus : result.node_cpus) {
        for (const auto cpu : cpus) highest = std::max(highest, cpu);
    }
    result.cpu_node.assign(static_cast<std::size_t>(highest + 1), -1);
    for (int node = 0; node < result.nodes; ++node) {
        for (const auto cpu : result.node_cpus[static_cast<std::size_t>(node)]) {
            result.cpu_node[static_cast<std::size_t>(cpu)] = node;
        }
    }
    return result;
}

int NumaTopology::node_of_cpu(int cpu) const noexcept {
    if (cpu < 0 || static_cast<std::size_t>(cpu) >= cpu_node.size()) return 0;
    const auto node = cpu_node[static_cast<std::size_t>(cpu)];
    return node < 0 ? 0 : node;
}

bool numa_bind_range(void* base, std::uint64_t bytes, int node) noexcept {
    if (node < 0) return false;
    const auto mask = node_mask(node + 1, node);
    return bind_range(base, bytes, kMpolBind, mask, node + 1);
}

bool numa_interleave_range(void* base, std::uint64_t bytes,
                           const NumaTopology& topology) noexcept {
    if (!topology.multi_node()) return false;
    const auto mask = node_mask(topology.nodes, -1);
    return bind_range(base, bytes, kMpolInterleave, mask, topology.nodes);
}

bool parse_routed_expert(std::string_view name,
                         std::uint32_t& expert) noexcept {
    constexpr std::string_view marker = ".ffn.experts.";
    const auto position = name.find(marker);
    if (position == std::string_view::npos) return false;
    auto tail = name.substr(position + marker.size());
    const auto dot = tail.find('.');
    if (dot == std::string_view::npos) return false;
    tail = tail.substr(0U, dot);
    if (tail.empty()) return false;
    std::uint32_t parsed = 0U;
    if (std::from_chars(tail.data(), tail.data() + tail.size(), parsed).ec !=
        std::errc{}) {
        return false;
    }
    expert = parsed;
    return true;
}

}  // namespace strata
