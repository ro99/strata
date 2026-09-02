#include "strata/platform/hardware_profile.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <string>
#include <string_view>
#include <thread>

#include <sched.h>

namespace strata {

namespace {

// MemTotal from /proc/meminfo, in bytes. Zero when unreadable -- the caller
// distinguishes that from a real answer, so an unreadable file must not fall
// back to a plausible-looking guess.
[[nodiscard]] std::uint64_t read_host_memory_bytes() noexcept {
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo) return 0U;
    std::string line;
    while (std::getline(meminfo, line)) {
        constexpr std::string_view key = "MemTotal:";
        if (line.compare(0U, key.size(), key) != 0) continue;
        std::string_view rest(line);
        rest.remove_prefix(key.size());
        while (!rest.empty() && rest.front() == ' ') rest.remove_prefix(1U);
        std::uint64_t kibibytes = 0U;
        const auto* begin = rest.data();
        const auto parsed =
            std::from_chars(begin, begin + rest.size(), kibibytes);
        if (parsed.ec != std::errc{}) return 0U;
        return kibibytes * 1024U;
    }
    return 0U;
}

// One `Key:  value kB` field of /proc/meminfo, in its own units. Returns zero
// when absent, which for a huge-page field means "this host reserved none"
// rather than "unknown": the kernel omits the fields entirely only on builds
// without huge-page support, where zero is also the right answer.
[[nodiscard]] std::uint64_t read_meminfo_field(std::string_view key) noexcept {
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo) return 0U;
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.compare(0U, key.size(), key) != 0) continue;
        std::string_view rest(line);
        rest.remove_prefix(key.size());
        while (!rest.empty() && rest.front() == ' ') rest.remove_prefix(1U);
        std::uint64_t value = 0U;
        const auto* begin = rest.data();
        if (std::from_chars(begin, begin + rest.size(), value).ec != std::errc{}) {
            return 0U;
        }
        return value;
    }
    return 0U;
}

// The process's CPU affinity mask, not the machine's CPU count: a cgroup or a
// taskset makes those differ, and threads beyond the mask only contend.
[[nodiscard]] std::vector<int> read_usable_cpu_ids() noexcept {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        std::vector<int> cpus;
        cpus.reserve(static_cast<std::size_t>(CPU_COUNT(&set)));
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &set)) cpus.push_back(cpu);
        }
        if (!cpus.empty()) return cpus;
    }
    const auto concurrency = std::thread::hardware_concurrency();
    const auto count = concurrency == 0U ? 1U : concurrency;
    std::vector<int> cpus(count);
    for (std::size_t index = 0U; index < cpus.size(); ++index) {
        cpus[index] = static_cast<int>(index);
    }
    return cpus;
}

[[nodiscard]] HardwareProfile probe() {
    HardwareProfile profile;
    profile.host_memory_bytes = read_host_memory_bytes();
    profile.usable_cpu_ids = read_usable_cpu_ids();
    profile.usable_cpus = profile.usable_cpu_ids.size();
    profile.numa = NumaTopology::detect();
    std::size_t smallest = 0U;
    for (const auto& cpus : profile.numa.node_cpus) {
        if (cpus.empty()) continue;
        if (smallest == 0U || cpus.size() < smallest) smallest = cpus.size();
    }
    // A machine sysfs reports as single-node still has all its CPUs on that
    // one node, so the minimum is the whole mask rather than zero.
    profile.minimum_cpus_per_node =
        smallest != 0U ? smallest : profile.usable_cpus;

    profile.huge_page_bytes = read_meminfo_field("Hugepagesize:") * 1024U;
    profile.explicit_huge_pages_total = read_meminfo_field("HugePages_Total:");
    profile.explicit_huge_pages_free = read_meminfo_field("HugePages_Free:");
    // The setting is rendered as a bracketed choice, e.g.
    // `always [madvise] never`, so the active mode is the bracketed token.
    std::ifstream thp("/sys/kernel/mm/transparent_hugepage/enabled");
    std::string mode;
    if (thp && std::getline(thp, mode)) {
        profile.transparent_huge_pages_always =
            mode.find("[always]") != std::string::npos;
        profile.transparent_huge_pages_enabled =
            profile.transparent_huge_pages_always ||
            mode.find("[madvise]") != std::string::npos;
    }
    return profile;
}

}  // namespace

std::uint64_t HardwareProfile::host_usable_bytes(
    double fraction) const noexcept {
    if (host_memory_bytes == 0U) return 0U;
    const double clamped = std::clamp(fraction, 0.0, 1.0);
    return static_cast<std::uint64_t>(
        static_cast<double>(host_memory_bytes) * clamped);
}

std::uint32_t HardwareProfile::worker_threads(double fraction) const noexcept {
    const double clamped = std::clamp(fraction, 0.0, 1.0);
    const auto scaled = static_cast<std::size_t>(
        static_cast<double>(usable_cpus) * clamped);
    return static_cast<std::uint32_t>(std::max<std::size_t>(scaled, 1U));
}

const HardwareProfile& host_hardware_profile() {
    static const HardwareProfile profile = probe();
    return profile;
}

}  // namespace strata
