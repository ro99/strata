// Standalone probe for the DeepSeek host-expert NUMA placement lever.
//
// Reproduces, in seconds, the exact mechanism the runtime depends on:
//   1. stage()  -- mmap a large arena, bind each ~13 MB expert slab to one
//                  node with MPOL_BIND (expert % nodes), first-touch the pages
//                  from unbound staging threads exactly as the checkpoint
//                  readers do.
//   2. host_moe -- a 28-worker pool pinned to physical cores 0-27 streams the
//                  slabs, with each lane walking only the slabs bound to its
//                  own node, exactly as run_rows() does.
//   3. A baseline arm streams the same arena without binding (the in-situ flat
//      rate the runtime currently measures).
//
// The probe then reports the aggregate GB/s for each arm plus the /proc/self/
// numa_maps placement for the arena, so a page that lands on the wrong node is
// visible instead of silently re-expensive.

#include "cli_common.hpp"

#include "strata/numa_topology.hpp"
#include "strata/worker_pool.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <thread>
#include <vector>

namespace {

constexpr std::uint64_t kSlabBytes = 13'400'000U;

struct Options {
    std::uint64_t arena_mib = 1280;
    std::uint64_t slab_bytes = kSlabBytes;
    std::uint32_t iterations = 21U;
    std::uint32_t threads = 28U;
    std::uint32_t read_workers = 8U;
    bool bind = true;
    bool interleave = false;
    bool dump_numa_maps = true;
    bool hugepage = false;
};

// The arena's slabs, each conceptually one expert triplet, the way
// Dsv4ResidentWeightStore::stage() structures the resident arena.
struct Arena {
    std::byte* base{};
    std::uint64_t bytes{};
    std::uint64_t slabs{};
    std::uint64_t slab_bytes{};
};

std::string numa_maps_snippet(std::byte* base, std::uint64_t bytes) {
    std::ifstream maps("/proc/self/numa_maps");
    std::string line;
    std::string out;
    std::array<std::uint64_t, 4> node_bytes{0, 0, 0, 0};
    std::uint64_t seen = 0U;
    while (std::getline(maps, line)) {
        std::istringstream istr(line);
        std::string address;
        istr >> address;
        const std::uint64_t begin = std::stoull(address, nullptr, 16);
        const std::uint64_t end = begin + 0x1000U;
        if (end < reinterpret_cast<std::uintptr_t>(base) ||
            begin >= reinterpret_cast<std::uintptr_t>(base) + bytes) {
            continue;
        }
        for (const char* token : {"N0=", "N1=", "N2=", "N3="}) {
            const auto at = line.find(token);
            if (at == std::string::npos) continue;
            node_bytes[static_cast<std::size_t>(token[1] - '0')] +=
                std::stoull(line.substr(at + 3U));
        }
        seen += 1U;
    }
    out = "N0=" + std::to_string(node_bytes[0]) + " N1=" +
          std::to_string(node_bytes[1]);
    if (node_bytes[2] != 0U || node_bytes[3] != 0U) {
        out += " N2=" + std::to_string(node_bytes[2]) + " N3=" +
               std::to_string(node_bytes[3]);
    }
    out += " (pages=" + std::to_string(seen) + ")";
    return out;
}

// First-touch the arena the way the staging readers do: unbound threads, each
// walking a contiguous run of slabs. Every page gets a write so it is resident
// before the timed pass.
void touch(std::byte* base, std::uint64_t bytes, std::uint32_t workers) {
    const auto pass = [&](std::uint64_t begin, std::uint64_t end) {
        auto* cursor = base + begin;
        const auto count = end - begin;
        std::fill_n(cursor, count, std::byte{0xAB});
    };
    const auto slices = static_cast<std::uint64_t>(workers);
    std::vector<std::thread> threads;
    threads.reserve(slices);
    for (std::uint64_t slice = 0U; slice < slices; ++slice) {
        threads.emplace_back([&, slice] {
            const auto begin = bytes * slice / slices;
            const auto end = bytes * (slice + 1U) / slices;
            pass(begin, end);
        });
    }
    for (auto& thread : threads) thread.join();
}

strata::ValidationResult place(const Options& options, const Arena& arena,
                               const strata::NumaTopology& topology) {
    strata::ValidationResult result;
    for (std::uint64_t slab = 0U; slab < arena.slabs; ++slab) {
        auto* const base = arena.base + slab * arena.slab_bytes;
        const auto bytes = std::min(arena.slab_bytes, arena.bytes - arena.slab_bytes * slab);
        if (options.interleave) {
            static_cast<void>(strata::numa_interleave_range(base, bytes, topology));
        } else if (options.bind) {
            const auto node = static_cast<int>(slab % static_cast<std::uint64_t>(topology.nodes));
            if (!strata::numa_bind_range(base, bytes, node)) {
                result.errors.emplace_back("mbind failed for slab " +
                                           std::to_string(slab));
            }
        }
    }
    return result;
}

// A lane walks only the slabs bound to its node, exactly the run_rows()
// partition in host_moe(): the lane's row span is a contiguous slice of its
// node's slab sequence.
std::uint64_t stream_slabs(const Options& options, const Arena& arena,
                           const strata::NumaTopology& topology,
                           strata::HostWorkerPool& pool) {
    std::vector<int> lane_node(options.threads, 0);
    for (std::size_t lane = 0U; lane < options.threads; ++lane) {
        lane_node[lane] = topology.node_of_cpu(static_cast<int>(lane));
    }
    std::vector<std::vector<std::uint64_t>> node_slabs(
        static_cast<std::size_t>(std::max(topology.nodes, 1)));
    for (std::uint64_t slab = 0U; slab < arena.slabs; ++slab) {
        node_slabs[static_cast<std::size_t>(
            slab % static_cast<std::uint64_t>(topology.nodes))].push_back(slab);
    }
    std::vector<std::vector<std::size_t>> node_lanes(node_slabs.size());
    for (std::size_t lane = 0U; lane < options.threads; ++lane) {
        node_lanes[static_cast<std::size_t>(lane_node[lane])].push_back(lane);
    }
    std::atomic<std::uint64_t> bytes_read{0U};
    auto dispatched = pool.parallel_for_addressed(options.threads, [&](std::size_t lane) {
        const auto node = static_cast<std::size_t>(lane_node[lane]);
        const auto& slabs = node_slabs[node];
        const auto& lanes = node_lanes[node];
        if (slabs.empty() || lanes.empty()) return;
        const auto position = static_cast<std::uint64_t>(
            std::find(lanes.begin(), lanes.end(), lane) - lanes.begin());
        const auto share = static_cast<std::uint64_t>(lanes.size());
        std::vector<std::uint64_t> slab_bytes(slabs.size(), arena.slab_bytes);
        for (std::size_t i = 0U; i < slabs.size(); ++i) {
            const auto end = slabs[i] * arena.slab_bytes;
            if (end + arena.slab_bytes > arena.bytes) {
                slab_bytes[i] = arena.bytes - end;
            }
        }
        const std::uint64_t total =
            std::accumulate(slab_bytes.begin(), slab_bytes.end(), std::uint64_t{0});
        const std::uint64_t begin = total * position / share;
        const std::uint64_t end = total * (position + 1U) / share;
        std::uint64_t accumulator = 0U;
        std::uint64_t consumed = 0U;
        for (std::size_t slab = 0U; slab < slabs.size() && consumed < end;
             ++slab) {
            const auto length = slab_bytes[slab];
            const auto row_begin = std::max(begin, consumed);
            const auto row_end = std::min(end, consumed + length);
            if (row_begin < row_end) {
                const auto* const source =
                    arena.base + slabs[slab] * arena.slab_bytes +
                    (row_begin - consumed);
                const auto read = row_end - row_begin;
                const auto* const bytes =
                    reinterpret_cast<const std::uint8_t*>(source);
                for (std::uint64_t i = 0U; i < read; ++i) {
                    accumulator += bytes[i];
                }
                bytes_read.fetch_add(read, std::memory_order_relaxed);
            }
            consumed += length;
        }
        volatile std::uint64_t sink = accumulator;
        (void)sink;
    });
    return bytes_read.load(std::memory_order_relaxed);
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto value = [&]() -> const char* {
            return index + 1 < argc ? argv[++index] : nullptr;
        };
        if (argument == "--arena-mib") {
            const auto* next = value();
            if (next == nullptr ||
                !strata::cli::parse_u64(next, options.arena_mib) ||
                options.arena_mib == 0U) {
                return 2;
            }
        } else if (argument == "--iterations") {
            const auto* next = value();
            if (next == nullptr ||
                !strata::cli::parse_u32(next, options.iterations) ||
                options.iterations == 0U) {
                return 2;
            }
        } else if (argument == "--threads") {
            const auto* next = value();
            if (next == nullptr ||
                !strata::cli::parse_u32(next, options.threads) ||
                options.threads == 0U || options.threads > 56U) {
                return 2;
            }
        } else if (argument == "--read-workers") {
            const auto* next = value();
            if (next == nullptr ||
                !strata::cli::parse_u32(next, options.read_workers) ||
                options.read_workers == 0U) {
                return 2;
            }
        } else if (argument == "--no-bind") {
            options.bind = false;
        } else if (argument == "--interleave") {
            options.interleave = true;
            options.bind = false;
        } else if (argument == "--no-numa-maps") {
            options.dump_numa_maps = false;
        } else if (argument == "--hugepage") {
            options.hugepage = true;
        } else {
            return 2;
        }
    }
    if (options.threads > 28U) {
        std::cerr << "warning: threads > 28 uses SMT siblings; lane->node "
                     "mapping mirrors host_moe() and is wrong beyond the "
                     "physical pass\n";
    }
    if (options.slab_bytes > options.arena_mib * 1024U * 1024U) return 2;

    const auto topology = strata::NumaTopology::detect();
    std::cout << "nodes=" << topology.nodes;
    for (int node = 0; node < topology.nodes; ++node) {
        std::cout << " node" << node << "=";
        const auto& cpus = topology.node_cpus[static_cast<std::size_t>(node)];
        for (std::size_t i = 0U; i < cpus.size(); ++i) {
            if (i != 0U) std::cout << ',';
            std::cout << cpus[i];
        }
    }
    std::cout << '\n';

    const auto arena_bytes = options.arena_mib * 1024U * 1024U;
    void* const allocation = mmap(nullptr, static_cast<std::size_t>(arena_bytes),
                                  PROT_READ | PROT_WRITE,
                                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (allocation == MAP_FAILED) {
        std::cerr << "mmap failed\n";
        return 1;
    }
    Arena arena{static_cast<std::byte*>(allocation), arena_bytes,
                (arena_bytes + options.slab_bytes - 1U) / options.slab_bytes,
                options.slab_bytes};

    if (options.hugepage) {
#if defined(MADV_HUGEPAGE)
        static_cast<void>(madvise(allocation,
                                  static_cast<std::size_t>(arena_bytes),
                                  MADV_HUGEPAGE));
#endif
    }

    auto placed = place(options, arena, topology);
    if (!placed.ok()) {
        for (const auto& error : placed.errors) std::cerr << error << '\n';
        std::cerr << "placement failed; pages will follow default policy\n";
    }

    touch(arena.base, arena.bytes, options.read_workers);
    if (options.dump_numa_maps) {
        std::cout << "numa_maps " << numa_maps_snippet(arena.base, arena.bytes)
                  << '\n';
        std::uint64_t anon_huge = 0U;
        std::uint64_t kernel_pages = 0U;
        {
            std::ifstream smaps("/proc/self/smaps");
            std::string line;
            while (std::getline(smaps, line)) {
                if (line.rfind("AnonHugePages:", 0U) == 0U) {
                    anon_huge += std::stoull(line.substr(15U));
                } else if (line.rfind("KernelPageSize:", 0U) == 0U) {
                    kernel_pages = std::stoull(line.substr(16U));
                }
            }
        }
        std::cout << "smaps AnonHugePages_kB=" << anon_huge
                  << " KernelPageSize_kB=" << kernel_pages << '\n';
    }

    strata::HostWorkerPool pool(options.threads);
    std::vector<double> milliseconds;
    milliseconds.reserve(options.iterations);
    for (std::uint32_t iteration = 0U; iteration < options.iterations;
         ++iteration) {
        const auto started = std::chrono::steady_clock::now();
        static_cast<void>(stream_slabs(options, arena, topology, pool));
        milliseconds.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count());
    }
    std::sort(milliseconds.begin(), milliseconds.end());
    const double median = milliseconds[milliseconds.size() / 2U];
    const std::uint64_t read_bytes =
        arena.slab_bytes * (arena.slabs - 1U) +
        std::min(arena.slab_bytes,
                 arena.bytes - arena.slab_bytes * (arena.slabs - 1U));
    std::cout << std::setprecision(10)
              << "{\"arena_bytes\":" << arena_bytes
              << ",\"slabs\":" << arena.slabs
              << ",\"threads\":" << options.threads
              << ",\"read_workers\":" << options.read_workers
              << ",\"bind\":" << (options.bind ? "true" : "false")
              << ",\"interleave\":" << (options.interleave ? "true" : "false")
              << ",\"hugepage\":" << (options.hugepage ? "true" : "false")
              << ",\"iterations\":" << options.iterations
              << ",\"median_ms\":" << median
              << ",\"gb_s\":" << static_cast<double>(read_bytes) /
                     (median * 1.0e6) << "}\n";
    static_cast<void>(munmap(allocation, static_cast<std::size_t>(arena_bytes)));
    return 0;
}
