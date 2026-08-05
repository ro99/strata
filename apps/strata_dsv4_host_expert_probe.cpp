// Production-pattern probe for the DeepSeek host MoE path.
//
// The stream probe (strata-numa-probe) proved placement moves a pure stream
// 22.9 -> 56.7 GB/s, but the host expert kernel is uop-bound at 26.23 GB/s
// standalone, not memory-bound, so a stream is not a faithful proxy: a
// latency-limited kernel stalls on remote DRAM where a stream tolerates it.
//
// This probe reproduces the *exact* production path against a synthetic arena:
//   - 13.4 MB slabs, each one expert's w1/w3/w2 packed+scale bytes;
//   - the same MPOL_BIND placement stage() applies (expert % nodes);
//   - the same host_moe() dispatch: per layer, 6 routed experts, run_rows()
//     splitting each node's rows across its lanes with the addressed pool, the
//     E4M3 input quantize, the per-expert scratch quantize, and the output
//     combine;
//   - the real dsv4_host_expert_gate_up/down kernels.
//
// Arms: bind vs no-bind vs interleave, so a placement defect in production
// shows up as bind >> no-bind here, and a kernel/dispatch defect shows up as
// all arms converging far below 26 GB/s.

#include "cli_common.hpp"

#include "strata/deepseek_host_expert.hpp"
#include "strata/numa_topology.hpp"
#include "strata/worker_pool.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <thread>
#include <vector>

namespace {

constexpr std::uint64_t kHidden = 4096U;
constexpr std::uint64_t kIntermediate = 2048U;
constexpr std::uint64_t kTopK = 6U;
constexpr std::uint64_t kLayers = 43U;
constexpr std::uint64_t kExperts = 100U;

// One expert triplet: 2*(intermediate*(hidden/2) + intermediate*(hidden/32)) +
// hidden*(intermediate/2) + hidden*(intermediate/32).
constexpr std::uint64_t kW1Packed = kIntermediate * (kHidden / 2U);
constexpr std::uint64_t kW1Scales = kIntermediate * (kHidden / 32U);
constexpr std::uint64_t kExpertBytes =
    2U * (kW1Packed + kW1Scales) + kHidden * (kIntermediate / 2U) +
    kHidden * (kIntermediate / 32U);

struct Options {
    bool bind = true;
    bool interleave = false;
    std::uint32_t iterations = 3U;
    std::uint32_t threads = 28U;
    std::uint64_t layers = kLayers;
};

struct Arena {
    std::byte* base{};
    std::uint64_t bytes{};
};

strata::Dsv4HostExpertWeights expert_weights(std::byte* base) {
    strata::Dsv4HostExpertWeights weights;
    weights.w1_packed = {base, kW1Packed};
    weights.w1_scales = {base + kW1Packed, kW1Scales};
    weights.w3_packed = {base + kW1Packed + kW1Scales, kW1Packed};
    weights.w3_scales = {base + 2U * kW1Packed + kW1Scales, kW1Scales};
    weights.w2_packed = {base + 2U * kW1Packed + 2U * kW1Scales,
                         kHidden * (kIntermediate / 2U)};
    weights.w2_scales = {base + 2U * kW1Packed + 2U * kW1Scales +
                             kHidden * (kIntermediate / 2U),
                         kHidden * (kIntermediate / 32U)};
    return weights;
}

struct LayerRoute {
    std::array<std::uint32_t, kTopK> experts{};
    std::array<float, kTopK> weights{};
};

std::vector<LayerRoute> make_routes(std::uint64_t seed) {
    std::vector<LayerRoute> routes(kLayers);
    std::uint64_t state = seed;
    const auto next = [&]() -> std::uint32_t {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return static_cast<std::uint32_t>(state);
    };
    for (std::size_t layer = 0U; layer < kLayers; ++layer) {
        auto& route = routes[layer];
        bool distinct = false;
        while (!distinct) {
            distinct = true;
            for (std::size_t rank = 0U; rank < kTopK; ++rank) {
                route.experts[static_cast<std::size_t>(rank)] =
                    static_cast<std::uint32_t>(next() % kExperts);
                route.weights[static_cast<std::size_t>(rank)] =
                    static_cast<float>(next() % 1000U) * 0.001F + 0.01F;
            }
            for (std::size_t a = 0U; a < kTopK; ++a) {
                for (std::size_t b = a + 1U; b < kTopK; ++b) {
                    if (route.experts[a] == route.experts[b]) distinct = false;
                }
            }
        }
    }
    return routes;
}

// Runs one token through the exact host_moe() structure.
double run_token(const Options& options, const Arena& arena,
                 const std::vector<LayerRoute>& routes,
                 const strata::NumaTopology& topology,
                 strata::HostWorkerPool& pool) {
    const auto use_avx2 = strata::dsv4_host_expert_avx2_supported();
    const auto lane_count = pool.size();
    std::vector<int> lane_node(lane_count, 0);
    for (std::size_t lane = 0U; lane < lane_count; ++lane) {
        lane_node[lane] = topology.node_of_cpu(static_cast<int>(lane));
    }

    std::vector<float> input(kHidden);
    for (std::size_t i = 0U; i < kHidden; ++i) {
        input[static_cast<std::size_t>(i)] =
            std::sin(static_cast<float>(i) * 0.013F);
    }
    std::vector<float> output(kHidden);
    std::vector<float> scratch(kTopK * kIntermediate);
    std::vector<float> routed(kTopK * kHidden);
    std::vector<strata::ValidationResult> lane_results;

    for (std::uint64_t layer = 0U; layer < options.layers; ++layer) {
        const auto& route = routes[static_cast<std::size_t>(layer)];
        // Match stage(): expert bytes live on expert % nodes, exactly the
        // node_experts / node_lanes split host_moe() builds per layer.
        std::vector<std::vector<std::size_t>> node_experts(
            static_cast<std::size_t>(std::max(topology.nodes, 1)));
        for (std::size_t rank = 0U; rank < kTopK; ++rank) {
            const auto node = static_cast<std::size_t>(
                route.experts[rank] % static_cast<std::uint32_t>(topology.nodes));
            node_experts[node].push_back(rank);
        }
        std::vector<std::vector<std::size_t>> node_lanes(node_experts.size());
        for (std::size_t lane = 0U; lane < lane_count; ++lane) {
            node_lanes[static_cast<std::size_t>(lane_node[lane])].push_back(lane);
        }

        // The E4M3 input quantize the device path owns.
        auto quantized = input;
        static_cast<void>(strata::dsv4_host_expert_quantize(quantized, kHidden));

        const auto run_rows = [&](std::uint64_t per_expert, const auto& body) {
            lane_results.assign(lane_count, {});
            const auto lane_body = [&](std::size_t lane) {
                const auto node = static_cast<std::size_t>(lane_node[lane]);
                const auto& experts = node_experts[node];
                const auto& lanes = node_lanes[node];
                if (experts.empty() || lanes.empty()) return;
                const auto position = static_cast<std::uint64_t>(
                    std::find(lanes.begin(), lanes.end(), lane) - lanes.begin());
                const auto share = static_cast<std::uint64_t>(lanes.size());
                const auto total =
                    static_cast<std::uint64_t>(experts.size()) * per_expert;
                const auto begin = total * position / share;
                const auto end = total * (position + 1U) / share;
                for (auto row = begin; row < end;) {
                    const auto slot = row / per_expert;
                    const auto row_begin = row - slot * per_expert;
                    const auto row_end =
                        std::min(per_expert, end - slot * per_expert);
                    lane_results[lane] =
                        body(experts[static_cast<std::size_t>(slot)],
                             row_begin, row_end);
                    row = (slot + 1U) * per_expert;
                }
            };
            static_cast<void>(
                pool.parallel_for_addressed(lane_count, lane_body));
        };

        run_rows(kIntermediate, [&](std::size_t rank, std::uint64_t row_begin,
                                    std::uint64_t row_end) {
            return strata::dsv4_host_expert_gate_up(
                std::span<float>(scratch).subspan(rank * kIntermediate,
                                                  kIntermediate),
                quantized,
                expert_weights(arena.base + route.experts[rank] * kExpertBytes),
                kHidden, kIntermediate, row_begin, row_end, route.weights[rank],
                10.0F, use_avx2);
        });

        for (std::size_t rank = 0U; rank < kTopK; ++rank) {
            static_cast<void>(strata::dsv4_host_expert_quantize(
                std::span<float>(scratch).subspan(rank * kIntermediate,
                                                  kIntermediate),
                kIntermediate));
        }

        run_rows(kHidden, [&](std::size_t rank, std::uint64_t row_begin,
                              std::uint64_t row_end) {
            return strata::dsv4_host_expert_down(
                std::span<float>(routed).subspan(rank * kHidden, kHidden),
                std::span<const float>(scratch).subspan(rank * kIntermediate,
                                                        kIntermediate),
                expert_weights(arena.base + route.experts[rank] * kExpertBytes),
                kHidden, kIntermediate, row_begin, row_end, use_avx2);
        });

        std::fill(output.begin(), output.end(), 0.0F);
        for (std::size_t rank = 0U; rank < kTopK; ++rank) {
            const auto contribution =
                std::span<const float>(routed).subspan(rank * kHidden, kHidden);
            for (std::uint32_t column = 0U; column < kHidden; ++column) {
                output[column] += contribution[column];
            }
        }
    }
    volatile float sink = output[0];
    (void)sink;
    return 0.0;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto value = [&]() -> const char* {
            return index + 1 < argc ? argv[++index] : nullptr;
        };
        if (argument == "--iterations") {
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
        } else if (argument == "--layers") {
            const auto* next = value();
            if (next == nullptr ||
                !strata::cli::parse_u64(next, options.layers) ||
                options.layers == 0U || options.layers > kLayers) {
                return 2;
            }
        } else if (argument == "--no-bind") {
            options.bind = false;
        } else if (argument == "--interleave") {
            options.interleave = true;
            options.bind = false;
        } else {
            return 2;
        }
    }

    const auto topology = strata::NumaTopology::detect();
    const auto arena_bytes = kExperts * kExpertBytes;
    void* const allocation = mmap(nullptr, static_cast<std::size_t>(arena_bytes),
                                  PROT_READ | PROT_WRITE,
                                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (allocation == MAP_FAILED) return 1;
    Arena arena{static_cast<std::byte*>(allocation), arena_bytes};

    for (std::uint64_t expert = 0U; expert < kExperts; ++expert) {
        auto* const base = arena.base + expert * kExpertBytes;
        if (options.interleave) {
            static_cast<void>(strata::numa_interleave_range(
                base, kExpertBytes, topology));
        } else if (options.bind) {
            const auto node = static_cast<int>(
                expert % static_cast<std::uint64_t>(topology.nodes));
            static_cast<void>(strata::numa_bind_range(base, kExpertBytes, node));
        }
    }

    // First-touch from unbound threads, like the staging readers.
    const auto touch = [&] {
        std::vector<std::thread> threads;
        for (std::uint32_t t = 0U; t < 8U; ++t) {
            threads.emplace_back([&, t] {
                const auto begin = arena_bytes * t / 8U;
                const auto end = arena_bytes * (t + 1U) / 8U;
                std::fill_n(arena.base + begin, end - begin, std::byte{0xAB});
            });
        }
        for (auto& thread : threads) thread.join();
    };
    touch();

    const auto routes = make_routes(0xDEC0DE);
    strata::HostWorkerPool pool(options.threads);
    std::vector<double> milliseconds;
    milliseconds.reserve(options.iterations);
    for (std::uint32_t iteration = 0U; iteration < options.iterations;
         ++iteration) {
        const auto started = std::chrono::steady_clock::now();
        static_cast<void>(run_token(options, arena, routes, topology, pool));
        milliseconds.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count());
    }
    std::sort(milliseconds.begin(), milliseconds.end());
    const double median = milliseconds[milliseconds.size() / 2U];
    const std::uint64_t weight_bytes = options.layers * kTopK * kExpertBytes;
    std::cout << std::setprecision(10)
              << "{\"threads\":" << options.threads
              << ",\"layers\":" << options.layers
              << ",\"iterations\":" << options.iterations
              << ",\"bind\":" << (options.bind ? "true" : "false")
              << ",\"interleave\":" << (options.interleave ? "true" : "false")
              << ",\"median_ms_per_token\":" << median
              << ",\"weight_gb_per_token\":" << weight_bytes / 1.0e9
              << ",\"effective_gb_s\":" << static_cast<double>(weight_bytes) /
                     (median * 1.0e6) << "}\n";
    static_cast<void>(munmap(allocation, static_cast<std::size_t>(arena_bytes)));
    return 0;
}
