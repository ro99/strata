// Production-shape screen for reusing one transformed DeepSeek FP4 expert
// tile across several prompt rows. This isolates the proposed CPU mechanism:
// same worker pool, weights, row arithmetic and output layout; only the order
// of weight decode changes from one row at a time to four rows per tile.

#include "cli_common.hpp"

#include "strata/deepseek_host_expert.hpp"
#include "strata/numa_topology.hpp"
#include "strata/worker_pool.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sched.h>
#include <span>
#include <vector>

namespace {

constexpr std::uint64_t kHidden = 4096U;
constexpr std::uint64_t kIntermediate = 2048U;
constexpr std::uint64_t kShards = 2U;
constexpr std::uint64_t kLocalIntermediate = kIntermediate / kShards;
constexpr std::uint64_t kBlock = 32U;
constexpr float kCoefficient = 0.75F;

struct HostFp4 {
    std::vector<std::byte> packed;
    std::vector<std::byte> scales;
};

HostFp4 make_fp4(std::uint64_t rows, std::uint64_t columns,
                 std::uint8_t seed) {
    HostFp4 result;
    const auto packed_columns = (columns + 1U) / 2U;
    const auto scale_columns = (columns + 31U) / 32U;
    result.packed.resize(static_cast<std::size_t>(rows * packed_columns));
    result.scales.resize(static_cast<std::size_t>(rows * scale_columns));
    for (std::size_t index = 0U; index < result.packed.size(); ++index) {
        const auto low = static_cast<std::uint8_t>(
            (seed + index * 5U + index / packed_columns * 3U) & 0x0FU);
        const auto high = static_cast<std::uint8_t>(
            (seed + index * 11U + index / packed_columns * 7U + 1U) & 0x0FU);
        result.packed[index] = static_cast<std::byte>(
            low | static_cast<std::uint8_t>(high << 4U));
    }
    for (std::size_t index = 0U; index < result.scales.size(); ++index) {
        result.scales[index] = static_cast<std::byte>(
            0x78U + static_cast<std::uint8_t>((index + seed) % 3U));
    }
    return result;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}

void print_samples(std::span<const double> values) {
    std::cout << '[';
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}

struct Workspace {
    std::vector<float> input;
    std::vector<float> activation;
    std::vector<float> output;
};

strata::ValidationResult run_expert(
    bool batched, std::uint32_t rows,
    const strata::Dsv4TiledExpertWeights& weights,
    strata::HostWorkerPool& workers, Workspace& workspace) {
    strata::ValidationResult result;
    std::fill(workspace.activation.begin(), workspace.activation.end(), 0.0F);
    std::fill(workspace.output.begin(), workspace.output.end(), 0.0F);
    std::vector<std::uint32_t> row_indices(rows);
    std::iota(row_indices.begin(), row_indices.end(), 0U);

    const auto gate_tasks = kLocalIntermediate / kBlock;
    result = workers.parallel_for(gate_tasks, [&](std::size_t task) {
        const auto offset = static_cast<std::uint64_t>(task) * kBlock;
        if (!batched) {
            for (std::uint32_t row = 0U; row < rows; ++row) {
                std::array<float, kBlock> gate{};
                std::array<float, kBlock> up{};
                const auto input = std::span<const float>(workspace.input)
                    .subspan(static_cast<std::size_t>(row) * kHidden, kHidden);
                for (std::size_t half = 0U; half < 2U; ++half) {
                    strata::dsv4_tiled_expert_matvec16(
                        std::span<float, 16U>(gate.data() + half * 16U, 16U),
                        input, weights.w13_packed, weights.w13_scales,
                        2U * kLocalIntermediate, offset + half * 16U);
                    strata::dsv4_tiled_expert_matvec16(
                        std::span<float, 16U>(up.data() + half * 16U, 16U),
                        input, weights.w13_packed, weights.w13_scales,
                        2U * kLocalIntermediate,
                        kLocalIntermediate + offset + half * 16U);
                }
                auto* destination = workspace.activation.data() +
                    static_cast<std::size_t>(row) * kLocalIntermediate + offset;
                for (std::size_t index = 0U; index < kBlock; ++index) {
                    destination[index] = gate[index] /
                        (1.0F + std::exp(-gate[index])) * up[index] *
                        kCoefficient;
                }
            }
            return;
        }

        std::vector<float> gate(static_cast<std::size_t>(rows) * kBlock);
        std::vector<float> up(static_cast<std::size_t>(rows) * kBlock);
        for (std::size_t half = 0U; half < 2U; ++half) {
            strata::dsv4_tiled_expert_matvec16_rows(
                std::span<float>(gate).subspan(half * 16U), kBlock,
                workspace.input, kHidden, row_indices, weights.w13_packed,
                weights.w13_scales, 2U * kLocalIntermediate,
                offset + half * 16U);
            strata::dsv4_tiled_expert_matvec16_rows(
                std::span<float>(up).subspan(half * 16U), kBlock,
                workspace.input, kHidden, row_indices, weights.w13_packed,
                weights.w13_scales, 2U * kLocalIntermediate,
                kLocalIntermediate + offset + half * 16U);
        }
        for (std::uint32_t row = 0U; row < rows; ++row) {
            auto* destination = workspace.activation.data() +
                static_cast<std::size_t>(row) * kLocalIntermediate + offset;
            for (std::size_t index = 0U; index < kBlock; ++index) {
                const auto source = static_cast<std::size_t>(row) * kBlock + index;
                destination[index] = gate[source] /
                    (1.0F + std::exp(-gate[source])) * up[source] *
                    kCoefficient;
            }
        }
    });
    if (!result.ok()) return result;
    for (std::uint32_t row = 0U; row < rows; ++row) {
        result = strata::dsv4_host_expert_quantize(
            std::span<float>(workspace.activation)
                .subspan(static_cast<std::size_t>(row) * kLocalIntermediate,
                         kLocalIntermediate),
            kLocalIntermediate);
        if (!result.ok()) return result;
    }

    const auto down_tasks = kHidden / kBlock;
    result = workers.parallel_for(down_tasks, [&](std::size_t task) {
        const auto offset = static_cast<std::uint64_t>(task) * kBlock;
        if (!batched) {
            for (std::uint32_t row = 0U; row < rows; ++row) {
                const auto input = std::span<const float>(workspace.activation)
                    .subspan(static_cast<std::size_t>(row) * kLocalIntermediate,
                             kLocalIntermediate);
                for (std::size_t half = 0U; half < 2U; ++half) {
                    strata::dsv4_tiled_expert_matvec16(
                        std::span<float, 16U>(
                            workspace.output.data() +
                                static_cast<std::size_t>(row) * kHidden +
                                offset + half * 16U,
                            16U),
                        input, weights.w2_packed, weights.w2_scales, kHidden,
                        offset + half * 16U);
                }
            }
            return;
        }
        for (std::size_t half = 0U; half < 2U; ++half) {
            strata::dsv4_tiled_expert_matvec16_rows(
                std::span<float>(workspace.output).subspan(offset + half * 16U),
                kHidden, workspace.activation, kLocalIntermediate, row_indices,
                weights.w2_packed, weights.w2_scales, kHidden,
                offset + half * 16U);
        }
    });
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    std::uint32_t rows = 12U;
    std::uint32_t repetitions = 5U;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&]() -> const char* {
            return index + 1 < argc ? argv[++index] : nullptr;
        };
        if (argument == "--rows") {
            const auto* value = next();
            if (value == nullptr || !strata::cli::parse_u32(value, rows) ||
                rows == 0U || rows > 64U) return 2;
        } else if (argument == "--repetitions") {
            const auto* value = next();
            if (value == nullptr ||
                !strata::cli::parse_u32(value, repetitions) ||
                repetitions == 0U || repetitions > 31U) return 2;
        } else {
            return 2;
        }
    }

    const auto topology = strata::NumaTopology::detect();
    if (topology.node_cpus.empty() || topology.node_cpus[0].size() < 24U) {
        std::cerr << "production probe requires 24 CPUs on NUMA node zero\n";
        return 1;
    }
    std::vector<int> cpus(topology.node_cpus[0].begin(),
                          topology.node_cpus[0].begin() + 24);
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    CPU_SET(cpus.front(), &affinity);
    if (sched_setaffinity(0, sizeof(affinity), &affinity) != 0) return 1;

    const auto w1 = make_fp4(kIntermediate, kHidden, 1U);
    const auto w3 = make_fp4(kIntermediate, kHidden, 6U);
    const auto w2 = make_fp4(kHidden, kIntermediate, 11U);
    const strata::Dsv4HostExpertWeights canonical{
        w1.packed, w1.scales, w3.packed, w3.scales, w2.packed, w2.scales};
    const auto shard_bytes = strata::dsv4_tiled_expert_shard_bytes(
        kHidden, kIntermediate, kShards);
    std::vector<std::byte> storage(static_cast<std::size_t>(shard_bytes));
    if (!strata::dsv4_transform_tiled_expert_shard(
             storage, canonical, kHidden, kIntermediate, 0U, kShards)
             .ok()) return 1;
    auto viewed = strata::dsv4_tiled_expert_weights(
        storage, kHidden, kIntermediate, kShards);
    if (!viewed.ok()) return 1;

    Workspace scalar;
    scalar.input.resize(static_cast<std::size_t>(rows) * kHidden);
    scalar.activation.resize(static_cast<std::size_t>(rows) * kLocalIntermediate);
    scalar.output.resize(static_cast<std::size_t>(rows) * kHidden);
    for (std::uint32_t row = 0U; row < rows; ++row) {
        auto input = std::span<float>(scalar.input).subspan(
            static_cast<std::size_t>(row) * kHidden, kHidden);
        for (std::size_t column = 0U; column < input.size(); ++column) {
            input[column] = std::sin(
                static_cast<float>(column) * 0.013F +
                static_cast<float>(row) * 0.071F);
        }
        if (!strata::dsv4_host_expert_quantize(input, kHidden).ok()) return 1;
    }
    Workspace batched = scalar;
    strata::HostWorkerPool workers(cpus, std::chrono::milliseconds(1));
    if (!run_expert(false, rows, viewed.value, workers, scalar).ok() ||
        !run_expert(true, rows, viewed.value, workers, batched).ok()) return 1;
    const bool exact = std::equal(
        scalar.output.begin(), scalar.output.end(), batched.output.begin(),
        [](float left, float right) {
            return std::bit_cast<std::uint32_t>(left) ==
                   std::bit_cast<std::uint32_t>(right);
        });
    if (!exact) return 1;

    std::vector<double> scalar_ms;
    std::vector<double> batched_ms;
    scalar_ms.reserve(repetitions);
    batched_ms.reserve(repetitions);
    for (std::uint32_t repetition = 0U; repetition < repetitions;
         ++repetition) {
        for (const bool batch : {false, true}) {
            auto& workspace = batch ? batched : scalar;
            const auto started = std::chrono::steady_clock::now();
            if (!run_expert(batch, rows, viewed.value, workers, workspace).ok()) {
                return 1;
            }
            const auto elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            (batch ? batched_ms : scalar_ms).push_back(elapsed);
        }
    }
    const auto scalar_median = median(scalar_ms);
    const auto batched_median = median(batched_ms);
    volatile float sink = scalar.output.front() + batched.output.back();
    static_cast<void>(sink);
    std::cout << std::setprecision(10)
              << "{\"rows\":" << rows
              << ",\"workers\":24"
              << ",\"exact\":" << (exact ? "true" : "false")
              << ",\"scalar_ms\":";
    print_samples(scalar_ms);
    std::cout << ",\"batched_ms\":";
    print_samples(batched_ms);
    std::cout << ",\"scalar_median_ms\":" << scalar_median
              << ",\"batched_median_ms\":" << batched_median
              << ",\"speedup\":" << scalar_median / batched_median
              << "}\n";
    return 0;
}
