// Times the canonical rebuild of one routed expert at the production shape.
// Prefill's device upload has to sustain link bandwidth out of the resident
// transformed arena, so the rebuild rate is what decides whether it can be a
// host step at all.
#include "strata/deepseek_host_expert.hpp"
#include "strata/model_adapter.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

constexpr std::uint64_t kHidden =
    strata::kDeepSeekV4ExecutionContract.hidden_size;
constexpr std::uint64_t kIntermediate =
    strata::kDeepSeekV4ExecutionContract.expert_intermediate_size;
constexpr std::uint64_t kShards = 2U;

}  // namespace

int main() {
    const auto shard_bytes = strata::dsv4_tiled_expert_shard_bytes(
        kHidden, kIntermediate, kShards);
    if (shard_bytes == 0U) {
        std::fputs("invalid expert shape\n", stderr);
        return 1;
    }
    std::vector<std::vector<std::byte>> storage(kShards);
    std::vector<std::span<const std::byte>> views(kShards);
    std::uint32_t seed = 0x13579bdfU;
    for (std::uint64_t shard = 0U; shard < kShards; ++shard) {
        storage[shard].resize(static_cast<std::size_t>(shard_bytes));
        for (auto& value : storage[shard]) {
            seed = seed * 1'664'525U + 1'013'904'223U;
            value = static_cast<std::byte>(seed >> 24U);
        }
        views[shard] = storage[shard];
    }

    struct Case {
        const char* name;
        strata::Dsv4ExpertMatrix matrix;
        std::uint64_t rows;
        std::uint64_t columns;
    };
    const Case cases[] = {
        {"w1", strata::Dsv4ExpertMatrix::Gate, kIntermediate, kHidden},
        {"w3", strata::Dsv4ExpertMatrix::Up, kIntermediate, kHidden},
        {"w2", strata::Dsv4ExpertMatrix::Down, kHidden, kIntermediate}};

    constexpr int repetitions = 40;
    double total_ms = 0.0;
    std::uint64_t total_bytes = 0U;
    for (const auto& entry : cases) {
        std::vector<std::byte> packed(
            static_cast<std::size_t>(entry.rows * (entry.columns / 2U)));
        std::vector<std::byte> scales(
            static_cast<std::size_t>(entry.rows * (entry.columns / 32U)));
        std::vector<double> samples;
        samples.reserve(repetitions);
        for (int repetition = 0; repetition < repetitions; ++repetition) {
            const auto started = std::chrono::steady_clock::now();
            const auto rebuilt = strata::dsv4_untile_expert_matrix(
                packed, scales, views, entry.matrix, kHidden, kIntermediate);
            const auto finished = std::chrono::steady_clock::now();
            if (!rebuilt.ok()) {
                std::fprintf(stderr, "rebuild failed for %s\n", entry.name);
                return 1;
            }
            samples.push_back(
                std::chrono::duration<double, std::milli>(
                    finished - started).count());
        }
        std::sort(samples.begin(), samples.end());
        const auto median = samples[samples.size() / 2U];
        const auto bytes = packed.size() + scales.size();
        std::printf("%s  %8.3f ms median  %6.2f GB/s of canonical output\n",
                    entry.name, median,
                    static_cast<double>(bytes) / (median * 1.0e6));
        total_ms += median;
        total_bytes += bytes;
    }
    std::printf("expert triplet %8.3f ms  %6.2f GB/s  "
                "=> %6.1f GB/s across 28 threads\n",
                total_ms, static_cast<double>(total_bytes) / (total_ms * 1.0e6),
                static_cast<double>(total_bytes) / (total_ms * 1.0e6) * 28.0);
    return 0;
}
