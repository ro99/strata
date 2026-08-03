// Measures the Kimi-K3 host compute terms of a batch-1 decode step, in
// isolation from storage.
//
// Why this exists: every Kimi-K3 measurement so far ran at ~70 s/token, where
// the SATA read term is so dominant that no other term is observable. That made
// the compute side an assumption, and the charter's first step is that a
// mechanism cannot be chosen before `argmax_r` is known. Before any GPU work is
// designed — and before a faster drive is bought — the question is whether host
// compute alone already exceeds the target step time. If it does, no storage
// upgrade reaches the target and the answer is offload, not a disk.
//
// The probe reproduces the production compute pattern and nothing else: weights
// already resident in host RAM, read once per token by a matvec that decodes
// them in place. Weight *values* do not change the cost, so the buffers are
// synthetic; what does change it is that they are cold in RAM rather than warm
// in cache, so the pool is sized far past L3 and walked in random order.
//
// Two terms per token, both memory-bound:
//   routed  16 experts x 92 MoE layers x (gate + up + down), MXFP4
//   dense   the BF16 spine, every weight read once per decoded token

#include "strata/kimi_k3_layer.hpp"
#include "strata/kimi_k3_ops.hpp"
#include "strata/model_adapter.hpp"
#include "strata/worker_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr auto& kContract = strata::kKimiK3ExecutionContract;

struct Options {
    std::vector<std::size_t> worker_counts;
    std::uint32_t repetitions = 5U;
    std::uint64_t pool_bytes = 4ULL << 30U;  // past any cache, so reads are cold
    std::uint64_t seed = 20260803U;
    double target_seconds = 5.0;  // 0.2 tok/s
};

void usage() {
    std::cout
        << "usage: strata-kimi-compute-probe [--workers 8,28,56]\n"
           "         [--repetitions N] [--pool-bytes N] [--target-seconds S]\n\n"
           "Reports the median host compute cost of one batch-1 decode step,\n"
           "split into the routed-expert term and the dense-spine term, and\n"
           "compares the total against the target step time.\n";
}

std::vector<std::size_t> parse_list(const std::string& text) {
    std::vector<std::size_t> values;
    std::size_t begin = 0U;
    while (begin <= text.size()) {
        const auto comma = text.find(',', begin);
        const auto piece = text.substr(
            begin, comma == std::string::npos ? std::string::npos : comma - begin);
        if (!piece.empty()) {
            values.push_back(static_cast<std::size_t>(std::stoul(piece)));
        }
        if (comma == std::string::npos) break;
        begin = comma + 1U;
    }
    return values;
}

double median(std::vector<double> samples) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

// One MXFP4 module of `rows x columns`, laid out exactly as the arena holds it:
// two E2M1 per byte, one E8M0 scale per 32-element group.
struct PackedModule {
    std::vector<std::uint8_t> packed;
    std::vector<std::uint8_t> scales;
    std::uint32_t rows{};
    std::uint32_t columns{};

    void reset(std::uint32_t row_count, std::uint32_t column_count,
               std::mt19937_64& engine) {
        rows = row_count;
        columns = column_count;
        packed.resize(static_cast<std::size_t>(rows) * columns / 2U);
        scales.resize(static_cast<std::size_t>(rows) * columns / 32U);
        // A biased-127 exponent keeps the decoded values around unity, so the
        // matvec does the same work it does on real weights without denormals.
        std::uniform_int_distribution<int> nibbles(0, 255);
        for (auto& byte : packed) {
            byte = static_cast<std::uint8_t>(nibbles(engine));
        }
        std::fill(scales.begin(), scales.end(), static_cast<std::uint8_t>(127U));
    }

    [[nodiscard]] strata::KimiExpertModuleView view() const noexcept {
        return {packed, scales, rows, columns};
    }

    [[nodiscard]] std::uint64_t bytes() const noexcept {
        return packed.size() + scales.size();
    }
};

struct ExpertTriple {
    PackedModule gate;
    PackedModule up;
    PackedModule down;

    void reset(std::mt19937_64& engine) {
        const auto latent = kContract.routed_expert_hidden_size;
        const auto inner = kContract.expert_intermediate_size;
        gate.reset(inner, latent, engine);
        up.reset(inner, latent, engine);
        down.reset(latent, inner, engine);
    }

    [[nodiscard]] std::uint64_t bytes() const noexcept {
        return gate.bytes() + up.bytes() + down.bytes();
    }
};

}  // namespace

int main(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string flag = argv[index];
        const auto next = [&]() -> std::string {
            return index + 1 < argc ? argv[++index] : std::string{};
        };
        if (flag == "--help" || flag == "-h") {
            usage();
            return 0;
        } else if (flag == "--workers") {
            options.worker_counts = parse_list(next());
        } else if (flag == "--repetitions") {
            options.repetitions = static_cast<std::uint32_t>(std::stoul(next()));
        } else if (flag == "--pool-bytes") {
            options.pool_bytes = std::stoull(next());
        } else if (flag == "--target-seconds") {
            options.target_seconds = std::stod(next());
        } else if (flag == "--seed") {
            options.seed = std::stoull(next());
        } else {
            std::cerr << "unknown option " << flag << "\n";
            usage();
            return 2;
        }
    }

    const auto hardware_threads =
        static_cast<std::size_t>(std::max(1U, std::thread::hardware_concurrency()));
    if (options.worker_counts.empty()) {
        options.worker_counts = {std::size_t{1U}, hardware_threads / 4U,
                                 hardware_threads / 2U, hardware_threads};
        std::sort(options.worker_counts.begin(), options.worker_counts.end());
        options.worker_counts.erase(
            std::unique(options.worker_counts.begin(), options.worker_counts.end()),
            options.worker_counts.end());
    }

    const auto latent = static_cast<std::size_t>(kContract.routed_expert_hidden_size);
    const auto inner = static_cast<std::size_t>(kContract.expert_intermediate_size);
    const auto top_k = static_cast<std::size_t>(kContract.experts_per_token);
    const auto moe_layers = static_cast<std::size_t>(
        kContract.layer_count - kContract.dense_prefix_layers);

    std::mt19937_64 engine(options.seed);
    ExpertTriple probe_triple;
    probe_triple.reset(engine);
    const auto triple_bytes = probe_triple.bytes();

    // Enough distinct experts to overflow every cache level, so each matvec
    // pulls its weights from DRAM the way a real routed step does.
    const auto pool_count = std::max<std::size_t>(
        4U, static_cast<std::size_t>(options.pool_bytes / triple_bytes));

    std::cout << "Kimi-K3 host compute probe\n"
              << "  routed expert triple   " << std::fixed << std::setprecision(2)
              << static_cast<double>(triple_bytes) / (1024.0 * 1024.0) << " MiB\n"
              << "  distinct experts held  " << pool_count << " ("
              << static_cast<double>(pool_count * triple_bytes) /
                     (1024.0 * 1024.0 * 1024.0)
              << " GiB, cold in RAM)\n"
              << "  routed set per token   " << top_k << " experts x " << moe_layers
              << " MoE layers = " << (top_k * moe_layers) << " expert-tokens, "
              << static_cast<double>(top_k * moe_layers * triple_bytes) /
                     (1024.0 * 1024.0 * 1024.0)
              << " GiB\n"
              << "  target step time       " << std::setprecision(2)
              << options.target_seconds << " s ("
              << 1.0 / options.target_seconds << " tok/s)\n"
              << "  hardware threads       " << hardware_threads << "\n\n";

    std::cout << "staging " << pool_count << " synthetic experts...\n";
    std::vector<ExpertTriple> pool(pool_count);
    for (auto& triple : pool) triple.reset(engine);

    std::vector<float> source(latent);
    std::uniform_real_distribution<float> values(-1.0F, 1.0F);
    for (auto& value : source) value = values(engine);
    std::vector<float> gate(inner), up(inner), activated(inner), output(latent);

    std::cout << "\n"
              << std::setw(9) << "workers" << std::setw(14) << "ms/expert"
              << std::setw(14) << "GiB/s" << std::setw(16) << "routed s/token"
              << std::setw(14) << "vs target" << "\n";

    double best_routed_seconds = 0.0;
    std::size_t best_workers = 0U;
    for (const auto workers : options.worker_counts) {
        if (workers == 0U) continue;
        strata::HostWorkerPool pool_of_workers(workers);
        auto* worker_pool = workers > 1U ? &pool_of_workers : nullptr;

        // One untimed pass so the pool's threads are up and the first-touch
        // page faults on the output buffers are already paid.
        std::uniform_int_distribution<std::size_t> pick(0U, pool_count - 1U);
        for (std::size_t warm = 0U; warm < 4U; ++warm) {
            const auto& triple = pool[pick(engine)];
            (void)strata::kimi_mxfp4_matvec(gate, source, triple.gate.view(), worker_pool);
            (void)strata::kimi_mxfp4_matvec(up, source, triple.up.view(), worker_pool);
            (void)strata::kimi_situ_glu(activated, gate, up, kContract.situ_gate_beta,
                                kContract.situ_linear_beta);
            (void)strata::kimi_mxfp4_matvec(output, activated, triple.down.view(),
                                    worker_pool);
        }

        std::vector<double> samples;
        constexpr std::size_t kExpertsPerSample = 32U;
        for (std::uint32_t repetition = 0U; repetition < options.repetitions;
             ++repetition) {
            const auto begin = std::chrono::steady_clock::now();
            for (std::size_t step = 0U; step < kExpertsPerSample; ++step) {
                const auto& triple = pool[pick(engine)];
                (void)strata::kimi_mxfp4_matvec(gate, source, triple.gate.view(),
                                        worker_pool);
                (void)strata::kimi_mxfp4_matvec(up, source, triple.up.view(), worker_pool);
                (void)strata::kimi_situ_glu(activated, gate, up, kContract.situ_gate_beta,
                                    kContract.situ_linear_beta);
                (void)strata::kimi_mxfp4_matvec(output, activated, triple.down.view(),
                                        worker_pool);
            }
            const std::chrono::duration<double> elapsed =
                std::chrono::steady_clock::now() - begin;
            samples.push_back(elapsed.count() /
                              static_cast<double>(kExpertsPerSample));
        }

        const auto seconds_per_expert = median(samples);
        const auto gib_per_second =
            static_cast<double>(triple_bytes) / seconds_per_expert /
            (1024.0 * 1024.0 * 1024.0);
        const auto routed_seconds =
            seconds_per_expert * static_cast<double>(top_k * moe_layers);
        if (best_routed_seconds == 0.0 || routed_seconds < best_routed_seconds) {
            best_routed_seconds = routed_seconds;
            best_workers = workers;
        }

        std::cout << std::setw(9) << workers << std::setw(14)
                  << std::setprecision(3) << seconds_per_expert * 1000.0
                  << std::setw(14) << std::setprecision(2) << gib_per_second
                  << std::setw(16) << std::setprecision(2) << routed_seconds
                  << std::setw(13) << std::setprecision(2)
                  << routed_seconds / options.target_seconds << "x" << "\n";
    }

    std::cout << "\nbest routed-expert compute (matvec grain): "
              << std::setprecision(2) << best_routed_seconds << " s/token at "
              << best_workers << " workers\n";

    // ------------------------------------------------- expert grain
    //
    // The same work at the grain the MoE block now uses: one whole layer's
    // sixteen experts spread across workers, each expert's three matvecs run
    // single-threaded on the worker that owns it. One barrier per layer
    // instead of forty-eight.
    std::cout << "\n"
              << std::setw(9) << "workers" << std::setw(14) << "ms/block"
              << std::setw(16) << "routed s/token" << std::setw(16)
              << "vs matvec grain" << "\n";

    double best_block_seconds = 0.0;
    std::size_t best_block_workers = 0U;
    for (const auto workers : options.worker_counts) {
        if (workers == 0U) continue;
        strata::HostWorkerPool block_pool(workers);
        const auto lanes = std::min<std::size_t>(workers, top_k);
        const auto chunk = (top_k + lanes - 1U) / lanes;
        std::vector<std::vector<float>> lane_gate(lanes), lane_up(lanes),
            lane_activated(lanes), lane_output(lanes);
        for (std::size_t lane = 0U; lane < lanes; ++lane) {
            lane_gate[lane].assign(inner, 0.0F);
            lane_up[lane].assign(inner, 0.0F);
            lane_activated[lane].assign(inner, 0.0F);
            lane_output[lane].assign(latent, 0.0F);
        }

        std::uniform_int_distribution<std::size_t> pick(0U, pool_count - 1U);
        std::vector<std::size_t> chosen(top_k);
        const auto run_block = [&](std::size_t lane) {
            const auto begin = lane * chunk;
            const auto end = std::min(begin + chunk, top_k);
            for (std::size_t slot = begin; slot < end; ++slot) {
                const auto& triple = pool[chosen[slot]];
                (void)strata::kimi_mxfp4_matvec(lane_gate[lane], source,
                                                triple.gate.view());
                (void)strata::kimi_mxfp4_matvec(lane_up[lane], source,
                                                triple.up.view());
                (void)strata::kimi_situ_glu(lane_activated[lane], lane_gate[lane],
                                            lane_up[lane], kContract.situ_gate_beta,
                                            kContract.situ_linear_beta);
                (void)strata::kimi_mxfp4_matvec(lane_output[lane],
                                                lane_activated[lane],
                                                triple.down.view());
            }
        };

        for (std::size_t warm = 0U; warm < 2U; ++warm) {
            for (auto& value : chosen) value = pick(engine);
            (void)block_pool.parallel_for(lanes, run_block);
        }

        std::vector<double> samples;
        constexpr std::size_t kBlocksPerSample = 4U;
        for (std::uint32_t repetition = 0U; repetition < options.repetitions;
             ++repetition) {
            const auto begin = std::chrono::steady_clock::now();
            for (std::size_t step = 0U; step < kBlocksPerSample; ++step) {
                for (auto& value : chosen) value = pick(engine);
                (void)block_pool.parallel_for(lanes, run_block);
            }
            const std::chrono::duration<double> elapsed =
                std::chrono::steady_clock::now() - begin;
            samples.push_back(elapsed.count() /
                              static_cast<double>(kBlocksPerSample));
        }
        const auto seconds_per_block = median(samples);
        const auto routed_seconds =
            seconds_per_block * static_cast<double>(moe_layers);
        if (best_block_seconds == 0.0 || routed_seconds < best_block_seconds) {
            best_block_seconds = routed_seconds;
            best_block_workers = workers;
        }
        std::cout << std::setw(9) << workers << std::setw(14)
                  << std::setprecision(3) << seconds_per_block * 1000.0
                  << std::setw(16) << std::setprecision(2) << routed_seconds
                  << std::setw(15) << std::setprecision(2)
                  << best_routed_seconds / routed_seconds << "x" << "\n";
    }
    std::cout << "\nbest routed-expert compute (expert grain): "
              << std::setprecision(2) << best_block_seconds << " s/token at "
              << best_block_workers << " workers\n";
    best_routed_seconds = std::min(best_routed_seconds, best_block_seconds);

    // ------------------------------------------------------- dense spine
    //
    // The routed set is not the larger term. `docs/experiments/0048` measures
    // the BF16 dense spine at 106.55 GiB, and batch-1 decode reads every one of
    // those bytes once per token -- 4.4x the routed traffic. It had never been
    // profiled, because at ~70 s/token nothing but storage was visible.
    constexpr double kSpineGiB = 106.55;
    {
        const auto rows = static_cast<std::uint32_t>(kContract.hidden_size);
        const auto columns = static_cast<std::uint32_t>(kContract.hidden_size);
        const auto matrix_values = static_cast<std::size_t>(rows) * columns;
        const auto matrix_bytes = matrix_values * sizeof(std::uint16_t);
        const auto matrices = std::max<std::size_t>(
            2U, static_cast<std::size_t>(options.pool_bytes / matrix_bytes));

        std::cout << "\nstaging " << matrices << " synthetic BF16 matrices ("
                  << std::setprecision(2)
                  << static_cast<double>(matrices * matrix_bytes) /
                         (1024.0 * 1024.0 * 1024.0)
                  << " GiB, cold in RAM)...\n";
        std::vector<std::vector<std::uint16_t>> dense(matrices);
        for (auto& matrix : dense) {
            matrix.resize(matrix_values);
            // A BF16 exponent near the bias, so decoded values sit around unity
            // and the multiply does the same work it does on real weights.
            for (auto& value : matrix) {
                value = static_cast<std::uint16_t>(0x3F00U | (engine() & 0x7FU));
            }
        }
        std::vector<float> dense_input(columns, 0.5F);
        std::vector<float> dense_output(rows, 0.0F);

        std::cout << "\n"
                  << std::setw(9) << "workers" << std::setw(14) << "ms/matrix"
                  << std::setw(14) << "GiB/s" << std::setw(16) << "spine s/token"
                  << "\n";

        double best_dense_seconds = 0.0;
        std::size_t best_dense_workers = 0U;
        for (const auto workers : options.worker_counts) {
            if (workers == 0U) continue;
            strata::HostWorkerPool workers_pool(workers);
            auto* worker_pool = workers > 1U ? &workers_pool : nullptr;
            std::uniform_int_distribution<std::size_t> pick(0U, matrices - 1U);

            for (std::size_t warm = 0U; warm < 2U; ++warm) {
                const strata::KimiBf16Matrix matrix{dense[pick(engine)], rows,
                                                    columns};
                (void)strata::kimi_bf16_matmul(dense_output, dense_input, matrix, 1U,
                                               worker_pool);
            }
            std::vector<double> samples;
            constexpr std::size_t kMatricesPerSample = 8U;
            for (std::uint32_t repetition = 0U; repetition < options.repetitions;
                 ++repetition) {
                const auto begin = std::chrono::steady_clock::now();
                for (std::size_t step = 0U; step < kMatricesPerSample; ++step) {
                    const strata::KimiBf16Matrix matrix{dense[pick(engine)], rows,
                                                        columns};
                    (void)strata::kimi_bf16_matmul(dense_output, dense_input, matrix,
                                                   1U, worker_pool);
                }
                const std::chrono::duration<double> elapsed =
                    std::chrono::steady_clock::now() - begin;
                samples.push_back(elapsed.count() /
                                  static_cast<double>(kMatricesPerSample));
            }
            const auto seconds = median(samples);
            const auto gib_per_second = static_cast<double>(matrix_bytes) / seconds /
                                        (1024.0 * 1024.0 * 1024.0);
            const auto spine_seconds = kSpineGiB / gib_per_second;
            if (best_dense_seconds == 0.0 || spine_seconds < best_dense_seconds) {
                best_dense_seconds = spine_seconds;
                best_dense_workers = workers;
            }
            std::cout << std::setw(9) << workers << std::setw(14)
                      << std::setprecision(3) << seconds * 1000.0 << std::setw(14)
                      << std::setprecision(2) << gib_per_second << std::setw(16)
                      << std::setprecision(2) << spine_seconds << "\n";
        }

        std::cout << "\nbest dense-spine compute: " << std::setprecision(2)
                  << best_dense_seconds << " s/token at " << best_dense_workers
                  << " workers (" << kSpineGiB << " GiB read once per token)\n";
        std::cout << "\nhost compute floor, both terms: " << std::setprecision(2)
                  << best_routed_seconds + best_dense_seconds << " s/token\n";
        best_routed_seconds += best_dense_seconds;
    }
    std::cout << "\nThis is the compute term alone, with every weight already in\n"
                 "host RAM. Storage is not in it. Any storage upgrade leaves it\n"
                 "unchanged, so it is a floor on the step time.\n";
    if (best_routed_seconds > options.target_seconds) {
        std::cout << "\nVERDICT: host routed compute alone exceeds the target.\n"
                     "A faster disk cannot reach " << options.target_seconds
                  << " s/token; the expert path has to leave the CPU.\n";
    } else {
        std::cout << "\nVERDICT: host routed compute fits under the target with "
                  << std::setprecision(2)
                  << options.target_seconds - best_routed_seconds
                  << " s/token of headroom for storage and the dense spine.\n";
    }
    return 0;
}
