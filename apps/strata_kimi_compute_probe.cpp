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

#include <numa.h>

namespace {

constexpr auto& kContract = strata::kKimiK3ExecutionContract;

struct Options {
    std::vector<std::size_t> worker_counts;
    std::uint32_t repetitions = 15U;
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

// A measurement and how much it moved. The probe reports the spread because
// without it a 1.5x run-to-run swing reads exactly like a 1.5x optimisation:
// three runs of one binary at identical settings once gave 2.32, 2.32 and 1.51
// s/token here, and that was mistaken for a result. Any arm whose spread is
// comparable to the difference under test has not measured anything.
struct Stats {
    double median{};
    double lower{};
    double upper{};

    // Quartile ratio rather than min/max. The cost here is heavy-tailed: a
    // sample is dozens of barriers and one descheduled thread poisons the whole
    // sample, so the extremes describe the worst scheduling accident rather
    // than the work. The quartiles describe the work.
    [[nodiscard]] double spread() const noexcept {
        return lower > 0.0 ? upper / lower : 0.0;
    }
};

Stats summarize(std::vector<double> samples) {
    if (samples.empty()) return {};
    std::sort(samples.begin(), samples.end());
    const auto at = [&samples](double quantile) {
        const auto index = static_cast<std::size_t>(
            quantile * static_cast<double>(samples.size() - 1U) + 0.5);
        return samples[std::min(index, samples.size() - 1U)];
    };
    return {at(0.5), at(0.25), at(0.75)};
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

    // Interleave every allocation across the NUMA nodes before anything is
    // allocated. Left to first-touch, the synthetic pool lands wherever the
    // staging thread happened to run, so the same binary measured a different
    // machine on each run -- which is what made this probe unable to resolve
    // its own effects. Interleaving is not the fastest policy for every arm; it
    // is the reproducible one, which is what a probe needs.
    bool interleaved = false;
    if (numa_available() != -1) {
        numa_set_interleave_mask(numa_all_nodes_ptr);
        interleaved = true;
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
              << "  hardware threads       " << hardware_threads << "\n"
              << "  NUMA policy            "
              << (interleaved ? "interleaved across all nodes"
                              : "default (first touch) -- results will drift")
              << "\n\n";

    std::cout << "staging " << pool_count << " synthetic experts...\n";
    std::vector<ExpertTriple> pool(pool_count);
    for (auto& triple : pool) triple.reset(engine);

    std::vector<float> source(latent);
    std::uniform_real_distribution<float> values(-1.0F, 1.0F);
    for (auto& value : source) value = values(engine);
    std::vector<float> gate(inner), up(inner), activated(inner), output(latent);

    // Bring the machine to steady state before each timed arm.
    //
    // A fresh process reads high on whatever runs first -- 3.24 s/token against
    // 1.78 for the same arm later -- because the governor is still ramping from
    // idle. Arms also inherit state from the arm before them: the block arm,
    // measured without this, read 1.20 alone and 1.43 in sequence. Whichever
    // arm ran first was being penalised and whichever ran later was being
    // flattered, which is exactly the ordering artefact that gets read as a
    // result. Every sweep now starts from the same place.
    const auto settle = [&]() {
        strata::HostWorkerPool warmup_pool(hardware_threads);
        std::mt19937_64 warmup_selector(options.seed);
        std::uniform_int_distribution<std::size_t> warmup_pick(0U, pool_count - 1U);
        const auto until =
            std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (std::chrono::steady_clock::now() < until) {
            const auto& triple = pool[warmup_pick(warmup_selector)];
            (void)strata::kimi_mxfp4_matvec(gate, source, triple.gate.view(),
                                            &warmup_pool);
            (void)strata::kimi_mxfp4_matvec(up, source, triple.up.view(),
                                            &warmup_pool);
            (void)strata::kimi_mxfp4_matvec(output, activated, triple.down.view(),
                                            &warmup_pool);
        }
    };
    settle();

    std::cout << "\n"
              << std::setw(9) << "workers" << std::setw(14) << "ms/expert"
              << std::setw(14) << "GiB/s" << std::setw(16) << "routed s/token"
              << std::setw(12) << "vs target" << std::setw(12) << "spread"
              << "\n";

    double best_routed_seconds = 0.0;
    std::size_t best_workers = 0U;
    for (const auto workers : options.worker_counts) {
        if (workers == 0U) continue;
        strata::HostWorkerPool pool_of_workers(workers);
        auto* worker_pool = workers > 1U ? &pool_of_workers : nullptr;

        // One untimed pass so the pool's threads are up and the first-touch
        // page faults on the output buffers are already paid.
        //
        // The selector is re-seeded per arm rather than drawn from the shared
        // engine. Sharing it meant each arm walked a different set of experts,
        // so a worker-count comparison was also comparing two different memory
        // access patterns -- which is most of why this arm would not reproduce
        // while the dense arm did.
        std::mt19937_64 selector(options.seed);
        std::uniform_int_distribution<std::size_t> pick(0U, pool_count - 1U);
        for (std::size_t warm = 0U; warm < 4U; ++warm) {
            const auto& triple = pool[pick(selector)];
            (void)strata::kimi_mxfp4_matvec(gate, source, triple.gate.view(), worker_pool);
            (void)strata::kimi_mxfp4_matvec(up, source, triple.up.view(), worker_pool);
            (void)strata::kimi_situ_glu(activated, gate, up, kContract.situ_gate_beta,
                                kContract.situ_linear_beta);
            (void)strata::kimi_mxfp4_matvec(output, activated, triple.down.view(),
                                    worker_pool);
        }

        std::vector<double> samples;
        // A full MoE layer is sixteen experts. Timing four layers per sample
        // keeps the unit the production one while giving the barrier tail
        // enough draws to average out.
        const std::size_t kExpertsPerSample = top_k * 4U;
        for (std::uint32_t repetition = 0U; repetition < options.repetitions;
             ++repetition) {
            const auto begin = std::chrono::steady_clock::now();
            for (std::size_t step = 0U; step < kExpertsPerSample; ++step) {
                const auto& triple = pool[pick(selector)];
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

        const auto expert_stats = summarize(samples);
        const auto seconds_per_expert = expert_stats.median;
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
                  << std::setw(11) << std::setprecision(2)
                  << routed_seconds / options.target_seconds << "x"
                  << std::setw(11) << std::setprecision(2) << expert_stats.spread()
                  << "x" << "\n";
    }

    std::cout << "\nbest routed-expert compute: " << std::setprecision(2)
              << best_routed_seconds << " s/token at " << best_workers
              << " workers\n";

    // ------------------------------------------------------- expert grain
    //
    // The same routed work at the grain the MoE block uses: one whole layer's
    // sixteen experts spread across workers, each expert's three matvecs run
    // single-threaded on the worker that owns it. One barrier per layer instead
    // of forty-eight.
    //
    // Two signatures, not one. If the barrier hypothesis is right this should
    // cut the median *and* the spread, because spread is what dispatch noise
    // produces. A mean that moves while the spread does not would be the same
    // lucky run that got this branch parked the first time.
    std::cout << "\n"
              << std::setw(9) << "workers" << std::setw(14) << "ms/block"
              << std::setw(16) << "routed s/token" << std::setw(12) << "spread"
              << std::setw(14) << "vs matvec" << "\n";

    settle();
    double best_block_seconds = 0.0;
    std::size_t best_block_workers = 0U;
    for (const auto workers : options.worker_counts) {
        if (workers == 0U) continue;
        strata::HostWorkerPool block_pool(workers);
        const auto lanes = std::min<std::size_t>(workers, top_k);
        const auto span = (top_k + lanes - 1U) / lanes;
        std::vector<std::vector<float>> lane_gate(lanes), lane_up(lanes),
            lane_activated(lanes), lane_output(lanes);
        for (std::size_t lane = 0U; lane < lanes; ++lane) {
            lane_gate[lane].assign(inner, 0.0F);
            lane_up[lane].assign(inner, 0.0F);
            lane_activated[lane].assign(inner, 0.0F);
            lane_output[lane].assign(latent, 0.0F);
        }
        std::mt19937_64 selector(options.seed);
        std::uniform_int_distribution<std::size_t> pick(0U, pool_count - 1U);
        std::vector<std::size_t> chosen(top_k);
        const auto run_block = [&](std::size_t lane) {
            const auto begin = lane * span;
            const auto end = std::min(begin + span, top_k);
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
        for (std::size_t warm = 0U; warm < 4U; ++warm) {
            for (auto& value : chosen) value = pick(selector);
            (void)block_pool.parallel_for(lanes, run_block);
        }
        std::vector<double> samples;
        constexpr std::size_t kBlocksPerSample = 4U;
        for (std::uint32_t repetition = 0U; repetition < options.repetitions;
             ++repetition) {
            const auto begin = std::chrono::steady_clock::now();
            for (std::size_t step = 0U; step < kBlocksPerSample; ++step) {
                for (auto& value : chosen) value = pick(selector);
                (void)block_pool.parallel_for(lanes, run_block);
            }
            const std::chrono::duration<double> elapsed =
                std::chrono::steady_clock::now() - begin;
            samples.push_back(elapsed.count() /
                              static_cast<double>(kBlocksPerSample));
        }
        const auto block_stats = summarize(samples);
        const auto routed_seconds =
            block_stats.median * static_cast<double>(moe_layers);
        if (best_block_seconds == 0.0 || routed_seconds < best_block_seconds) {
            best_block_seconds = routed_seconds;
            best_block_workers = workers;
        }
        std::cout << std::setw(9) << workers << std::setw(14)
                  << std::setprecision(3) << block_stats.median * 1000.0
                  << std::setw(16) << std::setprecision(2) << routed_seconds
                  << std::setw(11) << std::setprecision(2) << block_stats.spread()
                  << "x" << std::setw(13) << std::setprecision(2)
                  << best_routed_seconds / routed_seconds << "x" << "\n";
    }
    std::cout << "\nbest routed-expert compute (expert grain): "
              << std::setprecision(2) << best_block_seconds << " s/token at "
              << best_block_workers << " workers\n";

    // --------------------------------------------- prefill token batching
    //
    // The MoE block runs one matvec per (expert, token). A page of P tokens
    // selects each expert about P*16/896 times, so at the current page of 64
    // that is ~1.1 tokens per expert and nothing is re-read -- but at P=512 it
    // is ~9, and each of those nine matvecs walks the same 16.73 MiB again.
    //
    // Whether that costs anything is a cache question, not an arithmetic one,
    // and it decides whether batching an expert's tokens into one pass is worth
    // building. Measured here as cost per (expert, token) against one expert as
    // the token count rises: flat means the re-reads are free and batching buys
    // nothing; falling means the first read dominates and batching converts
    // every later token to the cheap case.
    std::cout << "\n"
              << std::setw(12) << "tokens/expert" << std::setw(16)
              << "ms/expert-token" << std::setw(14) << "GiB/s" << std::setw(14)
              << "vs 1 token" << std::setw(12) << "spread" << "\n";

    settle();
    double single_token_cost = 0.0;
    for (const std::size_t tokens : {1U, 4U, 16U, 64U}) {
        strata::HostWorkerPool batch_pool(hardware_threads);
        std::mt19937_64 selector(options.seed);
        std::uniform_int_distribution<std::size_t> pick(0U, pool_count - 1U);
        std::vector<double> samples;
        for (std::uint32_t repetition = 0U; repetition < options.repetitions;
             ++repetition) {
            const auto& triple = pool[pick(selector)];
            const auto begin = std::chrono::steady_clock::now();
            for (std::size_t token = 0U; token < tokens; ++token) {
                (void)strata::kimi_mxfp4_matvec(gate, source, triple.gate.view(),
                                                &batch_pool);
                (void)strata::kimi_mxfp4_matvec(up, source, triple.up.view(),
                                                &batch_pool);
                (void)strata::kimi_situ_glu(activated, gate, up,
                                            kContract.situ_gate_beta,
                                            kContract.situ_linear_beta);
                (void)strata::kimi_mxfp4_matvec(output, activated,
                                                triple.down.view(), &batch_pool);
            }
            const std::chrono::duration<double> elapsed =
                std::chrono::steady_clock::now() - begin;
            samples.push_back(elapsed.count() / static_cast<double>(tokens));
        }
        const auto stats = summarize(samples);
        if (tokens == 1U) single_token_cost = stats.median;
        std::cout << std::setw(12) << tokens << std::setw(16)
                  << std::setprecision(3) << stats.median * 1000.0 << std::setw(14)
                  << std::setprecision(2)
                  << static_cast<double>(triple_bytes) / stats.median /
                         (1024.0 * 1024.0 * 1024.0)
                  << std::setw(13) << std::setprecision(2)
                  << single_token_cost / stats.median << "x" << std::setw(11)
                  << std::setprecision(2) << stats.spread() << "x" << "\n";
    }

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
                  << std::setw(12) << "spread" << "\n";

        settle();
        double best_dense_seconds = 0.0;
        std::size_t best_dense_workers = 0U;
        for (const auto workers : options.worker_counts) {
            if (workers == 0U) continue;
            strata::HostWorkerPool workers_pool(workers);
            auto* worker_pool = workers > 1U ? &workers_pool : nullptr;
            std::mt19937_64 selector(options.seed);
            std::uniform_int_distribution<std::size_t> pick(0U, matrices - 1U);

            for (std::size_t warm = 0U; warm < 2U; ++warm) {
                const strata::KimiBf16Matrix matrix{dense[pick(selector)], rows,
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
                    const strata::KimiBf16Matrix matrix{dense[pick(selector)], rows,
                                                        columns};
                    (void)strata::kimi_bf16_matmul(dense_output, dense_input, matrix,
                                                   1U, worker_pool);
                }
                const std::chrono::duration<double> elapsed =
                    std::chrono::steady_clock::now() - begin;
                samples.push_back(elapsed.count() /
                                  static_cast<double>(kMatricesPerSample));
            }
            const auto dense_stats = summarize(samples);
            const auto seconds = dense_stats.median;
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
                      << std::setprecision(2) << spine_seconds << std::setw(11)
                      << std::setprecision(2) << dense_stats.spread() << "x"
                      << "\n";
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
