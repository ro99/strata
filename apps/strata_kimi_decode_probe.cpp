// Measures batch-K decode in the real runtime, which is the gate on the draft
// model.
//
// The compute probe measured 3.63x spine amortization at four tokens, but that
// is one matmul in isolation. A decode step is 93 layers of attention, routed
// experts, residual mixing, cache commits, and a 2.19 GiB head, and only some
// of it batches. If the step does not amortize, no draft model can rescue it and
// the arm dies here rather than after two thousand lines.
//
// Arm budget: one 106.55 GiB spine load, roughly 10 minutes on SATA, then four
// arms of a few steps each. The load is the fixed cost and it is paid once, so
// the sweep is what is under test rather than the initialization.

#include "strata/kimi_k3_runtime.hpp"
#include "strata/model_adapter.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr auto& kContract = strata::kKimiK3ExecutionContract;

double median(std::vector<double> samples) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

}  // namespace

int main(int argc, char** argv) {
    std::string model = "/data/kimi-k3";
    std::vector<std::uint32_t> batches{1U, 2U, 4U, 8U};
    std::uint32_t steps = 3U;
    std::size_t workers = 28U;

    for (int index = 1; index < argc; ++index) {
        const std::string flag = argv[index];
        const auto next = [&]() -> std::string {
            return index + 1 < argc ? argv[++index] : std::string{};
        };
        if (flag == "--model") model = next();
        else if (flag == "--steps") steps = static_cast<std::uint32_t>(std::stoul(next()));
        else if (flag == "--workers") workers = static_cast<std::size_t>(std::stoul(next()));
        else {
            std::cerr << "usage: strata-kimi-decode-probe [--model DIR] "
                         "[--steps N] [--workers N]\n";
            return 2;
        }
    }

    strata::KimiK3RuntimeConfig config;
    config.host_workers = workers;
    config.maximum_context_tokens = 256U;
    config.load_progress = true;

    std::cout << "loading " << model << " (dense spine 106.55 GiB)...\n" << std::flush;
    const auto load_begin = std::chrono::steady_clock::now();
    strata::KimiK3Runtime runtime;
    const auto ready = runtime.initialize(model, config);
    if (!ready.ok()) {
        for (const auto& error : ready.errors) std::cerr << "error: " << error << '\n';
        return 1;
    }
    const std::chrono::duration<double> load = std::chrono::steady_clock::now() - load_begin;
    std::cout << "loaded in " << std::fixed << std::setprecision(1) << load.count() / 60.0
              << " min\n\n";

    const auto vocabulary = static_cast<std::size_t>(kContract.vocabulary_size);
    std::cout << std::setw(8) << "tokens" << std::setw(14) << "s/step"
              << std::setw(16) << "s/token" << std::setw(14) << "vs K=1"
              << std::setw(12) << "tok/s" << "\n";

    double single = 0.0;
    for (const auto batch : batches) {
        // A fresh sequence per arm: the KDA half is recurrent, so a step's cost
        // must not depend on what a previous arm left in the state.
        if (!runtime.reset_sequence().ok()) {
            std::cerr << "error: reset failed\n";
            return 1;
        }
        std::vector<std::uint32_t> prompt{1U, 2U, 3U, 4U};
        std::vector<float> seed(vocabulary);
        if (!runtime.evaluate(prompt, 0U, seed).ok()) {
            std::cerr << "error: prefill failed\n";
            return 1;
        }

        // All rows, which is what verification consumes and what makes the head
        // part of the batched work rather than a fixed tail.
        std::vector<std::uint32_t> tokens(batch, 100U);
        std::vector<float> logits(vocabulary * batch);
        auto position = static_cast<std::uint32_t>(prompt.size());

        // One untimed step so the arena holds a routed set and the first timed
        // step is not paying every layer's first miss. Its result is checked:
        // swallowing it left `position` out of step with the runtime's own
        // length and the next arm failed with a confusing continuation error
        // rather than the real one.
        const auto warm = runtime.evaluate(tokens, position, logits);
        if (!warm.ok()) {
            for (const auto& error : warm.errors) {
                std::cerr << "error (warm-up, K=" << batch << "): " << error << '\n';
            }
            return 1;
        }
        position += batch;

        std::vector<double> samples;
        for (std::uint32_t step = 0U; step < steps; ++step) {
            const auto begin = std::chrono::steady_clock::now();
            const auto ok = runtime.evaluate(tokens, position, logits);
            const std::chrono::duration<double> elapsed =
                std::chrono::steady_clock::now() - begin;
            if (!ok.ok()) {
                for (const auto& error : ok.errors) std::cerr << "error: " << error << '\n';
                return 1;
            }
            position += batch;
            samples.push_back(elapsed.count());
        }

        const auto seconds = median(samples);
        const auto per_token = seconds / static_cast<double>(batch);
        if (batch == 1U) single = per_token;
        std::cout << std::setw(8) << batch << std::setw(14) << std::setprecision(2)
                  << seconds << std::setw(16) << std::setprecision(2) << per_token
                  << std::setw(13) << std::setprecision(2)
                  << (single > 0.0 ? single / per_token : 1.0) << "x"
                  << std::setw(12) << std::setprecision(3) << 1.0 / per_token << "\n"
                  << std::flush;
    }

    std::cout << "\nThis is the gate on the draft model. The compute probe "
                 "predicted\n3.63x on the dense spine at four tokens; a step "
                 "also carries routed\nexperts, which do not amortize, so the "
                 "step figure is the one that\ndecides whether speculation can "
                 "pay for a draft pass.\n";
    return 0;
}
