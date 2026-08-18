// Prices one static-expert-tier call in isolation.
//
// The tier's first implementation measured a >30x end-to-end regression, which
// is far more than its own design could account for, so the per-call cost has
// to be measured before anything is concluded about the design. This probe
// does exactly what the tier does for one layer -- upload one hidden row, run N
// routed FP4 experts already resident on the device, download the routed rows,
// synchronize -- and reports the cost per call.
//
// It loads only the experts it needs, so it runs in seconds rather than behind
// a 120-second model stage. That is the measurement that should have come
// first.

#include "cli_common.hpp"

#include "strata/cuda_backend.hpp"
#include "strata/deepseek_checkpoint.hpp"
#include "strata/model_adapter.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kHidden =
    static_cast<std::size_t>(strata::kDeepSeekV4ExecutionContract.hidden_size);
constexpr std::size_t kIntermediate = static_cast<std::size_t>(
    strata::kDeepSeekV4ExecutionContract.expert_intermediate_size);

struct Options {
    std::string model;
    int device{0};
    std::uint32_t experts{6U};
    std::uint32_t iterations{200U};
};

void usage() {
    std::cerr << "usage: strata-dsv4-static-tier-probe --model DIR "
                 "[--device N] [--experts N] [--iterations N]\n";
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--help" || argument == "-h") { usage(); std::exit(0); }
        if (index + 1 >= argc) return false;
        const std::string_view value(argv[++index]);
        std::uint32_t parsed = 0U;
        if (argument == "--model") options.model = value;
        else if (argument == "--device") {
            if (!strata::cli::parse_u32(value, parsed)) return false;
            options.device = static_cast<int>(parsed);
        } else if (argument == "--experts") {
            if (!strata::cli::parse_positive_u32(value, options.experts)) return false;
        } else if (argument == "--iterations") {
            if (!strata::cli::parse_positive_u32(value, options.iterations)) return false;
        } else return false;
    }
    return !options.model.empty();
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) { usage(); return 2; }

    auto opened = strata::Dsv4CheckpointReader::open(options.model, false);
    if (!opened.ok()) {
        for (const auto& error : opened.errors) std::cerr << "error: " << error << "\n";
        return 1;
    }
    auto& checkpoint = *opened.value;

    strata::CudaBackend backend;
    const std::array<int, 1U> devices{options.device};
    if (auto started = backend.initialize(devices, true); !started.ok()) {
        for (const auto& error : started.errors) std::cerr << "error: " << error << "\n";
        return 1;
    }
    // Room for the triplets plus headroom, matching how the tier reserves.
    const std::uint64_t arena =
        static_cast<std::uint64_t>(options.experts) * 20ULL * (1ULL << 20U);
    if (auto reserved = backend.reserve_weight_arena(options.device, arena);
        !reserved.ok()) {
        for (const auto& error : reserved.errors) std::cerr << "error: " << error << "\n";
        return 1;
    }

    std::vector<strata::CudaWeight> weights(options.experts * 3U);
    std::vector<strata::CudaDeepSeekMoeExpert> routed(options.experts);
    for (std::uint32_t index = 0U; index < options.experts; ++index) {
        const auto prefix = "layers.0.ffn.experts." + std::to_string(index) + ".";
        const auto load = [&](const char* suffix, std::uint64_t rows,
                              std::uint64_t columns, strata::CudaWeight& into) {
            auto loaded = strata::load_dsv4_cuda_linear(
                checkpoint, nullptr, prefix + suffix, rows, columns,
                options.device, backend, into, false);
            if (!loaded.ok()) {
                for (const auto& error : loaded.errors) {
                    std::cerr << "error: " << prefix << suffix << ": " << error << "\n";
                }
                return false;
            }
            return true;
        };
        auto& w1 = weights[index * 3U];
        auto& w3 = weights[index * 3U + 1U];
        auto& w2 = weights[index * 3U + 2U];
        if (!load("w1", kIntermediate, kHidden, w1) ||
            !load("w3", kIntermediate, kHidden, w3) ||
            !load("w2", kHidden, kIntermediate, w2)) {
            return 1;
        }
        routed[index].w1 = &w1;
        routed[index].w3 = &w3;
        routed[index].w2 = &w2;
        routed[index].coefficient = 1.0F;
    }

    std::vector<float> input(kHidden, 0.01F);
    std::vector<float> output(static_cast<std::size_t>(options.experts) * kHidden, 0.0F);
    const auto limit = strata::kDeepSeekV4ExecutionContract.swiglu_limit;

    const auto one_call = [&]() -> bool {
        auto enqueued = backend.enqueue_deepseek_moe(
            options.device, input, routed, nullptr, limit);
        if (!enqueued.ok()) {
            for (const auto& error : enqueued.errors) std::cerr << "error: " << error << "\n";
            return false;
        }
        auto collected = backend.collect_deepseek_moe(options.device, output, {});
        if (!collected.ok()) {
            for (const auto& error : collected.errors) std::cerr << "error: " << error << "\n";
            return false;
        }
        return true;
    };

    for (std::uint32_t warm = 0U; warm < 20U; ++warm) {
        if (!one_call()) return 1;
    }
    std::vector<double> samples;
    samples.reserve(options.iterations);
    for (std::uint32_t iteration = 0U; iteration < options.iterations; ++iteration) {
        const auto started = std::chrono::steady_clock::now();
        if (!one_call()) return 1;
        samples.push_back(
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count());
    }
    std::sort(samples.begin(), samples.end());
    const auto median = samples[samples.size() / 2U];
    std::printf(
        "device %d | %u routed experts, 1 row | %u iterations\n",
        options.device, options.experts, options.iterations);
    std::printf(
        "  per call: median %.4f ms   min %.4f   max %.4f\n",
        median, samples.front(), samples.back());
    std::printf(
        "  43 layers/token -> %.2f ms/token of tier calls\n", median * 43.0);
    return 0;
}
