// Phase profile for one Inkling-Small forward step. Emits the per-phase
// breakdown the tiered-memory cost model is instantiated from, so the
// bottleneck resource is named from measurement rather than assumed.
#include "strata/inkling_runtime.hpp"
#include "strata/tokenizer.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

void report(const char* label, std::uint64_t nanoseconds, std::uint64_t total) {
    const double milliseconds = static_cast<double>(nanoseconds) / 1.0e6;
    const double share = total == 0U ? 0.0
                                     : 100.0 * static_cast<double>(nanoseconds) /
                                           static_cast<double>(total);
    std::printf("  %-22s %10.1f ms  %5.1f%%\n", label, milliseconds, share);
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_directory = "models/inkling-s";
    std::string prompt = "The capital of France is";
    std::uint32_t new_tokens = 4U;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--model" && index + 1 < argc) {
            model_directory = argv[++index];
        } else if (argument == "--prompt" && index + 1 < argc) {
            prompt = argv[++index];
        } else if (argument == "--tokens" && index + 1 < argc) {
            new_tokens = static_cast<std::uint32_t>(std::stoul(argv[++index]));
        } else {
            std::fprintf(stderr,
                         "usage: strata-inkling-probe [--model DIR] "
                         "[--prompt TEXT] [--tokens N]\n");
            return 2;
        }
    }

    strata::InklingRuntimeConfig config;
    config.maximum_context_tokens = 512U;
    config.load_progress = true;
    strata::InklingRuntime runtime;
    const auto started = std::chrono::steady_clock::now();
    const auto initialized = runtime.initialize(model_directory, config);
    if (!initialized.ok()) {
        for (const auto& error : initialized.errors) {
            std::fprintf(stderr, "initialize: %s\n", error.c_str());
        }
        return 1;
    }
    std::printf("load %.2f s\n",
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count());

    const auto result = runtime.generate_stream(prompt, new_tokens);
    if (!result.ok()) {
        for (const auto& error : result.errors) {
            std::fprintf(stderr, "generate: %s\n", error.c_str());
        }
        return 1;
    }

    const auto& graph = result.metrics.graph;
    const auto total = graph.embedding_nanoseconds + graph.attention_nanoseconds +
                       graph.short_conv_nanoseconds + graph.dense_mlp_nanoseconds +
                       graph.moe_router_nanoseconds + graph.moe_routed_nanoseconds +
                       graph.moe_shared_nanoseconds + graph.output_head_nanoseconds;
    std::printf("prompt %llu tokens, generated %llu tokens\n",
                static_cast<unsigned long long>(result.metrics.prompt_tokens),
                static_cast<unsigned long long>(result.metrics.decode_tokens));
    std::printf("prefill %.2f s (%.3f tok/s), decode %.2f s (%.3f tok/s)\n",
                result.metrics.prefill_seconds,
                result.metrics.prefill_tokens_per_second(),
                result.metrics.decode_seconds,
                result.metrics.decode_tokens_per_second());
    std::printf("per-phase over %llu forward tokens:\n",
                static_cast<unsigned long long>(graph.forward_tokens));
    report("embedding", graph.embedding_nanoseconds, total);
    report("attention", graph.attention_nanoseconds, total);
    report("short conv", graph.short_conv_nanoseconds, total);
    report("dense mlp", graph.dense_mlp_nanoseconds, total);
    report("moe router", graph.moe_router_nanoseconds, total);
    report("moe routed experts", graph.moe_routed_nanoseconds, total);
    report("moe shared experts", graph.moe_shared_nanoseconds, total);
    report("output head", graph.output_head_nanoseconds, total);
    const double expert_gib = static_cast<double>(graph.routed_expert_bytes) /
                              (1024.0 * 1024.0 * 1024.0);
    std::printf("routed expert bytes touched: %.2f GiB (%.2f GiB/token)\n",
                expert_gib,
                graph.forward_tokens == 0U
                    ? 0.0
                    : expert_gib / static_cast<double>(graph.forward_tokens));
    std::printf("rss %.2f GiB\n",
                static_cast<double>(result.metrics.rss_bytes) /
                    (1024.0 * 1024.0 * 1024.0));
    std::printf("text: %s\n", result.text.c_str());
    return 0;
}
