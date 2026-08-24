// Phase profile for one Inkling-Small forward step. Emits the per-phase
// breakdown the tiered-memory cost model is instantiated from, so the
// bottleneck resource is named from measurement rather than assumed.
#include "strata/inkling_runtime.hpp"
#include "strata/chat_protocol.hpp"
#include "strata/tokenizer.hpp"

#include <array>

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
    std::uint32_t repeats = 1U;
    bool host_only = false;
    bool chat = false;
    bool warm_experts = true;
    bool direct_stage = true;
    bool weight_arena = true;
    bool deferred_expert_uploads = true;
    bool expert_parallel = false;
    bool device_attention = true;
    std::uint32_t device_attention_minimum_rows = 512U;
    std::uint32_t context_tokens = 512U;
    std::string route_trace;
    std::vector<int> devices;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--model" && index + 1 < argc) {
            model_directory = argv[++index];
        } else if (argument == "--prompt" && index + 1 < argc) {
            prompt = argv[++index];
        } else if (argument == "--tokens" && index + 1 < argc) {
            new_tokens = static_cast<std::uint32_t>(std::stoul(argv[++index]));
        } else if (argument == "--chat") {
            chat = true;
        } else if (argument == "--host") {
            host_only = true;
        } else if (argument == "--no-warm") {
            warm_experts = false;
        } else if (argument == "--direct-stage") {
            direct_stage = true;
        } else if (argument == "--pinned-stage") {
            direct_stage = false;
        } else if (argument == "--no-weight-arena") {
            weight_arena = false;
        } else if (argument == "--sync-expert-uploads") {
            deferred_expert_uploads = false;
        } else if (argument == "--expert-parallel") {
            expert_parallel = true;
        } else if (argument == "--host-attention") {
            device_attention = false;
        } else if (argument == "--device-attention-min-rows" &&
                   index + 1 < argc) {
            device_attention_minimum_rows =
                static_cast<std::uint32_t>(std::stoul(argv[++index]));
        } else if (argument == "--context" && index + 1 < argc) {
            context_tokens =
                static_cast<std::uint32_t>(std::stoul(argv[++index]));
        } else if (argument == "--route-trace" && index + 1 < argc) {
            route_trace = argv[++index];
        } else if (argument == "--devices" && index + 1 < argc) {
            std::string list = argv[++index];
            std::size_t begin = 0U;
            while (begin < list.size()) {
                const auto comma = list.find(',', begin);
                devices.push_back(std::stoi(list.substr(begin, comma - begin)));
                if (comma == std::string::npos) break;
                begin = comma + 1U;
            }
        } else if (argument == "--repeat" && index + 1 < argc) {
            repeats = static_cast<std::uint32_t>(std::stoul(argv[++index]));
        } else {
            std::fprintf(stderr,
                         "usage: strata-inkling-probe [--model DIR] "
                         "[--prompt TEXT] [--tokens N] [--devices LIST] "
                         "[--no-warm] [--direct-stage|--pinned-stage] "
                         "[--no-weight-arena] "
                         "[--sync-expert-uploads] "
                         "[--expert-parallel] "
                         "[--host-attention] [--device-attention-min-rows N] "
                         "[--context N] [--route-trace PATH]\n");
            return 2;
        }
    }

    if (chat) {
        const std::array<strata::ChatMessage, 1> messages{
            strata::ChatMessage{strata::ChatRole::User, prompt}};
        prompt = strata::render_inkling_chat_prompt(messages);
    }
    strata::InklingRuntimeConfig config;
    config.enable_cuda = !host_only;
    config.devices = devices;
    config.warm_expert_pages = warm_experts;
    config.direct_mapped_mxfp4_staging = direct_stage;
    config.use_weight_arena = weight_arena;
    config.defer_mapped_mxfp4_uploads = deferred_expert_uploads;
    config.enable_expert_parallel = expert_parallel;
    config.enable_device_kv_attention = device_attention;
    config.minimum_device_attention_rows = device_attention_minimum_rows;
    config.maximum_context_tokens = context_tokens;
    config.route_trace_path = route_trace;
    config.load_progress = true;
    strata::InklingRuntime runtime;
    strata::reset_cuda_matmul_route_census();
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

    // Greedy decoding is deterministic, so every repeat routes to exactly the
    // same experts and reads exactly the same bytes. A repeat that is much
    // faster than the first means the term is page-cache misses, not compute;
    // one that is not means it is compute. Nothing else separates the two.
    strata::InklingGenerationResult result;
    for (std::uint32_t repeat = 0U; repeat < repeats; ++repeat) {
        const auto pass_started = std::chrono::steady_clock::now();
        result = runtime.generate_stream(prompt, new_tokens);
        if (!result.ok()) break;
        std::printf("repeat %u: prefill %.2f s, decode %.2f s (%.3f tok/s), "
                    "wall %.2f s\n",
                    repeat, result.metrics.prefill_seconds,
                    result.metrics.decode_seconds,
                    result.metrics.decode_tokens_per_second(),
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - pass_started).count());
        std::fflush(stdout);
    }
    if (!result.ok()) {
        for (const auto& error : result.errors) {
            std::fprintf(stderr, "generate: %s\n", error.c_str());
        }
        return 1;
    }

    // Decode-only phase cost. Prefill runs cold and would otherwise set every
    // share in the table.
    const auto& total_graph = result.metrics.graph;
    const auto& before = result.metrics.prefill_graph;
    strata::InklingGraphStats graph;
    graph.forward_tokens = total_graph.forward_tokens - before.forward_tokens;
    graph.embedding_nanoseconds =
        total_graph.embedding_nanoseconds - before.embedding_nanoseconds;
    graph.attention_nanoseconds =
        total_graph.attention_nanoseconds - before.attention_nanoseconds;
    graph.short_conv_nanoseconds =
        total_graph.short_conv_nanoseconds - before.short_conv_nanoseconds;
    graph.dense_mlp_nanoseconds =
        total_graph.dense_mlp_nanoseconds - before.dense_mlp_nanoseconds;
    graph.moe_router_nanoseconds =
        total_graph.moe_router_nanoseconds - before.moe_router_nanoseconds;
    graph.moe_routed_nanoseconds =
        total_graph.moe_routed_nanoseconds - before.moe_routed_nanoseconds;
    graph.moe_shared_nanoseconds =
        total_graph.moe_shared_nanoseconds - before.moe_shared_nanoseconds;
    graph.output_head_nanoseconds =
        total_graph.output_head_nanoseconds - before.output_head_nanoseconds;
    graph.routed_expert_bytes =
        total_graph.routed_expert_bytes - before.routed_expert_bytes;
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
    std::printf("decode-only per-phase over %llu forward tokens:\n",
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
    const auto& device = result.metrics.device;
    const auto& prefill_device = result.metrics.prefill_device;
    std::printf("cuda %s", device.enabled ? "on" : "off");
    if (device.enabled) {
        std::printf(", expert cache %llu hits / %llu misses (%.1f%%), "
                    "%llu evictions",
                    static_cast<unsigned long long>(device.expert_hits),
                    static_cast<unsigned long long>(device.expert_misses),
                    100.0 * device.hit_rate(),
                    static_cast<unsigned long long>(device.expert_evictions));
        const double staged = static_cast<double>(device.expert_staged_bytes) /
                              (1024.0 * 1024.0 * 1024.0);
        const double seconds =
            static_cast<double>(device.expert_stage_nanoseconds) / 1.0e9;
        std::printf("\n  staged %.2f GiB in %.2f s (%.2f GiB/s H2D)", staged,
                    seconds, seconds > 0.0 ? staged / seconds : 0.0);
        const auto decode_stage_bytes = device.expert_staged_bytes -
                                        prefill_device.expert_staged_bytes;
        const auto decode_stage_ns = device.expert_stage_nanoseconds -
                                     prefill_device.expert_stage_nanoseconds;
        const double decode_stage_gib =
            static_cast<double>(decode_stage_bytes) /
            (1024.0 * 1024.0 * 1024.0);
        const double decode_stage_seconds =
            static_cast<double>(decode_stage_ns) / 1.0e9;
        std::printf("\n  decode staged %.2f GiB in %.3f s (%.2f GiB/s)",
                    decode_stage_gib, decode_stage_seconds,
                    decode_stage_seconds > 0.0
                        ? decode_stage_gib / decode_stage_seconds
                        : 0.0);
        for (std::size_t slot = 0U; slot < device.cache_capacity_bytes.size();
             ++slot) {
            const double capacity =
                static_cast<double>(device.cache_capacity_bytes[slot]) /
                (1024.0 * 1024.0 * 1024.0);
            const double spine =
                slot < device.resident_spine_bytes.size()
                    ? static_cast<double>(device.resident_spine_bytes[slot]) /
                          (1024.0 * 1024.0 * 1024.0)
                    : 0.0;
            std::printf("\n  device %zu: spine %.2f GiB, cache budget %.2f GiB "
                        "(%llu bytes)",
                        slot, spine, capacity,
                        static_cast<unsigned long long>(
                            device.cache_capacity_bytes[slot]));
            if (slot < device.resident_kv_bytes.size()) {
                std::printf(", KV %.2f GiB",
                            static_cast<double>(device.resident_kv_bytes[slot]) /
                                (1024.0 * 1024.0 * 1024.0));
            }
        }
    }
    const auto& cuda = result.metrics.cuda;
    const auto& prefill_cuda = result.metrics.prefill_cuda;
    std::printf("\n  upload split: alloc %.2f s, copy %.2f s, wait %.2f s, "
                "kernel %.2f s\n",
                static_cast<double>(cuda.weight_allocation_nanoseconds) / 1.0e9,
                static_cast<double>(cuda.weight_copy_nanoseconds) / 1.0e9,
                static_cast<double>(cuda.upload_wait_nanoseconds) / 1.0e9,
                static_cast<double>(cuda.kernel_nanoseconds) / 1.0e9);
    std::printf("  decode upload/kernel: alloc %.3f s, copy %.3f s, wait %.3f s, "
                "kernel %.3f s\n",
                static_cast<double>(cuda.weight_allocation_nanoseconds -
                                    prefill_cuda.weight_allocation_nanoseconds) /
                    1.0e9,
                static_cast<double>(cuda.weight_copy_nanoseconds -
                                    prefill_cuda.weight_copy_nanoseconds) /
                    1.0e9,
                static_cast<double>(cuda.upload_wait_nanoseconds -
                                    prefill_cuda.upload_wait_nanoseconds) /
                    1.0e9,
                static_cast<double>(cuda.kernel_nanoseconds -
                                    prefill_cuda.kernel_nanoseconds) /
                    1.0e9);
    std::printf("rss %.2f GiB\n",
                static_cast<double>(result.metrics.rss_bytes) /
                    (1024.0 * 1024.0 * 1024.0));
    // Print the prompt and the continuation delimited. The continuation alone
    // reads as a standalone claim: a truncated enumeration such as
    // " Paris. The capital of Germany" looks like an error when it is a
    // correct answer followed by the start of the next sentence.
    std::printf("prompt: %s\n", prompt.c_str());
    std::printf("continuation: <<%s>>\n", result.text.c_str());
    std::printf("full: %s%s\n", prompt.c_str(), result.text.c_str());
    const auto census = strata::cuda_matmul_route_census();
    std::printf("route census:");
    for (std::size_t index = 0U; index < census.counts.size(); ++index) {
        if (census.counts[index] == 0U) continue;
        std::printf(" %s=%llu",
                    strata::cuda_matmul_route_name(
                        static_cast<strata::CudaMatmulRoute>(index)),
                    static_cast<unsigned long long>(census.counts[index]));
    }
    std::printf("\n");
    return 0;
}
