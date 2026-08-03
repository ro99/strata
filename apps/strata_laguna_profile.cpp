// Instantiates the governing cost model for Laguna S 2.1-NVFP4 decode:
// measures W_r and B_r for every resource at a chosen operating point and
// prints the per-phase breakdown of one step in milliseconds, so argmax_r is
// named from measurement rather than assumed.
#include "strata/laguna_runtime.hpp"

#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::string model{"models/laguna-s-21"};
    std::vector<int> devices{0, 1, 2};
    std::uint32_t context{512U};
    std::uint32_t max_new{16U};
    std::uint32_t repetitions{1U};
    double vram_fraction{0.85};
    std::string prompt{
        "Explain in one short paragraph why a hash map gives constant expected "
        "lookup time."};
};

bool parse_u32(std::string_view text, std::uint32_t& output) {
    const auto* end = text.data() + text.size();
    return std::from_chars(text.data(), end, output).ptr == end;
}

bool parse_devices(std::string_view text, std::vector<int>& output) {
    output.clear();
    std::size_t start = 0U;
    while (start <= text.size()) {
        const auto comma = text.find(',', start);
        const auto piece = text.substr(
            start, comma == std::string_view::npos ? text.size() - start
                                                   : comma - start);
        std::uint32_t value = 0U;
        if (piece.empty() || !parse_u32(piece, value)) return false;
        output.push_back(static_cast<int>(value));
        if (comma == std::string_view::npos) break;
        start = comma + 1U;
    }
    return !output.empty();
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view flag(argv[index]);
        const auto next = [&]() -> std::string_view {
            return index + 1 < argc ? std::string_view(argv[++index])
                                    : std::string_view();
        };
        if (flag == "--model") options.model = std::string(next());
        else if (flag == "--devices") { if (!parse_devices(next(), options.devices)) return false; }
        else if (flag == "--context") { if (!parse_u32(next(), options.context)) return false; }
        else if (flag == "--max-new") { if (!parse_u32(next(), options.max_new)) return false; }
        else if (flag == "--repetitions") { if (!parse_u32(next(), options.repetitions)) return false; }
        else if (flag == "--vram-fraction") options.vram_fraction = std::atof(std::string(next()).c_str());
        else if (flag == "--prompt") options.prompt = std::string(next());
        else return false;
    }
    return !options.model.empty() && options.repetitions != 0U;
}

double milliseconds(std::uint64_t nanoseconds, std::uint64_t steps) {
    return steps == 0U ? 0.0
                       : static_cast<double>(nanoseconds) /
                             (1.0e6 * static_cast<double>(steps));
}

double mebibytes(std::uint64_t bytes, std::uint64_t steps) {
    return steps == 0U ? 0.0
                       : static_cast<double>(bytes) /
                             (1048576.0 * static_cast<double>(steps));
}

// B_r in GB/s from the bytes and the service time actually spent moving them.
double gigabytes_per_second(std::uint64_t bytes, std::uint64_t nanoseconds) {
    return nanoseconds == 0U ? 0.0
                             : static_cast<double>(bytes) /
                                   static_cast<double>(nanoseconds);
}

void report_phase(std::string_view label, std::uint64_t nanoseconds,
                  std::uint64_t steps, double step_milliseconds) {
    const double value = milliseconds(nanoseconds, steps);
    std::cout << "  " << std::left << std::setw(28) << label << std::right
              << std::setw(9) << std::fixed << std::setprecision(2) << value
              << " ms  " << std::setw(6) << std::setprecision(1)
              << (step_milliseconds > 0.0 ? 100.0 * value / step_milliseconds
                                          : 0.0)
              << " %\n";
}

void report(const strata::LagunaPhaseMetrics& phase, std::uint64_t steps,
            double seconds, std::string_view title) {
    const double step_milliseconds =
        steps == 0U ? 0.0 : 1000.0 * seconds / static_cast<double>(steps);
    std::cout << "\n== " << title << " ==\n"
              << "steps=" << steps << "  wall=" << std::fixed
              << std::setprecision(3) << seconds << " s  per-step="
              << std::setprecision(2) << step_milliseconds << " ms\n";

    const auto& graph = phase.graph;
    std::cout << "-- graph phases (host wall, per step) --\n";
    report_phase("embedding", graph.embedding_nanoseconds, steps, step_milliseconds);
    report_phase("attention", graph.attention_nanoseconds, steps, step_milliseconds);
    report_phase("  projections q/k/v/g", graph.attention_projection_nanoseconds, steps, step_milliseconds);
    report_phase("  qk-norm + rope (host)", graph.attention_rope_nanoseconds, steps, step_milliseconds);
    report_phase("  kv restage (host)", graph.attention_kv_stage_nanoseconds, steps, step_milliseconds);
    report_phase("  flash attention", graph.attention_flash_nanoseconds, steps, step_milliseconds);
    report_phase("  gate + o_proj", graph.attention_output_nanoseconds, steps, step_milliseconds);
    report_phase("dense mlp (layer 0)", graph.dense_mlp_nanoseconds, steps, step_milliseconds);
    report_phase("moe router", graph.moe_router_nanoseconds, steps, step_milliseconds);
    report_phase("moe routed", graph.moe_routed_nanoseconds, steps, step_milliseconds);
    report_phase("  gather (slowest device)", graph.moe_gather_nanoseconds, steps, step_milliseconds);
    report_phase("  experts (slowest device)", graph.moe_expert_nanoseconds, steps, step_milliseconds);
    report_phase("    of which cache calls", graph.moe_expert_cache_nanoseconds, steps, step_milliseconds);
    report_phase("  accumulate + scale", graph.moe_accumulate_nanoseconds, steps, step_milliseconds);
    report_phase("moe shared expert", graph.moe_shared_nanoseconds, steps, step_milliseconds);
    report_phase("output head", graph.output_head_nanoseconds, steps, step_milliseconds);
    const auto accounted =
        graph.embedding_nanoseconds + graph.attention_nanoseconds +
        graph.dense_mlp_nanoseconds + graph.moe_router_nanoseconds +
        graph.moe_routed_nanoseconds + graph.moe_shared_nanoseconds +
        graph.output_head_nanoseconds;
    std::cout << "  " << std::left << std::setw(28) << "unaccounted (residual)"
              << std::right << std::setw(9) << std::setprecision(2)
              << (step_milliseconds - milliseconds(accounted, steps))
              << " ms\n";

    const auto& cuda = phase.cuda;
    std::cout << "-- cuda service (aggregate: bytes sum devices, ns max device) --\n"
              << "  matmul calls/step        " << std::setw(9)
              << (steps == 0U ? 0.0
                              : static_cast<double>(cuda.matmul_calls) /
                                    static_cast<double>(steps))
              << "\n"
              << "  flash calls/step         " << std::setw(9)
              << (steps == 0U ? 0.0
                              : static_cast<double>(cuda.flash_attention_calls) /
                                    static_cast<double>(steps))
              << "\n";
    std::cout << std::setprecision(2)
              << "  weight upload            " << std::setw(9)
              << mebibytes(cuda.weight_upload_bytes, steps) << " MiB/step  "
              << std::setw(8) << milliseconds(cuda.upload_wait_nanoseconds, steps)
              << " ms  " << std::setw(6)
              << gigabytes_per_second(cuda.weight_upload_bytes,
                                      cuda.upload_wait_nanoseconds)
              << " GB/s\n"
              << "  weight cudaMalloc        " << std::setw(9)
              << (steps == 0U ? 0.0
                              : static_cast<double>(cuda.weight_allocation_calls) /
                                    static_cast<double>(steps))
              << " calls     " << std::setw(8)
              << milliseconds(cuda.weight_allocation_nanoseconds, steps) << " ms\n"
              << "  weight memcpy (in-call)  " << std::setw(9) << ' '
              << "            " << std::setw(8)
              << milliseconds(cuda.weight_copy_nanoseconds, steps) << " ms  "
              << std::setw(6)
              << gigabytes_per_second(cuda.weight_upload_bytes,
                                      cuda.weight_copy_nanoseconds)
              << " GB/s\n"
              << "  activation h2d           " << std::setw(9)
              << mebibytes(cuda.activation_h2d_bytes, steps) << " MiB/step  "
              << std::setw(8)
              << milliseconds(cuda.activation_h2d_nanoseconds, steps) << " ms  "
              << std::setw(6)
              << gigabytes_per_second(cuda.activation_h2d_bytes,
                                      cuda.activation_h2d_nanoseconds)
              << " GB/s\n"
              << "  activation d2h           " << std::setw(9)
              << mebibytes(cuda.activation_d2h_bytes, steps) << " MiB/step  "
              << std::setw(8)
              << milliseconds(cuda.activation_d2h_nanoseconds, steps) << " ms  "
              << std::setw(6)
              << gigabytes_per_second(cuda.activation_d2h_bytes,
                                      cuda.activation_d2h_nanoseconds)
              << " GB/s\n"
              << "  matmul kernels           " << std::setw(9) << ' '
              << "            " << std::setw(8)
              << milliseconds(cuda.kernel_nanoseconds, steps) << " ms\n"
              << "  flash h2d                " << std::setw(9)
              << mebibytes(cuda.flash_attention_h2d_bytes, steps)
              << " MiB/step  " << std::setw(8)
              << milliseconds(cuda.flash_attention_h2d_nanoseconds, steps)
              << " ms  " << std::setw(6)
              << gigabytes_per_second(cuda.flash_attention_h2d_bytes,
                                      cuda.flash_attention_h2d_nanoseconds)
              << " GB/s\n"
              << "  flash kernels            " << std::setw(9) << ' '
              << "            " << std::setw(8)
              << milliseconds(cuda.flash_attention_kernel_nanoseconds, steps)
              << " ms\n"
              << "  stream synchronize wait  " << std::setw(9) << ' '
              << "            " << std::setw(8)
              << milliseconds(cuda.synchronization_nanoseconds, steps)
              << " ms  (" << cuda.synchronization_calls << " calls)\n";

    for (const auto& device : cuda.devices) {
        std::cout << "  cuda=" << device.device << "  matmul="
                  << device.matmul_calls << "  sync="
                  << std::setprecision(2)
                  << milliseconds(device.synchronization_nanoseconds, steps)
                  << " ms  kernel="
                  << milliseconds(device.kernel_nanoseconds, steps)
                  << " ms  h2d_act="
                  << milliseconds(device.activation_h2d_nanoseconds, steps)
                  << " ms  d2h_act="
                  << milliseconds(device.activation_d2h_nanoseconds, steps)
                  << " ms  weight="
                  << mebibytes(device.weight_upload_bytes, steps) << " MiB ("
                  << "malloc "
                  << milliseconds(device.weight_allocation_nanoseconds, steps)
                  << " ms, memcpy "
                  << milliseconds(device.weight_copy_nanoseconds, steps)
                  << " ms, wait "
                  << milliseconds(device.upload_wait_nanoseconds, steps)
                  << " ms, "
                  << gigabytes_per_second(device.weight_upload_bytes,
                                          device.weight_copy_nanoseconds +
                                              device.upload_wait_nanoseconds)
                  << " GB/s)\n";
    }

    const auto& cache = phase.cache;
    const auto lookups = cache.hits + cache.misses;
    std::cout << "-- expert cache --\n"
              << "  lookups/step " << std::setprecision(1)
              << (steps == 0U ? 0.0
                              : static_cast<double>(lookups) /
                                    static_cast<double>(steps))
              << "  hit rate "
              << (lookups == 0U ? 0.0
                                : 100.0 * static_cast<double>(cache.hits) /
                                      static_cast<double>(lookups))
              << " %  misses/step "
              << (steps == 0U ? 0.0
                              : static_cast<double>(cache.misses) /
                                    static_cast<double>(steps))
              << "  evictions/step "
              << (steps == 0U ? 0.0
                              : static_cast<double>(cache.evictions) /
                                    static_cast<double>(steps))
              << "\n";
    for (std::size_t slot = 0U; slot < cache.device_stage_nanoseconds.size();
         ++slot) {
        std::cout << "  slot " << slot << "  lock wait "
                  << std::setprecision(2)
                  << milliseconds(cache.device_lock_nanoseconds[slot], steps)
                  << " ms  miss staging "
                  << milliseconds(cache.device_stage_nanoseconds[slot], steps)
                  << " ms  cache matmul "
                  << milliseconds(cache.device_matmul_nanoseconds[slot], steps)
                  << " ms  (" << cache.device_misses[slot] << " misses)\n";
    }
    for (std::size_t slot = 0U; slot < cache.capacity_bytes.size(); ++slot) {
        std::cout << "  slot " << slot << "  capacity "
                  << std::setprecision(2)
                  << static_cast<double>(cache.capacity_bytes[slot]) / 1073741824.0
                  << " GiB  pinned spine "
                  << static_cast<double>(cache.pinned_resident_bytes[slot]) /
                         1073741824.0
                  << " GiB  experts "
                  << static_cast<double>(cache.evictable_expert_bytes[slot]) /
                         1073741824.0
                  << " GiB\n";
    }

    const auto& reads = phase.checkpoint_reads;
    std::cout << "-- checkpoint (host mmap / page cache) --\n"
              << "  reads/step " << std::setprecision(1)
              << (steps == 0U ? 0.0
                              : static_cast<double>(reads.calls) /
                                    static_cast<double>(steps))
              << "  bytes/step " << std::setprecision(2)
              << mebibytes(reads.bytes, steps) << " MiB  service "
              << milliseconds(reads.nanoseconds, steps) << " ms  "
              << gigabytes_per_second(reads.bytes, reads.nanoseconds)
              << " GB/s\n";
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        std::cerr << "usage: strata-laguna-profile --model DIR [--devices 0,1,2]"
                     " [--context N] [--max-new N] [--repetitions N]"
                     " [--vram-fraction F] [--prompt TEXT]\n";
        return 2;
    }

    strata::LagunaRuntimeConfig config;
    config.devices = options.devices;
    config.maximum_context_tokens = options.context;
    config.vram_cache_fraction = options.vram_fraction;
    config.sampling_temperature = 0.0;
    config.load_progress = true;
    config.enable_flash_attention = true;
    config.enable_incremental_kv_continuation = false;
    config.detailed_cuda_timing = true;

    strata::LagunaRuntime runtime;
    const auto load_started = std::chrono::steady_clock::now();
    const auto initialized = runtime.initialize(options.model, config);
    if (!initialized.ok()) {
        for (const auto& error : initialized.errors) {
            std::cerr << "error: " << error << '\n';
        }
        return 1;
    }
    std::cout << "load_seconds " << std::fixed << std::setprecision(2)
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - load_started).count()
              << '\n';

    strata::SamplingOptions sampling;
    sampling.temperature = 0.0;
    const std::array messages{
        strata::ChatMessage{strata::ChatRole::User, options.prompt}};

    for (std::uint32_t repetition = 0U; repetition < options.repetitions;
         ++repetition) {
        const auto result = runtime.generate_chat_stream(
            messages, options.max_new, sampling, {}, {});
        if (!result.ok()) {
            for (const auto& error : result.errors) {
                std::cerr << "error: " << error << '\n';
            }
            return 1;
        }
        std::cout << "\n######## repetition " << (repetition + 1U) << '/'
                  << options.repetitions << " ########\n"
                  << "prompt_tokens " << result.metrics.prompt_tokens
                  << "  decode_tokens " << result.metrics.decode_tokens
                  << "  decode_tok_s " << std::setprecision(3)
                  << result.metrics.decode_tokens_per_second()
                  << "  prefill_tok_s "
                  << result.metrics.prefill_tokens_per_second() << '\n';
        report(result.metrics.prefill, result.metrics.prefill_tokens,
               result.metrics.prefill_seconds, "prefill");
        report(result.metrics.decode, result.metrics.decode_tokens,
               result.metrics.decode_seconds, "decode");
        std::cout << "\ntext: " << result.text << '\n';
    }
    return 0;
}
