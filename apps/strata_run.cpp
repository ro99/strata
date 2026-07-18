#include "strata/glm_runtime.hpp"

#include "cli_common.hpp"

#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::string model;
    std::string prompt{"What is the closer start to sun, and how distant it is from it?"};
    std::vector<int> devices{0, 1, 2};
    std::uint32_t maximum_new_tokens{64U};
    std::uint32_t maximum_context_tokens{256U};
    double vram_fraction{0.85};
    bool json{};
    bool quiet{};
    bool diagnostic_trace{};
    bool detailed_timing{};
    bool host_cold_experts{};
    std::uint32_t host_worker_threads{36U};
    std::string route_trace;
};

void usage() {
    std::cerr
        << "usage: strata-run --model DIR [--prompt TEXT] [--max-new N]\n"
        << "                  [--devices 0,1,2] [--max-context N]\n"
        << "                  [--vram-fraction F] [--json] [--quiet]\n"
        << "                  [--diagnostic-trace] [--detailed-timing]\n"
        << "                  [--host-cold-experts] [--host-workers N]\n"
        << "                  [--route-trace PATH]\n";
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto value = [&](std::string_view name) -> const char* {
            if (index + 1 >= argc) {
                std::cerr << "missing value for " << name << '\n';
                return nullptr;
            }
            return argv[++index];
        };
        if (argument == "--model") {
            const auto* next = value(argument);
            if (next == nullptr) return false;
            options.model = next;
        } else if (argument == "--prompt") {
            const auto* next = value(argument);
            if (next == nullptr) return false;
            options.prompt = next;
        } else if (argument == "--devices") {
            const auto* next = value(argument);
            if (next == nullptr || !strata::cli::parse_devices(next, options.devices)) return false;
        } else if (argument == "--max-new") {
            const auto* next = value(argument);
            if (next == nullptr || !strata::cli::parse_u32(next, options.maximum_new_tokens)) return false;
        } else if (argument == "--max-context") {
            const auto* next = value(argument);
            if (next == nullptr || !strata::cli::parse_u32(next, options.maximum_context_tokens)) return false;
        } else if (argument == "--vram-fraction") {
            const auto* next = value(argument);
            if (next == nullptr) return false;
            if (!strata::cli::parse_double(next, options.vram_fraction, 0.0, 0.95)) return false;
        } else if (argument == "--json") {
            options.json = true;
        } else if (argument == "--quiet") {
            options.quiet = true;
        } else if (argument == "--diagnostic-trace") {
            options.diagnostic_trace = true;
        } else if (argument == "--detailed-timing") {
            options.detailed_timing = true;
        } else if (argument == "--host-cold-experts") {
            options.host_cold_experts = true;
        } else if (argument == "--host-workers") {
            const auto* next = value(argument);
            if (next == nullptr || !strata::cli::parse_u32(next, options.host_worker_threads)) return false;
        } else if (argument == "--route-trace") {
            const auto* next = value(argument);
            if (next == nullptr) return false;
            options.route_trace = next;
        } else if (argument == "--help" || argument == "-h") {
            usage();
            std::exit(0);
        } else {
            std::cerr << "unknown argument: " << argument << '\n';
            return false;
        }
    }
    return !options.model.empty();
}

template <typename T>
void print_array(std::ostream& output, const std::vector<T>& values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0U) output << ',';
        output << values[index];
    }
    output << ']';
}

void print_cuda_stats(std::ostream& output, const strata::CudaBackendStats& stats) {
    output << "{\"weight_h2d_bytes\":" << stats.weight_upload_bytes
           << ",\"activation_h2d_bytes\":" << stats.activation_h2d_bytes
           << ",\"activation_d2h_bytes\":" << stats.activation_d2h_bytes
           << ",\"matmul_calls\":" << stats.matmul_calls
           << ",\"weight_allocation_calls\":" << stats.weight_allocation_calls
           << ",\"weight_allocation_bytes\":" << stats.weight_allocation_bytes
           << ",\"workspace_allocation_calls\":" << stats.workspace_allocation_calls
           << ",\"workspace_allocation_bytes\":" << stats.workspace_allocation_bytes
           << ",\"synchronization_calls\":" << stats.synchronization_calls
           << ",\"critical_path_synchronization_seconds\":"
           << static_cast<double>(stats.synchronization_nanoseconds) / 1.0e9
           << ",\"critical_path_upload_wait_seconds\":"
           << static_cast<double>(stats.upload_wait_nanoseconds) / 1.0e9
           << ",\"critical_path_activation_h2d_seconds\":"
           << static_cast<double>(stats.activation_h2d_nanoseconds) / 1.0e9
           << ",\"critical_path_kernel_seconds\":"
           << static_cast<double>(stats.kernel_nanoseconds) / 1.0e9
           << ",\"critical_path_activation_d2h_seconds\":"
           << static_cast<double>(stats.activation_d2h_nanoseconds) / 1.0e9
           << ",\"devices\":[";
    for (std::size_t index = 0; index < stats.devices.size(); ++index) {
        const auto& device = stats.devices[index];
        if (index != 0U) output << ',';
        output << "{\"device\":" << device.device
               << ",\"weight_h2d_bytes\":" << device.weight_upload_bytes
               << ",\"activation_h2d_bytes\":" << device.activation_h2d_bytes
               << ",\"activation_d2h_bytes\":" << device.activation_d2h_bytes
               << ",\"matmul_calls\":" << device.matmul_calls
               << ",\"weight_allocation_calls\":" << device.weight_allocation_calls
               << ",\"weight_allocation_bytes\":" << device.weight_allocation_bytes
               << ",\"workspace_allocation_calls\":" << device.workspace_allocation_calls
               << ",\"workspace_allocation_bytes\":" << device.workspace_allocation_bytes
               << ",\"synchronization_calls\":" << device.synchronization_calls
               << ",\"synchronization_seconds\":"
               << static_cast<double>(device.synchronization_nanoseconds) / 1.0e9
               << ",\"upload_wait_seconds\":"
               << static_cast<double>(device.upload_wait_nanoseconds) / 1.0e9
               << ",\"activation_h2d_seconds\":"
               << static_cast<double>(device.activation_h2d_nanoseconds) / 1.0e9
               << ",\"kernel_seconds\":"
               << static_cast<double>(device.kernel_nanoseconds) / 1.0e9
               << ",\"activation_d2h_seconds\":"
               << static_cast<double>(device.activation_d2h_nanoseconds) / 1.0e9 << '}';
    }
    output << "]}";
}

void print_cache_stats(std::ostream& output, const strata::Glm52CacheStats& stats) {
    output << "{\"hits\":" << stats.hits << ",\"misses\":" << stats.misses
           << ",\"evictions\":" << stats.evictions << ",\"used_bytes\":";
    print_array(output, stats.used_bytes);
    output << ",\"peak_bytes\":";
    print_array(output, stats.peak_bytes);
    output << ",\"capacity_bytes\":";
    print_array(output, stats.capacity_bytes);
    output << ",\"pinned_resident_spine_bytes\":";
    print_array(output, stats.pinned_resident_bytes);
    output << ",\"evictable_expert_bytes\":";
    print_array(output, stats.evictable_expert_bytes);
    output << ",\"device_hits\":";
    print_array(output, stats.device_hits);
    output << ",\"device_misses\":";
    print_array(output, stats.device_misses);
    output << ",\"device_evictions\":";
    print_array(output, stats.device_evictions);
    output << '}';
}

void print_phase(std::ostream& output, const strata::Glm52PhaseMetrics& phase) {
    output << "{\"checkpoint_read_calls\":" << phase.checkpoint_reads.calls
           << ",\"checkpoint_read_bytes\":" << phase.checkpoint_reads.bytes
           << ",\"checkpoint_read_seconds\":"
           << static_cast<double>(phase.checkpoint_reads.nanoseconds) / 1.0e9
           << ",\"checkpoint_read_wall_seconds\":"
           << static_cast<double>(phase.checkpoint_reads.wall_nanoseconds) / 1.0e9
           << ",\"host_aggregation_seconds\":"
           << static_cast<double>(phase.host_aggregation_nanoseconds) / 1.0e9
           << ",\"host_experts\":" << phase.host_experts.experts
           << ",\"host_expert_matvec_calls\":" << phase.host_experts.matvec_calls
           << ",\"host_expert_weight_bytes\":" << phase.host_experts.weight_bytes
           << ",\"host_expert_service_seconds\":"
           << static_cast<double>(phase.host_experts.service_nanoseconds) / 1.0e9
           << ",\"host_mapping_sweeps\":" << phase.host_experts.mapping_sweeps
           << ",\"host_mapping_sweep_seconds\":"
           << static_cast<double>(phase.host_experts.mapping_sweep_nanoseconds) / 1.0e9
           << ",\"cuda\":";
    print_cuda_stats(output, phase.cuda);
    output << ",\"cache\":";
    print_cache_stats(output, phase.cache);
    output << '}';
}

}  // namespace

int main(int argc, char** argv) {
    if (std::getenv("CUDA_DEVICE_ORDER") == nullptr) {
        static_cast<void>(setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 0));
    }
    Options options;
    if (!parse_options(argc, argv, options)) {
        usage();
        return 2;
    }
    strata::Glm52RuntimeConfig config;
    config.devices = options.devices;
    config.vram_cache_fraction = options.vram_fraction;
    config.maximum_context_tokens = options.maximum_context_tokens;
    config.verbose = !options.quiet;
    config.diagnostic_trace = options.diagnostic_trace;
    config.detailed_timing = options.detailed_timing;
    config.host_cold_experts = options.host_cold_experts;
    config.host_worker_threads = options.host_worker_threads;
    config.route_trace_path = options.route_trace;

    if (!options.quiet) {
        std::cerr << "[contract] exact main-model greedy decode; INT4/INT8 checkpoint unchanged\n"
                  << "[contract] DSA full-attention region enforced at <=2048 tokens\n"
                  << "[contract] MTP proposal acceleration disabled for this baseline\n";
    }
    strata::Glm52Runtime runtime;
    const auto initialization_started = std::chrono::steady_clock::now();
    const auto initialized = runtime.initialize(options.model, config);
    const auto initialization_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - initialization_started).count();
    if (!initialized.ok()) {
        for (const auto& error : initialized.errors) std::cerr << "error: " << error << '\n';
        return 1;
    }
    const auto generated = runtime.generate(options.prompt, options.maximum_new_tokens);
    if (!generated.ok()) {
        for (const auto& error : generated.errors) std::cerr << "error: " << error << '\n';
        return 1;
    }

    const auto& metrics = generated.metrics;
    if (options.json) {
        std::cout << std::setprecision(10)
                  << "{\"prompt\":\"" << strata::cli::json_escape(options.prompt)
                  << "\",\"answer\":\"" << strata::cli::json_escape(generated.text)
                  << "\",\"execution\":\"exact_base_autoregressive\""
                  << ",\"mtp\":\"disabled\""
                  << ",\"detailed_timing\":" << (metrics.detailed_timing ? "true" : "false")
                  << ",\"route_trace\":\""
                  << strata::cli::json_escape(options.route_trace) << '"'
                  << ",\"initialization_seconds\":" << initialization_seconds
                  << ",\"prompt_tokens\":" << metrics.prompt_tokens
                  << ",\"decode_tokens\":" << metrics.decode_tokens
                  << ",\"prompt_processing_seconds\":" << metrics.prefill_seconds
                  << ",\"prompt_processing_tok_s\":"
                  << metrics.prefill_tokens_per_second()
                  << ",\"generation_seconds\":" << metrics.decode_seconds
                  << ",\"generation_tok_s\":" << metrics.decode_tokens_per_second()
                  << ",\"checkpoint_read_calls\":" << metrics.checkpoint_reads.calls
                  << ",\"checkpoint_read_bytes\":" << metrics.checkpoint_reads.bytes
                  << ",\"checkpoint_read_seconds\":"
                  << static_cast<double>(metrics.checkpoint_reads.nanoseconds) / 1.0e9
                  << ",\"checkpoint_read_wall_seconds\":"
                  << static_cast<double>(metrics.checkpoint_reads.wall_nanoseconds) / 1.0e9
                  << ",\"weight_h2d_bytes\":" << metrics.cuda.weight_upload_bytes
                  << ",\"activation_h2d_bytes\":" << metrics.cuda.activation_h2d_bytes
                  << ",\"activation_d2h_bytes\":" << metrics.cuda.activation_d2h_bytes
                  << ",\"cuda_matmul_calls\":" << metrics.cuda.matmul_calls
                  << ",\"cuda_allocation_calls\":"
                  << metrics.cuda.weight_allocation_calls +
                         metrics.cuda.workspace_allocation_calls
                  << ",\"cuda_synchronization_calls\":"
                  << metrics.cuda.synchronization_calls
                  << ",\"vram_cache_hits\":" << metrics.cache.hits
                  << ",\"vram_cache_misses\":" << metrics.cache.misses
                  << ",\"vram_cache_evictions\":" << metrics.cache.evictions
                  << ",\"vram_cache_used_bytes\":";
        print_array(std::cout, metrics.cache.used_bytes);
        std::cout << ",\"vram_cache_peak_bytes\":";
        print_array(std::cout, metrics.cache.peak_bytes);
        std::cout << ",\"vram_cache_capacity_bytes\":";
        print_array(std::cout, metrics.cache.capacity_bytes);
        std::cout << ",\"pinned_resident_spine_bytes\":";
        print_array(std::cout, metrics.cache.pinned_resident_bytes);
        std::cout << ",\"evictable_expert_bytes\":";
        print_array(std::cout, metrics.cache.evictable_expert_bytes);
        std::cout << ",\"phases\":{\"prefill\":";
        print_phase(std::cout, metrics.prefill);
        std::cout << ",\"decode\":";
        print_phase(std::cout, metrics.decode);
        std::cout << '}';
        std::cout << ",\"prompt_token_ids\":";
        print_array(std::cout, generated.prompt_token_ids);
        std::cout << ",\"generated_token_ids\":";
        print_array(std::cout, generated.generated_token_ids);
        std::cout << "}\n";
    } else {
        std::cout << generated.text << "\n\n"
                  << "initialization/load: " << initialization_seconds << " s\n"
                  << "prompt processing: " << metrics.prompt_tokens << " tokens in "
                  << metrics.prefill_seconds << " s ("
                  << metrics.prefill_tokens_per_second() << " tok/s)\n"
                  << "generation: " << metrics.decode_tokens << " decode steps in "
                  << metrics.decode_seconds << " s ("
                  << metrics.decode_tokens_per_second() << " tok/s)\n"
                  << "checkpoint reads: " << metrics.checkpoint_reads.bytes << " bytes in "
                  << static_cast<double>(metrics.checkpoint_reads.nanoseconds) / 1.0e9
                  << " s\n"
                  << "VRAM cache: " << metrics.cache.hits << " hits, "
                  << metrics.cache.misses << " misses, " << metrics.cache.evictions
                  << " evictions\n";
    }
    return 0;
}
