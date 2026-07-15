#include "strata/deepseek_runtime.hpp"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::string model;
    std::string prompt{"Hello"};
    std::vector<int> devices{0, 1, 2};
    std::uint32_t maximum_new_tokens{16U};
    std::uint32_t maximum_context_tokens{2048U};
    std::uint64_t host_memory_bytes{216ULL << 30U};
    double vram_fraction{0.85};
    bool admission_only{};
    bool json{};
    bool quiet{};
    bool detailed_timing{};
    std::string route_trace;
};

void usage() {
    std::cerr
        << "usage: strata-deepseek-run --model DIR [--prompt TEXT] [--max-new N]\n"
        << "       [--devices 0,1,2] [--max-context N] [--host-memory 216G]\n"
        << "       [--vram-fraction F] [--admission-only] [--route-trace PATH]\n"
        << "       [--detailed-timing] [--json] [--quiet]\n";
}

bool parse_u32(std::string_view text, std::uint32_t& value) {
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

bool parse_bytes(std::string_view text, std::uint64_t& value) {
    if (text.empty()) return false;
    std::uint64_t multiplier = 1U;
    const char suffix = text.back();
    if (suffix == 'K' || suffix == 'k') {
        multiplier = 1ULL << 10U;
        text.remove_suffix(1U);
    } else if (suffix == 'M' || suffix == 'm') {
        multiplier = 1ULL << 20U;
        text.remove_suffix(1U);
    } else if (suffix == 'G' || suffix == 'g') {
        multiplier = 1ULL << 30U;
        text.remove_suffix(1U);
    }
    std::uint64_t base = 0U;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), base);
    if (text.empty() || parsed.ec != std::errc{} ||
        parsed.ptr != text.data() + text.size() ||
        base > std::numeric_limits<std::uint64_t>::max() / multiplier) {
        return false;
    }
    value = base * multiplier;
    return true;
}

bool parse_devices(std::string_view text, std::vector<int>& output) {
    output.clear();
    std::size_t begin = 0U;
    while (begin < text.size()) {
        const auto end = text.find(',', begin);
        const auto item = text.substr(begin, end == std::string_view::npos
            ? text.size() - begin : end - begin);
        int device = -1;
        const auto parsed = std::from_chars(item.data(), item.data() + item.size(), device);
        if (item.empty() || parsed.ec != std::errc{} ||
            parsed.ptr != item.data() + item.size() || device < 0) return false;
        output.push_back(device);
        if (end == std::string_view::npos) break;
        begin = end + 1U;
    }
    return !output.empty();
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](std::string_view name) -> const char* {
            if (index + 1 >= argc) {
                std::cerr << "missing value for " << name << '\n';
                return nullptr;
            }
            return argv[++index];
        };
        if (argument == "--model") {
            const auto* value = next(argument);
            if (value == nullptr) return false;
            options.model = value;
        } else if (argument == "--prompt") {
            const auto* value = next(argument);
            if (value == nullptr) return false;
            options.prompt = value;
        } else if (argument == "--devices") {
            const auto* value = next(argument);
            if (value == nullptr || !parse_devices(value, options.devices)) return false;
        } else if (argument == "--max-new") {
            const auto* value = next(argument);
            if (value == nullptr || !parse_u32(value, options.maximum_new_tokens)) return false;
        } else if (argument == "--max-context") {
            const auto* value = next(argument);
            if (value == nullptr || !parse_u32(value, options.maximum_context_tokens)) return false;
        } else if (argument == "--host-memory") {
            const auto* value = next(argument);
            if (value == nullptr || !parse_bytes(value, options.host_memory_bytes)) return false;
        } else if (argument == "--vram-fraction") {
            const auto* value = next(argument);
            if (value == nullptr) return false;
            char* end = nullptr;
            options.vram_fraction = std::strtod(value, &end);
            if (end == value || *end != '\0') return false;
        } else if (argument == "--route-trace") {
            const auto* value = next(argument);
            if (value == nullptr) return false;
            options.route_trace = value;
        } else if (argument == "--admission-only") {
            options.admission_only = true;
        } else if (argument == "--detailed-timing") {
            options.detailed_timing = true;
        } else if (argument == "--json") {
            options.json = true;
        } else if (argument == "--quiet") {
            options.quiet = true;
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

std::string json_escape(std::string_view input) {
    std::ostringstream output;
    for (const unsigned char value : input) {
        switch (value) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default: output << static_cast<char>(value); break;
        }
    }
    return output.str();
}

template <typename T>
void print_array(std::ostream& output, const std::vector<T>& values) {
    output << '[';
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) output << ',';
        output << values[index];
    }
    output << ']';
}

void print_plan(std::ostream& output, const strata::Dsv4MemoryPlan& plan) {
    output << "{\"required_host_bytes\":" << plan.required_host_bytes
           << ",\"routed_expert_host_bytes\":" << plan.routed_expert_host_bytes
           << ",\"host_parameter_bytes\":" << plan.host_parameter_bytes
           << ",\"kv_state_bytes\":" << plan.kv_state_bytes
           << ",\"host_workspace_bytes\":" << plan.host_workspace_bytes
           << ",\"total_vram_budget_bytes\":" << plan.total_vram_budget_bytes
           << ",\"resident_spine_vram_bytes\":" << plan.resident_spine_vram_bytes
           << ",\"vram_workspace_bytes\":" << plan.vram_workspace_bytes
           << ",\"expert_vram_cache_bytes\":" << plan.expert_vram_cache_bytes
           << ",\"maximum_expert_placement_bytes\":" << plan.maximum_expert_bytes
           << ",\"steady_state_nvme_bytes\":" << plan.steady_state_nvme_bytes
           << ",\"maximum_context_tokens\":" << plan.maximum_context_tokens
           << ",\"zero_nvme_decode\":" << (plan.zero_nvme_decode ? "true" : "false")
           << ",\"dspark_enabled\":" << (plan.dspark_enabled ? "true" : "false")
           << '}';
}

bool device_budgets(const Options& options, std::vector<std::uint64_t>& budgets,
                    std::vector<std::string>& errors) {
    if (!std::isfinite(options.vram_fraction) || options.vram_fraction <= 0.0 ||
        options.vram_fraction > 0.95) {
        errors.emplace_back("VRAM cache fraction must be in (0, 0.95]");
        return false;
    }
    for (const auto device : options.devices) {
        auto memory = strata::CudaBackend::device_memory(device);
        if (!memory.ok()) {
            errors.insert(errors.end(), memory.errors.begin(), memory.errors.end());
            return false;
        }
        budgets.push_back(static_cast<std::uint64_t>(
            static_cast<double>(memory.value.free_bytes) * options.vram_fraction));
    }
    return true;
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
    if (!options.quiet) {
        std::cerr
            << "[contract] exact DeepSeek-V4-Flash base-model greedy decode\n"
            << "[contract] FP4/FP8 checkpoint semantics unchanged; context <=2048\n"
            << "[contract] experts and embedding resident in RAM; decode NVMe bytes = 0\n"
            << "[contract] optional DSpark proposals disabled, never approximated\n";
    }

    if (options.admission_only) {
        auto checkpoint = strata::Dsv4CheckpointReader::open(options.model);
        if (!checkpoint.ok()) {
            for (const auto& error : checkpoint.errors) std::cerr << "error: " << error << '\n';
            return 1;
        }
        std::vector<std::uint64_t> budgets;
        std::vector<std::string> errors;
        if (!device_budgets(options, budgets, errors)) {
            for (const auto& error : errors) std::cerr << "error: " << error << '\n';
            return 1;
        }
        strata::Dsv4AdmissionConfig config;
        config.host_memory_ceiling_bytes = options.host_memory_bytes;
        config.vram_weight_budgets = budgets;
        config.maximum_context_tokens = options.maximum_context_tokens;
        config.require_zero_nvme_decode = true;
        const auto admission = strata::plan_dsv4_resident_topology(
            checkpoint.value->manifest(), config);
        if (!admission.ok()) {
            for (const auto& error : admission.errors) std::cerr << "error: " << error << '\n';
            return 1;
        }
        if (options.json) {
            std::cout << "{\"status\":\"admitted\",\"memory_plan\":";
            print_plan(std::cout, admission.plan);
            std::cout << "}\n";
        } else {
            std::cout << "admitted: required RAM " << admission.plan.required_host_bytes
                      << " bytes; VRAM budget " << admission.plan.total_vram_budget_bytes
                      << " bytes; steady-state NVMe 0 bytes\n";
        }
        return 0;
    }

    strata::Dsv4RuntimeConfig config;
    config.devices = options.devices;
    config.vram_cache_fraction = options.vram_fraction;
    config.host_memory_limit_bytes = options.host_memory_bytes;
    config.maximum_context_tokens = options.maximum_context_tokens;
    config.require_zero_nvme_decode = true;
    config.enable_dspark = false;
    config.detailed_timing = options.detailed_timing;
    config.verbose = !options.quiet;
    config.route_trace_path = options.route_trace;
    strata::DeepSeekV4Runtime runtime;
    const auto initialized = runtime.initialize(options.model, config);
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
                  << "{\"answer\":\"" << json_escape(generated.text)
                  << "\",\"execution\":\"exact_base_autoregressive\""
                  << ",\"dspark\":\"disabled\""
                  << ",\"initialization_seconds\":" << metrics.initialization_seconds
                  << ",\"resident_staging_seconds\":" << metrics.resident_staging_seconds
                  << ",\"prompt_tokens\":" << metrics.prompt_tokens
                  << ",\"generated_tokens\":" << generated.generated_token_ids.size()
                  << ",\"decode_steps\":" << metrics.decode_tokens
                  << ",\"prefill_seconds\":" << metrics.prefill_seconds
                  << ",\"decode_seconds\":" << metrics.decode_seconds
                  << ",\"decode_checkpoint_read_bytes\":"
                  << metrics.decode_checkpoint_reads.bytes
                  << ",\"generation_checkpoint_read_bytes\":"
                  << metrics.generation_checkpoint_reads.bytes
                  << ",\"resident_stage_bytes\":" << metrics.resident_stage.bytes
                  << ",\"vram_cache_hits\":" << metrics.cache.hits
                  << ",\"vram_cache_misses\":" << metrics.cache.misses
                  << ",\"vram_cache_evictions\":" << metrics.cache.evictions
                  << ",\"memory_plan\":";
        print_plan(std::cout, metrics.memory);
        std::cout << ",\"prompt_token_ids\":";
        print_array(std::cout, generated.prompt_token_ids);
        std::cout << ",\"generated_token_ids\":";
        print_array(std::cout, generated.generated_token_ids);
        std::cout << "}\n";
    } else {
        std::cout << generated.text << "\n\n"
                  << "initialization: " << metrics.initialization_seconds << " s\n"
                  << "resident staging: " << metrics.resident_staging_seconds << " s\n"
                  << "prefill: " << metrics.prompt_tokens << " tokens in "
                  << metrics.prefill_seconds << " s\n"
                  << "generation: " << generated.generated_token_ids.size()
                  << " tokens, " << metrics.decode_tokens << " decode steps in "
                  << metrics.decode_seconds << " s\n"
                  << "decode checkpoint reads: "
                  << metrics.decode_checkpoint_reads.bytes << " bytes\n";
    }
    return 0;
}
