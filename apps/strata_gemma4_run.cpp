// Minimal single-model driver for Gemma 4, mirroring strata_run.cpp (GLM) and
// strata_deepseek_run.cpp (DeepSeek): strata-chat only reaches Gemma4Runtime
// through the shared strata::RuntimeSession facade, which deliberately does
// not expose architecture-specific diagnostics (see runtime.hpp). This binary
// talks to Gemma4Runtime directly so --layer-hash-trace has somewhere to go.

#include "strata/diagnostics_json.hpp"
#include "strata/gemma4_runtime.hpp"

#include "cli_common.hpp"

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::string model;
    std::string prompt{"The capital of France is"};
    std::vector<int> devices{0, 1, 2};
    std::uint32_t maximum_new_tokens{8U};
    std::uint32_t maximum_context_tokens{256U};
    double vram_fraction{0.85};
    double temperature{};
    std::uint64_t seed{33'377'335U};
    bool json{};
    bool quiet{};
    bool layer_hash_trace{};
};

void usage() {
    std::cerr
        << "usage: strata-gemma4-run --model DIR [--prompt TEXT] [--max-new N]\n"
        << "                        [--devices 0,1,2] [--max-context N]\n"
        << "                        [--vram-fraction F] [--temperature F]\n"
        << "                        [--seed N] [--json] [--quiet]\n"
        << "                        [--layer-hash-trace]\n";
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
        } else if (argument == "--max-new") {
            const auto* next = value(argument);
            if (next == nullptr || !strata::cli::parse_positive_u32(
                                        next, options.maximum_new_tokens)) {
                return false;
            }
        } else if (argument == "--max-context" || argument == "--context-size") {
            const auto* next = value(argument);
            if (next == nullptr || !strata::cli::parse_positive_u32(
                                        next, options.maximum_context_tokens)) {
                return false;
            }
        } else if (argument == "--devices") {
            const auto* next = value(argument);
            if (next == nullptr || !strata::cli::parse_devices(next, options.devices)) {
                return false;
            }
        } else if (argument == "--vram-fraction") {
            const auto* next = value(argument);
            if (next == nullptr ||
                !strata::cli::parse_double(next, options.vram_fraction, 0.0, 1.0)) {
                return false;
            }
        } else if (argument == "--temperature") {
            const auto* next = value(argument);
            if (next == nullptr ||
                !strata::cli::parse_double(next, options.temperature, 0.0, 2.0)) {
                return false;
            }
        } else if (argument == "--seed") {
            const auto* next = value(argument);
            if (next == nullptr) return false;
            options.seed = std::strtoull(next, nullptr, 10);
        } else if (argument == "--json") {
            options.json = true;
        } else if (argument == "--quiet") {
            options.quiet = true;
        } else if (argument == "--layer-hash-trace") {
            options.layer_hash_trace = true;
        } else {
            std::cerr << "unrecognized argument: " << argument << '\n';
            return false;
        }
    }
    return !options.model.empty();
}

void print_diagnostics(std::ostream& output,
                       const strata::DiagnosticTrace& diagnostics) {
    output << "{\"hash_algorithm\":\"fnv1a64-little-endian\""
           << ",\"layer_hidden_hashes\":";
    strata::print_layer_hidden_hashes_object(output, diagnostics);
    if (diagnostics.layer_hash_trace_enabled) {
        strata::print_operation_hashes_fields(output, diagnostics);
    }
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
    strata::Gemma4RuntimeConfig config;
    config.devices = options.devices;
    config.vram_cache_fraction = options.vram_fraction;
    config.maximum_context_tokens = options.maximum_context_tokens;
    config.sampling_temperature = options.temperature;
    config.sampling_seed = options.seed;
    config.verbose = !options.quiet;
    config.enable_layer_hash_trace = options.layer_hash_trace;

    strata::Gemma4Runtime runtime;
    const auto initialized = runtime.initialize(options.model, config);
    if (!initialized.ok()) {
        for (const auto& error : initialized.errors) std::cerr << "error: " << error << '\n';
        return 1;
    }
    const auto generated =
        runtime.generate_stream(options.prompt, options.maximum_new_tokens);
    if (!generated.ok()) {
        for (const auto& error : generated.errors) std::cerr << "error: " << error << '\n';
        return 1;
    }

    if (options.json) {
        std::cout << std::setprecision(10)
                  << "{\"prompt\":\"" << strata::cli::json_escape(options.prompt)
                  << "\",\"answer\":\"" << strata::cli::json_escape(generated.text)
                  << "\",\"prompt_token_ids\":";
        strata::cli::print_array(std::cout, generated.prompt_token_ids);
        std::cout << ",\"generated_token_ids\":";
        strata::cli::print_array(std::cout, generated.generated_token_ids);
        std::cout << ",\"diagnostics\":";
        print_diagnostics(std::cout, generated.diagnostics);
        std::cout << "}\n";
    } else {
        std::cout << generated.text << "\n\n"
                  << "prompt tokens: " << generated.metrics.prompt_tokens << '\n'
                  << "decode tokens: " << generated.metrics.decode_tokens << '\n'
                  << "layer hash trace: "
                  << (generated.diagnostics.layer_hash_trace_enabled
                          ? std::to_string(generated.diagnostics.layer_hashes.size()) +
                                " entries, trace_hash=" +
                                strata::hex_u64(generated.diagnostics.layer_hash_trace_hash)
                          : "disabled")
                  << '\n';
    }
    return 0;
}
