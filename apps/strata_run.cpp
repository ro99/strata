#include "strata/glm_runtime.hpp"

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
};

void usage() {
    std::cerr
        << "usage: strata-run --model DIR [--prompt TEXT] [--max-new N]\n"
        << "                  [--devices 0,1,2] [--max-context N]\n"
        << "                  [--vram-fraction F] [--json] [--quiet]\n";
}

bool parse_u32(std::string_view text, std::uint32_t& value) {
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

bool parse_devices(std::string_view text, std::vector<int>& output) {
    output.clear();
    std::size_t begin = 0U;
    while (begin < text.size()) {
        const auto end = text.find(',', begin);
        const auto part = text.substr(begin, end == std::string_view::npos
                                                ? text.size() - begin
                                                : end - begin);
        int device = -1;
        const auto parsed = std::from_chars(part.data(), part.data() + part.size(), device);
        if (part.empty() || parsed.ec != std::errc{} ||
            parsed.ptr != part.data() + part.size() || device < 0) {
            return false;
        }
        output.push_back(device);
        if (end == std::string_view::npos) break;
        begin = end + 1U;
    }
    return !output.empty();
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
            if (next == nullptr || !parse_devices(next, options.devices)) return false;
        } else if (argument == "--max-new") {
            const auto* next = value(argument);
            if (next == nullptr || !parse_u32(next, options.maximum_new_tokens)) return false;
        } else if (argument == "--max-context") {
            const auto* next = value(argument);
            if (next == nullptr || !parse_u32(next, options.maximum_context_tokens)) return false;
        } else if (argument == "--vram-fraction") {
            const auto* next = value(argument);
            if (next == nullptr) return false;
            char* end = nullptr;
            options.vram_fraction = std::strtod(next, &end);
            if (end == next || *end != '\0') return false;
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

std::string json_escape(std::string_view text) {
    std::ostringstream output;
    for (const unsigned char value : text) {
        switch (value) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (value < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<unsigned int>(value) << std::dec;
                } else {
                    output << static_cast<char>(value);
                }
        }
    }
    return output.str();
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
                  << "{\"prompt\":\"" << json_escape(options.prompt)
                  << "\",\"answer\":\"" << json_escape(generated.text)
                  << "\",\"execution\":\"exact_base_autoregressive\""
                  << ",\"mtp\":\"disabled\""
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
                  << ",\"weight_h2d_bytes\":" << metrics.cuda.weight_upload_bytes
                  << ",\"activation_h2d_bytes\":" << metrics.cuda.activation_h2d_bytes
                  << ",\"activation_d2h_bytes\":" << metrics.cuda.activation_d2h_bytes
                  << ",\"cuda_matmul_calls\":" << metrics.cuda.matmul_calls
                  << ",\"vram_cache_hits\":" << metrics.cache.hits
                  << ",\"vram_cache_misses\":" << metrics.cache.misses
                  << ",\"vram_cache_evictions\":" << metrics.cache.evictions
                  << ",\"vram_cache_used_bytes\":";
        print_array(std::cout, metrics.cache.used_bytes);
        std::cout << ",\"vram_cache_peak_bytes\":";
        print_array(std::cout, metrics.cache.peak_bytes);
        std::cout << ",\"vram_cache_capacity_bytes\":";
        print_array(std::cout, metrics.cache.capacity_bytes);
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
