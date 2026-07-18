#include "strata/deepseek_runtime.hpp"
#include "strata/glm_runtime.hpp"

#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <unistd.h>

namespace {

struct Options {
    std::string model;
    std::string model_type;
    std::string prompt;
    std::vector<int> devices;
    std::uint32_t context_size{2048U};
    std::uint32_t max_new_tokens{256U};
    double temperature{};
    std::uint64_t seed{33'377'335U};
    bool devices_explicit{};
};

void usage() {
    std::cerr
        << "usage: strata-chat --model DIR --model-type deepseek|glm\n"
        << "                    [--context-size N] [--max-new N]\n"
        << "                    [--temperature F] [--seed N]\n"
        << "                    [--devices 0,1,2] [--prompt TEXT]\n\n"
        << "Without --prompt, read one question per line until EOF.\n";
}

bool parse_u32(std::string_view text, std::uint32_t& value) {
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && value > 0U;
}

bool parse_u64(std::string_view text, std::uint64_t& value) {
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

bool parse_double(std::string_view text, double& value) {
    const std::string owned(text);
    char* end = nullptr;
    value = std::strtod(owned.c_str(), &end);
    return end != owned.c_str() && *end == '\0' && value >= 0.0 && value <= 10.0;
}

bool parse_devices(std::string_view text, std::vector<int>& devices) {
    devices.clear();
    std::size_t begin = 0U;
    while (begin < text.size()) {
        const auto end = text.find(',', begin);
        const auto part = text.substr(begin, end == std::string_view::npos
                                                ? text.size() - begin : end - begin);
        int device = -1;
        const auto parsed = std::from_chars(part.data(), part.data() + part.size(), device);
        if (part.empty() || parsed.ec != std::errc{} ||
            parsed.ptr != part.data() + part.size() || device < 0) return false;
        devices.push_back(device);
        if (end == std::string_view::npos) break;
        begin = end + 1U;
    }
    return !devices.empty();
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--help" || argument == "-h") {
            usage();
            std::exit(0);
        }
        if (index + 1 >= argc) return false;
        const auto next = [&]() { return std::string_view(argv[++index]); };
        if (argument == "--model") options.model = std::string(next());
        else if (argument == "--model-type") options.model_type = std::string(next());
        else if (argument == "--prompt") options.prompt = std::string(next());
        else if (argument == "--context-size" || argument == "--max-context") {
            if (!parse_u32(next(), options.context_size)) return false;
        } else if (argument == "--max-new") {
            if (!parse_u32(next(), options.max_new_tokens)) return false;
        } else if (argument == "--temperature") {
            if (!parse_double(next(), options.temperature)) return false;
        } else if (argument == "--seed") {
            if (!parse_u64(next(), options.seed)) return false;
        } else if (argument == "--devices") {
            if (!parse_devices(next(), options.devices)) return false;
            options.devices_explicit = true;
        } else {
            std::cerr << "unknown argument: " << argument << '\n';
            return false;
        }
    }
    if (!options.devices_explicit) options.devices = {0, 1, 2};
    return !options.model.empty() &&
           (options.model_type == "glm" || options.model_type == "deepseek");
}

std::string devices_text(const std::vector<int>& devices) {
    std::ostringstream output;
    for (std::size_t index = 0U; index < devices.size(); ++index) {
        if (index != 0U) output << ',';
        output << devices[index];
    }
    return output.str();
}

class StreamDisplay {
public:
    StreamDisplay() : interactive_(isatty(STDOUT_FILENO) != 0) {
        if (interactive_) {
            std::cerr << "[decode] live token speed is shown in the terminal title\n";
            set_title("strata-chat | waiting for first token");
        }
        std::cout << "assistant> " << std::flush;
    }

    void token(std::string_view piece) {
        pending_utf8_.append(piece.data(), piece.size());
        flush_pending_utf8();
        ++emitted_;
        const auto now = std::chrono::steady_clock::now();
        if (!decode_started_) decode_started_ = now;
        if (interactive_) {
            if (pending_utf8_.empty()) {
                std::ostringstream title;
                title << "strata-chat | " << emitted_ << " tokens | "
                      << std::fixed << std::setprecision(2)
                      << tokens_per_second(now) << " tok/s";
                set_title(title.str());
            }
        } else if (emitted_ % 16U == 0U && pending_utf8_.empty()) {
            std::cerr << "[decode] " << emitted_ << " tokens | "
                      << tokens_per_second(now) << " tok/s\n";
        }
    }

    void finish(double runtime_tok_s) {
        flush_pending_utf8();
        if (!pending_utf8_.empty()) {
            const char replacement[] = "\xEF\xBF\xBD";
            std::cout.write(replacement, 3);
            pending_utf8_.clear();
            std::cout << std::flush;
        }
        if (interactive_) set_title("strata-chat | ready");
        std::cout << '\n';
        std::cerr << std::fixed << std::setprecision(2)
                  << "[done] " << emitted_ << " tokens | "
                  << runtime_tok_s << " tok/s\n";
    }

    void abort() {
        flush_pending_utf8();
        if (!pending_utf8_.empty()) {
            const char replacement[] = "\xEF\xBF\xBD";
            std::cout.write(replacement, 3);
            pending_utf8_.clear();
            std::cout << std::flush;
        }
        if (interactive_) set_title("strata-chat | error");
        std::cout << '\n';
    }

private:
    // Returns >0 for a complete valid UTF-8 sequence, 0 if incomplete, -1 if invalid.
    int consume_utf8_at(std::size_t pos) const {
        const unsigned char lead = static_cast<unsigned char>(pending_utf8_[pos]);
        const std::size_t remain = pending_utf8_.size() - pos;

        if (lead <= 0x7FU) return 1;

        if (lead >= 0xC2U && lead <= 0xDFU) {
            if (remain < 2) return 0;
            if ((static_cast<unsigned char>(pending_utf8_[pos + 1U]) & 0xC0U) != 0x80U) return -1;
            return 2;
        }

        if (lead == 0xE0U) {
            if (remain < 3) return 0;
            const auto b2 = static_cast<unsigned char>(pending_utf8_[pos + 1U]);
            if ((b2 & 0xC0U) != 0x80U || b2 < 0xA0U) return -1;
            if ((static_cast<unsigned char>(pending_utf8_[pos + 2U]) & 0xC0U) != 0x80U) return -1;
            return 3;
        }

        if (lead >= 0xE1U && lead <= 0xECU) {
            if (remain < 3) return 0;
            if ((static_cast<unsigned char>(pending_utf8_[pos + 1U]) & 0xC0U) != 0x80U) return -1;
            if ((static_cast<unsigned char>(pending_utf8_[pos + 2U]) & 0xC0U) != 0x80U) return -1;
            return 3;
        }

        if (lead == 0xEDU) {
            if (remain < 3) return 0;
            const auto b2 = static_cast<unsigned char>(pending_utf8_[pos + 1U]);
            if ((b2 & 0xC0U) != 0x80U || b2 > 0x9FU) return -1;
            if ((static_cast<unsigned char>(pending_utf8_[pos + 2U]) & 0xC0U) != 0x80U) return -1;
            return 3;
        }

        if (lead >= 0xEEU && lead <= 0xEFU) {
            if (remain < 3) return 0;
            if ((static_cast<unsigned char>(pending_utf8_[pos + 1U]) & 0xC0U) != 0x80U) return -1;
            if ((static_cast<unsigned char>(pending_utf8_[pos + 2U]) & 0xC0U) != 0x80U) return -1;
            return 3;
        }

        if (lead == 0xF0U) {
            if (remain < 4) return 0;
            const auto b2 = static_cast<unsigned char>(pending_utf8_[pos + 1U]);
            if ((b2 & 0xC0U) != 0x80U || b2 < 0x90U) return -1;
            if ((static_cast<unsigned char>(pending_utf8_[pos + 2U]) & 0xC0U) != 0x80U) return -1;
            if ((static_cast<unsigned char>(pending_utf8_[pos + 3U]) & 0xC0U) != 0x80U) return -1;
            return 4;
        }

        if (lead >= 0xF1U && lead <= 0xF3U) {
            if (remain < 4) return 0;
            if ((static_cast<unsigned char>(pending_utf8_[pos + 1U]) & 0xC0U) != 0x80U) return -1;
            if ((static_cast<unsigned char>(pending_utf8_[pos + 2U]) & 0xC0U) != 0x80U) return -1;
            if ((static_cast<unsigned char>(pending_utf8_[pos + 3U]) & 0xC0U) != 0x80U) return -1;
            return 4;
        }

        if (lead == 0xF4U) {
            if (remain < 4) return 0;
            const auto b2 = static_cast<unsigned char>(pending_utf8_[pos + 1U]);
            if ((b2 & 0xC0U) != 0x80U || b2 > 0x8FU) return -1;
            if ((static_cast<unsigned char>(pending_utf8_[pos + 2U]) & 0xC0U) != 0x80U) return -1;
            if ((static_cast<unsigned char>(pending_utf8_[pos + 3U]) & 0xC0U) != 0x80U) return -1;
            return 4;
        }

        return -1;
    }

    void flush_pending_utf8() {
        if (pending_utf8_.empty()) return;
        std::size_t pos = 0;
        while (pos < pending_utf8_.size()) {
            const int result = consume_utf8_at(pos);
            if (result > 0) {
                pos += static_cast<std::size_t>(result);
            } else if (result == 0) {
                break;
            } else {
                if (pos > 0) {
                    std::cout.write(pending_utf8_.data(), pos);
                    pending_utf8_.erase(0, pos);
                    pos = 0;
                }
                const char replacement[] = "\xEF\xBF\xBD";
                std::cout.write(replacement, 3);
                pending_utf8_.erase(0, 1);
                std::cout << std::flush;
            }
        }
        if (pos > 0) {
            std::cout.write(pending_utf8_.data(), pos);
            pending_utf8_.erase(0, pos);
            std::cout << std::flush;
        }
    }

    double tokens_per_second(std::chrono::steady_clock::time_point now) const {
        if (!decode_started_ || emitted_ < 2U) return 0.0;
        const double seconds = std::chrono::duration<double>(
            now - *decode_started_).count();
        return seconds > 0.0 ? static_cast<double>(emitted_ - 1U) / seconds : 0.0;
    }

    void set_title(std::string_view title) const {
        std::cerr << "\033]0;" << title << '\a' << std::flush;
    }

    bool interactive_{};
    std::uint64_t emitted_{};
    std::string pending_utf8_;
    std::optional<std::chrono::steady_clock::time_point> decode_started_;
};

template <typename Runtime, typename Result>
bool answer(Runtime& runtime, const Options& options, std::string_view prompt) {
    std::cerr << "[prefill] processing prompt...\n";
    StreamDisplay display;
    const strata::TokenStreamCallback stream = [&](std::uint32_t, std::string_view piece) {
        display.token(piece);
    };
    const Result result = runtime.generate_stream(prompt, options.max_new_tokens, stream);
    if (!result.ok()) {
        display.abort();
        for (const auto& error : result.errors) std::cerr << "error: " << error << '\n';
        return false;
    }
    const double runtime_tok_s = result.metrics.decode_seconds > 0.0
        ? static_cast<double>(result.metrics.decode_tokens) /
              result.metrics.decode_seconds
        : 0.0;
    display.finish(runtime_tok_s);
    std::cout << '\n';
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        usage();
        return 2;
    }

    strata::Glm52Runtime glm;
    strata::DeepSeekV4Runtime deepseek;
    std::cerr << "[startup] model_type=" << options.model_type
              << " devices=" << devices_text(options.devices)
              << " context=" << options.context_size
              << " max_new=" << options.max_new_tokens
              << " temperature=" << options.temperature
              << " seed=" << options.seed << '\n'
              << "[contract] "
              << (options.temperature == 0.0 ? "exact greedy" : "seeded Gumbel-max sampled")
              << " base-model decode; no hidden fallback\n"
              << "[startup] loading model; this can take several minutes...\n";
    const auto initialization_started = std::chrono::steady_clock::now();
    if (options.model_type == "glm") {
        strata::Glm52RuntimeConfig config;
        config.devices = options.devices;
        config.maximum_context_tokens = options.context_size;
        config.verbose = false;
        config.load_progress = true;
        config.sampling_temperature = options.temperature;
        config.sampling_seed = options.seed;
        const auto initialized = glm.initialize(options.model, config);
        if (!initialized.ok()) {
            for (const auto& error : initialized.errors) std::cerr << "error: " << error << '\n';
            return 1;
        }
        std::cerr << "[ready] model loaded in " << std::fixed << std::setprecision(2)
                  << std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - initialization_started).count()
                  << " s; enter a prompt\n";
        if (!options.prompt.empty()) return answer<strata::Glm52Runtime, strata::Glm52GenerationResult>(glm, options, options.prompt) ? 0 : 1;
        std::string prompt;
        while (std::cout << "> " && std::getline(std::cin, prompt)) {
            if (prompt.empty()) continue;
            if (!answer<strata::Glm52Runtime, strata::Glm52GenerationResult>(glm, options, prompt)) return 1;
        }
    } else {
        strata::Dsv4RuntimeConfig config;
        config.devices = options.devices;
        config.maximum_context_tokens = options.context_size;
        config.verbose = true;
        config.sampling_temperature = options.temperature;
        config.sampling_seed = options.seed;
        const auto initialized = deepseek.initialize(options.model, config);
        if (!initialized.ok()) {
            for (const auto& error : initialized.errors) std::cerr << "error: " << error << '\n';
            return 1;
        }
        std::cerr << "[ready] model loaded in " << std::fixed << std::setprecision(2)
                  << std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - initialization_started).count()
                  << " s; enter a prompt\n";
        if (!options.prompt.empty()) return answer<strata::DeepSeekV4Runtime, strata::Dsv4GenerationResult>(deepseek, options, options.prompt) ? 0 : 1;
        std::string prompt;
        while (std::cout << "> " && std::getline(std::cin, prompt)) {
            if (prompt.empty()) continue;
            if (!answer<strata::DeepSeekV4Runtime, strata::Dsv4GenerationResult>(deepseek, options, prompt)) return 1;
        }
    }
    return 0;
}
