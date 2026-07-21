#include "strata/chat_protocol.hpp"
#include "strata/runtime.hpp"

#include "cli_common.hpp"

#include <array>
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
    double vram_fraction{0.85};
    std::uint64_t seed{33'377'335U};
    bool devices_explicit{};
    bool flash_attention{};
    bool jsonl_protocol{};
};

void usage() {
    std::cerr
        << "usage: strata-chat --model DIR --model-type deepseek|glm\n"
        << "                    [--context-size N] [--max-new N]\n"
        << "                    [--temperature F] [--seed N]\n"
        << "                    [--devices 0,1,2] [--vram-fraction F]\n"
        << "                    [--flash-attention]\n"
        << "                    [--protocol jsonl]\n"
        << "                    [--prompt TEXT]\n\n"
        << "Without --prompt, read one question per line until EOF.\n"
        << "The jsonl protocol reads prompt text and an optional messages array.\n";
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--help" || argument == "-h") {
            usage();
            std::exit(0);
        }
        if (argument == "--flash-attention") {
            options.flash_attention = true;
            continue;
        }
        if (index + 1 >= argc) return false;
        const auto next = [&]() { return std::string_view(argv[++index]); };
        if (argument == "--model") options.model = std::string(next());
        else if (argument == "--model-type") options.model_type = std::string(next());
        else if (argument == "--prompt") options.prompt = std::string(next());
        else if (argument == "--context-size" || argument == "--max-context") {
            if (!strata::cli::parse_positive_u32(next(), options.context_size)) return false;
        } else if (argument == "--max-new") {
            if (!strata::cli::parse_positive_u32(next(), options.max_new_tokens)) return false;
        } else if (argument == "--temperature") {
            if (!strata::cli::parse_double(next(), options.temperature, 0.0, 10.0)) return false;
        } else if (argument == "--vram-fraction") {
            if (!strata::cli::parse_double(next(), options.vram_fraction, 0.0, 0.95)) return false;
        } else if (argument == "--seed") {
            if (!strata::cli::parse_u64(next(), options.seed)) return false;
        } else if (argument == "--devices") {
            if (!strata::cli::parse_devices(next(), options.devices)) return false;
            options.devices_explicit = true;
        } else if (argument == "--protocol") {
            if (next() != "jsonl") return false;
            options.jsonl_protocol = true;
        } else {
            std::cerr << "unknown argument: " << argument << '\n';
            return false;
        }
    }
    if (!options.devices_explicit) options.devices = {0, 1, 2};
    return !options.model.empty() &&
           (options.model_type == "glm" || options.model_type == "deepseek");
}

void protocol_event(std::string_view event, std::string_view fields = {}) {
    std::cout << "{\"protocol\":\"strata-chat\",\"version\":"
              << strata::chat_protocol_version << ",\"event\":\""
              << event << '"';
    if (!fields.empty()) std::cout << ',' << fields;
    std::cout << "}\n" << std::flush;
}

void protocol_message(std::string_view event, std::string_view message,
                      bool fatal = false) {
    std::ostringstream fields;
    fields << "\"message\":\"" << strata::cli::json_escape(message) << '"';
    if (fatal) fields << ",\"fatal\":true";
    protocol_event(event, fields.str());
}

class StreamDisplay {
public:
    explicit StreamDisplay(bool protocol)
        : protocol_(protocol), interactive_(!protocol && isatty(STDOUT_FILENO) != 0) {
        if (interactive_) {
            std::cerr << "[decode] live token speed is shown in the terminal title\n";
            set_title("strata-chat | waiting for first token");
        }
        if (!protocol_) std::cout << "assistant> " << std::flush;
    }

    void token(std::uint32_t token_id, std::string_view piece) {
        last_token_id_ = token_id;
        ++emitted_;
        const auto now = std::chrono::steady_clock::now();
        if (!decode_started_) decode_started_ = now;
        pending_utf8_.append(piece.data(), piece.size());
        flush_pending_utf8();
        if (interactive_) {
            if (pending_utf8_.empty()) {
                std::ostringstream title;
                title << "strata-chat | " << emitted_ << " tokens | "
                      << std::fixed << std::setprecision(2)
                      << tokens_per_second(now) << " tok/s";
                set_title(title.str());
            }
        } else if (!protocol_ && emitted_ % 16U == 0U && pending_utf8_.empty()) {
            std::cerr << "[decode] " << emitted_ << " tokens | "
                      << tokens_per_second(now) << " tok/s\n";
        }
    }

    void finish(double runtime_tok_s) {
        flush_pending_utf8();
        if (!pending_utf8_.empty()) {
            const char replacement[] = "\xEF\xBF\xBD";
            write_text(std::string_view(replacement, 3));
            pending_utf8_.clear();
        }
        if (interactive_) set_title("strata-chat | ready");
        if (!protocol_) {
            std::cout << '\n';
            std::cerr << std::fixed << std::setprecision(2)
                      << "[done] " << emitted_ << " tokens | "
                      << runtime_tok_s << " tok/s\n";
        }
    }

    void abort() {
        flush_pending_utf8();
        if (!pending_utf8_.empty()) {
            const char replacement[] = "\xEF\xBF\xBD";
            write_text(std::string_view(replacement, 3));
            pending_utf8_.clear();
        }
        if (interactive_) set_title("strata-chat | error");
        if (!protocol_) std::cout << '\n';
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
                    write_text(std::string_view(pending_utf8_.data(), pos));
                    pending_utf8_.erase(0, pos);
                    pos = 0;
                }
                const char replacement[] = "\xEF\xBF\xBD";
                write_text(std::string_view(replacement, 3));
                pending_utf8_.erase(0, 1);
            }
        }
        if (pos > 0) {
            write_text(std::string_view(pending_utf8_.data(), pos));
            pending_utf8_.erase(0, pos);
        }
    }

    void write_text(std::string_view text) const {
        if (!protocol_) {
            std::cout.write(text.data(), static_cast<std::streamsize>(text.size()));
            std::cout << std::flush;
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        std::ostringstream fields;
        fields << "\"token_id\":" << last_token_id_
               << ",\"tokens\":" << emitted_
               << ",\"tok_s\":" << std::fixed << std::setprecision(4)
               << tokens_per_second(now)
               << ",\"text\":\"" << strata::cli::json_escape(text) << '"';
        protocol_event("token", fields.str());
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

    bool protocol_{};
    bool interactive_{};
    std::uint32_t last_token_id_{};
    std::uint64_t emitted_{};
    std::string pending_utf8_;
    std::optional<std::chrono::steady_clock::time_point> decode_started_;
};

bool answer(strata::RuntimeSession& runtime, const Options& options,
            std::span<const strata::ChatMessage> messages,
            std::string& answer_text) {
    const auto& prompt = messages.back().content;
    std::cerr << "[prefill] processing prompt...\n";
    if (options.jsonl_protocol) {
        std::ostringstream fields;
        fields << "\"prompt_bytes\":" << prompt.size();
        protocol_event("turn_start", fields.str());
    }
    StreamDisplay display(options.jsonl_protocol);
    const strata::TokenStreamCallback stream = [&](std::uint32_t token,
                                                    std::string_view piece) {
        display.token(token, piece);
    };
    const auto result = runtime.generate_chat_stream(
        messages, options.max_new_tokens, stream);
    if (!result.ok()) {
        display.abort();
        for (const auto& error : result.errors) {
            std::cerr << "error: " << error << '\n';
            if (options.jsonl_protocol) protocol_message("error", error);
        }
        return false;
    }
    const double runtime_tok_s = result.metrics.decode_seconds > 0.0
        ? static_cast<double>(result.metrics.decode_tokens) /
              result.metrics.decode_seconds
        : 0.0;
    display.finish(runtime_tok_s);
    if (options.jsonl_protocol) {
        const double prefill_tok_s = result.metrics.prefill_seconds > 0.0
            ? static_cast<double>(result.metrics.prompt_tokens) /
                  result.metrics.prefill_seconds
            : 0.0;
        std::ostringstream fields;
        fields << std::fixed << std::setprecision(6)
               << "\"prompt_tokens\":" << result.metrics.prompt_tokens
               << ",\"decode_tokens\":" << result.metrics.decode_tokens
               << ",\"prefill_seconds\":" << result.metrics.prefill_seconds
               << ",\"prefill_tok_s\":" << prefill_tok_s
               << ",\"decode_seconds\":" << result.metrics.decode_seconds
               << ",\"decode_tok_s\":" << runtime_tok_s;
        protocol_event("turn_done", fields.str());
    } else {
        std::cout << '\n';
    }
    answer_text = result.text;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        usage();
        return 2;
    }

    if (options.jsonl_protocol) {
        std::ostringstream fields;
        fields << "\"model_type\":\"" << options.model_type
               << "\",\"devices\":\""
               << strata::cli::devices_text(options.devices)
               << "\",\"context_size\":" << options.context_size
               << ",\"max_new_tokens\":" << options.max_new_tokens
               << ",\"temperature\":" << options.temperature
               << ",\"exact\":" << (options.temperature == 0.0 ? "true" : "false")
               << ",\"flash_attention\":"
               << (options.flash_attention ? "true" : "false");
        protocol_event("hello", fields.str());
        protocol_message("status", "Loading model");
    }

    strata::RuntimeSession runtime;
    std::cerr << "[startup] model_type=" << options.model_type
              << " devices=" << strata::cli::devices_text(options.devices)
              << " context=" << options.context_size
              << " max_new=" << options.max_new_tokens
              << " temperature=" << options.temperature
              << " vram_fraction=" << options.vram_fraction
              << " seed=" << options.seed << '\n'
              << "[attention] "
              << (options.flash_attention ? "CUDA FlashAttention" : "scalar reference")
              << '\n'
              << "[contract] "
              << (options.temperature == 0.0 ? "exact greedy" : "seeded Gumbel-max sampled")
              << " base-model decode; no hidden fallback\n"
              << "[startup] loading model; this can take several minutes...\n";
    const auto initialization_started = std::chrono::steady_clock::now();
    strata::RuntimeConfig config;
    config.model = options.model_type == "glm"
                       ? strata::RuntimeModel::Glm52
                       : strata::RuntimeModel::DeepSeekV4;
    config.devices = options.devices;
    config.maximum_context_tokens = options.context_size;
    config.vram_cache_fraction = options.vram_fraction;
    config.verbose = options.model_type == "deepseek";
    config.load_progress = options.model_type == "glm";
    config.sampling_temperature = options.temperature;
    config.sampling_seed = options.seed;
    config.enable_flash_attention = options.flash_attention;
    const auto initialized = runtime.initialize(options.model, config);
    if (!initialized.ok()) {
        for (const auto& error : initialized.errors) {
            std::cerr << "error: " << error << '\n';
            if (options.jsonl_protocol) protocol_message("error", error, true);
        }
        return 1;
    }
    const double load_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - initialization_started).count();
    std::cerr << "[ready] model loaded in " << std::fixed << std::setprecision(2)
              << load_seconds
              << " s; enter a prompt\n";
    if (options.jsonl_protocol) {
        std::ostringstream fields;
        fields << std::fixed << std::setprecision(6)
               << "\"load_seconds\":" << load_seconds;
        protocol_event("ready", fields.str());
    }
    if (!options.prompt.empty()) {
        const std::array messages{strata::ChatMessage{
            strata::ChatRole::User, options.prompt}};
        std::string answer_text;
        return answer(runtime, options, messages, answer_text) ? 0 : 1;
    }
    std::string prompt;
    std::vector<strata::ChatMessage> conversation;
    if (options.jsonl_protocol) {
        while (std::getline(std::cin, prompt)) {
            strata::ChatRequest request;
            std::string error;
            if (!strata::parse_chat_request(prompt, request, error)) {
                protocol_message("error", error);
                continue;
            }
            auto messages = request.includes_history
                ? std::move(request.messages)
                : conversation;
            if (!request.includes_history) {
                messages.push_back({strata::ChatRole::User,
                                    std::move(request.prompt)});
            }
            std::string answer_text;
            if (!answer(runtime, options, messages, answer_text)) return 1;
            messages.push_back({strata::ChatRole::Assistant,
                                std::move(answer_text)});
            conversation = std::move(messages);
        }
    } else {
        while (std::cout << "> " && std::getline(std::cin, prompt)) {
            if (prompt.empty()) continue;
            conversation.push_back({strata::ChatRole::User, prompt});
            std::string answer_text;
            if (!answer(runtime, options, conversation, answer_text)) return 1;
            conversation.push_back({strata::ChatRole::Assistant,
                                    std::move(answer_text)});
        }
    }
    return 0;
}
