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
    strata::SamplingOptions sampling{strata::greedy_sampling()};
    double vram_fraction{0.85};
    bool devices_explicit{};
    bool flash_attention{};
    bool incremental_kv_continuation{true};
    bool block_kv_cache{};
    bool pin_resident_arena{};
    bool prepack_mhc{true};
    bool jsonl_protocol{};
    std::string plan_cache;
    bool dry_run{};
    bool use_plan_cache{true};
    bool replan{};
};

// Bundles that make the sampler usable without knowing what nine knobs do.
// A preset only writes defaults; any explicit flag after it wins.
bool apply_sampler_preset(std::string_view name, strata::SamplingOptions& sampling) {
    if (name == "precise") {
        sampling.temperature = 0.0;
        return true;
    }
    if (name == "balanced") {
        sampling.temperature = 1.0;
        sampling.min_p = 0.05;
        sampling.repetition_penalty = 1.05;
        sampling.penalty_window = 256U;
        return true;
    }
    if (name == "creative") {
        // Truncate on min_p rather than top_p so the tail is cut by relative
        // plausibility, then use XTC to drop the safe continuation and DRY to
        // break the loops that removing it tends to invite.
        sampling.temperature = 1.0;
        sampling.min_p = 0.02;
        sampling.xtc_probability = 0.5;
        sampling.xtc_threshold = 0.1;
        sampling.dry_multiplier = 0.8;
        sampling.dry_base = 1.75;
        sampling.dry_allowed_length = 2U;
        sampling.repetition_penalty = 1.03;
        sampling.penalty_window = 512U;
        return true;
    }
    if (name == "future-entropy") {
        // Costs 21 forward passes per token, not one. min_p is not optional
        // here: broken word-fragments have maximally uncertain futures, so
        // without a relative-plausibility cut the entropy term selects them.
        // DRY is what the reference implementation reports missing — an
        // entropy-scored loop, once entered, is locked in by min_p.
        sampling.temperature = 1.0;
        sampling.min_p = 0.05;
        sampling.future_entropy_candidates = 20U;
        sampling.future_entropy_top_n = 30U;
        sampling.future_entropy_alpha = 0.0;
        sampling.dry_multiplier = 0.8;
        sampling.dry_base = 1.75;
        sampling.dry_allowed_length = 2U;
        sampling.repetition_penalty = 1.03;
        sampling.penalty_window = 512U;
        return true;
    }
    return false;
}

void usage() {
    std::cerr
        << "usage: strata-chat --model DIR --model-type gemma4|deepseek|glm|kimi-k3\n"
        << "                    [--context-size N] [--max-new N]\n"
        << "                    [--devices 0,1,2] [--vram-fraction F]\n"
        << "                    [--flash-attention] [--full-reprefill]\n"
        << "                    [--block-kv-cache] [--pin-resident-arena]\n"
        << "                    [--no-prepack-mhc]\n"
        << "                    [--protocol jsonl]\n"
        << "                    [--prompt TEXT]\n\n"
        << "placement:          [--dry-run] [--replan]\n"
        << "                    [--plan-cache DIR] [--no-plan-cache]\n\n"
        << "sampler:            [--preset precise|balanced|creative|future-entropy]\n"
        << "                    [--temperature F] [--seed N]\n"
        << "                    [--top-k N] [--top-p F] [--min-p F] [--typical-p F]\n"
        << "                    [--xtc-probability F] [--xtc-threshold F]\n"
        << "                    [--presence-penalty F] [--frequency-penalty F]\n"
        << "                    [--repetition-penalty F] [--penalty-window N]\n"
        << "                    [--dry-multiplier F] [--dry-base F]\n"
        << "                    [--dry-allowed-length N] [--dry-window N]\n"
        << "                    [--no-repeat-ngram N]\n\n"
        << "future entropy:     [--future-entropy N] [--future-entropy-top-n N]\n"
        << "                    [--alpha F] [--future-entropy-curve article|crossfade]\n"
        << "                    [--alpha-wave-amplitude F] [--alpha-wave-period F]\n\n"
        << "Truncation reads the model's own distribution, so --min-p and\n"
        << "--top-p mean the same thing at any temperature. --preset writes\n"
        << "defaults; flags given after it override them.\n\n"
        << "--temperature alone, with no --preset, is plain temperature\n"
        << "sampling and nothing else: no truncation, no penalties, no XTC.\n\n"
        << "--future-entropy N scores each of the N likeliest candidates by how\n"
        << "open the distribution one step past it is, and costs one extra\n"
        << "forward pass per candidate: a token takes N+1 decode steps, not\n"
        << "one. --alpha crossfades from -1 (ordinary sampling, the stage is\n"
        << "a no-op) to +1 (future entropy alone). Pair it with --min-p 0.05\n"
        << "or higher; entropy favours broken word-fragments otherwise.\n\n"
        << "Without --prompt, read one question per line until EOF.\n"
        << "The jsonl protocol reads prompt text and an optional messages array.\n\n"
        << "--dry-run sizes every component against this machine, prints where\n"
        << "each one would go, writes the plan to the cache, and exits without\n"
        << "reading a single weight. The next real load reuses that plan, so it\n"
        << "places exactly what the dry run printed. --replan recomputes and\n"
        << "overwrites a cached plan; --no-plan-cache neither reads nor writes\n"
        << "one. A plan is keyed by checkpoint, GPUs, context size, and device\n"
        << "list: change any of them and it is recomputed automatically.\n";
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
        if (argument == "--full-reprefill") {
            options.incremental_kv_continuation = false;
            continue;
        }
        if (argument == "--block-kv-cache") {
            options.block_kv_cache = true;
            continue;
        }
        if (argument == "--pin-resident-arena") {
            options.pin_resident_arena = true;
            continue;
        }
        if (argument == "--no-prepack-mhc") {
            options.prepack_mhc = false;
            continue;
        }
        if (argument == "--dry-run") {
            options.dry_run = true;
            continue;
        }
        if (argument == "--no-plan-cache") {
            options.use_plan_cache = false;
            continue;
        }
        if (argument == "--replan") {
            options.replan = true;
            continue;
        }
        if (index + 1 >= argc) return false;
        const auto next = [&]() { return std::string_view(argv[++index]); };
        if (argument == "--model") options.model = std::string(next());
        else if (argument == "--model-type") options.model_type = std::string(next());
        else if (argument == "--prompt") options.prompt = std::string(next());
        else if (argument == "--plan-cache") options.plan_cache = std::string(next());
        else if (argument == "--context-size" || argument == "--max-context") {
            if (!strata::cli::parse_positive_u32(next(), options.context_size)) return false;
        } else if (argument == "--max-new") {
            if (!strata::cli::parse_positive_u32(next(), options.max_new_tokens)) return false;
        } else if (argument == "--temperature") {
            if (!strata::cli::parse_double(next(), options.sampling.temperature, 0.0, 10.0)) return false;
        } else if (argument == "--vram-fraction") {
            if (!strata::cli::parse_double(next(), options.vram_fraction, 0.0, 0.95)) return false;
        } else if (argument == "--seed") {
            if (!strata::cli::parse_u64(next(), options.sampling.seed)) return false;
        } else if (argument == "--preset") {
            if (!apply_sampler_preset(next(), options.sampling)) return false;
        } else if (argument == "--top-k") {
            if (!strata::cli::parse_u32(next(), options.sampling.top_k)) return false;
        } else if (argument == "--top-p") {
            if (!strata::cli::parse_double(next(), options.sampling.top_p, 0.0, 1.0)) return false;
        } else if (argument == "--min-p") {
            if (!strata::cli::parse_double(next(), options.sampling.min_p, 0.0, 1.0)) return false;
        } else if (argument == "--typical-p") {
            if (!strata::cli::parse_double(next(), options.sampling.typical_p, 0.0, 1.0)) return false;
        } else if (argument == "--xtc-probability") {
            if (!strata::cli::parse_double(next(), options.sampling.xtc_probability, 0.0, 1.0)) return false;
        } else if (argument == "--xtc-threshold") {
            if (!strata::cli::parse_double(next(), options.sampling.xtc_threshold, 0.0, 1.0)) return false;
        } else if (argument == "--presence-penalty") {
            if (!strata::cli::parse_double(next(), options.sampling.presence_penalty, -2.0, 2.0)) return false;
        } else if (argument == "--frequency-penalty") {
            if (!strata::cli::parse_double(next(), options.sampling.frequency_penalty, -2.0, 2.0)) return false;
        } else if (argument == "--repetition-penalty") {
            if (!strata::cli::parse_double(next(), options.sampling.repetition_penalty, 0.0, 10.0)) return false;
        } else if (argument == "--penalty-window") {
            if (!strata::cli::parse_u32(next(), options.sampling.penalty_window)) return false;
        } else if (argument == "--dry-multiplier") {
            if (!strata::cli::parse_double(next(), options.sampling.dry_multiplier, 0.0, 10.0)) return false;
        } else if (argument == "--dry-base") {
            if (!strata::cli::parse_double(next(), options.sampling.dry_base, 1.0, 8.0)) return false;
        } else if (argument == "--dry-allowed-length") {
            if (!strata::cli::parse_positive_u32(next(), options.sampling.dry_allowed_length)) return false;
        } else if (argument == "--dry-window") {
            if (!strata::cli::parse_u32(next(), options.sampling.dry_window)) return false;
        } else if (argument == "--no-repeat-ngram") {
            if (!strata::cli::parse_u32(next(), options.sampling.no_repeat_ngram)) return false;
        } else if (argument == "--future-entropy") {
            if (!strata::cli::parse_u32(next(), options.sampling.future_entropy_candidates)) return false;
        } else if (argument == "--future-entropy-top-n") {
            if (!strata::cli::parse_u32(next(), options.sampling.future_entropy_top_n)) return false;
        } else if (argument == "--alpha") {
            if (!strata::cli::parse_double(next(), options.sampling.future_entropy_alpha, -1.0, 1.0)) return false;
        } else if (argument == "--future-entropy-curve") {
            const auto curve = next();
            if (curve == "article") {
                options.sampling.future_entropy_curve = strata::FutureEntropyCurve::Article;
            } else if (curve == "crossfade") {
                options.sampling.future_entropy_curve = strata::FutureEntropyCurve::Crossfade;
            } else {
                return false;
            }
        } else if (argument == "--alpha-wave-amplitude") {
            if (!strata::cli::parse_double(next(), options.sampling.future_entropy_wave_amplitude, 0.0, 2.0)) return false;
        } else if (argument == "--alpha-wave-period") {
            if (!strata::cli::parse_double(next(), options.sampling.future_entropy_wave_period, 1.0, 1e6)) return false;
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
    std::string sampling_error;
    if (!strata::validate_sampling_options(options.sampling, sampling_error)) {
        std::cerr << "invalid sampler option: " << sampling_error << '\n';
        return false;
    }
    return !options.model.empty() &&
           (options.model_type == "glm" || options.model_type == "deepseek" ||
            options.model_type == "gemma4" || options.model_type == "kimi-k3");
}

// A run is reproducible when nothing stochastic is enabled. Temperature alone
// no longer decides that: XTC draws even at temperature zero.
bool deterministic(const strata::SamplingOptions& sampling) {
    return sampling.temperature == 0.0 && sampling.xtc_probability == 0.0;
}

// Compact description of which stages are actually on, for the startup banner
// and the JSONL handshake. Silent stages are omitted rather than printed at
// their identity value.
std::string sampler_summary(const strata::SamplingOptions& sampling) {
    std::ostringstream text;
    text << "temperature=" << sampling.temperature;
    if (sampling.top_k != 0U) text << " top_k=" << sampling.top_k;
    if (sampling.top_p < 1.0) text << " top_p=" << sampling.top_p;
    if (sampling.min_p > 0.0) text << " min_p=" << sampling.min_p;
    if (sampling.typical_p < 1.0) text << " typical_p=" << sampling.typical_p;
    if (sampling.xtc_probability > 0.0) {
        text << " xtc=" << sampling.xtc_probability << '@' << sampling.xtc_threshold;
    }
    if (sampling.presence_penalty != 0.0) {
        text << " presence=" << sampling.presence_penalty;
    }
    if (sampling.frequency_penalty != 0.0) {
        text << " frequency=" << sampling.frequency_penalty;
    }
    if (sampling.repetition_penalty != 1.0) {
        text << " repetition=" << sampling.repetition_penalty;
    }
    if (sampling.penalty_window != 0U) text << " window=" << sampling.penalty_window;
    if (sampling.dry_multiplier > 0.0) {
        text << " dry=" << sampling.dry_multiplier << '^' << sampling.dry_base;
    }
    if (sampling.no_repeat_ngram != 0U) {
        text << " no_repeat_ngram=" << sampling.no_repeat_ngram;
    }
    if (sampling.future_entropy_candidates != 0U) {
        text << " future_entropy=" << sampling.future_entropy_candidates
             << '/' << sampling.future_entropy_top_n
             << " alpha=" << sampling.future_entropy_alpha
             << (sampling.future_entropy_curve ==
                         strata::FutureEntropyCurve::Crossfade
                     ? " curve=crossfade" : "");
        if (sampling.future_entropy_wave_amplitude != 0.0) {
            text << " alpha_wave=" << sampling.future_entropy_wave_amplitude
                 << '@' << sampling.future_entropy_wave_period;
        }
    }
    return text.str();
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
        if (last_token_time_) {
            const double interval = std::chrono::duration<double>(
                now - *last_token_time_).count();
            if (interval_count_ == intervals_.size()) {
                interval_total_ -= intervals_[interval_cursor_];
            } else {
                ++interval_count_;
            }
            intervals_[interval_cursor_] = interval;
            interval_total_ += interval;
            interval_cursor_ = (interval_cursor_ + 1U) % intervals_.size();
        }
        last_token_time_ = now;
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
        (void)now;
        return interval_total_ > 0.0
            ? static_cast<double>(interval_count_) / interval_total_
            : 0.0;
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
    std::optional<std::chrono::steady_clock::time_point> last_token_time_;
    std::array<double, 16> intervals_{};
    std::size_t interval_cursor_{};
    std::size_t interval_count_{};
    double interval_total_{};
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
            ? static_cast<double>(result.metrics.prefill_tokens) /
                  result.metrics.prefill_seconds
            : 0.0;
        std::ostringstream fields;
        fields << std::fixed << std::setprecision(6)
               << "\"prompt_tokens\":" << result.metrics.prompt_tokens
               << ",\"prefill_tokens\":" << result.metrics.prefill_tokens
               << ",\"reused_prompt_tokens\":"
               << result.metrics.reused_prompt_tokens
               << ",\"incremental_kv_continuation\":"
               << (result.metrics.incremental_kv_continuation ? "true" : "false")
               << ",\"decode_tokens\":" << result.metrics.decode_tokens
               << ",\"prefill_seconds\":" << result.metrics.prefill_seconds
               << ",\"prefill_tok_s\":" << prefill_tok_s
               << ",\"decode_seconds\":" << result.metrics.decode_seconds
               << ",\"decode_tok_s\":" << runtime_tok_s
               << ",\"generated_token_ids\":[";
        for (std::size_t index = 0U;
             index < result.generated_token_ids.size(); ++index) {
            if (index != 0U) fields << ',';
            fields << result.generated_token_ids[index];
        }
        fields << ']';
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
               << ",\"temperature\":" << options.sampling.temperature
               << ",\"exact\":" << (deterministic(options.sampling) ? "true" : "false")
               << ",\"sampler\":\""
               << strata::cli::json_escape(sampler_summary(options.sampling)) << '"'
               << ",\"flash_attention\":"
               << (options.flash_attention ? "true" : "false")
               << ",\"incremental_kv_continuation\":"
               << (options.incremental_kv_continuation ? "true" : "false")
               << ",\"block_kv_cache\":"
               << (options.block_kv_cache ? "true" : "false")
               << ",\"pin_resident_arena\":"
               << (options.pin_resident_arena ? "true" : "false")
               << ",\"prepack_mhc\":"
               << (options.prepack_mhc ? "true" : "false");
        protocol_event("hello", fields.str());
        protocol_message("status", "Loading model");
    }

    strata::RuntimeConfig config;
    config.model = options.model_type == "glm"
                       ? strata::RuntimeModel::Glm52
                   : options.model_type == "gemma4"
                       ? strata::RuntimeModel::Gemma4
                   : options.model_type == "kimi-k3"
                       ? strata::RuntimeModel::KimiK3
                       : strata::RuntimeModel::DeepSeekV4;
    config.devices = options.devices;
    config.maximum_context_tokens = options.context_size;
    config.vram_cache_fraction = options.vram_fraction;
    config.verbose = options.model_type == "deepseek";
    config.load_progress = options.model_type != "deepseek";
    config.sampling = options.sampling;
    config.enable_flash_attention =
        options.flash_attention || options.model_type == "gemma4";
    config.enable_incremental_kv_continuation =
        options.incremental_kv_continuation;
    config.deepseek_block_kv_cache = options.block_kv_cache;
    config.pin_resident_arena = options.pin_resident_arena;
    config.prepack_mhc_projection = options.prepack_mhc;
    config.placement_cache_directory = options.plan_cache;
    config.use_placement_cache = options.use_plan_cache;
    config.refresh_placement_plan = options.replan;
    config.report_placement_plan = true;

    if (options.dry_run) {
        const auto resolved = strata::resolve_placement_plan(
            strata::placement_request_for(options.model, config),
            options.plan_cache, options.use_plan_cache, options.replan);
        if (!resolved.ok()) {
            for (const auto& error : resolved.errors) {
                std::cerr << "error: " << error << '\n';
                if (options.jsonl_protocol) protocol_message("error", error, true);
            }
            return 1;
        }
        if (options.jsonl_protocol) {
            std::cout << strata::encode_placement_plan(resolved.value.plan)
                      << std::flush;
        } else {
            std::cout << strata::render_placement_report(resolved.value.plan);
        }
        if (resolved.value.from_cache) {
            std::cerr << "[dry-run] reused cached plan "
                      << resolved.value.cache_path << '\n';
        } else if (resolved.value.stored) {
            std::cerr << "[dry-run] wrote plan " << resolved.value.cache_path
                      << '\n';
        } else {
            std::cerr << "[dry-run] plan not cached (--no-plan-cache)\n";
        }
        std::cerr << "[dry-run] no weights were read"
                  << (resolved.value.stored || resolved.value.from_cache
                          ? "; the next load reuses this plan\n" : "\n");
        return resolved.value.plan.fits ? 0 : 1;
    }

    strata::RuntimeSession runtime;
    std::cerr << "[startup] model_type=" << options.model_type
              << " devices=" << strata::cli::devices_text(options.devices)
              << " context=" << options.context_size
              << " max_new=" << options.max_new_tokens
              << " vram_fraction=" << options.vram_fraction
              << " seed=" << options.sampling.seed << '\n'
              << "[sampler] " << sampler_summary(options.sampling) << '\n'
              << "[attention] "
              << ((options.flash_attention || options.model_type == "gemma4")
                      ? "CUDA FlashAttention" : "scalar reference")
              << '\n'
              << "[contract] "
              << (deterministic(options.sampling) ? "exact greedy"
                                                  : "seeded Gumbel-max sampled")
              << " base-model decode; no hidden fallback\n";
    if (options.sampling.future_entropy_candidates != 0U) {
        std::cerr << "[sampler] future entropy looks ahead one step per "
                  << "candidate: expect about "
                  << (options.sampling.future_entropy_candidates + 1U)
                  << "x the decode time of the same settings without it\n";
    }
    std::cerr << "[startup] loading model; this can take several minutes...\n";
    const auto initialization_started = std::chrono::steady_clock::now();
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
