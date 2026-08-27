// strata-chat: the interactive front end.
//
// Two modes share one option surface:
//   * a tty gets the line editor, slash commands and a per-turn timing line;
//   * a pipe gets one prompt per line and plain output;
//
// Throughput is reported where it can be believed: once per turn, and as a
// session aggregate on exit. A live counter would only ever show the last few
// tokens, and none of it belongs in the terminal title.

#include "strata/app/runtime.hpp"

#include "strata/app/cli.hpp"
#include "strata/app/cli_console.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <unistd.h>

namespace term = strata::cli::term;

namespace {

// ------------------------------------------------------------- interrupt ---

// Set by SIGINT while a generation is running. The token callback returns
// false on it, which the runtimes' decode loops read as a cancellation; a
// second interrupt inside the same generation gives up and exits, because a
// model that does not check cancellation would otherwise trap the terminal.
std::atomic<int> g_interrupts{0};

extern "C" void handle_interrupt(int) {
    if (g_interrupts.fetch_add(1) + 1 >= 2) {
        const char message[] = "\n";
        (void)!write(2, message, sizeof(message) - 1U);
        std::_Exit(130);
    }
}

void install_interrupt_handler() {
    struct sigaction action {};
    action.sa_handler = handle_interrupt;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    sigaction(SIGINT, &action, nullptr);
}

// --------------------------------------------------------------- options ---

struct Options {
    std::string model;
    std::string model_type;
    std::string prompt;
    std::string system_prompt;
    std::vector<int> devices;
    std::uint32_t context_size{2048U};
    std::uint32_t max_new_tokens{256U};
    std::uint32_t prefill_page_tokens{};
    strata::SamplingOptions sampling{strata::greedy_sampling()};
    double vram_fraction{0.85};
    bool devices_explicit{};
    bool flash_attention{};
    bool incremental_kv_continuation{true};
    bool block_kv_cache{};
    bool device_resident_runtime{};
    bool rank_local_decode{};
    bool pin_resident_arena{};
    bool prepack_mhc{true};
    bool no_colour{};
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
        // DRY is what the reference implementation reports missing -- an
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

// Flags that consume the following argument. Listed beside the parser so an
// unknown flag in the last position is reported as unknown rather than as a
// known flag missing its value.
bool takes_value(std::string_view argument) {
    static constexpr std::string_view names[] = {
        "--model", "--model-type", "--prompt", "--system", "--plan-cache",
        "--decode-topology", "--context-size", "--max-context", "--max-new",
        "--prefill-page-tokens",
        "--temperature", "--vram-fraction", "--seed", "--preset", "--top-k",
        "--top-p", "--min-p", "--typical-p", "--xtc-probability",
        "--xtc-threshold", "--presence-penalty", "--frequency-penalty",
        "--repetition-penalty", "--penalty-window", "--dry-multiplier",
        "--dry-base", "--dry-allowed-length", "--dry-window",
        "--no-repeat-ngram", "--future-entropy", "--future-entropy-top-n",
        "--alpha", "--future-entropy-curve", "--alpha-wave-amplitude",
        "--alpha-wave-period", "--devices",
    };
    return std::find(std::begin(names), std::end(names), argument) !=
           std::end(names);
}

void usage() {
    std::cerr <<
R"(strata-chat -- interactive chat against a Strata runtime

usage:
  strata-chat --model DIR --model-type TYPE [options]
  strata-chat --model DIR --model-type TYPE --prompt "..."   one-shot
  strata-chat --model DIR --model-type TYPE --dry-run        place, do not load

required:
  --model DIR                 checkpoint directory
  --model-type TYPE           gemma4 | deepseek | glm | laguna | inkling | kimi-k3

session:
  --prompt TEXT               answer TEXT and exit instead of prompting
  --system TEXT               prepend a system message to the conversation
  --context-size N            context window in tokens (default 2048)
  --max-new N                 tokens generated per turn (default 256)
  --devices 0,1,2             CUDA devices to use
  --vram-fraction F           share of each device given to caches (default 0.85)
  --no-color                  plain output even on a terminal

execution:
  --flash-attention           CUDA FlashAttention instead of the scalar path
  --full-reprefill            re-prefill each turn instead of reusing the KV prefix
  --block-kv-cache            DeepSeek physical KV pages
  --device-resident-runtime   the whole DeepSeek device-resident decode contract
  --decode-topology T         centralized (default) | rank-local-tp2
  --prefill-page-tokens N     DeepSeek prompt rows per layer-major page
  --pin-resident-arena        pin the resident weight arena
  --no-prepack-mhc            keep mHC projections in their stored layout

placement:
  --dry-run                   size and place every component, then exit
  --replan                    recompute a cached plan
  --plan-cache DIR            where plans are read and written
  --no-plan-cache             neither read nor write a plan

sampler:
  --preset NAME               precise | balanced | creative | future-entropy
  --temperature F             --seed N --top-k N --top-p F --min-p F --typical-p F
  --xtc-probability F         --xtc-threshold F
  --presence-penalty F        --frequency-penalty F
  --repetition-penalty F      --penalty-window N
  --dry-multiplier F          --dry-base F --dry-allowed-length N --dry-window N
  --no-repeat-ngram N

future entropy:
  --future-entropy N          --future-entropy-top-n N --alpha F
  --future-entropy-curve C    article | crossfade
  --alpha-wave-amplitude F    --alpha-wave-period F

commands, once running:
  /help /clear /regen /stats /save FILE /exit

notes:
  --preset writes defaults; flags after it override them. --temperature with no
  preset is plain temperature sampling: no truncation, penalties, or XTC.
  Truncation reads the model's own distribution, so --min-p and --top-p mean
  the same thing at any temperature.

  --future-entropy N scores each of the N likeliest candidates by how open the
  distribution one step past it is, at one extra forward pass per candidate: a
  token takes N+1 decode steps. Pair it with --min-p 0.05 or higher.

  --device-resident-runtime is a bundle, not a knob: physical KV pages,
  device-resident mHC, CUDA attention, the scalar lightning indexer, and routed
  experts in the two NUMA-local CPU shards. It overrides the individual cache
  and attention flags. --decode-topology rank-local-tp2 adds rank-local decode
  on top and needs an NCCL build with exactly two devices; it is admitted
  fail-closed and supports at most 65,536 context tokens (issue #22). Both are
  DeepSeek-only. Each topology is exact against its own oracle but they are not
  token-identical to each other, so switching one mid-project changes the text.

  --dry-run reads no weights. It writes the plan to the cache, so the next real
  load places exactly what it printed. A plan is keyed by checkpoint, GPUs,
  context size and device list, and is recomputed when any of them change.
)";
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
            options.device_resident_runtime = false;
            continue;
        }
        if (argument == "--device-resident-runtime") {
            options.block_kv_cache = true;
            options.device_resident_runtime = true;
            // Implications of the contract, applied here so the banner and the
            // hello event report what will actually run. The runtime enforces
            // them again for embedders that never pass through this parser.
            options.flash_attention = true;
            options.prepack_mhc = false;
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
        if (argument == "--no-color" || argument == "--no-colour") {
            options.no_colour = true;
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
        if (!takes_value(argument)) {
            std::cerr << "error: unknown argument: " << argument << '\n';
            return false;
        }
        if (index + 1 >= argc) {
            std::cerr << "error: " << argument << " needs a value\n";
            return false;
        }
        const auto next = [&]() { return std::string_view(argv[++index]); };
        // Every rejected value says which flag rejected it; a bare "false"
        // from the parser leaves the user guessing which of thirty flags it
        // was.
        const auto reject = [&](std::string_view value) {
            std::cerr << "error: " << argument << ": bad value '" << value
                      << "'\n";
            return false;
        };
        if (argument == "--model") options.model = std::string(next());
        else if (argument == "--model-type") options.model_type = std::string(next());
        else if (argument == "--prompt") options.prompt = std::string(next());
        else if (argument == "--system") options.system_prompt = std::string(next());
        else if (argument == "--plan-cache") options.plan_cache = std::string(next());
        else if (argument == "--decode-topology") {
            const auto topology = next();
            if (topology == "centralized") {
                options.rank_local_decode = false;
            } else if (topology == "rank-local-tp2") {
                // Rank-local decode is a decode-shaped ownership of the same
                // weights the device-resident contract places; it does not
                // replace prefill, which stays centralized. Everything that
                // contract requires is required here too, so the opt-in
                // implies it rather than silently running rank-local against
                // a scalar cache.
                options.rank_local_decode = true;
                options.block_kv_cache = true;
                options.device_resident_runtime = true;
                options.flash_attention = true;
                options.prepack_mhc = false;
            } else {
                std::cerr << "error: unknown --decode-topology: " << topology << '\n';
                return false;
            }
        }
        else if (argument == "--context-size" || argument == "--max-context") {
            {
                const auto value = next();
                if (!strata::cli::parse_positive_u32(value, options.context_size)) return reject(value);
            }
        } else if (argument == "--max-new") {
            {
                const auto value = next();
                if (!strata::cli::parse_positive_u32(value, options.max_new_tokens)) return reject(value);
            }
        } else if (argument == "--prefill-page-tokens") {
            {
                const auto value = next();
                if (!strata::cli::parse_positive_u32(
                        value, options.prefill_page_tokens)) {
                    return reject(value);
                }
            }
        } else if (argument == "--temperature") {
            {
                const auto value = next();
                if (!strata::cli::parse_double(value, options.sampling.temperature, 0.0, 10.0)) return reject(value);
            }
        } else if (argument == "--vram-fraction") {
            {
                const auto value = next();
                if (!strata::cli::parse_double(value, options.vram_fraction, 0.0, 0.95)) return reject(value);
            }
        } else if (argument == "--seed") {
            {
                const auto value = next();
                if (!strata::cli::parse_u64(value, options.sampling.seed)) return reject(value);
            }
        } else if (argument == "--preset") {
            {
                const auto value = next();
                if (!apply_sampler_preset(value, options.sampling)) return reject(value);
            }
        } else if (argument == "--top-k") {
            {
                const auto value = next();
                if (!strata::cli::parse_u32(value, options.sampling.top_k)) return reject(value);
            }
        } else if (argument == "--top-p") {
            {
                const auto value = next();
                if (!strata::cli::parse_double(value, options.sampling.top_p, 0.0, 1.0)) return reject(value);
            }
        } else if (argument == "--min-p") {
            {
                const auto value = next();
                if (!strata::cli::parse_double(value, options.sampling.min_p, 0.0, 1.0)) return reject(value);
            }
        } else if (argument == "--typical-p") {
            {
                const auto value = next();
                if (!strata::cli::parse_double(value, options.sampling.typical_p, 0.0, 1.0)) return reject(value);
            }
        } else if (argument == "--xtc-probability") {
            {
                const auto value = next();
                if (!strata::cli::parse_double(value, options.sampling.xtc_probability, 0.0, 1.0)) return reject(value);
            }
        } else if (argument == "--xtc-threshold") {
            {
                const auto value = next();
                if (!strata::cli::parse_double(value, options.sampling.xtc_threshold, 0.0, 1.0)) return reject(value);
            }
        } else if (argument == "--presence-penalty") {
            {
                const auto value = next();
                if (!strata::cli::parse_double(value, options.sampling.presence_penalty, -2.0, 2.0)) return reject(value);
            }
        } else if (argument == "--frequency-penalty") {
            {
                const auto value = next();
                if (!strata::cli::parse_double(value, options.sampling.frequency_penalty, -2.0, 2.0)) return reject(value);
            }
        } else if (argument == "--repetition-penalty") {
            {
                const auto value = next();
                if (!strata::cli::parse_double(value, options.sampling.repetition_penalty, 0.0, 10.0)) return reject(value);
            }
        } else if (argument == "--penalty-window") {
            {
                const auto value = next();
                if (!strata::cli::parse_u32(value, options.sampling.penalty_window)) return reject(value);
            }
        } else if (argument == "--dry-multiplier") {
            {
                const auto value = next();
                if (!strata::cli::parse_double(value, options.sampling.dry_multiplier, 0.0, 10.0)) return reject(value);
            }
        } else if (argument == "--dry-base") {
            {
                const auto value = next();
                if (!strata::cli::parse_double(value, options.sampling.dry_base, 1.0, 8.0)) return reject(value);
            }
        } else if (argument == "--dry-allowed-length") {
            {
                const auto value = next();
                if (!strata::cli::parse_positive_u32(value, options.sampling.dry_allowed_length)) return reject(value);
            }
        } else if (argument == "--dry-window") {
            {
                const auto value = next();
                if (!strata::cli::parse_u32(value, options.sampling.dry_window)) return reject(value);
            }
        } else if (argument == "--no-repeat-ngram") {
            {
                const auto value = next();
                if (!strata::cli::parse_u32(value, options.sampling.no_repeat_ngram)) return reject(value);
            }
        } else if (argument == "--future-entropy") {
            {
                const auto value = next();
                if (!strata::cli::parse_u32(value, options.sampling.future_entropy_candidates)) return reject(value);
            }
        } else if (argument == "--future-entropy-top-n") {
            {
                const auto value = next();
                if (!strata::cli::parse_u32(value, options.sampling.future_entropy_top_n)) return reject(value);
            }
        } else if (argument == "--alpha") {
            {
                const auto value = next();
                if (!strata::cli::parse_double(value, options.sampling.future_entropy_alpha, -1.0, 1.0)) return reject(value);
            }
        } else if (argument == "--future-entropy-curve") {
            const auto curve = next();
            if (curve == "article") {
                options.sampling.future_entropy_curve = strata::FutureEntropyCurve::Article;
            } else if (curve == "crossfade") {
                options.sampling.future_entropy_curve = strata::FutureEntropyCurve::Crossfade;
            } else {
                std::cerr << "error: unknown --future-entropy-curve: " << curve << '\n';
                return false;
            }
        } else if (argument == "--alpha-wave-amplitude") {
            {
                const auto value = next();
                if (!strata::cli::parse_double(value, options.sampling.future_entropy_wave_amplitude, 0.0, 2.0)) return reject(value);
            }
        } else if (argument == "--alpha-wave-period") {
            {
                const auto value = next();
                if (!strata::cli::parse_double(value, options.sampling.future_entropy_wave_period, 1.0, 1e6)) return reject(value);
            }
        } else if (argument == "--devices") {
            {
                const auto value = next();
                if (!strata::cli::parse_devices(value, options.devices)) return reject(value);
            }
            options.devices_explicit = true;
        } else {
            // Unreachable while takes_value() and this chain agree; kept so a
            // flag added to one and not the other is rejected instead of
            // silently swallowing its value.
            std::cerr << "error: unknown argument: " << argument << '\n';
            return false;
        }
    }
    if (!options.devices_explicit) options.devices = {0, 1, 2};
    std::string sampling_error;
    if (!strata::validate_sampling_options(options.sampling, sampling_error)) {
        std::cerr << "error: invalid sampler option: " << sampling_error << '\n';
        return false;
    }
    if (options.model.empty()) {
        std::cerr << "error: --model is required\n";
        return false;
    }
    // Resolved through the model registry rather than a hardcoded list, so a
    // new model needs no edit here.
    if (strata::find_model_by_cli_name(options.model_type) == nullptr) {
        std::cerr << "error: unknown --model-type: "
                  << (options.model_type.empty() ? "(missing)" : options.model_type)
                  << '\n';
        return false;
    }
    return true;
}

// A run is reproducible when nothing stochastic is enabled. Temperature alone
// no longer decides that: XTC draws even at temperature zero.
bool deterministic(const strata::SamplingOptions& sampling) {
    return sampling.temperature == 0.0 && sampling.xtc_probability == 0.0;
}

// Compact description of which stages are actually on, for the banner and the
// JSONL handshake. Silent stages are omitted rather than printed at their
// identity value.
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

// ------------------------------------------------------------ statistics ---

std::string with_thousands(std::uint64_t value) {
    std::string digits = std::to_string(value);
    for (std::size_t position = digits.size(); position > 3U;) {
        position -= 3U;
        digits.insert(position, ",");
    }
    return digits;
}

double rate(std::uint64_t tokens, double seconds) {
    return seconds > 0.0 ? static_cast<double>(tokens) / seconds : 0.0;
}

// What one turn cost. The runtime measures prefill and decode separately, and
// they are never averaged together: a prompt token and a generated token do
// not cost the same thing and a combined figure means nothing.
struct TurnStats {
    std::uint64_t prompt_tokens{};
    std::uint64_t prefill_tokens{};
    std::uint64_t decode_tokens{};
    double prefill_seconds{};
    double decode_seconds{};
    bool interrupted{};
};

struct SessionStats {
    std::uint64_t turns{};
    std::uint64_t prompt_tokens{};
    std::uint64_t prefill_tokens{};
    std::uint64_t decode_tokens{};
    double prefill_seconds{};
    double decode_seconds{};
    double load_seconds{};

    void add(const TurnStats& turn) {
        ++turns;
        prompt_tokens += turn.prompt_tokens;
        prefill_tokens += turn.prefill_tokens;
        decode_tokens += turn.decode_tokens;
        prefill_seconds += turn.prefill_seconds;
        decode_seconds += turn.decode_seconds;
    }
};

// The per-turn line. Dim, one line, after the answer -- the only place a live
// counter could have gone that does not fight with the text being streamed.
void print_turn_stats(const TurnStats& turn) {
    std::ostringstream line;
    line << std::fixed << std::setprecision(2);
    line << "prefill " << with_thousands(turn.prefill_tokens) << " tok in "
         << turn.prefill_seconds << " s (" << rate(turn.prefill_tokens, turn.prefill_seconds)
         << " tok/s)   decode " << with_thousands(turn.decode_tokens) << " tok in "
         << turn.decode_seconds << " s (" << rate(turn.decode_tokens, turn.decode_seconds)
         << " tok/s)";
    if (turn.prefill_tokens != turn.prompt_tokens) {
        line << "   reused " << with_thousands(turn.prompt_tokens - turn.prefill_tokens)
             << " tok";
    }
    if (turn.interrupted) line << "   [interrupted]";
    std::cerr << '\n' << term::grey() << "  " << line.str() << term::reset()
              << "\n" << std::flush;
}

// The session aggregate, printed once on exit. Each rate is total tokens over
// total seconds for that phase, not a mean of per-turn rates: a two-token turn
// should not weigh the same as a two-hundred-token one.
void print_session_stats(const SessionStats& session) {
    if (session.turns == 0U) return;
    std::ostringstream report;
    report << std::fixed << std::setprecision(2);
    report << '\n' << term::bold() << "  session" << term::reset() << '\n'
           << term::grey()
           << "    turns     " << session.turns << '\n'
           << "    prefill   " << with_thousands(session.prefill_tokens)
           << " tok in " << session.prefill_seconds << " s   "
           << rate(session.prefill_tokens, session.prefill_seconds)
           << " tok/s average\n"
           << "    decode    " << with_thousands(session.decode_tokens)
           << " tok in " << session.decode_seconds << " s   "
           << rate(session.decode_tokens, session.decode_seconds)
           << " tok/s average\n";
    if (session.prompt_tokens > session.prefill_tokens) {
        report << "    reused    "
               << with_thousands(session.prompt_tokens - session.prefill_tokens)
               << " prompt tok not re-prefilled\n";
    }
    report << "    load      " << session.load_seconds << " s\n" << term::reset();
    std::cerr << report.str() << std::flush;
}

// --------------------------------------------------------------- display ---

// Streams one answer as complete UTF-8 runs so it can be selected, piped, or
// copied without stripping decoration out of it.
class AnswerStream {
public:
    void token(std::string_view piece) {
        assembler_.push(piece, [this](std::string_view text) { write(text); });
    }

    void finish() {
        assembler_.finish([this](std::string_view text) { write(text); });
        if (wrote_any_ && !ends_with_newline_) std::cout << '\n';
        std::cout << std::flush;
    }

private:
    void write(std::string_view text) {
        if (text.empty()) return;
        wrote_any_ = true;
        ends_with_newline_ = text.back() == '\n';
        std::cout.write(text.data(), static_cast<std::streamsize>(text.size()));
        std::cout << std::flush;
    }

    bool wrote_any_{};
    bool ends_with_newline_{};
    term::Utf8Assembler assembler_;
};

// ------------------------------------------------------------------ turn ---

struct TurnOutcome {
    bool ok{};
    std::string text;
    TurnStats stats;
};

TurnOutcome run_turn(strata::RuntimeSession& runtime, const Options& options,
                     std::span<const strata::ChatMessage> messages) {
    TurnOutcome outcome;

    // Echo off for the whole turn, so a keystroke during generation cannot
    // land in the middle of the answer.
    const term::QuietInput quiet;

    term::Spinner spinner;
    // The separator goes out before the spinner starts, or the spinner thread
    // races it and draws its first frame on the wrong line.
    std::cerr << '\n' << std::flush;
    // Prefill prints nothing and can run for minutes on a long prompt.
    spinner.start("thinking");
    bool spinning = true;

    g_interrupts.store(0);
    AnswerStream stream;
    const strata::TokenStreamCallback on_token =
        [&](std::uint32_t, std::string_view piece) {
            if (spinning) {
                spinner.stop();
                spinning = false;
            }
            stream.token(piece);
            return g_interrupts.load() == 0;
        };

    const auto result = runtime.generate_chat_stream(
        messages, options.max_new_tokens, on_token);
    if (spinning) spinner.stop();
    stream.finish();

    if (!result.ok()) {
        for (const auto& error : result.errors) {
            std::cerr << term::red() << "  error: " << term::reset()
                      << error << '\n';
        }
        return outcome;
    }

    outcome.ok = true;
    outcome.text = result.text;
    outcome.stats.prompt_tokens = result.metrics.prompt_tokens;
    outcome.stats.prefill_tokens = result.metrics.prefill_tokens;
    outcome.stats.decode_tokens = result.metrics.decode_tokens;
    outcome.stats.prefill_seconds = result.metrics.prefill_seconds;
    outcome.stats.decode_seconds = result.metrics.decode_seconds;
    outcome.stats.interrupted = g_interrupts.load() != 0;
    g_interrupts.store(0);
    return outcome;
}

// ---------------------------------------------------------------- banner ---

void print_field(std::string_view label, std::string_view value) {
    std::cerr << term::grey() << "  " << std::left << std::setw(12)
              << std::string(label) << term::reset() << value << '\n';
}

void print_banner(const Options& options,
                  const strata::ModelRegistration& registration) {
    std::cerr << '\n'
              << "  " << term::bold() << term::cyan() << "strata" << term::reset()
              << term::grey() << "  ·  " << term::reset() << registration.name
              << "\n\n";
    print_field("model", options.model);
    print_field("devices", strata::cli::devices_text(options.devices) +
                               "   context " + std::to_string(options.context_size) +
                               " tokens   vram " +
                               [&] {
                                   std::ostringstream text;
                                   text << std::fixed << std::setprecision(2)
                                        << options.vram_fraction;
                                   return text.str();
                               }());
    print_field("attention",
                (options.flash_attention || registration.flash_attention_by_default)
                    ? "CUDA FlashAttention" : "scalar reference");
    if (options.device_resident_runtime) {
        print_field("decode", options.rank_local_decode
                                  ? "device-resident, rank-local TP2"
                                  : "device-resident, centralized");
    }
    if (options.prefill_page_tokens != 0U) {
        print_field("prefill", std::to_string(options.prefill_page_tokens) +
                                   " tokens/page");
    }
    print_field("sampler", sampler_summary(options.sampling) +
                               (deterministic(options.sampling)
                                    ? "   exact greedy, no hidden fallback"
                                    : "   seeded Gumbel-max"));
    if (options.sampling.future_entropy_candidates != 0U) {
        std::cerr << term::yellow() << "  note        " << term::reset()
                  << "future entropy costs about "
                  << (options.sampling.future_entropy_candidates + 1U)
                  << "x the decode time of the same settings without it\n";
    }
    std::cerr << std::flush;
}

// -------------------------------------------------------- slash commands ---

enum class Command : std::uint8_t { None, Handled, Regenerate, Quit };

void print_commands() {
    std::cerr << term::grey()
              << "\n  /help          this list\n"
                 "  /clear         forget the conversation so far\n"
                 "  /regen         generate the last answer again\n"
                 "  /stats         throughput so far this session\n"
                 "  /save FILE     write the transcript to FILE\n"
                 "  /exit          quit (Ctrl+D does the same)\n\n"
                 "  Ctrl+C stops a generation; on an empty prompt it quits.\n"
              << term::reset() << std::flush;
}

void save_transcript(const std::string& path,
                     const std::vector<strata::ChatMessage>& conversation) {
    std::ofstream file(path);
    if (!file) {
        std::cerr << term::red() << "  cannot write " << path << term::reset() << '\n';
        return;
    }
    for (const auto& message : conversation) {
        const char* role = message.role == strata::ChatRole::User      ? "user"
                           : message.role == strata::ChatRole::System  ? "system"
                           : message.role == strata::ChatRole::Tool    ? "tool"
                                                                       : "assistant";
        file << "## " << role << "\n\n" << message.content << "\n\n";
    }
    std::cerr << term::grey() << "  wrote " << path << term::reset() << '\n';
}

Command handle_command(std::string_view line,
                       std::vector<strata::ChatMessage>& conversation,
                       const SessionStats& session,
                       std::size_t retained_prefix) {
    if (line.empty() || line.front() != '/') return Command::None;
    const auto space = line.find(' ');
    const auto name = line.substr(0U, space);
    const auto argument = space == std::string_view::npos
                              ? std::string_view{}
                              : line.substr(space + 1U);

    if (name == "/help" || name == "/?") {
        print_commands();
        return Command::Handled;
    }
    if (name == "/exit" || name == "/quit") return Command::Quit;
    if (name == "/clear") {
        conversation.resize(retained_prefix);
        std::cerr << term::grey() << "  conversation cleared\n" << term::reset();
        return Command::Handled;
    }
    if (name == "/stats") {
        print_session_stats(session);
        return Command::Handled;
    }
    if (name == "/save") {
        if (argument.empty()) {
            std::cerr << term::red() << "  /save needs a filename\n" << term::reset();
        } else {
            save_transcript(std::string(argument), conversation);
        }
        return Command::Handled;
    }
    if (name == "/regen") {
        if (conversation.size() < retained_prefix + 2U) {
            std::cerr << term::red() << "  nothing to regenerate\n" << term::reset();
            return Command::Handled;
        }
        conversation.pop_back();  // the previous answer
        return Command::Regenerate;
    }
    std::cerr << term::red() << "  unknown command: " << name << term::reset()
              << term::grey() << "   /help for the list\n" << term::reset();
    return Command::Handled;
}

// ------------------------------------------------------------------ main ---

int run_dry_run(const Options& options, const strata::RuntimeConfig& config) {
    const auto resolved = strata::resolve_placement_plan(
        strata::placement_request_for(options.model, config),
        options.plan_cache, options.use_plan_cache, options.replan);
    if (!resolved.ok()) {
        for (const auto& error : resolved.errors) {
            std::cerr << "error: " << error << '\n';
        }
        return 1;
    }
    std::cout << strata::render_placement_report(resolved.value.plan);
    if (resolved.value.from_cache) {
        std::cerr << "  reused cached plan " << resolved.value.cache_path << '\n';
    } else if (resolved.value.stored) {
        std::cerr << "  wrote plan " << resolved.value.cache_path << '\n';
    } else {
        std::cerr << "  plan not cached (--no-plan-cache)\n";
    }
    std::cerr << "  no weights were read"
              << (resolved.value.stored || resolved.value.from_cache
                      ? "; the next load reuses this plan\n" : "\n");
    return resolved.value.plan.fits ? 0 : 1;
}

int run_interactive(strata::RuntimeSession& runtime, const Options& options,
                    std::vector<strata::ChatMessage>& conversation,
                    SessionStats& session) {
    const std::size_t retained_prefix = conversation.size();
    term::LineEditor editor;
    const std::string prompt = term::colour_enabled()
                                   ? std::string(term::bold()) + term::cyan() +
                                         "› " + term::reset()
                                   : "> ";
    std::cerr << term::grey()
              << "\n  /help for commands · Ctrl+C stops a generation · Ctrl+D quits\n"
              << term::reset() << std::flush;

    std::string line;
    for (;;) {
        const auto status = editor.read(prompt, line);
        if (status == term::LineEditor::Status::Eof) break;
        if (status == term::LineEditor::Status::Interrupt) {
            // Ctrl+C on an empty prompt is the conventional way out; with text
            // in the buffer it just drops the text.
            if (line.empty()) break;
            continue;
        }

        // Trim, so a stray trailing space does not become a distinct history
        // entry or a one-space prompt.
        const auto first = line.find_first_not_of(" \t");
        const auto last = line.find_last_not_of(" \t");
        const std::string text = first == std::string::npos
                                     ? std::string{}
                                     : line.substr(first, last - first + 1U);
        if (text.empty()) continue;
        editor.remember(text);

        bool regenerate = false;
        if (text.front() == '/') {
            switch (handle_command(text, conversation, session, retained_prefix)) {
                case Command::Quit: return 0;
                case Command::Handled: continue;
                case Command::Regenerate: regenerate = true; break;
                case Command::None: break;
            }
        }
        if (!regenerate) {
            conversation.push_back({strata::ChatRole::User, text});
        }

        auto outcome = run_turn(runtime, options, conversation);
        if (!outcome.ok) {
            // Drop the turn that failed rather than leaving the conversation
            // holding a question with no answer.
            conversation.pop_back();
            continue;
        }
        session.add(outcome.stats);
        print_turn_stats(outcome.stats);
        conversation.push_back(
            {strata::ChatRole::Assistant, std::move(outcome.text)});
    }
    return 0;
}

int run_piped(strata::RuntimeSession& runtime, const Options& options,
              std::vector<strata::ChatMessage>& conversation,
              SessionStats& session) {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        conversation.push_back({strata::ChatRole::User, line});
        auto outcome = run_turn(runtime, options, conversation);
        if (!outcome.ok) return 1;
        session.add(outcome.stats);
        print_turn_stats(outcome.stats);
        conversation.push_back(
            {strata::ChatRole::Assistant, std::move(outcome.text)});
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        std::cerr << "\nrun with --help for the full option list\n";
        return 2;
    }

    term::detect_colour(!options.no_colour && term::stderr_is_tty());

    // parse_options already rejected an unregistered --model-type.
    const auto* registration = strata::find_model_by_cli_name(options.model_type);
    strata::RuntimeConfig config;
    config.model = registration->model;
    config.devices = options.devices;
    config.maximum_context_tokens = options.context_size;
    config.vram_cache_fraction = options.vram_fraction;
    config.verbose = registration->verbose_by_default;
    config.load_progress = registration->progress_by_default;
    config.sampling = options.sampling;
    config.enable_flash_attention =
        options.flash_attention || registration->flash_attention_by_default;
    config.enable_incremental_kv_continuation = options.incremental_kv_continuation;
    config.deepseek_block_kv_cache = options.block_kv_cache;
    config.deepseek_device_resident_runtime = options.device_resident_runtime;
    config.deepseek_rank_local_decode = options.rank_local_decode;
    config.deepseek_prefill_page_tokens = options.prefill_page_tokens;
    config.pin_resident_arena = options.pin_resident_arena;
    config.prepack_mhc_projection = options.prepack_mhc;
    config.placement_cache_directory = options.plan_cache;
    config.use_placement_cache = options.use_plan_cache;
    config.refresh_placement_plan = options.replan;
    config.report_placement_plan = true;

    if (options.dry_run) return run_dry_run(options, config);

    print_banner(options, *registration);

    strata::RuntimeSession runtime;
    std::cerr << '\n' << term::grey()
              << "  loading; a large checkpoint can take several minutes\n"
              << term::reset() << std::flush;
    const auto load_started = std::chrono::steady_clock::now();
    const auto initialized = runtime.initialize(options.model, config);
    if (!initialized.ok()) {
        for (const auto& error : initialized.errors) {
            std::cerr << term::red() << "  error: " << term::reset()
                      << error << '\n';
        }
        return 1;
    }
    SessionStats session;
    session.load_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - load_started).count();

    std::cerr << term::green() << "  ready" << term::reset() << term::grey()
              << " in " << std::fixed << std::setprecision(2)
              << session.load_seconds << " s\n"
              << term::reset() << std::flush;

    std::vector<strata::ChatMessage> conversation;
    if (!options.system_prompt.empty()) {
        conversation.push_back(
            {strata::ChatRole::System, options.system_prompt});
    }

    install_interrupt_handler();

    int status = 0;
    if (!options.prompt.empty()) {
        conversation.push_back({strata::ChatRole::User, options.prompt});
        auto outcome = run_turn(runtime, options, conversation);
        if (!outcome.ok) return 1;
        session.add(outcome.stats);
        print_turn_stats(outcome.stats);
    } else if (term::stdin_is_tty()) {
        status = run_interactive(runtime, options, conversation, session);
    } else {
        status = run_piped(runtime, options, conversation, session);
    }

    print_session_stats(session);
    return status;
}
