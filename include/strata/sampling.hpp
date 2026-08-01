#pragma once

#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace strata {

// Sampling runs as a fixed pipeline. Penalties rewrite the model's logits;
// every truncation stage then reads the resulting natural distribution, and
// temperature rescales only the surviving candidates immediately before the
// Gumbel-max draw. The order is:
//
//   repetition/presence/frequency -> DRY -> n-gram ban -> logit bias
//   -> top_k -> top_p -> min_p -> typical_p -> XTC -> temperature -> draw
//
// Truncating on the natural distribution is what makes the thresholds mean
// what they say: min_p 0.05 is "at least 5% as likely as the best token
// according to the model", independent of temperature.
struct SamplingOptions {
    // Rescales the surviving candidates before the draw. Zero is greedy.
    double temperature{1.0};

    // Truncation. Each stage is disabled at its identity value.
    double top_p{1.0};             // keep the smallest mass >= top_p
    std::uint32_t top_k{};         // 0 disables; keep the k most likely
    double min_p{};                // 0 disables; keep p >= min_p * p_max
    double typical_p{1.0};         // 1 disables; keep by |ln p + H| ascending

    // Exclude Top Choices: with probability xtc_probability, drop every
    // candidate above xtc_threshold except the least likely of them. Removes
    // the safe continuation while leaving a plausible one in place.
    double xtc_probability{};
    double xtc_threshold{0.1};

    // Repetition control. presence/frequency are subtractive and OpenAI
    // compatible; repetition_penalty is the multiplicative llama.cpp form.
    // penalty_window bounds all three to the last N generated tokens; 0 means
    // the entire generated sequence.
    double presence_penalty{};
    double frequency_penalty{};
    double repetition_penalty{1.0};
    std::uint32_t penalty_window{};

    // DRY: penalize the token that would extend the longest repeated suffix,
    // by dry_multiplier * dry_base^(match - dry_allowed_length).
    double dry_multiplier{};
    double dry_base{1.75};
    std::uint32_t dry_allowed_length{2U};
    std::uint32_t dry_window{512U};

    // Hard ban on any token completing an n-gram already in the sequence.
    std::uint32_t no_repeat_ngram{};

    std::uint64_t seed{33'377'335U};
    std::vector<std::pair<std::uint32_t, double>> logit_bias;
    std::uint32_t top_logprobs{};
    bool return_logprobs{};
};

// What the repetition-aware stages read. `counts` is an optional whole-history
// histogram indexed by token id, kept because the runtimes already maintain
// one. `tokens` is the generated sequence in order, newest last; DRY, the
// n-gram ban, and any windowed penalty require it.
struct SamplingHistory {
    std::span<const std::uint32_t> counts;
    std::span<const std::uint32_t> tokens;
};

// `logprob` and `top` are the model's natural log probabilities: computed from
// the unmodified logits, before penalties, truncation, and temperature. They
// describe the model, not the sampler settings, so they stay comparable across
// requests that used different knobs.
struct TokenLogprob {
    std::uint32_t token{};
    double logprob{};
    std::vector<std::pair<std::uint32_t, double>> top;
};

// Reproducible decode: no stage draws from the generator.
[[nodiscard]] inline SamplingOptions greedy_sampling() {
    SamplingOptions options;
    options.temperature = 0.0;
    return options;
}

[[nodiscard]] TokenLogprob sample_logits(
    std::span<const float> logits, const SamplingOptions& options,
    const SamplingHistory& history, std::mt19937_64& generator);

[[nodiscard]] std::uint32_t sample_logits_gumbel(
    std::span<const float> logits, double temperature, std::mt19937_64& generator);

// Single definition of the accepted ranges, shared by the runtimes and the
// HTTP surface so a new knob cannot be validated in one place and not another.
[[nodiscard]] bool validate_sampling_options(
    const SamplingOptions& options, std::string& error);

}  // namespace strata
