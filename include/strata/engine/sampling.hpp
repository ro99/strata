#pragma once

#include "strata/platform/result.hpp"

#include <cstdint>
#include <functional>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace strata {

// Selects the two exponents of the future-entropy score from one crossfader.
// Both mappings agree at the endpoints and differ only in between.
enum class FutureEntropyCurve : std::uint8_t {
    // a = 1 - max(0, alpha), b = 1 - max(0, -alpha). Count Bayesie's mapping:
    // alpha 0 is exactly s = p * H_hat, the article's headline score.
    Article,
    // a = 1 - t, b = t for t = (alpha + 1) / 2. Constant exponent sum, so
    // alpha 0 is sqrt(p * H_hat). The reference implementation's mapping,
    // kept because its measured operating points were tuned against it.
    Crossfade,
};

// Sampling runs as a fixed pipeline. Penalties rewrite the model's logits;
// every truncation stage then reads the resulting natural distribution, and
// temperature rescales only the surviving candidates immediately before the
// Gumbel-max draw. The order is:
//
//   repetition/presence/frequency -> DRY -> n-gram ban -> logit bias
//   -> top_k -> top_p -> min_p -> typical_p -> XTC -> future entropy
//   -> temperature -> draw
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

    // Future entropy: prefer candidates that keep the next step's options
    // open. The model is run one step past each surviving candidate w to get
    // q_w = p(V | c + w); the normalized entropy of that distribution's top-n
    //
    //   H_hat(w) = [-sum q~_w(v) ln q~_w(v)] / ln n   in [0, 1]
    //
    // reweights the candidate by how much future choice it unlocks:
    //
    //   s(w) = p(w | c)^a * H_hat(w)^b
    //
    // The draw is then made from s renormalized over the candidates, so
    // temperature rescales the blended score rather than the raw probability.
    //
    // This is the only stage that costs forward passes: one per candidate, so
    // a decode step costs (future_entropy_candidates + 1) of them. It also
    // needs a lookahead evaluator; without one the sample reports a failure
    // rather than quietly degrading to ordinary sampling.
    //
    // A truncation stage that cuts the tail on relative plausibility is
    // required in front of it, min_p 0.05 upwards. Broken word-fragments have
    // maximally uncertain futures, so entropy selects them unless something
    // has already removed them.
    std::uint32_t future_entropy_candidates{};   // 0 disables the stage
    std::uint32_t future_entropy_top_n{30U};     // width of q_w for H_hat
    double future_entropy_alpha{};               // -1 probability, +1 entropy
    FutureEntropyCurve future_entropy_curve{FutureEntropyCurve::Article};

    // Rhythmic decoding: alpha oscillates over the generated sequence as
    // alpha + amplitude * sin(2*pi*step/period), clamped back into [-1, 1],
    // so the text alternates between exploratory and consolidating phases.
    double future_entropy_wave_amplitude{};      // 0 disables the wave
    double future_entropy_wave_period{60.0};     // in generated tokens

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
    // Only the future-entropy stage can fail: it is the one stage that needs
    // the model rather than the logits it was handed.
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

// Runs the model one step past each candidate and reports how open the future
// it unlocks is. `candidates` is in descending model probability; write
// H_hat(w) in [0, 1] for each into `normalized_entropy`, same order.
//
// The evaluator owns the speculative state: it must leave the runtime exactly
// as it found it, because the token the sampler goes on to pick is decoded
// from that state. Returning errors fails the sample rather than falling back.
using FutureEntropyEvaluator = std::function<ValidationResult(
    std::span<const std::uint32_t> candidates, std::uint32_t top_n,
    std::span<double> normalized_entropy)>;

// H_hat over the renormalized top-n of softmax(logits), scaled to [0, 1] by
// ln n. Shared so both runtimes measure the future the same way.
[[nodiscard]] double normalized_top_n_entropy(
    std::span<const float> logits, std::uint32_t top_n);

// The alpha in force at generation step `step`, after the wave and the clamp.
[[nodiscard]] double future_entropy_alpha_at(
    const SamplingOptions& options, std::uint64_t step);

// Reproducible decode: no stage draws from the generator.
[[nodiscard]] inline SamplingOptions greedy_sampling() {
    SamplingOptions options;
    options.temperature = 0.0;
    return options;
}

// `lookahead` is required exactly when future_entropy_candidates is non-zero,
// and unused otherwise.
[[nodiscard]] TokenLogprob sample_logits(
    std::span<const float> logits, const SamplingOptions& options,
    const SamplingHistory& history, std::mt19937_64& generator,
    const FutureEntropyEvaluator& lookahead = {});

[[nodiscard]] std::uint32_t sample_logits_gumbel(
    std::span<const float> logits, double temperature, std::mt19937_64& generator);

// Single definition of the accepted ranges, shared by the runtimes and the
// HTTP surface so a new knob cannot be validated in one place and not another.
[[nodiscard]] bool validate_sampling_options(
    const SamplingOptions& options, std::string& error);

}  // namespace strata
