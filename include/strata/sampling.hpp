#pragma once

#include <cstdint>
#include <random>
#include <span>

namespace strata {

[[nodiscard]] std::uint32_t sample_logits_gumbel(
    std::span<const float> logits, double temperature, std::mt19937_64& generator);

/*
 * Exact speculative sampling (Leviathan et al., Chen et al.).
 *
 * Drafting a token from q and then accepting it with probability
 * min(1, p/q), or otherwise resampling from the normalized residual
 * max(0, p - q), produces a token distributed exactly as p. Losslessness does
 * not depend on how good q is, only on the acceptance test being exact.
 *
 * At temperature <= 0 every function here degenerates to greedy decoding and
 * consumes no randomness at all: softmax_probabilities returns the one-hot
 * indicator of argmax, acceptance reduces to "the drafted token is the target
 * argmax", and the residual is a point mass. That is what allows a speculative
 * run to be gated on bit-identical greedy output against plain decode, even
 * though the two paths draw a wildly different number of random values when
 * the temperature is positive.
 */

// Softmax of logits / temperature. At temperature <= 0 this is the one-hot
// indicator of the first maximal logit, matching sample_logits_gumbel's argmax.
void softmax_probabilities(std::span<const float> logits, double temperature,
                           std::span<double> probabilities);

// One uniform in (0, 1), drawn the way sample_logits_gumbel draws its
// per-token uniforms so both samplers share one randomness convention.
[[nodiscard]] double next_uniform(std::mt19937_64& generator);

// Inverse-CDF categorical sample. A point mass consumes no draw.
[[nodiscard]] std::uint32_t sample_categorical(
    std::span<const double> probabilities, std::mt19937_64& generator);

// Accepts `drafted` with probability min(1, target/draft). Consumes no draw
// when the outcome is forced, which includes every greedy case.
[[nodiscard]] bool speculative_accept(std::span<const double> target,
                                      std::span<const double> draft,
                                      std::uint32_t drafted,
                                      std::mt19937_64& generator);

// Samples the normalized residual max(0, target - draft). Call only after
// speculative_accept has rejected.
[[nodiscard]] std::uint32_t sample_residual(std::span<const double> target,
                                            std::span<const double> draft,
                                            std::mt19937_64& generator);

}  // namespace strata
