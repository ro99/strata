#include "strata/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace strata {
namespace {

constexpr long double kUniformDenominator =
    static_cast<long double>(std::numeric_limits<std::uint64_t>::max()) + 1.0L;

[[nodiscard]] std::uint32_t argmax(std::span<const float> logits) noexcept {
    return static_cast<std::uint32_t>(
        std::distance(logits.begin(),
                      std::max_element(logits.begin(), logits.end())));
}

}  // namespace

std::uint32_t sample_logits_gumbel(
    std::span<const float> logits, double temperature,
    std::mt19937_64& generator) {
    if (logits.empty()) return 0U;
    if (temperature <= 0.0) {
        return static_cast<std::uint32_t>(
            std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    }

    const long double denominator =
        static_cast<long double>(std::numeric_limits<std::uint64_t>::max()) + 1.0L;
    double best_score = -std::numeric_limits<double>::infinity();
    std::uint32_t best = 0U;
    for (std::size_t token = 0U; token < logits.size(); ++token) {
        const long double uniform =
            (static_cast<long double>(generator()) + 0.5L) / denominator;
        const double gumbel = -std::log(-std::log(static_cast<double>(uniform)));
        const double score = static_cast<double>(logits[token]) / temperature + gumbel;
        if (score > best_score) {
            best_score = score;
            best = static_cast<std::uint32_t>(token);
        }
    }
    return best;
}

void softmax_probabilities(std::span<const float> logits, double temperature,
                           std::span<double> probabilities) {
    if (logits.empty() || probabilities.size() != logits.size()) return;
    std::fill(probabilities.begin(), probabilities.end(), 0.0);
    if (temperature <= 0.0) {
        probabilities[argmax(logits)] = 1.0;
        return;
    }
    double maximum = -std::numeric_limits<double>::infinity();
    for (const float logit : logits) {
        maximum = std::max(maximum, static_cast<double>(logit) / temperature);
    }
    if (!std::isfinite(maximum)) {
        probabilities[argmax(logits)] = 1.0;
        return;
    }
    double total = 0.0;
    for (std::size_t token = 0U; token < logits.size(); ++token) {
        const double weight =
            std::exp(static_cast<double>(logits[token]) / temperature - maximum);
        probabilities[token] = weight;
        total += weight;
    }
    if (!(total > 0.0)) {
        std::fill(probabilities.begin(), probabilities.end(), 0.0);
        probabilities[argmax(logits)] = 1.0;
        return;
    }
    for (double& probability : probabilities) probability /= total;
}

double next_uniform(std::mt19937_64& generator) {
    return static_cast<double>(
        (static_cast<long double>(generator()) + 0.5L) / kUniformDenominator);
}

std::uint32_t sample_categorical(std::span<const double> probabilities,
                                 std::mt19937_64& generator) {
    if (probabilities.empty()) return 0U;
    // A point mass carries no information to sample, so drawing would only
    // desynchronize the generator against the greedy path.
    for (std::size_t token = 0U; token < probabilities.size(); ++token) {
        if (probabilities[token] >= 1.0) return static_cast<std::uint32_t>(token);
    }
    const double uniform = next_uniform(generator);
    double cumulative = 0.0;
    std::uint32_t last_positive = 0U;
    for (std::size_t token = 0U; token < probabilities.size(); ++token) {
        if (probabilities[token] > 0.0) last_positive = static_cast<std::uint32_t>(token);
        cumulative += probabilities[token];
        if (uniform < cumulative) return static_cast<std::uint32_t>(token);
    }
    return last_positive;
}

bool speculative_accept(std::span<const double> target,
                        std::span<const double> draft, std::uint32_t drafted,
                        std::mt19937_64& generator) {
    if (target.size() != draft.size() || drafted >= target.size()) return false;
    const double p = target[drafted];
    const double q = draft[drafted];
    // Both branches are forced, so neither draws. Greedy decoding always lands
    // here: q is one at the drafted token, and p is one exactly when the draft
    // agrees with the target argmax.
    if (!(q > 0.0) || p >= q) return true;
    if (!(p > 0.0)) return false;
    return next_uniform(generator) < p / q;
}

std::uint32_t sample_residual(std::span<const double> target,
                              std::span<const double> draft,
                              std::mt19937_64& generator) {
    if (target.empty() || target.size() != draft.size()) return 0U;
    std::vector<double> residual(target.size(), 0.0);
    double total = 0.0;
    for (std::size_t token = 0U; token < target.size(); ++token) {
        const double mass = target[token] - draft[token];
        if (mass <= 0.0) continue;
        residual[token] = mass;
        total += mass;
    }
    if (!(total > 0.0)) {
        // The residual vanished to rounding. Falling back to the target's mode
        // keeps the caller on a token the target itself assigns mass to.
        return static_cast<std::uint32_t>(std::distance(
            target.begin(), std::max_element(target.begin(), target.end())));
    }
    for (double& mass : residual) mass /= total;
    return sample_categorical(residual, generator);
}

}  // namespace strata
