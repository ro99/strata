#include "strata/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <vector>

namespace strata {

TokenLogprob sample_logits(
    std::span<const float> logits, const SamplingOptions& options,
    std::span<const std::uint32_t> token_counts,
    std::mt19937_64& generator) {
    TokenLogprob result;
    if (logits.empty()) return result;

    const bool plain = options.presence_penalty == 0.0 &&
        options.frequency_penalty == 0.0 && options.logit_bias.empty();
    if (plain && options.top_p >= 1.0 && options.top_logprobs == 0U &&
        !options.return_logprobs) {
        if (options.temperature <= 0.0) {
            result.token = static_cast<std::uint32_t>(std::distance(
                logits.begin(), std::max_element(logits.begin(), logits.end())));
            return result;
        }
        const long double denominator =
            static_cast<long double>(std::numeric_limits<std::uint64_t>::max()) + 1.0L;
        double best_score = -std::numeric_limits<double>::infinity();
        for (std::size_t token = 0U; token < logits.size(); ++token) {
            const long double uniform =
                (static_cast<long double>(generator()) + 0.5L) / denominator;
            const double gumbel = -std::log(-std::log(static_cast<double>(uniform)));
            const double score = static_cast<double>(logits[token]) /
                options.temperature + gumbel;
            if (score > best_score) {
                best_score = score;
                result.token = static_cast<std::uint32_t>(token);
            }
        }
        return result;
    }

    std::vector<double> scores(logits.begin(), logits.end());
    for (std::size_t token = 0U;
         token < scores.size() && token < token_counts.size(); ++token) {
        const auto count = token_counts[token];
        if (count != 0U) scores[token] -= options.presence_penalty;
        scores[token] -= options.frequency_penalty * static_cast<double>(count);
    }
    for (const auto& [token, bias] : options.logit_bias) {
        if (token < scores.size()) scores[token] += bias;
    }

    if (options.top_p >= 1.0 && options.top_logprobs == 0U &&
        !options.return_logprobs) {
        if (options.temperature <= 0.0) {
            result.token = static_cast<std::uint32_t>(std::distance(
                scores.begin(), std::max_element(scores.begin(), scores.end())));
            return result;
        }
        const long double denominator =
            static_cast<long double>(std::numeric_limits<std::uint64_t>::max()) + 1.0L;
        double best_score = -std::numeric_limits<double>::infinity();
        for (std::size_t token = 0U; token < scores.size(); ++token) {
            const long double uniform =
                (static_cast<long double>(generator()) + 0.5L) / denominator;
            const double gumbel = -std::log(-std::log(static_cast<double>(uniform)));
            const double score = scores[token] / options.temperature + gumbel;
            if (score > best_score) {
                best_score = score;
                result.token = static_cast<std::uint32_t>(token);
            }
        }
        return result;
    }

    std::vector<std::uint32_t> order(scores.size());
    std::iota(order.begin(), order.end(), 0U);
    std::sort(order.begin(), order.end(), [&scores](auto left, auto right) {
        return scores[left] > scores[right];
    });
    if (options.temperature <= 0.0) {
        result.token = order.front();
        result.logprob = 0.0;
        if (options.top_logprobs != 0U) result.top.emplace_back(result.token, 0.0);
        return result;
    }

    const double maximum = scores[order.front()] / options.temperature;
    std::vector<double> weights(scores.size());
    double total = 0.0;
    for (std::size_t token = 0U; token < scores.size(); ++token) {
        weights[token] = std::exp(scores[token] / options.temperature - maximum);
        total += weights[token];
    }
    std::size_t allowed = order.size();
    if (options.top_p < 1.0) {
        double cumulative = 0.0;
        allowed = 0U;
        do {
            cumulative += weights[order[allowed++]] / total;
        } while (allowed < order.size() && cumulative < options.top_p);
    }

    const long double denominator =
        static_cast<long double>(std::numeric_limits<std::uint64_t>::max()) + 1.0L;
    double best_score = -std::numeric_limits<double>::infinity();
    for (std::size_t index = 0U; index < allowed; ++index) {
        const auto token = order[index];
        const long double uniform =
            (static_cast<long double>(generator()) + 0.5L) / denominator;
        const double gumbel = -std::log(-std::log(static_cast<double>(uniform)));
        const double score = scores[token] / options.temperature + gumbel;
        if (score > best_score) {
            best_score = score;
            result.token = token;
        }
    }
    result.logprob = std::log(weights[result.token] / total);
    const auto top = std::min<std::size_t>(options.top_logprobs, order.size());
    for (std::size_t index = 0U; index < top; ++index) {
        const auto token = order[index];
        result.top.emplace_back(token, std::log(weights[token] / total));
    }
    return result;
}

std::uint32_t sample_logits_gumbel(
    std::span<const float> logits, double temperature,
    std::mt19937_64& generator) {
    SamplingOptions options;
    options.temperature = temperature;
    return sample_logits(logits, options, {}, generator).token;
}

}  // namespace strata
