#include "strata/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace strata {

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

}  // namespace strata
