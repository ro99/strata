#include "strata/engine/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <unordered_map>
#include <vector>

namespace strata {
namespace {

constexpr double kNegativeInfinity = -std::numeric_limits<double>::infinity();

// One uniform in (0,1). The 0.5 offset keeps the draw off both endpoints so
// the double logarithm below stays finite.
double draw_uniform(std::mt19937_64& generator) {
    const long double denominator =
        static_cast<long double>(std::numeric_limits<std::uint64_t>::max()) + 1.0L;
    return static_cast<double>(
        (static_cast<long double>(generator()) + 0.5L) / denominator);
}

double draw_gumbel(std::mt19937_64& generator) {
    return -std::log(-std::log(draw_uniform(generator)));
}

// Gumbel-max over a token range: argmax of score/T + Gumbel noise is an exact
// draw from softmax(score/T). One RNG draw per candidate, in ascending token
// order, so a seed reproduces a run given the same candidate set.
template <typename Range>
std::uint32_t gumbel_max(const Range& candidates, const std::vector<double>& scores,
                         double temperature, std::mt19937_64& generator) {
    std::uint32_t chosen = 0U;
    double best = kNegativeInfinity;
    bool seen = false;
    for (const auto token : candidates) {
        const double score = scores[token] / temperature + draw_gumbel(generator);
        if (!seen || score > best) {
            best = score;
            chosen = static_cast<std::uint32_t>(token);
            seen = true;
        }
    }
    return chosen;
}

double log_sum_exp(const std::vector<double>& scores, double maximum) {
    if (maximum == kNegativeInfinity) return kNegativeInfinity;
    double total = 0.0;
    for (const auto score : scores) total += std::exp(score - maximum);
    return maximum + std::log(total);
}

// The `count` highest-scoring token ids, descending. Linear selection plus a
// sort of the prefix only, so this stays O(V) for the small prefixes that
// truncation actually keeps.
void select_top(const std::vector<double>& scores, std::size_t count,
                std::vector<std::uint32_t>& out) {
    const auto better = [&scores](std::uint32_t left, std::uint32_t right) {
        if (scores[left] != scores[right]) return scores[left] > scores[right];
        return left < right;
    };
    out.resize(scores.size());
    std::iota(out.begin(), out.end(), 0U);
    if (count < out.size()) {
        std::nth_element(out.begin(),
                         out.begin() + static_cast<std::ptrdiff_t>(count),
                         out.end(), better);
        out.resize(count);
    }
    std::sort(out.begin(), out.end(), better);
}

// Smallest descending-probability prefix carrying at least `mass`. The prefix
// length is unknown up front, so grow it geometrically and fall back to a full
// ordering rather than guess: for a typical top_p the first step suffices, and
// a full sort of a 129k vocabulary costs more than four linear passes.
void select_nucleus(const std::vector<double>& scores, double log_total,
                    double mass, std::vector<std::uint32_t>& out) {
    const std::size_t limit = scores.size();
    for (std::size_t width = 128U; ; width *= 8U) {
        const bool exhaustive = width >= limit;
        select_top(scores, exhaustive ? limit : width, out);
        double cumulative = 0.0;
        std::size_t kept = 0U;
        while (kept < out.size() && cumulative < mass) {
            cumulative += std::exp(scores[out[kept]] - log_total);
            ++kept;
        }
        if (cumulative >= mass || exhaustive) {
            out.resize(kept);
            return;
        }
    }
}

// Windowed occurrence counts for the subtractive and multiplicative penalties.
// The generated sequence is the source of truth; the runtimes' whole-history
// histogram is only an precomputed shortcut for the unwindowed case.
std::unordered_map<std::uint32_t, std::uint32_t> window_counts(
    std::span<const std::uint32_t> tokens, std::uint32_t window) {
    std::unordered_map<std::uint32_t, std::uint32_t> counts;
    const std::size_t span = (window == 0U || window >= tokens.size())
        ? tokens.size()
        : static_cast<std::size_t>(window);
    for (std::size_t index = tokens.size() - span; index < tokens.size(); ++index) {
        ++counts[tokens[index]];
    }
    return counts;
}

void apply_penalty(double& score, std::uint32_t count,
                   const SamplingOptions& options) {
    if (count != 0U) score -= options.presence_penalty;
    score -= options.frequency_penalty * static_cast<double>(count);
    if (count != 0U && options.repetition_penalty != 1.0) {
        score = score > 0.0 ? score / options.repetition_penalty
                            : score * options.repetition_penalty;
    }
}

// DRY: find the longest suffix of the sequence that recurs earlier, and
// penalize whichever token continued the earlier occurrence. A longer repeated
// run means a larger penalty, so the model is pushed off a loop it has already
// committed to rather than off every repeated word.
void apply_dry(std::vector<double>& scores, std::span<const std::uint32_t> tokens,
               const SamplingOptions& options) {
    if (tokens.empty()) return;
    const std::size_t span = (options.dry_window == 0U ||
                              options.dry_window >= tokens.size())
        ? tokens.size()
        : static_cast<std::size_t>(options.dry_window);
    const auto window = tokens.subspan(tokens.size() - span);
    if (window.size() < 2U) return;

    // A token can continue several earlier occurrences of the same suffix; it
    // is penalized once, by the longest of them, not once per match.
    std::unordered_map<std::uint32_t, std::size_t> longest;
    const std::size_t last = window.size() - 1U;
    for (std::size_t index = 0U; index < last; ++index) {
        if (window[index] != window[last]) continue;
        std::size_t match = 1U;
        while (match <= index && match < last &&
               window[index - match] == window[last - match]) {
            ++match;
        }
        if (match < options.dry_allowed_length) continue;
        const auto continuation = window[index + 1U];
        if (continuation >= scores.size()) continue;
        auto& best = longest[continuation];
        best = std::max(best, match);
    }
    for (const auto& [token, match] : longest) {
        scores[token] -= options.dry_multiplier *
            std::pow(options.dry_base,
                     static_cast<double>(match - options.dry_allowed_length));
    }
}

// Hard ban on any token that would complete an n-gram already present in the
// sequence. Unlike DRY this is not a nudge: the token is removed outright.
void apply_ngram_ban(std::vector<double>& scores,
                     std::span<const std::uint32_t> tokens, std::uint32_t size) {
    if (size == 0U || tokens.size() + 1U < size) return;
    const std::size_t prefix = static_cast<std::size_t>(size) - 1U;
    for (std::size_t index = prefix; index < tokens.size(); ++index) {
        if (!std::equal(tokens.end() - static_cast<std::ptrdiff_t>(prefix),
                        tokens.end(),
                        tokens.begin() + static_cast<std::ptrdiff_t>(index - prefix))) {
            continue;
        }
        if (tokens[index] < scores.size()) scores[tokens[index]] = kNegativeInfinity;
    }
}

// Floor under H_hat before the logarithm. A candidate whose future is entirely
// decided scores zero on the entropy factor, and ln 0 would erase it from the
// draw outright rather than merely rank it last.
constexpr double kEntropyFloor = 1e-6;

// Natural log probabilities, from the unmodified logits: the model's own view,
// independent of whatever penalties and truncation the caller asked for.
void record_natural_logprobs(std::span<const float> logits, std::uint32_t chosen,
                             std::uint32_t top_count, TokenLogprob& result) {
    double maximum = kNegativeInfinity;
    for (const auto logit : logits) maximum = std::max(maximum, static_cast<double>(logit));
    double total = 0.0;
    for (const auto logit : logits) total += std::exp(static_cast<double>(logit) - maximum);
    const double log_total = maximum + std::log(total);
    result.logprob = static_cast<double>(logits[chosen]) - log_total;
    if (top_count == 0U) return;

    std::vector<double> widened(logits.begin(), logits.end());
    std::vector<std::uint32_t> order;
    select_top(widened, std::min<std::size_t>(top_count, widened.size()), order);
    for (const auto token : order) {
        result.top.emplace_back(token, widened[token] - log_total);
    }
}

}  // namespace

double normalized_top_n_entropy(std::span<const float> logits,
                                std::uint32_t top_n) {
    const auto width = std::min<std::size_t>(top_n, logits.size());
    // One option is no choice at all, and ln 1 = 0 leaves the ratio undefined.
    if (width < 2U) return 0.0;
    std::vector<float> values(logits.begin(), logits.end());
    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(width),
                     values.end(), std::greater<float>());
    values.resize(width);
    const double maximum = static_cast<double>(
        *std::max_element(values.begin(), values.end()));
    // Every candidate banned: no future to be open about.
    if (!std::isfinite(maximum)) return 0.0;
    double total = 0.0;
    for (const auto value : values) {
        total += std::exp(static_cast<double>(value) - maximum);
    }
    double entropy = 0.0;
    for (const auto value : values) {
        const double probability =
            std::exp(static_cast<double>(value) - maximum) / total;
        if (probability > 0.0) entropy -= probability * std::log(probability);
    }
    return std::clamp(entropy / std::log(static_cast<double>(width)), 0.0, 1.0);
}

double future_entropy_alpha_at(const SamplingOptions& options,
                               std::uint64_t step) {
    double alpha = options.future_entropy_alpha;
    if (options.future_entropy_wave_amplitude != 0.0 &&
        options.future_entropy_wave_period > 0.0) {
        constexpr double two_pi = 6.283185307179586;
        alpha += options.future_entropy_wave_amplitude *
                 std::sin(two_pi * static_cast<double>(step) /
                          options.future_entropy_wave_period);
    }
    return std::clamp(alpha, -1.0, 1.0);
}

TokenLogprob sample_logits(
    std::span<const float> logits, const SamplingOptions& options,
    const SamplingHistory& history, std::mt19937_64& generator,
    const FutureEntropyEvaluator& lookahead) {
    TokenLogprob result;
    if (logits.empty()) return result;

    const bool penalized = options.presence_penalty != 0.0 ||
        options.frequency_penalty != 0.0 || options.repetition_penalty != 1.0 ||
        options.dry_multiplier != 0.0 || options.no_repeat_ngram != 0U ||
        !options.logit_bias.empty();
    const bool ordered = options.top_k != 0U || options.top_p < 1.0 ||
        options.typical_p < 1.0;
    const bool future_entropy = options.future_entropy_candidates != 0U;
    const bool truncated = ordered || options.min_p > 0.0 ||
        options.xtc_probability > 0.0 || future_entropy;
    const bool reporting = options.top_logprobs != 0U || options.return_logprobs;

    // Default decode path: no stage touches the distribution, so draw straight
    // from the logits without materializing scores or an index vector.
    if (!penalized && !truncated && !reporting) {
        if (options.temperature <= 0.0) {
            result.token = static_cast<std::uint32_t>(std::distance(
                logits.begin(), std::max_element(logits.begin(), logits.end())));
            return result;
        }
        double best = kNegativeInfinity;
        for (std::size_t token = 0U; token < logits.size(); ++token) {
            const double score = static_cast<double>(logits[token]) /
                options.temperature + draw_gumbel(generator);
            if (score > best) {
                best = score;
                result.token = static_cast<std::uint32_t>(token);
            }
        }
        return result;
    }

    std::vector<double> scores(logits.begin(), logits.end());

    if (options.presence_penalty != 0.0 || options.frequency_penalty != 0.0 ||
        options.repetition_penalty != 1.0) {
        if (options.penalty_window != 0U || history.counts.empty()) {
            const auto counts = window_counts(history.tokens, options.penalty_window);
            for (const auto& [token, count] : counts) {
                if (token < scores.size()) apply_penalty(scores[token], count, options);
            }
        } else {
            for (std::size_t token = 0U;
                 token < scores.size() && token < history.counts.size(); ++token) {
                const auto count = history.counts[token];
                if (count != 0U) apply_penalty(scores[token], count, options);
            }
        }
    }
    if (options.dry_multiplier != 0.0) apply_dry(scores, history.tokens, options);
    apply_ngram_ban(scores, history.tokens, options.no_repeat_ngram);
    for (const auto& [token, bias] : options.logit_bias) {
        if (token < scores.size()) scores[token] += bias;
    }

    const double maximum = *std::max_element(scores.begin(), scores.end());
    // top_p, typical_p and XTC are all defined against probabilities; min_p and
    // the n-gram ban are pure logit comparisons and need no normalizer.
    const bool needs_total = options.top_p < 1.0 || options.typical_p < 1.0 ||
        options.xtc_probability > 0.0 || future_entropy;
    const double log_total = needs_total ? log_sum_exp(scores, maximum) : 0.0;

    std::vector<std::uint32_t> candidates;
    if (options.top_k != 0U && options.top_p >= 1.0) {
        select_top(scores, std::min<std::size_t>(options.top_k, scores.size()),
                   candidates);
    } else if (options.top_p < 1.0) {
        select_nucleus(scores, log_total, options.top_p, candidates);
        if (options.top_k != 0U && candidates.size() > options.top_k) {
            candidates.resize(options.top_k);
        }
    } else if (truncated) {
        candidates.resize(scores.size());
        std::iota(candidates.begin(), candidates.end(), 0U);
    }

    if (options.min_p > 0.0) {
        // p >= min_p * p_max is exactly score >= max_score + ln(min_p).
        const double floor_score = maximum + std::log(options.min_p);
        std::erase_if(candidates, [&scores, floor_score](std::uint32_t token) {
            return scores[token] < floor_score;
        });
    }

    if (options.typical_p < 1.0 && !candidates.empty()) {
        double entropy = 0.0;
        for (const auto score : scores) {
            if (score == kNegativeInfinity) continue;
            const double probability = std::exp(score - log_total);
            entropy -= probability * (score - log_total);
        }
        // Keep the tokens whose surprisal sits closest to the distribution's
        // own entropy, dropping both the flat-obvious and the wild tails.
        std::sort(candidates.begin(), candidates.end(),
                  [&scores, log_total, entropy](std::uint32_t left, std::uint32_t right) {
                      const double first = std::abs(scores[left] - log_total + entropy);
                      const double second = std::abs(scores[right] - log_total + entropy);
                      if (first != second) return first < second;
                      return left < right;
                  });
        double cumulative = 0.0;
        std::size_t kept = 0U;
        while (kept < candidates.size() && cumulative < options.typical_p) {
            cumulative += std::exp(scores[candidates[kept]] - log_total);
            ++kept;
        }
        candidates.resize(kept);
    }

    // XTC draws before the candidate loop so the RNG stream stays a function of
    // the options alone, not of how many candidates happened to survive.
    if (options.xtc_probability > 0.0 && candidates.size() > 1U) {
        if (draw_uniform(generator) < options.xtc_probability) {
            const double threshold_score = log_total + std::log(options.xtc_threshold);
            std::vector<std::uint32_t> above;
            for (const auto token : candidates) {
                if (scores[token] >= threshold_score) above.push_back(token);
            }
            if (above.size() > 1U) {
                const auto weakest = *std::min_element(
                    above.begin(), above.end(),
                    [&scores](std::uint32_t left, std::uint32_t right) {
                        if (scores[left] != scores[right]) return scores[left] < scores[right];
                        return left < right;
                    });
                std::erase_if(candidates, [&scores, threshold_score, weakest](
                                              std::uint32_t token) {
                    return token != weakest && scores[token] >= threshold_score;
                });
            }
        }
    }

    if (truncated && candidates.empty()) {
        // Every stage above preserves at least one token, so this only fires if
        // the n-gram ban erased the whole vocabulary. Fall back to the argmax.
        candidates.assign(1U, static_cast<std::uint32_t>(std::distance(
            scores.begin(), std::max_element(scores.begin(), scores.end()))));
    }

    // Future entropy runs last, so its forward passes are spent only on tokens
    // every cheaper stage has already accepted. With one candidate left there
    // is no ranking to change, and the lookahead is skipped rather than paid.
    if (future_entropy && candidates.size() > 1U) {
        if (!lookahead) {
            result.errors.emplace_back(
                "future-entropy sampling requires a lookahead evaluator");
            return result;
        }
        // Neither typical_p nor the unordered min_p/XTC path leaves the
        // survivors in probability order, so rank them here rather than
        // assume it: the lookahead budget must go to the likeliest tokens.
        const auto better = [&scores](std::uint32_t left, std::uint32_t right) {
            if (scores[left] != scores[right]) return scores[left] > scores[right];
            return left < right;
        };
        if (candidates.size() > options.future_entropy_candidates) {
            const auto width = static_cast<std::ptrdiff_t>(
                options.future_entropy_candidates);
            std::nth_element(candidates.begin(), candidates.begin() + width,
                             candidates.end(), better);
            candidates.resize(static_cast<std::size_t>(width));
        }
        std::sort(candidates.begin(), candidates.end(), better);

        std::vector<double> entropy(candidates.size(), 0.0);
        auto evaluated = lookahead(candidates, options.future_entropy_top_n,
                                   entropy);
        if (!evaluated.ok()) {
            result.errors = std::move(evaluated.errors);
            return result;
        }

        const double alpha = future_entropy_alpha_at(
            options, static_cast<std::uint64_t>(history.tokens.size()));
        double probability_exponent = 1.0 - std::max(0.0, alpha);
        double entropy_exponent = 1.0 - std::max(0.0, -alpha);
        if (options.future_entropy_curve == FutureEntropyCurve::Crossfade) {
            const double blend = (alpha + 1.0) / 2.0;
            probability_exponent = 1.0 - blend;
            entropy_exponent = blend;
        }

        // s(w) = p(w|c)^a * H_hat(w)^b, carried in logs so the Gumbel-max draw
        // below turns it into an exact draw from s renormalized over the
        // candidates. A zero exponent means that factor is identically 1, and
        // must not be formed at all: 0 * ln 0 is NaN where the identity needs
        // it to vanish.
        for (std::size_t index = 0U; index < candidates.size(); ++index) {
            const auto token = candidates[index];
            double blended = 0.0;
            if (probability_exponent != 0.0) {
                blended += probability_exponent * (scores[token] - log_total);
            }
            if (entropy_exponent != 0.0) {
                blended += entropy_exponent *
                    std::log(std::max(entropy[index], kEntropyFloor));
            }
            scores[token] = blended;
        }
    }

    if (options.temperature <= 0.0) {
        const auto better = [&scores](std::uint32_t left, std::uint32_t right) {
            return scores[left] < scores[right];
        };
        result.token = truncated
            ? *std::max_element(candidates.begin(), candidates.end(), better)
            : static_cast<std::uint32_t>(std::distance(
                  scores.begin(), std::max_element(scores.begin(), scores.end())));
    } else if (truncated) {
        result.token = gumbel_max(candidates, scores, options.temperature, generator);
    } else {
        std::vector<std::uint32_t> all(scores.size());
        std::iota(all.begin(), all.end(), 0U);
        result.token = gumbel_max(all, scores, options.temperature, generator);
    }

    if (reporting) {
        record_natural_logprobs(logits, result.token, options.top_logprobs, result);
    }
    return result;
}

std::uint32_t sample_logits_gumbel(
    std::span<const float> logits, double temperature,
    std::mt19937_64& generator) {
    SamplingOptions options;
    options.temperature = temperature;
    return sample_logits(logits, options, SamplingHistory{}, generator).token;
}

bool validate_sampling_options(const SamplingOptions& options, std::string& error) {
    const auto reject = [&error](const char* message) {
        error = message;
        return false;
    };
    const auto in_range = [](double value, double low, double high) {
        return std::isfinite(value) && value >= low && value <= high;
    };
    if (!in_range(options.temperature, 0.0, 10.0)) return reject("temperature");
    if (!in_range(options.top_p, 0.0, 1.0) || options.top_p <= 0.0) {
        return reject("top_p");
    }
    if (!in_range(options.min_p, 0.0, 1.0) || options.min_p >= 1.0) {
        return reject("min_p");
    }
    if (!in_range(options.typical_p, 0.0, 1.0) || options.typical_p <= 0.0) {
        return reject("typical_p");
    }
    if (!in_range(options.xtc_probability, 0.0, 1.0)) return reject("xtc_probability");
    if (!in_range(options.xtc_threshold, 0.0, 1.0) || options.xtc_threshold <= 0.0) {
        return reject("xtc_threshold");
    }
    if (!in_range(options.presence_penalty, -2.0, 2.0)) return reject("presence_penalty");
    if (!in_range(options.frequency_penalty, -2.0, 2.0)) {
        return reject("frequency_penalty");
    }
    if (!in_range(options.repetition_penalty, 0.0, 10.0) ||
        options.repetition_penalty <= 0.0) {
        return reject("repetition_penalty");
    }
    if (!in_range(options.dry_multiplier, 0.0, 10.0)) return reject("dry_multiplier");
    if (!in_range(options.dry_base, 1.0, 8.0)) return reject("dry_base");
    if (options.dry_allowed_length == 0U) return reject("dry_allowed_length");
    if (!in_range(options.future_entropy_alpha, -1.0, 1.0)) {
        return reject("future_entropy_alpha");
    }
    if (!in_range(options.future_entropy_wave_amplitude, 0.0, 2.0)) {
        return reject("future_entropy_wave_amplitude");
    }
    if (!in_range(options.future_entropy_wave_period, 0.0, 1e6) ||
        options.future_entropy_wave_period <= 0.0) {
        return reject("future_entropy_wave_period");
    }
    // Each candidate is a forward pass, so the ceiling is a cost guard rather
    // than a correctness one: 64 already makes a decode step 65 passes long.
    if (options.future_entropy_candidates > 64U) {
        return reject("future_entropy_candidates");
    }
    if (options.future_entropy_candidates == 1U) {
        return reject("future_entropy_candidates");
    }
    if (options.future_entropy_candidates != 0U &&
        options.future_entropy_top_n < 2U) {
        return reject("future_entropy_top_n");
    }
    error.clear();
    return true;
}

}  // namespace strata
