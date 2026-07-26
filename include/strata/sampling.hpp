#pragma once

#include <cstdint>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace strata {

struct SamplingOptions {
    double temperature{1.0};
    double top_p{1.0};
    double presence_penalty{};
    double frequency_penalty{};
    std::uint64_t seed{33'377'335U};
    std::vector<std::pair<std::uint32_t, double>> logit_bias;
    std::uint32_t top_logprobs{};
    bool return_logprobs{};
};

struct TokenLogprob {
    std::uint32_t token{};
    double logprob{};
    std::vector<std::pair<std::uint32_t, double>> top;
};

[[nodiscard]] TokenLogprob sample_logits(
    std::span<const float> logits, const SamplingOptions& options,
    std::span<const std::uint32_t> token_counts,
    std::mt19937_64& generator);

[[nodiscard]] std::uint32_t sample_logits_gumbel(
    std::span<const float> logits, double temperature, std::mt19937_64& generator);

}  // namespace strata
