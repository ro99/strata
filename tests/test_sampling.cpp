#include "test.hpp"

#include "strata/sampling.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

TEST_CASE("zero-temperature sampling is greedy") {
    constexpr std::array<float, 4> logits{-3.0F, 2.0F, 9.0F, 1.0F};
    std::mt19937_64 generator(7U);
    REQUIRE(strata::sample_logits_gumbel(logits, 0.0, generator) == 2U);
}

TEST_CASE("seeded Gumbel-max sampling is deterministic and non-degenerate") {
    constexpr std::array<float, 4> logits{};
    std::mt19937_64 first(33'377'335U);
    std::mt19937_64 second(33'377'335U);
    std::set<std::uint32_t> observed;
    for (std::size_t iteration = 0U; iteration < 32U; ++iteration) {
        const auto first_token = strata::sample_logits_gumbel(logits, 1.0, first);
        const auto second_token = strata::sample_logits_gumbel(logits, 1.0, second);
        REQUIRE(first_token == second_token);
        REQUIRE(first_token < logits.size());
        observed.insert(first_token);
    }
    REQUIRE(observed.size() > 1U);
}

TEST_CASE("softmax degenerates to a one-hot argmax at zero temperature") {
    constexpr std::array<float, 4> logits{-3.0F, 2.0F, 9.0F, 1.0F};
    std::array<double, 4> probabilities{};
    strata::softmax_probabilities(logits, 0.0, probabilities);
    REQUIRE(probabilities[2] == 1.0);
    REQUIRE(probabilities[0] == 0.0);
    REQUIRE(probabilities[1] == 0.0);
    REQUIRE(probabilities[3] == 0.0);

    strata::softmax_probabilities(logits, 1.0, probabilities);
    double total = 0.0;
    for (const double probability : probabilities) {
        REQUIRE(probability > 0.0);
        total += probability;
    }
    REQUIRE_NEAR(total, 1.0, 1e-12);
    REQUIRE(probabilities[2] > probabilities[1]);
    REQUIRE(probabilities[1] > probabilities[3]);
}

TEST_CASE("greedy speculative sampling consumes no randomness") {
    constexpr std::array<float, 4> target_logits{-3.0F, 2.0F, 9.0F, 1.0F};
    constexpr std::array<float, 4> draft_logits{-3.0F, 8.0F, 1.0F, 1.0F};
    std::array<double, 4> target{};
    std::array<double, 4> draft{};
    strata::softmax_probabilities(target_logits, 0.0, target);
    strata::softmax_probabilities(draft_logits, 0.0, draft);

    std::mt19937_64 generator(33'377'335U);
    std::mt19937_64 untouched(33'377'335U);

    // The draft proposes 1; the target's argmax is 2, so it must be rejected
    // and the residual must return the target argmax.
    REQUIRE(!strata::speculative_accept(target, draft, 1U, generator));
    REQUIRE(strata::sample_residual(target, draft, generator) == 2U);
    // Proposing the target argmax must be accepted.
    REQUIRE(strata::speculative_accept(target, draft, 2U, generator));
    REQUIRE(generator == untouched);
}

TEST_CASE("an identical draft is always accepted and a disjoint one never is") {
    constexpr std::array<float, 5> logits{0.5F, -1.0F, 2.0F, 0.0F, 1.5F};
    std::array<double, 5> target{};
    strata::softmax_probabilities(logits, 1.0, target);
    std::mt19937_64 generator(11U);
    for (std::uint32_t token = 0U; token < target.size(); ++token) {
        REQUIRE(strata::speculative_accept(target, target, token, generator));
    }

    // A draft supported only where the target is not: every proposal is
    // rejected and the residual is the target itself.
    const std::array<double, 5> shifted{0.0, 1.0, 0.0, 0.0, 0.0};
    std::array<double, 5> masked = target;
    masked[1] = 0.0;
    double total = 0.0;
    for (const double mass : masked) total += mass;
    for (double& mass : masked) mass /= total;
    REQUIRE(!strata::speculative_accept(masked, shifted, 1U, generator));
    const auto residual = strata::sample_residual(masked, shifted, generator);
    REQUIRE(residual != 1U);
}

TEST_CASE("speculative sampling reproduces the target distribution exactly") {
    // The load-bearing invariant: drafting from q, accepting with probability
    // min(1, p/q) and otherwise resampling max(0, p - q) is distributed as p,
    // no matter how poor q is.
    constexpr std::array<float, 6> target_logits{1.5F, -0.5F, 0.25F, 2.0F, -1.5F, 0.75F};
    constexpr std::array<float, 6> draft_logits{-2.0F, 3.0F, 0.5F, -1.0F, 1.25F, 0.0F};
    std::array<double, 6> target{};
    std::array<double, 6> draft{};
    strata::softmax_probabilities(target_logits, 1.0, target);
    strata::softmax_probabilities(draft_logits, 1.0, draft);

    constexpr std::size_t draws = 1'000'000U;
    std::vector<std::uint64_t> histogram(target.size(), 0U);
    std::mt19937_64 generator(20'260'724U);
    std::uint64_t accepted = 0U;
    for (std::size_t draw = 0U; draw < draws; ++draw) {
        const auto drafted = strata::sample_categorical(draft, generator);
        if (strata::speculative_accept(target, draft, drafted, generator)) {
            ++histogram[drafted];
            ++accepted;
        } else {
            ++histogram[strata::sample_residual(target, draft, generator)];
        }
    }

    REQUIRE(accepted > 0U);
    REQUIRE(accepted < draws);
    for (std::size_t token = 0U; token < target.size(); ++token) {
        const double expected = target[token];
        const double observed =
            static_cast<double>(histogram[token]) / static_cast<double>(draws);
        const double standard_error =
            std::sqrt(expected * (1.0 - expected) / static_cast<double>(draws));
        REQUIRE(std::fabs(observed - expected) <= 4.0 * standard_error);
    }
}

TEST_CASE("categorical sampling matches its own distribution") {
    const std::array<double, 3> probabilities{0.2, 0.5, 0.3};
    constexpr std::size_t draws = 500'000U;
    std::vector<std::uint64_t> histogram(probabilities.size(), 0U);
    std::mt19937_64 generator(5U);
    for (std::size_t draw = 0U; draw < draws; ++draw) {
        ++histogram[strata::sample_categorical(probabilities, generator)];
    }
    for (std::size_t token = 0U; token < probabilities.size(); ++token) {
        const double expected = probabilities[token];
        const double observed =
            static_cast<double>(histogram[token]) / static_cast<double>(draws);
        const double standard_error =
            std::sqrt(expected * (1.0 - expected) / static_cast<double>(draws));
        REQUIRE(std::fabs(observed - expected) <= 4.0 * standard_error);
    }
}
