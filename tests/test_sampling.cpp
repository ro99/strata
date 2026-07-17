#include "test.hpp"

#include "strata/sampling.hpp"

#include <array>
#include <random>
#include <set>

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
