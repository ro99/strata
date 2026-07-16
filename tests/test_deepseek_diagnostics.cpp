#include "test.hpp"

#include "strata/deepseek_diagnostics.hpp"

#include <array>
#include <limits>

TEST_CASE("DeepSeek diagnostic top-K ordering and summaries are deterministic") {
    constexpr std::array<float, 7> logits{
        1.0F, 3.0F, 3.0F, -2.0F, 2.0F,
        std::numeric_limits<float>::quiet_NaN(), 0.5F};
    const auto first = strata::analyze_dsv4_logits(logits, 4U);
    const auto second = strata::analyze_dsv4_logits(logits, 4U);

    REQUIRE(first.top.size() == 4U);
    REQUIRE(first.top[0].token_id == 1U);
    REQUIRE(first.top[1].token_id == 2U);
    REQUIRE(first.top[2].token_id == 4U);
    REQUIRE(first.top[3].token_id == 0U);
    REQUIRE(first.summary.value_count == logits.size());
    REQUIRE(first.summary.finite_count == 6U);
    REQUIRE(first.summary.non_finite_count == 1U);
    REQUIRE(first.summary.sum == 7.5);
    REQUIRE(first.summary.absolute_sum == 11.5);
    REQUIRE(first.summary.square_sum == 27.25);
    REQUIRE(first.summary.minimum == -2.0F);
    REQUIRE(first.summary.maximum == 3.0F);
    REQUIRE(first.summary.raw_f32_hash == second.summary.raw_f32_hash);
    REQUIRE(first.top[0].raw_logit == second.top[0].raw_logit);
}

TEST_CASE("DeepSeek BF16 hidden-state hash observes the declared rounding boundary") {
    constexpr std::array<float, 3> reference{1.0F, -2.0F, 0.5F};
    constexpr std::array<float, 3> same_bf16{1.0001F, -2.0001F, 0.5001F};
    constexpr std::array<float, 3> different_bf16{1.01F, -2.0F, 0.5F};

    const auto reference_hash = strata::dsv4_stable_bf16_hash(reference);
    REQUIRE(reference_hash == 0x5A24'B89C'D9CF'A12DULL);
    REQUIRE(reference_hash == strata::dsv4_stable_bf16_hash(reference));
    REQUIRE(reference_hash == strata::dsv4_stable_bf16_hash(same_bf16));
    REQUIRE(reference_hash != strata::dsv4_stable_bf16_hash(different_bf16));
}
