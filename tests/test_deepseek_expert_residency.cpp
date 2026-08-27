#include "test.hpp"

#include "strata/models/deepseek/deepseek_expert_residency.hpp"

#include <string>

namespace {

constexpr std::uint32_t kLayers = 43U;
constexpr std::uint32_t kExperts = 256U;

[[nodiscard]] std::string plan_text(const std::string& body,
                                    std::size_t pairs) {
    return "strata.dsv4_expert_residency 1\n"
           "layers 43 experts 256 triplet_bytes 13369344\n"
           "decode_activations 100 decode_rows 10\n"
           "pairs " + std::to_string(pairs) + " bytes 0\n" + body;
}

}  // namespace

TEST_CASE("dsv4 expert residency plan parses and answers residency") {
    const auto text = plan_text("0 5 40\n7 200 21\n42 255 3\n", 3U);
    auto parsed = strata::Dsv4ExpertResidencyPlan::parse(text, kLayers, kExperts);
    REQUIRE(parsed.ok());
    const auto& plan = parsed.value;
    REQUIRE(plan.size() == 3U);
    REQUIRE(!plan.empty());
    REQUIRE(plan.resident(0U, 5U));
    REQUIRE(plan.resident(7U, 200U));
    REQUIRE(plan.resident(42U, 255U));
    REQUIRE(!plan.resident(0U, 6U));
    REQUIRE(!plan.resident(1U, 5U));
    // Out of range must answer false rather than read past the bitmap.
    REQUIRE(!plan.resident(kLayers, 0U));
    REQUIRE(!plan.resident(0U, kExperts));
    REQUIRE(plan.bytes(13369344ULL) == 3ULL * 13369344ULL);
    // Ordered hottest first, which is the order to admit under a tight budget.
    REQUIRE(plan.pairs()[0].first == 0U && plan.pairs()[0].second == 5U);
}

TEST_CASE("dsv4 expert residency plan truncates to the hottest entries") {
    const auto text = plan_text("0 5 40\n7 200 21\n42 255 3\n", 3U);
    auto parsed = strata::Dsv4ExpertResidencyPlan::parse(text, kLayers, kExperts);
    REQUIRE(parsed.ok());
    auto plan = std::move(parsed.value);
    REQUIRE(plan.truncate(2U) == 2U);
    REQUIRE(plan.size() == 2U);
    REQUIRE(plan.resident(0U, 5U));
    REQUIRE(plan.resident(7U, 200U));
    // The dropped entry must stop reporting resident, or the device and the
    // host would disagree about who owes that expert.
    REQUIRE(!plan.resident(42U, 255U));
    REQUIRE(plan.truncate(99U) == 2U);
}

TEST_CASE("dsv4 expert residency plan rejects a geometry mismatch") {
    const auto text =
        "strata.dsv4_expert_residency 1\n"
        "layers 40 experts 256 triplet_bytes 13369344\n"
        "pairs 1 bytes 0\n"
        "0 5 40\n";
    auto parsed = strata::Dsv4ExpertResidencyPlan::parse(text, kLayers, kExperts);
    REQUIRE(!parsed.ok());
}

TEST_CASE("dsv4 expert residency plan rejects a repeated triplet") {
    // A repeat would make one expert resident once but owed twice, so the
    // partial sums would double-count it.
    const auto text = plan_text("0 5 40\n0 5 21\n", 2U);
    auto parsed = strata::Dsv4ExpertResidencyPlan::parse(text, kLayers, kExperts);
    REQUIRE(!parsed.ok());
}

TEST_CASE("dsv4 expert residency plan rejects an out-of-range entry") {
    const auto text = plan_text("43 0 40\n", 1U);
    auto parsed = strata::Dsv4ExpertResidencyPlan::parse(text, kLayers, kExperts);
    REQUIRE(!parsed.ok());
    const auto expert = plan_text("0 256 40\n", 1U);
    auto other = strata::Dsv4ExpertResidencyPlan::parse(expert, kLayers, kExperts);
    REQUIRE(!other.ok());
}

TEST_CASE("dsv4 expert residency plan rejects a miscounted pair list") {
    const auto text = plan_text("0 5 40\n7 200 21\n", 3U);
    auto parsed = strata::Dsv4ExpertResidencyPlan::parse(text, kLayers, kExperts);
    REQUIRE(!parsed.ok());
}

TEST_CASE("dsv4 expert residency plan rejects a foreign or versioned header") {
    auto missing = strata::Dsv4ExpertResidencyPlan::parse(
        "pairs 0\n", kLayers, kExperts);
    REQUIRE(!missing.ok());
    auto future = strata::Dsv4ExpertResidencyPlan::parse(
        "strata.dsv4_expert_residency 2\nlayers 43 experts 256\npairs 0\n",
        kLayers, kExperts);
    REQUIRE(!future.ok());
}

TEST_CASE("dsv4 expert residency plan accepts an empty tier") {
    // Zero resident triplets is a legal configuration: it is exactly today's
    // behaviour, and admission must be able to fall back to it.
    const auto text = plan_text("", 0U);
    auto parsed = strata::Dsv4ExpertResidencyPlan::parse(text, kLayers, kExperts);
    REQUIRE(parsed.ok());
    REQUIRE(parsed.value.empty());
    REQUIRE(!parsed.value.resident(0U, 0U));
}
