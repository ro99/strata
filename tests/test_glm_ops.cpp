#include "test.hpp"

#include "strata/glm_ops.hpp"

#include <array>
#include <cmath>
#include <numeric>

namespace {

strata::RouterSpec router_spec(std::uint32_t experts, std::uint32_t topk) {
    auto spec = strata::quanttrio_glm52_int4_int8_mix_spec().router;
    spec.routed_experts = experts;
    spec.experts_per_token = topk;
    return spec;
}

}  // namespace

TEST_CASE("GLM RMSNorm uses the pinned affine RMS definition") {
    const std::array<float, 3> input{1.0F, 2.0F, 3.0F};
    const std::array<float, 3> weight{1.0F, 0.5F, 2.0F};
    std::array<float, 3> output{};
    REQUIRE(strata::rms_norm_f32(
        output, input, weight, strata::kGlm52RmsNormEpsilon).ok());
    const float reciprocal = 1.0F / std::sqrt(14.0F / 3.0F + 1.0e-5F);
    REQUIRE(std::abs(output[0] - reciprocal) < 1.0e-6F);
    REQUIRE(std::abs(output[1] - reciprocal) < 1.0e-6F);
    REQUIRE(std::abs(output[2] - 6.0F * reciprocal) < 1.0e-6F);
}

TEST_CASE("GLM indexer LayerNorm applies learned bias") {
    std::array<float, 4> values{1.0F, 2.0F, 3.0F, 4.0F};
    const std::array<float, 4> weight{1.0F, 1.0F, 1.0F, 1.0F};
    const std::array<float, 4> bias{0.25F, 0.25F, 0.25F, 0.25F};
    REQUIRE(strata::glm_layer_norm_f32(values, weight, bias).ok());
    const float sum = std::accumulate(values.begin(), values.end(), 0.0F);
    REQUIRE(std::abs(sum - 1.0F) < 1.0e-5F);
}

TEST_CASE("GLM interleaved RoPE writes split real and imaginary halves") {
    std::array<float, 6> values{1.0F, 2.0F, 3.0F, 4.0F, 91.0F, 92.0F};
    REQUIRE(strata::glm_rope_interleaved_f32(values, 0U, 4U).ok());
    REQUIRE(values[0] == 1.0F);
    REQUIRE(values[1] == 3.0F);
    REQUIRE(values[2] == 2.0F);
    REQUIRE(values[3] == 4.0F);
    REQUIRE(values[4] == 91.0F);
    REQUIRE(values[5] == 92.0F);
}

TEST_CASE("GLM router correction bias changes selection but not coefficient source") {
    const std::array<float, 4> logits{4.0F, 3.0F, 2.0F, 1.0F};
    const std::array<float, 4> bias{0.0F, 0.0F, 0.0F, 2.0F};
    const auto result = strata::glm_route_logits_noaux_tc(logits, bias, router_spec(4U, 2U));
    REQUIRE(result.ok());
    REQUIRE(result.value.experts[0] == 3U);
    REQUIRE(result.value.experts[1] == 0U);
    const float first = strata::sigmoid_f32(logits[3]);
    const float second = strata::sigmoid_f32(logits[0]);
    REQUIRE(std::abs(result.value.weights[0] - first / (first + second) * 2.5F) < 1.0e-6F);
    REQUIRE(std::abs(result.value.weights[1] - second / (first + second) * 2.5F) < 1.0e-6F);
}

TEST_CASE("GLM router top-k resolves equal scores by lower expert index") {
    const std::array<float, 4> logits{};
    const std::array<float, 4> bias{};
    const auto result = strata::glm_route_logits_noaux_tc(logits, bias, router_spec(4U, 3U));
    REQUIRE(result.ok());
    REQUIRE(result.value.experts[0] == 0U);
    REQUIRE(result.value.experts[1] == 1U);
    REQUIRE(result.value.experts[2] == 2U);
    const float sum = std::accumulate(result.value.weights.begin(),
                                      result.value.weights.end(), 0.0F);
    REQUIRE(std::abs(sum - 2.5F) < 1.0e-6F);
}

TEST_CASE("GLM reference router projection is row-major") {
    const std::array<float, 2> hidden{2.0F, 3.0F};
    const std::array<float, 6> weights{1.0F, 0.0F, 0.0F, 1.0F, -1.0F, -1.0F};
    const std::array<float, 3> bias{};
    const auto result = strata::glm_route_noaux_tc_f32(hidden, weights, bias,
                                                        router_spec(3U, 1U));
    REQUIRE(result.ok());
    REQUIRE(result.value.experts[0] == 1U);
    REQUIRE(std::abs(result.value.weights[0] - 2.5F) < 1.0e-6F);
}
