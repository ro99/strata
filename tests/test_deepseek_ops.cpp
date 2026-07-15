#include "test.hpp"

#include "strata/deepseek_ops.hpp"
#include "strata/glm_ops.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

TEST_CASE("DeepSeek native FP4 fixture matches target checkpoint bytes") {
    // First 16 packed bytes and the first E8M0 scale from
    // layers.0.ffn.experts.0.w1 in revision 62af8fff... .
    constexpr std::array<std::uint8_t, 16> encoded{
        0xacU, 0x54U, 0xa2U, 0x44U, 0xccU, 0x54U, 0x6cU, 0x55U,
        0x2aU, 0x2cU, 0xe0U, 0xecU, 0x2dU, 0xfdU, 0x85U, 0x42U};
    std::array<std::byte, 16> weights{};
    for (std::size_t index = 0U; index < encoded.size(); ++index) {
        weights[index] = static_cast<std::byte>(encoded[index]);
    }
    constexpr std::array<std::byte, 1> scales{std::byte{0x78U}};
    std::array<float, 32> input{};
    std::fill_n(input.begin(), 8U, 1.0F);
    std::array<float, 1> output{};
    const auto result = strata::dsv4_fp4_e2m1_matvec_f32(
        output, input, weights, scales, 1U, 32U);
    REQUIRE(result.ok());
    REQUIRE_NEAR(output[0], 0.046875F, 1.0e-7F);
}

TEST_CASE("DeepSeek native FP8 fixture matches target checkpoint bytes") {
    // First eight values and block scale from layers.0.attn.wq_a.
    constexpr std::array<std::uint8_t, 8> encoded{
        0xe0U, 0xf0U, 0x6dU, 0x6cU, 0x68U, 0x41U, 0x63U, 0xefU};
    std::array<std::byte, 128> weights{};
    for (std::size_t index = 0U; index < encoded.size(); ++index) {
        weights[index] = static_cast<std::byte>(encoded[index]);
    }
    constexpr std::array<std::byte, 1> scales{std::byte{0x73U}};
    std::array<float, 128> input{};
    std::fill_n(input.begin(), encoded.size(), 1.0F);
    std::array<float, 1> output{};
    const auto result = strata::dsv4_fp8_e4m3_matvec_f32(
        output, input, weights, scales, 1U, 128U);
    REQUIRE(result.ok());
    REQUIRE_NEAR(output[0], 0.00738525390625F, 1.0e-9F);
    REQUIRE_NEAR(strata::dsv4_fp8_e4m3_f32(0x7eU), 448.0F, 0.0F);
    REQUIRE(std::isnan(strata::dsv4_fp8_e4m3_f32(0x7fU)));
}

TEST_CASE("DeepSeek sqrtsoftplus routing keeps selection bias out of weights") {
    auto spec = strata::deepseek_v4_flash_dspark_spec().router;
    spec.routed_experts = 4U;
    spec.experts_per_token = 2U;
    const std::array<float, 4> logits{-2.0F, 0.0F, 1.0F, 2.0F};
    const std::array<float, 4> bias{10.0F, 0.0F, 0.0F, 0.0F};
    const auto route = strata::dsv4_route_sqrtsoftplus_f32(logits, bias, spec);
    REQUIRE(route.ok());
    const std::vector<std::uint32_t> expected{0U, 3U};
    REQUIRE(route.value.experts == expected);
    REQUIRE_NEAR(route.value.weights[0] + route.value.weights[1], 1.5F, 1.0e-6F);
    REQUIRE(route.value.weights[0] < route.value.weights[1]);
}

TEST_CASE("DeepSeek mHC Sinkhorn preserves four residual lanes") {
    constexpr std::uint32_t multiplier = 4U;
    std::array<float, 24> mixes{};
    std::array<float, 24> base{};
    constexpr std::array<float, 3> scale{1.0F, 1.0F, 1.0F};
    const auto split = strata::dsv4_mhc_split_sinkhorn_f32(
        mixes, scale, base, multiplier, 20U, 1.0e-6F);
    REQUIRE(split.ok());
    for (const float value : split.value.pre) {
        REQUIRE_NEAR(value, 0.500001F, 1.0e-7F);
    }
    for (const float value : split.value.post) REQUIRE_NEAR(value, 1.0F, 1.0e-7F);
    for (std::uint32_t row = 0U; row < multiplier; ++row) {
        float sum = 0.0F;
        for (std::uint32_t column = 0U; column < multiplier; ++column) {
            sum += split.value.combination[row * multiplier + column];
        }
        REQUIRE_NEAR(sum, 1.0F, 1.0e-5F);
    }
}

TEST_CASE("DeepSeek SwiGLU applies asymmetric target clamping") {
    constexpr std::array<float, 3> gate{-20.0F, 20.0F, 1.0F};
    constexpr std::array<float, 3> up{-20.0F, 20.0F, 2.0F};
    std::array<float, 3> output{};
    const auto result = strata::dsv4_swiglu_f32(output, gate, up, 10.0F);
    REQUIRE(result.ok());
    REQUIRE_NEAR(output[0], strata::glm_silu_f32(-20.0F) * -10.0F, 1.0e-7F);
    REQUIRE_NEAR(output[1], strata::glm_silu_f32(10.0F) * 10.0F, 1.0e-5F);
}
