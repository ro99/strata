#include "test.hpp"

#include "strata/models/glm53/glm53_runtime.hpp"
#include "strata/platform/numerics.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr std::size_t kColumns = 4096U;  // one GLM-5.3 expert gate/up row

// The reference dequantization, written the way the format is defined rather
// than the way either decoder is implemented: value = e2m1(nibble) * 2^(e-127),
// column 2b in the low nibble of byte b, one exponent per 32 columns.
[[nodiscard]] float reference_fp4_dot(
    const std::vector<std::byte>& packed, const std::vector<std::byte>& scales,
    const std::vector<float>& input) {
    double sum = 0.0;
    for (std::size_t column = 0U; column < input.size(); ++column) {
        const auto byte = std::to_integer<std::uint8_t>(packed[column / 2U]);
        const auto nibble = static_cast<std::uint8_t>(
            column % 2U == 0U ? (byte & 0x0FU) : (byte >> 4U));
        const auto code = std::to_integer<std::uint8_t>(scales[column / 32U]);
        sum += static_cast<double>(input[column]) *
               static_cast<double>(strata::fp4_e2m1_f32(nibble)) *
               std::ldexp(1.0, static_cast<int>(code) - 127);
    }
    return static_cast<float>(sum);
}

}  // namespace

TEST_CASE("GLM-5.3 MXFP4 host dot agrees with a reference dequantization") {
    std::mt19937 engine(20260830U);
    std::uniform_int_distribution<int> nibbles(0, 255);
    // E8M0 codes near the bias keep the products in a range where an f32
    // accumulation and the double reference cannot drift on magnitude alone.
    std::uniform_int_distribution<int> exponents(120, 131);
    std::uniform_real_distribution<float> activations(-2.0F, 2.0F);

    std::vector<std::byte> packed(kColumns / 2U);
    std::vector<std::byte> scales(kColumns / 32U);
    std::vector<float> input(kColumns);
    for (auto& byte : packed) byte = static_cast<std::byte>(nibbles(engine));
    for (auto& byte : scales) byte = static_cast<std::byte>(exponents(engine));
    for (auto& value : input) value = activations(engine);

    const auto expected = reference_fp4_dot(packed, scales, input);
    const auto scalar = strata::glm53_host_fp4_group32_row_dot(
        packed, scales, input, false);
    const auto vectorized = strata::glm53_host_fp4_group32_row_dot(
        packed, scales, input, true);
    const auto tolerance = 1.0e-3F * std::max(1.0F, std::abs(expected));
    REQUIRE(std::abs(scalar - expected) <= tolerance);
    REQUIRE(std::abs(vectorized - expected) <= tolerance);
}

TEST_CASE("GLM-5.3 MXFP4 host dot decodes every E2M1 code in column order") {
    // One group of 32: the low nibble of byte b must be column 2b. A decoder
    // that swapped the nibbles would pass a random-input test on magnitude and
    // fail this one, which is the failure the checkpoint audit ruled out.
    std::vector<std::byte> packed(16U);
    std::vector<std::byte> scales(1U, static_cast<std::byte>(127U));
    std::vector<float> input(32U, 0.0F);
    for (std::size_t byte = 0U; byte < packed.size(); ++byte) {
        // Column 2b gets code b, column 2b+1 gets code 15-b.
        packed[byte] = static_cast<std::byte>(
            (byte & 0x0FU) | ((15U - byte) << 4U));
    }
    for (std::size_t column = 0U; column < input.size(); ++column) {
        input[column] = 0.0F;
        std::fill(input.begin(), input.end(), 0.0F);
        input[column] = 1.0F;
        const auto expected = strata::fp4_e2m1_f32(static_cast<std::uint8_t>(
            column % 2U == 0U ? column / 2U : 15U - column / 2U));
        REQUIRE(strata::glm53_host_fp4_group32_row_dot(
                    packed, scales, input, false) == expected);
        REQUIRE(strata::glm53_host_fp4_group32_row_dot(
                    packed, scales, input, true) == expected);
    }
}

TEST_CASE("GLM-5.3 MXFP4 host dot applies one E8M0 exponent per 32 columns") {
    std::vector<std::byte> packed(kColumns / 2U,
                                  static_cast<std::byte>(0x22U));  // 1.0, 1.0
    std::vector<std::byte> scales(kColumns / 32U);
    std::vector<float> input(kColumns, 1.0F);
    for (std::size_t group = 0U; group < scales.size(); ++group) {
        // Alternate 2^0 and 2^1 so a decoder reading the wrong group is off by
        // a factor the sum cannot hide.
        scales[group] = static_cast<std::byte>(group % 2U == 0U ? 127U : 128U);
    }
    const auto expected = static_cast<float>(kColumns / 2U) * 1.0F +
                          static_cast<float>(kColumns / 2U) * 2.0F;
    REQUIRE(strata::glm53_host_fp4_group32_row_dot(
                packed, scales, input, false) == expected);
    REQUIRE(strata::glm53_host_fp4_group32_row_dot(
                packed, scales, input, true) == expected);
}

TEST_CASE("GLM-5.3 BF16 host dot matches an exact reference") {
    std::mt19937 engine(20260831U);
    std::uniform_real_distribution<float> weights(-1.0F, 1.0F);
    std::uniform_real_distribution<float> activations(-2.0F, 2.0F);
    std::vector<std::byte> encoded(kColumns * 2U);
    std::vector<float> input(kColumns);
    double expected = 0.0;
    for (std::size_t column = 0U; column < kColumns; ++column) {
        const auto weight = strata::bf16_round_f32(weights(engine));
        const auto bits = static_cast<std::uint16_t>(
            std::bit_cast<std::uint32_t>(weight) >> 16U);
        std::memcpy(encoded.data() + column * 2U, &bits, sizeof(bits));
        input[column] = activations(engine);
        expected += static_cast<double>(weight) *
                    static_cast<double>(input[column]);
    }
    const auto reference = static_cast<float>(expected);
    const auto tolerance = 1.0e-3F * std::max(1.0F, std::abs(reference));
    REQUIRE(std::abs(strata::glm53_host_bf16_row_dot(encoded, input, false) -
                     reference) <= tolerance);
    REQUIRE(std::abs(strata::glm53_host_bf16_row_dot(encoded, input, true) -
                     reference) <= tolerance);
}
