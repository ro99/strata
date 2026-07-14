#include "test.hpp"

#include "strata/kernel_abi.h"

#include <array>
#include <cstdint>

TEST_CASE("q4 is the minimum valid quantized format") {
    REQUIRE(strata_quant_format_valid(STRATA_Q4));
    REQUIRE(!strata_quant_format_valid(3));
    REQUIRE(!strata_quant_format_valid(2));
}

TEST_CASE("q4 nibble packing round-trips the signed range") {
    std::array<std::uint8_t, 8> packed{};
    for (int value = -8; value <= 7; ++value) {
        strata_q4_pack(packed.data(), static_cast<std::size_t>(value + 8),
                       static_cast<std::int8_t>(value));
    }
    for (int value = -8; value <= 7; ++value) {
        REQUIRE(strata_q4_unpack(packed.data(), static_cast<std::size_t>(value + 8)) == value);
    }
}

TEST_CASE("q4 reference matvec handles odd row widths") {
    constexpr std::size_t rows = 2;
    constexpr std::size_t cols = 3;
    std::array<std::uint8_t, rows * ((cols + 1) / 2)> weights{};
    const std::array<std::int8_t, rows * cols> values{1, -2, 3, -4, 5, -6};
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            strata_q4_pack(weights.data() + row * ((cols + 1) / 2), col,
                           values[row * cols + col]);
        }
    }
    const std::array<float, cols> x{2.0F, 3.0F, -1.0F};
    const std::array<float, rows> scales{0.5F, 0.25F};
    std::array<float, rows> y{};
    REQUIRE(strata_q4_matvec_f32(y.data(), x.data(), weights.data(), scales.data(),
                                 rows, cols));
    REQUIRE_NEAR(y[0], -3.5F, 1.0e-6F);
    REQUIRE_NEAR(y[1], 3.25F, 1.0e-6F);
}
