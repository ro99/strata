#include "test.hpp"

#include "strata/glm_int4.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

TEST_CASE("QuantTrio INT4 group-128 AVX2 matvec matches its target-format oracle") {
    constexpr std::uint64_t rows = 5U;
    constexpr std::uint64_t columns = 256U;
    constexpr std::uint64_t packed_columns = columns / 8U;
    constexpr std::uint64_t scale_columns = columns / 128U;
    std::vector<std::byte> packed(rows * packed_columns * 4U);
    for (std::size_t index = 0; index < packed.size(); ++index) {
        packed[index] = static_cast<std::byte>((index * 37U + 11U) & 0xFFU);
    }
    std::vector<std::byte> scales(rows * scale_columns * 2U);
    for (std::size_t index = 0; index < rows * scale_columns; ++index) {
        const float value = 0.015625F * static_cast<float>(index + 1U);
        const auto encoded = static_cast<std::uint16_t>(
            std::bit_cast<std::uint32_t>(value) >> 16U);
        std::memcpy(scales.data() + index * 2U, &encoded, sizeof(encoded));
    }
    std::vector<float> input(columns);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = std::sin(static_cast<float>(index) * 0.17F);
    }
    const strata::GlmInt4MatrixView matrix{
        packed, scales, rows, columns, packed_columns, scale_columns, 128U};
    std::vector<float> expected(rows);
    std::vector<float> actual(rows);
    REQUIRE(strata::glm_int4_group128_matvec_reference(
                matrix, input, expected).ok());
    REQUIRE(strata::glm_int4_group128_matvec(matrix, input, actual).ok());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float tolerance = 2.0e-5F * std::max(1.0F, std::abs(expected[index]));
        REQUIRE(std::abs(actual[index] - expected[index]) <= tolerance);
    }
}
