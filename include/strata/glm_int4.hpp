#pragma once

#include "strata/model.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace strata {

struct GlmInt4MatrixView {
    std::span<const std::byte> packed;
    std::span<const std::byte> scales;
    std::uint64_t rows{};
    std::uint64_t columns{};
    std::uint64_t packed_columns{};
    std::uint64_t scale_columns{};
    std::uint32_t group_size{};
};

// Scalar target-format oracle for QuantTrio offset-packed INT4 with BF16
// group scales. This format stores q + 8 in each nibble; it is deliberately
// distinct from the two's-complement Q4 ABI reference.
[[nodiscard]] ValidationResult glm_int4_group128_matvec_reference(
    const GlmInt4MatrixView& matrix, std::span<const float> input,
    std::span<float> output);

// Native CPU implementation. Uses AVX2/FMA when available and otherwise
// executes the reference path. It preserves INT4 weights and group-128 scales.
[[nodiscard]] ValidationResult glm_int4_group128_matvec(
    const GlmInt4MatrixView& matrix, std::span<const float> input,
    std::span<float> output);
[[nodiscard]] ValidationResult glm_int4_group128_matvec_rows(
    const GlmInt4MatrixView& matrix, std::span<const float> input,
    std::span<float> output, std::uint64_t row_begin, std::uint64_t row_end);

}  // namespace strata
