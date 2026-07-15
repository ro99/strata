#include "strata/glm_int4.hpp"

#include <bit>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <limits>

namespace strata {

namespace {

ValidationResult validate(const GlmInt4MatrixView& matrix,
                          std::span<const float> input,
                          std::span<float> output) {
    ValidationResult result;
    if (matrix.rows == 0U || matrix.columns == 0U || matrix.group_size != 128U ||
        matrix.packed_columns != (matrix.columns + 7U) / 8U ||
        matrix.scale_columns != (matrix.columns + 127U) / 128U ||
        input.size() != matrix.columns || output.size() != matrix.rows) {
        result.errors.emplace_back("invalid QuantTrio INT4 group-128 matvec shape");
        return result;
    }
    if (matrix.rows > std::numeric_limits<std::uint64_t>::max() /
                          matrix.packed_columns ||
        matrix.rows * matrix.packed_columns >
            std::numeric_limits<std::uint64_t>::max() / 4U ||
        matrix.rows > std::numeric_limits<std::uint64_t>::max() /
                          matrix.scale_columns ||
        matrix.rows * matrix.scale_columns >
            std::numeric_limits<std::uint64_t>::max() / 2U ||
        matrix.packed.size() != matrix.rows * matrix.packed_columns * 4U ||
        matrix.scales.size() != matrix.rows * matrix.scale_columns * 2U) {
        result.errors.emplace_back("invalid QuantTrio INT4 group-128 payload size");
    }
    return result;
}

float bf16_value(const std::byte* source) noexcept {
    std::uint16_t encoded = 0U;
    std::memcpy(&encoded, source, sizeof(encoded));
    return std::bit_cast<float>(static_cast<std::uint32_t>(encoded) << 16U);
}

__attribute__((target("avx2,fma")))
void matvec_avx2(const GlmInt4MatrixView& matrix,
                 std::span<const float> input,
                 std::span<float> output, std::uint64_t row_begin,
                 std::uint64_t row_end) {
    const auto* packed = reinterpret_cast<const std::uint8_t*>(matrix.packed.data());
    const __m128i nibble_mask = _mm_set1_epi8(0x0F);
    const __m128i offset = _mm_set1_epi8(8);
    for (std::uint64_t row = row_begin; row < row_end; ++row) {
        const auto* packed_row = packed + row * matrix.packed_columns * 4U;
        const auto* scale_row = matrix.scales.data() + row * matrix.scale_columns * 2U;
        alignas(32) float lane_sums[256]{};
        const auto vector_columns = matrix.columns - matrix.columns % 256U;
        for (std::uint64_t lane_base = 0U; lane_base < 256U;
             lane_base += 32U) {
            __m256 accumulator0 = _mm256_setzero_ps();
            __m256 accumulator1 = _mm256_setzero_ps();
            __m256 accumulator2 = _mm256_setzero_ps();
            __m256 accumulator3 = _mm256_setzero_ps();
            for (std::uint64_t packed_column = lane_base;
                 packed_column < vector_columns; packed_column += 256U) {
                const __m128i bytes = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(
                        packed_row + packed_column / 2U));
                const __m128i low_nibbles = _mm_and_si128(bytes, nibble_mask);
                const __m128i high_nibbles = _mm_and_si128(
                    _mm_srli_epi16(bytes, 4), nibble_mask);
                const __m128i quantized0 = _mm_sub_epi8(
                    _mm_unpacklo_epi8(low_nibbles, high_nibbles), offset);
                const __m128i quantized1 = _mm_sub_epi8(
                    _mm_unpackhi_epi8(low_nibbles, high_nibbles), offset);
                const float scale_value = bf16_value(
                    scale_row +
                    (packed_column / matrix.group_size) * 2U);
                const __m256 scale = _mm256_set1_ps(scale_value);
                const __m256 q0 = _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(quantized0));
                const __m256 q1 = _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_srli_si128(quantized0, 8)));
                const __m256 q2 = _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(quantized1));
                const __m256 q3 = _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_srli_si128(quantized1, 8)));
                const __m256 product0 = _mm256_mul_ps(
                    _mm256_loadu_ps(input.data() + packed_column), q0);
                const __m256 product1 = _mm256_mul_ps(
                    _mm256_loadu_ps(input.data() + packed_column + 8U), q1);
                const __m256 product2 = _mm256_mul_ps(
                    _mm256_loadu_ps(input.data() + packed_column + 16U), q2);
                const __m256 product3 = _mm256_mul_ps(
                    _mm256_loadu_ps(input.data() + packed_column + 24U), q3);
                accumulator0 = _mm256_fmadd_ps(scale, product0, accumulator0);
                accumulator1 = _mm256_fmadd_ps(scale, product1, accumulator1);
                accumulator2 = _mm256_fmadd_ps(scale, product2, accumulator2);
                accumulator3 = _mm256_fmadd_ps(scale, product3, accumulator3);
            }
            _mm256_store_ps(lane_sums + lane_base, accumulator0);
            _mm256_store_ps(lane_sums + lane_base + 8U, accumulator1);
            _mm256_store_ps(lane_sums + lane_base + 16U, accumulator2);
            _mm256_store_ps(lane_sums + lane_base + 24U, accumulator3);
        }
        for (std::uint64_t column = vector_columns;
             column < matrix.columns; ++column) {
            const std::uint8_t byte = packed_row[column / 2U];
            const std::uint8_t raw = (column & 1U) == 0U
                ? byte & 0x0FU : byte >> 4U;
            const float scale = bf16_value(
                scale_row + (column / matrix.group_size) * 2U);
            const float product = input[static_cast<std::size_t>(column)] *
                                  static_cast<float>(static_cast<int>(raw) - 8);
            const auto lane = static_cast<std::size_t>(column % 256U);
            lane_sums[lane] = std::fma(scale, product, lane_sums[lane]);
        }
        float warp_sums[32]{};
        for (std::size_t warp = 0U; warp < 8U; ++warp) {
            float values[32];
            std::memcpy(values, lane_sums + warp * 32U, sizeof(values));
            for (std::size_t reduction = 16U; reduction > 0U; reduction >>= 1U) {
                for (std::size_t lane = 0U; lane < reduction; ++lane) {
                    values[lane] += values[lane + reduction];
                }
            }
            warp_sums[warp] = values[0];
        }
        for (std::size_t reduction = 16U; reduction > 0U; reduction >>= 1U) {
            for (std::size_t lane = 0U; lane < reduction; ++lane) {
                warp_sums[lane] += warp_sums[lane + reduction];
            }
        }
        output[static_cast<std::size_t>(row)] = warp_sums[0];
    }
}

}  // namespace

ValidationResult glm_int4_group128_matvec_reference(
    const GlmInt4MatrixView& matrix, std::span<const float> input,
    std::span<float> output) {
    auto result = validate(matrix, input, output);
    if (!result.ok()) return result;
    for (std::uint64_t row = 0U; row < matrix.rows; ++row) {
        float sum = 0.0F;
        for (std::uint64_t column = 0U; column < matrix.columns; ++column) {
            std::uint32_t word = 0U;
            const auto word_index = row * matrix.packed_columns + column / 8U;
            std::memcpy(&word, matrix.packed.data() + word_index * 4U, sizeof(word));
            const auto raw = (word >> ((column % 8U) * 4U)) & 0x0FU;
            const auto scale_index = row * matrix.scale_columns + column / 128U;
            const float scale = bf16_value(matrix.scales.data() + scale_index * 2U);
            sum += input[static_cast<std::size_t>(column)] *
                   static_cast<float>(static_cast<int>(raw) - 8) * scale;
        }
        output[static_cast<std::size_t>(row)] = sum;
    }
    return result;
}

ValidationResult glm_int4_group128_matvec(
    const GlmInt4MatrixView& matrix, std::span<const float> input,
    std::span<float> output) {
    auto result = validate(matrix, input, output);
    if (!result.ok()) return result;
#if defined(__x86_64__) || defined(__i386__)
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        matvec_avx2(matrix, input, output, 0U, matrix.rows);
        return result;
    }
#endif
    return glm_int4_group128_matvec_reference(matrix, input, output);
}

ValidationResult glm_int4_group128_matvec_rows(
    const GlmInt4MatrixView& matrix, std::span<const float> input,
    std::span<float> output, std::uint64_t row_begin, std::uint64_t row_end) {
    auto result = validate(matrix, input, output);
    if (!result.ok()) return result;
    if (row_begin >= row_end || row_end > matrix.rows) {
        result.errors.emplace_back("invalid QuantTrio INT4 matvec row range");
        return result;
    }
#if defined(__x86_64__) || defined(__i386__)
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        matvec_avx2(matrix, input, output, row_begin, row_end);
        return result;
    }
#endif
    for (std::uint64_t row = row_begin; row < row_end; ++row) {
        const GlmInt4MatrixView row_matrix{
            matrix.packed.subspan(
                static_cast<std::size_t>(row * matrix.packed_columns * 4U),
                static_cast<std::size_t>(matrix.packed_columns * 4U)),
            matrix.scales.subspan(
                static_cast<std::size_t>(row * matrix.scale_columns * 2U),
                static_cast<std::size_t>(matrix.scale_columns * 2U)),
            1U, matrix.columns, matrix.packed_columns, matrix.scale_columns,
            matrix.group_size};
        auto status = glm_int4_group128_matvec_reference(
            row_matrix, input, output.subspan(static_cast<std::size_t>(row), 1U));
        if (!status.ok()) return status;
    }
    return result;
}

}  // namespace strata
