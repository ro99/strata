#include "strata/compressed_tensors.hpp"

#include "../models/common/checkpoint_common.hpp"

#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace strata {

namespace {

std::uint64_t divide_round_up(std::uint64_t value, std::uint64_t divisor) {
    return value / divisor + static_cast<std::uint64_t>(value % divisor != 0);
}

std::uint32_t read_le_u32(const std::byte* bytes) {
    return static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[0])) |
           (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[1])) << 8U) |
           (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[2])) << 16U) |
           (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[3])) << 24U);
}

std::uint64_t read_le_u64(const std::byte* bytes) {
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < 8U; ++index) {
        value |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(bytes[index])) <<
                 (index * 8U);
    }
    return value;
}

float read_le_bf16(const std::byte* bytes) {
    const auto high = static_cast<std::uint32_t>(
        static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(bytes[0])) |
        (static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(bytes[1])) << 8U));
    return std::bit_cast<float>(high << 16U);
}

// Byte count a validated layout requires of each payload. Only called after
// `validate_compressed_tensor_layout` has agreed the shapes are consistent.
struct PayloadExtent {
    std::uint64_t packed_bytes{};
    std::uint64_t scale_bytes{};
    bool ok{};
};

PayloadExtent payload_extent(const CompressedTensorLayout& layout) {
    PayloadExtent extent;
    std::uint64_t packed_elements = 0;
    std::uint64_t scale_elements = 0;
    if (!detail::checked_product(layout.packed_rows, layout.packed_columns,
                                 packed_elements) ||
        !detail::checked_product(layout.scale_rows, layout.scale_columns,
                                 scale_elements)) {
        return extent;
    }
    const std::uint64_t packed_width =
        safetensors_dtype_bytes(layout.packed_dtype);
    const std::uint64_t scale_width = safetensors_dtype_bytes(layout.scale_dtype);
    if (packed_width == 0U || scale_width == 0U ||
        !detail::checked_product(packed_elements, packed_width,
                                 extent.packed_bytes) ||
        !detail::checked_product(scale_elements, scale_width, extent.scale_bytes)) {
        return extent;
    }
    extent.ok = true;
    return extent;
}

ValidationResult validate_mxfp4_layout(const CompressedTensorLayout& layout,
                                       const QuantizedWeightSpec& quantization) {
    ValidationResult result;
    if (layout.packed_dtype != SafetensorsDtype::U8) {
        result.errors.emplace_back("mxfp4 weight_packed must use Safetensors U8");
    }
    if (layout.scale_dtype != SafetensorsDtype::U8) {
        result.errors.emplace_back(
            "mxfp4 weight_scale must use Safetensors U8 holding E8M0 exponents");
    }
    if (layout.shape_elements != 0U) {
        result.errors.emplace_back(
            "mxfp4-pack-quantized ships no weight_shape tensor");
    }
    if (layout.logical_rows == 0U || layout.logical_columns == 0U) {
        result.errors.emplace_back(
            "logical weight shape must be two-dimensional and positive");
        return result;
    }
    // Four bits is the charter floor, and mxfp4 sits exactly on it: an E2M1
    // element plus a shared 8-bit block exponent. Anything narrower is
    // forbidden everywhere, so this is a hard reject rather than a warning.
    if (quantization.bits != 4U) {
        result.errors.emplace_back("mxfp4 weights must declare four bits");
        return result;
    }
    if (!quantization.symmetric) {
        result.errors.emplace_back("mxfp4 weights must be symmetric");
    }
    if (quantization.granularity != QuantizationGranularity::Group ||
        quantization.group_size == 0U) {
        result.errors.emplace_back("mxfp4 weights must be group quantized");
        return result;
    }
    if (layout.logical_columns % 2U != 0U) {
        result.errors.emplace_back(
            "mxfp4 packs two elements per byte, so the input dimension must be even");
    }
    if (layout.packed_rows != layout.logical_rows ||
        layout.packed_columns != layout.logical_columns / 2U) {
        result.errors.emplace_back(
            "mxfp4 weight_packed shape does not match logical weight shape");
    }
    if (layout.scale_rows != layout.logical_rows ||
        layout.scale_columns !=
            divide_round_up(layout.logical_columns, quantization.group_size)) {
        result.errors.emplace_back(
            "mxfp4 weight_scale shape does not match quantization granularity");
    }
    return result;
}

}  // namespace

float mxfp4_scale_from_e8m0(std::uint8_t bits) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 23U);
}

ValidationResult mxfp4_dequantize_row(std::span<float> output,
                                      std::span<const std::byte> packed,
                                      std::span<const std::byte> scales,
                                      const CompressedTensorLayout& layout,
                                      const QuantizedWeightSpec& quantization,
                                      std::uint64_t row) {
    auto result = validate_mxfp4_layout(layout, quantization);
    if (!result.ok()) return result;
    if (output.size() != layout.logical_columns) {
        result.errors.emplace_back("mxfp4 row buffer disagrees with logical width");
        return result;
    }
    if (row >= layout.logical_rows) {
        result.errors.emplace_back("mxfp4 row index is out of range");
        return result;
    }
    const auto extent = payload_extent(layout);
    if (!extent.ok || packed.size() != extent.packed_bytes ||
        scales.size() != extent.scale_bytes) {
        result.errors.emplace_back("mxfp4 payload byte count is invalid");
        return result;
    }

    const auto* packed_row = packed.data() + row * layout.packed_columns;
    const auto* scale_row = scales.data() + row * layout.scale_columns;
    for (std::uint64_t column = 0U; column < layout.logical_columns; ++column) {
        const auto byte = std::to_integer<std::uint8_t>(packed_row[column / 2U]);
        // Low nibble carries the even element, high nibble the odd one. This
        // ordering is the compressed-tensors convention and is checked against
        // the reference dequantizer by a real-tensor fixture.
        const auto nibble = static_cast<std::uint8_t>(
            column % 2U == 0U ? (byte & 0x0FU) : (byte >> 4U));
        const auto magnitude = kMxfp4Magnitudes[nibble & 0x07U];
        const auto scale = mxfp4_scale_from_e8m0(std::to_integer<std::uint8_t>(
            scale_row[column / quantization.group_size]));
        // 0xFF is the encoding's reserved slot; the bit shift renders it as
        // infinity rather than NaN, so guard on finiteness and not on NaN.
        if (!std::isfinite(scale)) {
            result.errors.emplace_back(
                "mxfp4 tensor contains a reserved (0xFF) block scale");
            return result;
        }
        const auto value = magnitude * scale;
        output[static_cast<std::size_t>(column)] =
            (nibble & 0x08U) != 0U ? -value : value;
    }
    return result;
}

ValidationResult validate_compressed_tensor_layout(
    const CompressedTensorLayout& layout, const QuantizedWeightSpec& quantization) {
    if (layout.codec == CompressedTensorCodec::Mxfp4E2m1) {
        return validate_mxfp4_layout(layout, quantization);
    }
    ValidationResult result;
    if (layout.packed_dtype != SafetensorsDtype::I32) {
        result.errors.emplace_back("weight_packed must use Safetensors I32");
    }
    if (layout.scale_dtype != SafetensorsDtype::Bf16) {
        result.errors.emplace_back("weight_scale must use Safetensors BF16");
    }
    if (layout.shape_dtype != SafetensorsDtype::I64 || layout.shape_elements != 2) {
        result.errors.emplace_back("weight_shape must contain two Safetensors I64 values");
    }
    if (layout.logical_rows == 0 || layout.logical_columns == 0) {
        result.errors.emplace_back("logical weight shape must be two-dimensional and positive");
        return result;
    }
    if (quantization.bits != 4 && quantization.bits != 8) {
        result.errors.emplace_back("pack-quantized weights must be INT4 or INT8");
        return result;
    }
    if (!quantization.symmetric) {
        result.errors.emplace_back("target pack-quantized weights must be symmetric");
    }

    const auto values_per_word = 32U / quantization.bits;
    const auto expected_packed_columns =
        divide_round_up(layout.logical_columns, values_per_word);
    if (layout.packed_rows != layout.logical_rows ||
        layout.packed_columns != expected_packed_columns) {
        result.errors.emplace_back("weight_packed shape does not match logical weight shape");
    }

    std::uint64_t expected_scale_columns = 1;
    if (quantization.granularity == QuantizationGranularity::Group) {
        if (quantization.group_size == 0) {
            result.errors.emplace_back("group quantization requires a positive group size");
            return result;
        }
        expected_scale_columns =
            divide_round_up(layout.logical_columns, quantization.group_size);
    } else if (quantization.group_size != 0) {
        result.errors.emplace_back("channel quantization cannot declare a group size");
    }
    if (layout.scale_rows != layout.logical_rows ||
        layout.scale_columns != expected_scale_columns) {
        result.errors.emplace_back("weight_scale shape does not match quantization granularity");
    }
    return result;
}

ValidationResult decode_compressed_logical_shape(
    std::span<const std::byte> encoded, std::array<std::uint64_t, 2>& shape) {
    ValidationResult result;
    if (encoded.size() != 16U) {
        result.errors.emplace_back("weight_shape payload must contain exactly two I64 values");
        return result;
    }
    shape[0] = read_le_u64(encoded.data());
    shape[1] = read_le_u64(encoded.data() + 8U);
    if (shape[0] == 0U || shape[1] == 0U ||
        shape[0] > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
        shape[1] > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        result.errors.emplace_back("weight_shape dimensions must be positive signed-I64 values");
    }
    return result;
}

ValidationResult compressed_tensor_matvec_f32(
    std::span<float> output, std::span<const float> input,
    std::span<const std::byte> packed, std::span<const std::byte> scales,
    const CompressedTensorLayout& layout, const QuantizedWeightSpec& quantization) {
    auto result = validate_compressed_tensor_layout(layout, quantization);
    if (!result.ok()) return result;
    if (output.size() != layout.logical_rows || input.size() != layout.logical_columns) {
        result.errors.emplace_back("matvec vectors disagree with logical weight shape");
        return result;
    }
    if (layout.codec == CompressedTensorCodec::Mxfp4E2m1) {
        // Accumulating group by group keeps the reduction order equal to the
        // decode order, so a matvec and a dequantize-then-dot agree exactly.
        std::vector<float> row(static_cast<std::size_t>(layout.logical_columns));
        for (std::uint64_t index = 0U; index < layout.logical_rows; ++index) {
            auto decoded = mxfp4_dequantize_row(row, packed, scales, layout,
                                                quantization, index);
            if (!decoded.ok()) return decoded;
            float sum = 0.0F;
            for (std::size_t column = 0U; column < row.size(); ++column) {
                sum += row[column] * input[column];
            }
            output[static_cast<std::size_t>(index)] = sum;
        }
        return result;
    }
    std::uint64_t packed_words = 0;
    std::uint64_t scale_values = 0;
    if (!detail::checked_product(layout.packed_rows, layout.packed_columns, packed_words) ||
        !detail::checked_product(layout.scale_rows, layout.scale_columns, scale_values) ||
        packed_words > std::numeric_limits<std::uint64_t>::max() / 4U ||
        scale_values > std::numeric_limits<std::uint64_t>::max() / 2U ||
        packed.size() != packed_words * 4U || scales.size() != scale_values * 2U) {
        result.errors.emplace_back("compressed tensor payload byte count is invalid");
        return result;
    }

    const auto values_per_word = 32U / quantization.bits;
    const auto mask = (1U << quantization.bits) - 1U;
    const auto offset = 1U << (quantization.bits - 1U);
    for (std::uint64_t row = 0; row < layout.logical_rows; ++row) {
        float sum = 0.0F;
        for (std::uint64_t column = 0; column < layout.logical_columns; ++column) {
            const auto word_index = row * layout.packed_columns + column / values_per_word;
            const auto lane = static_cast<std::uint32_t>(column % values_per_word);
            const auto word = read_le_u32(packed.data() + word_index * 4U);
            const auto raw = (word >> (lane * quantization.bits)) & mask;
            const auto quantized = static_cast<std::int32_t>(raw) -
                                   static_cast<std::int32_t>(offset);
            const auto scale_column = quantization.granularity ==
                                              QuantizationGranularity::Group
                                          ? column / quantization.group_size
                                          : 0U;
            const auto scale_index = row * layout.scale_columns + scale_column;
            const auto scale = read_le_bf16(scales.data() + scale_index * 2U);
            if (!std::isfinite(scale)) {
                result.errors.emplace_back("compressed tensor contains a non-finite scale");
                return result;
            }
            sum += static_cast<float>(quantized) * scale *
                   input[static_cast<std::size_t>(column)];
        }
        output[static_cast<std::size_t>(row)] = sum;
    }
    return result;
}

}  // namespace strata
