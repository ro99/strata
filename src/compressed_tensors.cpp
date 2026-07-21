#include "strata/compressed_tensors.hpp"

#include "checkpoint_common.hpp"

#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

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

}  // namespace

ValidationResult validate_compressed_tensor_layout(
    const CompressedTensorLayout& layout, const QuantizedWeightSpec& quantization) {
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
