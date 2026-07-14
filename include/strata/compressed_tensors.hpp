#pragma once

#include "strata/model.hpp"
#include "strata/safetensors.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace strata {

struct CompressedTensorLayout {
    SafetensorsDtype packed_dtype{SafetensorsDtype::I32};
    SafetensorsDtype scale_dtype{SafetensorsDtype::Bf16};
    SafetensorsDtype shape_dtype{SafetensorsDtype::I64};
    std::uint64_t logical_rows{};
    std::uint64_t logical_columns{};
    std::uint64_t packed_rows{};
    std::uint64_t packed_columns{};
    std::uint64_t scale_rows{};
    std::uint64_t scale_columns{};
    std::uint64_t shape_elements{2};
};

[[nodiscard]] ValidationResult validate_compressed_tensor_layout(
    const CompressedTensorLayout& layout, const QuantizedWeightSpec& quantization);

[[nodiscard]] ValidationResult decode_compressed_logical_shape(
    std::span<const std::byte> encoded, std::array<std::uint64_t, 2>& shape);

[[nodiscard]] ValidationResult compressed_tensor_matvec_f32(
    std::span<float> output, std::span<const float> input,
    std::span<const std::byte> packed, std::span<const std::byte> scales,
    const CompressedTensorLayout& layout, const QuantizedWeightSpec& quantization);

}  // namespace strata
