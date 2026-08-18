#pragma once

#include "strata/quantization.hpp"
#include "strata/safetensors.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace strata {

// How the packed payload encodes one weight element.
enum class CompressedTensorCodec : std::uint8_t {
    // `pack-quantized`: `bits`-wide two's-complement indices packed into I32
    // words, offset by 2^(bits-1), scaled by a BF16 group scale. GLM and
    // Gemma 4 use this.
    OffsetPackedInt,
    // `mxfp4-pack-quantized`: two E2M1 elements per U8 (low nibble first),
    // sign in bit 3, magnitude index in bits 0..2, scaled by a per-group E8M0
    // exponent stored in U8. Kimi-K3's routed experts use this.
    Mxfp4E2m1,
};

struct CompressedTensorLayout {
    CompressedTensorCodec codec{CompressedTensorCodec::OffsetPackedInt};
    SafetensorsDtype packed_dtype{SafetensorsDtype::I32};
    SafetensorsDtype scale_dtype{SafetensorsDtype::Bf16};
    SafetensorsDtype shape_dtype{SafetensorsDtype::I64};
    std::uint64_t logical_rows{};
    std::uint64_t logical_columns{};
    std::uint64_t packed_rows{};
    std::uint64_t packed_columns{};
    std::uint64_t scale_rows{};
    std::uint64_t scale_columns{};
    // Zero declares that the checkpoint ships no `weight_shape` tensor, which
    // is how `mxfp4-pack-quantized` is written. Two is the pack-quantized form.
    std::uint64_t shape_elements{2};
};

// E2M1 magnitude table, indexed by bits 0..2 of a nibble. Sign is bit 3.
inline constexpr std::array<float, 8> kMxfp4Magnitudes{
    0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};

// The same sixteen values with the sign bit folded in, indexed by the whole
// nibble. Every entry equals what negating `kMxfp4Magnitudes[bits 0..2]` on
// bit 3 produces, so a decoder that indexes this table is bit-identical to one
// that branches. What changes is that it does not branch: weight nibbles are
// effectively random, so a predicate on the sign bit mispredicts about half the
// time, and on a decode loop this small the misprediction costs more than the
// arithmetic it guards.
inline constexpr std::array<float, 16> kMxfp4Values{
    0.0F,  0.5F,  1.0F,  1.5F,  2.0F,  3.0F,  4.0F,  6.0F,
    -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F};

// One E8M0 byte to its power-of-two scale. The encoding is the float32
// exponent field: `2^(bits - 127)`, with 0 meaning zero. This is a bit shift,
// not an arithmetic conversion; keeping it exact is what makes the codec
// reproduce the reference dequantization bit for bit. The shift renders the
// encoding's reserved 0xFF slot as infinity rather than NaN, which is why
// callers must reject non-finite scales rather than testing for NaN.
[[nodiscard]] float mxfp4_scale_from_e8m0(std::uint8_t bits) noexcept;

// Decodes one logical row into `output`, which must hold `logical_columns`
// values. Exact: every output is a product of a table entry and a power of two.
[[nodiscard]] ValidationResult mxfp4_dequantize_row(
    std::span<float> output, std::span<const std::byte> packed,
    std::span<const std::byte> scales, const CompressedTensorLayout& layout,
    const QuantizedWeightSpec& quantization, std::uint64_t row);

[[nodiscard]] ValidationResult validate_compressed_tensor_layout(
    const CompressedTensorLayout& layout, const QuantizedWeightSpec& quantization);

[[nodiscard]] ValidationResult decode_compressed_logical_shape(
    std::span<const std::byte> encoded, std::array<std::uint64_t, 2>& shape);

[[nodiscard]] ValidationResult compressed_tensor_matvec_f32(
    std::span<float> output, std::span<const float> input,
    std::span<const std::byte> packed, std::span<const std::byte> scales,
    const CompressedTensorLayout& layout, const QuantizedWeightSpec& quantization);

}  // namespace strata
