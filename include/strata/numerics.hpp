#pragma once

#include "strata/result.hpp"

#include <bit>
#include <span>
#include <cstdint>

namespace strata {

[[nodiscard]] ValidationResult rms_norm_f32(
    std::span<float> output, std::span<const float> input,
    std::span<const float> weight, float epsilon);
[[nodiscard]] float sigmoid_f32(float value) noexcept;
[[nodiscard]] float silu_f32(float value) noexcept;
[[nodiscard]] float log_sigmoid_f32(float value) noexcept;
// Storage-format decoders shared by every NVFP4 adapter. E4M3 carries the
// group scales and E2M1 the packed weight nibbles; both are exact tables of
// the format, not model-specific policy.
[[nodiscard]] float fp8_e4m3_f32(std::uint8_t encoded) noexcept;
[[nodiscard]] float fp4_e2m1_f32(std::uint8_t nibble) noexcept;
// Defined inline: both sit in per-element loops over billions of activation
// values (paged-attention query validation and BF16 upload staging). Called
// out of line without LTO they cost a real call per element and cannot
// vectorize; experiment 0113 measured 3.3x on the staging loop from inlining
// alone. The arithmetic is unchanged.
[[nodiscard]] inline std::uint16_t bf16_encode(float value) noexcept {
    auto bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7F80'0000U) != 0x7F80'0000U) {
        bits += 0x7FFFU + ((bits >> 16U) & 1U);
    }
    return static_cast<std::uint16_t>(bits >> 16U);
}
[[nodiscard]] inline float bf16_round_f32(float value) noexcept {
    return std::bit_cast<float>(
        static_cast<std::uint32_t>(bf16_encode(value)) << 16U);
}
[[nodiscard]] float gelu_tanh_f32(float value) noexcept;

}  // namespace strata
