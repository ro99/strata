#pragma once

#include "strata/result.hpp"

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
[[nodiscard]] std::uint16_t bf16_encode(float value) noexcept;
[[nodiscard]] float bf16_round_f32(float value) noexcept;
[[nodiscard]] float gelu_tanh_f32(float value) noexcept;

}  // namespace strata
