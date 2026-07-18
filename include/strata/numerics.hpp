#pragma once

#include "strata/result.hpp"

#include <span>

namespace strata {

[[nodiscard]] ValidationResult rms_norm_f32(
    std::span<float> output, std::span<const float> input,
    std::span<const float> weight, float epsilon);
[[nodiscard]] float sigmoid_f32(float value) noexcept;
[[nodiscard]] float silu_f32(float value) noexcept;

}  // namespace strata
