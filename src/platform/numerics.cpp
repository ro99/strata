#include "strata/platform/numerics.hpp"

#include <array>
#include <cmath>
#include <bit>
#include <limits>

namespace strata {

ValidationResult rms_norm_f32(std::span<float> output,
                              std::span<const float> input,
                              std::span<const float> weight,
                              float epsilon) {
    ValidationResult result;
    if (output.size() != input.size() || input.size() != weight.size() || input.empty()) {
        result.errors.emplace_back("RMSNorm spans have incompatible sizes");
        return result;
    }
    if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
        result.errors.emplace_back("RMSNorm epsilon must be finite and positive");
        return result;
    }
    double squared_sum = 0.0;
    for (const float value : input) {
        if (!std::isfinite(value)) {
            result.errors.emplace_back("RMSNorm input contains a non-finite value");
            return result;
        }
        squared_sum += static_cast<double>(value) * static_cast<double>(value);
    }
    const auto mean_square = static_cast<float>(
        squared_sum / static_cast<double>(input.size()));
    const float reciprocal = 1.0F / std::sqrt(mean_square + epsilon);
    for (std::size_t index = 0; index < input.size(); ++index) {
        if (!std::isfinite(weight[index])) {
            result.errors.emplace_back("RMSNorm weight contains a non-finite value");
            return result;
        }
        output[index] = input[index] * reciprocal * weight[index];
    }
    return result;
}

float sigmoid_f32(float value) noexcept {
    if (value >= 0.0F) return 1.0F / (1.0F + std::exp(-value));
    const float exponential = std::exp(value);
    return exponential / (1.0F + exponential);
}

float silu_f32(float value) noexcept {
    return value * sigmoid_f32(value);
}

float log_sigmoid_f32(float value) noexcept {
    // min(x, 0) - log1p(exp(-|x|)) is the overflow-free form the reference
    // router uses; the naive log(sigmoid(x)) underflows to -inf for x well
    // below zero, which would silently drop an expert from the renormalization.
    return std::fmin(value, 0.0F) - std::log1p(std::exp(-std::fabs(value)));
}

float fp8_e4m3_f32(std::uint8_t encoded) noexcept {
    const bool negative = (encoded & 0x80U) != 0U;
    const std::uint32_t exponent = (encoded >> 3U) & 0x0FU;
    const std::uint32_t mantissa = encoded & 0x07U;
    float value = 0.0F;
    if (exponent == 0U) {
        value = std::ldexp(static_cast<float>(mantissa) / 8.0F, -6);
    } else if (exponent == 0x0FU && mantissa == 0x07U) {
        return std::numeric_limits<float>::quiet_NaN();
    } else {
        value = std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                           static_cast<int>(exponent) - 7);
    }
    return negative ? -value : value;
}

float fp4_e2m1_f32(std::uint8_t nibble) noexcept {
    // E2M1 has sixteen exact values, so a table is both the fastest and the
    // least error-prone decoder. The sign bit is the high nibble bit.
    constexpr std::array<float, 16> values{
        0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
        -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F};
    return values[nibble & 0x0FU];
}

float gelu_tanh_f32(float value) noexcept {
    constexpr float coefficient = 0.7978845608028654F;
    return 0.5F * value *
           (1.0F + std::tanh(coefficient *
                             (value + 0.044715F * value * value * value)));
}

}  // namespace strata
