#include "strata/numerics.hpp"

#include <cmath>

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

}  // namespace strata
