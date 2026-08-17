#include "strata/gemma4_ops.hpp"

#include "strata/numerics.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace strata {

ValidationResult gemma4_rms_norm_bf16(
    std::span<float> output, std::span<const float> input,
    std::span<const float> weight, float epsilon) {
    auto result = rms_norm_f32(output, input, weight, epsilon);
    if (result.ok()) {
        for (auto& value : output) value = bf16_round_f32(value);
    }
    return result;
}

ValidationResult gemma4_rope_bf16(
    std::span<float> values, std::uint64_t position, float theta,
    float rotary_proportion) {
    ValidationResult result;
    if (values.empty() || values.size() % 2U != 0U ||
        !std::isfinite(theta) || theta <= 0.0F ||
        !std::isfinite(rotary_proportion) || rotary_proportion <= 0.0F ||
        rotary_proportion > 1.0F) {
        result.errors.emplace_back("Gemma 4 RoPE dimensions or parameters are invalid");
        return result;
    }
    const auto half = values.size() / 2U;
    const auto angles = static_cast<std::size_t>(
        rotary_proportion * static_cast<float>(values.size()) / 2.0F);
    std::vector<float> input(values.begin(), values.end());
    for (std::size_t index = 0U; index < angles; ++index) {
        const float inverse_frequency = std::pow(
            theta, -2.0F * static_cast<float>(index) /
                       static_cast<float>(values.size()));
        const float angle = static_cast<float>(position) * inverse_frequency;
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        values[index] = bf16_round_f32(
            input[index] * cosine - input[half + index] * sine);
        values[half + index] = bf16_round_f32(
            input[half + index] * cosine + input[index] * sine);
    }
    return result;
}

ValidationResult gemma4_vision_rope_bf16(
    std::span<float> values, std::int32_t x, std::int32_t y, float theta) {
    ValidationResult result;
    if (values.empty() || values.size() % 4U != 0U || x < 0 || y < 0) {
        result.errors.emplace_back("Gemma 4 vision RoPE position or head shape is invalid");
        return result;
    }
    const auto spatial = values.size() / 2U;
    auto first = gemma4_rope_bf16(values.first(spatial),
                                  static_cast<std::uint64_t>(x), theta);
    auto second = gemma4_rope_bf16(values.subspan(spatial),
                                   static_cast<std::uint64_t>(y), theta);
    result.errors.insert(result.errors.end(), first.errors.begin(), first.errors.end());
    result.errors.insert(result.errors.end(), second.errors.begin(), second.errors.end());
    return result;
}

ValidationResult gemma4_geglu_bf16(
    std::span<float> output, std::span<const float> gate,
    std::span<const float> up) {
    ValidationResult result;
    if (output.empty() || output.size() != gate.size() || gate.size() != up.size()) {
        result.errors.emplace_back("Gemma 4 GeGLU spans have incompatible sizes");
        return result;
    }
    for (std::size_t index = 0U; index < output.size(); ++index) {
        if (!std::isfinite(gate[index]) || !std::isfinite(up[index])) {
            result.errors.emplace_back("Gemma 4 GeGLU input is non-finite");
            return result;
        }
        output[index] = bf16_round_f32(
            bf16_round_f32(gelu_tanh_f32(gate[index])) * up[index]);
    }
    return result;
}

bool gemma4_text_attention_visible(
    std::uint64_t query, std::uint64_t key, bool sliding,
    std::uint32_t sliding_window,
    std::span<const std::int32_t> multimodal_groups) noexcept {
    const bool same_multimodal_group = query < multimodal_groups.size() &&
        key < multimodal_groups.size() && multimodal_groups[query] >= 0 &&
        multimodal_groups[query] == multimodal_groups[key];
    if (same_multimodal_group) return true;
    if (key > query) return false;
    return !sliding || (sliding_window != 0U &&
        query - key < static_cast<std::uint64_t>(sliding_window));
}

}  // namespace strata
