#pragma once

#include "strata/models/common/model_adapter.hpp"
#include "strata/platform/result.hpp"

#include <cstdint>
#include <span>

namespace strata {

[[nodiscard]] ValidationResult gemma4_rms_norm_bf16(
    std::span<float> output, std::span<const float> input,
    std::span<const float> weight,
    float epsilon = kGemma4ExecutionContract.rms_epsilon);

[[nodiscard]] ValidationResult gemma4_rope_bf16(
    std::span<float> values, std::uint64_t position, float theta,
    float rotary_proportion = 1.0F);

[[nodiscard]] ValidationResult gemma4_vision_rope_bf16(
    std::span<float> values, std::int32_t x, std::int32_t y,
    float theta = kGemma4ExecutionContract.vision_rope_theta);

[[nodiscard]] ValidationResult gemma4_geglu_bf16(
    std::span<float> output, std::span<const float> gate,
    std::span<const float> up);

[[nodiscard]] bool gemma4_text_attention_visible(
    std::uint64_t query, std::uint64_t key, bool sliding,
    std::uint32_t sliding_window,
    std::span<const std::int32_t> multimodal_groups = {}) noexcept;

}  // namespace strata
