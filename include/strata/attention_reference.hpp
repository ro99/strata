#pragma once

#include "strata/attention.hpp"

namespace strata {

// Scalar oracle for the request's declared numerical contract. Architecture
// adapters own any BF16 rounding boundaries before and after this operation.
[[nodiscard]] ValidationResult flash_attention_reference_f32(
    const FlashAttentionRequest& request, std::span<float> output);

}  // namespace strata
