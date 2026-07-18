#pragma once

#include "strata/model.hpp"
#include "strata/numerics.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strata {

inline constexpr float kGlm52RmsNormEpsilon = 1.0e-5F;
inline constexpr float kGlm52IndexerNormEpsilon = 1.0e-6F;
inline constexpr float kGlm52RopeTheta = 8'000'000.0F;
inline constexpr std::uint32_t kGlm52RopeDimensions = 64U;

struct GlmRoute {
    std::vector<std::uint32_t> experts;
    std::vector<float> weights;
};

struct GlmRouteResult {
    GlmRoute value;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

[[nodiscard]] ValidationResult glm_layer_norm_f32(
    std::span<float> values, std::span<const float> weight,
    std::span<const float> bias, float epsilon = kGlm52IndexerNormEpsilon);

[[nodiscard]] ValidationResult glm_softmax_f32(std::span<float> values);

[[nodiscard]] ValidationResult glm_rope_interleaved_f32(
    std::span<float> values, std::uint64_t position,
    std::uint32_t rope_dimensions = kGlm52RopeDimensions,
    float theta = kGlm52RopeTheta);

// Applies the pinned sigmoid/noaux_tc rule to precomputed router logits. The
// correction bias participates in top-k selection only; returned weights come
// from the unbiased sigmoid scores, are normalized, then routed_scale is
// applied. Equal scores retain the lower expert index.
[[nodiscard]] GlmRouteResult glm_route_logits_noaux_tc(
    std::span<const float> logits, std::span<const float> correction_bias,
    const RouterSpec& spec);

// Reference FP32 router projection followed by glm_route_logits_noaux_tc.
// router_weight is row-major [routed_experts, hidden_size].
[[nodiscard]] GlmRouteResult glm_route_noaux_tc_f32(
    std::span<const float> hidden, std::span<const float> router_weight,
    std::span<const float> correction_bias, const RouterSpec& spec);

}  // namespace strata
