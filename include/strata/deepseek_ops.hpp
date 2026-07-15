#pragma once

#include "strata/model.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace strata {

inline constexpr float kDsv4NormEpsilon = 1.0e-6F;
inline constexpr std::uint32_t kDsv4MhcMultiplier = 4U;
inline constexpr std::uint32_t kDsv4MhcSinkhornIterations = 20U;

struct Dsv4Route {
    std::vector<std::uint32_t> experts;
    std::vector<float> weights;
};

struct Dsv4RouteResult {
    Dsv4Route value;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

struct Dsv4MhcMix {
    std::vector<float> pre;
    std::vector<float> post;
    std::vector<float> combination;
};

struct Dsv4MhcMixResult {
    Dsv4MhcMix value;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

[[nodiscard]] float dsv4_softplus_f32(float value) noexcept;
[[nodiscard]] float dsv4_fp4_e2m1_f32(std::uint8_t nibble) noexcept;
[[nodiscard]] float dsv4_fp8_e4m3_f32(std::uint8_t encoded) noexcept;
[[nodiscard]] float dsv4_fp8_e8m0_scale_f32(std::uint8_t encoded) noexcept;

// The selection bias changes only top-k membership. Returned coefficients are
// gathered from unbiased sqrt(softplus(logit)), normalized, then scaled.
[[nodiscard]] Dsv4RouteResult dsv4_route_sqrtsoftplus_f32(
    std::span<const float> logits, std::span<const float> selection_bias,
    const RouterSpec& spec);

// Hash layers use the checkpoint's token-to-expert row for membership while
// retaining the same score, normalization, and routed-scale semantics.
[[nodiscard]] Dsv4RouteResult dsv4_route_hash_sqrtsoftplus_f32(
    std::span<const float> logits, std::span<const std::uint32_t> token_experts,
    const RouterSpec& spec);

[[nodiscard]] ValidationResult dsv4_swiglu_f32(
    std::span<float> output, std::span<const float> gate,
    std::span<const float> up, float limit);

// Native target-checkpoint reference oracles. FP4 bytes contain the lower-K
// value in the low nibble and use one E8M0 scale for every 32 logical columns.
[[nodiscard]] ValidationResult dsv4_fp4_e2m1_matvec_f32(
    std::span<float> output, std::span<const float> input,
    std::span<const std::byte> packed_weights,
    std::span<const std::byte> e8m0_scales,
    std::uint64_t rows, std::uint64_t columns);

// FP8 weights use one E8M0 scale for each 128 x 128 output/input block.
[[nodiscard]] ValidationResult dsv4_fp8_e4m3_matvec_f32(
    std::span<float> output, std::span<const float> input,
    std::span<const std::byte> weights,
    std::span<const std::byte> e8m0_scales,
    std::uint64_t rows, std::uint64_t columns);

[[nodiscard]] Dsv4MhcMixResult dsv4_mhc_split_sinkhorn_f32(
    std::span<const float> mixes, std::span<const float> scale,
    std::span<const float> base,
    std::uint32_t multiplier = kDsv4MhcMultiplier,
    std::uint32_t iterations = kDsv4MhcSinkhornIterations,
    float epsilon = kDsv4NormEpsilon);

[[nodiscard]] ValidationResult dsv4_mhc_pre_f32(
    std::span<float> reduced, Dsv4MhcMix& mix,
    std::span<const float> hidden_copies, std::span<const float> projection,
    std::span<const float> scale, std::span<const float> base,
    std::uint32_t multiplier = kDsv4MhcMultiplier,
    std::uint32_t iterations = kDsv4MhcSinkhornIterations,
    float epsilon = kDsv4NormEpsilon);

[[nodiscard]] ValidationResult dsv4_mhc_post_f32(
    std::span<float> output_copies, std::span<const float> branch,
    std::span<const float> residual_copies, const Dsv4MhcMix& mix,
    std::uint32_t multiplier = kDsv4MhcMultiplier);

}  // namespace strata
