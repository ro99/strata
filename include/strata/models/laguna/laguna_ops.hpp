#pragma once

#include "strata/models/common/model.hpp"
#include "strata/models/common/model_adapter.hpp"
#include "strata/platform/numerics.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace strata {

// One NVFP4 ("nvfp4-pack-quantized") matrix as it is stored in the checkpoint.
// `packed` is row-major [rows, columns/2] with the even column in the low
// nibble. `scales` is row-major [rows, columns/group_size] FP8 E4M3. The
// per-tensor `global_scale` is FP32 and dequantization is
//   w[r][c] = e2m1(nibble) * (e4m3(scale[r][c / group_size]) / global_scale)
// which is the compressed-tensors tensor_group rule.
struct LagunaNvfp4MatrixView {
    std::span<const std::byte> packed;
    std::span<const std::byte> scales;
    float global_scale{1.0F};
    std::uint64_t rows{};
    std::uint64_t columns{};
    std::uint64_t packed_columns{};
    std::uint64_t scale_columns{};
    std::uint32_t group_size{kLagunaExecutionContract.nvfp4_group_size};
};

struct LagunaRoute {
    std::vector<std::uint32_t> experts;
    std::vector<float> weights;
};

struct LagunaRouteResult {
    LagunaRoute value;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

// Rotary schedule for one layer type. `inverse_frequencies` has rotary_dim / 2
// entries and `attention_scaling` multiplies both cos and sin, as YaRN
// prescribes. Sliding layers use plain RoPE and an attention scaling of one.
struct LagunaRopeSchedule {
    std::vector<float> inverse_frequencies;
    float attention_scaling{1.0F};
    std::uint32_t rotary_dimensions{};
};

[[nodiscard]] float laguna_fp8_e4m3_f32(std::uint8_t encoded) noexcept;
[[nodiscard]] float laguna_fp4_e2m1_f32(std::uint8_t nibble) noexcept;
[[nodiscard]] float laguna_softplus_f32(float value) noexcept;

// Scalar target-format oracle for NVFP4. Dequantizes exactly as the reference
// does and accumulates in FP32.
[[nodiscard]] ValidationResult laguna_nvfp4_matvec_reference(
    const LagunaNvfp4MatrixView& matrix, std::span<const float> input,
    std::span<float> output);
[[nodiscard]] ValidationResult laguna_nvfp4_matvec_rows(
    const LagunaNvfp4MatrixView& matrix, std::span<const float> input,
    std::span<float> output, std::uint64_t row_begin, std::uint64_t row_end);

// Builds the rotary schedule declared by the execution contract. Global layers
// use YaRN over the first half of each head; sliding layers use default RoPE
// over the whole head.
[[nodiscard]] LagunaRopeSchedule laguna_rope_schedule(bool global_layer);

// Rotates one head in place using the `rotate_half` convention: the leading
// `rotary_dimensions` values split at their midpoint, the tail is passed
// through unchanged.
[[nodiscard]] ValidationResult laguna_rope_half_f32(
    std::span<float> head, std::uint64_t position,
    const LagunaRopeSchedule& schedule);

// Applies the pinned Laguna router rule to precomputed router logits: optional
// soft-capping, sigmoid scoring, top-k over score + correction bias, then
// normalization of the unbiased scores. Equal selection scores retain the lower
// expert index. The routed scaling factor is NOT applied here; the reference
// scales the summed routed output, so the caller must do the same.
[[nodiscard]] LagunaRouteResult laguna_route_sigmoid_topk(
    std::span<const float> logits, std::span<const float> correction_bias,
    const RouterSpec& spec, float logit_softcapping);

// True when a query at `query` may attend to `key` under the layer's pattern.
[[nodiscard]] bool laguna_attention_visible(
    std::uint64_t query, std::uint64_t key, bool sliding,
    std::uint32_t sliding_window) noexcept;

}  // namespace strata
