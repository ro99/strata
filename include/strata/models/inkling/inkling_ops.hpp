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

// One NVFP4 matrix as Inkling-Small stores it. This is the ModelOpt/TensorRT
// convention, not the compressed-tensors one: the per-expert FP32 scale is a
// *multiplier* derived as amax / (6 * 448), so dequantization is
//   w[r][c] = e2m1(nibble) * e4m3(scale[r][c / group_size]) * global_scale
// Applying the compressed-tensors reciprocal rule here inflates every weight
// by about seven orders of magnitude, so the direction is pinned by the
// checkpoint's own scale magnitudes rather than by the algorithm name.
// `packed` is row-major [rows, columns/2] with the even column in the low
// nibble; `scales` is row-major [rows, columns/group_size] FP8 E4M3.
struct InklingNvfp4MatrixView {
    std::span<const std::byte> packed;
    std::span<const std::byte> scales;
    float global_scale{1.0F};
    std::uint64_t rows{};
    std::uint64_t columns{};
    std::uint64_t packed_columns{};
    std::uint64_t scale_columns{};
    std::uint32_t group_size{kInklingExecutionContract.nvfp4_group_size};
};

// One Inkling MXFP4 matrix in the MLX safetensors representation. The packed
// tensor is declared U32, eight low-nibble-first E2M1 codes per word; viewed as
// bytes it is the canonical two-codes-per-byte stream consumed by the backend.
// Scales are one E8M0 exponent byte per 32 logical columns.
struct InklingMxfp4MatrixView {
    std::span<const std::byte> packed;
    std::span<const std::byte> scales;
    std::uint64_t rows{};
    std::uint64_t columns{};
    std::uint64_t packed_columns{};
    std::uint64_t scale_columns{};
    std::uint32_t group_size{32U};
};

// The router's decision for one token. `experts` holds the selected routed
// expert ids followed by the shared sink ids, which are numbered from
// `routed_experts` upwards. `weights` is index-aligned and already carries the
// routed scale and the gate's learned global scale.
struct InklingRoute {
    std::vector<std::uint32_t> experts;
    std::vector<float> weights;
};

struct InklingRouteResult {
    InklingRoute value;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

// Scalar target-format oracle for NVFP4. Dequantizes exactly as the reference
// does and accumulates in FP32.
[[nodiscard]] ValidationResult inkling_nvfp4_matvec_reference(
    const InklingNvfp4MatrixView& matrix, std::span<const float> input,
    std::span<float> output);
[[nodiscard]] ValidationResult inkling_nvfp4_matvec_rows(
    const InklingNvfp4MatrixView& matrix, std::span<const float> input,
    std::span<float> output, std::uint64_t row_begin, std::uint64_t row_end);
[[nodiscard]] ValidationResult inkling_mxfp4_matvec_reference(
    const InklingMxfp4MatrixView& matrix, std::span<const float> input,
    std::span<float> output);
[[nodiscard]] ValidationResult inkling_mxfp4_matvec_rows(
    const InklingMxfp4MatrixView& matrix, std::span<const float> input,
    std::span<float> output, std::uint64_t row_begin, std::uint64_t row_end);

// Depthwise causal convolution over one channel-major stream, with the
// reference's residual:
//   y[t][c] = x[t][c] + sum_i w[c][i] * x[t - (W - 1) + i][c]
// Taps before the start of the sequence contribute nothing. `history` holds the
// W-1 rows preceding `input` in channel-major order, and may be empty at the
// start of a sequence. Accumulation is FP32, matching the reference, which
// keeps these modules in higher precision than the surrounding BF16.
[[nodiscard]] ValidationResult inkling_short_conv_f32(
    std::span<float> output, std::span<const float> input,
    std::span<const float> history, std::span<const float> weight,
    std::uint64_t tokens, std::uint64_t channels, std::uint32_t kernel);

// Long-context attention temperature. The reference clamps the ratio at one so
// the factor is exactly 1 below the floor, then applies it to the normalized
// queries and to the relative logits of global layers only.
[[nodiscard]] float inkling_log_scaling_tau(std::uint64_t position) noexcept;

// Projects one token's relative branch to per-distance logits:
//   bias[h][e] = sum_d r[h][d] * proj[d][e]
// `relative` is [heads, relative_dim] and `projection` is row-major
// [relative_dim, extent]. `tau` scales the result; pass 1 on local layers.
[[nodiscard]] ValidationResult inkling_relative_logits(
    std::span<float> output, std::span<const float> relative,
    std::span<const float> projection, std::uint32_t heads,
    std::uint32_t relative_dim, std::uint32_t extent, float tau);

// The pinned Inkling router. `logits` covers the routed experts followed by the
// shared sink experts. Selection scores are sigmoid(logit) + correction bias
// over the routed range only, with ties resolved to the lower index. The raw
// logits of the selected experts and of every sink are then renormalized
// together by log-sigmoid followed by a softmax over that joint set, and scaled
// by routed_scale * global_scale. The sinks are never selection candidates but
// always absorb probability mass, which is what makes a confident routed
// decision down-weight the shared experts.
[[nodiscard]] InklingRouteResult inkling_route_sigmoid_sink(
    std::span<const float> logits, std::span<const float> correction_bias,
    const RouterSpec& spec, std::uint32_t shared_experts, float global_scale);

// True when a query at `query` may attend to `key` under the layer's pattern.
// Local layers see the `sliding_window` most recent positions inclusive of
// themselves; global layers see the whole causal prefix.
[[nodiscard]] bool inkling_attention_visible(
    std::uint64_t query, std::uint64_t key, bool local,
    std::uint32_t sliding_window) noexcept;

// Interleaved SwiGLU over one fused gate/up row block. The checkpoint stores
// w13 as [g0, u0, g1, u1, ...], so the halves are strided rather than
// contiguous; reading it as two blocks silently mixes gate rows into up rows.
[[nodiscard]] ValidationResult inkling_interleaved_swiglu_f32(
    std::span<float> output, std::span<const float> gate_up);

}  // namespace strata
