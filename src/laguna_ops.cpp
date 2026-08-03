#include "strata/laguna_ops.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

namespace strata {

namespace {

constexpr auto& kContract = kLagunaExecutionContract;

// E2M1 has sixteen exact values, so a table is both the fastest and the least
// error-prone decoder. The sign bit is the high nibble bit.
constexpr std::array<float, 16> kFp4E2m1{
    0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
    -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F};

double correction_dimension(double rotations, double dimensions, double base,
                            double original_context) noexcept {
    constexpr double two_pi = 6.283185307179586476925286766559;
    return (dimensions *
            std::log(original_context / (rotations * two_pi))) /
           (2.0 * std::log(base));
}

}  // namespace

float laguna_fp8_e4m3_f32(std::uint8_t encoded) noexcept {
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

float laguna_fp4_e2m1_f32(std::uint8_t nibble) noexcept {
    return kFp4E2m1[nibble & 0x0FU];
}

float laguna_softplus_f32(float value) noexcept {
    // log1p(exp(x)) overflows for large x while softplus(x) -> x, and the
    // reference relies on PyTorch's threshold form. Reproduce it so a large
    // gate does not become an infinity.
    if (value > 20.0F) return value;
    return std::log1p(std::exp(value));
}

ValidationResult laguna_nvfp4_matvec_rows(
    const LagunaNvfp4MatrixView& matrix, std::span<const float> input,
    std::span<float> output, std::uint64_t row_begin, std::uint64_t row_end) {
    ValidationResult result;
    if (matrix.group_size == 0U || matrix.columns == 0U || matrix.rows == 0U ||
        matrix.columns % 2U != 0U ||
        matrix.packed_columns != matrix.columns / 2U ||
        matrix.scale_columns !=
            (matrix.columns + matrix.group_size - 1U) / matrix.group_size) {
        result.errors.emplace_back("NVFP4 matrix layout is not the target format");
        return result;
    }
    if (!std::isfinite(matrix.global_scale) || matrix.global_scale <= 0.0F) {
        result.errors.emplace_back("NVFP4 global scale must be finite and positive");
        return result;
    }
    if (matrix.packed.size() != matrix.rows * matrix.packed_columns ||
        matrix.scales.size() != matrix.rows * matrix.scale_columns) {
        result.errors.emplace_back("NVFP4 matrix payload size mismatch");
        return result;
    }
    if (input.size() != matrix.columns || output.size() != matrix.rows) {
        result.errors.emplace_back("NVFP4 matvec activation shape mismatch");
        return result;
    }
    if (row_begin > row_end || row_end > matrix.rows) {
        result.errors.emplace_back("NVFP4 matvec row range is out of bounds");
        return result;
    }
    const auto* packed =
        reinterpret_cast<const std::uint8_t*>(matrix.packed.data());
    const auto* scales =
        reinterpret_cast<const std::uint8_t*>(matrix.scales.data());
    for (std::uint64_t row = row_begin; row < row_end; ++row) {
        const auto* packed_row = packed + row * matrix.packed_columns;
        const auto* scale_row = scales + row * matrix.scale_columns;
        float sum = 0.0F;
        for (std::uint64_t group = 0U; group < matrix.scale_columns; ++group) {
            const float scale =
                laguna_fp8_e4m3_f32(scale_row[group]) / matrix.global_scale;
            const auto begin = group * matrix.group_size;
            const auto end = std::min<std::uint64_t>(
                begin + matrix.group_size, matrix.columns);
            for (auto column = begin; column < end; ++column) {
                const auto byte = packed_row[column / 2U];
                const auto nibble = static_cast<std::uint8_t>(
                    column % 2U == 0U ? (byte & 0x0FU) : (byte >> 4U));
                sum += input[column] * laguna_fp4_e2m1_f32(nibble) * scale;
            }
        }
        output[row] = sum;
    }
    return result;
}

ValidationResult laguna_nvfp4_matvec_reference(
    const LagunaNvfp4MatrixView& matrix, std::span<const float> input,
    std::span<float> output) {
    return laguna_nvfp4_matvec_rows(matrix, input, output, 0U, matrix.rows);
}

LagunaRopeSchedule laguna_rope_schedule(bool global_layer) {
    LagunaRopeSchedule schedule;
    const float partial = global_layer ? kContract.global_rope_partial
                                       : kContract.sliding_rope_partial;
    const auto dimensions = static_cast<std::uint32_t>(
        static_cast<float>(kContract.head_dim) * partial);
    schedule.rotary_dimensions = dimensions & ~1U;
    const auto pairs = schedule.rotary_dimensions / 2U;
    schedule.inverse_frequencies.resize(pairs);
    const double base = global_layer ? kContract.global_rope_theta
                                     : kContract.sliding_rope_theta;
    if (!global_layer) {
        schedule.attention_scaling = 1.0F;
        for (std::uint32_t index = 0U; index < pairs; ++index) {
            const double exponent = static_cast<double>(2U * index) /
                                    static_cast<double>(schedule.rotary_dimensions);
            schedule.inverse_frequencies[index] =
                static_cast<float>(1.0 / std::pow(base, exponent));
        }
        return schedule;
    }

    // YaRN: interpolate the low-frequency dimensions by `factor`, extrapolate
    // the high-frequency ones, and ramp linearly between the correction bounds.
    schedule.attention_scaling = kContract.global_rope_attention_factor;
    const double factor = kContract.global_rope_factor;
    const double original = kContract.global_rope_original_context;
    const double dim = schedule.rotary_dimensions;
    double low = std::floor(correction_dimension(
        kContract.global_rope_beta_fast, dim, base, original));
    double high = std::ceil(correction_dimension(
        kContract.global_rope_beta_slow, dim, base, original));
    low = std::max(low, 0.0);
    high = std::min(high, dim - 1.0);
    if (low == high) high += 0.001;
    for (std::uint32_t index = 0U; index < pairs; ++index) {
        const double exponent =
            static_cast<double>(2U * index) / static_cast<double>(dim);
        const double positional = std::pow(base, exponent);
        const double extrapolation = 1.0 / positional;
        const double interpolation = 1.0 / (factor * positional);
        const double ramp = std::clamp(
            (static_cast<double>(index) - low) / (high - low), 0.0, 1.0);
        // ramp is the interpolation weight; 1 - ramp keeps extrapolation.
        const double extrapolation_weight = 1.0 - ramp;
        schedule.inverse_frequencies[index] = static_cast<float>(
            interpolation * (1.0 - extrapolation_weight) +
            extrapolation * extrapolation_weight);
    }
    return schedule;
}

ValidationResult laguna_rope_half_f32(std::span<float> head,
                                      std::uint64_t position,
                                      const LagunaRopeSchedule& schedule) {
    ValidationResult result;
    const auto rotary = schedule.rotary_dimensions;
    if (rotary == 0U || head.size() < rotary ||
        schedule.inverse_frequencies.size() != rotary / 2U) {
        result.errors.emplace_back("Laguna rotary shape mismatch");
        return result;
    }
    const auto half = rotary / 2U;
    for (std::uint32_t index = 0U; index < half; ++index) {
        const double angle = static_cast<double>(position) *
                             static_cast<double>(schedule.inverse_frequencies[index]);
        const auto cosine = static_cast<float>(std::cos(angle)) *
                            schedule.attention_scaling;
        const auto sine = static_cast<float>(std::sin(angle)) *
                          schedule.attention_scaling;
        // rotate_half pairs value i with value i + rotary/2 over the rotated
        // prefix. This is not the interleaved convention GLM uses.
        const float first = head[index];
        const float second = head[half + index];
        head[index] = first * cosine - second * sine;
        head[half + index] = second * cosine + first * sine;
    }
    return result;
}

LagunaRouteResult laguna_route_sigmoid_topk(
    std::span<const float> logits, std::span<const float> correction_bias,
    const RouterSpec& spec, float logit_softcapping) {
    LagunaRouteResult result;
    if (spec.selection != RouterSelectionKind::TopK ||
        spec.scoring != RouterScoreKind::Sigmoid || !spec.selection_bias) {
        result.errors.emplace_back("router spec is not the Laguna sigmoid top-k rule");
        return result;
    }
    if (spec.routed_experts == 0U || spec.experts_per_token == 0U ||
        spec.experts_per_token > spec.routed_experts) {
        result.errors.emplace_back("router expert counts are out of range");
        return result;
    }
    if (logits.size() != spec.routed_experts ||
        correction_bias.size() != spec.routed_experts) {
        result.errors.emplace_back("router logit or correction bias shape mismatch");
        return result;
    }
    std::vector<float> scores(spec.routed_experts);
    std::vector<float> selection(spec.routed_experts);
    for (std::uint32_t expert = 0U; expert < spec.routed_experts; ++expert) {
        float logit = logits[expert];
        if (!std::isfinite(logit)) {
            result.errors.emplace_back("router logit is not finite");
            return result;
        }
        if (logit_softcapping > 0.0F) {
            logit = std::tanh(logit / logit_softcapping) * logit_softcapping;
        }
        scores[expert] = sigmoid_f32(logit);
        selection[expert] = scores[expert] + correction_bias[expert];
    }
    std::vector<std::uint32_t> order(spec.routed_experts);
    std::iota(order.begin(), order.end(), 0U);
    const auto better = [&selection](std::uint32_t left, std::uint32_t right) {
        if (selection[left] != selection[right]) {
            return selection[left] > selection[right];
        }
        return left < right;
    };
    std::partial_sort(order.begin(),
                      order.begin() + spec.experts_per_token,
                      order.end(), better);
    order.resize(spec.experts_per_token);
    result.value.experts = order;
    result.value.weights.resize(spec.experts_per_token);
    float total = 0.0F;
    for (std::uint32_t rank = 0U; rank < spec.experts_per_token; ++rank) {
        result.value.weights[rank] = scores[order[rank]];
        total += result.value.weights[rank];
    }
    if (spec.normalize_topk) {
        if (!(total > 0.0F) || !std::isfinite(total)) {
            result.errors.emplace_back("router top-k normalizer is not positive");
            result.value = {};
            return result;
        }
        for (auto& weight : result.value.weights) weight /= total;
    }
    return result;
}

bool laguna_attention_visible(std::uint64_t query, std::uint64_t key,
                              bool sliding,
                              std::uint32_t sliding_window) noexcept {
    if (key > query) return false;
    if (!sliding) return true;
    return query - key < sliding_window;
}

}  // namespace strata
