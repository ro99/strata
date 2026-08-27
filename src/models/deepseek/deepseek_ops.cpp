#include "strata/models/deepseek/deepseek_ops.hpp"

#include "strata/platform/numerics.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>

#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define STRATA_MHC_AVX2 1
#define STRATA_MHC_TARGET_AVX2 __attribute__((target("avx2")))
#else
#define STRATA_MHC_AVX2 0
#define STRATA_MHC_TARGET_AVX2
#endif

namespace strata {

namespace {

[[nodiscard]] std::uint16_t encode_bf16(float value) noexcept {
    auto bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7F80'0000U) != 0x7F80'0000U) {
        bits += 0x7FFFU + ((bits >> 16U) & 1U);
    }
    return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] float round_bf16(float value) noexcept {
    return std::bit_cast<float>(
        static_cast<std::uint32_t>(encode_bf16(value)) << 16U);
}

[[nodiscard]] bool valid_router_spec(const RouterSpec& spec) noexcept {
    return spec.selection == RouterSelectionKind::NoAuxTc &&
           spec.scoring == RouterScoreKind::SqrtSoftplus &&
           spec.routed_experts > 0U && spec.experts_per_token > 0U &&
           spec.experts_per_token <= spec.routed_experts &&
           spec.groups == 1U && spec.selected_groups == 1U &&
           spec.normalize_topk && spec.selection_bias &&
           std::isfinite(spec.routed_scale) && spec.routed_scale > 0.0F;
}

[[nodiscard]] Dsv4RouteResult gather_route(
    std::span<const float> logits, std::span<const std::uint32_t> experts,
    const RouterSpec& spec) {
    Dsv4RouteResult result;
    result.value.experts.assign(experts.begin(), experts.end());
    result.value.weights.reserve(experts.size());
    float sum = 0.0F;
    for (const auto expert : experts) {
        if (expert >= logits.size()) {
            result.errors.emplace_back("DeepSeek route contains an out-of-range expert");
            result.value = {};
            return result;
        }
        const float score = std::sqrt(dsv4_softplus_f32(logits[expert]));
        if (!std::isfinite(score)) {
            result.errors.emplace_back("DeepSeek router score is not finite");
            result.value = {};
            return result;
        }
        result.value.weights.push_back(score);
        sum += score;
    }
    if (!std::isfinite(sum) || sum <= 0.0F) {
        result.errors.emplace_back("DeepSeek selected router scores do not normalize");
        result.value = {};
        return result;
    }
    for (auto& weight : result.value.weights) {
        weight = weight / sum * spec.routed_scale;
    }
    return result;
}

[[nodiscard]] ValidationResult route_sqrtsoftplus_into_impl(
    std::span<const float> logits, std::span<const float> selection_bias,
    const RouterSpec& spec, std::span<float> scratch,
    std::span<std::uint32_t> expert_ids, std::span<float> coefficients) {
    const auto fail = [&](const char* message) {
        std::fill(scratch.begin(), scratch.end(), 0.0F);
        std::fill(expert_ids.begin(), expert_ids.end(), 0U);
        std::fill(coefficients.begin(), coefficients.end(), 0.0F);
        ValidationResult result;
        result.errors.emplace_back(message);
        return result;
    };
    if (!valid_router_spec(spec))
        return fail("router specification is not the DeepSeek V4 sqrtsoftplus/noaux_tc contract");
    if (logits.size() != spec.routed_experts || selection_bias.size() != logits.size() ||
        scratch.size() != spec.routed_experts || expert_ids.size() != spec.experts_per_token ||
        coefficients.size() != spec.experts_per_token)
        return fail("DeepSeek router tensor shapes disagree with its contract");
    for (std::size_t expert = 0U; expert < logits.size(); ++expert) {
        if (!std::isfinite(logits[expert]) || !std::isfinite(selection_bias[expert]))
            return fail("DeepSeek router tensors contain a non-finite value");
        scratch[expert] = std::sqrt(dsv4_softplus_f32(logits[expert])) + selection_bias[expert];
        if (!std::isfinite(scratch[expert])) return fail("DeepSeek router score is not finite");
    }
    for (std::uint32_t rank = 0U; rank < spec.experts_per_token; ++rank) {
        bool found = false;
        std::uint32_t best = 0U;
        float best_score = -std::numeric_limits<float>::infinity();
        for (std::uint32_t expert = 0U; expert < spec.routed_experts; ++expert) {
            if (std::find(expert_ids.begin(), expert_ids.begin() + rank, expert) !=
                expert_ids.begin() + rank) continue;
            if (!found || scratch[expert] > best_score) {
                found = true; best = expert; best_score = scratch[expert];
            }
        }
        if (!found || best >= logits.size()) return fail("DeepSeek route contains an out-of-range expert");
        expert_ids[rank] = best;
    }
    float sum = 0.0F;
    for (std::size_t rank = 0U; rank < expert_ids.size(); ++rank) {
        const auto expert = expert_ids[rank];
        coefficients[rank] = std::sqrt(dsv4_softplus_f32(logits[expert]));
        if (!std::isfinite(coefficients[rank])) return fail("DeepSeek router score is not finite");
        sum += coefficients[rank];
    }
    if (!std::isfinite(sum) || sum <= 0.0F) return fail("DeepSeek selected router scores do not normalize");
    for (auto& coefficient : coefficients) {
        coefficient = coefficient / sum * spec.routed_scale;
        if (!std::isfinite(coefficient)) return fail("DeepSeek normalized router scores are not finite");
    }
    return {};
}

}  // namespace

float dsv4_softplus_f32(float value) noexcept {
    if (value > 20.0F) return value;
    if (value < -20.0F) return std::exp(value);
    return std::log1p(std::exp(value));
}

float dsv4_fp4_e2m1_f32(std::uint8_t nibble) noexcept {
    constexpr float values[16]{
        0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
        0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F};
    return values[nibble & 0x0FU];
}

ValidationResult dsv4_hadamard_rotate_f32(std::span<float> values) {
    ValidationResult result;
    if (values.empty() || (values.size() & (values.size() - 1U)) != 0U ||
        std::any_of(values.begin(), values.end(),
                    [](float value) { return !std::isfinite(value); })) {
        result.errors.emplace_back(
            "DeepSeek Hadamard rotation requires finite power-of-two input");
        return result;
    }
    for (std::size_t width = 1U; width < values.size(); width *= 2U) {
        for (std::size_t begin = 0U; begin < values.size(); begin += width * 2U) {
            for (std::size_t offset = 0U; offset < width; ++offset) {
                const float first = values[begin + offset];
                const float second = values[begin + width + offset];
                values[begin + offset] = first + second;
                values[begin + width + offset] = first - second;
            }
        }
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>(values.size()));
    for (auto& value : values) value = round_bf16(value * scale);
    return result;
}

ValidationResult dsv4_fp4_e2m1_simulate_f32(
    std::span<float> values, std::uint32_t group_size) {
    ValidationResult result;
    if (values.empty() || group_size == 0U || values.size() % group_size != 0U ||
        std::any_of(values.begin(), values.end(),
                    [](float value) { return !std::isfinite(value); })) {
        result.errors.emplace_back(
            "DeepSeek FP4 activation simulation has invalid groups or values");
        return result;
    }
    constexpr std::array<float, 8> magnitudes{
        0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    for (std::size_t begin = 0U; begin < values.size(); begin += group_size) {
        const auto group = values.subspan(begin, group_size);
        float maximum = 0.0F;
        for (const float value : group) {
            maximum = std::max(maximum, std::abs(value));
        }
        const float bounded = std::max(maximum, std::ldexp(6.0F, -126));
        const float scale = std::exp2(std::ceil(std::log2(bounded / 6.0F)));
        for (auto& value : group) {
            const float magnitude = std::min(std::abs(value / scale), 6.0F);
            std::size_t nearest = 0U;
            float distance = std::abs(magnitude - magnitudes[nearest]);
            for (std::size_t candidate = 1U; candidate < magnitudes.size();
                 ++candidate) {
                const float candidate_distance =
                    std::abs(magnitude - magnitudes[candidate]);
                if (candidate_distance < distance ||
                    (candidate_distance == distance &&
                     (candidate & 1U) == 0U && (nearest & 1U) != 0U)) {
                    nearest = candidate;
                    distance = candidate_distance;
                }
            }
            value = round_bf16(
                std::copysign(magnitudes[nearest] * scale, value));
        }
    }
    return result;
}

ValidationResult dsv4_index_scores_f32(
    std::span<float> scores, std::span<const float> queries,
    std::span<const float> keys, std::span<const float> weights,
    std::uint32_t heads, std::uint32_t head_dim) {
    ValidationResult result;
    const auto query_elements = static_cast<std::size_t>(heads) * head_dim;
    if (heads == 0U || head_dim == 0U || queries.size() != query_elements ||
        weights.size() != heads || scores.empty() ||
        keys.size() != scores.size() * static_cast<std::size_t>(head_dim) ||
        std::any_of(queries.begin(), queries.end(),
                    [](float value) { return !std::isfinite(value); }) ||
        std::any_of(keys.begin(), keys.end(),
                    [](float value) { return !std::isfinite(value); }) ||
        std::any_of(weights.begin(), weights.end(),
                    [](float value) { return !std::isfinite(value); })) {
        result.errors.emplace_back(
            "DeepSeek learned-index score tensors are invalid");
        return result;
    }
    for (std::size_t row = 0U; row < scores.size(); ++row) {
        const auto key = keys.subspan(row * head_dim, head_dim);
        float score = 0.0F;
        for (std::uint32_t head = 0U; head < heads; ++head) {
            const auto query = queries.subspan(
                static_cast<std::size_t>(head) * head_dim, head_dim);
            float dot = 0.0F;
            for (std::uint32_t dimension = 0U; dimension < head_dim;
                 ++dimension) {
                dot += query[dimension] * key[dimension];
            }
            dot = round_bf16(dot);
            score += round_bf16(weights[head] * std::max(0.0F, dot));
        }
        scores[row] = round_bf16(score);
    }
    return result;
}

Dsv4IndexSelectionResult dsv4_index_topk_f32(
    std::span<const float> scores, std::uint32_t top_k) {
    Dsv4IndexSelectionResult result;
    if (scores.empty() || top_k == 0U || top_k > scores.size() ||
        std::any_of(scores.begin(), scores.end(),
                    [](float value) { return !std::isfinite(value); })) {
        result.errors.emplace_back(
            "DeepSeek learned-index top-k scores are invalid");
        return result;
    }
    result.positions.resize(scores.size());
    std::iota(result.positions.begin(), result.positions.end(), 0U);
    const auto better = [&scores](std::uint32_t first, std::uint32_t second) {
        if (scores[first] != scores[second]) return scores[first] > scores[second];
        return first < second;
    };
    std::partial_sort(result.positions.begin(),
                      result.positions.begin() + top_k,
                      result.positions.end(), better);
    result.positions.resize(top_k);
    return result;
}

float dsv4_fp8_e4m3_f32(std::uint8_t encoded) noexcept {
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

float dsv4_fp8_e8m0_scale_f32(std::uint8_t encoded) noexcept {
    if (encoded == 0xFFU) return std::numeric_limits<float>::quiet_NaN();
    return std::ldexp(1.0F, static_cast<int>(encoded) - 127);
}

ValidationResult dsv4_fp8_e4m3_block128_to_bf16(
    std::span<std::uint16_t> output,
    std::span<const std::byte> weights,
    std::span<const std::byte> e8m0_scales,
    std::uint64_t rows, std::uint64_t columns) {
    ValidationResult result;
    constexpr std::uint64_t block = 128U;
    if (rows == 0U || columns == 0U ||
        rows > std::numeric_limits<std::uint64_t>::max() - (block - 1U) ||
        columns > std::numeric_limits<std::uint64_t>::max() - (block - 1U) ||
        rows > std::numeric_limits<std::uint64_t>::max() / columns) {
        result.errors.emplace_back("DeepSeek FP8-to-BF16 shape is invalid");
        return result;
    }
    const auto elements = rows * columns;
    const auto scale_rows = (rows + block - 1U) / block;
    const auto scale_columns = (columns + block - 1U) / block;
    if (scale_rows > std::numeric_limits<std::uint64_t>::max() / scale_columns ||
        output.size() != elements || weights.size() != elements ||
        e8m0_scales.size() != scale_rows * scale_columns) {
        result.errors.emplace_back("DeepSeek FP8-to-BF16 extents are incompatible");
        return result;
    }

    static const auto fp8_values = [] {
        std::array<float, 256> values{};
        for (std::size_t encoded = 0U; encoded < values.size(); ++encoded) {
            values[encoded] = dsv4_fp8_e4m3_f32(
                static_cast<std::uint8_t>(encoded));
        }
        return values;
    }();

    for (std::uint64_t block_row = 0U; block_row < rows; block_row += block) {
        const auto row_end = std::min(block_row + block, rows);
        for (std::uint64_t block_column = 0U; block_column < columns;
             block_column += block) {
            const auto column_end = std::min(block_column + block, columns);
            const auto scale_encoded = std::to_integer<std::uint8_t>(
                e8m0_scales[(block_row / block) * scale_columns +
                            block_column / block]);
            const float scale = dsv4_fp8_e8m0_scale_f32(scale_encoded);
            for (std::uint64_t row = block_row; row < row_end; ++row) {
                const auto row_offset = row * columns;
                for (std::uint64_t column = block_column;
                     column < column_end; ++column) {
                    const auto index = row_offset + column;
                    const auto encoded = std::to_integer<std::uint8_t>(
                        weights[static_cast<std::size_t>(index)]);
                    output[static_cast<std::size_t>(index)] = encode_bf16(
                        fp8_values[encoded] * scale);
                }
            }
        }
    }
    return result;
}

Dsv4RouteResult dsv4_route_sqrtsoftplus_f32(
    std::span<const float> logits, std::span<const float> selection_bias,
    const RouterSpec& spec) {
    Dsv4RouteResult result;
    if (!valid_router_spec(spec)) {
        result.errors.emplace_back(
            "router specification is not the DeepSeek V4 sqrtsoftplus/noaux_tc contract");
        return result;
    }
    if (logits.size() != spec.routed_experts || selection_bias.size() != logits.size()) {
        result.errors.emplace_back("DeepSeek router tensor shapes disagree with its contract");
        return result;
    }
    result.value.experts.resize(spec.experts_per_token);
    result.value.weights.resize(spec.experts_per_token);
    std::vector<float> scratch(spec.routed_experts);
    const auto validation = route_sqrtsoftplus_into_impl(
        logits, selection_bias, spec, scratch, result.value.experts,
        result.value.weights);
    if (!validation.ok()) {
        result.errors = validation.errors;
        result.value = {};
    }
    return result;
}

ValidationResult dsv4_route_sqrtsoftplus_f32_into(
    std::span<const float> logits, std::span<const float> selection_bias,
    const RouterSpec& spec, std::span<float> selection_scratch,
    std::span<std::uint32_t> expert_ids, std::span<float> coefficients) {
    return route_sqrtsoftplus_into_impl(logits, selection_bias, spec,
                                        selection_scratch, expert_ids, coefficients);
}

Dsv4RouteResult dsv4_route_hash_sqrtsoftplus_f32(
    std::span<const float> logits, std::span<const std::uint32_t> token_experts,
    const RouterSpec& spec) {
    Dsv4RouteResult result;
    if (!valid_router_spec(spec)) {
        result.errors.emplace_back(
            "router specification is not the DeepSeek V4 hash-routing contract");
        return result;
    }
    if (logits.size() != spec.routed_experts ||
        token_experts.size() != spec.experts_per_token) {
        result.errors.emplace_back("DeepSeek hash router tensor shapes disagree with its contract");
        return result;
    }
    for (const float logit : logits) {
        if (!std::isfinite(logit)) {
            result.errors.emplace_back("DeepSeek hash router logits contain a non-finite value");
            return result;
        }
    }
    return gather_route(logits, token_experts, spec);
}

ValidationResult dsv4_route_hash_sqrtsoftplus_f32_into(
    std::span<const float> logits, std::span<const std::uint32_t> token_experts,
    const RouterSpec& spec, std::span<std::uint32_t> expert_ids,
    std::span<float> coefficients) {
    const auto fail = [&](const char* message) {
        std::fill(expert_ids.begin(), expert_ids.end(), 0U);
        std::fill(coefficients.begin(), coefficients.end(), 0.0F);
        ValidationResult result;
        result.errors.emplace_back(message);
        return result;
    };
    if (!valid_router_spec(spec)) {
        return fail(
            "router specification is not the DeepSeek V4 hash-routing contract");
    }
    if (logits.size() != spec.routed_experts ||
        token_experts.size() != spec.experts_per_token ||
        expert_ids.size() != spec.experts_per_token ||
        coefficients.size() != spec.experts_per_token) {
        return fail("DeepSeek hash router tensor shapes disagree with its contract");
    }
    if (std::any_of(logits.begin(), logits.end(),
                    [](float value) { return !std::isfinite(value); })) {
        return fail("DeepSeek hash router logits contain a non-finite value");
    }
    float sum = 0.0F;
    for (std::size_t rank = 0U; rank < token_experts.size(); ++rank) {
        const auto expert = token_experts[rank];
        if (expert >= logits.size()) {
            return fail("DeepSeek hash route contains an invalid expert");
        }
        expert_ids[rank] = expert;
        coefficients[rank] = std::sqrt(dsv4_softplus_f32(logits[expert]));
        if (!std::isfinite(coefficients[rank])) {
            return fail("DeepSeek hash router score is not finite");
        }
        sum += coefficients[rank];
    }
    if (!std::isfinite(sum) || sum <= 0.0F) {
        return fail("DeepSeek selected hash router scores do not normalize");
    }
    for (auto& coefficient : coefficients) {
        coefficient = coefficient / sum * spec.routed_scale;
        if (!std::isfinite(coefficient)) {
            return fail("DeepSeek normalized hash router score is not finite");
        }
    }
    return {};
}

ValidationResult dsv4_swiglu_f32(std::span<float> output,
                                 std::span<const float> gate,
                                 std::span<const float> up,
                                 float limit) {
    ValidationResult result;
    if (output.empty() || output.size() != gate.size() || gate.size() != up.size()) {
        result.errors.emplace_back("DeepSeek SwiGLU spans have incompatible sizes");
        return result;
    }
    if (!std::isfinite(limit) || limit <= 0.0F) {
        result.errors.emplace_back("DeepSeek SwiGLU limit must be finite and positive");
        return result;
    }
    for (std::size_t index = 0U; index < output.size(); ++index) {
        if (!std::isfinite(gate[index]) || !std::isfinite(up[index])) {
            result.errors.emplace_back("DeepSeek SwiGLU input contains a non-finite value");
            return result;
        }
        const float limited_gate = std::min(gate[index], limit);
        const float limited_up = std::clamp(up[index], -limit, limit);
        output[index] = silu_f32(limited_gate) * limited_up;
    }
    return result;
}

ValidationResult dsv4_fp4_e2m1_matvec_f32(
    std::span<float> output, std::span<const float> input,
    std::span<const std::byte> packed_weights,
    std::span<const std::byte> e8m0_scales,
    std::uint64_t rows, std::uint64_t columns) {
    ValidationResult result;
    if (rows == 0U || columns == 0U || columns % 32U != 0U ||
        output.size() != rows || input.size() != columns ||
        packed_weights.size() != rows * columns / 2U ||
        e8m0_scales.size() != rows * columns / 32U) {
        result.errors.emplace_back("DeepSeek FP4 matvec layout is incompatible");
        return result;
    }
    for (std::uint64_t row = 0U; row < rows; ++row) {
        double sum = 0.0;
        for (std::uint64_t column = 0U; column < columns; ++column) {
            const auto packed = std::to_integer<std::uint8_t>(
                packed_weights[row * (columns / 2U) + column / 2U]);
            const auto nibble = static_cast<std::uint8_t>(
                column % 2U == 0U ? packed & 0x0FU : packed >> 4U);
            const auto scale = dsv4_fp8_e8m0_scale_f32(std::to_integer<std::uint8_t>(
                e8m0_scales[row * (columns / 32U) + column / 32U]));
            if (!std::isfinite(input[column]) || !std::isfinite(scale)) {
                result.errors.emplace_back("DeepSeek FP4 matvec contains a non-finite value");
                return result;
            }
            sum += static_cast<double>(input[column]) *
                   static_cast<double>(dsv4_fp4_e2m1_f32(nibble)) *
                   static_cast<double>(scale);
        }
        output[row] = static_cast<float>(sum);
    }
    return result;
}

ValidationResult dsv4_fp8_e4m3_matvec_f32(
    std::span<float> output, std::span<const float> input,
    std::span<const std::byte> weights,
    std::span<const std::byte> e8m0_scales,
    std::uint64_t rows, std::uint64_t columns) {
    ValidationResult result;
    const auto scale_rows = (rows + 127U) / 128U;
    const auto scale_columns = (columns + 127U) / 128U;
    if (rows == 0U || columns == 0U || output.size() != rows ||
        input.size() != columns || weights.size() != rows * columns ||
        e8m0_scales.size() != scale_rows * scale_columns) {
        result.errors.emplace_back("DeepSeek FP8 matvec layout is incompatible");
        return result;
    }
    for (std::uint64_t row = 0U; row < rows; ++row) {
        double sum = 0.0;
        for (std::uint64_t column = 0U; column < columns; ++column) {
            const auto weight = dsv4_fp8_e4m3_f32(std::to_integer<std::uint8_t>(
                weights[row * columns + column]));
            const auto scale = dsv4_fp8_e8m0_scale_f32(std::to_integer<std::uint8_t>(
                e8m0_scales[(row / 128U) * scale_columns + column / 128U]));
            if (!std::isfinite(input[column]) || !std::isfinite(weight) ||
                !std::isfinite(scale)) {
                result.errors.emplace_back("DeepSeek FP8 matvec contains a non-finite value");
                return result;
            }
            sum += static_cast<double>(input[column]) * static_cast<double>(weight) *
                   static_cast<double>(scale);
        }
        output[row] = static_cast<float>(sum);
    }
    return result;
}

Dsv4MhcMixResult dsv4_mhc_split_sinkhorn_f32(
    std::span<const float> mixes, std::span<const float> scale,
    std::span<const float> base, std::uint32_t multiplier,
    std::uint32_t iterations, float epsilon) {
    Dsv4MhcMixResult result;
    const auto mix_size = static_cast<std::size_t>(multiplier) * (2U + multiplier);
    if (multiplier == 0U || iterations == 0U || mixes.size() != mix_size ||
        base.size() != mix_size || scale.size() != 3U ||
        !std::isfinite(epsilon) || epsilon <= 0.0F) {
        result.errors.emplace_back("DeepSeek mHC Sinkhorn shapes or parameters are invalid");
        return result;
    }
    for (const auto values : {mixes, scale, base}) {
        if (!std::all_of(values.begin(), values.end(), [](float value) {
                return std::isfinite(value);
            })) {
            result.errors.emplace_back("DeepSeek mHC Sinkhorn input is not finite");
            return result;
        }
    }
    result.value.pre.resize(multiplier);
    result.value.post.resize(multiplier);
    result.value.combination.resize(static_cast<std::size_t>(multiplier) * multiplier);
    for (std::uint32_t index = 0U; index < multiplier; ++index) {
        result.value.pre[index] =
            sigmoid_f32(mixes[index] * scale[0] + base[index]) + epsilon;
        result.value.post[index] = 2.0F * sigmoid_f32(
            mixes[multiplier + index] * scale[1] + base[multiplier + index]);
    }
    const auto offset = static_cast<std::size_t>(2U * multiplier);
    for (std::uint32_t row = 0U; row < multiplier; ++row) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::uint32_t column = 0U; column < multiplier; ++column) {
            const auto index = static_cast<std::size_t>(row) * multiplier + column;
            auto& value = result.value.combination[index];
            value = mixes[offset + index] * scale[2] + base[offset + index];
            maximum = std::max(maximum, value);
        }
        float sum = 0.0F;
        for (std::uint32_t column = 0U; column < multiplier; ++column) {
            auto& value = result.value.combination[
                static_cast<std::size_t>(row) * multiplier + column];
            value = std::exp(value - maximum);
            sum += value;
        }
        for (std::uint32_t column = 0U; column < multiplier; ++column) {
            auto& value = result.value.combination[
                static_cast<std::size_t>(row) * multiplier + column];
            value = value / sum + epsilon;
        }
    }
    for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration) {
        if (iteration != 0U) {
            for (std::uint32_t row = 0U; row < multiplier; ++row) {
                float sum = 0.0F;
                for (std::uint32_t column = 0U; column < multiplier; ++column) {
                    sum += result.value.combination[
                        static_cast<std::size_t>(row) * multiplier + column];
                }
                for (std::uint32_t column = 0U; column < multiplier; ++column) {
                    result.value.combination[
                        static_cast<std::size_t>(row) * multiplier + column] /= sum + epsilon;
                }
            }
        }
        for (std::uint32_t column = 0U; column < multiplier; ++column) {
            float sum = 0.0F;
            for (std::uint32_t row = 0U; row < multiplier; ++row) {
                sum += result.value.combination[
                    static_cast<std::size_t>(row) * multiplier + column];
            }
            for (std::uint32_t row = 0U; row < multiplier; ++row) {
                result.value.combination[
                    static_cast<std::size_t>(row) * multiplier + column] /= sum + epsilon;
            }
        }
    }
    return result;
}

ValidationResult dsv4_mhc_pre_f32(
    std::span<float> reduced, Dsv4MhcMix& mix,
    std::span<const float> hidden_copies, std::span<const float> projection,
    std::span<const float> scale, std::span<const float> base,
    std::uint32_t multiplier, std::uint32_t iterations, float epsilon) {
    ValidationResult result;
    if (multiplier == 0U || hidden_copies.empty() ||
        hidden_copies.size() % multiplier != 0U ||
        reduced.size() != hidden_copies.size() / multiplier ||
        projection.size() != base.size() * hidden_copies.size()) {
        result.errors.emplace_back("DeepSeek mHC pre-projection shapes are incompatible");
        return result;
    }
    double square_sum = 0.0;
    for (const float value : hidden_copies) {
        if (!std::isfinite(value)) {
            result.errors.emplace_back("DeepSeek mHC hidden state is not finite");
            return result;
        }
        square_sum += static_cast<double>(value) * value;
    }
    const float reciprocal = 1.0F / std::sqrt(
        static_cast<float>(square_sum / static_cast<double>(hidden_copies.size())) + epsilon);
    std::vector<float> projected(base.size(), 0.0F);
    for (std::size_t row = 0U; row < projected.size(); ++row) {
        double sum = 0.0;
        for (std::size_t column = 0U; column < hidden_copies.size(); ++column) {
            sum += static_cast<double>(projection[row * hidden_copies.size() + column]) *
                   hidden_copies[column];
        }
        projected[row] = static_cast<float>(sum) * reciprocal;
    }
    auto split = dsv4_mhc_split_sinkhorn_f32(
        projected, scale, base, multiplier, iterations, epsilon);
    if (!split.ok()) {
        result.errors = std::move(split.errors);
        return result;
    }
    mix = std::move(split.value);
    std::fill(reduced.begin(), reduced.end(), 0.0F);
    for (std::uint32_t copy = 0U; copy < multiplier; ++copy) {
        for (std::size_t column = 0U; column < reduced.size(); ++column) {
            reduced[column] += mix.pre[copy] *
                hidden_copies[static_cast<std::size_t>(copy) * reduced.size() + column];
        }
    }
    return result;
}

bool dsv4_mhc_prepacked_supported() noexcept {
#if STRATA_MHC_AVX2
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

ValidationResult dsv4_pack_mhc_projection_f32(
    std::span<float> output, std::span<const float> row_major,
    std::size_t rows, std::size_t columns) {
    ValidationResult result;
    constexpr std::size_t width = 4U;
    if (rows == 0U || columns == 0U || rows % width != 0U ||
        row_major.size() != rows * columns || output.size() != row_major.size()) {
        result.errors.emplace_back(
            "DeepSeek mHC projection prepack spans are incompatible");
        return result;
    }
    for (std::size_t row = 0U; row < rows; row += width) {
        const std::size_t block = row / width;
        for (std::size_t column = 0U; column < columns; ++column) {
            for (std::size_t lane = 0U; lane < width; ++lane) {
                output[(block * columns + column) * width + lane] =
                    row_major[(row + lane) * columns + column];
            }
        }
    }
    return result;
}

#if STRATA_MHC_AVX2
// One block of four output rows, carrying its own accumulator. Each lane still
// folds columns in index order with the same mul and add, so the emitted floats
// do not move.
template <std::size_t Tile>
STRATA_MHC_TARGET_AVX2 static void dsv4_mhc_project_prepacked_tile_avx2(
    float* projected, const float* hidden, std::size_t columns,
    const float* packed, std::size_t first_block) {
    constexpr std::size_t width = 4U;
    std::array<__m256d, Tile> sums;
    std::array<const float*, Tile> rows{};
    for (std::size_t lane = 0U; lane < Tile; ++lane) {
        sums[lane] = _mm256_setzero_pd();
        rows[lane] = packed + (first_block + lane) * columns * width;
    }
    for (std::size_t column = 0U; column < columns; ++column) {
        const __m256d value = _mm256_set1_pd(static_cast<double>(hidden[column]));
        for (std::size_t lane = 0U; lane < Tile; ++lane) {
            const __m256d weights = _mm256_cvtps_pd(
                _mm_loadu_ps(rows[lane] + column * width));
            sums[lane] = _mm256_add_pd(sums[lane],
                                       _mm256_mul_pd(weights, value));
        }
    }
    for (std::size_t lane = 0U; lane < Tile; ++lane) {
        std::array<double, width> values{};
        _mm256_storeu_pd(values.data(), sums[lane]);
        for (std::size_t slot = 0U; slot < width; ++slot) {
            projected[(first_block + lane) * width + slot] =
                static_cast<float>(values[slot]);
        }
    }
}

// The projection is 24 rows against a 16,384-wide mHC state, so one block is a
// 16,384-long chain of dependent `vaddpd`. At the port's three-cycle latency
// that chain, not the arithmetic, sets the cost: measured 321 us a call and 86
// calls a decode step. Four blocks in flight cover the latency with issue slots
// that were idle, and because the blocks are independent output rows no lane's
// summation order changes.
STRATA_MHC_TARGET_AVX2
static void dsv4_mhc_project_prepacked_range_avx2(
    std::span<float> projected, std::span<const float> hidden,
    std::span<const float> packed, std::size_t first_block,
    std::size_t block_count) {
    const auto columns = hidden.size();
    std::size_t block = first_block;
    const std::size_t last = first_block + block_count;
    for (; block + 4U <= last; block += 4U) {
        dsv4_mhc_project_prepacked_tile_avx2<4U>(
            projected.data(), hidden.data(), columns, packed.data(), block);
    }
    for (; block + 2U <= last; block += 2U) {
        dsv4_mhc_project_prepacked_tile_avx2<2U>(
            projected.data(), hidden.data(), columns, packed.data(), block);
    }
    for (; block < last; ++block) {
        dsv4_mhc_project_prepacked_tile_avx2<1U>(
            projected.data(), hidden.data(), columns, packed.data(), block);
    }
}
#endif

ValidationResult dsv4_mhc_prepacked_f32(
    std::span<float> reduced, Dsv4MhcMix& mix,
    std::span<const float> hidden_copies,
    std::span<const float> packed_projection,
    std::span<const float> scale, std::span<const float> base,
    std::uint32_t multiplier, std::uint32_t iterations, float epsilon,
    const Dsv4ParallelFor& projection_lanes) {
    ValidationResult result;
    constexpr std::size_t width = 4U;
    if (!dsv4_mhc_prepacked_supported()) {
        result.errors.emplace_back("DeepSeek prepacked mHC requires x86 AVX2");
        return result;
    }
    if (multiplier == 0U || hidden_copies.empty() ||
        hidden_copies.size() % multiplier != 0U ||
        reduced.size() != hidden_copies.size() / multiplier ||
        base.empty() || base.size() % width != 0U ||
        packed_projection.size() != base.size() * hidden_copies.size()) {
        result.errors.emplace_back(
            "DeepSeek prepacked mHC spans are incompatible");
        return result;
    }
    double square_sum = 0.0;
    for (const float value : hidden_copies) {
        if (!std::isfinite(value)) {
            result.errors.emplace_back("DeepSeek mHC hidden state is not finite");
            return result;
        }
        square_sum += static_cast<double>(value) * value;
    }
    const float reciprocal = 1.0F / std::sqrt(
        static_cast<float>(square_sum / static_cast<double>(hidden_copies.size())) +
        epsilon);
    std::vector<float> projected(base.size());
#if STRATA_MHC_AVX2
    const std::size_t blocks = projected.size() / width;
    const std::size_t lanes = projection_lanes && blocks > 1U ? blocks : 1U;
    if (lanes > 1U) {
        auto dispatched = projection_lanes(lanes, [&](std::size_t lane) {
            const auto first = blocks * lane / lanes;
            const auto last = blocks * (lane + 1U) / lanes;
            if (last <= first) return;
            dsv4_mhc_project_prepacked_range_avx2(
                projected, hidden_copies, packed_projection, first,
                last - first);
        });
        if (!dispatched.ok()) return dispatched;
    } else {
        dsv4_mhc_project_prepacked_range_avx2(
            projected, hidden_copies, packed_projection, 0U, blocks);
    }
#endif
    for (auto& value : projected) value *= reciprocal;
    auto split = dsv4_mhc_split_sinkhorn_f32(
        projected, scale, base, multiplier, iterations, epsilon);
    if (!split.ok()) {
        result.errors = std::move(split.errors);
        return result;
    }
    mix = std::move(split.value);
    std::fill(reduced.begin(), reduced.end(), 0.0F);
    for (std::uint32_t copy = 0U; copy < multiplier; ++copy) {
        for (std::size_t column = 0U; column < reduced.size(); ++column) {
            reduced[column] += mix.pre[copy] * hidden_copies[
                static_cast<std::size_t>(copy) * reduced.size() + column];
        }
    }
    return result;
}

ValidationResult dsv4_mhc_post_f32(
    std::span<float> output_copies, std::span<const float> branch,
    std::span<const float> residual_copies, const Dsv4MhcMix& mix,
    std::uint32_t multiplier) {
    ValidationResult result;
    if (multiplier == 0U || branch.empty() ||
        output_copies.size() != branch.size() * multiplier ||
        residual_copies.size() != output_copies.size() ||
        mix.post.size() != multiplier ||
        mix.combination.size() != static_cast<std::size_t>(multiplier) * multiplier) {
        result.errors.emplace_back("DeepSeek mHC post-projection shapes are incompatible");
        return result;
    }
    for (std::uint32_t destination = 0U; destination < multiplier; ++destination) {
        for (std::size_t column = 0U; column < branch.size(); ++column) {
            float value = mix.post[destination] * branch[column];
            for (std::uint32_t source = 0U; source < multiplier; ++source) {
                // The target contracts comb as [source, destination].
                value += mix.combination[
                             static_cast<std::size_t>(source) * multiplier + destination] *
                         residual_copies[
                             static_cast<std::size_t>(source) * branch.size() + column];
            }
            output_copies[static_cast<std::size_t>(destination) * branch.size() + column] =
                value;
        }
    }
    return result;
}

}  // namespace strata
