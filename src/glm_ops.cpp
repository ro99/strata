#include "strata/glm_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace strata {

namespace {

void append_shape_error(ValidationResult& result, const char* operation) {
    result.errors.emplace_back(std::string(operation) + " spans have incompatible sizes");
}

}  // namespace

ValidationResult glm_layer_norm_f32(std::span<float> values,
                                    std::span<const float> weight,
                                    std::span<const float> bias,
                                    float epsilon) {
    ValidationResult result;
    if (values.empty() || values.size() != weight.size() || values.size() != bias.size()) {
        append_shape_error(result, "LayerNorm");
        return result;
    }
    if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
        result.errors.emplace_back("LayerNorm epsilon must be finite and positive");
        return result;
    }
    double mean = 0.0;
    for (const float value : values) {
        if (!std::isfinite(value)) {
            result.errors.emplace_back("LayerNorm input contains a non-finite value");
            return result;
        }
        mean += static_cast<double>(value);
    }
    mean /= static_cast<double>(values.size());
    double variance = 0.0;
    for (const float value : values) {
        const double difference = static_cast<double>(value) - mean;
        variance += difference * difference;
    }
    variance /= static_cast<double>(values.size());
    const float reciprocal = 1.0F / std::sqrt(static_cast<float>(variance) + epsilon);
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(weight[index]) || !std::isfinite(bias[index])) {
            result.errors.emplace_back("LayerNorm affine parameters contain a non-finite value");
            return result;
        }
        values[index] = static_cast<float>(static_cast<double>(values[index]) - mean) *
                            reciprocal * weight[index] +
                        bias[index];
    }
    return result;
}

ValidationResult glm_softmax_f32(std::span<float> values) {
    ValidationResult result;
    if (values.empty()) {
        result.errors.emplace_back("softmax input cannot be empty");
        return result;
    }
    float maximum = -std::numeric_limits<float>::infinity();
    for (const float value : values) {
        if (!std::isfinite(value)) {
            result.errors.emplace_back("softmax input contains a non-finite value");
            return result;
        }
        maximum = std::max(maximum, value);
    }
    float sum = 0.0F;
    for (float& value : values) {
        value = std::exp(value - maximum);
        sum += value;
    }
    if (!std::isfinite(sum) || sum <= 0.0F) {
        result.errors.emplace_back("softmax normalization is not finite and positive");
        return result;
    }
    for (float& value : values) value /= sum;
    return result;
}

ValidationResult glm_rope_interleaved_f32(std::span<float> values,
                                          std::uint64_t position,
                                          std::uint32_t rope_dimensions,
                                          float theta) {
    ValidationResult result;
    if (rope_dimensions == 0U || (rope_dimensions % 2U) != 0U ||
        rope_dimensions > values.size()) {
        result.errors.emplace_back("RoPE dimensions must be positive, even, and fit the input");
        return result;
    }
    if (!std::isfinite(theta) || theta <= 0.0F) {
        result.errors.emplace_back("RoPE theta must be finite and positive");
        return result;
    }
    std::vector<float> input(values.begin(), values.begin() + rope_dimensions);
    const auto half = rope_dimensions / 2U;
    for (std::uint32_t index = 0; index < half; ++index) {
        const float exponent = -2.0F * static_cast<float>(index) /
                               static_cast<float>(rope_dimensions);
        const float inverse_frequency = std::pow(theta, exponent);
        const float angle = static_cast<float>(position) * inverse_frequency;
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        const float first = input[2U * index];
        const float second = input[2U * index + 1U];
        values[index] = first * cosine - second * sine;
        values[half + index] = second * cosine + first * sine;
    }
    return result;
}

GlmRouteResult glm_route_logits_noaux_tc(std::span<const float> logits,
                                         std::span<const float> correction_bias,
                                         const RouterSpec& spec) {
    GlmRouteResult result;
    if (spec.selection != RouterSelectionKind::NoAuxTc ||
        spec.scoring != RouterScoreKind::Sigmoid || spec.groups != 1U ||
        spec.selected_groups != 1U || !spec.normalize_topk ||
        !spec.selection_bias) {
        result.errors.emplace_back("router specification is not the pinned GLM noaux_tc contract");
        return result;
    }
    if (logits.size() != spec.routed_experts || correction_bias.size() != logits.size() ||
        spec.experts_per_token == 0U || spec.experts_per_token > logits.size()) {
        result.errors.emplace_back("router tensor shapes disagree with its specification");
        return result;
    }
    if (!std::isfinite(spec.routed_scale) || spec.routed_scale <= 0.0F) {
        result.errors.emplace_back("router scale must be finite and positive");
        return result;
    }

    std::vector<float> scores(logits.size());
    std::vector<float> choices(logits.size());
    for (std::size_t expert = 0; expert < logits.size(); ++expert) {
        if (!std::isfinite(logits[expert]) || !std::isfinite(correction_bias[expert])) {
            result.errors.emplace_back("router tensors contain a non-finite value");
            return result;
        }
        scores[expert] = sigmoid_f32(logits[expert]);
        choices[expert] = scores[expert] + correction_bias[expert];
    }

    result.value.experts.reserve(spec.experts_per_token);
    result.value.weights.reserve(spec.experts_per_token);
    for (std::uint32_t rank = 0; rank < spec.experts_per_token; ++rank) {
        std::uint32_t best = 0U;
        float best_value = -std::numeric_limits<float>::infinity();
        bool found = false;
        for (std::uint32_t expert = 0; expert < spec.routed_experts; ++expert) {
            if (std::find(result.value.experts.begin(), result.value.experts.end(), expert) !=
                result.value.experts.end()) {
                continue;
            }
            if (!found || choices[expert] > best_value) {
                best = expert;
                best_value = choices[expert];
                found = true;
            }
        }
        result.value.experts.push_back(best);
        result.value.weights.push_back(scores[best]);
    }

    float sum = 1.0e-20F;
    for (const float weight : result.value.weights) sum += weight;
    for (float& weight : result.value.weights) weight = weight / sum * spec.routed_scale;
    return result;
}

GlmRouteResult glm_route_noaux_tc_f32(std::span<const float> hidden,
                                      std::span<const float> router_weight,
                                      std::span<const float> correction_bias,
                                      const RouterSpec& spec) {
    GlmRouteResult result;
    if (hidden.empty() || router_weight.size() != hidden.size() * spec.routed_experts) {
        result.errors.emplace_back("router projection tensor shapes are incompatible");
        return result;
    }
    std::vector<float> logits(spec.routed_experts, 0.0F);
    for (std::uint32_t expert = 0; expert < spec.routed_experts; ++expert) {
        const auto row = router_weight.subspan(static_cast<std::size_t>(expert) * hidden.size(),
                                               hidden.size());
        float sum = 0.0F;
        for (std::size_t column = 0; column < hidden.size(); ++column) {
            sum += row[column] * hidden[column];
        }
        logits[expert] = sum;
    }
    return glm_route_logits_noaux_tc(logits, correction_bias, spec);
}

}  // namespace strata
