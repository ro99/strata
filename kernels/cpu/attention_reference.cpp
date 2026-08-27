#include "strata/kernels/attention_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <new>
#include <vector>

namespace strata {

ValidationResult flash_attention_reference_f32(
    const FlashAttentionRequest& request, std::span<float> output) {
    ValidationResult result;
    auto shape = validate_flash_attention_request(request, output);
    if (!shape.ok()) {
        result.errors = std::move(shape.errors);
        return result;
    }
    constexpr std::size_t tile_rows = 32U;
    std::array<double, tile_rows> scores{};
    const auto heads_per_kv = request.query_heads / request.key_value_heads;
    std::vector<double> accumulator;
    std::vector<float> reference_scores;
    try {
        accumulator.resize(request.value_dim);
        if (request.numerics !=
            FlashAttentionNumerics::tiled_online_f64) {
            reference_scores.resize(static_cast<std::size_t>(
                shape.value.logical_rows));
        }
    } catch (const std::bad_alloc&) {
        result.errors.emplace_back(
            "FlashAttention scalar accumulator allocation failed within its workspace contract");
        return result;
    }

    for (std::uint32_t query_row = 0U; query_row < request.query_rows;
         ++query_row) {
        const bool masked = !request.query_key_mask.empty();
        const auto visible = request.causal_key_counts.empty()
            ? shape.value.logical_rows
            : request.causal_key_counts[query_row];
        const auto key_visible = [&](std::uint64_t logical_row) {
            return !masked || request.query_key_mask[
                static_cast<std::size_t>(query_row) *
                    shape.value.logical_rows + logical_row] != 0U;
        };
        for (std::uint32_t head = 0U; head < request.query_heads; ++head) {
            const auto kv_head = head / heads_per_kv;
            const auto query = request.queries.subspan(
                (static_cast<std::size_t>(query_row) * request.query_heads + head) *
                    request.query_key_dim,
                request.query_key_dim);
            auto destination = output.subspan(
                (static_cast<std::size_t>(query_row) * request.query_heads + head) *
                    request.value_dim,
                request.value_dim);
            std::fill(destination.begin(), destination.end(), 0.0F);
            if (request.numerics !=
                FlashAttentionNumerics::tiled_online_f64) {
                const bool f32_softmax = request.numerics ==
                    FlashAttentionNumerics::f32_dot_f32_softmax_f32_accum;
                float maximum = request.head_sinks.empty()
                    ? -std::numeric_limits<float>::infinity()
                    : request.head_sinks[head];
                std::size_t score_index = 0U;
                std::uint64_t logical_base = 0U;
                for (const auto& segment : request.segments) {
                    auto segment_info = segment_shape(request, segment);
                    const auto segment_visible = masked
                        ? segment_info.value.logical_rows
                        : visible <= logical_base
                        ? 0U
                        : std::min(segment_info.value.logical_rows,
                                   visible - logical_base);
                    const auto key_row_stride = static_cast<std::size_t>(
                        request.key_value_heads) * request.query_key_dim;
                    for (std::uint64_t logical_row = 0U;
                         logical_row < segment_visible; ++logical_row) {
                        if (!key_visible(logical_base + logical_row)) continue;
                        const auto source_row = segment.row_indices.empty()
                            ? logical_row
                            : segment.row_indices[
                                  static_cast<std::size_t>(logical_row)];
                        const auto key = segment.keys.subspan(
                            static_cast<std::size_t>(source_row) * key_row_stride +
                                static_cast<std::size_t>(kv_head) *
                                    request.query_key_dim,
                            request.query_key_dim);
                        float score = 0.0F;
                        if (f32_softmax) {
                            for (std::uint32_t dimension = 0U;
                                 dimension < request.query_key_dim; ++dimension) {
                                score += query[dimension] * key[dimension];
                            }
                            score *= request.scale;
                        } else {
                            double dot = 0.0;
                            for (std::uint32_t dimension = 0U;
                                 dimension < request.query_key_dim; ++dimension) {
                                dot += static_cast<double>(query[dimension]) *
                                       key[dimension];
                            }
                            score = static_cast<float>(dot) * request.scale;
                        }
                        if (!std::isfinite(score)) {
                            result.errors.emplace_back(
                                "FlashAttention scalar score is non-finite");
                            return result;
                        }
                        reference_scores[score_index++] = score;
                        maximum = std::max(maximum, score);
                    }
                    logical_base += segment_info.value.logical_rows;
                    if (!masked && logical_base >= visible) break;
                }
                double denominator = 0.0;
                float denominator_f32 = 0.0F;
                if (f32_softmax) {
                    if (!request.head_sinks.empty()) {
                        denominator_f32 = std::exp(
                            request.head_sinks[head] - maximum);
                    }
                    for (std::size_t item = 0U; item < score_index; ++item) {
                        reference_scores[item] = std::exp(
                            reference_scores[item] - maximum);
                        denominator_f32 += reference_scores[item];
                    }
                } else {
                    denominator = request.head_sinks.empty()
                        ? 0.0
                        : std::exp(static_cast<double>(
                              request.head_sinks[head] - maximum));
                    for (std::size_t item = 0U; item < score_index; ++item) {
                        denominator += std::exp(static_cast<double>(
                            reference_scores[item] - maximum));
                    }
                }
                if ((f32_softmax && (!std::isfinite(denominator_f32) ||
                                     denominator_f32 <= 0.0F)) ||
                    (!f32_softmax && (!std::isfinite(denominator) ||
                                      denominator <= 0.0))) {
                    result.errors.emplace_back(
                        "FlashAttention scalar softmax produced an invalid denominator");
                    return result;
                }
                std::size_t probability_index = 0U;
                logical_base = 0U;
                for (const auto& segment : request.segments) {
                    auto segment_info = segment_shape(request, segment);
                    const auto segment_visible = masked
                        ? segment_info.value.logical_rows
                        : visible <= logical_base
                        ? 0U
                        : std::min(segment_info.value.logical_rows,
                                   visible - logical_base);
                    const auto value_row_stride = static_cast<std::size_t>(
                        request.key_value_heads) * request.value_dim;
                    const auto& values = segment.values.empty()
                        ? segment.keys : segment.values;
                    for (std::uint64_t logical_row = 0U;
                         logical_row < segment_visible; ++logical_row) {
                        if (!key_visible(logical_base + logical_row)) continue;
                        const auto source_row = segment.row_indices.empty()
                            ? logical_row
                            : segment.row_indices[
                                  static_cast<std::size_t>(logical_row)];
                        const auto value = values.subspan(
                            static_cast<std::size_t>(source_row) * value_row_stride +
                                static_cast<std::size_t>(kv_head) * request.value_dim,
                            request.value_dim);
                        const auto score_position = probability_index++;
                        const float probability = f32_softmax
                            ? reference_scores[score_position] / denominator_f32
                            : static_cast<float>(std::exp(static_cast<double>(
                                  reference_scores[score_position] - maximum)) /
                              denominator);
                        for (std::uint32_t dimension = 0U;
                             dimension < request.value_dim; ++dimension) {
                            destination[dimension] +=
                                probability * value[dimension];
                        }
                    }
                    logical_base += segment_info.value.logical_rows;
                    if (!masked && logical_base >= visible) break;
                }
                if (!std::all_of(destination.begin(), destination.end(),
                                 [](float value) { return std::isfinite(value); })) {
                    result.errors.emplace_back(
                        "FlashAttention scalar output is non-finite");
                    return result;
                }
                continue;
            }
            std::fill(accumulator.begin(), accumulator.end(), 0.0);
            double maximum = request.head_sinks.empty()
                ? -std::numeric_limits<double>::infinity()
                : request.head_sinks[head];
            double denominator = request.head_sinks.empty() ? 0.0 : 1.0;
            std::uint64_t logical_base = 0U;
            for (const auto& segment : request.segments) {
                auto segment_info = segment_shape(request, segment);
                const auto segment_visible = visible <= logical_base
                    ? 0U
                    : std::min(segment_info.value.logical_rows,
                               visible - logical_base);
                const auto key_row_stride = static_cast<std::size_t>(
                    request.key_value_heads) * request.query_key_dim;
                const auto value_row_stride = static_cast<std::size_t>(
                    request.key_value_heads) * request.value_dim;
                for (std::uint64_t tile = 0U; tile < segment_visible;
                     tile += tile_rows) {
                    const auto count = std::min<std::uint64_t>(
                        tile_rows, segment_visible - tile);
                    double tile_maximum = -std::numeric_limits<double>::infinity();
                    for (std::uint64_t item = 0U; item < count; ++item) {
                        const auto logical_row = tile + item;
                        const auto source_row = segment.row_indices.empty()
                            ? logical_row
                            : segment.row_indices[static_cast<std::size_t>(logical_row)];
                        const auto key = segment.keys.subspan(
                            static_cast<std::size_t>(source_row) * key_row_stride +
                                static_cast<std::size_t>(kv_head) * request.query_key_dim,
                            request.query_key_dim);
                        double dot = 0.0;
                        for (std::uint32_t dimension = 0U;
                             dimension < request.query_key_dim; ++dimension) {
                            dot += static_cast<double>(query[dimension]) * key[dimension];
                        }
                        scores[static_cast<std::size_t>(item)] =
                            dot * static_cast<double>(request.scale);
                        tile_maximum = std::max(
                            tile_maximum, scores[static_cast<std::size_t>(item)]);
                    }
                    const double next_maximum = std::max(maximum, tile_maximum);
                    const double correction = denominator == 0.0
                        ? 0.0
                        : std::exp(maximum - next_maximum);
                    denominator *= correction;
                    for (auto& value : accumulator) value *= correction;
                    for (std::uint64_t item = 0U; item < count; ++item) {
                        const auto logical_row = tile + item;
                        const auto source_row = segment.row_indices.empty()
                            ? logical_row
                            : segment.row_indices[static_cast<std::size_t>(logical_row)];
                        const auto weight = std::exp(
                            scores[static_cast<std::size_t>(item)] - next_maximum);
                        denominator += weight;
                        const auto& values = segment.values.empty()
                            ? segment.keys : segment.values;
                        const auto value = values.subspan(
                            static_cast<std::size_t>(source_row) * value_row_stride +
                                static_cast<std::size_t>(kv_head) * request.value_dim,
                            request.value_dim);
                        for (std::uint32_t dimension = 0U;
                             dimension < request.value_dim; ++dimension) {
                            accumulator[dimension] += weight * value[dimension];
                        }
                    }
                    maximum = next_maximum;
                }
                logical_base += segment_info.value.logical_rows;
                if (logical_base >= visible) break;
            }
            if (!std::isfinite(denominator) || denominator <= 0.0) {
                result.errors.emplace_back(
                    "FlashAttention scalar softmax produced an invalid denominator");
                return result;
            }
            for (std::uint32_t dimension = 0U; dimension < request.value_dim;
                 ++dimension) {
                destination[dimension] = static_cast<float>(
                    accumulator[dimension] / denominator);
                if (!std::isfinite(destination[dimension])) {
                    result.errors.emplace_back(
                        "FlashAttention scalar output is non-finite");
                    return result;
                }
            }
        }
    }
    return result;
}

}  // namespace strata
