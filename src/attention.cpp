#include "strata/attention.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <new>
#include <vector>

namespace strata {

namespace {

bool multiply(std::uint64_t left, std::uint64_t right,
              std::uint64_t& output) noexcept {
    if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    output = left * right;
    return true;
}

bool add(std::uint64_t left, std::uint64_t right,
         std::uint64_t& output) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) return false;
    output = left + right;
    return true;
}

struct SegmentShape {
    std::uint64_t source_rows{};
    std::uint64_t logical_rows{};
};

ParseResult<SegmentShape> segment_shape(const FlashAttentionRequest& request,
                                        const FlashAttentionSegment& segment) {
    ParseResult<SegmentShape> result;
    std::uint64_t key_row_elements = 0U;
    if (!multiply(request.key_value_heads, request.query_key_dim,
                  key_row_elements) || key_row_elements == 0U ||
        segment.keys.size() % key_row_elements != 0U) {
        result.errors.emplace_back(
            "FlashAttention segment key storage has an incompatible shape");
        return result;
    }
    result.value.source_rows = segment.keys.size() / key_row_elements;
    result.value.logical_rows = segment.row_indices.empty()
        ? result.value.source_rows
        : segment.row_indices.size();
    if (result.value.source_rows == 0U && result.value.logical_rows != 0U) {
        result.errors.emplace_back(
            "FlashAttention indexed segment has no source rows");
        return result;
    }
    for (const auto row : segment.row_indices) {
        if (row >= result.value.source_rows) {
            result.errors.emplace_back(
                "FlashAttention segment row index is out of range");
            return result;
        }
    }
    if (segment.values.empty()) {
        if (request.query_key_dim != request.value_dim) {
            result.errors.emplace_back(
                "FlashAttention values may alias keys only for equal dimensions");
        }
        return result;
    }
    std::uint64_t value_row_elements = 0U;
    std::uint64_t expected_value_elements = 0U;
    if (!multiply(request.key_value_heads, request.value_dim,
                  value_row_elements) || value_row_elements == 0U ||
        !multiply(result.value.source_rows, value_row_elements,
                  expected_value_elements) ||
        segment.values.size() != expected_value_elements) {
        result.errors.emplace_back(
            "FlashAttention segment value storage has an incompatible shape");
    }
    return result;
}

}  // namespace

ParseResult<FlashAttentionShape> validate_flash_attention_request(
    const FlashAttentionRequest& request, std::span<float> output) {
    ParseResult<FlashAttentionShape> result;
    if (request.query_rows == 0U || request.query_heads == 0U ||
        request.key_value_heads == 0U || request.query_key_dim == 0U ||
        request.value_dim == 0U ||
        request.query_heads % request.key_value_heads != 0U) {
        result.errors.emplace_back(
            "FlashAttention dimensions and head mapping must be positive and compatible");
        return result;
    }
    if (!std::isfinite(request.scale) || request.scale <= 0.0F) {
        result.errors.emplace_back("FlashAttention scale must be positive and finite");
        return result;
    }
    if (request.numerics != FlashAttentionNumerics::tiled_online_f64 &&
        request.numerics !=
            FlashAttentionNumerics::f64_dot_f32_score_f32_accum &&
        request.numerics !=
            FlashAttentionNumerics::f32_dot_f32_softmax_f32_accum) {
        result.errors.emplace_back(
            "FlashAttention numerical contract is unsupported");
        return result;
    }
    std::uint64_t query_elements = 0U;
    std::uint64_t output_elements = 0U;
    if (!multiply(request.query_rows, request.query_heads, query_elements) ||
        !multiply(query_elements, request.query_key_dim, query_elements) ||
        request.queries.size() != query_elements ||
        !multiply(request.query_rows, request.query_heads, output_elements) ||
        !multiply(output_elements, request.value_dim, output_elements) ||
        output.size() != output_elements) {
        result.errors.emplace_back(
            "FlashAttention query or output storage has an incompatible shape");
        return result;
    }
    if ((!request.head_sinks.empty() &&
         request.head_sinks.size() != request.query_heads) ||
        (!request.causal_key_counts.empty() &&
         request.causal_key_counts.size() != request.query_rows)) {
        result.errors.emplace_back(
            "FlashAttention sink or causal-limit storage has an incompatible shape");
        return result;
    }
    for (const auto sink : request.head_sinks) {
        if (!std::isfinite(sink)) {
            result.errors.emplace_back("FlashAttention sink must be finite");
            return result;
        }
    }
    if (request.segments.empty() && request.head_sinks.empty()) {
        result.errors.emplace_back(
            "FlashAttention requires at least one key row or an attention sink");
        return result;
    }

    result.value.values_alias_keys = true;
    for (const auto& segment : request.segments) {
        auto shape = segment_shape(request, segment);
        if (!shape.ok()) {
            result.errors.insert(result.errors.end(),
                                 std::make_move_iterator(shape.errors.begin()),
                                 std::make_move_iterator(shape.errors.end()));
            return result;
        }
        if (!add(result.value.logical_rows, shape.value.logical_rows,
                 result.value.logical_rows)) {
            result.errors.emplace_back("FlashAttention logical row count overflows");
            return result;
        }
        std::uint64_t keys = 0U;
        std::uint64_t values = 0U;
        if (!multiply(shape.value.logical_rows, request.key_value_heads, keys) ||
            !multiply(keys, request.query_key_dim, keys) ||
            !add(result.value.packed_key_elements, keys,
                 result.value.packed_key_elements) ||
            !multiply(shape.value.logical_rows, request.key_value_heads, values) ||
            !multiply(values, request.value_dim, values) ||
            !add(result.value.packed_value_elements, values,
                 result.value.packed_value_elements)) {
            result.errors.emplace_back("FlashAttention packed storage size overflows");
            return result;
        }
        result.value.values_alias_keys = result.value.values_alias_keys &&
            segment.values.empty();
    }
    if (result.value.logical_rows == 0U && request.head_sinks.empty()) {
        result.errors.emplace_back(
            "FlashAttention requires at least one visible normalization term");
        return result;
    }
    if (result.value.values_alias_keys &&
        result.value.packed_key_elements != result.value.packed_value_elements) {
        result.errors.emplace_back("FlashAttention aliased packed dimensions disagree");
        return result;
    }
    for (const auto limit : request.causal_key_counts) {
        if (limit > result.value.logical_rows) {
            result.errors.emplace_back(
                "FlashAttention causal key count exceeds the logical rows");
            return result;
        }
        if (limit == 0U && request.head_sinks.empty()) {
            result.errors.emplace_back(
                "FlashAttention query has no visible normalization term");
            return result;
        }
    }
    std::uint64_t packed_elements = result.value.packed_key_elements;
    const auto packed_values = result.value.values_alias_keys
        ? 0U : result.value.packed_value_elements;
    std::uint64_t score_elements = 0U;
    if (request.numerics ==
            FlashAttentionNumerics::f64_dot_f32_score_f32_accum &&
        (!multiply(request.query_rows, request.query_heads, score_elements) ||
         !multiply(score_elements, result.value.logical_rows,
                   score_elements))) {
        result.errors.emplace_back(
            "FlashAttention score scratch size overflows");
        return result;
    }
    if (!add(packed_elements, packed_values, packed_elements) ||
        !add(packed_elements, query_elements, packed_elements) ||
        !add(packed_elements, output_elements, packed_elements) ||
        !add(packed_elements, score_elements, packed_elements) ||
        !add(packed_elements, request.head_sinks.size(), packed_elements) ||
        !add(packed_elements, request.causal_key_counts.size(), packed_elements) ||
        !add(packed_elements, 1U, packed_elements)) {
        result.errors.emplace_back("FlashAttention total workspace size overflows");
        return result;
    }
    if (packed_elements > request.maximum_workspace_bytes / sizeof(float)) {
        result.errors.emplace_back(
            "FlashAttention request exceeds its bounded workspace contract");
    }
    return result;
}

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
        const auto visible = request.causal_key_counts.empty()
            ? shape.value.logical_rows
            : request.causal_key_counts[query_row];
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
                    const auto segment_visible = visible <= logical_base
                        ? 0U
                        : std::min(segment_info.value.logical_rows,
                                   visible - logical_base);
                    const auto key_row_stride = static_cast<std::size_t>(
                        request.key_value_heads) * request.query_key_dim;
                    for (std::uint64_t logical_row = 0U;
                         logical_row < segment_visible; ++logical_row) {
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
                    if (logical_base >= visible) break;
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
                    const auto segment_visible = visible <= logical_base
                        ? 0U
                        : std::min(segment_info.value.logical_rows,
                                   visible - logical_base);
                    const auto value_row_stride = static_cast<std::size_t>(
                        request.key_value_heads) * request.value_dim;
                    const auto& values = segment.values.empty()
                        ? segment.keys : segment.values;
                    for (std::uint64_t logical_row = 0U;
                         logical_row < segment_visible; ++logical_row) {
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
                    if (logical_base >= visible) break;
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
