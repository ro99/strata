#include "strata/attention.hpp"

#include <cmath>
#include <iterator>
#include <limits>

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

}  // namespace

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
         request.causal_key_counts.size() != request.query_rows) ||
        (!request.causal_key_counts.empty() &&
         !request.query_key_mask.empty())) {
        result.errors.emplace_back(
            "FlashAttention sink, causal-limit, or visibility storage has an incompatible shape");
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
    std::uint64_t mask_elements = 0U;
    if (!multiply(request.query_rows, result.value.logical_rows,
                  mask_elements) ||
        (!request.query_key_mask.empty() &&
         request.query_key_mask.size() != mask_elements)) {
        result.errors.emplace_back(
            "FlashAttention query visibility mask has an incompatible shape");
        return result;
    }
    if (!request.query_key_mask.empty()) {
        if (request.numerics == FlashAttentionNumerics::tiled_online_f64) {
            result.errors.emplace_back(
                "FlashAttention exact visibility masks require a compatibility contract");
            return result;
        }
        for (std::uint32_t query = 0U; query < request.query_rows; ++query) {
            const auto begin = request.query_key_mask.begin() +
                static_cast<std::ptrdiff_t>(query * result.value.logical_rows);
            const auto end = begin +
                static_cast<std::ptrdiff_t>(result.value.logical_rows);
            if (request.head_sinks.empty() &&
                std::none_of(begin, end, [](std::uint8_t value) {
                    return value != 0U;
                })) {
                result.errors.emplace_back(
                    "FlashAttention query visibility mask hides every normalization term");
                return result;
            }
        }
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
        !add(packed_elements,
             request.query_key_mask.size() / sizeof(float) +
                 (request.query_key_mask.size() % sizeof(float) != 0U),
             packed_elements) ||
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

}  // namespace strata
