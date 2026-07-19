#pragma once

#include "strata/model.hpp"

#include <cstdint>
#include <span>

namespace strata {

// One logical K/V region. Empty row_indices means that every source row is
// attended in storage order. Otherwise, row_indices gathers exact source rows
// in the supplied order. Empty values aliases keys and is valid only when the
// query/key and value dimensions are equal.
struct FlashAttentionSegment {
    std::span<const float> keys;
    std::span<const float> values;
    std::span<const std::uint32_t> row_indices;
};

// The model adapter declares the numerical contract it already exposes. The
// online contract is the normal FlashAttention-2 recurrence. The compatibility
// compatibility contracts reproduce runtimes that use global softmax and F32
// value accumulation with either F64-dot/F64-softmax or all-F32 arithmetic.
enum class FlashAttentionNumerics : std::uint32_t {
    tiled_online_f64 = 0U,
    f64_dot_f32_score_f32_accum = 1U,
    f32_dot_f32_softmax_f32_accum = 2U,
};

// Production adapters may retain their exact scalar implementation below a
// measured CUDA crossover. This is a declared dispatch policy, not an error
// fallback and never changes the request's numerical contract.
[[nodiscard]] constexpr bool should_dispatch_flash_attention_cuda(
    bool enabled, std::uint64_t logical_rows,
    std::uint64_t minimum_cuda_rows) noexcept {
    return enabled && logical_rows >= minimum_cuda_rows;
}

// Model-neutral forward attention request. Queries and outputs use
// [query_rows, query_heads, dimension]. Segment storage uses
// [source_rows, key_value_heads, dimension]. causal_key_counts, when present,
// gives the visible prefix of the logically concatenated segments per query.
// A head sink is a virtual score with value zero.
struct FlashAttentionRequest {
    std::span<const float> queries;
    std::span<const FlashAttentionSegment> segments;
    std::span<const float> head_sinks;
    std::span<const std::uint32_t> causal_key_counts;
    std::uint32_t query_rows{};
    std::uint32_t query_heads{};
    std::uint32_t key_value_heads{};
    std::uint32_t query_key_dim{};
    std::uint32_t value_dim{};
    float scale{};
    FlashAttentionNumerics numerics{FlashAttentionNumerics::tiled_online_f64};
    std::uint64_t maximum_workspace_bytes{256ULL << 20U};
};

struct FlashAttentionShape {
    std::uint64_t logical_rows{};
    std::uint64_t packed_key_elements{};
    std::uint64_t packed_value_elements{};
    bool values_alias_keys{};
};

[[nodiscard]] ParseResult<FlashAttentionShape> validate_flash_attention_request(
    const FlashAttentionRequest& request, std::span<float> output);

// Scalar oracle for the request's declared numerical contract. Architecture
// adapters own any BF16 rounding boundaries before and after this operation.
[[nodiscard]] ValidationResult flash_attention_reference_f32(
    const FlashAttentionRequest& request, std::span<float> output);

}  // namespace strata
