#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace strata {

struct Dsv4TopLogit {
    std::uint32_t token_id{};
    float raw_logit{};
};

struct Dsv4LogitSummary {
    std::uint64_t value_count{};
    std::uint64_t finite_count{};
    std::uint64_t non_finite_count{};
    double sum{};
    double absolute_sum{};
    double square_sum{};
    float minimum{};
    float maximum{};
    std::uint64_t raw_f32_hash{};
    bool has_finite{};
};

struct Dsv4LogitAnalysis {
    Dsv4LogitSummary summary;
    std::vector<Dsv4TopLogit> top;
};

// Produces a deterministic diagnostic ordering without participating in token
// selection. Equal logits are ordered by ascending token id; NaNs sort last.
[[nodiscard]] Dsv4LogitAnalysis analyze_dsv4_logits(
    std::span<const float> logits, std::uint32_t top_k);

// FNV-1a over the little-endian bytes of each value after the runtime's
// round-to-nearest-even BF16 boundary. This is stable across host endianness.
[[nodiscard]] std::uint64_t dsv4_stable_bf16_hash(
    std::span<const float> values) noexcept;

struct Dsv4LogitTraceRecord {
    std::uint32_t position{};
    std::uint32_t input_token{};
    std::uint32_t selected_token{};
    Dsv4LogitSummary summary;
    std::vector<Dsv4TopLogit> top;
};

struct Dsv4LogitTraceAggregate {
    std::uint64_t forward_count{};
    std::uint64_t value_count{};
    std::uint64_t finite_count{};
    std::uint64_t non_finite_count{};
    double sum{};
    double absolute_sum{};
    double square_sum{};
    float minimum{};
    float maximum{};
    std::uint64_t trace_hash{};
    bool has_finite{};
};

struct Dsv4LayerHashTraceRecord {
    std::uint32_t position{};
    std::uint32_t input_token{};
    std::uint32_t layer{};
    std::uint64_t bf16_hash{};
};

struct Dsv4OperationHashTraceRecord {
    std::uint32_t position{};
    std::uint32_t input_token{};
    std::uint32_t layer{};
    std::string operation;
    std::uint64_t bf16_hash{};
};

struct Dsv4DiagnosticTrace {
    bool logit_trace_enabled{};
    bool layer_hash_trace_enabled{};
    std::uint32_t logit_top_k{20U};
    Dsv4LogitTraceAggregate logit_aggregate;
    std::vector<Dsv4LogitTraceRecord> logits;
    std::uint64_t layer_hash_trace_hash{};
    std::vector<Dsv4LayerHashTraceRecord> layer_hashes;
    std::vector<Dsv4OperationHashTraceRecord> operation_hashes;
    std::uint64_t index_selection_count{};
    std::uint64_t index_selection_trace_hash{};
};

}  // namespace strata
