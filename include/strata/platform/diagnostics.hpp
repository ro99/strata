#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace strata {

struct TopLogit {
    std::uint32_t token_id{};
    float raw_logit{};
};

struct LogitSummary {
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

struct LogitAnalysis {
    LogitSummary summary;
    std::vector<TopLogit> top;
};

// Produces a deterministic diagnostic ordering without participating in token
// selection. Equal logits are ordered by ascending token id; NaNs sort last.
[[nodiscard]] LogitAnalysis analyze_logits(
    std::span<const float> logits, std::uint32_t top_k);

// FNV-1a over the little-endian bytes of each value after the runtime's
// round-to-nearest-even BF16 boundary. This is stable across host endianness.
[[nodiscard]] std::uint64_t stable_bf16_hash(
    std::span<const float> values) noexcept;

// FNV-1a basis and combinators for folding discrete diagnostic fields
// (position, token, layer, a record's own bf16_hash, ...) into a single
// rolling trace_hash aggregate. stable_bf16_hash uses the same algorithm
// internally for tensor values; these are the byte-at-a-time primitives a
// model's own diagnostic wiring combines scalar metadata with, the same way
// for every model rather than each reimplementing FNV-1a privately.
inline constexpr std::uint64_t kDiagnosticFnvOffset =
    14'695'981'039'346'656'037ULL;
inline constexpr std::uint64_t kDiagnosticFnvPrime = 1'099'511'628'211ULL;

[[nodiscard]] std::uint64_t diagnostic_hash_byte(
    std::uint64_t hash, std::uint8_t value) noexcept;
[[nodiscard]] std::uint64_t diagnostic_hash_u32(
    std::uint64_t hash, std::uint32_t value) noexcept;
[[nodiscard]] std::uint64_t diagnostic_hash_u64(
    std::uint64_t hash, std::uint64_t value) noexcept;

struct LogitTraceRecord {
    std::uint32_t position{};
    std::uint32_t input_token{};
    std::uint32_t selected_token{};
    LogitSummary summary;
    std::vector<TopLogit> top;
};

struct LogitTraceAggregate {
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

struct LayerHashTraceRecord {
    std::uint32_t position{};
    std::uint32_t input_token{};
    std::uint32_t layer{};
    std::uint64_t bf16_hash{};
};

struct OperationHashTraceRecord {
    std::uint32_t position{};
    std::uint32_t input_token{};
    std::uint32_t layer{};
    std::string operation;
    std::uint64_t bf16_hash{};
};

struct DiagnosticTrace {
    bool logit_trace_enabled{};
    bool layer_hash_trace_enabled{};
    std::uint32_t logit_top_k{20U};
    LogitTraceAggregate logit_aggregate;
    std::vector<LogitTraceRecord> logits;
    std::uint64_t layer_hash_trace_hash{};
    std::vector<LayerHashTraceRecord> layer_hashes;
    // Mirrors layer_hash_trace_hash: a rolling aggregate over every recorded
    // operation, so a caller can compare one hash instead of serializing the
    // whole list. operation_hashes has no per-model bound on entry count the
    // way layer_hashes does (layers x positions) -- it can be several times
    // larger (four per layer for Gemma 4's current wiring) -- which is
    // exactly why an aggregate matters here: the full list is still recorded
    // for localisation, but routine comparison need not serialize it.
    std::uint64_t operation_hash_trace_hash{};
    std::vector<OperationHashTraceRecord> operation_hashes;
    std::uint64_t index_selection_count{};
    std::uint64_t index_selection_trace_hash{};
};

}  // namespace strata
