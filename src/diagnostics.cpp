#include "strata/diagnostics.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

namespace strata {

namespace {

[[nodiscard]] std::uint64_t hash_u16(std::uint64_t hash,
                                     std::uint16_t value) noexcept {
    hash = diagnostic_hash_byte(hash, static_cast<std::uint8_t>(value & 0xFFU));
    return diagnostic_hash_byte(hash, static_cast<std::uint8_t>(value >> 8U));
}

[[nodiscard]] std::uint16_t encode_bf16(float value) noexcept {
    auto bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7F80'0000U) != 0x7F80'0000U) {
        bits += 0x7FFFU + ((bits >> 16U) & 1U);
    }
    return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] bool better_logit(const TopLogit& left,
                                const TopLogit& right) noexcept {
    const bool left_nan = std::isnan(left.raw_logit);
    const bool right_nan = std::isnan(right.raw_logit);
    if (left_nan != right_nan) return !left_nan;
    if (!left_nan && left.raw_logit != right.raw_logit) {
        return left.raw_logit > right.raw_logit;
    }
    return left.token_id < right.token_id;
}

}  // namespace

LogitAnalysis analyze_logits(std::span<const float> logits,
                             std::uint32_t top_k) {
    LogitAnalysis result;
    auto& summary = result.summary;
    summary.value_count = logits.size();
    summary.raw_f32_hash = kDiagnosticFnvOffset;

    std::vector<TopLogit> candidates;
    candidates.reserve(logits.size());
    for (std::size_t index = 0U; index < logits.size(); ++index) {
        const float value = logits[index];
        summary.raw_f32_hash = diagnostic_hash_u32(
            summary.raw_f32_hash, std::bit_cast<std::uint32_t>(value));
        candidates.push_back(
            {static_cast<std::uint32_t>(index), value});
        if (!std::isfinite(value)) {
            ++summary.non_finite_count;
            continue;
        }
        ++summary.finite_count;
        const double widened = value;
        summary.sum += widened;
        summary.absolute_sum += std::abs(widened);
        summary.square_sum += widened * widened;
        if (!summary.has_finite) {
            summary.minimum = value;
            summary.maximum = value;
            summary.has_finite = true;
        } else {
            summary.minimum = std::min(summary.minimum, value);
            summary.maximum = std::max(summary.maximum, value);
        }
    }

    const auto count = std::min<std::size_t>(top_k, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() +
                          static_cast<std::ptrdiff_t>(count),
                      candidates.end(), better_logit);
    candidates.resize(count);
    result.top = std::move(candidates);
    return result;
}

std::uint64_t stable_bf16_hash(std::span<const float> values) noexcept {
    std::uint64_t hash = kDiagnosticFnvOffset;
    for (const float value : values) hash = hash_u16(hash, encode_bf16(value));
    return hash;
}

std::uint64_t diagnostic_hash_byte(std::uint64_t hash,
                                   std::uint8_t value) noexcept {
    return (hash ^ value) * kDiagnosticFnvPrime;
}

std::uint64_t diagnostic_hash_u32(std::uint64_t hash,
                                  std::uint32_t value) noexcept {
    for (std::uint32_t shift = 0U; shift < 32U; shift += 8U) {
        hash = diagnostic_hash_byte(hash, static_cast<std::uint8_t>(value >> shift));
    }
    return hash;
}

std::uint64_t diagnostic_hash_u64(std::uint64_t hash,
                                  std::uint64_t value) noexcept {
    for (std::uint32_t shift = 0U; shift < 64U; shift += 8U) {
        hash = diagnostic_hash_byte(hash, static_cast<std::uint8_t>(value >> shift));
    }
    return hash;
}

}  // namespace strata
