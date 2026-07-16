#include "strata/deepseek_diagnostics.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

namespace strata {

namespace {

constexpr std::uint64_t kFnvOffset = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t kFnvPrime = 1'099'511'628'211ULL;

[[nodiscard]] std::uint64_t hash_byte(std::uint64_t hash,
                                      std::uint8_t value) noexcept {
    return (hash ^ value) * kFnvPrime;
}

[[nodiscard]] std::uint64_t hash_u16(std::uint64_t hash,
                                     std::uint16_t value) noexcept {
    hash = hash_byte(hash, static_cast<std::uint8_t>(value & 0xFFU));
    return hash_byte(hash, static_cast<std::uint8_t>(value >> 8U));
}

[[nodiscard]] std::uint64_t hash_u32(std::uint64_t hash,
                                     std::uint32_t value) noexcept {
    for (std::uint32_t shift = 0U; shift < 32U; shift += 8U) {
        hash = hash_byte(hash, static_cast<std::uint8_t>(value >> shift));
    }
    return hash;
}

[[nodiscard]] std::uint16_t encode_bf16(float value) noexcept {
    auto bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7F80'0000U) != 0x7F80'0000U) {
        bits += 0x7FFFU + ((bits >> 16U) & 1U);
    }
    return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] bool better_logit(const Dsv4TopLogit& left,
                                const Dsv4TopLogit& right) noexcept {
    const bool left_nan = std::isnan(left.raw_logit);
    const bool right_nan = std::isnan(right.raw_logit);
    if (left_nan != right_nan) return !left_nan;
    if (!left_nan && left.raw_logit != right.raw_logit) {
        return left.raw_logit > right.raw_logit;
    }
    return left.token_id < right.token_id;
}

}  // namespace

Dsv4LogitAnalysis analyze_dsv4_logits(std::span<const float> logits,
                                      std::uint32_t top_k) {
    Dsv4LogitAnalysis result;
    auto& summary = result.summary;
    summary.value_count = logits.size();
    summary.raw_f32_hash = kFnvOffset;

    std::vector<Dsv4TopLogit> candidates;
    candidates.reserve(logits.size());
    for (std::size_t index = 0U; index < logits.size(); ++index) {
        const float value = logits[index];
        summary.raw_f32_hash = hash_u32(
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

std::uint64_t dsv4_stable_bf16_hash(std::span<const float> values) noexcept {
    std::uint64_t hash = kFnvOffset;
    for (const float value : values) hash = hash_u16(hash, encode_bf16(value));
    return hash;
}

}  // namespace strata
