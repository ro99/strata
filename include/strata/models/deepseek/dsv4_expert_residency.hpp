#pragma once

#include "strata/platform/result.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace strata {

// The static routed-expert residency tier: which (layer, expert) triplets live
// permanently in one device's VRAM instead of host DRAM.
//
// Chosen offline by `strata-dsv4-expert-residency` from decode route traces,
// because decode routing is concentrated in a way that belongs to the model
// rather than to one conversation -- the hottest 10% of triplets chosen from
// one prompt cover 38.6% of a different prompt's decode activations against a
// 10.4% random null (experiment 0124). A tier this small therefore pays for
// itself without a replacement policy.
//
// The set never changes for the life of the process. That is not a limitation
// but the point: routing is decided inside a `cudaLaunchHostFunc` callback
// where CUDA calls are forbidden, so a tier whose contents could change would
// have to publish them to the device on every change. A fixed set can be a
// constant bitmap the device tests for itself.
//
// This is placement, not prediction. It selects where a weight lives, never
// which experts the router picks, so the charter's advisory-predictor rule is
// not engaged and no fallback path exists to diverge.
class Dsv4ExpertResidencyPlan {
public:
    Dsv4ExpertResidencyPlan() = default;

    // Parses the format `strata-dsv4-expert-residency` emits. Fails rather
    // than admitting a partial plan: a residency map that silently disagrees
    // with the device's contents would produce experts computed twice or not
    // at all.
    [[nodiscard]] static ParseResult<Dsv4ExpertResidencyPlan> parse(
        std::string_view text, std::uint32_t layers, std::uint32_t experts);
    [[nodiscard]] static ParseResult<Dsv4ExpertResidencyPlan> load(
        const std::string& path, std::uint32_t layers, std::uint32_t experts);

    // Hot on the decode path: one bounds-checked bit test.
    [[nodiscard]] bool resident(std::uint32_t layer,
                                std::uint32_t expert) const noexcept {
        const auto flat =
            static_cast<std::size_t>(layer) * experts_ + expert;
        return layer < layers_ && expert < experts_ && flat < bitmap_.size() &&
               bitmap_[flat] != 0U;
    }

    [[nodiscard]] std::size_t size() const noexcept { return pairs_.size(); }
    [[nodiscard]] bool empty() const noexcept { return pairs_.empty(); }
    [[nodiscard]] std::uint64_t bytes(std::uint64_t triplet_bytes) const noexcept {
        return static_cast<std::uint64_t>(pairs_.size()) * triplet_bytes;
    }
    // Ordered hottest first, which is the order to admit them in when the
    // device cannot hold the whole plan.
    [[nodiscard]] const std::vector<std::pair<std::uint32_t, std::uint32_t>>&
    pairs() const noexcept { return pairs_; }

    // Drops everything past `limit`, keeping the hottest. Returns what remains.
    std::size_t truncate(std::size_t limit);

    // Keeps every `stride`-th entry starting at `offset`, so N devices each
    // take a disjoint slice of one ranking and the hottest triplets spread
    // evenly rather than piling onto one card. Returns what remains.
    std::size_t slice(std::size_t offset, std::size_t stride);

private:
    std::uint32_t layers_{};
    std::uint32_t experts_{};
    std::vector<std::uint8_t> bitmap_;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> pairs_;
};

}  // namespace strata
