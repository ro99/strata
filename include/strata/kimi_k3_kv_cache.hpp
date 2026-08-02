#pragma once

#include "strata/model_adapter.hpp"
#include "strata/result.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace strata {

// Kimi-K3's per-sequence state has two halves that scale differently, and
// keeping them in one object is what makes the difference visible.
//
// The 24 gated MLA layers hold a compressed latent plus one shared RoPE-slot
// row per token: (kv_lora_rank + qk_rope_head_dim) values, growing with
// context. The 69 KDA layers hold a fixed [heads, key_dim, value_dim]
// recurrent state and a short-convolution history of `kernel - 1` rows, both
// **independent of context length**. That asymmetry is why a million-token
// context is not the constraint the layer count suggests: at 1M tokens the MLA
// half is the whole cost and the KDA half has not moved.
struct KimiCacheFootprint {
    std::uint64_t latent_bytes{};
    std::uint64_t recurrent_bytes{};
    std::uint64_t convolution_bytes{};

    [[nodiscard]] std::uint64_t total_bytes() const noexcept {
        return latent_bytes + recurrent_bytes + convolution_bytes;
    }
};

[[nodiscard]] KimiCacheFootprint kimi_cache_footprint(
    std::uint32_t context_tokens) noexcept;

class KimiStateCache {
public:
    KimiStateCache() = default;

    // Allocates for `context_tokens` of MLA latent and the fixed KDA state.
    // Exact mode has no hidden fallback, so a request beyond the contract's
    // maximum context is an error rather than a silent clamp.
    [[nodiscard]] ValidationResult reset(std::uint32_t context_tokens);
    void clear() noexcept;

    [[nodiscard]] std::uint32_t length() const noexcept { return length_; }
    [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] KimiCacheFootprint footprint() const noexcept;

    // --- Gated MLA ---------------------------------------------------------
    // Appends one token's compressed KV for one layer. `latent` is the
    // normalized `kv_lora_rank` block followed by the `qk_rope_head_dim` slot
    // shared across heads. `layer` is a backbone layer index; it must be a
    // full-attention layer.
    [[nodiscard]] ValidationResult append_latent(std::uint32_t layer,
                                                 std::uint32_t position,
                                                 std::span<const float> latent);
    // Rows [0, length) for one layer, contiguous and row-major.
    [[nodiscard]] std::span<const float> latent_rows(
        std::uint32_t layer) const noexcept;
    // Publishes `position + 1` tokens as visible. Called once per token after
    // every MLA layer has appended, so a partially written row is never read.
    [[nodiscard]] ValidationResult commit(std::uint32_t position);

    // --- KDA ---------------------------------------------------------------
    // The `[value_dim, key_dim]` state for one head of one KDA layer, mutable
    // so the recurrence updates it in place.
    [[nodiscard]] std::span<float> recurrent_state(std::uint32_t layer,
                                                   std::uint32_t head) noexcept;
    [[nodiscard]] std::span<const float> recurrent_state(
        std::uint32_t layer, std::uint32_t head) const noexcept;
    // The `kernel - 1` rows of convolution history for one layer's q, k, or v
    // projection, laid out per channel as `kimi_short_conv_step` expects.
    [[nodiscard]] std::span<float> convolution_history(
        std::uint32_t layer, std::uint32_t projection) noexcept;

    // Truncates to `tokens`, keeping the MLA rows below it. The KDA state is
    // recurrent and cannot be rewound, so this reports an error rather than
    // pretending a prefix can be restored: a shortened sequence must be
    // re-prefilled. Multi-turn continuation therefore holds for the MLA half
    // only when nothing before the reuse point changed.
    [[nodiscard]] ValidationResult truncate(std::uint32_t tokens);

private:
    [[nodiscard]] std::int32_t latent_slot(std::uint32_t layer) const noexcept;
    [[nodiscard]] std::int32_t recurrent_slot(std::uint32_t layer) const noexcept;

    std::uint32_t capacity_{};
    std::uint32_t length_{};
    // Latent rows per full-attention layer, indexed by that layer's ordinal
    // among the 24 rather than by backbone index.
    std::vector<std::vector<float>> latents_;
    std::vector<float> recurrent_;
    std::vector<float> convolution_;
};

}  // namespace strata
