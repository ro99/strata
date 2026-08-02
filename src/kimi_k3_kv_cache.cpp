#include "strata/kimi_k3_kv_cache.hpp"

#include <algorithm>
#include <string>

namespace strata {
namespace {

constexpr std::uint32_t kProjections = 3U;  // q, k, v short convolutions.

[[nodiscard]] std::uint32_t latent_width() noexcept {
    const auto& c = kKimiK3ExecutionContract;
    return c.kv_lora_rank + c.rope_head_dim;
}

[[nodiscard]] std::uint32_t full_attention_layers() noexcept {
    const auto& c = kKimiK3ExecutionContract;
    std::uint32_t count = 0U;
    for (std::uint32_t layer = 0U; layer < c.layer_count; ++layer) {
        if (kimi_k3_full_attention_layer(layer)) ++count;
    }
    return count;
}

[[nodiscard]] std::uint32_t kda_layers() noexcept {
    return kKimiK3ExecutionContract.layer_count - full_attention_layers();
}

[[nodiscard]] std::uint64_t recurrent_state_values() noexcept {
    const auto& c = kKimiK3ExecutionContract;
    return static_cast<std::uint64_t>(c.linear_attention_heads) *
           c.linear_head_dim * c.value_head_dim;
}

[[nodiscard]] std::uint64_t convolution_values() noexcept {
    const auto& c = kKimiK3ExecutionContract;
    return static_cast<std::uint64_t>(kProjections) * c.linear_attention_heads *
           c.linear_head_dim * (c.short_conv_kernel - 1U);
}

}  // namespace

KimiCacheFootprint kimi_cache_footprint(std::uint32_t context_tokens) noexcept {
    KimiCacheFootprint footprint;
    footprint.latent_bytes = static_cast<std::uint64_t>(context_tokens) *
                             latent_width() * sizeof(float) *
                             full_attention_layers();
    // Neither of the two below depends on `context_tokens`.
    footprint.recurrent_bytes =
        recurrent_state_values() * sizeof(float) * kda_layers();
    footprint.convolution_bytes =
        convolution_values() * sizeof(float) * kda_layers();
    return footprint;
}

ValidationResult KimiStateCache::reset(std::uint32_t context_tokens) {
    ValidationResult result;
    const auto& c = kKimiK3ExecutionContract;
    if (context_tokens == 0U || context_tokens > c.maximum_context_tokens) {
        result.errors.push_back(
            "Kimi-K3 context must be within [1, " +
            std::to_string(c.maximum_context_tokens) + "] tokens, asked for " +
            std::to_string(context_tokens));
        return result;
    }
    capacity_ = context_tokens;
    length_ = 0U;
    latents_.assign(full_attention_layers(), {});
    for (auto& rows : latents_) {
        rows.assign(static_cast<std::size_t>(context_tokens) * latent_width(),
                    0.0F);
    }
    recurrent_.assign(
        static_cast<std::size_t>(recurrent_state_values()) * kda_layers(), 0.0F);
    convolution_.assign(
        static_cast<std::size_t>(convolution_values()) * kda_layers(), 0.0F);
    return result;
}

void KimiStateCache::clear() noexcept {
    capacity_ = 0U;
    length_ = 0U;
    latents_.clear();
    recurrent_.clear();
    convolution_.clear();
}

KimiCacheFootprint KimiStateCache::footprint() const noexcept {
    return kimi_cache_footprint(capacity_);
}

std::int32_t KimiStateCache::latent_slot(std::uint32_t layer) const noexcept {
    const auto& c = kKimiK3ExecutionContract;
    if (layer >= c.layer_count || !kimi_k3_full_attention_layer(layer)) return -1;
    std::int32_t ordinal = 0;
    for (std::uint32_t index = 0U; index < layer; ++index) {
        if (kimi_k3_full_attention_layer(index)) ++ordinal;
    }
    return ordinal;
}

std::int32_t KimiStateCache::recurrent_slot(std::uint32_t layer) const noexcept {
    const auto& c = kKimiK3ExecutionContract;
    if (layer >= c.layer_count || !kimi_k3_kda_layer(layer)) return -1;
    std::int32_t ordinal = 0;
    for (std::uint32_t index = 0U; index < layer; ++index) {
        if (kimi_k3_kda_layer(index)) ++ordinal;
    }
    return ordinal;
}

ValidationResult KimiStateCache::append_latent(std::uint32_t layer,
                                               std::uint32_t position,
                                               std::span<const float> latent) {
    ValidationResult result;
    const auto slot = latent_slot(layer);
    if (slot < 0) {
        result.errors.push_back("Kimi-K3 layer " + std::to_string(layer) +
                                " is not a gated MLA layer and holds no latent");
        return result;
    }
    if (latent.size() != latent_width()) {
        result.errors.emplace_back(
            "Kimi-K3 latent row must be kv_lora_rank plus the shared rope slot");
        return result;
    }
    if (position >= capacity_) {
        result.errors.push_back(
            "Kimi-K3 latent position " + std::to_string(position) +
            " is beyond the admitted context of " + std::to_string(capacity_));
        return result;
    }
    auto& rows = latents_[static_cast<std::size_t>(slot)];
    std::copy(latent.begin(), latent.end(),
              rows.begin() + static_cast<std::ptrdiff_t>(
                                 static_cast<std::size_t>(position) *
                                 latent_width()));
    return result;
}

std::span<const float> KimiStateCache::latent_rows(
    std::uint32_t layer) const noexcept {
    const auto slot = latent_slot(layer);
    if (slot < 0) return {};
    return std::span<const float>(latents_[static_cast<std::size_t>(slot)])
        .subspan(0U, static_cast<std::size_t>(length_) * latent_width());
}

ValidationResult KimiStateCache::commit(std::uint32_t position) {
    ValidationResult result;
    if (position >= capacity_) {
        result.errors.push_back(
            "Kimi-K3 commit position " + std::to_string(position) +
            " is beyond the admitted context of " + std::to_string(capacity_));
        return result;
    }
    if (position + 1U < length_) {
        result.errors.emplace_back(
            "Kimi-K3 cache length cannot move backwards; use truncate");
        return result;
    }
    length_ = position + 1U;
    return result;
}

std::span<float> KimiStateCache::recurrent_state(std::uint32_t layer,
                                                 std::uint32_t head) noexcept {
    const auto& c = kKimiK3ExecutionContract;
    const auto slot = recurrent_slot(layer);
    if (slot < 0 || head >= c.linear_attention_heads) return {};
    const auto head_values =
        static_cast<std::size_t>(c.linear_head_dim) * c.value_head_dim;
    const auto base = static_cast<std::size_t>(slot) *
                          static_cast<std::size_t>(recurrent_state_values()) +
                      head * head_values;
    return std::span<float>(recurrent_).subspan(base, head_values);
}

std::span<const float> KimiStateCache::recurrent_state(
    std::uint32_t layer, std::uint32_t head) const noexcept {
    const auto& c = kKimiK3ExecutionContract;
    const auto slot = recurrent_slot(layer);
    if (slot < 0 || head >= c.linear_attention_heads) return {};
    const auto head_values =
        static_cast<std::size_t>(c.linear_head_dim) * c.value_head_dim;
    const auto base = static_cast<std::size_t>(slot) *
                          static_cast<std::size_t>(recurrent_state_values()) +
                      head * head_values;
    return std::span<const float>(recurrent_).subspan(base, head_values);
}

std::span<float> KimiStateCache::convolution_history(
    std::uint32_t layer, std::uint32_t projection) noexcept {
    const auto& c = kKimiK3ExecutionContract;
    const auto slot = recurrent_slot(layer);
    if (slot < 0 || projection >= kProjections) return {};
    const auto channels =
        static_cast<std::size_t>(c.linear_attention_heads) * c.linear_head_dim;
    const auto width = channels * (c.short_conv_kernel - 1U);
    const auto base = static_cast<std::size_t>(slot) *
                          static_cast<std::size_t>(convolution_values()) +
                      projection * width;
    return std::span<float>(convolution_).subspan(base, width);
}

ValidationResult KimiStateCache::truncate(std::uint32_t tokens) {
    ValidationResult result;
    if (tokens > length_) {
        result.errors.emplace_back(
            "Kimi-K3 cache cannot be truncated to more tokens than it holds");
        return result;
    }
    if (tokens == length_) return result;
    // The KDA half is a recurrence, not a log: S_t depends on every prior
    // token and no prefix of it can be recovered from the final state. Saying
    // so is the honest answer; silently keeping a stale state would make the
    // shortened sequence continue from a future it no longer has.
    result.errors.emplace_back(
        "Kimi-K3 has 69 recurrent KDA layers whose state cannot be rewound, so "
        "a shortened sequence must be re-prefilled rather than truncated");
    return result;
}

}  // namespace strata
