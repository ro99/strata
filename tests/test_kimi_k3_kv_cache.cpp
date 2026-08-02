#include "test.hpp"

#include "strata/kimi_k3_kv_cache.hpp"
#include "strata/model_adapter.hpp"

#include <vector>

TEST_CASE("Kimi-K3 recurrent state does not grow with context") {
    // The asymmetry that decides the cache design: the MLA half is linear in
    // context and the KDA half is a constant, so a 1M-token context costs what
    // its latents cost and nothing more.
    const auto small = strata::kimi_cache_footprint(2048U);
    const auto large = strata::kimi_cache_footprint(1'048'576U);
    REQUIRE(small.recurrent_bytes == large.recurrent_bytes);
    REQUIRE(small.convolution_bytes == large.convolution_bytes);
    REQUIRE(large.latent_bytes == small.latent_bytes * 512U);

    const auto& c = strata::kKimiK3ExecutionContract;
    // 24 gated MLA layers hold 512 + 64 F32 values per token.
    REQUIRE(small.latent_bytes ==
            2048ULL * (c.kv_lora_rank + c.rope_head_dim) * sizeof(float) * 24ULL);
    // 69 KDA layers hold a fixed [96, 128, 128] state each.
    REQUIRE(small.recurrent_bytes == 69ULL * c.linear_attention_heads *
                                         c.linear_head_dim * c.value_head_dim *
                                         sizeof(float));
    // At 2048 tokens the fixed half dominates; at 1M the latents do.
    REQUIRE(small.recurrent_bytes > small.latent_bytes);
    REQUIRE(large.latent_bytes > large.recurrent_bytes);
}

TEST_CASE("Kimi-K3 cache routes layers to the half that owns them") {
    strata::KimiStateCache cache;
    REQUIRE(cache.reset(64U).ok());
    REQUIRE(cache.capacity() == 64U);
    REQUIRE(cache.length() == 0U);

    const auto& c = strata::kKimiK3ExecutionContract;
    const auto width = c.kv_lora_rank + c.rope_head_dim;
    std::vector<float> latent(width, 1.5F);

    // Layer 3 is the first gated MLA layer; layer 0 is KDA.
    REQUIRE(cache.append_latent(3U, 0U, latent).ok());
    REQUIRE(!cache.append_latent(0U, 0U, latent).ok());
    REQUIRE(cache.recurrent_state(0U, 0U).size() ==
            static_cast<std::size_t>(c.linear_head_dim) * c.value_head_dim);
    REQUIRE(cache.recurrent_state(3U, 0U).empty());
    // Layers 91 and 92 are adjacent gated MLA layers, the paper's extra one at
    // the end of the backbone.
    REQUIRE(cache.append_latent(91U, 0U, latent).ok());
    REQUIRE(cache.append_latent(92U, 0U, latent).ok());
    REQUIRE(cache.recurrent_state(90U, 95U).size() ==
            static_cast<std::size_t>(c.linear_head_dim) * c.value_head_dim);
    REQUIRE(cache.recurrent_state(0U, c.linear_attention_heads).empty());

    // Rows only become visible on commit, so a partially written token is
    // never read by attention.
    REQUIRE(cache.latent_rows(3U).empty());
    REQUIRE(cache.commit(0U).ok());
    REQUIRE(cache.latent_rows(3U).size() == width);
    REQUIRE(cache.latent_rows(3U)[0] == 1.5F);
    REQUIRE(cache.length() == 1U);
}

TEST_CASE("Kimi-K3 recurrent slices are disjoint and independently mutable") {
    strata::KimiStateCache cache;
    REQUIRE(cache.reset(8U).ok());
    auto first = cache.recurrent_state(0U, 0U);
    auto second = cache.recurrent_state(0U, 1U);
    auto other_layer = cache.recurrent_state(1U, 0U);
    REQUIRE(!first.empty() && !second.empty() && !other_layer.empty());
    std::fill(first.begin(), first.end(), 3.0F);
    REQUIRE(second[0] == 0.0F);
    REQUIRE(other_layer[0] == 0.0F);
    REQUIRE(cache.recurrent_state(0U, 0U)[0] == 3.0F);

    // The three short convolutions of a layer have separate histories.
    auto q_history = cache.convolution_history(0U, 0U);
    auto v_history = cache.convolution_history(0U, 2U);
    const auto& c = strata::kKimiK3ExecutionContract;
    REQUIRE(q_history.size() ==
            static_cast<std::size_t>(c.linear_attention_heads) *
                c.linear_head_dim * (c.short_conv_kernel - 1U));
    std::fill(q_history.begin(), q_history.end(), 1.0F);
    REQUIRE(v_history[0] == 0.0F);
    REQUIRE(cache.convolution_history(0U, 3U).empty());
}

TEST_CASE("Kimi-K3 cache refuses what it cannot honestly do") {
    strata::KimiStateCache cache;
    REQUIRE(!cache.reset(0U).ok());
    REQUIRE(!cache.reset(strata::kKimiK3ExecutionContract.maximum_context_tokens +
                         1U).ok());
    REQUIRE(cache.reset(16U).ok());

    std::vector<float> latent(
        strata::kKimiK3ExecutionContract.kv_lora_rank +
            strata::kKimiK3ExecutionContract.rope_head_dim, 0.0F);
    REQUIRE(!cache.append_latent(3U, 16U, latent).ok());
    std::vector<float> narrow(latent.size() - 1U, 0.0F);
    REQUIRE(!cache.append_latent(3U, 0U, narrow).ok());

    REQUIRE(cache.commit(3U).ok());
    REQUIRE(cache.length() == 4U);
    REQUIRE(!cache.commit(1U).ok());

    // Truncation to the current length is a no-op and succeeds; a real
    // truncation is refused, because 69 recurrent layers have no recoverable
    // prefix and pretending otherwise would continue from a future the
    // shortened sequence no longer has.
    REQUIRE(cache.truncate(4U).ok());
    const auto shortened = cache.truncate(2U);
    REQUIRE(!shortened.ok());
    REQUIRE(shortened.errors.front().find("re-prefilled") != std::string::npos);
    REQUIRE(!cache.truncate(9U).ok());
}
