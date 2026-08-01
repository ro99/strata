#include "test.hpp"

#include "strata/gemma4_ops.hpp"

#include <array>
#include <cmath>
#include <vector>

TEST_CASE("Gemma 4 RMSNorm and GeGLU preserve BF16 graph boundaries") {
    const std::array input{1.0F, -2.0F, 3.0F, -4.0F};
    const std::array weight{0.5F, 1.0F, 1.5F, 2.0F};
    std::array<float, 4> normalized{};
    REQUIRE(strata::gemma4_rms_norm_bf16(normalized, input, weight).ok());
    const std::array expected_norm{
        0.1826171875F, -0.73046875F, 1.640625F, -2.921875F};
    REQUIRE(normalized == expected_norm);

    const std::array gate{-2.0F, -0.5F, 0.5F, 2.0F};
    const std::array up{1.0F, 2.0F, 3.0F, 4.0F};
    std::array<float, 4> geglu{};
    REQUIRE(strata::gemma4_geglu_bf16(geglu, gate, up).ok());
    const std::array expected_geglu{
        -0.04541015625F, -0.30859375F, 1.0390625F, 7.8125F};
    REQUIRE(geglu == expected_geglu);
}

TEST_CASE("Gemma 4 local proportional and vision RoPE use the target layouts") {
    std::array<float, 8> local{1, 2, 3, 4, 5, 6, 7, 8};
    REQUIRE(strata::gemma4_rope_bf16(local, 1U, 10000.0F).ok());
    const std::array expected_local{
        -3.671875F, 1.390625F, 2.9375F, 3.984375F,
        3.546875F, 6.15625F, 7.03125F, 8.0F};
    REQUIRE(local == expected_local);

    std::array<float, 8> proportional{1, 2, 3, 4, 5, 6, 7, 8};
    REQUIRE(strata::gemma4_rope_bf16(
        proportional, 1U, 1000000.0F, 0.25F).ok());
    REQUIRE(proportional[2] == 3.0F);
    REQUIRE(proportional[3] == 4.0F);
    REQUIRE(proportional[6] == 7.0F);
    REQUIRE(proportional[7] == 8.0F);

    std::array<float, 8> vision{1, 2, 3, 4, 5, 6, 7, 8};
    REQUIRE(strata::gemma4_vision_rope_bf16(vision, 1, 2, 100.0F).ok());
    REQUIRE(vision[0] != 1.0F);
    REQUIRE(vision[4] != 5.0F);
}

TEST_CASE("Gemma 4 hybrid mask is causal local and bidirectional within images") {
    const std::array<std::int32_t, 6> groups{-1, 0, 0, 0, -1, -1};
    REQUIRE(strata::gemma4_text_attention_visible(1U, 3U, true, 2U, groups));
    REQUIRE(!strata::gemma4_text_attention_visible(4U, 1U, true, 2U, groups));
    REQUIRE(strata::gemma4_text_attention_visible(4U, 1U, false, 2U, groups));
    REQUIRE(!strata::gemma4_text_attention_visible(4U, 5U, false, 2U, groups));
}
