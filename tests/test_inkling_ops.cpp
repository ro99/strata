#include "test.hpp"

#include "strata/models/inkling/inkling_ops.hpp"
#include "strata/models/common/model.hpp"
#include "strata/models/common/model_adapter.hpp"
#include "strata/platform/numerics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

strata::RouterSpec inkling_router_spec() {
    return strata::inkling_small_nvfp4_spec().router;
}

// Independent restatement of the reference renormalization: softmax over the
// log-sigmoid of the raw logits of the active set.
std::vector<float> expected_sink_weights(const std::vector<float>& raw,
                                         float scale) {
    std::vector<float> log_probabilities(raw.size());
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0U; i < raw.size(); ++i) {
        log_probabilities[i] = std::log(1.0F / (1.0F + std::exp(-raw[i])));
        maximum = std::max(maximum, log_probabilities[i]);
    }
    float total = 0.0F;
    for (auto& value : log_probabilities) {
        value = std::exp(value - maximum);
        total += value;
    }
    for (auto& value : log_probabilities) value = value / total * scale;
    return log_probabilities;
}

}  // namespace

TEST_CASE("Inkling layer pattern matches the pinned local layer ids") {
    // The checkpoint lists 35 local layers; the complement is every sixth
    // layer counting from index 5.
    const std::vector<std::uint32_t> global{5U, 11U, 17U, 23U, 29U, 35U, 41U};
    std::uint32_t global_count = 0U;
    for (std::uint32_t layer = 0U; layer < 42U; ++layer) {
        const bool is_global = strata::inkling_global_attention_layer(layer);
        if (is_global) ++global_count;
        const bool expected =
            std::find(global.begin(), global.end(), layer) != global.end();
        REQUIRE(is_global == expected);
        // Local layers span exactly their window; global layers span 1024.
        REQUIRE(strata::inkling_relative_extent(layer) ==
                (expected ? 1024U : 512U));
    }
    REQUIRE(global_count == 7U);
    REQUIRE(!strata::inkling_sparse_layer(0U));
    REQUIRE(!strata::inkling_sparse_layer(1U));
    REQUIRE(strata::inkling_sparse_layer(2U));
    // Layer 2 is the one MoE layer whose experts ship unquantized.
    REQUIRE(!strata::inkling_quantized_expert_layer(2U));
    REQUIRE(strata::inkling_quantized_expert_layer(3U));
    REQUIRE(strata::inkling_quantized_expert_layer(41U));
}

TEST_CASE("Inkling MTP depths follow the checkpoint's local depth mask") {
    const std::vector<std::uint32_t> local{0U, 2U, 4U, 5U, 6U, 7U};
    for (std::uint32_t depth = 0U; depth < 8U; ++depth) {
        const bool expected_local =
            std::find(local.begin(), local.end(), depth) != local.end();
        REQUIRE(strata::inkling_mtp_global_attention_depth(depth) !=
                expected_local);
        REQUIRE(strata::inkling_mtp_relative_extent(depth) ==
                (expected_local ? 512U : 1024U));
    }
}

TEST_CASE("Inkling short convolution is causal with a residual") {
    // Two channels, kernel 4. Channel 0 selects the current token, channel 1
    // selects the token three positions back.
    const std::vector<float> weight{0.0F, 0.0F, 0.0F, 1.0F,
                                    1.0F, 0.0F, 0.0F, 0.0F};
    const std::vector<float> input{1.0F, 10.0F, 2.0F, 20.0F, 3.0F, 30.0F,
                                   4.0F, 40.0F, 5.0F, 50.0F};
    std::vector<float> output(input.size());
    auto status = strata::inkling_short_conv_f32(output, input, {}, weight, 5U,
                                                 2U, 4U);
    REQUIRE(status.ok());
    for (std::uint64_t token = 0U; token < 5U; ++token) {
        // Channel 0: x + x (tap on itself).
        REQUIRE_NEAR(output[token * 2U], input[token * 2U] * 2.0F, 1e-6F);
    }
    // Channel 1 reaches three back; the first three tokens have no such tap,
    // so they are the residual alone.
    REQUIRE_NEAR(output[1], 10.0F, 1e-6F);
    REQUIRE_NEAR(output[3], 20.0F, 1e-6F);
    REQUIRE_NEAR(output[5], 30.0F, 1e-6F);
    REQUIRE_NEAR(output[7], 40.0F + 10.0F, 1e-6F);
    REQUIRE_NEAR(output[9], 50.0F + 20.0F, 1e-6F);
}

TEST_CASE("Inkling short convolution continues across a history window") {
    const std::vector<float> weight{0.25F, 0.5F, 0.75F, 1.0F};
    const std::vector<float> history{2.0F, 4.0F, 8.0F};
    const std::vector<float> input{16.0F};
    std::vector<float> output(1U);
    auto status = strata::inkling_short_conv_f32(output, input, history, weight,
                                                 1U, 1U, 4U);
    REQUIRE(status.ok());
    // 16 residual + 0.25*2 + 0.5*4 + 0.75*8 + 1.0*16
    REQUIRE_NEAR(output[0], 16.0F + 0.5F + 2.0F + 6.0F + 16.0F, 1e-6F);

    // A decode step fed the same window must agree with the prefill result for
    // the same token, which is the property the streaming cache relies on.
    const std::vector<float> prefill{2.0F, 4.0F, 8.0F, 16.0F};
    std::vector<float> whole(4U);
    status = strata::inkling_short_conv_f32(whole, prefill, {}, weight, 4U, 1U,
                                            4U);
    REQUIRE(status.ok());
    REQUIRE_NEAR(whole[3], output[0], 1e-6F);
}

TEST_CASE("Inkling log scaling is inert below the position floor") {
    const auto& contract = strata::kInklingExecutionContract;
    REQUIRE_NEAR(strata::inkling_log_scaling_tau(0U), 1.0F, 1e-6F);
    REQUIRE_NEAR(strata::inkling_log_scaling_tau(1000U), 1.0F, 1e-6F);
    // Exactly at the floor the ratio is one, so tau is still one.
    REQUIRE_NEAR(
        strata::inkling_log_scaling_tau(contract.log_scaling_position_floor - 1U),
        1.0F, 1e-6F);
    const auto beyond = contract.log_scaling_position_floor * 4U;
    const float expected =
        1.0F + contract.log_scaling_alpha *
                   std::log(static_cast<float>(beyond + 1U) /
                            static_cast<float>(contract.log_scaling_position_floor));
    REQUIRE_NEAR(strata::inkling_log_scaling_tau(beyond), expected, 1e-5F);
    REQUIRE(strata::inkling_log_scaling_tau(beyond) > 1.0F);
}

TEST_CASE("Inkling relative logits project the per-head branch by distance") {
    // Two heads, width 2, extent 3.
    const std::vector<float> relative{1.0F, 2.0F, 3.0F, 4.0F};
    const std::vector<float> projection{1.0F, 0.0F, -1.0F, 0.0F, 1.0F, 2.0F};
    std::vector<float> output(6U);
    auto status = strata::inkling_relative_logits(output, relative, projection,
                                                  2U, 2U, 3U, 1.0F);
    REQUIRE(status.ok());
    // head 0: 1*[1,0,-1] + 2*[0,1,2] = [1,2,3]
    REQUIRE_NEAR(output[0], 1.0F, 1e-6F);
    REQUIRE_NEAR(output[1], 2.0F, 1e-6F);
    REQUIRE_NEAR(output[2], 3.0F, 1e-6F);
    // head 1: 3*[1,0,-1] + 4*[0,1,2] = [3,4,5]
    REQUIRE_NEAR(output[3], 3.0F, 1e-6F);
    REQUIRE_NEAR(output[4], 4.0F, 1e-6F);
    REQUIRE_NEAR(output[5], 5.0F, 1e-6F);

    // tau scales the whole bias, which is how long-context temperature reaches
    // the position signal.
    status = strata::inkling_relative_logits(output, relative, projection, 2U,
                                             2U, 3U, 2.0F);
    REQUIRE(status.ok());
    REQUIRE_NEAR(output[0], 2.0F, 1e-6F);
    REQUIRE_NEAR(output[5], 10.0F, 1e-6F);
}

TEST_CASE("Inkling attention visibility follows the layer pattern") {
    // Local layers see exactly the window, inclusive of the query.
    REQUIRE(strata::inkling_attention_visible(600U, 600U, true, 512U));
    REQUIRE(strata::inkling_attention_visible(600U, 89U, true, 512U));
    REQUIRE(!strata::inkling_attention_visible(600U, 88U, true, 512U));
    REQUIRE(!strata::inkling_attention_visible(600U, 601U, true, 512U));
    // Global layers see the whole causal prefix.
    REQUIRE(strata::inkling_attention_visible(600U, 0U, false, 512U));
    REQUIRE(!strata::inkling_attention_visible(600U, 601U, false, 512U));
}

TEST_CASE("Inkling router selects routed experts and always keeps both sinks") {
    const auto spec = inkling_router_spec();
    const auto routed = spec.routed_experts;
    const std::uint32_t shared = 2U;
    std::vector<float> logits(routed + shared, -8.0F);
    std::vector<float> bias(routed, 0.0F);
    // Six clear winners in a deliberately unsorted order.
    const std::vector<std::uint32_t> winners{7U, 3U, 200U, 41U, 128U, 99U};
    for (std::size_t i = 0U; i < winners.size(); ++i) {
        logits[winners[i]] = 4.0F - static_cast<float>(i) * 0.25F;
    }
    logits[routed] = 1.5F;
    logits[routed + 1U] = -0.5F;

    auto route = strata::inkling_route_sigmoid_sink(logits, bias, spec, shared,
                                                    1.0F);
    REQUIRE(route.ok());
    REQUIRE(route.value.experts.size() == spec.experts_per_token + shared);
    REQUIRE(route.value.weights.size() == route.value.experts.size());
    for (std::size_t i = 0U; i < winners.size(); ++i) {
        REQUIRE(route.value.experts[i] == winners[i]);
    }
    // Sinks are appended with ids continuing past the routed range.
    REQUIRE(route.value.experts[6] == routed);
    REQUIRE(route.value.experts[7] == routed + 1U);

    std::vector<float> raw;
    for (const auto expert : route.value.experts) raw.push_back(logits[expert]);
    const auto expected = expected_sink_weights(raw, spec.routed_scale);
    for (std::size_t i = 0U; i < expected.size(); ++i) {
        REQUIRE_NEAR(route.value.weights[i], expected[i], 1e-5F);
    }
    // Weights sum to routed_scale * global_scale, not to one.
    float total = 0.0F;
    for (const auto weight : route.value.weights) total += weight;
    REQUIRE_NEAR(total, spec.routed_scale, 1e-4F);
}

TEST_CASE("Inkling router applies the correction bias to selection only") {
    const auto spec = inkling_router_spec();
    const auto routed = spec.routed_experts;
    const std::uint32_t shared = 2U;
    std::vector<float> logits(routed + shared, -8.0F);
    std::vector<float> bias(routed, 0.0F);
    for (std::uint32_t expert = 0U; expert < 6U; ++expert) {
        logits[expert] = 1.0F;
    }
    // Expert 100 loses on raw score but wins once the bias is applied.
    logits[100U] = 0.0F;
    bias[100U] = 5.0F;

    auto route = strata::inkling_route_sigmoid_sink(logits, bias, spec, shared,
                                                    1.0F);
    REQUIRE(route.ok());
    REQUIRE(route.value.experts[0] == 100U);
    // The bias must not reach the weights: expert 100's weight is derived from
    // its raw logit of 0, so it is strictly below the experts whose logit is 1.
    REQUIRE(route.value.weights[0] < route.value.weights[1]);
    std::vector<float> raw;
    for (const auto expert : route.value.experts) raw.push_back(logits[expert]);
    const auto expected = expected_sink_weights(raw, spec.routed_scale);
    for (std::size_t i = 0U; i < expected.size(); ++i) {
        REQUIRE_NEAR(route.value.weights[i], expected[i], 1e-5F);
    }
}

TEST_CASE("Inkling router ties resolve to the lower expert index") {
    const auto spec = inkling_router_spec();
    const auto routed = spec.routed_experts;
    std::vector<float> logits(routed + 2U, -8.0F);
    const std::vector<float> bias(routed, 0.0F);
    for (std::uint32_t expert = 10U; expert < 20U; ++expert) {
        logits[expert] = 2.0F;
    }
    auto route = strata::inkling_route_sigmoid_sink(logits, bias, spec, 2U, 1.0F);
    REQUIRE(route.ok());
    for (std::uint32_t i = 0U; i < spec.experts_per_token; ++i) {
        REQUIRE(route.value.experts[i] == 10U + i);
    }
}

TEST_CASE("Inkling router global scale multiplies every weight") {
    const auto spec = inkling_router_spec();
    const auto routed = spec.routed_experts;
    std::vector<float> logits(routed + 2U, -1.0F);
    const std::vector<float> bias(routed, 0.0F);
    for (std::uint32_t expert = 0U; expert < 8U; ++expert) logits[expert] = 1.0F;

    auto unit = strata::inkling_route_sigmoid_sink(logits, bias, spec, 2U, 1.0F);
    auto scaled = strata::inkling_route_sigmoid_sink(logits, bias, spec, 2U, 0.5F);
    REQUIRE(unit.ok());
    REQUIRE(scaled.ok());
    for (std::size_t i = 0U; i < unit.value.weights.size(); ++i) {
        REQUIRE_NEAR(scaled.value.weights[i], unit.value.weights[i] * 0.5F,
                      1e-5F);
    }
}

TEST_CASE("Inkling router rejects contracts it does not implement") {
    auto spec = inkling_router_spec();
    const auto routed = spec.routed_experts;
    const std::vector<float> logits(routed + 2U, 0.0F);
    const std::vector<float> bias(routed, 0.0F);
    spec.scoring = strata::RouterScoreKind::Softmax;
    REQUIRE(!strata::inkling_route_sigmoid_sink(logits, bias, spec, 2U, 1.0F).ok());
    spec = inkling_router_spec();
    spec.selection_bias = false;
    REQUIRE(!strata::inkling_route_sigmoid_sink(logits, bias, spec, 2U, 1.0F).ok());
    spec = inkling_router_spec();
    // A logit vector without the sink rows is the wrong width.
    const std::vector<float> without_sinks(routed, 0.0F);
    REQUIRE(!strata::inkling_route_sigmoid_sink(without_sinks, bias, spec, 2U,
                                                1.0F).ok());
}

TEST_CASE("Inkling interleaved SwiGLU reads strided gate and up rows") {
    // [g0, u0, g1, u1]
    const std::vector<float> gate_up{1.0F, 3.0F, -2.0F, 5.0F};
    std::vector<float> output(2U);
    auto status = strata::inkling_interleaved_swiglu_f32(output, gate_up);
    REQUIRE(status.ok());
    REQUIRE_NEAR(output[0], strata::silu_f32(1.0F) * 3.0F, 1e-6F);
    REQUIRE_NEAR(output[1], strata::silu_f32(-2.0F) * 5.0F, 1e-6F);
}

TEST_CASE("Inkling NVFP4 dequantization multiplies by the per-expert scale") {
    // One row, sixteen columns, so a single group scale.
    strata::InklingNvfp4MatrixView matrix;
    std::vector<std::byte> packed(8U);
    std::vector<std::byte> scales(1U);
    // Every nibble is the E2M1 code for 1.0 (0x2), both halves of the byte.
    for (auto& byte : packed) byte = std::byte{0x22};
    // E4M3 code 0x38 is exponent 7, mantissa 0 => 1.0.
    scales[0] = std::byte{0x38};
    matrix.packed = packed;
    matrix.scales = scales;
    matrix.global_scale = 0.25F;
    matrix.rows = 1U;
    matrix.columns = 16U;
    matrix.packed_columns = 8U;
    matrix.scale_columns = 1U;
    matrix.group_size = 16U;

    const std::vector<float> input(16U, 1.0F);
    std::vector<float> output(1U);
    auto status = strata::inkling_nvfp4_matvec_reference(matrix, input, output);
    REQUIRE(status.ok());
    // 16 columns * 1.0 weight * 1.0 group scale * 0.25 global scale.
    REQUIRE_NEAR(output[0], 4.0F, 1e-6F);
    REQUIRE_NEAR(strata::fp4_e2m1_f32(0x2U), 1.0F, 1e-6F);
    REQUIRE_NEAR(strata::fp8_e4m3_f32(0x38U), 1.0F, 1e-6F);
}

TEST_CASE("Inkling NVFP4 matvec rejects malformed views") {
    strata::InklingNvfp4MatrixView matrix;
    const std::vector<std::byte> packed(8U);
    const std::vector<std::byte> scales(1U);
    matrix.packed = packed;
    matrix.scales = scales;
    matrix.rows = 1U;
    matrix.columns = 16U;
    matrix.packed_columns = 8U;
    matrix.scale_columns = 1U;
    matrix.group_size = 16U;
    matrix.global_scale = 0.0F;
    const std::vector<float> input(16U, 1.0F);
    std::vector<float> output(1U);
    // A non-positive global scale would silently zero the whole expert.
    REQUIRE(!strata::inkling_nvfp4_matvec_reference(matrix, input, output).ok());
    matrix.global_scale = 1.0F;
    matrix.scale_columns = 2U;
    REQUIRE(!strata::inkling_nvfp4_matvec_reference(matrix, input, output).ok());
}

TEST_CASE("Inkling MXFP4 decodes U32 bytes low nibble first with E8M0 groups") {
    strata::InklingMxfp4MatrixView matrix;
    // One row and one group. Alternating codes +1 (0x2) and -2 (0xc), with
    // the even column in the low nibble: every byte is 0xc2.
    std::vector<std::byte> packed(16U, std::byte{0xc2});
    std::vector<std::byte> scales(1U, std::byte{127U});  // 2^(127-127)
    matrix.packed = packed;
    matrix.scales = scales;
    matrix.rows = 1U;
    matrix.columns = 32U;
    matrix.packed_columns = 16U;
    matrix.scale_columns = 1U;
    matrix.group_size = 32U;
    std::vector<float> input(32U, 1.0F);
    std::vector<float> output(1U);
    REQUIRE(strata::inkling_mxfp4_matvec_reference(matrix, input, output).ok());
    // Sixteen pairs of (+1) + (-2).
    REQUIRE(output[0] == -16.0F);
    matrix.scale_columns = 2U;
    REQUIRE(!strata::inkling_mxfp4_matvec_reference(matrix, input, output).ok());
}

TEST_CASE("shared log-sigmoid matches the naive form where it is stable") {
    for (const float value : {-4.0F, -1.0F, -0.25F, 0.0F, 0.25F, 1.0F, 4.0F}) {
        const float naive = std::log(1.0F / (1.0F + std::exp(-value)));
        REQUIRE_NEAR(strata::log_sigmoid_f32(value), naive, 1e-6F);
    }
    // The naive form underflows here; the stable one stays finite and linear.
    REQUIRE(std::isfinite(strata::log_sigmoid_f32(-200.0F)));
    REQUIRE_NEAR(strata::log_sigmoid_f32(-200.0F), -200.0F, 1e-3F);
}
