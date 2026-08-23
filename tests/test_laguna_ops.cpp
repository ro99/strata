#include "test.hpp"

#include "strata/laguna_ops.hpp"
#include "strata/model.hpp"
#include "strata/model_adapter.hpp"
#include "strata/tokenizer.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

bool close(float actual, float expected, float tolerance = 1.0e-5F) {
    return std::fabs(actual - expected) <= tolerance;
}

}  // namespace

TEST_CASE("Laguna S 2.1 pinned spec validates") {
    const auto spec = strata::laguna_s21_nvfp4_spec();
    REQUIRE(strata::validate_laguna_s21_nvfp4(spec).ok());
    REQUIRE(spec.quant_bits == 4U);
    REQUIRE(spec.router.experts_per_token == 10U);
    REQUIRE(spec.router.routed_experts == 256U);
    REQUIRE(spec.laguna.sliding_window == 512U);

    // A checkpoint that silently dropped below four bits, changed top-k, or
    // moved off sigmoid routing is a different model and must be rejected.
    auto sub_nibble = spec;
    sub_nibble.quant_bits = 2U;
    sub_nibble.mixed_quantization.routed_experts.bits = 2U;
    REQUIRE(!strata::validate_laguna_s21_nvfp4(sub_nibble).ok());
    auto retopped = spec;
    retopped.router.experts_per_token = 8U;
    REQUIRE(!strata::validate_laguna_s21_nvfp4(retopped).ok());
    auto softmax = spec;
    softmax.router.scoring = strata::RouterScoreKind::Softmax;
    REQUIRE(!strata::validate_laguna_s21_nvfp4(softmax).ok());
}

TEST_CASE("Laguna S 2.1 MXFP4 spec is distinct and keeps the model contract") {
    const auto nvfp4 = strata::laguna_s21_nvfp4_spec();
    const auto mxfp4 = strata::laguna_s21_mxfp4_spec();
    REQUIRE(strata::validate_laguna_s21_nvfp4(nvfp4).ok());
    REQUIRE(strata::validate_laguna_s21_mxfp4(mxfp4).ok());
    REQUIRE(mxfp4.source.repository == "olka-fi/Laguna-S-2.1-MXFP4");
    REQUIRE(mxfp4.source.repository != nvfp4.source.repository);
    REQUIRE(mxfp4.mixed_quantization.kind ==
            strata::QuantizationKind::CompressedTensorsW4A16);
    REQUIRE(mxfp4.mixed_quantization.routed_experts.group_size == 32U);
    REQUIRE(mxfp4.laguna.quantized_expert_layers == mxfp4.layer_count);
    REQUIRE(mxfp4.router.experts_per_token == nvfp4.router.experts_per_token);
    REQUIRE(mxfp4.hidden_size == nvfp4.hidden_size);

    auto wrong_group = mxfp4;
    wrong_group.mixed_quantization.routed_experts.group_size = 16U;
    REQUIRE(!strata::validate_laguna_s21_mxfp4(wrong_group).ok());
    auto wrong_source = mxfp4;
    wrong_source.source.repository = nvfp4.source.repository;
    REQUIRE(!strata::validate_laguna_s21_mxfp4(wrong_source).ok());
}

TEST_CASE("Laguna attention layout alternates global and sliding layers") {
    REQUIRE(strata::laguna_global_attention_layer(0U));
    REQUIRE(!strata::laguna_global_attention_layer(1U));
    REQUIRE(strata::laguna_global_attention_layer(44U));
    REQUIRE(strata::laguna_attention_heads(0U) == 48U);
    REQUIRE(strata::laguna_attention_heads(1U) == 72U);
    REQUIRE(!strata::laguna_sparse_layer(0U));
    REQUIRE(strata::laguna_sparse_layer(1U));
    // Routed experts are NVFP4 up to layer 39 and plain BF16 from 40 on.
    REQUIRE(strata::laguna_quantized_expert_layer(39U));
    REQUIRE(!strata::laguna_quantized_expert_layer(40U));
}

TEST_CASE("NVFP4 decodes E2M1 nibbles against E4M3 group scales") {
    REQUIRE(strata::laguna_fp4_e2m1_f32(0x0U) == 0.0F);
    REQUIRE(strata::laguna_fp4_e2m1_f32(0x7U) == 6.0F);
    REQUIRE(strata::laguna_fp4_e2m1_f32(0xFU) == -6.0F);
    REQUIRE(strata::laguna_fp4_e2m1_f32(0x9U) == -0.5F);
    // 0x40 is E4M3 exponent 8, mantissa 0, which is exactly 2.0.
    REQUIRE(strata::laguna_fp8_e4m3_f32(0x40U) == 2.0F);

    constexpr std::uint64_t columns = 16U;
    // Row 0 is every nibble 0x2 (1.0); row 1 alternates 0x1 (0.5) in the even
    // column and 0x9 (-0.5) in the odd one, which pins the packing order.
    std::vector<std::byte> packed(2U * columns / 2U);
    for (std::size_t index = 0U; index < columns / 2U; ++index) {
        packed[index] = std::byte{0x22U};
        packed[columns / 2U + index] = std::byte{0x91U};
    }
    const std::vector<std::byte> scales{std::byte{0x40U}, std::byte{0x40U}};

    strata::LagunaNvfp4MatrixView matrix;
    matrix.packed = packed;
    matrix.scales = scales;
    matrix.global_scale = 4.0F;
    matrix.rows = 2U;
    matrix.columns = columns;
    matrix.packed_columns = columns / 2U;
    matrix.scale_columns = 1U;
    matrix.group_size = 16U;

    std::vector<float> ones(columns, 1.0F);
    std::array<float, 2> output{};
    REQUIRE(strata::laguna_nvfp4_matvec_reference(matrix, ones, output).ok());
    // Group scale 2.0 over global scale 4.0 is 0.5 per element.
    REQUIRE(output[0] == 16.0F * 0.5F);
    REQUIRE(output[1] == 0.0F);

    std::vector<float> ramp(columns);
    for (std::uint64_t column = 0U; column < columns; ++column) {
        ramp[column] = static_cast<float>(column);
    }
    REQUIRE(strata::laguna_nvfp4_matvec_reference(matrix, ramp, output).ok());
    REQUIRE(output[0] == 60.0F);
    REQUIRE(output[1] == -2.0F);

    // A non-positive global scale would silently rescale every weight.
    auto broken = matrix;
    broken.global_scale = 0.0F;
    REQUIRE(!strata::laguna_nvfp4_matvec_reference(broken, ones, output).ok());
}

TEST_CASE("Laguna router selects on biased scores and returns unbiased weights") {
    strata::RouterSpec spec;
    spec.selection = strata::RouterSelectionKind::TopK;
    spec.scoring = strata::RouterScoreKind::Sigmoid;
    spec.routed_experts = 4U;
    spec.experts_per_token = 2U;
    spec.normalize_topk = true;
    spec.selection_bias = true;

    const std::array logits{0.0F, 0.0F, 1.0F, -1.0F};
    const std::array bias{1.0F, 0.0F, 0.0F, 0.0F};
    auto routed = strata::laguna_route_sigmoid_topk(logits, bias, spec, 0.0F);
    REQUIRE(routed.ok());
    // Expert 0 wins only because of the bias; expert 2 wins on its own score.
    REQUIRE(routed.value.experts == std::vector<std::uint32_t>({0U, 2U}));
    const float low = strata::sigmoid_f32(0.0F);
    const float high = strata::sigmoid_f32(1.0F);
    REQUIRE(close(routed.value.weights[0], low / (low + high)));
    REQUIRE(close(routed.value.weights[1], high / (low + high)));
    REQUIRE(close(routed.value.weights[0] + routed.value.weights[1], 1.0F));

    // Equal selection scores keep the lower expert index.
    const std::array flat{0.0F, 0.0F, 0.0F, 0.0F};
    const std::array no_bias{0.0F, 0.0F, 0.0F, 0.0F};
    routed = strata::laguna_route_sigmoid_topk(flat, no_bias, spec, 0.0F);
    REQUIRE(routed.ok());
    REQUIRE(routed.value.experts == std::vector<std::uint32_t>({0U, 1U}));
    REQUIRE(close(routed.value.weights[0], 0.5F));

    // Softmax scoring is a different router and must not be accepted.
    auto rescored = spec;
    rescored.scoring = strata::RouterScoreKind::Softmax;
    REQUIRE(!strata::laguna_route_sigmoid_topk(logits, bias, rescored, 0.0F).ok());
}

TEST_CASE("Laguna softplus gate matches the reference threshold form") {
    REQUIRE(close(strata::laguna_softplus_f32(0.0F), std::log(2.0F)));
    REQUIRE(close(strata::laguna_softplus_f32(-2.0F),
                  std::log1p(std::exp(-2.0F))));
    // Above the threshold softplus is the identity; log1p(exp(x)) would overflow.
    REQUIRE(strata::laguna_softplus_f32(90.0F) == 90.0F);
    REQUIRE(std::isfinite(strata::laguna_softplus_f32(1000.0F)));
}

TEST_CASE("Laguna rotary schedules follow the declared YaRN and default rope") {
    const auto& contract = strata::kLagunaExecutionContract;
    const auto global = strata::laguna_rope_schedule(true);
    REQUIRE(global.rotary_dimensions == 64U);
    REQUIRE(global.inverse_frequencies.size() == 32U);
    REQUIRE(global.attention_scaling == contract.global_rope_attention_factor);
    // The correction range for beta_fast 32 / beta_slow 1 at theta 500000 over
    // a 64-wide rotary is [9, 18]: below it the schedule extrapolates exactly,
    // above it it interpolates by the full factor.
    REQUIRE(close(global.inverse_frequencies[0], 1.0F));
    const auto interpolated = [&](std::uint32_t index) {
        return static_cast<float>(
            1.0 / (contract.global_rope_factor *
                   std::pow(static_cast<double>(contract.global_rope_theta),
                            static_cast<double>(2U * index) / 64.0)));
    };
    const auto extrapolated = [&](std::uint32_t index) {
        return static_cast<float>(
            1.0 / std::pow(static_cast<double>(contract.global_rope_theta),
                           static_cast<double>(2U * index) / 64.0));
    };
    REQUIRE(close(global.inverse_frequencies[9], extrapolated(9U), 1.0e-9F));
    REQUIRE(close(global.inverse_frequencies[18], interpolated(18U), 1.0e-9F));
    REQUIRE(close(global.inverse_frequencies[31], interpolated(31U), 1.0e-12F));
    // Inside the ramp the schedule is strictly between the two.
    REQUIRE(global.inverse_frequencies[13] > interpolated(13U));
    REQUIRE(global.inverse_frequencies[13] < extrapolated(13U));

    const auto sliding = strata::laguna_rope_schedule(false);
    REQUIRE(sliding.rotary_dimensions == 128U);
    REQUIRE(sliding.inverse_frequencies.size() == 64U);
    REQUIRE(sliding.attention_scaling == 1.0F);
    REQUIRE(close(sliding.inverse_frequencies[0], 1.0F));
    REQUIRE(close(sliding.inverse_frequencies[1],
                  static_cast<float>(1.0 / std::pow(10000.0, 2.0 / 128.0))));
}

TEST_CASE("Laguna rotary uses the rotate_half convention over a partial head") {
    strata::LagunaRopeSchedule schedule;
    schedule.inverse_frequencies = {1.0F};
    schedule.attention_scaling = 1.0F;
    schedule.rotary_dimensions = 2U;
    std::array<float, 4> head{1.0F, 2.0F, 3.0F, 4.0F};
    REQUIRE(strata::laguna_rope_half_f32(head, 1U, schedule).ok());
    const auto cosine = std::cos(1.0F);
    const auto sine = std::sin(1.0F);
    // Value 0 pairs with value 1 (rotary_dimensions / 2 apart), not with its
    // neighbour: this is rotate_half, not the interleaved GLM form.
    REQUIRE(close(head[0], 1.0F * cosine - 2.0F * sine));
    REQUIRE(close(head[1], 2.0F * cosine + 1.0F * sine));
    // The tail beyond the rotary prefix is passed through untouched.
    REQUIRE(head[2] == 3.0F);
    REQUIRE(head[3] == 4.0F);
}

TEST_CASE("Laguna sliding attention admits exactly one window") {
    constexpr std::uint32_t window = strata::kLagunaExecutionContract.sliding_window;
    REQUIRE(strata::laguna_attention_visible(600U, 600U, true, window));
    REQUIRE(strata::laguna_attention_visible(600U, 89U, true, window));
    REQUIRE(!strata::laguna_attention_visible(600U, 88U, true, window));
    REQUIRE(!strata::laguna_attention_visible(600U, 601U, true, window));
    // Global layers see the whole causal prefix.
    REQUIRE(strata::laguna_attention_visible(600U, 0U, false, window));
    REQUIRE(!strata::laguna_attention_visible(600U, 601U, false, window));
}

TEST_CASE("Laguna pre-tokenizer isolates digits and splits newline runs") {
    const auto digits = strata::pretokenize(
        strata::TokenizerContract::Laguna, "abc123");
    REQUIRE(digits.ok());
    REQUIRE(digits.value ==
            std::vector<std::string>({"abc", "1", "2", "3"}));

    // The first Split stage detaches a newline run from the whitespace before
    // it; without it " \n x" would pre-tokenize as {" \n", " x"}.
    const auto newlines = strata::pretokenize(
        strata::TokenizerContract::Laguna, "abc \n def");
    REQUIRE(newlines.ok());
    REQUIRE(newlines.value ==
            std::vector<std::string>({"abc", " ", "\n", " def"}));

    const auto contraction = strata::pretokenize(
        strata::TokenizerContract::Laguna, "it's");
    REQUIRE(contraction.ok());
    REQUIRE(contraction.value == std::vector<std::string>({"it", "'s"}));
}

TEST_CASE("Laguna pre-tokenizer classifies Unicode categories exactly") {
    const auto pretokens = [](std::string_view text) {
        const auto result = strata::pretokenize(
            strata::TokenizerContract::Laguna, text);
        REQUIRE(result.ok());
        return result.value;
    };
    // U+00BB is punctuation, so it joins the following '/' in one symbol run.
    // The heuristic classifier shared with GLM calls it a letter and would
    // produce {"»", "/c"} instead.
    REQUIRE(pretokens("»/c") == std::vector<std::string>({"»/", "c"}));
    // U+3000 is White_Space, so it can only be the optional single-character
    // prefix of a letter run, never part of a symbol run.
    REQUIRE(pretokens("　c") == std::vector<std::string>({"　c"}));
    REQUIRE(pretokens("　-Y") == std::vector<std::string>({"　", "-Y"}));
    REQUIRE(pretokens("　　$c") ==
            std::vector<std::string>({"　", "　", "$c"}));
    // U+00AA and U+00B5 are letters despite sitting among Latin-1 symbols.
    REQUIRE(pretokens("ªµ") == std::vector<std::string>({"ªµ"}));
    REQUIRE(pretokens("×÷") == std::vector<std::string>({"×÷"}));
}

TEST_CASE("Laguna chat template reproduces the checkpoint prompt shape") {
    const std::array messages{
        strata::ChatMessage{strata::ChatRole::User, "hi"}};
    const auto thinking = strata::render_laguna_chat_prompt(messages, true);
    REQUIRE(thinking.starts_with("〈|EOS|〉<system>You are a helpful"));
    REQUIRE(thinking.find("<user>hi</user>\n") != std::string::npos);
    REQUIRE(thinking.ends_with("<assistant><think>"));

    const auto direct = strata::render_laguna_chat_prompt(messages, false);
    REQUIRE(direct.ends_with("<assistant></think>"));

    // An explicit empty system message opts out of the <system> block.
    const std::array suppressed{
        strata::ChatMessage{strata::ChatRole::System, ""},
        strata::ChatMessage{strata::ChatRole::User, "hi"}};
    const auto no_system = strata::render_laguna_chat_prompt(suppressed, true);
    REQUIRE(no_system.find("<system>") == std::string::npos);
    REQUIRE(no_system.starts_with("〈|EOS|〉<user>hi</user>\n"));
}
