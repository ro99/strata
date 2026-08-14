#include "test.hpp"

#include "strata/deepseek_ops.hpp"
#include "strata/numerics.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

std::uint16_t reference_bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7F80'0000U) != 0x7F80'0000U) {
        bits += 0x7FFFU + ((bits >> 16U) & 1U);
    }
    return static_cast<std::uint16_t>(bits >> 16U);
}

float bf16_value(float value) {
    return std::bit_cast<float>(
        static_cast<std::uint32_t>(reference_bf16(value)) << 16U);
}

}  // namespace

TEST_CASE("DeepSeek native FP4 fixture matches target checkpoint bytes") {
    // First 16 packed bytes and the first E8M0 scale from
    // layers.0.ffn.experts.0.w1 in revision 9e165c30... .
    constexpr std::array<std::uint8_t, 16> encoded{
        0x8cU, 0x54U, 0xa2U, 0x44U, 0xccU, 0x54U, 0x6cU, 0x55U,
        0x20U, 0x2cU, 0xe0U, 0xeaU, 0x2dU, 0xfdU, 0x05U, 0x22U};
    std::array<std::byte, 16> weights{};
    for (std::size_t index = 0U; index < encoded.size(); ++index) {
        weights[index] = static_cast<std::byte>(encoded[index]);
    }
    constexpr std::array<std::byte, 1> scales{std::byte{0x78U}};
    std::array<float, 32> input{};
    std::fill_n(input.begin(), 8U, 1.0F);
    std::array<float, 1> output{};
    const auto result = strata::dsv4_fp4_e2m1_matvec_f32(
        output, input, weights, scales, 1U, 32U);
    REQUIRE(result.ok());
    REQUIRE_NEAR(output[0], 0.0546875F, 1.0e-7F);
}

TEST_CASE("DeepSeek indexer Hadamard and FP4 activation path matches target semantics") {
    std::array<float, 128> rotated{};
    rotated[0] = 1.0F;
    const auto rotation = strata::dsv4_hadamard_rotate_f32(rotated);
    REQUIRE(rotation.ok());
    const float expected_rotation = bf16_value(1.0F / std::sqrt(128.0F));
    for (const float value : rotated) {
        REQUIRE_NEAR(value, expected_rotation, 0.0F);
    }

    std::array<float, 32> quantized{};
    quantized[0] = 0.6F;
    quantized[1] = -1.4F;
    quantized[2] = 2.1F;
    quantized[3] = 6.0F;
    quantized[4] = 0.75F;
    quantized[5] = 1.75F;
    quantized[6] = 3.5F;
    const auto simulation = strata::dsv4_fp4_e2m1_simulate_f32(quantized);
    REQUIRE(simulation.ok());
    REQUIRE_NEAR(quantized[0], 0.5F, 0.0F);
    REQUIRE_NEAR(quantized[1], -1.5F, 0.0F);
    REQUIRE_NEAR(quantized[2], 2.0F, 0.0F);
    REQUIRE_NEAR(quantized[3], 6.0F, 0.0F);
    REQUIRE_NEAR(quantized[4], 1.0F, 0.0F);
    REQUIRE_NEAR(quantized[5], 2.0F, 0.0F);
    REQUIRE_NEAR(quantized[6], 4.0F, 0.0F);

    REQUIRE(!strata::dsv4_hadamard_rotate_f32(
                 std::span<float>(rotated).first(127U)).ok());
    REQUIRE(!strata::dsv4_fp4_e2m1_simulate_f32(
                 std::span<float>(quantized).first(31U)).ok());
}

TEST_CASE("DeepSeek learned index selects the scalar-oracle top 512 at the boundary") {
    constexpr std::uint32_t heads = 3U;
    constexpr std::uint32_t head_dim = 4U;
    constexpr std::uint32_t positions = 513U;
    constexpr std::uint32_t top_k = 512U;
    const std::array<float, heads * head_dim> queries{
        1.0F, -0.5F, 0.25F, 2.0F,
        -1.0F, 0.75F, 0.5F, -0.25F,
        0.5F, 1.0F, -1.5F, 0.25F};
    const std::array<float, heads> weights{0.5F, -0.25F, 1.5F};
    std::vector<float> keys(static_cast<std::size_t>(positions) * head_dim);
    for (std::uint32_t row = 0U; row < positions; ++row) {
        for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension) {
            keys[static_cast<std::size_t>(row) * head_dim + dimension] =
                bf16_value(static_cast<float>(
                    static_cast<int>((row * 17U + dimension * 11U) % 31U) - 15) /
                    8.0F);
        }
    }
    std::vector<float> expected(positions);
    for (std::uint32_t row = 0U; row < positions; ++row) {
        float score = 0.0F;
        for (std::uint32_t head = 0U; head < heads; ++head) {
            float dot = 0.0F;
            for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension) {
                dot += queries[head * head_dim + dimension] *
                       keys[row * head_dim + dimension];
            }
            score += bf16_value(weights[head] *
                                std::max(0.0F, bf16_value(dot)));
        }
        expected[row] = bf16_value(score);
    }
    std::vector<float> actual(positions);
    REQUIRE(strata::dsv4_index_scores_f32(
                actual, queries, keys, weights, heads, head_dim).ok());
    REQUIRE(actual == expected);

    std::vector<std::uint32_t> reference(positions);
    for (std::uint32_t index = 0U; index < positions; ++index) {
        reference[index] = index;
    }
    std::partial_sort(reference.begin(), reference.begin() + top_k,
                      reference.end(), [&expected](auto first, auto second) {
                          if (expected[first] != expected[second]) {
                              return expected[first] > expected[second];
                          }
                          return first < second;
                      });
    reference.resize(top_k);
    const auto selected = strata::dsv4_index_topk_f32(actual, top_k);
    REQUIRE(selected.ok());
    REQUIRE(selected.positions == reference);
    REQUIRE(std::find(selected.positions.begin(), selected.positions.end(),
                      reference.back()) != selected.positions.end());
    REQUIRE(!strata::dsv4_index_topk_f32(actual, positions + 1U).ok());
}

TEST_CASE("DeepSeek native FP8 fixture matches target checkpoint bytes") {
    // First eight values and block scale from layers.0.attn.wq_a.
    constexpr std::array<std::uint8_t, 8> encoded{
        0xe0U, 0xf0U, 0x6cU, 0x6cU, 0x69U, 0x41U, 0x63U, 0xefU};
    std::array<std::byte, 128> weights{};
    for (std::size_t index = 0U; index < encoded.size(); ++index) {
        weights[index] = static_cast<std::byte>(encoded[index]);
    }
    constexpr std::array<std::byte, 1> scales{std::byte{0x73U}};
    std::array<float, 128> input{};
    std::fill_n(input.begin(), encoded.size(), 1.0F);
    std::array<float, 1> output{};
    const auto result = strata::dsv4_fp8_e4m3_matvec_f32(
        output, input, weights, scales, 1U, 128U);
    REQUIRE(result.ok());
    REQUIRE_NEAR(output[0], 0.00738525390625F, 1.0e-9F);
    REQUIRE_NEAR(strata::dsv4_fp8_e4m3_f32(0x7eU), 448.0F, 0.0F);
    REQUIRE(std::isnan(strata::dsv4_fp8_e4m3_f32(0x7fU)));
}

TEST_CASE("DeepSeek block FP8-to-BF16 conversion matches the scalar oracle") {
    constexpr std::uint64_t rows = 130U;
    constexpr std::uint64_t columns = 259U;
    std::vector<std::byte> weights(static_cast<std::size_t>(rows * columns));
    for (std::size_t index = 0U; index < weights.size(); ++index) {
        auto encoded = static_cast<std::uint8_t>((index * 37U + 13U) & 0xFFU);
        if ((encoded & 0x7FU) == 0x7FU) encoded = 0x7EU;
        weights[index] = static_cast<std::byte>(encoded);
    }
    constexpr std::array<std::uint8_t, 6> encoded_scales{
        0x73U, 0x78U, 0x7fU, 0x82U, 0x69U, 0x80U};
    std::array<std::byte, encoded_scales.size()> scales{};
    for (std::size_t index = 0U; index < scales.size(); ++index) {
        scales[index] = static_cast<std::byte>(encoded_scales[index]);
    }
    std::vector<std::uint16_t> converted(static_cast<std::size_t>(rows * columns));
    const auto result = strata::dsv4_fp8_e4m3_block128_to_bf16(
        converted, weights, scales, rows, columns);
    REQUIRE(result.ok());

    constexpr std::uint64_t scale_columns = (columns + 127U) / 128U;
    for (std::uint64_t row = 0U; row < rows; ++row) {
        for (std::uint64_t column = 0U; column < columns; ++column) {
            const auto index = static_cast<std::size_t>(row * columns + column);
            const auto encoded = std::to_integer<std::uint8_t>(weights[index]);
            const auto scale_encoded = std::to_integer<std::uint8_t>(
                scales[static_cast<std::size_t>(
                    (row / 128U) * scale_columns + column / 128U)]);
            const auto expected = reference_bf16(
                strata::dsv4_fp8_e4m3_f32(encoded) *
                strata::dsv4_fp8_e8m0_scale_f32(scale_encoded));
            REQUIRE(converted[index] == expected);
        }
    }
    converted.pop_back();
    REQUIRE(!strata::dsv4_fp8_e4m3_block128_to_bf16(
                 converted, weights, scales, rows, columns).ok());
}

TEST_CASE("DeepSeek sqrtsoftplus routing keeps selection bias out of weights") {
    auto spec = strata::deepseek_v4_flash_0731_spec().router;
    spec.routed_experts = 4U;
    spec.experts_per_token = 2U;
    const std::array<float, 4> logits{-2.0F, 0.0F, 1.0F, 2.0F};
    const std::array<float, 4> bias{10.0F, 0.0F, 0.0F, 0.0F};
    const auto route = strata::dsv4_route_sqrtsoftplus_f32(logits, bias, spec);
    REQUIRE(route.ok());
    std::array<float, 4> scratch{};
    std::array<std::uint32_t, 2> experts{};
    std::array<float, 2> coefficients{};
    REQUIRE(strata::dsv4_route_sqrtsoftplus_f32_into(
                logits, bias, spec, scratch, experts, coefficients).ok());
    REQUIRE(std::memcmp(experts.data(), route.value.experts.data(), sizeof(experts)) == 0);
    REQUIRE(std::memcmp(coefficients.data(), route.value.weights.data(), sizeof(coefficients)) == 0);
    const std::vector<std::uint32_t> expected{0U, 3U};
    REQUIRE(route.value.experts == expected);
    REQUIRE_NEAR(route.value.weights[0] + route.value.weights[1], 1.5F, 1.0e-6F);
    REQUIRE(route.value.weights[0] < route.value.weights[1]);
}

TEST_CASE("DeepSeek sqrtsoftplus into routing is exact and allocation-free") {
    auto spec = strata::deepseek_v4_flash_0731_spec().router;
    spec.routed_experts = 256U;
    spec.experts_per_token = 6U;
    std::array<float, 256> logits{}, bias{};
    for (std::size_t index = 0U; index < logits.size(); ++index) {
        logits[index] = static_cast<float>((index * 37U) % 101U) / 17.0F - 3.0F;
        bias[index] = static_cast<float>(static_cast<int>((index * 13U) % 17U) - 8) / 100.0F;
    }
    const auto expected = strata::dsv4_route_sqrtsoftplus_f32(logits, bias, spec);
    REQUIRE(expected.ok());
    std::vector<float> scratch(spec.routed_experts), coefficients(spec.experts_per_token);
    std::vector<std::uint32_t> experts(spec.experts_per_token);
    const auto* scratch_data = scratch.data();
    const auto* experts_data = experts.data();
    const auto* coefficients_data = coefficients.data();
    const auto scratch_capacity = scratch.capacity();
    const auto experts_capacity = experts.capacity();
    const auto coefficients_capacity = coefficients.capacity();
    for (unsigned repeat = 0U; repeat < 100U; ++repeat) {
        REQUIRE(strata::dsv4_route_sqrtsoftplus_f32_into(
                    logits, bias, spec, scratch, experts, coefficients).ok());
        REQUIRE(std::memcmp(experts.data(), expected.value.experts.data(), experts.size() * sizeof(experts[0])) == 0);
        REQUIRE(std::memcmp(coefficients.data(), expected.value.weights.data(), coefficients.size() * sizeof(coefficients[0])) == 0);
        REQUIRE(scratch.data() == scratch_data);
        REQUIRE(experts.data() == experts_data);
        REQUIRE(coefficients.data() == coefficients_data);
        REQUIRE(scratch.capacity() == scratch_capacity && experts.capacity() == experts_capacity && coefficients.capacity() == coefficients_capacity);
        REQUIRE(scratch.size() == spec.routed_experts && experts.size() == spec.experts_per_token && coefficients.size() == spec.experts_per_token);
    }
    auto tie_spec = spec;
    tie_spec.routed_experts = 8U;
    tie_spec.experts_per_token = 3U;
    const std::array<float, 8> ties{};
    std::array<float, 8> tie_scratch{};
    std::array<std::uint32_t, 3> tie_experts{};
    std::array<float, 3> tie_coefficients{};
    REQUIRE(strata::dsv4_route_sqrtsoftplus_f32_into(
                ties, ties, tie_spec, tie_scratch, tie_experts, tie_coefficients).ok());
    const std::array<std::uint32_t, 3> expected_ties{0U, 1U, 2U};
    REQUIRE(tie_experts == expected_ties);
    std::fill(scratch.begin(), scratch.end(), 7.0F);
    std::fill(experts.begin(), experts.end(), 99U);
    std::fill(coefficients.begin(), coefficients.end(), 7.0F);
    REQUIRE(!strata::dsv4_route_sqrtsoftplus_f32_into(
                 logits, bias, spec, std::span<float>(scratch).first(255U),
                 experts, coefficients).ok());
    REQUIRE(std::all_of(scratch.begin(), scratch.begin() + 255U, [](float value) { return value == 0.0F; }));
    REQUIRE(std::all_of(experts.begin(), experts.end(), [](auto value) { return value == 0U; }));
    REQUIRE(std::all_of(coefficients.begin(), coefficients.end(), [](float value) { return value == 0.0F; }));
    std::fill(scratch.begin(), scratch.end(), 7.0F);
    std::fill(experts.begin(), experts.end(), 99U);
    std::fill(coefficients.begin(), coefficients.end(), 7.0F);
    REQUIRE(!strata::dsv4_route_sqrtsoftplus_f32_into(
                 logits, bias, spec, scratch,
                 std::span<std::uint32_t>(experts).first(5U), coefficients).ok());
    REQUIRE(std::all_of(experts.begin(), experts.begin() + 5U, [](auto value) { return value == 0U; }));
    auto invalid_spec = spec;
    invalid_spec.experts_per_token = invalid_spec.routed_experts + 1U;
    REQUIRE(!strata::dsv4_route_sqrtsoftplus_f32_into(
                 logits, bias, invalid_spec, scratch, experts, coefficients).ok());
    auto underflow_logits = logits;
    underflow_logits.fill(-std::numeric_limits<float>::max());
    std::fill(scratch.begin(), scratch.end(), 7.0F);
    REQUIRE(!strata::dsv4_route_sqrtsoftplus_f32_into(
                 underflow_logits, bias, spec, scratch, experts, coefficients).ok());
    auto invalid_logits = logits;
    invalid_logits[3] = std::numeric_limits<float>::quiet_NaN();
    std::fill(scratch.begin(), scratch.end(), 7.0F);
    std::fill(experts.begin(), experts.end(), 99U);
    std::fill(coefficients.begin(), coefficients.end(), 7.0F);
    REQUIRE(!strata::dsv4_route_sqrtsoftplus_f32_into(
                 invalid_logits, bias, spec, scratch, experts, coefficients).ok());
    REQUIRE(std::all_of(scratch.begin(), scratch.end(), [](float value) { return value == 0.0F; }));
    REQUIRE(std::all_of(experts.begin(), experts.end(), [](auto value) { return value == 0U; }));
    REQUIRE(std::all_of(coefficients.begin(), coefficients.end(), [](float value) { return value == 0.0F; }));
}

TEST_CASE("DeepSeek hash routing has an exact allocation-free form") {
    auto spec = strata::deepseek_v4_flash_0731_spec().router;
    spec.routed_experts = 8U;
    spec.experts_per_token = 3U;
    const std::array<float, 8> logits{
        -2.0F, -1.0F, 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    const std::array<std::uint32_t, 3> selected{6U, 1U, 4U};
    const auto expected = strata::dsv4_route_hash_sqrtsoftplus_f32(
        logits, selected, spec);
    REQUIRE(expected.ok());
    std::array<std::uint32_t, 3> experts{};
    std::array<float, 3> coefficients{};
    REQUIRE(strata::dsv4_route_hash_sqrtsoftplus_f32_into(
                logits, selected, spec, experts, coefficients).ok());
    REQUIRE(std::memcmp(experts.data(), expected.value.experts.data(),
                        sizeof(experts)) == 0);
    REQUIRE(std::memcmp(coefficients.data(), expected.value.weights.data(),
                        sizeof(coefficients)) == 0);

    auto invalid = selected;
    invalid[1] = 8U;
    experts.fill(99U);
    coefficients.fill(7.0F);
    REQUIRE(!strata::dsv4_route_hash_sqrtsoftplus_f32_into(
                 logits, invalid, spec, experts, coefficients).ok());
    REQUIRE(std::all_of(experts.begin(), experts.end(),
                        [](auto value) { return value == 0U; }));
    REQUIRE(std::all_of(coefficients.begin(), coefficients.end(),
                        [](float value) { return value == 0.0F; }));
    auto invalid_logits = logits;
    invalid_logits[0] = std::numeric_limits<float>::quiet_NaN();
    experts.fill(99U);
    coefficients.fill(7.0F);
    REQUIRE(!strata::dsv4_route_hash_sqrtsoftplus_f32_into(
                 invalid_logits, selected, spec, experts, coefficients).ok());
    REQUIRE(std::all_of(experts.begin(), experts.end(),
                        [](auto value) { return value == 0U; }));
    REQUIRE(std::all_of(coefficients.begin(), coefficients.end(),
                        [](float value) { return value == 0.0F; }));
}

TEST_CASE("DeepSeek mHC Sinkhorn preserves four residual lanes") {
    constexpr std::uint32_t multiplier = 4U;
    std::array<float, 24> mixes{};
    std::array<float, 24> base{};
    constexpr std::array<float, 3> scale{1.0F, 1.0F, 1.0F};
    const auto split = strata::dsv4_mhc_split_sinkhorn_f32(
        mixes, scale, base, multiplier, 20U, 1.0e-6F);
    REQUIRE(split.ok());
    for (const float value : split.value.pre) {
        REQUIRE_NEAR(value, 0.500001F, 1.0e-7F);
    }
    for (const float value : split.value.post) REQUIRE_NEAR(value, 1.0F, 1.0e-7F);
    for (std::uint32_t row = 0U; row < multiplier; ++row) {
        float sum = 0.0F;
        for (std::uint32_t column = 0U; column < multiplier; ++column) {
            sum += split.value.combination[row * multiplier + column];
        }
        REQUIRE_NEAR(sum, 1.0F, 1.0e-5F);
    }
}

TEST_CASE("DeepSeek prepacked mHC preserves exact projection order") {
    if (!strata::dsv4_mhc_prepacked_supported()) return;
    constexpr std::size_t columns = 16U;
    constexpr std::size_t rows = 24U;
    std::array<float, columns> hidden{};
    std::array<float, rows * columns> projection{};
    for (std::size_t index = 0U; index < hidden.size(); ++index) {
        hidden[index] = static_cast<float>(static_cast<int>(index) - 8) / 16.0F;
    }
    for (std::size_t index = 0U; index < projection.size(); ++index) {
        projection[index] =
            static_cast<float>(static_cast<int>(index % 13U) - 6) / 128.0F;
    }
    std::array<float, rows * columns> packed{};
    REQUIRE(strata::dsv4_pack_mhc_projection_f32(
                packed, projection, rows, columns).ok());
    constexpr std::array<float, 3> scale{0.75F, 0.5F, 0.25F};
    std::array<float, rows> base{};
    std::array<float, columns / 4U> oracle{};
    std::array<float, columns / 4U> candidate{};
    strata::Dsv4MhcMix oracle_mix;
    strata::Dsv4MhcMix candidate_mix;
    REQUIRE(strata::dsv4_mhc_pre_f32(
                oracle, oracle_mix, hidden, projection, scale, base).ok());
    REQUIRE(strata::dsv4_mhc_prepacked_f32(
                candidate, candidate_mix, hidden, packed, scale, base).ok());
    REQUIRE(std::memcmp(oracle.data(), candidate.data(), sizeof(oracle)) == 0);
    REQUIRE(oracle_mix.pre == candidate_mix.pre);
    REQUIRE(oracle_mix.post == candidate_mix.post);
    REQUIRE(oracle_mix.combination == candidate_mix.combination);
    REQUIRE(!strata::dsv4_mhc_prepacked_f32(
                 candidate, candidate_mix, hidden,
                 std::span<const float>(packed).first(packed.size() - 1U),
                 scale, base).ok());
}

TEST_CASE("DeepSeek mHC post mixing uses source rows and destination columns") {
    constexpr std::uint32_t multiplier = 2U;
    constexpr std::array<float, 1> branch{3.0F};
    constexpr std::array<float, 2> residual{10.0F, 20.0F};
    const strata::Dsv4MhcMix mix{
        {},
        {1.0F, 1.0F},
        {0.1F, 0.9F,
         0.8F, 0.2F},
    };
    std::array<float, 2> output{};

    const auto result = strata::dsv4_mhc_post_f32(
        output, branch, residual, mix, multiplier);

    REQUIRE(result.ok());
    REQUIRE_NEAR(output[0], 20.0F, 1.0e-6F);
    REQUIRE_NEAR(output[1], 16.0F, 1.0e-6F);
}

TEST_CASE("DeepSeek SwiGLU applies asymmetric target clamping") {
    constexpr std::array<float, 3> gate{-20.0F, 20.0F, 1.0F};
    constexpr std::array<float, 3> up{-20.0F, 20.0F, 2.0F};
    std::array<float, 3> output{};
    const auto result = strata::dsv4_swiglu_f32(output, gate, up, 10.0F);
    REQUIRE(result.ok());
    REQUIRE_NEAR(output[0], strata::silu_f32(-20.0F) * -10.0F, 1.0e-7F);
    REQUIRE_NEAR(output[1], strata::silu_f32(10.0F) * 10.0F, 1.0e-5F);
}
