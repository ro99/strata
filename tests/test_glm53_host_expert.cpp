#include "test.hpp"

#include "strata/models/glm53/glm53_runtime.hpp"
#include "strata/platform/numerics.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr std::size_t kColumns = 4096U;  // one GLM-5.3 expert gate/up row

// The reference dequantization, written the way the format is defined rather
// than the way either decoder is implemented: value = e2m1(nibble) * 2^(e-127),
// column 2b in the low nibble of byte b, one exponent per 32 columns.
[[nodiscard]] float reference_fp4_dot(
    const std::vector<std::byte>& packed, const std::vector<std::byte>& scales,
    const std::vector<float>& input) {
    double sum = 0.0;
    for (std::size_t column = 0U; column < input.size(); ++column) {
        const auto byte = std::to_integer<std::uint8_t>(packed[column / 2U]);
        const auto nibble = static_cast<std::uint8_t>(
            column % 2U == 0U ? (byte & 0x0FU) : (byte >> 4U));
        const auto code = std::to_integer<std::uint8_t>(scales[column / 32U]);
        sum += static_cast<double>(input[column]) *
               static_cast<double>(strata::fp4_e2m1_f32(nibble)) *
               std::ldexp(1.0, static_cast<int>(code) - 127);
    }
    return static_cast<float>(sum);
}

// The NVFP4 reference, written the same way: value = e2m1(nibble) *
// (e4m3(code) / global_scale), one E4M3 code per 16 columns. The division is
// applied to the group scale and not to the sum, which is the order the
// exporter, the device kernel and the other NVFP4 host paths all use.
[[nodiscard]] float reference_nvfp4_dot(
    const std::vector<std::byte>& packed, const std::vector<std::byte>& scales,
    float global_scale, const std::vector<float>& input) {
    double sum = 0.0;
    for (std::size_t column = 0U; column < input.size(); ++column) {
        const auto byte = std::to_integer<std::uint8_t>(packed[column / 2U]);
        const auto nibble = static_cast<std::uint8_t>(
            column % 2U == 0U ? (byte & 0x0FU) : (byte >> 4U));
        const auto code = std::to_integer<std::uint8_t>(scales[column / 16U]);
        sum += static_cast<double>(input[column]) *
               static_cast<double>(strata::fp4_e2m1_f32(nibble)) *
               static_cast<double>(strata::fp8_e4m3_f32(code) / global_scale);
    }
    return static_cast<float>(sum);
}

}  // namespace

TEST_CASE("GLM-5.3 NVFP4 host dot agrees with a reference dequantization") {
    std::mt19937 engine(20260903U);
    std::uniform_int_distribution<int> nibbles(0, 255);
    // E4M3 codes with a positive sign and a moderate exponent: the format's
    // NaN code is 0x7F, and the checkpoint's own divisor brings the products
    // back to unit range, so this keeps both decoders and the double reference
    // comparable on precision rather than on magnitude.
    std::uniform_int_distribution<int> codes(0x30, 0x50);
    std::uniform_real_distribution<float> activations(-2.0F, 2.0F);
    // 21,504 is (448 * 6) / 0.125, the divisor this checkpoint's exporter fits
    // to a routed-expert projection; it is deliberately not a power of two, so
    // a decoder that multiplied by a reciprocal instead of dividing would not
    // reproduce the reference bit for bit.
    constexpr float kGlobalScale = 21504.0F;

    std::vector<std::byte> packed(kColumns / 2U);
    std::vector<std::byte> scales(kColumns / 16U);
    std::vector<float> input(kColumns);
    for (auto& byte : packed) byte = static_cast<std::byte>(nibbles(engine));
    for (auto& byte : scales) byte = static_cast<std::byte>(codes(engine));
    for (auto& value : input) value = activations(engine);

    const auto expected =
        reference_nvfp4_dot(packed, scales, kGlobalScale, input);
    const auto scalar = strata::glm53_host_nvfp4_group16_row_dot(
        packed, scales, kGlobalScale, input, false);
    const auto vectorized = strata::glm53_host_nvfp4_group16_row_dot(
        packed, scales, kGlobalScale, input, true);
    const auto tolerance = 1.0e-3F * std::max(1.0F, std::abs(expected));
    REQUIRE(std::abs(scalar - expected) <= tolerance);
    REQUIRE(std::abs(vectorized - expected) <= tolerance);
}

TEST_CASE("GLM-5.3 NVFP4 host dot decodes every E2M1 code in column order") {
    // The same nibble-order gate MXFP4 gets, over one group of 16 rather than
    // 32. A scale of 1.0 and a divisor of 1.0 leave the decoded weight alone,
    // so any difference here is the nibble mapping and nothing else.
    std::vector<std::byte> packed(8U);
    std::vector<std::byte> scales(1U, static_cast<std::byte>(0x38U));  // 1.0
    std::vector<float> input(16U, 0.0F);
    for (std::size_t byte = 0U; byte < packed.size(); ++byte) {
        packed[byte] = static_cast<std::byte>(
            (byte & 0x0FU) | ((15U - byte) << 4U));
    }
    REQUIRE(strata::fp8_e4m3_f32(0x38U) == 1.0F);
    for (std::size_t column = 0U; column < input.size(); ++column) {
        std::fill(input.begin(), input.end(), 0.0F);
        input[column] = 1.0F;
        const auto expected = strata::fp4_e2m1_f32(static_cast<std::uint8_t>(
            column % 2U == 0U ? column / 2U : 15U - column / 2U));
        REQUIRE(strata::glm53_host_nvfp4_group16_row_dot(
                    packed, scales, 1.0F, input, false) == expected);
        REQUIRE(strata::glm53_host_nvfp4_group16_row_dot(
                    packed, scales, 1.0F, input, true) == expected);
    }
}

TEST_CASE("GLM-5.3 NVFP4 host dot applies one E4M3 scale per 16 columns") {
    // Group 16, not MXFP4's 32: a decoder that kept the wider group would read
    // the first scale for the first 32 columns and land on the wrong total.
    std::vector<std::byte> packed(kColumns / 2U,
                                  static_cast<std::byte>(0x22U));  // 1.0, 1.0
    std::vector<std::byte> scales(kColumns / 16U);
    std::vector<float> input(kColumns, 1.0F);
    for (std::size_t group = 0U; group < scales.size(); ++group) {
        // 1.0 and 2.0 in E4M3, alternating per group.
        scales[group] = static_cast<std::byte>(group % 2U == 0U ? 0x38U : 0x40U);
    }
    REQUIRE(strata::fp8_e4m3_f32(0x40U) == 2.0F);
    // Halving the divisor doubles every weight, so the expected total states
    // the divide direction rather than assuming it.
    constexpr float kGlobalScale = 0.5F;
    const auto expected = static_cast<float>(kColumns / 2U) * 2.0F +
                          static_cast<float>(kColumns / 2U) * 4.0F;
    REQUIRE(strata::glm53_host_nvfp4_group16_row_dot(
                packed, scales, kGlobalScale, input, false) == expected);
    REQUIRE(strata::glm53_host_nvfp4_group16_row_dot(
                packed, scales, kGlobalScale, input, true) == expected);
}

TEST_CASE("GLM-5.3 NVFP4 host dot decodes a row the vector step cannot cover") {
    // 208 columns is three 64-column vector steps and a 16-column remainder,
    // so the AVX2 path's scalar tail runs. The tail indexes the decoded group
    // scales the body built rather than re-deriving them, and reading the wrong
    // one there is the mistake this catches. The two paths associate their sums
    // differently by design, so each is checked against the reference and not
    // against the other.
    std::mt19937 engine(20260904U);
    std::uniform_int_distribution<int> nibbles(0, 255);
    std::uniform_int_distribution<int> codes(0x30, 0x50);
    std::uniform_real_distribution<float> activations(-2.0F, 2.0F);
    constexpr std::size_t kPartial = 208U;
    constexpr float kGlobalScale = 21504.0F;

    std::vector<std::byte> packed(kPartial / 2U);
    std::vector<std::byte> scales(kPartial / 16U);
    std::vector<float> input(kPartial);
    for (auto& byte : packed) byte = static_cast<std::byte>(nibbles(engine));
    for (auto& byte : scales) byte = static_cast<std::byte>(codes(engine));
    for (auto& value : input) value = activations(engine);

    const auto expected =
        reference_nvfp4_dot(packed, scales, kGlobalScale, input);
    const auto tolerance = 1.0e-3F * std::max(1.0F, std::abs(expected));
    REQUIRE(std::abs(strata::glm53_host_nvfp4_group16_row_dot(
                packed, scales, kGlobalScale, input, false) - expected) <=
            tolerance);
    REQUIRE(std::abs(strata::glm53_host_nvfp4_group16_row_dot(
                packed, scales, kGlobalScale, input, true) - expected) <=
            tolerance);
}

TEST_CASE("GLM-5.3 MXFP4 host dot agrees with a reference dequantization") {
    std::mt19937 engine(20260830U);
    std::uniform_int_distribution<int> nibbles(0, 255);
    // E8M0 codes near the bias keep the products in a range where an f32
    // accumulation and the double reference cannot drift on magnitude alone.
    std::uniform_int_distribution<int> exponents(120, 131);
    std::uniform_real_distribution<float> activations(-2.0F, 2.0F);

    std::vector<std::byte> packed(kColumns / 2U);
    std::vector<std::byte> scales(kColumns / 32U);
    std::vector<float> input(kColumns);
    for (auto& byte : packed) byte = static_cast<std::byte>(nibbles(engine));
    for (auto& byte : scales) byte = static_cast<std::byte>(exponents(engine));
    for (auto& value : input) value = activations(engine);

    const auto expected = reference_fp4_dot(packed, scales, input);
    const auto scalar = strata::glm53_host_fp4_group32_row_dot(
        packed, scales, input, false);
    const auto vectorized = strata::glm53_host_fp4_group32_row_dot(
        packed, scales, input, true);
    const auto tolerance = 1.0e-3F * std::max(1.0F, std::abs(expected));
    REQUIRE(std::abs(scalar - expected) <= tolerance);
    REQUIRE(std::abs(vectorized - expected) <= tolerance);
}

TEST_CASE("GLM-5.3 MXFP4 host dot decodes every E2M1 code in column order") {
    // One group of 32: the low nibble of byte b must be column 2b. A decoder
    // that swapped the nibbles would pass a random-input test on magnitude and
    // fail this one, which is the failure the checkpoint audit ruled out.
    std::vector<std::byte> packed(16U);
    std::vector<std::byte> scales(1U, static_cast<std::byte>(127U));
    std::vector<float> input(32U, 0.0F);
    for (std::size_t byte = 0U; byte < packed.size(); ++byte) {
        // Column 2b gets code b, column 2b+1 gets code 15-b.
        packed[byte] = static_cast<std::byte>(
            (byte & 0x0FU) | ((15U - byte) << 4U));
    }
    for (std::size_t column = 0U; column < input.size(); ++column) {
        input[column] = 0.0F;
        std::fill(input.begin(), input.end(), 0.0F);
        input[column] = 1.0F;
        const auto expected = strata::fp4_e2m1_f32(static_cast<std::uint8_t>(
            column % 2U == 0U ? column / 2U : 15U - column / 2U));
        REQUIRE(strata::glm53_host_fp4_group32_row_dot(
                    packed, scales, input, false) == expected);
        REQUIRE(strata::glm53_host_fp4_group32_row_dot(
                    packed, scales, input, true) == expected);
    }
}

TEST_CASE("GLM-5.3 MXFP4 host dot applies one E8M0 exponent per 32 columns") {
    std::vector<std::byte> packed(kColumns / 2U,
                                  static_cast<std::byte>(0x22U));  // 1.0, 1.0
    std::vector<std::byte> scales(kColumns / 32U);
    std::vector<float> input(kColumns, 1.0F);
    for (std::size_t group = 0U; group < scales.size(); ++group) {
        // Alternate 2^0 and 2^1 so a decoder reading the wrong group is off by
        // a factor the sum cannot hide.
        scales[group] = static_cast<std::byte>(group % 2U == 0U ? 127U : 128U);
    }
    const auto expected = static_cast<float>(kColumns / 2U) * 1.0F +
                          static_cast<float>(kColumns / 2U) * 2.0F;
    REQUIRE(strata::glm53_host_fp4_group32_row_dot(
                packed, scales, input, false) == expected);
    REQUIRE(strata::glm53_host_fp4_group32_row_dot(
                packed, scales, input, true) == expected);
}

TEST_CASE("GLM-5.3 BF16 host dot matches an exact reference") {
    std::mt19937 engine(20260831U);
    std::uniform_real_distribution<float> weights(-1.0F, 1.0F);
    std::uniform_real_distribution<float> activations(-2.0F, 2.0F);
    std::vector<std::byte> encoded(kColumns * 2U);
    std::vector<float> input(kColumns);
    double expected = 0.0;
    for (std::size_t column = 0U; column < kColumns; ++column) {
        const auto weight = strata::bf16_round_f32(weights(engine));
        const auto bits = static_cast<std::uint16_t>(
            std::bit_cast<std::uint32_t>(weight) >> 16U);
        std::memcpy(encoded.data() + column * 2U, &bits, sizeof(bits));
        input[column] = activations(engine);
        expected += static_cast<double>(weight) *
                    static_cast<double>(input[column]);
    }
    const auto reference = static_cast<float>(expected);
    const auto tolerance = 1.0e-3F * std::max(1.0F, std::abs(reference));
    REQUIRE(std::abs(strata::glm53_host_bf16_row_dot(encoded, input, false) -
                     reference) <= tolerance);
    REQUIRE(std::abs(strata::glm53_host_bf16_row_dot(encoded, input, true) -
                     reference) <= tolerance);
}

namespace {

// One decode query's indexer inputs, with every pool's key set to a distinct
// constant so the ranking is controllable from the test.
struct SparseIndexFixture {
    std::vector<float> query, keys, gate, ape, weights;
    explicit SparseIndexFixture(std::uint32_t history) {
        constexpr auto D = strata::Glm53SparseIndexParameters::head_dim;
        constexpr auto H = strata::Glm53SparseIndexParameters::heads;
        query.assign(static_cast<std::size_t>(H) * D, 0.0F);
        keys.assign(static_cast<std::size_t>(history) * D, 0.0F);
        gate.assign(static_cast<std::size_t>(history) * D, 0.0F);
        ape.assign(static_cast<std::size_t>(
                       strata::Glm53SparseIndexParameters::pool) * D, 0.0F);
        weights.assign(H, 1.0F);
        // A query that reads only channel 0.
        for (std::uint32_t head = 0U; head < H; ++head) query[head * D] = 1.0F;
    }
};

}  // namespace

TEST_CASE("GLM-5.3 sparse index selection is the identity below index_topk") {
    constexpr std::uint32_t history = 1500U;
    SparseIndexFixture f(history);
    std::vector<std::uint32_t> selected(
        strata::Glm53SparseIndexParameters::selection_width);
    const auto count = strata::glm53_sparse_index_select_for_test(
        selected, f.query, f.keys, f.gate, f.ape, f.weights, history);
    REQUIRE(count == history);
    for (std::uint32_t token = 0U; token < history; ++token) {
        REQUIRE(selected[token] == token);
    }
}

TEST_CASE("GLM-5.3 sparse index selection bounds attention above index_topk") {
    constexpr std::uint32_t history = 32768U;
    constexpr auto D = strata::Glm53SparseIndexParameters::head_dim;
    SparseIndexFixture f(history);
    // Make later pools score higher, so the ranking is unambiguous.
    for (std::uint32_t token = 0U; token < history; ++token) {
        f.keys[static_cast<std::size_t>(token) * D] =
            static_cast<float>(token) / static_cast<float>(history);
    }
    std::vector<std::uint32_t> selected(
        strata::Glm53SparseIndexParameters::selection_width);
    const auto count = strata::glm53_sparse_index_select_for_test(
        selected, f.query, f.keys, f.gate, f.ape, f.weights, history);

    // 512 pools of 4, and history is a multiple of 4 so there is no tail.
    REQUIRE(count == strata::Glm53SparseIndexParameters::top_k);
    REQUIRE(count <= strata::Glm53SparseIndexParameters::selection_width);
    // Ascending, unique, in range.
    for (std::size_t index = 0U; index < count; ++index) {
        REQUIRE(selected[index] < history);
        if (index != 0U) REQUIRE(selected[index] > selected[index - 1U]);
    }
    // Highest-scoring pools are the latest ones, so the selection must end at
    // the final complete pool.
    REQUIRE(selected[count - 1U] == history - 1U);
}

TEST_CASE("GLM-5.3 sparse index selection always appends the visible tail") {
    // history % 4 == 3, so three raw tail positions follow the pooled ones.
    constexpr std::uint32_t history = 32771U;
    SparseIndexFixture f(history);
    std::vector<std::uint32_t> selected(
        strata::Glm53SparseIndexParameters::selection_width);
    const auto count = strata::glm53_sparse_index_select_for_test(
        selected, f.query, f.keys, f.gate, f.ape, f.weights, history);
    REQUIRE(count == strata::Glm53SparseIndexParameters::top_k + 3U);
    // The most recent tokens are never dropped, whatever the pool scores say.
    REQUIRE(selected[count - 1U] == history - 1U);
    REQUIRE(selected[count - 2U] == history - 2U);
    REQUIRE(selected[count - 3U] == history - 3U);
}

TEST_CASE("GLM-5.3 MLA workspace stops growing with the admitted context") {
    constexpr std::uint32_t page_tokens = 64U;
    const auto dense = strata::glm53_mla_workspace_bytes(2048U, page_tokens);
    // Below the threshold a call expands the whole visible history, so the
    // reservation is the dense extent: history x 64 heads x 2 x 256 x 4 B.
    REQUIRE(dense.output == 2048ULL * 64ULL * 2ULL * 256ULL * sizeof(float));

    // Above it the indexer expands a bounded selection instead. This is the
    // regression the runbook's own context table describes: the workspace must
    // be flat in context, not 4 GiB at 32,768 and 32 GiB at the ceiling.
    const auto sparse = strata::glm53_mla_workspace_bytes(32768U, page_tokens);
    const auto ceiling = strata::glm53_mla_workspace_bytes(262144U, page_tokens);
    REQUIRE(sparse.input == ceiling.input);
    REQUIRE(sparse.output == ceiling.output);
    REQUIRE(ceiling.total() < (1ULL << 30U));

    // Every admitted context, in both regimes, stays inside the per-device
    // workspace reserve the weight-arena budget is planned against.
    constexpr std::uint64_t workspace_reserve = 2ULL << 30U;
    for (const std::uint32_t context :
         {1U, 2048U, 2049U, 4096U, 32000U, 32768U, 65536U, 131072U, 262144U}) {
        const auto bytes = strata::glm53_mla_workspace_bytes(
            context, page_tokens);
        REQUIRE(bytes.total() < workspace_reserve);
    }
}
