#include "test.hpp"

#include "strata/attention.hpp"

#include <array>
#include <cmath>
#include <vector>

TEST_CASE("generic online FlashAttention combines segments sinks and indexed rows") {
    constexpr std::uint32_t heads = 2U;
    constexpr std::uint32_t dimension = 4U;
    const std::array<float, heads * dimension> queries{
        1.0F, 0.5F, -0.5F, 0.25F,
        -0.25F, 0.75F, 0.5F, -1.0F};
    const std::array<float, 3U * dimension> first{
        1.0F, 0.0F, 0.5F, -0.5F,
        0.0F, 1.0F, -0.25F, 0.75F,
        -0.5F, 0.25F, 1.0F, 0.0F};
    const std::array<float, 2U * dimension> second{
        0.25F, -0.5F, 0.75F, 1.0F,
        1.0F, 1.0F, -1.0F, 0.5F};
    const std::array<std::uint32_t, 2> selected{1U, 0U};
    const std::array<float, heads> sinks{0.125F, -0.25F};
    const std::array<strata::FlashAttentionSegment, 2> segments{{
        {first, {}, {}}, {second, {}, selected}}};
    strata::FlashAttentionRequest request;
    request.queries = queries;
    request.segments = segments;
    request.head_sinks = sinks;
    request.query_rows = 1U;
    request.query_heads = heads;
    request.key_value_heads = 1U;
    request.query_key_dim = dimension;
    request.value_dim = dimension;
    request.scale = 0.5F;
    std::array<float, heads * dimension> output{};
    REQUIRE(strata::flash_attention_reference_f32(request, output).ok());

    const std::array<const float*, 5> rows{
        first.data(), first.data() + dimension, first.data() + 2U * dimension,
        second.data() + dimension, second.data()};
    for (std::uint32_t head = 0U; head < heads; ++head) {
        std::array<double, 5> logits{};
        double maximum = sinks[head];
        for (std::size_t row = 0U; row < rows.size(); ++row) {
            double dot = 0.0;
            for (std::uint32_t dim = 0U; dim < dimension; ++dim) {
                dot += queries[head * dimension + dim] * rows[row][dim];
            }
            logits[row] = dot * 0.5;
            maximum = std::max(maximum, logits[row]);
        }
        double denominator = std::exp(static_cast<double>(sinks[head]) - maximum);
        std::array<double, dimension> expected{};
        for (std::size_t row = 0U; row < rows.size(); ++row) {
            const double weight = std::exp(logits[row] - maximum);
            denominator += weight;
            for (std::uint32_t dim = 0U; dim < dimension; ++dim) {
                expected[dim] += weight * rows[row][dim];
            }
        }
        for (std::uint32_t dim = 0U; dim < dimension; ++dim) {
            REQUIRE_NEAR(output[head * dimension + dim],
                         static_cast<float>(expected[dim] / denominator), 1.0e-6F);
        }
    }
}

TEST_CASE("generic online FlashAttention observes per-query causal limits") {
    const std::array<float, 4> queries{1.0F, 0.0F, 0.0F, 1.0F};
    const std::array<float, 6> keys{1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F};
    const std::array<float, 6> values{2.0F, 0.0F, 0.0F, 4.0F, 8.0F, 8.0F};
    const std::array<std::uint32_t, 2> limits{1U, 3U};
    const std::array<strata::FlashAttentionSegment, 1> segments{{
        {keys, values, {}}}};
    strata::FlashAttentionRequest request;
    request.queries = queries;
    request.segments = segments;
    request.causal_key_counts = limits;
    request.query_rows = 2U;
    request.query_heads = 1U;
    request.key_value_heads = 1U;
    request.query_key_dim = 2U;
    request.value_dim = 2U;
    request.scale = 1.0F;
    std::array<float, 4> output{};
    REQUIRE(strata::flash_attention_reference_f32(request, output).ok());
    REQUIRE_NEAR(output[0], 2.0F, 1.0e-6F);
    REQUIRE_NEAR(output[1], 0.0F, 1.0e-6F);
    REQUIRE(output[2] > 2.0F);
    REQUIRE(output[3] > 4.0F);
}

TEST_CASE("generic FlashAttention scalar oracle preserves compatibility contracts") {
    const std::array<float, 2> query{1.0F, 0.0F};
    const std::array<float, 4> keys{1.0F, 0.0F, 0.0F, 1.0F};
    const std::array<float, 4> values{2.0F, 0.0F, 0.0F, 4.0F};
    const std::array<strata::FlashAttentionSegment, 1> segments{{
        {keys, values, {}}}};
    strata::FlashAttentionRequest request;
    request.queries = query;
    request.segments = segments;
    request.query_rows = 1U;
    request.query_heads = 1U;
    request.key_value_heads = 1U;
    request.query_key_dim = 2U;
    request.value_dim = 2U;
    request.scale = 1.0F;
    const float denominator = std::exp(1.0F) + 1.0F;
    const std::array contracts{
        strata::FlashAttentionNumerics::f64_dot_f32_score_f32_accum,
        strata::FlashAttentionNumerics::f32_dot_f32_softmax_f32_accum};
    for (const auto contract : contracts) {
        request.numerics = contract;
        std::array<float, 2> output{};
        REQUIRE(strata::flash_attention_reference_f32(request, output).ok());
        REQUIRE_NEAR(output[0], 2.0F * std::exp(1.0F) / denominator, 1.0e-6F);
        REQUIRE_NEAR(output[1], 4.0F / denominator, 1.0e-6F);
    }
}

TEST_CASE("FlashAttention production dispatch observes the measured row boundary") {
    REQUIRE(!strata::should_dispatch_flash_attention_cuda(false, 256U, 256U));
    REQUIRE(!strata::should_dispatch_flash_attention_cuda(true, 255U, 256U));
    REQUIRE(strata::should_dispatch_flash_attention_cuda(true, 256U, 256U));
    REQUIRE(strata::should_dispatch_flash_attention_cuda(true, 0U, 0U));
}

TEST_CASE("generic FlashAttention handles sink-only and adversarial logits") {
    const std::array<float, 4> sink_queries{1.0F, 2.0F, 3.0F, 4.0F};
    const std::array<float, 2> sinks{100.0F, -100.0F};
    strata::FlashAttentionRequest sink_request;
    sink_request.queries = sink_queries;
    sink_request.head_sinks = sinks;
    sink_request.query_rows = 1U;
    sink_request.query_heads = 2U;
    sink_request.key_value_heads = 1U;
    sink_request.query_key_dim = 2U;
    sink_request.value_dim = 2U;
    sink_request.scale = 1.0F;
    std::array<float, 4> sink_output{
        -1.0F, -1.0F, -1.0F, -1.0F};
    REQUIRE(strata::flash_attention_reference_f32(
                sink_request, sink_output).ok());
    for (const float value : sink_output) REQUIRE(value == 0.0F);

    const std::array<float, 1> query{1.0F};
    const std::array<float, 2> keys{10'000.0F, -10'000.0F};
    const std::array<float, 2> values{2.0F, 4.0F};
    const std::array<strata::FlashAttentionSegment, 1> segments{{
        {keys, values, {}}}};
    strata::FlashAttentionRequest adversarial_request;
    adversarial_request.queries = query;
    adversarial_request.segments = segments;
    adversarial_request.query_rows = 1U;
    adversarial_request.query_heads = 1U;
    adversarial_request.key_value_heads = 1U;
    adversarial_request.query_key_dim = 1U;
    adversarial_request.value_dim = 1U;
    adversarial_request.scale = 1.0F;
    std::array<float, 1> adversarial_output{};
    REQUIRE(strata::flash_attention_reference_f32(
                adversarial_request, adversarial_output).ok());
    REQUIRE(std::isfinite(adversarial_output[0]));
    REQUIRE_NEAR(adversarial_output[0], 2.0F, 1.0e-6F);

    adversarial_request.maximum_workspace_bytes = 1U;
    REQUIRE(!strata::flash_attention_reference_f32(
                 adversarial_request, adversarial_output).ok());
}

TEST_CASE("generic online FlashAttention rejects invalid exact requests") {
    const std::array<float, 2> query{1.0F, 2.0F};
    const std::array<float, 2> key{1.0F, 2.0F};
    const std::array<std::uint32_t, 1> invalid{2U};
    const std::array<strata::FlashAttentionSegment, 1> segments{{
        {key, {}, invalid}}};
    strata::FlashAttentionRequest request;
    request.queries = query;
    request.segments = segments;
    request.query_rows = 1U;
    request.query_heads = 1U;
    request.key_value_heads = 1U;
    request.query_key_dim = 2U;
    request.value_dim = 2U;
    request.scale = 1.0F;
    std::array<float, 2> output{};
    REQUIRE(!strata::flash_attention_reference_f32(request, output).ok());
}
