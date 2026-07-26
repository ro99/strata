#include "test.hpp"

#include "strata/sampling.hpp"
#include "strata/runtime_support.hpp"

#include <array>
#include <random>
#include <set>
#include <string>
#include <vector>

TEST_CASE("zero-temperature sampling is greedy") {
    constexpr std::array<float, 4> logits{-3.0F, 2.0F, 9.0F, 1.0F};
    std::mt19937_64 generator(7U);
    REQUIRE(strata::sample_logits_gumbel(logits, 0.0, generator) == 2U);
}

TEST_CASE("seeded Gumbel-max sampling is deterministic and non-degenerate") {
    constexpr std::array<float, 4> logits{};
    std::mt19937_64 first(33'377'335U);
    std::mt19937_64 second(33'377'335U);
    std::set<std::uint32_t> observed;
    for (std::size_t iteration = 0U; iteration < 32U; ++iteration) {
        const auto first_token = strata::sample_logits_gumbel(logits, 1.0, first);
        const auto second_token = strata::sample_logits_gumbel(logits, 1.0, second);
        REQUIRE(first_token == second_token);
        REQUIRE(first_token < logits.size());
        observed.insert(first_token);
    }
    REQUIRE(observed.size() > 1U);
}

TEST_CASE("sampling applies nucleus, penalties, bias, and logprobs") {
    constexpr std::array<float, 3> logits{3.0F, 2.0F, 1.0F};
    std::mt19937_64 generator(7U);
    strata::SamplingOptions options;
    options.temperature = 0.0;
    options.presence_penalty = 2.0;
    options.top_logprobs = 2U;
    constexpr std::array<std::uint32_t, 3> counts{1U, 0U, 0U};
    auto sampled = strata::sample_logits(logits, options, counts, generator);
    REQUIRE(sampled.token == 1U);
    REQUIRE(sampled.top.size() == 1U);

    options.temperature = 1.0;
    options.presence_penalty = 0.0;
    options.top_p = 0.01;
    options.logit_bias = {{2U, 100.0}};
    sampled = strata::sample_logits(logits, options, {}, generator);
    REQUIRE(sampled.token == 2U);
    REQUIRE(sampled.top.size() == 2U);
}

TEST_CASE("stop sequences are withheld across token-piece boundaries") {
    const std::array<std::string, 1> stops{"END"};
    strata::StopSequenceBuffer buffer(stops);
    std::string streamed;
    const strata::TokenStreamCallback callback =
        [&streamed](std::uint32_t, std::string_view piece) { streamed += piece; };
    buffer.append(1U, "hello E", callback);
    REQUIRE(streamed == "hello ");
    buffer.append(2U, "ND ignored", callback);
    buffer.finish(callback);
    REQUIRE(buffer.stopped());
    REQUIRE(buffer.text() == "hello ");
    REQUIRE(streamed == "hello ");
}

TEST_CASE("chat history trimming preserves system messages and the newest turn") {
    std::vector<strata::ChatMessage> messages{
        {strata::ChatRole::System, "rules"},
        {strata::ChatRole::User, "old"},
        {strata::ChatRole::Assistant, "answer"},
        {strata::ChatRole::Tool, "tool one"},
        {strata::ChatRole::Tool, "tool two"},
        {strata::ChatRole::User, "new"},
    };
    REQUIRE(strata::trim_oldest_chat_turn(messages));
    REQUIRE(messages.size() == 2U);
    REQUIRE(messages[0].role == strata::ChatRole::System);
    REQUIRE(messages[1].content == "new");
    REQUIRE(!strata::trim_oldest_chat_turn(messages));
}

TEST_CASE("UTF-8 streaming withholds incomplete code points") {
    const std::string first{"x\xE2\x98", 3U};
    REQUIRE(strata::complete_utf8_prefix(first) == 1U);
    REQUIRE(strata::complete_utf8_prefix(first + "\x83") == 4U);
    REQUIRE(strata::complete_utf8_prefix("\xC0\x80") == std::string_view::npos);
}
