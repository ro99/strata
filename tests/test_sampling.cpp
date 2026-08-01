#include "test.hpp"

#include "strata/sampling.hpp"
#include "strata/runtime_support.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <string_view>
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
    auto sampled = strata::sample_logits(
        logits, options, strata::SamplingHistory{counts, {}}, generator);
    REQUIRE(sampled.token == 1U);
    REQUIRE(sampled.top.size() == 2U);

    options.temperature = 1.0;
    options.presence_penalty = 0.0;
    options.top_p = 0.01;
    options.logit_bias = {{2U, 100.0}};
    sampled = strata::sample_logits(logits, options, {}, generator);
    REQUIRE(sampled.token == 2U);
    REQUIRE(sampled.top.size() == 2U);
}

TEST_CASE("reported logprobs describe the model, not the sampler settings") {
    // p = softmax(3,2,1): the top token is about 0.665, i.e. ln p = -0.4076.
    constexpr std::array<float, 3> logits{3.0F, 2.0F, 1.0F};
    std::mt19937_64 generator(7U);
    strata::SamplingOptions options;
    options.temperature = 0.0;
    options.return_logprobs = true;
    options.top_logprobs = 3U;

    const auto plain = strata::sample_logits(logits, options, {}, generator);
    REQUIRE(plain.token == 0U);
    REQUIRE(std::abs(plain.logprob - -0.40760596) < 1e-6);
    REQUIRE(plain.top.size() == 3U);
    REQUIRE(plain.top[0].first == 0U);
    REQUIRE(plain.top[2].first == 2U);

    // Temperature, truncation and a bias all change which token is drawn but
    // must not move the reported probability of whatever was drawn.
    options.temperature = 0.5;
    options.min_p = 0.5;
    options.frequency_penalty = 1.5;
    const auto shaped = strata::sample_logits(logits, options, {}, generator);
    REQUIRE(shaped.token == 0U);
    REQUIRE(std::abs(shaped.logprob - plain.logprob) < 1e-12);
    for (std::size_t index = 0U; index < shaped.top.size(); ++index) {
        REQUIRE(std::abs(shaped.top[index].second - plain.top[index].second) < 1e-12);
    }

    // Natural logprobs are a distribution: they must normalize to one.
    double mass = 0.0;
    for (const auto& [token, logprob] : plain.top) mass += std::exp(logprob);
    REQUIRE(std::abs(mass - 1.0) < 1e-9);
}

TEST_CASE("top_k and min_p truncate against the natural distribution") {
    constexpr std::array<float, 5> logits{5.0F, 4.0F, 1.0F, 0.0F, -8.0F};
    std::mt19937_64 generator(11U);
    strata::SamplingOptions options;
    options.temperature = 1.0;
    options.top_k = 2U;
    std::set<std::uint32_t> observed;
    for (std::size_t iteration = 0U; iteration < 256U; ++iteration) {
        observed.insert(
            strata::sample_logits(logits, options, {}, generator).token);
    }
    REQUIRE(observed.size() == 2U);
    REQUIRE(observed.count(0U) == 1U);
    REQUIRE(observed.count(1U) == 1U);

    // min_p 0.2 keeps p >= 0.2 * p_max, which here is tokens 0 and 1 only.
    // Raising temperature must not widen the survivor set: the threshold is
    // read off the model's distribution, not the rescaled one.
    options.top_k = 0U;
    options.min_p = 0.2;
    options.temperature = 4.0;
    observed.clear();
    for (std::size_t iteration = 0U; iteration < 256U; ++iteration) {
        observed.insert(
            strata::sample_logits(logits, options, {}, generator).token);
    }
    REQUIRE(observed.size() == 2U);
    REQUIRE(observed.count(2U) == 0U);
}

TEST_CASE("XTC drops the safe continuation but keeps a plausible one") {
    constexpr std::array<float, 4> logits{6.0F, 5.5F, 1.0F, 0.5F};
    std::mt19937_64 generator(3U);
    strata::SamplingOptions options;
    options.temperature = 1.0;
    options.xtc_probability = 1.0;
    options.xtc_threshold = 0.1;

    // Tokens 0 and 1 both clear the 10% threshold, so the more likely of them
    // is removed and the weaker survivor stands in for it.
    std::set<std::uint32_t> observed;
    for (std::size_t iteration = 0U; iteration < 256U; ++iteration) {
        observed.insert(
            strata::sample_logits(logits, options, {}, generator).token);
    }
    REQUIRE(observed.count(0U) == 0U);
    REQUIRE(observed.count(1U) == 1U);

    // Never empties the candidate set: with a single token above threshold
    // there is nothing to exclude and the draw is unchanged.
    constexpr std::array<float, 3> peaked{20.0F, -20.0F, -20.0F};
    options.temperature = 0.0;
    REQUIRE(strata::sample_logits(peaked, options, {}, generator).token == 0U);
}

TEST_CASE("repetition stages read the generated sequence") {
    constexpr std::array<float, 4> logits{1.0F, 1.0F, 1.0F, 1.0F};
    std::mt19937_64 generator(5U);
    constexpr std::array<std::uint32_t, 6> history{0U, 1U, 2U, 0U, 1U, 2U};

    // The suffix "1,2" recurred, so DRY penalizes whatever followed it before:
    // token 0. Greedy then prefers the lowest surviving id, token 1.
    strata::SamplingOptions options;
    options.temperature = 0.0;
    options.dry_multiplier = 4.0;
    options.dry_allowed_length = 2U;
    auto sampled = strata::sample_logits(
        logits, options, strata::SamplingHistory{{}, history}, generator);
    REQUIRE(sampled.token != 0U);

    // A hard n-gram ban removes the same continuation outright.
    options.dry_multiplier = 0.0;
    options.no_repeat_ngram = 3U;
    std::set<std::uint32_t> observed;
    options.temperature = 1.0;
    for (std::size_t iteration = 0U; iteration < 256U; ++iteration) {
        observed.insert(strata::sample_logits(
            logits, options, strata::SamplingHistory{{}, history}, generator).token);
    }
    REQUIRE(observed.count(0U) == 0U);
    REQUIRE(observed.size() == 3U);

    // The penalty window bounds repetition control to recent tokens: token 3
    // is outside a two-token window and so is left alone.
    options.no_repeat_ngram = 0U;
    options.temperature = 0.0;
    options.repetition_penalty = 4.0;
    options.penalty_window = 2U;
    constexpr std::array<std::uint32_t, 3> recent{3U, 0U, 1U};
    sampled = strata::sample_logits(
        logits, options, strata::SamplingHistory{{}, recent}, generator);
    REQUIRE(sampled.token == 2U);
}

TEST_CASE("typical sampling keeps the tokens nearest the distribution entropy") {
    // p is about 0.451 for token 0 and 0.061 for each of the nine others, so
    // H = 1.895 nats. The leader's surprisal (0.797) sits further from H than
    // the body's (2.797) does, and typical sampling drops the leader for it.
    // The point of the stage is that it is not a probability cutoff: a token
    // is kept for being unsurprising in the right amount, not for being likely.
    std::array<float, 10> logits{};
    logits[0] = 2.0F;
    std::mt19937_64 generator(13U);
    strata::SamplingOptions options;
    options.temperature = 1.0;
    options.typical_p = 0.5;
    std::set<std::uint32_t> observed;
    for (std::size_t iteration = 0U; iteration < 512U; ++iteration) {
        observed.insert(
            strata::sample_logits(logits, options, {}, generator).token);
    }
    REQUIRE(observed.count(0U) == 0U);
    REQUIRE(observed.size() == 9U);

    // On a peaked distribution H collapses and the leader becomes the typical
    // token, so the stage must keep it rather than exclude it on principle.
    constexpr std::array<float, 5> peaked{8.0F, 1.0F, 1.0F, 1.0F, 1.0F};
    options.temperature = 0.0;
    REQUIRE(strata::sample_logits(peaked, options, {}, generator).token == 0U);
}

TEST_CASE("the pipeline still draws from the distribution it reports") {
    // Gumbel-max must remain an exact softmax draw after truncation, not just
    // a plausible-looking one: the whole pipeline is only worth having if the
    // survivor set is sampled at its own renormalized probabilities.
    const std::array<float, 4> logits{
        std::log(0.5F), std::log(0.3F), std::log(0.15F), std::log(0.05F)};
    constexpr std::size_t kDraws = 40'000U;
    std::mt19937_64 generator(31U);

    strata::SamplingOptions options;
    options.temperature = 1.0;
    std::array<std::size_t, 4> tally{};
    for (std::size_t draw = 0U; draw < kDraws; ++draw) {
        ++tally[strata::sample_logits(logits, options, {}, generator).token];
    }
    const std::array<double, 4> expected{0.5, 0.3, 0.15, 0.05};
    for (std::size_t token = 0U; token < tally.size(); ++token) {
        const double observed =
            static_cast<double>(tally[token]) / static_cast<double>(kDraws);
        REQUIRE(std::abs(observed - expected[token]) < 0.01);
    }

    // top_k 2 keeps 0.5 and 0.3, which renormalize to 0.625 and 0.375.
    options.top_k = 2U;
    tally = {};
    for (std::size_t draw = 0U; draw < kDraws; ++draw) {
        ++tally[strata::sample_logits(logits, options, {}, generator).token];
    }
    REQUIRE(tally[2] == 0U);
    REQUIRE(tally[3] == 0U);
    const double leader =
        static_cast<double>(tally[0]) / static_cast<double>(kDraws);
    REQUIRE(std::abs(leader - 0.625) < 0.01);

    // Temperature 2 flattens the same survivor set toward uniform.
    options.temperature = 2.0;
    tally = {};
    for (std::size_t draw = 0U; draw < kDraws; ++draw) {
        ++tally[strata::sample_logits(logits, options, {}, generator).token];
    }
    const double flattened =
        static_cast<double>(tally[0]) / static_cast<double>(kDraws);
    REQUIRE(flattened < leader);
    REQUIRE(std::abs(flattened - 0.5642) < 0.01);
}

TEST_CASE("sampler validation rejects each out-of-range knob") {
    std::string error;
    REQUIRE(strata::validate_sampling_options(strata::greedy_sampling(), error));
    REQUIRE(error.empty());

    const auto rejects = [&error](auto mutate, std::string_view field) {
        strata::SamplingOptions options;
        mutate(options);
        return !strata::validate_sampling_options(options, error) && error == field;
    };
    REQUIRE(rejects([](auto& o) { o.temperature = 11.0; }, "temperature"));
    REQUIRE(rejects([](auto& o) { o.top_p = 0.0; }, "top_p"));
    REQUIRE(rejects([](auto& o) { o.min_p = 1.0; }, "min_p"));
    REQUIRE(rejects([](auto& o) { o.typical_p = 0.0; }, "typical_p"));
    REQUIRE(rejects([](auto& o) { o.xtc_probability = 1.5; }, "xtc_probability"));
    REQUIRE(rejects([](auto& o) { o.xtc_threshold = 0.0; }, "xtc_threshold"));
    REQUIRE(rejects([](auto& o) { o.repetition_penalty = 0.0; }, "repetition_penalty"));
    REQUIRE(rejects([](auto& o) { o.dry_base = 0.5; }, "dry_base"));
    REQUIRE(rejects([](auto& o) { o.dry_allowed_length = 0U; }, "dry_allowed_length"));
    REQUIRE(rejects([](auto& o) {
        o.temperature = std::numeric_limits<double>::quiet_NaN();
    }, "temperature"));
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
