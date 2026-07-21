#include "test.hpp"

#include "strata/tokenizer.hpp"

#include <array>
#include <filesystem>
#include <string>

namespace {

constexpr auto kPrompt = "What is the closer start to sun, and how distant it is from it?";

std::filesystem::path tokenizer_fixture() {
    auto path = std::filesystem::path(STRATA_SOURCE_DIR) / "models/glm52/tokenizer.json";
    if (!std::filesystem::exists(path)) {
        path = std::filesystem::path(STRATA_SOURCE_DIR) / "build/glm52-tokenizer.json";
    }
    return path;
}

std::filesystem::path deepseek_tokenizer_fixture() {
    return std::filesystem::path(STRATA_SOURCE_DIR) /
           "models/DeepSeek-V4-Flash-DSpark/tokenizer.json";
}

}  // namespace

TEST_CASE("GLM single-user chat rendering matches the pinned template") {
    REQUIRE(strata::render_glm52_user_prompt(kPrompt) ==
            "[gMASK]<sop>\n<|system|>Reasoning Effort: Medium-high<|user|>"
            "What is the closer start to sun, and how distant it is from it?"
            "<|assistant|><think>");
}

TEST_CASE("chat rendering includes prior user and assistant turns") {
    const std::array messages{
        strata::ChatMessage{strata::ChatRole::User, "Capital of France?"},
        strata::ChatMessage{strata::ChatRole::Assistant, "Paris"},
        strata::ChatMessage{strata::ChatRole::User, "And its population?"},
    };
    REQUIRE(strata::render_glm52_chat_prompt(messages, "medium-high", false) ==
            "[gMASK]<sop>\n<|user|>Capital of France?"
            "<|assistant|><think></think>Paris"
            "<|user|>And its population?<|assistant|><think></think>");
    REQUIRE(strata::render_deepseek_v4_chat_prompt(messages) ==
            "<｜begin▁of▁sentence｜><｜User｜>Capital of France?"
            "<｜Assistant｜></think>Paris<｜User｜>And its population?"
            "<｜Assistant｜></think>");
}

TEST_CASE("real GLM tokenizer produces the frozen baseline prompt ids when available") {
    const auto path = tokenizer_fixture();
    if (!std::filesystem::exists(path)) SKIP("pinned GLM tokenizer fixture is absent");
    const auto tokenizer = strata::ModelTokenizer::load(path.string());
    REQUIRE(tokenizer.ok());
    const auto rendered = strata::render_glm52_user_prompt(kPrompt);
    const auto encoded = tokenizer.value.encode(rendered);
    REQUIRE(encoded.ok());
    constexpr std::array<std::uint32_t, 30> expected{
        154822, 154824, 198,    154826, 25062, 287, 29905, 371, 25, 24283,
        27469,  154827, 3838,   374,    279,   12122, 1191, 311, 7015, 11,
        323,    1246,   28624,  432,    374,   504,   432,  30,  154828, 154841};
    REQUIRE(encoded.value.size() == expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        REQUIRE(encoded.value[index] == expected[index]);
    }
    const auto decoded = tokenizer.value.decode(encoded.value);
    REQUIRE(decoded.ok());
    REQUIRE(decoded.value == rendered);
}

TEST_CASE("GLM tokenizer matches the canonical Unicode byte-level BPE ids") {
    const auto path = tokenizer_fixture();
    if (!std::filesystem::exists(path)) SKIP("pinned GLM tokenizer fixture is absent");
    const auto tokenizer = strata::ModelTokenizer::load(path.string());
    REQUIRE(tokenizer.ok());
    const auto encoded = tokenizer.value.encode("olá");
    REQUIRE(encoded.ok());
    REQUIRE(encoded.value == std::vector<std::uint32_t>({337U, 1953U}));
}

TEST_CASE("committed tokenizer pretoken boundaries cover both model contracts") {
    const auto glm = strata::pretokenize(strata::TokenizerContract::Glm52,
                                         "olá  world");
    REQUIRE(glm.ok());
    REQUIRE(glm.value == std::vector<std::string>({"olá", " ", " world"}));

    const auto deepseek = strata::pretokenize(
        strata::TokenizerContract::DeepSeekV4, "é1234 日本語test !hello");
    REQUIRE(deepseek.ok());
    REQUIRE(deepseek.value == std::vector<std::string>(
        {"é", "123", "4", " ", "日本語", "test", " !", "hello"}));
}

TEST_CASE("real DeepSeek V4 tokenizer and single-user chat rendering are supported") {
    const auto path = deepseek_tokenizer_fixture();
    if (!std::filesystem::exists(path)) SKIP("pinned DeepSeek tokenizer fixture is absent");
    const auto tokenizer = strata::ModelTokenizer::load(path.string());
    REQUIRE(tokenizer.ok());
    const auto rendered = strata::render_deepseek_v4_user_prompt("hello");
    REQUIRE(rendered ==
            "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>");
    const auto encoded = tokenizer.value.encode(rendered);
    REQUIRE(encoded.ok());
    constexpr std::array<std::uint32_t, 5> expected{
        0U, 128803U, 33310U, 128804U, 128822U};
    REQUIRE(encoded.value.size() == expected.size());
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        REQUIRE(encoded.value[index] == expected[index]);
    }
}

TEST_CASE("DeepSeek V4 byte-level decode produces valid UTF-8 across adjacent tokens") {
    const auto path = deepseek_tokenizer_fixture();
    if (!std::filesystem::exists(path)) SKIP("pinned DeepSeek tokenizer fixture is absent");
    const auto tokenizer = strata::ModelTokenizer::load(path.string());
    REQUIRE(tokenizer.ok());

    const auto t130 = tokenizer.value.decode_token(130U);
    REQUIRE(t130.ok());
    REQUIRE(t130.value == std::string("\xC3", 1));

    const auto t105 = tokenizer.value.decode_token(105U);
    REQUIRE(t105.ok());
    REQUIRE(t105.value == std::string("\xA9", 1));

    const std::array<std::uint32_t, 2> pair{130U, 105U};
    const auto combined = tokenizer.value.decode(pair);
    REQUIRE(combined.ok());
    REQUIRE(combined.value == "\xC3\xA9");
}

TEST_CASE("DeepSeek V4 tokenizer matches Unicode and digit-split canonical ids") {
    const auto path = deepseek_tokenizer_fixture();
    if (!std::filesystem::exists(path)) SKIP("pinned DeepSeek tokenizer fixture is absent");
    const auto tokenizer = strata::ModelTokenizer::load(path.string());
    REQUIRE(tokenizer.ok());
    const auto encoded = tokenizer.value.encode("é1234");
    REQUIRE(encoded.ok());
    REQUIRE(encoded.value ==
            std::vector<std::uint32_t>({619U, 6895U, 22U}));
}
