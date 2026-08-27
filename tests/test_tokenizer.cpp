#include "test.hpp"

#include "strata/models/common/tokenizer.hpp"

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
           "models/dsv4f/tokenizer.json";
}

std::filesystem::path gemma4_tokenizer_fixture() {
    return std::filesystem::path(STRATA_SOURCE_DIR) /
           "models/gemma4/tokenizer.json";
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
            "<｜Assistant｜></think>Paris<｜end▁of▁sentence｜>"
            "<｜User｜>And its population?"
            "<｜Assistant｜></think>");
}

TEST_CASE("DeepSeek V4 Flash 0731 rendering follows system and tool-result encoding") {
    const std::array messages{
        strata::ChatMessage{strata::ChatRole::System, "Be concise."},
        strata::ChatMessage{strata::ChatRole::User, "Check it."},
        strata::ChatMessage{strata::ChatRole::Assistant, "Calling tools."},
        strata::ChatMessage{strata::ChatRole::Tool, "{\"ok\":true}"},
        strata::ChatMessage{strata::ChatRole::Tool, "done"},
    };
    REQUIRE(strata::render_deepseek_v4_chat_prompt(messages, true) ==
            "<｜begin▁of▁sentence｜>Be concise.<｜User｜>Check it."
            "<｜Assistant｜></think>Calling tools.<｜end▁of▁sentence｜>"
            "<｜User｜><tool_result>{\"ok\":true}</tool_result>\n\n"
            "<tool_result>done</tool_result><｜Assistant｜><think>");
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

TEST_CASE("Gemma 4 chat rendering and SentencePiece BPE match the target tokenizer") {
    const auto path = gemma4_tokenizer_fixture();
    if (!std::filesystem::exists(path)) SKIP("pinned Gemma 4 tokenizer fixture is absent");
    const auto tokenizer = strata::ModelTokenizer::load(path.string());
    REQUIRE(tokenizer.ok());
    REQUIRE(tokenizer.value.vocabulary_size() == 262144U);

    const auto rendered = strata::render_gemma4_user_prompt("hello");
    REQUIRE(rendered ==
            "<bos><|turn>user\nhello<turn|>\n<|turn>model\n"
            "<|channel>thought\n<channel|>");
    const auto encoded = tokenizer.value.encode(rendered);
    REQUIRE(encoded.ok());
    REQUIRE(encoded.value == std::vector<std::uint32_t>(
        {2U, 105U, 2364U, 107U, 23391U, 106U, 107U, 105U, 4368U,
         107U, 100U, 45518U, 107U, 101U}));

    const auto unicode = tokenizer.value.encode(" hello olá 日本語");
    REQUIRE(unicode.ok());
    REQUIRE(unicode.value == std::vector<std::uint32_t>(
        {29104U, 4276U, 236898U, 33375U, 238582U}));
    const auto decoded = tokenizer.value.decode(unicode.value);
    REQUIRE(decoded.ok());
    REQUIRE(decoded.value == " hello olá 日本語");
}

TEST_CASE("non-ASCII letters are not pretokenized as digit runs") {
    // `unicode_number` narrowed the codepoint to `unsigned char` before the
    // ASCII digit test, so every codepoint whose low byte landed in 0x30-0x39
    // classified as a number. Cyrillic а-й is U+0430-U+0439, so `Привет мир`
    // split into six pretokens instead of two, and every contract sharing the
    // classifier carried it.
    const auto cyrillic = strata::pretokenize(strata::TokenizerContract::Glm52,
                                              "Привет мир");
    REQUIRE(cyrillic.ok());
    REQUIRE(cyrillic.value == std::vector<std::string>({"Привет", " мир"}));

    const auto deepseek = strata::pretokenize(
        strata::TokenizerContract::DeepSeekV4, "Привет мир");
    REQUIRE(deepseek.ok());
    REQUIRE(deepseek.value == std::vector<std::string>({"Привет", " мир"}));

    // Armenian ա-ի (U+0561-U+056A) shares the low byte range with 0x30-0x39
    // one block up, and the digit runs the classifier does own must still split
    // from the letters around them.
    const auto armenian = strata::pretokenize(strata::TokenizerContract::Glm52,
                                              "աբգ");
    REQUIRE(armenian.ok());
    REQUIRE(armenian.value == std::vector<std::string>({"աբգ"}));

    const auto digits = strata::pretokenize(strata::TokenizerContract::Glm52,
                                            "abc123");
    REQUIRE(digits.ok());
    REQUIRE(digits.value == std::vector<std::string>({"abc", "123"}));
}
