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

TEST_CASE("real GLM tokenizer produces the frozen baseline prompt ids when available") {
    const auto path = tokenizer_fixture();
    if (!std::filesystem::exists(path)) return;
    const auto tokenizer = strata::GlmTokenizer::load(path.string());
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

TEST_CASE("GLM tokenizer rejects unsupported Unicode input explicitly") {
    const auto path = tokenizer_fixture();
    if (!std::filesystem::exists(path)) return;
    const auto tokenizer = strata::GlmTokenizer::load(path.string());
    REQUIRE(tokenizer.ok());
    REQUIRE(!tokenizer.value.encode("olá").ok());
}

TEST_CASE("real DeepSeek V4 tokenizer and single-user chat rendering are supported") {
    const auto path = deepseek_tokenizer_fixture();
    if (!std::filesystem::exists(path)) return;
    const auto tokenizer = strata::GlmTokenizer::load(path.string());
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
