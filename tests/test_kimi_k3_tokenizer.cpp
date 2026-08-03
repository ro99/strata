#include "kimi_fixture.hpp"
#include "test.hpp"

#include "strata/chat_protocol.hpp"
#include "strata/tokenizer.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// Gate 7: the Kimi-K3 tokenizer against the checkpoint's own tokenizer. The
// fixture comes from `scripts/kimi_k3_tokenizer_fixture.py`, which drives the
// real `TikTokenTokenizer` through `AutoTokenizer.from_pretrained`, so the
// comparison is against a running tokenizer rather than against a reading of
// its regex.
//
// The cases are chosen to separate the alternatives of Kimi's `pat_str`. Most
// of them look like ordinary text and would pass under a wrong pretokenizer;
// `CamelCaseIdentifier`, `HTTPServerError`, `漢字とかなとカナ`, and the
// whitespace cases are the ones that do not. A tokenizer that treats all CJK
// alike, or that has one `\p{L}+` alternative instead of the two cased ones,
// fails exactly there and nowhere else.
//
// Unlike the numeric gates this one has no tolerance: token ids are exact or
// they are wrong.

namespace {

using kimi_test::kimi_directory;
using kimi_test::kimi_present;

// The fixture is small, flat JSON. A full parser is not warranted for six keys,
// but the reads are checked rather than assumed: a silently missing case would
// turn this gate into a test that passes by not running.
class Json {
public:
    [[nodiscard]] bool load(const std::string& path) {
        std::ifstream stream(path, std::ios::binary);
        if (!stream) return false;
        text_.assign(std::istreambuf_iterator<char>(stream),
                     std::istreambuf_iterator<char>());
        return !text_.empty();
    }

    // Every `"name": "<value>"` object in the `cases`/`chats` arrays, returned
    // as the decoded string alongside its `ids` array.
    struct Record {
        std::string name;
        std::string text;
        std::vector<std::uint32_t> ids;
    };

    [[nodiscard]] std::vector<Record> records(const std::string& array,
                                              const std::string& text_key) const {
        std::vector<Record> found;
        auto cursor = text_.find("\"" + array + "\"");
        if (cursor == std::string::npos) return found;
        const auto limit = text_.find("\n ]", cursor);
        while (true) {
            const auto name = field(cursor, "name", limit);
            if (!name) break;
            Record record;
            record.name = decode(name->first);
            cursor = name->second;
            const auto body = field(cursor, text_key, limit);
            if (!body) break;
            record.text = decode(body->first);
            cursor = body->second;
            const auto ids = integers(cursor, "ids", limit);
            if (!ids) break;
            record.ids = ids->first;
            cursor = ids->second;
            found.push_back(std::move(record));
        }
        return found;
    }

private:
    // The raw JSON string body between the quotes following `"key":`.
    [[nodiscard]] std::optional<std::pair<std::string, std::size_t>> field(
        std::size_t from, const std::string& key, std::size_t limit) const {
        const auto at = text_.find("\"" + key + "\":", from);
        if (at == std::string::npos || at > limit) return std::nullopt;
        auto cursor = text_.find('"', at + key.size() + 3U);
        if (cursor == std::string::npos) return std::nullopt;
        ++cursor;
        std::string raw;
        while (cursor < text_.size() && text_[cursor] != '"') {
            if (text_[cursor] == '\\') {
                raw.push_back(text_[cursor++]);
                if (cursor < text_.size()) raw.push_back(text_[cursor++]);
                continue;
            }
            raw.push_back(text_[cursor++]);
        }
        return std::pair{raw, cursor};
    }

    [[nodiscard]] std::optional<std::pair<std::vector<std::uint32_t>, std::size_t>>
    integers(std::size_t from, const std::string& key, std::size_t limit) const {
        const auto at = text_.find("\"" + key + "\":", from);
        if (at == std::string::npos || at > limit) return std::nullopt;
        auto cursor = text_.find('[', at);
        if (cursor == std::string::npos) return std::nullopt;
        std::vector<std::uint32_t> values;
        std::string digits;
        for (++cursor; cursor < text_.size() && text_[cursor] != ']'; ++cursor) {
            if (text_[cursor] >= '0' && text_[cursor] <= '9') {
                digits.push_back(text_[cursor]);
                continue;
            }
            if (!digits.empty()) {
                values.push_back(static_cast<std::uint32_t>(std::stoul(digits)));
                digits.clear();
            }
        }
        if (!digits.empty()) {
            values.push_back(static_cast<std::uint32_t>(std::stoul(digits)));
        }
        return std::pair{values, cursor};
    }

    [[nodiscard]] static std::string decode(const std::string& raw) {
        std::string out;
        for (std::size_t index = 0U; index < raw.size(); ++index) {
            if (raw[index] != '\\') {
                out.push_back(raw[index]);
                continue;
            }
            if (++index >= raw.size()) break;
            switch (raw[index]) {
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'u': {
                    const auto hex = raw.substr(index + 1U, 4U);
                    index += 4U;
                    auto codepoint = static_cast<std::uint32_t>(
                        std::stoul(hex, nullptr, 16));
                    if (codepoint >= 0xD800U && codepoint <= 0xDBFFU &&
                        index + 6U < raw.size() && raw[index + 1U] == '\\' &&
                        raw[index + 2U] == 'u') {
                        const auto low = static_cast<std::uint32_t>(
                            std::stoul(raw.substr(index + 3U, 4U), nullptr, 16));
                        index += 6U;
                        codepoint = 0x10000U + ((codepoint - 0xD800U) << 10U) +
                                    (low - 0xDC00U);
                    }
                    if (codepoint < 0x80U) {
                        out.push_back(static_cast<char>(codepoint));
                    } else if (codepoint < 0x800U) {
                        out.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
                        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
                    } else if (codepoint < 0x10000U) {
                        out.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
                        out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
                        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
                    } else {
                        out.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
                        out.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
                        out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
                        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
                    }
                    break;
                }
                default: out.push_back(raw[index]); break;
            }
        }
        return out;
    }

    std::string text_;
};

std::string join(const std::vector<std::uint32_t>& ids, std::size_t limit = 24U) {
    std::string text;
    for (std::size_t index = 0U; index < ids.size() && index < limit; ++index) {
        if (index != 0U) text += ' ';
        text += std::to_string(ids[index]);
    }
    if (ids.size() > limit) text += " ...";
    return text;
}

}  // namespace

TEST_CASE("Kimi-K3 tokenizer matches the reference on every pretokenizer class") {
    if (!kimi_present()) SKIP("pinned Kimi-K3 checkpoint is absent");
    Json fixture;
    if (!fixture.load(kimi_test::fixture_path("kimi-k3-tokenizer.json"))) {
        SKIP("tokenizer fixture is absent; run "
             "scripts/kimi_k3_tokenizer_fixture.py");
    }
    const auto cases = fixture.records("cases", "text");
    REQUIRE(cases.size() >= 30U);

    auto loaded = strata::ModelTokenizer::load_kimi_k3(kimi_directory());
    REQUIRE(loaded.ok());
    const auto& tokenizer = loaded.value;
    REQUIRE(tokenizer.vocabulary_size() == 163'840U);

    std::uint32_t mismatches = 0U;
    for (const auto& record : cases) {
        const auto encoded = tokenizer.encode(record.text);
        if (!encoded.ok()) {
            std::cout << "  [gate 7] " << record.name << ": "
                      << encoded.errors.front() << '\n';
            ++mismatches;
            continue;
        }
        if (encoded.value != record.ids) {
            std::cout << "  [gate 7] " << record.name << " encode mismatch\n"
                      << "    measured  " << join(encoded.value) << '\n'
                      << "    reference " << join(record.ids) << '\n';
            ++mismatches;
            continue;
        }
        const auto decoded = tokenizer.decode(record.ids);
        if (!decoded.ok() || decoded.value != record.text) {
            std::cout << "  [gate 7] " << record.name << " decode mismatch\n";
            ++mismatches;
        }
    }
    std::cout << "  [gate 7] " << (cases.size() - mismatches) << '/'
              << cases.size() << " cases round-trip exactly\n";
    REQUIRE(mismatches == 0U);
}

TEST_CASE("Kimi-K3 XTML chat rendering matches the reference") {
    if (!kimi_present()) SKIP("pinned Kimi-K3 checkpoint is absent");
    Json fixture;
    if (!fixture.load(kimi_test::fixture_path("kimi-k3-tokenizer.json"))) {
        SKIP("tokenizer fixture is absent; run "
             "scripts/kimi_k3_tokenizer_fixture.py");
    }
    const auto chats = fixture.records("chats", "rendered");
    REQUIRE(chats.size() >= 4U);

    auto loaded = strata::ModelTokenizer::load_kimi_k3(kimi_directory());
    REQUIRE(loaded.ok());
    const auto& tokenizer = loaded.value;

    // The fixture's conversations, in the same order as `CHATS` in the oracle.
    const std::vector<std::pair<std::vector<strata::ChatMessage>, bool>> conversations{
        {{{strata::ChatRole::User, "Hello"}}, false},
        {{{strata::ChatRole::System, "You are terse."},
          {strata::ChatRole::User, "Hi"}}, false},
        {{{strata::ChatRole::User, "What is 2+2?"},
          {strata::ChatRole::Assistant, "4"},
          {strata::ChatRole::User, "And 3+3?"}}, false},
        {{{strata::ChatRole::User, "Think about it."}}, true},
    };
    REQUIRE(conversations.size() == chats.size());

    std::uint32_t mismatches = 0U;
    for (std::size_t index = 0U; index < chats.size(); ++index) {
        const auto& [messages, thinking] = conversations[index];
        const auto rendered =
            strata::render_kimi_k3_chat_prompt(messages, thinking);
        if (rendered != chats[index].text) {
            std::cout << "  [gate 7] " << chats[index].name
                      << " rendering mismatch\n    measured  " << rendered
                      << "\n    reference " << chats[index].text << '\n';
            ++mismatches;
            continue;
        }
        const auto encoded = tokenizer.encode(rendered);
        if (!encoded.ok() || encoded.value != chats[index].ids) {
            std::cout << "  [gate 7] " << chats[index].name
                      << " token mismatch\n    measured  "
                      << join(encoded.value) << "\n    reference "
                      << join(chats[index].ids) << '\n';
            ++mismatches;
        }
    }
    std::cout << "  [gate 7] " << (chats.size() - mismatches) << '/'
              << chats.size() << " conversations render and tokenize exactly\n";
    REQUIRE(mismatches == 0U);
}
