#include "strata/models/common/tokenizer.hpp"

#include "../../platform/json_cursor.hpp"
#include "../inkling/inkling_unicode.hpp"
#include "../laguna/laguna_unicode.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <optional>
#include <queue>
#include <utility>

namespace strata {

namespace {

using detail::JsonCursor;

[[nodiscard]] std::string utf8(std::uint32_t codepoint) {
    std::string output;
    if (codepoint < 0x80U) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint < 0x800U) {
        output.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
        output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
    return output;
}

[[nodiscard]] std::optional<std::pair<std::uint32_t, std::size_t>> decode_utf8(
    std::string_view text, std::size_t offset) {
    if (offset >= text.size()) return std::nullopt;
    const auto first = static_cast<unsigned char>(text[offset]);
    if (first < 0x80U) return std::pair{static_cast<std::uint32_t>(first), 1U};
    std::size_t bytes = 0;
    std::uint32_t codepoint = 0;
    if ((first & 0xE0U) == 0xC0U) {
        bytes = 2;
        codepoint = first & 0x1FU;
    } else if ((first & 0xF0U) == 0xE0U) {
        bytes = 3;
        codepoint = first & 0x0FU;
    } else if ((first & 0xF8U) == 0xF0U) {
        bytes = 4;
        codepoint = first & 0x07U;
    } else {
        return std::nullopt;
    }
    if (text.size() - offset < bytes) return std::nullopt;
    for (std::size_t index = 1; index < bytes; ++index) {
        const auto continuation = static_cast<unsigned char>(text[offset + index]);
        if ((continuation & 0xC0U) != 0x80U) return std::nullopt;
        codepoint = (codepoint << 6U) | (continuation & 0x3FU);
    }
    if ((bytes == 2U && codepoint < 0x80U) ||
        (bytes == 3U && codepoint < 0x800U) ||
        (bytes == 4U && codepoint < 0x10000U) ||
        codepoint > 0x10FFFFU ||
        (codepoint >= 0xD800U && codepoint <= 0xDFFFU)) {
        return std::nullopt;
    }
    return std::pair{codepoint, bytes};
}

[[nodiscard]] bool ascii_letter(unsigned char value) noexcept {
    return (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z');
}

[[nodiscard]] bool ascii_number(unsigned char value) noexcept {
    return value >= '0' && value <= '9';
}

[[nodiscard]] bool ascii_space(unsigned char value) noexcept {
    return value == ' ' || value == '\t' || value == '\n' || value == '\r' ||
           value == '\v' || value == '\f';
}

[[nodiscard]] unsigned char ascii_lower(unsigned char value) noexcept {
    return value >= 'A' && value <= 'Z' ? static_cast<unsigned char>(value + 32U) : value;
}

[[nodiscard]] std::string merge_key(std::string_view left, std::string_view right) {
    std::string key;
    key.reserve(left.size() + right.size() + 1U);
    key.append(left);
    key.push_back('\0');
    key.append(right);
    return key;
}

void parse_added_tokens(JsonCursor& cursor,
                        std::vector<std::pair<std::string, std::uint32_t>>& output) {
    cursor.expect('[');
    if (cursor.consume(']')) return;
    for (;;) {
        std::optional<std::uint32_t> id;
        std::optional<std::string> content;
        cursor.expect('{');
        if (!cursor.consume('}')) {
            for (;;) {
                const auto key = cursor.parse_string();
                cursor.expect(':');
                if (key == "id") {
                    const auto parsed = cursor.parse_uint64();
                    if (parsed > std::numeric_limits<std::uint32_t>::max()) {
                        throw std::runtime_error("added token id exceeds uint32");
                    }
                    id = static_cast<std::uint32_t>(parsed);
                } else if (key == "content") {
                    content = cursor.parse_string();
                } else {
                    cursor.skip_value();
                }
                if (cursor.consume('}')) break;
                cursor.expect(',');
            }
        }
        if (!id || !content) throw std::runtime_error("added token is missing id or content");
        output.emplace_back(std::move(*content), *id);
        if (cursor.consume(']')) return;
        cursor.expect(',');
    }
}

void parse_vocabulary(JsonCursor& cursor,
                      std::unordered_map<std::string, std::uint32_t>& vocabulary,
                      std::vector<std::string>& id_to_piece) {
    cursor.expect('{');
    if (cursor.consume('}')) return;
    for (;;) {
        auto piece = cursor.parse_string();
        cursor.expect(':');
        const auto parsed = cursor.parse_uint64();
        if (parsed > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("vocabulary id exceeds uint32");
        }
        const auto id = static_cast<std::uint32_t>(parsed);
        if (id_to_piece.size() <= id) id_to_piece.resize(static_cast<std::size_t>(id) + 1U);
        if (!id_to_piece[id].empty() || !vocabulary.emplace(piece, id).second) {
            throw std::runtime_error("duplicate vocabulary piece or id");
        }
        id_to_piece[id] = std::move(piece);
        if (cursor.consume('}')) return;
        cursor.expect(',');
    }
}

void parse_merges(JsonCursor& cursor,
                  std::unordered_map<std::string, std::uint32_t>& merge_ranks) {
    std::uint32_t rank = 0;
    cursor.expect('[');
    if (cursor.consume(']')) return;
    for (;;) {
        std::string left;
        std::string right;
        if (cursor.peek() == '[') {
            cursor.expect('[');
            left = cursor.parse_string();
            cursor.expect(',');
            right = cursor.parse_string();
            cursor.expect(']');
        } else {
            auto pair = cursor.parse_string();
            const auto divider = pair.find(' ');
            if (divider == std::string::npos || divider == 0U ||
                divider + 1U >= pair.size()) {
                throw std::runtime_error("string tokenizer merge is not a token pair");
            }
            left = pair.substr(0U, divider);
            right = pair.substr(divider + 1U);
        }
        if (!merge_ranks.emplace(merge_key(left, right), rank).second) {
            throw std::runtime_error("duplicate tokenizer merge");
        }
        ++rank;
        if (cursor.consume(']')) return;
        cursor.expect(',');
    }
}

void parse_model(JsonCursor& cursor, bool& ignore_merges,
                 std::unordered_map<std::string, std::uint32_t>& vocabulary,
                 std::vector<std::string>& id_to_piece,
                 std::unordered_map<std::string, std::uint32_t>& merge_ranks) {
    cursor.expect('{');
    if (cursor.consume('}')) return;
    for (;;) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        if (key == "type") {
            if (cursor.parse_string() != "BPE") throw std::runtime_error("tokenizer model is not BPE");
        } else if (key == "ignore_merges") {
            ignore_merges = cursor.parse_bool();
        } else if (key == "vocab") {
            parse_vocabulary(cursor, vocabulary, id_to_piece);
        } else if (key == "merges") {
            parse_merges(cursor, merge_ranks);
        } else {
            cursor.skip_value();
        }
        if (cursor.consume('}')) return;
        cursor.expect(',');
    }
}

enum class RuneKind : std::uint8_t {
    Letter,
    Number,
    Mark,
    Space,
    Punctuation,
};

struct Rune {
    std::uint32_t codepoint{};
    std::size_t begin{};
    std::size_t end{};
    RuneKind kind{RuneKind::Punctuation};
};

[[nodiscard]] bool in_range(std::uint32_t value, std::uint32_t begin,
                            std::uint32_t end) noexcept {
    return value >= begin && value <= end;
}

[[nodiscard]] bool unicode_space(std::uint32_t value) noexcept {
    return value == 0x85U || value == 0xA0U || value == 0x1680U ||
           in_range(value, 0x2000U, 0x200AU) || value == 0x2028U ||
           value == 0x2029U || value == 0x202FU || value == 0x205FU ||
           value == 0x3000U;
}

[[nodiscard]] bool unicode_number(std::uint32_t value) noexcept {
    // The `unsigned char` narrowing has to be guarded: without it every
    // codepoint whose low byte lands in 0x30-0x39 reads as an ASCII digit, so
    // Cyrillic а-й (U+0430-U+0439), Armenian, and a long tail of others were
    // classified as numbers and pretokenized as digit runs. Found by the
    // Kimi-K3 tokenizer gate on `Привет мир`, which the reference splits into
    // two words and this split into six pieces; it affected every contract
    // sharing this classifier.
    if (value < 0x80U) return ascii_number(static_cast<unsigned char>(value));
    constexpr std::array<std::uint32_t, 21> starts{
        0x0660U, 0x06F0U, 0x07C0U, 0x0966U, 0x09E6U, 0x0A66U, 0x0AE6U,
        0x0B66U, 0x0BE6U, 0x0C66U, 0x0CE6U, 0x0D66U, 0x0E50U, 0x0ED0U,
        0x0F20U, 0x1040U, 0x1090U, 0x17E0U, 0x1810U, 0xFF10U, 0x1D7CEU};
    return std::any_of(starts.begin(), starts.end(), [value](std::uint32_t start) {
        return value >= start && value <= start + 9U;
    });
}

[[nodiscard]] bool unicode_mark(std::uint32_t value) noexcept {
    return in_range(value, 0x0300U, 0x036FU) ||
           in_range(value, 0x1AB0U, 0x1AFFU) ||
           in_range(value, 0x1DC0U, 0x1DFFU) ||
           in_range(value, 0x20D0U, 0x20FFU) ||
           in_range(value, 0xFE20U, 0xFE2FU);
}

[[nodiscard]] bool unicode_punctuation_or_symbol(std::uint32_t value) noexcept {
    return in_range(value, 0x2000U, 0x206FU) ||
           in_range(value, 0x20A0U, 0x20CFU) ||
           in_range(value, 0x2100U, 0x2BFFU) ||
           in_range(value, 0x2E00U, 0x2E7FU) ||
           in_range(value, 0x3001U, 0x303FU) ||
           in_range(value, 0xFE10U, 0xFE1FU) ||
           in_range(value, 0xFE30U, 0xFE6FU) ||
           in_range(value, 0xFF01U, 0xFF0FU) ||
           in_range(value, 0xFF1AU, 0xFF20U) ||
           in_range(value, 0xFF3BU, 0xFF40U) ||
           in_range(value, 0xFF5BU, 0xFF65U) ||
           in_range(value, 0x1F000U, 0x1FAFFU);
}

[[nodiscard]] RuneKind classify(std::uint32_t value) noexcept {
    if (value < 0x80U) {
        const auto ascii = static_cast<unsigned char>(value);
        if (ascii_letter(ascii)) return RuneKind::Letter;
        if (ascii_number(ascii)) return RuneKind::Number;
        if (ascii_space(ascii)) return RuneKind::Space;
        return RuneKind::Punctuation;
    }
    if (unicode_space(value)) return RuneKind::Space;
    if (unicode_number(value)) return RuneKind::Number;
    if (unicode_mark(value)) return RuneKind::Mark;
    if (unicode_punctuation_or_symbol(value)) return RuneKind::Punctuation;
    return RuneKind::Letter;
}

[[nodiscard]] ParseResult<std::vector<Rune>> decode_runes(std::string_view text) {
    ParseResult<std::vector<Rune>> result;
    std::size_t cursor = 0U;
    while (cursor < text.size()) {
        const auto decoded = decode_utf8(text, cursor);
        if (!decoded) {
            result.errors.emplace_back("tokenizer input is not valid UTF-8");
            result.value.clear();
            return result;
        }
        const auto [codepoint, bytes] = *decoded;
        result.value.push_back({codepoint, cursor, cursor + bytes,
                                classify(codepoint)});
        cursor += bytes;
    }
    return result;
}

[[nodiscard]] bool newline(const Rune& rune) noexcept {
    return rune.codepoint == '\r' || rune.codepoint == '\n';
}

[[nodiscard]] bool ascii_letter_rune(const Rune& rune) noexcept {
    return rune.codepoint < 0x80U &&
           ascii_letter(static_cast<unsigned char>(rune.codepoint));
}

[[nodiscard]] bool cjk(const Rune& rune) noexcept {
    return in_range(rune.codepoint, 0x4E00U, 0x9FA5U) ||
           in_range(rune.codepoint, 0x3040U, 0x309FU) ||
           in_range(rune.codepoint, 0x30A0U, 0x30FFU);
}

// `\p{Han}` proper, which is narrower than `cjk` above: kana are letters but not
// Han, and Kimi-K3's pattern gives Han its own alternative while kana fall
// through to the letter ones. Conflating them splits Japanese differently from
// the reference.
[[nodiscard]] bool han(const Rune& rune) noexcept {
    return in_range(rune.codepoint, 0x4E00U, 0x9FFFU) ||
           in_range(rune.codepoint, 0x3400U, 0x4DBFU) ||
           in_range(rune.codepoint, 0xF900U, 0xFAFFU) ||
           in_range(rune.codepoint, 0x2E80U, 0x2EFFU) ||
           in_range(rune.codepoint, 0x2F00U, 0x2FDFU) ||
           in_range(rune.codepoint, 0x3005U, 0x3005U) ||
           in_range(rune.codepoint, 0x3007U, 0x3007U) ||
           in_range(rune.codepoint, 0x20000U, 0x2FA1FU);
}

// Kimi-K3's two letter alternatives are built from `\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}`
// and `\p{Ll}\p{Lm}\p{Lo}\p{M}`, both with Han removed. `\p{Lm}`, `\p{Lo}`, and
// `\p{M}` are in both, so only the cased letters tell the two apart, and it is
// exactly that difference that splits `CamelCase` into `Camel` and `Case`.
[[nodiscard]] bool uppercase(std::uint32_t value) noexcept {
    if (value < 0x80U) return value >= 'A' && value <= 'Z';
    if (in_range(value, 0xC0U, 0xDEU) && value != 0xD7U) return true;
    if (in_range(value, 0x100U, 0x17FU)) return (value % 2U) == 0U;
    if (in_range(value, 0x391U, 0x3A9U)) return true;   // Greek capitals
    if (in_range(value, 0x400U, 0x42FU)) return true;   // Cyrillic capitals
    if (in_range(value, 0x1E00U, 0x1EFFU)) return (value % 2U) == 0U;
    if (in_range(value, 0x1F08U, 0x1F0FU)) return true;
    return false;
}

[[nodiscard]] bool lowercase(std::uint32_t value) noexcept {
    if (value < 0x80U) return value >= 'a' && value <= 'z';
    if (value == 0xDFU || in_range(value, 0xDFU, 0xFFU)) return value != 0xF7U;
    if (in_range(value, 0x100U, 0x17FU)) return (value % 2U) == 1U;
    if (in_range(value, 0x3B1U, 0x3C9U)) return true;
    if (in_range(value, 0x430U, 0x45FU)) return true;
    if (in_range(value, 0x1E00U, 0x1EFFU)) return (value % 2U) == 1U;
    if (in_range(value, 0x1F00U, 0x1F07U)) return true;
    return false;
}

// `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]`: an uncased letter or a mark
// belongs to both classes, which is why this is not the complement of `lower`.
[[nodiscard]] bool upper_class(const Rune& rune) noexcept {
    if (han(rune)) return false;
    if (rune.kind == RuneKind::Mark) return true;
    if (rune.kind != RuneKind::Letter) return false;
    return !lowercase(rune.codepoint);
}

// `[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]`
[[nodiscard]] bool lower_class(const Rune& rune) noexcept {
    if (han(rune)) return false;
    if (rune.kind == RuneKind::Mark) return true;
    if (rune.kind != RuneKind::Letter) return false;
    return !uppercase(rune.codepoint);
}

// `[^\r\n\p{L}\p{N}]?` — the optional single rune both letter alternatives may
// start with. A leading space enters a word this way, which is what makes
// " hello" one piece rather than two.
[[nodiscard]] bool word_prefix(const Rune& rune) noexcept {
    return !newline(rune) && rune.kind != RuneKind::Letter &&
           rune.kind != RuneKind::Number;
}

// `(?i:'s|'t|'re|'ve|'m|'ll|'d)?` after a letter run. ASCII apostrophe only:
// the pattern spells a literal `'`, so a typographic quote is not a contraction.
[[nodiscard]] std::size_t contraction_length(const std::vector<Rune>& runes,
                                             std::size_t cursor) noexcept {
    if (cursor >= runes.size() || runes[cursor].codepoint != '\'') return 0U;
    const auto lower = [&](std::size_t offset) -> std::uint32_t {
        if (cursor + offset >= runes.size()) return 0U;
        const auto value = runes[cursor + offset].codepoint;
        return (value >= 'A' && value <= 'Z') ? value + 32U : value;
    };
    const auto first = lower(1U);
    const auto second = lower(2U);
    if (first == 's' || first == 't' || first == 'm' || first == 'd') return 2U;
    if ((first == 'r' && second == 'e') || (first == 'v' && second == 'e') ||
        (first == 'l' && second == 'l')) {
        return 3U;
    }
    return 0U;
}

void emit_pretoken(std::string_view text, const std::vector<Rune>& runes,
                   std::size_t begin, std::size_t end,
                   std::vector<std::string>& output) {
    output.emplace_back(text.substr(runes[begin].begin,
                                    runes[end - 1U].end - runes[begin].begin));
}

[[nodiscard]] std::size_t whitespace_token_end(
    const std::vector<Rune>& runes, std::size_t begin) noexcept {
    auto run_end = begin;
    std::size_t last_newline = runes.size();
    while (run_end < runes.size() && runes[run_end].kind == RuneKind::Space) {
        if (newline(runes[run_end])) last_newline = run_end;
        ++run_end;
    }
    if (last_newline != runes.size()) return last_newline + 1U;
    if (run_end == runes.size()) return run_end;
    if (run_end - begin > 1U) return run_end - 1U;
    return run_end;
}

// Kimi-K3's `pat_str`, one alternative per block, in the order the reference
// lists them — `fancy-regex` is leftmost-first, so the order is semantics.
[[nodiscard]] ParseResult<std::vector<std::string>> pretokenize_kimi(
    std::string_view text, const std::vector<Rune>& runes) {
    ParseResult<std::vector<std::string>> result;
    const auto count = runes.size();

    // `[^\r\n\p{L}\p{N}]? [upper]* [lower]+ contraction?` and, when that fails,
    // `[^\r\n\p{L}\p{N}]? [upper]+ [lower]* contraction?`. The `?` and `*` are
    // greedy with backtracking, so the first form tries the longest run of
    // upper-class runes that still leaves a lower-class rune behind it.
    const auto letter_run = [&](std::size_t begin) -> std::size_t {
        for (const bool prefix : {true, false}) {
            auto cursor = begin;
            if (prefix) {
                if (cursor >= count || !word_prefix(runes[cursor])) continue;
                ++cursor;
            }
            auto upper_end = cursor;
            while (upper_end < count && upper_class(runes[upper_end])) ++upper_end;

            // First alternative: back off the upper run until a lower-class rune
            // starts the mandatory `[lower]+`.
            for (auto split = upper_end + 1U; split-- > cursor;) {
                if (split >= count || !lower_class(runes[split])) continue;
                auto end = split;
                while (end < count && lower_class(runes[end])) ++end;
                return end + contraction_length(runes, end);
            }
            // Second alternative: at least one upper-class rune, then any
            // number of lower-class ones.
            if (upper_end > cursor) {
                auto end = upper_end;
                while (end < count && lower_class(runes[end])) ++end;
                return end + contraction_length(runes, end);
            }
        }
        return begin;
    };

    std::size_t cursor = 0U;
    while (cursor < count) {
        const auto begin = cursor;
        if (han(runes[cursor])) {
            while (cursor < count && han(runes[cursor])) ++cursor;
            emit_pretoken(text, runes, begin, cursor, result.value);
            continue;
        }
        if (const auto end = letter_run(cursor); end > cursor) {
            cursor = end;
            emit_pretoken(text, runes, begin, cursor, result.value);
            continue;
        }
        if (runes[cursor].kind == RuneKind::Number) {
            while (cursor < count && cursor - begin < 3U &&
                   runes[cursor].kind == RuneKind::Number) ++cursor;
            emit_pretoken(text, runes, begin, cursor, result.value);
            continue;
        }
        // ` ?[^\s\p{L}\p{N}]+[\r\n]*` — a literal space, not `\s`.
        {
            auto scan = cursor;
            if (runes[scan].codepoint == ' ') ++scan;
            const auto body = scan;
            while (scan < count && runes[scan].kind != RuneKind::Space &&
                   runes[scan].kind != RuneKind::Letter &&
                   runes[scan].kind != RuneKind::Number) {
                ++scan;
            }
            if (scan > body) {
                while (scan < count && newline(runes[scan])) ++scan;
                cursor = scan;
                emit_pretoken(text, runes, begin, cursor, result.value);
                continue;
            }
        }
        if (runes[cursor].kind == RuneKind::Space) {
            cursor = whitespace_token_end(runes, cursor);
            emit_pretoken(text, runes, begin, cursor, result.value);
            continue;
        }
        ++cursor;
        emit_pretoken(text, runes, begin, cursor, result.value);
    }
    return result;
}

[[nodiscard]] ParseResult<std::vector<std::string>> pretokenize_glm(
    std::string_view text, const std::vector<Rune>& runes) {
    ParseResult<std::vector<std::string>> result;
    std::size_t cursor = 0U;
    while (cursor < runes.size()) {
        const auto begin = cursor;
        if (runes[cursor].codepoint == '\'' && cursor + 1U < runes.size() &&
            runes[cursor + 1U].codepoint < 0x80U) {
            std::string contraction;
            for (std::size_t index = cursor + 1U;
                 index < runes.size() && index < cursor + 3U; ++index) {
                contraction.push_back(static_cast<char>(ascii_lower(
                    static_cast<unsigned char>(runes[index].codepoint))));
            }
            const std::size_t length =
                contraction.starts_with("re") || contraction.starts_with("ve") ||
                        contraction.starts_with("ll")
                    ? 3U
                    : (!contraction.empty() &&
                       std::string_view("stmd").find(contraction[0]) !=
                           std::string_view::npos ? 2U : 0U);
            if (length != 0U && cursor + length <= runes.size()) {
                cursor += length;
                emit_pretoken(text, runes, begin, cursor, result.value);
                continue;
            }
        }

        std::size_t scan = cursor;
        if (runes[scan].kind != RuneKind::Letter && !newline(runes[scan]) &&
            runes[scan].kind != RuneKind::Number && scan + 1U < runes.size() &&
            runes[scan + 1U].kind == RuneKind::Letter) {
            ++scan;
        }
        if (scan < runes.size() && runes[scan].kind == RuneKind::Letter) {
            while (scan < runes.size() && runes[scan].kind == RuneKind::Letter) ++scan;
            cursor = scan;
            emit_pretoken(text, runes, begin, cursor, result.value);
            continue;
        }
        if (runes[cursor].kind == RuneKind::Number) {
            while (cursor < runes.size() && cursor - begin < 3U &&
                   runes[cursor].kind == RuneKind::Number) ++cursor;
            emit_pretoken(text, runes, begin, cursor, result.value);
            continue;
        }
        scan = cursor;
        if (runes[scan].codepoint == ' ' && scan + 1U < runes.size() &&
            runes[scan + 1U].kind == RuneKind::Punctuation) ++scan;
        if (scan < runes.size() && runes[scan].kind == RuneKind::Punctuation) {
            while (scan < runes.size() &&
                   runes[scan].kind == RuneKind::Punctuation) ++scan;
            while (scan < runes.size() && newline(runes[scan])) ++scan;
            cursor = scan;
            emit_pretoken(text, runes, begin, cursor, result.value);
            continue;
        }
        if (runes[cursor].kind == RuneKind::Space) {
            cursor = whitespace_token_end(runes, cursor);
            emit_pretoken(text, runes, begin, cursor, result.value);
            continue;
        }
        ++cursor;
        emit_pretoken(text, runes, begin, cursor, result.value);
    }
    return result;
}

[[nodiscard]] ParseResult<std::vector<std::string>> pretokenize_deepseek(
    std::string_view text, const std::vector<Rune>& runes) {
    ParseResult<std::vector<std::string>> result;
    std::size_t cursor = 0U;
    while (cursor < runes.size()) {
        const auto begin = cursor;
        if (runes[cursor].kind == RuneKind::Number) {
            while (cursor < runes.size() && cursor - begin < 3U &&
                   runes[cursor].kind == RuneKind::Number) ++cursor;
            emit_pretoken(text, runes, begin, cursor, result.value);
            continue;
        }
        if (cjk(runes[cursor])) {
            while (cursor < runes.size() && cjk(runes[cursor])) ++cursor;
            emit_pretoken(text, runes, begin, cursor, result.value);
            continue;
        }
        if (runes[cursor].kind == RuneKind::Punctuation &&
            runes[cursor].codepoint < 0x80U && cursor + 1U < runes.size() &&
            ascii_letter_rune(runes[cursor + 1U])) {
            ++cursor;
            while (cursor < runes.size() && ascii_letter_rune(runes[cursor])) ++cursor;
            emit_pretoken(text, runes, begin, cursor, result.value);
            continue;
        }
        std::size_t scan = cursor;
        if (!newline(runes[scan]) && runes[scan].kind != RuneKind::Letter &&
            runes[scan].kind != RuneKind::Punctuation &&
            scan + 1U < runes.size() &&
            (runes[scan + 1U].kind == RuneKind::Letter ||
             runes[scan + 1U].kind == RuneKind::Mark)) {
            ++scan;
        }
        if (scan < runes.size() &&
            (runes[scan].kind == RuneKind::Letter ||
             runes[scan].kind == RuneKind::Mark)) {
            while (scan < runes.size() &&
                   (runes[scan].kind == RuneKind::Letter ||
                    runes[scan].kind == RuneKind::Mark) && !cjk(runes[scan])) ++scan;
            cursor = scan;
            emit_pretoken(text, runes, begin, cursor, result.value);
            continue;
        }
        scan = cursor;
        if (runes[scan].codepoint == ' ' && scan + 1U < runes.size() &&
            runes[scan + 1U].kind == RuneKind::Punctuation) ++scan;
        if (scan < runes.size() && runes[scan].kind == RuneKind::Punctuation) {
            while (scan < runes.size() &&
                   runes[scan].kind == RuneKind::Punctuation) ++scan;
            while (scan < runes.size() && newline(runes[scan])) ++scan;
            cursor = scan;
            emit_pretoken(text, runes, begin, cursor, result.value);
            continue;
        }
        if (runes[cursor].kind == RuneKind::Space) {
            cursor = whitespace_token_end(runes, cursor);
            emit_pretoken(text, runes, begin, cursor, result.value);
            continue;
        }
        ++cursor;
        emit_pretoken(text, runes, begin, cursor, result.value);
    }
    return result;
}

// The Laguna alternatives test \p{L}, \p{N} and \s directly, so they use the
// exact Unicode category tables rather than the heuristic RuneKind classifier
// shared with GLM and DeepSeek. That classifier calls every unknown non-ASCII
// codepoint a letter, which puts U+00BB (and every other Latin-1 punctuation
// mark) on the wrong side of [^\s\p{L}\p{N}]+ and splits "»/c" as {"»", "/c"}
// where the reference produces {"»/", "c"}.
[[nodiscard]] bool laguna_letter(const Rune& rune) noexcept {
    return detail::laguna_letter(rune.codepoint);
}

[[nodiscard]] bool laguna_number(const Rune& rune) noexcept {
    return detail::laguna_number(rune.codepoint);
}

[[nodiscard]] bool laguna_space(const Rune& rune) noexcept {
    return detail::laguna_space(rune.codepoint);
}

// The complement class [^\s\p{L}\p{N}].
[[nodiscard]] bool laguna_symbol(const Rune& rune) noexcept {
    return !laguna_letter(rune) && !laguna_number(rune) && !laguna_space(rune);
}

// `\s+(?!\S)` yields the run minus its last character when a non-space follows,
// and the whole run at end of input; `\s*[\r\n]+` ends just past the last
// newline in the run. Together with the bare `\s+` fallback that reproduces the
// last three alternatives.
[[nodiscard]] std::size_t laguna_whitespace_end(
    const std::vector<Rune>& runes, std::size_t begin) noexcept {
    auto run_end = begin;
    std::size_t last_newline = runes.size();
    while (run_end < runes.size() && laguna_space(runes[run_end])) {
        if (newline(runes[run_end])) last_newline = run_end;
        ++run_end;
    }
    if (last_newline != runes.size()) return last_newline + 1U;
    if (run_end == runes.size()) return run_end;
    if (run_end - begin > 1U) return run_end - 1U;
    return run_end;
}

// Second stage of the Laguna pre-tokenizer:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   |[^\r\n\p{L}\p{N}]?\p{L}+ |\p{N}
//   | ?[^\s\p{L}\p{N}]+[\r\n]* |\s*[\r\n]+ |\s+(?!\S) |\s+
// applied with Isolated behavior, so the alternatives are tried in order at
// each position. Unlike the GLM pattern, digits are isolated one at a time.
void pretokenize_laguna_chunk(std::string_view text,
                              const std::vector<Rune>& runes,
                              std::vector<std::string>& output) {
    std::size_t cursor = 0U;
    while (cursor < runes.size()) {
        const auto begin = cursor;
        if (runes[cursor].codepoint == '\'' && cursor + 1U < runes.size() &&
            runes[cursor + 1U].codepoint < 0x80U) {
            std::string contraction;
            for (std::size_t index = cursor + 1U;
                 index < runes.size() && index < cursor + 3U; ++index) {
                contraction.push_back(static_cast<char>(ascii_lower(
                    static_cast<unsigned char>(runes[index].codepoint))));
            }
            const std::size_t length =
                contraction.starts_with("re") || contraction.starts_with("ve") ||
                        contraction.starts_with("ll")
                    ? 3U
                    : (!contraction.empty() &&
                       std::string_view("stmd").find(contraction[0]) !=
                           std::string_view::npos ? 2U : 0U);
            if (length != 0U && cursor + length <= runes.size()) {
                cursor += length;
                emit_pretoken(text, runes, begin, cursor, output);
                continue;
            }
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+ : the optional prefix is any single
        // non-letter, non-digit that is not a line break, including a space.
        std::size_t scan = cursor;
        if (!laguna_letter(runes[scan]) && !newline(runes[scan]) &&
            !laguna_number(runes[scan]) && scan + 1U < runes.size() &&
            laguna_letter(runes[scan + 1U])) {
            ++scan;
        }
        if (scan < runes.size() && laguna_letter(runes[scan])) {
            while (scan < runes.size() && laguna_letter(runes[scan])) ++scan;
            cursor = scan;
            emit_pretoken(text, runes, begin, cursor, output);
            continue;
        }
        if (laguna_number(runes[cursor])) {
            ++cursor;
            emit_pretoken(text, runes, begin, cursor, output);
            continue;
        }
        //  ?[^\s\p{L}\p{N}]+[\r\n]*
        scan = cursor;
        if (runes[scan].codepoint == ' ' && scan + 1U < runes.size() &&
            laguna_symbol(runes[scan + 1U])) {
            ++scan;
        }
        if (scan < runes.size() && laguna_symbol(runes[scan])) {
            while (scan < runes.size() && laguna_symbol(runes[scan])) ++scan;
            while (scan < runes.size() && newline(runes[scan])) ++scan;
            cursor = scan;
            emit_pretoken(text, runes, begin, cursor, output);
            continue;
        }
        if (laguna_space(runes[cursor])) {
            cursor = laguna_whitespace_end(runes, cursor);
            emit_pretoken(text, runes, begin, cursor, output);
            continue;
        }
        ++cursor;
        emit_pretoken(text, runes, begin, cursor, output);
    }
}

// First stage: Split on (?:\r?\n)+(?!\r?\n) with MergedWithNext, so a maximal
// newline run starts a new chunk instead of merging with the whitespace that
// precedes it. Without it, " \n x" would pre-tokenize as [" \n", " x"] rather
// than [" ", "\n", " x"].
[[nodiscard]] std::vector<std::size_t> laguna_newline_boundaries(
    const std::vector<Rune>& runes) {
    std::vector<std::size_t> boundaries;
    std::size_t index = 0U;
    while (index < runes.size()) {
        std::size_t scan = index;
        for (;;) {
            if (runes[scan].codepoint == '\r' && scan + 1U < runes.size() &&
                runes[scan + 1U].codepoint == '\n') {
                scan += 2U;
            } else if (runes[scan].codepoint == '\n') {
                scan += 1U;
            } else {
                break;
            }
            if (scan >= runes.size()) break;
        }
        if (scan == index) {
            ++index;
            continue;
        }
        if (index != 0U) boundaries.push_back(index);
        index = scan;
    }
    return boundaries;
}

[[nodiscard]] ParseResult<std::vector<std::string>> pretokenize_laguna(
    std::string_view text, const std::vector<Rune>& runes) {
    ParseResult<std::vector<std::string>> result;
    const auto boundaries = laguna_newline_boundaries(runes);
    std::size_t begin = 0U;
    for (std::size_t index = 0U; index <= boundaries.size(); ++index) {
        const auto end = index < boundaries.size() ? boundaries[index]
                                                   : runes.size();
        if (end > begin) {
            const std::vector<Rune> chunk(runes.begin() +
                                              static_cast<std::ptrdiff_t>(begin),
                                          runes.begin() +
                                              static_cast<std::ptrdiff_t>(end));
            pretokenize_laguna_chunk(text, chunk, result.value);
        }
        begin = end;
    }
    return result;
}

// Length in runes of the contraction suffix at `cursor`, or zero. Inkling
// attaches it to the letter run rather than matching it as its own
// alternative, so it is a suffix test, not a leading one.
[[nodiscard]] std::size_t inkling_contraction(const std::vector<Rune>& runes,
                                              std::size_t cursor) noexcept {
    if (cursor >= runes.size() || runes[cursor].codepoint != '\'') return 0U;
    std::string tail;
    for (std::size_t index = cursor + 1U;
         index < runes.size() && index < cursor + 3U; ++index) {
        if (runes[index].codepoint >= 0x80U) break;
        tail.push_back(static_cast<char>(
            ascii_lower(static_cast<unsigned char>(runes[index].codepoint))));
    }
    if (tail.starts_with("re") || tail.starts_with("ve") ||
        tail.starts_with("ll")) {
        return 3U;
    }
    if (!tail.empty() &&
        std::string_view("stmd").find(tail[0]) != std::string_view::npos) {
        return 2U;
    }
    return 0U;
}

// Inkling's pre-tokenizer, applied with Isolated behavior so the alternatives
// are tried in order at each position:
//   [^\r\n\p{L}\p{N}]?[Upper]*[Lower]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//   |[^\r\n\p{L}\p{N}]?[Upper]+[Lower]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//   |\p{N}{1,3}
//   | ?[^\s\p{L}\p{N}]+[\r\n/]*
//   |\s*[\r\n]+ |\s+(?!\S) |\s+
// where Upper is Lu|Lt|Lm|Lo|M and Lower is Ll|Lm|Lo|M. Two properties
// distinguish it from the Laguna pattern: digits group in runs of up to three
// rather than one at a time, and the punctuation tail absorbs '/' alongside
// line breaks.
[[nodiscard]] ParseResult<std::vector<std::string>> pretokenize_inkling(
    std::string_view text, const std::vector<Rune>& runes) {
    ParseResult<std::vector<std::string>> result;
    const auto upper = [&](std::size_t index) {
        return index < runes.size() &&
               detail::inkling_upper_letter(runes[index].codepoint);
    };
    const auto lower = [&](std::size_t index) {
        return index < runes.size() &&
               detail::inkling_lower_letter(runes[index].codepoint);
    };
    const auto letter = [&](std::size_t index) {
        return index < runes.size() &&
               detail::laguna_letter(runes[index].codepoint);
    };
    const auto number = [&](std::size_t index) {
        return index < runes.size() &&
               detail::laguna_number(runes[index].codepoint);
    };
    const auto space = [&](std::size_t index) {
        return index < runes.size() &&
               detail::laguna_space(runes[index].codepoint);
    };

    std::size_t cursor = 0U;
    while (cursor < runes.size()) {
        const auto begin = cursor;

        // Both letter alternatives share the optional single-character prefix,
        // which is any non-letter, non-digit that is not a line break.
        std::size_t scan = cursor;
        const bool prefixed = !letter(scan) && !number(scan) &&
                              !newline(runes[scan]) && (upper(scan + 1U) ||
                                                        lower(scan + 1U));
        if (prefixed) ++scan;
        const auto run_start = scan;
        while (upper(scan) && !lower(scan)) ++scan;
        const auto after_upper = scan;
        while (lower(scan)) ++scan;
        // Caseless letters satisfy both classes, so a run that consumed
        // nothing under the strict-upper test still advances here.
        if (scan == run_start) {
            while (upper(scan) || lower(scan)) ++scan;
        }
        if (scan > run_start && (after_upper > run_start || scan > run_start)) {
            scan += inkling_contraction(runes, scan);
            cursor = scan;
            emit_pretoken(text, runes, begin, cursor, result.value);
            continue;
        }

        if (number(cursor)) {
            while (cursor < runes.size() && cursor - begin < 3U &&
                   number(cursor)) {
                ++cursor;
            }
            emit_pretoken(text, runes, begin, cursor, result.value);
            continue;
        }

        //  ?[^\s\p{L}\p{N}]+[\r\n/]*
        scan = cursor;
        const auto symbol = [&](std::size_t index) {
            return index < runes.size() && !space(index) && !letter(index) &&
                   !number(index);
        };
        if (runes[scan].codepoint == ' ' && symbol(scan + 1U)) ++scan;
        if (symbol(scan)) {
            while (symbol(scan)) ++scan;
            while (scan < runes.size() &&
                   (newline(runes[scan]) || runes[scan].codepoint == '/')) {
                ++scan;
            }
            cursor = scan;
            emit_pretoken(text, runes, begin, cursor, result.value);
            continue;
        }

        if (space(cursor)) {
            cursor = laguna_whitespace_end(runes, cursor);
            if (cursor == begin) ++cursor;
            emit_pretoken(text, runes, begin, cursor, result.value);
            continue;
        }

        ++cursor;
        emit_pretoken(text, runes, begin, cursor, result.value);
    }
    return result;
}

}  // namespace

ParseResult<std::vector<std::string>> pretokenize(
    TokenizerContract contract, std::string_view text) {
    const auto runes = decode_runes(text);
    if (!runes.ok()) {
        ParseResult<std::vector<std::string>> result;
        result.errors = runes.errors;
        return result;
    }
    if (contract == TokenizerContract::Glm52) {
        return pretokenize_glm(text, runes.value);
    }
    if (contract == TokenizerContract::DeepSeekV4) {
        return pretokenize_deepseek(text, runes.value);
    }
    if (contract == TokenizerContract::Laguna) {
        return pretokenize_laguna(text, runes.value);
    }
    if (contract == TokenizerContract::Inkling) {
        return pretokenize_inkling(text, runes.value);
    }
    if (contract == TokenizerContract::KimiK3) {
        return pretokenize_kimi(text, runes.value);
    }
    ParseResult<std::vector<std::string>> result;
    result.value.emplace_back(text);
    return result;
}

ModelTokenizer::ModelTokenizer() {
    codepoint_to_byte_.fill(-1);
    std::array<bool, 256> direct{};
    for (std::uint32_t value = 33U; value <= 126U; ++value) direct[value] = true;
    for (std::uint32_t value = 161U; value <= 172U; ++value) direct[value] = true;
    for (std::uint32_t value = 174U; value <= 255U; ++value) direct[value] = true;
    std::uint32_t substitute = 0;
    for (std::uint32_t value = 0; value < 256U; ++value) {
        const auto codepoint = direct[value] ? value : 256U + substitute++;
        byte_to_piece_[value] = utf8(codepoint);
        codepoint_to_byte_[codepoint] = static_cast<std::int16_t>(value);
    }
}

namespace {

[[nodiscard]] bool decode_base64(std::string_view encoded, std::string& out) {
    static constexpr std::string_view kAlphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::uint32_t accumulator = 0U;
    std::uint32_t bits = 0U;
    out.clear();
    for (const char symbol : encoded) {
        if (symbol == '=') break;
        const auto position = kAlphabet.find(symbol);
        if (position == std::string_view::npos) return false;
        accumulator = (accumulator << 6U) | static_cast<std::uint32_t>(position);
        bits += 6U;
        if (bits >= 8U) {
            bits -= 8U;
            out.push_back(static_cast<char>((accumulator >> bits) & 0xFFU));
        }
    }
    return true;
}

// The named reserved ids come from `tokenizer_config.json`; the rest of the 256
// reserved slots are `<|reserved_token_N|>`, exactly as the reference builds
// them. Parsed with the same cursor as everything else rather than by pattern
// matching, so a change in that file is an error and not a silent misread.
void parse_added_tokens_decoder(
    JsonCursor& cursor, std::vector<std::pair<std::string, std::uint32_t>>& out) {
    cursor.expect('{');
    if (cursor.consume('}')) return;
    for (;;) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        const auto id = static_cast<std::uint32_t>(std::stoul(key));
        std::string content;
        cursor.expect('{');
        if (!cursor.consume('}')) {
            for (;;) {
                const auto field = cursor.parse_string();
                cursor.expect(':');
                if (field == "content") {
                    content = cursor.parse_string();
                } else {
                    cursor.skip_value();
                }
                if (cursor.consume('}')) break;
                cursor.expect(',');
            }
        }
        if (!content.empty()) out.emplace_back(std::move(content), id);
        if (cursor.consume('}')) break;
        cursor.expect(',');
    }
}

}  // namespace

ParseResult<ModelTokenizer> ModelTokenizer::load_kimi_k3(
    const std::string& model_directory) {
    ParseResult<ModelTokenizer> result;
    const auto vocabulary = load_bounded_text_file(
        model_directory + "/tiktoken.model", 64ULL << 20U);
    if (!vocabulary.ok()) {
        result.errors = vocabulary.errors;
        return result;
    }
    const auto configuration = load_bounded_text_file(
        model_directory + "/tokenizer_config.json", 8ULL << 20U);
    if (!configuration.ok()) {
        result.errors = configuration.errors;
        return result;
    }
    try {
        // One `base64(piece) rank` per line, ranks dense from zero. The rank is
        // the token id, so the file is both the vocabulary and the merge order:
        // tiktoken has no separate merge table.
        result.value.vocabulary_.reserve(170000U);
        std::string_view remaining(vocabulary.value);
        std::string piece;
        std::uint32_t expected = 0U;
        while (!remaining.empty()) {
            const auto line_end = remaining.find('\n');
            const auto line = remaining.substr(0U, line_end);
            remaining = line_end == std::string_view::npos
                            ? std::string_view{}
                            : remaining.substr(line_end + 1U);
            if (line.empty()) continue;
            const auto space = line.find(' ');
            if (space == std::string_view::npos) {
                throw std::runtime_error("tiktoken line has no rank");
            }
            if (!decode_base64(line.substr(0U, space), piece)) {
                throw std::runtime_error("tiktoken piece is not base64");
            }
            const auto rank = static_cast<std::uint32_t>(
                std::stoul(std::string(line.substr(space + 1U))));
            if (rank != expected) {
                throw std::runtime_error("tiktoken ranks are not dense");
            }
            ++expected;
            result.value.vocabulary_.emplace(piece, rank);
            result.value.id_to_piece_.push_back(piece);
        }

        std::vector<std::pair<std::string, std::uint32_t>> named;
        JsonCursor cursor(configuration.value);
        cursor.expect('{');
        if (!cursor.consume('}')) {
            for (;;) {
                const auto key = cursor.parse_string();
                cursor.expect(':');
                if (key == "added_tokens_decoder") {
                    parse_added_tokens_decoder(cursor, named);
                } else {
                    cursor.skip_value();
                }
                if (cursor.consume('}')) break;
                cursor.expect(',');
            }
        }

        const auto base = static_cast<std::uint32_t>(result.value.id_to_piece_.size());
        constexpr std::uint32_t kReservedSpecialTokens = 256U;
        if (base != 163584U || named.size() != 16U) {
            throw std::runtime_error(
                "tokenizer does not match the pinned Kimi-K3 contract");
        }
        std::unordered_map<std::uint32_t, std::string> by_id;
        for (auto& [content, id] : named) by_id.emplace(id, std::move(content));
        result.value.id_to_piece_.resize(base + kReservedSpecialTokens);
        result.value.added_tokens_.reserve(kReservedSpecialTokens);
        for (std::uint32_t offset = 0U; offset < kReservedSpecialTokens; ++offset) {
            const auto id = base + offset;
            const auto found = by_id.find(id);
            auto content = found != by_id.end()
                               ? found->second
                               : "<|reserved_token_" + std::to_string(id) + "|>";
            result.value.id_to_piece_[id] = content;
            result.value.added_tokens_.push_back({std::move(content), id});
        }
        result.value.added_id_.assign(result.value.id_to_piece_.size(), false);
        for (const auto& token : result.value.added_tokens_) {
            result.value.added_id_[token.id] = true;
        }
        std::sort(result.value.added_tokens_.begin(),
                  result.value.added_tokens_.end(),
                  [](const auto& left, const auto& right) {
                      if (left.content.size() != right.content.size()) {
                          return left.content.size() > right.content.size();
                      }
                      return left.content < right.content;
                  });
        result.value.contract_ = TokenizerContract::KimiK3;
        result.value.tiktoken_bpe_ = true;
    } catch (const std::exception& error) {
        result.errors.emplace_back(std::string("Kimi-K3 tokenizer: ") + error.what());
        result.value = ModelTokenizer{};
    }
    return result;
}

ParseResult<ModelTokenizer> ModelTokenizer::load(const std::string& path) {
    ParseResult<ModelTokenizer> result;
    const auto text = load_bounded_text_file(path, 128ULL << 20U);
    if (!text.ok()) {
        result.errors = text.errors;
        return result;
    }
    try {
        result.value.vocabulary_.reserve(160000U);
        result.value.merge_ranks_.reserve(330000U);
        JsonCursor cursor(text.value);
        std::vector<std::pair<std::string, std::uint32_t>> added_tokens;
        bool saw_added = false;
        bool saw_model = false;
        cursor.expect('{');
        if (!cursor.consume('}')) {
            for (;;) {
                const auto key = cursor.parse_string();
                cursor.expect(':');
                if (key == "added_tokens") {
                    parse_added_tokens(cursor, added_tokens);
                    saw_added = true;
                } else if (key == "model") {
                    parse_model(cursor, result.value.ignore_merges_,
                                result.value.vocabulary_, result.value.id_to_piece_,
                                result.value.merge_ranks_);
                    saw_model = true;
                } else {
                    cursor.skip_value();
                }
                if (cursor.consume('}')) break;
                cursor.expect(',');
            }
        }
        if (!cursor.finished()) throw detail::JsonError(cursor.offset(), "trailing content");
        const bool glm52 = saw_added && saw_model &&
            result.value.vocabulary_.size() == 154820U &&
            result.value.merge_ranks_.size() == 321649U &&
            added_tokens.size() == 36U && result.value.ignore_merges_;
        const bool deepseek_v4 = saw_added && saw_model &&
            result.value.vocabulary_.size() == 128000U &&
            result.value.merge_ranks_.size() == 127741U &&
            added_tokens.size() == 1283U && !result.value.ignore_merges_;
        const bool gemma4 = saw_added && saw_model &&
            result.value.vocabulary_.size() == 262144U &&
            result.value.merge_ranks_.size() == 514906U &&
            added_tokens.size() == 24U && !result.value.ignore_merges_;
        const bool laguna = saw_added && saw_model &&
            result.value.vocabulary_.size() == 100352U &&
            result.value.merge_ranks_.size() == 100026U &&
            added_tokens.size() == 70U && !result.value.ignore_merges_;
        const bool inkling = saw_added && saw_model &&
            result.value.vocabulary_.size() == 199998U &&
            result.value.merge_ranks_.size() == 446189U &&
            added_tokens.size() == 60U && result.value.ignore_merges_;
        if (!glm52 && !deepseek_v4 && !gemma4 && !laguna && !inkling) {
            throw std::runtime_error(
                "tokenizer does not match a pinned GLM-5.2, DeepSeek-V4, Gemma-4, "
                "Laguna, or Inkling contract");
        }
        result.value.contract_ = gemma4 ? TokenizerContract::Gemma4
            : deepseek_v4 ? TokenizerContract::DeepSeekV4
            : laguna      ? TokenizerContract::Laguna
            : inkling     ? TokenizerContract::Inkling
                          : TokenizerContract::Glm52;
        result.value.sentencepiece_bpe_ = gemma4;
        result.value.added_tokens_.reserve(added_tokens.size());
        for (auto& [content, id] : added_tokens) {
            result.value.added_tokens_.push_back({std::move(content), id});
        }
        std::uint32_t maximum_id = 0;
        for (const auto& token : result.value.added_tokens_) maximum_id = std::max(maximum_id, token.id);
        if (result.value.id_to_piece_.size() <= maximum_id) {
            result.value.id_to_piece_.resize(static_cast<std::size_t>(maximum_id) + 1U);
        }
        result.value.added_id_.resize(result.value.id_to_piece_.size(), false);
        for (const auto& token : result.value.added_tokens_) {
            if (!result.value.id_to_piece_[token.id].empty() &&
                result.value.id_to_piece_[token.id] != token.content) {
                throw std::runtime_error(
                    "added token collides with a different base vocabulary piece");
            }
            result.value.id_to_piece_[token.id] = token.content;
            result.value.added_id_[token.id] = true;
        }
        std::sort(result.value.added_tokens_.begin(), result.value.added_tokens_.end(),
                  [](const auto& left, const auto& right) {
                      if (left.content.size() != right.content.size()) {
                          return left.content.size() > right.content.size();
                      }
                      return left.content < right.content;
                  });
    } catch (const std::exception& exception) {
        result.errors.emplace_back(exception.what());
    }
    return result;
}

std::int32_t ModelTokenizer::token_id(std::string_view piece) const noexcept {
    for (const auto& token : added_tokens_) {
        if (token.content == piece) return static_cast<std::int32_t>(token.id);
    }
    const auto found = vocabulary_.find(std::string(piece));
    return found == vocabulary_.end() ? -1 : static_cast<std::int32_t>(found->second);
}

ParseResult<std::vector<std::uint32_t>> ModelTokenizer::encode_piece(
    std::string_view bytes) const {
    ParseResult<std::vector<std::uint32_t>> result;
    if (tiktoken_bpe_) {
        // tiktoken has no merge table: the rank of a merge *is* the id of the
        // piece it produces, so the loop repeatedly joins the adjacent pair
        // whose concatenation has the lowest id in the vocabulary. Raw bytes,
        // not the GPT-2 byte alphabet — the vocabulary is byte strings.
        if (bytes.empty()) return result;
        if (const auto whole = vocabulary_.find(std::string(bytes));
            whole != vocabulary_.end()) {
            result.value.push_back(whole->second);
            return result;
        }
        std::vector<std::size_t> boundaries(bytes.size() + 1U);
        for (std::size_t index = 0U; index <= bytes.size(); ++index) {
            boundaries[index] = index;
        }
        const auto rank_of = [&](std::size_t part) -> std::uint32_t {
            if (part + 2U >= boundaries.size()) {
                return std::numeric_limits<std::uint32_t>::max();
            }
            const auto begin = boundaries[part];
            const auto end = boundaries[part + 2U];
            const auto found = vocabulary_.find(
                std::string(bytes.substr(begin, end - begin)));
            return found == vocabulary_.end()
                       ? std::numeric_limits<std::uint32_t>::max()
                       : found->second;
        };
        for (;;) {
            auto best = std::numeric_limits<std::uint32_t>::max();
            std::size_t chosen = boundaries.size();
            for (std::size_t part = 0U; part + 2U < boundaries.size(); ++part) {
                if (const auto rank = rank_of(part); rank < best) {
                    best = rank;
                    chosen = part;
                }
            }
            if (chosen == boundaries.size()) break;
            boundaries.erase(boundaries.begin() +
                             static_cast<std::ptrdiff_t>(chosen) + 1);
        }
        for (std::size_t part = 0U; part + 1U < boundaries.size(); ++part) {
            const auto begin = boundaries[part];
            const auto end = boundaries[part + 1U];
            const auto found = vocabulary_.find(
                std::string(bytes.substr(begin, end - begin)));
            if (found == vocabulary_.end()) {
                // Every single byte is in a tiktoken vocabulary, so this cannot
                // happen for well-formed input; reporting beats emitting an
                // unknown-token id the model never saw.
                result.errors.emplace_back(
                    "byte sequence is outside the pinned Kimi-K3 vocabulary");
                result.value.clear();
                return result;
            }
            result.value.push_back(found->second);
        }
        return result;
    }
    if (sentencepiece_bpe_) {
        std::string normalized;
        normalized.reserve(bytes.size());
        for (const char value : bytes) {
            if (value == ' ') normalized += "▁";
            else normalized.push_back(value);
        }
        struct Node {
            std::string symbol;
            std::size_t previous{};
            std::size_t next{};
            std::uint32_t generation{};
            bool alive{true};
        };
        struct Candidate {
            std::uint32_t rank{};
            std::size_t left{};
            std::size_t right{};
            std::uint32_t left_generation{};
            std::uint32_t right_generation{};
        };
        const auto later = [](const Candidate& left, const Candidate& right) {
            return left.rank > right.rank ||
                   (left.rank == right.rank && left.left > right.left);
        };
        std::vector<Node> nodes;
        for (std::size_t cursor = 0U; cursor < normalized.size();) {
            const auto decoded = decode_utf8(normalized, cursor);
            if (!decoded) {
                result.errors.emplace_back("tokenizer input is not valid UTF-8");
                return result;
            }
            const auto index = nodes.size();
            nodes.push_back({normalized.substr(cursor, decoded->second),
                             index == 0U ? index : index - 1U, index + 1U});
            cursor += decoded->second;
        }
        if (nodes.empty()) return result;
        nodes.back().next = nodes.size();
        std::priority_queue<Candidate, std::vector<Candidate>, decltype(later)>
            candidates(later);
        const auto enqueue = [&](std::size_t left) {
            if (left >= nodes.size() || !nodes[left].alive) return;
            const auto right = nodes[left].next;
            if (right >= nodes.size() || !nodes[right].alive) return;
            const auto found = merge_ranks_.find(
                merge_key(nodes[left].symbol, nodes[right].symbol));
            if (found != merge_ranks_.end()) {
                candidates.push({found->second, left, right,
                                 nodes[left].generation,
                                 nodes[right].generation});
            }
        };
        for (std::size_t index = 0U; index + 1U < nodes.size(); ++index) {
            enqueue(index);
        }
        while (!candidates.empty()) {
            const auto candidate = candidates.top();
            candidates.pop();
            if (!nodes[candidate.left].alive || !nodes[candidate.right].alive ||
                nodes[candidate.left].next != candidate.right ||
                nodes[candidate.left].generation != candidate.left_generation ||
                nodes[candidate.right].generation != candidate.right_generation) {
                continue;
            }
            auto& left = nodes[candidate.left];
            auto& right = nodes[candidate.right];
            left.symbol += right.symbol;
            ++left.generation;
            right.alive = false;
            ++right.generation;
            left.next = right.next;
            if (right.next < nodes.size()) {
                nodes[right.next].previous = candidate.left;
            }
            if (left.previous != candidate.left) enqueue(left.previous);
            enqueue(candidate.left);
        }
        for (std::size_t index = 0U; index < nodes.size(); index = nodes[index].next) {
            const auto found = vocabulary_.find(nodes[index].symbol);
            if (found != vocabulary_.end()) {
                result.value.push_back(found->second);
                continue;
            }
            for (const unsigned char value : nodes[index].symbol) {
                constexpr char digits[] = "0123456789ABCDEF";
                std::string fallback = "<0x00>";
                fallback[3] = digits[value >> 4U];
                fallback[4] = digits[value & 0x0FU];
                const auto byte = vocabulary_.find(fallback);
                if (byte == vocabulary_.end()) {
                    result.errors.emplace_back("Gemma byte fallback is absent from the vocabulary");
                    result.value.clear();
                    return result;
                }
                result.value.push_back(byte->second);
            }
        }
        return result;
    }
    std::string byte_level;
    for (const auto value : bytes) byte_level += byte_to_piece_[static_cast<unsigned char>(value)];
    if (ignore_merges_) {
        const auto whole = vocabulary_.find(byte_level);
        if (whole != vocabulary_.end()) {
            result.value.push_back(whole->second);
            return result;
        }
    }
    std::vector<std::string> symbols;
    symbols.reserve(bytes.size());
    for (const auto value : bytes) symbols.push_back(byte_to_piece_[static_cast<unsigned char>(value)]);
    while (symbols.size() > 1U) {
        auto best_rank = std::numeric_limits<std::uint32_t>::max();
        std::size_t best_position = symbols.size();
        for (std::size_t position = 0; position + 1U < symbols.size(); ++position) {
            const auto found = merge_ranks_.find(merge_key(symbols[position], symbols[position + 1U]));
            if (found != merge_ranks_.end() && found->second < best_rank) {
                best_rank = found->second;
                best_position = position;
            }
        }
        if (best_position == symbols.size()) break;
        symbols[best_position] += symbols[best_position + 1U];
        symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_position + 1U));
    }
    for (const auto& symbol : symbols) {
        const auto found = vocabulary_.find(symbol);
        if (found == vocabulary_.end()) {
            result.errors.emplace_back("BPE produced a piece outside the pinned vocabulary");
            result.value.clear();
            return result;
        }
        result.value.push_back(found->second);
    }
    return result;
}

ParseResult<std::vector<std::uint32_t>> ModelTokenizer::encode_plain_chunk(
    std::string_view text) const {
    ParseResult<std::vector<std::uint32_t>> result;
    const auto pieces = pretokenize(contract_, text);
    if (!pieces.ok()) {
        result.errors = pieces.errors;
        return result;
    }
    for (const auto& piece : pieces.value) {
        const auto encoded = encode_piece(piece);
        if (!encoded.ok()) {
            result.errors = encoded.errors;
            result.value.clear();
            return result;
        }
        result.value.insert(result.value.end(), encoded.value.begin(),
                            encoded.value.end());
    }
    return result;
}

ParseResult<std::vector<std::uint32_t>> ModelTokenizer::encode(std::string_view text) const {
    ParseResult<std::vector<std::uint32_t>> result;
    std::size_t cursor = 0;
    while (cursor < text.size()) {
        std::size_t hit = text.size();
        const AddedToken* selected = nullptr;
        for (const auto& token : added_tokens_) {
            const auto position = text.find(token.content, cursor);
            if (position < hit || (position == hit && selected != nullptr &&
                                   token.content.size() > selected->content.size())) {
                hit = position;
                selected = &token;
            }
        }
        if (hit > cursor) {
            const auto plain = encode_plain_chunk(text.substr(cursor, hit - cursor));
            if (!plain.ok()) {
                result.errors = plain.errors;
                return result;
            }
            result.value.insert(result.value.end(), plain.value.begin(), plain.value.end());
        }
        if (selected == nullptr || hit == text.size()) break;
        result.value.push_back(selected->id);
        cursor = hit + selected->content.size();
    }
    return result;
}

ParseResult<std::string> ModelTokenizer::decode_token(std::uint32_t token) const {
    ParseResult<std::string> result;
    if (token >= id_to_piece_.size() || id_to_piece_[token].empty()) {
        result.errors.emplace_back("token id is outside the pinned vocabulary");
        return result;
    }
    const auto& piece = id_to_piece_[token];
    if (added_id_[token]) {
        result.value = piece;
        return result;
    }
    if (tiktoken_bpe_) {
        // The vocabulary is raw byte strings, so a piece decodes to itself. A
        // multi-byte character split across two tokens therefore comes back
        // whole only after concatenation, which `decode` does.
        result.value = piece;
        return result;
    }
    if (sentencepiece_bpe_) {
        if (piece.size() == 6U && piece.starts_with("<0x") && piece[5] == '>') {
            const auto nibble = [](char value) -> int {
                if (value >= '0' && value <= '9') return value - '0';
                if (value >= 'A' && value <= 'F') return value - 'A' + 10;
                return -1;
            };
            const auto high = nibble(piece[3]);
            const auto low = nibble(piece[4]);
            if (high < 0 || low < 0) {
                result.errors.emplace_back("Gemma byte fallback token is malformed");
                return result;
            }
            result.value.push_back(static_cast<char>((high << 4) | low));
            return result;
        }
        for (std::size_t cursor = 0U; cursor < piece.size();) {
            if (piece.compare(cursor, std::string_view("▁").size(), "▁") == 0) {
                result.value.push_back(' ');
                cursor += std::string_view("▁").size();
            } else {
                result.value.push_back(piece[cursor++]);
            }
        }
        return result;
    }
    std::size_t cursor = 0;
    while (cursor < piece.size()) {
        const auto decoded = decode_utf8(piece, cursor);
        if (!decoded || decoded->first >= codepoint_to_byte_.size() ||
            codepoint_to_byte_[decoded->first] < 0) {
            result.errors.emplace_back("vocabulary piece is not valid byte-level encoding");
            result.value.clear();
            return result;
        }
        result.value.push_back(static_cast<char>(
            static_cast<unsigned char>(codepoint_to_byte_[decoded->first])));
        cursor += decoded->second;
    }
    return result;
}

ParseResult<std::string> ModelTokenizer::decode(
    std::span<const std::uint32_t> tokens) const {
    ParseResult<std::string> result;
    for (const auto token : tokens) {
        const auto piece = decode_token(token);
        if (!piece.ok()) {
            result.errors = piece.errors;
            result.value.clear();
            return result;
        }
        result.value += piece.value;
    }
    return result;
}

std::string render_glm52_user_prompt(std::string_view user_text,
                                     std::string_view reasoning_effort,
                                     bool enable_thinking) {
    const std::array messages{ChatMessage{ChatRole::User,
                                          std::string(user_text)}};
    return render_glm52_chat_prompt(messages, reasoning_effort,
                                    enable_thinking);
}

std::string render_glm52_chat_prompt(std::span<const ChatMessage> messages,
                                     std::string_view reasoning_effort,
                                     bool enable_thinking) {
    std::string output = "[gMASK]<sop>\n";
    if (enable_thinking) {
        std::string effort(reasoning_effort);
        if (!effort.empty()) {
            effort[0] = static_cast<char>(
                std::toupper(static_cast<unsigned char>(effort[0])));
            for (std::size_t index = 1; index < effort.size(); ++index) {
                effort[index] = static_cast<char>(
                    std::tolower(static_cast<unsigned char>(effort[index])));
            }
        }
        output += "<|system|>Reasoning Effort: " + effort;
    }
    for (const auto& message : messages) {
        switch (message.role) {
            case ChatRole::System: output += "<|system|>"; break;
            case ChatRole::User: output += "<|user|>"; break;
            case ChatRole::Assistant: output += "<|assistant|>"; break;
            case ChatRole::Tool: output += "<|tool|>"; break;
        }
        if (!message.name.empty()) output += message.name + ": ";
        if (message.role == ChatRole::Assistant) {
            output += enable_thinking ? "<think>" : "<think></think>";
        }
        output += message.content;
    }
    output += "<|assistant|>";
    output += enable_thinking ? "<think>" : "<think></think>";
    return output;
}

std::string render_glm53_chat_prompt(std::span<const ChatMessage> messages,
                                     std::string_view reasoning_effort,
                                     bool clear_thinking) {
    const auto effort = reasoning_effort == "low" ? "Low"
        : reasoning_effort == "high" ? "High" : "Max";
    std::string output =
        std::string("[gMASK]<sop><|system|>Reasoning Effort: ") + effort;
    bool observing = false;
    for (const auto& message : messages) {
        switch (message.role) {
            case ChatRole::System:
                output += "<|system|>" + message.content;
                observing = false;
                break;
            case ChatRole::User:
                output += "<|user|>" + message.content;
                observing = false;
                break;
            case ChatRole::Assistant: {
                output += "<|assistant|>";
                if (clear_thinking) output += "<think></think>";
                auto begin = message.content.find_first_not_of(" \t\r\n");
                if (begin != std::string::npos) {
                    const auto end = message.content.find_last_not_of(" \t\r\n");
                    output.append(message.content, begin, end - begin + 1U);
                }
                observing = false;
                break;
            }
            case ChatRole::Tool:
                if (!observing) output += "<|observation|>";
                output += "<tool_response>" + message.content +
                          "</tool_response>";
                observing = true;
                break;
        }
    }
    output += "<|assistant|><think>";
    return output;
}

std::string render_glm53_user_prompt(std::string_view user_text,
                                     std::string_view reasoning_effort,
                                     bool clear_thinking) {
    const std::array messages{ChatMessage{ChatRole::User,
                                          std::string(user_text)}};
    return render_glm53_chat_prompt(messages, reasoning_effort,
                                    clear_thinking);
}

std::string render_deepseek_v4_user_prompt(std::string_view user_text,
                                           bool enable_thinking) {
    const std::array messages{ChatMessage{ChatRole::User,
                                          std::string(user_text)}};
    return render_deepseek_v4_chat_prompt(messages, enable_thinking);
}

std::string render_deepseek_v4_chat_prompt(
    std::span<const ChatMessage> messages, bool enable_thinking) {
    std::string output = "<｜begin▁of▁sentence｜>";
    bool continuing_tool_results = false;
    for (const auto& message : messages) {
        switch (message.role) {
            case ChatRole::System:
                output += message.content;
                continuing_tool_results = false;
                continue;
            case ChatRole::User:
                output += "<｜User｜>" + message.content;
                continuing_tool_results = false;
                continue;
            case ChatRole::Assistant:
                output += "<｜Assistant｜></think>" + message.content +
                          "<｜end▁of▁sentence｜>";
                continuing_tool_results = false;
                continue;
            case ChatRole::Tool:
                output += continuing_tool_results ? "\n\n" : "<｜User｜>";
                output += "<tool_result>" + message.content + "</tool_result>";
                continuing_tool_results = true;
                continue;
        }
    }
    output += "<｜Assistant｜>";
    output += enable_thinking ? "<think>" : "</think>";
    return output;
}

std::string render_gemma4_user_prompt(std::string_view user_text,
                                      bool enable_thinking) {
    const std::array messages{ChatMessage{ChatRole::User,
                                          std::string(user_text)}};
    return render_gemma4_chat_prompt(messages, enable_thinking);
}

std::string render_gemma4_chat_prompt(
    std::span<const ChatMessage> messages, bool enable_thinking) {
    std::string output = "<bos>";
    for (const auto& message : messages) {
        switch (message.role) {
            case ChatRole::System: output += "<|turn>system\n"; break;
            case ChatRole::User: output += "<|turn>user\n"; break;
            case ChatRole::Assistant: output += "<|turn>model\n"; break;
            case ChatRole::Tool:
                output += "<|tool_response>response:";
                output += message.name.empty() ? "unknown" : message.name;
                output += "{value:<|\"|>";
                output += message.content;
                output += "<|\"|>}<tool_response|>";
                continue;
        }
        output += message.content;
        output += "<turn|>\n";
    }
    output += "<|turn>model\n";
    if (!enable_thinking) output += "<|channel>thought\n<channel|>";
    return output;
}

std::string render_laguna_user_prompt(std::string_view user_text,
                                      bool enable_thinking) {
    const std::array messages{ChatMessage{ChatRole::User,
                                          std::string(user_text)}};
    return render_laguna_chat_prompt(messages, enable_thinking);
}

std::string render_laguna_chat_prompt(std::span<const ChatMessage> messages,
                                      bool enable_thinking) {
    // The checkpoint's default system message. A leading system message with
    // non-empty content replaces it; one with empty content opts out of the
    // <system> block entirely, which is how the template was trained.
    constexpr std::string_view default_system =
        "You are a helpful, conversationally-fluent assistant made by Poolside. "
        "You are here to be helpful to users through natural language "
        "conversations.";
    const auto trailing_space = [](std::string_view text) {
        auto end = text.size();
        while (end > 0U) {
            const auto value = static_cast<unsigned char>(text[end - 1U]);
            if (value != ' ' && value != '\t' && value != '\n' && value != '\r') {
                break;
            }
            --end;
        }
        return text.substr(0U, end);
    };

    std::string output = "〈|EOS|〉";
    std::size_t first = 0U;
    std::string_view system = default_system;
    if (!messages.empty() && messages.front().role == ChatRole::System) {
        system = messages.front().content;
        first = 1U;
    }
    const auto stripped = trailing_space(system);
    if (!stripped.empty()) {
        output += "<system>";
        output += stripped;
        output += "</system>\n";
    }
    for (std::size_t index = first; index < messages.size(); ++index) {
        const auto& message = messages[index];
        switch (message.role) {
            case ChatRole::User:
                output += "<user>" + message.content + "</user>\n";
                continue;
            case ChatRole::Assistant:
                // Prior turns carry no separately stored reasoning here, so the
                // think block is emitted empty when thinking is enabled.
                output += "<assistant>";
                output += enable_thinking ? "<think></think>" : "</think>";
                output += message.content;
                output += "</assistant>\n";
                continue;
            case ChatRole::Tool:
                output += "<tool_response>" + message.content +
                          "</tool_response>\n";
                continue;
            case ChatRole::System:
                output += "<system>" + message.content + "</system>\n";
                continue;
        }
    }
    output += "<assistant>";
    output += enable_thinking ? "<think>" : "</think>";
    return output;
}

std::string render_inkling_user_prompt(std::string_view user_text,
                                       double reasoning_effort) {
    const std::array messages{ChatMessage{ChatRole::User,
                                          std::string(user_text)}};
    return render_inkling_chat_prompt(messages, reasoning_effort);
}

std::string render_inkling_chat_prompt(std::span<const ChatMessage> messages,
                                       double reasoning_effort) {
    // The template emits the thinking-effort system message exactly once,
    // immediately before the first non-system message, and again at the end if
    // no such message ever appeared. Emitting it twice, or in the wrong place,
    // changes the token sequence the model was trained on.
    const auto role_token = [](ChatRole role) -> std::string_view {
        switch (role) {
            case ChatRole::User: return "<|message_user|>";
            case ChatRole::Assistant: return "<|message_model|>";
            case ChatRole::System: return "<|message_system|>";
            case ChatRole::Tool: return "<|message_tool|>";
        }
        return "<|message_user|>";
    };
    const auto effort_block = [&]() {
        std::string block = "<|message_system|><|content_text|>Thinking effort level: ";
        const double clamped = std::clamp(reasoning_effort, 0.0, 0.99);
        if (clamped == 0.0) {
            block += "0";
        } else {
            // Jinja renders the float with Python's repr, which drops trailing
            // zeros; 0.9 must not become "0.900000".
            std::string rendered = std::to_string(clamped);
            while (rendered.size() > 1U && rendered.back() == '0') rendered.pop_back();
            if (!rendered.empty() && rendered.back() == '.') rendered.pop_back();
            block += rendered;
        }
        block += "<|end_message|>";
        return block;
    };

    std::string output;
    bool effort_emitted = false;
    for (const auto& message : messages) {
        if (!effort_emitted && message.role != ChatRole::System) {
            output += effort_block();
            effort_emitted = true;
        }
        const auto token = role_token(message.role);
        if (message.role == ChatRole::Tool) {
            output += token;
            output += message.name;
            output += "<|content_text|>";
            output += message.content;
            output += "<|end_message|>";
            continue;
        }
        output += token;
        output += "<|content_text|>";
        output += message.content;
        output += "<|end_message|>";
        if (message.role == ChatRole::Assistant) {
            output += "<|content_model_end_sampling|>";
        }
    }
    if (!effort_emitted) output += effort_block();
    output += "<|message_model|>";
    return output;
}

namespace {

// XTML, as `encoding_k3.py` builds it. `<|open|>`, `<|close|>`, `<|sep|>`, and
// `<|end_of_msg|>` are single special tokens; the tag names and attributes are
// ordinary text, so they go through BPE exactly as the reference's `_text`
// segments do.
constexpr std::string_view kXtmlOpen = "<|open|>";
constexpr std::string_view kXtmlClose = "<|close|>";
constexpr std::string_view kXtmlSeparator = "<|sep|>";
constexpr std::string_view kXtmlEndOfMessage = "<|end_of_msg|>";

// `_escape_attr_value`: only `&` and `"`, and `&` first so an escaped quote is
// not double-escaped.
[[nodiscard]] std::string escape_attribute(std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char character : value) {
        if (character == '&') {
            escaped += "&amp;";
        } else if (character == '"') {
            escaped += "&quot;";
        } else {
            escaped.push_back(character);
        }
    }
    return escaped;
}

void open_xtml_tag(std::string& output, std::string_view tag,
                   std::span<const std::pair<std::string_view, std::string_view>>
                       attributes = {}) {
    output += kXtmlOpen;
    output += tag;
    for (const auto& [key, value] : attributes) {
        output += ' ';
        output += key;
        output += "=\"";
        output += escape_attribute(value);
        output += '"';
    }
    output += kXtmlSeparator;
}

void close_xtml_tag(std::string& output, std::string_view tag) {
    output += kXtmlClose;
    output += tag;
    output += kXtmlSeparator;
}

}  // namespace

std::string render_kimi_k3_user_prompt(std::string_view user_text,
                                       bool enable_thinking) {
    const std::array messages{ChatMessage{ChatRole::User, std::string(user_text)}};
    return render_kimi_k3_chat_prompt(messages, enable_thinking);
}

std::string render_kimi_k3_chat_prompt(std::span<const ChatMessage> messages,
                                       bool enable_thinking) {
    std::string output;
    // `apply_chat_template` does `kwargs.setdefault("thinking_effort", "max")`,
    // so a thinking conversation opens with this internal system message. It is
    // part of the prompt the model was tuned on, not decoration: dropping it
    // changes the distribution.
    if (enable_thinking) {
        const std::array<std::pair<std::string_view, std::string_view>, 2U> attributes{
            std::pair<std::string_view, std::string_view>{"role", "system"},
            std::pair<std::string_view, std::string_view>{"type", "thinking-effort"}};
        open_xtml_tag(output, "message", attributes);
        output +=
            "`thinking_effort` guides on how much to think in your thinking "
            "channel (not including the response channel), supported values "
            "include `low`, `medium`, `high`, and `max`.\n"
            "Now the system is invoked with `thinking_effort=max`.";
        close_xtml_tag(output, "message");
        output += kXtmlEndOfMessage;
    }
    // `build_chat_segments` numbers tool results from one and resets the counter
    // at every assistant message, because the index refers to that message's
    // tool calls.
    std::uint32_t tool_index = 0U;
    for (const auto& message : messages) {
        std::string_view role;
        switch (message.role) {
            case ChatRole::System: role = "system"; break;
            case ChatRole::User: role = "user"; break;
            case ChatRole::Assistant: role = "assistant"; break;
            // A tool result carries its tool name and a one-based index. Strata's
            // `ChatMessage` has no ordinal, so the position among tool messages
            // is the index, which is what `build_chat_segments` counts.
            case ChatRole::Tool: role = "tool"; break;
        }
        std::vector<std::pair<std::string_view, std::string_view>> attributes;
        attributes.emplace_back("role", role);
        std::string index_text;
        if (message.role == ChatRole::Tool) {
            index_text = std::to_string(++tool_index);
            attributes.emplace_back("tool", message.name);
            attributes.emplace_back("index", index_text);
        } else if (!message.name.empty()) {
            attributes.emplace_back("name", message.name);
        }
        if (message.role == ChatRole::Assistant) tool_index = 0U;
        open_xtml_tag(output, "message", attributes);
        if (message.role == ChatRole::Assistant) {
            // The think channel is structural: in thinking mode every assistant
            // message carries the tags even with nothing in them, and in
            // non-thinking mode the channel is absent entirely.
            if (enable_thinking) {
                open_xtml_tag(output, "think");
                close_xtml_tag(output, "think");
            }
            open_xtml_tag(output, "response");
            output += message.content;
            close_xtml_tag(output, "response");
        } else {
            output += message.content;
        }
        close_xtml_tag(output, "message");
        output += kXtmlEndOfMessage;
    }
    const std::array<std::pair<std::string_view, std::string_view>, 1U> assistant{
        std::pair<std::string_view, std::string_view>{"role", "assistant"}};
    open_xtml_tag(output, "message", assistant);
    open_xtml_tag(output, enable_thinking ? "think" : "response");
    return output;
}

}  // namespace strata
