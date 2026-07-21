#include "strata/tokenizer.hpp"

#include "json_cursor.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <optional>
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
    if (ascii_number(static_cast<unsigned char>(value))) return true;
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

}  // namespace

ParseResult<std::vector<std::string>> pretokenize(
    TokenizerContract contract, std::string_view text) {
    const auto runes = decode_runes(text);
    if (!runes.ok()) {
        ParseResult<std::vector<std::string>> result;
        result.errors = runes.errors;
        return result;
    }
    return contract == TokenizerContract::Glm52
               ? pretokenize_glm(text, runes.value)
               : pretokenize_deepseek(text, runes.value);
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
        if (!glm52 && !deepseek_v4) {
            throw std::runtime_error(
                "tokenizer does not match a pinned GLM-5.2 or DeepSeek-V4 contract");
        }
        result.value.contract_ = deepseek_v4 ? TokenizerContract::DeepSeekV4
                                             : TokenizerContract::Glm52;
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
        output += message.role == ChatRole::User ? "<|user|>" : "<|assistant|>";
        if (message.role == ChatRole::Assistant) {
            output += enable_thinking ? "<think>" : "<think></think>";
        }
        output += message.content;
    }
    output += "<|assistant|>";
    output += enable_thinking ? "<think>" : "<think></think>";
    return output;
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
    for (const auto& message : messages) {
        output += message.role == ChatRole::User ? "<｜User｜>" : "<｜Assistant｜>";
        if (message.role == ChatRole::Assistant) {
            output += enable_thinking ? "<think>" : "</think>";
        }
        output += message.content;
    }
    output += "<｜Assistant｜>";
    output += enable_thinking ? "<think>" : "</think>";
    return output;
}

}  // namespace strata
