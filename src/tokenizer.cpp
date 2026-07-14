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
        cursor.expect('[');
        auto left = cursor.parse_string();
        cursor.expect(',');
        auto right = cursor.parse_string();
        cursor.expect(']');
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

}  // namespace

GlmTokenizer::GlmTokenizer() {
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

ParseResult<GlmTokenizer> GlmTokenizer::load(const std::string& path) {
    ParseResult<GlmTokenizer> result;
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
        if (!saw_added || !saw_model || result.value.vocabulary_.size() != 154820U ||
            result.value.merge_ranks_.size() != 321649U ||
            added_tokens.size() != 36U || !result.value.ignore_merges_) {
            throw std::runtime_error("tokenizer does not match the pinned GLM-5.2 contract");
        }
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
            if (!result.value.id_to_piece_[token.id].empty()) {
                throw std::runtime_error("added token collides with the base vocabulary");
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

std::int32_t GlmTokenizer::token_id(std::string_view piece) const noexcept {
    for (const auto& token : added_tokens_) {
        if (token.content == piece) return static_cast<std::int32_t>(token.id);
    }
    const auto found = vocabulary_.find(std::string(piece));
    return found == vocabulary_.end() ? -1 : static_cast<std::int32_t>(found->second);
}

ParseResult<std::vector<std::uint32_t>> GlmTokenizer::encode_piece(
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

ParseResult<std::vector<std::uint32_t>> GlmTokenizer::encode_plain_chunk(
    std::string_view text) const {
    ParseResult<std::vector<std::uint32_t>> result;
    for (const auto value : text) {
        if (static_cast<unsigned char>(value) >= 0x80U) {
            result.errors.emplace_back(
                "non-ASCII encoding is not enabled; refusing to silently change tokenization");
            return result;
        }
    }
    const auto emit = [&](std::size_t begin, std::size_t end,
                          std::vector<std::uint32_t>& output,
                          std::vector<std::string>& errors) {
        const auto encoded = encode_piece(text.substr(begin, end - begin));
        if (!encoded.ok()) {
            errors.insert(errors.end(), encoded.errors.begin(), encoded.errors.end());
        } else {
            output.insert(output.end(), encoded.value.begin(), encoded.value.end());
        }
    };

    std::size_t cursor = 0;
    while (cursor < text.size()) {
        const auto begin = cursor;
        const auto current = static_cast<unsigned char>(text[cursor]);
        if (current == '\'' && cursor + 1U < text.size()) {
            const auto next = ascii_lower(static_cast<unsigned char>(text[cursor + 1U]));
            if (cursor + 2U < text.size()) {
                const auto third = ascii_lower(static_cast<unsigned char>(text[cursor + 2U]));
                if ((next == 'r' && third == 'e') || (next == 'v' && third == 'e') ||
                    (next == 'l' && third == 'l')) {
                    cursor += 3U;
                    emit(begin, cursor, result.value, result.errors);
                    continue;
                }
            }
            if (next == 's' || next == 't' || next == 'm' || next == 'd') {
                cursor += 2U;
                emit(begin, cursor, result.value, result.errors);
                continue;
            }
        }

        auto scan = cursor;
        if (!ascii_letter(current) && current != '\r' && current != '\n' &&
            !ascii_number(current)) {
            if (scan + 1U < text.size() &&
                ascii_letter(static_cast<unsigned char>(text[scan + 1U]))) {
                ++scan;
            } else {
                scan = text.size();
            }
        }
        if (scan < text.size() && ascii_letter(static_cast<unsigned char>(text[scan]))) {
            while (scan < text.size() &&
                   ascii_letter(static_cast<unsigned char>(text[scan]))) {
                ++scan;
            }
            cursor = scan;
            emit(begin, cursor, result.value, result.errors);
            continue;
        }

        if (ascii_number(current)) {
            while (cursor < text.size() && cursor - begin < 3U &&
                   ascii_number(static_cast<unsigned char>(text[cursor]))) {
                ++cursor;
            }
            emit(begin, cursor, result.value, result.errors);
            continue;
        }

        scan = cursor;
        if (current == ' ' && scan + 1U < text.size()) {
            const auto next = static_cast<unsigned char>(text[scan + 1U]);
            if (!ascii_space(next) && !ascii_letter(next) && !ascii_number(next)) ++scan;
        }
        if (scan < text.size()) {
            const auto value = static_cast<unsigned char>(text[scan]);
            if (!ascii_space(value) && !ascii_letter(value) && !ascii_number(value)) {
                while (scan < text.size()) {
                    const auto punctuation = static_cast<unsigned char>(text[scan]);
                    if (ascii_space(punctuation) || ascii_letter(punctuation) ||
                        ascii_number(punctuation)) {
                        break;
                    }
                    ++scan;
                }
                while (scan < text.size() && (text[scan] == '\r' || text[scan] == '\n')) ++scan;
                cursor = scan;
                emit(begin, cursor, result.value, result.errors);
                continue;
            }
        }

        if (ascii_space(current)) {
            auto run_end = cursor;
            std::size_t last_newline = text.size();
            while (run_end < text.size() &&
                   ascii_space(static_cast<unsigned char>(text[run_end]))) {
                if (text[run_end] == '\r' || text[run_end] == '\n') last_newline = run_end;
                ++run_end;
            }
            if (last_newline != text.size()) {
                cursor = last_newline + 1U;
            } else if (run_end == text.size()) {
                cursor = run_end;
            } else if (run_end - cursor > 1U) {
                cursor = run_end - 1U;
            } else {
                cursor = run_end;
            }
            emit(begin, cursor, result.value, result.errors);
            continue;
        }

        ++cursor;
        emit(begin, cursor, result.value, result.errors);
    }
    return result;
}

ParseResult<std::vector<std::uint32_t>> GlmTokenizer::encode(std::string_view text) const {
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

ParseResult<std::string> GlmTokenizer::decode_token(std::uint32_t token) const {
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

ParseResult<std::string> GlmTokenizer::decode(
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
    output += "<|user|>";
    output.append(user_text);
    output += "<|assistant|>";
    output += enable_thinking ? "<think>" : "<think></think>";
    return output;
}

}  // namespace strata
