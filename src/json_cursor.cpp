#include "json_cursor.hpp"

#include <cctype>
#include <limits>

namespace strata::detail {

namespace {

void append_utf8(std::string& output, std::uint32_t codepoint) {
    if (codepoint <= 0x7FU) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FFU) {
        output.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint <= 0xFFFFU) {
        output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
        output.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
}

[[nodiscard]] int hex_value(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

}  // namespace

JsonError::JsonError(std::size_t offset, const std::string& message)
    : std::runtime_error("JSON byte " + std::to_string(offset) + ": " + message),
      offset_(offset) {}

void JsonCursor::skip_whitespace() {
    while (offset_ < input_.size() &&
           std::isspace(static_cast<unsigned char>(input_[offset_])) != 0) {
        ++offset_;
    }
}

void JsonCursor::fail(const std::string& message) const {
    throw JsonError(offset_, message);
}

void JsonCursor::expect(char wanted) {
    skip_whitespace();
    if (offset_ >= input_.size() || input_[offset_] != wanted) {
        fail(std::string("expected '") + wanted + "'");
    }
    ++offset_;
}

bool JsonCursor::consume(char wanted) {
    skip_whitespace();
    if (offset_ < input_.size() && input_[offset_] == wanted) {
        ++offset_;
        return true;
    }
    return false;
}

char JsonCursor::peek() {
    skip_whitespace();
    if (offset_ >= input_.size()) fail("unexpected end of input");
    return input_[offset_];
}

bool JsonCursor::finished() {
    skip_whitespace();
    return offset_ == input_.size();
}

std::uint32_t JsonCursor::parse_hex_quad() {
    if (input_.size() - offset_ < 4U) fail("truncated Unicode escape");
    std::uint32_t value = 0;
    for (std::size_t index = 0; index < 4U; ++index) {
        const auto digit = hex_value(input_[offset_++]);
        if (digit < 0) fail("invalid Unicode escape");
        value = (value << 4U) | static_cast<std::uint32_t>(digit);
    }
    return value;
}

std::string JsonCursor::parse_string() {
    skip_whitespace();
    if (offset_ >= input_.size() || input_[offset_] != '"') fail("expected string");
    ++offset_;
    std::string output;
    while (offset_ < input_.size()) {
        const auto value = static_cast<unsigned char>(input_[offset_++]);
        if (value == '"') return output;
        if (value < 0x20U) fail("unescaped control character in string");
        if (value != '\\') {
            output.push_back(static_cast<char>(value));
            continue;
        }
        if (offset_ >= input_.size()) fail("truncated string escape");
        const auto escape = input_[offset_++];
        switch (escape) {
            case '"': output.push_back('"'); break;
            case '\\': output.push_back('\\'); break;
            case '/': output.push_back('/'); break;
            case 'b': output.push_back('\b'); break;
            case 'f': output.push_back('\f'); break;
            case 'n': output.push_back('\n'); break;
            case 'r': output.push_back('\r'); break;
            case 't': output.push_back('\t'); break;
            case 'u': {
                auto codepoint = parse_hex_quad();
                if (codepoint >= 0xD800U && codepoint <= 0xDBFFU) {
                    if (input_.size() - offset_ < 6U || input_[offset_] != '\\' ||
                        input_[offset_ + 1U] != 'u') {
                        fail("high surrogate is missing its low surrogate");
                    }
                    offset_ += 2U;
                    const auto low = parse_hex_quad();
                    if (low < 0xDC00U || low > 0xDFFFU) fail("invalid low surrogate");
                    codepoint = 0x10000U + ((codepoint - 0xD800U) << 10U) +
                                (low - 0xDC00U);
                } else if (codepoint >= 0xDC00U && codepoint <= 0xDFFFU) {
                    fail("unexpected low surrogate");
                }
                append_utf8(output, codepoint);
                break;
            }
            default: fail("unsupported string escape");
        }
    }
    fail("unterminated string");
}

std::uint64_t JsonCursor::parse_uint64() {
    skip_whitespace();
    if (offset_ >= input_.size() || input_[offset_] < '0' || input_[offset_] > '9') {
        fail("expected unsigned integer");
    }
    std::uint64_t value = 0;
    do {
        const auto digit = static_cast<std::uint64_t>(input_[offset_] - '0');
        if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10U) {
            fail("unsigned integer overflow");
        }
        value = value * 10U + digit;
        ++offset_;
    } while (offset_ < input_.size() && input_[offset_] >= '0' && input_[offset_] <= '9');
    return value;
}

bool JsonCursor::parse_bool() {
    skip_whitespace();
    if (input_.substr(offset_, 4U) == "true") {
        offset_ += 4U;
        return true;
    }
    if (input_.substr(offset_, 5U) == "false") {
        offset_ += 5U;
        return false;
    }
    fail("expected boolean");
}

void JsonCursor::skip_number() {
    if (offset_ < input_.size() && input_[offset_] == '-') ++offset_;
    if (offset_ >= input_.size()) fail("truncated number");
    if (input_[offset_] == '0') {
        ++offset_;
    } else {
        if (input_[offset_] < '1' || input_[offset_] > '9') fail("invalid number");
        while (offset_ < input_.size() && input_[offset_] >= '0' &&
               input_[offset_] <= '9') {
            ++offset_;
        }
    }
    if (offset_ < input_.size() && input_[offset_] == '.') {
        ++offset_;
        if (offset_ >= input_.size() || input_[offset_] < '0' || input_[offset_] > '9') {
            fail("invalid fractional number");
        }
        while (offset_ < input_.size() && input_[offset_] >= '0' &&
               input_[offset_] <= '9') {
            ++offset_;
        }
    }
    if (offset_ < input_.size() && (input_[offset_] == 'e' || input_[offset_] == 'E')) {
        ++offset_;
        if (offset_ < input_.size() && (input_[offset_] == '+' || input_[offset_] == '-')) {
            ++offset_;
        }
        if (offset_ >= input_.size() || input_[offset_] < '0' || input_[offset_] > '9') {
            fail("invalid number exponent");
        }
        while (offset_ < input_.size() && input_[offset_] >= '0' &&
               input_[offset_] <= '9') {
            ++offset_;
        }
    }
}

void JsonCursor::skip_value() {
    skip_value(0);
}

void JsonCursor::skip_value(std::size_t depth) {
    if (depth > 256U) fail("JSON nesting exceeds 256 levels");
    skip_whitespace();
    if (offset_ >= input_.size()) fail("unexpected end of input");
    switch (input_[offset_]) {
        case '"': static_cast<void>(parse_string()); return;
        case '{': {
            ++offset_;
            if (consume('}')) return;
            for (;;) {
                static_cast<void>(parse_string());
                expect(':');
                skip_value(depth + 1U);
                if (consume('}')) return;
                expect(',');
            }
        }
        case '[': {
            ++offset_;
            if (consume(']')) return;
            for (;;) {
                skip_value(depth + 1U);
                if (consume(']')) return;
                expect(',');
            }
        }
        case 't':
            if (input_.substr(offset_, 4U) != "true") fail("invalid literal");
            offset_ += 4U;
            return;
        case 'f':
            if (input_.substr(offset_, 5U) != "false") fail("invalid literal");
            offset_ += 5U;
            return;
        case 'n':
            if (input_.substr(offset_, 4U) != "null") fail("invalid literal");
            offset_ += 4U;
            return;
        default: skip_number(); return;
    }
}

}  // namespace strata::detail
