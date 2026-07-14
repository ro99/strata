#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strata::detail {

class JsonError final : public std::runtime_error {
public:
    JsonError(std::size_t offset, const std::string& message);

    [[nodiscard]] std::size_t offset() const noexcept { return offset_; }

private:
    std::size_t offset_{};
};

class JsonCursor {
public:
    explicit JsonCursor(std::string_view input) : input_(input) {}

    void expect(char wanted);
    [[nodiscard]] bool consume(char wanted);
    [[nodiscard]] char peek();
    [[nodiscard]] bool finished();
    [[nodiscard]] std::string parse_string();
    [[nodiscard]] std::uint64_t parse_uint64();
    [[nodiscard]] bool parse_bool();
    void skip_value();

    [[nodiscard]] std::size_t offset() const noexcept { return offset_; }

private:
    void skip_whitespace();
    void skip_value(std::size_t depth);
    void skip_number();
    [[nodiscard]] std::uint32_t parse_hex_quad();
    [[noreturn]] void fail(const std::string& message) const;

    std::string_view input_;
    std::size_t offset_{};
};

}  // namespace strata::detail
