#pragma once

#include <string>
#include <vector>

namespace strata {

struct ValidationResult {
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

template <typename T>
struct ParseResult {
    T value;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

}  // namespace strata
