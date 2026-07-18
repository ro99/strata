#pragma once

#include "strata/result.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strata {

enum class RoutePhase : std::uint8_t {
    Unknown,
    Prefill,
    Decode,
};

struct RouteEvent {
    std::uint64_t request{};
    std::uint64_t token_position{};
    std::uint32_t layer{};
    std::vector<std::uint32_t> experts;
    std::vector<float> coefficients;
    RoutePhase phase{RoutePhase::Unknown};
};

struct TraceParseResult {
    std::vector<RouteEvent> events;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

/*
 * Text format v1:
 *   # strata-route-trace-v1
 *   REQUEST TOKEN LAYER EXPERT [EXPERT ...]
 */
[[nodiscard]] TraceParseResult parse_route_trace(const std::filesystem::path& path);

class RouteTraceWriter {
public:
    [[nodiscard]] ValidationResult open(const std::filesystem::path& path);
    [[nodiscard]] ValidationResult write(const RouteEvent& event);
    [[nodiscard]] ValidationResult flush();
    [[nodiscard]] bool is_open() const noexcept { return output_.is_open(); }

private:
    std::ofstream output_;
};

[[nodiscard]] std::string_view to_string(RoutePhase phase) noexcept;

}  // namespace strata
