#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace strata {

struct RouteEvent {
    std::uint64_t request{};
    std::uint64_t token{};
    std::uint32_t layer{};
    std::vector<std::uint32_t> experts;
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

}  // namespace strata
