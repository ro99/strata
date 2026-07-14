#include "strata/trace.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>

namespace strata {

TraceParseResult parse_route_trace(const std::filesystem::path& path) {
    TraceParseResult result;
    std::ifstream input(path);
    if (!input) {
        result.errors.push_back("cannot open route trace: " + path.string());
        return result;
    }

    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#') continue;

        std::istringstream fields(line);
        RouteEvent event;
        std::uint64_t layer = 0;
        if (!(fields >> event.request >> event.token >> layer) ||
            layer > std::numeric_limits<std::uint32_t>::max()) {
            result.errors.push_back("line " + std::to_string(line_number) +
                                    ": expected REQUEST TOKEN LAYER EXPERT...");
            continue;
        }
        event.layer = static_cast<std::uint32_t>(layer);

        std::uint64_t expert = 0;
        std::unordered_set<std::uint32_t> unique;
        bool expert_range_error = false;
        while (fields >> expert) {
            if (expert > std::numeric_limits<std::uint32_t>::max()) {
                expert_range_error = true;
                break;
            }
            const auto id = static_cast<std::uint32_t>(expert);
            if (unique.insert(id).second) event.experts.push_back(id);
        }
        if (expert_range_error) {
            result.errors.push_back("line " + std::to_string(line_number) +
                                    ": expert id exceeds uint32 range");
            continue;
        }
        if (!fields.eof()) {
            result.errors.push_back("line " + std::to_string(line_number) +
                                    ": non-numeric route field");
            continue;
        }
        if (event.experts.empty()) {
            result.errors.push_back("line " + std::to_string(line_number) +
                                    ": route contains no experts");
            continue;
        }
        result.events.push_back(std::move(event));
    }
    if (result.events.empty() && result.errors.empty()) {
        result.errors.emplace_back("route trace contains no events");
    }
    return result;
}

}  // namespace strata
