#include "strata/trace.hpp"

#include "json_cursor.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>

namespace strata {

namespace {

bool parse_json_event(std::string_view line, RouteEvent& event,
                      bool& header, std::string& error) {
    try {
        detail::JsonCursor cursor(line);
        bool saw_layer = false;
        bool saw_position = false;
        bool saw_experts = false;
        cursor.expect('{');
        if (!cursor.consume('}')) {
            for (;;) {
                const auto key = cursor.parse_string();
                cursor.expect(':');
                if (key == "schema") {
                    static_cast<void>(cursor.parse_string());
                    header = true;
                } else if (key == "request") {
                    event.request = cursor.parse_uint64();
                } else if (key == "token_position" || key == "position") {
                    event.token_position = cursor.parse_uint64();
                    saw_position = true;
                } else if (key == "layer") {
                    const auto layer = cursor.parse_uint64();
                    if (layer > std::numeric_limits<std::uint32_t>::max()) {
                        error = "layer exceeds uint32 range";
                        return false;
                    }
                    event.layer = static_cast<std::uint32_t>(layer);
                    saw_layer = true;
                } else if (key == "phase") {
                    const auto phase = cursor.parse_string();
                    if (phase == "prefill") event.phase = RoutePhase::Prefill;
                    else if (phase == "decode") event.phase = RoutePhase::Decode;
                    else if (phase == "unknown") event.phase = RoutePhase::Unknown;
                    else {
                        error = "unknown route phase";
                        return false;
                    }
                } else if (key == "experts") {
                    cursor.expect('[');
                    if (!cursor.consume(']')) {
                        for (;;) {
                            const auto expert = cursor.parse_uint64();
                            if (expert > std::numeric_limits<std::uint32_t>::max()) {
                                error = "expert id exceeds uint32 range";
                                return false;
                            }
                            event.experts.push_back(static_cast<std::uint32_t>(expert));
                            if (cursor.consume(']')) break;
                            cursor.expect(',');
                        }
                    }
                    saw_experts = true;
                } else {
                    cursor.skip_value();
                }
                if (cursor.consume('}')) break;
                cursor.expect(',');
            }
        }
        if (!cursor.finished()) {
            error = "trailing JSON content";
            return false;
        }
        if (header) return true;
        if (!saw_position || !saw_layer || !saw_experts || event.experts.empty()) {
            error = "route JSON lacks position, layer, or experts";
            return false;
        }
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

}  // namespace

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

        if (line[first] == '{') {
            RouteEvent event;
            bool header = false;
            std::string error;
            if (!parse_json_event(std::string_view(line).substr(first), event,
                                  header, error)) {
                result.errors.push_back("line " + std::to_string(line_number) +
                                        ": " + error);
            } else if (!header) {
                result.events.push_back(std::move(event));
            }
            continue;
        }

        std::istringstream fields(line);
        RouteEvent event;
        std::uint64_t layer = 0;
        if (!(fields >> event.request >> event.token_position >> layer) ||
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

std::string_view to_string(RoutePhase phase) noexcept {
    switch (phase) {
        case RoutePhase::Unknown: return "unknown";
        case RoutePhase::Prefill: return "prefill";
        case RoutePhase::Decode: return "decode";
    }
    return "unknown";
}

ValidationResult RouteTraceWriter::open(const std::filesystem::path& path) {
    ValidationResult result;
    if (output_.is_open()) {
        result.errors.emplace_back("route trace writer is already open");
        return result;
    }
    output_.open(path, std::ios::out | std::ios::trunc);
    if (!output_.is_open()) {
        result.errors.emplace_back("cannot open route trace: " + path.string());
        return result;
    }
    output_ << "{\"schema\":\"strata.route_trace\",\"version\":2,"
               "\"position_base\":0,\"layer_base\":0,"
               "\"coefficient_encoding\":\"float32-roundtrip-decimal\"}\n";
    if (!output_.good()) result.errors.emplace_back("cannot write route trace header");
    return result;
}

ValidationResult RouteTraceWriter::write(const RouteEvent& event) {
    ValidationResult result;
    if (!output_.is_open()) {
        result.errors.emplace_back("route trace writer is not open");
        return result;
    }
    if (event.experts.empty() ||
        (!event.coefficients.empty() &&
         event.coefficients.size() != event.experts.size())) {
        result.errors.emplace_back("route event has incompatible expert coefficients");
        return result;
    }
    output_ << "{\"request\":" << event.request
            << ",\"phase\":\"" << to_string(event.phase)
            << "\",\"token_position\":" << event.token_position
            << ",\"layer\":" << event.layer << ",\"experts\":[";
    for (std::size_t index = 0; index < event.experts.size(); ++index) {
        if (index != 0U) output_ << ',';
        output_ << event.experts[index];
    }
    output_ << "],\"coefficients\":[" << std::setprecision(
        std::numeric_limits<float>::max_digits10);
    for (std::size_t index = 0; index < event.coefficients.size(); ++index) {
        if (index != 0U) output_ << ',';
        output_ << event.coefficients[index];
    }
    output_ << "]}\n";
    if (!output_.good()) result.errors.emplace_back("cannot write route trace event");
    return result;
}

ValidationResult RouteTraceWriter::flush() {
    ValidationResult result;
    if (!output_.is_open()) return result;
    output_.flush();
    if (!output_.good()) result.errors.emplace_back("cannot flush route trace");
    return result;
}

}  // namespace strata
