#pragma once

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace strata::cli {

inline bool parse_u32(std::string_view text, std::uint32_t& value) {
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

inline bool parse_positive_u32(std::string_view text, std::uint32_t& value) {
    return parse_u32(text, value) && value > 0U;
}

inline bool parse_u64(std::string_view text, std::uint64_t& value) {
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

inline bool parse_double(std::string_view text, double& value,
                         double minimum, double maximum) {
    const std::string owned(text);
    char* end = nullptr;
    value = std::strtod(owned.c_str(), &end);
    return end != owned.c_str() && *end == '\0' && std::isfinite(value) &&
           value >= minimum && value <= maximum;
}

inline bool parse_devices(std::string_view text, std::vector<int>& devices) {
    devices.clear();
    std::size_t begin = 0U;
    while (begin < text.size()) {
        const auto end = text.find(',', begin);
        const auto part = text.substr(begin, end == std::string_view::npos
                                                ? text.size() - begin : end - begin);
        int device = -1;
        const auto parsed = std::from_chars(part.data(), part.data() + part.size(), device);
        if (part.empty() || parsed.ec != std::errc{} ||
            parsed.ptr != part.data() + part.size() || device < 0) return false;
        devices.push_back(device);
        if (end == std::string_view::npos) break;
        begin = end + 1U;
    }
    return !devices.empty();
}

inline std::string devices_text(const std::vector<int>& devices) {
    std::ostringstream output;
    for (std::size_t index = 0U; index < devices.size(); ++index) {
        if (index != 0U) output << ',';
        output << devices[index];
    }
    return output.str();
}

inline std::string json_escape(std::string_view text) {
    std::ostringstream output;
    for (const unsigned char value : text) {
        switch (value) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (value < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<unsigned int>(value)
                           << std::dec;
                } else {
                    output << static_cast<char>(value);
                }
        }
    }
    return output.str();
}

}  // namespace strata::cli
