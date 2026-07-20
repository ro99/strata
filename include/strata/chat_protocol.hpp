#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace strata {

inline constexpr unsigned int chat_protocol_version = 1U;
inline constexpr std::size_t maximum_chat_request_bytes = 16U * 1024U * 1024U;

struct ChatRequest {
    std::string prompt;
};

// Parses one frontend-to-runtime JSONL record. The protocol currently accepts
// {"command":"prompt","text":"..."}; unknown fields are ignored so the
// envelope can grow without breaking older runtimes.
[[nodiscard]] bool parse_chat_request(std::string_view line,
                                      ChatRequest& request,
                                      std::string& error);

}  // namespace strata
