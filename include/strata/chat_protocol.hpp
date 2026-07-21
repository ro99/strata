#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strata {

inline constexpr unsigned int chat_protocol_version = 1U;
inline constexpr std::size_t maximum_chat_request_bytes = 16U * 1024U * 1024U;

enum class ChatRole : std::uint8_t {
    User,
    Assistant,
};

struct ChatMessage {
    ChatRole role{ChatRole::User};
    std::string content;
};

struct ChatRequest {
    std::string prompt;
    std::vector<ChatMessage> messages;
    bool includes_history{};
};

// Parses one frontend-to-runtime JSONL record. The protocol currently accepts
// {"command":"prompt","text":"..."} and optionally a canonical messages
// array. New clients send both: old runtimes use text, while new runtimes use
// messages. Unknown fields remain ignored.
[[nodiscard]] bool parse_chat_request(std::string_view line,
                                      ChatRequest& request,
                                      std::string& error);
[[nodiscard]] bool validate_chat_messages(std::span<const ChatMessage> messages,
                                          std::string& error);

}  // namespace strata
