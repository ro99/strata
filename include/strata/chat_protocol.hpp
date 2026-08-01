#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strata {

inline constexpr unsigned int chat_protocol_version = 1U;
inline constexpr std::size_t maximum_chat_request_bytes = 16U * 1024U * 1024U;

enum class ChatRole : std::uint8_t {
    System,
    User,
    Assistant,
    Tool,
};

enum class ChatContentKind : std::uint8_t {
    Text,
    Image,
};

struct ChatContentPart {
    ChatContentKind kind{ChatContentKind::Text};
    std::string data;
    std::string mime_type;
};

struct ChatMessage {
    ChatMessage() = default;
    ChatMessage(ChatRole message_role, std::string message_content,
                std::string message_name = {})
        : role(message_role), content(std::move(message_content)),
          name(std::move(message_name)) {}

    ChatRole role{ChatRole::User};
    std::string content;
    std::string name;
    std::vector<ChatContentPart> parts;
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
