#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strata {

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

[[nodiscard]] bool validate_chat_messages(std::span<const ChatMessage> messages,
                                          std::string& error);

}  // namespace strata
