#include "strata/engine/chat_protocol.hpp"

#include <string>

namespace strata {

bool validate_chat_messages(std::span<const ChatMessage> messages,
                            std::string& error) {
    if (messages.empty()) {
        error = "prompt request must contain at least one chat message";
        return false;
    }
    ChatRole previous = ChatRole::System;
    bool started = false;
    for (const auto& message : messages) {
        if (message.role == ChatRole::System) {
            if (started) {
                error = "system messages must precede conversation messages";
                return false;
            }
            continue;
        }
        if ((message.role == ChatRole::User && started &&
             previous == ChatRole::User) ||
            (message.role == ChatRole::Assistant &&
             (!started || previous == ChatRole::Assistant)) ||
            (message.role == ChatRole::Tool && previous != ChatRole::Assistant &&
             previous != ChatRole::Tool)) {
            error = "chat messages must alternate valid conversation roles";
            return false;
        }
        if (message.role != ChatRole::Assistant && message.content.empty() &&
            message.parts.empty()) {
            error = "chat message content must not be empty";
            return false;
        }
        if (!message.parts.empty() && message.role != ChatRole::User) {
            error = "multimodal content is supported only in user messages";
            return false;
        }
        previous = message.role;
        started = true;
    }
    if (!started || (previous != ChatRole::User && previous != ChatRole::Tool)) {
        error = "prompt request must end with a user or tool message";
        return false;
    }
    error.clear();
    return true;
}

}  // namespace strata
