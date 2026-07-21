#include "strata/chat_protocol.hpp"

#include "json_cursor.hpp"

#include <exception>
#include <string>

namespace strata {
namespace {

bool parse_message(detail::JsonCursor& cursor, ChatMessage& message,
                   std::string& error) {
    cursor.expect('{');
    std::string role;
    bool has_role = false;
    bool has_content = false;
    if (!cursor.consume('}')) {
        for (;;) {
            const auto key = cursor.parse_string();
            cursor.expect(':');
            if (key == "role") {
                if (has_role) {
                    error = "chat message contains duplicate role fields";
                    return false;
                }
                role = cursor.parse_string();
                has_role = true;
            } else if (key == "content") {
                if (has_content) {
                    error = "chat message contains duplicate content fields";
                    return false;
                }
                message.content = cursor.parse_string();
                has_content = true;
            } else {
                cursor.skip_value();
            }
            if (cursor.consume('}')) break;
            cursor.expect(',');
        }
    }
    if (!has_role || !has_content) {
        error = "chat messages require role and content fields";
        return false;
    }
    if (role == "user") {
        message.role = ChatRole::User;
    } else if (role == "assistant") {
        message.role = ChatRole::Assistant;
    } else {
        error = "chat message role must be \"user\" or \"assistant\"";
        return false;
    }
    if (message.role == ChatRole::User && message.content.empty()) {
        error = "user message content must not be empty";
        return false;
    }
    return true;
}

}  // namespace

bool validate_chat_messages(std::span<const ChatMessage> messages,
                            std::string& error) {
    if (messages.empty()) {
        error = "prompt request must contain at least one chat message";
        return false;
    }
    for (std::size_t index = 0; index < messages.size(); ++index) {
        const auto expected = index % 2U == 0U ? ChatRole::User
                                               : ChatRole::Assistant;
        if (messages[index].role != expected) {
            error = "chat messages must alternate user and assistant roles";
            return false;
        }
        if (messages[index].role == ChatRole::User &&
            messages[index].content.empty()) {
            error = "user message content must not be empty";
            return false;
        }
    }
    if (messages.back().role != ChatRole::User) {
        error = "prompt request must end with a user message";
        return false;
    }
    error.clear();
    return true;
}

bool parse_chat_request(std::string_view line, ChatRequest& request,
                        std::string& error) {
    request = {};
    error.clear();
    if (line.size() > maximum_chat_request_bytes) {
        error = "request exceeds the 16 MiB protocol limit";
        return false;
    }

    try {
        detail::JsonCursor cursor(line);
        cursor.expect('{');
        std::string command;
        bool has_command = false;
        bool has_text = false;
        bool has_messages = false;
        if (!cursor.consume('}')) {
            for (;;) {
                const std::string key = cursor.parse_string();
                cursor.expect(':');
                if (key == "command") {
                    if (has_command) {
                        error = "request contains duplicate command fields";
                        return false;
                    }
                    command = cursor.parse_string();
                    has_command = true;
                } else if (key == "text") {
                    if (has_text) {
                        error = "request contains duplicate text fields";
                        return false;
                    }
                    request.prompt = cursor.parse_string();
                    has_text = true;
                } else if (key == "messages") {
                    if (has_messages) {
                        error = "request contains duplicate messages fields";
                        return false;
                    }
                    cursor.expect('[');
                    if (!cursor.consume(']')) {
                        for (;;) {
                            ChatMessage message;
                            if (!parse_message(cursor, message, error)) return false;
                            request.messages.push_back(std::move(message));
                            if (cursor.consume(']')) break;
                            cursor.expect(',');
                        }
                    }
                    has_messages = true;
                } else {
                    cursor.skip_value();
                }
                if (cursor.consume('}')) break;
                cursor.expect(',');
            }
        }
        if (!cursor.finished()) {
            error = "request has trailing JSON data";
            return false;
        }
        if (!has_command || command != "prompt") {
            error = "request command must be \"prompt\"";
            return false;
        }
        if (!has_text && !has_messages) {
            error = "prompt request is missing text or messages";
            return false;
        }
        if (has_text && request.prompt.empty()) {
            error = "prompt must not be empty";
            return false;
        }
        if (has_messages) {
            if (!validate_chat_messages(request.messages, error)) return false;
            if (has_text && request.prompt != request.messages.back().content) {
                error = "text must match the final user message";
                return false;
            }
            request.prompt = request.messages.back().content;
            request.includes_history = true;
        } else {
            request.messages.push_back({ChatRole::User, request.prompt});
        }
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

}  // namespace strata
