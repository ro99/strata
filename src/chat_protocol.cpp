#include "strata/chat_protocol.hpp"

#include "json_cursor.hpp"

#include <exception>
#include <string>

namespace strata {

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
        if (!has_text) {
            error = "prompt request is missing its text field";
            return false;
        }
        if (request.prompt.empty()) {
            error = "prompt must not be empty";
            return false;
        }
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

}  // namespace strata
