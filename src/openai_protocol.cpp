#include "strata/openai_protocol.hpp"

#include "json_cursor.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <exception>
#include <limits>

namespace strata {
namespace {

bool parse_content_part(detail::JsonCursor& cursor, std::string& content,
                        std::string& error) {
    cursor.expect('{');
    std::string type;
    std::string text;
    while (!cursor.consume('}')) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        if (key == "type") type = cursor.parse_string();
        else if (key == "text") text = cursor.parse_string();
        else cursor.skip_value();
        if (cursor.consume('}')) break;
        cursor.expect(',');
    }
    if (type == "text") {
        content += text;
        return true;
    }
    error = type == "image_url"
        ? "this loaded model does not support image content"
        : "unsupported message content part";
    return false;
}

bool parse_message(detail::JsonCursor& cursor, ChatMessage& message,
                   std::string& error) {
    cursor.expect('{');
    std::string role;
    bool has_role = false;
    bool has_content = false;
    while (!cursor.consume('}')) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        if (key == "role") {
            role = cursor.parse_string();
            has_role = true;
        } else if (key == "name") {
            message.name = cursor.parse_string();
        } else if (key == "content") {
            has_content = true;
            if (cursor.peek() == '"') {
                message.content = cursor.parse_string();
            } else if (cursor.peek() == '[') {
                cursor.expect('[');
                while (!cursor.consume(']')) {
                    if (!parse_content_part(cursor, message.content, error)) return false;
                    if (cursor.consume(']')) break;
                    cursor.expect(',');
                }
            } else {
                cursor.skip_value();
            }
        } else {
            cursor.skip_value();
        }
        if (cursor.consume('}')) break;
        cursor.expect(',');
    }
    if (!has_role || !has_content) {
        error = "messages require role and content";
        return false;
    }
    if (role == "system" || role == "developer") message.role = ChatRole::System;
    else if (role == "user") message.role = ChatRole::User;
    else if (role == "assistant") message.role = ChatRole::Assistant;
    else if (role == "tool") message.role = ChatRole::Tool;
    else {
        error = "unsupported message role";
        return false;
    }
    return true;
}

bool parse_stop(detail::JsonCursor& cursor, std::vector<std::string>& stop,
                std::string& error) {
    if (cursor.peek() == 'n') {
        cursor.skip_value();
        return true;
    }
    if (cursor.peek() == '"') {
        stop.push_back(cursor.parse_string());
    } else {
        cursor.expect('[');
        while (!cursor.consume(']')) {
            stop.push_back(cursor.parse_string());
            if (cursor.consume(']')) break;
            cursor.expect(',');
        }
    }
    if (stop.size() > 4U ||
        std::any_of(stop.begin(), stop.end(), [](const auto& value) {
            return value.empty();
        })) {
        error = "stop must contain one to four non-empty strings";
        return false;
    }
    return true;
}

bool parse_logit_bias(detail::JsonCursor& cursor, SamplingOptions& sampling,
                      std::string& error) {
    if (cursor.peek() == 'n') {
        cursor.skip_value();
        return true;
    }
    cursor.expect('{');
    while (!cursor.consume('}')) {
        const auto token_text = cursor.parse_string();
        cursor.expect(':');
        std::uint32_t token = 0U;
        const auto parsed = std::from_chars(
            token_text.data(), token_text.data() + token_text.size(), token);
        const double bias = cursor.parse_number();
        if (parsed.ec != std::errc{} ||
            parsed.ptr != token_text.data() + token_text.size() ||
            bias < -100.0 || bias > 100.0) {
            error = "logit_bias requires token-id keys and values within [-100, 100]";
            return false;
        }
        sampling.logit_bias.emplace_back(token, bias);
        if (cursor.consume('}')) break;
        cursor.expect(',');
    }
    return true;
}

bool parse_response_format(detail::JsonCursor& cursor, bool& json_object,
                           std::string& error) {
    cursor.expect('{');
    std::string type;
    while (!cursor.consume('}')) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        if (key == "type") type = cursor.parse_string();
        else cursor.skip_value();
        if (cursor.consume('}')) break;
        cursor.expect(',');
    }
    if (type != "text" && type != "json_object") {
        error = "response_format.type must be text or json_object";
        return false;
    }
    json_object = type == "json_object";
    return true;
}

}  // namespace

bool parse_openai_chat_request(
    std::string_view json, OpenAiChatRequest& request, std::string& error) {
    request = {};
    error.clear();
    if (json.size() > maximum_chat_request_bytes) {
        error = "request exceeds the 16 MiB limit";
        return false;
    }
    bool has_model = false;
    bool has_messages = false;
    bool tool_choice_none = false;
    try {
        detail::JsonCursor cursor(json);
        cursor.expect('{');
        while (!cursor.consume('}')) {
            const auto key = cursor.parse_string();
            cursor.expect(':');
            if (key == "model") {
                request.model = cursor.parse_string();
                has_model = true;
            } else if (key == "messages") {
                cursor.expect('[');
                while (!cursor.consume(']')) {
                    ChatMessage message;
                    if (!parse_message(cursor, message, error)) return false;
                    request.messages.push_back(std::move(message));
                    if (cursor.consume(']')) break;
                    cursor.expect(',');
                }
                has_messages = true;
            } else if (key == "temperature") {
                request.generation.sampling.temperature = cursor.parse_number();
            } else if (key == "top_p") {
                request.generation.sampling.top_p = cursor.parse_number();
            } else if (key == "n") {
                const auto value = cursor.parse_uint64();
                if (value > 16U) {
                    error = "n must be within [1, 16]";
                    return false;
                }
                request.n = static_cast<std::uint32_t>(value);
            } else if (key == "stream") {
                request.stream = cursor.parse_bool();
            } else if (key == "stop") {
                if (!parse_stop(cursor, request.generation.stop, error)) return false;
            } else if (key == "max_tokens" || key == "max_completion_tokens") {
                const auto value = cursor.parse_uint64();
                if (value > std::numeric_limits<std::uint32_t>::max()) {
                    error = "max_tokens is too large";
                    return false;
                }
                request.generation.maximum_new_tokens =
                    static_cast<std::uint32_t>(value);
                request.has_max_tokens = true;
            } else if (key == "presence_penalty") {
                request.generation.sampling.presence_penalty = cursor.parse_number();
            } else if (key == "frequency_penalty") {
                request.generation.sampling.frequency_penalty = cursor.parse_number();
            } else if (key == "logit_bias") {
                if (!parse_logit_bias(cursor, request.generation.sampling, error)) return false;
            } else if (key == "seed") {
                if (cursor.peek() == 'n') cursor.skip_value();
                else {
                    request.generation.sampling.seed = cursor.parse_uint64();
                    request.has_seed = true;
                }
            } else if (key == "user") {
                if (cursor.peek() == 'n') cursor.skip_value();
                else request.user = cursor.parse_string();
            } else if (key == "response_format") {
                if (!parse_response_format(cursor, request.json_object, error)) return false;
            } else if (key == "logprobs") {
                request.logprobs = cursor.parse_bool();
                request.generation.sampling.return_logprobs = request.logprobs;
            } else if (key == "top_logprobs") {
                const auto value = cursor.parse_uint64();
                if (value > 20U) {
                    error = "top_logprobs must be within [0, 20]";
                    return false;
                }
                request.generation.sampling.top_logprobs =
                    static_cast<std::uint32_t>(value);
            } else if (key == "tools") {
                request.has_tools = cursor.peek() != 'n';
                cursor.skip_value();
            } else if (key == "tool_choice") {
                if (cursor.peek() == '"') {
                    tool_choice_none = cursor.parse_string() == "none";
                    request.has_tools = request.has_tools || !tool_choice_none;
                } else {
                    request.has_tools = true;
                    cursor.skip_value();
                }
            } else {
                cursor.skip_value();
            }
            if (cursor.consume('}')) break;
            cursor.expect(',');
        }
        if (!cursor.finished()) {
            error = "request has trailing JSON data";
            return false;
        }
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
    if (!has_model || request.model.empty() || !has_messages) {
        error = "model and messages are required";
        return false;
    }
    if (tool_choice_none) request.has_tools = false;
    if (request.n == 0U || request.generation.maximum_new_tokens == 0U ||
        request.generation.sampling.temperature < 0.0 ||
        request.generation.sampling.temperature > 2.0 ||
        request.generation.sampling.top_p <= 0.0 ||
        request.generation.sampling.top_p > 1.0 ||
        std::abs(request.generation.sampling.presence_penalty) > 2.0 ||
        std::abs(request.generation.sampling.frequency_penalty) > 2.0) {
        error = "invalid generation parameter range";
        return false;
    }
    if (request.generation.sampling.top_logprobs != 0U && !request.logprobs) {
        error = "top_logprobs requires logprobs=true";
        return false;
    }
    if (!validate_chat_messages(request.messages, error)) return false;
    if (request.json_object) {
        request.messages.insert(request.messages.begin(),
            {ChatRole::System, "Return only one valid JSON object."});
    }
    return true;
}

bool parse_openai_completion_request(
    std::string_view json, OpenAiChatRequest& request, std::string& error) {
    request = {};
    error.clear();
    bool has_model = false;
    bool has_prompt = false;
    std::string prompt;
    try {
        detail::JsonCursor cursor(json);
        cursor.expect('{');
        while (!cursor.consume('}')) {
            const auto key = cursor.parse_string();
            cursor.expect(':');
            if (key == "model") {
                request.model = cursor.parse_string();
                has_model = true;
            } else if (key == "prompt") {
                prompt = cursor.parse_string();
                has_prompt = true;
            } else if (key == "temperature") {
                request.generation.sampling.temperature = cursor.parse_number();
            } else if (key == "top_p") {
                request.generation.sampling.top_p = cursor.parse_number();
            } else if (key == "n") {
                const auto value = cursor.parse_uint64();
                if (value > 16U) {
                    error = "n must be within [1, 16]";
                    return false;
                }
                request.n = static_cast<std::uint32_t>(value);
            } else if (key == "stream") {
                request.stream = cursor.parse_bool();
            } else if (key == "stop") {
                if (!parse_stop(cursor, request.generation.stop, error)) return false;
            } else if (key == "max_tokens") {
                const auto value = cursor.parse_uint64();
                if (value > std::numeric_limits<std::uint32_t>::max()) {
                    error = "max_tokens is too large";
                    return false;
                }
                request.generation.maximum_new_tokens = static_cast<std::uint32_t>(value);
                request.has_max_tokens = true;
            } else if (key == "presence_penalty") {
                request.generation.sampling.presence_penalty = cursor.parse_number();
            } else if (key == "frequency_penalty") {
                request.generation.sampling.frequency_penalty = cursor.parse_number();
            } else if (key == "logit_bias") {
                if (!parse_logit_bias(cursor, request.generation.sampling, error)) return false;
            } else if (key == "seed") {
                if (cursor.peek() == 'n') cursor.skip_value();
                else {
                    request.generation.sampling.seed = cursor.parse_uint64();
                    request.has_seed = true;
                }
            } else if (key == "user") {
                if (cursor.peek() == 'n') cursor.skip_value();
                else request.user = cursor.parse_string();
            } else if (key == "logprobs") {
                const auto count = cursor.parse_uint64();
                if (count > 20U) {
                    error = "logprobs must be within [0, 20]";
                    return false;
                }
                request.logprobs = count != 0U;
                request.generation.sampling.return_logprobs = request.logprobs;
                request.generation.sampling.top_logprobs =
                    static_cast<std::uint32_t>(count);
            } else {
                cursor.skip_value();
            }
            if (cursor.consume('}')) break;
            cursor.expect(',');
        }
        if (!cursor.finished()) {
            error = "request has trailing JSON data";
            return false;
        }
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
    if (!has_model || request.model.empty() || !has_prompt || prompt.empty()) {
        error = "model and a non-empty string prompt are required";
        return false;
    }
    if (request.n == 0U || request.n > 16U ||
        request.generation.maximum_new_tokens == 0U ||
        request.generation.sampling.temperature < 0.0 ||
        request.generation.sampling.temperature > 2.0 ||
        request.generation.sampling.top_p <= 0.0 ||
        request.generation.sampling.top_p > 1.0 ||
        std::abs(request.generation.sampling.presence_penalty) > 2.0 ||
        std::abs(request.generation.sampling.frequency_penalty) > 2.0) {
        error = "invalid generation parameter range";
        return false;
    }
    request.messages.push_back({ChatRole::User, std::move(prompt)});
    return true;
}

bool parse_openai_tokenize_request(
    std::string_view json, std::string& model, std::string& text,
    std::string& error) {
    model.clear();
    text.clear();
    error.clear();
    try {
        detail::JsonCursor cursor(json);
        cursor.expect('{');
        while (!cursor.consume('}')) {
            const auto key = cursor.parse_string();
            cursor.expect(':');
            if (key == "model") model = cursor.parse_string();
            else if (key == "text" || key == "prompt") text = cursor.parse_string();
            else cursor.skip_value();
            if (cursor.consume('}')) break;
            cursor.expect(',');
        }
        if (!cursor.finished()) throw detail::JsonError(cursor.offset(), "trailing data");
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
    if (model.empty() || text.empty()) {
        error = "model and non-empty text are required";
        return false;
    }
    return true;
}

bool is_json_object(std::string_view json) noexcept {
    try {
        detail::JsonCursor cursor(json);
        if (cursor.peek() != '{') return false;
        cursor.skip_value();
        return cursor.finished();
    } catch (...) {
        return false;
    }
}

}  // namespace strata
