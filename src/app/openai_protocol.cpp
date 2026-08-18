#include "strata/openai_protocol.hpp"

#include "../platform/json_cursor.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <exception>
#include <limits>
#include <sstream>

namespace strata {
namespace {

bool decode_base64(std::string_view encoded, std::string& decoded) {
    static constexpr std::string_view alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    if (encoded.empty() || encoded.size() % 4U != 0U) return false;
    decoded.clear();
    decoded.reserve(encoded.size() / 4U * 3U);
    for (std::size_t offset = 0U; offset < encoded.size(); offset += 4U) {
        std::array<unsigned int, 4> values{};
        unsigned int padding = 0U;
        for (std::size_t index = 0U; index < 4U; ++index) {
            const char value = encoded[offset + index];
            if (value == '=') {
                if (index < 2U || offset + 4U != encoded.size()) return false;
                values[index] = 0U;
                ++padding;
            } else {
                const auto found = alphabet.find(value);
                if (found == std::string_view::npos || padding != 0U) return false;
                values[index] = static_cast<unsigned int>(found);
            }
        }
        if (padding > 2U) return false;
        const auto bits = (values[0] << 18U) | (values[1] << 12U) |
                          (values[2] << 6U) | values[3];
        decoded.push_back(static_cast<char>((bits >> 16U) & 0xffU));
        if (padding < 2U) decoded.push_back(static_cast<char>((bits >> 8U) & 0xffU));
        if (padding == 0U) decoded.push_back(static_cast<char>(bits & 0xffU));
    }
    return true;
}

bool parse_image_url(detail::JsonCursor& cursor, std::string& url) {
    if (cursor.peek() == '"') {
        url = cursor.parse_string();
        return true;
    }
    cursor.expect('{');
    while (!cursor.consume('}')) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        if (key == "url") url = cursor.parse_string();
        else cursor.skip_value();
        if (cursor.consume('}')) break;
        cursor.expect(',');
    }
    return !url.empty();
}

bool parse_content_part(detail::JsonCursor& cursor, ChatMessage& message,
                        std::string& error) {
    cursor.expect('{');
    std::string type;
    std::string text;
    std::string image_url;
    while (!cursor.consume('}')) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        if (key == "type") type = cursor.parse_string();
        else if (key == "text") text = cursor.parse_string();
        else if (key == "image_url") {
            if (!parse_image_url(cursor, image_url)) {
                error = "image_url requires a URL";
                return false;
            }
        }
        else cursor.skip_value();
        if (cursor.consume('}')) break;
        cursor.expect(',');
    }
    if (type == "text") {
        message.content += text;
        message.parts.push_back({ChatContentKind::Text, std::move(text), {}});
        return true;
    }
    if (type != "image_url") {
        error = "unsupported message content part";
        return false;
    }
    constexpr std::string_view prefix = "data:";
    const auto separator = image_url.find(";base64,");
    if (!image_url.starts_with(prefix) || separator == std::string::npos) {
        error = "image_url must be a base64 data URL";
        return false;
    }
    const auto mime = std::string_view(image_url).substr(
        prefix.size(), separator - prefix.size());
    if (mime != "image/png" && mime != "image/jpeg" && mime != "image/webp") {
        error = "image_url MIME type must be PNG, JPEG, or WebP";
        return false;
    }
    std::string decoded;
    if (!decode_base64(std::string_view(image_url).substr(separator + 8U), decoded)) {
        error = "image_url contains invalid base64";
        return false;
    }
    message.parts.push_back(
        {ChatContentKind::Image, std::move(decoded), std::string(mime)});
    return true;
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
                    if (!parse_content_part(cursor, message, error)) return false;
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

// Sampler knobs beyond the OpenAI schema. Shared by the chat and completion
// parsers so a knob cannot reach one endpoint and not the other. `handled`
// reports whether the key belonged here at all.
bool parse_sampler_extension(std::string_view key, detail::JsonCursor& cursor,
                             SamplingOptions& sampling, bool& handled,
                             std::string& error) {
    handled = true;
    const auto bounded_u32 = [&cursor, &error](std::uint32_t& target,
                                               std::uint64_t limit,
                                               const char* name) {
        const auto value = cursor.parse_uint64();
        if (value > limit) {
            error = std::string(name) + " is out of range";
            return false;
        }
        target = static_cast<std::uint32_t>(value);
        return true;
    };
    if (key == "top_k") return bounded_u32(sampling.top_k, 1U << 20U, "top_k");
    if (key == "min_p") sampling.min_p = cursor.parse_number();
    else if (key == "typical_p") sampling.typical_p = cursor.parse_number();
    else if (key == "xtc_probability") sampling.xtc_probability = cursor.parse_number();
    else if (key == "xtc_threshold") sampling.xtc_threshold = cursor.parse_number();
    else if (key == "repetition_penalty") {
        sampling.repetition_penalty = cursor.parse_number();
    } else if (key == "penalty_window") {
        return bounded_u32(sampling.penalty_window, 1U << 24U, "penalty_window");
    } else if (key == "dry_multiplier") sampling.dry_multiplier = cursor.parse_number();
    else if (key == "dry_base") sampling.dry_base = cursor.parse_number();
    else if (key == "dry_allowed_length") {
        return bounded_u32(sampling.dry_allowed_length, 1U << 16U, "dry_allowed_length");
    } else if (key == "dry_window") {
        return bounded_u32(sampling.dry_window, 1U << 24U, "dry_window");
    } else if (key == "no_repeat_ngram") {
        return bounded_u32(sampling.no_repeat_ngram, 1U << 16U, "no_repeat_ngram");
    } else if (key == "future_entropy_candidates") {
        return bounded_u32(sampling.future_entropy_candidates, 64U,
                           "future_entropy_candidates");
    } else if (key == "future_entropy_top_n") {
        return bounded_u32(sampling.future_entropy_top_n, 1U << 20U,
                           "future_entropy_top_n");
    } else if (key == "alpha") sampling.future_entropy_alpha = cursor.parse_number();
    else if (key == "future_entropy_curve") {
        const auto curve = cursor.parse_string();
        if (curve == "article") {
            sampling.future_entropy_curve = FutureEntropyCurve::Article;
        } else if (curve == "crossfade") {
            sampling.future_entropy_curve = FutureEntropyCurve::Crossfade;
        } else {
            error = "future_entropy_curve is not a known curve";
            return false;
        }
    } else if (key == "alpha_wave_amplitude") {
        sampling.future_entropy_wave_amplitude = cursor.parse_number();
    } else if (key == "alpha_wave_period") {
        sampling.future_entropy_wave_period = cursor.parse_number();
    } else {
        handled = false;
    }
    return true;
}

// The HTTP surface caps temperature at the OpenAI-documented 2.0; every other
// range comes from the sampler's own validator.
bool validate_request_sampling(const SamplingOptions& sampling, std::string& error) {
    if (sampling.temperature > 2.0) {
        error = "invalid generation parameter range";
        return false;
    }
    std::string field;
    if (!validate_sampling_options(sampling, field)) {
        error = "invalid generation parameter range: " + field;
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
                bool handled = false;
                if (!parse_sampler_extension(key, cursor,
                                             request.generation.sampling,
                                             handled, error)) {
                    return false;
                }
                if (!handled) cursor.skip_value();
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
    if (request.n == 0U || request.generation.maximum_new_tokens == 0U) {
        error = "invalid generation parameter range";
        return false;
    }
    if (!validate_request_sampling(request.generation.sampling, error)) return false;
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
                bool handled = false;
                if (!parse_sampler_extension(key, cursor,
                                             request.generation.sampling,
                                             handled, error)) {
                    return false;
                }
                if (!handled) cursor.skip_value();
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
        request.generation.maximum_new_tokens == 0U) {
        error = "invalid generation parameter range";
        return false;
    }
    if (!validate_request_sampling(request.generation.sampling, error)) return false;
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

std::string render_openai_usage(std::uint64_t prompt_tokens,
                                std::uint64_t completion_tokens,
                                std::uint64_t cached_tokens) {
    std::ostringstream output;
    output << "{\"prompt_tokens\":" << prompt_tokens
           << ",\"completion_tokens\":" << completion_tokens
           << ",\"total_tokens\":" << prompt_tokens + completion_tokens
           << ",\"prompt_tokens_details\":{\"cached_tokens\":"
           << cached_tokens << "}}";
    return output.str();
}

std::string render_openai_timings(const GenerationMetrics& metrics) {
    std::ostringstream output;
    const double prompt_per_token_ms = metrics.prefill_tokens == 0U
        ? 0.0 : metrics.prefill_seconds * 1000.0 / static_cast<double>(metrics.prefill_tokens);
    const double predicted_per_token_ms = metrics.decode_tokens == 0U
        ? 0.0 : metrics.decode_seconds * 1000.0 / static_cast<double>(metrics.decode_tokens);
    const double prompt_per_second = metrics.prefill_seconds > 0.0
        ? static_cast<double>(metrics.prefill_tokens) / metrics.prefill_seconds : 0.0;
    const double predicted_per_second = metrics.decode_seconds > 0.0
        ? static_cast<double>(metrics.decode_tokens) / metrics.decode_seconds : 0.0;
    output << "{\"prompt_n\":" << metrics.prefill_tokens
           << ",\"prompt_ms\":" << metrics.prefill_seconds * 1000.0
           << ",\"prompt_per_token_ms\":" << prompt_per_token_ms
           << ",\"prompt_per_second\":" << prompt_per_second
           << ",\"predicted_n\":" << metrics.decode_tokens
           << ",\"predicted_ms\":" << metrics.decode_seconds * 1000.0
           << ",\"predicted_per_token_ms\":" << predicted_per_token_ms
           << ",\"predicted_per_second\":" << predicted_per_second
           << ",\"cache_n\":" << metrics.reused_prompt_tokens.value_or(0U) << '}';
    return output.str();
}

}  // namespace strata
