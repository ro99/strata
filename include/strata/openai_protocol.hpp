#pragma once

#include "strata/runtime.hpp"

#include <cstdint>
#include <string>

namespace strata {

struct OpenAiChatRequest {
    std::string model;
    std::vector<ChatMessage> messages;
    GenerationOptions generation;
    std::uint32_t n{1U};
    bool stream{};
    bool logprobs{};
    bool json_object{};
    bool has_tools{};
    bool has_max_tokens{};
    bool has_seed{};
    std::string user;
};

[[nodiscard]] bool parse_openai_chat_request(
    std::string_view json, OpenAiChatRequest& request, std::string& error);
[[nodiscard]] bool parse_openai_completion_request(
    std::string_view json, OpenAiChatRequest& request, std::string& error);
[[nodiscard]] bool parse_openai_tokenize_request(
    std::string_view json, std::string& model, std::string& text,
    std::string& error);
[[nodiscard]] bool is_json_object(std::string_view json) noexcept;

}  // namespace strata
