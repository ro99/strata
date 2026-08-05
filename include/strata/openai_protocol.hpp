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

// Serialize the completion-metadata fragments llama-swap and other proxies
// read from the response body to populate their activity/metrics views. These
// mirror the llama.cpp `usage` and `timings` shapes so proxies that parse
// llama.cpp responses report the same columns for Strata.
[[nodiscard]] std::string render_openai_usage(std::uint64_t prompt_tokens,
                                              std::uint64_t completion_tokens,
                                              std::uint64_t cached_tokens);
[[nodiscard]] std::string render_openai_timings(
    const GenerationMetrics& metrics);

}  // namespace strata
