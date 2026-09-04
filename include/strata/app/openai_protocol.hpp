#pragma once

#include "strata/app/runtime.hpp"

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
    bool has_temperature{};
    bool has_top_p{};
    bool has_seed{};
    // Whether the separated reasoning is returned to the client. False still
    // generates it -- a model whose template always reasons cannot be told not
    // to -- and only withholds it from the response.
    bool include_reasoning{true};
    std::string user;
};

// Splits a model's generated text into reasoning and answer while it streams.
//
// The delimiters arrive as ordinary text and a token boundary falls wherever
// the tokenizer put it, so a closing tag routinely straddles two pieces. Any
// suffix that could still become a delimiter is therefore held back rather than
// emitted, which is why a caller must call `finish` -- a stream that ended
// mid-delimiter has to release those bytes rather than drop them.
//
// A format that declares no closing tag passes everything through as content,
// so an unannotated model is unaffected. Scanning stops once the block closes:
// GLM-5.3 emits exactly one, and a model's answer is allowed to contain the
// literal text of a tag.
class ReasoningSplitter {
public:
    struct Delta {
        std::string reasoning;
        std::string content;

        [[nodiscard]] bool empty() const noexcept {
            return reasoning.empty() && content.empty();
        }
    };

    explicit ReasoningSplitter(const ReasoningFormat& format) noexcept;

    [[nodiscard]] Delta consume(std::string_view piece);
    [[nodiscard]] Delta finish();
    [[nodiscard]] bool reasoning_open() const noexcept { return reasoning_; }

private:
    std::string_view open_;
    std::string_view close_;
    std::string pending_;
    bool reasoning_{};
    bool done_{};
};

// Whole-text convenience for the non-streaming path: equivalent to one
// `consume` followed by `finish`.
[[nodiscard]] ReasoningSplitter::Delta split_reasoning(
    const ReasoningFormat& format, std::string_view text);

[[nodiscard]] bool parse_openai_chat_request(
    std::string_view json, OpenAiChatRequest& request, std::string& error);
[[nodiscard]] bool parse_openai_completion_request(
    std::string_view json, OpenAiChatRequest& request, std::string& error);
[[nodiscard]] bool parse_openai_tokenize_request(
    std::string_view json, std::string& model, std::string& text,
    std::string& error);
// Extract only the required top-level model string while preserving the
// original request body for a router to proxy byte-for-byte.
[[nodiscard]] bool parse_openai_model_field(
    std::string_view json, std::string& model, std::string& error);
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
