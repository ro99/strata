#pragma once

#include "strata/chat_protocol.hpp"
#include "strata/safetensors.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace strata {

enum class TokenizerContract : std::uint8_t {
    Glm52,
    DeepSeekV4,
};

[[nodiscard]] ParseResult<std::vector<std::string>> pretokenize(
    TokenizerContract contract, std::string_view text);

class ModelTokenizer {
public:
    ModelTokenizer();

    [[nodiscard]] static ParseResult<ModelTokenizer> load(const std::string& path);
    [[nodiscard]] ParseResult<std::vector<std::uint32_t>> encode(
        std::string_view text) const;
    [[nodiscard]] ParseResult<std::string> decode(
        std::span<const std::uint32_t> tokens) const;
    [[nodiscard]] ParseResult<std::string> decode_token(std::uint32_t token) const;

    [[nodiscard]] std::uint32_t vocabulary_size() const noexcept {
        return static_cast<std::uint32_t>(id_to_piece_.size());
    }
    [[nodiscard]] std::int32_t token_id(std::string_view piece) const noexcept;

private:
    struct AddedToken {
        std::string content;
        std::uint32_t id{};
    };

    [[nodiscard]] ParseResult<std::vector<std::uint32_t>> encode_plain_chunk(
        std::string_view text) const;
    [[nodiscard]] ParseResult<std::vector<std::uint32_t>> encode_piece(
        std::string_view bytes) const;

    std::unordered_map<std::string, std::uint32_t> vocabulary_;
    std::unordered_map<std::string, std::uint32_t> merge_ranks_;
    std::vector<std::string> id_to_piece_;
    std::vector<bool> added_id_;
    std::vector<AddedToken> added_tokens_;
    std::array<std::string, 256> byte_to_piece_;
    std::array<std::int16_t, 1024> codepoint_to_byte_{};
    bool ignore_merges_{};
    TokenizerContract contract_{TokenizerContract::Glm52};
};

[[nodiscard]] std::string render_glm52_user_prompt(
    std::string_view user_text, std::string_view reasoning_effort = "medium-high",
    bool enable_thinking = true);
[[nodiscard]] std::string render_glm52_chat_prompt(
    std::span<const ChatMessage> messages,
    std::string_view reasoning_effort = "medium-high",
    bool enable_thinking = true);
[[nodiscard]] std::string render_deepseek_v4_user_prompt(
    std::string_view user_text, bool enable_thinking = false);
[[nodiscard]] std::string render_deepseek_v4_chat_prompt(
    std::span<const ChatMessage> messages, bool enable_thinking = false);

}  // namespace strata
