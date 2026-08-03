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
    Gemma4,
    KimiK3,
};

[[nodiscard]] ParseResult<std::vector<std::string>> pretokenize(
    TokenizerContract contract, std::string_view text);

class ModelTokenizer {
public:
    ModelTokenizer();

    [[nodiscard]] static ParseResult<ModelTokenizer> load(const std::string& path);
    // Kimi-K3 ships a tiktoken vocabulary, not a `tokenizer.json`: one
    // base64 piece and rank per line, with the reserved special ids named by
    // `tokenizer_config.json`. Its BPE is also a different algorithm — merge
    // the adjacent pair of lowest vocabulary rank, with no merge table — so it
    // gets its own loader rather than a flag on the JSON one.
    [[nodiscard]] static ParseResult<ModelTokenizer> load_kimi_k3(
        const std::string& model_directory);
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
    bool sentencepiece_bpe_{};
    bool tiktoken_bpe_{};
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
[[nodiscard]] std::string render_gemma4_user_prompt(
    std::string_view user_text, bool enable_thinking = false);
[[nodiscard]] std::string render_gemma4_chat_prompt(
    std::span<const ChatMessage> messages, bool enable_thinking = false);
// Kimi-K3 renders XTML rather than header-delimited turns: every message is an
// element opened with `<|open|>`, its attributes as `key="value"` text, `<|sep|>`
// before the body, and `<|close|>message<|sep|>` plus `<|end_of_msg|>` after it.
[[nodiscard]] std::string render_kimi_k3_user_prompt(
    std::string_view user_text, bool enable_thinking = true);
[[nodiscard]] std::string render_kimi_k3_chat_prompt(
    std::span<const ChatMessage> messages, bool enable_thinking = true);

}  // namespace strata
