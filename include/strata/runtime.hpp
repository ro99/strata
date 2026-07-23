#pragma once

#include "strata/chat_protocol.hpp"
#include "strata/result.hpp"
#include "strata/types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace strata {

enum class RuntimeModel : std::uint8_t {
    Glm52,
    DeepSeekV4,
};

struct RuntimeConfig {
    RuntimeModel model{RuntimeModel::Glm52};
    std::vector<int> devices{0, 1, 2};
    double vram_cache_fraction{0.85};
    std::uint32_t maximum_context_tokens{2048U};
    double sampling_temperature{};
    std::uint64_t sampling_seed{33'377'335U};
    bool verbose{};
    bool load_progress{};
    bool enable_flash_attention{};
    bool enable_incremental_kv_continuation{true};
    bool deepseek_block_kv_cache{};
};

struct GenerationMetrics {
    std::uint64_t prompt_tokens{};
    std::uint64_t prefill_tokens{};
    std::uint64_t reused_prompt_tokens{};
    std::uint64_t decode_tokens{};
    double prefill_seconds{};
    double decode_seconds{};
    bool incremental_kv_continuation{};
};

struct GenerationResult {
    std::string text;
    std::vector<std::uint32_t> prompt_token_ids;
    std::vector<std::uint32_t> generated_token_ids;
    GenerationMetrics metrics;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

// Stable application-facing facade. Architecture-specific diagnostics and
// research controls remain available through the concrete runtimes.
class RuntimeSession {
public:
    RuntimeSession();
    ~RuntimeSession();
    RuntimeSession(RuntimeSession&&) noexcept;
    RuntimeSession& operator=(RuntimeSession&&) noexcept;
    RuntimeSession(const RuntimeSession&) = delete;
    RuntimeSession& operator=(const RuntimeSession&) = delete;

    [[nodiscard]] ValidationResult initialize(
        const std::string& model_directory, const RuntimeConfig& config);
    [[nodiscard]] GenerationResult generate_stream(
        std::string_view prompt, std::uint32_t maximum_new_tokens,
        const TokenStreamCallback& on_token = {});
    [[nodiscard]] GenerationResult generate_chat_stream(
        std::span<const ChatMessage> messages,
        std::uint32_t maximum_new_tokens,
        const TokenStreamCallback& on_token = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata
