#pragma once

#include "strata/engine/chat_protocol.hpp"
#include "strata/engine/sampling.hpp"
#include "strata/platform/result.hpp"
#include "strata/platform/types.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strata {

struct Glm53RuntimeConfig {
    std::vector<int> devices;
    double vram_cache_fraction{0.85};
    std::uint32_t maximum_context_tokens{2048U};
    double sampling_temperature{};
    std::uint64_t sampling_seed{33'377'335U};
    bool verbose{};
    bool load_progress{};
};

struct Glm53RunMetrics {
    std::uint64_t prompt_tokens{};
    std::uint64_t prefill_tokens{};
    std::uint64_t decode_tokens{};
    double prefill_seconds{};
    double decode_seconds{};
};

struct Glm53GenerationResult {
    std::string text;
    std::vector<std::uint32_t> prompt_token_ids;
    std::vector<std::uint32_t> generated_token_ids;
    std::vector<TokenLogprob> logprobs;
    Glm53RunMetrics metrics;
    std::vector<std::string> errors;
    bool stopped{};

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

class Glm53Runtime {
public:
    Glm53Runtime();
    ~Glm53Runtime();
    Glm53Runtime(Glm53Runtime&&) noexcept;
    Glm53Runtime& operator=(Glm53Runtime&&) noexcept;
    Glm53Runtime(const Glm53Runtime&) = delete;
    Glm53Runtime& operator=(const Glm53Runtime&) = delete;

    [[nodiscard]] ValidationResult initialize(
        const std::string& model_directory,
        const Glm53RuntimeConfig& config = {});
    [[nodiscard]] Glm53GenerationResult generate_chat_stream(
        std::span<const ChatMessage> messages,
        std::uint32_t maximum_new_tokens, const SamplingOptions& sampling,
        std::span<const std::string> stop,
        const TokenStreamCallback& on_token = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata
