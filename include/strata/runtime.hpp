#pragma once

// The application-facing facade.
//
// RuntimeModel, RuntimeConfig, GenerationResult and friends moved to
// model_executor.hpp in Phase 4 and are re-exported here, so every existing
// `#include "strata/runtime.hpp"` keeps working unchanged. They live a tier
// lower because the six models implement ModelExecutor against them: had they
// stayed here, every model would depend upward on the application tier, which
// is exactly the inversion check-symbols exists to catch.

#include "strata/model_executor.hpp"

#include <memory>
#include <string>
#include <string_view>

namespace strata {

// Placement inputs implied by a runtime configuration. Applications use this to
// dry-run a load without constructing a session.
[[nodiscard]] PlacementRequest placement_request_for(
    const std::string& model_directory, const RuntimeConfig& config);

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
    [[nodiscard]] GenerationResult generate_chat_stream(
        std::span<const ChatMessage> messages,
        const GenerationOptions& options,
        const TokenStreamCallback& on_token = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata
