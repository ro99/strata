#pragma once

#include "strata/chat_protocol.hpp"
#include "strata/kimi_k3_expert_arena.hpp"
#include "strata/placement.hpp"
#include "strata/sampling.hpp"
#include "strata/types.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strata {

struct KimiK3RuntimeConfig {
    std::uint32_t maximum_context_tokens{2048U};
    // Tokens per prefill page. A page amortizes every dense weight read across
    // its rows, but the routed set does not amortize: past roughly 256 tokens a
    // page touches essentially all 896 experts of every layer, so the page cost
    // stops falling. See `docs/experiments/0048`.
    std::uint32_t prefill_page_tokens{64U};
    // Host bytes for the routed-expert cache. Zero derives it from what is left
    // after the dense spine, against `host_budget_bytes`.
    std::uint64_t expert_arena_bytes{};
    std::uint64_t host_budget_bytes{};
    std::uint32_t expert_queue_depth{4U};
    bool direct_expert_reads{true};
    bool lock_expert_arena{true};
    // Disks that may not receive a byte derived from model weights. The guard
    // refuses to run rather than degrading quietly, because an unlocked arena
    // or an enabled swap file reintroduces the write path silently.
    std::vector<std::string> forbidden_disks{"nvme0n1", "sdb"};
    bool apply_write_guard{true};
    std::size_t host_workers{};  // zero uses the hardware concurrency
    double sampling_temperature{};
    std::uint64_t sampling_seed{33'377'335U};
    bool verbose{};
    bool load_progress{};
    // Optional pre-solved placement, borrowed for the duration of initialize.
    const PlacementPlan* placement{};
};

struct KimiK3RunMetrics {
    std::uint64_t prompt_tokens{};
    std::uint64_t prefill_tokens{};
    std::uint64_t decode_tokens{};
    double load_seconds{};
    double prefill_seconds{};
    double decode_seconds{};
    std::uint64_t resident_weight_bytes{};
    std::uint64_t expert_arena_bytes{};
    std::uint64_t expert_hits{};
    std::uint64_t expert_misses{};
    std::uint64_t expert_evictions{};
    std::uint64_t storage_bytes_read{};
    std::uint64_t rss_bytes{};
    bool expert_arena_locked{};
};

struct KimiK3GenerationResult {
    std::string text;
    std::vector<std::uint32_t> prompt_token_ids;
    std::vector<std::uint32_t> generated_token_ids;
    std::vector<TokenLogprob> logprobs;
    KimiK3RunMetrics metrics;
    std::vector<std::string> errors;
    bool stopped{};

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

class KimiK3Runtime {
public:
    KimiK3Runtime();
    ~KimiK3Runtime();
    KimiK3Runtime(KimiK3Runtime&&) noexcept;
    KimiK3Runtime& operator=(KimiK3Runtime&&) noexcept;
    KimiK3Runtime(const KimiK3Runtime&) = delete;
    KimiK3Runtime& operator=(const KimiK3Runtime&) = delete;

    [[nodiscard]] ValidationResult initialize(
        const std::string& model_directory,
        const KimiK3RuntimeConfig& config = {});

    // Clears the per-sequence state. The KDA half is recurrent and cannot be
    // rewound, so a new sequence resets rather than truncates.
    [[nodiscard]] ValidationResult reset_sequence();

    // Runs `tokens` through the backbone at `position` and writes the final
    // token's logits. This is the surface the teacher-forcing and generation
    // oracles drive: it needs no tokenizer, so gates 5 and 6 do not wait on one.
    [[nodiscard]] ValidationResult evaluate(
        std::span<const std::uint32_t> tokens, std::uint32_t position,
        std::span<float> logits);

    // Greedy or sampled continuation from token ids.
    [[nodiscard]] KimiK3GenerationResult generate_from_tokens(
        std::span<const std::uint32_t> prompt, std::uint32_t maximum_new_tokens,
        const SamplingOptions& sampling,
        const TokenStreamCallback& on_token = {});

    [[nodiscard]] KimiK3GenerationResult generate_stream(
        std::string_view prompt, std::uint32_t maximum_new_tokens,
        const TokenStreamCallback& on_token = {});
    [[nodiscard]] KimiK3GenerationResult generate_chat_stream(
        std::span<const ChatMessage> messages, std::uint32_t maximum_new_tokens,
        const SamplingOptions& sampling, std::span<const std::string> stop,
        const TokenStreamCallback& on_token = {});

    [[nodiscard]] const KimiK3RunMetrics& metrics() const noexcept;
    [[nodiscard]] std::uint32_t vocabulary_size() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata
