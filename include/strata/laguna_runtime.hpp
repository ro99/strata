#pragma once

#include "strata/checkpoint.hpp"
#include "strata/chat_protocol.hpp"
#include "strata/cuda_backend.hpp"
#include "strata/sampling.hpp"
#include "strata/types.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strata {

struct LagunaRuntimeConfig {
    std::vector<int> devices{0, 1, 2};
    double vram_cache_fraction{0.85};
    std::uint32_t maximum_context_tokens{2048U};
    double sampling_temperature{};
    std::uint64_t sampling_seed{33'377'335U};
    bool verbose{};
    bool load_progress{};
    bool enable_flash_attention{true};
    bool enable_incremental_kv_continuation{true};
    bool enable_thinking{true};
    std::string route_trace_path;
    std::uint64_t request_id{};
};

struct LagunaCacheStats {
    std::uint64_t hits{};
    std::uint64_t misses{};
    std::uint64_t evictions{};
    std::vector<std::uint64_t> used_bytes;
    std::vector<std::uint64_t> peak_bytes;
    std::vector<std::uint64_t> capacity_bytes;
    std::vector<std::uint64_t> pinned_resident_bytes;
    std::vector<std::uint64_t> evictable_expert_bytes;
    std::vector<std::uint64_t> device_hits;
    std::vector<std::uint64_t> device_misses;
    std::vector<std::uint64_t> device_evictions;
};

struct LagunaGraphStats {
    std::uint64_t forward_tokens{};
    std::uint64_t embedding_nanoseconds{};
    std::uint64_t attention_nanoseconds{};
    std::uint64_t dense_mlp_nanoseconds{};
    std::uint64_t moe_router_nanoseconds{};
    std::uint64_t moe_routed_nanoseconds{};
    std::uint64_t moe_shared_nanoseconds{};
    std::uint64_t output_head_nanoseconds{};
};

struct LagunaPhaseMetrics {
    CheckpointReadStats checkpoint_reads;
    CudaBackendStats cuda;
    LagunaCacheStats cache;
    LagunaGraphStats graph;
};

struct LagunaRunMetrics {
    std::uint64_t prompt_tokens{};
    std::uint64_t prefill_tokens{};
    std::uint64_t reused_prompt_tokens{};
    std::uint64_t decode_tokens{};
    double prefill_seconds{};
    double decode_seconds{};
    CheckpointReadStats checkpoint_reads;
    CudaBackendStats cuda;
    LagunaCacheStats cache;
    LagunaGraphStats graph;
    LagunaPhaseMetrics prefill;
    LagunaPhaseMetrics decode;
    std::uint64_t rss_bytes{};
    std::vector<std::uint64_t> device_vram_used_bytes;
    bool flash_attention_enabled{};
    bool incremental_kv_continuation{};

    [[nodiscard]] double prefill_tokens_per_second() const noexcept {
        return prefill_seconds > 0.0
            ? static_cast<double>(prefill_tokens) / prefill_seconds : 0.0;
    }
    [[nodiscard]] double decode_tokens_per_second() const noexcept {
        return decode_seconds > 0.0
            ? static_cast<double>(decode_tokens) / decode_seconds : 0.0;
    }
};

struct LagunaGenerationResult {
    std::string text;
    std::vector<std::uint32_t> prompt_token_ids;
    std::vector<std::uint32_t> generated_token_ids;
    std::vector<TokenLogprob> logprobs;
    LagunaRunMetrics metrics;
    std::vector<std::string> errors;
    bool stopped{};

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

class LagunaRuntime {
public:
    LagunaRuntime();
    ~LagunaRuntime();
    LagunaRuntime(LagunaRuntime&&) noexcept;
    LagunaRuntime& operator=(LagunaRuntime&&) noexcept;
    LagunaRuntime(const LagunaRuntime&) = delete;
    LagunaRuntime& operator=(const LagunaRuntime&) = delete;

    [[nodiscard]] ValidationResult initialize(
        const std::string& model_directory,
        const LagunaRuntimeConfig& config = {});
    [[nodiscard]] LagunaGenerationResult generate_stream(
        std::string_view prompt, std::uint32_t maximum_new_tokens,
        const TokenStreamCallback& on_token = {});
    [[nodiscard]] LagunaGenerationResult generate_chat_stream(
        std::span<const ChatMessage> messages,
        std::uint32_t maximum_new_tokens,
        const SamplingOptions& sampling,
        std::span<const std::string> stop,
        const TokenStreamCallback& on_token = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata
