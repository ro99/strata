#pragma once

#include "strata/models/common/checkpoint.hpp"
#include "strata/engine/chat_protocol.hpp"
#include "strata/engine/sampling.hpp"
#include "strata/platform/types.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace strata {

struct Glm52RuntimeConfig {
    // Empty means every visible device. The old default named three GPUs
    // because the development box had three; on a one- or two-GPU machine it
    // silently claimed devices that were not there.
    std::vector<int> devices;
    double vram_cache_fraction{0.85};
    std::uint32_t maximum_context_tokens{256};
    bool verbose{true};
    bool load_progress{};
    bool diagnostic_trace{};
    bool detailed_timing{};
    bool enable_flash_attention{};
    bool enable_incremental_kv_continuation{true};
    bool host_cold_experts{};
    // Zero derives from the process's CPU affinity mask.
    std::uint32_t host_worker_threads{};
    double sampling_temperature{};
    std::uint64_t sampling_seed{33'377'335U};
    std::string route_trace_path;
    std::uint64_t request_id{};
};

struct Glm52CacheStats {
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

struct Glm52HostExpertStats {
    std::uint64_t experts{};
    std::uint64_t matvec_calls{};
    std::uint64_t weight_bytes{};
    std::uint64_t service_nanoseconds{};
    std::uint64_t mapping_sweeps{};
    std::uint64_t mapping_sweep_nanoseconds{};
};

struct Glm52GraphStats {
    std::uint64_t forward_tokens{};
    std::uint64_t embedding_nanoseconds{};
    std::uint64_t input_norm_nanoseconds{};
    std::uint64_t attention_nanoseconds{};
    std::uint64_t attention_residual_nanoseconds{};
    std::uint64_t post_attention_norm_nanoseconds{};
    std::uint64_t dense_mlp_nanoseconds{};
    std::uint64_t moe_nanoseconds{};
    std::uint64_t moe_router_nanoseconds{};
    std::uint64_t moe_prepare_nanoseconds{};
    std::uint64_t moe_routed_nanoseconds{};
    std::uint64_t moe_shared_nanoseconds{};
    std::uint64_t mlp_residual_nanoseconds{};
    std::uint64_t output_head_nanoseconds{};
    std::uint64_t future_entropy_nanoseconds{};
    std::uint64_t future_entropy_passes{};
};

struct Glm52PhaseMetrics {
    CheckpointReadStats checkpoint_reads;
    CudaBackendStats cuda;
    Glm52CacheStats cache;
    Glm52HostExpertStats host_experts;
    Glm52GraphStats graph;
    std::uint64_t host_aggregation_nanoseconds{};
};

struct Glm52RunMetrics {
    std::uint64_t prompt_tokens{};
    std::uint64_t prefill_tokens{};
    std::uint64_t reused_prompt_tokens{};
    std::uint64_t decode_tokens{};
    double prefill_seconds{};
    double decode_seconds{};
    CheckpointReadStats checkpoint_reads;
    CudaBackendStats cuda;
    Glm52CacheStats cache;
    Glm52HostExpertStats host_experts;
    Glm52GraphStats graph;
    Glm52PhaseMetrics prefill;
    Glm52PhaseMetrics decode;
    std::uint64_t rss_bytes{};
    std::vector<std::uint64_t> device_vram_used_bytes;
    bool detailed_timing{};
    bool flash_attention_enabled{};
    bool incremental_kv_continuation{};

    [[nodiscard]] double prefill_tokens_per_second() const noexcept {
        return prefill_seconds > 0.0
            ? static_cast<double>(prefill_tokens) / prefill_seconds : 0.0;
    }
    [[nodiscard]] double decode_tokens_per_second() const noexcept {
        return decode_seconds > 0.0 ? static_cast<double>(decode_tokens) / decode_seconds : 0.0;
    }
};

struct Glm52GenerationResult {
    std::string text;
    std::vector<std::uint32_t> prompt_token_ids;
    std::vector<std::uint32_t> generated_token_ids;
    std::vector<TokenLogprob> logprobs;
    Glm52RunMetrics metrics;
    std::vector<std::string> errors;
    bool stopped{};

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

class Glm52Runtime {
public:
    Glm52Runtime();
    ~Glm52Runtime();
    Glm52Runtime(Glm52Runtime&&) noexcept;
    Glm52Runtime& operator=(Glm52Runtime&&) noexcept;
    Glm52Runtime(const Glm52Runtime&) = delete;
    Glm52Runtime& operator=(const Glm52Runtime&) = delete;

    [[nodiscard]] ValidationResult initialize(
        const std::string& model_directory, const Glm52RuntimeConfig& config = {});
    [[nodiscard]] Glm52GenerationResult generate(
        std::string_view prompt, std::uint32_t maximum_new_tokens);
    [[nodiscard]] Glm52GenerationResult generate_stream(
        std::string_view prompt, std::uint32_t maximum_new_tokens,
        const TokenStreamCallback& on_token);
    [[nodiscard]] Glm52GenerationResult generate_chat_stream(
        std::span<const ChatMessage> messages,
        std::uint32_t maximum_new_tokens,
        const TokenStreamCallback& on_token = {});
    [[nodiscard]] Glm52GenerationResult generate_chat_stream(
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
