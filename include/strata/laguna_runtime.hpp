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
    // Empty means every visible device. The old default named three GPUs
    // because the development box had three; on a one- or two-GPU machine it
    // silently claimed devices that were not there.
    std::vector<int> devices;
    double vram_cache_fraction{0.85};
    std::uint32_t maximum_context_tokens{2048U};
    double sampling_temperature{};
    std::uint64_t sampling_seed{33'377'335U};
    bool verbose{};
    bool load_progress{};
    bool enable_flash_attention{true};
    // Keep the exact BF16 decode KV ring on the attention layer's device. This
    // changes placement only: the host mirror remains authoritative for exact
    // prompt continuation and the numerical contract is unchanged.
    bool enable_device_resident_kv_decode{true};
    bool enable_incremental_kv_continuation{true};
    bool enable_thinking{true};
    // Records CUDA events around every activation upload, kernel, and download
    // so the per-phase cost model can be instantiated. Off by default: it adds
    // event records and elapsed-time queries to every matmul.
    bool detailed_cuda_timing{};
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
    // Host wall time inside the cache, split into the miss path (checkpoint
    // read plus device upload) and the matmul call itself, so a residual is
    // attributed rather than inferred by subtraction.
    std::vector<std::uint64_t> device_lock_nanoseconds;
    std::vector<std::uint64_t> device_stage_nanoseconds;
    std::vector<std::uint64_t> device_matmul_nanoseconds;
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
    // Sub-phases. The attention counters partition attention_nanoseconds and
    // the MoE counters are contained in moe_routed_nanoseconds; they exist to
    // separate transfer volume from serialization inside a phase.
    std::uint64_t attention_projection_nanoseconds{};
    std::uint64_t attention_rope_nanoseconds{};
    std::uint64_t attention_kv_stage_nanoseconds{};
    std::uint64_t attention_flash_nanoseconds{};
    std::uint64_t attention_output_nanoseconds{};
    std::uint64_t moe_gather_nanoseconds{};
    std::uint64_t moe_expert_nanoseconds{};
    std::uint64_t moe_expert_cache_nanoseconds{};
    std::uint64_t moe_accumulate_nanoseconds{};
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
    bool device_resident_kv_decode{};
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
