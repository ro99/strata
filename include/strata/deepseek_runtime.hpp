#pragma once

#include "strata/chat_protocol.hpp"
#include "strata/cuda_backend.hpp"
#include "strata/deepseek_admission.hpp"
#include "strata/deepseek_checkpoint.hpp"
#include "strata/deepseek_diagnostics.hpp"
#include "strata/deepseek_kv_cache.hpp"
#include "strata/types.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace strata {

struct Dsv4RuntimeConfig {
    std::vector<int> devices{0};
    double vram_cache_fraction{0.85};
    std::uint64_t host_memory_limit_bytes{216ULL << 30U};
    std::uint64_t host_kv_cache_bytes{};
    std::vector<std::uint64_t> device_kv_cache_bytes;
    std::uint32_t maximum_context_tokens{2048U};
    Dsv4KvCacheMode kv_cache_mode{Dsv4KvCacheMode::ScalarOracle};
    std::uint32_t kv_block_rows{64U};
    // Prefill is executed in bounded layer-major pages. Page 64 is the
    // accepted measured default; a value of one retains the oracle traversal.
    std::uint32_t prefill_page_tokens{64U};
    std::uint32_t logit_trace_top_k{20U};
    std::uint32_t host_attention_threads{28U};
    bool enable_flash_attention{};
    bool enable_gpu_lightning_indexer{};
    bool enable_incremental_kv_continuation{true};
    // CUDA offload has a fixed launch/staging cost. The production default
    // retains parallel host attention below the measured row crossover; zero
    // forces every supported shape through CUDA for diagnostics.
    std::uint32_t flash_attention_minimum_rows{256U};
    std::uint32_t resident_read_workers{8U};
    std::uint32_t spine_warmup_workers{3U};
    double sampling_temperature{};
    std::uint64_t sampling_seed{33'377'335U};
    bool require_zero_nvme_decode{true};
    bool enable_dspark{};
    bool enable_device_moe{true};
    bool enable_logit_trace{};
    bool enable_layer_hash_trace{};
    bool detailed_timing{};
    bool overlap_resident_warmup{true};
    bool verbose{};
    std::string route_trace_path;
};

struct Dsv4CacheStats {
    std::uint64_t hits{};
    std::uint64_t misses{};
    std::uint64_t evictions{};
    std::uint64_t lease_acquires{};
    std::uint64_t lease_releases{};
    std::vector<std::uint64_t> used_bytes;
    std::vector<std::uint64_t> capacity_bytes;
    std::vector<std::uint64_t> pinned_bytes;
    std::vector<std::uint64_t> leased_bytes;
    std::vector<std::uint64_t> active_leases;
};

struct Dsv4DeviceMoeStats {
    std::uint64_t batches{};
    std::uint64_t device_commands{};
    std::uint64_t routed_experts{};
    std::uint64_t shared_experts{};
    std::uint64_t nanoseconds{};
};

struct Dsv4GraphStats {
    std::uint64_t forward_tokens{};
    std::uint64_t prefill_pages{};
    std::uint64_t prefill_max_page_tokens{};
    std::uint64_t prefill_max_workspace_bytes{};
    std::uint64_t embedding_nanoseconds{};
    std::uint64_t mhc_pre_nanoseconds{};
    std::uint64_t branch_norm_nanoseconds{};
    std::uint64_t attention_nanoseconds{};
    std::uint64_t attention_query_nanoseconds{};
    std::uint64_t attention_kv_nanoseconds{};
    std::uint64_t attention_projection_matmul_calls{};
    std::uint64_t attention_projection_matmul_rows{};
    std::uint64_t attention_index_nanoseconds{};
    std::uint64_t attention_index_queries{};
    std::uint64_t attention_index_candidates{};
    std::uint64_t attention_index_selected{};
    std::uint64_t attention_index_cuda_dispatches{};
    std::uint64_t attention_index_scalar_dispatches{};
    std::uint64_t attention_cuda_dispatches{};
    std::uint64_t attention_scalar_dispatches{};
    std::uint64_t attention_score_nanoseconds{};
    std::uint64_t attention_output_nanoseconds{};
    std::uint64_t moe_nanoseconds{};
    std::uint64_t moe_router_nanoseconds{};
    std::uint64_t moe_prepare_nanoseconds{};
    std::uint64_t mhc_post_nanoseconds{};
    std::uint64_t output_head_nanoseconds{};
};

struct Dsv4PhaseMetrics {
    Dsv4CheckpointReadStats checkpoint_reads;
    CudaBackendStats cuda;
    Dsv4CacheStats cache;
    Dsv4KvCacheStats kv_cache;
    Dsv4DeviceMoeStats device_moe;
    Dsv4GraphStats graph;
};

struct Dsv4GenerationMetrics {
    double initialization_seconds{};
    double admission_seconds{};
    double resident_staging_seconds{};
    double resident_warmup_seconds{};
    double prefill_seconds{};
    double decode_seconds{};
    std::uint64_t prompt_tokens{};
    std::uint64_t prefill_tokens{};
    std::uint64_t reused_prompt_tokens{};
    std::uint64_t decode_tokens{};
    std::uint64_t rss_bytes{};
    std::vector<std::uint64_t> device_vram_used_bytes;
    Dsv4MemoryPlan memory;
    Dsv4ResidentStageStats resident_stage;
    Dsv4CheckpointReadStats generation_checkpoint_reads;
    Dsv4CheckpointReadStats decode_checkpoint_reads;
    CudaBackendStats cuda;
    Dsv4CacheStats cache;
    Dsv4KvCacheStats kv_cache;
    Dsv4DeviceMoeStats device_moe;
    Dsv4GraphStats graph;
    Dsv4PhaseMetrics prefill;
    Dsv4PhaseMetrics decode;
    bool detailed_timing{};
    bool dspark_enabled{};
    bool device_moe_enabled{};
    bool resident_warmup_overlapped{};
    bool block_kv_cache_enabled{};
    bool incremental_kv_continuation{};
    std::uint32_t kv_block_rows{};
    std::uint32_t host_attention_threads{};
    std::uint32_t prefill_page_tokens{};
    bool flash_attention_enabled{};
    bool gpu_lightning_indexer_enabled{};
    std::uint32_t flash_attention_minimum_rows{};
    std::uint32_t resident_read_workers{};
    std::uint32_t spine_warmup_workers{};
};

struct Dsv4GenerationResult {
    std::string text;
    std::vector<std::uint32_t> prompt_token_ids;
    std::vector<std::uint32_t> generated_token_ids;
    Dsv4GenerationMetrics metrics;
    Dsv4DiagnosticTrace diagnostics;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

class DeepSeekV4Runtime {
public:
    DeepSeekV4Runtime();
    ~DeepSeekV4Runtime();
    DeepSeekV4Runtime(DeepSeekV4Runtime&&) noexcept;
    DeepSeekV4Runtime& operator=(DeepSeekV4Runtime&&) noexcept;
    DeepSeekV4Runtime(const DeepSeekV4Runtime&) = delete;
    DeepSeekV4Runtime& operator=(const DeepSeekV4Runtime&) = delete;

    [[nodiscard]] ValidationResult initialize(
        const std::string& model_directory, const Dsv4RuntimeConfig& config);
    [[nodiscard]] Dsv4GenerationResult generate(
        std::string_view prompt, std::uint32_t maximum_new_tokens);
    [[nodiscard]] Dsv4GenerationResult generate_stream(
        std::string_view prompt, std::uint32_t maximum_new_tokens,
        const TokenStreamCallback& on_token);
    [[nodiscard]] Dsv4GenerationResult generate_chat_stream(
        std::span<const ChatMessage> messages,
        std::uint32_t maximum_new_tokens,
        const TokenStreamCallback& on_token = {});
    [[nodiscard]] const Dsv4MemoryPlan& memory_plan() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata
