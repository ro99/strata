#pragma once

#include "strata/chat_protocol.hpp"
#include "strata/cuda_backend.hpp"
#include "strata/deepseek_admission.hpp"
#include "strata/deepseek_checkpoint.hpp"
#include "strata/deepseek_diagnostics.hpp"
#include "strata/deepseek_kv_cache.hpp"
#include "strata/dsv4_rank_local_topology.hpp"
#include "strata/sampling.hpp"
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
    // Decode execution topology. Centralized is the default and is unchanged
    // by the rank-local feature. RankLocalTp2 is explicit opt-in, is admitted
    // fail-closed before model loading, and never falls back once admitted.
    Dsv4DecodeTopology decode_topology{Dsv4DecodeTopology::Centralized};
    double vram_cache_fraction{0.85};
    // Optional hard per-device admission ceiling. Zero preserves the
    // fractional-only contract; non-zero is combined with it as min().
    std::uint64_t explicit_vram_budget_bytes{};
    std::uint64_t host_memory_limit_bytes{216ULL << 30U};
    std::uint64_t host_kv_cache_bytes{};
    std::vector<std::uint64_t> device_kv_cache_bytes;
    std::uint32_t maximum_context_tokens{2048U};
    Dsv4KvCacheMode kv_cache_mode{Dsv4KvCacheMode::ScalarOracle};
    std::uint32_t kv_block_rows{64U};
    // Prefill is executed in bounded layer-major pages. Page 64 is the
    // accepted measured default; a value of one retains the oracle traversal.
    std::uint32_t prefill_page_tokens{64U};
    // Prefill visits layers outermost over a tile of this many tokens, so a
    // layer's routed experts are streamed once per tile rather than once per
    // page. Zero tiles the whole prefill range, which is the minimum possible
    // expert traffic; setting it equal to prefill_page_tokens restores the
    // page-major nest. Costs tile_tokens * mhc_multiplier * hidden_size * 4
    // bytes of resident activation.
    std::uint32_t prefill_layer_tile_tokens{};
    std::uint32_t logit_trace_top_k{20U};
    std::uint32_t host_attention_threads{28U};
    bool enable_flash_attention{};
    bool enable_gpu_lightning_indexer{};
    bool enable_incremental_kv_continuation{true};
    // Zero sends every supported shape to CUDA, the measured production
    // crossover. Keep the knob because launch/staging costs are hardware- and
    // context-dependent.
    std::uint32_t flash_attention_minimum_rows{};
    // Page-lock the resident weight arena after staging. Every routed expert
    // is a cold slice of a 147 GB mapping, so the driver's pageable staging
    // copy dominates the transfer: measured 1.32 ms pageable against 0.37 ms
    // pinned for one 4.46 MB projection. Costs about 2.7 GB/s of one-time
    // registration at load and locks the arena's pages for the process.
    bool pin_resident_arena{};
    // Waits out each routed-expert demand upload where it is issued, instead of
    // letting a layer's uploads to different devices run concurrently and
    // waiting once per device. This is the pre-0052 behaviour and is a
    // diagnostic and rollback switch only: it drives three independent PCIe
    // links one at a time, which measured 105.4 ms against 55.4 ms for the same
    // bytes. Nothing about the result changes -- the transfers are identical
    // and so is every byte of output.
    bool serial_expert_upload{};
    // Prefill pages execute their MoE expert-major by default: each distinct
    // expert of the page is acquired and read once and applied to every row
    // that selected it. Setting this restores the row-major nest, which reads
    // one expert triplet per row, and exists so the two can be A/B'd from one
    // build. Per-row arithmetic is identical either way.
    bool row_major_moe_page{};
    bool prepack_mhc_projection{true};
    std::uint32_t resident_read_workers{8U};
    std::uint32_t spine_warmup_workers{3U};
    // Zero predictions disables advisory expert prefetch. The remaining
    // defaults are bounded so one CLI switch is sufficient to enable it.
    std::uint32_t expert_prefetch_predictions{};
    std::uint32_t expert_prefetch_queue_depth{8U};
    std::uint64_t expert_prefetch_byte_budget{1ULL << 30U};
    std::uint64_t expert_prefetch_lease_ticks{16U};
    double expert_prefetch_minimum_confidence{0.75};
    double sampling_temperature{};
    std::uint64_t sampling_seed{33'377'335U};
    bool require_zero_nvme_decode{true};
    bool enable_dspark{};
    bool enable_device_moe{true};
    // Reproduces the device-resident decode placement: routed experts use two
    // output-tiled NUMA-local CPU shards while the shared expert stays on GPU.
    // Prefill remains on the existing grouped device path.
    bool enable_host_routed_moe{};
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
    std::uint64_t demand_h2d_bytes{};
    std::uint64_t demand_wait_nanoseconds{};
    std::uint64_t prefetch_requests{};
    std::uint64_t prefetch_h2d_bytes{};
    std::uint64_t useful_prefetch_bytes{};
    std::uint64_t late_prefetch_bytes{};
    std::uint64_t duplicate_prefetch_bytes{};
    std::uint64_t evicted_prefetch_bytes{};
    std::uint64_t wasted_prefetch_bytes{};
    std::uint64_t cancelled_prefetch_bytes{};
    std::uint64_t prefetch_lease_acquires{};
    std::uint64_t prefetch_lease_releases{};
    std::uint64_t active_prefetch_leases{};
    std::uint64_t prefetch_queue_peak{};
    std::vector<std::uint64_t> used_bytes;
    std::vector<std::uint64_t> capacity_bytes;
    std::vector<std::uint64_t> pinned_bytes;
    std::vector<std::uint64_t> leased_bytes;
    std::vector<std::uint64_t> active_leases;
};

struct Dsv4DeviceMoeStats {
    std::uint64_t batches{};
    std::uint64_t host_callback_batches{};
    std::uint64_t host_callback_failures{};
    std::uint64_t device_join_batches{};
    std::uint64_t device_commands{};
    std::uint64_t routed_experts{};
    std::uint64_t shared_experts{};
    std::uint64_t routed_gate_up_nanoseconds{};
    std::uint64_t routed_down_nanoseconds{};
    std::uint64_t routed_reduce_nanoseconds{};
    std::uint64_t routed_cpu_nanoseconds{};
    std::uint64_t shared_collect_nanoseconds{};
    std::uint64_t combine_nanoseconds{};
    std::uint64_t nanoseconds{};
};

struct Dsv4GraphStats {
    std::uint64_t forward_tokens{};
    std::uint64_t prefill_pages{};
    std::uint64_t prefill_max_page_tokens{};
    std::uint64_t prefill_max_workspace_bytes{};
    std::uint64_t embedding_nanoseconds{};
    std::uint64_t mhc_pre_nanoseconds{};
    std::uint64_t mhc_prepacked_calls{};
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
    // Rank-local decode attribution. The executor reports the device work it
    // performs; these account for the host-side glue around it, which is the
    // only place a rank-local step can be slower than the centralized step it
    // splits. Zero on the centralized path.
    //
    // `rank_local_layer_nanoseconds` is wall time inside run()/enqueue and
    // `rank_local_device_nanoseconds` is what the executor measured of that,
    // so their difference is submission plus the per-layer diagnostic boundary.
    std::uint64_t rank_local_layer_nanoseconds{};
    std::uint64_t rank_local_device_nanoseconds{};
    std::uint64_t rank_local_kv_nanoseconds{};
    std::uint64_t rank_local_candidate_nanoseconds{};
    std::uint64_t rank_local_boundary_nanoseconds{};
    std::uint64_t rank_local_collective_nanoseconds{};
    std::uint64_t rank_local_transition_nanoseconds{};
    std::uint64_t rank_local_shared_nanoseconds{};
    // Future-entropy lookahead, kept separate because it is whole speculative
    // forward passes rather than a phase of one. The per-phase counters above
    // include the work these passes did; this is how much of it was
    // speculative, and `future_entropy_passes` is how many passes that was.
    std::uint64_t future_entropy_nanoseconds{};
    std::uint64_t future_entropy_passes{};
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
    bool host_routed_moe_enabled{};
    bool resident_warmup_overlapped{};
    bool block_kv_cache_enabled{};
    bool incremental_kv_continuation{};
    std::uint32_t kv_block_rows{};
    std::uint32_t host_attention_threads{};
    std::uint32_t prefill_page_tokens{};
    std::uint32_t prefill_layer_tile_tokens{};
    bool flash_attention_enabled{};
    bool gpu_lightning_indexer_enabled{};
    std::uint32_t flash_attention_minimum_rows{};
    std::uint32_t resident_read_workers{};
    std::uint32_t spine_warmup_workers{};
    double resident_pin_seconds{};
    bool resident_arena_pinned{};
    std::uint32_t expert_prefetch_predictions{};
    std::uint32_t expert_prefetch_queue_depth{};
    std::uint64_t expert_prefetch_byte_budget{};
    std::uint64_t expert_prefetch_lease_ticks{};
    double expert_prefetch_minimum_confidence{};
};

struct Dsv4GenerationResult {
    std::string text;
    std::vector<std::uint32_t> prompt_token_ids;
    std::vector<std::uint32_t> generated_token_ids;
    std::vector<TokenLogprob> logprobs;
    Dsv4GenerationMetrics metrics;
    Dsv4DiagnosticTrace diagnostics;
    std::vector<std::string> errors;
    bool stopped{};

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
    [[nodiscard]] Dsv4GenerationResult generate_chat_stream(
        std::span<const ChatMessage> messages,
        std::uint32_t maximum_new_tokens,
        const SamplingOptions& sampling,
        std::span<const std::string> stop,
        const TokenStreamCallback& on_token = {});
    [[nodiscard]] const Dsv4MemoryPlan& memory_plan() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata
