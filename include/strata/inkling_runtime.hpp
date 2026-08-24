#pragma once

#include "strata/chat_protocol.hpp"
#include "strata/checkpoint.hpp"
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

struct InklingRuntimeConfig {
    // Empty means every visible device. The old default named three GPUs
    // because the development box had three; on a one- or two-GPU machine it
    // silently claimed devices that were not there.
    std::vector<int> devices;
    // Fraction of each device's free VRAM the runtime may claim. What the
    // resident spine does not use becomes routed-expert cache.
    double vram_cache_fraction{0.85};
    // Runs the experts and the resident spine on the devices. Off falls back
    // to the host reference path, which is the correctness oracle.
    bool enable_cuda{true};
    // Faults the routed expert set into page cache at load. Host memory is
    // larger than either supported checkpoint, so steady-state decode should
    // never touch NVMe; without this the VRAM cache misses fault cold pages and
    // the device waits on storage instead of on PCIe.
    bool warm_expert_pages{true};
    // MXFP4 experts already have the canonical three-stack layout, so upload
    // from the resident mapping without an extra host memcpy into pinned
    // scratch. False retains the measured control path for profiling. NVFP4
    // still needs scratch to de-interleave gate/up and is unaffected.
    bool direct_mapped_mxfp4_staging{true};
    // Keep the exact BF16 K/V ring on each layer's assigned device and run
    // Inkling's relative-bias attention there. False retains the scalar host
    // oracle for controlled comparisons.
    bool enable_device_kv_attention{true};
    // Keep short generations on the scalar oracle. Although the device
    // operation crosses over near 32 rows, a route-sensitive 64-token run
    // regressed after its numerically equivalent route changed. At 512 rows
    // the target-shape operation is already 35x faster on SM86, so this
    // conservative boundary preserves the short-context path while removing
    // the unbounded host-attention term at long context.
    std::uint32_t minimum_device_attention_rows{512U};
    std::uint32_t maximum_context_tokens{2048U};
    // Rows per prefill page. Zero or one keeps the token-at-a-time path.
    // Above one, a page runs attention and the short convolutions row by row
    // in order -- both carry row-ordered state -- and batches the routed MoE
    // between them expert-major, so each distinct expert of the page is
    // fetched once instead of once per row that selected it. Arithmetic is
    // identical either way; this is a scheduling change.
    // Opt-in until measured. Expert-major batching cuts expert stagings by
    // about 1.93x at page 64 (384 selections -> ~199 distinct experts under
    // uniform routing) but raises enqueue/collect round trips per layer from
    // 64 to ~199, and collect_moe ends in cudaStreamSynchronize -- so it
    // trades expert-fetch bytes for host stream drains, and Sigma_serial is
    // the term that usually dominates. The direction of that trade has not
    // been measured, so it does not default on.
    std::uint32_t prefill_page_tokens{};
    double sampling_temperature{};
    std::uint64_t sampling_seed{33'377'335U};
    bool verbose{};
    bool load_progress{};
    // Runs the MTP depth blocks to propose tokens the backbone then verifies.
    // Off by default: acceptance has not been measured on this checkpoint, and
    // an unmeasured draft adds compute to every step.
    bool enable_mtp_speculation{};
    std::uint32_t speculation_depth{};
    std::uint64_t request_id{};
};

struct InklingGraphStats {
    std::uint64_t forward_tokens{};
    std::uint64_t embedding_nanoseconds{};
    std::uint64_t attention_nanoseconds{};
    std::uint64_t short_conv_nanoseconds{};
    std::uint64_t dense_mlp_nanoseconds{};
    std::uint64_t moe_router_nanoseconds{};
    std::uint64_t moe_routed_nanoseconds{};
    std::uint64_t moe_shared_nanoseconds{};
    std::uint64_t output_head_nanoseconds{};
    // Routed-expert weight bytes actually touched, which is the term the
    // tiered-memory cost model is built around.
    std::uint64_t routed_expert_bytes{};
};

struct InklingDeviceStats {
    std::uint64_t expert_hits{};
    std::uint64_t expert_misses{};
    std::uint64_t expert_evictions{};
    std::uint64_t expert_stage_nanoseconds{};
    std::uint64_t expert_staged_bytes{};
    std::vector<std::uint64_t> cache_capacity_bytes;
    std::vector<std::uint64_t> cache_peak_bytes;
    std::vector<std::uint64_t> resident_spine_bytes;
    std::vector<std::uint64_t> resident_kv_bytes;
    bool enabled{};

    [[nodiscard]] double hit_rate() const noexcept {
        const auto total = expert_hits + expert_misses;
        return total == 0U ? 0.0
                           : static_cast<double>(expert_hits) /
                                 static_cast<double>(total);
    }
};

struct InklingSpeculationStats {
    std::uint64_t proposed{};
    std::uint64_t accepted{};

    [[nodiscard]] double acceptance_rate() const noexcept {
        return proposed > 0U ? static_cast<double>(accepted) /
                                   static_cast<double>(proposed)
                             : 0.0;
    }
};

struct InklingRunMetrics {
    std::uint64_t prompt_tokens{};
    std::uint64_t prefill_tokens{};
    std::uint64_t decode_tokens{};
    double prefill_seconds{};
    double decode_seconds{};
    CheckpointReadStats checkpoint_reads;
    InklingGraphStats graph;
    // Snapshot taken at the end of prefill. Decode-only cost is the difference
    // against `graph`; the charter requires the two phases be separated, and a
    // cold prefill otherwise dominates every per-phase share.
    InklingGraphStats prefill_graph;
    InklingSpeculationStats speculation;
    InklingDeviceStats device;
    InklingDeviceStats prefill_device;
    CudaBackendStats cuda;
    CudaBackendStats prefill_cuda;
    std::uint64_t rss_bytes{};

    [[nodiscard]] double prefill_tokens_per_second() const noexcept {
        return prefill_seconds > 0.0
            ? static_cast<double>(prefill_tokens) / prefill_seconds : 0.0;
    }
    [[nodiscard]] double decode_tokens_per_second() const noexcept {
        return decode_seconds > 0.0
            ? static_cast<double>(decode_tokens) / decode_seconds : 0.0;
    }
};

struct InklingGenerationResult {
    std::string text;
    std::vector<std::uint32_t> prompt_token_ids;
    std::vector<std::uint32_t> generated_token_ids;
    std::vector<TokenLogprob> logprobs;
    InklingRunMetrics metrics;
    std::vector<std::string> errors;
    bool stopped{};

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

class InklingRuntime {
public:
    InklingRuntime();
    ~InklingRuntime();
    InklingRuntime(InklingRuntime&&) noexcept;
    InklingRuntime& operator=(InklingRuntime&&) noexcept;
    InklingRuntime(const InklingRuntime&) = delete;
    InklingRuntime& operator=(const InklingRuntime&) = delete;

    [[nodiscard]] ValidationResult initialize(
        const std::string& model_directory,
        const InklingRuntimeConfig& config = {});
    [[nodiscard]] InklingGenerationResult generate_stream(
        std::string_view prompt, std::uint32_t maximum_new_tokens,
        const TokenStreamCallback& on_token = {});
    [[nodiscard]] InklingGenerationResult generate_chat_stream(
        std::span<const ChatMessage> messages,
        std::uint32_t maximum_new_tokens, const SamplingOptions& sampling,
        std::span<const std::string> stop,
        const TokenStreamCallback& on_token = {});

    // Teacher-forcing oracle: resets the sequence, runs the backbone over
    // `tokens`, and returns the next-token logits at each position. This is the
    // surface the correctness gates use, so it is exact and carries no
    // sampling state.
    [[nodiscard]] ValidationResult forward_logits(
        std::span<const std::uint32_t> tokens,
        std::vector<std::vector<float>>& logits);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata
