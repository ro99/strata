#pragma once

#include "strata/device/cuda_backend.hpp"
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

// Deterministic weighted assignment used by both warmup and execution. The
// same projection key therefore has exactly one CUDA home, while independent
// projections spread according to discovered cache capacity.
[[nodiscard]] std::vector<std::size_t> glm53_projection_slots(
    std::span<const std::string_view> keys,
    std::span<const std::uint64_t> costs,
    std::span<const std::uint64_t> capacities,
    std::size_t preferred_slot);

struct Glm53RuntimeConfig {
    std::vector<int> devices;
    double vram_cache_fraction{0.85};
    std::uint32_t maximum_context_tokens{2048U};
    double sampling_temperature{};
    std::uint64_t sampling_seed{33'377'335U};
    // Upper bound for a prefill scheduler page. The default preserves the
    // production path; experiments may override it to measure weight reuse.
    std::uint32_t prefill_page_tokens{64U};
    bool verbose{};
    bool load_progress{};
    // Opt-in request attribution. CUDA event timing is enabled only when this
    // is true; ordinary production execution retains its existing timing path.
    bool phase_profile{};
};

struct Glm53CacheMetrics {
    std::uint64_t hits{};
    std::uint64_t misses{};
    std::uint64_t evictions{};
    std::uint64_t prefetches{};
    std::uint64_t useful_prefetches{};
    std::uint64_t failed_prefetches{};
};

struct Glm53HostExpertMetrics {
    std::uint64_t calls{};
    std::uint64_t rows{};
    std::uint64_t gate_up_weight_bytes{};
    std::uint64_t down_weight_bytes{};
    std::uint64_t view_resolution_nanoseconds{};
    std::uint64_t input_quantization_nanoseconds{};
    std::uint64_t gate_up_nanoseconds{};
    std::uint64_t activation_nanoseconds{};
    std::uint64_t down_nanoseconds{};
    std::uint64_t reduction_nanoseconds{};
    std::uint64_t service_nanoseconds{};
    std::uint64_t temporary_allocation_calls{};
    std::uint64_t ep2_calls{};
    std::uint64_t ep2_owner0_experts{};
    std::uint64_t ep2_owner1_experts{};
    std::uint64_t ep2_imbalance_experts{};
};

struct Glm53GraphMetrics {
    std::uint64_t forward_calls{};
    std::uint64_t forward_rows{};
    std::uint64_t embedding_nanoseconds{};
    std::uint64_t layer_nanoseconds{};
    std::uint64_t attention_block_nanoseconds{};
    std::uint64_t kda_nanoseconds{};
    std::uint64_t mla_nanoseconds{};
    std::uint64_t feedforward_block_nanoseconds{};
    std::uint64_t output_head_nanoseconds{};
    std::uint64_t sampling_nanoseconds{};
};

struct Glm53PhaseMetrics {
    CudaBackendStats cuda;
    Glm53CacheMetrics cache;
    Glm53HostExpertMetrics host_experts;
    Glm53GraphMetrics graph;
};

struct Glm53RunMetrics {
    std::uint64_t prompt_tokens{};
    std::uint64_t prefill_tokens{};
    std::uint64_t reused_prompt_tokens{};
    std::uint64_t decode_tokens{};
    double prefill_seconds{};
    double decode_seconds{};
    double decode_prepare_seconds{};
    Glm53PhaseMetrics prefill;
    Glm53PhaseMetrics decode;
    std::uint64_t rss_bytes{};
    std::vector<std::uint64_t> device_vram_used_bytes;
    bool phase_profile{};
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
