#pragma once

#include "strata/chat_protocol.hpp"
#include "strata/diagnostics.hpp"
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

struct Gemma4RuntimeConfig {
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
    bool enable_incremental_kv_continuation{true};
    // Default off, no cost on the hot path either way. Records a per-layer
    // BF16 hash of the residual stream and a per-operation hash of each
    // layer's attention and MLP outputs, during the host-side forward_layers
    // path (prefill, and any decode step that has not yet uploaded its KV
    // cache to the fused device path). The fused device decode path has no
    // host-visible layer boundary, so it is not covered -- the same
    // limitation DeepSeek's own device-resident decode path has.
    bool enable_layer_hash_trace{};
    // Opt-in CUDA event attribution for the cost-model probe. Off by default;
    // event recording would otherwise perturb every production projection.
    bool enable_cuda_phase_timing{};
    // Optional pre-solved placement. When present and prescriptive it supplies
    // the layer-to-device assignment and the admitted per-device budgets, so
    // the load performs exactly the placement a dry run printed. Borrowed for
    // the duration of initialize only.
    const PlacementPlan* placement{};
};

struct Gemma4CudaPhaseMetrics {
    std::uint64_t h2d_bytes{};
    std::uint64_t d2h_bytes{};
    double h2d_seconds{};
    double kernel_seconds{};
    double d2h_seconds{};
    double synchronization_seconds{};
};

struct Gemma4RunMetrics {
    std::uint64_t prompt_tokens{};
    std::uint64_t prefill_tokens{};
    std::uint64_t reused_prompt_tokens{};
    std::uint64_t decode_tokens{};
    double prefill_seconds{};
    double decode_seconds{};
    double first_decode_seconds{};
    double steady_decode_seconds{};
    std::uint64_t steady_decode_tokens{};
    std::uint64_t rss_bytes{};
    std::vector<std::uint64_t> device_vram_used_bytes;
    bool incremental_kv_continuation{};
    Gemma4CudaPhaseMetrics prefill_cuda;
    Gemma4CudaPhaseMetrics decode_cuda;
    Gemma4CudaPhaseMetrics first_decode_cuda;
    Gemma4CudaPhaseMetrics steady_decode_cuda;
};

struct Gemma4GenerationResult {
    std::string text;
    std::vector<std::uint32_t> prompt_token_ids;
    std::vector<std::uint32_t> generated_token_ids;
    std::vector<TokenLogprob> logprobs;
    Gemma4RunMetrics metrics;
    DiagnosticTrace diagnostics;
    std::vector<std::string> errors;
    bool stopped{};

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

class Gemma4Runtime {
public:
    Gemma4Runtime();
    ~Gemma4Runtime();
    Gemma4Runtime(Gemma4Runtime&&) noexcept;
    Gemma4Runtime& operator=(Gemma4Runtime&&) noexcept;
    Gemma4Runtime(const Gemma4Runtime&) = delete;
    Gemma4Runtime& operator=(const Gemma4Runtime&) = delete;

    [[nodiscard]] ValidationResult initialize(
        const std::string& model_directory,
        const Gemma4RuntimeConfig& config = {});
    [[nodiscard]] Gemma4GenerationResult generate_stream(
        std::string_view prompt, std::uint32_t maximum_new_tokens,
        const TokenStreamCallback& on_token = {});
    [[nodiscard]] Gemma4GenerationResult generate_chat_stream(
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
