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

// Host decoders for the two expert storage formats the MXFP4 release adds.
// Exposed so the vectorized and scalar decoders can be compared against each
// other and against a reference dequantization; the host MoE calls exactly
// these functions. `use_avx2` false forces the scalar reference.
//
// MXFP4: `packed` holds one row's E2M1 nibbles, column 2b in the low nibble of
// byte b, and `scales` one E8M0 byte per 32 columns of that row.
[[nodiscard]] float glm53_host_fp4_group32_row_dot(
    std::span<const std::byte> packed, std::span<const std::byte> scales,
    std::span<const float> input, bool use_avx2) noexcept;
// FP8: `weights` holds one row's E4M3 codes and `scales` the F32 inverse
// scales for that row's 128-column blocks. Exposed alongside the other two so
// the three formats can be priced against each other on the same input.
[[nodiscard]] float glm53_host_fp8_block128_row_dot(
    std::span<const std::byte> weights, std::span<const float> scales,
    std::span<const float> input, bool use_avx2) noexcept;
// BF16 rows carry no scale.
[[nodiscard]] float glm53_host_bf16_row_dot(
    std::span<const std::byte> weights, std::span<const float> input,
    bool use_avx2) noexcept;

// GLM-5.3 k-pool sparse indexer selection (record 0237). Chooses which history
// positions one decode query attends to, so attention cost stops growing with
// context. Exposed for test: below `index_topk` the selection is the identity,
// which is the regression gate for everything above it.
struct Glm53SparseIndexParameters {
    static constexpr std::uint32_t heads = 32U;
    static constexpr std::uint32_t head_dim = 128U;
    static constexpr std::uint32_t top_k = 2048U;
    static constexpr std::uint32_t pool = 4U;
    static constexpr std::uint32_t selection_width = top_k + pool - 1U;
};

[[nodiscard]] std::size_t glm53_sparse_index_select_for_test(
    std::span<std::uint32_t> selected, std::span<const float> indexer_query,
    std::span<const float> indexer_keys, std::span<const float> gate_scores,
    std::span<const float> pool_ape, std::span<const float> head_weights,
    std::uint32_t history);

// The two projections the selection is fed from, exposed so the reference
// oracle can drive the whole indexer chain rather than only its ranking. The
// key norm is `nn.LayerNorm(head_dim, eps=1e-6)` -- mean subtracting, with a
// bias -- and not the RMSNorm this model uses everywhere else, which is a
// difference no gate at or below `index_topk` can see.
void glm53_indexer_gate_for_test(std::span<float> output,
                                 std::span<const float> input,
                                 std::span<const float> weight) noexcept;
void glm53_indexer_layer_norm_for_test(std::span<float> values,
                                       std::span<const float> weight,
                                       std::span<const float> bias) noexcept;

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

// Wall-clock attribution inside sparse resident MLA. CUDA kernel event time
// remains in CudaBackendStats; the device-scores wait deliberately includes
// queued device work because it measures the serialization boundary seen by
// the host thread.
struct Glm53SparseMlaMetrics {
    std::uint64_t calls{};
    std::uint64_t input_download_nanoseconds{};
    std::uint64_t indexer_projection_nanoseconds{};
    std::uint64_t indexer_state_nanoseconds{};
    std::uint64_t query_rank_projection_nanoseconds{};
    std::uint64_t pool_scoring_nanoseconds{};
    std::uint64_t topk_sort_nanoseconds{};
    std::uint64_t arena_bookkeeping_nanoseconds{};
    std::uint64_t index_upload_nanoseconds{};
    std::uint64_t device_scores_wait_nanoseconds{};
    std::uint64_t host_softmax_nanoseconds{};
    std::uint64_t coefficient_upload_nanoseconds{};
};

struct Glm53PhaseMetrics {
    CudaBackendStats cuda;
    Glm53CacheMetrics cache;
    Glm53HostExpertMetrics host_experts;
    Glm53GraphMetrics graph;
    Glm53SparseMlaMetrics sparse_mla;
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
