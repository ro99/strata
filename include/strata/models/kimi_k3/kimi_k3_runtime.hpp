#pragma once

#include "strata/engine/chat_protocol.hpp"
#include "strata/models/kimi_k3/kimi_k3_expert_arena.hpp"
#include "strata/engine/placement.hpp"
#include "strata/engine/sampling.hpp"
#include "strata/platform/types.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strata {

// Called after each decoder layer with the running attention-residual prefix —
// the same `[tokens, hidden]` tensor `KimiDecoderLayer` returns. Full-model
// teacher forcing needs it: one end-to-end comparison across 93 layers says
// only whether the backbone agrees, not where it stopped agreeing.
using KimiLayerObserver =
    std::function<void(std::uint32_t layer, std::span<const float> hidden)>;

// Called after each MoE layer's router runs, with the deduplicated, sorted set
// of experts that layer selected. The reference oracle records the same array,
// so the two can be compared position by position -- which is what separates a
// routing divergence from a composition bug when the backbone disagrees.
using KimiRouteObserver =
    std::function<void(std::uint32_t layer, std::span<const std::uint32_t> experts)>;

struct KimiK3RuntimeConfig {
    std::uint32_t maximum_context_tokens{2048U};
    // Tokens per prefill page.
    //
    // The previous value of 64 was justified by observing that past roughly 256
    // tokens a page touches essentially all 896 experts of every layer, "so the
    // page cost stops falling". That inverts the conclusion. Once a page
    // saturates the routed set its *total* cost stops rising -- it is capped at
    // the whole 1347 GiB -- so cost per token is `saturated_set / P` and keeps
    // falling as 1/P. The fact cited as the reason to keep pages small is the
    // reason to make them large.
    //
    // Exact, from the contract: a page of P tokens draws 16P experts per layer
    // from 896, so it touches `896 * (1 - (1 - 1/896)^(16P))` distinct ones.
    //
    //     page   distinct/layer   GiB/token   vs decode   residual stream
    //       64              610       14.34        1.7x            15 MB
    //      256              887        5.21        4.6x            59 MB
    //      512              896        2.63        9.1x           117 MB
    //     2048              896        0.66       36.6x           470 MB
    //
    // Batch-1 decode is 24.06 GiB/token. At 64 tokens a page was therefore only
    // 1.7x cheaper per token than decode, which is the shape the charter names
    // as a defect: a page exists to amortize weight reads across its rows, and
    // 1.7x is not amortization.
    //
    // 512 buys 9.1x for 117 MB of residual stream, plus about 205 MB of
    // per-worker MoE accumulator at 28 workers -- together under half a
    // gigabyte against a 238 GiB host budget. Past 512 the routed set is fully
    // saturated and the return is linear in page size, so the ceiling is
    // activation memory rather than anything about routing.
    //
    // Real routing is skewed rather than uniform, which touches *fewer* distinct
    // experts at a given P. That lowers the cost at every page size and moves
    // saturation later; it does not change the 1/P scaling this rests on.
    //
    // Still owed: an end-to-end prefill measurement. The arithmetic above is
    // exact but it is storage volume, not wall time, and no Kimi-K3 prefill has
    // been profiled at any page size.
    std::uint32_t prefill_page_tokens{512U};
    // Host bytes for the routed-expert cache. Zero derives it from what is left
    // after the dense spine, against `host_budget_bytes`.
    std::uint64_t expert_arena_bytes{};
    std::uint64_t host_budget_bytes{};
    std::uint32_t expert_queue_depth{4U};
    bool direct_expert_reads{true};
    bool lock_expert_arena{true};
    // Disks that may not receive a byte derived from model weights. Empty by
    // default: the runtime already writes no derived copy of the checkpoint on
    // any device, which is the property that makes a storage -> storage repack
    // impossible, and a device name in a runtime default is operator policy
    // rather than a contract. Populate it to have the guard refuse a run whose
    // incidental write paths (results, traces) land on a protected disk.
    std::vector<std::string> forbidden_disks{};
    bool apply_write_guard{true};
    std::size_t host_workers{};  // zero uses the hardware concurrency
    double sampling_temperature{};
    std::uint64_t sampling_seed{33'377'335U};
    bool verbose{};
    bool load_progress{};
    // Optional pre-solved placement, borrowed for the duration of initialize.
    const PlacementPlan* placement{};
    // Unset on the serving path; the teacher-forcing oracle binds it.
    KimiLayerObserver layer_observer{};
    // Unset on the serving path; the gate 5/6 discriminating experiment binds
    // it. Advisory only -- it observes the routed set, it cannot change it.
    KimiRouteObserver route_observer{};
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

    // Runs `tokens` through the backbone at `position` and writes logits.
    //
    // The span decides how many rows come back. Pass one vocabulary and get the
    // final token's logits, which is what a prefill page and single-token decode
    // want. Pass `tokens.size()` vocabularies and get every row, which is what
    // speculative verification needs -- accepting the longest correct prefix
    // means comparing a draft's token at position i against this model's argmax
    // at i, for every i.
    //
    // All rows come from one pass over the backbone. The dense spine is 106.55
    // GiB read once per pass regardless of how many tokens it carries, measured
    // to amortize 3.63x at four tokens, so a batch is where that term stops
    // dominating.
    //
    // This is the surface the teacher-forcing and generation oracles drive: it
    // needs no tokenizer, so gates 5 and 6 do not wait on one.
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
