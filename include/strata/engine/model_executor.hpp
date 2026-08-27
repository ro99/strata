#pragma once

// The model seam.
//
// Everything above this header (RuntimeSession, the CLIs, the server) speaks
// only ModelExecutor and the registry. Everything below it (the six concrete
// runtimes) implements ModelExecutor and registers itself from its own
// translation unit.
//
// The point of the file is what it removes. Before it, adding a model meant
// editing twelve shared files -- a closed enum here, a variant alternative
// there, an if-chain arm, a string ternary in two applications, a placement
// switch -- and forgetting any one of them was silent. Registration is now
// self-contained: a model contributes one translation unit in its own
// directory and nothing above it changes.
//
// Layering: this header and the registry are strata_engine. Models depend on
// it downward; the application facade depends on it downward. If the config
// and result types lived with RuntimeSession in strata_app, every model would
// depend upward on the application tier, which is the inversion check-symbols
// exists to catch.

#include "strata/engine/chat_protocol.hpp"
#include "strata/engine/placement.hpp"
#include "strata/platform/result.hpp"
#include "strata/engine/sampling.hpp"
#include "strata/platform/types.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strata {

enum class RuntimeModel : std::uint8_t {
    Glm52,
    DeepSeekV4,
    Gemma4,
    KimiK3,
    Laguna,
    Inkling,
};

struct RuntimeConfig {
    RuntimeModel model{RuntimeModel::Glm52};
    // Empty means every visible device. The old default named three GPUs
    // because the development box had three; on a one- or two-GPU machine it
    // silently claimed devices that were not there.
    std::vector<int> devices;
    double vram_cache_fraction{0.85};
    std::uint32_t maximum_context_tokens{2048U};
    // Session default sampler. Greedy unless the caller asks otherwise, so a
    // run stays reproducible until someone opts into a stochastic stage.
    SamplingOptions sampling{greedy_sampling()};
    bool verbose{};
    bool load_progress{};
    bool enable_flash_attention{};
    bool enable_incremental_kv_continuation{true};
    bool deepseek_block_kv_cache{};
    // The DeepSeek device-resident decode contract: physical KV pages,
    // device-resident mHC, CUDA attention, the scalar lightning indexer, and
    // routed experts in the two NUMA-local CPU shards. It is a bundle rather
    // than a knob, so setting it overrides the individual switches above
    // exactly as strata-deepseek-run's --device-resident-runtime does.
    bool deepseek_device_resident_runtime{};
    // Rank-local TP2 decode. Opt-in, admitted fail-closed before the
    // checkpoint is opened, and never falling back once admitted. Requires an
    // NCCL build and exactly two devices, and implies the device-resident
    // contract above.
    bool deepseek_rank_local_decode{};
    // Prompt rows per prefill page. Zero keeps the runtime default. The
    // device-resident path executes a page layer-major over one mHC slot per
    // row and groups the page's rows by expert; 1 restores row-at-a-time
    // prompt processing.
    std::uint32_t deepseek_prefill_page_tokens{};
    // Routed-expert tier: a plan produced by the offline experiment planner,
    // VRAM each rank device spends holding its slice of it. The bytes come out
    // of the centralized prefill expert cache, which decode never reads.
    // An empty path disables the tier, which is the default.
    std::string deepseek_static_expert_plan;
    std::uint64_t deepseek_static_expert_bytes{};
    bool pin_resident_arena{};
    bool prepack_mhc_projection{true};
    // Placement plan cache. An empty directory selects the default location;
    // see placement_cache_directory. A cached plan that matches this
    // checkpoint, hardware, and request is reused instead of recomputed.
    std::string placement_cache_directory;
    bool use_placement_cache{true};
    bool refresh_placement_plan{};
    bool report_placement_plan{};
};

struct GenerationMetrics {
    std::uint64_t prompt_tokens{};
    std::uint64_t prefill_tokens{};
    // Unset, not zero, when the loaded runtime does not implement incremental
    // prefix reuse at all (Inkling, Kimi-K3 today). GenerationMetrics has no
    // other way to say "not applicable" versus "measured, and the answer is
    // none reused" (Phase 2, B6) -- a defaulted 0/false was indistinguishable
    // from a genuine zero.
    std::optional<std::uint64_t> reused_prompt_tokens;
    std::uint64_t decode_tokens{};
    double prefill_seconds{};
    double decode_seconds{};
    std::optional<bool> incremental_kv_continuation;
};

struct GenerationResult {
    std::string text;
    std::vector<std::uint32_t> prompt_token_ids;
    std::vector<std::uint32_t> generated_token_ids;
    std::vector<TokenLogprob> logprobs;
    GenerationMetrics metrics;
    std::vector<std::string> errors;
    bool stopped{};

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

struct GenerationOptions {
    std::uint32_t maximum_new_tokens{256U};
    SamplingOptions sampling;
    std::vector<std::string> stop;
};

// One loaded model. Implementations wrap a concrete runtime and own the
// translation between its architecture-specific config/result types and the
// neutral ones above. Research tools that need architecture-specific
// diagnostics still reach past this to the concrete runtime directly; that is
// the seam's declared limit, not an oversight.
class ModelExecutor {
public:
    ModelExecutor() = default;
    virtual ~ModelExecutor() = default;
    ModelExecutor(const ModelExecutor&) = delete;
    ModelExecutor& operator=(const ModelExecutor&) = delete;

    [[nodiscard]] virtual ValidationResult initialize(
        const std::string& model_directory, const RuntimeConfig& config,
        const PlacementPlan* placement) = 0;

    [[nodiscard]] virtual GenerationResult generate_chat_stream(
        std::span<const ChatMessage> messages, const GenerationOptions& options,
        const TokenStreamCallback& on_token) = 0;

    // Image parts in a chat message are rejected above unless this is true.
    [[nodiscard]] virtual bool accepts_images() const noexcept { return false; }
};

struct ModelRegistration {
    RuntimeModel model{};
    // Human-readable, used in diagnostics and rejection messages.
    const char* name{};
    // The --model-type token. Applications resolve this through the registry
    // rather than carrying their own string-to-enum chain, so adding a model
    // does not touch any CLI.
    const char* cli_name{};
    PlacementModel placement{};
    // Presentation defaults the applications used to hardcode per model:
    // DeepSeek prints its own progress and wants verbose, the others want a
    // load-progress bar; Gemma 4 and Laguna always run flash attention.
    bool verbose_by_default{};
    bool progress_by_default{true};
    bool flash_attention_by_default{};
    // DeepSeek-only controls (rank-local decode, the device-resident bundle,
    // the block KV cache) are rejected rather than ignored for every model
    // that does not implement them: a request for rank-local decode that
    // quietly ran a centralized GLM would report the accepted path while
    // executing a different one.
    bool accepts_deepseek_controls{};
    std::unique_ptr<ModelExecutor> (*make)(){};
};

// Registers a model. Call once per model, from that model's own translation
// unit, via the ModelRegistrar below rather than directly.
void register_model(const ModelRegistration& registration);

// Null for a value that is not a registered model, including a value outside
// the enum -- which is legal for a scoped enum with an explicit underlying
// type, and is what makes this a runtime check rather than a formality.
[[nodiscard]] const ModelRegistration* find_model(RuntimeModel model) noexcept;

// Resolves a --model-type token. Null if no registered model claims it.
[[nodiscard]] const ModelRegistration* find_model_by_cli_name(
    std::string_view cli_name) noexcept;

// Comma-separated list of every registered --model-type token, for help text
// and rejection messages. Generated, so it cannot drift from the registry.
[[nodiscard]] std::string registered_model_names();

[[nodiscard]] std::span<const ModelRegistration> registered_models() noexcept;

// File-scope registrar. Declare one `const ModelRegistrar` in an anonymous
// namespace in the model's own .cpp; its constructor runs before main.
//
// strata_models is linked with --whole-archive precisely so these objects
// survive: a static library drops any member no symbol references, and
// nothing references a self-registering translation unit by definition. That
// link flag is what buys "adding a model touches zero shared files".
struct ModelRegistrar {
    explicit ModelRegistrar(const ModelRegistration& registration) {
        register_model(registration);
    }
};

}  // namespace strata
