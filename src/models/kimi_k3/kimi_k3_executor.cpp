#include "../common/executor_support.hpp"
#include "strata/models/kimi_k3/kimi_k3_runtime.hpp"

namespace strata {
namespace {

class KimiK3Executor final : public ModelExecutor {
public:
    ValidationResult initialize(const std::string& model_directory,
                                const RuntimeConfig& config,
                                const PlacementPlan* placement) override {
        KimiK3RuntimeConfig concrete;
        // No devices / vram_cache_fraction here, and that is not an omission:
        // Kimi's config surface is host-RAM and NUMA shaped because it streams
        // MXFP4 experts from storage. Device selection reaches it through the
        // placement plan below.
        concrete.maximum_context_tokens = config.maximum_context_tokens;
        concrete.sampling_temperature = config.sampling.temperature;
        concrete.sampling_seed = config.sampling.seed;
        concrete.verbose = config.verbose;
        concrete.load_progress = config.load_progress;
        concrete.placement = placement;
        return runtime_.initialize(model_directory, concrete);
    }

    GenerationResult generate_chat_stream(
        std::span<const ChatMessage> messages, const GenerationOptions& options,
        const TokenStreamCallback& on_token) override {
        GenerationResult result;
        auto concrete = runtime_.generate_chat_stream(
            messages, options.maximum_new_tokens, options.sampling,
            options.stop, on_token);
        detail::copy_common_generation(result, concrete);
        // Unset for the same reason as Inkling (B6).
        return result;
    }

private:
    KimiK3Runtime runtime_;
};

const ModelRegistrar registrar{{
    RuntimeModel::KimiK3, "Kimi-K3", "kimi-k3", PlacementModel::KimiK3,
    false, true, false, false,
    // This model\'s reasoning, if any, is not separated yet; its output is
    // passed through whole and it accepts no budget.
    ReasoningFormat{},
    [] { return std::unique_ptr<ModelExecutor>(new KimiK3Executor()); }}};

}  // namespace
}  // namespace strata
