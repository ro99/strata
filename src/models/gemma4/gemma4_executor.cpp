#include "../common/executor_support.hpp"
#include "strata/engine/runtime_support.hpp"
#include "strata/models/gemma4/gemma4_runtime.hpp"

namespace strata {
namespace {

class Gemma4Executor final : public ModelExecutor {
public:
    ValidationResult initialize(const std::string& model_directory,
                                const RuntimeConfig& config,
                                const PlacementPlan* placement) override {
        Gemma4RuntimeConfig concrete;
        concrete.devices = resolve_runtime_devices(config.devices);
        concrete.vram_cache_fraction = config.vram_cache_fraction;
        concrete.maximum_context_tokens = config.maximum_context_tokens;
        concrete.sampling_temperature = config.sampling.temperature;
        concrete.sampling_seed = config.sampling.seed;
        concrete.verbose = config.verbose;
        concrete.load_progress = config.load_progress;
        concrete.enable_flash_attention = config.enable_flash_attention;
        concrete.enable_incremental_kv_continuation =
            config.enable_incremental_kv_continuation;
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
        result.metrics.reused_prompt_tokens = concrete.metrics.reused_prompt_tokens;
        result.metrics.incremental_kv_continuation =
            concrete.metrics.incremental_kv_continuation;
        return result;
    }

    // The only multimodal model in the set.
    [[nodiscard]] bool accepts_images() const noexcept override { return true; }

private:
    Gemma4Runtime runtime_;
};

const ModelRegistrar registrar{{
    RuntimeModel::Gemma4, "Gemma 4", "gemma4", PlacementModel::Gemma4,
    false, true, true, false,
    [] { return std::unique_ptr<ModelExecutor>(new Gemma4Executor()); }}};

}  // namespace
}  // namespace strata
