#include "../common/executor_support.hpp"
#include "strata/glm_runtime.hpp"

namespace strata {
namespace {

class Glm52Executor final : public ModelExecutor {
public:
    ValidationResult initialize(const std::string& model_directory,
                                const RuntimeConfig& config,
                                const PlacementPlan*) override {
        Glm52RuntimeConfig concrete;
        concrete.devices = config.devices;
        concrete.vram_cache_fraction = config.vram_cache_fraction;
        concrete.maximum_context_tokens = config.maximum_context_tokens;
        concrete.sampling_temperature = config.sampling.temperature;
        concrete.sampling_seed = config.sampling.seed;
        concrete.verbose = config.verbose;
        concrete.load_progress = config.load_progress;
        concrete.enable_flash_attention = config.enable_flash_attention;
        concrete.enable_incremental_kv_continuation =
            config.enable_incremental_kv_continuation;
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

private:
    Glm52Runtime runtime_;
};

const ModelRegistrar registrar{{
    RuntimeModel::Glm52, "GLM-5.2", "glm", PlacementModel::Glm52,
    false, true, false, false,
    [] { return std::unique_ptr<ModelExecutor>(new Glm52Executor()); }}};

}  // namespace
}  // namespace strata
