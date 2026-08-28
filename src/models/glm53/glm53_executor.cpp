#include "../common/executor_support.hpp"
#include "strata/engine/runtime_support.hpp"
#include "strata/models/glm53/glm53_runtime.hpp"

namespace strata {
namespace {

class Glm53Executor final : public ModelExecutor {
public:
    ValidationResult initialize(const std::string& model_directory,
                                const RuntimeConfig& config,
                                const PlacementPlan*) override {
        Glm53RuntimeConfig concrete;
        concrete.devices = resolve_runtime_devices(config.devices);
        concrete.vram_cache_fraction = config.vram_cache_fraction;
        concrete.maximum_context_tokens = config.maximum_context_tokens;
        concrete.sampling_temperature = config.sampling.temperature;
        concrete.sampling_seed = config.sampling.seed;
        concrete.verbose = config.verbose;
        concrete.load_progress = config.load_progress;
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
        result.metrics.reused_prompt_tokens =
            concrete.metrics.reused_prompt_tokens;
        result.metrics.incremental_kv_continuation = true;
        return result;
    }

private:
    Glm53Runtime runtime_;
};

const ModelRegistrar registrar{{
    RuntimeModel::Glm53, "GLM-5.3-Flash", "glm53", PlacementModel::Glm53,
    false, true, false, false,
    [] { return std::unique_ptr<ModelExecutor>(new Glm53Executor()); }}};

}  // namespace
}  // namespace strata
