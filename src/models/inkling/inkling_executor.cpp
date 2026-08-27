#include "../common/executor_support.hpp"
#include "strata/engine/runtime_support.hpp"
#include "strata/models/inkling/inkling_runtime.hpp"

namespace strata {
namespace {

class InklingExecutor final : public ModelExecutor {
public:
    ValidationResult initialize(const std::string& model_directory,
                                const RuntimeConfig& config,
                                const PlacementPlan*) override {
        InklingRuntimeConfig concrete;
        // Phase 2, B5: these two were dropped by the old hand-written arm, so
        // --devices was accepted and discarded against a {0,1,2} default.
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
        // Left unset deliberately (B6): InklingRunMetrics has no such fields
        // because the runtime does not implement incremental prefix reuse.
        // "Not applicable" is the honest answer, and it is not zero.
        return result;
    }

private:
    InklingRuntime runtime_;
};

const ModelRegistrar registrar{{
    RuntimeModel::Inkling, "Inkling", "inkling", PlacementModel::Inkling,
    false, true, false, false,
    [] { return std::unique_ptr<ModelExecutor>(new InklingExecutor()); }}};

}  // namespace
}  // namespace strata
