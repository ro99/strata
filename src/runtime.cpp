#include "strata/runtime.hpp"

#include "strata/deepseek_runtime.hpp"
#include "strata/glm_runtime.hpp"

#include <utility>
#include <variant>

namespace strata {

struct RuntimeSession::Impl {
    std::variant<std::monostate, Glm52Runtime, DeepSeekV4Runtime> runtime;
};

RuntimeSession::RuntimeSession() : impl_(std::make_unique<Impl>()) {}
RuntimeSession::~RuntimeSession() = default;
RuntimeSession::RuntimeSession(RuntimeSession&&) noexcept = default;
RuntimeSession& RuntimeSession::operator=(RuntimeSession&&) noexcept = default;

ValidationResult RuntimeSession::initialize(
    const std::string& model_directory, const RuntimeConfig& config) {
    ValidationResult result;
    if (!std::holds_alternative<std::monostate>(impl_->runtime)) {
        result.errors.emplace_back("runtime session is already initialized");
        return result;
    }
    if (config.model == RuntimeModel::Glm52) {
        Glm52Runtime runtime;
        Glm52RuntimeConfig concrete;
        concrete.devices = config.devices;
        concrete.vram_cache_fraction = config.vram_cache_fraction;
        concrete.maximum_context_tokens = config.maximum_context_tokens;
        concrete.sampling_temperature = config.sampling_temperature;
        concrete.sampling_seed = config.sampling_seed;
        concrete.verbose = config.verbose;
        concrete.load_progress = config.load_progress;
        result = runtime.initialize(model_directory, concrete);
        if (result.ok()) impl_->runtime.emplace<Glm52Runtime>(std::move(runtime));
        return result;
    }
    DeepSeekV4Runtime runtime;
    Dsv4RuntimeConfig concrete;
    concrete.devices = config.devices;
    concrete.vram_cache_fraction = config.vram_cache_fraction;
    concrete.maximum_context_tokens = config.maximum_context_tokens;
    concrete.sampling_temperature = config.sampling_temperature;
    concrete.sampling_seed = config.sampling_seed;
    concrete.verbose = config.verbose;
    result = runtime.initialize(model_directory, concrete);
    if (result.ok()) impl_->runtime.emplace<DeepSeekV4Runtime>(std::move(runtime));
    return result;
}

GenerationResult RuntimeSession::generate_stream(
    std::string_view prompt, std::uint32_t maximum_new_tokens,
    const TokenStreamCallback& on_token) {
    GenerationResult result;
    if (auto* runtime = std::get_if<Glm52Runtime>(&impl_->runtime)) {
        auto concrete = runtime->generate_stream(prompt, maximum_new_tokens, on_token);
        result.text = std::move(concrete.text);
        result.prompt_token_ids = std::move(concrete.prompt_token_ids);
        result.generated_token_ids = std::move(concrete.generated_token_ids);
        result.metrics = {concrete.metrics.prompt_tokens,
                          concrete.metrics.decode_tokens,
                          concrete.metrics.prefill_seconds,
                          concrete.metrics.decode_seconds};
        result.errors = std::move(concrete.errors);
        return result;
    }
    if (auto* runtime = std::get_if<DeepSeekV4Runtime>(&impl_->runtime)) {
        auto concrete = runtime->generate_stream(prompt, maximum_new_tokens, on_token);
        result.text = std::move(concrete.text);
        result.prompt_token_ids = std::move(concrete.prompt_token_ids);
        result.generated_token_ids = std::move(concrete.generated_token_ids);
        result.metrics = {concrete.metrics.prompt_tokens,
                          concrete.metrics.decode_tokens,
                          concrete.metrics.prefill_seconds,
                          concrete.metrics.decode_seconds};
        result.errors = std::move(concrete.errors);
        return result;
    }
    result.errors.emplace_back("runtime session is not initialized");
    return result;
}

}  // namespace strata
