#include "strata/runtime.hpp"

#include "strata/deepseek_runtime.hpp"
#include "strata/gemma4_runtime.hpp"
#include "strata/glm_runtime.hpp"

#include <array>
#include <algorithm>
#include <utility>
#include <variant>

namespace strata {

struct RuntimeSession::Impl {
    std::variant<std::monostate, Glm52Runtime, DeepSeekV4Runtime,
                 Gemma4Runtime> runtime;
    SamplingOptions sampling;
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
    impl_->sampling = config.sampling;
    if (config.model == RuntimeModel::Glm52) {
        if (config.deepseek_block_kv_cache) {
            result.errors.emplace_back(
                "DeepSeek block KV cache cannot be used by the GLM runtime");
            return result;
        }
        Glm52Runtime runtime;
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
        result = runtime.initialize(model_directory, concrete);
        if (result.ok()) impl_->runtime.emplace<Glm52Runtime>(std::move(runtime));
        return result;
    }
    if (config.model == RuntimeModel::Gemma4) {
        if (config.deepseek_block_kv_cache) {
            result.errors.emplace_back(
                "DeepSeek block KV cache cannot be used by the Gemma 4 runtime");
            return result;
        }
        Gemma4Runtime runtime;
        Gemma4RuntimeConfig concrete;
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
        result = runtime.initialize(model_directory, concrete);
        if (result.ok()) impl_->runtime.emplace<Gemma4Runtime>(std::move(runtime));
        return result;
    }
    DeepSeekV4Runtime runtime;
    Dsv4RuntimeConfig concrete;
    concrete.devices = config.devices;
    concrete.vram_cache_fraction = config.vram_cache_fraction;
    concrete.maximum_context_tokens = config.maximum_context_tokens;
    concrete.sampling_temperature = config.sampling.temperature;
    concrete.sampling_seed = config.sampling.seed;
    concrete.verbose = config.verbose;
    concrete.enable_flash_attention = config.enable_flash_attention;
    concrete.enable_incremental_kv_continuation =
        config.enable_incremental_kv_continuation;
    concrete.pin_resident_arena = config.pin_resident_arena;
    concrete.prepack_mhc_projection = config.prepack_mhc_projection;
    concrete.kv_cache_mode = config.deepseek_block_kv_cache
        ? Dsv4KvCacheMode::Block : Dsv4KvCacheMode::ScalarOracle;
    result = runtime.initialize(model_directory, concrete);
    if (result.ok()) impl_->runtime.emplace<DeepSeekV4Runtime>(std::move(runtime));
    return result;
}

GenerationResult RuntimeSession::generate_stream(
    std::string_view prompt, std::uint32_t maximum_new_tokens,
    const TokenStreamCallback& on_token) {
    const std::array messages{ChatMessage{ChatRole::User,
                                          std::string(prompt)}};
    return generate_chat_stream(messages, maximum_new_tokens, on_token);
}

GenerationResult RuntimeSession::generate_chat_stream(
    std::span<const ChatMessage> messages,
    std::uint32_t maximum_new_tokens,
    const TokenStreamCallback& on_token) {
    GenerationOptions options;
    options.maximum_new_tokens = maximum_new_tokens;
    options.sampling = impl_->sampling;
    return generate_chat_stream(messages, options, on_token);
}

GenerationResult RuntimeSession::generate_chat_stream(
    std::span<const ChatMessage> messages,
    const GenerationOptions& options,
    const TokenStreamCallback& on_token) {
    GenerationResult result;
    const bool has_images = std::any_of(
        messages.begin(), messages.end(), [](const ChatMessage& message) {
            return std::any_of(message.parts.begin(), message.parts.end(),
                [](const ChatContentPart& part) {
                    return part.kind == ChatContentKind::Image;
                });
        });
    if (has_images && !std::holds_alternative<Gemma4Runtime>(impl_->runtime)) {
        result.errors.emplace_back(
            "this loaded model does not support image content");
        return result;
    }
    if (auto* runtime = std::get_if<Glm52Runtime>(&impl_->runtime)) {
        auto concrete = runtime->generate_chat_stream(
            messages, options.maximum_new_tokens, options.sampling,
            options.stop, on_token);
        result.text = std::move(concrete.text);
        result.prompt_token_ids = std::move(concrete.prompt_token_ids);
        result.generated_token_ids = std::move(concrete.generated_token_ids);
        result.logprobs = std::move(concrete.logprobs);
        result.metrics.prompt_tokens = concrete.metrics.prompt_tokens;
        result.metrics.prefill_tokens = concrete.metrics.prefill_tokens;
        result.metrics.reused_prompt_tokens =
            concrete.metrics.reused_prompt_tokens;
        result.metrics.decode_tokens = concrete.metrics.decode_tokens;
        result.metrics.prefill_seconds = concrete.metrics.prefill_seconds;
        result.metrics.decode_seconds = concrete.metrics.decode_seconds;
        result.metrics.incremental_kv_continuation =
            concrete.metrics.incremental_kv_continuation;
        result.errors = std::move(concrete.errors);
        result.stopped = concrete.stopped;
        return result;
    }
    if (auto* runtime = std::get_if<DeepSeekV4Runtime>(&impl_->runtime)) {
        auto concrete = runtime->generate_chat_stream(
            messages, options.maximum_new_tokens, options.sampling,
            options.stop, on_token);
        result.text = std::move(concrete.text);
        result.prompt_token_ids = std::move(concrete.prompt_token_ids);
        result.generated_token_ids = std::move(concrete.generated_token_ids);
        result.logprobs = std::move(concrete.logprobs);
        result.metrics.prompt_tokens = concrete.metrics.prompt_tokens;
        result.metrics.prefill_tokens = concrete.metrics.prefill_tokens;
        result.metrics.reused_prompt_tokens =
            concrete.metrics.reused_prompt_tokens;
        result.metrics.decode_tokens = concrete.metrics.decode_tokens;
        result.metrics.prefill_seconds = concrete.metrics.prefill_seconds;
        result.metrics.decode_seconds = concrete.metrics.decode_seconds;
        result.metrics.incremental_kv_continuation =
            concrete.metrics.incremental_kv_continuation;
        result.errors = std::move(concrete.errors);
        result.stopped = concrete.stopped;
        return result;
    }
    if (auto* runtime = std::get_if<Gemma4Runtime>(&impl_->runtime)) {
        auto concrete = runtime->generate_chat_stream(
            messages, options.maximum_new_tokens, options.sampling,
            options.stop, on_token);
        result.text = std::move(concrete.text);
        result.prompt_token_ids = std::move(concrete.prompt_token_ids);
        result.generated_token_ids = std::move(concrete.generated_token_ids);
        result.logprobs = std::move(concrete.logprobs);
        result.metrics.prompt_tokens = concrete.metrics.prompt_tokens;
        result.metrics.prefill_tokens = concrete.metrics.prefill_tokens;
        result.metrics.reused_prompt_tokens =
            concrete.metrics.reused_prompt_tokens;
        result.metrics.decode_tokens = concrete.metrics.decode_tokens;
        result.metrics.prefill_seconds = concrete.metrics.prefill_seconds;
        result.metrics.decode_seconds = concrete.metrics.decode_seconds;
        result.metrics.incremental_kv_continuation =
            concrete.metrics.incremental_kv_continuation;
        result.errors = std::move(concrete.errors);
        result.stopped = concrete.stopped;
        return result;
    }
    result.errors.emplace_back("runtime session is not initialized");
    return result;
}

}  // namespace strata
