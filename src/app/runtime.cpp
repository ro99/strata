#include "strata/runtime.hpp"

#include "strata/deepseek_runtime.hpp"
#include "strata/dsv4_attention_kv.hpp"
#include "strata/gemma4_runtime.hpp"
#include "strata/glm_runtime.hpp"
#include "strata/inkling_runtime.hpp"
#include "strata/kimi_k3_runtime.hpp"
#include "strata/laguna_runtime.hpp"

#include <array>
#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <variant>

namespace strata {

struct RuntimeSession::Impl {
    std::variant<std::monostate, Glm52Runtime, DeepSeekV4Runtime,
                 Gemma4Runtime, LagunaRuntime, InklingRuntime,
                 KimiK3Runtime> runtime;
    SamplingOptions sampling;
    PlacementPlan placement;
    bool placement_ready{};
};

namespace {

// Names a declared model, or nullptr for a value that is not an enumerator.
//
// This is the single admission point for RuntimeModel and it carries two
// separate guarantees. At compile time there is no default label, so
// -Werror=switch (scoped to this file in CMakeLists.txt) turns a seventh
// enumerator into a build failure right here. At run time a value outside the
// enum -- legal for a scoped enum with an explicit underlying type -- returns
// nullptr, so initialize can reject it instead of falling through to whichever
// runtime happens to be written last.
[[nodiscard]] const char* runtime_model_name(RuntimeModel model) noexcept {
    switch (model) {
        case RuntimeModel::Glm52: return "GLM-5.2";
        case RuntimeModel::DeepSeekV4: return "DeepSeek-V4";
        case RuntimeModel::Gemma4: return "Gemma 4";
        case RuntimeModel::KimiK3: return "Kimi-K3";
        case RuntimeModel::Laguna: return "Laguna";
        case RuntimeModel::Inkling: return "Inkling";
    }
    return nullptr;
}

[[nodiscard]] PlacementModel placement_model(RuntimeModel model) noexcept {
    switch (model) {
        case RuntimeModel::Glm52: return PlacementModel::Glm52;
        case RuntimeModel::DeepSeekV4: return PlacementModel::DeepSeekV4;
        case RuntimeModel::Gemma4: return PlacementModel::Gemma4;
        case RuntimeModel::Laguna: return PlacementModel::Laguna;
        case RuntimeModel::Inkling: return PlacementModel::Inkling;
        case RuntimeModel::KimiK3: return PlacementModel::KimiK3;
    }
    // Unreachable: initialize rejects an undeclared model before anything
    // reaches this, and -Werror=switch covers the missing-enumerator case.
    return PlacementModel::Gemma4;
}

// Names the DeepSeek-only switch a request carries, or nullptr if it carries
// none. Every other runtime rejects on it rather than ignoring it: a request
// for rank-local decode that quietly runs a centralized GLM would report the
// accepted path while executing a different one.
[[nodiscard]] const char* deepseek_only_control(
    const RuntimeConfig& config) noexcept {
    if (config.deepseek_rank_local_decode) return "rank-local decode";
    if (config.deepseek_device_resident_runtime) {
        return "device-resident runtime";
    }
    if (config.deepseek_block_kv_cache) return "block KV cache";
    return nullptr;
}

}  // namespace

PlacementRequest placement_request_for(const std::string& model_directory,
                                       const RuntimeConfig& config) {
    PlacementRequest request;
    request.model = placement_model(config.model);
    request.model_directory = model_directory;
    request.devices = config.devices;
    request.vram_cache_fraction = config.vram_cache_fraction;
    request.maximum_context_tokens = config.maximum_context_tokens;
    // The device-resident contract implies both of these, so the plan reports
    // the layout the runtime will actually build rather than the one the bare
    // flags describe. A request cannot express physical KV pages, so for that
    // mode the plan still sizes the compact cache; it is advisory for DeepSeek
    // and the runtime admits the physical geometry itself.
    request.flash_attention = config.enable_flash_attention ||
                              config.deepseek_device_resident_runtime;
    request.block_kv_cache = config.deepseek_block_kv_cache ||
                             config.deepseek_device_resident_runtime;
    return request;
}

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
    // Reject before any placement work or checkpoint access. The dispatch
    // below is an if-chain whose last arm is unconditional, so without this an
    // undeclared model would be constructed as DeepSeek against whatever
    // directory was supplied and report initialization success.
    if (runtime_model_name(config.model) == nullptr) {
        result.errors.emplace_back(
            "unhandled runtime model: " +
            std::to_string(static_cast<unsigned>(config.model)));
        return result;
    }
    impl_->sampling = config.sampling;

    // Resolve placement before anything is uploaded. The plan decides where
    // Gemma 4 puts each layer; for GLM and DeepSeek it reports and admits the
    // placement those runtimes already perform without changing it, so a
    // planning defect cannot regress a validated runtime.
    auto resolved = resolve_placement_plan(
        placement_request_for(model_directory, config),
        config.placement_cache_directory, config.use_placement_cache,
        config.refresh_placement_plan);
    if (resolved.ok()) {
        impl_->placement = std::move(resolved.value.plan);
        impl_->placement_ready = true;
        if (config.report_placement_plan) {
            std::cerr << render_placement_report(impl_->placement)
                      << "[placement] ";
            if (resolved.value.from_cache) {
                std::cerr << "reused cached plan " << resolved.value.cache_path;
            } else if (resolved.value.stored) {
                std::cerr << "computed and cached plan "
                          << resolved.value.cache_path;
            } else {
                std::cerr << "computed plan; not cached";
            }
            std::cerr << '\n';
        }
        auto hardware = probe_placement_hardware(config.devices);
        if (hardware.ok()) {
            const auto verified =
                verify_placement_plan(impl_->placement, hardware.value);
            if (!verified.ok() && impl_->placement.prescriptive) {
                result.errors = verified.errors;
                return result;
            }
            for (const auto& error : verified.errors) {
                std::cerr << "[placement] warning: " << error << '\n';
            }
        }
    } else if (config.report_placement_plan) {
        for (const auto& error : resolved.errors) {
            std::cerr << "[placement] warning: " << error << '\n';
        }
    }
    const auto* placement =
        impl_->placement_ready ? &impl_->placement : nullptr;

    if (config.model == RuntimeModel::Glm52) {
        if (const auto* control = deepseek_only_control(config)) {
            result.errors.emplace_back(
                std::string("DeepSeek ") + control +
                " cannot be used by the GLM runtime");
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
    if (config.model == RuntimeModel::KimiK3) {
        if (const auto* control = deepseek_only_control(config)) {
            result.errors.emplace_back(
                std::string("DeepSeek ") + control +
                " cannot be used by the Kimi-K3 runtime");
            return result;
        }
        KimiK3Runtime runtime;
        KimiK3RuntimeConfig concrete;
        concrete.maximum_context_tokens = config.maximum_context_tokens;
        concrete.sampling_temperature = config.sampling.temperature;
        concrete.sampling_seed = config.sampling.seed;
        concrete.verbose = config.verbose;
        concrete.load_progress = config.load_progress;
        concrete.placement = placement;
        result = runtime.initialize(model_directory, concrete);
        if (result.ok()) impl_->runtime.emplace<KimiK3Runtime>(std::move(runtime));
        return result;
    }
    if (config.model == RuntimeModel::Gemma4) {
        if (const auto* control = deepseek_only_control(config)) {
            result.errors.emplace_back(
                std::string("DeepSeek ") + control +
                " cannot be used by the Gemma 4 runtime");
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
        concrete.placement = placement;
        result = runtime.initialize(model_directory, concrete);
        if (result.ok()) impl_->runtime.emplace<Gemma4Runtime>(std::move(runtime));
        return result;
    }
    if (config.model == RuntimeModel::Inkling) {
        if (const auto* control = deepseek_only_control(config)) {
            result.errors.emplace_back(
                std::string("DeepSeek ") + control +
                " cannot be used by the Inkling runtime");
            return result;
        }
        InklingRuntime runtime;
        InklingRuntimeConfig concrete;
        concrete.maximum_context_tokens = config.maximum_context_tokens;
        concrete.sampling_temperature = config.sampling.temperature;
        concrete.sampling_seed = config.sampling.seed;
        concrete.verbose = config.verbose;
        concrete.load_progress = config.load_progress;
        result = runtime.initialize(model_directory, concrete);
        if (result.ok()) impl_->runtime.emplace<InklingRuntime>(std::move(runtime));
        return result;
    }
    if (config.model == RuntimeModel::Laguna) {
        if (const auto* control = deepseek_only_control(config)) {
            result.errors.emplace_back(
                std::string("DeepSeek ") + control +
                " cannot be used by the Laguna runtime");
            return result;
        }
        LagunaRuntime runtime;
        LagunaRuntimeConfig concrete;
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
        if (result.ok()) impl_->runtime.emplace<LagunaRuntime>(std::move(runtime));
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
    concrete.kv_cache_mode = config.deepseek_device_resident_runtime
        ? Dsv4KvCacheMode::PhysicalDevice
        : config.deepseek_block_kv_cache ? Dsv4KvCacheMode::Block
                                         : Dsv4KvCacheMode::ScalarOracle;
    concrete.kv_block_rows = config.deepseek_device_resident_runtime
        ? kDsv4PhysicalKvBlockRows : kDsv4KvBlockRows;
    if (config.deepseek_prefill_page_tokens != 0U) {
        concrete.prefill_page_tokens = config.deepseek_prefill_page_tokens;
    }
    if (config.deepseek_device_resident_runtime) {
        // The device-resident decode contract is a bundle, not a knob. Leaving
        // any member of it to the caller lets a run report the accepted
        // attention/mHC path while executing routed experts somewhere else,
        // which is what made this opt-in rather than a default. These are the
        // same implications strata-deepseek-run applies to the flag.
        concrete.enable_flash_attention = true;
        concrete.enable_gpu_lightning_indexer = false;
        concrete.enable_host_routed_moe = true;
        concrete.prepack_mhc_projection = false;
    }
    concrete.decode_topology = config.deepseek_rank_local_decode
        ? Dsv4DecodeTopology::RankLocalTp2
        : Dsv4DecodeTopology::Centralized;
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
    if (auto* runtime = std::get_if<InklingRuntime>(&impl_->runtime)) {
        auto concrete = runtime->generate_chat_stream(
            messages, options.maximum_new_tokens, options.sampling,
            options.stop, on_token);
        result.text = std::move(concrete.text);
        result.prompt_token_ids = std::move(concrete.prompt_token_ids);
        result.generated_token_ids = std::move(concrete.generated_token_ids);
        result.logprobs = std::move(concrete.logprobs);
        result.metrics.prompt_tokens = concrete.metrics.prompt_tokens;
        result.metrics.prefill_tokens = concrete.metrics.prefill_tokens;
        result.metrics.decode_tokens = concrete.metrics.decode_tokens;
        result.metrics.prefill_seconds = concrete.metrics.prefill_seconds;
        result.metrics.decode_seconds = concrete.metrics.decode_seconds;
        result.errors = std::move(concrete.errors);
        result.stopped = concrete.stopped;
        return result;
    }
    if (auto* runtime = std::get_if<KimiK3Runtime>(&impl_->runtime)) {
        auto concrete = runtime->generate_chat_stream(
            messages, options.maximum_new_tokens, options.sampling,
            options.stop, on_token);
        result.text = std::move(concrete.text);
        result.prompt_token_ids = std::move(concrete.prompt_token_ids);
        result.generated_token_ids = std::move(concrete.generated_token_ids);
        result.logprobs = std::move(concrete.logprobs);
        result.metrics.prompt_tokens = concrete.metrics.prompt_tokens;
        result.metrics.prefill_tokens = concrete.metrics.prefill_tokens;
        result.metrics.decode_tokens = concrete.metrics.decode_tokens;
        result.metrics.prefill_seconds = concrete.metrics.prefill_seconds;
        result.metrics.decode_seconds = concrete.metrics.decode_seconds;
        result.errors = std::move(concrete.errors);
        result.stopped = concrete.stopped;
        return result;
    }
    if (auto* runtime = std::get_if<LagunaRuntime>(&impl_->runtime)) {
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
