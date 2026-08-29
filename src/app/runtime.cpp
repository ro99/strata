#include "strata/app/runtime.hpp"

#include "strata/engine/load_report.hpp"
#include "strata/engine/placement.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <string>
#include <utility>

namespace strata {

struct RuntimeSession::Impl {
    std::unique_ptr<ModelExecutor> executor;
    SamplingOptions sampling;
    PlacementPlan placement;
    bool placement_ready{};
    // Contract section 8's load report. Started before placement resolution so
    // it covers header parsing and mapping, and closed at the first published
    // token, because on a lazily mapped checkpoint the load is nearly free and
    // the fault-in all lands in the first forward.
    LoadReport load_report;
    bool report_load{};
    bool load_report_emitted{};
};

namespace {

// Names the DeepSeek-only switch a request carries, or nullptr if it carries
// none. Rejected rather than ignored by every model whose registration does
// not claim them: a request for rank-local decode that quietly ran a
// centralized GLM would report the accepted path while executing a different
// one.
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
    const auto* registration = find_model(config.model);
    request.model = registration != nullptr ? registration->placement
                                            : PlacementModel::Gemma4;
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
    if (impl_->executor != nullptr) {
        result.errors.emplace_back("runtime session is already initialized");
        return result;
    }

    // Single admission point. A value outside the enum is legal for a scoped
    // enum with an explicit underlying type, so this is a real check and not a
    // formality; before Phase 4 an unregistered model fell through an if-chain
    // and was constructed as DeepSeek against whatever directory was supplied.
    const auto* registration = find_model(config.model);
    if (registration == nullptr) {
        result.errors.emplace_back(
            "unhandled runtime model: " +
            std::to_string(static_cast<unsigned>(config.model)));
        return result;
    }
    if (!registration->accepts_deepseek_controls) {
        if (const auto* control = deepseek_only_control(config)) {
            result.errors.emplace_back(
                std::string("DeepSeek ") + control + " cannot be used by the " +
                registration->name + " runtime");
            return result;
        }
    }
    impl_->sampling = config.sampling;
    impl_->report_load = config.report_placement_plan;
    impl_->load_report.begin(resolve_backing_storage(model_directory).disk);
    // Set before the first mark so `load` and `first-token` both carry a VRAM
    // sample; the difference between them is the device expert cache filling
    // during the first forward, which is the figure worth seeing.
    impl_->load_report.set_devices(config.devices);

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

    auto executor = registration->make();
    result = executor->initialize(model_directory, config, placement);
    // Commit only on success: a failed attempt leaves generation disabled and
    // a retry starts from a fresh implementation object.
    if (result.ok()) {
        impl_->executor = std::move(executor);
        impl_->load_report.mark("load");
        if (impl_->placement_ready) {
            impl_->load_report.set_storage_tier_bytes(
                impl_->placement.storage_resident_bytes);
            impl_->load_report.set_storage_tier_note(
                impl_->placement.io_dependent
                    ? "I/O dependent: steady-state decode reads the checkpoint"
                    : "resident: steady-state decode does not read the checkpoint");
        }
    }
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
    if (impl_->executor == nullptr) {
        result.errors.emplace_back("runtime session is not initialized");
        return result;
    }
    const bool has_images = std::any_of(
        messages.begin(), messages.end(), [](const ChatMessage& message) {
            return std::any_of(message.parts.begin(), message.parts.end(),
                [](const ChatContentPart& part) {
                    return part.kind == ChatContentKind::Image;
                });
        });
    if (has_images && !impl_->executor->accepts_images()) {
        result.errors.emplace_back(
            "this loaded model does not support image content");
        return result;
    }
    // The first published token closes the load report. Wrapping the callback
    // rather than timing around the whole call is deliberate: a generation runs
    // to many tokens, and the figure contract section 8 asks for is time to
    // *first* token.
    if (!impl_->load_report_emitted && impl_->load_report.started()) {
        impl_->load_report_emitted = true;
        bool first = true;
        const TokenStreamCallback wrapped =
            [&](std::uint32_t token, std::string_view text) {
                if (first) {
                    first = false;
                    impl_->load_report.mark("first-token");
                    if (impl_->report_load) {
                        std::cerr << impl_->load_report.render();
                    }
                }
                return on_token ? on_token(token, text) : true;
            };
        return impl_->executor->generate_chat_stream(messages, options, wrapped);
    }
    return impl_->executor->generate_chat_stream(messages, options, on_token);
}

}  // namespace strata
