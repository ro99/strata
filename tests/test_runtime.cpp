#include "test.hpp"

#include "strata/runtime.hpp"
#include "strata/runtime_support.hpp"

#include <algorithm>

TEST_CASE("common runtime validation rejects duplicate devices") {
    const std::vector<int> devices{0, 0};
    const auto validation = strata::validate_common_runtime_config(
        devices, 0.85, 0.0, "test");
    REQUIRE(!validation.ok());
    REQUIRE(std::any_of(validation.errors.begin(), validation.errors.end(),
                        [](const std::string& error) {
                            return error.find("unique") != std::string::npos;
                        }));
}

TEST_CASE("explicit runtime VRAM budget applies the stricter bound") {
    const auto capped = strata::compute_runtime_device_budget(
        25'007'554'560ULL, 0.95, 21'256'421'376ULL);
    REQUIRE(capped.fractional_budget_bytes == 23'757'176'832ULL);
    REQUIRE(capped.explicit_budget_bytes == 21'256'421'376ULL);
    REQUIRE(capped.applied_budget_bytes == 21'256'421'376ULL);
    REQUIRE(strata::runtime_budget_bound_name(
                capped.fractional_budget_bytes,
                capped.explicit_budget_bytes) == "explicit");

    const auto unchanged = strata::compute_runtime_device_budget(
        25'007'554'560ULL, 0.95, 0U);
    REQUIRE(unchanged.applied_budget_bytes ==
            unchanged.fractional_budget_bytes);
    REQUIRE(strata::runtime_budget_bound_name(
                unchanged.fractional_budget_bytes,
                unchanged.explicit_budget_bytes) == "fractional");

    const auto tie = strata::compute_runtime_device_budget(
        25'007'554'560ULL, 0.95, 23'757'176'832ULL);
    REQUIRE(strata::runtime_budget_bound_name(
                tie.fractional_budget_bytes, tie.explicit_budget_bytes) ==
            "equal");
}

TEST_CASE("incremental KV reuse requires a strict exact token prefix") {
    const std::vector<std::uint32_t> cached{1U, 2U, 3U};
    const std::vector<std::uint32_t> extended{1U, 2U, 3U, 4U, 5U};
    const std::vector<std::uint32_t> unchanged{1U, 2U, 3U};
    const std::vector<std::uint32_t> truncated{1U, 2U};
    const std::vector<std::uint32_t> changed{1U, 2U, 9U, 4U};
    const std::vector<std::uint32_t> empty;

    REQUIRE(strata::incremental_kv_prefix_tokens(cached, extended) == 3U);
    REQUIRE(strata::incremental_kv_prefix_tokens(cached, unchanged) == 0U);
    REQUIRE(strata::incremental_kv_prefix_tokens(cached, truncated) == 0U);
    REQUIRE(strata::incremental_kv_prefix_tokens(cached, changed) == 0U);
    REQUIRE(strata::incremental_kv_prefix_tokens(empty, extended) == 0U);
}

TEST_CASE("runtime session remains fresh after failed initialization") {
    strata::RuntimeSession runtime;
    strata::RuntimeConfig config;
    config.model = strata::RuntimeModel::DeepSeekV4;
    const auto missing = runtime.initialize("not-present", config);
    REQUIRE(!missing.ok());

    config.maximum_context_tokens = 1'048'577U;
    const auto invalid = runtime.initialize("not-present", config);
    REQUIRE(!invalid.ok());
    REQUIRE(std::any_of(invalid.errors.begin(), invalid.errors.end(),
                        [](const std::string& error) {
                            return error.find("model limit") != std::string::npos;
                        }));
}

TEST_CASE("common runtime rejects DeepSeek cache controls for GLM") {
    strata::RuntimeSession runtime;
    strata::RuntimeConfig config;
    config.deepseek_block_kv_cache = true;
    const auto initialized = runtime.initialize("not-used", config);
    REQUIRE(!initialized.ok());
    REQUIRE(std::any_of(initialized.errors.begin(), initialized.errors.end(),
                        [](const std::string& error) {
                            return error.find("DeepSeek block KV") !=
                                   std::string::npos;
                        }));
}

// A DeepSeek-only decode control must never be silently ignored by another
// runtime: a request for rank-local decode that quietly runs a centralized GLM
// would report the accepted path while executing a different one.
TEST_CASE("common runtime rejects DeepSeek decode controls for GLM") {
    const auto rejected = [](void (*apply)(strata::RuntimeConfig&),
                             std::string_view expected) {
        strata::RuntimeSession runtime;
        strata::RuntimeConfig config;
        apply(config);
        const auto initialized = runtime.initialize("not-used", config);
        REQUIRE(!initialized.ok());
        REQUIRE(std::any_of(initialized.errors.begin(), initialized.errors.end(),
                            [expected](const std::string& error) {
                                return error.find(expected) !=
                                       std::string::npos;
                            }));
    };
    rejected([](strata::RuntimeConfig& config) {
        config.deepseek_device_resident_runtime = true;
    }, "DeepSeek device-resident runtime");
    rejected([](strata::RuntimeConfig& config) {
        config.deepseek_rank_local_decode = true;
    }, "DeepSeek rank-local decode");
}

// The placement plan must describe the layout the runtime will build, not the
// bare flags: the device-resident contract implies both of these.
TEST_CASE("device-resident runtime implies its placement request flags") {
    strata::RuntimeConfig config;
    config.model = strata::RuntimeModel::DeepSeekV4;
    config.deepseek_device_resident_runtime = true;
    const auto request = strata::placement_request_for("not-used", config);
    REQUIRE(request.block_kv_cache);
    REQUIRE(request.flash_attention);
}

TEST_CASE("runtime session cannot generate before initialization") {
    strata::RuntimeSession runtime;
    REQUIRE(!runtime.generate_stream("hello", 1U).ok());
}

// Free test 1 (Phase 2 survey, brief 04 task 2): the only two conformance
// properties assertable across all six RuntimeModel values without a real
// checkpoint. A failed initialize() must never leave impl_ holding a runtime
// -- the facade falls through to std::monostate, so generation afterward
// must report the same "not initialized" error every model gets from a
// session that was never touched, not silently produce output or crash.
TEST_CASE("initialization failure leaves generation disabled, all six models") {
    constexpr std::array<strata::RuntimeModel, 6> models{
        strata::RuntimeModel::Glm52,     strata::RuntimeModel::DeepSeekV4,
        strata::RuntimeModel::Gemma4,    strata::RuntimeModel::Laguna,
        strata::RuntimeModel::Inkling,   strata::RuntimeModel::KimiK3,
    };
    for (const auto model : models) {
        strata::RuntimeSession runtime;
        strata::RuntimeConfig config;
        config.model = model;
        const auto initialized = runtime.initialize("not-present", config);
        REQUIRE(!initialized.ok());
        const auto generated = runtime.generate_stream("hello", 1U);
        REQUIRE(!generated.ok());
        REQUIRE(std::any_of(generated.errors.begin(), generated.errors.end(),
                            [](const std::string& error) {
                                return error.find("not initialized") !=
                                       std::string::npos;
                            }));
    }
}

// Free test 2 / Phase 2, B7: a RuntimeModel value outside the declared six
// (only reachable via an explicit cast -- every real caller derives the enum
// from a closed switch or ternary chain) must be rejected by name, not
// quietly executed as DeepSeek against whatever directory was supplied.
// Red for the right reason: before the fix, none of the if-conditions in
// RuntimeSession::initialize matched RuntimeModel(99), so it fell through to
// the unconditional trailing block and attempted to open "definitely-not-a-
// real-checkpoint-directory" as a DeepSeek checkpoint -- producing a real
// error, but one about a missing/invalid checkpoint, never one naming the
// model as unhandled. A test that only checked `!result.ok()` would have
// passed for the wrong reason on the old code; checking the error text is
// what makes this catch the actual defect.
TEST_CASE("an unhandled runtime model is rejected, not silently run as DeepSeek") {
    strata::RuntimeSession runtime;
    strata::RuntimeConfig config;
    config.model = static_cast<strata::RuntimeModel>(99);
    const auto initialized =
        runtime.initialize("definitely-not-a-real-checkpoint-directory", config);
    REQUIRE(!initialized.ok());
    REQUIRE(std::any_of(initialized.errors.begin(), initialized.errors.end(),
                        [](const std::string& error) {
                            return error.find("unhandled") != std::string::npos;
                        }));
}

// Phase 2, B6: GenerationMetrics must be able to say "not applicable" for a
// runtime that does not implement incremental prefix reuse at all (Inkling,
// Kimi-K3), distinct from "measured, and the answer is none reused". A
// hardcoded 0/false default cannot express that distinction; std::optional
// can. Pure struct-shape check: no runtime, no checkpoint, exactly the
// "cheap because honest" property the schema fix was chosen for.
TEST_CASE("generation metrics can express reuse fields as not applicable") {
    strata::GenerationMetrics metrics;
    REQUIRE(!metrics.reused_prompt_tokens.has_value());
    REQUIRE(!metrics.incremental_kv_continuation.has_value());
    metrics.reused_prompt_tokens = 4U;
    metrics.incremental_kv_continuation = true;
    REQUIRE(metrics.reused_prompt_tokens.value() == 4U);
    REQUIRE(metrics.incremental_kv_continuation.value());
}
