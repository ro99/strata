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

TEST_CASE("runtime session cannot generate before initialization") {
    strata::RuntimeSession runtime;
    REQUIRE(!runtime.generate_stream("hello", 1U).ok());
}
