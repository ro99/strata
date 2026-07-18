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

TEST_CASE("runtime session cannot generate before initialization") {
    strata::RuntimeSession runtime;
    REQUIRE(!runtime.generate_stream("hello", 1U).ok());
}
