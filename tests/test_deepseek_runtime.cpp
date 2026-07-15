#include "test.hpp"

#include "strata/deepseek_runtime.hpp"

#include <algorithm>
#include <string>

TEST_CASE("DeepSeek runtime rejects unverified DSpark execution explicitly") {
    strata::DeepSeekV4Runtime runtime;
    strata::Dsv4RuntimeConfig config;
    config.enable_dspark = true;
    const auto initialized = runtime.initialize("not-used", config);
    REQUIRE(!initialized.ok());
    REQUIRE(std::any_of(initialized.errors.begin(), initialized.errors.end(),
                        [](const std::string& error) {
                            return error.find("DSpark") != std::string::npos;
                        }));
}

TEST_CASE("DeepSeek runtime cannot generate before resident admission") {
    strata::DeepSeekV4Runtime runtime;
    const auto generated = runtime.generate("Hello", 1U);
    REQUIRE(!generated.ok());
}
