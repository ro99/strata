#include "test.hpp"

#include "strata/deepseek_runtime.hpp"

#include <algorithm>
#include <string>

TEST_CASE("DeepSeek correctness diagnostics are opt-in") {
    const strata::Dsv4RuntimeConfig config;
    REQUIRE(!config.enable_device_moe);
    REQUIRE(!config.enable_logit_trace);
    REQUIRE(!config.enable_layer_hash_trace);
    REQUIRE(config.logit_trace_top_k == 20U);
    REQUIRE(config.host_attention_threads == 0U);
    REQUIRE(config.resident_read_workers == 8U);
    REQUIRE(config.spine_warmup_workers == 3U);
    REQUIRE(config.overlap_resident_warmup);
}

TEST_CASE("DeepSeek runtime rejects excessive resident read workers") {
    strata::DeepSeekV4Runtime runtime;
    strata::Dsv4RuntimeConfig config;
    config.resident_read_workers = 65U;
    const auto initialized = runtime.initialize("not-used", config);
    REQUIRE(!initialized.ok());
    REQUIRE(std::any_of(initialized.errors.begin(), initialized.errors.end(),
                        [](const std::string& error) {
                            return error.find("resident read worker") !=
                                   std::string::npos;
                        }));
}

TEST_CASE("DeepSeek runtime rejects excessive spine warmup workers") {
    strata::DeepSeekV4Runtime runtime;
    strata::Dsv4RuntimeConfig config;
    config.spine_warmup_workers = 65U;
    const auto initialized = runtime.initialize("not-used", config);
    REQUIRE(!initialized.ok());
    REQUIRE(std::any_of(initialized.errors.begin(), initialized.errors.end(),
                        [](const std::string& error) {
                            return error.find("spine warmup worker") !=
                                   std::string::npos;
                        }));
}

TEST_CASE("DeepSeek runtime rejects excessive host attention workers") {
    strata::DeepSeekV4Runtime runtime;
    strata::Dsv4RuntimeConfig config;
    config.host_attention_threads = 65U;
    const auto initialized = runtime.initialize("not-used", config);
    REQUIRE(!initialized.ok());
    REQUIRE(std::any_of(initialized.errors.begin(), initialized.errors.end(),
                        [](const std::string& error) {
                            return error.find("attention worker") !=
                                   std::string::npos;
                        }));
}

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
