#include "test.hpp"

#include "strata/deepseek_runtime.hpp"

#include <algorithm>
#include <string>

TEST_CASE("DeepSeek fast exact execution defaults are enabled") {
    const strata::Dsv4RuntimeConfig config;
    REQUIRE(config.enable_device_moe);
    REQUIRE(!config.enable_flash_attention);
    REQUIRE(config.flash_attention_minimum_rows == 256U);
    REQUIRE(config.prefill_page_tokens == 64U);
    REQUIRE(!config.enable_logit_trace);
    REQUIRE(!config.enable_layer_hash_trace);
    REQUIRE(config.logit_trace_top_k == 20U);
    REQUIRE(config.host_attention_threads == 28U);
    REQUIRE(config.resident_read_workers == 8U);
    REQUIRE(config.spine_warmup_workers == 3U);
    REQUIRE(config.overlap_resident_warmup);
}

TEST_CASE("DeepSeek runtime rejects an unbounded prefill page") {
    strata::DeepSeekV4Runtime runtime;
    strata::Dsv4RuntimeConfig config;
    config.prefill_page_tokens = 513U;
    const auto initialized = runtime.initialize("not-used", config);
    REQUIRE(!initialized.ok());
    REQUIRE(std::any_of(initialized.errors.begin(), initialized.errors.end(),
                        [](const std::string& error) {
                            return error.find("prefill page") != std::string::npos;
                        }));
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

TEST_CASE("DeepSeek runtime accepts user context ceilings through the model limit") {
    strata::DeepSeekV4Runtime runtime;
    strata::Dsv4RuntimeConfig config;
    config.maximum_context_tokens = 1'048'576U;
    const auto accepted = runtime.initialize("not-used", config);
    REQUIRE(!accepted.ok());
    REQUIRE(std::none_of(accepted.errors.begin(), accepted.errors.end(),
                         [](const std::string& error) {
                             return error.find("context") != std::string::npos;
                         }));

    config.maximum_context_tokens = 1'048'577U;
    const auto rejected = runtime.initialize("not-used", config);
    REQUIRE(!rejected.ok());
    REQUIRE(std::any_of(rejected.errors.begin(), rejected.errors.end(),
                        [](const std::string& error) {
                            return error.find("model limit") != std::string::npos;
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
