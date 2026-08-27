#include "test.hpp"

#include "strata/app/model_catalog.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>

#include <unistd.h>

namespace {

struct CatalogFixture {
    std::filesystem::path root = std::filesystem::temp_directory_path() /
        ("strata-catalog-" + std::to_string(getpid()));

    CatalogFixture() {
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root / "models" / "deepseek");
        std::filesystem::create_directories(root / "models" / "gemma");
    }

    ~CatalogFixture() { std::filesystem::remove_all(root); }

    std::filesystem::path write(std::string_view text) const {
        const auto path = root / "models.ini";
        std::ofstream output(path);
        output << text;
        return path;
    }
};

bool has_pair(const std::vector<std::string>& values,
              const std::string& key, const std::string& value) {
    for (std::size_t index = 1U; index < values.size(); ++index) {
        if (values[index - 1U] == key && values[index] == value) return true;
    }
    return false;
}

}  // namespace

TEST_CASE("model catalog resolves exact per-model launch contracts") {
    CatalogFixture fixture;
    const auto result = strata::load_model_catalog(fixture.write(R"(
version = 1

[*]
context-size = 8192
temperature = 0.8
top-p = 0.95

[deepseek-v4]
name = DeepSeek V4 — two 3090s
model = models/deepseek
model-type = deepseek
devices = 1,2
device-resident-runtime = true
decode-topology = rank-local-tp2
max-new = 2048
seed = 42

[gemma-4]
model = models/gemma
model-type = gemma4
devices = 0
temperature = 0.4
load-on-startup = true
stop-timeout = 20
)"));
    REQUIRE(result.ok());
    REQUIRE(result.value.models.size() == 2U);
    const auto& deepseek = result.value.models[0];
    REQUIRE(deepseek.id == "deepseek-v4");
    REQUIRE(deepseek.name == "DeepSeek V4 — two 3090s");
    REQUIRE(deepseek.model_directory ==
            (fixture.root / "models" / "deepseek").string());
    REQUIRE(deepseek.temperature == 0.8);
    REQUIRE(deepseek.top_p == 0.95);
    REQUIRE(deepseek.seed == 42U);
    REQUIRE(has_pair(deepseek.launch_arguments, "--devices", "1,2"));
    REQUIRE(has_pair(deepseek.launch_arguments, "--context-size", "8192"));
    REQUIRE(std::find(deepseek.launch_arguments.begin(),
                      deepseek.launch_arguments.end(),
                      "--device-resident-runtime") !=
            deepseek.launch_arguments.end());

    const auto& gemma = result.value.models[1];
    REQUIRE(gemma.id == "gemma-4");
    REQUIRE(gemma.temperature == 0.4);
    REQUIRE(gemma.load_on_startup);
    REQUIRE(gemma.stop_timeout_seconds == 20U);
}

TEST_CASE("model catalog rejects ambiguous or silently ignored settings") {
    CatalogFixture fixture;
    const auto result = strata::load_model_catalog(fixture.write(R"(
version = 1

[gemma-4]
model = models/gemma
model-type = gemma4
device-resident-runtime = true
temperature = 0.7
temperature = 0.8
made-up-option = yes
)"));
    REQUIRE(!result.ok());
    std::string joined;
    for (const auto& error : result.errors) joined += error + '\n';
    REQUIRE(joined.find("duplicate option 'temperature'") != std::string::npos);
    REQUIRE(joined.find("unknown option 'made-up-option'") != std::string::npos);
}
