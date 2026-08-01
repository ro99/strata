#include "test.hpp"

#include "strata/gemma4_checkpoint.hpp"
#include "strata/model_adapter.hpp"

#include <filesystem>

TEST_CASE("real Gemma 4 31B W8A16 checkpoint validates target tensors") {
    const auto path = std::filesystem::path(STRATA_SOURCE_DIR) / "models/gemma4";
    if (!std::filesystem::exists(path / "model.safetensors.index.json")) {
        SKIP("pinned Gemma 4 checkpoint is absent");
    }
    const auto checkpoint = strata::Gemma4CheckpointReader::open(path.string());
    REQUIRE(checkpoint.ok());
    REQUIRE(checkpoint.value->tensors().size() == 2008U);
    const auto* global_q = checkpoint.value->find(
        "model.language_model.layers.5.self_attn.q_proj.weight_packed");
    REQUIRE(global_q != nullptr);
    REQUIRE(global_q->shape == std::vector<std::uint64_t>({16384U, 1344U}));
    REQUIRE(checkpoint.value->find(
        "model.language_model.layers.5.self_attn.v_proj.weight_packed") == nullptr);
}
