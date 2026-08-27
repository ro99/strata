#include "test.hpp"

#include "strata/engine/model_executor.hpp"
#include "strata/models/common/tokenizer.hpp"
#include "strata/models/glm53/glm53_manifest.hpp"

#include <array>
#include <cstdint>
#include <string_view>

TEST_CASE("GLM-5.3 schedule partitions KDA, sparse MLA, dense, and MoE layers") {
    std::uint32_t kda = 0U;
    std::uint32_t sparse = 0U;
    std::uint32_t dense = 0U;
    std::uint32_t moe = 0U;
    for (std::uint32_t layer = 0U; layer < 45U; ++layer) {
        if (strata::glm53_kda_layer(layer)) ++kda; else ++sparse;
        if (strata::glm53_moe_layer(layer)) ++moe; else ++dense;
    }
    REQUIRE(kda == 34U);
    REQUIRE(sparse == 11U);
    REQUIRE(dense == 3U);
    REQUIRE(moe == 42U);
    REQUIRE(strata::glm53_full_attention_layer(3U));
    REQUIRE(strata::glm53_kda_layer(44U));
}

TEST_CASE("GLM-5.3 tensor classifier separates text, MTP, and vision") {
    std::int32_t layer = -1;
    std::int32_t expert = -1;
    const auto classify = [&](std::string_view name) {
        return strata::classify_glm53_tensor(name, layer, expert);
    };
    REQUIRE(classify("model.language_model.layers.0.self_attn.A_log") ==
            strata::Glm53TensorRole::KdaAttention);
    REQUIRE(layer == 0);
    REQUIRE(classify("model.language_model.layers.3.self_attn.q_a_proj.weight") ==
            strata::Glm53TensorRole::SparseAttention);
    REQUIRE(classify("model.language_model.layers.3.self_attn.indexer.wk.weight") ==
            strata::Glm53TensorRole::AttentionIndexer);
    REQUIRE(classify("model.language_model.layers.2.mlp.gate_proj.weight") ==
            strata::Glm53TensorRole::DenseMlp);
    REQUIRE(classify("model.language_model.layers.10.mlp.experts.287.up_proj.weight") ==
            strata::Glm53TensorRole::RoutedExpert);
    REQUIRE(expert == 287);
    REQUIRE(classify("model.language_model.layers.10.mlp.shared_experts.down_proj.weight") ==
            strata::Glm53TensorRole::SharedExpert);
    REQUIRE(classify("model.language_model.layers.45.shared_head.norm.weight") ==
            strata::Glm53TensorRole::Mtp);
    REQUIRE(classify("model.visual.blocks.0.attn.qkv.weight") ==
            strata::Glm53TensorRole::Vision);
}

TEST_CASE("GLM-5.3 registers as a text-only chat and server model") {
    const auto* model = strata::find_model(strata::RuntimeModel::Glm53);
    REQUIRE(model != nullptr);
    REQUIRE(std::string_view(model->cli_name) == "glm53");
    REQUIRE(model->placement == strata::PlacementModel::Glm53);
    const auto executor = model->make();
    REQUIRE(executor != nullptr);
    REQUIRE(!executor->accepts_images());
}

TEST_CASE("GLM-5.3 chat rendering matches its text-only Jinja contract") {
    REQUIRE(strata::render_glm53_user_prompt("x") ==
            "[gMASK]<sop><|system|>Reasoning Effort: Max"
            "<|user|>x<|assistant|><think>");
    const std::array messages{
        strata::ChatMessage{strata::ChatRole::User, "x"},
        strata::ChatMessage{strata::ChatRole::Assistant, " answer "},
        strata::ChatMessage{strata::ChatRole::User, "y"},
    };
    REQUIRE(strata::render_glm53_chat_prompt(messages, "max", true) ==
            "[gMASK]<sop><|system|>Reasoning Effort: Max"
            "<|user|>x<|assistant|><think></think>answer"
            "<|user|>y<|assistant|><think>");
    REQUIRE(strata::render_glm53_chat_prompt(
                std::span<const strata::ChatMessage>(messages).first(1U),
                "unsupported", true) ==
            "[gMASK]<sop><|system|>Reasoning Effort: Max"
            "<|user|>x<|assistant|><think>");
}
