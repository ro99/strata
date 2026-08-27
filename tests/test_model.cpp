#include "test.hpp"

#include "strata/models/common/model.hpp"

TEST_CASE("precision below int4 is rejected") {
    strata::ModelSpec model;
    model.name = "forbidden-q3";
    model.quant_bits = 3;
    model.hidden_size = 4096;
    model.layer_count = 32;
    const auto result = strata::validate_model(model);
    REQUIRE(!result.ok());
}

TEST_CASE("dense model has no router") {
    strata::ModelSpec model;
    model.name = "dense";
    model.quant_bits = 4;
    model.hidden_size = 4096;
    model.layer_count = 32;
    REQUIRE(strata::validate_model(model).ok());
    model.router.routed_experts = 8;
    REQUIRE(!strata::validate_model(model).ok());
}

TEST_CASE("DeepSeek requires MLA shared experts and explicit routing") {
    strata::ModelSpec model;
    model.name = "deepseek-test";
    model.architecture = strata::ArchitectureKind::DeepSeek;
    model.attention = strata::AttentionKind::Mla;
    model.quant_bits = 4;
    model.hidden_size = 7168;
    model.layer_count = 61;
    model.dense_prefix_layers = 3;
    model.shared_experts = 1;
    model.expert_intermediate_size = 2048;
    model.router.selection = strata::RouterSelectionKind::NoAuxTc;
    model.router.scoring = strata::RouterScoreKind::SqrtSoftplus;
    model.router.routed_experts = 256;
    model.router.experts_per_token = 8;
    model.router.groups = 8;
    model.router.selected_groups = 4;
    REQUIRE(strata::validate_model(model).ok());
    model.shared_experts = 0;
    REQUIRE(!strata::validate_model(model).ok());
}

TEST_CASE("GLM W4A16 contract is valid") {
    const auto model = strata::glm52_w4a16_spec();
    REQUIRE(strata::validate_model(model).ok());
    REQUIRE(strata::validate_glm52_w4a16(model).ok());
    REQUIRE(model.mixed_quantization.routed_experts.bits == 4);
    REQUIRE(model.mixed_quantization.routed_experts.group_size == 128);
    REQUIRE(model.mixed_quantization.linears.bits == 4);
    REQUIRE(model.mixed_quantization.quantized_linear_start_layer == 0);
    REQUIRE(model.mixed_quantization.quantized_expert_start_layer == 3);
    REQUIRE(model.mixed_quantization.mtp_layer_index == 78);
    REQUIRE(model.mixed_quantization.mtp.bits == 4);
    REQUIRE(model.source.main_shards == 8);
    REQUIRE(model.source.mtp_shards == 0);
}

TEST_CASE("GLM W4A16 rejects silent precision and manifest changes") {
    auto model = strata::glm52_w4a16_spec();
    model.mixed_quantization.routed_experts.bits = 3;
    REQUIRE(!strata::validate_glm52_w4a16(model).ok());

    model = strata::glm52_w4a16_spec();
    model.mixed_quantization.routed_experts.group_size = 64;
    REQUIRE(!strata::validate_glm52_w4a16(model).ok());

    model = strata::glm52_w4a16_spec();
    --model.source.tensor_count;
    REQUIRE(!strata::validate_glm52_w4a16(model).ok());

    model = strata::glm52_w4a16_spec();
    model.router.experts_per_token = 4;
    REQUIRE(!strata::validate_glm52_w4a16(model).ok());

    model = strata::glm52_w4a16_spec();
    model.router.scoring = strata::RouterScoreKind::Softmax;
    REQUIRE(!strata::validate_glm52_w4a16(model).ok());
}

TEST_CASE("DeepSeek V4 Flash 0731 rejects the preview source identity") {
    auto model = strata::deepseek_v4_flash_0731_spec();
    REQUIRE(strata::validate_deepseek_v4_flash_0731(model).ok());
    REQUIRE(model.name == "deepseek-ai/DeepSeek-V4-Flash-0731");
    REQUIRE(model.source.revision ==
            "9e165c30e2704aec5d9d593cce3eebd58bbef1cb");

    model.source.repository = "deepseek-ai/DeepSeek-V4-Flash-DSpark";
    REQUIRE(!strata::validate_deepseek_v4_flash_0731(model).ok());
}

TEST_CASE("Gemma 4 31B contract pins hybrid attention vision and W8A16") {
    auto model = strata::gemma4_31b_it_w8a16_spec();
    REQUIRE(strata::validate_gemma4_31b_it_w8a16(model).ok());
    REQUIRE(model.architecture == strata::ArchitectureKind::Gemma4);
    REQUIRE(model.attention == strata::AttentionKind::HybridLocalGlobal);
    REQUIRE(model.mixed_quantization.linears.bits == 8U);
    REQUIRE(model.mixed_quantization.linears.group_size == 32U);
    REQUIRE(model.gemma4.global_key_equals_value);
    REQUIRE(model.gemma4.vision_bidirectional);

    model.gemma4.global_rope_proportion = 1.0F;
    REQUIRE(!strata::validate_gemma4_31b_it_w8a16(model).ok());
}
