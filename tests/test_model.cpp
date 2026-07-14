#include "test.hpp"

#include "strata/model.hpp"

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

TEST_CASE("pinned QuantTrio GLM-5.2 mixed-precision contract is valid") {
    const auto model = strata::quanttrio_glm52_int4_int8_mix_spec();
    REQUIRE(strata::validate_model(model).ok());
    REQUIRE(strata::validate_quanttrio_glm52_int4_int8_mix(model).ok());
    REQUIRE(model.mixed_quantization.routed_experts.bits == 4);
    REQUIRE(model.mixed_quantization.routed_experts.group_size == 128);
    REQUIRE(model.mixed_quantization.linears.bits == 8);
    REQUIRE(model.mixed_quantization.quantized_linear_start_layer == 1);
    REQUIRE(model.mixed_quantization.quantized_expert_start_layer == 3);
    REQUIRE(model.mixed_quantization.mtp_layer_index == 78);
    REQUIRE(model.mixed_quantization.mtp.granularity ==
            strata::QuantizationGranularity::Channel);
    REQUIRE(model.source.main_shards + model.source.mtp_shards == 128);
}

TEST_CASE("QuantTrio GLM-5.2 rejects silent precision and manifest changes") {
    auto model = strata::quanttrio_glm52_int4_int8_mix_spec();
    model.mixed_quantization.routed_experts.bits = 3;
    REQUIRE(!strata::validate_quanttrio_glm52_int4_int8_mix(model).ok());

    model = strata::quanttrio_glm52_int4_int8_mix_spec();
    model.mixed_quantization.routed_experts.group_size = 64;
    REQUIRE(!strata::validate_quanttrio_glm52_int4_int8_mix(model).ok());

    model = strata::quanttrio_glm52_int4_int8_mix_spec();
    --model.source.tensor_count;
    REQUIRE(!strata::validate_quanttrio_glm52_int4_int8_mix(model).ok());

    model = strata::quanttrio_glm52_int4_int8_mix_spec();
    model.router.experts_per_token = 4;
    REQUIRE(!strata::validate_quanttrio_glm52_int4_int8_mix(model).ok());

    model = strata::quanttrio_glm52_int4_int8_mix_spec();
    model.router.scoring = strata::RouterScoreKind::Softmax;
    REQUIRE(!strata::validate_quanttrio_glm52_int4_int8_mix(model).ok());
}
