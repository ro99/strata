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
    model.router.kind = strata::RouterKind::NoAuxTc;
    model.router.routed_experts = 256;
    model.router.experts_per_token = 8;
    model.router.groups = 8;
    model.router.selected_groups = 4;
    REQUIRE(strata::validate_model(model).ok());
    model.shared_experts = 0;
    REQUIRE(!strata::validate_model(model).ok());
}
