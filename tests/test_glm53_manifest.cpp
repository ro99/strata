#include "test.hpp"

#include "strata/engine/model_executor.hpp"
#include "strata/models/common/tokenizer.hpp"
#include "strata/models/glm53/glm53_manifest.hpp"
#include "strata/models/glm53/glm53_expert_arena.hpp"
#include "strata/platform/hardware_profile.hpp"
#include "strata/models/glm53/glm53_runtime.hpp"
#include "strata/models/glm53/glm53_sequence.hpp"

#include <algorithm>
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

TEST_CASE("GLM-5.3 projection assignment is deterministic and capacity weighted") {
    constexpr std::array<std::string_view, 8U> keys{
        "q", "k", "v", "f", "b", "g", "up", "down"};
    constexpr std::array<std::uint64_t, 8U> costs{
        10U, 10U, 10U, 10U, 10U, 10U, 10U, 10U};
    constexpr std::array<std::uint64_t, 2U> equal{20U, 20U};
    const auto first = strata::glm53_projection_slots(keys, costs, equal, 1U);
    const auto second = strata::glm53_projection_slots(keys, costs, equal, 1U);
    REQUIRE(first == second);
    REQUIRE(std::count(first.begin(), first.end(), 0U) == 4);
    REQUIRE(std::count(first.begin(), first.end(), 1U) == 4);

    constexpr std::array<std::uint64_t, 2U> weighted{30U, 10U};
    const auto asymmetric =
        strata::glm53_projection_slots(keys, costs, weighted, 0U);
    REQUIRE(std::count(asymmetric.begin(), asymmetric.end(), 0U) == 6);
    REQUIRE(std::count(asymmetric.begin(), asymmetric.end(), 1U) == 2);
    constexpr std::array<std::uint64_t, 2U> invalid{30U, 0U};
    REQUIRE(strata::glm53_projection_slots(keys, costs, invalid, 0U).empty());
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

TEST_CASE("GLM-5.3 physical sequence pages fork copy-on-write") {
    strata::Glm53PagedRows rows(4U, 2U);
    constexpr std::array<float, 4U> first{1, 2, 3, 4};
    constexpr std::array<float, 4U> second{5, 6, 7, 8};
    constexpr std::array<float, 4U> third{9, 10, 11, 12};
    REQUIRE(rows.append(first).ok());
    REQUIRE(rows.append(second).ok());
    auto fork = rows;
    REQUIRE(rows.private_bytes() == 0U);
    REQUIRE(fork.private_bytes() == 0U);
    REQUIRE(fork.append(third).ok());
    REQUIRE(rows.rows() == 2U);
    REQUIRE(fork.rows() == 3U);
    REQUIRE(fork.row(0U)[0] == 1.0F);
    REQUIRE(fork.row(2U)[3] == 12.0F);
    REQUIRE(fork.private_bytes() != 0U);
}

TEST_CASE("GLM-5.3 recurrent state forks lazily and exactly") {
    strata::Glm53SequenceState state;
    REQUIRE(state.reset(128U, 16U).ok());
    auto recurrent = state.recurrent(0U);
    REQUIRE(!recurrent.empty());
    recurrent[17U] = 3.25F;
    auto fork = state;
    auto forked = fork.recurrent(0U);
    REQUIRE(forked[17U] == 3.25F);
    forked[17U] = -2.0F;
    REQUIRE(state.recurrent(0U)[17U] == 3.25F);
    REQUIRE(fork.recurrent(0U)[17U] == -2.0F);
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

TEST_CASE("GLM-5.3 accepts the FP8 and quark MXFP4 releases and nothing else") {
    strata::Glm53TextConfig config;
    config.quantization_method = "fp8";
    config.quantization_format = "e4m3";
    REQUIRE(strata::glm53_config_quantization(config) ==
            strata::Glm53Quantization::Fp8E4m3Block128);
    config = {};
    config.quantization_method = "quark";
    config.quantization_weight_dtype = "fp4";
    REQUIRE(strata::glm53_config_quantization(config) ==
            strata::Glm53Quantization::Mxfp4Group32);
    config.quantization_weight_dtype = "int4";
    REQUIRE(strata::glm53_config_quantization(config) ==
            strata::Glm53Quantization::Unsupported);
    config = {};
    REQUIRE(strata::glm53_config_quantization(config) ==
            strata::Glm53Quantization::Unsupported);
}

TEST_CASE("GLM-5.3 config parser reads quark's nested weight format") {
    const auto parsed = strata::parse_glm53_config(
        R"({"architectures":["Glm5NextForConditionalGeneration"],)"
        R"("quantization_config":{"quant_method":"quark","exclude":["a","b"],)"
        R"("global_quant_config":{"input_tensors":{"dtype":"fp4"},)"
        R"("weight":{"dtype":"fp4","group_size":32,"scale_format":"e8m0",)"
        R"("observer_cls":"PerBlockMXObserver"}}}})");
    REQUIRE(parsed.ok());
    REQUIRE(parsed.value.quantization_method == "quark");
    REQUIRE(parsed.value.quantization_weight_dtype == "fp4");
    REQUIRE(parsed.value.quantization_scale_format == "e8m0");
    REQUIRE(parsed.value.quantization_group_size == 32U);
    // The FP8 release's flat form must still parse into the same fields.
    const auto fp8 = strata::parse_glm53_config(
        R"({"quantization_config":{"quant_method":"fp8","fmt":"e4m3",)"
        R"("weight_block_size":[128,128]}})");
    REQUIRE(fp8.ok());
    REQUIRE(fp8.value.fp8_block_rows == 128U);
    REQUIRE(fp8.value.fp8_block_columns == 128U);
}

TEST_CASE("GLM-5.3 classifier places MXFP4 scale tensors with their module") {
    std::int32_t layer = -1;
    std::int32_t expert = -1;
    REQUIRE(strata::classify_glm53_tensor(
                "model.language_model.layers.20.mlp.experts.7.up_proj"
                ".weight_scale",
                layer, expert) == strata::Glm53TensorRole::RoutedExpert);
    REQUIRE(layer == 20);
    REQUIRE(expert == 7);
}

TEST_CASE("GLM-5.3 expert arena capacity counts only exact host payloads") {
    strata::Glm53IndexManifest manifest;
    strata::Glm53ManifestTensor routed;
    routed.name = "routed.weight";
    routed.role = strata::Glm53TensorRole::RoutedExpert;
    routed.component = strata::Glm53TensorComponent::Weight;
    routed.source_bytes = 65U;
    manifest.tensors.push_back(routed);

    strata::Glm53ManifestTensor shared;
    shared.name = "shared.weight_scale";
    shared.role = strata::Glm53TensorRole::SharedExpert;
    shared.component = strata::Glm53TensorComponent::Scale;
    shared.source_bytes = 64U;
    manifest.tensors.push_back(shared);

    auto mtp = routed;
    mtp.name = "model.language_model.layers.45.mlp.experts.0.gate_proj.weight";
    mtp.role = strata::Glm53TensorRole::Mtp;
    mtp.source_bytes = 64U;
    manifest.tensors.push_back(mtp);

    auto ignored = routed;
    ignored.name = "routed.bias";
    ignored.component = strata::Glm53TensorComponent::Bias;
    ignored.source_bytes = 1ULL << 30U;
    manifest.tensors.push_back(ignored);
    ignored.name = "norm.weight";
    ignored.role = strata::Glm53TensorRole::Norm;
    ignored.component = strata::Glm53TensorComponent::Weight;
    manifest.tensors.push_back(ignored);

    const auto reported = strata::host_hardware_profile().huge_page_bytes;
    const auto huge = reported == 0U ? 2ULL << 20U : reported;
    // 65 rounds to two 64-byte extents; each 64-byte expert occupies one.
    const std::uint64_t default_logical = 192U;
    const auto expected =
        (default_logical + huge - 1U) / huge * huge;
    REQUIRE(strata::Glm53ExpertArena::required_bytes(manifest) == expected);
    const std::uint64_t mtp_logical = 256U;
    const auto expected_with_mtp =
        (mtp_logical + huge - 1U) / huge * huge;
    REQUIRE(strata::Glm53ExpertArena::required_bytes(manifest, true) ==
            expected_with_mtp);

    manifest.tensors.clear();
    REQUIRE(strata::Glm53ExpertArena::required_bytes(manifest) == 0U);
}
