#include "test.hpp"

#include "strata/kimi_k3_manifest.hpp"
#include "strata/model_adapter.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

std::string kimi_directory() {
    return (std::filesystem::path(STRATA_SOURCE_DIR) / "models/kimi-k3").string();
}

std::string load_config() {
    const auto path = std::filesystem::path(kimi_directory()) / "config.json";
    if (!std::filesystem::exists(path)) return {};
    std::ifstream file(path, std::ios::binary);
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Replaces the first occurrence of `from` with `to`. Used to mutate one field
// of the real config and confirm the contract check refuses it.
std::string mutate(std::string text, std::string_view from, std::string_view to) {
    const auto at = text.find(from);
    if (at == std::string::npos) return {};
    return text.replace(at, from.size(), to);
}

}  // namespace

TEST_CASE("Kimi-K3 layer schedule matches the checkpoint's own 1-based lists") {
    // full_attn_layers = [4, 8, ..., 88, 92, 93], 1-based, 24 entries.
    std::uint32_t full = 0U;
    std::uint32_t kda = 0U;
    for (std::uint32_t layer = 0U; layer < 93U; ++layer) {
        if (strata::kimi_k3_full_attention_layer(layer)) ++full; else ++kda;
        REQUIRE(strata::kimi_k3_full_attention_layer(layer) !=
                strata::kimi_k3_kda_layer(layer));
    }
    REQUIRE(full == 24U);
    REQUIRE(kda == 69U);
    // The paper's extra gated MLA layer at the end of the backbone: 92 and 93
    // are adjacent full-attention layers, which the plain "every fourth" rule
    // would not produce.
    REQUIRE(strata::kimi_k3_full_attention_layer(91U));
    REQUIRE(strata::kimi_k3_full_attention_layer(92U));
    REQUIRE(strata::kimi_k3_kda_layer(90U));
    // Layer 0 is the dense MLP prefix; every later layer is MoE.
    REQUIRE(!strata::kimi_k3_moe_layer(0U));
    REQUIRE(strata::kimi_k3_moe_layer(1U));
    REQUIRE(strata::kimi_k3_moe_layer(92U));
    REQUIRE(!strata::kimi_k3_moe_layer(93U));
}

TEST_CASE("Kimi-K3 model spec pins the routed-expert quantization at four bits") {
    const auto spec = strata::kimi_k3_mxfp4_spec();
    REQUIRE(strata::validate_kimi_k3_mxfp4(spec).ok());
    REQUIRE(spec.mixed_quantization.routed_experts.bits == 4U);
    REQUIRE(spec.mixed_quantization.routed_experts.group_size == 32U);
    REQUIRE(spec.mixed_quantization.routed_experts.symmetric);
    // The dense spine is not quantized at all: no layer has a quantized linear.
    REQUIRE(spec.mixed_quantization.quantized_linear_start_layer ==
            spec.layer_count);
    REQUIRE(spec.router.routed_experts == 896U);
    REQUIRE(spec.router.experts_per_token == 16U);
    REQUIRE(spec.router.selection == strata::RouterSelectionKind::NoAuxTc);
    REQUIRE(spec.router.scoring == strata::RouterScoreKind::Sigmoid);
    REQUIRE(spec.router.normalize_topk);
    REQUIRE(spec.router.selection_bias);
}

TEST_CASE("mutated Kimi-K3 model spec is rejected") {
    auto spec = strata::kimi_k3_mxfp4_spec();
    spec.router.experts_per_token = 8U;
    REQUIRE(!strata::validate_kimi_k3_mxfp4(spec).ok());

    spec = strata::kimi_k3_mxfp4_spec();
    spec.router.normalize_topk = false;
    REQUIRE(!strata::validate_kimi_k3_mxfp4(spec).ok());

    spec = strata::kimi_k3_mxfp4_spec();
    spec.mixed_quantization.routed_experts.bits = 2U;
    REQUIRE(!strata::validate_kimi_k3_mxfp4(spec).ok());
}

TEST_CASE("Kimi-K3 tensor classification separates KDA, MLA, and expert roles") {
    std::int32_t layer = 0;
    std::int32_t expert = 0;
    const auto classify = [&](std::string_view name) {
        return strata::classify_kimi_tensor(name, layer, expert);
    };

    // Layer 0 is KDA, so its shared-name projections belong to the KDA path.
    REQUIRE(classify("language_model.model.layers.0.self_attn.g_proj.weight") ==
            strata::KimiTensorRole::KdaAttention);
    REQUIRE(layer == 0);
    // Layer 3 (1-based 4) is the first gated MLA layer.
    REQUIRE(classify("language_model.model.layers.3.self_attn.o_proj.weight") ==
            strata::KimiTensorRole::MlaAttention);
    REQUIRE(layer == 3);
    REQUIRE(classify("language_model.model.layers.3.self_attn.kv_b_proj.weight") ==
            strata::KimiTensorRole::MlaAttention);
    REQUIRE(classify("language_model.model.layers.2.self_attn.A_log") ==
            strata::KimiTensorRole::KdaAttention);
    REQUIRE(classify("language_model.model.layers.2.self_attn.q_conv1d.weight") ==
            strata::KimiTensorRole::KdaAttention);

    REQUIRE(classify(
                "language_model.model.layers.1.block_sparse_moe.experts.895."
                "w2.weight_packed") == strata::KimiTensorRole::RoutedExpert);
    REQUIRE(layer == 1);
    REQUIRE(expert == 895);
    REQUIRE(classify("language_model.model.layers.1.block_sparse_moe."
                     "routed_expert_up_proj.weight") ==
            strata::KimiTensorRole::LatentMoeProjection);
    REQUIRE(classify("language_model.model.layers.1.block_sparse_moe."
                     "shared_experts.gate_proj.weight") ==
            strata::KimiTensorRole::SharedExpert);
    REQUIRE(classify("language_model.model.layers.1.block_sparse_moe.gate."
                     "e_score_correction_bias") == strata::KimiTensorRole::Router);
    REQUIRE(classify("language_model.model.layers.0.mlp.gate_proj.weight") ==
            strata::KimiTensorRole::DenseMlp);

    // All three attention-residual sites, including the backbone output one.
    REQUIRE(classify("language_model.model.layers.5.self_attention_res_proj.weight") ==
            strata::KimiTensorRole::AttentionResidual);
    REQUIRE(classify("language_model.model.layers.5.mlp_res_norm.weight") ==
            strata::KimiTensorRole::AttentionResidual);
    REQUIRE(classify("language_model.model.output_attn_res_proj.weight") ==
            strata::KimiTensorRole::AttentionResidual);

    REQUIRE(classify("vision_tower.encoder.blocks.0.wqkv.weight") ==
            strata::KimiTensorRole::Vision);
    REQUIRE(classify("mm_projector.proj.0.weight") ==
            strata::KimiTensorRole::VisionProjector);
    REQUIRE(classify("language_model.lm_head.weight") ==
            strata::KimiTensorRole::OutputHead);
    REQUIRE(classify("nonsense.tensor") == strata::KimiTensorRole::Count);
}

TEST_CASE("real Kimi-K3 config matches the pinned execution contract") {
    const auto text = load_config();
    if (text.empty()) SKIP("pinned Kimi-K3 checkpoint is absent");
    const auto parsed = strata::parse_kimi_k3_config(text);
    REQUIRE(parsed.ok());
    const auto& config = parsed.value;
    const auto& contract = strata::kKimiK3ExecutionContract;

    REQUIRE(config.hidden_size == contract.hidden_size);
    REQUIRE(config.layer_count == contract.layer_count);
    REQUIRE(config.routed_experts == contract.routed_experts);
    REQUIRE(config.experts_per_token == contract.experts_per_token);
    REQUIRE(config.routed_expert_hidden_size == contract.routed_expert_hidden_size);
    REQUIRE(config.expert_intermediate_size == contract.expert_intermediate_size);
    REQUIRE(config.hidden_activation == "situ");
    REQUIRE(config.topk_method == "noaux_tc");
    REQUIRE(config.quantization_format == "mxfp4-pack-quantized");
    REQUIRE(config.quantization_bits == 4U);
    REQUIRE(config.quantization_group_size == 32U);
    REQUIRE(config.mla_use_nope);
    REQUIRE(config.mla_use_output_gate);
    REQUIRE(config.latent_moe_use_norm);
    REQUIRE(config.use_full_rank_gate);
    REQUIRE(!config.tie_word_embeddings);

    // The layer lists arrive 1-based and are rebased on parse. Check both the
    // count and the endpoints so an off-by-one cannot pass.
    REQUIRE(config.full_attention_layers.size() == 24U);
    REQUIRE(config.kda_layers.size() == 69U);
    REQUIRE(config.full_attention_layers.front() == 3U);
    REQUIRE(config.full_attention_layers.back() == 92U);
    REQUIRE(config.kda_layers.front() == 0U);
    REQUIRE(config.kda_layers.back() == 90U);

    REQUIRE(strata::validate_kimi_k3_config(config).ok());
}

TEST_CASE("mutated Kimi-K3 config is rejected field by field") {
    const auto text = load_config();
    if (text.empty()) SKIP("pinned Kimi-K3 checkpoint is absent");

    // Each mutation is one the charter names as forbidden to change silently:
    // expert count, top-k, precision, router semantics, and the activation.
    const std::pair<std::string_view, std::string_view> mutations[] = {
        {"\"num_experts\": 896", "\"num_experts\": 512"},
        {"\"num_experts_per_token\": 16", "\"num_experts_per_token\": 8"},
        {"\"num_bits\": 4", "\"num_bits\": 2"},
        {"\"hidden_act\": \"situ\"", "\"hidden_act\": \"silu\""},
        {"\"topk_method\": \"noaux_tc\"", "\"topk_method\": \"greedy\""},
        {"\"moe_renormalize\": true", "\"moe_renormalize\": false"},
        {"\"mla_use_nope\": true", "\"mla_use_nope\": false"},
        {"\"latent_moe_use_norm\": true", "\"latent_moe_use_norm\": false"},
        {"\"gate_lower_bound\": -5.0", "\"gate_lower_bound\": -3.0"},
        {"\"group_size\": 32", "\"group_size\": 64"},
    };
    for (const auto& [from, to] : mutations) {
        const auto mutated = mutate(text, from, to);
        REQUIRE(!mutated.empty());
        const auto parsed = strata::parse_kimi_k3_config(mutated);
        REQUIRE(parsed.ok());
        REQUIRE(!strata::validate_kimi_k3_config(parsed.value).ok());
    }
}

TEST_CASE("Kimi-K3 config with a rearranged hybrid schedule is rejected") {
    const auto text = load_config();
    if (text.empty()) SKIP("pinned Kimi-K3 checkpoint is absent");
    // Move one full-attention layer off its contracted position. The lists
    // still partition the backbone, so only the per-layer check catches it.
    const auto mutated = mutate(text, "\n        4,\n        8,",
                                "\n        5,\n        8,");
    REQUIRE(!mutated.empty());
    const auto parsed = strata::parse_kimi_k3_config(mutated);
    REQUIRE(parsed.ok());
    REQUIRE(!strata::validate_kimi_k3_config(parsed.value).ok());
}
