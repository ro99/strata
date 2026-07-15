#include "strata/model.hpp"

#include <algorithm>
#include <cmath>
#include <string_view>

namespace strata {

std::string_view to_string(Tier tier) noexcept {
    switch (tier) {
        case Tier::Vram: return "vram";
        case Tier::Ram: return "ram";
        case Tier::Peer: return "peer";
        case Tier::Nvme: return "nvme";
    }
    return "unknown";
}

std::string_view to_string(ReplacementPolicy policy) noexcept {
    switch (policy) {
        case ReplacementPolicy::Lru: return "lru";
        case ReplacementPolicy::Lfu: return "lfu";
        case ReplacementPolicy::Lease: return "lease";
    }
    return "unknown";
}

ValidationResult validate_model(const ModelSpec& spec) {
    ValidationResult result;
    if (spec.name.empty()) result.errors.emplace_back("model name is empty");
    if (!quantization_allowed(spec.quant_bits)) {
        result.errors.emplace_back("weight precision below four bits is forbidden");
    }
    if (spec.hidden_size == 0) result.errors.emplace_back("hidden_size must be positive");
    if (spec.layer_count == 0) result.errors.emplace_back("layer_count must be positive");
    if (spec.dense_prefix_layers > spec.layer_count) {
        result.errors.emplace_back("dense_prefix_layers exceeds layer_count");
    }
    if (!std::isfinite(spec.router.routed_scale) || spec.router.routed_scale <= 0.0F) {
        result.errors.emplace_back("routed_scale must be finite and positive");
    }

    if (spec.mixed_quantization.kind == QuantizationKind::CompressedTensorsInt4Int8Mix) {
        const auto validate_quantized_role = [&result](const QuantizedWeightSpec& role,
                                                        std::string_view name) {
            if (!quantization_allowed(role.bits)) {
                result.errors.emplace_back(std::string(name) +
                                           " precision below four bits is forbidden");
            }
            if (role.granularity == QuantizationGranularity::Group &&
                role.group_size == 0) {
                result.errors.emplace_back(std::string(name) +
                                           " group quantization requires group_size");
            }
            if (role.granularity == QuantizationGranularity::Channel &&
                role.group_size != 0) {
                result.errors.emplace_back(std::string(name) +
                                           " channel quantization cannot declare group_size");
            }
        };
        validate_quantized_role(spec.mixed_quantization.routed_experts,
                                "routed expert");
        validate_quantized_role(spec.mixed_quantization.linears, "linear");
        validate_quantized_role(spec.mixed_quantization.mtp, "MTP");
        const auto minimum_role_bits = std::min(
            {spec.mixed_quantization.routed_experts.bits,
             spec.mixed_quantization.linears.bits, spec.mixed_quantization.mtp.bits});
        if (spec.quant_bits != minimum_role_bits) {
            result.errors.emplace_back(
                "quant_bits must equal the minimum declared role precision");
        }
    }
    if (spec.mixed_quantization.kind == QuantizationKind::NativeFp4Fp8) {
        const auto& quantization = spec.native_fp4_fp8;
        if (!quantization_allowed(quantization.fp4_weight_bits) ||
            !quantization_allowed(quantization.fp8_weight_bits)) {
            result.errors.emplace_back("native FP4/FP8 precision is below four bits");
        }
        if (quantization.fp4_weight_bits != 4U ||
            quantization.fp8_weight_bits != 8U ||
            quantization.activation_bits != 8U) {
            result.errors.emplace_back("native FP4/FP8 encoding widths are unsupported");
        }
        if (quantization.fp4_group_size == 0U ||
            quantization.fp8_block_rows == 0U ||
            quantization.fp8_block_columns == 0U ||
            quantization.activation_group_size == 0U) {
            result.errors.emplace_back("native FP4/FP8 block dimensions must be positive");
        }
        if (spec.quant_bits != std::min(quantization.fp4_weight_bits,
                                       quantization.fp8_weight_bits)) {
            result.errors.emplace_back(
                "quant_bits must equal the minimum native FP4/FP8 precision");
        }
    }

    if (spec.architecture == ArchitectureKind::Dense) {
        if (spec.router.selection != RouterSelectionKind::None ||
            spec.router.scoring != RouterScoreKind::None ||
            spec.router.routed_experts != 0 || spec.router.experts_per_token != 0) {
            result.errors.emplace_back("dense architecture cannot declare routed experts");
        }
        if (spec.shared_experts != 0) {
            result.errors.emplace_back("dense architecture cannot declare shared experts");
        }
        return result;
    }

    if (spec.router.selection == RouterSelectionKind::None) {
        result.errors.emplace_back("MoE architecture requires explicit router selection");
    }
    if (spec.router.scoring == RouterScoreKind::None) {
        result.errors.emplace_back("MoE architecture requires an explicit scoring function");
    }
    if (spec.router.routed_experts == 0) {
        result.errors.emplace_back("MoE architecture requires routed experts");
    }
    if (spec.router.experts_per_token == 0 ||
        spec.router.experts_per_token > spec.router.routed_experts) {
        result.errors.emplace_back("experts_per_token is outside the routed expert range");
    }
    if (spec.expert_intermediate_size == 0) {
        result.errors.emplace_back("MoE architecture requires expert_intermediate_size");
    }
    if (spec.router.groups == 0 || spec.router.selected_groups == 0 ||
        spec.router.selected_groups > spec.router.groups) {
        result.errors.emplace_back("invalid router group selection");
    }

    if (spec.architecture == ArchitectureKind::DeepSeek) {
        if (spec.attention != AttentionKind::Mla &&
            spec.attention != AttentionKind::HybridCompressedSparse) {
            result.errors.emplace_back(
                "DeepSeek adapter requires MLA or hybrid compressed sparse attention");
        }
        if (spec.shared_experts == 0) {
            result.errors.emplace_back("DeepSeek adapter requires explicit shared experts");
        }
        if (spec.router.selection != RouterSelectionKind::NoAuxTc &&
            spec.router.selection != RouterSelectionKind::GroupLimitedTopK &&
            spec.router.selection != RouterSelectionKind::TopK) {
            result.errors.emplace_back("unsupported DeepSeek router semantics");
        }
    }

    if (spec.architecture == ArchitectureKind::GlmMoeDsa) {
        if (spec.attention != AttentionKind::Mla) {
            result.errors.emplace_back("GLM MoE DSA adapter requires MLA attention");
        }
        if (spec.shared_experts == 0) {
            result.errors.emplace_back(
                "GLM MoE DSA adapter requires explicit shared experts");
        }
        if (spec.router.selection != RouterSelectionKind::NoAuxTc) {
            result.errors.emplace_back("GLM MoE DSA adapter requires noaux_tc routing");
        }
        if (spec.glm_moe_dsa.sparse_attention_topk == 0) {
            result.errors.emplace_back("GLM MoE DSA adapter requires sparse attention top-k");
        }
        if (spec.glm_moe_dsa.index_share_frequency == 0) {
            result.errors.emplace_back("GLM MoE DSA adapter requires IndexShare frequency");
        }
        if (spec.glm_moe_dsa.mtp_layers == 0) {
            result.errors.emplace_back("GLM MoE DSA adapter requires explicit MTP layers");
        }
    }
    return result;
}

ModelSpec deepseek_v4_flash_dspark_spec() {
    ModelSpec spec;
    spec.name = "deepseek-ai/DeepSeek-V4-Flash-DSpark";
    spec.architecture = ArchitectureKind::DeepSeek;
    spec.attention = AttentionKind::HybridCompressedSparse;
    spec.router.selection = RouterSelectionKind::NoAuxTc;
    spec.router.scoring = RouterScoreKind::SqrtSoftplus;
    spec.router.routed_experts = 256U;
    spec.router.experts_per_token = 6U;
    spec.router.groups = 1U;
    spec.router.selected_groups = 1U;
    spec.router.normalize_topk = true;
    spec.router.selection_bias = true;
    spec.router.routed_scale = 1.5F;

    spec.mixed_quantization.kind = QuantizationKind::NativeFp4Fp8;
    spec.native_fp4_fp8 = {8U, 128U, 8U, 128U, 128U, 4U, 32U, true};
    spec.source.repository = "deepseek-ai/DeepSeek-V4-Flash-DSpark";
    spec.source.revision = "62af8fffb2f7030cac4de2f0169f5b8d1101b646";
    spec.source.index_sha256 =
        "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b";
    spec.source.tensor_count = 72'317U;
    spec.source.indexed_tensor_bytes = 166'878'536'440ULL;
    spec.source.shard_file_bytes = 166'886'535'336ULL;
    spec.source.main_shards = 48U;
    spec.source.mtp_shards = 0U;

    spec.quant_bits = 4U;
    spec.hidden_size = 4096U;
    spec.layer_count = 43U;
    spec.max_context_tokens = 1'048'576U;
    spec.dense_prefix_layers = 0U;
    spec.shared_experts = 1U;
    spec.expert_intermediate_size = 2048U;

    spec.deepseek_v4.attention_heads = 64U;
    spec.deepseek_v4.key_value_heads = 1U;
    spec.deepseek_v4.head_dim = 512U;
    spec.deepseek_v4.rope_head_dim = 64U;
    spec.deepseek_v4.query_lora_rank = 1024U;
    spec.deepseek_v4.output_lora_rank = 1024U;
    spec.deepseek_v4.output_groups = 8U;
    spec.deepseek_v4.sliding_window = 128U;
    spec.deepseek_v4.index_heads = 64U;
    spec.deepseek_v4.index_head_dim = 128U;
    spec.deepseek_v4.index_topk = 512U;
    spec.deepseek_v4.hash_layers = 3U;
    spec.deepseek_v4.mhc_multiplier = 4U;
    spec.deepseek_v4.mhc_sinkhorn_iterations = 20U;
    spec.deepseek_v4.dspark_layers = 3U;
    spec.deepseek_v4.dspark_block_size = 5U;
    spec.deepseek_v4.dspark_noise_token_id = 128'799U;
    spec.deepseek_v4.dspark_markov_rank = 256U;
    spec.deepseek_v4.swiglu_limit = 10.0F;
    spec.deepseek_v4.compression_ratios = {
        0U, 0U, 4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U,
        4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U,
        4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U,
        4U, 128U, 4U, 128U, 4U, 128U, 4U, 0U, 0U, 0U};
    spec.deepseek_v4.dspark_target_layers = {40U, 41U, 42U};
    return spec;
}

ValidationResult validate_deepseek_v4_flash_dspark(const ModelSpec& spec) {
    auto result = validate_model(spec);
    const auto expected = deepseek_v4_flash_dspark_spec();
    const auto require = [&result](bool condition, std::string_view message) {
        if (!condition) result.errors.emplace_back(message);
    };
    require(spec.source.repository == expected.source.repository &&
                spec.source.revision == expected.source.revision &&
                spec.source.index_sha256 == expected.source.index_sha256,
            "unexpected DeepSeek-V4-Flash-DSpark source identity");
    require(spec.source.tensor_count == expected.source.tensor_count &&
                spec.source.indexed_tensor_bytes == expected.source.indexed_tensor_bytes &&
                spec.source.shard_file_bytes == expected.source.shard_file_bytes &&
                spec.source.main_shards == expected.source.main_shards &&
                spec.source.mtp_shards == expected.source.mtp_shards,
            "unexpected DeepSeek-V4-Flash-DSpark checkpoint extent");
    require(spec.architecture == expected.architecture &&
                spec.attention == expected.attention &&
                spec.hidden_size == expected.hidden_size &&
                spec.layer_count == expected.layer_count &&
                spec.max_context_tokens == expected.max_context_tokens &&
                spec.shared_experts == expected.shared_experts &&
                spec.expert_intermediate_size == expected.expert_intermediate_size,
            "unexpected DeepSeek-V4-Flash-DSpark architecture dimensions");
    require(spec.router.selection == expected.router.selection &&
                spec.router.scoring == expected.router.scoring &&
                spec.router.routed_experts == expected.router.routed_experts &&
                spec.router.experts_per_token == expected.router.experts_per_token &&
                spec.router.normalize_topk == expected.router.normalize_topk &&
                spec.router.selection_bias == expected.router.selection_bias &&
                spec.router.routed_scale == expected.router.routed_scale,
            "unexpected DeepSeek-V4-Flash-DSpark router semantics");
    const auto& actual_quantization = spec.native_fp4_fp8;
    const auto& wanted_quantization = expected.native_fp4_fp8;
    require(spec.mixed_quantization.kind == QuantizationKind::NativeFp4Fp8 &&
                actual_quantization.activation_bits == wanted_quantization.activation_bits &&
                actual_quantization.activation_group_size ==
                    wanted_quantization.activation_group_size &&
                actual_quantization.fp8_weight_bits == wanted_quantization.fp8_weight_bits &&
                actual_quantization.fp8_block_rows == wanted_quantization.fp8_block_rows &&
                actual_quantization.fp8_block_columns ==
                    wanted_quantization.fp8_block_columns &&
                actual_quantization.fp4_weight_bits == wanted_quantization.fp4_weight_bits &&
                actual_quantization.fp4_group_size == wanted_quantization.fp4_group_size &&
                actual_quantization.power_of_two_scales ==
                    wanted_quantization.power_of_two_scales,
            "unexpected DeepSeek-V4-Flash-DSpark native quantization");
    const auto& actual = spec.deepseek_v4;
    const auto& wanted = expected.deepseek_v4;
    require(actual.attention_heads == wanted.attention_heads &&
                actual.key_value_heads == wanted.key_value_heads &&
                actual.head_dim == wanted.head_dim &&
                actual.rope_head_dim == wanted.rope_head_dim &&
                actual.query_lora_rank == wanted.query_lora_rank &&
                actual.output_lora_rank == wanted.output_lora_rank &&
                actual.output_groups == wanted.output_groups &&
                actual.sliding_window == wanted.sliding_window &&
                actual.index_heads == wanted.index_heads &&
                actual.index_head_dim == wanted.index_head_dim &&
                actual.index_topk == wanted.index_topk &&
                actual.hash_layers == wanted.hash_layers &&
                actual.mhc_multiplier == wanted.mhc_multiplier &&
                actual.mhc_sinkhorn_iterations == wanted.mhc_sinkhorn_iterations &&
                actual.dspark_layers == wanted.dspark_layers &&
                actual.dspark_block_size == wanted.dspark_block_size &&
                actual.dspark_noise_token_id == wanted.dspark_noise_token_id &&
                actual.dspark_markov_rank == wanted.dspark_markov_rank &&
                actual.swiglu_limit == wanted.swiglu_limit &&
                actual.compression_ratios == wanted.compression_ratios &&
                actual.dspark_target_layers == wanted.dspark_target_layers,
            "unexpected DeepSeek-V4-Flash-DSpark hybrid attention, mHC, or DSpark contract");
    return result;
}

ModelSpec quanttrio_glm52_int4_int8_mix_spec() {
    ModelSpec spec;
    spec.name = "QuantTrio/GLM-5.2-Int4-Int8Mix";
    spec.architecture = ArchitectureKind::GlmMoeDsa;
    spec.attention = AttentionKind::Mla;
    spec.router.selection = RouterSelectionKind::NoAuxTc;
    spec.router.scoring = RouterScoreKind::Sigmoid;
    spec.router.routed_experts = 256;
    spec.router.experts_per_token = 8;
    spec.router.groups = 1;
    spec.router.selected_groups = 1;
    spec.router.normalize_topk = true;
    spec.router.selection_bias = true;
    spec.router.routed_scale = 2.5F;

    spec.mixed_quantization.kind = QuantizationKind::CompressedTensorsInt4Int8Mix;
    spec.mixed_quantization.activation_bits = 16;
    spec.mixed_quantization.quantized_linear_start_layer = 1;
    spec.mixed_quantization.quantized_expert_start_layer = 3;
    spec.mixed_quantization.mtp_layer_index = 78;
    spec.mixed_quantization.routed_experts = {
        4, QuantizationGranularity::Group, 128, true};
    spec.mixed_quantization.linears = {
        8, QuantizationGranularity::Group, 128, true};
    spec.mixed_quantization.mtp = {
        8, QuantizationGranularity::Channel, 0, true};

    spec.source.repository = "QuantTrio/GLM-5.2-Int4-Int8Mix";
    spec.source.revision = "1d3bcfe5ec549ecd000fd80b37f191183842e983";
    spec.source.index_sha256 =
        "43298345833417b1ad2a8b76d012a83d4f2275d532e5ab38e118566f1ac7b12b";
    spec.source.tensor_count = 177'569;
    spec.source.indexed_tensor_bytes = 405'459'090'304ULL;
    spec.source.shard_file_bytes = 405'481'014'016ULL;
    spec.source.main_shards = 124;
    spec.source.mtp_shards = 4;

    spec.quant_bits = 4;
    spec.hidden_size = 6144;
    spec.layer_count = 78;
    spec.max_context_tokens = 1'048'576;
    spec.dense_prefix_layers = 3;
    spec.shared_experts = 1;
    spec.expert_intermediate_size = 2048;
    spec.glm_moe_dsa.sparse_attention_topk = 2048;
    spec.glm_moe_dsa.index_share_frequency = 4;
    spec.glm_moe_dsa.mtp_layers = 1;
    spec.glm_moe_dsa.index_share_for_mtp = true;
    return spec;
}

ValidationResult validate_quanttrio_glm52_int4_int8_mix(const ModelSpec& spec) {
    auto result = validate_model(spec);
    const auto expected = quanttrio_glm52_int4_int8_mix_spec();
    const auto require = [&result](bool condition, std::string_view message) {
        if (!condition) result.errors.emplace_back(message);
    };

    require(spec.source.repository == expected.source.repository,
            "unexpected QuantTrio GLM-5.2 repository");
    require(spec.source.revision == expected.source.revision,
            "unexpected QuantTrio GLM-5.2 revision");
    require(spec.source.index_sha256 == expected.source.index_sha256,
            "unexpected QuantTrio GLM-5.2 index hash");
    require(spec.source.tensor_count == expected.source.tensor_count,
            "unexpected QuantTrio GLM-5.2 tensor count");
    require(spec.source.indexed_tensor_bytes == expected.source.indexed_tensor_bytes,
            "unexpected QuantTrio GLM-5.2 indexed byte count");
    require(spec.source.shard_file_bytes == expected.source.shard_file_bytes,
            "unexpected QuantTrio GLM-5.2 shard byte count");
    require(spec.source.main_shards == expected.source.main_shards &&
                spec.source.mtp_shards == expected.source.mtp_shards,
            "unexpected QuantTrio GLM-5.2 shard count");

    require(spec.architecture == expected.architecture && spec.attention == expected.attention,
            "unexpected QuantTrio GLM-5.2 architecture");
    require(spec.hidden_size == expected.hidden_size &&
                spec.layer_count == expected.layer_count &&
                spec.dense_prefix_layers == expected.dense_prefix_layers &&
                spec.shared_experts == expected.shared_experts &&
                spec.expert_intermediate_size == expected.expert_intermediate_size &&
                spec.max_context_tokens == expected.max_context_tokens,
            "unexpected QuantTrio GLM-5.2 model dimensions");
    require(spec.router.selection == expected.router.selection &&
                spec.router.scoring == expected.router.scoring &&
                spec.router.routed_experts == expected.router.routed_experts &&
                spec.router.experts_per_token == expected.router.experts_per_token &&
                spec.router.groups == expected.router.groups &&
                spec.router.selected_groups == expected.router.selected_groups &&
                spec.router.normalize_topk == expected.router.normalize_topk &&
                spec.router.selection_bias == expected.router.selection_bias &&
                spec.router.routed_scale == expected.router.routed_scale,
            "unexpected QuantTrio GLM-5.2 router semantics");
    require(spec.glm_moe_dsa.sparse_attention_topk ==
                    expected.glm_moe_dsa.sparse_attention_topk &&
                spec.glm_moe_dsa.index_share_frequency ==
                    expected.glm_moe_dsa.index_share_frequency &&
                spec.glm_moe_dsa.mtp_layers == expected.glm_moe_dsa.mtp_layers &&
                spec.glm_moe_dsa.index_share_for_mtp ==
                    expected.glm_moe_dsa.index_share_for_mtp,
            "unexpected QuantTrio GLM-5.2 DSA or MTP semantics");

    const auto& actual_quantization = spec.mixed_quantization;
    const auto& expected_quantization = expected.mixed_quantization;
    const auto same_role = [](const QuantizedWeightSpec& actual,
                              const QuantizedWeightSpec& wanted) {
        return actual.bits == wanted.bits && actual.granularity == wanted.granularity &&
               actual.group_size == wanted.group_size && actual.symmetric == wanted.symmetric;
    };
    require(actual_quantization.kind == expected_quantization.kind &&
                actual_quantization.activation_bits == expected_quantization.activation_bits &&
                actual_quantization.quantized_linear_start_layer ==
                    expected_quantization.quantized_linear_start_layer &&
                actual_quantization.quantized_expert_start_layer ==
                    expected_quantization.quantized_expert_start_layer &&
                actual_quantization.mtp_layer_index ==
                    expected_quantization.mtp_layer_index &&
                same_role(actual_quantization.routed_experts,
                          expected_quantization.routed_experts) &&
                same_role(actual_quantization.linears, expected_quantization.linears) &&
                same_role(actual_quantization.mtp, expected_quantization.mtp),
            "unexpected QuantTrio GLM-5.2 quantization semantics");
    return result;
}

}  // namespace strata
