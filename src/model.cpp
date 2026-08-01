#include "strata/model.hpp"

#include "strata/model_adapter.hpp"

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

    if (spec.mixed_quantization.kind == QuantizationKind::CompressedTensorsW4A16 ||
        spec.mixed_quantization.kind == QuantizationKind::CompressedTensorsW8A16) {
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

    if (spec.architecture == ArchitectureKind::Dense ||
        spec.architecture == ArchitectureKind::Gemma4) {
        if (spec.router.selection != RouterSelectionKind::None ||
            spec.router.scoring != RouterScoreKind::None ||
            spec.router.routed_experts != 0 || spec.router.experts_per_token != 0) {
            result.errors.emplace_back("dense architecture cannot declare routed experts");
        }
        if (spec.shared_experts != 0) {
            result.errors.emplace_back("dense architecture cannot declare shared experts");
        }
        if (spec.architecture == ArchitectureKind::Gemma4 &&
            spec.attention != AttentionKind::HybridLocalGlobal) {
            result.errors.emplace_back(
                "Gemma 4 adapter requires hybrid local/global attention");
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

ModelSpec deepseek_v4_flash_0731_spec() {
    ModelSpec spec;
    spec.name = "deepseek-ai/DeepSeek-V4-Flash-0731";
    spec.architecture = ArchitectureKind::DeepSeek;
    spec.attention = AttentionKind::HybridCompressedSparse;
    spec.router.selection = RouterSelectionKind::NoAuxTc;
    spec.router.scoring = RouterScoreKind::SqrtSoftplus;
    spec.router.routed_experts = kDeepSeekV4ExecutionContract.routed_experts;
    spec.router.experts_per_token = kDeepSeekV4ExecutionContract.experts_per_token;
    spec.router.groups = 1U;
    spec.router.selected_groups = 1U;
    spec.router.normalize_topk = true;
    spec.router.selection_bias = true;
    spec.router.routed_scale = kDeepSeekV4ExecutionContract.routed_scale;

    spec.mixed_quantization.kind = QuantizationKind::NativeFp4Fp8;
    spec.native_fp4_fp8 = {8U, 128U, 8U, 128U, 128U, 4U, 32U, true};
    spec.source.repository = "deepseek-ai/DeepSeek-V4-Flash-0731";
    spec.source.revision = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb";
    spec.source.index_sha256 =
        "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b";
    spec.source.tensor_count = 72'317U;
    spec.source.indexed_tensor_bytes = 166'878'536'440ULL;
    spec.source.shard_file_bytes = 166'886'535'336ULL;
    spec.source.main_shards = 48U;
    spec.source.mtp_shards = 0U;

    spec.quant_bits = 4U;
    spec.hidden_size = kDeepSeekV4ExecutionContract.hidden_size;
    spec.layer_count = kDeepSeekV4ExecutionContract.layer_count;
    spec.max_context_tokens = kDeepSeekV4ExecutionContract.maximum_context_tokens;
    spec.dense_prefix_layers = 0U;
    spec.shared_experts = 1U;
    spec.expert_intermediate_size =
        kDeepSeekV4ExecutionContract.expert_intermediate_size;

    spec.deepseek_v4.attention_heads = kDeepSeekV4ExecutionContract.attention_heads;
    spec.deepseek_v4.key_value_heads = kDeepSeekV4ExecutionContract.key_value_heads;
    spec.deepseek_v4.head_dim = kDeepSeekV4ExecutionContract.head_dim;
    spec.deepseek_v4.rope_head_dim = kDeepSeekV4ExecutionContract.rope_head_dim;
    spec.deepseek_v4.query_lora_rank = kDeepSeekV4ExecutionContract.query_lora_rank;
    spec.deepseek_v4.output_lora_rank = kDeepSeekV4ExecutionContract.output_lora_rank;
    spec.deepseek_v4.output_groups = kDeepSeekV4ExecutionContract.output_groups;
    spec.deepseek_v4.sliding_window = kDeepSeekV4ExecutionContract.sliding_window;
    spec.deepseek_v4.index_heads = kDeepSeekV4ExecutionContract.index_heads;
    spec.deepseek_v4.index_head_dim = kDeepSeekV4ExecutionContract.index_head_dim;
    spec.deepseek_v4.index_topk = kDeepSeekV4ExecutionContract.index_topk;
    spec.deepseek_v4.hash_layers = 3U;
    spec.deepseek_v4.mhc_multiplier = kDeepSeekV4ExecutionContract.mhc_multiplier;
    spec.deepseek_v4.mhc_sinkhorn_iterations =
        kDeepSeekV4ExecutionContract.mhc_sinkhorn_iterations;
    spec.deepseek_v4.dspark_layers = 3U;
    spec.deepseek_v4.dspark_block_size = 5U;
    spec.deepseek_v4.dspark_noise_token_id = 128'799U;
    spec.deepseek_v4.dspark_markov_rank = 256U;
    spec.deepseek_v4.swiglu_limit = kDeepSeekV4ExecutionContract.swiglu_limit;
    spec.deepseek_v4.compression_ratios.assign(
        kDeepSeekV4ExecutionContract.compression_ratios.begin(),
        kDeepSeekV4ExecutionContract.compression_ratios.end());
    spec.deepseek_v4.dspark_target_layers = {40U, 41U, 42U};
    return spec;
}

ValidationResult validate_deepseek_v4_flash_0731(const ModelSpec& spec) {
    auto result = validate_model(spec);
    const auto expected = deepseek_v4_flash_0731_spec();
    const auto require = [&result](bool condition, std::string_view message) {
        if (!condition) result.errors.emplace_back(message);
    };
    require(spec.source.repository == expected.source.repository &&
                spec.source.revision == expected.source.revision &&
                spec.source.index_sha256 == expected.source.index_sha256,
            "unexpected DeepSeek-V4-Flash-0731 source identity");
    require(spec.source.tensor_count == expected.source.tensor_count &&
                spec.source.indexed_tensor_bytes == expected.source.indexed_tensor_bytes &&
                spec.source.shard_file_bytes == expected.source.shard_file_bytes &&
                spec.source.main_shards == expected.source.main_shards &&
                spec.source.mtp_shards == expected.source.mtp_shards,
            "unexpected DeepSeek-V4-Flash-0731 checkpoint extent");
    require(spec.architecture == expected.architecture &&
                spec.attention == expected.attention &&
                spec.hidden_size == expected.hidden_size &&
                spec.layer_count == expected.layer_count &&
                spec.max_context_tokens == expected.max_context_tokens &&
                spec.shared_experts == expected.shared_experts &&
                spec.expert_intermediate_size == expected.expert_intermediate_size,
            "unexpected DeepSeek-V4-Flash-0731 architecture dimensions");
    require(spec.router.selection == expected.router.selection &&
                spec.router.scoring == expected.router.scoring &&
                spec.router.routed_experts == expected.router.routed_experts &&
                spec.router.experts_per_token == expected.router.experts_per_token &&
                spec.router.normalize_topk == expected.router.normalize_topk &&
                spec.router.selection_bias == expected.router.selection_bias &&
                spec.router.routed_scale == expected.router.routed_scale,
            "unexpected DeepSeek-V4-Flash-0731 router semantics");
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
            "unexpected DeepSeek-V4-Flash-0731 native quantization");
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
            "unexpected DeepSeek-V4-Flash-0731 hybrid attention, mHC, or DSpark contract");
    return result;
}

ModelSpec glm52_w4a16_spec() {
    ModelSpec spec;
    spec.name = "glm-moe-dsa-w4a16";
    spec.architecture = ArchitectureKind::GlmMoeDsa;
    spec.attention = AttentionKind::Mla;
    spec.router.selection = RouterSelectionKind::NoAuxTc;
    spec.router.scoring = RouterScoreKind::Sigmoid;
    spec.router.routed_experts = kGlm52ExecutionContract.routed_experts;
    spec.router.experts_per_token = kGlm52ExecutionContract.experts_per_token;
    spec.router.groups = 1;
    spec.router.selected_groups = 1;
    spec.router.normalize_topk = true;
    spec.router.selection_bias = true;
    spec.router.routed_scale = kGlm52ExecutionContract.routed_scale;

    spec.mixed_quantization.kind = QuantizationKind::CompressedTensorsW4A16;
    spec.mixed_quantization.activation_bits = 16;
    spec.mixed_quantization.quantized_linear_start_layer = 0;
    spec.mixed_quantization.quantized_expert_start_layer = 3;
    spec.mixed_quantization.mtp_layer_index = 78;
    spec.mixed_quantization.routed_experts = {
        4, QuantizationGranularity::Group, 128, true};
    spec.mixed_quantization.linears = {
        4, QuantizationGranularity::Group, 128, true};
    spec.mixed_quantization.mtp = {
        4, QuantizationGranularity::Group, 128, true};

    spec.source.index_sha256 =
        "74d73bfaa26425beaf618342f4a0851b21d9198138b76bfb678f88164d987beb";
    spec.source.tensor_count = 175'527;
    spec.source.indexed_tensor_bytes = 387'667'154'688ULL;
    spec.source.shard_file_bytes = 387'689'209'608ULL;
    spec.source.main_shards = 8;
    spec.source.mtp_shards = 0;

    spec.quant_bits = 4;
    spec.hidden_size = kGlm52ExecutionContract.hidden_size;
    spec.layer_count = kGlm52ExecutionContract.layer_count;
    spec.max_context_tokens = kGlm52ExecutionContract.maximum_context_tokens;
    spec.dense_prefix_layers = kGlm52ExecutionContract.dense_prefix_layers;
    spec.shared_experts = 1;
    spec.expert_intermediate_size = kGlm52ExecutionContract.expert_intermediate_size;
    spec.glm_moe_dsa.sparse_attention_topk =
        kGlm52ExecutionContract.sparse_attention_topk;
    spec.glm_moe_dsa.index_share_frequency = 4;
    spec.glm_moe_dsa.mtp_layers = 1;
    spec.glm_moe_dsa.index_share_for_mtp = true;
    return spec;
}

ValidationResult validate_glm52_w4a16(const ModelSpec& spec) {
    auto result = validate_model(spec);
    const auto expected = glm52_w4a16_spec();
    const auto require = [&result](bool condition, std::string_view message) {
        if (!condition) result.errors.emplace_back(message);
    };

    require(spec.source.index_sha256 == expected.source.index_sha256,
            "unexpected GLM W4A16 index hash");
    require(spec.source.tensor_count == expected.source.tensor_count,
            "unexpected GLM W4A16 tensor count");
    require(spec.source.indexed_tensor_bytes == expected.source.indexed_tensor_bytes,
            "unexpected GLM W4A16 indexed byte count");
    require(spec.source.shard_file_bytes == expected.source.shard_file_bytes,
            "unexpected GLM W4A16 shard byte count");
    require(spec.source.main_shards == expected.source.main_shards &&
                spec.source.mtp_shards == expected.source.mtp_shards,
            "unexpected GLM W4A16 shard count");

    require(spec.architecture == expected.architecture && spec.attention == expected.attention,
            "unexpected GLM W4A16 architecture");
    require(spec.hidden_size == expected.hidden_size &&
                spec.layer_count == expected.layer_count &&
                spec.dense_prefix_layers == expected.dense_prefix_layers &&
                spec.shared_experts == expected.shared_experts &&
                spec.expert_intermediate_size == expected.expert_intermediate_size &&
                spec.max_context_tokens == expected.max_context_tokens,
            "unexpected GLM W4A16 model dimensions");
    require(spec.router.selection == expected.router.selection &&
                spec.router.scoring == expected.router.scoring &&
                spec.router.routed_experts == expected.router.routed_experts &&
                spec.router.experts_per_token == expected.router.experts_per_token &&
                spec.router.groups == expected.router.groups &&
                spec.router.selected_groups == expected.router.selected_groups &&
                spec.router.normalize_topk == expected.router.normalize_topk &&
                spec.router.selection_bias == expected.router.selection_bias &&
                spec.router.routed_scale == expected.router.routed_scale,
            "unexpected GLM W4A16 router semantics");
    require(spec.glm_moe_dsa.sparse_attention_topk ==
                    expected.glm_moe_dsa.sparse_attention_topk &&
                spec.glm_moe_dsa.index_share_frequency ==
                    expected.glm_moe_dsa.index_share_frequency &&
                spec.glm_moe_dsa.mtp_layers == expected.glm_moe_dsa.mtp_layers &&
                spec.glm_moe_dsa.index_share_for_mtp ==
                    expected.glm_moe_dsa.index_share_for_mtp,
            "unexpected GLM W4A16 DSA or MTP semantics");

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
            "unexpected GLM W4A16 quantization semantics");
    return result;
}

ModelSpec gemma4_31b_it_w8a16_spec() {
    ModelSpec spec;
    const auto& contract = kGemma4ExecutionContract;
    spec.name = "google/gemma-4-31B-it";
    spec.architecture = ArchitectureKind::Gemma4;
    spec.attention = AttentionKind::HybridLocalGlobal;
    spec.mixed_quantization.kind = QuantizationKind::CompressedTensorsW8A16;
    spec.mixed_quantization.activation_bits = 16U;
    spec.mixed_quantization.quantized_linear_start_layer = 0U;
    spec.mixed_quantization.quantized_expert_start_layer = contract.layer_count;
    spec.mixed_quantization.mtp_layer_index = contract.layer_count;
    spec.mixed_quantization.linears = {
        8U, QuantizationGranularity::Group, 32U, true};
    spec.mixed_quantization.routed_experts = spec.mixed_quantization.linears;
    spec.mixed_quantization.mtp = spec.mixed_quantization.linears;
    spec.source.repository = "google/gemma-4-31B-it";
    spec.source.index_sha256 =
        "d6d47cb0547c6090c4b81aa5fe90d72026c2124b06c53c2d9efeec0e693fd421";
    spec.source.tensor_count = 2'008U;
    spec.source.indexed_tensor_bytes = 35'089'877'112ULL;
    spec.source.shard_file_bytes = 35'090'140'728ULL;
    spec.source.main_shards = 7U;
    spec.quant_bits = 8U;
    spec.hidden_size = contract.hidden_size;
    spec.layer_count = contract.layer_count;
    spec.max_context_tokens = contract.maximum_context_tokens;
    spec.gemma4 = {
        contract.attention_heads,
        contract.local_key_value_heads,
        contract.global_key_value_heads,
        contract.local_head_dim,
        contract.global_head_dim,
        contract.intermediate_size,
        contract.sliding_window,
        6U,
        contract.vision_hidden_size,
        contract.vision_layer_count,
        contract.vision_attention_heads,
        contract.vision_head_dim,
        contract.vision_intermediate_size,
        contract.vision_patch_size,
        contract.vision_position_embeddings,
        contract.vision_pooling_kernel,
        258'880U,
        contract.default_image_tokens,
        contract.rms_epsilon,
        contract.local_rope_theta,
        contract.global_rope_theta,
        contract.global_rope_proportion,
        contract.vision_rope_theta,
        contract.final_logit_softcap,
        true,
        true,
    };
    return spec;
}

ValidationResult validate_gemma4_31b_it_w8a16(const ModelSpec& spec) {
    auto result = validate_model(spec);
    const auto expected = gemma4_31b_it_w8a16_spec();
    const auto require = [&result](bool condition, std::string_view message) {
        if (!condition) result.errors.emplace_back(message);
    };
    require(spec.source.repository == expected.source.repository &&
                spec.source.index_sha256 == expected.source.index_sha256,
            "unexpected Gemma 4 31B source identity");
    require(spec.source.tensor_count == expected.source.tensor_count &&
                spec.source.indexed_tensor_bytes == expected.source.indexed_tensor_bytes &&
                spec.source.shard_file_bytes == expected.source.shard_file_bytes &&
                spec.source.main_shards == expected.source.main_shards,
            "unexpected Gemma 4 31B checkpoint extent");
    require(spec.architecture == expected.architecture &&
                spec.attention == expected.attention &&
                spec.hidden_size == expected.hidden_size &&
                spec.layer_count == expected.layer_count &&
                spec.max_context_tokens == expected.max_context_tokens,
            "unexpected Gemma 4 31B architecture dimensions");
    const auto& actual = spec.gemma4;
    const auto& wanted = expected.gemma4;
    require(actual.attention_heads == wanted.attention_heads &&
                actual.local_key_value_heads == wanted.local_key_value_heads &&
                actual.global_key_value_heads == wanted.global_key_value_heads &&
                actual.local_head_dim == wanted.local_head_dim &&
                actual.global_head_dim == wanted.global_head_dim &&
                actual.intermediate_size == wanted.intermediate_size &&
                actual.sliding_window == wanted.sliding_window &&
                actual.global_attention_stride == wanted.global_attention_stride &&
                actual.vision_hidden_size == wanted.vision_hidden_size &&
                actual.vision_layers == wanted.vision_layers &&
                actual.vision_attention_heads == wanted.vision_attention_heads &&
                actual.vision_head_dim == wanted.vision_head_dim &&
                actual.vision_intermediate_size == wanted.vision_intermediate_size &&
                actual.vision_patch_size == wanted.vision_patch_size &&
                actual.vision_position_embeddings == wanted.vision_position_embeddings &&
                actual.vision_pooling_kernel == wanted.vision_pooling_kernel &&
                actual.image_token_id == wanted.image_token_id &&
                actual.default_image_tokens == wanted.default_image_tokens &&
                actual.rms_epsilon == wanted.rms_epsilon &&
                actual.local_rope_theta == wanted.local_rope_theta &&
                actual.global_rope_theta == wanted.global_rope_theta &&
                actual.global_rope_proportion == wanted.global_rope_proportion &&
                actual.vision_rope_theta == wanted.vision_rope_theta &&
                actual.final_logit_softcap == wanted.final_logit_softcap &&
                actual.global_key_equals_value == wanted.global_key_equals_value &&
                actual.vision_bidirectional == wanted.vision_bidirectional,
            "unexpected Gemma 4 hybrid attention or vision contract");
    const auto& quantization = spec.mixed_quantization;
    require(quantization.kind == QuantizationKind::CompressedTensorsW8A16 &&
                quantization.activation_bits == 16U &&
                quantization.linears.bits == 8U &&
                quantization.linears.granularity == QuantizationGranularity::Group &&
                quantization.linears.group_size == 32U &&
                quantization.linears.symmetric && spec.quant_bits == 8U,
            "unexpected Gemma 4 W8A16 quantization semantics");
    return result;
}

}  // namespace strata
