#include "strata/models/glm53/glm53_manifest.hpp"

#include "../../platform/json_cursor.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <unordered_map>

namespace strata {
namespace {

using detail::JsonCursor;

constexpr std::string_view kLayerPrefix = "model.language_model.layers.";

// One pinned row per supported GLM-5.3-Flash release. The extents are the
// gate: a checkpoint that matches none of them is not a release this build has
// been validated against, and the manifest refuses it rather than guessing.
struct Glm53ReleaseProfile {
    Glm53Quantization quantization;
    // `metadata.total_size` as the checkpoint's index declares it, and the sum
    // of the tensor extents the shards actually hold. Both published FP8 and
    // MXFP4 exports follow the Hugging Face convention where those coincide;
    // the NVFP4 export declares the shard file total instead, so the two are
    // pinned separately rather than one being checked twice.
    std::uint64_t indexed_bytes;
    std::uint64_t payload_bytes;
    std::uint64_t shard_file_bytes;
    std::uint64_t tensor_count;
    std::uint32_t shard_count;
};

constexpr std::array<Glm53ReleaseProfile, 3U> kReleases{{
    {Glm53Quantization::Fp8E4m3Block128, 328'326'771'576ULL,
     328'326'771'576ULL, 328'337'455'672ULL, 76'108ULL, 62U},
    {Glm53Quantization::Mxfp4Group32, 227'486'055'288ULL, 227'486'055'288ULL,
     227'496'161'368ULL, 72'466ULL, 120U},
    {Glm53Quantization::Nvfp4Group16E4m3, 190'213'869'288ULL,
     190'198'265'848ULL, 190'213'869'288ULL, 110'457ULL, 62U},
}};

[[nodiscard]] const Glm53ReleaseProfile* release_profile(
    Glm53Quantization quantization) noexcept {
    for (const auto& profile : kReleases) {
        if (profile.quantization == quantization) return &profile;
    }
    return nullptr;
}

bool take_index(std::string_view name, std::size_t& cursor,
                std::uint32_t& value) noexcept {
    const auto begin = cursor;
    while (cursor < name.size() && name[cursor] >= '0' && name[cursor] <= '9') {
        ++cursor;
    }
    if (cursor == begin) return false;
    const auto parsed = std::from_chars(name.data() + begin,
                                        name.data() + cursor, value);
    return parsed.ec == std::errc{};
}

bool has_suffix(std::string_view value, std::string_view suffix) noexcept {
    return value.size() >= suffix.size() && value.ends_with(suffix);
}

void parse_uint_list(JsonCursor& cursor, std::vector<std::uint32_t>& output) {
    cursor.expect('[');
    if (cursor.consume(']')) return;
    for (;;) {
        output.push_back(static_cast<std::uint32_t>(cursor.parse_uint64()));
        if (cursor.consume(']')) return;
        cursor.expect(',');
    }
}

void parse_string_list(JsonCursor& cursor, std::vector<std::string>& output) {
    cursor.expect('[');
    if (cursor.consume(']')) return;
    for (;;) {
        output.push_back(cursor.parse_string());
        if (cursor.consume(']')) return;
        cursor.expect(',');
    }
}

void parse_linear_attention(JsonCursor& cursor, Glm53TextConfig& config) {
    cursor.expect('{');
    if (cursor.consume('}')) return;
    for (;;) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        if (key == "num_heads") {
            config.linear_attention_heads =
                static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "head_dim") {
            config.linear_head_dim =
                static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "short_conv_kernel_size") {
            config.short_conv_kernel =
                static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "gate_lower_bound") {
            config.kda_gate_lower_bound = static_cast<float>(cursor.parse_number());
        } else if (key == "full_attn_layers") {
            parse_uint_list(cursor, config.full_attention_layers);
        } else if (key == "kda_layers") {
            parse_uint_list(cursor, config.kda_layers);
        } else {
            cursor.skip_value();
        }
        if (cursor.consume('}')) return;
        cursor.expect(',');
    }
}

void parse_text(JsonCursor& cursor, Glm53TextConfig& config) {
    cursor.expect('{');
    if (cursor.consume('}')) return;
    for (;;) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        if (key == "model_type") config.model_type = cursor.parse_string();
        else if (key == "hidden_size") config.hidden_size = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "num_hidden_layers") config.layer_count = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "num_attention_heads") config.attention_heads = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "num_key_value_heads") config.key_value_heads = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "q_lora_rank") config.query_lora_rank = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "kv_lora_rank") config.kv_lora_rank = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "qk_nope_head_dim") config.nope_head_dim = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "qk_rope_head_dim") config.rope_head_dim = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "v_head_dim") config.value_head_dim = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "intermediate_size") config.dense_intermediate_size = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "moe_intermediate_size") config.expert_intermediate_size = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "n_routed_experts") config.routed_experts = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "num_experts_per_tok") config.experts_per_token = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "n_group") config.expert_groups = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "topk_group") config.selected_expert_groups = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "n_shared_experts") config.shared_experts = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "first_k_dense_replace") config.dense_prefix_layers = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "vocab_size") config.vocabulary_size = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "max_position_embeddings") config.maximum_context_tokens = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "hc_mult") config.mhc_multiplier = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "hc_sinkhorn_iters") config.mhc_sinkhorn_iterations = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "index_n_heads") config.index_heads = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "index_head_dim") config.index_head_dim = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "index_topk") config.index_topk = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "index_kpool") config.index_pool = static_cast<std::uint32_t>(cursor.parse_uint64());
        else if (key == "rms_norm_eps") config.rms_epsilon = static_cast<float>(cursor.parse_number());
        else if (key == "hc_eps") config.mhc_epsilon = static_cast<float>(cursor.parse_number());
        else if (key == "routed_scaling_factor") config.routed_scale = static_cast<float>(cursor.parse_number());
        else if (key == "swiglu_limit") config.swiglu_limit = static_cast<float>(cursor.parse_number());
        else if (key == "norm_topk_prob") config.normalize_topk = cursor.parse_bool();
        else if (key == "mhc") config.mhc = cursor.parse_bool();
        else if (key == "mla_use_nope") config.mla_use_nope = cursor.parse_bool();
        else if (key == "index_kpool_compress") config.index_pool_compress = cursor.parse_bool();
        else if (key == "index_kpool_always_select_tail") config.index_pool_select_tail = cursor.parse_bool();
        else if (key == "tie_word_embeddings") config.tie_word_embeddings = cursor.parse_bool();
        else if (key == "hidden_act") config.hidden_activation = cursor.parse_string();
        else if (key == "scoring_func") config.router_scoring = cursor.parse_string();
        else if (key == "topk_method") config.topk_method = cursor.parse_string();
        else if (key == "layer_types") parse_string_list(cursor, config.attention_layer_types);
        else if (key == "mlp_layer_types") parse_string_list(cursor, config.mlp_layer_types);
        else if (key == "linear_attn_config") parse_linear_attention(cursor, config);
        else cursor.skip_value();
        if (cursor.consume('}')) return;
        cursor.expect(',');
    }
}

// quark records the weight format under global_quant_config.weight; the FP8
// release records it flat. Both are read into the same fields so validation
// sees one shape of answer.
void parse_quark_weight(JsonCursor& cursor, Glm53TextConfig& config) {
    cursor.expect('{');
    if (cursor.consume('}')) return;
    for (;;) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        if (key == "dtype") {
            config.quantization_weight_dtype = cursor.parse_string();
        } else if (key == "scale_format") {
            config.quantization_scale_format = cursor.parse_string();
        } else if (key == "group_size") {
            config.quantization_group_size =
                static_cast<std::uint32_t>(cursor.parse_uint64());
        } else {
            cursor.skip_value();
        }
        if (cursor.consume('}')) return;
        cursor.expect(',');
    }
}

// compressed-tensors records the weight format per config group rather than
// once. Every GLM-5.3 NVFP4 group quantizes the same routed-expert projections
// with the same rule, so the first group is read into the same fields the
// quark and FP8 exports fill and validation sees one shape of answer.
void parse_compressed_tensors_weights(JsonCursor& cursor,
                                      Glm53TextConfig& config) {
    cursor.expect('{');
    if (cursor.consume('}')) return;
    std::uint32_t bits = 0U;
    std::string type;
    for (;;) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        if (key == "num_bits") {
            bits = static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "type") {
            type = cursor.parse_string();
        } else if (key == "group_size") {
            config.quantization_group_size =
                static_cast<std::uint32_t>(cursor.parse_uint64());
        } else {
            cursor.skip_value();
        }
        if (cursor.consume('}')) break;
        cursor.expect(',');
    }
    if (bits == 4U && type == "float") config.quantization_weight_dtype = "fp4";
}

void parse_compressed_tensors_group(JsonCursor& cursor,
                                    Glm53TextConfig& config) {
    cursor.expect('{');
    if (cursor.consume('}')) return;
    for (;;) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        if (key == "weights") parse_compressed_tensors_weights(cursor, config);
        else cursor.skip_value();
        if (cursor.consume('}')) return;
        cursor.expect(',');
    }
}

void parse_compressed_tensors_groups(JsonCursor& cursor,
                                     Glm53TextConfig& config) {
    cursor.expect('{');
    if (cursor.consume('}')) return;
    bool first = true;
    for (;;) {
        static_cast<void>(cursor.parse_string());
        cursor.expect(':');
        if (first) parse_compressed_tensors_group(cursor, config);
        else cursor.skip_value();
        first = false;
        if (cursor.consume('}')) return;
        cursor.expect(',');
    }
}

void parse_quark_global(JsonCursor& cursor, Glm53TextConfig& config) {
    cursor.expect('{');
    if (cursor.consume('}')) return;
    for (;;) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        if (key == "weight") parse_quark_weight(cursor, config);
        else cursor.skip_value();
        if (cursor.consume('}')) return;
        cursor.expect(',');
    }
}

void parse_quantization(JsonCursor& cursor, Glm53TextConfig& config) {
    cursor.expect('{');
    if (cursor.consume('}')) return;
    for (;;) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        if (key == "quant_method") config.quantization_method = cursor.parse_string();
        else if (key == "fmt") config.quantization_format = cursor.parse_string();
        // compressed-tensors spells the same field `format`. It also repeats it
        // inside each config group, but only the top level is read here, so the
        // two never contend.
        else if (key == "format") config.quantization_format = cursor.parse_string();
        else if (key == "config_groups") parse_compressed_tensors_groups(cursor, config);
        else if (key == "global_quant_config") parse_quark_global(cursor, config);
        else if (key == "weight_block_size") {
            std::vector<std::uint32_t> dimensions;
            parse_uint_list(cursor, dimensions);
            if (dimensions.size() == 2U) {
                config.fp8_block_rows = dimensions[0];
                config.fp8_block_columns = dimensions[1];
            }
        } else cursor.skip_value();
        if (cursor.consume('}')) return;
        cursor.expect(',');
    }
}

Glm53TensorComponent component_of(std::string_view name) noexcept {
    // NVFP4 carries two scale tensors per module -- the E4M3 group scales and
    // the F32 per-tensor divisor -- and names the payload `weight_packed`
    // rather than `weight`, so neither suffix is reachable from the rules the
    // other two releases need.
    if (has_suffix(name, ".weight_scale_inv") ||
        has_suffix(name, ".weight_global_scale") ||
        has_suffix(name, ".weight_scale")) {
        return Glm53TensorComponent::Scale;
    }
    if (has_suffix(name, ".bias") || has_suffix(name, "_bias")) {
        return Glm53TensorComponent::Bias;
    }
    if (has_suffix(name, ".weight") || has_suffix(name, ".weight_packed") ||
        has_suffix(name, "_fn")) {
        return Glm53TensorComponent::Weight;
    }
    return Glm53TensorComponent::State;
}

}  // namespace

Glm53ConfigResult parse_glm53_config(std::string_view json) {
    Glm53ConfigResult result;
    try {
        JsonCursor cursor(json);
        cursor.expect('{');
        if (!cursor.consume('}')) {
            for (;;) {
                const auto key = cursor.parse_string();
                cursor.expect(':');
                if (key == "architectures") {
                    std::vector<std::string> architectures;
                    parse_string_list(cursor, architectures);
                    if (architectures.size() == 1U) {
                        result.value.architecture = std::move(architectures[0]);
                    }
                } else if (key == "text_config") {
                    parse_text(cursor, result.value);
                } else if (key == "quantization_config") {
                    parse_quantization(cursor, result.value);
                } else {
                    cursor.skip_value();
                }
                if (cursor.consume('}')) break;
                cursor.expect(',');
            }
        }
    } catch (const detail::JsonError& error) {
        result.errors.push_back("GLM-5.3 config.json is malformed at offset " +
                                std::to_string(error.offset()) + ": " +
                                error.what());
    }
    return result;
}

Glm53Quantization glm53_config_quantization(
    const Glm53TextConfig& config) noexcept {
    if (config.quantization_method == "fp8" &&
        config.quantization_format == "e4m3") {
        return Glm53Quantization::Fp8E4m3Block128;
    }
    if (config.quantization_method == "quark" &&
        config.quantization_weight_dtype == "fp4") {
        return Glm53Quantization::Mxfp4Group32;
    }
    if (config.quantization_method == "compressed-tensors" &&
        config.quantization_format == "nvfp4-pack-quantized") {
        return Glm53Quantization::Nvfp4Group16E4m3;
    }
    return Glm53Quantization::Unsupported;
}

ValidationResult validate_glm53_config(const Glm53TextConfig& config) {
    ValidationResult result;
    const auto require = [&result](bool condition, std::string message) {
        if (!condition) result.errors.push_back(std::move(message));
    };
    const auto equal = [&require](auto actual, auto wanted, std::string_view name) {
        require(actual == wanted, "GLM-5.3 " + std::string(name) + " is " +
                    std::to_string(actual) + ", expected " +
                    std::to_string(wanted));
    };
    require(config.architecture == "Glm5NextForConditionalGeneration",
            "GLM-5.3 architecture must be Glm5NextForConditionalGeneration");
    require(config.model_type == "glm5_next_text",
            "GLM-5.3 text model_type must be glm5_next_text");
    equal(config.hidden_size, 4096U, "hidden_size");
    equal(config.layer_count, 45U, "num_hidden_layers");
    equal(config.attention_heads, 64U, "num_attention_heads");
    equal(config.key_value_heads, 64U, "num_key_value_heads");
    equal(config.query_lora_rank, 1536U, "q_lora_rank");
    equal(config.kv_lora_rank, 512U, "kv_lora_rank");
    equal(config.nope_head_dim, 256U, "qk_nope_head_dim");
    equal(config.rope_head_dim, 0U, "qk_rope_head_dim");
    equal(config.value_head_dim, 256U, "v_head_dim");
    equal(config.linear_attention_heads, 64U, "linear attention heads");
    equal(config.linear_head_dim, 128U, "linear attention head_dim");
    equal(config.short_conv_kernel, 4U, "linear convolution kernel");
    equal(config.dense_intermediate_size, 12288U, "intermediate_size");
    equal(config.expert_intermediate_size, 2048U, "moe_intermediate_size");
    equal(config.routed_experts, 288U, "n_routed_experts");
    equal(config.experts_per_token, 8U, "num_experts_per_tok");
    equal(config.expert_groups, 1U, "n_group");
    equal(config.selected_expert_groups, 1U, "topk_group");
    equal(config.shared_experts, 1U, "n_shared_experts");
    equal(config.dense_prefix_layers, 3U, "first_k_dense_replace");
    equal(config.vocabulary_size, 154880U, "vocab_size");
    equal(config.maximum_context_tokens, 1'048'576U, "max_position_embeddings");
    equal(config.mhc_multiplier, 4U, "hc_mult");
    equal(config.mhc_sinkhorn_iterations, 20U, "hc_sinkhorn_iters");
    equal(config.index_heads, 32U, "index_n_heads");
    equal(config.index_head_dim, 128U, "index_head_dim");
    equal(config.index_topk, 2048U, "index_topk");
    equal(config.index_pool, 4U, "index_kpool");
    require(std::abs(config.rms_epsilon - 1.0e-5F) <= 1.0e-12F,
            "GLM-5.3 rms_norm_eps must be 1e-5");
    require(std::abs(config.mhc_epsilon - 1.0e-6F) <= 1.0e-12F,
            "GLM-5.3 hc_eps must be 1e-6");
    require(config.routed_scale == 2.5F,
            "GLM-5.3 routed_scaling_factor must be 2.5");
    require(config.swiglu_limit == 10.0F, "GLM-5.3 swiglu_limit must be 10");
    require(config.kda_gate_lower_bound == -5.0F,
            "GLM-5.3 KDA gate lower bound must be -5");
    require(config.normalize_topk, "GLM-5.3 must normalize routed top-k weights");
    require(config.mhc, "GLM-5.3 must enable mHC");
    require(config.mla_use_nope, "GLM-5.3 sparse MLA must use NoPE");
    require(config.index_pool_compress && config.index_pool_select_tail,
            "GLM-5.3 index k-pool compression and visible tail are required");
    require(!config.tie_word_embeddings,
            "GLM-5.3 output head must not be tied to embeddings");
    require(config.hidden_activation == "silu", "GLM-5.3 activation must be silu");
    require(config.router_scoring == "sigmoid", "GLM-5.3 router must use sigmoid scores");
    require(config.topk_method == "noaux_tc", "GLM-5.3 router must use noaux_tc selection");
    switch (glm53_config_quantization(config)) {
        case Glm53Quantization::Fp8E4m3Block128:
            equal(config.fp8_block_rows, 128U, "FP8 block rows");
            equal(config.fp8_block_columns, 128U, "FP8 block columns");
            break;
        case Glm53Quantization::Mxfp4Group32:
            equal(config.quantization_group_size, 32U, "MXFP4 group size");
            require(config.quantization_scale_format == "e8m0",
                    "GLM-5.3 MXFP4 scales must be E8M0");
            break;
        case Glm53Quantization::Nvfp4Group16E4m3:
            equal(config.quantization_group_size, 16U, "NVFP4 group size");
            require(config.quantization_weight_dtype == "fp4",
                    "GLM-5.3 NVFP4 weights must be 4-bit float");
            break;
        case Glm53Quantization::Unsupported:
            require(false,
                    "GLM-5.3 quantization must be FP8 E4M3 block-128, quark "
                    "MXFP4 E2M1 group-32 or compressed-tensors NVFP4 E2M1 "
                    "group-16");
            break;
    }
    require(config.attention_layer_types.size() == config.layer_count,
            "GLM-5.3 attention layer_types must cover all 45 layers");
    require(config.mlp_layer_types.size() == config.layer_count,
            "GLM-5.3 mlp_layer_types must cover all 45 layers");
    for (std::uint32_t layer = 0U; layer < config.layer_count; ++layer) {
        const auto attention = glm53_kda_layer(layer) ? "linear_attention"
                                                     : "deepseek_sparse_attention";
        const auto mlp = glm53_moe_layer(layer) ? "sparse" : "dense";
        require(layer < config.attention_layer_types.size() &&
                    config.attention_layer_types[layer] == attention,
                "GLM-5.3 attention schedule differs at layer " +
                    std::to_string(layer));
        require(layer < config.mlp_layer_types.size() &&
                    config.mlp_layer_types[layer] == mlp,
                "GLM-5.3 MLP schedule differs at layer " +
                    std::to_string(layer));
    }
    std::vector<std::uint32_t> expected_full_attention;
    std::vector<std::uint32_t> expected_kda;
    for (std::uint32_t layer = 0U; layer < 45U; ++layer) {
        (glm53_kda_layer(layer) ? expected_kda : expected_full_attention)
            .push_back(layer);
    }
    require(config.full_attention_layers == expected_full_attention &&
                config.kda_layers == expected_kda,
            "GLM-5.3 linear attention lists must exactly partition the pinned "
            "34 KDA and 11 sparse layers");
    return result;
}

Glm53TensorRole classify_glm53_tensor(std::string_view name,
                                      std::int32_t& layer,
                                      std::int32_t& expert) noexcept {
    layer = -1;
    expert = -1;
    if (name.starts_with("model.visual.")) return Glm53TensorRole::Vision;
    if (name == "model.language_model.embed_tokens.weight") return Glm53TensorRole::Embedding;
    if (name == "model.language_model.norm.weight") return Glm53TensorRole::Norm;
    if (name == "lm_head.weight") return Glm53TensorRole::OutputHead;
    if (!name.starts_with(kLayerPrefix)) return Glm53TensorRole::Count;
    std::size_t cursor = kLayerPrefix.size();
    std::uint32_t ordinal = 0U;
    if (!take_index(name, cursor, ordinal) || cursor >= name.size() ||
        name[cursor] != '.') {
        return Glm53TensorRole::Count;
    }
    layer = static_cast<std::int32_t>(ordinal);
    if (ordinal >= 45U) return Glm53TensorRole::Mtp;
    const auto leaf = name.substr(cursor + 1U);
    if (leaf.starts_with("hc_")) return Glm53TensorRole::Mhc;
    if (leaf.starts_with("input_layernorm") ||
        leaf.starts_with("post_attention_layernorm")) return Glm53TensorRole::Norm;
    if (leaf.starts_with("self_attn.indexer.")) return Glm53TensorRole::AttentionIndexer;
    if (leaf.starts_with("self_attn.")) {
        return glm53_kda_layer(ordinal) ? Glm53TensorRole::KdaAttention
                                       : Glm53TensorRole::SparseAttention;
    }
    if (!leaf.starts_with("mlp.")) return Glm53TensorRole::Count;
    const auto mlp = leaf.substr(4U);
    if (ordinal < 3U) return Glm53TensorRole::DenseMlp;
    if (mlp.starts_with("gate.")) return Glm53TensorRole::Router;
    if (mlp.starts_with("shared_experts.")) return Glm53TensorRole::SharedExpert;
    if (mlp.starts_with("experts.")) {
        std::size_t inner = 8U;
        std::uint32_t selected = 0U;
        if (!take_index(mlp, inner, selected)) return Glm53TensorRole::Count;
        expert = static_cast<std::int32_t>(selected);
        return Glm53TensorRole::RoutedExpert;
    }
    return Glm53TensorRole::Count;
}

Glm53ManifestResult build_glm53_index_manifest(SafetensorsIndex index) {
    Glm53ManifestResult result;
    auto& manifest = result.manifest;
    manifest.indexed_tensor_bytes = index.total_size;
    manifest.shards = std::move(index.shards);
    manifest.tensors.reserve(index.entries.size());
    const Glm53ReleaseProfile* profile = nullptr;
    for (const auto& candidate : kReleases) {
        if (manifest.indexed_tensor_bytes == candidate.indexed_bytes &&
            index.entries.size() == candidate.tensor_count &&
            manifest.shards.size() == candidate.shard_count) {
            profile = &candidate;
            break;
        }
    }
    if (profile == nullptr) {
        result.errors.emplace_back(
            "GLM-5.3 checkpoint index extent does not match a pinned Flash "
            "release");
        return result;
    }
    manifest.quantization = profile->quantization;
    for (auto& entry : index.entries) {
        std::int32_t layer = -1;
        std::int32_t expert = -1;
        const auto role = classify_glm53_tensor(entry.name, layer, expert);
        if (role == Glm53TensorRole::Count) {
            if (result.errors.size() < 64U) {
                result.errors.push_back("unclassified GLM-5.3 tensor " + entry.name);
            }
            continue;
        }
        Glm53ManifestTensor tensor;
        tensor.name = std::move(entry.name);
        tensor.shard = std::move(entry.shard);
        tensor.role = role;
        tensor.component = component_of(tensor.name);
        tensor.encoding = Glm53TensorEncoding::Plain;
        if (tensor.component == Glm53TensorComponent::Scale) {
            switch (profile->quantization) {
                case Glm53Quantization::Mxfp4Group32:
                    tensor.encoding = Glm53TensorEncoding::Fp4E2m1Group32E8m0;
                    break;
                case Glm53Quantization::Nvfp4Group16E4m3:
                    tensor.encoding = Glm53TensorEncoding::Fp4E2m1Group16E4m3;
                    break;
                case Glm53Quantization::Fp8E4m3Block128:
                case Glm53Quantization::Unsupported:
                    tensor.encoding = Glm53TensorEncoding::Fp8E4m3Block128F32;
                    break;
            }
        }
        tensor.layer = layer;
        tensor.expert = expert;
        ++manifest.role_counts[static_cast<std::size_t>(role)];
        if (tensor.component == Glm53TensorComponent::Scale) ++manifest.fp8_modules;
        manifest.tensors.push_back(std::move(tensor));
    }
    return result;
}

Glm53ManifestResult validate_glm53_checkpoint(
    const std::string& model_directory, Glm53IndexManifest manifest,
    const Glm53CheckpointOptions& options) {
    Glm53ManifestResult result;
    result.manifest = std::move(manifest);
    auto& out = result.manifest;
    std::unordered_map<std::string_view, std::size_t> indexed;
    indexed.reserve(out.tensors.size());
    for (std::size_t index = 0U; index < out.tensors.size(); ++index) {
        if (!indexed.emplace(out.tensors[index].name, index).second) {
            result.errors.push_back("duplicate GLM-5.3 tensor " + out.tensors[index].name);
        }
    }
    if (!result.errors.empty()) return result;
    std::uint64_t resolved = 0U;
    for (const auto& shard_name : out.shards) {
        auto shard = load_safetensors_shard(
            (std::filesystem::path(model_directory) / shard_name).string());
        if (!shard.ok()) {
            if (options.require_all_shards) {
                for (auto& error : shard.errors) {
                    if (result.errors.size() < options.maximum_errors) {
                        result.errors.push_back(std::move(error));
                    }
                }
            }
            continue;
        }
        ++out.scanned_shards;
        out.shard_file_bytes += shard.value.file_size;
        for (const auto& source : shard.value.tensors) {
            const auto found = indexed.find(source.name);
            if (found == indexed.end()) {
                if (result.errors.size() < options.maximum_errors) {
                    result.errors.push_back("unindexed GLM-5.3 tensor " + source.name);
                }
                continue;
            }
            auto& target = out.tensors[found->second];
            if (target.shard != shard_name) {
                result.errors.push_back("misplaced GLM-5.3 tensor " + source.name);
                continue;
            }
            target.source_dtype = source.dtype;
            target.source_shape = source.shape;
            target.source_offset = source.absolute_begin;
            target.source_bytes = source.bytes();
            if (target.role == Glm53TensorRole::Vision) out.vision_bytes += target.source_bytes;
            else if (target.role == Glm53TensorRole::RoutedExpert) out.routed_expert_bytes += target.source_bytes;
            else out.dense_spine_bytes += target.source_bytes;
            out.tensor_payload_bytes += target.source_bytes;
            ++resolved;
        }
    }
    if (!result.errors.empty()) return result;
    const auto* profile = release_profile(out.quantization);
    if (profile == nullptr || out.scanned_shards != profile->shard_count ||
        out.shard_file_bytes != profile->shard_file_bytes ||
        resolved != out.tensors.size() ||
        out.tensor_payload_bytes != profile->payload_bytes) {
        result.errors.emplace_back(
            "GLM-5.3 shard or tensor extent does not match the pinned Flash release");
        return result;
    }
    const auto expect = [&](std::string_view name,
                            std::vector<std::uint64_t> shape,
                            SafetensorsDtype dtype) {
        const auto found = indexed.find(name);
        if (found == indexed.end()) {
            result.errors.push_back("GLM-5.3 checkpoint is missing " + std::string(name));
            return;
        }
        const auto& tensor = out.tensors[found->second];
        if (tensor.source_shape != shape || tensor.source_dtype != dtype) {
            result.errors.push_back("GLM-5.3 tensor " + std::string(name) +
                                    " has an unexpected shape or dtype");
        }
    };
    expect("model.language_model.embed_tokens.weight", {154880U, 4096U},
           SafetensorsDtype::Bf16);
    expect("model.language_model.norm.weight", {4096U}, SafetensorsDtype::Bf16);
    expect("lm_head.weight", {154880U, 4096U}, SafetensorsDtype::Bf16);
    for (const auto layer : {0U, 3U, 44U}) {
        const auto prefix = "model.language_model.layers." +
                            std::to_string(layer) + ".";
        expect(prefix + "hc_attn_fn", {24U, 16384U}, SafetensorsDtype::Bf16);
        expect(prefix + "hc_ffn_fn", {24U, 16384U}, SafetensorsDtype::Bf16);
        expect(prefix + "input_layernorm.weight", {4096U}, SafetensorsDtype::Bf16);
        if (glm53_kda_layer(layer)) {
            expect(prefix + "self_attn.q_proj.weight", {8192U, 4096U}, SafetensorsDtype::Bf16);
            expect(prefix + "self_attn.A_log", {64U}, SafetensorsDtype::F32);
        } else {
            // Only the routed experts are quantized in the two FP4 releases,
            // so their attention spine is BF16 where the FP8 release is E4M3.
            expect(prefix + "self_attn.q_a_proj.weight", {1536U, 4096U},
                   out.quantization == Glm53Quantization::Fp8E4m3Block128
                       ? SafetensorsDtype::F8E4M3
                       : SafetensorsDtype::Bf16);
            expect(prefix + "self_attn.kv_b_proj.weight", {32768U, 512U}, SafetensorsDtype::Bf16);
        }
        if (result.errors.size() >= options.maximum_errors) break;
    }
    // Layer 4 is the first MXFP4-quantized MoE layer: layers 3, 5 and 6 keep
    // BF16 routed experts under the publisher's mixed-precision correction, so
    // sampling layer 3 would prove nothing about the packed layout.
    if (out.quantization == Glm53Quantization::Mxfp4Group32) {
        const std::string expert =
            "model.language_model.layers.4.mlp.experts.0.";
        expect(expert + "gate_proj.weight", {2048U, 2048U}, SafetensorsDtype::U8);
        expect(expert + "gate_proj.weight_scale", {2048U, 128U},
               SafetensorsDtype::U8);
        expect(expert + "down_proj.weight", {4096U, 1024U}, SafetensorsDtype::U8);
        expect(expert + "down_proj.weight_scale", {4096U, 64U},
               SafetensorsDtype::U8);
    }
    // The NVFP4 release packs every MoE layer, so layer 3 -- the first one, and
    // the one the MXFP4 release leaves BF16 -- is the sample that proves it.
    // Group 16 halves the columns per scale byte against MXFP4's 32 and adds
    // the scalar F32 divisor, so all three tensors are checked per projection.
    if (out.quantization == Glm53Quantization::Nvfp4Group16E4m3) {
        const std::string expert =
            "model.language_model.layers.3.mlp.experts.0.";
        expect(expert + "gate_proj.weight_packed", {2048U, 2048U},
               SafetensorsDtype::U8);
        expect(expert + "gate_proj.weight_scale", {2048U, 256U},
               SafetensorsDtype::F8E4M3);
        expect(expert + "gate_proj.weight_global_scale", {1U},
               SafetensorsDtype::F32);
        expect(expert + "down_proj.weight_packed", {4096U, 1024U},
               SafetensorsDtype::U8);
        expect(expert + "down_proj.weight_scale", {4096U, 128U},
               SafetensorsDtype::F8E4M3);
        expect(expert + "down_proj.weight_global_scale", {1U},
               SafetensorsDtype::F32);
        // The shared expert stays BF16 in this release, which is why the host
        // and device expert paths must keep meeting both encodings per layer.
        expect("model.language_model.layers.3.mlp.shared_experts.gate_proj.weight",
               {2048U, 4096U}, SafetensorsDtype::Bf16);
    }
    return result;
}

std::string_view to_string(Glm53TensorRole role) noexcept {
    switch (role) {
        case Glm53TensorRole::Embedding: return "embedding";
        case Glm53TensorRole::OutputHead: return "output-head";
        case Glm53TensorRole::Norm: return "norm";
        case Glm53TensorRole::Mhc: return "mhc";
        case Glm53TensorRole::KdaAttention: return "kda";
        case Glm53TensorRole::SparseAttention: return "sparse-attention";
        case Glm53TensorRole::AttentionIndexer: return "attention-indexer";
        case Glm53TensorRole::DenseMlp: return "dense-mlp";
        case Glm53TensorRole::Router: return "router";
        case Glm53TensorRole::SharedExpert: return "shared-expert";
        case Glm53TensorRole::RoutedExpert: return "routed-expert";
        case Glm53TensorRole::Mtp: return "mtp";
        case Glm53TensorRole::Vision: return "vision";
        case Glm53TensorRole::Count: break;
    }
    return "unknown";
}

std::string_view to_string(Glm53Quantization quantization) noexcept {
    switch (quantization) {
        case Glm53Quantization::Fp8E4m3Block128: return "fp8-e4m3-block128";
        case Glm53Quantization::Mxfp4Group32: return "mxfp4-e2m1-group32";
        case Glm53Quantization::Nvfp4Group16E4m3: return "nvfp4-e2m1-group16";
        case Glm53Quantization::Unsupported: break;
    }
    return "unsupported";
}

std::string_view to_string(Glm53TensorEncoding encoding) noexcept {
    switch (encoding) {
        case Glm53TensorEncoding::Plain: return "plain";
        case Glm53TensorEncoding::Fp4E2m1Group32E8m0:
            return "fp4-e2m1-group32-e8m0";
        case Glm53TensorEncoding::Fp4E2m1Group16E4m3:
            return "fp4-e2m1-group16-e4m3";
        case Glm53TensorEncoding::Fp8E4m3Block128F32:
            return "fp8-e4m3-block128-f32-scale";
    }
    return "unknown";
}

}  // namespace strata
