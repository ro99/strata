#include "strata/kimi_k3_manifest.hpp"

#include "json_cursor.hpp"

#include "strata/model_adapter.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>

namespace strata {
namespace {

using detail::JsonCursor;

constexpr std::string_view kTextPrefix = "language_model.model.layers.";
constexpr std::string_view kVisionPrefix = "vision_tower.";

// Parses the decimal run starting at `cursor` and leaves `cursor` on the first
// character after it. Returns false when there is no digit there.
[[nodiscard]] bool take_index(std::string_view name, std::size_t& cursor,
                              std::uint32_t& value) noexcept {
    const auto start = cursor;
    while (cursor < name.size() && name[cursor] >= '0' && name[cursor] <= '9') {
        ++cursor;
    }
    if (cursor == start) return false;
    const auto parsed = std::from_chars(name.data() + start, name.data() + cursor,
                                        value);
    return parsed.ec == std::errc();
}

[[nodiscard]] bool has_suffix(std::string_view name,
                              std::string_view suffix) noexcept {
    return name.size() >= suffix.size() &&
           name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

[[nodiscard]] KimiTensorRole classify_attention(std::string_view leaf) noexcept {
    // Every layer carries `g_proj` and `o_proj`; only the layer kind tells them
    // apart, so the caller resolves those two by layer index.
    if (leaf.starts_with("q_a_proj") || leaf.starts_with("q_b_proj") ||
        leaf.starts_with("kv_a_proj_with_mqa") || leaf.starts_with("kv_b_proj") ||
        leaf.starts_with("q_a_layernorm") || leaf.starts_with("kv_a_layernorm")) {
        return KimiTensorRole::MlaAttention;
    }
    if (leaf.starts_with("q_proj") || leaf.starts_with("k_proj") ||
        leaf.starts_with("v_proj") || leaf.starts_with("b_proj") ||
        leaf.starts_with("f_a_proj") || leaf.starts_with("f_b_proj") ||
        leaf.starts_with("q_conv1d") || leaf.starts_with("k_conv1d") ||
        leaf.starts_with("v_conv1d") || leaf.starts_with("A_log") ||
        leaf.starts_with("dt_bias") || leaf.starts_with("o_norm")) {
        return KimiTensorRole::KdaAttention;
    }
    return KimiTensorRole::Count;
}

[[nodiscard]] KimiTensorComponent classify_component(
    std::string_view name) noexcept {
    if (has_suffix(name, ".weight_packed")) return KimiTensorComponent::PackedWeight;
    if (has_suffix(name, ".weight_scale")) return KimiTensorComponent::Scale;
    if (has_suffix(name, "_bias") || has_suffix(name, ".bias")) {
        return KimiTensorComponent::Bias;
    }
    return KimiTensorComponent::Weight;
}

void parse_uint_list(JsonCursor& cursor, std::vector<std::uint32_t>& values) {
    cursor.expect('[');
    if (cursor.consume(']')) return;
    for (;;) {
        values.push_back(static_cast<std::uint32_t>(cursor.parse_uint64()));
        if (cursor.consume(']')) return;
        cursor.expect(',');
    }
}

void parse_linear_attention(JsonCursor& cursor, KimiK3Config& config) {
    cursor.expect('{');
    if (cursor.consume('}')) return;
    for (;;) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        if (key == "head_dim") {
            config.linear_head_dim = static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "num_heads") {
            config.linear_attention_heads =
                static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "short_conv_kernel_size") {
            config.short_conv_kernel =
                static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "gate_lower_bound") {
            config.kda_gate_lower_bound = static_cast<float>(cursor.parse_number());
        } else if (key == "use_full_rank_gate") {
            config.use_full_rank_gate = cursor.parse_bool();
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

void parse_group_weights(JsonCursor& cursor, KimiK3Config& config) {
    cursor.expect('{');
    if (cursor.consume('}')) return;
    for (;;) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        if (key == "group_size") {
            config.quantization_group_size =
                static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "num_bits") {
            config.quantization_bits =
                static_cast<std::uint32_t>(cursor.parse_uint64());
        } else {
            cursor.skip_value();
        }
        if (cursor.consume('}')) return;
        cursor.expect(',');
    }
}

void parse_quantization(JsonCursor& cursor, KimiK3Config& config) {
    cursor.expect('{');
    if (cursor.consume('}')) return;
    for (;;) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        if (key == "format") {
            config.quantization_format = cursor.parse_string();
        } else if (key == "config_groups") {
            // One group named `group_0`, whose `weights` object carries the
            // bit width and group size the codec must agree with.
            cursor.expect('{');
            if (!cursor.consume('}')) {
                for (;;) {
                    (void)cursor.parse_string();
                    cursor.expect(':');
                    cursor.expect('{');
                    if (!cursor.consume('}')) {
                        for (;;) {
                            const auto inner = cursor.parse_string();
                            cursor.expect(':');
                            if (inner == "weights") {
                                parse_group_weights(cursor, config);
                            } else {
                                cursor.skip_value();
                            }
                            if (cursor.consume('}')) break;
                            cursor.expect(',');
                        }
                    }
                    if (cursor.consume('}')) break;
                    cursor.expect(',');
                }
            }
        } else {
            cursor.skip_value();
        }
        if (cursor.consume('}')) return;
        cursor.expect(',');
    }
}

void parse_vision(JsonCursor& cursor, KimiK3Config& config) {
    cursor.expect('{');
    if (cursor.consume('}')) return;
    for (;;) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        if (key == "vt_hidden_size") {
            config.vision_hidden_size =
                static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "vt_num_hidden_layers") {
            config.vision_layer_count =
                static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "vt_num_attention_heads") {
            config.vision_attention_heads =
                static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "qkv_hidden_size") {
            config.vision_qkv_hidden_size =
                static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "vt_intermediate_size") {
            config.vision_intermediate_size =
                static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "patch_size") {
            config.vision_patch_size =
                static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "merge_kernel_size") {
            std::vector<std::uint32_t> merge;
            parse_uint_list(cursor, merge);
            // Square merge only; anything else changes the projector width.
            config.vision_merge_kernel =
                merge.size() == 2U && merge[0] == merge[1] ? merge[0] : 0U;
        } else {
            cursor.skip_value();
        }
        if (cursor.consume('}')) return;
        cursor.expect(',');
    }
}

void parse_text(JsonCursor& cursor, KimiK3Config& config) {
    cursor.expect('{');
    if (cursor.consume('}')) return;
    for (;;) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        if (key == "hidden_size") {
            config.hidden_size = static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "num_hidden_layers") {
            config.layer_count = static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "num_attention_heads") {
            config.attention_heads = static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "num_key_value_heads") {
            config.key_value_heads = static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "q_lora_rank") {
            config.query_lora_rank = static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "kv_lora_rank") {
            config.kv_lora_rank = static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "qk_nope_head_dim") {
            config.nope_head_dim = static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "qk_rope_head_dim") {
            config.rope_head_dim = static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "v_head_dim") {
            config.value_head_dim = static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "intermediate_size") {
            config.dense_intermediate_size =
                static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "moe_intermediate_size") {
            config.expert_intermediate_size =
                static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "routed_expert_hidden_size") {
            config.routed_expert_hidden_size =
                static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "num_experts") {
            config.routed_experts = static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "num_experts_per_token") {
            config.experts_per_token =
                static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "num_expert_group") {
            config.expert_groups = static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "topk_group") {
            config.selected_expert_groups =
                static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "num_shared_experts") {
            config.shared_experts = static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "first_k_dense_replace") {
            config.dense_prefix_layers =
                static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "attn_res_block_size") {
            config.attention_residual_block_size =
                static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "vocab_size") {
            config.vocabulary_size = static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "max_position_embeddings") {
            config.maximum_context_tokens =
                static_cast<std::uint32_t>(cursor.parse_uint64());
        } else if (key == "rms_norm_eps") {
            config.rms_epsilon = static_cast<float>(cursor.parse_number());
        } else if (key == "routed_scaling_factor") {
            config.routed_scale = static_cast<float>(cursor.parse_number());
        } else if (key == "activation_situ_beta") {
            config.situ_gate_beta = static_cast<float>(cursor.parse_number());
        } else if (key == "activation_situ_linear_beta") {
            config.situ_linear_beta = static_cast<float>(cursor.parse_number());
        } else if (key == "mla_use_nope") {
            config.mla_use_nope = cursor.parse_bool();
        } else if (key == "mla_use_output_gate") {
            config.mla_use_output_gate = cursor.parse_bool();
        } else if (key == "latent_moe_use_norm") {
            config.latent_moe_use_norm = cursor.parse_bool();
        } else if (key == "moe_renormalize") {
            config.moe_renormalize = cursor.parse_bool();
        } else if (key == "tie_word_embeddings") {
            config.tie_word_embeddings = cursor.parse_bool();
        } else if (key == "hidden_act") {
            config.hidden_activation = cursor.parse_string();
        } else if (key == "moe_router_activation_func") {
            config.router_activation = cursor.parse_string();
        } else if (key == "topk_method") {
            config.topk_method = cursor.parse_string();
        } else if (key == "linear_attn_config") {
            parse_linear_attention(cursor, config);
        } else if (key == "quantization_config") {
            parse_quantization(cursor, config);
        } else {
            cursor.skip_value();
        }
        if (cursor.consume('}')) return;
        cursor.expect(',');
    }
}

}  // namespace

ModelSpec kimi_k3_mxfp4_spec() {
    ModelSpec spec;
    const auto& contract = kKimiK3ExecutionContract;
    spec.name = "moonshotai/Kimi-K3";
    spec.architecture = ArchitectureKind::StandardMoe;
    spec.attention = AttentionKind::HybridCompressedSparse;
    spec.router.selection = RouterSelectionKind::NoAuxTc;
    spec.router.scoring = RouterScoreKind::Sigmoid;
    spec.router.routed_experts = contract.routed_experts;
    spec.router.experts_per_token = contract.experts_per_token;
    spec.router.groups = contract.expert_groups;
    spec.router.selected_groups = contract.selected_expert_groups;
    spec.router.normalize_topk = true;
    spec.router.selection_bias = true;
    spec.router.routed_scale = contract.routed_scale;
    // Only the routed experts are quantized. The whole dense spine — attention,
    // shared experts, the dense MLP, the latent projections, lm_head, and the
    // vision tower — ships as plain BF16, which is why it dominates placement.
    spec.mixed_quantization.kind = QuantizationKind::CompressedTensorsW4A16;
    spec.mixed_quantization.activation_bits = 16U;
    spec.mixed_quantization.quantized_linear_start_layer = contract.layer_count;
    spec.mixed_quantization.quantized_expert_start_layer =
        contract.dense_prefix_layers;
    spec.mixed_quantization.mtp_layer_index = contract.layer_count;
    spec.mixed_quantization.routed_experts = {
        4U, QuantizationGranularity::Group, 32U, true};
    spec.mixed_quantization.linears = {16U, QuantizationGranularity::Channel, 0U,
                                       true};
    spec.mixed_quantization.mtp = spec.mixed_quantization.linears;
    spec.source.repository = "moonshotai/Kimi-K3";
    spec.source.index_sha256 =
        "a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd";
    spec.source.tensor_count = 497'220U;
    spec.source.indexed_tensor_bytes = 1'560'860'324'864ULL;
    spec.source.shard_file_bytes = 1'560'936'091'448ULL;
    spec.source.main_shards = 96U;
    spec.quant_bits = 4U;
    spec.hidden_size = contract.hidden_size;
    spec.layer_count = contract.layer_count;
    spec.max_context_tokens = contract.maximum_context_tokens;
    spec.dense_prefix_layers = contract.dense_prefix_layers;
    spec.shared_experts = contract.shared_experts;
    spec.expert_intermediate_size = contract.expert_intermediate_size;
    return spec;
}

ValidationResult validate_kimi_k3_mxfp4(const ModelSpec& spec) {
    auto result = validate_model(spec);
    const auto expected = kimi_k3_mxfp4_spec();
    const auto require = [&result](bool condition, std::string_view message) {
        if (!condition) result.errors.emplace_back(message);
    };
    require(spec.source.repository == expected.source.repository &&
                spec.source.index_sha256 == expected.source.index_sha256,
            "unexpected Kimi-K3 source identity");
    require(spec.source.tensor_count == expected.source.tensor_count &&
                spec.source.indexed_tensor_bytes ==
                    expected.source.indexed_tensor_bytes &&
                spec.source.shard_file_bytes == expected.source.shard_file_bytes &&
                spec.source.main_shards == expected.source.main_shards,
            "unexpected Kimi-K3 checkpoint extent");
    require(spec.architecture == expected.architecture &&
                spec.attention == expected.attention &&
                spec.hidden_size == expected.hidden_size &&
                spec.layer_count == expected.layer_count &&
                spec.dense_prefix_layers == expected.dense_prefix_layers &&
                spec.shared_experts == expected.shared_experts &&
                spec.expert_intermediate_size == expected.expert_intermediate_size,
            "unexpected Kimi-K3 architecture geometry");
    require(spec.router.selection == expected.router.selection &&
                spec.router.scoring == expected.router.scoring &&
                spec.router.routed_experts == expected.router.routed_experts &&
                spec.router.experts_per_token == expected.router.experts_per_token &&
                spec.router.groups == expected.router.groups &&
                spec.router.selected_groups == expected.router.selected_groups &&
                spec.router.normalize_topk == expected.router.normalize_topk &&
                spec.router.selection_bias == expected.router.selection_bias &&
                spec.router.routed_scale == expected.router.routed_scale,
            "unexpected Kimi-K3 router semantics");
    require(spec.quant_bits == expected.quant_bits &&
                spec.mixed_quantization.routed_experts.bits ==
                    expected.mixed_quantization.routed_experts.bits &&
                spec.mixed_quantization.routed_experts.group_size ==
                    expected.mixed_quantization.routed_experts.group_size &&
                spec.mixed_quantization.routed_experts.symmetric,
            "unexpected Kimi-K3 routed-expert quantization");
    return result;
}

KimiConfigResult parse_kimi_k3_config(std::string_view json) {
    KimiConfigResult result;
    try {
        JsonCursor cursor(json);
        cursor.expect('{');
        if (!cursor.consume('}')) {
            for (;;) {
                const auto key = cursor.parse_string();
                cursor.expect(':');
                if (key == "text_config") {
                    parse_text(cursor, result.value);
                } else if (key == "vision_config") {
                    parse_vision(cursor, result.value);
                } else if (key == "media_placeholder_token_id") {
                    result.value.image_token_id =
                        static_cast<std::uint32_t>(cursor.parse_uint64());
                } else {
                    cursor.skip_value();
                }
                if (cursor.consume('}')) break;
                cursor.expect(',');
            }
        }
    } catch (const detail::JsonError& error) {
        result.errors.push_back("Kimi-K3 config.json is malformed at offset " +
                                std::to_string(error.offset()) + ": " +
                                error.what());
        return result;
    }
    // The checkpoint lists attention layers 1-based. Convert once, here, so no
    // caller has to remember which convention it is holding.
    const auto rebase = [&result](std::vector<std::uint32_t>& layers,
                                  std::string_view label) {
        for (auto& layer : layers) {
            if (layer == 0U) {
                result.errors.push_back(std::string(label) +
                                        " is 1-based and cannot contain zero");
                return;
            }
            --layer;
        }
        std::sort(layers.begin(), layers.end());
    };
    rebase(result.value.full_attention_layers, "full_attn_layers");
    rebase(result.value.kda_layers, "kda_layers");
    return result;
}

ValidationResult validate_kimi_k3_config(const KimiK3Config& config) {
    ValidationResult result;
    const auto& c = kKimiK3ExecutionContract;
    const auto require = [&result](bool condition, std::string message) {
        if (!condition) result.errors.push_back(std::move(message));
    };
    const auto equal = [&require](std::uint32_t actual, std::uint32_t expected,
                                  std::string_view field) {
        require(actual == expected, "Kimi-K3 config " + std::string(field) +
                                        " is " + std::to_string(actual) +
                                        ", expected " + std::to_string(expected));
    };
    equal(config.hidden_size, c.hidden_size, "hidden_size");
    equal(config.layer_count, c.layer_count, "num_hidden_layers");
    equal(config.attention_heads, c.attention_heads, "num_attention_heads");
    equal(config.key_value_heads, c.key_value_heads, "num_key_value_heads");
    equal(config.query_lora_rank, c.query_lora_rank, "q_lora_rank");
    equal(config.kv_lora_rank, c.kv_lora_rank, "kv_lora_rank");
    equal(config.nope_head_dim, c.nope_head_dim, "qk_nope_head_dim");
    equal(config.rope_head_dim, c.rope_head_dim, "qk_rope_head_dim");
    equal(config.value_head_dim, c.value_head_dim, "v_head_dim");
    equal(config.linear_attention_heads, c.linear_attention_heads,
          "linear_attn_config.num_heads");
    equal(config.linear_head_dim, c.linear_head_dim, "linear_attn_config.head_dim");
    equal(config.short_conv_kernel, c.short_conv_kernel,
          "linear_attn_config.short_conv_kernel_size");
    equal(config.dense_intermediate_size, c.dense_intermediate_size,
          "intermediate_size");
    equal(config.expert_intermediate_size, c.expert_intermediate_size,
          "moe_intermediate_size");
    equal(config.routed_expert_hidden_size, c.routed_expert_hidden_size,
          "routed_expert_hidden_size");
    equal(config.routed_experts, c.routed_experts, "num_experts");
    equal(config.experts_per_token, c.experts_per_token, "num_experts_per_token");
    equal(config.expert_groups, c.expert_groups, "num_expert_group");
    equal(config.selected_expert_groups, c.selected_expert_groups, "topk_group");
    equal(config.shared_experts, c.shared_experts, "num_shared_experts");
    equal(config.dense_prefix_layers, c.dense_prefix_layers,
          "first_k_dense_replace");
    equal(config.attention_residual_block_size, c.attention_residual_block_size,
          "attn_res_block_size");
    equal(config.vocabulary_size, c.vocabulary_size, "vocab_size");
    equal(config.maximum_context_tokens, c.maximum_context_tokens,
          "max_position_embeddings");
    equal(config.image_token_id, c.image_token_id, "media_placeholder_token_id");
    equal(config.vision_hidden_size, c.vision_hidden_size, "vt_hidden_size");
    equal(config.vision_layer_count, c.vision_layer_count, "vt_num_hidden_layers");
    equal(config.vision_attention_heads, c.vision_attention_heads,
          "vt_num_attention_heads");
    equal(config.vision_qkv_hidden_size, c.vision_qkv_hidden_size,
          "qkv_hidden_size");
    equal(config.vision_intermediate_size, c.vision_intermediate_size,
          "vt_intermediate_size");
    equal(config.vision_patch_size, c.vision_patch_size, "patch_size");
    equal(config.vision_merge_kernel, c.vision_merge_kernel, "merge_kernel_size");

    const auto close = [&require](float actual, float expected,
                                  std::string_view field) {
        require(std::fabs(actual - expected) <=
                    std::fabs(expected) * 1.0e-6F + 1.0e-12F,
                "Kimi-K3 config " + std::string(field) + " is " +
                    std::to_string(actual) + ", expected " +
                    std::to_string(expected));
    };
    close(config.rms_epsilon, c.rms_epsilon, "rms_norm_eps");
    close(config.routed_scale, c.routed_scale, "routed_scaling_factor");
    close(config.situ_gate_beta, c.situ_gate_beta, "activation_situ_beta");
    close(config.situ_linear_beta, c.situ_linear_beta,
          "activation_situ_linear_beta");
    close(config.kda_gate_lower_bound, c.kda_gate_lower_bound,
          "linear_attn_config.gate_lower_bound");

    // Semantics, not sizes. A checkpoint that flips any of these is a different
    // model and must be refused rather than run with Strata's assumptions.
    require(config.hidden_activation == "situ",
            "Kimi-K3 requires the SiTU-GLU activation; a SwiGLU substitution "
            "would silently change expert semantics");
    require(config.router_activation == "sigmoid",
            "Kimi-K3 router scores must be sigmoid");
    require(config.topk_method == "noaux_tc",
            "Kimi-K3 routing must use the frozen-bias noaux_tc selection");
    require(config.moe_renormalize,
            "Kimi-K3 requires top-k renormalization of routed weights");
    require(config.latent_moe_use_norm,
            "Kimi-K3 requires the LatentMoE norm between aggregation and "
            "up-projection");
    require(config.mla_use_nope,
            "Kimi-K3 MLA layers must be NoPE; position information rides the "
            "KDA layers");
    require(config.mla_use_output_gate, "Kimi-K3 MLA requires the output gate");
    require(config.use_full_rank_gate,
            "Kimi-K3 KDA requires the full-rank output gate");
    require(!config.tie_word_embeddings,
            "Kimi-K3 does not tie the output head to the embedding table");
    require(config.quantization_format == "mxfp4-pack-quantized",
            "Kimi-K3 routed experts must be mxfp4-pack-quantized");
    require(config.quantization_bits == 4U,
            "Kimi-K3 expert weights must be four bits; below four is forbidden");
    equal(config.quantization_group_size, 32U, "quantization group_size");

    // The layer schedule is taken from the config's own lists, not inferred
    // from the repeat pattern, and then checked against the contract's
    // predicate so a checkpoint with a different hybrid layout is rejected.
    require(config.full_attention_layers.size() +
                    config.kda_layers.size() ==
                c.layer_count,
            "Kimi-K3 attention layer lists do not partition the backbone");
    for (const auto layer : config.full_attention_layers) {
        require(kimi_k3_full_attention_layer(layer),
                "Kimi-K3 full-attention layer " + std::to_string(layer) +
                    " is not where the contract places one");
    }
    for (const auto layer : config.kda_layers) {
        require(kimi_k3_kda_layer(layer),
                "Kimi-K3 KDA layer " + std::to_string(layer) +
                    " is not where the contract places one");
    }
    return result;
}

KimiTensorRole classify_kimi_tensor(std::string_view name, std::int32_t& layer,
                                    std::int32_t& expert) noexcept {
    layer = -1;
    expert = -1;
    if (name.starts_with(kVisionPrefix)) return KimiTensorRole::Vision;
    if (name.starts_with("mm_projector.")) return KimiTensorRole::VisionProjector;
    if (name == "language_model.model.embed_tokens.weight") {
        return KimiTensorRole::Embedding;
    }
    if (name == "language_model.lm_head.weight") return KimiTensorRole::OutputHead;
    if (name == "language_model.model.norm.weight") return KimiTensorRole::Norm;
    if (name.starts_with("language_model.model.output_attn_res_")) {
        return KimiTensorRole::AttentionResidual;
    }
    if (!name.starts_with(kTextPrefix)) return KimiTensorRole::Count;

    std::size_t cursor = kTextPrefix.size();
    std::uint32_t index = 0U;
    if (!take_index(name, cursor, index)) return KimiTensorRole::Count;
    if (cursor >= name.size() || name[cursor] != '.') return KimiTensorRole::Count;
    layer = static_cast<std::int32_t>(index);
    const auto rest = name.substr(cursor + 1U);

    if (rest.starts_with("self_attn.")) {
        const auto leaf = rest.substr(std::string_view("self_attn.").size());
        // `g_proj` and `o_proj` exist on every layer and belong to whichever
        // attention the layer runs, so they are resolved by layer kind.
        if (leaf.starts_with("g_proj") || leaf.starts_with("o_proj")) {
            return kimi_k3_full_attention_layer(index)
                       ? KimiTensorRole::MlaAttention
                       : KimiTensorRole::KdaAttention;
        }
        return classify_attention(leaf);
    }
    if (rest.starts_with("input_layernorm") ||
        rest.starts_with("post_attention_layernorm")) {
        return KimiTensorRole::Norm;
    }
    if (rest.starts_with("self_attention_res_") || rest.starts_with("mlp_res_")) {
        return KimiTensorRole::AttentionResidual;
    }
    if (rest.starts_with("mlp.")) return KimiTensorRole::DenseMlp;
    if (rest.starts_with("block_sparse_moe.")) {
        const auto leaf = rest.substr(std::string_view("block_sparse_moe.").size());
        if (leaf.starts_with("gate.")) return KimiTensorRole::Router;
        if (leaf.starts_with("shared_experts.")) return KimiTensorRole::SharedExpert;
        if (leaf.starts_with("routed_expert_")) {
            return KimiTensorRole::LatentMoeProjection;
        }
        if (leaf.starts_with("experts.")) {
            std::size_t inner = std::string_view("experts.").size();
            std::uint32_t ordinal = 0U;
            if (!take_index(leaf, inner, ordinal)) return KimiTensorRole::Count;
            expert = static_cast<std::int32_t>(ordinal);
            return KimiTensorRole::RoutedExpert;
        }
    }
    return KimiTensorRole::Count;
}

KimiManifestResult build_kimi_k3_index_manifest(SafetensorsIndex index) {
    KimiManifestResult result;
    auto& manifest = result.manifest;
    const auto& c = kKimiK3ExecutionContract;
    manifest.indexed_tensor_bytes = index.total_size;
    manifest.shards = std::move(index.shards);
    manifest.tensors.reserve(index.entries.size());

    std::unordered_set<std::uint32_t> kda_seen;
    std::unordered_set<std::uint32_t> full_seen;
    std::unordered_set<std::uint32_t> moe_seen;
    for (auto& entry : index.entries) {
        std::int32_t layer = -1;
        std::int32_t expert = -1;
        const auto role = classify_kimi_tensor(entry.name, layer, expert);
        if (role == KimiTensorRole::Count) {
            if (result.errors.size() < 64U) {
                result.errors.push_back("unclassified Kimi-K3 tensor " + entry.name);
            }
            continue;
        }
        KimiManifestTensor tensor;
        tensor.role = role;
        tensor.component = classify_component(entry.name);
        tensor.layer = layer;
        tensor.expert = expert;
        tensor.encoding = role == KimiTensorRole::RoutedExpert
                              ? KimiTensorEncoding::Mxfp4Group32
                              : KimiTensorEncoding::PlainBf16;
        tensor.name = std::move(entry.name);
        tensor.shard = std::move(entry.shard);
        manifest.tensors.push_back(std::move(tensor));
        ++manifest.role_counts[static_cast<std::size_t>(role)];
        if (layer >= 0) {
            const auto ordinal = static_cast<std::uint32_t>(layer);
            if (role == KimiTensorRole::KdaAttention) kda_seen.insert(ordinal);
            if (role == KimiTensorRole::MlaAttention) full_seen.insert(ordinal);
            if (role == KimiTensorRole::RoutedExpert) moe_seen.insert(ordinal);
        }
        if (role == KimiTensorRole::RoutedExpert &&
            manifest.tensors.back().component ==
                KimiTensorComponent::PackedWeight) {
            ++manifest.routed_expert_modules;
        }
    }
    manifest.kda_layers = kda_seen.size();
    manifest.full_attention_layers = full_seen.size();
    manifest.moe_layers = moe_seen.size();
    if (manifest.kda_layers != c.layer_count - 24U) {
        result.errors.push_back("Kimi-K3 index declares " +
                                std::to_string(manifest.kda_layers) +
                                " KDA layers, expected 69");
    }
    if (manifest.full_attention_layers != 24U) {
        result.errors.push_back("Kimi-K3 index declares " +
                                std::to_string(manifest.full_attention_layers) +
                                " gated MLA layers, expected 24");
    }
    if (manifest.moe_layers != c.layer_count - c.dense_prefix_layers) {
        result.errors.push_back("Kimi-K3 index declares " +
                                std::to_string(manifest.moe_layers) +
                                " MoE layers, expected 92");
    }
    return result;
}

KimiManifestResult validate_kimi_k3_checkpoint(
    const std::string& model_directory, KimiIndexManifest manifest,
    const KimiCheckpointOptions& options) {
    KimiManifestResult result;
    result.manifest = std::move(manifest);
    auto& out = result.manifest;
    const auto& c = kKimiK3ExecutionContract;

    std::unordered_map<std::string_view, std::size_t> by_name;
    by_name.reserve(out.tensors.size());
    for (std::size_t index = 0U; index < out.tensors.size(); ++index) {
        if (!by_name.emplace(out.tensors[index].name, index).second) {
            result.errors.push_back("duplicate Kimi-K3 tensor " +
                                    out.tensors[index].name);
        }
    }
    if (!result.errors.empty()) return result;

    std::uint64_t resolved = 0U;
    for (const auto& shard_name : out.shards) {
        const auto path =
            (std::filesystem::path(model_directory) / shard_name).string();
        auto shard = load_safetensors_shard(path);
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
        for (const auto& tensor : shard.value.tensors) {
            const auto found = by_name.find(tensor.name);
            if (found == by_name.end()) {
                if (result.errors.size() < options.maximum_errors) {
                    result.errors.push_back("unindexed Kimi-K3 tensor " +
                                            tensor.name);
                }
                continue;
            }
            auto& entry = out.tensors[found->second];
            if (entry.shard != shard_name) {
                if (result.errors.size() < options.maximum_errors) {
                    result.errors.push_back("misplaced Kimi-K3 tensor " +
                                            tensor.name);
                }
                continue;
            }
            entry.source_dtype = tensor.dtype;
            entry.source_shape = tensor.shape;
            entry.source_offset = tensor.absolute_begin;
            entry.source_bytes = tensor.bytes();
            if (entry.role == KimiTensorRole::RoutedExpert) {
                out.routed_expert_bytes += entry.source_bytes;
            } else {
                out.dense_spine_bytes += entry.source_bytes;
            }
            out.tensor_payload_bytes += entry.source_bytes;
            ++resolved;
        }
    }
    if (!result.errors.empty()) return result;
    if (resolved != out.tensors.size()) {
        result.errors.push_back("Kimi-K3 index lists " +
                                std::to_string(out.tensors.size()) +
                                " tensors but the shards resolved " +
                                std::to_string(resolved));
        return result;
    }

    // Shapes, not just names. The routed-expert geometry is what the MXFP4
    // codec indexes into, and the dense spine's dtype is what makes it 103 GiB
    // of the placement problem, so both are checked against the contract.
    const auto expect_shape = [&](std::string_view name,
                                  std::vector<std::uint64_t> shape,
                                  SafetensorsDtype dtype) {
        const auto found = by_name.find(name);
        if (found == by_name.end()) {
            result.errors.push_back("Kimi-K3 checkpoint is missing " +
                                    std::string(name));
            return;
        }
        const auto& entry = out.tensors[found->second];
        if (entry.source_shape != shape || entry.source_dtype != dtype) {
            result.errors.push_back("Kimi-K3 tensor " + std::string(name) +
                                    " has an unexpected shape or dtype");
        }
    };
    const auto hidden = static_cast<std::uint64_t>(c.hidden_size);
    const auto heads = static_cast<std::uint64_t>(c.attention_heads);
    const auto head_dim = static_cast<std::uint64_t>(c.value_head_dim);
    expect_shape("language_model.model.embed_tokens.weight",
                 {c.vocabulary_size, hidden}, SafetensorsDtype::Bf16);
    expect_shape("language_model.lm_head.weight", {c.vocabulary_size, hidden},
                 SafetensorsDtype::Bf16);
    for (std::uint32_t layer = 0U; layer < c.layer_count; ++layer) {
        const auto prefix =
            "language_model.model.layers." + std::to_string(layer) + ".";
        const auto attention = prefix + "self_attn.";
        expect_shape(attention + "g_proj.weight", {heads * head_dim, hidden},
                     SafetensorsDtype::Bf16);
        expect_shape(attention + "o_proj.weight", {hidden, heads * head_dim},
                     SafetensorsDtype::Bf16);
        expect_shape(prefix + "self_attention_res_proj.weight", {1U, hidden},
                     SafetensorsDtype::Bf16);
        expect_shape(prefix + "mlp_res_proj.weight", {1U, hidden},
                     SafetensorsDtype::Bf16);
        if (kimi_k3_full_attention_layer(layer)) {
            expect_shape(attention + "q_a_proj.weight", {c.query_lora_rank, hidden},
                         SafetensorsDtype::Bf16);
            expect_shape(attention + "q_b_proj.weight",
                         {heads * (c.nope_head_dim + c.rope_head_dim),
                          c.query_lora_rank},
                         SafetensorsDtype::Bf16);
            expect_shape(attention + "kv_a_proj_with_mqa.weight",
                         {c.kv_lora_rank + c.rope_head_dim, hidden},
                         SafetensorsDtype::Bf16);
            expect_shape(attention + "kv_b_proj.weight",
                         {heads * (c.nope_head_dim + c.value_head_dim),
                          c.kv_lora_rank},
                         SafetensorsDtype::Bf16);
        } else {
            for (const auto* leaf : {"q_proj", "k_proj", "v_proj"}) {
                expect_shape(attention + leaf + ".weight",
                             {heads * head_dim, hidden}, SafetensorsDtype::Bf16);
            }
            for (const auto* leaf : {"q_conv1d", "k_conv1d", "v_conv1d"}) {
                expect_shape(attention + leaf + ".weight",
                             {heads * head_dim, 1U, c.short_conv_kernel},
                             SafetensorsDtype::F32);
            }
            expect_shape(attention + "b_proj.weight", {heads, hidden},
                         SafetensorsDtype::Bf16);
            expect_shape(attention + "f_a_proj.weight", {c.decay_rank, hidden},
                         SafetensorsDtype::Bf16);
            expect_shape(attention + "f_b_proj.weight",
                         {heads * head_dim, c.decay_rank}, SafetensorsDtype::Bf16);
            // Per-head decay scale, stored zero-padded to head_dim. Only the
            // first `attention_heads` entries are live; treating the tensor as
            // per-channel would apply exp(0) = 1 to the padding and corrupt the
            // decay on a quarter of the channels.
            expect_shape(attention + "A_log", {c.linear_head_dim},
                         SafetensorsDtype::F32);
            expect_shape(attention + "dt_bias", {heads * head_dim},
                         SafetensorsDtype::F32);
            expect_shape(attention + "o_norm.weight", {c.linear_head_dim},
                         SafetensorsDtype::F32);
        }
        if (!kimi_k3_moe_layer(layer)) continue;
        const auto moe = prefix + "block_sparse_moe.";
        expect_shape(moe + "gate.weight", {c.routed_experts, hidden},
                     SafetensorsDtype::Bf16);
        expect_shape(moe + "gate.e_score_correction_bias", {c.routed_experts},
                     SafetensorsDtype::F32);
        expect_shape(moe + "routed_expert_down_proj.weight",
                     {c.routed_expert_hidden_size, hidden},
                     SafetensorsDtype::Bf16);
        expect_shape(moe + "routed_expert_up_proj.weight",
                     {hidden, c.routed_expert_hidden_size},
                     SafetensorsDtype::Bf16);
        expect_shape(moe + "routed_expert_norm.weight",
                     {c.routed_expert_hidden_size}, SafetensorsDtype::Bf16);
        const auto shared =
            static_cast<std::uint64_t>(c.shared_experts) * c.expert_intermediate_size;
        expect_shape(moe + "shared_experts.gate_proj.weight", {shared, hidden},
                     SafetensorsDtype::Bf16);
        expect_shape(moe + "shared_experts.up_proj.weight", {shared, hidden},
                     SafetensorsDtype::Bf16);
        expect_shape(moe + "shared_experts.down_proj.weight", {hidden, shared},
                     SafetensorsDtype::Bf16);
        if (result.errors.size() >= options.maximum_errors) return result;
    }
    if (!result.errors.empty()) return result;

    // Routed experts, checked on the layer that has to be right for every
    // other one to be indexable: the packed and scale extents are what the
    // codec strides through.
    const auto latent = static_cast<std::uint64_t>(c.routed_expert_hidden_size);
    const auto inner = static_cast<std::uint64_t>(c.expert_intermediate_size);
    const auto group = 32ULL;
    for (const auto* gate : {"w1", "w3"}) {
        const auto base = "language_model.model.layers.1.block_sparse_moe.experts.0." +
                          std::string(gate) + ".";
        expect_shape(base + "weight_packed", {inner, latent / 2U},
                     SafetensorsDtype::U8);
        expect_shape(base + "weight_scale", {inner, latent / group},
                     SafetensorsDtype::U8);
    }
    expect_shape(
        "language_model.model.layers.1.block_sparse_moe.experts.0.w2.weight_packed",
        {latent, inner / 2U}, SafetensorsDtype::U8);
    expect_shape(
        "language_model.model.layers.1.block_sparse_moe.experts.0.w2.weight_scale",
        {latent, inner / group}, SafetensorsDtype::U8);

    const auto expected_expert_modules =
        static_cast<std::uint64_t>(c.routed_experts) * 3ULL *
        (c.layer_count - c.dense_prefix_layers);
    if (out.routed_expert_modules != expected_expert_modules) {
        result.errors.push_back(
            "Kimi-K3 checkpoint carries " +
            std::to_string(out.routed_expert_modules) +
            " routed-expert modules, expected " +
            std::to_string(expected_expert_modules));
    }
    return result;
}

std::string_view to_string(KimiTensorRole role) noexcept {
    switch (role) {
        case KimiTensorRole::Embedding: return "embedding";
        case KimiTensorRole::OutputHead: return "output-head";
        case KimiTensorRole::Norm: return "norm";
        case KimiTensorRole::MlaAttention: return "gated-mla";
        case KimiTensorRole::KdaAttention: return "kda";
        case KimiTensorRole::AttentionResidual: return "attention-residual";
        case KimiTensorRole::DenseMlp: return "dense-mlp";
        case KimiTensorRole::Router: return "router";
        case KimiTensorRole::SharedExpert: return "shared-expert";
        case KimiTensorRole::LatentMoeProjection: return "latent-moe-projection";
        case KimiTensorRole::RoutedExpert: return "routed-expert";
        case KimiTensorRole::Vision: return "vision";
        case KimiTensorRole::VisionProjector: return "vision-projector";
        case KimiTensorRole::Count: break;
    }
    return "unknown";
}

std::string_view to_string(KimiTensorEncoding encoding) noexcept {
    switch (encoding) {
        case KimiTensorEncoding::PlainBf16: return "bf16";
        case KimiTensorEncoding::PlainF32: return "f32";
        case KimiTensorEncoding::Mxfp4Group32: return "mxfp4-group32";
    }
    return "unknown";
}

}  // namespace strata
