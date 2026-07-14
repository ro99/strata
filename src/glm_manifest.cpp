#include "strata/glm_manifest.hpp"

#include "strata/compressed_tensors.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <sys/stat.h>

namespace strata {

namespace {

constexpr std::array<std::uint64_t, static_cast<std::size_t>(GlmTensorRole::Count)>
    kExpectedRoleCounts{
        1U,       // embedding
        1U,       // output head
        317U,     // model and per-layer norms
        1175U,    // attention linears and their packed metadata
        110U,     // DSA indexers
        21U,      // three dense MLPs
        152U,     // router weights and correction biases
        684U,     // shared experts
        175104U,  // routed experts
        4U,       // MTP state projections and norms
    };

struct ClassifiedName {
    GlmTensorRole role{GlmTensorRole::Norm};
    GlmTensorComponent component{GlmTensorComponent::Weight};
    GlmTensorEncoding encoding{GlmTensorEncoding::Plain};
    std::int32_t layer{-1};
    std::int32_t expert{-1};
    bool mtp{};
    std::string quantized_base;
};

struct TripletState {
    std::uint8_t mask{};
    std::string shard;
    GlmTensorEncoding encoding{GlmTensorEncoding::Plain};
};

[[nodiscard]] std::string strip_component_suffix(std::string_view name,
                                                 GlmTensorComponent component) {
    std::string_view suffix;
    switch (component) {
        case GlmTensorComponent::PackedWeight: suffix = ".weight_packed"; break;
        case GlmTensorComponent::Scale: suffix = ".weight_scale"; break;
        case GlmTensorComponent::LogicalShape: suffix = ".weight_shape"; break;
        case GlmTensorComponent::Weight:
        case GlmTensorComponent::Bias: return std::string(name);
    }
    return std::string(name.substr(0, name.size() - suffix.size()));
}

[[nodiscard]] QuantizedWeightSpec quantization_for(GlmTensorEncoding encoding) {
    switch (encoding) {
        case GlmTensorEncoding::Int4Group128:
            return {4, QuantizationGranularity::Group, 128, true};
        case GlmTensorEncoding::Int8Group128:
            return {8, QuantizationGranularity::Group, 128, true};
        case GlmTensorEncoding::Int8Channel:
            return {8, QuantizationGranularity::Channel, 0, true};
        case GlmTensorEncoding::Plain: return {};
    }
    return {};
}

void append_error(GlmManifestResult& result, const GlmCheckpointOptions& options,
                  std::string error) {
    if (result.errors.size() < options.maximum_errors) result.errors.push_back(std::move(error));
}

[[nodiscard]] bool starts_with(std::string_view text, std::string_view prefix) noexcept {
    return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

[[nodiscard]] bool ends_with(std::string_view text, std::string_view suffix) noexcept {
    return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
}

[[nodiscard]] std::optional<std::int32_t> parse_segment_number(std::string_view text,
                                                               std::size_t& cursor) {
    const auto begin = cursor;
    while (cursor < text.size() && text[cursor] >= '0' && text[cursor] <= '9') ++cursor;
    if (begin == cursor) return std::nullopt;
    std::int32_t value = 0;
    const auto conversion = std::from_chars(text.data() + begin, text.data() + cursor, value);
    if (conversion.ec != std::errc{}) return std::nullopt;
    return value;
}

[[nodiscard]] GlmTensorEncoding encoding_for(GlmTensorRole role, std::int32_t layer) {
    if (layer == 78 && (role == GlmTensorRole::Attention ||
                        role == GlmTensorRole::SharedExpert ||
                        role == GlmTensorRole::RoutedExpert)) {
        return GlmTensorEncoding::Int8Channel;
    }
    if (role == GlmTensorRole::RoutedExpert) return GlmTensorEncoding::Int4Group128;
    return GlmTensorEncoding::Int8Group128;
}

[[nodiscard]] std::optional<ClassifiedName> classify(std::string_view name,
                                                     std::string& error) {
    ClassifiedName result;
    if (name == "model.embed_tokens.weight") {
        result.role = GlmTensorRole::Embedding;
        return result;
    }
    if (name == "lm_head.weight") {
        result.role = GlmTensorRole::OutputHead;
        return result;
    }
    if (name == "model.norm.weight") {
        result.role = GlmTensorRole::Norm;
        return result;
    }

    constexpr std::string_view layer_prefix = "model.layers.";
    if (!starts_with(name, layer_prefix)) {
        error = "unknown top-level tensor " + std::string(name);
        return std::nullopt;
    }
    std::size_t cursor = layer_prefix.size();
    const auto layer = parse_segment_number(name, cursor);
    if (!layer || cursor >= name.size() || name[cursor] != '.') {
        error = "invalid layer tensor name " + std::string(name);
        return std::nullopt;
    }
    if (*layer < 0 || *layer > 78) {
        error = "tensor layer is outside [0, 78]: " + std::string(name);
        return std::nullopt;
    }
    result.layer = *layer;
    result.mtp = *layer == 78;
    const auto tail = name.substr(++cursor);

    constexpr std::string_view packed_suffix = ".weight_packed";
    constexpr std::string_view scale_suffix = ".weight_scale";
    constexpr std::string_view shape_suffix = ".weight_shape";
    std::string_view operation = tail;
    if (ends_with(tail, packed_suffix)) {
        result.component = GlmTensorComponent::PackedWeight;
        operation = tail.substr(0, tail.size() - packed_suffix.size());
    } else if (ends_with(tail, scale_suffix)) {
        result.component = GlmTensorComponent::Scale;
        operation = tail.substr(0, tail.size() - scale_suffix.size());
    } else if (ends_with(tail, shape_suffix)) {
        result.component = GlmTensorComponent::LogicalShape;
        operation = tail.substr(0, tail.size() - shape_suffix.size());
    } else if (ends_with(tail, ".bias") || ends_with(tail, "_bias")) {
        result.component = GlmTensorComponent::Bias;
    } else if (!ends_with(tail, ".weight")) {
        error = "unknown tensor component " + std::string(name);
        return std::nullopt;
    }

    if (tail == "input_layernorm.weight" || tail == "post_attention_layernorm.weight" ||
        tail == "self_attn.q_a_layernorm.weight" ||
        tail == "self_attn.kv_a_layernorm.weight") {
        result.role = GlmTensorRole::Norm;
    } else if (starts_with(tail, "self_attn.indexer.")) {
        result.role = GlmTensorRole::AttentionIndexer;
    } else if (starts_with(tail, "self_attn.")) {
        result.role = GlmTensorRole::Attention;
    } else if (tail == "mlp.gate.weight" ||
               tail == "mlp.gate.e_score_correction_bias") {
        result.role = GlmTensorRole::Router;
    } else if (starts_with(tail, "mlp.shared_experts.")) {
        result.role = GlmTensorRole::SharedExpert;
    } else if (starts_with(tail, "mlp.experts.")) {
        std::size_t expert_cursor = std::string_view("mlp.experts.").size();
        const auto expert = parse_segment_number(tail, expert_cursor);
        if (!expert || *expert < 0 || *expert > 255 || expert_cursor >= tail.size() ||
            tail[expert_cursor] != '.') {
            error = "invalid expert tensor name " + std::string(name);
            return std::nullopt;
        }
        result.expert = *expert;
        result.role = GlmTensorRole::RoutedExpert;
    } else if (starts_with(tail, "mlp.gate_proj.") ||
               starts_with(tail, "mlp.up_proj.") ||
               starts_with(tail, "mlp.down_proj.")) {
        result.role = GlmTensorRole::DenseMlp;
    } else if (tail == "eh_proj.weight" || tail == "enorm.weight" ||
               tail == "hnorm.weight" || tail == "shared_head.norm.weight") {
        if (!result.mtp) {
            error = "MTP state tensor appears outside layer 78: " + std::string(name);
            return std::nullopt;
        }
        result.role = GlmTensorRole::MtpState;
    } else {
        error = "unknown GLM tensor role " + std::string(name);
        return std::nullopt;
    }

    const bool quantized_component = result.component == GlmTensorComponent::PackedWeight ||
                                     result.component == GlmTensorComponent::Scale ||
                                     result.component == GlmTensorComponent::LogicalShape;
    if (quantized_component) {
        if (result.role != GlmTensorRole::Attention &&
            result.role != GlmTensorRole::DenseMlp &&
            result.role != GlmTensorRole::SharedExpert &&
            result.role != GlmTensorRole::RoutedExpert) {
            error = "forbidden quantized tensor role " + std::string(name);
            return std::nullopt;
        }
        result.encoding = encoding_for(result.role, result.layer);
        result.quantized_base = std::string(name.substr(0, name.size() -
            (result.component == GlmTensorComponent::PackedWeight ? packed_suffix.size() :
             result.component == GlmTensorComponent::Scale ? scale_suffix.size() :
                                                             shape_suffix.size())));
    }
    static_cast<void>(operation);
    return result;
}

[[nodiscard]] bool parse_shard_name(std::string_view name, std::string_view prefix,
                                    std::uint32_t expected_total, std::uint32_t& ordinal) {
    const auto expected_size = prefix.size() + 5U + std::string_view("-of-").size() + 5U +
                               std::string_view(".safetensors").size();
    if (name.size() != expected_size || !starts_with(name, prefix)) return false;
    std::size_t cursor = prefix.size();
    const auto number = parse_segment_number(name, cursor);
    if (!number || cursor + 4U >= name.size() || name.substr(cursor, 4U) != "-of-") return false;
    cursor += 4U;
    const auto total = parse_segment_number(name, cursor);
    if (!total || *total != static_cast<std::int32_t>(expected_total) ||
        name.substr(cursor) != ".safetensors" || *number < 1 ||
        *number > static_cast<std::int32_t>(expected_total)) {
        return false;
    }
    ordinal = static_cast<std::uint32_t>(*number);
    return true;
}

}  // namespace

std::string_view to_string(GlmTensorRole role) noexcept {
    switch (role) {
        case GlmTensorRole::Embedding: return "embedding";
        case GlmTensorRole::OutputHead: return "output_head";
        case GlmTensorRole::Norm: return "norm";
        case GlmTensorRole::Attention: return "attention";
        case GlmTensorRole::AttentionIndexer: return "attention_indexer";
        case GlmTensorRole::DenseMlp: return "dense_mlp";
        case GlmTensorRole::Router: return "router";
        case GlmTensorRole::SharedExpert: return "shared_expert";
        case GlmTensorRole::RoutedExpert: return "routed_expert";
        case GlmTensorRole::MtpState: return "mtp_state";
        case GlmTensorRole::Count: return "count";
    }
    return "unknown";
}

std::string_view to_string(GlmTensorEncoding encoding) noexcept {
    switch (encoding) {
        case GlmTensorEncoding::Plain: return "plain";
        case GlmTensorEncoding::Int4Group128: return "int4_group128";
        case GlmTensorEncoding::Int8Group128: return "int8_group128";
        case GlmTensorEncoding::Int8Channel: return "int8_channel";
    }
    return "unknown";
}

GlmManifestResult build_quanttrio_glm52_index_manifest(SafetensorsIndex index) {
    GlmManifestResult result;
    const auto expected = quanttrio_glm52_int4_int8_mix_spec();
    result.manifest.indexed_tensor_bytes = index.total_size;
    result.manifest.shards = std::move(index.shards);

    if (index.total_size != expected.source.indexed_tensor_bytes) {
        result.errors.emplace_back("unexpected indexed tensor byte total");
    }
    if (index.entries.size() != expected.source.tensor_count) {
        result.errors.emplace_back("unexpected indexed tensor count");
    }

    std::array<bool, 124U> main_seen{};
    std::array<bool, 4U> mtp_seen{};
    for (const auto& shard : result.manifest.shards) {
        std::uint32_t ordinal = 0;
        if (parse_shard_name(shard, "model-", 124U, ordinal)) {
            if (main_seen[ordinal - 1U]) result.errors.emplace_back("duplicate main shard " + shard);
            main_seen[ordinal - 1U] = true;
        } else if (parse_shard_name(shard, "mtp-", 4U, ordinal)) {
            if (mtp_seen[ordinal - 1U]) result.errors.emplace_back("duplicate MTP shard " + shard);
            mtp_seen[ordinal - 1U] = true;
        } else {
            result.errors.emplace_back("unexpected shard name " + shard);
        }
    }
    if (!std::all_of(main_seen.begin(), main_seen.end(), [](bool seen) { return seen; }) ||
        !std::all_of(mtp_seen.begin(), mtp_seen.end(), [](bool seen) { return seen; })) {
        result.errors.emplace_back("Safetensors index does not cover all 124 main and four MTP shards");
    }

    result.manifest.tensors.reserve(index.entries.size());
    std::unordered_map<std::string, TripletState> triplets;
    triplets.reserve(60000U);
    for (auto& entry : index.entries) {
        std::string error;
        const auto classified = classify(entry.name, error);
        if (!classified) {
            result.errors.push_back(std::move(error));
            continue;
        }
        const auto role_index = static_cast<std::size_t>(classified->role);
        ++result.manifest.role_counts[role_index];
        if (!classified->quantized_base.empty()) {
            auto& triplet = triplets[classified->quantized_base];
            const std::uint8_t bit = classified->component == GlmTensorComponent::PackedWeight ? 1U :
                                     classified->component == GlmTensorComponent::Scale ? 2U : 4U;
            if ((triplet.mask & bit) != 0U) {
                result.errors.emplace_back("duplicate compressed-tensors component for " +
                                           classified->quantized_base);
            }
            if (triplet.mask == 0U) {
                triplet.shard = entry.shard;
                triplet.encoding = classified->encoding;
            } else if (triplet.shard != entry.shard || triplet.encoding != classified->encoding) {
                result.errors.emplace_back("compressed-tensors triplet crosses a shard or encoding for " +
                                           classified->quantized_base);
            }
            triplet.mask = static_cast<std::uint8_t>(triplet.mask | bit);
        }
        result.manifest.tensors.push_back({
            std::move(entry.name), std::move(entry.shard), classified->role,
            classified->component, classified->encoding, classified->layer,
            classified->expert, classified->mtp, SafetensorsDtype::Other, {}, 0U, 0U});
    }

    for (const auto& [base, triplet] : triplets) {
        if (triplet.mask != 7U) {
            result.errors.emplace_back("incomplete compressed-tensors triplet for " + base);
            continue;
        }
        ++result.manifest.quantized_modules;
        switch (triplet.encoding) {
            case GlmTensorEncoding::Int4Group128: ++result.manifest.int4_modules; break;
            case GlmTensorEncoding::Int8Group128: ++result.manifest.int8_group_modules; break;
            case GlmTensorEncoding::Int8Channel: ++result.manifest.int8_channel_modules; break;
            case GlmTensorEncoding::Plain:
                result.errors.emplace_back("quantized triplet has a plain encoding for " + base);
                break;
        }
    }

    if (result.manifest.quantized_modules != 58992U ||
        result.manifest.int4_modules != 57600U ||
        result.manifest.int8_group_modules != 616U ||
        result.manifest.int8_channel_modules != 776U) {
        result.errors.emplace_back("unexpected compressed-tensors module counts");
    }
    for (std::size_t role = 0; role < kExpectedRoleCounts.size(); ++role) {
        if (result.manifest.role_counts[role] != kExpectedRoleCounts[role]) {
            result.errors.emplace_back("unexpected " +
                std::string(to_string(static_cast<GlmTensorRole>(role))) + " tensor count");
        }
    }
    std::sort(result.manifest.tensors.begin(), result.manifest.tensors.end(),
              [](const auto& left, const auto& right) { return left.name < right.name; });
    return result;
}

GlmManifestResult validate_quanttrio_glm52_checkpoint(
    const std::string& model_directory, GlmIndexManifest manifest,
    const GlmCheckpointOptions& options) {
    GlmManifestResult result;
    result.manifest = std::move(manifest);
    const auto expected = quanttrio_glm52_int4_int8_mix_spec();
    std::unordered_map<std::string, std::size_t> tensor_by_name;
    tensor_by_name.reserve(result.manifest.tensors.size());
    for (std::size_t index = 0; index < result.manifest.tensors.size(); ++index) {
        tensor_by_name.emplace(result.manifest.tensors[index].name, index);
    }
    std::vector<bool> seen(result.manifest.tensors.size(), false);

    for (const auto& shard_name : result.manifest.shards) {
        const auto path = (std::filesystem::path(model_directory) / shard_name).string();
        struct stat status {};
        if (stat(path.c_str(), &status) != 0) {
            if (options.require_all_shards) append_error(result, options, "missing shard " + shard_name);
            continue;
        }
        if (options.require_read_only &&
            (status.st_mode & static_cast<mode_t>(S_IWUSR | S_IWGRP | S_IWOTH)) != 0) {
            append_error(result, options, "source shard is writable: " + shard_name);
            continue;
        }
        auto shard = load_safetensors_shard(path);
        if (!shard.ok()) {
            for (auto& error : shard.errors) append_error(result, options, std::move(error));
            continue;
        }
        ++result.manifest.scanned_shards;
        result.manifest.shard_file_bytes += shard.value.file_size;
        for (const auto& source : shard.value.tensors) {
            result.manifest.tensor_payload_bytes += source.bytes();
            const auto found = tensor_by_name.find(source.name);
            if (found == tensor_by_name.end()) {
                append_error(result, options, "unindexed tensor " + source.name + " in " + shard_name);
                continue;
            }
            auto& target = result.manifest.tensors[found->second];
            if (target.shard != shard_name) {
                append_error(result, options, "tensor " + source.name + " is in the wrong shard");
                continue;
            }
            if (seen[found->second]) {
                append_error(result, options, "tensor occurs in multiple shard headers: " + source.name);
                continue;
            }
            seen[found->second] = true;
            ++result.manifest.resolved_tensors;
            target.source_dtype = source.dtype;
            target.source_shape = source.shape;
            target.source_offset = source.absolute_begin;
            target.source_bytes = source.bytes();

            const auto expected_dtype = target.component == GlmTensorComponent::PackedWeight
                                            ? SafetensorsDtype::I32
                                        : target.component == GlmTensorComponent::Scale
                                            ? SafetensorsDtype::Bf16
                                        : target.component == GlmTensorComponent::LogicalShape
                                            ? SafetensorsDtype::I64
                                            : source.dtype;
            if (source.dtype != expected_dtype) {
                append_error(result, options, "tensor " + source.name + " has unexpected dtype " +
                                                  std::string(to_string(source.dtype)));
            }
            if ((target.component == GlmTensorComponent::Weight ||
                 target.component == GlmTensorComponent::Bias) &&
                source.dtype != SafetensorsDtype::Bf16 &&
                source.dtype != SafetensorsDtype::F32) {
                append_error(result, options, "plain tensor " + source.name +
                                                  " must be BF16 or F32");
            }
            if (target.component == GlmTensorComponent::LogicalShape &&
                (source.shape.size() != 1U || source.shape[0] != 2U)) {
                append_error(result, options, "weight_shape header shape is not [2] for " +
                                                  source.name);
            }
        }
    }

    if (options.validate_logical_shapes) {
        for (auto& packed : result.manifest.tensors) {
            if (packed.component != GlmTensorComponent::PackedWeight ||
                packed.source_dtype == SafetensorsDtype::Other) {
                continue;
            }
            const auto base = strip_component_suffix(packed.name, packed.component);
            const auto scale_found = tensor_by_name.find(base + ".weight_scale");
            const auto shape_found = tensor_by_name.find(base + ".weight_shape");
            if (scale_found == tensor_by_name.end() || shape_found == tensor_by_name.end()) {
                append_error(result, options, "missing joined tensor metadata for " + base);
                continue;
            }
            const auto& scale = result.manifest.tensors[scale_found->second];
            const auto& shape = result.manifest.tensors[shape_found->second];
            if (scale.source_dtype == SafetensorsDtype::Other ||
                shape.source_dtype == SafetensorsDtype::Other) {
                append_error(result, options, "partially resolved triplet for " + base);
                continue;
            }
            SafetensorsTensor source_shape;
            source_shape.name = shape.name;
            source_shape.dtype = shape.source_dtype;
            source_shape.shape = shape.source_shape;
            source_shape.absolute_begin = shape.source_offset;
            source_shape.relative_begin = 0;
            source_shape.relative_end = shape.source_bytes;
            const auto path = (std::filesystem::path(model_directory) / shape.shard).string();
            const auto encoded = read_safetensors_tensor(path, source_shape, 16U);
            if (!encoded.ok()) {
                for (auto error : encoded.errors) append_error(result, options, std::move(error));
                continue;
            }
            std::array<std::uint64_t, 2> logical_shape{};
            const auto decoded = decode_compressed_logical_shape(encoded.value, logical_shape);
            if (!decoded.ok()) {
                for (auto error : decoded.errors) append_error(result, options, std::move(error));
                continue;
            }
            if (packed.source_shape.size() != 2U || scale.source_shape.size() != 2U) {
                append_error(result, options, "packed or scale tensor is not rank two for " + base);
                continue;
            }
            CompressedTensorLayout layout;
            layout.logical_rows = logical_shape[0];
            layout.logical_columns = logical_shape[1];
            layout.packed_rows = packed.source_shape[0];
            layout.packed_columns = packed.source_shape[1];
            layout.scale_rows = scale.source_shape[0];
            layout.scale_columns = scale.source_shape[1];
            layout.packed_dtype = packed.source_dtype;
            layout.scale_dtype = scale.source_dtype;
            layout.shape_dtype = shape.source_dtype;
            const auto validated = validate_compressed_tensor_layout(
                layout, quantization_for(packed.encoding));
            if (!validated.ok()) {
                for (const auto& error : validated.errors) {
                    append_error(result, options, base + ": " + error);
                }
                continue;
            }
            ++result.manifest.validated_layouts;
        }
    }

    if (options.require_all_shards) {
        if (result.manifest.scanned_shards != expected.source.main_shards +
                                                  expected.source.mtp_shards) {
            append_error(result, options, "checkpoint does not contain all expected shard files");
        }
        if (result.manifest.resolved_tensors != expected.source.tensor_count) {
            append_error(result, options, "checkpoint headers do not resolve every indexed tensor");
        }
        if (result.manifest.shard_file_bytes != expected.source.shard_file_bytes) {
            append_error(result, options, "checkpoint shard byte total is unexpected");
        }
        if (result.manifest.tensor_payload_bytes != expected.source.indexed_tensor_bytes) {
            append_error(result, options, "checkpoint tensor payload total is unexpected");
        }
        if (options.validate_logical_shapes &&
            result.manifest.validated_layouts != result.manifest.quantized_modules) {
            append_error(result, options, "not every quantized module passed layout validation");
        }
    }
    return result;
}

}  // namespace strata
