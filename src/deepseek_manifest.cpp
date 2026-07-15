#include "strata/deepseek_manifest.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <filesystem>
#include <optional>
#include <string_view>
#include <sys/stat.h>
#include <unordered_map>
#include <unordered_set>

namespace strata {

namespace {

constexpr std::array<std::uint64_t,
                     static_cast<std::size_t>(Dsv4TensorRole::Count)>
    kExpectedRoleCounts{
        1U,      // embedding
        1U,      // output head
        95U,     // model, block, compressor, and DSpark norms
        598U,    // hybrid-attention parameters
        164U,    // compressed-KV parameters
        147U,    // learned sparse-indexer parameters
        92U,     // gate weights plus hash tables or selection biases
        276U,    // FP8 shared experts
        70'656U, // FP4 routed experts
        282U,    // mHC block and head state
        5U,      // DSpark projection, Markov, and confidence state
    };

struct ClassifiedName {
    Dsv4TensorRole role{Dsv4TensorRole::Norm};
    Dsv4TensorComponent component{Dsv4TensorComponent::State};
    Dsv4TensorEncoding encoding{Dsv4TensorEncoding::Plain};
    std::int32_t layer{-1};
    std::int32_t expert{-1};
    bool dspark{};
    std::string quantized_base;
};

struct PairState {
    std::uint8_t mask{};
    std::string shard;
    Dsv4TensorEncoding encoding{Dsv4TensorEncoding::Plain};
};

[[nodiscard]] bool starts_with(std::string_view text,
                               std::string_view prefix) noexcept {
    return text.size() >= prefix.size() && text.substr(0U, prefix.size()) == prefix;
}

[[nodiscard]] bool ends_with(std::string_view text,
                             std::string_view suffix) noexcept {
    return text.size() >= suffix.size() &&
           text.substr(text.size() - suffix.size()) == suffix;
}

[[nodiscard]] std::optional<std::int32_t> parse_number(std::string_view text,
                                                       std::size_t& cursor) {
    const auto begin = cursor;
    while (cursor < text.size() && text[cursor] >= '0' && text[cursor] <= '9') {
        ++cursor;
    }
    if (begin == cursor) return std::nullopt;
    std::int32_t value = 0;
    const auto parsed = std::from_chars(text.data() + begin, text.data() + cursor, value);
    if (parsed.ec != std::errc{}) return std::nullopt;
    return value;
}

[[nodiscard]] std::optional<ClassifiedName> classify(
    std::string_view name, const std::unordered_set<std::string_view>& indexed_names,
    std::string& error) {
    ClassifiedName result;
    if (name == "embed.weight") {
        result.role = Dsv4TensorRole::Embedding;
        result.component = Dsv4TensorComponent::Weight;
        return result;
    }
    if (name == "head.weight") {
        result.role = Dsv4TensorRole::OutputHead;
        result.component = Dsv4TensorComponent::Weight;
        return result;
    }
    if (name == "norm.weight") {
        result.role = Dsv4TensorRole::Norm;
        result.component = Dsv4TensorComponent::Weight;
        return result;
    }
    if (starts_with(name, "hc_head_")) {
        result.role = Dsv4TensorRole::Mhc;
        return result;
    }

    std::string_view tail;
    if (starts_with(name, "layers.")) {
        std::size_t cursor = std::string_view("layers.").size();
        const auto layer = parse_number(name, cursor);
        if (!layer || *layer < 0 || *layer >= 43 || cursor >= name.size() ||
            name[cursor] != '.') {
            error = "invalid DeepSeek layer tensor name " + std::string(name);
            return std::nullopt;
        }
        result.layer = *layer;
        tail = name.substr(cursor + 1U);
    } else if (starts_with(name, "mtp.")) {
        std::size_t cursor = std::string_view("mtp.").size();
        const auto layer = parse_number(name, cursor);
        if (!layer || *layer < 0 || *layer >= 3 || cursor >= name.size() ||
            name[cursor] != '.') {
            error = "invalid DSpark layer tensor name " + std::string(name);
            return std::nullopt;
        }
        result.layer = *layer;
        result.dspark = true;
        tail = name.substr(cursor + 1U);
    } else {
        error = "unknown DeepSeek top-level tensor " + std::string(name);
        return std::nullopt;
    }

    if (ends_with(name, ".scale")) {
        result.component = Dsv4TensorComponent::Scale;
    } else if (ends_with(name, ".weight")) {
        result.component = Dsv4TensorComponent::Weight;
    } else if (ends_with(name, ".bias")) {
        result.component = Dsv4TensorComponent::Bias;
    }

    if (tail == "attn_norm.weight" || tail == "ffn_norm.weight" ||
        tail == "main_norm.weight" || tail == "norm.weight") {
        result.role = Dsv4TensorRole::Norm;
    } else if (starts_with(tail, "attn.indexer.")) {
        result.role = Dsv4TensorRole::AttentionIndexer;
    } else if (starts_with(tail, "attn.compressor.")) {
        result.role = Dsv4TensorRole::AttentionCompressor;
    } else if (starts_with(tail, "attn.")) {
        result.role = Dsv4TensorRole::Attention;
    } else if (starts_with(tail, "ffn.experts.")) {
        std::size_t cursor = std::string_view("ffn.experts.").size();
        const auto expert = parse_number(tail, cursor);
        if (!expert || *expert < 0 || *expert >= 256 || cursor >= tail.size() ||
            tail[cursor] != '.') {
            error = "invalid DeepSeek routed expert name " + std::string(name);
            return std::nullopt;
        }
        result.expert = *expert;
        result.role = Dsv4TensorRole::RoutedExpert;
    } else if (starts_with(tail, "ffn.shared_experts.")) {
        result.role = Dsv4TensorRole::SharedExpert;
    } else if (starts_with(tail, "ffn.gate.")) {
        result.role = Dsv4TensorRole::Router;
    } else if (starts_with(tail, "hc_")) {
        result.role = Dsv4TensorRole::Mhc;
    } else if (result.dspark &&
               (starts_with(tail, "main_proj.") ||
                starts_with(tail, "markov_head.") ||
                starts_with(tail, "confidence_head."))) {
        result.role = Dsv4TensorRole::DsparkState;
    } else {
        error = "unknown DeepSeek tensor role " + std::string(name);
        return std::nullopt;
    }

    std::string base;
    if (result.component == Dsv4TensorComponent::Scale) {
        base.assign(name.substr(0U, name.size() - std::string_view(".scale").size()));
    } else if (result.component == Dsv4TensorComponent::Weight) {
        base.assign(name.substr(0U, name.size() - std::string_view(".weight").size()));
    }
    const bool has_quantized_pair = !base.empty() &&
        indexed_names.contains(result.component == Dsv4TensorComponent::Scale
                                   ? std::string_view(base + ".weight")
                                   : std::string_view(base + ".scale"));
    if (has_quantized_pair) {
        result.encoding = result.role == Dsv4TensorRole::RoutedExpert
                              ? Dsv4TensorEncoding::Fp4E2m1Group32
                              : Dsv4TensorEncoding::Fp8E4m3Block128;
        result.quantized_base = std::move(base);
    } else if (result.component == Dsv4TensorComponent::Scale) {
        error = "unpaired DeepSeek quantization scale " + std::string(name);
        return std::nullopt;
    }
    return result;
}

[[nodiscard]] bool parse_shard_name(std::string_view name,
                                    std::uint32_t& ordinal) {
    constexpr std::string_view prefix = "model-";
    constexpr std::string_view divider = "-of-";
    constexpr std::string_view suffix = ".safetensors";
    if (!starts_with(name, prefix)) return false;
    std::size_t cursor = prefix.size();
    const auto number = parse_number(name, cursor);
    if (!number || cursor - prefix.size() != 5U ||
        name.substr(cursor, divider.size()) != divider) {
        return false;
    }
    cursor += divider.size();
    const auto total = parse_number(name, cursor);
    if (!total || cursor < divider.size() ||
        name.substr(cursor) != suffix || *total != 48 || *number < 1 || *number > 48) {
        return false;
    }
    ordinal = static_cast<std::uint32_t>(*number);
    return true;
}

void append_error(Dsv4ManifestResult& result, const Dsv4CheckpointOptions& options,
                  std::string error) {
    if (result.errors.size() < options.maximum_errors) {
        result.errors.push_back(std::move(error));
    }
}

}  // namespace

std::string_view to_string(Dsv4TensorRole role) noexcept {
    switch (role) {
        case Dsv4TensorRole::Embedding: return "embedding";
        case Dsv4TensorRole::OutputHead: return "output_head";
        case Dsv4TensorRole::Norm: return "norm";
        case Dsv4TensorRole::Attention: return "attention";
        case Dsv4TensorRole::AttentionCompressor: return "attention_compressor";
        case Dsv4TensorRole::AttentionIndexer: return "attention_indexer";
        case Dsv4TensorRole::Router: return "router";
        case Dsv4TensorRole::SharedExpert: return "shared_expert";
        case Dsv4TensorRole::RoutedExpert: return "routed_expert";
        case Dsv4TensorRole::Mhc: return "mhc";
        case Dsv4TensorRole::DsparkState: return "dspark_state";
        case Dsv4TensorRole::Count: return "count";
    }
    return "unknown";
}

std::string_view to_string(Dsv4TensorEncoding encoding) noexcept {
    switch (encoding) {
        case Dsv4TensorEncoding::Plain: return "plain";
        case Dsv4TensorEncoding::Fp8E4m3Block128: return "fp8_e4m3_block128";
        case Dsv4TensorEncoding::Fp4E2m1Group32: return "fp4_e2m1_group32";
    }
    return "unknown";
}

Dsv4ManifestResult build_deepseek_v4_flash_dspark_index_manifest(
    SafetensorsIndex index) {
    Dsv4ManifestResult result;
    const auto expected = deepseek_v4_flash_dspark_spec();
    result.manifest.indexed_tensor_bytes = index.total_size;
    result.manifest.shards = std::move(index.shards);
    if (index.total_size != expected.source.indexed_tensor_bytes) {
        result.errors.emplace_back("unexpected DeepSeek indexed tensor byte total");
    }
    if (index.entries.size() != expected.source.tensor_count) {
        result.errors.emplace_back("unexpected DeepSeek indexed tensor count");
    }

    std::array<bool, 48U> shards_seen{};
    for (const auto& shard : result.manifest.shards) {
        std::uint32_t ordinal = 0U;
        if (!parse_shard_name(shard, ordinal)) {
            result.errors.emplace_back("unexpected DeepSeek shard name " + shard);
        } else if (shards_seen[ordinal - 1U]) {
            result.errors.emplace_back("duplicate DeepSeek shard " + shard);
        } else {
            shards_seen[ordinal - 1U] = true;
        }
    }
    if (!std::all_of(shards_seen.begin(), shards_seen.end(), [](bool seen) { return seen; })) {
        result.errors.emplace_back("Safetensors index does not cover all 48 DeepSeek shards");
    }

    std::unordered_set<std::string_view> indexed_names;
    indexed_names.reserve(index.entries.size());
    for (const auto& entry : index.entries) indexed_names.insert(entry.name);

    std::unordered_map<std::string, PairState> pairs;
    pairs.reserve(36'000U);
    result.manifest.tensors.reserve(index.entries.size());
    for (auto& entry : index.entries) {
        std::string error;
        const auto classified = classify(entry.name, indexed_names, error);
        if (!classified) {
            result.errors.push_back(std::move(error));
            continue;
        }
        ++result.manifest.role_counts[static_cast<std::size_t>(classified->role)];
        if (!classified->quantized_base.empty()) {
            auto& pair = pairs[classified->quantized_base];
            const std::uint8_t bit = classified->component == Dsv4TensorComponent::Weight
                                         ? 1U
                                         : 2U;
            if ((pair.mask & bit) != 0U) {
                result.errors.emplace_back("duplicate DeepSeek quantized component for " +
                                           classified->quantized_base);
            }
            if (pair.mask == 0U) {
                pair.shard = entry.shard;
                pair.encoding = classified->encoding;
            } else if (pair.shard != entry.shard || pair.encoding != classified->encoding) {
                result.errors.emplace_back(
                    "DeepSeek quantized pair crosses a shard or encoding for " +
                    classified->quantized_base);
            }
            pair.mask = static_cast<std::uint8_t>(pair.mask | bit);
        }
        result.manifest.tensors.push_back({
            std::move(entry.name), std::move(entry.shard), classified->role,
            classified->component, classified->encoding, classified->layer,
            classified->expert, classified->dspark, SafetensorsDtype::Other, {}, 0U, 0U});
    }

    for (const auto& [base, pair] : pairs) {
        if (pair.mask != 3U) {
            result.errors.emplace_back("incomplete DeepSeek quantized pair for " + base);
            continue;
        }
        ++result.manifest.quantized_modules;
        if (pair.encoding == Dsv4TensorEncoding::Fp4E2m1Group32) {
            ++result.manifest.fp4_modules;
        } else if (pair.encoding == Dsv4TensorEncoding::Fp8E4m3Block128) {
            ++result.manifest.fp8_modules;
        } else {
            result.errors.emplace_back("quantized DeepSeek pair has plain encoding for " + base);
        }
    }
    if (result.manifest.quantized_modules != 35'718U ||
        result.manifest.fp4_modules != 35'328U || result.manifest.fp8_modules != 390U) {
        result.errors.emplace_back("unexpected DeepSeek native quantized module counts");
    }
    for (std::size_t role = 0U; role < kExpectedRoleCounts.size(); ++role) {
        if (result.manifest.role_counts[role] != kExpectedRoleCounts[role]) {
            result.errors.emplace_back(
                "unexpected DeepSeek " +
                std::string(to_string(static_cast<Dsv4TensorRole>(role))) +
                " tensor count");
        }
    }
    std::sort(result.manifest.tensors.begin(), result.manifest.tensors.end(),
              [](const auto& left, const auto& right) { return left.name < right.name; });
    return result;
}

Dsv4ManifestResult validate_deepseek_v4_flash_dspark_checkpoint(
    const std::string& model_directory, Dsv4IndexManifest manifest,
    const Dsv4CheckpointOptions& options) {
    Dsv4ManifestResult result;
    result.manifest = std::move(manifest);
    const auto expected = deepseek_v4_flash_dspark_spec();
    std::unordered_map<std::string, std::size_t> tensors;
    tensors.reserve(result.manifest.tensors.size());
    for (std::size_t index = 0U; index < result.manifest.tensors.size(); ++index) {
        tensors.emplace(result.manifest.tensors[index].name, index);
    }
    std::vector<bool> seen(result.manifest.tensors.size(), false);

    for (const auto& shard_name : result.manifest.shards) {
        const auto path = (std::filesystem::path(model_directory) / shard_name).string();
        struct stat status {};
        if (stat(path.c_str(), &status) != 0) {
            if (options.require_all_shards) {
                append_error(result, options, "missing DeepSeek shard " + shard_name);
            }
            continue;
        }
        if (options.require_read_only &&
            (status.st_mode & static_cast<mode_t>(S_IWUSR | S_IWGRP | S_IWOTH)) != 0) {
            append_error(result, options, "source DeepSeek shard is writable: " + shard_name);
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
            const auto found = tensors.find(source.name);
            if (found == tensors.end()) {
                append_error(result, options,
                             "unindexed DeepSeek tensor " + source.name + " in " + shard_name);
                continue;
            }
            auto& target = result.manifest.tensors[found->second];
            if (target.shard != shard_name) {
                append_error(result, options,
                             "DeepSeek tensor " + source.name + " is in the wrong shard");
                continue;
            }
            if (seen[found->second]) {
                append_error(result, options,
                             "DeepSeek tensor occurs in multiple shard headers: " +
                                 source.name);
                continue;
            }
            seen[found->second] = true;
            ++result.manifest.resolved_tensors;
            target.source_dtype = source.dtype;
            target.source_shape = source.shape;
            target.source_offset = source.absolute_begin;
            target.source_bytes = source.bytes();

            SafetensorsDtype wanted = SafetensorsDtype::F32;
            if (target.encoding == Dsv4TensorEncoding::Fp4E2m1Group32) {
                wanted = target.component == Dsv4TensorComponent::Scale
                             ? SafetensorsDtype::F8E8M0
                             : SafetensorsDtype::I8;
            } else if (target.encoding == Dsv4TensorEncoding::Fp8E4m3Block128) {
                wanted = target.component == Dsv4TensorComponent::Scale
                             ? SafetensorsDtype::F8E8M0
                             : SafetensorsDtype::F8E4M3;
            } else if (target.component == Dsv4TensorComponent::Weight) {
                wanted = ends_with(target.name, ".tid2eid")
                             ? SafetensorsDtype::I64
                             : SafetensorsDtype::Bf16;
            } else if (ends_with(target.name, ".tid2eid")) {
                wanted = SafetensorsDtype::I64;
            }
            if (source.dtype != wanted) {
                append_error(result, options,
                             "DeepSeek tensor " + source.name + " has dtype " +
                                 std::string(to_string(source.dtype)) + ", expected " +
                                 std::string(to_string(wanted)));
            }
        }
    }

    if (options.validate_quantized_layouts) {
        for (const auto& weight : result.manifest.tensors) {
            if (weight.component != Dsv4TensorComponent::Weight ||
                weight.encoding == Dsv4TensorEncoding::Plain ||
                weight.source_dtype == SafetensorsDtype::Other) {
                continue;
            }
            const auto base = weight.name.substr(
                0U, weight.name.size() - std::string_view(".weight").size());
            const auto scale_found = tensors.find(base + ".scale");
            if (scale_found == tensors.end()) {
                append_error(result, options,
                             "missing DeepSeek quantization scale for " + base);
                continue;
            }
            const auto& scale = result.manifest.tensors[scale_found->second];
            if (weight.source_shape.size() != 2U || scale.source_shape.size() != 2U) {
                append_error(result, options,
                             "DeepSeek quantized tensors are not rank two for " + base);
                continue;
            }
            bool layout_ok = false;
            if (weight.encoding == Dsv4TensorEncoding::Fp4E2m1Group32) {
                const auto logical_columns = weight.source_shape[1] * 2U;
                layout_ok = logical_columns % 32U == 0U &&
                    scale.source_shape == std::vector<std::uint64_t>{
                        weight.source_shape[0], logical_columns / 32U};
            } else {
                layout_ok = scale.source_shape == std::vector<std::uint64_t>{
                    (weight.source_shape[0] + 127U) / 128U,
                    (weight.source_shape[1] + 127U) / 128U};
            }
            if (!layout_ok) {
                append_error(result, options,
                             "DeepSeek quantized layout mismatch for " + base);
                continue;
            }
            ++result.manifest.validated_layouts;
        }
    }

    if (options.require_all_shards) {
        if (result.manifest.scanned_shards != expected.source.main_shards) {
            append_error(result, options,
                         "checkpoint does not contain all expected DeepSeek shard files");
        }
        if (result.manifest.resolved_tensors != expected.source.tensor_count) {
            append_error(result, options,
                         "DeepSeek checkpoint headers do not resolve every indexed tensor");
        }
        if (result.manifest.shard_file_bytes != expected.source.shard_file_bytes) {
            append_error(result, options,
                         "DeepSeek checkpoint shard byte total is unexpected");
        }
        if (result.manifest.tensor_payload_bytes != expected.source.indexed_tensor_bytes) {
            append_error(result, options,
                         "DeepSeek checkpoint tensor payload total is unexpected");
        }
        if (options.validate_quantized_layouts &&
            result.manifest.validated_layouts != result.manifest.quantized_modules) {
            append_error(result, options,
                         "not every DeepSeek native quantized module passed layout validation");
        }
    }
    return result;
}

}  // namespace strata
