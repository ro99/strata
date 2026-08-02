#include "strata/kimi_k3_checkpoint.hpp"

#include "checkpoint_common.hpp"

#include "strata/model_adapter.hpp"

#include <array>
#include <cstdio>
#include <filesystem>
#include <limits>

namespace strata {
namespace {

constexpr std::uint64_t kMaximumIndexBytes = 64ULL << 20U;
constexpr std::uint64_t kMaximumConfigBytes = 4ULL << 20U;
// Gate, up, down. Fixed order so the dense expert index is pure arithmetic.
constexpr std::array<std::string_view, 3> kExpertModules{"w1", "w3", "w2"};

void append(std::vector<std::string>& destination,
            std::vector<std::string> source) {
    for (auto& error : source) destination.push_back(std::move(error));
}

[[nodiscard]] std::size_t expert_slot(std::uint32_t layer, std::uint32_t expert,
                                      std::size_t module, bool scale) noexcept {
    const auto& c = kKimiK3ExecutionContract;
    const auto sparse = layer - c.dense_prefix_layers;
    return (((static_cast<std::size_t>(sparse) * c.routed_experts + expert) *
             kExpertModules.size()) +
            module) *
               2U +
           (scale ? 1U : 0U);
}

}  // namespace

CompressedTensorLayout kimi_expert_layout(std::uint64_t rows,
                                          std::uint64_t columns) noexcept {
    CompressedTensorLayout layout;
    layout.codec = CompressedTensorCodec::Mxfp4E2m1;
    layout.packed_dtype = SafetensorsDtype::U8;
    layout.scale_dtype = SafetensorsDtype::U8;
    layout.logical_rows = rows;
    layout.logical_columns = columns;
    layout.packed_rows = rows;
    layout.packed_columns = columns / 2U;
    layout.scale_rows = rows;
    layout.scale_columns = columns / 32U;
    // `mxfp4-pack-quantized` ships no weight_shape tensor; the logical shape is
    // implied by the packed extent and the group size.
    layout.shape_elements = 0U;
    return layout;
}

QuantizedWeightSpec kimi_expert_quantization() noexcept {
    return {4U, QuantizationGranularity::Group, 32U, true};
}

KimiCheckpointOpenResult KimiCheckpointReader::open(std::string model_directory) {
    KimiCheckpointOpenResult result;
    const auto& c = kKimiK3ExecutionContract;
    const auto root = std::filesystem::path(model_directory);

    auto config_text =
        load_bounded_text_file((root / "config.json").string(), kMaximumConfigBytes);
    if (!config_text.ok()) {
        result.errors = std::move(config_text.errors);
        return result;
    }
    auto config = parse_kimi_k3_config(config_text.value);
    if (!config.ok()) {
        result.errors = std::move(config.errors);
        return result;
    }
    auto contract = validate_kimi_k3_config(config.value);
    if (!contract.ok()) {
        result.errors = std::move(contract.errors);
        return result;
    }

    auto text = load_bounded_text_file((root / "model.safetensors.index.json").string(),
                                       kMaximumIndexBytes);
    if (!text.ok()) {
        result.errors = std::move(text.errors);
        return result;
    }
    auto index = parse_safetensors_index(text.value);
    if (!index.ok()) {
        result.errors = std::move(index.errors);
        return result;
    }
    const auto spec = kimi_k3_mxfp4_spec();
    if (index.value.total_size != spec.source.indexed_tensor_bytes ||
        index.value.entries.size() != spec.source.tensor_count ||
        index.value.shards.size() != spec.source.main_shards) {
        result.errors.emplace_back(
            "Kimi-K3 checkpoint index extent is not the pinned MXFP4 release");
        return result;
    }
    for (const auto& shard : index.value.shards) {
        char expected[48]{};
        std::snprintf(expected, sizeof(expected),
                      "model-%05u-of-%06u.safetensors", 0U, 0U);
        bool matched = false;
        for (std::uint32_t ordinal = 1U; ordinal <= spec.source.main_shards;
             ++ordinal) {
            std::snprintf(expected, sizeof(expected),
                          "model-%05u-of-%06u.safetensors", ordinal,
                          spec.source.main_shards);
            if (shard == expected) {
                matched = true;
                break;
            }
        }
        if (!matched) {
            result.errors.push_back("unexpected Kimi-K3 shard name " + shard);
            return result;
        }
    }

    auto built = build_kimi_k3_index_manifest(std::move(index.value));
    if (!built.ok()) {
        result.errors = std::move(built.errors);
        return result;
    }
    auto validated = validate_kimi_k3_checkpoint(model_directory,
                                                 std::move(built.manifest));
    if (!validated.ok()) {
        result.errors = std::move(validated.errors);
        return result;
    }
    if (validated.manifest.shard_file_bytes != spec.source.shard_file_bytes) {
        result.errors.emplace_back(
            "Kimi-K3 shard extent does not match the pinned checkpoint");
        return result;
    }

    auto reader = std::unique_ptr<KimiCheckpointReader>(new KimiCheckpointReader());
    reader->model_directory_ = std::move(model_directory);
    reader->config_ = std::move(config.value);
    reader->manifest_ = std::move(validated.manifest);
    reader->by_name_.reserve(reader->manifest_.tensors.size());
    for (std::size_t index_of = 0U; index_of < reader->manifest_.tensors.size();
         ++index_of) {
        reader->by_name_.emplace(reader->manifest_.tensors[index_of].name, index_of);
    }

    const auto sparse_layers = c.layer_count - c.dense_prefix_layers;
    reader->expert_index_.assign(static_cast<std::size_t>(sparse_layers) *
                                     c.routed_experts * kExpertModules.size() * 2U,
                                 std::numeric_limits<std::size_t>::max());
    for (std::size_t index_of = 0U; index_of < reader->manifest_.tensors.size();
         ++index_of) {
        const auto& tensor = reader->manifest_.tensors[index_of];
        if (tensor.role != KimiTensorRole::RoutedExpert || tensor.layer < 0 ||
            tensor.expert < 0) {
            continue;
        }
        const auto layer = static_cast<std::uint32_t>(tensor.layer);
        const auto expert = static_cast<std::uint32_t>(tensor.expert);
        if (layer < c.dense_prefix_layers || layer >= c.layer_count ||
            expert >= c.routed_experts) {
            result.errors.push_back("Kimi-K3 routed expert out of range: " +
                                    tensor.name);
            return result;
        }
        std::size_t module = kExpertModules.size();
        for (std::size_t candidate = 0U; candidate < kExpertModules.size();
             ++candidate) {
            if (tensor.name.find("." + std::string(kExpertModules[candidate]) + ".") !=
                std::string::npos) {
                module = candidate;
                break;
            }
        }
        if (module == kExpertModules.size()) {
            result.errors.push_back("unrecognised Kimi-K3 expert module " +
                                    tensor.name);
            return result;
        }
        const auto scale = tensor.component == KimiTensorComponent::Scale;
        reader->expert_index_[expert_slot(layer, expert, module, scale)] = index_of;
    }
    for (const auto slot : reader->expert_index_) {
        if (slot == std::numeric_limits<std::size_t>::max()) {
            result.errors.emplace_back(
                "Kimi-K3 routed-expert index has a hole; the checkpoint does not "
                "carry every (layer, expert, module) triple");
            return result;
        }
    }

    auto opened = reader->shards_.open(reader->model_directory_,
                                       reader->manifest_.shards, "Kimi-K3");
    if (!opened.ok()) {
        result.errors = std::move(opened.errors);
        return result;
    }
    result.value = std::move(reader);
    return result;
}

const KimiManifestTensor* KimiCheckpointReader::find(
    std::string_view name) const noexcept {
    const auto found = by_name_.find(name);
    if (found == by_name_.end()) return nullptr;
    return &manifest_.tensors[found->second];
}

ParseResult<std::vector<std::byte>> KimiCheckpointReader::read(
    std::string_view name, std::uint64_t maximum_bytes) const {
    ParseResult<std::vector<std::byte>> result;
    const auto* tensor = find(name);
    if (tensor == nullptr) {
        result.errors.push_back("unknown Kimi-K3 tensor " + std::string(name));
        return result;
    }
    if (tensor->source_bytes > maximum_bytes) {
        result.errors.push_back("Kimi-K3 tensor " + std::string(name) +
                                " exceeds the caller's byte budget");
        return result;
    }
    result.value.resize(static_cast<std::size_t>(tensor->source_bytes));
    auto read_result = shards_.read(tensor->shard, tensor->source_offset,
                                    result.value, name);
    if (!read_result.ok()) {
        result.errors = std::move(read_result.errors);
        result.value.clear();
    }
    return result;
}

ValidationResult KimiCheckpointReader::read_into(
    std::string_view name, std::span<std::byte> destination) const {
    ValidationResult result;
    const auto* tensor = find(name);
    if (tensor == nullptr) {
        result.errors.push_back("unknown Kimi-K3 tensor " + std::string(name));
        return result;
    }
    if (destination.size() != tensor->source_bytes) {
        result.errors.push_back("Kimi-K3 tensor " + std::string(name) +
                                " does not fit the destination exactly");
        return result;
    }
    return shards_.read(tensor->shard, tensor->source_offset, destination, name);
}

ParseResult<std::vector<float>> KimiCheckpointReader::read_f32(
    std::string_view name, std::uint64_t maximum_elements) const {
    ParseResult<std::vector<float>> result;
    const auto* tensor = find(name);
    if (tensor == nullptr) {
        result.errors.push_back("unknown Kimi-K3 tensor " + std::string(name));
        return result;
    }
    if (tensor->source_dtype != SafetensorsDtype::Bf16 &&
        tensor->source_dtype != SafetensorsDtype::F32) {
        result.errors.push_back("Kimi-K3 tensor " + std::string(name) +
                                " is not a plain BF16 or F32 tensor");
        return result;
    }
    const auto width = safetensors_dtype_bytes(tensor->source_dtype);
    const auto elements = tensor->source_bytes / width;
    if (elements > maximum_elements) {
        result.errors.push_back("Kimi-K3 tensor " + std::string(name) +
                                " exceeds the caller's element budget");
        return result;
    }
    std::vector<std::byte> raw(static_cast<std::size_t>(tensor->source_bytes));
    auto read_result = shards_.read(tensor->shard, tensor->source_offset, raw, name);
    if (!read_result.ok()) {
        result.errors = std::move(read_result.errors);
        return result;
    }
    result.value.resize(static_cast<std::size_t>(elements));
    for (std::size_t index = 0U; index < result.value.size(); ++index) {
        result.value[index] =
            detail::decode_plain_scalar(raw.data() + index * width,
                                        tensor->source_dtype);
    }
    return result;
}

const KimiManifestTensor* KimiCheckpointReader::expert_tensor(
    std::uint32_t layer, std::uint32_t expert, std::string_view module,
    bool scale) const noexcept {
    const auto& c = kKimiK3ExecutionContract;
    if (layer < c.dense_prefix_layers || layer >= c.layer_count ||
        expert >= c.routed_experts) {
        return nullptr;
    }
    std::size_t ordinal = kExpertModules.size();
    for (std::size_t candidate = 0U; candidate < kExpertModules.size(); ++candidate) {
        if (kExpertModules[candidate] == module) {
            ordinal = candidate;
            break;
        }
    }
    if (ordinal == kExpertModules.size()) return nullptr;
    const auto slot = expert_slot(layer, expert, ordinal, scale);
    if (slot >= expert_index_.size()) return nullptr;
    return &manifest_.tensors[expert_index_[slot]];
}

bool KimiCheckpointReader::expert_modules(
    std::uint32_t layer, std::uint32_t expert,
    KimiExpertModules& modules) const noexcept {
    const auto fill = [&](std::string_view module, KimiExpertLocation& location) {
        const auto* packed = expert_tensor(layer, expert, module, false);
        const auto* scale = expert_tensor(layer, expert, module, true);
        if (packed == nullptr || scale == nullptr) return false;
        location.shard = packed->shard;
        location.packed_offset = packed->source_offset;
        location.packed_bytes = packed->source_bytes;
        location.scale_offset = scale->source_offset;
        location.scale_bytes = scale->source_bytes;
        return true;
    };
    return fill("w1", modules.gate) && fill("w3", modules.up) &&
           fill("w2", modules.down);
}

ValidationResult KimiCheckpointReader::read_expert_module_f32(
    const KimiExpertLocation& location, std::uint64_t rows,
    std::uint64_t columns, std::span<float> output) const {
    ValidationResult result;
    const auto layout = kimi_expert_layout(rows, columns);
    const auto quantization = kimi_expert_quantization();
    if (output.size() != rows * columns) {
        result.errors.emplace_back(
            "Kimi-K3 expert buffer disagrees with the module shape");
        return result;
    }
    if (location.packed_bytes != rows * (columns / 2U) ||
        location.scale_bytes != rows * (columns / 32U)) {
        result.errors.emplace_back(
            "Kimi-K3 expert module extent disagrees with the declared shape");
        return result;
    }
    std::vector<std::byte> packed(static_cast<std::size_t>(location.packed_bytes));
    std::vector<std::byte> scales(static_cast<std::size_t>(location.scale_bytes));
    auto read_packed = shards_.read(location.shard, location.packed_offset, packed,
                                    "routed expert weight_packed");
    if (!read_packed.ok()) return read_packed;
    auto read_scales = shards_.read(location.shard, location.scale_offset, scales,
                                    "routed expert weight_scale");
    if (!read_scales.ok()) return read_scales;
    for (std::uint64_t row = 0U; row < rows; ++row) {
        auto decoded = mxfp4_dequantize_row(
            output.subspan(static_cast<std::size_t>(row * columns),
                           static_cast<std::size_t>(columns)),
            packed, scales, layout, quantization, row);
        if (!decoded.ok()) {
            append(result.errors, std::move(decoded.errors));
            return result;
        }
    }
    return result;
}

}  // namespace strata
