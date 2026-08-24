#include "strata/gemma4_checkpoint.hpp"

#include "../common/checkpoint_common.hpp"

#include "strata/compressed_tensors.hpp"
#include "strata/model.hpp"
#include "strata/model_adapter.hpp"

#include <array>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <unordered_map>

namespace strata {
namespace {

constexpr std::uint64_t kMaximumIndexBytes = 16ULL << 20U;
constexpr std::uint64_t kMxfp4TensorCount = 1'598U;
constexpr std::uint64_t kMxfp4TensorBytes = 19'531'296'472ULL;
constexpr std::uint64_t kMxfp4ShardFileBytes = 19'531'513'296ULL;
constexpr std::string_view kSingleShardName = "model.safetensors";

void append(std::vector<std::string>& destination,
            std::vector<std::string> source) {
    for (auto& error : source) destination.push_back(std::move(error));
}

std::string layer_prefix(std::uint32_t layer) {
    return "model.language_model.layers." + std::to_string(layer) + ".";
}

std::string vision_layer_prefix(std::uint32_t layer) {
    return "model.vision_tower.encoder.layers." + std::to_string(layer) + ".";
}

}  // namespace

Gemma4CheckpointOpenResult Gemma4CheckpointReader::open(
    std::string model_directory) {
    Gemma4CheckpointOpenResult result;
    const auto spec = gemma4_31b_it_w8a16_spec();
    auto index = load_safetensors_index(model_directory, kMaximumIndexBytes);
    if (!index.ok()) {
        result.errors = std::move(index.errors);
        return result;
    }
    const bool w8a16 =
        index.value.total_size == spec.source.indexed_tensor_bytes &&
        index.value.entries.size() == spec.source.tensor_count &&
        index.value.shards.size() == spec.source.main_shards;
    const bool mxfp4 =
        index.value.total_size == kMxfp4TensorBytes &&
        index.value.entries.size() == kMxfp4TensorCount &&
        index.value.shards ==
            std::vector<std::string>{std::string(kSingleShardName)};
    if (!w8a16 && !mxfp4) {
        result.errors.emplace_back(
            "Gemma 4 checkpoint extent is neither pinned 31B W8A16 nor MXFP4");
        return result;
    }
    if (w8a16) {
        std::array<bool, 7> shards_seen{};
        for (const auto& shard : index.value.shards) {
            bool matched = false;
            for (std::size_t ordinal = 0U; ordinal < shards_seen.size(); ++ordinal) {
                char expected[40]{};
                std::snprintf(expected, sizeof(expected),
                              "model-%05zu-of-00007.safetensors", ordinal + 1U);
                if (shard == expected) {
                    shards_seen[ordinal] = true;
                    matched = true;
                    break;
                }
            }
            if (!matched) result.errors.emplace_back("unexpected Gemma 4 shard name " + shard);
        }
    }
    if (!result.errors.empty()) return result;

    auto reader = std::unique_ptr<Gemma4CheckpointReader>(new Gemma4CheckpointReader());
    reader->model_directory_ = std::move(model_directory);
    std::unordered_map<std::string, std::string> indexed;
    indexed.reserve(index.value.entries.size());
    for (const auto& entry : index.value.entries) indexed.emplace(entry.name, entry.shard);

    std::uint64_t shard_bytes = 0U;
    reader->tensors_.reserve(index.value.entries.size());
    for (const auto& shard_name : index.value.shards) {
        const auto path = (std::filesystem::path(reader->model_directory_) /
                           shard_name).string();
        auto shard = load_safetensors_shard(path);
        if (!shard.ok()) {
            append(result.errors, std::move(shard.errors));
            continue;
        }
        shard_bytes += shard.value.file_size;
        for (auto& tensor : shard.value.tensors) {
            const auto found = indexed.find(tensor.name);
            if (found == indexed.end() || found->second != shard_name) {
                result.errors.emplace_back("unindexed or misplaced Gemma 4 tensor " + tensor.name);
                continue;
            }
            reader->tensors_.push_back({std::move(tensor.name), shard_name,
                tensor.dtype, std::move(tensor.shape), tensor.absolute_begin,
                tensor.bytes()});
        }
    }
    if (!result.errors.empty()) return result;
    const auto expected_shard_bytes =
        w8a16 ? spec.source.shard_file_bytes : kMxfp4ShardFileBytes;
    const auto expected_tensor_count =
        w8a16 ? spec.source.tensor_count : kMxfp4TensorCount;
    if (shard_bytes != expected_shard_bytes ||
        reader->tensors_.size() != expected_tensor_count) {
        result.errors.emplace_back("Gemma 4 shard extent does not match the pinned checkpoint");
        return result;
    }
    reader->by_name_.reserve(reader->tensors_.size());
    for (std::size_t i = 0U; i < reader->tensors_.size(); ++i) {
        if (!reader->by_name_.emplace(reader->tensors_[i].name, i).second) {
            result.errors.emplace_back("duplicate Gemma 4 tensor " + reader->tensors_[i].name);
        }
    }
    if (!result.errors.empty()) return result;
    auto opened = reader->shards_.open(reader->model_directory_,
                                       index.value.shards, "Gemma 4");
    if (!opened.ok()) {
        result.errors = std::move(opened.errors);
        return result;
    }

    const auto require_plain = [&](std::string_view name,
                                   std::initializer_list<std::uint64_t> shape) {
        const auto* tensor = reader->find(name);
        if (tensor == nullptr || tensor->dtype != SafetensorsDtype::Bf16 ||
            tensor->shape != std::vector<std::uint64_t>(shape)) {
            result.errors.emplace_back("Gemma 4 plain tensor shape mismatch: " +
                                       std::string(name));
        }
    };
    const auto require_linear = [&](std::string_view base,
                                    std::uint64_t rows,
                                    std::uint64_t columns,
                                    bool quantized) {
        if (!quantized) {
            require_plain(std::string(base) + ".weight", {rows, columns});
            return;
        }
        const std::string packed_name = std::string(base) + ".weight_packed";
        const std::string scale_name = std::string(base) + ".weight_scale";
        const std::string shape_name = std::string(base) + ".weight_shape";
        const auto* packed = reader->find(packed_name);
        const auto* scales = reader->find(scale_name);
        const auto* logical = reader->find(shape_name);
        const auto w8_packed_columns = (columns + 3U) / 4U;
        const auto fp4_packed_columns = (columns + 1U) / 2U;
        const auto scale_columns = (columns + 31U) / 32U;
        const bool valid_w8 =
            packed != nullptr && scales != nullptr && logical != nullptr &&
            packed->dtype == SafetensorsDtype::I32 &&
            scales->dtype == SafetensorsDtype::Bf16 &&
            logical->dtype == SafetensorsDtype::I64 &&
            packed->shape == std::vector<std::uint64_t>{rows, w8_packed_columns} &&
            scales->shape == std::vector<std::uint64_t>{rows, scale_columns} &&
            logical->shape == std::vector<std::uint64_t>{2U};
        const bool valid_fp4 =
            packed != nullptr && scales != nullptr && logical == nullptr &&
            packed->dtype == SafetensorsDtype::U8 &&
            scales->dtype == SafetensorsDtype::U8 &&
            packed->shape == std::vector<std::uint64_t>{rows, fp4_packed_columns} &&
            scales->shape == std::vector<std::uint64_t>{rows, scale_columns};
        if (!valid_w8 && !valid_fp4) {
            result.errors.emplace_back("Gemma 4 compressed tensor shape mismatch: " +
                                       std::string(base));
            return;
        }
        if (valid_fp4) return;
        auto encoded = reader->read(shape_name, 16U);
        std::array<std::uint64_t, 2> decoded{};
        if (!encoded.ok()) {
            append(result.errors, std::move(encoded.errors));
        } else {
            auto status = decode_compressed_logical_shape(encoded.value, decoded);
            append(result.errors, std::move(status.errors));
            if (status.ok() && decoded != std::array<std::uint64_t, 2>{rows, columns}) {
                result.errors.emplace_back("Gemma 4 logical tensor shape mismatch: " +
                                           std::string(base));
            }
        }
    };

    const auto& c = kGemma4ExecutionContract;
    require_plain("model.language_model.embed_tokens.weight",
                  {c.vocabulary_size, c.hidden_size});
    require_plain("model.language_model.norm.weight", {c.hidden_size});
    require_linear("model.embed_vision.embedding_projection",
                   c.hidden_size, c.vision_hidden_size, false);
    for (std::uint32_t layer = 0U; layer < c.layer_count; ++layer) {
        const auto prefix = layer_prefix(layer);
        const auto attention = prefix + "self_attn.";
        const bool global = gemma4_global_attention_layer(layer);
        const auto head_dim = global ? c.global_head_dim : c.local_head_dim;
        const auto kv_heads = global ? c.global_key_value_heads : c.local_key_value_heads;
        require_plain(prefix + "input_layernorm.weight", {c.hidden_size});
        require_plain(prefix + "post_attention_layernorm.weight", {c.hidden_size});
        require_plain(prefix + "pre_feedforward_layernorm.weight", {c.hidden_size});
        require_plain(prefix + "post_feedforward_layernorm.weight", {c.hidden_size});
        require_plain(prefix + "layer_scalar", {1U});
        require_plain(attention + "q_norm.weight", {head_dim});
        require_plain(attention + "k_norm.weight", {head_dim});
        require_linear(attention + "q_proj", c.attention_heads * head_dim,
                       c.hidden_size, true);
        require_linear(attention + "k_proj", kv_heads * head_dim,
                       c.hidden_size, true);
        if (!global) {
            require_linear(attention + "v_proj", kv_heads * head_dim,
                           c.hidden_size, true);
        }
        require_linear(attention + "o_proj", c.hidden_size,
                       c.attention_heads * head_dim, true);
        require_linear(prefix + "mlp.gate_proj", c.intermediate_size,
                       c.hidden_size, true);
        require_linear(prefix + "mlp.up_proj", c.intermediate_size,
                       c.hidden_size, true);
        require_linear(prefix + "mlp.down_proj", c.hidden_size,
                       c.intermediate_size, true);
    }
    const auto patch_columns = 3U * c.vision_patch_size * c.vision_patch_size;
    require_linear("model.vision_tower.patch_embedder.input_proj",
                   c.vision_hidden_size, patch_columns, false);
    require_plain("model.vision_tower.patch_embedder.position_embedding_table",
                  {2U, c.vision_position_embeddings, c.vision_hidden_size});
    require_plain("model.vision_tower.std_bias", {c.vision_hidden_size});
    require_plain("model.vision_tower.std_scale", {c.vision_hidden_size});
    for (std::uint32_t layer = 0U; layer < c.vision_layer_count; ++layer) {
        const auto prefix = vision_layer_prefix(layer);
        const auto attention = prefix + "self_attn.";
        for (const auto* norm : {"input_layernorm.weight",
                                 "post_attention_layernorm.weight",
                                 "pre_feedforward_layernorm.weight",
                                 "post_feedforward_layernorm.weight"}) {
            require_plain(prefix + norm, {c.vision_hidden_size});
        }
        require_plain(attention + "q_norm.weight", {c.vision_head_dim});
        require_plain(attention + "k_norm.weight", {c.vision_head_dim});
        for (const auto* projection : {"q_proj.linear", "k_proj.linear",
                                       "v_proj.linear", "o_proj.linear"}) {
            require_linear(attention + projection, c.vision_hidden_size,
                           c.vision_hidden_size, false);
        }
        require_linear(prefix + "mlp.gate_proj.linear",
                       c.vision_intermediate_size, c.vision_hidden_size, false);
        require_linear(prefix + "mlp.up_proj.linear",
                       c.vision_intermediate_size, c.vision_hidden_size, false);
        require_linear(prefix + "mlp.down_proj.linear",
                       c.vision_hidden_size, c.vision_intermediate_size, false);
    }
    if (!result.errors.empty()) return result;
    result.value = std::move(reader);
    return result;
}

const Gemma4Tensor* Gemma4CheckpointReader::find(
    std::string_view name) const noexcept {
    const auto found = by_name_.find(name);
    return found == by_name_.end() ? nullptr : &tensors_[found->second];
}

ParseResult<std::vector<std::byte>> Gemma4CheckpointReader::read(
    std::string_view name, std::uint64_t maximum_bytes) const {
    ParseResult<std::vector<std::byte>> result;
    const auto* tensor = find(name);
    if (tensor == nullptr) {
        result.errors.emplace_back("Gemma 4 tensor does not exist: " + std::string(name));
        return result;
    }
    if (tensor->bytes > maximum_bytes ||
        tensor->bytes > std::numeric_limits<std::size_t>::max()) {
        result.errors.emplace_back("Gemma 4 tensor exceeds the read ceiling: " +
                                   std::string(name));
        return result;
    }
    result.value.resize(static_cast<std::size_t>(tensor->bytes));
    auto status = shards_.read(tensor->shard, tensor->offset,
                               result.value, tensor->name);
    if (!status.ok()) {
        result.errors = std::move(status.errors);
        result.value.clear();
    }
    return result;
}

ParseResult<std::vector<float>> Gemma4CheckpointReader::read_f32(
    std::string_view name, std::uint64_t maximum_elements) const {
    ParseResult<std::vector<float>> result;
    const auto* tensor = find(name);
    const auto width = tensor == nullptr ? 0U : safetensors_dtype_bytes(tensor->dtype);
    if (tensor == nullptr || (tensor->dtype != SafetensorsDtype::Bf16 &&
                              tensor->dtype != SafetensorsDtype::F16 &&
                              tensor->dtype != SafetensorsDtype::F32) ||
        width == 0U || tensor->bytes / width > maximum_elements) {
        result.errors.emplace_back("Gemma 4 tensor cannot be decoded as bounded FP32: " +
                                   std::string(name));
        return result;
    }
    auto encoded = read(name, tensor->bytes);
    if (!encoded.ok()) {
        result.errors = std::move(encoded.errors);
        return result;
    }
    result.value.resize(static_cast<std::size_t>(tensor->bytes / width));
    for (std::size_t i = 0U; i < result.value.size(); ++i) {
        result.value[i] = detail::decode_plain_scalar(
            encoded.value.data() + i * width, tensor->dtype);
    }
    return result;
}

ParseResult<std::vector<float>> Gemma4CheckpointReader::read_f32_row(
    std::string_view name, std::uint64_t row) const {
    ParseResult<std::vector<float>> result;
    const auto* tensor = find(name);
    if (tensor == nullptr || tensor->shape.size() != 2U || row >= tensor->shape[0]) {
        result.errors.emplace_back("Gemma 4 tensor row is out of range: " + std::string(name));
        return result;
    }
    const auto width = safetensors_dtype_bytes(tensor->dtype);
    if ((tensor->dtype != SafetensorsDtype::Bf16 &&
         tensor->dtype != SafetensorsDtype::F16 &&
         tensor->dtype != SafetensorsDtype::F32) || width == 0U) {
        result.errors.emplace_back("Gemma 4 tensor row is not floating point");
        return result;
    }
    const auto row_bytes = tensor->shape[1] * width;
    std::vector<std::byte> encoded(static_cast<std::size_t>(row_bytes));
    auto status = shards_.read(tensor->shard, tensor->offset + row * row_bytes,
                               encoded, tensor->name);
    if (!status.ok()) {
        result.errors = std::move(status.errors);
        return result;
    }
    result.value.resize(static_cast<std::size_t>(tensor->shape[1]));
    for (std::size_t i = 0U; i < result.value.size(); ++i) {
        result.value[i] = detail::decode_plain_scalar(
            encoded.data() + i * width, tensor->dtype);
    }
    return result;
}

ValidationResult load_gemma4_cuda_linear(
    const Gemma4CheckpointReader& checkpoint, std::string_view base_name,
    std::uint64_t rows, std::uint64_t columns, int device,
    CudaBackend& backend, CudaWeight& output) {
    ValidationResult result;
    const std::string plain_name = std::string(base_name) + ".weight";
    if (const auto* tensor = checkpoint.find(plain_name); tensor != nullptr) {
        if (tensor->shape != std::vector<std::uint64_t>{rows, columns}) {
            result.errors.emplace_back("Gemma 4 plain CUDA linear shape mismatch: " +
                                       std::string(base_name));
            return result;
        }
        auto data = checkpoint.read(plain_name, tensor->bytes);
        if (!data.ok()) {
            result.errors = std::move(data.errors);
            return result;
        }
        CudaWeightDescriptor descriptor;
        descriptor.encoding = CudaWeightEncoding::Plain;
        descriptor.dtype = tensor->dtype;
        descriptor.rows = rows;
        descriptor.columns = columns;
        return backend.upload(device, descriptor, data.value, {}, output);
    }
    const std::string packed_name = std::string(base_name) + ".weight_packed";
    const std::string scale_name = std::string(base_name) + ".weight_scale";
    const auto* packed = checkpoint.find(packed_name);
    const auto* scales = checkpoint.find(scale_name);
    if (packed == nullptr || scales == nullptr) {
        result.errors.emplace_back("Gemma 4 compressed CUDA linear is missing: " +
                                   std::string(base_name));
        return result;
    }
    auto described = describe_gemma4_cuda_linear(
        *packed, *scales, rows, columns);
    if (!described.ok()) {
        for (auto& error : described.errors) {
            result.errors.emplace_back(std::string(base_name) + ": " + error);
        }
        return result;
    }
    auto packed_data = checkpoint.read(packed_name, packed->bytes);
    auto scale_data = checkpoint.read(scale_name, scales->bytes);
    if (!packed_data.ok()) append(result.errors, std::move(packed_data.errors));
    if (!scale_data.ok()) append(result.errors, std::move(scale_data.errors));
    if (!result.ok()) return result;
    result = backend.upload(device, described.value, packed_data.value,
                            scale_data.value, output);
    if (result.ok() &&
        described.value.encoding == CudaWeightEncoding::Fp4E2m1Group32 &&
        register_fed_matmul_enabled()) {
        result = backend.prepack_marlin(device, output);
    }
    return result;
}

ParseResult<CudaWeightDescriptor> describe_gemma4_cuda_linear(
    const Gemma4Tensor& packed, const Gemma4Tensor& scales,
    std::uint64_t rows, std::uint64_t columns) {
    ParseResult<CudaWeightDescriptor> result;
    const auto scale_columns = columns / 32U;
    const bool w8a16 =
        columns % 4U == 0U && packed.dtype == SafetensorsDtype::I32 &&
        scales.dtype == SafetensorsDtype::Bf16 &&
        packed.shape == std::vector<std::uint64_t>{rows, columns / 4U} &&
        scales.shape == std::vector<std::uint64_t>{rows, scale_columns};
    const bool mxfp4 =
        columns % 2U == 0U && packed.dtype == SafetensorsDtype::U8 &&
        scales.dtype == SafetensorsDtype::U8 &&
        packed.shape == std::vector<std::uint64_t>{rows, columns / 2U} &&
        scales.shape == std::vector<std::uint64_t>{rows, scale_columns};
    if (columns % 32U != 0U || (!w8a16 && !mxfp4)) {
        result.errors.emplace_back("Gemma 4 CUDA linear dtype or shape mismatch");
        return result;
    }
    result.value.encoding = mxfp4
        ? CudaWeightEncoding::Fp4E2m1Group32
        : CudaWeightEncoding::OffsetPackedInt8;
    // Fp4E2m1Group32 names the byte-level E2M1 code stream as I8 in the
    // backend contract even when Safetensors stores the same bytes as U8.
    result.value.dtype = mxfp4 ? SafetensorsDtype::I8 : SafetensorsDtype::I32;
    result.value.rows = rows;
    result.value.columns = columns;
    result.value.packed_columns = mxfp4 ? columns / 2U : columns / 4U;
    result.value.scale_columns = scale_columns;
    result.value.group_size = 32U;
    return result;
}

}  // namespace strata
