#include "strata/models/glm53/glm53_checkpoint.hpp"

#include "../common/checkpoint_common.hpp"

#include <filesystem>

namespace strata {
namespace {

constexpr std::uint64_t kMaximumConfigBytes = 4ULL << 20U;
constexpr std::uint64_t kMaximumIndexBytes = 64ULL << 20U;

}  // namespace

Glm53CheckpointOpenResult Glm53CheckpointReader::open(
    std::string model_directory) {
    Glm53CheckpointOpenResult result;
    const auto root = std::filesystem::path(model_directory);
    auto config_text = load_bounded_text_file(
        (root / "config.json").string(), kMaximumConfigBytes);
    if (!config_text.ok()) {
        result.errors = std::move(config_text.errors);
        return result;
    }
    auto config = parse_glm53_config(config_text.value);
    if (!config.ok()) {
        result.errors = std::move(config.errors);
        return result;
    }
    auto config_gate = validate_glm53_config(config.value);
    if (!config_gate.ok()) {
        result.errors = std::move(config_gate.errors);
        return result;
    }
    auto index = load_safetensors_index(model_directory, kMaximumIndexBytes);
    if (!index.ok()) {
        result.errors = std::move(index.errors);
        return result;
    }
    auto built = build_glm53_index_manifest(std::move(index.value));
    if (!built.ok()) {
        result.errors = std::move(built.errors);
        return result;
    }
    auto validated = validate_glm53_checkpoint(
        model_directory, std::move(built.manifest));
    if (!validated.ok()) {
        result.errors = std::move(validated.errors);
        return result;
    }
    auto reader = std::unique_ptr<Glm53CheckpointReader>(
        new Glm53CheckpointReader());
    reader->model_directory_ = std::move(model_directory);
    reader->config_ = std::move(config.value);
    reader->manifest_ = std::move(validated.manifest);
    reader->by_name_.reserve(reader->manifest_.tensors.size());
    for (std::size_t index_of = 0U;
         index_of < reader->manifest_.tensors.size(); ++index_of) {
        reader->by_name_.emplace(reader->manifest_.tensors[index_of].name,
                                 index_of);
    }
    auto opened = reader->shards_.open(reader->model_directory_,
                                       reader->manifest_.shards,
                                       "GLM-5.3-Flash");
    if (!opened.ok()) {
        result.errors = std::move(opened.errors);
        return result;
    }
    result.value = std::move(reader);
    return result;
}

const Glm53ManifestTensor* Glm53CheckpointReader::find(
    std::string_view name) const noexcept {
    const auto found = by_name_.find(name);
    return found == by_name_.end() ? nullptr
                                   : &manifest_.tensors[found->second];
}

ParseResult<std::vector<std::byte>> Glm53CheckpointReader::read_slice(
    const Glm53ManifestTensor& tensor, std::uint64_t offset,
    std::uint64_t bytes) const {
    ParseResult<std::vector<std::byte>> result;
    if (offset > tensor.source_bytes || bytes > tensor.source_bytes - offset) {
        result.errors.push_back("GLM-5.3 tensor slice is out of bounds: " +
                                tensor.name);
        return result;
    }
    result.value.resize(static_cast<std::size_t>(bytes));
    auto status = shards_.read(tensor.shard, tensor.source_offset + offset,
                               result.value, tensor.name);
    if (!status.ok()) {
        result.errors = std::move(status.errors);
        result.value.clear();
    }
    return result;
}

ParseResult<std::vector<std::byte>> Glm53CheckpointReader::read(
    std::string_view name, std::uint64_t maximum_bytes) const {
    ParseResult<std::vector<std::byte>> result;
    const auto* tensor = find(name);
    if (tensor == nullptr) {
        result.errors.push_back("unknown GLM-5.3 tensor " + std::string(name));
        return result;
    }
    if (tensor->source_bytes > maximum_bytes) {
        result.errors.push_back("GLM-5.3 tensor exceeds the caller byte budget: " +
                                std::string(name));
        return result;
    }
    return read_slice(*tensor, 0U, tensor->source_bytes);
}

ParseResult<std::vector<float>> Glm53CheckpointReader::read_f32(
    std::string_view name, std::uint64_t maximum_elements) const {
    ParseResult<std::vector<float>> result;
    const auto* tensor = find(name);
    if (tensor == nullptr) {
        result.errors.push_back("unknown GLM-5.3 tensor " + std::string(name));
        return result;
    }
    const auto width = safetensors_dtype_bytes(tensor->source_dtype);
    if ((tensor->source_dtype != SafetensorsDtype::Bf16 &&
         tensor->source_dtype != SafetensorsDtype::F16 &&
         tensor->source_dtype != SafetensorsDtype::F32) || width == 0U ||
        tensor->source_bytes % width != 0U ||
        tensor->source_bytes / width != maximum_elements) {
        result.errors.push_back("GLM-5.3 tensor does not match the required F32 extent: " +
                                std::string(name));
        return result;
    }
    auto encoded = read_slice(*tensor, 0U, tensor->source_bytes);
    if (!encoded.ok()) {
        result.errors = std::move(encoded.errors);
        return result;
    }
    const auto elements = tensor->source_bytes / width;
    result.value.resize(static_cast<std::size_t>(elements));
    for (std::size_t index = 0U; index < result.value.size(); ++index) {
        result.value[index] = detail::decode_plain_scalar(
            encoded.value.data() + index * width, tensor->source_dtype);
    }
    return result;
}

ParseResult<std::vector<float>> Glm53CheckpointReader::read_f32_row(
    std::string_view name, std::uint64_t row) const {
    ParseResult<std::vector<float>> result;
    const auto* tensor = find(name);
    if (tensor == nullptr || tensor->source_shape.size() != 2U ||
        row >= tensor->source_shape[0]) {
        result.errors.push_back("GLM-5.3 tensor row is invalid: " +
                                std::string(name));
        return result;
    }
    const auto width = safetensors_dtype_bytes(tensor->source_dtype);
    if ((tensor->source_dtype != SafetensorsDtype::Bf16 &&
         tensor->source_dtype != SafetensorsDtype::F16 &&
         tensor->source_dtype != SafetensorsDtype::F32) || width == 0U) {
        result.errors.push_back("GLM-5.3 row tensor is not plain floating point: " +
                                std::string(name));
        return result;
    }
    const auto columns = tensor->source_shape[1];
    auto encoded = read_slice(*tensor, row * columns * width, columns * width);
    if (!encoded.ok()) {
        result.errors = std::move(encoded.errors);
        return result;
    }
    result.value.resize(static_cast<std::size_t>(columns));
    for (std::size_t column = 0U; column < result.value.size(); ++column) {
        result.value[column] = detail::decode_plain_scalar(
            encoded.value.data() + column * width, tensor->source_dtype);
    }
    return result;
}

std::uint64_t Glm53CheckpointReader::cuda_linear_storage_bytes(
    std::string_view base_name) const {
    const auto weight_name = std::string(base_name) + ".weight";
    const auto* weight = find(weight_name);
    if (weight == nullptr) return 0U;
    std::uint64_t scale_bytes = 0U;
    if (weight->source_dtype == SafetensorsDtype::F8E4M3) {
        const auto* scale = find(std::string(base_name) + ".weight_scale_inv");
        if (scale == nullptr) return 0U;
        scale_bytes = scale->source_bytes;
    }
    return CudaBackend::weight_storage_bytes(weight->source_bytes,
                                              scale_bytes);
}

ValidationResult Glm53CheckpointReader::load_cuda_linear(
    std::string_view base_name, std::uint64_t rows, std::uint64_t columns,
    int device, CudaBackend& backend, CudaWeight& output) const {
    ValidationResult result;
    const auto weight_name = std::string(base_name) + ".weight";
    const auto* weight = find(weight_name);
    if (weight == nullptr || weight->source_shape !=
            std::vector<std::uint64_t>{rows, columns}) {
        result.errors.push_back("GLM-5.3 linear has an unexpected or missing weight: " +
                                weight_name);
        return result;
    }
    auto weights = read(weight_name, rows * columns * 4U);
    if (!weights.ok()) {
        result.errors = std::move(weights.errors);
        return result;
    }
    CudaWeightDescriptor descriptor;
    descriptor.dtype = weight->source_dtype;
    descriptor.rows = rows;
    descriptor.columns = columns;
    std::vector<std::byte> scale_bytes;
    if (weight->source_dtype == SafetensorsDtype::F8E4M3) {
        const auto scale_name = std::string(base_name) + ".weight_scale_inv";
        const auto* scale = find(scale_name);
        const auto scale_rows = (rows + 127U) / 128U;
        const auto scale_columns = (columns + 127U) / 128U;
        if (scale == nullptr || scale->source_dtype != SafetensorsDtype::F32 ||
            scale->source_shape !=
                std::vector<std::uint64_t>{scale_rows, scale_columns}) {
            result.errors.push_back("GLM-5.3 FP8 linear has an invalid scale: " +
                                    scale_name);
            return result;
        }
        auto scales = read(scale_name, scale_rows * scale_columns * 4U);
        if (!scales.ok()) {
            result.errors = std::move(scales.errors);
            return result;
        }
        scale_bytes = std::move(scales.value);
        descriptor.encoding = CudaWeightEncoding::Fp8E4m3Block128F32;
        descriptor.packed_columns = columns;
        descriptor.scale_columns = scale_columns;
        descriptor.group_size = 128U;
    } else if (weight->source_dtype == SafetensorsDtype::Bf16 ||
               weight->source_dtype == SafetensorsDtype::F16 ||
               weight->source_dtype == SafetensorsDtype::F32) {
        descriptor.encoding = CudaWeightEncoding::Plain;
    } else {
        result.errors.push_back("GLM-5.3 linear uses an unsupported dtype: " +
                                weight_name);
        return result;
    }
    const auto fragment_layout =
        descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128F32 &&
                backend.fp8_f32_register_fed_supported(device)
            ? CudaBackend::FragmentLayout::Prepack
            : CudaBackend::FragmentLayout::Canonical;
    return backend.upload(device, descriptor, weights.value, scale_bytes,
                          output, CudaBackend::UploadCompletion::Synchronous,
                          fragment_layout);
}

}  // namespace strata
