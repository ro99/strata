#include "strata/models/glm53/glm53_checkpoint.hpp"

#include "../common/checkpoint_common.hpp"

#include <filesystem>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>

namespace strata {
namespace {

constexpr std::uint64_t kMaximumConfigBytes = 4ULL << 20U;
constexpr std::uint64_t kMaximumIndexBytes = 64ULL << 20U;

[[nodiscard]] bool deferred_weights_enabled() noexcept {
    const char* value = std::getenv("STRATA_GLM53_DEFERRED_WEIGHTS");
    return value == nullptr ||
           (std::string_view(value) != "0" &&
            std::string_view(value) != "false" &&
            std::string_view(value) != "off");
}

[[nodiscard]] bool phase_scheduled_reads_enabled() noexcept {
    const char* value = std::getenv("STRATA_GLM53_PHASE_SCHEDULER");
    return value == nullptr ||
           (std::string_view(value) != "0" &&
            std::string_view(value) != "false" &&
            std::string_view(value) != "off");
}

}  // namespace

Glm53CheckpointReader::~Glm53CheckpointReader() {
    std::scoped_lock lock(mapping_mutex_);
    for (const auto& [name, mapping] : mappings_) {
        static_cast<void>(name);
        if (mapping.address != nullptr && mapping.bytes != 0U) {
            static_cast<void>(munmap(mapping.address,
                                     static_cast<std::size_t>(mapping.bytes)));
        }
    }
}

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

ParseResult<std::span<const std::byte>> Glm53CheckpointReader::view(
    std::string_view name) const {
    ParseResult<std::span<const std::byte>> result;
    const auto* tensor = find(name);
    if (tensor == nullptr) {
        result.errors.push_back("unknown GLM-5.3 tensor " + std::string(name));
        return result;
    }
    std::scoped_lock lock(mapping_mutex_);
    auto mapping = mappings_.find(tensor->shard);
    if (mapping == mappings_.end()) {
        const int descriptor = shards_.descriptor(tensor->shard);
        struct stat status {};
        if (descriptor < 0 || fstat(descriptor, &status) != 0 ||
            status.st_size <= 0) {
            result.errors.push_back("cannot size GLM-5.3 checkpoint shard " +
                                    tensor->shard + ": " +
                                    std::strerror(errno));
            return result;
        }
        const auto bytes = static_cast<std::uint64_t>(status.st_size);
        void* address = mmap(nullptr, static_cast<std::size_t>(bytes), PROT_READ,
                             MAP_SHARED, descriptor, 0);
        if (address == MAP_FAILED) {
            result.errors.push_back("cannot map GLM-5.3 checkpoint shard " +
                                    tensor->shard + ": " +
                                    std::strerror(errno));
            return result;
        }
        mapping = mappings_.emplace(
            tensor->shard,
            ShardMapping{static_cast<std::byte*>(address), bytes}).first;
    }
    if (tensor->source_offset > mapping->second.bytes ||
        tensor->source_bytes > mapping->second.bytes - tensor->source_offset) {
        result.errors.push_back("mapped GLM-5.3 tensor exceeds its shard: " +
                                tensor->name);
        return result;
    }
    result.value = std::span<const std::byte>(
        mapping->second.address + tensor->source_offset,
        static_cast<std::size_t>(tensor->source_bytes));
    return result;
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
    const auto offset = row * columns * width;
    const auto bytes = columns * width;
    std::vector<std::byte> owned;
    std::span<const std::byte> encoded;
    if (phase_scheduled_reads_enabled()) {
        auto mapped = view(name);
        if (!mapped.ok()) {
            result.errors = std::move(mapped.errors);
            return result;
        }
        if (offset > mapped.value.size_bytes() ||
            bytes > mapped.value.size_bytes() - offset) {
            result.errors.push_back(
                "GLM-5.3 mapped row exceeds its tensor: " +
                std::string(name));
            return result;
        }
        encoded = mapped.value.subspan(static_cast<std::size_t>(offset),
                                       static_cast<std::size_t>(bytes));
    } else {
        auto loaded = read_slice(*tensor, offset, bytes);
        if (!loaded.ok()) {
            result.errors = std::move(loaded.errors);
            return result;
        }
        owned = std::move(loaded.value);
        encoded = owned;
    }
    result.value.resize(static_cast<std::size_t>(columns));
    for (std::size_t column = 0U; column < result.value.size(); ++column) {
        result.value[column] = detail::decode_plain_scalar(
            encoded.data() + column * width, tensor->source_dtype);
    }
    return result;
}

std::uint64_t Glm53CheckpointReader::cuda_linear_storage_bytes(
    std::string_view base_name) const {
    const auto weight_name = std::string(base_name) + ".weight";
    const auto* weight = find(weight_name);
    if (weight == nullptr) return 0U;
    std::uint64_t scale_bytes = 0U;
    if (weight->source_dtype == SafetensorsDtype::F8E4M3 ||
        weight->source_dtype == SafetensorsDtype::U8) {
        const auto* scale = find(
            std::string(base_name) +
            (weight->source_dtype == SafetensorsDtype::U8 ? ".weight_scale"
                                                          : ".weight_scale_inv"));
        if (scale == nullptr) return 0U;
        scale_bytes = scale->source_bytes;
    }
    return CudaBackend::weight_storage_bytes(weight->source_bytes,
                                              scale_bytes);
}

std::uint64_t Glm53CheckpointReader::cuda_linear_slice_storage_bytes(
    std::string_view base_name, std::uint64_t row_begin,
    std::uint64_t row_count) const {
    const auto* weight = find(std::string(base_name) + ".weight");
    if (weight == nullptr || weight->source_shape.size() != 2U ||
        row_count == 0U || row_begin > weight->source_shape[0] ||
        row_count > weight->source_shape[0] - row_begin) {
        return 0U;
    }
    if (weight->source_dtype == SafetensorsDtype::F8E4M3) {
        if (row_begin % 128U != 0U) return 0U;
        const auto* scale = find(
            std::string(base_name) + ".weight_scale_inv");
        if (scale == nullptr || scale->source_dtype != SafetensorsDtype::F32) {
            return 0U;
        }
        const auto scale_columns =
            (weight->source_shape[1] + 127U) / 128U;
        const auto scale_rows = (row_count + 127U) / 128U;
        return CudaBackend::weight_storage_bytes(
            row_count * weight->source_shape[1],
            scale_rows * scale_columns * sizeof(float));
    }
    if (weight->source_dtype != SafetensorsDtype::Bf16 &&
         weight->source_dtype != SafetensorsDtype::F16 &&
         weight->source_dtype != SafetensorsDtype::F32) {
        return 0U;
    }
    const auto width = safetensors_dtype_bytes(weight->source_dtype);
    return CudaBackend::weight_storage_bytes(
        row_count * weight->source_shape[1] * width, 0U);
}

ValidationResult Glm53CheckpointReader::load_cuda_linear(
    std::string_view base_name, std::uint64_t rows, std::uint64_t columns,
    int device, CudaBackend& backend, CudaWeight& output,
    bool concurrent_prefetch, bool canonical_layout) const {
    ValidationResult result;
    const auto weight_name = std::string(base_name) + ".weight";
    const auto* weight = find(weight_name);
    const auto expected_shape = weight != nullptr &&
                                weight->source_dtype == SafetensorsDtype::U8
        ? std::vector<std::uint64_t>{rows, columns / 2U}
        : std::vector<std::uint64_t>{rows, columns};
    if (weight == nullptr || columns == 0U ||
        (weight->source_dtype == SafetensorsDtype::U8 && columns % 32U != 0U) ||
        weight->source_shape != expected_shape) {
        result.errors.push_back("GLM-5.3 linear has an unexpected or missing weight: " +
                                weight_name);
        return result;
    }
    CudaWeightDescriptor descriptor;
    descriptor.dtype = weight->source_dtype;
    descriptor.rows = rows;
    descriptor.columns = columns;
    std::string scale_name;
    if (weight->source_dtype == SafetensorsDtype::F8E4M3) {
        scale_name = std::string(base_name) + ".weight_scale_inv";
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
        descriptor.encoding = CudaWeightEncoding::Fp8E4m3Block128F32;
        descriptor.packed_columns = columns;
        descriptor.scale_columns = scale_columns;
        descriptor.group_size = 128U;
    } else if (weight->source_dtype == SafetensorsDtype::U8) {
        scale_name = std::string(base_name) + ".weight_scale";
        const auto* scale = find(scale_name);
        const auto scale_columns = columns / 32U;
        if (scale == nullptr || scale->source_dtype != SafetensorsDtype::U8 ||
            scale->source_shape !=
                std::vector<std::uint64_t>{rows, scale_columns}) {
            result.errors.push_back(
                "GLM-5.3 MXFP4 linear has an invalid scale: " + scale_name);
            return result;
        }
        // The backend calls the byte-level E2M1 stream I8 even when the
        // Safetensors container declares those identical bytes U8.
        descriptor.dtype = SafetensorsDtype::I8;
        descriptor.encoding = CudaWeightEncoding::Fp4E2m1Group32;
        descriptor.packed_columns = columns / 2U;
        descriptor.scale_columns = scale_columns;
        descriptor.group_size = 32U;
    } else if (weight->source_dtype == SafetensorsDtype::Bf16 ||
               weight->source_dtype == SafetensorsDtype::F16 ||
               weight->source_dtype == SafetensorsDtype::F32) {
        descriptor.encoding = CudaWeightEncoding::Plain;
    } else {
        result.errors.push_back("GLM-5.3 linear uses an unsupported dtype: " +
                                weight_name);
        return result;
    }
    const auto fragment_layout = !canonical_layout &&
        descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128F32 &&
                backend.fp8_f32_register_fed_supported(device)
            ? CudaBackend::FragmentLayout::Prepack
            : CudaBackend::FragmentLayout::Canonical;
    if (deferred_weights_enabled()) {
        auto weights = view(weight_name);
        if (!weights.ok()) return {std::move(weights.errors)};
        std::span<const std::byte> scales;
        if (!scale_name.empty()) {
            auto mapped_scales = view(scale_name);
            if (!mapped_scales.ok()) return {std::move(mapped_scales.errors)};
            scales = mapped_scales.value;
        }
        // Both mapped views outlive every upload, so the copy stream may retain
        // them until its device-side completion event without a heap copy.
        return backend.upload(device, descriptor, weights.value, scales, output,
                              concurrent_prefetch
                                  ? CudaBackend::UploadCompletion::DeferredConcurrent
                                  : CudaBackend::UploadCompletion::Deferred,
                              fragment_layout);
    }

    // Same-binary control route for performance campaigns. This is the former
    // production contract: heap-copy each tensor and block each upload.
    auto weights = read(weight_name, rows * columns * 4U);
    if (!weights.ok()) return {std::move(weights.errors)};
    std::vector<std::byte> scales;
    if (!scale_name.empty()) {
        const auto scale_bytes =
            descriptor.encoding == CudaWeightEncoding::Fp4E2m1Group32
            ? rows * descriptor.scale_columns
            : descriptor.scale_columns * ((rows + 127U) / 128U) *
                  sizeof(float);
        auto loaded_scales = read(scale_name, scale_bytes);
        if (!loaded_scales.ok()) return {std::move(loaded_scales.errors)};
        scales = std::move(loaded_scales.value);
    }
    return backend.upload(device, descriptor, weights.value, scales, output,
                          CudaBackend::UploadCompletion::Synchronous,
                          fragment_layout);
}

ValidationResult Glm53CheckpointReader::load_cuda_linear_slice(
    std::string_view base_name, std::uint64_t total_rows,
    std::uint64_t columns, std::uint64_t row_begin,
    std::uint64_t row_count, int device, CudaBackend& backend,
    CudaWeight& output) const {
    ValidationResult result;
    const auto weight_name = std::string(base_name) + ".weight";
    const auto* weight = find(weight_name);
    if (weight == nullptr ||
        weight->source_shape != std::vector<std::uint64_t>{total_rows, columns} ||
        row_count == 0U || row_begin > total_rows ||
        row_count > total_rows - row_begin ||
        (weight->source_dtype != SafetensorsDtype::F8E4M3 &&
         weight->source_dtype != SafetensorsDtype::Bf16 &&
         weight->source_dtype != SafetensorsDtype::F16 &&
         weight->source_dtype != SafetensorsDtype::F32)) {
        return {{"GLM-5.3 sliced linear has an invalid weight or row range: " +
                 weight_name}};
    }
    const auto width = safetensors_dtype_bytes(weight->source_dtype);
    const auto row_bytes = columns * width;
    const auto offset = row_begin * row_bytes;
    const auto bytes = row_count * row_bytes;
    CudaWeightDescriptor descriptor;
    descriptor.dtype = weight->source_dtype;
    descriptor.rows = row_count;
    descriptor.columns = columns;
    std::span<const std::byte> scales;
    if (weight->source_dtype == SafetensorsDtype::F8E4M3) {
        if (row_begin % 128U != 0U) {
            return {{"GLM-5.3 FP8 slice must begin on a scale-block row"}};
        }
        const auto scale_name = std::string(base_name) + ".weight_scale_inv";
        auto mapped_scales = view(scale_name);
        const auto* scale = find(scale_name);
        const auto scale_columns = (columns + 127U) / 128U;
        const auto scale_row_begin = row_begin / 128U;
        const auto scale_rows = (row_count + 127U) / 128U;
        const auto scale_offset =
            scale_row_begin * scale_columns * sizeof(float);
        const auto scale_bytes = scale_rows * scale_columns * sizeof(float);
        if (!mapped_scales.ok()) return {std::move(mapped_scales.errors)};
        if (scale == nullptr || scale->source_dtype != SafetensorsDtype::F32 ||
            scale_offset > mapped_scales.value.size_bytes() ||
            scale_bytes > mapped_scales.value.size_bytes() - scale_offset) {
            return {{"GLM-5.3 FP8 slice has an invalid scale extent"}};
        }
        scales = mapped_scales.value.subspan(
            static_cast<std::size_t>(scale_offset),
            static_cast<std::size_t>(scale_bytes));
        descriptor.encoding = CudaWeightEncoding::Fp8E4m3Block128F32;
        descriptor.packed_columns = columns;
        descriptor.scale_columns = scale_columns;
        descriptor.group_size = 128U;
    } else {
        descriptor.encoding = CudaWeightEncoding::Plain;
    }
    if (deferred_weights_enabled()) {
        auto mapped = view(weight_name);
        if (!mapped.ok()) return {std::move(mapped.errors)};
        if (offset > mapped.value.size_bytes() ||
            bytes > mapped.value.size_bytes() - offset) {
            return {{"GLM-5.3 sliced linear exceeds its mapped weight"}};
        }
        return backend.upload(
            device, descriptor,
            mapped.value.subspan(static_cast<std::size_t>(offset),
                                 static_cast<std::size_t>(bytes)),
            scales, output, CudaBackend::UploadCompletion::Deferred,
            descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128F32 &&
                    backend.fp8_f32_register_fed_supported(device)
                ? CudaBackend::FragmentLayout::Prepack
                : CudaBackend::FragmentLayout::Canonical);
    }
    if (weight->source_dtype == SafetensorsDtype::F8E4M3) {
        return {{"GLM-5.3 FP8 slices require stable deferred mappings"}};
    }
    auto loaded = read_slice(*weight, offset, bytes);
    if (!loaded.ok()) return {std::move(loaded.errors)};
    return backend.upload(device, descriptor, loaded.value, {}, output);
}

}  // namespace strata
