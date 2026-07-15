#include "strata/checkpoint.hpp"

#include <bit>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>

namespace strata {

namespace {

constexpr std::uint64_t kMaximumIndexBytes = 64ULL << 20U;

float decode_scalar(const std::byte* bytes, SafetensorsDtype dtype) {
    if (dtype == SafetensorsDtype::Bf16) {
        std::uint16_t value = 0U;
        std::memcpy(&value, bytes, sizeof(value));
        return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
    }
    if (dtype == SafetensorsDtype::F16) {
        std::uint16_t value = 0U;
        std::memcpy(&value, bytes, sizeof(value));
        const std::uint32_t sign = (value & 0x8000U) << 16U;
        std::uint32_t exponent = (value >> 10U) & 0x1FU;
        std::uint32_t mantissa = value & 0x3FFU;
        if (exponent == 0U) {
            if (mantissa == 0U) return std::bit_cast<float>(sign);
            std::int32_t shift = 0;
            while ((mantissa & 0x400U) == 0U) {
                mantissa <<= 1U;
                ++shift;
            }
            mantissa &= 0x3FFU;
            exponent = static_cast<std::uint32_t>(127 - 15 - shift);
        } else if (exponent == 0x1FU) {
            exponent = 0xFFU;
        } else {
            exponent += 127U - 15U;
        }
        return std::bit_cast<float>(sign | (exponent << 23U) | (mantissa << 13U));
    }
    float value = 0.0F;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

bool checked_product(std::uint64_t left, std::uint64_t right, std::uint64_t& output) {
    if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left) return false;
    output = left * right;
    return true;
}

void append_errors(ValidationResult& output, std::vector<std::string> errors) {
    for (auto& error : errors) output.errors.push_back(std::move(error));
}

}  // namespace

GlmCheckpointReader::~GlmCheckpointReader() {
    for (const auto& [name, descriptor] : shard_fds_) {
        static_cast<void>(name);
        if (descriptor >= 0) static_cast<void>(close(descriptor));
    }
}

GlmCheckpointOpenResult GlmCheckpointReader::open(std::string model_directory,
                                                  bool require_read_only) {
    GlmCheckpointOpenResult result;
    const auto index_path =
        (std::filesystem::path(model_directory) / "model.safetensors.index.json").string();
    auto text = load_bounded_text_file(index_path, kMaximumIndexBytes);
    if (!text.ok()) {
        result.errors = std::move(text.errors);
        return result;
    }
    auto index = parse_safetensors_index(text.value);
    if (!index.ok()) {
        result.errors = std::move(index.errors);
        return result;
    }
    auto built = build_quanttrio_glm52_index_manifest(std::move(index.value));
    if (!built.ok()) {
        result.errors = std::move(built.errors);
        return result;
    }
    GlmCheckpointOptions options;
    options.require_all_shards = true;
    options.require_read_only = require_read_only;
    options.validate_logical_shapes = true;
    auto validated = validate_quanttrio_glm52_checkpoint(
        model_directory, std::move(built.manifest), options);
    if (!validated.ok()) {
        result.errors = std::move(validated.errors);
        return result;
    }

    auto reader = std::unique_ptr<GlmCheckpointReader>(new GlmCheckpointReader());
    reader->model_directory_ = std::move(model_directory);
    reader->manifest_ = std::move(validated.manifest);
    reader->tensors_.reserve(reader->manifest_.tensors.size());
    for (std::size_t index_value = 0; index_value < reader->manifest_.tensors.size();
         ++index_value) {
        reader->tensors_.emplace(reader->manifest_.tensors[index_value].name, index_value);
    }
    for (const auto& shard : reader->manifest_.shards) {
        const auto path = (std::filesystem::path(reader->model_directory_) / shard).string();
        const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor < 0) {
            result.errors.emplace_back("cannot open checkpoint shard " + path + ": " +
                                       std::strerror(errno));
            return result;
        }
        reader->shard_fds_.emplace(shard, descriptor);
    }
    result.value = std::move(reader);
    return result;
}

const GlmManifestTensor* GlmCheckpointReader::find(std::string_view name) const noexcept {
    const auto found = tensors_.find(name);
    return found == tensors_.end() ? nullptr : &manifest_.tensors[found->second];
}

ParseResult<std::vector<std::byte>> GlmCheckpointReader::pread_tensor(
    const GlmManifestTensor& tensor, std::uint64_t relative_offset,
    std::uint64_t bytes) const {
    ParseResult<std::vector<std::byte>> result;
    if (relative_offset > tensor.source_bytes || bytes > tensor.source_bytes - relative_offset) {
        result.errors.emplace_back("tensor slice exceeds " + tensor.name);
        return result;
    }
    const auto found = shard_fds_.find(tensor.shard);
    if (found == shard_fds_.end()) {
        result.errors.emplace_back("checkpoint shard is not open for " + tensor.name);
        return result;
    }
    if (tensor.source_offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) -
                                   relative_offset) {
        result.errors.emplace_back("tensor file offset exceeds the platform range");
        return result;
    }
    result.value.resize(static_cast<std::size_t>(bytes));
    {
        std::scoped_lock lock(read_interval_mutex_);
        if (active_reads_++ == 0U) read_interval_started_ = std::chrono::steady_clock::now();
    }
    const auto finish_interval = [this] {
        std::scoped_lock lock(read_interval_mutex_);
        if (--active_reads_ == 0U) {
            read_wall_nanoseconds_ += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - read_interval_started_).count());
        }
    };
    const auto started = std::chrono::steady_clock::now();
    std::uint64_t completed = 0U;
    while (completed < bytes) {
        const auto request = static_cast<std::size_t>(std::min<std::uint64_t>(
            bytes - completed, static_cast<std::uint64_t>(std::numeric_limits<ssize_t>::max())));
        const auto count = pread(found->second, result.value.data() + completed, request,
                                 static_cast<off_t>(tensor.source_offset + relative_offset +
                                                    completed));
        if (count < 0) {
            if (errno == EINTR) continue;
            result.errors.emplace_back("cannot read tensor " + tensor.name + ": " +
                                       std::strerror(errno));
            result.value.clear();
            finish_interval();
            return result;
        }
        if (count == 0) {
            result.errors.emplace_back("unexpected end of shard while reading " + tensor.name);
            result.value.clear();
            finish_interval();
            return result;
        }
        completed += static_cast<std::uint64_t>(count);
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started);
    read_calls_.fetch_add(1U, std::memory_order_relaxed);
    read_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    read_nanoseconds_.fetch_add(static_cast<std::uint64_t>(elapsed.count()),
                                std::memory_order_relaxed);
    finish_interval();
    return result;
}

ParseResult<std::vector<std::byte>> GlmCheckpointReader::read(
    std::string_view name, std::uint64_t maximum_bytes) const {
    ParseResult<std::vector<std::byte>> result;
    const auto* tensor = find(name);
    if (tensor == nullptr) {
        result.errors.emplace_back("checkpoint tensor does not exist: " + std::string(name));
        return result;
    }
    if (tensor->source_bytes > maximum_bytes) {
        result.errors.emplace_back("checkpoint tensor exceeds the caller's byte ceiling: " +
                                   tensor->name);
        return result;
    }
    return pread_tensor(*tensor, 0U, tensor->source_bytes);
}

ParseResult<std::vector<std::byte>> GlmCheckpointReader::read_slice(
    std::string_view name, std::uint64_t relative_offset, std::uint64_t bytes) const {
    ParseResult<std::vector<std::byte>> result;
    const auto* tensor = find(name);
    if (tensor == nullptr) {
        result.errors.emplace_back("checkpoint tensor does not exist: " + std::string(name));
        return result;
    }
    return pread_tensor(*tensor, relative_offset, bytes);
}

ParseResult<std::vector<float>> GlmCheckpointReader::read_f32(
    std::string_view name, std::uint64_t maximum_elements) const {
    ParseResult<std::vector<float>> result;
    const auto* tensor = find(name);
    if (tensor == nullptr) {
        result.errors.emplace_back("checkpoint tensor does not exist: " + std::string(name));
        return result;
    }
    const auto element_bytes = safetensors_dtype_bytes(tensor->source_dtype);
    if ((tensor->source_dtype != SafetensorsDtype::Bf16 &&
         tensor->source_dtype != SafetensorsDtype::F16 &&
         tensor->source_dtype != SafetensorsDtype::F32) ||
        element_bytes == 0U || tensor->source_bytes % element_bytes != 0U ||
        tensor->source_bytes / element_bytes > maximum_elements) {
        result.errors.emplace_back("tensor cannot be decoded within the requested FP32 ceiling: " +
                                   tensor->name);
        return result;
    }
    auto encoded = pread_tensor(*tensor, 0U, tensor->source_bytes);
    if (!encoded.ok()) {
        result.errors = std::move(encoded.errors);
        return result;
    }
    const auto elements = tensor->source_bytes / element_bytes;
    result.value.resize(static_cast<std::size_t>(elements));
    for (std::uint64_t index = 0U; index < elements; ++index) {
        result.value[static_cast<std::size_t>(index)] =
            decode_scalar(encoded.value.data() + index * element_bytes, tensor->source_dtype);
    }
    return result;
}

ParseResult<std::vector<float>> GlmCheckpointReader::read_f32_row(
    std::string_view name, std::uint64_t row) const {
    ParseResult<std::vector<float>> result;
    const auto* tensor = find(name);
    if (tensor == nullptr || tensor->source_shape.size() != 2U ||
        row >= (tensor == nullptr ? 0U : tensor->source_shape[0])) {
        result.errors.emplace_back("cannot address requested tensor row: " + std::string(name));
        return result;
    }
    const auto element_bytes = safetensors_dtype_bytes(tensor->source_dtype);
    if (tensor->source_dtype != SafetensorsDtype::Bf16 &&
        tensor->source_dtype != SafetensorsDtype::F16 &&
        tensor->source_dtype != SafetensorsDtype::F32) {
        result.errors.emplace_back("requested tensor row is not floating point");
        return result;
    }
    std::uint64_t row_bytes = 0U;
    if (!checked_product(tensor->source_shape[1], element_bytes, row_bytes)) {
        result.errors.emplace_back("tensor row byte count overflows");
        return result;
    }
    auto encoded = pread_tensor(*tensor, row * row_bytes, row_bytes);
    if (!encoded.ok()) {
        result.errors = std::move(encoded.errors);
        return result;
    }
    result.value.resize(static_cast<std::size_t>(tensor->source_shape[1]));
    for (std::size_t index = 0U; index < result.value.size(); ++index) {
        result.value[index] = decode_scalar(encoded.value.data() + index * element_bytes,
                                            tensor->source_dtype);
    }
    return result;
}

CheckpointReadStats GlmCheckpointReader::stats() const noexcept {
    std::scoped_lock lock(read_interval_mutex_);
    return {read_calls_.load(std::memory_order_relaxed),
            read_bytes_.load(std::memory_order_relaxed),
            read_nanoseconds_.load(std::memory_order_relaxed),
            read_wall_nanoseconds_};
}

ValidationResult load_glm_cuda_linear(const GlmCheckpointReader& checkpoint,
                                      std::string_view base_name,
                                      std::uint64_t expected_rows,
                                      std::uint64_t expected_columns,
                                      int device, CudaBackend& backend,
                                      CudaWeight& output) {
    ValidationResult result;
    const std::string plain_name = std::string(base_name) + ".weight";
    if (const auto* plain = checkpoint.find(plain_name); plain != nullptr) {
        if (plain->source_shape != std::vector<std::uint64_t>{expected_rows, expected_columns}) {
            result.errors.emplace_back("plain linear shape mismatch for " + plain_name);
            return result;
        }
        auto data = checkpoint.read(plain_name, plain->source_bytes);
        if (!data.ok()) {
            append_errors(result, std::move(data.errors));
            return result;
        }
        CudaWeightDescriptor descriptor;
        descriptor.encoding = CudaWeightEncoding::Plain;
        descriptor.dtype = plain->source_dtype;
        descriptor.rows = expected_rows;
        descriptor.columns = expected_columns;
        return backend.upload(device, descriptor, data.value, {}, output);
    }

    const std::string packed_name = std::string(base_name) + ".weight_packed";
    const std::string scale_name = std::string(base_name) + ".weight_scale";
    const auto* packed = checkpoint.find(packed_name);
    const auto* scales = checkpoint.find(scale_name);
    if (packed == nullptr || scales == nullptr || packed->source_shape.size() != 2U ||
        scales->source_shape.size() != 2U || packed->encoding != scales->encoding) {
        result.errors.emplace_back("compressed linear triplet is missing or inconsistent for " +
                                   std::string(base_name));
        return result;
    }
    const std::uint32_t bits = packed->encoding == GlmTensorEncoding::Int4Group128 ? 4U : 8U;
    const std::uint32_t group_size =
        packed->encoding == GlmTensorEncoding::Int8Channel ? 0U : 128U;
    const auto expected_packed_columns =
        (expected_columns + (32U / bits) - 1U) / (32U / bits);
    const auto expected_scale_columns = group_size == 0U
                                            ? 1U
                                            : (expected_columns + group_size - 1U) / group_size;
    if (packed->source_shape !=
            std::vector<std::uint64_t>{expected_rows, expected_packed_columns} ||
        scales->source_shape !=
            std::vector<std::uint64_t>{expected_rows, expected_scale_columns}) {
        result.errors.emplace_back("compressed linear shape mismatch for " +
                                   std::string(base_name));
        return result;
    }
    auto packed_data = checkpoint.read(packed_name, packed->source_bytes);
    auto scale_data = checkpoint.read(scale_name, scales->source_bytes);
    if (!packed_data.ok()) append_errors(result, std::move(packed_data.errors));
    if (!scale_data.ok()) append_errors(result, std::move(scale_data.errors));
    if (!result.ok()) return result;

    CudaWeightDescriptor descriptor;
    descriptor.encoding = bits == 4U ? CudaWeightEncoding::OffsetPackedInt4
                                     : CudaWeightEncoding::OffsetPackedInt8;
    descriptor.dtype = SafetensorsDtype::I32;
    descriptor.rows = expected_rows;
    descriptor.columns = expected_columns;
    descriptor.packed_columns = expected_packed_columns;
    descriptor.scale_columns = expected_scale_columns;
    descriptor.group_size = group_size;
    return backend.upload(device, descriptor, packed_data.value, scale_data.value, output);
}

}  // namespace strata
