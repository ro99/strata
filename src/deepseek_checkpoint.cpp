#include "strata/deepseek_checkpoint.hpp"

#include "strata/deepseek_ops.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <limits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace strata {

namespace {

constexpr std::uint64_t kMaximumIndexBytes = 16ULL << 20U;

[[nodiscard]] float decode_plain_scalar(const std::byte* bytes,
                                        SafetensorsDtype dtype) {
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

[[nodiscard]] std::uint16_t encode_bf16(float value) noexcept {
    auto bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7F80'0000U) != 0x7F80'0000U) {
        bits += 0x7FFFU + ((bits >> 16U) & 1U);
    }
    return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] bool checked_product(std::uint64_t left, std::uint64_t right,
                                   std::uint64_t& output) noexcept {
    if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left) return false;
    output = left * right;
    return true;
}

void move_errors(ValidationResult& result, std::vector<std::string> errors) {
    for (auto& error : errors) result.errors.push_back(std::move(error));
}

}  // namespace

Dsv4CheckpointReader::~Dsv4CheckpointReader() {
    for (const auto& [name, descriptor] : shard_fds_) {
        static_cast<void>(name);
        if (descriptor >= 0) static_cast<void>(close(descriptor));
    }
}

Dsv4CheckpointOpenResult Dsv4CheckpointReader::open(std::string model_directory,
                                                    bool require_read_only) {
    Dsv4CheckpointOpenResult result;
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
    auto built = build_deepseek_v4_flash_dspark_index_manifest(std::move(index.value));
    if (!built.ok()) {
        result.errors = std::move(built.errors);
        return result;
    }
    Dsv4CheckpointOptions options;
    options.require_read_only = require_read_only;
    auto validated = validate_deepseek_v4_flash_dspark_checkpoint(
        model_directory, std::move(built.manifest), options);
    if (!validated.ok()) {
        result.errors = std::move(validated.errors);
        return result;
    }

    auto reader = std::unique_ptr<Dsv4CheckpointReader>(new Dsv4CheckpointReader());
    reader->model_directory_ = std::move(model_directory);
    reader->manifest_ = std::move(validated.manifest);
    reader->tensors_.reserve(reader->manifest_.tensors.size());
    for (std::size_t index_value = 0U; index_value < reader->manifest_.tensors.size();
         ++index_value) {
        reader->tensors_.emplace(reader->manifest_.tensors[index_value].name, index_value);
    }
    for (const auto& shard : reader->manifest_.shards) {
        const auto path = (std::filesystem::path(reader->model_directory_) / shard).string();
        const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor < 0) {
            result.errors.emplace_back("cannot open DeepSeek checkpoint shard " + path +
                                       ": " + std::strerror(errno));
            return result;
        }
        reader->shard_fds_.emplace(shard, descriptor);
    }
    result.value = std::move(reader);
    return result;
}

const Dsv4ManifestTensor* Dsv4CheckpointReader::find(std::string_view name) const noexcept {
    const auto found = tensors_.find(name);
    return found == tensors_.end() ? nullptr : &manifest_.tensors[found->second];
}

ParseResult<std::vector<std::byte>> Dsv4CheckpointReader::pread_tensor(
    const Dsv4ManifestTensor& tensor, std::uint64_t relative_offset,
    std::uint64_t bytes) const {
    ParseResult<std::vector<std::byte>> result;
    if (relative_offset > tensor.source_bytes || bytes > tensor.source_bytes - relative_offset) {
        result.errors.emplace_back("tensor slice exceeds " + tensor.name);
        return result;
    }
    const auto found = shard_fds_.find(tensor.shard);
    if (found == shard_fds_.end()) {
        result.errors.emplace_back("DeepSeek checkpoint shard is not open for " + tensor.name);
        return result;
    }
    result.value.resize(static_cast<std::size_t>(bytes));
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
            result.errors.emplace_back("cannot read DeepSeek tensor " + tensor.name + ": " +
                                       std::strerror(errno));
            result.value.clear();
            return result;
        }
        if (count == 0) {
            result.errors.emplace_back("unexpected end of DeepSeek shard reading " +
                                       tensor.name);
            result.value.clear();
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
    return result;
}

ParseResult<std::vector<std::byte>> Dsv4CheckpointReader::read(
    std::string_view name, std::uint64_t maximum_bytes) const {
    ParseResult<std::vector<std::byte>> result;
    const auto* tensor = find(name);
    if (tensor == nullptr) {
        result.errors.emplace_back("DeepSeek checkpoint tensor does not exist: " +
                                   std::string(name));
        return result;
    }
    if (tensor->source_bytes > maximum_bytes) {
        result.errors.emplace_back("DeepSeek tensor exceeds the caller byte ceiling: " +
                                   tensor->name);
        return result;
    }
    return pread_tensor(*tensor, 0U, tensor->source_bytes);
}

ParseResult<std::vector<std::byte>> Dsv4CheckpointReader::read_slice(
    std::string_view name, std::uint64_t relative_offset, std::uint64_t bytes) const {
    ParseResult<std::vector<std::byte>> result;
    const auto* tensor = find(name);
    if (tensor == nullptr) {
        result.errors.emplace_back("DeepSeek checkpoint tensor does not exist: " +
                                   std::string(name));
        return result;
    }
    return pread_tensor(*tensor, relative_offset, bytes);
}

ParseResult<std::vector<float>> Dsv4CheckpointReader::read_f32(
    std::string_view name, std::uint64_t maximum_elements) const {
    ParseResult<std::vector<float>> result;
    const auto* tensor = find(name);
    if (tensor == nullptr) {
        result.errors.emplace_back("DeepSeek checkpoint tensor does not exist: " +
                                   std::string(name));
        return result;
    }
    const auto element_bytes = safetensors_dtype_bytes(tensor->source_dtype);
    if ((tensor->source_dtype != SafetensorsDtype::Bf16 &&
         tensor->source_dtype != SafetensorsDtype::F16 &&
         tensor->source_dtype != SafetensorsDtype::F32) ||
        element_bytes == 0U || tensor->source_bytes % element_bytes != 0U ||
        tensor->source_bytes / element_bytes > maximum_elements) {
        result.errors.emplace_back("DeepSeek tensor cannot be decoded under FP32 ceiling: " +
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
        result.value[static_cast<std::size_t>(index)] = decode_plain_scalar(
            encoded.value.data() + index * element_bytes, tensor->source_dtype);
    }
    return result;
}

ParseResult<std::vector<float>> Dsv4CheckpointReader::read_f32_row(
    std::string_view name, std::uint64_t row) const {
    ParseResult<std::vector<float>> result;
    const auto* tensor = find(name);
    if (tensor == nullptr || tensor->source_shape.size() != 2U ||
        row >= (tensor == nullptr ? 0U : tensor->source_shape[0])) {
        result.errors.emplace_back("cannot address DeepSeek tensor row: " + std::string(name));
        return result;
    }
    const auto element_bytes = safetensors_dtype_bytes(tensor->source_dtype);
    if (tensor->source_dtype != SafetensorsDtype::Bf16 &&
        tensor->source_dtype != SafetensorsDtype::F16 &&
        tensor->source_dtype != SafetensorsDtype::F32) {
        result.errors.emplace_back("requested DeepSeek tensor row is not plain floating point");
        return result;
    }
    std::uint64_t row_bytes = 0U;
    if (!checked_product(tensor->source_shape[1], element_bytes, row_bytes)) {
        result.errors.emplace_back("DeepSeek tensor row size overflows");
        return result;
    }
    auto encoded = pread_tensor(*tensor, row * row_bytes, row_bytes);
    if (!encoded.ok()) {
        result.errors = std::move(encoded.errors);
        return result;
    }
    result.value.resize(static_cast<std::size_t>(tensor->source_shape[1]));
    for (std::size_t index = 0U; index < result.value.size(); ++index) {
        result.value[index] = decode_plain_scalar(
            encoded.value.data() + index * element_bytes, tensor->source_dtype);
    }
    return result;
}

ParseResult<std::vector<std::uint32_t>> Dsv4CheckpointReader::read_u32_row_from_i64(
    std::string_view name, std::uint64_t row) const {
    ParseResult<std::vector<std::uint32_t>> result;
    const auto* tensor = find(name);
    if (tensor == nullptr || tensor->source_dtype != SafetensorsDtype::I64 ||
        tensor->source_shape.size() != 2U || row >= tensor->source_shape[0]) {
        result.errors.emplace_back("cannot address DeepSeek I64 routing row: " +
                                   std::string(name));
        return result;
    }
    std::uint64_t row_bytes = 0U;
    if (!checked_product(tensor->source_shape[1], 8U, row_bytes)) {
        result.errors.emplace_back("DeepSeek I64 routing row size overflows");
        return result;
    }
    auto encoded = pread_tensor(*tensor, row * row_bytes, row_bytes);
    if (!encoded.ok()) {
        result.errors = std::move(encoded.errors);
        return result;
    }
    result.value.resize(static_cast<std::size_t>(tensor->source_shape[1]));
    for (std::size_t index = 0U; index < result.value.size(); ++index) {
        std::int64_t value = 0;
        std::memcpy(&value, encoded.value.data() + index * 8U, sizeof(value));
        if (value < 0 || value > std::numeric_limits<std::uint32_t>::max()) {
            result.errors.emplace_back("DeepSeek hash routing expert is out of range");
            result.value.clear();
            return result;
        }
        result.value[index] = static_cast<std::uint32_t>(value);
    }
    return result;
}

Dsv4CheckpointReadStats Dsv4CheckpointReader::stats() const noexcept {
    return {read_calls_.load(std::memory_order_relaxed),
            read_bytes_.load(std::memory_order_relaxed),
            read_nanoseconds_.load(std::memory_order_relaxed)};
}

Dsv4ResidentWeightStore::~Dsv4ResidentWeightStore() {
    if (arena_ != nullptr) {
        static_cast<void>(munmap(arena_, static_cast<std::size_t>(arena_bytes_)));
    }
}

ValidationResult Dsv4ResidentWeightStore::stage(
    const Dsv4CheckpointReader& checkpoint,
    std::uint64_t host_memory_ceiling_bytes, bool include_dspark) {
    ValidationResult result;
    if (complete_ || arena_ != nullptr) {
        result.errors.emplace_back("DeepSeek resident expert store is already staged");
        return result;
    }
    std::vector<const Dsv4ManifestTensor*> tensors;
    for (const auto& tensor : checkpoint.manifest().tensors) {
        if (tensor.dspark && !include_dspark) continue;
        if (tensor.role == Dsv4TensorRole::RoutedExpert ||
            tensor.role == Dsv4TensorRole::Embedding) {
            tensors.push_back(&tensor);
        }
    }
    std::sort(tensors.begin(), tensors.end(), [](const auto* left, const auto* right) {
        if (left->shard != right->shard) return left->shard < right->shard;
        return left->source_offset < right->source_offset;
    });
    for (const auto* tensor : tensors) {
        if (tensor->source_bytes > std::numeric_limits<std::uint64_t>::max() - arena_bytes_) {
            result.errors.emplace_back("DeepSeek resident expert byte count overflows");
            return result;
        }
        arena_bytes_ += tensor->source_bytes;
    }
    if (arena_bytes_ == 0U || arena_bytes_ > host_memory_ceiling_bytes) {
        result.errors.emplace_back(
            "DeepSeek resident weights exceed the admitted host-memory ceiling");
        arena_bytes_ = 0U;
        return result;
    }
    void* allocation = mmap(nullptr, static_cast<std::size_t>(arena_bytes_),
                            PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (allocation == MAP_FAILED) {
        arena_bytes_ = 0U;
        result.errors.emplace_back("cannot allocate DeepSeek resident expert arena: " +
                                   std::string(std::strerror(errno)));
        return result;
    }
    arena_ = static_cast<std::byte*>(allocation);
    extents_.reserve(tensors.size());
    const auto started = std::chrono::steady_clock::now();
    std::uint64_t cursor = 0U;
    for (const auto* tensor : tensors) {
        const auto before = checkpoint.stats();
        auto payload = checkpoint.read(tensor->name, tensor->source_bytes);
        if (!payload.ok()) {
            move_errors(result, std::move(payload.errors));
            return result;
        }
        std::memcpy(arena_ + cursor, payload.value.data(), payload.value.size());
        const auto after = checkpoint.stats();
        if (after.bytes - before.bytes != tensor->source_bytes) {
            result.errors.emplace_back("resident staging read accounting mismatch for " +
                                       tensor->name);
            return result;
        }
        extents_.emplace(tensor->name, Extent{cursor, tensor->source_bytes});
        cursor += tensor->source_bytes;
    }
    static_cast<void>(mprotect(arena_, static_cast<std::size_t>(arena_bytes_), PROT_READ));
    stats_.tensors = tensors.size();
    stats_.bytes = arena_bytes_;
    stats_.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    complete_ = true;
    return result;
}

std::span<const std::byte> Dsv4ResidentWeightStore::find(
    std::string_view name) const noexcept {
    const auto found = extents_.find(name);
    if (found == extents_.end() || arena_ == nullptr) return {};
    return {arena_ + found->second.offset, static_cast<std::size_t>(found->second.bytes)};
}

ValidationResult load_dsv4_cuda_linear(
    const Dsv4CheckpointReader& checkpoint,
    const Dsv4ResidentWeightStore* resident_weights,
    std::string_view base_name, std::uint64_t expected_rows,
    std::uint64_t expected_columns, int device, CudaBackend& backend,
    CudaWeight& output) {
    ValidationResult result;
    const std::string weight_name = std::string(base_name) + ".weight";
    const auto* weight = checkpoint.find(weight_name);
    if (weight == nullptr || weight->source_shape.size() != 2U) {
        result.errors.emplace_back("DeepSeek linear weight is missing: " +
                                   std::string(base_name));
        return result;
    }
    auto expected_source_shape = std::vector<std::uint64_t>{expected_rows, expected_columns};
    if (weight->encoding == Dsv4TensorEncoding::Fp4E2m1Group32) {
        expected_source_shape[1] = (expected_columns + 1U) / 2U;
    }
    if (weight->source_shape != expected_source_shape) {
        result.errors.emplace_back("DeepSeek linear weight shape mismatch: " + weight_name);
        return result;
    }
    std::vector<std::byte> owned_weight;
    std::span<const std::byte> weight_data;
    if (resident_weights != nullptr) weight_data = resident_weights->find(weight_name);
    if (weight_data.empty()) {
        auto loaded = checkpoint.read(weight_name, weight->source_bytes);
        if (!loaded.ok()) {
            move_errors(result, std::move(loaded.errors));
            return result;
        }
        owned_weight = std::move(loaded.value);
        weight_data = owned_weight;
    }

    CudaWeightDescriptor descriptor;
    descriptor.dtype = weight->source_dtype;
    descriptor.rows = expected_rows;
    descriptor.columns = expected_columns;
    std::vector<std::byte> owned_scale;
    std::span<const std::byte> scale_data;
    if (weight->encoding == Dsv4TensorEncoding::Plain) {
        descriptor.encoding = CudaWeightEncoding::Plain;
    } else {
        const std::string scale_name = std::string(base_name) + ".scale";
        const auto* scale = checkpoint.find(scale_name);
        if (scale == nullptr || scale->encoding != weight->encoding) {
            result.errors.emplace_back("DeepSeek linear scale is missing: " + scale_name);
            return result;
        }
        if (resident_weights != nullptr) scale_data = resident_weights->find(scale_name);
        if (scale_data.empty()) {
            auto loaded = checkpoint.read(scale_name, scale->source_bytes);
            if (!loaded.ok()) {
                move_errors(result, std::move(loaded.errors));
                return result;
            }
            owned_scale = std::move(loaded.value);
            scale_data = owned_scale;
        }
        if (weight->encoding == Dsv4TensorEncoding::Fp4E2m1Group32) {
            descriptor.encoding = CudaWeightEncoding::Fp4E2m1Group32;
            descriptor.packed_columns = (expected_columns + 1U) / 2U;
            descriptor.scale_columns = (expected_columns + 31U) / 32U;
            descriptor.group_size = 32U;
        } else {
            if (base_name.ends_with(".attn.wo_a")) {
                const auto scale_columns = (expected_columns + 127U) / 128U;
                const auto scale_rows = (expected_rows + 127U) / 128U;
                if (weight_data.size() != expected_rows * expected_columns ||
                    scale_data.size() != scale_rows * scale_columns) {
                    result.errors.emplace_back(
                        "DeepSeek wo_a FP8 source layout is incompatible");
                    return result;
                }
                std::vector<std::byte> bf16(
                    static_cast<std::size_t>(expected_rows * expected_columns * 2U));
                for (std::uint64_t row = 0U; row < expected_rows; ++row) {
                    for (std::uint64_t column = 0U; column < expected_columns;
                         ++column) {
                        const auto encoded = std::to_integer<std::uint8_t>(
                            weight_data[static_cast<std::size_t>(
                                row * expected_columns + column)]);
                        const auto scale_encoded = std::to_integer<std::uint8_t>(
                            scale_data[static_cast<std::size_t>(
                                (row / 128U) * scale_columns + column / 128U)]);
                        const auto converted = encode_bf16(
                            dsv4_fp8_e4m3_f32(encoded) *
                            dsv4_fp8_e8m0_scale_f32(scale_encoded));
                        std::memcpy(bf16.data() + static_cast<std::size_t>(
                                        (row * expected_columns + column) * 2U),
                                    &converted, sizeof(converted));
                    }
                }
                descriptor.encoding = CudaWeightEncoding::Plain;
                descriptor.dtype = SafetensorsDtype::Bf16;
                return backend.upload(device, descriptor, bf16, {}, output);
            }
            descriptor.encoding = CudaWeightEncoding::Fp8E4m3Block128;
            descriptor.packed_columns = expected_columns;
            descriptor.scale_columns = (expected_columns + 127U) / 128U;
            descriptor.group_size = 128U;
        }
    }
    return backend.upload(device, descriptor, weight_data, scale_data, output);
}

}  // namespace strata
