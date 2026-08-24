#include "strata/laguna_checkpoint.hpp"

#include "../common/checkpoint_common.hpp"

#include "strata/model.hpp"
#include "strata/model_adapter.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <unordered_map>

namespace strata {
namespace {

constexpr auto& kContract = kLagunaExecutionContract;
constexpr std::uint64_t kMaximumIndexBytes = 64ULL << 20U;
constexpr std::size_t kMaximumOpenErrors = 32U;

void append(std::vector<std::string>& destination,
            std::vector<std::string> source) {
    for (auto& error : source) {
        if (destination.size() < kMaximumOpenErrors) {
            destination.push_back(std::move(error));
        }
    }
}

std::string layer_prefix(std::uint32_t layer) {
    return "model.layers." + std::to_string(layer) + ".";
}

}  // namespace

LagunaCheckpointReader::~LagunaCheckpointReader() { release_mapped_views(); }

LagunaCheckpointOpenResult LagunaCheckpointReader::open(
    std::string model_directory) {
    LagunaCheckpointOpenResult result;
    const auto nvfp4_spec = laguna_s21_nvfp4_spec();
    const auto mxfp4_spec = laguna_s21_mxfp4_spec();
    auto index = load_safetensors_index(model_directory, kMaximumIndexBytes);
    if (!index.ok()) {
        result.errors = std::move(index.errors);
        return result;
    }
    const auto matches = [&](const ModelSpec& spec) {
        return index.value.total_size == spec.source.indexed_tensor_bytes &&
               index.value.entries.size() == spec.source.tensor_count &&
               index.value.shards.size() == spec.source.main_shards;
    };
    const bool nvfp4 = matches(nvfp4_spec);
    const bool mxfp4 = matches(mxfp4_spec);
    if (!nvfp4 && !mxfp4) {
        result.errors.emplace_back(
            "Laguna checkpoint index extent is neither pinned S 2.1 NVFP4 nor MXFP4");
        return result;
    }
    const auto& spec = nvfp4 ? nvfp4_spec : mxfp4_spec;
    for (const auto& shard : index.value.shards) {
        char expected[48]{};
        bool matched = false;
        for (std::uint32_t ordinal = 1U; ordinal <= spec.source.main_shards;
             ++ordinal) {
            std::snprintf(expected, sizeof(expected),
                          "model-%05u-of-%05u.safetensors", ordinal,
                          spec.source.main_shards);
            if (shard == expected) {
                matched = true;
                break;
            }
        }
        if (!matched) {
            result.errors.emplace_back("unexpected Laguna shard name " + shard);
        }
    }
    if (!result.errors.empty()) return result;

    auto reader = std::unique_ptr<LagunaCheckpointReader>(
        new LagunaCheckpointReader());
    reader->model_directory_ = std::move(model_directory);
    reader->format_ = nvfp4 ? LagunaCheckpointFormat::Nvfp4Mixed
                            : LagunaCheckpointFormat::Mxfp4Group32;
    std::unordered_map<std::string, std::string> indexed;
    indexed.reserve(index.value.entries.size());
    for (const auto& entry : index.value.entries) {
        indexed.emplace(entry.name, entry.shard);
    }

    reader->tensors_.reserve(index.value.entries.size());
    for (const auto& shard_name : index.value.shards) {
        const auto path = (std::filesystem::path(reader->model_directory_) /
                           shard_name).string();
        auto shard = load_safetensors_shard(path);
        if (!shard.ok()) {
            append(result.errors, std::move(shard.errors));
            continue;
        }
        reader->shard_file_bytes_ += shard.value.file_size;
        for (auto& tensor : shard.value.tensors) {
            const auto found = indexed.find(tensor.name);
            if (found == indexed.end() || found->second != shard_name) {
                if (result.errors.size() < kMaximumOpenErrors) {
                    result.errors.emplace_back(
                        "unindexed or misplaced Laguna tensor " + tensor.name);
                }
                continue;
            }
            reader->tensors_.push_back({std::move(tensor.name), shard_name,
                                        tensor.dtype, std::move(tensor.shape),
                                        tensor.absolute_begin, tensor.bytes()});
        }
    }
    if (!result.errors.empty()) return result;
    if (reader->shard_file_bytes_ != spec.source.shard_file_bytes ||
        reader->tensors_.size() != spec.source.tensor_count) {
        result.errors.emplace_back(
            "Laguna shard extent does not match the pinned checkpoint");
        return result;
    }
    reader->by_name_.reserve(reader->tensors_.size());
    for (std::size_t i = 0U; i < reader->tensors_.size(); ++i) {
        if (!reader->by_name_.emplace(reader->tensors_[i].name, i).second) {
            result.errors.emplace_back("duplicate Laguna tensor " +
                                       reader->tensors_[i].name);
            return result;
        }
    }
    auto opened = reader->shards_.open(reader->model_directory_,
                                       index.value.shards, "Laguna");
    if (!opened.ok()) {
        result.errors = std::move(opened.errors);
        return result;
    }

    const auto require_plain = [&](std::string_view name,
                                   std::initializer_list<std::uint64_t> shape) {
        const auto* tensor = reader->find(name);
        if (tensor == nullptr || tensor->dtype != SafetensorsDtype::Bf16 ||
            tensor->shape != std::vector<std::uint64_t>(shape)) {
            if (result.errors.size() < kMaximumOpenErrors) {
                result.errors.emplace_back(
                    "Laguna plain tensor shape mismatch: " + std::string(name));
            }
        }
    };
    const auto require_linear = [&](const std::string& base, std::uint64_t rows,
                                    std::uint64_t columns, bool quantized) {
        if (!quantized) {
            require_plain(base + ".weight", {rows, columns});
            return;
        }
        const auto* packed = reader->find(base + ".weight_packed");
        const auto* scale = reader->find(base + ".weight_scale");
        const auto* global = reader->find(base + ".weight_global_scale");
        const auto* activation = reader->find(base + ".input_global_scale");
        const auto packed_columns = columns / 2U;
        const auto nvfp4_scale_columns =
            (columns + kContract.nvfp4_group_size - 1U) /
            kContract.nvfp4_group_size;
        const bool valid_nvfp4 =
            reader->format_ == LagunaCheckpointFormat::Nvfp4Mixed &&
            packed != nullptr && scale != nullptr && global != nullptr &&
            activation != nullptr && columns % 2U == 0U &&
            packed->dtype == SafetensorsDtype::U8 &&
            scale->dtype == SafetensorsDtype::F8E4M3 &&
            global->dtype == SafetensorsDtype::F32 &&
            activation->dtype == SafetensorsDtype::F32 &&
            packed->shape == std::vector<std::uint64_t>{rows, packed_columns} &&
            scale->shape ==
                std::vector<std::uint64_t>{rows, nvfp4_scale_columns} &&
            global->shape.empty() && activation->shape.empty();
        const bool valid_mxfp4 =
            reader->format_ == LagunaCheckpointFormat::Mxfp4Group32 &&
            packed != nullptr && scale != nullptr && global == nullptr &&
            activation == nullptr && columns % 32U == 0U &&
            packed->dtype == SafetensorsDtype::U8 &&
            scale->dtype == SafetensorsDtype::U8 &&
            packed->shape == std::vector<std::uint64_t>{rows, packed_columns} &&
            scale->shape == std::vector<std::uint64_t>{rows, columns / 32U};
        if (!valid_nvfp4 && !valid_mxfp4 &&
            result.errors.size() < kMaximumOpenErrors) {
            result.errors.emplace_back(
                "Laguna compressed tensor layout mismatch: " + base);
        }
    };

    const auto hidden = static_cast<std::uint64_t>(kContract.hidden_size);
    const auto head_dim = static_cast<std::uint64_t>(kContract.head_dim);
    const auto kv_columns =
        static_cast<std::uint64_t>(kContract.key_value_heads) * head_dim;
    require_plain("model.embed_tokens.weight", {kContract.vocabulary_size, hidden});
    require_plain("model.norm.weight", {hidden});
    require_plain("lm_head.weight", {kContract.vocabulary_size, hidden});
    for (std::uint32_t layer = 0U; layer < kContract.layer_count; ++layer) {
        const auto prefix = layer_prefix(layer);
        const auto attention = prefix + "self_attn.";
        const auto mlp = prefix + "mlp.";
        const auto heads = static_cast<std::uint64_t>(laguna_attention_heads(layer));
        require_plain(prefix + "input_layernorm.weight", {hidden});
        require_plain(prefix + "post_attention_layernorm.weight", {hidden});
        require_plain(attention + "q_norm.weight", {head_dim});
        require_plain(attention + "k_norm.weight", {head_dim});
        require_linear(attention + "q_proj", heads * head_dim, hidden, false);
        require_linear(attention + "k_proj", kv_columns, hidden, false);
        require_linear(attention + "v_proj", kv_columns, hidden, false);
        require_linear(attention + "o_proj", hidden, heads * head_dim, false);
        require_linear(attention + "g_proj", heads, hidden, false);
        if (!laguna_sparse_layer(layer)) {
            require_linear(mlp + "gate_proj", kContract.dense_intermediate_size,
                           hidden, false);
            require_linear(mlp + "up_proj", kContract.dense_intermediate_size,
                           hidden, false);
            require_linear(mlp + "down_proj", hidden,
                           kContract.dense_intermediate_size, false);
            continue;
        }
        require_linear(mlp + "gate", kContract.routed_experts, hidden, false);
        require_plain(mlp + "experts.e_score_correction_bias",
                      {kContract.routed_experts});
        require_linear(mlp + "shared_expert.gate_proj",
                       kContract.shared_expert_intermediate_size, hidden, false);
        require_linear(mlp + "shared_expert.up_proj",
                       kContract.shared_expert_intermediate_size, hidden, false);
        require_linear(mlp + "shared_expert.down_proj", hidden,
                       kContract.shared_expert_intermediate_size, false);
        const bool quantized = mxfp4 || laguna_quantized_expert_layer(layer);
        for (std::uint32_t expert = 0U; expert < kContract.routed_experts;
             ++expert) {
            const auto base = mlp + "experts." + std::to_string(expert) + ".";
            require_linear(base + "gate_proj", kContract.expert_intermediate_size,
                           hidden, quantized);
            require_linear(base + "up_proj", kContract.expert_intermediate_size,
                           hidden, quantized);
            require_linear(base + "down_proj", hidden,
                           kContract.expert_intermediate_size, quantized);
        }
        if (result.errors.size() >= kMaximumOpenErrors) break;
    }
    if (!result.errors.empty()) return result;
    result.value = std::move(reader);
    return result;
}

const LagunaTensor* LagunaCheckpointReader::find(
    std::string_view name) const noexcept {
    const auto found = by_name_.find(name);
    return found == by_name_.end() ? nullptr : &tensors_[found->second];
}

ParseResult<LagunaLinear> LagunaCheckpointReader::linear(
    std::string_view base_name, std::uint64_t rows,
    std::uint64_t columns) const {
    ParseResult<LagunaLinear> result;
    const std::string base(base_name);
    result.value.rows = rows;
    result.value.columns = columns;
    if (const auto* plain = find(base + ".weight"); plain != nullptr) {
        if (plain->dtype != SafetensorsDtype::Bf16 ||
            plain->shape != std::vector<std::uint64_t>{rows, columns}) {
            result.errors.emplace_back("Laguna plain linear shape mismatch: " + base);
            return result;
        }
        result.value.encoding = LagunaTensorEncoding::Plain;
        result.value.weight = plain;
        return result;
    }
    const auto* packed = find(base + ".weight_packed");
    const auto* scale = find(base + ".weight_scale");
    const auto* global = find(base + ".weight_global_scale");
    const auto packed_columns = columns / 2U;
    const auto nvfp4_scale_columns =
        (columns + kContract.nvfp4_group_size - 1U) / kContract.nvfp4_group_size;
    const bool valid_nvfp4 =
        format_ == LagunaCheckpointFormat::Nvfp4Mixed && packed != nullptr &&
        scale != nullptr && global != nullptr && columns % 2U == 0U &&
        packed->dtype == SafetensorsDtype::U8 &&
        scale->dtype == SafetensorsDtype::F8E4M3 &&
        global->dtype == SafetensorsDtype::F32 &&
        packed->shape == std::vector<std::uint64_t>{rows, packed_columns} &&
        scale->shape ==
            std::vector<std::uint64_t>{rows, nvfp4_scale_columns};
    const bool valid_mxfp4 =
        format_ == LagunaCheckpointFormat::Mxfp4Group32 && packed != nullptr &&
        scale != nullptr && global == nullptr && columns % 32U == 0U &&
        packed->dtype == SafetensorsDtype::U8 &&
        scale->dtype == SafetensorsDtype::U8 &&
        packed->shape == std::vector<std::uint64_t>{rows, packed_columns} &&
        scale->shape == std::vector<std::uint64_t>{rows, columns / 32U};
    if (!valid_nvfp4 && !valid_mxfp4) {
        result.errors.emplace_back(
            "Laguna compressed linear is missing or mismatched: " + base);
        return result;
    }
    result.value.encoding = valid_mxfp4
        ? LagunaTensorEncoding::Mxfp4Group32
        : LagunaTensorEncoding::Nvfp4Group16;
    result.value.packed = packed;
    result.value.scale = scale;
    result.value.global_scale = global;
    return result;
}

ParseResult<std::vector<std::byte>> LagunaCheckpointReader::read(
    std::string_view name, std::uint64_t maximum_bytes) const {
    ParseResult<std::vector<std::byte>> result;
    const auto* tensor = find(name);
    if (tensor == nullptr) {
        result.errors.emplace_back("Laguna tensor does not exist: " +
                                   std::string(name));
        return result;
    }
    if (tensor->bytes > maximum_bytes ||
        tensor->bytes > std::numeric_limits<std::size_t>::max()) {
        result.errors.emplace_back("Laguna tensor exceeds the read ceiling: " +
                                   std::string(name));
        return result;
    }
    result.value.resize(static_cast<std::size_t>(tensor->bytes));
    const auto started = std::chrono::steady_clock::now();
    auto status = shards_.read(tensor->shard, tensor->offset, result.value,
                               tensor->name);
    read_nanoseconds_.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count()),
        std::memory_order_relaxed);
    if (!status.ok()) {
        result.errors = std::move(status.errors);
        result.value.clear();
        return result;
    }
    read_calls_.fetch_add(1U, std::memory_order_relaxed);
    read_bytes_.fetch_add(tensor->bytes, std::memory_order_relaxed);
    return result;
}

ParseResult<std::vector<float>> LagunaCheckpointReader::read_f32(
    std::string_view name, std::uint64_t maximum_elements) const {
    ParseResult<std::vector<float>> result;
    const auto* tensor = find(name);
    const auto width = tensor == nullptr ? 0U
                                         : safetensors_dtype_bytes(tensor->dtype);
    if (tensor == nullptr || (tensor->dtype != SafetensorsDtype::Bf16 &&
                              tensor->dtype != SafetensorsDtype::F16 &&
                              tensor->dtype != SafetensorsDtype::F32) ||
        width == 0U || tensor->bytes / width > maximum_elements) {
        result.errors.emplace_back(
            "Laguna tensor cannot be decoded as bounded FP32: " + std::string(name));
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

ParseResult<std::vector<float>> LagunaCheckpointReader::read_f32_row(
    std::string_view name, std::uint64_t row) const {
    ParseResult<std::vector<float>> result;
    const auto* tensor = find(name);
    if (tensor == nullptr || tensor->shape.size() != 2U ||
        row >= tensor->shape[0]) {
        result.errors.emplace_back("Laguna tensor row is out of range: " +
                                   std::string(name));
        return result;
    }
    const auto width = safetensors_dtype_bytes(tensor->dtype);
    if ((tensor->dtype != SafetensorsDtype::Bf16 &&
         tensor->dtype != SafetensorsDtype::F16 &&
         tensor->dtype != SafetensorsDtype::F32) || width == 0U) {
        result.errors.emplace_back("Laguna tensor row is not floating point");
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
    read_calls_.fetch_add(1U, std::memory_order_relaxed);
    read_bytes_.fetch_add(row_bytes, std::memory_order_relaxed);
    result.value.resize(static_cast<std::size_t>(tensor->shape[1]));
    for (std::size_t i = 0U; i < result.value.size(); ++i) {
        result.value[i] = detail::decode_plain_scalar(
            encoded.data() + i * width, tensor->dtype);
    }
    return result;
}

ParseResult<std::span<const std::byte>> LagunaCheckpointReader::view(
    std::string_view name) const {
    ParseResult<std::span<const std::byte>> result;
    const auto* tensor = find(name);
    if (tensor == nullptr) {
        result.errors.emplace_back("Laguna tensor does not exist: " +
                                   std::string(name));
        return result;
    }
    std::scoped_lock lock(mapping_mutex_);
    auto mapping = mappings_.find(tensor->shard);
    if (mapping == mappings_.end()) {
        const int descriptor = shards_.descriptor(tensor->shard);
        if (descriptor < 0) {
            result.errors.emplace_back("Laguna shard is not open for " +
                                       tensor->name);
            return result;
        }
        struct stat status {};
        if (fstat(descriptor, &status) != 0 || status.st_size <= 0) {
            result.errors.emplace_back("cannot size Laguna shard " +
                                       tensor->shard + ": " +
                                       std::strerror(errno));
            return result;
        }
        const auto bytes = static_cast<std::uint64_t>(status.st_size);
        void* address = mmap(nullptr, static_cast<std::size_t>(bytes), PROT_READ,
                             MAP_SHARED, descriptor, 0);
        if (address == MAP_FAILED) {
            result.errors.emplace_back("cannot map Laguna shard " + tensor->shard +
                                       ": " + std::strerror(errno));
            return result;
        }
        mapping = mappings_.emplace(
            tensor->shard,
            ShardMapping{static_cast<std::byte*>(address), bytes}).first;
    }
    if (tensor->offset > mapping->second.bytes ||
        tensor->bytes > mapping->second.bytes - tensor->offset) {
        result.errors.emplace_back("mapped tensor exceeds its Laguna shard: " +
                                   tensor->name);
        return result;
    }
    result.value = std::span<const std::byte>(
        mapping->second.address + tensor->offset,
        static_cast<std::size_t>(tensor->bytes));
    read_calls_.fetch_add(1U, std::memory_order_relaxed);
    read_bytes_.fetch_add(tensor->bytes, std::memory_order_relaxed);
    return result;
}

void LagunaCheckpointReader::release_mapped_views() const noexcept {
    std::scoped_lock lock(mapping_mutex_);
    for (const auto& [shard, mapping] : mappings_) {
        static_cast<void>(shard);
        if (mapping.address != nullptr) {
            static_cast<void>(munmap(mapping.address,
                                     static_cast<std::size_t>(mapping.bytes)));
        }
    }
    mappings_.clear();
}

ParseResult<LagunaNvfp4MatrixView> LagunaCheckpointReader::nvfp4_view(
    const LagunaLinear& module) const {
    ParseResult<LagunaNvfp4MatrixView> result;
    if (module.encoding != LagunaTensorEncoding::Nvfp4Group16 ||
        module.packed == nullptr || module.scale == nullptr ||
        module.global_scale == nullptr) {
        result.errors.emplace_back("linear is not an NVFP4 module");
        return result;
    }
    auto packed = view(module.packed->name);
    if (!packed.ok()) {
        result.errors = std::move(packed.errors);
        return result;
    }
    auto scale = view(module.scale->name);
    if (!scale.ok()) {
        result.errors = std::move(scale.errors);
        return result;
    }
    auto global = read_f32(module.global_scale->name, 1U);
    if (!global.ok()) {
        result.errors = std::move(global.errors);
        return result;
    }
    if (global.value.size() != 1U) {
        result.errors.emplace_back("NVFP4 global scale is not a scalar: " +
                                   module.global_scale->name);
        return result;
    }
    result.value.packed = packed.value;
    result.value.scales = scale.value;
    result.value.global_scale = global.value[0];
    result.value.rows = module.rows;
    result.value.columns = module.columns;
    result.value.packed_columns = module.columns / 2U;
    result.value.scale_columns =
        (module.columns + kContract.nvfp4_group_size - 1U) /
        kContract.nvfp4_group_size;
    result.value.group_size = kContract.nvfp4_group_size;
    return result;
}

CheckpointReadStats LagunaCheckpointReader::stats() const noexcept {
    const auto nanoseconds = read_nanoseconds_.load(std::memory_order_relaxed);
    return {read_calls_.load(std::memory_order_relaxed),
            read_bytes_.load(std::memory_order_relaxed), nanoseconds,
            nanoseconds};
}

ValidationResult load_laguna_cuda_linear(
    const LagunaCheckpointReader& checkpoint, const LagunaLinear& module,
    int device, CudaBackend& backend, CudaWeight& output) {
    ValidationResult result;
    // The upload source is the shard mapping itself. `read` would copy every
    // byte into a fresh heap vector first, which measured 0.30 ms per routed
    // expert projection of pure duplication -- 1.9x on the whole staging path
    // once the per-weight cudaMalloc goes too.
    //
    // Deferred, not synchronous: the copy lands on the upload stream and the
    // consumer orders itself behind it with an event, so staging an expert no
    // longer blocks the host. The source is the reader's own mapping, which
    // outlives every batch, so the transfer cannot outlive its bytes -- that
    // precondition is what makes deferring legal here and not in a runtime
    // that stages through a temporary.
    if (module.encoding == LagunaTensorEncoding::Plain) {
        if (module.weight == nullptr) {
            result.errors.emplace_back("Laguna plain linear has no weight tensor");
            return result;
        }
        auto data = checkpoint.view(module.weight->name);
        if (!data.ok()) {
            result.errors = std::move(data.errors);
            return result;
        }
        if (data.value.size() != module.weight->bytes) {
            result.errors.emplace_back(
                "Laguna plain linear mapping size mismatch: " +
                module.weight->name);
            return result;
        }
        CudaWeightDescriptor descriptor;
        descriptor.encoding = CudaWeightEncoding::Plain;
        descriptor.dtype = module.weight->dtype;
        descriptor.rows = module.rows;
        descriptor.columns = module.columns;
        return backend.upload(device, descriptor, data.value, {}, output,
                              CudaBackend::UploadCompletion::Deferred);
    }
    if (module.encoding == LagunaTensorEncoding::Mxfp4Group32) {
        if (module.packed == nullptr || module.scale == nullptr ||
            module.global_scale != nullptr) {
            result.errors.emplace_back("Laguna MXFP4 linear is incomplete");
            return result;
        }
        auto packed = checkpoint.view(module.packed->name);
        auto scale = checkpoint.view(module.scale->name);
        if (!packed.ok()) append(result.errors, std::move(packed.errors));
        if (!scale.ok()) append(result.errors, std::move(scale.errors));
        if (!result.ok()) return result;
        if (packed.value.size() != module.packed->bytes ||
            scale.value.size() != module.scale->bytes) {
            result.errors.emplace_back("Laguna MXFP4 mapping size mismatch: " +
                                       module.packed->name);
            return result;
        }
        CudaWeightDescriptor descriptor;
        descriptor.encoding = CudaWeightEncoding::Fp4E2m1Group32;
        // The backend names the byte-level E2M1 stream I8 even when the
        // Safetensors container declares those same bytes U8.
        descriptor.dtype = SafetensorsDtype::I8;
        descriptor.rows = module.rows;
        descriptor.columns = module.columns;
        descriptor.packed_columns = module.columns / 2U;
        descriptor.scale_columns = module.columns / 32U;
        descriptor.group_size = 32U;
        // Ask for m16n8k16 fragment order. Both consumers of an MXFP4 linear
        // take a register-fed route when the weight carries it -- the fused MoE
        // batch in enqueue_moe and the generic matmul -- and a shape the layout
        // cannot express is left canonical by the backend rather than half
        // converted. Honoured only while the register-fed switch is on.
        return backend.upload(device, descriptor, packed.value, scale.value,
                              output,
                              CudaBackend::UploadCompletion::Deferred,
                              CudaBackend::FragmentLayout::Prepack);
    }
    if (module.packed == nullptr || module.scale == nullptr ||
        module.global_scale == nullptr) {
        result.errors.emplace_back("Laguna NVFP4 linear is incomplete");
        return result;
    }
    auto packed = checkpoint.view(module.packed->name);
    auto scale = checkpoint.view(module.scale->name);
    auto global = checkpoint.read_f32(module.global_scale->name, 1U);
    if (!packed.ok()) append(result.errors, std::move(packed.errors));
    if (!scale.ok()) append(result.errors, std::move(scale.errors));
    if (!global.ok()) append(result.errors, std::move(global.errors));
    if (!result.ok()) return result;
    if (global.value.size() != 1U || !(global.value[0] > 0.0F)) {
        result.errors.emplace_back("Laguna NVFP4 global scale is not positive: " +
                                   module.global_scale->name);
        return result;
    }
    if (packed.value.size() != module.packed->bytes ||
        scale.value.size() != module.scale->bytes) {
        result.errors.emplace_back("Laguna NVFP4 mapping size mismatch: " +
                                   module.packed->name);
        return result;
    }
    CudaWeightDescriptor descriptor;
    descriptor.encoding = CudaWeightEncoding::Nvfp4Group16;
    descriptor.dtype = SafetensorsDtype::U8;
    descriptor.rows = module.rows;
    descriptor.columns = module.columns;
    descriptor.packed_columns = module.columns / 2U;
    descriptor.scale_columns =
        (module.columns + kContract.nvfp4_group_size - 1U) /
        kContract.nvfp4_group_size;
    descriptor.group_size = kContract.nvfp4_group_size;
    descriptor.global_scale = global.value[0];
    return backend.upload(device, descriptor, packed.value, scale.value, output,
                          CudaBackend::UploadCompletion::Deferred);
}

}  // namespace strata
