#include "strata/inkling_checkpoint.hpp"

#include "checkpoint_common.hpp"

#include "strata/model.hpp"
#include "strata/model_adapter.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <unordered_map>

namespace strata {
namespace {

constexpr auto& kContract = kInklingExecutionContract;
constexpr std::uint64_t kMaximumIndexBytes = 64ULL << 20U;
constexpr std::size_t kMaximumOpenErrors = 32U;
constexpr std::string_view kMtpShard = "mtp.safetensors";

void append(std::vector<std::string>& destination,
            std::vector<std::string> source) {
    for (auto& error : source) {
        if (destination.size() < kMaximumOpenErrors) {
            destination.push_back(std::move(error));
        }
    }
}

}  // namespace

std::string inkling_layer_prefix(std::uint32_t layer) {
    return "model.llm.layers." + std::to_string(layer) + ".";
}

std::string inkling_mtp_prefix(std::uint32_t depth) {
    return "model.mtp.layers." + std::to_string(depth) + ".";
}

InklingCheckpointReader::~InklingCheckpointReader() { release_mapped_views(); }

InklingCheckpointOpenResult InklingCheckpointReader::open(
    std::string model_directory) {
    InklingCheckpointOpenResult result;
    const auto spec = inkling_small_nvfp4_spec();
    const auto index_path = (std::filesystem::path(model_directory) /
                             "model.safetensors.index.json").string();
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
    // The MTP heads ride in the same index as the backbone, so the shard count
    // is the main shards plus the single MTP shard.
    const auto expected_shards =
        static_cast<std::size_t>(spec.source.main_shards) + spec.source.mtp_shards;
    if (index.value.total_size != spec.source.indexed_tensor_bytes ||
        index.value.entries.size() != spec.source.tensor_count ||
        index.value.shards.size() != expected_shards) {
        result.errors.emplace_back(
            "Inkling checkpoint index extent is not pinned Small-NVFP4");
        return result;
    }
    for (const auto& shard : index.value.shards) {
        if (shard == kMtpShard) continue;
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
            result.errors.emplace_back("unexpected Inkling shard name " + shard);
        }
    }
    if (!result.errors.empty()) return result;

    auto reader = std::unique_ptr<InklingCheckpointReader>(
        new InklingCheckpointReader());
    reader->model_directory_ = std::move(model_directory);
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
                        "unindexed or misplaced Inkling tensor " + tensor.name);
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
            "Inkling shard extent does not match the pinned checkpoint");
        return result;
    }
    reader->by_name_.reserve(reader->tensors_.size());
    for (std::size_t i = 0U; i < reader->tensors_.size(); ++i) {
        if (!reader->by_name_.emplace(reader->tensors_[i].name, i).second) {
            result.errors.emplace_back("duplicate Inkling tensor " +
                                       reader->tensors_[i].name);
            return result;
        }
    }
    auto opened = reader->shards_.open(reader->model_directory_,
                                       index.value.shards, "Inkling");
    if (!opened.ok()) {
        result.errors = std::move(opened.errors);
        return result;
    }

    const auto require_dtype = [&](std::string_view name, SafetensorsDtype dtype,
                                   std::initializer_list<std::uint64_t> shape) {
        const auto* tensor = reader->find(name);
        if (tensor == nullptr || tensor->dtype != dtype ||
            tensor->shape != std::vector<std::uint64_t>(shape)) {
            if (result.errors.size() < kMaximumOpenErrors) {
                result.errors.emplace_back(
                    "Inkling tensor shape mismatch: " + std::string(name));
            }
        }
    };
    const auto require_plain = [&](std::string_view name,
                                   std::initializer_list<std::uint64_t> shape) {
        require_dtype(name, SafetensorsDtype::Bf16, shape);
    };
    const auto require_experts = [&](const std::string& base, bool quantized,
                                     std::uint64_t rows, std::uint64_t columns) {
        const auto experts = static_cast<std::uint64_t>(kContract.routed_experts);
        if (!quantized) {
            require_plain(base, {experts, rows, columns});
            return;
        }
        const auto* packed = reader->find(base);
        const auto* scale = reader->find(base + ".scale");
        const auto* global = reader->find(base + ".scale2");
        const auto packed_columns = columns / 2U;
        const auto scale_columns =
            (columns + kContract.nvfp4_group_size - 1U) / kContract.nvfp4_group_size;
        const bool valid =
            packed != nullptr && scale != nullptr && global != nullptr &&
            columns % 2U == 0U && packed->dtype == SafetensorsDtype::U8 &&
            scale->dtype == SafetensorsDtype::F8E4M3 &&
            global->dtype == SafetensorsDtype::F32 &&
            packed->shape ==
                std::vector<std::uint64_t>{experts, rows, packed_columns} &&
            scale->shape ==
                std::vector<std::uint64_t>{experts, rows, scale_columns} &&
            global->shape == std::vector<std::uint64_t>{experts};
        if (!valid && result.errors.size() < kMaximumOpenErrors) {
            result.errors.emplace_back(
                "Inkling NVFP4 expert stack layout mismatch: " + base);
        }
    };

    const auto hidden = static_cast<std::uint64_t>(kContract.hidden_size);
    const auto head_dim = static_cast<std::uint64_t>(kContract.head_dim);
    const auto query_columns =
        static_cast<std::uint64_t>(kContract.attention_heads) * head_dim;
    const auto kv_columns =
        static_cast<std::uint64_t>(kContract.key_value_heads) * head_dim;
    const auto relative_columns =
        static_cast<std::uint64_t>(kContract.attention_heads) *
        kContract.relative_dim;
    const auto kernel = static_cast<std::uint64_t>(kContract.short_conv_kernel);
    const auto dense_gate_up =
        2ULL * static_cast<std::uint64_t>(kContract.dense_intermediate_size);
    const auto expert_gate_up =
        2ULL * static_cast<std::uint64_t>(kContract.expert_intermediate_size);
    const auto padded_vocabulary =
        static_cast<std::uint64_t>(kContract.padded_vocabulary_size);

    require_plain("model.llm.embed.weight", {padded_vocabulary, hidden});
    require_plain("model.llm.embed_norm.weight", {hidden});
    require_plain("model.llm.norm.weight", {hidden});
    require_plain("model.llm.unembed.weight", {padded_vocabulary, hidden});

    // One attention block, shared by the backbone layers and the MTP depths.
    const auto require_attention = [&](const std::string& prefix,
                                       std::uint32_t relative_extent) {
        const auto attention = prefix + "attn.";
        require_plain(prefix + "attn_norm.weight", {hidden});
        require_plain(prefix + "mlp_norm.weight", {hidden});
        require_plain(prefix + "attn_sconv.weight", {hidden, 1U, kernel});
        require_plain(prefix + "mlp_sconv.weight", {hidden, 1U, kernel});
        require_plain(attention + "wq_du.weight", {query_columns, hidden});
        require_plain(attention + "wk_dv.weight", {kv_columns, hidden});
        require_plain(attention + "wv_dv.weight", {kv_columns, hidden});
        require_plain(attention + "wo_ud.weight", {hidden, query_columns});
        require_plain(attention + "wr_du.weight", {relative_columns, hidden});
        require_plain(attention + "q_norm.weight", {head_dim});
        require_plain(attention + "k_norm.weight", {head_dim});
        require_plain(attention + "k_sconv.weight", {kv_columns, 1U, kernel});
        require_plain(attention + "v_sconv.weight", {kv_columns, 1U, kernel});
        // Position lives here and nowhere else: local layers span exactly
        // their window, global layers the full relative extent.
        require_plain(attention + "rel_logits_proj.proj",
                      {kContract.relative_dim, relative_extent});
    };

    for (std::uint32_t layer = 0U; layer < kContract.layer_count; ++layer) {
        const auto prefix = inkling_layer_prefix(layer);
        const auto mlp = prefix + "mlp.";
        require_attention(prefix, inkling_relative_extent(layer));
        if (!inkling_sparse_layer(layer)) {
            require_plain(mlp + "w13_dn.weight", {dense_gate_up, hidden});
            require_plain(mlp + "w2_md.weight",
                          {hidden, kContract.dense_intermediate_size});
            require_plain(mlp + "global_scale", {1U});
            continue;
        }
        // The gate scores the routed experts and both sinks, so it has
        // routed + shared rows; the correction bias covers only the routed
        // range because the sinks never compete for selection.
        require_plain(mlp + "gate.weight",
                      {kContract.routed_experts + kContract.shared_experts, hidden});
        require_dtype(mlp + "gate.bias", SafetensorsDtype::F32,
                      {kContract.routed_experts});
        require_dtype(mlp + "gate.global_scale", SafetensorsDtype::F32, {1U});
        require_plain(mlp + "shared_experts.shared_w13_weight",
                      {kContract.shared_experts, expert_gate_up, hidden});
        require_plain(mlp + "shared_experts.shared_w2_weight",
                      {kContract.shared_experts, hidden,
                       kContract.expert_intermediate_size});
        const bool quantized = inkling_quantized_expert_layer(layer);
        require_experts(mlp + "experts.w13_weight", quantized, expert_gate_up,
                        hidden);
        require_experts(mlp + "experts.w2_weight", quantized, hidden,
                        kContract.expert_intermediate_size);
        if (result.errors.size() >= kMaximumOpenErrors) break;
    }

    for (std::uint32_t depth = 0U; depth < kContract.mtp_layers; ++depth) {
        const auto prefix = inkling_mtp_prefix(depth);
        const auto block = prefix + "transformer_block.";
        require_plain(prefix + "embed_norm.weight", {hidden});
        require_plain(prefix + "hidden_norm.weight", {hidden});
        // The depth block consumes the normalized hidden state concatenated
        // with the normalized embedding, so its input is twice the width.
        require_plain(prefix + "input_proj.weight", {hidden, 2U * hidden});
        require_attention(block, inkling_mtp_relative_extent(depth));
        require_plain(block + "mlp.w13_dn.weight", {dense_gate_up, hidden});
        require_plain(block + "mlp.w2_md.weight",
                      {hidden, kContract.dense_intermediate_size});
        require_plain(block + "mlp.global_scale", {1U});
        if (result.errors.size() >= kMaximumOpenErrors) break;
    }

    if (!result.errors.empty()) return result;
    result.value = std::move(reader);
    return result;
}

const InklingTensor* InklingCheckpointReader::find(
    std::string_view name) const noexcept {
    const auto found = by_name_.find(name);
    return found == by_name_.end() ? nullptr : &tensors_[found->second];
}

ParseResult<InklingLinear> InklingCheckpointReader::linear(
    std::string_view name, std::uint64_t rows, std::uint64_t columns) const {
    ParseResult<InklingLinear> result;
    result.value.rows = rows;
    result.value.columns = columns;
    const auto* weight = find(name);
    if (weight == nullptr || weight->dtype != SafetensorsDtype::Bf16 ||
        weight->shape != std::vector<std::uint64_t>{rows, columns}) {
        result.errors.emplace_back("Inkling linear is missing or mismatched: " +
                                   std::string(name));
        return result;
    }
    result.value.weight = weight;
    return result;
}

ParseResult<InklingExpertStack> InklingCheckpointReader::expert_stack(
    std::string_view base_name, std::uint32_t layer, std::uint64_t experts,
    std::uint64_t rows, std::uint64_t columns) const {
    ParseResult<InklingExpertStack> result;
    const std::string base(base_name);
    result.value.experts = experts;
    result.value.rows = rows;
    result.value.columns = columns;
    if (!inkling_quantized_expert_layer(layer)) {
        const auto* weight = find(base);
        if (weight == nullptr || weight->dtype != SafetensorsDtype::Bf16 ||
            weight->shape != std::vector<std::uint64_t>{experts, rows, columns}) {
            result.errors.emplace_back(
                "Inkling plain expert stack is missing or mismatched: " + base);
            return result;
        }
        result.value.encoding = InklingTensorEncoding::Plain;
        result.value.weight = weight;
        return result;
    }
    const auto* packed = find(base);
    const auto* scale = find(base + ".scale");
    const auto* global = find(base + ".scale2");
    const auto packed_columns = columns / 2U;
    const auto scale_columns =
        (columns + kContract.nvfp4_group_size - 1U) / kContract.nvfp4_group_size;
    if (packed == nullptr || scale == nullptr || global == nullptr ||
        columns % 2U != 0U || packed->dtype != SafetensorsDtype::U8 ||
        scale->dtype != SafetensorsDtype::F8E4M3 ||
        global->dtype != SafetensorsDtype::F32 ||
        packed->shape !=
            std::vector<std::uint64_t>{experts, rows, packed_columns} ||
        scale->shape != std::vector<std::uint64_t>{experts, rows, scale_columns} ||
        global->shape != std::vector<std::uint64_t>{experts}) {
        result.errors.emplace_back(
            "Inkling NVFP4 expert stack is missing or mismatched: " + base);
        return result;
    }
    result.value.encoding = InklingTensorEncoding::Nvfp4Group16;
    result.value.packed = packed;
    result.value.scale = scale;
    result.value.global_scale = global;
    return result;
}

ParseResult<std::vector<std::byte>> InklingCheckpointReader::read(
    std::string_view name, std::uint64_t maximum_bytes) const {
    ParseResult<std::vector<std::byte>> result;
    const auto* tensor = find(name);
    if (tensor == nullptr) {
        result.errors.emplace_back("Inkling tensor does not exist: " +
                                   std::string(name));
        return result;
    }
    if (tensor->bytes > maximum_bytes ||
        tensor->bytes > std::numeric_limits<std::size_t>::max()) {
        result.errors.emplace_back("Inkling tensor exceeds the read ceiling: " +
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

ParseResult<std::vector<float>> InklingCheckpointReader::read_f32(
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
            "Inkling tensor cannot be decoded as bounded FP32: " + std::string(name));
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

ParseResult<std::vector<float>> InklingCheckpointReader::read_f32_row(
    std::string_view name, std::uint64_t row) const {
    ParseResult<std::vector<float>> result;
    const auto* tensor = find(name);
    if (tensor == nullptr || tensor->shape.size() != 2U ||
        row >= tensor->shape[0]) {
        result.errors.emplace_back("Inkling tensor row is out of range: " +
                                   std::string(name));
        return result;
    }
    const auto width = safetensors_dtype_bytes(tensor->dtype);
    if ((tensor->dtype != SafetensorsDtype::Bf16 &&
         tensor->dtype != SafetensorsDtype::F16 &&
         tensor->dtype != SafetensorsDtype::F32) || width == 0U) {
        result.errors.emplace_back("Inkling tensor row is not floating point");
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

ParseResult<std::span<const std::byte>> InklingCheckpointReader::view(
    std::string_view name) const {
    ParseResult<std::span<const std::byte>> result;
    const auto* tensor = find(name);
    if (tensor == nullptr) {
        result.errors.emplace_back("Inkling tensor does not exist: " +
                                   std::string(name));
        return result;
    }
    std::scoped_lock lock(mapping_mutex_);
    auto mapping = mappings_.find(tensor->shard);
    if (mapping == mappings_.end()) {
        const int descriptor = shards_.descriptor(tensor->shard);
        if (descriptor < 0) {
            result.errors.emplace_back("Inkling shard is not open for " +
                                       tensor->name);
            return result;
        }
        struct stat status {};
        if (fstat(descriptor, &status) != 0 || status.st_size <= 0) {
            result.errors.emplace_back("cannot size Inkling shard " +
                                       tensor->shard + ": " +
                                       std::strerror(errno));
            return result;
        }
        const auto bytes = static_cast<std::uint64_t>(status.st_size);
        void* address = mmap(nullptr, static_cast<std::size_t>(bytes), PROT_READ,
                             MAP_SHARED, descriptor, 0);
        if (address == MAP_FAILED) {
            result.errors.emplace_back("cannot map Inkling shard " + tensor->shard +
                                       ": " + std::strerror(errno));
            return result;
        }
        mapping = mappings_.emplace(
            tensor->shard,
            ShardMapping{static_cast<std::byte*>(address), bytes}).first;
    }
    if (tensor->offset > mapping->second.bytes ||
        tensor->bytes > mapping->second.bytes - tensor->offset) {
        result.errors.emplace_back("mapped tensor exceeds its Inkling shard: " +
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

void InklingCheckpointReader::release_mapped_views() const noexcept {
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

ParseResult<InklingNvfp4MatrixView> InklingCheckpointReader::nvfp4_expert_view(
    const InklingExpertStack& stack, std::uint64_t expert) const {
    ParseResult<InklingNvfp4MatrixView> result;
    if (stack.encoding != InklingTensorEncoding::Nvfp4Group16 ||
        stack.packed == nullptr || stack.scale == nullptr ||
        stack.global_scale == nullptr) {
        result.errors.emplace_back("expert stack is not an NVFP4 module");
        return result;
    }
    if (expert >= stack.experts) {
        result.errors.emplace_back("expert index is outside the stack");
        return result;
    }
    auto packed = view(stack.packed->name);
    if (!packed.ok()) {
        result.errors = std::move(packed.errors);
        return result;
    }
    auto scale = view(stack.scale->name);
    if (!scale.ok()) {
        result.errors = std::move(scale.errors);
        return result;
    }
    auto global = read_f32(stack.global_scale->name, stack.experts);
    if (!global.ok()) {
        result.errors = std::move(global.errors);
        return result;
    }
    if (global.value.size() != stack.experts) {
        result.errors.emplace_back("NVFP4 global scale count does not match the stack");
        return result;
    }
    const auto packed_columns = stack.columns / 2U;
    const auto scale_columns =
        (stack.columns + kContract.nvfp4_group_size - 1U) /
        kContract.nvfp4_group_size;
    const auto packed_stride = stack.rows * packed_columns;
    const auto scale_stride = stack.rows * scale_columns;
    if (packed.value.size() < (expert + 1U) * packed_stride ||
        scale.value.size() < (expert + 1U) * scale_stride) {
        result.errors.emplace_back("NVFP4 expert slice exceeds its stack");
        return result;
    }
    result.value.packed = packed.value.subspan(
        static_cast<std::size_t>(expert * packed_stride),
        static_cast<std::size_t>(packed_stride));
    result.value.scales = scale.value.subspan(
        static_cast<std::size_t>(expert * scale_stride),
        static_cast<std::size_t>(scale_stride));
    result.value.global_scale = global.value[static_cast<std::size_t>(expert)];
    result.value.rows = stack.rows;
    result.value.columns = stack.columns;
    result.value.packed_columns = packed_columns;
    result.value.scale_columns = scale_columns;
    result.value.group_size = kContract.nvfp4_group_size;
    return result;
}

CheckpointReadStats InklingCheckpointReader::stats() const noexcept {
    const auto nanoseconds = read_nanoseconds_.load(std::memory_order_relaxed);
    return {read_calls_.load(std::memory_order_relaxed),
            read_bytes_.load(std::memory_order_relaxed), nanoseconds,
            nanoseconds};
}

}  // namespace strata
