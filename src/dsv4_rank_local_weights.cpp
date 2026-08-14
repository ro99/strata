#include "strata/dsv4_rank_local_weights.hpp"

#include "strata/deepseek_ops.hpp"
#include "strata/deepseek_rank_shard.hpp"
#include "strata/model_adapter.hpp"

#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <utility>

namespace strata {
namespace {

constexpr std::uint64_t kHidden = 4096U;
constexpr std::uint64_t kMhc = 4U * kHidden;
constexpr std::uint64_t kRouterExperts = 256U;
constexpr std::uint64_t kQueryNormWidth = 1024U;
constexpr std::uint64_t kKeyValueNormWidth = 512U;
constexpr std::uint64_t kIndexHeadDim = 128U;
// Layers below this route by the checkpoint token-to-expert table rather than
// by a learned selection bias.
constexpr std::uint32_t kHashRoutedLayers = 3U;

struct HostShard {
    Dsv4RankShardDescriptor descriptor;
    Dsv4RankShardPayload payload;
};

[[nodiscard]] std::string front_error(const std::vector<std::string>& errors,
                                     const char* fallback) {
    return errors.empty() ? std::string(fallback) : errors.front();
}

[[nodiscard]] ValidationResult load_shard(
    const Dsv4CheckpointReader& checkpoint, const std::string& name,
    Dsv4ShardOwnership ownership, std::uint32_t rank, HostShard& output) {
    ValidationResult result;
    auto described = describe_dsv4_rank_shard(
        checkpoint, name, ownership, rank,
        static_cast<std::uint32_t>(kDsv4RankLocalWorld));
    if (!described.ok()) {
        result.errors.emplace_back(
            "describe " + name + ": " +
            front_error(described.errors, "unknown"));
        return result;
    }
    auto payload = load_dsv4_rank_shard(checkpoint, described.value);
    if (!payload.ok()) {
        result.errors.emplace_back(
            "load " + name + ": " + front_error(payload.errors, "unknown"));
        return result;
    }
    output.descriptor = std::move(described.value);
    output.payload = std::move(payload.value);
    return result;
}

[[nodiscard]] CudaWeightDescriptor cuda_descriptor(const HostShard& host) {
    CudaWeightDescriptor descriptor;
    descriptor.encoding =
        host.descriptor.encoding == Dsv4TensorEncoding::Fp8E4m3Block128
            ? CudaWeightEncoding::Fp8E4m3Block128
            : CudaWeightEncoding::Plain;
    descriptor.dtype = host.descriptor.weight_dtype;
    descriptor.rows = host.descriptor.local_logical_shape[0];
    descriptor.columns = host.descriptor.local_logical_shape[1];
    descriptor.packed_columns = host.descriptor.local_packed_shape[1];
    descriptor.scale_columns = host.descriptor.local_scale_shape.empty()
        ? 0U
        : host.descriptor.local_scale_shape[1];
    descriptor.group_size =
        descriptor.encoding == CudaWeightEncoding::Plain ? 0U : 128U;
    return descriptor;
}

[[nodiscard]] ValidationResult upload_shard(
    CudaBackend& backend, int device, const HostShard& host,
    CudaWeight& output, std::uint64_t& device_bytes) {
    ValidationResult result;
    auto status = backend.upload(device, cuda_descriptor(host),
                                 host.payload.weight, host.payload.scale,
                                 output);
    if (!status.ok()) {
        result.errors = std::move(status.errors);
        return result;
    }
    device_bytes += output.device_bytes();
    return result;
}

// The target attention projections are consumed as BF16. Converting at load
// keeps decode free of any dequantization step and matches the accepted
// rank-local layer exactly.
[[nodiscard]] ValidationResult convert_fp8_to_bf16(
    const HostShard& source, HostShard& output) {
    ValidationResult result;
    if (source.descriptor.encoding != Dsv4TensorEncoding::Fp8E4m3Block128) {
        result.errors.emplace_back(
            "expected FP8 block-128 encoding for " +
            source.descriptor.weight_name);
        return result;
    }
    output = source;
    const auto rows = source.descriptor.local_logical_shape[0];
    const auto columns = source.descriptor.local_logical_shape[1];
    std::vector<std::uint16_t> converted(
        static_cast<std::size_t>(rows * columns));
    auto status = dsv4_fp8_e4m3_block128_to_bf16(
        converted, source.payload.weight, source.payload.scale, rows, columns);
    if (!status.ok()) {
        result.errors = std::move(status.errors);
        return result;
    }
    output.payload.weight.resize(converted.size() * sizeof(std::uint16_t));
    std::memcpy(output.payload.weight.data(), converted.data(),
                output.payload.weight.size());
    output.payload.scale.clear();
    output.descriptor.encoding = Dsv4TensorEncoding::Plain;
    output.descriptor.weight_dtype = SafetensorsDtype::Bf16;
    output.descriptor.scale_dtype = SafetensorsDtype::Other;
    output.descriptor.scale_name.clear();
    output.descriptor.scale_shard.clear();
    output.descriptor.scale_shape.clear();
    output.descriptor.local_scale_shape.clear();
    output.descriptor.local_packed_shape =
        output.descriptor.local_logical_shape;
    output.descriptor.local_scale_bytes = 0U;
    return result;
}

[[nodiscard]] std::vector<float> decode_plain_f32(const HostShard& host) {
    const auto width = host.descriptor.weight_dtype == SafetensorsDtype::F32
        ? sizeof(float)
        : sizeof(std::uint16_t);
    const auto elements = host.payload.weight.size() / width;
    std::vector<float> result(elements);
    for (std::size_t index = 0U; index < elements; ++index) {
        if (host.descriptor.weight_dtype == SafetensorsDtype::Bf16) {
            std::uint16_t encoded{};
            std::memcpy(&encoded, host.payload.weight.data() + index * width,
                        sizeof(encoded));
            const std::uint32_t widened = static_cast<std::uint32_t>(encoded)
                                          << 16U;
            float value{};
            std::memcpy(&value, &widened, sizeof(value));
            result[index] = value;
        } else if (host.descriptor.weight_dtype == SafetensorsDtype::F32) {
            std::memcpy(&result[index],
                        host.payload.weight.data() + index * width,
                        sizeof(float));
        }
    }
    return result;
}

[[nodiscard]] ValidationResult load_plain_cuda(
    const Dsv4CheckpointReader& checkpoint, const std::string& name,
    std::uint64_t rows, std::uint64_t columns, int device,
    CudaBackend& backend, CudaWeight& output, std::uint64_t& device_bytes) {
    ValidationResult result;
    const auto* tensor = checkpoint.find(name);
    if (tensor == nullptr || tensor->source_dtype != SafetensorsDtype::Bf16 ||
        tensor->source_shape.size() != 2U || tensor->source_shape[0] != rows ||
        tensor->source_shape[1] != columns) {
        result.errors.emplace_back(
            "plain tensor contract failed for " + name);
        return result;
    }
    auto bytes = checkpoint.read(name, tensor->source_bytes);
    if (!bytes.ok()) {
        result.errors = std::move(bytes.errors);
        return result;
    }
    CudaWeightDescriptor descriptor;
    descriptor.encoding = CudaWeightEncoding::Plain;
    descriptor.dtype = SafetensorsDtype::Bf16;
    descriptor.rows = rows;
    descriptor.columns = columns;
    auto status = backend.upload(device, descriptor, bytes.value, {}, output);
    if (!status.ok()) {
        result.errors = std::move(status.errors);
        return result;
    }
    device_bytes += output.device_bytes();
    return result;
}

[[nodiscard]] ValidationResult load_mhc(
    const Dsv4CheckpointReader& checkpoint, CudaBackend& backend, int device,
    std::uint32_t layer, bool ffn, CudaDsv4MhcWeights& output,
    std::uint64_t& device_bytes) {
    ValidationResult result;
    const auto prefix = "layers." + std::to_string(layer) + ".";
    const std::string stem = ffn ? "hc_ffn_" : "hc_attn_";
    auto projection = checkpoint.read_f32(prefix + stem + "fn", 24U * kMhc);
    auto scale = checkpoint.read_f32(prefix + stem + "scale", 3U);
    auto base = checkpoint.read_f32(prefix + stem + "base", 24U);
    auto norm = checkpoint.read_f32(
        prefix + (ffn ? "ffn_norm.weight" : "attn_norm.weight"), kHidden);
    if (!projection.ok() || !scale.ok() || !base.ok() || !norm.ok()) {
        result.errors.emplace_back(
            "mHC weight read failed for layer " + std::to_string(layer) +
            (ffn ? " ffn" : " attn"));
        return result;
    }
    auto uploaded = backend.upload_dsv4_mhc_weights(
        device, projection.value, scale.value, base.value, norm.value, output);
    if (!uploaded.ok()) {
        result.errors = std::move(uploaded.errors);
    } else {
        device_bytes += output.device_bytes();
    }
    return result;
}

[[nodiscard]] ValidationResult load_rank_attention(
    const Dsv4CheckpointReader& checkpoint, CudaBackend& backend, int device,
    std::uint32_t layer, std::uint32_t rank,
    Dsv4RankLocalRankLayerWeights& output, std::uint64_t& device_bytes) {
    ValidationResult result;
    const auto prefix = "layers." + std::to_string(layer) + ".attn.";
    HostShard query_a, query_b, key_value, output_a, output_b;
    HostShard query_norm, key_value_norm;
    const std::array<ValidationResult, 7U> reads{
        load_shard(checkpoint, prefix + "wq_a",
                   Dsv4ShardOwnership::Replicated, 0U, query_a),
        load_shard(checkpoint, prefix + "wq_b",
                   Dsv4ShardOwnership::Replicated, 0U, query_b),
        load_shard(checkpoint, prefix + "wkv",
                   Dsv4ShardOwnership::Replicated, 0U, key_value),
        load_shard(checkpoint, prefix + "wo_a",
                   Dsv4ShardOwnership::ContiguousRows, rank, output_a),
        load_shard(checkpoint, prefix + "wo_b",
                   Dsv4ShardOwnership::StridedColumns, rank, output_b),
        load_shard(checkpoint, prefix + "q_norm.weight",
                   Dsv4ShardOwnership::Replicated, 0U, query_norm),
        load_shard(checkpoint, prefix + "kv_norm.weight",
                   Dsv4ShardOwnership::Replicated, 0U, key_value_norm),
    };
    for (const auto& read : reads) {
        if (!read.ok()) {
            result.errors.insert(result.errors.end(), read.errors.begin(),
                                 read.errors.end());
        }
    }
    if (!result.ok()) return result;

    HostShard query_a_bf16, query_b_bf16, key_value_bf16;
    HostShard output_a_bf16, output_b_bf16;
    const std::array<ValidationResult, 5U> conversions{
        convert_fp8_to_bf16(query_a, query_a_bf16),
        convert_fp8_to_bf16(query_b, query_b_bf16),
        convert_fp8_to_bf16(key_value, key_value_bf16),
        convert_fp8_to_bf16(output_a, output_a_bf16),
        convert_fp8_to_bf16(output_b, output_b_bf16),
    };
    for (const auto& conversion : conversions) {
        if (!conversion.ok()) {
            result.errors.insert(result.errors.end(),
                                 conversion.errors.begin(),
                                 conversion.errors.end());
        }
    }
    if (!result.ok()) return result;

    const std::array<ValidationResult, 5U> uploads{
        upload_shard(backend, device, query_a_bf16, output.query_a, device_bytes),
        upload_shard(backend, device, query_b_bf16, output.query_b, device_bytes),
        upload_shard(backend, device, key_value_bf16, output.key_value, device_bytes),
        upload_shard(backend, device, output_a_bf16, output.output_a, device_bytes),
        upload_shard(backend, device, output_b_bf16, output.output_b, device_bytes),
    };
    for (const auto& upload : uploads) {
        if (!upload.ok()) {
            result.errors.insert(result.errors.end(), upload.errors.begin(),
                                 upload.errors.end());
        }
    }
    if (!result.ok()) return result;

    // Compressor state is one logical, replicated KV stream. Rank 0 owns its
    // projection and exact host pooling; the deferred page callback copies the
    // resulting encoded row to both ranks. Loading a second identical set
    // would add roughly 0.6 GiB without reducing the binding CPU term.
    if (rank == 0U) {
        const auto& ratios =
            deepseek_v4_flash_0731_spec().deepseek_v4.compression_ratios;
        if (layer >= ratios.size()) {
            result.errors.emplace_back(
                "compression-ratio contract is missing layer " +
                std::to_string(layer));
            return result;
        }
        const auto ratio = ratios[layer];
        if (ratio != 0U) {
            const auto coefficient = ratio == 4U ? 2U : 1U;
            const auto elements = coefficient * kKeyValueNormWidth;
            auto value = load_plain_cuda(
                checkpoint, prefix + "compressor.wkv.weight", elements, kHidden,
                device, backend, output.compressor_value, device_bytes);
            if (!value.ok()) return value;
            auto gate = load_plain_cuda(
                checkpoint, prefix + "compressor.wgate.weight", elements, kHidden,
                device, backend, output.compressor_gate, device_bytes);
            if (!gate.ok()) return gate;
            output.compressor_elements = static_cast<std::uint32_t>(elements);

            if (ratio == 4U) {
                constexpr auto index_elements = 2U * kIndexHeadDim;
                auto index_value = load_plain_cuda(
                    checkpoint, prefix + "indexer.compressor.wkv.weight",
                    index_elements, kHidden, device, backend,
                    output.index_compressor_value, device_bytes);
                if (!index_value.ok()) return index_value;
                auto index_gate = load_plain_cuda(
                    checkpoint, prefix + "indexer.compressor.wgate.weight",
                    index_elements, kHidden, device, backend,
                    output.index_compressor_gate, device_bytes);
                if (!index_gate.ok()) return index_gate;
                output.index_compressor_elements =
                    static_cast<std::uint32_t>(index_elements);
            }
        }
    }

    output.query_norm = decode_plain_f32(query_norm);
    output.key_value_norm = decode_plain_f32(key_value_norm);
    if (output.query_norm.size() != kQueryNormWidth ||
        output.key_value_norm.size() != kKeyValueNormWidth) {
        result.errors.emplace_back(
            "attention norm width contract failed at layer " +
            std::to_string(layer) + ": q_norm " +
            std::to_string(output.query_norm.size()) + ", kv_norm " +
            std::to_string(output.key_value_norm.size()));
    }
    return result;
}

[[nodiscard]] ValidationResult load_rank_shared(
    const Dsv4CheckpointReader& checkpoint, CudaBackend& backend, int device,
    std::uint32_t layer, std::uint32_t rank,
    Dsv4RankLocalRankLayerWeights& output, std::uint64_t& device_bytes) {
    ValidationResult result;
    const auto prefix =
        "layers." + std::to_string(layer) + ".ffn.shared_experts.";
    HostShard w1, w3, w2;
    const std::array<ValidationResult, 3U> reads{
        load_shard(checkpoint, prefix + "w1",
                   Dsv4ShardOwnership::ContiguousRows, rank, w1),
        load_shard(checkpoint, prefix + "w3",
                   Dsv4ShardOwnership::ContiguousRows, rank, w3),
        load_shard(checkpoint, prefix + "w2",
                   Dsv4ShardOwnership::StridedColumns, rank, w2),
    };
    for (const auto& read : reads) {
        if (!read.ok()) {
            result.errors.insert(result.errors.end(), read.errors.begin(),
                                 read.errors.end());
        }
    }
    if (!result.ok()) return result;

    const std::array<ValidationResult, 3U> uploads{
        upload_shard(backend, device, w1, output.shared_w1, device_bytes),
        upload_shard(backend, device, w3, output.shared_w3, device_bytes),
        upload_shard(backend, device, w2, output.shared_w2, device_bytes),
    };
    for (const auto& upload : uploads) {
        if (!upload.ok()) {
            result.errors.insert(result.errors.end(), upload.errors.begin(),
                                 upload.errors.end());
        }
    }
    if (!result.ok()) return result;
    output.shared = {&output.shared_w1, &output.shared_w3, &output.shared_w2,
                     1.0F};
    return result;
}

// Whole-table load. One row per token, `experts_per_token` entries wide.
[[nodiscard]] ValidationResult load_hash_router_table(
    const Dsv4CheckpointReader& checkpoint, std::uint32_t layer,
    Dsv4RankLocalLayerStorage& output) {
    ValidationResult result;
    const auto name =
        "layers." + std::to_string(layer) + ".ffn.gate.tid2eid";
    const auto* tensor = checkpoint.find(name);
    const auto width = static_cast<std::uint64_t>(
        kDeepSeekV4ExecutionContract.experts_per_token);
    if (tensor == nullptr || tensor->source_dtype != SafetensorsDtype::I64 ||
        tensor->source_shape.size() != 2U || tensor->source_shape[0] == 0U ||
        tensor->source_shape[1] != width) {
        result.errors.emplace_back(
            "hash router table contract failed for " + name);
        return result;
    }
    auto bytes = checkpoint.read(name, tensor->source_bytes);
    if (!bytes.ok()) {
        result.errors = std::move(bytes.errors);
        return result;
    }
    const auto rows = tensor->source_shape[0];
    const auto expected = rows * width * sizeof(std::int64_t);
    if (bytes.value.size() != expected) {
        result.errors.emplace_back(
            "hash router table for " + name + " read " +
            std::to_string(bytes.value.size()) + " B, expected " +
            std::to_string(expected) + " B");
        return result;
    }
    output.hash_router_table.resize(static_cast<std::size_t>(rows * width));
    for (std::size_t index = 0U; index < output.hash_router_table.size();
         ++index) {
        std::int64_t selected = -1;
        std::memcpy(&selected,
                    bytes.value.data() + index * sizeof(std::int64_t),
                    sizeof(selected));
        if (selected < 0 ||
            selected >= static_cast<std::int64_t>(kRouterExperts)) {
            result.errors.emplace_back(
                "hash router table for " + name + " holds invalid expert " +
                std::to_string(selected) + " at index " +
                std::to_string(index));
            output.hash_router_table.clear();
            return result;
        }
        output.hash_router_table[index] =
            static_cast<std::uint32_t>(selected);
    }
    output.hash_router_rows = static_cast<std::uint32_t>(rows);
    output.hash_router_width = static_cast<std::uint32_t>(width);
    return result;
}

}  // namespace

void Dsv4RankLocalWeightStore::clear() noexcept {
    layers_.clear();
    head_ = {};
    stats_ = {};
    loaded_ = false;
}

std::span<const std::uint32_t> Dsv4RankLocalWeightStore::hash_router_row(
    std::uint32_t layer, std::uint32_t token) const noexcept {
    if (layer >= layers_.size()) return {};
    const auto& storage = layers_[layer];
    if (storage.hash_router_width == 0U ||
        token >= storage.hash_router_rows) {
        return {};
    }
    const auto offset =
        static_cast<std::size_t>(token) * storage.hash_router_width;
    return std::span<const std::uint32_t>(
        storage.hash_router_table.data() + offset, storage.hash_router_width);
}

Dsv4RankLocalLayerWeights Dsv4RankLocalWeightStore::layer_view(
    std::uint32_t layer, std::uint32_t token) const {
    Dsv4RankLocalLayerWeights view;
    if (layer >= layers_.size()) return view;
    const auto& storage = layers_[layer];
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        const auto& source = storage.rank[rank];
        auto& target = view.rank[rank];
        target.query_a = &source.query_a;
        target.query_b = &source.query_b;
        target.key_value = &source.key_value;
        target.compressor_value = source.compressor_elements == 0U
            ? nullptr : &source.compressor_value;
        target.compressor_gate = source.compressor_elements == 0U
            ? nullptr : &source.compressor_gate;
        target.index_compressor_value =
            source.index_compressor_elements == 0U
                ? nullptr : &source.index_compressor_value;
        target.index_compressor_gate =
            source.index_compressor_elements == 0U
                ? nullptr : &source.index_compressor_gate;
        target.compressor_elements = source.compressor_elements;
        target.index_compressor_elements = source.index_compressor_elements;
        target.output_a = &source.output_a;
        target.output_b = &source.output_b;
        target.router = &source.router;
        target.shared = &source.shared;
        target.attention_mhc = &source.attention_mhc;
        target.ffn_mhc = &source.ffn_mhc;
        target.next_attention_mhc = source.has_next_attention_mhc
            ? &source.next_attention_mhc
            : nullptr;
        target.query_norm = source.query_norm;
        target.key_value_norm = source.key_value_norm;
    }
    // Exactly one routing membership is populated, matching the executor's
    // contract.
    if (layer < kHashRoutedLayers) {
        view.router_token_experts = hash_router_row(layer, token);
    } else {
        view.router_bias = storage.router_bias;
    }
    return view;
}

ValidationResult Dsv4RankLocalWeightStore::load(
    const Dsv4CheckpointReader& checkpoint, CudaBackend& backend,
    const std::array<int, kDsv4RankLocalWorld>& devices,
    std::uint32_t layer_count) {
    ValidationResult result;
    const auto started = std::chrono::steady_clock::now();
    const auto before = checkpoint.stats();
    clear();

    if (layer_count == 0U) {
        result.errors.emplace_back(
            "rank-local weight store requires at least one layer");
        return result;
    }
    if (devices[0] == devices[1]) {
        result.errors.emplace_back(
            "rank-local weight store requires two distinct devices");
        return result;
    }

    layers_.resize(layer_count);
    std::array<std::uint64_t, kDsv4RankLocalWorld> device_bytes{};
    std::array<ValidationResult, kDsv4RankLocalWorld> rank_results;
    std::array<std::thread, kDsv4RankLocalWorld> rank_workers;

    // Each rank owns a distinct CUDA device, arena, stream, and destination
    // slot. Load the two complete rank streams concurrently so their host FP8
    // conversion does not serialize startup after the resident spine warmup.
    for (std::uint32_t rank = 0U;
         rank < static_cast<std::uint32_t>(kDsv4RankLocalWorld); ++rank) {
        rank_workers[rank] = std::thread([&, rank] {
            auto& rank_result = rank_results[rank];
            const auto device = devices[rank];
            for (std::uint32_t layer = 0U;
                 layer < layer_count && rank_result.ok(); ++layer) {
                auto& storage = layers_[layer];
                auto& target = storage.rank[rank];
                auto attention = load_rank_attention(
                    checkpoint, backend, device, layer, rank, target,
                    device_bytes[rank]);
                if (!attention.ok()) {
                    rank_result.errors = std::move(attention.errors);
                    break;
                }
                auto shared = load_rank_shared(
                    checkpoint, backend, device, layer, rank, target,
                    device_bytes[rank]);
                if (!shared.ok()) {
                    rank_result.errors = std::move(shared.errors);
                    break;
                }
                auto router = load_plain_cuda(
                    checkpoint,
                    "layers." + std::to_string(layer) + ".ffn.gate.weight",
                    kRouterExperts, kHidden, device, backend, target.router,
                    device_bytes[rank]);
                if (!router.ok()) {
                    rank_result.errors = std::move(router.errors);
                    break;
                }
                auto attention_mhc = load_mhc(
                    checkpoint, backend, device, layer, false,
                    target.attention_mhc, device_bytes[rank]);
                auto ffn_mhc = load_mhc(
                    checkpoint, backend, device, layer, true,
                    target.ffn_mhc, device_bytes[rank]);
                if (!attention_mhc.ok() || !ffn_mhc.ok()) {
                    rank_result.errors = attention_mhc.ok()
                        ? std::move(ffn_mhc.errors)
                        : std::move(attention_mhc.errors);
                    break;
                }
                // The chain queues the next layer's attention mHC transition
                // from this layer, so every layer but the last owns its
                // successor's attention mHC weights.
                if (layer + 1U < layer_count) {
                    auto next = load_mhc(
                        checkpoint, backend, device, layer + 1U, false,
                        target.next_attention_mhc, device_bytes[rank]);
                    if (!next.ok()) {
                        rank_result.errors = std::move(next.errors);
                        break;
                    }
                    target.has_next_attention_mhc = true;
                }
            }
        });
    }
    for (auto& worker : rank_workers) worker.join();
    for (auto& rank_result : rank_results) {
        if (!rank_result.ok()) {
            for (auto& error : rank_result.errors) {
                result.errors.emplace_back(std::move(error));
            }
        }
    }
    if (!result.ok()) {
        clear();
        return result;
    }

    // Router membership/bias is shared host state, so populate it once after
    // both independent rank streams have completed.
    for (std::uint32_t layer = 0U; layer < layer_count; ++layer) {
        auto& storage = layers_[layer];
        if (layer < kHashRoutedLayers) {
            auto table = load_hash_router_table(checkpoint, layer, storage);
            if (!table.ok()) {
                result.errors = std::move(table.errors);
                clear();
                return result;
            }
        } else {
            auto bias = checkpoint.read_f32(
                "layers." + std::to_string(layer) + ".ffn.gate.bias",
                kRouterExperts);
            if (!bias.ok() || bias.value.size() != kRouterExperts) {
                result.errors.emplace_back(
                    "learned router bias read failed at layer " +
                    std::to_string(layer));
                clear();
                return result;
            }
            storage.router_bias = std::move(bias.value);
        }
    }

    // Terminal head: auxiliary weights are replicated, the projection is
    // sharded by contiguous rows so each rank publishes its own logit shard.
    auto projection = checkpoint.read_f32("hc_head_fn", 4U * kMhc);
    auto scale = checkpoint.read_f32("hc_head_scale", 1U);
    auto base = checkpoint.read_f32("hc_head_base", 4U);
    auto norm = checkpoint.read_f32("norm.weight", kHidden);
    if (!projection.ok() || !scale.ok() || !base.ok() || !norm.ok()) {
        result.errors.emplace_back(
            "terminal head auxiliary weight read failed");
        clear();
        return result;
    }
    head_.projection = std::move(projection.value);
    head_.scale = std::move(scale.value);
    head_.base = std::move(base.value);
    head_.norm = std::move(norm.value);
    for (std::uint32_t rank = 0U;
         rank < static_cast<std::uint32_t>(kDsv4RankLocalWorld); ++rank) {
        HostShard shard;
        auto read = load_shard(checkpoint, "head",
                               Dsv4ShardOwnership::ContiguousRows, rank, shard);
        if (!read.ok()) {
            result.errors = std::move(read.errors);
            clear();
            return result;
        }
        auto upload = upload_shard(backend, devices[rank], shard,
                                   head_.weights[rank], device_bytes[rank]);
        if (!upload.ok()) {
            result.errors = std::move(upload.errors);
            clear();
            return result;
        }
    }

    const auto after = checkpoint.stats();
    stats_.checkpoint_read_calls = after.calls - before.calls;
    stats_.checkpoint_read_bytes = after.bytes - before.bytes;
    stats_.device_weight_bytes = device_bytes;
    stats_.layers = layer_count;
    stats_.host_norm_bytes = 0U;
    for (const auto& storage : layers_) {
        for (const auto& rank : storage.rank) {
            stats_.host_norm_bytes +=
                (rank.query_norm.size() + rank.key_value_norm.size()) *
                sizeof(float);
        }
        stats_.host_norm_bytes +=
            storage.router_bias.size() * sizeof(float) +
            storage.hash_router_table.size() * sizeof(std::uint32_t);
    }
    stats_.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    loaded_ = true;
    return result;
}

}  // namespace strata
