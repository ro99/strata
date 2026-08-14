#pragma once

#include "strata/cuda_backend.hpp"
#include "strata/deepseek_checkpoint.hpp"
#include "strata/dsv4_rank_local_layer_executor.hpp"
#include "strata/dsv4_rank_local_topology.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace strata {

// One rank's device-resident weights for one layer. Owned by the store; the
// executor borrows pointers into this and never copies or frees them.
struct Dsv4RankLocalRankLayerWeights {
    CudaWeight query_a;
    CudaWeight query_b;
    CudaWeight key_value;
    CudaWeight compressor_value;
    CudaWeight compressor_gate;
    CudaWeight index_compressor_value;
    CudaWeight index_compressor_gate;
    std::uint32_t compressor_elements{};
    std::uint32_t index_compressor_elements{};
    CudaWeight output_a;
    CudaWeight output_b;
    CudaWeight router;
    CudaWeight shared_w1;
    CudaWeight shared_w3;
    CudaWeight shared_w2;
    CudaDeepSeekMoeExpert shared{};
    CudaDsv4MhcWeights attention_mhc{};
    CudaDsv4MhcWeights ffn_mhc{};
    CudaDsv4MhcWeights next_attention_mhc{};
    bool has_next_attention_mhc{};
    // Host-resident per-layer norms uploaded per command by the attention
    // preparation staging. Kept here so the spans handed to the executor
    // outlive every queued copy that reads them.
    std::vector<float> query_norm;
    std::vector<float> key_value_norm;
};

struct Dsv4RankLocalLayerStorage {
    std::array<Dsv4RankLocalRankLayerWeights, kDsv4RankLocalWorld> rank{};
    // Exactly one routing membership is populated per layer. Layers 3+ carry
    // the learned selection bias.
    std::vector<float> router_bias;
    // Layers 0-2 route by a per-token `tid2eid` row. The whole table is
    // resident because reading one row per token would be checkpoint I/O
    // inside the decode window, which the steady-state gate forbids. Stored
    // row-major as [rows][experts_per_token].
    std::vector<std::uint32_t> hash_router_table;
    std::uint32_t hash_router_rows{};
    std::uint32_t hash_router_width{};
};

struct Dsv4RankLocalHeadStorage {
    std::array<CudaWeight, kDsv4RankLocalWorld> weights{};
    std::vector<float> projection;
    std::vector<float> scale;
    std::vector<float> base;
    std::vector<float> norm;
};

struct Dsv4RankLocalWeightStats {
    std::uint64_t checkpoint_read_calls{};
    std::uint64_t checkpoint_read_bytes{};
    std::array<std::uint64_t, kDsv4RankLocalWorld> device_weight_bytes{};
    std::uint64_t host_norm_bytes{};
    std::uint32_t layers{};
    double seconds{};
};

// Runtime-owned rank-local weight store. Every tensor is read from the actual
// checkpoint through Dsv4RankShard descriptors -- never from a .d4c, .d4r or
// .d4m fixture. Loading is a setup-time operation: once load() returns, decode
// performs no checkpoint read and no weight allocation.
//
// Sharding contract, unchanged from the accepted rank-local layer:
//   attn.wq_a / wq_b / wkv              replicated
//   attn.wo_a                           contiguous rows by rank
//   attn.wo_b                           strided columns by rank
//   attn.q_norm / kv_norm               replicated, host-resident
//   ffn.shared_experts.w1 / w3          contiguous rows by rank
//   ffn.shared_experts.w2               strided columns by rank
//   ffn.gate.weight                     replicated
//   head                                contiguous rows by rank
class Dsv4RankLocalWeightStore {
public:
    Dsv4RankLocalWeightStore() = default;
    ~Dsv4RankLocalWeightStore() = default;
    Dsv4RankLocalWeightStore(const Dsv4RankLocalWeightStore&) = delete;
    Dsv4RankLocalWeightStore& operator=(const Dsv4RankLocalWeightStore&) = delete;
    Dsv4RankLocalWeightStore(Dsv4RankLocalWeightStore&&) = delete;
    Dsv4RankLocalWeightStore& operator=(Dsv4RankLocalWeightStore&&) = delete;

    // Loads all `layer_count` layers plus the terminal head. Atomic at the API
    // boundary: any failure clears every loaded layer and returns errors, so a
    // caller can never run against a partially resident set.
    [[nodiscard]] ValidationResult load(
        const Dsv4CheckpointReader& checkpoint, CudaBackend& backend,
        const std::array<int, kDsv4RankLocalWorld>& devices,
        std::uint32_t layer_count = kDsv4RankLocalLayerCount);

    // Borrowed view for one layer, shaped for the executor. The returned
    // spans and pointers are valid until the store is destroyed or reloaded.
    // `token` selects the hash-router row for layers 0-2 and is ignored by
    // later layers, which use the learned selection bias instead.
    [[nodiscard]] Dsv4RankLocalLayerWeights layer_view(
        std::uint32_t layer, std::uint32_t token) const;

    // The resident hash-router row for a layer below 3, or an empty span when
    // the layer routes by bias or the token is out of range.
    [[nodiscard]] std::span<const std::uint32_t> hash_router_row(
        std::uint32_t layer, std::uint32_t token) const noexcept;

    [[nodiscard]] bool loaded() const noexcept { return loaded_; }
    [[nodiscard]] std::uint32_t layer_count() const noexcept {
        return static_cast<std::uint32_t>(layers_.size());
    }
    [[nodiscard]] const Dsv4RankLocalHeadStorage& head() const noexcept {
        return head_;
    }
    [[nodiscard]] Dsv4RankLocalWeightStats stats() const noexcept {
        return stats_;
    }
    // Per-device resident bytes, for the admission account.
    [[nodiscard]] std::array<std::uint64_t, kDsv4RankLocalWorld>
    device_bytes() const noexcept { return stats_.device_weight_bytes; }

    void clear() noexcept;

private:
    std::vector<Dsv4RankLocalLayerStorage> layers_;
    Dsv4RankLocalHeadStorage head_;
    Dsv4RankLocalWeightStats stats_;
    bool loaded_{};
};

}  // namespace strata
