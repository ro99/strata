#include "strata/laguna_runtime.hpp"

#include "cuda_stats_delta.hpp"

#include "strata/attention_reference.hpp"
#include "strata/laguna_checkpoint.hpp"
#include "strata/laguna_ops.hpp"
#include "strata/model.hpp"
#include "strata/model_adapter.hpp"
#include "strata/runtime_support.hpp"
#include "strata/tokenizer.hpp"
#include "strata/trace.hpp"
#include "strata/worker_pool.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace strata {

namespace {

constexpr auto& kContract = kLagunaExecutionContract;
constexpr std::uint32_t kHidden = kContract.hidden_size;
constexpr std::uint32_t kLayers = kContract.layer_count;
constexpr std::uint32_t kHeadDim = kContract.head_dim;
constexpr std::uint32_t kKvHeads = kContract.key_value_heads;
constexpr std::uint32_t kKvColumns = kKvHeads * kHeadDim;
constexpr std::uint32_t kExperts = kContract.routed_experts;
constexpr std::uint32_t kTopK = kContract.experts_per_token;
constexpr std::uint32_t kExpertIntermediate = kContract.expert_intermediate_size;
constexpr std::uint32_t kSharedIntermediate =
    kContract.shared_expert_intermediate_size;
constexpr std::uint32_t kDenseIntermediate = kContract.dense_intermediate_size;
constexpr std::uint32_t kVocabulary = kContract.vocabulary_size;
constexpr std::uint32_t kSlidingWindow = kContract.sliding_window;
constexpr std::uint64_t kFlashAttentionWorkspaceReserve = 512ULL << 20U;
// 〈|EOS|〉 and </assistant>, the checkpoint's declared eos_token_id list.
constexpr std::array<std::uint32_t, 2> kStopTokenIds{2U, 24U};

float decode_bf16(std::uint16_t value) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

std::uint64_t elapsed_nanoseconds(
    std::chrono::steady_clock::time_point started) noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count());
}

std::string layer_prefix(std::uint32_t layer) {
    return "model.layers." + std::to_string(layer) + ".";
}

void move_errors(std::vector<std::string>& destination, ValidationResult source,
                 std::string_view context = {}) {
    for (auto& error : source.errors) {
        if (!context.empty()) error = std::string(context) + ": " + error;
        destination.push_back(std::move(error));
    }
}

// One routed expert projection triplet, resolved once at load so the decode
// path never rebuilds tensor names.
struct ExpertModules {
    LagunaLinear gate;
    LagunaLinear up;
    LagunaLinear down;
    std::size_t device{};
};

struct SpineWeight {
    CudaWeight weight;
    std::uint64_t bytes{};
};

struct LayerWeights {
    std::size_t device{};
    bool global{};
    std::uint32_t heads{};
    SpineWeight query;
    SpineWeight key;
    SpineWeight value;
    SpineWeight output;
    SpineWeight output_gate;
    SpineWeight router;
    SpineWeight shared_gate;
    SpineWeight shared_up;
    SpineWeight shared_down;
    SpineWeight dense_gate;
    SpineWeight dense_up;
    SpineWeight dense_down;
    std::vector<float> input_norm;
    std::vector<float> post_attention_norm;
    std::vector<float> query_norm;
    std::vector<float> key_norm;
    std::vector<float> correction_bias;
    std::vector<ExpertModules> experts;
};

struct LayerKv {
    // Rows are stored already rounded through BF16, which is the reference
    // cache dtype, but held as F32 so the attention request can point straight
    // at them. Holding BF16 and decoding on use cost a full O(context) decode
    // of the whole cache per layer per token to add one row; the values are
    // identical either way because decode(encode(x)) is a fixed point.
    // Sliding layers keep at most `sliding_window` rows and advance `start`;
    // global layers keep all.
    std::vector<float> keys;
    std::vector<float> values;
    std::uint32_t start{};
};

float bf16_round(float value) noexcept {
    return decode_bf16(bf16_encode(value));
}

struct ExpertJob {
    std::uint32_t expert{};
    std::size_t device{};
    std::vector<std::uint32_t> rows;
    std::vector<float> weights;
    std::vector<float> output;
    std::vector<std::string> errors;
    std::uint64_t gather_nanoseconds{};
    std::uint64_t expert_nanoseconds{};
    std::uint64_t cache_nanoseconds{};
};

CheckpointReadStats checkpoint_delta(const CheckpointReadStats& after,
                                     const CheckpointReadStats& before) {
    return {after.calls - before.calls, after.bytes - before.bytes,
            after.nanoseconds - before.nanoseconds,
            after.wall_nanoseconds - before.wall_nanoseconds};
}

std::vector<std::uint64_t> counter_delta(const std::vector<std::uint64_t>& after,
                                         const std::vector<std::uint64_t>& before) {
    std::vector<std::uint64_t> result;
    result.reserve(after.size());
    for (std::size_t index = 0; index < after.size(); ++index) {
        result.push_back(after[index] - (index < before.size() ? before[index] : 0U));
    }
    return result;
}

LagunaCacheStats cache_delta(const LagunaCacheStats& after,
                             const LagunaCacheStats& before) {
    LagunaCacheStats result = after;
    result.hits -= before.hits;
    result.misses -= before.misses;
    result.evictions -= before.evictions;
    result.device_hits = counter_delta(after.device_hits, before.device_hits);
    result.device_misses = counter_delta(after.device_misses, before.device_misses);
    result.device_evictions =
        counter_delta(after.device_evictions, before.device_evictions);
    result.device_lock_nanoseconds =
        counter_delta(after.device_lock_nanoseconds, before.device_lock_nanoseconds);
    result.device_stage_nanoseconds =
        counter_delta(after.device_stage_nanoseconds, before.device_stage_nanoseconds);
    result.device_matmul_nanoseconds =
        counter_delta(after.device_matmul_nanoseconds, before.device_matmul_nanoseconds);
    return result;
}

LagunaGraphStats graph_delta(const LagunaGraphStats& after,
                             const LagunaGraphStats& before) noexcept {
    return {
        after.forward_tokens - before.forward_tokens,
        after.embedding_nanoseconds - before.embedding_nanoseconds,
        after.attention_nanoseconds - before.attention_nanoseconds,
        after.dense_mlp_nanoseconds - before.dense_mlp_nanoseconds,
        after.moe_router_nanoseconds - before.moe_router_nanoseconds,
        after.moe_routed_nanoseconds - before.moe_routed_nanoseconds,
        after.moe_shared_nanoseconds - before.moe_shared_nanoseconds,
        after.output_head_nanoseconds - before.output_head_nanoseconds,
        after.attention_projection_nanoseconds -
            before.attention_projection_nanoseconds,
        after.attention_rope_nanoseconds - before.attention_rope_nanoseconds,
        after.attention_kv_stage_nanoseconds -
            before.attention_kv_stage_nanoseconds,
        after.attention_flash_nanoseconds - before.attention_flash_nanoseconds,
        after.attention_output_nanoseconds - before.attention_output_nanoseconds,
        after.moe_gather_nanoseconds - before.moe_gather_nanoseconds,
        after.moe_expert_nanoseconds - before.moe_expert_nanoseconds,
        after.moe_expert_cache_nanoseconds - before.moe_expert_cache_nanoseconds,
        after.moe_accumulate_nanoseconds - before.moe_accumulate_nanoseconds,
    };
}

// LRU cache of routed-expert projections in device memory. The spine is not
// managed here: it is uploaded once and its bytes are removed from the
// capacity this cache is given, so an expert can never evict an attention
// projection.
class ExpertCache {
    struct Entry {
        CudaWeight weight;
        std::uint64_t last_use{};
        std::uint64_t leases{};
    };
    struct State {
        mutable std::mutex mutex;
        std::unordered_map<std::uint64_t, Entry> entries;
        std::uint64_t capacity{};
        std::uint64_t used{};
        std::uint64_t peak{};
        std::uint64_t clock{};
        std::uint64_t hits{};
        std::uint64_t misses{};
        std::uint64_t evictions{};
        std::uint64_t stage_nanoseconds{};
        std::uint64_t matmul_nanoseconds{};
        std::uint64_t lock_nanoseconds{};
    };

public:
    ExpertCache(const LagunaCheckpointReader& checkpoint, CudaBackend& backend,
                std::vector<int> devices, std::vector<std::uint64_t> capacities)
        : checkpoint_(checkpoint), backend_(backend),
          devices_(std::move(devices)) {
        states_.reserve(capacities.size());
        for (const auto capacity : capacities) {
            auto state = std::make_unique<State>();
            state->capacity = capacity;
            states_.push_back(std::move(state));
        }
    }

    ValidationResult matmul(std::size_t device_slot, std::uint64_t key,
                            const LagunaLinear& module,
                            std::span<const float> input, std::uint32_t rows,
                            std::span<float> output) {
        ValidationResult result;
        if (device_slot >= states_.size()) {
            result.errors.emplace_back("expert targets an invalid runtime device slot");
            return result;
        }
        auto& state = *states_[device_slot];
        const auto lock_started = std::chrono::steady_clock::now();
        std::scoped_lock lock(state.mutex);
        const auto lock_nanoseconds = elapsed_nanoseconds(lock_started);
        state.lock_nanoseconds += lock_nanoseconds;
        Entry* entry = nullptr;
        const auto stage_started = std::chrono::steady_clock::now();
        result = ensure_locked(state, device_slot, key, module, entry);
        state.stage_nanoseconds += elapsed_nanoseconds(stage_started);
        if (!result.ok()) return result;
        const auto matmul_started = std::chrono::steady_clock::now();
        result = backend_.matmul(entry->weight, input, rows, output);
        state.matmul_nanoseconds += elapsed_nanoseconds(matmul_started);
        return result;
    }

    // Makes one expert's three projections resident and leases them, so a
    // batched device command can hold raw weight pointers across the enqueue
    // and collect pair without an eviction pulling storage out from under it.
    // Every successful acquire must be matched by one release.
    ValidationResult acquire(std::size_t device_slot, std::uint32_t layer,
                             const ExpertModules& modules,
                             std::uint32_t expert, CudaMoeExpert& output) {
        ValidationResult result;
        if (device_slot >= states_.size()) {
            result.errors.emplace_back("expert targets an invalid runtime device slot");
            return result;
        }
        auto& state = *states_[device_slot];
        std::scoped_lock lock(state.mutex);
        const std::array<const LagunaLinear*, 3> sources{
            &modules.gate, &modules.up, &modules.down};
        const CudaWeight* resolved[3]{};
        const auto stage_started = std::chrono::steady_clock::now();
        for (std::uint32_t index = 0U; index < 3U; ++index) {
            Entry* entry = nullptr;
            const auto key = (static_cast<std::uint64_t>(layer) << 40U) |
                             (static_cast<std::uint64_t>(expert) << 8U) | index;
            result = ensure_locked(state, device_slot, key, *sources[index],
                                   entry);
            if (!result.ok()) {
                // Unwind the leases already taken for this expert.
                for (std::uint32_t taken = 0U; taken < index; ++taken) {
                    const auto unwind_key =
                        (static_cast<std::uint64_t>(layer) << 40U) |
                        (static_cast<std::uint64_t>(expert) << 8U) | taken;
                    auto found = state.entries.find(unwind_key);
                    if (found != state.entries.end() &&
                        found->second.leases != 0U) {
                        --found->second.leases;
                    }
                }
                state.stage_nanoseconds += elapsed_nanoseconds(stage_started);
                return result;
            }
            ++entry->leases;
            resolved[index] = &entry->weight;
        }
        state.stage_nanoseconds += elapsed_nanoseconds(stage_started);
        output.gate = resolved[0];
        output.up = resolved[1];
        output.down = resolved[2];
        output.coefficient = 1.0F;
        return result;
    }

    void release(std::size_t device_slot, std::uint32_t layer,
                 std::uint32_t expert) {
        if (device_slot >= states_.size()) return;
        auto& state = *states_[device_slot];
        std::scoped_lock lock(state.mutex);
        for (std::uint32_t index = 0U; index < 3U; ++index) {
            const auto key = (static_cast<std::uint64_t>(layer) << 40U) |
                             (static_cast<std::uint64_t>(expert) << 8U) | index;
            auto found = state.entries.find(key);
            if (found != state.entries.end() && found->second.leases != 0U) {
                --found->second.leases;
            }
        }
    }

    void add_matmul_nanoseconds(std::size_t device_slot,
                                std::uint64_t nanoseconds) {
        if (device_slot >= states_.size()) return;
        auto& state = *states_[device_slot];
        std::scoped_lock lock(state.mutex);
        state.matmul_nanoseconds += nanoseconds;
    }

    [[nodiscard]] LagunaCacheStats stats(
        std::span<const std::uint64_t> pinned) const {
        LagunaCacheStats result;
        result.hits = hits_.load(std::memory_order_relaxed);
        result.misses = misses_.load(std::memory_order_relaxed);
        result.evictions = evictions_.load(std::memory_order_relaxed);
        for (std::size_t slot = 0U; slot < states_.size(); ++slot) {
            const auto& state = *states_[slot];
            std::scoped_lock lock(state.mutex);
            const auto pinned_bytes = slot < pinned.size() ? pinned[slot] : 0U;
            result.used_bytes.push_back(state.used + pinned_bytes);
            result.peak_bytes.push_back(state.peak + pinned_bytes);
            result.capacity_bytes.push_back(state.capacity + pinned_bytes);
            result.pinned_resident_bytes.push_back(pinned_bytes);
            result.evictable_expert_bytes.push_back(state.used);
            result.device_hits.push_back(state.hits);
            result.device_misses.push_back(state.misses);
            result.device_evictions.push_back(state.evictions);
            result.device_lock_nanoseconds.push_back(state.lock_nanoseconds);
            result.device_stage_nanoseconds.push_back(state.stage_nanoseconds);
            result.device_matmul_nanoseconds.push_back(state.matmul_nanoseconds);
        }
        return result;
    }

private:
    // Drops the least recently used unleased entry. Returns false when nothing
    // is evictable, which is the only condition that can fail a lookup.
    bool evict_one_locked(State& state) {
        auto victim = state.entries.end();
        for (auto candidate = state.entries.begin();
             candidate != state.entries.end(); ++candidate) {
            if (candidate->second.leases != 0U) continue;
            if (victim == state.entries.end() ||
                candidate->second.last_use < victim->second.last_use) {
                victim = candidate;
            }
        }
        if (victim == state.entries.end()) return false;
        state.used -= victim->second.weight.device_bytes();
        state.entries.erase(victim);
        ++state.evictions;
        evictions_.fetch_add(1U, std::memory_order_relaxed);
        return true;
    }

    ValidationResult ensure_locked(State& state, std::size_t device_slot,
                                   std::uint64_t key,
                                   const LagunaLinear& module, Entry*& output) {
        ValidationResult result;
        ++state.clock;
        auto found = state.entries.find(key);
        if (found != state.entries.end()) {
            found->second.last_use = state.clock;
            ++state.hits;
            hits_.fetch_add(1U, std::memory_order_relaxed);
            output = &found->second;
            return result;
        }
        const auto needed = module.source_bytes();
        if (needed == 0U) {
            result.errors.emplace_back("routed expert projection is unresolved");
            return result;
        }
        if (needed > state.capacity) {
            result.errors.emplace_back(
                "routed expert exceeds the admitted VRAM cache on device " +
                std::to_string(devices_[device_slot]));
            return result;
        }
        while (state.used + needed > state.capacity) {
            if (!evict_one_locked(state)) {
                result.errors.emplace_back(
                    "in-flight routed experts exceed VRAM cache capacity on device " +
                    std::to_string(devices_[device_slot]));
                return result;
            }
        }
        // Logical capacity accounting is not sufficient on its own: the arena
        // is a free list and routed experts come in two sizes (1.69 MiB NVFP4
        // and 18.87 MiB plain BF16), so a request can fail against a fragmented
        // arena that still has enough total free bytes. Evict the next LRU
        // victim and retry rather than failing the token. This changes only
        // which weights are resident, never the arithmetic.
        Entry entry;
        ValidationResult loaded;
        for (;;) {
            loaded = load_laguna_cuda_linear(
                checkpoint_, module, devices_[device_slot], backend_,
                entry.weight);
            if (loaded.ok()) break;
            if (!evict_one_locked(state)) return loaded;
        }
        entry.last_use = state.clock;
        state.used += entry.weight.device_bytes();
        state.peak = std::max(state.peak, state.used);
        found = state.entries.emplace(key, std::move(entry)).first;
        ++state.misses;
        misses_.fetch_add(1U, std::memory_order_relaxed);
        output = &found->second;
        return result;
    }

    const LagunaCheckpointReader& checkpoint_;
    CudaBackend& backend_;
    std::vector<int> devices_;
    std::vector<std::unique_ptr<State>> states_;
    std::atomic<std::uint64_t> hits_{};
    std::atomic<std::uint64_t> misses_{};
    std::atomic<std::uint64_t> evictions_{};
};

}  // namespace

struct LagunaRuntime::Impl {
    LagunaRuntimeConfig config;
    std::unique_ptr<LagunaCheckpointReader> checkpoint;
    ModelTokenizer tokenizer;
    CudaBackend cuda;
    std::unique_ptr<ExpertCache> experts;
    std::unique_ptr<HostWorkerPool> device_workers;
    std::vector<int> devices;
    std::vector<std::size_t> device_schedule;
    std::vector<std::uint64_t> pinned_bytes;
    std::vector<LayerWeights> layers{kLayers};
    std::vector<LayerKv> kv{kLayers};
    SpineWeight output_head;
    std::vector<float> final_norm;
    std::unordered_map<std::uint32_t, std::vector<float>> embedding_rows;
    LagunaRopeSchedule global_rope;
    LagunaRopeSchedule sliding_rope;
    RouterSpec router;
    std::vector<std::uint32_t> cached_token_ids;
    RouteTraceWriter route_trace;
    std::mt19937_64 sampler;
    SamplingOptions active_sampling;
    std::vector<std::uint32_t> sampled_token_counts;
    std::vector<std::uint32_t> sampled_token_ids;
    TokenLogprob last_sample;
    LagunaGraphStats graph_stats;
    std::uint64_t active_request_id{};
    std::uint64_t generated_requests{};
    bool initialized{};
    bool reusable_sequence{};

    [[nodiscard]] std::size_t layer_device(std::uint32_t layer) const {
        return device_schedule[layer % device_schedule.size()];
    }

    [[nodiscard]] std::size_t expert_device(std::uint32_t layer,
                                            std::uint32_t expert) const {
        const auto index =
            (static_cast<std::size_t>(layer) * kExperts + expert) %
            device_schedule.size();
        return device_schedule[index];
    }

    [[nodiscard]] std::uint64_t expert_key(std::uint32_t layer,
                                           std::uint32_t expert,
                                           std::uint32_t projection) const noexcept {
        return (static_cast<std::uint64_t>(layer) << 40U) |
               (static_cast<std::uint64_t>(expert) << 8U) | projection;
    }

    ValidationResult spine_matmul(const SpineWeight& spine,
                                  std::span<const float> input,
                                  std::uint32_t rows, std::span<float> output) {
        ValidationResult result;
        if (!spine.weight.valid()) {
            result.errors.emplace_back("resident Laguna projection is not loaded");
            return result;
        }
        return cuda.matmul(spine.weight, input, rows, output);
    }

    ValidationResult load_spine_weight(std::size_t device,
                                       const std::string& base,
                                       std::uint64_t rows,
                                       std::uint64_t columns,
                                       SpineWeight& output) {
        ValidationResult result;
        auto module = checkpoint->linear(base, rows, columns);
        if (!module.ok()) {
            result.errors = std::move(module.errors);
            return result;
        }
        result = load_laguna_cuda_linear(*checkpoint, module.value,
                                         devices[device], cuda, output.weight);
        if (!result.ok()) return result;
        output.bytes = output.weight.device_bytes();
        pinned_bytes[device] += output.bytes;
        return result;
    }

    ValidationResult load_host_vector(const std::string& name,
                                      std::uint64_t elements,
                                      std::vector<float>& output) {
        ValidationResult result;
        auto loaded = checkpoint->read_f32(name, elements);
        if (!loaded.ok()) {
            result.errors = std::move(loaded.errors);
            return result;
        }
        if (loaded.value.size() != elements) {
            result.errors.emplace_back("Laguna host vector shape mismatch: " + name);
            return result;
        }
        output = std::move(loaded.value);
        return result;
    }

    ValidationResult load_layer(std::uint32_t layer) {
        ValidationResult result;
        auto& weights = layers[layer];
        weights.device = layer_device(layer);
        weights.global = laguna_global_attention_layer(layer);
        weights.heads = laguna_attention_heads(layer);
        const auto prefix = layer_prefix(layer);
        const auto attention = prefix + "self_attn.";
        const auto mlp = prefix + "mlp.";
        const auto device = weights.device;
        const auto query_columns =
            static_cast<std::uint64_t>(weights.heads) * kHeadDim;

        result = load_spine_weight(device, attention + "q_proj", query_columns,
                                   kHidden, weights.query);
        if (!result.ok()) return result;
        result = load_spine_weight(device, attention + "k_proj", kKvColumns,
                                   kHidden, weights.key);
        if (!result.ok()) return result;
        result = load_spine_weight(device, attention + "v_proj", kKvColumns,
                                   kHidden, weights.value);
        if (!result.ok()) return result;
        result = load_spine_weight(device, attention + "o_proj", kHidden,
                                   query_columns, weights.output);
        if (!result.ok()) return result;
        result = load_spine_weight(device, attention + "g_proj", weights.heads,
                                   kHidden, weights.output_gate);
        if (!result.ok()) return result;

        result = load_host_vector(prefix + "input_layernorm.weight", kHidden,
                                  weights.input_norm);
        if (!result.ok()) return result;
        result = load_host_vector(prefix + "post_attention_layernorm.weight",
                                  kHidden, weights.post_attention_norm);
        if (!result.ok()) return result;
        result = load_host_vector(attention + "q_norm.weight", kHeadDim,
                                  weights.query_norm);
        if (!result.ok()) return result;
        result = load_host_vector(attention + "k_norm.weight", kHeadDim,
                                  weights.key_norm);
        if (!result.ok()) return result;

        if (!laguna_sparse_layer(layer)) {
            result = load_spine_weight(device, mlp + "gate_proj",
                                       kDenseIntermediate, kHidden,
                                       weights.dense_gate);
            if (!result.ok()) return result;
            result = load_spine_weight(device, mlp + "up_proj",
                                       kDenseIntermediate, kHidden,
                                       weights.dense_up);
            if (!result.ok()) return result;
            return load_spine_weight(device, mlp + "down_proj", kHidden,
                                     kDenseIntermediate, weights.dense_down);
        }

        result = load_spine_weight(device, mlp + "gate", kExperts, kHidden,
                                   weights.router);
        if (!result.ok()) return result;
        result = load_spine_weight(device, mlp + "shared_expert.gate_proj",
                                   kSharedIntermediate, kHidden,
                                   weights.shared_gate);
        if (!result.ok()) return result;
        result = load_spine_weight(device, mlp + "shared_expert.up_proj",
                                   kSharedIntermediate, kHidden,
                                   weights.shared_up);
        if (!result.ok()) return result;
        result = load_spine_weight(device, mlp + "shared_expert.down_proj",
                                   kHidden, kSharedIntermediate,
                                   weights.shared_down);
        if (!result.ok()) return result;
        result = load_host_vector(mlp + "experts.e_score_correction_bias",
                                  kExperts, weights.correction_bias);
        if (!result.ok()) return result;

        weights.experts.resize(kExperts);
        for (std::uint32_t expert = 0U; expert < kExperts; ++expert) {
            const auto base = mlp + "experts." + std::to_string(expert) + ".";
            auto gate = checkpoint->linear(base + "gate_proj",
                                           kExpertIntermediate, kHidden);
            auto up = checkpoint->linear(base + "up_proj", kExpertIntermediate,
                                         kHidden);
            auto down = checkpoint->linear(base + "down_proj", kHidden,
                                           kExpertIntermediate);
            for (auto* resolved : {&gate, &up, &down}) {
                if (!resolved->ok()) {
                    result.errors = resolved->errors;
                    return result;
                }
            }
            weights.experts[expert] = {gate.value, up.value, down.value,
                                       expert_device(layer, expert)};
        }
        return result;
    }

    ValidationResult warmup() {
        ValidationResult result;
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            result = load_layer(layer);
            if (!result.ok()) return result;
            if (config.verbose || config.load_progress) {
                std::cerr << "[load] Laguna resident spine layer " << (layer + 1U)
                          << '/' << kLayers << '\n';
            }
        }
        result = load_spine_weight(layer_device(kLayers - 1U), "lm_head",
                                   kVocabulary, kHidden, output_head);
        if (!result.ok()) return result;
        return load_host_vector("model.norm.weight", kHidden, final_norm);
    }

    ValidationResult embed(std::span<const std::uint32_t> token_ids,
                           std::span<float> output) {
        ValidationResult result;
        if (output.size() != token_ids.size() * kHidden) {
            result.errors.emplace_back("Laguna embedding output shape mismatch");
            return result;
        }
        for (std::size_t row = 0U; row < token_ids.size(); ++row) {
            if (token_ids[row] >= kVocabulary) {
                result.errors.emplace_back("token id exceeds the model vocabulary");
                return result;
            }
            auto found = embedding_rows.find(token_ids[row]);
            if (found == embedding_rows.end()) {
                auto loaded = checkpoint->read_f32_row("model.embed_tokens.weight",
                                                       token_ids[row]);
                if (!loaded.ok()) {
                    result.errors = std::move(loaded.errors);
                    return result;
                }
                found = embedding_rows.emplace(token_ids[row],
                                               std::move(loaded.value)).first;
            }
            std::copy(found->second.begin(), found->second.end(),
                      output.begin() + static_cast<std::ptrdiff_t>(row * kHidden));
        }
        return result;
    }

    ValidationResult norm_rows(std::span<float> output,
                               std::span<const float> input,
                               std::span<const float> weight,
                               std::uint32_t rows, std::uint32_t columns) {
        ValidationResult result;
        if (output.size() != input.size() ||
            output.size() != static_cast<std::size_t>(rows) * columns) {
            result.errors.emplace_back("Laguna RMSNorm row shape mismatch");
            return result;
        }
        for (std::uint32_t row = 0U; row < rows; ++row) {
            result = rms_norm_f32(
                output.subspan(static_cast<std::size_t>(row) * columns, columns),
                input.subspan(static_cast<std::size_t>(row) * columns, columns),
                weight, kContract.rms_epsilon);
            if (!result.ok()) return result;
        }
        return result;
    }

    ValidationResult attention(std::uint32_t layer,
                               std::span<const float> input,
                               std::uint32_t rows, std::uint32_t position_base,
                               std::span<float> output) {
        ValidationResult result;
        auto& weights = layers[layer];
        auto& cache = kv[layer];
        const auto heads = weights.heads;
        const auto query_columns = static_cast<std::size_t>(heads) * kHeadDim;
        const auto& schedule = weights.global ? global_rope : sliding_rope;

        std::vector<float> queries(static_cast<std::size_t>(rows) * query_columns);
        std::vector<float> new_keys(static_cast<std::size_t>(rows) * kKvColumns);
        std::vector<float> new_values(new_keys.size());
        std::vector<float> gates(static_cast<std::size_t>(rows) * heads);
        auto stage_started = std::chrono::steady_clock::now();
        result = spine_matmul(weights.query, input, rows, queries);
        if (!result.ok()) return result;
        result = spine_matmul(weights.key, input, rows, new_keys);
        if (!result.ok()) return result;
        result = spine_matmul(weights.value, input, rows, new_values);
        if (!result.ok()) return result;
        result = spine_matmul(weights.output_gate, input, rows, gates);
        if (!result.ok()) return result;
        graph_stats.attention_projection_nanoseconds +=
            elapsed_nanoseconds(stage_started);
        stage_started = std::chrono::steady_clock::now();

        // QK RMSNorm runs per head before RoPE; values are not normalized.
        for (std::uint32_t row = 0U; row < rows; ++row) {
            for (std::uint32_t head = 0U; head < heads; ++head) {
                auto query = std::span<float>(queries).subspan(
                    (static_cast<std::size_t>(row) * heads + head) * kHeadDim,
                    kHeadDim);
                result = rms_norm_f32(query, std::span<const float>(query),
                                      weights.query_norm, kContract.rms_epsilon);
                if (!result.ok()) return result;
                result = laguna_rope_half_f32(query, position_base + row, schedule);
                if (!result.ok()) return result;
            }
            for (std::uint32_t head = 0U; head < kKvHeads; ++head) {
                auto key = std::span<float>(new_keys).subspan(
                    (static_cast<std::size_t>(row) * kKvHeads + head) * kHeadDim,
                    kHeadDim);
                result = rms_norm_f32(key, std::span<const float>(key),
                                      weights.key_norm, kContract.rms_epsilon);
                if (!result.ok()) return result;
                result = laguna_rope_half_f32(key, position_base + row, schedule);
                if (!result.ok()) return result;
            }
        }

        graph_stats.attention_rope_nanoseconds +=
            elapsed_nanoseconds(stage_started);
        stage_started = std::chrono::steady_clock::now();

        const auto cached_rows = cache.keys.size() / kKvColumns;
        if (cache.values.size() != cache.keys.size() ||
            cache.start + cached_rows != position_base) {
            result.errors.emplace_back("Laguna KV cache is not contiguous at layer " +
                                       std::to_string(layer));
            return result;
        }
        // Append this step's rows to the cache and attend over the cache
        // itself. The rows are rounded through BF16 on the way in, so the
        // arithmetic is what a BF16 cache would produce.
        cache.keys.reserve(cache.keys.size() + new_keys.size());
        cache.values.reserve(cache.values.size() + new_values.size());
        for (const auto value : new_keys) cache.keys.push_back(bf16_round(value));
        for (const auto value : new_values) cache.values.push_back(bf16_round(value));

        const std::array<FlashAttentionSegment, 1> segments{
            {{cache.keys, cache.values, {}}}};
        FlashAttentionRequest request;
        request.queries = queries;
        request.segments = segments;
        request.query_rows = rows;
        request.query_heads = heads;
        request.key_value_heads = kKvHeads;
        request.query_key_dim = kHeadDim;
        request.value_dim = kHeadDim;
        request.scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
        request.numerics = FlashAttentionNumerics::f32_dot_f32_softmax_f32_accum;
        request.maximum_workspace_bytes = kFlashAttentionWorkspaceReserve;
        std::vector<std::uint32_t> causal;
        std::vector<std::uint8_t> mask;
        if (weights.global) {
            causal.resize(rows);
            for (std::uint32_t row = 0U; row < rows; ++row) {
                causal[row] = position_base + row + 1U - cache.start;
            }
            request.causal_key_counts = causal;
        } else {
            const auto total = cached_rows + rows;
            mask.resize(static_cast<std::size_t>(rows) * total);
            for (std::uint32_t row = 0U; row < rows; ++row) {
                const auto query = static_cast<std::uint64_t>(position_base + row);
                for (std::size_t key_row = 0U; key_row < total; ++key_row) {
                    mask[static_cast<std::size_t>(row) * total + key_row] =
                        static_cast<std::uint8_t>(laguna_attention_visible(
                            query, cache.start + key_row, true, kSlidingWindow));
                }
            }
            request.query_key_mask = mask;
        }
        std::vector<float> context(static_cast<std::size_t>(rows) * query_columns);
        graph_stats.attention_kv_stage_nanoseconds +=
            elapsed_nanoseconds(stage_started);
        stage_started = std::chrono::steady_clock::now();
        result = config.enable_flash_attention
            ? cuda.flash_attention(devices[weights.device], request, context)
            : flash_attention_reference_f32(request, context);
        graph_stats.attention_flash_nanoseconds +=
            elapsed_nanoseconds(stage_started);
        if (!result.ok()) return result;
        stage_started = std::chrono::steady_clock::now();

        // Laguna gates the attention output before o_proj: one softplus gate
        // per head, broadcast across head_dim.
        for (std::uint32_t row = 0U; row < rows; ++row) {
            for (std::uint32_t head = 0U; head < heads; ++head) {
                const float gate = laguna_softplus_f32(
                    gates[static_cast<std::size_t>(row) * heads + head]);
                auto slice = std::span<float>(context).subspan(
                    (static_cast<std::size_t>(row) * heads + head) * kHeadDim,
                    kHeadDim);
                for (auto& value : slice) value *= gate;
            }
        }
        result = spine_matmul(weights.output, context, rows, output);
        if (!result.ok()) return result;

        if (!weights.global && cached_rows + rows > kSlidingWindow) {
            const auto drop_rows = cached_rows + rows - kSlidingWindow;
            const auto drop = drop_rows * kKvColumns;
            cache.keys.erase(cache.keys.begin(),
                             cache.keys.begin() + static_cast<std::ptrdiff_t>(drop));
            cache.values.erase(
                cache.values.begin(),
                cache.values.begin() + static_cast<std::ptrdiff_t>(drop));
            cache.start += static_cast<std::uint32_t>(drop_rows);
        }
        graph_stats.attention_output_nanoseconds +=
            elapsed_nanoseconds(stage_started);
        return result;
    }

    ValidationResult swiglu_triplet(const SpineWeight& gate_projection,
                                    const SpineWeight& up_projection,
                                    const SpineWeight& down_projection,
                                    std::uint32_t intermediate,
                                    std::span<const float> input,
                                    std::uint32_t rows,
                                    std::span<float> output) {
        ValidationResult result;
        std::vector<float> gate(static_cast<std::size_t>(rows) * intermediate);
        std::vector<float> up(gate.size());
        result = spine_matmul(gate_projection, input, rows, gate);
        if (!result.ok()) return result;
        result = spine_matmul(up_projection, input, rows, up);
        if (!result.ok()) return result;
        for (std::size_t index = 0U; index < gate.size(); ++index) {
            gate[index] = silu_f32(gate[index]) * up[index];
        }
        return spine_matmul(down_projection, gate, rows, output);
    }

    ValidationResult run_expert(std::uint32_t layer, const ExpertModules& modules,
                                std::uint32_t expert,
                                std::span<const float> input, std::uint32_t rows,
                                std::span<float> output,
                                std::uint64_t& cache_nanoseconds) {
        ValidationResult result;
        const auto allocation_started = std::chrono::steady_clock::now();
        std::vector<float> gate(static_cast<std::size_t>(rows) * kExpertIntermediate);
        std::vector<float> up(gate.size());
        const auto allocation_nanoseconds =
            elapsed_nanoseconds(allocation_started);
        const auto cache_started = std::chrono::steady_clock::now();
        result = experts->matmul(modules.device, expert_key(layer, expert, 0U),
                                 modules.gate, input, rows, gate);
        if (!result.ok()) return result;
        result = experts->matmul(modules.device, expert_key(layer, expert, 1U),
                                 modules.up, input, rows, up);
        if (!result.ok()) return result;
        for (std::size_t index = 0U; index < gate.size(); ++index) {
            gate[index] = silu_f32(gate[index]) * up[index];
        }
        result = experts->matmul(modules.device, expert_key(layer, expert, 2U),
                                 modules.down, gate, rows, output);
        cache_nanoseconds +=
            elapsed_nanoseconds(cache_started) + allocation_nanoseconds;
        return result;
    }

    // Runs every routed expert this device owns for one decode row as a single
    // device command. Weights are leased across the enqueue/collect pair so an
    // eviction cannot free storage the in-flight kernels still point at.
    ValidationResult run_batched_experts(std::uint32_t layer,
                                         std::size_t slot,
                                         std::span<const float> input,
                                         std::vector<ExpertJob>& jobs) {
        ValidationResult result;
        auto& weights = layers[layer];
        std::vector<std::size_t> indices;
        indices.reserve(kTopK);
        for (std::size_t index = 0U; index < jobs.size(); ++index) {
            if (jobs[index].device == slot) indices.push_back(index);
        }
        if (indices.empty()) return result;

        const auto started = std::chrono::steady_clock::now();
        std::vector<CudaMoeExpert> batch(indices.size());
        std::vector<std::uint32_t> acquired;
        acquired.reserve(indices.size());
        const auto release_all = [&]() {
            for (const auto expert : acquired) experts->release(slot, layer, expert);
        };
        for (std::size_t position = 0U; position < indices.size(); ++position) {
            const auto& job = jobs[indices[position]];
            result = experts->acquire(slot, layer, weights.experts[job.expert],
                                      job.expert, batch[position]);
            if (!result.ok()) {
                release_all();
                return result;
            }
            acquired.push_back(job.expert);
        }

        const auto command_started = std::chrono::steady_clock::now();
        result = cuda.enqueue_moe(devices[slot], input.first(kHidden), 1U, batch,
                                  nullptr);
        if (!result.ok()) {
            release_all();
            return result;
        }
        std::vector<float> collected(indices.size() * kHidden);
        result = cuda.collect_moe(devices[slot], collected, {});
        release_all();
        experts->add_matmul_nanoseconds(slot, elapsed_nanoseconds(command_started));
        if (!result.ok()) return result;

        for (std::size_t position = 0U; position < indices.size(); ++position) {
            auto& job = jobs[indices[position]];
            job.output.assign(kHidden, 0.0F);
            std::copy_n(collected.begin() +
                            static_cast<std::ptrdiff_t>(position * kHidden),
                        kHidden, job.output.begin());
            job.expert_nanoseconds += elapsed_nanoseconds(started) / indices.size();
        }
        return result;
    }

    bool write_route(std::uint32_t position, std::uint32_t layer,
                     const LagunaRoute& route, bool prefill) {
        if (!route_trace.is_open()) return true;
        RouteEvent event;
        event.request = active_request_id;
        event.token_position = position;
        event.layer = layer;
        event.experts = route.experts;
        event.coefficients = route.weights;
        event.phase = prefill ? RoutePhase::Prefill : RoutePhase::Decode;
        return route_trace.write(event).ok();
    }

    ValidationResult sparse_mlp(std::uint32_t layer,
                                std::span<const float> input,
                                std::uint32_t rows, std::uint32_t position_base,
                                bool prefill, std::span<float> output) {
        ValidationResult result;
        auto& weights = layers[layer];
        const auto router_started = std::chrono::steady_clock::now();
        std::vector<float> logits(static_cast<std::size_t>(rows) * kExperts);
        result = spine_matmul(weights.router, input, rows, logits);
        if (!result.ok()) return result;
        std::vector<LagunaRoute> routes(rows);
        for (std::uint32_t row = 0U; row < rows; ++row) {
            auto routed = laguna_route_sigmoid_topk(
                std::span<const float>(logits).subspan(
                    static_cast<std::size_t>(row) * kExperts, kExperts),
                weights.correction_bias, router,
                kContract.router_logit_softcapping);
            if (!routed.ok()) {
                result.errors = std::move(routed.errors);
                return result;
            }
            routes[row] = std::move(routed.value);
            if (!write_route(position_base + row, layer, routes[row], prefill)) {
                result.errors.emplace_back("cannot write Laguna route trace");
                return result;
            }
        }
        graph_stats.moe_router_nanoseconds += elapsed_nanoseconds(router_started);

        std::array<std::int32_t, kExperts> job_by_expert{};
        job_by_expert.fill(-1);
        std::vector<ExpertJob> jobs;
        jobs.reserve(std::min<std::size_t>(
            static_cast<std::size_t>(rows) * kTopK, kExperts));
        for (std::uint32_t row = 0U; row < rows; ++row) {
            for (std::uint32_t rank = 0U; rank < kTopK; ++rank) {
                const auto expert = routes[row].experts[rank];
                if (job_by_expert[expert] < 0) {
                    job_by_expert[expert] =
                        static_cast<std::int32_t>(jobs.size());
                    ExpertJob job;
                    job.expert = expert;
                    job.device = weights.experts[expert].device;
                    jobs.push_back(std::move(job));
                }
                auto& job = jobs[static_cast<std::size_t>(job_by_expert[expert])];
                job.rows.push_back(row);
                job.weights.push_back(routes[row].weights[rank]);
            }
        }

        const auto routed_started = std::chrono::steady_clock::now();
        std::vector<std::vector<std::string>> device_errors(devices.size());
        // Batch decode. Every routed expert of a step sees the same single
        // hidden row, so one device command covers all of a device's experts:
        // one host-to-device copy of the row, two kernel launches, one copy
        // back, instead of three synchronous round trips per expert. At
        // rows > 1 the experts see disjoint row subsets, which this command
        // shape cannot express, so prefill keeps the per-expert path.
        const bool batched = rows == 1U;
        auto dispatched = batched
            ? device_workers->parallel_for(devices.size(), [&](std::size_t slot) {
                  auto status = run_batched_experts(layer, slot, input, jobs);
                  if (!status.ok()) move_errors(device_errors[slot], std::move(status));
              })
            : device_workers->parallel_for(
            devices.size(), [&](std::size_t slot) {
            for (auto& job : jobs) {
                if (job.device != slot) continue;
                const auto job_rows = static_cast<std::uint32_t>(job.rows.size());
                const auto gather_started = std::chrono::steady_clock::now();
                std::vector<float> gathered(
                    static_cast<std::size_t>(job_rows) * kHidden);
                for (std::size_t row = 0U; row < job.rows.size(); ++row) {
                    const auto source = input.subspan(
                        static_cast<std::size_t>(job.rows[row]) * kHidden, kHidden);
                    std::copy(source.begin(), source.end(),
                              gathered.begin() +
                                  static_cast<std::ptrdiff_t>(row * kHidden));
                }
                job.output.assign(gathered.size(), 0.0F);
                job.gather_nanoseconds += elapsed_nanoseconds(gather_started);
                const auto expert_started = std::chrono::steady_clock::now();
                auto status = run_expert(layer, weights.experts[job.expert],
                                         job.expert, gathered, job_rows,
                                         job.output, job.cache_nanoseconds);
                job.expert_nanoseconds += elapsed_nanoseconds(expert_started);
                if (!status.ok()) {
                    move_errors(device_errors[slot], std::move(status));
                    return;
                }
            }
        });
        graph_stats.moe_routed_nanoseconds += elapsed_nanoseconds(routed_started);
        if (!dispatched.ok()) return dispatched;
        for (auto& errors : device_errors) {
            result.errors.insert(result.errors.end(),
                                 std::make_move_iterator(errors.begin()),
                                 std::make_move_iterator(errors.end()));
        }
        if (!result.ok()) return result;
        // Per-device work runs concurrently, so the wall cost of a sub-phase is
        // the slowest device's share of it, not the sum across devices.
        std::vector<std::uint64_t> device_gather(devices.size());
        std::vector<std::uint64_t> device_expert(devices.size());
        std::vector<std::uint64_t> device_cache(devices.size());
        for (const auto& job : jobs) {
            device_gather[job.device] += job.gather_nanoseconds;
            device_expert[job.device] += job.expert_nanoseconds;
            device_cache[job.device] += job.cache_nanoseconds;
        }
        graph_stats.moe_expert_cache_nanoseconds +=
            *std::max_element(device_cache.begin(), device_cache.end());
        graph_stats.moe_gather_nanoseconds +=
            *std::max_element(device_gather.begin(), device_gather.end());
        graph_stats.moe_expert_nanoseconds +=
            *std::max_element(device_expert.begin(), device_expert.end());

        const auto accumulate_started = std::chrono::steady_clock::now();
        std::fill(output.begin(), output.end(), 0.0F);
        for (const auto& job : jobs) {
            for (std::size_t local = 0U; local < job.rows.size(); ++local) {
                auto destination = output.subspan(
                    static_cast<std::size_t>(job.rows[local]) * kHidden, kHidden);
                const auto* source = job.output.data() + local * kHidden;
                for (std::uint32_t column = 0U; column < kHidden; ++column) {
                    destination[column] += source[column] * job.weights[local];
                }
            }
        }
        // The reference scales the summed routed output, not the individual
        // routing weights, and only the routed branch is scaled.
        for (auto& value : output) value *= kContract.routed_scale;
        graph_stats.moe_accumulate_nanoseconds +=
            elapsed_nanoseconds(accumulate_started);

        const auto shared_started = std::chrono::steady_clock::now();
        std::vector<float> shared(output.size());
        result = swiglu_triplet(weights.shared_gate, weights.shared_up,
                                weights.shared_down, kSharedIntermediate, input,
                                rows, shared);
        graph_stats.moe_shared_nanoseconds += elapsed_nanoseconds(shared_started);
        if (!result.ok()) return result;
        for (std::size_t index = 0U; index < output.size(); ++index) {
            output[index] += shared[index];
        }
        return result;
    }

    ValidationResult forward_layers(std::span<float> hidden, std::uint32_t rows,
                                    std::uint32_t position_base, bool prefill) {
        ValidationResult result;
        std::vector<float> normalized(hidden.size());
        std::vector<float> branch(hidden.size());
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            const auto started = std::chrono::steady_clock::now();
            auto& weights = layers[layer];
            result = norm_rows(normalized, hidden, weights.input_norm, rows,
                               kHidden);
            if (!result.ok()) return result;
            auto phase_started = std::chrono::steady_clock::now();
            result = attention(layer, normalized, rows, position_base, branch);
            graph_stats.attention_nanoseconds += elapsed_nanoseconds(phase_started);
            if (!result.ok()) return result;
            for (std::size_t index = 0U; index < hidden.size(); ++index) {
                hidden[index] += branch[index];
            }
            result = norm_rows(normalized, hidden, weights.post_attention_norm,
                               rows, kHidden);
            if (!result.ok()) return result;
            phase_started = std::chrono::steady_clock::now();
            if (!laguna_sparse_layer(layer)) {
                result = swiglu_triplet(weights.dense_gate, weights.dense_up,
                                        weights.dense_down, kDenseIntermediate,
                                        normalized, rows, branch);
                graph_stats.dense_mlp_nanoseconds +=
                    elapsed_nanoseconds(phase_started);
            } else {
                result = sparse_mlp(layer, normalized, rows, position_base,
                                    prefill, branch);
            }
            if (!result.ok()) return result;
            for (std::size_t index = 0U; index < hidden.size(); ++index) {
                hidden[index] += branch[index];
            }
            if (config.verbose) {
                std::cerr << '[' << (rows > 1U ? "prefill" : "decode")
                          << "] Laguna layer " << (layer + 1U) << '/' << kLayers
                          << " rows=" << rows << " position=" << position_base
                          << " seconds="
                          << std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - started).count()
                          << '\n';
            }
        }
        return result;
    }

    ValidationResult compute_logits(std::span<const std::uint32_t> token_ids,
                                    std::uint32_t position_base, bool prefill,
                                    std::vector<float>& logits) {
        ValidationResult result;
        const auto rows = static_cast<std::uint32_t>(token_ids.size());
        if (rows == 0U || position_base + rows > config.maximum_context_tokens) {
            result.errors.emplace_back(
                "forward pass exceeds the configured context ceiling");
            return result;
        }
        graph_stats.forward_tokens += rows;
        std::vector<float> hidden(static_cast<std::size_t>(rows) * kHidden);
        auto phase_started = std::chrono::steady_clock::now();
        result = embed(token_ids, hidden);
        graph_stats.embedding_nanoseconds += elapsed_nanoseconds(phase_started);
        if (!result.ok()) return result;
        result = forward_layers(hidden, rows, position_base, prefill);
        if (!result.ok()) return result;
        phase_started = std::chrono::steady_clock::now();
        std::vector<float> normalized(kHidden);
        result = rms_norm_f32(normalized, std::span<const float>(hidden).last(kHidden),
                              final_norm, kContract.rms_epsilon);
        if (!result.ok()) return result;
        logits.assign(kVocabulary, 0.0F);
        result = spine_matmul(output_head, normalized, 1U, logits);
        graph_stats.output_head_nanoseconds += elapsed_nanoseconds(phase_started);
        return result;
    }

    ParseResult<std::uint32_t> forward(std::span<const std::uint32_t> token_ids,
                                       std::uint32_t position_base, bool prefill) {
        ParseResult<std::uint32_t> result;
        std::vector<float> logits;
        auto status = compute_logits(token_ids, position_base, prefill, logits);
        if (!status.ok()) {
            result.errors = std::move(status.errors);
            return result;
        }
        last_sample = sample_logits(
            logits, active_sampling,
            SamplingHistory{sampled_token_counts, sampled_token_ids}, sampler, {});
        if (!last_sample.ok()) {
            result.errors = last_sample.errors;
            return result;
        }
        result.value = last_sample.token;
        ++sampled_token_counts[result.value];
        sampled_token_ids.push_back(result.value);
        return result;
    }

    void reset_sequence() {
        reusable_sequence = false;
        cached_token_ids.clear();
        for (auto& layer : kv) {
            layer.keys.clear();
            layer.values.clear();
            layer.start = 0U;
        }
    }

    [[nodiscard]] LagunaCacheStats cache_stats() const {
        return experts == nullptr ? LagunaCacheStats{}
                                  : experts->stats(pinned_bytes);
    }
};

LagunaRuntime::LagunaRuntime() : impl_(std::make_unique<Impl>()) {}
LagunaRuntime::~LagunaRuntime() = default;
LagunaRuntime::LagunaRuntime(LagunaRuntime&&) noexcept = default;
LagunaRuntime& LagunaRuntime::operator=(LagunaRuntime&&) noexcept = default;

ValidationResult LagunaRuntime::initialize(const std::string& model_directory,
                                           const LagunaRuntimeConfig& config) {
    ValidationResult result;
    if (impl_->initialized) {
        result.errors.emplace_back("Laguna runtime is already initialized");
        return result;
    }
    result = validate_common_runtime_config(config.devices,
                                            config.vram_cache_fraction,
                                            config.sampling_temperature, "Laguna");
    if (!result.ok()) return result;
    if (config.maximum_context_tokens == 0U ||
        config.maximum_context_tokens > kContract.maximum_context_tokens) {
        result.errors.emplace_back(
            "Laguna runtime context exceeds the declared model ceiling");
        return result;
    }
    const auto spec = laguna_s21_nvfp4_spec();
    result = validate_laguna_s21_nvfp4(spec);
    if (!result.ok()) return result;

    impl_ = std::make_unique<Impl>();
    auto checkpoint = LagunaCheckpointReader::open(model_directory);
    if (!checkpoint.ok()) {
        result.errors = std::move(checkpoint.errors);
        return result;
    }
    auto tokenizer = ModelTokenizer::load(
        (std::filesystem::path(model_directory) / "tokenizer.json").string());
    if (!tokenizer.ok()) {
        result.errors = std::move(tokenizer.errors);
        return result;
    }
    result = impl_->cuda.initialize(config.devices, config.detailed_cuda_timing);
    if (!result.ok()) return result;
    if (config.enable_flash_attention) {
        for (const int device : config.devices) {
            result = impl_->cuda.validate_flash_attention_device(device);
            if (!result.ok()) return result;
        }
    }
    if (!config.route_trace_path.empty()) {
        result = impl_->route_trace.open(config.route_trace_path);
        if (!result.ok()) return result;
    }
    auto device_plan = plan_runtime_devices(
        config.devices, config.vram_cache_fraction,
        config.enable_flash_attention ? kFlashAttentionWorkspaceReserve : 0U,
        2ULL << 30U, "Laguna");
    if (!device_plan.ok()) {
        result.errors = std::move(device_plan.errors);
        return result;
    }

    impl_->config = config;
    impl_->sampler.seed(config.sampling_seed);
    impl_->checkpoint = std::move(checkpoint.value);
    impl_->tokenizer = std::move(tokenizer.value);
    impl_->devices = config.devices;
    impl_->device_schedule = std::move(device_plan.value.weighted_schedule);
    impl_->pinned_bytes.assign(config.devices.size(), 0U);
    impl_->global_rope = laguna_rope_schedule(true);
    impl_->sliding_rope = laguna_rope_schedule(false);
    impl_->router = spec.router;
    impl_->device_workers =
        std::make_unique<HostWorkerPool>(config.devices.size());

    // One arena per device covers the spine and the expert cache together.
    // Routed experts are allocated and freed on every miss, so the per-weight
    // cudaMalloc/cudaFree path costs measured driver time and fragments the
    // device until an upload fails outright with "out of memory". The arena is
    // a coalescing free list, so eviction still returns space.
    auto capacities = device_plan.value.weight_capacities;
    for (std::size_t slot = 0U; slot < config.devices.size(); ++slot) {
        result = impl_->cuda.reserve_weight_arena(config.devices[slot],
                                                  capacities[slot]);
        if (!result.ok()) return result;
        if (config.verbose || config.load_progress) {
            std::cerr << "[hardware] cuda=" << config.devices[slot]
                      << " weight_arena_bytes=" << capacities[slot] << '\n';
        }
    }

    // The resident spine is uploaded first; the expert cache then receives only
    // what each device has left, so a routed expert can never evict attention.
    result = impl_->warmup();
    if (!result.ok()) return result;
    for (std::size_t slot = 0U; slot < capacities.size(); ++slot) {
        if (impl_->pinned_bytes[slot] >= capacities[slot]) {
            result.errors.emplace_back(
                "the Laguna resident spine exhausts the admitted VRAM budget on device " +
                std::to_string(config.devices[slot]));
            return result;
        }
        capacities[slot] -= impl_->pinned_bytes[slot];
        if (config.verbose || config.load_progress) {
            std::cerr << "[hardware] cuda=" << config.devices[slot]
                      << " spine_bytes=" << impl_->pinned_bytes[slot]
                      << " expert_cache_bytes=" << capacities[slot] << '\n';
        }
    }
    impl_->experts = std::make_unique<ExpertCache>(
        *impl_->checkpoint, impl_->cuda, impl_->devices, std::move(capacities));

    try {
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            const auto rows = laguna_global_attention_layer(layer)
                ? config.maximum_context_tokens
                : std::min(config.maximum_context_tokens, kSlidingWindow);
            impl_->kv[layer].keys.reserve(
                static_cast<std::size_t>(rows) * kKvColumns);
            impl_->kv[layer].values.reserve(
                static_cast<std::size_t>(rows) * kKvColumns);
        }
    } catch (const std::bad_alloc&) {
        result.errors.emplace_back("cannot reserve the configured Laguna KV cache");
        return result;
    } catch (const std::length_error&) {
        result.errors.emplace_back("configured Laguna KV cache is too large");
        return result;
    }
    impl_->initialized = true;
    return result;
}

LagunaGenerationResult LagunaRuntime::generate_stream(
    std::string_view prompt, std::uint32_t maximum_new_tokens,
    const TokenStreamCallback& on_token) {
    const std::array messages{ChatMessage{ChatRole::User, std::string(prompt)}};
    SamplingOptions sampling;
    sampling.temperature = impl_->config.sampling_temperature;
    sampling.seed = impl_->config.sampling_seed;
    return generate_chat_stream(messages, maximum_new_tokens, sampling, {},
                                on_token);
}

LagunaGenerationResult LagunaRuntime::generate_chat_stream(
    std::span<const ChatMessage> messages, std::uint32_t maximum_new_tokens,
    const SamplingOptions& sampling, std::span<const std::string> stop,
    const TokenStreamCallback& on_token) {
    LagunaGenerationResult result;
    if (!impl_->initialized) {
        result.errors.emplace_back("Laguna runtime is not initialized");
        return result;
    }
    if (maximum_new_tokens == 0U) {
        result.errors.emplace_back("maximum_new_tokens must be positive");
        return result;
    }
    std::string sampling_error;
    if (!validate_sampling_options(sampling, sampling_error)) {
        result.errors.emplace_back("invalid sampling option: " + sampling_error);
        return result;
    }
    std::string validation_error;
    if (!validate_chat_messages(messages, validation_error)) {
        result.errors.push_back(std::move(validation_error));
        return result;
    }
    impl_->active_sampling = sampling;
    impl_->sampled_token_counts.assign(kVocabulary, 0U);
    impl_->sampled_token_ids.clear();
    impl_->sampler.seed(sampling.seed);

    std::vector<ChatMessage> active_messages(messages.begin(), messages.end());
    ParseResult<std::vector<std::uint32_t>> encoded;
    for (;;) {
        encoded = impl_->tokenizer.encode(render_laguna_chat_prompt(
            active_messages, impl_->config.enable_thinking));
        if (!encoded.ok()) {
            result.errors = std::move(encoded.errors);
            return result;
        }
        if (encoded.value.size() + maximum_new_tokens <=
            impl_->config.maximum_context_tokens) break;
        if (!trim_oldest_chat_turn(active_messages)) {
            result.errors.emplace_back(
                "prompt and requested generation exceed the context ceiling");
            return result;
        }
    }
    result.prompt_token_ids = std::move(encoded.value);
    impl_->active_request_id =
        impl_->config.request_id + impl_->generated_requests++;

    // Sliding layers keep only the last window, so a reused prefix is only
    // sound when the whole prompt prefix is still represented. Requiring the
    // reuse point to be inside the window keeps the sliding cache exact.
    std::size_t prefill_offset =
        impl_->config.enable_incremental_kv_continuation && impl_->reusable_sequence
            ? incremental_kv_prefix_tokens(impl_->cached_token_ids,
                                           result.prompt_token_ids)
            : 0U;
    if (prefill_offset != 0U &&
        prefill_offset != impl_->cached_token_ids.size()) {
        // A divergent suffix would need the sliding caches rewound past rows
        // that were already evicted; recompute instead of approximating.
        prefill_offset = 0U;
    }
    impl_->reusable_sequence = false;
    if (prefill_offset == 0U) impl_->reset_sequence();

    const auto reads_before = impl_->checkpoint->stats();
    const auto cuda_before = impl_->cuda.stats();
    const auto cache_before = impl_->cache_stats();
    const auto graph_before = impl_->graph_stats;
    const auto prefill_start = std::chrono::steady_clock::now();
    auto prefill_tokens = std::span<const std::uint32_t>(
        result.prompt_token_ids).subspan(prefill_offset);
    auto next = impl_->forward(prefill_tokens,
                               static_cast<std::uint32_t>(prefill_offset), true);
    result.metrics.prefill_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - prefill_start).count();
    result.metrics.prompt_tokens = result.prompt_token_ids.size();
    result.metrics.prefill_tokens = prefill_tokens.size();
    result.metrics.reused_prompt_tokens = prefill_offset;
    result.metrics.incremental_kv_continuation = prefill_offset != 0U;
    if (!next.ok()) {
        result.errors = std::move(next.errors);
        return result;
    }
    impl_->cached_token_ids = result.prompt_token_ids;
    const auto reads_after_prefill = impl_->checkpoint->stats();
    const auto cuda_after_prefill = impl_->cuda.stats();
    const auto cache_after_prefill = impl_->cache_stats();
    const auto graph_after_prefill = impl_->graph_stats;

    const auto is_stop = [](std::uint32_t token) {
        return std::find(kStopTokenIds.begin(), kStopTokenIds.end(), token) !=
               kStopTokenIds.end();
    };
    auto position = static_cast<std::uint32_t>(result.prompt_token_ids.size());
    StopSequenceBuffer output(stop);
    if (!is_stop(next.value)) {
        result.generated_token_ids.push_back(next.value);
        result.logprobs.push_back(impl_->last_sample);
        const auto piece = impl_->tokenizer.decode_token(next.value);
        if (!piece.ok()) {
            result.errors = std::move(piece.errors);
            return result;
        }
        output.append(next.value, piece.value, on_token);
    }
    const auto decode_start = std::chrono::steady_clock::now();
    while (!is_stop(next.value) && !output.stopped() && !output.cancelled() &&
           result.generated_token_ids.size() < maximum_new_tokens) {
        const std::array<std::uint32_t, 1> input{next.value};
        next = impl_->forward(input, position++, false);
        if (!next.ok()) {
            result.errors = std::move(next.errors);
            return result;
        }
        impl_->cached_token_ids.push_back(input[0]);
        ++result.metrics.decode_tokens;
        if (!is_stop(next.value)) {
            result.generated_token_ids.push_back(next.value);
            result.logprobs.push_back(impl_->last_sample);
            const auto piece = impl_->tokenizer.decode_token(next.value);
            if (!piece.ok()) {
                result.errors = std::move(piece.errors);
                return result;
            }
            output.append(next.value, piece.value, on_token);
        }
    }
    output.finish(on_token);
    result.stopped = output.stopped();
    result.metrics.decode_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - decode_start).count();
    result.text = output.text();

    const auto reads_after = impl_->checkpoint->stats();
    const auto cuda_after = impl_->cuda.stats();
    const auto cache_after = impl_->cache_stats();
    const auto graph_after = impl_->graph_stats;
    result.metrics.prefill.checkpoint_reads =
        checkpoint_delta(reads_after_prefill, reads_before);
    result.metrics.prefill.cuda =
        detail::cuda_delta(cuda_after_prefill, cuda_before);
    result.metrics.prefill.cache = cache_delta(cache_after_prefill, cache_before);
    result.metrics.prefill.graph = graph_delta(graph_after_prefill, graph_before);
    result.metrics.decode.checkpoint_reads =
        checkpoint_delta(reads_after, reads_after_prefill);
    result.metrics.decode.cuda =
        detail::cuda_delta(cuda_after, cuda_after_prefill);
    result.metrics.decode.cache = cache_delta(cache_after, cache_after_prefill);
    result.metrics.decode.graph = graph_delta(graph_after, graph_after_prefill);
    result.metrics.checkpoint_reads = checkpoint_delta(reads_after, reads_before);
    result.metrics.cuda = detail::cuda_delta(cuda_after, cuda_before);
    result.metrics.cache = cache_delta(cache_after, cache_before);
    result.metrics.graph = graph_delta(graph_after, graph_before);
    result.metrics.rss_bytes = process_resident_set_bytes();
    result.metrics.device_vram_used_bytes = device_vram_used_bytes(impl_->devices);
    result.metrics.flash_attention_enabled = impl_->config.enable_flash_attention;
    if (impl_->route_trace.is_open()) {
        auto flushed = impl_->route_trace.flush();
        move_errors(result.errors, std::move(flushed));
    }
    impl_->reusable_sequence =
        impl_->config.enable_incremental_kv_continuation && result.ok();
    return result;
}

}  // namespace strata
