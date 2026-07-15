#include "strata/glm_runtime.hpp"

#include "strata/glm_ops.hpp"
#include "strata/glm_int4.hpp"
#include "strata/tokenizer.hpp"
#include "strata/worker_pool.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <future>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <iostream>
#include <mutex>
#include <numeric>
#include <optional>
#include <unordered_map>

namespace strata {

namespace {

constexpr std::uint32_t kHidden = 6144U;
constexpr std::uint32_t kLayers = 78U;
constexpr std::uint32_t kHeads = 64U;
constexpr std::uint32_t kQueryLora = 2048U;
constexpr std::uint32_t kKvLora = 512U;
constexpr std::uint32_t kNope = 192U;
constexpr std::uint32_t kRope = 64U;
constexpr std::uint32_t kQueryHead = kNope + kRope;
constexpr std::uint32_t kValueHead = 256U;
constexpr std::uint32_t kDenseIntermediate = 12288U;
constexpr std::uint32_t kExpertIntermediate = 2048U;
constexpr std::uint32_t kExperts = 256U;
constexpr std::uint32_t kTopK = 8U;
constexpr std::uint32_t kVocabulary = 154880U;
constexpr std::uint32_t kDsaThreshold = 2048U;
constexpr float kAttentionScale = 1.0F / 16.0F;

std::uint64_t state_hash(std::span<const float> values) noexcept {
    constexpr std::uint64_t offset = 1469598103934665603ULL;
    constexpr std::uint64_t prime = 1099511628211ULL;
    std::uint64_t hash = offset;
    for (const float value : values) {
        hash ^= std::bit_cast<std::uint32_t>(value);
        hash *= prime;
    }
    return hash;
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

template <typename T>
void move_parse_errors(std::vector<std::string>& destination, ParseResult<T>& source,
                       std::string_view context = {}) {
    for (auto& error : source.errors) {
        if (!context.empty()) error = std::string(context) + ": " + error;
        destination.push_back(std::move(error));
    }
}

struct LinearSizeResult {
    std::uint64_t bytes{};
    std::vector<std::string> errors;
};

LinearSizeResult linear_bytes(const GlmCheckpointReader& checkpoint,
                              std::string_view base) {
    LinearSizeResult result;
    const std::string plain = std::string(base) + ".weight";
    if (const auto* tensor = checkpoint.find(plain); tensor != nullptr) {
        result.bytes = tensor->source_bytes;
        return result;
    }
    const std::string packed = std::string(base) + ".weight_packed";
    const std::string scales = std::string(base) + ".weight_scale";
    const auto* packed_tensor = checkpoint.find(packed);
    const auto* scale_tensor = checkpoint.find(scales);
    if (packed_tensor == nullptr || scale_tensor == nullptr) {
        result.errors.emplace_back("linear is absent from the checkpoint: " + std::string(base));
        return result;
    }
    result.bytes = packed_tensor->source_bytes + scale_tensor->source_bytes;
    return result;
}

ValidationResult validate_linear_shape(const GlmCheckpointReader& checkpoint,
                                       std::string_view base,
                                       std::uint64_t rows,
                                       std::uint64_t columns) {
    ValidationResult result;
    const std::string plain = std::string(base) + ".weight";
    if (const auto* tensor = checkpoint.find(plain); tensor != nullptr) {
        if (tensor->source_shape != std::vector<std::uint64_t>{rows, columns}) {
            result.errors.emplace_back("runtime linear shape mismatch: " + plain);
        }
        return result;
    }
    const std::string packed_name = std::string(base) + ".weight_packed";
    const std::string scale_name = std::string(base) + ".weight_scale";
    const auto* packed = checkpoint.find(packed_name);
    const auto* scales = checkpoint.find(scale_name);
    if (packed == nullptr || scales == nullptr || packed->encoding != scales->encoding) {
        result.errors.emplace_back("runtime compressed linear is missing: " + std::string(base));
        return result;
    }
    const std::uint32_t bits = packed->encoding == GlmTensorEncoding::Int4Group128 ? 4U : 8U;
    const std::uint32_t group = packed->encoding == GlmTensorEncoding::Int8Channel ? 0U : 128U;
    const auto packed_columns = (columns + (32U / bits) - 1U) / (32U / bits);
    const auto scale_columns = group == 0U ? 1U : (columns + group - 1U) / group;
    if (packed->source_shape != std::vector<std::uint64_t>{rows, packed_columns} ||
        scales->source_shape != std::vector<std::uint64_t>{rows, scale_columns}) {
        result.errors.emplace_back("runtime compressed linear shape mismatch: " +
                                   std::string(base));
    }
    return result;
}

ValidationResult validate_vector_shape(const GlmCheckpointReader& checkpoint,
                                       std::string_view name,
                                       std::uint64_t elements) {
    ValidationResult result;
    const auto* tensor = checkpoint.find(name);
    if (tensor == nullptr || tensor->source_shape != std::vector<std::uint64_t>{elements}) {
        result.errors.emplace_back("runtime vector shape mismatch: " + std::string(name));
    }
    return result;
}

ValidationResult validate_runtime_graph_contract(const GlmCheckpointReader& checkpoint) {
    ValidationResult result;
    const auto append = [&result](ValidationResult status) {
        for (auto& error : status.errors) {
            if (result.errors.size() < 64U) result.errors.push_back(std::move(error));
        }
    };
    const auto* embedding = checkpoint.find("model.embed_tokens.weight");
    if (embedding == nullptr ||
        embedding->source_shape != std::vector<std::uint64_t>{kVocabulary, kHidden}) {
        result.errors.emplace_back("runtime embedding shape mismatch");
    }
    append(validate_linear_shape(checkpoint, "lm_head", kVocabulary, kHidden));
    append(validate_vector_shape(checkpoint, "model.norm.weight", kHidden));
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        const auto prefix = layer_prefix(layer);
        const auto attention = prefix + "self_attn.";
        append(validate_vector_shape(checkpoint, prefix + "input_layernorm.weight", kHidden));
        append(validate_vector_shape(checkpoint, prefix + "post_attention_layernorm.weight",
                                     kHidden));
        append(validate_vector_shape(checkpoint, attention + "q_a_layernorm.weight", kQueryLora));
        append(validate_vector_shape(checkpoint, attention + "kv_a_layernorm.weight", kKvLora));
        append(validate_linear_shape(checkpoint, attention + "q_a_proj", kQueryLora, kHidden));
        append(validate_linear_shape(checkpoint, attention + "q_b_proj",
                                     kHeads * kQueryHead, kQueryLora));
        append(validate_linear_shape(checkpoint, attention + "kv_a_proj_with_mqa",
                                     kKvLora + kRope, kHidden));
        append(validate_linear_shape(checkpoint, attention + "kv_b_proj",
                                     kHeads * (kNope + kValueHead), kKvLora));
        append(validate_linear_shape(checkpoint, attention + "o_proj", kHidden,
                                     kHeads * kValueHead));
        const auto mlp = prefix + "mlp.";
        if (layer < 3U) {
            append(validate_linear_shape(checkpoint, mlp + "gate_proj",
                                         kDenseIntermediate, kHidden));
            append(validate_linear_shape(checkpoint, mlp + "up_proj",
                                         kDenseIntermediate, kHidden));
            append(validate_linear_shape(checkpoint, mlp + "down_proj",
                                         kHidden, kDenseIntermediate));
            continue;
        }
        append(validate_linear_shape(checkpoint, mlp + "gate", kExperts, kHidden));
        append(validate_vector_shape(checkpoint, mlp + "gate.e_score_correction_bias", kExperts));
        append(validate_linear_shape(checkpoint, mlp + "shared_experts.gate_proj",
                                     kExpertIntermediate, kHidden));
        append(validate_linear_shape(checkpoint, mlp + "shared_experts.up_proj",
                                     kExpertIntermediate, kHidden));
        append(validate_linear_shape(checkpoint, mlp + "shared_experts.down_proj",
                                     kHidden, kExpertIntermediate));
        for (std::uint32_t expert = 0; expert < kExperts; ++expert) {
            const auto expert_prefix = mlp + "experts." + std::to_string(expert) + ".";
            append(validate_linear_shape(checkpoint, expert_prefix + "gate_proj",
                                         kExpertIntermediate, kHidden));
            append(validate_linear_shape(checkpoint, expert_prefix + "up_proj",
                                         kExpertIntermediate, kHidden));
            append(validate_linear_shape(checkpoint, expert_prefix + "down_proj",
                                         kHidden, kExpertIntermediate));
        }
    }
    return result;
}

class WeightCache {
    struct Entry {
        CudaWeight weight;
        std::uint64_t last_use{};
        bool pinned{};
    };
    struct State {
        mutable std::mutex mutex;
        std::unordered_map<std::string, Entry> entries;
        std::uint64_t capacity{};
        std::uint64_t used{};
        std::uint64_t peak{};
        std::uint64_t pinned_used{};
        std::uint64_t clock{};
        std::uint64_t hits{};
        std::uint64_t misses{};
        std::uint64_t evictions{};
    };

public:
    WeightCache(GlmCheckpointReader& checkpoint, CudaBackend& backend,
                std::vector<int> devices, std::vector<std::uint64_t> capacities)
        : checkpoint_(checkpoint), backend_(backend), devices_(std::move(devices)) {
        states_.reserve(capacities.size());
        for (const auto capacity : capacities) {
            auto state = std::make_unique<State>();
            state->capacity = capacity;
            states_.push_back(std::move(state));
        }
    }

    ValidationResult matmul(std::size_t device_slot, std::string_view base,
                            std::uint64_t output_columns,
                            std::uint64_t input_columns,
                            std::span<const float> input, std::uint32_t rows,
                            std::span<float> output) {
        ValidationResult result;
        if (device_slot >= states_.size()) {
            result.errors.emplace_back("linear targets an invalid runtime device slot");
            return result;
        }
        auto& state = *states_[device_slot];
        std::scoped_lock lock(state.mutex);
        Entry* entry = nullptr;
        result = ensure_locked(state, device_slot, base, output_columns,
                               input_columns, false, entry);
        if (!result.ok()) return result;
        return backend_.matmul(entry->weight, input, rows, output);
    }

    ValidationResult preload(std::size_t device_slot, std::string_view base,
                             std::uint64_t output_columns,
                             std::uint64_t input_columns) {
        ValidationResult result;
        if (device_slot >= states_.size()) {
            result.errors.emplace_back("preload targets an invalid runtime device slot");
            return result;
        }
        auto& state = *states_[device_slot];
        std::scoped_lock lock(state.mutex);
        Entry* entry = nullptr;
        return ensure_locked(state, device_slot, base, output_columns,
                             input_columns, true, entry);
    }

    bool expert_resident(std::size_t device_slot, std::string_view prefix) const {
        if (device_slot >= states_.size()) return false;
        const auto& state = *states_[device_slot];
        std::scoped_lock lock(state.mutex);
        const std::string base(prefix);
        return state.entries.contains(base + "gate_proj") &&
               state.entries.contains(base + "up_proj") &&
               state.entries.contains(base + "down_proj");
    }

    [[nodiscard]] Glm52CacheStats stats() const {
        Glm52CacheStats result;
        result.hits = hits_.load(std::memory_order_relaxed);
        result.misses = misses_.load(std::memory_order_relaxed);
        result.evictions = evictions_.load(std::memory_order_relaxed);
        for (const auto& state_pointer : states_) {
            const auto& state = *state_pointer;
            std::scoped_lock lock(state.mutex);
            result.used_bytes.push_back(state.used);
            result.peak_bytes.push_back(state.peak);
            result.capacity_bytes.push_back(state.capacity);
            result.pinned_resident_bytes.push_back(state.pinned_used);
            result.evictable_expert_bytes.push_back(state.used - state.pinned_used);
            result.device_hits.push_back(state.hits);
            result.device_misses.push_back(state.misses);
            result.device_evictions.push_back(state.evictions);
        }
        return result;
    }

private:
    ValidationResult ensure_locked(State& state, std::size_t device_slot,
                                   std::string_view base,
                                   std::uint64_t output_columns,
                                   std::uint64_t input_columns,
                                   bool pin, Entry*& output) {
        ValidationResult result;
        ++state.clock;
        const std::string key(base);
        auto found = state.entries.find(key);
        if (found != state.entries.end()) {
            found->second.last_use = state.clock;
            if (pin && !found->second.pinned) {
                found->second.pinned = true;
                state.pinned_used += found->second.weight.device_bytes();
            }
            ++state.hits;
            hits_.fetch_add(1U, std::memory_order_relaxed);
            output = &found->second;
            return result;
        }
        const auto needed = linear_bytes(checkpoint_, base);
        if (!needed.errors.empty()) {
            result.errors = needed.errors;
            return result;
        }
        if (needed.bytes > state.capacity) {
            result.errors.emplace_back("linear exceeds the configured VRAM cache on device " +
                                       std::to_string(devices_[device_slot]));
            return result;
        }
        while (state.used + needed.bytes > state.capacity) {
            auto victim = state.entries.end();
            for (auto candidate = state.entries.begin(); candidate != state.entries.end();
                 ++candidate) {
                if (candidate->second.pinned) continue;
                if (victim == state.entries.end() ||
                    candidate->second.last_use < victim->second.last_use) {
                    victim = candidate;
                }
            }
            if (victim == state.entries.end()) {
                result.errors.emplace_back("pinned resident spine exceeds VRAM cache capacity on device " +
                                           std::to_string(devices_[device_slot]));
                return result;
            }
            state.used -= victim->second.weight.device_bytes();
            state.entries.erase(victim);
            ++state.evictions;
            evictions_.fetch_add(1U, std::memory_order_relaxed);
        }
        Entry entry;
        auto loaded = load_glm_cuda_linear(checkpoint_, base, output_columns,
                                           input_columns, devices_[device_slot],
                                           backend_, entry.weight);
        if (!loaded.ok()) return loaded;
        entry.last_use = state.clock;
        entry.pinned = pin;
        state.used += entry.weight.device_bytes();
        if (pin) state.pinned_used += entry.weight.device_bytes();
        state.peak = std::max(state.peak, state.used);
        found = state.entries.emplace(key, std::move(entry)).first;
        misses_.fetch_add(1U, std::memory_order_relaxed);
        ++state.misses;
        output = &found->second;
        return result;
    }

    GlmCheckpointReader& checkpoint_;
    CudaBackend& backend_;
    std::vector<int> devices_;
    std::vector<std::unique_ptr<State>> states_;
    std::atomic<std::uint64_t> hits_{};
    std::atomic<std::uint64_t> misses_{};
    std::atomic<std::uint64_t> evictions_{};
};

CheckpointReadStats checkpoint_delta(const CheckpointReadStats& after,
                                     const CheckpointReadStats& before) {
    return {after.calls - before.calls, after.bytes - before.bytes,
            after.nanoseconds - before.nanoseconds,
            after.wall_nanoseconds - before.wall_nanoseconds};
}

CudaBackendStats::Device cuda_device_delta(const CudaBackendStats::Device& after,
                                           const CudaBackendStats::Device& before) {
    CudaBackendStats::Device result;
    result.device = after.device;
#define STRATA_CUDA_DEVICE_DELTA(field) result.field = after.field - before.field
    STRATA_CUDA_DEVICE_DELTA(weight_upload_bytes);
    STRATA_CUDA_DEVICE_DELTA(activation_h2d_bytes);
    STRATA_CUDA_DEVICE_DELTA(activation_d2h_bytes);
    STRATA_CUDA_DEVICE_DELTA(matmul_calls);
    STRATA_CUDA_DEVICE_DELTA(weight_allocation_calls);
    STRATA_CUDA_DEVICE_DELTA(weight_allocation_bytes);
    STRATA_CUDA_DEVICE_DELTA(workspace_allocation_calls);
    STRATA_CUDA_DEVICE_DELTA(workspace_allocation_bytes);
    STRATA_CUDA_DEVICE_DELTA(synchronization_calls);
    STRATA_CUDA_DEVICE_DELTA(synchronization_nanoseconds);
    STRATA_CUDA_DEVICE_DELTA(upload_wait_nanoseconds);
    STRATA_CUDA_DEVICE_DELTA(activation_h2d_nanoseconds);
    STRATA_CUDA_DEVICE_DELTA(kernel_nanoseconds);
    STRATA_CUDA_DEVICE_DELTA(activation_d2h_nanoseconds);
#undef STRATA_CUDA_DEVICE_DELTA
    return result;
}

CudaBackendStats cuda_delta(const CudaBackendStats& after,
                            const CudaBackendStats& before) {
    CudaBackendStats result;
#define STRATA_CUDA_DELTA(field) result.field = after.field - before.field
    STRATA_CUDA_DELTA(weight_upload_bytes);
    STRATA_CUDA_DELTA(activation_h2d_bytes);
    STRATA_CUDA_DELTA(activation_d2h_bytes);
    STRATA_CUDA_DELTA(matmul_calls);
    STRATA_CUDA_DELTA(weight_allocation_calls);
    STRATA_CUDA_DELTA(weight_allocation_bytes);
    STRATA_CUDA_DELTA(workspace_allocation_calls);
    STRATA_CUDA_DELTA(workspace_allocation_bytes);
    STRATA_CUDA_DELTA(synchronization_calls);
    STRATA_CUDA_DELTA(synchronization_nanoseconds);
    STRATA_CUDA_DELTA(upload_wait_nanoseconds);
    STRATA_CUDA_DELTA(activation_h2d_nanoseconds);
    STRATA_CUDA_DELTA(kernel_nanoseconds);
    STRATA_CUDA_DELTA(activation_d2h_nanoseconds);
#undef STRATA_CUDA_DELTA
    result.synchronization_nanoseconds = 0U;
    result.upload_wait_nanoseconds = 0U;
    result.activation_h2d_nanoseconds = 0U;
    result.kernel_nanoseconds = 0U;
    result.activation_d2h_nanoseconds = 0U;
    for (const auto& device_after : after.devices) {
        const auto found = std::find_if(
            before.devices.begin(), before.devices.end(),
            [&device_after](const auto& value) { return value.device == device_after.device; });
        if (found == before.devices.end()) {
            result.devices.push_back(device_after);
        } else {
            result.devices.push_back(cuda_device_delta(device_after, *found));
        }
        const auto& delta = result.devices.back();
        result.synchronization_nanoseconds = std::max(
            result.synchronization_nanoseconds, delta.synchronization_nanoseconds);
        result.upload_wait_nanoseconds = std::max(
            result.upload_wait_nanoseconds, delta.upload_wait_nanoseconds);
        result.activation_h2d_nanoseconds = std::max(
            result.activation_h2d_nanoseconds, delta.activation_h2d_nanoseconds);
        result.kernel_nanoseconds = std::max(
            result.kernel_nanoseconds, delta.kernel_nanoseconds);
        result.activation_d2h_nanoseconds = std::max(
            result.activation_d2h_nanoseconds, delta.activation_d2h_nanoseconds);
    }
    return result;
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

Glm52CacheStats cache_delta(const Glm52CacheStats& after,
                            const Glm52CacheStats& before) {
    Glm52CacheStats result = after;
    result.hits -= before.hits;
    result.misses -= before.misses;
    result.evictions -= before.evictions;
    result.device_hits = counter_delta(after.device_hits, before.device_hits);
    result.device_misses = counter_delta(after.device_misses, before.device_misses);
    result.device_evictions = counter_delta(after.device_evictions, before.device_evictions);
    return result;
}

Glm52HostExpertStats host_expert_delta(const Glm52HostExpertStats& after,
                                       const Glm52HostExpertStats& before) {
    return {after.experts - before.experts,
            after.matvec_calls - before.matvec_calls,
            after.weight_bytes - before.weight_bytes,
            after.service_nanoseconds - before.service_nanoseconds,
            after.mapping_sweeps - before.mapping_sweeps,
            after.mapping_sweep_nanoseconds - before.mapping_sweep_nanoseconds};
}

struct LayerKv {
    std::vector<float> keys;
    std::vector<float> values;
    std::vector<float> rope;
};

struct ExpertJob {
    std::uint32_t expert{};
    std::size_t device_slot{};
    std::vector<std::uint32_t> rows;
    std::vector<float> weights;
    std::vector<float> output;
    std::vector<std::string> errors;
};

}  // namespace

struct Glm52Runtime::Impl {
    Glm52RuntimeConfig config;
    std::unique_ptr<GlmCheckpointReader> checkpoint;
    GlmTokenizer tokenizer;
    CudaBackend cuda;
    std::unique_ptr<WeightCache> weights;
    std::unique_ptr<HostWorkerPool> host_workers;
    std::vector<int> devices;
    std::vector<std::uint64_t> capacities;
    std::vector<std::size_t> device_schedule;
    std::unordered_map<std::string, std::vector<float>> host_tensors;
    std::unordered_map<std::uint32_t, std::vector<float>> embedding_rows;
    std::vector<LayerKv> kv{kLayers};
    std::ofstream route_trace;
    bool initialized{};
    std::uint32_t last_second_token{};
    float last_logit_margin{};
    std::uint64_t host_aggregation_nanoseconds{};
    std::uint64_t active_request_id{};
    std::uint64_t generated_requests{};
    std::atomic<std::uint64_t> host_expert_count{};
    std::atomic<std::uint64_t> host_expert_matvec_calls{};
    std::atomic<std::uint64_t> host_expert_weight_bytes{};
    std::atomic<std::uint64_t> host_expert_service_nanoseconds{};
    std::atomic<std::uint64_t> host_mapping_sweeps{};
    std::atomic<std::uint64_t> host_mapping_sweep_nanoseconds{};

    Glm52HostExpertStats host_stats() const noexcept {
        return {host_expert_count.load(std::memory_order_relaxed),
                host_expert_matvec_calls.load(std::memory_order_relaxed),
                host_expert_weight_bytes.load(std::memory_order_relaxed),
                host_expert_service_nanoseconds.load(std::memory_order_relaxed),
                host_mapping_sweeps.load(std::memory_order_relaxed),
                host_mapping_sweep_nanoseconds.load(std::memory_order_relaxed)};
    }

    std::size_t layer_device(std::uint32_t layer) const {
        return device_schedule[layer % device_schedule.size()];
    }

    std::size_t expert_device(std::uint32_t expert) const {
        return device_schedule[expert % device_schedule.size()];
    }

    bool write_route(std::uint32_t position, std::uint32_t layer,
                     const GlmRoute& route, bool prefill) {
        if (!route_trace.is_open()) return true;
        route_trace << "{\"request\":" << active_request_id
                    << ",\"phase\":\"" << (prefill ? "prefill" : "decode")
                    << "\",\"token_position\":" << position
                    << ",\"layer\":" << layer << ",\"experts\":[";
        for (std::size_t rank = 0; rank < route.experts.size(); ++rank) {
            if (rank != 0U) route_trace << ',';
            route_trace << route.experts[rank];
        }
        route_trace << "],\"coefficients\":[" << std::setprecision(
            std::numeric_limits<float>::max_digits10);
        for (std::size_t rank = 0; rank < route.weights.size(); ++rank) {
            if (rank != 0U) route_trace << ',';
            route_trace << route.weights[rank];
        }
        route_trace << "]}\n";
        return route_trace.good();
    }

    ParseResult<const std::vector<float>*> host_tensor(std::string name,
                                                       std::uint64_t ceiling) {
        ParseResult<const std::vector<float>*> result;
        const auto found = host_tensors.find(name);
        if (found != host_tensors.end()) {
            result.value = &found->second;
            return result;
        }
        auto loaded = checkpoint->read_f32(name, ceiling);
        if (!loaded.ok()) {
            result.errors = std::move(loaded.errors);
            return result;
        }
        auto inserted = host_tensors.emplace(std::move(name), std::move(loaded.value));
        result.value = &inserted.first->second;
        return result;
    }

    ValidationResult linear(std::size_t device, std::string_view base,
                            std::uint64_t output_columns,
                            std::uint64_t input_columns,
                            std::span<const float> input, std::uint32_t rows,
                            std::span<float> output) {
        return weights->matmul(device, base, output_columns, input_columns,
                               input, rows, output);
    }

    ValidationResult warmup_resident_spine() {
        ValidationResult result;
        const auto preload = [this, &result](std::size_t device, std::string base,
                                             std::uint64_t rows,
                                             std::uint64_t columns) {
            if (!result.ok()) return;
            result = weights->preload(device, base, rows, columns);
        };
        for (std::uint32_t layer = 0; layer < kLayers && result.ok(); ++layer) {
            const auto device = layer_device(layer);
            const auto prefix = layer_prefix(layer);
            const auto attention = prefix + "self_attn.";
            preload(device, attention + "q_a_proj", kQueryLora, kHidden);
            preload(device, attention + "q_b_proj", kHeads * kQueryHead, kQueryLora);
            preload(device, attention + "kv_a_proj_with_mqa", kKvLora + kRope, kHidden);
            preload(device, attention + "kv_b_proj", kHeads * (kNope + kValueHead), kKvLora);
            preload(device, attention + "o_proj", kHidden, kHeads * kValueHead);
            const auto mlp = prefix + "mlp.";
            if (layer < 3U) {
                preload(device, mlp + "gate_proj", kDenseIntermediate, kHidden);
                preload(device, mlp + "up_proj", kDenseIntermediate, kHidden);
                preload(device, mlp + "down_proj", kHidden, kDenseIntermediate);
            } else {
                preload(device, mlp + "gate", kExperts, kHidden);
                preload(device, mlp + "shared_experts.gate_proj",
                        kExpertIntermediate, kHidden);
                preload(device, mlp + "shared_experts.up_proj",
                        kExpertIntermediate, kHidden);
                preload(device, mlp + "shared_experts.down_proj",
                        kHidden, kExpertIntermediate);
            }
            auto input_norm = host_tensor(prefix + "input_layernorm.weight", kHidden);
            auto post_norm = host_tensor(prefix + "post_attention_layernorm.weight", kHidden);
            auto query_norm = host_tensor(attention + "q_a_layernorm.weight", kQueryLora);
            auto kv_norm = host_tensor(attention + "kv_a_layernorm.weight", kKvLora);
            for (auto* loaded : {&input_norm, &post_norm, &query_norm, &kv_norm}) {
                if (!loaded->ok() && result.ok()) result.errors = std::move(loaded->errors);
            }
            if (layer >= 3U) {
                auto bias = host_tensor(mlp + "gate.e_score_correction_bias", kExperts);
                if (!bias.ok() && result.ok()) result.errors = std::move(bias.errors);
            }
            if (config.verbose && result.ok()) {
                std::cerr << "[load] resident spine layer " << (layer + 1U) << '/'
                          << kLayers << '\n';
            }
        }
        if (result.ok()) {
            preload(layer_device(kLayers - 1U), "lm_head", kVocabulary, kHidden);
            auto final_norm = host_tensor("model.norm.weight", kHidden);
            if (!final_norm.ok()) result.errors = std::move(final_norm.errors);
        }
        return result;
    }

    ValidationResult embed(std::span<const std::uint32_t> token_ids,
                           std::span<float> output) {
        ValidationResult result;
        if (output.size() != token_ids.size() * kHidden) {
            result.errors.emplace_back("embedding output shape mismatch");
            return result;
        }
        for (std::size_t row = 0; row < token_ids.size(); ++row) {
            if (token_ids[row] >= kVocabulary) {
                result.errors.emplace_back("token id exceeds the model vocabulary");
                return result;
            }
            auto found = embedding_rows.find(token_ids[row]);
            if (found == embedding_rows.end()) {
                auto loaded = checkpoint->read_f32_row("model.embed_tokens.weight", token_ids[row]);
                if (!loaded.ok()) {
                    result.errors = std::move(loaded.errors);
                    return result;
                }
                found = embedding_rows.emplace(token_ids[row], std::move(loaded.value)).first;
            }
            std::copy(found->second.begin(), found->second.end(),
                      output.begin() + static_cast<std::ptrdiff_t>(row * kHidden));
        }
        return result;
    }

    ValidationResult norm_rows(std::span<float> output, std::span<const float> input,
                               std::uint32_t rows, std::string name) {
        ValidationResult result;
        auto weight = host_tensor(std::move(name), kHidden);
        if (!weight.ok()) {
            result.errors = std::move(weight.errors);
            return result;
        }
        if (input.size() != static_cast<std::size_t>(rows) * kHidden ||
            output.size() != input.size() || weight.value->size() != kHidden) {
            result.errors.emplace_back("RMSNorm runtime shape mismatch");
            return result;
        }
        for (std::uint32_t row = 0; row < rows; ++row) {
            auto status = glm_rms_norm_f32(
                output.subspan(static_cast<std::size_t>(row) * kHidden, kHidden),
                input.subspan(static_cast<std::size_t>(row) * kHidden, kHidden),
                *weight.value);
            if (!status.ok()) return status;
        }
        return result;
    }

    ValidationResult attention(std::uint32_t layer, std::span<const float> input,
                               std::uint32_t rows, std::uint32_t position_base,
                               std::span<float> output) {
        ValidationResult result;
        const auto prefix = layer_prefix(layer) + "self_attn.";
        const auto device = layer_device(layer);
        std::vector<float> query_residual(static_cast<std::size_t>(rows) * kQueryLora);
        result = linear(device, prefix + "q_a_proj", kQueryLora, kHidden,
                        input, rows, query_residual);
        if (!result.ok()) return result;
        auto query_norm = host_tensor(prefix + "q_a_layernorm.weight", kQueryLora);
        if (!query_norm.ok()) {
            result.errors = std::move(query_norm.errors);
            return result;
        }
        for (std::uint32_t row = 0; row < rows; ++row) {
            auto status = glm_rms_norm_f32(
                std::span<float>(query_residual).subspan(
                    static_cast<std::size_t>(row) * kQueryLora, kQueryLora),
                std::span<const float>(query_residual).subspan(
                    static_cast<std::size_t>(row) * kQueryLora, kQueryLora),
                *query_norm.value);
            if (!status.ok()) return status;
        }
        std::vector<float> queries(static_cast<std::size_t>(rows) * kHeads * kQueryHead);
        result = linear(device, prefix + "q_b_proj", kHeads * kQueryHead,
                        kQueryLora, query_residual, rows, queries);
        if (!result.ok()) return result;
        for (std::uint32_t row = 0; row < rows; ++row) {
            const auto position = position_base + row;
            for (std::uint32_t head = 0; head < kHeads; ++head) {
                auto rope_values = std::span<float>(queries).subspan(
                    (static_cast<std::size_t>(row) * kHeads + head) * kQueryHead + kNope,
                    kRope);
                auto status = glm_rope_interleaved_f32(rope_values, position);
                if (!status.ok()) return status;
            }
        }

        std::vector<float> compressed(static_cast<std::size_t>(rows) * (kKvLora + kRope));
        result = linear(device, prefix + "kv_a_proj_with_mqa", kKvLora + kRope,
                        kHidden, input, rows, compressed);
        if (!result.ok()) return result;
        auto kv_norm = host_tensor(prefix + "kv_a_layernorm.weight", kKvLora);
        if (!kv_norm.ok()) {
            result.errors = std::move(kv_norm.errors);
            return result;
        }
        std::vector<float> latent(static_cast<std::size_t>(rows) * kKvLora);
        std::vector<float> new_rope(static_cast<std::size_t>(rows) * kRope);
        for (std::uint32_t row = 0; row < rows; ++row) {
            const auto source = std::span<const float>(compressed).subspan(
                static_cast<std::size_t>(row) * (kKvLora + kRope), kKvLora + kRope);
            auto destination = std::span<float>(latent).subspan(
                static_cast<std::size_t>(row) * kKvLora, kKvLora);
            auto status = glm_rms_norm_f32(destination, source.first(kKvLora),
                                            *kv_norm.value);
            if (!status.ok()) return status;
            auto rope_destination = std::span<float>(new_rope).subspan(
                static_cast<std::size_t>(row) * kRope, kRope);
            std::copy(source.begin() + kKvLora, source.end(), rope_destination.begin());
            status = glm_rope_interleaved_f32(rope_destination, position_base + row);
            if (!status.ok()) return status;
        }

        constexpr std::uint32_t kv_output = kHeads * (kNope + kValueHead);
        std::vector<float> reconstructed(static_cast<std::size_t>(rows) * kv_output);
        result = linear(device, prefix + "kv_b_proj", kv_output, kKvLora,
                        latent, rows, reconstructed);
        if (!result.ok()) return result;

        auto& cache = kv[layer];
        const auto existing = cache.rope.size() / kRope;
        if (existing != position_base) {
            result.errors.emplace_back("KV cache position is not contiguous at layer " +
                                       std::to_string(layer));
            return result;
        }
        cache.rope.insert(cache.rope.end(), new_rope.begin(), new_rope.end());
        cache.keys.reserve(cache.keys.size() + static_cast<std::size_t>(rows) * kHeads * kNope);
        cache.values.reserve(cache.values.size() +
                             static_cast<std::size_t>(rows) * kHeads * kValueHead);
        for (std::uint32_t row = 0; row < rows; ++row) {
            const auto* source = reconstructed.data() + static_cast<std::size_t>(row) * kv_output;
            for (std::uint32_t head = 0; head < kHeads; ++head) {
                const auto* block = source + static_cast<std::size_t>(head) * (kNope + kValueHead);
                cache.keys.insert(cache.keys.end(), block, block + kNope);
                cache.values.insert(cache.values.end(), block + kNope,
                                    block + kNope + kValueHead);
            }
        }

        std::vector<float> context(static_cast<std::size_t>(rows) * kHeads * kValueHead);
        for (std::uint32_t row = 0; row < rows; ++row) {
            const std::uint32_t position = position_base + row;
            const std::size_t token_count = static_cast<std::size_t>(position) + 1U;
            std::vector<float> scores(token_count);
            for (std::uint32_t head = 0; head < kHeads; ++head) {
                const auto* query = queries.data() +
                    (static_cast<std::size_t>(row) * kHeads + head) * kQueryHead;
                for (std::size_t token = 0; token < token_count; ++token) {
                    const auto* key = cache.keys.data() +
                        (token * kHeads + head) * kNope;
                    const auto* rope_key = cache.rope.data() + token * kRope;
                    float score = 0.0F;
                    for (std::uint32_t index = 0; index < kNope; ++index) {
                        score += query[index] * key[index];
                    }
                    for (std::uint32_t index = 0; index < kRope; ++index) {
                        score += query[kNope + index] * rope_key[index];
                    }
                    scores[token] = score * kAttentionScale;
                }
                auto status = glm_softmax_f32(scores);
                if (!status.ok()) return status;
                auto* destination = context.data() +
                    (static_cast<std::size_t>(row) * kHeads + head) * kValueHead;
                std::fill(destination, destination + kValueHead, 0.0F);
                for (std::size_t token = 0; token < token_count; ++token) {
                    const auto* value = cache.values.data() +
                        (token * kHeads + head) * kValueHead;
                    for (std::uint32_t index = 0; index < kValueHead; ++index) {
                        destination[index] += scores[token] * value[index];
                    }
                }
            }
        }
        return linear(device, prefix + "o_proj", kHidden, kHeads * kValueHead,
                      context, rows, output);
    }

    ValidationResult mlp_triplet(std::size_t device, std::string_view prefix,
                                 std::uint32_t intermediate,
                                 std::span<const float> input,
                                 std::uint32_t rows,
                                 std::span<float> output) {
        ValidationResult result;
        std::vector<float> gate(static_cast<std::size_t>(rows) * intermediate);
        std::vector<float> up(gate.size());
        result = linear(device, std::string(prefix) + "gate_proj", intermediate,
                        kHidden, input, rows, gate);
        if (!result.ok()) return result;
        result = linear(device, std::string(prefix) + "up_proj", intermediate,
                        kHidden, input, rows, up);
        if (!result.ok()) return result;
        for (std::size_t index = 0; index < gate.size(); ++index) {
            gate[index] = glm_silu_f32(gate[index]) * up[index];
        }
        return linear(device, std::string(prefix) + "down_proj", kHidden,
                      intermediate, gate, rows, output);
    }

    struct HostMatrix {
        GlmInt4MatrixView view;
    };

    ParseResult<HostMatrix> host_matrix(std::string_view base,
                                        std::uint64_t rows,
                                        std::uint64_t columns) {
        ParseResult<HostMatrix> result;
        const std::string packed_name = std::string(base) + ".weight_packed";
        const std::string scale_name = std::string(base) + ".weight_scale";
        const auto* packed = checkpoint->find(packed_name);
        const auto* scales = checkpoint->find(scale_name);
        const auto packed_columns = (columns + 7U) / 8U;
        const auto scale_columns = (columns + 127U) / 128U;
        if (packed == nullptr || scales == nullptr ||
            packed->encoding != GlmTensorEncoding::Int4Group128 ||
            scales->encoding != GlmTensorEncoding::Int4Group128 ||
            packed->source_shape !=
                std::vector<std::uint64_t>{rows, packed_columns} ||
            scales->source_shape !=
                std::vector<std::uint64_t>{rows, scale_columns}) {
            result.errors.emplace_back(
                "host expert matrix is not target INT4 group-128: " +
                std::string(base));
            return result;
        }
        auto packed_view = checkpoint->view(packed_name);
        auto scale_view = checkpoint->view(scale_name);
        if (!packed_view.ok()) {
            result.errors = std::move(packed_view.errors);
            return result;
        }
        if (!scale_view.ok()) {
            checkpoint->release_view(packed_view.value);
            result.errors = std::move(scale_view.errors);
            return result;
        }
        result.value.view = {packed_view.value, scale_view.value, rows, columns,
                             packed_columns, scale_columns, 128U};
        return result;
    }

    void release_host_matrix(const HostMatrix& matrix) const noexcept {
        // The mapped host tier retains its file-backed working set across
        // layers and tokens. Immediate MADV_DONTNEED here forces thousands of
        // avoidable minor faults and discards route locality. The mapping is
        // still file-backed and remains reclaimable by the kernel.
        static_cast<void>(matrix);
    }

    ValidationResult host_matvec(const HostMatrix& matrix,
                                 std::span<const float> input,
                                 std::span<float> output) {
        ValidationResult result;
        if (host_workers == nullptr) {
            result.errors.emplace_back("host expert worker pool is unavailable");
            return result;
        }
        constexpr std::uint64_t rows_per_task = 128U;
        const auto tasks = static_cast<std::size_t>(
            (matrix.view.rows + rows_per_task - 1U) / rows_per_task);
        std::mutex error_mutex;
        auto dispatched = host_workers->parallel_for(tasks, [&](std::size_t task) {
            const auto begin = static_cast<std::uint64_t>(task) * rows_per_task;
            const auto end = std::min(matrix.view.rows, begin + rows_per_task);
            auto status = glm_int4_group128_matvec_rows(
                matrix.view, input, output, begin, end);
            if (!status.ok()) {
                std::scoped_lock lock(error_mutex);
                if (result.errors.empty()) result.errors = std::move(status.errors);
            }
        });
        if (!dispatched.ok() && result.errors.empty()) {
            result.errors = std::move(dispatched.errors);
        }
        return result;
    }

    void run_host_expert_job(std::uint32_t layer, ExpertJob& job,
                             std::span<const float> input) {
        const auto started = std::chrono::steady_clock::now();
        const auto prefix = layer_prefix(layer) + "mlp.experts." +
                            std::to_string(job.expert) + ".";
        auto gate = host_matrix(prefix + "gate_proj", kExpertIntermediate, kHidden);
        auto up = host_matrix(prefix + "up_proj", kExpertIntermediate, kHidden);
        auto down = host_matrix(prefix + "down_proj", kHidden, kExpertIntermediate);
        const auto release = [&] {
            if (gate.ok()) release_host_matrix(gate.value);
            if (up.ok()) release_host_matrix(up.value);
            if (down.ok()) release_host_matrix(down.value);
        };
        for (auto* loaded : {&gate, &up, &down}) {
            if (!loaded->ok()) {
                move_parse_errors(job.errors, *loaded);
                release();
                return;
            }
        }
        std::vector<float> gate_output(kExpertIntermediate);
        std::vector<float> up_output(kExpertIntermediate);
        job.output.resize(kHidden);
        auto status = host_matvec(gate.value, input, gate_output);
        if (status.ok()) status = host_matvec(up.value, input, up_output);
        if (status.ok()) {
            for (std::size_t index = 0; index < gate_output.size(); ++index) {
                gate_output[index] =
                    glm_silu_f32(gate_output[index]) * up_output[index];
            }
            status = host_matvec(down.value, gate_output, job.output);
        }
        const auto weight_bytes =
            gate.value.view.packed.size() + gate.value.view.scales.size() +
            up.value.view.packed.size() + up.value.view.scales.size() +
            down.value.view.packed.size() + down.value.view.scales.size();
        release();
        if (!status.ok()) {
            move_errors(job.errors, std::move(status));
            return;
        }
        host_expert_count.fetch_add(1U, std::memory_order_relaxed);
        host_expert_matvec_calls.fetch_add(3U, std::memory_order_relaxed);
        host_expert_weight_bytes.fetch_add(weight_bytes, std::memory_order_relaxed);
        host_expert_service_nanoseconds.fetch_add(
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - started).count()),
            std::memory_order_relaxed);
    }

    struct HostMatvecRequest {
        const HostMatrix* matrix{};
        std::span<const float> input;
        std::span<float> output;
    };

    ValidationResult host_matvec_wavefront(
        std::span<const HostMatvecRequest> requests) {
        ValidationResult result;
        if (requests.empty()) return result;
        if (host_workers == nullptr) {
            result.errors.emplace_back("host expert worker pool is unavailable");
            return result;
        }
        struct Task {
            const HostMatvecRequest* request{};
            std::uint64_t begin{};
            std::uint64_t end{};
        };
        constexpr std::uint64_t rows_per_task = 64U;
        std::vector<Task> tasks;
        for (const auto& request : requests) {
            for (std::uint64_t begin = 0U;
                 begin < request.matrix->view.rows; begin += rows_per_task) {
                tasks.push_back(
                    {&request, begin,
                     std::min(request.matrix->view.rows, begin + rows_per_task)});
            }
        }
        std::mutex error_mutex;
        auto dispatched = host_workers->parallel_for(
            tasks.size(), [&](std::size_t index) {
                const auto& task = tasks[index];
                auto status = glm_int4_group128_matvec_rows(
                    task.request->matrix->view, task.request->input,
                    task.request->output, task.begin, task.end);
                if (!status.ok()) {
                    std::scoped_lock lock(error_mutex);
                    if (result.errors.empty()) {
                        result.errors = std::move(status.errors);
                    }
                }
            });
        if (!dispatched.ok() && result.errors.empty()) {
            result.errors = std::move(dispatched.errors);
        }
        return result;
    }

    void run_host_expert_wavefront(std::uint32_t layer,
                                   std::span<ExpertJob*> jobs,
                                   std::span<const float> input) {
        if (jobs.empty()) return;
        const auto started = std::chrono::steady_clock::now();
        struct Work {
            ExpertJob* job{};
            HostMatrix gate;
            HostMatrix up;
            HostMatrix down;
            std::vector<float> gate_output =
                std::vector<float>(kExpertIntermediate);
            std::vector<float> up_output =
                std::vector<float>(kExpertIntermediate);
        };
        std::vector<Work> work;
        work.reserve(jobs.size());
        const auto release_all = [&] {
            for (const auto& item : work) {
                if (!item.gate.view.packed.empty()) release_host_matrix(item.gate);
                if (!item.up.view.packed.empty()) release_host_matrix(item.up);
                if (!item.down.view.packed.empty()) release_host_matrix(item.down);
            }
        };
        std::uint64_t weight_bytes = 0U;
        for (auto* job : jobs) {
            work.emplace_back();
            auto& item = work.back();
            item.job = job;
            const auto prefix = layer_prefix(layer) + "mlp.experts." +
                                std::to_string(job->expert) + ".";
            auto gate = host_matrix(
                prefix + "gate_proj", kExpertIntermediate, kHidden);
            auto up = host_matrix(
                prefix + "up_proj", kExpertIntermediate, kHidden);
            auto down = host_matrix(
                prefix + "down_proj", kHidden, kExpertIntermediate);
            for (auto* loaded : {&gate, &up, &down}) {
                if (!loaded->ok()) {
                    move_parse_errors(job->errors, *loaded);
                    if (gate.ok()) release_host_matrix(gate.value);
                    if (up.ok()) release_host_matrix(up.value);
                    if (down.ok()) release_host_matrix(down.value);
                    release_all();
                    return;
                }
            }
            item.gate = gate.value;
            item.up = up.value;
            item.down = down.value;
            item.job->output.resize(kHidden);
            weight_bytes +=
                item.gate.view.packed.size() + item.gate.view.scales.size() +
                item.up.view.packed.size() + item.up.view.scales.size() +
                item.down.view.packed.size() + item.down.view.scales.size();
        }

        std::vector<HostMatvecRequest> gate_up;
        gate_up.reserve(work.size() * 2U);
        for (auto& item : work) {
            gate_up.push_back({&item.gate, input, item.gate_output});
            gate_up.push_back({&item.up, input, item.up_output});
        }
        auto status = host_matvec_wavefront(gate_up);
        if (status.ok()) {
            for (auto& item : work) {
                for (std::size_t index = 0; index < item.gate_output.size(); ++index) {
                    item.gate_output[index] =
                        glm_silu_f32(item.gate_output[index]) *
                        item.up_output[index];
                }
            }
            std::vector<HostMatvecRequest> down;
            down.reserve(work.size());
            for (auto& item : work) {
                down.push_back(
                    {&item.down, item.gate_output, item.job->output});
            }
            status = host_matvec_wavefront(down);
        }
        release_all();
        if (!status.ok()) {
            move_errors(jobs.front()->errors, std::move(status));
            return;
        }
        host_expert_count.fetch_add(jobs.size(), std::memory_order_relaxed);
        host_expert_matvec_calls.fetch_add(
            jobs.size() * 3U, std::memory_order_relaxed);
        host_expert_weight_bytes.fetch_add(weight_bytes, std::memory_order_relaxed);
        host_expert_service_nanoseconds.fetch_add(
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - started).count()),
            std::memory_order_relaxed);
    }

    void run_expert_job(std::uint32_t layer, ExpertJob& job,
                        std::span<const float> input) {
        std::vector<float> gathered(job.rows.size() * kHidden);
        for (std::size_t row = 0; row < job.rows.size(); ++row) {
            const auto source = input.subspan(static_cast<std::size_t>(job.rows[row]) * kHidden,
                                              kHidden);
            std::copy(source.begin(), source.end(),
                      gathered.begin() + static_cast<std::ptrdiff_t>(row * kHidden));
        }
        job.output.resize(gathered.size());
        const auto prefix = layer_prefix(layer) + "mlp.experts." +
                            std::to_string(job.expert) + ".";
        const auto status = mlp_triplet(job.device_slot, prefix, kExpertIntermediate,
                                        gathered, static_cast<std::uint32_t>(job.rows.size()),
                                        job.output);
        move_errors(job.errors, status);
    }

    ValidationResult sparse_mlp(std::uint32_t layer, std::span<const float> input,
                                std::uint32_t rows, std::uint32_t position_base,
                                bool prefill, std::span<float> output) {
        ValidationResult result;
        const auto prefix = layer_prefix(layer) + "mlp.";
        std::vector<float> logits(static_cast<std::size_t>(rows) * kExperts);
        result = linear(layer_device(layer), prefix + "gate", kExperts, kHidden,
                        input, rows, logits);
        if (!result.ok()) return result;
        auto bias = host_tensor(prefix + "gate.e_score_correction_bias", kExperts);
        if (!bias.ok()) {
            result.errors = std::move(bias.errors);
            return result;
        }
        const auto router = quanttrio_glm52_int4_int8_mix_spec().router;
        std::vector<GlmRoute> routes(rows);
        for (std::uint32_t row = 0; row < rows; ++row) {
            auto routed = glm_route_logits_noaux_tc(
                std::span<const float>(logits).subspan(
                    static_cast<std::size_t>(row) * kExperts, kExperts),
                *bias.value, router);
            if (!routed.ok()) {
                result.errors = std::move(routed.errors);
                return result;
            }
            routes[row] = std::move(routed.value);
            if (!write_route(position_base + row, layer, routes[row], prefill)) {
                result.errors.emplace_back("cannot write GLM route trace");
                return result;
            }
        }

        std::array<std::int32_t, kExperts> job_by_expert{};
        job_by_expert.fill(-1);
        std::vector<ExpertJob> jobs;
        jobs.reserve(std::min<std::size_t>(static_cast<std::size_t>(rows) * kTopK,
                                           kExperts));
        for (std::uint32_t row = 0; row < rows; ++row) {
            for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
                const auto expert = routes[row].experts[rank];
                if (job_by_expert[expert] < 0) {
                    job_by_expert[expert] = static_cast<std::int32_t>(jobs.size());
                    ExpertJob job;
                    job.expert = expert;
                    job.device_slot = expert_device(expert);
                    jobs.push_back(std::move(job));
                }
                auto& job = jobs[static_cast<std::size_t>(job_by_expert[expert])];
                job.rows.push_back(row);
                job.weights.push_back(routes[row].weights[rank]);
            }
        }

        std::vector<std::uint8_t> host_jobs(jobs.size(), 0U);
        if (config.host_cold_experts && rows == 1U) {
            for (std::size_t index = 0; index < jobs.size(); ++index) {
                const auto expert_prefix = layer_prefix(layer) + "mlp.experts." +
                                           std::to_string(jobs[index].expert) + ".";
                if (!weights->expert_resident(jobs[index].device_slot,
                                              expert_prefix)) {
                    host_jobs[index] = 1U;
                }
            }
        }

        std::vector<std::future<void>> workers;
        if (std::find(host_jobs.begin(), host_jobs.end(), 1U) != host_jobs.end()) {
            workers.push_back(std::async(std::launch::async, [&] {
                std::vector<ExpertJob*> wavefront;
                for (std::size_t index = 0; index < jobs.size(); ++index) {
                    if (host_jobs[index] != 0U) {
                        wavefront.push_back(&jobs[index]);
                    }
                }
                run_host_expert_wavefront(layer, wavefront, input);
            }));
        }
        for (std::size_t device = 0; device < devices.size(); ++device) {
            workers.push_back(std::async(std::launch::async, [&, device] {
                for (std::size_t index = 0; index < jobs.size(); ++index) {
                    if (host_jobs[index] == 0U &&
                        jobs[index].device_slot == device) {
                        run_expert_job(layer, jobs[index], input);
                    }
                }
            }));
        }
        for (auto& worker : workers) worker.get();
        const auto aggregation_started = std::chrono::steady_clock::now();
        std::fill(output.begin(), output.end(), 0.0F);
        for (const auto& job : jobs) {
            if (!job.errors.empty()) {
                result.errors.insert(result.errors.end(), job.errors.begin(), job.errors.end());
                return result;
            }
            if (config.diagnostic_trace) {
                std::cerr << "[expert-state] layer=" << layer
                          << " expert=" << job.expert
                          << " device_slot=" << job.device_slot
                          << " rows=" << job.rows.size()
                          << " hash=" << std::hex << state_hash(job.output) << std::dec << '\n';
            }
            for (std::size_t local_row = 0; local_row < job.rows.size(); ++local_row) {
                auto destination = output.subspan(
                    static_cast<std::size_t>(job.rows[local_row]) * kHidden, kHidden);
                const auto* source = job.output.data() + local_row * kHidden;
                for (std::uint32_t column = 0; column < kHidden; ++column) {
                    destination[column] += source[column] * job.weights[local_row];
                }
            }
        }
        host_aggregation_nanoseconds += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - aggregation_started).count());

        std::vector<float> shared(output.size());
        result = mlp_triplet(layer_device(layer), prefix + "shared_experts.",
                             kExpertIntermediate, input, rows, shared);
        if (!result.ok()) return result;
        const auto shared_aggregation_started = std::chrono::steady_clock::now();
        for (std::size_t index = 0; index < output.size(); ++index) output[index] += shared[index];
        host_aggregation_nanoseconds += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - shared_aggregation_started).count());
        return result;
    }

    ValidationResult forward_layers(std::span<float> hidden, std::uint32_t rows,
                                    std::uint32_t position_base, bool prefill) {
        ValidationResult result;
        std::vector<float> normalized(hidden.size());
        std::vector<float> branch(hidden.size());
        for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
            const auto layer_started = std::chrono::steady_clock::now();
            const auto prefix = layer_prefix(layer);
            result = norm_rows(normalized, hidden, rows, prefix + "input_layernorm.weight");
            if (!result.ok()) return result;
            result = attention(layer, normalized, rows, position_base, branch);
            if (!result.ok()) return result;
            for (std::size_t index = 0; index < hidden.size(); ++index) hidden[index] += branch[index];
            if (config.diagnostic_trace) {
                std::cerr << "[state] phase=" << (rows > 1U ? "prefill" : "decode")
                          << " position=" << position_base << " rows=" << rows
                          << " layer=" << layer << " stage=attention hash=" << std::hex
                          << state_hash(hidden) << std::dec << '\n';
            }
            result = norm_rows(normalized, hidden, rows,
                               prefix + "post_attention_layernorm.weight");
            if (!result.ok()) return result;
            if (layer < 3U) {
                result = mlp_triplet(layer_device(layer), prefix + "mlp.",
                                     kDenseIntermediate, normalized, rows, branch);
            } else {
                result = sparse_mlp(layer, normalized, rows, position_base, prefill, branch);
            }
            if (!result.ok()) return result;
            for (std::size_t index = 0; index < hidden.size(); ++index) hidden[index] += branch[index];
            if (config.diagnostic_trace) {
                std::cerr << "[state] phase=" << (rows > 1U ? "prefill" : "decode")
                          << " position=" << position_base << " rows=" << rows
                          << " layer=" << layer << " stage=mlp hash=" << std::hex
                          << state_hash(hidden) << std::dec << '\n';
            }
            if (config.verbose) {
                const auto elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - layer_started).count();
                std::cerr << '[' << (rows > 1U ? "prefill" : "decode") << "] layer "
                          << (layer + 1U) << '/' << kLayers << " rows=" << rows
                          << " position=" << position_base << " seconds=" << elapsed << '\n';
            }
        }
        return result;
    }

    ParseResult<std::uint32_t> forward(std::span<const std::uint32_t> token_ids,
                                       std::uint32_t position_base, bool prefill) {
        ParseResult<std::uint32_t> result;
        const auto rows = static_cast<std::uint32_t>(token_ids.size());
        if (rows == 0U || position_base + rows > config.maximum_context_tokens) {
            result.errors.emplace_back("forward pass exceeds the configured context ceiling");
            return result;
        }
        std::vector<float> hidden(static_cast<std::size_t>(rows) * kHidden);
        auto status = embed(token_ids, hidden);
        if (!status.ok()) {
            result.errors = std::move(status.errors);
            return result;
        }
        status = forward_layers(hidden, rows, position_base, prefill);
        if (!status.ok()) {
            result.errors = std::move(status.errors);
            return result;
        }
        auto final_norm = host_tensor("model.norm.weight", kHidden);
        if (!final_norm.ok()) {
            result.errors = std::move(final_norm.errors);
            return result;
        }
        std::vector<float> normalized(kHidden);
        const auto last = std::span<const float>(hidden).last(kHidden);
        status = glm_rms_norm_f32(normalized, last, *final_norm.value);
        if (!status.ok()) {
            result.errors = std::move(status.errors);
            return result;
        }
        std::vector<float> logits(kVocabulary);
        status = linear(layer_device(kLayers - 1U), "lm_head", kVocabulary,
                        kHidden, normalized, 1U, logits);
        if (!status.ok()) {
            result.errors = std::move(status.errors);
            return result;
        }
        std::uint32_t best = 0U;
        std::uint32_t second = 1U;
        if (logits[second] > logits[best]) std::swap(best, second);
        for (std::uint32_t token = 2U; token < kVocabulary; ++token) {
            if (logits[token] > logits[best]) {
                second = best;
                best = token;
            } else if (logits[token] > logits[second]) {
                second = token;
            }
        }
        result.value = best;
        last_second_token = second;
        last_logit_margin = logits[best] - logits[second];
        return result;
    }

    void reset_sequence() {
        host_aggregation_nanoseconds = 0U;
        for (auto& layer : kv) {
            layer.keys.clear();
            layer.values.clear();
            layer.rope.clear();
        }
    }
};

Glm52Runtime::Glm52Runtime() : impl_(std::make_unique<Impl>()) {}
Glm52Runtime::~Glm52Runtime() = default;
Glm52Runtime::Glm52Runtime(Glm52Runtime&&) noexcept = default;
Glm52Runtime& Glm52Runtime::operator=(Glm52Runtime&&) noexcept = default;

ValidationResult Glm52Runtime::initialize(const std::string& model_directory,
                                          const Glm52RuntimeConfig& config) {
    ValidationResult result;
    if (config.devices.empty()) {
        result.errors.emplace_back("GLM runtime requires at least one CUDA device");
        return result;
    }
    if (!std::isfinite(config.vram_cache_fraction) ||
        config.vram_cache_fraction <= 0.0 || config.vram_cache_fraction > 0.95) {
        result.errors.emplace_back("VRAM cache fraction must be in (0, 0.95]");
        return result;
    }
    if (config.maximum_context_tokens == 0U ||
        config.maximum_context_tokens > kDsaThreshold) {
        result.errors.emplace_back(
            "baseline exact runtime context must be within the 2,048-token full-attention region");
        return result;
    }
    if (config.host_cold_experts &&
        (config.host_worker_threads == 0U || config.host_worker_threads > 256U)) {
        result.errors.emplace_back("host expert worker count must be in [1, 256]");
        return result;
    }
    auto checkpoint = GlmCheckpointReader::open(model_directory);
    if (!checkpoint.ok()) {
        result.errors = std::move(checkpoint.errors);
        return result;
    }
    result = validate_runtime_graph_contract(*checkpoint.value);
    if (!result.ok()) return result;
    auto tokenizer = GlmTokenizer::load(
        (std::filesystem::path(model_directory) / "tokenizer.json").string());
    if (!tokenizer.ok()) {
        result.errors = std::move(tokenizer.errors);
        return result;
    }
    result = impl_->cuda.initialize(config.devices, config.detailed_timing);
    if (!result.ok()) return result;
    if (!config.route_trace_path.empty()) {
        impl_->route_trace.open(config.route_trace_path, std::ios::out | std::ios::trunc);
        if (!impl_->route_trace.is_open()) {
            result.errors.emplace_back("cannot open GLM route trace: " +
                                       config.route_trace_path);
            return result;
        }
        impl_->route_trace
            << "{\"schema\":\"strata.glm52.route_trace\",\"version\":1,"
               "\"position_base\":0,\"layer_base\":0,\"router_order\":true,"
               "\"coefficient_encoding\":\"float32-roundtrip-decimal\"}\n";
    }

    std::vector<std::uint64_t> capacities;
    std::vector<std::uint64_t> totals;
    for (const auto device : config.devices) {
        auto memory = CudaBackend::device_memory(device);
        if (!memory.ok()) {
            result.errors = std::move(memory.errors);
            return result;
        }
        const auto capacity = static_cast<std::uint64_t>(
            static_cast<double>(memory.value.free_bytes) * config.vram_cache_fraction);
        if (capacity < (2ULL << 30U)) {
            result.errors.emplace_back("CUDA device has less than 2 GiB available for weights");
            return result;
        }
        capacities.push_back(capacity);
        totals.push_back(memory.value.total_bytes);
    }
    const auto smallest = *std::min_element(totals.begin(), totals.end());
    std::vector<std::size_t> schedule;
    for (std::size_t slot = 0; slot < totals.size(); ++slot) {
        const auto shares = std::max<std::uint64_t>(
            1U, (totals[slot] * 2U + smallest / 2U) / smallest);
        for (std::uint64_t count = 0; count < shares; ++count) schedule.push_back(slot);
    }

    impl_->config = config;
    impl_->checkpoint = std::move(checkpoint.value);
    impl_->tokenizer = std::move(tokenizer.value);
    impl_->devices = config.devices;
    impl_->capacities = capacities;
    impl_->device_schedule = std::move(schedule);
    impl_->weights = std::make_unique<WeightCache>(*impl_->checkpoint, impl_->cuda,
                                                   impl_->devices, capacities);
    if (config.host_cold_experts) {
        impl_->host_workers =
            std::make_unique<HostWorkerPool>(config.host_worker_threads);
    }
    if (config.verbose) {
        for (std::size_t slot = 0; slot < impl_->devices.size(); ++slot) {
            std::cerr << "[hardware] cuda=" << impl_->devices[slot]
                      << " vram_cache_bytes=" << capacities[slot] << '\n';
        }
    }
    result = impl_->warmup_resident_spine();
    if (!result.ok()) return result;
    impl_->initialized = true;
    return result;
}

Glm52GenerationResult Glm52Runtime::generate(std::string_view prompt,
                                             std::uint32_t maximum_new_tokens) {
    Glm52GenerationResult result;
    if (!impl_->initialized) {
        result.errors.emplace_back("GLM runtime is not initialized");
        return result;
    }
    if (maximum_new_tokens == 0U) {
        result.errors.emplace_back("maximum_new_tokens must be positive");
        return result;
    }
    const auto rendered = render_glm52_user_prompt(prompt);
    auto encoded = impl_->tokenizer.encode(rendered);
    if (!encoded.ok()) {
        result.errors = std::move(encoded.errors);
        return result;
    }
    if (encoded.value.size() + maximum_new_tokens > impl_->config.maximum_context_tokens) {
        result.errors.emplace_back("prompt and requested generation exceed the context ceiling");
        return result;
    }
    result.prompt_token_ids = encoded.value;
    impl_->active_request_id = impl_->config.request_id + impl_->generated_requests++;
    impl_->reset_sequence();
    const auto reads_before = impl_->checkpoint->stats();
    const auto cuda_before = impl_->cuda.stats();
    const auto cache_before = impl_->weights->stats();
    const auto host_before = impl_->host_stats();
    const auto aggregation_before = impl_->host_aggregation_nanoseconds;
    const auto prefill_start = std::chrono::steady_clock::now();
    auto next = impl_->forward(result.prompt_token_ids, 0U, true);
    result.metrics.prefill_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - prefill_start).count();
    result.metrics.prompt_tokens = result.prompt_token_ids.size();
    if (!next.ok()) {
        result.errors = std::move(next.errors);
        return result;
    }
    const auto reads_after_prefill = impl_->checkpoint->stats();
    const auto cuda_after_prefill = impl_->cuda.stats();
    const auto cache_after_prefill = impl_->weights->stats();
    const auto host_after_prefill = impl_->host_stats();
    const auto aggregation_after_prefill = impl_->host_aggregation_nanoseconds;
    if (impl_->config.verbose) {
        std::cerr << "[token] position=" << result.prompt_token_ids.size()
                  << " id=" << next.value << " second=" << impl_->last_second_token
                  << " margin=" << impl_->last_logit_margin << '\n';
    }

    constexpr std::array<std::uint32_t, 3> stop_ids{154820U, 154827U, 154829U};
    std::uint32_t position = static_cast<std::uint32_t>(result.prompt_token_ids.size());
    auto is_stop = [&stop_ids](std::uint32_t token) {
        return std::find(stop_ids.begin(), stop_ids.end(), token) != stop_ids.end();
    };
    if (!is_stop(next.value)) result.generated_token_ids.push_back(next.value);
    const auto decode_start = std::chrono::steady_clock::now();
    while (!is_stop(next.value) && result.generated_token_ids.size() < maximum_new_tokens) {
        const std::array<std::uint32_t, 1> input{next.value};
        next = impl_->forward(input, position++, false);
        if (!next.ok()) {
            result.errors = std::move(next.errors);
            return result;
        }
        ++result.metrics.decode_tokens;
        if (impl_->config.host_cold_experts &&
            result.metrics.decode_tokens % 32U == 0U) {
            const auto sweep_started = std::chrono::steady_clock::now();
            impl_->checkpoint->release_mapped_views();
            impl_->host_mapping_sweeps.fetch_add(1U, std::memory_order_relaxed);
            impl_->host_mapping_sweep_nanoseconds.fetch_add(
                static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - sweep_started).count()),
                std::memory_order_relaxed);
        }
        if (impl_->config.verbose) {
            std::cerr << "[token] position=" << position << " id=" << next.value
                      << " second=" << impl_->last_second_token
                      << " margin=" << impl_->last_logit_margin << '\n';
        }
        if (!is_stop(next.value)) result.generated_token_ids.push_back(next.value);
    }
    result.metrics.decode_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - decode_start).count();
    auto decoded = impl_->tokenizer.decode(result.generated_token_ids);
    if (!decoded.ok()) {
        result.errors = std::move(decoded.errors);
        return result;
    }
    result.text = std::move(decoded.value);
    const auto reads_after_decode = impl_->checkpoint->stats();
    const auto cuda_after_decode = impl_->cuda.stats();
    const auto cache_after_decode = impl_->weights->stats();
    const auto host_after_decode = impl_->host_stats();
    const auto aggregation_after_decode = impl_->host_aggregation_nanoseconds;
    result.metrics.prefill.checkpoint_reads = checkpoint_delta(reads_after_prefill, reads_before);
    result.metrics.prefill.cuda = cuda_delta(cuda_after_prefill, cuda_before);
    result.metrics.prefill.cache = cache_delta(cache_after_prefill, cache_before);
    result.metrics.prefill.host_experts = host_expert_delta(
        host_after_prefill, host_before);
    result.metrics.prefill.host_aggregation_nanoseconds =
        aggregation_after_prefill - aggregation_before;
    result.metrics.decode.checkpoint_reads = checkpoint_delta(
        reads_after_decode, reads_after_prefill);
    result.metrics.decode.cuda = cuda_delta(cuda_after_decode, cuda_after_prefill);
    result.metrics.decode.cache = cache_delta(cache_after_decode, cache_after_prefill);
    result.metrics.decode.host_experts = host_expert_delta(
        host_after_decode, host_after_prefill);
    result.metrics.decode.host_aggregation_nanoseconds =
        aggregation_after_decode - aggregation_after_prefill;
    result.metrics.checkpoint_reads = checkpoint_delta(reads_after_decode, reads_before);
    result.metrics.cuda = cuda_delta(cuda_after_decode, cuda_before);
    result.metrics.cache = cache_delta(cache_after_decode, cache_before);
    result.metrics.host_experts = host_expert_delta(host_after_decode, host_before);
    result.metrics.detailed_timing = impl_->config.detailed_timing;
    if (impl_->route_trace.is_open()) {
        impl_->route_trace.flush();
        if (!impl_->route_trace.good()) {
            result.errors.emplace_back("cannot flush GLM route trace");
        }
    }
    return result;
}

}  // namespace strata
