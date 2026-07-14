#include "strata/glm_runtime.hpp"

#include "strata/glm_ops.hpp"
#include "strata/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <filesystem>
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
        std::uint64_t clock{};
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
            found->second.pinned = found->second.pinned || pin;
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
        state.peak = std::max(state.peak, state.used);
        found = state.entries.emplace(key, std::move(entry)).first;
        misses_.fetch_add(1U, std::memory_order_relaxed);
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
    std::vector<int> devices;
    std::vector<std::uint64_t> capacities;
    std::vector<std::size_t> device_schedule;
    std::unordered_map<std::string, std::vector<float>> host_tensors;
    std::unordered_map<std::uint32_t, std::vector<float>> embedding_rows;
    std::vector<LayerKv> kv{kLayers};
    bool initialized{};
    std::uint32_t last_second_token{};
    float last_logit_margin{};

    std::size_t layer_device(std::uint32_t layer) const {
        return device_schedule[layer % device_schedule.size()];
    }

    std::size_t expert_device(std::uint32_t expert) const {
        return device_schedule[expert % device_schedule.size()];
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
                                std::uint32_t rows, std::span<float> output) {
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

        std::vector<std::future<void>> workers;
        for (std::size_t device = 0; device < devices.size(); ++device) {
            workers.push_back(std::async(std::launch::async, [&, device] {
                for (auto& job : jobs) {
                    if (job.device_slot == device) run_expert_job(layer, job, input);
                }
            }));
        }
        for (auto& worker : workers) worker.get();
        std::fill(output.begin(), output.end(), 0.0F);
        for (const auto& job : jobs) {
            if (!job.errors.empty()) {
                result.errors.insert(result.errors.end(), job.errors.begin(), job.errors.end());
                return result;
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

        std::vector<float> shared(output.size());
        result = mlp_triplet(layer_device(layer), prefix + "shared_experts.",
                             kExpertIntermediate, input, rows, shared);
        if (!result.ok()) return result;
        for (std::size_t index = 0; index < output.size(); ++index) output[index] += shared[index];
        return result;
    }

    ValidationResult forward_layers(std::span<float> hidden, std::uint32_t rows,
                                    std::uint32_t position_base) {
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
            result = norm_rows(normalized, hidden, rows,
                               prefix + "post_attention_layernorm.weight");
            if (!result.ok()) return result;
            if (layer < 3U) {
                result = mlp_triplet(layer_device(layer), prefix + "mlp.",
                                     kDenseIntermediate, normalized, rows, branch);
            } else {
                result = sparse_mlp(layer, normalized, rows, branch);
            }
            if (!result.ok()) return result;
            for (std::size_t index = 0; index < hidden.size(); ++index) hidden[index] += branch[index];
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
                                       std::uint32_t position_base) {
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
        status = forward_layers(hidden, rows, position_base);
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
        config.vram_cache_fraction <= 0.0 || config.vram_cache_fraction > 0.90) {
        result.errors.emplace_back("VRAM cache fraction must be in (0, 0.90]");
        return result;
    }
    if (config.maximum_context_tokens == 0U ||
        config.maximum_context_tokens > kDsaThreshold) {
        result.errors.emplace_back(
            "baseline exact runtime context must be within the 2,048-token full-attention region");
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
    result = impl_->cuda.initialize(config.devices);
    if (!result.ok()) return result;

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
    impl_->reset_sequence();
    const auto reads_before = impl_->checkpoint->stats();
    const auto cuda_before = impl_->cuda.stats();
    const auto cache_before = impl_->weights->stats();
    const auto prefill_start = std::chrono::steady_clock::now();
    auto next = impl_->forward(result.prompt_token_ids, 0U);
    result.metrics.prefill_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - prefill_start).count();
    result.metrics.prompt_tokens = result.prompt_token_ids.size();
    if (!next.ok()) {
        result.errors = std::move(next.errors);
        return result;
    }
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
        next = impl_->forward(input, position++);
        if (!next.ok()) {
            result.errors = std::move(next.errors);
            return result;
        }
        ++result.metrics.decode_tokens;
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
    result.metrics.checkpoint_reads = impl_->checkpoint->stats();
    result.metrics.checkpoint_reads.calls -= reads_before.calls;
    result.metrics.checkpoint_reads.bytes -= reads_before.bytes;
    result.metrics.checkpoint_reads.nanoseconds -= reads_before.nanoseconds;
    result.metrics.cuda = impl_->cuda.stats();
    result.metrics.cuda.weight_upload_bytes -= cuda_before.weight_upload_bytes;
    result.metrics.cuda.activation_h2d_bytes -= cuda_before.activation_h2d_bytes;
    result.metrics.cuda.activation_d2h_bytes -= cuda_before.activation_d2h_bytes;
    result.metrics.cuda.matmul_calls -= cuda_before.matmul_calls;
    result.metrics.cache = impl_->weights->stats();
    result.metrics.cache.hits -= cache_before.hits;
    result.metrics.cache.misses -= cache_before.misses;
    result.metrics.cache.evictions -= cache_before.evictions;
    return result;
}

}  // namespace strata
