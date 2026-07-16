#include "strata/deepseek_runtime.hpp"

#include "strata/deepseek_ops.hpp"
#include "strata/glm_ops.hpp"
#include "strata/tokenizer.hpp"
#include "strata/worker_pool.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <span>
#include <unordered_map>

namespace strata {

namespace {

constexpr std::uint32_t kHidden = 4096U;
constexpr std::uint32_t kLayers = 43U;
constexpr std::uint32_t kHeads = 64U;
constexpr std::uint32_t kHeadDim = 512U;
constexpr std::uint32_t kRopeDim = 64U;
constexpr std::uint32_t kQueryRank = 1024U;
constexpr std::uint32_t kOutputRank = 1024U;
constexpr std::uint32_t kOutputGroups = 8U;
constexpr std::uint32_t kWindow = 128U;
constexpr std::uint32_t kExperts = 256U;
constexpr std::uint32_t kTopK = 6U;
constexpr std::uint32_t kExpertIntermediate = 2048U;
constexpr std::uint32_t kVocabulary = 129280U;
constexpr std::uint32_t kMhc = 4U;
constexpr std::uint32_t kMix = 24U;
constexpr std::uint64_t kDeviceWorkspaceReserve = 256ULL << 20U;
constexpr float kRmsEpsilon = 1.0e-6F;
constexpr float kAttentionScale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
constexpr std::uint64_t kDiagnosticFnvOffset = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t kDiagnosticFnvPrime = 1'099'511'628'211ULL;

[[nodiscard]] std::string layer_prefix(std::uint32_t layer) {
    return "layers." + std::to_string(layer) + ".";
}

void append_errors(ValidationResult& result, std::vector<std::string> errors,
                   std::string_view context = {}) {
    for (auto& error : errors) {
        if (!context.empty()) error = std::string(context) + ": " + error;
        result.errors.push_back(std::move(error));
    }
}

[[nodiscard]] std::uint64_t diagnostic_hash_byte(
    std::uint64_t hash, std::uint8_t value) noexcept {
    return (hash ^ value) * kDiagnosticFnvPrime;
}

[[nodiscard]] std::uint64_t diagnostic_hash_u32(
    std::uint64_t hash, std::uint32_t value) noexcept {
    for (std::uint32_t shift = 0U; shift < 32U; shift += 8U) {
        hash = diagnostic_hash_byte(
            hash, static_cast<std::uint8_t>(value >> shift));
    }
    return hash;
}

[[nodiscard]] std::uint64_t diagnostic_hash_u64(
    std::uint64_t hash, std::uint64_t value) noexcept {
    for (std::uint32_t shift = 0U; shift < 64U; shift += 8U) {
        hash = diagnostic_hash_byte(
            hash, static_cast<std::uint8_t>(value >> shift));
    }
    return hash;
}

[[nodiscard]] float round_bf16(float value) noexcept {
    auto bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7F80'0000U) == 0x7F80'0000U) return value;
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return std::bit_cast<float>(bits & 0xFFFF'0000U);
}

void round_bf16(std::span<float> values) noexcept {
    for (auto& value : values) value = round_bf16(value);
}

[[nodiscard]] float quantize_e4m3(float value) noexcept {
    const float magnitude = std::min(std::abs(value), 448.0F);
    float quantized = 0.0F;
    if (magnitude < 0.015625F) {
        quantized = std::nearbyint(std::ldexp(magnitude, 9)) * std::ldexp(1.0F, -9);
    } else {
        int exponent = 0;
        static_cast<void>(std::frexp(magnitude, &exponent));
        exponent = std::clamp(exponent - 1, -6, 8);
        const float step = std::ldexp(1.0F, exponent - 3);
        quantized = std::min(std::nearbyint(magnitude / step) * step, 448.0F);
    }
    return std::copysign(quantized, value);
}

void quantize_activation_in_place(std::span<float> values,
                                  std::uint32_t group_size) {
    for (std::size_t begin = 0U; begin < values.size(); begin += group_size) {
        const auto count = std::min<std::size_t>(group_size, values.size() - begin);
        float maximum = 0.0F;
        for (std::size_t index = 0U; index < count; ++index) {
            maximum = std::max(maximum, std::abs(values[begin + index]));
        }
        float scale = 1.0F;
        if (maximum > 0.0F) {
            scale = std::exp2(std::ceil(std::log2(maximum / 448.0F)));
        }
        for (std::size_t index = 0U; index < count; ++index) {
            values[begin + index] = round_bf16(
                quantize_e4m3(values[begin + index] / scale) * scale);
        }
    }
}

[[nodiscard]] std::vector<float> rope_frequencies(std::uint32_t compression_ratio) {
    const float base = compression_ratio == 0U ? 10'000.0F : 160'000.0F;
    std::vector<float> result(kRopeDim / 2U);
    for (std::uint32_t index = 0U; index < result.size(); ++index) {
        result[index] = 1.0F /
            std::pow(base, static_cast<float>(2U * index) / static_cast<float>(kRopeDim));
    }
    if (compression_ratio == 0U) return result;
    constexpr float original = 65'536.0F;
    const auto correction = [base](float rotations) {
        return static_cast<float>(kRopeDim) *
               std::log(original / (rotations * 2.0F * std::numbers::pi_v<float>)) /
               (2.0F * std::log(base));
    };
    const float low = std::max(0.0F, std::floor(correction(32.0F)));
    const float high = std::min(static_cast<float>(kRopeDim - 1U),
                                std::ceil(correction(1.0F)));
    for (std::uint32_t index = 0U; index < result.size(); ++index) {
        const float ramp = std::clamp((static_cast<float>(index) - low) /
                                          std::max(0.001F, high - low),
                                      0.0F, 1.0F);
        const float smooth = 1.0F - ramp;
        result[index] = result[index] / 16.0F * (1.0F - smooth) +
                        result[index] * smooth;
    }
    return result;
}

void apply_rope(std::span<float> values, std::uint64_t position,
                std::span<const float> frequencies, bool inverse = false) {
    for (std::size_t index = 0U; index < frequencies.size(); ++index) {
        const float angle = static_cast<float>(position) * frequencies[index] *
                            (inverse ? -1.0F : 1.0F);
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        const float first = values[index * 2U];
        const float second = values[index * 2U + 1U];
        values[index * 2U] = first * cosine - second * sine;
        values[index * 2U + 1U] = second * cosine + first * sine;
    }
}

[[nodiscard]] Dsv4CheckpointReadStats read_delta(
    const Dsv4CheckpointReadStats& after,
    const Dsv4CheckpointReadStats& before) noexcept {
    return {after.calls - before.calls, after.bytes - before.bytes,
            after.nanoseconds - before.nanoseconds};
}

[[nodiscard]] CudaBackendStats::Device cuda_device_delta(
    const CudaBackendStats::Device& after,
    const CudaBackendStats::Device& before) {
    CudaBackendStats::Device result;
    result.device = after.device;
#define STRATA_DSV4_CUDA_DEVICE_DELTA(field) result.field = after.field - before.field
    STRATA_DSV4_CUDA_DEVICE_DELTA(weight_upload_bytes);
    STRATA_DSV4_CUDA_DEVICE_DELTA(activation_h2d_bytes);
    STRATA_DSV4_CUDA_DEVICE_DELTA(activation_d2h_bytes);
    STRATA_DSV4_CUDA_DEVICE_DELTA(matmul_calls);
    STRATA_DSV4_CUDA_DEVICE_DELTA(weight_allocation_calls);
    STRATA_DSV4_CUDA_DEVICE_DELTA(weight_allocation_bytes);
    STRATA_DSV4_CUDA_DEVICE_DELTA(workspace_allocation_calls);
    STRATA_DSV4_CUDA_DEVICE_DELTA(workspace_allocation_bytes);
    STRATA_DSV4_CUDA_DEVICE_DELTA(synchronization_calls);
    STRATA_DSV4_CUDA_DEVICE_DELTA(synchronization_nanoseconds);
    STRATA_DSV4_CUDA_DEVICE_DELTA(upload_wait_nanoseconds);
    STRATA_DSV4_CUDA_DEVICE_DELTA(activation_h2d_nanoseconds);
    STRATA_DSV4_CUDA_DEVICE_DELTA(kernel_nanoseconds);
    STRATA_DSV4_CUDA_DEVICE_DELTA(activation_d2h_nanoseconds);
    STRATA_DSV4_CUDA_DEVICE_DELTA(deepseek_moe_calls);
    STRATA_DSV4_CUDA_DEVICE_DELTA(deepseek_moe_kernel_launches);
    STRATA_DSV4_CUDA_DEVICE_DELTA(deepseek_moe_h2d_transfers);
    STRATA_DSV4_CUDA_DEVICE_DELTA(deepseek_moe_d2h_transfers);
    STRATA_DSV4_CUDA_DEVICE_DELTA(deepseek_moe_h2d_bytes);
    STRATA_DSV4_CUDA_DEVICE_DELTA(deepseek_moe_d2h_bytes);
    STRATA_DSV4_CUDA_DEVICE_DELTA(deepseek_moe_h2d_nanoseconds);
    STRATA_DSV4_CUDA_DEVICE_DELTA(deepseek_moe_kernel_nanoseconds);
    STRATA_DSV4_CUDA_DEVICE_DELTA(deepseek_moe_d2h_nanoseconds);
    STRATA_DSV4_CUDA_DEVICE_DELTA(deepseek_moe_nanoseconds);
#undef STRATA_DSV4_CUDA_DEVICE_DELTA
    return result;
}

[[nodiscard]] CudaBackendStats cuda_delta(const CudaBackendStats& after,
                                           const CudaBackendStats& before) {
    CudaBackendStats result;
#define STRATA_DSV4_CUDA_DELTA(field) result.field = after.field - before.field
    STRATA_DSV4_CUDA_DELTA(weight_upload_bytes);
    STRATA_DSV4_CUDA_DELTA(activation_h2d_bytes);
    STRATA_DSV4_CUDA_DELTA(activation_d2h_bytes);
    STRATA_DSV4_CUDA_DELTA(matmul_calls);
    STRATA_DSV4_CUDA_DELTA(weight_allocation_calls);
    STRATA_DSV4_CUDA_DELTA(weight_allocation_bytes);
    STRATA_DSV4_CUDA_DELTA(workspace_allocation_calls);
    STRATA_DSV4_CUDA_DELTA(workspace_allocation_bytes);
    STRATA_DSV4_CUDA_DELTA(synchronization_calls);
    STRATA_DSV4_CUDA_DELTA(synchronization_nanoseconds);
    STRATA_DSV4_CUDA_DELTA(upload_wait_nanoseconds);
    STRATA_DSV4_CUDA_DELTA(activation_h2d_nanoseconds);
    STRATA_DSV4_CUDA_DELTA(kernel_nanoseconds);
    STRATA_DSV4_CUDA_DELTA(activation_d2h_nanoseconds);
    STRATA_DSV4_CUDA_DELTA(deepseek_moe_calls);
    STRATA_DSV4_CUDA_DELTA(deepseek_moe_kernel_launches);
    STRATA_DSV4_CUDA_DELTA(deepseek_moe_h2d_transfers);
    STRATA_DSV4_CUDA_DELTA(deepseek_moe_d2h_transfers);
    STRATA_DSV4_CUDA_DELTA(deepseek_moe_h2d_bytes);
    STRATA_DSV4_CUDA_DELTA(deepseek_moe_d2h_bytes);
    STRATA_DSV4_CUDA_DELTA(deepseek_moe_h2d_nanoseconds);
    STRATA_DSV4_CUDA_DELTA(deepseek_moe_kernel_nanoseconds);
    STRATA_DSV4_CUDA_DELTA(deepseek_moe_d2h_nanoseconds);
    STRATA_DSV4_CUDA_DELTA(deepseek_moe_nanoseconds);
#undef STRATA_DSV4_CUDA_DELTA
    result.synchronization_nanoseconds = 0U;
    result.upload_wait_nanoseconds = 0U;
    result.activation_h2d_nanoseconds = 0U;
    result.kernel_nanoseconds = 0U;
    result.activation_d2h_nanoseconds = 0U;
    result.deepseek_moe_h2d_nanoseconds = 0U;
    result.deepseek_moe_kernel_nanoseconds = 0U;
    result.deepseek_moe_d2h_nanoseconds = 0U;
    result.deepseek_moe_nanoseconds = 0U;
    for (const auto& device_after : after.devices) {
        const auto found = std::find_if(
            before.devices.begin(), before.devices.end(),
            [&device_after](const auto& value) {
                return value.device == device_after.device;
            });
        result.devices.push_back(found == before.devices.end()
                                     ? device_after
                                     : cuda_device_delta(device_after, *found));
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
        result.deepseek_moe_h2d_nanoseconds = std::max(
            result.deepseek_moe_h2d_nanoseconds,
            delta.deepseek_moe_h2d_nanoseconds);
        result.deepseek_moe_kernel_nanoseconds = std::max(
            result.deepseek_moe_kernel_nanoseconds,
            delta.deepseek_moe_kernel_nanoseconds);
        result.deepseek_moe_d2h_nanoseconds = std::max(
            result.deepseek_moe_d2h_nanoseconds,
            delta.deepseek_moe_d2h_nanoseconds);
        result.deepseek_moe_nanoseconds = std::max(
            result.deepseek_moe_nanoseconds, delta.deepseek_moe_nanoseconds);
    }
    return result;
}

[[nodiscard]] Dsv4CacheStats cache_delta(const Dsv4CacheStats& after,
                                         const Dsv4CacheStats& before) {
    Dsv4CacheStats result = after;
    result.hits -= before.hits;
    result.misses -= before.misses;
    result.evictions -= before.evictions;
    result.lease_acquires -= before.lease_acquires;
    result.lease_releases -= before.lease_releases;
    return result;
}

[[nodiscard]] Dsv4DeviceMoeStats device_moe_delta(
    const Dsv4DeviceMoeStats& after,
    const Dsv4DeviceMoeStats& before) noexcept {
    return {after.batches - before.batches,
            after.device_commands - before.device_commands,
            after.routed_experts - before.routed_experts,
            after.shared_experts - before.shared_experts,
            after.nanoseconds - before.nanoseconds};
}

[[nodiscard]] std::uint64_t linear_bytes(const Dsv4CheckpointReader& checkpoint,
                                         std::string_view base) {
    const auto* weight = checkpoint.find(std::string(base) + ".weight");
    if (weight == nullptr) return 0U;
    if (base.ends_with(".attn.wo_a") &&
        weight->encoding == Dsv4TensorEncoding::Fp8E4m3Block128) {
        return weight->source_bytes * 2U;
    }
    std::uint64_t result = weight->source_bytes;
    if (weight->encoding != Dsv4TensorEncoding::Plain) {
        const auto* scale = checkpoint.find(std::string(base) + ".scale");
        if (scale == nullptr) return 0U;
        result += scale->source_bytes;
    }
    return result;
}

class Dsv4WeightCache {
    struct Entry {
        CudaWeight weight;
        std::uint64_t last_use{};
        std::uint32_t leases{};
        bool pinned{};
    };
    struct State {
        std::unordered_map<std::string, Entry> entries;
        std::uint64_t capacity{};
        std::uint64_t used{};
        std::uint64_t pinned{};
        std::uint64_t clock{};
    };

public:
    class Lease {
    public:
        Lease() = default;
        ~Lease() { reset(); }
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;

        Lease(Lease&& other) noexcept
            : owner_(other.owner_), entry_(other.entry_) {
            other.owner_ = nullptr;
            other.entry_ = nullptr;
        }

        Lease& operator=(Lease&& other) noexcept {
            if (this == &other) return *this;
            reset();
            owner_ = other.owner_;
            entry_ = other.entry_;
            other.owner_ = nullptr;
            other.entry_ = nullptr;
            return *this;
        }

        [[nodiscard]] const CudaWeight& weight() const noexcept {
            return entry_->weight;
        }

    private:
        friend class Dsv4WeightCache;

        Lease(Dsv4WeightCache* owner, Entry* entry) noexcept
            : owner_(owner), entry_(entry) {}

        void reset() noexcept {
            if (owner_ != nullptr && entry_ != nullptr) owner_->release(entry_);
            owner_ = nullptr;
            entry_ = nullptr;
        }

        Dsv4WeightCache* owner_{};
        Entry* entry_{};
    };

    Dsv4WeightCache(Dsv4CheckpointReader& checkpoint,
                    Dsv4ResidentWeightStore& resident, CudaBackend& backend,
                    std::vector<int> devices, std::vector<std::uint64_t> capacities)
        : checkpoint_(checkpoint), resident_(resident), backend_(backend),
          devices_(std::move(devices)) {
        for (const auto capacity : capacities) states_.push_back(State{{}, capacity});
    }

    ValidationResult preload(std::size_t slot, std::string_view base,
                             std::uint64_t rows, std::uint64_t columns) {
        Entry* entry = nullptr;
        return ensure(slot, base, rows, columns, true, entry);
    }

    ValidationResult acquire(std::size_t slot, std::string_view base,
                             std::uint64_t rows, std::uint64_t columns,
                             Lease& output) {
        output.reset();
        Entry* entry = nullptr;
        auto result = ensure(slot, base, rows, columns, false, entry);
        if (!result.ok()) return result;
        ++entry->leases;
        ++lease_acquires_;
        output = Lease(this, entry);
        return result;
    }

    ValidationResult validate_atomic_expert_capacity(
        std::uint64_t required_bytes) const {
        ValidationResult result;
        for (std::size_t slot = 0U; slot < states_.size(); ++slot) {
            const auto& state = states_[slot];
            const auto available = state.capacity >= state.pinned
                                       ? state.capacity - state.pinned
                                       : 0U;
            if (required_bytes > available) {
                result.errors.emplace_back(
                    "DeepSeek device " + std::to_string(devices_[slot]) +
                    " cannot lease the worst-case exact top-k expert set");
            }
        }
        return result;
    }

    ValidationResult matmul(std::size_t slot, std::string_view base,
                            std::uint64_t output_columns,
                            std::uint64_t input_columns,
                            std::span<const float> input, std::uint32_t rows,
                            std::span<float> output, bool bf16_output = true) {
        Entry* entry = nullptr;
        auto result = ensure(slot, base, output_columns, input_columns, false, entry);
        if (!result.ok()) return result;
        result = backend_.matmul(entry->weight, input, rows, output);
        if (result.ok() && bf16_output) round_bf16(output);
        return result;
    }

    ValidationResult grouped(std::size_t slot, std::string_view base,
                             std::uint64_t output_columns,
                             std::uint64_t input_columns,
                             std::span<const float> input,
                             std::uint32_t groups,
                             std::uint64_t rows_per_group,
                             std::span<float> output) {
        Entry* entry = nullptr;
        auto result = ensure(slot, base, output_columns, input_columns, false, entry);
        if (!result.ok()) return result;
        result = backend_.matmul_grouped(entry->weight, input, groups,
                                         rows_per_group, output);
        if (result.ok()) round_bf16(output);
        return result;
    }

    [[nodiscard]] Dsv4CacheStats stats() const {
        Dsv4CacheStats result;
        result.hits = hits_;
        result.misses = misses_;
        result.evictions = evictions_;
        result.lease_acquires = lease_acquires_;
        result.lease_releases = lease_releases_;
        for (const auto& state : states_) {
            result.used_bytes.push_back(state.used);
            result.capacity_bytes.push_back(state.capacity);
            result.pinned_bytes.push_back(state.pinned);
            std::uint64_t leased_bytes = 0U;
            std::uint64_t active_leases = 0U;
            for (const auto& [name, entry] : state.entries) {
                static_cast<void>(name);
                if (entry.leases == 0U) continue;
                leased_bytes += entry.weight.device_bytes();
                active_leases += entry.leases;
            }
            result.leased_bytes.push_back(leased_bytes);
            result.active_leases.push_back(active_leases);
        }
        return result;
    }

private:
    ValidationResult ensure(std::size_t slot, std::string_view base,
                            std::uint64_t rows, std::uint64_t columns,
                            bool pin, Entry*& output) {
        ValidationResult result;
        if (slot >= states_.size()) {
            result.errors.emplace_back("DeepSeek linear targets an invalid CUDA slot");
            return result;
        }
        auto& state = states_[slot];
        ++state.clock;
        const std::string key(base);
        auto found = state.entries.find(key);
        if (found != state.entries.end()) {
            found->second.last_use = state.clock;
            if (pin && !found->second.pinned) {
                found->second.pinned = true;
                state.pinned += found->second.weight.device_bytes();
            }
            ++hits_;
            output = &found->second;
            return result;
        }
        const auto bytes = linear_bytes(checkpoint_, base);
        if (bytes == 0U || bytes > state.capacity) {
            result.errors.emplace_back("DeepSeek linear is absent or exceeds device cache: " +
                                       key);
            return result;
        }
        while (state.used + bytes > state.capacity) {
            auto victim = state.entries.end();
            for (auto candidate = state.entries.begin(); candidate != state.entries.end();
                 ++candidate) {
                if (candidate->second.pinned || candidate->second.leases != 0U) continue;
                if (victim == state.entries.end() ||
                    candidate->second.last_use < victim->second.last_use) {
                    victim = candidate;
                }
            }
            if (victim == state.entries.end()) {
                const bool in_flight = std::any_of(
                    state.entries.begin(), state.entries.end(),
                    [](const auto& candidate) {
                        return candidate.second.leases != 0U;
                    });
                result.errors.emplace_back(in_flight
                    ? "DeepSeek atomic in-flight expert set exceeds a device VRAM budget"
                    : "DeepSeek pinned resident spine exceeds a device VRAM budget");
                return result;
            }
            state.used -= victim->second.weight.device_bytes();
            state.entries.erase(victim);
            ++evictions_;
        }
        Entry entry;
        result = load_dsv4_cuda_linear(checkpoint_, &resident_, base, rows, columns,
                                        devices_[slot], backend_, entry.weight);
        if (!result.ok()) return result;
        entry.last_use = state.clock;
        entry.pinned = pin;
        state.used += entry.weight.device_bytes();
        if (pin) state.pinned += entry.weight.device_bytes();
        found = state.entries.emplace(key, std::move(entry)).first;
        ++misses_;
        output = &found->second;
        return result;
    }

    void release(Entry* entry) noexcept {
        if (entry != nullptr && entry->leases != 0U) {
            --entry->leases;
            ++lease_releases_;
        }
    }

    Dsv4CheckpointReader& checkpoint_;
    Dsv4ResidentWeightStore& resident_;
    CudaBackend& backend_;
    std::vector<int> devices_;
    std::vector<State> states_;
    std::uint64_t hits_{};
    std::uint64_t misses_{};
    std::uint64_t evictions_{};
    std::uint64_t lease_acquires_{};
    std::uint64_t lease_releases_{};
};

struct CompressorState {
    std::uint32_t ratio{};
    std::uint32_t coefficient{};
    std::vector<float> values;
    std::vector<float> scores;
    std::vector<float> compressed;
};

struct AttentionState {
    std::vector<float> sliding;
    CompressorState compressor;
    std::vector<float> frequencies;
};

}  // namespace

struct DeepSeekV4Runtime::Impl {
    Dsv4RuntimeConfig config;
    Dsv4MemoryPlan memory;
    Dsv4GenerationMetrics initialization_metrics;
    std::unique_ptr<Dsv4CheckpointReader> checkpoint;
    Dsv4ResidentWeightStore resident;
    GlmTokenizer tokenizer;
    CudaBackend cuda;
    std::unique_ptr<Dsv4WeightCache> weights;
    std::unique_ptr<HostWorkerPool> attention_workers;
    Dsv4DiagnosticTrace diagnostics;
    Dsv4DeviceMoeStats device_moe_stats;
    std::vector<int> devices;
    std::vector<std::uint64_t> capacities;
    std::vector<std::size_t> schedule;
    std::unordered_map<std::string, std::vector<float>> host_tensors;
    std::unordered_map<std::string, std::vector<std::byte>> host_raw;
    std::array<AttentionState, kLayers> attention_state;
    std::ofstream route_trace;
    bool initialized{};

    [[nodiscard]] std::size_t layer_device(std::uint32_t layer) const {
        return schedule[layer % schedule.size()];
    }

    [[nodiscard]] std::size_t expert_device(std::uint32_t expert) const {
        return schedule[expert % schedule.size()];
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

    ParseResult<const std::vector<std::byte>*> raw_tensor(std::string name,
                                                          std::uint64_t ceiling) {
        ParseResult<const std::vector<std::byte>*> result;
        const auto found = host_raw.find(name);
        if (found != host_raw.end()) {
            result.value = &found->second;
            return result;
        }
        auto loaded = checkpoint->read(name, ceiling);
        if (!loaded.ok()) {
            result.errors = std::move(loaded.errors);
            return result;
        }
        auto inserted = host_raw.emplace(std::move(name), std::move(loaded.value));
        result.value = &inserted.first->second;
        return result;
    }

    ValidationResult linear(std::size_t slot, std::string_view base,
                            std::uint64_t outputs, std::uint64_t inputs,
                            std::span<const float> input,
                            std::span<float> output,
                            bool bf16_output = true) {
        return weights->matmul(slot, base, outputs, inputs, input, 1U,
                               output, bf16_output);
    }

    ValidationResult norm(std::span<float> output, std::span<const float> input,
                          std::string name) {
        ValidationResult result;
        auto weight = host_tensor(std::move(name), input.size());
        if (!weight.ok()) {
            append_errors(result, std::move(weight.errors));
            return result;
        }
        result = glm_rms_norm_f32(output, input, *weight.value, kRmsEpsilon);
        if (result.ok()) round_bf16(output);
        return result;
    }

    ValidationResult warmup();
    ValidationResult reset_sequence();
    void reset_diagnostics();
    void record_layer_hash(std::uint32_t position, std::uint32_t token,
                           std::uint32_t layer, std::span<const float> hidden);
    void record_logits(std::uint32_t position, std::uint32_t token,
                       std::uint32_t selected, std::span<const float> logits);
    ValidationResult embed(std::uint32_t token, std::span<float> output);
    ValidationResult compressor(std::uint32_t layer, std::span<const float> input,
                                std::uint32_t position);
    ValidationResult attention(std::uint32_t layer, std::span<const float> input,
                               std::uint32_t position, std::span<float> output);
    ValidationResult expert(std::uint32_t layer, std::uint32_t expert_id,
                            float routed_coefficient,
                            std::span<const float> input, std::span<float> output);
    ValidationResult device_moe(std::uint32_t layer,
                                const Dsv4Route& route,
                                std::span<const float> input,
                                std::span<float> output);
    ValidationResult moe(std::uint32_t layer, std::uint32_t token,
                         std::span<const float> input, std::span<float> output,
                         std::uint32_t position);
    ValidationResult block(std::uint32_t layer, std::uint32_t token,
                           std::span<float> hidden, std::uint32_t position);
    ParseResult<std::uint32_t> forward_token(std::uint32_t token,
                                             std::uint32_t position,
                                             bool logits);
};

void DeepSeekV4Runtime::Impl::reset_diagnostics() {
    diagnostics = {};
    diagnostics.logit_trace_enabled = config.enable_logit_trace;
    diagnostics.layer_hash_trace_enabled = config.enable_layer_hash_trace;
    diagnostics.logit_top_k = config.logit_trace_top_k;
    if (config.enable_logit_trace) {
        diagnostics.logit_aggregate.trace_hash = diagnostic_hash_u32(
            kDiagnosticFnvOffset, config.logit_trace_top_k);
        diagnostics.logits.reserve(config.maximum_context_tokens);
    }
    if (config.enable_layer_hash_trace) {
        diagnostics.layer_hash_trace_hash = diagnostic_hash_u32(
            kDiagnosticFnvOffset, kLayers);
        diagnostics.layer_hashes.reserve(
            static_cast<std::size_t>(config.maximum_context_tokens) * kLayers);
    }
}

void DeepSeekV4Runtime::Impl::record_layer_hash(
    std::uint32_t position, std::uint32_t token, std::uint32_t layer,
    std::span<const float> hidden) {
    const auto hash = dsv4_stable_bf16_hash(hidden);
    diagnostics.layer_hashes.push_back({position, token, layer, hash});
    auto aggregate = diagnostics.layer_hash_trace_hash;
    aggregate = diagnostic_hash_u32(aggregate, position);
    aggregate = diagnostic_hash_u32(aggregate, token);
    aggregate = diagnostic_hash_u32(aggregate, layer);
    diagnostics.layer_hash_trace_hash = diagnostic_hash_u64(aggregate, hash);
}

void DeepSeekV4Runtime::Impl::record_logits(
    std::uint32_t position, std::uint32_t token, std::uint32_t selected,
    std::span<const float> logits) {
    auto analysis = analyze_dsv4_logits(logits, config.logit_trace_top_k);
    const auto& summary = analysis.summary;
    auto& aggregate = diagnostics.logit_aggregate;
    ++aggregate.forward_count;
    aggregate.value_count += summary.value_count;
    aggregate.finite_count += summary.finite_count;
    aggregate.non_finite_count += summary.non_finite_count;
    aggregate.sum += summary.sum;
    aggregate.absolute_sum += summary.absolute_sum;
    aggregate.square_sum += summary.square_sum;
    if (summary.has_finite) {
        if (!aggregate.has_finite) {
            aggregate.minimum = summary.minimum;
            aggregate.maximum = summary.maximum;
            aggregate.has_finite = true;
        } else {
            aggregate.minimum = std::min(aggregate.minimum, summary.minimum);
            aggregate.maximum = std::max(aggregate.maximum, summary.maximum);
        }
    }
    auto hash = aggregate.trace_hash;
    hash = diagnostic_hash_u32(hash, position);
    hash = diagnostic_hash_u32(hash, token);
    hash = diagnostic_hash_u32(hash, selected);
    aggregate.trace_hash = diagnostic_hash_u64(hash, summary.raw_f32_hash);
    diagnostics.logits.push_back(
        {position, token, selected, summary, std::move(analysis.top)});
}

ValidationResult DeepSeekV4Runtime::Impl::warmup() {
    ValidationResult result;
    const auto preload = [this, &result](std::size_t slot, const std::string& base,
                                         std::uint64_t rows, std::uint64_t columns) {
        if (result.ok()) result = weights->preload(slot, base, rows, columns);
    };
    const auto load_host = [this, &result](const std::string& name,
                                           std::uint64_t ceiling) {
        if (!result.ok()) return;
        auto loaded = host_tensor(name, ceiling);
        if (!loaded.ok()) append_errors(result, std::move(loaded.errors));
    };
    const auto load_raw = [this, &result](const std::string& name,
                                          std::uint64_t ceiling) {
        if (!result.ok()) return;
        auto loaded = raw_tensor(name, ceiling);
        if (!loaded.ok()) append_errors(result, std::move(loaded.errors));
    };

    const auto ratios = deepseek_v4_flash_dspark_spec().deepseek_v4.compression_ratios;
    for (std::uint32_t layer = 0U; layer < kLayers && result.ok(); ++layer) {
        const auto slot = layer_device(layer);
        const auto prefix = layer_prefix(layer);
        const auto attention = prefix + "attn.";
        preload(slot, attention + "wq_a", kQueryRank, kHidden);
        preload(slot, attention + "wq_b", kHeads * kHeadDim, kQueryRank);
        preload(slot, attention + "wkv", kHeadDim, kHidden);
        preload(slot, attention + "wo_a", kOutputGroups * kOutputRank,
                kHeads * kHeadDim / kOutputGroups);
        preload(slot, attention + "wo_b", kHidden, kOutputGroups * kOutputRank);
        preload(slot, prefix + "ffn.gate", kExperts, kHidden);
        for (const auto* operation : {"w1", "w3"}) {
            preload(slot, prefix + "ffn.shared_experts." + operation,
                    kExpertIntermediate, kHidden);
        }
        preload(slot, prefix + "ffn.shared_experts.w2", kHidden,
                kExpertIntermediate);
        load_host(attention + "q_norm.weight", kQueryRank);
        load_host(attention + "kv_norm.weight", kHeadDim);
        load_host(attention + "attn_sink", kHeads);
        load_host(prefix + "attn_norm.weight", kHidden);
        load_host(prefix + "ffn_norm.weight", kHidden);
        for (const auto* branch : {"attn", "ffn"}) {
            load_host(prefix + "hc_" + branch + "_fn", kMix * kMhc * kHidden);
            load_host(prefix + "hc_" + branch + "_scale", 3U);
            load_host(prefix + "hc_" + branch + "_base", kMix);
        }
        if (layer < 3U) {
            load_raw(prefix + "ffn.gate.tid2eid", 8ULL * kVocabulary * kTopK);
        } else {
            load_host(prefix + "ffn.gate.bias", kExperts);
        }
        const auto ratio = ratios[layer];
        if (ratio != 0U) {
            const auto coefficient = ratio == 4U ? 2U : 1U;
            const auto dimensions = coefficient * kHeadDim;
            preload(slot, attention + "compressor.wkv", dimensions, kHidden);
            preload(slot, attention + "compressor.wgate", dimensions, kHidden);
            load_host(attention + "compressor.ape",
                      static_cast<std::uint64_t>(ratio) * dimensions);
            load_host(attention + "compressor.norm.weight", kHeadDim);
            if (ratio == 4U) {
                preload(slot, attention + "indexer.wq_b", 64U * 128U, kQueryRank);
                preload(slot, attention + "indexer.weights_proj", 64U, kHidden);
                preload(slot, attention + "indexer.compressor.wkv", 2U * 128U,
                        kHidden);
                preload(slot, attention + "indexer.compressor.wgate", 2U * 128U,
                        kHidden);
                load_host(attention + "indexer.compressor.ape", 4U * 2U * 128U);
                load_host(attention + "indexer.compressor.norm.weight", 128U);
            }
        }
        if (config.verbose) {
            std::cerr << "[deepseek-load] resident spine layer " << layer + 1U
                      << '/' << kLayers << '\n';
        }
    }
    if (result.ok()) {
        preload(layer_device(kLayers - 1U), "head", kVocabulary, kHidden);
        load_host("norm.weight", kHidden);
        load_host("hc_head_fn", kMhc * kMhc * kHidden);
        load_host("hc_head_scale", 1U);
        load_host("hc_head_base", kMhc);
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::reset_sequence() {
    ValidationResult result;
    const auto ratios = deepseek_v4_flash_dspark_spec().deepseek_v4.compression_ratios;
    for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
        auto& state = attention_state[layer];
        state.sliding.assign(static_cast<std::size_t>(kWindow) * kHeadDim, 0.0F);
        state.frequencies = rope_frequencies(ratios[layer]);
        auto& compressor_state = state.compressor;
        compressor_state = {};
        compressor_state.ratio = ratios[layer];
        if (compressor_state.ratio == 0U) continue;
        compressor_state.coefficient = compressor_state.ratio == 4U ? 2U : 1U;
        const auto rows = static_cast<std::size_t>(compressor_state.coefficient) *
                          compressor_state.ratio;
        const auto dimensions = static_cast<std::size_t>(
            compressor_state.coefficient) * kHeadDim;
        compressor_state.values.assign(rows * dimensions, 0.0F);
        compressor_state.scores.assign(
            rows * dimensions, -std::numeric_limits<float>::infinity());
        const auto compressed_rows =
            (static_cast<std::size_t>(config.maximum_context_tokens) +
             compressor_state.ratio - 1U) / compressor_state.ratio;
        compressor_state.compressed.assign(compressed_rows * kHeadDim, 0.0F);
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::embed(std::uint32_t token,
                                                 std::span<float> output) {
    ValidationResult result;
    if (token >= kVocabulary || output.size() != static_cast<std::size_t>(kMhc) * kHidden) {
        result.errors.emplace_back("DeepSeek embedding token or output shape is invalid");
        return result;
    }
    const auto embedding = resident.find("embed.weight");
    const auto row_bytes = static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t);
    const auto offset = static_cast<std::size_t>(token) * row_bytes;
    if (embedding.size() < offset + row_bytes) {
        result.errors.emplace_back("DeepSeek resident embedding extent is incomplete");
        return result;
    }
    for (std::uint32_t column = 0U; column < kHidden; ++column) {
        std::uint16_t encoded = 0U;
        std::memcpy(&encoded, embedding.data() + offset + column * sizeof(encoded),
                    sizeof(encoded));
        const float value = std::bit_cast<float>(
            static_cast<std::uint32_t>(encoded) << 16U);
        for (std::uint32_t copy = 0U; copy < kMhc; ++copy) {
            output[static_cast<std::size_t>(copy) * kHidden + column] = value;
        }
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::compressor(
    std::uint32_t layer, std::span<const float> input, std::uint32_t position) {
    ValidationResult result;
    auto& state = attention_state[layer].compressor;
    if (state.ratio == 0U) return result;
    const auto prefix = layer_prefix(layer) + "attn.compressor.";
    const auto dimensions = static_cast<std::size_t>(state.coefficient) * kHeadDim;
    std::vector<float> values(dimensions);
    std::vector<float> scores(dimensions);
    const auto slot = layer_device(layer);
    result = linear(slot, prefix + "wkv", dimensions, kHidden, input, values, false);
    if (!result.ok()) return result;
    result = linear(slot, prefix + "wgate", dimensions, kHidden, input, scores, false);
    if (!result.ok()) return result;
    auto ape = host_tensor(prefix + "ape",
                           static_cast<std::uint64_t>(state.ratio) * dimensions);
    if (!ape.ok()) {
        append_errors(result, std::move(ape.errors));
        return result;
    }
    const auto phase = position % state.ratio;
    const auto row = state.coefficient == 2U ? state.ratio + phase : phase;
    const auto row_offset = static_cast<std::size_t>(row) * dimensions;
    const auto ape_offset = static_cast<std::size_t>(phase) * dimensions;
    for (std::size_t dimension = 0U; dimension < dimensions; ++dimension) {
        state.values[row_offset + dimension] = values[dimension];
        state.scores[row_offset + dimension] =
            scores[dimension] + (*ape.value)[ape_offset + dimension];
    }
    if ((position + 1U) % state.ratio != 0U) return result;

    std::vector<float> pooled(kHeadDim, 0.0F);
    for (std::uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::uint32_t candidate = 0U;
             candidate < state.coefficient * state.ratio; ++candidate) {
            std::size_t index = 0U;
            if (state.coefficient == 2U) {
                const auto source_row = candidate < state.ratio ? candidate : candidate;
                const auto source_dimension = candidate < state.ratio ? dimension :
                    static_cast<std::uint32_t>(kHeadDim + dimension);
                index = static_cast<std::size_t>(source_row) * dimensions +
                        source_dimension;
            } else {
                index = static_cast<std::size_t>(candidate) * dimensions + dimension;
            }
            maximum = std::max(maximum, state.scores[index]);
        }
        double denominator = 0.0;
        double numerator = 0.0;
        for (std::uint32_t candidate = 0U;
             candidate < state.coefficient * state.ratio; ++candidate) {
            std::size_t index = 0U;
            if (state.coefficient == 2U) {
                const auto source_dimension = candidate < state.ratio ? dimension :
                    static_cast<std::uint32_t>(kHeadDim + dimension);
                index = static_cast<std::size_t>(candidate) * dimensions +
                        source_dimension;
            } else {
                index = static_cast<std::size_t>(candidate) * dimensions + dimension;
            }
            const double weight = std::exp(
                static_cast<double>(state.scores[index] - maximum));
            denominator += weight;
            numerator += weight * static_cast<double>(state.values[index]);
        }
        pooled[dimension] = static_cast<float>(numerator / denominator);
    }
    if (state.coefficient == 2U) {
        const auto block_bytes = static_cast<std::size_t>(state.ratio) * dimensions;
        std::copy_n(state.values.begin() + block_bytes, block_bytes,
                    state.values.begin());
        std::copy_n(state.scores.begin() + block_bytes, block_bytes,
                    state.scores.begin());
    }
    result = norm(pooled, pooled, prefix + "norm.weight");
    if (!result.ok()) return result;
    apply_rope(std::span<float>(pooled).last(kRopeDim),
               position + 1U - state.ratio, attention_state[layer].frequencies);
    round_bf16(pooled);
    quantize_activation_in_place(std::span<float>(pooled).first(kHeadDim - kRopeDim),
                                 64U);
    const auto compressed_row = position / state.ratio;
    std::copy(pooled.begin(), pooled.end(),
              state.compressed.begin() +
                  static_cast<std::size_t>(compressed_row) * kHeadDim);
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::attention(
    std::uint32_t layer, std::span<const float> input, std::uint32_t position,
    std::span<float> output) {
    ValidationResult result;
    if (input.size() != kHidden || output.size() != kHidden) {
        result.errors.emplace_back("DeepSeek attention spans have incompatible sizes");
        return result;
    }
    const auto slot = layer_device(layer);
    const auto prefix = layer_prefix(layer) + "attn.";
    std::vector<float> query_rank(kQueryRank);
    result = linear(slot, prefix + "wq_a", kQueryRank, kHidden, input, query_rank);
    if (!result.ok()) return result;
    result = norm(query_rank, query_rank, prefix + "q_norm.weight");
    if (!result.ok()) return result;
    std::vector<float> queries(static_cast<std::size_t>(kHeads) * kHeadDim);
    result = linear(slot, prefix + "wq_b", kHeads * kHeadDim, kQueryRank,
                    query_rank, queries);
    if (!result.ok()) return result;
    const auto normalize_query = [&](std::uint32_t head) {
        auto query = std::span<float>(queries).subspan(
            static_cast<std::size_t>(head) * kHeadDim, kHeadDim);
        double square_sum = 0.0;
        for (const float value : query) {
            square_sum += static_cast<double>(value) * value;
        }
        const float reciprocal = 1.0F / std::sqrt(
            static_cast<float>(square_sum / kHeadDim) + kRmsEpsilon);
        for (auto& value : query) value = round_bf16(value * reciprocal);
        apply_rope(query.last(kRopeDim), position,
                   attention_state[layer].frequencies);
        round_bf16(query.last(kRopeDim));
    };
    if (attention_workers != nullptr) {
        result = attention_workers->parallel_for(
            kHeads, [&](std::size_t head) {
                normalize_query(static_cast<std::uint32_t>(head));
            });
        if (!result.ok()) return result;
    } else {
        for (std::uint32_t head = 0U; head < kHeads; ++head) {
            normalize_query(head);
        }
    }

    std::vector<float> kv(kHeadDim);
    result = linear(slot, prefix + "wkv", kHeadDim, kHidden, input, kv);
    if (!result.ok()) return result;
    result = norm(kv, kv, prefix + "kv_norm.weight");
    if (!result.ok()) return result;
    apply_rope(std::span<float>(kv).last(kRopeDim), position,
               attention_state[layer].frequencies);
    round_bf16(std::span<float>(kv).last(kRopeDim));
    quantize_activation_in_place(std::span<float>(kv).first(kHeadDim - kRopeDim),
                                 64U);
    auto& layer_state = attention_state[layer];
    std::copy(kv.begin(), kv.end(),
              layer_state.sliding.begin() +
                  static_cast<std::size_t>(position % kWindow) * kHeadDim);
    result = compressor(layer, input, position);
    if (!result.ok()) return result;

    auto sink = host_tensor(prefix + "attn_sink", kHeads);
    if (!sink.ok()) {
        append_errors(result, std::move(sink.errors));
        return result;
    }
    std::vector<float> attended(static_cast<std::size_t>(kHeads) * kHeadDim, 0.0F);
    const auto window_count = std::min<std::uint32_t>(position + 1U, kWindow);
    const auto ratio = layer_state.compressor.ratio;
    const auto compressed_count = ratio == 0U ? 0U : (position + 1U) / ratio;
    const auto score_stride = static_cast<std::size_t>(window_count) +
                              compressed_count;
    const auto attend_head = [&](std::uint32_t head,
                                 std::span<float> scores) {
        const auto query = std::span<const float>(queries).subspan(
            static_cast<std::size_t>(head) * kHeadDim, kHeadDim);
        std::size_t next_score = 0U;
        float maximum = (*sink.value)[head];
        for (std::uint32_t item = 0U; item < window_count; ++item) {
            const auto absolute = position + 1U - window_count + item;
            const auto key = std::span<const float>(layer_state.sliding).subspan(
                static_cast<std::size_t>(absolute % kWindow) * kHeadDim, kHeadDim);
            double dot = 0.0;
            for (std::uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
                dot += static_cast<double>(query[dimension]) * key[dimension];
            }
            const float score = static_cast<float>(dot) * kAttentionScale;
            scores[next_score++] = score;
            maximum = std::max(maximum, score);
        }
        for (std::uint32_t item = 0U; item < compressed_count; ++item) {
            const auto key = std::span<const float>(layer_state.compressor.compressed)
                                 .subspan(static_cast<std::size_t>(item) * kHeadDim,
                                          kHeadDim);
            double dot = 0.0;
            for (std::uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
                dot += static_cast<double>(query[dimension]) * key[dimension];
            }
            const float score = static_cast<float>(dot) * kAttentionScale;
            scores[next_score++] = score;
            maximum = std::max(maximum, score);
        }
        double denominator = std::exp(
            static_cast<double>((*sink.value)[head] - maximum));
        for (const float score : scores) {
            denominator += std::exp(static_cast<double>(score - maximum));
        }
        auto destination = std::span<float>(attended).subspan(
            static_cast<std::size_t>(head) * kHeadDim, kHeadDim);
        std::size_t score_index = 0U;
        for (std::uint32_t item = 0U; item < window_count; ++item) {
            const auto absolute = position + 1U - window_count + item;
            const auto value = std::span<const float>(layer_state.sliding).subspan(
                static_cast<std::size_t>(absolute % kWindow) * kHeadDim, kHeadDim);
            const float probability = static_cast<float>(
                std::exp(static_cast<double>(scores[score_index++] - maximum)) /
                denominator);
            for (std::uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
                destination[dimension] += probability * value[dimension];
            }
        }
        for (std::uint32_t item = 0U; item < compressed_count; ++item) {
            const auto value = std::span<const float>(layer_state.compressor.compressed)
                                   .subspan(static_cast<std::size_t>(item) * kHeadDim,
                                            kHeadDim);
            const float probability = static_cast<float>(
                std::exp(static_cast<double>(scores[score_index++] - maximum)) /
                denominator);
            for (std::uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
                destination[dimension] += probability * value[dimension];
            }
        }
        round_bf16(destination);
        apply_rope(destination.last(kRopeDim), position,
                   layer_state.frequencies, true);
        round_bf16(destination.last(kRopeDim));
    };
    if (attention_workers != nullptr) {
        std::vector<float> parallel_scores(
            static_cast<std::size_t>(kHeads) * score_stride);
        result = attention_workers->parallel_for(
            kHeads, [&](std::size_t head) {
                attend_head(
                    static_cast<std::uint32_t>(head),
                    std::span<float>(parallel_scores).subspan(
                        head * score_stride, score_stride));
            });
        if (!result.ok()) return result;
    } else {
        std::vector<float> scores(score_stride);
        for (std::uint32_t head = 0U; head < kHeads; ++head) {
            attend_head(head, scores);
        }
    }

    std::vector<float> output_rank(static_cast<std::size_t>(kOutputGroups) *
                                   kOutputRank);
    result = weights->grouped(slot, prefix + "wo_a", kOutputGroups * kOutputRank,
                              kHeads * kHeadDim / kOutputGroups, attended,
                              kOutputGroups, kOutputRank, output_rank);
    if (!result.ok()) return result;
    return linear(slot, prefix + "wo_b", kHidden,
                  kOutputGroups * kOutputRank, output_rank, output);
}

ValidationResult DeepSeekV4Runtime::Impl::expert(
    std::uint32_t layer, std::uint32_t expert_id,
    float routed_coefficient,
    std::span<const float> input, std::span<float> output) {
    ValidationResult result;
    if (expert_id >= kExperts || input.size() != kHidden || output.size() != kHidden) {
        result.errors.emplace_back("DeepSeek expert id or span shape is invalid");
        return result;
    }
    const auto slot = expert_device(expert_id);
    const auto prefix = layer_prefix(layer) + "ffn.experts." +
                        std::to_string(expert_id) + ".";
    std::vector<float> gate(kExpertIntermediate);
    std::vector<float> up(kExpertIntermediate);
    std::vector<float> activated(kExpertIntermediate);
    result = linear(slot, prefix + "w1", kExpertIntermediate, kHidden, input, gate);
    if (!result.ok()) return result;
    result = linear(slot, prefix + "w3", kExpertIntermediate, kHidden, input, up);
    if (!result.ok()) return result;
    result = dsv4_swiglu_f32(activated, gate, up, 10.0F);
    if (!result.ok()) return result;
    for (auto& value : activated) {
        value *= routed_coefficient;
        value *= routed_coefficient;
    }
    round_bf16(activated);
    return linear(slot, prefix + "w2", kHidden, kExpertIntermediate,
                  activated, output);
}

ValidationResult DeepSeekV4Runtime::Impl::device_moe(
    std::uint32_t layer, const Dsv4Route& route,
    std::span<const float> input, std::span<float> output) {
    ValidationResult result;
    if (layer >= kLayers || input.size() != kHidden || output.size() != kHidden ||
        route.experts.size() != kTopK || route.weights.size() != kTopK) {
        result.errors.emplace_back("DeepSeek device MoE input or route shape is invalid");
        return result;
    }

    struct RoutePlacement {
        std::size_t slot{};
        std::size_t local_rank{};
    };
    struct PendingDevice {
        std::vector<Dsv4WeightCache::Lease> leases;
        std::vector<CudaDeepSeekMoeExpert> routed;
        CudaDeepSeekMoeExpert shared;
        std::vector<float> routed_output;
        std::vector<float> shared_output;
        bool has_shared{};
        bool enqueued{};
    };

    std::vector<PendingDevice> pending(devices.size());
    for (auto& device : pending) {
        device.leases.reserve((kTopK + 1U) * 3U);
        device.routed.reserve(kTopK);
    }
    std::array<RoutePlacement, kTopK> placements{};

    const auto acquire_triplet = [this, &result](
        std::size_t slot, std::string_view prefix, float coefficient,
        PendingDevice& pending_device, CudaDeepSeekMoeExpert& descriptor) {
        descriptor.coefficient = coefficient;
        const auto acquire = [this, &result, slot, &pending_device](
            std::string name, std::uint64_t rows, std::uint64_t columns,
            const CudaWeight*& weight) {
            pending_device.leases.emplace_back();
            auto loaded = weights->acquire(slot, name, rows, columns,
                                           pending_device.leases.back());
            if (!loaded.ok()) {
                append_errors(result, std::move(loaded.errors), name);
                pending_device.leases.pop_back();
                return false;
            }
            weight = &pending_device.leases.back().weight();
            return true;
        };
        return acquire(std::string(prefix) + "w1", kExpertIntermediate,
                       kHidden, descriptor.w1) &&
               acquire(std::string(prefix) + "w3", kExpertIntermediate,
                       kHidden, descriptor.w3) &&
               acquire(std::string(prefix) + "w2", kHidden,
                       kExpertIntermediate, descriptor.w2);
    };

    const auto routed_prefix = layer_prefix(layer) + "ffn.experts.";
    for (std::size_t rank = 0U; rank < kTopK; ++rank) {
        const auto expert_id = route.experts[rank];
        if (expert_id >= kExperts || !std::isfinite(route.weights[rank])) {
            result.errors.emplace_back(
                "DeepSeek device MoE expert id or coefficient is invalid");
            return result;
        }
        const auto slot = expert_device(expert_id);
        auto& pending_device = pending[slot];
        placements[rank] = {slot, pending_device.routed.size()};
        CudaDeepSeekMoeExpert descriptor;
        const auto prefix = routed_prefix + std::to_string(expert_id) + ".";
        if (!acquire_triplet(slot, prefix, route.weights[rank],
                             pending_device, descriptor)) {
            return result;
        }
        pending_device.routed.push_back(descriptor);
    }

    const auto shared_slot = layer_device(layer);
    auto& shared_device = pending[shared_slot];
    const auto shared_prefix = layer_prefix(layer) + "ffn.shared_experts.";
    if (!acquire_triplet(shared_slot, shared_prefix, 1.0F,
                         shared_device, shared_device.shared)) {
        return result;
    }
    shared_device.has_shared = true;

    const auto execution_started = std::chrono::steady_clock::now();
    const auto device_commands = static_cast<std::uint64_t>(std::count_if(
        pending.begin(), pending.end(), [](const auto& pending_device) {
            return !pending_device.routed.empty() || pending_device.has_shared;
        }));

    for (std::size_t slot = 0U; slot < pending.size(); ++slot) {
        auto& pending_device = pending[slot];
        if (pending_device.routed.empty() && !pending_device.has_shared) continue;
        pending_device.routed_output.resize(
            pending_device.routed.size() * kHidden);
        if (pending_device.has_shared) {
            pending_device.shared_output.resize(kHidden);
        }
        auto enqueued = cuda.enqueue_deepseek_moe(
            devices[slot], input, pending_device.routed,
            pending_device.has_shared ? &pending_device.shared : nullptr, 10.0F);
        if (!enqueued.ok()) {
            append_errors(result, std::move(enqueued.errors),
                          "DeepSeek device MoE enqueue");
            break;
        }
        pending_device.enqueued = true;
    }

    // Every accepted command must be observed before its cache leases leave
    // scope, including commands submitted before a later-device enqueue error.
    for (std::size_t slot = 0U; slot < pending.size(); ++slot) {
        auto& pending_device = pending[slot];
        if (!pending_device.enqueued) continue;
        auto collected = cuda.collect_deepseek_moe(
            devices[slot], pending_device.routed_output,
            pending_device.shared_output);
        pending_device.enqueued = false;
        if (!collected.ok()) {
            append_errors(result, std::move(collected.errors),
                          "DeepSeek device MoE collect");
        }
    }
    if (!result.ok()) return result;

    for (auto& pending_device : pending) {
        round_bf16(pending_device.routed_output);
        round_bf16(pending_device.shared_output);
    }
    std::fill(output.begin(), output.end(), 0.0F);
    for (std::size_t rank = 0U; rank < kTopK; ++rank) {
        const auto placement = placements[rank];
        const auto routed = std::span<const float>(
            pending[placement.slot].routed_output)
            .subspan(placement.local_rank * kHidden, kHidden);
        for (std::uint32_t column = 0U; column < kHidden; ++column) {
            output[column] += routed[column];
        }
    }
    for (std::uint32_t column = 0U; column < kHidden; ++column) {
        output[column] = round_bf16(
            output[column] + shared_device.shared_output[column]);
    }
    ++device_moe_stats.batches;
    device_moe_stats.device_commands += device_commands;
    device_moe_stats.routed_experts += kTopK;
    ++device_moe_stats.shared_experts;
    device_moe_stats.nanoseconds += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - execution_started).count());
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::moe(
    std::uint32_t layer, std::uint32_t token, std::span<const float> input,
    std::span<float> output, std::uint32_t position) {
    ValidationResult result;
    const auto prefix = layer_prefix(layer) + "ffn.";
    std::vector<float> logits(kExperts);
    result = linear(layer_device(layer), prefix + "gate", kExperts, kHidden,
                    input, logits, false);
    if (!result.ok()) return result;
    const auto& router = deepseek_v4_flash_dspark_spec().router;
    Dsv4RouteResult route;
    if (layer < 3U) {
        const auto name = prefix + "gate.tid2eid";
        const auto found = host_raw.find(name);
        if (found == host_raw.end()) {
            result.errors.emplace_back("DeepSeek resident hash-routing table is absent");
            return result;
        }
        const auto row_bytes = static_cast<std::size_t>(kTopK) * sizeof(std::int64_t);
        const auto offset = static_cast<std::size_t>(token) * row_bytes;
        if (token >= kVocabulary || found->second.size() < offset + row_bytes) {
            result.errors.emplace_back("DeepSeek hash-routing token row is out of range");
            return result;
        }
        std::array<std::uint32_t, kTopK> selected{};
        for (std::uint32_t rank = 0U; rank < kTopK; ++rank) {
            std::int64_t encoded = 0;
            std::memcpy(&encoded,
                        found->second.data() + offset + rank * sizeof(encoded),
                        sizeof(encoded));
            if (encoded < 0 || encoded >= static_cast<std::int64_t>(kExperts)) {
                result.errors.emplace_back("DeepSeek hash-routing expert is invalid");
                return result;
            }
            selected[rank] = static_cast<std::uint32_t>(encoded);
        }
        route = dsv4_route_hash_sqrtsoftplus_f32(logits, selected, router);
    } else {
        auto bias = host_tensor(prefix + "gate.bias", kExperts);
        if (!bias.ok()) {
            append_errors(result, std::move(bias.errors));
            return result;
        }
        route = dsv4_route_sqrtsoftplus_f32(logits, *bias.value, router);
    }
    if (!route.ok()) {
        append_errors(result, std::move(route.errors));
        return result;
    }
    if (route_trace.is_open()) {
        route_trace << "{\"position\":" << position << ",\"layer\":" << layer
                    << ",\"token\":" << token << ",\"experts\":[";
        for (std::size_t index = 0U; index < route.value.experts.size(); ++index) {
            if (index != 0U) route_trace << ',';
            route_trace << route.value.experts[index];
        }
        route_trace << "],\"weights\":[" << std::setprecision(
            std::numeric_limits<float>::max_digits10);
        for (std::size_t index = 0U; index < route.value.weights.size(); ++index) {
            if (index != 0U) route_trace << ',';
            route_trace << route.value.weights[index];
        }
        route_trace << "]}\n";
    }

    if (config.enable_device_moe) {
        return device_moe(layer, route.value, input, output);
    }

    std::fill(output.begin(), output.end(), 0.0F);
    std::vector<float> routed(kHidden);
    for (std::size_t rank = 0U; rank < route.value.experts.size(); ++rank) {
        result = expert(layer, route.value.experts[rank],
                        route.value.weights[rank], input, routed);
        if (!result.ok()) return result;
        for (std::uint32_t column = 0U; column < kHidden; ++column) {
            output[column] += routed[column];
        }
    }

    const auto slot = layer_device(layer);
    std::vector<float> shared_gate(kExpertIntermediate);
    std::vector<float> shared_up(kExpertIntermediate);
    std::vector<float> shared_activated(kExpertIntermediate);
    std::vector<float> shared_output(kHidden);
    result = linear(slot, prefix + "shared_experts.w1", kExpertIntermediate,
                    kHidden, input, shared_gate);
    if (!result.ok()) return result;
    result = linear(slot, prefix + "shared_experts.w3", kExpertIntermediate,
                    kHidden, input, shared_up);
    if (!result.ok()) return result;
    result = dsv4_swiglu_f32(shared_activated, shared_gate, shared_up, 10.0F);
    if (!result.ok()) return result;
    round_bf16(shared_activated);
    result = linear(slot, prefix + "shared_experts.w2", kHidden,
                    kExpertIntermediate, shared_activated, shared_output);
    if (!result.ok()) return result;
    for (std::uint32_t column = 0U; column < kHidden; ++column) {
        output[column] = round_bf16(output[column] + shared_output[column]);
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::block(
    std::uint32_t layer, std::uint32_t token, std::span<float> hidden,
    std::uint32_t position) {
    ValidationResult result;
    if (hidden.size() != static_cast<std::size_t>(kMhc) * kHidden) {
        result.errors.emplace_back("DeepSeek mHC hidden state has the wrong shape");
        return result;
    }
    const auto prefix = layer_prefix(layer);
    for (const auto* branch_name : {"attn", "ffn"}) {
        const std::string branch(branch_name);
        auto projection = host_tensor(prefix + "hc_" + branch + "_fn",
                                      kMix * kMhc * kHidden);
        auto scale = host_tensor(prefix + "hc_" + branch + "_scale", 3U);
        auto base = host_tensor(prefix + "hc_" + branch + "_base", kMix);
        if (!projection.ok()) append_errors(result, std::move(projection.errors));
        if (!scale.ok()) append_errors(result, std::move(scale.errors));
        if (!base.ok()) append_errors(result, std::move(base.errors));
        if (!result.ok()) return result;
        const std::vector<float> residual(hidden.begin(), hidden.end());
        std::vector<float> reduced(kHidden);
        Dsv4MhcMix mix;
        result = dsv4_mhc_pre_f32(reduced, mix, residual, *projection.value,
                                  *scale.value, *base.value);
        if (!result.ok()) return result;
        round_bf16(reduced);
        result = norm(reduced, reduced, prefix + branch + "_norm.weight");
        if (!result.ok()) return result;
        std::vector<float> branch_output(kHidden);
        if (branch == "attn") {
            result = attention(layer, reduced, position, branch_output);
        } else {
            result = moe(layer, token, reduced, branch_output, position);
        }
        if (!result.ok()) return result;
        result = dsv4_mhc_post_f32(hidden, branch_output, residual, mix);
        if (!result.ok()) return result;
        round_bf16(hidden);
    }
    return result;
}

ParseResult<std::uint32_t> DeepSeekV4Runtime::Impl::forward_token(
    std::uint32_t token, std::uint32_t position, bool logits_required) {
    ParseResult<std::uint32_t> result;
    std::vector<float> hidden(static_cast<std::size_t>(kMhc) * kHidden);
    auto validation = embed(token, hidden);
    if (!validation.ok()) {
        result.errors = std::move(validation.errors);
        return result;
    }
    for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
        validation = block(layer, token, hidden, position);
        if (!validation.ok()) {
            result.errors = std::move(validation.errors);
            return result;
        }
        if (config.enable_layer_hash_trace) {
            record_layer_hash(position, token, layer, hidden);
        }
    }
    if (!logits_required) {
        result.value = token;
        return result;
    }

    auto head_projection = host_tensor("hc_head_fn", kMhc * kMhc * kHidden);
    auto head_scale = host_tensor("hc_head_scale", 1U);
    auto head_base = host_tensor("hc_head_base", kMhc);
    if (!head_projection.ok()) append_errors(validation, std::move(head_projection.errors));
    if (!head_scale.ok()) append_errors(validation, std::move(head_scale.errors));
    if (!head_base.ok()) append_errors(validation, std::move(head_base.errors));
    if (!validation.ok()) {
        result.errors = std::move(validation.errors);
        return result;
    }
    double square_sum = 0.0;
    for (const float value : hidden) square_sum += static_cast<double>(value) * value;
    const float reciprocal = 1.0F / std::sqrt(
        static_cast<float>(square_sum / static_cast<double>(hidden.size())) +
        kRmsEpsilon);
    std::vector<float> reduced(kHidden, 0.0F);
    for (std::uint32_t copy = 0U; copy < kMhc; ++copy) {
        double projected = 0.0;
        const auto row = static_cast<std::size_t>(copy) * hidden.size();
        for (std::size_t column = 0U; column < hidden.size(); ++column) {
            projected += static_cast<double>((*head_projection.value)[row + column]) *
                         hidden[column];
        }
        const float coefficient = glm_sigmoid_f32(
            static_cast<float>(projected) * reciprocal * (*head_scale.value)[0] +
            (*head_base.value)[copy]) + kRmsEpsilon;
        for (std::uint32_t column = 0U; column < kHidden; ++column) {
            reduced[column] += coefficient *
                hidden[static_cast<std::size_t>(copy) * kHidden + column];
        }
    }
    round_bf16(reduced);
    validation = norm(reduced, reduced, "norm.weight");
    if (!validation.ok()) {
        result.errors = std::move(validation.errors);
        return result;
    }
    std::vector<float> logits(kVocabulary);
    validation = linear(layer_device(kLayers - 1U), "head", kVocabulary,
                        kHidden, reduced, logits, false);
    if (!validation.ok()) {
        result.errors = std::move(validation.errors);
        return result;
    }
    result.value = static_cast<std::uint32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    if (config.enable_logit_trace) {
        record_logits(position, token, result.value, logits);
    }
    return result;
}

DeepSeekV4Runtime::DeepSeekV4Runtime() : impl_(std::make_unique<Impl>()) {}
DeepSeekV4Runtime::~DeepSeekV4Runtime() = default;
DeepSeekV4Runtime::DeepSeekV4Runtime(DeepSeekV4Runtime&&) noexcept = default;
DeepSeekV4Runtime& DeepSeekV4Runtime::operator=(DeepSeekV4Runtime&&) noexcept = default;

ValidationResult DeepSeekV4Runtime::initialize(
    const std::string& model_directory, const Dsv4RuntimeConfig& config) {
    ValidationResult result;
    const auto initialization_started = std::chrono::steady_clock::now();
    if (config.devices.empty()) {
        result.errors.emplace_back("DeepSeek runtime requires at least one CUDA device");
        return result;
    }
    if (!std::isfinite(config.vram_cache_fraction) ||
        config.vram_cache_fraction <= 0.0 || config.vram_cache_fraction > 0.95) {
        result.errors.emplace_back("VRAM cache fraction must be in (0, 0.95]");
        return result;
    }
    if (config.maximum_context_tokens == 0U ||
        config.maximum_context_tokens > 2048U) {
        result.errors.emplace_back(
            "current exact DeepSeek runtime context must be within [1, 2048] tokens");
        return result;
    }
    if (config.enable_logit_trace &&
        (config.logit_trace_top_k == 0U ||
         config.logit_trace_top_k > kVocabulary)) {
        result.errors.emplace_back(
            "DeepSeek logit trace top-K must be within [1, 129280]");
        return result;
    }
    if (config.host_attention_threads > kHeads) {
        result.errors.emplace_back(
            "DeepSeek host attention worker count must not exceed 64");
        return result;
    }
    if (config.enable_dspark) {
        result.errors.emplace_back(
            "DSpark tensors are verified, but speculative execution is not enabled in "
            "the base-model executor; refusing a silent approximation");
        return result;
    }
    auto checkpoint = Dsv4CheckpointReader::open(model_directory);
    if (!checkpoint.ok()) {
        result.errors = std::move(checkpoint.errors);
        return result;
    }
    auto tokenizer = GlmTokenizer::load(
        (std::filesystem::path(model_directory) / "tokenizer.json").string());
    if (!tokenizer.ok()) {
        result.errors = std::move(tokenizer.errors);
        return result;
    }
    result = impl_->cuda.initialize(config.devices, config.detailed_timing);
    if (!result.ok()) return result;

    std::vector<std::uint64_t> capacities;
    std::vector<std::uint64_t> weight_capacities;
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
            result.errors.emplace_back(
                "CUDA device has less than 2 GiB available for DeepSeek weights");
            return result;
        }
        capacities.push_back(capacity);
        weight_capacities.push_back(capacity - kDeviceWorkspaceReserve);
        totals.push_back(memory.value.total_bytes);
    }
    const auto admission_started = std::chrono::steady_clock::now();
    Dsv4AdmissionConfig admission_config;
    admission_config.host_memory_ceiling_bytes = config.host_memory_limit_bytes;
    admission_config.vram_weight_budgets = capacities;
    admission_config.maximum_context_tokens = config.maximum_context_tokens;
    admission_config.enable_dspark = config.enable_dspark;
    admission_config.require_zero_nvme_decode = config.require_zero_nvme_decode;
    auto admission = plan_dsv4_resident_topology(checkpoint.value->manifest(),
                                                  admission_config);
    const double admission_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - admission_started).count();
    if (!admission.ok()) {
        result.errors = std::move(admission.errors);
        return result;
    }

    const auto smallest = *std::min_element(totals.begin(), totals.end());
    std::vector<std::size_t> schedule;
    for (std::size_t slot = 0U; slot < totals.size(); ++slot) {
        const auto shares = std::max<std::uint64_t>(
            1U, (totals[slot] * 2U + smallest / 2U) / smallest);
        for (std::uint64_t count = 0U; count < shares; ++count) {
            schedule.push_back(slot);
        }
    }
    impl_->config = config;
    impl_->memory = admission.plan;
    impl_->checkpoint = std::move(checkpoint.value);
    impl_->tokenizer = std::move(tokenizer.value);
    impl_->devices = config.devices;
    impl_->capacities = weight_capacities;
    impl_->schedule = std::move(schedule);
    if (!config.route_trace_path.empty()) {
        impl_->route_trace.open(config.route_trace_path,
                                std::ios::out | std::ios::trunc);
        if (!impl_->route_trace.is_open()) {
            result.errors.emplace_back("cannot open DeepSeek route trace: " +
                                       config.route_trace_path);
            return result;
        }
        impl_->route_trace
            << "{\"schema\":\"strata.deepseek_v4.route_trace\",\"version\":1,"
               "\"position_base\":0,\"layer_base\":0,\"router_order\":true,"
               "\"coefficient_encoding\":\"float32-roundtrip-decimal\"}\n";
    }
    if (config.verbose) {
        for (std::size_t slot = 0U; slot < impl_->devices.size(); ++slot) {
            std::cerr << "[hardware] cuda=" << impl_->devices[slot]
                      << " vram_budget_bytes=" << capacities[slot]
                      << " weight_cache_bytes=" << weight_capacities[slot]
                      << " workspace_reserve_bytes=" << kDeviceWorkspaceReserve
                      << '\n';
        }
    }
    const auto staging_started = std::chrono::steady_clock::now();
    result = impl_->resident.stage(*impl_->checkpoint,
                                   config.host_memory_limit_bytes,
                                   config.enable_dspark);
    const double staging_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - staging_started).count();
    if (!result.ok()) return result;
    impl_->weights = std::make_unique<Dsv4WeightCache>(
        *impl_->checkpoint, impl_->resident, impl_->cuda,
        impl_->devices, weight_capacities);
    if (config.host_attention_threads != 0U) {
        impl_->attention_workers = std::make_unique<HostWorkerPool>(
            config.host_attention_threads);
    } else {
        impl_->attention_workers.reset();
    }
    result = impl_->warmup();
    if (!result.ok()) return result;
    if (config.enable_device_moe) {
        if (impl_->memory.maximum_expert_bytes >
            std::numeric_limits<std::uint64_t>::max() / kTopK) {
            result.errors.emplace_back(
                "DeepSeek exact top-k expert lease size overflows");
            return result;
        }
        result = impl_->weights->validate_atomic_expert_capacity(
            impl_->memory.maximum_expert_bytes * kTopK);
        if (!result.ok()) return result;
    }
    result = impl_->reset_sequence();
    if (!result.ok()) return result;
    impl_->initialized = true;
    impl_->initialization_metrics.initialization_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      initialization_started).count();
    impl_->initialization_metrics.admission_seconds = admission_seconds;
    impl_->initialization_metrics.resident_staging_seconds = staging_seconds;
    impl_->initialization_metrics.memory = impl_->memory;
    impl_->initialization_metrics.resident_stage = impl_->resident.stats();
    impl_->initialization_metrics.cuda = impl_->cuda.stats();
    impl_->initialization_metrics.cache = impl_->weights->stats();
    impl_->initialization_metrics.detailed_timing = config.detailed_timing;
    impl_->initialization_metrics.dspark_enabled = false;
    impl_->initialization_metrics.device_moe_enabled = config.enable_device_moe;
    impl_->initialization_metrics.host_attention_threads =
        config.host_attention_threads;
    return result;
}

Dsv4GenerationResult DeepSeekV4Runtime::generate(
    std::string_view prompt, std::uint32_t maximum_new_tokens) {
    Dsv4GenerationResult result;
    if (!impl_->initialized) {
        result.errors.emplace_back("DeepSeek runtime is not initialized");
        return result;
    }
    if (maximum_new_tokens == 0U) {
        result.errors.emplace_back("maximum_new_tokens must be positive");
        return result;
    }
    auto encoded = impl_->tokenizer.encode(
        render_deepseek_v4_user_prompt(prompt, false));
    if (!encoded.ok()) {
        result.errors = std::move(encoded.errors);
        return result;
    }
    if (encoded.value.size() + maximum_new_tokens >
        impl_->config.maximum_context_tokens) {
        result.errors.emplace_back(
            "prompt and requested DeepSeek generation exceed the context ceiling");
        return result;
    }
    result.prompt_token_ids = encoded.value;
    impl_->reset_diagnostics();
    auto reset = impl_->reset_sequence();
    if (!reset.ok()) {
        result.errors = std::move(reset.errors);
        return result;
    }
    const auto reads_before = impl_->checkpoint->stats();
    const auto cuda_before = impl_->cuda.stats();
    const auto cache_before = impl_->weights->stats();
    const auto device_moe_before = impl_->device_moe_stats;
    const auto prefill_started = std::chrono::steady_clock::now();
    ParseResult<std::uint32_t> next;
    for (std::size_t position = 0U; position < result.prompt_token_ids.size(); ++position) {
        next = impl_->forward_token(
            result.prompt_token_ids[position], static_cast<std::uint32_t>(position),
            position + 1U == result.prompt_token_ids.size());
        if (!next.ok()) {
            result.errors = std::move(next.errors);
            return result;
        }
    }
    result.metrics.prefill_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - prefill_started).count();
    result.metrics.prompt_tokens = result.prompt_token_ids.size();
    const auto reads_after_prefill = impl_->checkpoint->stats();
    const auto cuda_after_prefill = impl_->cuda.stats();
    const auto cache_after_prefill = impl_->weights->stats();
    const auto device_moe_after_prefill = impl_->device_moe_stats;
    constexpr std::uint32_t stop_token = 1U;
    if (next.value != stop_token) result.generated_token_ids.push_back(next.value);
    std::uint32_t position = static_cast<std::uint32_t>(
        result.prompt_token_ids.size());
    std::uint64_t decode_steps = 0U;
    const auto decode_started = std::chrono::steady_clock::now();
    while (next.value != stop_token &&
           result.generated_token_ids.size() < maximum_new_tokens) {
        next = impl_->forward_token(next.value, position++, true);
        if (!next.ok()) {
            result.errors = std::move(next.errors);
            return result;
        }
        ++decode_steps;
        if (next.value != stop_token) result.generated_token_ids.push_back(next.value);
    }
    result.metrics.decode_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - decode_started).count();
    result.metrics.decode_tokens = decode_steps;
    auto decoded = impl_->tokenizer.decode(result.generated_token_ids);
    if (!decoded.ok()) {
        result.errors = std::move(decoded.errors);
        return result;
    }
    result.text = std::move(decoded.value);
    const auto reads_after_decode = impl_->checkpoint->stats();
    const auto cuda_after_decode = impl_->cuda.stats();
    const auto cache_after_decode = impl_->weights->stats();
    const auto device_moe_after_decode = impl_->device_moe_stats;
    const double prefill_seconds = result.metrics.prefill_seconds;
    const double decode_seconds = result.metrics.decode_seconds;
    result.metrics = impl_->initialization_metrics;
    result.metrics.prefill_seconds = prefill_seconds;
    result.metrics.decode_seconds = decode_seconds;
    result.metrics.prompt_tokens = result.prompt_token_ids.size();
    result.metrics.decode_tokens = decode_steps;
    result.metrics.generation_checkpoint_reads = read_delta(reads_after_decode,
                                                             reads_before);
    result.metrics.decode_checkpoint_reads = read_delta(reads_after_decode,
                                                         reads_after_prefill);
    result.metrics.cuda = impl_->cuda.stats();
    result.metrics.cache = impl_->weights->stats();
    result.metrics.device_moe = device_moe_delta(
        device_moe_after_decode, device_moe_before);
    result.metrics.prefill.checkpoint_reads = read_delta(reads_after_prefill,
                                                         reads_before);
    result.metrics.prefill.cuda = cuda_delta(cuda_after_prefill, cuda_before);
    result.metrics.prefill.cache = cache_delta(cache_after_prefill, cache_before);
    result.metrics.prefill.device_moe = device_moe_delta(
        device_moe_after_prefill, device_moe_before);
    result.metrics.decode.checkpoint_reads = read_delta(reads_after_decode,
                                                        reads_after_prefill);
    result.metrics.decode.cuda = cuda_delta(cuda_after_decode, cuda_after_prefill);
    result.metrics.decode.cache = cache_delta(cache_after_decode, cache_after_prefill);
    result.metrics.decode.device_moe = device_moe_delta(
        device_moe_after_decode, device_moe_after_prefill);
    result.diagnostics = std::move(impl_->diagnostics);
    if (result.metrics.cache.lease_acquires !=
            result.metrics.cache.lease_releases ||
        std::any_of(result.metrics.cache.active_leases.begin(),
                    result.metrics.cache.active_leases.end(),
                    [](std::uint64_t count) { return count != 0U; })) {
        result.errors.emplace_back(
            "DeepSeek generation completed with outstanding CUDA weight leases");
    }
    if (impl_->config.require_zero_nvme_decode &&
        (result.metrics.decode_checkpoint_reads.calls != 0U ||
         result.metrics.decode_checkpoint_reads.bytes != 0U)) {
        result.errors.emplace_back(
            "DeepSeek zero-NVMe decode contract was violated by checkpoint reads");
    }
    if (impl_->route_trace.is_open()) {
        impl_->route_trace.flush();
        if (!impl_->route_trace.good()) {
            result.errors.emplace_back("cannot flush DeepSeek route trace");
        }
    }
    return result;
}

const Dsv4MemoryPlan& DeepSeekV4Runtime::memory_plan() const noexcept {
    return impl_->memory;
}

}  // namespace strata
