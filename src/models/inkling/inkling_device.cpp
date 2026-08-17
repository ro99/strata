#include "strata/inkling_device.hpp"

#include "strata/model_adapter.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <vector>

namespace strata {
namespace {

constexpr auto& kContract = kInklingExecutionContract;

std::uint64_t elapsed_since(
    std::chrono::steady_clock::time_point started) noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count());
}

// Copies every other row out of an interleaved [rows, stride] block. `up`
// selects the odd rows. The result is contiguous and half as tall.
void deinterleave_rows(std::span<const std::byte> source,
                       std::span<std::byte> destination, std::uint64_t rows,
                       std::uint64_t stride, bool up) {
    const auto half = rows / 2U;
    for (std::uint64_t row = 0U; row < half; ++row) {
        const auto source_row = row * 2U + (up ? 1U : 0U);
        std::memcpy(destination.data() + row * stride,
                    source.data() + source_row * stride,
                    static_cast<std::size_t>(stride));
    }
}

}  // namespace

ValidationResult load_inkling_cuda_linear(
    const InklingCheckpointReader& checkpoint, const InklingLinear& module,
    int device, CudaBackend& backend, CudaWeight& output) {
    ValidationResult result;
    if (module.weight == nullptr) {
        result.errors.emplace_back("Inkling linear is not resolved");
        return result;
    }
    auto view = checkpoint.view(module.weight->name);
    if (!view.ok()) {
        result.errors = std::move(view.errors);
        return result;
    }
    CudaWeightDescriptor descriptor;
    descriptor.encoding = CudaWeightEncoding::Plain;
    descriptor.dtype = module.weight->dtype;
    descriptor.rows = module.rows;
    descriptor.columns = module.columns;
    return backend.upload(device, descriptor, view.value, {}, output);
}

ValidationResult load_inkling_cuda_interleaved_half(
    const InklingCheckpointReader& checkpoint, const std::string& name,
    std::uint64_t slice, std::uint64_t rows, std::uint64_t columns, bool up,
    int device, CudaBackend& backend, CudaWeight& output) {
    ValidationResult result;
    auto view = checkpoint.view(name);
    if (!view.ok()) {
        result.errors = std::move(view.errors);
        return result;
    }
    const auto stride = columns * sizeof(std::uint16_t);
    const auto block = rows * stride;
    if (view.value.size() < (slice + 1U) * block) {
        result.errors.emplace_back("Inkling interleaved slice exceeds " + name);
        return result;
    }
    std::vector<std::byte> half(static_cast<std::size_t>(block / 2U));
    deinterleave_rows(view.value.subspan(static_cast<std::size_t>(slice * block),
                                         static_cast<std::size_t>(block)),
                      half, rows, stride, up);
    CudaWeightDescriptor descriptor;
    descriptor.encoding = CudaWeightEncoding::Plain;
    descriptor.dtype = SafetensorsDtype::Bf16;
    descriptor.rows = rows / 2U;
    descriptor.columns = columns;
    return backend.upload(device, descriptor, half, {}, output);
}

InklingExpertCache::InklingExpertCache(
    const InklingCheckpointReader& checkpoint, CudaBackend& backend,
    std::vector<int> devices, std::vector<std::uint64_t> capacities)
    : checkpoint_(checkpoint), backend_(backend), devices_(std::move(devices)) {
    // The largest staged block is a plain BF16 half of layer 2's interleaved
    // w13 (2048 x 4096 x 2 bytes); the NVFP4 blocks are a quarter of that.
    constexpr std::size_t kWeightScratch = 2048U * 4096U * sizeof(std::uint16_t);
    constexpr std::size_t kScaleScratch = 4096U * 256U;
    states_.reserve(capacities.size());
    for (std::size_t slot = 0U; slot < capacities.size(); ++slot) {
        auto state = std::make_unique<State>();
        state->capacity = capacities[slot];
        state->scratch.weights.resize(kWeightScratch);
        state->scratch.scales.resize(kScaleScratch);
        const auto weights = backend_.register_host_memory(
            state->scratch.weights.data(), kWeightScratch);
        const auto scales = backend_.register_host_memory(
            state->scratch.scales.data(), kScaleScratch);
        state->scratch.registered = weights.ok() && scales.ok();
        states_.push_back(std::move(state));
    }
}

InklingExpertCache::~InklingExpertCache() {
    for (auto& state : states_) {
        if (!state->scratch.registered) continue;
        backend_.unregister_host_memory(state->scratch.weights.data());
        backend_.unregister_host_memory(state->scratch.scales.data());
    }
}

bool InklingExpertCache::evict_locked(State& state, std::uint64_t bytes) {
    if (bytes > state.capacity) return false;
    while (state.used + bytes > state.capacity) {
        std::uint64_t oldest = 0U;
        std::uint64_t oldest_clock = std::numeric_limits<std::uint64_t>::max();
        bool found = false;
        for (const auto& [key, entry] : state.entries) {
            // A leased entry is feeding an in-flight device command.
            if (entry.leases != 0U) continue;
            if (entry.last_use < oldest_clock) {
                oldest_clock = entry.last_use;
                oldest = key;
                found = true;
            }
        }
        if (!found) return false;
        state.used -= state.entries[oldest].bytes;
        state.entries.erase(oldest);
        ++state.evictions;
    }
    return true;
}

ParseResult<const InklingDeviceExpert*> InklingExpertCache::acquire(
    std::size_t device_slot, std::uint32_t layer, std::uint32_t expert,
    const InklingExpertStack& gate_up, const InklingExpertStack& down) {
    ParseResult<const InklingDeviceExpert*> result;
    if (device_slot >= states_.size()) {
        result.errors.emplace_back("expert targets an invalid device slot");
        return result;
    }
    auto& state = *states_[device_slot];
    const auto device = devices_[device_slot];
    const auto key = key_of(layer, expert);

    std::scoped_lock lock(state.mutex);
    if (const auto found = state.entries.find(key); found != state.entries.end()) {
        found->second.last_use = ++state.clock;
        ++found->second.leases;
        ++state.hits;
        result.value = &found->second.expert;
        return result;
    }
    ++state.misses;

    const auto started = std::chrono::steady_clock::now();
    Entry entry;
    const bool quantized = gate_up.encoding == InklingTensorEncoding::Nvfp4Group16;

    // Gate and up come out of one interleaved tensor; down is stored plainly.
    if (quantized) {
        auto matrix = checkpoint_.nvfp4_expert_view(gate_up, expert);
        if (!matrix.ok()) {
            result.errors = std::move(matrix.errors);
            return result;
        }
        const auto half_rows = matrix.value.rows / 2U;
        const auto packed_stride = matrix.value.packed_columns;
        const auto scale_stride = matrix.value.scale_columns;
        const auto packed_bytes =
            static_cast<std::size_t>(half_rows * packed_stride);
        const auto scale_bytes =
            static_cast<std::size_t>(half_rows * scale_stride);
        if (state.scratch.weights.size() < packed_bytes ||
            state.scratch.scales.size() < scale_bytes) {
            result.errors.emplace_back("Inkling staging scratch is too small");
            return result;
        }
        auto packed = std::span<std::byte>(state.scratch.weights).first(packed_bytes);
        auto scales = std::span<std::byte>(state.scratch.scales).first(scale_bytes);
        CudaWeightDescriptor descriptor;
        descriptor.encoding = CudaWeightEncoding::Nvfp4Group16;
        descriptor.dtype = SafetensorsDtype::U8;
        descriptor.rows = half_rows;
        descriptor.columns = matrix.value.columns;
        descriptor.packed_columns = packed_stride;
        descriptor.scale_columns = scale_stride;
        descriptor.group_size = matrix.value.group_size;
        // The kernel divides the group scale by global_scale, but Inkling's
        // ModelOpt scale is a multiplier, so the reciprocal is what makes the
        // shared kernel compute e2m1 * e4m3 * scale2.
        descriptor.global_scale = 1.0F / matrix.value.global_scale;
        for (const bool up : {false, true}) {
            deinterleave_rows(matrix.value.packed, packed, matrix.value.rows,
                              packed_stride, up);
            deinterleave_rows(matrix.value.scales, scales, matrix.value.rows,
                              scale_stride, up);
            auto status = backend_.upload(device, descriptor, packed, scales,
                                          up ? entry.expert.up
                                             : entry.expert.gate);
            if (!status.ok()) {
                result.errors = std::move(status.errors);
                return result;
            }
        }
        auto down_matrix = checkpoint_.nvfp4_expert_view(down, expert);
        if (!down_matrix.ok()) {
            result.errors = std::move(down_matrix.errors);
            return result;
        }
        CudaWeightDescriptor down_descriptor;
        down_descriptor.encoding = CudaWeightEncoding::Nvfp4Group16;
        down_descriptor.dtype = SafetensorsDtype::U8;
        down_descriptor.rows = down_matrix.value.rows;
        down_descriptor.columns = down_matrix.value.columns;
        down_descriptor.packed_columns = down_matrix.value.packed_columns;
        down_descriptor.scale_columns = down_matrix.value.scale_columns;
        down_descriptor.group_size = down_matrix.value.group_size;
        down_descriptor.global_scale = 1.0F / down_matrix.value.global_scale;
        const auto down_packed = down_matrix.value.packed.size();
        const auto down_scales = down_matrix.value.scales.size();
        if (state.scratch.weights.size() < down_packed ||
            state.scratch.scales.size() < down_scales) {
            result.errors.emplace_back("Inkling staging scratch is too small");
            return result;
        }
        // `down` needs no de-interleave, but it is still a cold mapped slice,
        // so it goes through the pinned buffer for the same reason.
        std::memcpy(state.scratch.weights.data(), down_matrix.value.packed.data(),
                    down_packed);
        std::memcpy(state.scratch.scales.data(), down_matrix.value.scales.data(),
                    down_scales);
        auto status = backend_.upload(
            device, down_descriptor,
            std::span<const std::byte>(state.scratch.weights).first(down_packed),
            std::span<const std::byte>(state.scratch.scales).first(down_scales),
            entry.expert.down);
        if (!status.ok()) {
            result.errors = std::move(status.errors);
            return result;
        }
    } else {
        // Layer 2 ships plain BF16 experts.
        auto gate_up_view = checkpoint_.view(gate_up.weight->name);
        if (!gate_up_view.ok()) {
            result.errors = std::move(gate_up_view.errors);
            return result;
        }
        const auto stride = gate_up.columns * sizeof(std::uint16_t);
        const auto block = gate_up.rows * stride;
        const auto half_bytes = static_cast<std::size_t>(block / 2U);
        if (state.scratch.weights.size() < half_bytes) {
            result.errors.emplace_back("Inkling staging scratch is too small");
            return result;
        }
        auto half = std::span<std::byte>(state.scratch.weights).first(half_bytes);
        CudaWeightDescriptor descriptor;
        descriptor.encoding = CudaWeightEncoding::Plain;
        descriptor.dtype = SafetensorsDtype::Bf16;
        descriptor.rows = gate_up.rows / 2U;
        descriptor.columns = gate_up.columns;
        for (const bool up : {false, true}) {
            deinterleave_rows(
                gate_up_view.value.subspan(
                    static_cast<std::size_t>(expert * block),
                    static_cast<std::size_t>(block)),
                half, gate_up.rows, stride, up);
            auto status = backend_.upload(device, descriptor, half, {},
                                          up ? entry.expert.up
                                             : entry.expert.gate);
            if (!status.ok()) {
                result.errors = std::move(status.errors);
                return result;
            }
        }
        auto down_view = checkpoint_.view(down.weight->name);
        if (!down_view.ok()) {
            result.errors = std::move(down_view.errors);
            return result;
        }
        const auto down_block = down.rows * down.columns * sizeof(std::uint16_t);
        CudaWeightDescriptor down_descriptor;
        down_descriptor.encoding = CudaWeightEncoding::Plain;
        down_descriptor.dtype = SafetensorsDtype::Bf16;
        down_descriptor.rows = down.rows;
        down_descriptor.columns = down.columns;
        if (state.scratch.weights.size() < down_block) {
            result.errors.emplace_back("Inkling staging scratch is too small");
            return result;
        }
        std::memcpy(state.scratch.weights.data(),
                    down_view.value.data() +
                        static_cast<std::size_t>(expert * down_block),
                    static_cast<std::size_t>(down_block));
        auto status = backend_.upload(
            device, down_descriptor,
            std::span<const std::byte>(state.scratch.weights)
                .first(static_cast<std::size_t>(down_block)),
            {}, entry.expert.down);
        if (!status.ok()) {
            result.errors = std::move(status.errors);
            return result;
        }
    }

    entry.bytes = entry.expert.device_bytes();
    state.stage_nanoseconds += elapsed_since(started);
    state.staged_bytes += entry.bytes;
    if (!evict_locked(state, entry.bytes)) {
        result.errors.emplace_back(
            "Inkling expert cache cannot make room for a routed expert");
        return result;
    }
    entry.last_use = ++state.clock;
    entry.leases = 1U;
    state.used += entry.bytes;
    state.peak = std::max(state.peak, state.used);
    const auto inserted = state.entries.emplace(key, std::move(entry));
    result.value = &inserted.first->second.expert;
    return result;
}

void InklingExpertCache::release(std::size_t device_slot, std::uint32_t layer,
                                 std::uint32_t expert) noexcept {
    if (device_slot >= states_.size()) return;
    auto& state = *states_[device_slot];
    std::scoped_lock lock(state.mutex);
    if (const auto found = state.entries.find(key_of(layer, expert));
        found != state.entries.end() && found->second.leases != 0U) {
        --found->second.leases;
    }
}

InklingCacheStats InklingExpertCache::stats() const {
    InklingCacheStats stats;
    for (const auto& state : states_) {
        std::scoped_lock lock(state->mutex);
        stats.hits += state->hits;
        stats.misses += state->misses;
        stats.evictions += state->evictions;
        stats.used_bytes.push_back(state->used);
        stats.capacity_bytes.push_back(state->capacity);
        stats.peak_bytes.push_back(state->peak);
        stats.stage_nanoseconds += state->stage_nanoseconds;
        stats.staged_bytes += state->staged_bytes;
    }
    return stats;
}

}  // namespace strata
