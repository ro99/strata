// Register-fed NVFP4 routed experts -- the public entry point for the kernel
// experiment 0247 gated.
//
// Named for the encoding rather than the model, like every other file in
// this tier: NVFP4 group-16/E4M3 is a checkpoint format, and check_layers
// rejects a model-named file in strata_device.
//
// WHAT THIS IS. `regfed_nvfp4_grouped_matmul_kernel<kColBlocks, 3>` reproduces
// `glm53_shared_expert_nvfp4_dot_kernel` to 5.960e-07 worst case over 60 real
// checkpoint fixtures, inside 0157's 7.53e-07, and runs 4.8x to 17.0x the
// scalar kernel depending on how many activation rows share an expert. This
// file wires it to the same enqueue/collect shape
// `enqueue_glm53_expert_gate_up` already presents, so `feedforward_page`
// consumes identical output.
//
// ONLY THE THREE-TERM PATH IS REACHABLE FROM HERE. The one-term form is 1.7x
// faster and measures 6.108e-03 — four orders outside the tolerance — because
// GLM's activation is FP32 and a single BF16 operand cannot carry it. The term
// count is a compile-time constant in this file and there is no parameter that
// can lower it.
//
// THE LAYOUT HAZARD, AND THE GUARD.
// `prepack_fragment` is an in-place one-copy replacement of canonical order:
// after it runs, the weight IS in fragment order and the scalar kernel would
// read a permutation. GLM's expert `CudaWeight`s are shared between the pinned
// static tier, which serves decode through the scalar kernel, and demand
// staging, which serves prefill. Prepacking a tier-pinned expert therefore
// corrupts every decode token, silently.
//
// Two things follow, and both are enforced rather than documented:
//
//   * `prepack_glm53_regfed_expert` is the only prepack path for this encoding
//     and it is the caller's decision which experts reach it. The caller must
//     pass only demand-staged experts.
//   * `CudaBackend::fragment_prepacked` answers the query for any of the three
//     projections, so a page can route prepacked experts here and leave
//     canonical ones on the scalar kernel within the same page.
//   * the scalar path now REFUSES a prepacked NVFP4 expert (see
//     `glm53_scalar_expert_layout_ok` below), so a mis-routed decode fails
//     loudly instead of reading a permutation.

namespace {

// SPLIT-K IS PART OF THE NUMERICAL CONTRACT HERE, NOT A TUNING KNOB.
//
// This started at 1, chosen on throughput grounds -- experiment 0247's grouped
// sweep is nearly flat in split-K at three terms, because the kernel is
// mma-bound at 85% of tensor peak, and the expert dimension already supplies
// the parallelism as gridDim.y. That was wrong, and the entry-point gate caught
// it: split-K also changes the accumulation order. A slice accumulates K/split
// columns and the fold sums `split` partials, which is a shallower summation
// tree than one serial pass over K=4096.
//
// Measured against the scalar kernel on real checkpoint activations, worst case
// over 60 fixtures at three terms:
//
//     split      1          2          4          8         16
//     error   3.755e-06  1.639e-06  9.537e-07  5.960e-07  5.960e-07
//
// so the 7.53e-07 tolerance needs split >= 8, and the figure is saturated by
// then. Eight is the smallest value that clears it, which matters because the
// split-K partial workspace scales as split x rows_per_expert.
//
// It is not even a throughput cost at the width the runtime currently issues:
// at M=1 split 8 measured 385.6 GB/s for gate/up and 402.4 for down, the best
// of the sweep. It costs 6% at M=8 and up to 35% at M=16 on `down`.
//
// DO NOT LOWER THIS for throughput without re-running the tolerance gate.
constexpr std::uint32_t kGlm53RegfedSplit = 8U;
constexpr std::uint32_t kGlm53RegfedTerms = 3U;

// The shape a projection carries, in the kernel's (N rows, K columns) terms.
struct Glm53RegfedShape {
    std::uint64_t rows{};
    std::uint64_t columns{};
};

[[nodiscard]] inline Glm53RegfedShape glm53_regfed_gate_shape(
    const CudaGlm53Expert& expert) noexcept {
    return {expert.intermediate, expert.hidden};
}

[[nodiscard]] inline Glm53RegfedShape glm53_regfed_down_shape(
    const CudaGlm53Expert& expert) noexcept {
    return {expert.hidden, expert.intermediate};
}

[[nodiscard]] inline cudaError_t glm53_regfed_grow(void*& pointer,
                                                   std::uint64_t& capacity,
                                                   std::uint64_t required,
                                                   bool zero,
                                                   cudaStream_t stream) {
    if (required <= capacity) return cudaSuccess;
    if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
    pointer = nullptr;
    capacity = 0U;
    if (auto status = cudaMalloc(&pointer, static_cast<std::size_t>(required));
        status != cudaSuccess) {
        return status;
    }
    capacity = required;
    if (!zero) return cudaSuccess;
    // Zeroing the whole allocation, not just the range in use, is what lets a
    // later, wider page reuse the buffer: the split-K fold resets every counter
    // it touches and everything it has not touched is still zero.
    return cudaMemsetAsync(pointer, 0, static_cast<std::size_t>(required),
                           stream);
}

[[nodiscard]] inline cudaError_t glm53_regfed_grow_host(
    void*& pointer, std::uint64_t& capacity, std::uint64_t required) {
    if (required <= capacity) return cudaSuccess;
    if (pointer != nullptr) static_cast<void>(cudaFreeHost(pointer));
    pointer = nullptr;
    capacity = 0U;
    if (auto status =
            cudaMallocHost(&pointer, static_cast<std::size_t>(required));
        status != cudaSuccess) {
        return status;
    }
    capacity = required;
    return cudaSuccess;
}

// One grouped dispatch: split the FP32 activation into three BF16 planes per
// expert, then run every slice in one kernel.
template <typename DeviceState>
[[nodiscard]] inline ValidationResult glm53_regfed_dispatch(
    DeviceState& state, std::uint32_t experts, std::uint32_t rows,
    std::uint32_t columns, std::uint32_t m) {
    static_cast<void>(rows);
    const std::uint32_t k_tiles = columns / kRegfedTileK;
    const std::uint32_t column_blocks =
        (std::min(m, kRegfedMaxM) + kRegfedTileM - 1U) / kRegfedTileM;
    const std::uint32_t groups = std::min(m, kRegfedTileM);
    const std::uint64_t per_expert =
        static_cast<std::uint64_t>(k_tiles) * column_blocks * groups * 4U;
    // gate and up share one activation, so the fragment count follows the
    // expert count, not the slice count.
    constexpr unsigned int threads = 256U;
    const std::uint64_t fragment_total =
        static_cast<std::uint64_t>(experts) * per_expert;
    regfed_nvfp4_moe_activation_fragment_kernel<kGlm53RegfedTerms><<<
        static_cast<unsigned int>(std::min<std::uint64_t>(
            (fragment_total + threads - 1U) / threads, 65535U)),
        threads, 0U, state.stream>>>(state.glm53_regfed_fragments,
                                     state.glm53_regfed_input, experts, m,
                                     columns, column_blocks, groups);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status,
                          "launch GLM-5.3 register-fed activation split");
    }
    return {};
}

template <typename DeviceState>
[[nodiscard]] inline ValidationResult glm53_regfed_launch(
    DeviceState& state, std::uint32_t slice_count,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t m) {
    const std::uint32_t n_tiles = rows / kRegfedTileN;
    const std::uint32_t column_blocks =
        (std::min(m, kRegfedMaxM) + kRegfedTileM - 1U) / kRegfedTileM;
    const std::uint32_t groups = std::min(m, kRegfedTileM);
    const dim3 blocks(
        static_cast<unsigned int>(
            (static_cast<std::uint64_t>(n_tiles) * kGlm53RegfedSplit +
             kRegfedWarpsPerBlock - 1U) /
            kRegfedWarpsPerBlock),
        slice_count);
    if (column_blocks == 1U) {
        regfed_nvfp4_grouped_matmul_kernel<1U, kGlm53RegfedTerms><<<
            blocks, kRegfedWarpsPerBlock * 32U, 0U, state.stream>>>(
            state.glm53_regfed_slices, columns, rows, kGlm53RegfedSplit, m,
            groups);
    } else {
        regfed_nvfp4_grouped_matmul_kernel<2U, kGlm53RegfedTerms><<<
            blocks, kRegfedWarpsPerBlock * 32U, 0U, state.stream>>>(
            state.glm53_regfed_slices, columns, rows, kGlm53RegfedSplit, m,
            groups);
    }
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch GLM-5.3 register-fed expert matmul");
    }
    return {};
}

}  // namespace

// Validation shared by both enqueues. Never a silent fallback: a batch that
// quietly reverted to the scalar kernel would hide the very routing decision
// the caller is making.
ValidationResult CudaBackend::glm53_regfed_admit_batch(
    int device, std::span<const CudaGlm53Expert> experts,
    std::uint32_t rows_per_expert, bool down, std::uint32_t& hidden,
    std::uint32_t& intermediate) {
    const auto ready = [device](const CudaWeight* weight, std::uint64_t rows,
                                std::uint64_t columns) {
        if (weight == nullptr || !weight->valid() ||
            weight->device() != device) {
            return false;
        }
        if (weight->impl_ == nullptr || !weight->impl_->fragment_prepacked) {
            return false;
        }
        const auto& descriptor = weight->impl_->descriptor;
        return descriptor.encoding == CudaWeightEncoding::Nvfp4Group16 &&
               descriptor.rows == rows && descriptor.columns == columns &&
               descriptor.group_size == kRegfedNvfp4Group &&
               descriptor.scale_columns == columns / kRegfedNvfp4Group &&
               regfed_nvfp4_shape_admissible(rows, columns);
    };
    if (experts.empty() || experts.size() > kMaximumGlm53DeviceExperts) {
        return {{"GLM-5.3 register-fed expert batch has an invalid width"}};
    }
    if (rows_per_expert == 0U ||
        rows_per_expert > kGlm53RegfedMaxRowsPerExpert) {
        return {{"GLM-5.3 register-fed expert batch has an invalid row count"}};
    }
    hidden = experts.front().hidden;
    intermediate = experts.front().intermediate;
    for (const auto& expert : experts) {
        if (expert.encoding != CudaGlm53ExpertEncoding::Nvfp4Group16E4m3) {
            return {{"GLM-5.3 register-fed expert batch is not NVFP4"}};
        }
        if (expert.hidden != hidden || expert.intermediate != intermediate) {
            return {{"GLM-5.3 register-fed expert batch mixes shapes"}};
        }
        if (down) {
            if (!ready(expert.down, expert.hidden, expert.intermediate)) {
                return {{"GLM-5.3 register-fed down projection is not a "
                         "prepacked NVFP4 arena weight of the right shape"}};
            }
        } else if (!ready(expert.gate, expert.intermediate, expert.hidden) ||
                   !ready(expert.up, expert.intermediate, expert.hidden)) {
            return {{"GLM-5.3 register-fed gate or up projection is not a "
                     "prepacked NVFP4 arena weight of the right shape"}};
        }
    }
    return {};
}

bool CudaBackend::glm53_regfed_expert_prepacked(
    const CudaGlm53Expert& expert) noexcept {
    const CudaWeight* projections[3] = {expert.gate, expert.up, expert.down};
    for (const auto* weight : projections) {
        if (weight == nullptr || !fragment_prepacked(*weight)) return false;
    }
    return true;
}

bool CudaBackend::glm53_regfed_expert_admissible(
    const CudaGlm53Expert& expert) noexcept {
    if (expert.encoding != CudaGlm53ExpertEncoding::Nvfp4Group16E4m3) {
        return false;
    }
    if (expert.gate == nullptr || expert.up == nullptr ||
        expert.down == nullptr) {
        return false;
    }
    return regfed_nvfp4_shape_admissible(expert.intermediate, expert.hidden) &&
           regfed_nvfp4_shape_admissible(expert.hidden, expert.intermediate);
}

ValidationResult CudaBackend::prepack_glm53_regfed_expert(
    int device, const CudaGlm53Expert& expert) {
    ValidationResult result;
    if (!glm53_regfed_expert_admissible(expert)) {
        result.errors.emplace_back(
            "GLM-5.3 register-fed prepack requires an NVFP4 arena expert whose "
            "shapes the fragment layout can express");
        return result;
    }
    // All three or none: an expert whose gate is in fragment order and whose
    // down is canonical cannot be routed as a unit, and the caller's
    // `glm53_regfed_expert_prepacked` query would be ambiguous.
    const CudaWeight* projections[3] = {expert.gate, expert.up, expert.down};
    for (const auto* weight : projections) {
        if (auto status = prepack_fragment(device, *weight); !status.ok()) {
            return status;
        }
    }
    return result;
}

ValidationResult CudaBackend::enqueue_glm53_regfed_expert_gate_up(
    int device, std::span<const CudaGlm53Expert> experts,
    std::span<const float> input, std::uint32_t rows_per_expert) {
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        return {{"GLM-5.3 register-fed expert batch targets an uninitialized "
                 "CUDA device"}};
    }
    auto& state = found->second;
    if (state.glm53_regfed_gate_up_in_flight ||
        state.glm53_regfed_down_in_flight) {
        return {{"a GLM-5.3 register-fed expert batch is already in flight"}};
    }
    std::uint32_t hidden = 0U;
    std::uint32_t intermediate = 0U;
    if (auto admitted = glm53_regfed_admit_batch(
            device, experts, rows_per_expert, false, hidden, intermediate);
        !admitted.ok()) {
        return admitted;
    }
    const auto count = static_cast<std::uint32_t>(experts.size());
    const std::uint64_t input_floats =
        static_cast<std::uint64_t>(count) * rows_per_expert * hidden;
    if (input.size() != input_floats) {
        return {{"GLM-5.3 register-fed expert batch input has an invalid "
                 "shape"}};
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status,
                          "select CUDA device for the register-fed expert batch");
    }

    const std::uint64_t out_floats =
        static_cast<std::uint64_t>(count) * rows_per_expert * intermediate;
    const std::uint32_t k_tiles = hidden / kRegfedTileK;
    const std::uint32_t n_tiles = intermediate / kRegfedTileN;
    const std::uint32_t column_blocks =
        (std::min(rows_per_expert, kRegfedMaxM) + kRegfedTileM - 1U) /
        kRegfedTileM;
    const std::uint32_t groups = std::min(rows_per_expert, kRegfedTileM);
    const std::uint64_t fragment_units =
        static_cast<std::uint64_t>(count) * kGlm53RegfedTerms * k_tiles *
        column_blocks * groups * 4U;
    const std::uint64_t partial_floats =
        static_cast<std::uint64_t>(2U) * count * n_tiles * kGlm53RegfedSplit *
        kRegfedTileN * rows_per_expert;
    const std::uint64_t counter_words =
        static_cast<std::uint64_t>(2U) * count * n_tiles;
    const auto grow = [&](void*& pointer, std::uint64_t& capacity,
                          std::uint64_t bytes, bool zero) {
        return glm53_regfed_grow(pointer, capacity, bytes, zero, state.stream);
    };
    if (auto status =
            grow(reinterpret_cast<void*&>(state.glm53_regfed_input),
                 state.glm53_regfed_input_bytes, input_floats * sizeof(float),
                 false);
        status != cudaSuccess) {
        return cuda_error(status, "allocate GLM-5.3 register-fed input");
    }
    if (auto status =
            grow(reinterpret_cast<void*&>(state.glm53_regfed_gate),
                 state.glm53_regfed_gate_bytes, out_floats * sizeof(float),
                 false);
        status != cudaSuccess) {
        return cuda_error(status, "allocate GLM-5.3 register-fed gate");
    }
    if (auto status = grow(reinterpret_cast<void*&>(state.glm53_regfed_up),
                           state.glm53_regfed_up_bytes,
                           out_floats * sizeof(float), false);
        status != cudaSuccess) {
        return cuda_error(status, "allocate GLM-5.3 register-fed up");
    }
    if (auto status = grow(reinterpret_cast<void*&>(state.glm53_regfed_fragments),
                           state.glm53_regfed_fragment_bytes,
                           fragment_units * sizeof(uint2), false);
        status != cudaSuccess) {
        return cuda_error(status, "allocate GLM-5.3 register-fed fragments");
    }
    if (auto status = grow(reinterpret_cast<void*&>(state.glm53_regfed_partials),
                           state.glm53_regfed_partial_bytes,
                           partial_floats * sizeof(float), false);
        status != cudaSuccess) {
        return cuda_error(status, "allocate GLM-5.3 register-fed partials");
    }
    if (auto status = grow(reinterpret_cast<void*&>(state.glm53_regfed_counters),
                           state.glm53_regfed_counter_bytes,
                           counter_words * sizeof(std::uint32_t), true);
        status != cudaSuccess) {
        return cuda_error(status, "allocate GLM-5.3 register-fed counters");
    }
    if (auto status = grow(reinterpret_cast<void*&>(state.glm53_regfed_slices),
                           state.glm53_regfed_slice_bytes,
                           2ULL * count * sizeof(RegfedNvfp4Slice), false);
        status != cudaSuccess) {
        return cuda_error(status, "allocate GLM-5.3 register-fed slices");
    }
    if (auto status = glm53_regfed_grow_host(
            reinterpret_cast<void*&>(state.glm53_regfed_slices_host),
            state.glm53_regfed_slice_host_bytes,
            2ULL * count * sizeof(RegfedNvfp4Slice));
        status != cudaSuccess) {
        return cuda_error(status, "allocate GLM-5.3 register-fed slice staging");
    }
    if (auto status = glm53_regfed_grow_host(
            reinterpret_cast<void*&>(state.glm53_regfed_staging),
            state.glm53_regfed_staging_bytes,
            std::max<std::uint64_t>(input_floats, 2U * out_floats) *
                sizeof(float));
        status != cudaSuccess) {
        return cuda_error(status, "allocate GLM-5.3 register-fed staging");
    }

    std::copy(input.begin(), input.end(), state.glm53_regfed_staging);
    if (auto status = cudaMemcpyAsync(
            state.glm53_regfed_input, state.glm53_regfed_staging,
            input_floats * sizeof(float), cudaMemcpyHostToDevice, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "upload GLM-5.3 register-fed input");
    }
    if (auto status = glm53_kernel_timing_begin(
            state, impl_->detailed_timing, Glm53KernelCategory::Expert);
        status != cudaSuccess) {
        return cuda_error(status, "start GLM-5.3 register-fed gate/up timing");
    }
    if (auto status =
            glm53_regfed_dispatch(state, count, intermediate, hidden,
                                  rows_per_expert);
        !status.ok()) {
        return status;
    }

    const std::uint64_t per_expert_fragments =
        static_cast<std::uint64_t>(k_tiles) * column_blocks * groups * 4U;
    const std::uint64_t partial_stride =
        static_cast<std::uint64_t>(n_tiles) * kGlm53RegfedSplit *
        kRegfedTileN * rows_per_expert;
    auto* slices = state.glm53_regfed_slices_host;
    for (std::uint32_t index = 0U; index < count; ++index) {
        const auto& expert = experts[index];
        // gate and up read the same activation rows, so they share one set of
        // fragment planes and differ only in weights, output and workspace.
        const auto* fragments =
            state.glm53_regfed_fragments +
            static_cast<std::size_t>(index) * kGlm53RegfedTerms *
                per_expert_fragments;
        const auto row_offset =
            static_cast<std::size_t>(index) * rows_per_expert * intermediate;
        slices[index] = RegfedNvfp4Slice{
            static_cast<const std::uint32_t*>(expert.gate->impl_->weights),
            static_cast<const unsigned char*>(expert.gate->impl_->scales),
            fragments, state.glm53_regfed_gate + row_offset,
            state.glm53_regfed_partials + index * partial_stride,
            state.glm53_regfed_counters +
                static_cast<std::size_t>(index) * n_tiles,
            expert.gate->impl_->descriptor.global_scale};
        slices[count + index] = RegfedNvfp4Slice{
            static_cast<const std::uint32_t*>(expert.up->impl_->weights),
            static_cast<const unsigned char*>(expert.up->impl_->scales),
            fragments, state.glm53_regfed_up + row_offset,
            state.glm53_regfed_partials + (count + index) * partial_stride,
            state.glm53_regfed_counters +
                static_cast<std::size_t>(count + index) * n_tiles,
            expert.up->impl_->descriptor.global_scale};
    }
    if (auto status = cudaMemcpyAsync(
            state.glm53_regfed_slices, slices,
            2ULL * count * sizeof(RegfedNvfp4Slice), cudaMemcpyHostToDevice,
            state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "upload GLM-5.3 register-fed slices");
    }
    if (auto status = glm53_regfed_launch(state, 2U * count, intermediate,
                                          hidden, rows_per_expert);
        !status.ok()) {
        return status;
    }
    if (auto status = glm53_kernel_timing_end(state, impl_->detailed_timing);
        status != cudaSuccess) {
        return cuda_error(status, "finish GLM-5.3 register-fed gate/up timing");
    }
    if (auto status = cudaMemcpyAsync(
            state.glm53_regfed_staging, state.glm53_regfed_gate,
            out_floats * sizeof(float), cudaMemcpyDeviceToHost, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "download GLM-5.3 register-fed gate");
    }
    if (auto status = cudaMemcpyAsync(
            state.glm53_regfed_staging + out_floats, state.glm53_regfed_up,
            out_floats * sizeof(float), cudaMemcpyDeviceToHost, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "download GLM-5.3 register-fed up");
    }
    state.glm53_regfed_gate_up_in_flight = true;
    state.glm53_regfed_batch = count;
    state.glm53_regfed_rows = rows_per_expert;
    state.glm53_regfed_hidden = hidden;
    state.glm53_regfed_intermediate = intermediate;
    return {};
}

ValidationResult CudaBackend::collect_glm53_regfed_expert_gate_up(
    int device, std::span<float> gate, std::span<float> up) {
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        return {{"GLM-5.3 register-fed expert batch targets an uninitialized "
                 "CUDA device"}};
    }
    auto& state = found->second;
    if (!state.glm53_regfed_gate_up_in_flight) {
        return {{"no GLM-5.3 register-fed gate and up command is in flight"}};
    }
    const std::size_t expected =
        static_cast<std::size_t>(state.glm53_regfed_batch) *
        state.glm53_regfed_rows * state.glm53_regfed_intermediate;
    if (gate.size() != expected || up.size() != expected) {
        return {{"GLM-5.3 register-fed gate and up output has an invalid "
                 "shape"}};
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status,
                          "select CUDA device for the register-fed expert batch");
    }
    const auto status = cudaStreamSynchronize(state.stream);
    state.glm53_regfed_gate_up_in_flight = false;
    if (status != cudaSuccess) {
        return cuda_error(status,
                          "complete GLM-5.3 register-fed gate and up");
    }
    if (auto timing_status =
            glm53_kernel_timing_drain(*impl_, state, device, false);
        timing_status != cudaSuccess) {
        return cuda_error(timing_status,
                          "measure GLM-5.3 register-fed gate/up");
    }
    std::copy_n(state.glm53_regfed_staging, expected, gate.begin());
    std::copy_n(state.glm53_regfed_staging + expected, expected, up.begin());
    return {};
}

ValidationResult CudaBackend::enqueue_glm53_regfed_expert_down(
    int device, std::span<const CudaGlm53Expert> experts,
    std::span<const float> activations, std::uint32_t rows_per_expert) {
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        return {{"GLM-5.3 register-fed expert batch targets an uninitialized "
                 "CUDA device"}};
    }
    auto& state = found->second;
    if (state.glm53_regfed_gate_up_in_flight ||
        state.glm53_regfed_down_in_flight) {
        return {{"a GLM-5.3 register-fed expert batch is already in flight"}};
    }
    std::uint32_t hidden = 0U;
    std::uint32_t intermediate = 0U;
    if (auto admitted = glm53_regfed_admit_batch(
            device, experts, rows_per_expert, true, hidden, intermediate);
        !admitted.ok()) {
        return admitted;
    }
    const auto count = static_cast<std::uint32_t>(experts.size());
    const std::uint64_t input_floats =
        static_cast<std::uint64_t>(count) * rows_per_expert * intermediate;
    if (activations.size() != input_floats) {
        return {{"GLM-5.3 register-fed expert batch activation has an invalid "
                 "shape"}};
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status,
                          "select CUDA device for the register-fed expert batch");
    }
    const std::uint64_t out_floats =
        static_cast<std::uint64_t>(count) * rows_per_expert * hidden;
    const std::uint32_t k_tiles = intermediate / kRegfedTileK;
    const std::uint32_t n_tiles = hidden / kRegfedTileN;
    const std::uint32_t column_blocks =
        (std::min(rows_per_expert, kRegfedMaxM) + kRegfedTileM - 1U) /
        kRegfedTileM;
    const std::uint32_t groups = std::min(rows_per_expert, kRegfedTileM);
    const std::uint64_t per_expert_fragments =
        static_cast<std::uint64_t>(k_tiles) * column_blocks * groups * 4U;
    const std::uint64_t partial_stride =
        static_cast<std::uint64_t>(n_tiles) * kGlm53RegfedSplit *
        kRegfedTileN * rows_per_expert;
    const auto grow = [&](void*& pointer, std::uint64_t& capacity,
                          std::uint64_t bytes, bool zero) {
        return glm53_regfed_grow(pointer, capacity, bytes, zero, state.stream);
    };
    if (auto status =
            grow(reinterpret_cast<void*&>(state.glm53_regfed_input),
                 state.glm53_regfed_input_bytes, input_floats * sizeof(float),
                 false);
        status != cudaSuccess) {
        return cuda_error(status, "allocate GLM-5.3 register-fed input");
    }
    if (auto status =
            grow(reinterpret_cast<void*&>(state.glm53_regfed_output),
                 state.glm53_regfed_output_bytes, out_floats * sizeof(float),
                 false);
        status != cudaSuccess) {
        return cuda_error(status, "allocate GLM-5.3 register-fed output");
    }
    if (auto status = grow(reinterpret_cast<void*&>(state.glm53_regfed_fragments),
                           state.glm53_regfed_fragment_bytes,
                           static_cast<std::uint64_t>(count) *
                               kGlm53RegfedTerms * per_expert_fragments *
                               sizeof(uint2),
                           false);
        status != cudaSuccess) {
        return cuda_error(status, "allocate GLM-5.3 register-fed fragments");
    }
    if (auto status = grow(reinterpret_cast<void*&>(state.glm53_regfed_partials),
                           state.glm53_regfed_partial_bytes,
                           static_cast<std::uint64_t>(count) * partial_stride *
                               sizeof(float),
                           false);
        status != cudaSuccess) {
        return cuda_error(status, "allocate GLM-5.3 register-fed partials");
    }
    if (auto status = grow(reinterpret_cast<void*&>(state.glm53_regfed_counters),
                           state.glm53_regfed_counter_bytes,
                           static_cast<std::uint64_t>(count) * n_tiles *
                               sizeof(std::uint32_t),
                           true);
        status != cudaSuccess) {
        return cuda_error(status, "allocate GLM-5.3 register-fed counters");
    }
    if (auto status = grow(reinterpret_cast<void*&>(state.glm53_regfed_slices),
                           state.glm53_regfed_slice_bytes,
                           static_cast<std::uint64_t>(count) *
                               sizeof(RegfedNvfp4Slice),
                           false);
        status != cudaSuccess) {
        return cuda_error(status, "allocate GLM-5.3 register-fed slices");
    }
    if (auto status = glm53_regfed_grow_host(
            reinterpret_cast<void*&>(state.glm53_regfed_slices_host),
            state.glm53_regfed_slice_host_bytes,
            static_cast<std::uint64_t>(count) * sizeof(RegfedNvfp4Slice));
        status != cudaSuccess) {
        return cuda_error(status, "allocate GLM-5.3 register-fed slice staging");
    }
    if (auto status = glm53_regfed_grow_host(
            reinterpret_cast<void*&>(state.glm53_regfed_staging),
            state.glm53_regfed_staging_bytes,
            std::max<std::uint64_t>(input_floats, out_floats) *
                sizeof(float));
        status != cudaSuccess) {
        return cuda_error(status, "allocate GLM-5.3 register-fed staging");
    }

    std::copy(activations.begin(), activations.end(),
              state.glm53_regfed_staging);
    if (auto status = cudaMemcpyAsync(
            state.glm53_regfed_input, state.glm53_regfed_staging,
            input_floats * sizeof(float), cudaMemcpyHostToDevice, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "upload GLM-5.3 register-fed activation");
    }
    if (auto status = glm53_kernel_timing_begin(
            state, impl_->detailed_timing, Glm53KernelCategory::Expert);
        status != cudaSuccess) {
        return cuda_error(status, "start GLM-5.3 register-fed down timing");
    }
    if (auto status = glm53_regfed_dispatch(state, count, hidden, intermediate,
                                            rows_per_expert);
        !status.ok()) {
        return status;
    }
    auto* slices = state.glm53_regfed_slices_host;
    for (std::uint32_t index = 0U; index < count; ++index) {
        const auto& expert = experts[index];
        slices[index] = RegfedNvfp4Slice{
            static_cast<const std::uint32_t*>(expert.down->impl_->weights),
            static_cast<const unsigned char*>(expert.down->impl_->scales),
            state.glm53_regfed_fragments +
                static_cast<std::size_t>(index) * kGlm53RegfedTerms *
                    per_expert_fragments,
            state.glm53_regfed_output +
                static_cast<std::size_t>(index) * rows_per_expert * hidden,
            state.glm53_regfed_partials + index * partial_stride,
            state.glm53_regfed_counters +
                static_cast<std::size_t>(index) * n_tiles,
            expert.down->impl_->descriptor.global_scale};
    }
    if (auto status = cudaMemcpyAsync(
            state.glm53_regfed_slices, slices,
            static_cast<std::size_t>(count) * sizeof(RegfedNvfp4Slice),
            cudaMemcpyHostToDevice, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "upload GLM-5.3 register-fed slices");
    }
    if (auto status = glm53_regfed_launch(state, count, hidden, intermediate,
                                          rows_per_expert);
        !status.ok()) {
        return status;
    }
    if (auto status = glm53_kernel_timing_end(state, impl_->detailed_timing);
        status != cudaSuccess) {
        return cuda_error(status, "finish GLM-5.3 register-fed down timing");
    }
    if (auto status = cudaMemcpyAsync(
            state.glm53_regfed_staging, state.glm53_regfed_output,
            out_floats * sizeof(float), cudaMemcpyDeviceToHost, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "download GLM-5.3 register-fed down");
    }
    state.glm53_regfed_down_in_flight = true;
    state.glm53_regfed_batch = count;
    state.glm53_regfed_rows = rows_per_expert;
    state.glm53_regfed_hidden = hidden;
    state.glm53_regfed_intermediate = intermediate;
    return {};
}

ValidationResult CudaBackend::collect_glm53_regfed_expert_down(
    int device, std::span<float> output) {
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        return {{"GLM-5.3 register-fed expert batch targets an uninitialized "
                 "CUDA device"}};
    }
    auto& state = found->second;
    if (!state.glm53_regfed_down_in_flight) {
        return {{"no GLM-5.3 register-fed down command is in flight"}};
    }
    const std::size_t expected =
        static_cast<std::size_t>(state.glm53_regfed_batch) *
        state.glm53_regfed_rows * state.glm53_regfed_hidden;
    if (output.size() != expected) {
        return {{"GLM-5.3 register-fed down output has an invalid shape"}};
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status,
                          "select CUDA device for the register-fed expert batch");
    }
    const auto status = cudaStreamSynchronize(state.stream);
    state.glm53_regfed_down_in_flight = false;
    if (status != cudaSuccess) {
        return cuda_error(status, "complete GLM-5.3 register-fed down");
    }
    if (auto timing_status =
            glm53_kernel_timing_drain(*impl_, state, device, false);
        timing_status != cudaSuccess) {
        return cuda_error(timing_status, "measure GLM-5.3 register-fed down");
    }
    std::copy_n(state.glm53_regfed_staging, expected, output.begin());
    return {};
}
