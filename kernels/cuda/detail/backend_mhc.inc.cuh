ValidationResult CudaBackend::upload_dsv4_mhc_weights(
    int device, std::span<const float> projection,
    std::span<const float> scale, std::span<const float> base,
    std::span<const float> norm_weight, CudaDsv4MhcWeights& output) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek device mHC weight upload targets an uninitialized device");
        return result;
    }
    auto& state = found->second;
    if (!state.dsv4_mhc_supported) {
        result.errors.emplace_back(
            "exact DeepSeek device mHC requires an SM86 device");
        return result;
    }
    if (state.moe_in_flight || state.dsv4_mhc_stage != 0U ||
        projection.size() != kDsv4MhcProjectionElements ||
        scale.size() != 3U || base.size() != kDsv4MhcMixes ||
        norm_weight.size() != kDsv4MhcHidden) {
        result.errors.emplace_back(
            "DeepSeek device mHC weight shapes or command state are invalid");
        return result;
    }
    for (const auto values : {projection, scale, base, norm_weight}) {
        if (!std::all_of(values.begin(), values.end(), [](float value) {
                return std::isfinite(value);
            })) {
            result.errors.emplace_back(
                "DeepSeek device mHC weights contain a non-finite value");
            return result;
        }
    }

    auto target = std::make_shared<CudaDsv4MhcWeights::Impl>();
    CudaWeightDescriptor descriptor;
    descriptor.encoding = CudaWeightEncoding::Plain;
    descriptor.dtype = SafetensorsDtype::F32;
    descriptor.rows = kDsv4MhcMixes;
    descriptor.columns = kDsv4MhcMultiplier * kDsv4MhcHidden;
    result = upload(device, descriptor, std::as_bytes(projection), {},
                    target->projection);
    if (!result.ok()) return result;

    std::vector<std::byte> auxiliary(kDsv4MhcAuxBytes);
    std::memcpy(auxiliary.data(), scale.data(), scale.size_bytes());
    std::memcpy(auxiliary.data() + scale.size_bytes(), base.data(),
                base.size_bytes());
    auto* norm_bf16 = reinterpret_cast<std::uint16_t*>(
        auxiliary.data() + kDsv4MhcAuxNormOffset);
    for (std::size_t index = 0U; index < norm_weight.size(); ++index) {
        norm_bf16[index] = bf16_encode(norm_weight[index]);
    }
    auto auxiliary_target = std::make_shared<CudaBuffer::Impl>();
    auxiliary_target->bytes = auxiliary.size();
    auxiliary_target->device = device;
    const auto allocation_started = std::chrono::steady_clock::now();
    if (auto status = cudaMalloc(
            &auxiliary_target->data, auxiliary.size());
        status != cudaSuccess) {
        return cuda_error(
            status, "allocate DeepSeek device mHC auxiliary weights");
    }
    const auto allocation_nanoseconds = elapsed_nanoseconds_since(
        allocation_started);
    const auto copy_started = std::chrono::steady_clock::now();
    if (auto status = cudaMemcpyAsync(
            auxiliary_target->data, auxiliary.data(), auxiliary.size(),
            cudaMemcpyHostToDevice, state.stream); status != cudaSuccess) {
        static_cast<void>(cudaStreamSynchronize(state.stream));
        return cuda_error(
            status, "upload DeepSeek device mHC auxiliary weights");
    }
    const auto copy_nanoseconds = elapsed_nanoseconds_since(copy_started);
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaStreamSynchronize(state.stream);
        status != cudaSuccess) {
        state.quarantined_buffers.push_back(std::move(auxiliary_target));
        return cuda_error(
            status, "synchronize DeepSeek device mHC auxiliary weights");
    }
    const auto wait_nanoseconds = elapsed_nanoseconds_since(wait_started);
    target->auxiliary.impl_ = std::move(auxiliary_target);
    {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        stats.weight_upload_bytes += auxiliary.size();
        ++stats.weight_allocation_calls;
        stats.weight_allocation_bytes += auxiliary.size();
        stats.weight_allocation_nanoseconds += allocation_nanoseconds;
        stats.weight_copy_nanoseconds += copy_nanoseconds;
        record_synchronization(stats, SynchronizationSubsystem::Weight,
                               1U, wait_nanoseconds);
        stats.upload_wait_nanoseconds += wait_nanoseconds;
        stats.dsv4_mhc_resident_weight_bytes +=
            target->projection.device_bytes() +
            target->auxiliary.device_bytes();
    }
    output.impl_ = std::move(target);
    return result;
}

ValidationResult CudaBackend::dsv4_mhc_select_slot(
    int device, std::uint32_t slot) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek device mHC slot selection targets an uninitialized device");
        return result;
    }
    auto& state = found->second;
    if (!state.dsv4_mhc_supported || slot >= kDsv4MhcMaximumSlots) {
        result.errors.emplace_back(
            "DeepSeek device mHC slot selection is out of range");
        return result;
    }
    if (state.moe_in_flight || state.dsv4_mhc_head_in_flight) {
        // A slot swap rebinds what every later command reads. Doing it while
        // asynchronous work still owns the current workspace would let that
        // work land in another row's state.
        result.errors.emplace_back(
            "DeepSeek device mHC slot selection is out of order");
        return result;
    }
    if (slot == state.dsv4_mhc_active_slot) return result;
    if (slot >= state.dsv4_mhc_slot_capacity) {
        result.errors.emplace_back(
            "DeepSeek device mHC slot selection exceeds the reservation");
        return result;
    }
    const auto required = static_cast<std::size_t>(
        std::max(slot, state.dsv4_mhc_active_slot)) + 1U;
    if (state.dsv4_mhc_saved_slots.size() < required) {
        state.dsv4_mhc_saved_slots.resize(required);
    }
    auto& outgoing = state.dsv4_mhc_saved_slots[state.dsv4_mhc_active_slot];
    outgoing.stage = state.dsv4_mhc_stage;
    outgoing.residual_index = state.dsv4_mhc_residual_index;
    outgoing.branch_ready = state.dsv4_mhc_branch_ready;
    const auto& incoming = state.dsv4_mhc_saved_slots[slot];
    state.dsv4_mhc_workspace = state.dsv4_mhc_slot_arena + slot;
    state.dsv4_mhc_stage = incoming.stage;
    state.dsv4_mhc_residual_index = incoming.residual_index;
    state.dsv4_mhc_branch_ready = incoming.branch_ready;
    state.dsv4_mhc_workspace_bytes = sizeof(Dsv4MhcWorkspace);
    state.dsv4_mhc_active_slot = slot;
    return result;
}

ValidationResult CudaBackend::dsv4_mhc_reserve_slots(
    int device, std::uint32_t slots) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek device mHC slot reservation targets an uninitialized device");
        return result;
    }
    auto& state = found->second;
    if (!state.dsv4_mhc_supported || slots == 0U ||
        slots > kDsv4MhcMaximumSlots) {
        result.errors.emplace_back(
            "DeepSeek device mHC slot reservation is out of range");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status,
                          "select CUDA device for device mHC slot reservation");
    }
    if (state.dsv4_mhc_stage != 0U || state.moe_in_flight) {
        // Growing the arena moves every slot, so no row may be mid-flight.
        // A stage left non-zero by an aborted request is the usual cause, so
        // report which of the two conditions held.
        result.errors.emplace_back(
            std::string("DeepSeek device mHC slot reservation is out of order"
                        " (stage=") +
            std::to_string(state.dsv4_mhc_stage) + " moe_in_flight=" +
            (state.moe_in_flight ? "true" : "false") + ")");
        return result;
    }
    std::uint64_t allocation_calls = 0U;
    std::uint64_t allocation_bytes = 0U;
    if (slots > state.dsv4_mhc_slot_capacity) {
        void* allocation = nullptr;
        const auto bytes = static_cast<std::size_t>(slots) *
                           sizeof(Dsv4MhcWorkspace);
        if (auto status = cudaMalloc(&allocation, bytes);
            status != cudaSuccess) {
            return cuda_error(
                status, "allocate DeepSeek device mHC slot arena");
        }
        if (state.dsv4_mhc_slot_arena != nullptr) {
            static_cast<void>(cudaFree(state.dsv4_mhc_slot_arena));
        } else if (state.dsv4_mhc_workspace != nullptr) {
            // The single-state allocation the token-major path made.
            static_cast<void>(cudaFree(state.dsv4_mhc_workspace));
        }
        state.dsv4_mhc_slot_arena = static_cast<Dsv4MhcWorkspace*>(allocation);
        state.dsv4_mhc_slot_capacity = slots;
        ++allocation_calls;
        allocation_bytes += bytes;
    }
    if (state.dsv4_mhc_saved_slots.size() < slots) {
        state.dsv4_mhc_saved_slots.resize(slots);
    }
    state.dsv4_mhc_workspace =
        state.dsv4_mhc_slot_arena + state.dsv4_mhc_active_slot;
    state.dsv4_mhc_workspace_bytes = sizeof(Dsv4MhcWorkspace);
    if (allocation_calls != 0U) {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        stats.workspace_allocation_calls += allocation_calls;
        stats.workspace_allocation_bytes += allocation_bytes;
    }
    return result;
}

ValidationResult CudaBackend::dsv4_mhc_begin(
    int device, const CudaDsv4MhcWeights& weights,
    std::span<const float> hidden, std::span<float> weighted,
    std::span<float> layer_input) {
    return dsv4_mhc_begin_impl(
        device, weights, hidden, weighted, layer_input, false);
}

ValidationResult CudaBackend::dsv4_mhc_begin_device(
    int device, const CudaDsv4MhcWeights& weights,
    std::span<const float> hidden) {
    return dsv4_mhc_begin_impl(device, weights, hidden, {}, {}, true);
}

ValidationResult CudaBackend::dsv4_mhc_begin_impl(
    int device, const CudaDsv4MhcWeights& weights,
    std::span<const float> hidden, std::span<float> weighted,
    std::span<float> layer_input, bool device_only) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek device mHC begin targets an uninitialized device");
        return result;
    }
    auto& state = found->second;
    if (!weights.valid() || weights.device() != device ||
        !state.dsv4_mhc_supported || state.moe_in_flight ||
        state.dsv4_mhc_stage != 0U ||
        hidden.size() != kDsv4MhcMultiplier * kDsv4MhcHidden ||
        (!weighted.empty() && weighted.size() != kDsv4MhcHidden) ||
        ((!device_only && layer_input.size() != kDsv4MhcHidden) ||
         (device_only && (!weighted.empty() || !layer_input.empty()))) ||
        !std::all_of(hidden.begin(), hidden.end(), [](float value) {
            return std::isfinite(value);
        })) {
        result.errors.emplace_back(
            "DeepSeek device mHC begin request or command order is invalid");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for device mHC begin");
    }
    std::uint64_t allocation_calls = 0U;
    std::uint64_t allocation_bytes = 0U;
    if (state.dsv4_mhc_workspace == nullptr) {
        if (auto status = cudaMalloc(
                &state.dsv4_mhc_workspace, sizeof(Dsv4MhcWorkspace));
            status != cudaSuccess) {
            return cuda_error(status,
                              "allocate DeepSeek device mHC workspace");
        }
        state.dsv4_mhc_workspace_bytes = sizeof(Dsv4MhcWorkspace);
        ++allocation_calls;
        allocation_bytes += sizeof(Dsv4MhcWorkspace);
    }
    if (state.dsv4_mhc_host_staging == nullptr) {
        void* staging = nullptr;
        if (auto status = cudaMallocHost(
                &staging,
                static_cast<std::size_t>(kDsv4MhcMaximumHostStagingBytes));
            status != cudaSuccess) {
            return cuda_error(
                status, "allocate pinned DeepSeek device mHC staging");
        }
        state.dsv4_mhc_host_staging = static_cast<std::byte*>(staging);
        state.dsv4_mhc_host_staging_bytes =
            kDsv4MhcMaximumHostStagingBytes;
        ++allocation_calls;
        allocation_bytes += kDsv4MhcMaximumHostStagingBytes;
    }
    auto* host_hidden = reinterpret_cast<std::uint16_t*>(
        state.dsv4_mhc_host_staging);
    for (std::size_t index = 0U; index < hidden.size(); ++index) {
        host_hidden[index] = bf16_encode(hidden[index]);
    }
    auto* workspace = state.dsv4_mhc_workspace;
    const auto* projection = static_cast<const float*>(
        weights.impl_->projection.impl_->weights);
    const auto* auxiliary = static_cast<const std::byte*>(
        weights.impl_->auxiliary.impl_->data);
    const auto* scale = reinterpret_cast<const float*>(auxiliary);
    const auto* base = reinterpret_cast<const float*>(
        auxiliary + 3U * sizeof(float));
    const auto* norm = reinterpret_cast<const __nv_bfloat16*>(
        auxiliary + kDsv4MhcAuxNormOffset);
    const auto h2d_bytes = hidden.size() * sizeof(std::uint16_t);
    const auto weighted_bytes = weighted.size() * sizeof(std::uint16_t);
    const auto layer_bytes = layer_input.size() * sizeof(std::uint16_t);
    const auto d2h_bytes = weighted_bytes + layer_bytes;
    const auto operation_started = std::chrono::steady_clock::now();
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_start, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "record DeepSeek device mHC begin upload start");
        }
    }
    if (auto status = cudaMemcpyAsync(
            workspace->residual[0], host_hidden,
            static_cast<std::size_t>(h2d_bytes), cudaMemcpyHostToDevice,
            state.stream); status != cudaSuccess) {
        return cuda_error(status, "upload DeepSeek device mHC residual");
    }
    if (auto status = cudaMemsetAsync(
            &workspace->failure, 0, sizeof(workspace->failure), state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "clear DeepSeek device mHC status");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                state.activation_uploaded, state.stream);
            status != cudaSuccess) {
            return cuda_error(
                status, "record DeepSeek device mHC begin upload completion");
        }
    }
    dsv4_mhc_standalone_projection<<<
        dim3{2U, kDsv4MhcStandaloneSplits}, 32U, 0U, state.stream>>>(
        workspace->residual[0], projection, workspace->partial_projection);
    dsv4_mhc_standalone_square_sum<<<
        kDsv4MhcStandaloneSplits, 8U, 0U, state.stream>>>(
        workspace->residual[0], workspace->partial_square_sum);
    dsv4_mhc_mix<<<1U, 32U, 0U, state.stream>>>(
        workspace->partial_projection, workspace->partial_square_sum,
        scale, base, kDsv4MhcStandaloneSplits, workspace->pre,
        workspace->post, workspace->combination);
    dsv4_mhc_weighted_norm<<<1U, kDsv4MhcWeightedNormThreads, 0U,
                             state.stream>>>(
        workspace->residual[0], workspace->pre, norm,
        workspace->weighted, workspace->layer_input);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch DeepSeek device mHC begin");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                state.kernel_finished, state.stream);
            status != cudaSuccess) {
            return cuda_error(
                status, "record DeepSeek device mHC begin kernels");
        }
    }
    if (device_only) {
        state.dsv4_mhc_stage = 1U;
        state.dsv4_mhc_residual_index = 0U;
        state.dsv4_mhc_branch_ready = false;
        state.dsv4_mhc_failed = false;
        const auto operation_nanoseconds = elapsed_nanoseconds_since(
            operation_started);
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        ++stats.dsv4_mhc_calls;
        ++stats.dsv4_mhc_standalone_calls;
        stats.dsv4_mhc_kernel_launches += 4U;
        stats.dsv4_mhc_h2d_bytes += h2d_bytes;
        stats.dsv4_mhc_nanoseconds += operation_nanoseconds;
        stats.dsv4_mhc_host_nanoseconds += operation_nanoseconds;
        stats.activation_h2d_bytes += h2d_bytes;
        stats.workspace_allocation_calls += allocation_calls;
        stats.workspace_allocation_bytes += allocation_bytes;
        return result;
    }
    if (!weighted.empty()) {
        if (auto status = cudaMemcpyAsync(
                state.dsv4_mhc_host_staging, workspace->weighted,
                weighted_bytes, cudaMemcpyDeviceToHost, state.stream);
            status != cudaSuccess) {
            return cuda_error(
                status, "download DeepSeek device mHC begin weighted state");
        }
    }
    if (auto status = cudaMemcpyAsync(
            state.dsv4_mhc_host_staging + weighted_bytes,
            workspace->layer_input, layer_bytes, cudaMemcpyDeviceToHost,
            state.stream); status != cudaSuccess) {
        return cuda_error(
            status, "download DeepSeek device mHC begin layer input");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                state.activation_downloaded, state.stream);
            status != cudaSuccess) {
            return cuda_error(
                status, "record DeepSeek device mHC begin download");
        }
    }
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaStreamSynchronize(state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "synchronize DeepSeek device mHC begin");
    }
    const auto wait_nanoseconds = elapsed_nanoseconds_since(wait_started);
    const auto operation_nanoseconds = elapsed_nanoseconds_since(
        operation_started);
    const auto* host_output = reinterpret_cast<const std::uint16_t*>(
        state.dsv4_mhc_host_staging);
    const auto decode = [&](std::span<float> target,
                            std::size_t offset) {
        for (std::size_t index = 0U; index < target.size(); ++index) {
            target[index] = std::bit_cast<float>(
                static_cast<std::uint32_t>(host_output[offset + index]) << 16U);
            if (!std::isfinite(target[index])) return false;
        }
        return true;
    };
    if (!decode(weighted, 0U) || !decode(layer_input, weighted.size())) {
        result.errors.emplace_back(
            "DeepSeek device mHC begin produced a non-finite value");
        return result;
    }
    std::uint64_t h2d_nanoseconds = 0U;
    std::uint64_t kernel_nanoseconds = 0U;
    std::uint64_t d2h_nanoseconds = 0U;
    std::uint64_t timing_clamped_samples = 0U;
    if (impl_->detailed_timing) {
        float h2d_ms = 0.0F;
        float kernel_ms = 0.0F;
        float d2h_ms = 0.0F;
        if (cudaEventElapsedTime(&h2d_ms, state.activation_start,
                                 state.activation_uploaded) != cudaSuccess ||
            cudaEventElapsedTime(&kernel_ms, state.activation_uploaded,
                                 state.kernel_finished) != cudaSuccess ||
            cudaEventElapsedTime(&d2h_ms, state.kernel_finished,
                                 state.activation_downloaded) != cudaSuccess) {
            result.errors.emplace_back(
                "measure DeepSeek device mHC begin failed");
            return result;
        }
        h2d_nanoseconds = event_milliseconds_to_nanoseconds(
            h2d_ms, timing_clamped_samples);
        kernel_nanoseconds = event_milliseconds_to_nanoseconds(
            kernel_ms, timing_clamped_samples);
        d2h_nanoseconds = event_milliseconds_to_nanoseconds(
            d2h_ms, timing_clamped_samples);
    }
    state.dsv4_mhc_stage = 1U;
    state.dsv4_mhc_residual_index = 0U;
    state.dsv4_mhc_branch_ready = false;
    state.dsv4_mhc_failed = false;
    const auto call_nanoseconds = elapsed_nanoseconds_since(operation_started);
    {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        ++stats.dsv4_mhc_calls;
        ++stats.dsv4_mhc_standalone_calls;
        stats.dsv4_mhc_kernel_launches += 4U;
        stats.dsv4_mhc_h2d_bytes += h2d_bytes;
        stats.dsv4_mhc_d2h_bytes += d2h_bytes;
        stats.dsv4_mhc_h2d_nanoseconds += h2d_nanoseconds;
        stats.dsv4_mhc_kernel_nanoseconds += kernel_nanoseconds;
        stats.dsv4_mhc_d2h_nanoseconds += d2h_nanoseconds;
        stats.dsv4_mhc_nanoseconds += operation_nanoseconds;
        stats.dsv4_mhc_device_nanoseconds +=
            h2d_nanoseconds + kernel_nanoseconds + d2h_nanoseconds;
        stats.dsv4_mhc_host_nanoseconds +=
            call_nanoseconds > wait_nanoseconds
                ? call_nanoseconds - wait_nanoseconds : 0U;
        stats.dsv4_mhc_timing_clamped_samples += timing_clamped_samples;
        stats.activation_h2d_bytes += h2d_bytes;
        stats.activation_d2h_bytes += d2h_bytes;
        stats.activation_h2d_nanoseconds += h2d_nanoseconds;
        stats.kernel_nanoseconds += kernel_nanoseconds;
        stats.activation_d2h_nanoseconds += d2h_nanoseconds;
        stats.workspace_allocation_calls += allocation_calls;
        stats.workspace_allocation_bytes += allocation_bytes;
        record_synchronization(stats, SynchronizationSubsystem::Mhc,
                               1U, wait_nanoseconds);
    }
    return result;
}

ValidationResult CudaBackend::dsv4_mhc_transition(
    int device, const CudaDsv4MhcWeights& next_weights,
    std::span<const float> branch_output, std::span<float> weighted,
    std::span<float> layer_input, std::span<float> post_residual) {
    return dsv4_mhc_transition_impl(
        device, next_weights, branch_output, weighted, layer_input,
        post_residual, false);
}

ValidationResult CudaBackend::dsv4_mhc_transition_device(
    int device, const CudaDsv4MhcWeights& next_weights,
    std::span<float> weighted, std::span<float> layer_input,
    std::span<float> post_residual) {
    return dsv4_mhc_transition_impl(
        device, next_weights, {}, weighted, layer_input, post_residual, true);
}

ValidationResult CudaBackend::dsv4_mhc_transition_impl(
    int device, const CudaDsv4MhcWeights& next_weights,
    std::span<const float> branch_output, std::span<float> weighted,
    std::span<float> layer_input, std::span<float> post_residual,
    bool device_branch) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek device mHC transition targets an uninitialized device");
        return result;
    }
    auto& state = found->second;
    if (!next_weights.valid() || next_weights.device() != device ||
        !state.dsv4_mhc_supported || state.moe_in_flight ||
        state.dsv4_mhc_stage != 1U ||
        state.dsv4_mhc_branch_ready != device_branch ||
        (device_branch ? !branch_output.empty()
                       : branch_output.size() != kDsv4MhcHidden) ||
        (!weighted.empty() && weighted.size() != kDsv4MhcHidden) ||
        layer_input.size() != kDsv4MhcHidden ||
        (!post_residual.empty() &&
         post_residual.size() != kDsv4MhcMultiplier * kDsv4MhcHidden) ||
        (!device_branch &&
         !std::all_of(branch_output.begin(), branch_output.end(),
                      [](float value) { return std::isfinite(value); }))) {
        result.errors.emplace_back(
            "DeepSeek device mHC transition request or command order is invalid");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(
            status, "select CUDA device for device mHC transition");
    }
    state.dsv4_mhc_branch_ready = false;
    auto* host_branch = reinterpret_cast<std::uint16_t*>(
        state.dsv4_mhc_host_staging);
    if (!device_branch) {
        for (std::size_t index = 0U; index < branch_output.size(); ++index) {
            host_branch[index] = bf16_encode(branch_output[index]);
        }
    }
    auto* workspace = state.dsv4_mhc_workspace;
    const auto current = state.dsv4_mhc_residual_index;
    const auto next = current ^ 1U;
    const auto* projection = static_cast<const float*>(
        next_weights.impl_->projection.impl_->weights);
    const auto* auxiliary = static_cast<const std::byte*>(
        next_weights.impl_->auxiliary.impl_->data);
    const auto* scale = reinterpret_cast<const float*>(auxiliary);
    const auto* base = reinterpret_cast<const float*>(
        auxiliary + 3U * sizeof(float));
    const auto* norm = reinterpret_cast<const __nv_bfloat16*>(
        auxiliary + kDsv4MhcAuxNormOffset);
    const auto h2d_bytes = device_branch
        ? 0U : branch_output.size() * sizeof(std::uint16_t);
    const auto output_elements = weighted.size() + layer_input.size() +
                                 post_residual.size();
    const auto d2h_bytes = output_elements * sizeof(std::uint16_t);
    const auto operation_started = std::chrono::steady_clock::now();
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_start, state.stream);
            status != cudaSuccess) {
            return cuda_error(
                status, "record DeepSeek device mHC transition upload start");
        }
    }
    if (!device_branch) {
        if (auto status = cudaMemcpyAsync(
                workspace->branch, host_branch,
                static_cast<std::size_t>(h2d_bytes), cudaMemcpyHostToDevice,
                state.stream); status != cudaSuccess) {
            return cuda_error(
                status, "upload DeepSeek device mHC branch output");
        }
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                state.activation_uploaded, state.stream);
            status != cudaSuccess) {
            return cuda_error(
                status, "record DeepSeek device mHC transition upload");
        }
    }
    dsv4_mhc_fused_post_projection<<<
        dim3{kDsv4MhcMixes / kDsv4MhcProjectionTile, kDsv4MhcSplits},
        kDsv4MhcProjectionThreads, 0U, state.stream>>>(
        workspace->combination, workspace->residual[current],
        workspace->post, workspace->branch, projection,
        workspace->partial_projection, workspace->partial_square_sum,
        workspace->residual[next]);
    dsv4_mhc_mix<<<1U, 32U, 0U, state.stream>>>(
        workspace->partial_projection, workspace->partial_square_sum,
        scale, base, kDsv4MhcSplits, workspace->pre, workspace->post,
        workspace->combination);
    dsv4_mhc_weighted_norm<<<1U, kDsv4MhcWeightedNormThreads, 0U,
                             state.stream>>>(
        workspace->residual[next], workspace->pre, norm,
        workspace->weighted, workspace->layer_input);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "launch DeepSeek device mHC transition");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                state.kernel_finished, state.stream);
            status != cudaSuccess) {
            state.dsv4_mhc_stage = 0U;
            return cuda_error(
                status, "record DeepSeek device mHC transition kernels");
        }
    }
    const auto weighted_bytes =
        weighted.size() * sizeof(std::uint16_t);
    const auto layer_bytes =
        layer_input.size() * sizeof(std::uint16_t);
    if (!weighted.empty()) {
        if (auto status = cudaMemcpyAsync(
                state.dsv4_mhc_host_staging, workspace->weighted,
                weighted_bytes, cudaMemcpyDeviceToHost, state.stream);
            status != cudaSuccess) {
            state.dsv4_mhc_stage = 0U;
            return cuda_error(
                status,
                "download DeepSeek device mHC transition weighted state");
        }
    }
    if (auto status = cudaMemcpyAsync(
            state.dsv4_mhc_host_staging + weighted_bytes,
            workspace->layer_input, layer_bytes, cudaMemcpyDeviceToHost,
            state.stream); status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(
            status, "download DeepSeek device mHC transition layer input");
    }
    if (!post_residual.empty()) {
        if (auto status = cudaMemcpyAsync(
                state.dsv4_mhc_host_staging + weighted_bytes + layer_bytes,
                workspace->residual[next],
                post_residual.size() * sizeof(std::uint16_t),
                cudaMemcpyDeviceToHost, state.stream);
            status != cudaSuccess) {
            state.dsv4_mhc_stage = 0U;
            return cuda_error(
                status, "download DeepSeek device mHC transition residual");
        }
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                state.activation_downloaded, state.stream);
            status != cudaSuccess) {
            state.dsv4_mhc_stage = 0U;
            return cuda_error(
                status, "record DeepSeek device mHC transition download");
        }
    }
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaStreamSynchronize(state.stream);
        status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(
            status, "synchronize DeepSeek device mHC transition");
    }
    const auto wait_nanoseconds = elapsed_nanoseconds_since(wait_started);
    const auto operation_nanoseconds = elapsed_nanoseconds_since(
        operation_started);
    const auto* host_output = reinterpret_cast<const std::uint16_t*>(
        state.dsv4_mhc_host_staging);
    const auto decode = [&](std::span<float> target,
                            std::size_t offset) {
        for (std::size_t index = 0U; index < target.size(); ++index) {
            target[index] = std::bit_cast<float>(
                static_cast<std::uint32_t>(host_output[offset + index]) << 16U);
            if (!std::isfinite(target[index])) return false;
        }
        return true;
    };
    if (!decode(weighted, 0U) ||
        !decode(layer_input, weighted.size()) ||
        (!post_residual.empty() &&
         !decode(post_residual, weighted.size() + layer_input.size()))) {
        state.dsv4_mhc_stage = 0U;
        result.errors.emplace_back(
            "DeepSeek device mHC transition produced a non-finite value");
        return result;
    }
    state.dsv4_mhc_residual_index = next;
    std::uint64_t h2d_nanoseconds = 0U;
    std::uint64_t kernel_nanoseconds = 0U;
    std::uint64_t d2h_nanoseconds = 0U;
    std::uint64_t timing_clamped_samples = 0U;
    if (impl_->detailed_timing) {
        float h2d_ms = 0.0F;
        float kernel_ms = 0.0F;
        float d2h_ms = 0.0F;
        if (cudaEventElapsedTime(&h2d_ms, state.activation_start,
                                 state.activation_uploaded) != cudaSuccess ||
            cudaEventElapsedTime(&kernel_ms, state.activation_uploaded,
                                 state.kernel_finished) != cudaSuccess ||
            cudaEventElapsedTime(&d2h_ms, state.kernel_finished,
                                 state.activation_downloaded) != cudaSuccess) {
            state.dsv4_mhc_stage = 0U;
            result.errors.emplace_back(
                "measure DeepSeek device mHC transition failed");
            return result;
        }
        h2d_nanoseconds = event_milliseconds_to_nanoseconds(
            h2d_ms, timing_clamped_samples);
        kernel_nanoseconds = event_milliseconds_to_nanoseconds(
            kernel_ms, timing_clamped_samples);
        d2h_nanoseconds = event_milliseconds_to_nanoseconds(
            d2h_ms, timing_clamped_samples);
    }
    const auto call_nanoseconds = elapsed_nanoseconds_since(operation_started);
    {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        ++stats.dsv4_mhc_calls;
        ++stats.dsv4_mhc_transition_calls;
        stats.dsv4_mhc_kernel_launches += 3U;
        stats.dsv4_mhc_h2d_bytes += h2d_bytes;
        stats.dsv4_mhc_d2h_bytes += d2h_bytes;
        stats.dsv4_mhc_h2d_nanoseconds += h2d_nanoseconds;
        stats.dsv4_mhc_kernel_nanoseconds += kernel_nanoseconds;
        stats.dsv4_mhc_d2h_nanoseconds += d2h_nanoseconds;
        stats.dsv4_mhc_nanoseconds += operation_nanoseconds;
        stats.dsv4_mhc_device_nanoseconds +=
            h2d_nanoseconds + kernel_nanoseconds + d2h_nanoseconds;
        stats.dsv4_mhc_host_nanoseconds +=
            call_nanoseconds > wait_nanoseconds
                ? call_nanoseconds - wait_nanoseconds : 0U;
        stats.dsv4_mhc_timing_clamped_samples += timing_clamped_samples;
        stats.activation_h2d_bytes += h2d_bytes;
        stats.activation_d2h_bytes += d2h_bytes;
        stats.activation_h2d_nanoseconds += h2d_nanoseconds;
        stats.kernel_nanoseconds += kernel_nanoseconds;
        stats.activation_d2h_nanoseconds += d2h_nanoseconds;
        record_synchronization(stats, SynchronizationSubsystem::Mhc,
                               1U, wait_nanoseconds);
    }
    return result;
}

ValidationResult CudaBackend::dsv4_mhc_finish(
    int device, std::span<const float> branch_output,
    std::span<float> hidden) {
    return dsv4_mhc_finish_impl(device, branch_output, hidden, false);
}

ValidationResult CudaBackend::dsv4_mhc_finish_device(
    int device, std::span<float> hidden) {
    return dsv4_mhc_finish_impl(device, {}, hidden, true);
}

ValidationResult CudaBackend::dsv4_mhc_device_view(
    int device, CudaDsv4MhcDeviceView& view) {
    view = {};
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        return {{"DeepSeek device mHC view targets an uninitialized device"}};
    }
    auto& state = found->second;
    if (!state.dsv4_mhc_supported || state.dsv4_mhc_stage != 1U ||
        state.dsv4_mhc_workspace == nullptr || state.moe_in_flight ||
        state.dsv4_mhc_failed) {
        return {{"DeepSeek device mHC view violates command order"}};
    }
    view.stream = state.stream;
    view.weighted = reinterpret_cast<std::uint16_t*>(
        state.dsv4_mhc_workspace->weighted);
    view.layer_input = reinterpret_cast<std::uint16_t*>(
        state.dsv4_mhc_workspace->layer_input);
    view.branch = reinterpret_cast<std::uint16_t*>(
        state.dsv4_mhc_workspace->branch);
    view.residual = reinterpret_cast<std::uint16_t*>(
        state.dsv4_mhc_workspace->residual[state.dsv4_mhc_residual_index]);
    view.router_logits = state.dsv4_mhc_workspace->router_logits;
    view.status = &state.dsv4_mhc_workspace->failure;
    if (view.stream == nullptr || view.weighted == nullptr ||
        view.layer_input == nullptr ||
        view.branch == nullptr || view.residual == nullptr ||
        view.router_logits == nullptr ||
        view.status == nullptr) {
        view = {};
        return {{"DeepSeek device mHC view is incomplete"}};
    }
    return {};
}

namespace {

bool dsv4_validate_device_pointer(
    int device, const void* pointer, const char* name,
    ValidationResult& result) {
    if (pointer == nullptr) {
        result.errors.emplace_back(std::string(name) + " is null");
        return false;
    }
    cudaPointerAttributes attributes{};
    if (const auto status = cudaPointerGetAttributes(&attributes, pointer);
        status != cudaSuccess) {
        result.errors.emplace_back(std::string(name) +
                                   " is not a live CUDA pointer");
        static_cast<void>(cudaGetLastError());
        return false;
    }
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device) {
        result.errors.emplace_back(std::string(name) +
                                   " is not resident on the requested device");
        return false;
    }
    return true;
}

}  // namespace

ValidationResult CudaBackend::dsv4_mhc_branch_to_fp32(
    int device, float* output) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek mHC branch conversion targets an uninitialized device");
        return result;
    }
    auto& state = found->second;
    if (!state.dsv4_mhc_supported || state.dsv4_mhc_stage != 1U ||
        state.dsv4_mhc_workspace == nullptr || state.dsv4_mhc_branch_ready ||
        state.moe_in_flight || state.dsv4_mhc_failed ||
        !dsv4_validate_device_pointer(device, output, "mHC FP32 branch output",
                                      result)) {
        if (result.errors.empty()) {
            result.errors.emplace_back(
                "DeepSeek mHC branch conversion violates command order");
        }
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for mHC branch conversion");
    }
    constexpr std::uint32_t threads = 256U;
    constexpr std::uint32_t blocks =
        (kDsv4MhcHidden + threads - 1U) / threads;
    dsv4_bf16_to_fp32<<<blocks, threads, 0U, state.stream>>>(
        state.dsv4_mhc_workspace->branch, output, kDsv4MhcHidden);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch mHC branch conversion");
    }
    return result;
}

ValidationResult CudaBackend::dsv4_mhc_commit_reduced_branch(
    int device, const float* reduced) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek mHC branch commit targets an uninitialized device");
        return result;
    }
    auto& state = found->second;
    if (!state.dsv4_mhc_supported || state.dsv4_mhc_stage != 1U ||
        state.dsv4_mhc_workspace == nullptr || state.dsv4_mhc_branch_ready ||
        state.dsv4_mhc_failed || state.dsv4_host_moe_input_pending ||
        !dsv4_validate_device_pointer(device, reduced, "reduced mHC branch",
                                      result)) {
        if (result.errors.empty()) {
            result.errors.emplace_back(
                "DeepSeek mHC branch commit violates command order");
        }
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for mHC branch commit");
    }
    constexpr std::uint32_t threads = 256U;
    constexpr std::uint32_t blocks =
        (kDsv4MhcHidden + threads - 1U) / threads;
    dsv4_fp32_to_bf16<<<blocks, threads, 0U, state.stream>>>(
        reduced, state.dsv4_mhc_workspace->branch, kDsv4MhcHidden);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch mHC reduced branch publication");
    }
    state.dsv4_mhc_branch_ready = true;
    return result;
}

ValidationResult CudaBackend::dsv4_mhc_abort_branch(int device) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek mHC branch abort targets an uninitialized device");
        return result;
    }
    auto& state = found->second;
    if (!state.dsv4_mhc_supported || state.dsv4_mhc_stage != 1U ||
        state.dsv4_mhc_workspace == nullptr || state.moe_in_flight) {
        result.errors.emplace_back(
            "DeepSeek mHC branch abort violates command order");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for mHC branch abort");
    }
    auto* workspace = state.dsv4_mhc_workspace;
    if (auto status = cudaMemsetAsync(
            workspace->branch, 0, sizeof(workspace->branch), state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "clear failed mHC branch");
    }
    if (auto status = cudaMemsetAsync(
            workspace->weighted, 0, sizeof(workspace->weighted), state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "clear failed mHC weighted state");
    }
    if (auto status = cudaMemsetAsync(
            workspace->layer_input, 0, sizeof(workspace->layer_input),
            state.stream); status != cudaSuccess) {
        return cuda_error(status, "clear failed mHC layer input");
    }
    if (auto status = cudaMemsetAsync(
            workspace->residual, 0, sizeof(workspace->residual), state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "clear failed mHC residual state");
    }
    if (auto status = cudaMemsetAsync(
            &workspace->failure, 1, sizeof(workspace->failure), state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "mark failed mHC state");
    }
    state.dsv4_mhc_stage = 0U;
    state.dsv4_mhc_residual_index = 0U;
    state.dsv4_mhc_branch_ready = false;
    state.dsv4_mhc_failed = true;
    state.dsv4_host_moe_input_pending = false;
    state.dsv4_host_moe_router_logits = nullptr;
    state.dsv4_host_moe_device_failure = nullptr;
    state.dsv4_host_moe_host_failure = nullptr;
    return result;
}

ValidationResult CudaBackend::dsv4_mhc_transition_router_device(
    int device, const CudaDsv4MhcWeights& next_weights,
    const CudaWeight& router) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek mHC router transition targets an uninitialized device");
        return result;
    }
    auto& state = found->second;
    if (!next_weights.valid() || next_weights.device() != device ||
        !router.valid() || router.device() != device ||
        !state.dsv4_mhc_supported || state.dsv4_mhc_stage != 1U ||
        state.dsv4_mhc_workspace == nullptr || !state.dsv4_mhc_branch_ready ||
        state.dsv4_host_moe_input_pending || state.moe_in_flight ||
        state.dsv4_mhc_failed) {
        result.errors.emplace_back(
            "DeepSeek mHC router transition violates command order or ownership");
        return result;
    }
    const auto& descriptor = router.impl_->descriptor;
    if (descriptor.encoding != CudaWeightEncoding::Plain ||
        descriptor.dtype != SafetensorsDtype::Bf16 ||
        descriptor.rows != kDsv4MhcRouterLogits ||
        descriptor.columns != kDsv4MhcHidden) {
        result.errors.emplace_back(
            "DeepSeek mHC router transition requires a 256x4096 BF16 router");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for mHC router transition");
    }
    auto* workspace = state.dsv4_mhc_workspace;
    const auto current = state.dsv4_mhc_residual_index;
    const auto next = current ^ 1U;
    const auto* projection = static_cast<const float*>(
        next_weights.impl_->projection.impl_->weights);
    const auto* auxiliary = static_cast<const std::byte*>(
        next_weights.impl_->auxiliary.impl_->data);
    const auto* scale = reinterpret_cast<const float*>(auxiliary);
    const auto* base = reinterpret_cast<const float*>(
        auxiliary + 3U * sizeof(float));
    const auto* norm = reinterpret_cast<const __nv_bfloat16*>(
        auxiliary + kDsv4MhcAuxNormOffset);
    dsv4_mhc_fused_post_projection<<<
        dim3{kDsv4MhcMixes / kDsv4MhcProjectionTile, kDsv4MhcSplits},
        kDsv4MhcProjectionThreads, 0U, state.stream>>>(
        workspace->combination, workspace->residual[current], workspace->post,
        workspace->branch, projection, workspace->partial_projection,
        workspace->partial_square_sum, workspace->residual[next]);
    dsv4_mhc_mix<<<1U, 32U, 0U, state.stream>>>(
        workspace->partial_projection, workspace->partial_square_sum, scale,
        base, kDsv4MhcSplits, workspace->pre, workspace->post,
        workspace->combination);
    dsv4_mhc_weighted_norm<<<1U, kDsv4MhcWeightedNormThreads, 0U,
                             state.stream>>>(
        workspace->residual[next], workspace->pre, norm, workspace->weighted,
        workspace->layer_input);
    constexpr unsigned int threads = 256U;
    const auto blocks = static_cast<unsigned int>(
        (kDsv4MhcRouterLogits + (threads / 32U) - 1U) / (threads / 32U));
    bf16_input_matvec_kernel<<<blocks, threads, 0U, state.stream>>>(
        workspace->router_logits, workspace->layer_input,
        static_cast<const __nv_bfloat16*>(router.impl_->weights),
        descriptor.columns, descriptor.rows);
    if (auto status = cudaMemsetAsync(
            workspace->branch, 0, sizeof(workspace->branch), state.stream);
        status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "clear consumed mHC attention branch");
    }
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "launch mHC router transition");
    }
    state.dsv4_mhc_residual_index = next;
    state.dsv4_mhc_branch_ready = false;
    state.dsv4_host_moe_input_pending = true;
    state.dsv4_host_moe_router_logits = workspace->router_logits;
    state.dsv4_host_moe_device_failure = &workspace->failure;
    state.dsv4_host_moe_host_failure = nullptr;
    return result;
}

ValidationResult CudaBackend::dsv4_mhc_transition_next_device(
    int device, const CudaDsv4MhcWeights& next_weights) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek final mHC transition targets an uninitialized device");
        return result;
    }
    auto& state = found->second;
    if (!next_weights.valid() || next_weights.device() != device ||
        !state.dsv4_mhc_supported || state.dsv4_mhc_stage != 1U ||
        state.dsv4_mhc_workspace == nullptr || !state.dsv4_mhc_branch_ready ||
        state.dsv4_host_moe_input_pending || state.moe_in_flight ||
        state.dsv4_mhc_failed) {
        result.errors.emplace_back(
            "DeepSeek final mHC transition violates command order");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for final mHC transition");
    }
    auto* workspace = state.dsv4_mhc_workspace;
    const auto current = state.dsv4_mhc_residual_index;
    const auto next = current ^ 1U;
    const auto* projection = static_cast<const float*>(
        next_weights.impl_->projection.impl_->weights);
    const auto* auxiliary = static_cast<const std::byte*>(
        next_weights.impl_->auxiliary.impl_->data);
    const auto* scale = reinterpret_cast<const float*>(auxiliary);
    const auto* base = reinterpret_cast<const float*>(
        auxiliary + 3U * sizeof(float));
    const auto* norm = reinterpret_cast<const __nv_bfloat16*>(
        auxiliary + kDsv4MhcAuxNormOffset);
    dsv4_mhc_fused_post_projection<<<
        dim3{kDsv4MhcMixes / kDsv4MhcProjectionTile, kDsv4MhcSplits},
        kDsv4MhcProjectionThreads, 0U, state.stream>>>(
        workspace->combination, workspace->residual[current], workspace->post,
        workspace->branch, projection, workspace->partial_projection,
        workspace->partial_square_sum, workspace->residual[next]);
    dsv4_mhc_mix<<<1U, 32U, 0U, state.stream>>>(
        workspace->partial_projection, workspace->partial_square_sum, scale,
        base, kDsv4MhcSplits, workspace->pre, workspace->post,
        workspace->combination);
    dsv4_mhc_weighted_norm<<<1U, kDsv4MhcWeightedNormThreads, 0U,
                             state.stream>>>(
        workspace->residual[next], workspace->pre, norm, workspace->weighted,
        workspace->layer_input);
    if (auto status = cudaMemsetAsync(
            workspace->branch, 0, sizeof(workspace->branch), state.stream);
        status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "clear consumed mHC MoE branch");
    }
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "launch final mHC transition");
    }
    state.dsv4_mhc_residual_index = next;
    state.dsv4_mhc_branch_ready = false;
    return result;
}

ValidationResult CudaBackend::dsv4_mhc_finish_impl(
    int device, std::span<const float> branch_output,
    std::span<float> hidden, bool device_branch) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek device mHC finish targets an uninitialized device");
        return result;
    }
    auto& state = found->second;
    if (!state.dsv4_mhc_supported || state.moe_in_flight ||
        state.dsv4_mhc_stage != 1U ||
        state.dsv4_mhc_branch_ready != device_branch ||
        (device_branch ? !branch_output.empty()
                       : branch_output.size() != kDsv4MhcHidden) ||
        hidden.size() != kDsv4MhcMultiplier * kDsv4MhcHidden ||
        (!device_branch &&
         !std::all_of(branch_output.begin(), branch_output.end(),
                      [](float value) { return std::isfinite(value); }))) {
        result.errors.emplace_back(
            "DeepSeek device mHC finish request or command order is invalid");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(
            status, "select CUDA device for device mHC finish");
    }
    state.dsv4_mhc_branch_ready = false;
    auto* host_branch = reinterpret_cast<std::uint16_t*>(
        state.dsv4_mhc_host_staging);
    if (!device_branch) {
        for (std::size_t index = 0U; index < branch_output.size(); ++index) {
            host_branch[index] = bf16_encode(branch_output[index]);
        }
    }
    auto* workspace = state.dsv4_mhc_workspace;
    const auto current = state.dsv4_mhc_residual_index;
    const auto next = current ^ 1U;
    const auto h2d_bytes = device_branch
        ? 0U : branch_output.size() * sizeof(std::uint16_t);
    const auto d2h_bytes = hidden.size() * sizeof(std::uint16_t);
    const auto operation_started = std::chrono::steady_clock::now();
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_start, state.stream);
            status != cudaSuccess) {
            return cuda_error(
                status, "record DeepSeek device mHC finish upload start");
        }
    }
    if (!device_branch) {
        if (auto status = cudaMemcpyAsync(
                workspace->branch, host_branch,
                static_cast<std::size_t>(h2d_bytes), cudaMemcpyHostToDevice,
                state.stream); status != cudaSuccess) {
            return cuda_error(
                status, "upload DeepSeek device mHC final branch output");
        }
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                state.activation_uploaded, state.stream);
            status != cudaSuccess) {
            return cuda_error(
                status, "record DeepSeek device mHC finish upload");
        }
    }
    constexpr std::uint32_t threads = 256U;
    constexpr std::uint32_t blocks =
        (kDsv4MhcHidden + threads - 1U) / threads;
    dsv4_mhc_final_post<<<blocks, threads, 0U, state.stream>>>(
        workspace->combination, workspace->residual[current],
        workspace->post, workspace->branch, workspace->residual[next]);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "launch DeepSeek device mHC finish");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                state.kernel_finished, state.stream);
            status != cudaSuccess) {
            state.dsv4_mhc_stage = 0U;
            return cuda_error(
                status, "record DeepSeek device mHC finish kernel");
        }
    }
    if (auto status = cudaMemcpyAsync(
            state.dsv4_mhc_host_staging, workspace->residual[next],
            static_cast<std::size_t>(d2h_bytes), cudaMemcpyDeviceToHost,
            state.stream); status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(
            status, "download DeepSeek device mHC final residual");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                state.activation_downloaded, state.stream);
            status != cudaSuccess) {
            state.dsv4_mhc_stage = 0U;
            return cuda_error(
                status, "record DeepSeek device mHC finish download");
        }
    }
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaStreamSynchronize(state.stream);
        status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(
            status, "synchronize DeepSeek device mHC finish");
    }
    const auto wait_nanoseconds = elapsed_nanoseconds_since(wait_started);
    const auto operation_nanoseconds = elapsed_nanoseconds_since(
        operation_started);
    const auto* host_output = reinterpret_cast<const std::uint16_t*>(
        state.dsv4_mhc_host_staging);
    for (std::size_t index = 0U; index < hidden.size(); ++index) {
        hidden[index] = std::bit_cast<float>(
            static_cast<std::uint32_t>(host_output[index]) << 16U);
        if (!std::isfinite(hidden[index])) {
            state.dsv4_mhc_stage = 0U;
            result.errors.emplace_back(
                "DeepSeek device mHC finish produced a non-finite value");
            return result;
        }
    }
    state.dsv4_mhc_stage = 0U;
    state.dsv4_mhc_residual_index = next;
    std::uint64_t h2d_nanoseconds = 0U;
    std::uint64_t kernel_nanoseconds = 0U;
    std::uint64_t d2h_nanoseconds = 0U;
    std::uint64_t timing_clamped_samples = 0U;
    if (impl_->detailed_timing) {
        float h2d_ms = 0.0F;
        float kernel_ms = 0.0F;
        float d2h_ms = 0.0F;
        if (cudaEventElapsedTime(&h2d_ms, state.activation_start,
                                 state.activation_uploaded) != cudaSuccess ||
            cudaEventElapsedTime(&kernel_ms, state.activation_uploaded,
                                 state.kernel_finished) != cudaSuccess ||
            cudaEventElapsedTime(&d2h_ms, state.kernel_finished,
                                 state.activation_downloaded) != cudaSuccess) {
            result.errors.emplace_back(
                "measure DeepSeek device mHC finish failed");
            return result;
        }
        h2d_nanoseconds = event_milliseconds_to_nanoseconds(
            h2d_ms, timing_clamped_samples);
        kernel_nanoseconds = event_milliseconds_to_nanoseconds(
            kernel_ms, timing_clamped_samples);
        d2h_nanoseconds = event_milliseconds_to_nanoseconds(
            d2h_ms, timing_clamped_samples);
    }
    const auto call_nanoseconds = elapsed_nanoseconds_since(operation_started);
    {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        ++stats.dsv4_mhc_calls;
        ++stats.dsv4_mhc_final_calls;
        ++stats.dsv4_mhc_kernel_launches;
        stats.dsv4_mhc_h2d_bytes += h2d_bytes;
        stats.dsv4_mhc_d2h_bytes += d2h_bytes;
        stats.dsv4_mhc_h2d_nanoseconds += h2d_nanoseconds;
        stats.dsv4_mhc_kernel_nanoseconds += kernel_nanoseconds;
        stats.dsv4_mhc_d2h_nanoseconds += d2h_nanoseconds;
        stats.dsv4_mhc_nanoseconds += operation_nanoseconds;
        stats.dsv4_mhc_device_nanoseconds +=
            h2d_nanoseconds + kernel_nanoseconds + d2h_nanoseconds;
        stats.dsv4_mhc_host_nanoseconds +=
            call_nanoseconds > wait_nanoseconds
                ? call_nanoseconds - wait_nanoseconds : 0U;
        stats.dsv4_mhc_timing_clamped_samples += timing_clamped_samples;
        stats.activation_h2d_bytes += h2d_bytes;
        stats.activation_d2h_bytes += d2h_bytes;
        stats.activation_h2d_nanoseconds += h2d_nanoseconds;
        stats.kernel_nanoseconds += kernel_nanoseconds;
        stats.activation_d2h_nanoseconds += d2h_nanoseconds;
        record_synchronization(stats, SynchronizationSubsystem::Mhc,
                               1U, wait_nanoseconds);
    }
    return result;
}

ValidationResult CudaBackend::reserve_dsv4_mhc_head(
    int device, std::uint64_t logits) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end() || logits == 0U) {
        result.errors.emplace_back(
            "DeepSeek device output-head reservation is invalid");
        return result;
    }
    auto& state = found->second;
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for output-head reserve");
    }
    constexpr std::uint64_t hidden_bytes =
        kDsv4MhcMultiplier * kDsv4MhcHidden * sizeof(std::uint16_t);
    constexpr std::uint64_t input_bytes =
        kDsv4MhcHidden * sizeof(float);
    const auto output_bytes = logits * sizeof(float);
    const auto host_bytes = hidden_bytes + input_bytes + output_bytes;
    if (state.dsv4_mhc_head_input == nullptr) {
        if (auto status = cudaMalloc(
                &state.dsv4_mhc_head_input,
                static_cast<std::size_t>(input_bytes));
            status != cudaSuccess) {
            return cuda_error(status, "allocate DeepSeek output-head input");
        }
        state.dsv4_mhc_head_input_bytes = input_bytes;
    }
    // A reservation is a capacity, not a shape. One device can serve two head
    // shapes: centralized prefill projects the full vocabulary while rank-local
    // decode projects one rank's row shard, and under the rank-local opt-in
    // both are resident on the same device. Growing is safe because the
    // staging layout is offset-addressed from a fixed prefix; shrinking is a
    // no-op that keeps the larger buffer. What must still match exactly is the
    // enqueue against its own completion, which is pinned separately below.
    if (state.dsv4_mhc_head_in_flight) {
        result.errors.emplace_back(
            "DeepSeek output-head reservation cannot change while a head is "
            "in flight");
        return result;
    }
    if (state.dsv4_mhc_head_output_bytes < output_bytes) {
        if (state.dsv4_mhc_head_output != nullptr) {
            static_cast<void>(cudaFree(state.dsv4_mhc_head_output));
            state.dsv4_mhc_head_output = nullptr;
            state.dsv4_mhc_head_output_bytes = 0U;
        }
        if (auto status = cudaMalloc(
                &state.dsv4_mhc_head_output,
                static_cast<std::size_t>(output_bytes));
            status != cudaSuccess) {
            return cuda_error(status, "allocate DeepSeek output-head output");
        }
        state.dsv4_mhc_head_output_bytes = output_bytes;
    }
    if (state.dsv4_mhc_head_host_staging_bytes < host_bytes) {
        if (state.dsv4_mhc_head_host_staging != nullptr) {
            static_cast<void>(
                cudaFreeHost(state.dsv4_mhc_head_host_staging));
            state.dsv4_mhc_head_host_staging = nullptr;
            state.dsv4_mhc_head_host_staging_bytes = 0U;
        }
        void* staging = nullptr;
        if (auto status = cudaMallocHost(
                &staging, static_cast<std::size_t>(host_bytes));
            status != cudaSuccess) {
            return cuda_error(
                status, "allocate pinned DeepSeek output-head staging");
        }
        state.dsv4_mhc_head_host_staging = static_cast<std::byte*>(staging);
        state.dsv4_mhc_head_host_staging_bytes = host_bytes;
    }
    if (state.dsv4_mhc_head_input_bytes != input_bytes ||
        state.dsv4_mhc_head_output_bytes < output_bytes ||
        state.dsv4_mhc_head_host_staging_bytes < host_bytes) {
        result.errors.emplace_back(
            "DeepSeek output-head reservation is smaller than the head it "
            "must serve");
        return result;
    }
    return result;
}

ValidationResult CudaBackend::enqueue_dsv4_mhc_finish_head_device(
    int device, const CudaWeight& head,
    CudaDsv4MhcHeadCallback callback, void* callback_context,
    CudaDsv4MhcHeadDeviceView* view) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end() || !head.valid() ||
        head.impl_->device != device) {
        result.errors.emplace_back(
            "DeepSeek device output-head enqueue is invalid");
        return result;
    }
    auto& state = found->second;
    const auto& descriptor = head.impl_->descriptor;
    constexpr std::uint64_t hidden_bytes =
        kDsv4MhcMultiplier * kDsv4MhcHidden * sizeof(std::uint16_t);
    constexpr std::uint64_t input_bytes =
        kDsv4MhcHidden * sizeof(float);
    const auto output_bytes = descriptor.rows * sizeof(float);
    if (!state.dsv4_mhc_supported || state.dsv4_mhc_stage != 1U ||
        !state.dsv4_mhc_branch_ready || state.dsv4_mhc_head_in_flight ||
        callback == nullptr || callback_context == nullptr ||
        descriptor.columns != kDsv4MhcHidden ||
        state.dsv4_mhc_head_input_bytes != input_bytes ||
        state.dsv4_mhc_head_output_bytes < output_bytes ||
        state.dsv4_mhc_head_host_staging_bytes <
            hidden_bytes + input_bytes + output_bytes) {
        result.errors.emplace_back(
            "DeepSeek device output-head command order or shape is invalid");
        return result;
    }
    // This head's own logit extent, so its completion cannot accept a span
    // sized for the other head shape resident on this device.
    state.dsv4_mhc_head_logits_bytes = output_bytes;
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for output-head enqueue");
    }
    auto* workspace = state.dsv4_mhc_workspace;
    const auto current = state.dsv4_mhc_residual_index;
    const auto next = current ^ 1U;
    constexpr std::uint32_t threads = 256U;
    constexpr std::uint32_t blocks =
        (kDsv4MhcHidden + threads - 1U) / threads;
    dsv4_mhc_final_post<<<blocks, threads, 0U, state.stream>>>(
        workspace->combination, workspace->residual[current],
        workspace->post, workspace->branch, workspace->residual[next]);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "launch DeepSeek output-head final mHC post");
    }
    auto* host_hidden = reinterpret_cast<std::uint16_t*>(
        state.dsv4_mhc_head_host_staging);
    auto* host_reduced = reinterpret_cast<float*>(
        state.dsv4_mhc_head_host_staging + hidden_bytes);
    auto* host_logits = reinterpret_cast<float*>(
        state.dsv4_mhc_head_host_staging + hidden_bytes + input_bytes);
    if (auto status = cudaMemcpyAsync(
            host_hidden, workspace->residual[next], hidden_bytes,
            cudaMemcpyDeviceToHost, state.stream); status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "stage DeepSeek final mHC residual");
    }
    state.dsv4_mhc_head_callback = {
        callback, callback_context, host_hidden, host_reduced, false};
    if (auto status = cudaLaunchHostFunc(
            state.stream, run_dsv4_mhc_head_callback,
            &state.dsv4_mhc_head_callback); status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "enqueue DeepSeek output-head host reduction");
    }
    if (auto status = cudaMemcpyAsync(
            state.dsv4_mhc_head_input, host_reduced, input_bytes,
            cudaMemcpyHostToDevice, state.stream); status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "upload DeepSeek output-head input");
    }
    const bool native =
        descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128 ||
        descriptor.encoding == CudaWeightEncoding::Fp4E2m1Group32;
    if (native) {
        const dim3 quantize_grid(
            static_cast<unsigned int>((descriptor.columns + 127U) / 128U),
            1U, 1U);
        quantize_activation_e4m3_kernel<<<
            quantize_grid, 128U, 0U, state.stream>>>(
            state.dsv4_mhc_head_input, descriptor.columns, 1U);
    }
    const dim3 grid(static_cast<unsigned int>(descriptor.rows), 1U, 1U);
    if (descriptor.encoding == CudaWeightEncoding::Plain &&
        descriptor.dtype == SafetensorsDtype::Bf16) {
        constexpr unsigned int warps_per_block = threads / 32U;
        const auto matvec_blocks = static_cast<unsigned int>(
            (descriptor.rows + warps_per_block - 1U) / warps_per_block);
        bf16_matvec_kernel<<<matvec_blocks, threads, 0U, state.stream>>>(
            state.dsv4_mhc_head_output, state.dsv4_mhc_head_input,
            static_cast<const __nv_bfloat16*>(head.impl_->weights),
            descriptor.columns, descriptor.rows);
    } else if (descriptor.encoding == CudaWeightEncoding::Plain) {
        plain_matmul_kernel<1U><<<grid, threads, 0U, state.stream>>>(
            state.dsv4_mhc_head_output, state.dsv4_mhc_head_input,
            head.impl_->weights, static_cast<int>(descriptor.dtype), 1U,
            descriptor.columns, descriptor.rows, 0U, 0U);
    } else if (descriptor.encoding == CudaWeightEncoding::OffsetPackedInt8 &&
               descriptor.group_size == 32U &&
               descriptor.columns % 32U == 0U) {
        constexpr unsigned int warps_per_block = threads / 32U;
        const auto matvec_blocks = static_cast<unsigned int>(
            (descriptor.rows + warps_per_block - 1U) / warps_per_block);
        packed_int8_group32_matvec_kernel<<<
            matvec_blocks, threads, 0U, state.stream>>>(
            state.dsv4_mhc_head_output, state.dsv4_mhc_head_input,
            static_cast<const std::uint32_t*>(head.impl_->weights),
            static_cast<const __nv_bfloat16*>(head.impl_->scales),
            descriptor.packed_columns, descriptor.scale_columns,
            descriptor.columns, descriptor.rows);
    } else if (descriptor.encoding == CudaWeightEncoding::OffsetPackedInt4 ||
               descriptor.encoding == CudaWeightEncoding::OffsetPackedInt8) {
        const auto bits = descriptor.encoding ==
            CudaWeightEncoding::OffsetPackedInt4 ? 4U : 8U;
        packed_matmul_kernel<<<grid, threads, 0U, state.stream>>>(
            state.dsv4_mhc_head_output, state.dsv4_mhc_head_input,
            static_cast<const std::uint32_t*>(head.impl_->weights),
            static_cast<const __nv_bfloat16*>(head.impl_->scales), bits,
            descriptor.group_size, descriptor.packed_columns,
            descriptor.scale_columns, 1U, descriptor.columns,
            descriptor.rows, 0U, 0U);
    } else if (descriptor.encoding == CudaWeightEncoding::Nvfp4Group16) {
        nvfp4_group16_matmul_kernel<<<grid, threads, 0U, state.stream>>>(
            state.dsv4_mhc_head_output, state.dsv4_mhc_head_input,
            static_cast<const unsigned char*>(head.impl_->weights),
            static_cast<const unsigned char*>(head.impl_->scales),
            descriptor.global_scale, descriptor.packed_columns,
            descriptor.scale_columns, descriptor.group_size, 1U,
            descriptor.columns, descriptor.rows, 0U, 0U);
    } else if (descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128) {
        native_fp8_matmul_kernel<<<grid, threads, 0U, state.stream>>>(
            state.dsv4_mhc_head_output, state.dsv4_mhc_head_input,
            static_cast<const unsigned char*>(head.impl_->weights),
            static_cast<const unsigned char*>(head.impl_->scales),
            descriptor.scale_columns, 1U, descriptor.columns,
            descriptor.rows, 0U, 0U);
    } else {
        native_fp4_matmul_kernel<<<grid, threads, 0U, state.stream>>>(
            state.dsv4_mhc_head_output, state.dsv4_mhc_head_input,
            static_cast<const unsigned char*>(head.impl_->weights),
            static_cast<const unsigned char*>(head.impl_->scales),
            descriptor.packed_columns, descriptor.scale_columns, 1U,
            descriptor.columns, descriptor.rows, 0U, 0U);
    }
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "launch DeepSeek output-head projection");
    }
    if (auto status = cudaMemcpyAsync(
            host_logits, state.dsv4_mhc_head_output, output_bytes,
            cudaMemcpyDeviceToHost, state.stream); status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "stage DeepSeek output-head logits");
    }
    state.dsv4_mhc_stage = 0U;
    state.dsv4_mhc_residual_index = next;
    state.dsv4_mhc_branch_ready = false;
    state.dsv4_mhc_head_in_flight = true;
    if (view != nullptr) {
        *view = {state.stream, state.dsv4_mhc_head_output,
                 reinterpret_cast<std::uint16_t*>(workspace->residual[next]),
                 descriptor.rows};
    }
    {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        ++stats.dsv4_mhc_calls;
        ++stats.dsv4_mhc_final_calls;
        ++stats.dsv4_mhc_kernel_launches;
        stats.dsv4_mhc_d2h_bytes += hidden_bytes;
        stats.activation_h2d_bytes += input_bytes;
        stats.activation_d2h_bytes += hidden_bytes + output_bytes;
        ++stats.matmul_calls;
    }
    return result;
}

ValidationResult CudaBackend::complete_dsv4_mhc_head_device(
    int device, std::span<float> logits) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek output-head completion targets an invalid device");
        return result;
    }
    auto& state = found->second;
    const auto output_bytes = logits.size_bytes();
    constexpr std::uint64_t hidden_bytes =
        kDsv4MhcMultiplier * kDsv4MhcHidden * sizeof(std::uint16_t);
    constexpr std::uint64_t input_bytes =
        kDsv4MhcHidden * sizeof(float);
    if (!state.dsv4_mhc_head_in_flight ||
        output_bytes != state.dsv4_mhc_head_logits_bytes) {
        result.errors.emplace_back(
            "DeepSeek output-head completion shape or order is invalid");
        return result;
    }
    state.dsv4_mhc_head_in_flight = false;
    if (state.dsv4_mhc_head_callback.failed) {
        result.errors.emplace_back(
            "DeepSeek output-head host reduction failed");
        return result;
    }
    const auto* host_logits = reinterpret_cast<const float*>(
        state.dsv4_mhc_head_host_staging + hidden_bytes + input_bytes);
    std::memcpy(logits.data(), host_logits, output_bytes);
    if (!std::all_of(logits.begin(), logits.end(), [](float value) {
            return std::isfinite(value);
        })) {
        result.errors.emplace_back(
            "DeepSeek output-head projection produced a non-finite value");
    }
    return result;
}
