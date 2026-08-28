ValidationResult CudaBackend::enqueue_deepseek_moe(
    int device, std::span<const float> hidden,
    std::span<const CudaDeepSeekMoeExpert> routed,
    const CudaDeepSeekMoeExpert* shared, float swiglu_limit) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek MoE command targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    if (state.moe_in_flight) {
        result.errors.emplace_back(
            "DeepSeek MoE workspace already has an in-flight command");
        return result;
    }
    if (routed.size() > kMaxDeepSeekRoutedExperts ||
        (routed.empty() && shared == nullptr)) {
        result.errors.emplace_back(
            "DeepSeek MoE command requires one to six routed experts or a shared expert");
        return result;
    }
    if (!std::isfinite(swiglu_limit) || swiglu_limit <= 0.0F) {
        result.errors.emplace_back(
            "DeepSeek MoE SwiGLU limit must be finite and positive");
        return result;
    }

    std::uint64_t hidden_columns = 0U;
    std::uint64_t intermediate_columns = 0U;
    auto validate_expert = [&](const CudaDeepSeekMoeExpert& expert,
                               CudaWeightEncoding encoding,
                               bool shared_expert) {
        const std::array<const CudaWeight*, 3> weights{
            expert.w1, expert.w3, expert.w2};
        for (const auto* weight : weights) {
            if (weight == nullptr || !weight->valid()) {
                result.errors.emplace_back(
                    "DeepSeek MoE command contains an invalid CUDA weight");
                return false;
            }
            if (weight->impl_->device != device) {
                result.errors.emplace_back(
                    "DeepSeek MoE weights do not belong to the command device");
                return false;
            }
            if (weight->impl_->descriptor.encoding != encoding) {
                result.errors.emplace_back(
                    "DeepSeek MoE weight encoding is incompatible with the expert kind");
                return false;
            }
        }
        const auto& w1 = expert.w1->impl_->descriptor;
        const auto& w3 = expert.w3->impl_->descriptor;
        const auto& w2 = expert.w2->impl_->descriptor;
        const auto expected_dtype = encoding == CudaWeightEncoding::Fp4E2m1Group32
                                        ? SafetensorsDtype::I8
                                        : SafetensorsDtype::F8E4M3;
        const auto expected_group = encoding == CudaWeightEncoding::Fp4E2m1Group32
                                        ? 32U
                                        : 128U;
        if (w1.dtype != expected_dtype || w3.dtype != expected_dtype ||
            w2.dtype != expected_dtype || w1.group_size != expected_group ||
            w3.group_size != expected_group || w2.group_size != expected_group ||
            w1.rows == 0U || w1.columns == 0U ||
            w3.rows != w1.rows || w3.columns != w1.columns ||
            w2.rows != w1.columns || w2.columns != w1.rows) {
            result.errors.emplace_back(
                "DeepSeek MoE W1/W3/W2 shapes or native encoding metadata are invalid");
            return false;
        }
        if (!std::isfinite(expert.coefficient) ||
            (shared_expert && expert.coefficient != 1.0F)) {
            result.errors.emplace_back(
                "DeepSeek MoE expert coefficient is invalid");
            return false;
        }
        if (hidden_columns == 0U) {
            hidden_columns = w1.columns;
            intermediate_columns = w1.rows;
        } else if (hidden_columns != w1.columns ||
                   intermediate_columns != w1.rows) {
            result.errors.emplace_back(
                "DeepSeek MoE experts do not share one exact activation shape");
            return false;
        }
        return true;
    };
    for (const auto& expert : routed) {
        if (!validate_expert(expert, CudaWeightEncoding::Fp4E2m1Group32,
                             false)) {
            return result;
        }
    }
    if (shared != nullptr &&
        !validate_expert(*shared, CudaWeightEncoding::Fp8E4m3Block128, true)) {
        return result;
    }
    if (hidden.empty() || hidden.size() != hidden_columns ||
        hidden_columns > std::numeric_limits<unsigned int>::max() ||
        intermediate_columns > std::numeric_limits<unsigned int>::max()) {
        result.errors.emplace_back(
            "DeepSeek MoE hidden row or expert dimensions are incompatible");
        return result;
    }
    if (!std::all_of(hidden.begin(), hidden.end(),
                     [](float value) { return std::isfinite(value); })) {
        result.errors.emplace_back(
            "DeepSeek MoE hidden row contains a non-finite value");
        return result;
    }

    const std::uint64_t expert_count =
        static_cast<std::uint64_t>(routed.size()) + (shared == nullptr ? 0U : 1U);
    std::uint64_t hidden_bytes = 0U;
    std::uint64_t activation_bytes = 0U;
    std::uint64_t output_bytes = 0U;
    std::uint64_t host_staging_bytes = 0U;
    if (!checked_bytes(1U, hidden_columns, sizeof(float), hidden_bytes) ||
        !checked_bytes(expert_count, intermediate_columns, sizeof(float),
                       activation_bytes) ||
        !checked_bytes(expert_count, hidden_columns, sizeof(float), output_bytes) ||
        hidden_bytes > std::numeric_limits<std::size_t>::max() ||
        activation_bytes > std::numeric_limits<std::size_t>::max() ||
        output_bytes > std::numeric_limits<std::size_t>::max() ||
        output_bytes > std::numeric_limits<std::uint64_t>::max() -
                           sizeof(unsigned int)) {
        result.errors.emplace_back("DeepSeek MoE workspace size overflows");
        return result;
    }
    host_staging_bytes = output_bytes + sizeof(unsigned int);
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for DeepSeek MoE");
    }

    std::uint64_t allocation_calls = 0U;
    std::uint64_t allocation_bytes = 0U;
    auto ensure_workspace = [&](float*& pointer, std::uint64_t& capacity,
                                std::uint64_t required, const char* operation) {
        if (required <= capacity) return true;
        if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
        pointer = nullptr;
        capacity = 0U;
        if (const auto status =
                cudaMalloc(&pointer, static_cast<std::size_t>(required));
            status != cudaSuccess) {
            result = cuda_error(status, operation);
            return false;
        }
        capacity = required;
        ++allocation_calls;
        allocation_bytes += required;
        return true;
    };
    if (!ensure_workspace(state.moe_hidden, state.moe_hidden_bytes, hidden_bytes,
                          "allocate DeepSeek MoE hidden workspace") ||
        !ensure_workspace(state.moe_activations, state.moe_activation_bytes,
                          activation_bytes,
                          "allocate DeepSeek MoE activation workspace") ||
        !ensure_workspace(state.moe_output, state.moe_output_bytes, output_bytes,
                          "allocate DeepSeek MoE output workspace")) {
        return result;
    }
    if (state.moe_bf16_silu == nullptr) {
        constexpr std::size_t bytes = kDsv4Bf16SiluEntries * sizeof(float);
        static const std::array<float, kDsv4Bf16SiluEntries> table = [] {
            std::array<float, kDsv4Bf16SiluEntries> values{};
            for (std::size_t index = 0U; index < kDsv4Bf16SiluEntries;
                 ++index) {
                const auto bits = static_cast<std::uint32_t>(index) << 16U;
                const float value = std::bit_cast<float>(bits);
                values[index] = std::isfinite(value) ? silu_f32(value) : value;
            }
            return values;
        }();
        if (const auto status = cudaMalloc(&state.moe_bf16_silu, bytes);
            status != cudaSuccess) {
            return cuda_error(status, "allocate DeepSeek BF16 SiLU table");
        }
        if (const auto status = cudaMemcpyAsync(
                state.moe_bf16_silu, table.data(), bytes,
                cudaMemcpyHostToDevice, state.stream);
            status != cudaSuccess) {
            static_cast<void>(cudaFree(state.moe_bf16_silu));
            state.moe_bf16_silu = nullptr;
            return cuda_error(status, "upload DeepSeek BF16 SiLU table");
        }
        ++allocation_calls;
        allocation_bytes += bytes;
    }
    if (state.moe_error == nullptr) {
        if (const auto status = cudaMalloc(&state.moe_error, sizeof(unsigned int));
            status != cudaSuccess) {
            return cuda_error(status, "allocate DeepSeek MoE error flag");
        }
        ++allocation_calls;
        allocation_bytes += sizeof(unsigned int);
    }
    if (host_staging_bytes > state.moe_host_staging_bytes) {
        if (state.moe_host_staging != nullptr) {
            static_cast<void>(cudaFreeHost(state.moe_host_staging));
        }
        state.moe_host_staging = nullptr;
        state.moe_host_staging_bytes = 0U;
        if (const auto status = cudaMallocHost(
                &state.moe_host_staging,
                static_cast<std::size_t>(host_staging_bytes));
            status != cudaSuccess) {
            return cuda_error(status, "allocate DeepSeek MoE host staging");
        }
        state.moe_host_staging_bytes = host_staging_bytes;
        ++allocation_calls;
        allocation_bytes += host_staging_bytes;
    }

    DeepSeekFp4Batch routed_batch;
    for (std::size_t index = 0U; index < routed.size(); ++index) {
        const auto& expert = routed[index];
        routed_batch.w1_weights[index] =
            static_cast<const unsigned char*>(expert.w1->impl_->weights);
        routed_batch.w1_scales[index] =
            static_cast<const unsigned char*>(expert.w1->impl_->scales);
        routed_batch.w3_weights[index] =
            static_cast<const unsigned char*>(expert.w3->impl_->weights);
        routed_batch.w3_scales[index] =
            static_cast<const unsigned char*>(expert.w3->impl_->scales);
        routed_batch.w2_weights[index] =
            static_cast<const unsigned char*>(expert.w2->impl_->weights);
        routed_batch.w2_scales[index] =
            static_cast<const unsigned char*>(expert.w2->impl_->scales);
        routed_batch.coefficients[index] = expert.coefficient;
    }
    routed_batch.count = static_cast<std::uint32_t>(routed.size());

    state.moe_weights.clear();
    state.moe_weights.reserve(static_cast<std::size_t>(expert_count * 3U));
    for (const auto& expert : routed) {
        state.moe_weights.push_back(expert.w1->impl_);
        state.moe_weights.push_back(expert.w3->impl_);
        state.moe_weights.push_back(expert.w2->impl_);
    }
    if (shared != nullptr) {
        state.moe_weights.push_back(shared->w1->impl_);
        state.moe_weights.push_back(shared->w3->impl_);
        state.moe_weights.push_back(shared->w2->impl_);
    }

    state.moe_hidden_columns = hidden_columns;
    state.moe_intermediate_columns = intermediate_columns;
    state.moe_rows = 1U;
    state.moe_shared_rows = 1U;
    state.moe_routed_count = routed_batch.count;
    state.moe_has_shared = shared != nullptr;
    state.moe_shared_phase_timing_valid = false;
    state.moe_host_join = false;
    state.moe_output_to_mhc = false;
    state.moe_host_callback = {};
    state.moe_kernel_launches = 0U;
    state.moe_in_flight = true;
    state.moe_poisoned = false;
    auto abort_enqueue = [&](cudaError_t status, const char* operation) {
        result = cuda_error(status, operation);
        const auto drain_status = cudaStreamSynchronize(state.stream);
        if (drain_status != cudaSuccess) {
            result.errors.emplace_back(
                std::string("drain failed DeepSeek MoE enqueue: ") +
                cudaGetErrorString(drain_status));
            state.moe_poisoned = true;
        } else {
            state.moe_in_flight = false;
            state.moe_weights.clear();
        }
    };

    if (auto status = cudaEventRecord(state.moe_start, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record DeepSeek MoE start");
        return result;
    }
    if (auto status = cudaMemsetAsync(
            state.moe_error, 0, sizeof(unsigned int), state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "reset DeepSeek MoE error flag");
        return result;
    }
    if (auto status = cudaMemcpyAsync(
            state.moe_hidden, hidden.data(), static_cast<std::size_t>(hidden_bytes),
            cudaMemcpyHostToDevice, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "upload DeepSeek MoE hidden row");
        return result;
    }
    if (auto status = cudaEventRecord(state.moe_hidden_uploaded, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record DeepSeek MoE hidden upload");
        return result;
    }

    constexpr unsigned int threads = 256U;
    const dim3 hidden_quantize_grid(
        static_cast<unsigned int>((hidden_columns + 127U) / 128U), 1U, 1U);
    quantize_activation_e4m3_kernel<<<hidden_quantize_grid, 128U, 0U,
                                      state.stream>>>(
        state.moe_hidden, hidden_columns, 1U);
    ++state.moe_kernel_launches;
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        abort_enqueue(status, "launch DeepSeek MoE hidden quantization");
        return result;
    }
    if (shared != nullptr) {
        if (auto status = cudaEventRecord(
                state.moe_shared_input_finished, state.stream);
            status != cudaSuccess) {
            abort_enqueue(status, "record DeepSeek shared input quantization");
            return result;
        }
    }

    if (!routed.empty()) {
        // Counted once per command on the gate/up dispatch; the down kernel
        // mirrors the branch, so counting both would double every entry.
        record_cuda_matmul_route(CudaMatmulRoute::Dsv4MoeRoutedFp4);
        const auto& w1 = routed.front().w1->impl_->descriptor;
        const auto& w2 = routed.front().w2->impl_->descriptor;
        const dim3 gate_grid(static_cast<unsigned int>(intermediate_columns),
                             routed_batch.count, 1U);
        deepseek_fp4_gate_up_kernel<<<gate_grid, threads, 0U, state.stream>>>(
            state.moe_activations, state.moe_hidden, routed_batch,
            hidden_columns, intermediate_columns, w1.packed_columns,
            w1.scale_columns, swiglu_limit, state.moe_bf16_silu,
            state.moe_error);
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek FP4 W1/W3 SwiGLU");
            return result;
        }
        const dim3 activation_grid(
            static_cast<unsigned int>((intermediate_columns + 127U) / 128U),
            routed_batch.count, 1U);
        quantize_activation_e4m3_kernel<<<activation_grid, 128U, 0U,
                                          state.stream>>>(
            state.moe_activations, intermediate_columns, routed_batch.count);
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek routed activation quantization");
            return result;
        }
        const dim3 down_grid(static_cast<unsigned int>(hidden_columns),
                             routed_batch.count, 1U);
        deepseek_fp4_down_kernel<<<down_grid, threads, 0U, state.stream>>>(
            state.moe_output, state.moe_activations, routed_batch,
            intermediate_columns, hidden_columns, w2.packed_columns,
            w2.scale_columns);
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek FP4 W2");
            return result;
        }
    }

    if (shared != nullptr) {
        const auto& w1 = shared->w1->impl_->descriptor;
        const auto& w2 = shared->w2->impl_->descriptor;
        float* shared_activation = state.moe_activations +
            static_cast<std::uint64_t>(routed_batch.count) * intermediate_columns;
        float* shared_output = state.moe_output +
            static_cast<std::uint64_t>(routed_batch.count) * hidden_columns;
        const auto& w3_descriptor = shared->w3->impl_->descriptor;
        const bool shared_regfed =
            regfed_matmul_enabled() &&
            // All three must already be permuted. These same weights are also read
            // canonically by enqueue_deepseek_moe_rows through the paged kernels, so
            // this site may not decide their layout on its own. An explicit
            // prepack_fragment call is the opt-in, and it opts every consumer in.
            shared->w1->impl_->fragment_prepacked &&
            shared->w3->impl_->fragment_prepacked &&
            shared->w2->impl_->fragment_prepacked &&
            regfed_fp8_shape_admissible(w1.rows, w1.columns) &&
            regfed_fp8_shape_admissible(w3_descriptor.rows, w3_descriptor.columns) &&
            regfed_fp8_shape_admissible(w2.rows, w2.columns);
        if (!shared_regfed &&
            (shared->w1->impl_->fragment_prepacked ||
             shared->w3->impl_->fragment_prepacked ||
             shared->w2->impl_->fragment_prepacked)) {
            // Refuse rather than let the scalar kernel read fragment order as
            // canonical weights, which is silent corruption, not degradation.
            abort_enqueue(cudaErrorInvalidValue,
                          "DeepSeek shared expert weights are fragment-prepacked "
                          "but the register-fed route is unavailable");
            return result;
        }
        if (shared_regfed) {
            record_cuda_matmul_route(
                CudaMatmulRoute::Dsv4MoeSharedFp8RegisterFed);
            // Gate and up share one buffer so a single allocation covers both.
            void* gate_buffer = state.moe_regfed_gate;
            if (auto status = regfed_grow(
                    gate_buffer, state.moe_regfed_gate_bytes,
                    intermediate_columns * 2U * sizeof(float), false, state.stream);
                status != cudaSuccess) {
                abort_enqueue(status, "allocate register-fed shared expert buffers");
                return result;
            }
            state.moe_regfed_gate = static_cast<float*>(gate_buffer);
            state.moe_regfed_up = state.moe_regfed_gate + intermediate_columns;
            if (auto status = launch_regfed_fp8_matvec(
                    state.moe_regfed, w1, shared->w1->impl_->weights,
                    shared->w1->impl_->scales,
                    shared->w1->impl_->fragment_prepacked, state.moe_hidden,
                    state.moe_regfed_gate, state.stream);
                status != cudaSuccess) {
                abort_enqueue(status, "launch register-fed shared expert gate");
                return result;
            }
            if (auto status = launch_regfed_fp8_matvec(
                    state.moe_regfed, w3_descriptor, shared->w3->impl_->weights,
                    shared->w3->impl_->scales,
                    shared->w3->impl_->fragment_prepacked, state.moe_hidden,
                    state.moe_regfed_up, state.stream, true);
                status != cudaSuccess) {
                abort_enqueue(status, "launch register-fed shared expert up");
                return result;
            }
            regfed_shared_swiglu_kernel<<<
                static_cast<unsigned int>((intermediate_columns + 255U) / 256U),
                256U, 0U, state.stream>>>(
                shared_activation, state.moe_regfed_gate, state.moe_regfed_up,
                static_cast<std::uint32_t>(intermediate_columns), swiglu_limit,
                state.moe_bf16_silu, state.moe_error);
            state.moe_kernel_launches += 4U;
        } else {
            record_cuda_matmul_route(CudaMatmulRoute::Dsv4MoeSharedFp8);
            deepseek_fp8_gate_up_kernel<<<
                static_cast<unsigned int>(intermediate_columns), threads, 0U,
                state.stream>>>(
                shared_activation, state.moe_hidden,
                static_cast<const unsigned char*>(shared->w1->impl_->weights),
                static_cast<const unsigned char*>(shared->w1->impl_->scales),
                static_cast<const unsigned char*>(shared->w3->impl_->weights),
                static_cast<const unsigned char*>(shared->w3->impl_->scales),
                hidden_columns, intermediate_columns, w1.scale_columns,
                swiglu_limit, state.moe_bf16_silu, state.moe_error);
            ++state.moe_kernel_launches;
        }
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek shared FP8 W1/W3 SwiGLU");
            return result;
        }
        if (auto status = cudaEventRecord(
                state.moe_shared_gate_up_finished, state.stream);
            status != cudaSuccess) {
            abort_enqueue(status, "record DeepSeek shared gate/up completion");
            return result;
        }
        const dim3 shared_activation_grid(
            static_cast<unsigned int>((intermediate_columns + 127U) / 128U),
            1U, 1U);
        quantize_activation_e4m3_kernel<<<shared_activation_grid, 128U, 0U,
                                          state.stream>>>(
            shared_activation, intermediate_columns, 1U);
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek shared activation quantization");
            return result;
        }
        if (auto status = cudaEventRecord(
                state.moe_shared_activation_finished, state.stream);
            status != cudaSuccess) {
            abort_enqueue(status,
                          "record DeepSeek shared activation quantization");
            return result;
        }
        if (shared_regfed) {
            if (auto status = launch_regfed_fp8_matvec(
                    state.moe_regfed, w2, shared->w2->impl_->weights,
                    shared->w2->impl_->scales,
                    shared->w2->impl_->fragment_prepacked, shared_activation,
                    shared_output, state.stream);
                status != cudaSuccess) {
                abort_enqueue(status, "launch register-fed shared expert down");
                return result;
            }
            state.moe_kernel_launches += 2U;
        } else {
            deepseek_fp8_down_kernel<<<
                static_cast<unsigned int>(hidden_columns), threads, 0U,
                state.stream>>>(
                shared_output, shared_activation,
                static_cast<const unsigned char*>(shared->w2->impl_->weights),
                static_cast<const unsigned char*>(shared->w2->impl_->scales),
                intermediate_columns, hidden_columns, w2.scale_columns);
            ++state.moe_kernel_launches;
        }
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek shared FP8 W2");
            return result;
        }
        if (auto status = cudaEventRecord(
                state.moe_shared_finished, state.stream);
            status != cudaSuccess) {
            abort_enqueue(status, "record DeepSeek shared down completion");
            return result;
        }
        state.moe_shared_phase_timing_valid = true;
    }
    if (auto status = cudaEventRecord(state.moe_kernel_finished, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record DeepSeek MoE kernel completion");
        return result;
    }
    {
        std::scoped_lock lock(impl_->mutex);
        auto& device_stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        device_stats.activation_h2d_bytes += hidden_bytes;
        device_stats.matmul_calls += 3U * expert_count;
        device_stats.workspace_allocation_calls += allocation_calls;
        device_stats.workspace_allocation_bytes += allocation_bytes;
        ++device_stats.deepseek_moe_calls;
        device_stats.deepseek_moe_kernel_launches += state.moe_kernel_launches;
        ++device_stats.deepseek_moe_h2d_transfers;
        device_stats.deepseek_moe_h2d_bytes += hidden_bytes;
    }
    return result;
}

ValidationResult CudaBackend::enqueue_dsv4_host_moe(
    int device, std::span<const float> hidden,
    const CudaDeepSeekMoeExpert& shared, float swiglu_limit,
    CudaDsv4HostMoeCallback callback, void* callback_context) {
    return enqueue_dsv4_host_moe_impl(
        device, hidden, shared, swiglu_limit, callback, callback_context,
        nullptr, false);
}

ValidationResult CudaBackend::enqueue_dsv4_host_moe_device_view(
    int device, std::span<const float> hidden,
    const CudaDeepSeekMoeExpert& shared, float swiglu_limit,
    CudaDsv4HostMoeCallback callback, void* callback_context,
    CudaDsv4HostMoeDeviceView& view) {
    view = {};
    auto result = enqueue_dsv4_host_moe(
        device, hidden, shared, swiglu_limit, callback, callback_context);
    if (!result.ok()) return result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end() ||
        found->second.moe_host_callback_count == 0U) {
        result.errors.emplace_back(
            "DeepSeek host MoE device view has no queued command");
        return result;
    }
    auto& state = found->second;
    view.stream = state.stream;
    view.output = state.moe_output;
    view.status = state.moe_error;
    if (view.stream == nullptr || view.output == nullptr ||
        view.status == nullptr) {
        view = {};
        result.errors.emplace_back(
            "DeepSeek host MoE device view is incomplete");
    }
    return result;
}

ValidationResult CudaBackend::enqueue_dsv4_host_moe_from_mhc(
    int device, const CudaDeepSeekMoeExpert& shared, float swiglu_limit,
    CudaDsv4HostMoeCallback callback, void* callback_context) {
    return enqueue_dsv4_host_moe_impl(
        device, {}, shared, swiglu_limit, callback, callback_context,
        nullptr, true);
}

ValidationResult CudaBackend::enqueue_dsv4_host_moe_from_device_input(
    int device, const CudaDeepSeekMoeExpert& shared, float swiglu_limit,
    CudaDsv4DeviceInputHostMoeCallback callback, void* callback_context) {
    return enqueue_dsv4_host_moe_impl(
        device, {}, shared, swiglu_limit, nullptr, callback_context,
        callback, true);
}

ValidationResult CudaBackend::enqueue_dsv4_host_moe_from_device_input_device_view(
    int device, const CudaDeepSeekMoeExpert& shared, float swiglu_limit,
    CudaDsv4DeviceInputHostMoeCallback callback, void* callback_context,
    CudaDsv4HostMoeDeviceView& view,
    CudaDsv4DeviceInputHostMoeRouteCallback route_callback) {
    view = {};
    auto result = enqueue_dsv4_host_moe_impl(
        device, {}, shared, swiglu_limit, nullptr, callback_context,
        callback, true, route_callback);
    if (!result.ok()) return result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end() ||
        found->second.moe_host_callback_count == 0U) {
        result.errors.emplace_back(
            "DeepSeek device-input host MoE view has no queued command");
        return result;
    }
    auto& state = found->second;
    // The reusable rank-local view deliberately leaves the backend branch
    // unpublished. The existing join still computes its local BF16 value in
    // stream order, but only the caller's FP32 NCCL result may commit the
    // branch through dsv4_mhc_commit_reduced_branch().
    state.dsv4_mhc_branch_ready = false;
    view.stream = state.stream;
    view.output = state.moe_output;
    view.status = state.moe_error;
    if (view.stream == nullptr || view.output == nullptr ||
        view.status == nullptr) {
        view = {};
        result.errors.emplace_back(
            "DeepSeek device-input host MoE view is incomplete");
    }
    return result;
}

ValidationResult CudaBackend::enqueue_dsv4_host_moe_impl(
    int device, std::span<const float> hidden,
    const CudaDeepSeekMoeExpert& shared, float swiglu_limit,
    CudaDsv4HostMoeCallback callback, void* callback_context,
    CudaDsv4DeviceInputHostMoeCallback device_input_callback,
    bool mhc_source_and_destination,
    CudaDsv4DeviceInputHostMoeRouteCallback route_callback) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek host MoE command targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    if (state.moe_in_flight) {
        result.errors.emplace_back(
            "DeepSeek MoE workspace already has an in-flight command");
        return result;
    }
    if (state.moe_host_callback_count >=
        state.moe_host_callbacks.size()) {
        result.errors.emplace_back(
            "DeepSeek fixed host-MoE command chain is full");
        return result;
    }
    const bool device_input = device_input_callback != nullptr;
    if ((callback == nullptr) == !device_input || callback_context == nullptr ||
        !std::isfinite(swiglu_limit) || swiglu_limit <= 0.0F) {
        result.errors.emplace_back(
            "DeepSeek host MoE requires a callback, context, and positive SwiGLU limit");
        return result;
    }

    const std::array<const CudaWeight*, 3U> weights{
        shared.w1, shared.w3, shared.w2};
    for (const auto* weight : weights) {
        if (weight == nullptr || !weight->valid()) {
            result.errors.emplace_back(
                "DeepSeek host MoE shared expert contains an invalid CUDA weight");
            return result;
        }
        if (weight->impl_->device != device ||
            weight->impl_->descriptor.encoding !=
                CudaWeightEncoding::Fp8E4m3Block128) {
            result.errors.emplace_back(
                "DeepSeek host MoE shared weight has the wrong device or encoding");
            return result;
        }
    }
    const auto& w1 = shared.w1->impl_->descriptor;
    const auto& w3 = shared.w3->impl_->descriptor;
    const auto& w2 = shared.w2->impl_->descriptor;
    if (shared.coefficient != 1.0F ||
        w1.dtype != SafetensorsDtype::F8E4M3 ||
        w3.dtype != SafetensorsDtype::F8E4M3 ||
        w2.dtype != SafetensorsDtype::F8E4M3 ||
        w1.group_size != 128U || w3.group_size != 128U ||
        w2.group_size != 128U || w1.rows == 0U || w1.columns == 0U ||
        w3.rows != w1.rows || w3.columns != w1.columns ||
        w2.rows != w1.columns || w2.columns != w1.rows ||
        (mhc_source_and_destination
             ? (!hidden.empty() || state.dsv4_mhc_stage != 1U ||
                state.dsv4_mhc_workspace == nullptr ||
                state.dsv4_mhc_branch_ready ||
                (device_input != state.dsv4_host_moe_input_pending) ||
                w1.columns != kDsv4MhcHidden)
             : hidden.size() != w1.columns) ||
        w1.columns > std::numeric_limits<unsigned int>::max() ||
        w1.rows > std::numeric_limits<unsigned int>::max()) {
        result.errors.emplace_back(
            "DeepSeek host MoE shared expert shape or metadata is invalid");
        return result;
    }
    if (!mhc_source_and_destination &&
        !std::all_of(hidden.begin(), hidden.end(),
                     [](float value) { return std::isfinite(value); })) {
        result.errors.emplace_back(
            "DeepSeek host MoE hidden row contains a non-finite value");
        return result;
    }
    if (device_input &&
        (state.dsv4_host_moe_router_logits == nullptr ||
         (state.dsv4_host_moe_device_failure == nullptr &&
          state.dsv4_host_moe_host_failure == nullptr))) {
        result.errors.emplace_back(
            "DeepSeek device-input host MoE has no deferred input or status");
        return result;
    }

    const auto hidden_columns = w1.columns;
    const auto intermediate_columns = w1.rows;
    std::uint64_t hidden_bytes = 0U;
    std::uint64_t activation_bytes = 0U;
    std::uint64_t output_bytes = 0U;
    std::uint64_t rank_partial_bytes = 0U;
    if (!checked_bytes(1U, hidden_columns, sizeof(float), hidden_bytes) ||
        !checked_bytes(1U, intermediate_columns, sizeof(float),
                       activation_bytes) ||
        !checked_bytes(3U, hidden_columns, sizeof(float), output_bytes) ||
        !checked_bytes(2U, hidden_columns, sizeof(float), rank_partial_bytes) ||
        hidden_bytes > std::numeric_limits<std::size_t>::max() ||
        activation_bytes > std::numeric_limits<std::size_t>::max() ||
        output_bytes > std::numeric_limits<std::size_t>::max() ||
        rank_partial_bytes > std::numeric_limits<std::size_t>::max() ||
        hidden_bytes > std::numeric_limits<std::uint64_t>::max() -
                           sizeof(unsigned int)) {
        result.errors.emplace_back("DeepSeek host MoE workspace size overflows");
        return result;
    }
    constexpr std::uint64_t encoded_hidden_bytes =
        kDsv4MhcHidden * sizeof(std::uint16_t);
    constexpr std::uint64_t router_bytes =
        kDsv4MhcRouterLogits * sizeof(float);
    const auto local_upstream_failure_bytes =
        device_input && state.dsv4_host_moe_device_failure != nullptr
            ? sizeof(unsigned int) : 0U;
    const auto device_input_bytes = device_input
        ? encoded_hidden_bytes + router_bytes +
              local_upstream_failure_bytes
        : 0U;
    const auto host_staging_bytes = std::max(
        rank_partial_bytes + device_input_bytes,
        hidden_bytes + sizeof(unsigned int));
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for DeepSeek host MoE");
    }

    std::uint64_t allocation_calls = 0U;
    std::uint64_t allocation_bytes = 0U;
    const auto ensure_workspace = [&](float*& pointer,
                                      std::uint64_t& capacity,
                                      std::uint64_t required,
                                      const char* operation) {
        if (required <= capacity) return true;
        if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
        pointer = nullptr;
        capacity = 0U;
        if (const auto status = cudaMalloc(
                &pointer, static_cast<std::size_t>(required));
            status != cudaSuccess) {
            result = cuda_error(status, operation);
            return false;
        }
        capacity = required;
        ++allocation_calls;
        allocation_bytes += required;
        return true;
    };
    if (!ensure_workspace(state.moe_hidden, state.moe_hidden_bytes,
                          hidden_bytes,
                          "allocate DeepSeek host MoE hidden workspace") ||
        !ensure_workspace(state.moe_activations, state.moe_activation_bytes,
                          activation_bytes,
                          "allocate DeepSeek host MoE activation workspace") ||
        !ensure_workspace(state.moe_output, state.moe_output_bytes,
                          output_bytes,
                          "allocate DeepSeek host MoE output workspace")) {
        return result;
    }
    if (state.moe_bf16_silu == nullptr) {
        constexpr std::size_t bytes = kDsv4Bf16SiluEntries * sizeof(float);
        static const std::array<float, kDsv4Bf16SiluEntries> table = [] {
            std::array<float, kDsv4Bf16SiluEntries> values{};
            for (std::size_t index = 0U; index < kDsv4Bf16SiluEntries;
                 ++index) {
                const auto bits = static_cast<std::uint32_t>(index) << 16U;
                const float value = std::bit_cast<float>(bits);
                values[index] = std::isfinite(value) ? silu_f32(value) : value;
            }
            return values;
        }();
        if (const auto status = cudaMalloc(&state.moe_bf16_silu, bytes);
            status != cudaSuccess) {
            return cuda_error(status, "allocate DeepSeek BF16 SiLU table");
        }
        if (const auto status = cudaMemcpyAsync(
                state.moe_bf16_silu, table.data(), bytes,
                cudaMemcpyHostToDevice, state.stream);
            status != cudaSuccess) {
            static_cast<void>(cudaFree(state.moe_bf16_silu));
            state.moe_bf16_silu = nullptr;
            return cuda_error(status, "upload DeepSeek BF16 SiLU table");
        }
        ++allocation_calls;
        allocation_bytes += bytes;
    }
    if (state.moe_error == nullptr) {
        if (const auto status = cudaMalloc(
                &state.moe_error, sizeof(unsigned int));
            status != cudaSuccess) {
            return cuda_error(status, "allocate DeepSeek MoE error flag");
        }
        ++allocation_calls;
        allocation_bytes += sizeof(unsigned int);
    }
    if (host_staging_bytes > state.moe_host_staging_bytes) {
        if (state.moe_host_staging != nullptr) {
            static_cast<void>(cudaFreeHost(state.moe_host_staging));
        }
        state.moe_host_staging = nullptr;
        state.moe_host_staging_bytes = 0U;
        if (const auto status = cudaMallocHost(
                &state.moe_host_staging,
                static_cast<std::size_t>(host_staging_bytes));
            status != cudaSuccess) {
            return cuda_error(status,
                              "allocate DeepSeek host MoE pinned staging");
        }
        state.moe_host_staging_bytes = host_staging_bytes;
        ++allocation_calls;
        allocation_bytes += host_staging_bytes;
    }

    if (state.moe_host_callback_count == 0U) {
        state.moe_weights.clear();
        state.moe_weights.reserve(3U * 43U);
    }
    state.moe_weights.push_back(shared.w1->impl_);
    state.moe_weights.push_back(shared.w3->impl_);
    state.moe_weights.push_back(shared.w2->impl_);
    state.moe_hidden_columns = hidden_columns;
    state.moe_intermediate_columns = intermediate_columns;
    state.moe_rows = 1U;
    state.moe_shared_rows = 1U;
    state.moe_routed_count = 0U;
    state.moe_has_shared = true;
    state.moe_shared_phase_timing_valid = false;
    state.moe_host_join = true;
    state.moe_output_to_mhc = mhc_source_and_destination;
    state.moe_kernel_launches = 0U;
    state.moe_in_flight = true;
    state.moe_poisoned = false;
    auto& host_callback_state = state.moe_host_callbacks[
        state.moe_host_callback_count];
    host_callback_state = {};
    host_callback_state.function = callback;
    host_callback_state.device_input_function = device_input_callback;
    host_callback_state.route_function = route_callback;
    host_callback_state.context = callback_context;
    host_callback_state.rank_partials =
        static_cast<float*>(state.moe_host_staging);
    host_callback_state.rank_partial_elements = 2U * hidden_columns;
    auto* host_bytes = static_cast<std::byte*>(state.moe_host_staging);
    auto* staged_hidden = reinterpret_cast<std::uint16_t*>(
        host_bytes + static_cast<std::ptrdiff_t>(rank_partial_bytes));
    auto* staged_router = reinterpret_cast<float*>(
        host_bytes + static_cast<std::ptrdiff_t>(
            rank_partial_bytes + encoded_hidden_bytes));
    auto* staged_upstream_failure = reinterpret_cast<unsigned int*>(
        host_bytes + static_cast<std::ptrdiff_t>(
            rank_partial_bytes + encoded_hidden_bytes + router_bytes));
    if (device_input) {
        host_callback_state.encoded_hidden = staged_hidden;
        host_callback_state.hidden_elements = hidden_columns;
        host_callback_state.router_logits = staged_router;
        host_callback_state.router_elements = kDsv4MhcRouterLogits;
        host_callback_state.upstream_failure =
            state.dsv4_host_moe_host_failure != nullptr
                ? state.dsv4_host_moe_host_failure
                : staged_upstream_failure;
    }

    const auto abort_enqueue = [&](cudaError_t status,
                                   const char* operation) {
        state.moe_shared_phase_timing_valid = false;
        result = cuda_error(status, operation);
        const auto main_status = cudaStreamSynchronize(state.stream);
        const auto shared_status = cudaStreamSynchronize(
            state.moe_shared_stream);
        if (main_status != cudaSuccess || shared_status != cudaSuccess) {
            result.errors.emplace_back(
                "drain failed DeepSeek host MoE enqueue");
            state.moe_poisoned = true;
        } else {
            state.moe_in_flight = false;
            state.moe_host_join = false;
            state.moe_output_to_mhc = false;
            state.moe_host_callback = {};
            state.moe_host_callback_count = 0U;
            state.moe_weights.clear();
            state.dsv4_host_moe_input_pending = false;
            state.dsv4_host_moe_router_logits = nullptr;
            state.dsv4_host_moe_device_failure = nullptr;
            state.dsv4_host_moe_host_failure = nullptr;
            state.dsv4_deferred_attention_source_device = -1;
            state.dsv4_deferred_attention_cross_transition = false;
        }
    };

    if (auto status = cudaEventRecord(state.moe_start, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record DeepSeek host MoE start");
        return result;
    }
    if (state.moe_host_callback_count == 0U) {
        if (auto status = cudaMemsetAsync(
                state.moe_error, 0, sizeof(unsigned int), state.stream);
            status != cudaSuccess) {
            abort_enqueue(status, "reset DeepSeek host MoE error flag");
            return result;
        }
    }
    if (!mhc_source_and_destination) {
        if (auto status = cudaMemcpyAsync(
                state.moe_hidden, hidden.data(),
                static_cast<std::size_t>(hidden_bytes), cudaMemcpyHostToDevice,
                state.stream);
            status != cudaSuccess) {
            abort_enqueue(status, "upload DeepSeek host MoE hidden row");
            return result;
        }
    }
    if (auto status = cudaEventRecord(
            state.moe_hidden_uploaded, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record DeepSeek host MoE hidden upload");
        return result;
    }
    if (auto status = cudaStreamWaitEvent(
            state.moe_shared_stream, state.moe_hidden_uploaded);
        status != cudaSuccess) {
        abort_enqueue(status, "fan out DeepSeek shared expert");
        return result;
    }
    if (device_input) {
        if (auto status = cudaMemcpyAsync(
                staged_hidden, state.dsv4_mhc_workspace->layer_input,
                static_cast<std::size_t>(encoded_hidden_bytes),
                cudaMemcpyDeviceToHost, state.stream);
            status != cudaSuccess) {
            abort_enqueue(status,
                          "stage DeepSeek device-input host MoE hidden row");
            return result;
        }
        if (auto status = cudaMemcpyAsync(
                staged_router, state.dsv4_host_moe_router_logits,
                static_cast<std::size_t>(router_bytes),
                cudaMemcpyDeviceToHost, state.stream);
            status != cudaSuccess) {
            abort_enqueue(status,
                          "stage DeepSeek device-input host MoE router logits");
            return result;
        }
        if (state.dsv4_host_moe_device_failure != nullptr) {
            if (auto status = cudaMemcpyAsync(
                    staged_upstream_failure,
                    state.dsv4_host_moe_device_failure,
                    sizeof(*staged_upstream_failure), cudaMemcpyDeviceToHost,
                    state.stream);
                status != cudaSuccess) {
                abort_enqueue(status,
                              "stage DeepSeek deferred attention status");
                return result;
            }
        }
    }

    const dim3 hidden_quantize_grid(
        static_cast<unsigned int>((hidden_columns + 127U) / 128U), 1U, 1U);
    if (mhc_source_and_destination) {
        quantize_bf16_activation_e4m3_kernel<<<
            hidden_quantize_grid, 128U, 0U, state.moe_shared_stream>>>(
            state.moe_hidden, state.dsv4_mhc_workspace->layer_input,
            hidden_columns);
    } else {
        quantize_activation_e4m3_kernel<<<
            hidden_quantize_grid, 128U, 0U, state.moe_shared_stream>>>(
            state.moe_hidden, hidden_columns, 1U);
    }
    ++state.moe_kernel_launches;
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        abort_enqueue(status,
                      "launch DeepSeek host MoE hidden quantization");
        return result;
    }
    if (auto status = cudaEventRecord(
            state.moe_shared_input_finished, state.moe_shared_stream);
        status != cudaSuccess) {
        abort_enqueue(status,
                      "record DeepSeek shared input quantization");
        return result;
    }

    constexpr unsigned int threads = 256U;
    const auto& w3_descriptor = shared.w3->impl_->descriptor;
    const bool shared_regfed =
        regfed_matmul_enabled() &&
        // All three must already be permuted. These same weights are also read
        // canonically by enqueue_deepseek_moe_rows through the paged kernels, so
        // this site may not decide their layout on its own. An explicit
        // prepack_fragment call is the opt-in, and it opts every consumer in.
        shared.w1->impl_->fragment_prepacked &&
        shared.w3->impl_->fragment_prepacked &&
        shared.w2->impl_->fragment_prepacked &&
        regfed_fp8_shape_admissible(w1.rows, w1.columns) &&
        regfed_fp8_shape_admissible(w3_descriptor.rows, w3_descriptor.columns) &&
        regfed_fp8_shape_admissible(w2.rows, w2.columns);
    if (!shared_regfed &&
        (shared.w1->impl_->fragment_prepacked ||
         shared.w3->impl_->fragment_prepacked ||
         shared.w2->impl_->fragment_prepacked)) {
        // Refuse rather than let the scalar kernel read fragment order as
        // canonical weights, which is silent corruption, not degradation.
        abort_enqueue(cudaErrorInvalidValue,
                      "DeepSeek shared expert weights are fragment-prepacked "
                      "but the register-fed route is unavailable");
        return result;
    }
    if (shared_regfed) {
        record_cuda_matmul_route(CudaMatmulRoute::Dsv4MoeSharedFp8RegisterFed);
        void* gate_buffer = state.moe_regfed_gate;
        if (auto status = regfed_grow(
                gate_buffer, state.moe_regfed_gate_bytes,
                intermediate_columns * 2U * sizeof(float), false,
                state.moe_shared_stream);
            status != cudaSuccess) {
            abort_enqueue(status, "allocate register-fed shared expert buffers");
            return result;
        }
        state.moe_regfed_gate = static_cast<float*>(gate_buffer);
        state.moe_regfed_up = state.moe_regfed_gate + intermediate_columns;
        if (auto status = launch_regfed_fp8_matvec(
                state.moe_regfed, w1, shared.w1->impl_->weights,
                shared.w1->impl_->scales, shared.w1->impl_->fragment_prepacked,
                state.moe_hidden, state.moe_regfed_gate,
                state.moe_shared_stream);
            status != cudaSuccess) {
            abort_enqueue(status, "launch register-fed shared expert gate");
            return result;
        }
        if (auto status = launch_regfed_fp8_matvec(
                state.moe_regfed, w3_descriptor, shared.w3->impl_->weights,
                shared.w3->impl_->scales, shared.w3->impl_->fragment_prepacked,
                state.moe_hidden, state.moe_regfed_up,
                state.moe_shared_stream, true);
            status != cudaSuccess) {
            abort_enqueue(status, "launch register-fed shared expert up");
            return result;
        }
        regfed_shared_swiglu_kernel<<<
            static_cast<unsigned int>((intermediate_columns + 255U) / 256U),
            256U, 0U, state.moe_shared_stream>>>(
            state.moe_activations, state.moe_regfed_gate, state.moe_regfed_up,
            static_cast<std::uint32_t>(intermediate_columns), swiglu_limit,
            state.moe_bf16_silu, state.moe_error);
        state.moe_kernel_launches += 4U;
    } else {
        record_cuda_matmul_route(CudaMatmulRoute::Dsv4MoeSharedFp8);
        deepseek_fp8_gate_up_kernel<<<
            static_cast<unsigned int>(intermediate_columns), threads, 0U,
            state.moe_shared_stream>>>(
            state.moe_activations, state.moe_hidden,
            static_cast<const unsigned char*>(shared.w1->impl_->weights),
            static_cast<const unsigned char*>(shared.w1->impl_->scales),
            static_cast<const unsigned char*>(shared.w3->impl_->weights),
            static_cast<const unsigned char*>(shared.w3->impl_->scales),
            hidden_columns, intermediate_columns, w1.scale_columns,
            swiglu_limit, state.moe_bf16_silu, state.moe_error);
        ++state.moe_kernel_launches;
    }
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        abort_enqueue(status, "launch DeepSeek host MoE shared W1/W3");
        return result;
    }
    if (auto status = cudaEventRecord(
            state.moe_shared_gate_up_finished, state.moe_shared_stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record DeepSeek shared gate/up completion");
        return result;
    }
    const dim3 activation_grid(
        static_cast<unsigned int>((intermediate_columns + 127U) / 128U),
        1U, 1U);
    quantize_activation_e4m3_kernel<<<
        activation_grid, 128U, 0U, state.moe_shared_stream>>>(
        state.moe_activations, intermediate_columns, 1U);
    ++state.moe_kernel_launches;
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        abort_enqueue(status,
                      "launch DeepSeek host MoE shared activation quantization");
        return result;
    }
    if (auto status = cudaEventRecord(
            state.moe_shared_activation_finished, state.moe_shared_stream);
        status != cudaSuccess) {
        abort_enqueue(status,
                      "record DeepSeek shared activation quantization");
        return result;
    }
    if (shared_regfed) {
        if (auto status = launch_regfed_fp8_matvec(
                state.moe_regfed, w2, shared.w2->impl_->weights,
                shared.w2->impl_->scales, shared.w2->impl_->fragment_prepacked,
                state.moe_activations, state.moe_output,
                state.moe_shared_stream);
            status != cudaSuccess) {
            abort_enqueue(status, "launch register-fed shared expert down");
            return result;
        }
        state.moe_kernel_launches += 2U;
    } else {
        deepseek_fp8_down_kernel<<<
            static_cast<unsigned int>(hidden_columns), threads, 0U,
            state.moe_shared_stream>>>(
            state.moe_output, state.moe_activations,
            static_cast<const unsigned char*>(shared.w2->impl_->weights),
            static_cast<const unsigned char*>(shared.w2->impl_->scales),
            intermediate_columns, hidden_columns, w2.scale_columns);
        ++state.moe_kernel_launches;
    }
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        abort_enqueue(status, "launch DeepSeek host MoE shared W2");
        return result;
    }
    if (auto status = cudaEventRecord(
            state.moe_shared_finished, state.moe_shared_stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record DeepSeek shared-expert completion");
        return result;
    }
    state.moe_shared_phase_timing_valid = true;

    // Overlapped dispatch is used only when the caller supplied a route half
    // and this device actually holds a tier. Everything else keeps the
    // original ordering byte for byte, which is the rollback.
    const bool tier_present =
        state.tier_committed && state.tier_installed != 0U;
    const bool overlapped = tier_present && route_callback != nullptr;
    if (overlapped) {
        if (auto status = cudaLaunchHostFunc(
                state.stream, run_dsv4_host_moe_route_callback,
                &host_callback_state);
            status != cudaSuccess) {
            abort_enqueue(status, "enqueue DeepSeek route callback");
            return result;
        }
        // Recorded between the two halves. This is the whole mechanism: the
        // tier stream is released here, while the host share below is still
        // to run.
        if (auto status = cudaEventRecord(
                state.tier_route_ready, state.stream);
            status != cudaSuccess) {
            abort_enqueue(status, "record DeepSeek tier route event");
            return result;
        }
    }
    if (auto status = cudaLaunchHostFunc(
            state.stream, run_dsv4_host_moe_callback,
            &host_callback_state);
        status != cudaSuccess) {
        abort_enqueue(status, "enqueue DeepSeek CPU-MoE callback");
        return result;
    }
    auto* rank_partials = state.moe_output + hidden_columns;
    if (auto status = cudaMemcpyAsync(
            rank_partials, state.moe_host_staging,
            static_cast<std::size_t>(rank_partial_bytes),
            cudaMemcpyHostToDevice, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "upload DeepSeek CPU-MoE rank partials");
        return result;
    }
    // Routed-expert tier. Enqueued after the callback, so it is stream-ordered
    // behind it and observes the selection the callback wrote into pinned
    // memory, and after the rank-partial upload, so it accumulates into the
    // same buffer the join below already consumes. No host wait, no worker
    // thread, no change to the join itself.
    if (tier_present) {
        // Overlapped, everything below runs on the tier stream, gated by the
        // route event and accumulating into its own partials. Serial, it is
        // the original: the rank stream, behind the one callback, straight
        // into the rank partials.
        const auto tier_stream = overlapped ? state.tier_stream : state.stream;
        auto* tier_destination = overlapped ? state.tier_partials : rank_partials;
        if (overlapped) {
            if (state.tier_partials == nullptr) {
                void* partials = nullptr;
                if (const auto status = cudaMalloc(
                        &partials,
                        static_cast<std::size_t>(hidden_columns) * sizeof(float));
                    status != cudaSuccess) {
                    abort_enqueue(status, "allocate DeepSeek tier partials");
                    return result;
                }
                state.tier_partials = static_cast<float*>(partials);
                tier_destination = state.tier_partials;
            }
            if (auto status = cudaStreamWaitEvent(
                    tier_stream, state.tier_route_ready, 0U);
                status != cudaSuccess) {
                abort_enqueue(status, "gate DeepSeek tier stream");
                return result;
            }
            if (auto status = cudaMemsetAsync(
                    tier_destination, 0,
                    static_cast<std::size_t>(hidden_columns) * sizeof(float),
                    tier_stream);
                status != cudaSuccess) {
                abort_enqueue(status, "clear DeepSeek tier partials");
                return result;
            }
        }
        const std::uint64_t tier_activation_bytes =
            kMaxDeepSeekRoutedExperts * intermediate_columns * sizeof(float);
        if (tier_activation_bytes > state.tier_activation_bytes) {
            if (state.tier_activations != nullptr) {
                static_cast<void>(cudaFree(state.tier_activations));
                state.tier_activations = nullptr;
                state.tier_activation_bytes = 0U;
            }
            void* scratch = nullptr;
            if (const auto status = cudaMalloc(
                    &scratch, static_cast<std::size_t>(tier_activation_bytes));
                status != cudaSuccess) {
                abort_enqueue(status, "allocate DeepSeek tier activations");
                return result;
            }
            state.tier_activations = static_cast<float*>(scratch);
            state.tier_activation_bytes = tier_activation_bytes;
        }
        if (auto status = cudaMemcpyAsync(
                state.tier_selection_device, state.tier_selection_host,
                sizeof(CudaDsv4TierSelection), cudaMemcpyHostToDevice,
                tier_stream);
            status != cudaSuccess) {
            abort_enqueue(status, "upload DeepSeek tier selection");
            return result;
        }
        DeepSeekTierTable table;
        table.w1_weights = state.tier_device_pointers[0];
        table.w1_scales = state.tier_device_pointers[1];
        table.w3_weights = state.tier_device_pointers[2];
        table.w3_scales = state.tier_device_pointers[3];
        table.w2_weights = state.tier_device_pointers[4];
        table.w2_scales = state.tier_device_pointers[5];
        table.experts = state.tier_experts;
        const auto* tier_selection =
            reinterpret_cast<const DeepSeekTierSelection*>(
                state.tier_selection_device);
        const dim3 tier_gate_grid(
            static_cast<unsigned int>(intermediate_columns),
            kMaxDeepSeekRoutedExperts, 1U);
        record_cuda_matmul_route(CudaMatmulRoute::Dsv4MoeTierFp4);
        deepseek_fp4_tier_gate_up_kernel<<<
            tier_gate_grid, threads, 0U, tier_stream>>>(
            state.tier_activations, state.moe_hidden, table, tier_selection,
            hidden_columns, intermediate_columns,
            state.tier_gate_packed_columns, state.tier_gate_scale_columns);
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek tier W1/W3");
            return result;
        }
        const dim3 tier_down_grid(static_cast<unsigned int>(hidden_columns),
                                  kMaxDeepSeekRoutedExperts, 1U);
        deepseek_fp4_tier_down_kernel<<<
            tier_down_grid, threads, 0U, tier_stream>>>(
            tier_destination, state.tier_activations, table, tier_selection,
            intermediate_columns, hidden_columns,
            state.tier_down_packed_columns, state.tier_down_scale_columns);
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek tier W2");
            return result;
        }
        if (overlapped) {
            // Rejoin before the join reads the tier partials. This also keeps
            // the next layer's route half from overwriting the one pinned
            // selection slot while this layer's tier is still reading it.
            if (auto status = cudaEventRecord(state.tier_finished, tier_stream);
                status != cudaSuccess) {
                abort_enqueue(status, "record DeepSeek tier completion");
                return result;
            }
            if (auto status = cudaStreamWaitEvent(
                    state.stream, state.tier_finished, 0U);
                status != cudaSuccess) {
                abort_enqueue(status, "rejoin DeepSeek tier stream");
                return result;
            }
        }
    }
    if (auto status = cudaStreamWaitEvent(
            state.stream, state.moe_shared_finished);
        status != cudaSuccess) {
        abort_enqueue(status, "join DeepSeek shared-expert stream");
        return result;
    }
    constexpr unsigned int join_threads = 256U;
    const auto join_blocks = static_cast<unsigned int>(
        (hidden_columns + join_threads - 1U) / join_threads);
    const float* join_tier_partials = overlapped ? state.tier_partials : nullptr;
    if (mhc_source_and_destination) {
        dsv4_host_moe_join_mhc_kernel<<<
            join_blocks, join_threads, 0U, state.stream>>>(
            state.moe_output, rank_partials, join_tier_partials,
            state.dsv4_mhc_workspace->branch, hidden_columns,
            state.moe_error);
    } else {
        dsv4_host_moe_join_kernel<<<
            join_blocks, join_threads, 0U, state.stream>>>(
            state.moe_output, rank_partials, join_tier_partials,
            hidden_columns, state.moe_error);
    }
    ++state.moe_kernel_launches;
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        abort_enqueue(status, "launch DeepSeek routed/shared join");
        return result;
    }
    if (auto status = cudaEventRecord(
            state.moe_kernel_finished, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record DeepSeek host MoE completion");
        return result;
    }

    const auto total_h2d_bytes = rank_partial_bytes +
        (mhc_source_and_destination ? 0U : hidden_bytes);
    const auto input_d2h_bytes = device_input
        ? encoded_hidden_bytes + router_bytes +
              (state.dsv4_host_moe_device_failure != nullptr
                   ? sizeof(unsigned int) : 0U)
        : 0U;
    {
        std::scoped_lock lock(impl_->mutex);
        auto& device_stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        device_stats.activation_h2d_bytes += total_h2d_bytes;
        device_stats.activation_d2h_bytes += input_d2h_bytes;
        device_stats.matmul_calls += 3U;
        device_stats.workspace_allocation_calls += allocation_calls;
        device_stats.workspace_allocation_bytes += allocation_bytes;
        ++device_stats.deepseek_moe_calls;
        device_stats.deepseek_moe_kernel_launches +=
            state.moe_kernel_launches;
        device_stats.deepseek_moe_h2d_transfers +=
            mhc_source_and_destination ? 1U : 2U;
        device_stats.deepseek_moe_h2d_bytes += total_h2d_bytes;
        device_stats.deepseek_moe_d2h_transfers += device_input
            ? (state.dsv4_host_moe_device_failure != nullptr ? 3U : 2U)
            : 0U;
        device_stats.deepseek_moe_d2h_bytes += input_d2h_bytes;
    }
    if (device_input) {
        state.dsv4_host_moe_input_pending = false;
        state.dsv4_host_moe_router_logits = nullptr;
        state.dsv4_host_moe_device_failure = nullptr;
        state.dsv4_host_moe_host_failure = nullptr;
    }
    if (mhc_source_and_destination) {
        // The branch is produced later in this stream, but every consumer is
        // also stream/event ordered. Publishing host ownership now allows the
        // fixed dependent command chain to be issued without a CPU drain.
        state.dsv4_mhc_branch_ready = true;
    }
    state.moe_in_flight = false;
    ++state.moe_host_callback_count;
    return result;
}

ValidationResult CudaBackend::enqueue_deepseek_moe_rows(
    int device, std::span<const float> hidden, std::uint32_t hidden_rows,
    std::span<const CudaDeepSeekMoeRowGroup> groups,
    const CudaDeepSeekMoeExpert* shared,
    std::span<const std::uint32_t> shared_rows, float swiglu_limit) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek MoE page command targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    if (state.moe_in_flight) {
        result.errors.emplace_back(
            "DeepSeek MoE workspace already has an in-flight command");
        return result;
    }
    if (hidden_rows == 0U || (groups.empty() && shared == nullptr)) {
        result.errors.emplace_back(
            "DeepSeek MoE page command requires rows and at least one expert");
        return result;
    }
    if (!std::isfinite(swiglu_limit) || swiglu_limit <= 0.0F) {
        result.errors.emplace_back(
            "DeepSeek MoE SwiGLU limit must be finite and positive");
        return result;
    }

    std::uint64_t hidden_columns = 0U;
    std::uint64_t intermediate_columns = 0U;
    const auto validate_triplet = [&](const CudaWeight* w1_weight,
                                      const CudaWeight* w3_weight,
                                      const CudaWeight* w2_weight,
                                      CudaWeightEncoding encoding) {
        for (const auto* weight : {w1_weight, w3_weight, w2_weight}) {
            if (weight == nullptr || !weight->valid()) {
                result.errors.emplace_back(
                    "DeepSeek MoE page command contains an invalid CUDA weight");
                return false;
            }
            if (weight->impl_->device != device) {
                result.errors.emplace_back(
                    "DeepSeek MoE page weights do not belong to the command device");
                return false;
            }
            if (weight->impl_->descriptor.encoding != encoding) {
                result.errors.emplace_back(
                    "DeepSeek MoE page weight encoding is incompatible with the expert kind");
                return false;
            }
        }
        const auto& w1 = w1_weight->impl_->descriptor;
        const auto& w3 = w3_weight->impl_->descriptor;
        const auto& w2 = w2_weight->impl_->descriptor;
        const auto expected_dtype = encoding == CudaWeightEncoding::Fp4E2m1Group32
                                        ? SafetensorsDtype::I8
                                        : SafetensorsDtype::F8E4M3;
        const auto expected_group =
            encoding == CudaWeightEncoding::Fp4E2m1Group32 ? 32U : 128U;
        if (w1.dtype != expected_dtype || w3.dtype != expected_dtype ||
            w2.dtype != expected_dtype || w1.group_size != expected_group ||
            w3.group_size != expected_group || w2.group_size != expected_group ||
            w1.rows == 0U || w1.columns == 0U ||
            w3.rows != w1.rows || w3.columns != w1.columns ||
            w2.rows != w1.columns || w2.columns != w1.rows) {
            result.errors.emplace_back(
                "DeepSeek MoE page W1/W3/W2 shapes or native encoding metadata are invalid");
            return false;
        }
        if (hidden_columns == 0U) {
            hidden_columns = w1.columns;
            intermediate_columns = w1.rows;
        } else if (hidden_columns != w1.columns ||
                   intermediate_columns != w1.rows) {
            result.errors.emplace_back(
                "DeepSeek MoE page experts do not share one exact activation shape");
            return false;
        }
        return true;
    };

    // A transformed expert arrives as its TP shards instead of a triplet. The
    // shards carry the same values, so they must agree with the triplets on
    // the activation shape they all share.
    std::uint64_t shard_intermediate = 0U;
    const auto validate_shards = [&](const CudaDeepSeekMoeRowGroup& group) {
        for (const auto* shard : group.tiled_shards) {
            if (shard == nullptr || !shard->valid()) {
                result.errors.emplace_back(
                    "DeepSeek MoE page command contains an invalid expert shard");
                return false;
            }
            if (shard->impl_->device != device ||
                shard->impl_->descriptor.encoding !=
                    CudaWeightEncoding::Fp4E2m1Tiled32) {
                result.errors.emplace_back(
                    "DeepSeek MoE page expert shard has the wrong device or encoding");
                return false;
            }
        }
        const auto& first = group.tiled_shards.front()->impl_->descriptor;
        for (const auto* shard : group.tiled_shards) {
            const auto& descriptor = shard->impl_->descriptor;
            if (descriptor.rows != first.rows ||
                descriptor.columns != first.columns) {
                result.errors.emplace_back(
                    "DeepSeek MoE page expert shards disagree on shape");
                return false;
            }
        }
        const auto shards =
            static_cast<std::uint64_t>(group.tiled_shards.size());
        if (hidden_columns == 0U) {
            hidden_columns = first.rows;
            intermediate_columns = first.columns * shards;
        } else if (hidden_columns != first.rows ||
                   intermediate_columns != first.columns * shards) {
            result.errors.emplace_back(
                "DeepSeek MoE page experts do not share one exact activation shape");
            return false;
        }
        if (shard_intermediate == 0U) {
            shard_intermediate = first.columns;
        } else if (shard_intermediate != first.columns) {
            result.errors.emplace_back(
                "DeepSeek MoE page expert shards disagree on width");
            return false;
        }
        return true;
    };

    std::uint64_t work_count = 0U;
    for (const auto& group : groups) {
        const bool tiled = group.tiled_shards.front() != nullptr;
        if (tiled ? !validate_shards(group)
                  : !validate_triplet(group.w1, group.w3, group.w2,
                                      CudaWeightEncoding::Fp4E2m1Group32)) {
            return result;
        }
        if (group.rows.empty() || group.rows.size() != group.coefficients.size()) {
            result.errors.emplace_back(
                "DeepSeek MoE page group rows and coefficients must be non-empty and equal in size");
            return result;
        }
        for (const auto row : group.rows) {
            if (row >= hidden_rows) {
                result.errors.emplace_back(
                    "DeepSeek MoE page group row index is out of range");
                return result;
            }
        }
        for (const auto coefficient : group.coefficients) {
            if (!std::isfinite(coefficient)) {
                result.errors.emplace_back(
                    "DeepSeek MoE page group coefficient is invalid");
                return result;
            }
        }
        work_count += group.rows.size();
    }
    if (shared != nullptr) {
        if (!validate_triplet(shared->w1, shared->w3, shared->w2,
                              CudaWeightEncoding::Fp8E4m3Block128) ||
            shared->coefficient != 1.0F) {
            if (result.ok()) {
                result.errors.emplace_back(
                    "DeepSeek MoE page shared expert coefficient is invalid");
            }
            return result;
        }
        if (shared_rows.empty()) {
            result.errors.emplace_back(
                "DeepSeek MoE page shared expert requires at least one row");
            return result;
        }
        for (const auto row : shared_rows) {
            if (row >= hidden_rows) {
                result.errors.emplace_back(
                    "DeepSeek MoE page shared row index is out of range");
                return result;
            }
        }
    } else if (!shared_rows.empty()) {
        result.errors.emplace_back(
            "DeepSeek MoE page shared rows were supplied without a shared expert");
        return result;
    }
    const auto shared_count =
        static_cast<std::uint64_t>(shared == nullptr ? 0U : shared_rows.size());
    if (work_count == 0U && shared_count == 0U) {
        result.errors.emplace_back("DeepSeek MoE page command has no work");
        return result;
    }
    if (hidden.size() !=
            static_cast<std::size_t>(hidden_rows) * hidden_columns ||
        hidden_columns > std::numeric_limits<unsigned int>::max() ||
        intermediate_columns > std::numeric_limits<unsigned int>::max()) {
        result.errors.emplace_back(
            "DeepSeek MoE page hidden rows or expert dimensions are incompatible");
        return result;
    }
    if (!std::all_of(hidden.begin(), hidden.end(),
                     [](float value) { return std::isfinite(value); })) {
        result.errors.emplace_back(
            "DeepSeek MoE page hidden rows contain a non-finite value");
        return result;
    }

    const auto activation_slots = work_count + shared_count;
    std::uint64_t hidden_bytes = 0U;
    std::uint64_t activation_bytes = 0U;
    std::uint64_t output_bytes = 0U;
    std::uint64_t row_bytes = 0U;
    std::uint64_t coefficient_bytes = 0U;
    std::uint64_t group_bytes = 0U;
    std::uint64_t shared_row_bytes = 0U;
    if (!checked_bytes(hidden_rows, hidden_columns, sizeof(float), hidden_bytes) ||
        !checked_bytes(activation_slots, intermediate_columns, sizeof(float),
                       activation_bytes) ||
        !checked_bytes(activation_slots, hidden_columns, sizeof(float),
                       output_bytes) ||
        !checked_bytes(work_count, 1U, sizeof(std::uint32_t), row_bytes) ||
        !checked_bytes(work_count, 1U, sizeof(float), coefficient_bytes) ||
        !checked_bytes(groups.size(), 1U, sizeof(DeepSeekFp4PageGroup),
                       group_bytes) ||
        !checked_bytes(std::max<std::uint64_t>(shared_count, 1U), 1U,
                       sizeof(std::uint32_t), shared_row_bytes) ||
        hidden_bytes > std::numeric_limits<std::size_t>::max() ||
        activation_bytes > std::numeric_limits<std::size_t>::max() ||
        output_bytes > std::numeric_limits<std::size_t>::max() ||
        output_bytes > std::numeric_limits<std::uint64_t>::max() -
                           sizeof(unsigned int)) {
        result.errors.emplace_back("DeepSeek MoE page workspace size overflows");
        return result;
    }
    const auto host_staging_bytes = output_bytes + sizeof(unsigned int);
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for DeepSeek MoE page");
    }

    std::uint64_t allocation_calls = 0U;
    std::uint64_t allocation_bytes = 0U;
    const auto ensure_bytes = [&](void*& pointer, std::uint64_t& capacity,
                                  std::uint64_t required,
                                  const char* operation) {
        if (required <= capacity) return true;
        if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
        pointer = nullptr;
        capacity = 0U;
        if (const auto status =
                cudaMalloc(&pointer, static_cast<std::size_t>(required));
            status != cudaSuccess) {
            result = cuda_error(status, operation);
            return false;
        }
        capacity = required;
        ++allocation_calls;
        allocation_bytes += required;
        return true;
    };
    const auto ensure_floats = [&](float*& pointer, std::uint64_t& capacity,
                                   std::uint64_t required,
                                   const char* operation) {
        auto* raw = static_cast<void*>(pointer);
        const auto ok = ensure_bytes(raw, capacity, required, operation);
        pointer = static_cast<float*>(raw);
        return ok;
    };
    const auto ensure_indices = [&](std::uint32_t*& pointer,
                                    std::uint64_t& capacity,
                                    std::uint64_t required,
                                    const char* operation) {
        auto* raw = static_cast<void*>(pointer);
        const auto ok = ensure_bytes(raw, capacity, required, operation);
        pointer = static_cast<std::uint32_t*>(raw);
        return ok;
    };
    if (!ensure_floats(state.moe_hidden, state.moe_hidden_bytes, hidden_bytes,
                       "allocate DeepSeek MoE page hidden workspace") ||
        !ensure_floats(state.moe_activations, state.moe_activation_bytes,
                       activation_bytes,
                       "allocate DeepSeek MoE page activation workspace") ||
        !ensure_floats(state.moe_output, state.moe_output_bytes, output_bytes,
                       "allocate DeepSeek MoE page output workspace") ||
        !ensure_indices(state.moe_page_rows, state.moe_page_rows_bytes,
                        row_bytes, "allocate DeepSeek MoE page row list") ||
        !ensure_floats(state.moe_page_coefficients,
                       state.moe_page_coefficient_bytes, coefficient_bytes,
                       "allocate DeepSeek MoE page coefficient list") ||
        !ensure_bytes(state.moe_page_groups, state.moe_page_group_bytes,
                      group_bytes, "allocate DeepSeek MoE page group table") ||
        !ensure_indices(state.moe_page_shared_rows,
                        state.moe_page_shared_row_bytes, shared_row_bytes,
                        "allocate DeepSeek MoE page shared row list")) {
        return result;
    }
    if (state.moe_bf16_silu == nullptr) {
        // Same table as the single-row command builds. Held in a vector rather
        // than a function-local static array because a second
        // `static const std::array<float, N>` in this translation unit does not
        // survive nvcc's host pass.
        constexpr std::size_t silu_entries = 1U << 16U;
        const std::size_t silu_bytes = silu_entries * sizeof(float);
        std::vector<float> silu_table(silu_entries);
        for (std::size_t index = 0U; index < silu_entries; ++index) {
            const auto bits = static_cast<std::uint32_t>(index) << 16U;
            const float value = std::bit_cast<float>(bits);
            silu_table[index] = std::isfinite(value) ? silu_f32(value) : value;
        }
        if (const auto status = cudaMalloc(&state.moe_bf16_silu, silu_bytes);
            status != cudaSuccess) {
            return cuda_error(status, "allocate DeepSeek BF16 SiLU page table");
        }
        if (const auto status = cudaMemcpy(state.moe_bf16_silu,
                                           silu_table.data(), silu_bytes,
                                           cudaMemcpyHostToDevice);
            status != cudaSuccess) {
            static_cast<void>(cudaFree(state.moe_bf16_silu));
            state.moe_bf16_silu = nullptr;
            return cuda_error(status, "upload DeepSeek BF16 SiLU page table");
        }
        ++allocation_calls;
        allocation_bytes += silu_bytes;
    }
    if (state.moe_error == nullptr) {
        if (const auto status = cudaMalloc(&state.moe_error, sizeof(unsigned int));
            status != cudaSuccess) {
            return cuda_error(status, "allocate DeepSeek MoE page error flag");
        }
        ++allocation_calls;
        allocation_bytes += sizeof(unsigned int);
    }
    if (host_staging_bytes > state.moe_host_staging_bytes) {
        if (state.moe_host_staging != nullptr) {
            static_cast<void>(cudaFreeHost(state.moe_host_staging));
        }
        state.moe_host_staging = nullptr;
        state.moe_host_staging_bytes = 0U;
        if (const auto status = cudaMallocHost(
                &state.moe_host_staging,
                static_cast<std::size_t>(host_staging_bytes));
            status != cudaSuccess) {
            return cuda_error(status, "allocate DeepSeek MoE page host staging");
        }
        state.moe_host_staging_bytes = host_staging_bytes;
        ++allocation_calls;
        allocation_bytes += host_staging_bytes;
    }

    std::vector<std::uint32_t> host_rows;
    std::vector<float> host_coefficients;
    std::vector<DeepSeekFp4PageGroup> host_groups;
    host_rows.reserve(static_cast<std::size_t>(work_count));
    host_coefficients.reserve(static_cast<std::size_t>(work_count));
    host_groups.reserve(groups.size());
    std::uint32_t maximum_group_rows = 0U;
    for (const auto& group : groups) {
        DeepSeekFp4PageGroup entry;
        if (group.tiled_shards.front() != nullptr) {
            for (std::size_t shard = 0U; shard < group.tiled_shards.size();
                 ++shard) {
                entry.tiled[shard] = static_cast<const unsigned char*>(
                    group.tiled_shards[shard]->impl_->weights);
            }
            entry.shard_intermediate =
                static_cast<std::uint32_t>(shard_intermediate);
        } else {
            entry.w1_weights = static_cast<const unsigned char*>(group.w1->impl_->weights);
            entry.w1_scales = static_cast<const unsigned char*>(group.w1->impl_->scales);
            entry.w3_weights = static_cast<const unsigned char*>(group.w3->impl_->weights);
            entry.w3_scales = static_cast<const unsigned char*>(group.w3->impl_->scales);
            entry.w2_weights = static_cast<const unsigned char*>(group.w2->impl_->weights);
            entry.w2_scales = static_cast<const unsigned char*>(group.w2->impl_->scales);
        }
        entry.row_offset = static_cast<std::uint32_t>(host_rows.size());
        entry.row_count = static_cast<std::uint32_t>(group.rows.size());
        maximum_group_rows = std::max(maximum_group_rows, entry.row_count);
        host_rows.insert(host_rows.end(), group.rows.begin(), group.rows.end());
        host_coefficients.insert(host_coefficients.end(),
                                 group.coefficients.begin(),
                                 group.coefficients.end());
        host_groups.push_back(entry);
    }

    state.moe_weights.clear();
    state.moe_weights.reserve((groups.size() + 1U) * 3U);
    for (const auto& group : groups) {
        if (group.tiled_shards.front() != nullptr) {
            for (const auto* shard : group.tiled_shards) {
                state.moe_weights.push_back(shard->impl_);
            }
            continue;
        }
        state.moe_weights.push_back(group.w1->impl_);
        state.moe_weights.push_back(group.w3->impl_);
        state.moe_weights.push_back(group.w2->impl_);
    }
    if (shared != nullptr) {
        state.moe_weights.push_back(shared->w1->impl_);
        state.moe_weights.push_back(shared->w3->impl_);
        state.moe_weights.push_back(shared->w2->impl_);
    }

    state.moe_hidden_columns = hidden_columns;
    state.moe_intermediate_columns = intermediate_columns;
    state.moe_rows = 1U;
    state.moe_routed_count = static_cast<std::uint32_t>(work_count);
    state.moe_shared_rows = static_cast<std::uint32_t>(shared_count);
    state.moe_page_work_count = static_cast<std::uint32_t>(work_count);
    state.moe_page_shared_count = static_cast<std::uint32_t>(shared_count);
    state.moe_has_shared = shared != nullptr;
    state.moe_shared_phase_timing_valid = false;
    state.moe_host_join = false;
    state.moe_output_to_mhc = false;
    state.moe_host_callback = {};
    state.moe_kernel_launches = 0U;
    state.moe_in_flight = true;
    state.moe_poisoned = false;
    auto abort_enqueue = [&](cudaError_t status, const char* operation) {
        result = cuda_error(status, operation);
        const auto drain_status = cudaStreamSynchronize(state.stream);
        if (drain_status != cudaSuccess) {
            result.errors.emplace_back(
                std::string("drain failed DeepSeek MoE page enqueue: ") +
                cudaGetErrorString(drain_status));
            state.moe_poisoned = true;
        } else {
            state.moe_in_flight = false;
            state.moe_weights.clear();
        }
    };

    if (auto status = cudaEventRecord(state.moe_start, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record DeepSeek MoE page start");
        return result;
    }
    if (auto status = cudaMemsetAsync(state.moe_error, 0, sizeof(unsigned int),
                                      state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "reset DeepSeek MoE page error flag");
        return result;
    }
    const auto upload = [&](void* destination, const void* source,
                            std::uint64_t bytes, const char* operation) {
        if (bytes == 0U) return true;
        if (auto status = cudaMemcpyAsync(destination, source,
                                          static_cast<std::size_t>(bytes),
                                          cudaMemcpyHostToDevice, state.stream);
            status != cudaSuccess) {
            abort_enqueue(status, operation);
            return false;
        }
        return true;
    };
    if (!upload(state.moe_hidden, hidden.data(), hidden_bytes,
                "upload DeepSeek MoE page hidden rows") ||
        !upload(state.moe_page_rows, host_rows.data(), row_bytes,
                "upload DeepSeek MoE page row list") ||
        !upload(state.moe_page_coefficients, host_coefficients.data(),
                coefficient_bytes,
                "upload DeepSeek MoE page coefficient list") ||
        !upload(state.moe_page_groups, host_groups.data(), group_bytes,
                "upload DeepSeek MoE page group table") ||
        (shared != nullptr &&
         !upload(state.moe_page_shared_rows, shared_rows.data(),
                 static_cast<std::uint64_t>(shared_rows.size()) *
                     sizeof(std::uint32_t),
                 "upload DeepSeek MoE page shared row list"))) {
        return result;
    }
    if (auto status = cudaEventRecord(state.moe_hidden_uploaded, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record DeepSeek MoE page hidden upload");
        return result;
    }

    constexpr unsigned int threads = 256U;
    // The quantizer takes its row from blockIdx.y, so grid.y is the row count.
    const dim3 hidden_quantize_grid(
        static_cast<unsigned int>((hidden_columns + 127U) / 128U), hidden_rows,
        1U);
    quantize_activation_e4m3_kernel<<<hidden_quantize_grid, 128U, 0U,
                                      state.stream>>>(
        state.moe_hidden, hidden_columns, hidden_rows);
    ++state.moe_kernel_launches;
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        abort_enqueue(status, "launch DeepSeek MoE page hidden quantization");
        return result;
    }

    const auto* device_groups =
        static_cast<const DeepSeekFp4PageGroup*>(state.moe_page_groups);
    if (!groups.empty()) {
        // Derived rather than read off a descriptor: a transformed group has
        // no canonical triplet to read, and the canonical one would carry
        // exactly these values anyway.
        const auto gate_packed_columns = (hidden_columns + 1U) / 2U;
        const auto gate_scale_columns = (hidden_columns + 31U) / 32U;
        const auto down_packed_columns = (intermediate_columns + 1U) / 2U;
        const auto down_scale_columns = (intermediate_columns + 31U) / 32U;
        const bool tiled = shard_intermediate != 0U;
        const auto row_tile =
            tiled ? kDeepSeekTiledRowTile : kDeepSeekPageRowTile;
        const auto row_tiles = static_cast<unsigned int>(
            (maximum_group_rows + row_tile - 1U) / row_tile);
        // The transformed kernel gives one warp a whole 32-row transform block
        // and one output row to each of its lanes.
        const auto tiled_output_blocks = static_cast<unsigned int>(
            (intermediate_columns / 32U + kDeepSeekTiledWarps - 1U) /
            kDeepSeekTiledWarps);
        const dim3 gate_grid(
            tiled ? tiled_output_blocks
                  : static_cast<unsigned int>(intermediate_columns),
            static_cast<unsigned int>(groups.size()), row_tiles);
        if (tiled) {
            deepseek_fp4_tiled_page_gate_up_kernel<<<
                gate_grid, threads, 0U, state.stream>>>(
                state.moe_activations, state.moe_hidden, state.moe_page_rows,
                state.moe_page_coefficients, device_groups,
                static_cast<std::uint32_t>(groups.size()), hidden_columns,
                intermediate_columns, swiglu_limit, state.moe_bf16_silu,
                state.moe_error);
        } else {
            deepseek_fp4_page_gate_up_kernel<<<gate_grid, threads, 0U, state.stream>>>(
                state.moe_activations, state.moe_hidden, state.moe_page_rows,
                state.moe_page_coefficients, device_groups,
                static_cast<std::uint32_t>(groups.size()), hidden_columns,
                intermediate_columns, gate_packed_columns, gate_scale_columns,
                swiglu_limit, state.moe_bf16_silu, state.moe_error);
        }
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek FP4 page W1/W3 SwiGLU");
            return result;
        }
        const dim3 activation_grid(
            static_cast<unsigned int>((intermediate_columns + 127U) / 128U),
            static_cast<unsigned int>(work_count), 1U);
        quantize_activation_e4m3_kernel<<<activation_grid, 128U, 0U,
                                          state.stream>>>(
            state.moe_activations, intermediate_columns,
            static_cast<std::uint32_t>(work_count));
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status,
                          "launch DeepSeek page routed activation quantization");
            return result;
        }
        const auto tiled_down_blocks = static_cast<unsigned int>(
            (hidden_columns / 32U + kDeepSeekTiledWarps - 1U) /
            kDeepSeekTiledWarps);
        const dim3 down_grid(
            tiled ? tiled_down_blocks
                  : static_cast<unsigned int>(hidden_columns),
            static_cast<unsigned int>(groups.size()), row_tiles);
        if (tiled) {
            deepseek_fp4_tiled_page_down_kernel<<<
                down_grid, threads, 0U, state.stream>>>(
                state.moe_output, state.moe_activations, device_groups,
                static_cast<std::uint32_t>(groups.size()),
                intermediate_columns, hidden_columns);
        } else {
            deepseek_fp4_page_down_kernel<<<down_grid, threads, 0U, state.stream>>>(
                state.moe_output, state.moe_activations, device_groups,
                static_cast<std::uint32_t>(groups.size()), intermediate_columns,
                hidden_columns, down_packed_columns, down_scale_columns);
        }
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek FP4 page W2");
            return result;
        }
    }

    if (shared != nullptr) {
        const auto& w1 = shared->w1->impl_->descriptor;
        const auto& w2 = shared->w2->impl_->descriptor;
        float* shared_activations =
            state.moe_activations + work_count * intermediate_columns;
        float* shared_output = state.moe_output + work_count * hidden_columns;
        const auto shared_tiles = static_cast<unsigned int>(
            (shared_count + kDeepSeekPageRowTile - 1U) / kDeepSeekPageRowTile);
        const dim3 shared_gate_grid(
            static_cast<unsigned int>(intermediate_columns), shared_tiles, 1U);
        deepseek_fp8_page_gate_up_kernel<<<shared_gate_grid, threads, 0U,
                                           state.stream>>>(
            shared_activations, state.moe_hidden, state.moe_page_shared_rows,
            static_cast<std::uint32_t>(shared_count),
            static_cast<const unsigned char*>(shared->w1->impl_->weights),
            static_cast<const unsigned char*>(shared->w1->impl_->scales),
            static_cast<const unsigned char*>(shared->w3->impl_->weights),
            static_cast<const unsigned char*>(shared->w3->impl_->scales),
            hidden_columns, intermediate_columns, w1.scale_columns,
            swiglu_limit, state.moe_bf16_silu, state.moe_error);
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek shared FP8 page W1/W3 SwiGLU");
            return result;
        }
        const dim3 shared_activation_grid(
            static_cast<unsigned int>((intermediate_columns + 127U) / 128U),
            static_cast<unsigned int>(shared_count), 1U);
        quantize_activation_e4m3_kernel<<<shared_activation_grid, 128U, 0U,
                                          state.stream>>>(
            shared_activations, intermediate_columns,
            static_cast<std::uint32_t>(shared_count));
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status,
                          "launch DeepSeek page shared activation quantization");
            return result;
        }
        const dim3 shared_down_grid(static_cast<unsigned int>(hidden_columns),
                                    shared_tiles, 1U);
        deepseek_fp8_page_down_kernel<<<shared_down_grid, threads, 0U,
                                        state.stream>>>(
            shared_output, shared_activations,
            static_cast<std::uint32_t>(shared_count),
            static_cast<const unsigned char*>(shared->w2->impl_->weights),
            static_cast<const unsigned char*>(shared->w2->impl_->scales),
            intermediate_columns, hidden_columns, w2.scale_columns);
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek shared FP8 page W2");
            return result;
        }
    }

    if (auto status = cudaEventRecord(state.moe_kernel_finished, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record DeepSeek MoE page kernel completion");
        return result;
    }
    {
        std::scoped_lock lock(impl_->mutex);
        auto& device_stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        device_stats.activation_h2d_bytes += hidden_bytes;
        device_stats.matmul_calls += 3U * (groups.size() + (shared ? 1U : 0U));
        device_stats.workspace_allocation_calls += allocation_calls;
        device_stats.workspace_allocation_bytes += allocation_bytes;
        ++device_stats.deepseek_moe_calls;
        device_stats.deepseek_moe_kernel_launches += state.moe_kernel_launches;
        ++device_stats.deepseek_moe_h2d_transfers;
        device_stats.deepseek_moe_h2d_bytes += hidden_bytes;
    }
    return result;
}

ValidationResult CudaBackend::collect_deepseek_moe_rows(
    int device, std::span<float> routed_output, std::span<float> shared_output) {
    return collect_deepseek_moe(device, routed_output, shared_output);
}

ValidationResult CudaBackend::enqueue_moe(
    int device, std::span<const float> hidden, std::uint32_t rows,
    std::span<const CudaMoeExpert> routed, const CudaMoeExpert* shared,
    float swiglu_limit) {
    return enqueue_moe_impl(device, hidden, rows, routed, shared,
                            swiglu_limit, false, {});
}

ValidationResult CudaBackend::enqueue_glm53_moe_from_mhc(
    int device, std::span<const CudaMoeExpert> routed,
    const CudaMoeExpert& shared, std::span<const float> coefficients,
    float swiglu_limit) {
    return enqueue_moe_impl(device, {}, 1U, routed, &shared, swiglu_limit,
                            true, coefficients);
}

ValidationResult CudaBackend::enqueue_moe_impl(
    int device, std::span<const float> hidden, std::uint32_t rows,
    std::span<const CudaMoeExpert> routed, const CudaMoeExpert* shared,
    float swiglu_limit, bool mhc_source_destination,
    std::span<const float> routed_coefficients) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back("MoE command targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    if (state.moe_in_flight ||
        (mhc_source_destination &&
         (!state.dsv4_mhc_supported || state.dsv4_mhc_stage != 1U ||
          state.dsv4_mhc_workspace == nullptr ||
          state.dsv4_mhc_branch_ready || state.dsv4_mhc_failed))) {
        result.errors.emplace_back("MoE workspace already has an in-flight command");
        return result;
    }
    const auto expert_count = routed.size() + (shared == nullptr ? 0U : 1U);
    if (rows == 0U || expert_count == 0U || expert_count > kMaxMoeExperts ||
        routed.size() > kMaxRoutedMoeExperts ||
        (mhc_source_destination &&
         (rows != 1U || shared == nullptr ||
          routed_coefficients.size() != routed.size())) ||
        (!mhc_source_destination && !routed_coefficients.empty())) {
        result.errors.emplace_back("MoE command has an unsupported row or expert count");
        return result;
    }

    // The batch is single-encoding. The first expert's gate fixes it and every
    // other weight must agree, so a mixed batch is rejected rather than
    // silently dispatched to the wrong decode rule.
    const auto* first_gate = routed.empty() ? (shared == nullptr ? nullptr
                                                                 : shared->gate)
                                            : routed.front().gate;
    if (first_gate == nullptr || !first_gate->valid()) {
        result.errors.emplace_back("MoE command has no valid leading expert");
        return result;
    }
    const auto batch_encoding = first_gate->impl_->descriptor.encoding;
    const bool nvfp4_batch = batch_encoding == CudaWeightEncoding::Nvfp4Group16;
    const bool mxfp4_batch =
        batch_encoding == CudaWeightEncoding::Fp4E2m1Group32;
    const bool fp8_f32_batch =
        batch_encoding == CudaWeightEncoding::Fp8E4m3Block128F32;
    const bool plain_batch = batch_encoding == CudaWeightEncoding::Plain;
    if (!nvfp4_batch && !mxfp4_batch && !fp8_f32_batch && !plain_batch &&
        batch_encoding != CudaWeightEncoding::OffsetPackedInt4) {
        result.errors.emplace_back("MoE command has an unsupported weight encoding");
        return result;
    }
    if (fp8_f32_batch &&
        (!std::isfinite(swiglu_limit) || swiglu_limit <= 0.0F)) {
        result.errors.emplace_back(
            "F32-scaled FP8 MoE requires a positive finite SwiGLU limit");
        return result;
    }

    std::uint64_t hidden_columns = 0U;
    std::uint64_t intermediate_columns = 0U;
    const auto validate_expert = [&](const CudaMoeExpert& expert,
                                     bool shared_expert) {
        const std::array<const CudaWeight*, 3> weights{
            expert.gate, expert.up, expert.down};
        for (const auto* weight : weights) {
            const bool compatible =
                weight != nullptr && weight->valid() &&
                weight->impl_->device == device &&
                weight->impl_->descriptor.encoding == batch_encoding &&
                (nvfp4_batch
                     ? (weight->impl_->descriptor.dtype == SafetensorsDtype::U8 &&
                        weight->impl_->descriptor.group_size == 16U &&
                        std::isfinite(weight->impl_->descriptor.global_scale) &&
                        weight->impl_->descriptor.global_scale > 0.0F)
                 : mxfp4_batch
                     ? (weight->impl_->descriptor.dtype == SafetensorsDtype::I8 &&
                        weight->impl_->descriptor.group_size == 32U)
                 : fp8_f32_batch
                     ? (weight->impl_->descriptor.dtype ==
                            SafetensorsDtype::F8E4M3 &&
                        weight->impl_->descriptor.group_size == 128U)
                 : plain_batch
                     ? weight->impl_->descriptor.dtype == SafetensorsDtype::Bf16
                     : (weight->impl_->descriptor.dtype == SafetensorsDtype::I32 &&
                        weight->impl_->descriptor.group_size == 128U));
            if (!compatible) {
                result.errors.emplace_back(
                    "MoE command contains an incompatible CUDA weight");
                return false;
            }
        }
        const auto& gate = expert.gate->impl_->descriptor;
        const auto& up = expert.up->impl_->descriptor;
        const auto& down = expert.down->impl_->descriptor;
        const auto expected_down_packed = fp8_f32_batch
            ? down.columns
            : (nvfp4_batch || mxfp4_batch)
                ? (down.columns + 1U) / 2U : (down.columns + 7U) / 8U;
        const auto expected_down_scales = nvfp4_batch
            ? (down.columns + 15U) / 16U
            : mxfp4_batch ? (down.columns + 31U) / 32U
                           : (down.columns + 127U) / 128U;
        const bool packing_valid = plain_batch ||
            (gate.packed_columns == up.packed_columns &&
             gate.scale_columns == up.scale_columns &&
             down.packed_columns == expected_down_packed &&
             down.scale_columns == expected_down_scales);
        if (gate.rows == 0U || gate.columns == 0U ||
            up.rows != gate.rows || up.columns != gate.columns ||
            down.rows != gate.columns || down.columns != gate.rows ||
            !packing_valid) {
            result.errors.emplace_back("MoE gate/up/down shapes are incompatible");
            return false;
        }
        // The NVFP4 kernels leave the routing coefficient to the caller,
        // because scaling before the down projection is not float-equal to
        // scaling after it and Laguna's reference scales after.
        if (!std::isfinite(expert.coefficient) ||
            ((shared_expert || nvfp4_batch || mxfp4_batch || fp8_f32_batch ||
              plain_batch) &&
             expert.coefficient != 1.0F)) {
            result.errors.emplace_back("MoE expert coefficient is invalid");
            return false;
        }
        if (hidden_columns == 0U) {
            hidden_columns = gate.columns;
            intermediate_columns = gate.rows;
        } else if (hidden_columns != gate.columns ||
                   intermediate_columns != gate.rows) {
            result.errors.emplace_back(
                "MoE experts do not share one activation shape");
            return false;
        }
        return true;
    };
    for (const auto& expert : routed) {
        if (!validate_expert(expert, false)) return result;
    }
    if (shared != nullptr && !validate_expert(*shared, true)) return result;

    std::uint64_t hidden_elements = 0U;
    if (!checked_bytes(rows, hidden_columns, 1U, hidden_elements) ||
        (mhc_source_destination ? !hidden.empty()
                                : hidden.size() != hidden_elements) ||
        std::any_of(hidden.begin(), hidden.end(),
                    [](float value) { return !std::isfinite(value); }) ||
        std::any_of(routed_coefficients.begin(), routed_coefficients.end(),
                    [](float value) { return !std::isfinite(value); })) {
        result.errors.emplace_back("MoE hidden rows are incompatible");
        return result;
    }

    std::uint64_t hidden_bytes = 0U;
    std::uint64_t activation_rows = 0U;
    std::uint64_t activation_bytes = 0U;
    std::uint64_t output_bytes = 0U;
    if (!checked_bytes(rows, hidden_columns, sizeof(float), hidden_bytes) ||
        !checked_bytes(expert_count, rows, 1U, activation_rows) ||
        !checked_bytes(activation_rows, intermediate_columns, sizeof(float),
                       activation_bytes) ||
        !checked_bytes(activation_rows, hidden_columns, sizeof(float),
                       output_bytes) ||
        output_bytes > std::numeric_limits<std::uint64_t>::max() -
                           sizeof(unsigned int)) {
        result.errors.emplace_back("MoE workspace size overflows");
        return result;
    }
    if (hidden_columns > std::numeric_limits<unsigned int>::max() ||
        intermediate_columns > std::numeric_limits<unsigned int>::max() ||
        activation_rows > std::numeric_limits<unsigned int>::max()) {
        result.errors.emplace_back("MoE CUDA grid dimensions overflow");
        return result;
    }
    const auto host_staging_bytes = output_bytes + sizeof(unsigned int);
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for MoE");
    }

    std::uint64_t allocation_calls = 0U;
    std::uint64_t allocation_bytes = 0U;
    const auto ensure_workspace = [&](float*& pointer, std::uint64_t& capacity,
                                      std::uint64_t required,
                                      const char* operation) {
        if (required <= capacity) return true;
        if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
        pointer = nullptr;
        capacity = 0U;
        if (const auto status =
                cudaMalloc(&pointer, static_cast<std::size_t>(required));
            status != cudaSuccess) {
            result = cuda_error(status, operation);
            return false;
        }
        capacity = required;
        ++allocation_calls;
        allocation_bytes += required;
        return true;
    };
    if (!ensure_workspace(state.moe_hidden, state.moe_hidden_bytes, hidden_bytes,
                          "allocate MoE hidden workspace") ||
        !ensure_workspace(state.moe_activations, state.moe_activation_bytes,
                          activation_bytes,
                          "allocate MoE activation workspace") ||
        !ensure_workspace(state.moe_output, state.moe_output_bytes, output_bytes,
                          "allocate MoE output workspace")) {
        return result;
    }
    if (state.moe_error == nullptr) {
        if (const auto status = cudaMalloc(&state.moe_error, sizeof(unsigned int));
            status != cudaSuccess) {
            return cuda_error(status, "allocate MoE error flag");
        }
        ++allocation_calls;
        allocation_bytes += sizeof(unsigned int);
    }
    if (host_staging_bytes > state.moe_host_staging_bytes) {
        if (state.moe_host_staging != nullptr) {
            static_cast<void>(cudaFreeHost(state.moe_host_staging));
        }
        state.moe_host_staging = nullptr;
        state.moe_host_staging_bytes = 0U;
        if (const auto status = cudaMallocHost(
                &state.moe_host_staging,
                static_cast<std::size_t>(host_staging_bytes));
            status != cudaSuccess) {
            return cuda_error(status, "allocate MoE host staging");
        }
        state.moe_host_staging_bytes = host_staging_bytes;
        ++allocation_calls;
        allocation_bytes += host_staging_bytes;
    }

    PackedInt4MoeBatch batch;
    Nvfp4MoeBatch nvfp4_batch_data;
    Mxfp4MoeBatch mxfp4_batch_data;
    Fp8F32MoeBatch fp8_f32_batch_data;
    PlainBf16MoeBatch plain_batch_data;
    state.moe_weights.clear();
    state.moe_weights.reserve(expert_count * 3U);
    const auto append_expert = [&](const CudaMoeExpert& expert,
                                   std::size_t index) {
        if (plain_batch) {
            plain_batch_data.gate_weights[index] =
                static_cast<const __nv_bfloat16*>(expert.gate->impl_->weights);
            plain_batch_data.up_weights[index] =
                static_cast<const __nv_bfloat16*>(expert.up->impl_->weights);
            plain_batch_data.down_weights[index] =
                static_cast<const __nv_bfloat16*>(expert.down->impl_->weights);
        } else if (fp8_f32_batch) {
            fp8_f32_batch_data.gate_weights[index] =
                static_cast<const unsigned char*>(expert.gate->impl_->weights);
            fp8_f32_batch_data.gate_scales[index] =
                static_cast<const float*>(expert.gate->impl_->scales);
            fp8_f32_batch_data.up_weights[index] =
                static_cast<const unsigned char*>(expert.up->impl_->weights);
            fp8_f32_batch_data.up_scales[index] =
                static_cast<const float*>(expert.up->impl_->scales);
            fp8_f32_batch_data.down_weights[index] =
                static_cast<const unsigned char*>(expert.down->impl_->weights);
            fp8_f32_batch_data.down_scales[index] =
                static_cast<const float*>(expert.down->impl_->scales);
            fp8_f32_batch_data.coefficients[index] =
                index < routed_coefficients.size()
                    ? routed_coefficients[index] : 1.0F;
        } else if (nvfp4_batch) {
            nvfp4_batch_data.gate_weights[index] =
                static_cast<const unsigned char*>(expert.gate->impl_->weights);
            nvfp4_batch_data.gate_scales[index] =
                static_cast<const unsigned char*>(expert.gate->impl_->scales);
            nvfp4_batch_data.up_weights[index] =
                static_cast<const unsigned char*>(expert.up->impl_->weights);
            nvfp4_batch_data.up_scales[index] =
                static_cast<const unsigned char*>(expert.up->impl_->scales);
            nvfp4_batch_data.down_weights[index] =
                static_cast<const unsigned char*>(expert.down->impl_->weights);
            nvfp4_batch_data.down_scales[index] =
                static_cast<const unsigned char*>(expert.down->impl_->scales);
            nvfp4_batch_data.gate_global_scales[index] =
                expert.gate->impl_->descriptor.global_scale;
            nvfp4_batch_data.up_global_scales[index] =
                expert.up->impl_->descriptor.global_scale;
            nvfp4_batch_data.down_global_scales[index] =
                expert.down->impl_->descriptor.global_scale;
        } else if (mxfp4_batch) {
            mxfp4_batch_data.gate_weights[index] =
                static_cast<const unsigned char*>(expert.gate->impl_->weights);
            mxfp4_batch_data.gate_scales[index] =
                static_cast<const unsigned char*>(expert.gate->impl_->scales);
            mxfp4_batch_data.up_weights[index] =
                static_cast<const unsigned char*>(expert.up->impl_->weights);
            mxfp4_batch_data.up_scales[index] =
                static_cast<const unsigned char*>(expert.up->impl_->scales);
            mxfp4_batch_data.down_weights[index] =
                static_cast<const unsigned char*>(expert.down->impl_->weights);
            mxfp4_batch_data.down_scales[index] =
                static_cast<const unsigned char*>(expert.down->impl_->scales);
        } else {
            batch.gate_weights[index] = static_cast<const std::uint32_t*>(
                expert.gate->impl_->weights);
            batch.gate_scales[index] = static_cast<const __nv_bfloat16*>(
                expert.gate->impl_->scales);
            batch.up_weights[index] = static_cast<const std::uint32_t*>(
                expert.up->impl_->weights);
            batch.up_scales[index] = static_cast<const __nv_bfloat16*>(
                expert.up->impl_->scales);
            batch.down_weights[index] = static_cast<const std::uint32_t*>(
                expert.down->impl_->weights);
            batch.down_scales[index] = static_cast<const __nv_bfloat16*>(
                expert.down->impl_->scales);
            batch.coefficients[index] = expert.coefficient;
        }
        state.moe_weights.push_back(expert.gate->impl_);
        state.moe_weights.push_back(expert.up->impl_);
        state.moe_weights.push_back(expert.down->impl_);
    };
    for (std::size_t index = 0U; index < routed.size(); ++index) {
        append_expert(routed[index], index);
    }
    if (shared != nullptr) append_expert(*shared, routed.size());
    batch.count = static_cast<std::uint32_t>(expert_count);
    batch.rows = rows;
    nvfp4_batch_data.count = batch.count;
    nvfp4_batch_data.rows = rows;
    mxfp4_batch_data.count = batch.count;
    mxfp4_batch_data.rows = rows;
    fp8_f32_batch_data.count = batch.count;
    fp8_f32_batch_data.rows = rows;
    plain_batch_data.count = batch.count;
    plain_batch_data.rows = rows;

    bool fp8_f32_prepacked = fp8_f32_batch;
    bool fp8_f32_any_prepacked = false;
    if (fp8_f32_batch) {
        for (std::uint32_t index = 0U; index < batch.count; ++index) {
            const auto* expert = index < routed.size() ? &routed[index] : shared;
            const bool ready = expert->gate->impl_->fragment_prepacked &&
                               expert->up->impl_->fragment_prepacked &&
                               expert->down->impl_->fragment_prepacked &&
                regfed_fp8_shape_admissible(
                    expert->gate->impl_->descriptor.rows,
                    expert->gate->impl_->descriptor.columns) &&
                regfed_fp8_shape_admissible(
                    expert->up->impl_->descriptor.rows,
                    expert->up->impl_->descriptor.columns) &&
                regfed_fp8_shape_admissible(
                    expert->down->impl_->descriptor.rows,
                    expert->down->impl_->descriptor.columns);
            fp8_f32_prepacked = fp8_f32_prepacked && ready;
            fp8_f32_any_prepacked = fp8_f32_any_prepacked ||
                expert->gate->impl_->fragment_prepacked ||
                expert->up->impl_->fragment_prepacked ||
                expert->down->impl_->fragment_prepacked;
        }
        if (fp8_f32_any_prepacked && !fp8_f32_prepacked) {
            result.errors.emplace_back(
                "F32-scaled FP8 MoE batch mixes fragment-prepacked and "
                "canonical experts");
            return result;
        }
        if (fp8_f32_prepacked && rows > kRegfedMaxM) {
            result.errors.emplace_back(
                "F32-scaled FP8 MoE batch is fragment-prepacked but exceeds "
                "the register-fed row width");
            return result;
        }
    }
    const bool fp8_f32_regfed =
        fp8_f32_prepacked && regfed_matmul_enabled() &&
        state.fp8_f32_register_fed_supported;

    state.moe_hidden_columns = hidden_columns;
    state.moe_intermediate_columns = intermediate_columns;
    state.moe_rows = rows;
    // The generic command's shared expert produces one row per input row, so
    // collection sizes its download from the same count.
    state.moe_shared_rows = rows;
    state.moe_routed_count = static_cast<std::uint32_t>(routed.size());
    state.moe_has_shared = shared != nullptr;
    state.moe_shared_phase_timing_valid = false;
    state.moe_host_join = false;
    state.moe_output_to_mhc = mhc_source_destination;
    state.moe_host_callback = {};
    state.moe_kernel_launches = 0U;
    state.moe_in_flight = true;
    state.moe_poisoned = false;
    const auto abort_enqueue = [&](cudaError_t status, const char* operation) {
        result = cuda_error(status, operation);
        const auto drain_status = cudaStreamSynchronize(state.stream);
        if (drain_status != cudaSuccess) {
            result.errors.emplace_back(
                std::string("drain failed MoE enqueue: ") +
                cudaGetErrorString(drain_status));
            state.moe_poisoned = true;
        } else {
            state.moe_in_flight = false;
            state.moe_weights.clear();
        }
    };

    if (fp8_f32_regfed) {
        const std::uint64_t hidden_compact = hidden_elements +
            static_cast<std::uint64_t>(rows) *
                (hidden_columns / 128U) * sizeof(float);
        const std::uint64_t activation_elements =
            activation_rows * intermediate_columns;
        const std::uint64_t activation_compact = activation_elements +
            activation_rows * (intermediate_columns / 128U) * sizeof(float);
        const auto compact_bytes = std::max(hidden_compact,
                                            activation_compact);
        if (regfed_grow(state.moe_regfed_compact,
                        state.moe_regfed_compact_bytes, compact_bytes, false,
                        state.stream) != cudaSuccess) {
            abort_enqueue(cudaErrorMemoryAllocation,
                          "allocate F32-scaled FP8 MoE compact workspace");
            return result;
        }
    }

    if (auto status = cudaEventRecord(state.moe_start, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record MoE start");
        return result;
    }
    if (auto status = cudaMemsetAsync(
            state.moe_error, 0, sizeof(unsigned int), state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "reset MoE error flag");
        return result;
    }
    if (mhc_source_destination) {
        constexpr std::uint32_t convert_threads = 256U;
        constexpr std::uint32_t convert_blocks =
            (kDsv4MhcHidden + convert_threads - 1U) / convert_threads;
        dsv4_bf16_to_fp32<<<convert_blocks, convert_threads, 0U,
                            state.stream>>>(
            state.dsv4_mhc_workspace->layer_input, state.moe_hidden,
            kDsv4MhcHidden);
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "convert resident GLM-5.3 MoE input");
            return result;
        }
    } else if (auto status = cudaMemcpyAsync(
                   state.moe_hidden, hidden.data(),
                   static_cast<std::size_t>(hidden_bytes),
                   cudaMemcpyHostToDevice, state.stream);
               status != cudaSuccess) {
        abort_enqueue(status, "upload MoE hidden rows");
        return result;
    }
    if (auto status = cudaEventRecord(state.moe_hidden_uploaded, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record MoE hidden upload");
        return result;
    }
    if (fp8_f32_regfed) {
        auto* compact_values =
            static_cast<unsigned char*>(state.moe_regfed_compact);
        auto* compact_scales = reinterpret_cast<float*>(
            compact_values + hidden_elements);
        const dim3 quantize_grid(
            static_cast<unsigned int>(hidden_columns / 128U), rows, 1U);
        quantize_activation_e4m3_f32_bytes_kernel<<<
            quantize_grid, 128U, 0U, state.stream>>>(
            compact_values, compact_scales, state.moe_hidden,
            hidden_columns, rows);
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status,
                          "compact F32-scaled FP8 MoE hidden activation");
            return result;
        }
    } else if (fp8_f32_batch) {
        const dim3 quantize_grid(
            static_cast<unsigned int>((hidden_columns + 127U) / 128U), rows,
            1U);
        quantize_activation_e4m3_f32_scale_kernel<<<
            quantize_grid, 128U, 0U, state.stream>>>(
            state.moe_hidden, hidden_columns, rows);
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch F32-scaled FP8 MoE hidden quantization");
            return result;
        }
    } else if (mxfp4_batch) {
        const dim3 quantize_grid(
            static_cast<unsigned int>((hidden_columns + 127U) / 128U), rows,
            1U);
        quantize_activation_e4m3_kernel<<<quantize_grid, 128U, 0U,
                                          state.stream>>>(
            state.moe_hidden, hidden_columns, rows);
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch MXFP4 MoE hidden quantization");
            return result;
        }
    }

    const auto& gate = (routed.empty() ? shared->gate : routed.front().gate)
                           ->impl_->descriptor;
    const auto& down = (routed.empty() ? shared->down : routed.front().down)
                           ->impl_->descriptor;
    constexpr unsigned int threads = 256U;
    const dim3 gate_grid(static_cast<unsigned int>(intermediate_columns),
                         static_cast<unsigned int>(activation_rows), 1U);
    constexpr unsigned int warps_per_block = 8U;
    const dim3 plain_gate_grid(
        static_cast<unsigned int>((intermediate_columns + warps_per_block - 1U) /
                                  warps_per_block),
        static_cast<unsigned int>(activation_rows), 1U);
    // Counted once per MoE command on the gate/up dispatch; the down kernel
    // mirrors the same branch, so counting both would double every entry.
    // Register-fed fused MoE. Fragment order replaces the canonical layout, so
    // the batch is all-or-nothing: a partially permuted batch is a defect, not
    // a mixed dispatch, and is refused rather than half-served.
    bool mxfp4_prepacked = mxfp4_batch;
    bool mxfp4_any_prepacked = false;
    if (mxfp4_batch) {
        for (std::uint32_t index = 0U; index < batch.count; ++index) {
            const auto* expert = index < routed.size() ? &routed[index] : shared;
            const bool ready = expert->gate->impl_->fragment_prepacked &&
                               expert->up->impl_->fragment_prepacked &&
                               expert->down->impl_->fragment_prepacked;
            mxfp4_prepacked = mxfp4_prepacked && ready;
            mxfp4_any_prepacked = mxfp4_any_prepacked ||
                                  expert->gate->impl_->fragment_prepacked ||
                                  expert->up->impl_->fragment_prepacked ||
                                  expert->down->impl_->fragment_prepacked;
        }
        if (mxfp4_any_prepacked && !mxfp4_prepacked) {
            abort_enqueue(cudaErrorInvalidValue,
                          "MXFP4 MoE batch mixes fragment-prepacked and "
                          "canonical experts");
            return result;
        }
        if (mxfp4_prepacked && rows > kRegfedMaxM) {
            // No hidden fallback: the scalar kernel would read fragment order
            // as canonical weights, which is silent corruption.
            abort_enqueue(cudaErrorInvalidValue,
                          "MXFP4 MoE batch is fragment-prepacked but the row "
                          "count exceeds the register-fed kernel's width");
            return result;
        }
    }
    const bool mxfp4_regfed = mxfp4_prepacked && regfed_matmul_enabled();
    record_cuda_matmul_route(
        plain_batch      ? CudaMatmulRoute::MoePlainBf16
        : fp8_f32_regfed ? CudaMatmulRoute::MoeFp8F32RegisterFed
        : fp8_f32_batch  ? CudaMatmulRoute::MoeFp8E4m3Block128F32
        : nvfp4_batch    ? CudaMatmulRoute::MoeNvfp4Group16
        : mxfp4_regfed   ? CudaMatmulRoute::MoeFp4RegisterFed
        : mxfp4_batch    ? CudaMatmulRoute::MoeFp4E2m1Group32
                         : CudaMatmulRoute::MoePackedInt4);
    if (plain_batch) {
        plain_bf16_moe_gate_up_kernel<<<plain_gate_grid, threads, 0U, state.stream>>>(
            state.moe_activations, state.moe_hidden, plain_batch_data,
            hidden_columns, intermediate_columns, state.moe_error);
    } else if (fp8_f32_regfed) {
        const auto experts = batch.count;
        const auto column_blocks = static_cast<std::uint32_t>(
            (rows + kRegfedTileM - 1U) / kRegfedTileM);
        const auto groups = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(rows, kRegfedTileM));
        const auto n_tiles =
            static_cast<std::uint32_t>(intermediate_columns / kRegfedTileN);
        const auto pairs =
            static_cast<std::uint32_t>(hidden_columns / 32U);
        std::uint32_t split = 1U;
        while (split < 16U && pairs % ((split * 2U) * 4U) == 0U &&
               static_cast<std::uint64_t>(experts) * n_tiles * split * 2U <=
                   4096U) {
            split *= 2U;
        }
        const std::uint64_t partial_bytes =
            static_cast<std::uint64_t>(experts) * intermediate_columns * rows *
            split * sizeof(float);
        const std::uint64_t fragment_total =
            (hidden_columns / kRegfedTileK) * column_blocks * groups * 4U;
        const auto grow = [&](void*& pointer, std::uint64_t& capacity,
                              std::uint64_t required) {
            return regfed_grow(pointer, capacity, required, false,
                               state.stream);
        };
        if (grow(state.moe_regfed_gate_partials,
                 state.moe_regfed_gate_partial_bytes, partial_bytes) !=
                cudaSuccess ||
            grow(state.moe_regfed_up_partials,
                 state.moe_regfed_up_partial_bytes, partial_bytes) !=
                cudaSuccess ||
            grow(state.moe_regfed_hidden_fragment,
                 state.moe_regfed_hidden_fragment_bytes,
                 fragment_total * sizeof(uint2)) != cudaSuccess) {
            abort_enqueue(cudaErrorMemoryAllocation,
                          "allocate register-fed FP8 MoE gate/up workspaces");
            return result;
        }
        const auto* compact_values =
            static_cast<const unsigned char*>(state.moe_regfed_compact);
        const auto* compact_scales = reinterpret_cast<const float*>(
            compact_values + hidden_elements);
        regfed_fp8_moe_activation_fragment_kernel<<<
            static_cast<unsigned int>(std::min<std::uint64_t>(
                (fragment_total + 255U) / 256U, 65535U)),
            256U, 0U, state.stream>>>(
            static_cast<uint2*>(state.moe_regfed_hidden_fragment),
            compact_values, 1U, rows,
            static_cast<std::uint32_t>(hidden_columns), column_blocks, groups);
        const auto blocks = static_cast<unsigned int>(
            std::min<std::uint64_t>(
                (static_cast<std::uint64_t>(experts) * n_tiles * split +
                 kRegfedWarpsPerBlock - 1U) /
                    kRegfedWarpsPerBlock,
                65535U));
        const auto launch = [&](auto tag) {
            constexpr std::uint32_t kBlocks = decltype(tag)::value;
            regfed_fp8_f32_moe_gate_up_kernel<kBlocks><<<
                blocks, kRegfedWarpsPerBlock * 32U, 0U, state.stream>>>(
                static_cast<float*>(state.moe_regfed_gate_partials),
                static_cast<float*>(state.moe_regfed_up_partials),
                static_cast<const uint2*>(state.moe_regfed_hidden_fragment),
                compact_scales, fp8_f32_batch_data,
                static_cast<std::uint32_t>(hidden_columns),
                static_cast<std::uint32_t>(intermediate_columns), split, rows,
                groups);
        };
        if (column_blocks == 1U) {
            launch(std::integral_constant<std::uint32_t, 1U>{});
        } else {
            launch(std::integral_constant<std::uint32_t, 2U>{});
        }
        const std::uint64_t swiglu_total =
            static_cast<std::uint64_t>(experts) * intermediate_columns * rows;
        regfed_fp8_f32_moe_swiglu_kernel<<<
            static_cast<unsigned int>(std::min<std::uint64_t>(
                (swiglu_total + 255U) / 256U, 65535U)),
            256U, 0U, state.stream>>>(
            state.moe_activations,
            static_cast<const float*>(state.moe_regfed_gate_partials),
            static_cast<const float*>(state.moe_regfed_up_partials), experts,
            static_cast<std::uint32_t>(intermediate_columns), rows, split,
            swiglu_limit, state.moe_error);
        state.moe_kernel_launches += 2U;
    } else if (fp8_f32_batch) {
        fp8_f32_moe_gate_up_kernel<<<gate_grid, threads, 0U, state.stream>>>(
            state.moe_activations, state.moe_hidden, fp8_f32_batch_data,
            hidden_columns, intermediate_columns, gate.scale_columns,
            swiglu_limit, state.moe_error);
    } else if (nvfp4_batch) {
        nvfp4_moe_gate_up_kernel<<<gate_grid, threads, 0U, state.stream>>>(
            state.moe_activations, state.moe_hidden, nvfp4_batch_data,
            hidden_columns, intermediate_columns, gate.packed_columns,
            gate.scale_columns, gate.group_size, state.moe_error);
    } else if (mxfp4_regfed) {
        const auto experts = batch.count;
        const auto column_blocks = static_cast<std::uint32_t>(
            (std::min<std::uint64_t>(rows, kRegfedMaxM) + kRegfedTileM - 1U) /
            kRegfedTileM);
        const auto groups = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(rows, kRegfedTileM));
        const auto n_tiles =
            static_cast<std::uint32_t>(intermediate_columns / kRegfedTileN);
        const auto k_blocks = static_cast<std::uint32_t>(
            (hidden_columns / kRegfedTileK) / kRegfedKPerLoad);
        std::uint32_t split = 1U;
        while (split < 16U && k_blocks % (split * 2U) == 0U &&
               static_cast<std::uint64_t>(experts) * n_tiles * split * 2U <= 4096U) {
            split *= 2U;
        }
        const std::uint64_t partial_bytes = static_cast<std::uint64_t>(experts) *
                                            intermediate_columns * rows * split *
                                            sizeof(float);
        const std::uint64_t fragment_bytes =
            static_cast<std::uint64_t>(hidden_columns / kRegfedTileK) *
            column_blocks * groups * 4U * sizeof(uint2);
        const auto grow = [&](void*& pointer, std::uint64_t& capacity,
                              std::uint64_t required) {
            return regfed_grow(pointer, capacity, required, false, state.stream);
        };
        if (grow(state.moe_regfed_gate_partials,
                 state.moe_regfed_gate_partial_bytes, partial_bytes) != cudaSuccess ||
            grow(state.moe_regfed_up_partials,
                 state.moe_regfed_up_partial_bytes, partial_bytes) != cudaSuccess ||
            grow(state.moe_regfed_hidden_fragment,
                 state.moe_regfed_hidden_fragment_bytes, fragment_bytes) != cudaSuccess) {
            abort_enqueue(cudaErrorMemoryAllocation,
                          "allocate register-fed MoE gate/up workspaces");
            return result;
        }
        const std::uint64_t fragment_total =
            static_cast<std::uint64_t>(hidden_columns / kRegfedTileK) *
            column_blocks * groups * 4U;
        regfed_moe_activation_fragment_kernel<<<
            static_cast<unsigned int>(
                std::min<std::uint64_t>((fragment_total + 255U) / 256U, 65535U)),
            256U, 0U, state.stream>>>(
            static_cast<uint2*>(state.moe_regfed_hidden_fragment),
            state.moe_hidden, 1U, static_cast<std::uint32_t>(rows),
            static_cast<std::uint32_t>(hidden_columns), column_blocks, groups);
        const auto blocks = static_cast<unsigned int>(std::min<std::uint64_t>(
            (static_cast<std::uint64_t>(experts) * n_tiles * split +
             kRegfedWarpsPerBlock - 1U) / kRegfedWarpsPerBlock, 65535U));
        const auto launch = [&](auto tag) {
            constexpr std::uint32_t kBlocks = decltype(tag)::value;
            regfed_mxfp4_moe_gate_up_kernel<kBlocks><<<
                blocks, kRegfedWarpsPerBlock * 32U, 0U, state.stream>>>(
                static_cast<float*>(state.moe_regfed_gate_partials),
                static_cast<float*>(state.moe_regfed_up_partials),
                static_cast<const uint2*>(state.moe_regfed_hidden_fragment),
                mxfp4_batch_data, static_cast<std::uint32_t>(hidden_columns),
                static_cast<std::uint32_t>(intermediate_columns), split,
                static_cast<std::uint32_t>(rows), groups);
        };
        if (column_blocks == 1U) launch(std::integral_constant<std::uint32_t, 1U>{});
        else launch(std::integral_constant<std::uint32_t, 2U>{});
        const std::uint64_t swiglu_total =
            static_cast<std::uint64_t>(experts) * intermediate_columns * rows;
        regfed_mxfp4_moe_swiglu_kernel<<<
            static_cast<unsigned int>(
                std::min<std::uint64_t>((swiglu_total + 255U) / 256U, 65535U)),
            256U, 0U, state.stream>>>(
            state.moe_activations,
            static_cast<const float*>(state.moe_regfed_gate_partials),
            static_cast<const float*>(state.moe_regfed_up_partials), experts,
            static_cast<std::uint32_t>(intermediate_columns),
            static_cast<std::uint32_t>(rows), split, state.moe_error);
        state.moe_kernel_launches += 2U;
    } else if (mxfp4_batch) {
        mxfp4_moe_gate_up_kernel<<<gate_grid, threads, 0U, state.stream>>>(
            state.moe_activations, state.moe_hidden, mxfp4_batch_data,
            hidden_columns, intermediate_columns, gate.packed_columns,
            gate.scale_columns, state.moe_error);
    } else {
        packed_int4_moe_gate_up_kernel<<<gate_grid, threads, 0U, state.stream>>>(
            state.moe_activations, state.moe_hidden, batch, hidden_columns,
            intermediate_columns, gate.packed_columns, gate.scale_columns,
            gate.group_size, state.moe_error);
    }
    ++state.moe_kernel_launches;
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        abort_enqueue(status, "launch MoE gate/up SwiGLU");
        return result;
    }
    if (fp8_f32_regfed) {
        const auto activation_elements =
            activation_rows * intermediate_columns;
        auto* compact_values =
            static_cast<unsigned char*>(state.moe_regfed_compact);
        auto* compact_scales = reinterpret_cast<float*>(
            compact_values + activation_elements);
        const dim3 quantize_grid(
            static_cast<unsigned int>(intermediate_columns / 128U),
            static_cast<unsigned int>(activation_rows), 1U);
        quantize_activation_e4m3_f32_bytes_kernel<<<
            quantize_grid, 128U, 0U, state.stream>>>(
            compact_values, compact_scales, state.moe_activations,
            intermediate_columns, static_cast<std::uint32_t>(activation_rows));
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status,
                          "compact F32-scaled FP8 MoE down activation");
            return result;
        }
    } else if (fp8_f32_batch) {
        const dim3 quantize_grid(
            static_cast<unsigned int>((intermediate_columns + 127U) / 128U),
            static_cast<unsigned int>(activation_rows), 1U);
        quantize_activation_e4m3_f32_scale_kernel<<<
            quantize_grid, 128U, 0U, state.stream>>>(
            state.moe_activations, intermediate_columns,
            static_cast<std::uint32_t>(activation_rows));
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status,
                          "launch F32-scaled FP8 MoE activation quantization");
            return result;
        }
    } else if (mxfp4_batch) {
        // Runs for both routes: it is what makes the activation E4M3, and an
        // E4M3 value's BF16 image is exact, which is why the register-fed
        // tensor op multiplies the same numbers the scalar kernel does.
        const dim3 quantize_grid(
            static_cast<unsigned int>((intermediate_columns + 127U) / 128U),
            static_cast<unsigned int>(activation_rows), 1U);
        quantize_activation_e4m3_kernel<<<quantize_grid, 128U, 0U,
                                          state.stream>>>(
            state.moe_activations, intermediate_columns,
            static_cast<std::uint32_t>(activation_rows));
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch MXFP4 MoE activation quantization");
            return result;
        }
    }
    const dim3 down_grid(static_cast<unsigned int>(hidden_columns),
                         static_cast<unsigned int>(activation_rows), 1U);
    const dim3 plain_down_grid(
        static_cast<unsigned int>((hidden_columns + warps_per_block - 1U) /
                                  warps_per_block),
        static_cast<unsigned int>(activation_rows), 1U);
    if (plain_batch) {
        plain_bf16_moe_down_kernel<<<plain_down_grid, threads, 0U, state.stream>>>(
            state.moe_output, state.moe_activations, plain_batch_data,
            intermediate_columns, hidden_columns, state.moe_error);
    } else if (fp8_f32_regfed) {
        const auto experts = batch.count;
        const auto column_blocks = static_cast<std::uint32_t>(
            (rows + kRegfedTileM - 1U) / kRegfedTileM);
        const auto groups = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(rows, kRegfedTileM));
        const auto n_tiles =
            static_cast<std::uint32_t>(hidden_columns / kRegfedTileN);
        const auto pairs =
            static_cast<std::uint32_t>(intermediate_columns / 32U);
        std::uint32_t split = 1U;
        while (split < 16U && pairs % ((split * 2U) * 4U) == 0U &&
               static_cast<std::uint64_t>(experts) * n_tiles * split * 2U <=
                   4096U) {
            split *= 2U;
        }
        const std::uint64_t partial_bytes =
            static_cast<std::uint64_t>(experts) * hidden_columns * rows * split *
            sizeof(float);
        const std::uint64_t fragment_total =
            static_cast<std::uint64_t>(experts) *
            (intermediate_columns / kRegfedTileK) * column_blocks * groups * 4U;
        const auto grow = [&](void*& pointer, std::uint64_t& capacity,
                              std::uint64_t required) {
            return regfed_grow(pointer, capacity, required, false,
                               state.stream);
        };
        if (grow(state.moe_regfed_down_partials,
                 state.moe_regfed_down_partial_bytes, partial_bytes) !=
                cudaSuccess ||
            grow(state.moe_regfed_activation_fragment,
                 state.moe_regfed_activation_fragment_bytes,
                 fragment_total * sizeof(uint2)) != cudaSuccess) {
            abort_enqueue(cudaErrorMemoryAllocation,
                          "allocate register-fed FP8 MoE down workspaces");
            return result;
        }
        const auto activation_elements =
            activation_rows * intermediate_columns;
        const auto* compact_values =
            static_cast<const unsigned char*>(state.moe_regfed_compact);
        const auto* compact_scales = reinterpret_cast<const float*>(
            compact_values + activation_elements);
        regfed_fp8_moe_activation_fragment_kernel<<<
            static_cast<unsigned int>(std::min<std::uint64_t>(
                (fragment_total + 255U) / 256U, 65535U)),
            256U, 0U, state.stream>>>(
            static_cast<uint2*>(state.moe_regfed_activation_fragment),
            compact_values, experts, rows,
            static_cast<std::uint32_t>(intermediate_columns), column_blocks,
            groups);
        const auto blocks = static_cast<unsigned int>(
            std::min<std::uint64_t>(
                (static_cast<std::uint64_t>(experts) * n_tiles * split +
                 kRegfedWarpsPerBlock - 1U) /
                    kRegfedWarpsPerBlock,
                65535U));
        const auto launch = [&](auto tag) {
            constexpr std::uint32_t kBlocks = decltype(tag)::value;
            regfed_fp8_f32_moe_down_kernel<kBlocks><<<
                blocks, kRegfedWarpsPerBlock * 32U, 0U, state.stream>>>(
                static_cast<float*>(state.moe_regfed_down_partials),
                static_cast<const uint2*>(state.moe_regfed_activation_fragment),
                compact_scales, fp8_f32_batch_data,
                static_cast<std::uint32_t>(intermediate_columns),
                static_cast<std::uint32_t>(hidden_columns), split, rows,
                groups);
        };
        if (column_blocks == 1U) {
            launch(std::integral_constant<std::uint32_t, 1U>{});
        } else {
            launch(std::integral_constant<std::uint32_t, 2U>{});
        }
        const std::uint64_t reduce_total =
            static_cast<std::uint64_t>(experts) * hidden_columns * rows;
        regfed_fp8_f32_moe_reduce_kernel<<<
            static_cast<unsigned int>(std::min<std::uint64_t>(
                (reduce_total + 255U) / 256U, 65535U)),
            256U, 0U, state.stream>>>(
            state.moe_output,
            static_cast<const float*>(state.moe_regfed_down_partials), experts,
            static_cast<std::uint32_t>(hidden_columns), rows, split,
            state.moe_error);
        state.moe_kernel_launches += 2U;
    } else if (fp8_f32_batch) {
        fp8_f32_moe_down_kernel<<<down_grid, threads, 0U, state.stream>>>(
            state.moe_output, state.moe_activations, fp8_f32_batch_data,
            intermediate_columns, hidden_columns, down.scale_columns,
            state.moe_error);
    } else if (nvfp4_batch) {
        nvfp4_moe_down_kernel<<<down_grid, threads, 0U, state.stream>>>(
            state.moe_output, state.moe_activations, nvfp4_batch_data,
            intermediate_columns, hidden_columns, down.packed_columns,
            down.scale_columns, down.group_size, state.moe_error);
    } else if (mxfp4_regfed) {
        const auto experts = batch.count;
        const auto column_blocks = static_cast<std::uint32_t>(
            (std::min<std::uint64_t>(rows, kRegfedMaxM) + kRegfedTileM - 1U) /
            kRegfedTileM);
        const auto groups = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(rows, kRegfedTileM));
        const auto n_tiles =
            static_cast<std::uint32_t>(hidden_columns / kRegfedTileN);
        const auto k_blocks = static_cast<std::uint32_t>(
            (intermediate_columns / kRegfedTileK) / kRegfedKPerLoad);
        std::uint32_t split = 1U;
        while (split < 16U && k_blocks % (split * 2U) == 0U &&
               static_cast<std::uint64_t>(experts) * n_tiles * split * 2U <= 4096U) {
            split *= 2U;
        }
        const std::uint64_t partial_bytes = static_cast<std::uint64_t>(experts) *
                                            hidden_columns * rows * split *
                                            sizeof(float);
        const std::uint64_t fragment_total =
            static_cast<std::uint64_t>(experts) *
            (intermediate_columns / kRegfedTileK) * column_blocks * groups * 4U;
        const auto grow = [&](void*& pointer, std::uint64_t& capacity,
                              std::uint64_t required) {
            return regfed_grow(pointer, capacity, required, false, state.stream);
        };
        if (grow(state.moe_regfed_down_partials,
                 state.moe_regfed_down_partial_bytes, partial_bytes) != cudaSuccess ||
            grow(state.moe_regfed_activation_fragment,
                 state.moe_regfed_activation_fragment_bytes,
                 fragment_total * sizeof(uint2)) != cudaSuccess) {
            abort_enqueue(cudaErrorMemoryAllocation,
                          "allocate register-fed MoE down workspaces");
            return result;
        }
        regfed_moe_activation_fragment_kernel<<<
            static_cast<unsigned int>(
                std::min<std::uint64_t>((fragment_total + 255U) / 256U, 65535U)),
            256U, 0U, state.stream>>>(
            static_cast<uint2*>(state.moe_regfed_activation_fragment),
            state.moe_activations, experts, static_cast<std::uint32_t>(rows),
            static_cast<std::uint32_t>(intermediate_columns), column_blocks, groups);
        const auto blocks = static_cast<unsigned int>(std::min<std::uint64_t>(
            (static_cast<std::uint64_t>(experts) * n_tiles * split +
             kRegfedWarpsPerBlock - 1U) / kRegfedWarpsPerBlock, 65535U));
        const auto launch = [&](auto tag) {
            constexpr std::uint32_t kBlocks = decltype(tag)::value;
            regfed_mxfp4_moe_down_kernel<kBlocks><<<
                blocks, kRegfedWarpsPerBlock * 32U, 0U, state.stream>>>(
                static_cast<float*>(state.moe_regfed_down_partials),
                static_cast<const uint2*>(state.moe_regfed_activation_fragment),
                mxfp4_batch_data,
                static_cast<std::uint32_t>(intermediate_columns),
                static_cast<std::uint32_t>(hidden_columns), split,
                static_cast<std::uint32_t>(rows), groups);
        };
        if (column_blocks == 1U) launch(std::integral_constant<std::uint32_t, 1U>{});
        else launch(std::integral_constant<std::uint32_t, 2U>{});
        const std::uint64_t reduce_total =
            static_cast<std::uint64_t>(experts) * hidden_columns * rows;
        regfed_mxfp4_moe_reduce_kernel<<<
            static_cast<unsigned int>(
                std::min<std::uint64_t>((reduce_total + 255U) / 256U, 65535U)),
            256U, 0U, state.stream>>>(
            state.moe_output,
            static_cast<const float*>(state.moe_regfed_down_partials), experts,
            static_cast<std::uint32_t>(hidden_columns),
            static_cast<std::uint32_t>(rows), split, state.moe_error);
        state.moe_kernel_launches += 2U;
    } else if (mxfp4_batch) {
        mxfp4_moe_down_kernel<<<down_grid, threads, 0U, state.stream>>>(
            state.moe_output, state.moe_activations, mxfp4_batch_data,
            intermediate_columns, hidden_columns, down.packed_columns,
            down.scale_columns, state.moe_error);
    } else {
        packed_int4_moe_down_kernel<<<down_grid, threads, 0U, state.stream>>>(
            state.moe_output, state.moe_activations, batch, intermediate_columns,
            hidden_columns, down.packed_columns, down.scale_columns,
            down.group_size, state.moe_error);
    }
    ++state.moe_kernel_launches;
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        abort_enqueue(status, "launch MoE down projection");
        return result;
    }
    if (mhc_source_destination) {
        constexpr unsigned int join_threads = 256U;
        const auto join_blocks = static_cast<unsigned int>(
            (hidden_columns + join_threads - 1U) / join_threads);
        glm53_moe_join_mhc_kernel<<<join_blocks, join_threads, 0U,
                                    state.stream>>>(
            state.moe_output, fp8_f32_batch_data.coefficients,
            static_cast<std::uint32_t>(routed.size()),
            state.dsv4_mhc_workspace->branch,
            static_cast<std::uint32_t>(hidden_columns), state.moe_error);
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "join resident GLM-5.3 MoE branch");
            return result;
        }
        state.dsv4_mhc_branch_ready = true;
    }
    if (auto status = cudaEventRecord(state.moe_kernel_finished, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record MoE kernel completion");
        return result;
    }
    {
        std::scoped_lock lock(impl_->mutex);
        auto& device_stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        device_stats.activation_h2d_bytes +=
            mhc_source_destination ? 0U : hidden_bytes;
        device_stats.matmul_calls += 3U * expert_count;
        device_stats.workspace_allocation_calls += allocation_calls;
        device_stats.workspace_allocation_bytes += allocation_bytes;
        ++device_stats.deepseek_moe_calls;
        device_stats.deepseek_moe_kernel_launches += state.moe_kernel_launches;
        device_stats.deepseek_moe_h2d_transfers +=
            mhc_source_destination ? 0U : 1U;
        device_stats.deepseek_moe_h2d_bytes +=
            mhc_source_destination ? 0U : hidden_bytes;
    }
    return result;
}

ValidationResult CudaBackend::dsv4_tier_reserve(
    int device, std::uint32_t layers, std::uint32_t experts) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek tier reserve targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    if (state.tier_layers != 0U) {
        result.errors.emplace_back("DeepSeek tier is already reserved");
        return result;
    }
    if (layers == 0U || experts == 0U) {
        result.errors.emplace_back("DeepSeek tier needs a positive geometry");
        return result;
    }
    if (const auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for DeepSeek tier");
    }
    const auto entries = static_cast<std::size_t>(layers) * experts;
    for (std::size_t array = 0U; array < 6U; ++array) {
        state.tier_host_pointers[array].assign(entries, nullptr);
        void* device_array = nullptr;
        if (const auto status = cudaMalloc(
                &device_array, entries * sizeof(const unsigned char*));
            status != cudaSuccess) {
            return cuda_error(status, "allocate DeepSeek tier pointer table");
        }
        state.tier_device_pointers[array] =
            static_cast<const unsigned char**>(device_array);
    }
    void* selection_host = nullptr;
    if (const auto status = cudaMallocHost(
            &selection_host, sizeof(CudaDsv4TierSelection));
        status != cudaSuccess) {
        return cuda_error(status, "allocate DeepSeek tier selection staging");
    }
    state.tier_selection_host =
        static_cast<CudaDsv4TierSelection*>(selection_host);
    *state.tier_selection_host = CudaDsv4TierSelection{};
    void* selection_device = nullptr;
    if (const auto status =
            cudaMalloc(&selection_device, sizeof(CudaDsv4TierSelection));
        status != cudaSuccess) {
        return cuda_error(status, "allocate DeepSeek tier selection");
    }
    state.tier_selection_device =
        static_cast<CudaDsv4TierSelection*>(selection_device);
    state.tier_layers = layers;
    state.tier_experts = experts;
    return result;
}

ValidationResult CudaBackend::dsv4_tier_add(
    int device, std::uint32_t layer, std::uint32_t expert,
    const CudaWeight& w1, const CudaWeight& w3, const CudaWeight& w2) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek tier add targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    if (state.tier_layers == 0U) {
        result.errors.emplace_back("DeepSeek tier is not reserved");
        return result;
    }
    if (layer >= state.tier_layers || expert >= state.tier_experts) {
        result.errors.emplace_back("DeepSeek tier entry is out of range");
        return result;
    }
    const std::array<const CudaWeight*, 3U> weights{&w1, &w3, &w2};
    for (const auto* weight : weights) {
        if (weight == nullptr || !weight->valid() ||
            weight->impl_->device != device) {
            result.errors.emplace_back(
                "DeepSeek tier entry weight is invalid or on another device");
            return result;
        }
        if (weight->impl_->descriptor.encoding !=
            CudaWeightEncoding::Fp4E2m1Group32) {
            result.errors.emplace_back(
                "DeepSeek tier entry weight is not FP4 E2M1 group 32");
            return result;
        }
    }
    const auto entry =
        static_cast<std::size_t>(layer) * state.tier_experts + expert;
    if (state.tier_host_pointers[0][entry] != nullptr) {
        result.errors.emplace_back(
            "DeepSeek tier already holds this layer and expert");
        return result;
    }
    const auto& gate_descriptor = w1.impl_->descriptor;
    const auto& down_descriptor = w2.impl_->descriptor;
    if (state.tier_gate_packed_columns == 0U) {
        state.tier_gate_packed_columns = gate_descriptor.packed_columns;
        state.tier_gate_scale_columns = gate_descriptor.scale_columns;
        state.tier_down_packed_columns = down_descriptor.packed_columns;
        state.tier_down_scale_columns = down_descriptor.scale_columns;
    } else if (gate_descriptor.packed_columns != state.tier_gate_packed_columns ||
               gate_descriptor.scale_columns != state.tier_gate_scale_columns ||
               down_descriptor.packed_columns != state.tier_down_packed_columns ||
               down_descriptor.scale_columns != state.tier_down_scale_columns) {
        result.errors.emplace_back(
            "DeepSeek tier entry shape differs from the tier's first entry");
        return result;
    }
    const std::array<const CudaWeight*, 6U> ordered{&w1, &w1, &w3, &w3, &w2, &w2};
    for (std::size_t array = 0U; array < 6U; ++array) {
        state.tier_host_pointers[array][entry] = static_cast<const unsigned char*>(
            array % 2U == 0U ? ordered[array]->impl_->weights
                             : ordered[array]->impl_->scales);
    }
    ++state.tier_installed;
    return result;
}

ValidationResult CudaBackend::dsv4_tier_commit(int device) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek tier commit targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    if (state.tier_layers == 0U) {
        result.errors.emplace_back("DeepSeek tier is not reserved");
        return result;
    }
    if (const auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for DeepSeek tier commit");
    }
    const auto entries =
        static_cast<std::size_t>(state.tier_layers) * state.tier_experts;
    for (std::size_t array = 0U; array < 6U; ++array) {
        if (const auto status = cudaMemcpy(
                state.tier_device_pointers[array],
                state.tier_host_pointers[array].data(),
                entries * sizeof(const unsigned char*), cudaMemcpyHostToDevice);
            status != cudaSuccess) {
            return cuda_error(status, "upload DeepSeek tier pointer table");
        }
    }
    // Overlapped dispatch needs a stream of its own, two events, and a
    // partial buffer the rank-partial upload does not also write. They are
    // built once here rather than per layer; a command that does not pass a
    // route half simply never touches them.
    if (state.tier_stream == nullptr) {
        if (const auto status = cudaStreamCreateWithFlags(
                &state.tier_stream, cudaStreamNonBlocking);
            status != cudaSuccess) {
            return cuda_error(status, "create DeepSeek tier stream");
        }
    }
    if (state.tier_route_ready == nullptr) {
        if (const auto status = cudaEventCreateWithFlags(
                &state.tier_route_ready, cudaEventDisableTiming);
            status != cudaSuccess) {
            return cuda_error(status, "create DeepSeek tier route event");
        }
    }
    if (state.tier_finished == nullptr) {
        if (const auto status = cudaEventCreateWithFlags(
                &state.tier_finished, cudaEventDisableTiming);
            status != cudaSuccess) {
            return cuda_error(status, "create DeepSeek tier completion event");
        }
    }
    state.tier_committed = true;
    return result;
}

bool CudaBackend::dsv4_tier_active(int device) const noexcept {
    const auto found = impl_->devices.find(device);
    return found != impl_->devices.end() && found->second.tier_installed != 0U &&
           found->second.tier_committed;
}

CudaDsv4TierSelection* CudaBackend::dsv4_tier_selection(int device) noexcept {
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) return nullptr;
    return found->second.tier_selection_host;
}

ValidationResult CudaBackend::collect_deepseek_moe(
    int device, std::span<float> routed_output,
    std::span<float> shared_output) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek MoE collect targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    const auto reset_host_join = [&state] {
        state.moe_host_join = false;
        state.moe_output_to_mhc = false;
        state.moe_host_callback = {};
        for (std::uint32_t index = 0U;
             index < state.moe_host_callback_count; ++index) {
            state.moe_host_callbacks[index] = {};
        }
        state.moe_host_callback_count = 0U;
    };
    if (!state.moe_in_flight && state.moe_host_callback_count == 0U) {
        result.errors.emplace_back(
            "DeepSeek MoE collect has no matching in-flight command");
        return result;
    }
    if (state.moe_poisoned) {
        result.errors.emplace_back(
            "DeepSeek MoE workspace is poisoned by an unconfirmed CUDA drain");
        if (const auto select_status = cudaSetDevice(device);
            select_status == cudaSuccess) {
            if (const auto drain_status = cudaDeviceSynchronize();
                drain_status == cudaSuccess) {
                state.moe_in_flight = false;
                state.moe_poisoned = false;
                state.moe_weights.clear();
                reset_host_join();
            } else {
                result.errors.emplace_back(
                    std::string("retry poisoned DeepSeek MoE drain: ") +
                    cudaGetErrorString(drain_status));
            }
        } else {
            result.errors.emplace_back(
                std::string("select poisoned DeepSeek MoE device: ") +
                cudaGetErrorString(select_status));
        }
        return result;
    }
    auto drain_without_output = [&]() {
        if (const auto status = cudaSetDevice(device); status != cudaSuccess) {
            result.errors.emplace_back(
                std::string("select CUDA device while draining DeepSeek MoE: ") +
                cudaGetErrorString(status));
        }
        const auto drain_status = cudaEventSynchronize(state.moe_kernel_finished);
        if (drain_status != cudaSuccess) {
            result.errors.emplace_back(
                std::string("drain DeepSeek MoE kernels: ") +
                cudaGetErrorString(drain_status));
            state.moe_poisoned = true;
        } else {
            state.moe_in_flight = false;
            state.moe_weights.clear();
            reset_host_join();
        }
    };
    std::uint64_t routed_rows = 0U;
    std::uint64_t routed_elements = 0U;
    std::uint64_t shared_elements = 0U;
    const bool output_to_mhc = state.moe_output_to_mhc;
    if (!checked_bytes(state.moe_routed_count, state.moe_rows, 1U,
                       routed_rows) ||
        !checked_bytes(routed_rows, state.moe_hidden_columns, 1U,
                       routed_elements) ||
        !checked_bytes(state.moe_shared_rows, state.moe_hidden_columns, 1U,
                       shared_elements) ||
        routed_output.size() != routed_elements ||
        (output_to_mhc
             ? (!shared_output.empty() &&
                shared_output.size() != shared_elements)
             : (state.moe_has_shared
                    ? shared_output.size() != shared_elements
                    : !shared_output.empty()))) {
        result.errors.emplace_back(
            "DeepSeek MoE collect output spans do not match the enqueued command");
        drain_without_output();
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        result = cuda_error(status, "select CUDA device for DeepSeek MoE collect");
        drain_without_output();
        return result;
    }
    auto abort_collect = [&](cudaError_t status, const char* operation) {
        result = cuda_error(status, operation);
        const auto drain_status = cudaStreamSynchronize(state.stream);
        if (drain_status != cudaSuccess) {
            result.errors.emplace_back(
                std::string("drain failed DeepSeek MoE collect: ") +
                cudaGetErrorString(drain_status));
            state.moe_poisoned = true;
        } else {
            state.moe_in_flight = false;
            state.moe_weights.clear();
            reset_host_join();
        }
    };
    if (auto status = cudaEventRecord(state.moe_download_started, state.stream);
        status != cudaSuccess) {
        abort_collect(status, "record DeepSeek MoE download start");
        return result;
    }
    const auto routed_bytes =
        static_cast<std::uint64_t>(routed_output.size_bytes());
    const auto shared_bytes =
        static_cast<std::uint64_t>(shared_output.size_bytes());
    const auto downloaded_bytes = routed_bytes + shared_bytes;
    auto* host_bytes = static_cast<std::byte*>(state.moe_host_staging);
    auto* host_error = reinterpret_cast<unsigned int*>(
        host_bytes + static_cast<std::ptrdiff_t>(downloaded_bytes));
    if (downloaded_bytes != 0U) {
        if (auto status = cudaMemcpyAsync(
                host_bytes, state.moe_output,
                static_cast<std::size_t>(downloaded_bytes),
                cudaMemcpyDeviceToHost, state.stream);
            status != cudaSuccess) {
            abort_collect(status, "stage DeepSeek MoE expert outputs");
            return result;
        }
    }
    if (auto status = cudaMemcpyAsync(
            host_error, state.moe_error, sizeof(unsigned int),
            cudaMemcpyDeviceToHost, state.stream);
        status != cudaSuccess) {
        abort_collect(status, "stage DeepSeek MoE error flag");
        return result;
    }
    if (auto status = cudaEventRecord(state.moe_completed, state.stream);
        status != cudaSuccess) {
        abort_collect(status, "record DeepSeek MoE completion");
        return result;
    }
    const auto wait_started = std::chrono::steady_clock::now();
    const auto wait_status = cudaEventSynchronize(state.moe_completed);
    const auto wait_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wait_started).count());
    if (wait_status != cudaSuccess) {
        result = cuda_error(wait_status, "synchronize DeepSeek MoE completion");
        const auto drain_status = cudaStreamSynchronize(state.stream);
        if (drain_status != cudaSuccess) {
            result.errors.emplace_back(
                std::string("drain failed DeepSeek MoE execution: ") +
                cudaGetErrorString(drain_status));
            state.moe_poisoned = true;
        } else {
            state.moe_in_flight = false;
            state.moe_weights.clear();
            reset_host_join();
        }
        return result;
    }
    for (auto& [attention_device, attention_state] : impl_->devices) {
        if (attention_state.dsv4_attention_prepare_host_command_count != 0U) {
            if (auto status = cudaSetDevice(attention_device);
                status != cudaSuccess) {
                result.errors.emplace_back(
                    std::string("select deferred attention callback device: ") +
                    cudaGetErrorString(status));
            } else if (auto status = cudaStreamSynchronize(
                           attention_state.stream);
                       status != cudaSuccess) {
                result.errors.emplace_back(
                    std::string("drain deferred attention callback stream: ") +
                    cudaGetErrorString(status));
            }
        }
        for (std::uint32_t index = 0U;
             index <
                 attention_state.dsv4_attention_prepare_host_command_count;
             ++index) {
            if (attention_state
                    .dsv4_attention_prepare_host_commands[index].failed) {
                result.errors.emplace_back(
                    "DeepSeek attention preparation host callback failed");
            }
        }
        attention_state.dsv4_attention_prepare_host_command_count = 0U;
        attention_state.dsv4_deferred_attention_command_count = 0U;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        result.errors.emplace_back(
            std::string("restore DeepSeek MoE device after callback drain: ") +
            cudaGetErrorString(status));
    }
    if (state.dsv4_deferred_attention_source_device >= 0) {
        const auto source_found = impl_->devices.find(
            state.dsv4_deferred_attention_source_device);
        if (source_found == impl_->devices.end()) {
            result.errors.emplace_back(
                "deferred DeepSeek attention source device disappeared");
        } else if (impl_->detailed_timing) {
            auto& source = source_found->second;
            const bool cross =
                state.dsv4_deferred_attention_cross_transition;
            float source_h2d_ms = 0.0F;
            float attention_ms = 0.0F;
            float mhc_ms = 0.0F;
            float kernel_ms = 0.0F;
            float d2h_ms = 0.0F;
            const auto measure = [&](float& output, cudaEvent_t begin,
                                     cudaEvent_t end,
                                     const char* operation) {
                if (auto status = cudaEventElapsedTime(&output, begin, end);
                    status != cudaSuccess) {
                    result.errors.emplace_back(
                        std::string(operation) + ": " +
                        cudaGetErrorString(status));
                    return false;
                }
                return true;
            };
            if (auto status = cudaSetDevice(
                    state.dsv4_deferred_attention_source_device);
                status != cudaSuccess) {
                result.errors.emplace_back(
                    std::string("select deferred attention source: ") +
                    cudaGetErrorString(status));
            } else if (cross) {
                float source_d2h_ms = 0.0F;
                float target_h2d_ms = 0.0F;
                float target_kernel_ms = 0.0F;
                float target_d2h_ms = 0.0F;
                const bool source_measured =
                    measure(source_h2d_ms, source.activation_start,
                            source.activation_uploaded,
                            "measure deferred attention upload") &&
                    measure(attention_ms, source.activation_uploaded,
                            source.mhc_transition_finished,
                            "measure deferred attention kernels") &&
                    measure(source_d2h_ms, source.mhc_transition_finished,
                            source.activation_downloaded,
                            "measure deferred attention source download");
                if (auto status = cudaSetDevice(device);
                    status != cudaSuccess) {
                    result.errors.emplace_back(
                        std::string("restore deferred attention target: ") +
                        cudaGetErrorString(status));
                } else {
                    const bool target_measured =
                        measure(target_h2d_ms, state.activation_start,
                                state.activation_uploaded,
                                "measure deferred mHC upload") &&
                        measure(target_kernel_ms, state.activation_uploaded,
                                state.kernel_finished,
                                "measure deferred mHC/router kernels") &&
                        measure(mhc_ms, state.activation_uploaded,
                                state.router_started,
                                "measure deferred mHC transition") &&
                        measure(target_d2h_ms, state.kernel_finished,
                                state.activation_downloaded,
                                "measure deferred target download");
                    if (source_measured && target_measured) {
                        source_h2d_ms += target_h2d_ms;
                        kernel_ms = attention_ms + target_kernel_ms;
                        d2h_ms = source_d2h_ms + target_d2h_ms;
                    }
                }
            } else {
                const bool measured =
                    measure(source_h2d_ms, source.activation_start,
                            source.activation_uploaded,
                            "measure deferred attention upload") &&
                    measure(kernel_ms, source.activation_uploaded,
                            source.kernel_finished,
                            "measure deferred attention/mHC kernels") &&
                    measure(attention_ms, source.activation_uploaded,
                            source.mhc_transition_finished,
                            "measure deferred attention kernels") &&
                    measure(mhc_ms, source.mhc_transition_finished,
                            source.router_started,
                            "measure deferred mHC transition") &&
                    measure(d2h_ms, source.kernel_finished,
                            source.activation_downloaded,
                            "measure deferred attention status download");
                static_cast<void>(measured);
            }
            if (result.ok()) {
                const auto to_nanoseconds = [](float milliseconds) {
                    return static_cast<std::uint64_t>(std::llround(
                        static_cast<double>(milliseconds) * 1.0e6));
                };
                std::uint64_t mhc_timing_clamped_samples = 0U;
                const auto h2d_nanoseconds =
                    to_nanoseconds(source_h2d_ms);
                const auto attention_nanoseconds =
                    to_nanoseconds(attention_ms);
                const auto mhc_nanoseconds =
                    event_milliseconds_to_nanoseconds(
                        mhc_ms, mhc_timing_clamped_samples);
                const auto kernel_nanoseconds = to_nanoseconds(kernel_ms);
                const auto d2h_nanoseconds = to_nanoseconds(d2h_ms);
                std::scoped_lock lock(impl_->mutex);
                auto& source_stats = *std::find_if(
                    impl_->stats.devices.begin(), impl_->stats.devices.end(),
                    [&](const auto& value) {
                        return value.device ==
                            state.dsv4_deferred_attention_source_device;
                    });
                source_stats.dsv4_paged_attention_h2d_nanoseconds +=
                    h2d_nanoseconds;
                source_stats.dsv4_paged_attention_kernel_nanoseconds +=
                    attention_nanoseconds;
                source_stats.dsv4_paged_attention_d2h_nanoseconds +=
                    d2h_nanoseconds;
                source_stats.dsv4_paged_attention_nanoseconds +=
                    h2d_nanoseconds + kernel_nanoseconds + d2h_nanoseconds;
                source_stats.activation_h2d_nanoseconds += h2d_nanoseconds;
                source_stats.kernel_nanoseconds += kernel_nanoseconds;
                source_stats.activation_d2h_nanoseconds += d2h_nanoseconds;
                source_stats.dsv4_mhc_kernel_nanoseconds += mhc_nanoseconds;
                source_stats.dsv4_mhc_d2h_nanoseconds += d2h_nanoseconds;
                source_stats.dsv4_mhc_nanoseconds +=
                    mhc_nanoseconds + d2h_nanoseconds;
                source_stats.dsv4_mhc_device_nanoseconds +=
                    mhc_nanoseconds + d2h_nanoseconds;
                source_stats.dsv4_mhc_timing_clamped_samples +=
                    mhc_timing_clamped_samples;
            }
        }
        state.dsv4_deferred_attention_source_device = -1;
        state.dsv4_deferred_attention_cross_transition = false;
        if (auto status = cudaSetDevice(device); status != cudaSuccess) {
            result.errors.emplace_back(
                std::string("restore DeepSeek MoE device after timing: ") +
                cudaGetErrorString(status));
        }
    }
    bool host_callback_failed =
        state.moe_host_join && state.moe_host_callback.failed;
    const bool shared_phase_timing_valid =
        state.moe_shared_phase_timing_valid;
    unsigned int first_upstream_failure =
        state.moe_host_callback.upstream_failure_value;
    for (std::uint32_t index = 0U;
         index < state.moe_host_callback_count; ++index) {
        host_callback_failed = host_callback_failed ||
            state.moe_host_callbacks[index].failed;
        if (first_upstream_failure == 0U) {
            first_upstream_failure =
                state.moe_host_callbacks[index].upstream_failure_value;
        }
    }
    state.moe_in_flight = false;
    state.moe_weights.clear();
    reset_host_join();

    float h2d_milliseconds = 0.0F;
    float kernel_milliseconds = 0.0F;
    float d2h_milliseconds = 0.0F;
    float shared_input_quantize_milliseconds = 0.0F;
    float shared_gate_up_milliseconds = 0.0F;
    float shared_activation_quantize_milliseconds = 0.0F;
    float shared_down_milliseconds = 0.0F;
    if (auto status = cudaEventElapsedTime(
            &h2d_milliseconds, state.moe_start, state.moe_hidden_uploaded);
        status != cudaSuccess) {
        return cuda_error(status, "measure DeepSeek MoE hidden upload");
    }
    if (auto status = cudaEventElapsedTime(
            &kernel_milliseconds, state.moe_hidden_uploaded,
            state.moe_kernel_finished);
        status != cudaSuccess) {
        return cuda_error(status, "measure DeepSeek MoE kernels");
    }
    if (auto status = cudaEventElapsedTime(
            &d2h_milliseconds, state.moe_download_started,
            state.moe_completed);
        status != cudaSuccess) {
        return cuda_error(status, "measure DeepSeek MoE output download");
    }
    if (shared_phase_timing_valid) {
        const auto measure_shared_phase = [&](float& output,
                                              cudaEvent_t begin,
                                              cudaEvent_t end,
                                              const char* operation) {
            if (auto status = cudaEventElapsedTime(&output, begin, end);
                status != cudaSuccess) {
                result.errors.emplace_back(
                    std::string(operation) + ": " +
                    cudaGetErrorString(status));
                return false;
            }
            return true;
        };
        if (!measure_shared_phase(
                shared_input_quantize_milliseconds,
                state.moe_hidden_uploaded,
                state.moe_shared_input_finished,
                "measure DeepSeek shared input quantization") ||
            !measure_shared_phase(
                shared_gate_up_milliseconds,
                state.moe_shared_input_finished,
                state.moe_shared_gate_up_finished,
                "measure DeepSeek shared gate/up") ||
            !measure_shared_phase(
                shared_activation_quantize_milliseconds,
                state.moe_shared_gate_up_finished,
                state.moe_shared_activation_finished,
                "measure DeepSeek shared activation quantization") ||
            !measure_shared_phase(
                shared_down_milliseconds,
                state.moe_shared_activation_finished,
                state.moe_shared_finished,
                "measure DeepSeek shared down")) {
            return result;
        }
    }
    const auto to_nanoseconds = [](float milliseconds) {
        return static_cast<std::uint64_t>(std::llround(
            static_cast<double>(milliseconds) * 1.0e6));
    };
    const auto h2d_nanoseconds = to_nanoseconds(h2d_milliseconds);
    const auto kernel_nanoseconds = to_nanoseconds(kernel_milliseconds);
    const auto d2h_nanoseconds = to_nanoseconds(d2h_milliseconds);
    const auto total_nanoseconds =
        h2d_nanoseconds + kernel_nanoseconds + d2h_nanoseconds;
    const auto shared_input_quantize_nanoseconds =
        to_nanoseconds(shared_input_quantize_milliseconds);
    const auto shared_gate_up_nanoseconds =
        to_nanoseconds(shared_gate_up_milliseconds);
    const auto shared_activation_quantize_nanoseconds =
        to_nanoseconds(shared_activation_quantize_milliseconds);
    const auto shared_down_nanoseconds =
        to_nanoseconds(shared_down_milliseconds);
    {
        std::scoped_lock lock(impl_->mutex);
        auto& device_stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        device_stats.activation_d2h_bytes += downloaded_bytes;
        record_synchronization(device_stats,
                               SynchronizationSubsystem::Moe, 1U,
                               wait_nanoseconds);
        device_stats.activation_h2d_nanoseconds += h2d_nanoseconds;
        device_stats.kernel_nanoseconds += kernel_nanoseconds;
        device_stats.activation_d2h_nanoseconds += d2h_nanoseconds;
        device_stats.deepseek_moe_d2h_transfers +=
            downloaded_bytes == 0U ? 1U : 2U;
        device_stats.deepseek_moe_d2h_bytes +=
            downloaded_bytes + sizeof(unsigned int);
        device_stats.deepseek_moe_h2d_nanoseconds += h2d_nanoseconds;
        device_stats.deepseek_moe_kernel_nanoseconds += kernel_nanoseconds;
        device_stats.deepseek_moe_input_quantize_nanoseconds +=
            shared_input_quantize_nanoseconds;
        device_stats.deepseek_moe_shared_gate_up_nanoseconds +=
            shared_gate_up_nanoseconds;
        device_stats.deepseek_moe_shared_activation_quantize_nanoseconds +=
            shared_activation_quantize_nanoseconds;
        device_stats.deepseek_moe_shared_down_nanoseconds +=
            shared_down_nanoseconds;
        device_stats.deepseek_moe_d2h_nanoseconds += d2h_nanoseconds;
        device_stats.deepseek_moe_nanoseconds += total_nanoseconds;
    }
    state.moe_shared_phase_timing_valid = false;
    if (*host_error != 0U) {
        result.errors.emplace_back(
            "MoE projection produced a non-finite activation");
        return result;
    }
    if (host_callback_failed) {
        if (first_upstream_failure != 0U) {
            result.errors.emplace_back(
                "DeepSeek CPU-MoE callback rejected upstream status " +
                std::to_string(first_upstream_failure));
        } else {
            result.errors.emplace_back("DeepSeek CPU-MoE callback failed");
        }
        return result;
    }
    static_cast<void>(output_to_mhc);
    if (!routed_output.empty()) {
        std::memcpy(routed_output.data(), host_bytes,
                    static_cast<std::size_t>(routed_bytes));
    }
    if (!shared_output.empty()) {
        std::memcpy(shared_output.data(),
                    host_bytes + static_cast<std::ptrdiff_t>(routed_bytes),
                    static_cast<std::size_t>(shared_bytes));
    }
    return result;
}

ValidationResult CudaBackend::finish_deepseek_moe_chain(int device) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek MoE chain finish targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    const auto clear_shared_phase_timing = [&]() noexcept {
        state.moe_shared_phase_timing_valid = false;
    };
    if (!state.moe_in_flight && state.moe_host_callback_count == 0U) {
        clear_shared_phase_timing();
        result.errors.emplace_back(
            "DeepSeek MoE chain finish has no matching in-flight command");
        return result;
    }
    if (state.moe_host_staging == nullptr || state.moe_error == nullptr) {
        clear_shared_phase_timing();
        result.errors.emplace_back(
            "DeepSeek MoE chain finish has no fixed status staging");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        clear_shared_phase_timing();
        return cuda_error(status, "select CUDA device for DeepSeek MoE chain finish");
    }
    auto* host_status = reinterpret_cast<unsigned int*>(state.moe_host_staging);
    if (auto status = cudaMemcpyAsync(
            host_status, state.moe_error, sizeof(*host_status),
            cudaMemcpyDeviceToHost, state.stream); status != cudaSuccess) {
        clear_shared_phase_timing();
        return cuda_error(status, "stage DeepSeek MoE chain status");
    }
    if (auto status = cudaEventRecord(state.moe_completed, state.stream);
        status != cudaSuccess) {
        clear_shared_phase_timing();
        return cuda_error(status, "record DeepSeek MoE chain finish");
    }
    if (auto status = cudaEventSynchronize(state.moe_completed);
        status != cudaSuccess) {
        clear_shared_phase_timing();
        return cuda_error(status, "synchronize DeepSeek MoE chain finish");
    }
    // The attention host node belongs to this device's stream.  A paired
    // rank-local finish must inspect and clear only this device's commands;
    // clearing the other rank here would release a callback before its stream
    // has drained.  Cross-device collection keeps its existing paired path.
    for (std::uint32_t index = 0U;
         index < state.dsv4_attention_prepare_host_command_count; ++index) {
        if (state.dsv4_attention_prepare_host_commands[index].failed) {
            result.errors.emplace_back(
                "DeepSeek attention preparation host callback failed");
        }
        state.dsv4_attention_prepare_host_commands[index] = {};
    }
    state.dsv4_attention_prepare_host_command_count = 0U;
    state.dsv4_deferred_attention_command_count = 0U;
    std::array<std::uint64_t, 4U> shared_phase_nanoseconds{};
    bool shared_phase_timing_ok = true;
    if (state.moe_shared_phase_timing_valid) {
        const auto measure_shared_phase = [&](std::size_t index,
                                              cudaEvent_t begin,
                                              cudaEvent_t end,
                                              const char* operation) {
            float milliseconds = 0.0F;
            if (auto status = cudaEventElapsedTime(&milliseconds, begin, end);
                status != cudaSuccess) {
                result.errors.emplace_back(
                    std::string(operation) + ": " +
                    cudaGetErrorString(status));
                shared_phase_timing_ok = false;
                return;
            }
            shared_phase_nanoseconds[index] = static_cast<std::uint64_t>(
                std::llround(static_cast<double>(milliseconds) * 1.0e6));
        };
        measure_shared_phase(
            0U, state.moe_hidden_uploaded,
            state.moe_shared_input_finished,
            "measure DeepSeek shared input quantization");
        measure_shared_phase(
            1U, state.moe_shared_input_finished,
            state.moe_shared_gate_up_finished,
            "measure DeepSeek shared gate/up");
        measure_shared_phase(
            2U, state.moe_shared_gate_up_finished,
            state.moe_shared_activation_finished,
            "measure DeepSeek shared activation quantization");
        measure_shared_phase(
            3U, state.moe_shared_activation_finished,
            state.moe_shared_finished,
            "measure DeepSeek shared down");
    }
    if (shared_phase_timing_ok && state.moe_shared_phase_timing_valid) {
        std::scoped_lock lock(impl_->mutex);
        auto& device_stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        device_stats.deepseek_moe_input_quantize_nanoseconds +=
            shared_phase_nanoseconds[0U];
        device_stats.deepseek_moe_shared_gate_up_nanoseconds +=
            shared_phase_nanoseconds[1U];
        device_stats.deepseek_moe_shared_activation_quantize_nanoseconds +=
            shared_phase_nanoseconds[2U];
        device_stats.deepseek_moe_shared_down_nanoseconds +=
            shared_phase_nanoseconds[3U];
    }
    clear_shared_phase_timing();
    bool callback_failed = false;
    for (std::uint32_t index = 0U;
         index < state.moe_host_callback_count; ++index) {
        callback_failed = callback_failed ||
            state.moe_host_callbacks[index].failed;
    }
    if (callback_failed) {
        result.errors.emplace_back("DeepSeek host-MoE callback failed");
    }
    if (*host_status != 0U) {
        result.errors.emplace_back("DeepSeek host-MoE device status failed");
    }
    {
        std::scoped_lock lock(impl_->mutex);
        auto& device_stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        ++device_stats.deepseek_moe_d2h_transfers;
        device_stats.deepseek_moe_d2h_bytes += sizeof(*host_status);
    }
    state.moe_in_flight = false;
    state.moe_weights.clear();
    state.moe_host_join = false;
    state.moe_output_to_mhc = false;
    state.moe_host_callback = {};
    for (std::uint32_t index = 0U;
         index < state.moe_host_callback_count; ++index) {
        state.moe_host_callbacks[index] = {};
    }
    state.moe_host_callback_count = 0U;
    return result;
}

ValidationResult CudaBackend::collect_moe(
    int device, std::span<float> routed_output,
    std::span<float> shared_output) {
    return collect_deepseek_moe(device, routed_output, shared_output);
}

ValidationResult CudaBackend::synchronize(int device) {
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) return {{"cannot synchronize an uninitialized CUDA device"}};
    if (found->second.moe_in_flight ||
        found->second.moe_host_callback_count != 0U) {
        return {{"use collect_deepseek_moe for an in-flight DeepSeek MoE command"}};
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for synchronization");
    }
    const auto started = std::chrono::steady_clock::now();
    const auto status = cudaStreamSynchronize(found->second.stream);
    const auto elapsed = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count());
    if (status == cudaSuccess) {
        std::scoped_lock lock(impl_->mutex);
        auto& device_stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        record_synchronization(device_stats,
                               SynchronizationSubsystem::Other, 1U,
                               elapsed);
    }
    return cuda_error(status, "synchronize CUDA device");
}

CudaBackendStats CudaBackend::stats() const noexcept {
    std::scoped_lock lock(impl_->mutex);
    auto result = impl_->stats;
    for (const auto& device : result.devices) {
        result.weight_upload_bytes += device.weight_upload_bytes;
        result.activation_h2d_bytes += device.activation_h2d_bytes;
        result.activation_d2h_bytes += device.activation_d2h_bytes;
        result.matmul_calls += device.matmul_calls;
        result.weight_allocation_calls += device.weight_allocation_calls;
        result.weight_allocation_bytes += device.weight_allocation_bytes;
        result.workspace_allocation_calls += device.workspace_allocation_calls;
        result.workspace_allocation_bytes += device.workspace_allocation_bytes;
        result.synchronization_calls += device.synchronization_calls;
        result.weight_synchronization.calls +=
            device.weight_synchronization.calls;
        result.attention_synchronization.calls +=
            device.attention_synchronization.calls;
        result.projection_synchronization.calls +=
            device.projection_synchronization.calls;
        result.mhc_synchronization.calls +=
            device.mhc_synchronization.calls;
        result.moe_synchronization.calls +=
            device.moe_synchronization.calls;
        result.other_synchronization.calls +=
            device.other_synchronization.calls;
        if (device.synchronization_nanoseconds >
            result.synchronization_nanoseconds) {
            result.synchronization_nanoseconds =
                device.synchronization_nanoseconds;
            result.weight_synchronization.nanoseconds =
                device.weight_synchronization.nanoseconds;
            result.attention_synchronization.nanoseconds =
                device.attention_synchronization.nanoseconds;
            result.projection_synchronization.nanoseconds =
                device.projection_synchronization.nanoseconds;
            result.mhc_synchronization.nanoseconds =
                device.mhc_synchronization.nanoseconds;
            result.moe_synchronization.nanoseconds =
                device.moe_synchronization.nanoseconds;
            result.other_synchronization.nanoseconds =
                device.other_synchronization.nanoseconds;
        }
        result.upload_wait_nanoseconds = std::max(
            result.upload_wait_nanoseconds, device.upload_wait_nanoseconds);
        result.weight_allocation_nanoseconds = std::max(
            result.weight_allocation_nanoseconds,
            device.weight_allocation_nanoseconds);
        result.weight_copy_nanoseconds = std::max(
            result.weight_copy_nanoseconds, device.weight_copy_nanoseconds);
        result.activation_h2d_nanoseconds = std::max(
            result.activation_h2d_nanoseconds, device.activation_h2d_nanoseconds);
        result.kernel_nanoseconds = std::max(
            result.kernel_nanoseconds, device.kernel_nanoseconds);
        result.activation_d2h_nanoseconds = std::max(
            result.activation_d2h_nanoseconds, device.activation_d2h_nanoseconds);
        result.deepseek_moe_calls += device.deepseek_moe_calls;
        result.deepseek_moe_kernel_launches += device.deepseek_moe_kernel_launches;
        result.deepseek_moe_h2d_transfers += device.deepseek_moe_h2d_transfers;
        result.deepseek_moe_d2h_transfers += device.deepseek_moe_d2h_transfers;
        result.deepseek_moe_h2d_bytes += device.deepseek_moe_h2d_bytes;
        result.deepseek_moe_d2h_bytes += device.deepseek_moe_d2h_bytes;
        result.deepseek_moe_h2d_nanoseconds = std::max(
            result.deepseek_moe_h2d_nanoseconds,
            device.deepseek_moe_h2d_nanoseconds);
        result.deepseek_moe_kernel_nanoseconds = std::max(
            result.deepseek_moe_kernel_nanoseconds,
            device.deepseek_moe_kernel_nanoseconds);
        result.deepseek_moe_d2h_nanoseconds = std::max(
            result.deepseek_moe_d2h_nanoseconds,
            device.deepseek_moe_d2h_nanoseconds);
        result.deepseek_moe_nanoseconds = std::max(
            result.deepseek_moe_nanoseconds, device.deepseek_moe_nanoseconds);
        result.flash_attention_calls += device.flash_attention_calls;
        result.flash_attention_kernel_launches +=
            device.flash_attention_kernel_launches;
        result.flash_attention_h2d_transfers +=
            device.flash_attention_h2d_transfers;
        result.flash_attention_d2h_transfers +=
            device.flash_attention_d2h_transfers;
        result.flash_attention_h2d_bytes += device.flash_attention_h2d_bytes;
        result.flash_attention_d2h_bytes += device.flash_attention_d2h_bytes;
        result.flash_attention_useful_staging_bytes +=
            device.flash_attention_useful_staging_bytes;
        result.flash_attention_wasted_staging_bytes +=
            device.flash_attention_wasted_staging_bytes;
        result.flash_attention_h2d_nanoseconds = std::max(
            result.flash_attention_h2d_nanoseconds,
            device.flash_attention_h2d_nanoseconds);
        result.flash_attention_kernel_nanoseconds = std::max(
            result.flash_attention_kernel_nanoseconds,
            device.flash_attention_kernel_nanoseconds);
        result.flash_attention_d2h_nanoseconds = std::max(
            result.flash_attention_d2h_nanoseconds,
            device.flash_attention_d2h_nanoseconds);
        result.flash_attention_nanoseconds = std::max(
            result.flash_attention_nanoseconds,
            device.flash_attention_nanoseconds);
        result.dsv4_paged_attention_calls +=
            device.dsv4_paged_attention_calls;
        result.dsv4_paged_attention_kernel_launches +=
            device.dsv4_paged_attention_kernel_launches;
        result.dsv4_paged_attention_h2d_bytes +=
            device.dsv4_paged_attention_h2d_bytes;
        result.dsv4_paged_attention_d2h_bytes +=
            device.dsv4_paged_attention_d2h_bytes;
        result.dsv4_paged_attention_page_bytes +=
            device.dsv4_paged_attention_page_bytes;
        result.dsv4_paged_attention_h2d_nanoseconds = std::max(
            result.dsv4_paged_attention_h2d_nanoseconds,
            device.dsv4_paged_attention_h2d_nanoseconds);
        result.dsv4_paged_attention_kernel_nanoseconds = std::max(
            result.dsv4_paged_attention_kernel_nanoseconds,
            device.dsv4_paged_attention_kernel_nanoseconds);
        result.dsv4_paged_attention_d2h_nanoseconds = std::max(
            result.dsv4_paged_attention_d2h_nanoseconds,
            device.dsv4_paged_attention_d2h_nanoseconds);
        result.dsv4_paged_attention_nanoseconds = std::max(
            result.dsv4_paged_attention_nanoseconds,
            device.dsv4_paged_attention_nanoseconds);
        result.dsv4_paged_attention_host_remainder_nanoseconds +=
            device.dsv4_paged_attention_host_remainder_nanoseconds;
        result.dsv4_paged_attention_stream_sync_nanoseconds +=
            device.dsv4_paged_attention_stream_sync_nanoseconds;
        result.dsv4_mhc_calls += device.dsv4_mhc_calls;
        result.dsv4_mhc_standalone_calls +=
            device.dsv4_mhc_standalone_calls;
        result.dsv4_mhc_transition_calls +=
            device.dsv4_mhc_transition_calls;
        result.dsv4_mhc_final_calls += device.dsv4_mhc_final_calls;
        result.dsv4_mhc_kernel_launches +=
            device.dsv4_mhc_kernel_launches;
        result.dsv4_mhc_resident_weight_bytes +=
            device.dsv4_mhc_resident_weight_bytes;
        result.dsv4_mhc_h2d_bytes += device.dsv4_mhc_h2d_bytes;
        result.dsv4_mhc_d2h_bytes += device.dsv4_mhc_d2h_bytes;
        result.dsv4_mhc_h2d_nanoseconds = std::max(
            result.dsv4_mhc_h2d_nanoseconds,
            device.dsv4_mhc_h2d_nanoseconds);
        result.dsv4_mhc_kernel_nanoseconds = std::max(
            result.dsv4_mhc_kernel_nanoseconds,
            device.dsv4_mhc_kernel_nanoseconds);
        result.dsv4_mhc_d2h_nanoseconds = std::max(
            result.dsv4_mhc_d2h_nanoseconds,
            device.dsv4_mhc_d2h_nanoseconds);
        result.dsv4_mhc_nanoseconds = std::max(
            result.dsv4_mhc_nanoseconds,
            device.dsv4_mhc_nanoseconds);
        result.dsv4_mhc_device_nanoseconds = std::max(
            result.dsv4_mhc_device_nanoseconds,
            device.dsv4_mhc_device_nanoseconds);
        result.dsv4_mhc_host_nanoseconds = std::max(
            result.dsv4_mhc_host_nanoseconds,
            device.dsv4_mhc_host_nanoseconds);
        result.dsv4_mhc_timing_clamped_samples +=
            device.dsv4_mhc_timing_clamped_samples;
        result.lightning_index_calls += device.lightning_index_calls;
        result.lightning_index_kernel_launches +=
            device.lightning_index_kernel_launches;
        result.lightning_index_candidates +=
            device.lightning_index_candidates;
        result.lightning_index_selected += device.lightning_index_selected;
        result.lightning_index_h2d_transfers +=
            device.lightning_index_h2d_transfers;
        result.lightning_index_d2h_transfers +=
            device.lightning_index_d2h_transfers;
        result.lightning_index_h2d_bytes +=
            device.lightning_index_h2d_bytes;
        result.lightning_index_d2h_bytes +=
            device.lightning_index_d2h_bytes;
        result.lightning_index_useful_selection_bytes +=
            device.lightning_index_useful_selection_bytes;
        result.lightning_index_h2d_nanoseconds = std::max(
            result.lightning_index_h2d_nanoseconds,
            device.lightning_index_h2d_nanoseconds);
        result.lightning_index_kernel_nanoseconds = std::max(
            result.lightning_index_kernel_nanoseconds,
            device.lightning_index_kernel_nanoseconds);
        result.lightning_index_d2h_nanoseconds = std::max(
            result.lightning_index_d2h_nanoseconds,
            device.lightning_index_d2h_nanoseconds);
        result.lightning_index_nanoseconds = std::max(
            result.lightning_index_nanoseconds,
            device.lightning_index_nanoseconds);
    }
    return result;
}
