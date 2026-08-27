ValidationResult CudaBackend::glm_absorbed_attention(
    const CudaWeight& key_value_projection,
    const CudaGlmAbsorbedAttentionRequest& request,
    std::span<float> output) {
    ValidationResult result;
    if (!key_value_projection.valid()) {
        result.errors.emplace_back(
            "GLM absorbed attention received an invalid projection");
        return result;
    }
    const auto& descriptor = key_value_projection.impl_->descriptor;
    constexpr std::uint64_t projection_rows =
        static_cast<std::uint64_t>(kGlmHeads) * (kGlmNope + kGlmValue);
    if (descriptor.encoding != CudaWeightEncoding::OffsetPackedInt4 ||
        descriptor.dtype != SafetensorsDtype::I32 ||
        descriptor.rows != projection_rows ||
        descriptor.columns != kGlmLatent || descriptor.group_size != 128U ||
        descriptor.packed_columns != kGlmLatent / 8U ||
        descriptor.scale_columns != kGlmLatent / 128U) {
        result.errors.emplace_back(
            "GLM absorbed attention requires the target OffsetPackedInt4 kv_b projection");
        return result;
    }
    const auto query_rows = request.causal_key_counts.size();
    if (query_rows == 0U || query_rows > 65'535U ||
        request.queries.size() !=
            query_rows * kGlmHeads * (kGlmNope + kGlmRope) ||
        request.latent.empty() || request.latent.size() % kGlmLatent != 0U ||
        output.size() != query_rows * kGlmHeads * kGlmValue ||
        !std::isfinite(request.scale) || request.scale <= 0.0F) {
        result.errors.emplace_back("GLM absorbed attention activation shape is invalid");
        return result;
    }
    const auto key_rows = request.latent.size() / kGlmLatent;
    if (key_rows > 2'048U ||
        request.rope.size() != key_rows * kGlmRope ||
        std::any_of(request.causal_key_counts.begin(),
                    request.causal_key_counts.end(),
                    [key_rows](std::uint32_t rows) {
                        return rows == 0U || rows > key_rows;
                    })) {
        result.errors.emplace_back(
            "GLM absorbed attention causal window is invalid or exceeds 2,048 tokens");
        return result;
    }
    const auto found = impl_->devices.find(key_value_projection.impl_->device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "GLM absorbed attention targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    if (state.moe_in_flight) {
        result.errors.emplace_back(
            "GLM absorbed attention cannot overlap an in-flight MoE command");
        return result;
    }
    if (auto status = cudaSetDevice(key_value_projection.impl_->device);
        status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for GLM absorbed attention");
    }

    const auto query_bytes = static_cast<std::uint64_t>(request.queries.size_bytes());
    const auto latent_bytes = static_cast<std::uint64_t>(request.latent.size_bytes());
    const auto rope_bytes = static_cast<std::uint64_t>(request.rope.size_bytes());
    const auto limit_bytes = static_cast<std::uint64_t>(
        request.causal_key_counts.size_bytes());
    const auto output_bytes = static_cast<std::uint64_t>(output.size_bytes());
    if (query_bytes > std::numeric_limits<std::uint64_t>::max() - latent_bytes ||
        query_bytes + latent_bytes >
            std::numeric_limits<std::uint64_t>::max() - rope_bytes ||
        query_bytes + latent_bytes + rope_bytes >
            std::numeric_limits<std::uint64_t>::max() - limit_bytes ||
        output_bytes > std::numeric_limits<std::uint64_t>::max() -
                           sizeof(unsigned int)) {
        result.errors.emplace_back("GLM absorbed attention workspace size overflows");
        return result;
    }
    const auto latent_offset = query_bytes;
    const auto rope_offset = latent_offset + latent_bytes;
    const auto limit_offset = rope_offset + rope_bytes;
    const auto input_bytes = limit_offset + limit_bytes;
    const auto error_offset = output_bytes;
    const auto output_workspace_bytes = output_bytes + sizeof(unsigned int);
    if (input_bytes > request.maximum_workspace_bytes ||
        output_workspace_bytes > request.maximum_workspace_bytes - input_bytes) {
        result.errors.emplace_back(
            "GLM absorbed attention exceeds its bounded CUDA workspace");
        return result;
    }

    std::uint64_t allocation_calls = 0U;
    std::uint64_t allocation_bytes = 0U;
    if (input_bytes > state.input_bytes) {
        if (state.input != nullptr) static_cast<void>(cudaFree(state.input));
        if (auto status = cudaMalloc(
                &state.input, static_cast<std::size_t>(input_bytes));
            status != cudaSuccess) {
            state.input = nullptr;
            state.input_bytes = 0U;
            return cuda_error(status,
                              "allocate GLM absorbed attention input workspace");
        }
        state.input_bytes = input_bytes;
        ++allocation_calls;
        allocation_bytes += input_bytes;
    }
    if (output_workspace_bytes > state.output_bytes) {
        if (state.output != nullptr) static_cast<void>(cudaFree(state.output));
        if (auto status = cudaMalloc(
                &state.output, static_cast<std::size_t>(output_workspace_bytes));
            status != cudaSuccess) {
            state.output = nullptr;
            state.output_bytes = 0U;
            return cuda_error(status,
                              "allocate GLM absorbed attention output workspace");
        }
        state.output_bytes = output_workspace_bytes;
        ++allocation_calls;
        allocation_bytes += output_workspace_bytes;
    }

    auto* device_input = reinterpret_cast<std::byte*>(state.input);
    auto* device_output = reinterpret_cast<std::byte*>(state.output);
    auto* device_error = reinterpret_cast<unsigned int*>(
        device_output + error_offset);
    const auto operation_started = std::chrono::steady_clock::now();
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_start, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "record GLM absorbed attention upload start");
        }
    }
    for (const auto& copy : {
             std::tuple{device_input,
                        static_cast<const void*>(request.queries.data()), query_bytes},
             std::tuple{device_input + latent_offset,
                        static_cast<const void*>(request.latent.data()), latent_bytes},
             std::tuple{device_input + rope_offset,
                        static_cast<const void*>(request.rope.data()), rope_bytes},
             std::tuple{device_input + limit_offset,
                        static_cast<const void*>(request.causal_key_counts.data()),
                        limit_bytes}}) {
        if (auto status = cudaMemcpyAsync(
                std::get<0>(copy), std::get<1>(copy),
                static_cast<std::size_t>(std::get<2>(copy)),
                cudaMemcpyHostToDevice, state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "upload GLM absorbed attention inputs");
        }
    }
    if (auto status = cudaMemsetAsync(device_error, 0, sizeof(*device_error),
                                      state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "clear GLM absorbed attention status");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_uploaded, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "record GLM absorbed attention upload completion");
        }
    }
    const dim3 grid(kGlmHeads, static_cast<unsigned int>(query_rows), 1U);
    constexpr unsigned int threads = 256U;
    const auto shared_bytes =
        static_cast<std::size_t>(2U * kGlmLatent + key_rows) * sizeof(float);
    glm_absorbed_attention_kernel<<<grid, threads, shared_bytes, state.stream>>>(
        reinterpret_cast<float*>(device_output),
        reinterpret_cast<const float*>(device_input),
        reinterpret_cast<const float*>(device_input + latent_offset),
        reinterpret_cast<const float*>(device_input + rope_offset),
        reinterpret_cast<const std::uint32_t*>(device_input + limit_offset),
        static_cast<const std::uint32_t*>(key_value_projection.impl_->weights),
        static_cast<const __nv_bfloat16*>(key_value_projection.impl_->scales),
        static_cast<std::uint32_t>(query_rows),
        static_cast<std::uint32_t>(key_rows), request.scale, device_error);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch GLM absorbed attention");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.kernel_finished, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "record GLM absorbed attention kernel completion");
        }
    }
    unsigned int numerical_error = 0U;
    if (auto status = cudaMemcpyAsync(
            output.data(), device_output, static_cast<std::size_t>(output_bytes),
            cudaMemcpyDeviceToHost, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "download GLM absorbed attention output");
    }
    if (auto status = cudaMemcpyAsync(
            &numerical_error, device_error, sizeof(numerical_error),
            cudaMemcpyDeviceToHost, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "download GLM absorbed attention status");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_downloaded, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "record GLM absorbed attention download completion");
        }
    }
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaStreamSynchronize(state.stream); status != cudaSuccess) {
        return cuda_error(status, "synchronize GLM absorbed attention");
    }
    const auto wait_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wait_started).count());
    const auto operation_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - operation_started).count());
    std::uint64_t h2d_nanoseconds = 0U;
    std::uint64_t kernel_nanoseconds = 0U;
    std::uint64_t d2h_nanoseconds = 0U;
    if (impl_->detailed_timing) {
        float h2d_ms = 0.0F;
        float kernel_ms = 0.0F;
        float d2h_ms = 0.0F;
        if (auto status = cudaEventElapsedTime(
                &h2d_ms, state.activation_start, state.activation_uploaded);
            status != cudaSuccess) {
            return cuda_error(status, "measure GLM absorbed attention upload");
        }
        if (auto status = cudaEventElapsedTime(
                &kernel_ms, state.activation_uploaded, state.kernel_finished);
            status != cudaSuccess) {
            return cuda_error(status, "measure GLM absorbed attention kernel");
        }
        if (auto status = cudaEventElapsedTime(
                &d2h_ms, state.kernel_finished, state.activation_downloaded);
            status != cudaSuccess) {
            return cuda_error(status, "measure GLM absorbed attention download");
        }
        h2d_nanoseconds = static_cast<std::uint64_t>(
            std::llround(static_cast<double>(h2d_ms) * 1.0e6));
        kernel_nanoseconds = static_cast<std::uint64_t>(
            std::llround(static_cast<double>(kernel_ms) * 1.0e6));
        d2h_nanoseconds = static_cast<std::uint64_t>(
            std::llround(static_cast<double>(d2h_ms) * 1.0e6));
    }
    {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [&](const auto& value) {
                return value.device == key_value_projection.impl_->device;
            });
        stats.activation_h2d_bytes += input_bytes;
        stats.activation_d2h_bytes += output_workspace_bytes;
        stats.matmul_calls += 2U;
        stats.workspace_allocation_calls += allocation_calls;
        stats.workspace_allocation_bytes += allocation_bytes;
        record_synchronization(stats, SynchronizationSubsystem::Other,
                               1U, wait_nanoseconds);
        stats.activation_h2d_nanoseconds += h2d_nanoseconds;
        stats.kernel_nanoseconds += kernel_nanoseconds;
        stats.activation_d2h_nanoseconds += d2h_nanoseconds;
        ++stats.flash_attention_calls;
        ++stats.flash_attention_kernel_launches;
        stats.flash_attention_h2d_transfers += 4U;
        stats.flash_attention_d2h_transfers += 2U;
        stats.flash_attention_h2d_bytes += input_bytes;
        stats.flash_attention_d2h_bytes += output_workspace_bytes;
        stats.flash_attention_useful_staging_bytes += latent_bytes + rope_bytes;
        stats.flash_attention_h2d_nanoseconds += h2d_nanoseconds;
        stats.flash_attention_kernel_nanoseconds += kernel_nanoseconds;
        stats.flash_attention_d2h_nanoseconds += d2h_nanoseconds;
        stats.flash_attention_nanoseconds += operation_nanoseconds;
    }
    if (numerical_error != 0U) {
        result.errors.emplace_back(
            numerical_error == 1U
                ? "GLM absorbed attention score is non-finite"
                : numerical_error == 2U
                    ? "GLM absorbed attention softmax denominator is invalid"
                    : "GLM absorbed attention output is non-finite");
    }
    return result;
}
