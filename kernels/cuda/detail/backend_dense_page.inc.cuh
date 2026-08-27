ValidationResult CudaBackend::upload_gemma4_kv(
    const CudaBuffer& cache, std::span<const std::uint16_t> keys,
    std::span<const std::uint16_t> values, std::uint32_t start,
    std::uint32_t capacity_rows, std::uint32_t columns) {
    ValidationResult result;
    if (!cache.valid() || capacity_rows == 0U || columns == 0U ||
        keys.size() != values.size() || keys.size() % columns != 0U ||
        keys.size() / columns > capacity_rows) {
        result.errors.emplace_back("Gemma 4 CUDA KV upload shape is invalid");
        return result;
    }
    std::uint64_t plane_bytes = 0U;
    if (!checked_bytes(capacity_rows, columns, sizeof(std::uint16_t),
                       plane_bytes) ||
        plane_bytes > std::numeric_limits<std::uint64_t>::max() / 2U ||
        cache.device_bytes() != plane_bytes * 2U) {
        result.errors.emplace_back("Gemma 4 CUDA KV cache capacity is invalid");
        return result;
    }
    if (keys.empty()) return result;
    auto& state = impl_->devices.at(cache.impl_->device);
    if (auto status = cudaSetDevice(cache.impl_->device);
        status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for Gemma 4 KV upload");
    }
    const auto rows = static_cast<std::uint32_t>(keys.size() / columns);
    const auto physical = start % capacity_rows;
    const auto first_rows = std::min(rows, capacity_rows - physical);
    const auto first_elements = static_cast<std::size_t>(first_rows) * columns;
    const auto second_elements = keys.size() - first_elements;
    auto* device_keys = static_cast<std::uint16_t*>(cache.impl_->data);
    auto* device_values = reinterpret_cast<std::uint16_t*>(
        static_cast<std::byte*>(cache.impl_->data) + plane_bytes);
    const auto copy_plane = [&](std::uint16_t* destination,
                                std::span<const std::uint16_t> source) {
        auto status = cudaMemcpyAsync(
            destination + static_cast<std::size_t>(physical) * columns,
            source.data(), first_elements * sizeof(std::uint16_t),
            cudaMemcpyHostToDevice, state.stream);
        if (status == cudaSuccess && second_elements != 0U) {
            status = cudaMemcpyAsync(
                destination, source.data() + first_elements,
                second_elements * sizeof(std::uint16_t),
                cudaMemcpyHostToDevice, state.stream);
        }
        return status;
    };
    if (auto status = copy_plane(device_keys, keys); status != cudaSuccess) {
        return cuda_error(status, "upload Gemma 4 CUDA keys");
    }
    if (auto status = copy_plane(device_values, values); status != cudaSuccess) {
        return cuda_error(status, "upload Gemma 4 CUDA values");
    }
    if (auto status = cudaStreamSynchronize(state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "synchronize Gemma 4 CUDA KV upload");
    }
    return result;
}

ValidationResult CudaBackend::bf16_kv_attention(
    int device, const CudaBf16KvAttentionRequest& request,
    std::span<float> output) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    const auto query_elements = static_cast<std::uint64_t>(request.query_heads) *
                                request.head_dim;
    const auto kv_elements = static_cast<std::uint64_t>(request.key_value_heads) *
                             request.head_dim;
    std::uint64_t plane_bytes = 0U;
    const auto relative_bias_elements =
        static_cast<std::uint64_t>(request.query_heads) *
        request.relative_bias_extent;
    const bool relative_bias_valid =
        (request.relative_bias.empty() && request.relative_bias_extent == 0U) ||
        (request.relative_bias_extent != 0U &&
         request.relative_bias.size() == relative_bias_elements &&
         std::all_of(request.relative_bias.begin(), request.relative_bias.end(),
                     [](float value) { return std::isfinite(value); }));
    if (found == impl_->devices.end() || request.cache == nullptr ||
        !request.cache->valid() || request.cache->device() != device ||
        request.query_heads == 0U || request.key_value_heads == 0U ||
        request.query_heads % request.key_value_heads != 0U ||
        request.head_dim == 0U || request.head_dim > 1'024U ||
        request.capacity_rows == 0U || request.cached_rows == 0U ||
        request.cached_rows > request.capacity_rows ||
        request.queries.size() != query_elements || output.size() != query_elements ||
        request.next_keys.size() != kv_elements ||
        request.next_values.size() != kv_elements ||
        request.cache_start > request.position ||
        static_cast<std::uint64_t>(request.cache_start) + request.cached_rows !=
            static_cast<std::uint64_t>(request.position) + 1U ||
        !std::isfinite(request.scale) || request.scale <= 0.0F ||
        !checked_bytes(request.capacity_rows, kv_elements,
                       sizeof(std::uint16_t), plane_bytes) ||
        plane_bytes > std::numeric_limits<std::uint64_t>::max() / 2U ||
        request.cache->device_bytes() != plane_bytes * 2U ||
        !relative_bias_valid ||
        std::any_of(request.queries.begin(), request.queries.end(),
                    [](float value) { return !std::isfinite(value); })) {
        result.errors.emplace_back("BF16 KV attention request is invalid");
        return result;
    }
    auto& state = found->second;
    if (!state.flash_attention_supported) {
        result.errors.emplace_back(
            "BF16 KV attention CUDA kernel supports only SM86 and SM120 devices");
        return result;
    }
    if (state.moe_in_flight) {
        result.errors.emplace_back(
            "BF16 KV attention cannot overlap an in-flight MoE command");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for BF16 KV attention");
    }

    const auto query_bytes = static_cast<std::uint64_t>(request.queries.size_bytes());
    const auto row_bytes = static_cast<std::uint64_t>(request.next_keys.size_bytes());
    const auto relative_bias_bytes =
        static_cast<std::uint64_t>(request.relative_bias.size_bytes());
    const auto upload_bytes = query_bytes + 2U * row_bytes + relative_bias_bytes;
    const auto output_bytes = static_cast<std::uint64_t>(output.size_bytes());
    const auto download_bytes = output_bytes + sizeof(unsigned int);
    std::uint64_t score_bytes = 0U;
    if (!checked_bytes(request.query_heads, request.capacity_rows,
                       sizeof(float), score_bytes)) {
        result.errors.emplace_back("BF16 KV attention score workspace overflows");
        return result;
    }
    std::uint64_t allocation_calls = 0U;
    std::uint64_t allocation_bytes = 0U;
    const auto ensure_device = [&](std::byte*& pointer, std::uint64_t& capacity,
                                   std::uint64_t required,
                                   const char* operation) {
        if (required <= capacity) return true;
        std::byte* replacement = nullptr;
        if (auto status = cudaMalloc(&replacement, static_cast<std::size_t>(required));
            status != cudaSuccess) {
            result = cuda_error(status, operation);
            return false;
        }
        if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
        pointer = replacement;
        capacity = required;
        ++allocation_calls;
        allocation_bytes += required;
        return true;
    };
    const auto ensure_host = [&](std::byte*& pointer, std::uint64_t& capacity,
                                 std::uint64_t required,
                                 const char* operation) {
        if (required <= capacity) return true;
        void* replacement = nullptr;
        if (auto status = cudaMallocHost(&replacement,
                                         static_cast<std::size_t>(required));
            status != cudaSuccess) {
            result = cuda_error(status, operation);
            return false;
        }
        if (pointer != nullptr) static_cast<void>(cudaFreeHost(pointer));
        pointer = static_cast<std::byte*>(replacement);
        capacity = required;
        return true;
    };
    if (!ensure_device(state.attention_upload, state.attention_upload_bytes,
                       upload_bytes, "allocate BF16 KV attention upload") ||
        !ensure_device(state.attention_download, state.attention_download_bytes,
                       download_bytes, "allocate BF16 KV attention download") ||
        !ensure_host(state.attention_host_upload,
                     state.attention_host_upload_bytes, upload_bytes,
                     "allocate BF16 KV attention host upload") ||
        !ensure_host(state.attention_host_download,
                     state.attention_host_download_bytes, download_bytes,
                     "allocate BF16 KV attention host download")) {
        return result;
    }
    if (score_bytes > state.attention_score_bytes) {
        float* replacement = nullptr;
        if (auto status = cudaMalloc(
                &replacement, static_cast<std::size_t>(score_bytes));
            status != cudaSuccess) {
            return cuda_error(status, "allocate BF16 KV attention scores");
        }
        if (state.attention_scores != nullptr) {
            static_cast<void>(cudaFree(state.attention_scores));
        }
        state.attention_scores = replacement;
        state.attention_score_bytes = score_bytes;
        ++allocation_calls;
        allocation_bytes += score_bytes;
    }

    std::memcpy(state.attention_host_upload, request.queries.data(), query_bytes);
    std::memcpy(state.attention_host_upload + query_bytes,
                request.next_keys.data(), row_bytes);
    std::memcpy(state.attention_host_upload + query_bytes + row_bytes,
                request.next_values.data(), row_bytes);
    if (relative_bias_bytes != 0U) {
        std::memcpy(state.attention_host_upload + query_bytes + 2U * row_bytes,
                    request.relative_bias.data(), relative_bias_bytes);
    }
    const auto operation_started = std::chrono::steady_clock::now();
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_start, state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "record BF16 KV attention upload start");
        }
    }
    if (auto status = cudaMemcpyAsync(
            state.attention_upload, state.attention_host_upload,
            static_cast<std::size_t>(upload_bytes), cudaMemcpyHostToDevice,
            state.stream); status != cudaSuccess) {
        return cuda_error(status, "upload BF16 KV attention inputs");
    }
    auto* cache_keys = static_cast<__nv_bfloat16*>(request.cache->impl_->data);
    auto* cache_values = reinterpret_cast<__nv_bfloat16*>(
        static_cast<std::byte*>(request.cache->impl_->data) + plane_bytes);
    const auto physical = request.position % request.capacity_rows;
    auto* device_next_keys = reinterpret_cast<const __nv_bfloat16*>(
        state.attention_upload + query_bytes);
    auto* device_next_values = reinterpret_cast<const __nv_bfloat16*>(
        state.attention_upload + query_bytes + row_bytes);
    auto* device_relative_bias = reinterpret_cast<const float*>(
        state.attention_upload + query_bytes + 2U * row_bytes);
    if (auto status = cudaMemcpyAsync(
            cache_keys + static_cast<std::uint64_t>(physical) * kv_elements,
            device_next_keys, static_cast<std::size_t>(row_bytes),
            cudaMemcpyDeviceToDevice, state.stream); status != cudaSuccess) {
        return cuda_error(status, "store BF16 KV attention key row");
    }
    if (auto status = cudaMemcpyAsync(
            cache_values + static_cast<std::uint64_t>(physical) * kv_elements,
            device_next_values, static_cast<std::size_t>(row_bytes),
            cudaMemcpyDeviceToDevice, state.stream); status != cudaSuccess) {
        return cuda_error(status, "store BF16 KV attention value row");
    }
    auto* device_output = reinterpret_cast<float*>(state.attention_download);
    auto* device_error = reinterpret_cast<unsigned int*>(
        state.attention_download + output_bytes);
    if (auto status = cudaMemsetAsync(device_error, 0, sizeof(*device_error),
                                      state.stream); status != cudaSuccess) {
        return cuda_error(status, "clear BF16 KV attention status");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_uploaded, state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "record BF16 KV attention upload completion");
        }
    }
    bf16_kv_attention_reference_all_f32_kernel<<<
        request.query_heads, 256U, 0U, state.stream>>>(
        device_output, state.attention_scores,
        reinterpret_cast<const float*>(state.attention_upload), cache_keys,
        cache_values, device_relative_bias, request.relative_bias_extent,
        request.query_heads, request.key_value_heads,
        request.head_dim, request.capacity_rows, request.cache_start,
        request.cached_rows, request.scale, device_error);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch BF16 KV attention");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.kernel_finished, state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "record BF16 KV attention kernel completion");
        }
    }
    if (auto status = cudaMemcpyAsync(
            state.attention_host_download, state.attention_download,
            static_cast<std::size_t>(download_bytes), cudaMemcpyDeviceToHost,
            state.stream); status != cudaSuccess) {
        return cuda_error(status, "download BF16 KV attention output");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_downloaded, state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "record BF16 KV attention download completion");
        }
    }
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaStreamSynchronize(state.stream); status != cudaSuccess) {
        return cuda_error(status, "synchronize BF16 KV attention");
    }
    const auto wait_nanoseconds = elapsed_nanoseconds_since(wait_started);
    const auto operation_nanoseconds = elapsed_nanoseconds_since(operation_started);
    unsigned int numerical_error = 0U;
    std::memcpy(&numerical_error, state.attention_host_download + output_bytes,
                sizeof(numerical_error));
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
            return cuda_error(status, "measure BF16 KV attention upload");
        }
        if (auto status = cudaEventElapsedTime(
                &kernel_ms, state.activation_uploaded, state.kernel_finished);
            status != cudaSuccess) {
            return cuda_error(status, "measure BF16 KV attention kernel");
        }
        if (auto status = cudaEventElapsedTime(
                &d2h_ms, state.kernel_finished, state.activation_downloaded);
            status != cudaSuccess) {
            return cuda_error(status, "measure BF16 KV attention download");
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
            [device](const auto& value) { return value.device == device; });
        ++stats.flash_attention_calls;
        ++stats.flash_attention_kernel_launches;
        ++stats.flash_attention_h2d_transfers;
        ++stats.flash_attention_d2h_transfers;
        stats.flash_attention_h2d_bytes += upload_bytes;
        stats.flash_attention_d2h_bytes += download_bytes;
        stats.flash_attention_useful_staging_bytes +=
            2U * row_bytes + relative_bias_bytes;
        stats.flash_attention_h2d_nanoseconds += h2d_nanoseconds;
        stats.flash_attention_kernel_nanoseconds += kernel_nanoseconds;
        stats.flash_attention_d2h_nanoseconds += d2h_nanoseconds;
        stats.flash_attention_nanoseconds += operation_nanoseconds;
        stats.workspace_allocation_calls += allocation_calls;
        stats.workspace_allocation_bytes += allocation_bytes;
        record_synchronization(stats, SynchronizationSubsystem::Attention, 1U,
                               wait_nanoseconds);
    }
    if (numerical_error != 0U) {
        result.errors.emplace_back(
            numerical_error == 1U
                ? "BF16 KV attention score is non-finite"
                : numerical_error == 2U
                    ? "BF16 KV attention softmax denominator is invalid"
                    : "BF16 KV attention output is non-finite");
        return result;
    }
    std::memcpy(output.data(), state.attention_host_download, output_bytes);
    return result;
}

ValidationResult CudaBackend::reserve_gemma4_workspace(
    int device, std::uint32_t hidden_columns,
    std::uint32_t maximum_query_columns,
    std::uint32_t maximum_kv_columns,
    std::uint32_t maximum_intermediate_columns) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end() || hidden_columns == 0U ||
        maximum_query_columns == 0U || maximum_kv_columns == 0U ||
        maximum_intermediate_columns == 0U) {
        result.errors.emplace_back(
            "Gemma 4 CUDA workspace reservation is invalid");
        return result;
    }
    auto& state = found->second;
    constexpr std::uint64_t padded_rows = 128U;
    const std::uint64_t workspace_elements =
        padded_rows * (static_cast<std::uint64_t>(hidden_columns) * 3U +
                       static_cast<std::uint64_t>(maximum_query_columns) * 2U +
                       static_cast<std::uint64_t>(maximum_kv_columns) * 2U +
                       static_cast<std::uint64_t>(
                           maximum_intermediate_columns) * 2U);
    std::uint64_t workspace_bytes = 0U;
    std::uint64_t hidden_bytes = 0U;
    if (!checked_bytes(1U, workspace_elements, sizeof(float), workspace_bytes) ||
        !checked_bytes(padded_rows, hidden_columns, sizeof(float),
                       hidden_bytes) ||
        hidden_bytes >
            (std::numeric_limits<std::uint64_t>::max() -
             sizeof(unsigned int)) / 2U) {
        result.errors.emplace_back(
            "Gemma 4 CUDA workspace reservation overflows");
        return result;
    }
    const auto host_bytes = hidden_bytes * 2U + sizeof(unsigned int);
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status,
                          "select CUDA device for Gemma 4 workspace");
    }
    const auto ensure_device = [&](auto*& pointer, std::uint64_t& capacity,
                                   std::uint64_t required) {
        if (required <= capacity) return cudaSuccess;
        using Pointer = std::remove_reference_t<decltype(pointer)>;
        Pointer replacement = nullptr;
        auto status = cudaMalloc(&replacement, static_cast<std::size_t>(required));
        if (status != cudaSuccess) return status;
        if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
        pointer = replacement;
        capacity = required;
        return cudaSuccess;
    };
    if (auto status = ensure_device(state.gemma_workspace,
                                    state.gemma_workspace_bytes,
                                    workspace_bytes);
        status != cudaSuccess) {
        return cuda_error(status, "reserve Gemma 4 page workspace");
    }
    if (state.gemma_error == nullptr) {
        if (auto status = cudaMalloc(&state.gemma_error,
                                     sizeof(*state.gemma_error));
            status != cudaSuccess) {
            return cuda_error(status, "reserve Gemma 4 page status");
        }
    }
    if (host_bytes > state.gemma_host_staging_bytes) {
        void* replacement = nullptr;
        if (auto status = cudaMallocHost(
                &replacement, static_cast<std::size_t>(host_bytes));
            status != cudaSuccess) {
            return cuda_error(status, "reserve Gemma 4 pinned staging");
        }
        if (state.gemma_host_staging != nullptr) {
            static_cast<void>(cudaFreeHost(state.gemma_host_staging));
        }
        state.gemma_host_staging = static_cast<std::byte*>(replacement);
        state.gemma_host_staging_bytes = host_bytes;
    }
    int multiprocessors = 0;
    if (auto status = cudaDeviceGetAttribute(
            &multiprocessors, cudaDevAttrMultiProcessorCount, device);
        status != cudaSuccess) {
        return cuda_error(status, "query Gemma 4 Marlin SM count");
    }
    int maximum_shared = 0;
    if (auto status = cudaDeviceGetAttribute(
            &maximum_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        status != cudaSuccess) {
        return cuda_error(status,
                          "query Gemma 4 page attention shared memory");
    }
    if (auto status = cudaFuncSetAttribute(
            gemma4_prefill_attention_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, maximum_shared);
        status != cudaSuccess) {
        return cuda_error(status,
                          "configure Gemma 4 page attention shared memory");
    }
    if (auto status = cudaFuncSetAttribute(
            gemma4_grouped_prefill_attention_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, maximum_shared);
        status != cudaSuccess) {
        return cuda_error(
            status, "configure grouped Gemma 4 page attention shared memory");
    }
    const auto activation_columns = std::max({
        hidden_columns, maximum_query_columns, maximum_kv_columns,
        maximum_intermediate_columns});
    const auto activation_bytes = padded_rows * activation_columns *
                                  sizeof(__nv_bfloat16);
    const auto reduce_bytes = static_cast<std::uint64_t>(multiprocessors) *
                              64U * 256U * sizeof(float);
    const auto reorder_bytes = static_cast<std::uint64_t>(multiprocessors) *
                               64U * 264U * sizeof(float);
    const auto lock_bytes = static_cast<std::uint64_t>(multiprocessors) *
                            sizeof(int);
    auto& marlin = state.gemma_marlin;
    if (auto status = regfed_grow(marlin.activation, marlin.activation_bytes,
                                  activation_bytes, false, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "reserve Gemma 4 Marlin activation");
    }
    if (auto status = regfed_grow(marlin.reduce, marlin.reduce_bytes,
                                  reduce_bytes, false, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "reserve Gemma 4 Marlin reduction");
    }
    if (auto status = regfed_grow(marlin.reorder, marlin.reorder_bytes,
                                  reorder_bytes, false, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "reserve Gemma 4 Marlin reorder");
    }
    if (auto status = regfed_grow(marlin.locks, marlin.lock_bytes,
                                  lock_bytes, true, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "reserve Gemma 4 Marlin locks");
    }
    return result;
}

ValidationResult CudaBackend::gemma4_prefill_layers(
    int device, std::span<const CudaGemma4DecodeLayer> layers,
    std::span<const float> input, std::uint32_t rows,
    std::uint32_t position_base, std::span<float> output) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end() || layers.empty() || rows < 2U ||
        rows > 128U || input.empty() ||
        input.size() % rows != 0U || output.size() != input.size()) {
        result.errors.emplace_back(
            "Gemma 4 CUDA page prefill request is invalid");
        return result;
    }
    auto& state = found->second;
    if (state.moe_in_flight) {
        result.errors.emplace_back(
            "Gemma 4 CUDA prefill cannot overlap an in-flight MoE command");
        return result;
    }
    const auto hidden_columns = static_cast<std::uint32_t>(input.size() / rows);
    std::uint64_t maximum_query_columns = 0U;
    std::uint64_t maximum_kv_columns = 0U;
    std::uint64_t maximum_intermediate = 0U;
    std::uint64_t matmul_calls = 0U;
    const auto valid_buffer = [device](const CudaBuffer* buffer,
                                       std::uint64_t bytes) {
        return buffer != nullptr && buffer->valid() &&
               buffer->device() == device && buffer->device_bytes() == bytes;
    };
    const auto valid_weight = [device](const CudaWeight* weight) {
        return weight != nullptr && weight->valid() &&
               weight->device() == device && weight->impl_->marlin_prepacked &&
               weight->impl_->descriptor.encoding ==
                   CudaWeightEncoding::Fp4E2m1Group32;
    };
    for (const auto& layer : layers) {
        if (!valid_weight(layer.query) || !valid_weight(layer.key) ||
            (layer.value != nullptr && !valid_weight(layer.value)) ||
            !valid_weight(layer.output) || !valid_weight(layer.gate) ||
            !valid_weight(layer.up) || !valid_weight(layer.down) ||
            layer.cached_rows > layer.cache_capacity_rows ||
            static_cast<std::uint64_t>(layer.cache_start) +
                    layer.cached_rows != position_base ||
            layer.cache_capacity_rows == 0U || !std::isfinite(layer.scalar)) {
            result.errors.emplace_back(
                "Gemma 4 CUDA page layer contract is invalid");
            return result;
        }
        const auto& query = layer.query->impl_->descriptor;
        const auto& key = layer.key->impl_->descriptor;
        const auto& projection = layer.output->impl_->descriptor;
        const auto& gate = layer.gate->impl_->descriptor;
        const auto& up = layer.up->impl_->descriptor;
        const auto& down = layer.down->impl_->descriptor;
        if (query.columns != hidden_columns || key.columns != hidden_columns ||
            projection.rows != hidden_columns ||
            projection.columns != query.rows || gate.columns != hidden_columns ||
            up.columns != hidden_columns || gate.rows != up.rows ||
            down.rows != hidden_columns || down.columns != gate.rows ||
            (layer.value != nullptr &&
             (layer.value->impl_->descriptor.columns != hidden_columns ||
              layer.value->impl_->descriptor.rows != key.rows))) {
            result.errors.emplace_back(
                "Gemma 4 CUDA page weight shapes are invalid");
            return result;
        }
        if (layer.query_norm == nullptr || layer.key_norm == nullptr ||
            layer.query_norm->device_bytes() != layer.key_norm->device_bytes() ||
            layer.query_norm->device_bytes() == 0U ||
            layer.query_norm->device_bytes() % sizeof(float) != 0U) {
            result.errors.emplace_back(
                "Gemma 4 CUDA page attention norm shape is invalid");
            return result;
        }
        const auto head_dim = static_cast<std::uint32_t>(
            layer.query_norm->device_bytes() / sizeof(float));
        if (!valid_buffer(layer.query_norm, head_dim * sizeof(float)) ||
            !valid_buffer(layer.key_norm, head_dim * sizeof(float)) ||
            query.rows % head_dim != 0U || key.rows % head_dim != 0U ||
            query.rows / head_dim == 0U || key.rows / head_dim == 0U ||
            (query.rows / head_dim) % (key.rows / head_dim) != 0U) {
            result.errors.emplace_back(
                "Gemma 4 CUDA page attention heads are invalid");
            return result;
        }
        const auto norm_bytes =
            static_cast<std::uint64_t>(hidden_columns) * sizeof(float);
        if (!valid_buffer(layer.input_norm, norm_bytes) ||
            !valid_buffer(layer.post_attention_norm, norm_bytes) ||
            !valid_buffer(layer.pre_feedforward_norm, norm_bytes) ||
            !valid_buffer(layer.post_feedforward_norm, norm_bytes)) {
            result.errors.emplace_back(
                "Gemma 4 CUDA page norm buffers are invalid");
            return result;
        }
        std::uint64_t plane_bytes = 0U;
        if (!checked_bytes(layer.cache_capacity_rows, key.rows,
                           sizeof(std::uint16_t), plane_bytes) ||
            plane_bytes > std::numeric_limits<std::uint64_t>::max() / 2U ||
            !valid_buffer(layer.kv_cache, plane_bytes * 2U)) {
            result.errors.emplace_back(
                "Gemma 4 CUDA page KV buffer is invalid");
            return result;
        }
        maximum_query_columns = std::max(maximum_query_columns, query.rows);
        maximum_kv_columns = std::max(maximum_kv_columns, key.rows);
        maximum_intermediate = std::max(maximum_intermediate, gate.rows);
        matmul_calls += layer.value == nullptr ? 6U : 7U;
    }

    constexpr std::uint64_t padded_rows = 128U;
    const std::uint64_t workspace_elements =
        padded_rows * (static_cast<std::uint64_t>(hidden_columns) * 3U +
                       maximum_query_columns * 2U +
                       maximum_kv_columns * 2U +
                       maximum_intermediate * 2U);
    std::uint64_t workspace_bytes = 0U;
    if (!checked_bytes(1U, workspace_elements, sizeof(float), workspace_bytes)) {
        result.errors.emplace_back("Gemma 4 CUDA prefill workspace overflows");
        return result;
    }
    const auto hidden_bytes = static_cast<std::uint64_t>(input.size_bytes());
    const auto host_bytes = hidden_bytes * 2U + sizeof(unsigned int);
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for Gemma 4 prefill");
    }
    const auto ensure_device = [&](auto*& pointer, std::uint64_t& capacity,
                                   std::uint64_t required) {
        if (required <= capacity) return cudaSuccess;
        using Pointer = std::remove_reference_t<decltype(pointer)>;
        Pointer replacement = nullptr;
        auto status = cudaMalloc(&replacement, required);
        if (status != cudaSuccess) return status;
        if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
        pointer = replacement;
        capacity = required;
        return cudaSuccess;
    };
    if (auto status = ensure_device(state.gemma_workspace,
                                    state.gemma_workspace_bytes,
                                    workspace_bytes);
        status != cudaSuccess) {
        return cuda_error(status, "allocate Gemma 4 prefill workspace");
    }
    if (state.gemma_error == nullptr) {
        if (auto status = cudaMalloc(&state.gemma_error,
                                     sizeof(*state.gemma_error));
            status != cudaSuccess) {
            return cuda_error(status, "allocate Gemma 4 prefill status");
        }
    }
    if (host_bytes > state.gemma_host_staging_bytes) {
        void* replacement = nullptr;
        if (auto status = cudaMallocHost(&replacement, host_bytes);
            status != cudaSuccess) {
            return cuda_error(status, "allocate Gemma 4 prefill host staging");
        }
        if (state.gemma_host_staging != nullptr) {
            static_cast<void>(cudaFreeHost(state.gemma_host_staging));
        }
        state.gemma_host_staging = static_cast<std::byte*>(replacement);
        state.gemma_host_staging_bytes = host_bytes;
    }

    auto* cursor = state.gemma_workspace;
    auto take = [&](std::uint64_t columns) {
        auto* pointer = cursor;
        cursor += padded_rows * columns;
        return pointer;
    };
    auto* hidden = take(hidden_columns);
    auto* normalized = take(hidden_columns);
    auto* branch = take(hidden_columns);
    auto* queries = take(maximum_query_columns);
    auto* keys = take(maximum_kv_columns);
    auto* values = take(maximum_kv_columns);
    auto* context = take(maximum_query_columns);
    auto* gate_output = take(maximum_intermediate);
    auto* up_output = take(maximum_intermediate);

    std::memcpy(state.gemma_host_staging, input.data(), input.size_bytes());
    const auto operation_started = std::chrono::steady_clock::now();
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_start, state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "record Gemma 4 prefill upload start");
        }
    }
    if (auto status = cudaMemcpyAsync(hidden, state.gemma_host_staging,
                                      input.size_bytes(), cudaMemcpyHostToDevice,
                                      state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "upload Gemma 4 prefill hidden state");
    }
    if (auto status = cudaMemsetAsync(state.gemma_error, 0,
                                      sizeof(*state.gemma_error), state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "clear Gemma 4 prefill status");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_uploaded, state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "record Gemma 4 prefill upload completion");
        }
    }
    const auto launch_matrix = [&](const CudaWeight* weight,
                                   const float* activation,
                                   float* destination,
                                   bool reuse_activation = false) -> cudaError_t {
        const auto& descriptor = weight->impl_->descriptor;
        auto status = launch_gemma_marlin(
            state.gemma_marlin, descriptor, weight->impl_->weights,
            weight->impl_->scales, activation, rows, destination, state.stream,
            reuse_activation);
        if (status != cudaSuccess) return status;
        record_cuda_matmul_route(CudaMatmulRoute::GemmaMarlin);
        return cudaSuccess;
    };

    for (const auto& layer : layers) {
        const auto& query = layer.query->impl_->descriptor;
        const auto& key = layer.key->impl_->descriptor;
        const auto& intermediate = layer.gate->impl_->descriptor;
        const auto head_dim = static_cast<std::uint32_t>(
            layer.query_norm->device_bytes() / sizeof(float));
        const auto query_heads = static_cast<std::uint32_t>(query.rows / head_dim);
        const auto kv_heads = static_cast<std::uint32_t>(key.rows / head_dim);
        gemma4_rms_norm_rows_kernel<<<rows, 1024U, 0U, state.stream>>>(
            normalized, hidden,
            static_cast<const float*>(layer.input_norm->impl_->data), rows,
            hidden_columns, 1.0e-6F, state.gemma_error);
        if (auto status = launch_matrix(layer.query, normalized, queries);
            status != cudaSuccess) {
            return cuda_error(status, "launch Gemma 4 page query projection");
        }
        if (auto status = launch_matrix(layer.key, normalized, keys, true);
            status != cudaSuccess) {
            return cuda_error(status, "launch Gemma 4 page key projection");
        }
        if (layer.value == nullptr) {
            if (auto status = cudaMemcpyAsync(
                    values, keys, rows * key.rows * sizeof(float),
                    cudaMemcpyDeviceToDevice, state.stream);
                status != cudaSuccess) {
                return cuda_error(status, "copy Gemma 4 page shared K/V");
            }
        } else if (auto status = launch_matrix(
                       layer.value, normalized, values, true);
                   status != cudaSuccess) {
            return cuda_error(status, "launch Gemma 4 page value projection");
        }
        const bool global = layer.value == nullptr;
        const float theta = global ? 1'000'000.0F : 10'000.0F;
        const float proportion = global ? 0.25F : 1.0F;
        const dim3 query_grid(query_heads, rows, 1U);
        const dim3 qkv_grid(query_heads + 2U * kv_heads, rows, 1U);
        gemma4_norm_rope_qkv_rows_kernel<<<
            qkv_grid, 32U, 0U, state.stream>>>(
            queries, keys, values,
            static_cast<const float*>(layer.query_norm->impl_->data),
            static_cast<const float*>(layer.key_norm->impl_->data), rows,
            query_heads, kv_heads, head_dim, position_base, theta, proportion,
            state.gemma_error);
        auto* cache = static_cast<__nv_bfloat16*>(layer.kv_cache->impl_->data);
        const auto maximum_visible = std::min(
            position_base + rows, layer.cache_capacity_rows);
        const auto scalar_attention_shared = static_cast<std::size_t>(
            maximum_visible + 1U) * sizeof(float);
        const auto group_size = query_heads / kv_heads;
        const std::uint32_t queries_per_block = group_size == 2U ? 4U : 1U;
        const auto combined = group_size * queries_per_block;
        const auto grouped_attention_shared =
            static_cast<std::size_t>(combined) *
                (maximum_visible + 1U) * sizeof(float) +
            8U * head_dim * sizeof(__nv_bfloat16);
        if (maximum_visible >= 64U && combined <= 8U &&
            grouped_attention_shared <=
                static_cast<std::size_t>(
                    state.gemma_marlin.maximum_shared)) {
            const dim3 grouped_grid(
                kv_heads, (rows + queries_per_block - 1U) / queries_per_block,
                1U);
            gemma4_grouped_prefill_attention_kernel<<<
                grouped_grid, 256U, grouped_attention_shared, state.stream>>>(
                context, queries, keys, values, cache, rows, position_base,
                layer.cache_capacity_rows, query_heads, kv_heads, head_dim,
                queries_per_block, state.gemma_error);
        } else {
            gemma4_prefill_attention_kernel<<<
                query_grid, 128U, scalar_attention_shared, state.stream>>>(
                context, queries, keys, values, cache, rows, position_base,
                layer.cache_capacity_rows, query_heads, kv_heads, head_dim,
                state.gemma_error);
        }
        const dim3 store_grid(
            static_cast<unsigned int>((key.rows + 255U) / 256U), rows, 1U);
        gemma4_store_kv_rows_kernel<<<store_grid, 256U, 0U, state.stream>>>(
            cache, keys, values, position_base, rows,
            layer.cache_capacity_rows, static_cast<std::uint32_t>(key.rows));
        if (auto status = launch_matrix(layer.output, context, branch);
            status != cudaSuccess) {
            return cuda_error(status, "launch Gemma 4 page output projection");
        }
        gemma4_post_attention_rows_kernel<<<rows, 1024U, 0U, state.stream>>>(
            hidden, normalized, branch,
            static_cast<const float*>(
                layer.post_attention_norm->impl_->data),
            static_cast<const float*>(
                layer.pre_feedforward_norm->impl_->data),
            rows, hidden_columns, state.gemma_error);
        if (auto status = launch_matrix(layer.gate, normalized, gate_output);
            status != cudaSuccess) {
            return cuda_error(status, "launch Gemma 4 page gate projection");
        }
        if (auto status = launch_matrix(layer.up, normalized, up_output, true);
            status != cudaSuccess) {
            return cuda_error(status, "launch Gemma 4 page up projection");
        }
        const auto activation_elements =
            static_cast<std::uint64_t>(rows) * intermediate.rows;
        gemma4_geglu_kernel<<<
            static_cast<unsigned int>(
                std::min<std::uint64_t>(
                    (activation_elements + 255U) / 256U, 65535U)),
            256U, 0U, state.stream>>>(
            gate_output, up_output,
            static_cast<std::uint32_t>(activation_elements),
            state.gemma_error);
        if (auto status = launch_matrix(layer.down, gate_output, branch);
            status != cudaSuccess) {
            return cuda_error(status, "launch Gemma 4 page down projection");
        }
        gemma4_post_feedforward_rows_kernel<<<
            rows, 1024U, 0U, state.stream>>>(
            hidden, normalized, branch,
            static_cast<const float*>(
                layer.post_feedforward_norm->impl_->data),
            rows, hidden_columns, layer.scalar, state.gemma_error);
    }
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch Gemma 4 CUDA prefill kernels");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.kernel_finished, state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "record Gemma 4 prefill kernel completion");
        }
    }
    const auto output_offset = hidden_bytes;
    const auto error_offset = hidden_bytes * 2U;
    if (auto status = cudaMemcpyAsync(
            state.gemma_host_staging + output_offset, hidden, hidden_bytes,
            cudaMemcpyDeviceToHost, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "download Gemma 4 prefill hidden state");
    }
    if (auto status = cudaMemcpyAsync(
            state.gemma_host_staging + error_offset, state.gemma_error,
            sizeof(*state.gemma_error), cudaMemcpyDeviceToHost, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "download Gemma 4 prefill status");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_downloaded,
                                          state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "record Gemma 4 prefill download");
        }
    }
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaStreamSynchronize(state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "synchronize Gemma 4 CUDA prefill");
    }
    const auto wait_nanoseconds = elapsed_nanoseconds_since(wait_started);
    unsigned int numerical_error = 0U;
    std::memcpy(&numerical_error, state.gemma_host_staging + error_offset,
                sizeof(numerical_error));
    if (numerical_error != 0U) {
        result.errors.emplace_back(
            "Gemma 4 CUDA prefill encountered a non-finite intermediate");
        return result;
    }
    std::memcpy(output.data(), state.gemma_host_staging + output_offset,
                output.size_bytes());

    std::uint64_t h2d_nanoseconds = 0U;
    std::uint64_t kernel_nanoseconds = 0U;
    std::uint64_t d2h_nanoseconds = 0U;
    if (impl_->detailed_timing) {
        float h2d_ms = 0.0F, kernel_ms = 0.0F, d2h_ms = 0.0F;
        if (auto status = cudaEventElapsedTime(
                &h2d_ms, state.activation_start, state.activation_uploaded);
            status != cudaSuccess) {
            return cuda_error(status, "measure Gemma 4 prefill upload");
        }
        if (auto status = cudaEventElapsedTime(
                &kernel_ms, state.activation_uploaded, state.kernel_finished);
            status != cudaSuccess) {
            return cuda_error(status, "measure Gemma 4 prefill kernels");
        }
        if (auto status = cudaEventElapsedTime(
                &d2h_ms, state.kernel_finished, state.activation_downloaded);
            status != cudaSuccess) {
            return cuda_error(status, "measure Gemma 4 prefill download");
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
            [device](const auto& value) { return value.device == device; });
        stats.matmul_calls += matmul_calls;
        stats.activation_h2d_bytes += hidden_bytes;
        stats.activation_d2h_bytes += hidden_bytes;
        stats.activation_h2d_nanoseconds += h2d_nanoseconds;
        stats.kernel_nanoseconds += kernel_nanoseconds;
        stats.activation_d2h_nanoseconds += d2h_nanoseconds;
        record_synchronization(stats, SynchronizationSubsystem::Projection, 1U,
                               wait_nanoseconds);
    }
    (void)operation_started;
    return result;
}

ValidationResult CudaBackend::gemma4_decode_layers(
    int device, std::span<const CudaGemma4DecodeLayer> layers,
    std::span<const float> input, std::uint32_t position,
    std::span<float> output) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end() || layers.empty() || input.empty() ||
        output.size() != input.size() ||
        input.size() > std::numeric_limits<std::uint32_t>::max()) {
        result.errors.emplace_back("Gemma 4 CUDA decode request is invalid");
        return result;
    }
    auto& state = found->second;
    if (state.moe_in_flight) {
        result.errors.emplace_back(
            "Gemma 4 CUDA decode cannot overlap an in-flight MoE command");
        return result;
    }
    const auto hidden_columns = static_cast<std::uint32_t>(input.size());
    std::uint64_t maximum_query_columns = 0U;
    std::uint64_t maximum_kv_columns = 0U;
    std::uint64_t maximum_intermediate = 0U;
    std::uint64_t score_elements = 0U;
    std::uint64_t next_kv_bytes = 0U;
    std::uint64_t matmul_calls = 0U;
    const auto valid_buffer = [device](const CudaBuffer* buffer,
                                       std::uint64_t bytes) {
        return buffer != nullptr && buffer->valid() &&
               buffer->device() == device && buffer->device_bytes() == bytes;
    };
    const bool regfed_enabled = regfed_matmul_enabled();
    const auto valid_weight = [device, regfed_enabled](const CudaWeight* weight) {
        if (weight == nullptr || !weight->valid() || weight->device() != device) {
            return false;
        }
        const auto& descriptor = weight->impl_->descriptor;
        const bool w8a16 =
            descriptor.encoding == CudaWeightEncoding::OffsetPackedInt8 &&
            descriptor.columns % 4U == 0U;
        const bool mxfp4 =
            descriptor.encoding == CudaWeightEncoding::Fp4E2m1Group32 &&
            descriptor.columns % 2U == 0U;
        if ((!w8a16 && !mxfp4) || descriptor.group_size != 32U) return false;
        return weight->impl_->marlin_prepacked ||
               !weight->impl_->fragment_prepacked ||
               (mxfp4 && regfed_enabled &&
                regfed_fp4_shape_admissible(descriptor.rows,
                                             descriptor.columns));
    };
    for (const auto& layer : layers) {
        const std::array<const CudaWeight*, 7U> layer_weights{
            layer.query, layer.key, layer.value, layer.output,
            layer.gate, layer.up, layer.down};
        for (const auto* weight : layer_weights) {
            if (weight == nullptr || !weight->valid() ||
                !weight->impl_->fragment_prepacked) {
                continue;
            }
            const auto& descriptor = weight->impl_->descriptor;
            if (!regfed_enabled ||
                descriptor.encoding != CudaWeightEncoding::Fp4E2m1Group32 ||
                !regfed_fp4_shape_admissible(descriptor.rows,
                                              descriptor.columns)) {
                result.errors.emplace_back(
                    "Gemma 4 CUDA decode received a fragment-prepacked weight "
                    "without an admissible enabled register-fed route; "
                    "refusing a canonical read");
                return result;
            }
        }
        for (const auto* weight : layer_weights) {
            if (weight != nullptr && weight->impl_->marlin_prepacked &&
                weight->impl_->descriptor.encoding !=
                    CudaWeightEncoding::Fp4E2m1Group32) {
                result.errors.emplace_back(
                    "Gemma 4 CUDA decode received a non-MXFP4 Marlin layout");
                return result;
            }
        }
        if (!valid_weight(layer.query) || !valid_weight(layer.key) ||
            (layer.value != nullptr && !valid_weight(layer.value)) ||
            !valid_weight(layer.output) || !valid_weight(layer.gate) ||
            !valid_weight(layer.up) || !valid_weight(layer.down) ||
            !std::isfinite(layer.scalar) || layer.cache_capacity_rows == 0U ||
            layer.cached_rows > layer.cache_capacity_rows ||
            static_cast<std::uint64_t>(layer.cache_start) + layer.cached_rows !=
                position) {
            result.errors.emplace_back("Gemma 4 CUDA decode layer contract is invalid");
            return result;
        }
        const auto& query = layer.query->impl_->descriptor;
        const auto& key = layer.key->impl_->descriptor;
        const auto& projection = layer.output->impl_->descriptor;
        const auto& gate = layer.gate->impl_->descriptor;
        const auto& up = layer.up->impl_->descriptor;
        const auto& down = layer.down->impl_->descriptor;
        if (query.columns != hidden_columns || key.columns != hidden_columns ||
            projection.rows != hidden_columns || projection.columns != query.rows ||
            gate.columns != hidden_columns || up.columns != hidden_columns ||
            gate.rows != up.rows || down.rows != hidden_columns ||
            down.columns != gate.rows ||
            (layer.value != nullptr &&
             (layer.value->impl_->descriptor.columns != hidden_columns ||
              layer.value->impl_->descriptor.rows != key.rows))) {
            result.errors.emplace_back("Gemma 4 CUDA decode weight shapes are invalid");
            return result;
        }
        if (layer.query_norm == nullptr || layer.key_norm == nullptr ||
            layer.query_norm->device_bytes() != layer.key_norm->device_bytes() ||
            layer.query_norm->device_bytes() == 0U ||
            layer.query_norm->device_bytes() % sizeof(float) != 0U) {
            result.errors.emplace_back("Gemma 4 CUDA attention norm shape is invalid");
            return result;
        }
        const auto head_dim = static_cast<std::uint32_t>(
            layer.query_norm->device_bytes() / sizeof(float));
        if (!valid_buffer(layer.query_norm, head_dim * sizeof(float)) ||
            !valid_buffer(layer.key_norm, head_dim * sizeof(float)) ||
            query.rows % head_dim != 0U || key.rows % head_dim != 0U ||
            query.rows / head_dim == 0U || key.rows / head_dim == 0U ||
            (query.rows / head_dim) % (key.rows / head_dim) != 0U) {
            result.errors.emplace_back("Gemma 4 CUDA attention head shape is invalid");
            return result;
        }
        const std::uint64_t norm_bytes =
            static_cast<std::uint64_t>(hidden_columns) * sizeof(float);
        if (!valid_buffer(layer.input_norm, norm_bytes) ||
            !valid_buffer(layer.post_attention_norm, norm_bytes) ||
            !valid_buffer(layer.pre_feedforward_norm, norm_bytes) ||
            !valid_buffer(layer.post_feedforward_norm, norm_bytes)) {
            result.errors.emplace_back("Gemma 4 CUDA layer norm buffer is invalid");
            return result;
        }
        std::uint64_t cache_plane_bytes = 0U;
        if (!checked_bytes(layer.cache_capacity_rows, key.rows,
                           sizeof(std::uint16_t), cache_plane_bytes) ||
            cache_plane_bytes > std::numeric_limits<std::uint64_t>::max() / 2U ||
            !valid_buffer(layer.kv_cache, cache_plane_bytes * 2U) ||
            layer.next_keys.empty() != layer.next_values.empty() ||
            (!layer.next_keys.empty() &&
             (layer.next_keys.size() != key.rows ||
              layer.next_values.size() != key.rows))) {
            result.errors.emplace_back("Gemma 4 CUDA KV buffer is invalid");
            return result;
        }
        maximum_query_columns = std::max(maximum_query_columns, query.rows);
        maximum_kv_columns = std::max(maximum_kv_columns, key.rows);
        maximum_intermediate = std::max(maximum_intermediate, gate.rows);
        score_elements = std::max(
            score_elements,
            (query.rows / head_dim) * layer.cache_capacity_rows);
        const auto layer_kv_bytes =
            static_cast<std::uint64_t>(layer.next_keys.size_bytes()) * 2U;
        if (layer_kv_bytes > std::numeric_limits<std::uint64_t>::max() -
                                 next_kv_bytes) {
            result.errors.emplace_back("Gemma 4 CUDA decode staging overflows");
            return result;
        }
        next_kv_bytes += layer_kv_bytes;
        matmul_calls += layer.value == nullptr ? 6U : 7U;
    }

    const std::uint64_t workspace_elements =
        static_cast<std::uint64_t>(hidden_columns) * 3U +
        maximum_query_columns * 2U + maximum_kv_columns * 2U +
        maximum_intermediate * 2U;
    std::uint64_t workspace_bytes = 0U;
    std::uint64_t score_bytes = 0U;
    if (!checked_bytes(1U, workspace_elements, sizeof(float), workspace_bytes) ||
        !checked_bytes(1U, score_elements, sizeof(float), score_bytes)) {
        result.errors.emplace_back("Gemma 4 CUDA decode workspace overflows");
        return result;
    }
    const auto hidden_bytes = static_cast<std::uint64_t>(input.size_bytes());
    if (hidden_bytes > (std::numeric_limits<std::uint64_t>::max() -
                        next_kv_bytes - sizeof(unsigned int)) / 2U) {
        result.errors.emplace_back("Gemma 4 CUDA host staging overflows");
        return result;
    }
    const auto host_bytes = hidden_bytes * 2U + next_kv_bytes +
                            sizeof(unsigned int);
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for Gemma 4 decode");
    }
    const auto ensure_device = [&](auto*& pointer, std::uint64_t& capacity,
                                   std::uint64_t required,
                                   const char* operation) {
        if (required <= capacity) return cudaSuccess;
        using Pointer = std::remove_reference_t<decltype(pointer)>;
        Pointer replacement = nullptr;
        auto status = cudaMalloc(&replacement, static_cast<std::size_t>(required));
        if (status != cudaSuccess) return status;
        if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
        pointer = replacement;
        capacity = required;
        (void)operation;
        return cudaSuccess;
    };
    if (auto status = ensure_device(
            state.gemma_workspace, state.gemma_workspace_bytes,
            workspace_bytes, "allocate Gemma 4 decode workspace");
        status != cudaSuccess) {
        return cuda_error(status, "allocate Gemma 4 decode workspace");
    }
    if (auto status = ensure_device(
            state.gemma_scores, state.gemma_score_bytes, score_bytes,
            "allocate Gemma 4 attention scores"); status != cudaSuccess) {
        return cuda_error(status, "allocate Gemma 4 attention scores");
    }
    if (state.gemma_error == nullptr) {
        if (auto status = cudaMalloc(&state.gemma_error,
                                     sizeof(*state.gemma_error));
            status != cudaSuccess) {
            return cuda_error(status, "allocate Gemma 4 decode status");
        }
    }
    if (host_bytes > state.gemma_host_staging_bytes) {
        void* replacement = nullptr;
        if (auto status = cudaMallocHost(
                &replacement, static_cast<std::size_t>(host_bytes));
            status != cudaSuccess) {
            return cuda_error(status, "allocate Gemma 4 pinned staging");
        }
        if (state.gemma_host_staging != nullptr) {
            static_cast<void>(cudaFreeHost(state.gemma_host_staging));
        }
        state.gemma_host_staging = static_cast<std::byte*>(replacement);
        state.gemma_host_staging_bytes = host_bytes;
    }

    auto* cursor = state.gemma_workspace;
    auto* hidden = cursor;
    cursor += hidden_columns;
    auto* normalized = cursor;
    cursor += hidden_columns;
    auto* branch = cursor;
    cursor += hidden_columns;
    auto* queries = cursor;
    cursor += maximum_query_columns;
    auto* keys = cursor;
    cursor += maximum_kv_columns;
    auto* values = cursor;
    cursor += maximum_kv_columns;
    auto* context = cursor;
    cursor += maximum_query_columns;
    auto* gate_output = cursor;
    cursor += maximum_intermediate;
    auto* up_output = cursor;
    std::memcpy(state.gemma_host_staging, input.data(), input.size_bytes());
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_start, state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "record Gemma 4 activation upload start");
        }
    }
    if (auto status = cudaMemcpyAsync(
            hidden, state.gemma_host_staging, input.size_bytes(),
            cudaMemcpyHostToDevice, state.stream); status != cudaSuccess) {
        return cuda_error(status, "upload Gemma 4 hidden state");
    }
    if (auto status = cudaMemsetAsync(
            state.gemma_error, 0, sizeof(*state.gemma_error), state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "clear Gemma 4 decode status");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_uploaded, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "record Gemma 4 activation upload completion");
        }
    }
    const auto launch_matvec = [&](const CudaWeight* weight,
                                   const float* activation,
                                   float* destination,
                                   bool reuse_activation = false) -> cudaError_t {
        const auto& descriptor = weight->impl_->descriptor;
        constexpr unsigned int threads = 256U;
        if (descriptor.encoding == CudaWeightEncoding::Fp4E2m1Group32) {
            if (weight->impl_->marlin_prepacked) {
                auto status = launch_gemma_marlin(
                    state.gemma_marlin, descriptor, weight->impl_->weights,
                    weight->impl_->scales, activation, 1U, destination,
                    state.stream, reuse_activation);
                if (status != cudaSuccess) return status;
                record_cuda_matmul_route(CudaMatmulRoute::GemmaMarlin);
                return cudaSuccess;
            }
            const dim3 quantize_grid(
                static_cast<unsigned int>((descriptor.columns + 127U) / 128U),
                1U, 1U);
            quantize_activation_e4m3_kernel<<<
                quantize_grid, 128U, 0U, state.stream>>>(
                const_cast<float*>(activation), descriptor.columns, 1U);
            if (auto status = cudaGetLastError(); status != cudaSuccess) {
                return status;
            }
            if (weight->impl_->fragment_prepacked) {
                const auto status = launch_regfed_fp4_matvec(
                    state.gemma_regfed, descriptor, weight->impl_->weights,
                    weight->impl_->scales, activation, destination,
                    state.stream);
                if (status == cudaSuccess) {
                    record_cuda_matmul_route(CudaMatmulRoute::Fp4RegisterFed);
                }
                return status;
            }
            const dim3 grid(static_cast<unsigned int>(descriptor.rows), 1U, 1U);
            native_fp4_matmul_kernel<<<grid, threads, 0U, state.stream>>>(
                destination, activation,
                static_cast<const unsigned char*>(weight->impl_->weights),
                static_cast<const unsigned char*>(weight->impl_->scales),
                descriptor.packed_columns, descriptor.scale_columns, 1U,
                descriptor.columns, descriptor.rows, 0U, 0U);
            record_cuda_matmul_route(CudaMatmulRoute::Fp4E2m1Group32);
            return cudaGetLastError();
        }
        constexpr unsigned int warps = threads / 32U;
        const auto blocks = static_cast<unsigned int>(
            (descriptor.rows + warps - 1U) / warps);
        packed_int8_group32_matvec_kernel<<<
            blocks, threads, 0U, state.stream>>>(
            destination, activation,
            static_cast<const std::uint32_t*>(weight->impl_->weights),
            static_cast<const __nv_bfloat16*>(weight->impl_->scales),
            descriptor.packed_columns, descriptor.scale_columns,
            descriptor.columns, descriptor.rows);
        record_cuda_matmul_route(CudaMatmulRoute::PackedInt8Group32);
        return cudaGetLastError();
    };
    for (const auto& layer : layers) {
        const auto& query = layer.query->impl_->descriptor;
        const auto& key = layer.key->impl_->descriptor;
        const auto& intermediate = layer.gate->impl_->descriptor;
        const auto head_dim = static_cast<std::uint32_t>(
            layer.query_norm->device_bytes() / sizeof(float));
        const auto query_heads = static_cast<std::uint32_t>(query.rows / head_dim);
        const auto kv_heads = static_cast<std::uint32_t>(key.rows / head_dim);
        gemma4_rms_norm_kernel<<<1U, 1024U, 0U, state.stream>>>(
            normalized, hidden,
            static_cast<const float*>(layer.input_norm->impl_->data),
            hidden_columns, 1.0e-6F, state.gemma_error);
        if (auto status = launch_matvec(layer.query, normalized, queries);
            status != cudaSuccess) {
            return cuda_error(status, "launch Gemma 4 query projection");
        }
        if (auto status = launch_matvec(layer.key, normalized, keys, true);
            status != cudaSuccess) {
            return cuda_error(status, "launch Gemma 4 key projection");
        }
        if (layer.value == nullptr) {
            if (auto status = cudaMemcpyAsync(
                    values, keys, key.rows * sizeof(float),
                    cudaMemcpyDeviceToDevice, state.stream);
                status != cudaSuccess) {
                return cuda_error(status, "copy Gemma 4 shared K/V projection");
            }
        } else {
            if (auto status = launch_matvec(
                    layer.value, normalized, values, true);
                status != cudaSuccess) {
                return cuda_error(status, "launch Gemma 4 value projection");
            }
        }
        const bool global = layer.value == nullptr;
        const float theta = global ? 1'000'000.0F : 10'000.0F;
        const float proportion = global ? 0.25F : 1.0F;
        gemma4_norm_rope_qkv_rows_kernel<<<
            query_heads + 2U * kv_heads, 32U, 0U, state.stream>>>(
            queries, keys, values,
            static_cast<const float*>(layer.query_norm->impl_->data),
            static_cast<const float*>(layer.key_norm->impl_->data), 1U,
            query_heads, kv_heads, head_dim, position, theta, proportion,
            state.gemma_error);
        auto* cache = static_cast<__nv_bfloat16*>(layer.kv_cache->impl_->data);
        gemma4_store_kv_kernel<<<
            static_cast<unsigned int>((key.rows + 255U) / 256U), 256U, 0U,
            state.stream>>>(cache, keys, values, position,
                            layer.cache_capacity_rows,
                            static_cast<std::uint32_t>(key.rows));
        gemma4_attention_kernel<<<query_heads, 256U, 0U, state.stream>>>(
            context, state.gemma_scores, queries, cache, position,
            layer.cache_capacity_rows, query_heads, kv_heads, head_dim,
            state.gemma_error);
        if (auto status = launch_matvec(layer.output, context, branch);
            status != cudaSuccess) {
            return cuda_error(status, "launch Gemma 4 output projection");
        }
        gemma4_post_attention_kernel<<<1U, 1024U, 0U, state.stream>>>(
            hidden, normalized, branch,
            static_cast<const float*>(layer.post_attention_norm->impl_->data),
            static_cast<const float*>(layer.pre_feedforward_norm->impl_->data),
            hidden_columns, state.gemma_error);
        if (auto status = launch_matvec(layer.gate, normalized, gate_output);
            status != cudaSuccess) {
            return cuda_error(status, "launch Gemma 4 gate projection");
        }
        if (auto status = launch_matvec(
                layer.up, normalized, up_output, true);
            status != cudaSuccess) {
            return cuda_error(status, "launch Gemma 4 up projection");
        }
        gemma4_geglu_kernel<<<
            static_cast<unsigned int>((intermediate.rows + 255U) / 256U),
            256U, 0U, state.stream>>>(
            gate_output, up_output,
            static_cast<std::uint32_t>(intermediate.rows), state.gemma_error);
        if (auto status = launch_matvec(layer.down, gate_output, branch);
            status != cudaSuccess) {
            return cuda_error(status, "launch Gemma 4 down projection");
        }
        gemma4_post_feedforward_kernel<<<1U, 1024U, 0U, state.stream>>>(
            hidden, normalized, branch,
            static_cast<const float*>(layer.post_feedforward_norm->impl_->data),
            hidden_columns, layer.scalar, state.gemma_error);
    }
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch Gemma 4 CUDA decode kernels");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.kernel_finished, state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "record Gemma 4 kernel completion");
        }
    }
    const auto output_offset = hidden_bytes;
    auto kv_offset = hidden_bytes * 2U;
    if (auto status = cudaMemcpyAsync(
            state.gemma_host_staging + output_offset, hidden,
            static_cast<std::size_t>(hidden_bytes), cudaMemcpyDeviceToHost,
            state.stream); status != cudaSuccess) {
        return cuda_error(status, "download Gemma 4 hidden state");
    }
    for (const auto& layer : layers) {
        if (layer.next_keys.empty()) continue;
        const auto columns = static_cast<std::uint64_t>(layer.next_keys.size());
        const auto bytes = columns * sizeof(std::uint16_t);
        const auto physical = position % layer.cache_capacity_rows;
        const auto plane = static_cast<std::uint64_t>(layer.cache_capacity_rows) *
                           columns;
        const auto* cache = static_cast<const std::uint16_t*>(
            layer.kv_cache->impl_->data);
        if (auto status = cudaMemcpyAsync(
                state.gemma_host_staging + kv_offset,
                cache + static_cast<std::uint64_t>(physical) * columns,
                static_cast<std::size_t>(bytes), cudaMemcpyDeviceToHost,
                state.stream); status != cudaSuccess) {
            return cuda_error(status, "download Gemma 4 next keys");
        }
        kv_offset += bytes;
        if (auto status = cudaMemcpyAsync(
                state.gemma_host_staging + kv_offset,
                cache + plane + static_cast<std::uint64_t>(physical) * columns,
                static_cast<std::size_t>(bytes), cudaMemcpyDeviceToHost,
                state.stream); status != cudaSuccess) {
            return cuda_error(status, "download Gemma 4 next values");
        }
        kv_offset += bytes;
    }
    const auto error_offset = hidden_bytes * 2U + next_kv_bytes;
    if (auto status = cudaMemcpyAsync(
            state.gemma_host_staging + error_offset, state.gemma_error,
            sizeof(*state.gemma_error), cudaMemcpyDeviceToHost, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "download Gemma 4 decode status");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_downloaded,
                                          state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "record Gemma 4 activation download completion");
        }
    }
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaStreamSynchronize(state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "synchronize Gemma 4 CUDA decode");
    }
    const auto wait_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wait_started).count());
    std::uint64_t activation_h2d_nanoseconds = 0U;
    std::uint64_t kernel_nanoseconds = 0U;
    std::uint64_t activation_d2h_nanoseconds = 0U;
    if (impl_->detailed_timing) {
        float h2d_milliseconds = 0.0F;
        float kernel_milliseconds = 0.0F;
        float d2h_milliseconds = 0.0F;
        if (auto status = cudaEventElapsedTime(
                &h2d_milliseconds, state.activation_start,
                state.activation_uploaded);
            status != cudaSuccess) {
            return cuda_error(status, "measure Gemma 4 activation upload");
        }
        if (auto status = cudaEventElapsedTime(
                &kernel_milliseconds, state.activation_uploaded,
                state.kernel_finished);
            status != cudaSuccess) {
            return cuda_error(status, "measure Gemma 4 CUDA kernels");
        }
        if (auto status = cudaEventElapsedTime(
                &d2h_milliseconds, state.kernel_finished,
                state.activation_downloaded);
            status != cudaSuccess) {
            return cuda_error(status, "measure Gemma 4 activation download");
        }
        activation_h2d_nanoseconds = static_cast<std::uint64_t>(
            std::llround(static_cast<double>(h2d_milliseconds) * 1.0e6));
        kernel_nanoseconds = static_cast<std::uint64_t>(
            std::llround(static_cast<double>(kernel_milliseconds) * 1.0e6));
        activation_d2h_nanoseconds = static_cast<std::uint64_t>(
            std::llround(static_cast<double>(d2h_milliseconds) * 1.0e6));
    }
    std::memcpy(output.data(), state.gemma_host_staging + output_offset,
                output.size_bytes());
    kv_offset = hidden_bytes * 2U;
    for (const auto& layer : layers) {
        if (layer.next_keys.empty()) continue;
        const auto bytes = layer.next_keys.size_bytes();
        std::memcpy(layer.next_keys.data(),
                    state.gemma_host_staging + kv_offset, bytes);
        kv_offset += bytes;
        std::memcpy(layer.next_values.data(),
                    state.gemma_host_staging + kv_offset, bytes);
        kv_offset += bytes;
    }
    unsigned int numerical_error = 0U;
    std::memcpy(&numerical_error, state.gemma_host_staging + error_offset,
                sizeof(numerical_error));
    {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        stats.activation_h2d_bytes += hidden_bytes;
        stats.activation_d2h_bytes += hidden_bytes + next_kv_bytes;
        stats.matmul_calls += matmul_calls;
        stats.flash_attention_calls += layers.size();
        stats.flash_attention_kernel_launches += layers.size();
        stats.activation_h2d_nanoseconds += activation_h2d_nanoseconds;
        stats.kernel_nanoseconds += kernel_nanoseconds;
        stats.activation_d2h_nanoseconds += activation_d2h_nanoseconds;
        record_synchronization(stats, SynchronizationSubsystem::Other, 1U,
                               wait_nanoseconds);
    }
    if (numerical_error != 0U) {
        result.errors.emplace_back("Gemma 4 CUDA decode produced a non-finite value");
    }
    return result;
}
