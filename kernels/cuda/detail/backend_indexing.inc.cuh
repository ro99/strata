ValidationResult CudaBackend::matmul(const CudaWeight& weight,
                                     std::span<const float> input,
                                     std::uint32_t rows,
                                     std::span<float> output,
                                     bool round_bf16_output,
                                     CudaMatmulProfile* profile,
                                     bool dsv4_fp8_tensor_page) {
    return matmul_impl(weight, input, rows, 0U, 0U, output, 0.0F,
                       round_bf16_output, profile,
                       dsv4_fp8_tensor_page);
}

ValidationResult CudaBackend::matmul_batch(
    std::span<const CudaMatmulBatchItem> items) {
    ValidationResult result;
    if (items.empty() || items.front().weight == nullptr ||
        !items.front().weight->valid()) {
        result.errors.emplace_back("CUDA matmul batch is empty or invalid");
        return result;
    }
    const int device = items.front().weight->device();
    if (impl_->detailed_timing) {
        for (const auto& item : items) {
            if (item.weight == nullptr || item.weight->device() != device) {
                return {{"CUDA matmul batch spans invalid or different devices"}};
            }
            auto completed = matmul(*item.weight, item.input, item.rows,
                                    item.output, item.round_bf16_output, nullptr,
                                    item.fp8_tensor_page);
            if (!completed.ok()) return completed;
        }
        return result;
    }

    std::vector<std::uint64_t> input_offsets(items.size());
    std::vector<std::uint64_t> output_offsets(items.size());
    std::uint64_t input_bytes = 0U;
    std::uint64_t output_bytes = 0U;
    for (std::size_t index = 0U; index < items.size(); ++index) {
        const auto& item = items[index];
        if (item.weight == nullptr || !item.weight->valid() ||
            item.weight->device() != device || item.input.empty() ||
            item.output.empty() || item.rows == 0U ||
            item.input.size_bytes() >
                std::numeric_limits<std::uint64_t>::max() - input_bytes ||
            item.output.size_bytes() >
                std::numeric_limits<std::uint64_t>::max() - output_bytes) {
            result.errors.emplace_back(
                "CUDA matmul batch spans invalid or different devices");
            return result;
        }
        input_offsets[index] = input_bytes;
        output_offsets[index] = output_bytes;
        input_bytes += item.input.size_bytes();
        output_bytes += item.output.size_bytes();
    }
    auto& state = impl_->devices.at(device);
    const auto grow_pinned = [](std::byte*& pointer, std::uint64_t& capacity,
                                std::uint64_t required) -> cudaError_t {
        if (required <= capacity) return cudaSuccess;
        void* replacement = nullptr;
        if (auto status = cudaMallocHost(&replacement,
                                         static_cast<std::size_t>(required));
            status != cudaSuccess) {
            return status;
        }
        if (pointer != nullptr) static_cast<void>(cudaFreeHost(pointer));
        pointer = static_cast<std::byte*>(replacement);
        capacity = required;
        return cudaSuccess;
    };
    if (auto status = grow_pinned(state.matmul_host_input,
                                  state.matmul_host_input_bytes, input_bytes);
        status != cudaSuccess) {
        return cuda_error(status, "allocate CUDA matmul batch input staging");
    }
    if (auto status = grow_pinned(state.matmul_host_output,
                                  state.matmul_host_output_bytes, output_bytes);
        status != cudaSuccess) {
        return cuda_error(status, "allocate CUDA matmul batch output staging");
    }
    for (std::size_t index = 0U; index < items.size(); ++index) {
        std::memcpy(state.matmul_host_input + input_offsets[index],
                    items[index].input.data(), items[index].input.size_bytes());
    }
    for (std::size_t index = 0U; index < items.size(); ++index) {
        const auto& item = items[index];
        auto issued = matmul_impl(
            *item.weight, item.input, item.rows, 0U, 0U, item.output, 0.0F,
            item.round_bf16_output, nullptr, item.fp8_tensor_page,
            state.matmul_host_input + input_offsets[index],
            state.matmul_host_output + output_offsets[index], true);
        if (!issued.ok()) {
            static_cast<void>(cudaStreamSynchronize(state.stream));
            return issued;
        }
    }
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaStreamSynchronize(state.stream); status != cudaSuccess) {
        return cuda_error(status, "synchronize CUDA matmul batch");
    }
    for (std::size_t index = 0U; index < items.size(); ++index) {
        std::memcpy(items[index].output.data(),
                    state.matmul_host_output + output_offsets[index],
                    items[index].output.size_bytes());
    }
    const auto wait_nanoseconds = elapsed_nanoseconds_since(wait_started);
    {
        std::scoped_lock lock(impl_->mutex);
        auto& device_stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        record_synchronization(device_stats,
                               SynchronizationSubsystem::Projection, 1U,
                               wait_nanoseconds);
    }
    return result;
}

ValidationResult CudaBackend::matmul_softcap(
    const CudaWeight& weight, std::span<const float> input,
    float softcap, std::span<float> output) {
    return matmul_impl(weight, input, 1U, 0U, 0U, output, softcap);
}

ValidationResult CudaBackend::matmul_grouped(
    const CudaWeight& weight, std::span<const float> input,
    std::uint32_t groups, std::uint64_t rows_per_group,
    std::span<float> output) {
    return matmul_impl(
        weight, input, 1U, groups, rows_per_group, output, 0.0F);
}

ValidationResult CudaBackend::matmul_grouped_rows(
    const CudaWeight& weight, std::span<const float> input,
    std::uint32_t rows, std::uint32_t groups,
    std::uint64_t rows_per_group, std::span<float> output) {
    return matmul_impl(
        weight, input, rows, groups, rows_per_group, output, 0.0F);
}

ValidationResult CudaBackend::validate_flash_attention_device(int device) const {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "FlashAttention targets an uninitialized CUDA device");
    } else if (!found->second.flash_attention_supported) {
        result.errors.emplace_back(
            "FlashAttention CUDA kernel supports only SM86 and SM120 devices");
    }
    return result;
}

ValidationResult CudaBackend::validate_lightning_index_device(int device) const {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "Lightning Indexer targets an uninitialized CUDA device");
    } else if (!found->second.lightning_index_supported) {
        result.errors.emplace_back(
            "Lightning Indexer CUDA kernel supports only SM86 and SM120 devices");
    }
    return result;
}

bool CudaBackend::dsv4_fp8_tensor_page_supported(int device) const noexcept {
    const auto found = impl_->devices.find(device);
    return found != impl_->devices.end() &&
           found->second.dsv4_fp8_tensor_page_supported;
}

bool CudaBackend::fp8_f32_tensor_page_supported(int device) const noexcept {
    const auto found = impl_->devices.find(device);
    return found != impl_->devices.end() &&
           found->second.fp8_f32_tensor_page_supported;
}

bool CudaBackend::fp8_f32_register_fed_supported(int device) const noexcept {
    const auto found = impl_->devices.find(device);
    return found != impl_->devices.end() &&
           found->second.fp8_f32_register_fed_supported;
}

ValidationResult CudaBackend::validate_dsv4_mhc_device(int device) const {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek device mHC targets an uninitialized CUDA device");
    } else if (!found->second.dsv4_mhc_supported) {
        result.errors.emplace_back(
            "exact DeepSeek device mHC requires an SM86 device");
    }
    return result;
}

ValidationResult CudaBackend::lightning_index(
    int device, const CudaLightningIndexRequest& request,
    std::span<std::uint32_t> output) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "Lightning Indexer targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    if (!state.lightning_index_supported) {
        result.errors.emplace_back(
            "Lightning Indexer CUDA kernel supports only SM86 and SM120 devices");
        return result;
    }
    if (state.moe_in_flight) {
        result.errors.emplace_back(
            "Lightning Indexer cannot overlap an in-flight DeepSeek MoE command");
        return result;
    }
    const auto query_elements = static_cast<std::uint64_t>(request.heads) *
                                request.head_dim;
    if (request.heads == 0U || request.heads > 64U ||
        request.head_dim < 32U || request.head_dim > 1'024U ||
        (request.head_dim & (request.head_dim - 1U)) != 0U ||
        request.head_dim % 32U != 0U || request.top_k == 0U ||
        request.queries.size() != query_elements ||
        request.weights.size() != request.heads ||
        std::any_of(request.queries.begin(), request.queries.end(),
                    [](float value) { return !std::isfinite(value); }) ||
        std::any_of(request.weights.begin(), request.weights.end(),
                    [](float value) { return !std::isfinite(value); })) {
        result.errors.emplace_back(
            "Lightning Indexer query shape or values are unsupported");
        return result;
    }
    const auto row_bytes = static_cast<std::uint64_t>(request.head_dim / 2U +
                                                       request.head_dim / 32U);
    std::uint64_t candidates64 = 0U;
    std::uint64_t host_key_bytes = 0U;
    for (const auto& segment : request.segments) {
        const bool device_source = segment.device_buffer != nullptr;
        const bool host_source = !segment.host_bytes.empty();
        std::uint64_t bytes = 0U;
        if (segment.rows == 0U || device_source == host_source ||
            !checked_bytes(segment.rows, row_bytes, 1U, bytes) ||
            segment.byte_offset > std::numeric_limits<std::uint64_t>::max() -
                                      bytes ||
            candidates64 > std::numeric_limits<std::uint64_t>::max() -
                               segment.rows) {
            result.errors.emplace_back(
                "Lightning Indexer key segment is invalid");
            return result;
        }
        if (device_source) {
            if (!segment.device_buffer->valid() ||
                segment.device_buffer->device() != device ||
                segment.byte_offset + bytes >
                    segment.device_buffer->device_bytes()) {
                result.errors.emplace_back(
                    "Lightning Indexer device key segment is invalid");
                return result;
            }
        } else {
            if (segment.byte_offset + bytes > segment.host_bytes.size() ||
                host_key_bytes > std::numeric_limits<std::uint64_t>::max() -
                                     bytes) {
                result.errors.emplace_back(
                    "Lightning Indexer host key segment is invalid");
                return result;
            }
            host_key_bytes += bytes;
        }
        candidates64 += segment.rows;
    }
    if (candidates64 > 1'048'576U ||
        request.segments.size() > std::numeric_limits<std::uint32_t>::max()) {
        result.errors.emplace_back(
            "Lightning Indexer candidate or segment count is unsupported");
        return result;
    }
    const auto candidates = static_cast<std::uint32_t>(candidates64);
    const auto selected = std::min(request.top_k, candidates);
    if (output.size() != selected) {
        result.errors.emplace_back(
            "Lightning Indexer output extent is incompatible");
        return result;
    }
    if (candidates == 0U) return result;
    if (request.top_k > candidates) {
        result.errors.emplace_back(
            "Lightning Indexer top-k exceeds the candidate count");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for Lightning Indexer");
    }

    std::uint64_t cursor = 0U;
    const auto region = [&](std::uint64_t bytes, std::uint64_t alignment,
                            std::uint64_t& offset) -> bool {
        if (!align_up(cursor, alignment, offset) ||
            bytes > std::numeric_limits<std::uint64_t>::max() - offset) {
            return false;
        }
        cursor = offset + bytes;
        return true;
    };
    std::uint64_t query_bytes = 0U;
    std::uint64_t weight_bytes = 0U;
    std::uint64_t packed_query_bytes = 0U;
    std::uint64_t segment_bytes = 0U;
    std::uint64_t score_bytes = 0U;
    std::uint64_t top_score_bytes = 0U;
    std::uint64_t top_position_bytes = 0U;
    if (!checked_bytes(request.queries.size(), 1U, sizeof(float), query_bytes) ||
        !checked_bytes(request.weights.size(), 1U, sizeof(float), weight_bytes) ||
        !checked_bytes(request.heads, row_bytes, 1U, packed_query_bytes) ||
        !checked_bytes(request.segments.size(), 1U,
                       sizeof(LightningDeviceSegment), segment_bytes) ||
        !checked_bytes(candidates, 1U, sizeof(float), score_bytes) ||
        !checked_bytes(request.top_k, 1U, sizeof(float), top_score_bytes) ||
        !checked_bytes(request.top_k, 1U, sizeof(std::uint32_t),
                       top_position_bytes)) {
        result.errors.emplace_back(
            "Lightning Indexer workspace size overflows");
        return result;
    }
    std::uint64_t query_offset = 0U;
    std::uint64_t weight_offset = 0U;
    std::uint64_t packed_query_offset = 0U;
    std::uint64_t segment_offset = 0U;
    std::uint64_t host_key_offset = 0U;
    std::uint64_t score_offset = 0U;
    std::uint64_t top_score_offset = 0U;
    std::uint64_t top_position_offset = 0U;
    std::uint64_t error_offset = 0U;
    if (!region(query_bytes, alignof(float), query_offset) ||
        !region(weight_bytes, alignof(float), weight_offset) ||
        !region(packed_query_bytes, 1U, packed_query_offset) ||
        !region(segment_bytes, alignof(LightningDeviceSegment), segment_offset) ||
        !region(host_key_bytes, 1U, host_key_offset) ||
        !region(score_bytes, alignof(float), score_offset) ||
        !region(top_score_bytes, alignof(float), top_score_offset) ||
        !region(top_position_bytes, alignof(std::uint32_t),
                top_position_offset) ||
        !region(sizeof(unsigned int), alignof(unsigned int), error_offset) ||
        cursor > request.maximum_workspace_bytes ||
        cursor > std::numeric_limits<std::size_t>::max()) {
        result.errors.emplace_back(
            "Lightning Indexer exceeds its bounded CUDA workspace");
        return result;
    }
    std::uint64_t allocation_calls = 0U;
    std::uint64_t allocation_bytes = 0U;
    if (state.lightning_workspace_bytes < cursor ||
        state.lightning_workspace_bytes > request.maximum_workspace_bytes) {
        if (state.lightning_workspace != nullptr) {
            static_cast<void>(cudaFree(state.lightning_workspace));
            state.lightning_workspace = nullptr;
            state.lightning_workspace_bytes = 0U;
        }
        if (auto status = cudaMalloc(&state.lightning_workspace,
                                     static_cast<std::size_t>(cursor));
            status != cudaSuccess) {
            return cuda_error(status,
                              "allocate bounded Lightning Indexer workspace");
        }
        state.lightning_workspace_bytes = cursor;
        allocation_calls = 1U;
        allocation_bytes = cursor;
    }
    auto* base = state.lightning_workspace;
    auto* device_queries = reinterpret_cast<float*>(base + query_offset);
    auto* device_weights = reinterpret_cast<float*>(base + weight_offset);
    auto* device_packed_queries = reinterpret_cast<unsigned char*>(
        base + packed_query_offset);
    auto* device_segments = reinterpret_cast<LightningDeviceSegment*>(
        base + segment_offset);
    auto* device_host_keys = reinterpret_cast<unsigned char*>(
        base + host_key_offset);
    auto* device_scores = reinterpret_cast<float*>(base + score_offset);
    auto* device_top_scores = reinterpret_cast<float*>(base + top_score_offset);
    auto* device_top_positions = reinterpret_cast<std::uint32_t*>(
        base + top_position_offset);
    auto* device_error = reinterpret_cast<unsigned int*>(base + error_offset);

    std::vector<LightningDeviceSegment> descriptors;
    descriptors.reserve(request.segments.size());
    std::uint64_t host_cursor = 0U;
    std::uint32_t row_begin = 0U;
    for (const auto& segment : request.segments) {
        const auto bytes = static_cast<std::uint64_t>(segment.rows) * row_bytes;
        const unsigned char* keys = nullptr;
        if (segment.device_buffer != nullptr) {
            keys = static_cast<const unsigned char*>(
                       segment.device_buffer->impl_->data) +
                   segment.byte_offset;
        } else {
            keys = device_host_keys + host_cursor;
            host_cursor += bytes;
        }
        descriptors.push_back({keys, row_begin, segment.rows});
        row_begin += segment.rows;
    }

    const auto operation_started = std::chrono::steady_clock::now();
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_start, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "record Lightning Indexer upload start");
        }
    }
    std::uint64_t h2d_transfers = 0U;
    std::uint64_t h2d_bytes = 0U;
    const auto upload = [&](void* destination, const void* source,
                            std::uint64_t bytes) -> bool {
        if (bytes == 0U) return true;
        if (auto status = cudaMemcpyAsync(destination, source,
                                          static_cast<std::size_t>(bytes),
                                          cudaMemcpyHostToDevice, state.stream);
            status != cudaSuccess) {
            result = cuda_error(status, "upload Lightning Indexer input");
            return false;
        }
        ++h2d_transfers;
        h2d_bytes += bytes;
        return true;
    };
    if (!upload(device_queries, request.queries.data(), query_bytes) ||
        !upload(device_weights, request.weights.data(), weight_bytes) ||
        !upload(device_segments, descriptors.data(), segment_bytes)) {
        return result;
    }
    host_cursor = 0U;
    for (const auto& segment : request.segments) {
        if (segment.device_buffer != nullptr) continue;
        const auto bytes = static_cast<std::uint64_t>(segment.rows) * row_bytes;
        if (!upload(device_host_keys + host_cursor,
                    segment.host_bytes.data() + segment.byte_offset, bytes)) {
            return result;
        }
        host_cursor += bytes;
    }
    if (auto status = cudaMemsetAsync(device_error, 0,
                                      sizeof(unsigned int), state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "clear Lightning Indexer error state");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_uploaded,
                                          state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "record Lightning Indexer upload completion");
        }
    }
    lightning_query_fp4_kernel<<<request.heads, 256U,
        static_cast<std::size_t>(request.head_dim) * sizeof(float),
        state.stream>>>(device_packed_queries, device_queries,
                        request.heads, request.head_dim);
    lightning_topk_initialize_kernel<<<1U, 256U, 0U, state.stream>>>(
        device_top_scores, device_top_positions, request.top_k);
    lightning_score_kernel<<<candidates, request.heads, 0U, state.stream>>>(
        device_scores, device_packed_queries, device_weights,
        device_segments, static_cast<std::uint32_t>(request.segments.size()),
        candidates, request.heads, request.head_dim, device_error);
    lightning_topk_merge_kernel<<<1U, 1U, 0U, state.stream>>>(
        device_scores, candidates, device_top_scores, device_top_positions,
        request.top_k);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch Lightning Indexer kernels");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.kernel_finished, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "record Lightning Indexer kernel completion");
        }
    }
    unsigned int host_error = 0U;
    const auto output_bytes = static_cast<std::uint64_t>(output.size_bytes());
    if (auto status = cudaMemcpyAsync(output.data(), device_top_positions,
                                      output.size_bytes(),
                                      cudaMemcpyDeviceToHost, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "download Lightning Indexer positions");
    }
    if (auto status = cudaMemcpyAsync(&host_error, device_error,
                                      sizeof(host_error),
                                      cudaMemcpyDeviceToHost, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "download Lightning Indexer error state");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_downloaded,
                                          state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "record Lightning Indexer download completion");
        }
    }
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaStreamSynchronize(state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "synchronize Lightning Indexer");
    }
    const auto wait_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wait_started).count());
    const auto total_nanoseconds = static_cast<std::uint64_t>(
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
            return cuda_error(status, "measure Lightning Indexer upload");
        }
        if (auto status = cudaEventElapsedTime(
                &kernel_ms, state.activation_uploaded, state.kernel_finished);
            status != cudaSuccess) {
            return cuda_error(status, "measure Lightning Indexer kernels");
        }
        if (auto status = cudaEventElapsedTime(
                &d2h_ms, state.kernel_finished, state.activation_downloaded);
            status != cudaSuccess) {
            return cuda_error(status, "measure Lightning Indexer download");
        }
        h2d_nanoseconds = static_cast<std::uint64_t>(h2d_ms * 1.0e6F);
        kernel_nanoseconds = static_cast<std::uint64_t>(kernel_ms * 1.0e6F);
        d2h_nanoseconds = static_cast<std::uint64_t>(d2h_ms * 1.0e6F);
    }
    {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        stats.activation_h2d_bytes += h2d_bytes;
        stats.activation_d2h_bytes += output_bytes + sizeof(host_error);
        stats.workspace_allocation_calls += allocation_calls;
        stats.workspace_allocation_bytes += allocation_bytes;
        record_synchronization(stats, SynchronizationSubsystem::Attention, 1U,
                               wait_nanoseconds);
        stats.activation_h2d_nanoseconds += h2d_nanoseconds;
        stats.kernel_nanoseconds += kernel_nanoseconds;
        stats.activation_d2h_nanoseconds += d2h_nanoseconds;
        ++stats.lightning_index_calls;
        stats.lightning_index_kernel_launches += 4U;
        stats.lightning_index_candidates += candidates;
        stats.lightning_index_selected += selected;
        stats.lightning_index_h2d_transfers += h2d_transfers;
        stats.lightning_index_d2h_transfers += 2U;
        stats.lightning_index_h2d_bytes += h2d_bytes;
        stats.lightning_index_d2h_bytes += output_bytes + sizeof(host_error);
        stats.lightning_index_useful_selection_bytes +=
            static_cast<std::uint64_t>(selected) * row_bytes;
        stats.lightning_index_h2d_nanoseconds += h2d_nanoseconds;
        stats.lightning_index_kernel_nanoseconds += kernel_nanoseconds;
        stats.lightning_index_d2h_nanoseconds += d2h_nanoseconds;
        stats.lightning_index_nanoseconds += total_nanoseconds;
    }
    if (host_error != 0U) {
        result.errors.emplace_back(
            "Lightning Indexer encountered corrupt FP4 values or incomplete keys");
    }
    return result;
}

ValidationResult CudaBackend::dsv4_index_query_rope_quantize(
    int device, std::span<float> queries, std::span<const float> cosines,
    std::span<const float> sines, std::uint32_t heads,
    std::uint32_t head_dim, std::uint32_t rope_dim, bool quantize) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "CUDA index-query preparation received an unknown device");
        return result;
    }
    auto& state = found->second;
    if (heads == 0U || head_dim == 0U || rope_dim == 0U ||
        rope_dim > head_dim || (rope_dim % 2U) != 0U ||
        head_dim > 1024U ||
        queries.size() != static_cast<std::size_t>(heads) * head_dim ||
        cosines.size() != rope_dim / 2U || sines.size() != rope_dim / 2U) {
        result.errors.emplace_back(
            "CUDA index-query preparation shapes are incompatible");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status,
                          "select CUDA device for index-query preparation");
    }
    const auto query_bytes =
        static_cast<std::uint64_t>(queries.size()) * sizeof(float);
    const auto rope_bytes =
        static_cast<std::uint64_t>(cosines.size()) * sizeof(float);
    const auto required = query_bytes + 2U * rope_bytes + sizeof(unsigned int);
    if (state.dsv4_index_query_workspace_bytes < required) {
        if (state.dsv4_index_query_workspace != nullptr) {
            static_cast<void>(cudaFree(state.dsv4_index_query_workspace));
            state.dsv4_index_query_workspace = nullptr;
            state.dsv4_index_query_workspace_bytes = 0U;
        }
        if (auto status = cudaMalloc(&state.dsv4_index_query_workspace,
                                     static_cast<std::size_t>(required));
            status != cudaSuccess) {
            return cuda_error(status,
                              "allocate index-query preparation workspace");
        }
        state.dsv4_index_query_workspace_bytes = required;
    }
    auto* base = state.dsv4_index_query_workspace;
    auto* device_queries = reinterpret_cast<float*>(base);
    auto* device_cosines = reinterpret_cast<float*>(base + query_bytes);
    auto* device_sines =
        reinterpret_cast<float*>(base + query_bytes + rope_bytes);
    auto* device_error = reinterpret_cast<unsigned int*>(
        base + query_bytes + 2U * rope_bytes);

    const auto copy = [&](void* destination, const void* source,
                          std::uint64_t bytes, cudaMemcpyKind kind,
                          const char* what) {
        const auto status = cudaMemcpyAsync(
            destination, source, static_cast<std::size_t>(bytes), kind,
            state.stream);
        if (status != cudaSuccess) result = cuda_error(status, what);
        return status == cudaSuccess;
    };
    if (!copy(device_queries, queries.data(), query_bytes,
              cudaMemcpyHostToDevice, "upload index queries") ||
        !copy(device_cosines, cosines.data(), rope_bytes,
              cudaMemcpyHostToDevice, "upload index rope cosines") ||
        !copy(device_sines, sines.data(), rope_bytes,
              cudaMemcpyHostToDevice, "upload index rope sines")) {
        return result;
    }
    if (auto status = cudaMemsetAsync(device_error, 0, sizeof(unsigned int),
                                      state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "clear index-query preparation error state");
    }
    dsv4_index_query_rope_quantize_kernel<<<heads, head_dim, 0U,
                                           state.stream>>>(
        device_queries, device_cosines, device_sines, head_dim, rope_dim,
        quantize ? 1U : 0U, device_error);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch index-query preparation");
    }
    unsigned int host_error = 0U;
    if (!copy(queries.data(), device_queries, query_bytes,
              cudaMemcpyDeviceToHost, "download index queries") ||
        !copy(&host_error, device_error, sizeof(host_error),
              cudaMemcpyDeviceToHost, "download index-query error state")) {
        return result;
    }
    if (auto status = cudaStreamSynchronize(state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "synchronize index-query preparation");
    }
    if (host_error != 0U) {
        result.errors.emplace_back(
            "DeepSeek V4 DeepSeek-V4 index query contains a non-finite value");
    }
    return result;
}

ValidationResult CudaBackend::dsv4_index_projections(
    int device, const CudaDsv4IndexProjectionRequest& request,
    std::span<float> queries, std::span<float> weights) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "CUDA index projections received an unknown device");
        return result;
    }
    auto& state = found->second;
    const auto query_elements =
        static_cast<std::uint64_t>(request.heads) * request.head_dim;
    // With both output spans empty the projections stay on the device for an
    // in-chain selection to consume: no download and no synchronize, which is
    // the whole point of running them here rather than through two host
    // round trips.
    const bool device_only = queries.empty() && weights.empty();
    if (request.heads == 0U || request.heads > 64U ||
        request.head_dim < 32U || request.head_dim > 1'024U ||
        request.rope_dim == 0U || request.rope_dim > request.head_dim ||
        (request.rope_dim % 2U) != 0U ||
        (!device_only && queries.size() != query_elements) ||
        (!device_only && weights.size() != request.heads) ||
        request.rope_cosines.size() != request.rope_dim / 2U ||
        request.rope_sines.size() != request.rope_dim / 2U ||
        !std::isfinite(request.weight_scale) ||
        request.query_projection == nullptr ||
        request.weight_projection == nullptr ||
        !request.query_projection->valid() ||
        !request.weight_projection->valid() ||
        request.query_projection->device() != device ||
        request.weight_projection->device() != device) {
        result.errors.emplace_back(
            "CUDA index projection shapes are incompatible");
        return result;
    }
    const auto& query_descriptor = request.query_projection->impl_->descriptor;
    const auto& weight_descriptor =
        request.weight_projection->impl_->descriptor;
    // The two sources are the preparation's own activations, and each one is
    // in the state exactly one encoding expects. A quantized activation fed to
    // a plain BF16 kernel, or a raw one fed to an FP8 kernel, would be silently
    // wrong rather than rejected, so the encodings are required rather than
    // dispatched over.
    if (query_descriptor.encoding != CudaWeightEncoding::Fp8E4m3Block128 ||
        query_descriptor.rows != query_elements ||
        weight_descriptor.encoding != CudaWeightEncoding::Plain ||
        weight_descriptor.dtype != SafetensorsDtype::Bf16 ||
        weight_descriptor.rows != request.heads) {
        result.errors.emplace_back(
            "CUDA index projections require an FP8 query projection and a "
            "BF16 weight projection");
        return result;
    }
    if (state.dsv4_prepared_index_query_source == nullptr ||
        state.dsv4_prepared_index_hidden_source == nullptr) {
        result.errors.emplace_back(
            std::string("CUDA index projections on device ") +
            std::to_string(device) + " have no prepared " +
            (state.dsv4_prepared_index_query_source == nullptr
                 ? "query rank"
                 : "layer input") +
            "; the preparation before them must publish its index source");
        return result;
    }
    if (state.moe_in_flight) {
        result.errors.emplace_back(
            "CUDA index projections cannot overlap an in-flight DeepSeek MoE "
            "command");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for index projections");
    }
    const auto query_bytes = query_elements * sizeof(float);
    const auto weight_bytes =
        static_cast<std::uint64_t>(request.heads) * sizeof(float);
    const auto rope_bytes =
        static_cast<std::uint64_t>(request.rope_cosines.size()) * sizeof(float);
    const auto required =
        query_bytes + weight_bytes + 2U * rope_bytes + sizeof(unsigned int);
    if (state.dsv4_index_query_workspace_bytes < required) {
        if (state.dsv4_index_query_workspace != nullptr) {
            static_cast<void>(cudaFree(state.dsv4_index_query_workspace));
            state.dsv4_index_query_workspace = nullptr;
            state.dsv4_index_query_workspace_bytes = 0U;
        }
        if (auto status = cudaMalloc(&state.dsv4_index_query_workspace,
                                     static_cast<std::size_t>(required));
            status != cudaSuccess) {
            return cuda_error(status, "allocate index projection workspace");
        }
        state.dsv4_index_query_workspace_bytes = required;
    }
    auto* base = state.dsv4_index_query_workspace;
    auto* device_queries = reinterpret_cast<float*>(base);
    auto* device_weights = reinterpret_cast<float*>(base + query_bytes);
    auto* device_cosines =
        reinterpret_cast<float*>(base + query_bytes + weight_bytes);
    auto* device_sines = reinterpret_cast<float*>(
        base + query_bytes + weight_bytes + rope_bytes);
    auto* device_error = reinterpret_cast<unsigned int*>(
        base + query_bytes + weight_bytes + 2U * rope_bytes);

    const auto copy = [&](void* destination, const void* source,
                          std::uint64_t bytes, cudaMemcpyKind kind,
                          const char* what) {
        const auto status = cudaMemcpyAsync(
            destination, source, static_cast<std::size_t>(bytes), kind,
            state.stream);
        if (status != cudaSuccess) result = cuda_error(status, what);
        return status == cudaSuccess;
    };
    if (!copy(device_cosines, request.rope_cosines.data(), rope_bytes,
              cudaMemcpyHostToDevice, "upload index rope cosines") ||
        !copy(device_sines, request.rope_sines.data(), rope_bytes,
              cudaMemcpyHostToDevice, "upload index rope sines")) {
        return result;
    }
    if (auto status = cudaMemsetAsync(device_error, 0, sizeof(unsigned int),
                                      state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "clear index projection error state");
    }
    constexpr unsigned int threads = 256U;
    native_fp8_matmul_kernel<<<
        dim3(static_cast<unsigned int>(query_descriptor.rows), 1U, 1U), threads,
        0U, state.stream>>>(
        device_queries, state.dsv4_prepared_index_query_source,
        static_cast<const unsigned char*>(
            request.query_projection->impl_->weights),
        static_cast<const unsigned char*>(
            request.query_projection->impl_->scales),
        query_descriptor.scale_columns, 1U, query_descriptor.columns,
        query_descriptor.rows, 0U, 0U);
    dsv4_index_projection_round_kernel<<<
        static_cast<unsigned int>((query_elements + threads - 1U) / threads),
        threads, 0U, state.stream>>>(
        device_queries, static_cast<std::uint32_t>(query_elements),
        device_error);
    dsv4_index_query_rope_quantize_kernel<<<request.heads, request.head_dim, 0U,
                                           state.stream>>>(
        device_queries, device_cosines, device_sines, request.head_dim,
        request.rope_dim, 1U, device_error);
    constexpr unsigned int warps_per_block = threads / 32U;
    bf16_matvec_kernel<<<
        static_cast<unsigned int>(
            (weight_descriptor.rows + warps_per_block - 1U) / warps_per_block),
        threads, 0U, state.stream>>>(
        device_weights, state.dsv4_prepared_index_hidden_source,
        static_cast<const __nv_bfloat16*>(
            request.weight_projection->impl_->weights),
        weight_descriptor.columns, weight_descriptor.rows);
    dsv4_index_weight_scale_kernel<<<1U, threads, 0U, state.stream>>>(
        device_weights, request.heads, request.weight_scale, device_error);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch index projection kernels");
    }
    // One preparation publishes one index projection. Retiring the sources here
    // means a second use without a fresh preparation is rejected instead of
    // silently projecting the previous layer's activations.
    state.dsv4_prepared_index_query_source = nullptr;
    state.dsv4_prepared_index_hidden_source = nullptr;
    if (device_only) {
        state.dsv4_index_projection_queries = device_queries;
        state.dsv4_index_projection_weights = device_weights;
        state.dsv4_index_projection_error = device_error;
        state.dsv4_index_projection_heads = request.heads;
        state.dsv4_index_projection_head_dim = request.head_dim;
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        stats.matmul_calls += 2U;
        stats.activation_h2d_bytes += 2U * rope_bytes;
        return result;
    }
    unsigned int host_error = 0U;
    if (!copy(queries.data(), device_queries, query_bytes,
              cudaMemcpyDeviceToHost, "download index queries") ||
        !copy(weights.data(), device_weights, weight_bytes,
              cudaMemcpyDeviceToHost, "download index weights") ||
        !copy(&host_error, device_error, sizeof(host_error),
              cudaMemcpyDeviceToHost, "download index projection state")) {
        return result;
    }
    if (auto status = cudaStreamSynchronize(state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "synchronize index projections");
    }
    if (host_error != 0U) {
        result.errors.emplace_back(
            "DeepSeek index projection produced a non-finite value");
        return result;
    }
    {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        stats.matmul_calls += 2U;
        stats.activation_h2d_bytes += 2U * rope_bytes;
        stats.activation_d2h_bytes +=
            query_bytes + weight_bytes + sizeof(unsigned int);
    }
    return result;
}

ValidationResult CudaBackend::dsv4_physical_lightning_index(
    int device, const CudaDsv4PhysicalIndexRequest& request,
    std::span<std::uint32_t> output,
    CudaDsv4DeviceIndexSelection* device_selection) {
    ValidationResult result;
    if (device_selection != nullptr) *device_selection = {};
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "physical Lightning Indexer targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    if (!state.lightning_index_supported) {
        result.errors.emplace_back(
            "physical Lightning Indexer CUDA kernel supports only SM86 and SM120 devices");
        return result;
    }
    if (state.moe_in_flight) {
        result.errors.emplace_back(
            "physical Lightning Indexer cannot overlap an in-flight DeepSeek MoE command");
        return result;
    }
    const auto query_elements = static_cast<std::uint64_t>(request.heads) *
                                request.head_dim;
    if (request.heads == 0U || request.heads > 64U ||
        request.head_dim < 32U || request.head_dim > 1'024U ||
        request.head_dim % 4U != 0U || request.top_k == 0U ||
        (request.device_projected
             ? (!request.queries.empty() || !request.weights.empty())
             : (request.queries.size() != query_elements ||
                request.weights.size() != request.heads)) ||
        std::any_of(request.queries.begin(), request.queries.end(),
                    [](float value) { return !std::isfinite(value); }) ||
        std::any_of(request.weights.begin(), request.weights.end(),
                    [](float value) { return !std::isfinite(value); })) {
        result.errors.emplace_back(
            "physical Lightning Indexer query shape or values are unsupported");
        return result;
    }
    if (request.device_projected &&
        (state.dsv4_index_projection_queries == nullptr ||
         state.dsv4_index_projection_weights == nullptr ||
         state.dsv4_index_projection_heads != request.heads ||
         state.dsv4_index_projection_head_dim != request.head_dim)) {
        result.errors.emplace_back(
            "physical Lightning Indexer has no matching device projection");
        return result;
    }
    std::uint64_t candidates64 = 0U;
    for (const auto& page : request.pages) {
        std::uint64_t bytes = 0U;
        if (page.buffer == nullptr || !page.buffer->valid() ||
            page.buffer->device() != device || page.rows == 0U ||
            page.block_rows == 0U || page.rows > page.block_rows ||
            !checked_bytes(page.block_rows,
                           static_cast<std::uint64_t>(request.head_dim) +
                               sizeof(float),
                           1U, bytes) ||
            bytes > page.buffer->device_bytes() ||
            page.byte_offset > page.buffer->device_bytes() - bytes ||
            candidates64 > std::numeric_limits<std::uint64_t>::max() -
                               page.rows) {
            result.errors.emplace_back(
                "physical Lightning Indexer page is invalid");
            return result;
        }
        candidates64 += page.rows;
    }
    if (candidates64 > 1'048'576U ||
        request.pages.size() > std::numeric_limits<std::uint32_t>::max()) {
        result.errors.emplace_back(
            "physical Lightning Indexer candidate or page count is unsupported");
        return result;
    }
    const auto candidates = static_cast<std::uint32_t>(candidates64);
    const auto selected = std::min(request.top_k, candidates);
    // A device-selection caller reads the positions on the device and need not
    // provide host storage at all; it may still pass a correctly sized span if
    // it wants both.
    if (output.size() != selected &&
        !(device_selection != nullptr && output.empty())) {
        result.errors.emplace_back(
            "physical Lightning Indexer output extent is incompatible");
        return result;
    }
    if (candidates == 0U) return result;
    if (request.top_k > candidates) {
        result.errors.emplace_back(
            "physical Lightning Indexer top-k exceeds the candidate count");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status,
                          "select CUDA device for physical Lightning Indexer");
    }

    std::uint64_t cursor = 0U;
    const auto region = [&](std::uint64_t bytes, std::uint64_t alignment,
                            std::uint64_t& offset) -> bool {
        if (!align_up(cursor, alignment, offset) ||
            bytes > std::numeric_limits<std::uint64_t>::max() - offset) {
            return false;
        }
        cursor = offset + bytes;
        return true;
    };
    std::uint64_t query_bytes = 0U;
    std::uint64_t weight_bytes = 0U;
    std::uint64_t segment_bytes = 0U;
    std::uint64_t score_bytes = 0U;
    std::uint64_t key_bytes = 0U;
    std::uint64_t active_bytes = 0U;
    std::uint64_t winner_bytes = 0U;
    constexpr std::uint64_t histogram_bytes =
        static_cast<std::uint64_t>(kPhysicalIndexRadixBins) *
        sizeof(std::uint32_t);
    // remaining, winner_count, active_count, next_count, pivot_bin,
    // above_count, error
    constexpr std::uint64_t counter_count = 7U;
    // Sized from the shape, not from the request spans: a device-projected
    // call carries no host spans, and reserving nothing for the query would
    // leave the transposed staging writing over the regions that follow it.
    if (!checked_bytes(query_elements, 1U, sizeof(float), query_bytes) ||
        !checked_bytes(request.heads, 1U, sizeof(float), weight_bytes) ||
        !checked_bytes(request.pages.size(), 1U,
                       sizeof(PhysicalIndexDeviceSegment), segment_bytes) ||
        !checked_bytes(candidates, 1U, sizeof(float), score_bytes) ||
        !checked_bytes(candidates, 1U, sizeof(unsigned long long), key_bytes) ||
        !checked_bytes(candidates, 1U, sizeof(std::uint32_t), active_bytes) ||
        !checked_bytes(request.top_k, 1U, sizeof(std::uint32_t),
                       winner_bytes)) {
        result.errors.emplace_back(
            "physical Lightning Indexer workspace size overflows");
        return result;
    }
    std::uint64_t query_offset = 0U;
    std::uint64_t weight_offset = 0U;
    std::uint64_t segment_offset = 0U;
    std::uint64_t score_offset = 0U;
    std::uint64_t key_offset = 0U;
    std::uint64_t active_offset = 0U;
    std::uint64_t next_offset = 0U;
    std::uint64_t winner_offset = 0U;
    std::uint64_t histogram_offset = 0U;
    std::uint64_t counter_offset = 0U;
    if (!region(query_bytes, alignof(float), query_offset) ||
        !region(weight_bytes, alignof(float), weight_offset) ||
        !region(segment_bytes, alignof(PhysicalIndexDeviceSegment),
                segment_offset) ||
        !region(score_bytes, alignof(float), score_offset) ||
        !region(key_bytes, alignof(unsigned long long), key_offset) ||
        !region(active_bytes, alignof(std::uint32_t), active_offset) ||
        !region(active_bytes, alignof(std::uint32_t), next_offset) ||
        !region(winner_bytes, alignof(std::uint32_t), winner_offset) ||
        // The pivot kernel reads each 64-bin group as 16 uint4, so this region
        // needs vector alignment, not merely uint32 alignment.
        !region(histogram_bytes, alignof(uint4), histogram_offset) ||
        !region(counter_count * sizeof(std::uint32_t), alignof(std::uint32_t),
                counter_offset) ||
        cursor > request.maximum_workspace_bytes ||
        cursor > std::numeric_limits<std::size_t>::max()) {
        result.errors.emplace_back(
            "physical Lightning Indexer exceeds its bounded CUDA workspace");
        return result;
    }
    std::uint64_t allocation_calls = 0U;
    std::uint64_t allocation_bytes = 0U;
    if (state.lightning_workspace_bytes < cursor ||
        state.lightning_workspace_bytes > request.maximum_workspace_bytes) {
        if (state.lightning_workspace != nullptr) {
            static_cast<void>(cudaFree(state.lightning_workspace));
            state.lightning_workspace = nullptr;
            state.lightning_workspace_bytes = 0U;
        }
        if (auto status = cudaMalloc(&state.lightning_workspace,
                                     static_cast<std::size_t>(cursor));
            status != cudaSuccess) {
            return cuda_error(
                status,
                "allocate bounded physical Lightning Indexer workspace");
        }
        state.lightning_workspace_bytes = cursor;
        allocation_calls = 1U;
        allocation_bytes = cursor;
    }
    auto* base = state.lightning_workspace;
    auto* device_queries = reinterpret_cast<float*>(base + query_offset);
    auto* device_weights = reinterpret_cast<float*>(base + weight_offset);
    auto* device_segments = reinterpret_cast<PhysicalIndexDeviceSegment*>(
        base + segment_offset);
    auto* device_scores = reinterpret_cast<float*>(base + score_offset);
    auto* device_keys = reinterpret_cast<unsigned long long*>(
        base + key_offset);
    auto* device_active = reinterpret_cast<std::uint32_t*>(
        base + active_offset);
    auto* device_next = reinterpret_cast<std::uint32_t*>(base + next_offset);
    auto* device_winners = reinterpret_cast<std::uint32_t*>(
        base + winner_offset);
    auto* device_histogram = reinterpret_cast<std::uint32_t*>(
        base + histogram_offset);
    auto* counters = reinterpret_cast<std::uint32_t*>(base + counter_offset);
    auto* device_remaining = counters + 0U;
    auto* device_winner_count = counters + 1U;
    auto* device_active_count = counters + 2U;
    auto* device_next_count = counters + 3U;
    auto* device_pivot = counters + 4U;
    auto* device_above = counters + 5U;
    auto* device_error = reinterpret_cast<unsigned int*>(counters + 6U);

    std::vector<PhysicalIndexDeviceSegment> descriptors;
    try {
        descriptors.reserve(request.pages.size());
    } catch (const std::bad_alloc&) {
        result.errors.emplace_back(
            "cannot allocate physical Lightning Indexer page metadata");
        return result;
    }
    std::uint32_t row_begin = 0U;
    for (const auto& page : request.pages) {
        descriptors.push_back(PhysicalIndexDeviceSegment{
            static_cast<const unsigned char*>(page.buffer->impl_->data) +
                page.byte_offset,
            row_begin, page.rows, page.block_rows});
        row_begin += page.rows;
    }

    const auto operation_started = std::chrono::steady_clock::now();
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_start, state.stream);
            status != cudaSuccess) {
            return cuda_error(
                status, "record physical Lightning Indexer upload start");
        }
    }
    std::uint64_t h2d_transfers = 0U;
    std::uint64_t h2d_bytes = 0U;
    const auto upload = [&](void* destination, const void* source,
                            std::uint64_t bytes) -> bool {
        if (bytes == 0U) return true;
        if (auto status = cudaMemcpyAsync(destination, source,
                                          static_cast<std::size_t>(bytes),
                                          cudaMemcpyHostToDevice, state.stream);
            status != cudaSuccess) {
            result = cuda_error(status,
                                "upload physical Lightning Indexer input");
            return false;
        }
        ++h2d_transfers;
        h2d_bytes += bytes;
        return true;
    };
    // Transposed to column-major on the way in so the score kernel's 64
    // threads read consecutive floats. The cost is one pass over 8,192 values
    // per call; the alternative is a strided read in the innermost loop.
    if (!request.device_projected) {
        std::vector<float> transposed;
        try {
            transposed.resize(request.queries.size());
        } catch (const std::bad_alloc&) {
            result.errors.emplace_back(
                "cannot allocate physical Lightning Indexer query staging");
            return result;
        }
        for (std::uint32_t head = 0U; head < request.heads; ++head) {
            for (std::uint32_t column = 0U; column < request.head_dim;
                 ++column) {
                transposed[static_cast<std::size_t>(column) * request.heads +
                           head] =
                    request.queries[static_cast<std::size_t>(head) *
                                        request.head_dim + column];
            }
        }
        if (!upload(device_queries, transposed.data(), query_bytes) ||
            !upload(device_weights, request.weights.data(), weight_bytes)) {
            return result;
        }
    }
    if (!upload(device_segments, descriptors.data(), segment_bytes)) {
        return result;
    }
    const std::array<std::uint32_t, counter_count> initial_counters{
        request.top_k, 0U, candidates, 0U, 0U, 0U, 0U};
    if (!upload(counters, initial_counters.data(),
                counter_count * sizeof(std::uint32_t))) {
        return result;
    }
    if (request.device_projected) {
        constexpr std::uint32_t transpose_threads = 256U;
        dsv4_index_query_transpose_kernel<<<
            static_cast<unsigned int>(
                (query_elements + transpose_threads - 1U) / transpose_threads),
            transpose_threads, 0U, state.stream>>>(
            device_queries, state.dsv4_index_projection_queries, request.heads,
            request.head_dim);
        if (auto status = cudaMemcpyAsync(
                device_weights, state.dsv4_index_projection_weights,
                static_cast<std::size_t>(weight_bytes),
                cudaMemcpyDeviceToDevice, state.stream);
            status != cudaSuccess) {
            return cuda_error(
                status, "stage device-projected Lightning Indexer weights");
        }
        // The projection's non-finite rejection replaces the host validation
        // above and has no other route back, so it seeds this command's error
        // word rather than being checked on the host.
        if (auto status = cudaMemcpyAsync(
                device_error, state.dsv4_index_projection_error,
                sizeof(unsigned int), cudaMemcpyDeviceToDevice, state.stream);
            status != cudaSuccess) {
            return cuda_error(
                status, "carry the device index projection rejection");
        }
        state.dsv4_index_projection_queries = nullptr;
        state.dsv4_index_projection_weights = nullptr;
        state.dsv4_index_projection_error = nullptr;
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_uploaded,
                                          state.stream);
            status != cudaSuccess) {
            return cuda_error(
                status, "record physical Lightning Indexer upload completion");
        }
    }

    constexpr std::uint32_t kPassThreads = 256U;
    const auto pass_blocks = (candidates + kPassThreads - 1U) / kPassThreads;
    // Slots are threads-per-block divided by heads; each slot scores
    // kPhysicalIndexRowsPerThread rows, so the block covers that many more
    // candidates without changing its thread count or its occupancy.
    const auto score_slots =
        std::max(1U, kPhysicalIndexBlockThreads / request.heads);
    const auto rows_per_block = score_slots * kPhysicalIndexRowsPerThread;
    const auto score_threads = score_slots * request.heads;
    const auto score_blocks =
        (candidates + rows_per_block - 1U) / rows_per_block;
    const auto score_shared = static_cast<std::size_t>(
        (256U + rows_per_block * request.head_dim +
         rows_per_block * request.heads + rows_per_block) * sizeof(float));
    dsv4_physical_index_score_kernel<kPhysicalIndexRowsPerThread>
        <<<score_blocks, score_threads, score_shared, state.stream>>>(
        device_scores, device_keys, device_queries, device_weights,
        device_segments, static_cast<std::uint32_t>(request.pages.size()),
        candidates, request.heads, request.head_dim, rows_per_block,
        device_error);
    // Three 16-bit passes cover the composite key's 48 live bits: the top 16
    // are the bf16 score, the low 32 the position's complement.
    const std::array<std::uint32_t, 3U> shifts{32U, 16U, 0U};
    const std::uint32_t* pass_active = nullptr;
    auto* pass_next = device_active;
    auto* pass_spare = device_next;
    for (const auto shift : shifts) {
        if (auto status = cudaMemsetAsync(device_histogram, 0,
                                          static_cast<std::size_t>(
                                              histogram_bytes),
                                          state.stream);
            status != cudaSuccess) {
            return cuda_error(
                status, "clear physical Lightning Indexer histogram");
        }
        dsv4_physical_index_histogram_kernel<<<pass_blocks, kPassThreads, 0U,
                                              state.stream>>>(
            device_keys, pass_active, device_active_count, shift,
            device_histogram);
        dsv4_physical_index_pivot_kernel<<<1U, kPhysicalIndexPivotThreads, 0U,
                                          state.stream>>>(
            device_histogram, device_remaining, device_pivot, device_above);
        dsv4_physical_index_partition_kernel<<<pass_blocks, kPassThreads, 0U,
                                              state.stream>>>(
            device_keys, pass_active, device_active_count, shift, device_pivot,
            device_winners, device_winner_count, pass_next, device_next_count);
        dsv4_physical_index_advance_kernel<<<1U, 1U, 0U, state.stream>>>(
            device_remaining, device_above, device_active_count,
            device_next_count);
        pass_active = pass_next;
        std::swap(pass_next, pass_spare);
    }
    dsv4_physical_index_finalize_kernel<<<1U, 1U, 0U, state.stream>>>(
        pass_active, device_active_count, device_remaining, device_winners,
        device_winner_count, device_error);
    std::uint32_t padded = 1U;
    while (padded < selected) padded *= 2U;
    const auto sort_shared =
        static_cast<std::size_t>(padded) *
        (sizeof(unsigned long long) + sizeof(std::uint32_t));
    dsv4_physical_index_sort_kernel<<<1U, std::min(padded, 1'024U), sort_shared,
                                     state.stream>>>(
        device_winners, device_keys, selected, padded);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status,
                          "launch physical Lightning Indexer kernels");
    }
    // The selection stays on the device for a caller that consumes it in
    // stream order. With no host output span this returns here, and everything
    // below -- a device-to-host copy and a stream synchronize -- is exactly
    // what a queued chain cannot afford per indexed layer. A caller may still
    // ask for both while the device form is being brought up.
    if (device_selection != nullptr) {
        device_selection->positions = device_winners;
        device_selection->error = device_error;
        device_selection->selected = selected;
    }
    if (device_selection != nullptr && output.empty()) {
        {
            std::scoped_lock lock(impl_->mutex);
            auto& stats = *std::find_if(
                impl_->stats.devices.begin(), impl_->stats.devices.end(),
                [device](const auto& value) { return value.device == device; });
            ++stats.lightning_index_calls;
            stats.lightning_index_kernel_launches += 2U + shifts.size() * 4U;
            stats.lightning_index_candidates += candidates;
            stats.lightning_index_selected += selected;
            stats.lightning_index_h2d_transfers += h2d_transfers;
            stats.lightning_index_h2d_bytes += h2d_bytes;
            stats.workspace_allocation_calls += allocation_calls;
            stats.workspace_allocation_bytes += allocation_bytes;
        }
        return result;
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.kernel_finished, state.stream);
            status != cudaSuccess) {
            return cuda_error(
                status, "record physical Lightning Indexer kernel completion");
        }
    }
    unsigned int host_error = 0U;
    const auto output_bytes = static_cast<std::uint64_t>(output.size_bytes());
    if (auto status = cudaMemcpyAsync(output.data(), device_winners,
                                      output.size_bytes(),
                                      cudaMemcpyDeviceToHost, state.stream);
        status != cudaSuccess) {
        return cuda_error(status,
                          "download physical Lightning Indexer positions");
    }
    if (auto status = cudaMemcpyAsync(&host_error, device_error,
                                      sizeof(host_error),
                                      cudaMemcpyDeviceToHost, state.stream);
        status != cudaSuccess) {
        return cuda_error(status,
                          "download physical Lightning Indexer error state");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_downloaded,
                                          state.stream);
            status != cudaSuccess) {
            return cuda_error(
                status,
                "record physical Lightning Indexer download completion");
        }
    }
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaStreamSynchronize(state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "synchronize physical Lightning Indexer");
    }
    const auto wait_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wait_started).count());
    const auto total_nanoseconds = static_cast<std::uint64_t>(
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
            return cuda_error(status,
                              "measure physical Lightning Indexer upload");
        }
        if (auto status = cudaEventElapsedTime(
                &kernel_ms, state.activation_uploaded, state.kernel_finished);
            status != cudaSuccess) {
            return cuda_error(status,
                              "measure physical Lightning Indexer kernels");
        }
        if (auto status = cudaEventElapsedTime(
                &d2h_ms, state.kernel_finished, state.activation_downloaded);
            status != cudaSuccess) {
            return cuda_error(status,
                              "measure physical Lightning Indexer download");
        }
        h2d_nanoseconds = static_cast<std::uint64_t>(h2d_ms * 1.0e6F);
        kernel_nanoseconds = static_cast<std::uint64_t>(kernel_ms * 1.0e6F);
        d2h_nanoseconds = static_cast<std::uint64_t>(d2h_ms * 1.0e6F);
    }
    const auto row_bytes =
        static_cast<std::uint64_t>(request.head_dim) + sizeof(float);
    {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        stats.activation_h2d_bytes += h2d_bytes;
        stats.activation_d2h_bytes += output_bytes + sizeof(host_error);
        stats.workspace_allocation_calls += allocation_calls;
        stats.workspace_allocation_bytes += allocation_bytes;
        record_synchronization(stats, SynchronizationSubsystem::Attention, 1U,
                               wait_nanoseconds);
        stats.activation_h2d_nanoseconds += h2d_nanoseconds;
        stats.kernel_nanoseconds += kernel_nanoseconds;
        stats.activation_d2h_nanoseconds += d2h_nanoseconds;
        ++stats.lightning_index_calls;
        stats.lightning_index_kernel_launches += 2U + shifts.size() * 4U;
        stats.lightning_index_candidates += candidates;
        stats.lightning_index_selected += selected;
        stats.lightning_index_h2d_transfers += h2d_transfers;
        stats.lightning_index_d2h_transfers += 2U;
        stats.lightning_index_h2d_bytes += h2d_bytes;
        stats.lightning_index_d2h_bytes += output_bytes + sizeof(host_error);
        stats.lightning_index_useful_selection_bytes +=
            static_cast<std::uint64_t>(selected) * row_bytes;
        stats.lightning_index_h2d_nanoseconds += h2d_nanoseconds;
        stats.lightning_index_kernel_nanoseconds += kernel_nanoseconds;
        stats.lightning_index_d2h_nanoseconds += d2h_nanoseconds;
        stats.lightning_index_nanoseconds += total_nanoseconds;
    }
    if (host_error != 0U) {
        result.errors.emplace_back(
            "physical Lightning Indexer encountered a corrupt E4M3 row, a "
            "non-bf16 score, or an incomplete key history");
    }
    return result;
}
