ValidationResult CudaBackend::flash_attention(
    int device, const FlashAttentionRequest& request,
    std::span<float> output) {
    ValidationResult result;
    auto shape = validate_flash_attention_request(request, output);
    if (!shape.ok()) {
        result.errors = std::move(shape.errors);
        return result;
    }
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "FlashAttention targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    if (state.moe_in_flight) {
        result.errors.emplace_back(
            "FlashAttention cannot overlap an in-flight DeepSeek MoE command");
        return result;
    }
    if (request.query_rows > 65'535U || request.query_key_dim > 1'024U ||
        request.value_dim > 1'024U ||
        shape.value.logical_rows > std::numeric_limits<std::uint32_t>::max()) {
        result.errors.emplace_back(
            "FlashAttention CUDA shape exceeds the supported query, row, or head dimension");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for FlashAttention");
    }
    if (!state.flash_attention_supported) {
        result.errors.emplace_back(
            "FlashAttention CUDA kernel supports only SM86 and SM120 devices");
        return result;
    }

    const auto query_bytes = static_cast<std::uint64_t>(request.queries.size_bytes());
    std::uint64_t key_bytes = 0U;
    std::uint64_t value_bytes = 0U;
    if (!checked_bytes(1U, shape.value.packed_key_elements, sizeof(float),
                       key_bytes) ||
        (!shape.value.values_alias_keys &&
         !checked_bytes(1U, shape.value.packed_value_elements, sizeof(float),
                        value_bytes))) {
        result.errors.emplace_back(
            "FlashAttention CUDA packed staging size overflows");
        return result;
    }
    const auto sink_bytes = static_cast<std::uint64_t>(request.head_sinks.size_bytes());
    const auto limit_bytes = static_cast<std::uint64_t>(
        request.causal_key_counts.size_bytes());
    const auto mask_bytes = static_cast<std::uint64_t>(
        request.query_key_mask.size_bytes());
    const auto output_bytes = static_cast<std::uint64_t>(output.size_bytes());
    std::uint64_t score_bytes = 0U;
    if (request.numerics ==
            FlashAttentionNumerics::f64_dot_f32_score_f32_accum &&
        !checked_bytes(
            static_cast<std::uint64_t>(request.query_rows) * request.query_heads,
            shape.value.logical_rows, sizeof(float), score_bytes)) {
        result.errors.emplace_back(
            "FlashAttention CUDA score scratch size overflows");
        return result;
    }
    const auto append_region = [&](std::uint64_t& total,
                                   std::uint64_t bytes) -> bool {
        if (bytes > std::numeric_limits<std::uint64_t>::max() - total) {
            return false;
        }
        total += bytes;
        return true;
    };
    const std::uint64_t query_offset = 0U;
    const std::uint64_t key_offset = query_bytes;
    std::uint64_t upload_bytes = query_bytes;
    if (!append_region(upload_bytes, key_bytes)) {
        result.errors.emplace_back("FlashAttention CUDA upload layout overflows");
        return result;
    }
    const std::uint64_t value_offset = upload_bytes;
    if (!append_region(upload_bytes, value_bytes)) {
        result.errors.emplace_back("FlashAttention CUDA upload layout overflows");
        return result;
    }
    const std::uint64_t sink_offset = upload_bytes;
    if (!append_region(upload_bytes, sink_bytes)) {
        result.errors.emplace_back("FlashAttention CUDA upload layout overflows");
        return result;
    }
    const std::uint64_t limit_offset = upload_bytes;
    if (!append_region(upload_bytes, limit_bytes)) {
        result.errors.emplace_back("FlashAttention CUDA upload layout overflows");
        return result;
    }
    const std::uint64_t mask_offset = upload_bytes;
    if (!append_region(upload_bytes, mask_bytes)) {
        result.errors.emplace_back("FlashAttention CUDA upload layout overflows");
        return result;
    }
    const std::uint64_t output_offset = 0U;
    const std::uint64_t error_offset = output_bytes;
    std::uint64_t download_bytes = output_bytes;
    if (!append_region(download_bytes, sizeof(unsigned int))) {
        result.errors.emplace_back("FlashAttention CUDA download layout overflows");
        return result;
    }

    std::uint64_t allocation_calls = 0U;
    std::uint64_t allocation_bytes = 0U;
    const auto workspace_capacity = [&]() -> std::uint64_t {
        const std::array capacities{
            state.attention_upload_bytes,
            state.attention_download_bytes,
            state.attention_score_bytes};
        std::uint64_t total = 0U;
        for (const auto bytes : capacities) {
            if (bytes > std::numeric_limits<std::uint64_t>::max() - total) {
                return std::numeric_limits<std::uint64_t>::max();
            }
            total += bytes;
        }
        return total;
    };
    const auto ensure_workspace = [&](auto*& pointer, std::uint64_t& capacity,
                                      std::uint64_t required,
                                      const char* operation) -> bool {
        if (required == 0U || required <= capacity) return true;

        // Decode grows the logical KV history one row at a time. Exact-sized
        // cudaFree/cudaMalloc on every token serializes the device and can cost
        // more than the attention kernel. Grow geometrically while keeping the
        // complete persistent workspace inside the request's declared ceiling.
        auto target = required;
        if (required <= (std::uint64_t{1U} << 63U)) {
            target = std::bit_ceil(required);
        }
        const auto current_total = workspace_capacity();
        if (current_total == std::numeric_limits<std::uint64_t>::max() ||
            capacity > current_total) {
            result.errors.emplace_back(
                "FlashAttention CUDA workspace capacity overflows");
            return false;
        }
        const auto retained = current_total - capacity;
        if (target > request.maximum_workspace_bytes -
                         std::min(request.maximum_workspace_bytes, retained)) {
            target = required;
        }
        if (retained > request.maximum_workspace_bytes ||
            target > request.maximum_workspace_bytes - retained) {
            result.errors.emplace_back(
                "FlashAttention reusable CUDA workspace exceeds its bounded contract");
            return false;
        }

        using Pointer = std::remove_reference_t<decltype(pointer)>;
        Pointer replacement = nullptr;
        const bool can_replace_before_free =
            current_total <= request.maximum_workspace_bytes &&
            target <= request.maximum_workspace_bytes - current_total;
        if (!can_replace_before_free && pointer != nullptr) {
            static_cast<void>(cudaFree(pointer));
            pointer = nullptr;
            capacity = 0U;
        }
        if (auto status = cudaMalloc(
                &replacement, static_cast<std::size_t>(target));
            status != cudaSuccess) {
            result = cuda_error(status, operation);
            return false;
        }
        if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
        pointer = replacement;
        capacity = target;
        ++allocation_calls;
        allocation_bytes += target;
        return true;
    };
    if (!ensure_workspace(state.attention_upload, state.attention_upload_bytes,
                          upload_bytes,
                          "allocate FlashAttention upload workspace") ||
        !ensure_workspace(state.attention_download,
                          state.attention_download_bytes, download_bytes,
                          "allocate FlashAttention download workspace") ||
        !ensure_workspace(state.attention_scores, state.attention_score_bytes,
                          score_bytes, "allocate FlashAttention score workspace")) {
        return result;
    }

    const auto ensure_host_workspace = [&](std::byte*& pointer,
                                           std::uint64_t& capacity,
                                           std::uint64_t required,
                                           const char* operation) -> bool {
        if (required <= capacity) return true;
        auto target = required;
        if (required <= (std::uint64_t{1U} << 63U)) {
            target = std::bit_ceil(required);
        }
        void* replacement = nullptr;
        if (auto status = cudaMallocHost(
                &replacement, static_cast<std::size_t>(target));
            status != cudaSuccess) {
            result = cuda_error(status, operation);
            return false;
        }
        if (pointer != nullptr) static_cast<void>(cudaFreeHost(pointer));
        pointer = static_cast<std::byte*>(replacement);
        capacity = target;
        return true;
    };
    if (!ensure_host_workspace(
            state.attention_host_upload, state.attention_host_upload_bytes,
            upload_bytes, "allocate pinned FlashAttention upload staging") ||
        !ensure_host_workspace(
            state.attention_host_download, state.attention_host_download_bytes,
            download_bytes, "allocate pinned FlashAttention download staging")) {
        return result;
    }

    auto* host_queries = reinterpret_cast<float*>(
        state.attention_host_upload + query_offset);
    auto* host_keys = reinterpret_cast<float*>(
        state.attention_host_upload + key_offset);
    auto* host_values = reinterpret_cast<float*>(
        state.attention_host_upload + value_offset);
    auto* host_sinks = reinterpret_cast<float*>(
        state.attention_host_upload + sink_offset);
    auto* host_limits = reinterpret_cast<std::uint32_t*>(
        state.attention_host_upload + limit_offset);
    auto* host_mask = reinterpret_cast<std::uint8_t*>(
        state.attention_host_upload + mask_offset);
    std::copy(request.queries.begin(), request.queries.end(), host_queries);
    std::copy(request.head_sinks.begin(), request.head_sinks.end(), host_sinks);
    std::copy(request.causal_key_counts.begin(),
              request.causal_key_counts.end(), host_limits);
    std::copy(request.query_key_mask.begin(),
              request.query_key_mask.end(), host_mask);
    const auto key_row_elements = static_cast<std::size_t>(
        request.key_value_heads) * request.query_key_dim;
    const auto value_row_elements = static_cast<std::size_t>(
        request.key_value_heads) * request.value_dim;
    std::size_t packed_key_offset = 0U;
    std::size_t packed_value_offset = 0U;
    for (const auto& segment : request.segments) {
        const auto source_rows = segment.keys.size() / key_row_elements;
        const auto logical_rows = segment.row_indices.empty()
            ? source_rows : segment.row_indices.size();
        for (std::size_t row = 0U; row < logical_rows; ++row) {
            const auto source_row = segment.row_indices.empty()
                ? row : segment.row_indices[row];
            const auto key = segment.keys.subspan(
                source_row * key_row_elements, key_row_elements);
            std::copy(key.begin(), key.end(), host_keys + packed_key_offset);
            packed_key_offset += key.size();
            if (!shape.value.values_alias_keys) {
                const auto& values = segment.values.empty()
                    ? segment.keys : segment.values;
                const auto value = values.subspan(
                    source_row * value_row_elements, value_row_elements);
                std::copy(value.begin(), value.end(),
                          host_values + packed_value_offset);
                packed_value_offset += value.size();
            }
        }
    }

    const auto operation_started = std::chrono::steady_clock::now();
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_start, state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "record FlashAttention upload start");
        }
    }
    if (auto status = cudaMemcpyAsync(
            state.attention_upload, state.attention_host_upload,
            static_cast<std::size_t>(upload_bytes), cudaMemcpyHostToDevice,
            state.stream); status != cudaSuccess) {
        return cuda_error(status, "upload pinned FlashAttention staging");
    }
    auto* device_queries = reinterpret_cast<float*>(
        state.attention_upload + query_offset);
    auto* device_keys = reinterpret_cast<float*>(
        state.attention_upload + key_offset);
    auto* device_values_storage = reinterpret_cast<float*>(
        state.attention_upload + value_offset);
    auto* device_sinks = reinterpret_cast<float*>(
        state.attention_upload + sink_offset);
    auto* device_limits = reinterpret_cast<std::uint32_t*>(
        state.attention_upload + limit_offset);
    auto* device_mask = reinterpret_cast<std::uint8_t*>(
        state.attention_upload + mask_offset);
    auto* device_output = reinterpret_cast<float*>(
        state.attention_download + output_offset);
    auto* device_error = reinterpret_cast<unsigned int*>(
        state.attention_download + error_offset);
    if (auto status = cudaMemsetAsync(device_error, 0,
                                      sizeof(*device_error), state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "clear FlashAttention status");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_uploaded, state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "record FlashAttention upload completion");
        }
    }
    const dim3 grid(request.query_heads, request.query_rows, 1U);
    constexpr unsigned int threads = 256U;
    const auto* device_values = shape.value.values_alias_keys
        ? device_keys : device_values_storage;
    if (request.numerics ==
        FlashAttentionNumerics::f32_dot_f32_softmax_f32_accum) {
        flash_attention_reference_all_f32_kernel<<<grid, threads, 0U, state.stream>>>(
            device_output, device_queries, device_keys,
            device_values,
            request.head_sinks.empty() ? nullptr : device_sinks,
            request.causal_key_counts.empty()
                ? nullptr : device_limits,
            request.query_key_mask.empty() ? nullptr : device_mask,
            request.query_rows, request.query_heads, request.key_value_heads,
            request.query_key_dim, request.value_dim,
            static_cast<std::uint32_t>(shape.value.logical_rows), request.scale,
            device_error);
    } else if (request.numerics ==
        FlashAttentionNumerics::f64_dot_f32_score_f32_accum) {
        // One double per key row lets the block evaluate the softmax
        // exponentials in parallel and hand thread 0 a plain sequential add.
        // Beyond the shared-memory budget the kernel keeps the single-thread
        // fold, so a long context stays correct rather than failing to launch.
        constexpr std::uint64_t exponential_shared_ceiling = 32U * 1024U;
        const auto exponential_capacity =
            shape.value.logical_rows * sizeof(double) <=
                    exponential_shared_ceiling
                ? static_cast<std::uint32_t>(shape.value.logical_rows)
                : 0U;
        const auto exponential_bytes =
            static_cast<std::size_t>(exponential_capacity) * sizeof(double);
        flash_attention_reference_f32_kernel<<<
            grid, threads, exponential_bytes, state.stream>>>(
            device_output, device_queries, device_keys,
            device_values, state.attention_scores,
            request.head_sinks.empty() ? nullptr : device_sinks,
            request.causal_key_counts.empty()
                ? nullptr : device_limits,
            request.query_key_mask.empty() ? nullptr : device_mask,
            request.query_rows, request.query_heads, request.key_value_heads,
            request.query_key_dim, request.value_dim,
            static_cast<std::uint32_t>(shape.value.logical_rows), request.scale,
            device_error, exponential_capacity);
    } else {
        flash_attention_forward_kernel<<<grid, threads, 0U, state.stream>>>(
            device_output, device_queries, device_keys,
            device_values,
            request.head_sinks.empty() ? nullptr : device_sinks,
            request.causal_key_counts.empty()
                ? nullptr : device_limits,
            request.query_rows, request.query_heads, request.key_value_heads,
            request.query_key_dim, request.value_dim,
            static_cast<std::uint32_t>(shape.value.logical_rows), request.scale,
            device_error);
    }
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch FlashAttention forward kernel");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.kernel_finished, state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "record FlashAttention kernel completion");
        }
    }
    if (auto status = cudaMemcpyAsync(
            state.attention_host_download, state.attention_download,
            static_cast<std::size_t>(download_bytes), cudaMemcpyDeviceToHost,
            state.stream); status != cudaSuccess) {
        return cuda_error(status, "download pinned FlashAttention staging");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_downloaded, state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "record FlashAttention download completion");
        }
    }
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaStreamSynchronize(state.stream); status != cudaSuccess) {
        return cuda_error(status, "synchronize FlashAttention forward");
    }
    const auto wait_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wait_started).count());
    const auto operation_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - operation_started).count());
    unsigned int numerical_error = 0U;
    std::memcpy(&numerical_error,
                state.attention_host_download + error_offset,
                sizeof(numerical_error));
    std::uint64_t h2d_nanoseconds = 0U;
    std::uint64_t kernel_nanoseconds = 0U;
    std::uint64_t d2h_nanoseconds = 0U;
    if (impl_->detailed_timing) {
        float h2d_milliseconds = 0.0F;
        float kernel_milliseconds = 0.0F;
        float d2h_milliseconds = 0.0F;
        if (auto status = cudaEventElapsedTime(
                &h2d_milliseconds, state.activation_start,
                state.activation_uploaded); status != cudaSuccess) {
            return cuda_error(status, "measure FlashAttention upload");
        }
        if (auto status = cudaEventElapsedTime(
                &kernel_milliseconds, state.activation_uploaded,
                state.kernel_finished); status != cudaSuccess) {
            return cuda_error(status, "measure FlashAttention kernel");
        }
        if (auto status = cudaEventElapsedTime(
                &d2h_milliseconds, state.kernel_finished,
                state.activation_downloaded); status != cudaSuccess) {
            return cuda_error(status, "measure FlashAttention download");
        }
        h2d_nanoseconds = static_cast<std::uint64_t>(std::llround(
            static_cast<double>(h2d_milliseconds) * 1.0e6));
        kernel_nanoseconds = static_cast<std::uint64_t>(std::llround(
            static_cast<double>(kernel_milliseconds) * 1.0e6));
        d2h_nanoseconds = static_cast<std::uint64_t>(std::llround(
            static_cast<double>(d2h_milliseconds) * 1.0e6));
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
        stats.flash_attention_useful_staging_bytes += key_bytes + value_bytes;
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
                ? "FlashAttention CUDA score is non-finite"
                : numerical_error == 2U
                    ? "FlashAttention CUDA softmax denominator is invalid"
                    : "FlashAttention CUDA output is non-finite");
        return result;
    }
    const auto* host_output = reinterpret_cast<const float*>(
        state.attention_host_download + output_offset);
    std::copy_n(host_output, output.size(), output.begin());
    return result;
}

ValidationResult CudaBackend::dsv4_paged_attention(
    int device, const CudaDsv4PagedAttentionRequest& request,
    std::span<float> output) {
    ValidationResult result;
    const auto call_started = std::chrono::steady_clock::now();
    constexpr std::uint64_t row_output_elements =
        static_cast<std::uint64_t>(kDsv4PagedHeads) * kDsv4PagedHeadDim;
    const auto rows = request.rows;
    const auto candidates = request.candidate_width == 0U && rows == 1U
        ? static_cast<std::uint32_t>(request.candidates.size())
        : request.candidate_width;
    const auto total_candidates = static_cast<std::uint64_t>(rows) *
                                  candidates;
    const auto output_elements = static_cast<std::uint64_t>(rows) *
                                 row_output_elements;
    if (rows == 0U || request.candidates.size() != total_candidates ||
        request.queries.size() != output_elements ||
        request.head_sinks.size() != kDsv4PagedHeads ||
        output.size() != output_elements || request.pages.empty() ||
        request.pages.size() > std::numeric_limits<std::uint32_t>::max() ||
        candidates == 0U || candidates > 640U ||
        candidates % kDsv4PagedCandidateBlock != 0U ||
        !std::isfinite(request.scale) || request.scale <= 0.0F ||
        request.maximum_workspace_bytes == 0U ||
        std::any_of(request.queries.begin(), request.queries.end(),
                    [](float value) {
                        return !std::isfinite(value) ||
                               bf16_round_f32(value) != value;
                    }) ||
        std::any_of(request.head_sinks.begin(), request.head_sinks.end(),
                    [](float value) { return !std::isfinite(value); })) {
        result.errors.emplace_back(
            "DeepSeek paged attention request shape, BF16 query, scale, or sink is invalid");
        return result;
    }
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek paged attention targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    if (state.moe_in_flight) {
        result.errors.emplace_back(
            "DeepSeek paged attention cannot overlap an in-flight MoE command");
        return result;
    }
    if (!state.dsv4_paged_attention_supported) {
        result.errors.emplace_back(
            "exact DeepSeek paged attention requires an SM86 device");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status,
                          "select CUDA device for DeepSeek paged attention");
    }

    std::uint64_t page_bytes = 0U;
    std::uint64_t flat_rows64 = 0U;
    std::uint32_t maximum_page_rows = 0U;
    for (const auto& page : request.pages) {
        if (page.buffer == nullptr || !page.buffer->valid() ||
            page.buffer->device() != device ||
            (page.rows != 2U && page.rows != 64U && page.rows != 256U) ||
            page.buffer->device_bytes() !=
                static_cast<std::uint64_t>(page.rows) * 584U ||
            page.buffer->device_bytes() >
                std::numeric_limits<std::uint64_t>::max() - page_bytes) {
            result.errors.emplace_back(
                "DeepSeek paged attention physical page is invalid");
            return result;
        }
        page_bytes += page.buffer->device_bytes();
        flat_rows64 += page.rows;
        maximum_page_rows = std::max(maximum_page_rows, page.rows);
    }
    if (flat_rows64 == 0U ||
        flat_rows64 > std::numeric_limits<std::uint32_t>::max()) {
        result.errors.emplace_back(
            "DeepSeek paged attention flat page extent overflows");
        return result;
    }
    const auto flat_rows = static_cast<std::uint32_t>(flat_rows64);
    for (const auto& candidate : request.candidates) {
        if (candidate.valid &&
            (candidate.page >= request.pages.size() ||
             candidate.row >= request.pages[candidate.page].rows)) {
            result.errors.emplace_back(
                "DeepSeek paged attention candidate is outside its physical page");
            return result;
        }
    }

    std::uint64_t cursor = 0U;
    const auto region = [&](std::uint64_t bytes,
                            std::uint64_t alignment,
                            std::uint64_t& offset) -> bool {
        if (!align_up(cursor, alignment, offset) ||
            bytes > std::numeric_limits<std::uint64_t>::max() - offset) {
            return false;
        }
        cursor = offset + bytes;
        return true;
    };
    std::uint64_t page_offset = 0U;
    std::uint64_t candidate_offset = 0U;
    std::uint64_t query_offset = 0U;
    std::uint64_t sink_offset = 0U;
    std::uint64_t page_descriptor_bytes = 0U;
    std::uint64_t candidate_bytes = 0U;
    std::uint64_t query_bytes = 0U;
    std::uint64_t sink_bytes = 0U;
    if (!checked_bytes(request.pages.size(), 1U,
                       sizeof(Dsv4DevicePhysicalPage),
                       page_descriptor_bytes) ||
        !checked_bytes(total_candidates, 1U,
                       sizeof(Dsv4DeviceAttentionCandidate),
                       candidate_bytes) ||
        !checked_bytes(output_elements, 1U, sizeof(std::uint16_t),
                       query_bytes) ||
        !checked_bytes(kDsv4PagedHeads, 1U, sizeof(float), sink_bytes) ||
        !region(page_descriptor_bytes, 16U, page_offset) ||
        !region(candidate_bytes, 16U, candidate_offset) ||
        !region(query_bytes, 16U, query_offset) ||
        !region(sink_bytes, 16U, sink_offset)) {
        result.errors.emplace_back(
            "DeepSeek paged attention upload layout overflows");
        return result;
    }
    const auto upload_bytes = cursor;

    std::uint64_t kv_offset = 0U;
    std::uint64_t score_offset = 0U;
    std::uint64_t maximum_offset = 0U;
    std::uint64_t denominator_offset = 0U;
    std::uint64_t value_offset = 0U;
    std::uint64_t output_offset = 0U;
    std::uint64_t failure_offset = 0U;
    std::uint64_t kv_bytes = 0U;
    std::uint64_t score_bytes = 0U;
    std::uint64_t maximum_bytes = 0U;
    std::uint64_t value_bytes = 0U;
    std::uint64_t output_bytes = 0U;
    if (!checked_bytes(flat_rows, kDsv4PagedHeadDim,
                       sizeof(std::uint16_t), kv_bytes) ||
        !checked_bytes(rows, static_cast<std::uint64_t>(kDsv4PagedHeads) *
                                 flat_rows,
                       sizeof(std::uint16_t), score_bytes) ||
        !checked_bytes(rows, kDsv4PagedHeads, sizeof(float), maximum_bytes) ||
        !checked_bytes(output_elements, 1U, sizeof(float), value_bytes) ||
        !checked_bytes(output_elements, 1U, sizeof(std::uint16_t),
                       output_bytes) ||
        !region(kv_bytes, 16U, kv_offset) ||
        !region(score_bytes, 16U, score_offset) ||
        !region(maximum_bytes, 16U, maximum_offset) ||
        !region(maximum_bytes, 16U, denominator_offset) ||
        !region(value_bytes, 16U, value_offset) ||
        !region(output_bytes, 16U, output_offset) ||
        !region(sizeof(unsigned int), 16U, failure_offset)) {
        result.errors.emplace_back(
            "DeepSeek paged attention workspace layout overflows");
        return result;
    }
    const auto workspace_bytes = cursor;
    if (workspace_bytes > request.maximum_workspace_bytes) {
        result.errors.emplace_back(
            "DeepSeek paged attention workspace exceeds its bounded contract");
        return result;
    }
    const auto download_bytes = output_bytes + sizeof(unsigned int);

    std::uint64_t allocation_calls = 0U;
    std::uint64_t allocation_bytes = 0U;
    if (workspace_bytes > state.dsv4_attention_workspace_bytes) {
        auto target = workspace_bytes <= (std::uint64_t{1U} << 63U)
            ? std::bit_ceil(workspace_bytes) : workspace_bytes;
        if (target > request.maximum_workspace_bytes) target = workspace_bytes;
        std::byte* replacement = nullptr;
        if (auto status = cudaMalloc(
                &replacement, static_cast<std::size_t>(target));
            status != cudaSuccess) {
            return cuda_error(status,
                              "allocate DeepSeek paged attention workspace");
        }
        if (state.dsv4_attention_workspace != nullptr) {
            static_cast<void>(cudaFree(state.dsv4_attention_workspace));
        }
        state.dsv4_attention_workspace = replacement;
        state.dsv4_attention_workspace_bytes = target;
        ++allocation_calls;
        allocation_bytes += target;
    }
    const auto ensure_host = [&](std::byte*& pointer,
                                 std::uint64_t& capacity,
                                 std::uint64_t required,
                                 const char* operation) -> bool {
        if (required <= capacity) return true;
        auto target = required <= (std::uint64_t{1U} << 63U)
            ? std::bit_ceil(required) : required;
        void* replacement = nullptr;
        if (auto status = cudaMallocHost(
                &replacement, static_cast<std::size_t>(target));
            status != cudaSuccess) {
            result = cuda_error(status, operation);
            return false;
        }
        if (pointer != nullptr) static_cast<void>(cudaFreeHost(pointer));
        pointer = static_cast<std::byte*>(replacement);
        capacity = target;
        return true;
    };
    if (!ensure_host(state.dsv4_attention_host_upload,
                     state.dsv4_attention_host_upload_bytes, upload_bytes,
                     "allocate pinned DeepSeek paged attention upload") ||
        !ensure_host(state.dsv4_attention_host_download,
                     state.dsv4_attention_host_download_bytes,
                     download_bytes,
                     "allocate pinned DeepSeek paged attention download")) {
        return result;
    }

    auto* host_pages = reinterpret_cast<Dsv4DevicePhysicalPage*>(
        state.dsv4_attention_host_upload + page_offset);
    std::uint32_t flat_begin = 0U;
    for (std::size_t index = 0U; index < request.pages.size(); ++index) {
        const auto& page = request.pages[index];
        host_pages[index] = {
            static_cast<const std::uint8_t*>(page.buffer->impl_->data),
            page.rows, flat_begin};
        flat_begin += page.rows;
    }
    auto* host_candidates =
        reinterpret_cast<Dsv4DeviceAttentionCandidate*>(
            state.dsv4_attention_host_upload + candidate_offset);
    for (std::size_t index = 0U; index < request.candidates.size(); ++index) {
        const auto& candidate = request.candidates[index];
        host_candidates[index] = {
            candidate.page, candidate.row, candidate.valid ? 1U : 0U};
    }
    auto* host_query = reinterpret_cast<std::uint16_t*>(
        state.dsv4_attention_host_upload + query_offset);
    for (std::size_t index = 0U; index < request.queries.size(); ++index) {
        host_query[index] = bf16_encode(request.queries[index]);
    }
    std::memcpy(state.dsv4_attention_host_upload + sink_offset,
                request.head_sinks.data(),
                static_cast<std::size_t>(sink_bytes));

    auto* workspace = state.dsv4_attention_workspace;
    auto* device_pages = reinterpret_cast<Dsv4DevicePhysicalPage*>(
        workspace + page_offset);
    auto* device_candidates =
        reinterpret_cast<Dsv4DeviceAttentionCandidate*>(
            workspace + candidate_offset);
    auto* device_query = reinterpret_cast<__nv_bfloat16*>(
        workspace + query_offset);
    auto* device_sink = reinterpret_cast<float*>(workspace + sink_offset);
    auto* device_kv = reinterpret_cast<__nv_bfloat16*>(workspace + kv_offset);
    auto* device_scores = reinterpret_cast<__nv_bfloat16*>(
        workspace + score_offset);
    auto* device_maximums = reinterpret_cast<float*>(
        workspace + maximum_offset);
    auto* device_denominators = reinterpret_cast<float*>(
        workspace + denominator_offset);
    auto* device_values = reinterpret_cast<float*>(workspace + value_offset);
    auto* device_output = reinterpret_cast<__nv_bfloat16*>(
        workspace + output_offset);
    auto* device_failure = reinterpret_cast<unsigned int*>(
        workspace + failure_offset);

    const auto operation_started = std::chrono::steady_clock::now();
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_start, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "record DeepSeek paged attention upload start");
        }
    }
    if (auto status = cudaMemcpyAsync(
            workspace, state.dsv4_attention_host_upload,
            static_cast<std::size_t>(upload_bytes), cudaMemcpyHostToDevice,
            state.stream); status != cudaSuccess) {
        return cuda_error(status, "upload DeepSeek paged attention metadata");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                state.activation_uploaded, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "record DeepSeek paged attention upload completion");
        }
    }
    if (auto status = cudaMemsetAsync(
            device_failure, 0, sizeof(*device_failure), state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "clear DeepSeek paged attention status");
    }
    constexpr std::uint32_t threads = 256U;
    const auto page_elements = static_cast<std::uint64_t>(maximum_page_rows) *
                               kDsv4PagedHeadDim;
    const dim3 kv_grid(
        static_cast<unsigned int>((page_elements + threads - 1U) / threads),
        static_cast<unsigned int>(request.pages.size()));
    dsv4_materialize_physical_pages<<<kv_grid, threads, 0U, state.stream>>>(
        device_pages, static_cast<std::uint32_t>(request.pages.size()),
        device_kv, device_failure);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status,
                          "launch DeepSeek physical-page materialization");
    }
    dsv4_sparse_scores_kernel<<<
        dim3{rows, kDsv4PagedHeads / kDsv4SparseScoreHeads},
        kDsv4SparseScoreHeads * 32U, 0U, state.stream>>>(
        device_scores, device_query, device_kv, device_pages,
        device_candidates, candidates, 0U);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch DeepSeek sparse attention scores");
    }
    const auto score_elements = static_cast<std::uint64_t>(
        rows) * kDsv4PagedHeads * candidates;
    const auto score_blocks = static_cast<std::uint32_t>(
        (score_elements + threads - 1U) / threads);
    dsv4_scale_scores<<<score_blocks, threads, 0U, state.stream>>>(
        device_scores, score_elements, request.scale);
    const auto boundaries = candidates / kDsv4PagedCandidateBlock;
    dsv4_finish_maximums<<<dim3{kDsv4PagedHeads, rows},
                           kDsv4PagedCandidateBlock, 0U, state.stream>>>(
        device_scores, device_pages, device_candidates, device_sink,
        device_maximums, candidates, candidates, boundaries);
    dsv4_finish_denominators<<<dim3{kDsv4PagedHeads, rows},
                              kDsv4PagedCandidateBlock, 0U, state.stream>>>(
        device_scores, device_pages, device_candidates, device_sink,
        device_maximums, device_denominators, candidates, candidates,
        boundaries);
    const dim3 value_grid(kDsv4PagedHeads,
                          kDsv4PagedHeadDim /
                              kDsv4PagedDimensionsPerBlock,
                          rows);
    dsv4_finish_values<<<value_grid, kDsv4PagedDimensionsPerBlock,
                         0U, state.stream>>>(
        device_scores, device_pages, device_candidates, device_maximums,
        device_kv, device_denominators, device_output,
        static_cast<std::uint64_t>(kDsv4PagedHeads) * kDsv4PagedHeadDim, 0U,
        candidates, candidates, boundaries);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status,
                          "launch DeepSeek paged attention finish kernels");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                state.kernel_finished, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "record DeepSeek paged attention kernel completion");
        }
    }
    if (auto status = cudaMemcpyAsync(
            state.dsv4_attention_host_download, device_output,
            static_cast<std::size_t>(download_bytes),
            cudaMemcpyDeviceToHost, state.stream); status != cudaSuccess) {
        return cuda_error(status, "download DeepSeek paged attention output");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                state.activation_downloaded, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "record DeepSeek paged attention download completion");
        }
    }
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaStreamSynchronize(state.stream);
        status != cudaSuccess) {
        return cuda_error(status,
                          "synchronize DeepSeek paged attention");
    }
    const auto wait_nanoseconds = elapsed_nanoseconds_since(wait_started);
    const auto operation_nanoseconds = elapsed_nanoseconds_since(
        operation_started);
    unsigned int numerical_error = 0U;
    std::memcpy(&numerical_error,
                state.dsv4_attention_host_download + output_bytes,
                sizeof(numerical_error));
    if (numerical_error != 0U) {
        result.errors.emplace_back(
            "DeepSeek paged attention encountered corrupt physical page data");
        return result;
    }
    const auto* host_output = reinterpret_cast<const std::uint16_t*>(
        state.dsv4_attention_host_download);
    for (std::size_t index = 0U; index < output.size(); ++index) {
        output[index] = std::bit_cast<float>(
            static_cast<std::uint32_t>(host_output[index]) << 16U);
    }

    std::uint64_t h2d_nanoseconds = 0U;
    std::uint64_t kernel_nanoseconds = 0U;
    std::uint64_t d2h_nanoseconds = 0U;
    if (impl_->detailed_timing) {
        float h2d_milliseconds = 0.0F;
        float kernel_milliseconds = 0.0F;
        float d2h_milliseconds = 0.0F;
        if (auto status = cudaEventElapsedTime(
                &h2d_milliseconds, state.activation_start,
                state.activation_uploaded); status != cudaSuccess) {
            return cuda_error(status,
                              "measure DeepSeek paged attention upload");
        }
        if (auto status = cudaEventElapsedTime(
                &kernel_milliseconds, state.activation_uploaded,
                state.kernel_finished); status != cudaSuccess) {
            return cuda_error(status,
                              "measure DeepSeek paged attention kernels");
        }
        if (auto status = cudaEventElapsedTime(
                &d2h_milliseconds, state.kernel_finished,
                state.activation_downloaded); status != cudaSuccess) {
            return cuda_error(status,
                              "measure DeepSeek paged attention download");
        }
        h2d_nanoseconds = static_cast<std::uint64_t>(std::llround(
            static_cast<double>(h2d_milliseconds) * 1.0e6));
        kernel_nanoseconds = static_cast<std::uint64_t>(std::llround(
            static_cast<double>(kernel_milliseconds) * 1.0e6));
        d2h_nanoseconds = static_cast<std::uint64_t>(std::llround(
            static_cast<double>(d2h_milliseconds) * 1.0e6));
    }
    {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        ++stats.dsv4_paged_attention_calls;
        stats.dsv4_paged_attention_kernel_launches += 7U;
        stats.dsv4_paged_attention_h2d_bytes += upload_bytes;
        stats.dsv4_paged_attention_d2h_bytes += download_bytes;
        stats.dsv4_paged_attention_page_bytes += page_bytes;
        stats.dsv4_paged_attention_h2d_nanoseconds += h2d_nanoseconds;
        stats.dsv4_paged_attention_kernel_nanoseconds += kernel_nanoseconds;
        stats.dsv4_paged_attention_d2h_nanoseconds += d2h_nanoseconds;
        stats.dsv4_paged_attention_nanoseconds += operation_nanoseconds;
        const auto call_nanoseconds = elapsed_nanoseconds_since(call_started);
        stats.dsv4_paged_attention_host_remainder_nanoseconds +=
            call_nanoseconds > wait_nanoseconds
                ? call_nanoseconds - wait_nanoseconds : 0U;
        stats.dsv4_paged_attention_stream_sync_nanoseconds += wait_nanoseconds;
        stats.activation_h2d_bytes += upload_bytes;
        stats.activation_d2h_bytes += download_bytes;
        stats.activation_h2d_nanoseconds += h2d_nanoseconds;
        stats.kernel_nanoseconds += kernel_nanoseconds;
        stats.activation_d2h_nanoseconds += d2h_nanoseconds;
        stats.workspace_allocation_calls += allocation_calls;
        stats.workspace_allocation_bytes += allocation_bytes;
        record_synchronization(stats, SynchronizationSubsystem::Attention,
                               1U, wait_nanoseconds);
    }
    return result;
}

