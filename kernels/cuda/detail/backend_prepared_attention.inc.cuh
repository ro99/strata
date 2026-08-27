ValidationResult CudaBackend::dsv4_prepare_attention(
    int device, const CudaDsv4AttentionPrepareRequest& request,
    std::span<float> query_rank, std::span<float> key_value,
    std::span<float> compressor_values,
    std::span<float> compressor_scores,
    std::span<float> index_compressor_values,
    std::span<float> index_compressor_scores) {
    ValidationResult result;
    constexpr std::uint64_t hidden = 4096U;
    constexpr std::uint64_t query_rank_elements = 1024U;
    constexpr std::uint64_t query_elements = 64U * 512U;
    constexpr std::uint64_t key_value_elements = 512U;
    constexpr std::uint64_t rope_pairs = 32U;
    const auto source_found = impl_->devices.find(device);
    const auto mhc_found = impl_->devices.find(request.mhc_device);
    const auto* query_a = request.query_a;
    const auto* query_b = request.query_b;
    const auto* key_value_weight = request.key_value;
    const bool prepare_compressor = !compressor_values.empty();
    const bool prepare_index_compressor = !index_compressor_values.empty();
    // The raw BF16 layer input, widened. Every compressor projection reads it,
    // and so does an in-chain index weight projection on a rank that owns no
    // compressor.
    const bool expand_input = prepare_compressor || prepare_index_compressor ||
                              request.publish_index_source;
    const bool transition_mhc = request.mhc_transition != nullptr;
    const bool host_deferred = request.host_callback != nullptr;
    const bool device_only = request.device_only;
    const bool ready_page_patch = !request.ready_page_patches.empty();
    // A host-only preparation returns its projections and publishes no
    // prepared device query, so nothing downstream may consume it.
    const bool host_only = request.host_only;
    // Both deferred and device-only preparations belong to a queued layer that
    // submits its upload and returns without synchronizing, so the next layer
    // may rewrite this pinned staging before the queued H2D has read it.  Their
    // staging must therefore be fixed per command, exactly as the rank-local
    // attention command's staging already is.  Only the fully synchronous
    // host-visible path may share one buffer.
    const bool fixed_command_staging = host_deferred || device_only;
    const bool device_mhc_input = request.mhc_device != device &&
        !transition_mhc && request.cross_device_input.empty();
    if (source_found == impl_->devices.end() ||
        mhc_found == impl_->devices.end() || query_a == nullptr ||
        query_b == nullptr || key_value_weight == nullptr ||
        !query_a->valid() || !query_b->valid() ||
        !key_value_weight->valid() || query_a->device() != device ||
        query_b->device() != device || key_value_weight->device() != device ||
        (device_only
             ? (!query_rank.empty() || !key_value.empty() ||
                prepare_compressor || prepare_index_compressor ||
                host_deferred)
             : (query_rank.size() != query_rank_elements ||
                key_value.size() != key_value_elements)) ||
        request.query_norm.size() != query_rank_elements ||
        request.key_value_norm.size() != key_value_elements ||
        request.rope_cosines.size() != rope_pairs ||
        request.rope_sines.size() != rope_pairs ||
        request.maximum_workspace_bytes == 0U ||
        compressor_values.size() != compressor_scores.size() ||
        index_compressor_values.size() != index_compressor_scores.size() ||
        ((request.host_callback == nullptr) !=
         (request.host_callback_context == nullptr)) ||
        ((request.page_patch_ready_event != nullptr) != ready_page_patch) ||
        (device_only
             ? (ready_page_patch != !request.page_writes.empty())
             : (host_deferred ? request.page_writes.empty()
                              : !request.page_writes.empty())) ||
        (prepare_compressor
             ? (request.compressor_value == nullptr ||
                request.compressor_gate == nullptr)
             : (request.compressor_value != nullptr ||
                request.compressor_gate != nullptr)) ||
        (prepare_index_compressor
             ? (request.index_compressor_value == nullptr ||
                request.index_compressor_gate == nullptr)
             : (request.index_compressor_value != nullptr ||
                request.index_compressor_gate != nullptr)) ||
        (request.mhc_device == device
             ? !request.cross_device_input.empty()
             : (transition_mhc
                    ? !request.cross_device_input.empty()
                    : (!device_mhc_input &&
                       request.cross_device_input.size() != hidden)))) {
        result.errors.emplace_back(
            "DeepSeek attention preparation request shape is invalid");
        return result;
    }
    std::uint64_t page_patch_bytes = 0U;
    for (const auto& write : request.page_writes) {
        const auto write_bytes = static_cast<std::uint64_t>(write.data_bytes) +
                                 write.scale_bytes;
        if (write.buffer == nullptr || !write.buffer->valid() ||
            write.buffer->device() != device || write.data_bytes == 0U ||
            write.scale_bytes == 0U ||
            write.data_offset > write.buffer->device_bytes() ||
            write.data_bytes >
                write.buffer->device_bytes() - write.data_offset ||
            write.scale_offset > write.buffer->device_bytes() ||
            write.scale_bytes >
                write.buffer->device_bytes() - write.scale_offset ||
            write_bytes > std::numeric_limits<std::uint64_t>::max() -
                              page_patch_bytes) {
            result.errors.emplace_back(
                "DeepSeek attention preparation page patch is invalid");
            return result;
        }
        page_patch_bytes += write_bytes;
    }
    if (ready_page_patch &&
        request.ready_page_patches.size() != page_patch_bytes) {
        result.errors.emplace_back(
            "DeepSeek ready page-patch staging has the wrong extent");
        return result;
    }
    const auto finite = [](float value) { return std::isfinite(value); };
    const auto bf16_finite = [](float value) {
        return std::isfinite(value) && bf16_round_f32(value) == value;
    };
    if (!std::all_of(request.query_norm.begin(), request.query_norm.end(),
                     finite) ||
        !std::all_of(request.key_value_norm.begin(),
                     request.key_value_norm.end(), finite) ||
        !std::all_of(request.rope_cosines.begin(),
                     request.rope_cosines.end(), finite) ||
        !std::all_of(request.rope_sines.begin(), request.rope_sines.end(),
                     finite) ||
        !std::all_of(request.cross_device_input.begin(),
                     request.cross_device_input.end(), bf16_finite)) {
        result.errors.emplace_back(
            "DeepSeek attention preparation input is non-finite or not BF16");
        return result;
    }
    const auto accepted_weight = [device, device_only, host_deferred](const CudaWeight* weight,
                                                        std::uint64_t rows,
                                                        std::uint64_t columns) {
        const auto& descriptor = weight->impl_->descriptor;
        const bool native =
            descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128 &&
            descriptor.dtype == SafetensorsDtype::F8E4M3 &&
            descriptor.group_size == 128U;
        // Expanded BF16 weights are valid for the deferred production host
        // node too; synchronous host-visible preparation remains native-only.
        const bool expanded = (device_only || host_deferred) &&
            descriptor.encoding == CudaWeightEncoding::Plain &&
            descriptor.dtype == SafetensorsDtype::Bf16;
        return weight->device() == device && (native || expanded) &&
               descriptor.rows == rows && descriptor.columns == columns;
    };
    const auto accepted_compressor_weight =
        [device](const CudaWeight* weight, std::uint64_t rows,
                 std::uint64_t columns) {
            const auto& descriptor = weight->impl_->descriptor;
            return weight->device() == device &&
                   descriptor.encoding == CudaWeightEncoding::Plain &&
                   descriptor.dtype == SafetensorsDtype::Bf16 &&
                   descriptor.rows == rows && descriptor.columns == columns;
        };
    if (!accepted_weight(query_a, query_rank_elements, hidden) ||
        !accepted_weight(query_b, query_elements, query_rank_elements) ||
        !accepted_weight(key_value_weight, key_value_elements, hidden) ||
        (prepare_compressor &&
         (!accepted_compressor_weight(request.compressor_value,
                                      compressor_values.size(), hidden) ||
          !accepted_compressor_weight(request.compressor_gate,
                                      compressor_scores.size(), hidden))) ||
        (prepare_index_compressor &&
         (!accepted_compressor_weight(request.index_compressor_value,
                                      index_compressor_values.size(), hidden) ||
          !accepted_compressor_weight(request.index_compressor_gate,
                                      index_compressor_scores.size(), hidden)))) {
        result.errors.emplace_back(
            "DeepSeek attention preparation weights violate their encoding contract");
        return result;
    }
    if (transition_mhc &&
        (!request.mhc_transition->valid() ||
         request.mhc_transition->device() != request.mhc_device)) {
        result.errors.emplace_back(
            "DeepSeek attention preparation mHC transition is invalid");
        return result;
    }
    auto& state = source_found->second;
    auto& mhc_state = mhc_found->second;
    if (state.moe_in_flight || mhc_state.moe_in_flight ||
        !state.dsv4_paged_attention_supported ||
        !mhc_state.dsv4_mhc_supported ||
        mhc_state.dsv4_mhc_workspace == nullptr ||
        mhc_state.dsv4_mhc_stage != 1U ||
        mhc_state.dsv4_mhc_branch_ready != transition_mhc ||
        state.dsv4_attention_prepared || (host_only && device_only) ||
        (fixed_command_staging &&
         state.dsv4_attention_prepare_host_command_count >=
             kDsv4FixedCommandCount)) {
        result.errors.emplace_back(
            "DeepSeek attention preparation command order is invalid");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status,
                          "select CUDA device for attention preparation");
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
    std::uint64_t cross_input_offset{}, query_norm_offset{};
    std::uint64_t key_value_norm_offset{}, cosine_offset{}, sine_offset{};
    const auto cross_input_bytes = request.mhc_device == device
        ? 0U : hidden * sizeof(std::uint16_t);
    constexpr auto query_norm_bytes = query_rank_elements * sizeof(float);
    constexpr auto key_value_norm_bytes = key_value_elements * sizeof(float);
    constexpr auto rope_bytes = rope_pairs * sizeof(float);
    if (!region(cross_input_bytes, 16U, cross_input_offset) ||
        !region(query_norm_bytes, 16U, query_norm_offset) ||
        !region(key_value_norm_bytes, 16U, key_value_norm_offset) ||
        !region(rope_bytes, 16U, cosine_offset) ||
        !region(rope_bytes, 16U, sine_offset)) {
        result.errors.emplace_back(
            "DeepSeek attention preparation upload layout overflows");
        return result;
    }
    const auto upload_bytes = cursor;
    if (fixed_command_staging &&
        upload_bytes >
            kDsv4DeferredAttentionPrepareUploadSlotBytes) {
        result.errors.emplace_back(
            "deferred attention preparation upload exceeds its fixed slot");
        return result;
    }
    std::uint64_t input_quant_offset{}, compressor_input_offset{};
    std::uint64_t query_rank_raw_offset{};
    std::uint64_t query_rank_bf16_offset{}, query_rank_quant_offset{};
    std::uint64_t query_raw_offset{}, prepared_query_offset{};
    std::uint64_t key_value_raw_offset{}, key_value_bf16_offset{};
    std::uint64_t compressor_value_offset{}, compressor_score_offset{};
    std::uint64_t index_compressor_value_offset{};
    std::uint64_t index_compressor_score_offset{};
    std::uint64_t failure_offset{};
    if (!region(hidden * sizeof(float), 16U, input_quant_offset) ||
        !region(expand_input ? hidden * sizeof(float) : 0U,
                16U, compressor_input_offset) ||
        !region(query_rank_elements * sizeof(float), 16U,
                query_rank_raw_offset) ||
        !region(query_rank_elements * sizeof(std::uint16_t), 16U,
                query_rank_bf16_offset) ||
        !region(query_rank_elements * sizeof(float), 16U,
                query_rank_quant_offset) ||
        !region(query_elements * sizeof(float), 16U, query_raw_offset) ||
        !region(query_elements * sizeof(std::uint16_t), 16U,
                prepared_query_offset) ||
        !region(key_value_elements * sizeof(float), 16U,
                key_value_raw_offset) ||
        !region(key_value_elements * sizeof(std::uint16_t), 16U,
                key_value_bf16_offset) ||
        !region(compressor_values.size_bytes(), 16U,
                compressor_value_offset) ||
        !region(compressor_scores.size_bytes(), 16U,
                compressor_score_offset) ||
        !region(index_compressor_values.size_bytes(), 16U,
                index_compressor_value_offset) ||
        !region(index_compressor_scores.size_bytes(), 16U,
                index_compressor_score_offset) ||
        !region(sizeof(unsigned int), 16U, failure_offset)) {
        result.errors.emplace_back(
            "DeepSeek attention preparation workspace layout overflows");
        return result;
    }
    const auto workspace_bytes = cursor;
    if (workspace_bytes > request.maximum_workspace_bytes) {
        result.errors.emplace_back(
            "DeepSeek attention preparation exceeds its workspace contract");
        return result;
    }
    constexpr auto query_rank_download_bytes =
        query_rank_elements * sizeof(std::uint16_t);
    constexpr auto key_value_download_bytes =
        key_value_elements * sizeof(std::uint16_t);
    const auto compressor_value_download_offset =
        query_rank_download_bytes + key_value_download_bytes;
    const auto compressor_score_download_offset =
        compressor_value_download_offset + compressor_values.size_bytes();
    const auto index_compressor_value_download_offset =
        compressor_score_download_offset + compressor_scores.size_bytes();
    const auto index_compressor_score_download_offset =
        index_compressor_value_download_offset +
        index_compressor_values.size_bytes();
    const auto failure_download_offset =
        index_compressor_score_download_offset +
        index_compressor_scores.size_bytes();
    const auto download_bytes = failure_download_offset +
                                sizeof(unsigned int);
    if (page_patch_bytes > std::numeric_limits<std::uint64_t>::max() -
                               download_bytes ||
        (host_deferred && download_bytes + page_patch_bytes >
                              request.maximum_workspace_bytes)) {
        result.errors.emplace_back(
            "DeepSeek attention preparation host staging overflows");
        return result;
    }
    const auto host_download_bytes = download_bytes + page_patch_bytes;
    std::uint64_t allocation_calls = 0U;
    std::uint64_t allocation_bytes = 0U;
    if (workspace_bytes > state.dsv4_attention_prepare_workspace_bytes ||
        (host_deferred && state.dsv4_attention_prepare_workspace_bytes <
                              request.maximum_workspace_bytes)) {
        auto target_bytes = host_deferred
            ? request.maximum_workspace_bytes : std::bit_ceil(workspace_bytes);
        if (target_bytes > request.maximum_workspace_bytes) {
            target_bytes = workspace_bytes;
        }
        std::byte* replacement = nullptr;
        if (auto status = cudaMalloc(
                &replacement, static_cast<std::size_t>(target_bytes));
            status != cudaSuccess) {
            return cuda_error(status,
                              "allocate attention preparation workspace");
        }
        if (state.dsv4_attention_prepare_workspace != nullptr) {
            static_cast<void>(
                cudaFree(state.dsv4_attention_prepare_workspace));
        }
        state.dsv4_attention_prepare_workspace = replacement;
        state.dsv4_attention_prepare_workspace_bytes = target_bytes;
        ++allocation_calls;
        allocation_bytes += target_bytes;
    }
    const auto ensure_host = [&](std::byte*& pointer,
                                 std::uint64_t& capacity,
                                 std::uint64_t required,
                                 const char* operation) -> bool {
        if (required <= capacity) return true;
        const auto target_bytes = std::bit_ceil(required);
        void* replacement = nullptr;
        if (auto status = cudaMallocHost(
                &replacement, static_cast<std::size_t>(target_bytes));
            status != cudaSuccess) {
            result = cuda_error(status, operation);
            return false;
        }
        if (pointer != nullptr) static_cast<void>(cudaFreeHost(pointer));
        pointer = static_cast<std::byte*>(replacement);
        capacity = target_bytes;
        return true;
    };
    if (fixed_command_staging &&
        state.dsv4_attention_prepare_fixed_host_upload == nullptr) {
        void* upload = nullptr;
        if (auto status = cudaMallocHost(
                &upload,
                static_cast<std::size_t>(
                    kDsv4FixedCommandCount *
                    kDsv4DeferredAttentionPrepareUploadSlotBytes));
            status != cudaSuccess) {
            return cuda_error(
                status,
                "allocate fixed attention preparation upload slots");
        }
        state.dsv4_attention_prepare_fixed_host_upload =
            static_cast<std::byte*>(upload);
    }
    if (!fixed_command_staging &&
        !ensure_host(state.dsv4_attention_prepare_host_upload,
                     state.dsv4_attention_prepare_host_upload_bytes,
                     upload_bytes,
                     "allocate pinned attention preparation upload")) {
        return result;
    }
    if (!device_only &&
        !ensure_host(state.dsv4_attention_prepare_host_download,
                     state.dsv4_attention_prepare_host_download_bytes,
                     host_deferred ? request.maximum_workspace_bytes
                                   : host_download_bytes,
                     "allocate pinned attention preparation download")) {
        return result;
    }
    auto* host_upload = fixed_command_staging
        ? state.dsv4_attention_prepare_fixed_host_upload +
              static_cast<std::uint64_t>(
                  state.dsv4_attention_prepare_host_command_count) *
                  kDsv4DeferredAttentionPrepareUploadSlotBytes
        : state.dsv4_attention_prepare_host_upload;
    if (cross_input_bytes != 0U && !transition_mhc && !device_mhc_input) {
        auto* encoded = reinterpret_cast<std::uint16_t*>(
            host_upload + cross_input_offset);
        for (std::size_t index = 0U;
             index < request.cross_device_input.size(); ++index) {
            encoded[index] = bf16_encode(request.cross_device_input[index]);
        }
    }
    std::memcpy(host_upload + query_norm_offset, request.query_norm.data(),
                query_norm_bytes);
    std::memcpy(host_upload + key_value_norm_offset,
                request.key_value_norm.data(), key_value_norm_bytes);
    std::memcpy(host_upload + cosine_offset, request.rope_cosines.data(),
                rope_bytes);
    std::memcpy(host_upload + sine_offset, request.rope_sines.data(),
                rope_bytes);

    auto* workspace = state.dsv4_attention_prepare_workspace;
    const auto* device_input = request.mhc_device == device
        ? mhc_state.dsv4_mhc_workspace->layer_input
        : reinterpret_cast<const __nv_bfloat16*>(
              workspace + cross_input_offset);
    auto* device_query_norm = reinterpret_cast<float*>(
        workspace + query_norm_offset);
    auto* device_key_value_norm = reinterpret_cast<float*>(
        workspace + key_value_norm_offset);
    auto* device_cosines = reinterpret_cast<float*>(workspace + cosine_offset);
    auto* device_sines = reinterpret_cast<float*>(workspace + sine_offset);
    auto* input_quant = reinterpret_cast<float*>(workspace + input_quant_offset);
    auto* compressor_input = reinterpret_cast<float*>(
        workspace + compressor_input_offset);
    auto* query_rank_raw = reinterpret_cast<float*>(
        workspace + query_rank_raw_offset);
    auto* query_rank_bf16 = reinterpret_cast<__nv_bfloat16*>(
        workspace + query_rank_bf16_offset);
    auto* query_rank_quant = reinterpret_cast<float*>(
        workspace + query_rank_quant_offset);
    auto* query_raw = reinterpret_cast<float*>(workspace + query_raw_offset);
    auto* prepared_query = reinterpret_cast<__nv_bfloat16*>(
        workspace + prepared_query_offset);
    auto* key_value_raw = reinterpret_cast<float*>(
        workspace + key_value_raw_offset);
    auto* key_value_bf16 = reinterpret_cast<__nv_bfloat16*>(
        workspace + key_value_bf16_offset);
    auto* compressor_value_raw = reinterpret_cast<float*>(
        workspace + compressor_value_offset);
    auto* compressor_score_raw = reinterpret_cast<float*>(
        workspace + compressor_score_offset);
    auto* index_compressor_value_raw = reinterpret_cast<float*>(
        workspace + index_compressor_value_offset);
    auto* index_compressor_score_raw = reinterpret_cast<float*>(
        workspace + index_compressor_score_offset);
    auto* failure = reinterpret_cast<unsigned int*>(workspace + failure_offset);
    const auto operation_started = std::chrono::steady_clock::now();
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_start, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "record attention preparation upload start");
        }
    }
    if (device_mhc_input) {
        constexpr auto layer_bytes = hidden * sizeof(std::uint16_t);
        if (auto status = cudaSetDevice(request.mhc_device);
            status != cudaSuccess) {
            return cuda_error(
                status, "select mHC device for initial attention input");
        }
        if (auto status = cudaMemcpyAsync(
                mhc_state.dsv4_mhc_host_staging,
                mhc_state.dsv4_mhc_workspace->layer_input, layer_bytes,
                cudaMemcpyDeviceToHost, mhc_state.stream);
            status != cudaSuccess) {
            static_cast<void>(cudaSetDevice(device));
            return cuda_error(
                status, "stage initial cross-device mHC layer input");
        }
        if (auto status = cudaEventRecord(
                mhc_state.dsv4_cross_device_ready, mhc_state.stream);
            status != cudaSuccess) {
            static_cast<void>(cudaSetDevice(device));
            return cuda_error(
                status, "publish initial cross-device mHC layer input");
        }
        if (auto status = cudaSetDevice(device); status != cudaSuccess) {
            return cuda_error(
                status, "restore initial attention preparation device");
        }
        if (auto status = cudaStreamWaitEvent(
                state.stream, mhc_state.dsv4_cross_device_ready);
            status != cudaSuccess) {
            return cuda_error(
                status, "wait for initial cross-device mHC layer input");
        }
        if (auto status = cudaMemcpyAsync(
                workspace + cross_input_offset,
                mhc_state.dsv4_mhc_host_staging, layer_bytes,
                cudaMemcpyHostToDevice, state.stream);
            status != cudaSuccess) {
            return cuda_error(
                status, "forward initial cross-device mHC layer input");
        }
    } else if (cross_input_bytes != 0U && !transition_mhc) {
        if (auto status = cudaMemcpyAsync(
                workspace + cross_input_offset,
                host_upload + cross_input_offset,
                static_cast<std::size_t>(cross_input_bytes),
                cudaMemcpyHostToDevice, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "upload cross-device attention input");
        }
    }
    const auto metadata_offset = cross_input_bytes == 0U
        ? 0U : query_norm_offset;
    if (auto status = cudaMemcpyAsync(
            workspace + metadata_offset, host_upload + metadata_offset,
            static_cast<std::size_t>(upload_bytes - metadata_offset),
            cudaMemcpyHostToDevice, state.stream); status != cudaSuccess) {
        return cuda_error(status, "upload attention preparation metadata");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                state.activation_uploaded, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "record attention preparation upload completion");
        }
    }
    if (transition_mhc) {
        const bool cross_transition = request.mhc_device != device;
        if (cross_transition) {
            if (auto status = cudaSetDevice(request.mhc_device);
                status != cudaSuccess) {
                return cuda_error(
                    status, "select mHC device for attention preparation");
            }
        }
        mhc_state.dsv4_mhc_branch_ready = false;
        const auto current = mhc_state.dsv4_mhc_residual_index;
        const auto next = current ^ 1U;
        const auto* projection = static_cast<const float*>(
            request.mhc_transition->impl_->projection.impl_->weights);
        const auto* auxiliary = static_cast<const std::byte*>(
            request.mhc_transition->impl_->auxiliary.impl_->data);
        const auto* scale = reinterpret_cast<const float*>(auxiliary);
        const auto* base = reinterpret_cast<const float*>(
            auxiliary + 3U * sizeof(float));
        const auto* norm = reinterpret_cast<const __nv_bfloat16*>(
            auxiliary + kDsv4MhcAuxNormOffset);
        const auto transition_stream = cross_transition
            ? mhc_state.stream : state.stream;
        if (cross_transition && impl_->detailed_timing) {
            if (auto status = cudaEventRecord(
                    mhc_state.activation_start, transition_stream);
                status != cudaSuccess) {
                static_cast<void>(cudaSetDevice(device));
                return cuda_error(
                    status, "record cross-device mHC transition start");
            }
        }
        dsv4_mhc_fused_post_projection<<<
            dim3{kDsv4MhcMixes / kDsv4MhcProjectionTile,
                 kDsv4MhcSplits},
            kDsv4MhcProjectionThreads, 0U, transition_stream>>>(
            mhc_state.dsv4_mhc_workspace->combination,
            mhc_state.dsv4_mhc_workspace->residual[current],
            mhc_state.dsv4_mhc_workspace->post,
            mhc_state.dsv4_mhc_workspace->branch, projection,
            mhc_state.dsv4_mhc_workspace->partial_projection,
            mhc_state.dsv4_mhc_workspace->partial_square_sum,
            mhc_state.dsv4_mhc_workspace->residual[next]);
        dsv4_mhc_mix<<<1U, 32U, 0U, transition_stream>>>(
            mhc_state.dsv4_mhc_workspace->partial_projection,
            mhc_state.dsv4_mhc_workspace->partial_square_sum,
            scale, base, kDsv4MhcSplits,
            mhc_state.dsv4_mhc_workspace->pre,
            mhc_state.dsv4_mhc_workspace->post,
            mhc_state.dsv4_mhc_workspace->combination);
        dsv4_mhc_weighted_norm<<<1U, kDsv4MhcWeightedNormThreads, 0U,
                                 transition_stream>>>(
            mhc_state.dsv4_mhc_workspace->residual[next],
            mhc_state.dsv4_mhc_workspace->pre, norm,
            mhc_state.dsv4_mhc_workspace->weighted,
            mhc_state.dsv4_mhc_workspace->layer_input);
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            mhc_state.dsv4_mhc_stage = 0U;
            if (cross_transition) static_cast<void>(cudaSetDevice(device));
            return cuda_error(status,
                              "launch combined attention mHC transition");
        }
        mhc_state.dsv4_mhc_residual_index = next;
        if (cross_transition) {
            constexpr auto layer_bytes =
                hidden * sizeof(std::uint16_t);
            if (impl_->detailed_timing) {
                if (auto status = cudaEventRecord(
                        mhc_state.kernel_finished, transition_stream);
                    status != cudaSuccess) {
                    mhc_state.dsv4_mhc_stage = 0U;
                    static_cast<void>(cudaSetDevice(device));
                    return cuda_error(
                        status, "record cross-device mHC kernels");
                }
            }
            if (auto status = cudaMemcpyAsync(
                    mhc_state.dsv4_mhc_host_staging,
                    mhc_state.dsv4_mhc_workspace->layer_input,
                    layer_bytes, cudaMemcpyDeviceToHost,
                    transition_stream);
                status != cudaSuccess) {
                mhc_state.dsv4_mhc_stage = 0U;
                static_cast<void>(cudaSetDevice(device));
                return cuda_error(
                    status, "stage cross-device mHC layer input");
            }
            if (impl_->detailed_timing) {
                if (auto status = cudaEventRecord(
                        mhc_state.activation_downloaded,
                        transition_stream);
                    status != cudaSuccess) {
                    mhc_state.dsv4_mhc_stage = 0U;
                    static_cast<void>(cudaSetDevice(device));
                    return cuda_error(
                        status, "record cross-device mHC download");
                }
            }
            if (auto status = cudaEventRecord(
                    mhc_state.dsv4_cross_device_ready,
                    transition_stream);
                status != cudaSuccess) {
                mhc_state.dsv4_mhc_stage = 0U;
                static_cast<void>(cudaSetDevice(device));
                return cuda_error(
                    status, "publish cross-device mHC layer input");
            }
            if (auto status = cudaSetDevice(device); status != cudaSuccess) {
                mhc_state.dsv4_mhc_stage = 0U;
                return cuda_error(
                    status, "restore attention preparation device");
            }
            if (auto status = cudaStreamWaitEvent(
                    state.stream, mhc_state.dsv4_cross_device_ready);
                status != cudaSuccess) {
                mhc_state.dsv4_mhc_stage = 0U;
                return cuda_error(
                    status, "wait for cross-device mHC layer input");
            }
            if (auto status = cudaMemcpyAsync(
                    workspace + cross_input_offset,
                    mhc_state.dsv4_mhc_host_staging, layer_bytes,
                    cudaMemcpyHostToDevice, state.stream);
                status != cudaSuccess) {
                mhc_state.dsv4_mhc_stage = 0U;
                return cuda_error(
                    status, "forward cross-device mHC layer input");
            }
        } else if (impl_->detailed_timing) {
            if (auto status = cudaEventRecord(
                    state.mhc_transition_finished, state.stream);
                status != cudaSuccess) {
                mhc_state.dsv4_mhc_stage = 0U;
                return cuda_error(
                    status,
                    "record combined mHC transition completion");
            }
        }
    }
    if (auto status = cudaMemsetAsync(
            failure, 0, sizeof(*failure), state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "clear attention preparation status");
    }
    quantize_bf16_activation_e4m3_kernel<<<32U, 128U, 0U, state.stream>>>(
        input_quant, device_input, hidden);
    if (expand_input) {
        expand_bf16_activation_kernel<<<32U, 128U, 0U, state.stream>>>(
            compressor_input, device_input, hidden);
    }
    if (auto status = cudaEventRecord(
            state.dsv4_attention_input_ready, state.stream);
        status != cudaSuccess) {
        if (transition_mhc) mhc_state.dsv4_mhc_stage = 0U;
        return cuda_error(status,
                          "publish DeepSeek attention input activation");
    }
    for (auto stream : state.dsv4_attention_aux_streams) {
        if (auto status = cudaStreamWaitEvent(
                stream, state.dsv4_attention_input_ready);
            status != cudaSuccess) {
            if (transition_mhc) mhc_state.dsv4_mhc_stage = 0U;
            return cuda_error(status,
                              "wait for DeepSeek attention input activation");
        }
    }
    constexpr std::uint32_t threads = 256U;
    // The rank-local expanded-BF16 reduction emulates the accepted Stage-4
    // two-half association with 128 physical threads (four warps). Its
    // shared reduction stores low/high halves in eight slots; launching it
    // with 256 threads would index past that contract and corrupt the sum.
    constexpr std::uint32_t rank_threads = 128U;
    const auto launch_projection = [&](float* output, const float* input,
                                       const CudaWeight* weight,
                                       cudaStream_t stream) {
        const auto& descriptor = weight->impl_->descriptor;
        if (descriptor.encoding == CudaWeightEncoding::Plain) {
            dsv4_rank_bf16_matmul<<<descriptor.rows, rank_threads, 0U, stream>>>(
                output, input,
                static_cast<const __nv_bfloat16*>(weight->impl_->weights),
                1U, descriptor.columns, descriptor.rows, 0U, 0U);
        } else {
            native_fp8_matmul_kernel<<<descriptor.rows, threads, 0U,
                                       stream>>>(
                output, input,
                static_cast<const unsigned char*>(weight->impl_->weights),
                static_cast<const unsigned char*>(weight->impl_->scales),
                descriptor.scale_columns, 1U, descriptor.columns,
                descriptor.rows, 0U, 0U);
        }
    };
    const auto launch_compressor = [&](float* output,
                                       const CudaWeight* weight,
                                       cudaStream_t stream) {
        const auto& descriptor = weight->impl_->descriptor;
        constexpr unsigned int warps_per_block = threads / 32U;
        const auto blocks = static_cast<unsigned int>(
            (descriptor.rows + warps_per_block - 1U) / warps_per_block);
        bf16_matvec_kernel<<<blocks, threads, 0U, stream>>>(
            output, compressor_input,
            static_cast<const __nv_bfloat16*>(weight->impl_->weights),
            descriptor.columns, descriptor.rows);
    };
    launch_projection(query_rank_raw, input_quant, query_a, state.stream);
    dsv4_query_rank_norm<<<1U, kDsv4QueryRankNormThreads, 0U, state.stream>>>(
        query_rank_raw, device_query_norm, query_rank_bf16, failure);
    quantize_bf16_activation_e4m3_kernel<<<8U, 128U, 0U, state.stream>>>(
        query_rank_quant, query_rank_bf16, query_rank_elements);
    launch_projection(query_raw, query_rank_quant, query_b, state.stream);
    dsv4_query_norm_rope<<<64U, kDsv4QueryNormRopeThreads, 0U, state.stream>>>(
        query_raw, device_cosines, device_sines, prepared_query, failure);
    const auto kv_stream = state.dsv4_attention_aux_streams[0U];
    launch_projection(key_value_raw, input_quant, key_value_weight, kv_stream);
    dsv4_key_value_norm_rope<<<1U, kDsv4KeyValueNormThreads, 0U, kv_stream>>>(
        key_value_raw, device_key_value_norm, device_cosines, device_sines,
        key_value_bf16, failure);
    if (auto status = cudaEventRecord(
            state.dsv4_attention_aux_finished[0U], kv_stream);
        status != cudaSuccess) {
        if (transition_mhc) mhc_state.dsv4_mhc_stage = 0U;
        return cuda_error(status,
                          "record DeepSeek key/value completion");
    }
    if (prepare_compressor) {
        const auto compressor_stream = state.dsv4_attention_aux_streams[1U];
        launch_compressor(compressor_value_raw, request.compressor_value,
                          compressor_stream);
        launch_compressor(compressor_score_raw, request.compressor_gate,
                          compressor_stream);
        if (auto status = cudaEventRecord(
                state.dsv4_attention_aux_finished[1U], compressor_stream);
            status != cudaSuccess) {
            if (transition_mhc) mhc_state.dsv4_mhc_stage = 0U;
            return cuda_error(status,
                              "record DeepSeek compressor completion");
        }
    }
    if (prepare_index_compressor) {
        const auto index_stream = state.dsv4_attention_aux_streams[2U];
        launch_compressor(index_compressor_value_raw,
                          request.index_compressor_value, index_stream);
        launch_compressor(index_compressor_score_raw,
                          request.index_compressor_gate, index_stream);
        if (auto status = cudaEventRecord(
                state.dsv4_attention_aux_finished[2U], index_stream);
            status != cudaSuccess) {
            if (transition_mhc) mhc_state.dsv4_mhc_stage = 0U;
            return cuda_error(
                status, "record DeepSeek index-compressor completion");
        }
    }
    if (auto status = cudaStreamWaitEvent(
            state.stream, state.dsv4_attention_aux_finished[0U]);
        status != cudaSuccess) {
        if (transition_mhc) mhc_state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "join DeepSeek key/value projection");
    }
    if (prepare_compressor) {
        if (auto status = cudaStreamWaitEvent(
                state.stream, state.dsv4_attention_aux_finished[1U]);
            status != cudaSuccess) {
            if (transition_mhc) mhc_state.dsv4_mhc_stage = 0U;
            return cuda_error(status, "join DeepSeek compressor projection");
        }
    }
    if (prepare_index_compressor) {
        if (auto status = cudaStreamWaitEvent(
                state.stream, state.dsv4_attention_aux_finished[2U]);
            status != cudaSuccess) {
            if (transition_mhc) mhc_state.dsv4_mhc_stage = 0U;
            return cuda_error(
                status, "join DeepSeek index-compressor projection");
        }
    }
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        if (transition_mhc) mhc_state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "launch attention preparation kernels");
    }
    // Publish the two index-projection sources for a following in-chain index
    // command.
    state.dsv4_prepared_index_query_source = query_rank_quant;
    state.dsv4_prepared_index_hidden_source =
        expand_input ? compressor_input : nullptr;
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.kernel_finished, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "record attention preparation kernel completion");
        }
    }
    if (device_only) {
        if (ready_page_patch) {
            if (auto status = cudaStreamWaitEvent(
                    state.stream,
                    static_cast<cudaEvent_t>(request.page_patch_ready_event));
                status != cudaSuccess) {
                return cuda_error(
                    status, "wait for canonical DeepSeek page patch");
            }
            std::uint64_t patch_cursor = 0U;
            for (const auto& write : request.page_writes) {
                auto* destination = static_cast<std::byte*>(
                    write.buffer->impl_->data);
                if (auto status = cudaMemcpyAsync(
                        destination + write.data_offset,
                        request.ready_page_patches.data() + patch_cursor,
                        write.data_bytes, cudaMemcpyHostToDevice,
                        state.stream); status != cudaSuccess) {
                    return cuda_error(
                        status, "replicate DeepSeek prepared page data");
                }
                patch_cursor += write.data_bytes;
                if (auto status = cudaMemcpyAsync(
                        destination + write.scale_offset,
                        request.ready_page_patches.data() + patch_cursor,
                        write.scale_bytes, cudaMemcpyHostToDevice,
                        state.stream); status != cudaSuccess) {
                    return cuda_error(
                        status, "replicate DeepSeek prepared page scale");
                }
                patch_cursor += write.scale_bytes;
            }
        }
        state.dsv4_prepared_queries = prepared_query;
        state.dsv4_attention_prepared = !host_only;
        // Retire this command's upload slot.  The device-only path enqueues no
        // host node, so its command record stays empty and reports no failure,
        // but the slot must not be reused until the chain has been drained.
        state.dsv4_attention_prepare_host_commands[
            state.dsv4_attention_prepare_host_command_count++] = {};
        {
            std::scoped_lock lock(impl_->mutex);
            auto& stats = *std::find_if(
                impl_->stats.devices.begin(), impl_->stats.devices.end(),
                [device](const auto& value) { return value.device == device; });
            stats.matmul_calls += 3U;
            stats.activation_h2d_bytes += upload_bytes + page_patch_bytes;
            stats.workspace_allocation_calls += allocation_calls;
            stats.workspace_allocation_bytes += allocation_bytes;
        }
        return result;
    }
    auto* host_download = state.dsv4_attention_prepare_host_download;
    if (auto status = cudaMemcpyAsync(
            host_download, query_rank_bf16, query_rank_download_bytes,
            cudaMemcpyDeviceToHost, state.stream); status != cudaSuccess) {
        return cuda_error(status, "download prepared query rank");
    }
    if (auto status = cudaMemcpyAsync(
            host_download + query_rank_download_bytes, key_value_bf16,
            key_value_download_bytes, cudaMemcpyDeviceToHost, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "download prepared key/value row");
    }
    if (prepare_compressor) {
        if (auto status = cudaMemcpyAsync(
                host_download + compressor_value_download_offset,
                compressor_value_raw, compressor_values.size_bytes(),
                cudaMemcpyDeviceToHost, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "download prepared compressor values");
        }
        if (auto status = cudaMemcpyAsync(
                host_download + compressor_score_download_offset,
                compressor_score_raw, compressor_scores.size_bytes(),
                cudaMemcpyDeviceToHost, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "download prepared compressor scores");
        }
    }
    if (prepare_index_compressor) {
        if (auto status = cudaMemcpyAsync(
                host_download + index_compressor_value_download_offset,
                index_compressor_value_raw,
                index_compressor_values.size_bytes(),
                cudaMemcpyDeviceToHost, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "download prepared index-compressor values");
        }
        if (auto status = cudaMemcpyAsync(
                host_download + index_compressor_score_download_offset,
                index_compressor_score_raw,
                index_compressor_scores.size_bytes(),
                cudaMemcpyDeviceToHost, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "download prepared index-compressor scores");
        }
    }
    if (auto status = cudaMemcpyAsync(
            host_download + failure_download_offset,
            failure, sizeof(*failure), cudaMemcpyDeviceToHost, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "download attention preparation status");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                state.activation_downloaded, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "record attention preparation download completion");
        }
    }
    if (host_deferred) {
        if (state.dsv4_attention_prepare_host_command_count >=
            state.dsv4_attention_prepare_host_commands.size()) {
            if (transition_mhc) mhc_state.dsv4_mhc_stage = 0U;
            result.errors.emplace_back(
                "DeepSeek attention preparation command chain is full");
            return result;
        }
        auto& command = state.dsv4_attention_prepare_host_commands[
            state.dsv4_attention_prepare_host_command_count++];
        command = {};
        command.function = request.host_callback;
        command.context = request.host_callback_context;
        command.query_rank = reinterpret_cast<const std::uint16_t*>(
            host_download);
        command.key_value = reinterpret_cast<const std::uint16_t*>(
            host_download + query_rank_download_bytes);
        command.compressor_values = reinterpret_cast<const float*>(
            host_download + compressor_value_download_offset);
        command.compressor_scores = reinterpret_cast<const float*>(
            host_download + compressor_score_download_offset);
        command.index_compressor_values = reinterpret_cast<const float*>(
            host_download + index_compressor_value_download_offset);
        command.index_compressor_scores = reinterpret_cast<const float*>(
            host_download + index_compressor_score_download_offset);
        command.compressor_elements = compressor_values.size();
        command.index_compressor_elements =
            index_compressor_values.size();
        command.page_patches = host_download + download_bytes;
        command.page_patch_bytes = page_patch_bytes;
        command.upstream_failure = reinterpret_cast<const unsigned int*>(
            host_download + failure_download_offset);
        if (auto status = cudaLaunchHostFunc(
                state.stream,
                run_dsv4_attention_prepare_host_callback, &command);
            status != cudaSuccess) {
            if (transition_mhc) mhc_state.dsv4_mhc_stage = 0U;
            return cuda_error(
                status, "enqueue DeepSeek attention preparation host node");
        }
        std::uint64_t patch_cursor = 0U;
        for (const auto& write : request.page_writes) {
            auto* destination = static_cast<std::byte*>(
                write.buffer->impl_->data);
            if (auto status = cudaMemcpyAsync(
                    destination + write.data_offset,
                    command.page_patches + patch_cursor,
                    write.data_bytes, cudaMemcpyHostToDevice, state.stream);
                status != cudaSuccess) {
                if (transition_mhc) mhc_state.dsv4_mhc_stage = 0U;
                return cuda_error(
                    status, "patch DeepSeek prepared page data");
            }
            patch_cursor += write.data_bytes;
            if (auto status = cudaMemcpyAsync(
                    destination + write.scale_offset,
                    command.page_patches + patch_cursor,
                    write.scale_bytes, cudaMemcpyHostToDevice, state.stream);
                status != cudaSuccess) {
                if (transition_mhc) mhc_state.dsv4_mhc_stage = 0U;
                return cuda_error(
                    status, "patch DeepSeek prepared page scale");
            }
            patch_cursor += write.scale_bytes;
        }
        {
            std::scoped_lock lock(impl_->mutex);
            auto& stats = *std::find_if(
                impl_->stats.devices.begin(), impl_->stats.devices.end(),
                [device](const auto& value) {
                    return value.device == device;
                });
            stats.matmul_calls += 3U +
                (prepare_compressor ? 2U : 0U) +
                (prepare_index_compressor ? 2U : 0U);
            stats.activation_h2d_bytes += upload_bytes + page_patch_bytes;
            stats.activation_d2h_bytes += download_bytes;
            stats.workspace_allocation_calls += allocation_calls;
            stats.workspace_allocation_bytes += allocation_bytes;
            if (transition_mhc && request.mhc_device == device) {
                ++stats.dsv4_mhc_calls;
                ++stats.dsv4_mhc_transition_calls;
                stats.dsv4_mhc_kernel_launches += 3U;
            }
            if (transition_mhc && request.mhc_device != device) {
                auto& target_stats = *std::find_if(
                    impl_->stats.devices.begin(), impl_->stats.devices.end(),
                    [&](const auto& value) {
                        return value.device == request.mhc_device;
                    });
                constexpr auto layer_bytes =
                    hidden * sizeof(std::uint16_t);
                ++target_stats.dsv4_mhc_calls;
                ++target_stats.dsv4_mhc_transition_calls;
                target_stats.dsv4_mhc_kernel_launches += 3U;
                target_stats.dsv4_mhc_d2h_bytes += layer_bytes;
                target_stats.activation_d2h_bytes += layer_bytes;
            }
            if (device_mhc_input) {
                auto& target_stats = *std::find_if(
                    impl_->stats.devices.begin(), impl_->stats.devices.end(),
                    [&](const auto& value) {
                        return value.device == request.mhc_device;
                    });
                constexpr auto layer_bytes =
                    hidden * sizeof(std::uint16_t);
                target_stats.dsv4_mhc_d2h_bytes += layer_bytes;
                target_stats.activation_d2h_bytes += layer_bytes;
            }
        }
        state.dsv4_prepared_queries = prepared_query;
        state.dsv4_attention_prepared = !host_only;
        return result;
    }
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaStreamSynchronize(state.stream);
        status != cudaSuccess) {
        if (transition_mhc) mhc_state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "synchronize attention preparation");
    }
    const auto wait_nanoseconds = elapsed_nanoseconds_since(wait_started);
    const auto operation_nanoseconds = elapsed_nanoseconds_since(
        operation_started);
    unsigned int numerical_error = 0U;
    std::memcpy(&numerical_error,
                host_download + failure_download_offset,
                sizeof(numerical_error));
    if (numerical_error != 0U) {
        result.errors.emplace_back(
            "DeepSeek attention preparation produced a non-finite value");
        return result;
    }
    const auto decode = [](const std::uint16_t* source,
                           std::span<float> destination) {
        for (std::size_t index = 0U; index < destination.size(); ++index) {
            destination[index] = std::bit_cast<float>(
                static_cast<std::uint32_t>(source[index]) << 16U);
        }
    };
    const auto* encoded = reinterpret_cast<const std::uint16_t*>(host_download);
    decode(encoded, query_rank);
    decode(encoded + query_rank_elements, key_value);
    if (prepare_compressor) {
        std::memcpy(compressor_values.data(),
                    host_download + compressor_value_download_offset,
                    compressor_values.size_bytes());
        std::memcpy(compressor_scores.data(),
                    host_download + compressor_score_download_offset,
                    compressor_scores.size_bytes());
    }
    if (prepare_index_compressor) {
        std::memcpy(index_compressor_values.data(),
                    host_download + index_compressor_value_download_offset,
                    index_compressor_values.size_bytes());
        std::memcpy(index_compressor_scores.data(),
                    host_download + index_compressor_score_download_offset,
                    index_compressor_scores.size_bytes());
    }
    std::uint64_t h2d_nanoseconds = 0U;
    std::uint64_t kernel_nanoseconds = 0U;
    std::uint64_t d2h_nanoseconds = 0U;
    std::uint64_t mhc_transition_nanoseconds = 0U;
    std::uint64_t mhc_transition_d2h_nanoseconds = 0U;
    std::uint64_t mhc_timing_clamped_samples = 0U;
    if (impl_->detailed_timing) {
        float h2d_ms = 0.0F;
        float kernel_ms = 0.0F;
        float d2h_ms = 0.0F;
        if (auto status = cudaEventElapsedTime(
                &h2d_ms, state.activation_start,
                state.activation_uploaded); status != cudaSuccess) {
            return cuda_error(status, "measure attention preparation upload");
        }
        if (auto status = cudaEventElapsedTime(
                &kernel_ms, state.activation_uploaded,
                state.kernel_finished); status != cudaSuccess) {
            return cuda_error(status, "measure attention preparation kernels");
        }
        if (auto status = cudaEventElapsedTime(
                &d2h_ms, state.kernel_finished,
                state.activation_downloaded); status != cudaSuccess) {
            return cuda_error(status, "measure attention preparation download");
        }
        if (transition_mhc) {
            float mhc_ms = 0.0F;
            if (request.mhc_device == device) {
                if (auto status = cudaEventElapsedTime(
                        &mhc_ms, state.activation_uploaded,
                        state.mhc_transition_finished);
                    status != cudaSuccess) {
                    return cuda_error(
                        status, "measure combined mHC transition");
                }
            } else {
                if (auto status = cudaSetDevice(request.mhc_device);
                    status != cudaSuccess) {
                    return cuda_error(
                        status, "select mHC device for transition timing");
                }
                float mhc_d2h_ms = 0.0F;
                if (cudaEventElapsedTime(
                        &mhc_ms, mhc_state.activation_start,
                        mhc_state.kernel_finished) != cudaSuccess ||
                    cudaEventElapsedTime(
                        &mhc_d2h_ms, mhc_state.kernel_finished,
                        mhc_state.activation_downloaded) != cudaSuccess) {
                    static_cast<void>(cudaSetDevice(device));
                    return {{"measure cross-device mHC transition failed"}};
                }
                if (auto status = cudaSetDevice(device);
                    status != cudaSuccess) {
                    return cuda_error(
                        status, "restore attention device after transition timing");
                }
                mhc_transition_d2h_nanoseconds =
                    event_milliseconds_to_nanoseconds(
                        mhc_d2h_ms, mhc_timing_clamped_samples);
            }
            mhc_transition_nanoseconds =
                event_milliseconds_to_nanoseconds(
                    mhc_ms, mhc_timing_clamped_samples);
        }
        h2d_nanoseconds = static_cast<std::uint64_t>(std::llround(
            static_cast<double>(h2d_ms) * 1.0e6));
        kernel_nanoseconds = static_cast<std::uint64_t>(std::llround(
            static_cast<double>(kernel_ms) * 1.0e6));
        d2h_nanoseconds = static_cast<std::uint64_t>(std::llround(
            static_cast<double>(d2h_ms) * 1.0e6));
    }
    {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        stats.matmul_calls += 3U + (prepare_compressor ? 2U : 0U) +
                              (prepare_index_compressor ? 2U : 0U);
        stats.activation_h2d_bytes += upload_bytes;
        stats.activation_d2h_bytes += download_bytes;
        stats.activation_h2d_nanoseconds += h2d_nanoseconds;
        stats.kernel_nanoseconds += kernel_nanoseconds;
        stats.activation_d2h_nanoseconds += d2h_nanoseconds;
        if (transition_mhc && request.mhc_device == device) {
            ++stats.dsv4_mhc_calls;
            ++stats.dsv4_mhc_transition_calls;
            stats.dsv4_mhc_kernel_launches += 3U;
            stats.dsv4_mhc_kernel_nanoseconds +=
                mhc_transition_nanoseconds;
            stats.dsv4_mhc_nanoseconds += mhc_transition_nanoseconds;
            stats.dsv4_mhc_device_nanoseconds +=
                mhc_transition_nanoseconds;
            stats.dsv4_mhc_timing_clamped_samples +=
                mhc_timing_clamped_samples;
        }
        stats.workspace_allocation_calls += allocation_calls;
        stats.workspace_allocation_bytes += allocation_bytes;
        record_synchronization(stats, SynchronizationSubsystem::Attention,
                               1U, wait_nanoseconds);
        if (transition_mhc && request.mhc_device != device) {
            auto& target_stats = *std::find_if(
                impl_->stats.devices.begin(), impl_->stats.devices.end(),
                [&](const auto& value) {
                    return value.device == request.mhc_device;
                });
            constexpr auto layer_bytes =
                hidden * sizeof(std::uint16_t);
            ++target_stats.dsv4_mhc_calls;
            ++target_stats.dsv4_mhc_transition_calls;
            target_stats.dsv4_mhc_kernel_launches += 3U;
            target_stats.dsv4_mhc_d2h_bytes += layer_bytes;
            target_stats.dsv4_mhc_kernel_nanoseconds +=
                mhc_transition_nanoseconds;
            target_stats.dsv4_mhc_d2h_nanoseconds +=
                mhc_transition_d2h_nanoseconds;
            target_stats.dsv4_mhc_nanoseconds +=
                mhc_transition_nanoseconds +
                mhc_transition_d2h_nanoseconds;
            target_stats.dsv4_mhc_device_nanoseconds +=
                mhc_transition_nanoseconds +
                mhc_transition_d2h_nanoseconds;
            target_stats.dsv4_mhc_timing_clamped_samples +=
                mhc_timing_clamped_samples;
            target_stats.activation_d2h_bytes += layer_bytes;
            target_stats.activation_d2h_nanoseconds +=
                mhc_transition_d2h_nanoseconds;
        }
    }
    static_cast<void>(operation_nanoseconds);
    state.dsv4_prepared_queries = prepared_query;
    state.dsv4_attention_prepared = !host_only;
    return result;
}

ValidationResult CudaBackend::dsv4_copy_prepared_queries(
    int device, std::span<float> output) {
    ValidationResult result;
    constexpr std::size_t query_elements = 64U * 512U;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end() || output.size() != query_elements ||
        found->second.dsv4_prepared_queries == nullptr) {
        result.errors.emplace_back(
            "DeepSeek prepared query capture request is invalid");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for prepared query capture");
    }
    if (auto status = cudaStreamSynchronize(found->second.stream);
        status != cudaSuccess) {
        return cuda_error(status, "synchronize DeepSeek prepared query capture");
    }
    std::vector<__nv_bfloat16> encoded(output.size());
    if (auto status = cudaMemcpy(encoded.data(),
                                 found->second.dsv4_prepared_queries,
                                 encoded.size() * sizeof(__nv_bfloat16),
                                 cudaMemcpyDeviceToHost); status != cudaSuccess) {
        return cuda_error(status, "copy DeepSeek prepared query capture");
    }
    for (std::size_t index = 0U; index < output.size(); ++index) {
        output[index] = __bfloat162float(encoded[index]);
        if (!std::isfinite(output[index]) ||
            bf16_round_f32(output[index]) != output[index]) {
            result.errors.emplace_back(
                "DeepSeek prepared query capture is non-finite or not BF16");
            return result;
        }
    }
    return result;
}

namespace {

bool dsv4_validate_device_pointer(
    int device, const void* pointer, const char* name,
    ValidationResult& result);

}  // namespace

ParseResult<std::uint64_t>
CudaBackend::dsv4_paged_attention_to_mhc_page_workspace_bytes(
    std::span<const CudaDsv4PhysicalPage> pages, std::uint32_t rows,
    std::uint32_t candidate_width, bool project_page_query) const {
    ParseResult<std::uint64_t> result{};
    if (rows < 2U || pages.empty() ||
        pages.size() > std::numeric_limits<std::uint32_t>::max() ||
        candidate_width == 0U || candidate_width > 640U ||
        candidate_width % kDsv4PagedCandidateBlock != 0U) {
        result.errors.emplace_back(
            "DeepSeek attention page workspace request is invalid");
        return result;
    }
    std::uint64_t flat_rows64 = 0U;
    for (const auto& page : pages) {
        if ((page.rows != 2U && page.rows != 64U && page.rows != 256U) ||
            page.rows > std::numeric_limits<std::uint64_t>::max() -
                            flat_rows64) {
            result.errors.emplace_back(
                "DeepSeek attention page workspace extent is invalid");
            return result;
        }
        flat_rows64 += page.rows;
    }
    if (flat_rows64 == 0U ||
        flat_rows64 > std::numeric_limits<std::uint32_t>::max()) {
        result.errors.emplace_back(
            "DeepSeek attention page workspace extent overflows");
        return result;
    }
    Dsv4AttentionMhcWorkspaceLayout layout;
    if (!dsv4_attention_mhc_workspace_layout(
            pages.size(), rows, 64U, 8U, candidate_width,
            static_cast<std::uint32_t>(flat_rows64), false,
            project_page_query, rows, 0U, 0U, layout)) {
        result.errors.emplace_back(
            "DeepSeek attention page workspace layout overflows");
        return result;
    }
    result.value = layout.workspace_bytes;
    return result;
}

ParseResult<std::uint32_t>
CudaBackend::dsv4_paged_attention_to_mhc_page_maximum_rows(
    std::span<const CudaDsv4PhysicalPage> pages,
    std::uint32_t requested_rows, std::uint32_t candidate_width,
    std::uint64_t maximum_workspace_bytes, bool project_page_query) const {
    ParseResult<std::uint32_t> result{};
    if (requested_rows < 2U || maximum_workspace_bytes == 0U) {
        result.errors.emplace_back(
            "DeepSeek attention page row admission request is invalid");
        return result;
    }
    auto minimum = dsv4_paged_attention_to_mhc_page_workspace_bytes(
        pages, 2U, candidate_width, project_page_query);
    if (!minimum.ok()) return {0U, std::move(minimum.errors)};
    if (minimum.value > maximum_workspace_bytes) {
        result.errors.emplace_back(
            "DeepSeek attention page cannot fit two rows in its bounded workspace");
        return result;
    }
    auto full = dsv4_paged_attention_to_mhc_page_workspace_bytes(
        pages, requested_rows, candidate_width, project_page_query);
    if (!full.ok()) return {0U, std::move(full.errors)};
    if (full.value <= maximum_workspace_bytes) {
        result.value = requested_rows;
        return result;
    }
    std::uint32_t lower = 2U;
    std::uint32_t upper = requested_rows - 1U;
    while (lower < upper) {
        const auto middle = lower + (upper - lower + 1U) / 2U;
        auto workspace = dsv4_paged_attention_to_mhc_page_workspace_bytes(
            pages, middle, candidate_width, project_page_query);
        if (!workspace.ok()) return {0U, std::move(workspace.errors)};
        if (workspace.value <= maximum_workspace_bytes) {
            lower = middle;
        } else {
            upper = middle - 1U;
        }
    }
    result.value = lower;
    return result;
}

ValidationResult CudaBackend::dsv4_paged_attention_to_mhc(
    int device, const CudaDsv4PagedAttentionMhcRequest& request,
    std::span<float> diagnostic_branch) {
    ValidationResult result;
    const auto call_started = std::chrono::steady_clock::now();
    const auto rows = request.attention.rows;
    const std::uint32_t total_heads = request.rank_local ? 32U : 64U;
    const std::uint32_t output_groups = request.rank_local ? 4U : 8U;
    constexpr std::uint32_t rope_pairs = 32U;
    const std::uint64_t attended_row_elements =
        static_cast<std::uint64_t>(total_heads) * kDsv4PagedHeadDim;
    const std::uint64_t attended_elements =
        static_cast<std::uint64_t>(rows) * attended_row_elements;
    constexpr std::uint64_t group_elements =
        static_cast<std::uint64_t>(kDsv4PagedHeads) * kDsv4PagedHeadDim;
    const std::uint64_t output_rank_row_elements =
        static_cast<std::uint64_t>(output_groups) * 1024U;
    const std::uint64_t output_rank_elements =
        static_cast<std::uint64_t>(rows) * output_rank_row_elements;
    constexpr std::uint64_t branch_row_elements = kDsv4MhcHidden;
    const std::uint64_t branch_elements =
        static_cast<std::uint64_t>(rows) * branch_row_elements;
    const auto candidates = request.attention.candidate_width == 0U &&
            rows == 1U
        ? static_cast<std::uint32_t>(request.attention.candidates.size())
        : request.attention.candidate_width;
    const auto total_candidates = static_cast<std::uint64_t>(rows) *
                                  candidates;
    const auto* output_a = request.output_a;
    const auto* output_b = request.output_b;
    const auto source_found = impl_->devices.find(device);
    const auto target_found = impl_->devices.find(request.mhc_device);
    const bool project_page_query = request.page_query_projection != nullptr;
    const bool use_prepared_query =
        request.attention.queries.empty() && !project_page_query;
    const bool transition_mhc = request.mhc_transition != nullptr;
    const bool project_router = request.router != nullptr;
    const bool defer_host_moe_input = request.defer_host_moe_input;
    const bool fixed_command_staging = defer_host_moe_input || request.rank_local;
    const bool page_request = rows > 1U;
    if (source_found == impl_->devices.end() ||
        target_found == impl_->devices.end() ||
        rows == 0U || request.attention.candidates.size() != total_candidates ||
        (page_request &&
         (request.rank_local || use_prepared_query || transition_mhc ||
          project_router || defer_host_moe_input ||
          request.attention.resolution != nullptr ||
          request.mhc_slots.size() != rows || diagnostic_branch.empty())) ||
        (!page_request && !request.mhc_slots.empty()) ||
        (use_prepared_query
             ? (!source_found->second.dsv4_attention_prepared ||
                source_found->second.dsv4_prepared_queries == nullptr ||
                !request.page_query_rank.empty())
             : project_page_query
                 ? (!page_request ||
                    !request.attention.queries.empty() ||
                    request.page_query_rank.size() !=
                        static_cast<std::size_t>(rows) * 1024U ||
                    source_found->second.dsv4_attention_prepared)
                 : (request.attention.queries.size() != attended_elements ||
                    !request.page_query_rank.empty() ||
                    source_found->second.dsv4_attention_prepared)) ||
        request.attention.head_sinks.size() != total_heads ||
        (request.rank_local &&
         (request.head_offset != 0U && request.head_offset != 32U)) ||
        (!request.rank_local && request.head_offset != 0U) ||
        (request.rank_local
             ? request.rank_local_raw_fp32_reduction == nullptr
             : request.rank_local_raw_fp32_reduction != nullptr) ||
        request.attention.pages.empty() ||
        request.attention.pages.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        candidates == 0U || candidates > 640U ||
        candidates % kDsv4PagedCandidateBlock != 0U ||
        !std::isfinite(request.attention.scale) ||
        request.attention.scale <= 0.0F ||
        request.attention.maximum_workspace_bytes == 0U ||
        request.inverse_rope_cosines.size() !=
            static_cast<std::size_t>(rows) * rope_pairs ||
        request.inverse_rope_sines.size() !=
            static_cast<std::size_t>(rows) * rope_pairs ||
        output_a == nullptr || output_b == nullptr ||
        !output_a->valid() || !output_b->valid() ||
        output_a->device() != device || output_b->device() != device ||
        (project_page_query &&
         (!request.page_query_projection->valid() ||
          request.page_query_projection->device() != device)) ||
        (request.rank_local && request.mhc_device != device) ||
        (transition_mhc
             ? (!request.mhc_transition->valid() ||
                request.mhc_transition->device() != request.mhc_device ||
                (defer_host_moe_input
                     ? !request.mhc_layer_input.empty()
                     : request.mhc_layer_input.size() != branch_row_elements) ||
                !diagnostic_branch.empty())
             : !request.mhc_layer_input.empty()) ||
        (project_router
             ? (!transition_mhc || !request.router->valid() ||
                request.router->device() != request.mhc_device ||
                (defer_host_moe_input
                     ? !request.router_logits.empty()
                     : request.router_logits.empty()))
             : !request.router_logits.empty()) ||
        (defer_host_moe_input &&
         (!transition_mhc || !project_router || !use_prepared_query)) ||
        (!diagnostic_branch.empty() &&
         diagnostic_branch.size() != branch_elements) ||
        (!request.page_query_diagnostic.empty() &&
         (!project_page_query ||
          request.page_query_diagnostic.size() != attended_elements)) ||
        std::any_of(request.attention.queries.begin(),
                    request.attention.queries.end(), [](float value) {
                        return !std::isfinite(value) ||
                               bf16_round_f32(value) != value;
                    }) ||
        std::any_of(request.page_query_rank.begin(),
                    request.page_query_rank.end(), [](float value) {
                        return !std::isfinite(value) ||
                               bf16_round_f32(value) != value;
                    }) ||
        std::any_of(request.attention.head_sinks.begin(),
                    request.attention.head_sinks.end(),
                    [](float value) { return !std::isfinite(value); }) ||
        std::any_of(request.inverse_rope_cosines.begin(),
                    request.inverse_rope_cosines.end(),
                    [](float value) { return !std::isfinite(value); }) ||
        std::any_of(request.inverse_rope_sines.begin(),
                    request.inverse_rope_sines.end(),
                    [](float value) { return !std::isfinite(value); })) {
        result.errors.emplace_back(
            "DeepSeek paged attention-to-mHC request is invalid");
        return result;
    }
    if (request.rank_local) {
        if (!dsv4_validate_device_pointer(
                device, request.rank_local_raw_fp32_reduction,
                "rank-local raw FP32 reduction destination", result)) {
            return result;
        }
    }
    const auto& a = output_a->impl_->descriptor;
    const auto& b = output_b->impl_->descriptor;
    const auto* page_query_descriptor = project_page_query
        ? &request.page_query_projection->impl_->descriptor : nullptr;
    const bool expanded_output_b = request.rank_local &&
        b.encoding == CudaWeightEncoding::Plain &&
        b.dtype == SafetensorsDtype::Bf16;
    const auto* router_descriptor = project_router
        ? &request.router->impl_->descriptor : nullptr;
    if (a.encoding != CudaWeightEncoding::Plain ||
        a.dtype != SafetensorsDtype::Bf16 ||
        a.rows != output_rank_row_elements || a.columns != 4096U ||
        (!((b.encoding == CudaWeightEncoding::Fp8E4m3Block128 &&
            b.dtype == SafetensorsDtype::F8E4M3 && b.group_size == 128U) ||
           expanded_output_b)) ||
        b.rows != branch_row_elements ||
        b.columns != output_rank_row_elements) {
        result.errors.emplace_back(
            "DeepSeek attention output weights violate the accepted mixed BF16/FP8 contract");
        return result;
    }
    if (project_page_query &&
        (page_query_descriptor->encoding !=
             CudaWeightEncoding::Fp8E4m3Block128 ||
         page_query_descriptor->dtype != SafetensorsDtype::F8E4M3 ||
         page_query_descriptor->group_size != 128U ||
         page_query_descriptor->rows != 2U * group_elements ||
         page_query_descriptor->columns != 1024U ||
         page_query_descriptor->columns % kDsv4Fp8TensorBlockK != 0U ||
         page_query_descriptor->rows % kDsv4Fp8TensorBlockN != 0U)) {
        result.errors.emplace_back(
            "DeepSeek page query projection violates the accepted FP8 contract");
        return result;
    }
    if (project_router &&
        (router_descriptor->encoding != CudaWeightEncoding::Plain ||
         router_descriptor->dtype != SafetensorsDtype::Bf16 ||
         router_descriptor->rows != kDsv4MhcRouterLogits ||
         (!defer_host_moe_input &&
          router_descriptor->rows != request.router_logits.size()) ||
         router_descriptor->columns != branch_elements)) {
        result.errors.emplace_back(
            "DeepSeek FFN router violates the accepted BF16 contract");
        return result;
    }
    auto& state = source_found->second;
    auto& target = target_found->second;
    if (state.moe_in_flight || target.moe_in_flight ||
        !state.dsv4_paged_attention_supported ||
        (project_page_query && !state.dsv4_fp8_tensor_page_supported) ||
        !target.dsv4_mhc_supported || target.dsv4_mhc_workspace == nullptr ||
        (!page_request &&
         (target.dsv4_mhc_stage != 1U || target.dsv4_mhc_branch_ready)) ||
        (page_request &&
         (target.dsv4_mhc_slot_arena == nullptr ||
          target.dsv4_mhc_slot_capacity < rows)) ||
        target.dsv4_host_moe_input_pending ||
        (fixed_command_staging &&
         state.dsv4_deferred_attention_command_count >=
             kDsv4FixedCommandCount)) {
        result.errors.emplace_back(
            "DeepSeek attention-to-mHC command order or device support is invalid");
        return result;
    }
    if (page_request) {
        for (const auto slot : request.mhc_slots) {
            if (slot >= target.dsv4_mhc_slot_capacity) {
                result.errors.emplace_back(
                    "DeepSeek attention page mHC slot is out of range");
                return result;
            }
            const auto stage = slot == target.dsv4_mhc_active_slot
                ? target.dsv4_mhc_stage
                : slot < target.dsv4_mhc_saved_slots.size()
                    ? target.dsv4_mhc_saved_slots[slot].stage : 0U;
            const auto branch_ready = slot == target.dsv4_mhc_active_slot
                ? target.dsv4_mhc_branch_ready
                : slot < target.dsv4_mhc_saved_slots.size() &&
                    target.dsv4_mhc_saved_slots[slot].branch_ready;
            if (stage != 1U || branch_ready) {
                result.errors.emplace_back(
                    "DeepSeek attention page mHC slot state is invalid");
                return result;
            }
        }
    }
    if (use_prepared_query) state.dsv4_attention_prepared = false;

    std::uint64_t page_bytes = 0U;
    std::uint64_t flat_rows64 = 0U;
    std::uint32_t maximum_page_rows = 0U;
    for (const auto& page : request.attention.pages) {
        if (page.buffer == nullptr || !page.buffer->valid() ||
            page.buffer->device() != device ||
            (page.rows != 2U && page.rows != 64U && page.rows != 256U) ||
            page.buffer->device_bytes() !=
                static_cast<std::uint64_t>(page.rows) * 584U ||
            page.buffer->device_bytes() >
                std::numeric_limits<std::uint64_t>::max() - page_bytes) {
            result.errors.emplace_back(
                "DeepSeek attention-to-mHC physical page is invalid");
            return result;
        }
        page_bytes += page.buffer->device_bytes();
        flat_rows64 += page.rows;
        maximum_page_rows = std::max(maximum_page_rows, page.rows);
    }
    if (flat_rows64 == 0U ||
        flat_rows64 > std::numeric_limits<std::uint32_t>::max()) {
        result.errors.emplace_back(
            "DeepSeek attention-to-mHC flat page extent overflows");
        return result;
    }
    const auto flat_rows = static_cast<std::uint32_t>(flat_rows64);
    // The resolved region is checked on the device against these same page
    // descriptors, because the host has not seen the selection that fills it.
    const auto* resolution = request.attention.resolution;
    const auto host_candidate_begin = resolution == nullptr
        ? 0U : resolution->compressed_width;
    if (resolution != nullptr &&
        (resolution->selection.positions == nullptr ||
         resolution->selection.error == nullptr ||
         resolution->blocks.empty() ||
         resolution->blocks.size() > request.attention.pages.size() ||
         resolution->compressed_width == 0U ||
         resolution->compressed_width > candidates ||
         resolution->selection.selected > resolution->compressed_width)) {
        result.errors.emplace_back(
            "DeepSeek attention-to-mHC device candidate resolution is invalid");
        return result;
    }
    for (std::size_t index = host_candidate_begin;
         index < request.attention.candidates.size(); ++index) {
        const auto& candidate = request.attention.candidates[index];
        if (candidate.valid &&
            (candidate.page >= request.attention.pages.size() ||
             candidate.row >=
                 request.attention.pages[candidate.page].rows)) {
            result.errors.emplace_back(
                "DeepSeek attention-to-mHC candidate is outside its page");
            return result;
        }
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status,
                          "select CUDA device for attention-to-mHC");
    }
    if (fixed_command_staging &&
        (state.dsv4_deferred_attention_host_upload == nullptr ||
         state.dsv4_deferred_attention_host_download == nullptr)) {
        if (state.dsv4_deferred_attention_host_upload != nullptr ||
            state.dsv4_deferred_attention_host_download != nullptr) {
            return {{"deferred attention staging is only partially allocated"}};
        }
        void* upload = nullptr;
        void* download = nullptr;
        if (auto status = cudaMallocHost(
                &upload,
                static_cast<std::size_t>(
                    kDsv4FixedCommandCount *
                    kDsv4DeferredAttentionUploadSlotBytes));
            status != cudaSuccess) {
            return cuda_error(
                status,
                "allocate fixed deferred attention upload slots");
        }
        if (auto status = cudaMallocHost(
                &download,
                static_cast<std::size_t>(
                    kDsv4FixedCommandCount *
                    kDsv4DeferredAttentionDownloadSlotBytes));
            status != cudaSuccess) {
            static_cast<void>(cudaFreeHost(upload));
            return cuda_error(
                status,
                "allocate fixed deferred attention download slots");
        }
        state.dsv4_deferred_attention_host_upload =
            static_cast<std::byte*>(upload);
        state.dsv4_deferred_attention_host_download =
            static_cast<std::byte*>(download);
    }

    const auto router_logits_bytes = project_router &&
            request.mhc_device == device
        ? static_cast<std::uint64_t>(router_descriptor->rows) * sizeof(float)
        : 0U;
    Dsv4AttentionMhcWorkspaceLayout layout;
    if (!dsv4_attention_mhc_workspace_layout(
            request.attention.pages.size(), rows, total_heads, output_groups,
            candidates, flat_rows, use_prepared_query, project_page_query,
            request.mhc_slots.size(),
            resolution == nullptr ? 0U : resolution->blocks.size(),
            router_logits_bytes, layout)) {
        result.errors.emplace_back(
            "DeepSeek attention-to-mHC workspace layout overflows");
        return result;
    }
    const auto page_offset = layout.page_offset;
    const auto candidate_offset = layout.candidate_offset;
    const auto query_offset = layout.query_offset;
    const auto sink_offset = layout.sink_offset;
    const auto cosine_offset = layout.cosine_offset;
    const auto sine_offset = layout.sine_offset;
    const auto slot_offset = layout.slot_offset;
    const auto block_offset = layout.block_offset;
    const auto page_query_rank_offset = layout.page_query_rank_offset;
    const auto kv_offset = layout.kv_offset;
    const auto score_offset = layout.score_offset;
    const auto maximum_offset = layout.maximum_offset;
    const auto denominator_offset = layout.denominator_offset;
    const auto value_offset = layout.value_offset;
    const auto attended_offset = layout.attended_offset;
    const auto decoded_offset = layout.decoded_offset;
    const auto output_rank_offset = layout.output_rank_offset;
    const auto tensor_values_offset = layout.tensor_values_offset;
    const auto tensor_scales_offset = layout.tensor_scales_offset;
    const auto page_query_values_offset = layout.page_query_values_offset;
    const auto page_query_scales_offset = layout.page_query_scales_offset;
    const auto page_query_raw_offset = layout.page_query_raw_offset;
    const auto page_query_output_offset = layout.page_query_output_offset;
    const auto branch_offset = layout.branch_offset;
    const auto encoded_branch_offset = layout.encoded_branch_offset;
    const auto router_logits_offset = layout.router_logits_offset;
    const auto failure_offset = layout.failure_offset;
    const auto sink_bytes = layout.sink_bytes;
    const auto rope_bytes = layout.rope_bytes;
    const auto slot_bytes = layout.slot_bytes;
    const auto page_query_rank_bytes = layout.page_query_rank_bytes;
    const auto upload_bytes = layout.upload_bytes;
    const auto workspace_bytes = layout.workspace_bytes;
    if (workspace_bytes > request.attention.maximum_workspace_bytes) {
        result.errors.emplace_back(
            "DeepSeek attention-to-mHC workspace exceeds its bounded contract");
        return result;
    }
    std::uint64_t allocation_calls = 0U;
    std::uint64_t allocation_bytes = 0U;
    if (workspace_bytes > state.dsv4_attention_workspace_bytes) {
        auto target_bytes = std::bit_ceil(workspace_bytes);
        if (target_bytes > request.attention.maximum_workspace_bytes) {
            target_bytes = workspace_bytes;
        }
        std::byte* replacement = nullptr;
        if (auto status = cudaMalloc(
                &replacement, static_cast<std::size_t>(target_bytes));
            status != cudaSuccess) {
            return cuda_error(status,
                              "allocate attention-to-mHC workspace");
        }
        if (state.dsv4_attention_workspace != nullptr) {
            static_cast<void>(cudaFree(state.dsv4_attention_workspace));
        }
        state.dsv4_attention_workspace = replacement;
        state.dsv4_attention_workspace_bytes = target_bytes;
        ++allocation_calls;
        allocation_bytes += target_bytes;
    }
    const auto cross_page_branch_bytes = page_request &&
            request.mhc_device != device
        ? branch_elements * sizeof(std::uint16_t) : 0U;
    const auto cross_page_slot_offset =
        (cross_page_branch_bytes + alignof(std::uint32_t) - 1U) &
        ~(static_cast<std::uint64_t>(alignof(std::uint32_t)) - 1U);
    const auto cross_page_staging_bytes = cross_page_slot_offset +
        (page_request && request.mhc_device != device
             ? static_cast<std::uint64_t>(rows) * sizeof(std::uint32_t)
             : 0U);
    if (cross_page_staging_bytes > target.dsv4_attention_workspace_bytes) {
        if (auto status = cudaSetDevice(request.mhc_device);
            status != cudaSuccess) {
            return cuda_error(status,
                              "select target for attention page staging");
        }
        std::byte* replacement = nullptr;
        if (auto status = cudaMalloc(
                &replacement,
                static_cast<std::size_t>(cross_page_staging_bytes));
            status != cudaSuccess) {
            static_cast<void>(cudaSetDevice(device));
            return cuda_error(status,
                              "allocate cross-device attention page staging");
        }
        if (target.dsv4_attention_workspace != nullptr) {
            static_cast<void>(cudaFree(target.dsv4_attention_workspace));
        }
        target.dsv4_attention_workspace = replacement;
        target.dsv4_attention_workspace_bytes = cross_page_staging_bytes;
        ++allocation_calls;
        allocation_bytes += cross_page_staging_bytes;
        if (auto status = cudaSetDevice(device); status != cudaSuccess) {
            return cuda_error(status,
                              "restore source after attention page allocation");
        }
    }
    const auto ensure_host = [&](std::byte*& pointer,
                                 std::uint64_t& capacity,
                                 std::uint64_t required,
                                 const char* operation) -> bool {
        if (required <= capacity) return true;
        const auto target_bytes = std::bit_ceil(required);
        void* replacement = nullptr;
        if (auto status = cudaMallocHost(
                &replacement, static_cast<std::size_t>(target_bytes));
            status != cudaSuccess) {
            result = cuda_error(status, operation);
            return false;
        }
        if (pointer != nullptr) static_cast<void>(cudaFreeHost(pointer));
        pointer = static_cast<std::byte*>(replacement);
        capacity = target_bytes;
        return true;
    };
    const auto download_branch_bytes =
        (request.mhc_device != device || !diagnostic_branch.empty())
            ? branch_elements * sizeof(std::uint16_t) : 0U;
    const auto diagnostic_query_bytes =
        request.page_query_diagnostic.empty()
            ? 0U : attended_elements * sizeof(std::uint16_t);
    const auto transition_layer_bytes = transition_mhc &&
        !defer_host_moe_input
        ? branch_elements * sizeof(std::uint16_t) : 0U;
    const auto router_download_bytes = project_router &&
        !defer_host_moe_input
        ? static_cast<std::uint64_t>(router_descriptor->rows) * sizeof(float)
        : 0U;
    constexpr std::uint64_t attention_failure_bytes = sizeof(unsigned int);
    const auto download_bytes = download_branch_bytes +
                                diagnostic_query_bytes +
                                transition_layer_bytes +
                                router_download_bytes +
                                attention_failure_bytes;
    if (fixed_command_staging &&
        (upload_bytes > kDsv4DeferredAttentionUploadSlotBytes ||
         download_bytes > kDsv4DeferredAttentionDownloadSlotBytes)) {
        result.errors.emplace_back(
            "deferred attention staging exceeds its fixed command slot");
        return result;
    }
    if (!fixed_command_staging &&
        (!ensure_host(state.dsv4_attention_host_upload,
                      state.dsv4_attention_host_upload_bytes, upload_bytes,
                      "allocate pinned attention-to-mHC upload") ||
         !ensure_host(state.dsv4_attention_host_download,
                      state.dsv4_attention_host_download_bytes,
                      download_bytes,
                      "allocate pinned attention-to-mHC download"))) {
        return result;
    }
    auto* command_host_upload = fixed_command_staging
        ? state.dsv4_deferred_attention_host_upload +
              static_cast<std::uint64_t>(
                  state.dsv4_deferred_attention_command_count) *
                  kDsv4DeferredAttentionUploadSlotBytes
        : state.dsv4_attention_host_upload;
    auto* command_host_download = fixed_command_staging
        ? state.dsv4_deferred_attention_host_download +
              static_cast<std::uint64_t>(
                  state.dsv4_deferred_attention_command_count) *
                  kDsv4DeferredAttentionDownloadSlotBytes
        : state.dsv4_attention_host_download;

    auto* host_pages = reinterpret_cast<Dsv4DevicePhysicalPage*>(
        command_host_upload + page_offset);
    std::uint32_t flat_begin = 0U;
    for (std::size_t index = 0U;
         index < request.attention.pages.size(); ++index) {
        const auto& page = request.attention.pages[index];
        host_pages[index] = {
            static_cast<const std::uint8_t*>(page.buffer->impl_->data),
            page.rows, flat_begin};
        flat_begin += page.rows;
    }
    auto* host_candidates =
        reinterpret_cast<Dsv4DeviceAttentionCandidate*>(
            command_host_upload + candidate_offset);
    for (std::size_t index = 0U;
         index < request.attention.candidates.size(); ++index) {
        const auto& candidate = request.attention.candidates[index];
        host_candidates[index] = {
            candidate.page, candidate.row, candidate.valid ? 1U : 0U};
    }
    if (!use_prepared_query && !project_page_query) {
        auto* host_query = reinterpret_cast<std::uint16_t*>(
            command_host_upload + query_offset);
        const auto group_count = request.rank_local ? 1U : 2U;
        for (std::uint32_t group = 0U; group < group_count; ++group) {
            for (std::uint32_t row = 0U; row < rows; ++row) {
                const auto source = static_cast<std::uint64_t>(row) *
                        attended_row_elements +
                    static_cast<std::uint64_t>(group) * group_elements;
                const auto destination =
                    (static_cast<std::uint64_t>(group) * rows + row) *
                    group_elements;
                for (std::uint64_t index = 0U; index < group_elements;
                     ++index) {
                    host_query[destination + index] = bf16_encode(
                        request.attention.queries[source + index]);
                }
            }
        }
    }
    std::memcpy(command_host_upload + sink_offset,
                request.attention.head_sinks.data(), sink_bytes);
    std::memcpy(command_host_upload + cosine_offset,
                request.inverse_rope_cosines.data(), rope_bytes);
    std::memcpy(command_host_upload + sine_offset,
                request.inverse_rope_sines.data(), rope_bytes);
    if (slot_bytes != 0U) {
        std::memcpy(command_host_upload + slot_offset,
                    request.mhc_slots.data(), slot_bytes);
    }
    if (resolution != nullptr) {
        auto* host_blocks = reinterpret_cast<Dsv4DeviceKvBlock*>(
            command_host_upload + block_offset);
        for (std::size_t index = 0U; index < resolution->blocks.size();
             ++index) {
            const auto& block = resolution->blocks[index];
            host_blocks[index] = {block.logical_begin, block.used_rows,
                                  block.compression_ratio};
        }
    }
    if (project_page_query) {
        std::memcpy(command_host_upload + page_query_rank_offset,
                    request.page_query_rank.data(), page_query_rank_bytes);
    }

    auto* workspace = state.dsv4_attention_workspace;
    auto* device_pages = reinterpret_cast<Dsv4DevicePhysicalPage*>(
        workspace + page_offset);
    auto* device_candidates =
        reinterpret_cast<Dsv4DeviceAttentionCandidate*>(
            workspace + candidate_offset);
    auto* device_query = use_prepared_query
        ? state.dsv4_prepared_queries +
              static_cast<std::uint64_t>(request.head_offset) *
                  kDsv4PagedHeadDim
        : project_page_query
            ? reinterpret_cast<__nv_bfloat16*>(
                  workspace + page_query_output_offset)
            : reinterpret_cast<__nv_bfloat16*>(workspace + query_offset);
    auto* device_sink = reinterpret_cast<float*>(workspace + sink_offset);
    auto* device_cosines = reinterpret_cast<float*>(workspace + cosine_offset);
    auto* device_sines = reinterpret_cast<float*>(workspace + sine_offset);
    auto* device_slots = reinterpret_cast<std::uint32_t*>(
        workspace + slot_offset);
    auto* device_kv = reinterpret_cast<__nv_bfloat16*>(workspace + kv_offset);
    auto* device_scores = reinterpret_cast<__nv_bfloat16*>(
        workspace + score_offset);
    auto* device_maximums = reinterpret_cast<float*>(
        workspace + maximum_offset);
    auto* device_denominators = reinterpret_cast<float*>(
        workspace + denominator_offset);
    auto* device_values = reinterpret_cast<float*>(workspace + value_offset);
    auto* device_attended = reinterpret_cast<__nv_bfloat16*>(
        workspace + attended_offset);
    auto* device_decoded =
        reinterpret_cast<__nv_bfloat16*>(workspace + decoded_offset);
    auto* device_output_rank = reinterpret_cast<float*>(
        workspace + output_rank_offset);
    auto* device_branch = reinterpret_cast<float*>(workspace + branch_offset);
    auto* device_encoded_branch = reinterpret_cast<__nv_bfloat16*>(
        workspace + encoded_branch_offset);
    auto* device_router_logits = request.mhc_device == device
        ? reinterpret_cast<float*>(workspace + router_logits_offset)
        : target.dsv4_mhc_workspace->router_logits;
    auto* device_failure = reinterpret_cast<unsigned int*>(
        workspace + failure_offset);
    auto* device_page_query_rank = reinterpret_cast<float*>(
        workspace + page_query_rank_offset);
    auto* device_page_query_values = reinterpret_cast<unsigned char*>(
        workspace + page_query_values_offset);
    auto* device_page_query_scales = reinterpret_cast<unsigned char*>(
        workspace + page_query_scales_offset);
    auto* device_page_query_raw = reinterpret_cast<float*>(
        workspace + page_query_raw_offset);

    const auto operation_started = std::chrono::steady_clock::now();
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_start, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "record attention-to-mHC upload start");
        }
    }
    if (auto status = cudaMemcpyAsync(
            workspace, command_host_upload,
            static_cast<std::size_t>(upload_bytes), cudaMemcpyHostToDevice,
            state.stream); status != cudaSuccess) {
        return cuda_error(status, "upload attention-to-mHC metadata");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                state.activation_uploaded, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "record attention-to-mHC upload completion");
        }
    }
    if (auto status = cudaMemsetAsync(
            device_failure, 0, sizeof(*device_failure), state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "clear attention-to-mHC status");
    }
    constexpr std::uint32_t threads = 256U;
    constexpr std::uint32_t rank_threads = 128U;
    if (project_page_query) {
        const auto& descriptor = *page_query_descriptor;
        const dim3 quantize_grid(
            static_cast<unsigned int>(descriptor.columns / 128U), rows, 1U);
        quantize_activation_e4m3_bytes_kernel<<<
            quantize_grid, 128U, 0U, state.stream>>>(
            device_page_query_values, device_page_query_scales,
            device_page_query_rank,
            static_cast<std::uint32_t>(descriptor.columns), rows);
        const auto padded_rows =
            (static_cast<std::uint64_t>(rows) + kDsv4Fp8TensorBlockM - 1U) /
            kDsv4Fp8TensorBlockM * kDsv4Fp8TensorBlockM;
        const dim3 projection_grid(
            static_cast<unsigned int>(descriptor.rows /
                                      kDsv4Fp8TensorBlockN),
            static_cast<unsigned int>(padded_rows /
                                      kDsv4Fp8TensorBlockM), 1U);
        dsv4_fp8_decode_bf16_tensor_kernel<<<
            projection_grid, threads, 0U, state.stream>>>(
            device_page_query_raw, device_page_query_values,
            device_page_query_scales,
            static_cast<const unsigned char*>(
                request.page_query_projection->impl_->weights),
            static_cast<const unsigned char*>(
                request.page_query_projection->impl_->scales),
            rows, static_cast<std::uint32_t>(descriptor.columns),
            static_cast<std::uint32_t>(descriptor.rows));
        dsv4_page_query_norm_rope<<<
            dim3{64U, rows}, kDsv4QueryNormRopeThreads, 0U,
            state.stream>>>(
            device_page_query_raw, device_cosines, device_sines,
            device_query, rows, device_failure);
        record_cuda_matmul_route(CudaMatmulRoute::Fp8TensorPage);
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            return cuda_error(status,
                              "launch DeepSeek page query projection");
        }
    }
    // Overwrites the compressed region the upload just staged. Stream order
    // makes that safe and keeps the upload one contiguous copy; the selection
    // it reads was enqueued on this same stream and has not been seen by the
    // host. Its failures land in the command's own status word, so an
    // unresolvable row fails the layer rather than attending a wrong page.
    if (resolution != nullptr) {
        dsv4_resolve_candidates_kernel<<<
            (resolution->compressed_width + threads - 1U) / threads, threads,
            0U, state.stream>>>(
            static_cast<const std::uint32_t*>(resolution->selection.positions),
            resolution->selection.selected,
            reinterpret_cast<const Dsv4DeviceKvBlock*>(workspace +
                                                       block_offset),
            static_cast<std::uint32_t>(resolution->blocks.size()),
            device_pages,
            static_cast<std::uint32_t>(request.attention.pages.size()),
            device_candidates, resolution->compressed_width,
            static_cast<const unsigned int*>(resolution->selection.error),
            device_failure);
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            return cuda_error(status,
                              "launch DeepSeek device candidate resolution");
        }
    }
    const auto page_elements = static_cast<std::uint64_t>(maximum_page_rows) *
                               kDsv4PagedHeadDim;
    const dim3 kv_grid(
        static_cast<unsigned int>((page_elements + threads - 1U) / threads),
        static_cast<unsigned int>(request.attention.pages.size()));
    dsv4_materialize_physical_pages<<<kv_grid, threads, 0U, state.stream>>>(
        device_pages,
        static_cast<std::uint32_t>(request.attention.pages.size()),
        device_kv, device_failure);
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    const auto boundaries = candidates / kDsv4PagedCandidateBlock;
    const auto score_elements = static_cast<std::uint64_t>(rows) *
        kDsv4PagedHeads * candidates;
    const auto score_blocks = static_cast<std::uint32_t>(
        (score_elements + threads - 1U) / threads);
    const auto output_blocks = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(rows) * group_elements + threads - 1U) /
        threads);
    const dim3 value_grid(kDsv4PagedHeads,
                          kDsv4PagedHeadDim /
                              kDsv4PagedDimensionsPerBlock,
                          rows);
    const auto group_count = request.rank_local ? 1U : 2U;
    for (std::uint32_t group = 0U; group < group_count; ++group) {
        auto* group_query = device_query +
            static_cast<std::uint64_t>(group) * rows * group_elements;
        auto* group_sink = device_sink + group * kDsv4PagedHeads;
        dsv4_sparse_scores_kernel<<<
            dim3{rows, kDsv4PagedHeads / kDsv4SparseScoreHeads},
            kDsv4SparseScoreHeads * 32U, 0U, state.stream>>>(
            device_scores, group_query, device_kv, device_pages,
            device_candidates, candidates, 0U);
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            return cuda_error(
                status, "launch attention-to-mHC sparse scores");
        }
        dsv4_scale_scores<<<score_blocks, threads, 0U, state.stream>>>(
            device_scores, score_elements, request.attention.scale);
        dsv4_finish_maximums<<<dim3{kDsv4PagedHeads, rows},
                               kDsv4PagedCandidateBlock, 0U,
                               state.stream>>>(
            device_scores, device_pages, device_candidates, group_sink,
            device_maximums, candidates, candidates, boundaries);
        dsv4_finish_denominators<<<dim3{kDsv4PagedHeads, rows},
                                   kDsv4PagedCandidateBlock, 0U,
                                   state.stream>>>(
            device_scores, device_pages, device_candidates, group_sink,
            device_maximums, device_denominators, candidates, candidates,
            boundaries);
        dsv4_finish_values<<<value_grid, kDsv4PagedDimensionsPerBlock,
                             0U, state.stream>>>(
            device_scores, device_pages, device_candidates, device_maximums,
            device_kv, device_denominators, device_attended,
            attended_row_elements,
            static_cast<std::uint64_t>(group) * group_elements,
            candidates, candidates, boundaries);
    }
    const auto attended_blocks = static_cast<std::uint32_t>(
        (attended_elements + threads - 1U) / threads);
    dsv4_inverse_rope_decode<<<attended_blocks, threads, 0U, state.stream>>>(
        device_attended, device_cosines, device_sines, device_decoded,
        rows, total_heads);
    if (request.rank_local) {
        dsv4_rank_bf16_matmul_bf16_input<<<
            output_rank_elements, rank_threads, 0U, state.stream>>>(
            device_output_rank, device_decoded,
            static_cast<const __nv_bfloat16*>(output_a->impl_->weights),
            1U, a.columns, a.rows, output_groups, 1024U);
    } else {
        const dim3 output_a_grid(
            static_cast<unsigned int>(a.rows),
            static_cast<unsigned int>(
                (rows + kPlainMatmulRowTile - 1U) /
                kPlainMatmulRowTile));
        plain_matmul_kernel_bf16_input<kPlainMatmulRowTile>
            <<<output_a_grid, threads, 0U, state.stream>>>(
            device_output_rank, device_decoded, output_a->impl_->weights,
            static_cast<int>(a.dtype), rows, a.columns, a.rows,
            output_groups, 1024U);
    }
    const auto output_rank_blocks = static_cast<std::uint32_t>(
        (output_rank_elements + threads - 1U) / threads);
    dsv4_round_float_bf16<<<output_rank_blocks, threads, 0U, state.stream>>>(
        device_output_rank, output_rank_elements);
    const dim3 output_rank_quantize_grid(
        static_cast<unsigned int>(output_rank_row_elements / 128U), rows, 1U);
    auto* raw_branch_output = request.rank_local
        ? request.rank_local_raw_fp32_reduction : device_branch;
    // The output projection is the largest remaining native_fp8_matmul_kernel
    // launch on the page path: its grid is one block per (output row, batch
    // row), so every batch row re-reads the whole 4096-row weight. Route the
    // multi-row case through the same SM86 tensor path accepted in experiment
    // 0105. It writes whole 64-row tiles, so it targets the padded branch
    // region and the exact rows are copied out when the caller owns the
    // destination.
    const bool tensor_output_b =
        !expanded_output_b && page_request &&
        state.dsv4_fp8_tensor_page_supported &&
        b.encoding == CudaWeightEncoding::Fp8E4m3Block128 &&
        b.columns % kDsv4Fp8TensorBlockK == 0U &&
        b.rows % kDsv4Fp8TensorBlockN == 0U;
    if (expanded_output_b) {
        dsv4_rank_bf16_matmul<<<branch_elements, rank_threads, 0U, state.stream>>>(
            raw_branch_output, device_output_rank,
            static_cast<const __nv_bfloat16*>(output_b->impl_->weights),
            1U, b.columns, b.rows, 0U, 0U);
    } else if (tensor_output_b) {
        auto* tensor_values = reinterpret_cast<unsigned char*>(
            workspace + tensor_values_offset);
        auto* tensor_scales = reinterpret_cast<unsigned char*>(
            workspace + tensor_scales_offset);
        quantize_activation_e4m3_bytes_kernel<<<
            output_rank_quantize_grid, 128U, 0U, state.stream>>>(
            tensor_values, tensor_scales, device_output_rank,
            output_rank_row_elements, rows);
        const auto tensor_padded_rows =
            (static_cast<std::uint64_t>(rows) + kDsv4Fp8TensorBlockM - 1U) /
            kDsv4Fp8TensorBlockM * kDsv4Fp8TensorBlockM;
        const dim3 output_b_tensor_grid(
            static_cast<unsigned int>(b.rows / kDsv4Fp8TensorBlockN),
            static_cast<unsigned int>(
                tensor_padded_rows / kDsv4Fp8TensorBlockM), 1U);
        dsv4_fp8_decode_bf16_tensor_kernel<<<
            output_b_tensor_grid, threads, 0U, state.stream>>>(
            device_branch, tensor_values, tensor_scales,
            static_cast<const unsigned char*>(output_b->impl_->weights),
            static_cast<const unsigned char*>(output_b->impl_->scales), rows,
            static_cast<std::uint32_t>(b.columns),
            static_cast<std::uint32_t>(b.rows));
        if (raw_branch_output != device_branch) {
            if (auto status = cudaMemcpyAsync(
                    raw_branch_output, device_branch,
                    static_cast<std::size_t>(branch_elements) * sizeof(float),
                    cudaMemcpyDeviceToDevice, state.stream);
                status != cudaSuccess) {
                return cuda_error(
                    status, "copy tensor output projection to its destination");
            }
        }
    } else {
        quantize_activation_e4m3_kernel<<<output_rank_quantize_grid, 128U, 0U,
                                          state.stream>>>(
            device_output_rank, output_rank_row_elements, rows);
        native_fp8_matmul_kernel<<<dim3{
            static_cast<unsigned int>(branch_row_elements), rows},
            threads, 0U, state.stream>>>(
            raw_branch_output, device_output_rank,
            static_cast<const unsigned char*>(output_b->impl_->weights),
            static_cast<const unsigned char*>(output_b->impl_->scales),
            b.scale_columns, rows, b.columns, b.rows, 0U, 0U);
    }
    const auto branch_blocks = static_cast<std::uint32_t>(
        (branch_elements + threads - 1U) / threads);
    if (page_request) {
        auto* slot_arena = request.mhc_device == device
            ? target.dsv4_mhc_slot_arena : nullptr;
        dsv4_store_mhc_page_branches<<<branch_blocks, threads, 0U,
                                        state.stream>>>(
            raw_branch_output, device_encoded_branch, slot_arena,
            device_slots, rows);
    } else {
        dsv4_store_mhc_branch<<<branch_blocks, threads, 0U, state.stream>>>(
            raw_branch_output, device_encoded_branch, branch_elements);
    }
    const auto cross_device = request.mhc_device != device;
    const auto cross_transition = cross_device && transition_mhc;
    const auto needs_branch_download = cross_device ||
                                       !diagnostic_branch.empty();
    auto* staged_branch = command_host_download;
    auto* staged_query = staged_branch + download_branch_bytes;
    auto* staged_layer = staged_query + diagnostic_query_bytes;
    auto* staged_router = staged_layer + transition_layer_bytes;
    auto* staged_failure = command_host_download + download_branch_bytes +
                           diagnostic_query_bytes + transition_layer_bytes +
                           router_download_bytes;
    if (!cross_device && !page_request) {
        if (auto status = cudaMemcpyAsync(
                target.dsv4_mhc_workspace->branch, device_encoded_branch,
                static_cast<std::size_t>(branch_elements *
                                         sizeof(std::uint16_t)),
                cudaMemcpyDeviceToDevice, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "retain attention branch in mHC workspace");
        }
    }
    // Rank-local attention is the reusable device-resident boundary.  The
    // local BF16 branch and page-status word are copied into the persistent
    // target workspace, but no host download or stream wait is permitted.
    // The caller obtains the borrowed status pointer through
    // dsv4_mhc_device_view and closes it with the global U32 MAX collective.
    if (request.rank_local) {
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            return cuda_error(status, "launch rank-local attention kernels");
        }
        if (auto status = cudaMemcpyAsync(
                &target.dsv4_mhc_workspace->failure, device_failure,
                sizeof(*device_failure), cudaMemcpyDeviceToDevice,
                state.stream); status != cudaSuccess) {
            return cuda_error(status,
                              "retain rank-local attention status");
        }
        target.dsv4_mhc_branch_ready = false;
        target.dsv4_mhc_failed = false;
        target.dsv4_host_moe_input_pending = false;
        target.dsv4_host_moe_router_logits = nullptr;
        target.dsv4_host_moe_device_failure = nullptr;
        target.dsv4_host_moe_host_failure = nullptr;
        {
            std::scoped_lock lock(impl_->mutex);
            auto& stats = *std::find_if(
                impl_->stats.devices.begin(), impl_->stats.devices.end(),
                [device](const auto& value) { return value.device == device; });
            ++stats.dsv4_paged_attention_calls;
            stats.dsv4_paged_attention_kernel_launches += 19U;
            stats.dsv4_paged_attention_h2d_bytes += upload_bytes;
            stats.dsv4_paged_attention_page_bytes += page_bytes;
            stats.activation_h2d_bytes += upload_bytes;
            stats.workspace_allocation_calls += allocation_calls;
            stats.workspace_allocation_bytes += allocation_bytes;
        }
        ++state.dsv4_deferred_attention_command_count;
        return result;
    }
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        if (transition_mhc) target.dsv4_mhc_stage = 0U;
        return cuda_error(status, "launch attention-to-mHC kernels");
    }
    if (transition_mhc) {
        if (impl_->detailed_timing) {
            if (auto status = cudaEventRecord(
                    state.mhc_transition_finished, state.stream);
                status != cudaSuccess) {
                return cuda_error(
                    status, "record combined attention kernel completion");
            }
        }
    }
    if (cross_transition) {
        if (auto status = cudaMemcpyAsync(
                staged_branch, device_encoded_branch,
                static_cast<std::size_t>(branch_elements *
                                         sizeof(std::uint16_t)),
                cudaMemcpyDeviceToHost, state.stream);
            status != cudaSuccess) {
            target.dsv4_mhc_stage = 0U;
            return cuda_error(status,
                              "stage cross-device attention branch");
        }
        if (auto status = cudaMemcpyAsync(
                staged_failure, device_failure, sizeof(*device_failure),
                cudaMemcpyDeviceToHost, state.stream);
            status != cudaSuccess) {
            target.dsv4_mhc_stage = 0U;
            return cuda_error(status,
                              "stage cross-device attention status");
        }
        if (impl_->detailed_timing) {
            if (auto status = cudaEventRecord(
                    state.activation_downloaded, state.stream);
                status != cudaSuccess) {
                target.dsv4_mhc_stage = 0U;
                return cuda_error(
                    status, "record cross-device attention download");
            }
        }
        if (auto status = cudaEventRecord(
                state.dsv4_cross_device_ready, state.stream);
            status != cudaSuccess) {
            target.dsv4_mhc_stage = 0U;
            return cuda_error(status,
                              "publish cross-device attention branch");
        }
        if (auto status = cudaSetDevice(request.mhc_device);
            status != cudaSuccess) {
            target.dsv4_mhc_stage = 0U;
            return cuda_error(status,
                              "select target for combined mHC transition");
        }
        if (auto status = cudaStreamWaitEvent(
                target.stream, state.dsv4_cross_device_ready);
            status != cudaSuccess) {
            target.dsv4_mhc_stage = 0U;
            static_cast<void>(cudaSetDevice(device));
            return cuda_error(status,
                              "wait for cross-device attention branch");
        }
        if (impl_->detailed_timing) {
            if (auto status = cudaEventRecord(
                    target.activation_start, target.stream);
                status != cudaSuccess) {
                target.dsv4_mhc_stage = 0U;
                static_cast<void>(cudaSetDevice(device));
                return cuda_error(
                    status, "record cross-device mHC upload start");
            }
        }
        if (auto status = cudaMemcpyAsync(
                target.dsv4_mhc_workspace->branch, staged_branch,
                static_cast<std::size_t>(branch_elements *
                                         sizeof(std::uint16_t)),
                cudaMemcpyHostToDevice, target.stream);
            status != cudaSuccess) {
            target.dsv4_mhc_stage = 0U;
            static_cast<void>(cudaSetDevice(device));
            return cuda_error(status,
                              "forward cross-device attention branch");
        }
        if (impl_->detailed_timing) {
            if (auto status = cudaEventRecord(
                    target.activation_uploaded, target.stream);
                status != cudaSuccess) {
                target.dsv4_mhc_stage = 0U;
                static_cast<void>(cudaSetDevice(device));
                return cuda_error(
                    status, "record cross-device mHC upload completion");
            }
        }
    }
    auto transition_stream = cross_transition ? target.stream : state.stream;
    auto& transition_state = cross_transition ? target : state;
    if (transition_mhc) {
        const auto current = target.dsv4_mhc_residual_index;
        const auto next = current ^ 1U;
        const auto* projection = static_cast<const float*>(
            request.mhc_transition->impl_->projection.impl_->weights);
        const auto* auxiliary = static_cast<const std::byte*>(
            request.mhc_transition->impl_->auxiliary.impl_->data);
        const auto* scale = reinterpret_cast<const float*>(auxiliary);
        const auto* base = reinterpret_cast<const float*>(
            auxiliary + 3U * sizeof(float));
        const auto* norm = reinterpret_cast<const __nv_bfloat16*>(
            auxiliary + kDsv4MhcAuxNormOffset);
        dsv4_mhc_fused_post_projection<<<
            dim3{kDsv4MhcMixes / kDsv4MhcProjectionTile, kDsv4MhcSplits},
            kDsv4MhcProjectionThreads, 0U, transition_stream>>>(
            target.dsv4_mhc_workspace->combination,
            target.dsv4_mhc_workspace->residual[current],
            target.dsv4_mhc_workspace->post,
            target.dsv4_mhc_workspace->branch, projection,
            target.dsv4_mhc_workspace->partial_projection,
            target.dsv4_mhc_workspace->partial_square_sum,
            target.dsv4_mhc_workspace->residual[next]);
        dsv4_mhc_mix<<<1U, 32U, 0U, transition_stream>>>(
            target.dsv4_mhc_workspace->partial_projection,
            target.dsv4_mhc_workspace->partial_square_sum,
            scale, base, kDsv4MhcSplits,
            target.dsv4_mhc_workspace->pre,
            target.dsv4_mhc_workspace->post,
            target.dsv4_mhc_workspace->combination);
        dsv4_mhc_weighted_norm<<<1U, kDsv4MhcWeightedNormThreads, 0U,
                                 transition_stream>>>(
            target.dsv4_mhc_workspace->residual[next],
            target.dsv4_mhc_workspace->pre, norm,
            target.dsv4_mhc_workspace->weighted,
            target.dsv4_mhc_workspace->layer_input);
    }
    if (project_router) {
        if (impl_->detailed_timing) {
            if (auto status = cudaEventRecord(
                    transition_state.router_started, transition_stream);
                status != cudaSuccess) {
                target.dsv4_mhc_stage = 0U;
                if (cross_transition) {
                    static_cast<void>(cudaSetDevice(device));
                }
                return cuda_error(
                    status, "record combined router projection start");
            }
        }
        constexpr unsigned int router_warps_per_block = threads / 32U;
        const auto router_blocks = static_cast<unsigned int>(
            (router_descriptor->rows + router_warps_per_block - 1U) /
            router_warps_per_block);
        bf16_input_matvec_kernel<<<router_blocks, threads, 0U,
                                    transition_stream>>>(
            device_router_logits, target.dsv4_mhc_workspace->layer_input,
            static_cast<const __nv_bfloat16*>(
                request.router->impl_->weights),
            router_descriptor->columns, router_descriptor->rows);
    }
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        if (transition_mhc) target.dsv4_mhc_stage = 0U;
        if (cross_transition) static_cast<void>(cudaSetDevice(device));
        return cuda_error(status, "launch attention-to-mHC kernels");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                transition_state.kernel_finished, transition_stream);
            status != cudaSuccess) {
            if (transition_mhc) target.dsv4_mhc_stage = 0U;
            if (cross_transition) static_cast<void>(cudaSetDevice(device));
            return cuda_error(status,
                              "record attention-to-mHC kernel completion");
        }
    }
    if (needs_branch_download && !cross_transition) {
        const auto* source_branch = device_encoded_branch;
        if (auto status = cudaMemcpyAsync(
                staged_branch, source_branch,
                static_cast<std::size_t>(branch_elements *
                                         sizeof(std::uint16_t)),
                cudaMemcpyDeviceToHost, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "download attention-to-mHC branch");
        }
    }
    if (diagnostic_query_bytes != 0U) {
        if (cross_transition) {
            target.dsv4_mhc_stage = 0U;
            static_cast<void>(cudaSetDevice(device));
            return {{"page query diagnostics do not support a cross-device transition"}};
        }
        if (auto status = cudaMemcpyAsync(
                staged_query, device_query,
                static_cast<std::size_t>(diagnostic_query_bytes),
                cudaMemcpyDeviceToHost, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "download DeepSeek page query diagnostic");
        }
    }
    if (transition_mhc) {
        if (auto status = cudaMemcpyAsync(
                staged_layer, target.dsv4_mhc_workspace->layer_input,
                static_cast<std::size_t>(transition_layer_bytes),
                cudaMemcpyDeviceToHost, transition_stream);
            status != cudaSuccess) {
            target.dsv4_mhc_stage = 0U;
            if (cross_transition) static_cast<void>(cudaSetDevice(device));
            return cuda_error(
                status, "download combined attention mHC layer input");
        }
    }
    if (project_router) {
        if (auto status = cudaMemcpyAsync(
                staged_router, device_router_logits,
                static_cast<std::size_t>(router_download_bytes),
                cudaMemcpyDeviceToHost, transition_stream);
            status != cudaSuccess) {
            target.dsv4_mhc_stage = 0U;
            if (cross_transition) static_cast<void>(cudaSetDevice(device));
            return cuda_error(
                status, "download combined FFN router logits");
        }
    }
    if (!cross_transition && attention_failure_bytes != 0U) {
        if (auto status = cudaMemcpyAsync(
                staged_failure, device_failure, sizeof(*device_failure),
                cudaMemcpyDeviceToHost, state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "download attention-to-mHC status");
        }
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                transition_state.activation_downloaded, transition_stream);
            status != cudaSuccess) {
            if (transition_mhc) target.dsv4_mhc_stage = 0U;
            if (cross_transition) static_cast<void>(cudaSetDevice(device));
            return cuda_error(status,
                              "record attention-to-mHC download completion");
        }
    }
    if (defer_host_moe_input) {
        if (cross_transition) {
            if (auto status = cudaSetDevice(device); status != cudaSuccess) {
                target.dsv4_mhc_stage = 0U;
                return cuda_error(
                    status, "restore deferred attention source device");
            }
        }
        target.dsv4_mhc_residual_index ^= 1U;
        target.dsv4_mhc_branch_ready = false;
        target.dsv4_host_moe_input_pending = true;
        target.dsv4_host_moe_router_logits = device_router_logits;
        target.dsv4_host_moe_device_failure = nullptr;
        target.dsv4_host_moe_host_failure =
            reinterpret_cast<const unsigned int*>(staged_failure);
        ++state.dsv4_deferred_attention_command_count;
        target.dsv4_deferred_attention_source_device = device;
        target.dsv4_deferred_attention_cross_transition = cross_transition;
        {
            std::scoped_lock lock(impl_->mutex);
            auto& stats = *std::find_if(
                impl_->stats.devices.begin(), impl_->stats.devices.end(),
                [device](const auto& value) { return value.device == device; });
            ++stats.dsv4_paged_attention_calls;
            stats.dsv4_paged_attention_kernel_launches += 19U;
            stats.dsv4_paged_attention_h2d_bytes += upload_bytes;
            stats.dsv4_paged_attention_d2h_bytes +=
                download_branch_bytes + attention_failure_bytes;
            stats.dsv4_paged_attention_page_bytes += page_bytes;
            stats.activation_h2d_bytes += upload_bytes;
            stats.activation_d2h_bytes += download_bytes;
            stats.workspace_allocation_calls += allocation_calls;
            stats.workspace_allocation_bytes += allocation_bytes;
            if (project_router) ++stats.matmul_calls;
            ++stats.dsv4_mhc_calls;
            ++stats.dsv4_mhc_transition_calls;
            stats.dsv4_mhc_kernel_launches += 3U;
            if (request.mhc_device != device) {
                auto& target_stats = *std::find_if(
                    impl_->stats.devices.begin(), impl_->stats.devices.end(),
                    [&](const auto& value) {
                        return value.device == request.mhc_device;
                    });
                target_stats.activation_h2d_bytes +=
                    branch_elements * sizeof(std::uint16_t);
            }
        }
        return result;
    }
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaStreamSynchronize(transition_stream);
        status != cudaSuccess) {
        if (transition_mhc) target.dsv4_mhc_stage = 0U;
        if (cross_transition) static_cast<void>(cudaSetDevice(device));
        return cuda_error(status, "synchronize attention-to-mHC completion");
    }
    const auto wait_nanoseconds = elapsed_nanoseconds_since(wait_started);
    const auto operation_nanoseconds = elapsed_nanoseconds_since(
        operation_started);
    if (cross_transition) {
        if (auto status = cudaSetDevice(device); status != cudaSuccess) {
            target.dsv4_mhc_stage = 0U;
            return cuda_error(
                status, "restore source after combined mHC transition");
        }
    }
    unsigned int numerical_error = 0U;
    std::memcpy(&numerical_error, staged_failure, sizeof(numerical_error));
    if (numerical_error != 0U) {
        if (transition_mhc) target.dsv4_mhc_stage = 0U;
        result.errors.emplace_back(
            "DeepSeek attention-to-mHC encountered corrupt page data");
        return result;
    }
    if (cross_device && !cross_transition) {
        if (auto status = cudaSetDevice(request.mhc_device);
            status != cudaSuccess) {
            return cuda_error(status,
                              "select mHC device for attention branch handoff");
        }
        if (page_request) {
            auto* target_encoded = reinterpret_cast<__nv_bfloat16*>(
                target.dsv4_attention_workspace);
            auto* target_slots = reinterpret_cast<std::uint32_t*>(
                target.dsv4_attention_workspace + cross_page_slot_offset);
            if (auto status = cudaMemcpyAsync(
                    target_encoded, staged_branch,
                    static_cast<std::size_t>(cross_page_branch_bytes),
                    cudaMemcpyHostToDevice, target.stream);
                status != cudaSuccess) {
                return cuda_error(
                    status, "upload cross-device attention page branches");
            }
            if (auto status = cudaMemcpyAsync(
                    target_slots, request.mhc_slots.data(),
                    static_cast<std::size_t>(rows) * sizeof(std::uint32_t),
                    cudaMemcpyHostToDevice, target.stream);
                status != cudaSuccess) {
                return cuda_error(
                    status, "upload cross-device attention page slots");
            }
            dsv4_scatter_encoded_mhc_page_branches<<<
                branch_blocks, threads, 0U, target.stream>>>(
                target_encoded, target.dsv4_mhc_slot_arena,
                target_slots, rows);
            if (auto status = cudaGetLastError(); status != cudaSuccess) {
                return cuda_error(
                    status, "scatter cross-device attention page branches");
            }
        } else if (auto status = cudaMemcpyAsync(
                       target.dsv4_mhc_workspace->branch, staged_branch,
                       static_cast<std::size_t>(branch_elements *
                                                sizeof(std::uint16_t)),
                       cudaMemcpyHostToDevice, target.stream);
                   status != cudaSuccess) {
            return cuda_error(status,
                              "upload cross-device attention branch to mHC");
        }
    }
    if (transition_mhc) {
        const auto* encoded = reinterpret_cast<const std::uint16_t*>(
            staged_layer);
        for (std::size_t index = 0U;
             index < request.mhc_layer_input.size(); ++index) {
            request.mhc_layer_input[index] = std::bit_cast<float>(
                static_cast<std::uint32_t>(encoded[index]) << 16U);
            if (!std::isfinite(request.mhc_layer_input[index])) {
                target.dsv4_mhc_stage = 0U;
                result.errors.emplace_back(
                    "combined attention mHC transition produced a non-finite value");
                return result;
            }
        }
        target.dsv4_mhc_residual_index ^= 1U;
        target.dsv4_mhc_branch_ready = false;
    } else if (page_request) {
        if (target.dsv4_mhc_saved_slots.size() <
            target.dsv4_mhc_slot_capacity) {
            target.dsv4_mhc_saved_slots.resize(
                target.dsv4_mhc_slot_capacity);
        }
        for (const auto slot : request.mhc_slots) {
            if (slot == target.dsv4_mhc_active_slot) {
                target.dsv4_mhc_branch_ready = true;
            } else {
                target.dsv4_mhc_saved_slots[slot].branch_ready = true;
            }
        }
    } else {
        target.dsv4_mhc_branch_ready = true;
    }
    if (project_router) {
        std::memcpy(request.router_logits.data(), staged_router,
                    router_download_bytes);
        if (!std::all_of(request.router_logits.begin(),
                         request.router_logits.end(),
                         [](float value) { return std::isfinite(value); })) {
            target.dsv4_mhc_stage = 0U;
            result.errors.emplace_back(
                "combined FFN router projection produced a non-finite value");
            return result;
        }
    }
    if (!diagnostic_branch.empty()) {
        const auto* encoded = reinterpret_cast<const std::uint16_t*>(
            staged_branch);
        for (std::size_t index = 0U; index < diagnostic_branch.size();
             ++index) {
            diagnostic_branch[index] = std::bit_cast<float>(
                static_cast<std::uint32_t>(encoded[index]) << 16U);
        }
    }
    if (!request.page_query_diagnostic.empty()) {
        const auto* encoded = reinterpret_cast<const std::uint16_t*>(
            staged_query);
        for (std::size_t index = 0U;
             index < request.page_query_diagnostic.size(); ++index) {
            request.page_query_diagnostic[index] = std::bit_cast<float>(
                static_cast<std::uint32_t>(encoded[index]) << 16U);
        }
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status,
                          "restore attention device after mHC handoff");
    }

    std::uint64_t h2d_nanoseconds = 0U;
    std::uint64_t kernel_nanoseconds = 0U;
    std::uint64_t attention_kernel_nanoseconds = 0U;
    std::uint64_t mhc_transition_nanoseconds = 0U;
    std::uint64_t d2h_nanoseconds = 0U;
    std::uint64_t mhc_timing_clamped_samples = 0U;
    if (impl_->detailed_timing) {
        float source_h2d_ms = 0.0F;
        if (auto status = cudaEventElapsedTime(
                &source_h2d_ms, state.activation_start,
                state.activation_uploaded); status != cudaSuccess) {
            return cuda_error(status, "measure attention-to-mHC upload");
        }
        if (cross_transition) {
            float attention_ms = 0.0F;
            float source_d2h_ms = 0.0F;
            if (cudaEventElapsedTime(
                    &attention_ms, state.activation_uploaded,
                    state.mhc_transition_finished) != cudaSuccess ||
                cudaEventElapsedTime(
                    &source_d2h_ms, state.mhc_transition_finished,
                    state.activation_downloaded) != cudaSuccess) {
                return {{"measure cross-device attention phase failed"}};
            }
            if (auto status = cudaSetDevice(request.mhc_device);
                status != cudaSuccess) {
                return cuda_error(
                    status, "select target to measure combined mHC");
            }
            float target_h2d_ms = 0.0F;
            float target_kernel_ms = 0.0F;
            float mhc_ms = 0.0F;
            float target_d2h_ms = 0.0F;
            if (cudaEventElapsedTime(
                    &target_h2d_ms, target.activation_start,
                    target.activation_uploaded) != cudaSuccess ||
                cudaEventElapsedTime(
                    &target_kernel_ms, target.activation_uploaded,
                    target.kernel_finished) != cudaSuccess ||
                cudaEventElapsedTime(
                    &mhc_ms, target.activation_uploaded,
                    project_router ? target.router_started
                                   : target.kernel_finished) != cudaSuccess ||
                cudaEventElapsedTime(
                    &target_d2h_ms, target.kernel_finished,
                    target.activation_downloaded) != cudaSuccess) {
                static_cast<void>(cudaSetDevice(device));
                return {{"measure cross-device combined mHC failed"}};
            }
            if (auto status = cudaSetDevice(device); status != cudaSuccess) {
                return cuda_error(
                    status, "restore source after measuring combined mHC");
            }
            h2d_nanoseconds = static_cast<std::uint64_t>(std::llround(
                static_cast<double>(source_h2d_ms + target_h2d_ms) *
                1.0e6));
            attention_kernel_nanoseconds =
                static_cast<std::uint64_t>(std::llround(
                    static_cast<double>(attention_ms) * 1.0e6));
            mhc_transition_nanoseconds =
                event_milliseconds_to_nanoseconds(
                    mhc_ms, mhc_timing_clamped_samples);
            kernel_nanoseconds = static_cast<std::uint64_t>(std::llround(
                static_cast<double>(attention_ms + target_kernel_ms) *
                1.0e6));
            d2h_nanoseconds = static_cast<std::uint64_t>(std::llround(
                static_cast<double>(source_d2h_ms + target_d2h_ms) *
                1.0e6));
        } else {
            float kernel_ms = 0.0F;
            float d2h_ms = 0.0F;
            if (auto status = cudaEventElapsedTime(
                    &kernel_ms, state.activation_uploaded,
                    state.kernel_finished); status != cudaSuccess) {
                return cuda_error(status,
                                  "measure attention-to-mHC kernels");
            }
            if (auto status = cudaEventElapsedTime(
                    &d2h_ms, state.kernel_finished,
                    state.activation_downloaded); status != cudaSuccess) {
                return cuda_error(status,
                                  "measure attention-to-mHC download");
            }
            h2d_nanoseconds = static_cast<std::uint64_t>(std::llround(
                static_cast<double>(source_h2d_ms) * 1.0e6));
            kernel_nanoseconds = static_cast<std::uint64_t>(std::llround(
                static_cast<double>(kernel_ms) * 1.0e6));
            attention_kernel_nanoseconds = kernel_nanoseconds;
            if (transition_mhc) {
                float attention_ms = 0.0F;
                float mhc_ms = 0.0F;
                if (cudaEventElapsedTime(
                        &attention_ms, state.activation_uploaded,
                        state.mhc_transition_finished) != cudaSuccess ||
                    cudaEventElapsedTime(
                        &mhc_ms, state.mhc_transition_finished,
                        project_router ? state.router_started
                                       : state.kernel_finished) != cudaSuccess) {
                    return {{"measure combined attention mHC transition failed"}};
                }
                attention_kernel_nanoseconds =
                    static_cast<std::uint64_t>(std::llround(
                        static_cast<double>(attention_ms) * 1.0e6));
                mhc_transition_nanoseconds =
                    event_milliseconds_to_nanoseconds(
                        mhc_ms, mhc_timing_clamped_samples);
            }
            d2h_nanoseconds = static_cast<std::uint64_t>(std::llround(
                static_cast<double>(d2h_ms) * 1.0e6));
        }
    }
    {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        ++stats.dsv4_paged_attention_calls;
        stats.dsv4_paged_attention_kernel_launches +=
            19U + (project_page_query ? 3U : 0U) +
            static_cast<std::uint64_t>(page_request && cross_device);
        stats.dsv4_paged_attention_h2d_bytes += upload_bytes;
        stats.dsv4_paged_attention_d2h_bytes +=
            download_branch_bytes + diagnostic_query_bytes +
            sizeof(unsigned int);
        stats.dsv4_paged_attention_page_bytes += page_bytes;
        stats.dsv4_paged_attention_h2d_nanoseconds += h2d_nanoseconds;
        stats.dsv4_paged_attention_kernel_nanoseconds +=
            attention_kernel_nanoseconds;
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
        if (project_page_query) ++stats.matmul_calls;
        if (project_router) ++stats.matmul_calls;
        if (transition_mhc) {
            ++stats.dsv4_mhc_calls;
            ++stats.dsv4_mhc_transition_calls;
            stats.dsv4_mhc_kernel_launches += 3U;
            stats.dsv4_mhc_d2h_bytes += transition_layer_bytes;
            stats.dsv4_mhc_kernel_nanoseconds +=
                mhc_transition_nanoseconds;
            stats.dsv4_mhc_d2h_nanoseconds += d2h_nanoseconds;
            stats.dsv4_mhc_nanoseconds +=
                mhc_transition_nanoseconds + d2h_nanoseconds;
            stats.dsv4_mhc_device_nanoseconds +=
                mhc_transition_nanoseconds + d2h_nanoseconds;
            stats.dsv4_mhc_timing_clamped_samples +=
                mhc_timing_clamped_samples;
        }
        record_synchronization(stats, SynchronizationSubsystem::Attention,
                               1U, wait_nanoseconds);
        if (request.mhc_device != device) {
            auto& target_stats = *std::find_if(
                impl_->stats.devices.begin(), impl_->stats.devices.end(),
                [&](const auto& value) {
                    return value.device == request.mhc_device;
                });
            target_stats.activation_h2d_bytes +=
                branch_elements * sizeof(std::uint16_t);
        }
    }
    return result;
}
