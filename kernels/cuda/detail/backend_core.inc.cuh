CudaWeight::CudaWeight() = default;
CudaWeight::~CudaWeight() = default;
CudaWeight::CudaWeight(CudaWeight&&) noexcept = default;
CudaWeight& CudaWeight::operator=(CudaWeight&&) noexcept = default;
bool CudaWeight::valid() const noexcept { return impl_ != nullptr && impl_->weights != nullptr; }
std::uint64_t CudaWeight::device_bytes() const noexcept { return impl_ ? impl_->bytes : 0U; }
int CudaWeight::device() const noexcept { return impl_ ? impl_->device : -1; }

CudaBuffer::CudaBuffer() = default;
CudaBuffer::~CudaBuffer() = default;
CudaBuffer::CudaBuffer(CudaBuffer&&) noexcept = default;
CudaBuffer& CudaBuffer::operator=(CudaBuffer&&) noexcept = default;
bool CudaBuffer::valid() const noexcept {
    return impl_ != nullptr && impl_->data != nullptr;
}
std::uint64_t CudaBuffer::device_bytes() const noexcept {
    return impl_ ? impl_->bytes : 0U;
}
int CudaBuffer::device() const noexcept { return impl_ ? impl_->device : -1; }

CudaDsv4MhcWeights::CudaDsv4MhcWeights() = default;
CudaDsv4MhcWeights::~CudaDsv4MhcWeights() = default;
CudaDsv4MhcWeights::CudaDsv4MhcWeights(CudaDsv4MhcWeights&&) noexcept = default;
CudaDsv4MhcWeights& CudaDsv4MhcWeights::operator=(
    CudaDsv4MhcWeights&&) noexcept = default;
bool CudaDsv4MhcWeights::valid() const noexcept {
    return impl_ != nullptr && impl_->projection.valid() &&
           impl_->auxiliary.valid();
}
std::uint64_t CudaDsv4MhcWeights::device_bytes() const noexcept {
    return impl_ == nullptr ? 0U : impl_->projection.device_bytes() +
                                  impl_->auxiliary.device_bytes();
}
int CudaDsv4MhcWeights::device() const noexcept {
    return impl_ == nullptr ? -1 : impl_->projection.device();
}

CudaBackend::CudaBackend() : impl_(std::make_unique<Impl>()) {}
CudaBackend::~CudaBackend() = default;
CudaBackend::CudaBackend(CudaBackend&&) noexcept = default;
CudaBackend& CudaBackend::operator=(CudaBackend&&) noexcept = default;
bool CudaBackend::compiled() noexcept { return true; }

std::vector<int> CudaBackend::available_devices() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return {};
    std::vector<int> result(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) result[static_cast<std::size_t>(index)] = index;
    return result;
}

ParseResult<CudaDeviceMemory> CudaBackend::device_memory(int device) {
    ParseResult<CudaDeviceMemory> result;
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        result.errors.emplace_back(std::string("select CUDA device: ") +
                                   cudaGetErrorString(status));
        return result;
    }
    std::size_t free_bytes = 0U;
    std::size_t total_bytes = 0U;
    if (auto status = cudaMemGetInfo(&free_bytes, &total_bytes); status != cudaSuccess) {
        result.errors.emplace_back(std::string("query CUDA memory: ") +
                                   cudaGetErrorString(status));
        return result;
    }
    result.value.free_bytes = free_bytes;
    result.value.total_bytes = total_bytes;
    return result;
}

std::uint64_t CudaBackend::weight_storage_bytes(
    std::uint64_t weight_bytes, std::uint64_t scale_bytes) noexcept {
    if (weight_bytes == 0U) return 0U;
    std::uint64_t scale_offset = 0U;
    if (!align_up(weight_bytes, kWeightPointerAlignment, scale_offset) ||
        scale_bytes > std::numeric_limits<std::uint64_t>::max() - scale_offset) {
        return 0U;
    }
    std::uint64_t result = 0U;
    if (!align_up(scale_offset + scale_bytes, kWeightArenaAlignment, result)) return 0U;
    return result;
}

ValidationResult CudaBackend::initialize(std::span<const int> devices,
                                         bool detailed_timing) {
    ValidationResult result;
    if (devices.empty()) {
        result.errors.emplace_back("CUDA backend requires at least one device");
        return result;
    }
    int count = 0;
    if (const auto status = cudaGetDeviceCount(&count); status != cudaSuccess) {
        return cuda_error(status, "enumerate CUDA devices");
    }
    impl_->detailed_timing = detailed_timing;
    for (const int device : devices) {
        if (device < 0 || device >= count || impl_->devices.contains(device)) {
            result.errors.emplace_back("CUDA device list contains an invalid or duplicate device");
            return result;
        }
        if (auto status = cudaSetDevice(device); status != cudaSuccess) {
            return cuda_error(status, "select CUDA device");
        }
        Impl::DeviceState state;
        cudaDeviceProp properties{};
        if (auto status = cudaGetDeviceProperties(&properties, device);
            status != cudaSuccess) {
            return cuda_error(status, "query CUDA device properties");
        }
        state.flash_attention_supported =
            (properties.major == 8 && properties.minor == 6) ||
            (properties.major == 12 && properties.minor == 0);
        state.dsv4_paged_attention_supported =
            properties.major == 8 && properties.minor == 6;
        state.dsv4_mhc_supported =
            properties.major == 8 && properties.minor == 6;
        state.lightning_index_supported = state.flash_attention_supported;
        state.dsv4_fp8_tensor_page_supported =
            properties.major == 8 && properties.minor == 6;
        if (auto status = cudaStreamCreateWithFlags(&state.stream, cudaStreamNonBlocking);
            status != cudaSuccess) {
            return cuda_error(status, "create CUDA stream");
        }
        if (auto status = cublasCreate(&state.cublas);
            status != CUBLAS_STATUS_SUCCESS) {
            return cublas_error(status, "create cuBLAS handle");
        }
        if (auto status = cublasSetStream(state.cublas, state.stream);
            status != CUBLAS_STATUS_SUCCESS) {
            return cublas_error(status, "set cuBLAS execution stream");
        }
        if (auto status = cublasSetMathMode(
                state.cublas, CUBLAS_TENSOR_OP_MATH);
            status != CUBLAS_STATUS_SUCCESS) {
            return cublas_error(status, "set cuBLAS tensor-op mode");
        }
        // Weight uploads get their own stream so the copy engine can run them
        // while the SMs are still on the previous command. On the execution
        // stream a demand expert transfer and the kernel that will read it are
        // strictly ordered, so a decode layer pays transfer plus compute in
        // series; the copy engine is idle for the compute half of that.
        // Ordering is restored explicitly by upload_ready, which every consumer
        // waits on before it reads a weight.
        if (auto status = cudaStreamCreateWithFlags(&state.upload_stream,
                                                    cudaStreamNonBlocking);
            status != cudaSuccess) {
            return cuda_error(status, "create CUDA upload stream");
        }
        if (auto status = cudaStreamCreateWithFlags(
                &state.moe_shared_stream, cudaStreamNonBlocking);
            status != cudaSuccess) {
            return cuda_error(status, "create DeepSeek shared-expert stream");
        }
        for (auto& stream : state.dsv4_attention_aux_streams) {
            if (auto status = cudaStreamCreateWithFlags(
                    &stream, cudaStreamNonBlocking);
                status != cudaSuccess) {
                return cuda_error(
                    status, "create DeepSeek attention auxiliary stream");
            }
        }
        if (auto status = cudaEventCreateWithFlags(&state.upload_ready,
                                                   cudaEventDisableTiming);
            status != cudaSuccess) {
            return cuda_error(status, "create CUDA upload event");
        }
        if (detailed_timing) {
            for (auto* event : {&state.activation_start,
                                &state.activation_uploaded,
                                &state.mhc_transition_finished,
                                &state.router_started,
                                &state.kernel_finished,
                                &state.activation_downloaded}) {
                if (auto status = cudaEventCreate(event); status != cudaSuccess) {
                    return cuda_error(status, "create CUDA timing event");
                }
            }
        }
        for (auto* event : {&state.moe_start, &state.moe_hidden_uploaded,
                            &state.moe_kernel_finished, &state.moe_download_started,
                            &state.moe_completed,
                            &state.moe_shared_input_finished,
                            &state.moe_shared_gate_up_finished,
                            &state.moe_shared_activation_finished,
                            &state.moe_shared_finished}) {
            if (auto status = cudaEventCreate(event); status != cudaSuccess) {
                return cuda_error(status, "create DeepSeek MoE event");
            }
        }
        if (auto status = cudaEventCreateWithFlags(
                &state.dsv4_cross_device_ready, cudaEventDisableTiming);
            status != cudaSuccess) {
            return cuda_error(status,
                              "create DeepSeek cross-device event");
        }
        if (auto status = cudaEventCreateWithFlags(
                &state.dsv4_attention_input_ready, cudaEventDisableTiming);
            status != cudaSuccess) {
            return cuda_error(
                status, "create DeepSeek attention input event");
        }
        for (auto& event : state.dsv4_attention_aux_finished) {
            if (auto status = cudaEventCreateWithFlags(
                    &event, cudaEventDisableTiming);
                status != cudaSuccess) {
                return cuda_error(
                    status, "create DeepSeek attention completion event");
            }
        }
        impl_->devices.emplace(device, state);
        CudaBackendStats::Device device_stats;
        device_stats.device = device;
        impl_->stats.devices.push_back(device_stats);
    }
    return result;
}

ValidationResult CudaBackend::register_host_memory(const void* base,
                                                   std::uint64_t bytes) {
    ValidationResult result;
    if (base == nullptr || bytes == 0U) {
        result.errors.emplace_back("host registration requires a non-empty region");
        return result;
    }
    // Register the arena as a single range. Chunking is faster (21.4 GB/s in
    // 4 GiB chunks against 2.38 GB/s for one 138 GiB call) but not correct
    // here: a weight read that straddles two separately registered ranges is
    // refused, which surfaced as "upload CUDA weights: invalid argument" on a
    // 4.46 MB projection crossing a chunk boundary. The caller must present a
    // writable mapping -- cudaHostRegisterReadOnly is unsupported on these
    // devices -- so try it first only for hosts where it is available.
    auto status = cudaHostRegister(const_cast<void*>(base),
                                   static_cast<std::size_t>(bytes),
                                   cudaHostRegisterPortable |
                                       cudaHostRegisterReadOnly);
    if (status != cudaSuccess) {
        static_cast<void>(cudaGetLastError());
        status = cudaHostRegister(const_cast<void*>(base),
                                  static_cast<std::size_t>(bytes),
                                  cudaHostRegisterPortable);
    }
    if (status != cudaSuccess) {
        // Registration is advisory, so the caller continues unpinned. Clear the
        // runtime's error state before returning or the next cudaGetLastError()
        // -- which belongs to an unrelated kernel launch -- reports this failure
        // as its own. That misattributed a failed cudaHostRegister to
        // "launch CUDA matmul: invalid argument".
        static_cast<void>(cudaGetLastError());
        return cuda_error(status, "register host memory");
    }
    return result;
}

void CudaBackend::unregister_host_memory(const void* base) noexcept {
    if (base == nullptr) return;
    static_cast<void>(cudaHostUnregister(const_cast<void*>(base)));
    static_cast<void>(cudaGetLastError());
    cudaGetLastError();
}

ValidationResult CudaBackend::reserve_weight_arena(int device,
                                                   std::uint64_t bytes) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "weight arena targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    if (state.weight_arena != nullptr) {
        result.errors.emplace_back("CUDA weight arena is already reserved");
        return result;
    }
    const auto stats = std::find_if(
        impl_->stats.devices.begin(), impl_->stats.devices.end(),
        [device](const auto& value) { return value.device == device; });
    if (stats->weight_upload_bytes != 0U) {
        result.errors.emplace_back(
            "CUDA weight arena must be reserved before the first weight upload");
        return result;
    }
    bytes -= bytes % kWeightArenaAlignment;
    if (bytes == 0U || bytes > std::numeric_limits<std::size_t>::max()) {
        result.errors.emplace_back("CUDA weight arena capacity is invalid");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for weight arena");
    }
    void* base = nullptr;
    if (auto status = cudaMalloc(&base, static_cast<std::size_t>(bytes));
        status != cudaSuccess) {
        return cuda_error(status, "reserve CUDA weight arena");
    }
    try {
        state.weight_arena = std::make_shared<WeightArena>(device, base, bytes);
    } catch (const std::bad_alloc&) {
        static_cast<void>(cudaFree(base));
        result.errors.emplace_back("allocate CUDA weight arena metadata");
        return result;
    }
    {
        std::scoped_lock lock(impl_->mutex);
        ++stats->weight_allocation_calls;
        stats->weight_allocation_bytes += bytes;
    }
    return result;
}

ValidationResult CudaBackend::upload(int device, const CudaWeightDescriptor& descriptor,
                                     std::span<const std::byte> weights,
                                     std::span<const std::byte> scales,
                                     CudaWeight& output,
                                     UploadCompletion completion,
                                     FragmentLayout prepack) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back("weight upload targets an uninitialized CUDA device");
        return result;
    }
    if (found->second.moe_in_flight) {
        result.errors.emplace_back(
            "weight upload cannot overlap an in-flight DeepSeek MoE command");
        return result;
    }
    if (descriptor.rows == 0U || descriptor.columns == 0U) {
        result.errors.emplace_back("CUDA weight dimensions must be positive");
        return result;
    }
    std::uint64_t expected_weights = 0U;
    std::uint64_t expected_scales = 0U;
    if (descriptor.encoding == CudaWeightEncoding::Plain) {
        const auto element_bytes = safetensors_dtype_bytes(descriptor.dtype);
        if ((descriptor.dtype != SafetensorsDtype::Bf16 &&
             descriptor.dtype != SafetensorsDtype::F16 &&
             descriptor.dtype != SafetensorsDtype::F32) ||
            !checked_bytes(descriptor.rows, descriptor.columns, element_bytes,
                           expected_weights) || !scales.empty()) {
            result.errors.emplace_back("invalid plain CUDA weight descriptor or payload");
            return result;
        }
    } else if (descriptor.encoding == CudaWeightEncoding::OffsetPackedInt4 ||
               descriptor.encoding == CudaWeightEncoding::OffsetPackedInt8) {
        const std::uint32_t bits = descriptor.encoding == CudaWeightEncoding::OffsetPackedInt4
                                       ? 4U
                                       : 8U;
        const auto expected_packed_columns =
            (descriptor.columns + (32U / bits) - 1U) / (32U / bits);
        if (descriptor.dtype != SafetensorsDtype::I32 ||
            descriptor.packed_columns != expected_packed_columns ||
            descriptor.scale_columns == 0U ||
            !checked_bytes(descriptor.rows, descriptor.packed_columns, 4U,
                           expected_weights) ||
            !checked_bytes(descriptor.rows, descriptor.scale_columns, 2U,
                           expected_scales)) {
            result.errors.emplace_back("invalid packed CUDA weight descriptor");
            return result;
        }
    } else if (descriptor.encoding == CudaWeightEncoding::Fp4E2m1Group32) {
        const auto expected_packed_columns = (descriptor.columns + 1U) / 2U;
        const auto expected_scale_columns = (descriptor.columns + 31U) / 32U;
        if (descriptor.dtype != SafetensorsDtype::I8 ||
            descriptor.packed_columns != expected_packed_columns ||
            descriptor.scale_columns != expected_scale_columns ||
            descriptor.group_size != 32U ||
            !checked_bytes(descriptor.rows, descriptor.packed_columns, 1U,
                           expected_weights) ||
            !checked_bytes(descriptor.rows, descriptor.scale_columns, 1U,
                           expected_scales)) {
            result.errors.emplace_back("invalid native FP4 CUDA weight descriptor");
            return result;
        }
    } else if (descriptor.encoding == CudaWeightEncoding::Fp4E2m1Tiled32) {
        // One blob, no separate scale payload: the four regions are contiguous
        // and the kernels index into them. `rows` is hidden, `columns` is this
        // shard's intermediate width.
        const bool shaped =
            descriptor.rows % 32U == 0U && descriptor.columns % 32U == 0U;
        if (!shaped || descriptor.dtype != SafetensorsDtype::I8 ||
            descriptor.group_size != 32U || descriptor.packed_columns != 0U ||
            descriptor.scale_columns != 0U || !scales.empty()) {
            result.errors.emplace_back(
                "invalid transformed FP4 expert shard descriptor");
            return result;
        }
        expected_weights = 2U * descriptor.columns * (descriptor.rows / 2U) +
                           2U * descriptor.columns * (descriptor.rows / 16U) +
                           descriptor.rows * (descriptor.columns / 2U) +
                           descriptor.rows * (descriptor.columns / 16U);
    } else if (descriptor.encoding == CudaWeightEncoding::Nvfp4Group16) {
        const auto expected_packed_columns = (descriptor.columns + 1U) / 2U;
        const auto expected_scale_columns =
            descriptor.group_size == 0U
                ? 0U
                : (descriptor.columns + descriptor.group_size - 1U) /
                      descriptor.group_size;
        if (descriptor.dtype != SafetensorsDtype::U8 ||
            descriptor.group_size == 0U || descriptor.columns % 2U != 0U ||
            descriptor.columns % descriptor.group_size != 0U ||
            descriptor.packed_columns != expected_packed_columns ||
            descriptor.scale_columns != expected_scale_columns ||
            !std::isfinite(descriptor.global_scale) ||
            descriptor.global_scale <= 0.0F ||
            !checked_bytes(descriptor.rows, descriptor.packed_columns, 1U,
                           expected_weights) ||
            !checked_bytes(descriptor.rows, descriptor.scale_columns, 1U,
                           expected_scales)) {
            result.errors.emplace_back("invalid NVFP4 CUDA weight descriptor");
            return result;
        }
    } else if (descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128) {
        const auto expected_scale_columns = (descriptor.columns + 127U) / 128U;
        const auto expected_scale_rows = (descriptor.rows + 127U) / 128U;
        if (descriptor.dtype != SafetensorsDtype::F8E4M3 ||
            descriptor.packed_columns != descriptor.columns ||
            descriptor.scale_columns != expected_scale_columns ||
            descriptor.group_size != 128U ||
            !checked_bytes(descriptor.rows, descriptor.columns, 1U,
                           expected_weights) ||
            !checked_bytes(expected_scale_rows, descriptor.scale_columns, 1U,
                           expected_scales)) {
            result.errors.emplace_back("invalid native FP8 CUDA weight descriptor");
            return result;
        }
    } else {
        result.errors.emplace_back("unsupported CUDA weight encoding");
        return result;
    }
    if (weights.size() != expected_weights || scales.size() != expected_scales) {
        result.errors.emplace_back("CUDA weight payload byte count is invalid");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for upload");
    }
    auto target = std::make_shared<CudaWeight::Impl>();
    target->descriptor = descriptor;
    target->device = device;
    const auto payload_bytes = expected_weights + expected_scales;
    auto& state = found->second;
    std::uint64_t allocation_calls = 0U;
    std::uint64_t allocation_nanoseconds = 0U;
    std::uint64_t copy_nanoseconds = 0U;
    const auto allocation_started = std::chrono::steady_clock::now();
    if (state.weight_arena != nullptr) {
        target->bytes = weight_storage_bytes(expected_weights, expected_scales);
        WeightArena::Allocation allocation;
        if (target->bytes == 0U ||
            !state.weight_arena->allocate(target->bytes, allocation)) {
            const auto report = state.weight_arena->occupancy();
            const auto mib = [](std::uint64_t bytes) {
                return static_cast<double>(bytes) / (1024.0 * 1024.0);
            };
            std::ostringstream detail;
            detail.setf(std::ios::fixed);
            detail.precision(1);
            detail << "CUDA weight arena is exhausted; refusing per-weight "
                      "allocation fallback (device " << device << ", wanted "
                   << mib(target->bytes) << " MiB, free " << mib(report.free_bytes)
                   << " MiB of " << mib(report.capacity) << " MiB in "
                   << report.free_blocks << " blocks, largest "
                   << mib(report.largest_free) << " MiB)";
            result.errors.emplace_back(detail.str());
            return result;
        }
        target->arena = state.weight_arena;
        target->arena_offset = allocation.offset;
        target->weights = allocation.address;
        if (expected_scales != 0U) {
            std::uint64_t scale_offset = 0U;
            static_cast<void>(align_up(expected_weights, kWeightPointerAlignment,
                                       scale_offset));
            target->scales = static_cast<std::byte*>(allocation.address) + scale_offset;
        }
    } else {
        target->bytes = payload_bytes;
        if (auto status = cudaMalloc(
                &target->weights, static_cast<std::size_t>(expected_weights));
            status != cudaSuccess) {
            return cuda_error(status, "allocate CUDA weights");
        }
        ++allocation_calls;
    }
    allocation_nanoseconds += elapsed_nanoseconds_since(allocation_started);
    // A deferred upload runs on the copy stream so it overlaps whatever the
    // execution stream is still doing; the execution stream is made to wait on
    // upload_ready before anything reads the weight. A synchronous upload keeps
    // the execution stream, because its caller's host payload dies at return
    // and the wait below is what keeps it alive long enough.
    const bool deferred = completion == UploadCompletion::Deferred;
    auto* const upload_stream = deferred ? state.upload_stream : state.stream;
    const auto upload_error = [&state, &target, upload_stream](
        cudaError_t status, const char* operation) {
        if (cudaStreamSynchronize(upload_stream) != cudaSuccess) {
            state.quarantined_weights.push_back(std::move(target));
        }
        return cuda_error(status, operation);
    };
    auto copy_started = std::chrono::steady_clock::now();
    if (auto status = cudaMemcpyAsync(target->weights, weights.data(), weights.size(),
                                      cudaMemcpyHostToDevice, upload_stream);
        status != cudaSuccess) {
        return upload_error(status, "upload CUDA weights");
    }
    copy_nanoseconds += elapsed_nanoseconds_since(copy_started);
    if (expected_scales != 0U) {
        if (state.weight_arena == nullptr) {
            const auto scale_allocation_started = std::chrono::steady_clock::now();
            if (auto status = cudaMalloc(
                    &target->scales, static_cast<std::size_t>(expected_scales));
                status != cudaSuccess) {
                return cuda_error(status, "allocate CUDA scales");
            }
            ++allocation_calls;
            allocation_nanoseconds +=
                elapsed_nanoseconds_since(scale_allocation_started);
        }
        copy_started = std::chrono::steady_clock::now();
        if (auto status = cudaMemcpyAsync(target->scales, scales.data(), scales.size(),
                                          cudaMemcpyHostToDevice, upload_stream);
            status != cudaSuccess) {
            return upload_error(status, "upload CUDA scales");
        }
        copy_nanoseconds += elapsed_nanoseconds_since(copy_started);
    }
    // Fragment prepack, stream-ordered behind the copies that just landed. This
    // is a device-side permutation of the staged bytes: for a streaming MoE it
    // costs one read and one write of what was staged, measured at 0.509
    // ms/token against Laguna's 65.05 ms staging term (experiment 0168), so it
    // does not need to move off the staging path.
    if (prepack == FragmentLayout::Prepack && regfed_matmul_enabled()) {
        const auto scratch_bytes = fragment_prepack_scratch_bytes(descriptor);
        if (scratch_bytes != 0U) {
            bool ready = true;
            {
                std::scoped_lock lock(impl_->mutex);
                ready = regfed_grow(state.upload_prepack_scratch,
                                    state.upload_prepack_scratch_bytes,
                                    scratch_bytes, false,
                                    upload_stream) == cudaSuccess;
            }
            if (ready) {
                if (auto status = launch_fragment_prepack(
                        descriptor, target->weights, target->scales,
                        state.upload_prepack_scratch, upload_stream);
                    status != cudaSuccess) {
                    return upload_error(status, "prepack weight into fragment order");
                }
                target->fragment_prepacked = true;
            }
        }
    }
    std::uint64_t wait_nanoseconds = 0U;
    std::uint64_t synchronizations = 0U;
    if (deferred) {
        // The copies stay in flight so the next device's can start immediately,
        // and so this device's copy engine runs them against whatever the
        // execution stream is doing. Ordering is re-established by
        // synchronize_uploads(), which the caller owes before any consumer.
        state.pending_uploads = true;
    } else {
        const auto wait_started = std::chrono::steady_clock::now();
        if (auto status = cudaStreamSynchronize(state.stream); status != cudaSuccess) {
            state.quarantined_weights.push_back(std::move(target));
            return cuda_error(status, "synchronize CUDA weight upload");
        }
        wait_nanoseconds = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - wait_started).count());
        synchronizations = 1U;
    }
    {
        std::scoped_lock lock(impl_->mutex);
        auto& device_stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        device_stats.weight_upload_bytes += payload_bytes;
        device_stats.weight_allocation_calls += allocation_calls;
        if (allocation_calls != 0U) {
            device_stats.weight_allocation_bytes += payload_bytes;
        }
        record_synchronization(device_stats, SynchronizationSubsystem::Weight,
                               synchronizations, wait_nanoseconds);
        device_stats.upload_wait_nanoseconds += wait_nanoseconds;
        device_stats.weight_allocation_nanoseconds += allocation_nanoseconds;
        device_stats.weight_copy_nanoseconds += copy_nanoseconds;
    }
    output.impl_ = std::move(target);
    return result;
}

ValidationResult CudaBackend::synchronize_uploads(int device) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "upload synchronization targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    if (!state.pending_uploads) return result;
    state.pending_uploads = false;
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for upload synchronization");
    }
    // The ordering the consumer needs is "the copies have landed before the
    // kernel reads them", which is a device-side dependency, not a host one.
    // Expressing it as an event the execution stream waits on lets the host
    // return immediately and enqueue the command, so the copy engine finishes
    // the transfer while the SMs start on work that does not depend on it.
    // Blocking the host here instead cost a measured 64.5 ms of a 235 ms
    // decode step, with both engines idle for most of it.
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaEventRecord(state.upload_ready, state.upload_stream);
        status != cudaSuccess) {
        return cuda_error(status, "record deferred CUDA weight upload");
    }
    if (auto status = cudaStreamWaitEvent(state.stream, state.upload_ready, 0U);
        status != cudaSuccess) {
        return cuda_error(status, "order CUDA execution behind weight uploads");
    }
    state.upload_ordered = true;
    const auto wait_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wait_started).count());
    {
        std::scoped_lock lock(impl_->mutex);
        auto& device_stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        record_synchronization(device_stats, SynchronizationSubsystem::Weight,
                               1U, wait_nanoseconds);
        device_stats.upload_wait_nanoseconds += wait_nanoseconds;
    }
    return result;
}

ValidationResult CudaBackend::upload_buffer(
    int device, std::span<const std::byte> bytes, CudaBuffer& output) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "buffer upload targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    if (state.moe_in_flight) {
        result.errors.emplace_back(
            "buffer upload cannot overlap an in-flight DeepSeek MoE command");
        return result;
    }
    if (bytes.empty()) {
        result.errors.emplace_back("CUDA buffer upload payload is empty");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for buffer upload");
    }
    auto target = std::make_shared<CudaBuffer::Impl>();
    target->bytes = bytes.size();
    target->device = device;
    if (auto status = cudaMalloc(&target->data, bytes.size());
        status != cudaSuccess) {
        return cuda_error(status, "allocate CUDA buffer");
    }
    if (auto status = cudaMemcpyAsync(target->data, bytes.data(), bytes.size(),
                                      cudaMemcpyHostToDevice, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "upload CUDA buffer");
    }
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaStreamSynchronize(state.stream); status != cudaSuccess) {
        state.quarantined_buffers.push_back(std::move(target));
        return cuda_error(status, "synchronize CUDA buffer upload");
    }
    const auto wait_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wait_started).count());
    {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        stats.activation_h2d_bytes += bytes.size();
        stats.activation_h2d_nanoseconds += wait_nanoseconds;
        ++stats.workspace_allocation_calls;
        stats.workspace_allocation_bytes += bytes.size();
        record_synchronization(stats, SynchronizationSubsystem::Other, 1U,
                               wait_nanoseconds);
    }
    output.impl_ = std::move(target);
    return result;
}

ValidationResult CudaBackend::download_buffer(
    const CudaBuffer& buffer, std::uint64_t offset,
    std::span<std::byte> output) {
    ValidationResult result;
    if (!buffer.impl_ || output.empty()) {
        result.errors.emplace_back("CUDA buffer download is invalid");
        return result;
    }
    if (offset > buffer.impl_->bytes ||
        output.size() > buffer.impl_->bytes - offset) {
        result.errors.emplace_back("CUDA buffer download is out of bounds");
        return result;
    }
    const auto found = impl_->devices.find(buffer.impl_->device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "buffer download targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    if (state.moe_in_flight) {
        result.errors.emplace_back(
            "buffer download cannot overlap an in-flight DeepSeek MoE command");
        return result;
    }
    if (auto status = cudaSetDevice(buffer.impl_->device);
        status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for buffer download");
    }
    const auto* source = static_cast<const std::byte*>(buffer.impl_->data);
    if (auto status = cudaMemcpyAsync(
            output.data(), source + offset, output.size(),
            cudaMemcpyDeviceToHost, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "download CUDA buffer");
    }
    if (auto status = cudaStreamSynchronize(state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "synchronize CUDA buffer download");
    }
    return result;
}

ValidationResult CudaBackend::update_buffer(
    const CudaBuffer& buffer, std::span<const CudaBufferPatch> patches) {
    ValidationResult result;
    if (!buffer.impl_ || patches.empty()) {
        result.errors.emplace_back("CUDA buffer update is invalid");
        return result;
    }
    const auto found = impl_->devices.find(buffer.impl_->device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "buffer update targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    if (state.moe_in_flight) {
        result.errors.emplace_back(
            "buffer update cannot overlap an in-flight DeepSeek MoE command");
        return result;
    }
    std::uint64_t total_bytes = 0U;
    for (const auto& patch : patches) {
        if (patch.bytes.empty() || patch.offset > buffer.impl_->bytes ||
            patch.bytes.size() > buffer.impl_->bytes - patch.offset ||
            total_bytes > std::numeric_limits<std::uint64_t>::max() -
                              patch.bytes.size()) {
            result.errors.emplace_back("CUDA buffer patch is out of bounds");
            return result;
        }
        total_bytes += patch.bytes.size();
    }
    if (auto status = cudaSetDevice(buffer.impl_->device);
        status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for buffer update");
    }
    auto* destination = static_cast<std::byte*>(buffer.impl_->data);
    for (const auto& patch : patches) {
        if (auto status = cudaMemcpyAsync(
                destination + patch.offset, patch.bytes.data(),
                patch.bytes.size(), cudaMemcpyHostToDevice, state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "update CUDA buffer");
        }
    }
    {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [&buffer](const auto& value) {
                return value.device == buffer.impl_->device;
            });
        stats.activation_h2d_bytes += total_bytes;
    }
    return result;
}

ValidationResult CudaBackend::allocate_buffer(
    int device, std::uint64_t bytes, CudaBuffer& output) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end() || bytes == 0U) {
        result.errors.emplace_back("CUDA buffer allocation is invalid");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for buffer allocation");
    }
    auto target = std::make_shared<CudaBuffer::Impl>();
    target->bytes = bytes;
    target->device = device;
    if (auto status = cudaMalloc(&target->data, static_cast<std::size_t>(bytes));
        status != cudaSuccess) {
        return cuda_error(status, "allocate CUDA buffer");
    }
    output.impl_ = std::move(target);
    {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        ++stats.workspace_allocation_calls;
        stats.workspace_allocation_bytes += bytes;
    }
    return result;
}
