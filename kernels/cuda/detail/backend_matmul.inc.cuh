namespace {
std::atomic<std::uint64_t>
    g_route_census[static_cast<std::size_t>(CudaMatmulRoute::Count)]{};
}  // namespace

bool register_fed_matmul_enabled() noexcept { return regfed_matmul_enabled(); }

void set_register_fed_matmul(bool enabled) noexcept {
    g_regfed_matmul_enabled.store(enabled ? 1 : 0, std::memory_order_relaxed);
}

void record_cuda_matmul_route(CudaMatmulRoute route) noexcept {
    g_route_census[static_cast<std::size_t>(route)].fetch_add(
        1U, std::memory_order_relaxed);
}

bool CudaBackend::fragment_prepacked(const CudaWeight& weight) noexcept {
    return weight.impl_ != nullptr && weight.impl_->fragment_prepacked;
}

bool CudaBackend::marlin_prepacked(const CudaWeight& weight) noexcept {
    return weight.impl_ != nullptr && weight.impl_->marlin_prepacked;
}

ValidationResult CudaBackend::prepack_marlin(int device,
                                             const CudaWeight& weight) {
    ValidationResult result;
    if (!weight.valid() || weight.device() != device) {
        result.errors.emplace_back("Marlin prepack received an invalid weight");
        return result;
    }
    if (weight.impl_->fragment_prepacked) {
        result.errors.emplace_back(
            "Marlin prepack refuses a register-fed fragment layout");
        return result;
    }
    if (weight.impl_->marlin_prepacked) return result;
    const auto& descriptor = weight.impl_->descriptor;
    const auto scratch_bytes = marlin_prepack_scratch_bytes(descriptor);
    if (scratch_bytes == 0U) {
        result.errors.emplace_back(
            "Marlin prepack has no layout for this weight encoding and shape");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select device for Marlin prepack");
    }
    void* scratch = nullptr;
    if (auto status = cudaMalloc(&scratch, scratch_bytes);
        status != cudaSuccess) {
        return cuda_error(status, "allocate Marlin prepack scratch");
    }
    const auto release = [&](cudaError_t status, const char* what) {
        static_cast<void>(cudaFree(scratch));
        return cuda_error(status, what);
    };
    if (auto status = launch_marlin_prepack(
            descriptor, weight.impl_->weights, weight.impl_->scales, scratch,
            nullptr);
        status != cudaSuccess) {
        return release(status, "launch Marlin prepack");
    }
    if (auto status = cudaDeviceSynchronize(); status != cudaSuccess) {
        return release(status, "finish Marlin prepack");
    }
    static_cast<void>(cudaFree(scratch));
    weight.impl_->marlin_prepacked = true;
    return result;
}

ValidationResult CudaBackend::prepack_fragment(int device,
                                               const CudaWeight& weight) {
    ValidationResult result;
    if (!weight.valid() || weight.device() != device) {
        result.errors.emplace_back("fragment prepack received an invalid weight");
        return result;
    }
    if (weight.impl_->marlin_prepacked) {
        result.errors.emplace_back(
            "fragment prepack refuses an existing Marlin layout");
        return result;
    }
    const auto& descriptor = weight.impl_->descriptor;
    const auto found = impl_->devices.find(device);
    if (descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128F32 &&
        (found == impl_->devices.end() ||
         !found->second.fp8_f32_register_fed_supported)) {
        result.errors.emplace_back(
            "F32-scaled FP8 fragment prepack requires BF16 tensor cores");
        return result;
    }
    const auto scratch_bytes = fragment_prepack_scratch_bytes(descriptor);
    if (scratch_bytes == 0U) {
        result.errors.emplace_back(
            "fragment prepack has no layout for this weight encoding and shape");
        return result;
    }
    if (weight.impl_->fragment_prepacked) return result;
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select device for fragment prepack");
    }
    void* scratch = nullptr;
    if (auto status =
            cudaMalloc(&scratch, static_cast<std::size_t>(scratch_bytes));
        status != cudaSuccess) {
        return cuda_error(status, "allocate fragment prepack scratch");
    }
    const auto release = [&](cudaError_t status, const char* what) {
        static_cast<void>(cudaFree(scratch));
        return cuda_error(status, what);
    };
    if (auto status = launch_fragment_prepack(
            descriptor, weight.impl_->weights, weight.impl_->scales, scratch,
            nullptr);
        status != cudaSuccess) {
        return release(status, "launch fragment prepack");
    }
    if (auto status = cudaDeviceSynchronize(); status != cudaSuccess) {
        return release(status, "finish fragment prepack");
    }
    static_cast<void>(cudaFree(scratch));
    weight.impl_->fragment_prepacked = true;
    return result;
}

CudaMatmulRouteCensus cuda_matmul_route_census() noexcept {
    CudaMatmulRouteCensus out;
    for (std::size_t i = 0; i < out.counts.size(); ++i)
        out.counts[i] = g_route_census[i].load(std::memory_order_relaxed);
    return out;
}

void reset_cuda_matmul_route_census() noexcept {
    for (auto& c : g_route_census) c.store(0U, std::memory_order_relaxed);
}

const char* cuda_matmul_route_name(CudaMatmulRoute route) noexcept {
    switch (route) {
        case CudaMatmulRoute::PlainBf16Matvec: return "plain_bf16_matvec";
        case CudaMatmulRoute::PlainGeneric: return "plain_generic";
        case CudaMatmulRoute::PackedInt8Group32: return "packed_int8_group32";
        case CudaMatmulRoute::PackedOffsetInt: return "packed_offset_int";
        case CudaMatmulRoute::Nvfp4Group16: return "nvfp4_group16";
        case CudaMatmulRoute::Fp8TensorPage: return "fp8_tensor_page";
        case CudaMatmulRoute::Fp8F32TensorPage:
            return "fp8_f32_tensor_page";
        case CudaMatmulRoute::Fp8E4m3Block128: return "fp8_e4m3_block128";
        case CudaMatmulRoute::Fp8E4m3Block128F32:
            return "fp8_e4m3_block128_f32";
        case CudaMatmulRoute::Fp4E2m1Group32: return "fp4_e2m1_group32";
        case CudaMatmulRoute::Fp8RegisterFed: return "fp8_register_fed";
        case CudaMatmulRoute::Fp8F32RegisterFed:
            return "fp8_f32_register_fed";
        case CudaMatmulRoute::Fp4RegisterFed: return "fp4_register_fed";
        case CudaMatmulRoute::GemmaMarlin: return "gemma_marlin";
        case CudaMatmulRoute::MoePlainBf16: return "moe_plain_bf16";
        case CudaMatmulRoute::MoeFp8E4m3Block128F32:
            return "moe_fp8_e4m3_block128_f32";
        case CudaMatmulRoute::MoeFp8F32RegisterFed:
            return "moe_fp8_f32_register_fed";
        case CudaMatmulRoute::MoeNvfp4Group16: return "moe_nvfp4_group16";
        case CudaMatmulRoute::MoeFp4E2m1Group32:
            return "moe_fp4_e2m1_group32";
        case CudaMatmulRoute::MoePackedInt4: return "moe_packed_int4";
        case CudaMatmulRoute::MoeFp4RegisterFed: return "moe_fp4_register_fed";
        case CudaMatmulRoute::Dsv4MoeRoutedFp4: return "dsv4_moe_routed_fp4";
        case CudaMatmulRoute::Dsv4MoeSharedFp8: return "dsv4_moe_shared_fp8";
        case CudaMatmulRoute::Dsv4MoeSharedFp8RegisterFed:
            return "dsv4_moe_shared_fp8_register_fed";
        case CudaMatmulRoute::Dsv4MoeTierFp4: return "dsv4_moe_tier_fp4";
        case CudaMatmulRoute::Unsupported: return "unsupported";
        default: return "invalid";
    }
}

ValidationResult CudaBackend::matmul_impl(
    const CudaWeight& weight, std::span<const float> input,
    std::uint32_t rows, std::uint32_t groups,
    std::uint64_t rows_per_group, std::span<float> output, float softcap,
    bool round_output, CudaMatmulProfile* profile,
    bool dsv4_fp8_tensor_page, const std::byte* batch_input,
    std::byte* batch_output, bool defer_completion) {
    ValidationResult result;
    if (profile != nullptr) *profile = {};
    if (!weight.valid()) {
        result.errors.emplace_back("CUDA matmul received an invalid weight");
        return result;
    }
    const auto& descriptor = weight.impl_->descriptor;
    const bool regular_shape = groups == 0U &&
        input.size() == descriptor.columns * rows &&
        output.size() == descriptor.rows * rows;
    const bool grouped_shape = groups != 0U && rows_per_group != 0U &&
        descriptor.rows == static_cast<std::uint64_t>(groups) * rows_per_group &&
        input.size() == descriptor.columns * groups * rows &&
        output.size() == descriptor.rows * rows;
    if (rows == 0U || (!regular_shape && !grouped_shape) ||
        !std::isfinite(softcap) || softcap < 0.0F ||
        (softcap != 0.0F && (rows != 1U || groups != 0U)) ||
        (defer_completion &&
         (batch_input == nullptr || batch_output == nullptr ||
          profile != nullptr || impl_->detailed_timing))) {
        result.errors.emplace_back("CUDA matmul activation shapes are incompatible");
        return result;
    }
    const auto issue_started = std::chrono::steady_clock::now();
    auto& state = impl_->devices.at(weight.impl_->device);
    if (state.moe_in_flight) {
        result.errors.emplace_back(
            "CUDA matmul cannot overlap an in-flight DeepSeek MoE command");
        return result;
    }
    if (auto status = cudaSetDevice(weight.impl_->device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for matmul");
    }
    const auto input_bytes = static_cast<std::uint64_t>(input.size_bytes());
    const auto output_bytes = static_cast<std::uint64_t>(output.size_bytes());
    // MIX-2 register-fed dispatch. The skinny kernels own M <= 16, which is the
    // whole decode regime; wider M keeps the tensor-page and scalar routes,
    // where the weight read is already amortized across many activation rows.
    //
    // The prepack is lazy rather than done at load. A weight is permuted the
    // first time a skinny call reaches it, so no architecture adapter has to
    // opt in and no large-M caller ever pays for a layout it does not want. It
    // is one-way: once fragment order has replaced the canonical layout, a
    // later wide call on that same weight has to chunk through the skinny
    // kernel, which is recorded as its own census route rather than hidden.
    const bool regfed_encoding =
        descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128 ||
        descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128F32 ||
        descriptor.encoding == CudaWeightEncoding::Fp4E2m1Group32;
    const bool regfed_shape =
        regfed_encoding && groups == 0U && softcap == 0.0F &&
        (descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128 ||
         descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128F32
             ? regfed_fp8_shape_admissible(descriptor.rows, descriptor.columns)
             : regfed_fp4_shape_admissible(descriptor.rows, descriptor.columns));
    // The register-fed route requires a weight already permuted by an explicit
    // prepack_fragment call. matmul_impl must NOT decide this for itself: the
    // layout is a property of the weight, and matmul_impl cannot see the
    // weight's other consumers. Deciding it here corrupted the DeepSeek V4
    // attention output projection, which matmul_impl touches 129 times a run
    // and the attention path then reads canonically.
    const bool f32_fragment_page_candidate =
        descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128F32 &&
        weight.impl_->fragment_prepacked && dsv4_fp8_tensor_page && rows > 16U &&
        state.fp8_f32_tensor_page_supported && groups == 0U && softcap == 0.0F &&
        descriptor.columns % kDsv4Fp8TensorBlockK == 0U &&
        descriptor.rows % 128U == 0U &&
        descriptor.columns <= std::numeric_limits<std::uint32_t>::max() &&
        descriptor.rows <= std::numeric_limits<std::uint32_t>::max();
    const bool regfed = regfed_shape && regfed_matmul_enabled() &&
                        weight.impl_->fragment_prepacked &&
                        (descriptor.encoding !=
                             CudaWeightEncoding::Fp8E4m3Block128F32 ||
                         state.fp8_f32_register_fed_supported) &&
                        !f32_fragment_page_candidate;
    const bool f32_regfed = regfed && descriptor.encoding ==
        CudaWeightEncoding::Fp8E4m3Block128F32;
    const bool marlin = weight.impl_->marlin_prepacked && groups == 0U &&
                        rows <= 128U &&
                        descriptor.encoding ==
                            CudaWeightEncoding::Fp4E2m1Group32 &&
                        descriptor.rows % 64U == 0U &&
                        descriptor.columns % 256U == 0U;
    // No hidden fallback. Fragment order replaces the canonical layout, so a
    // permuted weight reaching a canonical kernel does not degrade -- it
    // decodes a permutation as if it were weights. Refuse instead.
    if (weight.impl_->fragment_prepacked && !regfed &&
        !f32_fragment_page_candidate) {
        result.errors.emplace_back(
            "CUDA matmul received a fragment-prepacked weight but has no "
            "register-fed route for this call; refusing to read fragment order "
            "as canonical layout");
        return result;
    }
    if (weight.impl_->marlin_prepacked && !marlin) {
        result.errors.emplace_back(
            "CUDA matmul received a Marlin-prepacked weight but this call has "
            "no admissible Marlin route");
        return result;
    }
    const bool dsv4_tensor_page =
        dsv4_fp8_tensor_page && !regfed &&
        state.dsv4_fp8_tensor_page_supported && rows > 1U && groups == 0U &&
        descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128 &&
        descriptor.columns % kDsv4Fp8TensorBlockK == 0U &&
        descriptor.rows % kDsv4Fp8TensorBlockN == 0U &&
        descriptor.columns <= std::numeric_limits<std::uint32_t>::max() &&
        descriptor.rows <= std::numeric_limits<std::uint32_t>::max() &&
        (!weight.impl_->fragment_prepacked || f32_fragment_page_candidate);
    const bool f32_tensor_page =
        dsv4_fp8_tensor_page && !regfed &&
        state.fp8_f32_tensor_page_supported && rows > 1U && groups == 0U &&
        descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128F32 &&
        descriptor.columns % kDsv4Fp8TensorBlockK == 0U &&
        descriptor.rows % 128U == 0U &&
        descriptor.columns <= std::numeric_limits<std::uint32_t>::max() &&
        descriptor.rows <= std::numeric_limits<std::uint32_t>::max() &&
        (!weight.impl_->fragment_prepacked || f32_fragment_page_candidate);
    const bool tensor_page = dsv4_tensor_page || f32_tensor_page;
    const auto input_scale_bytes = tensor_page
        ? static_cast<std::uint64_t>(rows) * descriptor.scale_columns *
              (f32_tensor_page ? sizeof(float) : sizeof(unsigned char))
        : 0U;
    const auto compact_input_bytes = tensor_page
        ? static_cast<std::uint64_t>(input.size()) + input_scale_bytes
        : input_bytes;
    const auto padded_rows = tensor_page
        ? (static_cast<std::uint64_t>(rows) + kDsv4Fp8TensorBlockM - 1U) /
              kDsv4Fp8TensorBlockM * kDsv4Fp8TensorBlockM
        : static_cast<std::uint64_t>(rows);
    if (tensor_page &&
        descriptor.rows > std::numeric_limits<std::uint64_t>::max() /
                              padded_rows / sizeof(float)) {
        result.errors.emplace_back(
            "FP8 tensor page output workspace overflows");
        return result;
    }
    const auto tensor_output_bytes = tensor_page
        ? padded_rows * descriptor.rows * sizeof(float)
        : output_bytes;
    const auto required_input_bytes = compact_input_bytes;
    // The original FP32 activation is uploaded into the eventual result
    // buffer, compacted into state.input, and then overwritten by the tensor
    // result. This keeps one compact encoded activation plus one reused output
    // allocation, never the incumbent four-byte encoded activation beside it.
    const auto marlin_output_bytes = marlin && rows > 1U
        ? 128U * descriptor.rows * sizeof(float)
        : output_bytes;
    const auto required_output_bytes = tensor_page || f32_regfed
        ? std::max(input_bytes, tensor_output_bytes)
        : marlin_output_bytes;
    std::uint64_t workspace_allocation_calls = 0U;
    std::uint64_t workspace_allocation_bytes = 0U;
    if (required_input_bytes > state.input_bytes) {
        if (state.input != nullptr) static_cast<void>(cudaFree(state.input));
        if (auto status = cudaMalloc(
                &state.input, static_cast<std::size_t>(required_input_bytes));
            status != cudaSuccess) {
            return cuda_error(status, "allocate CUDA input workspace");
        }
        state.input_bytes = required_input_bytes;
        ++workspace_allocation_calls;
        workspace_allocation_bytes += required_input_bytes;
    }
    if (required_output_bytes > state.output_bytes) {
        if (state.output != nullptr) static_cast<void>(cudaFree(state.output));
        if (auto status = cudaMalloc(
                &state.output, static_cast<std::size_t>(required_output_bytes));
            status != cudaSuccess) {
            return cuda_error(status, "allocate CUDA output workspace");
        }
        state.output_bytes = required_output_bytes;
        ++workspace_allocation_calls;
        workspace_allocation_bytes += required_output_bytes;
    }
    // Grown geometrically and kept, so a decode step that repeats the same
    // shapes allocates nothing. Past the ceiling the copy falls back to the
    // pageable path rather than reserving an unbounded pinned region.
    constexpr std::uint64_t matmul_host_staging_ceiling = 64U * 1024U * 1024U;
    const auto ensure_host_staging = [](std::byte*& pointer,
                                        std::uint64_t& capacity,
                                        std::uint64_t required) -> bool {
        if (required <= capacity) return capacity != 0U;
        if (required > matmul_host_staging_ceiling) return false;
        const auto target = std::bit_ceil(required);
        void* replacement = nullptr;
        if (cudaMallocHost(&replacement, static_cast<std::size_t>(target)) !=
            cudaSuccess) {
            static_cast<void>(cudaGetLastError());
            return false;
        }
        if (pointer != nullptr) static_cast<void>(cudaFreeHost(pointer));
        pointer = static_cast<std::byte*>(replacement);
        capacity = target;
        return true;
    };
    const bool stage_input = batch_input != nullptr || ensure_host_staging(
        state.matmul_host_input, state.matmul_host_input_bytes, input_bytes);
    const bool stage_output = batch_output != nullptr || ensure_host_staging(
        state.matmul_host_output, state.matmul_host_output_bytes, output_bytes);
    if (stage_input && batch_input == nullptr) {
        std::memcpy(state.matmul_host_input, input.data(), input.size_bytes());
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_start, state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "record activation upload start");
        }
    }
    if (auto status = cudaMemcpyAsync(
            (tensor_page || f32_regfed) ? static_cast<void*>(state.output)
                        : static_cast<void*>(state.input),
            batch_input != nullptr ? static_cast<const void*>(batch_input)
                        : stage_input ? static_cast<const void*>(state.matmul_host_input)
                        : static_cast<const void*>(input.data()),
            input.size_bytes(), cudaMemcpyHostToDevice, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "upload CUDA activation");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_uploaded, state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "record activation upload completion");
        }
    }
    const bool native = descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128 ||
                        descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128F32 ||
                        descriptor.encoding == CudaWeightEncoding::Fp4E2m1Group32;
    const bool w8_group32 =
        descriptor.encoding == CudaWeightEncoding::OffsetPackedInt8 &&
        rows == 1U && groups == 0U && descriptor.group_size == 32U &&
        descriptor.columns % 32U == 0U;
    if (tensor_page || f32_regfed) {
        const dim3 quantize_grid(
            static_cast<unsigned int>(descriptor.scale_columns), rows, 1U);
        auto* compact_values = reinterpret_cast<unsigned char*>(state.input);
        if (f32_tensor_page || f32_regfed) {
            auto* compact_scales = reinterpret_cast<float*>(
                compact_values + input.size());
            quantize_activation_e4m3_f32_bytes_kernel<<<
                quantize_grid, 128U, 0U, state.stream>>>(
                compact_values, compact_scales, state.output,
                descriptor.columns, rows);
        } else {
            auto* compact_scales = compact_values + input.size();
            quantize_activation_e4m3_bytes_kernel<<<
                quantize_grid, 128U, 0U, state.stream>>>(
                compact_values, compact_scales, state.output,
                descriptor.columns, rows);
        }
    } else if (descriptor.encoding ==
                   CudaWeightEncoding::Fp8E4m3Block128F32 &&
               !marlin && !regfed) {
        const auto input_rows = groups == 0U ? rows : rows * groups;
        const dim3 quantize_grid(
            static_cast<unsigned int>((descriptor.columns + 127U) / 128U),
            input_rows, 1U);
        quantize_activation_e4m3_f32_scale_kernel<<<
            quantize_grid, 128U, 0U, state.stream>>>(
            state.input, descriptor.columns, input_rows);
    } else if (native && !marlin) {
        const auto input_rows = groups == 0U ? rows : rows * groups;
        const dim3 quantize_grid(
            static_cast<unsigned int>((descriptor.columns + 127U) / 128U),
            input_rows, 1U);
        quantize_activation_e4m3_kernel<<<quantize_grid, 128U, 0U, state.stream>>>(
            state.input, descriptor.columns, input_rows);
    }
    const dim3 grid(static_cast<unsigned int>(descriptor.rows), rows, 1U);
    // Only the plain kernel tiles its input rows; every other encoding here
    // still takes one block per (output row, input row).
    const dim3 plain_grid(
        static_cast<unsigned int>(descriptor.rows),
        (rows + kPlainMatmulRowTile - 1U) / kPlainMatmulRowTile, 1U);
    constexpr unsigned int threads = 256U;
    if (marlin) {
        record_cuda_matmul_route(CudaMatmulRoute::GemmaMarlin);
        if (auto status = launch_gemma_marlin(
                state.gemma_marlin, descriptor, weight.impl_->weights,
                weight.impl_->scales, state.input, rows, state.output,
                state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "launch Gemma Marlin matmul");
        }
        const auto elements = static_cast<std::uint64_t>(rows) * descriptor.rows;
        round_bf16_rows_kernel<<<
            static_cast<unsigned int>(
                std::min<std::uint64_t>((elements + 255U) / 256U, 65535U)),
            256U, 0U, state.stream>>>(state.output, elements);
    } else if (descriptor.encoding == CudaWeightEncoding::Plain &&
        descriptor.dtype == SafetensorsDtype::Bf16 && rows == 1U &&
        groups == 0U) {
        constexpr unsigned int warps_per_block = threads / 32U;
        const auto blocks = static_cast<unsigned int>(
            (descriptor.rows + warps_per_block - 1U) / warps_per_block);
        bf16_matvec_kernel<<<blocks, threads, 0U, state.stream>>>(
            state.output, state.input,
            static_cast<const __nv_bfloat16*>(weight.impl_->weights),
            descriptor.columns, descriptor.rows);
    } else if (descriptor.encoding == CudaWeightEncoding::Plain) {
        record_cuda_matmul_route(CudaMatmulRoute::PlainGeneric);
        if (rows == 1U) {
            plain_matmul_kernel<1U><<<grid, threads, 0, state.stream>>>(
                state.output, state.input, weight.impl_->weights,
                static_cast<int>(descriptor.dtype), rows, descriptor.columns,
                descriptor.rows, groups, rows_per_group);
        } else if (descriptor.dtype == SafetensorsDtype::Bf16 && groups == 0U) {
            constexpr unsigned int warps_per_block = threads / 32U;
            const dim3 matvec_grid(
                static_cast<unsigned int>(
                    (descriptor.rows + warps_per_block - 1U) / warps_per_block),
                (rows + kBf16MatvecRowTile - 1U) / kBf16MatvecRowTile, 1U);
            bf16_matvec_rows_kernel<kBf16MatvecRowTile><<<
                matvec_grid, threads, 0, state.stream>>>(
                state.output, state.input,
                static_cast<const __nv_bfloat16*>(weight.impl_->weights), rows,
                descriptor.columns, descriptor.rows);
        } else
        plain_matmul_kernel<kPlainMatmulRowTile><<<plain_grid, threads, 0, state.stream>>>(
            state.output, state.input, weight.impl_->weights,
            static_cast<int>(descriptor.dtype), rows, descriptor.columns,
            descriptor.rows, groups, rows_per_group);
    } else if (w8_group32) {
        record_cuda_matmul_route(CudaMatmulRoute::PackedInt8Group32);
        constexpr unsigned int warps_per_block = threads / 32U;
        const auto blocks = static_cast<unsigned int>(
            (descriptor.rows + warps_per_block - 1U) / warps_per_block);
        packed_int8_group32_matvec_kernel<<<
            blocks, threads, 0U, state.stream>>>(
            state.output, state.input,
            static_cast<const std::uint32_t*>(weight.impl_->weights),
            static_cast<const __nv_bfloat16*>(weight.impl_->scales),
            descriptor.packed_columns, descriptor.scale_columns,
            descriptor.columns, descriptor.rows);
    } else if (descriptor.encoding == CudaWeightEncoding::OffsetPackedInt4 ||
               descriptor.encoding == CudaWeightEncoding::OffsetPackedInt8) {
        record_cuda_matmul_route(CudaMatmulRoute::PackedOffsetInt);
        const auto bits = descriptor.encoding == CudaWeightEncoding::OffsetPackedInt4 ? 4U : 8U;
        packed_matmul_kernel<<<grid, threads, 0, state.stream>>>(
            state.output, state.input, static_cast<const std::uint32_t*>(weight.impl_->weights),
            static_cast<const __nv_bfloat16*>(weight.impl_->scales), bits,
            descriptor.group_size, descriptor.packed_columns,
            descriptor.scale_columns, rows, descriptor.columns, descriptor.rows,
            groups, rows_per_group);
    } else if (descriptor.encoding == CudaWeightEncoding::Nvfp4Group16) {
        record_cuda_matmul_route(CudaMatmulRoute::Nvfp4Group16);
        nvfp4_group16_matmul_kernel<<<grid, threads, 0, state.stream>>>(
            state.output, state.input,
            static_cast<const unsigned char*>(weight.impl_->weights),
            static_cast<const unsigned char*>(weight.impl_->scales),
            descriptor.global_scale, descriptor.packed_columns,
            descriptor.scale_columns, descriptor.group_size, rows,
            descriptor.columns, descriptor.rows, groups, rows_per_group);
    } else if (regfed) {
        // The activation permutation reads state.input, which already holds the
        // E4M3-rounded FP32 activation the scalar routes consume. An E4M3 value
        // has three mantissa bits, so its BF16 image is exact and the tensor op
        // multiplies the same real numbers the scalar kernel multiplies.
        const auto column_blocks = static_cast<std::uint32_t>(
            (std::min<std::uint64_t>(rows, kRegfedMaxM) + kRegfedTileM - 1U) /
            kRegfedTileM);
        const auto groups_per_block = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(rows, kRegfedTileM));
        const auto k_tiles =
            static_cast<std::uint32_t>(descriptor.columns / kRegfedTileK);
        const auto n_tiles =
            static_cast<std::uint32_t>(descriptor.rows / kRegfedTileN);
        const std::uint32_t units =
            (descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128 ||
             descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128F32)
                ? static_cast<std::uint32_t>(descriptor.columns / 32U)
                : k_tiles / kRegfedKPerLoad;
        std::uint32_t split = regfed_split_k(units, n_tiles);
        if (f32_regfed) {
            // A continuous scale belongs to a complete K128 block (four
            // 32-column pairs), so split-K boundaries may never bisect one.
            split = 1U;
            while (split < 16U && units % ((split * 2U) * 4U) == 0U &&
                   n_tiles * split * 2U <= 4096U) {
                split *= 2U;
            }
        }
        const std::uint64_t activation_bytes =
            static_cast<std::uint64_t>(k_tiles) * column_blocks *
            groups_per_block * 4U * sizeof(uint2);
        const std::uint64_t partial_bytes =
            static_cast<std::uint64_t>(n_tiles) * split * kRegfedTileN *
            kRegfedMaxM * sizeof(float);
        const std::uint64_t counter_bytes =
            static_cast<std::uint64_t>(n_tiles) * sizeof(std::uint32_t);
        const auto grow = [&](void*& pointer, std::uint64_t& capacity,
                              std::uint64_t required, bool zero) -> cudaError_t {
            if (required <= capacity) return cudaSuccess;
            if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
            pointer = nullptr;
            capacity = 0U;
            if (auto status = cudaMalloc(&pointer,
                                         static_cast<std::size_t>(required));
                status != cudaSuccess) {
                return status;
            }
            capacity = required;
            if (!zero) return cudaSuccess;
            return cudaMemsetAsync(pointer, 0,
                                   static_cast<std::size_t>(required),
                                   state.stream);
        };
        if (auto status = grow(state.regfed_activation,
                               state.regfed_activation_bytes, activation_bytes,
                               false);
            status != cudaSuccess) {
            return cuda_error(status, "allocate register-fed activation workspace");
        }
        auto* partials = static_cast<void*>(state.regfed_partials);
        if (auto status =
                grow(partials, state.regfed_partial_bytes, partial_bytes, false);
            status != cudaSuccess) {
            return cuda_error(status, "allocate register-fed partial workspace");
        }
        state.regfed_partials = static_cast<float*>(partials);
        auto* counters = static_cast<void*>(state.regfed_counters);
        if (auto status =
                grow(counters, state.regfed_counter_bytes, counter_bytes, true);
            status != cudaSuccess) {
            return cuda_error(status, "allocate register-fed counter workspace");
        }
        state.regfed_counters = static_cast<std::uint32_t*>(counters);
        const unsigned int blocks = static_cast<unsigned int>(std::min<std::uint64_t>(
            (static_cast<std::uint64_t>(n_tiles) * split +
             kRegfedWarpsPerBlock - 1U) / kRegfedWarpsPerBlock, 65535U));
        // A weight already in fragment order cannot be read canonically, so a
        // wide call chunks the activation through the skinny kernel rather than
        // silently taking a route that would misread the layout. Each chunk is
        // counted, so a run where this happens is visible in the census.
        for (std::uint32_t start = 0U; start < rows; start += kRegfedMaxM) {
            const auto chunk = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(kRegfedMaxM, rows - start));
            const auto chunk_blocks =
                (std::min<std::uint32_t>(chunk, kRegfedMaxM) + kRegfedTileM - 1U) /
                kRegfedTileM;
            const auto chunk_groups = std::min<std::uint32_t>(chunk, kRegfedTileM);
            const std::uint64_t chunk_activation_bytes =
                static_cast<std::uint64_t>(k_tiles) * chunk_blocks *
                chunk_groups * 4U * sizeof(uint2);
            static_cast<void>(chunk_activation_bytes);
            const auto fragment_total = static_cast<std::uint64_t>(k_tiles) *
                                        chunk_blocks * chunk_groups * 4U;
            if (f32_regfed) {
                const auto* compact_values =
                    reinterpret_cast<const unsigned char*>(state.input);
                regfed_fp8_activation_fragment_kernel<<<
                    static_cast<unsigned int>(std::min<std::uint64_t>(
                        (fragment_total + 255U) / 256U, 65535U)),
                    256U, 0U, state.stream>>>(
                    static_cast<uint2*>(state.regfed_activation),
                    compact_values + static_cast<std::size_t>(start) *
                                         descriptor.columns,
                    chunk, static_cast<std::uint32_t>(descriptor.columns),
                    chunk_blocks, chunk_groups);
            } else {
                regfed_activation_fragment_kernel<<<
                    static_cast<unsigned int>(std::min<std::uint64_t>(
                        (fragment_total + 255U) / 256U, 65535U)),
                    256U, 0U, state.stream>>>(
                    static_cast<uint2*>(state.regfed_activation),
                    state.input + static_cast<std::size_t>(start) *
                                      descriptor.columns,
                    chunk, static_cast<std::uint32_t>(descriptor.columns),
                    chunk_blocks, chunk_groups);
            }
            float* chunk_output =
                state.output + static_cast<std::size_t>(start) * descriptor.rows;
            if (descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128) {
                record_cuda_matmul_route(CudaMatmulRoute::Fp8RegisterFed);
                if (chunk_blocks == 1U) {
                    regfed_fp8_matmul_kernel<1U><<<
                        blocks, kRegfedWarpsPerBlock * 32U, 0U, state.stream>>>(
                        chunk_output,
                        static_cast<const uint4*>(weight.impl_->weights),
                        static_cast<const unsigned char*>(weight.impl_->scales),
                        static_cast<const uint2*>(state.regfed_activation),
                        static_cast<std::uint32_t>(descriptor.columns),
                        static_cast<std::uint32_t>(descriptor.rows),
                        static_cast<std::uint32_t>(descriptor.scale_columns),
                        split, chunk, chunk_groups, state.regfed_partials,
                        state.regfed_counters);
                } else {
                    regfed_fp8_matmul_kernel<2U><<<
                        blocks, kRegfedWarpsPerBlock * 32U, 0U, state.stream>>>(
                        chunk_output,
                        static_cast<const uint4*>(weight.impl_->weights),
                        static_cast<const unsigned char*>(weight.impl_->scales),
                        static_cast<const uint2*>(state.regfed_activation),
                        static_cast<std::uint32_t>(descriptor.columns),
                        static_cast<std::uint32_t>(descriptor.rows),
                        static_cast<std::uint32_t>(descriptor.scale_columns),
                        split, chunk, chunk_groups, state.regfed_partials,
                        state.regfed_counters);
                }
            } else if (descriptor.encoding ==
                       CudaWeightEncoding::Fp8E4m3Block128F32) {
                record_cuda_matmul_route(CudaMatmulRoute::Fp8F32RegisterFed);
                const auto* compact_values =
                    reinterpret_cast<const unsigned char*>(state.input);
                const auto* compact_scales = reinterpret_cast<const float*>(
                    compact_values + input.size());
                const auto launch = [&](auto tag) {
                    constexpr std::uint32_t kBlocks = decltype(tag)::value;
                    regfed_fp8_f32_matmul_kernel<kBlocks><<<
                        blocks, kRegfedWarpsPerBlock * 32U, 0U,
                        state.stream>>>(
                        chunk_output,
                        static_cast<const uint4*>(weight.impl_->weights),
                        static_cast<const float*>(weight.impl_->scales),
                        static_cast<const uint2*>(state.regfed_activation),
                        compact_scales + static_cast<std::size_t>(start) *
                                             descriptor.scale_columns,
                        static_cast<std::uint32_t>(descriptor.columns),
                        static_cast<std::uint32_t>(descriptor.rows),
                        static_cast<std::uint32_t>(descriptor.scale_columns),
                        split, chunk, chunk_groups, state.regfed_partials,
                        state.regfed_counters);
                };
                if (chunk_blocks == 1U) {
                    launch(std::integral_constant<std::uint32_t, 1U>{});
                } else {
                    launch(std::integral_constant<std::uint32_t, 2U>{});
                }
            } else {
                record_cuda_matmul_route(CudaMatmulRoute::Fp4RegisterFed);
                if (chunk_blocks == 1U) {
                    regfed_fp4_matmul_kernel<1U><<<
                        blocks, kRegfedWarpsPerBlock * 32U, 0U, state.stream>>>(
                        chunk_output,
                        static_cast<const std::uint32_t*>(weight.impl_->weights),
                        static_cast<const unsigned char*>(weight.impl_->scales),
                        static_cast<const uint2*>(state.regfed_activation),
                        static_cast<std::uint32_t>(descriptor.columns),
                        static_cast<std::uint32_t>(descriptor.rows), split,
                        chunk, chunk_groups, state.regfed_partials,
                        state.regfed_counters);
                } else {
                    regfed_fp4_matmul_kernel<2U><<<
                        blocks, kRegfedWarpsPerBlock * 32U, 0U, state.stream>>>(
                        chunk_output,
                        static_cast<const std::uint32_t*>(weight.impl_->weights),
                        static_cast<const unsigned char*>(weight.impl_->scales),
                        static_cast<const uint2*>(state.regfed_activation),
                        static_cast<std::uint32_t>(descriptor.columns),
                        static_cast<std::uint32_t>(descriptor.rows), split,
                        chunk, chunk_groups, state.regfed_partials,
                        state.regfed_counters);
                }
            }
        }
    } else if (dsv4_tensor_page) {
        record_cuda_matmul_route(CudaMatmulRoute::Fp8TensorPage);
        const dim3 tensor_grid(
            static_cast<unsigned int>(
                descriptor.rows / kDsv4Fp8TensorBlockN),
            static_cast<unsigned int>(
                padded_rows / kDsv4Fp8TensorBlockM), 1U);
        const auto* compact_values =
            reinterpret_cast<const unsigned char*>(state.input);
        const auto* compact_scales = compact_values + input.size();
        dsv4_fp8_decode_bf16_tensor_kernel<<<
            tensor_grid, threads, 0U, state.stream>>>(
            state.output, compact_values, compact_scales,
            static_cast<const unsigned char*>(weight.impl_->weights),
            static_cast<const unsigned char*>(weight.impl_->scales), rows,
            static_cast<std::uint32_t>(descriptor.columns),
            static_cast<std::uint32_t>(descriptor.rows));
    } else if (f32_tensor_page) {
        record_cuda_matmul_route(CudaMatmulRoute::Fp8F32TensorPage);
        const dim3 tensor_grid(
            static_cast<unsigned int>(
                descriptor.rows / kFp8F32TensorBlockN),
            static_cast<unsigned int>(
                padded_rows / kDsv4Fp8TensorBlockM), 1U);
        const auto* compact_values =
            reinterpret_cast<const unsigned char*>(state.input);
        const auto* compact_scales = reinterpret_cast<const float*>(
            compact_values + input.size());
        const auto launch = [&](auto tag) {
            constexpr bool kPrepacked = decltype(tag)::value;
            fp8_f32_decode_bf16_tensor_kernel<kPrepacked><<<
                tensor_grid, threads, 0U, state.stream>>>(
                state.output, compact_values, compact_scales,
                static_cast<const unsigned char*>(weight.impl_->weights),
                static_cast<const float*>(weight.impl_->scales), rows,
                static_cast<std::uint32_t>(descriptor.columns),
                static_cast<std::uint32_t>(descriptor.rows));
        };
        if (weight.impl_->fragment_prepacked) {
            launch(std::true_type{});
        } else {
            launch(std::false_type{});
        }
    } else if (descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128) {
        record_cuda_matmul_route(CudaMatmulRoute::Fp8E4m3Block128);
        native_fp8_matmul_kernel<<<grid, threads, 0, state.stream>>>(
            state.output, state.input,
            static_cast<const unsigned char*>(weight.impl_->weights),
            static_cast<const unsigned char*>(weight.impl_->scales),
            descriptor.scale_columns, rows, descriptor.columns, descriptor.rows,
            groups, rows_per_group);
    } else if (descriptor.encoding ==
               CudaWeightEncoding::Fp8E4m3Block128F32) {
        record_cuda_matmul_route(CudaMatmulRoute::Fp8E4m3Block128F32);
        native_fp8_f32_scale_matmul_kernel<<<grid, threads, 0, state.stream>>>(
            state.output, state.input,
            static_cast<const unsigned char*>(weight.impl_->weights),
            static_cast<const float*>(weight.impl_->scales),
            descriptor.scale_columns, rows, descriptor.columns, descriptor.rows,
            groups, rows_per_group);
    } else if (descriptor.encoding == CudaWeightEncoding::Fp4E2m1Group32) {
        record_cuda_matmul_route(CudaMatmulRoute::Fp4E2m1Group32);
        native_fp4_matmul_kernel<<<grid, threads, 0, state.stream>>>(
            state.output, state.input,
            static_cast<const unsigned char*>(weight.impl_->weights),
            static_cast<const unsigned char*>(weight.impl_->scales),
            descriptor.packed_columns, descriptor.scale_columns, rows,
            descriptor.columns, descriptor.rows, groups, rows_per_group);
    } else {
        // MIX-1: no hidden fallback. This branch previously routed every
        // unrecognised encoding into the FP4 kernel, which would decode the
        // wrong format silently. An unsupported case must fail explicitly.
        record_cuda_matmul_route(CudaMatmulRoute::Unsupported);
        ValidationResult unsupported;
        unsupported.errors.emplace_back(
            "CUDA matmul has no approved exact route for weight encoding " +
            std::to_string(
                static_cast<unsigned>(descriptor.encoding)) +
            "; refusing to substitute a different format");
        return unsupported;
    }
    if (softcap > 0.0F) {
        gemma4_softcap_logits_kernel<<<
            static_cast<unsigned int>((output.size() + threads - 1U) / threads),
            threads, 0U, state.stream>>>(
            state.output, output.size(), softcap);
    }
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch CUDA matmul");
    }
    if (round_output) {
        // The caller's BF16 boundary, applied where the values already are.
        // Rounding on the host is a single-threaded pass over the whole
        // activation: the 32,768-wide query projection alone is 954 million
        // floats over a 677-token prompt.
        const auto rounded_blocks = static_cast<unsigned int>(
            (output.size() + threads - 1U) / threads);
        dsv4_round_float_bf16<<<rounded_blocks, threads, 0, state.stream>>>(
            state.output, output.size());
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            return cuda_error(status, "launch CUDA activation rounding");
        }
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.kernel_finished, state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "record CUDA kernel completion");
        }
    }
    if (auto status = cudaMemcpyAsync(
            batch_output != nullptr ? static_cast<void*>(batch_output)
                         : stage_output ? static_cast<void*>(state.matmul_host_output)
                         : static_cast<void*>(output.data()),
            state.output, output.size_bytes(),
            cudaMemcpyDeviceToHost, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "download CUDA activation");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_downloaded, state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "record activation download completion");
        }
    }
    const auto wait_started = std::chrono::steady_clock::now();
    const auto issue_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            wait_started - issue_started).count());
    if (defer_completion) {
        std::scoped_lock lock(impl_->mutex);
        auto& device_stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [&weight](const auto& value) {
                return value.device == weight.impl_->device;
            });
        device_stats.activation_h2d_bytes += input_bytes;
        device_stats.activation_d2h_bytes += output_bytes;
        ++device_stats.matmul_calls;
        device_stats.workspace_allocation_calls += workspace_allocation_calls;
        device_stats.workspace_allocation_bytes += workspace_allocation_bytes;
        device_stats.matmul_issue_nanoseconds += issue_nanoseconds;
        return result;
    }
    if (auto status = cudaStreamSynchronize(state.stream); status != cudaSuccess) {
        return cuda_error(status, "synchronize CUDA matmul");
    }
    const auto synchronized = std::chrono::steady_clock::now();
    const auto synchronization_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            synchronized - wait_started).count());
    const auto finish_started = synchronized;
    if (stage_output) {
        std::memcpy(output.data(), state.matmul_host_output,
                    output.size_bytes());
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
                &h2d_milliseconds, state.activation_start, state.activation_uploaded);
            status != cudaSuccess) {
            return cuda_error(status, "measure activation upload");
        }
        if (auto status = cudaEventElapsedTime(
                &kernel_milliseconds, state.activation_uploaded, state.kernel_finished);
            status != cudaSuccess) {
            return cuda_error(status, "measure CUDA kernel");
        }
        if (auto status = cudaEventElapsedTime(
                &d2h_milliseconds, state.kernel_finished, state.activation_downloaded);
            status != cudaSuccess) {
            return cuda_error(status, "measure activation download");
        }
        activation_h2d_nanoseconds = static_cast<std::uint64_t>(
            std::llround(static_cast<double>(h2d_milliseconds) * 1.0e6));
        kernel_nanoseconds = static_cast<std::uint64_t>(
            std::llround(static_cast<double>(kernel_milliseconds) * 1.0e6));
        activation_d2h_nanoseconds = static_cast<std::uint64_t>(
            std::llround(static_cast<double>(d2h_milliseconds) * 1.0e6));
    }
    const auto finish_nanoseconds = elapsed_nanoseconds_since(finish_started);
    if (profile != nullptr) {
        profile->issue_nanoseconds = issue_nanoseconds;
        profile->finish_nanoseconds = finish_nanoseconds;
        profile->synchronization_nanoseconds = synchronization_nanoseconds;
        profile->h2d_nanoseconds = activation_h2d_nanoseconds;
        profile->kernel_nanoseconds = kernel_nanoseconds;
        profile->d2h_nanoseconds = activation_d2h_nanoseconds;
    }
    {
        std::scoped_lock lock(impl_->mutex);
        auto& device_stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [&weight](const auto& value) { return value.device == weight.impl_->device; });
        device_stats.activation_h2d_bytes += input_bytes;
        device_stats.activation_d2h_bytes += output_bytes;
        ++device_stats.matmul_calls;
        device_stats.workspace_allocation_calls += workspace_allocation_calls;
        device_stats.workspace_allocation_bytes += workspace_allocation_bytes;
        // Keep the historical synchronization total stable: it includes the
        // pinned-output memcpy after cudaStreamSynchronize. The exact stream
        // wait and post-wait finish are separately exposed in the profile.
        record_synchronization(device_stats,
                               SynchronizationSubsystem::Projection, 1U,
                               wait_nanoseconds);
        device_stats.matmul_issue_nanoseconds += issue_nanoseconds;
        device_stats.matmul_finish_nanoseconds += finish_nanoseconds;
        device_stats.activation_h2d_nanoseconds += activation_h2d_nanoseconds;
        device_stats.kernel_nanoseconds += kernel_nanoseconds;
        device_stats.activation_d2h_nanoseconds += activation_d2h_nanoseconds;
    }
    return result;
}
