// One register-fed FP8 matvec against a device-resident activation. Used by the
// DeepSeek shared expert, which never reaches matmul_impl because its operands
// never leave the device.
cudaError_t launch_regfed_fp8_matvec(RegfedWorkspace& workspace,
                                     const CudaWeightDescriptor& descriptor,
                                     void* weights, void* scales,
                                     bool& prepacked, const float* input,
                                     float* output, cudaStream_t stream,
                                     bool reuse_activation = false) {
    const auto rows = static_cast<std::uint32_t>(descriptor.rows);
    const auto columns = static_cast<std::uint32_t>(descriptor.columns);
    const std::uint32_t n_tiles = rows / kRegfedTileN;
    const std::uint32_t k_tiles = columns / kRegfedTileK;
    const std::uint32_t split = regfed_split_k(columns / 32U, n_tiles);
    const std::uint64_t activation_bytes =
        static_cast<std::uint64_t>(k_tiles) * 4U * sizeof(uint2);
    const std::uint64_t partial_bytes = static_cast<std::uint64_t>(n_tiles) *
                                        split * kRegfedTileN * kRegfedMaxM *
                                        sizeof(float);
    const std::uint64_t counter_bytes =
        static_cast<std::uint64_t>(n_tiles) * sizeof(std::uint32_t);
    if (auto status = regfed_grow(workspace.activation,
                                  workspace.activation_bytes, activation_bytes,
                                  false, stream);
        status != cudaSuccess) {
        return status;
    }
    if (auto status = regfed_grow(workspace.partials, workspace.partial_bytes,
                                  partial_bytes, false, stream);
        status != cudaSuccess) {
        return status;
    }
    if (auto status = regfed_grow(workspace.counters, workspace.counter_bytes,
                                  counter_bytes, true, stream);
        status != cudaSuccess) {
        return status;
    }
    if (!prepacked) {
        if (auto status = regfed_grow(
                workspace.scratch, workspace.scratch_bytes,
                fragment_prepack_scratch_bytes(descriptor), false, stream);
            status != cudaSuccess) {
            return status;
        }
        if (auto status = launch_fragment_prepack(descriptor, weights, scales,
                                                  workspace.scratch, stream);
            status != cudaSuccess) {
            return status;
        }
        prepacked = true;
    }
    constexpr unsigned int threads = 256U;
    // Gate and up read the same hidden vector, so the second of the pair reuses
    // the permutation the first one wrote rather than recomputing it.
    if (!reuse_activation) {
        const std::uint64_t fragment_total =
            static_cast<std::uint64_t>(k_tiles) * 4U;
        regfed_activation_fragment_kernel<<<
            static_cast<unsigned int>(
                std::min<std::uint64_t>((fragment_total + threads - 1U) / threads,
                                        65535U)),
            threads, 0U, stream>>>(
            static_cast<uint2*>(workspace.activation), input, 1U, columns, 1U,
            1U);
        if (auto status = cudaGetLastError(); status != cudaSuccess) return status;
    }
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<std::uint64_t>((static_cast<std::uint64_t>(n_tiles) * split +
                                 kRegfedWarpsPerBlock - 1U) /
                                    kRegfedWarpsPerBlock,
                                65535U));
    regfed_fp8_matmul_kernel<1U><<<blocks, kRegfedWarpsPerBlock * 32U, 0U,
                                   stream>>>(
        output, static_cast<const uint4*>(weights),
        static_cast<const unsigned char*>(scales),
        static_cast<const uint2*>(workspace.activation), columns, rows,
        static_cast<std::uint32_t>(descriptor.scale_columns), split, 1U, 1U,
        static_cast<float*>(workspace.partials),
        static_cast<std::uint32_t*>(workspace.counters));
    return cudaGetLastError();
}

// F32-scaled FP8 counterpart for device-resident activations.  GLM-5.3 uses
// weight_scale_inv tensors, not DeepSeek's E8M0 scale bytes.  Keep the same
// compact activation quantization and M<=16 chunking as matmul_impl so a row
// has the accepted projection operands and reduction association.
cudaError_t launch_regfed_fp8_f32_rows(
    RegfedWorkspace& workspace, const CudaWeightDescriptor& descriptor,
    void* weights, void* scales, bool& prepacked, const float* input,
    float* output, std::uint32_t batch, cudaStream_t stream) {
    if (descriptor.encoding != CudaWeightEncoding::Fp8E4m3Block128F32 ||
        batch == 0U) {
        return cudaErrorInvalidValue;
    }
    const auto rows = static_cast<std::uint32_t>(descriptor.rows);
    const auto columns = static_cast<std::uint32_t>(descriptor.columns);
    const auto scale_columns =
        static_cast<std::uint32_t>(descriptor.scale_columns);
    const std::uint32_t n_tiles = rows / kRegfedTileN;
    const std::uint32_t k_tiles = columns / kRegfedTileK;
    std::uint32_t split = 1U;
    const auto column_blocks = columns / 32U;
    while (split < 16U && column_blocks % ((split * 2U) * 4U) == 0U &&
           n_tiles * split * 2U <= 4096U) {
        split *= 2U;
    }
    const std::uint64_t activation_bytes =
        static_cast<std::uint64_t>(k_tiles) * kRegfedMaxColBlocks *
        kRegfedTileM * 4U * sizeof(uint2);
    const std::uint64_t partial_bytes = static_cast<std::uint64_t>(n_tiles) *
        split * kRegfedTileN * kRegfedMaxM * sizeof(float);
    const std::uint64_t counter_bytes =
        static_cast<std::uint64_t>(n_tiles) * sizeof(std::uint32_t);
    const std::uint64_t compact_bytes =
        static_cast<std::uint64_t>(batch) * columns +
        static_cast<std::uint64_t>(batch) * scale_columns * sizeof(float);
    const std::uint64_t scratch_bytes = std::max<std::uint64_t>(
        compact_bytes, fragment_prepack_scratch_bytes(descriptor));
    if (auto status = regfed_grow(workspace.activation,
                                  workspace.activation_bytes,
                                  activation_bytes, false, stream);
        status != cudaSuccess) return status;
    if (auto status = regfed_grow(workspace.partials, workspace.partial_bytes,
                                  partial_bytes, false, stream);
        status != cudaSuccess) return status;
    if (auto status = regfed_grow(workspace.counters, workspace.counter_bytes,
                                  counter_bytes, true, stream);
        status != cudaSuccess) return status;
    if (auto status = regfed_grow(workspace.scratch, workspace.scratch_bytes,
                                  scratch_bytes, false, stream);
        status != cudaSuccess) return status;
    if (!prepacked) {
        if (auto status = launch_fragment_prepack(
                descriptor, weights, scales, workspace.scratch, stream);
            status != cudaSuccess) return status;
        prepacked = true;
    }
    auto* compact_values = static_cast<unsigned char*>(workspace.scratch);
    auto* compact_scales = reinterpret_cast<float*>(
        compact_values + static_cast<std::uint64_t>(batch) * columns);
    quantize_activation_e4m3_f32_bytes_kernel<<<
        dim3{scale_columns, batch, 1U}, 128U, 0U, stream>>>(
        compact_values, compact_scales, input, columns, batch);
    if (auto status = cudaGetLastError(); status != cudaSuccess) return status;
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<std::uint64_t>(
            (static_cast<std::uint64_t>(n_tiles) * split +
             kRegfedWarpsPerBlock - 1U) / kRegfedWarpsPerBlock,
            65535U));
    for (std::uint32_t start = 0U; start < batch; start += kRegfedMaxM) {
        const auto chunk = std::min<std::uint32_t>(kRegfedMaxM, batch - start);
        const auto chunk_blocks =
            (chunk + kRegfedTileM - 1U) / kRegfedTileM;
        const auto chunk_groups = std::min<std::uint32_t>(chunk, kRegfedTileM);
        const auto fragment_total = static_cast<std::uint64_t>(k_tiles) *
            chunk_blocks * chunk_groups * 4U;
        regfed_fp8_activation_fragment_kernel<<<
            static_cast<unsigned int>(std::min<std::uint64_t>(
                (fragment_total + 255U) / 256U, 65535U)),
            256U, 0U, stream>>>(
            static_cast<uint2*>(workspace.activation),
            compact_values + static_cast<std::uint64_t>(start) * columns,
            chunk, columns, chunk_blocks, chunk_groups);
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            return status;
        }
        auto* chunk_output =
            output + static_cast<std::uint64_t>(start) * rows;
        const auto launch = [&](auto tag) {
            constexpr std::uint32_t kBlocks = decltype(tag)::value;
            regfed_fp8_f32_matmul_kernel<kBlocks><<<
                blocks, kRegfedWarpsPerBlock * 32U, 0U, stream>>>(
                chunk_output, static_cast<const uint4*>(weights),
                static_cast<const float*>(scales),
                static_cast<const uint2*>(workspace.activation),
                compact_scales + static_cast<std::uint64_t>(start) *
                                     scale_columns,
                columns, rows, scale_columns, split, chunk, chunk_groups,
                static_cast<float*>(workspace.partials),
                static_cast<std::uint32_t*>(workspace.counters));
        };
        if (chunk_blocks == 1U) {
            launch(std::integral_constant<std::uint32_t, 1U>{});
        } else {
            launch(std::integral_constant<std::uint32_t, 2U>{});
        }
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            return status;
        }
    }
    return cudaSuccess;
}

// One register-fed FP4 matvec against a device-resident activation. Unlike the
// DeepSeek helper above, Gemma weights have already been explicitly prepacked
// by their loader after all consumers have been audited.
cudaError_t launch_regfed_fp4_matvec(
    RegfedWorkspace& workspace, const CudaWeightDescriptor& descriptor,
    const void* weights, const void* scales, const float* input, float* output,
    cudaStream_t stream) {
    const auto rows = static_cast<std::uint32_t>(descriptor.rows);
    const auto columns = static_cast<std::uint32_t>(descriptor.columns);
    const std::uint32_t n_tiles = rows / kRegfedTileN;
    const std::uint32_t k_tiles = columns / kRegfedTileK;
    const std::uint32_t split =
        regfed_split_k(k_tiles / kRegfedKPerLoad, n_tiles);
    const std::uint64_t activation_bytes =
        static_cast<std::uint64_t>(k_tiles) * 4U * sizeof(uint2);
    const std::uint64_t partial_bytes = static_cast<std::uint64_t>(n_tiles) *
                                        split * kRegfedTileN * kRegfedMaxM *
                                        sizeof(float);
    const std::uint64_t counter_bytes =
        static_cast<std::uint64_t>(n_tiles) * sizeof(std::uint32_t);
    if (auto status = regfed_grow(workspace.activation,
                                  workspace.activation_bytes, activation_bytes,
                                  false, stream);
        status != cudaSuccess) {
        return status;
    }
    if (auto status = regfed_grow(workspace.partials, workspace.partial_bytes,
                                  partial_bytes, false, stream);
        status != cudaSuccess) {
        return status;
    }
    if (auto status = regfed_grow(workspace.counters, workspace.counter_bytes,
                                  counter_bytes, true, stream);
        status != cudaSuccess) {
        return status;
    }
    constexpr unsigned int threads = 256U;
    const std::uint64_t fragment_total =
        static_cast<std::uint64_t>(k_tiles) * 4U;
    regfed_activation_fragment_kernel<<<
        static_cast<unsigned int>(std::min<std::uint64_t>(
            (fragment_total + threads - 1U) / threads, 65535U)),
        threads, 0U, stream>>>(
        static_cast<uint2*>(workspace.activation), input, 1U, columns, 1U, 1U);
    if (auto status = cudaGetLastError(); status != cudaSuccess) return status;
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<std::uint64_t>(
            (static_cast<std::uint64_t>(n_tiles) * split +
             kRegfedWarpsPerBlock - 1U) /
                kRegfedWarpsPerBlock,
            65535U));
    regfed_fp4_matmul_kernel<1U><<<blocks, kRegfedWarpsPerBlock * 32U, 0U,
                                   stream>>>(
        output, static_cast<const std::uint32_t*>(weights),
        static_cast<const unsigned char*>(scales),
        static_cast<const uint2*>(workspace.activation), columns, rows, split,
        1U, 1U, static_cast<float*>(workspace.partials),
        static_cast<std::uint32_t*>(workspace.counters));
    return cudaGetLastError();
}

__global__ void deepseek_fp8_gate_up_kernel(
    float* activation, const float* hidden,
    const unsigned char* w1, const unsigned char* w1_scales,
    const unsigned char* w3, const unsigned char* w3_scales,
    std::uint64_t columns, std::uint64_t intermediate,
    std::uint64_t scale_columns, float swiglu_limit,
    const float* bf16_silu, unsigned int* error_flag) {
    const std::uint64_t output_row = blockIdx.x;
    if (output_row >= intermediate) return;
    const std::uint64_t weight_base = output_row * columns;
    const std::uint64_t scale_row = (output_row / 128U) * scale_columns;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint64_t groups = (columns + 31U) / 32U;
    float gate = 0.0F;
    float up = 0.0F;
    for (std::uint64_t group = warp; group < groups; group += 8U) {
        const std::uint64_t column = group * 32U + lane;
        float gate_scale = lane == 0U
                               ? fp8_e8m0_scale_bits(
                                     w1_scales[scale_row + (group * 32U) / 128U])
                               : 0.0F;
        float up_scale = lane == 0U
                             ? fp8_e8m0_scale_bits(
                                   w3_scales[scale_row + (group * 32U) / 128U])
                             : 0.0F;
        gate_scale = __shfl_sync(0xFFFF'FFFFU, gate_scale, 0);
        up_scale = __shfl_sync(0xFFFF'FFFFU, up_scale, 0);
        if (column < columns) {
            const float input = hidden[column];
            gate += input * fp8_e4m3_value(w1[weight_base + column]) * gate_scale;
            up += input * fp8_e4m3_value(w3[weight_base + column]) * up_scale;
        }
    }
    gate = reduce_block(gate);
    __syncthreads();
    up = reduce_block(up);
    if (threadIdx.x == 0U) {
        const float rounded_gate = bf16_round(gate);
        const float rounded_up = bf16_round(up);
        if (!isfinite(rounded_gate) || !isfinite(rounded_up)) {
            atomicExch(error_flag, 1U);
            activation[output_row] = __uint_as_float(0x7FC0'0000U);
            return;
        }
        const float limited_gate = fminf(rounded_gate, swiglu_limit);
        const float limited_up = fmaxf(-swiglu_limit,
                                       fminf(rounded_up, swiglu_limit));
        const auto gate_bits = static_cast<std::uint16_t>(
            __float_as_uint(limited_gate) >> 16U);
        activation[output_row] = bf16_round(
            bf16_silu[gate_bits] * limited_up);
    }
}

__global__ void deepseek_fp8_down_kernel(
    float* output, const float* activation,
    const unsigned char* weights, const unsigned char* scales,
    std::uint64_t columns, std::uint64_t rows,
    std::uint64_t scale_columns) {
    const std::uint64_t output_row = blockIdx.x;
    if (output_row >= rows) return;
    const std::uint64_t weight_base = output_row * columns;
    const std::uint64_t scale_row = (output_row / 128U) * scale_columns;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint64_t groups = (columns + 31U) / 32U;
    float sum = 0.0F;
    for (std::uint64_t group = warp; group < groups; group += 8U) {
        const std::uint64_t column = group * 32U + lane;
        float scale = lane == 0U
                          ? fp8_e8m0_scale_bits(
                                scales[scale_row + (group * 32U) / 128U])
                          : 0.0F;
        scale = __shfl_sync(0xFFFF'FFFFU, scale, 0);
        if (column < columns) {
            sum += activation[column] *
                   fp8_e4m3_value(weights[weight_base + column]) * scale;
        }
    }
    sum = reduce_block(sum);
    if (threadIdx.x == 0U) output[output_row] = bf16_round(sum);
}

// The CPU callback emits one partial per reconstructed TP rank. Match the
// retained host association exactly: round the rank sum first, round the
// shared output independently, then round their final sum.
// `tier_partials` is null unless the routed-expert tier ran overlapped, in
// which case its contribution could not be accumulated into the rank partials
// directly: the tier stream and the rank-partial upload would then be writing
// the same buffer concurrently. It is summed into the first rank's term, which
// is where the serial ordering accumulated it, before the same rounding.
// Cross-slot order inside the tier's own sum is unfixed either way -- the
// down kernel accumulates its slots by atomicAdd.
__global__ void dsv4_host_moe_join_kernel(
    float* shared_and_output, const float* rank_partials,
    const float* tier_partials, std::uint64_t hidden_columns,
    unsigned int* error_flag) {
    const auto column = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (column >= hidden_columns) return;
    const float first = tier_partials == nullptr
        ? rank_partials[column]
        : rank_partials[column] + tier_partials[column];
    const float routed = bf16_round(
        first + rank_partials[hidden_columns + column]);
    const float shared = bf16_round(shared_and_output[column]);
    const float output = bf16_round(routed + shared);
    shared_and_output[column] = output;
    if (!isfinite(output)) atomicExch(error_flag, 3U);
}

__global__ void dsv4_host_moe_join_mhc_kernel(
    float* shared_and_output, const float* rank_partials,
    const float* tier_partials, __nv_bfloat16* mhc_branch,
    std::uint64_t hidden_columns, unsigned int* error_flag) {
    const auto column = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (column >= hidden_columns) return;
    const float first = tier_partials == nullptr
        ? rank_partials[column]
        : rank_partials[column] + tier_partials[column];
    const float routed = bf16_round(
        first + rank_partials[hidden_columns + column]);
    const float shared = bf16_round(shared_and_output[column]);
    const float output = bf16_round(routed + shared);
    shared_and_output[column] = output;
    mhc_branch[column] = __float2bfloat16_rn(output);
    if (!isfinite(output)) atomicExch(error_flag, 3U);
}

// Row-tiled shared expert, the FP8 counterpart of the routed page kernels
// above. The shared expert fires for every row of a page, so its triplet is
// read once per tile instead of once per row.
__global__ void deepseek_fp8_page_gate_up_kernel(
    float* activations, const float* hidden, const std::uint32_t* work_rows,
    std::uint32_t work_count, const unsigned char* w1,
    const unsigned char* w1_scales, const unsigned char* w3,
    const unsigned char* w3_scales, std::uint64_t columns,
    std::uint64_t intermediate, std::uint64_t scale_columns,
    float swiglu_limit, const float* bf16_silu, unsigned int* error_flag) {
    const std::uint64_t output_row = blockIdx.x;
    if (output_row >= intermediate) return;
    const std::uint32_t tile_begin = blockIdx.y * kDeepSeekPageRowTile;
    if (tile_begin >= work_count) return;
    const std::uint32_t tile_rows =
        min(kDeepSeekPageRowTile, work_count - tile_begin);

    const std::uint64_t weight_base = output_row * columns;
    const std::uint64_t scale_row = (output_row / 128U) * scale_columns;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint64_t groups = (columns + 31U) / 32U;
    float gate[kDeepSeekPageRowTile];
    float up[kDeepSeekPageRowTile];
    for (std::uint32_t index = 0U; index < kDeepSeekPageRowTile; ++index) {
        gate[index] = 0.0F;
        up[index] = 0.0F;
    }

    for (std::uint64_t group = warp; group < groups; group += 8U) {
        const std::uint64_t column = group * 32U + lane;
        float gate_scale =
            lane == 0U ? fp8_e8m0_scale_bits(
                             w1_scales[scale_row + (group * 32U) / 128U])
                       : 0.0F;
        float up_scale =
            lane == 0U ? fp8_e8m0_scale_bits(
                             w3_scales[scale_row + (group * 32U) / 128U])
                       : 0.0F;
        gate_scale = __shfl_sync(0xFFFF'FFFFU, gate_scale, 0);
        up_scale = __shfl_sync(0xFFFF'FFFFU, up_scale, 0);
        if (column >= columns) continue;
        const float gate_weight = fp8_e4m3_value(w1[weight_base + column]);
        const float up_weight = fp8_e4m3_value(w3[weight_base + column]);
#pragma unroll
        for (std::uint32_t index = 0U; index < kDeepSeekPageRowTile; ++index) {
            const std::uint32_t local = index < tile_rows ? index : 0U;
            const std::uint64_t row = work_rows[tile_begin + local];
            const float input = hidden[row * columns + column];
            gate[index] += input * gate_weight * gate_scale;
            up[index] += input * up_weight * up_scale;
        }
    }

#pragma unroll 1
    for (std::uint32_t index = 0U; index < kDeepSeekPageRowTile; ++index) {
        __syncthreads();
        const float reduced_gate = reduce_block(gate[index]);
        __syncthreads();
        const float reduced_up = reduce_block(up[index]);
        if (threadIdx.x != 0U || index >= tile_rows) continue;
        const std::uint64_t destination =
            static_cast<std::uint64_t>(tile_begin + index) * intermediate +
            output_row;
        const float rounded_gate = bf16_round(reduced_gate);
        const float rounded_up = bf16_round(reduced_up);
        if (!isfinite(rounded_gate) || !isfinite(rounded_up)) {
            atomicExch(error_flag, 1U);
            activations[destination] = __uint_as_float(0x7FC0'0000U);
            continue;
        }
        const float limited_gate = fminf(rounded_gate, swiglu_limit);
        const float limited_up =
            fmaxf(-swiglu_limit, fminf(rounded_up, swiglu_limit));
        const auto gate_bits =
            static_cast<std::uint16_t>(__float_as_uint(limited_gate) >> 16U);
        activations[destination] =
            bf16_round(bf16_silu[gate_bits] * limited_up);
    }
}

__global__ void deepseek_fp8_page_down_kernel(
    float* output, const float* activations, std::uint32_t work_count,
    const unsigned char* weights, const unsigned char* scales,
    std::uint64_t columns, std::uint64_t rows, std::uint64_t scale_columns) {
    const std::uint64_t output_row = blockIdx.x;
    if (output_row >= rows) return;
    const std::uint32_t tile_begin = blockIdx.y * kDeepSeekPageRowTile;
    if (tile_begin >= work_count) return;
    const std::uint32_t tile_rows =
        min(kDeepSeekPageRowTile, work_count - tile_begin);

    const std::uint64_t weight_base = output_row * columns;
    const std::uint64_t scale_row = (output_row / 128U) * scale_columns;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint64_t groups = (columns + 31U) / 32U;
    float sum[kDeepSeekPageRowTile];
    for (std::uint32_t index = 0U; index < kDeepSeekPageRowTile; ++index) {
        sum[index] = 0.0F;
    }

    for (std::uint64_t group = warp; group < groups; group += 8U) {
        const std::uint64_t column = group * 32U + lane;
        float scale =
            lane == 0U
                ? fp8_e8m0_scale_bits(scales[scale_row + (group * 32U) / 128U])
                : 0.0F;
        scale = __shfl_sync(0xFFFF'FFFFU, scale, 0);
        if (column >= columns) continue;
        const float weight = fp8_e4m3_value(weights[weight_base + column]);
#pragma unroll
        for (std::uint32_t index = 0U; index < kDeepSeekPageRowTile; ++index) {
            const std::uint32_t local = index < tile_rows ? index : 0U;
            const std::uint64_t slot = tile_begin + local;
            sum[index] += activations[slot * columns + column] * weight * scale;
        }
    }

#pragma unroll 1
    for (std::uint32_t index = 0U; index < kDeepSeekPageRowTile; ++index) {
        __syncthreads();
        const float reduced = reduce_block(sum[index]);
        if (threadIdx.x != 0U || index >= tile_rows) continue;
        output[static_cast<std::uint64_t>(tile_begin + index) * rows +
               output_row] = bf16_round(reduced);
    }
}

// Decode-oriented FlashAttention-2 forward specialization. One CTA owns one
// query/head row while K/V are streamed in bounded tiles. Scores never leave
// registers/shared memory; the running maximum, denominator, and output are
// rescaled at every tile boundary.
__global__ void flash_attention_forward_kernel(
    float* output, const float* queries, const float* keys, const float* values,
    const float* sinks, const std::uint32_t* causal_key_counts,
    std::uint32_t query_rows, std::uint32_t query_heads,
    std::uint32_t key_value_heads, std::uint32_t query_key_dim,
    std::uint32_t value_dim, std::uint32_t key_rows, float scale,
    unsigned int* error_flag) {
    constexpr std::uint32_t tile_rows = 32U;
    constexpr std::uint32_t threads = 256U;
    constexpr std::uint32_t values_per_thread = 4U;
    const auto head = static_cast<std::uint32_t>(blockIdx.x);
    const auto query_row = static_cast<std::uint32_t>(blockIdx.y);
    if (head >= query_heads || query_row >= query_rows) return;
    const auto heads_per_kv = query_heads / key_value_heads;
    const auto kv_head = head / heads_per_kv;
    const auto visible_rows = causal_key_counts == nullptr
        ? key_rows : causal_key_counts[query_row];
    const auto* query = queries +
        (static_cast<std::uint64_t>(query_row) * query_heads + head) *
            query_key_dim;
    double accumulator[values_per_thread]{0.0, 0.0, 0.0, 0.0};
    __shared__ double scores[tile_rows];
    __shared__ double running_maximum;
    __shared__ double denominator;
    __shared__ double correction;
    if (threadIdx.x == 0U) {
        running_maximum = sinks == nullptr ? -INFINITY : sinks[head];
        denominator = sinks == nullptr ? 0.0 : 1.0;
    }
    __syncthreads();

    for (std::uint32_t tile = 0U; tile < visible_rows; tile += tile_rows) {
        const auto count = min(tile_rows, visible_rows - tile);
        for (std::uint32_t item = 0U; item < count; ++item) {
            const auto row = tile + item;
            const auto* key = keys +
                (static_cast<std::uint64_t>(row) * key_value_heads + kv_head) *
                    query_key_dim;
            double dot = 0.0;
            for (std::uint32_t dimension = threadIdx.x;
                 dimension < query_key_dim; dimension += blockDim.x) {
                dot += static_cast<double>(query[dimension]) * key[dimension];
            }
            dot = reduce_block_double(dot);
            if (threadIdx.x == 0U) {
                scores[item] = dot * static_cast<double>(scale);
                if (!isfinite(scores[item])) atomicExch(error_flag, 1U);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0U) {
            double tile_maximum = -INFINITY;
            for (std::uint32_t item = 0U; item < count; ++item) {
                tile_maximum = fmax(tile_maximum, scores[item]);
            }
            const double next_maximum = fmax(running_maximum, tile_maximum);
            correction = denominator == 0.0
                ? 0.0 : exp(running_maximum - next_maximum);
            denominator *= correction;
            for (std::uint32_t item = 0U; item < count; ++item) {
                scores[item] = exp(scores[item] - next_maximum);
                denominator += scores[item];
            }
            running_maximum = next_maximum;
            if (!isfinite(denominator) || denominator <= 0.0F) {
                atomicExch(error_flag, 2U);
            }
        }
        __syncthreads();
        for (auto& value : accumulator) value *= correction;
        for (std::uint32_t item = 0U; item < count; ++item) {
            const auto row = tile + item;
            const auto* value = values +
                (static_cast<std::uint64_t>(row) * key_value_heads + kv_head) *
                    value_dim;
#pragma unroll
            for (std::uint32_t slot = 0U; slot < values_per_thread; ++slot) {
                const auto dimension = threadIdx.x + slot * threads;
                if (dimension < value_dim) {
                    accumulator[slot] += scores[item] * value[dimension];
                }
            }
        }
        __syncthreads();
    }

    auto* destination = output +
        (static_cast<std::uint64_t>(query_row) * query_heads + head) * value_dim;
#pragma unroll
    for (std::uint32_t slot = 0U; slot < values_per_thread; ++slot) {
        const auto dimension = threadIdx.x + slot * threads;
        if (dimension < value_dim) {
            const float value = static_cast<float>(accumulator[slot] / denominator);
            destination[dimension] = value;
            if (!isfinite(value)) atomicExch(error_flag, 3U);
        }
    }
}

__device__ double flash_attention_sequential_dot(
    const float* query, const float* key, std::uint32_t dimensions) {
    double dot = 0.0;
    for (std::uint32_t dimension = 0U; dimension < dimensions; ++dimension) {
        dot = __dadd_rn(dot, __dmul_rn(
            static_cast<double>(query[dimension]),
            static_cast<double>(key[dimension])));
    }
    return dot;
}

// Decode specialization for model oracles whose public numerical contract
// predates online softmax: every key row owns one CUDA thread, but its F64 dot
// remains sequential and therefore bit-compatible with the scalar oracle.
// Scores are transient bounded scratch, then thread zero performs the original
// ordered global softmax while value dimensions accumulate in parallel.
__global__ void flash_attention_reference_f32_kernel(
    float* output, const float* queries, const float* keys, const float* values,
    float* score_scratch,
    const float* sinks, const std::uint32_t* causal_key_counts,
    const std::uint8_t* query_key_mask,
    std::uint32_t query_rows, std::uint32_t query_heads,
    std::uint32_t key_value_heads, std::uint32_t query_key_dim,
    std::uint32_t value_dim, std::uint32_t key_rows, float scale,
    unsigned int* error_flag, std::uint32_t exponential_capacity) {
    constexpr std::uint32_t threads = 256U;
    constexpr std::uint32_t values_per_thread = 4U;
    // Softmax scratch for the block-parallel path below: one double per key
    // row, holding exp(score - maximum) so the exponentials are evaluated once,
    // by every thread, instead of twice by thread 0.
    extern __shared__ double flash_attention_exponentials[];
    const auto head = static_cast<std::uint32_t>(blockIdx.x);
    const auto query_row = static_cast<std::uint32_t>(blockIdx.y);
    if (head >= query_heads || query_row >= query_rows) return;
    const auto heads_per_kv = query_heads / key_value_heads;
    const auto kv_head = head / heads_per_kv;
    const auto visible_rows = causal_key_counts == nullptr
        ? key_rows : causal_key_counts[query_row];
    const auto* key_mask = query_key_mask == nullptr
        ? nullptr
        : query_key_mask + static_cast<std::uint64_t>(query_row) * key_rows;
    const auto* query = queries +
        (static_cast<std::uint64_t>(query_row) * query_heads + head) *
            query_key_dim;
    float accumulator[values_per_thread]{0.0F, 0.0F, 0.0F, 0.0F};
    __shared__ float maximum;
    __shared__ double denominator;

    auto* scores = score_scratch +
        (static_cast<std::uint64_t>(query_row) * query_heads + head) * key_rows;
    for (std::uint32_t row = threadIdx.x; row < visible_rows;
         row += blockDim.x) {
        if (key_mask != nullptr && key_mask[row] == 0U) continue;
        const auto* key = keys +
            (static_cast<std::uint64_t>(row) * key_value_heads + kv_head) *
                query_key_dim;
        const float score = __fmul_rn(static_cast<float>(
            flash_attention_sequential_dot(query, key, query_key_dim)), scale);
        scores[row] = score;
        if (!isfinite(score)) atomicExch(error_flag, 1U);
    }
    __syncthreads();

    // The softmax below is split so that only the one step whose result depends
    // on evaluation order stays on a single thread. `fmaxf` ignores NaN from
    // either side, so the maximum is order independent and reduces; `exp` and
    // the final divide are per row. The denominator is a sequential
    // `__dadd_rn` fold over rows and stays exactly that, reading exponentials
    // the block already computed -- so every emitted float is unchanged, and
    // thread 0's work drops from 2 * visible_rows double exponentials to
    // visible_rows double adds.
    const bool block_softmax = exponential_capacity >= visible_rows;
    if (block_softmax) {
        __shared__ float warp_maxima[threads / 32U];
        float local = -INFINITY;
        for (std::uint32_t row = threadIdx.x; row < visible_rows;
             row += blockDim.x) {
            if (key_mask != nullptr && key_mask[row] == 0U) continue;
            local = fmaxf(local, scores[row]);
        }
#pragma unroll
        for (std::uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
            local = fmaxf(local, __shfl_down_sync(0xFFFFFFFFU, local, offset));
        }
        if ((threadIdx.x & 31U) == 0U) warp_maxima[threadIdx.x >> 5U] = local;
        __syncthreads();
        if (threadIdx.x == 0U) {
            float value = sinks == nullptr ? -INFINITY : sinks[head];
#pragma unroll
            for (std::uint32_t warp = 0U; warp < threads / 32U; ++warp) {
                value = fmaxf(value, warp_maxima[warp]);
            }
            maximum = value;
        }
        __syncthreads();
        for (std::uint32_t row = threadIdx.x; row < visible_rows;
             row += blockDim.x) {
            if (key_mask != nullptr && key_mask[row] == 0U) continue;
            flash_attention_exponentials[row] = exp(static_cast<double>(
                __fsub_rn(scores[row], maximum)));
        }
        __syncthreads();
        if (threadIdx.x == 0U) {
            double total = sinks == nullptr
                ? 0.0
                : exp(static_cast<double>(__fsub_rn(sinks[head], maximum)));
            for (std::uint32_t row = 0U; row < visible_rows; ++row) {
                if (key_mask != nullptr && key_mask[row] == 0U) continue;
                total = __dadd_rn(total, flash_attention_exponentials[row]);
            }
            if (!isfinite(total) || total <= 0.0) atomicExch(error_flag, 2U);
            denominator = total;
        }
        __syncthreads();
        for (std::uint32_t row = threadIdx.x; row < visible_rows;
             row += blockDim.x) {
            if (key_mask != nullptr && key_mask[row] == 0U) continue;
            scores[row] = static_cast<float>(
                flash_attention_exponentials[row] / denominator);
        }
    } else if (threadIdx.x == 0U) {
        maximum = sinks == nullptr ? -INFINITY : sinks[head];
        for (std::uint32_t row = 0U; row < visible_rows; ++row) {
            if (key_mask != nullptr && key_mask[row] == 0U) continue;
            maximum = fmaxf(maximum, scores[row]);
        }
        denominator = sinks == nullptr
            ? 0.0
            : exp(static_cast<double>(__fsub_rn(sinks[head], maximum)));
        for (std::uint32_t row = 0U; row < visible_rows; ++row) {
            if (key_mask != nullptr && key_mask[row] == 0U) continue;
            denominator = __dadd_rn(denominator, exp(static_cast<double>(
                __fsub_rn(scores[row], maximum))));
        }
        if (!isfinite(denominator) || denominator <= 0.0) {
            atomicExch(error_flag, 2U);
        }
        for (std::uint32_t row = 0U; row < visible_rows; ++row) {
            if (key_mask != nullptr && key_mask[row] == 0U) continue;
            scores[row] = static_cast<float>(exp(static_cast<double>(
                __fsub_rn(scores[row], maximum))) / denominator);
        }
    }
    __syncthreads();

    for (std::uint32_t row = 0U; row < visible_rows; ++row) {
        if (key_mask != nullptr && key_mask[row] == 0U) continue;
        const auto* value = values +
            (static_cast<std::uint64_t>(row) * key_value_heads + kv_head) *
                value_dim;
#pragma unroll
        for (std::uint32_t slot = 0U; slot < values_per_thread; ++slot) {
            const auto dimension = threadIdx.x + slot * threads;
            if (dimension < value_dim) {
                accumulator[slot] = __fadd_rn(
                    accumulator[slot],
                    __fmul_rn(scores[row], value[dimension]));
            }
        }
    }

    auto* destination = output +
        (static_cast<std::uint64_t>(query_row) * query_heads + head) * value_dim;
#pragma unroll
    for (std::uint32_t slot = 0U; slot < values_per_thread; ++slot) {
        const auto dimension = threadIdx.x + slot * threads;
        if (dimension < value_dim) {
            destination[dimension] = accumulator[slot];
            if (!isfinite(accumulator[slot])) atomicExch(error_flag, 3U);
        }
    }
}

__device__ float flash_attention_sequential_dot_f32(
    const float* query, const float* key, std::uint32_t dimensions) {
    float dot = 0.0F;
    for (std::uint32_t dimension = 0U; dimension < dimensions; ++dimension) {
        dot = __fadd_rn(dot, __fmul_rn(query[dimension], key[dimension]));
    }
    return dot;
}

// F32 compatibility specialization used by adapters whose scalar oracle has
// an F32 dot, exp, denominator, probability, and V accumulation contract.
__global__ void flash_attention_reference_all_f32_kernel(
    float* output, const float* queries, const float* keys, const float* values,
    const float* sinks, const std::uint32_t* causal_key_counts,
    const std::uint8_t* query_key_mask,
    std::uint32_t query_rows, std::uint32_t query_heads,
    std::uint32_t key_value_heads, std::uint32_t query_key_dim,
    std::uint32_t value_dim, std::uint32_t key_rows, float scale,
    unsigned int* error_flag) {
    constexpr std::uint32_t threads = 256U;
    constexpr std::uint32_t values_per_thread = 4U;
    const auto head = static_cast<std::uint32_t>(blockIdx.x);
    const auto query_row = static_cast<std::uint32_t>(blockIdx.y);
    if (head >= query_heads || query_row >= query_rows) return;
    const auto heads_per_kv = query_heads / key_value_heads;
    const auto kv_head = head / heads_per_kv;
    const auto visible_rows = causal_key_counts == nullptr
        ? key_rows : causal_key_counts[query_row];
    const auto* key_mask = query_key_mask == nullptr
        ? nullptr
        : query_key_mask + static_cast<std::uint64_t>(query_row) * key_rows;
    const auto* query = queries +
        (static_cast<std::uint64_t>(query_row) * query_heads + head) *
            query_key_dim;
    float accumulator[values_per_thread]{0.0F, 0.0F, 0.0F, 0.0F};
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float probability;

    if (threadIdx.x == 0U) {
        maximum = sinks == nullptr ? -INFINITY : sinks[head];
        for (std::uint32_t row = 0U; row < visible_rows; ++row) {
            if (key_mask != nullptr && key_mask[row] == 0U) continue;
            const auto* key = keys +
                (static_cast<std::uint64_t>(row) * key_value_heads + kv_head) *
                    query_key_dim;
            const float score = __fmul_rn(
                flash_attention_sequential_dot_f32(
                    query, key, query_key_dim), scale);
            if (!isfinite(score)) atomicExch(error_flag, 1U);
            maximum = fmaxf(maximum, score);
        }
        denominator = sinks == nullptr
            ? 0.0F : expf(__fsub_rn(sinks[head], maximum));
        for (std::uint32_t row = 0U; row < visible_rows; ++row) {
            if (key_mask != nullptr && key_mask[row] == 0U) continue;
            const auto* key = keys +
                (static_cast<std::uint64_t>(row) * key_value_heads + kv_head) *
                    query_key_dim;
            const float score = __fmul_rn(
                flash_attention_sequential_dot_f32(
                    query, key, query_key_dim), scale);
            denominator = __fadd_rn(
                denominator, expf(__fsub_rn(score, maximum)));
        }
        if (!isfinite(denominator) || denominator <= 0.0F) {
            atomicExch(error_flag, 2U);
        }
    }
    __syncthreads();

    for (std::uint32_t row = 0U; row < visible_rows; ++row) {
        if (key_mask != nullptr && key_mask[row] == 0U) continue;
        if (threadIdx.x == 0U) {
            const auto* key = keys +
                (static_cast<std::uint64_t>(row) * key_value_heads + kv_head) *
                    query_key_dim;
            const float score = __fmul_rn(
                flash_attention_sequential_dot_f32(
                    query, key, query_key_dim), scale);
            probability = __fdiv_rn(
                expf(__fsub_rn(score, maximum)), denominator);
        }
        __syncthreads();
        const auto* value = values +
            (static_cast<std::uint64_t>(row) * key_value_heads + kv_head) *
                value_dim;
#pragma unroll
        for (std::uint32_t slot = 0U; slot < values_per_thread; ++slot) {
            const auto dimension = threadIdx.x + slot * threads;
            if (dimension < value_dim) {
                accumulator[slot] = __fadd_rn(
                    accumulator[slot],
                    __fmul_rn(probability, value[dimension]));
            }
        }
        __syncthreads();
    }

    auto* destination = output +
        (static_cast<std::uint64_t>(query_row) * query_heads + head) * value_dim;
#pragma unroll
    for (std::uint32_t slot = 0U; slot < values_per_thread; ++slot) {
        const auto dimension = threadIdx.x + slot * threads;
        if (dimension < value_dim) {
            destination[dimension] = accumulator[slot];
            if (!isfinite(accumulator[slot])) atomicExch(error_flag, 3U);
        }
    }
}

__device__ float bf16_kv_sequential_dot_f32(
    const float* query, const __nv_bfloat16* key,
    std::uint32_t dimensions) {
    float dot = 0.0F;
    for (std::uint32_t dimension = 0U; dimension < dimensions; ++dimension) {
        dot = __fadd_rn(
            dot, __fmul_rn(query[dimension],
                           __bfloat162float(key[dimension])));
    }
    return dot;
}

// Batch-1 counterpart of flash_attention_reference_all_f32_kernel over a
// persistent two-plane BF16 KV ring. Reading a BF16 cache element through
// __bfloat162float produces exactly the F32 fixed point the host compatibility
// path used to upload, while preserving every sequential reduction order.
__global__ void bf16_kv_attention_reference_all_f32_kernel(
    float* output, float* scores, const float* queries,
    const __nv_bfloat16* keys,
    const __nv_bfloat16* values, const float* relative_bias,
    std::uint32_t relative_bias_extent, std::uint32_t query_heads,
    std::uint32_t key_value_heads, std::uint32_t head_dim,
    std::uint32_t capacity_rows, std::uint32_t cache_start,
    std::uint32_t cached_rows, float scale, unsigned int* error_flag) {
    constexpr std::uint32_t threads = 256U;
    const auto head = static_cast<std::uint32_t>(blockIdx.x);
    if (head >= query_heads) return;
    const auto heads_per_kv = query_heads / key_value_heads;
    const auto kv_head = head / heads_per_kv;
    const auto* query = queries + static_cast<std::uint64_t>(head) * head_dim;
    auto* head_scores = scores +
        static_cast<std::uint64_t>(head) * capacity_rows;
    __shared__ float maximum;
    __shared__ float denominator;

    // Rows are independent. Parallelizing them preserves the reference dot's
    // dimension order while removing the old three serial score passes from
    // thread zero. Softmax and value accumulation below retain their original
    // logical-row order exactly.
    for (std::uint32_t row = threadIdx.x; row < cached_rows;
         row += blockDim.x) {
        const auto physical = (cache_start + row) % capacity_rows;
        const auto* key = keys +
            (static_cast<std::uint64_t>(physical) * key_value_heads + kv_head) *
                head_dim;
        float score = __fmul_rn(
            bf16_kv_sequential_dot_f32(query, key, head_dim), scale);
        const auto distance = cached_rows - 1U - row;
        if (distance < relative_bias_extent) {
            score = __fadd_rn(
                score, relative_bias[
                    static_cast<std::uint64_t>(head) * relative_bias_extent +
                    distance]);
        }
        head_scores[row] = score;
        if (!isfinite(head_scores[row])) atomicExch(error_flag, 1U);
    }
    __syncthreads();

    if (threadIdx.x == 0U) {
        maximum = -INFINITY;
        for (std::uint32_t row = 0U; row < cached_rows; ++row) {
            maximum = fmaxf(maximum, head_scores[row]);
        }
        denominator = 0.0F;
        for (std::uint32_t row = 0U; row < cached_rows; ++row) {
            head_scores[row] = expf(__fsub_rn(head_scores[row], maximum));
            denominator = __fadd_rn(denominator, head_scores[row]);
        }
        if (!isfinite(denominator) || denominator <= 0.0F) {
            atomicExch(error_flag, 2U);
        }
        for (std::uint32_t row = 0U; row < cached_rows; ++row) {
            head_scores[row] = __fdiv_rn(head_scores[row], denominator);
        }
    }
    __syncthreads();

    auto* destination = output + static_cast<std::uint64_t>(head) * head_dim;
    for (std::uint32_t dimension = threadIdx.x; dimension < head_dim;
         dimension += threads) {
        float accumulator = 0.0F;
        for (std::uint32_t row = 0U; row < cached_rows; ++row) {
            const auto physical = (cache_start + row) % capacity_rows;
            const auto* value = values +
                (static_cast<std::uint64_t>(physical) * key_value_heads +
                 kv_head) * head_dim;
            accumulator = __fadd_rn(
                accumulator,
                __fmul_rn(head_scores[row],
                           __bfloat162float(value[dimension])));
        }
        destination[dimension] = accumulator;
        if (!isfinite(accumulator)) atomicExch(error_flag, 3U);
    }
}

constexpr std::uint32_t kGlmHeads = 64U;
constexpr std::uint32_t kGlmNope = 192U;
constexpr std::uint32_t kGlmRope = 64U;
constexpr std::uint32_t kGlmValue = 256U;
constexpr std::uint32_t kGlmLatent = 512U;

__device__ float glm_int4_product(
    float activation, const std::uint32_t* packed,
    const __nv_bfloat16* scales, std::uint32_t row,
    std::uint32_t column) {
    constexpr std::uint32_t packed_columns = kGlmLatent / 8U;
    constexpr std::uint32_t scale_columns = kGlmLatent / 128U;
    const auto word = packed[static_cast<std::uint64_t>(row) * packed_columns +
                             column / 8U];
    const auto raw = (word >> ((column % 8U) * 4U)) & 0x0FU;
    const float quantized = static_cast<float>(static_cast<int>(raw) - 8);
    const float scale = __bfloat162float(
        scales[static_cast<std::uint64_t>(row) * scale_columns +
               column / 128U]);
    return __fmul_rn(__fmul_rn(activation, quantized), scale);
}

__global__ void glm_absorbed_attention_kernel(
    float* output, const float* queries, const float* latent,
    const float* rope, const std::uint32_t* causal_key_counts,
    const std::uint32_t* packed, const __nv_bfloat16* scales,
    std::uint32_t query_rows, std::uint32_t key_rows, float attention_scale,
    unsigned int* error_flag) {
    const auto head = static_cast<std::uint32_t>(blockIdx.x);
    const auto query_row = static_cast<std::uint32_t>(blockIdx.y);
    if (head >= kGlmHeads || query_row >= query_rows) return;
    const auto visible_rows = causal_key_counts[query_row];
    const auto* query = queries +
        (static_cast<std::uint64_t>(query_row) * kGlmHeads + head) *
            (kGlmNope + kGlmRope);
    const auto weight_row = head * (kGlmNope + kGlmValue);
    extern __shared__ float scratch[];
    auto* absorbed_query = scratch;
    auto* context_latent = absorbed_query + kGlmLatent;
    auto* scores = context_latent + kGlmLatent;

    for (std::uint32_t column = threadIdx.x; column < kGlmLatent;
         column += blockDim.x) {
        float sum = 0.0F;
        for (std::uint32_t dimension = 0U; dimension < kGlmNope;
             ++dimension) {
            sum = __fadd_rn(sum, glm_int4_product(
                query[dimension], packed, scales,
                weight_row + dimension, column));
        }
        absorbed_query[column] = sum;
    }
    __syncthreads();

    for (std::uint32_t row = threadIdx.x; row < visible_rows;
         row += blockDim.x) {
        float score = 0.0F;
        const auto* latent_row = latent +
            static_cast<std::uint64_t>(row) * kGlmLatent;
        const auto* rope_row = rope +
            static_cast<std::uint64_t>(row) * kGlmRope;
        for (std::uint32_t column = 0U; column < kGlmLatent; ++column) {
            score = __fadd_rn(score, __fmul_rn(
                absorbed_query[column], latent_row[column]));
        }
        for (std::uint32_t dimension = 0U; dimension < kGlmRope;
             ++dimension) {
            score = __fadd_rn(score, __fmul_rn(
                query[kGlmNope + dimension], rope_row[dimension]));
        }
        scores[row] = __fmul_rn(score, attention_scale);
        if (!isfinite(scores[row])) atomicExch(error_flag, 1U);
    }
    __syncthreads();

    if (threadIdx.x == 0U) {
        float maximum = scores[0];
        for (std::uint32_t row = 1U; row < visible_rows; ++row) {
            maximum = fmaxf(maximum, scores[row]);
        }
        float denominator = 0.0F;
        for (std::uint32_t row = 0U; row < visible_rows; ++row) {
            scores[row] = expf(__fsub_rn(scores[row], maximum));
            denominator = __fadd_rn(denominator, scores[row]);
        }
        if (!isfinite(denominator) || denominator <= 0.0F) {
            atomicExch(error_flag, 2U);
            denominator = 1.0F;
        }
        for (std::uint32_t row = 0U; row < visible_rows; ++row) {
            scores[row] = __fdiv_rn(scores[row], denominator);
        }
    }
    __syncthreads();

    for (std::uint32_t column = threadIdx.x; column < kGlmLatent;
         column += blockDim.x) {
        float sum = 0.0F;
        for (std::uint32_t row = 0U; row < visible_rows; ++row) {
            sum = __fadd_rn(sum, __fmul_rn(
                scores[row], latent[static_cast<std::uint64_t>(row) *
                                    kGlmLatent + column]));
        }
        context_latent[column] = sum;
    }
    __syncthreads();

    if (threadIdx.x < kGlmValue) {
        float sum = 0.0F;
        const auto row = weight_row + kGlmNope + threadIdx.x;
        for (std::uint32_t column = 0U; column < kGlmLatent; ++column) {
            sum = __fadd_rn(sum, glm_int4_product(
                context_latent[column], packed, scales, row, column));
        }
        output[(static_cast<std::uint64_t>(query_row) * kGlmHeads + head) *
                   kGlmValue + threadIdx.x] = sum;
        if (!isfinite(sum)) atomicExch(error_flag, 3U);
    }
    (void)key_rows;
}

constexpr std::uint32_t kDsv4PagedHeads = 32U;
constexpr std::uint32_t kDsv4PagedHeadDim = 512U;
constexpr std::uint32_t kDsv4PagedCandidateBlock = 128U;
constexpr std::uint32_t kDsv4PagedDimensionsPerBlock = 256U;
constexpr std::uint32_t kDsv4PagedCandidateGroups = 4U;
constexpr std::uint32_t kDsv4PagedCandidatesPerGroup = 32U;

struct Dsv4DevicePhysicalPage {
    const std::uint8_t* data{};
    std::uint32_t rows{};
    std::uint32_t flat_begin{};
};

struct Dsv4DeviceAttentionCandidate {
    std::uint32_t page{};
    std::uint32_t row{};
    std::uint32_t valid{};
};

// Status bits an in-chain resolution can raise. Bit 0 is left to the plain
// failure every other DeepSeek attention kernel already reports.
constexpr unsigned int kDsv4ResolveSelectionRejected = 1U << 1U;
constexpr unsigned int kDsv4ResolveRowUnowned = 1U << 2U;
constexpr unsigned int kDsv4ResolveOutsidePage = 1U << 3U;

struct Dsv4DeviceKvBlock {
    std::uint64_t logical_begin{};
    std::uint32_t used_rows{};
    std::uint32_t compression_ratio{};
};

// Resolves each selected logical row to its physical page and row without the
// host having seen the selection. This is locate_physical_kv_block()'s three
// tiers -- uniform guess, binary search, exhaustive scan -- with the same
// ownership predicate validated on every path, so a row resolves to the block
// the host would have chosen or to nothing at all.
//
// The page index is the block-table index: under device selection there is no
// first-touch order to compact against. The bounds the host checks per
// candidate are checked here too, against the same uploaded page descriptors,
// so an out-of-range resolution fails the command instead of reading a page it
// does not own.
//
// Screened against the host over three geometries and 516 probes each -- short
// final blocks, block boundaries, ratio 4 and ratio 128, at 2,685 and
// 1,048,576 tokens -- with zero block and zero row mismatches on both
// architectures.
__global__ void dsv4_resolve_candidates_kernel(
    const std::uint32_t* selected, std::uint32_t selected_count,
    const Dsv4DeviceKvBlock* blocks, std::uint32_t block_count,
    const Dsv4DevicePhysicalPage* pages, std::uint32_t page_count,
    Dsv4DeviceAttentionCandidate* candidates, std::uint32_t candidate_width,
    const unsigned int* selection_error, unsigned int* error) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    // The selection ran without a host boundary, so its own rejection has no
    // other way back. Carry it into the command that consumes it.
    //
    // The three causes are distinguished in the status word because they are
    // otherwise indistinguishable from a decode failure downstream, and this
    // command has no host boundary of its own to report at. They are set as
    // bits so a later kernel's plain failure does not hide them.
    if (index == 0U && selection_error != nullptr && *selection_error != 0U) {
        atomicOr(error, kDsv4ResolveSelectionRejected);
    }
    if (index >= candidate_width) return;
    if (index >= selected_count) {
        candidates[index] = {0U, 0U, 0U};
        return;
    }
    const std::uint64_t logical_row = selected[index];

    const auto owns = [&](std::uint32_t slot) {
        const auto& block = blocks[slot];
        if (block.compression_ratio == 0U) return false;
        const auto begin = block.logical_begin / block.compression_ratio;
        return logical_row >= begin && logical_row < begin + block.used_rows;
    };

    std::uint32_t found = block_count;
    if (block_count != 0U) {
        const auto& first = blocks[0];
        if (first.compression_ratio != 0U && first.used_rows != 0U) {
            const auto base = first.logical_begin / first.compression_ratio;
            if (logical_row >= base) {
                const auto guess = static_cast<std::uint32_t>(
                    (logical_row - base) / first.used_rows);
                if (guess < block_count && owns(guess)) found = guess;
            }
        }
    }
    if (found == block_count) {
        std::uint32_t low = 0U;
        std::uint32_t high = block_count;
        while (low < high) {
            const auto middle = low + (high - low) / 2U;
            const auto& block = blocks[middle];
            if (block.compression_ratio == 0U) break;
            const auto begin = block.logical_begin / block.compression_ratio;
            if (logical_row < begin) high = middle;
            else if (logical_row >= begin + block.used_rows) low = middle + 1U;
            else { found = middle; break; }
        }
    }
    if (found == block_count) {
        for (std::uint32_t slot = 0U; slot < block_count; ++slot) {
            if (owns(slot)) { found = slot; break; }
        }
    }
    if (found == block_count) {
        atomicOr(error, kDsv4ResolveRowUnowned);
        candidates[index] = {0U, 0U, 0U};
        return;
    }
    const auto& block = blocks[found];
    const auto begin = block.logical_begin / block.compression_ratio;
    const auto row = static_cast<std::uint32_t>(logical_row - begin);
    if (found >= page_count || row >= pages[found].rows) {
        atomicOr(error, kDsv4ResolveOutsidePage);
        candidates[index] = {0U, 0U, 0U};
        return;
    }
    candidates[index] = {found, row, 1U};
}

__device__ __forceinline__ float dsv4_decode_e4m3fn(
    std::uint8_t code, unsigned int* failure) {
    const auto exponent = static_cast<std::uint32_t>((code >> 3U) & 0x0fU);
    const auto mantissa = static_cast<std::uint32_t>(code & 0x07U);
    if (exponent == 15U && mantissa == 7U) {
        atomicExch(failure, 1U);
        return 0.0F;
    }
    if (exponent == 0U && mantissa == 0U) return 0.0F;
    float value;
    if (exponent == 0U) {
        value = ldexpf(static_cast<float>(mantissa), -9);
    } else {
        value = ldexpf(static_cast<float>(8U + mantissa),
                       static_cast<int>(exponent) - 10);
    }
    return (code & 0x80U) == 0U ? value : -value;
}

__global__ void dsv4_materialize_physical_pages(
    const Dsv4DevicePhysicalPage* pages, std::uint32_t page_count,
    __nv_bfloat16* kv,
    unsigned int* failure) {
    const auto page_index = static_cast<std::uint32_t>(blockIdx.y);
    if (page_index >= page_count) return;
    const auto page = pages[page_index];
    const auto local_index = static_cast<std::uint64_t>(blockIdx.x) *
                                 blockDim.x + threadIdx.x;
    const auto count = static_cast<std::uint64_t>(page.rows) *
                       kDsv4PagedHeadDim;
    if (local_index >= count) return;
    const auto index = static_cast<std::uint64_t>(page.flat_begin) *
                           kDsv4PagedHeadDim + local_index;
    auto* output_bits = reinterpret_cast<std::uint16_t*>(kv);
    const auto row = static_cast<std::uint32_t>(
        local_index / kDsv4PagedHeadDim);
    const auto dimension = static_cast<std::uint32_t>(
        local_index % kDsv4PagedHeadDim);
    if (page.data == nullptr || row >= page.rows) {
        atomicExch(failure, 3U);
        output_bits[index] = 0U;
        return;
    }
    const auto data_offset = static_cast<std::uint64_t>(row) * 576U;
    if (dimension < 448U) {
        const auto scale_code =
            page.data[static_cast<std::uint64_t>(page.rows) * 576U +
                      static_cast<std::uint64_t>(row) * 8U +
                      dimension / 64U];
        if (scale_code == 255U) {
            atomicExch(failure, 4U);
            output_bits[index] = 0U;
            return;
        }
        const auto value = dsv4_decode_e4m3fn(
            page.data[data_offset + dimension], failure);
        const auto scale = ldexpf(1.0F, static_cast<int>(scale_code) - 127);
        kv[index] = __float2bfloat16_rn(__fmul_rn(value, scale));
    } else {
        const auto rope = data_offset + 448U +
            static_cast<std::uint64_t>(dimension - 448U) * 2U;
        output_bits[index] = static_cast<std::uint16_t>(page.data[rope]) |
            (static_cast<std::uint16_t>(page.data[rope + 1U]) << 8U);
    }
}

// Scores only the candidates a row actually attends, instead of every gathered
// KV row. The dense form computed rows x heads x flat_rows and then discarded
// all but 640 entries per row, which is both quadratic in context and the
// reason the score workspace forced sub-chunking. One block owns one row and
// kDsv4SparseScoreHeads heads; the KV row is staged once in shared and every
// warp in the block dots its own head against it.
constexpr std::uint32_t kDsv4SparseScoreHeads = 8U;
__global__ void dsv4_sparse_scores_kernel(
    __nv_bfloat16* scores, const __nv_bfloat16* queries,
    const __nv_bfloat16* kv, const Dsv4DevicePhysicalPage* pages,
    const Dsv4DeviceAttentionCandidate* candidates,
    std::uint32_t candidate_count, std::uint32_t group_offset) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t head = blockIdx.y * kDsv4SparseScoreHeads +
                               (threadIdx.x >> 5U);
    const std::uint32_t lane = threadIdx.x & 31U;
    if (head >= kDsv4PagedHeads) return;
    candidates += static_cast<std::uint64_t>(row) * candidate_count;
    const auto* query = queries +
        (static_cast<std::uint64_t>(group_offset) +
         static_cast<std::uint64_t>(row)) *
            kDsv4PagedHeads * kDsv4PagedHeadDim +
        static_cast<std::uint64_t>(head) * kDsv4PagedHeadDim;
    constexpr std::uint32_t kPerLane = kDsv4PagedHeadDim / 32U;
    float own[kPerLane];
#pragma unroll
    for (std::uint32_t index = 0U; index < kPerLane; ++index) {
        own[index] = __bfloat162float(query[lane * kPerLane + index]);
    }
    __shared__ __nv_bfloat16 staged[kDsv4PagedHeadDim];
    auto* base = scores +
        (static_cast<std::uint64_t>(row) * kDsv4PagedHeads + head) *
            candidate_count;
    for (std::uint32_t candidate = 0U; candidate < candidate_count;
         ++candidate) {
        const auto descriptor = candidates[candidate];
        if (descriptor.valid == 0U) {
            if (lane == 0U) base[candidate] = __float2bfloat16_rn(0.0F);
            __syncthreads();
            continue;
        }
        const auto flat = static_cast<std::uint64_t>(
            pages[descriptor.page].flat_begin + descriptor.row);
        const auto* source = kv + flat * kDsv4PagedHeadDim;
        for (std::uint32_t index = threadIdx.x; index < kDsv4PagedHeadDim;
             index += blockDim.x) {
            staged[index] = source[index];
        }
        __syncthreads();
        float sum = 0.0F;
#pragma unroll
        for (std::uint32_t index = 0U; index < kPerLane; ++index) {
            sum = __fmaf_rn(own[index],
                            __bfloat162float(staged[lane * kPerLane + index]),
                            sum);
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_xor_sync(0xffff'ffffU, sum, offset);
        }
        if (lane == 0U) base[candidate] = __float2bfloat16_rn(sum);
        __syncthreads();
    }
}

__global__ void dsv4_scale_scores(__nv_bfloat16* scores,
                                  std::uint64_t count, float scale) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < count) {
        scores[index] = __float2bfloat16_rn(
            __bfloat162float(scores[index]) * scale);
    }
}

__device__ __forceinline__ float dsv4_warp_max(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value = fmaxf(value,
                      __shfl_down_sync(0xffff'ffffU, value, offset));
    }
    return value;
}

__global__ void dsv4_finish_maximums(
    const __nv_bfloat16* scores,
    const Dsv4DevicePhysicalPage* pages,
    const Dsv4DeviceAttentionCandidate* candidates, const float* sink,
    float* maximums, std::uint32_t candidate_count,
    std::uint32_t score_width, std::uint32_t boundaries) {
    const auto head = blockIdx.x;
    const auto row = blockIdx.y;
    candidates += static_cast<std::uint64_t>(row) * candidate_count;
    maximums += static_cast<std::uint64_t>(row) * kDsv4PagedHeads;
    const auto lane = threadIdx.x & 31U;
    const auto warp = threadIdx.x >> 5U;
    __shared__ float warp_maximums[4];
    __shared__ float running_maximum;
    if (threadIdx.x == 0U) running_maximum = sink[head];
    __syncthreads();
    for (std::uint32_t boundary = 0U; boundary < boundaries; ++boundary) {
        const auto candidate = boundary * kDsv4PagedCandidateBlock +
                               threadIdx.x;
        float value = __int_as_float(0xff80'0000);
        if (candidate < candidate_count && candidates[candidate].valid != 0U) {
            value = __bfloat162float(
                scores[(static_cast<std::uint64_t>(row) *
                            kDsv4PagedHeads + head) * score_width +
                       candidate]);
        }
        value = dsv4_warp_max(value);
        if (lane == 0U) warp_maximums[warp] = value;
        __syncthreads();
        if (threadIdx.x == 0U) {
            float block_maximum = warp_maximums[0];
            for (std::uint32_t index = 1U; index < 4U; ++index) {
                block_maximum = fmaxf(block_maximum, warp_maximums[index]);
            }
            running_maximum = fmaxf(running_maximum, block_maximum);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) maximums[head] = running_maximum;
}

__device__ __forceinline__ float dsv4_triton_exp(float value) {
    const float scaled = value * __int_as_float(0x3fb8'aa3b);
    float result;
    asm("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(scaled));
    return result;
}

__global__ void dsv4_finish_denominators(
    const __nv_bfloat16* scores,
    const Dsv4DevicePhysicalPage* pages,
    const Dsv4DeviceAttentionCandidate* candidates, const float* sink,
    const float* maximums, float* denominators,
    std::uint32_t candidate_count, std::uint32_t score_width,
    std::uint32_t boundaries) {
    const auto head = blockIdx.x;
    const auto row = blockIdx.y;
    candidates += static_cast<std::uint64_t>(row) * candidate_count;
    maximums += static_cast<std::uint64_t>(row) * kDsv4PagedHeads;
    denominators += static_cast<std::uint64_t>(row) * kDsv4PagedHeads;
    const auto lane = threadIdx.x & 31U;
    const auto warp = threadIdx.x >> 5U;
    __shared__ float warp_sums[4];
    __shared__ float running_denominator;
    if (threadIdx.x == 0U) {
        running_denominator = dsv4_triton_exp(sink[head] - maximums[head]);
    }
    __syncthreads();
    for (std::uint32_t boundary = 0U; boundary < boundaries; ++boundary) {
        const auto candidate = boundary * kDsv4PagedCandidateBlock +
                               threadIdx.x;
        float weight = 0.0F;
        if (candidate < candidate_count && candidates[candidate].valid != 0U) {
            const auto score = __bfloat162float(
                scores[(static_cast<std::uint64_t>(row) *
                            kDsv4PagedHeads + head) * score_width +
                       candidate]);
            weight = dsv4_triton_exp(score - maximums[head]);
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            weight += __shfl_xor_sync(0xffff'ffffU, weight, offset);
        }
        if (lane == 0U) warp_sums[warp] = weight;
        __syncthreads();
        float block_sum = threadIdx.x < 4U ? warp_sums[threadIdx.x] : 0.0F;
        block_sum += __shfl_xor_sync(0xffff'ffffU, block_sum, 2);
        block_sum += __shfl_xor_sync(0xffff'ffffU, block_sum, 1);
        if (threadIdx.x == 0U) running_denominator += block_sum;
        __syncthreads();
    }
    if (threadIdx.x == 0U) denominators[head] = running_denominator;
}

__device__ __forceinline__ float dsv4_candidate_value(
    const __nv_bfloat16* kv,
    const Dsv4DevicePhysicalPage* pages,
    const Dsv4DeviceAttentionCandidate* candidates,
    std::uint32_t candidate, std::uint32_t dimension) {
    if (candidates[candidate].valid == 0U) return 0.0F;
    const auto descriptor = candidates[candidate];
    const auto flat = pages[descriptor.page].flat_begin + descriptor.row;
    return __bfloat162float(
        kv[static_cast<std::uint64_t>(flat) * kDsv4PagedHeadDim +
           dimension]);
}

__device__ __forceinline__ float dsv4_candidate_group_sum(
    const __nv_bfloat16* kv, const Dsv4DevicePhysicalPage* pages,
    const float* weights,
    const Dsv4DeviceAttentionCandidate* candidates, std::uint32_t dimension,
    std::uint32_t group) {
    const auto second = group + kDsv4PagedCandidateGroups;
    float sum = __fmul_rn(
        weights[second],
        dsv4_candidate_value(kv, pages, candidates, second, dimension));
    sum = __fmaf_rn(
        weights[group],
        dsv4_candidate_value(kv, pages, candidates, group, dimension), sum);
#pragma unroll
    for (std::uint32_t index = 2U;
         index < kDsv4PagedCandidatesPerGroup; ++index) {
        const auto offset = group + kDsv4PagedCandidateGroups * index;
        sum = __fmaf_rn(
            weights[offset],
            dsv4_candidate_value(
                kv, pages, candidates, offset, dimension), sum);
    }
    return sum;
}

// Divides by the denominator and stores BF16 in place of writing an FP32
// accumulator the next kernel would immediately consume. The division, its
// div.full.f32 form and the BF16 rounding are exactly what dsv4_divide_and_store
// performed, so the stored values are unchanged; only the 62 MB intermediate
// region disappears.
__global__ void dsv4_finish_values(
    const __nv_bfloat16* scores,
    const Dsv4DevicePhysicalPage* pages,
    const Dsv4DeviceAttentionCandidate* candidates, const float* maximums,
    const __nv_bfloat16* kv, const float* denominators,
    __nv_bfloat16* attended, std::uint64_t attended_row_stride,
    std::uint64_t attended_group_offset,
    std::uint32_t candidate_count, std::uint32_t score_width,
    std::uint32_t boundaries) {
    const auto head = blockIdx.x;
    const auto row = blockIdx.z;
    candidates += static_cast<std::uint64_t>(row) * candidate_count;
    maximums += static_cast<std::uint64_t>(row) * kDsv4PagedHeads;
    const auto dimension = blockIdx.y * kDsv4PagedDimensionsPerBlock +
                           threadIdx.x;
    __shared__ float weights[kDsv4PagedCandidateBlock];
    __shared__ Dsv4DeviceAttentionCandidate
        block_candidates[kDsv4PagedCandidateBlock];
    float running_value = 0.0F;
    for (std::uint32_t boundary = 0U; boundary < boundaries; ++boundary) {
        const auto candidate_start = boundary * kDsv4PagedCandidateBlock;
        if (threadIdx.x < kDsv4PagedCandidateBlock) {
            const auto candidate = candidate_start + threadIdx.x;
            const auto descriptor = candidate < candidate_count
                ? candidates[candidate] : Dsv4DeviceAttentionCandidate{};
            block_candidates[threadIdx.x] = descriptor;
            if (descriptor.valid != 0U) {
                const auto score = __bfloat162float(
                    scores[(static_cast<std::uint64_t>(row) *
                                kDsv4PagedHeads + head) * score_width +
                           candidate]);
                weights[threadIdx.x] = dsv4_triton_exp(
                    score - maximums[head]);
            } else {
                weights[threadIdx.x] = 0.0F;
            }
        }
        __syncthreads();
        const auto group0 = dsv4_candidate_group_sum(
            kv, pages, weights, block_candidates, dimension, 0U);
        const auto group1 = dsv4_candidate_group_sum(
            kv, pages, weights, block_candidates, dimension, 1U);
        const auto group2 = dsv4_candidate_group_sum(
            kv, pages, weights, block_candidates, dimension, 2U);
        const auto group3 = dsv4_candidate_group_sum(
            kv, pages, weights, block_candidates, dimension, 3U);
        const auto pair02 = __fadd_rn(group0, group2);
        const auto pair13 = __fadd_rn(group1, group3);
        running_value = __fadd_rn(
            running_value, __fadd_rn(pair02, pair13));
        __syncthreads();
    }
    float divided;
    asm("div.full.f32 %0, %1, %2;"
        : "=f"(divided)
        : "f"(running_value),
          "f"(denominators[static_cast<std::uint64_t>(row) *
                               kDsv4PagedHeads + head]));
    attended[static_cast<std::uint64_t>(row) * attended_row_stride +
             attended_group_offset +
             static_cast<std::uint64_t>(head) * kDsv4PagedHeadDim +
             dimension] = __float2bfloat16_rn(divided);
}

__global__ void dsv4_divide_and_store(
    const float* values, const float* denominators,
    __nv_bfloat16* output, std::uint64_t elements,
    std::uint64_t row_elements, std::uint64_t output_row_stride,
    std::uint64_t output_group_offset) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= elements) return;
    const auto row_offset = index % row_elements;
    const auto row = index / row_elements;
    const auto head = static_cast<std::uint32_t>(
        row_offset / kDsv4PagedHeadDim);
    float divided;
    asm("div.full.f32 %0, %1, %2;"
        : "=f"(divided)
        : "f"(values[index]),
          "f"(denominators[row * kDsv4PagedHeads + head]));
    output[row * output_row_stride + output_group_offset + row_offset] =
        __float2bfloat16_rn(divided);
}

__global__ void dsv4_inverse_rope_decode(
    const __nv_bfloat16* attended, const float* cosines,
    const float* sines, __nv_bfloat16* output, std::uint32_t rows,
    std::uint32_t heads) {
    constexpr std::uint32_t rope = 64U;
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    const auto row_elements = static_cast<std::uint64_t>(heads) *
                              kDsv4PagedHeadDim;
    const auto elements = static_cast<std::uint64_t>(rows) *
                          row_elements;
    if (index >= elements) return;
    const auto dimension = static_cast<std::uint32_t>(
        index % kDsv4PagedHeadDim);
    const auto row = index / row_elements;
    if (dimension < kDsv4PagedHeadDim - rope) {
        output[index] = attended[index];
        return;
    }
    const auto rope_dimension = dimension - (kDsv4PagedHeadDim - rope);
    const auto pair = rope_dimension / 2U;
    const auto pair_begin = index - rope_dimension + pair * 2U;
    const float first = __bfloat162float(attended[pair_begin]);
    const float second = __bfloat162float(attended[pair_begin + 1U]);
    const float cosine = cosines[row * (rope / 2U) + pair];
    const float sine = sines[row * (rope / 2U) + pair];
    const float value = (rope_dimension & 1U) == 0U
        ? __fsub_rn(__fmul_rn(first, cosine), __fmul_rn(second, sine))
        : __fadd_rn(__fmul_rn(second, cosine), __fmul_rn(first, sine));
    // The value was already rounded to BF16 here before being widened into an
    // FP32 region, so storing BF16 is lossless and halves the largest region
    // in the page workspace.
    output[index] = __float2bfloat16_rn(bf16_round(value));
}

__global__ void dsv4_bf16_to_fp32(
    const __nv_bfloat16* input, float* output, std::uint64_t elements) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < elements) output[index] = __bfloat162float(input[index]);
}

__global__ void dsv4_fp32_to_bf16(
    const float* input, __nv_bfloat16* output, std::uint64_t elements) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < elements) output[index] = __float2bfloat16_rn(input[index]);
}

__global__ void dsv4_round_float_bf16(float* values,
                                      std::uint64_t elements) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < elements) values[index] = bf16_round(values[index]);
}

__global__ void dsv4_store_mhc_branch(
    const float* values, __nv_bfloat16* branch,
    std::uint64_t elements) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < elements) branch[index] = __float2bfloat16_rn(values[index]);
}

__device__ __forceinline__ float dsv4_rope_first(
    float first, float second, float cosine, float sine) {
    return __fsub_rn(__fmul_rn(first, cosine),
                     __fmul_rn(second, sine));
}

__device__ __forceinline__ float dsv4_rope_second(
    float first, float second, float cosine, float sine) {
    return __fadd_rn(__fmul_rn(second, cosine),
                     __fmul_rn(first, sine));
}

// Reproduces encode_e4m3_half_up() from src/deepseek_attention_kv.cpp exactly.
//
// Two things here are deliberate and neither is the obvious choice. The
// backend's other quantizer, quantize_e4m3_value(), rounds ties to even via
// rintf; this contract is half-up, floor(x * 8 + 0.5), and using the wrong one
// silently changes which candidates a hard top-k selects. And the exponent
// comes from log2f rather than the mathematically exact frexpf binade: just
// below a power of two the host's log2 rounds up to the next integer, after
// which its mantissa falls below 1 and it takes the sub-1 branch. That is a
// suspected defect in the reference, recorded separately, but exactness against
// the declared scalar reference is the binding contract, so it is reproduced
// rather than corrected here.
//
// Screened against the host over 4,000,663 probes -- power-of-two boundaries,
// half-up ties, saturation, zero, denormal and a random sweep -- with zero
// mismatches on both supported architectures.
__device__ unsigned char dsv4_encode_e4m3_half_up(float value) {
    const unsigned int sign = value < 0.0F ? 1U : 0U;
    float magnitude = fminf(fabsf(value), 448.0F);
    if (!isfinite(magnitude)) magnitude = 0.0F;
    if (magnitude == 0.0F) return 0U;
    float exponent = floorf(log2f(magnitude));
    exponent = fminf(fmaxf(exponent, -6.0F), 8.0F);
    const float mantissa = magnitude / exp2f(exponent);
    int exponent_field = 0;
    int mantissa_field = 0;
    if (mantissa >= 1.0F) {
        exponent_field = static_cast<int>(exponent) + 7;
        mantissa_field = static_cast<int>(
            floorf((mantissa - 1.0F) * 8.0F + 0.5F));
        if (mantissa_field >= 8) {
            mantissa_field = 0;
            ++exponent_field;
        }
    } else {
        mantissa_field = static_cast<int>(floorf(mantissa * 8.0F + 0.5F));
        if (mantissa_field >= 8) {
            mantissa_field = 0;
            exponent_field = 1;
        }
    }
    exponent_field = min(exponent_field, 15);
    return static_cast<unsigned char>(
        (sign << 7U) | (static_cast<unsigned int>(exponent_field) << 3U) |
        static_cast<unsigned int>(mantissa_field));
}

// One block per index head, one thread per head dimension. Mirrors the per-head
// sequence index_select() runs on the host: RoPE over the trailing rope_dim
// elements, then bf16 rounding of that region *only*, then E4M3 quantization of
// the whole head.
//
// The cosines and sines are computed host-side and uploaded rather than
// evaluated here, because host libm and device trigonometry differ in the last
// ulp and the angles depend only on the position and the layer frequencies,
// both known before the call. The rotation itself uses the existing
// non-contracted helpers; a probe confirms the host does not contract its
// equivalent expression into an fma at -O3, so the two agree.
//
// The non-finite check covers the whole head and leaves it unmodified on
// failure, as dsv4_physical_quantize_query_e4m3_f32 does.
__global__ void dsv4_index_query_rope_quantize_kernel(
    float* queries, const float* cosines, const float* sines,
    std::uint32_t head_dim, std::uint32_t rope_dim, unsigned int quantize,
    unsigned int* error) {
    __shared__ unsigned int rejected;
    auto* query = queries +
        static_cast<std::uint64_t>(blockIdx.x) * head_dim;
    const auto rope_begin = head_dim - rope_dim;
    if (threadIdx.x < rope_dim / 2U) {
        const auto pair = threadIdx.x;
        const float first = query[rope_begin + pair * 2U];
        const float second = query[rope_begin + pair * 2U + 1U];
        const float cosine = cosines[pair];
        const float sine = sines[pair];
        query[rope_begin + pair * 2U] =
            dsv4_rope_first(first, second, cosine, sine);
        query[rope_begin + pair * 2U + 1U] =
            dsv4_rope_second(first, second, cosine, sine);
    }
    __syncthreads();
    if (threadIdx.x < rope_dim) {
        auto& value = query[rope_begin + threadIdx.x];
        value = bf16_round(value);
    }
    if (quantize == 0U) return;
    if (threadIdx.x == 0U) rejected = 0U;
    __syncthreads();
    for (std::uint32_t index = threadIdx.x; index < head_dim;
         index += blockDim.x) {
        if (!isfinite(query[index])) atomicExch(&rejected, 1U);
    }
    __syncthreads();
    if (rejected != 0U) {
        if (threadIdx.x == 0U) atomicExch(error, 1U);
        return;
    }
    for (std::uint32_t index = threadIdx.x; index < head_dim;
         index += blockDim.x) {
        query[index] =
            fp8_e4m3_value(dsv4_encode_e4m3_half_up(query[index]));
    }
}

// The BF16 rounding linear() applies to every projection output, plus the
// non-finite rejection the host performs before scoring. Rejecting here rather
// than after a download keeps the failure closed when the projection never
// crosses back to the host.
__global__ void dsv4_index_projection_round_kernel(
    float* values, std::uint32_t count, unsigned int* error) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float rounded = bf16_round(values[index]);
    if (!isfinite(rounded)) atomicExch(error, 1U);
    values[index] = rounded;
}

// index_select()'s per-head weight tail: the projection's own BF16 rounding,
// then the scale, then a second rounding of the product. The multiply is
// explicitly non-contracted so no fma can absorb it.
__global__ void dsv4_index_weight_scale_kernel(
    float* values, std::uint32_t count, float scale, unsigned int* error) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float scaled =
        bf16_round(__fmul_rn(bf16_round(values[index]), scale));
    if (!isfinite(scaled)) atomicExch(error, 1U);
    values[index] = scaled;
}

// Column-major staging for the score kernel's 64 consecutive-float reads. The
// host form does this pass on the CPU on its way to the upload; with the query
// already on the device there is nothing to upload, so the same permutation
// runs here.
__global__ void dsv4_index_query_transpose_kernel(
    float* destination, const float* source, std::uint32_t heads,
    std::uint32_t head_dim) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= heads * head_dim) return;
    const auto head = index / head_dim;
    const auto column = index % head_dim;
    destination[static_cast<std::uint64_t>(column) * heads + head] =
        source[index];
}

// The declared contract pins the accumulation *order* of the FP64 sum, not
// where its operands are read from. The dequantize/square phase and the
// scale/store phase carry no cross-column dependency, so only the reduction
// itself has to stay on one thread, reading shared memory instead of stalling
// on global latency 1,024 times in a row.
constexpr std::uint32_t kDsv4QueryRankNormColumns = 1024U;
constexpr std::uint32_t kDsv4QueryRankNormThreads = kDsv4QueryRankNormColumns;

__global__ void dsv4_query_rank_norm(
    const float* input, const float* weight, __nv_bfloat16* output,
    unsigned int* error) {
    constexpr std::uint32_t columns = kDsv4QueryRankNormColumns;
    __shared__ double squares[columns];
    __shared__ float rounded[columns];
    __shared__ unsigned int rejected;
    __shared__ float shared_reciprocal;

    if (threadIdx.x == 0U) rejected = 0U;
    __syncthreads();

    for (std::uint32_t column = threadIdx.x; column < columns;
         column += kDsv4QueryRankNormThreads) {
        const float value = bf16_round(input[column]);
        if (!isfinite(value) || !isfinite(weight[column])) {
            atomicExch(&rejected, 1U);
            continue;
        }
        rounded[column] = value;
        squares[column] = __dmul_rn(static_cast<double>(value),
                                    static_cast<double>(value));
    }
    __syncthreads();
    // The sequential shape returned before writing any output on the first
    // non-finite column, so a rejected row leaves the destination untouched.
    if (rejected != 0U) {
        if (threadIdx.x == 0U) atomicExch(error, 1U);
        return;
    }

    if (threadIdx.x == 0U) {
        double squared_sum = 0.0;
        for (std::uint32_t column = 0U; column < columns; ++column) {
            squared_sum = __dadd_rn(squared_sum, squares[column]);
        }
        shared_reciprocal = 1.0F / sqrtf(
            static_cast<float>(squared_sum / static_cast<double>(columns)) +
            1.0e-6F);
    }
    __syncthreads();

    const float reciprocal = shared_reciprocal;
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += kDsv4QueryRankNormThreads) {
        output[column] = __float2bfloat16_rn(
            __fmul_rn(__fmul_rn(rounded[column], reciprocal), weight[column]));
    }
}

// One block per head, as before. Within the head the same rule applies: the
// FP64 accumulation keeps its ascending order on one thread, everything else
// is per-column independent. The RoPE tail rewrites disjoint output pairs, so
// it parallelizes across pairs once the store phase has been synchronized.
constexpr std::uint32_t kDsv4QueryNormRopeColumns = 512U;
constexpr std::uint32_t kDsv4QueryNormRopeThreads = kDsv4QueryNormRopeColumns;

__global__ void dsv4_query_norm_rope(
    const float* input, const float* cosines, const float* sines,
    __nv_bfloat16* output, unsigned int* error) {
    constexpr std::uint32_t heads = 64U;
    constexpr std::uint32_t columns = kDsv4QueryNormRopeColumns;
    constexpr std::uint32_t rope = 64U;
    const auto head = static_cast<std::uint32_t>(blockIdx.x);
    if (head >= heads) return;
    const auto base = static_cast<std::uint64_t>(head) * columns;

    __shared__ double squares[columns];
    __shared__ float rounded[columns];
    __shared__ unsigned int rejected;
    __shared__ float shared_reciprocal;

    if (threadIdx.x == 0U) rejected = 0U;
    __syncthreads();

    for (std::uint32_t column = threadIdx.x; column < columns;
         column += kDsv4QueryNormRopeThreads) {
        const float value = bf16_round(input[base + column]);
        if (!isfinite(value)) {
            atomicExch(&rejected, 1U);
            continue;
        }
        rounded[column] = value;
        squares[column] = __dmul_rn(static_cast<double>(value),
                                    static_cast<double>(value));
    }
    __syncthreads();
    if (rejected != 0U) {
        if (threadIdx.x == 0U) atomicExch(error, 1U);
        return;
    }

    if (threadIdx.x == 0U) {
        double squared_sum = 0.0;
        for (std::uint32_t column = 0U; column < columns; ++column) {
            squared_sum = __dadd_rn(squared_sum, squares[column]);
        }
        shared_reciprocal = 1.0F / sqrtf(
            static_cast<float>(squared_sum / static_cast<double>(columns)) +
            1.0e-6F);
    }
    __syncthreads();

    const float reciprocal = shared_reciprocal;
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += kDsv4QueryNormRopeThreads) {
        output[base + column] = __float2bfloat16_rn(
            __fmul_rn(rounded[column], reciprocal));
    }
    __syncthreads();

    constexpr std::uint32_t rope_begin = columns - rope;
    for (std::uint32_t pair = threadIdx.x; pair < rope / 2U;
         pair += kDsv4QueryNormRopeThreads) {
        const auto first_index = base + rope_begin + pair * 2U;
        const float first = __bfloat162float(output[first_index]);
        const float second = __bfloat162float(output[first_index + 1U]);
        output[first_index] = __float2bfloat16_rn(dsv4_rope_first(
            first, second, cosines[pair], sines[pair]));
        output[first_index + 1U] = __float2bfloat16_rn(dsv4_rope_second(
            first, second, cosines[pair], sines[pair]));
    }
}

// Multi-row form of the accepted query RMS/RoPE boundary. The input is the
// raw FP32 tensor-page wq_b output. The destination is group-major because the
// two 32-head attention groups consume [group, row, head, column] without a
// host transpose. Cosines/sines describe inverse RoPE for the later output
// decode, so forward query RoPE uses the negated sine.
__global__ void dsv4_page_query_norm_rope(
    const float* input, const float* inverse_cosines,
    const float* inverse_sines, __nv_bfloat16* output,
    std::uint32_t rows, unsigned int* error) {
    constexpr std::uint32_t heads = 64U;
    constexpr std::uint32_t heads_per_group = 32U;
    constexpr std::uint32_t columns = kDsv4QueryNormRopeColumns;
    constexpr std::uint32_t rope = 64U;
    const auto head = static_cast<std::uint32_t>(blockIdx.x);
    const auto row = static_cast<std::uint32_t>(blockIdx.y);
    if (head >= heads || row >= rows) return;
    const auto input_base =
        (static_cast<std::uint64_t>(row) * heads + head) * columns;
    const auto group = head / heads_per_group;
    const auto local_head = head % heads_per_group;
    const auto output_base =
        ((static_cast<std::uint64_t>(group) * rows + row) *
             heads_per_group + local_head) * columns;

    __shared__ double squares[columns];
    __shared__ float rounded[columns];
    __shared__ unsigned int rejected;
    __shared__ float shared_reciprocal;

    if (threadIdx.x == 0U) rejected = 0U;
    __syncthreads();
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += kDsv4QueryNormRopeThreads) {
        const float value = bf16_round(input[input_base + column]);
        if (!isfinite(value)) {
            atomicExch(&rejected, 1U);
            continue;
        }
        rounded[column] = value;
        squares[column] = __dmul_rn(static_cast<double>(value),
                                    static_cast<double>(value));
    }
    __syncthreads();
    if (rejected != 0U) {
        if (threadIdx.x == 0U) atomicExch(error, 1U);
        return;
    }
    if (threadIdx.x == 0U) {
        double squared_sum = 0.0;
        for (std::uint32_t column = 0U; column < columns; ++column) {
            squared_sum = __dadd_rn(squared_sum, squares[column]);
        }
        shared_reciprocal = 1.0F / sqrtf(
            static_cast<float>(squared_sum / static_cast<double>(columns)) +
            1.0e-6F);
    }
    __syncthreads();
    const float reciprocal = shared_reciprocal;
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += kDsv4QueryNormRopeThreads) {
        output[output_base + column] = __float2bfloat16_rn(
            __fmul_rn(rounded[column], reciprocal));
    }
    __syncthreads();
    constexpr std::uint32_t rope_begin = columns - rope;
    const auto rope_base = static_cast<std::uint64_t>(row) * (rope / 2U);
    for (std::uint32_t pair = threadIdx.x; pair < rope / 2U;
         pair += kDsv4QueryNormRopeThreads) {
        const auto first_index = output_base + rope_begin + pair * 2U;
        const float first = __bfloat162float(output[first_index]);
        const float second = __bfloat162float(output[first_index + 1U]);
        const float cosine = inverse_cosines[rope_base + pair];
        const float sine = -inverse_sines[rope_base + pair];
        output[first_index] = __float2bfloat16_rn(
            dsv4_rope_first(first, second, cosine, sine));
        output[first_index + 1U] = __float2bfloat16_rn(
            dsv4_rope_second(first, second, cosine, sine));
    }
}

constexpr std::uint32_t kDsv4KeyValueNormColumns = 512U;
constexpr std::uint32_t kDsv4KeyValueNormThreads = kDsv4KeyValueNormColumns;

__global__ void dsv4_key_value_norm_rope(
    const float* input, const float* weight, const float* cosines,
    const float* sines, __nv_bfloat16* output, unsigned int* error) {
    constexpr std::uint32_t columns = kDsv4KeyValueNormColumns;
    constexpr std::uint32_t rope = 64U;

    __shared__ double squares[columns];
    __shared__ float rounded[columns];
    __shared__ unsigned int rejected;
    __shared__ float shared_reciprocal;

    if (threadIdx.x == 0U) rejected = 0U;
    __syncthreads();

    for (std::uint32_t column = threadIdx.x; column < columns;
         column += kDsv4KeyValueNormThreads) {
        const float value = bf16_round(input[column]);
        if (!isfinite(value) || !isfinite(weight[column])) {
            atomicExch(&rejected, 1U);
            continue;
        }
        rounded[column] = value;
        squares[column] = __dmul_rn(static_cast<double>(value),
                                    static_cast<double>(value));
    }
    __syncthreads();
    if (rejected != 0U) {
        if (threadIdx.x == 0U) atomicExch(error, 1U);
        return;
    }

    if (threadIdx.x == 0U) {
        double squared_sum = 0.0;
        for (std::uint32_t column = 0U; column < columns; ++column) {
            squared_sum = __dadd_rn(squared_sum, squares[column]);
        }
        shared_reciprocal = 1.0F / sqrtf(
            static_cast<float>(squared_sum / static_cast<double>(columns)) +
            1.0e-6F);
    }
    __syncthreads();

    const float reciprocal = shared_reciprocal;
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += kDsv4KeyValueNormThreads) {
        output[column] = __float2bfloat16_rn(
            __fmul_rn(__fmul_rn(rounded[column], reciprocal), weight[column]));
    }
    __syncthreads();

    constexpr std::uint32_t rope_begin = columns - rope;
    for (std::uint32_t pair = threadIdx.x; pair < rope / 2U;
         pair += kDsv4KeyValueNormThreads) {
        const auto first_index = rope_begin + pair * 2U;
        const float first = __bfloat162float(output[first_index]);
        const float second = __bfloat162float(output[first_index + 1U]);
        output[first_index] = __float2bfloat16_rn(dsv4_rope_first(
            first, second, cosines[pair], sines[pair]));
        output[first_index + 1U] = __float2bfloat16_rn(dsv4_rope_second(
            first, second, cosines[pair], sines[pair]));
    }
}

constexpr std::uint32_t kDsv4MhcHidden = 4096U;
constexpr std::uint32_t kDsv4MhcMultiplier = 4U;
constexpr std::uint32_t kDsv4MhcMixes = 24U;
constexpr std::uint32_t kDsv4MhcSplits = 8U;
constexpr std::uint32_t kDsv4MhcProjectionThreads = 256U;
constexpr std::uint32_t kDsv4MhcProjectionTile = 2U;
constexpr std::uint32_t kDsv4MhcStandaloneSplits = 64U;
constexpr std::uint64_t kDsv4MhcProjectionElements =
    static_cast<std::uint64_t>(kDsv4MhcMixes) * kDsv4MhcMultiplier *
    kDsv4MhcHidden;
constexpr std::uint64_t kDsv4MhcAuxNormOffset = 112U;
constexpr std::uint64_t kDsv4MhcAuxBytes =
    kDsv4MhcAuxNormOffset +
    static_cast<std::uint64_t>(kDsv4MhcHidden) * sizeof(std::uint16_t);

__device__ float dsv4_mhc_warp_xor_sum(float value) {
    constexpr unsigned int mask = 0xFFFF'FFFFU;
    value += __shfl_xor_sync(mask, value, 16);
    value += __shfl_xor_sync(mask, value, 8);
    value += __shfl_xor_sync(mask, value, 4);
    value += __shfl_xor_sync(mask, value, 2);
    value += __shfl_xor_sync(mask, value, 1);
    return value;
}

__global__ void dsv4_mhc_fused_post_projection(
    const float* combination, const __nv_bfloat16* residual,
    const float* post, const __nv_bfloat16* branch,
    const float* projection, float* partial_projection,
    float* partial_square_sum, __nv_bfloat16* residual_output) {
    const auto tile = static_cast<std::uint32_t>(blockIdx.x);
    const auto split = static_cast<std::uint32_t>(blockIdx.y);
    const auto thread = static_cast<std::uint32_t>(threadIdx.x);
    const auto warp = thread >> 5U;
    const auto lane = thread & 31U;
    const auto hidden_begin = split * (kDsv4MhcHidden / kDsv4MhcSplits);
    float accumulators[kDsv4MhcProjectionTile]{0.0F, 0.0F};
    float square_sum = 0.0F;

    #pragma unroll
    for (std::uint32_t iteration = 0U; iteration < 2U; ++iteration) {
        const auto hidden_index = hidden_begin +
            iteration * kDsv4MhcProjectionThreads + thread;
        float new_residual[kDsv4MhcMultiplier];
        #pragma unroll
        for (std::uint32_t destination = 0U;
             destination < kDsv4MhcMultiplier; ++destination) {
            float value = post[destination] *
                          __bfloat162float(branch[hidden_index]);
            #pragma unroll
            for (std::uint32_t source = 0U;
                 source < kDsv4MhcMultiplier; ++source) {
                value += combination[source * kDsv4MhcMultiplier +
                                     destination] *
                         __bfloat162float(
                             residual[source * kDsv4MhcHidden + hidden_index]);
            }
            new_residual[destination] = value;
            if (tile == 0U) {
                residual_output[destination * kDsv4MhcHidden + hidden_index] =
                    __float2bfloat16_rn(value);
                square_sum += value * value;
            }
        }
        #pragma unroll
        for (std::uint32_t output = 0U;
             output < kDsv4MhcProjectionTile; ++output) {
            const auto row = tile * kDsv4MhcProjectionTile + output;
            #pragma unroll
            for (std::uint32_t copy = 0U; copy < kDsv4MhcMultiplier; ++copy) {
                accumulators[output] +=
                    projection[(row * kDsv4MhcMultiplier + copy) *
                                   kDsv4MhcHidden + hidden_index] *
                    new_residual[copy];
            }
        }
    }

    #pragma unroll
    for (auto& accumulator : accumulators) {
        accumulator = dsv4_mhc_warp_xor_sum(accumulator);
    }
    if (tile == 0U) square_sum = dsv4_mhc_warp_xor_sum(square_sum);

    __shared__ float warp_results[8][3];
    if (lane == 0U) {
        warp_results[warp][0] = accumulators[0];
        warp_results[warp][1] = accumulators[1];
        if (tile == 0U) warp_results[warp][2] = square_sum;
    }
    __syncthreads();
    if (warp != 0U) return;
    if (lane < kDsv4MhcProjectionTile) {
        float value = 0.0F;
        #pragma unroll
        for (std::uint32_t source_warp = 0U; source_warp < 8U;
             ++source_warp) {
            value += warp_results[source_warp][lane];
        }
        partial_projection[split * kDsv4MhcMixes +
                           tile * kDsv4MhcProjectionTile + lane] = value;
    }
    if (tile == 0U && lane == 0U) {
        float value = 0.0F;
        #pragma unroll
        for (std::uint32_t source_warp = 0U; source_warp < 8U;
             ++source_warp) {
            value += warp_results[source_warp][2];
        }
        partial_square_sum[split] = value;
    }
}

__global__ void dsv4_mhc_standalone_projection(
    const __nv_bfloat16* residual, const float* projection,
    float* partial_projection) {
#if __CUDA_ARCH__ >= 800
    namespace wmma = nvcuda::wmma;
    const auto tile = static_cast<std::uint32_t>(blockIdx.x);
    const auto split = static_cast<std::uint32_t>(blockIdx.y);
    const auto thread = static_cast<std::uint32_t>(threadIdx.x);
    constexpr std::uint32_t split_columns =
        kDsv4MhcMultiplier * kDsv4MhcHidden / kDsv4MhcStandaloneSplits;
    __shared__ float matrix_a[16U * 8U];
    __shared__ float matrix_b[8U * 16U];
    __shared__ float matrix_c[16U * 16U];
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0F);
    #pragma unroll
    for (std::uint32_t column_begin = 0U; column_begin < split_columns;
         column_begin += 8U) {
        for (std::uint32_t index = thread; index < 16U * 8U; index += 32U) {
            const auto row = index / 8U;
            const auto column = index % 8U;
            matrix_a[index] = row == 0U
                ? __bfloat162float(residual[split * split_columns +
                                             column_begin + column])
                : 0.0F;
        }
        for (std::uint32_t index = thread; index < 8U * 16U; index += 32U) {
            const auto column = index / 8U;
            const auto reduction = index % 8U;
            const auto output = tile * 16U + column;
            matrix_b[index] = output < kDsv4MhcMixes
                ? projection[output * kDsv4MhcMultiplier * kDsv4MhcHidden +
                             split * split_columns + column_begin + reduction]
                : 0.0F;
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 8,
                       wmma::precision::tf32, wmma::row_major> left;
        wmma::fragment<wmma::matrix_b, 16, 16, 8,
                       wmma::precision::tf32, wmma::col_major> right;
        wmma::load_matrix_sync(left, matrix_a, 8U);
        wmma::load_matrix_sync(right, matrix_b, 8U);
        wmma::mma_sync(accumulator, left, right, accumulator);
        __syncthreads();
    }
    wmma::store_matrix_sync(matrix_c, accumulator, 16U,
                            wmma::mem_row_major);
    __syncthreads();
    if (thread < 16U) {
        const auto output = tile * 16U + thread;
        if (output < kDsv4MhcMixes) {
            partial_projection[split * kDsv4MhcMixes + output] =
                matrix_c[thread];
        }
    }
#endif
}

__global__ void dsv4_mhc_standalone_square_sum(
    const __nv_bfloat16* residual, float* partial_square_sum) {
    const auto split = static_cast<std::uint32_t>(blockIdx.x);
    const auto thread = static_cast<std::uint32_t>(threadIdx.x);
    constexpr std::uint32_t split_columns =
        kDsv4MhcMultiplier * kDsv4MhcHidden / kDsv4MhcStandaloneSplits;
    float total = 0.0F;
    #pragma unroll
    for (std::uint32_t chunk = 0U; chunk < 4U; ++chunk) {
        float values[8U]{};
        #pragma unroll
        for (std::uint32_t lane = 0U; lane < 8U; ++lane) {
            values[lane] = __bfloat162float(
                residual[split * split_columns + chunk * 64U +
                         thread * 8U + lane]);
        }
        float sum = values[1] * values[1];
        sum = fmaf(values[0], values[0], sum);
        #pragma unroll
        for (std::uint32_t lane = 2U; lane < 8U; ++lane) {
            sum = fmaf(values[lane], values[lane], sum);
        }
        sum += __shfl_xor_sync(0xFFU, sum, 4);
        sum += __shfl_xor_sync(0xFFU, sum, 2);
        sum += __shfl_xor_sync(0xFFU, sum, 1);
        total += sum;
    }
    if (thread == 0U) partial_square_sum[split] = total;
}

__device__ float dsv4_mhc_sigmoid(float value) {
    return 1.0F / (1.0F + expf(0.0F - value));
}

__global__ void dsv4_mhc_mix(
    const float* partial_projection, const float* partial_square_sum,
    const float* scale, const float* base, std::uint32_t split_count,
    float* pre, float* post, float* combination) {
    const auto thread = static_cast<std::uint32_t>(threadIdx.x);
    __shared__ float mixes[kDsv4MhcMixes];
    if (thread < kDsv4MhcMixes) {
        float square_sum = 0.0F;
        for (std::uint32_t split = 0U; split < split_count; ++split) {
            square_sum += partial_square_sum[split];
        }
        const float reciprocal = rsqrtf(
            square_sum /
                static_cast<float>(kDsv4MhcMultiplier * kDsv4MhcHidden) +
            1.0e-6F);
        float value = 0.0F;
        for (std::uint32_t split = 0U; split < split_count; ++split) {
            value += partial_projection[split * kDsv4MhcMixes + thread];
        }
        mixes[thread] = value * reciprocal;
    }
    __syncthreads();
    if (thread < kDsv4MhcMultiplier) {
        pre[thread] = dsv4_mhc_sigmoid(
                          mixes[thread] * scale[0] + base[thread]) +
                      1.0e-6F;
        post[thread] = dsv4_mhc_sigmoid(
                           mixes[thread + kDsv4MhcMultiplier] * scale[1] +
                           base[thread + kDsv4MhcMultiplier]) *
                       2.0F;
    }
    if (thread >= 16U) return;
    float value = mixes[thread + 2U * kDsv4MhcMultiplier] * scale[2] +
                  base[thread + 2U * kDsv4MhcMultiplier];
    constexpr unsigned int mask = 0xFFFFU;
    float row_max = fmaxf(value, __shfl_xor_sync(mask, value, 2));
    row_max = fmaxf(row_max, __shfl_xor_sync(mask, row_max, 1));
    value = expf(value - row_max);
    float row_sum = value + __shfl_xor_sync(mask, value, 2);
    row_sum += __shfl_xor_sync(mask, row_sum, 1);
    value = value / row_sum + 1.0e-6F;
    float column_sum = value + __shfl_xor_sync(mask, value, 8);
    column_sum += __shfl_xor_sync(mask, column_sum, 4);
    value /= column_sum + 1.0e-6F;
    for (std::uint32_t iteration = 1U; iteration < 20U; ++iteration) {
        row_sum = value + __shfl_xor_sync(mask, value, 2);
        row_sum += __shfl_xor_sync(mask, row_sum, 1);
        value /= row_sum + 1.0e-6F;
        column_sum = value + __shfl_xor_sync(mask, value, 8);
        column_sum += __shfl_xor_sync(mask, column_sum, 4);
        value /= column_sum + 1.0e-6F;
    }
    combination[thread] = value;
}

// Unlike the attention norms, this reduction is already a tree: 64 per-thread
// accumulators, each summing four blocks in order, combined by a fixed xor
// pattern. That shape *is* the contract, so the accumulator count and its
// combination order are preserved exactly. Only the two elementwise phases,
// which carry no cross-element dependency, are widened; the FP32 values are
// staged in shared memory so the pinned reduction consumes identical operands.
constexpr std::uint32_t kDsv4MhcWeightedNormThreads = 512U;
constexpr std::uint32_t kDsv4MhcWeightedNormAccumulators = 64U;

__global__ void dsv4_mhc_weighted_norm(
    const __nv_bfloat16* residual, const float* pre,
    const __nv_bfloat16* norm_weight, __nv_bfloat16* weighted_bf16,
    __nv_bfloat16* layer_input) {
    const auto thread = static_cast<std::uint32_t>(threadIdx.x);
    __shared__ float staged[kDsv4MhcHidden];
    __shared__ float cross_warp[kDsv4MhcWeightedNormAccumulators];
    __shared__ float shared_reciprocal;

    for (std::uint32_t index = thread; index < kDsv4MhcHidden;
         index += kDsv4MhcWeightedNormThreads) {
        float value = 0.0F;
        #pragma unroll
        for (std::uint32_t copy = 0U; copy < kDsv4MhcMultiplier; ++copy) {
            value += pre[copy] * __bfloat162float(
                residual[copy * kDsv4MhcHidden + index]);
        }
        staged[index] = value;
        weighted_bf16[index] = __float2bfloat16_rn(value);
    }
    __syncthreads();

    // Threads 0-63 are exactly two warps, so the full-mask shuffles below keep
    // every lane they require.
    float sum = 0.0F;
    if (thread < kDsv4MhcWeightedNormAccumulators) {
        float per_position[16]{};
        #pragma unroll
        for (std::uint32_t block = 0U; block < 4U; ++block) {
            #pragma unroll
            for (std::uint32_t lane = 0U; lane < 16U; ++lane) {
                const auto hidden_index = block * 1024U + thread * 16U + lane;
                const float value = staged[hidden_index];
                per_position[lane] += value * value;
            }
        }
        #pragma unroll
        for (std::uint32_t lane = 0U; lane < 16U; lane += 2U) {
            sum += per_position[lane];
        }
        #pragma unroll
        for (std::uint32_t lane = 1U; lane < 16U; lane += 2U) {
            sum += per_position[lane];
        }
        cross_warp[thread] = sum;
    }
    __syncthreads();
    if (thread < kDsv4MhcWeightedNormAccumulators) {
        sum += cross_warp[thread ^ 32U];
        sum += __shfl_xor_sync(0xFFFF'FFFFU, sum, 16);
        sum += __shfl_xor_sync(0xFFFF'FFFFU, sum, 8);
        sum += __shfl_xor_sync(0xFFFF'FFFFU, sum, 4);
        sum += __shfl_xor_sync(0xFFFF'FFFFU, sum, 2);
        sum += __shfl_xor_sync(0xFFFF'FFFFU, sum, 1);
        if (thread == 0U) {
            shared_reciprocal = rsqrtf(
                sum / static_cast<float>(kDsv4MhcHidden) + 1.0e-6F);
        }
    }
    __syncthreads();

    // Reads the BF16 round trip rather than the staged FP32 value, exactly as
    // the sequential shape did.
    const float reciprocal = shared_reciprocal;
    for (std::uint32_t index = thread; index < kDsv4MhcHidden;
         index += kDsv4MhcWeightedNormThreads) {
        const float value = __bfloat162float(weighted_bf16[index]) *
                            reciprocal *
                            __bfloat162float(norm_weight[index]);
        layer_input[index] = __float2bfloat16_rn(value);
    }
}

__global__ void dsv4_mhc_final_post(
    const float* combination, const __nv_bfloat16* residual,
    const float* post, const __nv_bfloat16* branch,
    __nv_bfloat16* residual_output) {
    const auto hidden_index =
        static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (hidden_index >= kDsv4MhcHidden) return;
    #pragma unroll
    for (std::uint32_t destination = 0U;
         destination < kDsv4MhcMultiplier; ++destination) {
        float value = post[destination] *
                      __bfloat162float(branch[hidden_index]);
        #pragma unroll
        for (std::uint32_t source = 0U;
             source < kDsv4MhcMultiplier; ++source) {
            value += combination[source * kDsv4MhcMultiplier + destination] *
                     __bfloat162float(
                         residual[source * kDsv4MhcHidden + hidden_index]);
        }
        residual_output[destination * kDsv4MhcHidden + hidden_index] =
            __float2bfloat16_rn(value);
    }
}

constexpr std::uint32_t kDsv4MhcRouterLogits = 256U;

struct alignas(256) Dsv4MhcWorkspace {
    __nv_bfloat16 residual[2][kDsv4MhcMultiplier * kDsv4MhcHidden];
    __nv_bfloat16 branch[kDsv4MhcHidden];
    __nv_bfloat16 weighted[kDsv4MhcHidden];
    __nv_bfloat16 layer_input[kDsv4MhcHidden];
    float partial_projection[kDsv4MhcStandaloneSplits * kDsv4MhcMixes];
    float partial_square_sum[kDsv4MhcStandaloneSplits];
    float pre[kDsv4MhcMultiplier];
    float post[kDsv4MhcMultiplier];
    float combination[kDsv4MhcMultiplier * kDsv4MhcMultiplier];
    float router_logits[kDsv4MhcRouterLogits];
    float glm53_router_logits[288U];
    unsigned int failure{};
};

__global__ void dsv4_store_mhc_page_branches(
    const float* values, __nv_bfloat16* diagnostic,
    Dsv4MhcWorkspace* slot_arena, const std::uint32_t* slots,
    std::uint32_t rows) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    const auto elements = static_cast<std::uint64_t>(rows) * kDsv4MhcHidden;
    if (index >= elements) return;
    const auto row = static_cast<std::uint32_t>(index / kDsv4MhcHidden);
    const auto column = static_cast<std::uint32_t>(index % kDsv4MhcHidden);
    const auto encoded = __float2bfloat16_rn(values[index]);
    diagnostic[index] = encoded;
    if (slot_arena != nullptr) {
        slot_arena[slots[row]].branch[column] = encoded;
    }
}

__global__ void dsv4_scatter_encoded_mhc_page_branches(
    const __nv_bfloat16* branches, Dsv4MhcWorkspace* slot_arena,
    const std::uint32_t* slots, std::uint32_t rows) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    const auto elements = static_cast<std::uint64_t>(rows) * kDsv4MhcHidden;
    if (index >= elements) return;
    const auto row = static_cast<std::uint32_t>(index / kDsv4MhcHidden);
    const auto column = static_cast<std::uint32_t>(index % kDsv4MhcHidden);
    slot_arena[slots[row]].branch[column] = branches[index];
}

// One prompt row's fused mHC state. The device workspace holds the residual
// pair, branch, weighted, layer input, partial reductions, pre/post, the
// Sinkhorn combination, and the router logits; the three scalars are the host
// half of the same state machine. Together they are the complete state a
// transition reads and writes, which is what makes swapping slots exact.
// Workspaces live in one arena, so a slot is an index rather than an
// allocation and a wide page costs a single cudaMalloc.
struct Dsv4MhcSlotState {
    std::uint32_t stage{};
    std::uint32_t residual_index{};
    bool branch_ready{};
};

// One slot per prompt row of the widest admitted page, at 95 KB each.
constexpr std::uint32_t kDsv4MhcMaximumSlots = 8192U;

constexpr std::uint64_t kDsv4MhcMaximumHostStagingBytes =
    static_cast<std::uint64_t>(
        kDsv4MhcMultiplier * kDsv4MhcHidden + 2U * kDsv4MhcHidden) *
    sizeof(std::uint16_t);

bool checked_bytes(std::uint64_t left, std::uint64_t right, std::uint64_t element_bytes,
                   std::uint64_t& result) {
    if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left) return false;
    const auto elements = left * right;
    if (elements != 0U && element_bytes > std::numeric_limits<std::uint64_t>::max() / elements) {
        return false;
    }
    result = elements * element_bytes;
    return true;
}

constexpr std::uint64_t kWeightPointerAlignment = 16U;
constexpr std::uint64_t kWeightArenaAlignment = 256U;

bool align_up(std::uint64_t value, std::uint64_t alignment,
              std::uint64_t& result) {
    const auto remainder = value % alignment;
    const auto padding = remainder == 0U ? 0U : alignment - remainder;
    if (value > std::numeric_limits<std::uint64_t>::max() - padding) return false;
    result = value + padding;
    return true;
}

struct Dsv4AttentionMhcWorkspaceLayout {
    std::uint64_t page_offset{};
    std::uint64_t candidate_offset{};
    std::uint64_t query_offset{};
    std::uint64_t sink_offset{};
    std::uint64_t cosine_offset{};
    std::uint64_t sine_offset{};
    std::uint64_t slot_offset{};
    std::uint64_t block_offset{};
    std::uint64_t page_query_rank_offset{};
    std::uint64_t kv_offset{};
    std::uint64_t score_offset{};
    std::uint64_t maximum_offset{};
    std::uint64_t denominator_offset{};
    std::uint64_t value_offset{};
    std::uint64_t attended_offset{};
    std::uint64_t decoded_offset{};
    std::uint64_t output_rank_offset{};
    // Compact E4M3 activation for the SM86 tensor output projection: one
    // value byte per element plus one E8M0 byte per row/K128 group.
    std::uint64_t tensor_values_offset{};
    std::uint64_t tensor_scales_offset{};
    std::uint64_t page_query_values_offset{};
    std::uint64_t page_query_scales_offset{};
    std::uint64_t page_query_raw_offset{};
    std::uint64_t page_query_output_offset{};
    std::uint64_t branch_offset{};
    std::uint64_t encoded_branch_offset{};
    std::uint64_t router_logits_offset{};
    std::uint64_t failure_offset{};
    std::uint64_t page_descriptor_bytes{};
    std::uint64_t candidate_bytes{};
    std::uint64_t query_bytes{};
    std::uint64_t sink_bytes{};
    std::uint64_t rope_bytes{};
    std::uint64_t slot_bytes{};
    std::uint64_t block_bytes{};
    std::uint64_t page_query_rank_bytes{};
    std::uint64_t kv_bytes{};
    std::uint64_t score_bytes{};
    std::uint64_t upload_bytes{};
    std::uint64_t workspace_bytes{};
};

bool dsv4_attention_mhc_workspace_layout(
    std::uint64_t page_count, std::uint32_t rows,
    std::uint32_t total_heads, std::uint32_t output_groups,
    std::uint32_t candidates, std::uint32_t flat_rows,
    bool use_prepared_query, bool project_page_query,
    std::uint64_t mhc_slot_count,
    std::uint64_t resolution_block_count,
    std::uint64_t router_logits_bytes,
    Dsv4AttentionMhcWorkspaceLayout& layout) {
    constexpr std::uint64_t rope_pairs = 32U;
    constexpr std::uint64_t group_elements =
        static_cast<std::uint64_t>(kDsv4PagedHeads) * kDsv4PagedHeadDim;
    constexpr std::uint64_t branch_row_elements = kDsv4MhcHidden;
    const auto attended_row_elements =
        static_cast<std::uint64_t>(total_heads) * kDsv4PagedHeadDim;
    const auto output_rank_row_elements =
        static_cast<std::uint64_t>(output_groups) * 1024U;
    std::uint64_t total_candidates{};
    std::uint64_t attended_elements{};
    std::uint64_t output_rank_elements{};
    std::uint64_t branch_elements{};
    if (!checked_bytes(rows, candidates, 1U, total_candidates) ||
        !checked_bytes(rows, attended_row_elements, 1U, attended_elements) ||
        !checked_bytes(rows, output_rank_row_elements, 1U,
                       output_rank_elements) ||
        !checked_bytes(rows, branch_row_elements, 1U, branch_elements) ||
        !checked_bytes(page_count, 1U, sizeof(Dsv4DevicePhysicalPage),
                       layout.page_descriptor_bytes) ||
        !checked_bytes(total_candidates, 1U,
                       sizeof(Dsv4DeviceAttentionCandidate),
                       layout.candidate_bytes) ||
        !checked_bytes((use_prepared_query || project_page_query)
                           ? 0U : attended_elements, 1U,
                       sizeof(std::uint16_t), layout.query_bytes) ||
        !checked_bytes(total_heads, 1U, sizeof(float), layout.sink_bytes) ||
        !checked_bytes(rows, rope_pairs, sizeof(float), layout.rope_bytes) ||
        !checked_bytes(mhc_slot_count, 1U, sizeof(std::uint32_t),
                       layout.slot_bytes) ||
        !checked_bytes(resolution_block_count, 1U,
                       sizeof(Dsv4DeviceKvBlock), layout.block_bytes) ||
        !checked_bytes(project_page_query ? rows : 0U, 1024U,
                       sizeof(float), layout.page_query_rank_bytes) ||
        !checked_bytes(flat_rows, kDsv4PagedHeadDim,
                       sizeof(std::uint16_t), layout.kv_bytes) ||
        !checked_bytes(rows,
                       static_cast<std::uint64_t>(kDsv4PagedHeads) * candidates,
                       sizeof(std::uint16_t), layout.score_bytes)) {
        return false;
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
    if (!region(layout.page_descriptor_bytes, 16U, layout.page_offset) ||
        !region(layout.candidate_bytes, 16U, layout.candidate_offset) ||
        !region(layout.query_bytes, 16U, layout.query_offset) ||
        !region(layout.sink_bytes, 16U, layout.sink_offset) ||
        !region(layout.rope_bytes, 16U, layout.cosine_offset) ||
        !region(layout.rope_bytes, 16U, layout.sine_offset) ||
        !region(layout.slot_bytes, 16U, layout.slot_offset) ||
        !region(layout.block_bytes, 16U, layout.block_offset) ||
        !region(layout.page_query_rank_bytes, 16U,
                layout.page_query_rank_offset)) {
        return false;
    }
    layout.upload_bytes = cursor;

    std::uint64_t maximum_bytes{};
    std::uint64_t denominator_bytes{};
    std::uint64_t value_bytes{};
    std::uint64_t attended_bytes{};
    std::uint64_t decoded_bytes{};
    std::uint64_t output_rank_bytes{};
    std::uint64_t branch_bytes{};
    std::uint64_t encoded_branch_bytes{};
    // The tensor output projection writes whole 64-row tiles, so the branch
    // region is sized to the padded row count while branch_elements keeps its
    // exact meaning for the caller-facing contracts.
    const auto tensor_padded_rows =
        (static_cast<std::uint64_t>(rows) + kDsv4Fp8TensorBlockM - 1U) /
        kDsv4Fp8TensorBlockM * kDsv4Fp8TensorBlockM;
    std::uint64_t branch_capacity_elements{};
    std::uint64_t tensor_values_bytes{};
    std::uint64_t tensor_scales_bytes{};
    std::uint64_t page_query_values_bytes{};
    std::uint64_t page_query_scales_bytes{};
    std::uint64_t page_query_raw_bytes{};
    std::uint64_t page_query_output_bytes{};
    if (!checked_bytes(tensor_padded_rows, branch_row_elements, 1U,
                       branch_capacity_elements) ||
        !checked_bytes(rows, output_rank_row_elements, 1U,
                       tensor_values_bytes) ||
        !checked_bytes(rows, output_rank_row_elements / 128U, 1U,
                       tensor_scales_bytes) ||
        !checked_bytes(project_page_query ? rows : 0U, 1024U, 1U,
                       page_query_values_bytes) ||
        !checked_bytes(project_page_query ? rows : 0U, 8U, 1U,
                       page_query_scales_bytes) ||
        !checked_bytes(project_page_query ? tensor_padded_rows : 0U,
                       2U * group_elements, sizeof(float),
                       page_query_raw_bytes) ||
        !checked_bytes(project_page_query ? rows : 0U, 2U * group_elements,
                       sizeof(std::uint16_t), page_query_output_bytes)) {
        return false;
    }
    if (!checked_bytes(rows, kDsv4PagedHeads, sizeof(float), maximum_bytes) ||
        !checked_bytes(rows, kDsv4PagedHeads, sizeof(float),
                       denominator_bytes) ||
        !checked_bytes(0U, group_elements, sizeof(float), value_bytes) ||
        !checked_bytes(attended_elements, 1U, sizeof(std::uint16_t),
                       attended_bytes) ||
        !checked_bytes(attended_elements, 1U, sizeof(std::uint16_t),
                       decoded_bytes) ||
        !checked_bytes(output_rank_elements, 1U, sizeof(float),
                       output_rank_bytes) ||
        !checked_bytes(branch_capacity_elements, 1U, sizeof(float),
                       branch_bytes) ||
        !checked_bytes(branch_elements, 1U, sizeof(std::uint16_t),
                       encoded_branch_bytes) ||
        !region(layout.kv_bytes, 16U, layout.kv_offset) ||
        !region(layout.score_bytes, 16U, layout.score_offset) ||
        !region(maximum_bytes, 16U, layout.maximum_offset) ||
        !region(denominator_bytes, 16U, layout.denominator_offset) ||
        !region(value_bytes, 16U, layout.value_offset) ||
        !region(attended_bytes, 16U, layout.attended_offset) ||
        !region(decoded_bytes, 16U, layout.decoded_offset) ||
        !region(output_rank_bytes, 16U, layout.output_rank_offset) ||
        !region(tensor_values_bytes, 16U, layout.tensor_values_offset) ||
        !region(tensor_scales_bytes, 16U, layout.tensor_scales_offset) ||
        !region(page_query_values_bytes, 16U,
                layout.page_query_values_offset) ||
        !region(page_query_scales_bytes, 16U,
                layout.page_query_scales_offset) ||
        !region(page_query_raw_bytes, 16U,
                layout.page_query_raw_offset) ||
        !region(page_query_output_bytes, 16U,
                layout.page_query_output_offset) ||
        !region(branch_bytes, 16U, layout.branch_offset) ||
        !region(encoded_branch_bytes, 16U, layout.encoded_branch_offset) ||
        !region(router_logits_bytes, 16U, layout.router_logits_offset) ||
        !region(sizeof(unsigned int), 16U, layout.failure_offset)) {
        return false;
    }
    layout.workspace_bytes = cursor;
    if (const char* trace = std::getenv("STRATA_TRACE_ATTENTION_LAYOUT");
        trace != nullptr && *trace == '1') {
        static std::atomic<int> emitted{0};
        if (emitted.fetch_add(1) < 2) {
            std::fprintf(
                stderr,
                "attention layout rows=%llu flat_rows=%llu candidates=%llu "
                "total=%.2f MB | kv=%.2f score=%.2f value=%.2f attended=%.2f "
                "decoded=%.2f output_rank=%.2f branch=%.2f encoded=%.2f "
                "router=%.2f cand=%.2f query=%.2f\n",
                (unsigned long long)rows, (unsigned long long)flat_rows,
                (unsigned long long)candidates, cursor / 1048576.0,
                layout.kv_bytes / 1048576.0, layout.score_bytes / 1048576.0,
                value_bytes / 1048576.0, attended_bytes / 1048576.0,
                decoded_bytes / 1048576.0, output_rank_bytes / 1048576.0,
                branch_bytes / 1048576.0, encoded_branch_bytes / 1048576.0,
                router_logits_bytes / 1048576.0,
                layout.candidate_bytes / 1048576.0,
                layout.query_bytes / 1048576.0);
        }
    }
    return true;
}

class WeightArena {
public:
    struct Allocation {
        std::uint64_t offset{};
        std::uint64_t bytes{};
        void* address{};
    };

    WeightArena(int device, void* base, std::uint64_t capacity)
        : device_(device), base_(static_cast<std::byte*>(base)),
          capacity_(capacity) {
        free_.reserve(16'384U);
        free_.push_back({0U, capacity});
    }

    // Exhaustion and fragmentation both surface as a failed allocate() and
    // need opposite fixes: one is a budget that is too small, the other is a
    // budget that is big enough but cut into pieces none of which fit. The
    // error path reports both so the next reader does not have to guess.
    struct Occupancy {
        std::uint64_t capacity{};
        std::uint64_t free_bytes{};
        std::uint64_t largest_free{};
        std::size_t free_blocks{};
    };

    [[nodiscard]] Occupancy occupancy() {
        std::scoped_lock lock(mutex_);
        Occupancy report;
        report.capacity = capacity_;
        report.free_blocks = free_.size();
        for (const auto& block : free_) {
            report.free_bytes += block.bytes;
            if (block.bytes > report.largest_free) {
                report.largest_free = block.bytes;
            }
        }
        return report;
    }

    ~WeightArena() {
        if (device_ >= 0) static_cast<void>(cudaSetDevice(device_));
        if (base_ != nullptr) static_cast<void>(cudaFree(base_));
    }

    WeightArena(const WeightArena&) = delete;
    WeightArena& operator=(const WeightArena&) = delete;

    [[nodiscard]] bool allocate(std::uint64_t bytes, Allocation& output) {
        std::scoped_lock lock(mutex_);
        if (metadata_failed_) return false;
        const auto found = std::find_if(
            free_.begin(), free_.end(),
            [bytes](const Block& block) { return block.bytes >= bytes; });
        if (found == free_.end()) return false;
        output.offset = found->offset;
        output.bytes = bytes;
        output.address = base_ + found->offset;
        found->offset += bytes;
        found->bytes -= bytes;
        if (found->bytes == 0U) free_.erase(found);
        return true;
    }

    void release(std::uint64_t offset, std::uint64_t bytes) noexcept {
        if (bytes == 0U) return;
        std::scoped_lock lock(mutex_);
        auto next = std::lower_bound(
            free_.begin(), free_.end(), offset,
            [](const Block& block, std::uint64_t value) {
                return block.offset < value;
            });
        if (next != free_.begin()) {
            auto previous = std::prev(next);
            if (previous->offset + previous->bytes == offset) {
                previous->bytes += bytes;
                if (next != free_.end() &&
                    previous->offset + previous->bytes == next->offset) {
                    previous->bytes += next->bytes;
                    free_.erase(next);
                }
                return;
            }
        }
        if (next != free_.end() && offset + bytes == next->offset) {
            next->offset = offset;
            next->bytes += bytes;
            return;
        }
        try {
            free_.insert(next, Block{offset, bytes});
        } catch (const std::bad_alloc&) {
            // Destructors cannot surface allocation failure. Quarantine the
            // untracked span and make future allocation fail explicitly.
            metadata_failed_ = true;
        }
    }

private:
    struct Block {
        std::uint64_t offset{};
        std::uint64_t bytes{};
    };

    int device_{-1};
    std::byte* base_{};
    std::uint64_t capacity_{};
    std::vector<Block> free_;
    std::mutex mutex_;
    bool metadata_failed_{};
};

struct Dsv4HostMoeCallbackState {
    CudaDsv4HostMoeCallback function{};
    CudaDsv4DeviceInputHostMoeCallback device_input_function{};
    // Optional first half. When present it runs as its own host node, an
    // event is recorded behind it, and the tier stream is released by that
    // event while this node's second half is still computing the host share.
    CudaDsv4DeviceInputHostMoeRouteCallback route_function{};
    bool route_failed{};
    void* context{};
    float* rank_partials{};
    std::uint64_t rank_partial_elements{};
    const std::uint16_t* encoded_hidden{};
    std::uint64_t hidden_elements{};
    const float* router_logits{};
    std::uint64_t router_elements{};
    const unsigned int* upstream_failure{};
    unsigned int upstream_failure_value{};
    bool failed{};
    std::chrono::steady_clock::time_point started{};
    std::chrono::steady_clock::time_point finished{};
};

constexpr std::uint32_t kDsv4FixedCommandCount = 43U;
constexpr std::uint64_t kDsv4DeferredAttentionPrepareUploadSlotBytes =
    16ULL << 10U;
constexpr std::uint64_t kDsv4DeferredAttentionUploadSlotBytes = 32ULL << 10U;
constexpr std::uint64_t kDsv4DeferredAttentionDownloadSlotBytes = 16ULL << 10U;

// Route half. Deliberately does no more than decide the route and publish the
// tier selection: every microsecond spent here is a microsecond the tier
// stream is still waiting, and the whole point of the split is that the tier
// starts early.
void CUDART_CB run_dsv4_host_moe_route_callback(void* opaque) {
    auto& state = *static_cast<Dsv4HostMoeCallbackState*>(opaque);
    bool accepted = false;
    try {
        state.upstream_failure_value = state.upstream_failure == nullptr
            ? 0U : *state.upstream_failure;
        if (state.upstream_failure_value == 0U &&
            state.route_function != nullptr &&
            state.encoded_hidden != nullptr && state.router_logits != nullptr) {
            accepted = state.route_function(
                state.context,
                std::span<const std::uint16_t>(state.encoded_hidden,
                                               state.hidden_elements),
                std::span<const float>(state.router_logits,
                                       state.router_elements));
        }
    } catch (...) {
        accepted = false;
    }
    state.route_failed = !accepted;
}

void CUDART_CB run_dsv4_host_moe_callback(void* opaque) {
    auto& state = *static_cast<Dsv4HostMoeCallbackState*>(opaque);
    state.started = std::chrono::steady_clock::now();
    bool accepted = false;
    try {
        state.upstream_failure_value = state.upstream_failure == nullptr
            ? 0U : *state.upstream_failure;
        // A failed route half is not short-circuited here: the callback owns
        // zeroing its own rank partials on failure, and skipping it would
        // leave the join reading whatever the previous layer left behind. It
        // sees the failure through its own state and fails the command.
        const bool upstream_accepted =
            state.upstream_failure_value == 0U;
        if (upstream_accepted && state.device_input_function != nullptr &&
            state.encoded_hidden != nullptr && state.router_logits != nullptr &&
            state.rank_partials != nullptr) {
            accepted = state.device_input_function(
                state.context,
                std::span<const std::uint16_t>(state.encoded_hidden,
                                               state.hidden_elements),
                std::span<const float>(state.router_logits,
                                       state.router_elements),
                std::span<float>(state.rank_partials,
                                 state.rank_partial_elements));
        } else if (upstream_accepted && state.function != nullptr &&
                   state.rank_partials != nullptr) {
            accepted = state.function(
                state.context,
                std::span<float>(state.rank_partials,
                                 state.rank_partial_elements));
        }
    } catch (...) {
        accepted = false;
    }
    state.failed = !accepted;
    state.finished = std::chrono::steady_clock::now();
}

struct Dsv4MhcHeadCallbackState {
    CudaDsv4MhcHeadCallback function{};
    void* context{};
    const std::uint16_t* encoded_hidden{};
    float* reduced{};
    bool failed{};
};

void CUDART_CB run_dsv4_mhc_head_callback(void* opaque) {
    auto& state = *static_cast<Dsv4MhcHeadCallbackState*>(opaque);
    bool accepted = false;
    try {
        if (state.function != nullptr && state.context != nullptr &&
            state.encoded_hidden != nullptr && state.reduced != nullptr) {
            accepted = state.function(
                state.context,
                std::span<const std::uint16_t>(
                    state.encoded_hidden, 4U * 4096U),
                std::span<float>(state.reduced, 4096U));
        }
    } catch (...) {
        accepted = false;
    }
    state.failed = !accepted;
}

struct Dsv4AttentionPrepareHostCommand {
    CudaDsv4AttentionPrepareHostCallback function{};
    void* context{};
    const std::uint16_t* query_rank{};
    const std::uint16_t* key_value{};
    const float* compressor_values{};
    const float* compressor_scores{};
    const float* index_compressor_values{};
    const float* index_compressor_scores{};
    std::uint64_t compressor_elements{};
    std::uint64_t index_compressor_elements{};
    std::byte* page_patches{};
    std::uint64_t page_patch_bytes{};
    const unsigned int* upstream_failure{};
    bool failed{};
};

void CUDART_CB run_dsv4_attention_prepare_host_callback(void* opaque) {
    auto& command = *static_cast<Dsv4AttentionPrepareHostCommand*>(opaque);
    bool accepted = false;
    try {
        if (command.function != nullptr && command.context != nullptr &&
            command.query_rank != nullptr && command.key_value != nullptr &&
            command.upstream_failure != nullptr &&
            *command.upstream_failure == 0U) {
            const CudaDsv4AttentionPrepareHostView view{
                std::span<const std::uint16_t>(command.query_rank, 1024U),
                std::span<const std::uint16_t>(command.key_value, 512U),
                std::span<const float>(command.compressor_values,
                                       command.compressor_elements),
                std::span<const float>(command.compressor_scores,
                                       command.compressor_elements),
                std::span<const float>(command.index_compressor_values,
                                       command.index_compressor_elements),
                std::span<const float>(command.index_compressor_scores,
                                       command.index_compressor_elements),
                std::span<std::byte>(command.page_patches,
                                     command.page_patch_bytes)};
            accepted = command.function(command.context, view);
        }
    } catch (...) {
        accepted = false;
    }
    command.failed = !accepted;
}


// One GLM-5.3 shared-expert dot per eight threads, associated exactly as the
// host AVX2 path associates it.
//
// The host sums an FP8 row into eight `__m256` accumulators -- 64 independent
// partial sums -- and combines them with a fixed tree. Floating-point addition
// is not associative, so a block reduction over the same products lands on a
// different float. `accumulator[lane]` here is one of those `__m256`
// registers: thread `group` owns `accumulators[group]` and all eight of its
// lanes, the fma is the intrinsic's fma(mul(decoded, scale), activation, acc),
// and the combine below is the same tree, so the device result is
// bit-identical rather than merely close.
//
// One thread per *group* rather than per row is what makes the reads
// coalesced: the eight threads of a row read the 64 bytes of a block
// contiguously and in order, as one `uint2` and two `float4` each. The
// row-per-thread form this replaced was equally exact and ran at 9.8 GB/s
// because each thread streamed a whole 4,096-byte row alone; this form
// measures 95.6 GB/s on gate and up and 159.0 on down, 9.8x and 8.1x, with
// zero mismatches against the host dot over all 8,192 output rows.
//
// The dot is returned raw. Every rounding and the SwiGLU stay on the host, so
// no device libm function -- expf above all -- enters the comparison.
__global__ void glm53_shared_expert_dot_kernel(
    float* output, const unsigned char* weights, const float* scales,
    const float* input, std::uint32_t rows, std::uint32_t columns,
    std::uint32_t scale_columns) {
    constexpr std::uint32_t kGroups = 8U;
    const std::uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t row = thread / kGroups;
    const std::uint32_t group = thread % kGroups;
    if (row >= rows) return;
    const unsigned char* weight_row =
        weights + static_cast<std::uint64_t>(row) * columns;
    const float* scale_row =
        scales + static_cast<std::uint64_t>(row / 128U) * scale_columns;
    float accumulator[8];
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) accumulator[lane] = 0.0F;
    for (std::uint32_t column = 0U; column + 64U <= columns; column += 64U) {
        const float scale = scale_row[column / 128U];
        const unsigned char* block = weight_row + column + group * 8U;
        const float* activation = input + column + group * 8U;
        const uint2 packed = *reinterpret_cast<const uint2*>(block);
        const float4 low = *reinterpret_cast<const float4*>(activation);
        const float4 high = *reinterpret_cast<const float4*>(activation + 4);
        const unsigned char* bytes =
            reinterpret_cast<const unsigned char*>(&packed);
        const float values[8] = {low.x, low.y, low.z, low.w,
                                 high.x, high.y, high.z, high.w};
#pragma unroll
        for (int lane = 0; lane < 8; ++lane) {
            accumulator[lane] = __fmaf_rn(
                __fmul_rn(fp8_e4m3_value(bytes[lane]), scale), values[lane],
                accumulator[lane]);
        }
    }
    // `for (width = 4; width; width >>= 1) acc[i] += acc[i + width]`, which is
    // a vector add over the eight lanes and therefore eight shuffles here.
    const unsigned mask = __activemask();
#pragma unroll
    for (int width = 4; width != 0; width >>= 1) {
#pragma unroll
        for (int lane = 0; lane < 8; ++lane) {
            const float other =
                __shfl_down_sync(mask, accumulator[lane], width, kGroups);
            if (static_cast<int>(group) < width) {
                accumulator[lane] = __fadd_rn(accumulator[lane], other);
            }
        }
    }
    if (group != 0U) return;
    // _mm_add_ps of the two 128-bit halves, then the two _mm_hadd_ps.
    float total[4];
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        total[lane] = __fadd_rn(accumulator[lane], accumulator[lane + 4]);
    }
    output[row] = __fadd_rn(__fadd_rn(total[0], total[1]),
                            __fadd_rn(total[2], total[3]));
}

}  // namespace
