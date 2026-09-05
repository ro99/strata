namespace {

constexpr std::size_t kDsv4Bf16SiluEntries = 1U << 16U;

std::uint64_t elapsed_nanoseconds_since(
    std::chrono::steady_clock::time_point started) noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count());
}

enum class SynchronizationSubsystem {
    Weight,
    Attention,
    Projection,
    Mhc,
    Moe,
    Other,
};

void record_synchronization(CudaBackendStats::Device& stats,
                            SynchronizationSubsystem subsystem,
                            std::uint64_t calls,
                            std::uint64_t nanoseconds) noexcept {
    stats.synchronization_calls += calls;
    stats.synchronization_nanoseconds += nanoseconds;
    CudaSynchronizationStats* target = nullptr;
    switch (subsystem) {
        case SynchronizationSubsystem::Weight:
            target = &stats.weight_synchronization;
            break;
        case SynchronizationSubsystem::Attention:
            target = &stats.attention_synchronization;
            break;
        case SynchronizationSubsystem::Projection:
            target = &stats.projection_synchronization;
            break;
        case SynchronizationSubsystem::Mhc:
            target = &stats.mhc_synchronization;
            break;
        case SynchronizationSubsystem::Moe:
            target = &stats.moe_synchronization;
            break;
        case SynchronizationSubsystem::Other:
            target = &stats.other_synchronization;
            break;
    }
    target->calls += calls;
    target->nanoseconds += nanoseconds;
}

std::uint64_t event_milliseconds_to_nanoseconds(
    float milliseconds, std::uint64_t& clamped_samples) noexcept {
    if (!std::isfinite(milliseconds) || milliseconds < 0.0F) {
        ++clamped_samples;
        return 0U;
    }
    return static_cast<std::uint64_t>(std::llround(
        static_cast<double>(milliseconds) * 1.0e6));
}

ValidationResult cuda_error(cudaError_t status, const char* operation) {
    ValidationResult result;
    if (status != cudaSuccess) {
        result.errors.emplace_back(std::string(operation) + ": " + cudaGetErrorString(status));
    }
    return result;
}

ValidationResult cublas_error(cublasStatus_t status, const char* operation) {
    ValidationResult result;
    if (status != CUBLAS_STATUS_SUCCESS) {
        result.errors.emplace_back(
            std::string(operation) + ": cuBLAS status " +
            std::to_string(static_cast<int>(status)));
    }
    return result;
}

__device__ float reduce_block(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFF'FFFFU, value, offset);
    }
    __shared__ float warps[8];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) warps[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? warps[lane] : 0.0F;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xFFFF'FFFFU, value, offset);
        }
    }
    return value;
}

__device__ double reduce_block_double(double value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFF'FFFFU, value, offset);
    }
    __shared__ double warps[32];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int warp_count = (static_cast<int>(blockDim.x) + 31) >> 5;
    if (lane == 0) warps[warp] = value;
    __syncthreads();
    value = static_cast<int>(threadIdx.x) < warp_count ? warps[lane] : 0.0;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xFFFF'FFFFU, value, offset);
        }
    }
    return value;
}

__device__ float bf16_round(float value);

__device__ float plain_value(const void* weights, int dtype, std::uint64_t index) {
    if (dtype == static_cast<int>(SafetensorsDtype::Bf16)) {
        return __bfloat162float(static_cast<const __nv_bfloat16*>(weights)[index]);
    }
    if (dtype == static_cast<int>(SafetensorsDtype::F16)) {
        return __half2float(static_cast<const __half*>(weights)[index]);
    }
    return static_cast<const float*>(weights)[index];
}

__device__ float fp8_e8m0_scale(unsigned char encoded) {
    return encoded == 0xFFU ? nanf("") : ldexpf(1.0F, static_cast<int>(encoded) - 127);
}

__device__ float fp8_e4m3_value(unsigned char encoded) {
    const bool negative = (encoded & 0x80U) != 0U;
    const unsigned int exponent = (encoded >> 3U) & 0x0FU;
    const unsigned int mantissa = encoded & 0x07U;
    float value = 0.0F;
    if (exponent == 0U) {
        value = ldexpf(static_cast<float>(mantissa) / 8.0F, -6);
    } else if (exponent == 0x0FU && mantissa == 0x07U) {
        return nanf("");
    } else {
        value = ldexpf(1.0F + static_cast<float>(mantissa) / 8.0F,
                       static_cast<int>(exponent) - 7);
    }
    return negative ? -value : value;
}

// E2M1 decode, branch-free.
//
// This ran as a sixteen-way switch, which nvcc lowers to a jump table or a
// branch tree and which executes once per weight element. Measured with ncu on
// the DeepSeek batch-1 expert kernels, that left them at 77% SM throughput
// against 6% DRAM throughput -- issue-bound, with the memory system idle
// (experiment 0124). The bytes were never the constraint; the decode was.
//
// E2M1 is a sign bit, a two-bit exponent with bias 1, and a one-bit mantissa,
// so the value is exactly a float built from bits. For exponent e >= 1 the
// magnitude is 2^(e-1) * (1 + m/2), which is the float whose exponent field is
// 126 + e and whose top mantissa bit is m. For e == 0 it is subnormal, m/2,
// giving 0 or 0.5. That case is the only one the bit construction cannot
// express, and it collapses to a single select.
//
// Exhaustively bit-identical to the switch over all sixteen encodings;
// tests/test_fp4_decode.cpp mirrors this construction and pins it against the
// declared table. Measured effect on the batch-1 expert path: 2.09 -> 1.06 ms
// per six-expert call on the 5060 Ti and 1.47 -> 0.77 ms on a 3090.
__device__ float fp4_e2m1_value(unsigned int encoded) {
    const unsigned int magnitude = encoded & 0x07U;
    const unsigned int exponent = magnitude >> 1U;
    const unsigned int mantissa = magnitude & 0x01U;
    const unsigned int normal =
        ((126U + exponent) << 23U) | (mantissa << 22U);
    // exponent 0 is subnormal: 0.0 for mantissa 0, 0.5 for mantissa 1.
    const unsigned int subnormal = mantissa == 0U ? 0U : 0x3F00'0000U;
    const unsigned int bits = exponent == 0U ? subnormal : normal;
    // Encoding 0x8 is negative zero in the bit construction but the declared
    // table gives +0.0, and the two differ in their bits even though they
    // compare equal. Suppress the sign when the magnitude is zero so the
    // decode is bit-identical, not merely numerically equal.
    const unsigned int sign =
        magnitude == 0U ? 0U : ((encoded & 0x08U) << 28U);
    return __uint_as_float(bits | sign);
}

__device__ float quantize_e4m3_value(float value) {
    const float magnitude = fminf(fabsf(value), 448.0F);
    float quantized = 0.0F;
    if (magnitude < 0.015625F) {
        quantized = rintf(ldexpf(magnitude, 9)) * ldexpf(1.0F, -9);
    } else {
        int exponent = 0;
        static_cast<void>(frexpf(magnitude, &exponent));
        exponent = max(-6, min(8, exponent - 1));
        const float step = ldexpf(1.0F, exponent - 3);
        quantized = fminf(rintf(magnitude / step) * step, 448.0F);
    }
    return copysignf(quantized, value);
}

__device__ unsigned char encode_e4m3_value(float value) {
    const unsigned char sign = signbit(value) ? 0x80U : 0U;
    const float magnitude = fabsf(value);
    if (magnitude == 0.0F) return sign;
    if (magnitude < 0.015625F) {
        const auto mantissa = static_cast<unsigned int>(
            rintf(ldexpf(magnitude, 9)));
        return static_cast<unsigned char>(sign | min(mantissa, 7U));
    }
    int exponent = 0;
    const float fraction = frexpf(magnitude, &exponent);
    const auto encoded_exponent = static_cast<unsigned int>(exponent + 6);
    const auto mantissa = static_cast<unsigned int>(
        rintf((fraction * 2.0F - 1.0F) * 8.0F));
    const auto maximum_mantissa = encoded_exponent == 15U ? 6U : 7U;
    return static_cast<unsigned char>(
        sign | (min(encoded_exponent, 15U) << 3U) |
        min(mantissa, maximum_mantissa));
}

__global__ void quantize_activation_e4m3_kernel(float* values,
                                                std::uint64_t columns,
                                                std::uint32_t rows) {
    const std::uint32_t row = blockIdx.y;
    const std::uint64_t group_begin = static_cast<std::uint64_t>(blockIdx.x) * 128U;
    if (row >= rows || group_begin >= columns) return;
    const std::uint64_t index = group_begin + threadIdx.x;
    const float magnitude = index < columns
                                ? fabsf(values[static_cast<std::uint64_t>(row) * columns + index])
                                : 0.0F;
    __shared__ float maximum[128];
    maximum[threadIdx.x] = magnitude;
    __syncthreads();
    for (unsigned int stride = 64U; stride != 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            maximum[threadIdx.x] = fmaxf(maximum[threadIdx.x],
                                         maximum[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (index >= columns) return;
    float scale = 1.0F;
    if (maximum[0] > 0.0F) scale = exp2f(ceilf(log2f(maximum[0] / 448.0F)));
    auto& value = values[static_cast<std::uint64_t>(row) * columns + index];
    value = quantize_e4m3_value(value / scale) * scale;
}

// GLM-5.3's compressed-tensors dynamic activation contract stores ordinary
// F32 inverse scales, not E8M0 powers of two. Simulate its per-token K128
// quantization in place so the scalar matmul consumes the same values as the
// fused FP8 kernel while retaining the existing F32 activation workspace.
__global__ void quantize_activation_e4m3_f32_scale_kernel(
    float* values, std::uint64_t columns, std::uint32_t rows) {
    const std::uint32_t row = blockIdx.y;
    const std::uint64_t group_begin =
        static_cast<std::uint64_t>(blockIdx.x) * 128U;
    if (row >= rows || group_begin >= columns) return;
    const std::uint64_t index = group_begin + threadIdx.x;
    const float magnitude = index < columns
        ? fabsf(values[static_cast<std::uint64_t>(row) * columns + index])
        : 0.0F;
    __shared__ float maximum[128];
    maximum[threadIdx.x] = magnitude;
    __syncthreads();
    for (unsigned int stride = 64U; stride != 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            maximum[threadIdx.x] = fmaxf(maximum[threadIdx.x],
                                         maximum[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (index >= columns) return;
    const float scale = maximum[0] > 0.0F ? maximum[0] / 448.0F : 1.0F;
    auto& value = values[static_cast<std::uint64_t>(row) * columns + index];
    value = quantize_e4m3_value(value / scale) * scale;
}

__global__ void round_bf16_rows_kernel(float* values, std::uint64_t elements) {
    for (std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += gridDim.x * blockDim.x) {
        values[index] = bf16_round(values[index]);
    }
}

// Compact form of the same activation simulation. The value byte is the
// unscaled E4M3 code and one E8M0 byte carries the per-row, per-K128 scale.
// Decoding code*scale reproduces quantize_activation_e4m3_kernel exactly but
// avoids retaining its four-byte encoded-value workspace beside the tensor
// projection output.
__global__ void quantize_activation_e4m3_bytes_kernel(
    unsigned char* values, unsigned char* scales, const float* source,
    std::uint64_t columns, std::uint32_t rows) {
    const std::uint32_t row = blockIdx.y;
    const std::uint64_t group_begin =
        static_cast<std::uint64_t>(blockIdx.x) * 128U;
    if (row >= rows || group_begin >= columns) return;
    const std::uint64_t index = group_begin + threadIdx.x;
    const float value = index < columns
                            ? source[static_cast<std::uint64_t>(row) * columns +
                                     index]
                            : 0.0F;
    __shared__ float maximum[128];
    maximum[threadIdx.x] = fabsf(value);
    __syncthreads();
    for (unsigned int stride = 64U; stride != 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            maximum[threadIdx.x] = fmaxf(maximum[threadIdx.x],
                                         maximum[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    int scale_exponent = 0;
    float scale = 1.0F;
    if (maximum[0] > 0.0F) {
        scale_exponent = static_cast<int>(ceilf(log2f(maximum[0] / 448.0F)));
        scale = exp2f(static_cast<float>(scale_exponent));
    }
    if (threadIdx.x == 0U) {
        scales[static_cast<std::uint64_t>(row) * gridDim.x + blockIdx.x] =
            static_cast<unsigned char>(scale_exponent + 127);
    }
    if (index < columns) {
        values[static_cast<std::uint64_t>(row) * columns + index] =
            encode_e4m3_value(quantize_e4m3_value(value / scale));
    }
}

// Compact continuous-scale activation used by GLM-5.3. This is byte-for-byte
// the same E4M3 quantization simulated by
// quantize_activation_e4m3_f32_scale_kernel, but keeps one raw code per value
// and one F32 scale per row/K128 block. Keeping the scale separate lets tensor
// cores dot exactly representable raw E4M3 values before the two continuous
// scales are applied in F32.
__global__ void quantize_activation_e4m3_f32_bytes_kernel(
    unsigned char* values, float* scales, const float* source,
    std::uint64_t columns, std::uint32_t rows) {
    const std::uint32_t row = blockIdx.y;
    const std::uint64_t group_begin =
        static_cast<std::uint64_t>(blockIdx.x) * 128U;
    if (row >= rows || group_begin >= columns) return;
    const std::uint64_t index = group_begin + threadIdx.x;
    const float value = index < columns
                            ? source[static_cast<std::uint64_t>(row) * columns +
                                     index]
                            : 0.0F;
    __shared__ float maximum[128];
    maximum[threadIdx.x] = fabsf(value);
    __syncthreads();
    for (unsigned int stride = 64U; stride != 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            maximum[threadIdx.x] = fmaxf(maximum[threadIdx.x],
                                         maximum[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float scale = maximum[0] > 0.0F ? maximum[0] / 448.0F : 1.0F;
    if (threadIdx.x == 0U) {
        scales[static_cast<std::uint64_t>(row) * gridDim.x + blockIdx.x] =
            scale;
    }
    if (index < columns) {
        values[static_cast<std::uint64_t>(row) * columns + index] =
            encode_e4m3_value(quantize_e4m3_value(value / scale));
    }
}

// The persistent mHC workspace stores its layer input as BF16. Decode and
// apply the same 128-column FP8 activation simulation in one launch so the
// shared expert sees exactly the values produced by the existing host bridge
// without an intervening H2D copy or an extra kernel.
__global__ void quantize_bf16_activation_e4m3_kernel(
    float* destination, const __nv_bfloat16* source,
    std::uint64_t columns) {
    const std::uint64_t group_begin =
        static_cast<std::uint64_t>(blockIdx.x) * 128U;
    if (group_begin >= columns) return;
    const std::uint64_t index = group_begin + threadIdx.x;
    const float value = index < columns
                            ? __bfloat162float(source[index])
                            : 0.0F;
    __shared__ float maximum[128];
    maximum[threadIdx.x] = fabsf(value);
    __syncthreads();
    for (unsigned int stride = 64U; stride != 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            maximum[threadIdx.x] = fmaxf(maximum[threadIdx.x],
                                         maximum[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (index >= columns) return;
    float scale = 1.0F;
    if (maximum[0] > 0.0F) {
        scale = exp2f(ceilf(log2f(maximum[0] / 448.0F)));
    }
    destination[index] = quantize_e4m3_value(value / scale) * scale;
}

__global__ void expand_bf16_activation_kernel(
    float* destination, const __nv_bfloat16* source,
    std::uint64_t columns) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < columns) destination[index] = __bfloat162float(source[index]);
}

// A block owns one output row and a tile of input rows. One block per (output
// row, input row) reads the whole weight row again for every input row, which
// for the 32,768 by 1,536 query projection is 263 GB a layer at a 2,612-token
// page -- 11.3 TB over the prompt, for 11.3 TFLOP of arithmetic. Tiling the
// input rows reads the weight once for the tile and leaves each output's
// reduction exactly as it was: the same block-wide tree over the same terms.
constexpr std::uint32_t kPlainMatmulRowTile = 16U;

// BF16-activation twin of plain_matmul_kernel for the attention output
// projection, whose input region is now BF16.
template <unsigned int Tile>
__global__ void plain_matmul_kernel_bf16_input(
    float* output, const __nv_bfloat16* input, const void* weights, int dtype,
    std::uint32_t batch, std::uint64_t columns, std::uint64_t rows,
    std::uint32_t groups, std::uint64_t rows_per_group) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t tile_begin = blockIdx.y * Tile;
    if (output_row >= rows || tile_begin >= batch) return;
    const std::uint32_t tile_rows = min(Tile, batch - tile_begin);
    const std::uint64_t weight_base = output_row * columns;
    float sum[Tile];
    std::uint64_t input_base[Tile];
#pragma unroll
    for (std::uint32_t index = 0U; index < Tile; ++index) {
        sum[index] = 0.0F;
        const std::uint32_t local = index < tile_rows ? index : 0U;
        const std::uint64_t batch_row = tile_begin + local;
        const std::uint64_t input_row = groups == 0U
            ? batch_row
            : batch_row * groups + output_row / rows_per_group;
        input_base[index] = input_row * columns;
    }
    for (std::uint64_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        const float weight = plain_value(weights, dtype, weight_base + column);
#pragma unroll
        for (std::uint32_t index = 0U; index < Tile; ++index) {
            sum[index] = fmaf(
                __bfloat162float(input[input_base[index] + column]), weight,
                sum[index]);
        }
    }
    __shared__ float reduction[Tile][32];
    const auto lane = threadIdx.x & 31U;
    const auto warp = threadIdx.x >> 5U;
#pragma unroll
    for (std::uint32_t index = 0U; index < Tile; ++index) {
        float value = sum[index];
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffff'ffffU, value, offset);
        }
        if (lane == 0U) reduction[index][warp] = value;
    }
    __syncthreads();
    if (threadIdx.x < Tile) {
        const auto warps = (blockDim.x + 31U) / 32U;
        float total = 0.0F;
        for (std::uint32_t index = 0U; index < warps; ++index) {
            total += reduction[threadIdx.x][index];
        }
        if (threadIdx.x < tile_rows) {
            output[(static_cast<std::uint64_t>(tile_begin) + threadIdx.x) *
                       rows + output_row] = total;
        }
    }
}

template <std::uint32_t Tile>
__global__ void plain_matmul_kernel(float* output, const float* input,
                                    const void* weights, int dtype,
                                    std::uint32_t batch, std::uint64_t columns,
                                    std::uint64_t rows, std::uint32_t groups,
                                    std::uint64_t rows_per_group) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t tile_begin = blockIdx.y * Tile;
    if (output_row >= rows || tile_begin >= batch) return;
    const std::uint32_t tile_rows = min(Tile, batch - tile_begin);
    const std::uint64_t weight_base = output_row * columns;
    float sum[Tile];
    std::uint64_t input_base[Tile];
#pragma unroll
    for (std::uint32_t index = 0U; index < Tile; ++index) {
        sum[index] = 0.0F;
        // Rows past the tail recompute row zero and are never written, which
        // keeps the accumulators statically indexed and in registers.
        const std::uint32_t local = index < tile_rows ? index : 0U;
        const std::uint64_t batch_row = tile_begin + local;
        const std::uint64_t input_row = groups == 0U
                                            ? batch_row
                                            : batch_row * groups +
                                                  output_row / rows_per_group;
        input_base[index] = input_row * columns;
    }
    for (std::uint64_t column = threadIdx.x; column < columns; column += blockDim.x) {
        const float weight = plain_value(weights, dtype, weight_base + column);
#pragma unroll
        for (std::uint32_t index = 0U; index < Tile; ++index) {
            sum[index] += weight * input[input_base[index] + column];
        }
    }
    // `tile_rows` is uniform across the block, so skipping the tail skips the
    // barrier for every thread alike. A batch of one therefore pays exactly
    // one reduction, as it did before the tile existed.
#pragma unroll
    for (std::uint32_t index = 0U; index < Tile; ++index) {
        if (index >= tile_rows) continue;
        __syncthreads();
        const float reduced = reduce_block(sum[index]);
        if (threadIdx.x != 0U) continue;
        const std::uint64_t output_index =
            static_cast<std::uint64_t>(tile_begin + index) * rows + output_row;
        output[output_index] = reduced;
    }
}

// Stage-4's rank-local attention contract uses a BF16-expanded weight and a
// two-half reduction.  Keep that association when the reusable executor
// supplies the expanded form; the native FP8 path below is intentionally not
// substituted because its unrounded product association is a different
// fixture contract.
// Reads a BF16 activation instead of FP32. Used for the attention output
// projection, whose input is the RoPE-decoded attention result: holding that
// as BF16 rather than FP32 halves the largest region in the page workspace.
__global__ void dsv4_rank_bf16_matmul_bf16_input(
    float* output, const __nv_bfloat16* input, const __nv_bfloat16* weights,
    std::uint32_t batch, std::uint64_t columns, std::uint64_t rows,
    std::uint32_t groups, std::uint64_t rows_per_group) {
    const auto output_row = static_cast<std::uint64_t>(blockIdx.x);
    const auto batch_row = static_cast<std::uint32_t>(blockIdx.y);
    if (output_row >= rows || batch_row >= batch) return;
    const auto input_row = groups == 0U
        ? batch_row
        : static_cast<std::uint64_t>(batch_row) * groups +
              output_row / rows_per_group;
    const auto input_base = input_row * columns;
    const auto weight_base = output_row * columns;
    float low = 0.0F;
    float high = 0.0F;
    for (std::uint64_t column = threadIdx.x; column < columns;
         column += 256U) {
        low += __bfloat162float(input[input_base + column]) *
               __bfloat162float(weights[weight_base + column]);
    }
    for (std::uint64_t column = threadIdx.x + 128U; column < columns;
         column += 256U) {
        high += __bfloat162float(input[input_base + column]) *
                __bfloat162float(weights[weight_base + column]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        low += __shfl_down_sync(0xffff'ffffU, low, offset);
        high += __shfl_down_sync(0xffff'ffffU, high, offset);
    }
    __shared__ float warps[8];
    const auto lane = threadIdx.x & 31U;
    const auto warp = threadIdx.x >> 5U;
    if (lane == 0U) {
        warps[warp] = low;
        warps[warp + 4U] = high;
    }
    __syncthreads();
    float reduced = threadIdx.x < 8U ? warps[lane] : 0.0F;
    for (int offset = 4; offset > 0; offset >>= 1) {
        reduced += __shfl_down_sync(0xffff'ffffU, reduced, offset);
    }
    if (threadIdx.x == 0U) {
        output[static_cast<std::uint64_t>(batch_row) * rows + output_row] =
            reduced;
    }
}

__global__ void dsv4_rank_bf16_matmul(
    float* output, const float* input, const __nv_bfloat16* weights,
    std::uint32_t batch, std::uint64_t columns, std::uint64_t rows,
    std::uint32_t groups, std::uint64_t rows_per_group) {
    const auto output_row = static_cast<std::uint64_t>(blockIdx.x);
    const auto batch_row = static_cast<std::uint32_t>(blockIdx.y);
    if (output_row >= rows || batch_row >= batch) return;
    const auto input_row = groups == 0U
        ? batch_row
        : static_cast<std::uint64_t>(batch_row) * groups +
              output_row / rows_per_group;
    const auto input_base = input_row * columns;
    const auto weight_base = output_row * columns;
    float low = 0.0F;
    float high = 0.0F;
    for (std::uint64_t column = threadIdx.x; column < columns;
         column += 256U) {
        low += input[input_base + column] *
               __bfloat162float(weights[weight_base + column]);
    }
    for (std::uint64_t column = threadIdx.x + 128U; column < columns;
         column += 256U) {
        high += input[input_base + column] *
                __bfloat162float(weights[weight_base + column]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        low += __shfl_down_sync(0xffff'ffffU, low, offset);
        high += __shfl_down_sync(0xffff'ffffU, high, offset);
    }
    __shared__ float warps[8];
    const auto lane = threadIdx.x & 31U;
    const auto warp = threadIdx.x >> 5U;
    if (lane == 0U) {
        warps[warp] = low;
        warps[warp + 4U] = high;
    }
    __syncthreads();
    float reduced = threadIdx.x < 8U ? warps[lane] : 0.0F;
    if (warp == 0U) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            reduced += __shfl_down_sync(0xffff'ffffU, reduced, offset);
        }
    }
    if (threadIdx.x == 0U) {
        output[static_cast<std::uint64_t>(batch_row) * rows + output_row] =
            reduced;
    }
}

__global__ void packed_matmul_kernel(float* output, const float* input,
                                     const std::uint32_t* packed,
                                     const __nv_bfloat16* scales,
                                     std::uint32_t bits, std::uint32_t group_size,
                                     std::uint64_t packed_columns,
                                     std::uint64_t scale_columns,
                                     std::uint32_t batch, std::uint64_t columns,
                                     std::uint64_t rows, std::uint32_t groups,
                                     std::uint64_t rows_per_group) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t batch_row = blockIdx.y;
    if (output_row >= rows || batch_row >= batch) return;
    const std::uint32_t lanes = 32U / bits;
    const std::uint32_t mask = (1U << bits) - 1U;
    const std::int32_t offset = 1 << (bits - 1U);
    float sum = 0.0F;
    const std::uint64_t packed_base = output_row * packed_columns;
    const std::uint64_t scale_base = output_row * scale_columns;
    const std::uint64_t input_row = groups == 0U
                                        ? batch_row
                                        : static_cast<std::uint64_t>(batch_row) *
                                              groups +
                                              output_row / rows_per_group;
    const std::uint64_t input_base = input_row * columns;
    for (std::uint64_t column = threadIdx.x; column < columns; column += blockDim.x) {
        const std::uint32_t word = packed[packed_base + column / lanes];
        const std::uint32_t raw = (word >> ((column % lanes) * bits)) & mask;
        const std::int32_t quantized = static_cast<std::int32_t>(raw) - offset;
        const std::uint64_t scale_column = group_size == 0U ? 0U : column / group_size;
        const float scale = __bfloat162float(scales[scale_base + scale_column]);
        sum += input[input_base + column] * static_cast<float>(quantized) * scale;
    }
    sum = reduce_block(sum);
    if (threadIdx.x == 0) {
        const std::uint64_t output_index =
            static_cast<std::uint64_t>(batch_row) * rows + output_row;
        output[output_index] = sum;
    }
}

__global__ void packed_int8_group32_matvec_kernel(
    float* output, const float* input, const std::uint32_t* packed,
    const __nv_bfloat16* scales, std::uint64_t packed_columns,
    std::uint64_t scale_columns, std::uint64_t columns,
    std::uint64_t rows) {
    constexpr unsigned int warps_per_block = 8U;
    const auto warp = threadIdx.x / warpSize;
    const auto lane = threadIdx.x % warpSize;
    const auto output_row = static_cast<std::uint64_t>(blockIdx.x) *
                                warps_per_block + warp;
    if (output_row >= rows) return;

    const auto packed_base = output_row * packed_columns;
    const auto scale_base = output_row * scale_columns;
    float sum = 0.0F;
    for (std::uint64_t packed_column = lane; packed_column < packed_columns;
         packed_column += warpSize) {
        const auto word = packed[packed_base + packed_column];
        const auto column = packed_column * 4U;
        const float scale = __bfloat162float(
            scales[scale_base + column / 32U]);
#pragma unroll
        for (unsigned int item = 0U; item < 4U; ++item) {
            const auto raw = static_cast<std::uint8_t>(word >> (item * 8U));
            sum += input[column + item] *
                   static_cast<float>(static_cast<int>(raw) - 128) * scale;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum = __fadd_rn(
            sum, __shfl_down_sync(0xFFFF'FFFFU, sum, offset));
    }
    if (lane == 0U) output[output_row] = bf16_round(sum);
}

__global__ void bf16_matvec_kernel(
    float* output, const float* input, const __nv_bfloat16* weights,
    std::uint64_t columns, std::uint64_t rows) {
    constexpr unsigned int warps_per_block = 8U;
    const auto warp = threadIdx.x / warpSize;
    const auto lane = threadIdx.x % warpSize;
    const auto output_row = static_cast<std::uint64_t>(blockIdx.x) *
                                warps_per_block + warp;
    if (output_row >= rows) return;

    const auto base = output_row * columns;
    float sum = 0.0F;
    for (std::uint64_t column = lane; column < columns; column += warpSize) {
        sum = __fadd_rn(
            sum,
            __fmul_rn(input[column],
                      __bfloat162float(weights[base + column])));
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum = __fadd_rn(
            sum, __shfl_down_sync(0xFFFF'FFFFU, sum, offset));
    }
    if (lane == 0U) output[output_row] = sum;
}

// bf16_matvec_kernel over a tile of input rows. The 256-thread block kernel is
// shaped for a long reduction: at the 32,768 by 1,024 query projection it gives
// each thread four columns and then runs a block-wide tree per output, so the
// reductions cost more than the arithmetic. One warp per output row keeps the
// same __fadd_rn/__fmul_rn accumulation and the same shuffle tree as the
// batch-of-one path, so a row of this is bit-identical to a call of that.
constexpr std::uint32_t kBf16MatvecRowTile = 16U;

template <std::uint32_t Tile>
__global__ void bf16_matvec_rows_kernel(
    float* output, const float* input, const __nv_bfloat16* weights,
    std::uint32_t batch, std::uint64_t columns, std::uint64_t rows) {
    constexpr unsigned int warps_per_block = 8U;
    const auto warp = threadIdx.x / warpSize;
    const auto lane = threadIdx.x % warpSize;
    const auto output_row = static_cast<std::uint64_t>(blockIdx.x) *
                                warps_per_block + warp;
    const std::uint32_t tile_begin = blockIdx.y * Tile;
    if (output_row >= rows || tile_begin >= batch) return;
    const std::uint32_t tile_rows = min(Tile, batch - tile_begin);

    const auto base = output_row * columns;
    float sum[Tile];
#pragma unroll
    for (std::uint32_t index = 0U; index < Tile; ++index) sum[index] = 0.0F;
    for (std::uint64_t column = lane; column < columns; column += warpSize) {
        const float weight = __bfloat162float(weights[base + column]);
#pragma unroll
        for (std::uint32_t index = 0U; index < Tile; ++index) {
            const std::uint32_t local = index < tile_rows ? index : 0U;
            const auto input_base =
                static_cast<std::uint64_t>(tile_begin + local) * columns;
            sum[index] = __fadd_rn(
                sum[index], __fmul_rn(input[input_base + column], weight));
        }
    }
#pragma unroll
    for (std::uint32_t index = 0U; index < Tile; ++index) {
        float value = sum[index];
        for (int offset = 16; offset > 0; offset >>= 1) {
            value = __fadd_rn(
                value, __shfl_down_sync(0xFFFF'FFFFU, value, offset));
        }
        if (lane == 0U && index < tile_rows) {
            output[static_cast<std::uint64_t>(tile_begin + index) * rows +
                   output_row] = value;
        }
    }
}

// The same reduction as bf16_matvec_rows_kernel, with the model's existing
// BF16 projection boundary fused into the store. Each output is converted only
// after the identical lane-local accumulation and shuffle tree have completed,
// so reading it back as F32 is bit-equivalent to the old F32-output kernel
// followed by round_bf16_rows_kernel.
template <std::uint32_t Tile>
__global__ void bf16_matvec_rows_to_bf16_kernel(
    __nv_bfloat16* output, const float* input,
    const __nv_bfloat16* weights, std::uint32_t batch,
    std::uint64_t columns, std::uint64_t rows) {
    constexpr unsigned int warps_per_block = 8U;
    const auto warp = threadIdx.x / warpSize;
    const auto lane = threadIdx.x % warpSize;
    const auto output_row = static_cast<std::uint64_t>(blockIdx.x) *
                                warps_per_block + warp;
    const std::uint32_t tile_begin = blockIdx.y * Tile;
    if (output_row >= rows || tile_begin >= batch) return;
    const std::uint32_t tile_rows = min(Tile, batch - tile_begin);

    const auto base = output_row * columns;
    float sum[Tile];
#pragma unroll
    for (std::uint32_t index = 0U; index < Tile; ++index) sum[index] = 0.0F;
    for (std::uint64_t column = lane; column < columns; column += warpSize) {
        const float weight = __bfloat162float(weights[base + column]);
#pragma unroll
        for (std::uint32_t index = 0U; index < Tile; ++index) {
            const std::uint32_t local = index < tile_rows ? index : 0U;
            const auto input_base =
                static_cast<std::uint64_t>(tile_begin + local) * columns;
            sum[index] = __fadd_rn(
                sum[index], __fmul_rn(input[input_base + column], weight));
        }
    }
#pragma unroll
    for (std::uint32_t index = 0U; index < Tile; ++index) {
        float value = sum[index];
        for (int offset = 16; offset > 0; offset >>= 1) {
            value = __fadd_rn(
                value, __shfl_down_sync(0xFFFF'FFFFU, value, offset));
        }
        if (lane == 0U && index < tile_rows) {
            output[static_cast<std::uint64_t>(tile_begin + index) * rows +
                   output_row] = __float2bfloat16_rn(value);
        }
    }
}

// Sparse MLA's gathered expansion keeps a separate entry point from the
// dense cache builder above so the <=index_topk identity oracle retains two
// independent implementations. Apart from selecting the input row, its
// multiply/add order and BF16 store are element-for-element identical to
// bf16_matvec_rows_to_bf16_kernel.
template <std::uint32_t Tile>
__global__ void bf16_gathered_matvec_rows_to_bf16_kernel(
    __nv_bfloat16* output, const float* input,
    const std::uint32_t* selected_rows,
    const std::uint32_t* destination_rows,
    const __nv_bfloat16* weights, std::uint32_t batch,
    std::uint64_t columns, std::uint64_t rows) {
    constexpr unsigned int warps_per_block = 8U;
    const auto warp = threadIdx.x / warpSize;
    const auto lane = threadIdx.x % warpSize;
    const auto output_row = static_cast<std::uint64_t>(blockIdx.x) *
                                warps_per_block + warp;
    const std::uint32_t tile_begin = blockIdx.y * Tile;
    if (output_row >= rows || tile_begin >= batch) return;
    const std::uint32_t tile_rows = min(Tile, batch - tile_begin);

    const auto base = output_row * columns;
    float sum[Tile];
#pragma unroll
    for (std::uint32_t index = 0U; index < Tile; ++index) sum[index] = 0.0F;
    for (std::uint64_t column = lane; column < columns; column += warpSize) {
        const float weight = __bfloat162float(weights[base + column]);
#pragma unroll
        for (std::uint32_t index = 0U; index < Tile; ++index) {
            const std::uint32_t local = index < tile_rows ? index : 0U;
            const auto selected = selected_rows[tile_begin + local];
            const auto input_base =
                static_cast<std::uint64_t>(selected) * columns;
            sum[index] = __fadd_rn(
                sum[index], __fmul_rn(input[input_base + column], weight));
        }
    }
#pragma unroll
    for (std::uint32_t index = 0U; index < Tile; ++index) {
        float value = sum[index];
        for (int offset = 16; offset > 0; offset >>= 1) {
            value = __fadd_rn(
                value, __shfl_down_sync(0xFFFF'FFFFU, value, offset));
        }
        if (lane == 0U && index < tile_rows) {
            output[static_cast<std::uint64_t>(
                       destination_rows[tile_begin + index]) * rows +
                   output_row] = __float2bfloat16_rn(value);
        }
    }
}

// Same accumulation contract as bf16_matvec_kernel, with an already-resident
// BF16 activation source. The generic host bridge expands this exact source to
// FP32 before upload; decoding in the multiply loop is bit-equivalent and
// avoids both that upload and a separate expansion kernel.
__global__ void bf16_input_matvec_kernel(
    float* output, const __nv_bfloat16* input,
    const __nv_bfloat16* weights, std::uint64_t columns,
    std::uint64_t rows) {
    constexpr unsigned int warps_per_block = 8U;
    const auto warp = threadIdx.x / warpSize;
    const auto lane = threadIdx.x % warpSize;
    const auto output_row = static_cast<std::uint64_t>(blockIdx.x) *
                                warps_per_block + warp;
    if (output_row >= rows) return;
    const auto base = output_row * columns;
    float sum = 0.0F;
    for (std::uint64_t column = lane; column < columns; column += warpSize) {
        sum = __fadd_rn(
            sum,
            __fmul_rn(__bfloat162float(input[column]),
                      __bfloat162float(weights[base + column])));
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum = __fadd_rn(
            sum, __shfl_down_sync(0xFFFF'FFFFU, sum, offset));
    }
    if (lane == 0U) output[output_row] = sum;
}

__global__ void gemma4_softcap_logits_kernel(
    float* values, std::uint64_t count, float softcap) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= count) return;
    float value = bf16_round(values[index]);
    value = bf16_round(value / softcap);
    value = bf16_round(tanhf(value));
    values[index] = bf16_round(value * softcap);
}

__device__ void gemma4_norm_vector_block(
    float* output, const float* input, const float* weight,
    std::uint32_t columns, float epsilon, unsigned int* error_flag) {
    double squared_sum = 0.0;
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        const float value = bf16_round(input[column]);
        if (!isfinite(value) || !isfinite(weight[column])) {
            atomicExch(error_flag, 1U);
        }
        squared_sum = __dadd_rn(
            squared_sum,
            __dmul_rn(static_cast<double>(value), static_cast<double>(value)));
    }
    squared_sum = reduce_block_double(squared_sum);
    __shared__ float reciprocal;
    if (threadIdx.x == 0U) {
        reciprocal = 1.0F / sqrtf(
            static_cast<float>(squared_sum / static_cast<double>(columns)) +
            epsilon);
    }
    __syncthreads();
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        output[column] = bf16_round(
            bf16_round(input[column]) * reciprocal * weight[column]);
    }
    __syncthreads();
}

__global__ void gemma4_rms_norm_kernel(
    float* output, const float* input, const float* weight,
    std::uint32_t columns, float epsilon, unsigned int* error_flag) {
    if (blockIdx.x != 0U) return;
    gemma4_norm_vector_block(
        output, input, weight, columns, epsilon, error_flag);
}

__global__ void gemma4_rms_norm_rows_kernel(
    float* output, const float* input, const float* weight,
    std::uint32_t rows, std::uint32_t columns, float epsilon,
    unsigned int* error_flag) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x);
    if (row >= rows) return;
    const auto offset = static_cast<std::uint64_t>(row) * columns;
    gemma4_norm_vector_block(output + offset, input + offset, weight, columns,
                             epsilon, error_flag);
}

__global__ void gemma4_norm_rope_kernel(
    float* values, const float* weight, std::uint32_t heads,
    std::uint32_t head_dim, std::uint32_t position, float theta,
    float rotary_proportion, unsigned int* error_flag) {
    const auto head = static_cast<std::uint32_t>(blockIdx.x);
    if (head >= heads || threadIdx.x >= 32U) return;
    auto* row = values + static_cast<std::uint64_t>(head) * head_dim;
    double squared_sum = 0.0;
    int valid = 1;
    for (std::uint32_t column = threadIdx.x; column < head_dim;
         column += 32U) {
        const float value = bf16_round(row[column]);
        if (!isfinite(value) ||
            (weight != nullptr && !isfinite(weight[column]))) {
            valid = 0;
        } else {
            squared_sum = __dadd_rn(
                squared_sum, __dmul_rn(static_cast<double>(value),
                                       static_cast<double>(value)));
        }
    }
    if (__ballot_sync(0xffff'ffffU, valid != 0) != 0xffff'ffffU) {
        if (threadIdx.x == 0U) atomicExch(error_flag, 1U);
        return;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        squared_sum = __dadd_rn(
            squared_sum,
            __shfl_down_sync(0xffff'ffffU, squared_sum, offset));
    }
    float reciprocal = threadIdx.x == 0U
        ? 1.0F / sqrtf(
              static_cast<float>(squared_sum / static_cast<double>(head_dim)) +
              1.0e-6F)
        : 0.0F;
    reciprocal = __shfl_sync(0xffff'ffffU, reciprocal, 0);
    for (std::uint32_t column = threadIdx.x; column < head_dim;
         column += 32U) {
        row[column] = bf16_round(
            bf16_round(row[column]) * reciprocal *
            (weight == nullptr ? 1.0F : weight[column]));
    }
    __syncwarp();
    if (theta == 0.0F) return;
    const auto half = head_dim / 2U;
    const auto angles = static_cast<std::uint32_t>(
        rotary_proportion * static_cast<float>(head_dim) / 2.0F);
    for (std::uint32_t index = threadIdx.x; index < angles; index += 32U) {
        const float first = row[index];
        const float second = row[half + index];
        const float inverse_frequency = powf(
            theta, -2.0F * static_cast<float>(index) /
                       static_cast<float>(head_dim));
        const float angle = static_cast<float>(position) * inverse_frequency;
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        row[index] = bf16_round(first * cosine - second * sine);
        row[half + index] = bf16_round(second * cosine + first * sine);
    }
}

__global__ void gemma4_norm_rope_qkv_rows_kernel(
    float* queries, float* keys, float* values,
    const float* query_weight, const float* key_weight, std::uint32_t rows,
    std::uint32_t query_heads, std::uint32_t key_value_heads,
    std::uint32_t head_dim,
    std::uint32_t position_base, float theta, float rotary_proportion,
    unsigned int* error_flag) {
    const auto combined_head = static_cast<std::uint32_t>(blockIdx.x);
    const auto token_row = static_cast<std::uint32_t>(blockIdx.y);
    if (combined_head >= query_heads + 2U * key_value_heads ||
        token_row >= rows || threadIdx.x >= 32U) return;
    float* tensor = queries;
    const float* weight = query_weight;
    std::uint32_t heads = query_heads;
    std::uint32_t head = combined_head;
    float own_theta = theta;
    float own_rotary_proportion = rotary_proportion;
    if (combined_head >= query_heads + key_value_heads) {
        tensor = values;
        weight = nullptr;
        heads = key_value_heads;
        head = combined_head - query_heads - key_value_heads;
        own_theta = 0.0F;
        own_rotary_proportion = 1.0F;
    } else if (combined_head >= query_heads) {
        tensor = keys;
        weight = key_weight;
        heads = key_value_heads;
        head = combined_head - query_heads;
    }
    auto* row = tensor +
        (static_cast<std::uint64_t>(token_row) * heads + head) * head_dim;
    double squared_sum = 0.0;
    int valid = 1;
    for (std::uint32_t column = threadIdx.x; column < head_dim;
         column += 32U) {
        const float value = bf16_round(row[column]);
        if (!isfinite(value) ||
            (weight != nullptr && !isfinite(weight[column]))) {
            valid = 0;
        } else {
            squared_sum = __dadd_rn(
                squared_sum, __dmul_rn(static_cast<double>(value),
                                       static_cast<double>(value)));
        }
    }
    if (__ballot_sync(0xffff'ffffU, valid != 0) != 0xffff'ffffU) {
        if (threadIdx.x == 0U) atomicExch(error_flag, 1U);
        return;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        squared_sum = __dadd_rn(
            squared_sum,
            __shfl_down_sync(0xffff'ffffU, squared_sum, offset));
    }
    float reciprocal = threadIdx.x == 0U
        ? 1.0F / sqrtf(
              static_cast<float>(squared_sum / static_cast<double>(head_dim)) +
              1.0e-6F)
        : 0.0F;
    reciprocal = __shfl_sync(0xffff'ffffU, reciprocal, 0);
    for (std::uint32_t column = threadIdx.x; column < head_dim;
         column += 32U) {
        row[column] = bf16_round(
            bf16_round(row[column]) * reciprocal *
            (weight == nullptr ? 1.0F : weight[column]));
    }
    __syncwarp();
    if (own_theta == 0.0F) return;
    const auto half = head_dim / 2U;
    const auto angles = static_cast<std::uint32_t>(
        own_rotary_proportion * static_cast<float>(head_dim) / 2.0F);
    const auto position = position_base + token_row;
    for (std::uint32_t index = threadIdx.x; index < angles; index += 32U) {
        const float first = row[index];
        const float second = row[half + index];
        const float inverse_frequency = powf(
            own_theta, -2.0F * static_cast<float>(index) /
                           static_cast<float>(head_dim));
        const float angle = static_cast<float>(position) * inverse_frequency;
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        row[index] = bf16_round(first * cosine - second * sine);
        row[half + index] = bf16_round(second * cosine + first * sine);
    }
}

__global__ void gemma4_store_kv_kernel(
    __nv_bfloat16* cache, const float* keys, const float* values,
    std::uint32_t position, std::uint32_t capacity_rows,
    std::uint32_t columns) {
    const auto column = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (column >= columns) return;
    const auto row = position % capacity_rows;
    const auto offset = static_cast<std::uint64_t>(row) * columns + column;
    const auto plane = static_cast<std::uint64_t>(capacity_rows) * columns;
    cache[offset] = __float2bfloat16_rn(keys[column]);
    cache[plane + offset] = __float2bfloat16_rn(values[column]);
}

__global__ void gemma4_store_kv_rows_kernel(
    __nv_bfloat16* cache, const float* keys, const float* values,
    std::uint32_t position_base, std::uint32_t rows,
    std::uint32_t capacity_rows, std::uint32_t columns) {
    const auto column = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    const auto row = static_cast<std::uint32_t>(blockIdx.y);
    if (column >= columns || row >= rows) return;
    const auto physical = (position_base + row) % capacity_rows;
    const auto destination =
        static_cast<std::uint64_t>(physical) * columns + column;
    const auto source = static_cast<std::uint64_t>(row) * columns + column;
    const auto plane = static_cast<std::uint64_t>(capacity_rows) * columns;
    cache[destination] = __float2bfloat16_rn(keys[source]);
    cache[plane + destination] = __float2bfloat16_rn(values[source]);
}

// Text-page attention over the persistent history plus the current page. A
// warp owns each key-row dot so its reads are coalesced; thread zero preserves
// the reference softmax order, and each value column preserves the scalar row
// accumulation order. Current-page K/V is consumed directly and rounded at
// the cache boundary. Committing it only after attention avoids overwriting
// history that early rows still need when a sliding-window ring wraps.
__global__ void gemma4_prefill_attention_kernel(
    float* output, const float* queries, const float* current_keys,
    const float* current_values, const __nv_bfloat16* cache,
    std::uint32_t rows, std::uint32_t position_base,
    std::uint32_t capacity_rows, std::uint32_t query_heads,
    std::uint32_t key_value_heads, std::uint32_t head_dim,
    unsigned int* error_flag) {
    extern __shared__ float scores[];
    const auto head = static_cast<std::uint32_t>(blockIdx.x);
    const auto query_row = static_cast<std::uint32_t>(blockIdx.y);
    if (head >= query_heads || query_row >= rows) return;
    const auto query_position = position_base + query_row;
    const auto visible = min(query_position + 1U, capacity_rows);
    const auto first_position = query_position + 1U - visible;
    const auto kv_head = head / (query_heads / key_value_heads);
    const auto safe_query_row = min(query_row, rows - 1U);
    const auto* query = queries +
        (static_cast<std::uint64_t>(safe_query_row) * query_heads + head) *
            head_dim;
    const auto plane = static_cast<std::uint64_t>(capacity_rows) *
                       key_value_heads * head_dim;
    const auto lane = static_cast<std::uint32_t>(threadIdx.x) & 31U;
    const auto warp = static_cast<std::uint32_t>(threadIdx.x) >> 5U;
    const auto warps = static_cast<std::uint32_t>(blockDim.x) >> 5U;
    for (std::uint32_t key_row = warp; key_row < visible; key_row += warps) {
        const auto absolute = first_position + key_row;
        float score = 0.0F;
        for (std::uint32_t column = lane; column < head_dim; column += 32U) {
            float key_value = 0.0F;
            if (absolute < position_base) {
                const auto physical = absolute % capacity_rows;
                const auto offset =
                    (static_cast<std::uint64_t>(physical) * key_value_heads +
                     kv_head) * head_dim + column;
                key_value = __bfloat162float(cache[offset]);
            } else {
                const auto current_row = absolute - position_base;
                const auto offset =
                    (static_cast<std::uint64_t>(current_row) * key_value_heads +
                     kv_head) * head_dim + column;
                key_value = __bfloat162float(
                    __float2bfloat16_rn(current_keys[offset]));
            }
            score = __fadd_rn(
                score, __fmul_rn(query[column], key_value));
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            score = __fadd_rn(
                score, __shfl_down_sync(0xffff'ffffU, score, offset));
        }
        if (lane == 0U) {
            scores[key_row] = score;
            if (!isfinite(score)) atomicExch(error_flag, 2U);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0U) {
        float maximum = -INFINITY;
        for (std::uint32_t row = 0U; row < visible; ++row) {
            maximum = fmaxf(maximum, scores[row]);
        }
        scores[visible] = 0.0F;
        for (std::uint32_t row = 0U; row < visible; ++row) {
            scores[row] = expf(__fsub_rn(scores[row], maximum));
            scores[visible] = __fadd_rn(scores[visible], scores[row]);
        }
        if (!isfinite(scores[visible]) || scores[visible] <= 0.0F) {
            atomicExch(error_flag, 3U);
        }
        for (std::uint32_t row = 0U; row < visible; ++row) {
            scores[row] = __fdiv_rn(scores[row], scores[visible]);
        }
    }
    __syncthreads();
    auto* destination = output +
        (static_cast<std::uint64_t>(query_row) * query_heads + head) * head_dim;
    for (std::uint32_t column = threadIdx.x; column < head_dim;
         column += blockDim.x) {
        float sum = 0.0F;
        for (std::uint32_t row = 0U; row < visible; ++row) {
            const auto absolute = first_position + row;
            float value = 0.0F;
            if (absolute < position_base) {
                const auto physical = absolute % capacity_rows;
                const auto offset =
                    (static_cast<std::uint64_t>(physical) * key_value_heads +
                     kv_head) * head_dim + column;
                value = __bfloat162float(cache[plane + offset]);
            } else {
                const auto current_row = absolute - position_base;
                const auto offset =
                    (static_cast<std::uint64_t>(current_row) * key_value_heads +
                     kv_head) * head_dim + column;
                value = __bfloat162float(
                    __float2bfloat16_rn(current_values[offset]));
            }
            sum = __fadd_rn(sum, __fmul_rn(scores[row], value));
        }
        destination[column] = bf16_round(sum);
        if (!isfinite(sum)) atomicExch(error_flag, 4U);
    }
}

// Grouped-query page attention. One CTA shares each BF16 K/V tile across all
// query heads that own it. Local attention also shares the tile across four
// adjacent query rows, filling the eight warps of the CTA. Every individual
// dot, softmax, and value sum retains the scalar order used above; only the
// redundant cache reads are removed.
__global__ void gemma4_grouped_prefill_attention_kernel(
    float* output, const float* queries, const float* current_keys,
    const float* current_values, const __nv_bfloat16* cache,
    std::uint32_t rows, std::uint32_t position_base,
    std::uint32_t capacity_rows, std::uint32_t query_heads,
    std::uint32_t key_value_heads, std::uint32_t head_dim,
    std::uint32_t queries_per_block, unsigned int* error_flag) {
    constexpr std::uint32_t tile_rows = 8U;
    extern __shared__ unsigned char shared_bytes[];
    const auto group_size = query_heads / key_value_heads;
    const auto combined = queries_per_block * group_size;
    const auto maximum_visible = min(position_base + rows, capacity_rows);
    const auto score_stride = maximum_visible + 1U;
    auto* scores = reinterpret_cast<float*>(shared_bytes);
    auto* tile = reinterpret_cast<__nv_bfloat16*>(
        scores + static_cast<std::uint64_t>(combined) * score_stride);

    const auto kv_head = static_cast<std::uint32_t>(blockIdx.x);
    const auto query_begin = static_cast<std::uint32_t>(blockIdx.y) *
                             queries_per_block;
    const auto query_end = min(query_begin + queries_per_block, rows);
    const auto warp = static_cast<std::uint32_t>(threadIdx.x) >> 5U;
    const auto lane = static_cast<std::uint32_t>(threadIdx.x) & 31U;
    const auto local_query = warp / group_size;
    const auto group_head = warp % group_size;
    const auto query_row = query_begin + local_query;
    const bool active = warp < combined && query_row < query_end;
    const auto head = kv_head * group_size + group_head;
    const auto query_position = position_base + query_row;
    const auto visible = active
        ? min(query_position + 1U, capacity_rows) : 0U;
    const auto first_position = active
        ? query_position + 1U - visible : 0U;
    const auto first_query_position = position_base + query_begin;
    const auto common_visible = min(first_query_position + 1U, capacity_rows);
    const auto common_first = first_query_position + 1U - common_visible;
    const auto common_last = position_base + query_end;
    const auto plane = static_cast<std::uint64_t>(capacity_rows) *
                       key_value_heads * head_dim;
    const auto safe_query_row = min(query_row, rows - 1U);
    const auto* query = queries +
        (static_cast<std::uint64_t>(safe_query_row) * query_heads + head) *
            head_dim;

    for (std::uint32_t base = common_first; base < common_last;
         base += tile_rows) {
        for (std::uint32_t element = threadIdx.x;
             element < tile_rows * head_dim; element += blockDim.x) {
            const auto tile_row = element / head_dim;
            const auto column = element % head_dim;
            const auto absolute = base + tile_row;
            float value = 0.0F;
            if (absolute < common_last) {
                if (absolute < position_base) {
                    const auto physical = absolute % capacity_rows;
                    const auto offset =
                        (static_cast<std::uint64_t>(physical) *
                             key_value_heads + kv_head) * head_dim + column;
                    value = __bfloat162float(cache[offset]);
                } else {
                    const auto current_row = absolute - position_base;
                    const auto offset =
                        (static_cast<std::uint64_t>(current_row) *
                             key_value_heads + kv_head) * head_dim + column;
                    value = current_keys[offset];
                }
            }
            tile[element] = __float2bfloat16_rn(value);
        }
        __syncthreads();
        if (active) {
            for (std::uint32_t tile_row = 0U; tile_row < tile_rows;
                 ++tile_row) {
                const auto absolute = base + tile_row;
                if (absolute < first_position || absolute > query_position) {
                    continue;
                }
                float score = 0.0F;
                for (std::uint32_t column = lane; column < head_dim;
                     column += 32U) {
                    score = __fadd_rn(
                        score, __fmul_rn(
                            query[column],
                            __bfloat162float(tile[tile_row * head_dim +
                                                  column])));
                }
                for (int offset = 16; offset > 0; offset /= 2) {
                    score = __fadd_rn(
                        score,
                        __shfl_down_sync(0xffff'ffffU, score, offset));
                }
                if (lane == 0U) {
                    scores[static_cast<std::uint64_t>(warp) * score_stride +
                           absolute - first_position] = score;
                    if (!isfinite(score)) atomicExch(error_flag, 2U);
                }
            }
        }
        __syncthreads();
    }

    if (active && lane == 0U) {
        auto* own_scores = scores +
            static_cast<std::uint64_t>(warp) * score_stride;
        float maximum = -INFINITY;
        for (std::uint32_t row = 0U; row < visible; ++row) {
            maximum = fmaxf(maximum, own_scores[row]);
        }
        own_scores[visible] = 0.0F;
        for (std::uint32_t row = 0U; row < visible; ++row) {
            own_scores[row] = expf(__fsub_rn(own_scores[row], maximum));
            own_scores[visible] =
                __fadd_rn(own_scores[visible], own_scores[row]);
        }
        if (!isfinite(own_scores[visible]) || own_scores[visible] <= 0.0F) {
            atomicExch(error_flag, 3U);
        }
        for (std::uint32_t row = 0U; row < visible; ++row) {
            own_scores[row] = __fdiv_rn(own_scores[row], own_scores[visible]);
        }
    }
    __syncthreads();

    float sums[16]{};
    for (std::uint32_t base = common_first; base < common_last;
         base += tile_rows) {
        for (std::uint32_t element = threadIdx.x;
             element < tile_rows * head_dim; element += blockDim.x) {
            const auto tile_row = element / head_dim;
            const auto column = element % head_dim;
            const auto absolute = base + tile_row;
            float value = 0.0F;
            if (absolute < common_last) {
                if (absolute < position_base) {
                    const auto physical = absolute % capacity_rows;
                    const auto offset =
                        (static_cast<std::uint64_t>(physical) *
                             key_value_heads + kv_head) * head_dim + column;
                    value = __bfloat162float(cache[plane + offset]);
                } else {
                    const auto current_row = absolute - position_base;
                    const auto offset =
                        (static_cast<std::uint64_t>(current_row) *
                             key_value_heads + kv_head) * head_dim + column;
                    value = current_values[offset];
                }
            }
            tile[element] = __float2bfloat16_rn(value);
        }
        __syncthreads();
        if (active) {
            const auto* own_scores = scores +
                static_cast<std::uint64_t>(warp) * score_stride;
            for (std::uint32_t tile_row = 0U; tile_row < tile_rows;
                 ++tile_row) {
                const auto absolute = base + tile_row;
                if (absolute < first_position || absolute > query_position) {
                    continue;
                }
                const auto coefficient = own_scores[absolute - first_position];
#pragma unroll
                for (std::uint32_t item = 0U; item < 16U; ++item) {
                    const auto column = lane + item * 32U;
                    if (column < head_dim) {
                        sums[item] = __fadd_rn(
                            sums[item], __fmul_rn(
                                coefficient,
                                __bfloat162float(
                                    tile[tile_row * head_dim + column])));
                    }
                }
            }
        }
        __syncthreads();
    }
    if (active) {
        auto* destination = output +
            (static_cast<std::uint64_t>(query_row) * query_heads + head) *
                head_dim;
#pragma unroll
        for (std::uint32_t item = 0U; item < 16U; ++item) {
            const auto column = lane + item * 32U;
            if (column < head_dim) {
                destination[column] = bf16_round(sums[item]);
                if (!isfinite(sums[item])) atomicExch(error_flag, 4U);
            }
        }
    }
}

__global__ void gemma4_attention_kernel(
    float* output, float* scores, const float* queries,
    const __nv_bfloat16* cache, std::uint32_t position,
    std::uint32_t capacity_rows, std::uint32_t query_heads,
    std::uint32_t key_value_heads, std::uint32_t head_dim,
    unsigned int* error_flag) {
    const auto head = static_cast<std::uint32_t>(blockIdx.x);
    if (head >= query_heads) return;
    const auto visible_rows = min(position + 1U, capacity_rows);
    const auto first_position = position + 1U - visible_rows;
    const auto kv_head = head / (query_heads / key_value_heads);
    const auto* query = queries + static_cast<std::uint64_t>(head) * head_dim;
    const auto plane = static_cast<std::uint64_t>(capacity_rows) *
                       key_value_heads * head_dim;
    auto* head_scores = scores + static_cast<std::uint64_t>(head) *
                                  capacity_rows;
    for (std::uint32_t row = threadIdx.x; row < visible_rows;
         row += blockDim.x) {
        const auto absolute = first_position + row;
        const auto physical = absolute % capacity_rows;
        const auto* key = cache +
            (static_cast<std::uint64_t>(physical) * key_value_heads + kv_head) *
                head_dim;
        float score = 0.0F;
        for (std::uint32_t column = 0U; column < head_dim; ++column) {
            score = __fadd_rn(
                score,
                __fmul_rn(query[column], __bfloat162float(key[column])));
        }
        head_scores[row] = score;
        if (!isfinite(score)) atomicExch(error_flag, 2U);
    }
    __syncthreads();

    __shared__ float denominator;
    if (threadIdx.x == 0U) {
        float maximum = -INFINITY;
        for (std::uint32_t row = 0U; row < visible_rows; ++row) {
            maximum = fmaxf(maximum, head_scores[row]);
        }
        denominator = 0.0F;
        for (std::uint32_t row = 0U; row < visible_rows; ++row) {
            const float probability = expf(__fsub_rn(head_scores[row], maximum));
            head_scores[row] = probability;
            denominator = __fadd_rn(denominator, probability);
        }
        if (!isfinite(denominator) || denominator <= 0.0F) {
            atomicExch(error_flag, 3U);
        }
        for (std::uint32_t row = 0U; row < visible_rows; ++row) {
            head_scores[row] = __fdiv_rn(head_scores[row], denominator);
        }
    }
    __syncthreads();

    for (std::uint32_t column = threadIdx.x; column < head_dim;
         column += blockDim.x) {
        float sum = 0.0F;
        for (std::uint32_t row = 0U; row < visible_rows; ++row) {
            const auto absolute = first_position + row;
            const auto physical = absolute % capacity_rows;
            const auto value_offset =
                (static_cast<std::uint64_t>(physical) * key_value_heads +
                 kv_head) * head_dim + column;
            sum = __fadd_rn(
                sum,
                __fmul_rn(head_scores[row],
                          __bfloat162float(cache[plane + value_offset])));
        }
        output[static_cast<std::uint64_t>(head) * head_dim + column] =
            bf16_round(sum);
        if (!isfinite(sum)) atomicExch(error_flag, 4U);
    }
}

__global__ void gemma4_post_attention_kernel(
    float* hidden, float* normalized, const float* branch,
    const float* post_attention_norm, const float* pre_feedforward_norm,
    std::uint32_t columns, unsigned int* error_flag) {
    if (blockIdx.x != 0U) return;
    gemma4_norm_vector_block(
        normalized, branch, post_attention_norm, columns, 1.0e-6F,
        error_flag);
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        hidden[column] = bf16_round(hidden[column] + normalized[column]);
    }
    __syncthreads();
    gemma4_norm_vector_block(
        normalized, hidden, pre_feedforward_norm, columns, 1.0e-6F,
        error_flag);
}

__global__ void gemma4_post_attention_rows_kernel(
    float* hidden, float* normalized, const float* branch,
    const float* post_attention_norm, const float* pre_feedforward_norm,
    std::uint32_t rows, std::uint32_t columns, unsigned int* error_flag) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x);
    if (row >= rows) return;
    const auto offset = static_cast<std::uint64_t>(row) * columns;
    gemma4_norm_vector_block(normalized + offset, branch + offset,
                             post_attention_norm, columns, 1.0e-6F,
                             error_flag);
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        hidden[offset + column] = bf16_round(
            hidden[offset + column] + normalized[offset + column]);
    }
    __syncthreads();
    gemma4_norm_vector_block(normalized + offset, hidden + offset,
                             pre_feedforward_norm, columns, 1.0e-6F,
                             error_flag);
}

__global__ void gemma4_geglu_kernel(
    float* gate, const float* up, std::uint32_t columns,
    unsigned int* error_flag) {
    const auto column = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (column >= columns) return;
    const float gate_value = bf16_round(gate[column]);
    const float up_value = bf16_round(up[column]);
    if (!isfinite(gate_value) || !isfinite(up_value)) {
        atomicExch(error_flag, 1U);
        return;
    }
    constexpr float coefficient = 0.7978845608028654F;
    const float value = gate_value;
    const float activated = 0.5F * value *
        (1.0F + tanhf(coefficient *
                      (value + 0.044715F * value * value * value)));
    gate[column] = bf16_round(bf16_round(activated) * up_value);
}

__global__ void gemma4_post_feedforward_kernel(
    float* hidden, float* normalized, const float* branch,
    const float* post_feedforward_norm, std::uint32_t columns,
    float scalar, unsigned int* error_flag) {
    if (blockIdx.x != 0U) return;
    gemma4_norm_vector_block(
        normalized, branch, post_feedforward_norm, columns, 1.0e-6F,
        error_flag);
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        hidden[column] = bf16_round(
            bf16_round(hidden[column] + normalized[column]) * scalar);
    }
}

__global__ void gemma4_post_feedforward_rows_kernel(
    float* hidden, float* normalized, const float* branch,
    const float* post_feedforward_norm, std::uint32_t rows,
    std::uint32_t columns, float scalar, unsigned int* error_flag) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x);
    if (row >= rows) return;
    const auto offset = static_cast<std::uint64_t>(row) * columns;
    gemma4_norm_vector_block(normalized + offset, branch + offset,
                             post_feedforward_norm, columns, 1.0e-6F,
                             error_flag);
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        hidden[offset + column] = bf16_round(
            bf16_round(hidden[offset + column] + normalized[offset + column]) *
            scalar);
    }
}

__global__ void native_fp8_matmul_kernel(
    float* output, const float* input, const unsigned char* weights,
    const unsigned char* scales, std::uint64_t scale_columns,
    std::uint32_t batch, std::uint64_t columns, std::uint64_t rows,
    std::uint32_t groups, std::uint64_t rows_per_group) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t batch_row = blockIdx.y;
    if (output_row >= rows || batch_row >= batch) return;
    const std::uint64_t input_row = groups == 0U
                                        ? batch_row
                                        : static_cast<std::uint64_t>(batch_row) *
                                              groups +
                                              output_row / rows_per_group;
    const std::uint64_t input_base = input_row * columns;
    const std::uint64_t weight_base = output_row * columns;
    float sum = 0.0F;
    for (std::uint64_t column = threadIdx.x; column < columns; column += blockDim.x) {
        const float weight = fp8_e4m3_value(weights[weight_base + column]);
        const float scale = fp8_e8m0_scale(
            scales[(output_row / 128U) * scale_columns + column / 128U]);
        sum += input[input_base + column] * weight * scale;
    }
    sum = reduce_block(sum);
    if (threadIdx.x == 0) {
        const std::uint64_t output_index =
            static_cast<std::uint64_t>(batch_row) * rows + output_row;
        output[output_index] = sum;
    }
}

__global__ void native_fp8_f32_scale_matmul_kernel(
    float* output, const float* input, const unsigned char* weights,
    const float* scales, std::uint64_t scale_columns,
    std::uint32_t batch, std::uint64_t columns, std::uint64_t rows,
    std::uint32_t groups, std::uint64_t rows_per_group) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t batch_row = blockIdx.y;
    if (output_row >= rows || batch_row >= batch) return;
    const std::uint64_t input_row = groups == 0U
        ? batch_row
        : static_cast<std::uint64_t>(batch_row) * groups +
              output_row / rows_per_group;
    const std::uint64_t input_base = input_row * columns;
    const std::uint64_t weight_base = output_row * columns;
    float sum = 0.0F;
    for (std::uint64_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        const float weight = fp8_e4m3_value(weights[weight_base + column]);
        const float scale = scales[(output_row / 128U) * scale_columns +
                                   column / 128U];
        sum += input[input_base + column] * weight * scale;
    }
    sum = reduce_block(sum);
    if (threadIdx.x == 0U) {
        output[static_cast<std::uint64_t>(batch_row) * rows + output_row] = sum;
    }
}

constexpr std::uint32_t kDsv4Fp8TensorBlockM = 64U;
constexpr std::uint32_t kDsv4Fp8TensorBlockN = 128U;
constexpr std::uint32_t kDsv4Fp8TensorBlockK = 128U;
constexpr std::uint32_t kFp8F32TensorBlockN = 64U;

// SM86 page-projection path. Both operands remain byte FP8 in global memory;
// each tile widens them exactly to BF16 in shared memory and uses BF16 WMMA.
// Activation scales are powers of two and are applied during the exact widen;
// the weight block scale is applied after each K128 tensor dot, matching the
// block-scaled checkpoint arithmetic screened in experiments 0103/0104.
__global__ void dsv4_fp8_decode_bf16_tensor_kernel(
    float* output, const unsigned char* input,
    const unsigned char* input_scales, const unsigned char* weights,
    const unsigned char* weight_scales, std::uint32_t batch,
    std::uint32_t columns, std::uint32_t rows) {
    using namespace nvcuda;
    __shared__ __nv_bfloat16 shared_a[
        kDsv4Fp8TensorBlockM * kDsv4Fp8TensorBlockK];
    __shared__ __nv_bfloat16 shared_b[
        kDsv4Fp8TensorBlockK * kDsv4Fp8TensorBlockN];

    const std::uint32_t tile_m = blockIdx.y * kDsv4Fp8TensorBlockM;
    const std::uint32_t tile_n = blockIdx.x * kDsv4Fp8TensorBlockN;
    const std::uint32_t warp = threadIdx.x / warpSize;
    const std::uint32_t warp_m = warp & 3U;
    const std::uint32_t warp_n_group = warp >> 2U;
    constexpr std::uint32_t fragments_per_warp = 4U;
    float totals[fragments_per_warp][8]{};
    const std::uint32_t scale_columns = columns / kDsv4Fp8TensorBlockK;

    for (std::uint32_t tile_k = 0U; tile_k < columns;
         tile_k += kDsv4Fp8TensorBlockK) {
        for (std::uint32_t index = threadIdx.x;
             index < kDsv4Fp8TensorBlockM * kDsv4Fp8TensorBlockK;
             index += blockDim.x) {
            const std::uint32_t local_m = index / kDsv4Fp8TensorBlockK;
            const std::uint32_t local_k = index % kDsv4Fp8TensorBlockK;
            const std::uint32_t global_m = tile_m + local_m;
            float value = 0.0F;
            if (global_m < batch) {
                const auto encoded = input[
                    static_cast<std::uint64_t>(global_m) * columns + tile_k +
                    local_k];
                const auto scale = input_scales[
                    static_cast<std::uint64_t>(global_m) * scale_columns +
                    tile_k / kDsv4Fp8TensorBlockK];
                value = fp8_e4m3_value(encoded) * fp8_e8m0_scale(scale);
            }
            shared_a[index] = __float2bfloat16_rn(value);
        }
        for (std::uint32_t index = threadIdx.x;
             index < kDsv4Fp8TensorBlockK * kDsv4Fp8TensorBlockN;
             index += blockDim.x) {
            const std::uint32_t local_k = index / kDsv4Fp8TensorBlockN;
            const std::uint32_t local_n = index % kDsv4Fp8TensorBlockN;
            const std::uint32_t global_n = tile_n + local_n;
            const auto encoded = weights[
                static_cast<std::uint64_t>(global_n) * columns + tile_k +
                local_k];
            shared_b[index] = __float2bfloat16_rn(fp8_e4m3_value(encoded));
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                       wmma::row_major> a_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                       wmma::row_major> b_fragment;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float>
            accumulators[fragments_per_warp];
        for (std::uint32_t fragment = 0U; fragment < fragments_per_warp;
             ++fragment) {
            wmma::fill_fragment(accumulators[fragment], 0.0F);
        }
        for (std::uint32_t local_k = 0U;
             local_k < kDsv4Fp8TensorBlockK; local_k += 16U) {
            wmma::load_matrix_sync(
                a_fragment,
                shared_a + warp_m * 16U * kDsv4Fp8TensorBlockK + local_k,
                kDsv4Fp8TensorBlockK);
            for (std::uint32_t fragment = 0U;
                 fragment < fragments_per_warp; ++fragment) {
                const std::uint32_t fragment_n =
                    warp_n_group * fragments_per_warp + fragment;
                wmma::load_matrix_sync(
                    b_fragment,
                    shared_b + local_k * kDsv4Fp8TensorBlockN +
                        fragment_n * 16U,
                    kDsv4Fp8TensorBlockN);
                wmma::mma_sync(accumulators[fragment], a_fragment, b_fragment,
                               accumulators[fragment]);
            }
        }
        const float scale = fp8_e8m0_scale(weight_scales[
            (tile_n / kDsv4Fp8TensorBlockN) * scale_columns +
            tile_k / kDsv4Fp8TensorBlockK]);
        for (std::uint32_t fragment = 0U; fragment < fragments_per_warp;
             ++fragment) {
#pragma unroll
            for (std::uint32_t element = 0U;
                 element < accumulators[fragment].num_elements; ++element) {
                totals[fragment][element] +=
                    accumulators[fragment].x[element] * scale;
            }
        }
        __syncthreads();
    }

    for (std::uint32_t fragment = 0U; fragment < fragments_per_warp;
         ++fragment) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> result;
#pragma unroll
        for (std::uint32_t element = 0U;
             element < result.num_elements; ++element) {
            result.x[element] = totals[fragment][element];
        }
        const std::uint32_t fragment_n =
            warp_n_group * fragments_per_warp + fragment;
        float* destination = output +
            static_cast<std::uint64_t>(tile_m + warp_m * 16U) * rows +
            tile_n + fragment_n * 16U;
        wmma::store_matrix_sync(destination, result, rows,
                                wmma::mem_row_major);
    }
}

// QPN-derived continuous-scale page projection. Unlike DeepSeek's E8M0 route,
// neither GLM scale is a power of two, so widening a scaled activation to BF16
// would discard part of the checkpoint's declared arithmetic. Instead each
// K128 tile performs a tensor-core dot over the raw E4M3 values (all exactly
// representable in BF16), publishes that partial, and applies the activation
// and weight scales in F32. A 64x64 output tile leaves the raw A/B tiles, the
// published partial and the accumulated result within the 48 KiB SM86 shared
// memory budget without assuming a particular installed GPU.
__device__ __forceinline__ unsigned char fp8_fragment_prepacked_code(
    const unsigned char* codes, std::uint32_t row, std::uint32_t column,
    std::uint32_t columns) {
    const std::uint32_t pair = column / 32U;
    const std::uint32_t within16 = column & 15U;
    const std::uint32_t lane = (row & 7U) * 4U +
                               ((within16 & 7U) / 2U);
    const auto packed = reinterpret_cast<const uint4*>(codes)[
        (static_cast<std::size_t>(row / 16U) * (columns / 32U) + pair) * 32U +
        lane];
    const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    const bool upper_columns = within16 >= 8U;
    const std::uint32_t word = (column & 31U) / 16U * 2U +
                               (upper_columns ? 1U : 0U);
    const std::uint32_t i = (upper_columns ? 4U : 0U) +
                            ((row & 15U) >= 8U ? 2U : 0U) +
                            (within16 & 1U);
    return static_cast<unsigned char>((words[word] >> ((i & 3U) * 8U)) &
                                      0xFFU);
}

template <bool kFragmentPrepacked>
__global__ void fp8_f32_decode_bf16_tensor_kernel(
    float* output, const unsigned char* input, const float* input_scales,
    const unsigned char* weights, const float* weight_scales,
    std::uint32_t batch, std::uint32_t columns, std::uint32_t rows) {
    using namespace nvcuda;
    union SharedAOrPartial {
        __nv_bfloat16 a[kDsv4Fp8TensorBlockM * kDsv4Fp8TensorBlockK];
        float partial[kDsv4Fp8TensorBlockM * kFp8F32TensorBlockN];
    };
    __shared__ SharedAOrPartial shared_a_or_partial;
    __shared__ __nv_bfloat16 shared_b[
        kDsv4Fp8TensorBlockK * kFp8F32TensorBlockN];
    __shared__ float totals[
        kDsv4Fp8TensorBlockM * kFp8F32TensorBlockN];

    const std::uint32_t tile_m = blockIdx.y * kDsv4Fp8TensorBlockM;
    const std::uint32_t tile_n = blockIdx.x * kFp8F32TensorBlockN;
    const std::uint32_t warp = threadIdx.x / warpSize;
    const std::uint32_t warp_m = warp & 3U;
    const std::uint32_t warp_n_group = warp >> 2U;
    constexpr std::uint32_t fragments_per_warp = 2U;
    const std::uint32_t scale_columns = columns / kDsv4Fp8TensorBlockK;

    for (std::uint32_t index = threadIdx.x;
         index < kDsv4Fp8TensorBlockM * kFp8F32TensorBlockN;
         index += blockDim.x) {
        totals[index] = 0.0F;
    }
    __syncthreads();

    for (std::uint32_t tile_k = 0U; tile_k < columns;
         tile_k += kDsv4Fp8TensorBlockK) {
        for (std::uint32_t index = threadIdx.x;
             index < kDsv4Fp8TensorBlockM * kDsv4Fp8TensorBlockK;
             index += blockDim.x) {
            const std::uint32_t local_m = index / kDsv4Fp8TensorBlockK;
            const std::uint32_t local_k = index % kDsv4Fp8TensorBlockK;
            const std::uint32_t global_m = tile_m + local_m;
            const auto encoded = global_m < batch
                ? input[static_cast<std::uint64_t>(global_m) * columns +
                        tile_k + local_k]
                : 0U;
            shared_a_or_partial.a[index] =
                __float2bfloat16_rn(fp8_e4m3_value(encoded));
        }
        for (std::uint32_t index = threadIdx.x;
             index < kDsv4Fp8TensorBlockK * kFp8F32TensorBlockN;
             index += blockDim.x) {
            const std::uint32_t local_k = index / kFp8F32TensorBlockN;
            const std::uint32_t local_n = index % kFp8F32TensorBlockN;
            const std::uint32_t global_n = tile_n + local_n;
            const auto encoded = kFragmentPrepacked
                ? fp8_fragment_prepacked_code(weights, global_n,
                                              tile_k + local_k, columns)
                : weights[static_cast<std::uint64_t>(global_n) * columns +
                          tile_k + local_k];
            shared_b[index] =
                __float2bfloat16_rn(fp8_e4m3_value(encoded));
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                       wmma::row_major> a_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                       wmma::row_major> b_fragment;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float>
            accumulators[fragments_per_warp];
        for (std::uint32_t fragment = 0U; fragment < fragments_per_warp;
             ++fragment) {
            wmma::fill_fragment(accumulators[fragment], 0.0F);
        }
        for (std::uint32_t local_k = 0U;
             local_k < kDsv4Fp8TensorBlockK; local_k += 16U) {
            wmma::load_matrix_sync(
                a_fragment,
                shared_a_or_partial.a +
                    warp_m * 16U * kDsv4Fp8TensorBlockK + local_k,
                kDsv4Fp8TensorBlockK);
            for (std::uint32_t fragment = 0U;
                 fragment < fragments_per_warp; ++fragment) {
                const std::uint32_t fragment_n =
                    warp_n_group * fragments_per_warp + fragment;
                wmma::load_matrix_sync(
                    b_fragment,
                    shared_b + local_k * kFp8F32TensorBlockN +
                        fragment_n * 16U,
                    kFp8F32TensorBlockN);
                wmma::mma_sync(accumulators[fragment], a_fragment, b_fragment,
                               accumulators[fragment]);
            }
        }
        __syncthreads();
        for (std::uint32_t fragment = 0U; fragment < fragments_per_warp;
             ++fragment) {
            const std::uint32_t fragment_n =
                warp_n_group * fragments_per_warp + fragment;
            float* destination = shared_a_or_partial.partial +
                warp_m * 16U * kFp8F32TensorBlockN + fragment_n * 16U;
            wmma::store_matrix_sync(destination, accumulators[fragment],
                                    kFp8F32TensorBlockN,
                                    wmma::mem_row_major);
        }
        __syncthreads();

        for (std::uint32_t index = threadIdx.x;
             index < kDsv4Fp8TensorBlockM * kFp8F32TensorBlockN;
             index += blockDim.x) {
            const std::uint32_t local_m = index / kFp8F32TensorBlockN;
            const std::uint32_t local_n = index % kFp8F32TensorBlockN;
            const std::uint32_t global_m = tile_m + local_m;
            const std::uint32_t global_n = tile_n + local_n;
            if (global_m < batch) {
                const float activation_scale = input_scales[
                    static_cast<std::uint64_t>(global_m) * scale_columns +
                    tile_k / kDsv4Fp8TensorBlockK];
                const float weight_scale = weight_scales[
                    static_cast<std::uint64_t>(global_n / 128U) *
                        scale_columns + tile_k / kDsv4Fp8TensorBlockK];
                totals[index] += shared_a_or_partial.partial[index] *
                                 activation_scale * weight_scale;
            }
        }
        __syncthreads();
    }

    for (std::uint32_t index = threadIdx.x;
         index < kDsv4Fp8TensorBlockM * kFp8F32TensorBlockN;
         index += blockDim.x) {
        const std::uint32_t local_m = index / kFp8F32TensorBlockN;
        const std::uint32_t local_n = index % kFp8F32TensorBlockN;
        const std::uint32_t global_m = tile_m + local_m;
        if (global_m < batch) {
            output[static_cast<std::uint64_t>(global_m) * rows + tile_n +
                   local_n] = totals[index];
        }
    }
}

__global__ void native_fp4_matmul_kernel(
    float* output, const float* input, const unsigned char* weights,
    const unsigned char* scales, std::uint64_t packed_columns,
    std::uint64_t scale_columns, std::uint32_t batch,
    std::uint64_t columns, std::uint64_t rows, std::uint32_t groups,
    std::uint64_t rows_per_group) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t batch_row = blockIdx.y;
    if (output_row >= rows || batch_row >= batch) return;
    const std::uint64_t input_row = groups == 0U
                                        ? batch_row
                                        : static_cast<std::uint64_t>(batch_row) *
                                              groups +
                                              output_row / rows_per_group;
    const std::uint64_t input_base = input_row * columns;
    const std::uint64_t weight_base = output_row * packed_columns;
    const std::uint64_t scale_base = output_row * scale_columns;
    float sum = 0.0F;
    for (std::uint64_t column = threadIdx.x; column < columns; column += blockDim.x) {
        const unsigned char packed = weights[weight_base + column / 2U];
        const unsigned int encoded = column % 2U == 0U ? packed & 0x0FU : packed >> 4U;
        const float scale = fp8_e8m0_scale(scales[scale_base + column / 32U]);
        sum += input[input_base + column] * fp4_e2m1_value(encoded) * scale;
    }
    sum = reduce_block(sum);
    if (threadIdx.x == 0) {
        const std::uint64_t output_index =
            static_cast<std::uint64_t>(batch_row) * rows + output_row;
        output[output_index] = sum;
    }
}

// NVFP4 ("nvfp4-pack-quantized"): E2M1 nibble pairs with FP8 E4M3 group scales
// and one FP32 per-tensor global scale. Unlike the DeepSeek FP4 path this does
// not quantize the activation; the declared contract is W4A16 with FP32
// accumulation, matching the reference implementation's dequantized BF16 GEMM.
__global__ void nvfp4_group16_matmul_kernel(
    float* output, const float* input, const unsigned char* weights,
    const unsigned char* scales, float global_scale,
    std::uint64_t packed_columns, std::uint64_t scale_columns,
    std::uint32_t group_size, std::uint32_t batch, std::uint64_t columns,
    std::uint64_t rows, std::uint32_t groups, std::uint64_t rows_per_group) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t batch_row = blockIdx.y;
    if (output_row >= rows || batch_row >= batch) return;
    const std::uint64_t input_row = groups == 0U
                                        ? batch_row
                                        : static_cast<std::uint64_t>(batch_row) *
                                              groups +
                                              output_row / rows_per_group;
    const std::uint64_t input_base = input_row * columns;
    const std::uint64_t weight_base = output_row * packed_columns;
    const std::uint64_t scale_base = output_row * scale_columns;
    float sum = 0.0F;
    for (std::uint64_t column = threadIdx.x; column < columns; column += blockDim.x) {
        const unsigned char packed = weights[weight_base + column / 2U];
        const unsigned int encoded = column % 2U == 0U ? packed & 0x0FU : packed >> 4U;
        const float scale =
            fp8_e4m3_value(scales[scale_base + column / group_size]) / global_scale;
        sum += input[input_base + column] * fp4_e2m1_value(encoded) * scale;
    }
    sum = reduce_block(sum);
    if (threadIdx.x == 0) {
        const std::uint64_t output_index =
            static_cast<std::uint64_t>(batch_row) * rows + output_row;
        output[output_index] = sum;
    }
}

constexpr std::uint32_t kMaxDeepSeekRoutedExperts = 6U;
// Laguna routes top-10 of 256 and all ten can hash to one device, so a batch
// must hold ten routed experts plus an optional shared one.
constexpr std::uint32_t kMaxMoeExperts = 11U;
constexpr std::uint32_t kMaxRoutedMoeExperts = 10U;

struct DeepSeekFp4Batch {
    const unsigned char* w1_weights[kMaxDeepSeekRoutedExperts]{};
    const unsigned char* w1_scales[kMaxDeepSeekRoutedExperts]{};
    const unsigned char* w3_weights[kMaxDeepSeekRoutedExperts]{};
    const unsigned char* w3_scales[kMaxDeepSeekRoutedExperts]{};
    const unsigned char* w2_weights[kMaxDeepSeekRoutedExperts]{};
    const unsigned char* w2_scales[kMaxDeepSeekRoutedExperts]{};
    float coefficients[kMaxDeepSeekRoutedExperts]{};
    std::uint32_t count{};
};

struct PackedInt4MoeBatch {
    const std::uint32_t* gate_weights[kMaxMoeExperts]{};
    const __nv_bfloat16* gate_scales[kMaxMoeExperts]{};
    const std::uint32_t* up_weights[kMaxMoeExperts]{};
    const __nv_bfloat16* up_scales[kMaxMoeExperts]{};
    const std::uint32_t* down_weights[kMaxMoeExperts]{};
    const __nv_bfloat16* down_scales[kMaxMoeExperts]{};
    float coefficients[kMaxMoeExperts]{};
    std::uint32_t count{};
    std::uint32_t rows{};
};

// compressed-tensors "nvfp4-pack-quantized": E2M1 nibble pairs, FP8 E4M3 group
// scales, one FP32 per-tensor divisor. The divisor is per weight tensor, so it
// travels per expert and per projection rather than per batch.
struct Nvfp4MoeBatch {
    const unsigned char* gate_weights[kMaxMoeExperts]{};
    const unsigned char* gate_scales[kMaxMoeExperts]{};
    const unsigned char* up_weights[kMaxMoeExperts]{};
    const unsigned char* up_scales[kMaxMoeExperts]{};
    const unsigned char* down_weights[kMaxMoeExperts]{};
    const unsigned char* down_scales[kMaxMoeExperts]{};
    float gate_global_scales[kMaxMoeExperts]{};
    float up_global_scales[kMaxMoeExperts]{};
    float down_global_scales[kMaxMoeExperts]{};
    std::uint32_t count{};
    std::uint32_t rows{};
};

// compressed-tensors MXFP4: E2M1 nibble pairs with power-of-two E8M0 scales
// per group of 32 columns. Unlike NVFP4 there is no per-tensor divisor.
struct Mxfp4MoeBatch {
    const unsigned char* gate_weights[kMaxMoeExperts]{};
    const unsigned char* gate_scales[kMaxMoeExperts]{};
    const unsigned char* up_weights[kMaxMoeExperts]{};
    const unsigned char* up_scales[kMaxMoeExperts]{};
    const unsigned char* down_weights[kMaxMoeExperts]{};
    const unsigned char* down_scales[kMaxMoeExperts]{};
    std::uint32_t count{};
    std::uint32_t rows{};
};

// Standard compressed-tensors FP8: E4M3 payloads with ordinary F32 inverse
// scales per 128x128 weight block. Kept separate from the E8M0-scaled FP8
// batches so their scale bytes can never be reinterpreted silently.
struct Fp8F32MoeBatch {
    const unsigned char* gate_weights[kMaxMoeExperts]{};
    const float* gate_scales[kMaxMoeExperts]{};
    const unsigned char* up_weights[kMaxMoeExperts]{};
    const float* up_scales[kMaxMoeExperts]{};
    const unsigned char* down_weights[kMaxMoeExperts]{};
    const float* down_scales[kMaxMoeExperts]{};
    float coefficients[kMaxMoeExperts]{};
    std::uint32_t count{};
    std::uint32_t rows{};
};

// Laguna carries the routed experts of layers 40-47 as plain BF16.
struct PlainBf16MoeBatch {
    const __nv_bfloat16* gate_weights[kMaxMoeExperts]{};
    const __nv_bfloat16* up_weights[kMaxMoeExperts]{};
    const __nv_bfloat16* down_weights[kMaxMoeExperts]{};
    std::uint32_t count{};
    std::uint32_t rows{};
};

__device__ float fp8_e8m0_scale_bits(unsigned char encoded) {
    // E8M0 is exactly a float exponent field, except that 0xff is NaN and
    // exponent zero denotes 2^-127 rather than float zero.
    if (encoded == 0xFFU) return __uint_as_float(0x7FC0'0000U);
    if (encoded == 0U) return __uint_as_float(0x0040'0000U);
    return __uint_as_float(static_cast<unsigned int>(encoded) << 23U);
}

__device__ float bf16_round(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

struct LightningDeviceSegment {
    const unsigned char* keys{};
    std::uint32_t begin{};
    std::uint32_t rows{};
};

__device__ float lightning_fp4_magnitude(std::uint32_t encoded) {
    constexpr float values[8]{0.0F, 0.5F, 1.0F, 1.5F,
                              2.0F, 3.0F, 4.0F, 6.0F};
    return values[encoded & 7U];
}

__global__ void lightning_query_fp4_kernel(
    unsigned char* packed, const float* queries,
    std::uint32_t heads, std::uint32_t head_dim) {
    const auto head = blockIdx.x;
    if (head >= heads) return;
    extern __shared__ float values[];
    for (std::uint32_t column = threadIdx.x; column < head_dim;
         column += blockDim.x) {
        values[column] = queries[static_cast<std::uint64_t>(head) * head_dim +
                                 column];
    }
    __syncthreads();
    for (std::uint32_t width = 1U; width < head_dim; width *= 2U) {
        for (std::uint32_t pair = threadIdx.x; pair < head_dim / 2U;
             pair += blockDim.x) {
            const auto begin = pair / width * width * 2U;
            const auto offset = pair % width;
            const float first = values[begin + offset];
            const float second = values[begin + width + offset];
            values[begin + offset] = first + second;
            values[begin + width + offset] = first - second;
        }
        __syncthreads();
    }
    const float normalization = rsqrtf(static_cast<float>(head_dim));
    for (std::uint32_t column = threadIdx.x; column < head_dim;
         column += blockDim.x) {
        values[column] = bf16_round(values[column] * normalization);
    }
    __syncthreads();

    const auto packed_columns = head_dim / 2U;
    const auto groups = head_dim / 32U;
    auto* output = packed + static_cast<std::uint64_t>(head) *
                                (packed_columns + groups);
    for (std::uint32_t group = threadIdx.x; group < groups;
         group += blockDim.x) {
        float maximum = 0.0F;
        for (std::uint32_t column = 0U; column < 32U; ++column) {
            maximum = fmaxf(maximum, fabsf(values[group * 32U + column]));
        }
        const float bounded = fmaxf(maximum, ldexpf(6.0F, -126));
        const float scale = exp2f(ceilf(log2f(bounded / 6.0F)));
        output[packed_columns + group] = static_cast<unsigned char>(
            ilogbf(scale) + 127);
        for (std::uint32_t column = 0U; column < 32U; column += 2U) {
            unsigned int encoded[2]{};
            for (std::uint32_t item = 0U; item < 2U; ++item) {
                const float value = values[group * 32U + column + item];
                const float magnitude = fminf(fabsf(value / scale), 6.0F);
                std::uint32_t nearest = 0U;
                float distance = fabsf(magnitude - lightning_fp4_magnitude(0U));
                for (std::uint32_t candidate = 1U; candidate < 8U;
                     ++candidate) {
                    const float candidate_distance = fabsf(
                        magnitude - lightning_fp4_magnitude(candidate));
                    if (candidate_distance < distance ||
                        (candidate_distance == distance &&
                         (candidate & 1U) == 0U && (nearest & 1U) != 0U)) {
                        nearest = candidate;
                        distance = candidate_distance;
                    }
                }
                encoded[item] = nearest | (signbit(value) ? 8U : 0U);
            }
            output[group * 16U + column / 2U] =
                static_cast<unsigned char>(encoded[0] | (encoded[1] << 4U));
        }
    }
}

__global__ void lightning_score_kernel(
    float* scores, const unsigned char* packed_queries,
    const float* weights, const LightningDeviceSegment* segments,
    std::uint32_t segment_count, std::uint32_t candidates,
    std::uint32_t heads, std::uint32_t head_dim,
    unsigned int* error_flag) {
    const auto row = blockIdx.x;
    if (row >= candidates || threadIdx.x >= heads) return;
    std::uint32_t low = 0U;
    std::uint32_t high = segment_count;
    while (low < high) {
        const auto middle = low + (high - low) / 2U;
        if (segments[middle].begin + segments[middle].rows <= row) {
            low = middle + 1U;
        } else {
            high = middle;
        }
    }
    if (low >= segment_count || row < segments[low].begin) {
        if (threadIdx.x == 0U) atomicExch(error_flag, 1U);
        return;
    }
    const auto packed_columns = head_dim / 2U;
    const auto scale_columns = head_dim / 32U;
    const auto row_bytes = packed_columns + scale_columns;
    const auto* key = segments[low].keys +
        static_cast<std::uint64_t>(row - segments[low].begin) * row_bytes;
    const auto* query = packed_queries +
        static_cast<std::uint64_t>(threadIdx.x) * row_bytes;
    float dot = 0.0F;
    for (std::uint32_t column = 0U; column < head_dim; ++column) {
        const auto key_values = key[column / 2U];
        const auto query_values = query[column / 2U];
        const auto key_encoded = column % 2U == 0U
            ? key_values & 0x0FU : key_values >> 4U;
        const auto query_encoded = column % 2U == 0U
            ? query_values & 0x0FU : query_values >> 4U;
        const auto key_scale = key[packed_columns + column / 32U];
        const auto query_scale = query[packed_columns + column / 32U];
        if (key_scale == 0xFFU || query_scale == 0xFFU) {
            atomicExch(error_flag, 1U);
            dot = nanf("");
            break;
        }
        const float key_value = bf16_round(
            fp4_e2m1_value(key_encoded) * fp8_e8m0_scale_bits(key_scale));
        const float query_value = bf16_round(
            fp4_e2m1_value(query_encoded) *
            fp8_e8m0_scale_bits(query_scale));
        dot = fmaf(query_value, key_value, dot);
    }
    __shared__ float head_scores[64];
    head_scores[threadIdx.x] = bf16_round(dot);
    __syncthreads();
    if (threadIdx.x == 0U) {
        float score = 0.0F;
        for (std::uint32_t head = 0U; head < heads; ++head) {
            score += bf16_round(weights[head] * fmaxf(0.0F, head_scores[head]));
        }
        scores[row] = bf16_round(score);
        if (!isfinite(scores[row])) atomicExch(error_flag, 1U);
    }
}

// ---------------------------------------------------------------------------
// Physical-format (E4M3, one f32 scale per row) learned-index scoring and an
// exact parallel top-k.
//
// The FP4 path above resolves top-k with a single-thread insertion merge over
// every candidate. That is affordable at block-KV scale but not here: the
// physical cache holds `context / 4` index rows per layer, which is 262,144
// rows at the declared 1,048,576-token context, and 43 layers of a serial
// 262,144-element merge dominates a decode step on its own. Selection below is
// a radix select over a composite key, so every stage is parallel.
// ---------------------------------------------------------------------------

struct PhysicalIndexDeviceSegment {
    const unsigned char* payload{};
    std::uint32_t begin{};
    std::uint32_t rows{};
    std::uint32_t block_rows{};
};

// Order-preserving map from float to unsigned so that a larger unsigned always
// means a larger float. Finite inputs only; the score kernel flags anything
// else before selection runs.
__device__ unsigned int physical_index_sortable(float value) {
    const unsigned int bits = __float_as_uint(value);
    return (bits & 0x80000000U) != 0U ? ~bits : (bits | 0x80000000U);
}

// Composite selection key: score in the high 32 bits, the position's
// complement in the low 32. Descending order on this single key therefore
// reproduces dsv4_index_topk_f32's "higher score first, lower position wins
// ties" without a separate tie-break stage. Keys are unique because positions
// are, so the radix select never has to split a bucket it cannot resolve.
//
// Only the top 16 bits of the sortable score carry information: every score is
// round_bf16'd, so the low 16 mantissa bits of the fp32 are zero (and after
// complementing a negative value, uniformly one). The score kernel enforces
// that invariant rather than assuming it, which is what lets the select run in
// three 16-bit passes instead of four.
__device__ unsigned long long physical_index_key(float score,
                                                 std::uint32_t position) {
    const unsigned long long score_bits =
        static_cast<unsigned long long>(physical_index_sortable(score) >> 16U);
    return (score_bits << 32U) |
           static_cast<unsigned long long>(~position);
}

// Shared footprint is sized for the DSV4 contract: 64 index heads of 128
// dimensions. `rows_per_block` candidates share one block so that a block is
// 256 threads rather than 64 -- at 64 the hardware's 16-blocks-per-SM cap, not
// the register or shared budget, holds occupancy to two thirds.
constexpr std::uint32_t kPhysicalIndexBlockThreads = 256U;
constexpr std::uint32_t kPhysicalIndexMaxHeadDim = 1'024U;
// Candidate rows scored per thread. One row per thread leaves each thread with
// a single dependent __fadd_rn chain fed by one load per iteration, which at
// full occupancy still issues at about 0.28 instructions per cycle: the loop
// stalls on latency rather than saturating any resource. Scoring several rows
// against the same query element gives the scheduler independent chains and
// amortizes the query load, which measurement attributes at about 32% of the
// kernel. Rows are added to the block, not threads, so occupancy is unchanged.
constexpr std::uint32_t kPhysicalIndexRowsPerThread = 4U;

template <std::uint32_t kRowsPerThread>
__global__ void dsv4_physical_index_score_kernel(
    float* scores, unsigned long long* keys, const float* queries,
    const float* weights, const PhysicalIndexDeviceSegment* segments,
    std::uint32_t segment_count, std::uint32_t candidates,
    std::uint32_t heads, std::uint32_t head_dim,
    std::uint32_t rows_per_block, unsigned int* error_flag) {
    // The E4M3 decode is a branch plus ldexpf. Executed once per element it
    // would run head_dim times per thread; a byte has only 256 possible
    // values, so the whole decode collapses to one table built per block and
    // read as a shared load.
    //
    // Sized dynamically from the real shapes. A static array big enough for
    // the widest supported head_dim would reserve 32 KiB per block and hold an
    // SM to two blocks; at the DSV4 contract the same layout needs about 4 KiB
    // and reaches full occupancy.
    extern __shared__ float physical_index_shared[];
    auto* e4m3_table = physical_index_shared;
    auto* key_values = e4m3_table + 256U;
    auto* head_scores = key_values +
        static_cast<std::uint64_t>(rows_per_block) * head_dim;
    auto* row_live = reinterpret_cast<unsigned int*>(
        head_scores + static_cast<std::uint64_t>(rows_per_block) * heads);

    const auto lane = threadIdx.x % heads;
    const auto slot = threadIdx.x / heads;
    const auto first_row = blockIdx.x * rows_per_block;

    for (std::uint32_t entry = threadIdx.x; entry < 256U;
         entry += blockDim.x) {
        e4m3_table[entry] =
            fp8_e4m3_value(static_cast<unsigned char>(entry));
    }
    for (std::uint32_t index = threadIdx.x; index < rows_per_block;
         index += blockDim.x) {
        row_live[index] = 1U;
    }
    __syncthreads();

    // Dequantize each of the block's rows once, cooperatively, into shared
    // floats. Every head then reads a plain float instead of chasing a byte
    // through two dependent shared lookups on every iteration.
    for (std::uint32_t index = slot; index < rows_per_block;
         index += blockDim.x / heads) {
        const auto row = first_row + index;
        if (row >= candidates) {
            if (lane == 0U) row_live[index] = 0U;
            continue;
        }
        std::uint32_t low = 0U;
        std::uint32_t high = segment_count;
        while (low < high) {
            const auto middle = low + (high - low) / 2U;
            if (segments[middle].begin + segments[middle].rows <= row) {
                low = middle + 1U;
            } else {
                high = middle;
            }
        }
        if (low >= segment_count || row < segments[low].begin) {
            if (lane == 0U) {
                row_live[index] = 0U;
                atomicExch(error_flag, 1U);
            }
            continue;
        }
        const auto& segment = segments[low];
        const auto local = row - segment.begin;
        const auto* data = segment.payload +
            static_cast<std::uint64_t>(local) * head_dim;
        // Assembled byte by byte: `byte_offset` places a page anywhere inside
        // its block buffer, so the scale region carries no alignment
        // guarantee. The order matches the little-endian host memcpy in
        // dsv4_physical_encode_kv_row.
        const auto* scale_bytes = segment.payload +
            static_cast<std::uint64_t>(segment.block_rows) * head_dim +
            static_cast<std::uint64_t>(local) * sizeof(float);
        const unsigned int scale_raw =
            static_cast<unsigned int>(scale_bytes[0]) |
            (static_cast<unsigned int>(scale_bytes[1]) << 8U) |
            (static_cast<unsigned int>(scale_bytes[2]) << 16U) |
            (static_cast<unsigned int>(scale_bytes[3]) << 24U);
        const float scale = __uint_as_float(scale_raw);
        if (!isfinite(scale) || scale <= 0.0F) {
            if (lane == 0U) {
                row_live[index] = 0U;
                atomicExch(error_flag, 1U);
            }
            continue;
        }
        auto* destination = key_values +
            static_cast<std::uint64_t>(index) * head_dim;
        for (std::uint32_t column = lane; column < head_dim; column += heads) {
            // Exactly the value dsv4_physical_decode_kv_row materializes.
            destination[column] = __fmul_rn(e4m3_table[data[column]], scale);
        }
    }
    __syncthreads();

    // Queries arrive column-major (column * heads + head) so the heads of one
    // row read consecutive floats. Head-major queries would stride by head_dim
    // and cost a separate transaction per thread.
    //
    // Each thread owns `kRowsPerThread` consecutive local rows. The query
    // element for this column is loaded once and reused across them, and each
    // row keeps its own accumulator. dsv4_index_scores_f32 accumulates
    // query * key with a separate multiply and add in ascending column order;
    // every accumulator here still walks columns in that same ascending order
    // under explicit round-to-nearest intrinsics, so no fma contraction or
    // reassociation occurs and each row's result is bit identical to the
    // reference. Only which thread computes a row has changed.
    const auto* query = queries + lane;
    float dot[kRowsPerThread];
    #pragma unroll
    for (std::uint32_t index = 0U; index < kRowsPerThread; ++index) {
        dot[index] = 0.0F;
    }
    const auto first_local = slot * kRowsPerThread;
    for (std::uint32_t column = 0U; column < head_dim; ++column) {
        const float query_value =
            query[static_cast<std::uint64_t>(column) * heads];
        #pragma unroll
        for (std::uint32_t index = 0U; index < kRowsPerThread; ++index) {
            const auto local = first_local + index;
            dot[index] = __fadd_rn(
                dot[index],
                __fmul_rn(query_value,
                          key_values[static_cast<std::uint64_t>(local) *
                                         head_dim + column]));
        }
    }
    #pragma unroll
    for (std::uint32_t index = 0U; index < kRowsPerThread; ++index) {
        const auto local = first_local + index;
        if (local >= rows_per_block) continue;
        const auto candidate_row = first_row + local;
        const bool live = candidate_row < candidates &&
                          row_live[local] != 0U;
        head_scores[static_cast<std::uint64_t>(local) * heads + lane] =
            live ? bf16_round(dot[index]) : 0.0F;
    }
    __syncthreads();

    // One thread per local row performs the weighted reduction, in the same
    // ascending head order as before.
    if (threadIdx.x >= rows_per_block) return;
    const auto local = threadIdx.x;
    const auto row = first_row + local;
    if (row >= candidates || row_live[local] == 0U) return;
    const auto* row_scores = head_scores +
        static_cast<std::uint64_t>(local) * heads;
    float score = 0.0F;
    for (std::uint32_t head = 0U; head < heads; ++head) {
        score = __fadd_rn(
            score,
            bf16_round(__fmul_rn(weights[head],
                                 fmaxf(0.0F, row_scores[head]))));
    }
    score = bf16_round(score);
    // The composite key drops the low 16 bits of the score. That is exact only
    // while the score really is bf16-valued, so verify it here instead of
    // trusting bf16_round's contract from a distance.
    if (!isfinite(score) || (__float_as_uint(score) & 0xFFFFU) != 0U) {
        atomicExch(error_flag, 1U);
        return;
    }
    scores[row] = score;
    keys[row] = physical_index_key(score, row);
}

constexpr std::uint32_t kPhysicalIndexRadixBins = 65'536U;

// Every pass reads its element count from device memory and is launched over
// the worst-case grid, because the surviving set shrinks between passes and
// reading its size on the host would cost a stream synchronize per pass --
// 129 of them per token across 43 layers. Threads past the live count exit
// immediately instead.
__global__ void dsv4_physical_index_histogram_kernel(
    const unsigned long long* keys, const std::uint32_t* active,
    const std::uint32_t* active_count, std::uint32_t shift,
    std::uint32_t* histogram) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= *active_count) return;
    const auto position = active == nullptr ? index : active[index];
    const auto digit = static_cast<std::uint32_t>(
        (keys[position] >> shift) & 0xFFFFULL);
    atomicAdd(&histogram[digit], 1U);
}

// Walks the histogram from the highest digit down, accumulating until the
// running count reaches the number of selections still owed. Everything above
// the bin it stops on is an outright winner; the bin itself is the only one
// that has to be split further.
constexpr std::uint32_t kPhysicalIndexPivotThreads = 1'024U;
constexpr std::uint32_t kPhysicalIndexPivotGroup =
    kPhysicalIndexRadixBins / kPhysicalIndexPivotThreads;

// Locates the bin holding the k-th largest key. A single thread walking all
// 65,536 bins costs about 0.48 ms per pass here, which at three passes and 43
// layers is 62 ms per token of pure serial scan -- the same defect the FP4
// path's <<<1,1>>> merge has, moved rather than fixed. This reduces per group
// in parallel, suffix-scans 1,024 group totals, and leaves one thread to walk
// only the 64 bins of the group that actually contains the pivot.
__global__ void dsv4_physical_index_pivot_kernel(
    const std::uint32_t* histogram, const std::uint32_t* remaining,
    std::uint32_t* pivot_bin, std::uint32_t* above_count) {
    __shared__ std::uint32_t suffix[kPhysicalIndexPivotThreads];
    __shared__ std::uint32_t pivot_bins[kPhysicalIndexPivotGroup];
    __shared__ std::uint32_t pivot_group;
    const auto thread = threadIdx.x;
    const auto owed = *remaining;
    // Bin counts are integers, so the group sum is order-independent and only
    // the access pattern matters. Reading the group one uint32 at a time makes
    // a warp touch 32 separate 256 B regions, and each 4 B request still pulls
    // a whole 32 B sector: 8x read amplification, turning 256 KiB of histogram
    // into about 2 MiB of traffic. A group is 64 contiguous uint32 and the
    // allocation is 256 B aligned, so it can be read as 16 uint4 instead,
    // which cuts the amplification to 2x for the same arithmetic.
    std::uint32_t total = 0U;
    const auto* group_words = reinterpret_cast<const uint4*>(
        histogram + thread * kPhysicalIndexPivotGroup);
    #pragma unroll
    for (std::uint32_t index = 0U;
         index < kPhysicalIndexPivotGroup / 4U; ++index) {
        const uint4 quad = group_words[index];
        total += quad.x;
        total += quad.y;
        total += quad.z;
        total += quad.w;
    }
    suffix[thread] = total;
    if (thread == 0U) pivot_group = 0U;
    __syncthreads();
    // Inclusive suffix sum: suffix[t] ends as the population of every bin at
    // or above group t, which is the count "from the top" the walk needs.
    for (std::uint32_t stride = 1U; stride < kPhysicalIndexPivotThreads;
         stride *= 2U) {
        const std::uint32_t addend =
            thread + stride < kPhysicalIndexPivotThreads
                ? suffix[thread + stride] : 0U;
        __syncthreads();
        suffix[thread] += addend;
        __syncthreads();
    }
    // Exactly one group satisfies both halves: it reaches the quota while the
    // group above it does not.
    const bool reaches = suffix[thread] >= owed;
    const bool above_short = thread + 1U == kPhysicalIndexPivotThreads ||
                             suffix[thread + 1U] < owed;
    if (reaches && above_short) pivot_group = thread;
    __syncthreads();

    // The final walk is inherently sequential -- it stops at the first bin that
    // reaches the quota -- but it does not have to chase global memory. Staging
    // the pivot group's 64 bins coalesced turns up to 64 dependent global loads
    // on one thread into 64 shared reads.
    const auto group = pivot_group;
    const auto first = group * kPhysicalIndexPivotGroup;
    if (thread < kPhysicalIndexPivotGroup) {
        pivot_bins[thread] = histogram[first + thread];
    }
    __syncthreads();
    if (thread != 0U) return;
    std::uint32_t accumulated = group + 1U == kPhysicalIndexPivotThreads
        ? 0U : suffix[group + 1U];
    for (std::uint32_t index = kPhysicalIndexPivotGroup; index-- > 0U;) {
        const auto count = pivot_bins[index];
        if (accumulated + count >= owed) {
            *pivot_bin = first + index;
            *above_count = accumulated;
            return;
        }
        accumulated += count;
    }
    *pivot_bin = first;
    *above_count = accumulated;
}

__global__ void dsv4_physical_index_partition_kernel(
    const unsigned long long* keys, const std::uint32_t* active,
    const std::uint32_t* active_count, std::uint32_t shift,
    const std::uint32_t* pivot, std::uint32_t* winners,
    std::uint32_t* winner_count, std::uint32_t* next_active,
    std::uint32_t* next_count) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= *active_count) return;
    const auto position = active == nullptr ? index : active[index];
    const auto digit = static_cast<std::uint32_t>(
        (keys[position] >> shift) & 0xFFFFULL);
    if (digit > *pivot) {
        winners[atomicAdd(winner_count, 1U)] = position;
    } else if (digit == *pivot) {
        next_active[atomicAdd(next_count, 1U)] = position;
    }
}

// Retires one pass: the winners above the pivot are already banked, so the
// outstanding count drops by that many and the tied bucket becomes the next
// pass's input.
__global__ void dsv4_physical_index_advance_kernel(
    std::uint32_t* remaining, const std::uint32_t* above_count,
    std::uint32_t* active_count, std::uint32_t* next_count) {
    if (threadIdx.x != 0U || blockIdx.x != 0U) return;
    *remaining -= *above_count;
    *active_count = *next_count;
    *next_count = 0U;
}

// After the last pass the survivors agree on all 48 key bits. Keys are unique,
// so that set holds exactly the outstanding selections.
__global__ void dsv4_physical_index_finalize_kernel(
    const std::uint32_t* active, const std::uint32_t* active_count,
    const std::uint32_t* remaining, std::uint32_t* winners,
    std::uint32_t* winner_count, unsigned int* error_flag) {
    if (threadIdx.x != 0U || blockIdx.x != 0U) return;
    if (*active_count < *remaining) {
        atomicExch(error_flag, 1U);
        return;
    }
    for (std::uint32_t index = 0U; index < *remaining; ++index) {
        winners[(*winner_count)++] = active[index];
    }
}

// Bitonic sort of the selected positions by descending composite key. The
// radix select emits winners in bucket order, not in final order, so this is
// what makes the output match dsv4_index_topk_f32 element for element.
__global__ void dsv4_physical_index_sort_kernel(
    std::uint32_t* selected, const unsigned long long* keys,
    std::uint32_t count, std::uint32_t padded) {
    extern __shared__ unsigned long long sort_keys[];
    auto* sort_values = reinterpret_cast<std::uint32_t*>(sort_keys + padded);
    for (std::uint32_t index = threadIdx.x; index < padded;
         index += blockDim.x) {
        const bool live = index < count;
        sort_values[index] = live ? selected[index] : 0xFFFFFFFFU;
        sort_keys[index] = live ? keys[selected[index]] : 0ULL;
    }
    __syncthreads();
    for (std::uint32_t size = 2U; size <= padded; size *= 2U) {
        for (std::uint32_t stride = size / 2U; stride > 0U; stride /= 2U) {
            for (std::uint32_t index = threadIdx.x; index < padded;
                 index += blockDim.x) {
                const auto partner = index ^ stride;
                if (partner <= index) continue;
                const bool ascending = (index & size) != 0U;
                const bool swap = ascending
                    ? sort_keys[index] > sort_keys[partner]
                    : sort_keys[index] < sort_keys[partner];
                if (!swap) continue;
                const auto key = sort_keys[index];
                sort_keys[index] = sort_keys[partner];
                sort_keys[partner] = key;
                const auto value = sort_values[index];
                sort_values[index] = sort_values[partner];
                sort_values[partner] = value;
            }
            __syncthreads();
        }
    }
    for (std::uint32_t index = threadIdx.x; index < count;
         index += blockDim.x) {
        selected[index] = sort_values[index];
    }
}

__global__ void lightning_topk_initialize_kernel(
    float* top_scores, std::uint32_t* top_positions,
    std::uint32_t top_k) {
    for (std::uint32_t index = threadIdx.x; index < top_k;
         index += blockDim.x) {
        top_scores[index] = -INFINITY;
        top_positions[index] = UINT_MAX;
    }
}

__global__ void lightning_topk_merge_kernel(
    const float* scores, std::uint32_t candidates,
    float* top_scores, std::uint32_t* top_positions,
    std::uint32_t top_k) {
    if (threadIdx.x != 0U || blockIdx.x != 0U) return;
    for (std::uint32_t position = 0U; position < candidates; ++position) {
        const float score = scores[position];
        const auto last = top_k - 1U;
        if (score < top_scores[last] ||
            (score == top_scores[last] && position >= top_positions[last])) {
            continue;
        }
        std::uint32_t insert = last;
        while (insert != 0U &&
               (score > top_scores[insert - 1U] ||
                (score == top_scores[insert - 1U] &&
                 position < top_positions[insert - 1U]))) {
            top_scores[insert] = top_scores[insert - 1U];
            top_positions[insert] = top_positions[insert - 1U];
            --insert;
        }
        top_scores[insert] = score;
        top_positions[insert] = position;
    }
}

__global__ void packed_int4_moe_gate_up_kernel(
    float* activations, const float* hidden, PackedInt4MoeBatch batch,
    std::uint64_t columns, std::uint64_t intermediate,
    std::uint64_t packed_columns, std::uint64_t scale_columns,
    std::uint32_t group_size, unsigned int* error_flag) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t batch_row = blockIdx.y;
    const auto expert = batch_row / batch.rows;
    const auto row = batch_row % batch.rows;
    if (output_row >= intermediate || expert >= batch.count) return;

    const auto* gate_weights = batch.gate_weights[expert];
    const auto* gate_scales = batch.gate_scales[expert];
    const auto* up_weights = batch.up_weights[expert];
    const auto* up_scales = batch.up_scales[expert];
    const auto weight_base = output_row * packed_columns;
    const auto scale_base = output_row * scale_columns;
    const auto input_base = static_cast<std::uint64_t>(row) * columns;
    float gate = 0.0F;
    float up = 0.0F;
    for (std::uint64_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        const auto shift = static_cast<unsigned int>((column % 8U) * 4U);
        const auto gate_raw = (gate_weights[weight_base + column / 8U] >> shift) & 0x0FU;
        const auto up_raw = (up_weights[weight_base + column / 8U] >> shift) & 0x0FU;
        const auto scale_column = column / group_size;
        const float input = hidden[input_base + column];
        gate += input * static_cast<float>(static_cast<int>(gate_raw) - 8) *
                __bfloat162float(gate_scales[scale_base + scale_column]);
        up += input * static_cast<float>(static_cast<int>(up_raw) - 8) *
              __bfloat162float(up_scales[scale_base + scale_column]);
    }
    gate = reduce_block(gate);
    __syncthreads();
    up = reduce_block(up);
    if (threadIdx.x == 0U) {
        if (!isfinite(gate) || !isfinite(up)) {
            atomicExch(error_flag, 1U);
            return;
        }
        const float exponential = gate >= 0.0F ? expf(-gate) : expf(gate);
        const float sigmoid = gate >= 0.0F
                                  ? 1.0F / (1.0F + exponential)
                                  : exponential / (1.0F + exponential);
        const auto activation =
            (static_cast<std::uint64_t>(expert) * batch.rows + row) *
                intermediate + output_row;
        activations[activation] = gate * sigmoid * up * batch.coefficients[expert];
    }
}

__global__ void packed_int4_moe_down_kernel(
    float* output, const float* activations, PackedInt4MoeBatch batch,
    std::uint64_t columns, std::uint64_t rows,
    std::uint64_t packed_columns, std::uint64_t scale_columns,
    std::uint32_t group_size, unsigned int* error_flag) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t batch_row = blockIdx.y;
    const auto expert = batch_row / batch.rows;
    const auto row = batch_row % batch.rows;
    if (output_row >= rows || expert >= batch.count) return;

    const auto* weights = batch.down_weights[expert];
    const auto* scales = batch.down_scales[expert];
    const auto weight_base = output_row * packed_columns;
    const auto scale_base = output_row * scale_columns;
    const auto input_base =
        (static_cast<std::uint64_t>(expert) * batch.rows + row) * columns;
    float sum = 0.0F;
    for (std::uint64_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        const auto shift = static_cast<unsigned int>((column % 8U) * 4U);
        const auto raw = (weights[weight_base + column / 8U] >> shift) & 0x0FU;
        sum += activations[input_base + column] *
               static_cast<float>(static_cast<int>(raw) - 8) *
               __bfloat162float(scales[scale_base + column / group_size]);
    }
    sum = reduce_block(sum);
    if (threadIdx.x == 0U) {
        if (!isfinite(sum)) atomicExch(error_flag, 1U);
        output[(static_cast<std::uint64_t>(expert) * batch.rows + row) * rows +
               output_row] = sum;
    }
}

// NVFP4 group-16 counterparts of the INT4 pair above. The per-column decode is
// character for character the one in nvfp4_group16_matmul_kernel, and the block
// reduction is the same, so a batched expert produces the same value the
// per-expert matmul path produced. The routing coefficient is deliberately not
// folded into the activation here: Laguna's reference scales the expert output
// after the down projection, and down(act * c) is not float-equal to
// c * down(act).
__global__ void nvfp4_moe_gate_up_kernel(
    float* activations, const float* hidden, Nvfp4MoeBatch batch,
    std::uint64_t columns, std::uint64_t intermediate,
    std::uint64_t packed_columns, std::uint64_t scale_columns,
    std::uint32_t group_size, unsigned int* error_flag) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t batch_row = blockIdx.y;
    const auto expert = batch_row / batch.rows;
    const auto row = batch_row % batch.rows;
    if (output_row >= intermediate || expert >= batch.count) return;

    const auto* gate_weights = batch.gate_weights[expert];
    const auto* gate_scales = batch.gate_scales[expert];
    const auto* up_weights = batch.up_weights[expert];
    const auto* up_scales = batch.up_scales[expert];
    const float gate_global = batch.gate_global_scales[expert];
    const float up_global = batch.up_global_scales[expert];
    const auto weight_base = output_row * packed_columns;
    const auto scale_base = output_row * scale_columns;
    const auto input_base = static_cast<std::uint64_t>(row) * columns;
    float gate = 0.0F;
    float up = 0.0F;
    for (std::uint64_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        const auto packed_index = weight_base + column / 2U;
        const auto scale_index = scale_base + column / group_size;
        const unsigned char gate_packed = gate_weights[packed_index];
        const unsigned char up_packed = up_weights[packed_index];
        const unsigned int gate_encoded =
            column % 2U == 0U ? gate_packed & 0x0FU : gate_packed >> 4U;
        const unsigned int up_encoded =
            column % 2U == 0U ? up_packed & 0x0FU : up_packed >> 4U;
        const float input = hidden[input_base + column];
        gate += input * fp4_e2m1_value(gate_encoded) *
                (fp8_e4m3_value(gate_scales[scale_index]) / gate_global);
        up += input * fp4_e2m1_value(up_encoded) *
              (fp8_e4m3_value(up_scales[scale_index]) / up_global);
    }
    gate = reduce_block(gate);
    __syncthreads();
    up = reduce_block(up);
    if (threadIdx.x == 0U) {
        if (!isfinite(gate) || !isfinite(up)) {
            atomicExch(error_flag, 1U);
            return;
        }
        const float exponential = gate >= 0.0F ? expf(-gate) : expf(gate);
        const float sigmoid = gate >= 0.0F
                                  ? 1.0F / (1.0F + exponential)
                                  : exponential / (1.0F + exponential);
        const auto activation =
            (static_cast<std::uint64_t>(expert) * batch.rows + row) *
                intermediate + output_row;
        activations[activation] = gate * sigmoid * up;
    }
}

__global__ void nvfp4_moe_down_kernel(
    float* output, const float* activations, Nvfp4MoeBatch batch,
    std::uint64_t columns, std::uint64_t rows,
    std::uint64_t packed_columns, std::uint64_t scale_columns,
    std::uint32_t group_size, unsigned int* error_flag) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t batch_row = blockIdx.y;
    const auto expert = batch_row / batch.rows;
    const auto row = batch_row % batch.rows;
    if (output_row >= rows || expert >= batch.count) return;

    const auto* weights = batch.down_weights[expert];
    const auto* scales = batch.down_scales[expert];
    const float global_scale = batch.down_global_scales[expert];
    const auto weight_base = output_row * packed_columns;
    const auto scale_base = output_row * scale_columns;
    const auto input_base =
        (static_cast<std::uint64_t>(expert) * batch.rows + row) * columns;
    float sum = 0.0F;
    for (std::uint64_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        const unsigned char packed = weights[weight_base + column / 2U];
        const unsigned int encoded =
            column % 2U == 0U ? packed & 0x0FU : packed >> 4U;
        sum += activations[input_base + column] * fp4_e2m1_value(encoded) *
               (fp8_e4m3_value(scales[scale_base + column / group_size]) /
                global_scale);
    }
    sum = reduce_block(sum);
    if (threadIdx.x == 0U) {
        if (!isfinite(sum)) atomicExch(error_flag, 1U);
        output[(static_cast<std::uint64_t>(expert) * batch.rows + row) * rows +
               output_row] = sum;
    }
}

// Canonical MXFP4 counterparts. The hidden and intermediate activations are
// rounded to E4M3 by enqueue_moe immediately before these kernels, matching
// the scalar Fp4E2m1Group32 matmul route on the same weights.
__global__ void mxfp4_moe_gate_up_kernel(
    float* activations, const float* hidden, Mxfp4MoeBatch batch,
    std::uint64_t columns, std::uint64_t intermediate,
    std::uint64_t packed_columns, std::uint64_t scale_columns,
    unsigned int* error_flag) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t batch_row = blockIdx.y;
    const auto expert = batch_row / batch.rows;
    const auto row = batch_row % batch.rows;
    if (output_row >= intermediate || expert >= batch.count) return;

    const auto* gate_weights = batch.gate_weights[expert];
    const auto* gate_scales = batch.gate_scales[expert];
    const auto* up_weights = batch.up_weights[expert];
    const auto* up_scales = batch.up_scales[expert];
    const auto weight_base = output_row * packed_columns;
    const auto scale_base = output_row * scale_columns;
    const auto input_base = static_cast<std::uint64_t>(row) * columns;
    float gate = 0.0F;
    float up = 0.0F;
    for (std::uint64_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        const auto packed_index = weight_base + column / 2U;
        const auto scale_index = scale_base + column / 32U;
        const unsigned char gate_packed = gate_weights[packed_index];
        const unsigned char up_packed = up_weights[packed_index];
        const unsigned int gate_encoded =
            column % 2U == 0U ? gate_packed & 0x0FU : gate_packed >> 4U;
        const unsigned int up_encoded =
            column % 2U == 0U ? up_packed & 0x0FU : up_packed >> 4U;
        const float input = hidden[input_base + column];
        gate += input * fp4_e2m1_value(gate_encoded) *
                fp8_e8m0_scale_bits(gate_scales[scale_index]);
        up += input * fp4_e2m1_value(up_encoded) *
              fp8_e8m0_scale_bits(up_scales[scale_index]);
    }
    gate = reduce_block(gate);
    __syncthreads();
    up = reduce_block(up);
    if (threadIdx.x == 0U) {
        if (!isfinite(gate) || !isfinite(up)) {
            atomicExch(error_flag, 1U);
            return;
        }
        const float exponential = gate >= 0.0F ? expf(-gate) : expf(gate);
        const float sigmoid = gate >= 0.0F
                                  ? 1.0F / (1.0F + exponential)
                                  : exponential / (1.0F + exponential);
        const auto activation =
            (static_cast<std::uint64_t>(expert) * batch.rows + row) *
                intermediate + output_row;
        activations[activation] = gate * sigmoid * up;
    }
}

__global__ void mxfp4_moe_down_kernel(
    float* output, const float* activations, Mxfp4MoeBatch batch,
    std::uint64_t columns, std::uint64_t rows,
    std::uint64_t packed_columns, std::uint64_t scale_columns,
    unsigned int* error_flag) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t batch_row = blockIdx.y;
    const auto expert = batch_row / batch.rows;
    const auto row = batch_row % batch.rows;
    if (output_row >= rows || expert >= batch.count) return;

    const auto* weights = batch.down_weights[expert];
    const auto* scales = batch.down_scales[expert];
    const auto weight_base = output_row * packed_columns;
    const auto scale_base = output_row * scale_columns;
    const auto input_base =
        (static_cast<std::uint64_t>(expert) * batch.rows + row) * columns;
    float sum = 0.0F;
    for (std::uint64_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        const unsigned char packed = weights[weight_base + column / 2U];
        const unsigned int encoded =
            column % 2U == 0U ? packed & 0x0FU : packed >> 4U;
        sum += activations[input_base + column] * fp4_e2m1_value(encoded) *
               fp8_e8m0_scale_bits(scales[scale_base + column / 32U]);
    }
    sum = reduce_block(sum);
    if (threadIdx.x == 0U) {
        if (!isfinite(sum)) atomicExch(error_flag, 1U);
        output[(static_cast<std::uint64_t>(expert) * batch.rows + row) * rows +
               output_row] = sum;
    }
}

__global__ void fp8_f32_moe_gate_up_kernel(
    float* activations, const float* hidden, Fp8F32MoeBatch batch,
    std::uint64_t columns, std::uint64_t intermediate,
    std::uint64_t scale_columns, float swiglu_limit,
    unsigned int* error_flag) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t batch_row = blockIdx.y;
    const auto expert = batch_row / batch.rows;
    const auto row = batch_row % batch.rows;
    if (output_row >= intermediate || expert >= batch.count) return;

    const auto* gate_weights = batch.gate_weights[expert];
    const auto* gate_scales = batch.gate_scales[expert];
    const auto* up_weights = batch.up_weights[expert];
    const auto* up_scales = batch.up_scales[expert];
    const auto weight_base = output_row * columns;
    const auto scale_base = (output_row / 128U) * scale_columns;
    const auto input_base = static_cast<std::uint64_t>(row) * columns;
    float gate = 0.0F;
    float up = 0.0F;
    for (std::uint64_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        const float input = hidden[input_base + column];
        gate += input * fp8_e4m3_value(gate_weights[weight_base + column]) *
                gate_scales[scale_base + column / 128U];
        up += input * fp8_e4m3_value(up_weights[weight_base + column]) *
              up_scales[scale_base + column / 128U];
    }
    gate = reduce_block(gate);
    __syncthreads();
    up = reduce_block(up);
    if (threadIdx.x == 0U) {
        gate = bf16_round(gate);
        up = bf16_round(up);
        if (!isfinite(gate) || !isfinite(up)) {
            atomicExch(error_flag, 1U);
            return;
        }
        const float limited_gate = swiglu_limit > 0.0F
            ? fminf(gate, swiglu_limit) : gate;
        const float limited_up = swiglu_limit > 0.0F
            ? fminf(fmaxf(up, -swiglu_limit), swiglu_limit) : up;
        const float exponential = limited_gate >= 0.0F
            ? expf(-limited_gate) : expf(limited_gate);
        const float sigmoid = limited_gate >= 0.0F
            ? 1.0F / (1.0F + exponential)
            : exponential / (1.0F + exponential);
        const auto activation =
            (static_cast<std::uint64_t>(expert) * batch.rows + row) *
                intermediate + output_row;
        activations[activation] = bf16_round(
            limited_gate * sigmoid * limited_up);
    }
}

__global__ void fp8_f32_moe_down_kernel(
    float* output, const float* activations, Fp8F32MoeBatch batch,
    std::uint64_t columns, std::uint64_t rows,
    std::uint64_t scale_columns, unsigned int* error_flag) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t batch_row = blockIdx.y;
    const auto expert = batch_row / batch.rows;
    const auto row = batch_row % batch.rows;
    if (output_row >= rows || expert >= batch.count) return;

    const auto* weights = batch.down_weights[expert];
    const auto* scales = batch.down_scales[expert];
    const auto weight_base = output_row * columns;
    const auto scale_base = (output_row / 128U) * scale_columns;
    const auto input_base =
        (static_cast<std::uint64_t>(expert) * batch.rows + row) * columns;
    float sum = 0.0F;
    for (std::uint64_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        sum += activations[input_base + column] *
               fp8_e4m3_value(weights[weight_base + column]) *
               scales[scale_base + column / 128U];
    }
    sum = reduce_block(sum);
    if (threadIdx.x == 0U) {
        if (!isfinite(sum)) atomicExch(error_flag, 1U);
        output[(static_cast<std::uint64_t>(expert) * batch.rows + row) * rows +
               output_row] = bf16_round(sum);
    }
}

// Plain BF16 counterparts. These keep bf16_matvec_kernel's one-warp-per-output-
// row layout and its __fadd_rn/__shfl_down_sync reduction rather than the
// block reduction the quantized batches use, so the dot product is summed in
// the same order the per-expert path summed it.
__global__ void plain_bf16_moe_gate_up_kernel(
    float* activations, const float* hidden, PlainBf16MoeBatch batch,
    std::uint64_t columns, std::uint64_t intermediate,
    unsigned int* error_flag) {
    constexpr unsigned int warps_per_block = 8U;
    const auto warp = threadIdx.x / warpSize;
    const auto lane = threadIdx.x % warpSize;
    const auto output_row =
        static_cast<std::uint64_t>(blockIdx.x) * warps_per_block + warp;
    const std::uint32_t batch_row = blockIdx.y;
    const auto expert = batch_row / batch.rows;
    const auto row = batch_row % batch.rows;
    if (output_row >= intermediate || expert >= batch.count) return;

    const auto base = output_row * columns;
    const auto input_base = static_cast<std::uint64_t>(row) * columns;
    const auto* gate_weights = batch.gate_weights[expert];
    const auto* up_weights = batch.up_weights[expert];
    float gate = 0.0F;
    float up = 0.0F;
    for (std::uint64_t column = lane; column < columns; column += warpSize) {
        const float input = hidden[input_base + column];
        gate = __fadd_rn(
            gate,
            __fmul_rn(input, __bfloat162float(gate_weights[base + column])));
        up = __fadd_rn(
            up, __fmul_rn(input, __bfloat162float(up_weights[base + column])));
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate = __fadd_rn(gate, __shfl_down_sync(0xFFFF'FFFFU, gate, offset));
        up = __fadd_rn(up, __shfl_down_sync(0xFFFF'FFFFU, up, offset));
    }
    if (lane == 0U) {
        if (!isfinite(gate) || !isfinite(up)) {
            atomicExch(error_flag, 1U);
            return;
        }
        const float exponential = gate >= 0.0F ? expf(-gate) : expf(gate);
        const float sigmoid = gate >= 0.0F
                                  ? 1.0F / (1.0F + exponential)
                                  : exponential / (1.0F + exponential);
        activations[(static_cast<std::uint64_t>(expert) * batch.rows + row) *
                        intermediate + output_row] = gate * sigmoid * up;
    }
}

__global__ void plain_bf16_moe_down_kernel(
    float* output, const float* activations, PlainBf16MoeBatch batch,
    std::uint64_t columns, std::uint64_t rows, unsigned int* error_flag) {
    constexpr unsigned int warps_per_block = 8U;
    const auto warp = threadIdx.x / warpSize;
    const auto lane = threadIdx.x % warpSize;
    const auto output_row =
        static_cast<std::uint64_t>(blockIdx.x) * warps_per_block + warp;
    const std::uint32_t batch_row = blockIdx.y;
    const auto expert = batch_row / batch.rows;
    const auto row = batch_row % batch.rows;
    if (output_row >= rows || expert >= batch.count) return;

    const auto base = output_row * columns;
    const auto input_base =
        (static_cast<std::uint64_t>(expert) * batch.rows + row) * columns;
    const auto* weights = batch.down_weights[expert];
    float sum = 0.0F;
    for (std::uint64_t column = lane; column < columns; column += warpSize) {
        sum = __fadd_rn(
            sum, __fmul_rn(activations[input_base + column],
                           __bfloat162float(weights[base + column])));
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum = __fadd_rn(sum, __shfl_down_sync(0xFFFF'FFFFU, sum, offset));
    }
    if (lane == 0U) {
        if (!isfinite(sum)) atomicExch(error_flag, 1U);
        output[(static_cast<std::uint64_t>(expert) * batch.rows + row) * rows +
               output_row] = sum;
    }
}

// Device-resident routed-expert tier.
//
// The routed set is 143.7 GB against 64 GB of VRAM, so most experts must stay
// in host DRAM where the CPU reads them at 0.282 ms each. A 3090 computes the
// same expert from VRAM in 0.128 ms, so every expert that fits is worth moving.
//
// The obstacle was never the arithmetic, it was the ordering: routing is
// decided inside a cudaLaunchHostFunc callback, where CUDA calls are illegal,
// and a previous attempt that blocked that callback waiting on a worker thread
// stalled the whole context (experiment 0124).
//
// This avoids the callback entirely. Host functions are stream-ordered, so a
// kernel enqueued behind the callback is guaranteed to observe its writes. The
// callback therefore writes the experts it is NOT going to compute into pinned
// memory -- a plain store, no CUDA API -- and the kernel behind it reads that
// selection, looks the weights up in a table built once at load, and
// accumulates into the same rank-partial buffer the existing join already
// consumes. No worker thread, no events, no host wait, no change to the join.
struct DeepSeekTierTable {
    // Indexed by layer * experts + expert. Null weight means not resident, in
    // which case the host owes that expert and this kernel must skip it.
    const unsigned char* const* w1_weights{};
    const unsigned char* const* w1_scales{};
    const unsigned char* const* w3_weights{};
    const unsigned char* const* w3_scales{};
    const unsigned char* const* w2_weights{};
    const unsigned char* const* w2_scales{};
    std::uint32_t experts{};
};

// One selection per layer, written by the callback into pinned memory and
// uploaded on the same stream. `expert` is kDeepSeekTierAbsent for a slot this
// tier does not serve.
constexpr std::uint32_t kDeepSeekTierAbsent = 0xFFFF'FFFFU;
struct DeepSeekTierSelection {
    std::uint32_t experts[kMaxDeepSeekRoutedExperts]{};
    float coefficients[kMaxDeepSeekRoutedExperts]{};
    std::uint32_t layer{};
    std::uint32_t count{};
};

// Gate/up for the tier's slots. Grid is (intermediate, kMaxDeepSeekRoutedExperts);
// a block whose slot is absent exits immediately, which costs a launch but keeps
// the grid shape independent of a route only the host knows.
__global__ void deepseek_fp4_tier_gate_up_kernel(
    float* activations, const float* hidden, DeepSeekTierTable table,
    const DeepSeekTierSelection* selection, std::uint64_t columns,
    std::uint64_t intermediate, std::uint64_t packed_columns,
    std::uint64_t scale_columns) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t slot = blockIdx.y;
    if (output_row >= intermediate || slot >= kMaxDeepSeekRoutedExperts) return;
    const std::uint32_t expert = selection->experts[slot];
    if (expert == kDeepSeekTierAbsent) return;
    const std::uint64_t entry =
        static_cast<std::uint64_t>(selection->layer) * table.experts + expert;
    const auto* w1 = table.w1_weights[entry];
    const auto* w3 = table.w3_weights[entry];
    if (w1 == nullptr || w3 == nullptr) return;
    const auto* w1_scales = table.w1_scales[entry];
    const auto* w3_scales = table.w3_scales[entry];

    const std::uint64_t packed_base = output_row * packed_columns;
    const std::uint64_t scale_base = output_row * scale_columns;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    float gate = 0.0F;
    float up = 0.0F;
    for (std::uint64_t group = warp; group < scale_columns; group += 8U) {
        float gate_scale = lane == 0U
                               ? fp8_e8m0_scale_bits(w1_scales[scale_base + group])
                               : 0.0F;
        float up_scale = lane == 0U
                             ? fp8_e8m0_scale_bits(w3_scales[scale_base + group])
                             : 0.0F;
        gate_scale = __shfl_sync(0xFFFF'FFFFU, gate_scale, 0);
        up_scale = __shfl_sync(0xFFFF'FFFFU, up_scale, 0);
        const std::uint64_t column = group * 32U + lane;
        if (column < columns) {
            const float input = hidden[column];
            const unsigned char gate_packed = w1[packed_base + column / 2U];
            const unsigned char up_packed = w3[packed_base + column / 2U];
            const unsigned int gate_encoded = column % 2U == 0U
                                                  ? gate_packed & 0x0FU
                                                  : gate_packed >> 4U;
            const unsigned int up_encoded = column % 2U == 0U
                                                ? up_packed & 0x0FU
                                                : up_packed >> 4U;
            gate += input * fp4_e2m1_value(gate_encoded) * gate_scale;
            up += input * fp4_e2m1_value(up_encoded) * up_scale;
        }
    }
    gate = reduce_block(gate);
    __syncthreads();
    up = reduce_block(up);
    if (threadIdx.x == 0U) {
        const float exponential = gate >= 0.0F ? expf(-gate) : expf(gate);
        const float sigmoid = gate >= 0.0F
                                  ? 1.0F / (1.0F + exponential)
                                  : exponential / (1.0F + exponential);
        activations[static_cast<std::uint64_t>(slot) * intermediate + output_row] =
            gate * sigmoid * up;
    }
}

// Down projection, accumulating each served expert's contribution into the
// rank partial the join already reads. Accumulation is by atomicAdd across
// slots because the slots are independent blocks; within a row the order is
// therefore not fixed, which is the same reassociation the CPU path's own
// cross-slot sum performs.
__global__ void deepseek_fp4_tier_down_kernel(
    float* rank_partials, const float* activations, DeepSeekTierTable table,
    const DeepSeekTierSelection* selection, std::uint64_t columns,
    std::uint64_t rows, std::uint64_t packed_columns,
    std::uint64_t scale_columns) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t slot = blockIdx.y;
    if (output_row >= rows || slot >= kMaxDeepSeekRoutedExperts) return;
    const std::uint32_t expert = selection->experts[slot];
    if (expert == kDeepSeekTierAbsent) return;
    const std::uint64_t entry =
        static_cast<std::uint64_t>(selection->layer) * table.experts + expert;
    const auto* weights = table.w2_weights[entry];
    if (weights == nullptr) return;
    const auto* scales = table.w2_scales[entry];
    const float coefficient = selection->coefficients[slot];

    const std::uint64_t packed_base = output_row * packed_columns;
    const std::uint64_t scale_base = output_row * scale_columns;
    const std::uint64_t input_base = static_cast<std::uint64_t>(slot) * columns;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    float sum = 0.0F;
    for (std::uint64_t group = warp; group < scale_columns; group += 8U) {
        float scale = lane == 0U
                          ? fp8_e8m0_scale_bits(scales[scale_base + group])
                          : 0.0F;
        scale = __shfl_sync(0xFFFF'FFFFU, scale, 0);
        const std::uint64_t column = group * 32U + lane;
        if (column < columns) {
            const unsigned char packed = weights[packed_base + column / 2U];
            const unsigned int encoded = column % 2U == 0U
                                             ? packed & 0x0FU
                                             : packed >> 4U;
            sum += activations[input_base + column] *
                   fp4_e2m1_value(encoded) * scale;
        }
    }
    sum = reduce_block(sum);
    if (threadIdx.x == 0U) {
        atomicAdd(&rank_partials[output_row], sum * coefficient);
    }
}

__global__ void deepseek_fp4_gate_up_kernel(
    float* activations, const float* hidden, DeepSeekFp4Batch batch,
    std::uint64_t columns, std::uint64_t intermediate,
    std::uint64_t packed_columns, std::uint64_t scale_columns,
    float swiglu_limit, const float* bf16_silu, unsigned int* error_flag) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t expert = blockIdx.y;
    if (output_row >= intermediate || expert >= batch.count) return;

    const auto* w1 = batch.w1_weights[expert];
    const auto* w1_scales = batch.w1_scales[expert];
    const auto* w3 = batch.w3_weights[expert];
    const auto* w3_scales = batch.w3_scales[expert];
    const std::uint64_t packed_base = output_row * packed_columns;
    const std::uint64_t scale_base = output_row * scale_columns;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    float gate = 0.0F;
    float up = 0.0F;

    // One warp owns each 32-weight group. Its lane-zero scale decode is shared
    // by shuffle, and each hidden value feeds both W1 and W3 accumulators.
    for (std::uint64_t group = warp; group < scale_columns; group += 8U) {
        float gate_scale = lane == 0U
                               ? fp8_e8m0_scale_bits(w1_scales[scale_base + group])
                               : 0.0F;
        float up_scale = lane == 0U
                             ? fp8_e8m0_scale_bits(w3_scales[scale_base + group])
                             : 0.0F;
        gate_scale = __shfl_sync(0xFFFF'FFFFU, gate_scale, 0);
        up_scale = __shfl_sync(0xFFFF'FFFFU, up_scale, 0);
        const std::uint64_t column = group * 32U + lane;
        if (column < columns) {
            const float input = hidden[column];
            const unsigned char gate_packed = w1[packed_base + column / 2U];
            const unsigned char up_packed = w3[packed_base + column / 2U];
            const unsigned int gate_encoded = column % 2U == 0U
                                                  ? gate_packed & 0x0FU
                                                  : gate_packed >> 4U;
            const unsigned int up_encoded = column % 2U == 0U
                                                ? up_packed & 0x0FU
                                                : up_packed >> 4U;
            gate += input * fp4_e2m1_value(gate_encoded) * gate_scale;
            up += input * fp4_e2m1_value(up_encoded) * up_scale;
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
            activations[static_cast<std::uint64_t>(expert) * intermediate +
                        output_row] = __uint_as_float(0x7FC0'0000U);
            return;
        }
        const float limited_gate = fminf(rounded_gate, swiglu_limit);
        const float limited_up = fmaxf(-swiglu_limit,
                                       fminf(rounded_up, swiglu_limit));
        const auto gate_bits = static_cast<std::uint16_t>(
            __float_as_uint(limited_gate) >> 16U);
        float activated = bf16_silu[gate_bits] * limited_up;
        activated *= batch.coefficients[expert];
        activations[static_cast<std::uint64_t>(expert) * intermediate + output_row] =
            bf16_round(activated);
    }
}

__global__ void deepseek_fp4_down_kernel(
    float* output, const float* activations, DeepSeekFp4Batch batch,
    std::uint64_t columns, std::uint64_t rows,
    std::uint64_t packed_columns, std::uint64_t scale_columns) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t expert = blockIdx.y;
    if (output_row >= rows || expert >= batch.count) return;
    const auto* weights = batch.w2_weights[expert];
    const auto* scales = batch.w2_scales[expert];
    const std::uint64_t packed_base = output_row * packed_columns;
    const std::uint64_t scale_base = output_row * scale_columns;
    const std::uint64_t input_base = static_cast<std::uint64_t>(expert) * columns;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    float sum = 0.0F;
    for (std::uint64_t group = warp; group < scale_columns; group += 8U) {
        float scale = lane == 0U
                          ? fp8_e8m0_scale_bits(scales[scale_base + group])
                          : 0.0F;
        scale = __shfl_sync(0xFFFF'FFFFU, scale, 0);
        const std::uint64_t column = group * 32U + lane;
        if (column < columns) {
            const unsigned char packed = weights[packed_base + column / 2U];
            const unsigned int encoded = column % 2U == 0U
                                             ? packed & 0x0FU
                                             : packed >> 4U;
            sum += activations[input_base + column] *
                   fp4_e2m1_value(encoded) * scale;
        }
    }
    sum = reduce_block(sum);
    if (threadIdx.x == 0U) {
        output[static_cast<std::uint64_t>(expert) * rows + output_row] =
            bf16_round(sum);
    }
}

// Row tile of the page-batched FP4 path. Eight rows hold sixteen accumulators
// per thread, which fits without spilling at 256 threads, and amortises each
// 32-weight group's nibble and scale decode across eight rows instead of one.
constexpr std::uint32_t kDeepSeekPageRowTile = 32U;

// One routed expert of a page and the slice of the work list it owns. The work
// list is flattened group-major; `row_offset` is where this group's rows begin
// in `work_rows`/`work_coefficients` and is also where its outputs begin.
struct DeepSeekFp4PageGroup {
    const unsigned char* w1_weights{};
    const unsigned char* w1_scales{};
    const unsigned char* w3_weights{};
    const unsigned char* w3_scales{};
    const unsigned char* w2_weights{};
    const unsigned char* w2_scales{};
    // Transformed shards of this expert, one per intermediate-dimension TP
    // shard. Set together with a non-zero shard_intermediate, and then the six
    // canonical pointers above are unused.
    const unsigned char* tiled[kCudaDsv4TiledShards]{};
    std::uint32_t shard_intermediate{};
    std::uint32_t row_offset{};
    std::uint32_t row_count{};
};

__global__ void deepseek_fp4_page_gate_up_kernel(
    float* activations, const float* hidden, const std::uint32_t* work_rows,
    const float* work_coefficients, const DeepSeekFp4PageGroup* groups,
    std::uint32_t group_count, std::uint64_t columns, std::uint64_t intermediate,
    std::uint64_t packed_columns, std::uint64_t scale_columns,
    float swiglu_limit, const float* bf16_silu, unsigned int* error_flag) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t group_index = blockIdx.y;
    if (output_row >= intermediate || group_index >= group_count) return;
    const DeepSeekFp4PageGroup group = groups[group_index];
    const std::uint32_t tile_begin = blockIdx.z * kDeepSeekPageRowTile;
    if (tile_begin >= group.row_count) return;
    const std::uint32_t tile_rows =
        min(kDeepSeekPageRowTile, group.row_count - tile_begin);

    const std::uint64_t packed_base = output_row * packed_columns;
    const std::uint64_t scale_base = output_row * scale_columns;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    float gate[kDeepSeekPageRowTile];
    float up[kDeepSeekPageRowTile];
    for (std::uint32_t index = 0U; index < kDeepSeekPageRowTile; ++index) {
        gate[index] = 0.0F;
        up[index] = 0.0F;
    }

    for (std::uint64_t group_column = warp; group_column < scale_columns;
         group_column += 8U) {
        float gate_scale =
            lane == 0U
                ? fp8_e8m0_scale_bits(group.w1_scales[scale_base + group_column])
                : 0.0F;
        float up_scale =
            lane == 0U
                ? fp8_e8m0_scale_bits(group.w3_scales[scale_base + group_column])
                : 0.0F;
        gate_scale = __shfl_sync(0xFFFF'FFFFU, gate_scale, 0);
        up_scale = __shfl_sync(0xFFFF'FFFFU, up_scale, 0);
        const std::uint64_t column = group_column * 32U + lane;
        if (column >= columns) continue;
        const unsigned char gate_packed = group.w1_weights[packed_base + column / 2U];
        const unsigned char up_packed = group.w3_weights[packed_base + column / 2U];
        const unsigned int gate_encoded =
            column % 2U == 0U ? gate_packed & 0x0FU : gate_packed >> 4U;
        const unsigned int up_encoded =
            column % 2U == 0U ? up_packed & 0x0FU : up_packed >> 4U;
        const float gate_weight = fp4_e2m1_value(gate_encoded);
        const float up_weight = fp4_e2m1_value(up_encoded);
        // Fixed trip count with a clamped index: a runtime bound here would
        // make `gate`/`up` dynamically indexed, which spills them out of
        // registers into local memory and costs more than the tail it saves.
        // Rows past `tile_rows` recompute row zero and are never written.
#pragma unroll
        for (std::uint32_t index = 0U; index < kDeepSeekPageRowTile; ++index) {
            const std::uint32_t local = index < tile_rows ? index : 0U;
            const std::uint64_t row =
                work_rows[group.row_offset + tile_begin + local];
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
        const std::uint32_t slot = group.row_offset + tile_begin + index;
        const std::uint64_t destination =
            static_cast<std::uint64_t>(slot) * intermediate + output_row;
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
        float activated = bf16_silu[gate_bits] * limited_up;
        activated *= work_coefficients[slot];
        activations[destination] = bf16_round(activated);
    }
}

__global__ void deepseek_fp4_page_down_kernel(
    float* output, const float* activations,
    const DeepSeekFp4PageGroup* groups, std::uint32_t group_count,
    std::uint64_t columns, std::uint64_t rows, std::uint64_t packed_columns,
    std::uint64_t scale_columns) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t group_index = blockIdx.y;
    if (output_row >= rows || group_index >= group_count) return;
    const DeepSeekFp4PageGroup group = groups[group_index];
    const std::uint32_t tile_begin = blockIdx.z * kDeepSeekPageRowTile;
    if (tile_begin >= group.row_count) return;
    const std::uint32_t tile_rows =
        min(kDeepSeekPageRowTile, group.row_count - tile_begin);

    const std::uint64_t packed_base = output_row * packed_columns;
    const std::uint64_t scale_base = output_row * scale_columns;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    float sum[kDeepSeekPageRowTile];
    for (std::uint32_t index = 0U; index < kDeepSeekPageRowTile; ++index) {
        sum[index] = 0.0F;
    }

    for (std::uint64_t group_column = warp; group_column < scale_columns;
         group_column += 8U) {
        float scale =
            lane == 0U
                ? fp8_e8m0_scale_bits(group.w2_scales[scale_base + group_column])
                : 0.0F;
        scale = __shfl_sync(0xFFFF'FFFFU, scale, 0);
        const std::uint64_t column = group_column * 32U + lane;
        if (column >= columns) continue;
        const unsigned char packed = group.w2_weights[packed_base + column / 2U];
        const unsigned int encoded =
            column % 2U == 0U ? packed & 0x0FU : packed >> 4U;
        const float weight = fp4_e2m1_value(encoded);
#pragma unroll
        for (std::uint32_t index = 0U; index < kDeepSeekPageRowTile; ++index) {
            const std::uint32_t local = index < tile_rows ? index : 0U;
            const std::uint64_t slot = group.row_offset + tile_begin + local;
            sum[index] += activations[slot * columns + column] * weight * scale;
        }
    }

#pragma unroll 1
    for (std::uint32_t index = 0U; index < kDeepSeekPageRowTile; ++index) {
        __syncthreads();
        const float reduced = reduce_block(sum[index]);
        if (threadIdx.x != 0U || index >= tile_rows) continue;
        const std::uint64_t slot = group.row_offset + tile_begin + index;
        output[slot * rows + output_row] = bf16_round(reduced);
    }
}

// The transformed layout puts 32 consecutive output rows of one block in 32
// consecutive bytes, so a warp that owns those rows reads a whole sector per
// fetch. The canonical decomposition cannot: its block owns one output row and
// spreads 256 threads across the reduction, so it uses a thirty-second of
// every sector it pulls -- measured at 2.87x on the kernel.
//
// Owning the block instead means one lane per output row, and each lane sums
// its own row over the whole reduction in increasing column order. That is a
// reassociation of the canonical 256-partial tree, not a different
// computation: the same terms, the same per-term
// `fma(input * fp4_value, scale, accumulator)`, summed in a different order.
// It is therefore gated against the scalar oracle rather than against the
// canonical kernel bit for bit.
constexpr std::uint32_t kDeepSeekTiledWarps = 8U;
constexpr std::uint32_t kDeepSeekTiledRowTile = 8U;

__global__ void deepseek_fp4_tiled_page_gate_up_kernel(
    float* activations, const float* hidden, const std::uint32_t* work_rows,
    const float* work_coefficients, const DeepSeekFp4PageGroup* groups,
    std::uint32_t group_count, std::uint64_t columns,
    std::uint64_t intermediate, float swiglu_limit, const float* bf16_silu,
    unsigned int* error_flag) {
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint64_t output_block =
        static_cast<std::uint64_t>(blockIdx.x) * kDeepSeekTiledWarps + warp;
    const std::uint32_t group_index = blockIdx.y;
    const std::uint64_t output_row = output_block * 32U + lane;
    if (output_row >= intermediate || group_index >= group_count) return;
    const DeepSeekFp4PageGroup group = groups[group_index];
    const std::uint32_t tile_begin = blockIdx.z * kDeepSeekTiledRowTile;
    if (tile_begin >= group.row_count) return;
    const std::uint32_t tile_rows =
        min(kDeepSeekTiledRowTile, group.row_count - tile_begin);

    const std::uint64_t shard_intermediate = group.shard_intermediate;
    const std::uint32_t shard =
        static_cast<std::uint32_t>(output_row / shard_intermediate);
    const std::uint64_t row_in_shard =
        output_row - static_cast<std::uint64_t>(shard) * shard_intermediate;
    const unsigned char* base = group.tiled[shard];
    const unsigned char* scales_base =
        base + 2U * shard_intermediate * (columns / 2U);
    // A shard width is a multiple of 32, so every lane of the warp shares the
    // transform block and differs only in `within` -- which is the lane index.
    // That is what makes each fetch one contiguous 32-byte run.
    const std::uint64_t gate_block = row_in_shard >> 5U;
    const std::uint64_t up_block = (shard_intermediate + row_in_shard) >> 5U;
    const unsigned char* gate_packed =
        base + gate_block * (columns / 2U) * 32U + lane;
    const unsigned char* up_packed =
        base + up_block * (columns / 2U) * 32U + lane;
    const unsigned char* gate_scales =
        scales_base + gate_block * (columns / 16U) * 32U + lane;
    const unsigned char* up_scales =
        scales_base + up_block * (columns / 16U) * 32U + lane;

    float gate[kDeepSeekTiledRowTile];
    float up[kDeepSeekTiledRowTile];
    std::uint64_t input_base[kDeepSeekTiledRowTile];
#pragma unroll
    for (std::uint32_t index = 0U; index < kDeepSeekTiledRowTile; ++index) {
        gate[index] = 0.0F;
        up[index] = 0.0F;
        const std::uint32_t local = index < tile_rows ? index : 0U;
        input_base[index] =
            static_cast<std::uint64_t>(
                work_rows[group.row_offset + tile_begin + local]) *
            columns;
    }

    float gate_scale = 0.0F;
    float up_scale = 0.0F;
    for (std::uint64_t pair = 0U; pair < columns / 2U; ++pair) {
        const std::uint64_t column = pair * 2U;
        if ((column & 31U) == 0U) {
            // Each canonical group-32 scale is stored twice, once per group-16
            // half; the first copy is taken.
            const std::uint64_t slot = (column >> 5U) * 64U;
            gate_scale = fp8_e8m0_scale_bits(gate_scales[slot]);
            up_scale = fp8_e8m0_scale_bits(up_scales[slot]);
        }
        const unsigned char gate_byte = gate_packed[pair * 32U];
        const unsigned char up_byte = up_packed[pair * 32U];
        const float gate_low = fp4_e2m1_value(gate_byte & 0x0FU);
        const float gate_high = fp4_e2m1_value(gate_byte >> 4U);
        const float up_low = fp4_e2m1_value(up_byte & 0x0FU);
        const float up_high = fp4_e2m1_value(up_byte >> 4U);
#pragma unroll
        for (std::uint32_t index = 0U; index < kDeepSeekTiledRowTile; ++index) {
            const float first = hidden[input_base[index] + column];
            const float second = hidden[input_base[index] + column + 1U];
            gate[index] = fmaf(first * gate_low, gate_scale, gate[index]);
            gate[index] = fmaf(second * gate_high, gate_scale, gate[index]);
            up[index] = fmaf(first * up_low, up_scale, up[index]);
            up[index] = fmaf(second * up_high, up_scale, up[index]);
        }
    }

#pragma unroll
    for (std::uint32_t index = 0U; index < kDeepSeekTiledRowTile; ++index) {
        if (index >= tile_rows) continue;
        const std::uint32_t slot = group.row_offset + tile_begin + index;
        const std::uint64_t destination =
            static_cast<std::uint64_t>(slot) * intermediate + output_row;
        const float rounded_gate = bf16_round(gate[index]);
        const float rounded_up = bf16_round(up[index]);
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
        float activated = bf16_silu[gate_bits] * limited_up;
        activated *= work_coefficients[slot];
        activations[destination] = bf16_round(activated);
    }
}

__global__ void deepseek_fp4_tiled_page_down_kernel(
    float* output, const float* activations,
    const DeepSeekFp4PageGroup* groups, std::uint32_t group_count,
    std::uint64_t columns, std::uint64_t rows) {
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint64_t output_block =
        static_cast<std::uint64_t>(blockIdx.x) * kDeepSeekTiledWarps + warp;
    const std::uint32_t group_index = blockIdx.y;
    const std::uint64_t output_row = output_block * 32U + lane;
    if (output_row >= rows || group_index >= group_count) return;
    const DeepSeekFp4PageGroup group = groups[group_index];
    const std::uint32_t tile_begin = blockIdx.z * kDeepSeekTiledRowTile;
    if (tile_begin >= group.row_count) return;
    const std::uint32_t tile_rows =
        min(kDeepSeekTiledRowTile, group.row_count - tile_begin);

    const std::uint64_t shard_intermediate = group.shard_intermediate;
    const std::uint64_t block = output_block;
    float sum[kDeepSeekTiledRowTile];
    std::uint64_t input_base[kDeepSeekTiledRowTile];
#pragma unroll
    for (std::uint32_t index = 0U; index < kDeepSeekTiledRowTile; ++index) {
        sum[index] = 0.0F;
        const std::uint32_t local = index < tile_rows ? index : 0U;
        input_base[index] =
            static_cast<std::uint64_t>(
                group.row_offset + tile_begin + local) * columns;
    }

    // Down reduces over the intermediate dimension, which is what the
    // transform shards, so the shards are visited in order and each supplies a
    // contiguous run of the reduction.
    for (std::uint32_t shard = 0U; shard < kCudaDsv4TiledShards; ++shard) {
        const std::uint64_t shard_base =
            static_cast<std::uint64_t>(shard) * shard_intermediate;
        if (shard_base >= columns) break;
        const unsigned char* base = group.tiled[shard] +
            2U * shard_intermediate * (rows / 2U) +
            2U * shard_intermediate * (rows / 16U);
        const unsigned char* packed =
            base + block * (shard_intermediate / 2U) * 32U + lane;
        const unsigned char* scales = base +
            rows * (shard_intermediate / 2U) +
            block * (shard_intermediate / 16U) * 32U + lane;
        float scale = 0.0F;
        for (std::uint64_t pair = 0U; pair < shard_intermediate / 2U; ++pair) {
            const std::uint64_t local_column = pair * 2U;
            if ((local_column & 31U) == 0U) {
                scale = fp8_e8m0_scale_bits(scales[(local_column >> 5U) * 64U]);
            }
            const unsigned char byte = packed[pair * 32U];
            const float low = fp4_e2m1_value(byte & 0x0FU);
            const float high = fp4_e2m1_value(byte >> 4U);
            const std::uint64_t column = shard_base + local_column;
#pragma unroll
            for (std::uint32_t index = 0U; index < kDeepSeekTiledRowTile;
                 ++index) {
                const float first = activations[input_base[index] + column];
                const float second = activations[input_base[index] + column + 1U];
                sum[index] = fmaf(first * low, scale, sum[index]);
                sum[index] = fmaf(second * high, scale, sum[index]);
            }
        }
    }

#pragma unroll
    for (std::uint32_t index = 0U; index < kDeepSeekTiledRowTile; ++index) {
        if (index >= tile_rows) continue;
        const std::uint64_t slot = group.row_offset + tile_begin + index;
        output[slot * rows + output_row] = bf16_round(sum[index]);
    }
}

// ---------------------------------------------------------------------------
// Register-fed W8A16 shared expert (campaign milestone MIX-1).
//
// The incumbent deepseek_fp8_gate_up/down kernels are scalar matvecs: one block
// per output row, one weight byte per lane, no tensor cores. Experiment 0143
// measured that shape at about 13.5% of DRAM and 68% issue-bound. The accepted
// QPN8-derived path (experiments 0158-0159) reaches 83-86% of the local read
// roofline on these same E4M3/E8M0 block-128 shapes by keeping codes compressed
// through HBM, pre-permuting them into m16n8k16 fragment order at load, and
// decoding straight into MMA operand registers.
//
// Weights are prepacked in place: the fragment order REPLACES the canonical
// device layout rather than adding a second copy, which is what the contract's
// one-copy residency rule requires.
// ---------------------------------------------------------------------------

// Canonical [N][K] E4M3 bytes to m16n8k16 A-fragment order. One uint4 per lane
// per 32-column pair-group, matching the layout the decode below expects.
__global__ void dsv4_fp8_fragment_prepack_kernel(
    uint4* __restrict__ destination, const unsigned char* __restrict__ source,
    std::uint32_t rows, std::uint32_t columns) {
    const std::uint32_t pairs = columns / 32U;
    const std::uint32_t tiles = rows / 16U;
    const std::uint32_t total = tiles * pairs * 32U;
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += gridDim.x * blockDim.x) {
        const std::uint32_t lane = index & 31U;
        const std::uint32_t pair = (index >> 5U) % pairs;
        const std::uint32_t tile = (index >> 5U) / pairs;
        std::uint32_t word[4]{};
        for (std::uint32_t j = 0U; j < 2U; ++j) {
            for (std::uint32_t i = 0U; i < 8U; ++i) {
                const bool upper = i == 2U || i == 3U || i == 6U || i == 7U;
                const std::uint32_t row =
                    tile * 16U + (lane >> 2U) + (upper ? 8U : 0U);
                const std::uint32_t column = (pair * 2U + j) * 16U +
                                             (lane & 3U) * 2U + (i & 1U) +
                                             (i >= 4U ? 8U : 0U);
                word[j * 2U + (i >= 4U ? 1U : 0U)] |=
                    static_cast<std::uint32_t>(
                        source[static_cast<std::size_t>(row) * columns + column])
                    << ((i & 3U) * 8U);
            }
        }
        destination[index] = make_uint4(word[0], word[1], word[2], word[3]);
    }
}

// Canonical packed E2M1 [N][K/2] to Marlin's K16/N64 tensor-fragment order.
// One destination word contains the eight nibbles consumed by one lane.
__global__ void gemma_marlin_prepack_codes_kernel(
    std::uint32_t* destination, const unsigned char* source,
    std::uint32_t rows, std::uint32_t columns) {
    constexpr std::uint32_t n_tile = 64U;
    constexpr std::uint32_t k_tile = 16U;
    constexpr std::uint32_t tensor_offsets[4] = {0U, 1U, 8U, 9U};
    constexpr std::uint32_t pack_order[8] = {0U, 2U, 4U, 6U,
                                              1U, 3U, 5U, 7U};
    const std::uint64_t total =
        static_cast<std::uint64_t>(rows) * columns / 8U;
    for (std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += gridDim.x * blockDim.x) {
        const auto within_tile = static_cast<std::uint32_t>(index % 128U);
        const auto tile = index / 128U;
        const auto lane = within_tile / 4U;
        const auto warp = within_tile % 4U;
        const auto n_tiles = rows / n_tile;
        const auto kt = static_cast<std::uint32_t>(tile / n_tiles);
        const auto nt = static_cast<std::uint32_t>(tile % n_tiles);
        const auto tensor_column = lane / 4U;
        const auto tensor_row = (lane % 4U) * 2U;
        const auto first_n = nt * n_tile + warp * 16U + tensor_column;
        std::uint32_t values[8]{};
#pragma unroll
        for (std::uint32_t i = 0U; i < 4U; ++i) {
            const auto column =
                kt * k_tile + tensor_row + tensor_offsets[i];
            const auto read_code = [&](std::uint32_t row) {
                const auto packed = source[
                    static_cast<std::uint64_t>(row) * (columns / 2U) +
                    column / 2U];
                return static_cast<std::uint32_t>(
                    column & 1U ? packed >> 4U : packed & 0x0FU);
            };
            values[i] = read_code(first_n);
            values[4U + i] = read_code(first_n + 8U);
        }
        std::uint32_t packed = 0U;
#pragma unroll
        for (std::uint32_t i = 0U; i < 8U; ++i) {
            packed |= values[pack_order[i]] << (i * 4U);
        }
        destination[index] = packed;
    }
}

// Canonical E8M0 [N][K/32] to Marlin's transposed 64-value tensor order,
// followed by the MXFP4 [0,2,1,3] lane permutation.
__global__ void gemma_marlin_prepack_scales_kernel(
    unsigned char* destination, const unsigned char* source,
    std::uint32_t rows, std::uint32_t groups) {
    constexpr std::uint32_t permutation[64] = {
         0,  8, 16, 24, 32, 40, 48, 56,
         1,  9, 17, 25, 33, 41, 49, 57,
         2, 10, 18, 26, 34, 42, 50, 58,
         3, 11, 19, 27, 35, 43, 51, 59,
         4, 12, 20, 28, 36, 44, 52, 60,
         5, 13, 21, 29, 37, 45, 53, 61,
         6, 14, 22, 30, 38, 46, 54, 62,
         7, 15, 23, 31, 39, 47, 55, 63};
    const std::uint64_t total = static_cast<std::uint64_t>(rows) * groups;
    for (std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += gridDim.x * blockDim.x) {
        const auto base = index & ~std::uint64_t{63U};
        auto local = static_cast<std::uint32_t>(index & 63U);
        if ((local & 3U) == 1U) ++local;
        else if ((local & 3U) == 2U) --local;
        const auto transposed = base + permutation[local];
        const auto group = static_cast<std::uint32_t>(transposed / rows);
        const auto row = static_cast<std::uint32_t>(transposed % rows);
        destination[index] = source[static_cast<std::uint64_t>(row) * groups +
                                    group];
    }
}

// Two packed E4M3 codes to a packed BF16 pair, scaled, entirely in registers.
__device__ __forceinline__ std::uint32_t dsv4_fp8_decode_pair(
    std::uint32_t pair, std::uint32_t factor) {
    const auto permuted = __byte_perm(pair, 0U, 0x4140U);
    const std::uint32_t widened =
        ((permuted << 8U) & 0x8000'8000U) | ((permuted << 4U) & 0x07F0'07F0U);
    const auto scaled =
        __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&widened),
                *reinterpret_cast<const __nv_bfloat162*>(&factor));
    return *reinterpret_cast<const std::uint32_t*>(&scaled);
}

__device__ __forceinline__ void dsv4_mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3, std::uint32_t a0,
    std::uint32_t a1, std::uint32_t a2, std::uint32_t a3, std::uint32_t b0,
    std::uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ---------------------------------------------------------------------------
// Generic register-fed W4A16 / W8A16 matmul (campaign milestone MIX-2).
//
// The two kernels below are the accepted QPN skinny-kernel shape made model
// agnostic. Nothing here knows about DeepSeek: they take a weight in
// m16n8k16 fragment order, an activation already permuted into B-fragment
// order, and produce the same [M][N] output the scalar kernels produce, so any
// architecture whose weights carry Fp4E2m1Group32 or Fp8E4m3Block128 encoding
// reaches them through CudaBackend::matmul_impl.
//
// Numerical contract against the incumbent scalar kernels. matmul_impl rounds
// the activation to E4M3 before either path runs, and an E4M3 value has three
// mantissa bits, so its BF16 image is exact. An E2M1 code has one mantissa bit
// and an E4M3 code three; an E8M0 scale is a power of two. Every operand of the
// tensor op is therefore the same real number the scalar kernel multiplies --
// the paths differ only in FP32 accumulation order, not in operand precision.
//
// Shape admission is explicit and there is no silent fallback: a weight whose
// shape the fragment layout cannot express keeps the scalar route, and the
// census records which one ran.
// ---------------------------------------------------------------------------

constexpr std::uint32_t kRegfedTileN = 16U;   // MMA M dimension = weight rows
constexpr std::uint32_t kRegfedTileM = 8U;    // MMA N dimension = activation cols
constexpr std::uint32_t kRegfedTileK = 16U;   // MMA K dimension
constexpr std::uint32_t kRegfedWarp = 32U;
constexpr std::uint32_t kRegfedGroup = 32U;   // E8M0 group along K for FP4
constexpr std::uint32_t kRegfedNvfp4Group = 16U;  // E4M3 group along K, NVFP4
// Experiment 0140 measured argmax as load granularity: one uint4 per lane per
// four K-tiles makes a warp issue one fully coalesced 512-byte transaction that
// feeds four MMAs.
constexpr std::uint32_t kRegfedKPerLoad = 4U;
constexpr std::uint32_t kRegfedWarpsPerBlock = 4U;
// One m16n8k16 covers eight activation columns. Two column blocks cover M<=16,
// which is the whole skinny regime; wider M keeps the scalar route, where the
// weight read is already amortized across many rows.
constexpr std::uint32_t kRegfedMaxColBlocks = 2U;
constexpr std::uint32_t kRegfedMaxM = kRegfedTileM * kRegfedMaxColBlocks;
// FP8 decode folds a 2^120 exponent correction into the block scale, so the
// E8M0 code must leave the BF16 multiplier normal: code + 120 in [1, 254].
constexpr std::uint32_t kRegfedFp8ScaleCodeMaximum = 134U;

__constant__ std::uint32_t kRegfedFp4MagnitudeHigh[2] = {0x3F3F'3F00U,
                                                         0x4040'4040U};
__constant__ std::uint32_t kRegfedFp4MagnitudeLow[2] = {0xC080'0000U,
                                                        0xC080'4000U};

__device__ __forceinline__ std::uint32_t regfed_fp4_scale_pair(
    std::uint32_t code) {
    return (code << 7U) * 0x0001'0001U;
}

// One NVFP4 E4M3 group scale to the same broadcast BF16 pair.
//
// The E8M0 form above is two instructions because a power-of-two scale is only
// an exponent field. An E4M3 scale carries three mantissa bits, so the code has
// to be widened into BF16's fields and re-biased: E4M3's exponent bias is 7 and
// BF16's is 127, which is the constant 120 added to the exponent. Both formats
// then reach `regfed_fp4_decode_fragment` as an ordinary BF16 multiplier, so
// the decode itself is shared and only this helper differs.
//
// Exactness. The widened field is exact -- an E4M3 value has three mantissa
// bits and BF16 has seven -- and so is the product with an E2M1 code, whose
// significand needs one bit: (1 + a/8)(1 + b/2) has at most four fractional
// bits before normalisation and five after, still inside BF16's seven. The
// scaled weight the tensor op consumes is therefore the same real number the
// scalar kernel multiplies, exactly as the header's contract requires.
//
// Subnormals are normalised rather than widened. An E4M3 subnormal is
// mmm * 2^-9, which BF16 represents as a normal number, and feeding the mma a
// subnormal operand would risk flush-to-zero on a value the scalar oracle
// keeps. The branch is uniform-false for every scale in the shipped
// checkpoint and predicates away.
//
// Codes 0x7F and 0xFF are E4M3 NaN, and this construction maps them to a large
// finite BF16 rather than to NaN, where `fp8_e4m3_value` returns NaN. That is
// the E4M3 counterpart of 0148's open E8M0 0/255 admission check and it is a
// load-time question, not a kernel one: it belongs beside
// `dsv4_admit_e8m0_scales` when this path is integrated. Experiment 0247
// censuses the shipped checkpoint for these codes and reports the count.
__device__ __forceinline__ std::uint32_t regfed_nvfp4_scale_pair(
    std::uint32_t code) {
    const std::uint32_t sign = (code << 8U) & 0x0000'8000U;
    // eeeemmm lands in bits 10..4: exponent at BF16's bit 7, mantissa below it.
    std::uint32_t bits = ((code & 0x7FU) << 4U) + (120U << 7U);
    if ((code & 0x78U) == 0U) {
        const std::uint32_t mantissa = code & 0x07U;
        // mmm * 2^-9 == 2^(s-9) * (1 + f), s = floor(log2(mmm)).
        const std::uint32_t shift = 31U - __clz(mantissa | 1U);
        bits = mantissa == 0U
                   ? 0U
                   : (((127U - 9U + shift) << 7U) |
                      ((mantissa << (7U - shift)) & 0x7FU));
    }
    return (bits | sign) * 0x0001'0001U;
}

// Eight E2M1 codes to four packed BF16 pairs, in MMA A-fragment register order.
// Registers 0 and 2 carry weight row g, registers 1 and 3 carry row g+8, so the
// two rows' E8M0 scales are selected by register parity -- experiment 0140's
// scale-to-K binding defect was exactly this selection applied flat.
__device__ __forceinline__ void regfed_fp4_decode_fragment(
    std::uint32_t word, std::uint32_t scale_low_row,
    std::uint32_t scale_high_row, std::uint32_t (&out)[4]) {
    const std::uint32_t mag = word & 0x7777'7777U;
    const std::uint32_t ha =
        __byte_perm(kRegfedFp4MagnitudeHigh[0], kRegfedFp4MagnitudeHigh[1], mag);
    const std::uint32_t la =
        __byte_perm(kRegfedFp4MagnitudeLow[0], kRegfedFp4MagnitudeLow[1], mag);
    const std::uint32_t hb = __byte_perm(
        kRegfedFp4MagnitudeHigh[0], kRegfedFp4MagnitudeHigh[1], mag >> 16U);
    const std::uint32_t lb = __byte_perm(
        kRegfedFp4MagnitudeLow[0], kRegfedFp4MagnitudeLow[1], mag >> 16U);
    const std::uint32_t pa = __byte_perm(la, ha, 0x5140U);
    const std::uint32_t qa = __byte_perm(la, ha, 0x7362U);
    const std::uint32_t pb = __byte_perm(lb, hb, 0x5140U);
    const std::uint32_t qb = __byte_perm(lb, hb, 0x7362U);
    std::uint32_t value[4];
    value[0] = __byte_perm(pa, pb, 0x5410U);
    value[1] = __byte_perm(pa, pb, 0x7632U);
    value[2] = __byte_perm(qa, qb, 0x5410U);
    value[3] = __byte_perm(qa, qb, 0x7632U);
#pragma unroll
    for (std::uint32_t i = 0U; i < 4U; ++i) {
        const std::uint32_t scale =
            ((i & 1U) == 0U) ? scale_low_row : scale_high_row;
        // Code 0x8 is negative zero in E2M1 but the oracle's zero is positive,
        // so the sign is suppressed when the magnitude is zero.
        const std::uint32_t signs = (word << (12U - i * 4U)) & 0x8000'8000U;
        const std::uint32_t non_zero = value[i] + 0x7F80'7F80U;
        const __nv_bfloat162 scaled =
            __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&value[i]),
                    *reinterpret_cast<const __nv_bfloat162*>(&scale));
        out[i] = *reinterpret_cast<const std::uint32_t*>(&scaled) ^
                 (signs & non_zero & 0x8000'8000U);
    }
}

// ---- layout transforms -----------------------------------------------------

// Canonical FP4 [N][K/2] nibble pairs to fragment order. A pure permutation:
// the destination holds N*K/2 bytes, exactly what the source holds, so the
// prepack can run in place through transient scratch.
__global__ void regfed_fp4_prepack_codes_kernel(
    std::uint32_t* __restrict__ destination,
    const unsigned char* __restrict__ source, std::uint32_t rows,
    std::uint32_t columns) {
    const std::uint32_t k_tiles = columns / kRegfedTileK;
    const std::uint32_t n_tiles = rows / kRegfedTileN;
    const std::uint32_t total = n_tiles * k_tiles * kRegfedWarp;
    const std::size_t packed_columns = columns / 2U;
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += gridDim.x * blockDim.x) {
        const std::uint32_t lane = index & 31U;
        const std::uint32_t k_tile = (index >> 5U) % k_tiles;
        const std::uint32_t n_tile = (index >> 5U) / k_tiles;
        const std::uint32_t group = lane >> 2U;
        const std::uint32_t thread = lane & 3U;
        std::uint32_t word = 0U;
#pragma unroll
        for (std::uint32_t i = 0U; i < 4U; ++i) {
            const std::uint32_t row =
                n_tile * kRegfedTileN + group + (((i & 1U) != 0U) ? 8U : 0U);
            const std::uint32_t column =
                k_tile * kRegfedTileK + thread * 2U + ((i >= 2U) ? 8U : 0U);
            // The column is always even, so one byte carries both codes.
            const unsigned char pair =
                source[static_cast<std::size_t>(row) * packed_columns +
                       column / 2U];
            word |= static_cast<std::uint32_t>(pair & 0x0FU) << (i * 4U);
            word |= static_cast<std::uint32_t>(pair >> 4U) << ((i + 4U) * 4U);
        }
        const std::uint32_t block = k_tile / kRegfedKPerLoad;
        const std::uint32_t slot = k_tile % kRegfedKPerLoad;
        destination[((static_cast<std::size_t>(n_tile) *
                          (k_tiles / kRegfedKPerLoad) + block) * kRegfedWarp +
                     lane) * kRegfedKPerLoad + slot] = word;
    }
}

// Canonical FP4 [N][K/32] E8M0 scales to tile-major order, so one uint4 load
// per lane covers all sixteen rows of an N-tile for one K-group.
__global__ void regfed_fp4_prepack_scales_kernel(
    unsigned char* __restrict__ destination,
    const unsigned char* __restrict__ source, std::uint32_t rows,
    std::uint32_t scale_columns) {
    const std::uint32_t n_tiles = rows / kRegfedTileN;
    const std::uint32_t total = n_tiles * scale_columns * kRegfedTileN;
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += gridDim.x * blockDim.x) {
        const std::uint32_t row = index % kRegfedTileN;
        const std::uint32_t group = (index / kRegfedTileN) % scale_columns;
        const std::uint32_t n_tile = (index / kRegfedTileN) / scale_columns;
        destination[index] =
            source[(static_cast<std::size_t>(n_tile) * kRegfedTileN + row) *
                       scale_columns + group];
    }
}

// FP32 activation [M][K] to MMA B-fragment order. For lane (group, thread) of
// column block c, b0 carries K rows {2t, 2t+1} and b1 carries {2t+8, 2t+9} of
// activation column c*8 + group. Columns past M are a stored zero rather than a
// branch in the inner loop.
__global__ void regfed_activation_fragment_kernel(
    uint2* __restrict__ destination, const float* __restrict__ source,
    std::uint32_t m, std::uint32_t columns, std::uint32_t column_blocks,
    std::uint32_t groups_per_block) {
    const std::uint32_t k_tiles = columns / kRegfedTileK;
    const std::uint32_t total = k_tiles * column_blocks * groups_per_block * 4U;
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += gridDim.x * blockDim.x) {
        const std::uint32_t thread = index % 4U;
        const std::uint32_t group = (index / 4U) % groups_per_block;
        const std::uint32_t block =
            (index / (4U * groups_per_block)) % column_blocks;
        const std::uint32_t k_tile = index / (4U * groups_per_block * column_blocks);
        const std::uint32_t column = block * kRegfedTileM + group;
        std::uint32_t b0 = 0U;
        std::uint32_t b1 = 0U;
        if (column < m) {
            const float* row = source + static_cast<std::size_t>(column) * columns;
            const auto bits = [&](std::uint32_t offset) {
                return static_cast<std::uint32_t>(
                    __bfloat16_as_ushort(__float2bfloat16_rn(
                        row[k_tile * kRegfedTileK + offset])));
            };
            b0 = bits(thread * 2U) | (bits(thread * 2U + 1U) << 16U);
            b1 = bits(thread * 2U + 8U) | (bits(thread * 2U + 9U) << 16U);
        }
        destination[index] = make_uint2(b0, b1);
    }
}

// Compact E4M3 activation codes to MMA B-fragment order. GLM keeps its
// continuous activation scale beside these codes; the tensor dot sees only
// the raw representable values and the scale is applied to each K128 partial
// in F32 by regfed_fp8_f32_matmul_kernel.
__global__ void regfed_fp8_activation_fragment_kernel(
    uint2* __restrict__ destination,
    const unsigned char* __restrict__ source, std::uint32_t m,
    std::uint32_t columns, std::uint32_t column_blocks,
    std::uint32_t groups_per_block) {
    const std::uint32_t k_tiles = columns / kRegfedTileK;
    const std::uint32_t total =
        k_tiles * column_blocks * groups_per_block * 4U;
    constexpr std::uint32_t unit_factor =
        ((127U + 120U) << 7U) * 0x0001'0001U;
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += gridDim.x * blockDim.x) {
        const std::uint32_t thread = index % 4U;
        const std::uint32_t group = (index / 4U) % groups_per_block;
        const std::uint32_t block =
            (index / (4U * groups_per_block)) % column_blocks;
        const std::uint32_t k_tile =
            index / (4U * groups_per_block * column_blocks);
        const std::uint32_t column = block * kRegfedTileM + group;
        uint2 value = make_uint2(0U, 0U);
        if (column < m) {
            const auto* row = source +
                static_cast<std::size_t>(column) * columns +
                k_tile * kRegfedTileK;
            const std::uint32_t first =
                static_cast<std::uint32_t>(row[thread * 2U]) |
                (static_cast<std::uint32_t>(row[thread * 2U + 1U]) << 8U);
            const std::uint32_t second =
                static_cast<std::uint32_t>(row[thread * 2U + 8U]) |
                (static_cast<std::uint32_t>(row[thread * 2U + 9U]) << 8U);
            value = make_uint2(dsv4_fp8_decode_pair(first, unit_factor),
                               dsv4_fp8_decode_pair(second, unit_factor));
        }
        destination[index] = value;
    }
}

// ---- the kernels -----------------------------------------------------------

// One warp owns one (N-tile, K-slice). The last slice of a tile folds the
// split-K reduction itself: a separate reduce kernel cost a full 4.10 us of
// dispatch for trivial work, 29% of the step at these matrix sizes.
template <std::uint32_t kColBlocks>
__global__ __launch_bounds__(128) void regfed_fp4_matmul_kernel(
    float* __restrict__ output, const std::uint32_t* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const uint2* __restrict__ activations, std::uint32_t columns,
    std::uint32_t rows, std::uint32_t split, std::uint32_t m,
    std::uint32_t groups_per_block, float* __restrict__ partials,
    std::uint32_t* __restrict__ counters) {
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint32_t n_tiles = rows / kRegfedTileN;
    const std::uint32_t k_tiles = columns / kRegfedTileK;
    const std::uint32_t k_blocks = k_tiles / kRegfedKPerLoad;
    const std::uint32_t blocks_per_slice = k_blocks / split;
    const std::uint32_t scale_columns = columns / kRegfedGroup;
    const std::uint32_t group = lane >> 2U;
    const std::uint32_t thread = lane & 3U;
    const std::uint32_t shift = (group & 3U) * 8U;
    __shared__ std::uint32_t arrived[kRegfedWarpsPerBlock];

    bool live[kColBlocks];
    std::size_t activation_offset[kColBlocks];
#pragma unroll
    for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
        live[c] = group < groups_per_block && c * kRegfedTileM + group < m;
        activation_offset[c] =
            (static_cast<std::size_t>(c) * groups_per_block + group) * 4U + thread;
    }

    for (std::uint32_t work = blockIdx.x * kRegfedWarpsPerBlock + warp;
         work < n_tiles * split; work += gridDim.x * kRegfedWarpsPerBlock) {
        const std::uint32_t n_tile = work / split;
        const std::uint32_t slice = work % split;
        float acc[kColBlocks][4];
#pragma unroll
        for (std::uint32_t c = 0U; c < kColBlocks; ++c)
#pragma unroll
            for (std::uint32_t i = 0U; i < 4U; ++i) acc[c][i] = 0.0F;

        const uint4* code4 = reinterpret_cast<const uint4*>(codes);
        const std::uint32_t begin = slice * blocks_per_slice;
        const std::uint32_t end = begin + blocks_per_slice;
        for (std::uint32_t block = begin; block < end; ++block) {
            const uint4 packed =
                code4[(static_cast<std::size_t>(n_tile) * k_blocks + block) *
                          kRegfedWarp + lane];
            const unsigned char* base =
                scales + (static_cast<std::size_t>(n_tile) * scale_columns +
                          block * 2U) * kRegfedTileN;
            const uint4 even = *reinterpret_cast<const uint4*>(base);
            const uint4 odd =
                *reinterpret_cast<const uint4*>(base + kRegfedTileN);
            const std::uint32_t word[kRegfedKPerLoad] = {packed.x, packed.y,
                                                         packed.z, packed.w};
#pragma unroll
            for (std::uint32_t j = 0U; j < kRegfedKPerLoad; ++j) {
                // K-tiles 0 and 1 of the block sit in the even E8M0 group, 2
                // and 3 in the odd one; the select is compile time.
                const uint4 chosen = (j < 2U) ? even : odd;
                const std::uint32_t low_word = (group < 4U) ? chosen.x : chosen.y;
                const std::uint32_t high_word = (group < 4U) ? chosen.z : chosen.w;
                std::uint32_t a[4];
                regfed_fp4_decode_fragment(
                    word[j], regfed_fp4_scale_pair((low_word >> shift) & 0xFFU),
                    regfed_fp4_scale_pair((high_word >> shift) & 0xFFU), a);
                const std::size_t tile_base =
                    (static_cast<std::size_t>(block) * kRegfedKPerLoad + j) *
                    kColBlocks * groups_per_block * 4U;
#pragma unroll
                for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
                    const uint2 b =
                        live[c] ? activations[tile_base + activation_offset[c]]
                                : make_uint2(0U, 0U);
                    dsv4_mma_m16n8k16(acc[c][0], acc[c][1], acc[c][2], acc[c][3],
                                      a[0], a[1], a[2], a[3], b.x, b.y);
                }
            }
        }

        // D fragment: row = group + (i>=2 ? 8 : 0), column = thread*2 + (i&1),
        // offset by the column block. Columns past M are never stored, which
        // keeps split-K partial traffic proportional to the real M rather than
        // to the padded tile.
        float* slot = partials + (static_cast<std::size_t>(work)) *
                                     kRegfedTileN * kRegfedMaxM;
#pragma unroll
        for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
#pragma unroll
            for (std::uint32_t i = 0U; i < 4U; ++i) {
                const std::uint32_t row = group + ((i >= 2U) ? 8U : 0U);
                const std::uint32_t column =
                    c * kRegfedTileM + thread * 2U + (i & 1U);
                if (column < m) slot[row * m + column] = acc[c][i];
            }
        }

        __threadfence();
        __syncwarp();
        if (lane == 0U) {
            arrived[warp] = atomicAdd(&counters[n_tile], 1U);
        }
        __syncwarp();
        if (arrived[warp] == split - 1U) {
            if (lane < kRegfedTileN) {
                for (std::uint32_t column = 0U; column < m; ++column) {
                    float sum = 0.0F;
                    for (std::uint32_t s = 0U; s < split; ++s) {
                        sum += partials[(static_cast<std::size_t>(n_tile) *
                                             split + s) * kRegfedTileN *
                                            kRegfedMaxM + lane * m + column];
                    }
                    // matmul_impl's output is [M][N], not [N][M].
                    output[static_cast<std::size_t>(column) * rows +
                           n_tile * kRegfedTileN + lane] = sum;
                }
            }
            if (lane == 0U) counters[n_tile] = 0U;
        }
    }
}

// Split one FP32 activation into kTerms BF16 B-fragments, stored as kTerms
// planes of the layout `regfed_activation_fragment_kernel` already produces.
//
// WHY THIS EXISTS. Experiment 0247 measured the single-term NVFP4 kernel at
// 6.108e-03 against the scalar kernel over 60 real fixtures -- four orders
// outside 0157's 7.53e-07 -- while its control arm, which feeds both kernels a
// BF16-exact activation, measured 5.621e-07 on the same fixtures. So the whole
// departure is this boundary and none of it is the weight path: BF16 keeps
// eight significand bits, so rounding an FP32 activation costs 2^-8 relative
// per element, and an exact weight operand cannot recover it.
//
// FP32 carries 24 significand bits, so THREE BF16 terms carry an FP32
// activation exactly. `hi = bf16(x)` takes the top eight; `x - hi` is exact in
// FP32 because it needs only the sixteen bits the rounding discarded; `mid`
// takes the next eight and `lo` the last eight. Accumulating
// sum(w*hi) + sum(w*mid) + sum(w*lo) therefore multiplies the same real number
// the scalar kernel multiplies, and the production arm collapses onto the
// control arm -- what is left is summation order, which the contract already
// permits. Two terms leave 2^-16 and one leaves 2^-8.
//
// The cost is kTerms mma per K-tile and kTerms times the activation traffic.
// The weight stream is unchanged and dominates: one 2048x4096 expert is 4.5 MB
// of codes and scales against 64 KB of activation fragments at M=8.
__device__ __forceinline__ void regfed_bf16_split(float value,
                                                  std::uint32_t terms,
                                                  std::uint32_t* out) {
    for (std::uint32_t t = 0U; t < terms; ++t) {
        const __nv_bfloat16 rounded = __float2bfloat16_rn(value);
        out[t] = static_cast<std::uint32_t>(__bfloat16_as_ushort(rounded));
        // Exact in FP32: the residual needs only the bits the rounding dropped.
        value -= __bfloat162float(rounded);
    }
}

template <std::uint32_t kTerms>
__global__ void regfed_nvfp4_activation_fragment_kernel(
    uint2* __restrict__ destination, const float* __restrict__ source,
    std::uint32_t m, std::uint32_t columns, std::uint32_t column_blocks,
    std::uint32_t groups_per_block) {
    const std::uint32_t k_tiles = columns / kRegfedTileK;
    const std::uint32_t total = k_tiles * column_blocks * groups_per_block * 4U;
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += gridDim.x * blockDim.x) {
        const std::uint32_t thread = index % 4U;
        const std::uint32_t group = (index / 4U) % groups_per_block;
        const std::uint32_t block =
            (index / (4U * groups_per_block)) % column_blocks;
        const std::uint32_t k_tile =
            index / (4U * groups_per_block * column_blocks);
        const std::uint32_t column = block * kRegfedTileM + group;
        // b0 carries K rows {2t, 2t+1} and b1 carries {2t+8, 2t+9}, exactly as
        // the single-term kernel lays them out.
        std::uint32_t part[4][kTerms];
#pragma unroll
        for (std::uint32_t i = 0U; i < 4U; ++i)
#pragma unroll
            for (std::uint32_t t = 0U; t < kTerms; ++t) part[i][t] = 0U;
        if (column < m) {
            const float* row = source +
                static_cast<std::size_t>(column) * columns +
                k_tile * kRegfedTileK;
            const std::uint32_t offset[4] = {thread * 2U, thread * 2U + 1U,
                                             thread * 2U + 8U,
                                             thread * 2U + 9U};
#pragma unroll
            for (std::uint32_t i = 0U; i < 4U; ++i) {
                regfed_bf16_split(row[offset[i]], kTerms, part[i]);
            }
        }
#pragma unroll
        for (std::uint32_t t = 0U; t < kTerms; ++t) {
            destination[static_cast<std::size_t>(t) * total + index] =
                make_uint2(part[0][t] | (part[1][t] << 16U),
                           part[2][t] | (part[3][t] << 16U));
        }
    }
}

// The batched activation split, over experts, for the grouped dispatch. One
// launch covers a whole page's activations for the same reason
// regfed_nvfp4_grouped_matmul_kernel covers its matmuls: a page presents
// hundreds of experts and a per-expert launch is the starvation experiment 0247
// measured at 337 GB/s.
//
// Layout is expert-major with the kTerms planes inside, so slice `e`'s
// activation pointer is `base + e * kTerms * per_expert` and the matmul body's
// own plane stride is unchanged.
template <std::uint32_t kTerms>
__global__ void regfed_nvfp4_moe_activation_fragment_kernel(
    uint2* __restrict__ destination, const float* __restrict__ source,
    std::uint32_t experts, std::uint32_t m, std::uint32_t columns,
    std::uint32_t column_blocks, std::uint32_t groups_per_block) {
    const std::uint32_t k_tiles = columns / kRegfedTileK;
    const std::uint32_t per_expert =
        k_tiles * column_blocks * groups_per_block * 4U;
    const std::uint64_t total =
        static_cast<std::uint64_t>(experts) * per_expert;
    for (std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += gridDim.x * blockDim.x) {
        const auto local = static_cast<std::uint32_t>(index % per_expert);
        const auto expert = static_cast<std::uint32_t>(index / per_expert);
        const std::uint32_t thread = local % 4U;
        const std::uint32_t group = (local / 4U) % groups_per_block;
        const std::uint32_t block =
            (local / (4U * groups_per_block)) % column_blocks;
        const std::uint32_t k_tile =
            local / (4U * groups_per_block * column_blocks);
        const std::uint32_t column = block * kRegfedTileM + group;
        std::uint32_t part[4][kTerms];
#pragma unroll
        for (std::uint32_t i = 0U; i < 4U; ++i)
#pragma unroll
            for (std::uint32_t t = 0U; t < kTerms; ++t) part[i][t] = 0U;
        if (column < m) {
            const float* row = source +
                (static_cast<std::size_t>(expert) * m + column) * columns +
                k_tile * kRegfedTileK;
            const std::uint32_t offset[4] = {thread * 2U, thread * 2U + 1U,
                                             thread * 2U + 8U,
                                             thread * 2U + 9U};
#pragma unroll
            for (std::uint32_t i = 0U; i < 4U; ++i) {
                regfed_bf16_split(row[offset[i]], kTerms, part[i]);
            }
        }
        const std::size_t base =
            static_cast<std::size_t>(expert) * kTerms * per_expert;
#pragma unroll
        for (std::uint32_t t = 0U; t < kTerms; ++t) {
            destination[base + static_cast<std::size_t>(t) * per_expert + local] =
                make_uint2(part[0][t] | (part[1][t] << 16U),
                           part[2][t] | (part[3][t] << 16U));
        }
    }
}

// The NVFP4 counterpart of `regfed_fp4_matmul_kernel`. Codes are the same E2M1
// nibble pairs in the same fragment order, so `regfed_fp4_prepack_codes_kernel`
// serves both formats unchanged and `regfed_fp4_prepack_scales_kernel` is
// already parameterised by `scale_columns`. Three things differ:
//
//   * groups of 16, so one K-tile is exactly one scale group and a block of
//     four K-tiles reads four uint4 of scales where MXFP4 reads two;
//   * an E4M3 scale, decoded by `regfed_nvfp4_scale_pair`;
//   * a per-tensor FP32 divisor.
//
// THE DIVISOR IS A REASSOCIATION, AND IT IS THE ONE DELIBERATE NUMERICAL
// DIFFERENCE IN THIS KERNEL. `glm53_shared_expert_nvfp4_dot_kernel` divides
// every group scale by `global_scale` before multiplying, accumulating
// sum(x_i * w_i * (s_i / g)). It cannot fold into the BF16 weight here: the
// quotient s_i/g is an arbitrary FP32 value, and rounding it into BF16 would
// destroy the exactness the scaled weight otherwise has. So it applies once to
// the FP32 accumulator after the mma and after the split-K reduction, giving
// (sum(x_i * w_i * s_i)) / g. Division is used rather than a reciprocal
// multiply so the single rounding matches the scalar kernel's operation.
//
// This path is held to a tolerance, not to an output hash, under the owner
// ruling of 2026-09-04. With kTerms = 3 the activation boundary is exact and
// the tolerance is the reassociation alone; experiment 0247 measured both.
//
// The body is shared with the grouped form below so the two cannot drift.
template <std::uint32_t kColBlocks, std::uint32_t kTerms>

__device__ __forceinline__ void regfed_nvfp4_matmul_body(
    float* __restrict__ output, const std::uint32_t* __restrict__ codes,
    const unsigned char* __restrict__ scales, float global_scale,
    const uint2* __restrict__ activations, std::uint32_t columns,
    std::uint32_t rows, std::uint32_t split, std::uint32_t m,
    std::uint32_t groups_per_block, float* __restrict__ partials,
    std::uint32_t* __restrict__ counters) {
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint32_t n_tiles = rows / kRegfedTileN;
    const std::uint32_t k_tiles = columns / kRegfedTileK;
    const std::uint32_t k_blocks = k_tiles / kRegfedKPerLoad;
    const std::uint32_t blocks_per_slice = k_blocks / split;
    const std::uint32_t scale_columns = columns / kRegfedNvfp4Group;
    const std::uint32_t group = lane >> 2U;
    const std::uint32_t thread = lane & 3U;
    const std::uint32_t shift = (group & 3U) * 8U;
    // One activation plane per BF16 term, in the order the split kernel wrote.
    const std::size_t plane = static_cast<std::size_t>(k_tiles) * kColBlocks *
                              groups_per_block * 4U;
    __shared__ std::uint32_t arrived[kRegfedWarpsPerBlock];

    bool live[kColBlocks];
    std::size_t activation_offset[kColBlocks];
#pragma unroll
    for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
        live[c] = group < groups_per_block && c * kRegfedTileM + group < m;
        activation_offset[c] =
            (static_cast<std::size_t>(c) * groups_per_block + group) * 4U + thread;
    }

    for (std::uint32_t work = blockIdx.x * kRegfedWarpsPerBlock + warp;
         work < n_tiles * split; work += gridDim.x * kRegfedWarpsPerBlock) {
        const std::uint32_t n_tile = work / split;
        const std::uint32_t slice = work % split;
        // ONE ACCUMULATOR PER TERM, not one shared accumulator.
        //
        // Experiment 0247 measured the shared form at 2.222e-06 against the
        // scalar kernel with a three-term split that represents the activation
        // EXACTLY -- so the residual was not the split. It was this: a mid-term
        // product is 2^-8 of a hi-term product, and adding it into a running
        // sum of hi-terms rounds at 2^-24 of that running sum, which discards
        // most of what the extra term was computed to recover. Summing each
        // term in its own accumulator keeps the mid sum accurate relative to
        // its own magnitude, and the three are combined once at publication,
        // smallest first.
        float acc[kColBlocks][kTerms][4];
#pragma unroll
        for (std::uint32_t c = 0U; c < kColBlocks; ++c)
#pragma unroll
            for (std::uint32_t t = 0U; t < kTerms; ++t)
#pragma unroll
                for (std::uint32_t i = 0U; i < 4U; ++i) acc[c][t][i] = 0.0F;

        const uint4* code4 = reinterpret_cast<const uint4*>(codes);
        const std::uint32_t begin = slice * blocks_per_slice;
        const std::uint32_t end = begin + blocks_per_slice;
        for (std::uint32_t block = begin; block < end; ++block) {
            const uint4 packed =
                code4[(static_cast<std::size_t>(n_tile) * k_blocks + block) *
                          kRegfedWarp + lane];
            // One K-tile is one group of 16, so the block's four K-tiles need
            // four scale groups where the E8M0 path needs two.
            const unsigned char* base =
                scales + (static_cast<std::size_t>(n_tile) * scale_columns +
                          block * kRegfedKPerLoad) * kRegfedTileN;
            const std::uint32_t word[kRegfedKPerLoad] = {packed.x, packed.y,
                                                         packed.z, packed.w};
#pragma unroll
            for (std::uint32_t j = 0U; j < kRegfedKPerLoad; ++j) {
                const uint4 chosen = *reinterpret_cast<const uint4*>(
                    base + j * kRegfedTileN);
                const std::uint32_t low_word = (group < 4U) ? chosen.x : chosen.y;
                const std::uint32_t high_word = (group < 4U) ? chosen.z : chosen.w;
                std::uint32_t a[4];
                regfed_fp4_decode_fragment(
                    word[j], regfed_nvfp4_scale_pair((low_word >> shift) & 0xFFU),
                    regfed_nvfp4_scale_pair((high_word >> shift) & 0xFFU), a);
                const std::size_t tile_base =
                    (static_cast<std::size_t>(block) * kRegfedKPerLoad + j) *
                    kColBlocks * groups_per_block * 4U;
#pragma unroll
                for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
                    // Every term shares the decoded weight fragment, so the
                    // extra cost is the mma and the fragment read, never a
                    // second pass over the weight stream.
#pragma unroll
                    for (std::uint32_t t = 0U; t < kTerms; ++t) {
                        const uint2 b =
                            live[c] ? activations[static_cast<std::size_t>(t) *
                                                      plane + tile_base +
                                                  activation_offset[c]]
                                    : make_uint2(0U, 0U);
                        dsv4_mma_m16n8k16(acc[c][t][0], acc[c][t][1],
                                          acc[c][t][2], acc[c][t][3], a[0],
                                          a[1], a[2], a[3], b.x, b.y);
                    }
                }
            }
        }

        // Stride on the live m, not on kRegfedMaxM: only 16 x m of each slot
        // is ever written, and a wide prefill page allocates one slot per
        // expert, so the padded stride costs 16x the partial workspace at M=1.
        float* slot = partials + (static_cast<std::size_t>(work)) *
                                     kRegfedTileN * m;
#pragma unroll
        for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
#pragma unroll
            for (std::uint32_t i = 0U; i < 4U; ++i) {
                const std::uint32_t row = group + ((i >= 2U) ? 8U : 0U);
                const std::uint32_t column =
                    c * kRegfedTileM + thread * 2U + (i & 1U);
                // Smallest term first, so the correction is not lost to the
                // rounding of the sum it is correcting.
                float total = acc[c][kTerms - 1U][i];
#pragma unroll
                for (std::uint32_t t = kTerms - 1U; t > 0U; --t) {
                    total += acc[c][t - 1U][i];
                }
                if (column < m) slot[row * m + column] = total;
            }
        }

        __threadfence();
        __syncwarp();
        if (lane == 0U) {
            arrived[warp] = atomicAdd(&counters[n_tile], 1U);
        }
        __syncwarp();
        if (arrived[warp] == split - 1U) {
            if (lane < kRegfedTileN) {
                for (std::uint32_t column = 0U; column < m; ++column) {
                    float sum = 0.0F;
                    for (std::uint32_t s = 0U; s < split; ++s) {
                        sum += partials[(static_cast<std::size_t>(n_tile) *
                                             split + s) * kRegfedTileN * m +
                                        lane * m + column];
                    }
                    // The per-tensor divisor, once, on the completed FP32 sum.
                    output[static_cast<std::size_t>(column) * rows +
                           n_tile * kRegfedTileN + lane] = sum / global_scale;
                }
            }
            if (lane == 0U) counters[n_tile] = 0U;
        }
    }
}

template <std::uint32_t kColBlocks, std::uint32_t kTerms = 1U>
__global__ __launch_bounds__(128) void regfed_nvfp4_matmul_kernel(
    float* __restrict__ output, const std::uint32_t* __restrict__ codes,
    const unsigned char* __restrict__ scales, float global_scale,
    const uint2* __restrict__ activations, std::uint32_t columns,
    std::uint32_t rows, std::uint32_t split, std::uint32_t m,
    std::uint32_t groups_per_block, float* __restrict__ partials,
    std::uint32_t* __restrict__ counters) {
    regfed_nvfp4_matmul_body<kColBlocks, kTerms>(
        output, codes, scales, global_scale, activations, columns, rows, split,
        m, groups_per_block, partials, counters);
}

// One routed expert of a grouped dispatch. Partials and counters are per
// expert because blocks of different experts are resident at the same time and
// the split-K fold is a counter handshake, so sharing either is a race.
struct RegfedNvfp4Slice {
    const std::uint32_t* codes;
    const unsigned char* scales;
    const uint2* activations;
    float* output;
    float* partials;
    std::uint32_t* counters;
    float global_scale;
};

// The grouped form, on the same shape as
// `glm53_shared_expert_nvfp4_dot_grouped_kernel`: `blockIdx.y` selects the
// expert and one launch covers every expert of a page.
//
// WHY THIS EXISTS. Experiment 0247 measured the single-expert launch at 337
// GB/s against 0148's >=632 gate, and found the split-K sweep monotone all the
// way to split 16 -- where split-K partial traffic is 45% of the useful weight
// bytes and still wins. A kernel that gains from paying 45% overhead is starved
// of parallelism, not bandwidth-bound: one 2048x4096 expert is 128 N-tiles, so
// a single-expert launch offers 32 blocks at split 1 on 82 SMs. 0148's accepted
// 704-750 GB/s was measured at 32 experts per launch, and this is that shape.
template <std::uint32_t kColBlocks, std::uint32_t kTerms = 1U>
__global__ __launch_bounds__(128) void regfed_nvfp4_grouped_matmul_kernel(
    const RegfedNvfp4Slice* __restrict__ slices, std::uint32_t columns,
    std::uint32_t rows, std::uint32_t split, std::uint32_t m,
    std::uint32_t groups_per_block) {
    const RegfedNvfp4Slice slice = slices[blockIdx.y];
    regfed_nvfp4_matmul_body<kColBlocks, kTerms>(
        slice.output, slice.codes, slice.scales, slice.global_scale,
        slice.activations, columns, rows, split, m, groups_per_block,
        slice.partials, slice.counters);
}

template <std::uint32_t kColBlocks>
__global__ __launch_bounds__(128) void regfed_fp8_matmul_kernel(
    float* __restrict__ output, const uint4* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const uint2* __restrict__ activations, std::uint32_t columns,
    std::uint32_t rows, std::uint32_t scale_columns, std::uint32_t split,
    std::uint32_t m, std::uint32_t groups_per_block,
    float* __restrict__ partials, std::uint32_t* __restrict__ counters) {
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint32_t n_tiles = rows / kRegfedTileN;
    const std::uint32_t pairs = columns / 32U;
    const std::uint32_t pairs_per_slice = pairs / split;
    const std::uint32_t group = lane >> 2U;
    const std::uint32_t thread = lane & 3U;
    __shared__ std::uint32_t arrived[kRegfedWarpsPerBlock];

    bool live[kColBlocks];
    std::size_t activation_offset[kColBlocks];
#pragma unroll
    for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
        live[c] = group < groups_per_block && c * kRegfedTileM + group < m;
        activation_offset[c] =
            (static_cast<std::size_t>(c) * groups_per_block + group) * 4U + thread;
    }

    for (std::uint32_t work = blockIdx.x * kRegfedWarpsPerBlock + warp;
         work < n_tiles * split; work += gridDim.x * kRegfedWarpsPerBlock) {
        const std::uint32_t n_tile = work / split;
        const std::uint32_t slice = work % split;
        float acc[kColBlocks][4];
#pragma unroll
        for (std::uint32_t c = 0U; c < kColBlocks; ++c)
#pragma unroll
            for (std::uint32_t i = 0U; i < 4U; ++i) acc[c][i] = 0.0F;

        for (std::uint32_t pair = slice * pairs_per_slice;
             pair < (slice + 1U) * pairs_per_slice; ++pair) {
            const std::uint32_t k_tile = pair * 2U;
            // Block-128 scales are indexed by row block and column block, so
            // they need no permutation and stay in canonical order.
            const std::uint32_t factor =
                ((static_cast<std::uint32_t>(
                      scales[(n_tile / 8U) * scale_columns + k_tile / 8U]) +
                  120U) << 7U) * 0x0001'0001U;
            const uint4 packed =
                codes[(static_cast<std::size_t>(n_tile) * pairs + pair) * 32U +
                      lane];
            const std::uint32_t word[4] = {packed.x, packed.y, packed.z,
                                           packed.w};
#pragma unroll
            for (std::uint32_t half = 0U; half < 2U; ++half) {
                const std::uint32_t low = word[half * 2U];
                const std::uint32_t high = word[half * 2U + 1U];
                const std::uint32_t a0 =
                    dsv4_fp8_decode_pair(low & 0xFFFFU, factor);
                const std::uint32_t a1 =
                    dsv4_fp8_decode_pair(low >> 16U, factor);
                const std::uint32_t a2 =
                    dsv4_fp8_decode_pair(high & 0xFFFFU, factor);
                const std::uint32_t a3 =
                    dsv4_fp8_decode_pair(high >> 16U, factor);
                const std::size_t tile_base =
                    (static_cast<std::size_t>(k_tile) + half) * kColBlocks *
                    groups_per_block * 4U;
#pragma unroll
                for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
                    const uint2 b =
                        live[c] ? activations[tile_base + activation_offset[c]]
                                : make_uint2(0U, 0U);
                    dsv4_mma_m16n8k16(acc[c][0], acc[c][1], acc[c][2], acc[c][3],
                                      a0, a1, a2, a3, b.x, b.y);
                }
            }
        }

        float* slot = partials + (static_cast<std::size_t>(work)) *
                                     kRegfedTileN * kRegfedMaxM;
#pragma unroll
        for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
#pragma unroll
            for (std::uint32_t i = 0U; i < 4U; ++i) {
                const std::uint32_t row = group + ((i >= 2U) ? 8U : 0U);
                const std::uint32_t column =
                    c * kRegfedTileM + thread * 2U + (i & 1U);
                if (column < m) slot[row * m + column] = acc[c][i];
            }
        }

        __threadfence();
        __syncwarp();
        if (lane == 0U) {
            arrived[warp] = atomicAdd(&counters[n_tile], 1U);
        }
        __syncwarp();
        if (arrived[warp] == split - 1U) {
            if (lane < kRegfedTileN) {
                for (std::uint32_t column = 0U; column < m; ++column) {
                    float sum = 0.0F;
                    for (std::uint32_t s = 0U; s < split; ++s) {
                        sum += partials[(static_cast<std::size_t>(n_tile) *
                                             split + s) * kRegfedTileN *
                                            kRegfedMaxM + lane * m + column];
                    }
                    output[static_cast<std::size_t>(column) * rows +
                           n_tile * kRegfedTileN + lane] = sum;
                }
            }
            if (lane == 0U) counters[n_tile] = 0U;
        }
    }
}

// Register-fed GLM W8A8. Weight and activation codes remain one byte through
// HBM and are decoded directly into MMA registers. Each K128 dot is completed
// before its two arbitrary F32 scales are applied, preserving continuous
// dynamic activation scaling without widening scaled operands to BF16.
template <std::uint32_t kColBlocks>
__global__ __launch_bounds__(128) void regfed_fp8_f32_matmul_kernel(
    float* __restrict__ output, const uint4* __restrict__ codes,
    const float* __restrict__ weight_scales,
    const uint2* __restrict__ activations,
    const float* __restrict__ activation_scales, std::uint32_t columns,
    std::uint32_t rows, std::uint32_t scale_columns, std::uint32_t split,
    std::uint32_t m, std::uint32_t groups_per_block,
    float* __restrict__ partials, std::uint32_t* __restrict__ counters) {
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint32_t n_tiles = rows / kRegfedTileN;
    const std::uint32_t pairs = columns / 32U;
    const std::uint32_t pairs_per_slice = pairs / split;
    const std::uint32_t group = lane >> 2U;
    const std::uint32_t thread = lane & 3U;
    constexpr std::uint32_t unit_factor =
        ((127U + 120U) << 7U) * 0x0001'0001U;
    __shared__ std::uint32_t arrived[kRegfedWarpsPerBlock];

    bool live[kColBlocks];
    std::size_t activation_offset[kColBlocks];
#pragma unroll
    for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
        live[c] = group < groups_per_block &&
                  c * kRegfedTileM + group < m;
        activation_offset[c] =
            (static_cast<std::size_t>(c) * groups_per_block + group) * 4U +
            thread;
    }

    for (std::uint32_t work = blockIdx.x * kRegfedWarpsPerBlock + warp;
         work < n_tiles * split; work += gridDim.x * kRegfedWarpsPerBlock) {
        const std::uint32_t n_tile = work / split;
        const std::uint32_t slice = work % split;
        float totals_for_slice[kColBlocks][4]{};
        const std::uint32_t begin = slice * pairs_per_slice;
        const std::uint32_t end = begin + pairs_per_slice;
        for (std::uint32_t pair_block = begin; pair_block < end;
             pair_block += 4U) {
            float block_acc[kColBlocks][4]{};
#pragma unroll
            for (std::uint32_t within = 0U; within < 4U; ++within) {
                const std::uint32_t pair = pair_block + within;
                const std::uint32_t k_tile = pair * 2U;
                const uint4 packed = codes[
                    (static_cast<std::size_t>(n_tile) * pairs + pair) * 32U +
                    lane];
                const std::uint32_t word[4] = {packed.x, packed.y, packed.z,
                                               packed.w};
#pragma unroll
                for (std::uint32_t half = 0U; half < 2U; ++half) {
                    const std::uint32_t low = word[half * 2U];
                    const std::uint32_t high = word[half * 2U + 1U];
                    const std::uint32_t a0 =
                        dsv4_fp8_decode_pair(low & 0xFFFFU, unit_factor);
                    const std::uint32_t a1 =
                        dsv4_fp8_decode_pair(low >> 16U, unit_factor);
                    const std::uint32_t a2 =
                        dsv4_fp8_decode_pair(high & 0xFFFFU, unit_factor);
                    const std::uint32_t a3 =
                        dsv4_fp8_decode_pair(high >> 16U, unit_factor);
                    const std::size_t tile_base =
                        (static_cast<std::size_t>(k_tile) + half) *
                        kColBlocks * groups_per_block * 4U;
#pragma unroll
                    for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
                        const uint2 b = live[c]
                            ? activations[tile_base + activation_offset[c]]
                            : make_uint2(0U, 0U);
                        dsv4_mma_m16n8k16(
                            block_acc[c][0], block_acc[c][1],
                            block_acc[c][2], block_acc[c][3], a0, a1, a2, a3,
                            b.x, b.y);
                    }
                }
            }
            const std::uint32_t scale_column = pair_block / 4U;
            const float weight_scale = weight_scales[
                static_cast<std::size_t>(n_tile / 8U) * scale_columns +
                scale_column];
#pragma unroll
            for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
#pragma unroll
                for (std::uint32_t i = 0U; i < 4U; ++i) {
                    const std::uint32_t input_row =
                        c * kRegfedTileM + thread * 2U + (i & 1U);
                    if (input_row < m) {
                        totals_for_slice[c][i] += block_acc[c][i] *
                            activation_scales[
                                static_cast<std::size_t>(input_row) *
                                    scale_columns + scale_column] *
                            weight_scale;
                    }
                }
            }
        }

        float* slot = partials + static_cast<std::size_t>(work) *
                                     kRegfedTileN * kRegfedMaxM;
#pragma unroll
        for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
#pragma unroll
            for (std::uint32_t i = 0U; i < 4U; ++i) {
                const std::uint32_t row = group + ((i >= 2U) ? 8U : 0U);
                const std::uint32_t column =
                    c * kRegfedTileM + thread * 2U + (i & 1U);
                if (column < m) slot[row * m + column] = totals_for_slice[c][i];
            }
        }

        __threadfence();
        __syncwarp();
        if (lane == 0U) arrived[warp] = atomicAdd(&counters[n_tile], 1U);
        __syncwarp();
        if (arrived[warp] == split - 1U) {
            if (lane < kRegfedTileN) {
                for (std::uint32_t column = 0U; column < m; ++column) {
                    float sum = 0.0F;
                    for (std::uint32_t s = 0U; s < split; ++s) {
                        sum += partials[(static_cast<std::size_t>(n_tile) *
                                             split + s) * kRegfedTileN *
                                            kRegfedMaxM + lane * m + column];
                    }
                    output[static_cast<std::size_t>(column) * rows +
                           n_tile * kRegfedTileN + lane] = sum;
                }
            }
            if (lane == 0U) counters[n_tile] = 0U;
        }
    }
}

// Shape admission for the register-fed routes. Stated once, used by both the
// load-time prepack and the dispatch, so a weight can never be prepacked into a
// layout the kernel will not read.
[[nodiscard]] inline bool regfed_fp4_shape_admissible(
    std::uint64_t rows, std::uint64_t columns) noexcept {
    return rows % kRegfedTileN == 0U &&
           columns % (kRegfedTileK * kRegfedKPerLoad) == 0U &&
           columns % kRegfedGroup == 0U && rows >= kRegfedTileN &&
           columns >= kRegfedTileK * kRegfedKPerLoad;
}

// NVFP4 admits strictly more shapes than MXFP4: a group of 16 is exactly one
// K-tile, so the group constraint is implied by the K-block one and only the
// tile extents remain. Both of GLM's routed expert shapes -- 2048x4096 for
// gate and up, 4096x2048 for down -- satisfy it.
[[nodiscard]] inline bool regfed_nvfp4_shape_admissible(
    std::uint64_t rows, std::uint64_t columns) noexcept {
    return rows % kRegfedTileN == 0U &&
           columns % (kRegfedTileK * kRegfedKPerLoad) == 0U &&
           rows >= kRegfedTileN &&
           columns >= kRegfedTileK * kRegfedKPerLoad;
}

[[nodiscard]] inline bool regfed_fp8_shape_admissible(
    std::uint64_t rows, std::uint64_t columns) noexcept {
    // Block-128 scales are shared by sixteen-row tiles and by eight K-tiles, so
    // both extents must be whole multiples of 128 for one code to cover a tile.
    return rows % 128U == 0U && columns % 128U == 0U;
}

// Splits K so the tail of the machine stays busy without inflating partial
// traffic: every slice must be a whole number of load blocks, and the warp
// count is grown only while the grid is still short of the device.
[[nodiscard]] inline std::uint32_t regfed_split_k(
    std::uint32_t units, std::uint32_t n_tiles) noexcept {
    std::uint32_t split = 1U;
    while (split < 16U && units % (split * 2U) == 0U &&
           n_tiles * split * 2U <= 4096U) {
        split *= 2U;
    }
    return split;
}

// A/B switch for the register-fed routes. Default on: the campaign's gates were
// measured on these kernels and the scalar routes are the incumbent, so the
// interesting arm is the one that runs by default and the control is the one
// that has to be asked for.
std::atomic<int> g_regfed_matmul_enabled{-1};

[[nodiscard]] bool regfed_matmul_enabled() noexcept {
    auto current = g_regfed_matmul_enabled.load(std::memory_order_relaxed);
    if (current < 0) {
        const char* value = std::getenv("STRATA_REGFED_MATMUL");
        current = (value == nullptr || (value[0] != '0' && value[0] != 'n' &&
                                        value[0] != 'N'))
                      ? 1
                      : 0;
        g_regfed_matmul_enabled.store(current, std::memory_order_relaxed);
    }
    return current != 0;
}

// ---------------------------------------------------------------------------
// Register-fed fused MXFP4 MoE (campaign task A).
//
// The scalar mxfp4_moe_gate_up/down kernels above reach 15.3% and 9.9% of this
// card's measured read roofline at Laguna's real dispatch width, and hold that
// figure from width 8 to 64 -- bandwidth-bound, not dispatch-bound (experiment
// 0168). The register-fed decode collects 5.04x and 6.30x of that headroom at
// the same width by keeping the codes four bits through HBM and decoding
// straight into MMA operand registers.
//
// enqueue_moe already rounds both activations to E4M3 -- the hidden vector
// before gate/up and the SwiGLU output before down -- and an E4M3 value has
// three mantissa bits, so its BF16 image is exact. E2M1 codes and power-of-two
// E8M0 scales are exact in BF16 too, so these kernels multiply the same real
// numbers the scalar kernels multiply and differ only in accumulation order.
// ---------------------------------------------------------------------------

// Activation permute into MMA B-fragment order, batched over experts. gate/up
// share one hidden vector, so they pass experts = 1; down reads a distinct
// activation block per expert.
__global__ void regfed_moe_activation_fragment_kernel(
    uint2* __restrict__ destination, const float* __restrict__ source,
    std::uint32_t experts, std::uint32_t m, std::uint32_t columns,
    std::uint32_t column_blocks, std::uint32_t groups_per_block) {
    const std::uint32_t k_tiles = columns / kRegfedTileK;
    const std::uint32_t per_expert = k_tiles * column_blocks * groups_per_block * 4U;
    const std::uint64_t total = static_cast<std::uint64_t>(experts) * per_expert;
    for (std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += gridDim.x * blockDim.x) {
        const std::uint32_t local = static_cast<std::uint32_t>(index % per_expert);
        const std::uint32_t expert = static_cast<std::uint32_t>(index / per_expert);
        const std::uint32_t thread = local % 4U;
        const std::uint32_t group = (local / 4U) % groups_per_block;
        const std::uint32_t block = (local / (4U * groups_per_block)) % column_blocks;
        const std::uint32_t k_tile = local / (4U * groups_per_block * column_blocks);
        const std::uint32_t column = block * kRegfedTileM + group;
        std::uint32_t b0 = 0U;
        std::uint32_t b1 = 0U;
        if (column < m) {
            const float* row =
                source + (static_cast<std::size_t>(expert) * m + column) * columns;
            const auto bits = [&](std::uint32_t offset) {
                return static_cast<std::uint32_t>(__bfloat16_as_ushort(
                    __float2bfloat16_rn(row[k_tile * kRegfedTileK + offset])));
            };
            b0 = bits(thread * 2U) | (bits(thread * 2U + 1U) << 16U);
            b1 = bits(thread * 2U + 8U) | (bits(thread * 2U + 9U) << 16U);
        }
        destination[index] = make_uint2(b0, b1);
    }
}

__global__ void regfed_fp8_moe_activation_fragment_kernel(
    uint2* __restrict__ destination,
    const unsigned char* __restrict__ source, std::uint32_t experts,
    std::uint32_t m, std::uint32_t columns, std::uint32_t column_blocks,
    std::uint32_t groups_per_block) {
    const std::uint32_t k_tiles = columns / kRegfedTileK;
    const std::uint32_t per_expert =
        k_tiles * column_blocks * groups_per_block * 4U;
    const std::uint64_t total =
        static_cast<std::uint64_t>(experts) * per_expert;
    constexpr std::uint32_t unit_factor =
        ((127U + 120U) << 7U) * 0x0001'0001U;
    for (std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += gridDim.x * blockDim.x) {
        const auto local = static_cast<std::uint32_t>(index % per_expert);
        const auto expert = static_cast<std::uint32_t>(index / per_expert);
        const std::uint32_t thread = local % 4U;
        const std::uint32_t group = (local / 4U) % groups_per_block;
        const std::uint32_t block =
            (local / (4U * groups_per_block)) % column_blocks;
        const std::uint32_t k_tile =
            local / (4U * groups_per_block * column_blocks);
        const std::uint32_t column = block * kRegfedTileM + group;
        uint2 value = make_uint2(0U, 0U);
        if (column < m) {
            const auto* row = source +
                (static_cast<std::size_t>(expert) * m + column) * columns +
                k_tile * kRegfedTileK;
            const std::uint32_t first =
                static_cast<std::uint32_t>(row[thread * 2U]) |
                (static_cast<std::uint32_t>(row[thread * 2U + 1U]) << 8U);
            const std::uint32_t second =
                static_cast<std::uint32_t>(row[thread * 2U + 8U]) |
                (static_cast<std::uint32_t>(row[thread * 2U + 9U]) << 8U);
            value = make_uint2(dsv4_fp8_decode_pair(first, unit_factor),
                               dsv4_fp8_decode_pair(second, unit_factor));
        }
        destination[index] = value;
    }
}

template <std::uint32_t kColBlocks>
__global__ __launch_bounds__(128) void regfed_fp8_f32_moe_gate_up_kernel(
    float* __restrict__ gate_partials, float* __restrict__ up_partials,
    const uint2* __restrict__ activations,
    const float* __restrict__ activation_scales, Fp8F32MoeBatch batch,
    std::uint32_t columns, std::uint32_t intermediate, std::uint32_t split,
    std::uint32_t m, std::uint32_t groups_per_block) {
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint32_t n_tiles = intermediate / kRegfedTileN;
    const std::uint32_t pairs = columns / 32U;
    const std::uint32_t pairs_per_slice = pairs / split;
    const std::uint32_t scale_columns = columns / 128U;
    const std::uint32_t group = lane >> 2U;
    const std::uint32_t thread = lane & 3U;
    const std::uint32_t total = batch.count * n_tiles * split;
    constexpr std::uint32_t unit_factor =
        ((127U + 120U) << 7U) * 0x0001'0001U;
    bool live[kColBlocks];
    std::size_t offset[kColBlocks];
#pragma unroll
    for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
        live[c] = group < groups_per_block &&
                  c * kRegfedTileM + group < m;
        offset[c] =
            (static_cast<std::size_t>(c) * groups_per_block + group) * 4U +
            thread;
    }

    for (std::uint32_t work = blockIdx.x * kRegfedWarpsPerBlock + warp;
         work < total; work += gridDim.x * kRegfedWarpsPerBlock) {
        const std::uint32_t slice = work % split;
        const std::uint32_t flat = work / split;
        const std::uint32_t n_tile = flat % n_tiles;
        const std::uint32_t expert = flat / n_tiles;
        const auto* gate4 =
            reinterpret_cast<const uint4*>(batch.gate_weights[expert]);
        const auto* up4 =
            reinterpret_cast<const uint4*>(batch.up_weights[expert]);
        const auto* gate_scales = batch.gate_scales[expert];
        const auto* up_scales = batch.up_scales[expert];
        float gate_totals[kColBlocks][4]{};
        float up_totals[kColBlocks][4]{};
        const std::uint32_t begin = slice * pairs_per_slice;
        const std::uint32_t end = begin + pairs_per_slice;
        for (std::uint32_t pair_block = begin; pair_block < end;
             pair_block += 4U) {
            float gate_block[kColBlocks][4]{};
            float up_block[kColBlocks][4]{};
#pragma unroll
            for (std::uint32_t within = 0U; within < 4U; ++within) {
                const std::uint32_t pair = pair_block + within;
                const std::size_t code_index =
                    (static_cast<std::size_t>(n_tile) * pairs + pair) * 32U +
                    lane;
                const uint4 gate_packed = gate4[code_index];
                const uint4 up_packed = up4[code_index];
                const std::uint32_t gate_words[4] = {
                    gate_packed.x, gate_packed.y, gate_packed.z, gate_packed.w};
                const std::uint32_t up_words[4] = {
                    up_packed.x, up_packed.y, up_packed.z, up_packed.w};
#pragma unroll
                for (std::uint32_t half = 0U; half < 2U; ++half) {
                    const auto decode = [&](const std::uint32_t* words,
                                            std::uint32_t (&decoded)[4]) {
                        const auto low = words[half * 2U];
                        const auto high = words[half * 2U + 1U];
                        decoded[0] = dsv4_fp8_decode_pair(
                            low & 0xFFFFU, unit_factor);
                        decoded[1] = dsv4_fp8_decode_pair(
                            low >> 16U, unit_factor);
                        decoded[2] = dsv4_fp8_decode_pair(
                            high & 0xFFFFU, unit_factor);
                        decoded[3] = dsv4_fp8_decode_pair(
                            high >> 16U, unit_factor);
                    };
                    std::uint32_t gate_a[4];
                    std::uint32_t up_a[4];
                    decode(gate_words, gate_a);
                    decode(up_words, up_a);
                    const std::size_t base =
                        (static_cast<std::size_t>(pair * 2U) + half) *
                        kColBlocks * groups_per_block * 4U;
#pragma unroll
                    for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
                        const uint2 b = live[c]
                            ? activations[base + offset[c]]
                            : make_uint2(0U, 0U);
                        dsv4_mma_m16n8k16(
                            gate_block[c][0], gate_block[c][1],
                            gate_block[c][2], gate_block[c][3], gate_a[0],
                            gate_a[1], gate_a[2], gate_a[3], b.x, b.y);
                        dsv4_mma_m16n8k16(
                            up_block[c][0], up_block[c][1], up_block[c][2],
                            up_block[c][3], up_a[0], up_a[1], up_a[2], up_a[3],
                            b.x, b.y);
                    }
                }
            }
            const std::uint32_t scale_column = pair_block / 4U;
            const auto weight_scale_index =
                static_cast<std::size_t>(n_tile / 8U) * scale_columns +
                scale_column;
#pragma unroll
            for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
#pragma unroll
                for (std::uint32_t i = 0U; i < 4U; ++i) {
                    const std::uint32_t input_row =
                        c * kRegfedTileM + thread * 2U + (i & 1U);
                    if (input_row < m) {
                        const float activation_scale = activation_scales[
                            static_cast<std::size_t>(input_row) *
                                scale_columns + scale_column];
                        gate_totals[c][i] += gate_block[c][i] *
                            activation_scale *
                            gate_scales[weight_scale_index];
                        up_totals[c][i] += up_block[c][i] * activation_scale *
                            up_scales[weight_scale_index];
                    }
                }
            }
        }
#pragma unroll
        for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
#pragma unroll
            for (std::uint32_t i = 0U; i < 4U; ++i) {
                const std::uint32_t row = group + ((i >= 2U) ? 8U : 0U);
                const std::uint32_t column =
                    c * kRegfedTileM + thread * 2U + (i & 1U);
                if (column >= m) continue;
                const std::size_t slot =
                    ((static_cast<std::size_t>(expert) * intermediate +
                      n_tile * kRegfedTileN + row) * m + column) * split +
                    slice;
                gate_partials[slot] = gate_totals[c][i];
                up_partials[slot] = up_totals[c][i];
            }
        }
    }
}

__global__ void regfed_fp8_f32_moe_swiglu_kernel(
    float* __restrict__ activations, const float* __restrict__ gate_partials,
    const float* __restrict__ up_partials, std::uint32_t experts,
    std::uint32_t intermediate, std::uint32_t m, std::uint32_t split,
    float swiglu_limit, unsigned int* __restrict__ error_flag) {
    const std::uint64_t total =
        static_cast<std::uint64_t>(experts) * intermediate * m;
    for (std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += gridDim.x * blockDim.x) {
        const std::uint32_t column = static_cast<std::uint32_t>(index % m);
        float gate = 0.0F;
        float up = 0.0F;
        for (std::uint32_t slice = 0U; slice < split; ++slice) {
            gate += gate_partials[index * split + slice];
            up += up_partials[index * split + slice];
        }
        gate = bf16_round(gate);
        up = bf16_round(up);
        if (!isfinite(gate) || !isfinite(up)) {
            atomicExch(error_flag, 1U);
            continue;
        }
        const float limited_gate = fminf(gate, swiglu_limit);
        const float limited_up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
        const float exponential = limited_gate >= 0.0F
            ? expf(-limited_gate) : expf(limited_gate);
        const float sigmoid = limited_gate >= 0.0F
            ? 1.0F / (1.0F + exponential)
            : exponential / (1.0F + exponential);
        const std::uint32_t rest = static_cast<std::uint32_t>(index / m);
        const std::uint32_t output_row = rest % intermediate;
        const std::uint32_t expert = rest / intermediate;
        activations[(static_cast<std::size_t>(expert) * m + column) *
                        intermediate + output_row] =
            bf16_round(limited_gate * sigmoid * limited_up);
    }
}

template <std::uint32_t kColBlocks>
__global__ __launch_bounds__(128) void regfed_fp8_f32_moe_down_kernel(
    float* __restrict__ partials, const uint2* __restrict__ activations,
    const float* __restrict__ activation_scales, Fp8F32MoeBatch batch,
    std::uint32_t columns, std::uint32_t rows, std::uint32_t split,
    std::uint32_t m, std::uint32_t groups_per_block) {
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint32_t n_tiles = rows / kRegfedTileN;
    const std::uint32_t pairs = columns / 32U;
    const std::uint32_t pairs_per_slice = pairs / split;
    const std::uint32_t scale_columns = columns / 128U;
    const std::uint32_t group = lane >> 2U;
    const std::uint32_t thread = lane & 3U;
    const std::uint32_t total = batch.count * n_tiles * split;
    const std::size_t per_expert_activation =
        static_cast<std::size_t>(columns / kRegfedTileK) * kColBlocks *
        groups_per_block * 4U;
    constexpr std::uint32_t unit_factor =
        ((127U + 120U) << 7U) * 0x0001'0001U;
    bool live[kColBlocks];
    std::size_t offset[kColBlocks];
#pragma unroll
    for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
        live[c] = group < groups_per_block &&
                  c * kRegfedTileM + group < m;
        offset[c] =
            (static_cast<std::size_t>(c) * groups_per_block + group) * 4U +
            thread;
    }
    for (std::uint32_t work = blockIdx.x * kRegfedWarpsPerBlock + warp;
         work < total; work += gridDim.x * kRegfedWarpsPerBlock) {
        const std::uint32_t slice = work % split;
        const std::uint32_t flat = work / split;
        const std::uint32_t n_tile = flat % n_tiles;
        const std::uint32_t expert = flat / n_tiles;
        const auto* codes4 =
            reinterpret_cast<const uint4*>(batch.down_weights[expert]);
        const auto* weight_scales = batch.down_scales[expert];
        float totals_for_slice[kColBlocks][4]{};
        const std::uint32_t begin = slice * pairs_per_slice;
        const std::uint32_t end = begin + pairs_per_slice;
        for (std::uint32_t pair_block = begin; pair_block < end;
             pair_block += 4U) {
            float block_acc[kColBlocks][4]{};
#pragma unroll
            for (std::uint32_t within = 0U; within < 4U; ++within) {
                const std::uint32_t pair = pair_block + within;
                const uint4 packed = codes4[
                    (static_cast<std::size_t>(n_tile) * pairs + pair) * 32U +
                    lane];
                const std::uint32_t words[4] = {
                    packed.x, packed.y, packed.z, packed.w};
#pragma unroll
                for (std::uint32_t half = 0U; half < 2U; ++half) {
                    const auto low = words[half * 2U];
                    const auto high = words[half * 2U + 1U];
                    const std::uint32_t a[4] = {
                        dsv4_fp8_decode_pair(low & 0xFFFFU, unit_factor),
                        dsv4_fp8_decode_pair(low >> 16U, unit_factor),
                        dsv4_fp8_decode_pair(high & 0xFFFFU, unit_factor),
                        dsv4_fp8_decode_pair(high >> 16U, unit_factor)};
                    const std::size_t base =
                        static_cast<std::size_t>(expert) *
                            per_expert_activation +
                        (static_cast<std::size_t>(pair * 2U) + half) *
                            kColBlocks * groups_per_block * 4U;
#pragma unroll
                    for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
                        const uint2 b = live[c]
                            ? activations[base + offset[c]]
                            : make_uint2(0U, 0U);
                        dsv4_mma_m16n8k16(
                            block_acc[c][0], block_acc[c][1],
                            block_acc[c][2], block_acc[c][3], a[0], a[1], a[2],
                            a[3], b.x, b.y);
                    }
                }
            }
            const std::uint32_t scale_column = pair_block / 4U;
            const float weight_scale = weight_scales[
                static_cast<std::size_t>(n_tile / 8U) * scale_columns +
                scale_column];
#pragma unroll
            for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
#pragma unroll
                for (std::uint32_t i = 0U; i < 4U; ++i) {
                    const std::uint32_t input_row =
                        c * kRegfedTileM + thread * 2U + (i & 1U);
                    if (input_row < m) {
                        totals_for_slice[c][i] += block_acc[c][i] *
                            activation_scales[
                                (static_cast<std::size_t>(expert) * m +
                                 input_row) * scale_columns + scale_column] *
                            weight_scale;
                    }
                }
            }
        }
#pragma unroll
        for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
#pragma unroll
            for (std::uint32_t i = 0U; i < 4U; ++i) {
                const std::uint32_t row = group + ((i >= 2U) ? 8U : 0U);
                const std::uint32_t column =
                    c * kRegfedTileM + thread * 2U + (i & 1U);
                if (column >= m) continue;
                const std::size_t slot =
                    ((static_cast<std::size_t>(expert) * rows +
                      n_tile * kRegfedTileN + row) * m + column) * split +
                    slice;
                partials[slot] = totals_for_slice[c][i];
            }
        }
    }
}

__global__ void regfed_fp8_f32_moe_reduce_kernel(
    float* __restrict__ output, const float* __restrict__ partials,
    std::uint32_t experts, std::uint32_t rows, std::uint32_t m,
    std::uint32_t split, unsigned int* __restrict__ error_flag) {
    const std::uint64_t total = static_cast<std::uint64_t>(experts) * rows * m;
    for (std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += gridDim.x * blockDim.x) {
        const std::uint32_t column = static_cast<std::uint32_t>(index % m);
        const std::uint32_t rest = static_cast<std::uint32_t>(index / m);
        const std::uint32_t output_row = rest % rows;
        const std::uint32_t expert = rest / rows;
        float sum = 0.0F;
        for (std::uint32_t slice = 0U; slice < split; ++slice) {
            sum += partials[index * split + slice];
        }
        if (!isfinite(sum)) atomicExch(error_flag, 1U);
        output[(static_cast<std::size_t>(expert) * m + column) * rows +
               output_row] = bf16_round(sum);
    }
}

// One warp owns one (expert, N-tile, K-slice). Gate and up share the activation
// fragment, so a single pass over the hidden vector feeds both weight streams.
template <std::uint32_t kColBlocks>
__global__ __launch_bounds__(128) void regfed_mxfp4_moe_gate_up_kernel(
    float* __restrict__ gate_partials, float* __restrict__ up_partials,
    const uint2* __restrict__ activations, Mxfp4MoeBatch batch,
    std::uint32_t columns, std::uint32_t intermediate, std::uint32_t split,
    std::uint32_t m, std::uint32_t groups_per_block) {
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint32_t n_tiles = intermediate / kRegfedTileN;
    const std::uint32_t k_blocks = (columns / kRegfedTileK) / kRegfedKPerLoad;
    const std::uint32_t per_slice = k_blocks / split;
    const std::uint32_t scale_columns = columns / kRegfedGroup;
    const std::uint32_t group = lane >> 2U;
    const std::uint32_t thread = lane & 3U;
    const std::uint32_t shift = (group & 3U) * 8U;
    const std::uint32_t total = batch.count * n_tiles * split;

    bool live[kColBlocks];
    std::size_t offset[kColBlocks];
#pragma unroll
    for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
        live[c] = group < groups_per_block && c * kRegfedTileM + group < m;
        offset[c] = (static_cast<std::size_t>(c) * groups_per_block + group) * 4U + thread;
    }

    for (std::uint32_t work = blockIdx.x * kRegfedWarpsPerBlock + warp;
         work < total; work += gridDim.x * kRegfedWarpsPerBlock) {
        const std::uint32_t slice = work % split;
        const std::uint32_t flat = work / split;
        const std::uint32_t n_tile = flat % n_tiles;
        const std::uint32_t expert = flat / n_tiles;
        const auto* gate4 = reinterpret_cast<const uint4*>(batch.gate_weights[expert]);
        const auto* up4 = reinterpret_cast<const uint4*>(batch.up_weights[expert]);
        const unsigned char* gs = batch.gate_scales[expert];
        const unsigned char* us = batch.up_scales[expert];
        float gate[kColBlocks][4]{};
        float up[kColBlocks][4]{};

        for (std::uint32_t block = slice * per_slice;
             block < (slice + 1U) * per_slice; ++block) {
            const std::size_t index =
                (static_cast<std::size_t>(n_tile) * k_blocks + block) * kRegfedWarp + lane;
            const uint4 gpacked = gate4[index];
            const uint4 upacked = up4[index];
            const std::size_t sbase =
                (static_cast<std::size_t>(n_tile) * scale_columns + block * 2U) *
                kRegfedTileN;
            const uint4 g_even = *reinterpret_cast<const uint4*>(gs + sbase);
            const uint4 g_odd = *reinterpret_cast<const uint4*>(gs + sbase + kRegfedTileN);
            const uint4 u_even = *reinterpret_cast<const uint4*>(us + sbase);
            const uint4 u_odd = *reinterpret_cast<const uint4*>(us + sbase + kRegfedTileN);
            const std::uint32_t gw[4] = {gpacked.x, gpacked.y, gpacked.z, gpacked.w};
            const std::uint32_t uw[4] = {upacked.x, upacked.y, upacked.z, upacked.w};
#pragma unroll
            for (std::uint32_t j = 0U; j < kRegfedKPerLoad; ++j) {
                const uint4 gsel = (j < 2U) ? g_even : g_odd;
                const uint4 usel = (j < 2U) ? u_even : u_odd;
                std::uint32_t ga[4];
                std::uint32_t ua[4];
                regfed_fp4_decode_fragment(
                    gw[j],
                    regfed_fp4_scale_pair((((group < 4U) ? gsel.x : gsel.y) >> shift) & 0xFFU),
                    regfed_fp4_scale_pair((((group < 4U) ? gsel.z : gsel.w) >> shift) & 0xFFU),
                    ga);
                regfed_fp4_decode_fragment(
                    uw[j],
                    regfed_fp4_scale_pair((((group < 4U) ? usel.x : usel.y) >> shift) & 0xFFU),
                    regfed_fp4_scale_pair((((group < 4U) ? usel.z : usel.w) >> shift) & 0xFFU),
                    ua);
                const std::size_t base =
                    (static_cast<std::size_t>(block) * kRegfedKPerLoad + j) *
                    kColBlocks * groups_per_block * 4U;
#pragma unroll
                for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
                    const uint2 b = live[c] ? activations[base + offset[c]]
                                            : make_uint2(0U, 0U);
                    dsv4_mma_m16n8k16(gate[c][0], gate[c][1], gate[c][2], gate[c][3],
                                      ga[0], ga[1], ga[2], ga[3], b.x, b.y);
                    dsv4_mma_m16n8k16(up[c][0], up[c][1], up[c][2], up[c][3],
                                      ua[0], ua[1], ua[2], ua[3], b.x, b.y);
                }
            }
        }
#pragma unroll
        for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
#pragma unroll
            for (std::uint32_t i = 0U; i < 4U; ++i) {
                const std::uint32_t row = group + ((i >= 2U) ? 8U : 0U);
                const std::uint32_t column = c * kRegfedTileM + thread * 2U + (i & 1U);
                if (column >= m) continue;
                const std::size_t slot =
                    ((static_cast<std::size_t>(expert) * intermediate +
                      n_tile * kRegfedTileN + row) * m + column) * split + slice;
                gate_partials[slot] = gate[c][i];
                up_partials[slot] = up[c][i];
            }
        }
    }
}

// Sums the split-K partials and applies the SwiGLU. The branchy sigmoid is
// reproduced from mxfp4_moe_gate_up_kernel exactly, so this path is
// indistinguishable from the incumbent rather than merely close.
__global__ void regfed_mxfp4_moe_swiglu_kernel(
    float* __restrict__ activations, const float* __restrict__ gate_partials,
    const float* __restrict__ up_partials, std::uint32_t experts,
    std::uint32_t intermediate, std::uint32_t m, std::uint32_t split,
    unsigned int* __restrict__ error_flag) {
    const std::uint64_t total =
        static_cast<std::uint64_t>(experts) * intermediate * m;
    for (std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += gridDim.x * blockDim.x) {
        const std::uint32_t column = static_cast<std::uint32_t>(index % m);
        const std::uint32_t rest = static_cast<std::uint32_t>(index / m);
        const std::uint32_t output_row = rest % intermediate;
        const std::uint32_t expert = rest / intermediate;
        float gate = 0.0F;
        float up = 0.0F;
        for (std::uint32_t slice = 0U; slice < split; ++slice) {
            gate += gate_partials[index * split + slice];
            up += up_partials[index * split + slice];
        }
        if (!isfinite(gate) || !isfinite(up)) {
            atomicExch(error_flag, 1U);
            continue;
        }
        const float exponential = gate >= 0.0F ? expf(-gate) : expf(gate);
        const float sigmoid = gate >= 0.0F ? 1.0F / (1.0F + exponential)
                                           : exponential / (1.0F + exponential);
        activations[(static_cast<std::size_t>(expert) * m + column) * intermediate +
                    output_row] = gate * sigmoid * up;
    }
}

template <std::uint32_t kColBlocks>
__global__ __launch_bounds__(128) void regfed_mxfp4_moe_down_kernel(
    float* __restrict__ partials, const uint2* __restrict__ activations,
    Mxfp4MoeBatch batch, std::uint32_t columns, std::uint32_t rows,
    std::uint32_t split, std::uint32_t m, std::uint32_t groups_per_block) {
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint32_t n_tiles = rows / kRegfedTileN;
    const std::uint32_t k_blocks = (columns / kRegfedTileK) / kRegfedKPerLoad;
    const std::uint32_t per_slice = k_blocks / split;
    const std::uint32_t scale_columns = columns / kRegfedGroup;
    const std::uint32_t group = lane >> 2U;
    const std::uint32_t thread = lane & 3U;
    const std::uint32_t shift = (group & 3U) * 8U;
    const std::uint32_t total = batch.count * n_tiles * split;
    const std::size_t per_expert_activation =
        static_cast<std::size_t>(columns / kRegfedTileK) * kColBlocks *
        groups_per_block * 4U;

    bool live[kColBlocks];
    std::size_t offset[kColBlocks];
#pragma unroll
    for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
        live[c] = group < groups_per_block && c * kRegfedTileM + group < m;
        offset[c] = (static_cast<std::size_t>(c) * groups_per_block + group) * 4U + thread;
    }

    for (std::uint32_t work = blockIdx.x * kRegfedWarpsPerBlock + warp;
         work < total; work += gridDim.x * kRegfedWarpsPerBlock) {
        const std::uint32_t slice = work % split;
        const std::uint32_t flat = work / split;
        const std::uint32_t n_tile = flat % n_tiles;
        const std::uint32_t expert = flat / n_tiles;
        const auto* codes4 = reinterpret_cast<const uint4*>(batch.down_weights[expert]);
        const unsigned char* sc = batch.down_scales[expert];
        float acc[kColBlocks][4]{};

        for (std::uint32_t block = slice * per_slice;
             block < (slice + 1U) * per_slice; ++block) {
            const uint4 packed =
                codes4[(static_cast<std::size_t>(n_tile) * k_blocks + block) *
                           kRegfedWarp + lane];
            const std::size_t sbase =
                (static_cast<std::size_t>(n_tile) * scale_columns + block * 2U) *
                kRegfedTileN;
            const uint4 even = *reinterpret_cast<const uint4*>(sc + sbase);
            const uint4 odd = *reinterpret_cast<const uint4*>(sc + sbase + kRegfedTileN);
            const std::uint32_t w[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
            for (std::uint32_t j = 0U; j < kRegfedKPerLoad; ++j) {
                const uint4 sel = (j < 2U) ? even : odd;
                std::uint32_t a[4];
                regfed_fp4_decode_fragment(
                    w[j],
                    regfed_fp4_scale_pair((((group < 4U) ? sel.x : sel.y) >> shift) & 0xFFU),
                    regfed_fp4_scale_pair((((group < 4U) ? sel.z : sel.w) >> shift) & 0xFFU),
                    a);
                const std::size_t base =
                    static_cast<std::size_t>(expert) * per_expert_activation +
                    (static_cast<std::size_t>(block) * kRegfedKPerLoad + j) *
                        kColBlocks * groups_per_block * 4U;
#pragma unroll
                for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
                    const uint2 b = live[c] ? activations[base + offset[c]]
                                            : make_uint2(0U, 0U);
                    dsv4_mma_m16n8k16(acc[c][0], acc[c][1], acc[c][2], acc[c][3],
                                      a[0], a[1], a[2], a[3], b.x, b.y);
                }
            }
        }
#pragma unroll
        for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
#pragma unroll
            for (std::uint32_t i = 0U; i < 4U; ++i) {
                const std::uint32_t row = group + ((i >= 2U) ? 8U : 0U);
                const std::uint32_t column = c * kRegfedTileM + thread * 2U + (i & 1U);
                if (column >= m) continue;
                const std::size_t slot =
                    ((static_cast<std::size_t>(expert) * rows +
                      n_tile * kRegfedTileN + row) * m + column) * split + slice;
                partials[slot] = acc[c][i];
            }
        }
    }
}

__global__ void regfed_mxfp4_moe_reduce_kernel(
    float* __restrict__ output, const float* __restrict__ partials,
    std::uint32_t experts, std::uint32_t rows, std::uint32_t m,
    std::uint32_t split, unsigned int* __restrict__ error_flag) {
    const std::uint64_t total = static_cast<std::uint64_t>(experts) * rows * m;
    for (std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += gridDim.x * blockDim.x) {
        const std::uint32_t column = static_cast<std::uint32_t>(index % m);
        const std::uint32_t rest = static_cast<std::uint32_t>(index / m);
        const std::uint32_t output_row = rest % rows;
        const std::uint32_t expert = rest / rows;
        float sum = 0.0F;
        for (std::uint32_t slice = 0U; slice < split; ++slice) {
            sum += partials[index * split + slice];
        }
        if (!isfinite(sum)) atomicExch(error_flag, 1U);
        output[(static_cast<std::size_t>(expert) * m + column) * rows + output_row] = sum;
    }
}

// Scratch needed to permute this weight into fragment order, or zero if the
// encoding or the extents are inadmissible. The permutation cannot be done in
// place, but the scratch is transient and shared across every weight on the
// device, so no second persistent copy of any weight exists.
[[nodiscard]] std::uint64_t fragment_prepack_scratch_bytes(
    const CudaWeightDescriptor& descriptor) noexcept {
    if (descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128 ||
        descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128F32) {
        if (!regfed_fp8_shape_admissible(descriptor.rows, descriptor.columns)) {
            return 0U;
        }
        return descriptor.rows * descriptor.columns;
    }
    if (descriptor.encoding == CudaWeightEncoding::Fp4E2m1Group32) {
        if (!regfed_fp4_shape_admissible(descriptor.rows, descriptor.columns)) {
            return 0U;
        }
        return descriptor.rows * descriptor.packed_columns;
    }
    // NVFP4 shares the FP4 code permutation exactly -- the same E2M1 nibble
    // pairs in the same fragment order -- and differs only in the group width
    // the scale permutation is told about, which it already reads from the
    // descriptor. The scratch is the larger of the two copies it stages, and
    // packed_columns is columns/2 against scale_columns of columns/16, so the
    // code copy bounds it.
    if (descriptor.encoding == CudaWeightEncoding::Nvfp4Group16) {
        if (!regfed_nvfp4_shape_admissible(descriptor.rows,
                                           descriptor.columns) ||
            descriptor.group_size != kRegfedNvfp4Group) {
            return 0U;
        }
        return descriptor.rows * descriptor.packed_columns;
    }
    return 0U;
}

// Stream-ordered fragment prepack. Ordering behind the upload copy is a device
// dependency, not a host one, so this enqueues on the same stream the copy used
// and never blocks the loader.
cudaError_t launch_fragment_prepack(const CudaWeightDescriptor& descriptor,
                                    void* weights, void* scales, void* scratch,
                                    cudaStream_t stream) {
    const auto rows = static_cast<std::uint32_t>(descriptor.rows);
    const auto columns = static_cast<std::uint32_t>(descriptor.columns);
    constexpr unsigned int threads = 256U;
    const auto grid = [](std::uint64_t total) {
        return static_cast<unsigned int>(
            std::min<std::uint64_t>((total + threads - 1U) / threads, 65535U));
    };
    if (descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128 ||
        descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128F32) {
        const std::uint64_t bytes = descriptor.rows * descriptor.columns;
        if (auto status = cudaMemcpyAsync(scratch, weights,
                                          static_cast<std::size_t>(bytes),
                                          cudaMemcpyDeviceToDevice, stream);
            status != cudaSuccess) {
            return status;
        }
        const std::uint64_t total =
            (descriptor.rows / 16U) * (descriptor.columns / 32U) * 32U;
        dsv4_fp8_fragment_prepack_kernel<<<grid(total), threads, 0U, stream>>>(
            static_cast<uint4*>(weights),
            static_cast<const unsigned char*>(scratch), rows, columns);
        return cudaGetLastError();
    }
    const std::uint64_t code_bytes = descriptor.rows * descriptor.packed_columns;
    if (auto status = cudaMemcpyAsync(scratch, weights,
                                      static_cast<std::size_t>(code_bytes),
                                      cudaMemcpyDeviceToDevice, stream);
        status != cudaSuccess) {
        return status;
    }
    const std::uint64_t code_total =
        (descriptor.rows / kRegfedTileN) * (descriptor.columns / kRegfedTileK) *
        kRegfedWarp;
    regfed_fp4_prepack_codes_kernel<<<grid(code_total), threads, 0U, stream>>>(
        static_cast<std::uint32_t*>(weights),
        static_cast<const unsigned char*>(scratch), rows, columns);
    if (auto status = cudaGetLastError(); status != cudaSuccess) return status;
    const std::uint64_t scale_bytes = descriptor.rows * descriptor.scale_columns;
    if (auto status = cudaMemcpyAsync(scratch, scales,
                                      static_cast<std::size_t>(scale_bytes),
                                      cudaMemcpyDeviceToDevice, stream);
        status != cudaSuccess) {
        return status;
    }
    regfed_fp4_prepack_scales_kernel<<<grid(scale_bytes), threads, 0U, stream>>>(
        static_cast<unsigned char*>(scales),
        static_cast<const unsigned char*>(scratch), rows,
        static_cast<std::uint32_t>(descriptor.scale_columns));
    return cudaGetLastError();
}

[[nodiscard]] std::uint64_t marlin_prepack_scratch_bytes(
    const CudaWeightDescriptor& descriptor) noexcept {
    if (descriptor.encoding != CudaWeightEncoding::Fp4E2m1Group32 ||
        descriptor.rows == 0U || descriptor.columns == 0U ||
        descriptor.rows % 64U != 0U || descriptor.columns % 256U != 0U) {
        return 0U;
    }
    return std::max(descriptor.rows * descriptor.packed_columns,
                    descriptor.rows * descriptor.scale_columns);
}

cudaError_t launch_marlin_prepack(const CudaWeightDescriptor& descriptor,
                                  void* weights, void* scales, void* scratch,
                                  cudaStream_t stream) {
    constexpr unsigned int threads = 256U;
    const auto grid = [](std::uint64_t total) {
        return static_cast<unsigned int>(
            std::min<std::uint64_t>((total + threads - 1U) / threads, 65535U));
    };
    const auto code_bytes = descriptor.rows * descriptor.packed_columns;
    if (auto status = cudaMemcpyAsync(scratch, weights, code_bytes,
                                      cudaMemcpyDeviceToDevice, stream);
        status != cudaSuccess) {
        return status;
    }
    const auto code_words = code_bytes / sizeof(std::uint32_t);
    gemma_marlin_prepack_codes_kernel<<<grid(code_words), threads, 0U, stream>>>(
        static_cast<std::uint32_t*>(weights),
        static_cast<const unsigned char*>(scratch),
        static_cast<std::uint32_t>(descriptor.rows),
        static_cast<std::uint32_t>(descriptor.columns));
    if (auto status = cudaGetLastError(); status != cudaSuccess) return status;
    const auto scale_bytes = descriptor.rows * descriptor.scale_columns;
    if (auto status = cudaMemcpyAsync(scratch, scales, scale_bytes,
                                      cudaMemcpyDeviceToDevice, stream);
        status != cudaSuccess) {
        return status;
    }
    gemma_marlin_prepack_scales_kernel<<<grid(scale_bytes), threads, 0U, stream>>>(
        static_cast<unsigned char*>(scales),
        static_cast<const unsigned char*>(scratch),
        static_cast<std::uint32_t>(descriptor.rows),
        static_cast<std::uint32_t>(descriptor.scale_columns));
    return cudaGetLastError();
}

// Standalone SwiGLU for the register-fed shared expert. The rounding, the
// finite check, the clamp order and the BF16 SiLU table lookup are reproduced
// from deepseek_fp8_gate_up_kernel exactly, so substituting the projections
// underneath cannot move the activation by itself.
__global__ void regfed_shared_swiglu_kernel(
    float* __restrict__ activation, const float* __restrict__ gate,
    const float* __restrict__ up, std::uint32_t intermediate,
    float swiglu_limit, const float* __restrict__ bf16_silu,
    unsigned int* __restrict__ error_flag) {
    const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= intermediate) return;
    const float rounded_gate = bf16_round(gate[row]);
    const float rounded_up = bf16_round(up[row]);
    if (!isfinite(rounded_gate) || !isfinite(rounded_up)) {
        atomicExch(error_flag, 1U);
        activation[row] = __uint_as_float(0x7FC0'0000U);
        return;
    }
    const float limited_gate = fminf(rounded_gate, swiglu_limit);
    const float limited_up =
        fmaxf(-swiglu_limit, fminf(rounded_up, swiglu_limit));
    const auto gate_bits =
        static_cast<std::uint16_t>(__float_as_uint(limited_gate) >> 16U);
    activation[row] = bf16_round(bf16_silu[gate_bits] * limited_up);
}

// Workspaces for a register-fed dispatch made outside matmul_impl. Every buffer
// is grown geometrically and kept, so a decode step that repeats the same shapes
// allocates nothing after the first token.
struct RegfedWorkspace {
    void* activation{};
    void* partials{};
    void* counters{};
    void* scratch{};
    std::uint64_t activation_bytes{};
    std::uint64_t partial_bytes{};
    std::uint64_t counter_bytes{};
    std::uint64_t scratch_bytes{};
};

cudaError_t regfed_grow(void*& pointer, std::uint64_t& capacity,
                        std::uint64_t required, bool zero,
                        cudaStream_t stream) {
    if (required <= capacity) return cudaSuccess;
    if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
    pointer = nullptr;
    capacity = 0U;
    if (auto status = cudaMalloc(&pointer, static_cast<std::size_t>(required));
        status != cudaSuccess) {
        return status;
    }
    capacity = required;
    if (!zero) return cudaSuccess;
    // Zeroing the whole allocation, not just the range in use, is what lets a
    // later call with more N-tiles reuse the buffer: the fold resets every
    // counter it touches, and everything it has not touched is still zero.
    return cudaMemsetAsync(pointer, 0, static_cast<std::size_t>(required),
                           stream);
}
