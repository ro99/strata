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

// Gemma's Marlin path keeps the existing per-row/K128 E4M3 activation
// simulation, but publishes the exact simulated value as BF16 for W4A16 MMA.
// Rows above `rows` are padding for the fixed M=128 page specialization.
__global__ void quantize_activation_e4m3_bf16_kernel(
    __nv_bfloat16* output, const float* input, std::uint64_t columns,
    std::uint32_t rows, std::uint32_t padded_rows) {
    const std::uint32_t row = blockIdx.y;
    const std::uint64_t group_begin =
        static_cast<std::uint64_t>(blockIdx.x) * 128U;
    if (row >= padded_rows || group_begin >= columns) return;
    const std::uint64_t column = group_begin + threadIdx.x;
    const float magnitude = row < rows && column < columns
        ? fabsf(input[static_cast<std::uint64_t>(row) * columns + column])
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
    if (column >= columns) return;
    float value = 0.0F;
    if (row < rows) {
        float scale = 1.0F;
        if (maximum[0] > 0.0F) {
            scale = exp2f(ceilf(log2f(maximum[0] / 448.0F)));
        }
        value = quantize_e4m3_value(
            input[static_cast<std::uint64_t>(row) * columns + column] /
            scale) * scale;
    }
    output[static_cast<std::uint64_t>(row) * columns + column] =
        __float2bfloat16_rn(value);
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

constexpr std::uint32_t kDsv4Fp8TensorBlockM = 64U;
constexpr std::uint32_t kDsv4Fp8TensorBlockN = 128U;
constexpr std::uint32_t kDsv4Fp8TensorBlockK = 128U;

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
    if (descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128) {
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
    if (descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128) {
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

struct GemmaMarlinWorkspace {
    void* activation{};
    void* reduce{};
    void* reorder{};
    void* locks{};
    std::uint64_t activation_bytes{};
    std::uint64_t reduce_bytes{};
    std::uint64_t reorder_bytes{};
    std::uint64_t lock_bytes{};
    int multiprocessors{};
    int maximum_shared{};
    bool configured{};
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

cudaError_t launch_gemma_marlin(
    GemmaMarlinWorkspace& workspace, const CudaWeightDescriptor& descriptor,
    const void* weights, const void* scales, const float* input,
    std::uint32_t rows, float* output, cudaStream_t stream,
    bool reuse_activation = false) {
    if (rows == 0U || rows > 128U ||
        descriptor.encoding != CudaWeightEncoding::Fp4E2m1Group32 ||
        descriptor.rows % 64U != 0U || descriptor.columns % 256U != 0U) {
        return cudaErrorInvalidValue;
    }
    // The old integration padded every multi-row call to M=128.  Marlin has
    // exact partial-M kernels; keep the useful row count so ordinary chat
    // prompts do not pay a full-page schedule.
    const std::uint32_t kernel_rows = rows;
    if (!workspace.configured) {
        int device = 0;
        if (auto status = cudaGetDevice(&device); status != cudaSuccess) {
            return status;
        }
        if (auto status = cudaDeviceGetAttribute(
                &workspace.multiprocessors, cudaDevAttrMultiProcessorCount,
                device);
            status != cudaSuccess) {
            return status;
        }
        if (auto status = cudaDeviceGetAttribute(
                &workspace.maximum_shared,
                cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
            status != cudaSuccess) {
            return status;
        }
        if (auto status = cudaFuncSetAttribute(
                ::marlin::Marlin<
                    vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                    vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                    256, 1, 8, 8, true, 4, 2, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                workspace.maximum_shared);
            status != cudaSuccess) {
            return status;
        }
        if (auto status = cudaFuncSetAttribute(
                ::marlin::Marlin<
                    vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                    vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                    256, 1, 8, 8, false, 4, 2, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                workspace.maximum_shared);
            status != cudaSuccess) {
            return status;
        }
        if (auto status = cudaFuncSetAttribute(
                ::marlin::Marlin<
                    vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                    vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                    256, 2, 16, 4, false, 4, 2, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                workspace.maximum_shared);
            status != cudaSuccess) {
            return status;
        }
        if (auto status = cudaFuncSetAttribute(
                ::marlin::Marlin<
                    vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                    vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                    256, 3, 16, 4, false, 4, 2, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                workspace.maximum_shared);
            status != cudaSuccess) {
            return status;
        }
        if (auto status = cudaFuncSetAttribute(
                ::marlin::Marlin<
                    vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                    vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),
                    256, 4, 16, 4, false, 4, 2, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                workspace.maximum_shared);
            status != cudaSuccess) {
            return status;
        }
        workspace.configured = true;
    }
    const auto activation_bytes =
        static_cast<std::uint64_t>(kernel_rows) * descriptor.columns *
        sizeof(__nv_bfloat16);
    const auto reduce_bytes =
        static_cast<std::uint64_t>(workspace.multiprocessors) * 64U * 256U *
        sizeof(float);
    const auto reorder_bytes =
        static_cast<std::uint64_t>(workspace.multiprocessors) * 64U * 264U *
        sizeof(float);
    const auto lock_bytes =
        static_cast<std::uint64_t>(workspace.multiprocessors) * sizeof(int);
    if (auto status = regfed_grow(workspace.activation,
                                  workspace.activation_bytes,
                                  activation_bytes, false, stream);
        status != cudaSuccess) {
        return status;
    }
    if (auto status = regfed_grow(workspace.reduce, workspace.reduce_bytes,
                                  reduce_bytes, false, stream);
        status != cudaSuccess) {
        return status;
    }
    if (auto status = regfed_grow(workspace.reorder, workspace.reorder_bytes,
                                  reorder_bytes, false, stream);
        status != cudaSuccess) {
        return status;
    }
    if (auto status = regfed_grow(workspace.locks, workspace.lock_bytes,
                                  lock_bytes, true, stream);
        status != cudaSuccess) {
        return status;
    }
    if (!reuse_activation) {
        const dim3 quantize_grid(
            static_cast<unsigned int>((descriptor.columns + 127U) / 128U),
            kernel_rows, 1U);
        quantize_activation_e4m3_bf16_kernel<<<
            quantize_grid, 128U, 0U, stream>>>(
            static_cast<__nv_bfloat16*>(workspace.activation), input,
            descriptor.columns, rows, kernel_rows);
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            return status;
        }
    }
    const auto launch_segment = [&](std::uint32_t row_offset,
                                    std::uint32_t segment_rows) {
        const auto activation_offset = static_cast<std::uint64_t>(row_offset) *
                                       descriptor.columns / 8U;
        const auto output_offset = static_cast<std::uint64_t>(row_offset) *
                                   descriptor.rows / 4U;
        const auto* activation =
            static_cast<const int4*>(workspace.activation) + activation_offset;
        auto* destination = reinterpret_cast<int4*>(output) + output_offset;

#define STRATA_LAUNCH_GEMMA_MARLIN(TM, TN, TK, M8)                         \
        ::marlin::Marlin<                                                   \
            vllm::kBFloat16.id(), vllm::kFE2M1f.id(),                       \
            vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(),                     \
            256, TM, TN, TK, M8, 4, 2, false>                               \
            <<<workspace.multiprocessors, 256U, workspace.maximum_shared,   \
               stream>>>(                                                   \
                activation, static_cast<const int4*>(weights), destination, \
                static_cast<int4*>(workspace.reduce), nullptr, nullptr,      \
                static_cast<const int4*>(scales),                            \
                static_cast<const float*>(workspace.reorder), nullptr,       \
                nullptr, static_cast<int>(descriptor.scale_columns),         \
                static_cast<int>(segment_rows),                              \
                static_cast<int>(descriptor.rows),                           \
                static_cast<int>(descriptor.columns),                        \
                static_cast<int>(descriptor.columns),                        \
                static_cast<int*>(workspace.locks), false, false, true,      \
                workspace.maximum_shared)

        if (segment_rows <= 8U) {
            STRATA_LAUNCH_GEMMA_MARLIN(1, 8, 8, true);
        } else if (segment_rows <= 16U) {
            STRATA_LAUNCH_GEMMA_MARLIN(1, 8, 8, false);
        } else if (segment_rows <= 32U) {
            STRATA_LAUNCH_GEMMA_MARLIN(2, 16, 4, false);
        } else if (segment_rows <= 48U) {
            STRATA_LAUNCH_GEMMA_MARLIN(3, 16, 4, false);
        } else {
            STRATA_LAUNCH_GEMMA_MARLIN(4, 16, 4, false);
        }
#undef STRATA_LAUNCH_GEMMA_MARLIN
    };
    // Marlin's internal parallel count is integer division by the M tile.  A
    // single 65..127-row launch would therefore drop the remainder; mirror the
    // upstream dispatcher with a 64-row body and an exact partial tail.
    if (rows > 64U && rows < 128U) {
        launch_segment(0U, 64U);
        launch_segment(64U, rows - 64U);
    } else {
        launch_segment(0U, rows);
    }
    return cudaGetLastError();
}

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

// Reproduces encode_e4m3_half_up() from src/dsv4_attention_kv.cpp exactly.
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

}  // namespace
