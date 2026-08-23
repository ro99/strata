#include "strata/cuda_backend.hpp"
#include "strata/numerics.hpp"

#include <cublas_v2.h>
#include <cstdlib>
#include <cstdio>
#include <atomic>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <mutex>
#include <new>
#include <tuple>
#include <type_traits>
#include <unordered_map>

namespace strata {

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
    __shared__ double warps[8];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) warps[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? warps[lane] : 0.0;
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
        const float value = input[column];
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
        output[column] = bf16_round(input[column] * reciprocal * weight[column]);
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

__global__ void gemma4_norm_rope_kernel(
    float* values, const float* weight, std::uint32_t heads,
    std::uint32_t head_dim, std::uint32_t position, float theta,
    float rotary_proportion, unsigned int* error_flag) {
    const auto head = static_cast<std::uint32_t>(blockIdx.x);
    if (head >= heads || threadIdx.x != 0U) return;
    auto* row = values + static_cast<std::uint64_t>(head) * head_dim;
    double squared_sum = 0.0;
    for (std::uint32_t column = 0U; column < head_dim; ++column) {
        const float value = row[column];
        if (!isfinite(value) || (weight != nullptr && !isfinite(weight[column]))) {
            atomicExch(error_flag, 1U);
            return;
        }
        squared_sum = __dadd_rn(
            squared_sum,
            __dmul_rn(static_cast<double>(value), static_cast<double>(value)));
    }
    const float reciprocal = 1.0F / sqrtf(
        static_cast<float>(squared_sum / static_cast<double>(head_dim)) +
        1.0e-6F);
    for (std::uint32_t column = 0U; column < head_dim; ++column) {
        row[column] = bf16_round(
            row[column] * reciprocal * (weight == nullptr ? 1.0F : weight[column]));
    }
    if (theta == 0.0F) return;
    const auto half = head_dim / 2U;
    const auto angles = static_cast<std::uint32_t>(
        rotary_proportion * static_cast<float>(head_dim) / 2.0F);
    for (std::uint32_t index = 0U; index < angles; ++index) {
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

__global__ void gemma4_geglu_kernel(
    float* gate, const float* up, std::uint32_t columns,
    unsigned int* error_flag) {
    const auto column = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (column >= columns) return;
    if (!isfinite(gate[column]) || !isfinite(up[column])) {
        atomicExch(error_flag, 1U);
        return;
    }
    constexpr float coefficient = 0.7978845608028654F;
    const float value = gate[column];
    const float activated = 0.5F * value *
        (1.0F + tanhf(coefficient *
                      (value + 0.044715F * value * value * value)));
    gate[column] = bf16_round(bf16_round(activated) * up[column]);
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
__global__ void dsv4_host_moe_join_kernel(
    float* shared_and_output, const float* rank_partials,
    std::uint64_t hidden_columns, unsigned int* error_flag) {
    const auto column = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (column >= hidden_columns) return;
    const float routed = bf16_round(
        rank_partials[column] + rank_partials[hidden_columns + column]);
    const float shared = bf16_round(shared_and_output[column]);
    const float output = bf16_round(routed + shared);
    shared_and_output[column] = output;
    if (!isfinite(output)) atomicExch(error_flag, 3U);
}

__global__ void dsv4_host_moe_join_mhc_kernel(
    float* shared_and_output, const float* rank_partials,
    __nv_bfloat16* mhc_branch, std::uint64_t hidden_columns,
    unsigned int* error_flag) {
    const auto column = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (column >= hidden_columns) return;
    const float routed = bf16_round(
        rank_partials[column] + rank_partials[hidden_columns + column]);
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
    std::uint64_t kv_bytes{};
    std::uint64_t score_bytes{};
    std::uint64_t upload_bytes{};
    std::uint64_t workspace_bytes{};
};

bool dsv4_attention_mhc_workspace_layout(
    std::uint64_t page_count, std::uint32_t rows,
    std::uint32_t total_heads, std::uint32_t output_groups,
    std::uint32_t candidates, std::uint32_t flat_rows,
    bool use_prepared_query, std::uint64_t mhc_slot_count,
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
        !checked_bytes(use_prepared_query ? 0U : attended_elements, 1U,
                       sizeof(std::uint16_t), layout.query_bytes) ||
        !checked_bytes(total_heads, 1U, sizeof(float), layout.sink_bytes) ||
        !checked_bytes(rows, rope_pairs, sizeof(float), layout.rope_bytes) ||
        !checked_bytes(mhc_slot_count, 1U, sizeof(std::uint32_t),
                       layout.slot_bytes) ||
        !checked_bytes(resolution_block_count, 1U,
                       sizeof(Dsv4DeviceKvBlock), layout.block_bytes) ||
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
        !region(layout.block_bytes, 16U, layout.block_offset)) {
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
    if (!checked_bytes(tensor_padded_rows, branch_row_elements, 1U,
                       branch_capacity_elements) ||
        !checked_bytes(rows, output_rank_row_elements, 1U,
                       tensor_values_bytes) ||
        !checked_bytes(rows, output_rank_row_elements / 128U, 1U,
                       tensor_scales_bytes)) {
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
        : device_(device), base_(static_cast<std::byte*>(base)) {
        free_.reserve(16'384U);
        free_.push_back({0U, capacity});
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
    std::vector<Block> free_;
    std::mutex mutex_;
    bool metadata_failed_{};
};

struct Dsv4HostMoeCallbackState {
    CudaDsv4HostMoeCallback function{};
    CudaDsv4DeviceInputHostMoeCallback device_input_function{};
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

void CUDART_CB run_dsv4_host_moe_callback(void* opaque) {
    auto& state = *static_cast<Dsv4HostMoeCallbackState*>(opaque);
    state.started = std::chrono::steady_clock::now();
    bool accepted = false;
    try {
        state.upstream_failure_value = state.upstream_failure == nullptr
            ? 0U : *state.upstream_failure;
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

struct CudaWeight::Impl {
    void* weights{};
    void* scales{};
    CudaWeightDescriptor descriptor;
    // Set once, at load, by CudaBackend::prepack_fragment. The fragment order
    // REPLACES the canonical device layout -- one-copy residency -- so every
    // consumer of this weight must dispatch a register-fed kernel once this is
    // true. A consumer that reads it canonically would read a permutation.
    bool fragment_prepacked{};
    std::uint64_t bytes{};
    int device{-1};
    std::shared_ptr<WeightArena> arena;
    std::uint64_t arena_offset{};

    ~Impl() {
        if (arena != nullptr) {
            arena->release(arena_offset, bytes);
            return;
        }
        if (device >= 0) static_cast<void>(cudaSetDevice(device));
        if (weights != nullptr) static_cast<void>(cudaFree(weights));
        if (scales != nullptr) static_cast<void>(cudaFree(scales));
    }
};

struct CudaBuffer::Impl {
    void* data{};
    std::uint64_t bytes{};
    int device{-1};

    ~Impl() {
        if (device >= 0) static_cast<void>(cudaSetDevice(device));
        if (data != nullptr) static_cast<void>(cudaFree(data));
    }
};

struct CudaDsv4MhcWeights::Impl {
    CudaWeight projection;
    CudaBuffer auxiliary;
};

struct CudaBackend::Impl {
    struct DeviceState {
        cudaStream_t stream{};
        cudaStream_t moe_shared_stream{};
        std::array<cudaStream_t, 3U> dsv4_attention_aux_streams{};
        cublasHandle_t cublas{};
        cudaStream_t upload_stream{};
        cudaEvent_t upload_ready{};
        bool upload_ordered{};
        cudaEvent_t activation_start{};
        cudaEvent_t activation_uploaded{};
        cudaEvent_t mhc_transition_finished{};
        cudaEvent_t router_started{};
        cudaEvent_t kernel_finished{};
        cudaEvent_t activation_downloaded{};
        cudaEvent_t moe_start{};
        cudaEvent_t moe_hidden_uploaded{};
        cudaEvent_t moe_kernel_finished{};
        cudaEvent_t moe_download_started{};
        cudaEvent_t moe_completed{};
        cudaEvent_t moe_shared_input_finished{};
        cudaEvent_t moe_shared_gate_up_finished{};
        cudaEvent_t moe_shared_activation_finished{};
        cudaEvent_t moe_shared_finished{};
        cudaEvent_t dsv4_cross_device_ready{};
        cudaEvent_t dsv4_attention_input_ready{};
        std::array<cudaEvent_t, 3U> dsv4_attention_aux_finished{};
        float* input{};
        float* output{};
        std::uint64_t input_bytes{};
        std::uint64_t output_bytes{};
        // Pinned staging for matmul activations. A cudaMemcpyAsync whose host
        // side is pageable is not asynchronous: the driver stages it itself and
        // blocks, which decode pays on both legs of every one of its ~430
        // matmul round trips a step. FlashAttention and the MoE command already
        // stage through pinned host memory; this is the same for the generic
        // matmul.
        std::byte* matmul_host_input{};
        std::byte* matmul_host_output{};
        std::uint64_t matmul_host_input_bytes{};
        std::uint64_t matmul_host_output_bytes{};
        // Register-fed matmul workspaces: the B-fragment activation, the
        // split-K partials, and one arrival counter per N-tile. All three are
        // grown geometrically and kept, so a decode step that repeats the same
        // shapes allocates nothing after the first call.
        // The shared expert dispatches on its own stream, concurrently with the
        // routed path, so it keeps workspaces separate from the generic
        // matmul's rather than sharing them.
        RegfedWorkspace moe_regfed{};
        float* moe_regfed_gate{};
        float* moe_regfed_up{};
        std::uint64_t moe_regfed_gate_bytes{};
        void* regfed_activation{};
        float* regfed_partials{};
        std::uint32_t* regfed_counters{};
        void* regfed_scratch{};
        std::uint64_t regfed_activation_bytes{};
        std::uint64_t regfed_partial_bytes{};
        std::uint64_t regfed_counter_bytes{};
        std::uint64_t regfed_scratch_bytes{};
        std::byte* attention_upload{};
        std::byte* attention_download{};
        std::byte* attention_host_upload{};
        std::byte* attention_host_download{};
        float* attention_scores{};
        std::uint64_t attention_upload_bytes{};
        std::uint64_t attention_download_bytes{};
        std::uint64_t attention_host_upload_bytes{};
        std::uint64_t attention_host_download_bytes{};
        std::uint64_t attention_score_bytes{};
        // Index-query preparation: the head-major query block plus its rope
        // cosines and sines. Grown once and reused, never on a timed path.
        std::byte* dsv4_index_query_workspace{};
        std::uint64_t dsv4_index_query_workspace_bytes{};
        std::byte* dsv4_attention_workspace{};
        std::byte* dsv4_attention_host_upload{};
        std::byte* dsv4_attention_host_download{};
        std::uint64_t dsv4_attention_workspace_bytes{};
        std::uint64_t dsv4_attention_host_upload_bytes{};
        std::uint64_t dsv4_attention_host_download_bytes{};
        std::byte* dsv4_attention_prepare_workspace{};
        std::byte* dsv4_attention_prepare_host_upload{};
        std::byte* dsv4_attention_prepare_host_download{};
        std::byte* dsv4_attention_prepare_fixed_host_upload{};
        std::uint64_t dsv4_attention_prepare_workspace_bytes{};
        std::uint64_t dsv4_attention_prepare_host_upload_bytes{};
        std::uint64_t dsv4_attention_prepare_host_download_bytes{};
        __nv_bfloat16* dsv4_prepared_queries{};
        bool dsv4_attention_prepared{};
        // Index-projection sources left behind by the last preparation on this
        // device: the E4M3-quantized query rank, and the expanded BF16 layer
        // input. Both point into the preparation workspace, so the next
        // preparation on this device overwrites them.
        const float* dsv4_prepared_index_query_source{};
        const float* dsv4_prepared_index_hidden_source{};
        // Device-only index projections awaiting an in-chain selection. Set by
        // dsv4_index_projections when it returns no host output, consumed by
        // the next dsv4_physical_lightning_index on this device.
        const float* dsv4_index_projection_queries{};
        const float* dsv4_index_projection_weights{};
        const unsigned int* dsv4_index_projection_error{};
        std::uint32_t dsv4_index_projection_heads{};
        std::uint32_t dsv4_index_projection_head_dim{};
        std::array<Dsv4AttentionPrepareHostCommand, 43U>
            dsv4_attention_prepare_host_commands{};
        std::uint32_t dsv4_attention_prepare_host_command_count{};
        bool dsv4_host_moe_input_pending{};
        float* dsv4_host_moe_router_logits{};
        unsigned int* dsv4_host_moe_device_failure{};
        const unsigned int* dsv4_host_moe_host_failure{};
        std::byte* dsv4_deferred_attention_host_upload{};
        std::byte* dsv4_deferred_attention_host_download{};
        std::uint32_t dsv4_deferred_attention_command_count{};
        int dsv4_deferred_attention_source_device{-1};
        bool dsv4_deferred_attention_cross_transition{};
        Dsv4MhcWorkspace* dsv4_mhc_workspace{};
        std::byte* dsv4_mhc_host_staging{};
        std::uint64_t dsv4_mhc_workspace_bytes{};
        std::uint64_t dsv4_mhc_host_staging_bytes{};
        std::uint32_t dsv4_mhc_stage{};
        std::uint32_t dsv4_mhc_residual_index{};
        bool dsv4_mhc_branch_ready{};
        // Saved fused mHC state of every slot that is not currently selected.
        // The three scalars above are the selected slot's live copy; selecting
        // a different slot writes them back here and loads that slot's copy.
        // The workspace pointer is the arena base plus the slot index, so it
        // is derived rather than stored.
        std::vector<Dsv4MhcSlotState> dsv4_mhc_saved_slots{};
        std::uint32_t dsv4_mhc_active_slot{};
        Dsv4MhcWorkspace* dsv4_mhc_slot_arena{};
        std::uint32_t dsv4_mhc_slot_capacity{};
        bool dsv4_mhc_failed{};
        float* dsv4_mhc_head_input{};
        float* dsv4_mhc_head_output{};
        std::byte* dsv4_mhc_head_host_staging{};
        std::uint64_t dsv4_mhc_head_input_bytes{};
        std::uint64_t dsv4_mhc_head_output_bytes{};
        std::uint64_t dsv4_mhc_head_host_staging_bytes{};
        // Logit bytes of the head currently in flight. The reservation above
        // is a capacity, because one device may serve heads of two shapes:
        // centralized prefill projects the full vocabulary while rank-local
        // decode projects one rank's row shard. A completion must still match
        // the enqueue that produced it exactly, which is what this pins.
        std::uint64_t dsv4_mhc_head_logits_bytes{};
        Dsv4MhcHeadCallbackState dsv4_mhc_head_callback{};
        bool dsv4_mhc_head_in_flight{};
        float* gemma_workspace{};
        float* gemma_scores{};
        unsigned int* gemma_error{};
        std::byte* gemma_host_staging{};
        std::uint64_t gemma_workspace_bytes{};
        std::uint64_t gemma_score_bytes{};
        std::uint64_t gemma_host_staging_bytes{};
        std::byte* lightning_workspace{};
        std::uint64_t lightning_workspace_bytes{};
        float* moe_hidden{};
        float* moe_activations{};
        float* moe_output{};
        float* moe_bf16_silu{};
        unsigned int* moe_error{};
        void* moe_host_staging{};
        // Routed-expert tier: host-side pointer arrays mirrored to device, the
        // pinned selection the callback writes, and its device copy.
        std::vector<const unsigned char*> tier_host_pointers[6];
        const unsigned char** tier_device_pointers[6]{};
        CudaDsv4TierSelection* tier_selection_host{};
        CudaDsv4TierSelection* tier_selection_device{};
        std::uint32_t tier_layers{};
        std::uint32_t tier_experts{};
        std::uint64_t tier_installed{};
        bool tier_committed{};
        // Taken from the first installed triplet rather than assumed, and
        // every later triplet must match: a tier holding two shapes would
        // index one of them wrongly.
        std::uint64_t tier_gate_packed_columns{};
        std::uint64_t tier_gate_scale_columns{};
        std::uint64_t tier_down_packed_columns{};
        std::uint64_t tier_down_scale_columns{};
        float* tier_activations{};
        std::uint64_t tier_activation_bytes{};
        std::uint64_t moe_hidden_bytes{};
        std::uint64_t moe_activation_bytes{};
        std::uint64_t moe_output_bytes{};
        std::uint64_t moe_host_staging_bytes{};
        std::uint64_t moe_hidden_columns{};
        std::uint64_t moe_intermediate_columns{};
        std::uint32_t moe_rows{1U};
        std::uint32_t moe_routed_count{};
        std::uint64_t moe_kernel_launches{};
        // Page-path work list: row indices, per-row coefficients, and the group
        // table, all device-side. Sized to the largest page seen so far and
        // reused, because a prefill visits every layer with the same shape.
        std::uint32_t* moe_page_rows{};
        float* moe_page_coefficients{};
        void* moe_page_groups{};
        std::uint32_t* moe_page_shared_rows{};
        std::uint64_t moe_page_rows_bytes{};
        std::uint64_t moe_page_coefficient_bytes{};
        std::uint64_t moe_page_group_bytes{};
        std::uint64_t moe_page_shared_row_bytes{};
        std::uint32_t moe_page_work_count{};
        std::uint32_t moe_page_shared_count{};
        // Rows the shared expert produced. One for the single-row command; the
        // page command sets it to the page's shared row count so collection
        // sizes the download from the command that actually ran.
        std::uint32_t moe_shared_rows{1U};
        // Set while a deferred upload's copies are still in flight on `stream`.
        // Cleared by synchronize_uploads(), which the caller owes before the
        // upload's host source may be released.
        bool pending_uploads{};
        std::vector<std::shared_ptr<CudaWeight::Impl>> moe_weights;
        std::vector<std::shared_ptr<CudaWeight::Impl>> quarantined_weights;
        std::vector<std::shared_ptr<CudaBuffer::Impl>> quarantined_buffers;
        std::shared_ptr<WeightArena> weight_arena;
        bool moe_has_shared{};
        // True only for the single-row FP8 shared-expert path that records all
        // four phase events below. Host-join, page, and generic MoE commands
        // reuse moe_shared_finished for ordering but do not populate the
        // intermediate events.
        bool moe_shared_phase_timing_valid{};
        bool moe_host_join{};
        bool moe_output_to_mhc{};
        bool moe_in_flight{};
        bool moe_poisoned{};
        Dsv4HostMoeCallbackState moe_host_callback;
        std::array<Dsv4HostMoeCallbackState, 43U>
            moe_host_callbacks{};
        std::uint32_t moe_host_callback_count{};
        bool flash_attention_supported{};
        bool dsv4_paged_attention_supported{};
        bool dsv4_mhc_supported{};
        bool lightning_index_supported{};
        bool dsv4_fp8_tensor_page_supported{};
    };

    std::unordered_map<int, DeviceState> devices;
    CudaBackendStats stats;
    bool detailed_timing{};
    mutable std::mutex mutex;

    ~Impl() {
        for (auto& [device, state] : devices) {
            static_cast<void>(cudaSetDevice(device));
            if (state.input != nullptr) static_cast<void>(cudaFree(state.input));
            if (state.output != nullptr) static_cast<void>(cudaFree(state.output));
            if (state.attention_upload != nullptr) {
                static_cast<void>(cudaFree(state.attention_upload));
            }
            if (state.attention_download != nullptr) {
                static_cast<void>(cudaFree(state.attention_download));
            }
            if (state.attention_scores != nullptr) {
                static_cast<void>(cudaFree(state.attention_scores));
            }
            if (state.dsv4_attention_workspace != nullptr) {
                static_cast<void>(cudaFree(state.dsv4_attention_workspace));
            }
            if (state.dsv4_index_query_workspace != nullptr) {
                static_cast<void>(cudaFree(state.dsv4_index_query_workspace));
            }
            if (state.dsv4_attention_prepare_workspace != nullptr) {
                static_cast<void>(
                    cudaFree(state.dsv4_attention_prepare_workspace));
            }
            if (state.dsv4_mhc_slot_arena != nullptr) {
                static_cast<void>(cudaFree(state.dsv4_mhc_slot_arena));
            } else if (state.dsv4_mhc_workspace != nullptr) {
                static_cast<void>(cudaFree(state.dsv4_mhc_workspace));
            }
            if (state.gemma_workspace != nullptr) {
                static_cast<void>(cudaFree(state.gemma_workspace));
            }
            if (state.gemma_scores != nullptr) {
                static_cast<void>(cudaFree(state.gemma_scores));
            }
            if (state.gemma_error != nullptr) {
                static_cast<void>(cudaFree(state.gemma_error));
            }
            if (state.gemma_host_staging != nullptr) {
                static_cast<void>(cudaFreeHost(state.gemma_host_staging));
            }
            // moe_regfed_up is an interior pointer into the gate allocation,
            // not an allocation of its own: freeing it is cudaErrorInvalidValue.
            for (void* pointer : {state.moe_regfed.activation,
                                  state.moe_regfed.partials,
                                  state.moe_regfed.counters,
                                  state.moe_regfed.scratch,
                                  static_cast<void*>(state.moe_regfed_gate)}) {
                if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
            }
            if (state.regfed_activation != nullptr) {
                static_cast<void>(cudaFree(state.regfed_activation));
            }
            if (state.regfed_partials != nullptr) {
                static_cast<void>(cudaFree(state.regfed_partials));
            }
            if (state.regfed_counters != nullptr) {
                static_cast<void>(cudaFree(state.regfed_counters));
            }
            if (state.regfed_scratch != nullptr) {
                static_cast<void>(cudaFree(state.regfed_scratch));
            }
            if (state.matmul_host_input != nullptr) {
                static_cast<void>(cudaFreeHost(state.matmul_host_input));
            }
            if (state.matmul_host_output != nullptr) {
                static_cast<void>(cudaFreeHost(state.matmul_host_output));
            }
            if (state.attention_host_upload != nullptr) {
                static_cast<void>(cudaFreeHost(state.attention_host_upload));
            }
            if (state.attention_host_download != nullptr) {
                static_cast<void>(cudaFreeHost(state.attention_host_download));
            }
            if (state.dsv4_attention_host_upload != nullptr) {
                static_cast<void>(cudaFreeHost(
                    state.dsv4_attention_host_upload));
            }
            if (state.dsv4_attention_host_download != nullptr) {
                static_cast<void>(cudaFreeHost(
                    state.dsv4_attention_host_download));
            }
            if (state.dsv4_deferred_attention_host_upload != nullptr) {
                static_cast<void>(cudaFreeHost(
                    state.dsv4_deferred_attention_host_upload));
            }
            if (state.dsv4_deferred_attention_host_download != nullptr) {
                static_cast<void>(cudaFreeHost(
                    state.dsv4_deferred_attention_host_download));
            }
            if (state.dsv4_attention_prepare_host_upload != nullptr) {
                static_cast<void>(cudaFreeHost(
                    state.dsv4_attention_prepare_host_upload));
            }
            if (state.dsv4_attention_prepare_host_download != nullptr) {
                static_cast<void>(cudaFreeHost(
                    state.dsv4_attention_prepare_host_download));
            }
            if (state.dsv4_attention_prepare_fixed_host_upload != nullptr) {
                static_cast<void>(cudaFreeHost(
                    state.dsv4_attention_prepare_fixed_host_upload));
            }
            if (state.dsv4_mhc_host_staging != nullptr) {
                static_cast<void>(cudaFreeHost(state.dsv4_mhc_host_staging));
            }
            if (state.dsv4_mhc_head_input != nullptr) {
                static_cast<void>(cudaFree(state.dsv4_mhc_head_input));
            }
            if (state.dsv4_mhc_head_output != nullptr) {
                static_cast<void>(cudaFree(state.dsv4_mhc_head_output));
            }
            if (state.dsv4_mhc_head_host_staging != nullptr) {
                static_cast<void>(cudaFreeHost(
                    state.dsv4_mhc_head_host_staging));
            }
            if (state.lightning_workspace != nullptr) {
                static_cast<void>(cudaFree(state.lightning_workspace));
            }
            if (state.moe_hidden != nullptr) static_cast<void>(cudaFree(state.moe_hidden));
            if (state.moe_activations != nullptr) {
                static_cast<void>(cudaFree(state.moe_activations));
            }
            if (state.moe_output != nullptr) static_cast<void>(cudaFree(state.moe_output));
            if (state.moe_bf16_silu != nullptr) {
                static_cast<void>(cudaFree(state.moe_bf16_silu));
            }
            if (state.moe_error != nullptr) static_cast<void>(cudaFree(state.moe_error));
            if (state.moe_page_rows != nullptr) {
                static_cast<void>(cudaFree(state.moe_page_rows));
            }
            if (state.moe_page_coefficients != nullptr) {
                static_cast<void>(cudaFree(state.moe_page_coefficients));
            }
            if (state.moe_page_groups != nullptr) {
                static_cast<void>(cudaFree(state.moe_page_groups));
            }
            if (state.moe_page_shared_rows != nullptr) {
                static_cast<void>(cudaFree(state.moe_page_shared_rows));
            }
            if (state.moe_host_staging != nullptr) {
                static_cast<void>(cudaFreeHost(state.moe_host_staging));
            }
            if (state.activation_start != nullptr) {
                static_cast<void>(cudaEventDestroy(state.activation_start));
                static_cast<void>(cudaEventDestroy(state.activation_uploaded));
                static_cast<void>(cudaEventDestroy(
                    state.mhc_transition_finished));
                static_cast<void>(cudaEventDestroy(state.router_started));
                static_cast<void>(cudaEventDestroy(state.kernel_finished));
                static_cast<void>(cudaEventDestroy(state.activation_downloaded));
            }
            if (state.moe_start != nullptr) {
                static_cast<void>(cudaEventDestroy(state.moe_start));
                static_cast<void>(cudaEventDestroy(state.moe_hidden_uploaded));
                static_cast<void>(cudaEventDestroy(state.moe_kernel_finished));
                static_cast<void>(cudaEventDestroy(state.moe_download_started));
                static_cast<void>(cudaEventDestroy(state.moe_completed));
                static_cast<void>(cudaEventDestroy(
                    state.moe_shared_input_finished));
                static_cast<void>(cudaEventDestroy(
                    state.moe_shared_gate_up_finished));
                static_cast<void>(cudaEventDestroy(
                    state.moe_shared_activation_finished));
                static_cast<void>(cudaEventDestroy(state.moe_shared_finished));
                static_cast<void>(cudaEventDestroy(
                    state.dsv4_cross_device_ready));
                static_cast<void>(cudaEventDestroy(
                    state.dsv4_attention_input_ready));
                for (auto event : state.dsv4_attention_aux_finished) {
                    static_cast<void>(cudaEventDestroy(event));
                }
            }
            if (state.upload_ready != nullptr) {
                static_cast<void>(cudaEventDestroy(state.upload_ready));
            }
            if (state.upload_stream != nullptr) {
                static_cast<void>(cudaStreamSynchronize(state.upload_stream));
                static_cast<void>(cudaStreamDestroy(state.upload_stream));
            }
            if (state.moe_shared_stream != nullptr) {
                static_cast<void>(cudaStreamSynchronize(
                    state.moe_shared_stream));
                static_cast<void>(cudaStreamDestroy(state.moe_shared_stream));
            }
            for (auto stream : state.dsv4_attention_aux_streams) {
                if (stream == nullptr) continue;
                static_cast<void>(cudaStreamSynchronize(stream));
                static_cast<void>(cudaStreamDestroy(stream));
            }
            if (state.cublas != nullptr) {
                static_cast<void>(cublasDestroy(state.cublas));
            }
            if (state.stream != nullptr) static_cast<void>(cudaStreamDestroy(state.stream));
        }
    }
};

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
                                     UploadCompletion completion) {
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
            result.errors.emplace_back(
                "CUDA weight arena is exhausted; refusing per-weight allocation fallback");
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
    const auto valid_weight = [device](const CudaWeight* weight) {
        return weight != nullptr && weight->valid() && weight->device() == device &&
               weight->impl_->descriptor.encoding ==
                   CudaWeightEncoding::OffsetPackedInt8 &&
               weight->impl_->descriptor.group_size == 32U &&
               weight->impl_->descriptor.columns % 4U == 0U;
    };
    for (const auto& layer : layers) {
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
            layer.next_keys.size() != key.rows ||
            layer.next_values.size() != key.rows) {
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
    const auto launch_matvec = [&](const CudaWeight* weight,
                                   const float* activation,
                                   float* destination) {
        const auto& descriptor = weight->impl_->descriptor;
        constexpr unsigned int threads = 256U;
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
    };
    for (const auto& layer : layers) {
        const auto& query = layer.query->impl_->descriptor;
        const auto& key = layer.key->impl_->descriptor;
        const auto& intermediate = layer.gate->impl_->descriptor;
        const auto head_dim = static_cast<std::uint32_t>(
            layer.query_norm->device_bytes() / sizeof(float));
        const auto query_heads = static_cast<std::uint32_t>(query.rows / head_dim);
        const auto kv_heads = static_cast<std::uint32_t>(key.rows / head_dim);
        gemma4_rms_norm_kernel<<<1U, 256U, 0U, state.stream>>>(
            normalized, hidden,
            static_cast<const float*>(layer.input_norm->impl_->data),
            hidden_columns, 1.0e-6F, state.gemma_error);
        launch_matvec(layer.query, normalized, queries);
        launch_matvec(layer.key, normalized, keys);
        if (layer.value == nullptr) {
            if (auto status = cudaMemcpyAsync(
                    values, keys, key.rows * sizeof(float),
                    cudaMemcpyDeviceToDevice, state.stream);
                status != cudaSuccess) {
                return cuda_error(status, "copy Gemma 4 shared K/V projection");
            }
        } else {
            launch_matvec(layer.value, normalized, values);
        }
        const bool global = layer.value == nullptr;
        const float theta = global ? 1'000'000.0F : 10'000.0F;
        const float proportion = global ? 0.25F : 1.0F;
        gemma4_norm_rope_kernel<<<query_heads, 1U, 0U, state.stream>>>(
            queries, static_cast<const float*>(layer.query_norm->impl_->data),
            query_heads, head_dim, position, theta, proportion,
            state.gemma_error);
        gemma4_norm_rope_kernel<<<kv_heads, 1U, 0U, state.stream>>>(
            keys, static_cast<const float*>(layer.key_norm->impl_->data),
            kv_heads, head_dim, position, theta, proportion,
            state.gemma_error);
        gemma4_norm_rope_kernel<<<kv_heads, 1U, 0U, state.stream>>>(
            values, nullptr, kv_heads, head_dim, position, 0.0F, 1.0F,
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
        launch_matvec(layer.output, context, branch);
        gemma4_post_attention_kernel<<<1U, 256U, 0U, state.stream>>>(
            hidden, normalized, branch,
            static_cast<const float*>(layer.post_attention_norm->impl_->data),
            static_cast<const float*>(layer.pre_feedforward_norm->impl_->data),
            hidden_columns, state.gemma_error);
        launch_matvec(layer.gate, normalized, gate_output);
        launch_matvec(layer.up, normalized, up_output);
        gemma4_geglu_kernel<<<
            static_cast<unsigned int>((intermediate.rows + 255U) / 256U),
            256U, 0U, state.stream>>>(
            gate_output, up_output,
            static_cast<std::uint32_t>(intermediate.rows), state.gemma_error);
        launch_matvec(layer.down, gate_output, branch);
        gemma4_post_feedforward_kernel<<<1U, 256U, 0U, state.stream>>>(
            hidden, normalized, branch,
            static_cast<const float*>(layer.post_feedforward_norm->impl_->data),
            hidden_columns, layer.scalar, state.gemma_error);
    }
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch Gemma 4 CUDA decode kernels");
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
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaStreamSynchronize(state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "synchronize Gemma 4 CUDA decode");
    }
    const auto wait_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wait_started).count());
    std::memcpy(output.data(), state.gemma_host_staging + output_offset,
                output.size_bytes());
    kv_offset = hidden_bytes * 2U;
    for (const auto& layer : layers) {
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
        record_synchronization(stats, SynchronizationSubsystem::Other, 1U,
                               wait_nanoseconds);
    }
    if (numerical_error != 0U) {
        result.errors.emplace_back("Gemma 4 CUDA decode produced a non-finite value");
    }
    return result;
}

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
    std::uint32_t candidate_width) const {
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
            static_cast<std::uint32_t>(flat_rows64), false, rows, 0U, 0U,
            layout)) {
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
    std::uint64_t maximum_workspace_bytes) const {
    ParseResult<std::uint32_t> result{};
    if (requested_rows < 2U || maximum_workspace_bytes == 0U) {
        result.errors.emplace_back(
            "DeepSeek attention page row admission request is invalid");
        return result;
    }
    auto minimum = dsv4_paged_attention_to_mhc_page_workspace_bytes(
        pages, 2U, candidate_width);
    if (!minimum.ok()) return {0U, std::move(minimum.errors)};
    if (minimum.value > maximum_workspace_bytes) {
        result.errors.emplace_back(
            "DeepSeek attention page cannot fit two rows in its bounded workspace");
        return result;
    }
    auto full = dsv4_paged_attention_to_mhc_page_workspace_bytes(
        pages, requested_rows, candidate_width);
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
            pages, middle, candidate_width);
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
    const bool use_prepared_query = request.attention.queries.empty();
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
                source_found->second.dsv4_prepared_queries == nullptr)
             : (request.attention.queries.size() != attended_elements ||
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
        std::any_of(request.attention.queries.begin(),
                    request.attention.queries.end(), [](float value) {
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
            candidates, flat_rows, use_prepared_query,
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
    const auto branch_offset = layout.branch_offset;
    const auto encoded_branch_offset = layout.encoded_branch_offset;
    const auto router_logits_offset = layout.router_logits_offset;
    const auto failure_offset = layout.failure_offset;
    const auto sink_bytes = layout.sink_bytes;
    const auto rope_bytes = layout.rope_bytes;
    const auto slot_bytes = layout.slot_bytes;
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
    const auto transition_layer_bytes = transition_mhc &&
        !defer_host_moe_input
        ? branch_elements * sizeof(std::uint16_t) : 0U;
    const auto router_download_bytes = project_router &&
        !defer_host_moe_input
        ? static_cast<std::uint64_t>(router_descriptor->rows) * sizeof(float)
        : 0U;
    constexpr std::uint64_t attention_failure_bytes = sizeof(unsigned int);
    const auto download_bytes = download_branch_bytes +
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
    if (!use_prepared_query) {
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
    auto* staged_layer = staged_branch + download_branch_bytes;
    auto* staged_router = staged_layer + transition_layer_bytes;
    auto* staged_failure = command_host_download + download_branch_bytes +
                           transition_layer_bytes + router_download_bytes;
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
            19U + static_cast<std::uint64_t>(page_request && cross_device);
        stats.dsv4_paged_attention_h2d_bytes += upload_bytes;
        stats.dsv4_paged_attention_d2h_bytes +=
            download_branch_bytes + sizeof(unsigned int);
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

ValidationResult CudaBackend::upload_dsv4_mhc_weights(
    int device, std::span<const float> projection,
    std::span<const float> scale, std::span<const float> base,
    std::span<const float> norm_weight, CudaDsv4MhcWeights& output) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek device mHC weight upload targets an uninitialized device");
        return result;
    }
    auto& state = found->second;
    if (!state.dsv4_mhc_supported) {
        result.errors.emplace_back(
            "exact DeepSeek device mHC requires an SM86 device");
        return result;
    }
    if (state.moe_in_flight || state.dsv4_mhc_stage != 0U ||
        projection.size() != kDsv4MhcProjectionElements ||
        scale.size() != 3U || base.size() != kDsv4MhcMixes ||
        norm_weight.size() != kDsv4MhcHidden) {
        result.errors.emplace_back(
            "DeepSeek device mHC weight shapes or command state are invalid");
        return result;
    }
    for (const auto values : {projection, scale, base, norm_weight}) {
        if (!std::all_of(values.begin(), values.end(), [](float value) {
                return std::isfinite(value);
            })) {
            result.errors.emplace_back(
                "DeepSeek device mHC weights contain a non-finite value");
            return result;
        }
    }

    auto target = std::make_shared<CudaDsv4MhcWeights::Impl>();
    CudaWeightDescriptor descriptor;
    descriptor.encoding = CudaWeightEncoding::Plain;
    descriptor.dtype = SafetensorsDtype::F32;
    descriptor.rows = kDsv4MhcMixes;
    descriptor.columns = kDsv4MhcMultiplier * kDsv4MhcHidden;
    result = upload(device, descriptor, std::as_bytes(projection), {},
                    target->projection);
    if (!result.ok()) return result;

    std::vector<std::byte> auxiliary(kDsv4MhcAuxBytes);
    std::memcpy(auxiliary.data(), scale.data(), scale.size_bytes());
    std::memcpy(auxiliary.data() + scale.size_bytes(), base.data(),
                base.size_bytes());
    auto* norm_bf16 = reinterpret_cast<std::uint16_t*>(
        auxiliary.data() + kDsv4MhcAuxNormOffset);
    for (std::size_t index = 0U; index < norm_weight.size(); ++index) {
        norm_bf16[index] = bf16_encode(norm_weight[index]);
    }
    auto auxiliary_target = std::make_shared<CudaBuffer::Impl>();
    auxiliary_target->bytes = auxiliary.size();
    auxiliary_target->device = device;
    const auto allocation_started = std::chrono::steady_clock::now();
    if (auto status = cudaMalloc(
            &auxiliary_target->data, auxiliary.size());
        status != cudaSuccess) {
        return cuda_error(
            status, "allocate DeepSeek device mHC auxiliary weights");
    }
    const auto allocation_nanoseconds = elapsed_nanoseconds_since(
        allocation_started);
    const auto copy_started = std::chrono::steady_clock::now();
    if (auto status = cudaMemcpyAsync(
            auxiliary_target->data, auxiliary.data(), auxiliary.size(),
            cudaMemcpyHostToDevice, state.stream); status != cudaSuccess) {
        static_cast<void>(cudaStreamSynchronize(state.stream));
        return cuda_error(
            status, "upload DeepSeek device mHC auxiliary weights");
    }
    const auto copy_nanoseconds = elapsed_nanoseconds_since(copy_started);
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaStreamSynchronize(state.stream);
        status != cudaSuccess) {
        state.quarantined_buffers.push_back(std::move(auxiliary_target));
        return cuda_error(
            status, "synchronize DeepSeek device mHC auxiliary weights");
    }
    const auto wait_nanoseconds = elapsed_nanoseconds_since(wait_started);
    target->auxiliary.impl_ = std::move(auxiliary_target);
    {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        stats.weight_upload_bytes += auxiliary.size();
        ++stats.weight_allocation_calls;
        stats.weight_allocation_bytes += auxiliary.size();
        stats.weight_allocation_nanoseconds += allocation_nanoseconds;
        stats.weight_copy_nanoseconds += copy_nanoseconds;
        record_synchronization(stats, SynchronizationSubsystem::Weight,
                               1U, wait_nanoseconds);
        stats.upload_wait_nanoseconds += wait_nanoseconds;
        stats.dsv4_mhc_resident_weight_bytes +=
            target->projection.device_bytes() +
            target->auxiliary.device_bytes();
    }
    output.impl_ = std::move(target);
    return result;
}

ValidationResult CudaBackend::dsv4_mhc_select_slot(
    int device, std::uint32_t slot) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek device mHC slot selection targets an uninitialized device");
        return result;
    }
    auto& state = found->second;
    if (!state.dsv4_mhc_supported || slot >= kDsv4MhcMaximumSlots) {
        result.errors.emplace_back(
            "DeepSeek device mHC slot selection is out of range");
        return result;
    }
    if (state.moe_in_flight || state.dsv4_mhc_head_in_flight) {
        // A slot swap rebinds what every later command reads. Doing it while
        // asynchronous work still owns the current workspace would let that
        // work land in another row's state.
        result.errors.emplace_back(
            "DeepSeek device mHC slot selection is out of order");
        return result;
    }
    if (slot == state.dsv4_mhc_active_slot) return result;
    if (slot >= state.dsv4_mhc_slot_capacity) {
        result.errors.emplace_back(
            "DeepSeek device mHC slot selection exceeds the reservation");
        return result;
    }
    const auto required = static_cast<std::size_t>(
        std::max(slot, state.dsv4_mhc_active_slot)) + 1U;
    if (state.dsv4_mhc_saved_slots.size() < required) {
        state.dsv4_mhc_saved_slots.resize(required);
    }
    auto& outgoing = state.dsv4_mhc_saved_slots[state.dsv4_mhc_active_slot];
    outgoing.stage = state.dsv4_mhc_stage;
    outgoing.residual_index = state.dsv4_mhc_residual_index;
    outgoing.branch_ready = state.dsv4_mhc_branch_ready;
    const auto& incoming = state.dsv4_mhc_saved_slots[slot];
    state.dsv4_mhc_workspace = state.dsv4_mhc_slot_arena + slot;
    state.dsv4_mhc_stage = incoming.stage;
    state.dsv4_mhc_residual_index = incoming.residual_index;
    state.dsv4_mhc_branch_ready = incoming.branch_ready;
    state.dsv4_mhc_workspace_bytes = sizeof(Dsv4MhcWorkspace);
    state.dsv4_mhc_active_slot = slot;
    return result;
}

ValidationResult CudaBackend::dsv4_mhc_reserve_slots(
    int device, std::uint32_t slots) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek device mHC slot reservation targets an uninitialized device");
        return result;
    }
    auto& state = found->second;
    if (!state.dsv4_mhc_supported || slots == 0U ||
        slots > kDsv4MhcMaximumSlots) {
        result.errors.emplace_back(
            "DeepSeek device mHC slot reservation is out of range");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status,
                          "select CUDA device for device mHC slot reservation");
    }
    if (state.dsv4_mhc_stage != 0U || state.moe_in_flight) {
        // Growing the arena moves every slot, so no row may be mid-flight.
        // A stage left non-zero by an aborted request is the usual cause, so
        // report which of the two conditions held.
        result.errors.emplace_back(
            std::string("DeepSeek device mHC slot reservation is out of order"
                        " (stage=") +
            std::to_string(state.dsv4_mhc_stage) + " moe_in_flight=" +
            (state.moe_in_flight ? "true" : "false") + ")");
        return result;
    }
    std::uint64_t allocation_calls = 0U;
    std::uint64_t allocation_bytes = 0U;
    if (slots > state.dsv4_mhc_slot_capacity) {
        void* allocation = nullptr;
        const auto bytes = static_cast<std::size_t>(slots) *
                           sizeof(Dsv4MhcWorkspace);
        if (auto status = cudaMalloc(&allocation, bytes);
            status != cudaSuccess) {
            return cuda_error(
                status, "allocate DeepSeek device mHC slot arena");
        }
        if (state.dsv4_mhc_slot_arena != nullptr) {
            static_cast<void>(cudaFree(state.dsv4_mhc_slot_arena));
        } else if (state.dsv4_mhc_workspace != nullptr) {
            // The single-state allocation the token-major path made.
            static_cast<void>(cudaFree(state.dsv4_mhc_workspace));
        }
        state.dsv4_mhc_slot_arena = static_cast<Dsv4MhcWorkspace*>(allocation);
        state.dsv4_mhc_slot_capacity = slots;
        ++allocation_calls;
        allocation_bytes += bytes;
    }
    if (state.dsv4_mhc_saved_slots.size() < slots) {
        state.dsv4_mhc_saved_slots.resize(slots);
    }
    state.dsv4_mhc_workspace =
        state.dsv4_mhc_slot_arena + state.dsv4_mhc_active_slot;
    state.dsv4_mhc_workspace_bytes = sizeof(Dsv4MhcWorkspace);
    if (allocation_calls != 0U) {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        stats.workspace_allocation_calls += allocation_calls;
        stats.workspace_allocation_bytes += allocation_bytes;
    }
    return result;
}

ValidationResult CudaBackend::dsv4_mhc_begin(
    int device, const CudaDsv4MhcWeights& weights,
    std::span<const float> hidden, std::span<float> weighted,
    std::span<float> layer_input) {
    return dsv4_mhc_begin_impl(
        device, weights, hidden, weighted, layer_input, false);
}

ValidationResult CudaBackend::dsv4_mhc_begin_device(
    int device, const CudaDsv4MhcWeights& weights,
    std::span<const float> hidden) {
    return dsv4_mhc_begin_impl(device, weights, hidden, {}, {}, true);
}

ValidationResult CudaBackend::dsv4_mhc_begin_impl(
    int device, const CudaDsv4MhcWeights& weights,
    std::span<const float> hidden, std::span<float> weighted,
    std::span<float> layer_input, bool device_only) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek device mHC begin targets an uninitialized device");
        return result;
    }
    auto& state = found->second;
    if (!weights.valid() || weights.device() != device ||
        !state.dsv4_mhc_supported || state.moe_in_flight ||
        state.dsv4_mhc_stage != 0U ||
        hidden.size() != kDsv4MhcMultiplier * kDsv4MhcHidden ||
        (!weighted.empty() && weighted.size() != kDsv4MhcHidden) ||
        ((!device_only && layer_input.size() != kDsv4MhcHidden) ||
         (device_only && (!weighted.empty() || !layer_input.empty()))) ||
        !std::all_of(hidden.begin(), hidden.end(), [](float value) {
            return std::isfinite(value);
        })) {
        result.errors.emplace_back(
            "DeepSeek device mHC begin request or command order is invalid");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for device mHC begin");
    }
    std::uint64_t allocation_calls = 0U;
    std::uint64_t allocation_bytes = 0U;
    if (state.dsv4_mhc_workspace == nullptr) {
        if (auto status = cudaMalloc(
                &state.dsv4_mhc_workspace, sizeof(Dsv4MhcWorkspace));
            status != cudaSuccess) {
            return cuda_error(status,
                              "allocate DeepSeek device mHC workspace");
        }
        state.dsv4_mhc_workspace_bytes = sizeof(Dsv4MhcWorkspace);
        ++allocation_calls;
        allocation_bytes += sizeof(Dsv4MhcWorkspace);
    }
    if (state.dsv4_mhc_host_staging == nullptr) {
        void* staging = nullptr;
        if (auto status = cudaMallocHost(
                &staging,
                static_cast<std::size_t>(kDsv4MhcMaximumHostStagingBytes));
            status != cudaSuccess) {
            return cuda_error(
                status, "allocate pinned DeepSeek device mHC staging");
        }
        state.dsv4_mhc_host_staging = static_cast<std::byte*>(staging);
        state.dsv4_mhc_host_staging_bytes =
            kDsv4MhcMaximumHostStagingBytes;
        ++allocation_calls;
        allocation_bytes += kDsv4MhcMaximumHostStagingBytes;
    }
    auto* host_hidden = reinterpret_cast<std::uint16_t*>(
        state.dsv4_mhc_host_staging);
    for (std::size_t index = 0U; index < hidden.size(); ++index) {
        host_hidden[index] = bf16_encode(hidden[index]);
    }
    auto* workspace = state.dsv4_mhc_workspace;
    const auto* projection = static_cast<const float*>(
        weights.impl_->projection.impl_->weights);
    const auto* auxiliary = static_cast<const std::byte*>(
        weights.impl_->auxiliary.impl_->data);
    const auto* scale = reinterpret_cast<const float*>(auxiliary);
    const auto* base = reinterpret_cast<const float*>(
        auxiliary + 3U * sizeof(float));
    const auto* norm = reinterpret_cast<const __nv_bfloat16*>(
        auxiliary + kDsv4MhcAuxNormOffset);
    const auto h2d_bytes = hidden.size() * sizeof(std::uint16_t);
    const auto weighted_bytes = weighted.size() * sizeof(std::uint16_t);
    const auto layer_bytes = layer_input.size() * sizeof(std::uint16_t);
    const auto d2h_bytes = weighted_bytes + layer_bytes;
    const auto operation_started = std::chrono::steady_clock::now();
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_start, state.stream);
            status != cudaSuccess) {
            return cuda_error(status,
                              "record DeepSeek device mHC begin upload start");
        }
    }
    if (auto status = cudaMemcpyAsync(
            workspace->residual[0], host_hidden,
            static_cast<std::size_t>(h2d_bytes), cudaMemcpyHostToDevice,
            state.stream); status != cudaSuccess) {
        return cuda_error(status, "upload DeepSeek device mHC residual");
    }
    if (auto status = cudaMemsetAsync(
            &workspace->failure, 0, sizeof(workspace->failure), state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "clear DeepSeek device mHC status");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                state.activation_uploaded, state.stream);
            status != cudaSuccess) {
            return cuda_error(
                status, "record DeepSeek device mHC begin upload completion");
        }
    }
    dsv4_mhc_standalone_projection<<<
        dim3{2U, kDsv4MhcStandaloneSplits}, 32U, 0U, state.stream>>>(
        workspace->residual[0], projection, workspace->partial_projection);
    dsv4_mhc_standalone_square_sum<<<
        kDsv4MhcStandaloneSplits, 8U, 0U, state.stream>>>(
        workspace->residual[0], workspace->partial_square_sum);
    dsv4_mhc_mix<<<1U, 32U, 0U, state.stream>>>(
        workspace->partial_projection, workspace->partial_square_sum,
        scale, base, kDsv4MhcStandaloneSplits, workspace->pre,
        workspace->post, workspace->combination);
    dsv4_mhc_weighted_norm<<<1U, kDsv4MhcWeightedNormThreads, 0U,
                             state.stream>>>(
        workspace->residual[0], workspace->pre, norm,
        workspace->weighted, workspace->layer_input);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch DeepSeek device mHC begin");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                state.kernel_finished, state.stream);
            status != cudaSuccess) {
            return cuda_error(
                status, "record DeepSeek device mHC begin kernels");
        }
    }
    if (device_only) {
        state.dsv4_mhc_stage = 1U;
        state.dsv4_mhc_residual_index = 0U;
        state.dsv4_mhc_branch_ready = false;
        state.dsv4_mhc_failed = false;
        const auto operation_nanoseconds = elapsed_nanoseconds_since(
            operation_started);
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        ++stats.dsv4_mhc_calls;
        ++stats.dsv4_mhc_standalone_calls;
        stats.dsv4_mhc_kernel_launches += 4U;
        stats.dsv4_mhc_h2d_bytes += h2d_bytes;
        stats.dsv4_mhc_nanoseconds += operation_nanoseconds;
        stats.dsv4_mhc_host_nanoseconds += operation_nanoseconds;
        stats.activation_h2d_bytes += h2d_bytes;
        stats.workspace_allocation_calls += allocation_calls;
        stats.workspace_allocation_bytes += allocation_bytes;
        return result;
    }
    if (!weighted.empty()) {
        if (auto status = cudaMemcpyAsync(
                state.dsv4_mhc_host_staging, workspace->weighted,
                weighted_bytes, cudaMemcpyDeviceToHost, state.stream);
            status != cudaSuccess) {
            return cuda_error(
                status, "download DeepSeek device mHC begin weighted state");
        }
    }
    if (auto status = cudaMemcpyAsync(
            state.dsv4_mhc_host_staging + weighted_bytes,
            workspace->layer_input, layer_bytes, cudaMemcpyDeviceToHost,
            state.stream); status != cudaSuccess) {
        return cuda_error(
            status, "download DeepSeek device mHC begin layer input");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                state.activation_downloaded, state.stream);
            status != cudaSuccess) {
            return cuda_error(
                status, "record DeepSeek device mHC begin download");
        }
    }
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaStreamSynchronize(state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "synchronize DeepSeek device mHC begin");
    }
    const auto wait_nanoseconds = elapsed_nanoseconds_since(wait_started);
    const auto operation_nanoseconds = elapsed_nanoseconds_since(
        operation_started);
    const auto* host_output = reinterpret_cast<const std::uint16_t*>(
        state.dsv4_mhc_host_staging);
    const auto decode = [&](std::span<float> target,
                            std::size_t offset) {
        for (std::size_t index = 0U; index < target.size(); ++index) {
            target[index] = std::bit_cast<float>(
                static_cast<std::uint32_t>(host_output[offset + index]) << 16U);
            if (!std::isfinite(target[index])) return false;
        }
        return true;
    };
    if (!decode(weighted, 0U) || !decode(layer_input, weighted.size())) {
        result.errors.emplace_back(
            "DeepSeek device mHC begin produced a non-finite value");
        return result;
    }
    std::uint64_t h2d_nanoseconds = 0U;
    std::uint64_t kernel_nanoseconds = 0U;
    std::uint64_t d2h_nanoseconds = 0U;
    std::uint64_t timing_clamped_samples = 0U;
    if (impl_->detailed_timing) {
        float h2d_ms = 0.0F;
        float kernel_ms = 0.0F;
        float d2h_ms = 0.0F;
        if (cudaEventElapsedTime(&h2d_ms, state.activation_start,
                                 state.activation_uploaded) != cudaSuccess ||
            cudaEventElapsedTime(&kernel_ms, state.activation_uploaded,
                                 state.kernel_finished) != cudaSuccess ||
            cudaEventElapsedTime(&d2h_ms, state.kernel_finished,
                                 state.activation_downloaded) != cudaSuccess) {
            result.errors.emplace_back(
                "measure DeepSeek device mHC begin failed");
            return result;
        }
        h2d_nanoseconds = event_milliseconds_to_nanoseconds(
            h2d_ms, timing_clamped_samples);
        kernel_nanoseconds = event_milliseconds_to_nanoseconds(
            kernel_ms, timing_clamped_samples);
        d2h_nanoseconds = event_milliseconds_to_nanoseconds(
            d2h_ms, timing_clamped_samples);
    }
    state.dsv4_mhc_stage = 1U;
    state.dsv4_mhc_residual_index = 0U;
    state.dsv4_mhc_branch_ready = false;
    state.dsv4_mhc_failed = false;
    const auto call_nanoseconds = elapsed_nanoseconds_since(operation_started);
    {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        ++stats.dsv4_mhc_calls;
        ++stats.dsv4_mhc_standalone_calls;
        stats.dsv4_mhc_kernel_launches += 4U;
        stats.dsv4_mhc_h2d_bytes += h2d_bytes;
        stats.dsv4_mhc_d2h_bytes += d2h_bytes;
        stats.dsv4_mhc_h2d_nanoseconds += h2d_nanoseconds;
        stats.dsv4_mhc_kernel_nanoseconds += kernel_nanoseconds;
        stats.dsv4_mhc_d2h_nanoseconds += d2h_nanoseconds;
        stats.dsv4_mhc_nanoseconds += operation_nanoseconds;
        stats.dsv4_mhc_device_nanoseconds +=
            h2d_nanoseconds + kernel_nanoseconds + d2h_nanoseconds;
        stats.dsv4_mhc_host_nanoseconds +=
            call_nanoseconds > wait_nanoseconds
                ? call_nanoseconds - wait_nanoseconds : 0U;
        stats.dsv4_mhc_timing_clamped_samples += timing_clamped_samples;
        stats.activation_h2d_bytes += h2d_bytes;
        stats.activation_d2h_bytes += d2h_bytes;
        stats.activation_h2d_nanoseconds += h2d_nanoseconds;
        stats.kernel_nanoseconds += kernel_nanoseconds;
        stats.activation_d2h_nanoseconds += d2h_nanoseconds;
        stats.workspace_allocation_calls += allocation_calls;
        stats.workspace_allocation_bytes += allocation_bytes;
        record_synchronization(stats, SynchronizationSubsystem::Mhc,
                               1U, wait_nanoseconds);
    }
    return result;
}

ValidationResult CudaBackend::dsv4_mhc_transition(
    int device, const CudaDsv4MhcWeights& next_weights,
    std::span<const float> branch_output, std::span<float> weighted,
    std::span<float> layer_input, std::span<float> post_residual) {
    return dsv4_mhc_transition_impl(
        device, next_weights, branch_output, weighted, layer_input,
        post_residual, false);
}

ValidationResult CudaBackend::dsv4_mhc_transition_device(
    int device, const CudaDsv4MhcWeights& next_weights,
    std::span<float> weighted, std::span<float> layer_input,
    std::span<float> post_residual) {
    return dsv4_mhc_transition_impl(
        device, next_weights, {}, weighted, layer_input, post_residual, true);
}

ValidationResult CudaBackend::dsv4_mhc_transition_impl(
    int device, const CudaDsv4MhcWeights& next_weights,
    std::span<const float> branch_output, std::span<float> weighted,
    std::span<float> layer_input, std::span<float> post_residual,
    bool device_branch) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek device mHC transition targets an uninitialized device");
        return result;
    }
    auto& state = found->second;
    if (!next_weights.valid() || next_weights.device() != device ||
        !state.dsv4_mhc_supported || state.moe_in_flight ||
        state.dsv4_mhc_stage != 1U ||
        state.dsv4_mhc_branch_ready != device_branch ||
        (device_branch ? !branch_output.empty()
                       : branch_output.size() != kDsv4MhcHidden) ||
        (!weighted.empty() && weighted.size() != kDsv4MhcHidden) ||
        layer_input.size() != kDsv4MhcHidden ||
        (!post_residual.empty() &&
         post_residual.size() != kDsv4MhcMultiplier * kDsv4MhcHidden) ||
        (!device_branch &&
         !std::all_of(branch_output.begin(), branch_output.end(),
                      [](float value) { return std::isfinite(value); }))) {
        result.errors.emplace_back(
            "DeepSeek device mHC transition request or command order is invalid");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(
            status, "select CUDA device for device mHC transition");
    }
    state.dsv4_mhc_branch_ready = false;
    auto* host_branch = reinterpret_cast<std::uint16_t*>(
        state.dsv4_mhc_host_staging);
    if (!device_branch) {
        for (std::size_t index = 0U; index < branch_output.size(); ++index) {
            host_branch[index] = bf16_encode(branch_output[index]);
        }
    }
    auto* workspace = state.dsv4_mhc_workspace;
    const auto current = state.dsv4_mhc_residual_index;
    const auto next = current ^ 1U;
    const auto* projection = static_cast<const float*>(
        next_weights.impl_->projection.impl_->weights);
    const auto* auxiliary = static_cast<const std::byte*>(
        next_weights.impl_->auxiliary.impl_->data);
    const auto* scale = reinterpret_cast<const float*>(auxiliary);
    const auto* base = reinterpret_cast<const float*>(
        auxiliary + 3U * sizeof(float));
    const auto* norm = reinterpret_cast<const __nv_bfloat16*>(
        auxiliary + kDsv4MhcAuxNormOffset);
    const auto h2d_bytes = device_branch
        ? 0U : branch_output.size() * sizeof(std::uint16_t);
    const auto output_elements = weighted.size() + layer_input.size() +
                                 post_residual.size();
    const auto d2h_bytes = output_elements * sizeof(std::uint16_t);
    const auto operation_started = std::chrono::steady_clock::now();
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_start, state.stream);
            status != cudaSuccess) {
            return cuda_error(
                status, "record DeepSeek device mHC transition upload start");
        }
    }
    if (!device_branch) {
        if (auto status = cudaMemcpyAsync(
                workspace->branch, host_branch,
                static_cast<std::size_t>(h2d_bytes), cudaMemcpyHostToDevice,
                state.stream); status != cudaSuccess) {
            return cuda_error(
                status, "upload DeepSeek device mHC branch output");
        }
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                state.activation_uploaded, state.stream);
            status != cudaSuccess) {
            return cuda_error(
                status, "record DeepSeek device mHC transition upload");
        }
    }
    dsv4_mhc_fused_post_projection<<<
        dim3{kDsv4MhcMixes / kDsv4MhcProjectionTile, kDsv4MhcSplits},
        kDsv4MhcProjectionThreads, 0U, state.stream>>>(
        workspace->combination, workspace->residual[current],
        workspace->post, workspace->branch, projection,
        workspace->partial_projection, workspace->partial_square_sum,
        workspace->residual[next]);
    dsv4_mhc_mix<<<1U, 32U, 0U, state.stream>>>(
        workspace->partial_projection, workspace->partial_square_sum,
        scale, base, kDsv4MhcSplits, workspace->pre, workspace->post,
        workspace->combination);
    dsv4_mhc_weighted_norm<<<1U, kDsv4MhcWeightedNormThreads, 0U,
                             state.stream>>>(
        workspace->residual[next], workspace->pre, norm,
        workspace->weighted, workspace->layer_input);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "launch DeepSeek device mHC transition");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                state.kernel_finished, state.stream);
            status != cudaSuccess) {
            state.dsv4_mhc_stage = 0U;
            return cuda_error(
                status, "record DeepSeek device mHC transition kernels");
        }
    }
    const auto weighted_bytes =
        weighted.size() * sizeof(std::uint16_t);
    const auto layer_bytes =
        layer_input.size() * sizeof(std::uint16_t);
    if (!weighted.empty()) {
        if (auto status = cudaMemcpyAsync(
                state.dsv4_mhc_host_staging, workspace->weighted,
                weighted_bytes, cudaMemcpyDeviceToHost, state.stream);
            status != cudaSuccess) {
            state.dsv4_mhc_stage = 0U;
            return cuda_error(
                status,
                "download DeepSeek device mHC transition weighted state");
        }
    }
    if (auto status = cudaMemcpyAsync(
            state.dsv4_mhc_host_staging + weighted_bytes,
            workspace->layer_input, layer_bytes, cudaMemcpyDeviceToHost,
            state.stream); status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(
            status, "download DeepSeek device mHC transition layer input");
    }
    if (!post_residual.empty()) {
        if (auto status = cudaMemcpyAsync(
                state.dsv4_mhc_host_staging + weighted_bytes + layer_bytes,
                workspace->residual[next],
                post_residual.size() * sizeof(std::uint16_t),
                cudaMemcpyDeviceToHost, state.stream);
            status != cudaSuccess) {
            state.dsv4_mhc_stage = 0U;
            return cuda_error(
                status, "download DeepSeek device mHC transition residual");
        }
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                state.activation_downloaded, state.stream);
            status != cudaSuccess) {
            state.dsv4_mhc_stage = 0U;
            return cuda_error(
                status, "record DeepSeek device mHC transition download");
        }
    }
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaStreamSynchronize(state.stream);
        status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(
            status, "synchronize DeepSeek device mHC transition");
    }
    const auto wait_nanoseconds = elapsed_nanoseconds_since(wait_started);
    const auto operation_nanoseconds = elapsed_nanoseconds_since(
        operation_started);
    const auto* host_output = reinterpret_cast<const std::uint16_t*>(
        state.dsv4_mhc_host_staging);
    const auto decode = [&](std::span<float> target,
                            std::size_t offset) {
        for (std::size_t index = 0U; index < target.size(); ++index) {
            target[index] = std::bit_cast<float>(
                static_cast<std::uint32_t>(host_output[offset + index]) << 16U);
            if (!std::isfinite(target[index])) return false;
        }
        return true;
    };
    if (!decode(weighted, 0U) ||
        !decode(layer_input, weighted.size()) ||
        (!post_residual.empty() &&
         !decode(post_residual, weighted.size() + layer_input.size()))) {
        state.dsv4_mhc_stage = 0U;
        result.errors.emplace_back(
            "DeepSeek device mHC transition produced a non-finite value");
        return result;
    }
    state.dsv4_mhc_residual_index = next;
    std::uint64_t h2d_nanoseconds = 0U;
    std::uint64_t kernel_nanoseconds = 0U;
    std::uint64_t d2h_nanoseconds = 0U;
    std::uint64_t timing_clamped_samples = 0U;
    if (impl_->detailed_timing) {
        float h2d_ms = 0.0F;
        float kernel_ms = 0.0F;
        float d2h_ms = 0.0F;
        if (cudaEventElapsedTime(&h2d_ms, state.activation_start,
                                 state.activation_uploaded) != cudaSuccess ||
            cudaEventElapsedTime(&kernel_ms, state.activation_uploaded,
                                 state.kernel_finished) != cudaSuccess ||
            cudaEventElapsedTime(&d2h_ms, state.kernel_finished,
                                 state.activation_downloaded) != cudaSuccess) {
            state.dsv4_mhc_stage = 0U;
            result.errors.emplace_back(
                "measure DeepSeek device mHC transition failed");
            return result;
        }
        h2d_nanoseconds = event_milliseconds_to_nanoseconds(
            h2d_ms, timing_clamped_samples);
        kernel_nanoseconds = event_milliseconds_to_nanoseconds(
            kernel_ms, timing_clamped_samples);
        d2h_nanoseconds = event_milliseconds_to_nanoseconds(
            d2h_ms, timing_clamped_samples);
    }
    const auto call_nanoseconds = elapsed_nanoseconds_since(operation_started);
    {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        ++stats.dsv4_mhc_calls;
        ++stats.dsv4_mhc_transition_calls;
        stats.dsv4_mhc_kernel_launches += 3U;
        stats.dsv4_mhc_h2d_bytes += h2d_bytes;
        stats.dsv4_mhc_d2h_bytes += d2h_bytes;
        stats.dsv4_mhc_h2d_nanoseconds += h2d_nanoseconds;
        stats.dsv4_mhc_kernel_nanoseconds += kernel_nanoseconds;
        stats.dsv4_mhc_d2h_nanoseconds += d2h_nanoseconds;
        stats.dsv4_mhc_nanoseconds += operation_nanoseconds;
        stats.dsv4_mhc_device_nanoseconds +=
            h2d_nanoseconds + kernel_nanoseconds + d2h_nanoseconds;
        stats.dsv4_mhc_host_nanoseconds +=
            call_nanoseconds > wait_nanoseconds
                ? call_nanoseconds - wait_nanoseconds : 0U;
        stats.dsv4_mhc_timing_clamped_samples += timing_clamped_samples;
        stats.activation_h2d_bytes += h2d_bytes;
        stats.activation_d2h_bytes += d2h_bytes;
        stats.activation_h2d_nanoseconds += h2d_nanoseconds;
        stats.kernel_nanoseconds += kernel_nanoseconds;
        stats.activation_d2h_nanoseconds += d2h_nanoseconds;
        record_synchronization(stats, SynchronizationSubsystem::Mhc,
                               1U, wait_nanoseconds);
    }
    return result;
}

ValidationResult CudaBackend::dsv4_mhc_finish(
    int device, std::span<const float> branch_output,
    std::span<float> hidden) {
    return dsv4_mhc_finish_impl(device, branch_output, hidden, false);
}

ValidationResult CudaBackend::dsv4_mhc_finish_device(
    int device, std::span<float> hidden) {
    return dsv4_mhc_finish_impl(device, {}, hidden, true);
}

ValidationResult CudaBackend::dsv4_mhc_device_view(
    int device, CudaDsv4MhcDeviceView& view) {
    view = {};
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        return {{"DeepSeek device mHC view targets an uninitialized device"}};
    }
    auto& state = found->second;
    if (!state.dsv4_mhc_supported || state.dsv4_mhc_stage != 1U ||
        state.dsv4_mhc_workspace == nullptr || state.moe_in_flight ||
        state.dsv4_mhc_failed) {
        return {{"DeepSeek device mHC view violates command order"}};
    }
    view.stream = state.stream;
    view.weighted = reinterpret_cast<std::uint16_t*>(
        state.dsv4_mhc_workspace->weighted);
    view.layer_input = reinterpret_cast<std::uint16_t*>(
        state.dsv4_mhc_workspace->layer_input);
    view.branch = reinterpret_cast<std::uint16_t*>(
        state.dsv4_mhc_workspace->branch);
    view.residual = reinterpret_cast<std::uint16_t*>(
        state.dsv4_mhc_workspace->residual[state.dsv4_mhc_residual_index]);
    view.router_logits = state.dsv4_mhc_workspace->router_logits;
    view.status = &state.dsv4_mhc_workspace->failure;
    if (view.stream == nullptr || view.weighted == nullptr ||
        view.layer_input == nullptr ||
        view.branch == nullptr || view.residual == nullptr ||
        view.router_logits == nullptr ||
        view.status == nullptr) {
        view = {};
        return {{"DeepSeek device mHC view is incomplete"}};
    }
    return {};
}

namespace {

bool dsv4_validate_device_pointer(
    int device, const void* pointer, const char* name,
    ValidationResult& result) {
    if (pointer == nullptr) {
        result.errors.emplace_back(std::string(name) + " is null");
        return false;
    }
    cudaPointerAttributes attributes{};
    if (const auto status = cudaPointerGetAttributes(&attributes, pointer);
        status != cudaSuccess) {
        result.errors.emplace_back(std::string(name) +
                                   " is not a live CUDA pointer");
        static_cast<void>(cudaGetLastError());
        return false;
    }
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device) {
        result.errors.emplace_back(std::string(name) +
                                   " is not resident on the requested device");
        return false;
    }
    return true;
}

}  // namespace

ValidationResult CudaBackend::dsv4_mhc_branch_to_fp32(
    int device, float* output) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek mHC branch conversion targets an uninitialized device");
        return result;
    }
    auto& state = found->second;
    if (!state.dsv4_mhc_supported || state.dsv4_mhc_stage != 1U ||
        state.dsv4_mhc_workspace == nullptr || state.dsv4_mhc_branch_ready ||
        state.moe_in_flight || state.dsv4_mhc_failed ||
        !dsv4_validate_device_pointer(device, output, "mHC FP32 branch output",
                                      result)) {
        if (result.errors.empty()) {
            result.errors.emplace_back(
                "DeepSeek mHC branch conversion violates command order");
        }
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for mHC branch conversion");
    }
    constexpr std::uint32_t threads = 256U;
    constexpr std::uint32_t blocks =
        (kDsv4MhcHidden + threads - 1U) / threads;
    dsv4_bf16_to_fp32<<<blocks, threads, 0U, state.stream>>>(
        state.dsv4_mhc_workspace->branch, output, kDsv4MhcHidden);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch mHC branch conversion");
    }
    return result;
}

ValidationResult CudaBackend::dsv4_mhc_commit_reduced_branch(
    int device, const float* reduced) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek mHC branch commit targets an uninitialized device");
        return result;
    }
    auto& state = found->second;
    if (!state.dsv4_mhc_supported || state.dsv4_mhc_stage != 1U ||
        state.dsv4_mhc_workspace == nullptr || state.dsv4_mhc_branch_ready ||
        state.dsv4_mhc_failed || state.dsv4_host_moe_input_pending ||
        !dsv4_validate_device_pointer(device, reduced, "reduced mHC branch",
                                      result)) {
        if (result.errors.empty()) {
            result.errors.emplace_back(
                "DeepSeek mHC branch commit violates command order");
        }
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for mHC branch commit");
    }
    constexpr std::uint32_t threads = 256U;
    constexpr std::uint32_t blocks =
        (kDsv4MhcHidden + threads - 1U) / threads;
    dsv4_fp32_to_bf16<<<blocks, threads, 0U, state.stream>>>(
        reduced, state.dsv4_mhc_workspace->branch, kDsv4MhcHidden);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch mHC reduced branch publication");
    }
    state.dsv4_mhc_branch_ready = true;
    return result;
}

ValidationResult CudaBackend::dsv4_mhc_abort_branch(int device) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek mHC branch abort targets an uninitialized device");
        return result;
    }
    auto& state = found->second;
    if (!state.dsv4_mhc_supported || state.dsv4_mhc_stage != 1U ||
        state.dsv4_mhc_workspace == nullptr || state.moe_in_flight) {
        result.errors.emplace_back(
            "DeepSeek mHC branch abort violates command order");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for mHC branch abort");
    }
    auto* workspace = state.dsv4_mhc_workspace;
    if (auto status = cudaMemsetAsync(
            workspace->branch, 0, sizeof(workspace->branch), state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "clear failed mHC branch");
    }
    if (auto status = cudaMemsetAsync(
            workspace->weighted, 0, sizeof(workspace->weighted), state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "clear failed mHC weighted state");
    }
    if (auto status = cudaMemsetAsync(
            workspace->layer_input, 0, sizeof(workspace->layer_input),
            state.stream); status != cudaSuccess) {
        return cuda_error(status, "clear failed mHC layer input");
    }
    if (auto status = cudaMemsetAsync(
            workspace->residual, 0, sizeof(workspace->residual), state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "clear failed mHC residual state");
    }
    if (auto status = cudaMemsetAsync(
            &workspace->failure, 1, sizeof(workspace->failure), state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "mark failed mHC state");
    }
    state.dsv4_mhc_stage = 0U;
    state.dsv4_mhc_residual_index = 0U;
    state.dsv4_mhc_branch_ready = false;
    state.dsv4_mhc_failed = true;
    state.dsv4_host_moe_input_pending = false;
    state.dsv4_host_moe_router_logits = nullptr;
    state.dsv4_host_moe_device_failure = nullptr;
    state.dsv4_host_moe_host_failure = nullptr;
    return result;
}

ValidationResult CudaBackend::dsv4_mhc_transition_router_device(
    int device, const CudaDsv4MhcWeights& next_weights,
    const CudaWeight& router) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek mHC router transition targets an uninitialized device");
        return result;
    }
    auto& state = found->second;
    if (!next_weights.valid() || next_weights.device() != device ||
        !router.valid() || router.device() != device ||
        !state.dsv4_mhc_supported || state.dsv4_mhc_stage != 1U ||
        state.dsv4_mhc_workspace == nullptr || !state.dsv4_mhc_branch_ready ||
        state.dsv4_host_moe_input_pending || state.moe_in_flight ||
        state.dsv4_mhc_failed) {
        result.errors.emplace_back(
            "DeepSeek mHC router transition violates command order or ownership");
        return result;
    }
    const auto& descriptor = router.impl_->descriptor;
    if (descriptor.encoding != CudaWeightEncoding::Plain ||
        descriptor.dtype != SafetensorsDtype::Bf16 ||
        descriptor.rows != kDsv4MhcRouterLogits ||
        descriptor.columns != kDsv4MhcHidden) {
        result.errors.emplace_back(
            "DeepSeek mHC router transition requires a 256x4096 BF16 router");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for mHC router transition");
    }
    auto* workspace = state.dsv4_mhc_workspace;
    const auto current = state.dsv4_mhc_residual_index;
    const auto next = current ^ 1U;
    const auto* projection = static_cast<const float*>(
        next_weights.impl_->projection.impl_->weights);
    const auto* auxiliary = static_cast<const std::byte*>(
        next_weights.impl_->auxiliary.impl_->data);
    const auto* scale = reinterpret_cast<const float*>(auxiliary);
    const auto* base = reinterpret_cast<const float*>(
        auxiliary + 3U * sizeof(float));
    const auto* norm = reinterpret_cast<const __nv_bfloat16*>(
        auxiliary + kDsv4MhcAuxNormOffset);
    dsv4_mhc_fused_post_projection<<<
        dim3{kDsv4MhcMixes / kDsv4MhcProjectionTile, kDsv4MhcSplits},
        kDsv4MhcProjectionThreads, 0U, state.stream>>>(
        workspace->combination, workspace->residual[current], workspace->post,
        workspace->branch, projection, workspace->partial_projection,
        workspace->partial_square_sum, workspace->residual[next]);
    dsv4_mhc_mix<<<1U, 32U, 0U, state.stream>>>(
        workspace->partial_projection, workspace->partial_square_sum, scale,
        base, kDsv4MhcSplits, workspace->pre, workspace->post,
        workspace->combination);
    dsv4_mhc_weighted_norm<<<1U, kDsv4MhcWeightedNormThreads, 0U,
                             state.stream>>>(
        workspace->residual[next], workspace->pre, norm, workspace->weighted,
        workspace->layer_input);
    constexpr unsigned int threads = 256U;
    const auto blocks = static_cast<unsigned int>(
        (kDsv4MhcRouterLogits + (threads / 32U) - 1U) / (threads / 32U));
    bf16_input_matvec_kernel<<<blocks, threads, 0U, state.stream>>>(
        workspace->router_logits, workspace->layer_input,
        static_cast<const __nv_bfloat16*>(router.impl_->weights),
        descriptor.columns, descriptor.rows);
    if (auto status = cudaMemsetAsync(
            workspace->branch, 0, sizeof(workspace->branch), state.stream);
        status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "clear consumed mHC attention branch");
    }
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "launch mHC router transition");
    }
    state.dsv4_mhc_residual_index = next;
    state.dsv4_mhc_branch_ready = false;
    state.dsv4_host_moe_input_pending = true;
    state.dsv4_host_moe_router_logits = workspace->router_logits;
    state.dsv4_host_moe_device_failure = &workspace->failure;
    state.dsv4_host_moe_host_failure = nullptr;
    return result;
}

ValidationResult CudaBackend::dsv4_mhc_transition_next_device(
    int device, const CudaDsv4MhcWeights& next_weights) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek final mHC transition targets an uninitialized device");
        return result;
    }
    auto& state = found->second;
    if (!next_weights.valid() || next_weights.device() != device ||
        !state.dsv4_mhc_supported || state.dsv4_mhc_stage != 1U ||
        state.dsv4_mhc_workspace == nullptr || !state.dsv4_mhc_branch_ready ||
        state.dsv4_host_moe_input_pending || state.moe_in_flight ||
        state.dsv4_mhc_failed) {
        result.errors.emplace_back(
            "DeepSeek final mHC transition violates command order");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for final mHC transition");
    }
    auto* workspace = state.dsv4_mhc_workspace;
    const auto current = state.dsv4_mhc_residual_index;
    const auto next = current ^ 1U;
    const auto* projection = static_cast<const float*>(
        next_weights.impl_->projection.impl_->weights);
    const auto* auxiliary = static_cast<const std::byte*>(
        next_weights.impl_->auxiliary.impl_->data);
    const auto* scale = reinterpret_cast<const float*>(auxiliary);
    const auto* base = reinterpret_cast<const float*>(
        auxiliary + 3U * sizeof(float));
    const auto* norm = reinterpret_cast<const __nv_bfloat16*>(
        auxiliary + kDsv4MhcAuxNormOffset);
    dsv4_mhc_fused_post_projection<<<
        dim3{kDsv4MhcMixes / kDsv4MhcProjectionTile, kDsv4MhcSplits},
        kDsv4MhcProjectionThreads, 0U, state.stream>>>(
        workspace->combination, workspace->residual[current], workspace->post,
        workspace->branch, projection, workspace->partial_projection,
        workspace->partial_square_sum, workspace->residual[next]);
    dsv4_mhc_mix<<<1U, 32U, 0U, state.stream>>>(
        workspace->partial_projection, workspace->partial_square_sum, scale,
        base, kDsv4MhcSplits, workspace->pre, workspace->post,
        workspace->combination);
    dsv4_mhc_weighted_norm<<<1U, kDsv4MhcWeightedNormThreads, 0U,
                             state.stream>>>(
        workspace->residual[next], workspace->pre, norm, workspace->weighted,
        workspace->layer_input);
    if (auto status = cudaMemsetAsync(
            workspace->branch, 0, sizeof(workspace->branch), state.stream);
        status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "clear consumed mHC MoE branch");
    }
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "launch final mHC transition");
    }
    state.dsv4_mhc_residual_index = next;
    state.dsv4_mhc_branch_ready = false;
    return result;
}

ValidationResult CudaBackend::dsv4_mhc_finish_impl(
    int device, std::span<const float> branch_output,
    std::span<float> hidden, bool device_branch) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek device mHC finish targets an uninitialized device");
        return result;
    }
    auto& state = found->second;
    if (!state.dsv4_mhc_supported || state.moe_in_flight ||
        state.dsv4_mhc_stage != 1U ||
        state.dsv4_mhc_branch_ready != device_branch ||
        (device_branch ? !branch_output.empty()
                       : branch_output.size() != kDsv4MhcHidden) ||
        hidden.size() != kDsv4MhcMultiplier * kDsv4MhcHidden ||
        (!device_branch &&
         !std::all_of(branch_output.begin(), branch_output.end(),
                      [](float value) { return std::isfinite(value); }))) {
        result.errors.emplace_back(
            "DeepSeek device mHC finish request or command order is invalid");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(
            status, "select CUDA device for device mHC finish");
    }
    state.dsv4_mhc_branch_ready = false;
    auto* host_branch = reinterpret_cast<std::uint16_t*>(
        state.dsv4_mhc_host_staging);
    if (!device_branch) {
        for (std::size_t index = 0U; index < branch_output.size(); ++index) {
            host_branch[index] = bf16_encode(branch_output[index]);
        }
    }
    auto* workspace = state.dsv4_mhc_workspace;
    const auto current = state.dsv4_mhc_residual_index;
    const auto next = current ^ 1U;
    const auto h2d_bytes = device_branch
        ? 0U : branch_output.size() * sizeof(std::uint16_t);
    const auto d2h_bytes = hidden.size() * sizeof(std::uint16_t);
    const auto operation_started = std::chrono::steady_clock::now();
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_start, state.stream);
            status != cudaSuccess) {
            return cuda_error(
                status, "record DeepSeek device mHC finish upload start");
        }
    }
    if (!device_branch) {
        if (auto status = cudaMemcpyAsync(
                workspace->branch, host_branch,
                static_cast<std::size_t>(h2d_bytes), cudaMemcpyHostToDevice,
                state.stream); status != cudaSuccess) {
            return cuda_error(
                status, "upload DeepSeek device mHC final branch output");
        }
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                state.activation_uploaded, state.stream);
            status != cudaSuccess) {
            return cuda_error(
                status, "record DeepSeek device mHC finish upload");
        }
    }
    constexpr std::uint32_t threads = 256U;
    constexpr std::uint32_t blocks =
        (kDsv4MhcHidden + threads - 1U) / threads;
    dsv4_mhc_final_post<<<blocks, threads, 0U, state.stream>>>(
        workspace->combination, workspace->residual[current],
        workspace->post, workspace->branch, workspace->residual[next]);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "launch DeepSeek device mHC finish");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                state.kernel_finished, state.stream);
            status != cudaSuccess) {
            state.dsv4_mhc_stage = 0U;
            return cuda_error(
                status, "record DeepSeek device mHC finish kernel");
        }
    }
    if (auto status = cudaMemcpyAsync(
            state.dsv4_mhc_host_staging, workspace->residual[next],
            static_cast<std::size_t>(d2h_bytes), cudaMemcpyDeviceToHost,
            state.stream); status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(
            status, "download DeepSeek device mHC final residual");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(
                state.activation_downloaded, state.stream);
            status != cudaSuccess) {
            state.dsv4_mhc_stage = 0U;
            return cuda_error(
                status, "record DeepSeek device mHC finish download");
        }
    }
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaStreamSynchronize(state.stream);
        status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(
            status, "synchronize DeepSeek device mHC finish");
    }
    const auto wait_nanoseconds = elapsed_nanoseconds_since(wait_started);
    const auto operation_nanoseconds = elapsed_nanoseconds_since(
        operation_started);
    const auto* host_output = reinterpret_cast<const std::uint16_t*>(
        state.dsv4_mhc_host_staging);
    for (std::size_t index = 0U; index < hidden.size(); ++index) {
        hidden[index] = std::bit_cast<float>(
            static_cast<std::uint32_t>(host_output[index]) << 16U);
        if (!std::isfinite(hidden[index])) {
            state.dsv4_mhc_stage = 0U;
            result.errors.emplace_back(
                "DeepSeek device mHC finish produced a non-finite value");
            return result;
        }
    }
    state.dsv4_mhc_stage = 0U;
    state.dsv4_mhc_residual_index = next;
    std::uint64_t h2d_nanoseconds = 0U;
    std::uint64_t kernel_nanoseconds = 0U;
    std::uint64_t d2h_nanoseconds = 0U;
    std::uint64_t timing_clamped_samples = 0U;
    if (impl_->detailed_timing) {
        float h2d_ms = 0.0F;
        float kernel_ms = 0.0F;
        float d2h_ms = 0.0F;
        if (cudaEventElapsedTime(&h2d_ms, state.activation_start,
                                 state.activation_uploaded) != cudaSuccess ||
            cudaEventElapsedTime(&kernel_ms, state.activation_uploaded,
                                 state.kernel_finished) != cudaSuccess ||
            cudaEventElapsedTime(&d2h_ms, state.kernel_finished,
                                 state.activation_downloaded) != cudaSuccess) {
            result.errors.emplace_back(
                "measure DeepSeek device mHC finish failed");
            return result;
        }
        h2d_nanoseconds = event_milliseconds_to_nanoseconds(
            h2d_ms, timing_clamped_samples);
        kernel_nanoseconds = event_milliseconds_to_nanoseconds(
            kernel_ms, timing_clamped_samples);
        d2h_nanoseconds = event_milliseconds_to_nanoseconds(
            d2h_ms, timing_clamped_samples);
    }
    const auto call_nanoseconds = elapsed_nanoseconds_since(operation_started);
    {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        ++stats.dsv4_mhc_calls;
        ++stats.dsv4_mhc_final_calls;
        ++stats.dsv4_mhc_kernel_launches;
        stats.dsv4_mhc_h2d_bytes += h2d_bytes;
        stats.dsv4_mhc_d2h_bytes += d2h_bytes;
        stats.dsv4_mhc_h2d_nanoseconds += h2d_nanoseconds;
        stats.dsv4_mhc_kernel_nanoseconds += kernel_nanoseconds;
        stats.dsv4_mhc_d2h_nanoseconds += d2h_nanoseconds;
        stats.dsv4_mhc_nanoseconds += operation_nanoseconds;
        stats.dsv4_mhc_device_nanoseconds +=
            h2d_nanoseconds + kernel_nanoseconds + d2h_nanoseconds;
        stats.dsv4_mhc_host_nanoseconds +=
            call_nanoseconds > wait_nanoseconds
                ? call_nanoseconds - wait_nanoseconds : 0U;
        stats.dsv4_mhc_timing_clamped_samples += timing_clamped_samples;
        stats.activation_h2d_bytes += h2d_bytes;
        stats.activation_d2h_bytes += d2h_bytes;
        stats.activation_h2d_nanoseconds += h2d_nanoseconds;
        stats.kernel_nanoseconds += kernel_nanoseconds;
        stats.activation_d2h_nanoseconds += d2h_nanoseconds;
        record_synchronization(stats, SynchronizationSubsystem::Mhc,
                               1U, wait_nanoseconds);
    }
    return result;
}

ValidationResult CudaBackend::reserve_dsv4_mhc_head(
    int device, std::uint64_t logits) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end() || logits == 0U) {
        result.errors.emplace_back(
            "DeepSeek device output-head reservation is invalid");
        return result;
    }
    auto& state = found->second;
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for output-head reserve");
    }
    constexpr std::uint64_t hidden_bytes =
        kDsv4MhcMultiplier * kDsv4MhcHidden * sizeof(std::uint16_t);
    constexpr std::uint64_t input_bytes =
        kDsv4MhcHidden * sizeof(float);
    const auto output_bytes = logits * sizeof(float);
    const auto host_bytes = hidden_bytes + input_bytes + output_bytes;
    if (state.dsv4_mhc_head_input == nullptr) {
        if (auto status = cudaMalloc(
                &state.dsv4_mhc_head_input,
                static_cast<std::size_t>(input_bytes));
            status != cudaSuccess) {
            return cuda_error(status, "allocate DeepSeek output-head input");
        }
        state.dsv4_mhc_head_input_bytes = input_bytes;
    }
    // A reservation is a capacity, not a shape. One device can serve two head
    // shapes: centralized prefill projects the full vocabulary while rank-local
    // decode projects one rank's row shard, and under the rank-local opt-in
    // both are resident on the same device. Growing is safe because the
    // staging layout is offset-addressed from a fixed prefix; shrinking is a
    // no-op that keeps the larger buffer. What must still match exactly is the
    // enqueue against its own completion, which is pinned separately below.
    if (state.dsv4_mhc_head_in_flight) {
        result.errors.emplace_back(
            "DeepSeek output-head reservation cannot change while a head is "
            "in flight");
        return result;
    }
    if (state.dsv4_mhc_head_output_bytes < output_bytes) {
        if (state.dsv4_mhc_head_output != nullptr) {
            static_cast<void>(cudaFree(state.dsv4_mhc_head_output));
            state.dsv4_mhc_head_output = nullptr;
            state.dsv4_mhc_head_output_bytes = 0U;
        }
        if (auto status = cudaMalloc(
                &state.dsv4_mhc_head_output,
                static_cast<std::size_t>(output_bytes));
            status != cudaSuccess) {
            return cuda_error(status, "allocate DeepSeek output-head output");
        }
        state.dsv4_mhc_head_output_bytes = output_bytes;
    }
    if (state.dsv4_mhc_head_host_staging_bytes < host_bytes) {
        if (state.dsv4_mhc_head_host_staging != nullptr) {
            static_cast<void>(
                cudaFreeHost(state.dsv4_mhc_head_host_staging));
            state.dsv4_mhc_head_host_staging = nullptr;
            state.dsv4_mhc_head_host_staging_bytes = 0U;
        }
        void* staging = nullptr;
        if (auto status = cudaMallocHost(
                &staging, static_cast<std::size_t>(host_bytes));
            status != cudaSuccess) {
            return cuda_error(
                status, "allocate pinned DeepSeek output-head staging");
        }
        state.dsv4_mhc_head_host_staging = static_cast<std::byte*>(staging);
        state.dsv4_mhc_head_host_staging_bytes = host_bytes;
    }
    if (state.dsv4_mhc_head_input_bytes != input_bytes ||
        state.dsv4_mhc_head_output_bytes < output_bytes ||
        state.dsv4_mhc_head_host_staging_bytes < host_bytes) {
        result.errors.emplace_back(
            "DeepSeek output-head reservation is smaller than the head it "
            "must serve");
        return result;
    }
    return result;
}

ValidationResult CudaBackend::enqueue_dsv4_mhc_finish_head_device(
    int device, const CudaWeight& head,
    CudaDsv4MhcHeadCallback callback, void* callback_context,
    CudaDsv4MhcHeadDeviceView* view) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end() || !head.valid() ||
        head.impl_->device != device) {
        result.errors.emplace_back(
            "DeepSeek device output-head enqueue is invalid");
        return result;
    }
    auto& state = found->second;
    const auto& descriptor = head.impl_->descriptor;
    constexpr std::uint64_t hidden_bytes =
        kDsv4MhcMultiplier * kDsv4MhcHidden * sizeof(std::uint16_t);
    constexpr std::uint64_t input_bytes =
        kDsv4MhcHidden * sizeof(float);
    const auto output_bytes = descriptor.rows * sizeof(float);
    if (!state.dsv4_mhc_supported || state.dsv4_mhc_stage != 1U ||
        !state.dsv4_mhc_branch_ready || state.dsv4_mhc_head_in_flight ||
        callback == nullptr || callback_context == nullptr ||
        descriptor.columns != kDsv4MhcHidden ||
        state.dsv4_mhc_head_input_bytes != input_bytes ||
        state.dsv4_mhc_head_output_bytes < output_bytes ||
        state.dsv4_mhc_head_host_staging_bytes <
            hidden_bytes + input_bytes + output_bytes) {
        result.errors.emplace_back(
            "DeepSeek device output-head command order or shape is invalid");
        return result;
    }
    // This head's own logit extent, so its completion cannot accept a span
    // sized for the other head shape resident on this device.
    state.dsv4_mhc_head_logits_bytes = output_bytes;
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for output-head enqueue");
    }
    auto* workspace = state.dsv4_mhc_workspace;
    const auto current = state.dsv4_mhc_residual_index;
    const auto next = current ^ 1U;
    constexpr std::uint32_t threads = 256U;
    constexpr std::uint32_t blocks =
        (kDsv4MhcHidden + threads - 1U) / threads;
    dsv4_mhc_final_post<<<blocks, threads, 0U, state.stream>>>(
        workspace->combination, workspace->residual[current],
        workspace->post, workspace->branch, workspace->residual[next]);
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "launch DeepSeek output-head final mHC post");
    }
    auto* host_hidden = reinterpret_cast<std::uint16_t*>(
        state.dsv4_mhc_head_host_staging);
    auto* host_reduced = reinterpret_cast<float*>(
        state.dsv4_mhc_head_host_staging + hidden_bytes);
    auto* host_logits = reinterpret_cast<float*>(
        state.dsv4_mhc_head_host_staging + hidden_bytes + input_bytes);
    if (auto status = cudaMemcpyAsync(
            host_hidden, workspace->residual[next], hidden_bytes,
            cudaMemcpyDeviceToHost, state.stream); status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "stage DeepSeek final mHC residual");
    }
    state.dsv4_mhc_head_callback = {
        callback, callback_context, host_hidden, host_reduced, false};
    if (auto status = cudaLaunchHostFunc(
            state.stream, run_dsv4_mhc_head_callback,
            &state.dsv4_mhc_head_callback); status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "enqueue DeepSeek output-head host reduction");
    }
    if (auto status = cudaMemcpyAsync(
            state.dsv4_mhc_head_input, host_reduced, input_bytes,
            cudaMemcpyHostToDevice, state.stream); status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "upload DeepSeek output-head input");
    }
    const bool native =
        descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128 ||
        descriptor.encoding == CudaWeightEncoding::Fp4E2m1Group32;
    if (native) {
        const dim3 quantize_grid(
            static_cast<unsigned int>((descriptor.columns + 127U) / 128U),
            1U, 1U);
        quantize_activation_e4m3_kernel<<<
            quantize_grid, 128U, 0U, state.stream>>>(
            state.dsv4_mhc_head_input, descriptor.columns, 1U);
    }
    const dim3 grid(static_cast<unsigned int>(descriptor.rows), 1U, 1U);
    if (descriptor.encoding == CudaWeightEncoding::Plain &&
        descriptor.dtype == SafetensorsDtype::Bf16) {
        constexpr unsigned int warps_per_block = threads / 32U;
        const auto matvec_blocks = static_cast<unsigned int>(
            (descriptor.rows + warps_per_block - 1U) / warps_per_block);
        bf16_matvec_kernel<<<matvec_blocks, threads, 0U, state.stream>>>(
            state.dsv4_mhc_head_output, state.dsv4_mhc_head_input,
            static_cast<const __nv_bfloat16*>(head.impl_->weights),
            descriptor.columns, descriptor.rows);
    } else if (descriptor.encoding == CudaWeightEncoding::Plain) {
        plain_matmul_kernel<1U><<<grid, threads, 0U, state.stream>>>(
            state.dsv4_mhc_head_output, state.dsv4_mhc_head_input,
            head.impl_->weights, static_cast<int>(descriptor.dtype), 1U,
            descriptor.columns, descriptor.rows, 0U, 0U);
    } else if (descriptor.encoding == CudaWeightEncoding::OffsetPackedInt8 &&
               descriptor.group_size == 32U &&
               descriptor.columns % 32U == 0U) {
        constexpr unsigned int warps_per_block = threads / 32U;
        const auto matvec_blocks = static_cast<unsigned int>(
            (descriptor.rows + warps_per_block - 1U) / warps_per_block);
        packed_int8_group32_matvec_kernel<<<
            matvec_blocks, threads, 0U, state.stream>>>(
            state.dsv4_mhc_head_output, state.dsv4_mhc_head_input,
            static_cast<const std::uint32_t*>(head.impl_->weights),
            static_cast<const __nv_bfloat16*>(head.impl_->scales),
            descriptor.packed_columns, descriptor.scale_columns,
            descriptor.columns, descriptor.rows);
    } else if (descriptor.encoding == CudaWeightEncoding::OffsetPackedInt4 ||
               descriptor.encoding == CudaWeightEncoding::OffsetPackedInt8) {
        const auto bits = descriptor.encoding ==
            CudaWeightEncoding::OffsetPackedInt4 ? 4U : 8U;
        packed_matmul_kernel<<<grid, threads, 0U, state.stream>>>(
            state.dsv4_mhc_head_output, state.dsv4_mhc_head_input,
            static_cast<const std::uint32_t*>(head.impl_->weights),
            static_cast<const __nv_bfloat16*>(head.impl_->scales), bits,
            descriptor.group_size, descriptor.packed_columns,
            descriptor.scale_columns, 1U, descriptor.columns,
            descriptor.rows, 0U, 0U);
    } else if (descriptor.encoding == CudaWeightEncoding::Nvfp4Group16) {
        nvfp4_group16_matmul_kernel<<<grid, threads, 0U, state.stream>>>(
            state.dsv4_mhc_head_output, state.dsv4_mhc_head_input,
            static_cast<const unsigned char*>(head.impl_->weights),
            static_cast<const unsigned char*>(head.impl_->scales),
            descriptor.global_scale, descriptor.packed_columns,
            descriptor.scale_columns, descriptor.group_size, 1U,
            descriptor.columns, descriptor.rows, 0U, 0U);
    } else if (descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128) {
        native_fp8_matmul_kernel<<<grid, threads, 0U, state.stream>>>(
            state.dsv4_mhc_head_output, state.dsv4_mhc_head_input,
            static_cast<const unsigned char*>(head.impl_->weights),
            static_cast<const unsigned char*>(head.impl_->scales),
            descriptor.scale_columns, 1U, descriptor.columns,
            descriptor.rows, 0U, 0U);
    } else {
        native_fp4_matmul_kernel<<<grid, threads, 0U, state.stream>>>(
            state.dsv4_mhc_head_output, state.dsv4_mhc_head_input,
            static_cast<const unsigned char*>(head.impl_->weights),
            static_cast<const unsigned char*>(head.impl_->scales),
            descriptor.packed_columns, descriptor.scale_columns, 1U,
            descriptor.columns, descriptor.rows, 0U, 0U);
    }
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "launch DeepSeek output-head projection");
    }
    if (auto status = cudaMemcpyAsync(
            host_logits, state.dsv4_mhc_head_output, output_bytes,
            cudaMemcpyDeviceToHost, state.stream); status != cudaSuccess) {
        state.dsv4_mhc_stage = 0U;
        return cuda_error(status, "stage DeepSeek output-head logits");
    }
    state.dsv4_mhc_stage = 0U;
    state.dsv4_mhc_residual_index = next;
    state.dsv4_mhc_branch_ready = false;
    state.dsv4_mhc_head_in_flight = true;
    if (view != nullptr) {
        *view = {state.stream, state.dsv4_mhc_head_output,
                 reinterpret_cast<std::uint16_t*>(workspace->residual[next]),
                 descriptor.rows};
    }
    {
        std::scoped_lock lock(impl_->mutex);
        auto& stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        ++stats.dsv4_mhc_calls;
        ++stats.dsv4_mhc_final_calls;
        ++stats.dsv4_mhc_kernel_launches;
        stats.dsv4_mhc_d2h_bytes += hidden_bytes;
        stats.activation_h2d_bytes += input_bytes;
        stats.activation_d2h_bytes += hidden_bytes + output_bytes;
        ++stats.matmul_calls;
    }
    return result;
}

ValidationResult CudaBackend::complete_dsv4_mhc_head_device(
    int device, std::span<float> logits) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek output-head completion targets an invalid device");
        return result;
    }
    auto& state = found->second;
    const auto output_bytes = logits.size_bytes();
    constexpr std::uint64_t hidden_bytes =
        kDsv4MhcMultiplier * kDsv4MhcHidden * sizeof(std::uint16_t);
    constexpr std::uint64_t input_bytes =
        kDsv4MhcHidden * sizeof(float);
    if (!state.dsv4_mhc_head_in_flight ||
        output_bytes != state.dsv4_mhc_head_logits_bytes) {
        result.errors.emplace_back(
            "DeepSeek output-head completion shape or order is invalid");
        return result;
    }
    state.dsv4_mhc_head_in_flight = false;
    if (state.dsv4_mhc_head_callback.failed) {
        result.errors.emplace_back(
            "DeepSeek output-head host reduction failed");
        return result;
    }
    const auto* host_logits = reinterpret_cast<const float*>(
        state.dsv4_mhc_head_host_staging + hidden_bytes + input_bytes);
    std::memcpy(logits.data(), host_logits, output_bytes);
    if (!std::all_of(logits.begin(), logits.end(), [](float value) {
            return std::isfinite(value);
        })) {
        result.errors.emplace_back(
            "DeepSeek output-head projection produced a non-finite value");
    }
    return result;
}

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

ValidationResult CudaBackend::prepack_fragment(int device,
                                               const CudaWeight& weight) {
    ValidationResult result;
    if (!weight.valid()) {
        result.errors.emplace_back("fragment prepack received an invalid weight");
        return result;
    }
    const auto& descriptor = weight.impl_->descriptor;
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
        case CudaMatmulRoute::Fp8E4m3Block128: return "fp8_e4m3_block128";
        case CudaMatmulRoute::Fp4E2m1Group32: return "fp4_e2m1_group32";
        case CudaMatmulRoute::Fp8RegisterFed: return "fp8_register_fed";
        case CudaMatmulRoute::Fp4RegisterFed: return "fp4_register_fed";
        case CudaMatmulRoute::MoePlainBf16: return "moe_plain_bf16";
        case CudaMatmulRoute::MoeNvfp4Group16: return "moe_nvfp4_group16";
        case CudaMatmulRoute::MoePackedInt4: return "moe_packed_int4";
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
    bool dsv4_fp8_tensor_page) {
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
        (softcap != 0.0F && (rows != 1U || groups != 0U))) {
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
        descriptor.encoding == CudaWeightEncoding::Fp4E2m1Group32;
    const bool regfed_shape =
        regfed_encoding && groups == 0U && softcap == 0.0F &&
        (descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128
             ? regfed_fp8_shape_admissible(descriptor.rows, descriptor.columns)
             : regfed_fp4_shape_admissible(descriptor.rows, descriptor.columns));
    // The register-fed route requires a weight already permuted by an explicit
    // prepack_fragment call. matmul_impl must NOT decide this for itself: the
    // layout is a property of the weight, and matmul_impl cannot see the
    // weight's other consumers. Deciding it here corrupted the DeepSeek V4
    // attention output projection, which matmul_impl touches 129 times a run
    // and the attention path then reads canonically.
    const bool regfed = regfed_shape && regfed_matmul_enabled() &&
                        weight.impl_->fragment_prepacked;
    // No hidden fallback. Fragment order replaces the canonical layout, so a
    // permuted weight reaching a canonical kernel does not degrade -- it
    // decodes a permutation as if it were weights. Refuse instead.
    if (weight.impl_->fragment_prepacked && !regfed) {
        result.errors.emplace_back(
            "CUDA matmul received a fragment-prepacked weight but has no "
            "register-fed route for this call; refusing to read fragment order "
            "as canonical layout");
        return result;
    }
    const bool tensor_page =
        dsv4_fp8_tensor_page && !regfed &&
        state.dsv4_fp8_tensor_page_supported && rows > 1U && groups == 0U &&
        descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128 &&
        descriptor.columns % kDsv4Fp8TensorBlockK == 0U &&
        descriptor.rows % kDsv4Fp8TensorBlockN == 0U &&
        descriptor.columns <= std::numeric_limits<std::uint32_t>::max() &&
        descriptor.rows <= std::numeric_limits<std::uint32_t>::max();
    const auto input_scale_bytes = tensor_page
        ? static_cast<std::uint64_t>(rows) * descriptor.scale_columns
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
            "DeepSeek FP8 tensor page output workspace overflows");
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
    const auto required_output_bytes = tensor_page
        ? std::max(input_bytes, tensor_output_bytes)
        : output_bytes;
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
    const bool stage_input = ensure_host_staging(
        state.matmul_host_input, state.matmul_host_input_bytes, input_bytes);
    const bool stage_output = ensure_host_staging(
        state.matmul_host_output, state.matmul_host_output_bytes, output_bytes);
    if (stage_input) {
        std::memcpy(state.matmul_host_input, input.data(), input.size_bytes());
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_start, state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "record activation upload start");
        }
    }
    if (auto status = cudaMemcpyAsync(
            tensor_page ? static_cast<void*>(state.output)
                        : static_cast<void*>(state.input),
            stage_input ? static_cast<const void*>(state.matmul_host_input)
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
                        descriptor.encoding == CudaWeightEncoding::Fp4E2m1Group32;
    const bool w8_group32 =
        descriptor.encoding == CudaWeightEncoding::OffsetPackedInt8 &&
        rows == 1U && groups == 0U && descriptor.group_size == 32U &&
        descriptor.columns % 32U == 0U;
    if (tensor_page) {
        const dim3 quantize_grid(
            static_cast<unsigned int>(descriptor.scale_columns), rows, 1U);
        auto* compact_values = reinterpret_cast<unsigned char*>(state.input);
        auto* compact_scales = compact_values + input.size();
        quantize_activation_e4m3_bytes_kernel<<<
            quantize_grid, 128U, 0U, state.stream>>>(
            compact_values, compact_scales, state.output,
            descriptor.columns, rows);
    } else if (native) {
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
    if (descriptor.encoding == CudaWeightEncoding::Plain &&
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
            descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128
                ? static_cast<std::uint32_t>(descriptor.columns / 32U)
                : k_tiles / kRegfedKPerLoad;
        const std::uint32_t split = regfed_split_k(units, n_tiles);
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
            regfed_activation_fragment_kernel<<<
                static_cast<unsigned int>(std::min<std::uint64_t>(
                    (fragment_total + 255U) / 256U, 65535U)),
                256U, 0U, state.stream>>>(
                static_cast<uint2*>(state.regfed_activation),
                state.input + static_cast<std::size_t>(start) *
                                  descriptor.columns,
                chunk, static_cast<std::uint32_t>(descriptor.columns),
                chunk_blocks, chunk_groups);
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
    } else if (tensor_page) {
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
    } else if (descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128) {
        record_cuda_matmul_route(CudaMatmulRoute::Fp8E4m3Block128);
        native_fp8_matmul_kernel<<<grid, threads, 0, state.stream>>>(
            state.output, state.input,
            static_cast<const unsigned char*>(weight.impl_->weights),
            static_cast<const unsigned char*>(weight.impl_->scales),
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
            stage_output ? static_cast<void*>(state.matmul_host_output)
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

ValidationResult CudaBackend::enqueue_deepseek_moe(
    int device, std::span<const float> hidden,
    std::span<const CudaDeepSeekMoeExpert> routed,
    const CudaDeepSeekMoeExpert* shared, float swiglu_limit) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek MoE command targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    if (state.moe_in_flight) {
        result.errors.emplace_back(
            "DeepSeek MoE workspace already has an in-flight command");
        return result;
    }
    if (routed.size() > kMaxDeepSeekRoutedExperts ||
        (routed.empty() && shared == nullptr)) {
        result.errors.emplace_back(
            "DeepSeek MoE command requires one to six routed experts or a shared expert");
        return result;
    }
    if (!std::isfinite(swiglu_limit) || swiglu_limit <= 0.0F) {
        result.errors.emplace_back(
            "DeepSeek MoE SwiGLU limit must be finite and positive");
        return result;
    }

    std::uint64_t hidden_columns = 0U;
    std::uint64_t intermediate_columns = 0U;
    auto validate_expert = [&](const CudaDeepSeekMoeExpert& expert,
                               CudaWeightEncoding encoding,
                               bool shared_expert) {
        const std::array<const CudaWeight*, 3> weights{
            expert.w1, expert.w3, expert.w2};
        for (const auto* weight : weights) {
            if (weight == nullptr || !weight->valid()) {
                result.errors.emplace_back(
                    "DeepSeek MoE command contains an invalid CUDA weight");
                return false;
            }
            if (weight->impl_->device != device) {
                result.errors.emplace_back(
                    "DeepSeek MoE weights do not belong to the command device");
                return false;
            }
            if (weight->impl_->descriptor.encoding != encoding) {
                result.errors.emplace_back(
                    "DeepSeek MoE weight encoding is incompatible with the expert kind");
                return false;
            }
        }
        const auto& w1 = expert.w1->impl_->descriptor;
        const auto& w3 = expert.w3->impl_->descriptor;
        const auto& w2 = expert.w2->impl_->descriptor;
        const auto expected_dtype = encoding == CudaWeightEncoding::Fp4E2m1Group32
                                        ? SafetensorsDtype::I8
                                        : SafetensorsDtype::F8E4M3;
        const auto expected_group = encoding == CudaWeightEncoding::Fp4E2m1Group32
                                        ? 32U
                                        : 128U;
        if (w1.dtype != expected_dtype || w3.dtype != expected_dtype ||
            w2.dtype != expected_dtype || w1.group_size != expected_group ||
            w3.group_size != expected_group || w2.group_size != expected_group ||
            w1.rows == 0U || w1.columns == 0U ||
            w3.rows != w1.rows || w3.columns != w1.columns ||
            w2.rows != w1.columns || w2.columns != w1.rows) {
            result.errors.emplace_back(
                "DeepSeek MoE W1/W3/W2 shapes or native encoding metadata are invalid");
            return false;
        }
        if (!std::isfinite(expert.coefficient) ||
            (shared_expert && expert.coefficient != 1.0F)) {
            result.errors.emplace_back(
                "DeepSeek MoE expert coefficient is invalid");
            return false;
        }
        if (hidden_columns == 0U) {
            hidden_columns = w1.columns;
            intermediate_columns = w1.rows;
        } else if (hidden_columns != w1.columns ||
                   intermediate_columns != w1.rows) {
            result.errors.emplace_back(
                "DeepSeek MoE experts do not share one exact activation shape");
            return false;
        }
        return true;
    };
    for (const auto& expert : routed) {
        if (!validate_expert(expert, CudaWeightEncoding::Fp4E2m1Group32,
                             false)) {
            return result;
        }
    }
    if (shared != nullptr &&
        !validate_expert(*shared, CudaWeightEncoding::Fp8E4m3Block128, true)) {
        return result;
    }
    if (hidden.empty() || hidden.size() != hidden_columns ||
        hidden_columns > std::numeric_limits<unsigned int>::max() ||
        intermediate_columns > std::numeric_limits<unsigned int>::max()) {
        result.errors.emplace_back(
            "DeepSeek MoE hidden row or expert dimensions are incompatible");
        return result;
    }
    if (!std::all_of(hidden.begin(), hidden.end(),
                     [](float value) { return std::isfinite(value); })) {
        result.errors.emplace_back(
            "DeepSeek MoE hidden row contains a non-finite value");
        return result;
    }

    const std::uint64_t expert_count =
        static_cast<std::uint64_t>(routed.size()) + (shared == nullptr ? 0U : 1U);
    std::uint64_t hidden_bytes = 0U;
    std::uint64_t activation_bytes = 0U;
    std::uint64_t output_bytes = 0U;
    std::uint64_t host_staging_bytes = 0U;
    if (!checked_bytes(1U, hidden_columns, sizeof(float), hidden_bytes) ||
        !checked_bytes(expert_count, intermediate_columns, sizeof(float),
                       activation_bytes) ||
        !checked_bytes(expert_count, hidden_columns, sizeof(float), output_bytes) ||
        hidden_bytes > std::numeric_limits<std::size_t>::max() ||
        activation_bytes > std::numeric_limits<std::size_t>::max() ||
        output_bytes > std::numeric_limits<std::size_t>::max() ||
        output_bytes > std::numeric_limits<std::uint64_t>::max() -
                           sizeof(unsigned int)) {
        result.errors.emplace_back("DeepSeek MoE workspace size overflows");
        return result;
    }
    host_staging_bytes = output_bytes + sizeof(unsigned int);
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for DeepSeek MoE");
    }

    std::uint64_t allocation_calls = 0U;
    std::uint64_t allocation_bytes = 0U;
    auto ensure_workspace = [&](float*& pointer, std::uint64_t& capacity,
                                std::uint64_t required, const char* operation) {
        if (required <= capacity) return true;
        if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
        pointer = nullptr;
        capacity = 0U;
        if (const auto status =
                cudaMalloc(&pointer, static_cast<std::size_t>(required));
            status != cudaSuccess) {
            result = cuda_error(status, operation);
            return false;
        }
        capacity = required;
        ++allocation_calls;
        allocation_bytes += required;
        return true;
    };
    if (!ensure_workspace(state.moe_hidden, state.moe_hidden_bytes, hidden_bytes,
                          "allocate DeepSeek MoE hidden workspace") ||
        !ensure_workspace(state.moe_activations, state.moe_activation_bytes,
                          activation_bytes,
                          "allocate DeepSeek MoE activation workspace") ||
        !ensure_workspace(state.moe_output, state.moe_output_bytes, output_bytes,
                          "allocate DeepSeek MoE output workspace")) {
        return result;
    }
    if (state.moe_bf16_silu == nullptr) {
        constexpr std::size_t bytes = kDsv4Bf16SiluEntries * sizeof(float);
        static const std::array<float, kDsv4Bf16SiluEntries> table = [] {
            std::array<float, kDsv4Bf16SiluEntries> values{};
            for (std::size_t index = 0U; index < kDsv4Bf16SiluEntries;
                 ++index) {
                const auto bits = static_cast<std::uint32_t>(index) << 16U;
                const float value = std::bit_cast<float>(bits);
                values[index] = std::isfinite(value) ? silu_f32(value) : value;
            }
            return values;
        }();
        if (const auto status = cudaMalloc(&state.moe_bf16_silu, bytes);
            status != cudaSuccess) {
            return cuda_error(status, "allocate DeepSeek BF16 SiLU table");
        }
        if (const auto status = cudaMemcpyAsync(
                state.moe_bf16_silu, table.data(), bytes,
                cudaMemcpyHostToDevice, state.stream);
            status != cudaSuccess) {
            static_cast<void>(cudaFree(state.moe_bf16_silu));
            state.moe_bf16_silu = nullptr;
            return cuda_error(status, "upload DeepSeek BF16 SiLU table");
        }
        ++allocation_calls;
        allocation_bytes += bytes;
    }
    if (state.moe_error == nullptr) {
        if (const auto status = cudaMalloc(&state.moe_error, sizeof(unsigned int));
            status != cudaSuccess) {
            return cuda_error(status, "allocate DeepSeek MoE error flag");
        }
        ++allocation_calls;
        allocation_bytes += sizeof(unsigned int);
    }
    if (host_staging_bytes > state.moe_host_staging_bytes) {
        if (state.moe_host_staging != nullptr) {
            static_cast<void>(cudaFreeHost(state.moe_host_staging));
        }
        state.moe_host_staging = nullptr;
        state.moe_host_staging_bytes = 0U;
        if (const auto status = cudaMallocHost(
                &state.moe_host_staging,
                static_cast<std::size_t>(host_staging_bytes));
            status != cudaSuccess) {
            return cuda_error(status, "allocate DeepSeek MoE host staging");
        }
        state.moe_host_staging_bytes = host_staging_bytes;
        ++allocation_calls;
        allocation_bytes += host_staging_bytes;
    }

    DeepSeekFp4Batch routed_batch;
    for (std::size_t index = 0U; index < routed.size(); ++index) {
        const auto& expert = routed[index];
        routed_batch.w1_weights[index] =
            static_cast<const unsigned char*>(expert.w1->impl_->weights);
        routed_batch.w1_scales[index] =
            static_cast<const unsigned char*>(expert.w1->impl_->scales);
        routed_batch.w3_weights[index] =
            static_cast<const unsigned char*>(expert.w3->impl_->weights);
        routed_batch.w3_scales[index] =
            static_cast<const unsigned char*>(expert.w3->impl_->scales);
        routed_batch.w2_weights[index] =
            static_cast<const unsigned char*>(expert.w2->impl_->weights);
        routed_batch.w2_scales[index] =
            static_cast<const unsigned char*>(expert.w2->impl_->scales);
        routed_batch.coefficients[index] = expert.coefficient;
    }
    routed_batch.count = static_cast<std::uint32_t>(routed.size());

    state.moe_weights.clear();
    state.moe_weights.reserve(static_cast<std::size_t>(expert_count * 3U));
    for (const auto& expert : routed) {
        state.moe_weights.push_back(expert.w1->impl_);
        state.moe_weights.push_back(expert.w3->impl_);
        state.moe_weights.push_back(expert.w2->impl_);
    }
    if (shared != nullptr) {
        state.moe_weights.push_back(shared->w1->impl_);
        state.moe_weights.push_back(shared->w3->impl_);
        state.moe_weights.push_back(shared->w2->impl_);
    }

    state.moe_hidden_columns = hidden_columns;
    state.moe_intermediate_columns = intermediate_columns;
    state.moe_rows = 1U;
    state.moe_shared_rows = 1U;
    state.moe_routed_count = routed_batch.count;
    state.moe_has_shared = shared != nullptr;
    state.moe_shared_phase_timing_valid = false;
    state.moe_host_join = false;
    state.moe_output_to_mhc = false;
    state.moe_host_callback = {};
    state.moe_kernel_launches = 0U;
    state.moe_in_flight = true;
    state.moe_poisoned = false;
    auto abort_enqueue = [&](cudaError_t status, const char* operation) {
        result = cuda_error(status, operation);
        const auto drain_status = cudaStreamSynchronize(state.stream);
        if (drain_status != cudaSuccess) {
            result.errors.emplace_back(
                std::string("drain failed DeepSeek MoE enqueue: ") +
                cudaGetErrorString(drain_status));
            state.moe_poisoned = true;
        } else {
            state.moe_in_flight = false;
            state.moe_weights.clear();
        }
    };

    if (auto status = cudaEventRecord(state.moe_start, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record DeepSeek MoE start");
        return result;
    }
    if (auto status = cudaMemsetAsync(
            state.moe_error, 0, sizeof(unsigned int), state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "reset DeepSeek MoE error flag");
        return result;
    }
    if (auto status = cudaMemcpyAsync(
            state.moe_hidden, hidden.data(), static_cast<std::size_t>(hidden_bytes),
            cudaMemcpyHostToDevice, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "upload DeepSeek MoE hidden row");
        return result;
    }
    if (auto status = cudaEventRecord(state.moe_hidden_uploaded, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record DeepSeek MoE hidden upload");
        return result;
    }

    constexpr unsigned int threads = 256U;
    const dim3 hidden_quantize_grid(
        static_cast<unsigned int>((hidden_columns + 127U) / 128U), 1U, 1U);
    quantize_activation_e4m3_kernel<<<hidden_quantize_grid, 128U, 0U,
                                      state.stream>>>(
        state.moe_hidden, hidden_columns, 1U);
    ++state.moe_kernel_launches;
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        abort_enqueue(status, "launch DeepSeek MoE hidden quantization");
        return result;
    }
    if (shared != nullptr) {
        if (auto status = cudaEventRecord(
                state.moe_shared_input_finished, state.stream);
            status != cudaSuccess) {
            abort_enqueue(status, "record DeepSeek shared input quantization");
            return result;
        }
    }

    if (!routed.empty()) {
        // Counted once per command on the gate/up dispatch; the down kernel
        // mirrors the branch, so counting both would double every entry.
        record_cuda_matmul_route(CudaMatmulRoute::Dsv4MoeRoutedFp4);
        const auto& w1 = routed.front().w1->impl_->descriptor;
        const auto& w2 = routed.front().w2->impl_->descriptor;
        const dim3 gate_grid(static_cast<unsigned int>(intermediate_columns),
                             routed_batch.count, 1U);
        deepseek_fp4_gate_up_kernel<<<gate_grid, threads, 0U, state.stream>>>(
            state.moe_activations, state.moe_hidden, routed_batch,
            hidden_columns, intermediate_columns, w1.packed_columns,
            w1.scale_columns, swiglu_limit, state.moe_bf16_silu,
            state.moe_error);
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek FP4 W1/W3 SwiGLU");
            return result;
        }
        const dim3 activation_grid(
            static_cast<unsigned int>((intermediate_columns + 127U) / 128U),
            routed_batch.count, 1U);
        quantize_activation_e4m3_kernel<<<activation_grid, 128U, 0U,
                                          state.stream>>>(
            state.moe_activations, intermediate_columns, routed_batch.count);
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek routed activation quantization");
            return result;
        }
        const dim3 down_grid(static_cast<unsigned int>(hidden_columns),
                             routed_batch.count, 1U);
        deepseek_fp4_down_kernel<<<down_grid, threads, 0U, state.stream>>>(
            state.moe_output, state.moe_activations, routed_batch,
            intermediate_columns, hidden_columns, w2.packed_columns,
            w2.scale_columns);
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek FP4 W2");
            return result;
        }
    }

    if (shared != nullptr) {
        const auto& w1 = shared->w1->impl_->descriptor;
        const auto& w2 = shared->w2->impl_->descriptor;
        float* shared_activation = state.moe_activations +
            static_cast<std::uint64_t>(routed_batch.count) * intermediate_columns;
        float* shared_output = state.moe_output +
            static_cast<std::uint64_t>(routed_batch.count) * hidden_columns;
        const auto& w3_descriptor = shared->w3->impl_->descriptor;
        const bool shared_regfed =
            regfed_matmul_enabled() &&
            // All three must already be permuted. These same weights are also read
            // canonically by enqueue_deepseek_moe_rows through the paged kernels, so
            // this site may not decide their layout on its own. An explicit
            // prepack_fragment call is the opt-in, and it opts every consumer in.
            shared->w1->impl_->fragment_prepacked &&
            shared->w3->impl_->fragment_prepacked &&
            shared->w2->impl_->fragment_prepacked &&
            regfed_fp8_shape_admissible(w1.rows, w1.columns) &&
            regfed_fp8_shape_admissible(w3_descriptor.rows, w3_descriptor.columns) &&
            regfed_fp8_shape_admissible(w2.rows, w2.columns);
        if (!shared_regfed &&
            (shared->w1->impl_->fragment_prepacked ||
             shared->w3->impl_->fragment_prepacked ||
             shared->w2->impl_->fragment_prepacked)) {
            // Refuse rather than let the scalar kernel read fragment order as
            // canonical weights, which is silent corruption, not degradation.
            abort_enqueue(cudaErrorInvalidValue,
                          "DeepSeek shared expert weights are fragment-prepacked "
                          "but the register-fed route is unavailable");
            return result;
        }
        if (shared_regfed) {
            record_cuda_matmul_route(
                CudaMatmulRoute::Dsv4MoeSharedFp8RegisterFed);
            // Gate and up share one buffer so a single allocation covers both.
            void* gate_buffer = state.moe_regfed_gate;
            if (auto status = regfed_grow(
                    gate_buffer, state.moe_regfed_gate_bytes,
                    intermediate_columns * 2U * sizeof(float), false, state.stream);
                status != cudaSuccess) {
                abort_enqueue(status, "allocate register-fed shared expert buffers");
                return result;
            }
            state.moe_regfed_gate = static_cast<float*>(gate_buffer);
            state.moe_regfed_up = state.moe_regfed_gate + intermediate_columns;
            if (auto status = launch_regfed_fp8_matvec(
                    state.moe_regfed, w1, shared->w1->impl_->weights,
                    shared->w1->impl_->scales,
                    shared->w1->impl_->fragment_prepacked, state.moe_hidden,
                    state.moe_regfed_gate, state.stream);
                status != cudaSuccess) {
                abort_enqueue(status, "launch register-fed shared expert gate");
                return result;
            }
            if (auto status = launch_regfed_fp8_matvec(
                    state.moe_regfed, w3_descriptor, shared->w3->impl_->weights,
                    shared->w3->impl_->scales,
                    shared->w3->impl_->fragment_prepacked, state.moe_hidden,
                    state.moe_regfed_up, state.stream, true);
                status != cudaSuccess) {
                abort_enqueue(status, "launch register-fed shared expert up");
                return result;
            }
            regfed_shared_swiglu_kernel<<<
                static_cast<unsigned int>((intermediate_columns + 255U) / 256U),
                256U, 0U, state.stream>>>(
                shared_activation, state.moe_regfed_gate, state.moe_regfed_up,
                static_cast<std::uint32_t>(intermediate_columns), swiglu_limit,
                state.moe_bf16_silu, state.moe_error);
            state.moe_kernel_launches += 4U;
        } else {
            record_cuda_matmul_route(CudaMatmulRoute::Dsv4MoeSharedFp8);
            deepseek_fp8_gate_up_kernel<<<
                static_cast<unsigned int>(intermediate_columns), threads, 0U,
                state.stream>>>(
                shared_activation, state.moe_hidden,
                static_cast<const unsigned char*>(shared->w1->impl_->weights),
                static_cast<const unsigned char*>(shared->w1->impl_->scales),
                static_cast<const unsigned char*>(shared->w3->impl_->weights),
                static_cast<const unsigned char*>(shared->w3->impl_->scales),
                hidden_columns, intermediate_columns, w1.scale_columns,
                swiglu_limit, state.moe_bf16_silu, state.moe_error);
            ++state.moe_kernel_launches;
        }
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek shared FP8 W1/W3 SwiGLU");
            return result;
        }
        if (auto status = cudaEventRecord(
                state.moe_shared_gate_up_finished, state.stream);
            status != cudaSuccess) {
            abort_enqueue(status, "record DeepSeek shared gate/up completion");
            return result;
        }
        const dim3 shared_activation_grid(
            static_cast<unsigned int>((intermediate_columns + 127U) / 128U),
            1U, 1U);
        quantize_activation_e4m3_kernel<<<shared_activation_grid, 128U, 0U,
                                          state.stream>>>(
            shared_activation, intermediate_columns, 1U);
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek shared activation quantization");
            return result;
        }
        if (auto status = cudaEventRecord(
                state.moe_shared_activation_finished, state.stream);
            status != cudaSuccess) {
            abort_enqueue(status,
                          "record DeepSeek shared activation quantization");
            return result;
        }
        if (shared_regfed) {
            if (auto status = launch_regfed_fp8_matvec(
                    state.moe_regfed, w2, shared->w2->impl_->weights,
                    shared->w2->impl_->scales,
                    shared->w2->impl_->fragment_prepacked, shared_activation,
                    shared_output, state.stream);
                status != cudaSuccess) {
                abort_enqueue(status, "launch register-fed shared expert down");
                return result;
            }
            state.moe_kernel_launches += 2U;
        } else {
            deepseek_fp8_down_kernel<<<
                static_cast<unsigned int>(hidden_columns), threads, 0U,
                state.stream>>>(
                shared_output, shared_activation,
                static_cast<const unsigned char*>(shared->w2->impl_->weights),
                static_cast<const unsigned char*>(shared->w2->impl_->scales),
                intermediate_columns, hidden_columns, w2.scale_columns);
            ++state.moe_kernel_launches;
        }
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek shared FP8 W2");
            return result;
        }
        if (auto status = cudaEventRecord(
                state.moe_shared_finished, state.stream);
            status != cudaSuccess) {
            abort_enqueue(status, "record DeepSeek shared down completion");
            return result;
        }
        state.moe_shared_phase_timing_valid = true;
    }
    if (auto status = cudaEventRecord(state.moe_kernel_finished, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record DeepSeek MoE kernel completion");
        return result;
    }
    {
        std::scoped_lock lock(impl_->mutex);
        auto& device_stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        device_stats.activation_h2d_bytes += hidden_bytes;
        device_stats.matmul_calls += 3U * expert_count;
        device_stats.workspace_allocation_calls += allocation_calls;
        device_stats.workspace_allocation_bytes += allocation_bytes;
        ++device_stats.deepseek_moe_calls;
        device_stats.deepseek_moe_kernel_launches += state.moe_kernel_launches;
        ++device_stats.deepseek_moe_h2d_transfers;
        device_stats.deepseek_moe_h2d_bytes += hidden_bytes;
    }
    return result;
}

ValidationResult CudaBackend::enqueue_dsv4_host_moe(
    int device, std::span<const float> hidden,
    const CudaDeepSeekMoeExpert& shared, float swiglu_limit,
    CudaDsv4HostMoeCallback callback, void* callback_context) {
    return enqueue_dsv4_host_moe_impl(
        device, hidden, shared, swiglu_limit, callback, callback_context,
        nullptr, false);
}

ValidationResult CudaBackend::enqueue_dsv4_host_moe_device_view(
    int device, std::span<const float> hidden,
    const CudaDeepSeekMoeExpert& shared, float swiglu_limit,
    CudaDsv4HostMoeCallback callback, void* callback_context,
    CudaDsv4HostMoeDeviceView& view) {
    view = {};
    auto result = enqueue_dsv4_host_moe(
        device, hidden, shared, swiglu_limit, callback, callback_context);
    if (!result.ok()) return result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end() ||
        found->second.moe_host_callback_count == 0U) {
        result.errors.emplace_back(
            "DeepSeek host MoE device view has no queued command");
        return result;
    }
    auto& state = found->second;
    view.stream = state.stream;
    view.output = state.moe_output;
    view.status = state.moe_error;
    if (view.stream == nullptr || view.output == nullptr ||
        view.status == nullptr) {
        view = {};
        result.errors.emplace_back(
            "DeepSeek host MoE device view is incomplete");
    }
    return result;
}

ValidationResult CudaBackend::enqueue_dsv4_host_moe_from_mhc(
    int device, const CudaDeepSeekMoeExpert& shared, float swiglu_limit,
    CudaDsv4HostMoeCallback callback, void* callback_context) {
    return enqueue_dsv4_host_moe_impl(
        device, {}, shared, swiglu_limit, callback, callback_context,
        nullptr, true);
}

ValidationResult CudaBackend::enqueue_dsv4_host_moe_from_device_input(
    int device, const CudaDeepSeekMoeExpert& shared, float swiglu_limit,
    CudaDsv4DeviceInputHostMoeCallback callback, void* callback_context) {
    return enqueue_dsv4_host_moe_impl(
        device, {}, shared, swiglu_limit, nullptr, callback_context,
        callback, true);
}

ValidationResult CudaBackend::enqueue_dsv4_host_moe_from_device_input_device_view(
    int device, const CudaDeepSeekMoeExpert& shared, float swiglu_limit,
    CudaDsv4DeviceInputHostMoeCallback callback, void* callback_context,
    CudaDsv4HostMoeDeviceView& view) {
    view = {};
    auto result = enqueue_dsv4_host_moe_from_device_input(
        device, shared, swiglu_limit, callback, callback_context);
    if (!result.ok()) return result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end() ||
        found->second.moe_host_callback_count == 0U) {
        result.errors.emplace_back(
            "DeepSeek device-input host MoE view has no queued command");
        return result;
    }
    auto& state = found->second;
    // The reusable rank-local view deliberately leaves the backend branch
    // unpublished. The existing join still computes its local BF16 value in
    // stream order, but only the caller's FP32 NCCL result may commit the
    // branch through dsv4_mhc_commit_reduced_branch().
    state.dsv4_mhc_branch_ready = false;
    view.stream = state.stream;
    view.output = state.moe_output;
    view.status = state.moe_error;
    if (view.stream == nullptr || view.output == nullptr ||
        view.status == nullptr) {
        view = {};
        result.errors.emplace_back(
            "DeepSeek device-input host MoE view is incomplete");
    }
    return result;
}

ValidationResult CudaBackend::enqueue_dsv4_host_moe_impl(
    int device, std::span<const float> hidden,
    const CudaDeepSeekMoeExpert& shared, float swiglu_limit,
    CudaDsv4HostMoeCallback callback, void* callback_context,
    CudaDsv4DeviceInputHostMoeCallback device_input_callback,
    bool mhc_source_and_destination) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek host MoE command targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    if (state.moe_in_flight) {
        result.errors.emplace_back(
            "DeepSeek MoE workspace already has an in-flight command");
        return result;
    }
    if (state.moe_host_callback_count >=
        state.moe_host_callbacks.size()) {
        result.errors.emplace_back(
            "DeepSeek fixed host-MoE command chain is full");
        return result;
    }
    const bool device_input = device_input_callback != nullptr;
    if ((callback == nullptr) == !device_input || callback_context == nullptr ||
        !std::isfinite(swiglu_limit) || swiglu_limit <= 0.0F) {
        result.errors.emplace_back(
            "DeepSeek host MoE requires a callback, context, and positive SwiGLU limit");
        return result;
    }

    const std::array<const CudaWeight*, 3U> weights{
        shared.w1, shared.w3, shared.w2};
    for (const auto* weight : weights) {
        if (weight == nullptr || !weight->valid()) {
            result.errors.emplace_back(
                "DeepSeek host MoE shared expert contains an invalid CUDA weight");
            return result;
        }
        if (weight->impl_->device != device ||
            weight->impl_->descriptor.encoding !=
                CudaWeightEncoding::Fp8E4m3Block128) {
            result.errors.emplace_back(
                "DeepSeek host MoE shared weight has the wrong device or encoding");
            return result;
        }
    }
    const auto& w1 = shared.w1->impl_->descriptor;
    const auto& w3 = shared.w3->impl_->descriptor;
    const auto& w2 = shared.w2->impl_->descriptor;
    if (shared.coefficient != 1.0F ||
        w1.dtype != SafetensorsDtype::F8E4M3 ||
        w3.dtype != SafetensorsDtype::F8E4M3 ||
        w2.dtype != SafetensorsDtype::F8E4M3 ||
        w1.group_size != 128U || w3.group_size != 128U ||
        w2.group_size != 128U || w1.rows == 0U || w1.columns == 0U ||
        w3.rows != w1.rows || w3.columns != w1.columns ||
        w2.rows != w1.columns || w2.columns != w1.rows ||
        (mhc_source_and_destination
             ? (!hidden.empty() || state.dsv4_mhc_stage != 1U ||
                state.dsv4_mhc_workspace == nullptr ||
                state.dsv4_mhc_branch_ready ||
                (device_input != state.dsv4_host_moe_input_pending) ||
                w1.columns != kDsv4MhcHidden)
             : hidden.size() != w1.columns) ||
        w1.columns > std::numeric_limits<unsigned int>::max() ||
        w1.rows > std::numeric_limits<unsigned int>::max()) {
        result.errors.emplace_back(
            "DeepSeek host MoE shared expert shape or metadata is invalid");
        return result;
    }
    if (!mhc_source_and_destination &&
        !std::all_of(hidden.begin(), hidden.end(),
                     [](float value) { return std::isfinite(value); })) {
        result.errors.emplace_back(
            "DeepSeek host MoE hidden row contains a non-finite value");
        return result;
    }
    if (device_input &&
        (state.dsv4_host_moe_router_logits == nullptr ||
         (state.dsv4_host_moe_device_failure == nullptr &&
          state.dsv4_host_moe_host_failure == nullptr))) {
        result.errors.emplace_back(
            "DeepSeek device-input host MoE has no deferred input or status");
        return result;
    }

    const auto hidden_columns = w1.columns;
    const auto intermediate_columns = w1.rows;
    std::uint64_t hidden_bytes = 0U;
    std::uint64_t activation_bytes = 0U;
    std::uint64_t output_bytes = 0U;
    std::uint64_t rank_partial_bytes = 0U;
    if (!checked_bytes(1U, hidden_columns, sizeof(float), hidden_bytes) ||
        !checked_bytes(1U, intermediate_columns, sizeof(float),
                       activation_bytes) ||
        !checked_bytes(3U, hidden_columns, sizeof(float), output_bytes) ||
        !checked_bytes(2U, hidden_columns, sizeof(float), rank_partial_bytes) ||
        hidden_bytes > std::numeric_limits<std::size_t>::max() ||
        activation_bytes > std::numeric_limits<std::size_t>::max() ||
        output_bytes > std::numeric_limits<std::size_t>::max() ||
        rank_partial_bytes > std::numeric_limits<std::size_t>::max() ||
        hidden_bytes > std::numeric_limits<std::uint64_t>::max() -
                           sizeof(unsigned int)) {
        result.errors.emplace_back("DeepSeek host MoE workspace size overflows");
        return result;
    }
    constexpr std::uint64_t encoded_hidden_bytes =
        kDsv4MhcHidden * sizeof(std::uint16_t);
    constexpr std::uint64_t router_bytes =
        kDsv4MhcRouterLogits * sizeof(float);
    const auto local_upstream_failure_bytes =
        device_input && state.dsv4_host_moe_device_failure != nullptr
            ? sizeof(unsigned int) : 0U;
    const auto device_input_bytes = device_input
        ? encoded_hidden_bytes + router_bytes +
              local_upstream_failure_bytes
        : 0U;
    const auto host_staging_bytes = std::max(
        rank_partial_bytes + device_input_bytes,
        hidden_bytes + sizeof(unsigned int));
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for DeepSeek host MoE");
    }

    std::uint64_t allocation_calls = 0U;
    std::uint64_t allocation_bytes = 0U;
    const auto ensure_workspace = [&](float*& pointer,
                                      std::uint64_t& capacity,
                                      std::uint64_t required,
                                      const char* operation) {
        if (required <= capacity) return true;
        if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
        pointer = nullptr;
        capacity = 0U;
        if (const auto status = cudaMalloc(
                &pointer, static_cast<std::size_t>(required));
            status != cudaSuccess) {
            result = cuda_error(status, operation);
            return false;
        }
        capacity = required;
        ++allocation_calls;
        allocation_bytes += required;
        return true;
    };
    if (!ensure_workspace(state.moe_hidden, state.moe_hidden_bytes,
                          hidden_bytes,
                          "allocate DeepSeek host MoE hidden workspace") ||
        !ensure_workspace(state.moe_activations, state.moe_activation_bytes,
                          activation_bytes,
                          "allocate DeepSeek host MoE activation workspace") ||
        !ensure_workspace(state.moe_output, state.moe_output_bytes,
                          output_bytes,
                          "allocate DeepSeek host MoE output workspace")) {
        return result;
    }
    if (state.moe_bf16_silu == nullptr) {
        constexpr std::size_t bytes = kDsv4Bf16SiluEntries * sizeof(float);
        static const std::array<float, kDsv4Bf16SiluEntries> table = [] {
            std::array<float, kDsv4Bf16SiluEntries> values{};
            for (std::size_t index = 0U; index < kDsv4Bf16SiluEntries;
                 ++index) {
                const auto bits = static_cast<std::uint32_t>(index) << 16U;
                const float value = std::bit_cast<float>(bits);
                values[index] = std::isfinite(value) ? silu_f32(value) : value;
            }
            return values;
        }();
        if (const auto status = cudaMalloc(&state.moe_bf16_silu, bytes);
            status != cudaSuccess) {
            return cuda_error(status, "allocate DeepSeek BF16 SiLU table");
        }
        if (const auto status = cudaMemcpyAsync(
                state.moe_bf16_silu, table.data(), bytes,
                cudaMemcpyHostToDevice, state.stream);
            status != cudaSuccess) {
            static_cast<void>(cudaFree(state.moe_bf16_silu));
            state.moe_bf16_silu = nullptr;
            return cuda_error(status, "upload DeepSeek BF16 SiLU table");
        }
        ++allocation_calls;
        allocation_bytes += bytes;
    }
    if (state.moe_error == nullptr) {
        if (const auto status = cudaMalloc(
                &state.moe_error, sizeof(unsigned int));
            status != cudaSuccess) {
            return cuda_error(status, "allocate DeepSeek MoE error flag");
        }
        ++allocation_calls;
        allocation_bytes += sizeof(unsigned int);
    }
    if (host_staging_bytes > state.moe_host_staging_bytes) {
        if (state.moe_host_staging != nullptr) {
            static_cast<void>(cudaFreeHost(state.moe_host_staging));
        }
        state.moe_host_staging = nullptr;
        state.moe_host_staging_bytes = 0U;
        if (const auto status = cudaMallocHost(
                &state.moe_host_staging,
                static_cast<std::size_t>(host_staging_bytes));
            status != cudaSuccess) {
            return cuda_error(status,
                              "allocate DeepSeek host MoE pinned staging");
        }
        state.moe_host_staging_bytes = host_staging_bytes;
        ++allocation_calls;
        allocation_bytes += host_staging_bytes;
    }

    if (state.moe_host_callback_count == 0U) {
        state.moe_weights.clear();
        state.moe_weights.reserve(3U * 43U);
    }
    state.moe_weights.push_back(shared.w1->impl_);
    state.moe_weights.push_back(shared.w3->impl_);
    state.moe_weights.push_back(shared.w2->impl_);
    state.moe_hidden_columns = hidden_columns;
    state.moe_intermediate_columns = intermediate_columns;
    state.moe_rows = 1U;
    state.moe_shared_rows = 1U;
    state.moe_routed_count = 0U;
    state.moe_has_shared = true;
    state.moe_shared_phase_timing_valid = false;
    state.moe_host_join = true;
    state.moe_output_to_mhc = mhc_source_and_destination;
    state.moe_kernel_launches = 0U;
    state.moe_in_flight = true;
    state.moe_poisoned = false;
    auto& host_callback_state = state.moe_host_callbacks[
        state.moe_host_callback_count];
    host_callback_state = {};
    host_callback_state.function = callback;
    host_callback_state.device_input_function = device_input_callback;
    host_callback_state.context = callback_context;
    host_callback_state.rank_partials =
        static_cast<float*>(state.moe_host_staging);
    host_callback_state.rank_partial_elements = 2U * hidden_columns;
    auto* host_bytes = static_cast<std::byte*>(state.moe_host_staging);
    auto* staged_hidden = reinterpret_cast<std::uint16_t*>(
        host_bytes + static_cast<std::ptrdiff_t>(rank_partial_bytes));
    auto* staged_router = reinterpret_cast<float*>(
        host_bytes + static_cast<std::ptrdiff_t>(
            rank_partial_bytes + encoded_hidden_bytes));
    auto* staged_upstream_failure = reinterpret_cast<unsigned int*>(
        host_bytes + static_cast<std::ptrdiff_t>(
            rank_partial_bytes + encoded_hidden_bytes + router_bytes));
    if (device_input) {
        host_callback_state.encoded_hidden = staged_hidden;
        host_callback_state.hidden_elements = hidden_columns;
        host_callback_state.router_logits = staged_router;
        host_callback_state.router_elements = kDsv4MhcRouterLogits;
        host_callback_state.upstream_failure =
            state.dsv4_host_moe_host_failure != nullptr
                ? state.dsv4_host_moe_host_failure
                : staged_upstream_failure;
    }

    const auto abort_enqueue = [&](cudaError_t status,
                                   const char* operation) {
        state.moe_shared_phase_timing_valid = false;
        result = cuda_error(status, operation);
        const auto main_status = cudaStreamSynchronize(state.stream);
        const auto shared_status = cudaStreamSynchronize(
            state.moe_shared_stream);
        if (main_status != cudaSuccess || shared_status != cudaSuccess) {
            result.errors.emplace_back(
                "drain failed DeepSeek host MoE enqueue");
            state.moe_poisoned = true;
        } else {
            state.moe_in_flight = false;
            state.moe_host_join = false;
            state.moe_output_to_mhc = false;
            state.moe_host_callback = {};
            state.moe_host_callback_count = 0U;
            state.moe_weights.clear();
            state.dsv4_host_moe_input_pending = false;
            state.dsv4_host_moe_router_logits = nullptr;
            state.dsv4_host_moe_device_failure = nullptr;
            state.dsv4_host_moe_host_failure = nullptr;
            state.dsv4_deferred_attention_source_device = -1;
            state.dsv4_deferred_attention_cross_transition = false;
        }
    };

    if (auto status = cudaEventRecord(state.moe_start, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record DeepSeek host MoE start");
        return result;
    }
    if (state.moe_host_callback_count == 0U) {
        if (auto status = cudaMemsetAsync(
                state.moe_error, 0, sizeof(unsigned int), state.stream);
            status != cudaSuccess) {
            abort_enqueue(status, "reset DeepSeek host MoE error flag");
            return result;
        }
    }
    if (!mhc_source_and_destination) {
        if (auto status = cudaMemcpyAsync(
                state.moe_hidden, hidden.data(),
                static_cast<std::size_t>(hidden_bytes), cudaMemcpyHostToDevice,
                state.stream);
            status != cudaSuccess) {
            abort_enqueue(status, "upload DeepSeek host MoE hidden row");
            return result;
        }
    }
    if (auto status = cudaEventRecord(
            state.moe_hidden_uploaded, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record DeepSeek host MoE hidden upload");
        return result;
    }
    if (auto status = cudaStreamWaitEvent(
            state.moe_shared_stream, state.moe_hidden_uploaded);
        status != cudaSuccess) {
        abort_enqueue(status, "fan out DeepSeek shared expert");
        return result;
    }
    if (device_input) {
        if (auto status = cudaMemcpyAsync(
                staged_hidden, state.dsv4_mhc_workspace->layer_input,
                static_cast<std::size_t>(encoded_hidden_bytes),
                cudaMemcpyDeviceToHost, state.stream);
            status != cudaSuccess) {
            abort_enqueue(status,
                          "stage DeepSeek device-input host MoE hidden row");
            return result;
        }
        if (auto status = cudaMemcpyAsync(
                staged_router, state.dsv4_host_moe_router_logits,
                static_cast<std::size_t>(router_bytes),
                cudaMemcpyDeviceToHost, state.stream);
            status != cudaSuccess) {
            abort_enqueue(status,
                          "stage DeepSeek device-input host MoE router logits");
            return result;
        }
        if (state.dsv4_host_moe_device_failure != nullptr) {
            if (auto status = cudaMemcpyAsync(
                    staged_upstream_failure,
                    state.dsv4_host_moe_device_failure,
                    sizeof(*staged_upstream_failure), cudaMemcpyDeviceToHost,
                    state.stream);
                status != cudaSuccess) {
                abort_enqueue(status,
                              "stage DeepSeek deferred attention status");
                return result;
            }
        }
    }

    const dim3 hidden_quantize_grid(
        static_cast<unsigned int>((hidden_columns + 127U) / 128U), 1U, 1U);
    if (mhc_source_and_destination) {
        quantize_bf16_activation_e4m3_kernel<<<
            hidden_quantize_grid, 128U, 0U, state.moe_shared_stream>>>(
            state.moe_hidden, state.dsv4_mhc_workspace->layer_input,
            hidden_columns);
    } else {
        quantize_activation_e4m3_kernel<<<
            hidden_quantize_grid, 128U, 0U, state.moe_shared_stream>>>(
            state.moe_hidden, hidden_columns, 1U);
    }
    ++state.moe_kernel_launches;
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        abort_enqueue(status,
                      "launch DeepSeek host MoE hidden quantization");
        return result;
    }
    if (auto status = cudaEventRecord(
            state.moe_shared_input_finished, state.moe_shared_stream);
        status != cudaSuccess) {
        abort_enqueue(status,
                      "record DeepSeek shared input quantization");
        return result;
    }

    constexpr unsigned int threads = 256U;
    const auto& w3_descriptor = shared.w3->impl_->descriptor;
    const bool shared_regfed =
        regfed_matmul_enabled() &&
        // All three must already be permuted. These same weights are also read
        // canonically by enqueue_deepseek_moe_rows through the paged kernels, so
        // this site may not decide their layout on its own. An explicit
        // prepack_fragment call is the opt-in, and it opts every consumer in.
        shared.w1->impl_->fragment_prepacked &&
        shared.w3->impl_->fragment_prepacked &&
        shared.w2->impl_->fragment_prepacked &&
        regfed_fp8_shape_admissible(w1.rows, w1.columns) &&
        regfed_fp8_shape_admissible(w3_descriptor.rows, w3_descriptor.columns) &&
        regfed_fp8_shape_admissible(w2.rows, w2.columns);
    if (!shared_regfed &&
        (shared.w1->impl_->fragment_prepacked ||
         shared.w3->impl_->fragment_prepacked ||
         shared.w2->impl_->fragment_prepacked)) {
        // Refuse rather than let the scalar kernel read fragment order as
        // canonical weights, which is silent corruption, not degradation.
        abort_enqueue(cudaErrorInvalidValue,
                      "DeepSeek shared expert weights are fragment-prepacked "
                      "but the register-fed route is unavailable");
        return result;
    }
    if (shared_regfed) {
        record_cuda_matmul_route(CudaMatmulRoute::Dsv4MoeSharedFp8RegisterFed);
        void* gate_buffer = state.moe_regfed_gate;
        if (auto status = regfed_grow(
                gate_buffer, state.moe_regfed_gate_bytes,
                intermediate_columns * 2U * sizeof(float), false,
                state.moe_shared_stream);
            status != cudaSuccess) {
            abort_enqueue(status, "allocate register-fed shared expert buffers");
            return result;
        }
        state.moe_regfed_gate = static_cast<float*>(gate_buffer);
        state.moe_regfed_up = state.moe_regfed_gate + intermediate_columns;
        if (auto status = launch_regfed_fp8_matvec(
                state.moe_regfed, w1, shared.w1->impl_->weights,
                shared.w1->impl_->scales, shared.w1->impl_->fragment_prepacked,
                state.moe_hidden, state.moe_regfed_gate,
                state.moe_shared_stream);
            status != cudaSuccess) {
            abort_enqueue(status, "launch register-fed shared expert gate");
            return result;
        }
        if (auto status = launch_regfed_fp8_matvec(
                state.moe_regfed, w3_descriptor, shared.w3->impl_->weights,
                shared.w3->impl_->scales, shared.w3->impl_->fragment_prepacked,
                state.moe_hidden, state.moe_regfed_up,
                state.moe_shared_stream, true);
            status != cudaSuccess) {
            abort_enqueue(status, "launch register-fed shared expert up");
            return result;
        }
        regfed_shared_swiglu_kernel<<<
            static_cast<unsigned int>((intermediate_columns + 255U) / 256U),
            256U, 0U, state.moe_shared_stream>>>(
            state.moe_activations, state.moe_regfed_gate, state.moe_regfed_up,
            static_cast<std::uint32_t>(intermediate_columns), swiglu_limit,
            state.moe_bf16_silu, state.moe_error);
        state.moe_kernel_launches += 4U;
    } else {
        record_cuda_matmul_route(CudaMatmulRoute::Dsv4MoeSharedFp8);
        deepseek_fp8_gate_up_kernel<<<
            static_cast<unsigned int>(intermediate_columns), threads, 0U,
            state.moe_shared_stream>>>(
            state.moe_activations, state.moe_hidden,
            static_cast<const unsigned char*>(shared.w1->impl_->weights),
            static_cast<const unsigned char*>(shared.w1->impl_->scales),
            static_cast<const unsigned char*>(shared.w3->impl_->weights),
            static_cast<const unsigned char*>(shared.w3->impl_->scales),
            hidden_columns, intermediate_columns, w1.scale_columns,
            swiglu_limit, state.moe_bf16_silu, state.moe_error);
        ++state.moe_kernel_launches;
    }
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        abort_enqueue(status, "launch DeepSeek host MoE shared W1/W3");
        return result;
    }
    if (auto status = cudaEventRecord(
            state.moe_shared_gate_up_finished, state.moe_shared_stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record DeepSeek shared gate/up completion");
        return result;
    }
    const dim3 activation_grid(
        static_cast<unsigned int>((intermediate_columns + 127U) / 128U),
        1U, 1U);
    quantize_activation_e4m3_kernel<<<
        activation_grid, 128U, 0U, state.moe_shared_stream>>>(
        state.moe_activations, intermediate_columns, 1U);
    ++state.moe_kernel_launches;
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        abort_enqueue(status,
                      "launch DeepSeek host MoE shared activation quantization");
        return result;
    }
    if (auto status = cudaEventRecord(
            state.moe_shared_activation_finished, state.moe_shared_stream);
        status != cudaSuccess) {
        abort_enqueue(status,
                      "record DeepSeek shared activation quantization");
        return result;
    }
    if (shared_regfed) {
        if (auto status = launch_regfed_fp8_matvec(
                state.moe_regfed, w2, shared.w2->impl_->weights,
                shared.w2->impl_->scales, shared.w2->impl_->fragment_prepacked,
                state.moe_activations, state.moe_output,
                state.moe_shared_stream);
            status != cudaSuccess) {
            abort_enqueue(status, "launch register-fed shared expert down");
            return result;
        }
        state.moe_kernel_launches += 2U;
    } else {
        deepseek_fp8_down_kernel<<<
            static_cast<unsigned int>(hidden_columns), threads, 0U,
            state.moe_shared_stream>>>(
            state.moe_output, state.moe_activations,
            static_cast<const unsigned char*>(shared.w2->impl_->weights),
            static_cast<const unsigned char*>(shared.w2->impl_->scales),
            intermediate_columns, hidden_columns, w2.scale_columns);
        ++state.moe_kernel_launches;
    }
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        abort_enqueue(status, "launch DeepSeek host MoE shared W2");
        return result;
    }
    if (auto status = cudaEventRecord(
            state.moe_shared_finished, state.moe_shared_stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record DeepSeek shared-expert completion");
        return result;
    }
    state.moe_shared_phase_timing_valid = true;

    if (auto status = cudaLaunchHostFunc(
            state.stream, run_dsv4_host_moe_callback,
            &host_callback_state);
        status != cudaSuccess) {
        abort_enqueue(status, "enqueue DeepSeek CPU-MoE callback");
        return result;
    }
    auto* rank_partials = state.moe_output + hidden_columns;
    if (auto status = cudaMemcpyAsync(
            rank_partials, state.moe_host_staging,
            static_cast<std::size_t>(rank_partial_bytes),
            cudaMemcpyHostToDevice, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "upload DeepSeek CPU-MoE rank partials");
        return result;
    }
    // Routed-expert tier. Enqueued after the callback, so it is stream-ordered
    // behind it and observes the selection the callback wrote into pinned
    // memory, and after the rank-partial upload, so it accumulates into the
    // same buffer the join below already consumes. No host wait, no worker
    // thread, no change to the join itself.
    if (state.tier_committed && state.tier_installed != 0U) {
        const std::uint64_t tier_activation_bytes =
            kMaxDeepSeekRoutedExperts * intermediate_columns * sizeof(float);
        if (tier_activation_bytes > state.tier_activation_bytes) {
            if (state.tier_activations != nullptr) {
                static_cast<void>(cudaFree(state.tier_activations));
                state.tier_activations = nullptr;
                state.tier_activation_bytes = 0U;
            }
            void* scratch = nullptr;
            if (const auto status = cudaMalloc(
                    &scratch, static_cast<std::size_t>(tier_activation_bytes));
                status != cudaSuccess) {
                abort_enqueue(status, "allocate DeepSeek tier activations");
                return result;
            }
            state.tier_activations = static_cast<float*>(scratch);
            state.tier_activation_bytes = tier_activation_bytes;
        }
        if (auto status = cudaMemcpyAsync(
                state.tier_selection_device, state.tier_selection_host,
                sizeof(CudaDsv4TierSelection), cudaMemcpyHostToDevice,
                state.stream);
            status != cudaSuccess) {
            abort_enqueue(status, "upload DeepSeek tier selection");
            return result;
        }
        DeepSeekTierTable table;
        table.w1_weights = state.tier_device_pointers[0];
        table.w1_scales = state.tier_device_pointers[1];
        table.w3_weights = state.tier_device_pointers[2];
        table.w3_scales = state.tier_device_pointers[3];
        table.w2_weights = state.tier_device_pointers[4];
        table.w2_scales = state.tier_device_pointers[5];
        table.experts = state.tier_experts;
        const auto* tier_selection =
            reinterpret_cast<const DeepSeekTierSelection*>(
                state.tier_selection_device);
        const dim3 tier_gate_grid(
            static_cast<unsigned int>(intermediate_columns),
            kMaxDeepSeekRoutedExperts, 1U);
        record_cuda_matmul_route(CudaMatmulRoute::Dsv4MoeTierFp4);
        deepseek_fp4_tier_gate_up_kernel<<<
            tier_gate_grid, threads, 0U, state.stream>>>(
            state.tier_activations, state.moe_hidden, table, tier_selection,
            hidden_columns, intermediate_columns,
            state.tier_gate_packed_columns, state.tier_gate_scale_columns);
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek tier W1/W3");
            return result;
        }
        const dim3 tier_down_grid(static_cast<unsigned int>(hidden_columns),
                                  kMaxDeepSeekRoutedExperts, 1U);
        deepseek_fp4_tier_down_kernel<<<
            tier_down_grid, threads, 0U, state.stream>>>(
            rank_partials, state.tier_activations, table, tier_selection,
            intermediate_columns, hidden_columns,
            state.tier_down_packed_columns, state.tier_down_scale_columns);
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek tier W2");
            return result;
        }
    }
    if (auto status = cudaStreamWaitEvent(
            state.stream, state.moe_shared_finished);
        status != cudaSuccess) {
        abort_enqueue(status, "join DeepSeek shared-expert stream");
        return result;
    }
    constexpr unsigned int join_threads = 256U;
    const auto join_blocks = static_cast<unsigned int>(
        (hidden_columns + join_threads - 1U) / join_threads);
    if (mhc_source_and_destination) {
        dsv4_host_moe_join_mhc_kernel<<<
            join_blocks, join_threads, 0U, state.stream>>>(
            state.moe_output, rank_partials,
            state.dsv4_mhc_workspace->branch, hidden_columns,
            state.moe_error);
    } else {
        dsv4_host_moe_join_kernel<<<
            join_blocks, join_threads, 0U, state.stream>>>(
            state.moe_output, rank_partials, hidden_columns,
            state.moe_error);
    }
    ++state.moe_kernel_launches;
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        abort_enqueue(status, "launch DeepSeek routed/shared join");
        return result;
    }
    if (auto status = cudaEventRecord(
            state.moe_kernel_finished, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record DeepSeek host MoE completion");
        return result;
    }

    const auto total_h2d_bytes = rank_partial_bytes +
        (mhc_source_and_destination ? 0U : hidden_bytes);
    const auto input_d2h_bytes = device_input
        ? encoded_hidden_bytes + router_bytes +
              (state.dsv4_host_moe_device_failure != nullptr
                   ? sizeof(unsigned int) : 0U)
        : 0U;
    {
        std::scoped_lock lock(impl_->mutex);
        auto& device_stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        device_stats.activation_h2d_bytes += total_h2d_bytes;
        device_stats.activation_d2h_bytes += input_d2h_bytes;
        device_stats.matmul_calls += 3U;
        device_stats.workspace_allocation_calls += allocation_calls;
        device_stats.workspace_allocation_bytes += allocation_bytes;
        ++device_stats.deepseek_moe_calls;
        device_stats.deepseek_moe_kernel_launches +=
            state.moe_kernel_launches;
        device_stats.deepseek_moe_h2d_transfers +=
            mhc_source_and_destination ? 1U : 2U;
        device_stats.deepseek_moe_h2d_bytes += total_h2d_bytes;
        device_stats.deepseek_moe_d2h_transfers += device_input
            ? (state.dsv4_host_moe_device_failure != nullptr ? 3U : 2U)
            : 0U;
        device_stats.deepseek_moe_d2h_bytes += input_d2h_bytes;
    }
    if (device_input) {
        state.dsv4_host_moe_input_pending = false;
        state.dsv4_host_moe_router_logits = nullptr;
        state.dsv4_host_moe_device_failure = nullptr;
        state.dsv4_host_moe_host_failure = nullptr;
    }
    if (mhc_source_and_destination) {
        // The branch is produced later in this stream, but every consumer is
        // also stream/event ordered. Publishing host ownership now allows the
        // fixed dependent command chain to be issued without a CPU drain.
        state.dsv4_mhc_branch_ready = true;
    }
    state.moe_in_flight = false;
    ++state.moe_host_callback_count;
    return result;
}

ValidationResult CudaBackend::enqueue_deepseek_moe_rows(
    int device, std::span<const float> hidden, std::uint32_t hidden_rows,
    std::span<const CudaDeepSeekMoeRowGroup> groups,
    const CudaDeepSeekMoeExpert* shared,
    std::span<const std::uint32_t> shared_rows, float swiglu_limit) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek MoE page command targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    if (state.moe_in_flight) {
        result.errors.emplace_back(
            "DeepSeek MoE workspace already has an in-flight command");
        return result;
    }
    if (hidden_rows == 0U || (groups.empty() && shared == nullptr)) {
        result.errors.emplace_back(
            "DeepSeek MoE page command requires rows and at least one expert");
        return result;
    }
    if (!std::isfinite(swiglu_limit) || swiglu_limit <= 0.0F) {
        result.errors.emplace_back(
            "DeepSeek MoE SwiGLU limit must be finite and positive");
        return result;
    }

    std::uint64_t hidden_columns = 0U;
    std::uint64_t intermediate_columns = 0U;
    const auto validate_triplet = [&](const CudaWeight* w1_weight,
                                      const CudaWeight* w3_weight,
                                      const CudaWeight* w2_weight,
                                      CudaWeightEncoding encoding) {
        for (const auto* weight : {w1_weight, w3_weight, w2_weight}) {
            if (weight == nullptr || !weight->valid()) {
                result.errors.emplace_back(
                    "DeepSeek MoE page command contains an invalid CUDA weight");
                return false;
            }
            if (weight->impl_->device != device) {
                result.errors.emplace_back(
                    "DeepSeek MoE page weights do not belong to the command device");
                return false;
            }
            if (weight->impl_->descriptor.encoding != encoding) {
                result.errors.emplace_back(
                    "DeepSeek MoE page weight encoding is incompatible with the expert kind");
                return false;
            }
        }
        const auto& w1 = w1_weight->impl_->descriptor;
        const auto& w3 = w3_weight->impl_->descriptor;
        const auto& w2 = w2_weight->impl_->descriptor;
        const auto expected_dtype = encoding == CudaWeightEncoding::Fp4E2m1Group32
                                        ? SafetensorsDtype::I8
                                        : SafetensorsDtype::F8E4M3;
        const auto expected_group =
            encoding == CudaWeightEncoding::Fp4E2m1Group32 ? 32U : 128U;
        if (w1.dtype != expected_dtype || w3.dtype != expected_dtype ||
            w2.dtype != expected_dtype || w1.group_size != expected_group ||
            w3.group_size != expected_group || w2.group_size != expected_group ||
            w1.rows == 0U || w1.columns == 0U ||
            w3.rows != w1.rows || w3.columns != w1.columns ||
            w2.rows != w1.columns || w2.columns != w1.rows) {
            result.errors.emplace_back(
                "DeepSeek MoE page W1/W3/W2 shapes or native encoding metadata are invalid");
            return false;
        }
        if (hidden_columns == 0U) {
            hidden_columns = w1.columns;
            intermediate_columns = w1.rows;
        } else if (hidden_columns != w1.columns ||
                   intermediate_columns != w1.rows) {
            result.errors.emplace_back(
                "DeepSeek MoE page experts do not share one exact activation shape");
            return false;
        }
        return true;
    };

    // A transformed expert arrives as its TP shards instead of a triplet. The
    // shards carry the same values, so they must agree with the triplets on
    // the activation shape they all share.
    std::uint64_t shard_intermediate = 0U;
    const auto validate_shards = [&](const CudaDeepSeekMoeRowGroup& group) {
        for (const auto* shard : group.tiled_shards) {
            if (shard == nullptr || !shard->valid()) {
                result.errors.emplace_back(
                    "DeepSeek MoE page command contains an invalid expert shard");
                return false;
            }
            if (shard->impl_->device != device ||
                shard->impl_->descriptor.encoding !=
                    CudaWeightEncoding::Fp4E2m1Tiled32) {
                result.errors.emplace_back(
                    "DeepSeek MoE page expert shard has the wrong device or encoding");
                return false;
            }
        }
        const auto& first = group.tiled_shards.front()->impl_->descriptor;
        for (const auto* shard : group.tiled_shards) {
            const auto& descriptor = shard->impl_->descriptor;
            if (descriptor.rows != first.rows ||
                descriptor.columns != first.columns) {
                result.errors.emplace_back(
                    "DeepSeek MoE page expert shards disagree on shape");
                return false;
            }
        }
        const auto shards =
            static_cast<std::uint64_t>(group.tiled_shards.size());
        if (hidden_columns == 0U) {
            hidden_columns = first.rows;
            intermediate_columns = first.columns * shards;
        } else if (hidden_columns != first.rows ||
                   intermediate_columns != first.columns * shards) {
            result.errors.emplace_back(
                "DeepSeek MoE page experts do not share one exact activation shape");
            return false;
        }
        if (shard_intermediate == 0U) {
            shard_intermediate = first.columns;
        } else if (shard_intermediate != first.columns) {
            result.errors.emplace_back(
                "DeepSeek MoE page expert shards disagree on width");
            return false;
        }
        return true;
    };

    std::uint64_t work_count = 0U;
    for (const auto& group : groups) {
        const bool tiled = group.tiled_shards.front() != nullptr;
        if (tiled ? !validate_shards(group)
                  : !validate_triplet(group.w1, group.w3, group.w2,
                                      CudaWeightEncoding::Fp4E2m1Group32)) {
            return result;
        }
        if (group.rows.empty() || group.rows.size() != group.coefficients.size()) {
            result.errors.emplace_back(
                "DeepSeek MoE page group rows and coefficients must be non-empty and equal in size");
            return result;
        }
        for (const auto row : group.rows) {
            if (row >= hidden_rows) {
                result.errors.emplace_back(
                    "DeepSeek MoE page group row index is out of range");
                return result;
            }
        }
        for (const auto coefficient : group.coefficients) {
            if (!std::isfinite(coefficient)) {
                result.errors.emplace_back(
                    "DeepSeek MoE page group coefficient is invalid");
                return result;
            }
        }
        work_count += group.rows.size();
    }
    if (shared != nullptr) {
        if (!validate_triplet(shared->w1, shared->w3, shared->w2,
                              CudaWeightEncoding::Fp8E4m3Block128) ||
            shared->coefficient != 1.0F) {
            if (result.ok()) {
                result.errors.emplace_back(
                    "DeepSeek MoE page shared expert coefficient is invalid");
            }
            return result;
        }
        if (shared_rows.empty()) {
            result.errors.emplace_back(
                "DeepSeek MoE page shared expert requires at least one row");
            return result;
        }
        for (const auto row : shared_rows) {
            if (row >= hidden_rows) {
                result.errors.emplace_back(
                    "DeepSeek MoE page shared row index is out of range");
                return result;
            }
        }
    } else if (!shared_rows.empty()) {
        result.errors.emplace_back(
            "DeepSeek MoE page shared rows were supplied without a shared expert");
        return result;
    }
    const auto shared_count =
        static_cast<std::uint64_t>(shared == nullptr ? 0U : shared_rows.size());
    if (work_count == 0U && shared_count == 0U) {
        result.errors.emplace_back("DeepSeek MoE page command has no work");
        return result;
    }
    if (hidden.size() !=
            static_cast<std::size_t>(hidden_rows) * hidden_columns ||
        hidden_columns > std::numeric_limits<unsigned int>::max() ||
        intermediate_columns > std::numeric_limits<unsigned int>::max()) {
        result.errors.emplace_back(
            "DeepSeek MoE page hidden rows or expert dimensions are incompatible");
        return result;
    }
    if (!std::all_of(hidden.begin(), hidden.end(),
                     [](float value) { return std::isfinite(value); })) {
        result.errors.emplace_back(
            "DeepSeek MoE page hidden rows contain a non-finite value");
        return result;
    }

    const auto activation_slots = work_count + shared_count;
    std::uint64_t hidden_bytes = 0U;
    std::uint64_t activation_bytes = 0U;
    std::uint64_t output_bytes = 0U;
    std::uint64_t row_bytes = 0U;
    std::uint64_t coefficient_bytes = 0U;
    std::uint64_t group_bytes = 0U;
    std::uint64_t shared_row_bytes = 0U;
    if (!checked_bytes(hidden_rows, hidden_columns, sizeof(float), hidden_bytes) ||
        !checked_bytes(activation_slots, intermediate_columns, sizeof(float),
                       activation_bytes) ||
        !checked_bytes(activation_slots, hidden_columns, sizeof(float),
                       output_bytes) ||
        !checked_bytes(work_count, 1U, sizeof(std::uint32_t), row_bytes) ||
        !checked_bytes(work_count, 1U, sizeof(float), coefficient_bytes) ||
        !checked_bytes(groups.size(), 1U, sizeof(DeepSeekFp4PageGroup),
                       group_bytes) ||
        !checked_bytes(std::max<std::uint64_t>(shared_count, 1U), 1U,
                       sizeof(std::uint32_t), shared_row_bytes) ||
        hidden_bytes > std::numeric_limits<std::size_t>::max() ||
        activation_bytes > std::numeric_limits<std::size_t>::max() ||
        output_bytes > std::numeric_limits<std::size_t>::max() ||
        output_bytes > std::numeric_limits<std::uint64_t>::max() -
                           sizeof(unsigned int)) {
        result.errors.emplace_back("DeepSeek MoE page workspace size overflows");
        return result;
    }
    const auto host_staging_bytes = output_bytes + sizeof(unsigned int);
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for DeepSeek MoE page");
    }

    std::uint64_t allocation_calls = 0U;
    std::uint64_t allocation_bytes = 0U;
    const auto ensure_bytes = [&](void*& pointer, std::uint64_t& capacity,
                                  std::uint64_t required,
                                  const char* operation) {
        if (required <= capacity) return true;
        if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
        pointer = nullptr;
        capacity = 0U;
        if (const auto status =
                cudaMalloc(&pointer, static_cast<std::size_t>(required));
            status != cudaSuccess) {
            result = cuda_error(status, operation);
            return false;
        }
        capacity = required;
        ++allocation_calls;
        allocation_bytes += required;
        return true;
    };
    const auto ensure_floats = [&](float*& pointer, std::uint64_t& capacity,
                                   std::uint64_t required,
                                   const char* operation) {
        auto* raw = static_cast<void*>(pointer);
        const auto ok = ensure_bytes(raw, capacity, required, operation);
        pointer = static_cast<float*>(raw);
        return ok;
    };
    const auto ensure_indices = [&](std::uint32_t*& pointer,
                                    std::uint64_t& capacity,
                                    std::uint64_t required,
                                    const char* operation) {
        auto* raw = static_cast<void*>(pointer);
        const auto ok = ensure_bytes(raw, capacity, required, operation);
        pointer = static_cast<std::uint32_t*>(raw);
        return ok;
    };
    if (!ensure_floats(state.moe_hidden, state.moe_hidden_bytes, hidden_bytes,
                       "allocate DeepSeek MoE page hidden workspace") ||
        !ensure_floats(state.moe_activations, state.moe_activation_bytes,
                       activation_bytes,
                       "allocate DeepSeek MoE page activation workspace") ||
        !ensure_floats(state.moe_output, state.moe_output_bytes, output_bytes,
                       "allocate DeepSeek MoE page output workspace") ||
        !ensure_indices(state.moe_page_rows, state.moe_page_rows_bytes,
                        row_bytes, "allocate DeepSeek MoE page row list") ||
        !ensure_floats(state.moe_page_coefficients,
                       state.moe_page_coefficient_bytes, coefficient_bytes,
                       "allocate DeepSeek MoE page coefficient list") ||
        !ensure_bytes(state.moe_page_groups, state.moe_page_group_bytes,
                      group_bytes, "allocate DeepSeek MoE page group table") ||
        !ensure_indices(state.moe_page_shared_rows,
                        state.moe_page_shared_row_bytes, shared_row_bytes,
                        "allocate DeepSeek MoE page shared row list")) {
        return result;
    }
    if (state.moe_bf16_silu == nullptr) {
        // Same table as the single-row command builds. Held in a vector rather
        // than a function-local static array because a second
        // `static const std::array<float, N>` in this translation unit does not
        // survive nvcc's host pass.
        constexpr std::size_t silu_entries = 1U << 16U;
        const std::size_t silu_bytes = silu_entries * sizeof(float);
        std::vector<float> silu_table(silu_entries);
        for (std::size_t index = 0U; index < silu_entries; ++index) {
            const auto bits = static_cast<std::uint32_t>(index) << 16U;
            const float value = std::bit_cast<float>(bits);
            silu_table[index] = std::isfinite(value) ? silu_f32(value) : value;
        }
        if (const auto status = cudaMalloc(&state.moe_bf16_silu, silu_bytes);
            status != cudaSuccess) {
            return cuda_error(status, "allocate DeepSeek BF16 SiLU page table");
        }
        if (const auto status = cudaMemcpy(state.moe_bf16_silu,
                                           silu_table.data(), silu_bytes,
                                           cudaMemcpyHostToDevice);
            status != cudaSuccess) {
            static_cast<void>(cudaFree(state.moe_bf16_silu));
            state.moe_bf16_silu = nullptr;
            return cuda_error(status, "upload DeepSeek BF16 SiLU page table");
        }
        ++allocation_calls;
        allocation_bytes += silu_bytes;
    }
    if (state.moe_error == nullptr) {
        if (const auto status = cudaMalloc(&state.moe_error, sizeof(unsigned int));
            status != cudaSuccess) {
            return cuda_error(status, "allocate DeepSeek MoE page error flag");
        }
        ++allocation_calls;
        allocation_bytes += sizeof(unsigned int);
    }
    if (host_staging_bytes > state.moe_host_staging_bytes) {
        if (state.moe_host_staging != nullptr) {
            static_cast<void>(cudaFreeHost(state.moe_host_staging));
        }
        state.moe_host_staging = nullptr;
        state.moe_host_staging_bytes = 0U;
        if (const auto status = cudaMallocHost(
                &state.moe_host_staging,
                static_cast<std::size_t>(host_staging_bytes));
            status != cudaSuccess) {
            return cuda_error(status, "allocate DeepSeek MoE page host staging");
        }
        state.moe_host_staging_bytes = host_staging_bytes;
        ++allocation_calls;
        allocation_bytes += host_staging_bytes;
    }

    std::vector<std::uint32_t> host_rows;
    std::vector<float> host_coefficients;
    std::vector<DeepSeekFp4PageGroup> host_groups;
    host_rows.reserve(static_cast<std::size_t>(work_count));
    host_coefficients.reserve(static_cast<std::size_t>(work_count));
    host_groups.reserve(groups.size());
    std::uint32_t maximum_group_rows = 0U;
    for (const auto& group : groups) {
        DeepSeekFp4PageGroup entry;
        if (group.tiled_shards.front() != nullptr) {
            for (std::size_t shard = 0U; shard < group.tiled_shards.size();
                 ++shard) {
                entry.tiled[shard] = static_cast<const unsigned char*>(
                    group.tiled_shards[shard]->impl_->weights);
            }
            entry.shard_intermediate =
                static_cast<std::uint32_t>(shard_intermediate);
        } else {
            entry.w1_weights = static_cast<const unsigned char*>(group.w1->impl_->weights);
            entry.w1_scales = static_cast<const unsigned char*>(group.w1->impl_->scales);
            entry.w3_weights = static_cast<const unsigned char*>(group.w3->impl_->weights);
            entry.w3_scales = static_cast<const unsigned char*>(group.w3->impl_->scales);
            entry.w2_weights = static_cast<const unsigned char*>(group.w2->impl_->weights);
            entry.w2_scales = static_cast<const unsigned char*>(group.w2->impl_->scales);
        }
        entry.row_offset = static_cast<std::uint32_t>(host_rows.size());
        entry.row_count = static_cast<std::uint32_t>(group.rows.size());
        maximum_group_rows = std::max(maximum_group_rows, entry.row_count);
        host_rows.insert(host_rows.end(), group.rows.begin(), group.rows.end());
        host_coefficients.insert(host_coefficients.end(),
                                 group.coefficients.begin(),
                                 group.coefficients.end());
        host_groups.push_back(entry);
    }

    state.moe_weights.clear();
    state.moe_weights.reserve((groups.size() + 1U) * 3U);
    for (const auto& group : groups) {
        if (group.tiled_shards.front() != nullptr) {
            for (const auto* shard : group.tiled_shards) {
                state.moe_weights.push_back(shard->impl_);
            }
            continue;
        }
        state.moe_weights.push_back(group.w1->impl_);
        state.moe_weights.push_back(group.w3->impl_);
        state.moe_weights.push_back(group.w2->impl_);
    }
    if (shared != nullptr) {
        state.moe_weights.push_back(shared->w1->impl_);
        state.moe_weights.push_back(shared->w3->impl_);
        state.moe_weights.push_back(shared->w2->impl_);
    }

    state.moe_hidden_columns = hidden_columns;
    state.moe_intermediate_columns = intermediate_columns;
    state.moe_rows = 1U;
    state.moe_routed_count = static_cast<std::uint32_t>(work_count);
    state.moe_shared_rows = static_cast<std::uint32_t>(shared_count);
    state.moe_page_work_count = static_cast<std::uint32_t>(work_count);
    state.moe_page_shared_count = static_cast<std::uint32_t>(shared_count);
    state.moe_has_shared = shared != nullptr;
    state.moe_shared_phase_timing_valid = false;
    state.moe_host_join = false;
    state.moe_output_to_mhc = false;
    state.moe_host_callback = {};
    state.moe_kernel_launches = 0U;
    state.moe_in_flight = true;
    state.moe_poisoned = false;
    auto abort_enqueue = [&](cudaError_t status, const char* operation) {
        result = cuda_error(status, operation);
        const auto drain_status = cudaStreamSynchronize(state.stream);
        if (drain_status != cudaSuccess) {
            result.errors.emplace_back(
                std::string("drain failed DeepSeek MoE page enqueue: ") +
                cudaGetErrorString(drain_status));
            state.moe_poisoned = true;
        } else {
            state.moe_in_flight = false;
            state.moe_weights.clear();
        }
    };

    if (auto status = cudaEventRecord(state.moe_start, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record DeepSeek MoE page start");
        return result;
    }
    if (auto status = cudaMemsetAsync(state.moe_error, 0, sizeof(unsigned int),
                                      state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "reset DeepSeek MoE page error flag");
        return result;
    }
    const auto upload = [&](void* destination, const void* source,
                            std::uint64_t bytes, const char* operation) {
        if (bytes == 0U) return true;
        if (auto status = cudaMemcpyAsync(destination, source,
                                          static_cast<std::size_t>(bytes),
                                          cudaMemcpyHostToDevice, state.stream);
            status != cudaSuccess) {
            abort_enqueue(status, operation);
            return false;
        }
        return true;
    };
    if (!upload(state.moe_hidden, hidden.data(), hidden_bytes,
                "upload DeepSeek MoE page hidden rows") ||
        !upload(state.moe_page_rows, host_rows.data(), row_bytes,
                "upload DeepSeek MoE page row list") ||
        !upload(state.moe_page_coefficients, host_coefficients.data(),
                coefficient_bytes,
                "upload DeepSeek MoE page coefficient list") ||
        !upload(state.moe_page_groups, host_groups.data(), group_bytes,
                "upload DeepSeek MoE page group table") ||
        (shared != nullptr &&
         !upload(state.moe_page_shared_rows, shared_rows.data(),
                 static_cast<std::uint64_t>(shared_rows.size()) *
                     sizeof(std::uint32_t),
                 "upload DeepSeek MoE page shared row list"))) {
        return result;
    }
    if (auto status = cudaEventRecord(state.moe_hidden_uploaded, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record DeepSeek MoE page hidden upload");
        return result;
    }

    constexpr unsigned int threads = 256U;
    // The quantizer takes its row from blockIdx.y, so grid.y is the row count.
    const dim3 hidden_quantize_grid(
        static_cast<unsigned int>((hidden_columns + 127U) / 128U), hidden_rows,
        1U);
    quantize_activation_e4m3_kernel<<<hidden_quantize_grid, 128U, 0U,
                                      state.stream>>>(
        state.moe_hidden, hidden_columns, hidden_rows);
    ++state.moe_kernel_launches;
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        abort_enqueue(status, "launch DeepSeek MoE page hidden quantization");
        return result;
    }

    const auto* device_groups =
        static_cast<const DeepSeekFp4PageGroup*>(state.moe_page_groups);
    if (!groups.empty()) {
        // Derived rather than read off a descriptor: a transformed group has
        // no canonical triplet to read, and the canonical one would carry
        // exactly these values anyway.
        const auto gate_packed_columns = (hidden_columns + 1U) / 2U;
        const auto gate_scale_columns = (hidden_columns + 31U) / 32U;
        const auto down_packed_columns = (intermediate_columns + 1U) / 2U;
        const auto down_scale_columns = (intermediate_columns + 31U) / 32U;
        const bool tiled = shard_intermediate != 0U;
        const auto row_tile =
            tiled ? kDeepSeekTiledRowTile : kDeepSeekPageRowTile;
        const auto row_tiles = static_cast<unsigned int>(
            (maximum_group_rows + row_tile - 1U) / row_tile);
        // The transformed kernel gives one warp a whole 32-row transform block
        // and one output row to each of its lanes.
        const auto tiled_output_blocks = static_cast<unsigned int>(
            (intermediate_columns / 32U + kDeepSeekTiledWarps - 1U) /
            kDeepSeekTiledWarps);
        const dim3 gate_grid(
            tiled ? tiled_output_blocks
                  : static_cast<unsigned int>(intermediate_columns),
            static_cast<unsigned int>(groups.size()), row_tiles);
        if (tiled) {
            deepseek_fp4_tiled_page_gate_up_kernel<<<
                gate_grid, threads, 0U, state.stream>>>(
                state.moe_activations, state.moe_hidden, state.moe_page_rows,
                state.moe_page_coefficients, device_groups,
                static_cast<std::uint32_t>(groups.size()), hidden_columns,
                intermediate_columns, swiglu_limit, state.moe_bf16_silu,
                state.moe_error);
        } else {
            deepseek_fp4_page_gate_up_kernel<<<gate_grid, threads, 0U, state.stream>>>(
                state.moe_activations, state.moe_hidden, state.moe_page_rows,
                state.moe_page_coefficients, device_groups,
                static_cast<std::uint32_t>(groups.size()), hidden_columns,
                intermediate_columns, gate_packed_columns, gate_scale_columns,
                swiglu_limit, state.moe_bf16_silu, state.moe_error);
        }
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek FP4 page W1/W3 SwiGLU");
            return result;
        }
        const dim3 activation_grid(
            static_cast<unsigned int>((intermediate_columns + 127U) / 128U),
            static_cast<unsigned int>(work_count), 1U);
        quantize_activation_e4m3_kernel<<<activation_grid, 128U, 0U,
                                          state.stream>>>(
            state.moe_activations, intermediate_columns,
            static_cast<std::uint32_t>(work_count));
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status,
                          "launch DeepSeek page routed activation quantization");
            return result;
        }
        const auto tiled_down_blocks = static_cast<unsigned int>(
            (hidden_columns / 32U + kDeepSeekTiledWarps - 1U) /
            kDeepSeekTiledWarps);
        const dim3 down_grid(
            tiled ? tiled_down_blocks
                  : static_cast<unsigned int>(hidden_columns),
            static_cast<unsigned int>(groups.size()), row_tiles);
        if (tiled) {
            deepseek_fp4_tiled_page_down_kernel<<<
                down_grid, threads, 0U, state.stream>>>(
                state.moe_output, state.moe_activations, device_groups,
                static_cast<std::uint32_t>(groups.size()),
                intermediate_columns, hidden_columns);
        } else {
            deepseek_fp4_page_down_kernel<<<down_grid, threads, 0U, state.stream>>>(
                state.moe_output, state.moe_activations, device_groups,
                static_cast<std::uint32_t>(groups.size()), intermediate_columns,
                hidden_columns, down_packed_columns, down_scale_columns);
        }
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek FP4 page W2");
            return result;
        }
    }

    if (shared != nullptr) {
        const auto& w1 = shared->w1->impl_->descriptor;
        const auto& w2 = shared->w2->impl_->descriptor;
        float* shared_activations =
            state.moe_activations + work_count * intermediate_columns;
        float* shared_output = state.moe_output + work_count * hidden_columns;
        const auto shared_tiles = static_cast<unsigned int>(
            (shared_count + kDeepSeekPageRowTile - 1U) / kDeepSeekPageRowTile);
        const dim3 shared_gate_grid(
            static_cast<unsigned int>(intermediate_columns), shared_tiles, 1U);
        deepseek_fp8_page_gate_up_kernel<<<shared_gate_grid, threads, 0U,
                                           state.stream>>>(
            shared_activations, state.moe_hidden, state.moe_page_shared_rows,
            static_cast<std::uint32_t>(shared_count),
            static_cast<const unsigned char*>(shared->w1->impl_->weights),
            static_cast<const unsigned char*>(shared->w1->impl_->scales),
            static_cast<const unsigned char*>(shared->w3->impl_->weights),
            static_cast<const unsigned char*>(shared->w3->impl_->scales),
            hidden_columns, intermediate_columns, w1.scale_columns,
            swiglu_limit, state.moe_bf16_silu, state.moe_error);
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek shared FP8 page W1/W3 SwiGLU");
            return result;
        }
        const dim3 shared_activation_grid(
            static_cast<unsigned int>((intermediate_columns + 127U) / 128U),
            static_cast<unsigned int>(shared_count), 1U);
        quantize_activation_e4m3_kernel<<<shared_activation_grid, 128U, 0U,
                                          state.stream>>>(
            shared_activations, intermediate_columns,
            static_cast<std::uint32_t>(shared_count));
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status,
                          "launch DeepSeek page shared activation quantization");
            return result;
        }
        const dim3 shared_down_grid(static_cast<unsigned int>(hidden_columns),
                                    shared_tiles, 1U);
        deepseek_fp8_page_down_kernel<<<shared_down_grid, threads, 0U,
                                        state.stream>>>(
            shared_output, shared_activations,
            static_cast<std::uint32_t>(shared_count),
            static_cast<const unsigned char*>(shared->w2->impl_->weights),
            static_cast<const unsigned char*>(shared->w2->impl_->scales),
            intermediate_columns, hidden_columns, w2.scale_columns);
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek shared FP8 page W2");
            return result;
        }
    }

    if (auto status = cudaEventRecord(state.moe_kernel_finished, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record DeepSeek MoE page kernel completion");
        return result;
    }
    {
        std::scoped_lock lock(impl_->mutex);
        auto& device_stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        device_stats.activation_h2d_bytes += hidden_bytes;
        device_stats.matmul_calls += 3U * (groups.size() + (shared ? 1U : 0U));
        device_stats.workspace_allocation_calls += allocation_calls;
        device_stats.workspace_allocation_bytes += allocation_bytes;
        ++device_stats.deepseek_moe_calls;
        device_stats.deepseek_moe_kernel_launches += state.moe_kernel_launches;
        ++device_stats.deepseek_moe_h2d_transfers;
        device_stats.deepseek_moe_h2d_bytes += hidden_bytes;
    }
    return result;
}

ValidationResult CudaBackend::collect_deepseek_moe_rows(
    int device, std::span<float> routed_output, std::span<float> shared_output) {
    return collect_deepseek_moe(device, routed_output, shared_output);
}

ValidationResult CudaBackend::enqueue_moe(
    int device, std::span<const float> hidden, std::uint32_t rows,
    std::span<const CudaMoeExpert> routed, const CudaMoeExpert* shared) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back("MoE command targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    if (state.moe_in_flight) {
        result.errors.emplace_back("MoE workspace already has an in-flight command");
        return result;
    }
    const auto expert_count = routed.size() + (shared == nullptr ? 0U : 1U);
    if (rows == 0U || expert_count == 0U || expert_count > kMaxMoeExperts ||
        routed.size() > kMaxRoutedMoeExperts) {
        result.errors.emplace_back("MoE command has an unsupported row or expert count");
        return result;
    }

    // The batch is single-encoding. The first expert's gate fixes it and every
    // other weight must agree, so a mixed batch is rejected rather than
    // silently dispatched to the wrong decode rule.
    const auto* first_gate = routed.empty() ? (shared == nullptr ? nullptr
                                                                 : shared->gate)
                                            : routed.front().gate;
    if (first_gate == nullptr || !first_gate->valid()) {
        result.errors.emplace_back("MoE command has no valid leading expert");
        return result;
    }
    const auto batch_encoding = first_gate->impl_->descriptor.encoding;
    const bool nvfp4_batch = batch_encoding == CudaWeightEncoding::Nvfp4Group16;
    const bool plain_batch = batch_encoding == CudaWeightEncoding::Plain;
    if (!nvfp4_batch && !plain_batch &&
        batch_encoding != CudaWeightEncoding::OffsetPackedInt4) {
        result.errors.emplace_back("MoE command has an unsupported weight encoding");
        return result;
    }

    std::uint64_t hidden_columns = 0U;
    std::uint64_t intermediate_columns = 0U;
    const auto validate_expert = [&](const CudaMoeExpert& expert,
                                     bool shared_expert) {
        const std::array<const CudaWeight*, 3> weights{
            expert.gate, expert.up, expert.down};
        for (const auto* weight : weights) {
            const bool compatible =
                weight != nullptr && weight->valid() &&
                weight->impl_->device == device &&
                weight->impl_->descriptor.encoding == batch_encoding &&
                (nvfp4_batch
                     ? (weight->impl_->descriptor.dtype == SafetensorsDtype::U8 &&
                        weight->impl_->descriptor.group_size == 16U &&
                        std::isfinite(weight->impl_->descriptor.global_scale) &&
                        weight->impl_->descriptor.global_scale > 0.0F)
                 : plain_batch
                     ? weight->impl_->descriptor.dtype == SafetensorsDtype::Bf16
                     : (weight->impl_->descriptor.dtype == SafetensorsDtype::I32 &&
                        weight->impl_->descriptor.group_size == 128U));
            if (!compatible) {
                result.errors.emplace_back(
                    "MoE command contains an incompatible CUDA weight");
                return false;
            }
        }
        const auto& gate = expert.gate->impl_->descriptor;
        const auto& up = expert.up->impl_->descriptor;
        const auto& down = expert.down->impl_->descriptor;
        const auto expected_down_packed = nvfp4_batch
            ? (down.columns + 1U) / 2U : (down.columns + 7U) / 8U;
        const auto expected_down_scales = nvfp4_batch
            ? (down.columns + 15U) / 16U : (down.columns + 127U) / 128U;
        const bool packing_valid = plain_batch ||
            (gate.packed_columns == up.packed_columns &&
             gate.scale_columns == up.scale_columns &&
             down.packed_columns == expected_down_packed &&
             down.scale_columns == expected_down_scales);
        if (gate.rows == 0U || gate.columns == 0U ||
            up.rows != gate.rows || up.columns != gate.columns ||
            down.rows != gate.columns || down.columns != gate.rows ||
            !packing_valid) {
            result.errors.emplace_back("MoE gate/up/down shapes are incompatible");
            return false;
        }
        // The NVFP4 kernels leave the routing coefficient to the caller,
        // because scaling before the down projection is not float-equal to
        // scaling after it and Laguna's reference scales after.
        if (!std::isfinite(expert.coefficient) ||
            ((shared_expert || nvfp4_batch || plain_batch) &&
             expert.coefficient != 1.0F)) {
            result.errors.emplace_back("MoE expert coefficient is invalid");
            return false;
        }
        if (hidden_columns == 0U) {
            hidden_columns = gate.columns;
            intermediate_columns = gate.rows;
        } else if (hidden_columns != gate.columns ||
                   intermediate_columns != gate.rows) {
            result.errors.emplace_back(
                "MoE experts do not share one activation shape");
            return false;
        }
        return true;
    };
    for (const auto& expert : routed) {
        if (!validate_expert(expert, false)) return result;
    }
    if (shared != nullptr && !validate_expert(*shared, true)) return result;

    std::uint64_t hidden_elements = 0U;
    if (!checked_bytes(rows, hidden_columns, 1U, hidden_elements) ||
        hidden.size() != hidden_elements ||
        std::any_of(hidden.begin(), hidden.end(),
                    [](float value) { return !std::isfinite(value); })) {
        result.errors.emplace_back("MoE hidden rows are incompatible");
        return result;
    }

    std::uint64_t hidden_bytes = 0U;
    std::uint64_t activation_rows = 0U;
    std::uint64_t activation_bytes = 0U;
    std::uint64_t output_bytes = 0U;
    if (!checked_bytes(rows, hidden_columns, sizeof(float), hidden_bytes) ||
        !checked_bytes(expert_count, rows, 1U, activation_rows) ||
        !checked_bytes(activation_rows, intermediate_columns, sizeof(float),
                       activation_bytes) ||
        !checked_bytes(activation_rows, hidden_columns, sizeof(float),
                       output_bytes) ||
        output_bytes > std::numeric_limits<std::uint64_t>::max() -
                           sizeof(unsigned int)) {
        result.errors.emplace_back("MoE workspace size overflows");
        return result;
    }
    if (hidden_columns > std::numeric_limits<unsigned int>::max() ||
        intermediate_columns > std::numeric_limits<unsigned int>::max() ||
        activation_rows > std::numeric_limits<unsigned int>::max()) {
        result.errors.emplace_back("MoE CUDA grid dimensions overflow");
        return result;
    }
    const auto host_staging_bytes = output_bytes + sizeof(unsigned int);
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for MoE");
    }

    std::uint64_t allocation_calls = 0U;
    std::uint64_t allocation_bytes = 0U;
    const auto ensure_workspace = [&](float*& pointer, std::uint64_t& capacity,
                                      std::uint64_t required,
                                      const char* operation) {
        if (required <= capacity) return true;
        if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
        pointer = nullptr;
        capacity = 0U;
        if (const auto status =
                cudaMalloc(&pointer, static_cast<std::size_t>(required));
            status != cudaSuccess) {
            result = cuda_error(status, operation);
            return false;
        }
        capacity = required;
        ++allocation_calls;
        allocation_bytes += required;
        return true;
    };
    if (!ensure_workspace(state.moe_hidden, state.moe_hidden_bytes, hidden_bytes,
                          "allocate MoE hidden workspace") ||
        !ensure_workspace(state.moe_activations, state.moe_activation_bytes,
                          activation_bytes,
                          "allocate MoE activation workspace") ||
        !ensure_workspace(state.moe_output, state.moe_output_bytes, output_bytes,
                          "allocate MoE output workspace")) {
        return result;
    }
    if (state.moe_error == nullptr) {
        if (const auto status = cudaMalloc(&state.moe_error, sizeof(unsigned int));
            status != cudaSuccess) {
            return cuda_error(status, "allocate MoE error flag");
        }
        ++allocation_calls;
        allocation_bytes += sizeof(unsigned int);
    }
    if (host_staging_bytes > state.moe_host_staging_bytes) {
        if (state.moe_host_staging != nullptr) {
            static_cast<void>(cudaFreeHost(state.moe_host_staging));
        }
        state.moe_host_staging = nullptr;
        state.moe_host_staging_bytes = 0U;
        if (const auto status = cudaMallocHost(
                &state.moe_host_staging,
                static_cast<std::size_t>(host_staging_bytes));
            status != cudaSuccess) {
            return cuda_error(status, "allocate MoE host staging");
        }
        state.moe_host_staging_bytes = host_staging_bytes;
        ++allocation_calls;
        allocation_bytes += host_staging_bytes;
    }

    PackedInt4MoeBatch batch;
    Nvfp4MoeBatch nvfp4_batch_data;
    PlainBf16MoeBatch plain_batch_data;
    state.moe_weights.clear();
    state.moe_weights.reserve(expert_count * 3U);
    const auto append_expert = [&](const CudaMoeExpert& expert,
                                   std::size_t index) {
        if (plain_batch) {
            plain_batch_data.gate_weights[index] =
                static_cast<const __nv_bfloat16*>(expert.gate->impl_->weights);
            plain_batch_data.up_weights[index] =
                static_cast<const __nv_bfloat16*>(expert.up->impl_->weights);
            plain_batch_data.down_weights[index] =
                static_cast<const __nv_bfloat16*>(expert.down->impl_->weights);
        } else if (nvfp4_batch) {
            nvfp4_batch_data.gate_weights[index] =
                static_cast<const unsigned char*>(expert.gate->impl_->weights);
            nvfp4_batch_data.gate_scales[index] =
                static_cast<const unsigned char*>(expert.gate->impl_->scales);
            nvfp4_batch_data.up_weights[index] =
                static_cast<const unsigned char*>(expert.up->impl_->weights);
            nvfp4_batch_data.up_scales[index] =
                static_cast<const unsigned char*>(expert.up->impl_->scales);
            nvfp4_batch_data.down_weights[index] =
                static_cast<const unsigned char*>(expert.down->impl_->weights);
            nvfp4_batch_data.down_scales[index] =
                static_cast<const unsigned char*>(expert.down->impl_->scales);
            nvfp4_batch_data.gate_global_scales[index] =
                expert.gate->impl_->descriptor.global_scale;
            nvfp4_batch_data.up_global_scales[index] =
                expert.up->impl_->descriptor.global_scale;
            nvfp4_batch_data.down_global_scales[index] =
                expert.down->impl_->descriptor.global_scale;
        } else {
            batch.gate_weights[index] = static_cast<const std::uint32_t*>(
                expert.gate->impl_->weights);
            batch.gate_scales[index] = static_cast<const __nv_bfloat16*>(
                expert.gate->impl_->scales);
            batch.up_weights[index] = static_cast<const std::uint32_t*>(
                expert.up->impl_->weights);
            batch.up_scales[index] = static_cast<const __nv_bfloat16*>(
                expert.up->impl_->scales);
            batch.down_weights[index] = static_cast<const std::uint32_t*>(
                expert.down->impl_->weights);
            batch.down_scales[index] = static_cast<const __nv_bfloat16*>(
                expert.down->impl_->scales);
            batch.coefficients[index] = expert.coefficient;
        }
        state.moe_weights.push_back(expert.gate->impl_);
        state.moe_weights.push_back(expert.up->impl_);
        state.moe_weights.push_back(expert.down->impl_);
    };
    for (std::size_t index = 0U; index < routed.size(); ++index) {
        append_expert(routed[index], index);
    }
    if (shared != nullptr) append_expert(*shared, routed.size());
    batch.count = static_cast<std::uint32_t>(expert_count);
    batch.rows = rows;
    nvfp4_batch_data.count = batch.count;
    nvfp4_batch_data.rows = rows;
    plain_batch_data.count = batch.count;
    plain_batch_data.rows = rows;

    state.moe_hidden_columns = hidden_columns;
    state.moe_intermediate_columns = intermediate_columns;
    state.moe_rows = rows;
    // The generic command's shared expert produces one row per input row, so
    // collection sizes its download from the same count.
    state.moe_shared_rows = rows;
    state.moe_routed_count = static_cast<std::uint32_t>(routed.size());
    state.moe_has_shared = shared != nullptr;
    state.moe_shared_phase_timing_valid = false;
    state.moe_host_join = false;
    state.moe_output_to_mhc = false;
    state.moe_host_callback = {};
    state.moe_kernel_launches = 0U;
    state.moe_in_flight = true;
    state.moe_poisoned = false;
    const auto abort_enqueue = [&](cudaError_t status, const char* operation) {
        result = cuda_error(status, operation);
        const auto drain_status = cudaStreamSynchronize(state.stream);
        if (drain_status != cudaSuccess) {
            result.errors.emplace_back(
                std::string("drain failed MoE enqueue: ") +
                cudaGetErrorString(drain_status));
            state.moe_poisoned = true;
        } else {
            state.moe_in_flight = false;
            state.moe_weights.clear();
        }
    };

    if (auto status = cudaEventRecord(state.moe_start, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record MoE start");
        return result;
    }
    if (auto status = cudaMemsetAsync(
            state.moe_error, 0, sizeof(unsigned int), state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "reset MoE error flag");
        return result;
    }
    if (auto status = cudaMemcpyAsync(
            state.moe_hidden, hidden.data(), static_cast<std::size_t>(hidden_bytes),
            cudaMemcpyHostToDevice, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "upload MoE hidden rows");
        return result;
    }
    if (auto status = cudaEventRecord(state.moe_hidden_uploaded, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record MoE hidden upload");
        return result;
    }

    const auto& gate = (routed.empty() ? shared->gate : routed.front().gate)
                           ->impl_->descriptor;
    const auto& down = (routed.empty() ? shared->down : routed.front().down)
                           ->impl_->descriptor;
    constexpr unsigned int threads = 256U;
    const dim3 gate_grid(static_cast<unsigned int>(intermediate_columns),
                         static_cast<unsigned int>(activation_rows), 1U);
    constexpr unsigned int warps_per_block = 8U;
    const dim3 plain_gate_grid(
        static_cast<unsigned int>((intermediate_columns + warps_per_block - 1U) /
                                  warps_per_block),
        static_cast<unsigned int>(activation_rows), 1U);
    // Counted once per MoE command on the gate/up dispatch; the down kernel
    // mirrors the same branch, so counting both would double every entry.
    record_cuda_matmul_route(plain_batch    ? CudaMatmulRoute::MoePlainBf16
                             : nvfp4_batch  ? CudaMatmulRoute::MoeNvfp4Group16
                                            : CudaMatmulRoute::MoePackedInt4);
    if (plain_batch) {
        plain_bf16_moe_gate_up_kernel<<<plain_gate_grid, threads, 0U, state.stream>>>(
            state.moe_activations, state.moe_hidden, plain_batch_data,
            hidden_columns, intermediate_columns, state.moe_error);
    } else if (nvfp4_batch) {
        nvfp4_moe_gate_up_kernel<<<gate_grid, threads, 0U, state.stream>>>(
            state.moe_activations, state.moe_hidden, nvfp4_batch_data,
            hidden_columns, intermediate_columns, gate.packed_columns,
            gate.scale_columns, gate.group_size, state.moe_error);
    } else {
        packed_int4_moe_gate_up_kernel<<<gate_grid, threads, 0U, state.stream>>>(
            state.moe_activations, state.moe_hidden, batch, hidden_columns,
            intermediate_columns, gate.packed_columns, gate.scale_columns,
            gate.group_size, state.moe_error);
    }
    ++state.moe_kernel_launches;
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        abort_enqueue(status, "launch MoE gate/up SwiGLU");
        return result;
    }
    const dim3 down_grid(static_cast<unsigned int>(hidden_columns),
                         static_cast<unsigned int>(activation_rows), 1U);
    const dim3 plain_down_grid(
        static_cast<unsigned int>((hidden_columns + warps_per_block - 1U) /
                                  warps_per_block),
        static_cast<unsigned int>(activation_rows), 1U);
    if (plain_batch) {
        plain_bf16_moe_down_kernel<<<plain_down_grid, threads, 0U, state.stream>>>(
            state.moe_output, state.moe_activations, plain_batch_data,
            intermediate_columns, hidden_columns, state.moe_error);
    } else if (nvfp4_batch) {
        nvfp4_moe_down_kernel<<<down_grid, threads, 0U, state.stream>>>(
            state.moe_output, state.moe_activations, nvfp4_batch_data,
            intermediate_columns, hidden_columns, down.packed_columns,
            down.scale_columns, down.group_size, state.moe_error);
    } else {
        packed_int4_moe_down_kernel<<<down_grid, threads, 0U, state.stream>>>(
            state.moe_output, state.moe_activations, batch, intermediate_columns,
            hidden_columns, down.packed_columns, down.scale_columns,
            down.group_size, state.moe_error);
    }
    ++state.moe_kernel_launches;
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        abort_enqueue(status, "launch MoE down projection");
        return result;
    }
    if (auto status = cudaEventRecord(state.moe_kernel_finished, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record MoE kernel completion");
        return result;
    }
    {
        std::scoped_lock lock(impl_->mutex);
        auto& device_stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        device_stats.activation_h2d_bytes += hidden_bytes;
        device_stats.matmul_calls += 3U * expert_count;
        device_stats.workspace_allocation_calls += allocation_calls;
        device_stats.workspace_allocation_bytes += allocation_bytes;
        ++device_stats.deepseek_moe_calls;
        device_stats.deepseek_moe_kernel_launches += state.moe_kernel_launches;
        ++device_stats.deepseek_moe_h2d_transfers;
        device_stats.deepseek_moe_h2d_bytes += hidden_bytes;
    }
    return result;
}

ValidationResult CudaBackend::dsv4_tier_reserve(
    int device, std::uint32_t layers, std::uint32_t experts) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek tier reserve targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    if (state.tier_layers != 0U) {
        result.errors.emplace_back("DeepSeek tier is already reserved");
        return result;
    }
    if (layers == 0U || experts == 0U) {
        result.errors.emplace_back("DeepSeek tier needs a positive geometry");
        return result;
    }
    if (const auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for DeepSeek tier");
    }
    const auto entries = static_cast<std::size_t>(layers) * experts;
    for (std::size_t array = 0U; array < 6U; ++array) {
        state.tier_host_pointers[array].assign(entries, nullptr);
        void* device_array = nullptr;
        if (const auto status = cudaMalloc(
                &device_array, entries * sizeof(const unsigned char*));
            status != cudaSuccess) {
            return cuda_error(status, "allocate DeepSeek tier pointer table");
        }
        state.tier_device_pointers[array] =
            static_cast<const unsigned char**>(device_array);
    }
    void* selection_host = nullptr;
    if (const auto status = cudaMallocHost(
            &selection_host, sizeof(CudaDsv4TierSelection));
        status != cudaSuccess) {
        return cuda_error(status, "allocate DeepSeek tier selection staging");
    }
    state.tier_selection_host =
        static_cast<CudaDsv4TierSelection*>(selection_host);
    *state.tier_selection_host = CudaDsv4TierSelection{};
    void* selection_device = nullptr;
    if (const auto status =
            cudaMalloc(&selection_device, sizeof(CudaDsv4TierSelection));
        status != cudaSuccess) {
        return cuda_error(status, "allocate DeepSeek tier selection");
    }
    state.tier_selection_device =
        static_cast<CudaDsv4TierSelection*>(selection_device);
    state.tier_layers = layers;
    state.tier_experts = experts;
    return result;
}

ValidationResult CudaBackend::dsv4_tier_add(
    int device, std::uint32_t layer, std::uint32_t expert,
    const CudaWeight& w1, const CudaWeight& w3, const CudaWeight& w2) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek tier add targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    if (state.tier_layers == 0U) {
        result.errors.emplace_back("DeepSeek tier is not reserved");
        return result;
    }
    if (layer >= state.tier_layers || expert >= state.tier_experts) {
        result.errors.emplace_back("DeepSeek tier entry is out of range");
        return result;
    }
    const std::array<const CudaWeight*, 3U> weights{&w1, &w3, &w2};
    for (const auto* weight : weights) {
        if (weight == nullptr || !weight->valid() ||
            weight->impl_->device != device) {
            result.errors.emplace_back(
                "DeepSeek tier entry weight is invalid or on another device");
            return result;
        }
        if (weight->impl_->descriptor.encoding !=
            CudaWeightEncoding::Fp4E2m1Group32) {
            result.errors.emplace_back(
                "DeepSeek tier entry weight is not FP4 E2M1 group 32");
            return result;
        }
    }
    const auto entry =
        static_cast<std::size_t>(layer) * state.tier_experts + expert;
    if (state.tier_host_pointers[0][entry] != nullptr) {
        result.errors.emplace_back(
            "DeepSeek tier already holds this layer and expert");
        return result;
    }
    const auto& gate_descriptor = w1.impl_->descriptor;
    const auto& down_descriptor = w2.impl_->descriptor;
    if (state.tier_gate_packed_columns == 0U) {
        state.tier_gate_packed_columns = gate_descriptor.packed_columns;
        state.tier_gate_scale_columns = gate_descriptor.scale_columns;
        state.tier_down_packed_columns = down_descriptor.packed_columns;
        state.tier_down_scale_columns = down_descriptor.scale_columns;
    } else if (gate_descriptor.packed_columns != state.tier_gate_packed_columns ||
               gate_descriptor.scale_columns != state.tier_gate_scale_columns ||
               down_descriptor.packed_columns != state.tier_down_packed_columns ||
               down_descriptor.scale_columns != state.tier_down_scale_columns) {
        result.errors.emplace_back(
            "DeepSeek tier entry shape differs from the tier's first entry");
        return result;
    }
    const std::array<const CudaWeight*, 6U> ordered{&w1, &w1, &w3, &w3, &w2, &w2};
    for (std::size_t array = 0U; array < 6U; ++array) {
        state.tier_host_pointers[array][entry] = static_cast<const unsigned char*>(
            array % 2U == 0U ? ordered[array]->impl_->weights
                             : ordered[array]->impl_->scales);
    }
    ++state.tier_installed;
    return result;
}

ValidationResult CudaBackend::dsv4_tier_commit(int device) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek tier commit targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    if (state.tier_layers == 0U) {
        result.errors.emplace_back("DeepSeek tier is not reserved");
        return result;
    }
    if (const auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for DeepSeek tier commit");
    }
    const auto entries =
        static_cast<std::size_t>(state.tier_layers) * state.tier_experts;
    for (std::size_t array = 0U; array < 6U; ++array) {
        if (const auto status = cudaMemcpy(
                state.tier_device_pointers[array],
                state.tier_host_pointers[array].data(),
                entries * sizeof(const unsigned char*), cudaMemcpyHostToDevice);
            status != cudaSuccess) {
            return cuda_error(status, "upload DeepSeek tier pointer table");
        }
    }
    state.tier_committed = true;
    return result;
}

bool CudaBackend::dsv4_tier_active(int device) const noexcept {
    const auto found = impl_->devices.find(device);
    return found != impl_->devices.end() && found->second.tier_installed != 0U &&
           found->second.tier_committed;
}

CudaDsv4TierSelection* CudaBackend::dsv4_tier_selection(int device) noexcept {
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) return nullptr;
    return found->second.tier_selection_host;
}

ValidationResult CudaBackend::collect_deepseek_moe(
    int device, std::span<float> routed_output,
    std::span<float> shared_output) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek MoE collect targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    const auto reset_host_join = [&state] {
        state.moe_host_join = false;
        state.moe_output_to_mhc = false;
        state.moe_host_callback = {};
        for (std::uint32_t index = 0U;
             index < state.moe_host_callback_count; ++index) {
            state.moe_host_callbacks[index] = {};
        }
        state.moe_host_callback_count = 0U;
    };
    if (!state.moe_in_flight && state.moe_host_callback_count == 0U) {
        result.errors.emplace_back(
            "DeepSeek MoE collect has no matching in-flight command");
        return result;
    }
    if (state.moe_poisoned) {
        result.errors.emplace_back(
            "DeepSeek MoE workspace is poisoned by an unconfirmed CUDA drain");
        if (const auto select_status = cudaSetDevice(device);
            select_status == cudaSuccess) {
            if (const auto drain_status = cudaDeviceSynchronize();
                drain_status == cudaSuccess) {
                state.moe_in_flight = false;
                state.moe_poisoned = false;
                state.moe_weights.clear();
                reset_host_join();
            } else {
                result.errors.emplace_back(
                    std::string("retry poisoned DeepSeek MoE drain: ") +
                    cudaGetErrorString(drain_status));
            }
        } else {
            result.errors.emplace_back(
                std::string("select poisoned DeepSeek MoE device: ") +
                cudaGetErrorString(select_status));
        }
        return result;
    }
    auto drain_without_output = [&]() {
        if (const auto status = cudaSetDevice(device); status != cudaSuccess) {
            result.errors.emplace_back(
                std::string("select CUDA device while draining DeepSeek MoE: ") +
                cudaGetErrorString(status));
        }
        const auto drain_status = cudaEventSynchronize(state.moe_kernel_finished);
        if (drain_status != cudaSuccess) {
            result.errors.emplace_back(
                std::string("drain DeepSeek MoE kernels: ") +
                cudaGetErrorString(drain_status));
            state.moe_poisoned = true;
        } else {
            state.moe_in_flight = false;
            state.moe_weights.clear();
            reset_host_join();
        }
    };
    std::uint64_t routed_rows = 0U;
    std::uint64_t routed_elements = 0U;
    std::uint64_t shared_elements = 0U;
    const bool output_to_mhc = state.moe_output_to_mhc;
    if (!checked_bytes(state.moe_routed_count, state.moe_rows, 1U,
                       routed_rows) ||
        !checked_bytes(routed_rows, state.moe_hidden_columns, 1U,
                       routed_elements) ||
        !checked_bytes(state.moe_shared_rows, state.moe_hidden_columns, 1U,
                       shared_elements) ||
        routed_output.size() != routed_elements ||
        (output_to_mhc
             ? (!shared_output.empty() &&
                shared_output.size() != shared_elements)
             : (state.moe_has_shared
                    ? shared_output.size() != shared_elements
                    : !shared_output.empty()))) {
        result.errors.emplace_back(
            "DeepSeek MoE collect output spans do not match the enqueued command");
        drain_without_output();
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        result = cuda_error(status, "select CUDA device for DeepSeek MoE collect");
        drain_without_output();
        return result;
    }
    auto abort_collect = [&](cudaError_t status, const char* operation) {
        result = cuda_error(status, operation);
        const auto drain_status = cudaStreamSynchronize(state.stream);
        if (drain_status != cudaSuccess) {
            result.errors.emplace_back(
                std::string("drain failed DeepSeek MoE collect: ") +
                cudaGetErrorString(drain_status));
            state.moe_poisoned = true;
        } else {
            state.moe_in_flight = false;
            state.moe_weights.clear();
            reset_host_join();
        }
    };
    if (auto status = cudaEventRecord(state.moe_download_started, state.stream);
        status != cudaSuccess) {
        abort_collect(status, "record DeepSeek MoE download start");
        return result;
    }
    const auto routed_bytes =
        static_cast<std::uint64_t>(routed_output.size_bytes());
    const auto shared_bytes =
        static_cast<std::uint64_t>(shared_output.size_bytes());
    const auto downloaded_bytes = routed_bytes + shared_bytes;
    auto* host_bytes = static_cast<std::byte*>(state.moe_host_staging);
    auto* host_error = reinterpret_cast<unsigned int*>(
        host_bytes + static_cast<std::ptrdiff_t>(downloaded_bytes));
    if (downloaded_bytes != 0U) {
        if (auto status = cudaMemcpyAsync(
                host_bytes, state.moe_output,
                static_cast<std::size_t>(downloaded_bytes),
                cudaMemcpyDeviceToHost, state.stream);
            status != cudaSuccess) {
            abort_collect(status, "stage DeepSeek MoE expert outputs");
            return result;
        }
    }
    if (auto status = cudaMemcpyAsync(
            host_error, state.moe_error, sizeof(unsigned int),
            cudaMemcpyDeviceToHost, state.stream);
        status != cudaSuccess) {
        abort_collect(status, "stage DeepSeek MoE error flag");
        return result;
    }
    if (auto status = cudaEventRecord(state.moe_completed, state.stream);
        status != cudaSuccess) {
        abort_collect(status, "record DeepSeek MoE completion");
        return result;
    }
    const auto wait_started = std::chrono::steady_clock::now();
    const auto wait_status = cudaEventSynchronize(state.moe_completed);
    const auto wait_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wait_started).count());
    if (wait_status != cudaSuccess) {
        result = cuda_error(wait_status, "synchronize DeepSeek MoE completion");
        const auto drain_status = cudaStreamSynchronize(state.stream);
        if (drain_status != cudaSuccess) {
            result.errors.emplace_back(
                std::string("drain failed DeepSeek MoE execution: ") +
                cudaGetErrorString(drain_status));
            state.moe_poisoned = true;
        } else {
            state.moe_in_flight = false;
            state.moe_weights.clear();
            reset_host_join();
        }
        return result;
    }
    for (auto& [attention_device, attention_state] : impl_->devices) {
        if (attention_state.dsv4_attention_prepare_host_command_count != 0U) {
            if (auto status = cudaSetDevice(attention_device);
                status != cudaSuccess) {
                result.errors.emplace_back(
                    std::string("select deferred attention callback device: ") +
                    cudaGetErrorString(status));
            } else if (auto status = cudaStreamSynchronize(
                           attention_state.stream);
                       status != cudaSuccess) {
                result.errors.emplace_back(
                    std::string("drain deferred attention callback stream: ") +
                    cudaGetErrorString(status));
            }
        }
        for (std::uint32_t index = 0U;
             index <
                 attention_state.dsv4_attention_prepare_host_command_count;
             ++index) {
            if (attention_state
                    .dsv4_attention_prepare_host_commands[index].failed) {
                result.errors.emplace_back(
                    "DeepSeek attention preparation host callback failed");
            }
        }
        attention_state.dsv4_attention_prepare_host_command_count = 0U;
        attention_state.dsv4_deferred_attention_command_count = 0U;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        result.errors.emplace_back(
            std::string("restore DeepSeek MoE device after callback drain: ") +
            cudaGetErrorString(status));
    }
    if (state.dsv4_deferred_attention_source_device >= 0) {
        const auto source_found = impl_->devices.find(
            state.dsv4_deferred_attention_source_device);
        if (source_found == impl_->devices.end()) {
            result.errors.emplace_back(
                "deferred DeepSeek attention source device disappeared");
        } else if (impl_->detailed_timing) {
            auto& source = source_found->second;
            const bool cross =
                state.dsv4_deferred_attention_cross_transition;
            float source_h2d_ms = 0.0F;
            float attention_ms = 0.0F;
            float mhc_ms = 0.0F;
            float kernel_ms = 0.0F;
            float d2h_ms = 0.0F;
            const auto measure = [&](float& output, cudaEvent_t begin,
                                     cudaEvent_t end,
                                     const char* operation) {
                if (auto status = cudaEventElapsedTime(&output, begin, end);
                    status != cudaSuccess) {
                    result.errors.emplace_back(
                        std::string(operation) + ": " +
                        cudaGetErrorString(status));
                    return false;
                }
                return true;
            };
            if (auto status = cudaSetDevice(
                    state.dsv4_deferred_attention_source_device);
                status != cudaSuccess) {
                result.errors.emplace_back(
                    std::string("select deferred attention source: ") +
                    cudaGetErrorString(status));
            } else if (cross) {
                float source_d2h_ms = 0.0F;
                float target_h2d_ms = 0.0F;
                float target_kernel_ms = 0.0F;
                float target_d2h_ms = 0.0F;
                const bool source_measured =
                    measure(source_h2d_ms, source.activation_start,
                            source.activation_uploaded,
                            "measure deferred attention upload") &&
                    measure(attention_ms, source.activation_uploaded,
                            source.mhc_transition_finished,
                            "measure deferred attention kernels") &&
                    measure(source_d2h_ms, source.mhc_transition_finished,
                            source.activation_downloaded,
                            "measure deferred attention source download");
                if (auto status = cudaSetDevice(device);
                    status != cudaSuccess) {
                    result.errors.emplace_back(
                        std::string("restore deferred attention target: ") +
                        cudaGetErrorString(status));
                } else {
                    const bool target_measured =
                        measure(target_h2d_ms, state.activation_start,
                                state.activation_uploaded,
                                "measure deferred mHC upload") &&
                        measure(target_kernel_ms, state.activation_uploaded,
                                state.kernel_finished,
                                "measure deferred mHC/router kernels") &&
                        measure(mhc_ms, state.activation_uploaded,
                                state.router_started,
                                "measure deferred mHC transition") &&
                        measure(target_d2h_ms, state.kernel_finished,
                                state.activation_downloaded,
                                "measure deferred target download");
                    if (source_measured && target_measured) {
                        source_h2d_ms += target_h2d_ms;
                        kernel_ms = attention_ms + target_kernel_ms;
                        d2h_ms = source_d2h_ms + target_d2h_ms;
                    }
                }
            } else {
                const bool measured =
                    measure(source_h2d_ms, source.activation_start,
                            source.activation_uploaded,
                            "measure deferred attention upload") &&
                    measure(kernel_ms, source.activation_uploaded,
                            source.kernel_finished,
                            "measure deferred attention/mHC kernels") &&
                    measure(attention_ms, source.activation_uploaded,
                            source.mhc_transition_finished,
                            "measure deferred attention kernels") &&
                    measure(mhc_ms, source.mhc_transition_finished,
                            source.router_started,
                            "measure deferred mHC transition") &&
                    measure(d2h_ms, source.kernel_finished,
                            source.activation_downloaded,
                            "measure deferred attention status download");
                static_cast<void>(measured);
            }
            if (result.ok()) {
                const auto to_nanoseconds = [](float milliseconds) {
                    return static_cast<std::uint64_t>(std::llround(
                        static_cast<double>(milliseconds) * 1.0e6));
                };
                std::uint64_t mhc_timing_clamped_samples = 0U;
                const auto h2d_nanoseconds =
                    to_nanoseconds(source_h2d_ms);
                const auto attention_nanoseconds =
                    to_nanoseconds(attention_ms);
                const auto mhc_nanoseconds =
                    event_milliseconds_to_nanoseconds(
                        mhc_ms, mhc_timing_clamped_samples);
                const auto kernel_nanoseconds = to_nanoseconds(kernel_ms);
                const auto d2h_nanoseconds = to_nanoseconds(d2h_ms);
                std::scoped_lock lock(impl_->mutex);
                auto& source_stats = *std::find_if(
                    impl_->stats.devices.begin(), impl_->stats.devices.end(),
                    [&](const auto& value) {
                        return value.device ==
                            state.dsv4_deferred_attention_source_device;
                    });
                source_stats.dsv4_paged_attention_h2d_nanoseconds +=
                    h2d_nanoseconds;
                source_stats.dsv4_paged_attention_kernel_nanoseconds +=
                    attention_nanoseconds;
                source_stats.dsv4_paged_attention_d2h_nanoseconds +=
                    d2h_nanoseconds;
                source_stats.dsv4_paged_attention_nanoseconds +=
                    h2d_nanoseconds + kernel_nanoseconds + d2h_nanoseconds;
                source_stats.activation_h2d_nanoseconds += h2d_nanoseconds;
                source_stats.kernel_nanoseconds += kernel_nanoseconds;
                source_stats.activation_d2h_nanoseconds += d2h_nanoseconds;
                source_stats.dsv4_mhc_kernel_nanoseconds += mhc_nanoseconds;
                source_stats.dsv4_mhc_d2h_nanoseconds += d2h_nanoseconds;
                source_stats.dsv4_mhc_nanoseconds +=
                    mhc_nanoseconds + d2h_nanoseconds;
                source_stats.dsv4_mhc_device_nanoseconds +=
                    mhc_nanoseconds + d2h_nanoseconds;
                source_stats.dsv4_mhc_timing_clamped_samples +=
                    mhc_timing_clamped_samples;
            }
        }
        state.dsv4_deferred_attention_source_device = -1;
        state.dsv4_deferred_attention_cross_transition = false;
        if (auto status = cudaSetDevice(device); status != cudaSuccess) {
            result.errors.emplace_back(
                std::string("restore DeepSeek MoE device after timing: ") +
                cudaGetErrorString(status));
        }
    }
    bool host_callback_failed =
        state.moe_host_join && state.moe_host_callback.failed;
    const bool shared_phase_timing_valid =
        state.moe_shared_phase_timing_valid;
    unsigned int first_upstream_failure =
        state.moe_host_callback.upstream_failure_value;
    for (std::uint32_t index = 0U;
         index < state.moe_host_callback_count; ++index) {
        host_callback_failed = host_callback_failed ||
            state.moe_host_callbacks[index].failed;
        if (first_upstream_failure == 0U) {
            first_upstream_failure =
                state.moe_host_callbacks[index].upstream_failure_value;
        }
    }
    state.moe_in_flight = false;
    state.moe_weights.clear();
    reset_host_join();

    float h2d_milliseconds = 0.0F;
    float kernel_milliseconds = 0.0F;
    float d2h_milliseconds = 0.0F;
    float shared_input_quantize_milliseconds = 0.0F;
    float shared_gate_up_milliseconds = 0.0F;
    float shared_activation_quantize_milliseconds = 0.0F;
    float shared_down_milliseconds = 0.0F;
    if (auto status = cudaEventElapsedTime(
            &h2d_milliseconds, state.moe_start, state.moe_hidden_uploaded);
        status != cudaSuccess) {
        return cuda_error(status, "measure DeepSeek MoE hidden upload");
    }
    if (auto status = cudaEventElapsedTime(
            &kernel_milliseconds, state.moe_hidden_uploaded,
            state.moe_kernel_finished);
        status != cudaSuccess) {
        return cuda_error(status, "measure DeepSeek MoE kernels");
    }
    if (auto status = cudaEventElapsedTime(
            &d2h_milliseconds, state.moe_download_started,
            state.moe_completed);
        status != cudaSuccess) {
        return cuda_error(status, "measure DeepSeek MoE output download");
    }
    if (shared_phase_timing_valid) {
        const auto measure_shared_phase = [&](float& output,
                                              cudaEvent_t begin,
                                              cudaEvent_t end,
                                              const char* operation) {
            if (auto status = cudaEventElapsedTime(&output, begin, end);
                status != cudaSuccess) {
                result.errors.emplace_back(
                    std::string(operation) + ": " +
                    cudaGetErrorString(status));
                return false;
            }
            return true;
        };
        if (!measure_shared_phase(
                shared_input_quantize_milliseconds,
                state.moe_hidden_uploaded,
                state.moe_shared_input_finished,
                "measure DeepSeek shared input quantization") ||
            !measure_shared_phase(
                shared_gate_up_milliseconds,
                state.moe_shared_input_finished,
                state.moe_shared_gate_up_finished,
                "measure DeepSeek shared gate/up") ||
            !measure_shared_phase(
                shared_activation_quantize_milliseconds,
                state.moe_shared_gate_up_finished,
                state.moe_shared_activation_finished,
                "measure DeepSeek shared activation quantization") ||
            !measure_shared_phase(
                shared_down_milliseconds,
                state.moe_shared_activation_finished,
                state.moe_shared_finished,
                "measure DeepSeek shared down")) {
            return result;
        }
    }
    const auto to_nanoseconds = [](float milliseconds) {
        return static_cast<std::uint64_t>(std::llround(
            static_cast<double>(milliseconds) * 1.0e6));
    };
    const auto h2d_nanoseconds = to_nanoseconds(h2d_milliseconds);
    const auto kernel_nanoseconds = to_nanoseconds(kernel_milliseconds);
    const auto d2h_nanoseconds = to_nanoseconds(d2h_milliseconds);
    const auto total_nanoseconds =
        h2d_nanoseconds + kernel_nanoseconds + d2h_nanoseconds;
    const auto shared_input_quantize_nanoseconds =
        to_nanoseconds(shared_input_quantize_milliseconds);
    const auto shared_gate_up_nanoseconds =
        to_nanoseconds(shared_gate_up_milliseconds);
    const auto shared_activation_quantize_nanoseconds =
        to_nanoseconds(shared_activation_quantize_milliseconds);
    const auto shared_down_nanoseconds =
        to_nanoseconds(shared_down_milliseconds);
    {
        std::scoped_lock lock(impl_->mutex);
        auto& device_stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        device_stats.activation_d2h_bytes += downloaded_bytes;
        record_synchronization(device_stats,
                               SynchronizationSubsystem::Moe, 1U,
                               wait_nanoseconds);
        device_stats.activation_h2d_nanoseconds += h2d_nanoseconds;
        device_stats.kernel_nanoseconds += kernel_nanoseconds;
        device_stats.activation_d2h_nanoseconds += d2h_nanoseconds;
        device_stats.deepseek_moe_d2h_transfers +=
            downloaded_bytes == 0U ? 1U : 2U;
        device_stats.deepseek_moe_d2h_bytes +=
            downloaded_bytes + sizeof(unsigned int);
        device_stats.deepseek_moe_h2d_nanoseconds += h2d_nanoseconds;
        device_stats.deepseek_moe_kernel_nanoseconds += kernel_nanoseconds;
        device_stats.deepseek_moe_input_quantize_nanoseconds +=
            shared_input_quantize_nanoseconds;
        device_stats.deepseek_moe_shared_gate_up_nanoseconds +=
            shared_gate_up_nanoseconds;
        device_stats.deepseek_moe_shared_activation_quantize_nanoseconds +=
            shared_activation_quantize_nanoseconds;
        device_stats.deepseek_moe_shared_down_nanoseconds +=
            shared_down_nanoseconds;
        device_stats.deepseek_moe_d2h_nanoseconds += d2h_nanoseconds;
        device_stats.deepseek_moe_nanoseconds += total_nanoseconds;
    }
    state.moe_shared_phase_timing_valid = false;
    if (*host_error != 0U) {
        result.errors.emplace_back(
            "MoE projection produced a non-finite activation");
        return result;
    }
    if (host_callback_failed) {
        if (first_upstream_failure != 0U) {
            result.errors.emplace_back(
                "DeepSeek CPU-MoE callback rejected upstream status " +
                std::to_string(first_upstream_failure));
        } else {
            result.errors.emplace_back("DeepSeek CPU-MoE callback failed");
        }
        return result;
    }
    static_cast<void>(output_to_mhc);
    if (!routed_output.empty()) {
        std::memcpy(routed_output.data(), host_bytes,
                    static_cast<std::size_t>(routed_bytes));
    }
    if (!shared_output.empty()) {
        std::memcpy(shared_output.data(),
                    host_bytes + static_cast<std::ptrdiff_t>(routed_bytes),
                    static_cast<std::size_t>(shared_bytes));
    }
    return result;
}

ValidationResult CudaBackend::finish_deepseek_moe_chain(int device) {
    ValidationResult result;
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) {
        result.errors.emplace_back(
            "DeepSeek MoE chain finish targets an uninitialized CUDA device");
        return result;
    }
    auto& state = found->second;
    const auto clear_shared_phase_timing = [&]() noexcept {
        state.moe_shared_phase_timing_valid = false;
    };
    if (!state.moe_in_flight && state.moe_host_callback_count == 0U) {
        clear_shared_phase_timing();
        result.errors.emplace_back(
            "DeepSeek MoE chain finish has no matching in-flight command");
        return result;
    }
    if (state.moe_host_staging == nullptr || state.moe_error == nullptr) {
        clear_shared_phase_timing();
        result.errors.emplace_back(
            "DeepSeek MoE chain finish has no fixed status staging");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        clear_shared_phase_timing();
        return cuda_error(status, "select CUDA device for DeepSeek MoE chain finish");
    }
    auto* host_status = reinterpret_cast<unsigned int*>(state.moe_host_staging);
    if (auto status = cudaMemcpyAsync(
            host_status, state.moe_error, sizeof(*host_status),
            cudaMemcpyDeviceToHost, state.stream); status != cudaSuccess) {
        clear_shared_phase_timing();
        return cuda_error(status, "stage DeepSeek MoE chain status");
    }
    if (auto status = cudaEventRecord(state.moe_completed, state.stream);
        status != cudaSuccess) {
        clear_shared_phase_timing();
        return cuda_error(status, "record DeepSeek MoE chain finish");
    }
    if (auto status = cudaEventSynchronize(state.moe_completed);
        status != cudaSuccess) {
        clear_shared_phase_timing();
        return cuda_error(status, "synchronize DeepSeek MoE chain finish");
    }
    // The attention host node belongs to this device's stream.  A paired
    // rank-local finish must inspect and clear only this device's commands;
    // clearing the other rank here would release a callback before its stream
    // has drained.  Cross-device collection keeps its existing paired path.
    for (std::uint32_t index = 0U;
         index < state.dsv4_attention_prepare_host_command_count; ++index) {
        if (state.dsv4_attention_prepare_host_commands[index].failed) {
            result.errors.emplace_back(
                "DeepSeek attention preparation host callback failed");
        }
        state.dsv4_attention_prepare_host_commands[index] = {};
    }
    state.dsv4_attention_prepare_host_command_count = 0U;
    state.dsv4_deferred_attention_command_count = 0U;
    std::array<std::uint64_t, 4U> shared_phase_nanoseconds{};
    bool shared_phase_timing_ok = true;
    if (state.moe_shared_phase_timing_valid) {
        const auto measure_shared_phase = [&](std::size_t index,
                                              cudaEvent_t begin,
                                              cudaEvent_t end,
                                              const char* operation) {
            float milliseconds = 0.0F;
            if (auto status = cudaEventElapsedTime(&milliseconds, begin, end);
                status != cudaSuccess) {
                result.errors.emplace_back(
                    std::string(operation) + ": " +
                    cudaGetErrorString(status));
                shared_phase_timing_ok = false;
                return;
            }
            shared_phase_nanoseconds[index] = static_cast<std::uint64_t>(
                std::llround(static_cast<double>(milliseconds) * 1.0e6));
        };
        measure_shared_phase(
            0U, state.moe_hidden_uploaded,
            state.moe_shared_input_finished,
            "measure DeepSeek shared input quantization");
        measure_shared_phase(
            1U, state.moe_shared_input_finished,
            state.moe_shared_gate_up_finished,
            "measure DeepSeek shared gate/up");
        measure_shared_phase(
            2U, state.moe_shared_gate_up_finished,
            state.moe_shared_activation_finished,
            "measure DeepSeek shared activation quantization");
        measure_shared_phase(
            3U, state.moe_shared_activation_finished,
            state.moe_shared_finished,
            "measure DeepSeek shared down");
    }
    if (shared_phase_timing_ok && state.moe_shared_phase_timing_valid) {
        std::scoped_lock lock(impl_->mutex);
        auto& device_stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        device_stats.deepseek_moe_input_quantize_nanoseconds +=
            shared_phase_nanoseconds[0U];
        device_stats.deepseek_moe_shared_gate_up_nanoseconds +=
            shared_phase_nanoseconds[1U];
        device_stats.deepseek_moe_shared_activation_quantize_nanoseconds +=
            shared_phase_nanoseconds[2U];
        device_stats.deepseek_moe_shared_down_nanoseconds +=
            shared_phase_nanoseconds[3U];
    }
    clear_shared_phase_timing();
    bool callback_failed = false;
    for (std::uint32_t index = 0U;
         index < state.moe_host_callback_count; ++index) {
        callback_failed = callback_failed ||
            state.moe_host_callbacks[index].failed;
    }
    if (callback_failed) {
        result.errors.emplace_back("DeepSeek host-MoE callback failed");
    }
    if (*host_status != 0U) {
        result.errors.emplace_back("DeepSeek host-MoE device status failed");
    }
    {
        std::scoped_lock lock(impl_->mutex);
        auto& device_stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        ++device_stats.deepseek_moe_d2h_transfers;
        device_stats.deepseek_moe_d2h_bytes += sizeof(*host_status);
    }
    state.moe_in_flight = false;
    state.moe_weights.clear();
    state.moe_host_join = false;
    state.moe_output_to_mhc = false;
    state.moe_host_callback = {};
    for (std::uint32_t index = 0U;
         index < state.moe_host_callback_count; ++index) {
        state.moe_host_callbacks[index] = {};
    }
    state.moe_host_callback_count = 0U;
    return result;
}

ValidationResult CudaBackend::collect_moe(
    int device, std::span<float> routed_output,
    std::span<float> shared_output) {
    return collect_deepseek_moe(device, routed_output, shared_output);
}

ValidationResult CudaBackend::synchronize(int device) {
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) return {{"cannot synchronize an uninitialized CUDA device"}};
    if (found->second.moe_in_flight ||
        found->second.moe_host_callback_count != 0U) {
        return {{"use collect_deepseek_moe for an in-flight DeepSeek MoE command"}};
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for synchronization");
    }
    const auto started = std::chrono::steady_clock::now();
    const auto status = cudaStreamSynchronize(found->second.stream);
    const auto elapsed = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count());
    if (status == cudaSuccess) {
        std::scoped_lock lock(impl_->mutex);
        auto& device_stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        record_synchronization(device_stats,
                               SynchronizationSubsystem::Other, 1U,
                               elapsed);
    }
    return cuda_error(status, "synchronize CUDA device");
}

CudaBackendStats CudaBackend::stats() const noexcept {
    std::scoped_lock lock(impl_->mutex);
    auto result = impl_->stats;
    for (const auto& device : result.devices) {
        result.weight_upload_bytes += device.weight_upload_bytes;
        result.activation_h2d_bytes += device.activation_h2d_bytes;
        result.activation_d2h_bytes += device.activation_d2h_bytes;
        result.matmul_calls += device.matmul_calls;
        result.weight_allocation_calls += device.weight_allocation_calls;
        result.weight_allocation_bytes += device.weight_allocation_bytes;
        result.workspace_allocation_calls += device.workspace_allocation_calls;
        result.workspace_allocation_bytes += device.workspace_allocation_bytes;
        result.synchronization_calls += device.synchronization_calls;
        result.weight_synchronization.calls +=
            device.weight_synchronization.calls;
        result.attention_synchronization.calls +=
            device.attention_synchronization.calls;
        result.projection_synchronization.calls +=
            device.projection_synchronization.calls;
        result.mhc_synchronization.calls +=
            device.mhc_synchronization.calls;
        result.moe_synchronization.calls +=
            device.moe_synchronization.calls;
        result.other_synchronization.calls +=
            device.other_synchronization.calls;
        if (device.synchronization_nanoseconds >
            result.synchronization_nanoseconds) {
            result.synchronization_nanoseconds =
                device.synchronization_nanoseconds;
            result.weight_synchronization.nanoseconds =
                device.weight_synchronization.nanoseconds;
            result.attention_synchronization.nanoseconds =
                device.attention_synchronization.nanoseconds;
            result.projection_synchronization.nanoseconds =
                device.projection_synchronization.nanoseconds;
            result.mhc_synchronization.nanoseconds =
                device.mhc_synchronization.nanoseconds;
            result.moe_synchronization.nanoseconds =
                device.moe_synchronization.nanoseconds;
            result.other_synchronization.nanoseconds =
                device.other_synchronization.nanoseconds;
        }
        result.upload_wait_nanoseconds = std::max(
            result.upload_wait_nanoseconds, device.upload_wait_nanoseconds);
        result.weight_allocation_nanoseconds = std::max(
            result.weight_allocation_nanoseconds,
            device.weight_allocation_nanoseconds);
        result.weight_copy_nanoseconds = std::max(
            result.weight_copy_nanoseconds, device.weight_copy_nanoseconds);
        result.activation_h2d_nanoseconds = std::max(
            result.activation_h2d_nanoseconds, device.activation_h2d_nanoseconds);
        result.kernel_nanoseconds = std::max(
            result.kernel_nanoseconds, device.kernel_nanoseconds);
        result.activation_d2h_nanoseconds = std::max(
            result.activation_d2h_nanoseconds, device.activation_d2h_nanoseconds);
        result.deepseek_moe_calls += device.deepseek_moe_calls;
        result.deepseek_moe_kernel_launches += device.deepseek_moe_kernel_launches;
        result.deepseek_moe_h2d_transfers += device.deepseek_moe_h2d_transfers;
        result.deepseek_moe_d2h_transfers += device.deepseek_moe_d2h_transfers;
        result.deepseek_moe_h2d_bytes += device.deepseek_moe_h2d_bytes;
        result.deepseek_moe_d2h_bytes += device.deepseek_moe_d2h_bytes;
        result.deepseek_moe_h2d_nanoseconds = std::max(
            result.deepseek_moe_h2d_nanoseconds,
            device.deepseek_moe_h2d_nanoseconds);
        result.deepseek_moe_kernel_nanoseconds = std::max(
            result.deepseek_moe_kernel_nanoseconds,
            device.deepseek_moe_kernel_nanoseconds);
        result.deepseek_moe_d2h_nanoseconds = std::max(
            result.deepseek_moe_d2h_nanoseconds,
            device.deepseek_moe_d2h_nanoseconds);
        result.deepseek_moe_nanoseconds = std::max(
            result.deepseek_moe_nanoseconds, device.deepseek_moe_nanoseconds);
        result.flash_attention_calls += device.flash_attention_calls;
        result.flash_attention_kernel_launches +=
            device.flash_attention_kernel_launches;
        result.flash_attention_h2d_transfers +=
            device.flash_attention_h2d_transfers;
        result.flash_attention_d2h_transfers +=
            device.flash_attention_d2h_transfers;
        result.flash_attention_h2d_bytes += device.flash_attention_h2d_bytes;
        result.flash_attention_d2h_bytes += device.flash_attention_d2h_bytes;
        result.flash_attention_useful_staging_bytes +=
            device.flash_attention_useful_staging_bytes;
        result.flash_attention_wasted_staging_bytes +=
            device.flash_attention_wasted_staging_bytes;
        result.flash_attention_h2d_nanoseconds = std::max(
            result.flash_attention_h2d_nanoseconds,
            device.flash_attention_h2d_nanoseconds);
        result.flash_attention_kernel_nanoseconds = std::max(
            result.flash_attention_kernel_nanoseconds,
            device.flash_attention_kernel_nanoseconds);
        result.flash_attention_d2h_nanoseconds = std::max(
            result.flash_attention_d2h_nanoseconds,
            device.flash_attention_d2h_nanoseconds);
        result.flash_attention_nanoseconds = std::max(
            result.flash_attention_nanoseconds,
            device.flash_attention_nanoseconds);
        result.dsv4_paged_attention_calls +=
            device.dsv4_paged_attention_calls;
        result.dsv4_paged_attention_kernel_launches +=
            device.dsv4_paged_attention_kernel_launches;
        result.dsv4_paged_attention_h2d_bytes +=
            device.dsv4_paged_attention_h2d_bytes;
        result.dsv4_paged_attention_d2h_bytes +=
            device.dsv4_paged_attention_d2h_bytes;
        result.dsv4_paged_attention_page_bytes +=
            device.dsv4_paged_attention_page_bytes;
        result.dsv4_paged_attention_h2d_nanoseconds = std::max(
            result.dsv4_paged_attention_h2d_nanoseconds,
            device.dsv4_paged_attention_h2d_nanoseconds);
        result.dsv4_paged_attention_kernel_nanoseconds = std::max(
            result.dsv4_paged_attention_kernel_nanoseconds,
            device.dsv4_paged_attention_kernel_nanoseconds);
        result.dsv4_paged_attention_d2h_nanoseconds = std::max(
            result.dsv4_paged_attention_d2h_nanoseconds,
            device.dsv4_paged_attention_d2h_nanoseconds);
        result.dsv4_paged_attention_nanoseconds = std::max(
            result.dsv4_paged_attention_nanoseconds,
            device.dsv4_paged_attention_nanoseconds);
        result.dsv4_paged_attention_host_remainder_nanoseconds +=
            device.dsv4_paged_attention_host_remainder_nanoseconds;
        result.dsv4_paged_attention_stream_sync_nanoseconds +=
            device.dsv4_paged_attention_stream_sync_nanoseconds;
        result.dsv4_mhc_calls += device.dsv4_mhc_calls;
        result.dsv4_mhc_standalone_calls +=
            device.dsv4_mhc_standalone_calls;
        result.dsv4_mhc_transition_calls +=
            device.dsv4_mhc_transition_calls;
        result.dsv4_mhc_final_calls += device.dsv4_mhc_final_calls;
        result.dsv4_mhc_kernel_launches +=
            device.dsv4_mhc_kernel_launches;
        result.dsv4_mhc_resident_weight_bytes +=
            device.dsv4_mhc_resident_weight_bytes;
        result.dsv4_mhc_h2d_bytes += device.dsv4_mhc_h2d_bytes;
        result.dsv4_mhc_d2h_bytes += device.dsv4_mhc_d2h_bytes;
        result.dsv4_mhc_h2d_nanoseconds = std::max(
            result.dsv4_mhc_h2d_nanoseconds,
            device.dsv4_mhc_h2d_nanoseconds);
        result.dsv4_mhc_kernel_nanoseconds = std::max(
            result.dsv4_mhc_kernel_nanoseconds,
            device.dsv4_mhc_kernel_nanoseconds);
        result.dsv4_mhc_d2h_nanoseconds = std::max(
            result.dsv4_mhc_d2h_nanoseconds,
            device.dsv4_mhc_d2h_nanoseconds);
        result.dsv4_mhc_nanoseconds = std::max(
            result.dsv4_mhc_nanoseconds,
            device.dsv4_mhc_nanoseconds);
        result.dsv4_mhc_device_nanoseconds = std::max(
            result.dsv4_mhc_device_nanoseconds,
            device.dsv4_mhc_device_nanoseconds);
        result.dsv4_mhc_host_nanoseconds = std::max(
            result.dsv4_mhc_host_nanoseconds,
            device.dsv4_mhc_host_nanoseconds);
        result.dsv4_mhc_timing_clamped_samples +=
            device.dsv4_mhc_timing_clamped_samples;
        result.lightning_index_calls += device.lightning_index_calls;
        result.lightning_index_kernel_launches +=
            device.lightning_index_kernel_launches;
        result.lightning_index_candidates +=
            device.lightning_index_candidates;
        result.lightning_index_selected += device.lightning_index_selected;
        result.lightning_index_h2d_transfers +=
            device.lightning_index_h2d_transfers;
        result.lightning_index_d2h_transfers +=
            device.lightning_index_d2h_transfers;
        result.lightning_index_h2d_bytes +=
            device.lightning_index_h2d_bytes;
        result.lightning_index_d2h_bytes +=
            device.lightning_index_d2h_bytes;
        result.lightning_index_useful_selection_bytes +=
            device.lightning_index_useful_selection_bytes;
        result.lightning_index_h2d_nanoseconds = std::max(
            result.lightning_index_h2d_nanoseconds,
            device.lightning_index_h2d_nanoseconds);
        result.lightning_index_kernel_nanoseconds = std::max(
            result.lightning_index_kernel_nanoseconds,
            device.lightning_index_kernel_nanoseconds);
        result.lightning_index_d2h_nanoseconds = std::max(
            result.lightning_index_d2h_nanoseconds,
            device.lightning_index_d2h_nanoseconds);
        result.lightning_index_nanoseconds = std::max(
            result.lightning_index_nanoseconds,
            device.lightning_index_nanoseconds);
    }
    return result;
}

}  // namespace strata
