#include "strata/cuda_backend.hpp"
#include "strata/numerics.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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
#include <type_traits>
#include <unordered_map>

namespace strata {

namespace {

ValidationResult cuda_error(cudaError_t status, const char* operation) {
    ValidationResult result;
    if (status != cudaSuccess) {
        result.errors.emplace_back(std::string(operation) + ": " + cudaGetErrorString(status));
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

__device__ float fp4_e2m1_value(unsigned int encoded) {
    switch (encoded & 0x0FU) {
        case 0x0U: return 0.0F;
        case 0x1U: return 0.5F;
        case 0x2U: return 1.0F;
        case 0x3U: return 1.5F;
        case 0x4U: return 2.0F;
        case 0x5U: return 3.0F;
        case 0x6U: return 4.0F;
        case 0x7U: return 6.0F;
        case 0x8U: return 0.0F;
        case 0x9U: return -0.5F;
        case 0xAU: return -1.0F;
        case 0xBU: return -1.5F;
        case 0xCU: return -2.0F;
        case 0xDU: return -3.0F;
        case 0xEU: return -4.0F;
        case 0xFU: return -6.0F;
        default: return 0.0F;
    }
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

__global__ void plain_matmul_kernel(float* output, const float* input,
                                    const void* weights, int dtype,
                                    std::uint32_t batch, std::uint64_t columns,
                                    std::uint64_t rows, std::uint32_t groups,
                                    std::uint64_t rows_per_group) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t batch_row = blockIdx.y;
    if (output_row >= rows || batch_row >= batch) return;
    float sum = 0.0F;
    const std::uint64_t weight_base = output_row * columns;
    const std::uint64_t input_row = groups == 0U
                                        ? batch_row
                                        : output_row / rows_per_group;
    const std::uint64_t input_base = input_row * columns;
    for (std::uint64_t column = threadIdx.x; column < columns; column += blockDim.x) {
        sum += plain_value(weights, dtype, weight_base + column) * input[input_base + column];
    }
    sum = reduce_block(sum);
    if (threadIdx.x == 0) {
        const std::uint64_t output_index = groups == 0U
                                               ? static_cast<std::uint64_t>(batch_row) * rows +
                                                     output_row
                                               : output_row;
        output[output_index] = sum;
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
                                        : output_row / rows_per_group;
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
        const std::uint64_t output_index = groups == 0U
                                               ? static_cast<std::uint64_t>(batch_row) * rows +
                                                     output_row
                                               : output_row;
        output[output_index] = sum;
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
                                        : output_row / rows_per_group;
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
        const std::uint64_t output_index = groups == 0U
                                               ? static_cast<std::uint64_t>(batch_row) * rows +
                                                     output_row
                                               : output_row;
        output[output_index] = sum;
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
                                        : output_row / rows_per_group;
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
        const std::uint64_t output_index = groups == 0U
                                               ? static_cast<std::uint64_t>(batch_row) * rows +
                                                     output_row
                                               : output_row;
        output[output_index] = sum;
    }
}

struct DeepSeekFp4Work {
    const unsigned char* w1_weights{};
    const unsigned char* w1_scales{};
    const unsigned char* w3_weights{};
    const unsigned char* w3_scales{};
    const unsigned char* w2_weights{};
    const unsigned char* w2_scales{};
    float coefficient{};
    std::uint32_t input_row{};
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

__global__ void deepseek_fp4_gate_up_kernel(
    float* activations, const float* hidden, const DeepSeekFp4Work* work,
    std::uint32_t work_count,
    std::uint64_t columns, std::uint64_t intermediate,
    std::uint64_t packed_columns, std::uint64_t scale_columns,
    float swiglu_limit, const float* bf16_silu, unsigned int* error_flag) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t assignment = blockIdx.y;
    if (output_row >= intermediate || assignment >= work_count) return;

    const auto assignment_work = work[assignment];
    const auto* w1 = assignment_work.w1_weights;
    const auto* w1_scales = assignment_work.w1_scales;
    const auto* w3 = assignment_work.w3_weights;
    const auto* w3_scales = assignment_work.w3_scales;
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
            const float input = hidden[
                static_cast<std::uint64_t>(assignment_work.input_row) * columns +
                column];
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
            activations[static_cast<std::uint64_t>(assignment) * intermediate +
                        output_row] = __uint_as_float(0x7FC0'0000U);
            return;
        }
        const float limited_gate = fminf(rounded_gate, swiglu_limit);
        const float limited_up = fmaxf(-swiglu_limit,
                                       fminf(rounded_up, swiglu_limit));
        const auto gate_bits = static_cast<std::uint16_t>(
            __float_as_uint(limited_gate) >> 16U);
        float activated = bf16_silu[gate_bits] * limited_up;
        activated *= assignment_work.coefficient;
        activations[static_cast<std::uint64_t>(assignment) * intermediate +
                    output_row] = bf16_round(activated);
    }
}

__global__ void deepseek_fp4_down_kernel(
    float* output, const float* activations, const DeepSeekFp4Work* work,
    std::uint32_t work_count,
    std::uint64_t columns, std::uint64_t rows,
    std::uint64_t packed_columns, std::uint64_t scale_columns) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t assignment = blockIdx.y;
    if (output_row >= rows || assignment >= work_count) return;
    const auto assignment_work = work[assignment];
    const auto* weights = assignment_work.w2_weights;
    const auto* scales = assignment_work.w2_scales;
    const std::uint64_t packed_base = output_row * packed_columns;
    const std::uint64_t scale_base = output_row * scale_columns;
    const std::uint64_t input_base =
        static_cast<std::uint64_t>(assignment) * columns;
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
        output[static_cast<std::uint64_t>(assignment) * rows + output_row] =
            bf16_round(sum);
    }
}

__global__ void deepseek_fp8_gate_up_kernel(
    float* activation, const float* hidden,
    const unsigned char* w1, const unsigned char* w1_scales,
    const unsigned char* w3, const unsigned char* w3_scales,
    std::uint64_t columns, std::uint64_t intermediate,
    std::uint64_t scale_columns, float swiglu_limit,
    const float* bf16_silu, unsigned int* error_flag) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t batch_row = blockIdx.y;
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
            const float input = hidden[
                static_cast<std::uint64_t>(batch_row) * columns + column];
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
            activation[static_cast<std::uint64_t>(batch_row) * intermediate +
                       output_row] = __uint_as_float(0x7FC0'0000U);
            return;
        }
        const float limited_gate = fminf(rounded_gate, swiglu_limit);
        const float limited_up = fmaxf(-swiglu_limit,
                                       fminf(rounded_up, swiglu_limit));
        const auto gate_bits = static_cast<std::uint16_t>(
            __float_as_uint(limited_gate) >> 16U);
        activation[static_cast<std::uint64_t>(batch_row) * intermediate +
                   output_row] = bf16_round(
            bf16_silu[gate_bits] * limited_up);
    }
}

__global__ void deepseek_fp8_down_kernel(
    float* output, const float* activation,
    const unsigned char* weights, const unsigned char* scales,
    std::uint64_t columns, std::uint64_t rows,
    std::uint64_t scale_columns) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t batch_row = blockIdx.y;
    if (output_row >= rows) return;
    const std::uint64_t weight_base = output_row * columns;
    const std::uint64_t scale_row = (output_row / 128U) * scale_columns;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint64_t groups = (columns + 31U) / 32U;
    const std::uint64_t input_base =
        static_cast<std::uint64_t>(batch_row) * columns;
    float sum = 0.0F;
    for (std::uint64_t group = warp; group < groups; group += 8U) {
        const std::uint64_t column = group * 32U + lane;
        float scale = lane == 0U
                          ? fp8_e8m0_scale_bits(
                                scales[scale_row + (group * 32U) / 128U])
                          : 0.0F;
        scale = __shfl_sync(0xFFFF'FFFFU, scale, 0);
        if (column < columns) {
            sum += activation[input_base + column] *
                   fp8_e4m3_value(weights[weight_base + column]) * scale;
        }
    }
    sum = reduce_block(sum);
    if (threadIdx.x == 0U) {
        output[static_cast<std::uint64_t>(batch_row) * rows + output_row] =
            bf16_round(sum);
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
        const auto* key = keys +
            (static_cast<std::uint64_t>(row) * key_value_heads + kv_head) *
                query_key_dim;
        const float score = __fmul_rn(static_cast<float>(
            flash_attention_sequential_dot(query, key, query_key_dim)), scale);
        scores[row] = score;
        if (!isfinite(score)) atomicExch(error_flag, 1U);
    }
    __syncthreads();

    if (threadIdx.x == 0U) {
        maximum = sinks == nullptr ? -INFINITY : sinks[head];
        for (std::uint32_t row = 0U; row < visible_rows; ++row) {
            maximum = fmaxf(maximum, scores[row]);
        }
        denominator = sinks == nullptr
            ? 0.0
            : exp(static_cast<double>(__fsub_rn(sinks[head], maximum)));
        for (std::uint32_t row = 0U; row < visible_rows; ++row) {
            denominator = __dadd_rn(denominator, exp(static_cast<double>(
                __fsub_rn(scores[row], maximum))));
        }
        if (!isfinite(denominator) || denominator <= 0.0) {
            atomicExch(error_flag, 2U);
        }
        for (std::uint32_t row = 0U; row < visible_rows; ++row) {
            scores[row] = static_cast<float>(exp(static_cast<double>(
                __fsub_rn(scores[row], maximum))) / denominator);
        }
    }
    __syncthreads();

    for (std::uint32_t row = 0U; row < visible_rows; ++row) {
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

}  // namespace

struct CudaWeight::Impl {
    void* weights{};
    void* scales{};
    CudaWeightDescriptor descriptor;
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

struct CudaBackend::Impl {
    struct DeviceState {
        cudaStream_t stream{};
        cudaEvent_t activation_start{};
        cudaEvent_t activation_uploaded{};
        cudaEvent_t kernel_finished{};
        cudaEvent_t activation_downloaded{};
        cudaEvent_t moe_start{};
        cudaEvent_t moe_hidden_uploaded{};
        cudaEvent_t moe_kernel_finished{};
        cudaEvent_t moe_download_started{};
        cudaEvent_t moe_completed{};
        float* input{};
        float* output{};
        std::uint64_t input_bytes{};
        std::uint64_t output_bytes{};
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
        float* moe_hidden{};
        float* moe_activations{};
        float* moe_output{};
        DeepSeekFp4Work* moe_work{};
        float* moe_bf16_silu{};
        unsigned int* moe_error{};
        void* moe_host_staging{};
        std::uint64_t moe_hidden_bytes{};
        std::uint64_t moe_activation_bytes{};
        std::uint64_t moe_output_bytes{};
        std::uint64_t moe_work_bytes{};
        std::uint64_t moe_host_staging_bytes{};
        std::uint64_t moe_hidden_columns{};
        std::uint64_t moe_intermediate_columns{};
        std::uint32_t moe_routed_count{};
        std::uint32_t moe_shared_rows{};
        std::uint64_t moe_kernel_launches{};
        std::vector<DeepSeekFp4Work> moe_host_work;
        std::vector<std::shared_ptr<CudaWeight::Impl>> moe_weights;
        std::vector<std::shared_ptr<CudaWeight::Impl>> quarantined_weights;
        std::shared_ptr<WeightArena> weight_arena;
        bool moe_has_shared{};
        bool moe_in_flight{};
        bool moe_poisoned{};
        bool flash_attention_supported{};
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
            if (state.attention_host_upload != nullptr) {
                static_cast<void>(cudaFreeHost(state.attention_host_upload));
            }
            if (state.attention_host_download != nullptr) {
                static_cast<void>(cudaFreeHost(state.attention_host_download));
            }
            if (state.moe_hidden != nullptr) static_cast<void>(cudaFree(state.moe_hidden));
            if (state.moe_activations != nullptr) {
                static_cast<void>(cudaFree(state.moe_activations));
            }
            if (state.moe_output != nullptr) static_cast<void>(cudaFree(state.moe_output));
            if (state.moe_work != nullptr) static_cast<void>(cudaFree(state.moe_work));
            if (state.moe_bf16_silu != nullptr) {
                static_cast<void>(cudaFree(state.moe_bf16_silu));
            }
            if (state.moe_error != nullptr) static_cast<void>(cudaFree(state.moe_error));
            if (state.moe_host_staging != nullptr) {
                static_cast<void>(cudaFreeHost(state.moe_host_staging));
            }
            if (state.activation_start != nullptr) {
                static_cast<void>(cudaEventDestroy(state.activation_start));
                static_cast<void>(cudaEventDestroy(state.activation_uploaded));
                static_cast<void>(cudaEventDestroy(state.kernel_finished));
                static_cast<void>(cudaEventDestroy(state.activation_downloaded));
            }
            if (state.moe_start != nullptr) {
                static_cast<void>(cudaEventDestroy(state.moe_start));
                static_cast<void>(cudaEventDestroy(state.moe_hidden_uploaded));
                static_cast<void>(cudaEventDestroy(state.moe_kernel_finished));
                static_cast<void>(cudaEventDestroy(state.moe_download_started));
                static_cast<void>(cudaEventDestroy(state.moe_completed));
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
        if (auto status = cudaStreamCreateWithFlags(&state.stream, cudaStreamNonBlocking);
            status != cudaSuccess) {
            return cuda_error(status, "create CUDA stream");
        }
        if (detailed_timing) {
            for (auto* event : {&state.activation_start, &state.activation_uploaded,
                                &state.kernel_finished, &state.activation_downloaded}) {
                if (auto status = cudaEventCreate(event); status != cudaSuccess) {
                    return cuda_error(status, "create CUDA timing event");
                }
            }
        }
        for (auto* event : {&state.moe_start, &state.moe_hidden_uploaded,
                            &state.moe_kernel_finished, &state.moe_download_started,
                            &state.moe_completed}) {
            if (auto status = cudaEventCreate(event); status != cudaSuccess) {
                return cuda_error(status, "create DeepSeek MoE event");
            }
        }
        impl_->devices.emplace(device, state);
        CudaBackendStats::Device device_stats;
        device_stats.device = device;
        impl_->stats.devices.push_back(device_stats);
    }
    return result;
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
                                     CudaWeight& output) {
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
    // This stream is nonblocking with respect to the legacy default stream.
    // Keep uploads on the execution stream and finish them before the caller's
    // host payload is released or the cache publishes the weight.
    const auto upload_error = [&state, &target](cudaError_t status,
                                                const char* operation) {
        if (cudaStreamSynchronize(state.stream) != cudaSuccess) {
            state.quarantined_weights.push_back(std::move(target));
        }
        return cuda_error(status, operation);
    };
    if (auto status = cudaMemcpyAsync(target->weights, weights.data(), weights.size(),
                                      cudaMemcpyHostToDevice, state.stream);
        status != cudaSuccess) {
        return upload_error(status, "upload CUDA weights");
    }
    if (expected_scales != 0U) {
        if (state.weight_arena == nullptr) {
            if (auto status = cudaMalloc(
                    &target->scales, static_cast<std::size_t>(expected_scales));
                status != cudaSuccess) {
                return cuda_error(status, "allocate CUDA scales");
            }
            ++allocation_calls;
        }
        if (auto status = cudaMemcpyAsync(target->scales, scales.data(), scales.size(),
                                          cudaMemcpyHostToDevice, state.stream);
            status != cudaSuccess) {
            return upload_error(status, "upload CUDA scales");
        }
    }
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaStreamSynchronize(state.stream); status != cudaSuccess) {
        state.quarantined_weights.push_back(std::move(target));
        return cuda_error(status, "synchronize CUDA weight upload");
    }
    const auto wait_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wait_started).count());
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
        ++device_stats.synchronization_calls;
        device_stats.synchronization_nanoseconds += wait_nanoseconds;
        device_stats.upload_wait_nanoseconds += wait_nanoseconds;
    }
    output.impl_ = std::move(target);
    return result;
}

ValidationResult CudaBackend::matmul(const CudaWeight& weight,
                                     std::span<const float> input,
                                     std::uint32_t rows,
                                     std::span<float> output) {
    return matmul_impl(weight, input, rows, 0U, 0U, output);
}

ValidationResult CudaBackend::matmul_grouped(
    const CudaWeight& weight, std::span<const float> input,
    std::uint32_t groups, std::uint64_t rows_per_group,
    std::span<float> output) {
    return matmul_impl(weight, input, groups, groups, rows_per_group, output);
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
    std::copy(request.queries.begin(), request.queries.end(), host_queries);
    std::copy(request.head_sinks.begin(), request.head_sinks.end(), host_sinks);
    std::copy(request.causal_key_counts.begin(),
              request.causal_key_counts.end(), host_limits);
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
            request.query_rows, request.query_heads, request.key_value_heads,
            request.query_key_dim, request.value_dim,
            static_cast<std::uint32_t>(shape.value.logical_rows), request.scale,
            device_error);
    } else if (request.numerics ==
        FlashAttentionNumerics::f64_dot_f32_score_f32_accum) {
        flash_attention_reference_f32_kernel<<<grid, threads, 0U, state.stream>>>(
            device_output, device_queries, device_keys,
            device_values, state.attention_scores,
            request.head_sinks.empty() ? nullptr : device_sinks,
            request.causal_key_counts.empty()
                ? nullptr : device_limits,
            request.query_rows, request.query_heads, request.key_value_heads,
            request.query_key_dim, request.value_dim,
            static_cast<std::uint32_t>(shape.value.logical_rows), request.scale,
            device_error);
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
        ++stats.synchronization_calls;
        stats.synchronization_nanoseconds += wait_nanoseconds;
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

ValidationResult CudaBackend::matmul_impl(
    const CudaWeight& weight, std::span<const float> input,
    std::uint32_t rows, std::uint32_t groups,
    std::uint64_t rows_per_group, std::span<float> output) {
    ValidationResult result;
    if (!weight.valid()) {
        result.errors.emplace_back("CUDA matmul received an invalid weight");
        return result;
    }
    const auto& descriptor = weight.impl_->descriptor;
    const bool regular_shape = groups == 0U &&
        input.size() == descriptor.columns * rows &&
        output.size() == descriptor.rows * rows;
    const bool grouped_shape = groups != 0U && rows == groups && rows_per_group != 0U &&
        descriptor.rows == static_cast<std::uint64_t>(groups) * rows_per_group &&
        input.size() == descriptor.columns * groups && output.size() == descriptor.rows;
    if (rows == 0U || (!regular_shape && !grouped_shape)) {
        result.errors.emplace_back("CUDA matmul activation shapes are incompatible");
        return result;
    }
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
    std::uint64_t workspace_allocation_calls = 0U;
    std::uint64_t workspace_allocation_bytes = 0U;
    if (input_bytes > state.input_bytes) {
        if (state.input != nullptr) static_cast<void>(cudaFree(state.input));
        if (auto status = cudaMalloc(&state.input, static_cast<std::size_t>(input_bytes));
            status != cudaSuccess) {
            return cuda_error(status, "allocate CUDA input workspace");
        }
        state.input_bytes = input_bytes;
        ++workspace_allocation_calls;
        workspace_allocation_bytes += input_bytes;
    }
    if (output_bytes > state.output_bytes) {
        if (state.output != nullptr) static_cast<void>(cudaFree(state.output));
        if (auto status = cudaMalloc(&state.output, static_cast<std::size_t>(output_bytes));
            status != cudaSuccess) {
            return cuda_error(status, "allocate CUDA output workspace");
        }
        state.output_bytes = output_bytes;
        ++workspace_allocation_calls;
        workspace_allocation_bytes += output_bytes;
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.activation_start, state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "record activation upload start");
        }
    }
    if (auto status = cudaMemcpyAsync(state.input, input.data(), input.size_bytes(),
                                      cudaMemcpyHostToDevice, state.stream);
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
    if (native) {
        const dim3 quantize_grid(
            static_cast<unsigned int>((descriptor.columns + 127U) / 128U), rows, 1U);
        quantize_activation_e4m3_kernel<<<quantize_grid, 128U, 0U, state.stream>>>(
            state.input, descriptor.columns, rows);
    }
    const auto output_batches = groups == 0U ? rows : 1U;
    const dim3 grid(static_cast<unsigned int>(descriptor.rows), output_batches, 1U);
    constexpr unsigned int threads = 256U;
    if (descriptor.encoding == CudaWeightEncoding::Plain) {
        plain_matmul_kernel<<<grid, threads, 0, state.stream>>>(
            state.output, state.input, weight.impl_->weights,
            static_cast<int>(descriptor.dtype), rows, descriptor.columns,
            descriptor.rows, groups, rows_per_group);
    } else if (descriptor.encoding == CudaWeightEncoding::OffsetPackedInt4 ||
               descriptor.encoding == CudaWeightEncoding::OffsetPackedInt8) {
        const auto bits = descriptor.encoding == CudaWeightEncoding::OffsetPackedInt4 ? 4U : 8U;
        packed_matmul_kernel<<<grid, threads, 0, state.stream>>>(
            state.output, state.input, static_cast<const std::uint32_t*>(weight.impl_->weights),
            static_cast<const __nv_bfloat16*>(weight.impl_->scales), bits,
            descriptor.group_size, descriptor.packed_columns,
            descriptor.scale_columns, rows, descriptor.columns, descriptor.rows,
            groups, rows_per_group);
    } else if (descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128) {
        native_fp8_matmul_kernel<<<grid, threads, 0, state.stream>>>(
            state.output, state.input,
            static_cast<const unsigned char*>(weight.impl_->weights),
            static_cast<const unsigned char*>(weight.impl_->scales),
            descriptor.scale_columns, rows, descriptor.columns, descriptor.rows,
            groups, rows_per_group);
    } else {
        native_fp4_matmul_kernel<<<grid, threads, 0, state.stream>>>(
            state.output, state.input,
            static_cast<const unsigned char*>(weight.impl_->weights),
            static_cast<const unsigned char*>(weight.impl_->scales),
            descriptor.packed_columns, descriptor.scale_columns, rows,
            descriptor.columns, descriptor.rows, groups, rows_per_group);
    }
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        return cuda_error(status, "launch CUDA matmul");
    }
    if (impl_->detailed_timing) {
        if (auto status = cudaEventRecord(state.kernel_finished, state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "record CUDA kernel completion");
        }
    }
    if (auto status = cudaMemcpyAsync(output.data(), state.output, output.size_bytes(),
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
    if (auto status = cudaStreamSynchronize(state.stream); status != cudaSuccess) {
        return cuda_error(status, "synchronize CUDA matmul");
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
        ++device_stats.synchronization_calls;
        device_stats.synchronization_nanoseconds += wait_nanoseconds;
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
    if (routed.size() > 6U) {
        return {{"DeepSeek MoE command requires one to six routed experts or a shared expert"}};
    }
    std::array<CudaDeepSeekMoeRow, 6> rows{};
    std::array<CudaDeepSeekMoeGroup, 6> groups{};
    for (std::size_t index = 0U; index < routed.size(); ++index) {
        rows[index].coefficient = routed[index].coefficient;
        groups[index] = {routed[index].w1, routed[index].w3,
                         routed[index].w2,
                         std::span<const CudaDeepSeekMoeRow>(&rows[index], 1U)};
    }
    return enqueue_deepseek_moe_batch(
        device, hidden, 1U,
        std::span<const CudaDeepSeekMoeGroup>(groups).first(routed.size()),
        shared, swiglu_limit);
}

ValidationResult CudaBackend::enqueue_deepseek_moe_batch(
    int device, std::span<const float> hidden, std::uint32_t hidden_rows,
    std::span<const CudaDeepSeekMoeGroup> routed,
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
    if (hidden_rows == 0U || (routed.empty() && shared == nullptr)) {
        result.errors.emplace_back(
            "DeepSeek MoE batch requires hidden rows and routed or shared work");
        return result;
    }
    if (!std::isfinite(swiglu_limit) || swiglu_limit <= 0.0F) {
        result.errors.emplace_back(
            "DeepSeek MoE SwiGLU limit must be finite and positive");
        return result;
    }

    std::uint64_t hidden_columns = 0U;
    std::uint64_t intermediate_columns = 0U;
    auto validate_expert = [&](const CudaWeight* expert_w1,
                               const CudaWeight* expert_w3,
                               const CudaWeight* expert_w2,
                               CudaWeightEncoding encoding) {
        const std::array<const CudaWeight*, 3> weights{
            expert_w1, expert_w3, expert_w2};
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
        const auto& w1 = expert_w1->impl_->descriptor;
        const auto& w3 = expert_w3->impl_->descriptor;
        const auto& w2 = expert_w2->impl_->descriptor;
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
    std::uint64_t routed_count = 0U;
    for (const auto& group : routed) {
        if (group.rows.empty()) {
            result.errors.emplace_back(
                "DeepSeek MoE routed group contains no rows");
            return result;
        }
        if (!validate_expert(group.w1, group.w3, group.w2,
                             CudaWeightEncoding::Fp4E2m1Group32)) {
            return result;
        }
        for (const auto& row : group.rows) {
            if (row.row >= hidden_rows || !std::isfinite(row.coefficient)) {
                result.errors.emplace_back(
                    "DeepSeek MoE routed row or coefficient is invalid");
                return result;
            }
        }
        if (group.rows.size() >
            std::numeric_limits<std::uint64_t>::max() - routed_count) {
            result.errors.emplace_back("DeepSeek MoE routed row count overflows");
            return result;
        }
        routed_count += group.rows.size();
    }
    if (shared != nullptr) {
        if (!std::isfinite(shared->coefficient) ||
            shared->coefficient != 1.0F) {
            result.errors.emplace_back(
                "DeepSeek MoE expert coefficient is invalid");
            return result;
        }
        if (!validate_expert(shared->w1, shared->w3, shared->w2,
                             CudaWeightEncoding::Fp8E4m3Block128)) {
            return result;
        }
    }
    if (hidden_rows > std::numeric_limits<std::uint16_t>::max() ||
        routed_count > std::numeric_limits<std::uint16_t>::max()) {
        result.errors.emplace_back(
            "DeepSeek MoE routed row count exceeds the CUDA grid limit");
        return result;
    }
    if (hidden.empty() ||
        hidden_columns > std::numeric_limits<unsigned int>::max() ||
        intermediate_columns > std::numeric_limits<unsigned int>::max() ||
        hidden.size() != static_cast<std::uint64_t>(hidden_rows) * hidden_columns) {
        result.errors.emplace_back(
            "DeepSeek MoE hidden rows or expert dimensions are incompatible");
        return result;
    }
    if (!std::all_of(hidden.begin(), hidden.end(),
                     [](float value) { return std::isfinite(value); })) {
        result.errors.emplace_back(
            "DeepSeek MoE hidden rows contain a non-finite value");
        return result;
    }

    const auto shared_rows = shared == nullptr ? 0U : hidden_rows;
    const std::uint64_t expert_rows = routed_count + shared_rows;
    const std::uint64_t expert_groups =
        static_cast<std::uint64_t>(routed.size()) + (shared == nullptr ? 0U : 1U);
    std::uint64_t hidden_bytes = 0U;
    std::uint64_t activation_bytes = 0U;
    std::uint64_t output_bytes = 0U;
    std::uint64_t work_bytes = 0U;
    std::uint64_t host_staging_bytes = 0U;
    if (!checked_bytes(hidden_rows, hidden_columns, sizeof(float), hidden_bytes) ||
        !checked_bytes(expert_rows, intermediate_columns, sizeof(float),
                       activation_bytes) ||
        !checked_bytes(expert_rows, hidden_columns, sizeof(float), output_bytes) ||
        !checked_bytes(routed_count, 1U, sizeof(DeepSeekFp4Work), work_bytes) ||
        hidden_bytes > std::numeric_limits<std::size_t>::max() ||
        activation_bytes > std::numeric_limits<std::size_t>::max() ||
        output_bytes > std::numeric_limits<std::size_t>::max() ||
        work_bytes > std::numeric_limits<std::size_t>::max() ||
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
    if (work_bytes > state.moe_work_bytes) {
        if (state.moe_work != nullptr) static_cast<void>(cudaFree(state.moe_work));
        state.moe_work = nullptr;
        state.moe_work_bytes = 0U;
        if (const auto status = cudaMalloc(
                &state.moe_work, static_cast<std::size_t>(work_bytes));
            status != cudaSuccess) {
            return cuda_error(status, "allocate DeepSeek MoE row workspace");
        }
        state.moe_work_bytes = work_bytes;
        ++allocation_calls;
        allocation_bytes += work_bytes;
    }
    if (state.moe_bf16_silu == nullptr) {
        constexpr std::size_t entries = 1U << 16U;
        constexpr std::size_t bytes = entries * sizeof(float);
        static const std::array<float, entries> table = [] {
            std::array<float, entries> values{};
            for (std::size_t index = 0U; index < entries; ++index) {
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

    state.moe_host_work.clear();
    state.moe_host_work.reserve(static_cast<std::size_t>(routed_count));
    for (const auto& group : routed) {
        for (const auto& row : group.rows) {
            state.moe_host_work.push_back({
                static_cast<const unsigned char*>(group.w1->impl_->weights),
                static_cast<const unsigned char*>(group.w1->impl_->scales),
                static_cast<const unsigned char*>(group.w3->impl_->weights),
                static_cast<const unsigned char*>(group.w3->impl_->scales),
                static_cast<const unsigned char*>(group.w2->impl_->weights),
                static_cast<const unsigned char*>(group.w2->impl_->scales),
                row.coefficient, row.row});
        }
    }

    state.moe_weights.clear();
    state.moe_weights.reserve(static_cast<std::size_t>(expert_groups * 3U));
    for (const auto& group : routed) {
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
    state.moe_routed_count = static_cast<std::uint32_t>(routed_count);
    state.moe_shared_rows = shared_rows;
    state.moe_has_shared = shared != nullptr;
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
    if (work_bytes != 0U) {
        if (auto status = cudaMemcpyAsync(
                state.moe_work, state.moe_host_work.data(),
                static_cast<std::size_t>(work_bytes), cudaMemcpyHostToDevice,
                state.stream); status != cudaSuccess) {
            abort_enqueue(status, "upload DeepSeek MoE routed row descriptors");
            return result;
        }
    }
    if (auto status = cudaEventRecord(state.moe_hidden_uploaded, state.stream);
        status != cudaSuccess) {
        abort_enqueue(status, "record DeepSeek MoE hidden upload");
        return result;
    }

    constexpr unsigned int threads = 256U;
    const dim3 hidden_quantize_grid(
        static_cast<unsigned int>((hidden_columns + 127U) / 128U), hidden_rows, 1U);
    quantize_activation_e4m3_kernel<<<hidden_quantize_grid, 128U, 0U,
                                      state.stream>>>(
        state.moe_hidden, hidden_columns, hidden_rows);
    ++state.moe_kernel_launches;
    if (auto status = cudaGetLastError(); status != cudaSuccess) {
        abort_enqueue(status, "launch DeepSeek MoE hidden quantization");
        return result;
    }

    if (!routed.empty()) {
        const auto& w1 = routed.front().w1->impl_->descriptor;
        const auto& w2 = routed.front().w2->impl_->descriptor;
        const dim3 gate_grid(static_cast<unsigned int>(intermediate_columns),
                             state.moe_routed_count, 1U);
        deepseek_fp4_gate_up_kernel<<<gate_grid, threads, 0U, state.stream>>>(
            state.moe_activations, state.moe_hidden, state.moe_work,
            state.moe_routed_count,
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
            state.moe_routed_count, 1U);
        quantize_activation_e4m3_kernel<<<activation_grid, 128U, 0U,
                                          state.stream>>>(
            state.moe_activations, intermediate_columns,
            state.moe_routed_count);
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek routed activation quantization");
            return result;
        }
        const dim3 down_grid(static_cast<unsigned int>(hidden_columns),
                             state.moe_routed_count, 1U);
        deepseek_fp4_down_kernel<<<down_grid, threads, 0U, state.stream>>>(
            state.moe_output, state.moe_activations, state.moe_work,
            state.moe_routed_count,
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
            routed_count * intermediate_columns;
        float* shared_output = state.moe_output +
            routed_count * hidden_columns;
        const dim3 shared_gate_grid(
            static_cast<unsigned int>(intermediate_columns), hidden_rows, 1U);
        deepseek_fp8_gate_up_kernel<<<shared_gate_grid, threads, 0U, state.stream>>>(
            shared_activation, state.moe_hidden,
            static_cast<const unsigned char*>(shared->w1->impl_->weights),
            static_cast<const unsigned char*>(shared->w1->impl_->scales),
            static_cast<const unsigned char*>(shared->w3->impl_->weights),
            static_cast<const unsigned char*>(shared->w3->impl_->scales),
            hidden_columns, intermediate_columns, w1.scale_columns,
            swiglu_limit, state.moe_bf16_silu, state.moe_error);
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek shared FP8 W1/W3 SwiGLU");
            return result;
        }
        const dim3 shared_activation_grid(
            static_cast<unsigned int>((intermediate_columns + 127U) / 128U),
            hidden_rows, 1U);
        quantize_activation_e4m3_kernel<<<shared_activation_grid, 128U, 0U,
                                          state.stream>>>(
            shared_activation, intermediate_columns, hidden_rows);
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek shared activation quantization");
            return result;
        }
        const dim3 shared_down_grid(
            static_cast<unsigned int>(hidden_columns), hidden_rows, 1U);
        deepseek_fp8_down_kernel<<<shared_down_grid, threads, 0U, state.stream>>>(
            shared_output, shared_activation,
            static_cast<const unsigned char*>(shared->w2->impl_->weights),
            static_cast<const unsigned char*>(shared->w2->impl_->scales),
            intermediate_columns, hidden_columns, w2.scale_columns);
        ++state.moe_kernel_launches;
        if (auto status = cudaGetLastError(); status != cudaSuccess) {
            abort_enqueue(status, "launch DeepSeek shared FP8 W2");
            return result;
        }
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
        device_stats.matmul_calls += 3U * expert_rows;
        device_stats.workspace_allocation_calls += allocation_calls;
        device_stats.workspace_allocation_bytes += allocation_bytes;
        ++device_stats.deepseek_moe_calls;
        device_stats.deepseek_moe_kernel_launches += state.moe_kernel_launches;
        device_stats.deepseek_moe_h2d_transfers += work_bytes == 0U ? 1U : 2U;
        device_stats.deepseek_moe_h2d_bytes += hidden_bytes + work_bytes;
    }
    return result;
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
    if (!state.moe_in_flight) {
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
        }
    };
    std::uint64_t routed_elements = 0U;
    std::uint64_t shared_elements = 0U;
    if (!checked_bytes(state.moe_routed_count, state.moe_hidden_columns, 1U,
                       routed_elements) ||
        !checked_bytes(state.moe_shared_rows, state.moe_hidden_columns, 1U,
                       shared_elements) ||
        routed_output.size() != routed_elements ||
        (state.moe_has_shared
             ? shared_output.size() != shared_elements
             : !shared_output.empty())) {
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
    if (auto status = cudaMemcpyAsync(
            host_bytes, state.moe_output,
            static_cast<std::size_t>(downloaded_bytes),
            cudaMemcpyDeviceToHost, state.stream);
        status != cudaSuccess) {
        abort_collect(status, "stage DeepSeek MoE expert outputs");
        return result;
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
        }
        return result;
    }
    state.moe_in_flight = false;
    state.moe_weights.clear();

    float h2d_milliseconds = 0.0F;
    float kernel_milliseconds = 0.0F;
    float d2h_milliseconds = 0.0F;
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
    const auto to_nanoseconds = [](float milliseconds) {
        return static_cast<std::uint64_t>(std::llround(
            static_cast<double>(milliseconds) * 1.0e6));
    };
    const auto h2d_nanoseconds = to_nanoseconds(h2d_milliseconds);
    const auto kernel_nanoseconds = to_nanoseconds(kernel_milliseconds);
    const auto d2h_nanoseconds = to_nanoseconds(d2h_milliseconds);
    const auto total_nanoseconds =
        h2d_nanoseconds + kernel_nanoseconds + d2h_nanoseconds;
    {
        std::scoped_lock lock(impl_->mutex);
        auto& device_stats = *std::find_if(
            impl_->stats.devices.begin(), impl_->stats.devices.end(),
            [device](const auto& value) { return value.device == device; });
        device_stats.activation_d2h_bytes += downloaded_bytes;
        ++device_stats.synchronization_calls;
        device_stats.synchronization_nanoseconds += wait_nanoseconds;
        device_stats.activation_h2d_nanoseconds += h2d_nanoseconds;
        device_stats.kernel_nanoseconds += kernel_nanoseconds;
        device_stats.activation_d2h_nanoseconds += d2h_nanoseconds;
        device_stats.deepseek_moe_d2h_transfers += 2U;
        device_stats.deepseek_moe_d2h_bytes +=
            downloaded_bytes + sizeof(unsigned int);
        device_stats.deepseek_moe_h2d_nanoseconds += h2d_nanoseconds;
        device_stats.deepseek_moe_kernel_nanoseconds += kernel_nanoseconds;
        device_stats.deepseek_moe_d2h_nanoseconds += d2h_nanoseconds;
        device_stats.deepseek_moe_nanoseconds += total_nanoseconds;
    }
    if (*host_error != 0U) {
        result.errors.emplace_back(
            "DeepSeek MoE W1/W3 produced a non-finite BF16 activation");
        return result;
    }
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

ValidationResult CudaBackend::synchronize(int device) {
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) return {{"cannot synchronize an uninitialized CUDA device"}};
    if (found->second.moe_in_flight) {
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
        ++device_stats.synchronization_calls;
        device_stats.synchronization_nanoseconds += elapsed;
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
        result.synchronization_nanoseconds = std::max(
            result.synchronization_nanoseconds, device.synchronization_nanoseconds);
        result.upload_wait_nanoseconds = std::max(
            result.upload_wait_nanoseconds, device.upload_wait_nanoseconds);
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
    }
    return result;
}

}  // namespace strata
