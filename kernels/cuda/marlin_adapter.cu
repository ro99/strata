#include "detail/marlin_adapter.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstddef>

// This is the only production-code import boundary for the vendored Marlin
// subset. Keep third-party types and Strata's required FP32 epilogue switch
// out of the rest of the backend.
#define STRATA_MARLIN_FP32_OUTPUT 1
#include "../../vendor/marlin/marlin_template.h"
#undef STRATA_MARLIN_FP32_OUTPUT

namespace strata {
namespace {

__device__ float marlin_quantize_e4m3_value(float value) {
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

// Preserve the existing per-row/K128 E4M3 simulation and publish the exact
// simulated value as BF16 for W4A16 MMA.
__global__ void marlin_quantize_activation_e4m3_bf16_kernel(
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
        value = marlin_quantize_e4m3_value(
            input[static_cast<std::uint64_t>(row) * columns + column] /
            scale) * scale;
    }
    output[static_cast<std::uint64_t>(row) * columns + column] =
        __float2bfloat16_rn(value);
}

cudaError_t marlin_grow(void*& pointer, std::uint64_t& capacity,
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
    return cudaMemsetAsync(pointer, 0, static_cast<std::size_t>(required),
                           stream);
}

}  // namespace

cudaError_t launch_gemma_marlin(
    GemmaMarlinWorkspace& workspace, const CudaWeightDescriptor& descriptor,
    const void* weights, const void* scales, const float* input,
    std::uint32_t rows, float* output, cudaStream_t stream,
    bool reuse_activation) {
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
    if (auto status = marlin_grow(workspace.activation,
                                  workspace.activation_bytes,
                                  activation_bytes, false, stream);
        status != cudaSuccess) {
        return status;
    }
    if (auto status = marlin_grow(workspace.reduce, workspace.reduce_bytes,
                                  reduce_bytes, false, stream);
        status != cudaSuccess) {
        return status;
    }
    if (auto status = marlin_grow(workspace.reorder, workspace.reorder_bytes,
                                  reorder_bytes, false, stream);
        status != cudaSuccess) {
        return status;
    }
    if (auto status = marlin_grow(workspace.locks, workspace.lock_bytes,
                                  lock_bytes, true, stream);
        status != cudaSuccess) {
        return status;
    }
    if (!reuse_activation) {
        const dim3 quantize_grid(
            static_cast<unsigned int>((descriptor.columns + 127U) / 128U),
            kernel_rows, 1U);
        marlin_quantize_activation_e4m3_bf16_kernel<<<
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

}  // namespace strata
