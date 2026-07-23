#pragma once

#include "strata/cuda_backend.hpp"

#include <algorithm>

namespace strata::detail {

inline void assign_cuda_device_delta(CudaBackendStats::Device& result,
                                     const CudaBackendStats::Device& after,
                                     const CudaBackendStats::Device& before) {
    result.device = after.device;
#define STRATA_CUDA_DEVICE_DELTA(field) result.field = after.field - before.field
    STRATA_CUDA_DEVICE_DELTA(weight_upload_bytes);
    STRATA_CUDA_DEVICE_DELTA(activation_h2d_bytes);
    STRATA_CUDA_DEVICE_DELTA(activation_d2h_bytes);
    STRATA_CUDA_DEVICE_DELTA(matmul_calls);
    STRATA_CUDA_DEVICE_DELTA(weight_allocation_calls);
    STRATA_CUDA_DEVICE_DELTA(weight_allocation_bytes);
    STRATA_CUDA_DEVICE_DELTA(workspace_allocation_calls);
    STRATA_CUDA_DEVICE_DELTA(workspace_allocation_bytes);
    STRATA_CUDA_DEVICE_DELTA(synchronization_calls);
    STRATA_CUDA_DEVICE_DELTA(synchronization_nanoseconds);
    STRATA_CUDA_DEVICE_DELTA(upload_wait_nanoseconds);
    STRATA_CUDA_DEVICE_DELTA(activation_h2d_nanoseconds);
    STRATA_CUDA_DEVICE_DELTA(kernel_nanoseconds);
    STRATA_CUDA_DEVICE_DELTA(activation_d2h_nanoseconds);
    STRATA_CUDA_DEVICE_DELTA(deepseek_moe_calls);
    STRATA_CUDA_DEVICE_DELTA(deepseek_moe_kernel_launches);
    STRATA_CUDA_DEVICE_DELTA(deepseek_moe_h2d_transfers);
    STRATA_CUDA_DEVICE_DELTA(deepseek_moe_d2h_transfers);
    STRATA_CUDA_DEVICE_DELTA(deepseek_moe_h2d_bytes);
    STRATA_CUDA_DEVICE_DELTA(deepseek_moe_d2h_bytes);
    STRATA_CUDA_DEVICE_DELTA(deepseek_moe_h2d_nanoseconds);
    STRATA_CUDA_DEVICE_DELTA(deepseek_moe_kernel_nanoseconds);
    STRATA_CUDA_DEVICE_DELTA(deepseek_moe_d2h_nanoseconds);
    STRATA_CUDA_DEVICE_DELTA(deepseek_moe_nanoseconds);
    STRATA_CUDA_DEVICE_DELTA(flash_attention_calls);
    STRATA_CUDA_DEVICE_DELTA(flash_attention_kernel_launches);
    STRATA_CUDA_DEVICE_DELTA(flash_attention_h2d_transfers);
    STRATA_CUDA_DEVICE_DELTA(flash_attention_d2h_transfers);
    STRATA_CUDA_DEVICE_DELTA(flash_attention_h2d_bytes);
    STRATA_CUDA_DEVICE_DELTA(flash_attention_d2h_bytes);
    STRATA_CUDA_DEVICE_DELTA(flash_attention_useful_staging_bytes);
    STRATA_CUDA_DEVICE_DELTA(flash_attention_wasted_staging_bytes);
    STRATA_CUDA_DEVICE_DELTA(flash_attention_h2d_nanoseconds);
    STRATA_CUDA_DEVICE_DELTA(flash_attention_kernel_nanoseconds);
    STRATA_CUDA_DEVICE_DELTA(flash_attention_d2h_nanoseconds);
    STRATA_CUDA_DEVICE_DELTA(flash_attention_nanoseconds);
    STRATA_CUDA_DEVICE_DELTA(lightning_index_calls);
    STRATA_CUDA_DEVICE_DELTA(lightning_index_kernel_launches);
    STRATA_CUDA_DEVICE_DELTA(lightning_index_candidates);
    STRATA_CUDA_DEVICE_DELTA(lightning_index_selected);
    STRATA_CUDA_DEVICE_DELTA(lightning_index_h2d_transfers);
    STRATA_CUDA_DEVICE_DELTA(lightning_index_d2h_transfers);
    STRATA_CUDA_DEVICE_DELTA(lightning_index_h2d_bytes);
    STRATA_CUDA_DEVICE_DELTA(lightning_index_d2h_bytes);
    STRATA_CUDA_DEVICE_DELTA(lightning_index_useful_selection_bytes);
    STRATA_CUDA_DEVICE_DELTA(lightning_index_h2d_nanoseconds);
    STRATA_CUDA_DEVICE_DELTA(lightning_index_kernel_nanoseconds);
    STRATA_CUDA_DEVICE_DELTA(lightning_index_d2h_nanoseconds);
    STRATA_CUDA_DEVICE_DELTA(lightning_index_nanoseconds);
#undef STRATA_CUDA_DEVICE_DELTA
}

inline void assign_cuda_delta(CudaBackendStats& result,
                              const CudaBackendStats& after,
                              const CudaBackendStats& before) {
#define STRATA_CUDA_DELTA(field) result.field = after.field - before.field
    STRATA_CUDA_DELTA(weight_upload_bytes);
    STRATA_CUDA_DELTA(activation_h2d_bytes);
    STRATA_CUDA_DELTA(activation_d2h_bytes);
    STRATA_CUDA_DELTA(matmul_calls);
    STRATA_CUDA_DELTA(weight_allocation_calls);
    STRATA_CUDA_DELTA(weight_allocation_bytes);
    STRATA_CUDA_DELTA(workspace_allocation_calls);
    STRATA_CUDA_DELTA(workspace_allocation_bytes);
    STRATA_CUDA_DELTA(synchronization_calls);
    STRATA_CUDA_DELTA(synchronization_nanoseconds);
    STRATA_CUDA_DELTA(upload_wait_nanoseconds);
    STRATA_CUDA_DELTA(activation_h2d_nanoseconds);
    STRATA_CUDA_DELTA(kernel_nanoseconds);
    STRATA_CUDA_DELTA(activation_d2h_nanoseconds);
    STRATA_CUDA_DELTA(deepseek_moe_calls);
    STRATA_CUDA_DELTA(deepseek_moe_kernel_launches);
    STRATA_CUDA_DELTA(deepseek_moe_h2d_transfers);
    STRATA_CUDA_DELTA(deepseek_moe_d2h_transfers);
    STRATA_CUDA_DELTA(deepseek_moe_h2d_bytes);
    STRATA_CUDA_DELTA(deepseek_moe_d2h_bytes);
    STRATA_CUDA_DELTA(deepseek_moe_h2d_nanoseconds);
    STRATA_CUDA_DELTA(deepseek_moe_kernel_nanoseconds);
    STRATA_CUDA_DELTA(deepseek_moe_d2h_nanoseconds);
    STRATA_CUDA_DELTA(deepseek_moe_nanoseconds);
    STRATA_CUDA_DELTA(flash_attention_calls);
    STRATA_CUDA_DELTA(flash_attention_kernel_launches);
    STRATA_CUDA_DELTA(flash_attention_h2d_transfers);
    STRATA_CUDA_DELTA(flash_attention_d2h_transfers);
    STRATA_CUDA_DELTA(flash_attention_h2d_bytes);
    STRATA_CUDA_DELTA(flash_attention_d2h_bytes);
    STRATA_CUDA_DELTA(flash_attention_useful_staging_bytes);
    STRATA_CUDA_DELTA(flash_attention_wasted_staging_bytes);
    STRATA_CUDA_DELTA(flash_attention_h2d_nanoseconds);
    STRATA_CUDA_DELTA(flash_attention_kernel_nanoseconds);
    STRATA_CUDA_DELTA(flash_attention_d2h_nanoseconds);
    STRATA_CUDA_DELTA(flash_attention_nanoseconds);
    STRATA_CUDA_DELTA(lightning_index_calls);
    STRATA_CUDA_DELTA(lightning_index_kernel_launches);
    STRATA_CUDA_DELTA(lightning_index_candidates);
    STRATA_CUDA_DELTA(lightning_index_selected);
    STRATA_CUDA_DELTA(lightning_index_h2d_transfers);
    STRATA_CUDA_DELTA(lightning_index_d2h_transfers);
    STRATA_CUDA_DELTA(lightning_index_h2d_bytes);
    STRATA_CUDA_DELTA(lightning_index_d2h_bytes);
    STRATA_CUDA_DELTA(lightning_index_useful_selection_bytes);
    STRATA_CUDA_DELTA(lightning_index_h2d_nanoseconds);
    STRATA_CUDA_DELTA(lightning_index_kernel_nanoseconds);
    STRATA_CUDA_DELTA(lightning_index_d2h_nanoseconds);
    STRATA_CUDA_DELTA(lightning_index_nanoseconds);
#undef STRATA_CUDA_DELTA
}

inline void clear_cuda_critical_path(CudaBackendStats& result) noexcept {
    result.synchronization_nanoseconds = 0U;
    result.upload_wait_nanoseconds = 0U;
    result.activation_h2d_nanoseconds = 0U;
    result.kernel_nanoseconds = 0U;
    result.activation_d2h_nanoseconds = 0U;
    result.deepseek_moe_h2d_nanoseconds = 0U;
    result.deepseek_moe_kernel_nanoseconds = 0U;
    result.deepseek_moe_d2h_nanoseconds = 0U;
    result.deepseek_moe_nanoseconds = 0U;
    result.flash_attention_h2d_nanoseconds = 0U;
    result.flash_attention_kernel_nanoseconds = 0U;
    result.flash_attention_d2h_nanoseconds = 0U;
    result.flash_attention_nanoseconds = 0U;
}

inline void accumulate_cuda_critical_path(CudaBackendStats& result,
                                          const CudaBackendStats::Device& delta) noexcept {
    result.synchronization_nanoseconds = std::max(
        result.synchronization_nanoseconds, delta.synchronization_nanoseconds);
    result.upload_wait_nanoseconds = std::max(
        result.upload_wait_nanoseconds, delta.upload_wait_nanoseconds);
    result.activation_h2d_nanoseconds = std::max(
        result.activation_h2d_nanoseconds, delta.activation_h2d_nanoseconds);
    result.kernel_nanoseconds = std::max(
        result.kernel_nanoseconds, delta.kernel_nanoseconds);
    result.activation_d2h_nanoseconds = std::max(
        result.activation_d2h_nanoseconds, delta.activation_d2h_nanoseconds);
    result.deepseek_moe_h2d_nanoseconds = std::max(
        result.deepseek_moe_h2d_nanoseconds, delta.deepseek_moe_h2d_nanoseconds);
    result.deepseek_moe_kernel_nanoseconds = std::max(
        result.deepseek_moe_kernel_nanoseconds, delta.deepseek_moe_kernel_nanoseconds);
    result.deepseek_moe_d2h_nanoseconds = std::max(
        result.deepseek_moe_d2h_nanoseconds, delta.deepseek_moe_d2h_nanoseconds);
    result.deepseek_moe_nanoseconds = std::max(
        result.deepseek_moe_nanoseconds, delta.deepseek_moe_nanoseconds);
    result.flash_attention_h2d_nanoseconds = std::max(
        result.flash_attention_h2d_nanoseconds, delta.flash_attention_h2d_nanoseconds);
    result.flash_attention_kernel_nanoseconds = std::max(
        result.flash_attention_kernel_nanoseconds, delta.flash_attention_kernel_nanoseconds);
    result.flash_attention_d2h_nanoseconds = std::max(
        result.flash_attention_d2h_nanoseconds, delta.flash_attention_d2h_nanoseconds);
    result.flash_attention_nanoseconds = std::max(
        result.flash_attention_nanoseconds, delta.flash_attention_nanoseconds);
}

inline CudaBackendStats cuda_delta(const CudaBackendStats& after,
                                   const CudaBackendStats& before) {
    CudaBackendStats result;
    assign_cuda_delta(result, after, before);
    clear_cuda_critical_path(result);
    result.devices.reserve(after.devices.size());
    for (const auto& device_after : after.devices) {
        const auto found = std::find_if(
            before.devices.begin(), before.devices.end(),
            [&device_after](const auto& value) { return value.device == device_after.device; });
        if (found == before.devices.end()) {
            result.devices.push_back(device_after);
        } else {
            CudaBackendStats::Device delta;
            assign_cuda_device_delta(delta, device_after, *found);
            result.devices.push_back(delta);
        }
        accumulate_cuda_critical_path(result, result.devices.back());
    }
    return result;
}

}  // namespace strata::detail
