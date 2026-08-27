#pragma once

#include "strata/device/cuda_backend.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace strata {

// Backend-owned storage whose layout is shared with the adapter but whose
// implementation and vendored types remain confined to marlin_adapter.cu.
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

cudaError_t launch_gemma_marlin(
    GemmaMarlinWorkspace& workspace, const CudaWeightDescriptor& descriptor,
    const void* weights, const void* scales, const float* input,
    std::uint32_t rows, float* output, cudaStream_t stream,
    bool reuse_activation = false);

}  // namespace strata
