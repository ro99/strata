#include "strata/device/cuda_backend.hpp"
#include "strata/platform/numerics.hpp"

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
#include <iomanip>
#include <iterator>
#include <limits>
#include <mutex>
#include <new>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <unordered_map>

#include "detail/marlin_adapter.cuh"

namespace strata {

// Keep the CUDA backend in one translation unit for now: several device
// helpers and the private PImpl state intentionally have internal linkage.
// The fragments below are ownership boundaries, not separately linked CUDA
// objects, so this reorganization cannot change dispatch or device linking.
#include "detail/backend_kernels.cuh"
#include "detail/backend_state.cuh"
#include "detail/backend_core.inc.cuh"
#include "detail/backend_gemma.inc.cuh"
#include "detail/backend_attention.inc.cuh"
#include "detail/backend_mhc.inc.cuh"
#include "detail/backend_glm.inc.cuh"
#include "detail/backend_matmul.inc.cuh"
#include "detail/backend_moe.inc.cuh"

}  // namespace strata
