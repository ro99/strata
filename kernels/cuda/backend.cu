#include "strata/device/cuda_backend.hpp"
#include "strata/platform/numerics.hpp"

#include <cublas_v2.h>
#include <cstdlib>
#include <cstdio>
#include <atomic>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <mma.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
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

// Core backend helpers and the private PImpl intentionally share this owning
// translation unit. Vendored Marlin code has its own narrow adapter TU.
#include "detail/backend_kernels.cuh"
#include "detail/backend_model_kernels.cuh"
#include "detail/backend_state.cuh"
#include "detail/backend_core.inc.cuh"
#include "detail/backend_dense_page.inc.cuh"
#include "detail/backend_indexing.inc.cuh"
#include "detail/backend_hybrid_recurrence.inc.cuh"
#include "detail/backend_flash_attention.inc.cuh"
#include "detail/backend_prepared_attention.inc.cuh"
#include "detail/backend_mhc.inc.cuh"
#include "detail/backend_absorbed_attention.inc.cuh"
#include "detail/backend_matmul.inc.cuh"
#include "detail/backend_moe.inc.cuh"

}  // namespace strata
