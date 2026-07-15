#include "strata/cuda_backend.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
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

__device__ float plain_value(const void* weights, int dtype, std::uint64_t index) {
    if (dtype == static_cast<int>(SafetensorsDtype::Bf16)) {
        return __bfloat162float(static_cast<const __nv_bfloat16*>(weights)[index]);
    }
    if (dtype == static_cast<int>(SafetensorsDtype::F16)) {
        return __half2float(static_cast<const __half*>(weights)[index]);
    }
    return static_cast<const float*>(weights)[index];
}

__global__ void plain_matmul_kernel(float* output, const float* input,
                                    const void* weights, int dtype,
                                    std::uint32_t batch, std::uint64_t columns,
                                    std::uint64_t rows) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t batch_row = blockIdx.y;
    if (output_row >= rows || batch_row >= batch) return;
    float sum = 0.0F;
    const std::uint64_t weight_base = output_row * columns;
    const std::uint64_t input_base = static_cast<std::uint64_t>(batch_row) * columns;
    for (std::uint64_t column = threadIdx.x; column < columns; column += blockDim.x) {
        sum += plain_value(weights, dtype, weight_base + column) * input[input_base + column];
    }
    sum = reduce_block(sum);
    if (threadIdx.x == 0) {
        output[static_cast<std::uint64_t>(batch_row) * rows + output_row] = sum;
    }
}

__global__ void packed_matmul_kernel(float* output, const float* input,
                                     const std::uint32_t* packed,
                                     const __nv_bfloat16* scales,
                                     std::uint32_t bits, std::uint32_t group_size,
                                     std::uint64_t packed_columns,
                                     std::uint64_t scale_columns,
                                     std::uint32_t batch, std::uint64_t columns,
                                     std::uint64_t rows) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t batch_row = blockIdx.y;
    if (output_row >= rows || batch_row >= batch) return;
    const std::uint32_t lanes = 32U / bits;
    const std::uint32_t mask = (1U << bits) - 1U;
    const std::int32_t offset = 1 << (bits - 1U);
    float sum = 0.0F;
    const std::uint64_t packed_base = output_row * packed_columns;
    const std::uint64_t scale_base = output_row * scale_columns;
    const std::uint64_t input_base = static_cast<std::uint64_t>(batch_row) * columns;
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
        output[static_cast<std::uint64_t>(batch_row) * rows + output_row] = sum;
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

}  // namespace

struct CudaWeight::Impl {
    void* weights{};
    void* scales{};
    CudaWeightDescriptor descriptor;
    std::uint64_t bytes{};
    int device{-1};

    ~Impl() {
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
        float* input{};
        float* output{};
        std::uint64_t input_bytes{};
        std::uint64_t output_bytes{};
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
            if (state.activation_start != nullptr) {
                static_cast<void>(cudaEventDestroy(state.activation_start));
                static_cast<void>(cudaEventDestroy(state.activation_uploaded));
                static_cast<void>(cudaEventDestroy(state.kernel_finished));
                static_cast<void>(cudaEventDestroy(state.activation_downloaded));
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
        impl_->devices.emplace(device, state);
        CudaBackendStats::Device device_stats;
        device_stats.device = device;
        impl_->stats.devices.push_back(device_stats);
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
    } else {
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
    }
    if (weights.size() != expected_weights || scales.size() != expected_scales) {
        result.errors.emplace_back("CUDA weight payload byte count is invalid");
        return result;
    }
    if (auto status = cudaSetDevice(device); status != cudaSuccess) {
        return cuda_error(status, "select CUDA device for upload");
    }
    auto target = std::make_unique<CudaWeight::Impl>();
    target->descriptor = descriptor;
    target->device = device;
    target->bytes = expected_weights + expected_scales;
    if (auto status = cudaMalloc(&target->weights, static_cast<std::size_t>(expected_weights));
        status != cudaSuccess) {
        return cuda_error(status, "allocate CUDA weights");
    }
    auto& state = found->second;
    // This stream is nonblocking with respect to the legacy default stream.
    // Keep uploads on the execution stream and finish them before the caller's
    // host payload is released or the cache publishes the weight.
    if (auto status = cudaMemcpyAsync(target->weights, weights.data(), weights.size(),
                                      cudaMemcpyHostToDevice, state.stream);
        status != cudaSuccess) {
        return cuda_error(status, "upload CUDA weights");
    }
    if (expected_scales != 0U) {
        if (auto status = cudaMalloc(&target->scales, static_cast<std::size_t>(expected_scales));
            status != cudaSuccess) {
            return cuda_error(status, "allocate CUDA scales");
        }
        if (auto status = cudaMemcpyAsync(target->scales, scales.data(), scales.size(),
                                          cudaMemcpyHostToDevice, state.stream);
            status != cudaSuccess) {
            return cuda_error(status, "upload CUDA scales");
        }
    }
    const auto wait_started = std::chrono::steady_clock::now();
    if (auto status = cudaStreamSynchronize(state.stream); status != cudaSuccess) {
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
        device_stats.weight_upload_bytes += target->bytes;
        device_stats.weight_allocation_calls += expected_scales == 0U ? 1U : 2U;
        device_stats.weight_allocation_bytes += target->bytes;
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
    ValidationResult result;
    if (!weight.valid()) {
        result.errors.emplace_back("CUDA matmul received an invalid weight");
        return result;
    }
    const auto& descriptor = weight.impl_->descriptor;
    if (rows == 0U || input.size() != descriptor.columns * rows ||
        output.size() != descriptor.rows * rows) {
        result.errors.emplace_back("CUDA matmul activation shapes are incompatible");
        return result;
    }
    auto& state = impl_->devices.at(weight.impl_->device);
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
    const dim3 grid(static_cast<unsigned int>(descriptor.rows), rows, 1U);
    constexpr unsigned int threads = 256U;
    if (descriptor.encoding == CudaWeightEncoding::Plain) {
        plain_matmul_kernel<<<grid, threads, 0, state.stream>>>(
            state.output, state.input, weight.impl_->weights,
            static_cast<int>(descriptor.dtype), rows, descriptor.columns,
            descriptor.rows);
    } else {
        const auto bits = descriptor.encoding == CudaWeightEncoding::OffsetPackedInt4 ? 4U : 8U;
        packed_matmul_kernel<<<grid, threads, 0, state.stream>>>(
            state.output, state.input, static_cast<const std::uint32_t*>(weight.impl_->weights),
            static_cast<const __nv_bfloat16*>(weight.impl_->scales), bits,
            descriptor.group_size, descriptor.packed_columns,
            descriptor.scale_columns, rows, descriptor.columns, descriptor.rows);
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

ValidationResult CudaBackend::synchronize(int device) {
    const auto found = impl_->devices.find(device);
    if (found == impl_->devices.end()) return {{"cannot synchronize an uninitialized CUDA device"}};
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
    }
    return result;
}

}  // namespace strata
