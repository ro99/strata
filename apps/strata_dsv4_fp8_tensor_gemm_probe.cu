#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {

constexpr std::uint32_t kRows = 677U;
constexpr std::uint32_t kLayers = 43U;
constexpr std::uint32_t kBlockM = 64U;
constexpr std::uint32_t kBlockN = 128U;
constexpr std::uint32_t kBlockK = 128U;
constexpr std::uint32_t kThreads = 256U;
constexpr std::uint32_t kWarmups = 3U;
constexpr std::uint32_t kSamples = 11U;
constexpr std::uint32_t kOracleSamples = 4096U;
constexpr double kRtx3090DenseBf16TensorGflops = 142'000.0;

struct Shape {
    const char* name;
    std::uint32_t n;
    std::uint32_t k;
};

constexpr Shape kShapes[] = {
    {"wq_a", 1024U, 4096U},
    {"wq_b", 32768U, 1024U},
    {"wkv", 512U, 4096U},
};

void check(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

std::uint32_t xorshift(std::uint32_t& state) {
    state ^= state << 13U;
    state ^= state >> 17U;
    state ^= state << 5U;
    return state;
}

__host__ __device__ float decode_e4m3(std::uint8_t encoded) {
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

__host__ __device__ float decode_e8m0(std::uint8_t encoded) {
    return encoded == 0xFFU
               ? nanf("")
               : ldexpf(1.0F, static_cast<int>(encoded) - 127);
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
    value = threadIdx.x < 8U ? warps[lane] : 0.0F;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xFFFF'FFFFU, value, offset);
        }
    }
    return value;
}

// Exact production kernel shape and arithmetic, with groups fixed to zero.
__global__ void native_fp8_matmul_probe(
    float* output, const float* input, const std::uint8_t* weights,
    const std::uint8_t* scales, std::uint64_t scale_columns,
    std::uint32_t batch, std::uint64_t columns, std::uint64_t rows) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t batch_row = blockIdx.y;
    if (output_row >= rows || batch_row >= batch) return;
    const std::uint64_t input_base =
        static_cast<std::uint64_t>(batch_row) * columns;
    const std::uint64_t weight_base = output_row * columns;
    float sum = 0.0F;
    for (std::uint64_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        const float weight = decode_e4m3(weights[weight_base + column]);
        const float scale = decode_e8m0(
            scales[(output_row / 128U) * scale_columns + column / 128U]);
        sum += input[input_base + column] * weight * scale;
    }
    sum = reduce_block(sum);
    if (threadIdx.x == 0U) {
        output[static_cast<std::uint64_t>(batch_row) * rows + output_row] = sum;
    }
}

// SM86 form of the reference's DECODE_E4M3 mechanism. FP8 operands remain
// byte-resident globally, widen exactly to BF16 in shared-memory tiles, and
// feed BF16 tensor-core dots. E8M0 block scale is applied to each K=128 dot
// before the FP32 partials are accumulated.
__global__ void fp8_decode_bf16_tensor_probe(
    float* output, const std::uint8_t* input, const std::uint8_t* weights,
    const std::uint8_t* scales, std::uint32_t batch, std::uint32_t columns,
    std::uint32_t rows) {
    using namespace nvcuda;
    __shared__ __nv_bfloat16 shared_a[kBlockM * kBlockK];
    __shared__ __nv_bfloat16 shared_b[kBlockK * kBlockN];

    const std::uint32_t tile_m = blockIdx.y * kBlockM;
    const std::uint32_t tile_n = blockIdx.x * kBlockN;
    const std::uint32_t warp = threadIdx.x / warpSize;
    const std::uint32_t warp_m = warp & 3U;
    const std::uint32_t warp_n_group = warp >> 2U;
    constexpr std::uint32_t fragments_per_warp = 4U;
    float totals[fragments_per_warp][8]{};
    const std::uint32_t scale_columns = columns / kBlockK;

    for (std::uint32_t tile_k = 0U; tile_k < columns;
         tile_k += kBlockK) {
        for (std::uint32_t index = threadIdx.x;
             index < kBlockM * kBlockK; index += blockDim.x) {
            const std::uint32_t local_m = index / kBlockK;
            const std::uint32_t local_k = index % kBlockK;
            const std::uint32_t global_m = tile_m + local_m;
            const auto encoded = global_m < batch
                ? input[static_cast<std::uint64_t>(global_m) * columns +
                        tile_k + local_k]
                : 0U;
            shared_a[index] = __float2bfloat16_rn(decode_e4m3(encoded));
        }
        for (std::uint32_t index = threadIdx.x;
             index < kBlockK * kBlockN; index += blockDim.x) {
            const std::uint32_t local_k = index / kBlockN;
            const std::uint32_t local_n = index % kBlockN;
            const std::uint32_t global_n = tile_n + local_n;
            const auto encoded = weights[
                static_cast<std::uint64_t>(global_n) * columns + tile_k +
                local_k];
            shared_b[index] = __float2bfloat16_rn(decode_e4m3(encoded));
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
        for (std::uint32_t local_k = 0U; local_k < kBlockK; local_k += 16U) {
            wmma::load_matrix_sync(
                a_fragment,
                shared_a + warp_m * 16U * kBlockK + local_k, kBlockK);
            for (std::uint32_t fragment = 0U;
                 fragment < fragments_per_warp; ++fragment) {
                const std::uint32_t fragment_n =
                    warp_n_group * fragments_per_warp + fragment;
                wmma::load_matrix_sync(
                    b_fragment,
                    shared_b + local_k * kBlockN + fragment_n * 16U,
                    kBlockN);
                wmma::mma_sync(accumulators[fragment], a_fragment, b_fragment,
                               accumulators[fragment]);
            }
        }
        const float scale = decode_e8m0(
            scales[(tile_n / kBlockN) * scale_columns + tile_k / kBlockK]);
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
        for (std::uint32_t element = 0U; element < result.num_elements;
             ++element) {
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

struct DeviceBuffer {
    void* value = nullptr;
    explicit DeviceBuffer(std::size_t bytes) {
        check(cudaMalloc(&value, bytes), "allocate FP8 probe buffer");
    }
    ~DeviceBuffer() {
        if (value != nullptr) static_cast<void>(cudaFree(value));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

template <typename Launch>
float time_once(cudaEvent_t start, cudaEvent_t finish, cudaStream_t stream,
                Launch&& launch) {
    check(cudaEventRecord(start, stream), "record FP8 probe start");
    launch();
    check(cudaEventRecord(finish, stream), "record FP8 probe finish");
    check(cudaEventSynchronize(finish), "synchronize FP8 probe sample");
    float milliseconds = 0.0F;
    check(cudaEventElapsedTime(&milliseconds, start, finish),
          "measure FP8 probe sample");
    return milliseconds;
}

float median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}

struct ErrorMetrics {
    double maximum_absolute = 0.0;
    double maximum_relative = 0.0;
    double mean_absolute = 0.0;
    double rms = 0.0;
};

struct BoundaryMetrics {
    ErrorMetrics error;
    std::uint64_t mismatches = 0U;
};

struct OracleMetrics {
    ErrorMetrics fp32;
    BoundaryMetrics bf16;
};

struct Measurement {
    Shape shape;
    float current_milliseconds = 0.0F;
    float tensor_milliseconds = 0.0F;
    OracleMetrics current_error;
    OracleMetrics tensor_error;
    std::uint64_t current_weight_read_bytes = 0U;
    std::uint64_t tensor_weight_read_bytes = 0U;
    std::uint64_t one_read_weight_bytes = 0U;
    std::uint64_t maximum_device_bytes = 0U;
    bool tensor_no_worse_fp32 = false;
    bool tensor_no_worse_bf16 = false;
};

std::uint16_t float_to_bf16_bits(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto rounded = bits + 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(rounded >> 16U);
}

float bf16_bits_to_float(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

std::vector<std::uint64_t> oracle_indices(std::uint64_t elements,
                                          std::uint32_t seed) {
    std::vector<std::uint64_t> result;
    result.reserve(kOracleSamples);
    std::unordered_set<std::uint64_t> seen;
    seen.reserve(kOracleSamples * 2U);
    seen.insert(0U);
    seen.insert(elements - 1U);
    result.push_back(0U);
    result.push_back(elements - 1U);
    std::uint64_t state = 0x9E37'79B9'7F4A'7C15ULL ^ seed;
    while (result.size() < kOracleSamples) {
        state ^= state << 13U;
        state ^= state >> 7U;
        state ^= state << 17U;
        const auto index = state % elements;
        if (seen.insert(index).second) result.push_back(index);
    }
    return result;
}

OracleMetrics compare_oracle(
    const Shape& shape, const std::vector<std::uint8_t>& input,
    const std::vector<std::uint8_t>& weights,
    const std::vector<std::uint8_t>& scales,
    const std::vector<float>& output,
    const std::vector<std::uint64_t>& indices) {
    long double absolute_sum = 0.0;
    long double squared_sum = 0.0;
    long double boundary_absolute_sum = 0.0;
    long double boundary_squared_sum = 0.0;
    OracleMetrics result;
    const std::uint32_t scale_columns = shape.k / kBlockK;
    for (const auto index : indices) {
        const std::uint32_t row = static_cast<std::uint32_t>(index / shape.n);
        const std::uint32_t column = static_cast<std::uint32_t>(index % shape.n);
        double oracle = 0.0;
        for (std::uint32_t k = 0U; k < shape.k; ++k) {
            const double activation = decode_e4m3(
                input[static_cast<std::uint64_t>(row) * shape.k + k]);
            const double weight = decode_e4m3(
                weights[static_cast<std::uint64_t>(column) * shape.k + k]);
            const double scale = decode_e8m0(
                scales[(column / kBlockN) * scale_columns + k / kBlockK]);
            oracle += activation * weight * scale;
        }
        const double absolute = std::abs(static_cast<double>(output[index]) - oracle);
        const double relative = absolute / std::max(std::abs(oracle), 1.0e-9);
        result.fp32.maximum_absolute = std::max(
            result.fp32.maximum_absolute, absolute);
        result.fp32.maximum_relative = std::max(
            result.fp32.maximum_relative, relative);
        absolute_sum += absolute;
        squared_sum += absolute * absolute;

        // The runtime carries projection output in FP32 only as an ABI. It
        // rounds that value to BF16 on-device before any query/KV consumer.
        // The oracle follows the same FP32-carrier then BF16 boundary.
        const auto actual_bits = float_to_bf16_bits(output[index]);
        const auto oracle_bits = float_to_bf16_bits(static_cast<float>(oracle));
        const double actual_bf16 = bf16_bits_to_float(actual_bits);
        const double oracle_bf16 = bf16_bits_to_float(oracle_bits);
        const double boundary_absolute = std::abs(actual_bf16 - oracle_bf16);
        const double boundary_relative = boundary_absolute /
            std::max(std::abs(oracle_bf16), 1.0e-9);
        result.bf16.error.maximum_absolute = std::max(
            result.bf16.error.maximum_absolute, boundary_absolute);
        result.bf16.error.maximum_relative = std::max(
            result.bf16.error.maximum_relative, boundary_relative);
        boundary_absolute_sum += boundary_absolute;
        boundary_squared_sum += boundary_absolute * boundary_absolute;
        if (actual_bits != oracle_bits) ++result.bf16.mismatches;
    }
    result.fp32.mean_absolute = static_cast<double>(
        absolute_sum / static_cast<long double>(indices.size()));
    result.fp32.rms = std::sqrt(static_cast<double>(
        squared_sum / static_cast<long double>(indices.size())));
    result.bf16.error.mean_absolute = static_cast<double>(
        boundary_absolute_sum / static_cast<long double>(indices.size()));
    result.bf16.error.rms = std::sqrt(static_cast<double>(
        boundary_squared_sum / static_cast<long double>(indices.size())));
    return result;
}

Measurement run_shape(const Shape& shape, cudaStream_t stream) {
    const std::uint64_t input_elements =
        static_cast<std::uint64_t>(kRows) * shape.k;
    const std::uint64_t weight_elements =
        static_cast<std::uint64_t>(shape.n) * shape.k;
    const std::uint64_t scale_elements =
        static_cast<std::uint64_t>(shape.n / kBlockN) *
        (shape.k / kBlockK);
    const std::uint32_t padded_rows =
        (kRows + kBlockM - 1U) / kBlockM * kBlockM;
    const std::uint64_t output_elements =
        static_cast<std::uint64_t>(padded_rows) * shape.n;

    std::vector<std::uint8_t> input(input_elements);
    std::vector<float> current_input(input_elements);
    std::vector<std::uint8_t> weights(weight_elements);
    std::vector<std::uint8_t> scales(scale_elements);
    std::uint32_t random_state = 0xA341'316CU ^ shape.n ^ (shape.k << 8U);
    const auto random_fp8 = [&] {
        const auto sign = (xorshift(random_state) & 1U) << 7U;
        const auto exponent = (4U + xorshift(random_state) % 8U) << 3U;
        const auto mantissa = xorshift(random_state) & 7U;
        return static_cast<std::uint8_t>(sign | exponent | mantissa);
    };
    for (std::uint64_t index = 0U; index < input_elements; ++index) {
        input[index] = random_fp8();
        current_input[index] = decode_e4m3(input[index]);
    }
    for (auto& weight : weights) weight = random_fp8();
    for (auto& scale : scales) {
        scale = static_cast<std::uint8_t>(123U + xorshift(random_state) % 9U);
    }

    DeviceBuffer device_input(input_elements);
    DeviceBuffer device_current_input(input_elements * sizeof(float));
    DeviceBuffer device_weights(weight_elements);
    DeviceBuffer device_scales(scale_elements);
    DeviceBuffer device_output(output_elements * sizeof(float));
    check(cudaMemcpyAsync(device_input.value, input.data(), input_elements,
                          cudaMemcpyHostToDevice, stream),
          "upload encoded FP8 activation");
    check(cudaMemcpyAsync(device_current_input.value, current_input.data(),
                          input_elements * sizeof(float),
                          cudaMemcpyHostToDevice, stream),
          "upload decoded production activation");
    check(cudaMemcpyAsync(device_weights.value, weights.data(), weight_elements,
                          cudaMemcpyHostToDevice, stream),
          "upload encoded FP8 weights");
    check(cudaMemcpyAsync(device_scales.value, scales.data(), scale_elements,
                          cudaMemcpyHostToDevice, stream),
          "upload E8M0 scales");
    check(cudaStreamSynchronize(stream), "finish FP8 probe uploads");

    const auto launch_current = [&] {
        native_fp8_matmul_probe<<<dim3{shape.n, kRows}, kThreads, 0U, stream>>>(
            static_cast<float*>(device_output.value),
            static_cast<const float*>(device_current_input.value),
            static_cast<const std::uint8_t*>(device_weights.value),
            static_cast<const std::uint8_t*>(device_scales.value),
            shape.k / kBlockK, kRows, shape.k, shape.n);
        check(cudaGetLastError(), "launch native FP8 probe");
    };
    const auto launch_tensor = [&] {
        fp8_decode_bf16_tensor_probe<<<
            dim3{shape.n / kBlockN,
                 (kRows + kBlockM - 1U) / kBlockM},
            kThreads, 0U, stream>>>(
            static_cast<float*>(device_output.value),
            static_cast<const std::uint8_t*>(device_input.value),
            static_cast<const std::uint8_t*>(device_weights.value),
            static_cast<const std::uint8_t*>(device_scales.value), kRows,
            shape.k, shape.n);
        check(cudaGetLastError(), "launch FP8 tensor-core probe");
    };

    cudaEvent_t started = nullptr;
    cudaEvent_t finished = nullptr;
    check(cudaEventCreate(&started), "create FP8 probe start event");
    check(cudaEventCreate(&finished), "create FP8 probe finish event");
    for (std::uint32_t warmup = 0U; warmup < kWarmups; ++warmup) {
        launch_current();
        launch_tensor();
    }
    check(cudaStreamSynchronize(stream), "finish FP8 probe warmups");
    std::vector<float> current_times;
    std::vector<float> tensor_times;
    current_times.reserve(kSamples);
    tensor_times.reserve(kSamples);
    for (std::uint32_t sample = 0U; sample < kSamples; ++sample) {
        if ((sample & 1U) == 0U) {
            current_times.push_back(time_once(
                started, finished, stream, launch_current));
            tensor_times.push_back(time_once(
                started, finished, stream, launch_tensor));
        } else {
            tensor_times.push_back(time_once(
                started, finished, stream, launch_tensor));
            current_times.push_back(time_once(
                started, finished, stream, launch_current));
        }
    }

    std::vector<float> current_output(output_elements);
    std::vector<float> tensor_output(output_elements);
    launch_current();
    check(cudaMemcpyAsync(current_output.data(), device_output.value,
                          output_elements * sizeof(float),
                          cudaMemcpyDeviceToHost, stream),
          "download native FP8 output");
    check(cudaStreamSynchronize(stream), "finish native FP8 output");
    launch_tensor();
    check(cudaMemcpyAsync(tensor_output.data(), device_output.value,
                          output_elements * sizeof(float),
                          cudaMemcpyDeviceToHost, stream),
          "download tensor FP8 output");
    check(cudaStreamSynchronize(stream), "finish tensor FP8 output");
    check(cudaEventDestroy(started), "destroy FP8 probe start event");
    check(cudaEventDestroy(finished), "destroy FP8 probe finish event");

    const auto indices = oracle_indices(
        static_cast<std::uint64_t>(kRows) * shape.n,
        shape.n ^ (shape.k << 8U));
    Measurement result;
    result.shape = shape;
    result.current_milliseconds = median(std::move(current_times));
    result.tensor_milliseconds = median(std::move(tensor_times));
    result.current_error = compare_oracle(
        shape, input, weights, scales, current_output, indices);
    result.tensor_error = compare_oracle(
        shape, input, weights, scales, tensor_output, indices);
    result.current_weight_read_bytes =
        static_cast<std::uint64_t>(kRows) * weight_elements;
    result.tensor_weight_read_bytes =
        static_cast<std::uint64_t>(
            (kRows + kBlockM - 1U) / kBlockM) * weight_elements;
    result.one_read_weight_bytes = weight_elements;
    result.maximum_device_bytes = input_elements * (1U + sizeof(float)) +
        weight_elements + scale_elements + output_elements * sizeof(float);
    result.tensor_no_worse_fp32 =
        result.tensor_error.fp32.maximum_absolute <=
            result.current_error.fp32.maximum_absolute &&
        result.tensor_error.fp32.rms <= result.current_error.fp32.rms;
    result.tensor_no_worse_bf16 =
        result.tensor_error.bf16.error.maximum_absolute <=
            result.current_error.bf16.error.maximum_absolute &&
        result.tensor_error.bf16.error.maximum_relative <=
            result.current_error.bf16.error.maximum_relative &&
        result.tensor_error.bf16.error.rms <=
            result.current_error.bf16.error.rms &&
        result.tensor_error.bf16.mismatches <=
            result.current_error.bf16.mismatches;
    return result;
}

void print_error(const char* prefix, const ErrorMetrics& error) {
    std::cout << "      \"" << prefix << "_maximum_absolute_error\": "
              << error.maximum_absolute << ",\n"
              << "      \"" << prefix << "_maximum_relative_error_floor_1e_9\": "
              << error.maximum_relative << ",\n"
              << "      \"" << prefix << "_mean_absolute_error\": "
              << error.mean_absolute << ",\n"
              << "      \"" << prefix << "_rms_error\": "
              << error.rms << ",\n";
}

void print_boundary(const char* prefix, const BoundaryMetrics& boundary) {
    std::cout << "      \"" << prefix << "_bf16_maximum_absolute_error\": "
              << boundary.error.maximum_absolute << ",\n"
              << "      \"" << prefix << "_bf16_maximum_relative_error_floor_1e_9\": "
              << boundary.error.maximum_relative << ",\n"
              << "      \"" << prefix << "_bf16_mean_absolute_error\": "
              << boundary.error.mean_absolute << ",\n"
              << "      \"" << prefix << "_bf16_rms_error\": "
              << boundary.error.rms << ",\n"
              << "      \"" << prefix << "_bf16_mismatches\": "
              << boundary.mismatches << ",\n"
              << "      \"" << prefix << "_bf16_mismatch_rate\": "
              << static_cast<double>(boundary.mismatches) / kOracleSamples
              << ",\n";
}

void print_measurement(const Measurement& value, bool final) {
    const double flops = 2.0 * kRows * value.shape.n * value.shape.k;
    const double current_gflops =
        flops / (static_cast<double>(value.current_milliseconds) * 1.0e6);
    const double tensor_gflops =
        flops / (static_cast<double>(value.tensor_milliseconds) * 1.0e6);
    std::cout << "    {\n"
              << "      \"name\": \"" << value.shape.name << "\",\n"
              << "      \"m\": " << kRows << ",\n"
              << "      \"n\": " << value.shape.n << ",\n"
              << "      \"k\": " << value.shape.k << ",\n"
              << "      \"current_median_ms\": "
              << value.current_milliseconds << ",\n"
              << "      \"tensor_median_ms\": "
              << value.tensor_milliseconds << ",\n"
              << "      \"tensor_over_current\": "
              << value.current_milliseconds / value.tensor_milliseconds
              << ",\n"
              << "      \"current_gflops\": " << current_gflops << ",\n"
              << "      \"tensor_gflops\": " << tensor_gflops << ",\n"
              << "      \"current_fraction_of_peak\": "
              << current_gflops / kRtx3090DenseBf16TensorGflops << ",\n"
              << "      \"tensor_fraction_of_peak\": "
              << tensor_gflops / kRtx3090DenseBf16TensorGflops << ",\n"
              << "      \"current_weight_read_bytes\": "
              << value.current_weight_read_bytes << ",\n"
              << "      \"tensor_weight_read_bytes\": "
              << value.tensor_weight_read_bytes << ",\n"
              << "      \"one_read_weight_bytes\": "
              << value.one_read_weight_bytes << ",\n";
    print_error("current", value.current_error.fp32);
    print_error("tensor", value.tensor_error.fp32);
    print_boundary("current", value.current_error.bf16);
    print_boundary("tensor", value.tensor_error.bf16);
    std::cout << "      \"tensor_no_worse_fp32_max_and_rms\": "
              << (value.tensor_no_worse_fp32 ? "true" : "false") << ",\n"
              << "      \"tensor_no_worse_bf16_all_metrics\": "
              << (value.tensor_no_worse_bf16 ? "true" : "false") << ",\n"
              << "      \"oracle_samples\": " << kOracleSamples << ",\n"
              << "      \"maximum_device_bytes\": "
              << value.maximum_device_bytes << "\n"
              << "    }" << (final ? "\n" : ",\n");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        int device = 1;
        if (argc == 3 && std::string_view(argv[1]) == "--device") {
            device = std::stoi(argv[2]);
        } else if (argc != 1) {
            std::cerr << "usage: " << argv[0] << " [--device INDEX]\n";
            return EXIT_FAILURE;
        }
        check(cudaSetDevice(device), "select FP8 probe device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device),
              "query FP8 probe device");
        if (properties.major != 8 || properties.minor != 6) {
            throw std::runtime_error("FP8 tensor probe requires SM86");
        }
        cudaStream_t stream = nullptr;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "create FP8 probe stream");
        std::vector<Measurement> measurements;
        measurements.reserve(std::size(kShapes));
        for (const auto& shape : kShapes) {
            measurements.push_back(run_shape(shape, stream));
        }

        std::cout << std::fixed << std::setprecision(9)
                  << "{\n"
                  << "  \"device_index\": " << device << ",\n"
                  << "  \"device_name\": \"" << properties.name << "\",\n"
                  << "  \"rated_dense_bf16_tensor_gflops\": "
                  << kRtx3090DenseBf16TensorGflops << ",\n"
                  << "  \"layers\": " << kLayers << ",\n"
                  << "  \"block_m\": " << kBlockM << ",\n"
                  << "  \"block_n\": " << kBlockN << ",\n"
                  << "  \"block_k\": " << kBlockK << ",\n"
                  << "  \"warmups\": " << kWarmups << ",\n"
                  << "  \"samples\": " << kSamples << ",\n"
                  << "  \"shapes\": [\n";
        for (std::size_t index = 0U; index < measurements.size(); ++index) {
            print_measurement(measurements[index],
                              index + 1U == measurements.size());
        }
        std::cout << "  ]\n}\n";
        check(cudaStreamDestroy(stream), "destroy FP8 probe stream");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
