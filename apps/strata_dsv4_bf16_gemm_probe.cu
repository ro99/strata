#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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
#include <vector>

namespace {

constexpr std::uint32_t kRows = 677U;
constexpr std::uint32_t kTile = 16U;
constexpr std::uint32_t kWarmups = 5U;
constexpr std::uint32_t kSamples = 21U;
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

void check(cublasStatus_t status, std::string_view operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) +
                                 ": cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

std::uint16_t float_to_bf16_bits(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto rounded = bits + 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(rounded >> 16U);
}

float bf16_bits_to_float(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

float round_bf16(float value) {
    return bf16_bits_to_float(float_to_bf16_bits(value));
}

std::uint32_t xorshift(std::uint32_t& state) {
    state ^= state << 13U;
    state ^= state >> 17U;
    state ^= state << 5U;
    return state;
}

std::uint16_t random_finite_bf16(std::uint32_t& state,
                                 std::uint32_t minimum_exponent,
                                 std::uint32_t exponent_count) {
    const auto sign = (xorshift(state) & 1U) << 15U;
    const auto exponent =
        (minimum_exponent + xorshift(state) % exponent_count) << 7U;
    const auto mantissa = xorshift(state) & 0x7FU;
    return static_cast<std::uint16_t>(sign | exponent | mantissa);
}

// Exact copy of the production multi-row plain-BF16 kernel. Keeping the probe
// standalone avoids changing or exposing the runtime path under measurement.
template <std::uint32_t Tile>
__global__ void bf16_matvec_rows_kernel_probe(
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

__global__ void round_float_bf16_probe(float* values, std::uint64_t count) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < count) {
        values[index] = __bfloat162float(__float2bfloat16_rn(values[index]));
    }
}

struct DeviceBuffer {
    void* value = nullptr;

    explicit DeviceBuffer(std::size_t bytes) {
        check(cudaMalloc(&value, bytes), "allocate probe device buffer");
    }
    ~DeviceBuffer() {
        if (value != nullptr) static_cast<void>(cudaFree(value));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

template <typename Launch>
float time_once(cudaEvent_t started, cudaEvent_t finished,
                cudaStream_t stream, Launch&& launch) {
    check(cudaEventRecord(started, stream), "record probe start");
    launch();
    check(cudaEventRecord(finished, stream), "record probe finish");
    check(cudaEventSynchronize(finished), "synchronize probe sample");
    float milliseconds = 0.0F;
    check(cudaEventElapsedTime(&milliseconds, started, finished),
          "measure probe sample");
    return milliseconds;
}

float median(std::vector<float> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

struct Difference {
    double maximum_absolute = 0.0;
    double maximum_relative = 0.0;
    double mean_absolute = 0.0;
    double maximum_bf16_absolute = 0.0;
    double maximum_bf16_relative = 0.0;
    double canonical_magnitude = 0.0;
    std::uint64_t bf16_mismatches = 0U;
    std::uint32_t maximum_bf16_ulp_distance = 0U;
};

std::uint16_t ordered_bf16(std::uint16_t bits) {
    return (bits & 0x8000U) != 0U
               ? static_cast<std::uint16_t>(~bits)
               : static_cast<std::uint16_t>(bits | 0x8000U);
}

Difference compare(const std::vector<float>& canonical,
                   const std::vector<float>& candidate) {
    Difference result;
    long double absolute_sum = 0.0;
    for (std::size_t index = 0U; index < canonical.size(); ++index) {
        const double expected = canonical[index];
        const double actual = candidate[index];
        const double absolute = std::abs(actual - expected);
        const double denominator = std::max(std::abs(expected), 1.0e-7);
        result.maximum_absolute = std::max(result.maximum_absolute, absolute);
        result.maximum_relative = std::max(
            result.maximum_relative, absolute / denominator);
        result.canonical_magnitude = std::max(
            result.canonical_magnitude, std::abs(expected));
        absolute_sum += absolute;

        const float expected_bf16 = round_bf16(canonical[index]);
        const float actual_bf16 = round_bf16(candidate[index]);
        const auto expected_bits = float_to_bf16_bits(expected_bf16);
        const auto actual_bits = float_to_bf16_bits(actual_bf16);
        const double bf16_absolute =
            std::abs(static_cast<double>(actual_bf16) - expected_bf16);
        const double bf16_denominator =
            std::max(std::abs(static_cast<double>(expected_bf16)), 1.0e-7);
        result.maximum_bf16_absolute = std::max(
            result.maximum_bf16_absolute, bf16_absolute);
        result.maximum_bf16_relative = std::max(
            result.maximum_bf16_relative,
            bf16_absolute / bf16_denominator);
        if (expected_bits != actual_bits) {
            ++result.bf16_mismatches;
        }
        const auto expected_ordered = ordered_bf16(expected_bits);
        const auto actual_ordered = ordered_bf16(actual_bits);
        const auto ulp_distance = expected_ordered > actual_ordered
            ? expected_ordered - actual_ordered
            : actual_ordered - expected_ordered;
        result.maximum_bf16_ulp_distance = std::max(
            result.maximum_bf16_ulp_distance,
            static_cast<std::uint32_t>(ulp_distance));
    }
    result.mean_absolute = static_cast<double>(
        absolute_sum / static_cast<long double>(canonical.size()));
    return result;
}

struct Measurement {
    Shape shape;
    float current_milliseconds = 0.0F;
    float current_rounded_milliseconds = 0.0F;
    float cublas_milliseconds = 0.0F;
    float cublas_rounded_milliseconds = 0.0F;
    Difference difference;
    std::uint64_t cublas_repeat_fp32_mismatches = 0U;
    std::uint64_t output_elements = 0U;
    std::uint64_t maximum_device_bytes = 0U;
};

Measurement run_shape(const Shape& shape, cublasHandle_t handle,
                      cudaStream_t stream) {
    const std::uint64_t input_elements =
        static_cast<std::uint64_t>(kRows) * shape.k;
    const std::uint64_t weight_elements =
        static_cast<std::uint64_t>(shape.n) * shape.k;
    const std::uint64_t output_elements =
        static_cast<std::uint64_t>(kRows) * shape.n;

    std::vector<std::uint16_t> input_bf16(input_elements);
    std::vector<float> input_f32(input_elements);
    std::vector<std::uint16_t> weights(weight_elements);
    std::uint32_t random_state = 0x9E37'79B9U ^ shape.n ^ (shape.k << 8U);
    for (std::uint64_t index = 0U; index < input_elements; ++index) {
        // Vary exponent and mantissa. A fixed binary scale makes a BF16 dot
        // product an exact integer sum at these K values and falsely hides
        // reassociation differences between the two reduction trees.
        input_bf16[index] = random_finite_bf16(random_state, 123U, 5U);
        input_f32[index] = bf16_bits_to_float(input_bf16[index]);
    }
    for (std::uint64_t index = 0U; index < weight_elements; ++index) {
        weights[index] = random_finite_bf16(random_state, 119U, 5U);
    }

    DeviceBuffer device_input_f32(input_elements * sizeof(float));
    DeviceBuffer device_input_bf16(input_elements * sizeof(std::uint16_t));
    DeviceBuffer device_weights(weight_elements * sizeof(std::uint16_t));
    DeviceBuffer device_output(output_elements * sizeof(float));
    check(cudaMemcpyAsync(device_input_f32.value, input_f32.data(),
                          input_elements * sizeof(float),
                          cudaMemcpyHostToDevice, stream),
          "upload FP32 activation");
    check(cudaMemcpyAsync(device_input_bf16.value, input_bf16.data(),
                          input_elements * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice, stream),
          "upload BF16 activation");
    check(cudaMemcpyAsync(device_weights.value, weights.data(),
                          weight_elements * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice, stream),
          "upload BF16 weights");
    check(cudaStreamSynchronize(stream), "finish probe uploads");

    constexpr unsigned int threads = 256U;
    constexpr unsigned int warps_per_block = threads / 32U;
    const dim3 current_grid(
        (shape.n + warps_per_block - 1U) / warps_per_block,
        (kRows + kTile - 1U) / kTile, 1U);
    const auto output_blocks = static_cast<unsigned int>(
        (output_elements + threads - 1U) / threads);
    const auto launch_current = [&] {
        bf16_matvec_rows_kernel_probe<kTile><<<
            current_grid, threads, 0U, stream>>>(
            static_cast<float*>(device_output.value),
            static_cast<const float*>(device_input_f32.value),
            static_cast<const __nv_bfloat16*>(device_weights.value), kRows,
            shape.k, shape.n);
        check(cudaGetLastError(), "launch current BF16 row kernel");
    };
    const auto launch_round = [&] {
        round_float_bf16_probe<<<output_blocks, threads, 0U, stream>>>(
            static_cast<float*>(device_output.value), output_elements);
        check(cudaGetLastError(), "launch BF16 publication round");
    };
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    const auto launch_cublas = [&] {
        // Row-major X[M,K] * W[N,K]^T is column-major W^T[N,K] * X[K,M].
        check(cublasGemmEx(
                  handle, CUBLAS_OP_T, CUBLAS_OP_N,
                  static_cast<int>(shape.n), static_cast<int>(kRows),
                  static_cast<int>(shape.k), &alpha, device_weights.value,
                  CUDA_R_16BF, static_cast<int>(shape.k),
                  device_input_bf16.value, CUDA_R_16BF,
                  static_cast<int>(shape.k), &beta, device_output.value,
                  CUDA_R_32F, static_cast<int>(shape.n), CUBLAS_COMPUTE_32F,
                  CUBLAS_GEMM_DEFAULT_TENSOR_OP),
              "launch cuBLAS BF16 GEMM");
    };

    cudaEvent_t started = nullptr;
    cudaEvent_t finished = nullptr;
    check(cudaEventCreate(&started), "create probe start event");
    check(cudaEventCreate(&finished), "create probe finish event");
    for (std::uint32_t index = 0U; index < kWarmups; ++index) {
        launch_current();
        launch_cublas();
    }
    check(cudaStreamSynchronize(stream), "finish probe warmups");

    std::vector<float> current_samples;
    std::vector<float> current_rounded_samples;
    std::vector<float> cublas_samples;
    std::vector<float> cublas_rounded_samples;
    current_samples.reserve(kSamples);
    current_rounded_samples.reserve(kSamples);
    cublas_samples.reserve(kSamples);
    cublas_rounded_samples.reserve(kSamples);
    for (std::uint32_t sample = 0U; sample < kSamples; ++sample) {
        if ((sample & 1U) == 0U) {
            current_samples.push_back(time_once(
                started, finished, stream, launch_current));
            cublas_samples.push_back(time_once(
                started, finished, stream, launch_cublas));
        } else {
            cublas_samples.push_back(time_once(
                started, finished, stream, launch_cublas));
            current_samples.push_back(time_once(
                started, finished, stream, launch_current));
        }
        current_rounded_samples.push_back(time_once(
            started, finished, stream, [&] {
                launch_current();
                launch_round();
            }));
        cublas_rounded_samples.push_back(time_once(
            started, finished, stream, [&] {
                launch_cublas();
                launch_round();
            }));
    }

    std::vector<float> current_output(output_elements);
    std::vector<float> cublas_output(output_elements);
    std::vector<float> cublas_repeated_output(output_elements);
    launch_current();
    check(cudaMemcpyAsync(current_output.data(), device_output.value,
                          output_elements * sizeof(float),
                          cudaMemcpyDeviceToHost, stream),
          "download current output");
    check(cudaStreamSynchronize(stream), "finish current output download");
    launch_cublas();
    check(cudaMemcpyAsync(cublas_output.data(), device_output.value,
                          output_elements * sizeof(float),
                          cudaMemcpyDeviceToHost, stream),
          "download cuBLAS output");
    check(cudaStreamSynchronize(stream), "finish cuBLAS output download");
    launch_cublas();
    check(cudaMemcpyAsync(cublas_repeated_output.data(), device_output.value,
                          output_elements * sizeof(float),
                          cudaMemcpyDeviceToHost, stream),
          "download repeated cuBLAS output");
    check(cudaStreamSynchronize(stream),
          "finish repeated cuBLAS output download");

    check(cudaEventDestroy(started), "destroy probe start event");
    check(cudaEventDestroy(finished), "destroy probe finish event");

    Measurement result;
    result.shape = shape;
    result.current_milliseconds = median(std::move(current_samples));
    result.current_rounded_milliseconds =
        median(std::move(current_rounded_samples));
    result.cublas_milliseconds = median(std::move(cublas_samples));
    result.cublas_rounded_milliseconds =
        median(std::move(cublas_rounded_samples));
    result.difference = compare(current_output, cublas_output);
    for (std::uint64_t index = 0U; index < output_elements; ++index) {
        if (std::bit_cast<std::uint32_t>(cublas_output[index]) !=
            std::bit_cast<std::uint32_t>(cublas_repeated_output[index])) {
            ++result.cublas_repeat_fp32_mismatches;
        }
    }
    result.output_elements = output_elements;
    result.maximum_device_bytes =
        input_elements * (sizeof(float) + sizeof(std::uint16_t)) +
        weight_elements * sizeof(std::uint16_t) +
        output_elements * sizeof(float);
    return result;
}

void print_measurement(const Measurement& measurement, bool final) {
    const double flops = 2.0 * kRows * measurement.shape.n *
                         measurement.shape.k;
    const double current_gflops =
        flops / (static_cast<double>(measurement.current_milliseconds) * 1.0e6);
    const double cublas_gflops =
        flops / (static_cast<double>(measurement.cublas_milliseconds) * 1.0e6);
    const double bf16_step = measurement.difference.canonical_magnitude / 256.0;
    const auto magnitude_bits = float_to_bf16_bits(
        static_cast<float>(measurement.difference.canonical_magnitude));
    const double actual_bf16_spacing =
        static_cast<double>(bf16_bits_to_float(
            static_cast<std::uint16_t>(magnitude_bits + 1U))) -
        bf16_bits_to_float(magnitude_bits);
    std::cout << "    {\n"
              << "      \"name\": \"" << measurement.shape.name << "\",\n"
              << "      \"m\": " << kRows << ",\n"
              << "      \"n\": " << measurement.shape.n << ",\n"
              << "      \"k\": " << measurement.shape.k << ",\n"
              << "      \"flops\": " << static_cast<std::uint64_t>(flops) << ",\n"
              << "      \"current_kernel_median_ms\": "
              << measurement.current_milliseconds << ",\n"
              << "      \"current_kernel_plus_bf16_round_median_ms\": "
              << measurement.current_rounded_milliseconds << ",\n"
              << "      \"cublas_gemm_median_ms\": "
              << measurement.cublas_milliseconds << ",\n"
              << "      \"cublas_gemm_plus_bf16_round_median_ms\": "
              << measurement.cublas_rounded_milliseconds << ",\n"
              << "      \"current_gflops\": " << current_gflops << ",\n"
              << "      \"cublas_gflops\": " << cublas_gflops << ",\n"
              << "      \"cublas_over_current\": "
              << measurement.current_milliseconds /
                     measurement.cublas_milliseconds
              << ",\n"
              << "      \"current_fraction_of_rated_dense_bf16_tensor_peak\": "
              << current_gflops / kRtx3090DenseBf16TensorGflops << ",\n"
              << "      \"cublas_fraction_of_rated_dense_bf16_tensor_peak\": "
              << cublas_gflops / kRtx3090DenseBf16TensorGflops << ",\n"
              << "      \"maximum_absolute_difference_fp32\": "
              << measurement.difference.maximum_absolute << ",\n"
              << "      \"maximum_relative_difference_fp32_floor_1e_7\": "
              << measurement.difference.maximum_relative << ",\n"
              << "      \"mean_absolute_difference_fp32\": "
              << measurement.difference.mean_absolute << ",\n"
              << "      \"maximum_absolute_difference_after_bf16\": "
              << measurement.difference.maximum_bf16_absolute << ",\n"
              << "      \"maximum_relative_difference_after_bf16_floor_1e_7\": "
              << measurement.difference.maximum_bf16_relative << ",\n"
              << "      \"bf16_output_mismatches\": "
              << measurement.difference.bf16_mismatches << ",\n"
              << "      \"maximum_bf16_ulp_distance\": "
              << measurement.difference.maximum_bf16_ulp_distance << ",\n"
              << "      \"cublas_repeat_fp32_mismatches\": "
              << measurement.cublas_repeat_fp32_mismatches << ",\n"
              << "      \"output_elements\": "
              << measurement.output_elements << ",\n"
              << "      \"canonical_maximum_magnitude\": "
              << measurement.difference.canonical_magnitude << ",\n"
              << "      \"one_global_bf16_mantissa_step\": " << bf16_step << ",\n"
              << "      \"maximum_absolute_over_one_global_bf16_step\": "
              << (bf16_step == 0.0
                      ? std::numeric_limits<double>::infinity()
                      : measurement.difference.maximum_bf16_absolute /
                            bf16_step)
              << ",\n"
              << "      \"actual_bf16_spacing_at_canonical_maximum\": "
              << actual_bf16_spacing << ",\n"
              << "      \"maximum_absolute_over_actual_global_bf16_spacing\": "
              << measurement.difference.maximum_bf16_absolute /
                     actual_bf16_spacing
              << ",\n"
              << "      \"maximum_device_bytes\": "
              << measurement.maximum_device_bytes << "\n"
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
        check(cudaSetDevice(device), "select reference device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device),
              "query reference device");
        if (std::string_view(properties.name).find("RTX 3090") ==
            std::string_view::npos) {
            throw std::runtime_error(
                "rated-peak denominator is declared only for RTX 3090");
        }
        cudaStream_t stream = nullptr;
        cublasHandle_t handle = nullptr;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "create probe stream");
        check(cublasCreate(&handle), "create probe cuBLAS handle");
        check(cublasSetStream(handle, stream), "set probe cuBLAS stream");
        check(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH),
              "enable probe tensor-op math");

        std::vector<Measurement> measurements;
        measurements.reserve(std::size(kShapes));
        for (const auto& shape : kShapes) {
            measurements.push_back(run_shape(shape, handle, stream));
        }

        std::cout << std::fixed << std::setprecision(9)
                  << "{\n"
                  << "  \"device_index\": " << device << ",\n"
                  << "  \"device_name\": \"" << properties.name << "\",\n"
                  << "  \"rated_dense_bf16_tensor_gflops\": "
                  << kRtx3090DenseBf16TensorGflops << ",\n"
                  << "  \"warmups\": " << kWarmups << ",\n"
                  << "  \"samples\": " << kSamples << ",\n"
                  << "  \"relative_error_denominator_floor\": 0.000000100,\n"
                  << "  \"shapes\": [\n";
        for (std::size_t index = 0U; index < measurements.size(); ++index) {
            print_measurement(measurements[index],
                              index + 1U == measurements.size());
        }
        std::cout << "  ]\n}\n";

        check(cublasDestroy(handle), "destroy probe cuBLAS handle");
        check(cudaStreamDestroy(stream), "destroy probe stream");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
