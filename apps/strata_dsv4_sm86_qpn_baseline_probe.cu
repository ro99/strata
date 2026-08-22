// Clean C1 baseline for the RTX 3090/SM86 QPN campaign contract.
//
// This is deliberately not the target QPN implementation. It contains only:
//   1. the corrected 128 MiB ILP-4 cold-read ruler;
//   2. an exact replica of Strata's production E2M1/E8M0 group-32 matvec;
//   3. the conventional N=64 BF16 WMMA control whose decoded operands pass
//      through shared memory.
//
// Packed-weight bandwidth counts codes plus scales:
//   N*K/2 packed-code bytes + N*K/32 scale bytes = N*K*17/32 bytes.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {

constexpr std::uint32_t kGroupSize = 32U;
constexpr std::uint32_t kWarmups = 3U;
constexpr std::uint32_t kSamples = 11U;
constexpr std::uint32_t kOracleSamples = 4096U;
constexpr std::uint32_t kArenaReplicas = 8U;
constexpr std::size_t kScrubBytes = 256ULL * 1024ULL * 1024ULL;
constexpr std::size_t kRooflineStreamBytes = 128ULL * 1024ULL * 1024ULL;

constexpr std::uint32_t kTensorBlockN = 64U;
constexpr std::uint32_t kTensorBlockK = 128U;
constexpr std::uint32_t kTensorPadM = 16U;
constexpr std::uint32_t kTensorSlices = 4U;
constexpr std::uint32_t kTensorThreads = 128U;
constexpr std::uint32_t kTensorPitch = kTensorBlockK + 16U;
constexpr std::uint32_t kTensorSmemBytes =
    (kTensorBlockN * kTensorPitch + kTensorPadM * kTensorPitch) *
    sizeof(__nv_bfloat16);

struct Shape {
    const char* name;
    std::uint32_t n;
    std::uint32_t k;
};

constexpr Shape kShapes[] = {
    {"gate_up_w1", 2048U, 4096U},
    {"down_w2", 4096U, 2048U},
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

__host__ __device__ float bits_to_float(unsigned int bits) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(bits);
#else
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}

__host__ __device__ float fp4_e2m1_value(unsigned int encoded) {
    const unsigned int magnitude = encoded & 0x07U;
    const unsigned int exponent = magnitude >> 1U;
    const unsigned int mantissa = magnitude & 0x01U;
    const unsigned int normal =
        ((126U + exponent) << 23U) | (mantissa << 22U);
    const unsigned int subnormal = mantissa == 0U ? 0U : 0x3F00'0000U;
    const unsigned int bits = exponent == 0U ? subnormal : normal;
    const unsigned int sign =
        magnitude == 0U ? 0U : ((encoded & 0x08U) << 28U);
    return bits_to_float(bits | sign);
}

__host__ __device__ float e8m0_scale(unsigned char encoded) {
    return encoded == 0xFFU
               ? nanf("")
               : ldexpf(1.0F, static_cast<int>(encoded) - 127);
}

__device__ float bf16_round(float value) {
    const unsigned int bits = __float_as_uint(value);
    const unsigned int rounded = bits + 0x7FFFU + ((bits >> 16U) & 1U);
    return __uint_as_float(rounded & 0xFFFF'0000U);
}

__device__ float reduce_block(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFF'FFFFU, value, offset);
    }
    __shared__ float warp_sums[8];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8U ? warp_sums[lane] : 0.0F;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xFFFF'FFFFU, value, offset);
        }
    }
    return value;
}

__global__ void scrub_kernel(unsigned char* data, std::size_t bytes) {
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) *
                                 blockDim.x + threadIdx.x;
         index < bytes; index += stride) {
        data[index] = static_cast<unsigned char>(index);
    }
}

__global__ void read_roofline_kernel(const uint4* data,
                                     std::uint64_t vectors, float* sink) {
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    unsigned int accumulator_a = 0U;
    unsigned int accumulator_b = 0U;
    unsigned int accumulator_c = 0U;
    unsigned int accumulator_d = 0U;
    for (; index + 3U * stride < vectors; index += 4U * stride) {
        const uint4 value_a = data[index];
        const uint4 value_b = data[index + stride];
        const uint4 value_c = data[index + 2U * stride];
        const uint4 value_d = data[index + 3U * stride];
        accumulator_a ^= value_a.x ^ value_a.y ^ value_a.z ^ value_a.w;
        accumulator_b ^= value_b.x ^ value_b.y ^ value_b.z ^ value_b.w;
        accumulator_c ^= value_c.x ^ value_c.y ^ value_c.z ^ value_c.w;
        accumulator_d ^= value_d.x ^ value_d.y ^ value_d.z ^ value_d.w;
    }
    for (; index < vectors; index += stride) {
        const uint4 value = data[index];
        accumulator_a ^= value.x ^ value.y ^ value.z ^ value.w;
    }
    if ((accumulator_a ^ accumulator_b ^ accumulator_c ^ accumulator_d) ==
        0xDEAD'BEEFU) {
        sink[threadIdx.x] = 1.0F;
    }
}

__global__ void production_matvec_kernel(
    float* output, const float* hidden, const unsigned char* weights,
    const unsigned char* scales, std::uint32_t rows, std::uint32_t columns) {
    const std::uint64_t output_row = blockIdx.x;
    if (output_row >= rows) return;
    const std::uint64_t packed_base = output_row * (columns / 2U);
    const std::uint64_t scale_base = output_row * (columns / kGroupSize);
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint32_t scale_columns = columns / kGroupSize;
    float sum = 0.0F;
    for (std::uint64_t group = warp; group < scale_columns; group += 8U) {
        float scale = lane == 0U
                          ? e8m0_scale(scales[scale_base + group])
                          : 0.0F;
        scale = __shfl_sync(0xFFFF'FFFFU, scale, 0);
        const std::uint64_t column = group * kGroupSize + lane;
        const unsigned char packed = weights[packed_base + column / 2U];
        const unsigned int encoded =
            column % 2U == 0U ? packed & 0x0FU : packed >> 4U;
        sum += hidden[column] * fp4_e2m1_value(encoded) * scale;
    }
    sum = reduce_block(sum);
    if (threadIdx.x == 0U) output[output_row] = bf16_round(sum);
}

constexpr unsigned int kPackSign = 0x8000'8000U;
constexpr unsigned int kPackEm = 0x01C0'01C0U;

__device__ unsigned int pack_decode_pair(unsigned int q, unsigned int p) {
    const unsigned int sign_shift = 12U - 4U * p;
    const int payload_shift = 6 - static_cast<int>(4U * p);
    const unsigned int payload =
        payload_shift >= 0
            ? (q << payload_shift) & kPackEm
            : (q >> static_cast<unsigned int>(-payload_shift)) & kPackEm;
    return ((q << sign_shift) & kPackSign) | payload;
}

__device__ unsigned int pack_decode_scaled(unsigned int q, unsigned int p,
                                           unsigned int encoded_scale) {
    const unsigned int bits = pack_decode_pair(q, p);
    const unsigned int delta =
        (((encoded_scale - 1U) << 7U) * 0x0001'0001U);
    const unsigned int magnitudes = bits & 0x7FFF'7FFFU;
    const unsigned int magnitude_indicators =
        (magnitudes + 0x7FFF'7FFFU) & 0x8000'8000U;
    const unsigned int magnitude_nonzero =
        magnitude_indicators |
        (magnitude_indicators - (magnitude_indicators >> 15U));
    const unsigned int exponent_indicators =
        ((bits & 0x7F80'7F80U) + 0x7F80'7F80U) & 0x8000'8000U;
    const unsigned int exponent_zero =
        ~(exponent_indicators |
          (exponent_indicators - (exponent_indicators >> 15U)));
    const unsigned int sign_only = bits & 0x8000'8000U;
    unsigned int scaled =
        (bits & ~exponent_zero) | (sign_only & exponent_zero);
    scaled += delta;
    return scaled & magnitude_nonzero;
}

__global__ void conventional_wmma_n64_kernel(
    float* partials, const float* hidden, const unsigned char* weights,
    const unsigned char* scales, std::uint32_t rows, std::uint32_t columns) {
    using namespace nvcuda;
    extern __shared__ char tensor_smem[];
    __nv_bfloat16* weight_tile =
        reinterpret_cast<__nv_bfloat16*>(tensor_smem);
    __nv_bfloat16* activation_tile =
        weight_tile + kTensorBlockN * kTensorPitch;

    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t warp = tid >> 5U;
    const std::uint32_t warp_n = warp % (kTensorBlockN / 16U);
    const std::uint32_t n_begin = blockIdx.x * kTensorBlockN;
    const std::uint32_t slice = blockIdx.y;
    const std::uint32_t tiles_k = columns / kTensorBlockK;
    const std::uint32_t tile_begin = (tiles_k * slice) / kTensorSlices;
    const std::uint32_t tile_end =
        (tiles_k * (slice + 1U)) / kTensorSlices;

    constexpr std::uint32_t kCodeSegments =
        kTensorBlockN * (kTensorBlockK / 16U) / kTensorThreads;
    constexpr std::uint32_t kActivationSegments =
        kTensorPadM * (kTensorBlockK / 4U) / kTensorThreads;
    static_assert(kCodeSegments * kTensorThreads ==
                  kTensorBlockN * (kTensorBlockK / 16U));
    static_assert(kActivationSegments * kTensorThreads ==
                  kTensorPadM * (kTensorBlockK / 4U));

    uint2 staged_codes[kCodeSegments];
    unsigned char staged_scales[kCodeSegments];
    float4 staged_activations[kActivationSegments];

    const auto load_stage = [&](std::uint32_t k_begin) {
#pragma unroll
        for (std::uint32_t i = 0U; i < kCodeSegments; ++i) {
            const std::uint32_t index = tid + i * kTensorThreads;
            const std::uint32_t n = index / (kTensorBlockK / 16U);
            const std::uint32_t segment = index % (kTensorBlockK / 16U);
            staged_codes[i] = *reinterpret_cast<const uint2*>(
                weights +
                static_cast<std::uint64_t>(n_begin + n) * (columns / 2U) +
                (k_begin / 2U) + segment * 8U);
            staged_scales[i] =
                scales[static_cast<std::uint64_t>(n_begin + n) *
                           (columns / kGroupSize) +
                       ((k_begin + segment * 16U) / kGroupSize)];
        }
#pragma unroll
        for (std::uint32_t i = 0U; i < kActivationSegments; ++i) {
            const std::uint32_t index = tid + i * kTensorThreads;
            const std::uint32_t row = index / (kTensorBlockK / 4U);
            const std::uint32_t vector = index % (kTensorBlockK / 4U);
            staged_activations[i] =
                row == 0U
                    ? *reinterpret_cast<const float4*>(hidden + k_begin +
                                                       vector * 4U)
                    : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        }
    };

    const auto store_stage = [&]() {
#pragma unroll
        for (std::uint32_t i = 0U; i < kCodeSegments; ++i) {
            const std::uint32_t index = tid + i * kTensorThreads;
            const std::uint32_t n = index / (kTensorBlockK / 16U);
            const std::uint32_t segment = index % (kTensorBlockK / 16U);
            __nv_bfloat16* row =
                weight_tile + n * kTensorPitch + segment * 16U;
            const unsigned int words[2] = {staged_codes[i].x,
                                           staged_codes[i].y};
#pragma unroll
            for (std::uint32_t word = 0U; word < 2U; ++word) {
                const unsigned int codes = words[word];
#pragma unroll
                for (std::uint32_t pair = 0U; pair < 4U; ++pair) {
                    const unsigned int decoded = pack_decode_scaled(
                        codes, pair,
                        static_cast<unsigned int>(staged_scales[i]));
                    row[word * 8U + pair] = __ushort_as_bfloat16(
                        static_cast<unsigned short>(decoded & 0xFFFFU));
                    row[word * 8U + pair + 4U] = __ushort_as_bfloat16(
                        static_cast<unsigned short>(decoded >> 16U));
                }
            }
        }
#pragma unroll
        for (std::uint32_t i = 0U; i < kActivationSegments; ++i) {
            const std::uint32_t index = tid + i * kTensorThreads;
            const std::uint32_t row = index / (kTensorBlockK / 4U);
            const std::uint32_t vector = index % (kTensorBlockK / 4U);
            __nv_bfloat16* destination =
                activation_tile + row * kTensorPitch + vector * 4U;
            destination[0] =
                __float2bfloat16_rn(staged_activations[i].x);
            destination[1] =
                __float2bfloat16_rn(staged_activations[i].y);
            destination[2] =
                __float2bfloat16_rn(staged_activations[i].z);
            destination[3] =
                __float2bfloat16_rn(staged_activations[i].w);
        }
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0F);

    if (tile_begin < tile_end) load_stage(tile_begin * kTensorBlockK);
    for (std::uint32_t tile_k = tile_begin; tile_k < tile_end; ++tile_k) {
        const std::uint32_t k_begin = tile_k * kTensorBlockK;
        __syncthreads();
        store_stage();
        __syncthreads();
        if (tile_k + 1U < tile_end) load_stage(k_begin + kTensorBlockK);

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                       wmma::row_major>
            a_fragments[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                       wmma::col_major>
            b_fragments[2];
        wmma::load_matrix_sync(
            a_fragments[0],
            weight_tile + warp_n * 16U * kTensorPitch, kTensorPitch);
        wmma::load_matrix_sync(b_fragments[0], activation_tile,
                               kTensorPitch);
#pragma unroll
        for (std::uint32_t k_tile = 0U;
             k_tile < kTensorBlockK / 16U; ++k_tile) {
            const std::uint32_t current = k_tile & 1U;
            const std::uint32_t next = current ^ 1U;
            if (k_tile + 1U < kTensorBlockK / 16U) {
                wmma::load_matrix_sync(
                    a_fragments[next],
                    weight_tile + warp_n * 16U * kTensorPitch +
                        (k_tile + 1U) * 16U,
                    kTensorPitch);
                wmma::load_matrix_sync(
                    b_fragments[next],
                    activation_tile + (k_tile + 1U) * 16U, kTensorPitch);
            }
            wmma::mma_sync(accumulator, a_fragments[current],
                           b_fragments[current], accumulator);
        }
    }

    __syncthreads();
    float* accumulator_smem =
        reinterpret_cast<float*>(tensor_smem) + warp * 256U;
    wmma::store_matrix_sync(accumulator_smem, accumulator, 16U,
                            wmma::mem_row_major);
    __syncwarp();
    const std::uint64_t slice_base =
        static_cast<std::uint64_t>(slice) * kTensorPadM * rows;
    for (std::uint32_t element = (tid & 31U); element < 256U;
         element += 32U) {
        const std::uint32_t local_n = element >> 4U;
        const std::uint32_t local_m = element & 15U;
        const std::uint32_t global_n =
            n_begin + warp_n * 16U + local_n;
        partials[slice_base +
                 static_cast<std::uint64_t>(local_m) * rows + global_n] =
            accumulator_smem[element];
    }
}

__global__ void tensor_reduce_kernel(const float* partials, float* output,
                                     std::uint32_t rows) {
    const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= rows) return;
    float sum = 0.0F;
#pragma unroll
    for (std::uint32_t slice = 0U; slice < kTensorSlices; ++slice) {
        sum += partials[static_cast<std::uint64_t>(slice) *
                        kTensorPadM * rows + column];
    }
    output[column] = bf16_round(sum);
}

class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        check(cudaMalloc(&value_, bytes), "allocate SM86 baseline buffer");
    }

    ~DeviceBuffer() {
        if (value_ != nullptr) static_cast<void>(cudaFree(value_));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    [[nodiscard]] void* get() const noexcept { return value_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

  private:
    void* value_{nullptr};
    std::size_t bytes_{0U};
};

template <typename Launch>
float time_once(cudaEvent_t start, cudaEvent_t finish, cudaStream_t stream,
                Launch&& launch) {
    check(cudaEventRecord(start, stream), "record baseline start");
    launch();
    check(cudaEventRecord(finish, stream), "record baseline finish");
    check(cudaEventSynchronize(finish), "synchronize baseline sample");
    float milliseconds = 0.0F;
    check(cudaEventElapsedTime(&milliseconds, start, finish),
          "measure baseline sample");
    return milliseconds;
}

float median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}

float bf16_round_host(float value) {
    std::uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t rounded = bits + 0x7FFFU + ((bits >> 16U) & 1U);
    const std::uint32_t truncated = rounded & 0xFFFF'0000U;
    std::memcpy(&value, &truncated, sizeof(value));
    return value;
}

struct ErrorMetrics {
    double maximum_absolute{0.0};
    double maximum_relative{0.0};
};

struct ArmResult {
    float hot_milliseconds{0.0F};
    float cold_milliseconds{0.0F};
    std::uint64_t bytes{0U};
    ErrorMetrics oracle_error;
    double versus_production_max_abs{0.0};
};

struct Measurement {
    Shape shape{};
    std::uint64_t code_bytes{0U};
    std::uint64_t scale_bytes{0U};
    std::uint64_t matrix_bytes{0U};
    std::uint64_t allocated_bytes{0U};
    ArmResult roofline;
    ArmResult production;
    ArmResult conventional_wmma_n64;
};

double oracle_dot(const std::vector<float>& hidden,
                  const std::vector<unsigned char>& weights,
                  const std::vector<unsigned char>& scales,
                  std::uint32_t columns, std::uint32_t row) {
    double sum = 0.0;
    const std::uint64_t packed_base =
        static_cast<std::uint64_t>(row) * (columns / 2U);
    const std::uint64_t scale_base =
        static_cast<std::uint64_t>(row) * (columns / kGroupSize);
    for (std::uint32_t column = 0U; column < columns; ++column) {
        const unsigned char packed = weights[packed_base + column / 2U];
        const unsigned int encoded =
            column % 2U == 0U ? packed & 0x0FU : packed >> 4U;
        sum += static_cast<double>(hidden[column]) *
               static_cast<double>(fp4_e2m1_value(encoded)) *
               static_cast<double>(e8m0_scale(
                   scales[scale_base + column / kGroupSize]));
    }
    return sum;
}

Measurement run_shape(const Shape& shape, cudaStream_t stream,
                      const cudaDeviceProp& properties) {
    const std::uint64_t code_bytes =
        static_cast<std::uint64_t>(shape.n) * (shape.k / 2U);
    const std::uint64_t scale_bytes =
        static_cast<std::uint64_t>(shape.n) * (shape.k / kGroupSize);
    const std::uint64_t matrix_bytes = code_bytes + scale_bytes;
    const std::uint64_t arena_bytes = matrix_bytes * kArenaReplicas;
    const std::uint64_t hidden_bytes =
        static_cast<std::uint64_t>(shape.k) * sizeof(float);
    const std::uint64_t output_bytes =
        static_cast<std::uint64_t>(shape.n) * sizeof(float);
    const std::uint64_t partial_bytes =
        static_cast<std::uint64_t>(kTensorSlices) * kTensorPadM * shape.n *
        sizeof(float);

    std::vector<unsigned char> weights(code_bytes);
    std::vector<unsigned char> scales(scale_bytes);
    std::vector<float> hidden(shape.k);
    std::uint32_t random_state =
        0xA341'316CU ^ shape.n ^ (shape.k << 8U);
    for (auto& value : weights) {
        value = static_cast<unsigned char>(xorshift(random_state));
    }
    for (auto& value : scales) {
        value = static_cast<unsigned char>(123U +
                                           xorshift(random_state) % 9U);
    }
    for (auto& value : hidden) {
        const float uniform =
            static_cast<float>(xorshift(random_state) % 4096U) / 1024.0F -
            2.0F;
        value = bf16_round_host(uniform);
    }

    DeviceBuffer device_arena(arena_bytes);
    DeviceBuffer device_roofline(kRooflineStreamBytes);
    DeviceBuffer device_scrub(kScrubBytes);
    DeviceBuffer device_hidden(hidden_bytes);
    DeviceBuffer device_production(output_bytes);
    DeviceBuffer device_candidate(output_bytes);
    DeviceBuffer device_partials(partial_bytes);

    std::vector<unsigned char> arena_host(arena_bytes);
    for (std::uint32_t replica = 0U; replica < kArenaReplicas; ++replica) {
        std::memcpy(arena_host.data() + matrix_bytes * replica,
                    weights.data(), code_bytes);
        std::memcpy(arena_host.data() + matrix_bytes * replica + code_bytes,
                    scales.data(), scale_bytes);
    }
    check(cudaMemcpyAsync(device_arena.get(), arena_host.data(), arena_bytes,
                          cudaMemcpyHostToDevice, stream),
          "upload baseline arena");
    check(cudaMemcpyAsync(device_hidden.get(), hidden.data(), hidden_bytes,
                          cudaMemcpyHostToDevice, stream),
          "upload baseline activations");
    check(cudaMemsetAsync(device_roofline.get(), 0xA5,
                          device_roofline.bytes(), stream),
          "initialize roofline stream");
    check(cudaStreamSynchronize(stream), "finish baseline uploads");

    const auto arena_matrix = [&](std::uint32_t replica) {
        return static_cast<unsigned char*>(device_arena.get()) +
               matrix_bytes * replica;
    };
    const std::uint32_t roofline_blocks =
        static_cast<std::uint32_t>(properties.multiProcessorCount) * 8U;
    const auto launch_roofline = [&] {
        read_roofline_kernel<<<roofline_blocks, 256U, 0U, stream>>>(
            static_cast<const uint4*>(device_roofline.get()),
            kRooflineStreamBytes / sizeof(uint4),
            static_cast<float*>(device_candidate.get()));
        check(cudaGetLastError(), "launch roofline");
    };
    const auto launch_production_at = [&](std::uint32_t replica) {
        production_matvec_kernel<<<shape.n, 256U, 0U, stream>>>(
            static_cast<float*>(device_production.get()),
            static_cast<const float*>(device_hidden.get()),
            arena_matrix(replica), arena_matrix(replica) + code_bytes,
            shape.n, shape.k);
        check(cudaGetLastError(), "launch production control");
    };
    const auto launch_wmma_at = [&](std::uint32_t replica) {
        conventional_wmma_n64_kernel<<<
            dim3{shape.n / kTensorBlockN, kTensorSlices}, kTensorThreads,
            kTensorSmemBytes, stream>>>(
            static_cast<float*>(device_partials.get()),
            static_cast<const float*>(device_hidden.get()),
            arena_matrix(replica), arena_matrix(replica) + code_bytes,
            shape.n, shape.k);
        check(cudaGetLastError(), "launch conventional WMMA control");
        tensor_reduce_kernel<<<(shape.n + 255U) / 256U, 256U, 0U, stream>>>(
            static_cast<const float*>(device_partials.get()),
            static_cast<float*>(device_candidate.get()), shape.n);
        check(cudaGetLastError(), "launch conventional WMMA reduction");
    };
    const auto scrub = [&] {
        scrub_kernel<<<static_cast<std::uint32_t>(
                           kScrubBytes / (256U * 16U)),
                       256U, 0U, stream>>>(
            static_cast<unsigned char*>(device_scrub.get()), kScrubBytes);
        check(cudaGetLastError(), "launch L2 scrub");
    };

    cudaEvent_t started = nullptr;
    cudaEvent_t finished = nullptr;
    check(cudaEventCreate(&started), "create baseline start event");
    check(cudaEventCreate(&finished), "create baseline finish event");

    for (std::uint32_t warmup = 0U; warmup < kWarmups; ++warmup) {
        launch_roofline();
        launch_production_at(0U);
        launch_wmma_at(0U);
    }
    check(cudaStreamSynchronize(stream), "finish baseline warmups");

    std::vector<float> hot_roofline;
    std::vector<float> hot_production;
    std::vector<float> hot_wmma;
    std::vector<float> cold_roofline;
    std::vector<float> cold_production;
    std::vector<float> cold_wmma;
    hot_roofline.reserve(kSamples);
    hot_production.reserve(kSamples);
    hot_wmma.reserve(kSamples);
    cold_roofline.reserve(kSamples);
    cold_production.reserve(kSamples);
    cold_wmma.reserve(kSamples);

    for (std::uint32_t sample = 0U; sample < kSamples; ++sample) {
        hot_roofline.push_back(
            time_once(started, finished, stream, launch_roofline));
        hot_production.push_back(time_once(
            started, finished, stream,
            [&] { launch_production_at(0U); }));
        hot_wmma.push_back(time_once(started, finished, stream,
                                     [&] { launch_wmma_at(0U); }));

        const std::uint32_t replica = sample % kArenaReplicas;
        scrub();
        cold_roofline.push_back(
            time_once(started, finished, stream, launch_roofline));
        scrub();
        cold_production.push_back(time_once(
            started, finished, stream,
            [&] { launch_production_at(replica); }));
        scrub();
        cold_wmma.push_back(time_once(
            started, finished, stream,
            [&] { launch_wmma_at(replica); }));
    }

    Measurement result;
    result.shape = shape;
    result.code_bytes = code_bytes;
    result.scale_bytes = scale_bytes;
    result.matrix_bytes = matrix_bytes;
    result.allocated_bytes = arena_bytes + kRooflineStreamBytes +
                             kScrubBytes + hidden_bytes + 2U * output_bytes +
                             partial_bytes;
    result.roofline.hot_milliseconds = median(std::move(hot_roofline));
    result.roofline.cold_milliseconds = median(std::move(cold_roofline));
    result.roofline.bytes = kRooflineStreamBytes;
    result.production.hot_milliseconds = median(std::move(hot_production));
    result.production.cold_milliseconds = median(std::move(cold_production));
    result.production.bytes = matrix_bytes;
    result.conventional_wmma_n64.hot_milliseconds =
        median(std::move(hot_wmma));
    result.conventional_wmma_n64.cold_milliseconds =
        median(std::move(cold_wmma));
    result.conventional_wmma_n64.bytes = matrix_bytes;

    std::vector<float> production_output(shape.n);
    std::vector<float> candidate_output(shape.n);
    launch_production_at(0U);
    check(cudaMemcpyAsync(production_output.data(), device_production.get(),
                          output_bytes, cudaMemcpyDeviceToHost, stream),
          "download production output");
    launch_wmma_at(0U);
    check(cudaMemcpyAsync(candidate_output.data(), device_candidate.get(),
                          output_bytes, cudaMemcpyDeviceToHost, stream),
          "download conventional WMMA output");
    check(cudaStreamSynchronize(stream), "finish correctness downloads");

    const std::uint32_t oracle_samples =
        std::min(kOracleSamples, shape.n);
    std::unordered_set<std::uint32_t> seen;
    std::vector<std::uint32_t> indices;
    seen.reserve(oracle_samples * 2U);
    indices.reserve(oracle_samples);
    while (indices.size() < oracle_samples) {
        const std::uint32_t index = xorshift(random_state) % shape.n;
        if (seen.insert(index).second) indices.push_back(index);
    }
    for (const std::uint32_t index : indices) {
        const double reference =
            oracle_dot(hidden, weights, scales, shape.k, index);
        const double production_absolute =
            std::abs(static_cast<double>(production_output[index]) -
                     reference);
        const double candidate_absolute =
            std::abs(static_cast<double>(candidate_output[index]) -
                     reference);
        result.production.oracle_error.maximum_absolute = std::max(
            result.production.oracle_error.maximum_absolute,
            production_absolute);
        result.production.oracle_error.maximum_relative = std::max(
            result.production.oracle_error.maximum_relative,
            production_absolute / std::max(std::abs(reference), 1.0e-9));
        result.conventional_wmma_n64.oracle_error.maximum_absolute = std::max(
            result.conventional_wmma_n64.oracle_error.maximum_absolute,
            candidate_absolute);
        result.conventional_wmma_n64.oracle_error.maximum_relative = std::max(
            result.conventional_wmma_n64.oracle_error.maximum_relative,
            candidate_absolute / std::max(std::abs(reference), 1.0e-9));
    }
    for (std::uint32_t index = 0U; index < shape.n; ++index) {
        result.conventional_wmma_n64.versus_production_max_abs = std::max(
            result.conventional_wmma_n64.versus_production_max_abs,
            std::abs(static_cast<double>(candidate_output[index]) -
                     static_cast<double>(production_output[index])));
    }

    check(cudaEventDestroy(started), "destroy baseline start event");
    check(cudaEventDestroy(finished), "destroy baseline finish event");
    return result;
}

double gbps(std::uint64_t bytes, float milliseconds) {
    return static_cast<double>(bytes) /
           (static_cast<double>(milliseconds) * 1.0e6);
}

void print_arm(std::ostream& output, const char* name, const ArmResult& arm,
               bool final) {
    output << "      \"" << name << "\": {\n"
           << "        \"hot_ms\": " << arm.hot_milliseconds << ",\n"
           << "        \"hot_gb_s\": "
           << gbps(arm.bytes, arm.hot_milliseconds) << ",\n"
           << "        \"cold_ms\": " << arm.cold_milliseconds << ",\n"
           << "        \"cold_gb_s\": "
           << gbps(arm.bytes, arm.cold_milliseconds) << ",\n"
           << "        \"oracle_max_abs\": "
           << arm.oracle_error.maximum_absolute << ",\n"
           << "        \"oracle_max_rel_floor_1e_9\": "
           << arm.oracle_error.maximum_relative << ",\n"
           << "        \"versus_production_max_abs\": "
           << arm.versus_production_max_abs << "\n"
           << "      }" << (final ? "\n" : ",\n");
}

void print_result(std::ostream& output, int device,
                  const cudaDeviceProp& properties,
                  const std::vector<Measurement>& measurements) {
    output << std::fixed << std::setprecision(6)
           << "{\n"
           << "  \"device_index\": " << device << ",\n"
           << "  \"device_name\": \"" << properties.name << "\",\n"
           << "  \"device_capability\": \"" << properties.major << "."
           << properties.minor << "\",\n"
           << "  \"multiprocessors\": " << properties.multiProcessorCount
           << ",\n"
           << "  \"batch\": 1,\n"
           << "  \"group_size\": " << kGroupSize << ",\n"
           << "  \"warmups\": " << kWarmups << ",\n"
           << "  \"samples\": " << kSamples << ",\n"
           << "  \"arena_replicas\": " << kArenaReplicas << ",\n"
           << "  \"byte_formula\": \"N*K/2 code bytes + N*K/32 scale bytes\",\n"
           << "  \"shapes\": [\n";
    for (std::size_t index = 0U; index < measurements.size(); ++index) {
        const Measurement& measurement = measurements[index];
        output << "    {\n"
               << "      \"name\": \"" << measurement.shape.name
               << "\",\n"
               << "      \"n\": " << measurement.shape.n << ",\n"
               << "      \"k\": " << measurement.shape.k << ",\n"
               << "      \"code_bytes\": " << measurement.code_bytes
               << ",\n"
               << "      \"scale_bytes\": " << measurement.scale_bytes
               << ",\n"
               << "      \"matrix_bytes\": " << measurement.matrix_bytes
               << ",\n"
               << "      \"allocated_bytes\": "
               << measurement.allocated_bytes << ",\n";
        print_arm(output, "roofline", measurement.roofline, false);
        print_arm(output, "production", measurement.production, false);
        print_arm(output, "conventional_wmma_n64",
                  measurement.conventional_wmma_n64, true);
        output << "    }"
               << (index + 1U == measurements.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        int device = 0;
        std::string output_path;
        for (int index = 1; index < argc; ++index) {
            const std::string_view flag = argv[index];
            if (flag == "--device" && index + 1 < argc) {
                device = std::stoi(argv[++index]);
            } else if (flag == "--output" && index + 1 < argc) {
                output_path = argv[++index];
            } else {
                std::cerr << "usage: " << argv[0]
                          << " [--device INDEX] [--output PATH]\n";
                return EXIT_FAILURE;
            }
        }

        check(cudaSetDevice(device), "select baseline device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device),
              "query baseline device");
        if (properties.major != 8 || properties.minor != 6) {
            throw std::runtime_error(
                "C1 baseline requires an SM86 device, selected " +
                std::to_string(properties.major) + "." +
                std::to_string(properties.minor));
        }

        cudaStream_t stream = nullptr;
        check(cudaStreamCreate(&stream), "create baseline stream");
        std::vector<Measurement> measurements;
        measurements.reserve(2U);
        for (const Shape& shape : kShapes) {
            measurements.push_back(run_shape(shape, stream, properties));
        }

        if (output_path.empty()) {
            print_result(std::cout, device, properties, measurements);
        } else {
            std::ofstream output(output_path);
            if (!output) {
                throw std::runtime_error("cannot open output path: " +
                                         output_path);
            }
            print_result(output, device, properties, measurements);
        }
        check(cudaStreamDestroy(stream), "destroy baseline stream");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
