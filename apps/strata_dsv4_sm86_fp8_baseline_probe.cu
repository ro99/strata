// F8-0 baseline for the RTX 3090/SM86 QPN8 campaign.
//
// This is deliberately not a QPN8 implementation and is never dispatched by
// production code. It measures, at one requested (M,N,K) operating point:
//   * the corrected 128 MiB ILP-4 cold-read ruler;
//   * Strata's scalar E4M3/E8M0 block-128 kernel with a 16-bit activation
//     boundary (the W8A16 baseline primitive); and
//   * the existing 48 KiB shared-memory W8A8-style WMMA page control.
//
// Useful compressed bytes are exact checkpoint bytes:
//   N*K E4M3 code bytes + ceil(N/128)*ceil(K/128) E8M0 scale bytes.

// The fixture uses E4M3-exact activation values and unit E8M0 activation
// scales so both controls consume the same mathematical activation. The
// scalar arm receives them through the runtime's FP32 carrier after BF16
// rounding; the tensor arm receives their compact E4M3 representation.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint32_t kBlockM = 64U;
constexpr std::uint32_t kBlockN = 128U;
constexpr std::uint32_t kBlockK = 128U;
constexpr std::uint32_t kThreads = 256U;
constexpr std::uint32_t kWarmups = 3U;
constexpr std::uint32_t kSamples = 11U;
constexpr std::uint32_t kOracleSamples = 512U;
constexpr std::size_t kScrubBytes = 256ULL << 20U;
constexpr std::size_t kRooflineBytes = 128ULL << 20U;
constexpr std::size_t kMemoryCeiling = 512ULL << 20U;
constexpr std::size_t kArenaBudget = 112ULL << 20U;

struct Shape {
    const char* name;
    std::uint32_t n;
    std::uint32_t k;
    const char* regions;
    bool active;
};

constexpr Shape kShapes[] = {
    {"wq_a", 1024U, 4096U, "layers[*].attn.wq_a;mtp[*].attn.wq_a", true},
    {"wq_b", 32768U, 1024U, "layers[*].attn.wq_b;mtp[*].attn.wq_b", true},
    {"wkv", 512U, 4096U, "layers[*].attn.wkv;mtp[*].attn.wkv", true},
    {"wo_a", 8192U, 4096U, "layers[*].attn.wo_a;mtp[*].attn.wo_a", true},
    {"wo_b", 4096U, 8192U, "layers[*].attn.wo_b;mtp[*].attn.wo_b", true},
    {"shared_w1_w3", 2048U, 4096U,
     "layers[*].ffn.shared_experts.w1,w3;mtp[*].ffn.shared_experts.w1,w3", true},
    {"shared_w2", 4096U, 2048U,
     "layers[*].ffn.shared_experts.w2;mtp[*].ffn.shared_experts.w2", true},
    {"indexer_wq_b", 8192U, 1024U, "layers[ratio4].attn.indexer.wq_b", true},
    {"mtp_main_proj", 4096U, 12288U, "mtp[*].main_proj", false},
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

__host__ __device__ float e4m3_value(std::uint8_t encoded) {
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

__host__ __device__ float e8m0_scale(std::uint8_t encoded) {
    return encoded == 0xFFU
               ? nanf("")
               : ldexpf(1.0F, static_cast<int>(encoded) - 127);
}

__host__ __device__ float bf16_round(float value) {
#ifdef __CUDA_ARCH__
    const unsigned int bits = __float_as_uint(value);
    const unsigned int rounded = bits + 0x7FFFU + ((bits >> 16U) & 1U);
    return __uint_as_float(rounded & 0xFFFF'0000U);
#else
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return std::bit_cast<float>(bits & 0xFFFF'0000U);
#endif
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

__device__ std::uint8_t encode_e4m3_value(float value) {
    const std::uint8_t sign = signbit(value) ? 0x80U : 0U;
    const float magnitude = fabsf(value);
    if (magnitude == 0.0F) return sign;
    if (magnitude < 0.015625F) {
        const auto mantissa = static_cast<unsigned int>(
            rintf(ldexpf(magnitude, 9)));
        return static_cast<std::uint8_t>(sign | min(mantissa, 7U));
    }
    int exponent = 0;
    const float fraction = frexpf(magnitude, &exponent);
    const auto encoded_exponent = static_cast<unsigned int>(exponent + 6);
    const auto mantissa = static_cast<unsigned int>(
        rintf((fraction * 2.0F - 1.0F) * 8.0F));
    const auto maximum_mantissa = encoded_exponent == 15U ? 6U : 7U;
    return static_cast<std::uint8_t>(
        sign | (min(encoded_exponent, 15U) << 3U) |
        min(mantissa, maximum_mantissa));
}

__global__ void f8_quantize_activation_bytes_kernel(
    std::uint8_t* values, std::uint8_t* scales, const float* source,
    std::uint32_t columns, std::uint32_t rows) {
    const std::uint32_t row = blockIdx.y;
    const std::uint64_t group_begin =
        static_cast<std::uint64_t>(blockIdx.x) * kBlockK;
    if (row >= rows || group_begin >= columns) return;
    const std::uint64_t index = group_begin + threadIdx.x;
    const float value = index < columns
                            ? source[static_cast<std::uint64_t>(row) * columns +
                                     index]
                            : 0.0F;
    __shared__ float maximum[kBlockK];
    maximum[threadIdx.x] = fabsf(value);
    __syncthreads();
    for (unsigned int stride = kBlockK / 2U; stride != 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            maximum[threadIdx.x] = fmaxf(maximum[threadIdx.x],
                                         maximum[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    int scale_exponent = 0;
    float scale = 1.0F;
    if (maximum[0] > 0.0F) {
        scale_exponent = static_cast<int>(
            ceilf(log2f(maximum[0] / 448.0F)));
        scale = exp2f(static_cast<float>(scale_exponent));
    }
    if (threadIdx.x == 0U) {
        scales[static_cast<std::uint64_t>(row) * gridDim.x + blockIdx.x] =
            static_cast<std::uint8_t>(scale_exponent + 127);
    }
    if (index < columns) {
        values[static_cast<std::uint64_t>(row) * columns + index] =
            encode_e4m3_value(quantize_e4m3_value(value / scale));
    }
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

__global__ void f8_scrub_kernel(std::uint8_t* data, std::size_t bytes) {
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) *
                                 blockDim.x + threadIdx.x;
         index < bytes; index += stride) {
        data[index] = static_cast<std::uint8_t>(index);
    }
}

__global__ void f8_read_roofline_kernel(const uint4* data,
                                        std::uint64_t vectors,
                                        unsigned int* sink) {
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    unsigned int a = 0U;
    unsigned int b = 0U;
    unsigned int c = 0U;
    unsigned int d = 0U;
    for (; index + 3U * stride < vectors; index += 4U * stride) {
        const uint4 va = data[index];
        const uint4 vb = data[index + stride];
        const uint4 vc = data[index + 2U * stride];
        const uint4 vd = data[index + 3U * stride];
        a ^= va.x ^ va.y ^ va.z ^ va.w;
        b ^= vb.x ^ vb.y ^ vb.z ^ vb.w;
        c ^= vc.x ^ vc.y ^ vc.z ^ vc.w;
        d ^= vd.x ^ vd.y ^ vd.z ^ vd.w;
    }
    for (; index < vectors; index += stride) {
        const uint4 value = data[index];
        a ^= value.x ^ value.y ^ value.z ^ value.w;
    }
    if ((a ^ b ^ c ^ d) == 0xDEAD'BEEFU) sink[threadIdx.x] = a;
}

// Exact production kernel arithmetic, with groups fixed to zero. Unlike the
// current generic dispatcher, this F8-0 arm does not quantize its activation
// carrier to E4M3 before launch: it is the scalar W8A16 baseline.
__global__ void f8_native_w8a16_kernel(
    float* output, const float* input, const std::uint8_t* weights,
    const std::uint8_t* scales, std::uint32_t scale_columns,
    std::uint32_t batch, std::uint32_t columns, std::uint32_t rows) {
    const std::uint32_t output_row = blockIdx.x;
    const std::uint32_t batch_row = blockIdx.y;
    if (output_row >= rows || batch_row >= batch) return;
    const std::uint64_t input_base =
        static_cast<std::uint64_t>(batch_row) * columns;
    const std::uint64_t weight_base =
        static_cast<std::uint64_t>(output_row) * columns;
    float sum = 0.0F;
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        const float weight = e4m3_value(weights[weight_base + column]);
        const float scale = e8m0_scale(
            scales[(output_row / kBlockN) * scale_columns +
                   column / kBlockK]);
        sum += input[input_base + column] * weight * scale;
    }
    sum = reduce_block(sum);
    if (threadIdx.x == 0U) {
        output[static_cast<std::uint64_t>(batch_row) * rows + output_row] =
            bf16_round(sum);
    }
}

// Exact existing page-control dataflow: compact E4M3 activations and weights
// widen through 48 KiB of shared BF16 storage before BF16 WMMA.
__global__ void f8_w8a8_tensor_page_kernel(
    float* output, const std::uint8_t* input,
    const std::uint8_t* input_scales, const std::uint8_t* weights,
    const std::uint8_t* weight_scales, std::uint32_t batch,
    std::uint32_t columns, std::uint32_t rows) {
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

    for (std::uint32_t tile_k = 0U; tile_k < columns; tile_k += kBlockK) {
        for (std::uint32_t index = threadIdx.x;
             index < kBlockM * kBlockK; index += blockDim.x) {
            const std::uint32_t local_m = index / kBlockK;
            const std::uint32_t local_k = index % kBlockK;
            const std::uint32_t global_m = tile_m + local_m;
            float value = 0.0F;
            if (global_m < batch) {
                const auto encoded = input[
                    static_cast<std::uint64_t>(global_m) * columns + tile_k +
                    local_k];
                const auto scale = input_scales[
                    static_cast<std::uint64_t>(global_m) * scale_columns +
                    tile_k / kBlockK];
                value = e4m3_value(encoded) * e8m0_scale(scale);
            }
            shared_a[index] = __float2bfloat16_rn(value);
        }
        for (std::uint32_t index = threadIdx.x;
             index < kBlockK * kBlockN; index += blockDim.x) {
            const std::uint32_t local_k = index / kBlockN;
            const std::uint32_t local_n = index % kBlockN;
            const std::uint32_t global_n = tile_n + local_n;
            shared_b[index] = __float2bfloat16_rn(e4m3_value(weights[
                static_cast<std::uint64_t>(global_n) * columns + tile_k +
                local_k]));
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                       wmma::row_major> a_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                       wmma::row_major> b_fragment;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float>
            accumulators[fragments_per_warp];
        for (auto& accumulator : accumulators) {
            wmma::fill_fragment(accumulator, 0.0F);
        }
        for (std::uint32_t local_k = 0U; local_k < kBlockK;
             local_k += 16U) {
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
                wmma::mma_sync(accumulators[fragment], a_fragment,
                               b_fragment, accumulators[fragment]);
            }
        }
        const float scale = e8m0_scale(weight_scales[
            (tile_n / kBlockN) * scale_columns + tile_k / kBlockK]);
        for (std::uint32_t fragment = 0U;
             fragment < fragments_per_warp; ++fragment) {
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

__global__ void f8_round_output_kernel(float* output, std::uint64_t elements) {
    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = bf16_round(output[index]);
}

class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        check(cudaMalloc(&value_, bytes), "allocate F8-0 device buffer");
    }
    ~DeviceBuffer() {
        if (value_ != nullptr) static_cast<void>(cudaFree(value_));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    [[nodiscard]] void* get() const noexcept { return value_; }

  private:
    void* value_{nullptr};
    std::size_t bytes_{0U};
};

template <typename Launch>
float time_once(cudaEvent_t start, cudaEvent_t finish, cudaStream_t stream,
                Launch&& launch) {
    check(cudaEventRecord(start, stream), "record F8-0 start");
    launch();
    check(cudaEventRecord(finish, stream), "record F8-0 finish");
    check(cudaEventSynchronize(finish), "synchronize F8-0 sample");
    float milliseconds = 0.0F;
    check(cudaEventElapsedTime(&milliseconds, start, finish),
          "measure F8-0 sample");
    return milliseconds;
}

float median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}

double gbps(std::uint64_t bytes, float milliseconds) {
    return static_cast<double>(bytes) /
           (static_cast<double>(milliseconds) * 1.0e6);
}

struct BoundaryError {
    double maximum_absolute = 0.0;
    double rms = 0.0;
    std::uint64_t mismatches = 0U;
};

BoundaryError oracle_error(
    const Shape& shape, std::uint32_t m,
    const std::vector<float>& input, const std::vector<std::uint8_t>& weights,
    const std::vector<std::uint8_t>& scales,
    const std::vector<float>& output) {
    const std::uint64_t elements = static_cast<std::uint64_t>(m) * shape.n;
    const std::uint32_t samples = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(kOracleSamples, elements));
    const std::uint64_t step = std::max<std::uint64_t>(1U, elements / samples);
    long double squared = 0.0;
    BoundaryError result;
    for (std::uint32_t sample = 0U; sample < samples; ++sample) {
        const std::uint64_t output_index =
            std::min<std::uint64_t>(elements - 1U, sample * step);
        const std::uint32_t batch_row =
            static_cast<std::uint32_t>(output_index / shape.n);
        const std::uint32_t output_row =
            static_cast<std::uint32_t>(output_index % shape.n);
        double oracle = 0.0;
        for (std::uint32_t column = 0U; column < shape.k; ++column) {
            oracle += static_cast<double>(
                          input[static_cast<std::uint64_t>(batch_row) *
                                    shape.k + column]) *
                      static_cast<double>(e4m3_value(
                          weights[static_cast<std::uint64_t>(output_row) *
                                      shape.k + column])) *
                      static_cast<double>(e8m0_scale(scales[
                          static_cast<std::uint64_t>(output_row / kBlockN) *
                              (shape.k / kBlockK) +
                          column / kBlockK]));
        }
        const float rounded_oracle = bf16_round(static_cast<float>(oracle));
        const double absolute = std::abs(
            static_cast<double>(output[output_index]) - rounded_oracle);
        result.maximum_absolute = std::max(result.maximum_absolute, absolute);
        squared += absolute * absolute;
        if (std::bit_cast<std::uint32_t>(output[output_index]) !=
            std::bit_cast<std::uint32_t>(rounded_oracle)) {
            ++result.mismatches;
        }
    }
    result.rms = std::sqrt(static_cast<double>(
        squared / static_cast<long double>(samples)));
    return result;
}

void print_samples(std::string_view name, const std::vector<float>& values,
                   bool comma = true) {
    std::cout << "  \"" << name << "\": [";
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']' << (comma ? ",\n" : "\n");
}

const Shape& find_shape(std::string_view name) {
    for (const auto& shape : kShapes) {
        if (name == shape.name) return shape;
    }
    throw std::runtime_error("unknown F8-0 shape: " + std::string(name));
}

struct Options {
    int device = 0;
    const Shape* shape = &kShapes[0];
    std::uint32_t m = 1U;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--device" && index + 1 < argc) {
            options.device = std::stoi(argv[++index]);
        } else if (argument == "--shape" && index + 1 < argc) {
            options.shape = &find_shape(argv[++index]);
        } else if (argument == "--m" && index + 1 < argc) {
            options.m = static_cast<std::uint32_t>(std::stoul(argv[++index]));
        } else {
            throw std::runtime_error(
                "usage: probe [--device INDEX] [--shape NAME] [--m ROWS]");
        }
    }
    if (options.m == 0U || options.m > kBlockM) {
        throw std::runtime_error("F8-0 probe supports M in [1,64]");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const auto& shape = *options.shape;
        check(cudaSetDevice(options.device), "select F8-0 device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, options.device),
              "query F8-0 device");
        if (properties.major != 8 || properties.minor != 6) {
            throw std::runtime_error("F8-0 baseline requires SM86");
        }

        const std::uint64_t code_bytes =
            static_cast<std::uint64_t>(shape.n) * shape.k;
        const std::uint64_t scale_rows =
            (static_cast<std::uint64_t>(shape.n) + kBlockN - 1U) / kBlockN;
        const std::uint64_t scale_columns =
            (static_cast<std::uint64_t>(shape.k) + kBlockK - 1U) / kBlockK;
        const std::uint64_t scale_bytes = scale_rows * scale_columns;
        const std::uint64_t matrix_bytes = code_bytes + scale_bytes;
        const std::uint32_t replicas = static_cast<std::uint32_t>(
            std::clamp<std::uint64_t>(kArenaBudget / matrix_bytes, 2U, 8U));
        const std::uint64_t arena_bytes = matrix_bytes * replicas;
        const std::uint64_t input_elements =
            static_cast<std::uint64_t>(options.m) * shape.k;
        const std::uint64_t input_scale_bytes =
            static_cast<std::uint64_t>(options.m) * scale_columns;
        const std::uint64_t padded_m =
            (static_cast<std::uint64_t>(options.m) + kBlockM - 1U) /
            kBlockM * kBlockM;
        const std::uint64_t output_elements = padded_m * shape.n;
        const std::uint64_t allocated_bytes =
            arena_bytes + kScrubBytes + kRooflineBytes + input_elements +
            input_elements * sizeof(float) + input_scale_bytes +
            output_elements * sizeof(float) + 256U * sizeof(unsigned int);
        if (allocated_bytes > kMemoryCeiling) {
            throw std::runtime_error("F8-0 allocation exceeds 512 MiB ceiling");
        }

        std::vector<std::uint8_t> codes(code_bytes);
        std::vector<std::uint8_t> scales(scale_bytes);
        std::vector<std::uint8_t> compact_input(input_elements);
        std::vector<std::uint8_t> input_scales(input_scale_bytes, 127U);
        std::vector<float> wide_input(input_elements);
        std::uint32_t random_state =
            0xA341'316CU ^ shape.n ^ (shape.k << 8U) ^ options.m;
        const auto random_e4m3 = [&] {
            const auto sign = (xorshift(random_state) & 1U) << 7U;
            const auto exponent = (4U + xorshift(random_state) % 8U) << 3U;
            const auto mantissa = xorshift(random_state) & 7U;
            return static_cast<std::uint8_t>(sign | exponent | mantissa);
        };
        for (auto& code : codes) code = random_e4m3();
        for (auto& scale : scales) {
            scale = static_cast<std::uint8_t>(123U +
                                              xorshift(random_state) % 9U);
        }
        for (std::uint64_t index = 0U; index < input_elements; ++index) {
            compact_input[index] = random_e4m3();
            wide_input[index] = bf16_round(e4m3_value(compact_input[index]));
        }
        std::vector<std::uint8_t> arena(arena_bytes);
        for (std::uint32_t replica = 0U; replica < replicas; ++replica) {
            auto* destination = arena.data() + replica * matrix_bytes;
            std::memcpy(destination, codes.data(), code_bytes);
            std::memcpy(destination + code_bytes, scales.data(), scale_bytes);
        }

        DeviceBuffer device_arena(arena_bytes);
        DeviceBuffer device_scrub(kScrubBytes);
        DeviceBuffer device_roofline(kRooflineBytes);
        DeviceBuffer device_compact_input(input_elements);
        DeviceBuffer device_wide_input(input_elements * sizeof(float));
        DeviceBuffer device_input_scales(input_scale_bytes);
        DeviceBuffer device_output(output_elements * sizeof(float));
        DeviceBuffer device_sink(256U * sizeof(unsigned int));
        cudaStream_t stream = nullptr;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "create F8-0 stream");
        check(cudaMemcpyAsync(device_arena.get(), arena.data(), arena_bytes,
                              cudaMemcpyHostToDevice, stream),
              "upload F8-0 arena");
        check(cudaMemcpyAsync(device_compact_input.get(), compact_input.data(),
                              input_elements, cudaMemcpyHostToDevice, stream),
              "upload F8-0 compact input");
        check(cudaMemcpyAsync(device_wide_input.get(), wide_input.data(),
                              input_elements * sizeof(float),
                              cudaMemcpyHostToDevice, stream),
              "upload F8-0 W8A16 input");
        check(cudaMemcpyAsync(device_input_scales.get(), input_scales.data(),
                              input_scale_bytes, cudaMemcpyHostToDevice,
                              stream),
              "upload F8-0 input scales");
        check(cudaMemsetAsync(device_roofline.get(), 0xA5, kRooflineBytes,
                              stream),
              "initialize F8-0 roofline stream");
        check(cudaStreamSynchronize(stream), "finish F8-0 uploads");

        const auto matrix = [&](std::uint32_t replica) {
            return static_cast<std::uint8_t*>(device_arena.get()) +
                   replica * matrix_bytes;
        };
        const auto scrub = [&] {
            f8_scrub_kernel<<<static_cast<unsigned int>(
                                  kScrubBytes / (256U * 16U)),
                              256U, 0U, stream>>>(
                static_cast<std::uint8_t*>(device_scrub.get()), kScrubBytes);
            check(cudaGetLastError(), "launch F8-0 L2 scrub");
        };
        const auto roofline = [&] {
            f8_read_roofline_kernel<<<properties.multiProcessorCount * 8U,
                                      256U, 0U, stream>>>(
                static_cast<const uint4*>(device_roofline.get()),
                kRooflineBytes / sizeof(uint4),
                static_cast<unsigned int*>(device_sink.get()));
            check(cudaGetLastError(), "launch F8-0 roofline");
        };
        const auto scalar = [&](std::uint32_t replica) {
            f8_native_w8a16_kernel<<<dim3{shape.n, options.m}, kThreads, 0U,
                                     stream>>>(
                static_cast<float*>(device_output.get()),
                static_cast<const float*>(device_wide_input.get()),
                matrix(replica), matrix(replica) + code_bytes,
                static_cast<std::uint32_t>(scale_columns), options.m, shape.k,
                shape.n);
            check(cudaGetLastError(), "launch F8-0 scalar W8A16");
        };
        const auto tensor = [&](std::uint32_t replica) {
            const dim3 quantize_grid(
                static_cast<unsigned int>(scale_columns), options.m, 1U);
            f8_quantize_activation_bytes_kernel<<<
                quantize_grid, kBlockK, 0U, stream>>>(
                static_cast<std::uint8_t*>(device_compact_input.get()),
                static_cast<std::uint8_t*>(device_input_scales.get()),
                static_cast<const float*>(device_wide_input.get()), shape.k,
                options.m);
            f8_w8a8_tensor_page_kernel<<<
                dim3{shape.n / kBlockN,
                     (options.m + kBlockM - 1U) / kBlockM},
                kThreads, 0U, stream>>>(
                static_cast<float*>(device_output.get()),
                static_cast<const std::uint8_t*>(device_compact_input.get()),
                static_cast<const std::uint8_t*>(device_input_scales.get()),
                matrix(replica), matrix(replica) + code_bytes, options.m,
                shape.k, shape.n);
            f8_round_output_kernel<<<
                static_cast<unsigned int>(
                    (static_cast<std::uint64_t>(options.m) * shape.n + 255U) /
                    256U),
                256U, 0U, stream>>>(
                static_cast<float*>(device_output.get()),
                static_cast<std::uint64_t>(options.m) * shape.n);
            check(cudaGetLastError(), "launch F8-0 tensor-page control");
        };

        cudaEvent_t start = nullptr;
        cudaEvent_t finish = nullptr;
        check(cudaEventCreate(&start), "create F8-0 start event");
        check(cudaEventCreate(&finish), "create F8-0 finish event");
        for (std::uint32_t warmup = 0U; warmup < kWarmups; ++warmup) {
            roofline();
            scalar(0U);
            tensor(0U);
        }
        check(cudaStreamSynchronize(stream), "finish F8-0 warmups");

        std::vector<float> ruler_samples;
        std::vector<float> scalar_samples;
        std::vector<float> tensor_samples;
        ruler_samples.reserve(kSamples);
        scalar_samples.reserve(kSamples);
        tensor_samples.reserve(kSamples);
        for (std::uint32_t sample = 0U; sample < kSamples; ++sample) {
            const std::uint32_t replica = sample % replicas;
            if ((sample & 1U) == 0U) {
                scrub();
                ruler_samples.push_back(time_once(start, finish, stream,
                                                   roofline));
                scrub();
                scalar_samples.push_back(time_once(
                    start, finish, stream, [&] { scalar(replica); }));
                scrub();
                tensor_samples.push_back(time_once(
                    start, finish, stream, [&] { tensor(replica); }));
            } else {
                scrub();
                tensor_samples.push_back(time_once(
                    start, finish, stream, [&] { tensor(replica); }));
                scrub();
                scalar_samples.push_back(time_once(
                    start, finish, stream, [&] { scalar(replica); }));
                scrub();
                ruler_samples.push_back(time_once(start, finish, stream,
                                                   roofline));
            }
        }

        std::vector<float> scalar_output(
            static_cast<std::uint64_t>(options.m) * shape.n);
        std::vector<float> tensor_output(scalar_output.size());
        scalar(0U);
        check(cudaMemcpyAsync(scalar_output.data(), device_output.get(),
                              scalar_output.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream),
              "download F8-0 scalar output");
        check(cudaStreamSynchronize(stream), "finish scalar output download");
        tensor(0U);
        check(cudaMemcpyAsync(tensor_output.data(), device_output.get(),
                              tensor_output.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream),
              "download F8-0 tensor output");
        check(cudaStreamSynchronize(stream), "finish tensor output download");

        double maximum_absolute = 0.0;
        std::uint64_t bf16_mismatches = 0U;
        for (std::size_t index = 0U; index < scalar_output.size(); ++index) {
            maximum_absolute = std::max(
                maximum_absolute,
                std::abs(static_cast<double>(scalar_output[index]) -
                         static_cast<double>(tensor_output[index])));
            if (std::bit_cast<std::uint32_t>(scalar_output[index]) !=
                std::bit_cast<std::uint32_t>(tensor_output[index])) {
                ++bf16_mismatches;
            }
        }
        const auto scalar_oracle = oracle_error(
            shape, options.m, wide_input, codes, scales, scalar_output);
        const auto tensor_oracle = oracle_error(
            shape, options.m, wide_input, codes, scales, tensor_output);
        const bool tensor_no_worse =
            tensor_oracle.maximum_absolute <=
                scalar_oracle.maximum_absolute &&
            tensor_oracle.rms <= scalar_oracle.rms &&
            tensor_oracle.mismatches <= scalar_oracle.mismatches;

        check(cudaEventDestroy(start), "destroy F8-0 start event");
        check(cudaEventDestroy(finish), "destroy F8-0 finish event");
        check(cudaStreamDestroy(stream), "destroy F8-0 stream");

        const float ruler_ms = median(ruler_samples);
        const float scalar_ms = median(scalar_samples);
        const float tensor_ms = median(tensor_samples);
        const std::uint64_t scalar_physical_weight_bytes =
            matrix_bytes * options.m;
        const std::uint64_t tensor_physical_weight_bytes =
            matrix_bytes * ((options.m + kBlockM - 1U) / kBlockM);
        std::cout << std::fixed << std::setprecision(9)
                  << "{\n"
                  << "  \"milestone\": \"F8-0\",\n"
                  << "  \"candidate\": false,\n"
                  << "  \"device_index\": " << options.device << ",\n"
                  << "  \"device_name\": \"" << properties.name << "\",\n"
                  << "  \"compute_capability\": \"" << properties.major
                  << '.' << properties.minor << "\",\n"
                  << "  \"shape\": \"" << shape.name << "\",\n"
                  << "  \"regions\": \"" << shape.regions << "\",\n"
                  << "  \"active_runtime_region\": "
                  << (shape.active ? "true" : "false") << ",\n"
                  << "  \"m\": " << options.m << ",\n"
                  << "  \"n\": " << shape.n << ",\n"
                  << "  \"k\": " << shape.k << ",\n"
                  << "  \"code_bytes\": " << code_bytes << ",\n"
                  << "  \"scale_bytes\": " << scale_bytes << ",\n"
                  << "  \"one_read_checkpoint_bytes\": " << matrix_bytes
                  << ",\n"
                  << "  \"arena_replicas\": " << replicas << ",\n"
                  << "  \"allocated_device_bytes\": " << allocated_bytes
                  << ",\n"
                  << "  \"memory_ceiling_bytes\": " << kMemoryCeiling
                  << ",\n"
                  << "  \"warmups\": " << kWarmups << ",\n"
                  << "  \"samples\": " << kSamples << ",\n"
                  << "  \"ruler_median_ms\": " << ruler_ms << ",\n"
                  << "  \"ruler_bytes\": " << kRooflineBytes << ",\n"
                  << "  \"ruler_gbps\": " << gbps(kRooflineBytes, ruler_ms)
                  << ",\n"
                  << "  \"scalar_w8a16_median_ms\": " << scalar_ms << ",\n"
                  << "  \"scalar_w8a16_physical_weight_bytes\": "
                  << scalar_physical_weight_bytes << ",\n"
                  << "  \"scalar_w8a16_physical_weight_gbps\": "
                  << gbps(scalar_physical_weight_bytes, scalar_ms) << ",\n"
                  << "  \"scalar_w8a16_one_read_equivalent_gbps\": "
                  << gbps(matrix_bytes, scalar_ms) << ",\n"
                  << "  \"tensor_w8a8_median_ms\": " << tensor_ms << ",\n"
                  << "  \"tensor_w8a8_includes_activation_quantization\": true,\n"
                  << "  \"tensor_w8a8_physical_weight_bytes\": "
                  << tensor_physical_weight_bytes << ",\n"
                  << "  \"tensor_w8a8_physical_weight_gbps\": "
                  << gbps(tensor_physical_weight_bytes, tensor_ms) << ",\n"
                  << "  \"tensor_w8a8_one_read_equivalent_gbps\": "
                  << gbps(matrix_bytes, tensor_ms) << ",\n"
                  << "  \"control_maximum_absolute_difference\": "
                  << maximum_absolute << ",\n"
                  << "  \"control_bf16_mismatches\": " << bf16_mismatches
                  << ",\n";
        std::cout << "  \"oracle_samples\": "
                  << std::min<std::uint64_t>(
                         kOracleSamples,
                         static_cast<std::uint64_t>(options.m) * shape.n)
                  << ",\n"
                  << "  \"scalar_oracle_bf16_maximum_absolute_error\": "
                  << scalar_oracle.maximum_absolute << ",\n"
                  << "  \"scalar_oracle_bf16_rms_error\": "
                  << scalar_oracle.rms << ",\n"
                  << "  \"scalar_oracle_bf16_mismatches\": "
                  << scalar_oracle.mismatches << ",\n"
                  << "  \"tensor_oracle_bf16_maximum_absolute_error\": "
                  << tensor_oracle.maximum_absolute << ",\n"
                  << "  \"tensor_oracle_bf16_rms_error\": "
                  << tensor_oracle.rms << ",\n"
                  << "  \"tensor_oracle_bf16_mismatches\": "
                  << tensor_oracle.mismatches << ",\n"
                  << "  \"tensor_no_worse_bf16_oracle\": "
                  << (tensor_no_worse ? "true" : "false") << ",\n";
        print_samples("ruler_sample_ms", ruler_samples);
        print_samples("scalar_w8a16_sample_ms", scalar_samples);
        print_samples("tensor_w8a8_sample_ms", tensor_samples, false);
        std::cout << "}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
