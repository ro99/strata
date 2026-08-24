// F8-1 phase-A falsifier: exact E4M3/E8M0 block-128 weights are decoded
// directly into the verified SM86 BF16 A-fragment registers.
//
// This is an isolated correctness and code-generation probe. It deliberately
// crosses both N=128 and K=128 scale boundaries, publishes no widened weight
// matrix, changes no production dispatch, and makes no throughput claim.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint32_t kN = 256U;
constexpr std::uint32_t kM = 8U;
constexpr std::uint32_t kK = 256U;
constexpr std::uint32_t kTileN = 16U;
constexpr std::uint32_t kTileK = 16U;
constexpr std::uint32_t kScaleBlock = 128U;
constexpr std::uint32_t kWarp = 32U;
constexpr std::uint32_t kNTiles = kN / kTileN;
constexpr std::uint32_t kKTiles = kK / kTileK;
constexpr std::uint32_t kScaleColumns = kK / kScaleBlock;
constexpr std::uint32_t kDecodeScaleCount = 255U;
constexpr std::uint32_t kDecodeCodeCount = 254U;

void check(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        check(cudaMalloc(&data_, bytes), "allocate FP8 register-feed buffer");
    }
    ~DeviceBuffer() {
        if (data_ != nullptr) static_cast<void>(cudaFree(data_));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    [[nodiscard]] void* get() const noexcept { return data_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

  private:
    void* data_{nullptr};
    std::size_t bytes_{0U};
};

std::uint32_t xorshift(std::uint32_t& state) {
    state ^= state << 13U;
    state ^= state >> 17U;
    state ^= state << 5U;
    return state;
}

float e4m3_value(std::uint8_t code) {
    const bool negative = (code & 0x80U) != 0U;
    const std::uint32_t exponent = (code >> 3U) & 0x0FU;
    const std::uint32_t mantissa = code & 0x07U;
    double magnitude = 0.0;
    if (exponent == 0U) {
        magnitude = std::ldexp(static_cast<double>(mantissa), -9);
    } else if (exponent == 15U && mantissa == 7U) {
        return std::numeric_limits<float>::quiet_NaN();
    } else {
        magnitude = std::ldexp(1.0 + static_cast<double>(mantissa) / 8.0,
                               static_cast<int>(exponent) - 7);
    }
    return static_cast<float>(negative ? -magnitude : magnitude);
}

float e8m0_value(std::uint8_t code) {
    return code == 0xFFU
               ? std::numeric_limits<float>::quiet_NaN()
               : std::ldexp(1.0F, static_cast<int>(code) - 127);
}

std::uint16_t bf16_bits(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7F80'0000U) == 0x7F80'0000U) {
        if ((bits & 0x007F'FFFFU) != 0U) return 0x7FC0U;
        return static_cast<std::uint16_t>(bits >> 16U);
    }
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

std::uint32_t pack_bf16(std::uint16_t low, std::uint16_t high) {
    return static_cast<std::uint32_t>(low) |
           (static_cast<std::uint32_t>(high) << 16U);
}

float bf16_value(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

// QPN8's transferable decode idea is retained, but re-derived for BF16:
// placing S/EEEE/MMM directly in BF16 produces exact E4M3 / 2^120, including
// E4M3 subnormals. Multiplication by 2^(scale-7) folds both that factor and
// the checkpoint's E8M0 scale into the register value. The fast path covers
// scale bytes 0..134; the high-scale path materializes exact unscaled E4M3
// BF16 first so the folding factor cannot overflow before the product.
__device__ __forceinline__ std::uint32_t e4m3_div_2p120_pair(
    std::uint32_t pair) {
    const std::uint32_t spread = __byte_perm(pair, 0U, 0x4140U);
    return ((spread << 8U) & 0x8000'8000U) |
           ((spread << 4U) & 0x07F0'07F0U);
}

__device__ __forceinline__ std::uint16_t e4m3_bf16_bits(
    std::uint32_t code) {
    const std::uint16_t sign = (code & 0x80U) != 0U ? 0x8000U : 0U;
    const std::uint32_t exponent = (code >> 3U) & 0x0FU;
    const std::uint32_t mantissa = code & 0x07U;
    if (exponent == 0U) {
        if (mantissa == 0U) return sign;
        const std::uint32_t top = 31U - __clz(mantissa);
        const std::uint32_t field = 118U + top;
        const std::uint32_t fraction =
            (mantissa - (1U << top)) << (7U - top);
        return static_cast<std::uint16_t>(sign | (field << 7U) | fraction);
    }
    if (exponent == 15U && mantissa == 7U) {
        return static_cast<std::uint16_t>(sign | 0x7FC0U);
    }
    return static_cast<std::uint16_t>(
        sign | ((exponent + 120U) << 7U) | (mantissa << 4U));
}

__device__ __forceinline__ std::uint32_t e4m3_bf16_pair(
    std::uint32_t pair) {
    return static_cast<std::uint32_t>(e4m3_bf16_bits(pair & 0xFFU)) |
           (static_cast<std::uint32_t>(e4m3_bf16_bits((pair >> 8U) & 0xFFU))
            << 16U);
}

__device__ __forceinline__ std::uint32_t multiply_bf16_pair(
    std::uint32_t values, std::uint32_t scale_pair) {
    const __nv_bfloat162 result = __hmul2(
        *reinterpret_cast<const __nv_bfloat162*>(&values),
        *reinterpret_cast<const __nv_bfloat162*>(&scale_pair));
    return *reinterpret_cast<const std::uint32_t*>(&result);
}

__device__ __forceinline__ std::uint32_t decode_e4m3_pair_scaled(
    std::uint32_t pair, std::uint32_t scale) {
    if (scale <= 134U) {
        const std::uint32_t factor = (scale + 120U) << 7U;
        return multiply_bf16_pair(e4m3_div_2p120_pair(pair),
                                  factor * 0x0001'0001U);
    }
    const std::uint32_t factor = scale << 7U;
    return multiply_bf16_pair(e4m3_bf16_pair(pair),
                              factor * 0x0001'0001U);
}

__global__ void exhaustive_decode_kernel(std::uint32_t* decoded) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr std::uint32_t count = kDecodeCodeCount * kDecodeScaleCount;
    if (index >= count) return;
    const std::uint32_t ordinal = index % kDecodeCodeCount;
    const std::uint32_t code = ordinal < 127U ? ordinal : ordinal + 1U;
    const std::uint32_t scale = index / kDecodeCodeCount;
    decoded[index] = decode_e4m3_pair_scaled(code | (code << 8U), scale);
}

template <bool kBreakKScale, bool kBreakNScale, bool kBreakPermutation>
__global__ void fp8_register_feed_kernel(
    const uint2* __restrict__ codes,
    const std::uint8_t* __restrict__ scales,
    const std::uint32_t* __restrict__ activations,
    float* __restrict__ output) {
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t n_tile = blockIdx.x;
    float d0 = 0.0F, d1 = 0.0F, d2 = 0.0F, d3 = 0.0F;

    for (std::uint32_t kt = 0U; kt < kKTiles; ++kt) {
        const uint2 q = codes[(n_tile * kKTiles + kt) * kWarp + lane];
        const std::uint32_t n_block = kBreakNScale ? 0U : n_tile / 8U;
        const std::uint32_t k_block = kBreakKScale ? 0U : kt / 8U;
        const std::uint32_t scale = scales[n_block * kScaleColumns + k_block];
        std::uint32_t a0 = decode_e4m3_pair_scaled(q.x & 0xFFFFU, scale);
        std::uint32_t a1 = decode_e4m3_pair_scaled(q.x >> 16U, scale);
        std::uint32_t a2 = decode_e4m3_pair_scaled(q.y & 0xFFFFU, scale);
        std::uint32_t a3 = decode_e4m3_pair_scaled(q.y >> 16U, scale);
        if constexpr (kBreakPermutation) {
            const std::uint32_t swap = a1;
            a1 = a2;
            a2 = swap;
        }
        const std::uint32_t* b = activations +
            (static_cast<std::size_t>(kt) * kWarp + lane) * 2U;
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b[0]), "r"(b[1]));
    }
    float* destination = output +
        (static_cast<std::size_t>(n_tile) * kWarp + lane) * 4U;
    destination[0] = d0;
    destination[1] = d1;
    destination[2] = d2;
    destination[3] = d3;
}

__global__ void bf16_fragment_control_kernel(
    const std::uint32_t* __restrict__ weights,
    const std::uint32_t* __restrict__ activations,
    float* __restrict__ output) {
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t n_tile = blockIdx.x;
    float d0 = 0.0F, d1 = 0.0F, d2 = 0.0F, d3 = 0.0F;
    for (std::uint32_t kt = 0U; kt < kKTiles; ++kt) {
        const std::uint32_t* a = weights +
            (static_cast<std::size_t>(n_tile) * kKTiles + kt) * kWarp * 4U +
            lane * 4U;
        const std::uint32_t* b = activations +
            (static_cast<std::size_t>(kt) * kWarp + lane) * 2U;
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
              "r"(b[1]));
    }
    float* destination = output +
        (static_cast<std::size_t>(n_tile) * kWarp + lane) * 4U;
    destination[0] = d0;
    destination[1] = d1;
    destination[2] = d2;
    destination[3] = d3;
}

struct DecodeResult {
    std::uint32_t cases{0U};
    std::uint32_t mismatches{0U};
};

struct MatrixResult {
    const char* name{nullptr};
    std::uint32_t bit_mismatches{0U};
    double maximum_absolute{0.0};
};

struct OracleResult {
    const char* name{nullptr};
    std::uint32_t violations{0U};
    double maximum_error_over_sum_abs{0.0};
};

struct KernelAttributes {
    const char* name{nullptr};
    int registers{0};
    std::size_t local_bytes{0U};
    std::size_t shared_bytes{0U};
    int active_blocks_per_sm{0};
    double waves{0.0};
};

template <typename Kernel>
KernelAttributes inspect_kernel(const char* name, Kernel kernel, int device) {
    cudaFuncAttributes attributes{};
    check(cudaFuncGetAttributes(&attributes, kernel), "query kernel attributes");
    int blocks = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel,
                                                        kWarp, 0U),
          "query kernel occupancy");
    int sms = 0;
    check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device),
          "query SM count");
    return {name, attributes.numRegs, attributes.localSizeBytes,
            attributes.sharedSizeBytes, blocks,
            static_cast<double>(kNTiles) / static_cast<double>(sms)};
}

DecodeResult run_decode(cudaStream_t stream) {
    constexpr std::uint32_t count = kDecodeCodeCount * kDecodeScaleCount;
    DeviceBuffer device(count * sizeof(std::uint32_t));
    exhaustive_decode_kernel<<<(count + 255U) / 256U, 256U, 0U, stream>>>(
        static_cast<std::uint32_t*>(device.get()));
    check(cudaGetLastError(), "launch exhaustive FP8 decoder");
    std::vector<std::uint32_t> actual(count);
    check(cudaMemcpyAsync(actual.data(), device.get(), device.bytes(),
                          cudaMemcpyDeviceToHost, stream),
          "download exhaustive FP8 decoder");
    check(cudaStreamSynchronize(stream), "finish exhaustive FP8 decoder");

    DecodeResult result{count, 0U};
    for (std::uint32_t index = 0U; index < count; ++index) {
        const std::uint32_t ordinal = index % kDecodeCodeCount;
        const std::uint8_t code = static_cast<std::uint8_t>(
            ordinal < 127U ? ordinal : ordinal + 1U);
        const std::uint8_t scale =
            static_cast<std::uint8_t>(index / kDecodeCodeCount);
        const float product = e4m3_value(code) * e8m0_value(scale);
        const std::uint16_t expected = bf16_bits(product);
        if ((actual[index] & 0xFFFFU) != expected ||
            (actual[index] >> 16U) != expected) {
            ++result.mismatches;
        }
    }
    return result;
}

template <bool kBreakK, bool kBreakN, bool kBreakP>
void launch_candidate(const uint2* codes, const std::uint8_t* scales,
                      const std::uint32_t* activations, float* output,
                      cudaStream_t stream) {
    fp8_register_feed_kernel<kBreakK, kBreakN, kBreakP>
        <<<kNTiles, kWarp, 0U, stream>>>(codes, scales, activations, output);
}

MatrixResult compare(const char* name, const std::vector<float>& actual,
                     const std::vector<float>& expected) {
    MatrixResult result{name, 0U, 0.0};
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        const double absolute = std::abs(static_cast<double>(actual[index]) -
                                         static_cast<double>(expected[index]));
        result.maximum_absolute = std::max(result.maximum_absolute, absolute);
        if (std::bit_cast<std::uint32_t>(actual[index]) !=
            std::bit_cast<std::uint32_t>(expected[index])) {
            ++result.bit_mismatches;
        }
    }
    return result;
}

std::vector<float> unpack_output(const std::vector<float>& fragments) {
    std::vector<float> matrix(static_cast<std::size_t>(kN) * kM);
    for (std::uint32_t nt = 0U; nt < kNTiles; ++nt) {
        for (std::uint32_t lane = 0U; lane < kWarp; ++lane) {
            const std::uint32_t group = lane >> 2U;
            const std::uint32_t thread = lane & 3U;
            for (std::uint32_t index = 0U; index < 4U; ++index) {
                const std::uint32_t row =
                    nt * kTileN + group + (index >= 2U ? 8U : 0U);
                const std::uint32_t column = thread * 2U + (index & 1U);
                matrix[static_cast<std::size_t>(row) * kM + column] =
                    fragments[(static_cast<std::size_t>(nt) * kWarp + lane) *
                                  4U +
                              index];
            }
        }
    }
    return matrix;
}

OracleResult compare_canonical_oracle(
    const std::vector<float>& actual,
    const std::vector<std::uint8_t>& canonical_codes,
    const std::array<std::uint8_t, 4U>& canonical_scales,
    const std::vector<std::uint16_t>& activation_bits) {
    OracleResult result{"canonical_bf16_matrix_oracle", 0U, 0.0};
    for (std::uint32_t row = 0U; row < kN; ++row) {
        for (std::uint32_t column = 0U; column < kM; ++column) {
            double expected = 0.0;
            double sum_abs = 0.0;
            for (std::uint32_t k = 0U; k < kK; ++k) {
                const std::uint8_t scale = canonical_scales[
                    (row / kScaleBlock) * kScaleColumns + k / kScaleBlock];
                const std::uint8_t code =
                    canonical_codes[static_cast<std::size_t>(row) * kK + k];
                const double weight = static_cast<double>(bf16_value(bf16_bits(
                    e4m3_value(code) * e8m0_value(scale))));
                const double activation = static_cast<double>(bf16_value(
                    activation_bits[static_cast<std::size_t>(k) * kM +
                                    column]));
                const double product = weight * activation;
                expected += product;
                sum_abs += std::abs(product);
            }
            const double error = std::abs(
                static_cast<double>(actual[static_cast<std::size_t>(row) * kM +
                                           column]) -
                expected);
            const double normalized = sum_abs == 0.0 ? error : error / sum_abs;
            result.maximum_error_over_sum_abs =
                std::max(result.maximum_error_over_sum_abs, normalized);
            // This is an association bound, not a format tolerance. The exact
            // register-fed-versus-BF16-fragment gate above remains bitwise.
            if (normalized > 5.0e-5) ++result.violations;
        }
    }
    return result;
}

void print_kernel(std::ostream& out, const KernelAttributes& value, bool last) {
    out << "    {\"name\": \"" << value.name
        << "\", \"registers_per_thread\": " << value.registers
        << ", \"local_bytes\": " << value.local_bytes
        << ", \"shared_bytes\": " << value.shared_bytes
        << ", \"active_blocks_per_sm\": " << value.active_blocks_per_sm
        << ", \"fixture_waves\": " << value.waves << "}"
        << (last ? "\n" : ",\n");
}

void print_matrix(std::ostream& out, const MatrixResult& value, bool last) {
    out << "    {\"name\": \"" << value.name
        << "\", \"bit_mismatches\": " << value.bit_mismatches
        << ", \"maximum_absolute\": " << value.maximum_absolute << "}"
        << (last ? "\n" : ",\n");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        int device_index = 0;
        std::string output_path;
        for (int i = 1; i < argc; ++i) {
            const std::string_view flag = argv[i];
            if (flag == "--device" && i + 1 < argc) {
                device_index = std::stoi(argv[++i]);
            } else if (flag == "--output" && i + 1 < argc) {
                output_path = argv[++i];
            } else {
                std::cerr << "usage: " << argv[0]
                          << " [--device INDEX] [--output PATH]\n";
                return EXIT_FAILURE;
            }
        }

        check(cudaSetDevice(device_index), "select FP8 register-feed device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device_index),
              "query FP8 register-feed device");
        if (properties.major != 8 || properties.minor != 6) {
            throw std::runtime_error("FP8 register-feed probe requires SM86");
        }

        std::vector<std::uint8_t> canonical_codes(
            static_cast<std::size_t>(kN) * kK);
        // Distinct powers across both block axes keep the deliberate bugs
        // observable while even max-finite BF16 activations remain finite.
        std::array<std::uint8_t, 4U> canonical_scales{88U, 92U, 96U, 100U};
        std::vector<std::uint16_t> activation_bits(
            static_cast<std::size_t>(kK) * kM);
        constexpr std::array<std::uint16_t, 24U> broad_bf16{
            0x0000U, 0x8000U, 0x0001U, 0x8001U, 0x007FU, 0x807FU,
            0x0080U, 0x8080U, 0x3F00U, 0xBF00U, 0x3F80U, 0xBF80U,
            0x4000U, 0xC000U, 0x4049U, 0xC049U, 0x3EAAU, 0xBEAAU,
            0x7F7FU, 0xFF7FU, 0x0100U, 0x8100U, 0x7E80U, 0xFE80U};
        std::uint32_t state = 0xF801'B10CU;
        for (std::uint8_t& code : canonical_codes) {
            code = static_cast<std::uint8_t>(xorshift(state) & 0xFFU);
            if ((code & 0x7FU) == 0x7FU) code ^= 0x01U;
        }
        for (std::size_t i = 0U; i < activation_bits.size(); ++i) {
            activation_bits[i] = broad_bf16[i % broad_bf16.size()];
        }

        std::vector<uint2> fragment_codes(
            static_cast<std::size_t>(kNTiles) * kKTiles * kWarp);
        std::vector<std::uint32_t> fragment_weights(
            static_cast<std::size_t>(kNTiles) * kKTiles * kWarp * 4U);
        for (std::uint32_t nt = 0U; nt < kNTiles; ++nt) {
            for (std::uint32_t kt = 0U; kt < kKTiles; ++kt) {
                const std::uint8_t scale = canonical_scales[
                    (nt / 8U) * kScaleColumns + kt / 8U];
                for (std::uint32_t lane = 0U; lane < kWarp; ++lane) {
                    const std::uint32_t group = lane >> 2U;
                    const std::uint32_t thread = lane & 3U;
                    std::uint64_t packed = 0U;
                    for (std::uint32_t index = 0U; index < 8U; ++index) {
                        const bool high_row =
                            index == 2U || index == 3U || index == 6U ||
                            index == 7U;
                        const std::uint32_t row =
                            nt * kTileN + group + (high_row ? 8U : 0U);
                        const std::uint32_t column =
                            kt * kTileK + thread * 2U + (index & 1U) +
                            (index >= 4U ? 8U : 0U);
                        packed |= static_cast<std::uint64_t>(
                                      canonical_codes[
                                          static_cast<std::size_t>(row) * kK +
                                          column])
                                  << (index * 8U);
                    }
                    const std::size_t slot =
                        (static_cast<std::size_t>(nt) * kKTiles + kt) * kWarp +
                        lane;
                    fragment_codes[slot] = make_uint2(
                        static_cast<std::uint32_t>(packed),
                        static_cast<std::uint32_t>(packed >> 32U));
                    for (std::uint32_t reg = 0U; reg < 4U; ++reg) {
                        const std::uint8_t low = static_cast<std::uint8_t>(
                            packed >> (reg * 16U));
                        const std::uint8_t high = static_cast<std::uint8_t>(
                            packed >> (reg * 16U + 8U));
                        fragment_weights[slot * 4U + reg] = pack_bf16(
                            bf16_bits(e4m3_value(low) * e8m0_value(scale)),
                            bf16_bits(e4m3_value(high) * e8m0_value(scale)));
                    }
                }
            }
        }

        std::vector<std::uint8_t> inverted_codes(canonical_codes.size());
        for (std::uint32_t nt = 0U; nt < kNTiles; ++nt) {
            for (std::uint32_t kt = 0U; kt < kKTiles; ++kt) {
                for (std::uint32_t lane = 0U; lane < kWarp; ++lane) {
                    const std::uint32_t group = lane >> 2U;
                    const std::uint32_t thread = lane & 3U;
                    const uint2 q = fragment_codes[
                        (static_cast<std::size_t>(nt) * kKTiles + kt) * kWarp +
                        lane];
                    const std::uint64_t packed =
                        static_cast<std::uint64_t>(q.x) |
                        (static_cast<std::uint64_t>(q.y) << 32U);
                    for (std::uint32_t index = 0U; index < 8U; ++index) {
                        const bool high_row =
                            index == 2U || index == 3U || index == 6U ||
                            index == 7U;
                        const std::uint32_t row =
                            nt * kTileN + group + (high_row ? 8U : 0U);
                        const std::uint32_t column =
                            kt * kTileK + thread * 2U + (index & 1U) +
                            (index >= 4U ? 8U : 0U);
                        inverted_codes[static_cast<std::size_t>(row) * kK +
                                       column] = static_cast<std::uint8_t>(
                            packed >> (index * 8U));
                    }
                }
            }
        }
        std::uint32_t prepack_inverse_mismatches = 0U;
        for (std::size_t index = 0U; index < canonical_codes.size(); ++index) {
            if (canonical_codes[index] != inverted_codes[index]) {
                ++prepack_inverse_mismatches;
            }
        }

        std::vector<std::uint32_t> fragment_activations(
            static_cast<std::size_t>(kKTiles) * kWarp * 2U);
        for (std::uint32_t kt = 0U; kt < kKTiles; ++kt) {
            for (std::uint32_t lane = 0U; lane < kWarp; ++lane) {
                const std::uint32_t group = lane >> 2U;
                const std::uint32_t thread = lane & 3U;
                const std::uint32_t r0 = kt * kTileK + thread * 2U;
                const std::size_t slot =
                    (static_cast<std::size_t>(kt) * kWarp + lane) * 2U;
                fragment_activations[slot] = pack_bf16(
                    activation_bits[static_cast<std::size_t>(r0) * kM + group],
                    activation_bits[static_cast<std::size_t>(r0 + 1U) * kM +
                                    group]);
                fragment_activations[slot + 1U] = pack_bf16(
                    activation_bits[static_cast<std::size_t>(r0 + 8U) * kM +
                                    group],
                    activation_bits[static_cast<std::size_t>(r0 + 9U) * kM +
                                    group]);
            }
        }

        DeviceBuffer d_codes(fragment_codes.size() * sizeof(uint2));
        DeviceBuffer d_scales(canonical_scales.size());
        DeviceBuffer d_weights(fragment_weights.size() * sizeof(std::uint32_t));
        DeviceBuffer d_activations(fragment_activations.size() *
                                   sizeof(std::uint32_t));
        const std::size_t output_bytes =
            static_cast<std::size_t>(kNTiles) * kWarp * 4U * sizeof(float);
        DeviceBuffer d_reference(output_bytes);
        DeviceBuffer d_candidate(output_bytes);
        DeviceBuffer d_bug_k(output_bytes);
        DeviceBuffer d_bug_n(output_bytes);
        DeviceBuffer d_bug_p(output_bytes);
        cudaStream_t stream = nullptr;
        check(cudaStreamCreate(&stream), "create FP8 register-feed stream");
        check(cudaMemcpyAsync(d_codes.get(), fragment_codes.data(), d_codes.bytes(),
                              cudaMemcpyHostToDevice, stream),
              "upload FP8 fragment codes");
        check(cudaMemcpyAsync(d_scales.get(), canonical_scales.data(),
                              d_scales.bytes(), cudaMemcpyHostToDevice, stream),
              "upload E8M0 scales");
        check(cudaMemcpyAsync(d_weights.get(), fragment_weights.data(),
                              d_weights.bytes(), cudaMemcpyHostToDevice, stream),
              "upload BF16 control fragments");
        check(cudaMemcpyAsync(d_activations.get(), fragment_activations.data(),
                              d_activations.bytes(), cudaMemcpyHostToDevice,
                              stream),
              "upload broad BF16 activation fragments");

        bf16_fragment_control_kernel<<<kNTiles, kWarp, 0U, stream>>>(
            static_cast<const std::uint32_t*>(d_weights.get()),
            static_cast<const std::uint32_t*>(d_activations.get()),
            static_cast<float*>(d_reference.get()));
        launch_candidate<false, false, false>(
            static_cast<const uint2*>(d_codes.get()),
            static_cast<const std::uint8_t*>(d_scales.get()),
            static_cast<const std::uint32_t*>(d_activations.get()),
            static_cast<float*>(d_candidate.get()), stream);
        launch_candidate<true, false, false>(
            static_cast<const uint2*>(d_codes.get()),
            static_cast<const std::uint8_t*>(d_scales.get()),
            static_cast<const std::uint32_t*>(d_activations.get()),
            static_cast<float*>(d_bug_k.get()), stream);
        launch_candidate<false, true, false>(
            static_cast<const uint2*>(d_codes.get()),
            static_cast<const std::uint8_t*>(d_scales.get()),
            static_cast<const std::uint32_t*>(d_activations.get()),
            static_cast<float*>(d_bug_n.get()), stream);
        launch_candidate<false, false, true>(
            static_cast<const uint2*>(d_codes.get()),
            static_cast<const std::uint8_t*>(d_scales.get()),
            static_cast<const std::uint32_t*>(d_activations.get()),
            static_cast<float*>(d_bug_p.get()), stream);
        check(cudaGetLastError(), "launch FP8 register-feed fixtures");

        const std::size_t output_count = output_bytes / sizeof(float);
        std::vector<float> reference(output_count), candidate(output_count),
            bug_k(output_count), bug_n(output_count), bug_p(output_count);
        check(cudaMemcpyAsync(reference.data(), d_reference.get(), output_bytes,
                              cudaMemcpyDeviceToHost, stream),
              "download BF16 fragment control");
        check(cudaMemcpyAsync(candidate.data(), d_candidate.get(), output_bytes,
                              cudaMemcpyDeviceToHost, stream),
              "download FP8 register feed");
        check(cudaMemcpyAsync(bug_k.data(), d_bug_k.get(), output_bytes,
                              cudaMemcpyDeviceToHost, stream),
              "download broken K-scale control");
        check(cudaMemcpyAsync(bug_n.data(), d_bug_n.get(), output_bytes,
                              cudaMemcpyDeviceToHost, stream),
              "download broken N-scale control");
        check(cudaMemcpyAsync(bug_p.data(), d_bug_p.get(), output_bytes,
                              cudaMemcpyDeviceToHost, stream),
              "download broken permutation control");
        check(cudaStreamSynchronize(stream), "finish FP8 register-feed fixtures");

        const DecodeResult decode = run_decode(stream);
        check(cudaStreamDestroy(stream), "destroy FP8 register-feed stream");
        const MatrixResult correct = compare("register_feed", candidate, reference);
        const MatrixResult broken_k = compare("break_k_scale", bug_k, reference);
        const MatrixResult broken_n = compare("break_n_scale", bug_n, reference);
        const MatrixResult broken_p = compare("break_permutation", bug_p, reference);
        const OracleResult canonical_oracle = compare_canonical_oracle(
            unpack_output(candidate), canonical_codes, canonical_scales,
            activation_bits);

        const KernelAttributes candidate_attributes = inspect_kernel(
            "fp8_register_feed_kernel", fp8_register_feed_kernel<false, false, false>,
            device_index);
        const KernelAttributes control_attributes = inspect_kernel(
            "bf16_fragment_control_kernel", bf16_fragment_control_kernel,
            device_index);
        const std::size_t peak_device_bytes =
            d_codes.bytes() + d_scales.bytes() + d_weights.bytes() +
            d_activations.bytes() + d_reference.bytes() + d_candidate.bytes() +
            d_bug_k.bytes() + d_bug_n.bytes() + d_bug_p.bytes() +
            static_cast<std::size_t>(kDecodeCodeCount) * kDecodeScaleCount *
                sizeof(std::uint32_t);

        std::ofstream file;
        std::ostream* output = &std::cout;
        if (!output_path.empty()) {
            file.open(output_path);
            if (!file) throw std::runtime_error("cannot open output path");
            output = &file;
        }
        *output << std::fixed << std::setprecision(9)
                << "{\n"
                << "  \"milestone\": \"F8-1 phase-A register feed\",\n"
                << "  \"device_index\": " << device_index << ",\n"
                << "  \"device_name\": \"" << properties.name << "\",\n"
                << "  \"device_capability\": \"" << properties.major << "."
                << properties.minor << "\",\n"
                << "  \"shape\": {\"M\": " << kM << ", \"N\": " << kN
                << ", \"K\": " << kK << "},\n"
                << "  \"scale_block\": 128,\n"
                << "  \"canonical_code_bytes\": " << canonical_codes.size()
                << ",\n"
                << "  \"prepacked_code_bytes\": " << d_codes.bytes() << ",\n"
                << "  \"canonical_scale_bytes\": " << canonical_scales.size()
                << ",\n"
                << "  \"prepacked_scale_bytes\": " << d_scales.bytes() << ",\n"
                << "  \"prepack_inverse_mismatches\": "
                << prepack_inverse_mismatches << ",\n"
                << "  \"peak_device_bytes\": " << peak_device_bytes << ",\n"
                << "  \"broad_finite_bf16_patterns\": " << broad_bf16.size()
                << ",\n"
                << "  \"decode_cases\": " << decode.cases << ",\n"
                << "  \"decode_mismatches\": " << decode.mismatches << ",\n"
                << "  \"rejected_e4m3_nan_codes\": 2,\n"
                << "  \"rejected_e8m0_nan_codes\": 1,\n"
                << "  \"canonical_oracle\": {\"violations\": "
                << canonical_oracle.violations
                << ", \"maximum_error_over_sum_abs\": "
                << canonical_oracle.maximum_error_over_sum_abs << "},\n"
                << "  \"matrices\": [\n";
        print_matrix(*output, correct, false);
        print_matrix(*output, broken_k, false);
        print_matrix(*output, broken_n, false);
        print_matrix(*output, broken_p, true);
        *output << "  ],\n  \"kernels\": [\n";
        print_kernel(*output, candidate_attributes, false);
        print_kernel(*output, control_attributes, true);
        *output << "  ]\n}\n";

        const bool clean_resources = candidate_attributes.local_bytes == 0U &&
                                     candidate_attributes.shared_bytes == 0U;
        const bool bug_controls_fire = broken_k.bit_mismatches != 0U &&
                                       broken_n.bit_mismatches != 0U &&
                                       broken_p.bit_mismatches != 0U;
        return decode.mismatches == 0U && correct.bit_mismatches == 0U &&
                       canonical_oracle.violations == 0U &&
                       prepack_inverse_mismatches == 0U && bug_controls_fire &&
                       clean_resources &&
                       d_codes.bytes() == canonical_codes.size() &&
                       d_scales.bytes() == canonical_scales.size()
                   ? EXIT_SUCCESS
                   : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
