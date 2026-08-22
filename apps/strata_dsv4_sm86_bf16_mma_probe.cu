// C2 phase-A feasibility probe: native register-fed SM86 BF16 MMA.
// This does not decode FP4 and makes no throughput claim.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr std::uint32_t kM = 16U;
constexpr std::uint32_t kN = 8U;
constexpr std::uint32_t kK = 16U;
constexpr std::uint32_t kWarp = 32U;

void check(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t bytes) {
        check(cudaMalloc(&data_, bytes), "allocate BF16 MMA probe buffer");
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) static_cast<void>(cudaFree(data_));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    [[nodiscard]] void* get() const noexcept { return data_; }

  private:
    void* data_{nullptr};
};

std::uint16_t bf16_bits(float value) {
    std::uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t rounded = bits + 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(rounded >> 16U);
}

float bf16_value(float value) {
    const std::uint32_t bits =
        static_cast<std::uint32_t>(bf16_bits(value)) << 16U;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::uint32_t pack_bf16(std::uint16_t low, std::uint16_t high) {
    return static_cast<std::uint32_t>(low) |
           (static_cast<std::uint32_t>(high) << 16U);
}

float fp4_e2m1_value(std::uint8_t code) {
    constexpr std::array<float, 8U> magnitudes{
        0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    const float magnitude = magnitudes[code & 0x07U];
    return (code & 0x08U) != 0U && magnitude != 0.0F
               ? -magnitude
               : magnitude;
}

float e8m0_scale(std::uint8_t encoded) {
    return std::ldexp(1.0F, static_cast<int>(encoded) - 127);
}

__device__ std::uint32_t decode_e2m1_pair_scaled(std::uint32_t byte,
                                                 std::uint32_t scale) {
    const std::uint32_t bits =
        ((byte & 0x08U) << 12U) | ((byte & 0x80U) << 24U) |
        ((byte & 0x07U) << 6U) | ((byte & 0x70U) << 18U);
    const std::uint32_t delta =
        (((scale - 1U) << 7U) * 0x0001'0001U);
    const std::uint32_t magnitudes = bits & 0x7FFF'7FFFU;
    const std::uint32_t magnitude_indicators =
        (magnitudes + 0x7FFF'7FFFU) & 0x8000'8000U;
    const std::uint32_t magnitude_nonzero =
        magnitude_indicators |
        (magnitude_indicators - (magnitude_indicators >> 15U));
    const std::uint32_t exponent_indicators =
        ((bits & 0x7F80'7F80U) + 0x7F80'7F80U) & 0x8000'8000U;
    const std::uint32_t exponent_zero =
        ~(exponent_indicators |
          (exponent_indicators - (exponent_indicators >> 15U)));
    const std::uint32_t sign_only = bits & 0x8000'8000U;
    std::uint32_t scaled =
        (bits & ~exponent_zero) | (sign_only & exponent_zero);
    scaled += delta;
    return scaled & magnitude_nonzero;
}

__device__ std::uint32_t pack_bf16_representable(float low, float high) {
    return (__float_as_uint(low) >> 16U) |
           (__float_as_uint(high) & 0xFFFF'0000U);
}

__global__ void exhaustive_decode_kernel(std::uint32_t* decoded) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr std::uint32_t kScaleCount = 249U;
    constexpr std::uint32_t kCount = 256U * kScaleCount;
    if (index >= kCount) return;
    const std::uint32_t byte = index & 0xFFU;
    const std::uint32_t scale = 2U + index / 256U;
    decoded[index] = decode_e2m1_pair_scaled(byte, scale);
}

__global__ void bf16_m16n8k16_kernel(const std::uint32_t* a_fragments,
                                     const std::uint32_t* b_fragments,
                                     float* d_fragments) {
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t* a = a_fragments + lane * 4U;
    const std::uint32_t* b = b_fragments + lane * 2U;
    float d0 = 0.0F;
    float d1 = 0.0F;
    float d2 = 0.0F;
    float d3 = 0.0F;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
    float* output = d_fragments + lane * 4U;
    output[0] = d0;
    output[1] = d1;
    output[2] = d2;
    output[3] = d3;
}

__global__ void fp4_m16n8k16_kernel(const std::uint32_t* packed_codes,
                                    const std::uint32_t* packed_scales,
                                    const float* b_matrix,
                                    float* d_fragments) {
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t group = lane >> 2U;
    const std::uint32_t thread = lane & 3U;
    const std::uint32_t codes = packed_codes[lane];
    const std::uint32_t scales = packed_scales[lane];
    const std::uint32_t scale_low = scales & 0xFFU;
    const std::uint32_t scale_high = (scales >> 8U) & 0xFFU;
    const std::uint32_t a0 =
        decode_e2m1_pair_scaled(codes & 0xFFU, scale_low);
    const std::uint32_t a1 =
        decode_e2m1_pair_scaled((codes >> 8U) & 0xFFU, scale_high);
    const std::uint32_t a2 =
        decode_e2m1_pair_scaled((codes >> 16U) & 0xFFU, scale_low);
    const std::uint32_t a3 =
        decode_e2m1_pair_scaled((codes >> 24U) & 0xFFU, scale_high);

    const std::uint32_t row0 = thread * 2U;
    const std::uint32_t row1 = row0 + 1U;
    const std::uint32_t b0 = pack_bf16_representable(
        b_matrix[row0 * kN + group], b_matrix[row1 * kN + group]);
    const std::uint32_t b1 = pack_bf16_representable(
        b_matrix[(row0 + 8U) * kN + group],
        b_matrix[(row1 + 8U) * kN + group]);

    float d0 = 0.0F;
    float d1 = 0.0F;
    float d2 = 0.0F;
    float d3 = 0.0F;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    float* output = d_fragments + lane * 4U;
    output[0] = d0;
    output[1] = d1;
    output[2] = d2;
    output[3] = d3;
}

using MatrixA = std::array<float, kM * kK>;
using MatrixB = std::array<float, kK * kN>;
using MatrixD = std::array<float, kM * kN>;
using FragmentA = std::array<std::uint32_t, kWarp * 4U>;
using FragmentB = std::array<std::uint32_t, kWarp * 2U>;
using FragmentD = std::array<float, kWarp * 4U>;
using Codes = std::array<std::uint8_t, kM * kK>;
using Scales = std::array<std::uint8_t, kM>;

FragmentA pack_a(const MatrixA& matrix) {
    FragmentA fragments{};
    for (std::uint32_t lane = 0U; lane < kWarp; ++lane) {
        const std::uint32_t group = lane >> 2U;
        const std::uint32_t thread = lane & 3U;
        std::array<std::uint16_t, 8U> elements{};
        for (std::uint32_t index = 0U; index < elements.size(); ++index) {
            const bool upper_row =
                !((index < 2U) || (index >= 4U && index < 6U));
            const std::uint32_t row = group + (upper_row ? 8U : 0U);
            const std::uint32_t column =
                thread * 2U + (index & 1U) + (index >= 4U ? 8U : 0U);
            elements[index] = bf16_bits(matrix[row * kK + column]);
        }
        for (std::uint32_t reg = 0U; reg < 4U; ++reg) {
            fragments[lane * 4U + reg] =
                pack_bf16(elements[reg * 2U], elements[reg * 2U + 1U]);
        }
    }
    return fragments;
}

FragmentB pack_b(const MatrixB& matrix) {
    FragmentB fragments{};
    for (std::uint32_t lane = 0U; lane < kWarp; ++lane) {
        const std::uint32_t group = lane >> 2U;
        const std::uint32_t thread = lane & 3U;
        std::array<std::uint16_t, 4U> elements{};
        for (std::uint32_t index = 0U; index < elements.size(); ++index) {
            const std::uint32_t row =
                thread * 2U + (index & 1U) + (index >= 2U ? 8U : 0U);
            const std::uint32_t column = group;
            elements[index] = bf16_bits(matrix[row * kN + column]);
        }
        fragments[lane * 2U] = pack_bf16(elements[0], elements[1]);
        fragments[lane * 2U + 1U] = pack_bf16(elements[2], elements[3]);
    }
    return fragments;
}

MatrixD unpack_d(const FragmentD& fragments) {
    MatrixD matrix{};
    for (std::uint32_t lane = 0U; lane < kWarp; ++lane) {
        const std::uint32_t group = lane >> 2U;
        const std::uint32_t thread = lane & 3U;
        for (std::uint32_t index = 0U; index < 4U; ++index) {
            const std::uint32_t row = group + (index >= 2U ? 8U : 0U);
            const std::uint32_t column = thread * 2U + (index & 1U);
            matrix[row * kN + column] = fragments[lane * 4U + index];
        }
    }
    return matrix;
}

MatrixD oracle(const MatrixA& a, const MatrixB& b) {
    MatrixD expected{};
    for (std::uint32_t row = 0U; row < kM; ++row) {
        for (std::uint32_t column = 0U; column < kN; ++column) {
            float sum = 0.0F;
            for (std::uint32_t k = 0U; k < kK; ++k) {
                sum += bf16_value(a[row * kK + k]) *
                       bf16_value(b[k * kN + column]);
            }
            expected[row * kN + column] = sum;
        }
    }
    return expected;
}

struct FixtureResult {
    const char* name;
    double maximum_absolute{0.0};
    std::uint32_t mismatches{0U};
};

struct DecodeResult {
    std::uint32_t cases{0U};
    std::uint32_t mismatches{0U};
};

struct KernelAttributes {
    const char* name;
    int registers_per_thread{0};
    std::size_t local_bytes{0U};
    std::size_t shared_bytes{0U};
    int max_active_blocks_per_sm{0};
    int warps_per_sm{0};
    double occupancy{0.0};
};

template <typename Kernel>
KernelAttributes inspect_kernel(const char* name, Kernel kernel, int device) {
    cudaFuncAttributes attributes{};
    check(cudaFuncGetAttributes(&attributes, kernel),
          "query kernel attributes");
    int blocks = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &blocks, kernel, static_cast<int>(kWarp), 0U),
          "query kernel occupancy");
    int max_threads_per_sm = 0;
    check(cudaDeviceGetAttribute(&max_threads_per_sm,
                                 cudaDevAttrMaxThreadsPerMultiProcessor,
                                 device),
          "query device threads per multiprocessor");
    KernelAttributes result{name};
    result.registers_per_thread = attributes.numRegs;
    result.local_bytes = attributes.localSizeBytes;
    result.shared_bytes = attributes.sharedSizeBytes;
    result.max_active_blocks_per_sm = blocks;
    result.warps_per_sm = blocks;
    result.occupancy =
        max_threads_per_sm > 0
            ? static_cast<double>(blocks) * static_cast<double>(kWarp) /
                  static_cast<double>(max_threads_per_sm)
            : 0.0;
    return result;
}

DecodeResult run_exhaustive_decode(cudaStream_t stream) {
    constexpr std::uint32_t kScaleCount = 249U;
    constexpr std::uint32_t kCount = 256U * kScaleCount;
    DeviceBuffer device_decoded(kCount * sizeof(std::uint32_t));
    exhaustive_decode_kernel<<<(kCount + 255U) / 256U, 256U, 0U, stream>>>(
        static_cast<std::uint32_t*>(device_decoded.get()));
    check(cudaGetLastError(), "launch exhaustive E2M1/E8M0 decoder");
    std::array<std::uint32_t, kCount> decoded{};
    check(cudaMemcpyAsync(decoded.data(), device_decoded.get(),
                          sizeof(decoded), cudaMemcpyDeviceToHost, stream),
          "download exhaustive decoder output");
    check(cudaStreamSynchronize(stream), "finish exhaustive decoder");

    DecodeResult result{kCount, 0U};
    for (std::uint32_t index = 0U; index < kCount; ++index) {
        const std::uint8_t byte = static_cast<std::uint8_t>(index & 0xFFU);
        const std::uint8_t scale =
            static_cast<std::uint8_t>(2U + index / 256U);
        const std::uint16_t low = bf16_bits(
            fp4_e2m1_value(byte & 0x0FU) * e8m0_scale(scale));
        const std::uint16_t high = bf16_bits(
            fp4_e2m1_value(byte >> 4U) * e8m0_scale(scale));
        if (decoded[index] != pack_bf16(low, high)) ++result.mismatches;
    }
    return result;
}

FixtureResult run_fp4_fixture(const Codes& codes, const Scales& scales,
                              const MatrixB& b, cudaStream_t stream) {
    std::array<std::uint32_t, kWarp> packed_codes{};
    std::array<std::uint32_t, kWarp> packed_scales{};
    for (std::uint32_t lane = 0U; lane < kWarp; ++lane) {
        const std::uint32_t group = lane >> 2U;
        const std::uint32_t thread = lane & 3U;
        std::uint32_t packed = 0U;
        std::uint32_t low_row = 0U;
        std::uint32_t high_row = 0U;
        for (std::uint32_t index = 0U; index < 8U; ++index) {
            const bool upper_row =
                !((index < 2U) || (index >= 4U && index < 6U));
            const std::uint32_t row = group + (upper_row ? 8U : 0U);
            const std::uint32_t column =
                thread * 2U + (index & 1U) + (index >= 4U ? 8U : 0U);
            packed |= static_cast<std::uint32_t>(
                          codes[row * kK + column])
                      << (index * 4U);
            if (upper_row) {
                high_row = row;
            } else {
                low_row = row;
            }
        }
        packed_codes[lane] = packed;
        packed_scales[lane] =
            static_cast<std::uint32_t>(scales[low_row]) |
            (static_cast<std::uint32_t>(scales[high_row]) << 8U);
    }

    DeviceBuffer device_codes(sizeof(packed_codes));
    DeviceBuffer device_scales(sizeof(packed_scales));
    DeviceBuffer device_b(sizeof(b));
    DeviceBuffer device_d(sizeof(FragmentD));
    check(cudaMemcpyAsync(device_codes.get(), packed_codes.data(),
                          sizeof(packed_codes), cudaMemcpyHostToDevice, stream),
          "upload fragment-order FP4 codes");
    check(cudaMemcpyAsync(device_scales.get(), packed_scales.data(),
                          sizeof(packed_scales), cudaMemcpyHostToDevice,
                          stream),
          "upload fragment-order E8M0 scales");
    check(cudaMemcpyAsync(device_b.get(), b.data(), sizeof(b),
                          cudaMemcpyHostToDevice, stream),
          "upload BF16-representable activations");
    fp4_m16n8k16_kernel<<<1U, kWarp, 0U, stream>>>(
        static_cast<const std::uint32_t*>(device_codes.get()),
        static_cast<const std::uint32_t*>(device_scales.get()),
        static_cast<const float*>(device_b.get()),
        static_cast<float*>(device_d.get()));
    check(cudaGetLastError(), "launch direct FP4-to-BF16 MMA");
    FragmentD d_fragments{};
    check(cudaMemcpyAsync(d_fragments.data(), device_d.get(),
                          sizeof(d_fragments), cudaMemcpyDeviceToHost, stream),
          "download direct FP4 MMA output");
    check(cudaStreamSynchronize(stream), "finish direct FP4 MMA fixture");

    MatrixA decoded_a{};
    for (std::uint32_t row = 0U; row < kM; ++row) {
        for (std::uint32_t column = 0U; column < kK; ++column) {
            decoded_a[row * kK + column] =
                fp4_e2m1_value(codes[row * kK + column]) *
                e8m0_scale(scales[row]);
        }
    }
    const MatrixD actual = unpack_d(d_fragments);
    const MatrixD expected = oracle(decoded_a, b);
    FixtureResult result{"fp4_e8m0_register_feed"};
    for (std::uint32_t index = 0U; index < actual.size(); ++index) {
        const double absolute =
            std::abs(static_cast<double>(actual[index]) - expected[index]);
        result.maximum_absolute = std::max(result.maximum_absolute, absolute);
        if (actual[index] != expected[index]) ++result.mismatches;
    }
    return result;
}

FixtureResult run_fixture(const char* name, const MatrixA& a,
                          const MatrixB& b, cudaStream_t stream) {
    const FragmentA a_fragments = pack_a(a);
    const FragmentB b_fragments = pack_b(b);
    DeviceBuffer device_a(sizeof(a_fragments));
    DeviceBuffer device_b(sizeof(b_fragments));
    DeviceBuffer device_d(sizeof(FragmentD));
    check(cudaMemcpyAsync(device_a.get(), a_fragments.data(),
                          sizeof(a_fragments), cudaMemcpyHostToDevice, stream),
          "upload A fragments");
    check(cudaMemcpyAsync(device_b.get(), b_fragments.data(),
                          sizeof(b_fragments), cudaMemcpyHostToDevice, stream),
          "upload B fragments");
    bf16_m16n8k16_kernel<<<1U, kWarp, 0U, stream>>>(
        static_cast<const std::uint32_t*>(device_a.get()),
        static_cast<const std::uint32_t*>(device_b.get()),
        static_cast<float*>(device_d.get()));
    check(cudaGetLastError(), "launch native BF16 MMA");
    FragmentD d_fragments{};
    check(cudaMemcpyAsync(d_fragments.data(), device_d.get(),
                          sizeof(d_fragments), cudaMemcpyDeviceToHost, stream),
          "download D fragments");
    check(cudaStreamSynchronize(stream), "finish native BF16 MMA fixture");

    const MatrixD actual = unpack_d(d_fragments);
    const MatrixD expected = oracle(a, b);
    FixtureResult result{name};
    for (std::uint32_t index = 0U; index < actual.size(); ++index) {
        const double absolute =
            std::abs(static_cast<double>(actual[index]) - expected[index]);
        result.maximum_absolute = std::max(result.maximum_absolute, absolute);
        if (actual[index] != expected[index]) ++result.mismatches;
    }
    return result;
}

void print_kernel(std::ostream& output, const KernelAttributes& kernel,
                  bool last) {
    output << "    {\"name\": \"" << kernel.name
           << "\", \"registers_per_thread\": " << kernel.registers_per_thread
           << ", \"local_bytes\": " << kernel.local_bytes
           << ", \"shared_bytes\": " << kernel.shared_bytes
           << ", \"max_active_warps_per_sm\": "
           << kernel.max_active_blocks_per_sm
           << ", \"occupancy\": " << std::setprecision(4) << kernel.occupancy
           << std::setprecision(9) << "}" << (last ? "\n" : ",\n");
}

void print_result(std::ostream& output, int device,
                  const cudaDeviceProp& properties,
                  const FixtureResult& identity,
                  const FixtureResult& random,
                  const DecodeResult& decode,
                  const FixtureResult& fp4,
                  const KernelAttributes& bf16_kernel,
                  const KernelAttributes& fp4_kernel) {
    output << std::fixed << std::setprecision(9)
           << "{\n"
           << "  \"device_index\": " << device << ",\n"
           << "  \"device_name\": \"" << properties.name << "\",\n"
           << "  \"device_capability\": \"" << properties.major << "."
           << properties.minor << "\",\n"
           << "  \"instruction\": \"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\",\n"
           << "  \"bf16_allocated_bytes_per_fixture\": "
           << (sizeof(FragmentA) + sizeof(FragmentB) + sizeof(FragmentD))
           << ",\n"
           << "  \"decode_cases\": " << decode.cases << ",\n"
           << "  \"decode_mismatches\": " << decode.mismatches << ",\n"
           << "  \"fixtures\": [\n"
           << "    {\"name\": \"" << identity.name
           << "\", \"maximum_absolute\": " << identity.maximum_absolute
           << ", \"mismatches\": " << identity.mismatches << "},\n"
           << "    {\"name\": \"" << random.name
           << "\", \"maximum_absolute\": " << random.maximum_absolute
           << ", \"mismatches\": " << random.mismatches << "},\n"
           << "    {\"name\": \"" << fp4.name
           << "\", \"maximum_absolute\": " << fp4.maximum_absolute
           << ", \"mismatches\": " << fp4.mismatches << "}\n"
           << "  ],\n"
           << "  \"kernels\": [\n";
    print_kernel(output, bf16_kernel, false);
    print_kernel(output, fp4_kernel, true);
    output << "  ]\n"
           << "}\n";
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

        check(cudaSetDevice(device), "select BF16 MMA probe device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device),
              "query BF16 MMA probe device");
        if (properties.major != 8 || properties.minor != 6) {
            throw std::runtime_error("BF16 MMA probe requires SM86");
        }

        MatrixA identity_a{};
        MatrixB identity_b{};
        for (std::uint32_t index = 0U; index < kM; ++index) {
            identity_a[index * kK + index] = 1.0F;
        }
        for (std::uint32_t row = 0U; row < kK; ++row) {
            for (std::uint32_t column = 0U; column < kN; ++column) {
                identity_b[row * kN + column] =
                    static_cast<float>(static_cast<int>(row) - 7) +
                    static_cast<float>(column) * 0.125F;
            }
        }

        MatrixA random_a{};
        MatrixB random_b{};
        std::uint32_t state = 0xC001'D00DU;
        auto next = [&]() {
            state ^= state << 13U;
            state ^= state >> 17U;
            state ^= state << 5U;
            return state;
        };
        for (float& value : random_a) {
            value = static_cast<float>(static_cast<int>(next() % 9U) - 4);
        }
        for (float& value : random_b) {
            value = static_cast<float>(static_cast<int>(next() % 7U) - 3);
        }

        cudaStream_t stream = nullptr;
        check(cudaStreamCreate(&stream), "create BF16 MMA probe stream");
        const FixtureResult identity =
            run_fixture("identity_unique", identity_a, identity_b, stream);
        const FixtureResult random =
            run_fixture("random_small_integer", random_a, random_b, stream);
        const DecodeResult decode = run_exhaustive_decode(stream);

        Codes fp4_codes{};
        Scales fp4_scales{};
        MatrixB fp4_activations{};
        for (std::uint8_t& code : fp4_codes) {
            code = static_cast<std::uint8_t>(next() & 0x0FU);
        }
        for (std::uint8_t& scale : fp4_scales) {
            scale = static_cast<std::uint8_t>(125U + next() % 5U);
        }
        for (float& value : fp4_activations) {
            value = static_cast<float>(static_cast<int>(next() % 9U) - 4);
        }
        const FixtureResult fp4 =
            run_fp4_fixture(fp4_codes, fp4_scales, fp4_activations, stream);
        check(cudaStreamDestroy(stream), "destroy BF16 MMA probe stream");

        const KernelAttributes bf16_kernel = inspect_kernel(
            "bf16_m16n8k16_kernel", bf16_m16n8k16_kernel, device);
        const KernelAttributes fp4_kernel = inspect_kernel(
            "fp4_m16n8k16_kernel", fp4_m16n8k16_kernel, device);

        if (output_path.empty()) {
            print_result(std::cout, device, properties, identity, random,
                         decode, fp4, bf16_kernel, fp4_kernel);
        } else {
            std::ofstream output(output_path);
            if (!output) {
                throw std::runtime_error("cannot open output path: " +
                                         output_path);
            }
            print_result(output, device, properties, identity, random,
                         decode, fp4, bf16_kernel, fp4_kernel);
        }
        const bool register_fed =
            bf16_kernel.local_bytes == 0U && bf16_kernel.shared_bytes == 0U &&
            fp4_kernel.local_bytes == 0U && fp4_kernel.shared_bytes == 0U;
        return identity.mismatches == 0U && random.mismatches == 0U &&
                       decode.mismatches == 0U && fp4.mismatches == 0U &&
                       register_fed
                   ? EXIT_SUCCESS
                   : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
