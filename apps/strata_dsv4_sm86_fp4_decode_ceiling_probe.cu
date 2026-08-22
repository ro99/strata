// F4-1 phase-A falsifier: the SM86 ceiling on E2M1/E8M0 group-32 decode.
//
// This probe measures an UPPER BOUND on any register-fed W4A16 candidate. It
// reads the packed code and scale streams at their production shapes and
// decodes them into BF16 operand registers, but it does no fragment prepack,
// no activation feed, no output publication, and no split-K reduction. A real
// candidate must do strictly more work, so a ceiling below the F4-2 parity
// gate falsifies the decoder; a ceiling above it is necessary, not sufficient.
//
// Milestone F4-1, FP4 track only. Nothing here is FP8 evidence.

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
#include <vector>

namespace {

constexpr std::uint32_t kGroupSize = 32U;
constexpr std::uint32_t kWarmups = 3U;
constexpr std::uint32_t kSamples = 11U;
// Experiment 0136 defect fix: at one production matrix per launch the timed
// window is about 10 us against 4.1 us of measured launch/event overhead, a
// 40% deflation that reproduces the contract's forbidden 435-484 GB/s ruler.
// The arena is therefore swept at the established 128 MiB ruler scale so the
// window is ~170 us and launch overhead is under 3%.
constexpr std::uint32_t kArenaReplicas = 30U;
constexpr std::size_t kScrubBytes = 256ULL * 1024ULL * 1024ULL;
constexpr std::uint32_t kThreads = 256U;
constexpr std::uint32_t kOracleGroups = 4096U;

// E8M0 codes outside this window drive the smallest E2M1 magnitudes into the
// BF16 subnormal range or the largest into infinity/NaN. Experiment 0135
// limitation 1. The stimulus stays inside it and the runtime must admit on it.
std::uint32_t g_scale_minimum = 2U;
std::uint32_t g_scale_maximum = 250U;

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

class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t bytes) {
        check(cudaMalloc(&data_, bytes), "allocate FP4 ceiling probe buffer");
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

std::uint32_t xorshift(std::uint32_t& state) {
    state ^= state << 13U;
    state ^= state >> 17U;
    state ^= state << 5U;
    return state;
}

std::uint16_t bf16_bits(float value) {
    std::uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t rounded = bits + 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(rounded >> 16U);
}

float fp4_e2m1_value(std::uint8_t code) {
    constexpr std::array<float, 8U> magnitudes{
        0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    const float magnitude = magnitudes[code & 0x07U];
    return (code & 0x08U) != 0U && magnitude != 0.0F ? -magnitude : magnitude;
}

float e8m0_scale(std::uint8_t encoded) {
    return std::ldexp(1.0F, static_cast<int>(encoded) - 127);
}

// The exact decoder audited in experiment 0135: two E2M1 codes plus one E8M0
// exponent to a packed pair of BF16 values, entirely in registers.
__device__ __forceinline__ std::uint32_t
decode_e2m1_pair(std::uint32_t byte, std::uint32_t delta) {
    const std::uint32_t bits =
        ((byte & 0x08U) << 12U) | ((byte & 0x80U) << 24U) |
        ((byte & 0x07U) << 6U) | ((byte & 0x70U) << 18U);
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

__device__ __forceinline__ std::uint32_t scale_delta(std::uint32_t scale) {
    return ((scale - 1U) << 7U) * 0x0001'0001U;
}

// Arm 1: stream the same bytes with no decode. The ruler for this access
// pattern, so the decode arm's shortfall is attributable to the decoder.
__global__ void read_only_kernel(const uint4* __restrict__ codes,
                                 const unsigned char* __restrict__ scales,
                                 std::uint32_t groups, std::uint32_t* sink) {
    const std::uint32_t stride = gridDim.x * blockDim.x;
    std::uint32_t accumulator = 0U;
    for (std::uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
         group < groups; group += stride) {
        const uint4 packed = codes[group];
        accumulator ^= packed.x ^ packed.y ^ packed.z ^ packed.w;
        accumulator ^= scales[group];
    }
    if (accumulator == 0xFFFF'FFFFU) sink[0] = accumulator;
}

// Arm 2: read plus the full group-32 E2M1/E8M0 decode into operand registers.
__global__ void decode_kernel(const uint4* __restrict__ codes,
                              const unsigned char* __restrict__ scales,
                              std::uint32_t groups, std::uint32_t* sink) {
    const std::uint32_t stride = gridDim.x * blockDim.x;
    std::uint32_t accumulator = 0U;
    for (std::uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
         group < groups; group += stride) {
        const uint4 packed = codes[group];
        const std::uint32_t delta = scale_delta(scales[group]);
        const std::uint32_t words[4] = {packed.x, packed.y, packed.z,
                                        packed.w};
#pragma unroll
        for (std::uint32_t word = 0U; word < 4U; ++word) {
#pragma unroll
            for (std::uint32_t byte = 0U; byte < 4U; ++byte) {
                accumulator ^= decode_e2m1_pair(
                    (words[word] >> (byte * 8U)) & 0xFFU, delta);
            }
        }
    }
    if (accumulator == 0xFFFF'FFFFU) sink[0] = accumulator;
}

// Arm 3: the decode arm plus the native SM86 MMA that consumes it, using the
// lane/register contract verified in experiment 0135. This is a TIMING
// ceiling only: the arena is not in fragment order, so no correctness claim
// attaches to this arm. The fragment prepack is F4-1's remaining work.
__global__ void decode_mma_kernel(const uint4* __restrict__ codes,
                                  const unsigned char* __restrict__ scales,
                                  std::uint32_t groups, float* sink) {
    const std::uint32_t stride = gridDim.x * blockDim.x;
    float d0 = 0.0F;
    float d1 = 0.0F;
    float d2 = 0.0F;
    float d3 = 0.0F;
    const std::uint32_t b0 = 0x3F80'3F80U;
    const std::uint32_t b1 = 0x3F80'3F80U;
    for (std::uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
         group < groups; group += stride) {
        const uint4 packed = codes[group];
        const std::uint32_t delta = scale_delta(scales[group]);
        const std::uint32_t words[4] = {packed.x, packed.y, packed.z,
                                        packed.w};
#pragma unroll
        for (std::uint32_t word = 0U; word < 4U; ++word) {
            const std::uint32_t a0 =
                decode_e2m1_pair(words[word] & 0xFFU, delta);
            const std::uint32_t a1 =
                decode_e2m1_pair((words[word] >> 8U) & 0xFFU, delta);
            const std::uint32_t a2 =
                decode_e2m1_pair((words[word] >> 16U) & 0xFFU, delta);
            const std::uint32_t a3 =
                decode_e2m1_pair((words[word] >> 24U) & 0xFFU, delta);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    if (d0 + d1 + d2 + d3 == 12345.678F) sink[0] = d0;
}

// A cheaper successor decoder, screened against experiment 0136's measured
// budget. Two differences from the 0135 shift/rebias decoder:
//
//   1. The E2M1 magnitude is a table lookup done with PRMT rather than a
//      bit-twiddling reconstruction. The eight E2M1 magnitudes have only three
//      distinct BF16 high bytes and four distinct low bytes, so both halves of
//      the table fit in the eight source bytes one PRMT can index.
//   2. The E8M0 scale is applied as a native BF16 multiply instead of an
//      exponent add, which removes all of the zero-detection and
//      subnormal-masking machinery the additive trick needs, and extends the
//      valid scale window from codes 2-250 to 1-254.
//
// The nibble order is a load-time prepack choice, not a format change: the two
// codes of a BF16 pair sit 16 bits apart in the loaded word, so all four sign
// bits move into place with a single shift each. Experiment 0135's transferable
// thesis item 2 is exactly this permission.
__constant__ std::uint32_t kMagnitudeHigh[2] = {0x3F3F'3F00U, 0x4040'4040U};
__constant__ std::uint32_t kMagnitudeLow[2] = {0xC080'0000U, 0xC080'4000U};

__device__ __forceinline__ void decode_prmt_word(std::uint32_t word,
                                                 std::uint32_t scale_pair,
                                                 std::uint32_t (&out)[4]) {
    const std::uint32_t magnitudes = word & 0x7777'7777U;
    const std::uint32_t high_a =
        __byte_perm(kMagnitudeHigh[0], kMagnitudeHigh[1], magnitudes);
    const std::uint32_t low_a =
        __byte_perm(kMagnitudeLow[0], kMagnitudeLow[1], magnitudes);
    const std::uint32_t high_b = __byte_perm(
        kMagnitudeHigh[0], kMagnitudeHigh[1], magnitudes >> 16U);
    const std::uint32_t low_b =
        __byte_perm(kMagnitudeLow[0], kMagnitudeLow[1], magnitudes >> 16U);

    const std::uint32_t pair_a = __byte_perm(low_a, high_a, 0x5140U);
    const std::uint32_t quad_a = __byte_perm(low_a, high_a, 0x7362U);
    const std::uint32_t pair_b = __byte_perm(low_b, high_b, 0x5140U);
    const std::uint32_t quad_b = __byte_perm(low_b, high_b, 0x7362U);

    std::uint32_t value[4];
    value[0] = __byte_perm(pair_a, pair_b, 0x5410U);
    value[1] = __byte_perm(pair_a, pair_b, 0x7632U);
    value[2] = __byte_perm(quad_a, quad_b, 0x5410U);
    value[3] = __byte_perm(quad_a, quad_b, 0x7632U);

#pragma unroll
    for (std::uint32_t index = 0U; index < 4U; ++index) {
        // Code at nibble `index` pairs with the code at nibble `index + 4`,
        // so both sign bits reach bits 15 and 31 with one shared shift.
        const std::uint32_t signs =
            (word << (12U - index * 4U)) & 0x8000'8000U;
        // E2M1 code 0x8 is sign-set with magnitude zero. Applying its sign
        // would publish -0.0 where the reference oracle publishes +0.0, so the
        // sign is suppressed on zero magnitudes. Only the zero table entry has
        // a zero exponent field, and the largest entry plus 0x7F80 stays inside
        // its half, so one add carries into bit 15 exactly when the half is
        // non-zero and cannot carry across the half boundary.
        const std::uint32_t non_zero = value[index] + 0x7F80'7F80U;
        const __nv_bfloat162 scaled = __hmul2(
            *reinterpret_cast<const __nv_bfloat162*>(&value[index]),
            *reinterpret_cast<const __nv_bfloat162*>(&scale_pair));
        out[index] = *reinterpret_cast<const std::uint32_t*>(&scaled) ^
                     (signs & non_zero & 0x8000'8000U);
    }
}

// E8M0 code s means 2^(s-127), whose BF16 encoding is simply s in the exponent
// field with a zero mantissa. Broadcasting it into both halves costs one IMAD.
__device__ __forceinline__ std::uint32_t scale_pair_bf16(std::uint32_t scale) {
    return (scale << 7U) * 0x0001'0001U;
}

__global__ void decode_prmt_kernel(const uint4* __restrict__ codes,
                                   const unsigned char* __restrict__ scales,
                                   std::uint32_t groups, std::uint32_t* sink) {
    const std::uint32_t stride = gridDim.x * blockDim.x;
    std::uint32_t accumulator = 0U;
    for (std::uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
         group < groups; group += stride) {
        const uint4 packed = codes[group];
        const std::uint32_t scale = scale_pair_bf16(scales[group]);
        const std::uint32_t words[4] = {packed.x, packed.y, packed.z,
                                        packed.w};
#pragma unroll
        for (std::uint32_t word = 0U; word < 4U; ++word) {
            std::uint32_t out[4];
            decode_prmt_word(words[word], scale, out);
#pragma unroll
            for (std::uint32_t index = 0U; index < 4U; ++index) {
                accumulator ^= out[index];
            }
        }
    }
    if (accumulator == 0xFFFF'FFFFU) sink[0] = accumulator;
}

__global__ void oracle_prmt_kernel(const uint4* __restrict__ codes,
                                   const unsigned char* __restrict__ scales,
                                   std::uint32_t groups,
                                   std::uint32_t* decoded) {
    const std::uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups) return;
    const uint4 packed = codes[group];
    const std::uint32_t scale = scale_pair_bf16(scales[group]);
    const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (std::uint32_t word = 0U; word < 4U; ++word) {
        std::uint32_t out[4];
        decode_prmt_word(words[word], scale, out);
#pragma unroll
        for (std::uint32_t index = 0U; index < 4U; ++index) {
            decoded[group * 16U + word * 4U + index] = out[index];
        }
    }
}

// Arm 4: the decode arm with every byte decoded twice under two different
// deltas. DRAM traffic is byte-identical to arm 2; decoder ALU work is
// doubled. If the decode arm is ALU-bound this arm costs about twice as much;
// if it is DRAM-bound it costs about the same. This is the attribution test,
// not a candidate.
__global__ void decode_x2_kernel(const uint4* __restrict__ codes,
                                 const unsigned char* __restrict__ scales,
                                 std::uint32_t groups, std::uint32_t* sink) {
    const std::uint32_t stride = gridDim.x * blockDim.x;
    std::uint32_t accumulator = 0U;
    for (std::uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
         group < groups; group += stride) {
        const uint4 packed = codes[group];
        const std::uint32_t scale = scales[group];
        const std::uint32_t delta_a = scale_delta(scale);
        const std::uint32_t delta_b = scale_delta(scale ^ 1U);
        const std::uint32_t words[4] = {packed.x, packed.y, packed.z,
                                        packed.w};
#pragma unroll
        for (std::uint32_t word = 0U; word < 4U; ++word) {
#pragma unroll
            for (std::uint32_t byte = 0U; byte < 4U; ++byte) {
                const std::uint32_t code =
                    (words[word] >> (byte * 8U)) & 0xFFU;
                // The two codes must differ, or common-subexpression
                // elimination collapses the second decode to its final add
                // and the arm silently tests nothing.
                accumulator ^= decode_e2m1_pair(code, delta_a);
                accumulator ^= decode_e2m1_pair(code ^ 0x5AU, delta_b);
            }
        }
    }
    if (accumulator == 0xFFFF'FFFFU) sink[0] = accumulator;
}

__global__ void oracle_kernel(const uint4* __restrict__ codes,
                              const unsigned char* __restrict__ scales,
                              std::uint32_t groups, std::uint32_t* decoded) {
    const std::uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups) return;
    const uint4 packed = codes[group];
    const std::uint32_t delta = scale_delta(scales[group]);
    const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (std::uint32_t word = 0U; word < 4U; ++word) {
#pragma unroll
        for (std::uint32_t byte = 0U; byte < 4U; ++byte) {
            decoded[group * 16U + word * 4U + byte] = decode_e2m1_pair(
                (words[word] >> (byte * 8U)) & 0xFFU, delta);
        }
    }
}

__global__ void scrub_kernel(unsigned char* data, std::size_t bytes) {
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < bytes; index += stride) {
        data[index] = static_cast<unsigned char>(index);
    }
}

float median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}

struct ArmResult {
    const char* name;
    float cold_milliseconds{0.0F};
    float hot_milliseconds{0.0F};
    double cold_gigabytes_per_second{0.0};
    double hot_gigabytes_per_second{0.0};
    double derived_matrix_microseconds{0.0};
};

struct ShapeResult {
    const char* name;
    std::uint32_t n{0U};
    std::uint32_t k{0U};
    std::uint64_t useful_weight_bytes{0U};
    std::uint64_t code_bytes{0U};
    std::uint64_t scale_bytes{0U};
    std::uint64_t arena_bytes{0U};
    std::uint64_t swept_bytes{0U};
    std::uint32_t oracle_pairs{0U};
    std::uint32_t oracle_mismatches{0U};
    std::uint32_t successor_mismatches{0U};
    std::array<ArmResult, 5U> arms{};
};

ShapeResult run_shape(const Shape& shape, cudaStream_t stream,
                      const cudaDeviceProp& properties) {
    ShapeResult result{shape.name, shape.n, shape.k};
    result.code_bytes =
        static_cast<std::uint64_t>(shape.n) * (shape.k / 2U);
    result.scale_bytes =
        static_cast<std::uint64_t>(shape.n) * (shape.k / kGroupSize);
    result.useful_weight_bytes = result.code_bytes + result.scale_bytes;

    // One group is 32 weights: 16 packed code bytes (one uint4) and one E8M0
    // byte. So the uint4 count and the scale-byte count are both `groups`,
    // and a flat arena of kArenaReplicas copies indexes linearly.
    const std::uint32_t groups = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(shape.n) * shape.k / kGroupSize);
    const std::uint64_t replica_code_bytes = result.code_bytes;
    const std::uint64_t replica_scale_bytes = result.scale_bytes;
    result.arena_bytes =
        (replica_code_bytes + replica_scale_bytes) * kArenaReplicas;
    result.swept_bytes = result.useful_weight_bytes * kArenaReplicas;
    const std::uint32_t swept_groups = groups * kArenaReplicas;

    std::vector<unsigned char> host_codes(replica_code_bytes);
    std::vector<unsigned char> host_scales(replica_scale_bytes);
    std::uint32_t state = 0xA341'316CU ^ shape.n ^ (shape.k << 8U);
    for (auto& value : host_codes) {
        value = static_cast<unsigned char>(xorshift(state));
    }
    for (auto& value : host_scales) {
        value = static_cast<unsigned char>(
            g_scale_minimum +
            xorshift(state) % (g_scale_maximum - g_scale_minimum + 1U));
    }

    DeviceBuffer device_codes(replica_code_bytes * kArenaReplicas);
    DeviceBuffer device_scales(replica_scale_bytes * kArenaReplicas);
    DeviceBuffer device_scrub(kScrubBytes);
    DeviceBuffer device_sink(sizeof(float) * 4U);
    for (std::uint32_t replica = 0U; replica < kArenaReplicas; ++replica) {
        check(cudaMemcpyAsync(
                  static_cast<unsigned char*>(device_codes.get()) +
                      replica * replica_code_bytes,
                  host_codes.data(), replica_code_bytes,
                  cudaMemcpyHostToDevice, stream),
              "upload FP4 code arena replica");
        check(cudaMemcpyAsync(
                  static_cast<unsigned char*>(device_scales.get()) +
                      replica * replica_scale_bytes,
                  host_scales.data(), replica_scale_bytes,
                  cudaMemcpyHostToDevice, stream),
              "upload E8M0 scale arena replica");
    }
    check(cudaStreamSynchronize(stream), "finish arena upload");

    const std::uint32_t blocks =
        std::min<std::uint32_t>((swept_groups + kThreads - 1U) / kThreads,
                                static_cast<std::uint32_t>(
                                    properties.multiProcessorCount) * 16U);

    const auto arena_codes = [&]() {
        return reinterpret_cast<const uint4*>(
            static_cast<const unsigned char*>(device_codes.get()));
    };
    const auto arena_scales = [&]() {
        return static_cast<const unsigned char*>(device_scales.get());
    };

    const auto launch = [&](std::uint32_t arm) {
        if (arm == 0U) {
            read_only_kernel<<<blocks, kThreads, 0U, stream>>>(
                arena_codes(), arena_scales(), swept_groups,
                static_cast<std::uint32_t*>(device_sink.get()));
        } else if (arm == 1U) {
            decode_kernel<<<blocks, kThreads, 0U, stream>>>(
                arena_codes(), arena_scales(), swept_groups,
                static_cast<std::uint32_t*>(device_sink.get()));
        } else if (arm == 2U) {
            decode_mma_kernel<<<blocks, kThreads, 0U, stream>>>(
                arena_codes(), arena_scales(), swept_groups,
                static_cast<float*>(device_sink.get()));
        } else if (arm == 3U) {
            decode_x2_kernel<<<blocks, kThreads, 0U, stream>>>(
                arena_codes(), arena_scales(), swept_groups,
                static_cast<std::uint32_t*>(device_sink.get()));
        } else {
            decode_prmt_kernel<<<blocks, kThreads, 0U, stream>>>(
                arena_codes(), arena_scales(), swept_groups,
                static_cast<std::uint32_t*>(device_sink.get()));
        }
        check(cudaGetLastError(), "launch FP4 ceiling arm");
    };

    const auto scrub = [&] {
        scrub_kernel<<<1024U, 256U, 0U, stream>>>(
            static_cast<unsigned char*>(device_scrub.get()), kScrubBytes);
        check(cudaGetLastError(), "launch L2 scrub");
    };

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check(cudaEventCreate(&start), "create start event");
    check(cudaEventCreate(&stop), "create stop event");

    const auto time_arm = [&](std::uint32_t arm, bool cold) {
        if (cold) scrub();
        check(cudaEventRecord(start, stream), "record start");
        launch(arm);
        check(cudaEventRecord(stop, stream), "record stop");
        check(cudaEventSynchronize(stop), "synchronize sample");
        float milliseconds = 0.0F;
        check(cudaEventElapsedTime(&milliseconds, start, stop),
              "measure sample");
        return milliseconds;
    };

    for (std::uint32_t warmup = 0U; warmup < kWarmups; ++warmup) {
        for (std::uint32_t arm = 0U; arm < 5U; ++arm) {
            static_cast<void>(time_arm(arm, false));
        }
    }

    std::array<std::vector<float>, 5U> cold{};
    std::array<std::vector<float>, 5U> hot{};
    for (auto& samples : cold) samples.reserve(kSamples);
    for (auto& samples : hot) samples.reserve(kSamples);

    for (std::uint32_t sample = 0U; sample < kSamples; ++sample) {
        for (std::uint32_t arm = 0U; arm < 5U; ++arm) {
            cold[arm].push_back(time_arm(arm, true));
            hot[arm].push_back(time_arm(arm, false));
        }
    }

    static constexpr std::array<const char*, 5U> kArmNames{
        "read_only", "decode_0135_shift_rebias", "decode_mma",
        "decode_x2_attribution", "decode_prmt_lut_successor"};
    for (std::uint32_t arm = 0U; arm < 5U; ++arm) {
        ArmResult entry{kArmNames[arm]};
        entry.cold_milliseconds = median(std::move(cold[arm]));
        entry.hot_milliseconds = median(std::move(hot[arm]));
        entry.cold_gigabytes_per_second =
            static_cast<double>(result.swept_bytes) /
            (static_cast<double>(entry.cold_milliseconds) * 1.0e-3) / 1.0e9;
        entry.hot_gigabytes_per_second =
            static_cast<double>(result.swept_bytes) /
            (static_cast<double>(entry.hot_milliseconds) * 1.0e-3) / 1.0e9;
        // Derived, not measured: one production matrix pass at this rate. A
        // single-matrix launch cannot be timed here without reintroducing the
        // 40% launch deflation this arena size exists to remove.
        entry.derived_matrix_microseconds =
            static_cast<double>(entry.cold_milliseconds) * 1.0e3 /
            static_cast<double>(kArenaReplicas);
        result.arms[arm] = entry;
    }

    check(cudaEventDestroy(start), "destroy start event");
    check(cudaEventDestroy(stop), "destroy stop event");

    // Correctness: decode the first kOracleGroups groups on device and check
    // every produced BF16 pair against an independent host computation.
    const std::uint32_t oracle_groups = std::min(groups, kOracleGroups);
    DeviceBuffer device_decoded(
        static_cast<std::size_t>(oracle_groups) * 16U * sizeof(std::uint32_t));
    oracle_kernel<<<(oracle_groups + kThreads - 1U) / kThreads, kThreads, 0U,
                    stream>>>(arena_codes(), arena_scales(),
                              oracle_groups,
                              static_cast<std::uint32_t*>(
                                  device_decoded.get()));
    check(cudaGetLastError(), "launch FP4 decode oracle");
    std::vector<std::uint32_t> decoded(
        static_cast<std::size_t>(oracle_groups) * 16U);
    check(cudaMemcpyAsync(decoded.data(), device_decoded.get(),
                          decoded.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost, stream),
          "download FP4 decode oracle");
    check(cudaStreamSynchronize(stream), "finish FP4 decode oracle");

    DeviceBuffer device_prmt(
        static_cast<std::size_t>(oracle_groups) * 16U * sizeof(std::uint32_t));
    oracle_prmt_kernel<<<(oracle_groups + kThreads - 1U) / kThreads, kThreads,
                         0U, stream>>>(
        arena_codes(), arena_scales(), oracle_groups,
        static_cast<std::uint32_t*>(device_prmt.get()));
    check(cudaGetLastError(), "launch PRMT successor oracle");
    std::vector<std::uint32_t> prmt(
        static_cast<std::size_t>(oracle_groups) * 16U);
    check(cudaMemcpyAsync(prmt.data(), device_prmt.get(),
                          prmt.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost, stream),
          "download PRMT successor oracle");
    check(cudaStreamSynchronize(stream), "finish PRMT successor oracle");

    // The successor pairs the code at nibble j with the code at nibble j+4 of
    // the same 32-bit word. That pairing is a load-time prepack choice, so the
    // host oracle must reproduce it exactly rather than assume byte order.
    for (std::uint32_t group = 0U; group < oracle_groups; ++group) {
        const float scale = e8m0_scale(host_scales[group]);
        for (std::uint32_t word = 0U; word < 4U; ++word) {
            std::uint32_t packed = 0U;
            for (std::uint32_t byte = 0U; byte < 4U; ++byte) {
                packed |= static_cast<std::uint32_t>(
                              host_codes[group * 16U + word * 4U + byte])
                          << (byte * 8U);
            }
            for (std::uint32_t index = 0U; index < 4U; ++index) {
                const std::uint8_t low_code =
                    static_cast<std::uint8_t>((packed >> (index * 4U)) & 0xFU);
                const std::uint8_t high_code = static_cast<std::uint8_t>(
                    (packed >> ((index + 4U) * 4U)) & 0xFU);
                const std::uint32_t expected =
                    static_cast<std::uint32_t>(
                        bf16_bits(fp4_e2m1_value(low_code) * scale)) |
                    (static_cast<std::uint32_t>(
                         bf16_bits(fp4_e2m1_value(high_code) * scale))
                     << 16U);
                if (prmt[group * 16U + word * 4U + index] != expected) {
                    ++result.successor_mismatches;
                }
            }
        }
    }

    result.oracle_pairs = oracle_groups * 16U;
    for (std::uint32_t group = 0U; group < oracle_groups; ++group) {
        const float scale = e8m0_scale(host_scales[group]);
        for (std::uint32_t pair = 0U; pair < 16U; ++pair) {
            const std::uint8_t byte = host_codes[group * 16U + pair];
            const std::uint16_t low =
                bf16_bits(fp4_e2m1_value(byte & 0x0FU) * scale);
            const std::uint16_t high =
                bf16_bits(fp4_e2m1_value(byte >> 4U) * scale);
            const std::uint32_t expected =
                static_cast<std::uint32_t>(low) |
                (static_cast<std::uint32_t>(high) << 16U);
            if (decoded[group * 16U + pair] != expected) {
                ++result.oracle_mismatches;
            }
        }
    }
    return result;
}

void print_result(std::ostream& output, int device,
                  const cudaDeviceProp& properties,
                  const std::vector<ShapeResult>& shapes) {
    output << std::fixed << "{\n"
           << "  \"device_index\": " << device << ",\n"
           << "  \"device_name\": \"" << properties.name << "\",\n"
           << "  \"device_capability\": \"" << properties.major << "."
           << properties.minor << "\",\n"
           << "  \"milestone\": \"F4-1 phase A decode ceiling\",\n"
           << "  \"claim\": \"upper bound only: no fragment prepack, no "
              "activation feed, no output publication, no split-K\",\n"
           << "  \"warmups\": " << kWarmups << ",\n"
           << "  \"samples\": " << kSamples << ",\n"
           << "  \"arena_replicas\": " << kArenaReplicas << ",\n"
           << "  \"scrub_bytes\": " << kScrubBytes << ",\n"
           << "  \"e8m0_code_window\": [" << g_scale_minimum << ", "
           << g_scale_maximum << "],\n"
           << "  \"shapes\": [\n";
    for (std::size_t index = 0U; index < shapes.size(); ++index) {
        const ShapeResult& shape = shapes[index];
        output << "    {\n"
               << "      \"name\": \"" << shape.name << "\",\n"
               << "      \"n\": " << shape.n << ", \"k\": " << shape.k
               << ",\n"
               << "      \"useful_weight_bytes\": "
               << shape.useful_weight_bytes << ",\n"
               << "      \"code_bytes\": " << shape.code_bytes
               << ", \"scale_bytes\": " << shape.scale_bytes << ",\n"
               << "      \"arena_bytes\": " << shape.arena_bytes
               << ", \"swept_bytes\": " << shape.swept_bytes << ",\n"
               << "      \"oracle_pairs\": " << shape.oracle_pairs
               << ", \"oracle_mismatches\": " << shape.oracle_mismatches
               << ", \"successor_mismatches\": "
               << shape.successor_mismatches
               << ",\n"
               << "      \"arms\": [\n";
        for (std::size_t arm = 0U; arm < shape.arms.size(); ++arm) {
            const ArmResult& entry = shape.arms[arm];
            output << "        {\"name\": \"" << entry.name
                   << "\", \"cold_ms\": " << std::setprecision(6)
                   << entry.cold_milliseconds
                   << ", \"hot_ms\": " << entry.hot_milliseconds
                   << ", \"cold_gbps\": " << std::setprecision(2)
                   << entry.cold_gigabytes_per_second
                   << ", \"hot_gbps\": " << entry.hot_gigabytes_per_second
                   << ", \"derived_matrix_us\": " << std::setprecision(3)
                   << entry.derived_matrix_microseconds << "}" << (arm + 1U < shape.arms.size() ? ",\n" : "\n");
        }
        output << "      ]\n"
               << "    }" << (index + 1U < shapes.size() ? ",\n" : "\n");
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
            } else if (flag == "--scale-window" && index + 2 < argc) {
                // Experiment 0135 limitation 1: the additive decoder is only
                // valid for E8M0 codes 2-250. The successor applies the scale
                // as a multiply, so its valid window is an open question that
                // must be measured rather than asserted.
                g_scale_minimum =
                    static_cast<std::uint32_t>(std::stoul(argv[++index]));
                g_scale_maximum =
                    static_cast<std::uint32_t>(std::stoul(argv[++index]));
            } else {
                std::cerr << "usage: " << argv[0]
                          << " [--device INDEX] [--output PATH]\n";
                return EXIT_FAILURE;
            }
        }

        check(cudaSetDevice(device), "select FP4 ceiling probe device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device),
              "query FP4 ceiling probe device");
        if (properties.major != 8 || properties.minor != 6) {
            throw std::runtime_error("FP4 ceiling probe requires SM86");
        }

        cudaStream_t stream = nullptr;
        check(cudaStreamCreate(&stream), "create FP4 ceiling probe stream");
        std::vector<ShapeResult> shapes;
        shapes.reserve(std::size(kShapes));
        for (const Shape& shape : kShapes) {
            shapes.push_back(run_shape(shape, stream, properties));
        }
        check(cudaStreamDestroy(stream), "destroy FP4 ceiling probe stream");

        if (output_path.empty()) {
            print_result(std::cout, device, properties, shapes);
        } else {
            std::ofstream output(output_path);
            if (!output) {
                throw std::runtime_error("cannot open output path: " +
                                         output_path);
            }
            print_result(output, device, properties, shapes);
        }

        for (const ShapeResult& shape : shapes) {
            if (shape.oracle_mismatches != 0U) return EXIT_FAILURE;
            if (shape.successor_mismatches != 0U) return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
