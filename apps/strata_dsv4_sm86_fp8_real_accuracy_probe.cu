// Real-checkpoint/real-activation accuracy gate for the SM86 QPN8-derived
// W8A16 kernel. This is an experiment executable, not a production dispatch.

#include "strata/safetensors.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint32_t kWarp = 32U;
constexpr std::uint32_t kWarps = 4U;
constexpr std::uint32_t kThreads = kWarp * kWarps;

void check(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

class Buffer {
  public:
    explicit Buffer(std::size_t bytes) : bytes_(bytes) {
        check(cudaMalloc(&data_, bytes), "allocate real-accuracy buffer");
    }
    ~Buffer() {
        if (data_ != nullptr) static_cast<void>(cudaFree(data_));
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    [[nodiscard]] void* get() const noexcept { return data_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

  private:
    void* data_{};
    std::size_t bytes_{};
};

__host__ __device__ float bf16_value(std::uint16_t bits) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(static_cast<std::uint32_t>(bits) << 16U);
#else
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
#endif
}

std::uint16_t bf16_bits(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

__host__ __device__ float e4m3_value(std::uint8_t code) {
    const auto exponent = (code >> 3U) & 15U;
    const auto mantissa = code & 7U;
    if (exponent == 15U && mantissa == 7U) {
        return nanf("");
    }
    const float magnitude = exponent == 0U
        ? std::ldexp(static_cast<float>(mantissa), -9)
        : std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                     static_cast<int>(exponent) - 7);
    return (code & 0x80U) != 0U ? -magnitude : magnitude;
}

__host__ __device__ float e8m0_value(std::uint8_t code) {
    return code == 0xffU ? nanf("")
                         : ldexpf(1.0F, static_cast<int>(code) - 127);
}

__device__ __forceinline__ std::uint32_t decode_pair(
    std::uint32_t pair, std::uint32_t factor) {
    const auto unpacked = __byte_perm(pair, 0U, 0x4140U);
    const std::uint32_t values =
        ((unpacked << 8U) & 0x80008000U) |
        ((unpacked << 4U) & 0x07f007f0U);
    const auto result = __hmul2(
        *reinterpret_cast<const __nv_bfloat162*>(&values),
        *reinterpret_cast<const __nv_bfloat162*>(&factor));
    return *reinterpret_cast<const std::uint32_t*>(&result);
}

__device__ __forceinline__ void mma(
    float& d0, float& d1, float& d2, float& d3,
    std::uint32_t a0, std::uint32_t a1, std::uint32_t a2,
    std::uint32_t a3, std::uint32_t b0, std::uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ std::uint16_t round_bf16(float value) {
    std::uint32_t bits = __float_as_uint(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

__device__ __forceinline__ std::uint8_t packed_code_at(
    const uint4* codes, std::uint32_t row, std::uint32_t column,
    std::uint32_t k) {
    const std::uint32_t nt = row / 16U;
    const std::uint32_t local_row = row & 15U;
    const std::uint32_t kp = column / 32U;
    const std::uint32_t within = column & 31U;
    const std::uint32_t half = within / 16U;
    const std::uint32_t remainder = within & 15U;
    const std::uint32_t lane = (local_row & 7U) * 4U +
                               ((remainder & 7U) / 2U);
    const std::uint32_t index = (local_row >= 8U ? 2U : 0U) +
                                (remainder >= 8U ? 4U : 0U) +
                                (remainder & 1U);
    const std::uint32_t word = half * 2U + (remainder >= 8U ? 1U : 0U);
    const auto packed = codes[(static_cast<std::size_t>(nt) * (k / 32U) + kp) *
                              kWarp + lane];
    const auto* words = reinterpret_cast<const std::uint32_t*>(&packed);
    return static_cast<std::uint8_t>(words[word] >> ((index & 3U) * 8U));
}

template<int Split, bool Grouped, int Accs = 2, bool Replay = true>
__global__ __launch_bounds__(kThreads) void qpn8_accuracy_kernel(
    const uint4* codes, const std::uint8_t* scales,
    const std::uint16_t* input, float* partials, std::uint32_t* counters,
    std::uint16_t* output, float* raw_output, std::uint32_t* fallback_count,
    std::uint32_t n, std::uint32_t k) {
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint32_t work = blockIdx.x * kWarps + warp;
    const std::uint32_t works = (n / 16U) * Split;
    if (work >= works) return;
    const std::uint32_t nt = work / Split;
    const std::uint32_t slice = work % Split;
    const std::uint32_t kpairs = k / 32U;
    const std::uint32_t pairs_per_slice = kpairs / Split;
    const std::uint32_t pbegin = slice * pairs_per_slice;
    const std::uint32_t pend = pbegin + pairs_per_slice;
    const std::uint32_t group = lane >> 2U;
    const std::uint32_t thread = lane & 3U;
    const std::uint32_t input_group = Grouped ? nt / 64U : 0U;
    input += static_cast<std::size_t>(input_group) * k;
    float accum[Accs][4]{};
    for (std::uint32_t pb = pbegin; pb < pend; pb += 4U) {
        const std::uint32_t kt_begin = pb * 2U;
        std::uint32_t scale = lane == 0U
            ? scales[(nt / 8U) * (k / 128U) + kt_begin / 8U]
            : 0U;
        scale = __shfl_sync(0xffffffffU, scale, 0);
        const std::uint32_t factor = ((scale + 120U) << 7U) * 0x00010001U;
        uint4 packed[4]{};
#pragma unroll
        for (int unroll = 0; unroll < 4; ++unroll) {
            if (pb + static_cast<std::uint32_t>(unroll) < pend) {
                packed[unroll] =
                    codes[(static_cast<std::size_t>(nt) * kpairs + pb +
                           static_cast<std::uint32_t>(unroll)) * kWarp + lane];
            }
        }
#pragma unroll
        for (int unroll = 0; unroll < 4; ++unroll) {
            if (pb + static_cast<std::uint32_t>(unroll) >= pend) continue;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const std::uint32_t low = half == 0
                    ? packed[unroll].x : packed[unroll].z;
                const std::uint32_t high = half == 0
                    ? packed[unroll].y : packed[unroll].w;
                const auto a0 = decode_pair(low & 0xffffU, factor);
                const auto a1 = decode_pair(low >> 16U, factor);
                const auto a2 = decode_pair(high & 0xffffU, factor);
                const auto a3 = decode_pair(high >> 16U, factor);
                std::uint32_t b0 = 0U;
                std::uint32_t b1 = 0U;
                if (group == 0U) {
                    const std::uint32_t kt = kt_begin +
                        static_cast<std::uint32_t>(unroll) * 2U +
                        static_cast<std::uint32_t>(half);
                    const auto* base = input + kt * 16U + thread * 2U;
                    b0 = *reinterpret_cast<const std::uint32_t*>(base);
                    b1 = *reinterpret_cast<const std::uint32_t*>(base + 8U);
                }
                auto& accumulator = accum[(pb * 2U +
                    static_cast<std::uint32_t>(unroll) * 2U +
                    static_cast<std::uint32_t>(half)) % Accs];
                mma(accumulator[0], accumulator[1], accumulator[2],
                    accumulator[3], a0, a1, a2, a3, b0, b1);
            }
        }
    }
#pragma unroll
    for (int step = 1; step < Accs; step *= 2) {
#pragma unroll
        for (int base = 0; base < Accs; base += step * 2) {
#pragma unroll
            for (int reg = 0; reg < 4; ++reg) {
                accum[base][reg] += accum[base + step][reg];
            }
        }
    }
    auto* destination = partials +
        (static_cast<std::size_t>(nt) * Split + slice) * 128U + lane * 4U;
#pragma unroll
    for (int index = 0; index < 4; ++index) destination[index] = accum[0][index];
    __threadfence();
    __shared__ std::uint32_t arrived[kWarps];
    if (lane == 0U) arrived[warp] = atomicAdd(counters + nt, 1U);
    __syncwarp();
    if (arrived[warp] != Split - 1U) return;
    float fast_value = 0.0F;
    std::uint32_t midpoint_distance = 0xffffffffU;
    if (lane < 16U) {
        const std::uint32_t row = lane;
        const std::uint32_t source_lane = (row & 7U) * 4U;
        const std::uint32_t reg = (row >> 3U) * 2U;
        double sum = 0.0;
#pragma unroll
        for (std::uint32_t split = 0U; split < Split; ++split) {
            sum += partials[(static_cast<std::size_t>(nt) * Split + split) *
                            128U + source_lane * 4U + reg];
        }
        fast_value = static_cast<float>(sum);
        output[nt * 16U + row] = round_bf16(fast_value);
        raw_output[nt * 16U + row] = fast_value;
        const auto tail = __float_as_uint(fast_value) & 0xffffU;
        midpoint_distance = tail > 0x8000U ? tail - 0x8000U
                                            : 0x8000U - tail;
    }
    // HMMA is the fast path. Only sums close enough to a BF16 rounding
    // midpoint to make reassociation observable are replayed. The replay uses
    // the same invertibly prepacked bytes and a warp-voted FP64 dot; it is not
    // a widened weight copy or a precision fallback.
    std::uint32_t ambiguous = Replay ? __ballot_sync(
        0xffffffffU, lane < 16U &&
        (midpoint_distance <= (Grouped ? 1024U : 512U) ||
         (fast_value != 0.0F && fabsf(fast_value) <= 1.0e-6F))) : 0U;
    while (ambiguous != 0U) {
        const std::uint32_t owner = __ffs(ambiguous) - 1U;
        const std::uint32_t logical_row = nt * 16U + owner;
        double exact = 0.0;
        for (std::uint32_t column = lane; column < k; column += kWarp) {
            const auto code = packed_code_at(codes, logical_row, column, k);
            const auto scale = scales[
                static_cast<std::size_t>(logical_row / 128U) * (k / 128U) +
                column / 128U];
            exact += static_cast<double>(bf16_value(input[column])) *
                     static_cast<double>(e4m3_value(code)) *
                     static_cast<double>(e8m0_value(scale));
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            exact += __shfl_down_sync(0xffffffffU, exact, offset);
        }
        if (lane == 0U) {
            output[logical_row] = round_bf16(static_cast<float>(exact));
            atomicAdd(fallback_count, 1U);
        }
        ambiguous &= ambiguous - 1U;
    }
    if (lane == 0U) counters[nt] = 0U;
}

__device__ float warp_reduce(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__global__ void incumbent_w8a16_kernel(
    float* output, const float* input, const std::uint8_t* weights,
    const std::uint8_t* scales, std::uint32_t n, std::uint32_t k,
    bool grouped) {
    const std::uint32_t row = blockIdx.x;
    if (row >= n) return;
    const std::uint32_t input_group = grouped ? row / 1024U : 0U;
    const auto* x = input + static_cast<std::size_t>(input_group) * k;
    const auto* w = weights + static_cast<std::size_t>(row) * k;
    float sum = 0.0F;
    for (std::uint32_t column = threadIdx.x; column < k;
         column += blockDim.x) {
        sum += x[column] * e4m3_value(w[column]) *
               e8m0_value(scales[(row / 128U) * (k / 128U) +
                                  column / 128U]);
    }
    sum = warp_reduce(sum);
    __shared__ float warp_sums[8];
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    if (lane == 0U) warp_sums[warp] = sum;
    __syncthreads();
    sum = threadIdx.x < 8U ? warp_sums[lane] : 0.0F;
    if (warp == 0U) sum = warp_reduce(sum);
    if (threadIdx.x == 0U) output[row] = bf16_value(round_bf16(sum));
}

struct Matrix {
    std::string name;
    std::uint32_t n{};
    std::uint32_t k{};
    std::vector<std::uint8_t> canonical;
    std::vector<std::uint8_t> scales;
    std::vector<uint4> packed;

    void pack() {
        packed.resize(static_cast<std::size_t>(n / 16U) * (k / 32U) * kWarp);
        for (std::uint32_t nt = 0U; nt < n / 16U; ++nt) {
            for (std::uint32_t kp = 0U; kp < k / 32U; ++kp) {
                for (std::uint32_t lane = 0U; lane < kWarp; ++lane) {
                    std::uint32_t words[4]{};
                    for (std::uint32_t half = 0U; half < 2U; ++half) {
                        for (std::uint32_t index = 0U; index < 8U; ++index) {
                            const bool high = index == 2U || index == 3U ||
                                              index == 6U || index == 7U;
                            const std::uint32_t row = nt * 16U + (lane >> 2U) +
                                                      (high ? 8U : 0U);
                            const std::uint32_t column =
                                (kp * 2U + half) * 16U + (lane & 3U) * 2U +
                                (index & 1U) + (index >= 4U ? 8U : 0U);
                            words[half * 2U + (index >= 4U ? 1U : 0U)] |=
                                static_cast<std::uint32_t>(canonical[
                                    static_cast<std::size_t>(row) * k + column])
                                << ((index & 3U) * 8U);
                        }
                    }
                    packed[(static_cast<std::size_t>(nt) * (k / 32U) + kp) *
                           kWarp + lane] =
                        make_uint4(words[0], words[1], words[2], words[3]);
                }
            }
        }
    }
};

const strata::SafetensorsTensor& find_tensor(
    const strata::SafetensorsShard& shard, std::string_view name) {
    const auto iterator = std::find_if(
        shard.tensors.begin(), shard.tensors.end(),
        [name](const auto& tensor) { return tensor.name == name; });
    if (iterator == shard.tensors.end()) {
        throw std::runtime_error("checkpoint tensor is absent: " +
                                 std::string(name));
    }
    return *iterator;
}

std::vector<std::uint8_t> read_tensor(
    const std::string& path, const strata::SafetensorsTensor& tensor) {
    auto result = strata::read_safetensors_tensor(path, tensor, tensor.bytes());
    if (!result.ok()) {
        throw std::runtime_error("cannot read " + tensor.name + ": " +
                                 result.errors.front());
    }
    std::vector<std::uint8_t> bytes(result.value.size());
    std::memcpy(bytes.data(), result.value.data(), result.value.size());
    return bytes;
}

Matrix load_matrix(const std::string& path,
                   const strata::SafetensorsShard& shard,
                   std::string name) {
    const auto& weight = find_tensor(shard, name + ".weight");
    const auto& scale = find_tensor(shard, name + ".scale");
    if (weight.dtype != strata::SafetensorsDtype::F8E4M3 ||
        scale.dtype != strata::SafetensorsDtype::F8E8M0 ||
        weight.shape.size() != 2U || scale.shape.size() != 2U ||
        weight.shape[0] % 128U != 0U || weight.shape[1] % 128U != 0U ||
        scale.shape[0] != weight.shape[0] / 128U ||
        scale.shape[1] != weight.shape[1] / 128U) {
        throw std::runtime_error("incompatible E4M3/E8M0 matrix: " + name);
    }
    Matrix matrix;
    matrix.name = std::move(name);
    matrix.n = static_cast<std::uint32_t>(weight.shape[0]);
    matrix.k = static_cast<std::uint32_t>(weight.shape[1]);
    matrix.canonical = read_tensor(path, weight);
    matrix.scales = read_tensor(path, scale);
    matrix.pack();
    return matrix;
}

std::vector<std::uint16_t> load_bf16_vector(
    const std::string& path, const strata::SafetensorsShard& shard,
    std::string_view name, std::size_t elements) {
    const auto& tensor = find_tensor(shard, name);
    if (tensor.dtype != strata::SafetensorsDtype::Bf16 ||
        tensor.shape.size() != 1U || tensor.shape[0] != elements) {
        throw std::runtime_error("incompatible BF16 vector: " +
                                 std::string(name));
    }
    const auto bytes = read_tensor(path, tensor);
    std::vector<std::uint16_t> result(elements);
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
}

std::vector<std::uint16_t> normalize_query_rank(
    std::span<const std::uint16_t> input,
    std::span<const std::uint16_t> weight, bool exact) {
    double squared_sum = 0.0;
    if (exact) {
        for (const auto bits : input) {
            const double value = bf16_value(bits);
            squared_sum += value * value;
        }
    } else {
        std::array<float, kWarp> partial{};
        for (std::uint32_t lane = 0U; lane < kWarp; ++lane) {
            for (std::uint32_t index = lane; index < input.size();
                 index += kWarp) {
                const float value = bf16_value(input[index]);
                partial[lane] += value * value;
            }
        }
        for (std::uint32_t offset = kWarp / 2U; offset != 0U; offset >>= 1U) {
            for (std::uint32_t lane = 0U; lane < offset; ++lane) {
                partial[lane] += partial[lane + offset];
            }
        }
        squared_sum = partial[0];
    }
    const float inverse = 1.0F / std::sqrt(
        static_cast<float>(squared_sum / input.size()) + 1.0e-6F);
    std::vector<std::uint16_t> output(input.size());
    for (std::size_t index = 0U; index < input.size(); ++index) {
        output[index] = bf16_bits(
            bf16_value(input[index]) * inverse * bf16_value(weight[index]));
    }
    return output;
}

std::uint64_t mismatch_count(std::span<const std::uint16_t> left,
                             std::span<const std::uint16_t> right) {
    std::uint64_t result = 0U;
    for (std::size_t index = 0U; index < left.size(); ++index) {
        if (left[index] != right[index]) ++result;
    }
    return result;
}

Matrix concatenate(Matrix main, const Matrix& suffix) {
    if (main.k != suffix.k) throw std::runtime_error("fusion K mismatch");
    main.name += "+" + suffix.name;
    main.n += suffix.n;
    main.canonical.insert(main.canonical.end(), suffix.canonical.begin(),
                          suffix.canonical.end());
    main.scales.insert(main.scales.end(), suffix.scales.begin(),
                       suffix.scales.end());
    main.pack();
    return main;
}

struct Activations {
    std::uint32_t layer{};
    std::vector<std::uint16_t> hidden;
    std::vector<std::uint16_t> query_rank;
    std::vector<std::uint16_t> attention;
};

template<class T, std::size_t Extent>
void read_exact(std::ifstream& input, std::span<T, Extent> output,
                std::string_view field) {
    if (!input.read(reinterpret_cast<char*>(output.data()),
                    static_cast<std::streamsize>(output.size_bytes()))) {
        throw std::runtime_error("truncated activation fixture at " +
                                 std::string(field));
    }
}

Activations load_activations(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open activation fixture");
    std::array<char, 8> magic{};
    std::array<std::uint32_t, 2> header{};
    read_exact(input, std::span(magic), "magic");
    read_exact(input, std::span(header), "header");
    constexpr std::array<char, 8> expected{
        'D', '4', 'F', '8', 'A', '0', '1', '\0'};
    if (magic != expected || header[0] != 1U) {
        throw std::runtime_error("incompatible activation fixture");
    }
    Activations result;
    result.layer = header[1];
    result.hidden.resize(4096U);
    result.query_rank.resize(1024U);
    result.attention.resize(32768U);
    read_exact(input, std::span(result.hidden), "hidden");
    read_exact(input, std::span(result.query_rank), "query rank");
    read_exact(input, std::span(result.attention), "attention");
    char trailing{};
    if (input.read(&trailing, 1)) {
        throw std::runtime_error("activation fixture contains trailing bytes");
    }
    return result;
}

struct Metrics {
    double maximum_absolute{};
    double maximum_relative{};
    double rms{};
    std::uint64_t mismatches{};
    std::uint32_t maximum_mismatch_midpoint_distance{};
    double maximum_mismatch_raw_magnitude{};
};

Metrics measure(const Matrix& matrix, std::span<const std::uint16_t> input,
                std::span<const std::uint16_t> output, bool grouped,
                std::span<const float> raw = {}) {
    long double squared = 0.0L;
    Metrics result;
    for (std::uint32_t row = 0U; row < matrix.n; ++row) {
        const std::uint32_t group = grouped ? row / 1024U : 0U;
        double oracle = 0.0;
        for (std::uint32_t column = 0U; column < matrix.k; ++column) {
            oracle += static_cast<double>(bf16_value(
                          input[static_cast<std::size_t>(group) * matrix.k +
                                column])) *
                      static_cast<double>(e4m3_value(matrix.canonical[
                          static_cast<std::size_t>(row) * matrix.k + column])) *
                      static_cast<double>(e8m0_value(matrix.scales[
                          static_cast<std::size_t>(row / 128U) *
                              (matrix.k / 128U) + column / 128U]));
        }
        const auto expected = bf16_bits(static_cast<float>(oracle));
        const double expected_value = bf16_value(expected);
        const double actual_value = bf16_value(output[row]);
        const double absolute = std::abs(actual_value - expected_value);
        result.maximum_absolute = std::max(result.maximum_absolute, absolute);
        result.maximum_relative = std::max(
            result.maximum_relative,
            absolute / std::max(std::abs(expected_value), 1.0e-9));
        squared += static_cast<long double>(absolute) * absolute;
        if (output[row] != expected) {
            ++result.mismatches;
            if (!raw.empty()) {
                const auto tail = std::bit_cast<std::uint32_t>(raw[row]) & 0xffffU;
                const auto distance = tail > 0x8000U ? tail - 0x8000U
                                                     : 0x8000U - tail;
                result.maximum_mismatch_midpoint_distance = std::max(
                    result.maximum_mismatch_midpoint_distance, distance);
                result.maximum_mismatch_raw_magnitude = std::max(
                    result.maximum_mismatch_raw_magnitude,
                    std::abs(static_cast<double>(raw[row])));
            }
        }
    }
    result.rms = std::sqrt(static_cast<double>(squared / matrix.n));
    return result;
}

struct Comparison {
    Metrics incumbent;
    Metrics candidate;
    std::uint64_t candidate_incumbent_mismatches{};
    bool no_worse{};
    std::vector<std::uint16_t> incumbent_output;
    std::vector<std::uint16_t> candidate_output;
    std::vector<float> candidate_raw;
    std::uint32_t maximum_changed_midpoint_distance{};
    std::uint32_t fallback_rows{};
    std::uint32_t near_zero_fast_rows{};
};

template<int Split, bool Grouped, int Accs = 2, bool Replay = true>
Comparison run_matrix(const Matrix& matrix,
                      std::span<const std::uint16_t> input);

Comparison run_ungrouped(const Matrix& matrix,
                         std::span<const std::uint16_t> input,
                         std::uint32_t split) {
    switch (split) {
        case 1U: return run_matrix<1, false>(matrix, input);
        case 2U: return run_matrix<2, false>(matrix, input);
        case 4U: return run_matrix<4, false>(matrix, input);
        case 8U: return run_matrix<8, false>(matrix, input);
        case 16U: return run_matrix<16, false>(matrix, input);
        case 32U: return run_matrix<32, false>(matrix, input);
        default: throw std::runtime_error("unsupported accuracy split");
    }
}

Comparison run_grouped(const Matrix& matrix,
                       std::span<const std::uint16_t> input,
                       std::uint32_t split) {
    switch (split) {
        case 1U: return run_matrix<1, true>(matrix, input);
        case 2U: return run_matrix<2, true>(matrix, input);
        case 4U: return run_matrix<4, true>(matrix, input);
        case 8U: return run_matrix<8, true>(matrix, input);
        case 16U: return run_matrix<16, true>(matrix, input);
        case 32U: return run_matrix<32, true>(matrix, input);
        default: throw std::runtime_error("unsupported grouped accuracy split");
    }
}

Comparison run_nacc8_ungrouped(const Matrix& matrix,
                               std::span<const std::uint16_t> input,
                               std::uint32_t split) {
    if (split != 1U) throw std::runtime_error("NACC8 query requires split 1");
    return run_matrix<1, false, 8, false>(matrix, input);
}

Comparison run_nacc8_grouped(const Matrix& matrix,
                             std::span<const std::uint16_t> input,
                             std::uint32_t split) {
    if (split != 4U) throw std::runtime_error("NACC8 grouped output requires split 4");
    return run_matrix<4, true, 8, false>(matrix, input);
}

Comparison run_nacc16_ungrouped(const Matrix& matrix,
                                std::span<const std::uint16_t> input,
                                std::uint32_t split) {
    if (split != 1U) throw std::runtime_error("NACC16 query requires split 1");
    return run_matrix<1, false, 16, false>(matrix, input);
}

Comparison run_nacc16_grouped(const Matrix& matrix,
                              std::span<const std::uint16_t> input,
                              std::uint32_t split) {
    if (split != 4U) throw std::runtime_error("NACC16 grouped output requires split 4");
    return run_matrix<4, true, 16, false>(matrix, input);
}

template<int Split, bool Grouped, int Accs, bool Replay>
Comparison run_matrix(const Matrix& matrix,
                      std::span<const std::uint16_t> input) {
    const std::size_t partial_bytes =
        static_cast<std::size_t>(matrix.n / 16U) * Split * 128U * sizeof(float);
    Buffer codes(matrix.canonical.size());
    Buffer scales(matrix.scales.size());
    Buffer bf16_input(input.size_bytes());
    Buffer float_input(input.size() * sizeof(float));
    Buffer partials(partial_bytes);
    Buffer counters(static_cast<std::size_t>(matrix.n / 16U) *
                    sizeof(std::uint32_t));
    Buffer candidate_output(static_cast<std::size_t>(matrix.n) *
                            sizeof(std::uint16_t));
    Buffer candidate_raw(static_cast<std::size_t>(matrix.n) * sizeof(float));
    Buffer fallback_count(sizeof(std::uint32_t));
    Buffer incumbent_output(static_cast<std::size_t>(matrix.n) * sizeof(float));
    std::vector<float> expanded(input.size());
    std::transform(input.begin(), input.end(), expanded.begin(), bf16_value);
    check(cudaMemcpy(codes.get(), matrix.packed.data(), codes.bytes(),
                     cudaMemcpyHostToDevice), "upload packed checkpoint weights");
    check(cudaMemcpy(scales.get(), matrix.scales.data(), scales.bytes(),
                     cudaMemcpyHostToDevice), "upload checkpoint scales");
    check(cudaMemcpy(bf16_input.get(), input.data(), bf16_input.bytes(),
                     cudaMemcpyHostToDevice), "upload BF16 fixture");
    check(cudaMemcpy(float_input.get(), expanded.data(), float_input.bytes(),
                     cudaMemcpyHostToDevice), "upload incumbent fixture");
    check(cudaMemset(counters.get(), 0, counters.bytes()), "clear counters");
    check(cudaMemset(fallback_count.get(), 0, fallback_count.bytes()),
          "clear ambiguity count");
    const std::uint32_t works = (matrix.n / 16U) * Split;
    qpn8_accuracy_kernel<Split, Grouped, Accs, Replay>
        <<<(works + kWarps - 1U) / kWarps, kThreads>>>(
            static_cast<const uint4*>(codes.get()),
            static_cast<const std::uint8_t*>(scales.get()),
            static_cast<const std::uint16_t*>(bf16_input.get()),
            static_cast<float*>(partials.get()),
            static_cast<std::uint32_t*>(counters.get()),
            static_cast<std::uint16_t*>(candidate_output.get()),
            static_cast<float*>(candidate_raw.get()),
            static_cast<std::uint32_t*>(fallback_count.get()), matrix.n,
            matrix.k);
    // The incumbent consumes canonical checkpoint order, not fragment order.
    check(cudaMemcpy(codes.get(), matrix.canonical.data(), codes.bytes(),
                     cudaMemcpyHostToDevice), "restore canonical incumbent weights");
    incumbent_w8a16_kernel<<<matrix.n, 256U>>>(
        static_cast<float*>(incumbent_output.get()),
        static_cast<const float*>(float_input.get()),
        static_cast<const std::uint8_t*>(codes.get()),
        static_cast<const std::uint8_t*>(scales.get()), matrix.n, matrix.k,
        Grouped);
    check(cudaGetLastError(), "launch real-accuracy kernels");
    check(cudaDeviceSynchronize(), "synchronize real-accuracy kernels");
    std::vector<std::uint16_t> candidate(matrix.n);
    std::vector<float> raw(matrix.n);
    std::vector<float> incumbent_float(matrix.n);
    check(cudaMemcpy(candidate.data(), candidate_output.get(),
                     candidate_output.bytes(), cudaMemcpyDeviceToHost),
          "download candidate output");
    check(cudaMemcpy(incumbent_float.data(), incumbent_output.get(),
                     incumbent_output.bytes(), cudaMemcpyDeviceToHost),
          "download incumbent output");
    check(cudaMemcpy(raw.data(), candidate_raw.get(), candidate_raw.bytes(),
                     cudaMemcpyDeviceToHost), "download candidate FP32 sums");
    Comparison result;
    result.incumbent_output.resize(matrix.n);
    result.candidate_output = candidate;
    result.candidate_raw = raw;
    result.near_zero_fast_rows = static_cast<std::uint32_t>(std::count_if(
        raw.begin(), raw.end(), [](float value) {
            return value != 0.0F && std::abs(value) <= 1.0e-6F;
        }));
    check(cudaMemcpy(&result.fallback_rows, fallback_count.get(),
                     sizeof(result.fallback_rows), cudaMemcpyDeviceToHost),
          "download ambiguity count");
    std::transform(incumbent_float.begin(), incumbent_float.end(),
                   result.incumbent_output.begin(), bf16_bits);
    result.incumbent = measure(matrix, input, result.incumbent_output, Grouped);
    result.candidate = measure(matrix, input, candidate, Grouped, raw);
    for (std::uint32_t row = 0U; row < matrix.n; ++row) {
        if (candidate[row] != result.incumbent_output[row]) {
            ++result.candidate_incumbent_mismatches;
            const auto tail = std::bit_cast<std::uint32_t>(raw[row]) & 0xffffU;
            const auto distance = tail > 0x8000U ? tail - 0x8000U
                                                 : 0x8000U - tail;
            result.maximum_changed_midpoint_distance = std::max(
                result.maximum_changed_midpoint_distance, distance);
        }
    }
    result.no_worse =
        result.candidate.maximum_absolute <= result.incumbent.maximum_absolute &&
        result.candidate.maximum_relative <= result.incumbent.maximum_relative &&
        result.candidate.rms <= result.incumbent.rms &&
        result.candidate.mismatches <= result.incumbent.mismatches;
    return result;
}

void print_metrics(std::ostream& output, std::string_view name,
                   const Comparison& result, bool comma) {
    const auto emit = [&](std::string_view path, const Metrics& metrics) {
        output << "\"" << path << "\":{\"maximum_absolute\":"
               << metrics.maximum_absolute << ",\"maximum_relative\":"
               << metrics.maximum_relative << ",\"rms\":" << metrics.rms
               << ",\"oracle_mismatches\":" << metrics.mismatches
               << ",\"maximum_mismatch_midpoint_distance_fp32_ulp\":"
               << metrics.maximum_mismatch_midpoint_distance
               << ",\"maximum_mismatch_raw_magnitude\":"
               << metrics.maximum_mismatch_raw_magnitude << '}';
    };
    output << "    \"" << name << "\":{";
    emit("incumbent", result.incumbent);
    output << ',';
    emit("candidate", result.candidate);
    output << ",\"candidate_incumbent_mismatches\":"
           << result.candidate_incumbent_mismatches
           << ",\"maximum_changed_midpoint_distance_fp32_ulp\":"
           << result.maximum_changed_midpoint_distance
           << ",\"fp64_replay_rows\":" << result.fallback_rows
           << ",\"near_zero_fast_rows_1e_6\":" << result.near_zero_fast_rows
           << ",\"no_worse\":" << (result.no_worse ? "true" : "false")
           << '}' << (comma ? ",\n" : "\n");
}

struct Options {
    int device{};
    std::uint32_t layer{2U};
    std::string shard;
    std::string activations;
    std::string output;
    std::uint32_t qb_split{1U};
    std::uint32_t oa_split{4U};
    bool nacc8_no_replay{};
    bool nacc16_no_replay{};
};

Options parse(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto value = [&]() -> std::string {
            if (++index >= argc) throw std::runtime_error("missing option value");
            return argv[index];
        };
        if (argument == "--device") options.device = std::stoi(value());
        else if (argument == "--layer") options.layer = std::stoul(value());
        else if (argument == "--shard") options.shard = value();
        else if (argument == "--activations") options.activations = value();
        else if (argument == "--output") options.output = value();
        else if (argument == "--qb-split") options.qb_split = std::stoul(value());
        else if (argument == "--oa-split") options.oa_split = std::stoul(value());
        else if (argument == "--nacc8-no-replay") options.nacc8_no_replay = true;
        else if (argument == "--nacc16-no-replay") options.nacc16_no_replay = true;
        else throw std::runtime_error("unknown option: " + std::string(argument));
    }
    if (options.shard.empty() || options.activations.empty()) {
        throw std::runtime_error(
            "usage: probe --layer N --shard FILE --activations FILE [--output FILE]");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        check(cudaSetDevice(options.device), "select SM86 device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, options.device),
              "read device properties");
        if (properties.major != 8 || properties.minor != 6) {
            throw std::runtime_error("real-accuracy probe requires SM86");
        }
        const auto activations = load_activations(options.activations);
        if (activations.layer != options.layer) {
            throw std::runtime_error("activation/checkpoint layer mismatch");
        }
        auto shard_result = strata::load_safetensors_shard(options.shard);
        if (!shard_result.ok()) {
            throw std::runtime_error("cannot load shard: " +
                                     shard_result.errors.front());
        }
        const std::string prefix =
            "layers." + std::to_string(options.layer) + ".attn.";
        auto qa = load_matrix(options.shard, shard_result.value, prefix + "wq_a");
        const auto query_norm = load_bf16_vector(
            options.shard, shard_result.value, prefix + "q_norm.weight", 1024U);
        auto qb = load_matrix(options.shard, shard_result.value, prefix + "wq_b");
        const bool has_indexer = std::any_of(
            shard_result.value.tensors.begin(), shard_result.value.tensors.end(),
            [&](const auto& tensor) {
                return tensor.name == prefix + "indexer.wq_b.weight";
            });
        if (has_indexer) {
            qb = concatenate(std::move(qb), load_matrix(
                options.shard, shard_result.value, prefix + "indexer.wq_b"));
        }
        auto kv = load_matrix(options.shard, shard_result.value, prefix + "wkv");
        auto oa = load_matrix(options.shard, shard_result.value, prefix + "wo_a");
        auto ob = load_matrix(options.shard, shard_result.value, prefix + "wo_b");
        const auto qa_result = run_matrix<16, false>(qa, activations.hidden);
        const auto exact_query_rank = normalize_query_rank(
            qa_result.candidate_output, query_norm, true);
        const auto warp_query_rank = normalize_query_rank(
            qa_result.candidate_output, query_norm, false);
        const auto exact_fixture_mismatches = mismatch_count(
            exact_query_rank, activations.query_rank);
        const auto warp_fixture_mismatches = mismatch_count(
            warp_query_rank, activations.query_rank);
        const auto warp_exact_mismatches = mismatch_count(
            warp_query_rank, exact_query_rank);
        const auto qb_result = options.nacc16_no_replay
            ? run_nacc16_ungrouped(qb, activations.query_rank, options.qb_split)
            : options.nacc8_no_replay
                ? run_nacc8_ungrouped(qb, activations.query_rank, options.qb_split)
                : run_ungrouped(qb, activations.query_rank, options.qb_split);
        const auto kv_result = run_matrix<16, false>(kv, activations.hidden);
        const auto oa_result = options.nacc16_no_replay
            ? run_nacc16_grouped(oa, activations.attention, options.oa_split)
            : options.nacc8_no_replay
                ? run_nacc8_grouped(oa, activations.attention, options.oa_split)
                : run_grouped(oa, activations.attention, options.oa_split);
        const auto ob_result = run_matrix<8, false>(ob,
            std::span<const std::uint16_t>(oa_result.incumbent_output));
        const bool query_norm_accepted = exact_fixture_mismatches == 0U &&
            warp_fixture_mismatches == 0U && warp_exact_mismatches == 0U;
        const bool accepted = qa_result.no_worse && query_norm_accepted &&
            qb_result.no_worse &&
            kv_result.no_worse && oa_result.no_worse && ob_result.no_worse;
        std::ofstream file;
        std::ostream* output = &std::cout;
        if (!options.output.empty()) {
            file.open(options.output);
            if (!file) throw std::runtime_error("cannot open output file");
            output = &file;
        }
        *output << std::setprecision(12)
                << "{\n  \"schema\":\"strata.dsv4.sm86_fp8_real_accuracy.v1\","
                << "\n  \"device\":\"" << properties.name << "\","
                << "\n  \"layer\":" << options.layer
                << ",\n  \"has_indexer\":"
                << (has_indexer ? "true" : "false")
                << ",\n  \"q_b_split\":" << options.qb_split
                << ",\n  \"wo_a_split\":" << options.oa_split
                << ",\n  \"nacc8_no_replay\":"
                << (options.nacc8_no_replay ? "true" : "false")
                << ",\n  \"nacc16_no_replay\":"
                << (options.nacc16_no_replay ? "true" : "false")
                << ",\n  \"accepted\":" << (accepted ? "true" : "false")
                << ",\n  \"query_norm\":{\"accepted\":"
                << (query_norm_accepted ? "true" : "false")
                << ",\"exact_fixture_mismatches\":"
                << exact_fixture_mismatches
                << ",\"warp_fixture_mismatches\":"
                << warp_fixture_mismatches
                << ",\"warp_exact_mismatches\":"
                << warp_exact_mismatches << "}"
                << ",\n  \"operations\":{\n";
        print_metrics(*output, "wq_a", qa_result, true);
        print_metrics(*output, has_indexer ? "wq_b+indexer.wq_b" : "wq_b",
                      qb_result, true);
        print_metrics(*output, "wkv", kv_result, true);
        print_metrics(*output, "wo_a_grouped", oa_result, true);
        print_metrics(*output, "wo_b", ob_result, false);
        *output << "  }\n}\n";
        return accepted ? EXIT_SUCCESS : 3;
    } catch (const std::exception& exception) {
        std::cerr << "error: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}
