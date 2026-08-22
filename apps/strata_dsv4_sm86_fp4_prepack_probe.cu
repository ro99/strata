// F4-1 step 2: the E2M1/E8M0 group-32 fragment prepack, and the scale-to-K
// binding proof across group boundaries.
//
// Experiment 0137's decoder applies ONE E8M0 scale to all four A registers.
// That is correct for a flat code stream but WRONG in fragment order: the
// m16n8k16 A fragment puts registers 0 and 2 on N-row g and registers 1 and 3
// on N-row g+8, which carry different scales. This probe carries two scales per
// word, selected by register parity, which the unrolled loop folds at compile
// time.
//
// The prepack is a pure permutation. Codes stay N*K/2 bytes and scales stay
// N*K/32 bytes, so one-copy residency is preserved and W_FP4 is unchanged.
//
// Milestone F4-1, FP4 track only. Experimentation operating point.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint32_t kGroup = 32U;      // E8M0 scale group along K
constexpr std::uint32_t kTileN = 16U;      // MMA M dimension = weight N rows
constexpr std::uint32_t kTileM = 8U;       // MMA N dimension = activation cols
constexpr std::uint32_t kTileK = 16U;      // MMA K dimension
constexpr std::uint32_t kWarp = 32U;
// Experiment 0140 measured argmax as load granularity: 4 bytes per lane per
// K-tile against the ceiling probe's 16. Each lane now loads one uint4 holding
// its fragment words for kKPerLoad consecutive K-tiles, so consecutive lanes
// read consecutive 16-byte chunks and a warp issues one fully coalesced
// 512-byte transaction that feeds kKPerLoad MMAs.
constexpr std::uint32_t kKPerLoad = 4U;
constexpr std::uint32_t kWarpsPerBlock = 4U;
std::uint32_t g_split_k = 8U;
constexpr std::uint32_t kWarmups = 3U;
constexpr std::uint32_t kSamples = 11U;
constexpr std::size_t kScrubBytes = 256ULL * 1024ULL * 1024ULL;

struct Shape {
    const char* name;
    std::uint32_t n;
    std::uint32_t k;
};

bool g_break_scale_binding = false;
bool g_no_mma = false;
bool g_split_reduce = false;

constexpr Shape kShapes[] = {
    {"gate_up_w1", 2048U, 4096U},
    {"down_w2", 4096U, 2048U},
};

void check(cudaError_t status, std::string_view op) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " +
                                 cudaGetErrorString(status));
    }
}

class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        check(cudaMalloc(&data_, bytes), "allocate prepack probe buffer");
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

std::uint32_t xorshift(std::uint32_t& s) {
    s ^= s << 13U; s ^= s >> 17U; s ^= s << 5U; return s;
}

std::uint16_t bf16_bits(float v) {
    std::uint32_t b = 0U;
    std::memcpy(&b, &v, sizeof(b));
    return static_cast<std::uint16_t>((b + 0x7FFFU + ((b >> 16U) & 1U)) >> 16U);
}

float bf16_value(float v) {
    const std::uint32_t b = static_cast<std::uint32_t>(bf16_bits(v)) << 16U;
    std::memcpy(&v, &b, sizeof(v));
    return v;
}

double fp4_value(std::uint8_t code) {
    static const double mag[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double m = mag[code & 0x07U];
    return ((code & 0x08U) != 0U && m != 0.0) ? -m : m;
}

double e8m0(std::uint8_t s) { return std::ldexp(1.0, static_cast<int>(s) - 127); }

__constant__ std::uint32_t kMagHigh[2] = {0x3F3F'3F00U, 0x4040'4040U};
__constant__ std::uint32_t kMagLow[2] = {0xC080'0000U, 0xC080'4000U};

__device__ __forceinline__ std::uint32_t scale_pair_bf16(std::uint32_t s) {
    return (s << 7U) * 0x0001'0001U;
}

// The 0137 successor decoder, corrected for fragment order: register parity
// selects between the two N-rows' scales.
template <bool kBreakScaleBinding = false>
__device__ __forceinline__ void decode_fragment(std::uint32_t word,
                                                std::uint32_t scale_low_row,
                                                std::uint32_t scale_high_row,
                                                std::uint32_t (&out)[4]) {
    const std::uint32_t mag = word & 0x7777'7777U;
    const std::uint32_t ha = __byte_perm(kMagHigh[0], kMagHigh[1], mag);
    const std::uint32_t la = __byte_perm(kMagLow[0], kMagLow[1], mag);
    const std::uint32_t hb = __byte_perm(kMagHigh[0], kMagHigh[1], mag >> 16U);
    const std::uint32_t lb = __byte_perm(kMagLow[0], kMagLow[1], mag >> 16U);
    const std::uint32_t pa = __byte_perm(la, ha, 0x5140U);
    const std::uint32_t qa = __byte_perm(la, ha, 0x7362U);
    const std::uint32_t pb = __byte_perm(lb, hb, 0x5140U);
    const std::uint32_t qb = __byte_perm(lb, hb, 0x7362U);
    std::uint32_t value[4];
    value[0] = __byte_perm(pa, pb, 0x5410U);
    value[1] = __byte_perm(pa, pb, 0x7632U);
    value[2] = __byte_perm(qa, qb, 0x5410U);
    value[3] = __byte_perm(qa, qb, 0x7632U);
#pragma unroll
    for (std::uint32_t i = 0U; i < 4U; ++i) {
        // Registers 0 and 2 are N-row g; 1 and 3 are N-row g+8. This selection
        // is compile-time because the loop is fully unrolled.
        // kBreakScaleBinding reproduces experiment 0137's single-scale
        // behaviour, which is correct for a flat stream and wrong here. It
        // exists to prove this probe's oracle detects the defect.
        const std::uint32_t scale =
            (kBreakScaleBinding || (i & 1U) == 0U) ? scale_low_row
                                                   : scale_high_row;
        const std::uint32_t signs = (word << (12U - i * 4U)) & 0x8000'8000U;
        const std::uint32_t non_zero = value[i] + 0x7F80'7F80U;
        const __nv_bfloat162 scaled =
            __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&value[i]),
                    *reinterpret_cast<const __nv_bfloat162*>(&scale));
        out[i] = *reinterpret_cast<const std::uint32_t*>(&scaled) ^
                 (signs & non_zero & 0x8000'8000U);
    }
}

// kUseMma=false keeps every load and the full decode but drops the tensor op,
// which breaks the accumulator dependency chain. It is an attribution arm, not
// a candidate: its output is meaningless and its correctness is not checked.
template <bool kBreakScaleBinding, bool kUseMma = true>
__global__ void prepacked_matmul_kernel(
    const std::uint32_t* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const uint2* __restrict__ activations, std::uint32_t k_extent,
    std::uint32_t k_tiles_per_slice, std::uint32_t split_k,
    float* __restrict__ partials, std::uint32_t* __restrict__ counters,
    float* __restrict__ output, bool fold_reduction) {
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint32_t global_warp = blockIdx.x * kWarpsPerBlock + warp;
    const std::uint32_t n_tile = global_warp / split_k;
    const std::uint32_t slice = global_warp % split_k;
    const std::uint32_t k_tiles = k_extent / kTileK;
    const std::uint32_t group = lane >> 2U;
    const std::uint32_t thread = lane & 3U;
    __shared__ std::uint32_t arrived[kWarpsPerBlock];

    float d0 = 0.0F, d1 = 0.0F, d2 = 0.0F, d3 = 0.0F;

    const std::uint32_t k_blocks = k_tiles / kKPerLoad;
    const std::uint32_t shift = (group & 3U) * 8U;
    const std::uint32_t begin = slice * (k_tiles_per_slice / kKPerLoad);
    const std::uint32_t end = begin + k_tiles_per_slice / kKPerLoad;
    for (std::uint32_t kb = begin; kb < end && kb < k_blocks; ++kb) {
        // One coalesced 16-byte load per lane covers kKPerLoad K-tiles.
        const uint4 packed_words =
            reinterpret_cast<const uint4*>(codes)[(n_tile * k_blocks + kb) *
                                                      kWarp + lane];
        const std::uint32_t word[kKPerLoad] = {packed_words.x, packed_words.y,
                                               packed_words.z, packed_words.w};

        // kKPerLoad K-tiles span exactly two E8M0 groups, both read as
        // 16-byte broadcast loads rather than per-lane byte loads.
        const unsigned char* group_base =
            scales + static_cast<std::size_t>(n_tile * (k_extent / kGroup) +
                                              kb * 2U) * kTileN;
        const uint4 scales_even = *reinterpret_cast<const uint4*>(group_base);
        const uint4 scales_odd =
            *reinterpret_cast<const uint4*>(group_base + kTileN);

#pragma unroll
        for (std::uint32_t j = 0U; j < kKPerLoad; ++j) {
            // Compile-time select: K-tiles 0,1 of the block are in the even
            // E8M0 group and 2,3 in the odd one.
            const uint4 sc = (j < 2U) ? scales_even : scales_odd;
            const std::uint32_t word_low = (group < 4U) ? sc.x : sc.y;
            const std::uint32_t word_high = (group < 4U) ? sc.z : sc.w;
            const std::uint32_t scale_low =
                scale_pair_bf16((word_low >> shift) & 0xFFU);
            const std::uint32_t scale_high =
                scale_pair_bf16((word_high >> shift) & 0xFFU);

            std::uint32_t a[4];
            decode_fragment<kBreakScaleBinding>(word[j], scale_low, scale_high,
                                                a);

            const uint2 b =
                activations[(kb * kKPerLoad + j) * kWarp + lane];
            if constexpr (kUseMma) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x),
                      "r"(b.y));
            } else {
                d0 += __int_as_float(a[0] ^ b.x);
                d1 += __int_as_float(a[1] ^ b.y);
                d2 += __int_as_float(a[2]);
                d3 += __int_as_float(a[3]);
            }
        }
    }

    // D fragment: row = group + (i>=2 ? 8 : 0), column = thread*2 + (i&1).
    // At M=1 only column 0 is real, which lives in d0 and d2 of the lanes with
    // thread == 0. Writing the full 16x8 tile would make 7/8 of the partial
    // traffic zeros -- measured at 24-94% of the useful weight bytes -- so only
    // the real column is stored.
    if (thread == 0U) {
        float* slot = partials + static_cast<std::size_t>(
                                     n_tile * split_k + slice) * kTileN;
        slot[group] = d0;
        slot[group + 8U] = d2;
    }
    (void)d1;
    (void)d3;

    // Single-launch reduction. A separate reduce kernel cost a full 4.10 us of
    // dispatch for trivial work -- 29% of the whole step at this matrix size.
    // Instead the last slice to finish an N-tile reduces it, and resets the
    // counter so the kernel is re-runnable without a separate memset.
    if (!fold_reduction) return;
    __threadfence();
    __syncwarp();
    if (lane == 0U) {
        // One slot per warp in the block, not per N-tile: warps in a block
        // carry different N-tiles.
        arrived[warp] = atomicAdd(&counters[n_tile], 1U);
    }
    __syncwarp();
    if (fold_reduction && arrived[warp] == split_k - 1U) {
        if (lane < kTileN) {
            float sum = 0.0F;
            for (std::uint32_t sl = 0U; sl < split_k; ++sl) {
                sum += partials[(static_cast<std::size_t>(n_tile) * split_k +
                                 sl) * kTileN + lane];
            }
            output[n_tile * kTileN + lane] = sum;
        }
        if (lane == 0U) counters[n_tile] = 0U;
    }
}

__global__ void reduce_kernel(const float* __restrict__ partials,
                              std::uint32_t n_tiles, std::uint32_t split_k,
                              float* __restrict__ out) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = n_tiles * kTileN;
    if (index >= total) return;
    const std::uint32_t n_tile = index / kTileN;
    const std::uint32_t row = index % kTileN;
    float sum = 0.0F;
    for (std::uint32_t slice = 0U; slice < split_k; ++slice) {
        sum += partials[(static_cast<std::size_t>(n_tile) * split_k + slice) *
                            kTileN + row];
    }
    out[index] = sum;
}

__global__ void empty_kernel() {}

__global__ void scrub_kernel(unsigned char* data, std::size_t bytes) {
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
         i < bytes; i += stride) {
        data[i] = static_cast<unsigned char>(i);
    }
}

float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2U];
}

struct Result {
    const char* name;
    std::uint32_t n{0U}, k{0U};
    std::uint64_t useful_bytes{0U};
    std::uint64_t code_bytes{0U}, scale_bytes{0U};
    std::uint64_t prepacked_code_bytes{0U}, prepacked_scale_bytes{0U};
    std::uint64_t device_bytes{0U};
    double max_absolute{0.0}, max_relative{0.0};
    double output_max_abs{0.0}, oracle_max_abs{0.0};
    std::uint32_t nonzero_outputs{0U};
    std::uint32_t k_groups_crossed{0U};
    float cold_ms{0.0F}, hot_ms{0.0F};
    float matmul_ms{0.0F}, reduce_ms{0.0F}, empty_ms{0.0F};
    float pipelined_us{0.0F};
    double pipelined_gbps{0.0};
    double cold_gbps{0.0};
    double prepack_seconds{0.0};
};

Result run_shape(const Shape& shape, cudaStream_t stream) {
    Result r{shape.name, shape.n, shape.k};
    const std::uint32_t n = shape.n, k = shape.k;
    r.code_bytes = static_cast<std::uint64_t>(n) * (k / 2U);
    r.scale_bytes = static_cast<std::uint64_t>(n) * (k / kGroup);
    r.useful_bytes = r.code_bytes + r.scale_bytes;
    r.k_groups_crossed = k / kGroup;

    // Canonical checkpoint layout: codes [n][k/2] two per byte, scales [n][k/32]
    std::vector<std::uint8_t> canon_codes(static_cast<std::size_t>(n) * k);
    std::vector<std::uint8_t> canon_scales(r.scale_bytes);
    std::vector<float> activation(k);
    std::uint32_t state = 0x5EED'1234U ^ n ^ (k << 7U);
    for (auto& c : canon_codes) c = static_cast<std::uint8_t>(xorshift(state) & 0x0FU);
    for (auto& s : canon_scales)
        s = static_cast<std::uint8_t>(120U + xorshift(state) % 15U);
    for (auto& a : activation)
        a = bf16_value(static_cast<float>(static_cast<int>(xorshift(state) % 9U) - 4));

    // ---- prepack: pure permutation into fragment order ----
    const std::uint32_t n_tiles = n / kTileN;
    const std::uint32_t k_tiles = k / kTileK;
    std::vector<std::uint32_t> frag_codes(
        static_cast<std::size_t>(n_tiles) * k_tiles * kWarp);
    std::vector<std::uint8_t> frag_scales(
        static_cast<std::size_t>(n_tiles) * (k / kGroup) * kTileN);

    const auto t0 = std::chrono::steady_clock::now();
    for (std::uint32_t nt = 0U; nt < n_tiles; ++nt) {
        for (std::uint32_t kt = 0U; kt < k_tiles; ++kt) {
            for (std::uint32_t lane = 0U; lane < kWarp; ++lane) {
                const std::uint32_t g = lane >> 2U, t = lane & 3U;
                std::uint32_t word = 0U;
                for (std::uint32_t i = 0U; i < 4U; ++i) {
                    const std::uint32_t row =
                        nt * kTileN + g + (((i & 1U) != 0U) ? 8U : 0U);
                    const std::uint32_t col =
                        kt * kTileK + t * 2U + ((i >= 2U) ? 8U : 0U);
                    word |= static_cast<std::uint32_t>(
                                canon_codes[static_cast<std::size_t>(row) * k +
                                            col])
                            << (i * 4U);
                    word |= static_cast<std::uint32_t>(
                                canon_codes[static_cast<std::size_t>(row) * k +
                                            col + 1U])
                            << ((i + 4U) * 4U);
                }
                const std::uint32_t kb = kt / kKPerLoad;
                const std::uint32_t j = kt % kKPerLoad;
                frag_codes[((static_cast<std::size_t>(nt) *
                                 (k_tiles / kKPerLoad) + kb) * kWarp + lane) *
                               kKPerLoad + j] = word;
            }
        }
        for (std::uint32_t kg = 0U; kg < k / kGroup; ++kg) {
            for (std::uint32_t row = 0U; row < kTileN; ++row) {
                frag_scales[(static_cast<std::size_t>(nt) * (k / kGroup) + kg) *
                                kTileN + row] =
                    canon_scales[static_cast<std::size_t>(nt * kTileN + row) *
                                     (k / kGroup) + kg];
            }
        }
    }
    r.prepack_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    r.prepacked_code_bytes = frag_codes.size() * sizeof(std::uint32_t);
    r.prepacked_scale_bytes = frag_scales.size();

    // Pre-permute activations into B-fragment order: for lane (group, thread),
    // b0 = rows {2t, 2t+1} and b1 = rows {2t+8, 2t+9} of column `group`. Only
    // column 0 is real at M=1, so lanes with group != 0 carry zeros.
    std::vector<std::uint32_t> activation_frag(
        static_cast<std::size_t>(k_tiles) * kWarp * 2U);
    for (std::uint32_t kt = 0U; kt < k_tiles; ++kt) {
        for (std::uint32_t lane = 0U; lane < kWarp; ++lane) {
            const std::uint32_t g = lane >> 2U, t = lane & 3U;
            std::uint32_t b0 = 0U, b1 = 0U;
            if (g == 0U) {
                const std::uint32_t base = kt * kTileK;
                b0 = static_cast<std::uint32_t>(
                         bf16_bits(activation[base + t * 2U])) |
                     (static_cast<std::uint32_t>(
                          bf16_bits(activation[base + t * 2U + 1U]))
                      << 16U);
                b1 = static_cast<std::uint32_t>(
                         bf16_bits(activation[base + t * 2U + 8U])) |
                     (static_cast<std::uint32_t>(
                          bf16_bits(activation[base + t * 2U + 9U]))
                      << 16U);
            }
            activation_frag[(static_cast<std::size_t>(kt) * kWarp + lane) * 2U] = b0;
            activation_frag[(static_cast<std::size_t>(kt) * kWarp + lane) * 2U + 1U] = b1;
        }
    }

    DeviceBuffer d_codes(r.prepacked_code_bytes);
    DeviceBuffer d_scales(r.prepacked_scale_bytes);
    DeviceBuffer d_act(activation_frag.size() * sizeof(std::uint32_t));
    DeviceBuffer d_partials(static_cast<std::size_t>(n_tiles) * g_split_k *
                            kTileN * sizeof(float));
    DeviceBuffer d_out(static_cast<std::size_t>(n) * sizeof(float));
    DeviceBuffer d_counters(static_cast<std::size_t>(n_tiles) *
                            sizeof(std::uint32_t));
    check(cudaMemsetAsync(d_counters.get(), 0, d_counters.bytes(), stream),
          "zero split-K counters");
    DeviceBuffer d_scrub(kScrubBytes);
    r.device_bytes = d_codes.bytes() + d_scales.bytes() + d_act.bytes() +
                     d_partials.bytes() + d_out.bytes();

    check(cudaMemcpyAsync(d_codes.get(), frag_codes.data(),
                          r.prepacked_code_bytes, cudaMemcpyHostToDevice,
                          stream), "upload fragment codes");
    check(cudaMemcpyAsync(d_scales.get(), frag_scales.data(),
                          r.prepacked_scale_bytes, cudaMemcpyHostToDevice,
                          stream), "upload fragment scales");
    check(cudaMemcpyAsync(d_act.get(), activation_frag.data(),
                          d_act.bytes(), cudaMemcpyHostToDevice, stream),
          "upload B-fragment activations");
    check(cudaStreamSynchronize(stream), "finish uploads");

    const std::uint32_t warps = n_tiles * g_split_k;
    const std::uint32_t blocks = warps / kWarpsPerBlock;
    const std::uint32_t k_tiles_per_slice = k_tiles / g_split_k;

    const auto launch_reduce = [&] {
        reduce_kernel<<<(n + 255U) / 256U, 256U, 0U, stream>>>(
            static_cast<const float*>(d_partials.get()), n_tiles, g_split_k,
            static_cast<float*>(d_out.get()));
    };
    const auto launch_matmul = [&] {
        if (g_no_mma)
            prepacked_matmul_kernel<false, false>
                <<<blocks, kWarpsPerBlock * kWarp, 0U, stream>>>(
                    static_cast<const std::uint32_t*>(d_codes.get()),
                    static_cast<const unsigned char*>(d_scales.get()),
                    static_cast<const uint2*>(d_act.get()), k,
                    k_tiles_per_slice, g_split_k,
                    static_cast<float*>(d_partials.get()),
                    static_cast<std::uint32_t*>(d_counters.get()),
                    static_cast<float*>(d_out.get()), !g_split_reduce);
        else if (g_break_scale_binding)
            prepacked_matmul_kernel<true>
                <<<blocks, kWarpsPerBlock * kWarp, 0U, stream>>>(
                    static_cast<const std::uint32_t*>(d_codes.get()),
                    static_cast<const unsigned char*>(d_scales.get()),
                    static_cast<const uint2*>(d_act.get()), k,
                    k_tiles_per_slice, g_split_k,
                    static_cast<float*>(d_partials.get()),
                    static_cast<std::uint32_t*>(d_counters.get()),
                    static_cast<float*>(d_out.get()), !g_split_reduce);
        else
        prepacked_matmul_kernel<false>
            <<<blocks, kWarpsPerBlock * kWarp, 0U, stream>>>(
            static_cast<const std::uint32_t*>(d_codes.get()),
            static_cast<const unsigned char*>(d_scales.get()),
            static_cast<const uint2*>(d_act.get()), k,
            k_tiles_per_slice, g_split_k,
            static_cast<float*>(d_partials.get()),
            static_cast<std::uint32_t*>(d_counters.get()),
            static_cast<float*>(d_out.get()), !g_split_reduce);
    };
    const auto launch = [&] {
        launch_matmul();
        if (g_split_reduce) launch_reduce();
    };

    launch();
    check(cudaGetLastError(), "launch prepacked matmul");
    check(cudaStreamSynchronize(stream), "finish correctness launch");

    std::vector<float> out(n);
    check(cudaMemcpyAsync(out.data(), d_out.get(), n * sizeof(float),
                          cudaMemcpyDeviceToHost, stream), "download output");
    check(cudaStreamSynchronize(stream), "finish download");

    // ---- oracle: double precision, from the CANONICAL layout ----
    for (std::uint32_t row = 0U; row < n; ++row) {
        double sum = 0.0;
        for (std::uint32_t col = 0U; col < k; ++col) {
            const double w =
                fp4_value(canon_codes[static_cast<std::size_t>(row) * k + col]) *
                e8m0(canon_scales[static_cast<std::size_t>(row) * (k / kGroup) +
                                  col / kGroup]);
            sum += w * static_cast<double>(activation[col]);
        }
        r.oracle_max_abs = std::max(r.oracle_max_abs, std::abs(sum));
        r.output_max_abs =
            std::max(r.output_max_abs, std::abs(static_cast<double>(out[row])));
        if (out[row] != 0.0F) ++r.nonzero_outputs;
        const double diff = std::abs(static_cast<double>(out[row]) - sum);
        r.max_absolute = std::max(r.max_absolute, diff);
        const double denom = std::max(std::abs(sum), 1.0);
        r.max_relative = std::max(r.max_relative, diff / denom);
    }

    // ---- timing ----
    cudaEvent_t start = nullptr, stop = nullptr;
    check(cudaEventCreate(&start), "create start");
    check(cudaEventCreate(&stop), "create stop");
    const auto scrub = [&] {
        scrub_kernel<<<1024U, 256U, 0U, stream>>>(
            static_cast<unsigned char*>(d_scrub.get()), kScrubBytes);
    };
    // Phase attribution. A single production matrix pass is only a few
    // microseconds, which is the same order as a kernel launch, so the split
    // between real work and dispatch has to be measured, not assumed.
    cudaEvent_t e0 = nullptr, e1 = nullptr, e2 = nullptr;
    check(cudaEventCreate(&e0), "create e0");
    check(cudaEventCreate(&e1), "create e1");
    check(cudaEventCreate(&e2), "create e2");
    {
        std::vector<float> mm, rd, em;
        for (std::uint32_t i = 0U; i < kSamples; ++i) {
            check(cudaEventRecord(e0, stream), "e0");
            launch_matmul();
            check(cudaEventRecord(e1, stream), "e1");
            launch_reduce();
            check(cudaEventRecord(e2, stream), "e2");
            check(cudaEventSynchronize(e2), "sync phases");
            float a = 0.0F, b = 0.0F;
            check(cudaEventElapsedTime(&a, e0, e1), "matmul ms");
            check(cudaEventElapsedTime(&b, e1, e2), "reduce ms");
            mm.push_back(a);
            rd.push_back(b);
            check(cudaEventRecord(e0, stream), "e0 empty");
            empty_kernel<<<1U, 32U, 0U, stream>>>();
            check(cudaEventRecord(e1, stream), "e1 empty");
            check(cudaEventSynchronize(e1), "sync empty");
            float c = 0.0F;
            check(cudaEventElapsedTime(&c, e0, e1), "empty ms");
            em.push_back(c);
        }
        r.matmul_ms = median(std::move(mm));
        r.reduce_ms = median(std::move(rd));
        r.empty_ms = median(std::move(em));
    }
    check(cudaEventDestroy(e0), "destroy e0");
    check(cudaEventDestroy(e1), "destroy e1");
    check(cudaEventDestroy(e2), "destroy e2");

    // Steady-state cost. Timing one launch inside an event pair charges the
    // whole event-measurement floor to a single 4.5 MB matrix. Production
    // dispatches these back to back inside a graph, so the marginal per-matrix
    // cost is measured over a run of launches in one window -- the same reason
    // the campaign ruler is measured on a 128 MiB sweep rather than one matrix.
    {
        constexpr std::uint32_t kChain = 32U;
        std::vector<float> chain;
        for (std::uint32_t i = 0U; i < kSamples; ++i) {
            check(cudaEventRecord(start, stream), "chain start");
            for (std::uint32_t j = 0U; j < kChain; ++j) launch();
            check(cudaEventRecord(stop, stream), "chain stop");
            check(cudaEventSynchronize(stop), "chain sync");
            float ms = 0.0F;
            check(cudaEventElapsedTime(&ms, start, stop), "chain ms");
            chain.push_back(ms / static_cast<float>(kChain));
        }
        r.pipelined_us = median(std::move(chain)) * 1.0e3F;
        r.pipelined_gbps = static_cast<double>(r.useful_bytes) /
                           (static_cast<double>(r.pipelined_us) * 1.0e-6) / 1.0e9;
    }

    const auto sample = [&](bool cold) {
        if (cold) scrub();
        check(cudaEventRecord(start, stream), "record start");
        launch();
        check(cudaEventRecord(stop, stream), "record stop");
        check(cudaEventSynchronize(stop), "sync sample");
        float ms = 0.0F;
        check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
        return ms;
    };
    for (std::uint32_t w = 0U; w < kWarmups; ++w) static_cast<void>(sample(false));
    std::vector<float> cold, hot;
    for (std::uint32_t s = 0U; s < kSamples; ++s) {
        cold.push_back(sample(true));
        hot.push_back(sample(false));
    }
    r.cold_ms = median(std::move(cold));
    r.hot_ms = median(std::move(hot));
    r.cold_gbps = static_cast<double>(r.useful_bytes) /
                  (static_cast<double>(r.cold_ms) * 1.0e-3) / 1.0e9;
    check(cudaEventDestroy(start), "destroy start");
    check(cudaEventDestroy(stop), "destroy stop");
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        int device = 0;
        std::string output_path;
        for (int i = 1; i < argc; ++i) {
            const std::string_view f = argv[i];
            if (f == "--device" && i + 1 < argc) device = std::stoi(argv[++i]);
            else if (f == "--output" && i + 1 < argc) output_path = argv[++i];
            else if (f == "--break-scale-binding") g_break_scale_binding = true;
            else if (f == "--no-mma") g_no_mma = true;
            else if (f == "--split-reduce") g_split_reduce = true;
            else if (f == "--split-k" && i + 1 < argc)
                g_split_k = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            else { std::cerr << "usage: " << argv[0]
                             << " [--device INDEX] [--output PATH]\n";
                   return EXIT_FAILURE; }
        }
        check(cudaSetDevice(device), "select device");
        cudaDeviceProp p{};
        check(cudaGetDeviceProperties(&p, device), "query device");
        if (p.major != 8 || p.minor != 6)
            throw std::runtime_error("prepack probe requires SM86");

        cudaStream_t stream = nullptr;
        check(cudaStreamCreate(&stream), "create stream");
        std::vector<Result> results;
        for (const Shape& s : kShapes) results.push_back(run_shape(s, stream));
        check(cudaStreamDestroy(stream), "destroy stream");

        std::ostringstream json;
        json << std::fixed << "{\n"
             << "  \"device_name\": \"" << p.name << "\",\n"
             << "  \"device_capability\": \"" << p.major << "." << p.minor
             << "\",\n"
             << "  \"milestone\": \"F4-1 step 2 fragment prepack\",\n"
             << "  \"operating_point\": \"experimentation: single RTX 3090, "
                "350 W, unlocked clocks\",\n"
             << "  \"split_k\": " << g_split_k << ",\n"
             << "  \"shapes\": [\n";
        for (std::size_t i = 0U; i < results.size(); ++i) {
            const Result& r = results[i];
            json << "    {\"name\": \"" << r.name << "\", \"n\": " << r.n
                 << ", \"k\": " << r.k
                 << ", \"useful_weight_bytes\": " << r.useful_bytes
                 << ", \"canonical_code_bytes\": " << r.code_bytes
                 << ", \"prepacked_code_bytes\": " << r.prepacked_code_bytes
                 << ", \"canonical_scale_bytes\": " << r.scale_bytes
                 << ", \"prepacked_scale_bytes\": " << r.prepacked_scale_bytes
                 << ", \"k_groups_crossed\": " << r.k_groups_crossed
                 << ", \"device_bytes\": " << r.device_bytes
                 << ", \"prepack_seconds\": " << std::setprecision(3)
                 << r.prepack_seconds
                 << ", \"output_max_abs\": " << std::setprecision(3)
                 << r.output_max_abs
                 << ", \"oracle_max_abs\": " << r.oracle_max_abs
                 << ", \"nonzero_outputs\": " << r.nonzero_outputs
                 << ", \"max_absolute\": " << std::setprecision(9)
                 << r.max_absolute
                 << ", \"max_relative\": " << r.max_relative
                 << ", \"pipelined_us\": " << std::setprecision(3)
                 << r.pipelined_us
                 << ", \"pipelined_gbps\": " << std::setprecision(2)
                 << r.pipelined_gbps
                 << ", \"matmul_ms\": " << std::setprecision(6) << r.matmul_ms
                 << ", \"reduce_ms\": " << r.reduce_ms
                 << ", \"empty_launch_ms\": " << r.empty_ms
                 << ", \"cold_ms\": " << r.cold_ms
                 << ", \"cold_gbps\": " << std::setprecision(2) << r.cold_gbps
                 << "}" << (i + 1U < results.size() ? ",\n" : "\n");
        }
        json << "  ]\n}\n";
        if (output_path.empty()) std::cout << json.str();
        else {
            std::ofstream o(output_path);
            if (!o) throw std::runtime_error("cannot open " + output_path);
            o << json.str();
        }
        if (g_no_mma) return EXIT_SUCCESS;  // attribution arm: output is meaningless
        for (const Result& r : results)
            if (!(r.max_relative < 1.0e-4)) return EXIT_FAILURE;
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
