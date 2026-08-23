// Does the fused MXFP4 MoE kernel leave bandwidth on the table, and is it
// bound by reading weights or by dispatch?
//
// Experiments 0166 and 0167 stopped the register-fed substitution for Laguna
// and Inkling on a cost gate: GPU matmul is 6.2% and 5.5% of a decode step, so
// Amdahl caps the substitution near 1.05x however fast the kernel is. That
// arithmetic used the achieved bandwidth as an inference, not a measurement.
// This probe measures it.
//
// It reproduces the incumbent kernels exactly -- mxfp4_moe_gate_up_kernel and
// mxfp4_moe_down_kernel from kernels/cuda/backend.cu, copied so the probe times
// the production arithmetic rather than an approximation of it -- at Laguna's
// real routed-expert shapes, and sweeps the dispatch width. A kernel that is
// bandwidth-bound holds its GB/s as the width grows; one that is dispatch-bound
// climbs with it.
//
// The ruler is a pure streaming read over the same arena, so "percent of
// roofline" is measured on this card at this operating point rather than quoted
// from a datasheet.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace {

// Laguna S 2.1 MXFP4 routed expert, from experiment 0166: gate/up packed U8
// [1024,1536] with scale U8 [1024,96] for logical [1024,3072]; down packed U8
// [3072,512] with scale U8 [3072,32] for logical [3072,1024].
constexpr std::uint32_t kHidden = 3072U;        // gate/up input columns
constexpr std::uint32_t kIntermediate = 1024U;  // gate/up output rows
constexpr std::uint32_t kGroup = 32U;
constexpr std::uint32_t kThreads = 256U;
constexpr std::uint32_t kWarmups = 3U;
constexpr std::uint32_t kSamples = 11U;
// Enough experts that one sweep cannot sit in the 3090's 6 MB L2. Timing a
// single reused matrix is how a previous probe reported a forbidden
// 435-484 GB/s ruler: it measured cache, then launch overhead.
constexpr std::uint32_t kResidentExperts = 96U;

struct Batch {
    const unsigned char* gate_weights[64];
    const unsigned char* gate_scales[64];
    const unsigned char* up_weights[64];
    const unsigned char* up_scales[64];
    const unsigned char* down_weights[64];
    const unsigned char* down_scales[64];
    std::uint32_t count;
    std::uint32_t rows;
};

void check(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "error: %s: %s\n", what, cudaGetErrorString(status));
    std::exit(1);
}

__device__ __forceinline__ float fp4_e2m1_value(unsigned int encoded) {
    const bool negative = (encoded & 0x08U) != 0U;
    const unsigned int magnitude = encoded & 0x07U;
    const float table[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    const float value = table[magnitude];
    return negative ? -value : value;
}

__device__ __forceinline__ float fp8_e8m0_scale_bits(unsigned char encoded) {
    if (encoded == 0xFFU) return __uint_as_float(0x7FC0'0000U);
    if (encoded == 0U) return __uint_as_float(0x0040'0000U);
    return __uint_as_float(static_cast<unsigned int>(encoded) << 23U);
}

__device__ __forceinline__ float reduce_block(float value) {
    __shared__ float shared[kThreads / 32U];
    const unsigned int lane = threadIdx.x & 31U;
    const unsigned int warp = threadIdx.x >> 5U;
    for (unsigned int offset = 16U; offset != 0U; offset >>= 1U) {
        value += __shfl_down_sync(0xFFFF'FFFFU, value, offset);
    }
    if (lane == 0U) shared[warp] = value;
    __syncthreads();
    if (warp != 0U) return 0.0F;
    value = lane < (kThreads / 32U) ? shared[lane] : 0.0F;
    for (unsigned int offset = 16U; offset != 0U; offset >>= 1U) {
        value += __shfl_down_sync(0xFFFF'FFFFU, value, offset);
    }
    return value;
}

// Copied from mxfp4_moe_gate_up_kernel.
__global__ void gate_up_kernel(float* activations, const float* hidden,
                               Batch batch, std::uint64_t columns,
                               std::uint64_t intermediate,
                               std::uint64_t packed_columns,
                               std::uint64_t scale_columns) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t batch_row = blockIdx.y;
    const auto expert = batch_row / batch.rows;
    const auto row = batch_row % batch.rows;
    if (output_row >= intermediate || expert >= batch.count) return;
    const auto* gate_weights = batch.gate_weights[expert];
    const auto* gate_scales = batch.gate_scales[expert];
    const auto* up_weights = batch.up_weights[expert];
    const auto* up_scales = batch.up_scales[expert];
    const auto weight_base = output_row * packed_columns;
    const auto scale_base = output_row * scale_columns;
    const auto input_base = static_cast<std::uint64_t>(row) * columns;
    float gate = 0.0F;
    float up = 0.0F;
    for (std::uint64_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        const auto packed_index = weight_base + column / 2U;
        const auto scale_index = scale_base + column / kGroup;
        const unsigned char gate_packed = gate_weights[packed_index];
        const unsigned char up_packed = up_weights[packed_index];
        const unsigned int gate_encoded =
            column % 2U == 0U ? gate_packed & 0x0FU : gate_packed >> 4U;
        const unsigned int up_encoded =
            column % 2U == 0U ? up_packed & 0x0FU : up_packed >> 4U;
        const float input = hidden[input_base + column];
        gate += input * fp4_e2m1_value(gate_encoded) *
                fp8_e8m0_scale_bits(gate_scales[scale_index]);
        up += input * fp4_e2m1_value(up_encoded) *
              fp8_e8m0_scale_bits(up_scales[scale_index]);
    }
    gate = reduce_block(gate);
    __syncthreads();
    up = reduce_block(up);
    if (threadIdx.x == 0U) {
        const float exponential = gate >= 0.0F ? expf(-gate) : expf(gate);
        const float sigmoid = gate >= 0.0F ? 1.0F / (1.0F + exponential)
                                           : exponential / (1.0F + exponential);
        const auto activation =
            (static_cast<std::uint64_t>(expert) * batch.rows + row) *
                intermediate + output_row;
        activations[activation] = gate * sigmoid * up;
    }
}

// Copied from mxfp4_moe_down_kernel.
__global__ void down_kernel(float* output, const float* activations,
                            Batch batch, std::uint64_t columns,
                            std::uint64_t rows, std::uint64_t packed_columns,
                            std::uint64_t scale_columns) {
    const std::uint64_t output_row = blockIdx.x;
    const std::uint32_t batch_row = blockIdx.y;
    const auto expert = batch_row / batch.rows;
    const auto row = batch_row % batch.rows;
    if (output_row >= rows || expert >= batch.count) return;
    const auto* weights = batch.down_weights[expert];
    const auto* scales = batch.down_scales[expert];
    const auto weight_base = output_row * packed_columns;
    const auto scale_base = output_row * scale_columns;
    const auto input_base =
        (static_cast<std::uint64_t>(expert) * batch.rows + row) * columns;
    float sum = 0.0F;
    for (std::uint64_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        const unsigned char packed = weights[weight_base + column / 2U];
        const unsigned int encoded =
            column % 2U == 0U ? packed & 0x0FU : packed >> 4U;
        sum += activations[input_base + column] * fp4_e2m1_value(encoded) *
               fp8_e8m0_scale_bits(scales[scale_base + column / kGroup]);
    }
    sum = reduce_block(sum);
    if (threadIdx.x == 0U) {
        output[(static_cast<std::uint64_t>(expert) * batch.rows + row) * rows +
               output_row] = sum;
    }
}

// ---------------------------------------------------------------------------
// Register-fed arm. The decode is copied from regfed_fp4_decode_fragment in
// kernels/cuda/backend.cu so the probe measures the production arithmetic.
// ---------------------------------------------------------------------------

constexpr std::uint32_t kTileN = 16U;
constexpr std::uint32_t kTileK = 16U;
constexpr std::uint32_t kWarp = 32U;
constexpr std::uint32_t kKPerLoad = 4U;
constexpr std::uint32_t kWarpsPerBlock = 4U;

__constant__ std::uint32_t kMagHigh[2] = {0x3F3F'3F00U, 0x4040'4040U};
__constant__ std::uint32_t kMagLow[2] = {0xC080'0000U, 0xC080'4000U};

__device__ __forceinline__ std::uint32_t scale_pair(std::uint32_t code) {
    return (code << 7U) * 0x0001'0001U;
}

__device__ __forceinline__ void decode_fragment(std::uint32_t word,
                                                std::uint32_t scale_low,
                                                std::uint32_t scale_high,
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
        const std::uint32_t sc = ((i & 1U) == 0U) ? scale_low : scale_high;
        const std::uint32_t signs = (word << (12U - i * 4U)) & 0x8000'8000U;
        const std::uint32_t non_zero = value[i] + 0x7F80'7F80U;
        const __nv_bfloat162 scaled =
            __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&value[i]),
                    *reinterpret_cast<const __nv_bfloat162*>(&sc));
        out[i] = *reinterpret_cast<const std::uint32_t*>(&scaled) ^
                 (signs & non_zero & 0x8000'8000U);
    }
}

__device__ __forceinline__ void mma_m16n8k16(float& d0, float& d1, float& d2,
                                             float& d3, std::uint32_t a0,
                                             std::uint32_t a1, std::uint32_t a2,
                                             std::uint32_t a3, std::uint32_t b0,
                                             std::uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

struct FragBatch {
    const std::uint32_t* gate_codes[64];
    const unsigned char* gate_scales[64];
    const std::uint32_t* up_codes[64];
    const unsigned char* up_scales[64];
    const std::uint32_t* down_codes[64];
    const unsigned char* down_scales[64];
    std::uint32_t count;
};

// Gate and up share the activation, so one pass over the hidden vector feeds
// both weight streams -- the same fusion the incumbent kernel uses.
__global__ __launch_bounds__(128) void regfed_gate_up_kernel(
    float* __restrict__ partials, const uint2* __restrict__ activations,
    FragBatch batch, std::uint32_t columns, std::uint32_t intermediate,
    std::uint32_t split) {
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint32_t n_tiles = intermediate / kTileN;
    const std::uint32_t k_blocks = (columns / kTileK) / kKPerLoad;
    const std::uint32_t per_slice = k_blocks / split;
    const std::uint32_t scale_columns = columns / kGroup;
    const std::uint32_t group = lane >> 2U;
    const std::uint32_t thread = lane & 3U;
    const std::uint32_t shift = (group & 3U) * 8U;
    const std::uint32_t total = batch.count * n_tiles * split;

    for (std::uint32_t work = blockIdx.x * kWarpsPerBlock + warp; work < total;
         work += gridDim.x * kWarpsPerBlock) {
        const std::uint32_t slice = work % split;
        const std::uint32_t flat = work / split;
        const std::uint32_t n_tile = flat % n_tiles;
        const std::uint32_t expert = flat / n_tiles;
        const uint4* gate4 =
            reinterpret_cast<const uint4*>(batch.gate_codes[expert]);
        const uint4* up4 =
            reinterpret_cast<const uint4*>(batch.up_codes[expert]);
        const unsigned char* gs = batch.gate_scales[expert];
        const unsigned char* us = batch.up_scales[expert];
        float gate[4]{}, up[4]{};
        const bool live = group == 0U;
        for (std::uint32_t block = slice * per_slice;
             block < (slice + 1U) * per_slice; ++block) {
            const std::size_t index =
                (static_cast<std::size_t>(n_tile) * k_blocks + block) * kWarp +
                lane;
            const uint4 gpacked = gate4[index];
            const uint4 upacked = up4[index];
            const std::size_t sbase =
                (static_cast<std::size_t>(n_tile) * scale_columns +
                 block * 2U) * kTileN;
            const uint4 g_even = *reinterpret_cast<const uint4*>(gs + sbase);
            const uint4 g_odd =
                *reinterpret_cast<const uint4*>(gs + sbase + kTileN);
            const uint4 u_even = *reinterpret_cast<const uint4*>(us + sbase);
            const uint4 u_odd =
                *reinterpret_cast<const uint4*>(us + sbase + kTileN);
            const std::uint32_t gw[4] = {gpacked.x, gpacked.y, gpacked.z,
                                         gpacked.w};
            const std::uint32_t uw[4] = {upacked.x, upacked.y, upacked.z,
                                         upacked.w};
#pragma unroll
            for (std::uint32_t j = 0U; j < kKPerLoad; ++j) {
                const uint4 gsel = (j < 2U) ? g_even : g_odd;
                const uint4 usel = (j < 2U) ? u_even : u_odd;
                const std::uint32_t glo = (group < 4U) ? gsel.x : gsel.y;
                const std::uint32_t ghi = (group < 4U) ? gsel.z : gsel.w;
                const std::uint32_t ulo = (group < 4U) ? usel.x : usel.y;
                const std::uint32_t uhi = (group < 4U) ? usel.z : usel.w;
                std::uint32_t ga[4], ua[4];
                decode_fragment(gw[j], scale_pair((glo >> shift) & 0xFFU),
                                scale_pair((ghi >> shift) & 0xFFU), ga);
                decode_fragment(uw[j], scale_pair((ulo >> shift) & 0xFFU),
                                scale_pair((uhi >> shift) & 0xFFU), ua);
                const std::uint32_t k_tile = block * kKPerLoad + j;
                const uint2 b = live ? activations[k_tile * 4U + thread]
                                     : make_uint2(0U, 0U);
                mma_m16n8k16(gate[0], gate[1], gate[2], gate[3], ga[0], ga[1],
                             ga[2], ga[3], b.x, b.y);
                mma_m16n8k16(up[0], up[1], up[2], up[3], ua[0], ua[1], ua[2],
                             ua[3], b.x, b.y);
            }
        }
        if (thread == 0U) {
            const std::uint32_t rows[2] = {group, group + 8U};
            const int regs[2] = {0, 2};
            for (int which = 0; which < 2; ++which) {
                const std::size_t row =
                    (static_cast<std::size_t>(expert) * intermediate +
                     n_tile * kTileN + rows[which]);
                float* slot = partials + (row * split + slice) * 2U;
                slot[0] = gate[regs[which]];
                slot[1] = up[regs[which]];
            }
        }
    }
}

__global__ __launch_bounds__(128) void regfed_down_kernel(
    float* __restrict__ partials, const uint2* __restrict__ activations,
    FragBatch batch, std::uint32_t columns, std::uint32_t rows,
    std::uint32_t split) {
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint32_t n_tiles = rows / kTileN;
    const std::uint32_t k_blocks = (columns / kTileK) / kKPerLoad;
    const std::uint32_t per_slice = k_blocks / split;
    const std::uint32_t scale_columns = columns / kGroup;
    const std::uint32_t group = lane >> 2U;
    const std::uint32_t thread = lane & 3U;
    const std::uint32_t shift = (group & 3U) * 8U;
    const std::uint32_t total = batch.count * n_tiles * split;

    for (std::uint32_t work = blockIdx.x * kWarpsPerBlock + warp; work < total;
         work += gridDim.x * kWarpsPerBlock) {
        const std::uint32_t slice = work % split;
        const std::uint32_t flat = work / split;
        const std::uint32_t n_tile = flat % n_tiles;
        const std::uint32_t expert = flat / n_tiles;
        const uint4* codes4 =
            reinterpret_cast<const uint4*>(batch.down_codes[expert]);
        const unsigned char* sc = batch.down_scales[expert];
        float acc[4]{};
        const bool live = group == 0U;
        for (std::uint32_t block = slice * per_slice;
             block < (slice + 1U) * per_slice; ++block) {
            const uint4 packed =
                codes4[(static_cast<std::size_t>(n_tile) * k_blocks + block) *
                           kWarp + lane];
            const std::size_t sbase =
                (static_cast<std::size_t>(n_tile) * scale_columns +
                 block * 2U) * kTileN;
            const uint4 even = *reinterpret_cast<const uint4*>(sc + sbase);
            const uint4 odd =
                *reinterpret_cast<const uint4*>(sc + sbase + kTileN);
            const std::uint32_t w[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
            for (std::uint32_t j = 0U; j < kKPerLoad; ++j) {
                const uint4 sel = (j < 2U) ? even : odd;
                const std::uint32_t lo = (group < 4U) ? sel.x : sel.y;
                const std::uint32_t hi = (group < 4U) ? sel.z : sel.w;
                std::uint32_t a[4];
                decode_fragment(w[j], scale_pair((lo >> shift) & 0xFFU),
                                scale_pair((hi >> shift) & 0xFFU), a);
                const std::uint32_t k_tile = block * kKPerLoad + j;
                const std::size_t base =
                    static_cast<std::size_t>(expert) * (columns / kTileK) * 4U;
                const uint2 b = live ? activations[base + k_tile * 4U + thread]
                                     : make_uint2(0U, 0U);
                mma_m16n8k16(acc[0], acc[1], acc[2], acc[3], a[0], a[1], a[2],
                             a[3], b.x, b.y);
            }
        }
        if (thread == 0U) {
            const std::uint32_t r[2] = {group, group + 8U};
            const int regs[2] = {0, 2};
            for (int which = 0; which < 2; ++which) {
                const std::size_t row =
                    static_cast<std::size_t>(expert) * rows +
                    n_tile * kTileN + r[which];
                partials[row * split + slice] = acc[regs[which]];
            }
        }
    }
}

// The ruler: pure streaming read over the same arena, nothing else.
__global__ void read_ruler_kernel(const uint4* __restrict__ data,
                                  std::size_t count, float* sink) {
    std::uint32_t accumulator = 0U;
    for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += gridDim.x * blockDim.x) {
        const uint4 value = data[index];
        accumulator ^= value.x ^ value.y ^ value.z ^ value.w;
    }
    if (accumulator == 0xDEAD'BEEFU) sink[0] = 1.0F;
}

double median(std::vector<double>& values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}

}  // namespace

int main(int argc, char** argv) {
    int device = 0;
    std::vector<std::uint32_t> widths{1U, 2U, 4U, 8U, 10U, 16U, 32U, 64U};
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument{argv[i]};
        if (argument == "--device" && i + 1 < argc) device = std::atoi(argv[++i]);
    }
    check(cudaSetDevice(device), "select device");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device), "query device");

    const std::uint64_t packed_gate = kIntermediate * (kHidden / 2U);
    const std::uint64_t scales_gate = kIntermediate * (kHidden / kGroup);
    const std::uint64_t packed_down = kHidden * (kIntermediate / 2U);
    const std::uint64_t scales_down = kHidden * (kIntermediate / kGroup);
    const std::uint64_t per_expert =
        2U * (packed_gate + scales_gate) + packed_down + scales_down;

    std::printf("device %d: %s, %.1f GB, L2 %.1f MB\n", device, properties.name,
                properties.totalGlobalMem / 1e9,
                properties.l2CacheSize / 1e6);
    std::printf("Laguna routed expert: gate/up packed %llu + scales %llu, "
                "down packed %llu + scales %llu; %.2f MiB per expert\n",
                (unsigned long long)packed_gate, (unsigned long long)scales_gate,
                (unsigned long long)packed_down, (unsigned long long)scales_down,
                per_expert / 1048576.0);
    std::printf("resident pool: %u experts, %.1f MiB (L2 is %.1f MiB)\n\n",
                kResidentExperts, kResidentExperts * per_expert / 1048576.0,
                properties.l2CacheSize / 1048576.0);

    std::vector<unsigned char*> gate_w(kResidentExperts), gate_s(kResidentExperts);
    std::vector<unsigned char*> up_w(kResidentExperts), up_s(kResidentExperts);
    std::vector<unsigned char*> down_w(kResidentExperts), down_s(kResidentExperts);
    const auto allocate = [](std::uint64_t bytes, unsigned char seed) {
        void* pointer = nullptr;
        check(cudaMalloc(&pointer, bytes), "allocate expert");
        check(cudaMemset(pointer, static_cast<int>(seed), bytes), "fill expert");
        return static_cast<unsigned char*>(pointer);
    };
    for (std::uint32_t e = 0U; e < kResidentExperts; ++e) {
        gate_w[e] = allocate(packed_gate, 0x42U + e);
        gate_s[e] = allocate(scales_gate, 0x7FU);
        up_w[e] = allocate(packed_gate, 0x24U + e);
        up_s[e] = allocate(scales_gate, 0x7FU);
        down_w[e] = allocate(packed_down, 0x13U + e);
        down_s[e] = allocate(scales_down, 0x7FU);
    }

    float* hidden = nullptr;
    float* activations = nullptr;
    float* output = nullptr;
    check(cudaMalloc(&hidden, kHidden * sizeof(float)), "hidden");
    check(cudaMemset(hidden, 0, kHidden * sizeof(float)), "hidden fill");
    check(cudaMalloc(&activations, 64ULL * kIntermediate * sizeof(float)), "acts");
    check(cudaMalloc(&output, 64ULL * kHidden * sizeof(float)), "out");

    cudaEvent_t start{}, stop{};
    check(cudaEventCreate(&start), "event");
    check(cudaEventCreate(&stop), "event");

    // ---- ruler ----
    const std::size_t ruler_bytes = 512ULL * 1024ULL * 1024ULL;
    uint4* ruler = nullptr;
    check(cudaMalloc(&ruler, ruler_bytes), "ruler arena");
    check(cudaMemset(ruler, 0x5AU, ruler_bytes), "ruler fill");
    float* sink = nullptr;
    check(cudaMalloc(&sink, sizeof(float)), "sink");
    const std::size_t ruler_count = ruler_bytes / sizeof(uint4);
    std::vector<double> ruler_samples;
    for (std::uint32_t i = 0U; i < kWarmups + kSamples; ++i) {
        check(cudaEventRecord(start), "record");
        read_ruler_kernel<<<properties.multiProcessorCount * 8, 256>>>(
            ruler, ruler_count, sink);
        check(cudaEventRecord(stop), "record");
        check(cudaEventSynchronize(stop), "sync");
        float ms = 0.0F;
        check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
        if (i >= kWarmups) ruler_samples.push_back(ruler_bytes / (ms / 1e3) / 1e9);
    }
    const double roofline = median(ruler_samples);
    std::printf("read ruler: %.1f GB/s (median of %u)\n\n", roofline, kSamples);

    std::vector<double> scalar_gate_bw(65, 0.0), scalar_down_bw(65, 0.0);
    std::printf("%-10s %8s %10s %10s %10s %10s\n", "kernel", "experts",
                "MiB read", "ms", "GB/s", "%% roofline");
    std::printf("%s\n", std::string(64, '-').c_str());

    for (const auto width : widths) {
        if (width > 64U) continue;
        Batch batch{};
        batch.count = width;
        batch.rows = 1U;
        for (std::uint32_t e = 0U; e < width; ++e) {
            const auto slot = e % kResidentExperts;
            batch.gate_weights[e] = gate_w[slot];
            batch.gate_scales[e] = gate_s[slot];
            batch.up_weights[e] = up_w[slot];
            batch.up_scales[e] = up_s[slot];
            batch.down_weights[e] = down_w[slot];
            batch.down_scales[e] = down_s[slot];
        }
        const double gate_bytes =
            static_cast<double>(width) * 2.0 * (packed_gate + scales_gate);
        const double down_bytes =
            static_cast<double>(width) * (packed_down + scales_down);

        std::vector<double> gate_ms, down_ms;
        for (std::uint32_t i = 0U; i < kWarmups + kSamples; ++i) {
            check(cudaEventRecord(start), "record");
            gate_up_kernel<<<dim3(kIntermediate, width), kThreads>>>(
                activations, hidden, batch, kHidden, kIntermediate,
                kHidden / 2U, kHidden / kGroup);
            check(cudaEventRecord(stop), "record");
            check(cudaEventSynchronize(stop), "sync");
            float ms = 0.0F;
            check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
            if (i >= kWarmups) gate_ms.push_back(ms);

            check(cudaEventRecord(start), "record");
            down_kernel<<<dim3(kHidden, width), kThreads>>>(
                output, activations, batch, kIntermediate, kHidden,
                kIntermediate / 2U, kIntermediate / kGroup);
            check(cudaEventRecord(stop), "record");
            check(cudaEventSynchronize(stop), "sync");
            check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
            if (i >= kWarmups) down_ms.push_back(ms);
        }
        check(cudaGetLastError(), "launch");
        const double g = median(gate_ms);
        const double d = median(down_ms);
        const double g_bw = gate_bytes / (g / 1e3) / 1e9;
        const double d_bw = down_bytes / (d / 1e3) / 1e9;
        scalar_gate_bw[width] = g_bw;
        scalar_down_bw[width] = d_bw;
        std::printf("%-10s %8u %10.2f %10.4f %10.1f %9.1f%%\n", "gate_up",
                    width, gate_bytes / 1048576.0, g, g_bw,
                    100.0 * g_bw / roofline);
        std::printf("%-10s %8u %10.2f %10.4f %10.1f %9.1f%%\n", "down", width,
                    down_bytes / 1048576.0, d, d_bw, 100.0 * d_bw / roofline);
    }
    // ---- register-fed arm ----
    // Fragment order is a pure permutation at identical byte count, so the same
    // allocations serve it and the memory traffic is unchanged. This arm
    // measures bandwidth only; the decode's numerics are established elsewhere
    // by the route-against-route test and by Gemma 4's 3.367x (experiment 0165).
    float* partials = nullptr;
    check(cudaMalloc(&partials, 64ULL * kHidden * 16ULL * sizeof(float)),
          "partials");
    uint2* act_frag = nullptr;
    check(cudaMalloc(&act_frag, 64ULL * (kHidden / 16U) * 4U * sizeof(uint2)),
          "activation fragments");
    check(cudaMemset(act_frag, 0, 64ULL * (kHidden / 16U) * 4U * sizeof(uint2)),
          "activation fill");

    std::printf("\n%-14s %8s %10s %10s %10s %10s %8s\n", "register-fed",
                "experts", "MiB read", "ms", "GB/s", "%% roofline", "vs scalar");
    std::printf("%s\n", std::string(74, '-').c_str());

    for (const auto width : widths) {
        if (width > 64U) continue;
        FragBatch batch{};
        batch.count = width;
        for (std::uint32_t e = 0U; e < width; ++e) {
            const auto slot = e % kResidentExperts;
            batch.gate_codes[e] = reinterpret_cast<const std::uint32_t*>(gate_w[slot]);
            batch.gate_scales[e] = gate_s[slot];
            batch.up_codes[e] = reinterpret_cast<const std::uint32_t*>(up_w[slot]);
            batch.up_scales[e] = up_s[slot];
            batch.down_codes[e] = reinterpret_cast<const std::uint32_t*>(down_w[slot]);
            batch.down_scales[e] = down_s[slot];
        }
        const double gate_bytes =
            static_cast<double>(width) * 2.0 * (packed_gate + scales_gate);
        const double down_bytes =
            static_cast<double>(width) * (packed_down + scales_down);

        // Enough warps to cover the device; partial traffic scales as split x M
        // and M is 1 here, so a wide split is cheap.
        const auto pick_split = [&](std::uint32_t k_blocks,
                                    std::uint32_t n_tiles) {
            std::uint32_t split = 1U;
            while (split < 16U && k_blocks % (split * 2U) == 0U &&
                   width * n_tiles * split * 2U <=
                       static_cast<std::uint32_t>(
                           properties.multiProcessorCount) * 32U) {
                split *= 2U;
            }
            return split;
        };
        const std::uint32_t gate_split =
            pick_split((kHidden / 16U) / 4U, kIntermediate / 16U);
        const std::uint32_t down_split =
            pick_split((kIntermediate / 16U) / 4U, kHidden / 16U);
        const auto blocks = [&](std::uint32_t n_tiles, std::uint32_t split) {
            return static_cast<unsigned int>(
                std::min<std::uint64_t>((static_cast<std::uint64_t>(width) *
                                             n_tiles * split + 3U) / 4U,
                                        65535U));
        };

        std::vector<double> gate_ms, down_ms;
        for (std::uint32_t i = 0U; i < kWarmups + kSamples; ++i) {
            check(cudaEventRecord(start), "record");
            regfed_gate_up_kernel<<<blocks(kIntermediate / 16U, gate_split),
                                    kWarpsPerBlock * 32U>>>(
                partials, act_frag, batch, kHidden, kIntermediate, gate_split);
            check(cudaEventRecord(stop), "record");
            check(cudaEventSynchronize(stop), "sync");
            float ms = 0.0F;
            check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
            if (i >= kWarmups) gate_ms.push_back(ms);

            check(cudaEventRecord(start), "record");
            regfed_down_kernel<<<blocks(kHidden / 16U, down_split),
                                 kWarpsPerBlock * 32U>>>(
                partials, act_frag, batch, kIntermediate, kHidden, down_split);
            check(cudaEventRecord(stop), "record");
            check(cudaEventSynchronize(stop), "sync");
            check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
            if (i >= kWarmups) down_ms.push_back(ms);
        }
        check(cudaGetLastError(), "launch register-fed");
        const double g = median(gate_ms);
        const double d = median(down_ms);
        const double g_bw = gate_bytes / (g / 1e3) / 1e9;
        const double d_bw = down_bytes / (d / 1e3) / 1e9;
        std::printf("%-14s %8u %10.2f %10.4f %10.1f %9.1f%% %7.2fx\n",
                    "gate_up", width, gate_bytes / 1048576.0, g, g_bw,
                    100.0 * g_bw / roofline, g_bw / scalar_gate_bw[width]);
        std::printf("%-14s %8u %10.2f %10.4f %10.1f %9.1f%% %7.2fx\n", "down",
                    width, down_bytes / 1048576.0, d, d_bw,
                    100.0 * d_bw / roofline, d_bw / scalar_down_bw[width]);
    }

    std::printf("\nA bandwidth-bound kernel holds its GB/s as the width grows.\n"
                "One that climbs with width was dispatch-bound at the narrow end.\n");
    return 0;
}
