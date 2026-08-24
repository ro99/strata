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

#include <strata/result.hpp>
#include <strata/safetensors.hpp>

#include <map>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

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
// Memory-level parallelism. At M=1 this kernel is latency-bound, not
// bandwidth- or compute-bound: decode is free (0139) and DRAM can serve
// 847 GB/s, but a warp that loads, waits, decodes and issues has only one
// request in flight. Hoisting kUnroll independent uint4 loads to the top of
// the iteration multiplies requests-in-flight per warp by kUnroll, which is
// the Ampere lever -- 4 loads cost 16 registers out of 255.
constexpr std::uint32_t kUnroll = 1U;
// F4-3 needs the M curve. One m16n8k16 covers 8 activation columns, so M<=8 is
// one column block and M=16 is two. Weight traffic is identical at every M --
// the same bytes serve more tokens -- which is the whole point of a skinny
// kernel.
constexpr std::uint32_t kMaxColBlocks = 2U;
constexpr std::uint32_t kWarpsPerBlock = 4U;
constexpr std::uint32_t kPageWarps = 8U;
constexpr std::uint32_t kPageM = kPageWarps * kMaxColBlocks * kTileM;
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
int g_inject_bad_scale = 0;
std::uint32_t g_sweep_layer = 0U;
bool g_correctness_only = false;
std::string g_checkpoint_shard;   // single shard, or empty when using --index
std::string g_tensor_prefix;      // e.g. layers.0.ffn.experts.0
std::string g_checkpoint_index;   // model.safetensors.index.json
std::string g_sweep;              // "L:E,L:E,..." multi-layer/expert fixture
int g_activation_token = -1;      // >=0 selects a real checkpoint-derived vector

// Resolves a tensor name to the shard holding it, so a fixture can span layers
// that live in different shards.
class Checkpoint {
  public:
    void open(const std::string& index_path, const std::string& root) {
        root_ = root;
        const auto text = strata::load_bounded_text_file(index_path, 64ULL << 20U);
        if (!text.ok()) throw std::runtime_error("cannot read index: " + index_path);
        const auto index = strata::parse_safetensors_index(text.value);
        if (!index.ok()) throw std::runtime_error("cannot parse index: " + index_path);
        for (const auto& e : index.value.entries) shard_of_[e.name] = e.shard;
    }

    [[nodiscard]] bool opened() const noexcept { return !shard_of_.empty(); }

    // Returns the shard path and tensor descriptor for `name`.
    std::pair<std::string, strata::SafetensorsTensor> find(
        const std::string& name) {
        std::string path = g_checkpoint_shard;
        if (opened()) {
            const auto it = shard_of_.find(name);
            if (it == shard_of_.end())
                throw std::runtime_error("tensor not in index: " + name);
            path = root_ + "/" + it->second;
        }
        auto& shard = shard_cache_[path];
        if (shard.tensors.empty()) {
            const auto loaded = strata::load_safetensors_shard(path);
            if (!loaded.ok())
                throw std::runtime_error("cannot read shard: " + path);
            shard = loaded.value;
        }
        for (const auto& t : shard.tensors) {
            if (t.name == name) return {path, t};
        }
        throw std::runtime_error("tensor not in shard: " + name);
    }

  private:
    std::string root_;
    std::map<std::string, std::string> shard_of_;
    std::map<std::string, strata::SafetensorsShard> shard_cache_;
};

Checkpoint g_checkpoint;

// Reads a single row of a 2-D tensor without materialising the whole thing.
// embed.weight alone is 1.06 GB; the fixture needs 8 KiB of it.
std::vector<std::byte> read_tensor_row(const std::string& path,
                                       const strata::SafetensorsTensor& t,
                                       std::uint64_t row) {
    if (t.shape.size() != 2U || row >= t.shape[0])
        throw std::runtime_error("bad row request for " + t.name);
    const std::uint64_t row_bytes = t.bytes() / t.shape[0];
    strata::SafetensorsTensor slice = t;
    slice.relative_begin = t.relative_begin + row * row_bytes;
    slice.relative_end = slice.relative_begin + row_bytes;
    slice.absolute_begin = t.absolute_begin + row * row_bytes;
    const auto bytes = strata::read_safetensors_tensor(path, slice, row_bytes);
    if (!bytes.ok()) throw std::runtime_error("cannot read row of " + t.name);
    return bytes.value;
}

float bf16_value(float value);  // defined below

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16U;
    float out = 0.0F;
    std::memcpy(&out, &wide, sizeof(out));
    return out;
}

// A checkpoint-derived activation: a real token embedding row put through the
// production RMSNorm with the layer's real ffn_norm weight. This is NOT a
// captured forward-pass activation -- it skips attention -- but every value and
// the per-channel scale structure come from the checkpoint, and the result is
// BF16-representable at the expert boundary as the contract requires.
bool load_real_activation(std::uint32_t layer, std::uint32_t k,
                          std::vector<float>& activation, std::uint32_t m) {
    if (g_activation_token < 0) return false;
    const auto [embed_path, embed] = g_checkpoint.find("embed.weight");
    const auto [norm_path, norm] = g_checkpoint.find(
        "layers." + std::to_string(layer) + ".ffn_norm.weight");
    const auto norm_bytes =
        strata::read_safetensors_tensor(norm_path, norm, norm.bytes());
    if (!norm_bytes.ok()) throw std::runtime_error("cannot read ffn_norm");

    for (std::uint32_t col = 0U; col < m; ++col) {
        const auto row_bytes = read_tensor_row(
            embed_path, embed,
            (static_cast<std::uint64_t>(g_activation_token) + col) %
                embed.shape[0]);
        const auto* hidden =
            reinterpret_cast<const std::uint16_t*>(row_bytes.data());
        const auto* gain =
            reinterpret_cast<const std::uint16_t*>(norm_bytes.value.data());
        double sum = 0.0;
        for (std::uint32_t i = 0U; i < k; ++i) {
            const double v = bf16_to_float(hidden[i]);
            sum += v * v;
        }
        const float scale =
            static_cast<float>(1.0 / std::sqrt(sum / static_cast<double>(k) +
                                               1.0e-6));
        for (std::uint32_t i = 0U; i < k; ++i) {
            activation[static_cast<std::size_t>(i) * m + col] = bf16_value(
                bf16_to_float(hidden[i]) * scale * bf16_to_float(gain[i]));
        }
    }
    return true;
}

// Load a real expert's packed E2M1 codes and E8M0 scales straight from the
// checkpoint, so the measured path is exercised on production bytes rather than
// synthetic ones. Returns false when no checkpoint was requested.
bool load_real_expert(const Shape& shape, std::vector<std::uint8_t>& codes,
                      std::vector<std::uint8_t>& scales) {
    if (g_checkpoint_shard.empty() && !g_checkpoint.opened()) return false;
    // gate_up_w1 is the checkpoint's w1, down_w2 is w2. Both are exactly
    // 4,194,304 packed bytes, so a byte-count check alone cannot tell them
    // apart - the declared shape must be checked too.
    const std::string suffix =
        (std::string_view(shape.name) == "down_w2") ? ".w2" : ".w1";
    const std::string weight_name = g_tensor_prefix + suffix + ".weight";
    const std::string scale_name = g_tensor_prefix + suffix + ".scale";
    const auto [weight_path, weight_t] = g_checkpoint.find(weight_name);
    const auto [scale_path, scale_t] = g_checkpoint.find(scale_name);
    const auto* weight = &weight_t;
    const auto* scale = &scale_t;

    // The checkpoint must actually carry the declared FP4 contract: packed
    // E2M1 at two codes per byte, and one E8M0 byte per group of 32 along K.
    if (weight->dtype != strata::SafetensorsDtype::I8)
        throw std::runtime_error(weight_name + " is not I8-packed E2M1");
    if (scale->dtype != strata::SafetensorsDtype::F8E8M0)
        throw std::runtime_error(scale_name + " is not E8M0");
    const std::uint64_t want_codes =
        static_cast<std::uint64_t>(shape.n) * (shape.k / 2U);
    const std::uint64_t want_scales =
        static_cast<std::uint64_t>(shape.n) * (shape.k / kGroup);
    if (weight->bytes() != want_codes || scale->bytes() != want_scales)
        throw std::runtime_error(weight_name + " byte count does not match " +
                                 std::string(shape.name));
    // Shape, not just size: [N, K/2] codes and [N, K/32] scales.
    if (weight->shape.size() != 2U || weight->shape[0] != shape.n ||
        weight->shape[1] != shape.k / 2U)
        throw std::runtime_error(weight_name + " shape is not [" +
                                 std::to_string(shape.n) + "," +
                                 std::to_string(shape.k / 2U) + "]");
    if (scale->shape.size() != 2U || scale->shape[0] != shape.n ||
        scale->shape[1] != shape.k / kGroup)
        throw std::runtime_error(scale_name + " shape is not [" +
                                 std::to_string(shape.n) + "," +
                                 std::to_string(shape.k / kGroup) + "]");

    const auto wbytes =
        strata::read_safetensors_tensor(weight_path, *weight, want_codes);
    const auto sbytes =
        strata::read_safetensors_tensor(scale_path, *scale, want_scales);
    if (!wbytes.ok() || !sbytes.ok())
        throw std::runtime_error("failed reading real expert tensors");
    codes.resize(static_cast<std::size_t>(shape.n) * shape.k);
    for (std::size_t byte = 0U; byte < want_codes; ++byte) {
        const auto packed = static_cast<std::uint8_t>(wbytes.value[byte]);
        codes[byte * 2U] = static_cast<std::uint8_t>(packed & 0x0FU);
        codes[byte * 2U + 1U] = static_cast<std::uint8_t>(packed >> 4U);
    }
    scales.assign(reinterpret_cast<const std::uint8_t*>(sbytes.value.data()),
                  reinterpret_cast<const std::uint8_t*>(sbytes.value.data()) +
                      want_scales);
    return true;
}
bool g_split_reduce = false;
// A single 4.46 MB matrix at M=1 cannot fill an RTX 3090: it yields 0.31 waves
// per SM, measured. Production MoE decode dispatches many routed experts per
// layer, so this batches independent expert matrices into one launch, which is
// the operating point the kernel will actually run at.
std::uint32_t g_batch = 1U;
std::uint32_t g_m = 1U;
bool g_gemma_page = false;
bool g_page_shared = false;
bool g_page_wmma = false;
std::uint32_t g_page_warps = kPageWarps;

constexpr Shape kShapes[] = {
    {"gate_up_w1", 2048U, 4096U},
    {"down_w2", 4096U, 2048U},
};

constexpr Shape kGemmaPageShapes[] = {
    {"gemma_gate_up", 21504U, 5376U},
    {"gemma_down", 5376U, 21504U},
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

// E8M0 admission. Both FP4 decoders map the scale code straight into a BF16
// exponent field, which is exact for codes 1-254 and silently wrong outside it:
// code 0 means 2^-127, subnormal in BF16 whose smallest normal is 2^-126, and
// encodes as +0; code 255 is the E8M0 NaN encoding and encodes as +inf. The
// contract requires exact mode to execute an approved route or report failure,
// never to substitute silently, so a region carrying either code must fail
// admission at load rather than produce a wrong value at decode.
constexpr std::uint8_t kE8m0AdmissibleMinimum = 1U;
constexpr std::uint8_t kE8m0AdmissibleMaximum = 254U;

struct ScaleAdmission {
    std::uint64_t inadmissible{0U};
    std::uint64_t subnormal_code_zero{0U};
    std::uint64_t nan_code_255{0U};
    std::size_t first_offset{0U};
    std::uint8_t first_code{0U};
    [[nodiscard]] bool admitted() const noexcept { return inadmissible == 0U; }
};

ScaleAdmission admit_e8m0_scales(const std::vector<std::uint8_t>& scales) {
    ScaleAdmission a{};
    for (std::size_t i = 0U; i < scales.size(); ++i) {
        const std::uint8_t code = scales[i];
        if (code >= kE8m0AdmissibleMinimum && code <= kE8m0AdmissibleMaximum) {
            continue;
        }
        if (a.inadmissible == 0U) {
            a.first_offset = i;
            a.first_code = code;
        }
        ++a.inadmissible;
        if (code == 0U) ++a.subnormal_code_zero; else ++a.nan_code_255;
    }
    return a;
}

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
// kColBlocks is a template parameter, not a runtime value: with it runtime the
// inner loop stops being fully unrolled and M=1 lost 24% (700.7 -> 531.0).
template <bool kBreakScaleBinding, bool kUseMma, std::uint32_t kColBlocks>
__global__ void prepacked_matmul_kernel(
    const std::uint32_t* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const uint2* __restrict__ activations, std::uint32_t k_extent,
    std::uint32_t k_tiles_per_slice, std::uint32_t split_k,
    float* __restrict__ partials, std::uint32_t* __restrict__ counters,
    float* __restrict__ output, bool fold_reduction,
    std::uint32_t n_tiles_per_matrix, std::uint32_t m,
    std::uint32_t groups_per_block) {
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint32_t global_warp = blockIdx.x * kWarpsPerBlock + warp;
    const std::uint32_t flat_tile = global_warp / split_k;
    const std::uint32_t n_tile = flat_tile % n_tiles_per_matrix;
    const std::uint32_t batch = flat_tile / n_tiles_per_matrix;
    const std::uint32_t slice = global_warp % split_k;
    const std::uint32_t k_tiles = k_extent / kTileK;
    // Each batch element is an independent expert matrix laid out end to end.
    codes += static_cast<std::size_t>(batch) * n_tiles_per_matrix * k_tiles *
             kWarp;
    scales += static_cast<std::size_t>(batch) * n_tiles_per_matrix *
              (k_extent / kGroup) * kTileN;
    const std::uint32_t group = lane >> 2U;
    const std::uint32_t thread = lane & 3U;
    __shared__ std::uint32_t arrived[kWarpsPerBlock];
    // Per-lane activation predicate and offset are loop-invariant, so they are
    // resolved once here rather than every K-tile.
    bool live[kColBlocks];
    std::size_t act_off[kColBlocks];
#pragma unroll
    for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
        live[c] = group < groups_per_block && c * kTileM + group < m;
        act_off[c] = (static_cast<std::size_t>(c) * groups_per_block + group) *
                         4U + thread;
    }

    float acc[kColBlocks][4];
#pragma unroll
    for (std::uint32_t c = 0U; c < kColBlocks; ++c)
#pragma unroll
        for (std::uint32_t i = 0U; i < 4U; ++i) acc[c][i] = 0.0F;

    const std::uint32_t k_blocks = k_tiles / kKPerLoad;
    const std::uint32_t shift = (group & 3U) * 8U;
    const std::uint32_t begin = slice * (k_tiles_per_slice / kKPerLoad);
    const std::uint32_t end = begin + k_tiles_per_slice / kKPerLoad;
    const uint4* code4 = reinterpret_cast<const uint4*>(codes);
    for (std::uint32_t kb = begin; kb < end && kb < k_blocks; kb += kUnroll) {
        // Every load for this iteration is issued before any of them is
        // consumed, so kUnroll DRAM requests per lane are in flight at once.
        uint4 packed[kUnroll];
        uint4 even[kUnroll];
        uint4 odd[kUnroll];

#pragma unroll
        for (std::uint32_t u = 0U; u < kUnroll; ++u) {
            const std::uint32_t block = kb + u;
            packed[u] = code4[(n_tile * k_blocks + block) * kWarp + lane];
            const unsigned char* base =
                scales + static_cast<std::size_t>(
                             n_tile * (k_extent / kGroup) + block * 2U) *
                             kTileN;
            even[u] = *reinterpret_cast<const uint4*>(base);
            odd[u] = *reinterpret_cast<const uint4*>(base + kTileN);

        }

#pragma unroll
        for (std::uint32_t u = 0U; u < kUnroll; ++u) {
        const std::uint32_t word[kKPerLoad] = {packed[u].x, packed[u].y,
                                               packed[u].z, packed[u].w};
        const uint4 scales_even = even[u];
        const uint4 scales_odd = odd[u];

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

            const std::uint32_t k_tile = (kb + u) * kKPerLoad + j;
            const std::size_t tile_base = static_cast<std::size_t>(k_tile) *
                                          kColBlocks * groups_per_block * 4U;
#pragma unroll
            for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
                // Only min(M,8) column groups are ever non-zero, so the store
                // holds groups_per_block x 4 fragments per column block rather
                // than a full 32-lane tile. At M=1 that is 4 entries read as a
                // broadcast; out-of-range columns are a predicated zero.
                const uint2 b = live[c]
                                    ? activations[tile_base + act_off[c]]
                                    : make_uint2(0U, 0U);
                if constexpr (kUseMma) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                        "{%0,%1,%2,%3};\n"
                        : "+f"(acc[c][0]), "+f"(acc[c][1]), "+f"(acc[c][2]),
                          "+f"(acc[c][3])
                        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                          "r"(b.x), "r"(b.y));
                } else {
                    acc[c][0] += __int_as_float(a[0] ^ b.x);
                    acc[c][1] += __int_as_float(a[1] ^ b.y);
                    acc[c][2] += __int_as_float(a[2]);
                    acc[c][3] += __int_as_float(a[3]);
                }
            }
        }
        }
    }

    // D fragment: row = group + (i>=2 ? 8 : 0), column = thread*2 + (i&1).
    // At M=1 only column 0 is real, which lives in d0 and d2 of the lanes with
    // thread == 0. Writing the full 16x8 tile would make 7/8 of the partial
    // traffic zeros -- measured at 24-94% of the useful weight bytes -- so only
    // the real column is stored.
    // D fragment: row = group + (i>=2 ? 8 : 0), column = thread*2 + (i&1),
    // offset by the column block. Columns beyond M are not stored.
    float* slot = partials + static_cast<std::size_t>(
                                 flat_tile * split_k + slice) * kTileN * m;
#pragma unroll
    for (std::uint32_t c = 0U; c < kColBlocks; ++c) {
#pragma unroll
        for (std::uint32_t i = 0U; i < 4U; ++i) {
            const std::uint32_t row = group + ((i >= 2U) ? 8U : 0U);
            const std::uint32_t col = c * kTileM + thread * 2U + (i & 1U);
            if (col < m) slot[row * m + col] = acc[c][i];
        }
    }

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
        arrived[warp] = atomicAdd(&counters[flat_tile], 1U);
    }
    __syncwarp();
    if (fold_reduction && arrived[warp] == split_k - 1U) {
        if (lane < kTileN) {
            for (std::uint32_t col = 0U; col < m; ++col) {
                float sum = 0.0F;
                for (std::uint32_t sl = 0U; sl < split_k; ++sl) {
                    sum += partials[(static_cast<std::size_t>(flat_tile) *
                                         split_k + sl) * kTileN * m +
                                    lane * m + col];
                }
                output[(flat_tile * kTileN + lane) * m + col] = sum;
            }
        }
        if (lane == 0U) counters[flat_tile] = 0U;
    }
}

__device__ __forceinline__ void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3, std::uint32_t a0,
    std::uint32_t a1, std::uint32_t a2, std::uint32_t a3, std::uint32_t b0,
    std::uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Gemma prefill page falsifier. One CTA owns one 16-row weight tile and its
// eight warps own disjoint 16-token bands. Every warp must decode the compact
// weight into its own MMA A registers, but the weight bytes themselves should
// reach HBM only once. The two arms decide whether Ampere's cache/miss merging
// is sufficient or whether the compact 544-byte code+scale tile must be
// broadcast explicitly through shared memory. No widened weight tile exists.
template <bool kSharedBroadcast>
__global__ __launch_bounds__(kPageWarps * kWarp) void page_matmul_kernel(
    const std::uint32_t* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const uint2* __restrict__ activations, std::uint32_t k_extent,
    float* __restrict__ output, std::uint32_t n_tiles) {
    constexpr std::uint32_t kPageColBlocks = kPageM / kTileM;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint32_t group = lane >> 2U;
    const std::uint32_t thread = lane & 3U;
    const std::uint32_t shift = (group & 3U) * 8U;
    const std::uint32_t k_tiles = k_extent / kTileK;
    const std::uint32_t k_blocks = k_tiles / kKPerLoad;
    const std::uint32_t scale_columns = k_extent / kGroup;
    __shared__ uint4 shared_codes[kWarp];
    __shared__ uint4 shared_scales[2];

    for (std::uint32_t n_tile = blockIdx.x; n_tile < n_tiles;
         n_tile += gridDim.x) {
        float acc[kMaxColBlocks][4]{};
        const uint4* code4 = reinterpret_cast<const uint4*>(codes);
        for (std::uint32_t block = 0U; block < k_blocks; ++block) {
            const unsigned char* scale_base =
                scales + (static_cast<std::size_t>(n_tile) * scale_columns +
                          block * 2U) * kTileN;
            uint4 packed{};
            uint4 even{};
            uint4 odd{};
            if constexpr (kSharedBroadcast) {
                if (warp == 0U) {
                    shared_codes[lane] =
                        code4[(static_cast<std::size_t>(n_tile) * k_blocks +
                               block) * kWarp + lane];
                    if (lane < 2U) {
                        shared_scales[lane] = *reinterpret_cast<const uint4*>(
                            scale_base + lane * kTileN);
                    }
                }
                __syncthreads();
                packed = shared_codes[lane];
                even = shared_scales[0];
                odd = shared_scales[1];
            } else {
                packed = code4[(static_cast<std::size_t>(n_tile) * k_blocks +
                                block) * kWarp + lane];
                even = *reinterpret_cast<const uint4*>(scale_base);
                odd = *reinterpret_cast<const uint4*>(scale_base + kTileN);
            }
            const std::uint32_t words[kKPerLoad] = {
                packed.x, packed.y, packed.z, packed.w};
#pragma unroll
            for (std::uint32_t j = 0U; j < kKPerLoad; ++j) {
                const uint4 selected = j < 2U ? even : odd;
                const std::uint32_t low = group < 4U ? selected.x : selected.y;
                const std::uint32_t high = group < 4U ? selected.z : selected.w;
                std::uint32_t a[4];
                decode_fragment<false>(
                    words[j], scale_pair_bf16((low >> shift) & 0xFFU),
                    scale_pair_bf16((high >> shift) & 0xFFU), a);
                const std::uint32_t k_tile = block * kKPerLoad + j;
#pragma unroll
                for (std::uint32_t c = 0U; c < kMaxColBlocks; ++c) {
                    const std::uint32_t column_block =
                        warp * kMaxColBlocks + c;
                    const std::size_t activation_index =
                        ((static_cast<std::size_t>(k_tile) * kPageColBlocks +
                          column_block) * kTileM + group) * 4U + thread;
                    const uint2 b = activations[activation_index];
                    mma_m16n8k16(acc[c][0], acc[c][1], acc[c][2], acc[c][3],
                                 a[0], a[1], a[2], a[3], b.x, b.y);
                }
            }
            if constexpr (kSharedBroadcast) __syncthreads();
        }

#pragma unroll
        for (std::uint32_t c = 0U; c < kMaxColBlocks; ++c) {
#pragma unroll
            for (std::uint32_t i = 0U; i < 4U; ++i) {
                const std::uint32_t row =
                    n_tile * kTileN + group + (i >= 2U ? 8U : 0U);
                const std::uint32_t column =
                    (warp * kMaxColBlocks + c) * kTileM +
                    thread * 2U + (i & 1U);
                output[static_cast<std::size_t>(row) * kPageM + column] =
                    acc[c][i];
            }
        }
    }
}

// Conventional page GEMM control. A 64x128 CTA widens one compact K32 scale
// tile into BF16 shared memory, then reuses it across 64 activation rows and
// feeds BF16 tensor cores. At M=128 every compact weight is read twice rather
// than the decode-oriented route's eight times. The checkpoint remains in its
// canonical compact representation; widened weights are transient only.
constexpr std::uint32_t kWmmaBlockM = 64U;
constexpr std::uint32_t kWmmaBlockN = 128U;
constexpr std::uint32_t kWmmaBlockK = 32U;

__device__ __forceinline__ float page_fp4_value(unsigned int code) {
    const unsigned int magnitude = code & 7U;
    float value = 0.0F;
    if (magnitude <= 3U) value = 0.5F * static_cast<float>(magnitude);
    else if (magnitude == 4U) value = 2.0F;
    else if (magnitude == 5U) value = 3.0F;
    else if (magnitude == 6U) value = 4.0F;
    else value = 6.0F;
    return (code & 8U) != 0U ? -value : value;
}

__global__ __launch_bounds__(8U * kWarp) void page_wmma_matmul_kernel(
    const unsigned char* __restrict__ packed_codes,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ activations, std::uint32_t m_extent,
    std::uint32_t n_extent, std::uint32_t k_extent,
    float* __restrict__ output) {
    namespace wmma = nvcuda::wmma;
    __shared__ __nv_bfloat16 shared_a[kWmmaBlockM * kWmmaBlockK];
    __shared__ __nv_bfloat16 shared_b[kWmmaBlockK * kWmmaBlockN];
    __shared__ float shared_output[8U * 16U * 16U];

    const std::uint32_t tile_m = blockIdx.y * kWmmaBlockM;
    const std::uint32_t tile_n = blockIdx.x * kWmmaBlockN;
    const std::uint32_t warp = threadIdx.x / kWarp;
    const std::uint32_t warp_m = warp & 3U;
    const std::uint32_t warp_n_group = warp >> 2U;
    constexpr std::uint32_t kFragmentsPerWarp = 4U;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>
        accumulators[kFragmentsPerWarp];
#pragma unroll
    for (std::uint32_t fragment = 0U; fragment < kFragmentsPerWarp;
         ++fragment) {
        wmma::fill_fragment(accumulators[fragment], 0.0F);
    }

    const std::uint32_t packed_columns = k_extent / 2U;
    const std::uint32_t scale_columns = k_extent / kGroup;
    for (std::uint32_t tile_k = 0U; tile_k < k_extent;
         tile_k += kWmmaBlockK) {
        for (std::uint32_t index = threadIdx.x;
             index < kWmmaBlockM * kWmmaBlockK; index += blockDim.x) {
            const std::uint32_t local_k = index / kWmmaBlockM;
            const std::uint32_t local_m = index % kWmmaBlockM;
            const std::uint32_t global_m = tile_m + local_m;
            const float value = global_m < m_extent
                ? activations[static_cast<std::size_t>(tile_k + local_k) *
                                  m_extent + global_m]
                : 0.0F;
            shared_a[local_m * kWmmaBlockK + local_k] =
                __float2bfloat16_rn(value);
        }
        constexpr std::uint32_t kPackedTileBytes =
            kWmmaBlockN * kWmmaBlockK / 2U;
        for (std::uint32_t index = threadIdx.x; index < kPackedTileBytes;
             index += blockDim.x) {
            const std::uint32_t local_n = index / (kWmmaBlockK / 2U);
            const std::uint32_t local_pair = index % (kWmmaBlockK / 2U);
            const std::uint32_t global_n = tile_n + local_n;
            unsigned char packed = 0U;
            unsigned char encoded_scale = 127U;
            if (global_n < n_extent) {
                packed = packed_codes[
                    static_cast<std::size_t>(global_n) * packed_columns +
                    tile_k / 2U + local_pair];
                encoded_scale = scales[
                    static_cast<std::size_t>(global_n) * scale_columns +
                    tile_k / kGroup];
            }
            const float scale = ldexpf(1.0F,
                                       static_cast<int>(encoded_scale) - 127);
            const std::uint32_t local_k = local_pair * 2U;
            shared_b[local_k * kWmmaBlockN + local_n] =
                __float2bfloat16_rn(
                    page_fp4_value(static_cast<unsigned int>(packed & 0x0FU)) *
                    scale);
            shared_b[(local_k + 1U) * kWmmaBlockN + local_n] =
                __float2bfloat16_rn(
                    page_fp4_value(static_cast<unsigned int>(packed >> 4U)) *
                    scale);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                       wmma::row_major> a_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                       wmma::row_major> b_fragment;
#pragma unroll
        for (std::uint32_t local_k = 0U; local_k < kWmmaBlockK;
             local_k += 16U) {
            wmma::load_matrix_sync(
                a_fragment,
                shared_a + warp_m * 16U * kWmmaBlockK + local_k,
                kWmmaBlockK);
#pragma unroll
            for (std::uint32_t fragment = 0U;
                 fragment < kFragmentsPerWarp; ++fragment) {
                const std::uint32_t fragment_n =
                    warp_n_group * kFragmentsPerWarp + fragment;
                wmma::load_matrix_sync(
                    b_fragment,
                    shared_b + local_k * kWmmaBlockN + fragment_n * 16U,
                    kWmmaBlockN);
                wmma::mma_sync(accumulators[fragment], a_fragment, b_fragment,
                               accumulators[fragment]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (std::uint32_t fragment = 0U; fragment < kFragmentsPerWarp;
         ++fragment) {
        float* warp_output = shared_output + warp * 16U * 16U;
        wmma::store_matrix_sync(warp_output, accumulators[fragment], 16U,
                                wmma::mem_row_major);
        __syncwarp();
        const std::uint32_t fragment_n =
            warp_n_group * kFragmentsPerWarp + fragment;
        for (std::uint32_t element = threadIdx.x & 31U; element < 256U;
             element += kWarp) {
            const std::uint32_t local_m = element / 16U;
            const std::uint32_t local_n = element % 16U;
            const std::uint32_t global_m = tile_m + warp_m * 16U + local_m;
            const std::uint32_t global_n =
                tile_n + fragment_n * 16U + local_n;
            if (global_m < m_extent && global_n < n_extent) {
                output[static_cast<std::size_t>(global_m) * n_extent +
                       global_n] = warp_output[element];
            }
        }
        __syncwarp();
    }
}

// Ownership sweep prompted by the profile of page_matmul_kernel<false>:
// DRAM was only 11% busy while duplicate cache hits saturated L2 at 89.8% and
// duplicate FP4 decode drove ALU to 68.4%. Fewer warps per tile give each warp
// more activation columns, keeping the MMA count fixed while reducing both
// measured terms in direct proportion. Four warps per CTA pack independent
// N-tiles when one tile needs fewer than four warps.
template <std::uint32_t kWarpsPerTile>
__global__ __launch_bounds__(4U * kWarp) void page_owned_matmul_kernel(
    const std::uint32_t* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const uint2* __restrict__ activations, std::uint32_t k_extent,
    float* __restrict__ output, std::uint32_t n_tiles) {
    static_assert(kWarpsPerTile == 1U || kWarpsPerTile == 2U ||
                  kWarpsPerTile == 4U);
    constexpr std::uint32_t kPageColBlocks = kPageM / kTileM;
    constexpr std::uint32_t kLocalColBlocks =
        kPageColBlocks / kWarpsPerTile;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp_in_block = threadIdx.x >> 5U;
    const std::uint32_t global_warp = blockIdx.x * 4U + warp_in_block;
    const std::uint32_t n_tile = global_warp / kWarpsPerTile;
    if (n_tile >= n_tiles) return;
    const std::uint32_t tile_warp = global_warp % kWarpsPerTile;
    const std::uint32_t group = lane >> 2U;
    const std::uint32_t thread = lane & 3U;
    const std::uint32_t shift = (group & 3U) * 8U;
    const std::uint32_t k_tiles = k_extent / kTileK;
    const std::uint32_t k_blocks = k_tiles / kKPerLoad;
    const std::uint32_t scale_columns = k_extent / kGroup;
    float acc[kLocalColBlocks][4]{};
    const uint4* code4 = reinterpret_cast<const uint4*>(codes);

    for (std::uint32_t block = 0U; block < k_blocks; ++block) {
        const uint4 packed =
            code4[(static_cast<std::size_t>(n_tile) * k_blocks + block) *
                      kWarp + lane];
        const unsigned char* scale_base =
            scales + (static_cast<std::size_t>(n_tile) * scale_columns +
                      block * 2U) * kTileN;
        const uint4 even = *reinterpret_cast<const uint4*>(scale_base);
        const uint4 odd =
            *reinterpret_cast<const uint4*>(scale_base + kTileN);
        const std::uint32_t words[kKPerLoad] = {
            packed.x, packed.y, packed.z, packed.w};
#pragma unroll
        for (std::uint32_t j = 0U; j < kKPerLoad; ++j) {
            const uint4 selected = j < 2U ? even : odd;
            const std::uint32_t low = group < 4U ? selected.x : selected.y;
            const std::uint32_t high = group < 4U ? selected.z : selected.w;
            std::uint32_t a[4];
            decode_fragment<false>(
                words[j], scale_pair_bf16((low >> shift) & 0xFFU),
                scale_pair_bf16((high >> shift) & 0xFFU), a);
            const std::uint32_t k_tile = block * kKPerLoad + j;
#pragma unroll
            for (std::uint32_t c = 0U; c < kLocalColBlocks; ++c) {
                const std::uint32_t column_block =
                    tile_warp * kLocalColBlocks + c;
                const std::size_t activation_index =
                    ((static_cast<std::size_t>(k_tile) * kPageColBlocks +
                      column_block) * kTileM + group) * 4U + thread;
                const uint2 b = activations[activation_index];
                mma_m16n8k16(acc[c][0], acc[c][1], acc[c][2], acc[c][3],
                             a[0], a[1], a[2], a[3], b.x, b.y);
            }
        }
    }

#pragma unroll
    for (std::uint32_t c = 0U; c < kLocalColBlocks; ++c) {
#pragma unroll
        for (std::uint32_t i = 0U; i < 4U; ++i) {
            const std::uint32_t row =
                n_tile * kTileN + group + (i >= 2U ? 8U : 0U);
            const std::uint32_t column =
                (tile_warp * kLocalColBlocks + c) * kTileM +
                thread * 2U + (i & 1U);
            output[static_cast<std::size_t>(row) * kPageM + column] = acc[c][i];
        }
    }
}

__global__ void reduce_kernel(const float* __restrict__ partials,
                              std::uint32_t n_tiles, std::uint32_t split_k,
                              std::uint32_t m, float* __restrict__ out) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = n_tiles * kTileN * m;
    if (index >= total) return;
    const std::uint32_t n_tile = index / (kTileN * m);
    const std::uint32_t row = (index / m) % kTileN;
    const std::uint32_t col = index % m;
    float sum = 0.0F;
    for (std::uint32_t slice = 0U; slice < split_k; ++slice) {
        sum += partials[(static_cast<std::size_t>(n_tile) * split_k + slice) *
                            kTileN * m + row * m + col];
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
    ScaleAdmission admission{};
    bool real_weights{false};
    bool real_activations{false};
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
    std::vector<float> activation(static_cast<std::size_t>(k) * g_m);
    std::uint32_t state = 0x5EED'1234U ^ n ^ (k << 7U);
    r.real_weights = load_real_expert(shape, canon_codes, canon_scales);
    if (!r.real_weights) {
        for (auto& c : canon_codes)
            c = static_cast<std::uint8_t>(xorshift(state) & 0x0FU);
        for (auto& s : canon_scales)
            s = static_cast<std::uint8_t>(120U + xorshift(state) % 15U);
    }
    if (g_inject_bad_scale != 0) {
        // Proves the admission check can fail. A check that never fires its
        // own control is not a check.
        canon_scales[canon_scales.size() / 3U] =
            static_cast<std::uint8_t>(g_inject_bad_scale > 0 ? 255U : 0U);
    }
    r.admission = admit_e8m0_scales(canon_scales);
    if (!r.admission.admitted()) {
        // Report and stop for this shape rather than decoding a wrong value.
        return r;
    }
    r.real_activations =
        load_real_activation(g_sweep_layer, k, activation, g_m);
    if (!r.real_activations) {
        for (auto& a : activation)
            a = bf16_value(
                static_cast<float>(static_cast<int>(xorshift(state) % 9U) - 4));
    }

    std::vector<std::uint8_t> canonical_packed_codes(r.code_bytes);
    for (std::size_t byte = 0U; byte < canonical_packed_codes.size(); ++byte) {
        canonical_packed_codes[byte] = static_cast<std::uint8_t>(
            canon_codes[byte * 2U] | (canon_codes[byte * 2U + 1U] << 4U));
    }

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
    const std::uint32_t col_blocks = (g_m + kTileM - 1U) / kTileM;
    const std::uint32_t groups_per_block = std::min(g_m, kTileM);
    std::vector<std::uint32_t> activation_frag(
        static_cast<std::size_t>(k_tiles) * col_blocks * groups_per_block * 4U *
        2U);
    for (std::uint32_t kt = 0U; kt < k_tiles; ++kt) {
        for (std::uint32_t cb = 0U; cb < col_blocks; ++cb) {
            for (std::uint32_t g = 0U; g < groups_per_block; ++g) {
                for (std::uint32_t t = 0U; t < 4U; ++t) {
                    const std::uint32_t col = cb * kTileM + g;
                    std::uint32_t b0 = 0U, b1 = 0U;
                    if (col < g_m) {
                        const auto at = [&](std::uint32_t row) {
                            return static_cast<std::uint32_t>(bf16_bits(
                                activation[static_cast<std::size_t>(
                                               kt * kTileK + row) * g_m +
                                           col]));
                        };
                        b0 = at(t * 2U) | (at(t * 2U + 1U) << 16U);
                        b1 = at(t * 2U + 8U) | (at(t * 2U + 9U) << 16U);
                    }
                    const std::size_t idx =
                        (((static_cast<std::size_t>(kt) * col_blocks + cb) *
                              groups_per_block + g) * 4U + t) * 2U;
                    activation_frag[idx] = b0;
                    activation_frag[idx + 1U] = b1;
                }
            }
        }
    }

    DeviceBuffer d_codes(r.prepacked_code_bytes * g_batch);
    DeviceBuffer d_scales(r.prepacked_scale_bytes * g_batch);
    DeviceBuffer d_act(activation_frag.size() * sizeof(std::uint32_t));
    DeviceBuffer d_canonical_codes(canonical_packed_codes.size());
    DeviceBuffer d_canonical_scales(canon_scales.size());
    DeviceBuffer d_canonical_act(activation.size() * sizeof(float));
    DeviceBuffer d_partials(static_cast<std::size_t>(n_tiles) * g_batch *
                            g_split_k * kTileN * g_m * sizeof(float));
    DeviceBuffer d_out(static_cast<std::size_t>(n) * g_batch * g_m *
                       sizeof(float));
    DeviceBuffer d_counters(static_cast<std::size_t>(n_tiles) * g_batch *
                            sizeof(std::uint32_t));
    check(cudaMemsetAsync(d_counters.get(), 0, d_counters.bytes(), stream),
          "zero split-K counters");
    DeviceBuffer d_scrub(kScrubBytes);
    r.device_bytes = d_codes.bytes() + d_scales.bytes() + d_act.bytes() +
                     d_partials.bytes() + d_out.bytes() +
                     d_canonical_codes.bytes() + d_canonical_scales.bytes() +
                     d_canonical_act.bytes();

    for (std::uint32_t b = 0U; b < g_batch; ++b) {
        check(cudaMemcpyAsync(
                  static_cast<unsigned char*>(d_codes.get()) +
                      static_cast<std::size_t>(b) * r.prepacked_code_bytes,
                  frag_codes.data(), r.prepacked_code_bytes,
                  cudaMemcpyHostToDevice, stream), "upload fragment codes");
        check(cudaMemcpyAsync(
                  static_cast<unsigned char*>(d_scales.get()) +
                      static_cast<std::size_t>(b) * r.prepacked_scale_bytes,
                  frag_scales.data(), r.prepacked_scale_bytes,
                  cudaMemcpyHostToDevice, stream), "upload fragment scales");
    }
    check(cudaMemcpyAsync(d_act.get(), activation_frag.data(),
                          d_act.bytes(), cudaMemcpyHostToDevice, stream),
          "upload B-fragment activations");
    check(cudaMemcpyAsync(d_canonical_codes.get(), canonical_packed_codes.data(),
                          d_canonical_codes.bytes(), cudaMemcpyHostToDevice,
                          stream), "upload canonical packed codes");
    check(cudaMemcpyAsync(d_canonical_scales.get(), canon_scales.data(),
                          d_canonical_scales.bytes(), cudaMemcpyHostToDevice,
                          stream), "upload canonical scales");
    check(cudaMemcpyAsync(d_canonical_act.get(), activation.data(),
                          d_canonical_act.bytes(), cudaMemcpyHostToDevice,
                          stream), "upload canonical activations");
    check(cudaStreamSynchronize(stream), "finish uploads");

    const std::uint32_t k_blocks_total = k_tiles / kKPerLoad;
    if (k_blocks_total % (g_split_k * kUnroll) != 0U) {
        throw std::runtime_error(
            "split-K " + std::to_string(g_split_k) +
            " leaves " + std::to_string(k_blocks_total / g_split_k) +
            " blocks per slice, which is not a multiple of the unroll " +
            std::to_string(kUnroll) + "; this would silently skip work");
    }
    const std::uint32_t warps = n_tiles * g_batch * g_split_k;
    const std::uint32_t blocks = warps / kWarpsPerBlock;
    const std::uint32_t k_tiles_per_slice = k_tiles / g_split_k;

    const auto launch_reduce = [&] {
        if (g_gemma_page) return;
        reduce_kernel<<<(n * g_batch * g_m + 255U) / 256U, 256U, 0U, stream>>>(
            static_cast<const float*>(d_partials.get()), n_tiles * g_batch,
            g_split_k, g_m, static_cast<float*>(d_out.get()));
    };
    const auto launch_matmul = [&] {
        if (g_gemma_page) {
            if (g_page_wmma) {
                const dim3 grid((n + kWmmaBlockN - 1U) / kWmmaBlockN,
                                (g_m + kWmmaBlockM - 1U) / kWmmaBlockM, 1U);
                page_wmma_matmul_kernel<<<grid, 8U * kWarp, 0U, stream>>>(
                    static_cast<const unsigned char*>(d_canonical_codes.get()),
                    static_cast<const unsigned char*>(d_canonical_scales.get()),
                    static_cast<const float*>(d_canonical_act.get()), g_m, n, k,
                    static_cast<float*>(d_out.get()));
            } else if (g_page_shared) {
                page_matmul_kernel<true><<<
                    std::min<std::uint32_t>(n_tiles, 65535U),
                    kPageWarps * kWarp, 0U, stream>>>(
                    static_cast<const std::uint32_t*>(d_codes.get()),
                    static_cast<const unsigned char*>(d_scales.get()),
                    static_cast<const uint2*>(d_act.get()), k,
                    static_cast<float*>(d_out.get()), n_tiles);
            } else if (g_page_warps == 1U) {
                page_owned_matmul_kernel<1U><<<
                    (n_tiles + 3U) / 4U, 4U * kWarp, 0U, stream>>>(
                    static_cast<const std::uint32_t*>(d_codes.get()),
                    static_cast<const unsigned char*>(d_scales.get()),
                    static_cast<const uint2*>(d_act.get()), k,
                    static_cast<float*>(d_out.get()), n_tiles);
            } else if (g_page_warps == 2U) {
                page_owned_matmul_kernel<2U><<<
                    (n_tiles * 2U + 3U) / 4U, 4U * kWarp, 0U, stream>>>(
                    static_cast<const std::uint32_t*>(d_codes.get()),
                    static_cast<const unsigned char*>(d_scales.get()),
                    static_cast<const uint2*>(d_act.get()), k,
                    static_cast<float*>(d_out.get()), n_tiles);
            } else if (g_page_warps == 4U) {
                page_owned_matmul_kernel<4U><<<
                    n_tiles, 4U * kWarp, 0U, stream>>>(
                    static_cast<const std::uint32_t*>(d_codes.get()),
                    static_cast<const unsigned char*>(d_scales.get()),
                    static_cast<const uint2*>(d_act.get()), k,
                    static_cast<float*>(d_out.get()), n_tiles);
            } else {
                page_matmul_kernel<false><<<
                    std::min<std::uint32_t>(n_tiles, 65535U),
                    kPageWarps * kWarp, 0U, stream>>>(
                    static_cast<const std::uint32_t*>(d_codes.get()),
                    static_cast<const unsigned char*>(d_scales.get()),
                    static_cast<const uint2*>(d_act.get()), k,
                    static_cast<float*>(d_out.get()), n_tiles);
            }
        } else if (g_no_mma)
            prepacked_matmul_kernel<false, false, 1U>
                <<<blocks, kWarpsPerBlock * kWarp, 0U, stream>>>(
                    static_cast<const std::uint32_t*>(d_codes.get()),
                    static_cast<const unsigned char*>(d_scales.get()),
                    static_cast<const uint2*>(d_act.get()), k,
                    k_tiles_per_slice, g_split_k,
                    static_cast<float*>(d_partials.get()),
                    static_cast<std::uint32_t*>(d_counters.get()),
                    static_cast<float*>(d_out.get()), !g_split_reduce, n_tiles, g_m,
            groups_per_block);
        else if (g_break_scale_binding)
            prepacked_matmul_kernel<true, true, 1U>
                <<<blocks, kWarpsPerBlock * kWarp, 0U, stream>>>(
                    static_cast<const std::uint32_t*>(d_codes.get()),
                    static_cast<const unsigned char*>(d_scales.get()),
                    static_cast<const uint2*>(d_act.get()), k,
                    k_tiles_per_slice, g_split_k,
                    static_cast<float*>(d_partials.get()),
                    static_cast<std::uint32_t*>(d_counters.get()),
                    static_cast<float*>(d_out.get()), !g_split_reduce, n_tiles, g_m,
            groups_per_block);
        else if (col_blocks == 2U)
            prepacked_matmul_kernel<false, true, 2U>
                <<<blocks, kWarpsPerBlock * kWarp, 0U, stream>>>(
                    static_cast<const std::uint32_t*>(d_codes.get()),
                    static_cast<const unsigned char*>(d_scales.get()),
                    static_cast<const uint2*>(d_act.get()), k,
                    k_tiles_per_slice, g_split_k,
                    static_cast<float*>(d_partials.get()),
                    static_cast<std::uint32_t*>(d_counters.get()),
                    static_cast<float*>(d_out.get()), !g_split_reduce, n_tiles,
                    g_m, groups_per_block);
        else
        prepacked_matmul_kernel<false, true, 1U>
            <<<blocks, kWarpsPerBlock * kWarp, 0U, stream>>>(
            static_cast<const std::uint32_t*>(d_codes.get()),
            static_cast<const unsigned char*>(d_scales.get()),
            static_cast<const uint2*>(d_act.get()), k,
            k_tiles_per_slice, g_split_k,
            static_cast<float*>(d_partials.get()),
            static_cast<std::uint32_t*>(d_counters.get()),
            static_cast<float*>(d_out.get()), !g_split_reduce, n_tiles, g_m,
            groups_per_block);
    };
    const auto launch = [&] {
        launch_matmul();
        if (g_split_reduce) launch_reduce();
    };

    launch();
    check(cudaGetLastError(), "launch prepacked matmul");
    check(cudaStreamSynchronize(stream), "finish correctness launch");

    std::vector<float> out(static_cast<std::size_t>(n) * g_m);
    check(cudaMemcpyAsync(out.data(), d_out.get(),
                          out.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, stream), "download output");
    check(cudaStreamSynchronize(stream), "finish download");

    // ---- oracle: double precision, from the CANONICAL layout ----
    const std::uint32_t oracle_row_step =
        g_gemma_page ? std::max<std::uint32_t>(n / 32U, 1U) : 1U;
    const std::uint32_t oracle_column_step = g_gemma_page ? 16U : 1U;
    for (std::uint32_t row = 0U; row < n; row += oracle_row_step) {
        for (std::uint32_t mcol = 0U; mcol < g_m;
             mcol += oracle_column_step) {
            double sum = 0.0;
            for (std::uint32_t col = 0U; col < k; ++col) {
                const double w =
                    fp4_value(
                        canon_codes[static_cast<std::size_t>(row) * k + col]) *
                    e8m0(canon_scales[static_cast<std::size_t>(row) *
                                          (k / kGroup) + col / kGroup]);
                sum += w * static_cast<double>(
                               activation[static_cast<std::size_t>(col) * g_m +
                                          mcol]);
            }
            const float got = g_page_wmma
                ? out[static_cast<std::size_t>(mcol) * n + row]
                : out[static_cast<std::size_t>(row) * g_m + mcol];
            r.oracle_max_abs = std::max(r.oracle_max_abs, std::abs(sum));
            r.output_max_abs =
                std::max(r.output_max_abs, std::abs(static_cast<double>(got)));
            if (got != 0.0F) ++r.nonzero_outputs;
            const double diff = std::abs(static_cast<double>(got) - sum);
            r.max_absolute = std::max(r.max_absolute, diff);
            r.max_relative =
                std::max(r.max_relative, diff / std::max(std::abs(sum), 1.0));
        }
    }

    // ---- timing ----
    if (g_correctness_only) return r;
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
        r.pipelined_gbps = static_cast<double>(r.useful_bytes * g_batch) /
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
            else if (f == "--checkpoint" && i + 1 < argc)
                g_checkpoint_shard = argv[++i];
            else if (f == "--index" && i + 1 < argc) {
                const std::string idx = argv[++i];
                const auto slash = idx.find_last_of('/');
                g_checkpoint.open(idx, slash == std::string::npos
                                           ? std::string(".")
                                           : idx.substr(0U, slash));
            }
            else if (f == "--sweep" && i + 1 < argc) g_sweep = argv[++i];
            else if (f == "--correctness-only") g_correctness_only = true;
            else if (f == "--activation-token" && i + 1 < argc)
                g_activation_token = std::stoi(argv[++i]);
            else if (f == "--tensor" && i + 1 < argc) {
                g_tensor_prefix = argv[++i];
                // The activation's RMSNorm gain must come from the same layer
                // as the expert, so derive it rather than defaulting to 0.
                const auto begin = g_tensor_prefix.find("layers.");
                if (begin != std::string::npos) {
                    g_sweep_layer = static_cast<std::uint32_t>(
                        std::stoul(g_tensor_prefix.substr(begin + 7U)));
                }
            }
            else if (f == "--inject-scale-nan") g_inject_bad_scale = 1;
            else if (f == "--inject-scale-zero") g_inject_bad_scale = -1;
            else if (f == "--split-reduce") g_split_reduce = true;
            else if (f == "--gemma-page") g_gemma_page = true;
            else if (f == "--page-shared") {
                g_gemma_page = true;
                g_page_shared = true;
            }
            else if (f == "--page-wmma") {
                g_gemma_page = true;
                g_page_wmma = true;
            }
            else if (f == "--page-warps" && i + 1 < argc) {
                g_gemma_page = true;
                g_page_warps =
                    static_cast<std::uint32_t>(std::stoul(argv[++i]));
                if (g_page_warps != 1U && g_page_warps != 2U &&
                    g_page_warps != 4U && g_page_warps != 8U)
                    throw std::runtime_error("page warps must be 1, 2, 4, or 8");
            }
            else if (f == "--m" && i + 1 < argc) {
                g_m = static_cast<std::uint32_t>(std::stoul(argv[++i]));
                if (g_m == 0U || g_m > kPageM)
                    throw std::runtime_error("M must be 1..128");
            }
            else if (f == "--batch" && i + 1 < argc)
                g_batch = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            else if (f == "--split-k" && i + 1 < argc)
                g_split_k = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            else { std::cerr << "usage: " << argv[0]
                             << " [--device INDEX] [--output PATH]\n";
                   return EXIT_FAILURE; }
        }
        if (g_gemma_page) {
            if (g_m != 1U && g_m != kPageM)
                throw std::runtime_error("Gemma page arm requires M=128");
            g_m = kPageM;
            g_split_k = 1U;
            g_batch = 1U;
            g_split_reduce = false;
        } else if (g_m > kMaxColBlocks * kTileM) {
            throw std::runtime_error("skinny arm requires M=1..16");
        }
        check(cudaSetDevice(device), "select device");
        cudaDeviceProp p{};
        check(cudaGetDeviceProperties(&p, device), "query device");
        if (p.major != 8 || p.minor != 6)
            throw std::runtime_error("prepack probe requires SM86");

        cudaStream_t stream = nullptr;
        check(cudaStreamCreate(&stream), "create stream");
        std::vector<Result> results;
        if (g_gemma_page) {
            for (const Shape& s : kGemmaPageShapes)
                results.push_back(run_shape(s, stream));
        } else {
            for (const Shape& s : kShapes) results.push_back(run_shape(s, stream));
        }
        check(cudaStreamDestroy(stream), "destroy stream");

        std::ostringstream json;
        json << std::fixed << "{\n"
             << "  \"device_name\": \"" << p.name << "\",\n"
             << "  \"device_capability\": \"" << p.major << "." << p.minor
             << "\",\n"
             << "  \"milestone\": \""
             << (g_page_wmma
                     ? "Gemma MXFP4 M128 conventional WMMA control"
                     : g_gemma_page
                         ? "Gemma MXFP4 M128 page-kernel falsifier"
                         : "F4-1 step 2 fragment prepack")
             << "\",\n"
             << "  \"operating_point\": \""
             << (g_gemma_page
                     ? "production: single RTX 3090, 250 W, 1605 MHz locked"
                     : "experimentation: single RTX 3090, 350 W, unlocked clocks")
             << "\",\n"
             << "  \"page_shared_broadcast\": "
             << (g_page_shared ? "true" : "false") << ",\n"
             << "  \"page_wmma\": " << (g_page_wmma ? "true" : "false")
             << ",\n"
             << "  \"page_warps_per_weight_tile\": " << g_page_warps
             << ",\n"
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
                 << ", \"real_activations\": "
                 << (r.real_activations ? "true" : "false")
                 << ", \"real_weights\": "
                 << (r.real_weights ? "true" : "false")
                 << ", \"e8m0_admitted\": "
                 << (r.admission.admitted() ? "true" : "false")
                 << ", \"e8m0_inadmissible\": " << r.admission.inadmissible
                 << ", \"e8m0_code_zero\": " << r.admission.subnormal_code_zero
                 << ", \"e8m0_code_255\": " << r.admission.nan_code_255
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
        for (const Result& r : results) {
            if (!r.admission.admitted()) {
                std::cerr << "admission failure: " << r.name << " carries "
                          << r.admission.inadmissible
                          << " inadmissible E8M0 scale codes (first: code "
                          << static_cast<unsigned>(r.admission.first_code)
                          << " at byte offset " << r.admission.first_offset
                          << "); exact mode reports failure rather than "
                             "substituting\n";
                return EXIT_FAILURE;
            }
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
