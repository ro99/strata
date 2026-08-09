// Host FP4 routed expert, bit-identical to deepseek_fp4_gate_up_kernel and
// deepseek_fp4_down_kernel. See include/strata/deepseek_host_expert.hpp for the
// contract this file is obliged to reproduce.

#include "strata/deepseek_host_expert.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>

#include "strata/numerics.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#define STRATA_DSV4_EXPERT_AVX2 1
#include <immintrin.h>
#else
#define STRATA_DSV4_EXPERT_AVX2 0
#endif

namespace strata {
namespace {

// The device kernel launches 256 threads per output row: 8 warps of 32 lanes.
constexpr std::size_t kThreads = 256U;
constexpr std::size_t kWarps = 8U;
constexpr std::size_t kLanes = 32U;
constexpr std::size_t kGroup = 32U;

// fp4_e2m1_value, as a table rather than a switch. Every entry is exactly
// representable, so the table is not an approximation of the device switch.
constexpr std::array<float, 16> kFp4Value = {
    0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
    -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F,
};

// The magnitudes indexed by the low three bits, for the AVX2 permute path. The
// sign comes from bit 3, applied as an XOR of the sign bit, which reproduces
// the negative-zero entry at index 8 exactly.
constexpr std::array<float, 8> kFp4Magnitude = {
    0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
};

[[nodiscard]] inline float e8m0_scale(std::uint8_t bits) noexcept {
    // fp8_e8m0_scale_bits: a float exponent field, except that 0xff is NaN and
    // a zero exponent denotes 2^-127 rather than float zero.
    if (bits == 0xFFU) return std::bit_cast<float>(0x7FC0'0000U);
    if (bits == 0U) return std::bit_cast<float>(0x0040'0000U);
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 23U);
}

// quantize_e4m3_value, replayed. The routed activation is quantized to FP8
// E4M3 in place between the SwiGLU and the down projection -- the checkpoint's
// declared dynamic E4M3 activation scheme -- so a host expert that skips it is
// wrong by the mantissa width of E4M3, not by a rounding step.
[[nodiscard]] float quantize_e4m3_value(float value) noexcept {
    const float magnitude = fminf(fabsf(value), 448.0F);
    float quantized = 0.0F;
    if (magnitude < 0.015625F) {
        quantized = rintf(ldexpf(magnitude, 9)) * ldexpf(1.0F, -9);
    } else {
        int exponent = 0;
        static_cast<void>(frexpf(magnitude, &exponent));
        exponent = std::max(-6, std::min(8, exponent - 1));
        const float step = ldexpf(1.0F, exponent - 3);
        quantized = fminf(rintf(magnitude / step) * step, 448.0F);
    }
    return copysignf(quantized, value);
}

// quantize_activation_e4m3_kernel, replayed: one dynamic power-of-two scale per
// 128-column block, chosen from that block's maximum magnitude. The device
// reduction is a tree, but max is associative and exact, so a linear scan gives
// the same value.
void quantize_activation_e4m3(std::span<float> values,
                              std::uint64_t columns) noexcept {
    constexpr std::uint64_t block = 128U;
    for (std::uint64_t begin = 0U; begin < columns; begin += block) {
        float maximum = 0.0F;
        const auto end = std::min(begin + block, columns);
        for (std::uint64_t index = begin; index < end; ++index) {
            maximum = fmaxf(maximum, fabsf(values[index]));
        }
        float scale = 1.0F;
        if (maximum > 0.0F) scale = exp2f(ceilf(log2f(maximum / 448.0F)));
        for (std::uint64_t index = begin; index < end; ++index) {
            values[index] = quantize_e4m3_value(values[index] / scale) * scale;
        }
    }
}

[[nodiscard]] inline float bf16_round(float value) noexcept {
    // __float2bfloat16_rn: round to nearest even on the truncated mantissa.
    const auto bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7FFF'FFFFU) > 0x7F80'0000U) {
        // NaN: CUDA's conversion quiets it and keeps the high mantissa bit.
        return std::bit_cast<float>((bits | 0x0040'0000U) & 0xFFFF'0000U);
    }
    const std::uint32_t lsb = (bits >> 16U) & 1U;
    const std::uint32_t rounded = bits + 0x7FFFU + lsb;
    return std::bit_cast<float>(rounded & 0xFFFF'0000U);
}

// reduce_block, replayed exactly: five shuffle-down rounds inside each warp,
// then the eight warp sums placed in lanes 0-7 of warp 0 with the remaining
// lanes zero, then five more rounds. The two rounds that add the zeroed lanes
// are performed rather than skipped, because adding +0.0 is not a no-op for a
// negative-zero partial.
[[nodiscard]] float reduce_lane0(const float* lanes) noexcept {
    // Only lane 0's value is ever read, and lane 0 only ever reads lanes below
    // the current offset, so the upper half of each round can be skipped. This
    // is the same arithmetic, not a reassociation.
    std::array<float, 16U> a{};
    for (std::size_t lane = 0U; lane < 16U; ++lane) {
        a[lane] = lanes[lane] + lanes[lane + 16U];
    }
    for (std::size_t lane = 0U; lane < 8U; ++lane) a[lane] += a[lane + 8U];
    for (std::size_t lane = 0U; lane < 4U; ++lane) a[lane] += a[lane + 4U];
    a[0] += a[2];
    a[1] += a[3];
    return a[0] + a[1];
}

[[nodiscard]] float reduce_block(const float* partials) noexcept {
    std::array<float, kLanes> warp_sums{};
    for (std::size_t warp = 0U; warp < kWarps; ++warp) {
        warp_sums[warp] = reduce_lane0(partials + warp * kLanes);
    }
    // Lanes 8..31 hold 0.0F, exactly as `threadIdx.x < 8 ? warps[lane] : 0.0F`,
    // so the offset-16 and offset-8 rounds add +0.0 and the tree over the eight
    // warp sums is ((w0+w4)+(w2+w6)) + ((w1+w5)+(w3+w7)).
    for (std::size_t lane = 0U; lane < 8U; ++lane) warp_sums[lane] += 0.0F;
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
        warp_sums[lane] += warp_sums[lane + 4U];
    }
    warp_sums[0] += warp_sums[2];
    warp_sums[1] += warp_sums[3];
    return warp_sums[0] + warp_sums[1];
}

// Partials for one output row. Thread t owns columns t, t+256, t+512, ... in
// increasing order, which is exactly the device's `group = warp; group += 8`
// stride once `column = group * 32 + lane` is substituted.
void row_partials_scalar(float* partials, const float* input,
                         const std::uint8_t* packed, const std::uint8_t* scales,
                         std::uint64_t columns, std::uint64_t packed_base,
                         std::uint64_t scale_base) noexcept {
    std::memset(partials, 0, kThreads * sizeof(float));
    for (std::size_t thread = 0U; thread < kThreads; ++thread) {
        float sum = 0.0F;
        for (std::uint64_t column = thread; column < columns;
             column += kThreads) {
            const auto byte = packed[packed_base + column / 2U];
            const unsigned int encoded =
                column % 2U == 0U ? (byte & 0x0FU) : (byte >> 4U);
            const float scale = e8m0_scale(scales[scale_base + column / kGroup]);
            sum = std::fma(input[column] * kFp4Value[encoded], scale, sum);
        }
        partials[thread] = sum;
    }
}

#if STRATA_DSV4_EXPERT_AVX2
__attribute__((target("avx2,fma")))
void row_partials_avx2(float* partials, const float* input,
                       const std::uint8_t* packed, const std::uint8_t* scales,
                       std::uint64_t columns, std::uint64_t packed_base,
                       std::uint64_t scale_base) noexcept {
    // E2M1 doubled, so every entry is an exact int8 and one _mm_shuffle_epi8
    // decodes sixteen nibbles including their sign. The factor of two is folded
    // into the group scale below, which is exact: the scale is a power of two,
    // so halving it and doubling the weight leaves the FMA's exact product
    // unchanged.
    const __m128i value_lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2,
                                            -3, -4, -6, -8, -12);
    const __m128i low_nibble = _mm_set1_epi8(0x0F);
    // Decode this row's scales once. The inner step needs a different group
    // every iteration, so leaving the branchy E8M0 decode in the loop costs one
    // unpredictable branch pair per step.
    // Deliberately not value-initialized: only the first `groups` entries are
    // read, and zeroing 2 KiB per output row costs more than the decode does.
    alignas(64) std::array<float, 512U> decoded_scales;
    const auto groups = static_cast<std::size_t>(columns / kGroup);
    for (std::size_t group = 0U; group < groups; ++group) {
        decoded_scales[group] = e8m0_scale(scales[scale_base + group]) * 0.5F;
    }
    // Every accumulator group below is stored unconditionally, so the partials
    // only need clearing when there are fewer columns than threads and some
    // groups are never reached.
    if (columns < kThreads) {
        std::memset(partials, 0, kThreads * sizeof(float));
    }
    // Sixty-four columns per step, as two independent 32-column halves. Eight
    // accumulators is what it takes to cover the ~25-cycle load-to-accumulate
    // chain: with four the step is latency-bound and retires under one uop per
    // cycle. Each accumulator still visits its own columns in ascending order,
    // which is what bit-identity requires. `base` and `offset` are multiples of
    // 64 and 256, so each half lies inside one scale group.
    for (std::size_t base = 0U; base < kThreads; base += 64U) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();
        for (std::uint64_t offset = 0U; offset < columns; offset += kThreads) {
            const std::uint64_t column = offset + base;
            if (column >= columns) break;
            // Byte b carries column 2b in its low nibble and 2b+1 in its high
            // nibble, so interleaving the two decoded halves restores column
            // order.
            const __m128i raw_a = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(packed + packed_base +
                                                 column / 2U));
            const __m128i raw_b = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(packed + packed_base +
                                                 (column + 32U) / 2U));
            const __m128i even_a = _mm_shuffle_epi8(
                value_lut, _mm_and_si128(raw_a, low_nibble));
            const __m128i odd_a = _mm_shuffle_epi8(
                value_lut,
                _mm_and_si128(_mm_srli_epi16(raw_a, 4), low_nibble));
            const __m128i even_b = _mm_shuffle_epi8(
                value_lut, _mm_and_si128(raw_b, low_nibble));
            const __m128i odd_b = _mm_shuffle_epi8(
                value_lut,
                _mm_and_si128(_mm_srli_epi16(raw_b, 4), low_nibble));
            const __m128i first_a = _mm_unpacklo_epi8(even_a, odd_a);
            const __m128i second_a = _mm_unpackhi_epi8(even_a, odd_a);
            const __m128i first_b = _mm_unpacklo_epi8(even_b, odd_b);
            const __m128i second_b = _mm_unpackhi_epi8(even_b, odd_b);
            const __m256 scale_a = _mm256_set1_ps(
                decoded_scales[static_cast<std::size_t>(column / kGroup)]);
            const __m256 scale_b = _mm256_set1_ps(
                decoded_scales[static_cast<std::size_t>(column / kGroup) + 1U]);
            // Written out rather than factored into a lambda: a lambda does not
            // inherit the enclosing function's target attribute, so its AVX2
            // intrinsics fail to inline.
            acc0 = _mm256_fmadd_ps(
                _mm256_mul_ps(_mm256_loadu_ps(input + column),
                              _mm256_cvtepi32_ps(
                                  _mm256_cvtepi8_epi32(first_a))),
                scale_a, acc0);
            acc1 = _mm256_fmadd_ps(
                _mm256_mul_ps(_mm256_loadu_ps(input + column + 8U),
                              _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                                  _mm_srli_si128(first_a, 8)))),
                scale_a, acc1);
            acc2 = _mm256_fmadd_ps(
                _mm256_mul_ps(_mm256_loadu_ps(input + column + 16U),
                              _mm256_cvtepi32_ps(
                                  _mm256_cvtepi8_epi32(second_a))),
                scale_a, acc2);
            acc3 = _mm256_fmadd_ps(
                _mm256_mul_ps(_mm256_loadu_ps(input + column + 24U),
                              _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                                  _mm_srli_si128(second_a, 8)))),
                scale_a, acc3);
            acc4 = _mm256_fmadd_ps(
                _mm256_mul_ps(_mm256_loadu_ps(input + column + 32U),
                              _mm256_cvtepi32_ps(
                                  _mm256_cvtepi8_epi32(first_b))),
                scale_b, acc4);
            acc5 = _mm256_fmadd_ps(
                _mm256_mul_ps(_mm256_loadu_ps(input + column + 40U),
                              _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                                  _mm_srli_si128(first_b, 8)))),
                scale_b, acc5);
            acc6 = _mm256_fmadd_ps(
                _mm256_mul_ps(_mm256_loadu_ps(input + column + 48U),
                              _mm256_cvtepi32_ps(
                                  _mm256_cvtepi8_epi32(second_b))),
                scale_b, acc6);
            acc7 = _mm256_fmadd_ps(
                _mm256_mul_ps(_mm256_loadu_ps(input + column + 56U),
                              _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                                  _mm_srli_si128(second_b, 8)))),
                scale_b, acc7);
        }
        _mm256_storeu_ps(partials + base, acc0);
        _mm256_storeu_ps(partials + base + 8U, acc1);
        _mm256_storeu_ps(partials + base + 16U, acc2);
        _mm256_storeu_ps(partials + base + 24U, acc3);
        _mm256_storeu_ps(partials + base + 32U, acc4);
        _mm256_storeu_ps(partials + base + 40U, acc5);
        _mm256_storeu_ps(partials + base + 48U, acc6);
        _mm256_storeu_ps(partials + base + 56U, acc7);
    }
}
#endif

}  // namespace

const float* dsv4_host_bf16_silu_table() noexcept {
    static const std::vector<float> table = [] {
        constexpr std::size_t entries = 1U << 16U;
        std::vector<float> values(entries);
        for (std::size_t index = 0U; index < entries; ++index) {
            const auto bits = static_cast<std::uint32_t>(index) << 16U;
            const float value = std::bit_cast<float>(bits);
            values[index] = std::isfinite(value) ? silu_f32(value) : value;
        }
        return values;
    }();
    return table.data();
}

bool dsv4_host_expert_avx2_supported() noexcept {
#if STRATA_DSV4_EXPERT_AVX2
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

std::uint64_t dsv4_tiled_expert_shard_bytes(
    std::uint64_t hidden, std::uint64_t intermediate,
    std::uint64_t shards) noexcept {
    if (hidden == 0U || intermediate == 0U || shards == 0U ||
        hidden % 32U != 0U || intermediate % (32U * shards) != 0U) {
        return 0U;
    }
    const auto shard_intermediate = intermediate / shards;
    return 2U * shard_intermediate * (hidden / 2U) +
           2U * shard_intermediate * (hidden / 16U) +
           hidden * (shard_intermediate / 2U) +
           hidden * (shard_intermediate / 16U);
}

ParseResult<Dsv4TiledExpertWeights> dsv4_tiled_expert_weights(
    std::span<const std::byte> storage, std::uint64_t hidden,
    std::uint64_t intermediate, std::uint64_t shards) {
    ParseResult<Dsv4TiledExpertWeights> result;
    const auto bytes = dsv4_tiled_expert_shard_bytes(
        hidden, intermediate, shards);
    if (bytes == 0U || storage.size() != bytes) {
        result.errors.emplace_back(
            "DeepSeek tiled expert shard has an incompatible extent");
        return result;
    }
    const auto shard_intermediate = intermediate / shards;
    const auto w13_packed = 2U * shard_intermediate * (hidden / 2U);
    const auto w13_scales = 2U * shard_intermediate * (hidden / 16U);
    const auto w2_packed = hidden * (shard_intermediate / 2U);
    const auto w2_scales = hidden * (shard_intermediate / 16U);
    result.value.w13_packed = storage.first(w13_packed);
    storage = storage.subspan(w13_packed);
    result.value.w13_scales = storage.first(w13_scales);
    storage = storage.subspan(w13_scales);
    result.value.w2_packed = storage.first(w2_packed);
    result.value.w2_scales = storage.subspan(w2_packed, w2_scales);
    return result;
}

ValidationResult dsv4_transform_tiled_expert_shard(
    std::span<std::byte> destination, const Dsv4HostExpertWeights& canonical,
    std::uint64_t hidden, std::uint64_t intermediate, std::uint64_t shard,
    std::uint64_t shards) {
    ValidationResult result;
    const auto shard_bytes = dsv4_tiled_expert_shard_bytes(
        hidden, intermediate, shards);
    if (shard >= shards || destination.size() != shard_bytes ||
        canonical.w1_packed.size() != intermediate * (hidden / 2U) ||
        canonical.w3_packed.size() != intermediate * (hidden / 2U) ||
        canonical.w1_scales.size() != intermediate * (hidden / 32U) ||
        canonical.w3_scales.size() != intermediate * (hidden / 32U) ||
        canonical.w2_packed.size() != hidden * (intermediate / 2U) ||
        canonical.w2_scales.size() != hidden * (intermediate / 32U)) {
        result.errors.emplace_back(
            "DeepSeek tiled expert transform has incompatible extents");
        return result;
    }
    constexpr std::uint64_t block_rows = 32U;
    const auto shard_intermediate = intermediate / shards;
    const auto w13_packed_bytes =
        2U * shard_intermediate * (hidden / 2U);
    const auto w13_scale_bytes =
        2U * shard_intermediate * (hidden / 16U);
    const auto w2_packed_bytes = hidden * (shard_intermediate / 2U);
    auto* packed13 = reinterpret_cast<std::uint8_t*>(destination.data());
    auto* scales13 = packed13 + w13_packed_bytes;
    auto* packed2 = scales13 + w13_scale_bytes;
    auto* scales2 = packed2 + w2_packed_bytes;
    const std::uint8_t* source[] = {
        reinterpret_cast<const std::uint8_t*>(canonical.w1_packed.data()),
        reinterpret_cast<const std::uint8_t*>(canonical.w3_packed.data())};
    const std::uint8_t* source_scales[] = {
        reinterpret_cast<const std::uint8_t*>(canonical.w1_scales.data()),
        reinterpret_cast<const std::uint8_t*>(canonical.w3_scales.data())};
    const auto* source_w2 =
        reinterpret_cast<const std::uint8_t*>(canonical.w2_packed.data());
    const auto* source_s2 =
        reinterpret_cast<const std::uint8_t*>(canonical.w2_scales.data());
    for (std::uint64_t projection = 0U; projection < 2U; ++projection) {
        for (std::uint64_t row = 0U; row < shard_intermediate; ++row) {
            const auto source_row = shard * shard_intermediate + row;
            const auto output = projection * shard_intermediate + row;
            const auto block = output / block_rows;
            const auto within = output % block_rows;
            for (std::uint64_t pair = 0U; pair < hidden / 2U; ++pair) {
                packed13[(block * (hidden / 2U) + pair) * block_rows + within] =
                    source[projection][source_row * (hidden / 2U) + pair];
            }
            for (std::uint64_t group = 0U; group < hidden / 16U; ++group) {
                scales13[(block * (hidden / 16U) + group) * block_rows + within] =
                    source_scales[projection][
                        source_row * (hidden / 32U) + group / 2U];
            }
        }
    }
    for (std::uint64_t row = 0U; row < hidden; ++row) {
        const auto block = row / block_rows;
        const auto within = row % block_rows;
        for (std::uint64_t pair = 0U; pair < shard_intermediate / 2U; ++pair) {
            packed2[(block * (shard_intermediate / 2U) + pair) * block_rows +
                    within] =
                source_w2[row * (intermediate / 2U) +
                          shard * (shard_intermediate / 2U) + pair];
        }
        for (std::uint64_t group = 0U; group < shard_intermediate / 16U;
             ++group) {
            scales2[(block * (shard_intermediate / 16U) + group) * block_rows +
                    within] =
                source_s2[row * (intermediate / 32U) +
                          shard * (shard_intermediate / 32U) + group / 2U];
        }
    }
    return result;
}

namespace {

[[maybe_unused]] void tiled_matvec16_scalar(
    float* output, const float* input, const std::uint8_t* packed,
    const std::uint8_t* scales, std::uint64_t inputs) noexcept {
    std::fill_n(output, 16U, 0.0F);
    for (std::uint64_t column = 0U; column < inputs; ++column) {
        const auto* weights = packed + (column / 2U) * 32U;
        const auto* scale = scales + (column / 16U) * 32U;
        for (std::uint64_t row = 0U; row < 16U; ++row) {
            const auto encoded = column % 2U == 0U
                ? weights[row] & 0x0FU : weights[row] >> 4U;
            output[row] = std::fma(
                input[column],
                kFp4Value[encoded] * e8m0_scale(scale[row]), output[row]);
        }
    }
}

#if STRATA_DSV4_EXPERT_AVX2
__attribute__((target("avx2,fma")))
__m256 tiled_decode_fp4(__m256i bytes, __m256 magnitude) noexcept {
    const auto index = _mm256_and_si256(bytes, _mm256_set1_epi32(7));
    const auto sign = _mm256_slli_epi32(
        _mm256_and_si256(bytes, _mm256_set1_epi32(8)), 28);
    return _mm256_xor_ps(_mm256_permutevar8x32_ps(magnitude, index),
                         _mm256_castsi256_ps(sign));
}

__attribute__((target("avx2,fma")))
__m256 tiled_decode_e8m0(__m256i bytes) noexcept {
    auto bits = _mm256_slli_epi32(bytes, 23);
    bits = _mm256_blendv_epi8(
        bits, _mm256_set1_epi32(0x0040'0000),
        _mm256_cmpeq_epi32(bytes, _mm256_setzero_si256()));
    bits = _mm256_blendv_epi8(
        bits, _mm256_set1_epi32(0x7FC0'0000),
        _mm256_cmpeq_epi32(bytes, _mm256_set1_epi32(0xFF)));
    return _mm256_castsi256_ps(bits);
}

__attribute__((target("avx2,fma")))
void tiled_matvec16_avx2(float* output, const float* input,
                         const std::uint8_t* packed,
                         const std::uint8_t* scales,
                         std::uint64_t inputs) noexcept {
    const __m256 magnitude = _mm256_loadu_ps(kFp4Magnitude.data());
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    for (std::uint64_t group = 0U; group < inputs / 16U; ++group) {
        const auto raw_scale = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(scales + group * 32U));
        const __m256 scale0 =
            tiled_decode_e8m0(_mm256_cvtepu8_epi32(raw_scale));
        const __m256 scale1 = tiled_decode_e8m0(
            _mm256_cvtepu8_epi32(_mm_srli_si128(raw_scale, 8)));
        for (std::uint64_t pair = 0U; pair < 8U; ++pair) {
            const auto raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                packed + (group * 8U + pair) * 32U));
            const auto bytes0 = _mm256_cvtepu8_epi32(raw);
            const auto bytes1 =
                _mm256_cvtepu8_epi32(_mm_srli_si128(raw, 8));
            const __m256 even =
                _mm256_set1_ps(input[group * 16U + pair * 2U]);
            const __m256 odd =
                _mm256_set1_ps(input[group * 16U + pair * 2U + 1U]);
            sum0 = _mm256_fmadd_ps(
                _mm256_mul_ps(tiled_decode_fp4(bytes0, magnitude), scale0),
                even, sum0);
            sum1 = _mm256_fmadd_ps(
                _mm256_mul_ps(tiled_decode_fp4(bytes1, magnitude), scale1),
                even, sum1);
            sum0 = _mm256_fmadd_ps(
                _mm256_mul_ps(tiled_decode_fp4(
                    _mm256_srli_epi32(bytes0, 4), magnitude), scale0),
                odd, sum0);
            sum1 = _mm256_fmadd_ps(
                _mm256_mul_ps(tiled_decode_fp4(
                    _mm256_srli_epi32(bytes1, 4), magnitude), scale1),
                odd, sum1);
        }
    }
    _mm256_storeu_ps(output, sum0);
    _mm256_storeu_ps(output + 8U, sum1);
}
#endif

}  // namespace

void dsv4_tiled_expert_matvec16(
    std::span<float, 16U> output, std::span<const float> input,
    std::span<const std::byte> packed, std::span<const std::byte> scales,
    std::uint64_t outputs, std::uint64_t output_begin) noexcept {
    constexpr std::uint64_t block_rows = 32U;
    const auto block = output_begin / block_rows;
    const auto within = output_begin % block_rows;
    const auto* packed_data = reinterpret_cast<const std::uint8_t*>(
        packed.data() + block * (input.size() / 2U) * block_rows + within);
    const auto* scale_data = reinterpret_cast<const std::uint8_t*>(
        scales.data() + block * (input.size() / 16U) * block_rows + within);
    static_cast<void>(outputs);
#if STRATA_DSV4_EXPERT_AVX2
    static const bool vector = dsv4_host_expert_avx2_supported();
    if (vector) {
        tiled_matvec16_avx2(output.data(), input.data(), packed_data, scale_data,
                            input.size());
        return;
    }
#endif
    tiled_matvec16_scalar(output.data(), input.data(), packed_data, scale_data,
                          input.size());
}

namespace {

[[nodiscard]] ValidationResult validate_shape(
    const Dsv4HostExpertWeights& weights, std::uint64_t hidden,
    std::uint64_t intermediate) {
    ValidationResult result;
    if (hidden == 0U || intermediate == 0U || hidden % kGroup != 0U ||
        intermediate % kGroup != 0U) {
        result.errors.emplace_back(
            "DeepSeek host expert dimensions must be non-zero multiples of 32");
        return result;
    }
    if (weights.w1_packed.size() != intermediate * (hidden / 2U) ||
        weights.w3_packed.size() != intermediate * (hidden / 2U) ||
        weights.w1_scales.size() != intermediate * (hidden / kGroup) ||
        weights.w3_scales.size() != intermediate * (hidden / kGroup) ||
        weights.w2_packed.size() != hidden * (intermediate / 2U) ||
        weights.w2_scales.size() != hidden * (intermediate / kGroup)) {
        result.errors.emplace_back(
            "DeepSeek host expert weight extents do not match the declared shape");
    }
    return result;
}

void row_partials(float* destination, const float* source,
                  const std::uint8_t* packed, const std::uint8_t* scales,
                  std::uint64_t columns, std::uint64_t packed_base,
                  std::uint64_t scale_base, bool vector) noexcept {
#if STRATA_DSV4_EXPERT_AVX2
    // The vector path consumes 64 columns per step and reads the scale group
    // above the one it starts in, so it needs whole 64-column rows.
    if (vector && columns % 64U == 0U) {
        row_partials_avx2(destination, source, packed, scales, columns,
                          packed_base, scale_base);
        return;
    }
#else
    static_cast<void>(vector);
#endif
    row_partials_scalar(destination, source, packed, scales, columns,
                        packed_base, scale_base);
}

[[nodiscard]] bool vector_enabled(bool use_avx2) noexcept {
#if STRATA_DSV4_EXPERT_AVX2
    return use_avx2 && dsv4_host_expert_avx2_supported();
#else
    static_cast<void>(use_avx2);
    return false;
#endif
}

}  // namespace

ValidationResult dsv4_host_expert_gate_up(
    std::span<float> scratch, std::span<const float> input,
    const Dsv4HostExpertWeights& weights, std::uint64_t hidden,
    std::uint64_t intermediate, std::uint64_t row_begin, std::uint64_t row_end,
    float coefficient, float swiglu_limit, bool use_avx2) {
    auto result = validate_shape(weights, hidden, intermediate);
    if (!result.ok()) return result;
    if (input.size() != hidden || scratch.size() != intermediate ||
        row_end > intermediate || row_begin > row_end) {
        result.errors.emplace_back(
            "DeepSeek host expert gate/up row range is out of bounds");
        return result;
    }
    if (!std::isfinite(swiglu_limit) || swiglu_limit <= 0.0F) {
        result.errors.emplace_back(
            "DeepSeek host expert SwiGLU limit must be finite and positive");
        return result;
    }
    const bool vector = vector_enabled(use_avx2);
    const auto* silu = dsv4_host_bf16_silu_table();
    const auto* w1 = reinterpret_cast<const std::uint8_t*>(weights.w1_packed.data());
    const auto* w1s = reinterpret_cast<const std::uint8_t*>(weights.w1_scales.data());
    const auto* w3 = reinterpret_cast<const std::uint8_t*>(weights.w3_packed.data());
    const auto* w3s = reinterpret_cast<const std::uint8_t*>(weights.w3_scales.data());
    alignas(64) std::array<float, kThreads> gate_partials;
    alignas(64) std::array<float, kThreads> up_partials;
    for (std::uint64_t row = row_begin; row < row_end; ++row) {
        const auto packed_base = row * (hidden / 2U);
        const auto scale_base = row * (hidden / kGroup);
        row_partials(gate_partials.data(), input.data(), w1, w1s, hidden,
                     packed_base, scale_base, vector);
        row_partials(up_partials.data(), input.data(), w3, w3s, hidden,
                     packed_base, scale_base, vector);
        const float rounded_gate = bf16_round(reduce_block(gate_partials.data()));
        const float rounded_up = bf16_round(reduce_block(up_partials.data()));
        if (!std::isfinite(rounded_gate) || !std::isfinite(rounded_up)) {
            result.errors.emplace_back(
                "DeepSeek host expert gate/up produced a non-finite value");
            return result;
        }
        const float limited_gate = std::fmin(rounded_gate, swiglu_limit);
        const float limited_up =
            std::fmax(-swiglu_limit, std::fmin(rounded_up, swiglu_limit));
        const auto gate_bits = static_cast<std::uint16_t>(
            std::bit_cast<std::uint32_t>(limited_gate) >> 16U);
        float activated = silu[gate_bits] * limited_up;
        activated *= coefficient;
        scratch[row] = bf16_round(activated);
    }
    return result;
}

ValidationResult dsv4_host_expert_quantize(std::span<float> scratch,
                                           std::uint64_t intermediate) {
    ValidationResult result;
    if (scratch.size() != intermediate || intermediate == 0U) {
        result.errors.emplace_back(
            "DeepSeek host expert activation span does not match the shape");
        return result;
    }
    quantize_activation_e4m3(scratch, intermediate);
    return result;
}

ValidationResult dsv4_host_expert_down(
    std::span<float> output, std::span<const float> scratch,
    const Dsv4HostExpertWeights& weights, std::uint64_t hidden,
    std::uint64_t intermediate, std::uint64_t row_begin, std::uint64_t row_end,
    bool use_avx2) {
    auto result = validate_shape(weights, hidden, intermediate);
    if (!result.ok()) return result;
    if (output.size() != hidden || scratch.size() != intermediate ||
        row_end > hidden || row_begin > row_end) {
        result.errors.emplace_back(
            "DeepSeek host expert down row range is out of bounds");
        return result;
    }
    const bool vector = vector_enabled(use_avx2);
    const auto* w2 = reinterpret_cast<const std::uint8_t*>(weights.w2_packed.data());
    const auto* w2s = reinterpret_cast<const std::uint8_t*>(weights.w2_scales.data());
    alignas(64) std::array<float, kThreads> down_partials;
    for (std::uint64_t row = row_begin; row < row_end; ++row) {
        row_partials(down_partials.data(), scratch.data(), w2, w2s, intermediate,
                     row * (intermediate / 2U), row * (intermediate / kGroup),
                     vector);
        output[row] = bf16_round(reduce_block(down_partials.data()));
    }
    return result;
}

ValidationResult dsv4_host_expert_fp4(
    std::span<float> output, std::span<const float> input,
    const Dsv4HostExpertWeights& weights, std::span<float> scratch,
    std::uint64_t hidden, std::uint64_t intermediate, float coefficient,
    float swiglu_limit, bool use_avx2) {
    // enqueue_deepseek_moe quantizes the MoE input to E4M3 before the gate/up
    // kernel, exactly as it quantizes the activation before the down kernel.
    // A caller handing over the raw hidden state must get the same treatment or
    // this stops being an oracle.
    std::vector<float> quantized(input.begin(), input.end());
    auto result = dsv4_host_expert_quantize(quantized, hidden);
    if (!result.ok()) return result;
    result = dsv4_host_expert_gate_up(
        scratch, quantized, weights, hidden, intermediate, 0U, intermediate,
        coefficient, swiglu_limit, use_avx2);
    if (!result.ok()) return result;
    result = dsv4_host_expert_quantize(scratch, intermediate);
    if (!result.ok()) return result;
    return dsv4_host_expert_down(output, scratch, weights, hidden, intermediate,
                                 0U, hidden, use_avx2);
}

}  // namespace strata
