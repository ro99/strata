#pragma once

// Host execution of a DeepSeek-V4 FP4 routed expert, bit-identical to the CUDA
// path.
//
// STANDING: this kernel has no caller in the runtime, deliberately. Both
// scheduling policies built on it are measured rejections, and it is kept
// because it is bit-exact, tested, and a prerequisite for the one use that has
// not been priced yet.
//
//   * Diverting cache-miss experts here (`--host-expert-misses`, experiment
//     0051) measured 0.76x and is not on main. Decode routes about 0.9 diverted
//     experts per layer, so one expert splits across the pool twice per layer
//     and the path caps near 11 GB/s in situ against 26 standalone. The cache
//     also starves: with no decode-time admission the resident set goes stale
//     and the miss rate rises from ~15% to ~47%.
//   * Placing all six routed experts per layer here statically (experiment
//     0054) is rejected by arithmetic, without code. Break-even is 30.5 GB/s
//     in situ; the 86 barriers/step at 73.6 us make that 32.3 GB/s standalone,
//     against 26.23 measured. Best case 0.939x.
//
// What would change the answer is work per barrier. Experiment 0051 named
// ">= 4 diverted experts per layer" as the precondition, and DSpark/MTP at
// block size 5 supplies it by verifying several tokens per forward pass --
// which also amortizes the weight read, the only lever that reaches the
// throughput target on this hardware. Re-derive the gate at that operating
// point before wiring this to anything; do not model a new path on the
// rejected one.
//
// The arithmetic that makes it worth keeping regardless: the bytes already sit
// in the resident host arena, so computing the expert where the weights are
// costs a DRAM read (76.3 GB/s measured node-local, experiment 0050) instead
// of a PCIe transfer at 6.9 GB/s. The kernel is 26.23 GB/s standalone at 28
// threads, 28.62 under `numactl --interleave=all`, and uop-bound rather than
// memory-bound -- which is why populating the empty DIMM slots does not
// multiply it.
//
// Bit-exactness contract. This is NOT a numerics trade like --fast-mhc. It
// reproduces `deepseek_fp4_gate_up_kernel` and `deepseek_fp4_down_kernel`
// exactly:
//
//   * the same decomposition into 256 partial accumulators, where CUDA thread
//     t = warp*32 + lane owns columns t, t+256, t+512, ... in that order;
//   * the same f32 accumulation with the same FMA contraction, `partial =
//     fma(input * fp4_value, scale, partial)`, which is what nvcc emits for
//     `sum += input * fp4_e2m1_value(e) * scale` under the default -fmad=true;
//   * the same `reduce_block` tree -- five shuffle-down rounds within each
//     warp, then a second five-round tree over the eight warp sums;
//   * the same BF16 rounding points, the same SwiGLU clamp order, and the same
//     65,536-entry BF16-indexed SiLU table.
//
// A mismatch against the device is a defect, not a tolerance.

#include <cstdint>
#include <span>

#include "strata/result.hpp"

namespace strata {

// One 65,536-entry SiLU table indexed by the BF16 bit pattern of the clamped
// gate, matching `state.moe_bf16_silu` on the device. Built once, shared by
// every host expert call.
[[nodiscard]] const float* dsv4_host_bf16_silu_table() noexcept;

// True when this build and CPU can run the vectorized path. The scalar path is
// always available and is the reference both are tested against.
[[nodiscard]] bool dsv4_host_expert_avx2_supported() noexcept;

struct Dsv4HostExpertWeights {
    // `<prefix>w1.weight` / `.scale`: [intermediate, hidden] FP4 E2M1 packed
    // two per byte, with one E8M0 scale byte per 32 columns.
    std::span<const std::byte> w1_packed;
    std::span<const std::byte> w1_scales;
    std::span<const std::byte> w3_packed;
    std::span<const std::byte> w3_scales;
    // `<prefix>w2`: [hidden, intermediate], same encoding.
    std::span<const std::byte> w2_packed;
    std::span<const std::byte> w2_scales;
};

// Runs gate/up, the clamped SwiGLU, the routed coefficient and the down
// projection for one expert. `scratch` must hold `intermediate` floats and is
// overwritten. `output` receives `hidden` floats and is overwritten, not
// accumulated: the caller combines experts exactly as the device path does.
[[nodiscard]] ValidationResult dsv4_host_expert_fp4(
    std::span<float> output, std::span<const float> input,
    const Dsv4HostExpertWeights& weights, std::span<float> scratch,
    std::uint64_t hidden, std::uint64_t intermediate, float coefficient,
    float swiglu_limit, bool use_avx2);

// The same computation split into its three phases, so a caller with a worker
// pool can spread one expert's rows across cores. Decode routes about one
// cache-missing expert per layer, so per-expert parallelism alone would leave
// most of the machine idle.
//
// Rows are independent within a phase and each row's arithmetic is unchanged,
// so any row partition gives the same bits as `dsv4_host_expert_fp4`. The
// phases are ordered: every gate/up row must be finished before `quantize`,
// and `quantize` before any down row, because the E4M3 scale is chosen from a
// 128-column block of the completed activation.
//
// The caller of the phased form owns one extra step the wrapper does for it:
// the MoE *input* must be E4M3-quantized with `dsv4_host_expert_quantize`
// before `gate_up`, because `enqueue_deepseek_moe` quantizes it on the device.
// Skipping it diverges from the device only for inputs that are not already
// E4M3-representable, which is why it survived a test whose inputs were all
// multiples of 0.125.
[[nodiscard]] ValidationResult dsv4_host_expert_gate_up(
    std::span<float> scratch, std::span<const float> input,
    const Dsv4HostExpertWeights& weights, std::uint64_t hidden,
    std::uint64_t intermediate, std::uint64_t row_begin, std::uint64_t row_end,
    float coefficient, float swiglu_limit, bool use_avx2);

[[nodiscard]] ValidationResult dsv4_host_expert_quantize(
    std::span<float> scratch, std::uint64_t intermediate);

[[nodiscard]] ValidationResult dsv4_host_expert_down(
    std::span<float> output, std::span<const float> scratch,
    const Dsv4HostExpertWeights& weights, std::uint64_t hidden,
    std::uint64_t intermediate, std::uint64_t row_begin, std::uint64_t row_end,
    bool use_avx2);

}  // namespace strata
