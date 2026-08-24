# Experiment 0182 — conventional Gemma page WMMA is rejected

**Date:** 2026-08-24  
**Branch:** `fix/gemma4-page-tiled-prefill`  
**Origin:** experiments 0165, 0180, and corrected reference 0181  
**Verdict:** **REJECTED — do not integrate the shared-BF16 WMMA control**

## Hypothesis and gates

A conventional 64x128x32 page GEMM that widens each canonical compact MXFP4
weight tile into transient BF16 shared memory, reuses it over 64 activation
rows, and feeds BF16 WMMA might be fast enough to support a device-resident
M=128 Gemma executor.

The primary system gate was at least 10x over the measured 6.354801 s page,
or at most 635.5 ms, with prefill/decode per-token ratio at most 0.11. The
corrected external target is vLLM's measured 0.141776 s / 881.67 tok/s TP=1
short page. Projection correctness required maximum relative error below 1e-4;
full-model integration would additionally require the existing teacher and
generation oracles, unchanged W8A16 dispatch, and `make check`. Persistent
weights could not exceed one compact FP4 copy, transient page workspace was
capped at 256 MiB, and one-3090 admission at VRAM fraction 0.95 was binding.

Rollback was mandatory if the isolated projection arithmetic alone exhausted
the 635.5 ms full-page budget. It does, so no runtime executor was built.

## Cost model and targeted term

The production M=128 profile remains:

| Resource/term | Measured time |
|---|---:|
| CUDA/host synchronization and handoffs (`sum_serial`) | **2,419.841 ms** |
| q/k host normalization and RoPE | 1,610.519 ms |
| attention | 1,305.041 ms |
| gate/up projections | 888.632 ms |
| exact host GeGLU | 597.662 ms |
| H2D, 1.752 GB at 5.80 GB/s | 302.228 ms |
| D2H, 2.166 GB at 4.54 GB/s | 476.937 ms |
| all recorded CUDA kernels | 293.972 ms |

`argmax_r` is the 2.420 s serial CUDA/host handoff term. The intended complete
mechanism was a device-resident page executor: the tiled projection is its
prerequisite, while keeping activations, normalization, RoPE, attention, KV,
and residual work on device is what reduces `argmax_r`. Signs on the other
resources were favorable in principle: compact weight reads fall from eight
passes toward one or two, H2D/D2H activation volume nearly disappears, host
compute disappears, tensor compute increases, and transient workspace grows.
The control isolates the compute sign before building the executor.

The measured PCIe rates are 74%/58% of the real Gen3 x8 ceiling, not an
order-of-magnitude link deficit. Their large wall contribution comes from
serial materialization. This remains an overlap/ownership defect rather than
a PCIe bandwidth-limit claim.

## Cheap experiment and reference ruler

The rejected alternative was another full vLLM or Strata server profile: model
startup costs tens of seconds and the known-bad Strata page costs 6.4 s, while
the mechanism is decided by two sub-10-ms projections. Each Strata process
spent about 0.06 s in host prepack plus warmup/oracle/timing; all six shape arms
completed in 16.5 s. Fixed setup was roughly 17x the approximately 10 ms pair
of useful kernel windows per process, and the total arm budget was under two
minutes.

Before the Strata arm, vLLM 2.3.8's own Marlin operator was measured on the
same exact MLP shapes at M=128 and the same locked point. Thirty-one 20-launch
windows gave:

| Kernel | Median | Effective tensor rate | Useful compact rate |
|---|---:|---:|---:|
| Marlin gate/up `[21504,5376]` | **0.490811 ms** | 60.30 TFLOP/s | 125.13 GB/s |
| Marlin down `[5376,21504]` | **0.506624 ms** | 58.42 TFLOP/s | 121.22 GB/s |

This is the successor gate derived from a measured mechanism, replacing
0180's invalid 600 GB/s external premise.

## Three fresh-process Strata repetitions

The control uses a 64x128 CTA, K32 tiles aligned to the checkpoint scale group,
eight warps, BF16 shared A/B tiles, BF16 tensor-core MMA, and FP32 accumulation.
The checkpoint representation is unchanged. Probe-only memory includes both
the old fragment copy and the canonical control copy and peaks at 148,979,712
bytes, below the 256 MiB experiment ceiling; neither duplicate was admitted to
the runtime.

| Kernel | Run 1 | Run 2 | Run 3 | Median | vs Marlin |
|---|---:|---:|---:|---:|---:|
| gate/up | 3.642848 ms | 3.643649 ms | 3.643520 ms | **3.643520 ms** | **7.42x slower** |
| down | 6.293433 ms | 6.292672 ms | 6.293600 ms | **6.293433 ms** | **12.42x slower** |

Observed spread is 0.000801 ms gate/up and 0.000928 ms down, so the rejection
is far outside variance. Useful compact rates are 16.86 and 9.76 GB/s.

The sampled canonical double oracle passes in every run: maximum relative
error is 4.14e-7 gate/up and 5.564e-6 down. The rejection is performance-only.

Gemma has 60 layers and separately dispatches gate and up. These three MLP
projections alone project to
`60 * (2 * 3.643520 + 6.293433) = 814.83 ms`, already 1.28x the entire 635.5 ms
10x budget before attention projections, normalization, attention, KV, GeGLU,
residuals, logits, or any handoff. Integration is forbidden by the rollback
condition.

## Verdict and next bounded mechanism

**REJECTED.** Shared-memory widening plus ordinary WMMA is not a competitive
Marlin substitute. K32 forces 168 barriers for gate/up and 672 for down;
compact weights are decoded into shared BF16 by every M tile, and neither the
decode nor the global load is overlapped deeply enough.

The next bounded mechanism must reproduce the features the measured ruler
shows are material: load-time fragment repack, register dequantization,
multi-stage `cp.async` global-to-shared loading, and Marlin-style N/K work
scheduling. It must first approach the measured 0.491/0.507 ms shape ruler.
No device-resident full executor may be built on the rejected control.

Raw JSON remains ignored under `results/gemma4-prefill/wmma-page/`; the Marlin
ruler log remains outside Git at `/tmp/gemma4-marlin-shapes.log`.
