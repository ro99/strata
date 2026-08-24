# Experiment 0185 — Marlin page layout also clears the M=1 gate

**Date:** 2026-08-24  
**Branch:** `fix/gemma4-device-page-executor`  
**Origin:** experiment 0184's accepted page primitive  
**Verdict:** **ACCEPTED — one compact layout can serve prefill and decode**

## Hypothesis and gates

The exact Marlin code/scale permutation accepted at M=128 can also feed the
M=1 specialization with Strata's FP32 output boundary. This is the cheapest
falsifier for a one-representation runtime: a page-only layout that regresses
decode would require a forbidden duplicate weight or make the integration
negative.

The primary metrics were M=1 projection latency and useful compact-weight
bandwidth on the exact Gemma MLP shapes. Correctness remained maximum relative
error below 1e-4 against the canonical FP32 oracle. The transient probe ceiling
was 256 MiB. Rollback was any correctness failure or a median below the
accepted register-fed kernel's useful service range.

The production decode cost model has GPU kernel/HBM service as `argmax_r` at
99.1% of the old scalar step. This mechanism reduces that term. Compared with
the current register-fed route it removes activation E4M3 quantization and its
separate fragment workspace, retains BF16 activation at the MMA boundary, and
adds only the accepted FP32 output reorder traffic. H2D/D2H, attention, norms,
and persistent compact bytes do not change in this isolated screen.

## Cheap arm and exact configuration

One standalone process directly decides both layout compatibility and kernel
speed. The rejected alternative was a six-minute checkpoint load before the
backend could consume this layout. Three fresh processes completed in 5.5
seconds total. Each uses 32 chained launches per timing sample, so setup and
oracle dominate wall time while the measured window is still long enough to
amortize event and launch floor.

The specialization is BF16/E2M1/BF16/E8M0, M-block 1 with the 8-row form,
N-block 8, K-block 8, 256 threads, four stages and group-block 2. All runs use
CUDA device 1 under `CUDA_DEVICE_ORDER=PCI_BUS_ID`, one RTX 3090 locked at
1605 MHz and 250 W.

## Every run and median

| Shape | Run 1 | Run 2 | Run 3 | Median | Useful rate | Worst max-relative |
|---|---:|---:|---:|---:|---:|---:|
| gate/up `[21504,5376]` | 81.212 us | 81.216 us | 81.184 us | **81.212 us** | **756.22 GB/s** | **0.0** |
| down `[5376,21504]` | 78.560 us | 78.496 us | 78.496 us | **78.496 us** | **782.40 GB/s** | **5.6e-8** |

Useful bytes are unchanged at 61,415,424 per shape: 57,802,752 compact code
bytes plus 3,612,672 E8M0 scale bytes. The code and scale buffers remain pure
permutations with no widened persistent copy. Peak simultaneous probe memory
is 195,463,496 bytes, below 256 MiB.

## Verdict

**ACCEPTED.** The load-time Marlin layout is a viable single compact Gemma
MXFP4 representation for both M=128 prefill and M=1 decode. This does not
claim an end-to-end decode win: the full attention shapes and device layer
loop still need measurement. It removes the representation blocker from the
next experiment, which may replace Gemma's register-fed layout only if every
consumer is moved together and the full-model oracle passes.

Raw results remain outside Git at `/tmp/gemma4-marlin-m1-run-{1,2,3}.json`.
