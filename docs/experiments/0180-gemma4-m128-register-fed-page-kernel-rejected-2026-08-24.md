# Experiment 0180 — Gemma 4 M=128 register-fed page kernel is rejected

**Date:** 2026-08-24  
**Branch:** `fix/gemma4-device-page-prefill`  
**Origin:** experiment 0165's separate prefill defect  
**Verdict:** **REJECTED — do not integrate this page-kernel family**

> **Subsequent reference correction:** the 3000 tok/s vLLM figure used to set
> this experiment's 600 GB/s gate had not been reproduced. Experiment 0181
> later measured the exact checkpoint and operating point at 881.67 tok/s for
> the comparable TP=1, 127-token case. The predeclared gate and the result
> against it remain part of this record; 600 GB/s must not be reused as an
> admission threshold for the successor experiment.

## Question and predeclared gate

Gemma 4 MXFP4 prefill invokes the accepted M<=16 register-fed kernel eight
times for a 128-token page. Can one CTA share a compact weight tile across the
eight token bands and reduce the full projected-weight read from eight passes
to one?

The primary gate was at least **600 GB/s useful single-pass weight bandwidth**
on both real Gemma MLP shapes at M=128. The end-to-end target supplied by the
owner is about 3000 prefill tok/s in vLLM on this machine. A 128-token page at
that rate is 42.67 ms; one 18,377,146,368-byte projected-weight pass alone
requires 430.7 GB/s of the whole-page budget, so the isolated kernel needs
substantial margin above that. Correctness required the admitted E2M1/E8M0
group-32 representation, the existing BF16 operand boundary, relative error
below 1e-4, unchanged W8A16 dispatch, no persistent widened weight copy, and
one-3090 admission at `--vram-fraction 0.95`.

Expected before measurement: prefill/decode per-token cost below 0.25. The
measured 128-token reproduction is 49.647 ms/prefill token against 59.882 ms
for the following decode forward, ratio **0.829**, and only **20.14 tok/s**.
Against the owner's 3000 tok/s reference it is about **149x slower**.

That last comparison was the information available before launch, not a
validated baseline. The subsequent direct TP=1 measurement is about **43.8x**
the Strata token rate and **44.8x** the page wall-time performance.

## Mandatory cost model at the production point

The operating point was one PCI-ordered RTX 3090, device 1 / PCI 82:00.0,
250 W, SM clock locked to 1605 MHz, 24 GiB, with the model wholly resident at
VRAM fraction 0.95. This task's direct operating-point requirement overrides
the QPN campaign's otherwise-unlocked performance point; no clocks or caps
were changed.

The exact 128-token profile took 6,354.801 ms wall:

| Phase | Wall ms |
|---|---:|
| embedding | 0.479 |
| input norms | 92.954 |
| q/k/v projections | 436.385 |
| q/k norm and RoPE host transform | **1,610.519** |
| KV prepare | 42.868 |
| attention | 1,305.041 |
| attention output projections | 180.311 |
| KV commit | 301.771 |
| post-attention norms | 92.860 |
| attention residual | 23.954 |
| pre-feedforward norms | 93.030 |
| gate/up projections | 888.632 |
| exact GeGLU | 597.662 |
| down projections | 342.803 |
| post-feedforward norms | 94.600 |
| MLP residual | 38.229 |
| output head | 3.748 |
| KV upload | 38.762 |
| explained sum | 6,184.924 |
| residual | 169.877 |

Resource instantiation of `tau = max_r W_r/B_r + sum_serial`:

| Resource/term | Work and measured service | Time |
|---|---|---:|
| projected MXFP4 weights | 18.377 GB useful, but 8 chunks move about 147.017 GB; about 500 GB/s useful kernel service | 293.972 ms CUDA kernels |
| activation H2D | 1.752 GB at 5.80 GB/s versus a 7.88 GB/s Gen3 x8 link | 302.228 ms |
| result D2H | 2.166 GB at 4.54 GB/s versus a 7.88 GB/s Gen3 x8 link | 476.937 ms |
| CUDA/host handoff synchronization | repeated blocking generic matmul/attention calls | **2,419.841 ms** |
| largest host-compute phase | q/k norm plus RoPE transform | **1,610.519 ms** |

`argmax_r` is the CUDA/host handoff serial term at 2,419.841 ms; the largest
host-compute component is q/k normalization plus RoPE. The target mechanism
reduces the eight-pass weight term and, after device-resident integration,
would reduce dependent handoffs. It does not change arithmetic or activation
precision. Its positive signs were 8x less HBM weight volume and fewer
launches; negative signs were more per-CTA accumulators or a transient compact
shared tile. H2D/D2H already reach 74%/58% of the actual Gen3 x8 ceiling, so
they are not an order-of-magnitude bandwidth collapse. The large sync term is
an overlap defect, not a PCIe volume limit.

## Cheap experiment and budget

The mechanism probe uses the two 61,415,424-byte Gemma MLP projections:

`W = N*K/2 + N*K/32`

- gate/up: `[21504,5376]`
- down: `[5376,21504]`

Each process spends about 0.06 s prepacking a shape, then 3 warmups and 11
samples with a 256 MiB L2 scrub, plus a 32-launch steady window. Fixed setup is
therefore roughly two orders of magnitude larger than a single measured
projection, but the whole five-arm sweep completes in seconds. A full-model
arm was rejected because it would pay about 20 s of load and 6.4 s of known-bad
prefill to answer a sub-millisecond kernel question. Planned wall budget was
under two minutes; it completed inside that budget.

Raw JSON and the Nsight report are ignored under
`results/gemma4-prefill/page-kernel/`. Every arm ran in a fresh process, in
interleaved order, three times. The table reports the three steady-window
results and their median in useful GB/s:

| Arm | gate/up runs; median | down runs; median |
|---|---:|---:|
| 1 warp per weight tile | 89.11, 89.13, 89.08; **89.11** | 36.94, 36.92, 37.01; **36.94** |
| 2 warps per tile | 111.50, 111.43, 110.58; **111.43** | 63.33, 63.33, 63.32; **63.33** |
| 4 warps per tile | 97.86, 98.03, 96.87; **97.86** | 93.63, 93.64, 93.63; **93.63** |
| 8 warps, cache broadcast | 81.83, 79.77, 79.71; **79.77** | 76.31, 74.86, 74.84; **74.86** |
| 8 warps, compact shared broadcast | 75.21, 74.20, 73.42; **74.20** | 72.57, 70.96, 70.36; **70.96** |

All individual results miss 600 GB/s by at least 5.38x. This is not a
variance call.

## Attribution

Nsight Compute profiled the first real-shape cache-broadcast launch
(`1344 x 256`, 40 registers/thread):

- duration 748.67 us;
- DRAM throughput 11.11%;
- L2 throughput **89.83%**, L2 hit rate 95.17%;
- ALU utilization **68.4%**;
- 88.02% achieved occupancy, 2.73 waves/SM;
- 65.9% of cycles between issued instructions were long-scoreboard stalls.

The first arm did avoid repeated HBM misses, but it merely moved the defect on
chip: eight warps issue eight L2 reads and perform eight FP4 decodes. Explicit
compact shared broadcast removes those L2 requests but adds two barriers per
four K tiles and is slower. Giving one warp more M columns reduces duplicate
decode but raises accumulator pressure and starves the long-K down projection.
The shape-dependent ownership optimum proves there is no setting in this
family that approaches the gate.

Every arm has the same sampled canonical double-oracle result: maximum relative
error 4.14e-7 gate/up and 5.564e-6 down, below 1e-4. The rejection is purely
performance. No runtime path was integrated.

## Verdict and exact next action at close

**REJECTED.** Do not extend the decode-oriented register-fed kernel to M=128,
do not select a favorable ownership setting per shape, and do not integrate
the shared-broadcast arm. The eight-pass production defect remains open.

The next bounded hypothesis is a conventional page-shaped tensor-core GEMM:
decode compact MXFP4 once into a transient shared/BF16 tile and reuse it across
an M=128 output tile, with no persistent widened weight representation. Its
first action is an isolated real-shape bandwidth/oracle probe at the same
locked production point. Only a >=600 GB/s result permits runtime integration.

The final sentence records the predeclared gate; experiment 0181 invalidates
its external 3000 tok/s premise. The successor must derive a new staged gate
from the measured 881.67 tok/s reference and a projection-level profile, not
silently inherit 600 GB/s.
