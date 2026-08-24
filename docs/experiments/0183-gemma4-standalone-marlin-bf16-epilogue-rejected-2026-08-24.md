# Experiment 0183 — standalone Marlin core passes speed, fails precision

**Date:** 2026-08-24  
**Branch:** `fix/gemma4-marlin-page-kernel`  
**Origin:** experiment 0182's exact next mechanism  
**Verdict:** **REJECTED — BF16 output epilogue violates Strata's contract**

## Hypothesis and predeclared gate

A standalone BF16/MXFP4 specialization of the Apache-licensed Marlin dataflow
can reproduce vLLM's measured exact-shape ruler within 1.28x, at most 0.63 ms
for both M=128 Gemma MLP shapes, without Torch or a framework runtime.

The numerical gate remained Strata's declared compact MXFP4 contract: maximum
relative difference below 1e-4 against the canonical FP32-accumulation oracle.
Transient probe memory was capped at 256 MiB; checkpoint codes and E8M0 scales
had to remain a single compact persistent representation. A failure of any
gate was a stop before repeated arms or runtime integration.

## Cost-model position

The production `argmax_r` remains 2,419.841 ms of serial CUDA/host handoffs.
This isolated projection does not reduce that term and therefore cannot carry
a system throughput claim. It is a prerequisite falsifier for the later
device-resident executor that would remove host normalization, RoPE, attention,
KV and activation materialization from `sum_serial`.

The signs were predeclared as: projection compute and compact weight traffic
must fall sharply; no H2D/D2H or host term changes in the isolated probe;
load-time code/scale permutation adds no bytes; temporary reduction/work locks
add bounded device memory. Executor work was forbidden until both speed and
precision gates passed.

## Mechanism and cheap experiment

The probe specializes vLLM 2.3.8's Apache-licensed Marlin CUDA template for
BF16 activation, E2M1 weights, E8M0 group-32 scales, four M blocks, sixteen N
blocks, four K blocks, 256 threads and four async stages. It implements the
K16/N64 load-time code permutation and both Marlin scale permutations locally.
The compile-time scalar shim contains no Torch API and no framework is linked.

The rejected alternative was another model load: the two projection kernels
decide this stage directly. Compilation took 11.2 s; the one process completed
both prepack/oracle/timing arms in 3.5 s. The sub-millisecond measured window is
small relative to setup, but the 32-launch steady window removes event-launch
floor and matches the vLLM ruler method. Planned total budget was under five
minutes.

## Result and binding stop

| Shape | Standalone median | vLLM Marlin ruler | Speed gate | Max relative | Precision gate |
|---|---:|---:|---:|---:|---:|
| gate/up `[21504,5376]` | **0.486875 ms** | 0.490811 ms | pass, 0.992x | **0.003474** | **fail, 34.7x over** |
| down `[5376,21504]` | **0.493824 ms** | 0.506624 ms | pass, 0.975x | **0.003669** | **fail, 36.7x over** |

Useful compact rates are 126.14 and 124.37 GB/s, matching the external
125.13/121.22 GB/s ruler. Probe device bytes peak at 222,650,696, below the
256 MiB ceiling. The extra bytes are probe-only simultaneous canonical,
register-fed, and Marlin copies; no duplicate was admitted to the runtime.

The numerical failure has the exact shape of the upstream epilogue: Marlin
converts each FP32 accumulator pair to BF16 before the final global write,
whereas Strata's accepted scalar and register-fed routes return FP32
accumulations and defer rounding to the declared architecture operation. The
observed approximately half-BF16-ULP relative difference is not admissible
under 1e-4. Treating vLLM's BF16 output boundary as permission to change
Strata's boundary would be a silent precision change.

The correctness gate failed on the first process, so the planned second and
third repetitions were not run. This is a gate stop, not a variance verdict.

## Verdict and exact next action

**REJECTED.** Do not integrate the upstream BF16-output specialization and do
not build a device-resident executor on it. The experiment proves that the
Marlin load/repack/dequant/MMA/scheduler core is the missing performance
mechanism, but it has not produced an admissible Strata kernel.

The next bounded experiment may change only the result/reduction epilogue:
preserve FP32 accumulators through cross-CTA reduction and write FP32 output in
Strata's canonical `[M,N]` layout. It must retain <=0.63 ms on both shapes and
restore <=1e-4. Only then may the projection primitive be considered accepted.

The raw result remains outside Git at `/tmp/gemma4-marlin-standalone.json`.
