# Experiment 0134 — clean SM86 QPN campaign baseline

Status: **C1 COMPLETE. A fresh branch from `main`, containing only the
campaign-contract documentation commit, reproduces the identified RTX 3090,
the 842-class cold-read ruler, the exact production control, and the latest
conventional N64 WMMA control in three independent processes. No QPN candidate
or production runtime change exists in this experiment.**

## Question, contract, and budget

Hypothesis: the clean `main` baseline can reproduce the measurement controls
needed for the Ampere campaign without inheriting any runtime or probe code
from `exp/dsv4-qpn-packed-decode`.

Primary metrics were three independent cold process medians for the ruler and
both production shapes. Correctness required the production and conventional
WMMA arms to have identical double-oracle error and zero full-output delta at
M=1. The device-allocation ceiling was 512 MiB. Rollback was deletion of this
new experiment branch; the archived branch was never modified.

The experiment does not claim a throughput improvement and targets no resource.
Its measured context remains the archived attribution: legacy SIMT was
ALU/instruction-bound and conventional WMMA had a larger no-eligible/barrier
serial residual than its DRAM term. No C2 mechanism may inherit those constants;
it must be profiled at its own operating point.

Each process took about 0.7 seconds. The directly timed kernel windows total
roughly 10 ms per process; allocation, deterministic stimulus generation,
uploads, full-output correctness, and the required 256 MiB scrubs make fixed
work approximately 70 times the event windows. A roofline-only probe would be
cheaper but was rejected because it could not satisfy C1's production and
WMMA-control correctness gates. A model load or Nsight profile would answer no
additional C1 question and was rejected.

## Clean lineage and implementation

The branch `exp/dsv4-sm86-qpn-register-feed` was created directly from
`main@b895f82`. Only documentation commit `750cf20`, recreated by cherry-pick
as `b0cd35d`, was brought across before the probe was written. Therefore none
of experiments 0127–0133's probe or runtime commits are ancestors of this
work.

The new target is
`strata-dsv4-sm86-qpn-baseline-probe`, compiled only for SM86. It contains
exactly three arms:

1. `roofline`: the corrected 128 MiB ILP-4 read stream from experiment 0129;
2. `production`: the one-block-per-output-row E2M1/E8M0 group-32 production
   arithmetic and launch geometry;
3. `conventional_wmma_n64`: packed shift/rebias decode into BF16 shared
   memory, BF16 WMMA 16x16x16, N=64, split-K=4, followed by the common fp32
   partial reduction.

The third arm is intentionally a control, not the target architecture. It
materializes decoded weights and activations in shared memory and executes
two block-wide barriers per K tile, unlike the register-fed QPN thesis.

The probe uses three warmups and eleven interleaved samples in each process.
Cold samples scrub 256 MiB before every arm and rotate over eight separately
placed packed matrices. Deterministic activations are BF16-representable.
Unlike the archived harness stimulus, both nibbles of each generated weight
byte are populated.

Build and run:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target \
  strata-dsv4-sm86-qpn-baseline-probe -j 48
./build-release/strata-dsv4-sm86-qpn-baseline-probe \
  --device 0 --output results/qpn-sm86/0134-run1.json
```

Runs 2 and 3 use the corresponding deterministic output filenames.

## Device, bytes, and memory

Every raw result identifies CUDA ordinal 0 as `NVIDIA GeForce RTX 3090`,
compute capability 8.6, with 82 SMs.

Both production shapes contain the same number of logical weights. Effective
packed-weight bandwidth uses this explicit formula:

```text
code bytes  = N * K / 2  = 4,194,304
scale bytes = N * K / 32 =   262,144
total bytes =                 4,456,448
GB/s        = total bytes / event time in nanoseconds
```

No activation, partial, or output byte is included in the headline numerator.
Those bytes still consume hardware resources and must be reported by C2
profiles; the formula measures useful packed-weight progress consistently.

| Shape | Declared device allocation | MiB | Ceiling |
|---|---:|---:|---:|
| gate_up_w1 `[2048,4096]` | 438,861,824 B | 418.53 | 512 MiB |
| down_w2 `[4096,2048]` | 439,394,304 B | 419.04 | 512 MiB |

The allocation consists principally of a 256 MiB scrub buffer, 128 MiB ruler,
eight packed matrix replicas (34 MiB total), activations, two output buffers,
and split-K partials. Host stimulus storage is not included in device bytes.
The replicas are a declared probe-only cold-placement mechanism, not a proposed
persistent production layout.

## Results

All values are each process's median of eleven interleaved samples.

| Run | Shape | Ruler cold GB/s | Production cold GB/s | N64 WMMA cold GB/s | WMMA vs production max abs |
|---:|---|---:|---:|---:|---:|
| 1 | gate_up_w1 | 840.71 | 87.04 | 162.12 | 0 |
| 1 | down_w2 | 845.63 | 87.04 | 173.65 | 0 |
| 2 | gate_up_w1 | 840.21 | 87.04 | 161.37 | 0 |
| 2 | down_w2 | 845.63 | 87.04 | 174.08 | 0 |
| 3 | gate_up_w1 | 845.63 | 87.04 | 161.19 | 0 |
| 3 | down_w2 | 840.21 | 87.04 | 174.08 | 0 |
| **median** | **gate_up_w1** | **840.71** | **87.04** | **161.37** | **0** |
| **median** | **down_w2** | **845.63** | **87.04** | **174.08** | **0** |

Median event times were 159.648/158.720 us for the 128 MiB ruler,
51.200 us for production on either shape, and 27.616/25.600 us for N64 WMMA.
The ruler reproduces experiment 0129's 840–846 GB/s range. The controls are
reported as the new clean-baseline operating point; archived timings are not
silently reused.

## Correctness

All 2,048 gate-up outputs and all 4,096 down outputs were compared. The N64
WMMA output is bit-identical to production at M=1 on both shapes.

| Shape | Production oracle max abs | N64 WMMA oracle max abs | Full-output WMMA vs production |
|---|---:|---:|---:|
| gate_up_w1 | 8.682556 | 8.682556 | 0 |
| down_w2 | 7.921265 | 7.921265 | 0 |

The oracle errors are the established fp32 summation/BF16-output class, and
the candidate reproduces the production result exactly at the binding baseline
M=1 point. No fallback exists in the standalone executable.

## Gate verdict and next dependency

C1 passes every binding gate:

- fresh branch from `main`, with only the contract transferred;
- target verified as RTX 3090 SM86;
- 842-class ruler reproduced in three processes;
- explicit packed-code plus scale byte formula recorded;
- production and conventional WMMA controls reproduced;
- correctness clean with no new delta;
- 512 MiB device-allocation ceiling respected.

This result authorizes C2; it does not validate QPN on Ampere. The exact next
falsifier is an isolated inline-PTX SM86 BF16 MMA fragment-map probe. It must
first establish the lane/register mapping and intended SASS for
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`. Only then may the same
bounded experiment add fragment-order E2M1/E8M0 group-32 prepacking and direct
register decode. No runtime dispatch or conventional WMMA tuning is next.

## Artifacts

Raw JSONs are ignored and deterministic:

- `results/qpn-sm86/0134-run1.json`
- `results/qpn-sm86/0134-run2.json`
- `results/qpn-sm86/0134-run3.json`
