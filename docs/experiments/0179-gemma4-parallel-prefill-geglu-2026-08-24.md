# Experiment 0179 — Gemma 4 parallel prefill GeGLU removes one host bottleneck

Status: **POSITIVE BOUNDED MECHANISM; PREFILL DEFECT REMAINS OPEN.** Exact
physical-core parallelism removes 45.4% of the measured prefill wall and moves
prefill below batch-1 decode per token, but the resulting 22.85 tok/s remains
about 131x below the owner's approximately 3,000 tok/s vLLM reference. This is
an intermediate correction, not closure of the defect opened by experiment
0165.

## Branch, operating point, and gates

- Branch: `fix/gemma4-prefill-batching`, based on local `main@1acb047`. The
  local main was one user-owned CLI commit ahead of `origin/main`; it was
  preserved. The unrelated untracked DeepSeek script was untouched.
- Checkpoint: `models/gemma4/model.safetensors`, the single
  19,531,513,296-byte MXFP4 shard used by 0165.
- GPU: PCI-order device 1, RTX 3090 at bus `00000000:82:00.0`, alone at the
  start of every accepted arm. `CUDA_DEVICE_ORDER=PCI_BUS_ID` was explicit.
- Operating point: SM 1,605 MHz, 250 W, 24,576 MiB, one-device placement,
  `--vram-fraction 0.95`, context 512. The card's actual host link is PCIe Gen3
  x8, not the device's Gen4 capability.
- Build: `build-release`, verified `CMAKE_BUILD_TYPE=Release` and
  `-O3 -DNDEBUG` before measurement.
- Workload: the supplied prompt, rendered to 23 tokens, 32-token greedy cap,
  seed 33377335. There are 31 decode calls and 30 steady decode calls.
- Hypothesis: independent GeGLU elements are the measured host `argmax_r`; a
  persistent pool over physical cores reduces that term without changing any
  arithmetic order within an element.
- Primary metric: prefill wall ms/token. The predeclared architecture
  expectation was prefill/decode below 0.25; the mechanism kill gate was less
  than a 30% projected full-prefill saving after charging 60 pool barriers.
- Correctness: bit-identical operation probe, identical full-model prompt and
  generated IDs across all arms, unchanged route census, then `make check`.
- Memory: one resident 19.5 GB checkpoint plus the admitted attention/KV
  workspace on one 24 GiB card. Worker stacks add host virtual memory only;
  observed RSS rose by about 4.3 MiB.
- Rollback/control: `STRATA_GEMMA4_PARALLEL_PREFILL=0`.

## Cost model before mechanism design

The initial production profile measured 23-token prefill at 1,830.617 ms and
30-token steady decode at 55.419 ms/token: a ratio of 1.436. A second opt-in
host attribution run measured 1,835.640 ms and explained 1,820.896 ms (99.2%).

| Prefill phase or resource | Total | Share of wall |
|---|---:|---:|
| CPU GeGLU | **981.056 ms** | **53.4%** |
| CPU Q/K norm + RoPE | 284.098 ms | 15.5% |
| gate/up/down projection wall | 196.802 ms | 10.7% |
| Q/K/V/O projection wall | 109.707 ms | 6.0% |
| attention call | 100.186 ms | 5.5% |
| KV host commit | 54.642 ms | 3.0% |
| four host layer norms | 69.578 ms | 3.8% |
| all other attributed work | 24.827 ms | 1.4% |
| unattributed serial residual | 14.744 ms | 0.8% |

The makespan resources at this operating point were:

| Resource | Measured work and service |
|---|---|
| CPU/host | about 1,515 ms outside CUDA synchronization; `argmax_r` |
| CUDA stream critical service | 281.537 ms |
| MXFP4 kernel/HBM | about 36.75 GB over two <=16-row chunks in 73.154 ms = about 502 GB/s, 54% of the RTX 3090's 936 GB/s rating |
| activation H2D | 314.9 MB in 55.069 ms = 5.72 GB/s |
| activation D2H | 390.1 MB in 62.915 ms = 6.20 GB/s |

The measured PCIe rates are 72--79% of the actual Gen3 x8 ceiling of about
7.88 GB/s per direction. Neither HBM nor PCIe was an order-of-magnitude
collapse. The weight read was already amortized: kernel service was only 3.18
ms per prefill token, 5.7% of decode's per-token kernel service. Experiment
0165 correctly opened the implausible-ratio defect, but its specific statement
that register-fed prefill reread every weight per row was falsified here.

The target term was CPU GeGLU. Parallelism changes its effective CPU service
rate while preserving its work. It leaves HBM, PCIe, GPU kernels, attention,
KV volume, precision, and layout unchanged, and adds one host-pool barrier per
layer.

## Cheapest mechanism screen

A production-shape C++ probe used exactly `23 * 21,504 = 494,592` elements,
seven repetitions, the scalar function as oracle, and the persistent worker
pool. Build plus run took 2.6 seconds.

| Width | Median/call | Speedup | Projected 60-layer time |
|---:|---:|---:|---:|
| scalar | 18.656 ms | 1.00x | 1,119.4 ms |
| 7 physical | 3.276 ms | 5.70x | 196.5 ms |
| 14 physical | 2.728 ms | 6.84x | 163.7 ms |
| **28 physical** | **1.477 ms** | **12.63x** | **88.6 ms** |
| 56 logical | 2.392 ms | 7.80x | 143.5 ms |

Every width was bit-identical to the scalar oracle. Twenty-eight physical
cores cleared the 30% gate by projecting an 892 ms system saving. SMT was
negative, so the runtime discovers physical-core primaries from sysfs and does
not use logical width.

Fixed setup was about 19.5 seconds. The measured system window was 3.54 seconds
for control and projected at 2.65 seconds for candidate, fixed-to-measured
ratios of about 5.5:1 and 7.4:1. One profile arm cost about 22--23 seconds; the
six-arm matrix cost about 2.3 minutes. A longer prompt was rejected because it
would change the operating point and was unnecessary to measure the identified
23-row term.

One first candidate run was discarded as a wiring defect, not reported as a
negative result: the parallel block had landed in `vision_mlp`, so text GeGLU
remained exactly 981.5 ms. Moving it to the intended text `mlp` path produced
the predicted shape.

## Counterbalanced A/B

Order was parallel/scalar, scalar/parallel, parallel/scalar. All times below
are complete prefill wall; phase-event profiling was disabled equally in both
arms.

| Arm | Every run, ms | Median | ms/token | tok/s | Full spread |
|---|---|---:|---:|---:|---:|
| parallel | 996.031, 1,026.039, 1,006.482 | **1,006.482** | **43.760** | **22.851** | 30.008 ms |
| scalar | 1,842.570, 1,852.090, 1,815.983 | **1,842.570** | **80.112** | **12.483** | 36.107 ms |

Result: 45.38% less prefill wall, or 1.8307x throughput. The 836.09 ms median
gap is more than 23 times the larger full observed spread.

Steady decode medians were 55.198 ms/token parallel and 55.034 ms/token
control, a 0.30% difference. The candidate prefill/decode ratio is 0.793,
against control's 1.456. All six prompt/generated ID records had SHA-256
`2ccaa433...b4bb`; all six route censuses had SHA-256
`c4afe0a2...d2a6`. Candidate RSS was 3,140,908--3,141,360 KiB versus control
3,136,624--3,137,076 KiB.

The profiled candidate's GeGLU phase was 118.448 ms, 8.28x below the profiled
981.056 ms control while H2D bytes, D2H bytes, and kernel service were
unchanged within noise. Q/K norm + RoPE became the largest host term at
290.407 ms, essentially tied with 280.521 ms CUDA stream service.

## Correctness

The production-shape operation probe was bit-identical at every width. Every
A/B arm generated the same IDs. The binding suite passed:

```text
strata-tests                  Passed  78.59 s
strata-sim-smoke              Passed   0.09 s
strata-equivalence-gemma4     Passed  23.11 s
100% tests passed, 0 failed
Total                         101.79 s
```

## Conclusion and next term

Parallel GeGLU is accepted as an exact intermediate fix. It proves that the
implausible ratio was a host-orchestrated prefill defect, not a property of
Gemma 4. It does not satisfy the architecture expectation or the external
throughput target: 22.85 tok/s is still about 131x below the owner's reported
approximately 3,000 tok/s vLLM result.

The remaining implementation executes norms, RoPE, KV preparation/commit, and
each projection through host-visible synchronous boundaries. A genuine
device-resident page graph with a wide page GEMM is therefore the next bounded
hypothesis; another isolated host-loop optimization cannot plausibly close a
two-order-of-magnitude gap.

Raw ignored artifacts:

- `results/gemma4-prefill/profile-main/`
- `results/gemma4-prefill/profile-host-phases-r2/`
- `results/gemma4-prefill/parallel-geglu-profile-r2/`
- `results/gemma4-prefill/geglu-ab/`
- `results/gemma4-prefill/geglu-check.log`
