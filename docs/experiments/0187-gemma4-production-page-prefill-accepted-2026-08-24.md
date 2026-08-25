# Experiment 0187 — Gemma 4 production page prefill

**Date:** 2026-08-24

**Branch:** `fix/gemma4-production-prefill`

**Origin:** experiment 0165's prefill defect, experiment 0186, and issue #36

**Verdict:** **ACCEPTED — the production multi-page collapse is removed**

## Correction to the prior handoff

Experiment 0186 measured 668.993 tok/s only at exactly M=128. Its runtime
admitted that executor only when the entire rendered prompt was at most 128
tokens and started at position zero. Ordinary shorter prompts still paid an
M=128 Marlin schedule, while any longer or later page silently returned to the
serialized path. Reporting the M=128 result as the production chat speed was
therefore misleading.

The user reproduced both failures in normal `strata-chat`: a 28-token first
turn measured 169.41 tok/s and a 350-token rendered second turn measured 15.76
tok/s. This experiment treats those production shapes, rather than the special
M=128 prompt, as the binding workload.

## Hypothesis and predeclared gates

The bounded hypothesis was that production throughput was lost in three
dispatch/ownership defects: every partial Marlin page was padded to M=128, the
device executor was restricted to the first page, and continuation attention
reread grouped K/V history independently for every query head and row.

- Primary metric: complete production prefill wall and tok/s at M=28 and at a
  348-token, three-page rendered prompt matching the reported M=350 shape.
- Correctness: unchanged first token in every performance arm; candidate and
  serialized generation equality beyond one page and beyond the 1,024-token
  local KV ring; maximum Marlin relative residual below 1e-4 at M=1, M=28,
  M=65, and M=128; the committed Gemma equivalence oracle; then `make check`.
- Memory ceiling: one RTX 3090 at `--vram-fraction 0.95`, including the existing
  16,384-token 20.93 GiB admission. No second persistent weight or KV copy.
- Rollback: any numerical divergence, cache-ring failure, OOM, or improvement
  within observed variance rejects the relevant mechanism.

Before measurement, a full M=128 page was expected to retain the approximately
0.045 prefill/decode per-token ratio from 0186. A three-page prompt should add
causal-attention work with position but remain far below batch-one decode per
token. Using 0186's 33.617 ms steady decode cost, the accepted 348-token result
predicts a ratio near 0.054, still well below the architecture gate of 0.25.

## Mandatory cost model before design

The real operating point was the exact 19,531,513,296-byte MXFP4 shard, PCI
ordered CUDA device 1 / bus `82:00.0`, one RTX 3090 at 1,605 MHz and 250 W,
VRAM fraction 0.95, Release build, and zero cross-device hops.

The exact 28-token production profile on `main` instantiated
`tau = max_r W_r/B_r + sum_serial` as:

| Resource/term | Work and measured service | Time |
|---|---|---:|
| CUDA kernel/HBM service | complete 60-layer page, hidden M=128 schedule | **159.657 ms** |
| H2D | 623,616 bytes at 4.84 GB/s | 0.129 ms |
| D2H | 1,650,688 bytes at 5.92 GB/s | 0.279 ms |
| host/serial residual | embedding, sampling, bookkeeping | about 2.69 ms |
| **complete prefill** | 28 tokens | **162.751 ms** |

`argmax_r` was GPU kernel/HBM service. PCIe reached 61--75% of the Gen3 x8
payload ceiling and contributed under 0.3% of wall, so neither link volume nor
link overlap could close the defect. The first mechanism therefore reduced
the padded projection schedule, the measured bottleneck. It did not increase
weight bytes, link traffic, host work, or persistent memory.

At the 348-token operating point, profiling the first arbitrary-position page
implementation gave this device breakdown:

| GPU phase | Time |
|---|---:|
| Marlin projection kernels | 380.941 ms |
| causal page attention | **259.208 ms** |
| activation quantization | 15.945 ms |
| norms, RoPE, residuals, GeGLU, KV store | 36.651 ms |
| output head | 3.190 ms |
| **CUDA subtotal** | **695.935 ms** |

The second `argmax_r` was again GPU service, now specifically attention's
redundant K/V-read volume. Local layers have two query heads per KV head and
global layers have eight. The grouped mechanism shares each exact BF16 K/V
tile across those heads, and across four adjacent local query rows. Its signs
on other resources are: unchanged projected-weight and PCIe volume, lower KV
read volume, bounded dynamic shared memory and registers, and no arithmetic,
cache-layout, precision, or visibility change.

## Cheap screens and arm budget

Source inspection was the cheapest dispatch screen and found the unconditional
`M=128` padding and first-page/position-zero gates. One exact M=28 profile then
cost about 19.5 seconds of load for 0.163 seconds measured, roughly 120:1. It
rejected a PCIe mechanism and selected variable-M projection service.

The long mechanism profile cost about 19 seconds of setup for a 0.7-second
window, about 27:1. It selected grouped attention. An 8-row tile improved the
real-model long screen from 494.46 to 554.89 tok/s with the same output. A
32-row tile was rejected at 533.72 tok/s: fewer barriers lost to shared-memory
pressure and occupancy. The final six-arm counterbalanced matrix was budgeted
at 2.5--3 minutes and completed within that budget.

## Equal-shape vLLM correction

Experiment 0181's 881.67 tok/s vLLM figure is a 127-token point, not a
28-token point. A fresh local vLLM 2.3.8 server using this exact checkpoint,
device, clock, power cap, prefix-cache-off contract, and exact 28-token chat
prompt measured:

| Run | Prompt tokens | Server prefill ms | tok/s |
|---:|---:|---:|---:|
| 1 | 28 | 78.992 | 354.468 |
| 2 | 28 | 78.551 | 356.455 |
| 3 | 28 | 78.545 | 356.482 |
| **median** | **28** | **78.551** | **356.455** |

Thus the prior 668.99 tok/s M=128 result was not evidence for this 28-row
shape: even the equal-shape vLLM reference is much slower there. Final Strata
measured 508.950, 509.498, and 510.540 tok/s, a **509.498 tok/s median**, with
token 818 in every run. The production defect was the 350-token collapse, not
normal batch-shape scaling.

## Implementation

- Marlin now selects exact partial-M tiles instead of padding every multi-row
  call to 128. M=65--127 is split into a correct 64-row body and exact tail.
- The FP32-correct Marlin reorder tile reuses dynamic shared memory after MMA,
  removing an immediate global-scratch write/read round trip.
- Every text-only one-device MXFP4 chunk can use the device page executor at
  any position. Intermediate chunks no longer compute unused vocabulary
  logits.
- Page attention reads historical BF16 cache rows plus current BF16-rounded
  K/V before committing the current page. A wrapped local ring therefore
  cannot overwrite history still needed by early page rows.
- Explicit `(start, cached_rows)` metadata advances by page and remains the
  authority; no fake host KV allocation or hidden fallback was reintroduced.
- At visible extents of at least 64 rows, a grouped-query kernel shares exact
  K/V tiles across the declared 2-way local or 8-way global GQA groups. If its
  score/tile allocation exceeds the device's dynamic-shared limit, execution
  retains the exact scalar page kernel.

## Final production A/B

The prompt rendered to 348 tokens and therefore exercised three device pages,
including two nonzero positions. Order was candidate/control,
control/candidate, candidate/control. All arms used the same checkpoint,
Release build, one locked 3090, memory bound, prompt, one output token,
temperature zero, and seed.

| Arm | Every run, tok/s | Median tok/s | Median wall | First token |
|---|---:|---:|---:|---:|
| production pages | 554.978, 555.434, 556.010 | **555.434** | **626.537 ms** | 1509 |
| serialized control | 15.955, 15.979, 16.605 | **15.979** | **21,778.818 ms** | 1509 |

The median rate improvement is **34.76x**, and median wall falls by 21.152
seconds. Candidate spread is 1.032 tok/s; the result is far outside both arms'
variance. Prefill costs 1.800 ms/token, versus 33.617 ms for 0186's steady
decode, a per-token ratio of **0.0536**. Prefill is 18.67 times cheaper per
token, matching the architecture's required direction.

The exact M=128 production ruler also remains real rather than special-cased:
the three final runs were 690.015, 685.963, and 689.469 tok/s, for a **689.469
tok/s median**. All produced token 3810. This is above the previously reported
668.993 tok/s median while using the same executor that owns later pages.

## Correctness and promotion gates

The 130-token page-boundary generation gate compared the production executor
with `STRATA_GEMMA4_DEVICE_PAGE=0`. Both arms produced the same eight IDs:

```text
3810 659 5213 236770 236771 236771 1018 623
```

and identical text `There are **100** "`.

The separate KV-ring gate rendered 1,050 prompt tokens, 26 beyond the local
1,024-token capacity. The candidate attended before committing wrapped page
rows; the serialized oracle retained the host cache. Both generated ID 3048
and text `You`. Candidate prefill was 3.279 seconds; the deliberately slow
serialized oracle took 137.649 seconds. This validates ring contents and
metadata across wrap rather than merely checking an in-capacity prefix.

The committed MXFP4 full-model oracle passed: all 1,080 operation/layer hashes,
generated token IDs, and answer match
`tests/fixtures/gemma4/layer-hash-trace-mxfp4.json`. The final Release
`make check` also passed its layer and symbol audits and all three CTest
targets:

```text
strata-tests                  Passed   78.40 sec
strata-sim-smoke              Passed    0.10 sec
strata-equivalence-gemma4     Passed   22.72 sec
100% tests passed, 0 tests failed out of 3
```

## Verdict

**ACCEPTED.** Production chat no longer depends on a special exactly-128-token
first prompt. Partial pages, long prompts, later pages, and the local KV ring
retain the device-owned path and exact cache semantics. Issue #36 remains the
appropriate tracker for further vLLM parity work; this experiment corrects the
production regression and the misleading shape-independent 668.99 claim.

Raw model outputs, profiler captures, and logs remain ignored under
`results/gemma4-production-prefill/`.
