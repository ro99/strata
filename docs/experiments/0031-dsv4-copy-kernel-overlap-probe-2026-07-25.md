# Experiment 0031 — does an H2D copy overlap a running kernel?

## Contract

- Hypothesis under test: experiment 0026 rejected a per-device copy stream, but
  its kill criterion measured copy-queue depth — sync-per-copy against batch-64
  on one device, with nothing else running. That is not the property a copy
  stream exists for. `CudaBackend::upload` issues every H2D on `state.stream`,
  the same stream every kernel runs on, and then synchronizes it
  (`kernels/cuda/backend.cu:1598`), so no compute is ever in flight during an
  upload and a deeper queue has nothing to recover. The unmeasured claim is
  whether a copy and a kernel can proceed at the same time at all.
- Primary metric: `overlap_efficiency = (shared_stream - split_stream) /
  min(copy_only, kernel_only)`. 0.0 means a copy stream is worth nothing here
  and the mechanism is dead; 1.0 means the smaller term disappears completely.
  Reported as a ratio rather than a speedup so it does not depend on the
  calibrated duration mix.
- Kill criterion, stated before the work: if `split_stream` does not beat
  `shared_stream` outside run variance, the copy stream is rejected a second
  time and the two mechanisms that depend on it — intra-layer overlap of MoE
  compute with a demand load, and the never-run advisory expert prefetch of
  experiment 0022 — are closed with it.
- Correctness: no runtime code changed. This experiment is measurement only.
  The probe verifies both engines did the work its arms claim.
- Cost: no model load. About 90 seconds of probe time, three repetitions.

## Result

The hypothesis is **confirmed**. A copy and a kernel on one stream are strictly
additive; on separate streams the smaller term vanishes entirely. The mechanism
is worth its full theoretical value at the CUDA level on all three devices.

## Cheap measurement first

`strata-topology-probe` gained a `copy_kernel_overlap` stage. It reuses the
existing cold arena — a registered 16 GiB anonymous mapping sampled at randomly
permuted, non-repeating 4,456,448-byte offsets, the access pattern experiment
0024 established — and adds a kernel that streams one routed-expert triplet
(13,369,344 B, three matrices) out of HBM per pass. The arithmetic is not the
model's and does not need to be; what has to be reproduced is the device-memory
pressure and the launch duration, because HBM contention against the copy's own
HBM writes is the thing that could prevent overlap.

Four arms per device, medians of three repetitions:

| arm | what it does |
|---|---|
| `copy_only` | cold slices, synchronize per copy |
| `kernel_only` | launches calibrated to cost about what a copy costs |
| `shared_stream` | copy and kernel per iteration on one stream — today's runtime |
| `split_stream` | copy on its own stream, kernel on the compute stream |

Equal terms is where a sum and a max are furthest apart, so calibration targets
`kernel_only / copy_only = 1.0`.

## Measured

`results/deepseek-v4-copy-kernel-overlap-v2/`, 256 iterations per arm:

| | dev 0 (5060 Ti) | dev 1 (3090) | dev 2 (3090) |
|---|---:|---:|---:|
| calibrated passes | 90 | 339 | 154 |
| duration ratio (target 1.0) | 0.997 | 0.918 | 0.893 |
| `copy_only` s | 0.17345 | 0.19136 | 0.09626 |
| `kernel_only` s | 0.17294 | 0.17570 | 0.08598 |
| `shared_stream` s | 0.34695 | 0.37674 | 0.18273 |
| `split_stream` s | **0.17804** | **0.19261** | **0.09679** |
| shared / (copy + kernel) | 1.002 | 1.026 | 1.003 |
| split / max(copy, kernel) | 1.026 | 1.007 | 1.006 |
| **overlap efficiency** | **0.977** | **1.048** | **0.999** |
| shared / split | 1.949 | 1.956 | 1.888 |

Two bounding checks decide this, and both hold on every device:

**A single stream is strictly additive.** `shared_stream` is the sum of the two
arms to within 0.2%, 2.6% and 0.3%. There is no partial overlap to recover on
the stream the runtime uses; the copy waits for the kernel and the kernel waits
for the copy, exactly as the stream semantics require.

**A split stream collapses to the max.** `split_stream` equals
`max(copy_only, kernel_only)` to within 2.6%, 0.7% and 0.6%. The smaller term
is not reduced — it is gone.

At equal terms that is **1.89x–1.96x on the combined copy-plus-kernel term**.
Overlap efficiency is 0.98–1.05, so the ceiling is the full theoretical one.

## Gates

`copy_kernel_overlap[].verified` is true on all three devices: the last copy's
destination was read back and compared byte-for-byte against its source slice,
and the kernel's accumulator sink was confirmed nonzero, so neither engine was
idle in an arm that claims it worked. `make check` passes 2/2. No runtime code
changed.

The cold-slice arms ran unchanged in the same process as a control and
reproduce experiment 0026 in the v1 run: serial round-robin 6.41 GB/s against
0026's 6.37, overlapped 10.81 against 10.52.

## Two defects found

**The first calibration missed by 6x on two of three devices.** v1 measured one
launch at `passes = 1` and extrapolated linearly. Cost per pass is not constant:
the first pass reads the triplet from HBM, later passes hit L2, and a one-pass
launch carries undiluted launch overhead — so the extrapolation overshot and
`kernel_only` came out at 0.15–0.16 of `copy_only` instead of 1.0. Per the
charter that is a defect, not a datapoint. The calibration now measures at the
current pass count and corrects, converging inside a 0.85–1.15 band.

It is worth recording that the conclusion did not change: v1's overlap
efficiencies were 1.005 / 0.990 / 1.013 at a 6:1 duration mix, against
0.977 / 1.048 / 0.999 at 1:1. Both bounding checks held in both runs. The fix
makes the arm discriminating, not the answer different.

**The arena's NUMA placement moved between runs, on the same binary.** Device 2's
`copy_only` rate was 7.17 GB/s in v1 and 11.03 GB/s in v2, and the cold-slice
overlapped aggregate was 10.81 against 15.92 — the v2 figures sitting at 0026's
`membind=1` measurements. This independently reproduces 0026's finding that
`Dsv4ResidentWeightStore::stage` allocates with no NUMA policy, so the step's
largest term has a rate that depends on which staging worker touched which page.
It does not affect this experiment's metrics, which are ratios taken within one
device in one process; the overlap conclusion held across a 1.65x change in the
copy rate, which is a useful robustness check.

## What this closes and what it does not

**Closed.** The claim that a dedicated copy stream is worth nothing is
withdrawn. 0026 measured that a *deeper copy queue* recovers nothing, which is
true and remains true — a 4.46 MB copy already saturates its link and no second
copy is queued behind it. The copy stream's value was never queue depth. It is
that the copy engine and the SMs are separate resources, and today's single
stream bills them serially, which is a `Σ_serial` defect in the cost model's
terms, not a bandwidth one.

**Not closed, and this is the important part.** This measures the *mechanism*
ceiling, not the step. It says nothing about how much of the 86.25 ms/step
demand-wait term is actually hideable, because that depends on ordering the
probe does not model:

- Within a layer, the missing experts are not known until the router runs, so
  only MoE compute can hide a demand load. Experiment 0026's structure gives
  ~2.0 ms of transfer against ~1.09 ms of MoE compute per layer-step, and only
  the resident-expert fraction can be enqueued early.
- Across layers, the advisory prefetch of experiment 0022 has the larger window
  — the next layer's attention and mHC, ~3.4 ms/layer at this operating point,
  against the same ~2.0 ms transfer. Its simulation gate is cleared (7.61% fewer
  modelled bytes, 84.66% useful) and its runtime experiment has never been run.
  It is now clear why running it against the current backend would have measured
  the stream rather than the predictor: `prefetch_loop` reaches the device
  through the same `upload()` on the same stream as the kernels it is supposed
  to hide behind.

Neither number may be assumed. The next gate is a cheap replay of the recorded
decode route trace against the per-layer ordering, projecting how much of the
86.25 ms each of the two mechanisms can actually reach, before any runtime code
is written. Per the charter's dependency rule, the copy stream is a prerequisite
for both, but a prerequisite clearing its gate is not authorization to build the
things above it.

## Commands

```bash
scripts/run_deepseek_v4_copy_kernel_overlap_probe.sh
```

Evidence: `results/deepseek-v4-copy-kernel-overlap-v2/` (corrected calibration)
and `results/deepseek-v4-copy-kernel-overlap/` (v1, retained for the defect).
