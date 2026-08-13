# DeepSeek V4 Stage 5R.1 residual attribution — 2026-08-11

Status: **REJECTED / BLOCKED / REVIEW REQUIRED**. This experiment preserves
the Stage 5R rejection and does not authorize Stage 6. No performance
correction was implemented.

Branch: `exp/dsv4-rank-local-moe-r21`, based on Stage 5R result commit
`74dd7cfd02e464e4ae75a5d96caf02fd2b724b7f`.

Operating point: `CUDA_VISIBLE_DEVICES=1,2`, runtime devices 0/1, two RTX
3090s without CUDA P2P, batch-one actual replay at position 104, all 43
layers, production tiled NUMA arenas and hot-worker scheduler, one final
completion boundary.

## Predeclared question and gates

Stage 5R's binding candidate was 77.172562 ms with a 72.299077 ms maximum
rank CPU body, so its derived residual was:

```
Sigma_serial = 77.172562 - max(72.299077, 5.735872) = 4.873485 ms/43 layers
required removable serial work = 4.873485 - 2.125 = 2.748485 ms
```

The hypothesis was that at least 2.748485 ms is non-overlapping lifecycle
serialization—callback scheduling, partial H2D, join, NCCL/status,
publication, final drain, or pipeline bubbles—rather than unavoidable load.
The primary metric was causally attributed, non-overlapping removable
milliseconds. The measured bottleneck remains the concurrent routed CPU body;
its sign is unchanged. GPU shared work, H2D, NCCL, status, publication, and
other non-body resources are either overlapped or added load, so a correction
targeting them would be negative unless it removed the binding serial term.

The exactness gates were all 43 routes, coefficients, CPU rank partials,
accepted outputs, BF16 local join association, FP32 NCCL data/U32 status
association, BF16 publication, four injected pre/post-collective failures on
both ranks, no non-finite output, and fail-closed publication. Resource gates
were 21,287,272,448 B per GPU and 231,928,233,984 B host; zero timed
checkpoint reads, CUDA workspace allocations, explicit host allocations,
per-layer host collection, and dependent output D2H. The cheaper rejected
alternative was to launch a full-model decode or claim attribution from the
old 0077 probe; one replay position with 43 real layers is the bounded
falsifier. The predeclared setup/window expectation was about 24.7 s setup for
an approximately 80 ms chain (about 300:1 per single chain). The final matrix
staged in 28.521 s and priced roughly 0.96 s of 12 measured chains (about
30:1).

## Instrumentation perturbation and redesign

The first opt-in arm recorded fixed CUDA timing events at every layer and
collective boundary. It passed exactness, failure, memory, and I/O checks but
failed the perturbation gate:

| arm | wall median/range (ms/43 layers) |
|---|---:|
| event-off control | 83.083179 (single measured arm) |
| event-on control | 87.894008 |
| event-off candidate | 80.866959 |
| event-on candidate | 86.127517 |

The candidate residual moved from 4.834031 to 6.420033 ms. The raw rejected
arm is preserved at
`results/dsv4-rank-local-moe-r21/stage5r21-attribution-one/attribution.log`
(SHA256
`037930d68488177d81244194687c2fcd4da5a8d09a078368e6edff9541f15b77`).
Those event numbers are not used as causal evidence.

The measurement was redesigned to fixed-lifetime `steady_clock` timestamps
only: enqueue-to-callback start, callback body start/finish, first fill,
between-callback idle, and the final completion span. Per-layer CUDA/NCCL
event recording was removed from the tracked source because it perturbed the
production chain. Hidden H2D, partial H2D, shared completion, local join,
NCCL data/status, BF16 publication, and callback-finish-to-H2D clock alignment
are therefore explicitly **unmeasured**, not zero and not inferred from the
wall residual. The prior event arm remains raw evidence of the measurement
defect.

A one-pair host-only perturbation check passed structurally and showed no
systematic positive on-arm shift:

| arm | off (ms) | on (ms) |
|---|---:|---:|
| control | 83.811903 | 80.537731 |
| candidate | 84.145320 | 77.915845 |

Its raw log is
`results/dsv4-rank-local-moe-r21/stage5r21-host-one/attribution.log`
(SHA256
`25ec0814040809f520851bd8e25ffd9121c649a285024a6a9c9e910a07404c1f`).
No speedup is credited to instrumentation; the final three-run matrix below
contains visible scheduling variance.

## Binding three-run host attribution

The final artifact is
`results/dsv4-rank-local-moe-r21/stage5r21-final-clean/attribution.log`
(SHA256
`f30d447ea1bf9792eb3fecf9a934a1cb0e74145438e793c7ec130f4e385e0201`).
It used one warmup and three interleaved measured off/on repetitions. The
program's three-sample median is the middle sorted sample.

| arm | median wall | range wall |
|---|---:|---:|
| control, instrumentation off | 80.616859 | 79.348700–81.193217 |
| control, instrumentation on | 79.837465 | 78.577415–79.872613 |
| candidate, instrumentation off | 80.352462 | 76.347796–80.619033 |
| candidate, instrumentation on | 77.043135 | 76.672821–81.345846 |

The on medians and ranges do not show a positive instrumentation shift. All
samples are retained; no faster arm is converted into an attribution win and
no slower arm is discarded.

The host-only attributed spans are dependency-ordered and per-rank sums, not
additive wall components. Final on-arm medians/ranges were:

| span | control rank 0 | candidate rank 0 | candidate rank 1 |
|---|---:|---:|---:|
| enqueue → callback start | 1,595.576065 (1,531.828379–1,601.531118) | 1,389.650782 (1,353.413906–1,457.474638) | 1,385.920540 (1,349.213183–1,453.472247) |
| callback body | 77.753233 (75.634207–77.815289) | 71.880472 (70.501757–75.006645) | 70.914506 (69.827488–73.283732) |
| first fill | 0.188772 (0.155652–0.373142) | 0.154350 (0.153367–0.155334) | 0.178592 (0.155189–0.315235) |
| between-layer idle | 1.905519 (1.782356–2.496657) | 5.528870 (4.838588–5.849032) | 6.592254 (5.095895–7.026809) |
| final drain | 76.003759 (72.904045–76.019235) | 66.273102 (64.034547–70.514913) | same candidate chain span |

The candidate on-arm maximum callback body per repetition was 75.006645,
71.880472, and 70.914506 ms (median 71.880472). The candidate off-arm
maximum CPU body was 73.426832, 73.542878, and 71.219915 ms (median
73.426832). These are CPU-body measurements, not removable serial work.

The large enqueue-to-callback sums count queueing across 43 dependent
commands and overlap one another. Between-layer idle is measured separately
per rank and overlaps the other rank and device work. Final drain spans the
single completion wait and overlaps callbacks and GPU work. None can be
subtracted from `Sigma_serial` without a clock-aligned dependency interval.

## Resource and correctness evidence

The final log reports 43 exact outputs, partials, and routes in both control
and candidate, four failure cases with failure closure, and no changed
workload or arithmetic. Control routed payload remains 3,449,290,752 B per
43-layer chain. The diagnostic all-arm control line measured 77.265010 ms of
CPU phase and 44.642339 GB/s; it is not used to replace the accepted Stage 5R
binding control because the attribution matrix contains intentional on/off
instrumentation and visible variance. The accepted production comparison is
still Stage 5R's 84.915 ms / approximately 40.6 GB/s, with the 36.7 GB/s
gate.

The candidate retained the Stage 5R logical volumes: 1,409,024 B FP32 data
collective, 344 B U32 status collective, 2,818,048 B rank-partial H2D, and
8 B final status D2H per chain; physical NCCL SHM traffic was not measured.
The final run reports:

```
rss_bytes=158627143680 resident_bytes=156885843968
gpu0_used_bytes=2062680064 gpu1_used_bytes=837877760
measured_checkpoint_calls=0 measured_checkpoint_bytes=0
measured_cuda_workspace_alloc_calls=0 measured_cuda_workspace_alloc_bytes=0
measured_host_allocations=0
```

All ceilings and zero-I/O/allocation gates pass. The binding Stage 5R cost
model is unchanged:

```
tau_control   = max(77.899226 CPU, 5.735872 shared GPU, other terms)
                + 3.313316 = 81.212542 ms
tau_candidate = max(72.299077 CPU, 5.735872 shared GPU, other terms)
                + 4.873485 = 77.172562 ms
```

The candidate CPU body remains the `argmax`; the host-only attribution does
not establish a smaller `Sigma_serial`, nor does it establish that any other
resource is the bottleneck. The accepted Stage-2 approximately 1.427 ms
reduction term and the candidate lifecycle spans must not be stacked as
independent savings.

## Decision and review boundary

**REJECT/BLOCKED.** The experiment did not identify 2.748485 ms of causally
removable, non-overlapping serial work. The only low-perturbation spans that
were measured are overlapping host scheduling/body intervals; the device and
NCCL boundaries needed to separate the residual were unmeasured after the
event arm failed its perturbation gate. Therefore no narrow correction is
selected and no optimistic subtraction is reported.

This closes the Stage 5R.1 measurement at its review boundary while preserving
Stage 5R and 0077 as rejected experiments. Stage 6 remains **BLOCKED / NOT
AUTHORIZED**. Human review must decide whether to accept this technical
negative/measurement-defect closure and explicitly authorize a new bounded
measurement, or leave the roadmap blocked; this branch does not begin Stage 6.
