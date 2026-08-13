# Experiment 0077 — DSV4 rank-local MoE actual replay (2026-08-11)

Status: **REJECTED / REVIEW REQUIRED**
Stage: 5 — actual-replay rank-local shared and routed MoE
Branch: `exp/dsv4-rank-local-moe`
Base: `b4566c39daeeadb3f25fb6ed10841c83140ed7c4`
Operating point: `models/dsv4f`, two RTX 3090s, `CUDA_VISIBLE_DEVICES=1,2`,
runtime devices/ranks `0,1`, CUDA P2P disabled, batch one, no speculation,
104-token prompt, 15 timed decode positions (104–118), 28 host attention
threads, admission `0.95`, no decode checkpoint reads.

This document records the implementation, exactness gate, cost-model
measurement, upgraded Nsight read, and binding rejection at the Stage 5 review
boundary. Stage 6 was not started.

## Question and predeclared gates

The hypothesis was that intermediate-sharded routed experts, two independent
NUMA-local CPU worker pools, and a concurrent rank-local shared expert would
reduce the equal-scope MoE dependent critical path. The primary metric was the
centralized-control versus rank-local-candidate wall time for the same actual
replay fixtures. The governing model was instantiated as

```text
tau = max_r(W_r / B_r) + Sigma_serial
```

The correctness gate required exact route IDs, coefficients, CPU partials,
shared/routed association, BF16 local join order, FP32 NCCL reduction, BF16
publication, non-finite closure, and injected pre/post-collective failure
closure on both ranks. The memory ceilings were 21,287,272,448 bytes per GPU
and 231,928,233,984 bytes (216 GiB) RSS. The rollback condition was any exactness
mismatch, fallback, failure to close, memory breach, timed I/O/allocation, or
candidate that did not materially beat control outside observed variance.

The bottleneck was measured before deciding whether to add a correction. The
candidate was not compared on a friendlier workload and no Stage 6 or CPU
arithmetic optimization was introduced.

## Replay capture

The v2 `.d4m` format records, for each fixture, layer and position; hidden input
`[4096]`; router logits `[256]`; six expert IDs; six coefficients; two routed
CPU rank partials `[2,4096]`; and centralized accepted output `[4096]`.

The final capture is under:

```text
results/dsv4-rank-local-moe/stage5-capture/
results/dsv4-rank-local-moe/stage5-capture/replays/
```

It contains 5,117 replay files, of which 645 are the decode scope (43 layers ×
15 positions). The generation and decode-list hashes are:

```text
generation.json          cfe6c738a581480f3dcab64998d1fe287cd24527916591de054e2a38e9e821e1
decode-replay-files.txt  5e69c8e9284bf795923865124aeb1575ac2933a1282ad297b9c341ec423b0c92
```

The capture used the legacy/diagnostic `--layer-hash-trace` path because it
exposes a host accepted-output boundary. The fused device-output path does not
expose that boundary; it remains fail-closed and was not used to fabricate an
oracle. The capture script states an estimated fixed setup of 188 seconds and
an 11.3-second decode window (16.6× setup/window); the shorter existing exact
capture was reused where possible rather than running another model arm.

## Candidate architecture

The implementation is a bounded fixture/probe, not a full-runtime integration.
It gives each rank:

- actual FP4 routed expert intermediate shards in NUMA-bound storage;
- a 24-thread persistent CPU pool (rank 0 on NUMA node 1, rank 1 on node 0);
- independent persistent CPU and GPU coordinators;
- fixed process-lifetime measured-path buffers;
- CPU gate/up, exact SwiGLU, down, and weighted reduction with the existing
  arithmetic and route coefficients;
- a concurrent FP8 shared expert;
- the Stage 3 exact BF16 local `bf16(bf16(cpu_rank) + bf16(shared_rank))` join;
- real NCCL FP32 data all-reduce and U32 MAX status all-reduce; and
- final BF16 publication, with failed output withheld and published as zeros.

The conservative bridge uses backend host-visible shared output: each rank's
shared result is D2H, then CPU/shared partials and status are H2D to the
standalone rank-local join. This is the measured candidate under review; no
device-resident lifecycle correction was added.

## Exactness and failure results

The probe covered actual-format layers 2, 21, and 42 at positions 104–118: 45
fixtures per arm. All 45 candidate fixtures passed:

```text
route IDs and coefficients                  exact
captured routed CPU rank partials            exact
centralized accepted output association      exact
rank-local BF16 join and rank order          exact
FP32 NCCL data reduction                     exact
U32 MAX status reduction                     exact
BF16 publication                             exact
injected failures                            4/4 closed
non-finite output                            0
callback failures                            0
```

The four failure cases cover pre- and post-collective injection on both ranks.
No failed output became externally visible.

## Binding performance result

The binding artifact is `results/dsv4-rank-local-moe/stage5-probe/`. It used
one warmup and three interleaved control/candidate repetitions over 45 actual
fixtures. The probe deliberately exits status 3 when its performance gate is
rejected; this status is expected and is not a correctness failure.

| Repetition | Centralized control (ms/layer) | Rank-local candidate (ms/layer) |
|---:|---:|---:|
| 0 | 12.384058 | 12.676090 |
| 1 | 12.254986 | 12.614193 |
| 2 | 12.279593 | 12.567971 |
| Median | **12.279593** | **12.614193** |
| Range | 12.254986–12.384058 | 12.567971–12.676090 |

Every candidate repetition is slower than every control repetition. The
candidate/control speedup is `0.973474`, a 2.73% regression. The projected
43-layer values are 528.022479 ms control and 542.410307 ms candidate, or a
14.387828 ms/forward regression. The later verification arm under
`stage5-probe-verification-20260811` also remained slower (candidate median
12.611297 versus control 12.244690 ms/layer) and is retained as corroboration,
not substituted for the binding arm.

The binding phase means are:

| Phase | Control | Candidate |
|---|---:|---:|
| wall | 12.306212 ms | 12.619418 ms |
| CPU routed wall | 12.032943 ms | 11.863160 ms |
| gate critical | 8.121749 ms | 8.048353 ms |
| down critical | 3.759346 ms | 3.672239 ms |
| weighted reduction critical | 0.157446 ms | 0.111836 ms |
| rank imbalance | 0.237842 ms | 0.212175 ms |
| GPU rank critical | 0 | 0.285123 ms |
| NCCL/join | 0 | 0.214988 ms |

The control residual `wall - CPU` is 0.273269 ms/layer. The candidate residual
is 0.756258 ms/layer, an additional 0.482989 ms/layer. The candidate's CPU body
is slightly shorter, but that reduction does not pay for the bridge and
coordinator work.

## Cost-model instantiation

The routed CPU payload is 53,477,376 bytes/layer for gate/up and 26,738,688
bytes/layer for down, or 80,216,064 bytes/layer including weighted reduction.
At the measured body times, the effective aggregate CPU rates are:

```text
control B_cpu = 80,216,064 / 0.012032943 = 6.665 GB/s
candidate B_cpu = 80,216,064 / 0.011863160 = 6.761 GB/s
```

The CPU routed body is therefore `argmax_r(W_r/B_r)` for both arms. The
isolated full shared expert median is 0.266851 ms/layer, with mean phase times
of 0.004805 ms input quantization, 0.109701 ms gate/up, 0.004900 ms activation
quantization, and 0.061711 ms down. Its projected 43-layer work is 11.474593
ms and it overlaps the CPU body; it is not the bottleneck. Candidate GPU rank
critical time is 0.285123 ms/layer, also non-binding.

The measured model is consequently:

```text
control:   tau = max(12.032943 ms CPU,
                     0.266851 ms shared GPU,
                     transfer/NCCL terms) + 0.273269 ms
                    = 12.306212 ms/layer mean wall

candidate: tau = max(11.863160 ms CPU,
                     0.285123 ms GPU rank,
                     transfer/NCCL terms) + 0.756258 ms
                    = 12.619418 ms/layer mean wall
```

The sign is negative on the candidate: it does not reduce the CPU `argmax`
and adds serial work. It does not turn the host bridge into a bandwidth claim:
the bridge is a handoff/overlap defect, and physical NCCL SHM bytes are not
instrumented. The optimistic, unmeasured ceiling obtained by removing the
entire additional residual is:

```text
max(11.863160 ms, ...) + 0.273269 ms = 12.136429 ms/layer
```

This is only a bound: it assumes a free device-resident correction, creates no
new synchronization resource, and does not show that the implementation would
beat control by a material amount. It cannot be used as an acceptance result.

## Upgraded Nsight decision

The older artifact `results/dsv4-rank-local-moe/stage5-profile/trace.nsys-rep`
was read with Nsight Systems 2026.1.3 and reported no CUDA kernel or NVTX data;
it cannot answer the phase question. A production-shaped replacement already
present at `stage5-profile-nsys20260811/trace.nsys-rep` was then read
read-only with:

```bash
nsys stats --force-export=true --sqlite=trace.sqlite \
  --report=cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,\
nvtx_sum,nvtx_pushpop_sum --format=column trace.nsys-rep
```

That trace covers the 45-fixture control/candidate scope with one warmup and
three repetitions and contains CUDA kernel, API, memory, synchronization, and
NVTX summaries. The key kernel totals are:

| Kernel family | Instances | Total |
|---|---:|---:|
| shared gate/up | 812 | 67,409,533 ns |
| shared down | 812 | 37,518,303 ns |
| NCCL FP32 sum | 458 | 18,007,573 ns |
| NCCL U32 status | 458 | 6,784,598 ns |
| local join | 458 | 748,000 ns |
| publication | 458 | 688,736 ns |

The local join and publication kernels are approximately 3.1 and 1.5
microseconds per fixture. The aggregate memory summary reports 2493 H2D
operations / 187,263,264 bytes / 31,436,195 ns and 2082 D2H operations /
17,058,992 bytes / 4,016,885 ns; those totals include setup and are not a
per-layer bandwidth measurement. CUDA API synchronization is dominated by
`cudaEventSynchronize` (812 calls, 3,057,503,796 ns aggregate), while setup
`cudaFuncGetAttributes` accounts for 676,358,981 ns across 156 calls. NVTX
contains NCCL group/all-reduce ranges but no hidden device-resident join.

Nsight Compute remains unavailable: the prior attempt returned
`ERR_NVGPUCTRPERM`, so this experiment reports no hardware counter values. No
long replacement profile was needed because the upgraded existing trace is
production-shaped enough to distinguish the candidate kernels, copies, and
serial API path.

The trace confirms that a host bridge exists, but it does not justify a new
device-output lifecycle at this Stage 5 boundary. The candidate's CPU body is
the measured bottleneck; the local join is microsecond-scale; and the measured
end-to-end result is a variance-clearing regression. A single narrowly scoped
device-resident correction was therefore considered and rejected. No second
candidate arm was built, so there are no post-hoc passing numbers to launder
the binding negative result.

## Memory, I/O, and transfer gates

The candidate's measured resource accounting is:

| Resource | Value | Gate |
|---|---:|---|
| process RSS | 3,036,856,320 B | below 231,928,233,984 B |
| projected rank VRAM | 10,384,097,480 B | below 21,287,272,448 B |
| fixture GPU usage | 552,730,624 / 477,233,152 B | observed, below ceiling |
| setup checkpoint reads | 25,810 calls / 2,369,842,184 B | setup only |
| measured checkpoint reads | 0 calls / 0 B | pass |
| measured CUDA workspace allocations | 0 calls / 0 B | pass |
| measured explicit host allocations | 0 | pass |
| control application H2D/D2H | 49,152 / 16,384 B per fixture | recorded |
| candidate application H2D/D2H | 98,312 / 49,152 B per fixture | recorded |
| candidate NCCL logical data/status | 32,768 / 8 B per fixture | recorded |
| NCCL physical SHM | not measured | not claimed zero |

The Stage 4 projection component is 10,371,440,068 B/rank and Stage 5's extra
resident projection is 12,657,412 B/rank. CPU routed payload over all 43 layers
is 3,449,290,752 bytes. There was no decode NVMe demand/prefetch, KV miss, or
promotion in the measured path.

## Rejected and preserved arms

The following raw evidence remains untouched and is not replaced by the
binding arm:

```text
results/dsv4-rank-local-moe/stage5-probe-negative-v1
results/dsv4-rank-local-moe/stage5-probe-negative-v2-pre-split
results/dsv4-rank-local-moe/stage5-probe-negative-v3-pre-persistent
results/dsv4-rank-local-moe/stage5-probe-negative-v4-prewarm-stats
results/dsv4-rank-local-moe/stage5-probe-negative-v5-pre-output-oracle
results/dsv4-rank-local-moe/stage5-probe-failed-stale-view
results/dsv4-rank-local-moe/stage5-probe-verification-20260811
results/dsv4-rank-local-moe/stage5-capture-routed-boundary-v1
results/dsv4-rank-local-moe/stage5-capture-failed-callback-io
results/dsv4-rank-local-moe/stage5-capture-failed-position-gate
results/dsv4-rank-local-moe/stage5-capture-failed-diagnostic-path
results/dsv4-rank-local-moe/stage5-capture-failed-fused-upstream
```

The stale-view arm was a pointer-lifetime defect after vector movement; the
current source rebinds raw CUDA expert pointers after storage stabilization.
The failed capture arms document callback I/O, position-gate, diagnostic-path,
and fused-upstream failures. None is a performance result.

## Files, commands, and disposition

Stage 5 source and test files are:

```text
CMakeLists.txt
include/strata/cuda_backend.hpp
kernels/cuda/backend.cu
src/deepseek_runtime.cpp
apps/strata_dsv4_rank_local_moe_probe.cu
include/strata/dsv4_moe_replay.hpp
src/dsv4_moe_replay.cpp
tests/test_dsv4_moe_replay.cpp
scripts/run_dsv4_stage5_capture.sh
scripts/run_dsv4_stage5_probe.sh
scripts/run_dsv4_stage5_profile.sh
```

The measurement scripts were:

```bash
scripts/run_dsv4_stage5_capture.sh
scripts/run_dsv4_stage5_probe.sh
scripts/run_dsv4_stage5_profile.sh
nsys stats --force-export=true --sqlite=results/dsv4-rank-local-moe/stage5-profile-nsys20260811/trace.sqlite \
  --report=cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,nvtx_sum,nvtx_pushpop_sum \
  --format=column results/dsv4-rank-local-moe/stage5-profile-nsys20260811/trace.nsys-rep
```

The focused replay round-trip test, relevant CUDA target rebuild, root
`make check`, and `git diff --check` are required before the single result
commit. The final validation transcript and commit hash belong to the review
handoff; the branch remains `exp/dsv4-rank-local-moe`.

Final disposition: **Stage 5 REJECTED / REVIEW REQUIRED**. The exactness,
memory, I/O, and failure gates pass, but the performance gate fails. Stage 6
is **BLOCKED and not authorized**. The next action requires human review of
this negative result; this experiment does not authorize building a correction,
full mHC chain, graph capture, CPU arithmetic optimization, or an end-to-end
benchmark.
