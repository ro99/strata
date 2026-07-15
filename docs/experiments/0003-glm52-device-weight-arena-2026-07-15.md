# Experiment 0003 — GLM-5.2 persistent device weight arena

Status: **complete; rejected on performance evidence**.

## Hypothesis and contract

Replacing per-miss routed-weight `cudaMalloc` and per-eviction `cudaFree` with
fixed slots in one persistent arena per GPU, while replacing the full-cache LRU
victim scan with an O(1) list, might remove enough unmeasured allocator and
cache-management time to exceed 0.40 decode tok/s.

- Primary metric: median decode tok/s from three interleaved current/arena
  repetitions; the predeclared decisive threshold was 0.40 tok/s.
- Correctness: `make check`, identical greedy tokens, and identical checkpoint,
  H2D, and cache-event counts.
- Memory: unchanged `0.85` VRAM cache fraction and at most 216 GiB RSS.
- Rollback: reject on any semantic/accounting/memory failure, or when median
  throughput remains below 0.40 tok/s or inside observed variance.

The branch is `exp/glm52-device-weight-arena`, based on accepted T0/T1 runtime
commit `b8321bb`. Implementation commit `fed3dd7` is opt-in through
`--persistent-weight-arena`.

## Implementation

After the resident spine loads, the candidate allocates one device arena from
the remaining declared weight-cache capacity on each GPU. Every routed linear
uses an equal-size slot. A cache entry holds a non-owning slot reference; its
destructor returns the slot to an O(1) free list rather than calling
`cudaFree`. Candidate-mode LRU hits use list splice and eviction uses the front
entry rather than scanning the unordered map.

Baseline mode retains the original allocation and victim-scan behavior, so the
same binary provides an attributable A/B comparison. Placement, replacement
order, reads, uploads, kernels, routing, and numerical aggregation are
unchanged.

## Correctness and smoke

`make check` passed 2/2. The CUDA test covers arena exhaustion and slot reuse.
The two-token smoke generated exact IDs `[16,13]`, retained established
checkpoint/H2D/cache counts, and reported zero decode routed-weight allocation
calls in arena mode.

## Full interleaved result

The matrix used three rounds in order current/arena, arena/current,
current/arena. Every run used the pinned checkpoint, exact greedy graph, 30
prompt tokens, 127 decode steps, all three GPUs, detailed timing, and VRAM
fraction `0.85`.

| Round | Current tok/s | Arena tok/s | Current decode s | Arena decode s | Current checkpoint wall s | Arena checkpoint wall s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.313244 | 0.313688 | 405.44 | 404.86 | 154.89 | 171.96 |
| 2 | 0.320010 | 0.359391 | 396.86 | 353.38 | 148.50 | 145.08 |
| 3 | 0.315945 | 0.284696 | 401.97 | 446.09 | 150.98 | 179.65 |
| **Median** | **0.315945** | **0.313688** | **401.97** | **404.86** | **150.98** | **171.96** |

Arena throughput ranged from 0.284696 to 0.359391 tok/s. The median is 0.71%
below current and no run reached the predeclared 0.40 tok/s threshold. The
result is within large storage/page-cache variance and is not a win.

## Accounting

All six runs produced one unique generated-token sequence. In every paired
run, logical checkpoint bytes, weight H2D bytes, and cache hit/miss/eviction
counts were identical. Decode routed-weight allocation calls changed from
235,608 current to zero arena. Peak RSS remained approximately 2.17 GiB and
the declared VRAM cache capacities were unchanged.

CUDA critical-path kernel time stayed near 14.6–15.1 seconds per run. Current
checkpoint-read wall time ranged from 148.50 to 154.89 seconds; arena ranged
from 145.08 to 179.65 seconds and tracked most of its throughput variance.

Ignored artifacts are under `results/glm52-weight-arena/full-v1/`, with the
aggregate decision in `comparison.json`. A short Nsight Systems capture under
`results/glm52-weight-arena/profile/` did not collect CUDA trace tables on this
driver/tool combination; its OS-runtime table is diagnostic only and is not a
performance decision input.

## Decision

Reject the runtime behavior and do not merge `fed3dd7` into `main`. The
experiment proves that the allocation-call count was not a valid proxy for
critical-path allocator cost. The T1 bulk `cudaMalloc`/`cudaFree` microbenchmark
cannot be multiplied by hot-path call count to predict runtime savings.

The remaining structural evidence is that decode still performs 317,500 CUDA
matmuls and 435,304 synchronizations per run—2,500 matmuls and 3,428
synchronizations per token—while round-tripping about 96 MB of activations per
token through host memory. A future optimization must measure and reduce this
host-synchronous graph rather than infer a win from allocation or prediction
counters.

## Handoff record

Stage: A0

Status: rejected

Base revision: `b8321bb7cb551bd9f2d9f0e317360f3385a6cff9`

Branch and commit: `exp/glm52-device-weight-arena`, implementation `fed3dd7`

Commands and tmux session: `make check`; smoke under
`results/glm52-weight-arena/smoke/`; three-round matrix in tmux session
`strata-glm52-weight-arena` using `scripts/run_glm52_weight_arena.sh`.

Correctness results: 2/2 tests passed; identical tokens, checkpoint bytes, H2D
bytes, and cache counts; zero arena-mode decode weight allocations.

Decision: reject and preserve the branch for audit; no child experiment is
authorized by A0.
