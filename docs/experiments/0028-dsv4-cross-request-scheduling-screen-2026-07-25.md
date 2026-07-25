# Experiment 0028 — cross-request DeepSeek scheduling screen

Status: **rejected before implementation**. The proposed batching does not
reduce the measured bottleneck, and its two execution prerequisites did not
clear their own performance gates.

## Contract

- Hypothesis: a bounded layer-major wavefront over independent decode rows
  reduces routed-weight demand H2D through expert union and improves aggregate
  decode throughput without changing exact per-request state.
- Primary metric: projected demand-H2D bytes per exact decode token at batch
  sizes 2, 4, and 8. Latency, queue wait, RSS, VRAM, and per-request KV bytes are
  regression constraints.
- Correctness gate: identical per-request tokens, routes, coefficients, logits,
  layer hashes, stop/error handling, and isolated block-KV sequence handles.
- Memory ceiling: the existing 216 GiB host admission, 0.85 VRAM fraction,
  configured host/device KV budgets, and zero steady-state NVMe reads.
- Rollback: stop before runtime work if the replay does not materially reduce
  the measured `argmax` term.

## Bottleneck and target term

The current 511-token-prompt operating point is the pinned run at
`results/deepseek-v4-pinned-arena-ab/pinned/run-01`: 127 decode steps in
40.351 seconds, or 317.7 ms/step. Experiment 0026 instantiated the phase cost
model at this point:

| term | ms/step |
|---|---:|
| serial routed-weight demand H2D | **86.25** |
| attention score | 51.97 |
| MoE compute | 46.92 |
| mHC pre | 37.45 |
| attention query projection | 26.96 |
| attention output projection | 24.64 |

Demand H2D is `argmax_r`. Cross-request aggregation must therefore reduce its
volume or overlap its serial wait. This screen tests the volume claim before a
queue, scheduler, or runtime API is built. Explicit H2D overlap is roadmap stage
A4.5, after the ticket wavefront, and is not supplied by issue 16's batching
mechanism.

## Cheap replay

No new model load was needed: the existing route trace and metrics are from the
same model, hardware, precision, cache bounds, and operating point. Re-running
several full-model arms would spend about 51 seconds per arm on fixed setup and
about 40 seconds on the relevant decode window; replaying the exact trace takes
seconds and directly tests the proposed cache-volume mechanism.

The replay uses the runtime's capacity-weighted expert schedule
`[0,0,1,1,1,2,2,2]`, the measured per-device cache capacities minus pinned
bytes, and the admitted 13,369,344-byte atomic expert placement. Those bounds
hold 842, 1,242, and 1,344 routed experts on devices 0, 1, and 2. Prefill warms
both arms identically. The baseline then visits each decode row through all 43
layers; the candidate visits all rows in a bounded batch at each layer.

As a validation, batch 1 predicts 5,524 atomic expert misses. Each expert has
three projections, so that is exactly the run's measured 16,572 cache misses
and 73,852,256,256 demand-H2D bytes.

Adjacent decode rows are a favorable input for expert union because their
routes are more correlated than unrelated requests:

| batch | mean union | expert rows | apparent union reuse |
|---:|---:|---:|---:|
| 2 | 9.95 | 12 | 17.1% |
| 4 | 16.21 | 24 | 32.4% |
| 8 | 26.20 | 48 | 45.4% |
| 16 | 41.70 | 96 | 56.6% |

The shared LRU already retains that reuse across ordinary row-at-a-time decode:

| batch | tokens replayed | baseline misses | layer-batch misses | H2D change |
|---:|---:|---:|---:|---:|
| 1 | 127 | 5,524 | 5,524 | 0.00% |
| 2 | 126 | 5,433 | 5,436 | **+0.06%** |
| 4 | 124 | 5,361 | 5,379 | **+0.34%** |
| 8 | 120 | 5,212 | 5,231 | **+0.36%** |
| 16 | 112 | 5,085 | 5,228 | **+2.81%** |

Layer-major ordering slightly worsens cache replacement. It exposes no routed
weight reuse that the current cache was missing.

## Sign on other resources

- GPU launches and synchronization can fall, but they are not the measured
  bottleneck. The accepted grouped-expert prerequisite cut MoE launches 97.9%
  and MoE execution 21.3% while remaining end-to-end neutral.
- Activation H2D increases: that prerequisite measured +7.7% routed MoE H2D.
- Scheduler bookkeeping, ticket descriptors, deadline checks, and queue wait
  add CPU work and latency.
- Every admitted request adds exact KV/index state and block-table metadata;
  batching cannot share mutable state.
- NVMe remains zero only if every request continues to pass the existing
  admission contract.

The separate device-activation prerequisite was rejected for promotion at
0.9793x median decode throughput. Neither performance prerequisite is present
on `main`; only the block-KV manager is.

## Decision

Do not implement issue 16 at this operating point. Its expert aggregation does
not reduce the current bottleneck and both missing prerequisite mechanisms are
neutral or rejected. Building the request queue and multi-sequence graph before
that gate would violate the repository's binding dependency and cost-model
rules.

Reopen the mechanism only after a cheap measured prototype demonstrates one of:

1. demand-H2D overlap across ready rows that materially reduces the 86.25
   ms/step serial term;
2. a measured operating point where layer batching reduces exact demand-H2D
   bytes versus the shared-LRU baseline; or
3. a promoted device-resident grouped executor whose end-to-end gain exceeds
   observed variance before scheduler overhead.
