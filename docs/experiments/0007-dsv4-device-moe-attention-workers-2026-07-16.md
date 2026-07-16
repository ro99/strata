# Experiment 0007 — device MoE with exact host-attention workers

Status: **screened experimental baseline; not yet eligible for `main`**.

## Hypothesis and contract

After removing the device FP4 table-load penalty and grouping DeepSeek MoE on
device, the two head-independent host-attention regions are again a material
part of exact decode. Dispatching the 64 heads across 28 persistent,
affinity-pinned workers should materially improve throughput while preserving
the arithmetic order within every head.

- Primary metric: median exact decode steps/s over at least three interleaved
  128-token `Hello` repetitions. The working baseline is 2.9757502295 steps/s;
  the project objective remains greater than 5.0 steps/s.
- Correctness: `make check`; identical greedy token IDs, sequential routes,
  top-logit traces, and per-layer BF16 hashes; unchanged device-MoE command and
  cache accounting; zero decode checkpoint reads.
- Memory: at most 216 GiB host RSS and the same 0.85 per-device VRAM fraction.
- Rollback: reject if correctness, admission, exactness, or zero-NVMe gates
  fail, or if throughput does not improve materially beyond observed variance.

The branch is `exp/dsv4-device-moe-attention-workers`, based on FP4/device-MoE
revision `3742f1a`. The serial path remains the default and the worker count is
explicit; no precision, routing, top-k, scoring, or accumulation order changes.

## Projection

The earlier isolated host-attention screen reduced the pre-device-MoE decode
from 87.3346 to 73.3371 seconds, a roughly 14-second single-run saving. Applying
that measured saving to the current 42.6783-second device-MoE/FP4 screen
projects approximately 28.68 seconds for 127 decode steps, or 4.43 steps/s.

That projection is material but does not itself clear 5.0 steps/s. The screen
is still justified because it isolates the remaining exact host-attention cost
and establishes whether a subsequent device-resident attention projection
boundary has enough remaining value.

## Design

Only the independent head loops are parallelized:

1. per-head query RMS normalization, BF16 rounding, and RoPE;
2. per-head sliding/compressed score calculation, softmax, value accumulation,
   BF16 rounding, and inverse RoPE.

Each head retains its original scalar traversal and floating-point operation
order. Heads write disjoint query, score, and attended-output ranges. The
serial path retains the existing loops when the configured worker count is
zero. Persistent workers use the repository's existing host worker pool and
physical-core affinity policy; no per-layer threads are created.

The existing DeepSeek A/B harness is extended without changing its defaults so
both variants can require device MoE while independently declaring exact host
attention worker counts. The harness validates the reported modes rather than
inferring them from variant names.

## Pre-benchmark validation

- `make check`: 2/2 CTest targets passed.
- The harness script passes `bash -n` validation and preserves its original
  reference-serial/candidate-device-MoE defaults.
- A full-model two-token diagnostic smoke completed in tmux session
  `dsv4-attn-workers-correctness`. Serial/device-MoE and
  28-worker/device-MoE execution produced identical greedy tokens, sequential
  routes, logits, and every per-layer BF16 hash.
- Both diagnostic variants read zero checkpoint bytes during decode, completed
  exact execution, stayed within the host-memory ceiling, and finished with
  balanced cache leases and no active leases.

The ignored correctness artifact is
`results/deepseek-v4-device-moe-attention-workers-correctness/summary.json`.
The one traced decode step is not a performance measurement: dispatch overhead
made the worker variant slower at that tiny workload.

## Performance screen

One interleaved 128-token reference/candidate pair completed in tmux session
`dsv4-attn-workers-screen`. Both variants used the FP4 switch and exact device
MoE; only the explicit host-attention worker count changed from zero to 28.

| Metric | Serial attention | 28 workers |
|---|---:|---:|
| Decode seconds | 42.01521735 | **29.00231959** |
| Decode steps | 127 | 127 |
| Decode steps/s | 3.0227143404 | **4.3789600899** |
| Matched speedup | 1.000x | **1.4486847240x** |
| Prefill seconds | 5.388675491 | 5.114124290 |
| Decode activation H2D bytes | 1,605,947,392 | 1,605,947,392 |
| Decode activation D2H bytes | 1,747,585,024 | 1,747,585,024 |
| Decode synchronization calls | 62,285 | 62,285 |
| Maximum resident set, KiB | 146,298,332 | 146,297,372 |

The artifact is
`results/deepseek-v4-device-moe-attention-workers-screen/summary.json`.
Tokens and sequential routes were identical, decode checkpoint reads were
zero, cache leases were balanced with none active at completion, and both runs
stayed within the memory ceiling. The only failed acceptance gate was the
5.0 steps/s objective.

This is a material single-run result and raises the working screen baseline
from 2.9757502295 to 4.3789600899 steps/s. It is not yet a three-repetition
validated median.

The follow-up exact mHC-worker screen on branch
`exp/dsv4-device-moe-mhc-workers` (negative-result revision `1166175`) reached
4.3299886195 steps/s versus its matched 4.3326939996 reference, a
`0.9993755894x` ratio. That path was rejected without thread-count tuning or a
three-run matrix. The attention-worker revision `c1dada2` therefore remains the
best working screen baseline, and host micro-experiments stop here.

## Main-branch promotion status

The implementation is suitable for continued experimental use but is not yet
mature enough for the repository's validated `main` branch. The exact
serial/parallel comparison proves that this optimization preserves the current
Strata executor; it does not prove that the executor matches the supplied
target implementation. The outstanding promotion gates already declared in
`docs/deepseek-v4-runtime.md` and `docs/model-bringup.md` are:

- frozen target-executor outputs at operation and layer boundaries;
- full-model teacher-forcing agreement;
- greedy full-generation agreement;
- independent physical-I/O confirmation of the zero-NVMe claim; and
- a median of at least three interleaved 128-token repetitions before treating
  4.3789600899 steps/s as a validated throughput result.

Until those gates pass, this branch should remain an experimental checkpoint
and its performance number should be described as a screen, not a validated
research win. The next performance change, after correctness promotion, should
be structural: reduce host/device boundaries with a larger device-resident
attention/mHC block rather than resume small host-loop experiments.
