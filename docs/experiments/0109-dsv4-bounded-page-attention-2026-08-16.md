# Experiment 0109 — bounded DSV4 page-attention chunks

Status: **rejected on the functional gate.** The bounded row planner passes its
bit-exact fixture and does not split the 677-token prefill page, but the
2,612-token arm still does not complete. It now reports a KV host-cache
capacity failure before emitting model JSON.

## Predecessor and defect

Experiment 0108 measured landed commit `54505ba`. Its 677-token arm completed,
but the 2,612-token arm stopped safely after approximately 113 seconds with
`DeepSeek attention-to-mHC workspace exceeds its bounded contract`. The landed
page-attention path submitted every row in the scheduling page as one CUDA
command while fixing that command's workspace ceiling at 384 MiB. Workspace
grows with query rows, so the longer page could not be admitted.

The pre-merge per-row path had completed this operating point. This is therefore
a functional regression in the landed row-batched path, not a throughput
hypothesis.

## Pre-change contract

- Hypothesis: splitting only the query-row dimension at the exact 384 MiB
  workspace boundary restores long-prompt admission without changing any row's
  gather, score, output projection, or mHC slot arithmetic.
- Primary metric: the 2,612-token, page-8192, untraced arm completes and emits
  its model JSON.
- Correctness gate: forced split versus unsplit output is bit-exact; the
  677-token generated IDs remain `2107, 8777, 1277, 440`; decode checkpoint
  reads remain zero; decode is unregressed; `make check` passes.
- Memory ceiling: the existing 384 MiB per-device attention workspace cap is
  binding and unchanged. No second workspace or persistent allocation is
  added.
- Rollback/stop condition: preserve and report if the forced boundary differs
  by one bit, if 2,612 tokens still fails admission, or if the 677-token path
  splits or regresses outside observed variance.

The reduced resource is peak source-device attention workspace per command.
The expected negative signs are additional host submissions, kernel launches,
synchronizations, and repeated reads/materialization of the shared KV pages for
each sub-chunk. Weight precision/residency, router semantics, expert count,
top-k, KV contents, and mHC slots are unchanged.

## Reference shape

The implementation follows `split_indexer_prefill_chunks` in the local
reference stack's `vllm/v1/attention/backends/mla/indexer.py`: host admission is
against a bounded workspace and an oversized request is sub-chunked on the
query dimension. Strata keeps its scheduling page intact and applies the
workspace bound only to contiguous row slices.

## Implementation and fixture

The CUDA backend now exposes the exact multi-row physical-page
attention-to-mHC workspace calculation and uses that same layout object during
execution. Runtime admission binary-searches the largest row extent under
384 MiB. Every command shares the page descriptors and output weights while
queries, candidates, RoPE vectors, mHC slots, and diagnostic output are sliced
by row. A possible singleton tail is rebalanced into the preceding chunk
because the backend's one-row command intentionally has a different mHC state
contract.

The existing CUDA bit-exact page fixture now sets a deliberately small
attention-to-mHC workspace bound, verifies that the full row extent exceeds
it, obtains the admitted row count from the production planner, and compares
all split output bits against unsplit per-row truth. The focused fixture passed.
`make check` passed with 277 tests passed and 34 fixture-dependent skips.

## Arm budget

After the implementation checkpoint, run exactly two untraced page-8192 arms
from the same binary and devices 1,2: one 677-token arm (estimated four minutes)
and one 2,612-token arm. The landed implementation has no successful long-arm
timing; estimate 8–15 minutes from the reduced dispatch path, with a total
planned model window of approximately 12–19 minutes. Preserve actual wall
times. No repetitions, baseline variant, or additional prompt length are in
scope.

## Results

### 677-token no-regression arm

The arm completed successfully. `/usr/bin/time` measured 2:50.24 total wall
time, including 105.12 seconds of initialization. Raw captures are under
`results/dsv4-0109-workspace-recovery/677-w420/`-named files.

| Metric | 0108 landed | 0109 bounded | Difference |
|---|---:|---:|---:|
| Prompt tokens | 677 | 677 | 0 |
| Total prefill | 53.793264 s | 55.705522 s | +1.912258 s |
| Prefill throughput | 12.5852 tok/s | 12.1532 tok/s | -3.43% |
| Attention | 22.377361 s | 23.256031 s | +0.878670 s |
| Query | 3.764747 s | 3.896011 s | +0.131264 s |
| KV | 3.915576 s | 4.061156 s | +0.145580 s |
| Score | 14.280580 s | 14.913645 s | +0.633065 s |
| MoE | 27.145546 s | 28.015122 s | +0.869576 s |
| mHC | 2.741972 s | 2.883348 s | +0.141376 s |
| Query matmul device kernel | 0.335592 s | 0.337143 s | +0.001551 s |
| Prefill paged-attention calls | 86 | 86 | 0 |
| Prefill kernel launches | 1,655 | 1,655 | 0 |
| Prefill page bytes | 30,564,224 B | 30,564,224 B | 0 |
| Expert demand H2D bytes | 74,380,770,816 B | 74,380,770,816 B | 0 |
| Expert demand wait | 17.217182 s | 18.017977 s | +0.800795 s |
| Demand bytes / wait | 4.320 GB/s | 4.128 GB/s | -4.44% |
| Demand bytes / MoE | 2.740 GB/s | 2.655 GB/s | -3.10% |
| Cache hits / misses / evictions | 85,230 / 10,512 / 7,888 | 85,230 / 10,512 / 7,888 | 0 / 0 / 0 |
| Decode checkpoint reads | 0 B | 0 B | 0 |
| RSS | 158,772,056,064 B | 158,858,911,744 B | +86,855,680 B |
| GPU 1 VRAM | 22,941,925,376 B | 22,971,285,504 B | +29,360,128 B |
| GPU 2 VRAM | 22,864,330,752 B | 22,895,788,032 B | +31,457,280 B |

The arm generated exactly `2107, 8777, 1277, 440`, and its four generated
tokens over 0.457407 seconds give 8.745 tok/s. That is 1.7% below 0107's lowest
candidate rate of 8.897 tok/s. Three measured decode steps are too short to
separate that gap from run variance, so the timing check is inconclusive rather
than rounded into a pass. The changed admission API is called only by the
multi-row page path and is unreachable for single-row decode; generated IDs
and checkpoint-read bytes pass exactly.

The 1.912-second total-prefill increase is inside 0107's measured 2.628-second
candidate range. More importantly for this change, all three prefill attention
structure counters are bit-for-bit unchanged, the query device term differs by
0.46%, and expert bytes/cache events are identical. No 677-token sub-chunk was
introduced.

An initial reading incorrectly compared the global CUDA call counters against
0108's prefill-only totals. The global 0109 value was 344 calls: 86 prefill plus
258 from three decode steps over 43 layers and two ranks. Launches reconcile
identically: 1,655 prefill plus `258 * 19` decode launches equals 6,557. The
long arm was stopped conservatively on that apparent contradiction. Once the
phase-scoped counters explained it exactly, the already-authorized long arm was
restarted under a new filename; all interrupted captures were preserved.

### 2,612-token functional arm

The preserved first attempt was interrupted after 55.73 seconds during model
loading for the counter-scope check above. The retry ran for 3:15.13 wall time
and exited nonzero with:

`DeepSeek KV host cache capacity is exhausted`

It emitted no model JSON. Therefore prompt-token confirmation, phase buckets,
expert demand H2D bytes, cache events, decode counters, and VRAM-at-completion
do not exist for this arm. Peak process RSS from `/usr/bin/time` was
159,786,729,472 bytes. The error differs from 0108's bounded attention
workspace rejection, but this run cannot establish that the attention command
itself was reached before the new failure. The KV capacity failure is recorded
as an open blocker rather than attributed to the row planner without evidence.

## Verdict

The forced-boundary correctness fixture and the 677 no-regression checks pass,
but the primary functional gate does not: 2,612 tokens still fails before
producing output. Experiment 0109 is therefore not a completed regression fix.
The implementation and every raw capture remain preserved on the experiment
branch. No workspace cap was raised, and no second capacity mechanism was
attempted after the gate fired.
