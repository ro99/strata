# Experiment 0001 — GLM-5.2 runtime observability

Status: **complete; observability accepted**.

## Hypothesis and contract

The existing aggregate runtime counters hide enough phase and device behavior
to make later placement experiments unsafe. T0 is instrumentation-only: model
precision, exact top-8 router semantics, placement, replacement, kernel math,
prefetch, and I/O policy remain unchanged.

- Primary metric: complete reconciliation of logical checkpoint reads, weight
  and activation transfers, cache events, allocations, synchronization, and
  timed phases.
- Correctness: `make check`, identical greedy token IDs, and a valid sequential
  route record for every routed layer and token position.
- Memory: process RSS at most 216 GiB and the unchanged 0.85 VRAM cache fraction.
- Rollback: any token divergence, accounting mismatch, hidden fallback, memory
  ceiling breach, or unexplained I/O rejects the instrumentation.

## Interfaces and metric definitions

`strata-run --route-trace PATH` writes JSON Lines. The first record declares
schema `strata.glm52.route_trace` version 1. Every following record contains:

- `request`: request sequence starting at the configured request base;
- `phase`: `prefill` or `decode`;
- `token_position` and `layer`: zero-based absolute token position and layer;
- `experts`: selected expert IDs in exact router order;
- `coefficients`: normalized routed coefficients written with enough decimal
  digits to round-trip the float32 values exactly.

The baseline script writes per-run traces to
`results/glm52-observability/.../run-NN.routes.jsonl`. These artifacts remain
ignored. Arrays in cache metrics follow the configured CUDA-device order.

Runtime JSON retains the previous aggregate fields and adds `phases.prefill`
and `phases.decode`. Each phase reports:

- logical checkpoint calls/bytes, summed read service seconds, and overlap-safe
  read wall seconds;
- weight and activation bytes, matmul calls, allocation calls/bytes,
  synchronization calls/waits, and per-device values;
- per-device cache hits, misses, evictions, occupancy, capacity, resident-spine
  bytes, and evictable-expert bytes;
- host expert-output aggregation wall time.

Detailed CUDA event timings are enabled only by `--detailed-timing`. Aggregate
CUDA durations are the maximum phase delta among devices (the concurrent-device
critical path), while each device retains its own service total. They are never
summed across devices. Byte and event counts do sum across devices.

## Reusable commands

```bash
make check

tmux new-session -d -s strata-glm52-t0-short \
  'RESULT_DIR=results/glm52-observability/short-v2 \
   REPETITIONS=1 MAX_NEW_TOKENS=2 TRACE_ROUTES=1 DETAILED_TIMING=1 \
   scripts/run_glm52_baseline.sh'

tmux new-session -d -s strata-glm52-t0-baseline \
  'RESULT_ROOT=results/glm52-observability/baseline-v2 ROUNDS=3 \
   scripts/run_glm52_observability.sh'
```

The baseline matrix alternates counters-only and full route/event
instrumentation in odd/even order for three repetitions of each mode.

## Short-run reconciliation

The 30-token prefill plus one decode step produced 2,326 JSONL records: one
schema record plus `75 routed layers * 31 positions`. JSON parsing passed, all
phase/device counters reconciled, and generated token IDs were `[16, 13]`.
Peak RSS was 2,268,856 KiB. Logical checkpoint reads were 156,064,247,808 bytes;
overlap-safe checkpoint read wall time was 49.568 seconds.

At the end of the short run, per-device resident-spine bytes in configured
device order were `[5,000,759,296, 6,552,244,224, 7,845,552,128]`; evictable
expert bytes were `[8,998,944,768, 14,708,441,088, 13,417,316,352]`.

## Full baseline results

The six runs were interleaved as counters/full, full/counters, counters/full.
Every run used the same model, prompt, precision, 0.85 VRAM fraction, and exact
128-token greedy workload.

| Run | Mode | Decode tok/s | Prefill tok/s | Init s | Physical read bytes | Physical write bytes | Peak RSS KiB |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | counters | 0.283925 | 0.685773 | 20.755 | 79,845,515,264 | 62,795,776 | 2,269,600 |
| 1 | full | 0.298241 | 0.659891 | 23.303 | 88,895,029,248 | 190,078,976 | 2,269,244 |
| 2 | full | 0.307441 | 0.630143 | 26.413 | 103,226,802,176 | 21,610,496 | 2,267,540 |
| 2 | counters | 0.306120 | 0.648044 | 24.004 | 71,984,971,776 | 29,745,152 | 2,268,880 |
| 3 | counters | 0.259472 | 0.539248 | 25.610 | 72,781,975,552 | 164,450,304 | 2,267,436 |
| 3 | full | 0.305400 | 0.636563 | 27.239 | 100,134,309,888 | 141,393,920 | 2,267,992 |

Counters-only median decode was 0.283925 tok/s; full route/event timing median
was 0.305400 tok/s. The computed full-mode overhead is -7.56%, but this is not a
speedup: counters-only results ranged from 0.259472 to 0.306120 tok/s, so the
difference is within observed run variance. Detailed event timing and route
output remain opt-in. Counters-only median prefill was 0.648044 tok/s and median
initialization was 24.004 seconds.

All six runs produced exactly 910,224,494,592 logical checkpoint-read bytes,
910,223,474,688 weight H2D bytes, 7,945,728,000 activation H2D bytes,
7,126,483,456 activation D2H bytes, 200,396 cache hits, 140,292 misses, and
134,570 evictions. Phase totals and device totals reconciled in every run.

One representative detailed run split those totals as follows:

| Phase | Checkpoint bytes | Weight H2D | Activation H2D / D2H | Cache hit / miss / eviction | Allocation / sync calls |
|---|---:|---:|---:|---:|---:|
| Prefill | 145,903,927,296 | 145,903,583,232 | 1,517,715,456 / 1,347,215,360 | 700 / 22,488 / 16,766 | 44,998 / 45,676 |
| Decode | 764,320,567,296 | 764,319,891,456 | 6,428,012,544 / 5,779,268,096 | 199,696 / 117,804 / 117,804 | 235,608 / 435,304 |

Its overlap-safe checkpoint read wall time was 30.302 seconds for prefill and
167.598 seconds for decode. Critical-device kernel time was 3.090 and 14.881
seconds; critical-device upload wait was 1.895 and 9.342 seconds; host expert
aggregation was 0.093 and 0.474 seconds. These concurrent-device critical-path
figures are not summed across GPUs.

Each full run emitted 11,776 valid JSONL records: one schema record plus
`75 routed layers * (30 prefill + 127 decode positions)`. All three trace files
have SHA-256
`275a420014969d23e57d977c399041731a6824dc634bdfb28c282954d91b6a8e`.
All six generated-token sequences and answers match each other and the retained
pre-T0 baseline exactly.

Maximum sampled framebuffer use was 13,530 MiB on CUDA device 0 and 20,598 MiB
on devices 1 and 2. Runtime-accounted peak weight-cache bytes were
`[13,999,704,064, 21,260,685,312, 21,262,868,480]`. Resident-spine bytes were
`[5,000,759,296, 6,552,244,224, 7,845,552,128]`; final evictable-expert bytes
were `[8,998,944,768, 14,708,441,088, 13,417,316,352]`.

Physical reads were not stable enough to compare modes: the counters median was
72,781,975,552 bytes and the full-mode median was 100,134,309,888 bytes despite
identical logical reads. This is reported as page-cache/storage variance, not an
instrumentation win or a runtime-policy result.

## Decision

Accept T0. Low-cost phase/device counters remain always enabled; detailed CUDA
events and sequential route output remain explicit opt-ins. No placement,
replacement, kernel math, prefetch, or I/O policy changed. The accounting,
token-identity, trace-completeness, and memory gates pass, so T1 may use the
frozen trace schema and this evidence.

Unresolved risks are the large physical-read variance, one-second framebuffer
sampling granularity, and the inability to infer detailed-timing overhead from
this noisy six-run matrix. Later performance stages must continue interleaving
runs and must not treat these infrastructure measurements as a speedup.

## Handoff record

Stage: T0

Status: complete

Branch and commit: `infra/glm52-observability`, implementation `a75ea85`

Hypothesis: aggregate counters hid phase and per-device behavior needed for safe
placement research; accepted.

Files changed: runtime/checkpoint/CUDA metric interfaces and implementations,
`strata-run` JSON/CLI, CUDA tests, baseline and observability scripts, this
experiment record, README handoff link, and the T0 ledger.

Commands and tmux session: `make check`; short reconciliation in
`strata-glm52-t0-short-v2`; interleaved matrix in
`strata-glm52-t0-baseline` using `scripts/run_glm52_observability.sh`.

Ignored artifacts: `results/glm52-observability/short-v2/`,
`results/glm52-observability/baseline-v2/`,
`results/glm52-t0-short-v2-session.log`, and
`results/glm52-t0-baseline-v2-session.log`.

Correctness results: `make check` passed; six of six full-baseline token
sequences and answers are identical to the retained baseline; all traces parse
and have identical hashes; all phase/device reconciliations pass.

Run results and median: counters decode `[0.283925, 0.306120, 0.259472]`, median
0.283925 tok/s; full decode `[0.298241, 0.307441, 0.305400]`, median 0.305400
tok/s. The apparent -7.56% overhead is within variance and is not a win.

Peak RSS and per-GPU VRAM: maximum RSS 2,269,600 KiB; maximum sampled FB memory
13,530 / 20,598 / 20,598 MiB for CUDA devices 0 / 1 / 2.

NVMe, H2D/D2H, cache, allocation, and synchronization counters: per-run logical
checkpoint 910,224,494,592 bytes; weight H2D 910,223,474,688 bytes; activation
H2D/D2H 7,945,728,000 / 7,126,483,456 bytes; cache hit/miss/eviction
200,396 / 140,292 / 134,570. Representative phase allocation/sync calls are
44,998 / 45,676 prefill and 235,608 / 435,304 decode. Physical-read medians are
72,781,975,552 counters and 100,134,309,888 full; the difference is not claimed.

Decision: accept instrumentation; keep detailed timing and route files opt-in;
freeze schema version 1 and the metric definitions above.

Unresolved risks: physical-I/O variance, one-second VRAM sampling, and noisy
overhead attribution.

Next stage ready: yes
