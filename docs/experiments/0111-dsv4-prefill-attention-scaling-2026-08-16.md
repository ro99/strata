# Experiment 0111 — DSV4 prefill attention scaling

Status: **implementation correctness gate passed; model arms pending.**

## Predecessor and measured target

Experiment 0110 produced the first landed two-point curve with the KV host
budget made explicit. Attention rose from 22.803820 s at 677 tokens to
96.499184 s at 2,612 tokens, a 38.0868 ms/token marginal term. MoE demand H2D
rose only 23.9% while prompt tokens rose 286%, so expert volume is not the
scaling defect at this operating point.

The 2,612-token attention attribution before this change was:

- query 14.449289 s, including **8.801146 s query allocation**;
- KV 14.911487 s, including 0.120044 s KV allocation;
- score 61.008242 s;
- maximum device paged attention 15.736416 s;
- total critical-path synchronization 39.420888 s.

The query path allocated and value-initialized a 2,611 x 32,768 float output
for every layer. `linear_rows` then overwrote its complete extent. The repeated
zero-fill is the directly measured first target.

## Pre-change contract

- Hypothesis: persistent max-seen, uninitialized page-projection scratch
  removes the 8.801 s allocation term without changing any projection,
  normalization, RoPE, KV append, attention, or mHC arithmetic.
- Primary metric: `attention_seconds` at 2,612 tokens, baseline 96.499184 s.
- Diagnostic metric: `attention_query_allocation_seconds` must fall near zero;
  recompute the attention marginal term from 677 and 2,612 tokens.
- Correctness: projection output is fully overwritten; generated IDs are
  `2107, 8777, 1277, 440` at both lengths; the existing forced attention
  sub-chunk fixture remains bit-exact; decode checkpoint reads remain zero;
  decode is unregressed; `make check` passes.
- Memory ceiling: scratch capacity grows only to the largest admitted page and
  replaces same-sized transient projection buffers. At 2,612 tokens the three
  retained extents total about 358 MiB; report RSS and do not add a duplicate
  initialized copy.
- Stop condition: any unwritten output, generated-ID mismatch, checkpoint
  read, decode regression, non-negligible allocation residual, or attention
  regression triggers preserve-and-report before another mechanism.

The reduced resource is host memory write bandwidth and allocator work. The
adverse sign is longer host residency of the largest query-rank/query/KV
scratch. Device compute, H2D/D2H volume, expert residency, and the bounded
384 MiB device attention workspace should not change.

## Implementation and correctness fixture

`DeepSeekV4Runtime::Impl` now owns three uninitialized float scratch buffers
for page query-rank, full query, and KV projection output. Each grows only when
an admitted page exceeds its previous capacity and is reused by subsequent
layers. C++20 `make_unique_for_overwrite` avoids scalar value-initialization.

The production-shape SM86 FP8 projection fixture now poisons both output paths
with NaNs before dispatch and requires every output element to be finite after
the projection, in addition to its existing BF16-boundary oracle checks. This
gates the overwrite premise directly. The fixture passed, as did the existing
forced sub-chunk bit-exact fixture. `make check` passed.

## Score attribution to report after the curve

No new instrumentation is added. Existing counters split the 0110 long score
bucket into paged-attention device time, host remainder, stream wait,
learned-index work, candidate resolution, page-set build and weight acquire.
The raw page-byte counter is also checked against source: every command launches
`dsv4_materialize_physical_pages` over every supplied device page, so repeated
sub-chunks cause real device global-memory reads rather than lease-only logical
accounting. Whether that traffic is material relative to measured bandwidth is
decided from the bytes and device time, not assumed.

`moe_prepare_seconds` was 28.08 s of the 64.63 s long-prompt MoE bucket in
0110. It is recorded as an open lead and is not investigated here.

## Arm budget

Run exactly one untraced 677-token arm and one untraced 2,612-token arm from the
same build, page8192, context8192, devices1,2, `--kv-host-cache 4G`. Expected
wall times are about three and five minutes, roughly eight minutes total. No
repetitions or other mechanisms are in scope before the result gate.

## Results

Pending.
