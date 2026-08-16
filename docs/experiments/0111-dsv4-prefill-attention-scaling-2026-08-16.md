# Experiment 0111 — DSV4 prefill attention scaling

Status: **partial pass; allocation removed and long attention improved 4.1%,
with the next score bottleneck attributed but not changed.**

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

Both arms used commit `a6d284d44db75f44d6b263f10135c4cebf630779`
and binary SHA-256
`aa9f2420993663e9ce9e6cbbde51dd1647d176331950bfb876934f3d4316e10c`.
They were untraced, page 8192, context 8192, devices 1,2, and used an
explicit 4 GiB host KV budget. Wall time including initialization was 2:44.77
at 677 tokens and 5:03.33 at 2,612 tokens, against the three- and five-minute
estimates.

| metric | 677 tokens | 2,612 tokens |
|---|---:|---:|
| prefill seconds | 52.768807 | 192.905603 |
| prefill tok/s | 12.8295 | 13.5403 |
| attention seconds | 20.615734 | 92.520060 |
| query seconds | 1.638891 | 9.379094 |
| KV seconds | 4.043504 | 14.002381 |
| score seconds | 14.731123 | 63.957948 |
| MoE seconds | 27.703056 | 83.612777 |
| mHC-post seconds | 2.821239 | 10.791964 |
| query allocation seconds | **0.000236** | **0.000274** |
| KV allocation seconds | 0.000356 | 0.000081 |
| query matmul device seconds | 0.334752 | 1.130366 |
| paged-attention calls / launches | 86 / 1,655 | 236 / 4,579 |
| paged-attention page bytes | 30,564,224 | 383,225,472 |
| expert demand bytes | 74,380,770,816 | 92,121,114,624 |
| expert demand wait seconds | 17.684029 | 41.880322 |
| effective GB/s, bytes / demand wait | 4.206 | 2.200 |
| effective GB/s, bytes / MoE seconds | 2.685 | 1.102 |
| cache hits / misses / evictions | 85,230 / 10,512 / 7,888 | 460,413 / 13,034 / 10,395 |
| decode tok/s | 8.7953 | 8.1307 |
| decode checkpoint read bytes | 0 | 0 |
| RSS bytes | 158,955,638,784 | 159,666,851,840 |
| GPU 1 / GPU 2 VRAM bytes | 22,971,285,504 / 22,895,788,032 | 23,808,049,152 / 23,524,933,632 |

Both arms generated `2107, 8777, 1277, 440`; both exited zero and decode
checkpoint reads remained zero. Relative to 0110, decode changed from 8.9918
to 8.7953 tok/s at 677 (-2.2%) and from 8.0643 to 8.1307 tok/s at 2,612
(+0.8%). With one run per point this is mixed rather than evidence of a decode
regression. VRAM is byte-for-byte unchanged from 0110. Retaining the largest
host scratch raised long-arm RSS by 351,084,544 bytes (334.8 MiB), inside the
declared approximately 358 MiB ceiling; the short RSS change was 96,849,920
bytes.

The diagnostic passed decisively: query allocation fell from 2.400819 to
0.000236 s at 677 and from 8.801146 to 0.000274 s at 2,612. Long query fell
from 14.449289 to 9.379094 s, a 5.070194 s reduction. The saved 8.800872 s did
not transfer one-for-one because query D2H rose from 2.377787 to 6.308950 s.
The old value initialization first-touched the transient output pages before
CUDA wrote them, whereas the uninitialized allocation first presents untouched
pages to D2H; that is the source-level adverse sign expected from this change.
The 3.931162 s D2H increase sizes the offset but was not separately
first-touch-countered, so this explanation remains falsifiable rather than a
new measurement. The buffer is then reused, and the net query result remains
positive. KV allocation also fell from 0.120044 to 0.000081 s, and long KV fell
by 0.909106 s.

The primary value result is a partial pass. Long attention fell from
96.499184 to 92.520060 s, a 3.979124 s or 4.1% reduction. The two-point
attention marginal fell from 38.085459 to **37.159859 ms/token**, a
0.925601 ms/token improvement. At 677, attention fell from 22.803820 to
20.615734 s. Total-prefill values carry no verdict in this single pair:
expert demand wait moved from 26.992911 to 17.684029 s at 677 and from
28.135119 to 41.880322 s at 2,612, overwhelming the local attention change.

## Remaining score attribution

The 2,612-token score bucket rose from 61.008242 to 63.957948 s in this single
arm. Existing counters divide it as follows:

| score component | seconds |
|---|---:|
| paged-attention host remainder, both devices | 27.659238 |
| paged-attention stream wait, both devices | 29.944906 |
| learned-index work | 5.061226 |
| candidate resolution | 1.629104 |
| page-set construction | 0.084401 |
| weight acquisition | 0.000503 |
| sum | 64.379378 |
| score wall time | 63.957948 |
| accounting residual | -0.421430 |

The stream-wait counter is the exact interval around
`cudaStreamSynchronize`; it includes device execution. Per device, stream wait
was 15.435488 and 14.509418 s, while device attention was 15.670587 and
14.572557 s. `maximum_device_dsv4_paged_attention_seconds` therefore reports
15.670587 s, not the two-device sum. Host remainder is call wall time outside
that synchronization and was 14.236377 plus 13.422861 s. The two devices are
called synchronously, so the summed counters reconcile the score wall; adding
the maximum-device counter again would double-count kernel time. No score
bucket larger than 0.422 s remains unattributed.

The 383,225,472 page bytes are real repeated global-memory reads:
`dsv4_paged_attention_to_mhc` launches
`dsv4_materialize_physical_pages` over every supplied page on every sub-chunk,
and the counter sums each page buffer's device extent. It is not lease-only
accounting. The volume is nevertheless only 0.383 GB over the whole prefill
and cannot explain either approximately 30-second term at device-memory
bandwidth. A gather-once change is therefore not justified as a byte-volume
optimization by this measurement.

The measured dominant terms are the synchronous page command's 27.659 s host
remainder and 29.945 s stream wait. Source inspection provides one falsifiable
lead inside the host remainder: request validation performs a full host
`std::any_of` scan over every BF16-rounded query element for every layer and
sub-chunk before dispatch. That scan is not timed separately, so it is not yet
claimed as the cause. The cheapest next step is to time or microbenchmark that
exact scan before changing validation or command handoff. No second mechanism
was implemented in this experiment.

`moe_prepare_seconds` measured 17.685382 s at 677 and 41.835992 s at 2,612.
The latter supersedes 0110's 28.08 s observation for this arm and remains an
open lead; it was not investigated.

## Verdict

Correctness and the allocation diagnostic pass. The retained scratch removes
the measured repeated zero-fill and yields a real partial timing improvement,
but attention still has a 37.16 ms/token marginal term. Per the standing
protocol, preserve the implementation and report before deciding whether to
instrument the full-query validation scan or pursue another score mechanism.
The final post-arm `make check` passed both tests.
