# Experiment 0099 — row-batched attention with prepared index selection

Status: **rejected at the unchanged 677-token production timing gate.** The
row-batched structure and correctness gates passed, but removing the duplicate
compressor pass left `attention_score_seconds` at 14.159 s against the binding
`< 12 s` gate. The candidate and all measurements are preserved.

This is the second attempt at experiment 0098's mechanism. The operating
point, structural expectations and `attention_score_seconds < 12 s` gate are
unchanged. The sole semantic correction is that page attention selects against
the learned-index compressor state already advanced by the append pass; it no
longer invokes a second compressor pass for every row.

## Pre-change contract

Experiment 0098 reduced physical-attention calls from 29,111 to 86, kernel
launches from 553,109 to 1,677 and page bytes from 6.990 GB to 30.564 MB. Its
attention-score bucket fell from 18.324 s to 13.945 s and device attention fell
from 6.383 s to 4.140 s. The remaining non-device remainder was therefore
9.805 s.

Inspection identified a candidate defect in that remainder. The append loop
called `attention_append_prepared(..., append_index_compressor=true)` for every
row, then the page score path called `index_positions`, which enters
`compress_state` again before selection. The pre-existing per-row split path
called `index_select` when append had already prepared the compressor.

Hypothesis: changing only the page path from `index_positions` to
`index_select` removes duplicate learned-index preparation from the serial
score remainder while preserving the selected positions exactly. The measured
target is serial host/device preparation inside `attention_score_seconds`; it
does not reduce attention arithmetic, query/KV projection, routed-expert
traffic or MoE arithmetic.

Primary metric and unchanged gate: one untraced 677-token candidate at page
8192 must retain approximately 86 paged-attention calls, 1,677 launches and
30 MB of page reads, and `attention_score_seconds` must be below 12 s. The
non-device remainder, score seconds minus maximum device-attention seconds,
must fall well below 0098's 9.805 s. If either direction of the gate is reached,
stop and report before any repetition, 2,612-token arm, cleanup or next
mechanism.

Correctness gate: page attention is bit-exact against repeated single-row
attention at page sizes 4 and 64 for prompts of 8, 52 and 144 tokens. Sparse
selection is gated beyond the identity regime: the scalar-oracle index fixture
selects top-512 from 513 positions, and page attention receives a different
512-of-513 descriptor set for every row and must remain bit-exact when batched.
The later model gate additionally requires equal generated tokens, zero decode
checkpoint reads and unregressed decode.

Memory ceiling: the page command is hard-bounded at 384 MiB per attention
device. At the 677-row shape its computed workspace is expected near 310 MiB;
when the geometric 512 MiB allocation would exceed the bound, the backend
allocates the exact required extent instead. This prevents the approximately
512 MiB per-GPU increase observed in 0098. The 216 GiB host ceiling is
unchanged.

Rollback condition: any bit mismatch, selection mismatch, workspace-bound
failure at the admitted shape, structural counter regression, decode read or
decode regression stops the experiment. Per the standing protocol, the code
and results are preserved and no rollback or discard occurs without explicit
approval.

## Restored mechanism and correction

The recoverable 0098 host/header/test blobs were restored from Git's shared
object store. `kernels/cuda/backend.cu` was reconstructed because its
uncommitted version had no recoverable object.

The restored implementation:

- appends every page row before taking any physical-KV device lease;
- gives queries and candidates a row dimension;
- materializes each shared physical page once into a flat device workspace;
- scores all page rows in one matrix operation per 32-head group;
- finishes each row against its own candidate descriptors without changing
  score, softmax or value accumulation order;
- applies inverse RoPE and both output projections across the row dimension;
- scatters each BF16 branch to its existing device mHC slot;
- acquires `wo_a` and `wo_b` once per layer/page;
- leaves the single-row decode request and fused decode path on their existing
  defaults.

The 0099 correction is deliberately narrower: after append has advanced the
index compressor, each page row calls `index_select` directly. No second
`compress_state` call remains on the page path.

Four counters split the score remainder in both mechanisms into prepared
learned-index selection, output-weight acquisition, paged-attention host
remainder (staging, dispatch and branch handoff/scatter), and stream-
synchronization wait. The latter two are recorded inside the backend's existing
completion boundary, so the 29,111-call baseline does not acquire and copy the
aggregate stats object once per row. Existing candidate-resolution and device
attention counters remain available; host remainder is complete backend call
wall time after subtracting its measured stream wait.

## Cheapest correctness measurement

The CUDA fixture passed on the local SM86 device:

- repeated single-row versus row-batched physical attention, bit for bit;
- page sizes 4 and 64;
- prompt sizes 8, 52 and 144;
- sparse history of 513 compressed positions with a row-dependent 512-entry
  selected set and a fixed 640-entry candidate layout;
- a 384 MiB maximum workspace contract.

The existing Lightning Indexer fixture independently passed exact top-512
selection against its scalar oracle at 513 candidates and at the declared
1,048,576-token maximum. The local test binary reported 276 passed and 34
fixture-dependent skips.

## Production mechanism gate

The authorized budget was one untraced baseline and one untraced candidate at
677 tokens and `--prefill-page-tokens 8192`, approximately four minutes each
and eight minutes total. The candidate ran first so its deterministic
structural counters could gate further model work. It completed in 2:53 wall
time, including 104.528 s of initialization. Its result was:

| Metric | 0098 candidate | 0099 candidate | Gate / reading |
|---|---:|---:|---|
| Prefill | 59.833 s | 60.195 s | 11.247 tok/s; single runs, not a timing comparison |
| Attention total | — | 28.794 s | query 10.338 s, KV 3.908 s, score 14.159 s |
| Attention score | 13.945 s | **14.159 s** | **fail: required < 12 s** |
| Maximum device attention | 4.140 s | 4.108 s | device kernels are not the residual |
| Score minus device | 9.805 s | **10.051 s** | did not fall well below 0098 |
| Paged-attention calls | 86 | **86** | structural pass |
| Kernel launches | 1,677 | **1,655** | structural pass, order 10^3 |
| Page bytes | 30.564 MB | **30.564 MB** | structural pass |
| Prepared index selection | — | **0.000 s** | identity regime at 677 tokens |
| Output-weight acquisition | — | **0.000390 s** | negligible |
| Backend host remainder | — | **6.234 s** | staging/dispatch/handoff/scatter aggregate |
| Backend stream synchronization | — | **7.620 s** | includes the 4.108 s device interval |

The launch count is 22 below 0098's 1,677 because the reconstructed counter
charges the extra page-scatter launch only when a cross-device scatter is
actually issued. Calls, launches and page bytes nevertheless prove the same
row-batched structure; this is not a relaxation of the structural gate.

The remainder counters explain the second failure sufficiently to reject the
stated correction: prepared index selection was exactly zero at this operating
point, weight acquisition was 0.390 ms, and
`6.234 + (7.620 - 4.108) = 9.746 s` of backend host work plus synchronization
accounts for nearly all of the measured 10.051 s score-minus-device remainder.
The removed duplicate compressor therefore was not the dominant 0098 residual.
This is a negative result against the same threshold, prompt and page size, not
a gate change or a friendlier regime.

The baseline arm was not launched. Once the candidate's unchanged timing gate
failed, the standing protocol required a stop before spending the second arm;
there is consequently no 0099 A/B timing claim and no baseline remainder split.
The two preflight attempts that failed before model loading (one NCCL-disabled
build and one incorrect model path) remain preserved under the result directory
and consumed no measured arm.

## Correctness and resource accounting

The production arm generated token IDs `2107, 8777, 1277, 440`, matching the
0098 gate arms. Decode checkpoint reads remained zero. Four generated tokens
took 0.440 s; under the runner's existing first-token accounting this is 6.819
decode tok/s, versus 0098's 6.798 baseline, so this single run shows no decode
regression but is not a throughput-win claim.

Prefill expert demand H2D was 74,083,499,520 bytes, with 10,470 cache misses
and 7,842 evictions. RSS was 158,858,715,136 bytes. Per-GPU used VRAM was
22,994,354,176 and 22,916,759,552 bytes. Relative to 0098's candidate this is
226,492,416 bytes less on each GPU; relative to 0098's baseline it is
310,378,496 bytes more on each GPU. The latter is the admitted row-batched
workspace cost and remains below the explicit 384 MiB-per-attention-device
ceiling. It did not grow further.

The known `maximum_device_dsv4_mhc_kernel_seconds = 18446744070` unsigned
underflow is present and was not used. The final `make check` passed 2/2 tests.

Raw candidate JSON, system capture and full `/usr/bin/time -v` log are under
`results/dsv4-0099-prepared-selection-gate/`. No repetition, 2,612-token arm,
fixed/marginal fit, query/KV projection work or MoE work was run. Per the
standing protocol, the experiment stops here with its code and data preserved;
the next mechanism requires explicit direction.
