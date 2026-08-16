# Experiment 0099 — row-batched attention with prepared index selection

Status: **implementation and isolated correctness gates complete; production
mechanism gate pending.** No model arm has been launched from this experiment.

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

Four candidate-only counters split the score remainder into prepared learned-
index selection, output-weight acquisition, branch handoff/scatter host work,
and stream-synchronization wait. Existing candidate-resolution and device
attention counters remain available; the branch-handoff counter is the batched
backend call's host wall remainder after subtracting its measured stream wait.

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

No production counter or timing number belongs to 0099 yet. The next action is
the predeclared two-arm mechanism check, but it requires a budget check-in
before launch.

## Planned production mechanism check

One baseline and one candidate arm, each untraced, at 677 tokens and
`--prefill-page-tokens 8192`. Expected wall time is approximately four minutes
per arm after the model is resident/loaded as accounted by the existing
script, eight minutes measured-arm total. No repetition, 2,612-token arm or
fixed/marginal fit is authorized at this stage.

The production JSON must report the structural counters, score and maximum
device-attention seconds, the four new remainder counters, expert H2D bytes,
cache misses/evictions, decode checkpoint reads, decode rate, RSS and per-GPU
VRAM. The unsigned-underflow mHC maximum counter remains a separate known
defect and is not used.

