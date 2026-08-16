# Experiment 0098 — row-batched physical prefill attention

Status: **rejected by the predeclared timing gate.** The 676-row production
page was reduced from one physical-attention dispatch per row and layer to one
dispatch per layer. Page reads collapsed from 6.99 GB to 30.56 MB and the
measured device term fell from 6.38 s to 4.14 s, but the attention-score bucket
was 13.945 s, above the binding 12 s stop threshold. No timing repetitions and
no 2,612-token arm were run.

**One-sentence result:** row batching transferred to production exactly at the
page boundary, but it did not clear the timing gate because only 4.14 s of the
remaining 13.95 s score bucket is measured device attention and the candidate
also reruns the learned-index compressor per row inside that bucket after the
append pass already prepared it, leaving a different serial term to isolate
before any further mechanism work.

The candidate runtime is not promotable. Its changes remain outside the result
commit and must be rolled back; this record is the checkpoint.

## Pre-change contract

At the 677-token, page-8192 production operating point, attention was the
measured bottleneck: 33.8 s of roughly 64 s prefill, with 18.3 s in scoring.
The scoring command performed 29,111 host dispatches and 553,109 kernel
launches and read 6.99 GB from an approximately 30 MB physical KV working set.

Hypothesis: adding a row dimension to physical paged attention and dispatching
one complete page per layer would reduce the serial launch/synchronization term
and eliminate redundant physical-page reads. The target resources were host
dispatch/synchronization and device KV-read volume. Query/KV projection work,
MoE traffic and arithmetic, router semantics, precision, expert residency and
mHC semantics were intended to remain unchanged.

Primary gate: at 677 tokens, `dsv4_paged_attention_calls` had to become about
43 and kernel launches order `10^3`; these counters were checked before timing.
Secondary gate: after that structural change, `attention_score_seconds` had to
be below 12 s. Correctness required bit-exact batched attention against the
per-row path, unchanged generated tokens, zero decode checkpoint reads and no
decode regression. The declared host ceiling was 216 GiB; per-device VRAM was
reported rather than hidden by the attention result.

Rollback was binding: if calls collapsed but scoring remained at or above
12 s, stop and report instead of extending the mechanism.

## Reference-derived mechanism

The design followed the local reference stack's page/chunk shape rather than a
new host data structure:

- the host chooses the page boundary and appends all KV rows before acquiring
  read leases;
- physical KV pages are gathered once into one flat device workspace;
- queries and candidates carry a row dimension;
- one score operation covers every row/head against the shared flat KV;
- a device kernel maps compressed logical positions to physical page/row and
  derives the complete sliding tail from `position_base + row`;
- output projections are row-batched and their BF16 branches are scattered to
  the existing per-row mHC slots;
- `wo_a` and `wo_b` are acquired once per layer/page.

The old append/attend split was retained because a KV block refuses mutation
while a device lease is outstanding.

## Cheapest mechanism gate

Before loading the model, the CUDA backend compared batched and repeated
single-row attention bit for bit at page sizes 4 and 64 and prompt sizes 8, 52
and 144. Every float bit matched. The same fixture asserts one backend call and
seven launches per page chunk and charges the shared physical page only once.
The complete CTest suite passed 2/2.

This isolated test established arithmetic equivalence and the intended
dispatch/read accounting. It did not predict that every other operation in the
runtime score bucket had been removed.

## Production mechanism check

One untraced baseline and one candidate were run at 677 tokens with
`--prefill-page-tokens 8192`, devices 1 and 2, 216 GiB host admission and four
generated tokens. The budget was at most four minutes per arm and eight minutes
total. Initialization was 100.3 s baseline and 104.2 s candidate, approximately
61% of each setup-plus-prefill arm; the cheaper isolated mechanism fixture had
already passed.

| metric | baseline | candidate | ratio / delta |
| --- | ---: | ---: | ---: |
| paged-attention calls | 29,111 | 86 | 338.5x fewer |
| page-body graph dispatches | 29,068 | 43 | 676.0x fewer |
| paged-attention kernel launches | 553,109 | 1,677 | 329.8x fewer |
| physical page bytes | 6.990 GB | 30.564 MB | 228.7x fewer |
| maximum device attention | 6.383 s | 4.140 s | -2.243 s |
| attention score | 18.324 s | **13.945 s** | -4.379 s; gate failed |
| total attention | 33.762 s | 28.567 s | -5.196 s |
| prefill | 65.223 s | 59.833 s | not a promoted timing result |
| prefill rate | 10.380 tok/s | 11.315 tok/s | one repetition only |

The 86 aggregate CUDA calls are explained exactly by the existing prefill
driver. It page-processes all prompt rows but the last: 676 rows produce 43
row-batched calls, one per layer. The final prompt row stays on the token-major
path so it can produce logits and contributes another 43 calls. The launch
count is correspondingly `43 * 20 + 43 * 19 = 1,677`. Thus the page mechanism
itself met its structural gate; the remaining final-row calls are not hidden
row dispatch inside the page.

The clock gate did not pass. Subtracting measured device attention leaves
9.804 s in the candidate's score bucket. Reading the integrated path exposed a
specific accounting and implementation defect: after the append loop calls
`attention_append_prepared(..., append_index_compressor=true)` for every row,
`physical_paged_attention_page` calls `index_positions` for those rows. That
entry point calls `compress_state` again before selection. The old split path
used `index_select` when the compressor was already prepared. At 677 tokens
selection itself is the identity because compressed history is below top-k,
so the duplicate compressor work cannot be justified as selection cost and is
now charged inside `attention_score_seconds`.

This observation does not rescue the result. The declared gate required a
sub-12-second integrated score term. Removing or restructuring learned-index
preparation is a new measured subproblem and was not built after the gate
failed.

## Other-resource signs and correctness

| metric | baseline | candidate |
| --- | ---: | ---: |
| attention H2D | 2.074 GB | 2.065 GB |
| attention D2H | 238.414 MB | 238.297 MB |
| expert/weight H2D | 73.786 GB | 74.083 GB |
| cache misses | 10,428 | 10,470 |
| cache evictions | 7,800 | 7,842 |
| decode checkpoint reads | 0 | 0 |
| decode rate, 3 measured steps | 6.798 tok/s | 6.791 tok/s |
| RSS | 147.79 GiB | 147.95 GiB |
| GPU 1 VRAM | 21.13 GiB | 21.63 GiB |
| GPU 2 VRAM | 21.05 GiB | 21.55 GiB |

The generated token IDs were equal: `[2107, 8777, 1277, 440]`. Decode
checkpoint reads remained zero. A single three-step decode tail cannot promote
the 0.1% rate difference as either a regression or a win. The candidate added
approximately 512 MiB per GPU for its bounded page workspace; that cost must be
reduced or explicitly admitted before any future candidate can promote.

The full-model output check at this short operating point does not erase the
learned-index double-update defect: compressed history is below the 512-entry
top-k threshold, so selection is identity and the four generated tokens do not
exercise the later sparse regime.

The expert-upload term remained independently NUMA-sensitive. Its bytes and
cache counts are reported, but total prefill from one arm is not used to claim
a throughput win.

## Stopped work

Per the kill criterion, the following were deliberately not run:

- three interleaved 677-token timing repetitions;
- any 2,612-token arm;
- a fixed/marginal fit;
- any follow-on query projection, KV projection, MoE NUMA or workspace change.

Those fields are absent because the experiment stopped, not because a partial
result was promoted. A future experiment must first split the 9.804 s
non-device remainder into learned-index preparation, weight acquisition,
branch handoff and synchronization with counters, then state its own gate.

Raw mechanism data (ignored):
`results/dsv4-0098-attention-page-dispatch-check/`.

## Reproduce the bounded check

```bash
scripts/run_dsv4_attention_page_dispatch_check.sh
```

The production arm must remain untraced: `--layer-hash-trace` disables the
fused attention command and does not exercise this mechanism.
