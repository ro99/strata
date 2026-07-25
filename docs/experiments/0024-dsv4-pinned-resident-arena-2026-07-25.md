# Experiment 0024 — page-locking the DeepSeek resident weight arena

## Contract

- Hypothesis: the routed-expert arena is a 138 GiB `mmap` and every decode
  demand load is a cold, randomly placed slice of it, so `cudaMemcpyAsync` pays
  the driver's pageable staging copy. Page-locking the arena removes that copy.
- Bottleneck measured first, at a 3,565-token operating point: a 576 ms decode
  step is 271 ms MoE prepare (255 ms of it demand wait), 197 ms attention,
  43 ms mHC, 47 ms MoE compute. The transfer moved 582 MB/step at
  **2.28 GB/s** against a link that does 11.9 GB/s pinned. Per the charter an
  order-of-magnitude gap against the rated figure is a serialization defect,
  not a bandwidth cost.
- Target term: that staging copy, the largest single term in the step.
- Sign on every other resource: no compute added, no VRAM added, no change to
  routing, precision, top-k or expert count. One-time registration at load, and
  the arena's pages become unswappable for the process.
- Correctness: pinning moves no byte, so output must be bit-identical.
- Rollback: median decode not above baseline beyond run variance, or any output
  byte changing.

## Cheap measurement first

The topology probe reported pinned and pageable H2D as identical, 12.1 GB/s
both, and would have killed this idea. It was wrong because it timed a warm
reused buffer. Re-run against the production access pattern -- a cold, randomly
placed slice of the real mapping, at the real 4.46 MB projection size:

| path | ms | GB/s |
|---|---:|---:|
| pageable, warm reused buffer | 0.458 | 9.7 |
| pageable, cold arena slice | 1.316 | 3.4 |
| pinned, cold arena slice | 0.375 | 11.9 |

That five-minute standalone benchmark, not the end-to-end A/B, is what settled
the question. It predicts a 3.51x reduction in demand wait.

## Three defects found before any number could be produced

`--pin-resident-arena` had never pinned anything, and it aborted the run.

1. `stage()` seals the arena with `mprotect(PROT_READ)`. `cudaHostRegister`
   refuses a read-only mapping with `cudaErrorInvalidValue`, and
   `cudaHostRegisterReadOnly` is no escape: all three devices report
   `cudaDevAttrHostRegisterReadOnlySupported == 0`, so that flag returns
   "operation not supported". `pin()` now unseals for the registration call and
   reseals immediately; staging and warm-up have both joined by then, and H2D
   out of a registered-then-resealed mapping was verified to work.
2. `register_host_memory` returned without clearing CUDA's error state, so the
   failure above survived to the next `cudaGetLastError()` -- which belongs to
   an unrelated kernel launch -- and was reported as
   `launch CUDA matmul: invalid argument`. A best-effort optimization was
   killing the run and blaming a kernel it had nothing to do with.
3. The pin failure was printed only under `config.verbose`, which
   `strata-deepseek-run` never sets. An unpinned run was indistinguishable from
   a pinned one.

Chunked registration was tried and rejected: 4 GiB chunks register at
21.4 GB/s against 2.38 GB/s for one call, but a 4.46 MB weight read straddling
two separately registered ranges is refused, which surfaced as
`upload CUDA weights: invalid argument` on `layers.2.ffn.experts.132.w2`. The
arena is registered as a single range.

## Commands

```bash
scripts/run_deepseek_v4_pinned_arena_ab.sh
```

Two arms, interleaved, three repetitions. Prompt 34 sentences (~512 tokens),
`--max-new 128`, `--prefill-page-tokens 64`. Decode throughput is the
hypothesis, so the prompt is the shortest that still exercises the 128-row
sliding window and leaves the expert cache cold. About 5 minutes per arm,
30 minutes total.

## Result

Median of three interleaved repetitions:

| | reference | pinned | |
|---|---:|---:|---|
| Decode steps/s | 1.913 | 3.110 | **1.626x** |
| Decode seconds | 66.39 | 40.84 | |
| Demand wait (s) | 36.49 | 10.95 | 3.33x less |
| Registration (s) | 0 | 16.88 | one-time, at load |

Every run, not just the median. Demand H2D bytes are identical in all six runs
at 73,852,256,256, so the workload is fixed and only the rate differs.

| rep | arm | steps/s | demand wait (s) | effective GB/s | pin (s) |
|---|---|---:|---:|---:|---:|
| 1 | reference | 2.333 | 24.61 | 3.00 | 0 |
| 2 | reference | 1.776 | 41.27 | 1.79 | 0 |
| 3 | reference | 1.913 | 36.49 | 2.02 | 0 |
| 1 | pinned | 3.147 | 10.95 | 6.74 | 16.88 |
| 2 | pinned | 3.110 | 11.01 | 6.71 | 17.23 |
| 3 | pinned | 3.040 | 10.94 | 6.75 | 17.56 |

The reference arm spans 1.776-2.333 steps/s (+/-15%) while the pinned arm spans
3.040-3.147 (+/-1.7%). The two ranges do not overlap -- the slowest pinned run
beats the fastest reference run by 30% -- so this is not a variance artifact.
The pageable path's rate depends on host memory state at the moment of each
copy; page-locking removes that dependence, which is why the candidate is also
the more repeatable arm.

Gates, all passing: `pinning_actually_happened`, `generated_tokens_equal`,
`logits_equal`, `layer_hashes_equal`, `operation_hashes_equal`,
`zero_decode_checkpoint_reads`, `faster`. `make check` passes.

The measured 3.33x reduction in demand wait lands on the micro-benchmark's
predicted 3.51x. The mechanism is confirmed, not merely the outcome: this is
the pageable staging copy disappearing, which is what the model said it was.

**The gap is not closed.** Pinned reaches 6.74 GB/s effective, against 11.9 GB/s
for the same transfer in isolation. Pinning bought 3.33x of an available ~5.9x;
the residual is that `CudaBackend::upload` issues H2D on the compute stream and
then synchronizes it, so no copy overlaps any compute. That is a separate
`Sigma_serial` defect and needs its own branch -- a per-device copy stream with
`cudaStreamWaitEvent` before `enqueue_deepseek_moe`.

## Cost accounting

Registration costs 16.88 s at load and decode saves 25.55 s on this 128-token
run, so the run is net positive by 8.7 s -- modestly. The saving scales with
generated tokens while the registration does not, so the ratio improves with
generation length and is worst exactly here. Quoting the 1.626x decode figure
without this line would overstate the end-to-end effect on short runs.

## Operating point

This is a ~512-token prompt. The 576 ms/step and 2.28 GB/s figures that
motivated the work were measured at 3,565 tokens. **The 1.626x must not be
transplanted to another context length.** Attention cost grows with context
while expert traffic per step does not, so the share of the step this mechanism
addresses falls as context grows.

## Decision

Promote the fix. `pin_resident_arena` stays opt-in via `--pin-resident-arena`
until it has been measured at the long-context operating point, because the
registration cost is fixed and the benefit is context-dependent.

## Next term

Decode attention is now the candidate `argmax_r`: 197 ms/step at 3,565 tokens
with `attention_cuda_dispatches: 0` -- all 5,461 decode dispatches take the
host scalar path, because `flash_attention_minimum_rows` is 256 and batch-1
decode presents one row. That needs its own branch, hypothesis and gate.
