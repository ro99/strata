# Experiment 0030 — exact DeepSeek batched attention rows

Status: **promoted**. Exact multi-query attention batching improves the
production-context prefill by **1.308x**. Its three-repetition base-decode
effect is **1.008x**, not the earlier one-pair estimate of 1.089x. Forced CUDA
dispatch beats the old 256-row hybrid policy, so opt-in FlashAttention now uses
a zero-row crossover by default.

## Contract

- Hypothesis: packing a page's shared sliding/compressed KV union once and
  scoring its query rows in one exact CUDA call reduces serial attention
  dispatch and staging work.
- Operating point: 511-token prompt, 128 outputs, 64-row prefill pages, block
  KV, three GPUs, 216 GiB host ceiling, 0.85 VRAM fraction, and 256 MiB of
  admitted device KV per GPU.
- Primary metrics: median prefill seconds and base-decode rows/s over three
  interleaved repetitions. Re-test forced CUDA against the prior 256-row
  hybrid crossover.
- Correctness: identical tokens, logits, layer/operation hashes, and routes;
  zero checkpoint reads and bounded admitted memory.
- Rollback: any exactness/I/O/memory failure, overlapping prefill ranges, or no
  decode improvement outside the baseline range.

The frozen baseline is `main` at `9c170bd`; the candidate contains only exact
attention batching commits `4484aff` and `61a81c5` plus the matrix harness.
The cheaper 12-token oracle was used for exactness but rejected as a performance
proxy because it does not reproduce the production context or dispatch mix.
Initialization was 43--44 seconds. Relative to the measured windows, fixed
setup was 0.37x prefill and about 3.6x decode; both phases are reported
separately.

## Measured bottleneck and mechanism

With forced CUDA on `main`, prefill attention was the largest graph phase at
58.708 seconds, ahead of MoE at 41.128 seconds and mHC pre at 17.303 seconds.
Per-row FlashAttention made 21,973 calls, staged 9,359,709,440 bytes H2D, and
returned 2,880,132,948 bytes D2H. The target is the serial dispatch/staging
term: batching shares the KV input across query rows and reduces call count; it
does not change model arithmetic, routing, precision, or cache semantics.

The candidate made 344 calls and staged 3,027,679,215 bytes H2D, reductions of
98.4% and 67.7%. D2H, routed-weight H2D, and general activation bytes were
unchanged. Critical upload wait fell 6.729 to 6.698 seconds, synchronization
18.345 to 17.613 seconds, and kernels 19.110 to 18.920 seconds. MoE and mHC
were effectively unchanged.

## Three-repetition result

The arm order rotated each repetition:

1. baseline-forced, candidate-hybrid, candidate-forced
2. candidate-hybrid, candidate-forced, baseline-forced
3. candidate-forced, baseline-forced, candidate-hybrid

| arm / repetition | prefill seconds | decode rows/s |
|---|---:|---:|
| baseline forced 1 | 122.178 | 2.7635 |
| baseline forced 2 | 120.124 | 2.7488 |
| baseline forced 3 | 120.645 | 2.7538 |
| candidate forced 1 | 93.626 | 2.7705 |
| candidate forced 2 | 92.211 | 2.8132 |
| candidate forced 3 | 91.384 | 2.7755 |
| candidate hybrid 1 | 129.001 | 2.7194 |
| candidate hybrid 2 | 128.986 | 2.6931 |
| candidate hybrid 3 | 128.997 | 2.7604 |

Median resource accounting:

| metric | main forced | candidate forced | candidate hybrid |
|---|---:|---:|---:|
| prefill seconds | 120.645 | **92.211** | 128.997 |
| prefill attention | 58.708 s | **30.402 s** | 66.755 s |
| prefill attention score | 13.202 s | **7.714 s** | 19.545 s |
| prefill FlashAttention calls | 21,973 | **344** | 0 |
| prefill attention H2D | 9,359,709,440 B | **3,027,679,215 B** | 0 B |
| prefill weight H2D | 104,962,719,744 B | same | same |
| prefill activation H2D / D2H | 6,465,441,792 / 6,767,890,432 B | same | same |
| decode seconds / 127 rows | 46.117 | **45.757** | 46.701 |
| decode rows/s | 2.7538 | **2.7755** | 2.7194 |
| decode attention | 20.004 s | **19.813 s** | 21.050 s |
| decode score plus KV | 5.536 s | 5.561 s | 6.563 s |
| decode synchronization | 7.619 s | **7.601 s** | 7.309 s |
| decode kernels | 5.021 s | 5.048 s | 5.015 s |
| decode upload wait | 5.000 s | 5.012 s | 4.983 s |
| decode weight H2D | 73,852,256,256 B | same | same |
| decode activation H2D / D2H | 1,608,339,456 / 1,747,585,024 B | same | same |

Candidate forced is 1.308x on prefill and 1.0079x on decode against main
forced. Every candidate-forced decode run beats every baseline run, though the
effect is small. The prior 1.089x decode estimate and the claimed 123.9 to 93.9
ms/row score-plus-KV reduction did not replicate and are superseded by this
matrix.

## Crossover and gates

On the candidate, forced dispatch beats hybrid by 1.399x on prefill and 1.0206x
on decode. Hybrid makes no batched prefill CUDA calls at this operating point;
forced batches the eligible non-sparse pages and also moves all 5,461 decode
attention calls to CUDA rather than 2,667. The added decode synchronization is
7.601 versus 7.309 seconds, but the attention reduction is larger, so the old
256-row crossover no longer minimizes end-to-end time. The opt-in default moves
to zero; the explicit threshold knob remains for other hardware and contexts.

All gates pass. Generated tokens match in all nine arms; the clean real-model
oracle separately matches logits, layer hashes, operation hashes, and routes.
Prefill and decode checkpoint reads are zero. Median RSS is 148.945 GB against
the 150.546 GB plan; host KV peaks at 7,923,712 bytes, and device KV peaks at
zero within the admitted 256 MiB per GPU. `make check` passes.

Evidence: `results/deepseek-v4-batched-attention-rows-matrix/` and
`results/deepseek-v4-batched-attention-correctness/`. Re-run with
`scripts/run_deepseek_v4_batched_attention_rows_matrix.sh` and
`scripts/run_deepseek_v4_batched_attention_correctness.sh`.
