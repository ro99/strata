# Experiment 0172 — Laguna two-3090 spine sharding fails at layer shape

Status: **REJECTED.** Exact row sharding across the two RTX 3090s does not pass
the predeclared 1.5x gate for any recurring Laguna layer projection. Q improves
only 1.115x, O improves 1.011x, and the small shared projection regresses to
0.597x. The output head passes at 1.54x, but it is a 1.6 ms/token phase and its
0.29 ms standalone saving cannot materially change a 53.69 ms step. No TP
runtime was built.

## Contract and cheapest falsifier

After experiment 0171, attention is `argmax` at 26.72 ms/token. Its BF16
projection subterms are 9.83 ms for Q/K/V/G and 5.59 ms for gating plus O. The
hypothesis was that row-sharding each resident BF16 matrix across the two 3090s
would reduce this HBM/linear-service term while preserving exact arithmetic.

The primary metric was median end-to-end projection latency and effective
weight GB/s over 21 interleaved samples at real Laguna shapes. Correctness
required bit-exact F32 carriers for every output row. Aggregate resident spine
bytes could not grow. The adverse terms were the second input upload, an output
download from each device, persistent-worker dispatch, and redistribution of
spine bytes away from expert-cache capacity. The binding rollback was any
mismatch or less than 1.5x on the recurring layer shapes.

`strata-laguna-spine-tp-probe` is the mechanism-only falsifier. It maps the real
checkpoint BF16 tensors, puts the canonical matrix on one 3090 and equal row
shards on both 3090s, uses the production CUDA backend and persistent host
workers, and includes H2D, kernel, D2H, synchronization, and dispatch in the
wall measurement. Each row still executes the identical one-warp sequential
dot and shuffle tree; sharding only assigns disjoint rows to devices.

The complete probe took 1.23 seconds. A real-model TP implementation and A/B
would have cost minutes and changed placement throughout the runtime, so it was
rejected in favor of this direct measurement.

## Result

| real tensor | bytes | one 3090 | two 3090 | speedup | one-card GB/s | aggregate GB/s | extra transfer | mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sliding Q, 9216x3072 | 54 MiB | 0.111 ms | 0.100 ms | **1.115x** | 508.2 | 566.6 | 12 KiB | 0 |
| sliding O, 3072x9216 | 54 MiB | 0.100 ms | 0.099 ms | **1.011x** | 564.2 | 570.2 | 36 KiB | 0 |
| shared gate, 1024x3072 | 6 MiB | 0.040 ms | 0.066 ms | **0.597x** | 159.0 | 95.0 | 12 KiB | 0 |
| LM head, 100352x3072 | 588 MiB | 0.825 ms | 0.535 ms | **1.540x** | 747.7 | 1151.6 | 12 KiB | 0 |

The shape prediction is decisive. Only the 588 MiB matrix is long enough to
amortize a second-device dispatch. Laguna's recurring 54 MiB and 6 MiB matrices
are not: a second 3090 adds almost no bandwidth at 54 MiB and loses at 6 MiB.
Under the measured 53.69 ms cost model, even applying the LM-head standalone
saving perfectly changes the step by about 0.5%. Building TP on that exception
would launder a failed layer gate.

## Consequence

DeepSeek's TP=2 is not a transferable explanation by itself. DeepSeek uses a
rank-local graph built around much larger projection shapes and amortized
collectives. Laguna's immediate remaining term is serial service across many
small, synchronous projections and host numerical boundaries. Any next
mechanism must reduce those boundaries at the whole-chain level while retaining
the exact Laguna operation order; duplicating or sharding individual matrices
does not.

Raw ignored result: `results/laguna-spine-tp-gate.log`.

