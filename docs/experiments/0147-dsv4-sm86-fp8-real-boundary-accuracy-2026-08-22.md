# Experiment 0147: SM86 FP8 real-boundary accuracy

## Result

**The guarded QPN8-derived W8A16 reduction passes the real-fixture no-worse
gate at layers 2, 21, and 42.** The first unguarded run did not: layer 2
`wq_b+indexer.wq_b` produced 9 oracle BF16 differences versus the incumbent's
4, and grouped `wo_a` produced 1 versus 0. That negative result was binding for
the unguarded reduction order and was not relabelled as acceptable.

The retained successor keeps HMMA as the fast path and uses an SM86 warp vote
to identify outputs close to a BF16 rounding midpoint. Only those rows, plus
nonzero near-zero cancellations, are recomputed in FP64 by one warp directly
from the same invertibly prepacked E4M3 bytes. It creates no widened or duplicate
weight representation. Across the three real fixtures, 1.63--1.68% of query
rows and 1.46--1.90% of grouped-output rows take the replay. Every declared
maximum-absolute, maximum-relative, RMS, and mismatch metric is no worse than
the scalar W8A16 incumbent.

This closes F8-1's real-fixture numerical prerequisite. It is not an F8-2
throughput verdict. The replay has not yet been installed in or timed through
the layer-resident scheduler.

## Contract and cheapest falsifier

- **Hypothesis:** real checkpoint distributions make the observed synthetic
  BF16 differences either harmless against the incumbent or sparse enough for
  a same-layout guarded replay without erasing the Ampere scheduler's measured
  DRAM-bound advantage.
- **Primary metric:** candidate versus scalar-incumbent maximum absolute,
  maximum relative, RMS, and BF16 oracle mismatch count at each immediate
  projection publication boundary.
- **Correctness gate:** candidate is no worse on every metric and operation at
  all three retained real layers; checkpoint E4M3/E8M0 bytes, BF16 activation
  boundary, FP32 accumulation, and BF16 publication remain unchanged.
- **Memory ceiling:** 512 MiB. The largest sequential operation allocation is
  43,682,308 bytes (41.66 MiB); weights exist once on device in fragment order.
- **Rollback:** the unguarded reduction was rejected immediately when layer 2
  failed. The guarded successor would also be rejected if any of layers 2, 21,
  or 42 were worse; the M curve remained blocked during this experiment.

The layer-2 arm takes about five seconds including reading and prepacking the
real checkpoint tensors and evaluating every output against the FP64 oracle.
It was therefore cheaper and more discriminating than a full model load. The
middle and late fixtures were run only after that first arm established a
plausible correction.

## Exact production inputs

The probe reads the untouched tensors and E8M0 scales directly from the actual
safetensors shards:

| Layer | Shard | Operations |
|---:|---|---|
| 2 | `model-00004-of-00048.safetensors` | all five projections plus indexer `wq_b` |
| 21 | `model-00023-of-00048.safetensors` | all five projections |
| 42 | `model-00044-of-00048.safetensors` | all five projections plus indexer `wq_b` |

The retained reference traces provide actual BF16 `hidden_states`, normalized
`qr`, and attention output. Thus `wq_a` and `wkv` consume the real layer input,
and `wq_b` consumes the real post-query-RMSNorm activation rather than the raw
`wq_a` carrier used by experiment 0146's performance-only chain. Before
grouped `wo_a`, the fixture applies the reference inverse-RoPE formula using
the captured position's real cosine/sine vector and rounds to the declared
BF16 W8A16 operand. Each 1,024-row `wo_a` group selects its own 4,096-element
attention slice. `wo_b` is isolated with the scalar incumbent's real-derived
`wo_a` BF16 publication so a preceding candidate difference cannot hide its
own operation error.

This audit exposed two semantic gaps in the 0146 scheduler probe: it omitted
query RMSNorm between `wq_a` and `wq_b`, and it reused one attention slice for
all `wo_a` groups. Those choices preserve the byte-stream performance question
but are not an exact graph. They must be corrected before the protected
scheduler M curve or production integration.

## Negative arm and guarded mechanism

At layer 2, the original performance splits `{wq_b=1, wo_a=4}` measured:

| Operation | Incumbent mismatches | Unguarded candidate mismatches | Verdict |
|---|---:|---:|---|
| `wq_a` | 0 | 0 | pass |
| `wq_b+indexer.wq_b` | 4 | 9 | **fail** |
| `wkv` | 0 | 0 | pass |
| grouped `wo_a` | 0 | 1 | **fail** |
| `wo_b` | 0 | 0 | pass |

Split-K `{2,4,8,16,32}` changed the association but did not make every metric
no worse. Split 32 reduced the fused query candidate to one oracle mismatch,
showing that decode and scale binding were not the cause, but its execution
geometry is not the M=1 performance architecture.

The retained correction acts only after the fast HMMA sum:

1. the winning warp computes the FP32 distance to the nearest BF16 midpoint;
2. `__ballot_sync` collects ambiguous rows without another kernel launch;
3. one warp replays each selected row in FP64, inverse-indexing the existing
   fragment prepack; and
4. the result publishes through the same RNE BF16 boundary.

Ungrouped projections use a 512-FP32-ulp midpoint guard. Grouped `wo_a` uses
1,024 ulps, which covers its measured 768-ulp late-layer tail. A separate
nonzero `abs(sum) <= 1e-6` condition catches the one real layer-21 cancellation
that lands on a BF16 value rather than a midpoint. Exact structural zero rows
are excluded; the TP2 trace pads 32 inactive attention heads, and replaying
those zeros would be false work.

The replay reduces, rather than relaxes, numerical error. It is observable via
the `fp64_replay_rows` counter and never changes storage or activation format.

## Accepted real-layer result

| Layer / operation | Incumbent mismatches | Candidate mismatches | Incumbent max abs | Candidate max abs | Replay rows |
|---|---:|---:|---:|---:|---:|
| 2 `wq_a` | 0 | 0 | 0 | 0 | 17 / 1,024 |
| 2 `wq_b+indexer` | 4 | **0** | 3.815e-6 | **0** | 688 / 40,960 |
| 2 `wkv` | 0 | 0 | 0 | 0 | 3 / 512 |
| 2 grouped `wo_a` | 0 | 0 | 0 | 0 | 156 / 8,192 |
| 2 `wo_b` | 0 | 0 | 0 | 0 | 69 / 4,096 |
| 21 `wq_a` | 0 | 0 | 0 | 0 | 17 / 1,024 |
| 21 `wq_b` | 3 | **0** | 9.537e-7 | **0** | 535 / 32,768 |
| 21 `wkv` | 0 | 0 | 0 | 0 | 6 / 512 |
| 21 grouped `wo_a` | 0 | 0 | 0 | 0 | 134 / 8,192 |
| 21 `wo_b` | 1 | 1 | 3.815e-6 | 3.815e-6 | 54 / 4,096 |
| 42 `wq_a` | 0 | 0 | 0 | 0 | 16 / 1,024 |
| 42 `wq_b+indexer` | 5 | **1** | 2.441e-4 | **5.960e-8** | 660 / 40,960 |
| 42 `wkv` | 0 | 0 | 0 | 0 | 9 / 512 |
| 42 grouped `wo_a` | 0 | 0 | 0 | 0 | 120 / 8,192 |
| 42 `wo_b` | 0 | 0 | 0 | 0 | 61 / 4,096 |

Layer 42's one remaining fused-query mismatch is still strictly better than
the incumbent on all four gates: maximum absolute `5.960e-8` versus
`2.441e-4`, maximum relative `0.004329` versus `0.25`, RMS `2.945e-10` versus
`1.207e-6`, and one mismatch versus five. No tolerance moved.

Raw accepted JSON hashes are:

- layer 2: `9a80026b415e4697eda7edd3110743a15d32c6d34f15ebb47144a3d674c1d441`;
- layer 21: `06869dade1165b1e4c71eaa6b71882b5273f05f6abc85fb0559d9cca324a358f`;
- layer 42: `9d34e35c7e9d31f719b92cc76378cc5c56f4bd5765b617b305d0b3683a1b5dc1`.

`make check` passed all registered tests; the Gemma equivalence target skipped
as expected when its external fixture/model was absent.

## Verdict and next action

The real-boundary no-worse gate is closed positively for the guarded kernel.
F8-2 remains open. Before measuring the protected M curve, install the same
warp-voted replay in the layer-resident probe, add exact query RMSNorm, correct
grouped `wo_a` input selection, and measure the replay counters and new
resource signs. Then run M `{1,2,3,4,8,16}` against the unchanged
82%/81%/64% gates. A guarded scheduler that falls below those gates is a
performance rejection; numerical success cannot rescue it.
