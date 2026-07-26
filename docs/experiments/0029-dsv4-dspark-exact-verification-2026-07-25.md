# Experiment 0029 — exact DSpark verification and rollback

Status: **closed: exact but structurally unable to break even at this operating
point**. Exact attention batching reduced the attention marginal below the
required threshold, but the complete target graph measured `m=0.659`, above the
binding `m<0.512` gate. Base-model autoregressive generation remains the
default; the DSpark runtime is rejected from promotion.

The implementation and rerun scripts remain on the rejected
`exp/dsv4-dspark-attention-integration` branch. Only this experiment record is
retained on `main`.

## Contract

- Hypothesis: the checkpoint's three-stage, five-row DSpark draft plus
  layer-major target verification can reduce work per committed token enough to
  beat exact batch-1 decode.
- Primary metric: median end-to-end decode tok/s over three interleaved
  repetitions, but only after a one-pair operating-point mechanism screen.
- Correctness: committed tokens, logits, layer/operation hashes, and routes must
  match the non-speculative target; rejected state must not affect later output.
- Memory: 216 GiB host ceiling, 0.85 of free VRAM, bounded block KV, and zero
  decode checkpoint reads.
- Rollback: stop before the repetition matrix on an exactness, I/O, or memory
  failure, or if target verification has no batch amortization. Measure that
  dependency directly before classifying the DSpark mechanism.

The pre-change scalar-cache profile at the 511-token operating point named
routed-weight H2D as `argmax` at 86.25 ms/step. The required block KV path was
therefore re-instantiated in the production screen. There, attention was the
largest aggregate phase (23.58 s over 127 steps), while measured upload wait was
4.84 s. DSpark could not improve this point unless it reduced transfer without
inflating attention, compute, or synchronization.

## Implementation

DSpark is enabled with `--dspark` and a configurable confidence threshold. The
runtime executes all three MTP stages and five draft rows, applies the Markov and
confidence heads, and uses confidence only to shorten target verification.
Sampling uses exact speculative acceptance `min(1, p/q)` and residual
`max(0, p-q)` sampling.

Target verification runs on a forked block-KV sequence. Compressor state,
DSpark main KV, routes, index selections, logit/layer diagnostics, and output
callbacks advance only for the committed prefix. Rejected rows are truncated;
failure before commit restores the original sequence and compressor state. MTP
expert unions affect residency scheduling only, never routing or coefficients.

## Correctness gate

`make check` passes both CTest targets. The real-checkpoint forced-depth run used
confidence threshold zero so every proposal computed and verified the declared
five rows. Against the block-KV reference:

- all eight greedy tokens, committed logits, layer hashes, operation hashes,
  and route-trace bytes matched;
- three target forwards committed seven decode inputs, accepted four draft
  tokens, and exercised two partial rollbacks;
- provisional target KV peaked at 2,392,064 bytes;
- decode checkpoint reads were zero;
- RSS was 159,327,518,720 bytes under the 160,815,682,808-byte admission plan.

Evidence: `results/deepseek-v4-dspark-correctness/`.

## Production mechanism screen

One interleaved pair used the pinned 511-token prompt, 128 generated tokens,
block KV, identical devices/budgets, and the default 0.5 confidence threshold.
The expected arm budget was 3.5 minutes for reference and 4–6 minutes for
DSpark; actual wall times were 3:56 and 4:25. Initialization plus prefill was
175.5/179.4 seconds, so it remained separate from decode. The shorter
correctness run was rejected as a performance proxy because it recorded zero
cache evictions and did not reproduce the production mechanism.

| metric | reference | DSpark | sign |
|---|---:|---:|---:|
| decode seconds / 127 steps | 49.493 | 73.980 | negative |
| decode tok/s | 2.566 | 1.717 | **0.669x** |
| weight H2D bytes | 72,341,520,384 | 92,502,491,136 | +27.9% |
| upload wait | 4.840 s | 6.189 s | +27.9% |
| attention | 23.585 s | 34.693 s | +47.1% |
| MoE | 18.589 s | 24.228 s | +30.3% |
| synchronization | 6.223 s | 8.619 s | +38.5% |
| cache misses / evictions | 16,233 / 16,233 | 20,757 / 20,757 | +27.9% |
| activation D2H bytes | 1,747,585,024 | 3,181,683,712 | +82.1% |

The DSpark arm made 63 target forwards, committed 2.016 tokens per target
forward, accepted 64 draft tokens, cut 58 proposals by confidence, and rolled
back rejected suffixes 33 times. Draft/verification cost was 4.912/68.290
seconds. Provisional KV peaked at 5,531,648 bytes; maximum MTP expert union was
22 and 5,490 expert rows were executed. Outputs matched, decode checkpoint reads
were zero, and memory stayed admitted.

This result rejects the **current unbatched verification graph**, not the
DSpark draft or acceptance mechanism. The reference executed 127 target rows
in 49.493 seconds (389.7 ms/row); DSpark executed 188 target rows in 73.980
seconds including draft work (393.5 ms/row). Its 2.984 verified rows per target
forward therefore cost almost exactly 2.984 single-row forwards. With 2.016
committed tokens per forward and 4.912 seconds of draft work, the observed
0.669x follows mechanically from absent target batching.

For `c(k) = 1 + m(k - 1)`, the measured `k=2.984` and acceptance require
`m <= 0.512` to break even. At `k=3`, that is equivalent to average per-row
time at or below `(1 + 2 * 0.512) / 3 = 0.675x` the `k=1` row time.

## Target-page dependency gate

The direct `k={1,2,3,4,6}` sweep used a 12-token prompt, three interleaved
repetitions, the same three GPUs and memory budgets, block KV, and zero
checkpoint reads. The median results were:

| page rows `k` | prefill | ms/row | per-row ratio | marginal `m` |
|---:|---:|---:|---:|---:|
| 1 | 4.472 s | 372.7 | 1.000 | - |
| 2 | 4.384 s | 365.3 | 0.980 | 0.960 |
| 3 | 4.307 s | 358.9 | 0.963 | **0.945** |
| 4 | 4.311 s | 359.2 | 0.964 | 0.952 |
| 6 | 4.292 s | 357.6 | 0.960 | 0.952 |

### Operating-point limitation

This sweep does **not** reproduce the production attention mix. Total row time
nearly matches production, but the 12-token prompt removes almost all
context-scaling score and KV-gather work:

| phase | sweep `k=1`, 12-token context | production decode, context about 570 |
|---|---:|---:|
| total | 372.7 ms/row | 389.7 ms/row |
| attention | 71.8 ms/row (19.3%) | 185.7 ms/row (47.7%) |
| MoE | 255.2 ms/row (68.5%) | 146.4 ms/row (37.6%) |
| mHC pre | 37.6 ms/row (10.1%) | 48.2 ms/row (12.4%) |

The projection costs agree within 3%; the missing work is the context-dependent
term:

| attention sub-term | sweep | production |
|---|---:|---:|
| query projection | 26.9 ms/row | 27.5 ms/row |
| KV projection | 10.0 ms/row | 10.5 ms/row |
| output projection | 24.2 ms/row | 23.9 ms/row |
| score plus KV gather | 10.8 ms/row | 123.9 ms/row |

That last term shares key rows across batched queries and is therefore the
largest plausible source of target-row amortization. The cheap prompt reduced
it from 29% of the production step to 3% of the sweep step. The sweep is valid
for its measured context, but `m=0.945` must not be reported as a
production-context measurement. It also ran with no cache evictions: weight
H2D stayed exactly 16,257,122,304 bytes and cache misses stayed 3,648 at every
page size.

### Result and production estimate

At the measured `k=3` point, attention fell from 0.861 to 0.758 seconds, but
MoE remained the largest phase at 3.062 versus 3.019 seconds. Upload wait was
1.048 versus 1.031 seconds and critical-path synchronization was 1.181 versus
1.168 seconds. The per-phase `k=1` to `k=3` ratios imply:

| phase | per-row ratio | marginal `m` |
|---|---:|---:|
| attention | 0.880 | 0.820 |
| MoE | 0.986 | 0.979 |
| MoE prepare | 0.987 | 0.981 |
| mHC pre | 0.963 | 0.945 |
| total | 0.963 | **0.945** |

Reweighting those measured marginal ratios to the production phase mix gives

`m_est = (185.7*0.820 + 146.4*0.979 + 48.2*0.945 + 9.4) / 389.7 = 0.900`.

This is an estimate, not a substitute for a production-context measurement,
but it remains far above the required `m<0.512`. It gives
`c(2.984)=1+0.900*(2.984-1)=2.79`, so the target-only upper bound is
`2.016/2.79=0.72x` before paying the 4.912-second draft cost. That is consistent
with the measured 0.669x.

Generated tokens matched in every arm, all requested page sizes were
exercised, RSS stayed inside the admission plan, and prefill/decode reads were
zero.

### Closed escape hatches

- Shallower `k=2` drafting does not rescue the measured graph. With
  `m=0.945`, `c(2)=1.945`, so even ignoring draft cost the single proposal must
  be accepted more than 94.5% of the time. The production screen accepted 64
  of 125 verified draft rows, or 51.2%.
- Advisory-only DSpark prefetch has no positive bound in this measurement. The
  entire reference critical-path upload wait was 4.840 seconds, while the
  observed draft alone cost 4.912 seconds. The target-page sweep also changed
  neither H2D bytes nor cache misses, so there is no measured transfer saving
  to fund the draft.

## Production-context attention batching re-test

The prerequisite was implemented on a separate branch, then integrated with
DSpark. The exact FlashAttention request now accepts a per-query visibility
mask, so a target page packs the shared sliding/compressed KV union once and
scores every query row in one call without changing each row's visible keys.
Query/KV and output projections use the existing row-batched CUDA paths. The
sparse long-context indexer keeps its exact row-wise fallback.

The real-checkpoint 12-token correctness A/B reduced FlashAttention calls from
516 to 172 while generated tokens, logits, layer hashes, operation hashes, and
routes matched exactly. A later 128-output rollback probe exposed and fixed two
DSpark integration defects: batched rows now capture compressor snapshots, and
the block cache retains the declared five-row rollback margin in addition to
the 128-row attention window. Admission accounts for the extra physical block.
That probe completed 63 proposals and 27 rollbacks with zero decode reads and
bounded memory.

The binding screen used the same 511-token prompt, 128 outputs, block KV,
three GPUs, and memory budgets as the original production result. Both arms
forced the exact CUDA attention path. The base arm changed as follows:

| metric | prior scalar-attention screen | batched-attention screen | sign |
|---|---:|---:|---:|
| decode seconds / 127 rows | 49.493 | 45.452 | **1.089x** |
| attention | 23.585 s | 19.649 s | -16.7% |
| context score plus KV gather | 123.9 ms/row | 93.9 ms/row | -24.2% |
| weight H2D | 72,341,520,384 B | 72,341,520,384 B | unchanged |
| activation H2D / D2H | 1,608,339,456 / 1,747,585,024 B | same | unchanged |
| upload wait | 4.840 s | 4.883 s | +0.9% |
| critical synchronization | 6.223 s | 7.492 s | +20.4% |
| critical GPU kernels | 5.017 s | 5.013 s | unchanged |
| decode checkpoint reads | 0 | 0 | exact contract kept |

Attention remained the largest base phase at 154.7 ms/row, followed by MoE at
146.9 ms/row and mHC pre at 47.1 ms/row. At the measured `k=2.984`, target
verification produced these marginal costs:

| target phase | base ms/row | verified ms/row | marginal `m` |
|---|---:|---:|---:|
| attention | 154.7 | 100.0 | **0.468** |
| attention score timer | 32.5 | 22.2 | 0.520 |
| MoE | 146.9 | 128.4 | 0.810 |
| MoE prepare | 97.8 | 80.8 | 0.739 |
| mHC pre | 47.1 | 39.1 | 0.745 |
| complete target verification | 357.9 | 276.8 | **0.659** |

Thus attention batching met the phase-local threshold, but it did not make the
complete target graph meet the production gate. Verification cost 52.032
seconds over 63 forwards, giving `c=2.308` and
`m=(2.308-1)/(2.984-1)=0.659`. The binding requirement was `m<0.512`.

The DSpark arm remained exact but took 57.865 seconds versus 45.452 seconds:
2.195 versus 2.794 tok/s, or 0.785x. Draft time was 4.976 seconds, accepted
tokens per target forward remained 2.016, provisional KV peaked at 5,531,648
bytes, outputs matched, decode checkpoint reads were zero, and RSS stayed below
the 160,825,396,984-byte admission plan.

This one-pair mechanism screen is a binding rejection, so repetitions two and
three were not launched and the confidence threshold was not changed.

## Final decision

The rejection is stronger than the measured `m=0.659`. Speculative decoding
wins only if committed tokens per verified row exceed target cost per verified
row by enough to pay for the draft. Here:

`a/k = 2.016/2.984 = 0.676`, while even the best achievable target ratio from
the measured phase costs is `c(k)/k = 1.944/2.984 = 0.651`. The entire target
margin is therefore `0.025`, against a draft costing `4.976/45.452 = 11.0%` of
base decode. The margin cannot fund the draft.

Batching cannot materially widen it. Roughly 60% of marginal target cost is
genuinely row-proportional MoE arithmetic, projections, and mHC on this
host-compute-bound decode. Critical GPU kernels account for only 5.013 of
45.452 seconds, so the GPU is idle for about 89% of the step; PCIe wait is only
10.7%. The fixed per-forward weight-read dominance required by speculative
decoding does not exist at this operating point.

Acceptance closes the other direction and is not a runtime lever. A shallower
`k=2` needs more than 94.5% acceptance before draft cost, against 51.2%
measured. The forced-depth correctness run at `k=5` committed seven tokens over
three target forwards, `a/k=0.467`, worse than the production `0.676`; despite
the small sample, it agrees with the confidence head cutting 58 of 63 proposals
as MTP drafts decay with depth. Shallower fails on acceptance, deeper fails on
decay, and the middle fails on marginal cost. DSpark is closed and should not
return at this operating point.

Evidence: `results/deepseek-v4-dspark-screen/`,
`results/deepseek-v4-batched-attention-correctness/`,
`results/deepseek-v4-dspark-retention-fix/`, and
`results/deepseek-v4-dspark-attention-final/`. Re-run scripts are
`scripts/run_deepseek_v4_dspark_correctness.sh`,
`scripts/run_deepseek_v4_dspark_ab.sh`, and
`scripts/run_deepseek_v4_target_page_sweep.sh`.
