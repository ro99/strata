# Experiment 0113 — inline the BF16 helpers on the attention host path

Status: **accepted on its declared gate.** Attention at 2,612 tokens falls
92.520 s to 78.139 s, and the paged-attention host remainder falls 27.659 s to
15.472 s, against a gate of at least 5 s. Device attention time is unchanged at
15.671 s to 15.675 s, which confirms the saving is entirely host-side.

Run by the orchestrator directly; the codex executor was unavailable (usage
limit until 2026-08-21).

## Premise

Experiment 0111 attributed the 2,612-token score bucket to within 0.42 s and
left a 27.659 s paged-attention host remainder as the largest unexplained term.
Source inspection found two scalar passes over the same activation buffer per
(layer, sub-chunk):

- `CudaBackend::dsv4_paged_attention` validates every query value with
  `std::any_of(..., !std::isfinite(v) || bf16_round_f32(v) != v)`. Production
  queries are BF16-valid, so the predicate never short-circuits and the scan
  always walks the whole buffer.
- The upload path converts every query value with `bf16_encode` in a scalar
  triple loop before staging.

Both call `bf16_encode`/`bf16_round_f32`, which were declared in
`include/strata/numerics.hpp` but defined out of line in `src/numerics.cpp`.
The build sets no LTO or IPO, so each element cost a real function call and
neither loop could vectorize.

Total elements per prefill is `rows * kHeads * kHeadDim * kLayers`, which is
0.954 G at 677 tokens and 3.679 G at 2,612 — walked twice.

## Mechanism screen

Standalone microbenchmark at the production shapes, out-of-line versus inlined,
seven samples, median. Preserved under `results/dsv4-0113-inline-bf16/`.

| pass | 2,611 rows, out-of-line | inlined | ratio |
| --- | ---: | ---: | ---: |
| validation `any_of` | 226.21 ms | 168.96 ms | 1.34x |
| BF16 staging convert | 169.82 ms | 51.97 ms | 3.27x |

Scaled across 43 layers: validation 9.73 s, conversion 7.30 s, together 17.03 s
or 62% of the measured 27.659 s host remainder. Predicted saving from inlining
alone was 7.53 s.

## Change

`bf16_encode` and `bf16_round_f32` moved into `numerics.hpp` as `inline`,
removed from `numerics.cpp`. The arithmetic is byte-identical; this is a
linkage change only. Verified at the object level: `strata::bf16_encode` no
longer appears as an external reference in `backend.cu.o`.

## Result, 677 and 2,612 tokens, page 8192, `--kv-host-cache 4G`, untraced

| metric | 0111 @2,612 | 0113 @2,612 | delta |
| --- | ---: | ---: | ---: |
| paged-attention host remainder | 27.659 s | 15.472 s | -12.187 |
| attention score | 63.958 s | 51.363 s | -12.595 |
| attention total | 92.520 s | 78.139 s | -14.381 |
| attention query | 9.379 s | 7.986 s | -1.393 |
| maximum device paged attention | 15.671 s | 15.675 s | +0.004 |

At 677 tokens attention falls 20.616 s to 18.255 s and the host remainder is
3.743 s.

The measured saving exceeds the 7.53 s prediction. The screen modelled only the
two attention passes; `bf16_encode` has eight other callers, so the linkage
change plausibly helps elsewhere. That excess is not attributed and is not
claimed.

Attention marginal falls 37.16 to 30.95 ms/token. Whole-prefill marginal falls
59.19 to 47.66 ms/token, against the reference's 1.14 ms/token.

## What is not claimed

Total prefill moved 192.906 s to 144.946 s, but 32.897 s of that is the MoE
bucket and 26.897 s is expert demand wait, which is the known unexplained
page-8192 variance — 0107 measured a 13.53 s baseline range and 0110/0111
measured demand wait between 14.98 s and 41.88 s on identical 2,612 arms. Only
the attention terms are attributed to this change. The unchanged device
attention time is the control that makes that attribution safe.

## Correctness

Generated IDs `2107, 8777, 1277, 440` at both lengths, identical to every prior
arm. Decode checkpoint reads 0. RSS and per-GPU VRAM unchanged. `make check`
100%, 2/2. The change is a linkage move of identical arithmetic, so bit
equality is structural rather than sampled.

## Open follow-ups, not started

- The validation proves every query is already BF16-valid, so the staging
  encode is a pure truncation of the high 16 bits. Measured truncate-only is
  50.47 ms against 51.97 ms inlined and 169.82 ms out of line at 2,611 rows.
- The two passes read the same buffer back to back and could be fused, removing
  the scan rather than accelerating it.
- Roughly 10.6 s of the original host remainder is still unattributed.
- `moe_prepare_seconds` was 41.84 s at 2,612 in 0111 and remains unexamined.
