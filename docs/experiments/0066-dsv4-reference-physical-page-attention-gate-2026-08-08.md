# Experiment 0066: DeepSeek reference physical-page attention gate

## Status

**Rejected at the declared native CUDA correctness gate.** A fresh external
position-2,175 trace proved that the captured physical FP8 DS-MLA main and SWA
pages reconstruct every valid BF16 candidate bit exactly. A native fixed-address
CUDA candidate was fast and performed no decode H2D/D2H, but real layer 42
repeatedly reached `0.078125` maximum output error against the installed
reference, above the predeclared `0.0625` bound. Neither BF16 tensor-core QK nor
a source-shaped 128-candidate blocked finish changed that error. The failed CUDA
candidate and build target were removed; no runtime integration or device-1
expansion followed.

This rejects Strata's attempted native attention implementation, not persistent
FP8 KV or the external architecture. The bit-exact physical-page result is
retained as evidence. The complete device chain is now blocked on reproducing
the installed SM86 attention finish's actual compiled numerical association; it
must not silently relax the accepted layer oracle.

## Hypothesis and operating point

The bounded whole experiment remains a device-resident hidden/attention/KV/mHC
chain with stream-ordered CPU-MoE boundaries. This gate tested the smallest live
component that could falsify native integration: real physical FP8 pages at
fixed device addresses through sink-aware attention output.

```text
tau = max_r W_r / B_r + sum_serial

routed CPU standalone floor              87.879 ms
accepted fixed-buffer envelope        2.182--2.318 ms
remaining margin to 100 ms            9.939--9.803 ms
current serialized non-MoE path         130.145 ms
current total                           244.812 ms/token
target                                  <=100.000 ms/token
```

- **Target term.** The 71.140 ms attention share inside the serialized 130.145
  ms non-MoE/handoff term. Routed CPU was not modified.
- **Primary metric.** Median captured physical-page-to-attention-output graph
  span, conditional on the complete correctness gate.
- **Correctness gate.** Zero BF16 materialization mismatches and attention at
  each real early/middle/late fixture within 0.0625 maximum, 0.002 mean, with
  no non-finite values. Exact SM86 mode has no fallback.
- **Memory ceiling.** Host below 216 GiB, 0.95 per-GPU VRAM admission, and the
  previously accepted full-context KV/state slot reservation. No precision
  change, host KV tier, or NVMe dependency was allowed.
- **Resource signs.** The candidate reduced the GPU attention term and decode
  transfers to zero, added fixed HBM workspaces inside the admitted reservation,
  did not change CPU payload, routing, expert work, or storage traffic, and
  added no host compute.
- **Rollback.** Any real layer above the correctness bounds, hidden fallback,
  VRAM admission failure, or projected GPU lane beyond the envelope margin
  rejects the candidate before runtime wiring.

## Cheapest exact physical-page gate

The v2 trace retained only physical pages addressed by the production global
slot lists, rather than exporting an already materialized candidate matrix. The
checker independently decoded the block-major layout: 576 data bytes per row,
followed by the page's eight-byte scale rows. The traced external request took
20.924 seconds after server admission.

| Layer | Ratio | Valid compressed rows / pages | SWA rows / pages | BF16 mismatches | Independent attention max / mean |
|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 512 / 9 | 128 / 2 | **0** | 0.017702 / 0.000694 |
| 21 | 128 | 17 / 9 | 128 / 2 | **0** | 0.019200 / 0.000698 |
| 42 | 4 | 512 / 9 | 128 / 2 | **0** | 0.044325 / 0.000922 |

All three physical-page reconstructions are bit exact. Learned ratio-4
selection also remains an exact 512-element set at layers 2 and 42. This
strengthens experiment 0064: page addressing and dequantization are not the
source of the remaining native CUDA mismatch.

## Native candidate results

The removed SM86-only probe reserved the complete slot-0 KV/state envelope,
uploaded pages and inputs once, captured fixed-address page materialization plus
attention, and synchronized once per measured replay. Each reported median is
from eleven repetitions after three warm-ups.

| Candidate / layer | Complete median (min--max) | Page mismatches | Attention max / mean | Decision |
|---|---:|---:|---:|---|
| scalar/tree QK and global-normalized finish, layer 2 | 0.101248 (0.100352--0.103360) ms | 0 | 0.015625 / 0.000526 | component pass |
| same, layer 21 | 0.032768 (0.032768--0.034816) ms | 0 | 0.031250 / 0.000326 | component pass |
| same, layer 42 | 0.100352 (0.100352--0.103424) ms | 0 | **0.078125** / 0.000761 | **reject** |
| BF16 WMMA QK, layer 42 | 0.056320 (0.056128--0.058368) ms | 0 | **0.078125** / 0.000761 | **reject** |
| WMMA QK plus 128-candidate blocked finish, layer 42 | 0.136064 (0.135168--0.139264) ms | 0 | **0.078125** / 0.000761 | **reject** |

The unchanged layer-42 error across all three arithmetic shapes localizes the
failure beyond physical-page decoding and disproves the initial attribution to
the QK reduction. The source-shaped finish also raised its value phase from
about 0.041 to 0.119 ms without improving correctness. This is a negative gate,
not variance: the output error is identical while timing ranges do not affect
the declared correctness threshold.

The final arm reserved 36,694,016 bytes for full-context KV/state and reported a
37,951,788-byte total working set under a 24,033,234,124-byte ceiling. Load H2D
was 446,636 bytes; decode H2D/D2H were zero; the 688,128-byte verification
download occurred after measurement. It reported zero swaps and 16 filesystem
input blocks outside the decode window. Prefill was not applicable.

## Decision

Do not integrate or retain the failed native CUDA attention candidate. Do not
reinterpret its sub-millisecond timing as an accepted speedup. The following
conclusions remain valid:

1. real global-slot page addressing and FP8 DS-MLA dequantization are bit exact;
2. the admitted persistent page representation still matches the external
   stack and remains necessary for the known route to 10 tok/s;
3. Strata has not yet reproduced the compiled SM86 attention finish's numerical
   contract, even though the readable Triton fusion boundary is known; and
4. the complete live device-chain experiment cannot proceed through this native
   candidate without silently weakening correctness.

The next eligible experiment must first isolate the installed finish's compiled
arithmetic using the same real layer-42 fixture—for example, retained score and
finish intermediates or disassembly-level association evidence. It must be a
new bounded numerical-contract gate. Reopening CPU scheduling, wiring this
failed candidate into the runtime, or relaxing the oracle is not authorized by
these results.

Raw traces and JSON remain ignored under
`results/dsv4-reference-device-page-gate/`.
