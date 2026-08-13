# Experiment 0086 — rank-local adjacent-chain correctness rejection

Date: 2026-08-12
Branch: `exp/dsv4-rank-local-executor`
Disposition: **M3 REJECTED/BLOCKED at the A2 correctness gate; no timing arm**

This record closes the M3-A2 attempt after root review. It does not invalidate
the accepted P0 one-layer live-page capability or the M2 exact one-layer
readiness result. The A1 queue commit is preserved as historical evidence and
is explicitly reverted after this documentation checkpoint; it is not accepted
multi-layer readiness.

## Hypothesis and gates

The A1 queued boundary might preserve exact live state across two distinct
adjacent layers (21→22), two persistent rank contexts, and two independent
position-104 physical KV page sets. The required control was a sequential
diagnostic layer21 followed by layer22 from one live prefix/page seed. The
candidate was two `enqueue_chain_layer` calls followed by exactly one
`finish_chain`, with no intermediate host continuation, mHC inspection, or
page reseed.

The binding gates were exact fixture identity and candidate layout, exact
prefix records 0..42, rank equality, page callbacks and historical-page
preservation, terminal residual/weighted/input equality against the sequential
control, and stable repeated queued output. Routes were control-only evidence;
the candidate route trace was intentionally `not_exposed_A3_gate`. No timing,
performance, memory, or full-forward cost claim was allowed.

## Fixture and control evidence

The fixed M1/d4r/d4m inputs remained bound:

```text
d4c 1712565387b3565c983da3860c17d6e0648b25792b57da172fd638738adff24
d4r layer21 da8f59de692b2c63aa8e83a09aabbf39c0aa0f1c829cf2934d0193f66cff5d00
d4m f754e8ebd54a8ae57d16fb77e626434a46e163534a1e2f4380d28fe7a7ea6bdf
layer22 d4r be4ea825facb32e26ae410addca06caa4a42d987495081bf19345f0d8ee4a6a6
```

The corrected candidate contracts were validated from the production
`physical_paged_attention` formulas, not from copied replay assumptions:

- layer21: ratio 128, `compressed_count=floor(105/128)=0`, compressed width
  128, one sliding page, 256 candidates; all compressed-region entries are
  invalid, followed by sliding rows 0..104 and invalid padding;
- layer22: ratio 4, `compressed_count=26`, `kIndexTopK=512`, compressed width
  512, two pages, 640 candidates; compressed rows 0..25, invalid padding,
  sliding rows 0..104, then invalid padding;
- layer22 selection reported `selection_query_dependent=no_at_position104`;
  layer21 had no compressed rows.

Both ranks replayed prefix records 0..42 exactly. The sequential 21→22 control
passed and produced rank-equal control routes
`239,156,216,0,163,179`. Layer21 queued callbacks were independently exact on
both ranks: query, KV, page patch, historical bytes, and restored row checks all
passed.

## A2 falsification

The queued candidate failed during layer22 host preparation with
`query_mismatch` at index 0 on both ranks. Candidate query raw BF16 was
`0xbbb1`; sequential-control query raw BF16 was `0xbb6a`. Candidate rank0 and
rank1 remained equal, so this was not a rank divergence.

The same-stream diagnostic copied layer21 terminal residual, weighted, and
layer-input device state immediately after `enqueue_chain_layer(layer21)` and
before enqueueing layer22. After the single final finish/drain, the copies were
compared with a sequential snapshot taken immediately after diagnostic
layer21:

| state | rank0 mismatches | rank1 mismatches | first index |
|---|---:|---:|---:|
| residual, 16,384 BF16 | 16,302 | 16,302 | 0 |
| weighted, 4,096 BF16 | 4,085 | 4,085 | 0 |
| layer input, 4,096 BF16 | 4,091 | 4,091 | 0 |

Thus the divergence is classified **after queued layer21**, before layer22
preparation, rather than as a layer22 page/KV or query computation defect.
The sequential control passed, but the queued two-command-resident state did
not preserve the same terminal boundary. The A2 gate therefore fails before
timing or resource measurement.

## Raw evidence

All artifacts remain ignored under
`results/dsv4-rank-local-executor/m3-a2-adjacent/`; the concise manifest is
`a2-rejection-manifest.md`. Key hashes are:

```text
a2-preflight.log                    10abcd81cdf3d11b87a0a3b0df41a83e3de110b4765ce20f5e3d12046c19e335
a2-corrected.log                    27f202a86591d3a83635d3b075dc6d64a2c6e932f6a6280973c0810c0db4850a
a2-diagnostic.log                   24ea843f5c709766494516a84be557fe21cf0da2d8ec8fb0a0e83e92f0c92796
a2-scratch-diagnostic.log           03694fbfc9f2ece47e45f2445ad8ab530c6910993c693454d2396cb8e4c8b0aa
a2-scratch-diagnostic-corrected.log 03694fbfc9f2ece47e45f2445ad8ab530c6910993c693454d2396cb8e4c8b0aa
```

`a2-corrected2.log` is byte-identical to `a2-corrected.log`. The A1 historical
one-layer queue evidence remains at
`results/dsv4-rank-local-executor/m3-a1-queued-boundary/a1-chain.log`, SHA-256
`19c5493148e58a49f0ede1cccfa66bba39ec3de32e8be0f10f8708aefcbc21fc`.

## Decision and pivot

The accepted P0 implementation remains the last accepted production capability
(`f565c3b`). A1 commit `3be4da4` is historical/reverted evidence only: its
one-layer queue proof did not establish safe multi-command residency, and A2
exposed the post-layer21 state divergence. The root decision is that safely
multi-command-resident backend workspaces/staging are unbounded and unjustified
under the available exact evidence. No further A2 correction, backend queue
redesign, timing, or full-forward arm is authorized in this branch.

M3 is **REJECTED/BLOCKED before timing**. M4 remains **BLOCKED**. The next
authorized work is a direct same-operating-point Strata-versus-external
`94.282 ms` reference phase-gap reconstruction using the accepted Strata
`151.155686 ms` navigation point. This is a diagnostic phase-gap pivot, not an
equal-scope 94.282 ms pass and not a 10 tok/s claim.
