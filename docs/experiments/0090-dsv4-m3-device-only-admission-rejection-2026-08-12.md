# Experiment 0090: M3 callback-free admission rejection

Date: 2026-08-12
Branch: `exp/dsv4-a2-ownership-screen`
Base: `aba772c`
Disposition: **M3 CURRENT IMPLEMENTATION REJECTED BEFORE TIMING; M4 BLOCKED**

## Question and kill rule

Experiment 0089 completed correctness with live attention page callbacks, but
explicitly left production timing open. This bounded falsifier asked whether
the same 43-layer chain remained exact after fixture row 104 had been
materialized and the diagnostic page callbacks were removed. Only an exact
callback-free warm-up could authorize the planned three timing repetitions.

The predeclared rule was binding: any terminal or per-layer mismatch stops the
arm before timing. No latency, bandwidth, or `tau` value may be reported from
an inexact arm.

## Results

Two setup-sharing arms were preserved:

```text
performance-r1 combined.log SHA-256 6602535bf692da07faeb80564445b3c30febf1e10b44b4231e5007384cd9e709
performance-r1 source.diff  SHA-256 7afb6e3c7204beb45cf90dd3bc8785928be3b37cb052c9d8e5c2f75cb3702206
performance-r2 combined.log SHA-256 d9cd96bdf5e58a6bfcab4603b0b2af07a0f0ada02e89413d068a77339afabc49
performance-r2 source.diff  SHA-256 a87e4c7669230128395b5f31f63395956d08cf8fd29ea1ecc7902ddb88e7e0aa
exit status                         1 / 1
```

Both arms first reproduced the accepted callback-backed result exactly:

```text
weighted hash e1a9a77f0b01a361
input hash    122a716defe84e1b
hidden hash   5017083817dd2848
logits hash   343d766f3f5c0af3
next token    8806
```

Both then failed the first callback-free warm-up at
`timing_vs_sequential`. Therefore zero measured repetitions ran and the
`120 ms`, `36.7 GB/s`, and `30 ms` performance gates were not evaluated.

The second arm tested one narrow hypothesis: reusable prepared-query storage
was guarded until the preceding attention score work consumed it. The result
was identical, so that event-only correction is falsified and was removed.
It is not accepted runtime code.

## Interpretation

The exact callback-backed chain does not establish an exact production-shaped
chain. Removing the callback changes more than instrumentation: the current
callback participates in query/KV visibility and physical-page publication,
while the device-only preparation path has no equivalent declared page-write
contract. A remaining asynchronous ownership defect is also possible. The two
arms do not distinguish those causes, and the failed event-only correction is
evidence against naming prepared-query reuse as the proven root cause.

This is a correctness rejection before performance, not a slow result. It
does not invalidate Experiments 0088/0089 within their callback-backed scope,
and those results must not be relabeled as production timing readiness.

## Decision and cheapest next discriminator

The current M3 implementation is rejected at production admission and M4
remains blocked. Failed probe/runtime changes are excluded from the result
commit; the ignored raw directories preserve their exact source diffs.

If M3 is reopened, the next slice must be a one-layer callback-free
discriminator, not another 43-layer run:

1. pre-materialize the exact current-token physical row;
2. run the same layer once with and without the callback;
3. compare prepared query, physical row, attention reduction, and outgoing mHC
   state at the first differing boundary; and
4. authorize either a device-resident physical-row publication primitive or a
   single proven ownership correction, never both speculatively.

Only one-layer exactness may authorize adjacent-layer and then 43-layer
rechecks. Profiling, graph capture, CPU optimization, teacher forcing, and M4
remain blocked meanwhile.
