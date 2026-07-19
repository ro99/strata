# Issue draft: production-scale batched prefill for DeepSeek

Tracked upstream as [issue #2](https://github.com/ro99/strata/issues/2).

## Title

`feat: add production-scale batched prefill for long-context DeepSeek`

## Body

The DeepSeek runtime now admits arbitrary logical context ceilings up to the
checkpoint limit (1,048,576 tokens), lazily commits large host-cache pages, and
implements the declared learned top-512 index selection for ratio-4 layers
after the 2,048-token index window. Full-model execution evidence currently
covers the first selection boundary.

The runtime now has an accepted bounded layer-major prefill path with multi-row
attention projections and router projections. Page 64 is the measured default;
page size one remains the independent token-at-a-time oracle. Short and first
learned-index-boundary full-model correctness pass, but practical 32k/200k/1m
execution validation remains open. Exact expert execution remains row-at-a-time
within each bounded page.

The batched path must preserve the existing DeepSeek contract: compressed KV
and index state, causal masking, routing and top-k normalization, shared expert
execution, mHC state, and teacher-forcing output.

## Acceptance criteria

- Support batched/paged prefill at 32k and 200k, with a path that can scale to
  1m context without allocating an unbounded temporary tensor.
- Match a token-by-token teacher-forcing oracle within the declared numerical
  contract, including the learned index selection boundary.
- Report prefill separately from decode, including index work, staging,
  synchronization, RSS, VRAM, and cache/index allocation metrics.
- Preserve the current short-request decode behavior when a large logical cache
  is configured.
- Demonstrate median-of-three interleaved measurements and bounded workspace.
