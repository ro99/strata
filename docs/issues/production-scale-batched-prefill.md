# Issue draft: production-scale batched prefill for DeepSeek

GitHub issue creation is pending repository authentication in the development
environment (`gh` is not installed and no GitHub write credential is available).

## Title

`feat: add production-scale batched prefill for long-context DeepSeek`

## Body

The DeepSeek runtime now admits arbitrary logical context ceilings up to the
checkpoint limit (1,048,576 tokens), lazily commits large host-cache pages, and
implements the declared learned top-512 index selection for ratio-4 layers
after the 2,048-token index window. Full-model execution evidence currently
covers the first selection boundary.

Production-scale ingestion will need a batched prefill path. The current
full-model ingestion advances one token at a time, so practical and complete
32k/200k/1m execution validation remains future work.

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
