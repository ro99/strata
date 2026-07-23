# Exact incremental KV continuation for multi-turn chat

Status: **implementation and short full-model correctness accepted; performance not classified**.

## Contract

- Hypothesis: when the tokens already committed to KV are an exact prefix of
  the next rendered chat prompt, forwarding only the appended suffix at its
  absolute positions preserves full-reprefill generation.
- Primary metric: second-turn prefill latency and processed prompt tokens;
  generated-token equality is the release gate.
- Correctness gate: strict-prefix, unchanged, changed, and truncated token
  fixtures; `make check`; full-model GLM-5.2 and DeepSeek V4 generation against
  forced full re-prefill; DeepSeek scalar and compact block cache coverage.
- Memory ceiling: existing context and KV admission limits plus one token-ID
  vector bounded by the admitted context.
- Rollback: `--full-reprefill` disables reuse. Any token-prefix mismatch,
  failed request, truncated history, or unavailable learned-index state resets
  the sequence and takes that path automatically.

## Implementation

Each runtime records only tokens actually forwarded into its KV state. This is
important when generation stops at `--max-new`: the final emitted token has not
yet been forwarded and therefore remains part of the next turn's suffix. A
completed request becomes reusable only after all runtime invariants and trace
flushes pass.

The next request still renders and tokenizes the complete retained transcript.
An exact strict-prefix check chooses between reuse and reset. GLM forwards the
suffix through its existing nonzero position base. DeepSeek's bounded prefill
now also accepts a position base and preserves sliding, compressed, learned-
index, and partial compressor state across turns. If a continued DeepSeek
request crosses the learned-index boundary before index state was admitted, it
does one full re-prefill to construct the historical index exactly.

The JSON protocol reports full prompt tokens, processed prefill tokens, reused
tokens, and whether continuation was used. The TUI presents processed prefill
and KV-reuse counts. DeepSeek chat exposes the compact block mode, and the
reusable A/B script interleaves variants when repetitions exceed one.

## Validation

`make check` passed both CTest targets with 98/98 C++ cases, and all 11 TUI
tests passed. Three one-pair, two-turn greedy smokes used frozen runners:

| Runtime/cache | Full turn-2 tokens / seconds | Incremental turn-2 processed + reused / seconds | Generated IDs |
|---|---:|---:|---|
| GLM-5.2 scalar | 44 / 53.138 | 14 + 30 / 21.879 | equal |
| DeepSeek V4 scalar | 27 / 7.108 | 14 + 13 / 4.241 | equal |
| DeepSeek V4 compact block | 27 / 7.171 | 14 + 13 / 4.544 | equal |

The DeepSeek runs used two generated tokens per turn, so continuation also
covered the not-yet-forwarded final-token boundary. Both turns matched their
forced-full generated IDs: `[43, 6360]` then `[3476, 4869]`. GLM matched all
16 emitted IDs across the two turns.

Evidence is under `results/issue8-glm-smoke/`,
`results/issue8-deepseek-scalar-smoke/`, and
`results/issue8-deepseek-block-smoke/`.

These timings are descriptive one-pair observations, not wins. The required
median of three interleaved repetitions, long-context learned-index crossing,
and full teacher-forcing matrix remain open before closing issue #8.
