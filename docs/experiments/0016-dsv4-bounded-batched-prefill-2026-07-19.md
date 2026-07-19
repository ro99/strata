# DeepSeek bounded batched prefill

Status: **accepted; page 64 is the measured default and page 1 remains the
independent oracle**.

## Contract

- Hypothesis: batching resident attention projections across bounded prompt
  pages materially reduces attention preparation and end-to-end prefill time
  without changing exact causal state.
- Primary metric: median 2,054-token learned-index-boundary prefill tokens/s
  over three interleaved repetitions.
- Correctness gate: page-size-1 token-at-a-time oracle agreement for generated
  tokens, logits, every layer and operation hash, ordered route traces,
  learned-index counts, finite values, and zero checkpoint reads.
- Memory ceiling: 216 GiB admitted host memory, 0.85 VRAM fraction, the
  existing 256 MiB per-device workspace, and page-local temporary state
  independent of the logical prompt length.
- Rollback: select `--prefill-page-tokens 1` on any oracle disagreement,
  unbounded allocation, I/O-contract violation, or repeated regression.

## Implementation

The DeepSeek executor accepts `--prefill-page-tokens N` in `[1, 512]`, with a
default of 64 after the matrix gate. Page 1 remains the independent
token-at-a-time oracle. Each page owns only its hidden, residual, reduced,
branch-output, routing, and attention-projection state. Pages and attention
rows advance in increasing causal position. Decode remains token-at-a-time.

Within a page, mHC and branch work execute layer-major. The MoE router is one
multi-row CUDA projection per layer/page. Attention `wq_a`, `wq_b`, and `wkv`
are also multi-row projections; query/KV normalization is batched, while
compressor, learned-index selection, cache writes, causal scoring, grouped
`wo_a`, `wo_b`, and exact routed/shared expert execution retain row and rank
order. Optional route events are buffered only for the current page and
emitted in the original token-major order. Diagnostics are restored to
token/layer order before their aggregate hashes are finalized.

Page size 1 enters the original `forward_token` loop and remains the independent
runtime oracle. JSON phase metrics expose page count, maximum page rows, and
maximum page-owned workspace bytes.

## Short full-model correctness pair

The reusable script
`scripts/run_deepseek_v4_bounded_prefill_correctness.sh` ran a 213-token prompt
and two requested output tokens on devices `0,1,2`. Page 1 and page 64 matched:

- generated token IDs;
- every logit diagnostic;
- all 9,202 layer hashes and all operation hashes;
- the complete byte-identical ordered route trace;
- zero prefill and decode checkpoint-read bytes.

The oracle used 213 pages and reported 65,536 bytes of page-owned workspace.
The candidate used four pages, reported a 64-row maximum, and 19,267,584 bytes
of page-owned workspace. Both paths reported finite logits and equal projection
rows; the candidate reduced attention projection calls from 27,477 to 516.

| metric | page 1 oracle | page 64 candidate | change |
| --- | ---: | ---: | ---: |
| prefill seconds | 71.335 | 60.056 | -15.8% |
| attention query seconds | 5.881 | 3.576 | -39.2% |
| attention KV seconds | 1.904 | 1.252 | -34.3% |
| attention score seconds | 5.705 | 5.631 | -1.3% |
| attention output seconds | 5.390 | 5.182 | -3.9% |

This is one non-interleaved pair and is descriptive only. Artifacts are under
`results/deepseek-v4-attention-page-correctness/`.

## Learned-index boundary pair

The same script ran with
`PROMPT_REPETITIONS=2050 MAX_CONTEXT_TOKENS=32768`, producing a 2,054-token
prompt and one decode step. Page 1 and page 64 again matched every short-gate
artifact and additionally reported identical learned-index work:

- 63 prefill queries;
- 32,319 candidates;
- 32,256 selected positions;
- nonzero post-boundary decode index work;
- zero prefill and decode checkpoint reads.

| metric | page 1 oracle | page 64 candidate | change |
| --- | ---: | ---: | ---: |
| prefill seconds | 502.651 | 484.278 | -3.7% |
| attention query seconds | 56.207 | 35.303 | -37.2% |
| attention KV seconds | 18.678 | 13.413 | -28.2% |
| attention index seconds | 4.062 | 4.260 | +4.9% |
| attention score seconds | 127.388 | 128.620 | +1.0% |
| attention output seconds | 50.805 | 51.797 | +2.0% |
| projection matmul calls | 264,966 | 4,257 | -98.4% |
| projection matmul rows | 264,966 | 264,966 | equal |

This single pair is not the acceptance decision; artifacts are under
`results/deepseek-v4-attention-page-index-boundary/`.

## Three-repetition attention-projection matrix

The boundary matrix interleaved page 1 and page 64 three times with the same
2,054-token prompt, devices, memory, context, and decode settings. All runs
completed with identical generated tokens and zero decode checkpoint reads.

| median metric | page 1 | page 64 | change |
| --- | ---: | ---: | ---: |
| prefill seconds | 485.377 | 444.659 | -8.4% |
| prefill tokens/s | 4.232 | 4.619 | +9.2% |
| attention query seconds | 56.003 | 34.785 | -37.9% |
| attention KV seconds | 18.687 | 13.016 | -30.4% |
| attention index seconds | 4.089 | 4.075 | -0.4% |
| attention score seconds | 127.924 | 128.015 | +0.1% |
| attention output seconds | 50.973 | 51.315 | +0.7% |
| projection matmul calls | 264,966 | 4,257 | -98.4% |
| projection matmul rows | 264,966 | 264,966 | equal |

The CUDA phase reported equal activation H2D/D2H bytes, while total matmul
calls fell from 2,639,517 to 2,291,905 and synchronization calls fell from
1,045,954 to 698,312. Critical-path synchronization was 48.233 s versus
47.065 s; kernel time was 81.991 s versus 79.758 s. Median maximum RSS was
145,355,920 KiB versus 145,368,760 KiB, and per-device cache occupancy was
identical. This is a material, attributable end-to-end improvement beyond
observed run variance, so page 64 is accepted as the default. Artifacts are
under `results/deepseek-v4-attention-page-matrix/`.

## Remaining gates

Production-scale 32k/200k/1m execution remains open. Host scalar learned-index
cost remains a known quadratic limit for 200k and requires the separate
Lightning Indexer work.
