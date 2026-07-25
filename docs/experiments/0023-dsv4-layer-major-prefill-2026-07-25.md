# Experiment 0023 — layer-major prefill tiling

## Defect

Prefill ran at 647 ms/token against decode's 576 ms/token on the same run. A
prefill page batches 64 rows through one weight read, so it must be far cheaper
per token than batch-1 decode; a ratio of 1.12 is not a cost, it is a bug
report. The charter rule "treat an implausible measurement as a defect" was
written because this number was reported unexamined across two experiments.

`forward_prefill` nested the loops page-major:

```
for page (56 pages of 64 tokens):
    for layer (43):
        block_page(layer, 64 rows)
```

Each outer iteration sweeps every layer's routed experts, so the working set per
page is 43 x 256 x 3 x 4.46 MB = 147 GB against a ~38 GB VRAM cache. The cache
is evicted and refilled once per page.

Measured on `results/deepseek-v4-offload-regime-profile/cache-unbounded`
(3,565-token prompt):

| Prefill | Measured |
|---|---:|
| Demand H2D bytes | 3,366,668,206,080 |
| Routed-expert set | 147.4 GB |
| Overstream factor | 22.9x |
| Evictions | 745,172 |
| Demand wait | 1,355.96 s of 2,308.45 s |

22.9 ~ kLayers is the signature of this nest, not a coincidence: it is the
number of times a page-major sweep re-reads a working set that does not fit.

## Contract

- Hypothesis: visiting layers outermost over a tile of pages streams each
  layer's experts once per tile instead of once per page, reducing prefill
  demand H2D and prefill wall time.
- Bottleneck: prefill demand H2D is `argmax_r` for the phase, 59% of it in
  serial demand wait.
- Sign on other resources: routing, precision, top-k, expert count and all
  arithmetic are untouched. The only added cost is one resident activation tile,
  `tile_tokens x mhc_multiplier x hidden_size x 4 B` (134 MB at 512 tokens).
- Correctness: reordering independent layer/page work changes no arithmetic, so
  output must be bit-identical.
- Memory: RSS must not rise beyond the activation tile.
- Rollback: prefill not faster beyond run variance, or any output byte moves.

## Commands

`prefill_layer_tile_tokens` is a tile width, so the previous page-major nest is
reproducible from the same binary by tiling at the page width. Both arms are one
build; only the tile differs.

```bash
scripts/run_deepseek_v4_prefill_layer_major_ab.sh
```

Prompt 34 sentences (~512 tokens), `--prefill-page-tokens 64`, `--max-new 8`.
Decode is not under test here, so the arm is sized to prefill: about 5 minutes
for the reference arm and 3 for the candidate.

## Result

| | reference (tile 64) | candidate (tile 0) | |
|---|---:|---:|---|
| Prefill seconds | 284.16 | 150.39 | 1.89x |
| Prefill demand H2D bytes | 437,204,287,488 | 104,962,719,744 | 4.17x less |
| Prefill evictions | 87,818 | 13,265 | 6.6x fewer |
| Prefill demand wait (s) | 181.27 | 54.38 | |
| Prefill ms/token | 556.09 | 294.31 | |
| RSS bytes | 148,866,752,512 | 148,863,647,744 | |

Gates, all passing: `tiling_actually_differed`, `generated_tokens_equal`,
`logits_equal`, `layer_hashes_equal`, `operation_hashes_equal`,
`same_prefill_tokens`, `prefill_faster`, `moved_less`, plus `cmp` on
`routes.jsonl` byte-for-byte. `make check` passes.

RSS moved by -3.1 MB, inside noise: the 134 MB activation tile replaces the
per-page buffers and 148 GB of resident weights dominate the figure.

## Operating point

The reduction factor is the tile count, `ceil(tokens / page)`, floored by the
147 GB expert set. At 512 tokens that is 8 pages collapsing to 1 tile, and the
candidate's 105 GB is already near the floor — the whole expert set is touched
once. **This 4.17x must not be transplanted to other prompt lengths.** At 3,565
tokens the reference moved 3,367 GB while the candidate's floor is unchanged at
roughly 105-147 GB, so the reduction there should be far larger and the speedup
correspondingly bigger. That is an extrapolation from the model, not a
measurement, and it is stated here as an assumption to be re-measured, not a
result.

## Decision

Promote. `prefill_layer_tile_tokens` defaults to 0, tiling the whole prefill
range, which is the minimum possible expert traffic. Setting it equal to
`prefill_page_tokens` restores the previous behaviour for regression work.

## Open, not addressed here

Decode remains 576 ms/step (1.74 tok/s) at the 3,565-token operating point, and
that is two separate terms, neither of them this defect:

- MoE prepare 271 ms/step, of which 255 ms is demand wait moving 582 MB at
  **2.28 GB/s**. A micro-benchmark on the identical cold-arena access pattern
  reaches 11.9 GB/s pinned. Per the charter an order-of-magnitude gap against
  the rated figure is a serialization defect, not a bandwidth limit.
- Attention 197 ms/step with `attention_cuda_dispatches: 0` — all 5,461 decode
  dispatches take the host scalar path, because `flash_attention_minimum_rows`
  is 256 and decode presents one row.

Closing the first alone projects ~370 ms/step, at which point attention becomes
`argmax_r`. Each needs its own branch, hypothesis and gate.
