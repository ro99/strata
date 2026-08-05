# Experiment 0056 — row-grouped expert-major MoE for prefill

Status: **promoted.** Prefill is **1.262x** faster at a 586-token operating
point (154.0 → 122.0 ms/token) with byte-identical output, unchanged weight
transfer, unchanged cache behaviour and unchanged RSS. Decode is untouched by
construction and measures 1.018x, inside variance.

This is the first mechanism taken from the external stack teardown in
experiment 0055, and it is the one structural difference that teardown found
between the two engines' MoE.

## Contract

- Hypothesis: a prefill page executes its MoE row-major — one `execute_moe` per
  row, each acquiring and reading its own six expert triplets — so a hot expert
  of a 64-row page is read from HBM once for every row that selected it.
  Executing the page expert-major, so each distinct expert is read once and
  applied to all its rows, reduces MoE weight reads and dispatches without
  changing any row's arithmetic.
- Bottleneck at the operating point, measured before the change:
  prefill 92.03 s of a 100.0 s run; within prefill, attention 41.54 s (45.1%)
  and MoE 39.96 s (43.4%), of which 26.60 s is execution and 12.83 s is the
  demand-transfer wait. `argmax_r` for the MoE execution term is HBM weight
  reads: 25,198 row-layers x 80.2 MB = 2,021 GB read in 15.56 s of kernel time,
  **130 GB/s against a 936 GB/s card** — a batch-1 GEMV at 14% of peak.
- Which resource the mechanism reduces: HBM weight-read volume and per-command
  dispatch/synchronisation count.
- Sign on every other resource: PCIe weight H2D **unchanged** (the layer-major
  tile already holds it at the 96.46 GB expert-set floor); host DRAM +1
  activation staging buffer per page; VRAM workspace grows with page work
  items; routing, precision, top-k, expert count and every arithmetic operation
  untouched.
- Correctness gate, stated before the work: **generated token ids and answer
  byte-identical** to the row-major nest. Per-row arithmetic must not move.
- Memory ceiling: unchanged 216 GiB host admission; RSS must not rise.
- Kill criterion: prefill not faster beyond run variance, or any output byte
  moves.
- Rollback: `--row-major-moe-page` restores the previous nest from the same
  binary; deleting `execute_moe_page` and the two kernel pairs restores `main`.

## What was wrong

`moe_page` batched the router across the page and then looped:

```cpp
for (std::uint32_t row = 0U; row < rows; ++row) {
    result = execute_moe(layer, routes[row],
                         input.subspan(row * kHidden, kHidden), ...);
}
```

`enqueue_deepseek_moe` takes `std::span<const float> hidden` — a single row —
and `kMaxDeepSeekRoutedExperts = 6U`. So a 64-row page issued 64 independent
MoE commands per layer per device, each re-reading its own expert triplets. The
external stack's boundary is one call for all rows with the routing table
passed in (`cpu_decode(stream, num_tokens, top_k, hidden, topk_ids,
topk_weights, out)`), which is what lets it group rows by expert.

Two flag-only screens run first, both negative, both informative:

| screen | prefill | note |
|---|---:|---|
| `--prefill-page-tokens 512` on `main` | 0.96x | every phase flat; nothing was dispatch-bound |
| `--block-kv-cache` on `main` | 1.084x | unlocks batched prefill attention (25,198 → 430 FlashAttention calls) but costs 0.806x on decode |

The page-width screen is the load-bearing one: widening the page changed
nothing because the MoE cost was strictly proportional to rows regardless of
page width. That is the signature of a row-major nest, and it is why the fix
had to be the grouping and not the page size.

## The change

`execute_moe_page` groups the page's `(row, rank)` selections by expert per
device, acquires each distinct expert once, and issues one
`enqueue_deepseek_moe_rows` per device. Two new kernel pairs
(`deepseek_fp4_page_{gate_up,down}_kernel` and the FP8 shared-expert
counterparts) put the row loop **inside** the weight-group loop, so a 32-weight
group's nibble unpack and E8M0 scale decode are performed once and applied to a
tile of eight rows.

Bit-exactness is structural rather than tested-in: each row still accumulates
over the same groups in the same order with the same values, `reduce_block`
runs per row, and the final combine still sums the six routed contributions in
rank order before adding the shared expert and rounding. The row tile is a
fixed unrolled eight with a clamped index — a runtime trip count makes the
accumulators dynamically indexed, which spills them to local memory and costs
more than the tail it saves.

## Result

Three interleaved repetitions, one binary, `--row-major-moe-page` the only
difference. 586-token prompt, 32 generated tokens, three GPUs, 216 GiB host
ceiling, 0.95 VRAM fraction, `--flash-attention --pin-resident-arena`.

| metric | row-major (all runs) | grouped (all runs) | speedup |
|---|---:|---:|---:|
| prefill seconds | 88.885 90.216 91.273 | 70.728 71.499 73.418 | **1.262** |
| prefill ms/token | 151.7 154.0 155.8 | 120.7 122.0 125.3 | 1.262 |
| decode seconds | 7.682 7.885 7.925 | 7.662 7.742 8.328 | 1.018 |
| prefill MoE | 38.918 39.295 39.652 | 20.749 20.807 21.431 | 1.889 |
| — MoE GPU kernel | 15.537 15.546 15.546 | 8.580 8.581 8.581 | 1.812 |
| — MoE prepare (demand wait) | 12.231 12.532 12.577 | 8.518 8.605 8.708 | 1.456 |
| prefill attention | 40.098 40.939 41.149 | 39.976 40.476 41.628 | 1.011 |
| DeepSeek MoE commands | 70,483 | 1,290 | **54.6x fewer** |
| MoE kernel launches | 349,681 | 6,450 | 54.2x fewer |
| prefill weight H2D | 96.46 GB | 96.46 GB | **1.000** |
| prefill cache misses | 21,645 | 21,645 | **1.000** |
| RSS | 138.76 GiB | 138.76 GiB | **1.000** |

The arm ranges do not overlap on prefill, so the result is outside run variance.
End to end at this operating point: **98.10 s → 79.24 s, 1.238x.**

**Correctness: all six runs emit identical generated token ids and identical
answer text.** `make check` passes 260/260 (1 skipped).

Weight H2D, cache misses, evictions and RSS are identical to the digit, which is
the check that the speedup is not a transfer or admission trade.

## Why decode does not move, and that being correct

Decode presents one row, so `moe_page` takes the single-row path unchanged. The
measured 1.018x is variance. Nothing in this experiment targets decode, and the
decode path's object code is untouched — which is also why it carries no
regression risk.

## What page width does now

Re-running the width screen on the new nest, since the earlier answer was a
property of the old one:

| page | prefill | MoE | MoE kernel |
|---:|---:|---:|---:|
| 64 (default) | 71.45 s | 21.13 s | 8.58 s |
| 512 | 73.62 s | 19.57 s | 6.03 s |

Widening the page keeps helping the MoE — the group serves more rows, so the
weight read amortises further, and the kernel falls to 6.03 s — but attention
and the query projection get worse by more than the MoE gains. 64 stays the
default. The MoE term is no longer kernel-bound at either width: at 512 it is
19.57 s of which 6.03 s is kernel and 8.27 s is the demand wait.

## Where prefill is now

| term | s | share |
|---|---:|---:|
| attention | 40.48 | **56.6%** |
| — score | 14.48 | 20.3% |
| — output projection | 13.22 | 18.5% |
| — query projection | 9.90 | 13.9% |
| — KV | 2.64 | 3.7% |
| MoE | 21.13 | 29.6% |
| — demand-transfer wait | 8.61 | 12.0% |
| — execution | 12.52 | 17.5% |
| mHC pre + post | 8.50 | 11.9% |
| router + branch norm | 0.91 | 1.3% |

`argmax_r` for prefill is now **attention at 56.6%**, of which only 2.3 s is GPU
kernel time. The next mechanism is there, not in the MoE.

## Next, in order

1. **Prefill attention, 40.5 s, of which ~2.3 s is GPU kernel.** The batched
   path exists but is gated on `kv_cache != nullptr` — `block_page`'s
   `batch_cuda` predicate — so with the default scalar KV cache prefill issues
   one FlashAttention call per (token, layer): 25,198 calls. `--block-kv-cache`
   takes it to 430 calls and 1.084x on prefill, but costs 0.806x on decode
   (consistent with experiment 0032). Decoupling the batched-attention path
   from the block KV representation is worth both.
2. **Decode's 88.9 ms/step demand-transfer wait**, 34.6% of a 249 ms step, which
   the external stack pays zero for. See experiment 0055 for why 0054's
   32 GB/s re-open bar is now known to be clearable on this box.
3. **Decode's 91.5 ms/step attention**, of which 10.0 ms is GPU kernel.

## Artifacts

`kernels/cuda/backend.cu` (four page kernels, `enqueue_deepseek_moe_rows`,
`collect_deepseek_moe_rows`), `include/strata/cuda_backend.hpp`
(`CudaDeepSeekMoeRowGroup`), `src/deepseek_runtime.cpp` (`execute_moe_page`),
`src/cuda_backend_stub.cpp`, and the `--row-major-moe-page` rollback flag.
Run JSON under `results/dsv4-basepass/` (ignored). The A/B driver is throwaway
and lives in the session scratchpad.
