# Experiment 0171 — Laguna decode was restaging and rescoring its KV history

Status: **ACCEPTED.** Laguna now keeps its exact BF16 decode KV rings on the
owning CUDA device and uploads only the query plus the new K/V row. At the
steady operating point this cuts the median step from 105.87 to 53.69 ms and
raises decode from 9.446 to 18.626 tok/s, a **1.971x** controlled improvement.
All six arms generated identical text. Relative to the reported 3.75 tok/s
starting point the accepted path is 4.97x faster, not the requested 10x; the
remaining 30 tok/s gap is stated rather than hidden.

## Predeclared contract

- Hypothesis: Laguna's generic compatibility attention is an accidental
  `O(context)` host-to-device restage and serial-score path on every layer and
  decode token.
- Primary metric: median steady-state decode ms/token and tok/s over three
  interleaved old-path/candidate repetitions, with phase attribution.
- Correctness: exact target-shape attention equality, sliding-ring wrap
  equality, identical real-model greedy text, and `make check`.
- Memory ceiling: preserve the admitted host and VRAM budgets. Precision,
  routing, expert count, top-k, and attention semantics may not change.
- Rollback: no material wall-clock gain, any oracle mismatch, or an unbounded
  memory increase.

The target resource was the serial attention-history PCIe and score term. The
old detailed 15-token profile moved 22.17 MiB/token through FlashAttention H2D
at 13.17 GB/s and spent 35.03 ms/token in the flash phase. Expert service was
then the overall `argmax` because the cache was still admitting experts, so the
first cheap A/B correctly showed only 4.125 -> 4.378 tok/s. A longer same-load
repetition reached the steady expert-cache operating point and exposed
attention as `argmax`: 54.35--56.59 ms of a 105.36--106.34 ms step.

The candidate changes no routed-weight, checkpoint, activation, or expert-cache
volume. It adds bounded KV storage on the layer owner and a shared score
workspace, removes repeated KV-history H2D, and replaces three serial score
passes with one row-parallel pass while preserving the sequential F32 dot,
softmax-row, and value-row accumulation orders.

## The defects

The Laguna adapter retained BF16-rounded keys and values in F32 host vectors.
For every new token and every one of 48 attention layers it constructed a
generic `FlashAttentionRequest`; that backend packed and uploaded the complete
visible history again. Transfer volume therefore grew linearly with context.

The exact all-F32 compatibility kernel also calculated each Q/K score three
times on thread zero: once for the maximum, once for the denominator, and once
for the normalized value pass. This was serial across cache rows and repeated
identical dot products. Scores for distinct rows are independent, so the new
kernel calculates every dot once in parallel, then preserves logical-row order
for softmax and value accumulation.

The implementation keeps a two-plane BF16 ring per layer on its assigned
device. Multi-row prefill still uses the existing generic numerical path, then
syncs the completed cache once. Batch-1 decode uploads Q and the new BF16 K/V
row, writes that row at `position % capacity`, and attends directly over the
ring. The F32 host mirror remains authoritative for continuation.

Persistent KV storage is exactly:

```text
12 * maximum_context * 4096 bytes
  + 36 * min(maximum_context, 512) * 4096 bytes
```

That is 48 MiB at the experiment's 256-token admission, and 168 MiB at 2,048
tokens. Score workspace is reused per device and is at most
`72 * maximum_context * sizeof(float)` bytes. Allocation fails explicitly if
the admitted device cannot hold it; there is no precision or host fallback.

## Controlled A/B

`scripts/laguna_device_kv_ab.sh` ran host/device, device/host, host/device.
Every process loaded the same 63.665 GiB checkpoint on CUDA devices 0,1,2 at
VRAM fraction 0.85, context 256, 57 prompt tokens, 79 decoded tokens, and two
generations per load. Detailed CUDA event timing was disabled because normal
serving disables it; the second generation is the steady expert-cache result.
Each arm took under one minute and the complete matrix took under five minutes.

| arm | steady decode tok/s | ms/token |
|---|---:|---:|
| host 1 | 9.404 | 106.34 |
| device 1 | 18.484 | 54.10 |
| device 2 | 18.701 | 53.47 |
| host 2 | 9.446 | 105.87 |
| host 3 | 9.491 | 105.36 |
| device 3 | 18.626 | 53.69 |
| **median host** | **9.446** | **105.87** |
| **median device** | **18.626** | **53.69** |

The candidate is outside both arms' complete spread. At the median pair:

| decode term | host | device | delta |
|---|---:|---:|---:|
| complete step | 105.87 ms | 53.69 ms | **-52.18 ms** |
| attention | 78.70 ms | 26.72 ms | **-51.98 ms** |
| flash attention | 56.59 ms | 4.20 ms | **-52.39 ms** |
| flash H2D | 37.93 MiB | 1.73 MiB | **-95.4%** |
| Q/K/V/G projections | 9.99 ms | 9.83 ms | -0.16 ms |
| host QK norm + RoPE | 5.58 ms | 6.11 ms | +0.53 ms |
| gate + output projection | 5.55 ms | 5.59 ms | +0.04 ms |
| routed MoE | 15.40 ms | 15.23 ms | -0.17 ms |
| shared expert | 5.37 ms | 5.43 ms | +0.06 ms |
| expert cache | 99.4% hit, 8.8 misses | same | unchanged |
| expert weight upload | 14.10 MiB | 14.10 MiB | unchanged |

The result has the predicted shape: the wall saving equals the attention
saving, which equals the flash saving; unrelated work is flat. The first
generation of each process is not used for the headline because page-cache and
expert-admission state differed across arms. It ranged from 5.154--6.439 tok/s
on the host path and 9.109--9.183 tok/s on the device path.

## Correctness and rejected broader fusion

The CUDA fixture uses Laguna's real 72 query heads, 8 KV heads, and head
dimension 128. It compares every F32 output element exactly with the generic
`f32_dot_f32_softmax_f32_accum` oracle and separately exercises the physical
ring wrap used by sliding attention. All twelve real-model generations emitted
one identical continuation.

A broader device attention-chain experiment was built after the KV gate. It
kept projections, norm/RoPE, attention gating, and output projection in one
backend call. It was rejected and removed: the 80-token continuation diverged,
the first generation regressed from about 8.65 to 7.56--7.69 tok/s, and the
steady gain was only about 20.3 to 21.2--21.4 tok/s. A numerically divergent,
route-dependent 4% hot result does not pass the contract.

## TP=2 and the 30 tok/s comparison

DeepSeek's TP=2 is not the sole reason Laguna was slow. The old Laguna-specific
history path alone cost about 52 ms/token, and removing it almost doubles a
controlled steady run. Recent local DeepSeek TP=2 records are about 8.9 tok/s,
not 30 tok/s, so Laguna now exceeds that single-stream point when its expert
working set is admitted.

The checkpoint's own vLLM number is not a same-system batch-1 control. It is
about 600 aggregate tok/s at batch 16 on one RTX PRO 6000 Blackwell with 96 GB,
where the entire 64 GiB checkpoint and KV fit. It benefits from continuous
batching, a newer GPU, and no multi-device layer handoffs.

After this fix, 30 tok/s still requires 33.3 ms/token against the measured
53.69 ms. The new `argmax` is attention at 26.72 ms, followed by routed MoE at
15.23 ms and the shared expert at 5.43 ms. Laguna currently assigns whole
layers to one GPU, so its BF16 spine projections consume only one card's HBM
bandwidth at a time; unlike DeepSeek's rank-local TP=2 path, it does not split
those projections across the two 3090s. Experiment 0172 ran that target-shape
gate. Exact row sharding failed on the layer shapes (1.115x Q, 1.011x O, and a
0.597x shared-projection regression), so TP=2 is not promoted as the next fix.
Only the 588 MiB output head passed at 1.54x, and its complete runtime phase is
too small for that isolated result to move step throughput materially.

Raw ignored logs: `results/laguna-device-kv-final-ab/*.log`.
