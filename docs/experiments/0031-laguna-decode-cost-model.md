# 0031 — Laguna S 2.1-NVFP4 decode cost model

Branch: `perf/laguna-decode-cost-model` (from `feat/laguna-s21-nvfp4`, because the
Laguna runtime exists only there).

## Why

Decode measured 1.37 tok/s on the first correctness run. Before designing any
mechanism the governing model `τ = max_r W_r/B_r + Σ_serial` has to be
instantiated on this hardware at the real operating point.

## Operating point

3 GPUs: RTX 5060 Ti 16 GiB, 2x RTX 3090 24 GiB. All three link at **PCIe gen3
x16** (rated 15.75 GB/s each; `pcie.link.gen.max` is 3, the gen1/gen2 readings
at idle are downclocking). 251 GiB host RAM, 243 GiB in page cache, so the
99.7 GB checkpoint is warm. Context 512, greedy, prompt 5 tokens, 7-8 decode
tokens, `detailed_cuda_timing` on.

Admitted budgets: 19.31 / 19.31 / 12.54 GiB. Resident spine 2.44 / 2.25 / 2.21
GiB. Expert cache 16.87 / 17.06 / 10.32 GiB = 44.2 GiB against 85.36 GiB of
routed experts, so 51.8% of the expert set can be resident.

## Measured breakdown of one decode step (624.62 ms)

| phase | ms | % |
| --- | ---: | ---: |
| MoE routed | 519.52 | 83.2 |
| ...experts (slowest device) | 514.60 | 82.4 |
| attention | 82.29 | 13.2 |
| ...flash attention | 29.00 | 4.6 |
| ...KV restage (host BF16→F32) | 14.56 | 2.3 |
| ...QK-norm + RoPE (host) | 13.49 | 2.2 |
| ...q/k/v/g projections | 16.02 | 2.6 |
| MoE shared expert | 11.41 | 1.8 |
| MoE router | 3.39 | 0.5 |
| output head | 1.77 | 0.3 |

`argmax_r` is the routed-expert path, and within it the **miss staging** term:
215.85 / 248.25 / 191.11 ms per device per step, against only 56.80 / 50.06 /
34.29 ms of actual expert matmul.

Expert cache hit rate is already **82.3%**; volume is 565 MiB/step. At the link's
rated bandwidth that volume is ~36 ms aggregate. **Transfer volume is not the
bottleneck — the staging mechanism is.** Measured weight-path bandwidth is
1.8-3.3 GB/s against 15.75 GB/s rated: an order-of-magnitude gap, which the
charter classifies as a serialization defect rather than a bandwidth limit.

Matmul kernels total 35.81 ms/step. Compute is nowhere near the bottleneck, so
any mechanism that trades transfer for recompute is negative by construction.

## Mechanism microbenchmark

`strata-laguna-stage-probe`, 192 distinct routed-expert projections (324 MiB),
cold randomly placed slices of the mapping, never the same buffer twice.

| arm | ms/module | GB/s |
| --- | ---: | ---: |
| A `read()` + `cudaMalloc` + pageable H2D (current) | 0.780 | 2.27 |
| B `read()` host heap copy only, no device | 0.382 | 4.11 |
| C `view()` page touch only, no copy | 0.087 | 18.15 |
| D `view()` + arena + pageable H2D | **0.416** | **4.26** |
| E `view()` + arena + pinned bounce buffer | 0.516 | 3.43 |

Readings:

- C against B: the mapping itself delivers 18 GB/s. The heap copy inside
  `read()` is ~0.30 ms/module of pure waste, and `view()` already exists.
- D against A: removing the heap copy and the per-weight `cudaMalloc` is
  **1.9x** on the staging path.
- **E is a falsification.** A page-locked bounce buffer is *slower* than D: the
  extra host memcpy costs more than the DMA saves at these transfer sizes. The
  repo's DeepSeek note that pinning wins 3.5x was measured at a different
  operating point and does not transfer. Recorded as rejected; it would need
  transfers large enough for DMA rate to dominate the extra copy.

`cudaMalloc` also caused a hard `out of memory` on the second repetition of the
first profiling run: ~465 allocate/free pairs per step fragment the device.
The arena fixes correctness here, not only throughput.

## Probe limitation

The probe samples only NVFP4 layers (1.688 MiB/module). Production also misses
on the plain-BF16 routed experts of layers 40-47, where a module is 18.87 MiB.
Measured production average is 2.27 MiB/miss, so the probe under-represents the
large transfers. The A-vs-D ordering is unaffected and larger transfers favour
D further, but the absolute ms/module here is not the production constant.

## Stage 1 decision

Build D: zero-copy `view()` upload plus a per-device weight arena. It reduces
`argmax_r` and adds nothing to any other resource — no extra bytes, no extra
FLOPs, no precision or routing change.

Projected: staging 248 -> ~132 ms, step 624 -> ~508 ms (~1.97 tok/s). That
alone does not reach the 2 tok/s silver bar, so stage 2 is expected to be
needed; it will be chosen from the re-measured `argmax_r`, not from this list.

## Closing the unattributed expert-loop time

A first pass showed ~210 ms/step inside the expert loop attributed to neither
staging nor the matmul call. Two candidate explanations were measured and one
survived.

- **Lock contention: rejected.** Time spent acquiring the per-device cache mutex
  is 0.04-0.07 ms/step. There is one worker per device slot, so there is nothing
  to contend for.
- **Per-layer barrier imbalance: confirmed.** `moe_expert_nanoseconds`
  accumulates `Σ_layers max_device(...)` while the cache counters accumulate
  `max_device Σ_layers(...)`. These differ by Jensen, and the difference is real
  work time, not a counting error: `sparse_mlp` calls `parallel_for` once per
  layer, so the pool **joins 47 times per token** and every layer pays for its
  slowest device.

  Measured: `max_d Σ_layers` = 354 ms/step, `Σ_layers max_d` = 601 ms/step. The
  barrier converts per-device load imbalance into **~247 ms/step of serial
  time**, 35% of the step.

This is a second serialization defect, larger than expected, and it is partly
downstream of the first: a device that takes a miss pays a large serial cost, so
reducing staging cost also reduces the variance the barrier exposes. Stage 1
therefore targets staging first and re-measures the barrier tax afterwards
rather than assuming both are independent.
