# Experiment 0174 — Inkling long-context attention leaves the host

Status: **ACCEPTED FOR 512+ ROWS; THE 32-ROW ARM IS REJECTED.** Inkling now
keeps an exact BF16 K/V ring on the layer's assigned device and applies its
learned relative-position bias in the persistent CUDA attention operation.
The scalar host oracle remains the default below 512 visible rows. No device KV
allocation is made when the admitted context cannot cross that boundary.

## Contract and operating points

- Branch: `exp/inkling-device-kv-attention`, based on
  `main@685d3e4`.
- Checkpoint: `models/inkling`, `mlx-community/Inkling-Small-mxfp4`.
- Hardware order: `CUDA_DEVICE_ORDER=FASTEST_FIRST`; logical devices 0 and 1
  are RTX 3090s and logical device 2 is an RTX 5060 Ti 16 GB.
- Full-model screen: the ordinary six-token prompt, 32 and 64 generated tokens,
  all three devices, expert pages resident in host RAM.
- Long-context mechanism screen: one target-shape Inkling attention layer,
  32 query heads, 8 KV heads, head dimension 128, bias extent 1024, and exact
  BF16 K/V at 512, 4,096, and 16,384 visible rows.
- Primary metric: attention milliseconds per forward at the real shape;
  whole-model decode tokens/s is binding whenever both arms follow the same
  route.
- Correctness gate: the CUDA operation must match the sequential-F32 scalar
  operation within `1e-6`, followed by the full `make check` gate.
- Memory ceiling: exact 16K K/V must stay below 600 MiB over all 42 layers and
  must be charged before expert-cache admission.
- Rollback: an operation mismatch, excess allocation, or a throughput
  regression at the selected operating point.

## Instantiated cost model

At the 128-token host-attention baseline, one decode forward spent 36.17 ms in
attention and 91.93 ms in routed MoE. Routed expert H2D/compute was therefore
the overall `argmax`, while attention was already the second largest term and
grew with context. The mechanism targets host memory bandwidth and serial CPU
work in attention. It adds exact BF16 K/V capacity to HBM, uploads the 16 KiB
query, 4 KiB new K/V row and 128 KiB relative bias per layer, and adds the
device attention kernel. It does not change routed-expert bytes, precision,
router semantics, expert count, or top-k.

The host scan is not a constant. Repeating its 128-token cost at 16K would be an
invalid extrapolation, so the target-shape mechanism was measured directly:

| device | visible rows | host scalar median | device median | speedup |
|---|---:|---:|---:|---:|
| RTX 3090 | 512 | 6.367 ms | 0.181 ms | 35.2x |
| RTX 3090 | 4,096 | 78.097 ms | 1.214 ms | 64.3x |
| RTX 3090 | 16,384 | 825.734 ms | 4.777 ms | 172.9x |
| RTX 5060 Ti | 512 | 6.383 ms | 0.113 ms | 56.7x |
| RTX 5060 Ti | 4,096 | 77.517 ms | 0.575 ms | 134.9x |
| RTX 5060 Ti | 16,384 | 843.534 ms | 2.471 ms | 341.4x |

Each row is the median of three separate processes; each device measurement is
the mean of 20 operations after one warm/correctness operation. Maximum absolute
error was `2.68e-9` at 512, `1.50e-9` at 4K, and `1.38e-9` at 16K.

Inkling has 35 sliding layers retaining 512 rows and seven global layers. At
16K, exact device K/V is therefore:

```
35 * 2 MiB + 7 * 64 MiB = 518 MiB
```

That is charged before routed-expert cache capacity is computed and reported per
device in the probe metrics.

## Why the boundary is 512, not the component crossover

The operation itself crossed over near 32 rows. Turning it on there nevertheless
failed the whole-system gate:

| arm | decode rate | attention per forward | decode cache misses | decode stage |
|---|---:|---:|---:|---:|
| host, 64 tokens | 7.212 tok/s | 24.81 ms | 2,478 | 6.020 s |
| device from row 32, 64 tokens | 6.778 tok/s | 19.93 ms | 2,696 | 6.895 s |

The CUDA and host operations agree within the numerical contract, but their last
bits are not identical. Over dozens of layers that changed the greedy token
trajectory, which changed the expert route and made the cache workload unequal.
The 4.9 ms attention saving is real; the headline whole-model comparison is not
an equal-route A/B, and the 64-token candidate was slower in any case. It is
rejected rather than laundered as a win.

The production threshold is consequently 512. Short commands keep the previous
oracle and do not reserve K/V when their admitted context is at most 512. At the
boundary, one bulk upload initializes the exact ring; subsequent calls upload
only the new row. By then the measured component margin is 35x–57x, and at 16K
the seven global layers alone avoid roughly 5.7 seconds of host attention per
forward on this machine.

## Reproduction

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON
cmake --build build-release --target strata-inkling-attention-probe -j
scripts/inkling_attention_microbench.sh
scripts/inkling_attention_crossover.sh
```

Ignored raw logs are in `results/inkling-attention-microbench/` and
`results/inkling-attention-crossover/`.

## Decision

Keep the persistent relative-bias BF16 KV operation and the conservative
512-row gate. This closes the context-growing host-attention defect, but it does
not close the campaign: on short and moderate diverse routes the current
`argmax` remains routed-expert staging/compute. That term must be attacked on a
separate branch.
