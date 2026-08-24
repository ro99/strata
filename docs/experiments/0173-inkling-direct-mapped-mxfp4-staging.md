# Experiment 0173 — Inkling MXFP4 direct mapped staging removes a harmful host copy

Status: **ACCEPTED.** Inkling MXFP4 cache misses now upload directly from the
resident checkpoint mapping. Copying every already-canonical expert projection
through one reusable pinned scratch buffer made cold decode 1.31x slower.
Inkling NVFP4 keeps the scratch path because it must de-interleave gate/up rows.

## Operating point and gate

- Branch base: `main@e294b4f`.
- Checkpoint: `models/inkling`, `mlx-community/Inkling-Small-mxfp4`, 130.638
  GiB.
- Devices: `CUDA_DEVICE_ORDER=FASTEST_FIRST`, two RTX 3090 24 GiB cards and
  one RTX 5060 Ti 16 GiB, `--devices 0,1,2`, default 0.85 VRAM fraction.
- Workload: five-token raw prompt, 16 generated tokens, 15 decode forwards,
  greedy sampling. Three interleaved process-level repetitions in order
  pinned/direct/direct/pinned/pinned/direct.
- Page state: the full mapping was faulted into RAM before the matrix. Matrix
  arms used `--no-warm`, so they measured mechanism rather than repeating an
  85-second page-cache setup. No arm touched NVMe in its measured window.
- Memory ceiling: 2.59 GiB resident spine plus 49.99 GiB expert cache across
  the three GPUs; the 130.638 GiB checkpoint remains RAM/PCIe dependent.
- Correctness: identical generated token text and identical route census in all
  six arms, followed by `make check`.
- Rollback: reject if total miss service or median decode did not improve
  outside the interleaved spread, or if any generated token/route changed.

The baseline cost model named serial cache-miss staging as `argmax_r`. A cold
16-token arm touched 44.82 GiB of routed weights and staged 19.97 GiB. Its 7.45
seconds of miss service contained only 0.40--0.42 seconds of CUDA allocation,
0.23--0.24 seconds in the CUDA copy calls, 1.08 seconds of upload wait, and 0.13
seconds of kernels. The unaccounted majority included the explicit checkpoint
mapping to pinned-scratch memcpy. The change targets that host-memory and
serial term. PCIe bytes, GPU prepack, kernels, precision, cache capacity, RAM,
and VRAM are unchanged.

## Result

| Arm | Decode tok/s | Total miss stage | Decode-only miss stage |
|---|---:|---:|---:|
| pinned copy 1 | 3.654 | 7.45 s | 3.788 s |
| direct mapping 1 | 4.765 | 5.39 s | 2.777 s |
| direct mapping 2 | 4.811 | 5.35 s | 2.750 s |
| pinned copy 2 | 3.654 | 7.46 s | 3.788 s |
| pinned copy 3 | 3.652 | 7.44 s | 3.788 s |
| direct mapping 3 | 4.783 | 5.38 s | 2.769 s |
| **median pinned** | **3.654** | **7.45 s** | **3.788 s** |
| **median direct** | **4.783** | **5.38 s** | **2.769 s** |

Direct staging is **1.309x** faster end to end and **1.368x** faster on the
decode miss-service term. The arm spreads do not overlap. The CUDA copy call
itself grows to 1.53--1.54 seconds because a pageable `cudaMemcpyAsync` may
block while the driver stages, while explicit upload wait falls to 0.26
seconds. Those labels are not the objective: total miss service falls by 2.07
seconds, proving that removing the separate host copy is positive.

All arms produced exactly:

```text
 Paris. The capital of Germany is Berlin. The capital of Italy is Rome.
```

and the same route census:

```text
fp4_register_fed=4336 moe_fp4_register_fed=1600
```

## Longer route and remaining bottleneck

A separate 128-token non-repeated continuation measured the accepted direct
path at 7.240 tok/s. Over 127 decode forwards it touched 379.51 GiB, staged
31.56 GiB in 10.310 seconds, hit 89.4% of expert accesses, and evicted 166
entries. Routed experts remained `argmax_r` at 11.675 seconds total (92.0
ms/forward), followed by host attention at 4.593 seconds (36.2 ms/forward).

This acceptance does not claim the campaign target is reached. It removes one
real defect; remaining steady-state work requires reducing or overlapping the
roughly 249 MiB/forward of misses and moving Inkling's exact relative-bias
attention off its scalar host loop.

## Reproduce

```bash
scripts/inkling_direct_stage_ab.sh
```

Raw ignored artifacts are under `results/inkling-direct-stage-ab/` and
`results/inkling-decode-profile/long-route.log`.
