# Experiment 0177 — Inkling decode expert parallelism is rejected

Status: **REJECTED; opt-in experiment is not for promotion.** Static expert
ownership balanced the route on paper and eliminated cache evictions, but the
extra per-device command/barrier cost made the zero-miss path substantially
slower. Heterogeneous execution also changed later routing. Inkling keeps its
one batched six-expert command on the layer device.

## Cost model and gate

At the accepted 128-token baseline, a forward costs about 110 ms. Routed MoE is
the `argmax` at 56 ms/forward, including 31.58 GiB staged in 7.16 seconds over
the decode window; attention is about 34 ms/forward. The candidate gives every
`(layer, expert)` one stable capacity-weighted device owner and runs each
device's selected experts concurrently.

- Target terms: critical-device H2D misses and routed-expert compute.
- Unchanged: selected experts, top-k six, coefficients and accumulation order,
  MXFP4 bytes, total compute, aggregate VRAM admission, and cache policy.
- Increased: up to three activation H2D/D2H command pairs and one host worker
  barrier per sparse layer. No transfer-volume saving was claimed.
- Correctness gate: identical continuation and route sequence, then
  `make check`.
- Rollback: no material end-to-end gain, a slower zero-miss path, or changed
  generation.

## Sequential trace and placement simulation

The trace contains the actual five-token prefill plus 127 decode forwards:
5,280 sparse-layer events and 31,680 routed choices. Expert storage is exactly
13,369,344 bytes. Cache capacities printed by the runtime were
19,959,078,912 / 20,396,507,136 / 13,206,365,798 bytes; the weighted device
schedule was `0,0,0,1,1,1,2,2`.

The simulator reproduced the control's **exact 3,358 runtime misses**, which
validates its LRU and capacity inputs. It projected:

| scheduler | misses | critical miss slots | critical compute slots |
|---|---:|---:|---:|
| layer-local | 3,358 | 3,358 | 31,680 |
| expert-parallel | 3,341 | 2,545 | 17,103 |

That is unchanged transfer volume with projected 1.32x critical H2D and 1.85x
critical compute speedups, so it cleared the pre-implementation gate.

## Runtime result

The implementation leased all selected weights through enqueue/collect, issued
one batch per active owner concurrently, then accumulated collected blocks in
the original route-choice order. The result contradicted the simulation because
the simulation did not price the command/barrier term or pageable-copy
contention:

- 128-token expert-parallel repetitions: `3.889, 3.938, 3.937 tok/s`.
- Restored layer-local scheduler at the same operating point: **9.037 tok/s**.
- Expert-parallel zero-miss repetitions after its cache was populated:
  `17.840, 18.142 tok/s`.
- Existing layer-local zero-miss measurement: about **28.0 tok/s**.
- Expert-parallel staging was 34.95 GiB in a 26.57-second median, while the
  restored layer-local path staged 31.58 GiB in 7.16 seconds.

The heterogeneous schedule moved a quarter of experts to the RTX 5060 Ti.
Although each operation remained within its numerical contract, later router
decisions diverged and the continuation changed. That fails this experiment's
strong generation/route equality gate.

An early control in the interleaved matrix accidentally paid the new worker
barrier even with expert ownership disabled and measured only 3.79--4.51 tok/s.
It is not used as the baseline. Restoring the original direct caller path
returned 9.037 tok/s and exposed the scheduler's real regression. The
implausible control was treated as a defect rather than laundered into a win.

Pinned bounce staging was also falsified cheaply on the layer-local scheduler:
3.374 tok/s and 24.53 seconds of staging. The host copy plus three synchronous
projection uploads costs more than direct mapped staging's 9.037 tok/s and
7.16 seconds.

## Decision

Do not promote expert-parallel runtime code. The result explains why copying
Laguna's scheduler is not automatically correct for Inkling: Inkling already
batches six register-fed experts into one command, so splitting that command
raises `Sigma_serial`; Laguna's smaller per-expert command shape and different
balance priced that trade differently.

The production diagnosis remains the positive one: Inkling is fast when the
expert cache hits (about 28 tok/s), while fresh diverse decode pays roughly
31.6 GiB of H2D over 128 tokens. TP/EP is not the sole vLLM difference. Cache
residency and the cold pageable miss path dominate Strata's observed gap.

## Reproduction

```bash
cmake --build build-release --target strata-inkling-probe \
  strata-inkling-ep-sim -j

build-release/strata-inkling-ep-sim \
  --trace results/inkling-expert-parallel/control.routes \
  --capacities 19959078912,20396507136,13206365798 \
  --schedule 0,0,0,1,1,1,2,2 --all-phases

scripts/inkling_expert_parallel_ab.sh
```

Raw traces and logs remain ignored under `results/inkling-expert-parallel`.
