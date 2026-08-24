# Experiment 0170 — Laguna register-fed A/B: routes substituted, wall clock unmoved

> **2026-08-23 correction:** the route census below is process-global and
> includes the 47-row prefill. The 1,307 generic expert projections were
> incorrectly attributed to decode. `LagunaRuntime::sparse_mlp` dispatches
> every `rows == 1` decode layer through `run_batched_experts` and
> `CudaBackend::enqueue_moe`; only the multi-row prefill uses the generic
> per-expert path. The measured 4.20 ms kernel reduction remains valid, but the
> proposed decode-batching follow-up and the claim that experiment 0166 was
> contradicted are withdrawn. A phase-scoped census is required before making
> any future route-volume claim.

Status: **SUBSTITUTION CONFIRMED, NO MEASURABLE END-TO-END WIN.** The census
proves both the generic matmul and the fused MoE batch served the run on the
register-fed route with zero scalar dispatches. The matmul-kernel term fell
14.54 -> 10.34 ms, but per-step wall moved 183.72 -> 182.74 ms, which is inside
the run's own attention variation. One repetition per arm, so no spread estimate
exists and no win may be claimed.

## Arms

`scripts/laguna_regfed_ab.sh`, `models/laguna`, devices 0,1,2, context 128,
`--max-new 16`, one repetition, prompt "The capital of France is".
Build `build-release`, verified Release. 3090s at the owner's production point,
250 W and clock-locked to 1605 MHz.

## Two defects in the first attempt, both mine

The first run reported **1.89x** and it was an artifact. Both are worth
recording because either alone would have produced a false positive.

- **Page-cache confound.** The same expert copy is 0.52 GB/s cold and 6.09 GB/s
  warm (experiment 0169). The control ran first, warming the checkpoint for the
  candidate: weight memcpy read 1.99 GB/s in the control and 6.31 GB/s in the
  candidate, and the step "improved" 352.89 -> 186.27 ms. The script now warms
  the mapping and discards a warm-up run so both arms see the same state. After
  that, memcpy is 6.35 and 6.49 GB/s -- the arms agree, as they must.
- **Wrong phase.** The profiler prints prefill before decode, and the extraction
  read the first match. Laguna prefill runs at a 0% cache hit rate and 418
  generic matmul calls per step -- a separate open defect from 0166 -- so its
  numbers describe nothing about decode. The script now reads only the section
  after `== decode ==`.

## Result, warm and phase-correct

| metric | scalar | register-fed | delta |
|---|---:|---:|---:|
| per-step wall | 183.72 ms | 182.74 ms | -0.98 ms (1.005x) |
| matmul kernels | 14.54 ms | 10.34 ms | **-4.20 ms (1.41x)** |
| moe routed | 95.23 ms | 91.12 ms | -4.11 ms |
| attention | 68.86 ms | 71.08 ms | +2.22 ms |
| weight memcpy | 37.89 ms | 37.28 ms | -0.61 ms |
| cache hit rate | 89.8% | 89.7% | -- |

Census, which is why this is a measurement and not a guess:

| route | scalar | register-fed |
|---|---:|---:|
| `fp4_e2m1_group32` | 19233 | 0 |
| `fp4_register_fed` | 0 | 19608 |
| `moe_fp4_e2m1_group32` | 2068 | 0 |
| `moe_fp4_register_fed` | 0 | 2064 |
| `plain_generic` | 431 | 431 |

The substitution happened completely. The kernel term it targets fell by 4.20
ms. The step fell by 0.98 ms, and attention moved 2.22 ms the other way in the
same pair of runs, so the step delta is not distinguishable from noise. With one
repetition there is no spread to test against, and the campaign's rule is that a
result inside observed variance is not a win.

## Why 1.41x on the kernel term and not the measured 5x

Experiment 0168 measured 5.04x and 6.30x for the fused MoE batch at Laguna's
top-k of 10. The census shows why that rate does not reach the step:

- **1307 of 1410 expert projections per step go through the generic matmul, one
  projection at a time**, not through the fused batch;
- the fused batch serves about **138 dispatches per step**.

At width 1 the same probe measures 2.89x and 2.67x, not 5.04x and 6.30x, because
a single expert gives the split-K heuristic too few warps to cover the device.
So most of Laguna's expert work is taking the register-fed route at its worst
operating point, and `matmul kernels` also aggregates attention and output-head
work that the substitution does not touch. 1.41x on that blended term is
consistent with both facts.

This contradicts experiment 0166's statement that "Laguna decode does not call
generic matmul for its routed experts; it batches the selected experts through
`CudaBackend::enqueue_moe`." The census says the batched path serves under 10%
of expert dispatches. That is recorded here as an observation, not chased.

## Verdict

The register-fed substitution is correct, complete and free on Laguna: it moves
the kernel term the right way, costs nothing measurable, and the model still
generates. It does not pay end to end at this operating point, and this
experiment cannot claim it does.

The larger lever is visible but out of scope here: batching the routed experts
would move ~1307 dispatches per step from the kernel's worst width to its best,
where the same code is already measured at 5.04x and 6.30x.

## Reproduce

```bash
scripts/laguna_regfed_ab.sh results/laguna-regfed-ab
```
Raw arms, ignored: `results/laguna-regfed-ab2/{warmup,scalar,regfed}.log`.
