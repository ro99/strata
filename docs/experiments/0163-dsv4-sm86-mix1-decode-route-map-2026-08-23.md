# Experiment 0163 — where DeepSeek V4 decode actually dispatches

Status: **MIX-1 DECODE ROUTE MAP COMPLETE.** The census now covers the paths
production decode uses, and the picture is settled:

| Route | Per 4 tokens | Per 16 tokens | Nature |
|---|---:|---:|---|
| `fp8_tensor_page` | 129 | 129 | **load-only** — 43 layers x 3, fixed |
| `dsv4_moe_shared_fp8` | 172 | 688 | **per-token** — exactly 43 layers x forward passes |
| `dsv4_moe_tier_fp4` | 0 | 0 | opt-in, requires `--static-expert-plan` |
| everything else, incl. `unsupported` | 0 | 0 | — |

**The shared expert, FP8 E4M3/E8M0 block-128, is the only thing DeepSeek V4
decode dispatches to CUDA per token.** The routed FP4 experts run on the host.

## The route map

Three MoE families exist and only one is DeepSeek V4's:

- `CudaBackend::enqueue_moe` — generic; used by GLM-5.2, Laguna and Inkling,
  and **never called by the DeepSeek V4 runtime**.
- `CudaBackend::enqueue_deepseek_moe` — full device MoE, routed FP4 plus shared
  FP8. Not reached in any measured configuration, because 147 GB of routed
  experts cannot be resident on 2 x 24 GB.
- `CudaBackend::enqueue_dsv4_host_moe_impl` — **the live path.** Routed experts
  are computed on the host; the device runs the shared expert through
  `deepseek_fp8_gate_up/down_kernel`, plus `deepseek_fp4_tier_gate_up/down_kernel`
  when a resident routed-expert tier is installed.

The FP4 tier dispatch is guarded by `state.tier_committed && state.tier_installed
!= 0U`, and the tier is only populated by `dsv4_static_expert_tier.cpp` when
`--static-expert-plan` is supplied. That guard fully explains the zero; it was
read from the source rather than probed with another run.

## What this means for MIX-1

**FP8 has a real production dispatch site and FP4 does not.** MIX-1 requires
"eligible FP4 uses accepted F4 path; eligible FP8 uses accepted F8 path". On
this hardware:

- The accepted **F8** W8A16 path has an obvious target: the shared expert, 43
  dispatches per token, already FP8 E4M3 with block-128 scales — the exact
  format experiments 0158–0159 measured.
- The accepted **F4** path has **no unconditional target**. Its only per-token
  CUDA site is the opt-in resident tier. Everything else is host compute.

That is a scope finding, not a defect: the campaign's FP4 kernel work stands on
its own microbenchmark evidence, but "integrate eligible FP4" needs the owner
to decide whether the routed-expert tier is the intended production
configuration, because it is currently opt-in and off by default.

## Method note

This map cost five model loads that reading the call graph would have answered
in one pass. `grep` for callers of `enqueue_deepseek_moe` shows immediately that
the DeepSeek V4 runtime never calls it. The campaign's own rule — do the
cheapest measurement that can falsify first — applies to static structure as
much as to profiling: **trace the call graph before loading 156 GB.**

## What this does not establish

- No throughput claim; token counts were deliberately tiny.
- The tier FP4 route has never been observed firing. Its guard is understood,
  not exercised.
- `enqueue_deepseek_moe` remains uncensused in practice because no measured
  configuration reaches it.

## Exact next action

Owner decision: is the resident routed-expert tier the intended production
configuration for FP4? If yes, the accepted F4 path substitutes into
`deepseek_fp4_tier_gate_up/down_kernel`. If no, FP4 integration has no
per-token CUDA site on this hardware and MIX-1's FP4 half should be re-scoped.

Independently, the accepted F8 path can be substituted into the shared-expert
dispatch, which needs no such decision.
