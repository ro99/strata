# 0049 — Kimi-K3 decode: instantiating the cost model, and what it cost not to

**Date:** 2026-08-03
**Branch:** `feat/kimi-k3-storage-agnostic-and-gpu`
**Status:** closed. Decode is storage bound at ~80% of the step. The dominant
lever is hardware, not software.

## Summary

Twenty-three commits of optimization, correctness, and instrument repair. The
result that matters was the last measurement taken, and it invalidates the
premise most of the others were chosen under:

```
warm decode step, one token, per phase
  attention        6,006 ms   16.2%
  feedforward     30,895 ms   83.2%   <- argmax_r
  residual mix        39 ms    0.1%
  head               175 ms    0.5%
  accounted       37,115 ms   96% of the measured 38.6 s step
```

Feedforward is not compute. The routed matvecs measure 1.43 s/token directly and
the block's dense parts about 1.1 s, so ~2.5 s is expected against 30.9 measured.
The missing 28 s is **storage**: each step advances the position, the hidden
state differs, the router selects a different set, and 129 GiB of resident
experts do not help across steps. Every step pulls its own 24.06 GiB from SATA,
which at the measured 0.8 GB/s is ~30 s.

All the compute work below is real, measured, and A/B'd. Together it moved
**1.4 s of a 38.6 s step**.

## The governing failure

`CLAUDE.md` step 1: *instantiate the model — measure `B_r` and `W_r` for every
resource at the real operating point, emit the per-phase breakdown of a step, and
state which resource is `argmax_r`.*

That was done rigorously for isolated components and skipped for the step. Every
conclusion about where time goes — including two public reversals — was drawn
from measurements that together covered **6% of the step**. The breakdown cost
one model load that had already been paid for.

## What was measured, and what held

### Kernels (all direct A/B, all survive)

| change | before | after | commit |
|---|---|---|---|
| MXFP4 sign-bit branch removed (bit-exact) | 6.39 s/token | 2.51 | `f245f7d` |
| MXFP4 AVX2 + chunked row dispatch | 2.51 | 1.46 | `12e72fc` |
| BF16 dense spine AVX2 | 5.41 | 3.65 | `08a6229` |
| MoE expert-grain dispatch | 1.95 | 1.43 | `c08f3d0` |
| per-worker task addressing (SMT collision) | 2.89 @56w | 1.20 | `7847c65` |

The MXFP4 kernel was branch-bound, not bandwidth-bound: 3.76 GiB/s against ~100
GB/s of memory bandwidth is a 25x gap, which the charter says to read as a bug.
Weight nibbles are random, so a predicate on the sign bit mispredicted about half
the time — two mispredictions per byte at ~15 cycles matched the measured ~33
cycles/byte.

The SMT finding is the one worth keeping. A 16-lane dispatch onto a 56-thread
pool ran at exactly 2.0x the cost of a 16-thread pool, because a shared queue
let hyperthread siblings of already-running lanes win tasks: 16 lanes on ~8
physical cores. Addressing task *i* to worker *i* fixed it.

### Correctness: gates 5 and 6 were unsound, not failing

Gate 5 compared per-layer hidden states against a reference computed at BF16
activation precision while the runtime carries F32. A control run settled it:

| comparison | prompt layers routing differently | decode |
|---|---|---|
| runtime vs BF16 reference | 77 / 92 | 74 / 92 |
| **BF16 reference vs F32 reference** | **82 / 92** | **77 / 92** |

The reference disagrees with itself more than the runtime disagrees with it. A
top-16-of-896 router is discrete and non-Lipschitz, so it amplifies a 0.4%
activation difference instead of attenuating it. The prediction that error would
be flat in depth argued from the residual stream being additive and ignored that
a discrete selection sits between the additions.

Gate 6's decode arm asserted greedy equality across a 0.3125 logit gap on a 14.6
scale against 1.146 of pass noise — it was testing the arithmetic's tie-break.
Now gap-aware.

Output agrees everywhere: prompt 488, decode 810, across BF16 reference, F32
reference, and runtime. **The runtime reproduces its reference.**

### Storage policy

`forbid_nvme_residency = (model == KimiK3)` refused a run that read its routed
set from NVMe, so the fast device was the one configuration the planner would not
admit. Removed (`0c2d874`). Which block device backs the checkpoint sets
`B_storage`; it changes the reported cost, not legality. The endurance property
the runtime owes is that it never *writes* a derived copy, which holds on every
device.

### Prefill

`prefill_page_tokens` was 64, justified by a comment stating the inverted
conclusion — that page cost "stops falling" once a page saturates the routed set.
Once saturated the *total* stops rising, so per-token cost is `saturated_set / P`
and keeps falling as 1/P. Raised to 512 (`914683b`): 14.34 -> 2.63 GiB/token,
1.7x -> 9.1x cheaper than decode. Storage volume is exact from the contract;
wall time is still unmeasured.

## Rejections

- **Per-expert token batching** (`2440487`). Cost per (expert, token) is flat
  from 1 to 64 tokens — DRAM and L3 cost the same, so the routed matvec is
  arithmetic bound at ~13 GiB/s, not memory bound.
- **Atomic completion counter** in the worker pool. Cuts N lock-and-notify pairs
  to one, measured identical at every width. Reverted rather than merged for
  tidiness: a concurrency change to a shared component with no measured benefit
  is pure risk.
- **The DSpark draft model** (`models/kimi-k3-draft`, present, 5 dense layers,
  4.5 GB). Screened three times, and the history is instructive. First screen
  rejected it on routed-expert reuse: the captured trace gives a median 54-expert
  union from 64 draws against 61.8 for independent routing, worth 1.19x. Second
  screen reversed that after measuring the dense spine amortizing 3.63x at K=4.
  The phase breakdown killed it again and confirmed the first: the spine is 3.72
  s of a 38.6 s step, and the dominant term is routed-expert *reads*, which
  amortize 1.19x. **A term worth 10% of the step was allowed to reverse a
  correct rejection.**

## The instrument

Three commits repairing the probe (`699aaf4`, `5f3a092`, `431db56`), after it
produced 1.5x run-to-run variance on identical binaries and I read a grain change
as first a 1.71x win and then a loss. Causes: pools allocated by first touch on a
two-socket box, a selector shared with staging so each arm walked different
experts, min/max spread over five samples on a heavy-tailed cost, and a cold
governor penalising whichever arm ran first.

After: NUMA interleave before any allocation, per-arm seeded selector, quartile
spread over fifteen samples, and a steady-state settle before every arm. Dense
arm reproduces to three significant figures.

It also retracted a finding: there is **no two-socket penalty**. Interleaved, the
dense spine runs 28.6 GiB/s at 28 workers against 18.2 at 14. The earlier
conclusion was a first-touch artifact read through the noisy arm.

## What to do next, in order

1. **Buy the NVMe.** 24.06 GiB/token at ~3.5 GB/s is 6.9 s against 30, taking the
   step from 38.6 to about 15 s before anything else changes. Nothing in software
   on this machine competes. See `hardware/nvme-upgrade.md`; note that the
   one-drive-versus-two arithmetic there assumed uniform routing and should be
   re-derived against the captured trace.
2. **Arena replacement policy.** Worth something for the first time: 24.06
   GiB/token against 129 GiB resident means a policy anticipating the next
   step's selection converts misses to hits. The route trace now exists and the
   simulator is the right place to screen it before any runtime code.
3. **Attention, 6.0 s and never profiled internally.** The next software target,
   ahead of any further MoE work.
4. Prefill wall time at 64 vs 512, which the page change predicts but has not
   measured.

## Open defects

- The 56-worker block arm reads 1.20 in sequence and 1.44 alone; unattributed.
  Candidates are thread churn from pools of other widths and allocator state.
- The NVMe write gate samples a 30 s idle control against a 975 s run window, a
  32x ratio. Background writes here are bursty, so one flush trips a 2x
  threshold.
- `host_workers` still defaults to hardware concurrency. Width matters less after
  `7847c65`, but the default should come from an end-to-end measurement.
- Gate 5 is retired as formulated and its replacement — routed-set agreement
  against an F32-activation reference — reports but does not assert.

## The lesson, stated plainly

The cheapest measurement in this session was the one the charter puts first, and
it was taken last. Four of the twenty-three commits are instrument repair or
retraction. Two conclusions were reversed in public and one of those reversals
was itself wrong. Every one of those costs traces to the same omission: a step
was never profiled, so components were used as proxies for it, and a proxy
covering 6% of a step will mislead about the other 94%.
