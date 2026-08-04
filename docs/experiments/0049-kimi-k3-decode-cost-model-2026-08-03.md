# 0049 — Kimi-K3 decode: instantiating the cost model, and what it cost not to

**Date:** 2026-08-03
**Branch:** `feat/kimi-k3-storage-agnostic-and-gpu`
**Status:** closed with one large item left open. Decode is storage bound at
~80% of the step. A faster drive is the biggest single step, but storage and
compute never overlap, and closing that is worth 18% today and 50% once the drive
lands. See "storage and compute never overlap" at the end, which corrects the
ordering given in the body.

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
   step from 38.6 to about 15 s before anything else changes. It is the largest
   single step; it is *not* the only one, and the claim originally written here
   that nothing in software competes was wrong — see the overlap section at the
   end of this document. `hardware/nvme-upgrade.md`'s one-drive-versus-two
   arithmetic assumed uniform routing and should be re-derived against the
   captured trace.
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

## OPEN, and larger than it looks: storage and compute never overlap

Recorded after the conclusions above, because those conclusions understated it.
The claim "nothing in software on this machine competes with the drive" was
wrong, and the data contradicting it is in this document.

### The defect

```
attention      6.0 s
feedforward   30.9 s
step          38.6 s      <- 36.9 s accounted, additive
```

They add. If storage and compute overlapped at all the step would be
`max(30, 7) ~ 30 s`, not 37. They do not overlap.

`ArenaExpertSource::prepare()` in `src/kimi_k3_runtime.cpp` calls
`reader_->stage(...)` **synchronously**. `kimi_latent_moe_layer` calls it and
then enters the compute loop, so every one of the 92 MoE layers is
read-then-compute: the disk is idle while the layer computes, and the cores are
idle while the next layer's experts are read. That is `Sigma_serial` in the
charter's model, and the charter's rule 4 says to fix overlap before volume
because it is cheaper and usually strictly larger. It was skipped here in favour
of a hardware recommendation.

### Why it is worth more *after* the NVMe, not less

| | storage | compute | step |
|---|---|---|---|
| SATA, serial (measured) | 30 | 7 | **37 s** |
| SATA, overlapped | 30 | 7 | **30 s** (-18%) |
| NVMe, serial | 6.9 | 7 | **14 s** |
| NVMe, overlapped | 6.9 | 7 | **7 s** (-50%) |

On SATA the storage term buries everything and overlap is worth 18%. On NVMe the
two terms are balanced — 6.9 against 7.0 — which is exactly the regime where
hiding one behind the other doubles throughput. **Do not treat the drive as a
substitute for this work; it is what makes this work decisive.**

### Three tiers, cheapest first

1. **Within a layer, no prediction needed.** All 16 experts are known before any
   of them is read — that is why `prepare` takes the whole set. Today the block
   waits for all 16 to stage, then computes all 16. Streaming instead — compute
   expert *i* while expert *i+1* is in flight — requires no oracle and cannot
   change results. Bounded by `min(read, compute)` per layer, about 27 ms against
   326 ms, so ~2.5 s of the step. Small, but it is free correctness-wise and it
   is the scaffolding the next tier needs.

2. **Across layers, prefetch on a predictor.** Layer L+1's routed set depends on
   layer L's output, so it cannot be known exactly. It does not need to be:
   the charter permits prediction that only drives prefetch ("*a predictor is
   advisory; prediction may affect scheduling or prefetch only*"), Strata already
   has `src/route_predictor.cpp`, and a miss costs exactly what today costs. Load
   a predicted top-32 for L+1 while computing L. This is where the full 7 s is
   won, and after the NVMe it is where the step halves.

3. **Volume, which is the only thing that attacks the 30 s itself.** Overlap
   hides compute; it cannot hide 24.06 GiB of transfer. The arena holds 129 GiB
   of 1347 GiB, 9.6% of experts. If routing is skewed — MoE routing usually is —
   those 9.6% may capture far more than 9.6% of selections, and every 10% of
   converted misses is 3 s today and 0.7 s on NVMe.

### What does *not* work, so nobody rebuilds it

**The DSpark draft model cannot be the routing oracle.** The idea is well formed
— prediction driving prefetch is exactly what the charter allows — but the draft
is 5 layers predicting the next *token*, and what prefetch needs is layer 47's
expert set for the *current* token, which depends on layer 46's hidden state. A
5-layer model has no representation of layer 47. The draft was screened and
rejected three times in this document on separate grounds; this is a fourth, and
it is structural rather than economic.

### The measurement that gates all three, and needs no run

Hot-expert skew is computable from the route trace already written to
`/data/strata-results/kimi-k3-fixtures/kimi-k3-backbone.fixture`, in the
`{prompt,decode}.routed.{layer}` arrays. It sizes tier 3 directly, tells tier 2
how large a predicted set must be to cover the true 16, and re-derives the
one-drive-versus-two arithmetic in `hardware/nvme-upgrade.md`, which currently
assumes uniform routing. Minutes, no model load. **Do this first.**

### Corrected ordering

1. Hot-expert skew from the captured trace. No run.
2. Overlap `prepare()` with compute — tier 1, then tier 2.
3. The NVMe upgrade. Still the largest single step, but not the only one.
4. Attention, 6.0 s, never profiled internally.
