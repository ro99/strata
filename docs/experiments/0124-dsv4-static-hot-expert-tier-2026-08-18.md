# Experiment 0124 — a static hot-expert tier on the idle GPU

Status: **placement validated on held-out data; first implementation is
defective and under repair.** The projection was 8.63 -> 10.79 tok/s. The first
implementation measures worse than a 30x regression, which is far outside what
its own added latency can explain, so it is being treated as a defect in the
code rather than as evidence against the design.

## The question this answers

The box has three GPUs. Rank-local TP2 uses the two RTX 3090s and leaves the
RTX 5060 Ti idle. Every previous attempt to use it treated it as a third tensor
-parallel rank and failed, for good reasons recorded elsewhere: experiment 0051
measured adding it to a sequential layer schedule at 0.79x, every shardable axis
of this model is a power of two so nothing divides by three, and
`dsv4_rank_local_topology.cpp` hard-requires exactly two ranks.

Those are all answers to the wrong question. The card's problem is not that it
is idle; it is that its two properties — 16 GB and 448 GB/s — disqualify it as a
peer of a 24 GB, 936 GB/s card. This experiment asks what those two numbers
*are* good for.

## The constraint they have to beat

Decode's dominant term is the routed-expert MoE: 3.449 GB of weight per token,
read from host DRAM by two CPU pools. Measured after experiment 0123's fix, it
is **72.8 ms of a 115.9 ms step (63%)** and runs at **47.4 GB/s aggregate**
against DDR4's 76.3 GB/s peak and experiment 0058's 56.7–60.9 GB/s kernel
bound. So no amount of CPU-side work can take that term below about 45 ms, and
`argmax_r` stays where it is.

The only store on this machine that is not DRAM is VRAM, and the routed set is
143.7 GB against 64 GB of VRAM in total. It does not fit.

## It does not have to fit

Decode routing is concentrated. Over a 127-token decode window at the
production operating point the model touches **120.2 distinct experts per layer
against 243.4 under a uniform-random null** (60 trials) — a **2.03x**
concentration, replicated at 2.00x on a second trace with a different prompt.

The load-bearing question is whether that concentration is a property of the
*conversation* or of the *model*. If it is conversational, only a dynamic cache
can exploit it. If it is the model's, a fixed set chosen once is enough.

Measured by choosing the hottest triplets from trace A — a 3,565-token prompt,
127 decode tokens — and evaluating them against trace B, an unrelated 104-token
prompt, never seen when choosing:

| tier size | triplets | GB | covers trace A (self) | **covers trace B (held out)** |
| ---: | ---: | ---: | ---: | ---: |
| top 5% | 550 | 7.2 | 49.5% | 25.8% |
| **top 10%** | **1100** | **14.4** | 66.0% | **38.6%** |
| top 15% | 1651 | 21.6 | 76.9% | 46.4% |
| top 20% | 2201 | 28.7 | 84.2% | 51.8% |
| random 10% (null) | 1100 | 14.4 | — | **10.4%** |

**38.6% against a 10.4% null is a 3.7x lift on held-out data.** The
concentration is the model's, not the conversation's. The self-measured column
is reported only to show how much of it is overfitting: two thirds of the
apparent coverage does not transfer, and any design costed from the self figure
would have been costed at nearly double its real value.

And 14.4 GB is the size of the card.

## The design

Three tiers of routed-expert residency, by what each store is actually for:

- **Tier 0 — RTX 5060 Ti, ~14.4 GB, static.** The globally hottest triplets,
  chosen offline from route traces, pinned for the process lifetime. No
  eviction, no replacement policy, no coherency traffic, and no cache lookup on
  the decode path — residency is a constant bitmap.
- **Tier 1 — 2x RTX 3090, LRU.** The 17.7 GB of routed-expert cache admission
  already grants and rank-local decode never reads. Conversation-local experts.
- **Tier 2 — host DRAM.** The remaining tail, on the existing CPU kernel.

A layer's six experts split by ownership; each engine computes the ones it
holds and contributes a partial. The rank shard split is over the intermediate
dimension, so summing partials across engines is the same reduction the two
ranks already perform — one device computing a whole expert while the others
contribute zero for it is exact. Partials are 4,096 floats, 16 KB, so the joins
are free at this size (measured: the production NCCL all-reduce is 15.8 us).

Nothing here needs P2P, divisibility by three, a third NCCL rank, or a change
to the two-rank topology. The card's 448 GB/s does not need parity with a
3090 — it needs to not be DRAM, and it is 9.5x DDR4.

### Why static is what makes it implementable

Routing is computed on the host *inside* a `cudaLaunchHostFunc` callback, where
CUDA calls are forbidden, so GPU work for the selected experts cannot be
enqueued before the route is known. That blocks a device-side dynamic cache: its
contents would have to be published to the device every time they changed.

A static map never changes. The device holds the residency bitmap permanently,
runs its own top-k on router logits it already has, and executes its experts
with no host round trip, while the host independently derives the same top-k
from the same logits and runs only the non-resident ones. Tier 0 is therefore
buildable on its own, ahead of tier 1.

## Simulation, held out

Prefill then decode replayed against trace B, with the tier chosen from trace A.
`MoE ms` is DRAM bytes at the measured 47.4 GB/s; VRAM-side compute is ~3 ms and
concurrent. The 41.54 ms non-MoE layer term and 7.42 ms remainder are held
constant — they are a separate lever, not this one.

| design | static | LRU | DRAM GB/tok | MoE ms | step ms | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| today, no expert tier | 0.0% | 0.0% | 3.369 | 71.1 | 120.0 | 8.33 |
| 3090s LRU 17.7 GB only | 0.0% | 62.8% | 1.253 | 26.4 | 75.4 | 13.26 |
| 5060 Ti static 14.4 GB only | 38.6% | 0.0% | 2.070 | 43.7 | 92.6 | **10.79** |
| **5060 Ti static + 3090s LRU 17.7** | 38.6% | 36.3% | 0.847 | 17.9 | 66.8 | **14.96** |
| 5060 Ti static + 3090s LRU 30 | 38.6% | 41.8% | 0.662 | 14.0 | 62.9 | **15.89** |

The tiers **compose better than either alone** — 0.847 GB/token against 1.253
and 2.070 — because tier 0 absorbs the global skew and leaves the LRU free to
hold what is local to the conversation. That is the argument for keeping both
rather than spending all the VRAM on one policy.

## Path to 20 tok/s

Tiering reaches 15.89 tok/s with the non-MoE term untouched. Closing the rest
needs the 41.54 ms layer term, of which NCCL is only 1.36 ms (0123): at 25 ms
the step is 14.0 + 25 + 7.42 = 46.4 ms, or **21.6 tok/s**. Neither lever alone
suffices — a free MoE still leaves 48.96 ms — and both are now costed on
measured, held-out data.

## What landed

`strata-dsv4-expert-residency` (`apps/strata_dsv4_expert_residency.cpp`): reads
route traces, counts decode activations per (layer, expert), and emits the
hottest triplets that fit a byte budget, ordered, with ties broken by index so
the plan is byte-identical across runs on the same traces.

Prefill activations are deliberately excluded. Prefill touches 82% of the set at
3,565 tokens, so counting it flattens the ranking toward uniform and describes a
phase this tier does not serve.

```
$ strata-dsv4-expert-residency --trace <trace>.jsonl --budget 14G \
    --output hot-14g.plan
traces 1 | decode rows 5461 | activations 32766 | prefill rows skipped 153295
budget 15.03 GB -> 1124 triplets (15.03 GB), covering 66.63% of the traces'
decode activations
```

Verified end to end: the emitted plan re-read and evaluated against the held-out
trace covers **38.5%** of its decode activations, matching the analysis that
motivated it, and projecting the step 120.0 -> 92.7 ms, **1.30x**, from tier 0
alone.

This is placement, not prediction. It changes which device holds a weight, never
which experts the router selects, and the charter's advisory-predictor rule is
not engaged.

## Not yet built

The runtime side: static residency loading on device 0, the device-side top-k
and bitmap test, ownership-split dispatch, and the partial join. Tier 0 is the
first increment and stands alone at 1.30x.

## Risks

- Two traces, one model, one machine. The held-out result is a genuine
  cross-prompt validation but both prompts come from this repository's own
  benchmark set. A trace from real served traffic is the next input, and
  `--route-trace` already emits one.
- The 15.03 GB plan is close to the card's 16,311 MiB. Admission must size the
  tier against real free VRAM after context and workspace, and cap rather than
  reject.
- Coverage is measured in activations, not in bytes moved. They coincide here
  only because every triplet is the same size, which is true for this model and
  should be asserted rather than assumed for another.


## Runtime result: first implementation defective

The tier was built and run end to end. It loads correctly -- device weight
uploads rise from 25.2 GB to 40.2 GB, the extra 15 GB being the 1,124 triplets
on device 0 -- and after fixing the collect contract (the command returns one
unsummed row per routed expert, not their weighted sum) it executes without
error and produces output.

It is drastically slower. Against a baseline arm of about 32 seconds of prefill
plus decode, the tier arm was still running after **15 minutes** past model
load and was killed.

That magnitude is itself the diagnosis. The design's own worst case -- 43
synchronous device round trips per token at a few milliseconds each -- is on the
order of 130 ms/token added, which would be a 2-3x regression. A 30x regression
is not that; it is a defect. Reporting a number one cannot explain is not
neutral, so this is recorded as an open defect with the mechanism still
unproven either way, not as a result about the design.

### Why, and what the simulation missed

The simulation costed **bytes**: it moved 38.5% of the routed-expert reads off
DRAM and divided by the measured 47.4 GB/s. It never costed the
**synchronizations** the design adds.

Routing is decided inside a `cudaLaunchHostFunc` callback, where CUDA calls are
forbidden, so the tier hands its work to a worker thread and the callback waits
for it. That wait happens **once per layer, per token** -- 43 times a token --
and each one is a full device round trip on device 0: input upload, kernel,
output download, event synchronize. Worse, the thread being blocked is a CUDA
driver callback thread, so blocking it stalls the rank's own stream for the
duration.

The design tried to hide this by submitting before the host share and
collecting after, so the device work would overlap the CPU work. That window is
exactly what the tier shrinks: serving 38.5% of the experts on the device
leaves only 61.5% of the host work to overlap against, so the more the tier
succeeds at removing DRAM bytes, the less time it has to hide its own latency
behind.

This is the charter's "separate volume from overlap", and the failure is the
same shape as experiment 0025: a mechanism that reduces the volume of a term
while adding to a serial one, justified by an arithmetic that only modelled the
volume. The rule existed; the simulation still did not apply it. **A residency
simulation must cost the synchronizations its dispatch implies, not only the
bytes it relocates.**

### The defect, measured

`strata-dsv4-static-tier-probe` prices one tier call in isolation -- upload one
hidden row, run N resident FP4 experts, download the routed rows, synchronize --
without the 120-second model stage.

| routed experts per call | ms | per expert |
| ---: | ---: | ---: |
| 1 | 0.3343 | 0.334 |
| 2 | 0.6423 | 0.321 |
| 6 | 2.1045 | 0.351 |

It is **linear in the expert count with no measurable fixed cost**, so this is
not handoff latency, which was the obvious suspect and is wrong. It is the
kernel.

One expert triplet is 13.37 MB. At 0.34 ms that is **39 GB/s on a card with
448 GB/s of VRAM bandwidth -- 8.7% of it.** The batch-1 FP4 expert GEMV is the
defect, and it is not specific to this card: 0058 records the same kernel at
44.6 GB/s of a 3090's 936 GB/s.

The consequence is decisive and explains the direction of the result. The CPU
path moves 3.449 GB/token at 47.4 GB/s aggregate; the device path moves the
same experts at 39 GB/s. **The CPU is currently faster per expert than the GPU
is**, so relocating experts onto the device makes decode slower no matter how
good the placement is. At 6 experts a layer the tier costs 2.10 ms where the
host share it replaces costs about 1.65 ms, and 43 layers turn that into 90
ms/token of tier calls against roughly 27 ms of DRAM reads removed.

This is a property of the kernel, not of the placement. The placement decides
*which* bytes move off DRAM; it cannot help if the engine reading them is
slower than the one it replaced. At even 200 GB/s -- still under half the
card's bandwidth -- six experts would cost 0.4 ms, the tier calls would be 17
ms/token, and the projection would hold.

Worth noting where that fix likely lives: the tier device is the RTX 5060 Ti,
which is SM120 and the only device on this machine with **native FP4 tensor
cores**. The kernel it is running was written for SM86, which has none, so it
decodes FP4 in-register and feeds CUDA cores. That is the same gap experiments
0103-0105 closed for FP8 projections on SM86, and it is the obvious first place
to look for the missing 5x.

### Other defects in this implementation

Four, all in the code rather than the idea:

1. **The callback is blocked.** `collect` waits on a `cudaLaunchHostFunc`
   callback thread, which stalls the rank's own stream for the duration. The
   submit-early/collect-late structure was meant to overlap the device work
   with the host share; blocking that thread means it does not overlap at all.
2. **A full device round trip per layer.** `enqueue_deepseek_moe` plus
   `collect_deepseek_moe` uploads the input, launches, downloads every routed
   row and event-synchronizes -- 43 times a token, when the layer's own command
   stream is already going to the device anyway.
3. **The tier's own counters were never surfaced.** `device_nanoseconds` and
   `wait_nanoseconds` exist on the class and reach no JSON, so the 30x has no
   attribution. That is the same dead-instrumentation mistake 0123 found twice
   elsewhere, repeated here.
4. **Nothing was measured at small scale first.** The mechanism went straight
   into a 120-second model load. A standalone probe of one layer's tier call
   would have priced the round trip in minutes.

The correct next step is (3) then (4): instrument, then price the round trip in
isolation, and only then judge the design.

### What survives regardless

The *placement* finding stands and is independent of this mechanism:
concentration is 2.03x, it is the model's rather than the conversation's, and a
14.4 GB set chosen from one prompt covers 38.6% of another's decode activations
against a 10.4% null. The planner and the plan format are landed and tested.

What is in doubt is only this dispatch: serving that set **from a device outside
the rank pair via a blocking per-layer handoff**. Two directions for the repair:

1. The simulation's own table already prefers the other tier. The 3090s' LRU
   alone projects 13.26 tok/s against this tier's 10.79, and those experts live
   on the devices that are already executing the layer -- no cross-device
   handoff, no callback blocking, no extra synchronization per layer. That is
   the mechanism to build next, and this experiment is the argument for
   building it first.
2. Any device-served expert tier must be reachable **without blocking the
   routing callback**. That means the selection has to happen on the device --
   a top-k over router logits it already holds, tested against a residency
   bitmap in constant memory -- so the expert executes inside the layer's
   existing command stream rather than behind a host round trip. The static
   map is still what makes that possible; the blocking handoff was the error.

The tier remains in the tree behind `--static-expert-plan`, defaulted off, so
the defect is reproducible from one binary.
