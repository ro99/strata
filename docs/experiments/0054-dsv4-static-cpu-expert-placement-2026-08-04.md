# Experiment 0054 — static CPU expert placement, rejected by arithmetic before code

Status: **rejected at the screening gate; no runtime code written, no benchmark
run.** The mechanism's break-even requires **32.3 GB/s** sustained from the host
FP4 expert kernel. Experiment 0051 already measured that kernel at **26.23 GB/s**
standalone (28.62 under `numactl --interleave=all`) in its *most favourable*
decomposition. The ceiling is below the floor, so the mechanism cannot win at
this operating point and the planned microbenchmark cannot change the answer.

Predecessor documents live on other branches:
`git show infra/dsv4-external-stack-teardown:docs/experiments/0050-dsv4-external-stack-teardown-2026-08-04.md`,
`...:docs/experiments/0051-dsv4-host-expert-kernel-2026-08-04.md`, and
`git show perf/dsv4-device-attention-chain:docs/experiments/0053-dsv4-device-attention-chain-2026-08-04.md`.

## The mechanism under examination

Place **all** 6 routed experts of every layer on the host, permanently. No VRAM
expert cache, no demand H2D, no admission policy. PCIe carries only the
activation down and the result up (~1 MB/token against 3.449 GB/token of expert
weights). This is what `lk_moe` does in the external stack teardown (0050).

It is genuinely a different mechanism from the `--host-expert-misses` diversion
that 0051 rejected. That policy failed for two reasons that static placement
does remove: the cache starved itself with no decode-time admission, and decode
routed only ~0.9 diverted experts per layer, so one expert had to be split
across a 28-thread pool twice per layer. Static placement gives the pool 6
experts per barrier, which is the ">= 4 diverted experts per layer" regime 0051
named as its own precondition. The reasoning behind the proposal is sound.

It still loses, for a reason neither prior experiment stated in this form.

## Contract

- **Hypothesis.** Running all routed experts on the host removes both
  `moe_prepare` (the expert demand H2D wait) and the GPU MoE execution, and
  replaces them with host compute that is cheaper.
- **Primary metric.** Median decode ms/step at the chat operating point
  (18-token prompt, 64 generated tokens, 3 GPUs,
  `--flash-attention --pin-resident-arena`), three interleaved repetitions.
- **Baseline.** 209.24 ms/step, 4.78 tok/s, Release, measured 2026-08-04.
- **Correctness gate.** Bit-identical generated token ids. 0051's kernel already
  clears this at four fixtures including the production 4096/2048 shape.
- **Kill criterion, declared before the work.** Sustained in-situ host rate below
  **30.5 GB/s** rejects the mechanism without an end-to-end run.

## The screen

Routed-expert weights per token, from `config.json` — 43 layers, top-6, hidden
4096, `moe_intermediate_size` 2048, E2M1 + per-32 E8M0 at 0.53125 B/weight:

```
per triplet   3 x 4096 x 2048 x 0.53125 B  =  13.37 MB
per token     43 x 6 x 13.37 MB            =   3.449 GB
```

Static placement removes `moe_prepare` (64.80 ms/step) and the GPU MoE execution
(48.43 ms/step) and pays host compute instead, so it is competing against
**113.23 ms/step**:

```
break-even = 3.449 GB / 113.23 ms = 30.46 GB/s sustained, in situ
```

That is the gate as the handoff stated it, and it is correct. The screen is what
happens when the *in-situ* rate is decomposed instead of guessed.

### The barrier cost is fixed and known, so the gate can be restated on the kernel

Static placement runs two barrier phases per layer — gate/up, then down —
i.e. **86 barriers per step**, unchanged from the diversion policy, because
barrier count is set by layer count and not by experts per barrier. 0051
measured an empty `parallel_for` at 28 workers at **73.6 µs**:

```
fixed barrier cost = 86 x 73.6 us = 6.33 ms/step, irreducible
```

Solving the gate for the kernel rate the pool would have to sustain:

```
3.449 GB / (3.449 GB / R + 6.33 ms) >= 30.46 GB/s   =>   R >= 32.27 GB/s
```

**The host kernel must reach 32.3 GB/s standalone merely to break even** — and
that assumes straggler cost is exactly zero, which 0051 measured as the
*dominant* overhead term (30.1 ms of the 36.4 ms overhead at 1 expert/barrier;
the empty barrier was only 6.3).

### Measured against required

| host kernel rate | provenance | ms/step for 3.449 GB | vs 113.23 | step | ratio |
|---|---|---:|---:|---:|---:|
| 11.0 GB/s | 0051, measured **in situ** | 319.9 | +206.7 | 415.9 | 0.503x |
| **26.23 GB/s** | **0051, measured standalone, 28 threads** | **137.8** | **+24.6** | **233.8** | **0.895x** |
| **28.62 GB/s** | **0051, standalone + `numactl --interleave=all`** | **126.8** | **+13.6** | **222.9** | **0.939x** |
| 32.27 GB/s | **required to break even** | 113.2 | 0.0 | 209.2 | 1.000x |
| 40.0 GB/s | 0050 planning constant — **superseded** | 92.6 | −20.6 | 188.6 | 1.110x |

The two rows in bold are the only measured rates for the kernel that would
actually run. Both are regressions, and both are **upper bounds**: they credit
the pool with its whole-expert-per-thread standalone rate and charge it nothing
for straggler, neither of which survives contact with 6 experts split across 28
threads.

**Rejected.** Best case 0.939x against a gate of 1.15x.

## Why the proposed microbenchmark could not have changed this

The plan was to re-run `bench_pool.cpp` at 6 experts per barrier and read the
effective GB/s against the 30.5 gate. That measurement is not capable of
clearing it.

The pool's effective rate is bounded above by the kernel's standalone rate, and
the standalone 26.23 GB/s was measured in the *best possible* decomposition —
56 distinct experts over 28 threads, one whole expert per thread, no intra-expert
split at all. Six experts over 28 threads is a strictly worse decomposition of
the same kernel: every thread now owns a fraction of an expert's output rows, so
the arrangement can approach 26.23 GB/s from below but cannot exceed it. Adding
6.33 ms of unavoidable barrier on top puts the ceiling at 25.03 GB/s, against a
30.46 GB/s floor.

Going from 1 expert to 6 experts per barrier is a real improvement — it should
move the in-situ rate from the measured 11.0 GB/s toward the low-to-mid 20s. It
is simply an improvement that lands entirely inside the losing region.

## The defect in the inherited estimate

The handoff's own gate table contains both the rejection and the rescue, and
takes the rescue. Its `26.2 GB/s` row reads `+18.4 ms vs moe total` — a
regression, correctly computed from 0051's measurement — and its `40.0 GB/s` row
reads `−27.0` and carries the recommendation to proceed.

40 GB/s is **0050's planning constant, which 0051 was the experiment that
re-measured.** Its provenance is `scratchpad/fp4gemv.c`: an **L3-resident**
AVX2 W4A16 screen, so a pure compute ceiling with no DRAM streaming, and one
that did not implement the E4M3 activation quantizer that bit-exactness
requires — the omission 0051 records as putting all 512 outputs wrong by 2-5%.
0050 said so itself: *"treat 40 GB/s as the planning constant and re-measure
before building."* 0051 re-measured it and got 26.23 GB/s streaming and
bit-exact. Keeping both numbers in one table lets the superseded one carry the
decision.

This is the charter's *"never reuse a measured constant across operating points"*
in its sharper form: the constant had not merely moved operating point, it had
been directly superseded by a later, more faithful measurement of the same
quantity, in the same document the gate was otherwise derived from.

## Correction: the DIMM population does not multiply this mechanism

The handoff carries the four empty DIMM slots as "the cheapest 2x available
anywhere in this project", landing "exactly on the resource a CPU-expert design
makes `argmax_r`". At today's kernel that is not true, and 0051 measured why.

The host kernel runs at 26-29 GB/s against this machine's measured **76.3 GB/s**
DRAM ceiling — 34-38% utilisation — and 0051 concludes it is **uop-bound, not
memory-bound**, on Broadwell AVX2 with no AVX-512, no VNNI and no AMX. Taking
DRAM from 76 to 153 GB/s relaxes a constraint that is not binding. 0050's own
ranking already says this (item 4: *"Only worth buying after item 3 is what makes
DRAM the bottleneck; before that it changes nothing"*); the handoff inverted it.

The DIMMs become the right purchase once the kernel clears roughly 60-70 GB/s.
It is 26.

## Two costs the estimate omits, both in the same direction

Neither is large enough to matter given the margin, but both make the gate
harder, so no version of this accounting rescues the mechanism:

- **The shared expert.** The 48.43 ms/step of MoE execution includes the shared
  expert, which static routed placement does not remove. The replacement target
  is therefore below 113.23 ms/step, so the true break-even is *above*
  30.46 GB/s.
- **86 added host↔device round-trips per step** for the activation down and the
  result up. Experiment 0053 has just measured that the sign here is not
  obviously favourable: removing 128.8 synchronizations per step made that step
  *slower*, which falsifies the premise that these round-trips cost what their
  count suggests. They are not free either.

## What this does not reject

**DSpark is untouched by this screen, and it is where the argument actually
went.** At depth 5 with ~3.7 accepted tokens per weight read, one 3.449 GB
expert read serves 3.7 tokens on *whatever* placement it runs on. That is the
amortisation the 50 tok/s target requires — 0050's standing arithmetic is that
50 tok/s needs 172 GB/s for the expert term alone against 76.3 GB/s of DDR4 and
~12.5 GB/s of aggregate PCIe, so single-token decode cannot reach it at any
placement.

The handoff proposes planning static placement and DSpark together as "one
mechanism, not two". The dependency runs one way only:

- DSpark amortises the weight read on the **current GPU placement** as well as on
  a host placement, so it does not need static placement to be worth measuring.
- Static placement is a measured regression **today**, so it cannot be carried as
  a co-requisite. It has to be re-priced at DSpark's operating point, where the
  work per barrier rises ~5x and 0051's ">= 4 diverted experts per layer"
  precondition is met by the block size rather than by the placement.

So the order is DSpark first, then re-derive this gate — not the two together.
DSpark keeps its own gate unchanged: it multiplies host arithmetic, and host
arithmetic is `argmax_r`. That is the experiment 0025 shape and it is not
waived by anything here.

## The one measurement that could reopen this

The kernel would need **1.23x** standalone (26.23 → 32.3 GB/s) to reach zero
gain, and roughly **1.65x** (→ 43 GB/s) to reach the 1.15x step-level target that
would justify building it. On a kernel already seven optimisation revisions deep
on AVX2 Broadwell, that is not a credible ask from further tuning.

The single untested lever is **SMT**: 0051 ran 28 threads on 28 physical cores,
and this box has 56 logical CPUs. A uop-bound kernel is the case where SMT
sometimes pays. Against it: 0051 broke the accumulator chain eight ways
specifically to cover the ~25-cycle load-to-accumulate latency, which is most of
what the second thread would otherwise buy.

If anyone regenerates `bench_host_expert.cpp`, run the 56-thread arm — it is one
extra arm on a benchmark that has to be rebuilt anyway. **Under 32 GB/s, this
record stands and should not be re-opened at a different operating point.**

## Cost of this experiment

One screening calculation over constants already measured in 0050, 0051 and the
2026-08-04 Release baseline. No build, no benchmark, no A/B, no runtime code.
The rejected plan was a benchmark regeneration plus a placement implementation
plus an 18-minute A/B.

## Artifacts

None. Every constant is sourced above from 0050, 0051, 0053, or `config.json`.
