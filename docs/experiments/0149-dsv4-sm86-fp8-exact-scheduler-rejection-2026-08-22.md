# Experiment 0149: SM86 FP8 exact-scheduler rejection

## Result

**The F8-1 layer-resident successor is rejected at the binding guarded-M=1
performance gate.** Restoring the production query RMSNorm and true grouped
`wo_a` operand selection reduces the three-process median from experiment
0146's 83.27% feasibility result to **678.60 GB/s / 80.77%** of the local-read
roofline. All three independent processes are below the unchanged 82% gate.

That retained arm deliberately performs no numerical replay, even though the
real layer-2 and layer-21 fixtures prove that an unguarded reduction cannot be
promoted. It is therefore an optimistic performance upper bound, not a correct
candidate. Every measured correction adds work: packed-fragment FP64 replay
reaches 48.40%, compensated FP32 replay reaches 59.04%, and eight-way FP32
replay reaches 61.72%. A four-accumulator no-replay association reaches 80.35%
but fails the real-fixture no-worse gate. Since even the no-replay upper bound
misses, no member of this scheduler family can open the protected M curve.

This is not evidence that Ampere cannot execute the checkpoint's FP8 format
efficiently. Experiments 0145 and 0146 already show that its BF16 tensor cores,
wide grids, persistent CTAs, and independent-work overlap can clear 82%. The
rejected architecture spends that margin on a dependent normalization
publication and on reproducing the incumbent BF16 boundary. The result rejects
this composition, not the Ampere-native primitive.

## Contract and cheapest falsifier

- **Hypothesis:** Ampere's persistent six-CTA/SM layer scheduler has enough
  margin to absorb production query RMSNorm, grouped `wo_a` selection, and the
  sparse real-fixture replay while retaining at least 82% local-read-roofline
  efficiency at M=1.
- **Primary metric:** useful packed checkpoint bytes divided by complete
  scheduler time, normalized by a same-process cold local-read ruler. The
  binding threshold is 82% in every process, with the three-process median
  reported.
- **Correctness gate:** exact production graph boundaries, followed by the
  experiment-0147 no-worse real-fixture gate. An unguarded scheduler is an
  upper bound only and cannot be promoted.
- **Memory ceiling:** 512 MiB. Peak device allocation is 522,400,028 bytes
  (498.20 MiB).
- **Rollback:** stop this architecture if any M=1 process is below 82%, or if
  any putatively faster reduction fails a retained layer fixture. Both
  conditions fired, so M `{2,3,4,8,16}` was deliberately not run.

The cheapest experiment was one complete M=1 scheduler arm: roughly 170 us per
measured iteration after the existing allocation and warm-up, versus a full
protected curve and later full-model integration. Fixed setup dominates the
sub-millisecond measured window, but the process completes in seconds; no long
system run was justified.

## Instantiated cost model

For 115,350,400 useful compact weight bytes,
`tau = max_r(W_r/B_r) + Sigma_serial` gives:

- median ruler: **840.21 GB/s**;
- ideal read time: **137.29 us**;
- 82% gate time: **167.425 us**;
- retained no-replay scheduler: **169.984 us**;
- gate miss: **2.559 us**;
- measured throughput: **678.60 GB/s / 80.77%**.

The weight-read term remains the largest useful-volume term. The mechanism
under test adds a serial dependent boundary between `wq_a` and `wq_b`: query
RMSNorm must consume the complete 1,024-element BF16 `wq_a` publication before
`wq_b` can start. True grouped `wo_a` also replaces the feasibility probe's
single reused activation slice with eight production slices. Those corrections
add no checkpoint traffic, but the normalization adds FP32 reduction, BF16
weight reads and a grid-wide publish handoff. The numerical replay adds extra
weight/decode/ALU traffic and cannot reduce the already-binding scheduler time.

## Three-process binding result

| Process | Ruler GB/s | Scheduler us | Effective GB/s | Efficiency | 82% budget us | Verdict |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 829.57 | 171.008 | 674.53 | 81.31% | 169.571 | **fail** |
| 2 | 840.54 | 169.984 | 678.60 | 80.73% | 167.358 | **fail** |
| 3 | 840.21 | 169.984 | 678.60 | 80.77% | 167.425 | **fail** |
| **median** | **840.21** | **169.984** | **678.60** | **80.77%** | **167.425** | **reject** |

The retained kernel uses 72 registers, 32 bytes of static shared memory, and
has zero stack, spill, or local-memory bytes. Query RMSNorm matches its host
association exactly. Replay counters are zero because this is intentionally
the fastest uncorrected upper bound. Its synthetic `q_b+indexer` and grouped
`wo_a` mismatches are additional confirmation that the arm is not promotable.

Raw retained-result SHA-256 hashes:

- process 1: `96d0954e61d2a77874d5711ca74c3811f047f1b7d4e3edcdd1bd2a10e2d482e7`;
- process 2: `55a5d77dac5e1b69735debd1f63d3aaf64b0838d17f2a3f8a455dd08c37c6cfd`;
- process 3: `afb2ae730f71c824e7e59fd81395223b66d6fc87aaa9e3568a6954691e471b4b`.

The ignored raw artifacts are under
`results/dsv4-0149-sm86-fp8-exact-scheduler/`.

Both CUDA probes compile cleanly. `make check` passed all registered tests;
the Gemma equivalence target skipped as expected because its external fixture
was unavailable.

## Falsified mechanisms

| Arm | Scheduler us | Roofline efficiency | Why it is not retained |
|---|---:|---:|---|
| byte-wise FP64 replay | 1,044.480 | 13.14% | serialized scalar decode and replay |
| packed-fragment FP64 replay | 283.648 | 48.40% | PRMT/HMUL2 decode helps, FP64 work still dominates |
| compensated FP32 replay | 235.520 | 59.04% | fewer expensive operations, still far outside budget |
| eight-way-ILP FP32 replay | 225.280 | 61.72% | more ILP cannot hide replay volume |
| four HMMA accumulator chains, no replay | 173.056 | 80.35% | real layers 2 and 21 fail no-worse correctness |
| per-work-item/register query norm | 216.064 | 63.95% | repeats a dependent reduction across work items |
| 64-warp cooperative query norm | 176.128 | 78.95% | coordination costs more than the last-CTA publish |
| move independent `wkv` before query publication | 172.032 | 79.80% | changes ordering, not the dependent critical path |
| retained last-CTA query norm, no replay | **169.984** | **80.77% median** | fastest exact-graph upper bound, below gate and numerically unpromotable |

The NACC4 real-fixture control is particularly decisive. Layer 2 has seven
candidate `wq_b+indexer` oracle mismatches versus four for the incumbent and
one grouped-`wo_a` mismatch versus zero. Layer 21 also fails the declared
relative-error gate (0.01688 candidate versus 0.00543 incumbent). Layer 42
alone passes. A favorable layer cannot launder the two binding failures.

## Verdict and next action

F8-1's current successor is closed negatively and F8-2 remains closed. The
protected M curve was not run because its M=1 prerequisite failed. Do not add
this scheduler to production, lower D-F8-GATE, or hide protected shapes behind
a scalar fallback.

A future FP8 attempt must be a new bounded architecture, not another tuning
arm of this one. Its cheap falsifier must show how it removes the dependent
query-normalization publication and how it obtains incumbent-no-worse BF16
results without SIMT/FP64 replay on the critical path. Ampere-specific assets
remain the right raw material: BF16 HMMA, much larger useful execution grids,
resident CTAs, and overlap of genuinely independent layer work. The missing
asset is a semantics-preserving way to keep those engines fed across the
normalization boundary.
