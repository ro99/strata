# Experiment 0141 — the uint4 granularity fix, and how the F4-2 gate is measured

Status: **F4-1 STEP 3 PARTIAL. The candidate improves 1.67x to 427.6 / 449.2
GB/s steady state, remains correct at 0.0 relative error, and is still short of
the >600.0 GB/s gate and of its own 826 GB/s decoder ceiling.** Four
optimisations were applied and measured individually. A structural finding
about the gate's *measurement* is the more important result: **a single-launch
wall-clock measurement of one 4.5 MB matrix cannot reach 600 GB/s even with a
zero-cost kernel**, because the event-pair floor alone exceeds the gate's entire
slack over the DRAM floor.

Operating point: **experimentation** — single RTX 3090, 350 W, unlocked clocks.

## What was applied, and what each was worth

Experiment 0140 attributed the gap to load granularity. That was **partly
right**: it was one of four terms, and not the largest.

| Change | `gate_up_w1` | `down_w2` | Worth |
|---|---:|---:|---|
| Experiment 0140 baseline | 256.00 | 255.53 | — |
| `uint4` load: 4 K-tiles per lane per load | 288.93 | 310.86 | +13% / +22% |
| Write only the real output column | 310.86 | 334.77 | +8% / +8% |
| Fold the reduction into one launch | 334.8 | 362.7 | +8% / +8% |
| Measured as steady-state rather than one launch | **427.6** | **449.2** | +28% / +24% |

Cumulative **1.67x and 1.76x** over 0140, with correctness unchanged at 0.0
relative error and the deliberate-bug control still firing.

**The `uint4` fix delivered 13–22%, not the 3–4x predicted.** The prediction in
0140 was that granularity was *the* `argmax`; it was one contributor. Recording
that plainly: the 0140 attribution over-claimed, and the correction is that
this candidate had four comparable terms rather than one dominant one.

### The output-column fix

At M=1 the MMA produces a 16x8 D tile of which **only column 0 is real**, so
writing the whole tile made 7/8 of the split-K partial traffic zeros — measured
at 24% to 94% of the useful weight bytes depending on split-K, and the reason
throughput *degraded* past split-K 16 in 0140. Column 0 lives in `d0` and `d2`
of the lanes with `thread == 0`, so storing just those cuts partial traffic 8x.

### The reduction fold, and why it is nearly a wash

A separate reduce kernel cost a full 4.10 µs of dispatch for trivial work. It
was folded into the matmul with a last-slice-reduces pattern: `__threadfence`,
an atomic counter per N-tile, and the finishing warp reducing and resetting the
counter so the kernel stays re-runnable without a separate memset.

Measured steady state, split-K 16:

| Reduction | `gate_up_w1` | `down_w2` |
|---|---:|---:|
| Folded in-kernel | 428.6 | 396.8 |
| Separate kernel | 418.2 | 428.5 |

**Nearly a wash.** The fold trades 4.10 µs of dispatch for roughly 2–3 µs of
`__threadfence` and atomic cost inside the kernel. It helps single-launch wall
time and does not help steady state, so the reduction is **not** the remaining
bottleneck either. Both are kept behind `--split-reduce` so neither claim rests
on a single configuration.

## The measurement finding, which matters more than the kernel

Phase attribution of the two-launch version:

| Shape | Total | matmul | reduce | empty-kernel event pair |
|---|---:|---:|---:|---:|
| `gate_up_w1` | 14.30 µs | 11.07 | 4.10 | **4.10** |
| `down_w2` | 14.34 µs | 10.24 | 4.10 | **4.13** |

An **empty kernel** measures 4.10 µs inside an event pair, identical to the
reduce kernel that does almost nothing. That figure is the event-pair
measurement floor, not work.

Now the arithmetic:

- The gate at 600.0 GB/s on 4,456,448 bytes allows **7.43 µs**.
- The measured read floor of 847.79 GB/s costs **5.26 µs** and is unavoidable.
- The gate's entire slack over the DRAM floor is therefore **2.17 µs**.
- The event-pair floor is **4.10 µs**, which is **1.9x that slack**.

**Even a zero-cost kernel measured this way tops out at 476 GB/s.** The gate is
unreachable by construction when a single 4.5 MB matrix pass is timed inside one
event pair, regardless of how good the kernel is.

This is the same defect experiment 0136 caught and fixed for the decode ceiling
probe — timing a window too short relative to dispatch — reappearing in the
candidate, where it is harder to see because the "window" is a real production
shape rather than an arbitrary one.

The campaign's own ruler is not measured this way: it sweeps 128 MiB in one
launch. Experiment 0139's 826 GB/s ceiling sweeps a 30-replica arena. **The
ceiling and the candidate have been measured on different footings**, and the
candidate's footing is the deflated one.

This experiment therefore reports a **steady-state** number: 32 back-to-back
launches in one event window, divided. That matches how production dispatches
these — many layers and experts inside a CUDA graph — and it is the same
principle that fixed 0136.

| Shape | Single-launch wall clock | Steady state, three processes | Median |
|---|---:|---|---:|
| `gate_up_w1` | 334.8 | 427.3, 427.6, 428.5 | **427.6** |
| `down_w2` | 362.7 | 449.2, 449.2, 449.3 | **449.2** |

Steady state is 0.3% spread across processes — far more reproducible than the
single-launch figure, which is another sign that the single-launch number is
dominated by a fixed cost rather than by the kernel.

## Cost model

`tau = max_r(W_r/B_r) + Sigma_serial` at M=1, `gate_up_w1`:

- DRAM term **5.26 µs** at 847.79 GB/s.
- Decoder-plus-MMA ceiling **5.39 µs** (experiment 0139).
- This candidate, steady state **10.42 µs**.

`argmax` is still not DRAM and still not the decoder. The candidate is 1.93x its
own ceiling. Falsified as the cause, by measurement: parallelism (0140), scale
loads (0140), activation divergence (0140), the MMA accumulator chain (0140's
`--no-mma` arm), load granularity (this experiment, worth only 13–22%), partial
traffic (worth 8%), and the reduction structure (a wash). What remains
un-attributed is the per-warp prologue and the short trip count: at split-K 8 a
warp runs only 8 K-tile blocks, so index setup, scale extraction, and the tail
are amortised over very little work.

**No further optimisation should be attempted until that residual is
attributed**, which this experiment does not do. Guessing has now been wrong
twice — 0140 predicted granularity would be worth 3–4x and it was worth 13–22%.

## Correctness

Unchanged and re-verified after every change:

| Shape | max relative error | Deliberate-bug control |
|---|---:|---|
| `gate_up_w1` | **0.0** | fails at 766.9, detected |
| `down_w2` | **0.0** | fails at 1116.8, detected |

One real defect was caught by the oracle during this work: the first
single-launch reduction indexed its shared arrival slot by `n_tile`, which is
out of range and wrong because warps in a block carry different N-tiles. It
measured *faster* and failed correctness; the number was discarded and the
indexing fixed to one slot per warp. A second was caught by the gate: split-K 64
on `down_w2` truncates `k_tiles_per_slice / kKPerLoad` to zero, so the kernel
silently did no work and the oracle reported a relative error of 1.

40 registers, 0 spills, 0 barriers, 16 bytes of shared memory.

## Gate verdict

| Gate | Required | `gate_up_w1` | `down_w2` | Verdict |
|---|---|---:|---:|---|
| Correctness vs canonical double oracle | 0 error | 0.0 | 0.0 | PASS |
| Oracle sensitivity control | must fail | 766.9 | 1116.8 | PASS |
| One-copy residency | permutation only | yes | yes | PASS |
| F4-2 parity, steady state | > 600.0 GB/s | 427.6 | 449.2 | **not yet** |
| F4-2 parity, single-launch wall clock | > 600.0 GB/s | 334.8 | 362.7 | **unreachable by construction** |

**F4-2 is not passed.**

## The owner question this raises

**How is the F4-2 gate to be measured?** The threshold is not in question and
must not move. The measurement footing is:

1. **Steady state**, 32 back-to-back launches in one window — matches
   production graph dispatch, matches how the 842 GB/s ruler and the 826 GB/s
   ceiling were measured, and is what this record reports. Candidate is at
   427.6 / 449.2 against 600.0.
2. **Single-launch wall clock** for one production matrix — includes dispatch,
   but is then **arithmetically unable to reach 600.0 GB/s with any kernel**,
   because the 4.10 µs event floor is 1.9x the gate's 2.17 µs slack over the
   DRAM floor.

Option 2 makes the gate unachievable by construction rather than by engineering.
Option 1 is consistent with every other number in the campaign. **No agent
should pick between them silently**, and this record does not: it reports both
and recommends option 1 on the grounds of consistency with the ruler.

## Exact next action

1. **Owner decision on the F4-2 measurement footing**, per above. The 600.0
   threshold is unchanged either way.
2. **Attribute the residual 1.93x** between the candidate and its own 826 GB/s
   ceiling before optimising further. The cheapest falsifier is an Nsight
   profile of the candidate at split-K 8 reporting warp-state sampling, issue
   efficiency and memory throughput — not another guess. Two guesses have now
   been wrong.
3. **Still owed from step 3:** the admission check for E8M0 codes 0 and 255, and
   a guard rejecting split-K values where `k_tiles_per_slice / kKPerLoad` is
   zero, which currently produces a silently empty kernel.
4. No M curve has been measured.

**F8-0 remains open, unblocked, and independent. D-F8-GATE remains an open
owner decision.**
