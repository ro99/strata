# Experiment 0143 — F4-3 surpass curve cleared; Ampere gets FP4 from software

Status: **F4-3 COMPLETE. Every point of the surpass curve passes.** At 32 routed
experts per launch and split-K 2: **742.9 / 749.9 GB/s at M=1**, **731.4 /
736.2 at M=4**, **704.1 / 666.7 at M=8** against the **>=632 GB/s** gate, and
**523.2 / 478.9 at M=16** against **>301.9**. Correctness is bit-exact at M=1
and within 1.5e-07 relative at wider M. **The FP4 track's performance gates are
now all cleared: F4-1, F4-2 and F4-3.**

Operating point: single RTX 3090, 350 W, stock unlocked clocks — the campaign's
only operating point since the 2026-08-22 owner amendment.

## Results

Steady state, 32 back-to-back launches per timed window, median of three
independent interleaved processes.

| M | Shape | GB/s, three runs | Median | Gate | Verdict | Max rel. error |
|---:|---|---|---:|---:|---|---:|
| 1 | `gate_up_w1` | 739.8, 742.9, 762.2 | **742.9** | 632.0 | **PASS** | 0.0 |
| 1 | `down_w2` | 745.4, 749.9, 787.1 | **749.9** | 632.0 | **PASS** | 0.0 |
| 4 | `gate_up_w1` | 725.1, 731.4, 756.9 | **731.4** | 632.0 | **PASS** | 5.6e-08 |
| 4 | `down_w2` | 709.3, 736.2, 741.8 | **736.2** | 632.0 | **PASS** | 0.0 |
| 8 | `gate_up_w1` | 699.7, 704.1, 712.6 | **704.1** | 632.0 | **PASS** | 1.5e-07 |
| 8 | `down_w2` | 665.7, 666.7, 686.0 | **666.7** | 632.0 | **PASS** | 5.6e-08 |
| 16 | `gate_up_w1` | 514.6, 523.2, 523.8 | **523.2** | 301.9 | **PASS** | 6.0e-08 |
| 16 | `down_w2` | 476.8, 478.9, 483.0 | **478.9** | 301.9 | **PASS** | 5.8e-08 |

Bandwidth is on **useful packed weight bytes**, `W_FP4 = N*K/2 + N*K/32` =
4,456,448 per expert, which is independent of M — the same weights serve more
tokens as M grows, which is the entire point of a skinny kernel.

**Numerical contract.** M=1 is bit-exact, 0.0 relative error against a
double-precision oracle computed from the canonical layout. Wider M carries at
most 1.5e-07 relative, which is FP32 epsilon-level and is the summation-order
delta the contract already permits for wider M. No precision, scale semantics
or activation boundary changed.

## Two defects found and fixed while generalising to M > 1

Both were caught by comparing against the M=1 result rather than by inspection.

**1. A 24% regression at M=1, from a runtime loop bound.** Generalising the
kernel put `col_blocks` in the hot loop as a runtime value, which stopped the
inner loop being fully unrolled: M=1 fell from 700.7 to 531.0 GB/s. Making it a
template parameter and hoisting the per-lane activation predicate and offset out
of the K loop restored it to 692.5 and then 742.9 with a better configuration.
**A runtime bound on a four-iteration inner loop cost a quarter of the
throughput.**

**2. Activation traffic re-inflated 8x.** The first M-generalisation stored a
full 32-lane B-fragment tile per column block, of which only `min(M,8)` column
groups are ever non-zero. At M=1 that is 7/8 zeros. Storing only
`min(M,8) x 4` fragments per column block and predicating the rest to zero
removed it. This is the same M=1 waste already fixed on the output side in 0141,
reappearing on the input side during a refactor.

## Why throughput initially fell with M, and what fixed it

Throughput on weight bytes should be flat or rising in M, since weight traffic
is identical. It fell — 692.5 at M=1 to 455.7 at M=8 — which by the charter's
own rule is a defect, not a datapoint.

The cause is that **split-K partial traffic scales as `split_k x M`**:

| M | split-K 8 | split-K 2 |
|---:|---:|---:|
| 1 | 3% of useful bytes | 1% |
| 8 | **23%** | 6% |
| 16 | **47%** | 12% |

At larger M each warp already produces more useful output, so it needs less
split-K to fill the machine. Dropping split-K from 8 to 2 and raising the expert
batch to 32 makes the curve flat again: 742.9, 731.4, 704.1 at M=1, 4, 8. The
residual decline at M=16 is the second column block's activation and partial
traffic, and it still clears its gate by 1.6x.

## Against upstream

Upstream's V100 QPN numbers are a different device, a different format and a
different scale layout, so this is a roofline-efficiency comparison, not a
throughput one.

| M | This kernel, GB/s | % of the measured 847.79 GB/s SM86 read floor | Upstream QPN8 % of its 879 GB/s V100 ceiling |
|---:|---:|---:|---:|
| 1 | 742.9 | **88%** | 82% |
| 4 | 731.4 | **86%** | 82% |
| 8 | 704.1 | **83%** | 81% |
| 16 | 523.2 | 62% | 64% |

The SM86 W4A16 path matches or exceeds upstream's roofline efficiency at every M
up to 8, and tracks it at M=16 — while carrying **four-bit** weights rather than
eight-bit.

Against upstream's own FP4 figure at M=16, 301.9 GB/s, this kernel measures
**523.2 GB/s, a factor of 1.73**, which was the F4-3 gate's explicit
requirement.

## The claim this earns, stated precisely

Upstream's line is *"Blackwell gets NVFP4 support in silicon. Volta gets it from
software."* The Ampere equivalent is now supported by measurement, with one
correction that matters:

> **Blackwell gets FP4 in silicon. Ampere gets it from software — at 88% of
> memory roofline.**

**It is FP4, not NVFP4, and the distinction is not pedantic.** NVFP4 is E2M1
with **FP8 E4M3 group-16** scales. Strata's checkpoint format, which this kernel
implements exactly, is E2M1 with **E8M0 group-32** scales — an MX-style layout.
The contract records this difference explicitly, and upstream's claim is about
NVFP4 specifically. Claiming NVFP4 here would be claiming a format this kernel
does not decode.

**What is earned:** on SM86, with no hardware FP4 support, a register-fed
W4A16 kernel decoding E2M1 codes and applying E8M0 group-32 scales entirely in
software runs at 83–88% of the device's pure-read bandwidth for M from 1 to 8.
Software decode costs roughly 12% over simply reading the same bytes and doing
nothing.

**What is not earned:** this is a kernel result, not a serving result. There is
no production dispatch, admission, route census, graph integration or end-to-end
token throughput. MIX-1 and MIX-2 are untouched. The honest full sentence is
*"...from software, in a kernel, not yet in a server."*

## Cost model

`tau = max_r(W_r/B_r) + Sigma_serial`, M=1, 32 experts, `gate_up_w1`:

- DRAM term **5.26 µs per expert** at the 847.79 GB/s read floor.
- Candidate **6.00 µs per expert**.
- `argmax` is **DRAM at 88% utilisation**. Decode, E8M0 scaling, the MMA,
  split-K reduction and dispatch together account for the remaining 12%.

The bottleneck is the resource the campaign wanted it on, and there is roughly
12% of headroom left before the kernel is indistinguishable from a pure read.

## Gate verdict

| Gate | Required | Result | Verdict |
|---|---|---|---|
| F4-3 M curve | >= 632 GB/s at M in {1,4,8}, both shapes | 742.9/749.9, 731.4/736.2, 704.1/666.7 | **PASS** |
| F4-3 M=16 | > 301.9 GB/s | 523.2 / 478.9 | **PASS** |
| Numerical contract | M=1 exact; wider M only a summation-order delta | 0.0 at M=1, <=1.5e-07 above | PASS |
| Oracle sensitivity control | must fail | 766.9 / 1116.8 | PASS |
| One-copy residency | permutation only | codes and scales byte-identical to canonical | PASS |

**F4-1, F4-2 and F4-3 are all cleared. The FP4 track has no remaining
performance gate.**

## What this does not establish

- No production integration. MIX-1 (mixed one-copy dispatch, route census,
  admission, graph, VRAM) and MIX-2 (end-to-end workload) are untouched, and a
  kernel bandwidth figure is explicitly not an end-to-end claim.
- The 32-expert operating point is not yet tied to the target workload's
  measured routed-expert dispatch width. If real dispatch is narrower, these
  numbers must be re-stated at that width.
- E8M0 codes 0 and 255 still require an admission check.
- Nothing about FP8, E4M3, block-128 scales, or D-F8-GATE.

## Exact next action

1. **Measure the target workload's routed-expert dispatch width per layer**, and
   re-state the F4-2 and F4-3 results at that width. This is the cheapest
   remaining falsifier of the whole FP4 result: if production dispatches experts
   two at a time, the operating point that cleared these gates does not occur.
2. **Add the E8M0 0/255 admission check.**
3. **Begin MIX-1** only after (1): one-copy mixed production dispatch, route
   census, admission, prepack cost at load, VRAM accounting and graph
   integration.
4. **F8-0 remains open, unblocked, and independent. D-F8-GATE remains an open
   owner decision** and still blocks any FP8 performance claim.
