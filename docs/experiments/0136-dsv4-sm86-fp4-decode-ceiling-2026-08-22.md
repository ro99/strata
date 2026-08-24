# Experiment 0136 — the SM86 FP4 decode ceiling, and the rejection of the 0135 decoder

Status: **F4-1 GATE NEGATIVE. REJECTED: the experiment 0135 shift/rebias
E2M1/E8M0 decoder cannot support the F4-2 parity gate on SM86.** An upper-bound
arm that decodes the production streams and does none of the remaining required
work measures **514.02 GB/s** on `gate_up_w1` and **512.00 GB/s** on `down_w2`,
against a gate of **>600.0 GB/s**. The shortfall is 86–88 GB/s and a real
candidate must do strictly more work. The bottleneck is measured, not inferred:
the arm is ALU/issue-throughput bound. **The QPN register-fed dataflow is not
rejected** — the native MMA that consumes the decoder is free to sample
resolution.

## Question, contract, and budget

Hypothesis: the register-fed E2M1/E8M0 group-32 decoder audited in experiment
0135 can sustain the F4-2 parity gate of >600.0 GB/s cold effective
packed-weight bandwidth at the two production expert shapes.

Milestone F4-1, FP4 track only. Nothing here is FP8 evidence and nothing here
touches D-F8-GATE.

**Kill criterion, stated before the run:** the decode arm is an upper bound on
any F4-1 candidate. It performs no fragment prepack, no activation feed, no
output publication and no split-K reduction, all of which a real candidate must
add. If its cold effective packed-weight bandwidth is below 600.0 GB/s on
either production shape, the parity gate is unreachable with this decoder and
F4-1 must change the decoder or be rejected. The threshold was not to be moved,
and the shape and M were not to be changed to find a friendlier regime.

Primary metric: cold effective packed-weight bandwidth,
`W_FP4 / step`, with `W_FP4 = N*K/2 + N*K/32` = 4,456,448 bytes for both
shapes. Activation, output, partial and scrub traffic are excluded from useful
weight bytes by construction — this probe moves no such traffic at all.

Correctness gate: every BF16 pair produced by the timed decoder path, over
65,536 sampled pairs per shape, bit-exact against an independent host
computation built from a magnitude table and `ldexp`. Zero mismatches, all
runs, both shapes.

Memory ceiling: 512 MiB. Measured peak device allocation is 127.5 MiB of arena
plus the 256 MiB scrub buffer, 383.5 MiB.

Rollback: delete the probe and its CMake entry. No runtime or production
dispatch is touched.

Budget: three interleaved processes at about 4 s each. The whole experiment is
under a minute. The cheaper experiment rejected: an arithmetic screen alone,
which put the budget at 15.8 ALU ops per code-pair against an estimated
decoder cost near 17 — too close to the line for arithmetic to settle, so it
justified the measurement rather than replacing it. The more expensive
experiment rejected: building the fragment prepack and a full candidate kernel
first, which would have spent days to discover the same ceiling.

## The defect found and fixed before any result was recorded

The first version of this probe timed one production matrix per launch and
reported `read_only` at **435.20 and 483.56 GB/s**. That is precisely the band
the campaign contract forbids: *"Never use the older 435–484 GB/s
launch-deflated ruler."* The reproduction of a known-forbidden artifact was
treated as a defect, not a datapoint, and the cause was measured rather than
guessed: an empty kernel on this device is event-measured at a median of
**4.096 µs**, against a timed window of 10.24 µs — **40% launch and event
overhead**. The timings were also quantized to multiples of 1.024 µs.

Fix: sweep a 127.5 MiB arena of 30 matrix replicas per launch, matching the
scale of the established 128 MiB ruler, which puts the window near 160 µs and
launch overhead under 3%. Per-matrix step time is then **derived** by division
and is labeled as derived everywhere; it is not measured directly, because a
single-matrix launch cannot be timed here without reintroducing the deflation.

The fix validates itself: `read_only` now reports **831.59–842.32 GB/s cold
and 858.95 GB/s hot**, independently reproducing the campaign's 842 GB/s cold
and ~857 GB/s hot ruler on a different access pattern and a different byte
stream. A probe that reproduces the accepted ruler is measuring the link; one
that reports 435 is measuring its own launch overhead.

## A second defect: the attribution arm that tested nothing

The first attribution arm decoded each byte twice under two different E8M0
deltas and measured only **1.087x** for supposedly doubled work, which reads as
a refutation of the ALU-bound hypothesis. It was not. Everything in the decoder
except the final `scaled += delta` depends only on the code byte, so
common-subexpression elimination collapsed the second decode to a single add.
The SASS confirms it: the arm carried **399** ALU instructions against the
decode arm's **357**, an 11.8% increase that measured 8.7% — proportional, and
no test of the hypothesis at all.

Corrected by decoding a genuinely different code (`code ^ 0x5A`), the arm
carries **667** ALU instructions, a 1.868x increase. Both defects are recorded
because each would have produced a confidently wrong verdict: the first a false
rejection of the whole campaign, the second a false acquittal of the decoder.

## Device, protocol, and shapes

RTX 3090, compute capability 8.6, verified inline in every artifact; the probe
hard-fails any device that is not SM86. Toolchain nvcc 12.8, V12.8.93.
Measured device limits: 82 SMs, 65,536 registers per SM, 1,536 threads per SM.

Protocol per contract section 3 and 8: 256 MiB L2 scrub before every cold
sample, 30-replica arena rotation, 3 warmups, 11 samples, median, and three
independent interleaved process repetitions. Shapes are the actual production
per-rank expert shapes, `gate_up_w1 [N=2048,K=4096]` and
`down_w2 [N=4096,K=2048]`. The E8M0 stimulus is confined to codes 2–250, the
window experiment 0135 recorded as valid.

## Results

Cold effective packed-weight bandwidth, median of three interleaved processes.
Per-matrix microseconds are derived from the swept window.

| Shape | Arm | Cold GB/s, three runs | Median GB/s | Derived µs/matrix |
|---|---|---|---:|---:|
| `gate_up_w1` | `read_only` | 831.6, 831.6, 831.6 | **831.59** | 5.359 |
| `gate_up_w1` | `decode` | 514.0, 514.0, 514.0 | **514.02** | 8.670 |
| `gate_up_w1` | `decode_mma` | 514.0, 514.0, 514.0 | **514.02** | 8.670 |
| `gate_up_w1` | `decode_x2_attribution` | 264.3, 264.3, 264.3 | 264.29 | 16.862 |
| `down_w2` | `read_only` | 831.6, 842.3, 842.3 | **842.32** | 5.291 |
| `down_w2` | `decode` | 489.0, 512.0, 516.0 | **512.00** | 8.704 |
| `down_w2` | `decode_mma` | 487.2, 512.0, 514.0 | **512.00** | 8.704 |
| `down_w2` | `decode_x2_attribution` | 250.1, 262.7, 263.2 | 262.70 | 16.964 |

Correctness: 65,536 sampled pairs per shape, **0 mismatches**, every run.

Raw artifacts: `results/qpn-sm86/0136-run1.json`, `0136-run2.json`,
`0136-run3.json`, all under the ignored `results/` tree.

## Cost model, instantiated at the real operating point

`tau = max_r(W_r/B_r) + Sigma_serial` at `M=1`, `gate_up_w1`:

- `W_DRAM` = 4,456,448 useful packed bytes. `B_DRAM` measured at 831.59–842.32
  GB/s cold by the `read_only` arm on this exact access pattern. DRAM term
  **5.29–5.36 µs**.
- `W_ALU` = 4,194,304 code-pairs at a static SASS count of
  (357 − 15)/16 = **21.4 ALU instructions per code-pair**, so 89.8M ALU
  instructions. Measured ALU term **8.670 µs**, giving a measured
  `B_ALU` of **10.35 Tops/s**.
- `Sigma_serial` is not material here: zero barriers, zero shared memory, zero
  spills, and a single grid-stride loop.

**`argmax_r` is ALU**, at 1.62–1.65x the DRAM term. The attribution is measured,
not inferred: 1.868x the ALU instructions produced **1.945x and 1.949x** the
time, at identical DRAM traffic, identical zero spill/shared/barrier profile,
and identical achievable occupancy — all decode kernels use 39–40 registers,
which still permits the full 48 warps per SM, the same as `read_only`'s 14.
Near-proportional scaling with instruction count is the signature of a
throughput bound; a latency or occupancy bound would have absorbed the extra
ALU into existing stall slots and scaled sub-linearly.

The pre-run arithmetic screen assumed `B_ALU` = 8.92 Tops/s from 82 SMs x 64
INT32 lanes x 1.7 GHz and predicted a budget of 15.8 ops per code-pair. The
measurement puts the budget at 18.3 and `B_ALU` at 10.35 Tops/s, so the model
was **1.16x conservative** — integer work is not confined to the 64 INT32
lanes, because `IMAD` issues on the FMA pipe. The measured constant supersedes
the modeled one and must be used going forward. It was measured at `M=1` on
these two shapes and may not be reused at another operating point.

**The MMA is not the problem.** `decode_mma` equals `decode` to sample
resolution on both shapes: adding the native `HMMA.16816.F32.BF16` that
consumes the decoder costs nothing measurable. The transferable QPN dataflow —
codes compressed through HBM, decoded at point of consumption, straight into
operand registers with no shared-memory staging — is intact and is not what
failed.

## Gate verdict

| Gate | Required | `gate_up_w1` | `down_w2` | Verdict |
|---|---|---:|---:|---|
| F4-2 parity, cold M=1 | > 600.0 GB/s | 514.02 | 512.00 | **FAIL** |
| Decoder correctness | 0 mismatches | 0 | 0 | PASS |
| One-copy residency | no widened persistent copy | none | none | PASS |
| Memory ceiling | < 512 MiB | 383.5 MiB | 383.5 MiB | PASS |
| Ruler reproduction | ~842 cold / ~857 hot | 831.6 / 859.0 | 842.3 / 859.0 | PASS |

The kill criterion fired on **both** shapes. Per the contract, this is recorded
as `REJECTED` and the line stops here. The threshold was not moved, the shape
and M were not changed, the activation precision was not swapped, and no
dependent milestone was entered.

**What is rejected:** the experiment 0135 shift/rebias E2M1/E8M0 decoder as the
basis for an F4-2 pass. It is preserved as a correct, audited control.

**What is not rejected:** the QPN2-derived register-fed dataflow, the native
SM86 BF16 MMA, the C2 lane/register map, or the FP4 track itself.

## The budget any successor decoder must meet

Derived from this experiment's measured constants, for a decoder doing nothing
else:

| Target | Derived µs/matrix | Maximum ALU ops per code-pair |
|---|---:|---:|
| Current 0135 decoder | 8.670 | 21.4 |
| F4-2 parity, 600 GB/s | 7.427 | **18.3** |
| F4-3 surpass, 632 GB/s | 7.051 | 17.4 |
| Measured read floor, decoder stops being `argmax` | 5.325 | **13.1** |

A successor must clear 18.3 to touch the parity gate **with zero budget left**
for the fragment prepack, activation fragment feed, accumulator handling,
output publication, and any split-K reduction that a real candidate must add.
The engineering target is therefore **at or below 13.1 ops per code-pair**,
where the decoder ceases to be `argmax` and DRAM becomes the bound again. That
is a required reduction of **1.6x or better** from the current decoder.

Whether such a decoder exists on SM86 is an open question this experiment does
not answer. The contract forbids reopening PRMT-LUT variants "without a profile
showing that the rejected premise changed"; this experiment is such a profile,
because the earlier rejection was of PRMT-LUT inside a shared-memory SIMT
staging architecture, whereas the open question now is a decoder cost budget
inside a register-fed architecture whose MMA feed is measured free. That makes
a successor arm **permissible**, not proven, and it requires its own gate.

## What this does not establish

- Nothing about FP8, E4M3, block-128 scales, or D-F8-GATE.
- No claim that a 13.1-ops-per-pair decoder exists or is reachable.
- No M curve. This experiment measures M=1-equivalent decode throughput only;
  F4-3's M `{1,4,8,16}` curve was not run and must not be extrapolated from it.
- No production integration, dispatch, admission, or route-census evidence.
- The group-32 scale-to-K binding of 0135 limitation 2 remains unproven. This
  probe applies one scale per contiguous 32-weight group, which is the correct
  production semantics for the stream layout used, but it does not exercise the
  MMA fragment K ordering, because it does no fragment prepack.

## Exact next action

F4-1 is blocked on a decoder that fits the measured budget. Two options, and
the choice is the owner's because it decides whether the FP4 track continues:

1. **Screen a cheaper decoder against the 13.1 ops-per-code-pair budget**,
   before any prepack or kernel work. The cheapest falsifier is a static SASS
   instruction count of the candidate decoder alone — no benchmark needed to
   reject one that cannot fit. Only a decoder that fits should be measured in
   this probe's `decode` arm, which is already built and takes seconds per arm.
2. **Record F4-1 as REJECTED for SM86** and re-scope the FP4 track, on the
   evidence that DRAM offers 842 GB/s while the cheapest known-correct
   E2M1/E8M0 decode of that stream costs 1.6x the DRAM time.

Option 1 is cheap enough to be worth one attempt before option 2. Neither may
proceed by moving the 600.0 GB/s threshold.

Independently, **F8-0 remains open and unblocked** and does not depend on any
of this.
