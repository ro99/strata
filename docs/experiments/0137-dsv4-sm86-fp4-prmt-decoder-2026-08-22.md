# Experiment 0137 — a successor FP4 decoder that clears experiment 0136's budget

Status: **F4-1's feasible-cost-model blocker is CLEARED.** A PRMT/LUT decoder
with a native BF16 multiply for the E8M0 scale measures **810.93 GB/s cold on
both production shapes**, against experiment 0136's 514.02/512.00 GB/s and the
600.0 GB/s parity gate — **1.58x faster, and 97.5% of the measured read floor**,
with **zero oracle mismatches**. It also widens the valid E8M0 window from
codes 2–250 to 1–254. This is a decoder ceiling, **not** an F4-2 pass: no
fragment prepack, activation feed, output publication or split-K is included.

## Question, contract, and budget

Hypothesis: a decoder meeting experiment 0136's measured budget of at most 13.1
ALU operations per code-pair exists on SM86, and reaching it removes the
decoder as `argmax`.

Milestone F4-1, FP4 track only. Nothing here is FP8 evidence and nothing here
touches D-F8-GATE.

Gate, unchanged and not moved: the F4-2 parity threshold is >600.0 GB/s cold at
M=1 on both production shapes. This experiment measures the decoder ceiling
only, so clearing it is **necessary and not sufficient**.

Correctness gate: bit-exact against an independent host oracle that reproduces
the successor's own nibble pairing, over 65,536 sampled pairs per shape, in
three independent processes. Not a tolerance — exact equality.

Memory ceiling: 512 MiB. Measured 383.5 MiB, unchanged from 0136.

Rollback: delete the successor decoder; the 0136 arms and verdict are untouched
and remain committed.

Budget: the screen is a static SASS count costing seconds; the measurement is
three interleaved processes of about 5 s each. The expensive experiment
rejected: building the fragment prepack and a candidate kernel to find out
whether the decoder fits. The contract's own next-step names the static count
as the cheapest falsifier, and it is what rejected the first draft.

## The screen, which is what made this cheap

Experiment 0136 derived a budget of at most **18.3** ALU operations per
code-pair to touch the parity gate with zero margin, and at most **13.1** for
the decoder to stop being `argmax`. The 0135 decoder costs 21.4. A candidate
was therefore screened by static SASS instruction count **before** being
benchmarked:

| Decoder | ALU instructions | ALU ops per code-pair | Fits 13.1? |
|---|---:|---:|---|
| 0135 shift/rebias | 357 | 21.4 | no |
| PRMT/LUT, first draft | 141 | 7.9 | yes |
| PRMT/LUT, after the negative-zero fix | 166 | **9.4** | **yes** |

## What the successor does differently

Two changes, both of which remove work rather than reorganize it.

**1. The magnitude is a PRMT table lookup, not a bit reconstruction.** The
eight E2M1 magnitudes have only three distinct BF16 high bytes
(`0x00, 0x3F, 0x40`) and four distinct low bytes (`0x00, 0x40, 0x80, 0xC0`), so
each half-table fits in the eight source bytes a single `PRMT` can index, using
the magnitude nibbles themselves as the selector.

**2. The E8M0 scale is a native BF16 multiply, not an exponent add.** E8M0 code
`s` means `2^(s-127)`, whose BF16 encoding is simply `s` in the exponent field
with a zero mantissa, so the broadcast scale costs one `IMAD`. Replacing the
additive trick removes **all** of the zero-detection, subnormal-masking and
sign-preservation machinery that dominated the 0135 decoder. The SASS confirms
a real `HMUL2` — 16 of them — so this is native BF16 arithmetic, not an FP32
round trip.

**The nibble order is a prepack choice, not a format change.** The two codes of
a BF16 pair are stored 16 bits apart in the loaded word, so all four sign bits
reach bits 15 and 31 with a single shift each instead of four different ones.
The contract's transferable thesis item 2 — "pre-permute codes at load time
into the SM86 fragment order" — is exactly this permission. No weight bit is
added, removed or reinterpreted; `W_FP4` is unchanged at 4,456,448 bytes.

## The defect found by the oracle, and why the oracle mattered

The first working draft measured **821.13 GB/s** and **failed its oracle** with
7,938 and 8,040 mismatches of 65,536. Per the contract — "do not cite a fast
kernel that failed its oracle" — that number was discarded, not reported.

The rate was diagnostic. E2M1 code `0x8` is sign-set with magnitude zero, so
the device published `-0.0` where the reference publishes `+0.0`. The predicted
frequency of a BF16 pair containing at least one such code is
`2/16 - 1/256 = 12.11%`; the observed rates were 12.11% and 12.27%. The
prediction identified the bug before any debugging.

It was fixed rather than excused. `-0.0` is numerically harmless in the
subsequent MMA, and arguing for a declared delta would have been available —
but the accepted controls are bit-identical at M=1, and buying 1.5 ops per
code-pair to stay bit-identical is the right trade. The fix exploits the fact
that only the zero table entry has a zero exponent field, and that the largest
entry plus `0x7F80` stays inside its half, so one add carries into bit 15
exactly when the half is non-zero and never across the half boundary.

## Results

Cold effective packed-weight bandwidth, `W_FP4 / step`, median of three
independent interleaved processes, 30-replica 127.5 MiB arena, 256 MiB L2 scrub
before every cold sample, 3 warmups, 11 samples.

| Shape | Arm | Cold GB/s, three runs | Median |
|---|---|---|---:|
| `gate_up_w1` | `read_only` | 831.6, 831.6, 831.6 | 831.59 |
| `gate_up_w1` | `decode_0135_shift_rebias` | 483.6, 514.0, 514.0 | 514.02 |
| `gate_up_w1` | `decode_mma` | 481.8, 514.0, 514.0 | 514.02 |
| `gate_up_w1` | `decode_x2_attribution` | 247.7, 264.3, 264.3 | 264.29 |
| `gate_up_w1` | **`decode_prmt_lut_successor`** | 805.9, 810.9, 810.9 | **810.93** |
| `down_w2` | `read_only` | 831.6, 831.6, 831.6 | 831.59 |
| `down_w2` | `decode_0135_shift_rebias` | 487.2, 512.0, 514.0 | 512.00 |
| `down_w2` | `decode_mma` | 483.6, 514.0, 514.0 | 514.02 |
| `down_w2` | `decode_x2_attribution` | 247.7, 252.1, 264.3 | 252.05 |
| `down_w2` | **`decode_prmt_lut_successor`** | 805.9, 810.9, 816.0 | **810.93** |

Correctness: **0 mismatches** on both the 0135 oracle and the successor oracle,
every run, both shapes, 65,536 sampled pairs each.

The successor reaches **97.5%** of the measured read floor. The decoder has
effectively stopped being `argmax`: there is only 2.5% of headroom left between
it and the cost of simply reading the same bytes.

Raw artifacts: `results/qpn-sm86/0137-run1.json`, `0137-run2.json`,
`0137-run3.json`.

## Correction to experiment 0136's operating point

Experiment 0136 reported no SM clock or power state. That was a real omission,
and it was investigated after the owner asked whether this machine's 250 W cap —
against a 350 W default, required by the owner's electrical installation — was
distorting the result.

**During the probe it is not.** Sampled while the probe runs, the device holds
**1605 MHz at 101–120 W with `SW Power Cap: Not Active`**. The arms are
160–260 µs each with a 256 MiB scrub between them, so the probe never sustains
long enough to reach the limit. **Experiment 0136's verdict stands unqualified
and was not clock-distorted.**

**Under sustained load it very much does.** Running each kernel back to back for
10 s, which is closer to a production duty cycle:

| Arm | Cold probe GB/s | Sustained GB/s | Sustained SM clock | `SW Power Cap` | Throughput lost |
|---|---:|---:|---:|---|---:|
| `read_only` | 831.59 | 869.7 | 1605 MHz | Not Active | — |
| 0135 decoder | 514.02 | 470.6 | **1440 MHz** | **Active** | **−8.4%** |
| PRMT successor | 810.93 | 802.8 | **1065 MHz** | **Active** | **−1.0%** |

This is a result in its own right, and it points the same way as the headline.
**Under a power cap an ALU-bound FP4 kernel is penalized twice — once in
instructions and again in clock — while a memory-bound one is nearly immune.**
The successor clocks down 34%, far harder than the 0135 decoder's 10%, and
still loses only 1% of throughput, because ALU is no longer its `argmax` and
the memory system saturates regardless of core clock. On this machine, at this
cap, the successor's advantage is larger in production than the cold numbers
alone suggest.

The 250 W cap is an owner constraint and was not changed. All numbers in this
campaign are measured under it, which is the operating point that matters.

## The E8M0 window, measured rather than asserted

The successor applies the scale as a multiply, which suggested its valid range
should be wider than the additive decoder's 2–250. That was tested, not
claimed, by re-running with the stimulus window as a flag:

| E8M0 stimulus window | 0135 decoder mismatches | Successor mismatches |
|---|---:|---:|
| codes 2–250 | 0 | 0 |
| codes 1–254 | **172** | **0** |
| codes 0–255 | 750 | 363 |

The successor is exact across **codes 1–254**, the full range of E8M0 values
representable as normal BF16, and this closes experiment 0135's limitation 1
for every scale that matters. Code 0 means `2^-127`, which is subnormal in
BF16, and code 255 is the E8M0 NaN encoding; both still require an explicit
admission check, in either decoder. The default stimulus stays at 2–250 so the
timing comparison against 0136 remains apples-to-apples.

## Cost model, re-instantiated

`tau = max_r(W_r/B_r) + Sigma_serial` at M=1, `gate_up_w1`, with the successor:

- DRAM term **5.36 µs** at the measured 831.59 GB/s read floor, unchanged.
- ALU term now **9.4 ops per code-pair**, down from 21.4.
- Measured step **5.427 µs**, which is 1.3% above the read floor.

**`argmax` has moved from ALU to DRAM.** That is the point of the change, and
it is what the 13.1 ops-per-code-pair budget was constructed to predict. The
budget was derived in 0136 before this decoder existed and it held: 9.4 fits
under 13.1, and the decoder duly stopped being the bottleneck.

## Gate verdict

| Gate | Required | `gate_up_w1` | `down_w2` | Verdict |
|---|---|---:|---:|---|
| 0136 successor budget | <= 13.1 ops/code-pair | 9.4 | 9.4 | PASS |
| Decoder ceiling vs F4-2 parity | > 600.0 GB/s | 810.93 | 810.93 | PASS |
| Successor oracle | 0 mismatches | 0 | 0 | PASS |
| 0135 control oracle | 0 mismatches | 0 | 0 | PASS |
| One-copy residency | no widened persistent copy | none | none | PASS |
| Memory ceiling | < 512 MiB | 383.5 MiB | 383.5 MiB | PASS |

**F4-1's feasible-cost-model blocker is cleared. F4-2 is NOT passed and must
not be reported as passed.**

## What this does not establish

- **This is not an F4-2 result.** It is a decoder ceiling. A candidate must add
  the fragment prepack, the activation fragment feed, accumulator handling,
  output publication and any split-K reduction, and be timed as a full
  candidate step. Only 2.5% of headroom remains against the read floor, so
  those costs now have to come out of a very thin margin — the next gate is
  genuinely open.
- **The `decode_mma` arm still uses the 0135 decoder.** When the MMA was free
  to sample resolution, the decoder was `argmax` with 1.6x of slack to hide it
  in. At 97.5% of the read floor that slack is gone, so **whether the MMA is
  still free must be re-measured with the successor**, not inherited from 0136.
- The group-32 scale-to-K binding of 0135 limitation 2 remains unproven. This
  probe applies one scale per contiguous 32-weight group, correct for the
  stream layout used, but it performs no fragment prepack and so never
  exercises the MMA's K ordering.
- No M curve. F4-3's M `{1,4,8,16}` behavior was not measured and must not be
  extrapolated from an M=1-equivalent decode ceiling.
- Nothing about FP8, E4M3, block-128 scales, or D-F8-GATE.

## Exact next action

F4-1 proper is now unblocked and is the next work, in this order because each
step gates the next:

1. **Re-measure whether the MMA is still free** by adding a `decode_mma` arm
   built on the successor. This is a few lines in an existing probe and it
   decides whether the remaining 2.5% of headroom survives contact with the
   tensor op. If it does not, the candidate's budget must be rebuilt before any
   prepack work.
2. **Build the E2M1/E8M0 fragment prepack** for the real shapes and prove the
   group-32 scale-to-K binding across a group boundary, pinning K to the PTX
   coordinate rather than to 0135's self-consistent gauge.
3. **Carry an admission check for E8M0 codes 0 and 255**, which both decoders
   get wrong and which the widened 1–254 window does not cover.
4. Only then time a full candidate step against the unmoved >600.0 GB/s parity
   gate on both shapes, three interleaved process medians, oracle clean.

Independently, **F8-0 remains open, unblocked, and untouched by any of this.
D-F8-GATE remains an open owner decision.**
