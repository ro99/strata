# Experiment 0140 — the FP4 fragment prepack, and the scale-to-K binding proof

Status: **F4-1 STEP 2 COMPLETE.** The E2M1/E8M0 group-32 fragment prepack is
built for both production shapes and is **bit-exact against a double-precision
oracle computed from the canonical layout** — max relative error **0.0** across
128 and 64 K-group boundaries. The prepack is a pure permutation: codes stay
4,194,304 bytes and scales stay 262,144 bytes, so one-copy residency holds.

Deriving the layout exposed a **real defect in experiment 0137's decoder**, and
a deliberate-bug control proves this probe detects it. Throughput is **256.00 /
255.53 GB/s**, well below the 826 GB/s decoder ceiling, and the cause is
attributed by measurement rather than guessed: **load granularity**, not the
MMA, not the scales, not the activations, not parallelism.

Operating point: **experimentation** — single RTX 3090, 350 W, unlocked clocks.

## The defect the prepack derivation exposed

The `m16n8k16` A fragment, per the lane map verified in experiment 0135, places
registers 0 and 2 on N-row `g` and registers 1 and 3 on N-row `g+8`. E8M0
scales are per `(N-row, K/32)`. **Therefore one lane's 32-bit code word spans
two N-rows and needs two different scales.**

Experiment 0137's decoder applies a **single** `scale_pair` to all four
registers. That is correct for the flat code stream 0137 measured and **wrong in
fragment order**. The contract's next-step instruction — re-derive the nibble
pairing against the real fragment layout rather than assume it survives — was
what caught this.

The fix is free. Register parity selects the scale, and because the decode loop
is `#pragma unroll`ed over four registers, the selection folds at compile time:

```cpp
const std::uint32_t scale = ((i & 1U) == 0U) ? scale_low_row : scale_high_row;
```

What survives unchanged from 0137: the PRMT magnitude table, the `HMUL2` scale
application, and the 16-bit-apart nibble pairing, which the prepack simply
honours by construction.

## The prepack

Pure permutation of the canonical checkpoint arrays. For tile `nt`, K-tile
`kt`, lane `lane` with `g = lane>>2` and `t = lane&3`, the 32-bit word carries:

| Nibble | Weight | Nibble | Weight |
|---|---|---|---|
| 0 | `W[nt*16+g][kt*16+2t]` | 4 | `W[nt*16+g][kt*16+2t+1]` |
| 1 | `W[nt*16+g+8][kt*16+2t]` | 5 | `W[nt*16+g+8][kt*16+2t+1]` |
| 2 | `W[nt*16+g][kt*16+2t+8]` | 6 | `W[nt*16+g][kt*16+2t+9]` |
| 3 | `W[nt*16+g+8][kt*16+2t+8]` | 7 | `W[nt*16+g+8][kt*16+2t+9]` |

Scales are stored `[n_tile][k_group][16 bytes, one per tile row]`.

| Shape | Canonical codes | Prepacked codes | Canonical scales | Prepacked scales |
|---|---:|---:|---:|---:|
| `gate_up_w1` | 4,194,304 | **4,194,304** | 262,144 | **262,144** |
| `down_w2` | 4,194,304 | **4,194,304** | 262,144 | **262,144** |

**Byte-for-byte identical. No widened path, no duplicate copy, `W_FP4`
unchanged.** Prepack cost is 4–5 ms per matrix, one-time at load.

## Correctness

Oracle: double precision, computed from the **canonical** arrays, so a broken
prepack or a mis-bound scale cannot cancel out.

| Shape | K-groups crossed | Non-zero outputs | max abs output | max relative error |
|---|---:|---:|---:|---:|
| `gate_up_w1` | 128 | 2048 / 2048 | 59,917.45 | **0.0** |
| `down_w2` | 64 | 4096 / 4096 | 57,954.48 | **0.0** |

Bit-exact, three processes. The result is exact rather than merely close
because every term is a dyadic rational — an E2M1 code times a power of two
times a small integer activation — so FP32 accumulation is exact here.

**The scale-to-K binding is proven across group boundaries**, which was the
open item from experiment 0135's limitation 2.

### The test is not vacuous, and it is proven sensitive

A perfect 0.0 demands verification, so two checks were run. Every output is
non-zero with magnitudes near 60,000, so the kernel is doing real work. And a
`--break-scale-binding` control reproduces 0137's single-scale behaviour:

| Variant | `gate_up_w1` max rel | `down_w2` max rel | Exit |
|---|---:|---:|---|
| Correct, two scales by register parity | **0.0** | **0.0** | 0 |
| 0137 single-scale, deliberately broken | **766.9** | **1116.8** | 1 |

The oracle detects the exact defect the derivation found, by three orders of
magnitude. A probe that cannot fail its own control is not evidence.

## Throughput, and where the gap actually is

| Shape | Cold, three processes | Median |
|---|---|---:|
| `gate_up_w1` | 256.0, 256.0, 256.0 | **256.00 GB/s** |
| `down_w2` | 241.8, 255.5, 256.0 | **255.53 GB/s** |

Against experiment 0139's 826.33/831.59 GB/s decoder-plus-MMA ceiling, this is
**3.2x short**, and short of the >600.0 gate. Four hypotheses were tested and
**three were falsified**:

| Hypothesis | Test | Result |
|---|---|---|
| Insufficient parallelism | split-K sweep 4/8/16/32/64 | **Falsified.** Peaks at 16 and *degrades* past it as reduction traffic grows |
| Scale loads: 64 byte-loads per warp per K-tile | replaced with one 16-byte broadcast load plus extraction | **Falsified.** 254.1 to 256.0, no material change |
| Activation loads divergent, only `group==0` active | pre-permuted into B-fragment order, one coalesced 8-byte load per lane | **Falsified.** No material change |
| MMA accumulator dependency chain | `--no-mma` arm keeps every load and the full decode, drops the tensor op | **Falsified.** 256.0 with and without the MMA |

Removing the tensor op entirely changes nothing, so the bottleneck is upstream
of it. What remains is **load granularity**: this kernel loads **4 bytes per
lane per K-tile** where experiment 0139's ceiling probe loads **16**. Per useful
byte it therefore issues four times the loads, four times the loop control, and
four times the per-tile index arithmetic. That ratio, 4x, is close to the
measured 3.2x shortfall.

This is a structural property of the current loop, not of the prepack, the
decoder, the scale binding, or the MMA — all of which are now proven correct and
measured free.

## Cost model

`tau = max_r(W_r/B_r) + Sigma_serial` at M=1, `gate_up_w1`, experimentation
point:

- DRAM term **5.36 µs** at the 847.79 GB/s measured read floor.
- Decoder-plus-MMA ceiling **5.39 µs** (experiment 0139).
- This kernel **17.42 µs**.

`argmax` is neither DRAM nor the decoder. It is per-K-tile instruction and load
issue overhead — a `Sigma_serial`-style term created by processing 4 bytes per
iteration. The charter's rule applies directly: this is an overlap and
granularity defect, not a volume problem, and the volume is already right
because every byte is read exactly once.

## Gate verdict

| Gate | Required | `gate_up_w1` | `down_w2` | Verdict |
|---|---|---:|---:|---|
| Scale-to-K binding across group boundaries | exact | 0.0 over 128 groups | 0.0 over 64 groups | **PASS** |
| Prepack is a permutation | no size change | 4,194,304 / 262,144 | 4,194,304 / 262,144 | **PASS** |
| Oracle sensitivity | control must fail | 766.9 | 1116.8 | **PASS** |
| One-copy residency | no duplicate/widened weights | none | none | **PASS** |
| Device allocation | < 512 MiB | 5.3 MiB | 6.3 MiB | PASS |
| F4-2 parity gate | > 600.0 GB/s | 256.00 | 255.53 | **not yet** |

**F4-1 step 2 passes. F4-2 is not passed and this experiment does not claim it.**

## What this does not establish

- No F4-2 result. The candidate is 3.2x short of its own decoder ceiling.
- M=1 only. Seven of the MMA's eight output columns are zero at this shape,
  which is inherent to skinny decode and is what the campaign exists to attack,
  but no M curve was measured.
- The split-K reduction is a separate kernel whose traffic is included in the
  timing but not separately attributed.
- Nothing about FP8 or D-F8-GATE.

## Exact next action

F4-1 step 3, then step 4, with the granularity finding as step 4's opening move:

1. **Add the admission check for E8M0 codes 0 and 255**, which both decoders
   get wrong.
2. **Raise the load granularity to a `uint4` per lane.** Restack the prepack so
   a lane's four consecutive K-tile words are contiguous — index
   `((n_tile*(k_tiles/4) + kb)*32 + lane)*4 + j` — so one 16-byte load feeds
   four MMAs, matching the ceiling probe's access pattern. Consecutive lanes
   then read consecutive 16-byte chunks, giving fully coalesced 512-byte warp
   transactions. This is the measured `argmax`, and it is the only remaining
   term between the candidate and its 826 GB/s ceiling.
3. Consider multiple N-tiles per warp sharing one B fragment, which is
   transferable-thesis item 5 and would further amortise the activation load —
   but only if step 2 above does not already close the gap, and only on measured
   evidence.
4. Then time the full candidate against the unmoved **>600.0 GB/s** gate on both
   shapes, three interleaved process medians, reporting sustained SM clock and
   run spread alongside the median.

**F8-0 remains open, unblocked, and independent. D-F8-GATE remains an open
owner decision.**
