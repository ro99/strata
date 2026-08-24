# Experiment 0155 — FP4 gates re-stated at the real dispatch width, and E8M0 admission

Status: **BOTH REMAINING FP4 ITEMS CLOSED.** The routed-expert dispatch width is
**6**, not the 8 and 32 the gates were originally cleared at — and after
re-tuning split-K for that width, **every gate still passes**, including the
worst case of single-token decode with only 6 experts active. The E8M0 0/255
admission check is implemented and proven to fire.

Operating point: single RTX 3090, 350 W, unlocked clocks.

**Track coordination.** A concurrent session owns the FP8 track and reserved
experiment numbers 0150-0154; FP4 records use 0155 and above. Nothing in
`apps/strata_dsv4_sm86_fp8_*` or the FP8 contract status was touched.

## 1. The dispatch-width falsifier, fired and survived

Experiments 0142 and 0148 cleared F4-2 and F4-3 at **8** and **32** independent
expert matrices per launch, with the open question of whether production ever
dispatches that wide. It does not dispatch that wide at decode. From
`kDeepSeekV4ExecutionContract`, confirmed independently by
`kDsv4RankLocalTopK`:

- **256 routed experts, `experts_per_token` = 6**, `expert_intermediate_size` =
  2048, which matches `gate_up_w1 [N=2048,K=4096]` and `down_w2 [N=4096,K=2048]`
  exactly.

At the previous split-K 2 tuning, 6 experts gives only **0.39 waves per SM** and
the M=1 result fell to **597.3 GB/s — 3 GB/s short of the 600.0 parity gate.**
The falsifier fired.

It was not fatal, because split-K is the free variable: with a narrower batch
the same machine can be filled by splitting K further. Re-tuned to split-K 4,
6 experts gives 1.56 waves per SM and M=1 clears the gate.

### M and dispatch width are coupled, and pairing them wrongly invents a workload

The more important finding is that **M and the number of active experts are not
independent**. With 256 experts and top-k 6, `T` concurrent decode tokens
produce `6T` expert slots spread over `min(256, 6T)` active experts:

| Concurrent tokens | Active experts | M per expert |
|---:|---:|---:|
| 1 | 6 | 1.0 |
| 8 | 48 | 1.0 |
| 171 | 256 | 4.0 |
| 341 | 256 | 8.0 |
| 683 | 256 | 16.0 |

So **M=1 is the only case that occurs with a narrow dispatch**, and any M >= 4
requires enough concurrent tokens to activate all 256 experts. Measuring M=8
against a 6-wide dispatch — as a naive reading of "re-state the gates at the
real width" would demand — describes a workload where 341 concurrent tokens all
route to the same 6 experts out of 256. That point does not exist.

The honest re-statement therefore pairs each M with the dispatch width that
produces it: M=1 at the **worst case** of 6 experts, and higher M at a wide
dispatch. Batch 64 is used as a **conservative** stand-in for the real 256,
because 128 matrices already exceeds the campaign's 512 MiB probe allocation
ceiling at 544 MiB.

### Results at production-realistic operating points

Steady state, median of three independent interleaved processes.

| M | Active experts | split-K | Shape | GB/s, three runs | Median | Gate | Verdict |
|---:|---:|---:|---|---|---:|---:|---|
| 1 | **6 (worst case)** | 4 | `gate_up_w1` | 634.9, 635.4, 635.9 | **635.4** | 600.0 | **PASS** |
| 1 | **6 (worst case)** | 4 | `down_w2` | 667.8, 668.4, 668.5 | **668.4** | 600.0 | **PASS** |
| 4 | 64 (real 256) | 1 | `gate_up_w1` | 735.8, 735.9, 737.0 | **735.9** | 632.0 | **PASS** |
| 4 | 64 (real 256) | 1 | `down_w2` | 726.2, 726.2, 726.3 | **726.2** | 632.0 | **PASS** |
| 8 | 64 (real 256) | 1 | `gate_up_w1` | 709.2, 711.9, 714.9 | **711.9** | 632.0 | **PASS** |
| 8 | 64 (real 256) | 1 | `down_w2` | 687.1, 694.3, 694.3 | **694.3** | 632.0 | **PASS** |
| 16 | 64 (real 256) | 1 | `gate_up_w1` | 570.1, 571.1, 571.1 | **571.1** | 301.9 | **PASS** |
| 16 | 64 (real 256) | 1 | `down_w2` | 545.6, 551.3, 553.2 | **551.3** | 301.9 | **PASS** |

M=1 is bit-exact at 0.0 relative error; wider M carries at most 6.5e-07, the
permitted summation-order delta.

**M=1 at 6 experts also clears the stricter 632 F4-3 threshold on both shapes**,
so the surpass curve holds at the worst-case width too.

### The split-K dispatch rule this yields

Two competing terms set split-K, and they now have measured shapes:

- Too little split-K starves the machine: 6 experts at split-K 2 is 0.39 waves
  per SM.
- Too much inflates split-K partial traffic, which scales as **`split_k x M`**
  and reached 94% of useful weight bytes at M=8 with split-K 4 on `down_w2`.

The measured rule for MIX-1 dispatch: **choose the smallest split-K that reaches
about 1.5 waves per SM**, which is split-K 4 for a 6-expert M=1 decode and
split-K 1 once the dispatch is wide or M >= 4. Higher M needs *less* split-K,
not more, because it already produces more useful output per warp.

## 2. E8M0 0/255 admission, owed since experiment 0137

Both FP4 decoders map the E8M0 scale code directly into a BF16 exponent field.
That is exact for codes 1-254 and **silently wrong outside it**:

| Code | Means | Encodes as | Result |
|---:|---|---|---|
| 0 | 2^-127 | `0x0000` | **+0** — subnormal in BF16, whose smallest normal is 2^-126 |
| 1 | 2^-126 | `0x0080` | exact |
| 254 | 2^127 | `0x7F00` | exact |
| 255 | E8M0 NaN | `0x7F80` | **+inf** |

The contract requires exact mode to execute an approved route or report failure,
never to substitute silently. `admit_e8m0_scales` therefore scans the canonical
scale array at load and **fails admission** on any code outside 1-254, naming
the offending code and its byte offset.

**The check is proven to fire.** A control injects a single bad code into one of
262,144 scale bytes:

| Case | Result | Exit |
|---|---|---|
| Real checkpoint | admitted, 0 inadmissible, max rel 0.0, 635.5 / 667.9 GB/s | 0 |
| One code 255 injected | `admission failure: gate_up_w1 carries 1 inadmissible E8M0 scale codes (first: code 255 at byte offset 87381)` | **1** |
| One code 0 injected | same, reporting code 0 | **1** |

Admission runs on the canonical array before the prepack, costs one linear scan
of `N*K/32` bytes, and does not touch the measured path — throughput and
correctness are unchanged when the checkpoint is clean.

## Gate verdict

| Gate | Required | Result | Verdict |
|---|---|---|---|
| F4-2 parity at the **real** decode width | > 600.0 GB/s, 6 experts, M=1 | 635.4 / 668.4 | **PASS** |
| F4-3 curve at coupled widths | >= 632 at M in {1,4,8}; > 301.9 at M=16 | 635.4/668.4, 735.9/726.2, 711.9/694.3, 571.1/551.3 | **PASS** |
| Numerical contract | M=1 exact, wider M only a summation-order delta | 0.0 at M=1, <= 6.5e-07 above | PASS |
| E8M0 admission | reject codes 0 and 255, report rather than substitute | fires on both, exit 1 | **PASS** |
| Admission does not disturb the clean path | unchanged | 0 inadmissible, 0.0 error | PASS |

**Both remaining FP4 items are closed. The FP4 track's open work is now
integration only.**

## What this does not establish

- Batch 64 is a **conservative stand-in** for the real 256-expert wide dispatch;
  128 exceeds the 512 MiB probe ceiling. The wide-M numbers should improve at
  the true width, not degrade, but that is a prediction.
- The M-to-token coupling is derived from the routing contract (256 experts,
  top-k 6), not from a measured routing trace. A trace with strongly skewed
  routing could concentrate more tokens on fewer experts; the campaign's own
  decode traces show 2.03x concentration, which shifts the table but does not
  create the M=8-at-6-experts point.
- Still a kernel result. MIX-1 and MIX-2 are untouched, and no serving claim
  exists.
- Nothing about FP8. That track is owned by a concurrent session.

## Exact next action

**FP4 has no remaining kernel work.** The next FP4 step is **MIX-1**: one-copy
mixed production dispatch with a route census, admission wired to
`admit_e8m0_scales`, load-time prepack cost, VRAM accounting, graph integration
and fixtures. MIX-1 additionally depends on the FP8 track reaching an accepted
F8-2, which the concurrent session owns.

Carry into MIX-1: the split-K dispatch rule above, and the requirement that the
runtime dispatch routed experts **together per layer** rather than one at a
time — at 6 experts and split-K 4 that is 1.56 waves per SM, while one expert at
a time is 0.26 and loses roughly 40% of throughput.
