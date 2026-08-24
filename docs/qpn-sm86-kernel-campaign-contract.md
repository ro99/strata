# RTX 3090 mixed MXFP4/FP8 skinny-kernel campaign contract

Status: **ACTIVE — authoritative campaign contract**

Established: 2026-08-22

Owner amendment: 2026-08-22 — expanded from an FP4-only campaign to the
checkpoint's mixed FP4/FP8 execution stack

Owner amendment: 2026-08-22 — declared two distinct operating points and moved
experimentation to a single unlocked RTX 3090, after the owner identified that
this machine's 1605 MHz clock lock was distorting gate results (experiment
0138)

Owner amendment: 2026-08-22 — **the capped/locked configuration is retired as a
campaign gate.** The single unlocked RTX 3090 is now the only operating point
the campaign measures or optimises for. The owner will cap the cards when
running TP=2 and accepts the resulting slowdown as their own concern; it is not
a campaign obligation and no result requires re-measurement under it.

Owner amendment: 2026-08-22 — bound the campaign to software-native execution
of DeepSeek V4's published mixed MXFP4/FP8 weight representation, analogous to
`v100-skinny` on Volta, and resolved D-F8-GATE at equal local-read-roofline
efficiency: 82% for M=1-4, 81% for M=8, and 64% for M=16.

Target: NVIDIA RTX 3090, Ampere GA102, SM86

Umbrella objective:

> Run DeepSeek V4's published mixed MXFP4/FP8 weight representation unchanged
> on RTX 3090/SM86, providing software-native low-bit execution analogous to
> `v100-skinny` on Volta. MXFP4 regions use a QPN2-derived W4A16 path and FP8
> regions use a QPN8-derived W8A16 path. Codes and scales remain compressed
> through HBM and preserve their exact checkpoint values and semantics without
> arithmetic requantization; a verified invertible fragment-order permutation
> may be the single resident device representation. Each kernel decodes and
> scales weights directly into BF16 or FP16 operand registers consumed by
> native SM86 HMMA, without a materialized or persistent widened weight tile.
> This is software-native checkpoint execution, not a claim that SM86 gains
> hardware FP4 or FP8 MMA opcodes. Numerical semantics, one-copy residency, and
> the measured dispatch contract remain binding.

Binding FP4 headline goal: **greater than 600 GB/s cold effective
packed-weight bandwidth**

Binding FP8 headline goal: **match `v100-skinny`'s local-read-roofline
efficiency: at least 82% at every M in `{1,2,3,4}`, 81% at M=8, and 64% at
M=16 on every eligible protected production shape. At the 842 GB/s campaign
ruler these are 690, 682, and 539 GB/s respectively; the percentage and the
corresponding same-session local ruler are authoritative.**

Archived implementation evidence head: `exp/dsv4-qpn-packed-decode@ecfb50d`

Active implementation branch: `exp/dsv4-sm86-qpn-register-feed`, created from
`main@b895f82`

This document is the single source of truth for the RTX 3090 mixed FP4/FP8
skinny-kernel campaign. Every agent must read it in full, including the final
row of the append-only log, before planning, coding, profiling, interpreting a
result, or handing off work.

The **Contract** section changes only when the owner explicitly changes it.
The 2026-08-22 mixed-format amendment is such a change and supersedes the
FP4-only scope without erasing it from history. Agents may update milestone
status, Current position, blockers, Exact next step, and the append-only log.
Older log rows must never be edited or deleted; correct an error with a new
row and, when necessary, a dated amendment.

## Contract

### 1. What is expected

Strata must execute the checkpoint's two low-bit region types without
requantizing one into the other:

- **MXFP4 regions:** E2M1 codes with E8M0 group-32 scales through an
  Ampere-adapted, QPN2-derived, register-fed **W4A16** architecture.
- **FP8 regions:** E4M3 weight codes with E8M0 block-128 scales through an
  Ampere-adapted, QPN8-derived, register-fed **W8A16** architecture.

This is an RTX 3090/SM86 campaign analogous to the mixed-format execution
stack in `dnv2003/v100-skinny`. It is not a literal Volta source, format, lane
map, or opcode port. “QPN2-derived” and “QPN8-derived” name the register-fed,
fragment-prepacked dataflow. They do not claim that Strata's formats or
numerical boundaries are identical to upstream.

The final production result is mixed dispatch:

1. eligible MXFP4 regions use the accepted QPN2-derived W4A16 path;
2. eligible FP8 regions use the accepted QPN8-derived W8A16 path; and
3. unsupported shapes use an explicit, named, contract-approved exact route.

An unsupported case may fail admission when no approved exact route exists.
It may never disappear into a hidden fallback. Dispatch counters and a route
census must make every production choice observable.

### 2. Shared invariants

- Target hardware is exactly RTX 3090, GA102, SM86. Successful compilation or
  performance on Volta is not target evidence.
- Weights stay in their checkpoint precision and scale semantics. No FP8-to-FP4
  conversion, FP4-to-FP8 conversion, or persistent FP16/BF16 expansion is
  permitted.
- W4A16 and W8A16 both retain 16-bit activation operands at the MMA boundary.
  An existing FP32 carrier may be converted according to the operation's
  declared BF16/FP16 boundary. The campaign does not silently introduce
  activation E4M3 or otherwise turn W8A16 into W8A8.
- Accumulation, publication, routing, expert count, top-k, and model semantics
  remain those declared by each production operation. A performance result
  does not authorize a numerical-contract change.
- Only one persistent packed representation per weight region is allowed in
  the promoted design. A verified invertible fragment-order prepack may
  replace the canonical device layout after all consumers are accounted for;
  it may not become a second unnoticed full copy.
- Exact mode either executes an approved exact route or reports failure. It
  never silently substitutes a different precision, format, or activation
  contract.

### 3. FP4/W4A16 track

#### Format and numerical contract

- Weight codes: packed E2M1, two codes per byte.
- Weight scales: one E8M0 byte per group of 32 weights.
- Activation boundary: FP32 carrier values known to be BF16-representable at
  the current expert-projection boundary; the candidate is W4A16.
- Accumulation/publication: existing FP32 behavior under Strata's declared
  int4 numerical contract.
- Current production per-rank expert shapes:
  `gate_up_w1 [N=2048,K=4096]` and
  `down_w2 [N=4096,K=2048]`.

Every candidate must pass the existing int4 reference oracle. M=1 has so far
been bit-identical for accepted controls. Wider M may retain only an already
declared summation-order delta. Exhaustive code/scale decode, random matrices,
both production shapes, real weights/activations, operation/layer fixtures,
and the full runtime oracle remain independent gates.

#### Byte accounting and resource model

For divisible production shapes, useful packed weight bytes per matrix pass
are:

`W_FP4 = N*K/2 + N*K/32` bytes.

Both current expert shapes therefore contain 4,456,448 useful packed bytes.
Headline bandwidth is `W_FP4 / full_candidate_step_time`, including required
split-K reduction. Activation, output, partial-reduction, scrub, and duplicate
probe traffic must be reported separately and never folded into useful weight
bytes to inflate the result.

The baseline resource model is instruction/ALU plus serial eligibility loss,
not DRAM: the legacy SIMT arm is instruction-bound, while conventional packed
WMMA exposes 62–65% no-eligible cycles and only about 20% DRAM utilization.
The candidate targets decoder/instruction work and shared-memory/barrier serial
terms. It adds fragment prepack, register pressure, activation conversion,
scale work, and possibly partial reduction; every sign must be measured.

#### Performance and coverage gates

1. **Parity:** greater than 600.0 GB/s cold at M=1 on both production shapes.
2. **Surpass:** at least 75% of the measured 842 GB/s cold SM86 ruler,
   currently 632 GB/s after rounding, at M in `{1,4,8}` on both shapes; also
   greater than upstream's 301.9 GB/s FP4 result at M=16.
3. **Production:** preserve a material outside-variance win after prepack,
   reduction, dispatch, memory, graph, and routed-decode costs.

The FP4 microbenchmark must use cold arena rotation/L2 scrub, the actual
shapes, full candidate timing, three warmups, eleven samples, and at least
three independent interleaved process repetitions. Any failure of correctness,
one-copy residency, either M=1 shape, the required M curve, or the measured
cost-model feasibility gate rejects that candidate architecture. The threshold
may not be moved to a favorable shape or M.

### 4. FP8/W8A16 track

#### Format and numerical contract

- Weight codes: one E4M3 byte per weight.
- Weight scales: one E8M0 byte per checkpoint `[128,128]` weight block.
- Activation boundary: a separately declared 16-bit BF16 or FP16 MMA operand
  derived from the operation's existing carrier; the intended candidate is
  W8A16, not W8A8.
- Accumulation and publication are defined per production operation. Existing
  attention-page projections publish through BF16; that does not authorize an
  agent to assume all FP8 consumers have the same boundary.
- Before setting dispatch coverage, inventory every real FP8 region, operation,
  `(M,N,K)`, activation carrier, publication boundary, and frequency in the
  target workload. Upstream's Qwen region census and M ranges are not Strata's
  dispatch contract.

FP8 correctness is independent of FP4 correctness. Required gates are an
exhaustive E4M3/E8M0 decoder test, random matrix tests, every protected
production shape, real-weight/real-activation fixtures, comparison to an FP64
decoded oracle at the operation's immediate production boundary, and full
teacher-forcing/generation oracles. Experiment 0104's no-worse-after-BF16
result applies only to the tested existing path and shapes; it is evidence,
not a blanket waiver for a new W8A16 reduction order.

#### Byte accounting and resource model

For N and K divisible by 128, useful packed weight bytes per matrix pass are:

`W_FP8 = N*K + (N/128)*(K/128)` bytes.

For padded or non-divisible tensors, use the manifest's actual encoded storage
layout and report padding explicitly; do not replace block-128 scaling with a
per-row or group-32 formula. Headline bandwidth is
`W_FP8 / full_candidate_step_time`, including required reduction. Report
activation, output, scale, partial, scrub, prepack, and any duplicate probe
traffic independently.

Experiments 0103–0105 measured a large production weight-reread defect for
multi-row attention pages and removed it by tiling. That operating point and
its current incumbent must be re-measured separately from skinny decode. The
new QPN8-derived path must instantiate `tau = max_r(W_r/B_r)+Sigma_serial` at
each real M band before implementation. It is expected to reduce expanded
tile/shared-memory traffic, barriers, and decode/feed instructions, while
increasing fragment-prepack work and possibly registers, activation-conversion
work, and split-K/NACC costs. A QPN8 mechanism that does not reduce the current
`argmax` at its intended operating point is rejected before production work.

#### Microbenchmark, coverage, and rejection gates

The FP8 track requires its own clean SM86 control, measured read ceiling,
correct byte formula, resource profile, and independent interleaved matrix.
Report every real dispatch M band; at minimum preserve distinct results for
single-row decode, verification batches, and multi-row attention-page shapes
that are actually eligible. Do not copy upstream's M≤8, MT=2 M≤16, or chunked
M≤96 boundaries. Re-derive fragment layout, instruction shape, split-K, NACC,
and dispatch boundaries for SM86 and Strata.

The FP8 throughput threshold is owner-bound to equal `v100-skinny`'s fraction
of its local read ceiling. Upstream QPN8 on a V100 read ceiling of 879 GB/s
reports:

| M | Upstream QPN8 | V100 roofline efficiency | Same efficiency against the measured 842 GB/s SM86 ruler |
|---:|---:|---:|---:|
| 1–4 | 718.6–720.5 GB/s | 82% | approximately 690 GB/s |
| 8 | 712.4 GB/s | 81% | approximately 682 GB/s |
| 16 | 558.5 GB/s | 64% | approximately 539 GB/s |

The binding SM86 gate is the local-efficiency column, evaluated against the
cold read ceiling measured in the same clean harness and operating point as the
candidate:

- every M in `{1,2,3,4}`: at least **82%** of the local read ceiling;
- M=8: at least **81%** of the local read ceiling; and
- M=16: at least **64%** of the local read ceiling.

At the 842 GB/s campaign ruler these round to fixed reference values of **690,
682, and 539 GB/s** respectively. The percentages are authoritative when the
same-session ruler changes, so a faster measured ruler cannot lower the
required efficiency. Each threshold applies to every eligible protected
production shape at that M; unsupported shapes remain subject to the explicit
exact-route or admission-failure contract. These are W8A16 gates only. No agent
may apply the FP4 >600/632 thresholds or substitute W8A8 evidence.

Independently of the bound numeric threshold, reject any FP8 candidate
that changes W8A16 to W8A8, changes E4M3/E8M0 block-128 semantics, creates a
persistent widened/duplicate weight copy, fails an operation boundary oracle,
hides an unsupported shape, or cannot plausibly reduce the measured bottleneck.

### 5. Existing FP8 paths are distinct

The following paths must never be conflated in plans, results, or dispatch:

1. **Existing scalar/native production path:** `native_fp8_matmul_kernel` and
   its current exact dispatch. It is an incumbent/control, not QPN8.
2. **Existing SM86 tensor-page path from experiments 0103–0105:** E4M3 weights
   are decoded to BF16 through a conventional WMMA tile with 48 KiB shared
   memory per CTA. Production quantizes each activation row/K128 block to E4M3
   with an E8M0 activation scale. This is an activation-quantized W8A8-style
   behavior and a valuable production/control path; it is not the intended
   register-fed W8A16 QPN8 architecture.
3. **Intended QPN8-derived path:** checkpoint E4M3/E8M0 block-128 weights feed
   an SM86-native register-fragment architecture while activations retain their
   declared 16-bit boundary. It requires a new correctness and performance
   verdict and cannot inherit 0103–0105's gate automatically.

An explicit future owner amendment could approve activation-quantized W8A8 as
a separate optimization, but it would remain a separate named dispatch and
could not be reported as completion of this W8A16 track.

### 6. Shared Ampere-native prerequisite and transferable thesis

The native SM86 MMA lane/register-map work is a shared prerequisite only where
the operand type and instruction are truly shared. It must establish:

- exact lane/register-to-matrix coordinates for the selected native SM86 MMA;
- intended tensor instructions in SASS;
- no spills, unintended local/shared widened path, or hidden memory operands;
- the activation fragment contract used by both W4A16 and W8A16 candidates;
- measured register/occupancy feasibility before format-specific kernels.

After that shared fact is established, FP4 and FP8 split into independent
tracks. Code decoders, scale semantics, fragment prepack, byte accounting,
split-K, NACC, correctness, benchmarks, M coverage, dispatch, and rejection
criteria remain format-specific.

The transferable QPN thesis is dataflow:

1. keep codes compressed through HBM;
2. pre-permute codes at load time into the SM86 fragment order;
3. arrange decoder output to occupy operand registers without inner-loop
   shuffles or a materialized widened tile;
4. decode and apply each format's exact scales at point of consumption;
5. share activation fragments across as much N work as the measured SM86 lane
   map permits;
6. avoid shared-memory staging and block-wide barriers in the main weight loop
   when the native fragment path permits; and
7. select split-K, NACC, multi-tile geometry, and dispatch boundaries from SM86
   evidence rather than copying constants.

Tensor cores remain plausible. The rejected default is the high-level,
shared-memory-heavy WMMA architecture as the target dataflow, not tensor cores
as a class.

### 7. What does not transfer automatically from upstream

- Volta's `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`, its quadpair/lane
  map, all-four-quadpairs-on-N geometry, FP16 operand details, and dispatcher.
- Upstream QPN2's NVFP4 scale semantics. The upstream source describes E2M1
  with FP8 group-16 scaling; Strata requires E2M1 with E8M0 group-32 scaling.
- Upstream QPN8's Volta-specific E4M3 decoder, FP16 scale folding, fragment
  permutation, MT=2, split-K values `{4,8,16,32}`, NACC values `{1,2}`, shared
  reduction, register count, occupancy, or M≤96 dispatch.
- V100 throughput values as proof that an SM86 mechanism works, and successful
  SM86 compilation of a legacy opcode as evidence that it is efficient.

For both tracks, fragment layouts, native MMA instructions, scale handling,
split-K, NACC, reduction, and dispatch ranges must be re-derived on SM86.
Neither high-level WMMA nor a literal Volta source port is presumed correct.

### 8. Measurement and memory protocol

### Declared operating points (owner amendment, 2026-08-22)

This machine boots `apply-3090-tuning.service`, which applies `-pl 250` and
`-lgc 1605,1605 --mode=1` to every RTX 3090. The SM maximum is 2100 MHz, so
that lock is roughly a 24% underclock. The **memory clock is not locked**.
Experiment 0138 established that this distorts gate results asymmetrically:
the gates derive from the memory ruler, which the lock does not touch, while a
candidate's ability to meet them is set by ALU throughput, which it cuts by a
quarter.

**The campaign has ONE operating point: a single RTX 3090 at 350 W with stock
unlocked clocks, the second card idle.** Restore it with
`sudo nvidia-smi -i 1 -pl 350 && sudo nvidia-smi -i 1 -rgc`; it is not
reboot-persistent, since `apply-3090-tuning.service` reapplies 250 W and the
1605 MHz lock at boot.

The 2026-08-22 owner amendment retires the capped/locked configuration as a
gate. The objective is to extract maximum throughput from one RTX 3090. The
owner caps both cards when running TP=2 and accepts that slowdown; **no campaign
result requires re-measurement under the cap, and none may be blocked on it.**

Experiment 0138's cap measurements are preserved as evidence of how the two
configurations differ — an ALU-bound kernel loses 8–10%, a DRAM-bound one about
1% — but that difference is now context, not a gate.

The **FP4 gates did not move when the operating point changed**: the 842-class
ruler is memory-bound and re-measured at 845.63 GB/s cold with clocks unlocked,
so >600.0 and 632 stand exactly as written. Unlocking makes them honestly
meetable; it does not lower them.

**The cold protocol overstates ALU-bound candidates.** Its 160-260 us arms
boost to 1935 MHz at 150-200 W while sustained load settles at 1755 MHz against
the power limit, a measured 9.4% gap for an ALU-bound arm and none for a
DRAM-bound one. Every candidate must report its sustained SM clock alongside
its cold number, and an ALU-bound candidate that clears a gate only on the cold
protocol has not cleared it.

All headline kernel numbers are cold measurements on an identified RTX 3090.
Use the established L2 scrub and arena rotation, three warmups, eleven
interleaved samples, and the median of at least three independent interleaved
process repetitions. The current ruler is 842 GB/s cold and approximately
857 GB/s hot, measured in experiment 0129 and reproduced in 0134. Re-measure
when clocks, power, driver, CUDA, device, or benchmark traffic changes. Never
use the older 435–484 GB/s launch-deflated ruler.

For every optimization, instantiate
`tau = max_r(W_r/B_r) + Sigma_serial` at its actual operating point. Report
phase time, traffic and bandwidth, instruction/tensor utilization, registers,
shared memory, spills, grid/waves, occupancy, eligible/no-eligible cycles, and
barrier/scoreboard stalls. Name `argmax` and the serial term before selecting a
mechanism. Profile replay time is attribution evidence, not a headline time.

Prepack time, temporary workspace, scale storage, partial reductions, output
buffers, device allocation, RSS, and persistent-copy count must be measured.
Probe-only duplicate buffers must be declared. Promotion requires one
persistent packed representation for each region and no persistent widened
weights.

CUDA device enumeration differs between tools on this machine. Verify and
record `device_name` each session; a result accidentally collected on the RTX
5060 Ti is invalid.

### 9. Source-of-truth evidence

Read these sources before changing interpretation:

1. Upstream implementation:
   <https://raw.githubusercontent.com/dnv2003/v100-skinny/refs/heads/main/kernels/skinny_kernels.cu>
   The audited local copy is
   `results/qpn-surpass/reference/skinny_kernels.cu`, SHA-256
   `112ae57bdae0a0763d9898ad7b49eaf927be411cda0bde5344a76821d166533e`.
   Read both `skinny_nvfp4_qpn2` and `skinny_fp8_qpn8`/`qpn8_mt2`, their
   prepack, decode, split-K, NACC, and dispatcher—not only early SIMT/WMMA code.
2. Upstream README snapshot:
   `results/qpn-surpass/reference/v100-skinny-README.md`, SHA-256
   `433e66b4b538297893223431fe6da56a4df3d2ec33d175809b12e73331a0162b`.
   It defines the mixed QPN2 W4A16/QPN8 W8A16 execution architecture and the
   V100 performance evidence used above.
3. Reddit discussion:
   <https://www.reddit.com/r/LocalLLaMA/comments/1vsq3zg/nvfp4_on_volta_despite_being_built_for_blackwell/>
   The author's RTX 3090 statement is only that it “should work” and must be
   measured; the thread contains no Ampere result.
4. Experiments 0103–0105 under `docs/experiments/`: authoritative for Strata's
   E4M3/E8M0 block-128 behavior, production boundary, existing activation-
   quantized tensor-page path, and measured multi-row resource defect.
5. Experiments 0127–0134 under `docs/experiments/`: authoritative for Strata's
   FP4 controls, corrected SM86 ruler, bottleneck profiles, clean baseline, and
   rejected conventional architectures.

Evidence ranks as: target-hardware measurement with raw artifacts and oracle;
inspected source/SASS; upstream measurement on other hardware; author claim;
inference. Never promote lower-ranked contradictory evidence.

### 10. Explicit corrections and caveats

- This is a mixed FP4/FP8 campaign. The original contract and first two log
  rows described an FP4-only campaign; their work and chronology remain valid
  within that narrower scope, but their umbrella scope is superseded by this
  owner amendment.
- Upstream's M=1–8 FP4 headline is QPN2, not legacy SIMT. Experiment 0130
  rejects only the selected SIMT architecture.
- Experiments 0127, 0128, and 0132–0133 are conventional shared-memory WMMA
  controls, not QPN2's register-fed target architecture.
- Experiments 0103–0105 establish useful SM86 FP8 behavior and an accepted
  production control, but their 48 KiB shared-memory WMMA and E4M3-quantized
  activations are not the intended register-fed W8A16 QPN8 path.
- The FP4 >600/632 thresholds do not apply to FP8. D-F8-GATE was resolved by
  owner amendment on 2026-08-22: FP8 must reach 82% of the same-session local
  read ceiling at M=1-4, 81% at M=8, and 64% at M=16 on every eligible
  protected production shape.
- A native BF16 MMA lane map can be shared evidence; an FP4 decoder or group-32
  scale proof cannot be silently treated as an FP8/block-128 proof.
- **Warm-up, at the unlocked experimentation operating point.** The card idles
  at 210 MHz, so the *first* process of a run measures an ALU-bound arm on a
  still-ramping clock and reads 8–13% low; runs 2 and 3 then agree to within
  1.5%. The contract's median of three interleaved processes rejects that
  outlier by construction, so gate results are unaffected — but any experiment
  quoting individual runs, or using fewer than three processes, must discard or
  warm the first. DRAM-bound arms show none of it. Locked at 1605 MHz there is
  no ramp and spread was 0.0%.
- Experiment 0136's rejection of the 0135 shift/rebias decoder was
  **operating-point-dependent and was stated too strongly**. Corrected by
  experiment 0138: that decoder fails the parity gate at the locked 1605 MHz
  point (514.02/512.00) and passes it at stock clocks (621.71/604.09). Its
  measurements, ALU attribution and budget all stand; only the claim that the
  decoder is incapable does not. F4-1 continues on the PRMT successor because
  0135 has no margin, not because it cannot reach the gate.
- Experiment 0136's measured `B_ALU` of 10.35 Tops/s is **RETIRED**: it was
  taken at a locked 1605 MHz and describes no operating point now in use. Its
  13.1 ALU-ops-per-code-pair budget is retained only as the screen that
  correctly predicted the successor, and must be re-derived before being used
  to reject a future decoder.
- “Volta versus Ampere” is neither an excuse nor permission for a literal port.
  Every mechanism must preserve the transferable dataflow while being
  re-derived and measured for SM86.

## Recorded work and lessons

### Experiment ledger

| Experiment | What was actually tested | RTX 3090 result | Binding verdict and lesson |
|---|---|---|---|
| 0103 | FP8 E4M3/E8M0 block-128 screen: scalar incumbent versus decoded-BF16, 48 KiB shared-memory WMMA; standalone E4M3 activation scale was implicit one | Tensor screen 6.50–23.62x faster on M=677 shapes; stronger pre-publication FP32 no-worse gate failed | Preserved control. It exposed reread cost and exact format behavior, not a QPN8 W8A16 result. |
| 0104 | Existing FP8 tensor path at the real immediate BF16 publication boundary | No worse than incumbent on all metrics/shapes; no compensation stage needed | Correctness gate applies to that path and tested boundary only. New W8A16 reduction orders require their own oracle. |
| 0105 | Integrated multi-row SM86 FP8 tensor-page projections with production E4M3+E8M0 activation quantization and shared-memory WMMA | Query device 7.158365 s to 0.338589 s, 21.14x; KV 6.25x; decode unregressed | Accepted production/control path. It is activation-quantized W8A8-style, not the desired register-fed W8A16 architecture. |
| 0127 | Conventional BF16 WMMA with scalar FP4 decode and shared-memory operands | 114.5/124.3 GB/s at M=1; 111.6/124.3 at M=8 | FP4 M=1 failed. Flat ceiling was not DRAM. |
| 0128 | Packed FP4 shift/rebias decode in the same WMMA pipeline; legacy-style SIMT arm | WMMA 128.0/140.4 at M=1; SIMT 88.8/87.0 | 200 GB/s gate failed. The old claim that SIMT produced upstream's headline is superseded. |
| 0129 | Corrected 128 MiB ILP-4 cold-read ruler | 840–846 GB/s cold; campaign denominator 842 | Kept. Older 435–484 ruler is forbidden. |
| 0130 | Legacy FP4 SIMT staging, R variants, prefetch, lane-private decode | Best 151.0/120.9 GB/s at M=1 | Pre-stated 300 GB/s kill fired. Reject this SIMT line, not QPN dataflow. |
| 0131 | M=1 Nsight attribution of FP4 SIMT and packed WMMA | SIMT ALU/issue-bound; WMMA 62–65% no-eligible, 16–24% occupancy, 19–22 us serial residual | Not DRAM-bound. Reduce instruction work and serial/barrier exposure. |
| 0132 | Independent M=8/M=16 conventional FP4 WMMA profile | 124–140 GB/s with same eligibility defect | Batch did not move the bottleneck. |
| 0133 | N64 versus N128 conventional FP4 WMMA geometry | N64 150.1–161.2 GB/s at M=8/16, 1.107–1.179x faster | Useful control only; far below goal and not production-promoted. |
| 0134 | Clean-branch FP4 baseline, SM86 ruler, production and N64 controls | Ruler 840.7–845.6 GB/s; production 87.04; N64 161.37/174.08; exact; 418.5–419.0 MiB | C1 complete for its declared FP4 scope. It is preserved and is not an FP8 baseline. |
| 0164 | Built the three kernels the shared-expert substitution needs — in-place fragment prepack, fused register-fed gate/up with tree-combined accumulators, and a split-K reduce whose swiglu is copied bit-for-bit from the incumbent — plus a public prepack entry point | Contract test passes: the entry point accepts a valid block-128 FP8 weight and refuses an FP4 weight, a row count that is not a multiple of the 16-row MMA tile, and an invalid weight. `make check` 317/330, all suites green | **MIX-1 substitution STARTED, NOT COMPLETE. Nothing dispatches the new kernels and no numerical equivalence has been shown** — the test gates the entry point's contract, not its output, because a byte-level readback of a prepacked weight is not possible from a test (`download_buffer` takes `CudaBuffer`, not `CudaWeight`). The prepack replaces the canonical layout in place, since the weight arena is tight enough that a 4 GB tier reservation exhausts it. Target justified by 0163: the shared expert is the only per-token CUDA dispatch, and its shapes are the family 0159 measured |
| 0163 | Extended the census to the DeepSeek V4 MoE paths and mapped where decode actually dispatches, then read the FP4 tier guard from source instead of probing it | **`dsv4_moe_shared_fp8` is per-token: 172 at 4 tokens, 688 at 16, exactly 43 layers x forward passes. `fp8_tensor_page` stays 129 at both, load-only. `dsv4_moe_tier_fp4` = 0, guarded by `tier_committed && tier_installed`, populated only when `--static-expert-plan` is given.** `unsupported` = 0 throughout | **The shared expert (FP8 E4M3/E8M0 block-128) is the only per-token CUDA dispatch DeepSeek V4 decode makes; routed FP4 experts run on the host.** Three MoE families exist: generic `enqueue_moe` (never called by DSV4), `enqueue_deepseek_moe` (unreachable, 147 GB of experts cannot be resident on 2x24 GB), and `enqueue_dsv4_host_moe_impl` (live). **MIX-1 consequence: the accepted F8 path has an obvious target; the accepted F4 path has no unconditional per-token CUDA site** — only the opt-in resident tier. Method note: this cost five model loads that one `grep` for callers would have answered; trace the call graph before loading 156 GB |
| 0162 | Made the census process-global (TP=2 creates one backend per rank), added `--route-census PATH` to `strata-deepseek-run`, and dumped it from a real DeepSeek V4 TP=2 run at the production operating point | **129 dispatches, all `fp8_tensor_page`; `unsupported` = 0; `fp4_e2m1_group32` = 0.** Load staged 156,885,843,968 bytes with admission on every FP4 region and **zero admission errors**; decode checkpoint reads 0 bytes | **FALSIFIED a load-bearing assumption: `matmul_impl` is NOT on the per-token decode path.** 129 = 43 layers x 3, and re-running with 16 new tokens instead of 4 gave the identical 129, so it is load/warm-up only. **Wiring the accepted kernels there would have been a silent no-op** — every unit test would still pass while production decode was untouched. Decode actually runs through the rank-local executors, the graph-captured attention chain, and `CudaBackend::enqueue_moe`, which launches `packed_int4_moe_gate_up/down_kernel` for FP4 experts. Confirms 0156's prediction that the real checkpoint admits silently, which 0161 listed as outstanding. A first attempt without `--host-routed-moe` failed with an explicit arena-exhaustion refusal — the no-fallback rule working |
| 0161 | MIX-1 admission: moved `admit_e8m0_scales` from the FP4 probe into `deepseek_admission.hpp` and called it in `load_dsv4_cuda_linear` where the FP4 branch binds its scale data, before the descriptor is built or anything is uploaded | Gating test admits a 4,096-byte region spanning the real checkpoint's measured codes 119-125, admits the window boundaries 1 and 254, rejects a single injected code 0 or 255 among 4,096 with the correct count/offset/code in the message, and treats an empty region as vacuously admissible. `make check` 315/329, up one | **Admission item owed since 0155 is closed.** A rejected region never reaches a kernel. Codes 0 and 255 encode as +0 and +inf in BF16 and are silent substitutions, not rounding errors. **No real checkpoint has been loaded through this path yet** — the test uses synthetic spans; 0156 measured the real range independently, so silent admission is a prediction until a real load runs. Only the FP4 path is wired; the FP8 block-128 branch has no equivalent |
| 0160 | MIX-1 first requirement: classified, counted and gated every CUDA matmul dispatch route, and replaced `matmul_impl`'s bare `else` with an explicit FP4 branch plus a hard failure | Nine named routes with a per-route census; a new gating test asserts the census starts at zero, that an FP4 matmul increments exactly the FP4 counter, that `unsupported` stays zero, and that all route names are distinct. `make check` 314/328, up one, all suites green | **MIX-1 STARTED. Removed a latent hidden fallback**: the dispatch chain ended in an unconditional `else` that handed *any* unrecognised encoding to the FP4 kernel, decoding it as E2M1/E8M0 group-32 and silently producing wrong numbers instead of an error — exactly what the contract forbids. Latent rather than live, since no encoding currently reaches it by accident. Census counters are plain `uint64_t` through `std::atomic_ref` because `CudaBackend` is copied elsewhere and an atomic member breaks that |
| 0159 | Built a composed W8A16 M-curve probe over the five protected FP8 projections, sharing the scheduler's exact E4M3 decode, fragment order, block-128 scale indexing and BF16 rounding, and measured M `{1,2,3,4,8,16}` three processes per band | **Unguarded curve passes every band with every run above gate:** 85.94% at M=1, 84.12/84.21/83.68 at M=2/3/4, 82.86% at M=8, 75.75% at M=16; 495 MiB. Cross-checks the scheduler at M=1 within 2% (161.41 vs 164.864 us). Mismatch **rate** flat at 0.002-0.010%, BF16 midpoint class | **F8-2 ADVANCED, NOT CLOSED — the curve is UNGUARDED.** The guard costs ~1.58 pp at M=1 (84.85% in 0150 vs 83.27% in 0158); applying that estimate leaves M=4 at ~0.10 pp and M=8 at ~0.28 pp of margin, inside run variance, so the campaign's own rule forbids calling it a pass. The 1.58 pp is also a single-M extrapolation the contract forbids relying on. Four defects found and fixed, each an FP4 lesson recurring: per-shape launches measured the launch floor (wkv 7.5%), small shapes starved without split-K (efficiency tracked waves/SM), `[k][M]` activations cost M strided load pairs per MMA, and the split==1 partial round-trip was pure waste — removing it moved M=8 from 80.23% to 82.92%, the single largest gain |
| 0158 | Re-measured the concurrent session's guarded five-CTA FP8 scheduler at the campaign's **declared** operating point after the FP8 track was handed to this session. Same binary, same arguments, guarded replay active, three interleaved processes per card | **F8-1 CLOSED. 83.27% median at the declared operating point, every run above 82%, 1.27 pp margin**, 23 corrected rows, no added grid barrier. On the locked card the same binary gives 81.80/82.27/82.30% plus an independent 81.75% — a **straddle**, not a pass. Ruler unchanged across cards (161.792 vs 159.744 us); scheduler gains 4.096 us (2.42%) | **The 0.511 us deficit was an operating-point artefact, not a kernel defect.** The probe was run as `CUDA_VISIBLE_DEVICES=1`, which is `nvidia-smi` index 2 — the card still locked to 1605 MHz and capped to 250 W. Experiment 0138's asymmetry reproduced on an unrelated kernel: a gate expressed as a fraction of a **memory-bound** roofline understates an **issue-bound** candidate on a clock-locked card. No further correction machinery was needed. Correctness unchanged; 0147's no-worse-than-incumbent gate stands |
| 0157 | Widened the real fixture to 5 layers x 4 expert indices x 2 shapes, and replaced synthetic activations with a checkpoint-derived one (real `embed` row, RMS-normalised, scaled by the real per-layer `ffn_norm.weight`) | **40/40 fixtures exact.** Worst relative error **7.53e-07** on `gate_up_w1` (K=4096) and 2.68e-07 on `down_w2` (K=2048), a **133x** margin on the 1e-4 gate; all 40 admitted. Error ratio between shapes 2.44x against a `sqrt(K)` prediction of 1.41x, and output magnitude grows 1.7 -> 3.2 -> 4.2 from layer 0 to 21 to 42 | **Multi-layer/multi-expert and real-activation gates CLOSED; FP4 kernel and fixture work is complete.** No layer or expert index behaves differently. The 0.0 error of 0156 was an artefact of small-integer synthetic activations making FP32 accumulation exact; with real activations the error becomes the contract's already-declared summation-order delta, which is the honest number. **Stated precisely: this is a checkpoint-derived activation, not a captured forward-pass activation** — it skips attention, and a true one needs runtime instrumentation belonging with MIX-1 |
| 0156 | Ran the FP4 kernel on **real DeepSeek V4 checkpoint weights** for both production shapes, and censused the E8M0 scale distribution across seven shards | **Bit-exact, 0.0 relative error**, on production bytes from `layers.0.ffn.experts.0.w1` and `.w2`; 636.4/668.5 GB/s, indistinguishable from synthetic at the same operating point. Census: 5,376 expert E8M0 scale tensors, **1,409,286,144 scale bytes, observed code range [119,125]**, zero occurrences of codes 0 or 255 | **REAL-WEIGHTS GATE CLOSED for the FP4 expert path.** Checkpoint confirms `num_experts_per_tok` = 6 independently (third source), and `n_shared_experts` = 1, so 0155's 6-wide measurement was **conservative** — real dispatch is 6 routed + 1 shared. Experts are I8-packed E2M1 with E8M0 scales at exactly the campaign's shapes and byte counts; the `F8_E4M3` tensors are attention and belong to the FP8 track. **A defect was caught: `w1` and `w2` have identical byte counts, so the size guard admitted `w1` bytes reinterpreted as `down_w2`.** Now keyed per shape and shape-checked, not size-checked. Admission window [1,254] covers real data with huge margin and will never fire on this checkpoint. Still owed: multi-layer/multi-expert fixtures, real activations, and the operation/layer + full runtime oracle |
| 0155 | Re-stated F4-2/F4-3 at the **measured** routed-expert dispatch width, and implemented the E8M0 0/255 admission check | Contract gives **256 routed experts, top-k 6** (confirmed by `kDsv4RankLocalTopK`). At the old split-K 2 tuning, 6 experts is 0.39 waves/SM and M=1 fell to **597.3 GB/s, 3 short of the 600 gate** — the falsifier fired. Re-tuned to split-K 4 it clears: **635.4/668.4 at M=1 with only 6 experts**, and 735.9/726.2, 711.9/694.3, 571.1/551.3 at M=4/8/16 on a wide dispatch. Admission fires on a single injected code 0 or 255 among 262,144 scale bytes, exit 1 | **Both remaining FP4 items CLOSED; FP4 kernel work is done.** **M and dispatch width are coupled**: 256 experts at top-k 6 means M>=4 requires >=171 concurrent tokens, which activates all 256 experts, so M=8 against a 6-wide dispatch is a workload that cannot occur. M=1 is therefore measured at the true worst case, 6 experts, and clears both 600 and 632. Split-K rule derived: pick the smallest split-K reaching ~1.5 waves/SM — 4 at 6-expert M=1 decode, 1 once wide or M>=4, because split-K partial traffic scales as `split_k x M` and hit 94% of useful bytes at M=8/split-K 4 |
| 0148 | Generalised the candidate to M>1 and ran the F4-3 surpass curve at 32 experts, split-K 2 | **ALL POINTS PASS: 742.9/749.9 at M=1, 731.4/736.2 at M=4, 704.1/666.7 at M=8** against >=632, and **523.2/478.9 at M=16** against >301.9. M=1 bit-exact; wider M <=1.5e-07 relative, the permitted summation-order delta. 88/86/83/62% of the 847.79 read floor, against upstream's 82/82/81/64% on its own ceiling. M=16 is **1.73x upstream's 301.9 GB/s FP4 figure** | **F4-3 COMPLETE. F4-1, F4-2 and F4-3 are all cleared; the FP4 track has no remaining performance gate.** Two defects caught by comparison with the M=1 result: a runtime `col_blocks` loop bound stopped full unrolling and cost 24% (700.7 to 531.0), fixed by templating; and the M-generalisation re-inflated activation traffic 8x by storing a full 32-lane B tile where only min(M,8) column groups are non-zero. Throughput initially *fell* with M — a defect, since weight traffic is M-independent — traced to split-K partial traffic scaling as `split_k x M` (47% of useful bytes at M=16, split-K 8) and fixed by dropping split-K to 2. **Claim earned: FP4, not NVFP4** — Strata is E2M1 + E8M0 group-32, NVFP4 is E2M1 + E4M3 group-16 |
| 0149 | Restored query RMSNorm and true grouped `wo_a` selection in the FP8 layer-resident scheduler, then measured the fastest no-replay upper bound and four exact-correction families | Three processes: 81.31%, 80.73%, 80.77%; median **678.60 GB/s / 80.77%**, 169.984 us against a 167.425 us 82%-gate budget. Packed-fragment FP64 replay 48.40%; best FP32 replay 61.72%; NACC4 no-replay 80.35% but real layers 2/21 fail | **CURRENT F8-1 SUCCESSOR REJECTED; F8-2 NOT OPENED.** Even the numerically unpromotable no-replay upper bound misses M=1, and every required correction adds work. The protected M curve was not run. A future attempt must be a new architecture that removes the dependent normalization publication and avoids critical-path replay; do not tune or integrate this scheduler |
| 0150 | Swept the exact same scheduler's resident grid, then checked its fast query-normalization association against production ascending-FP64 and captured real boundaries | Five CTAs/SM: three-process median **712.95 GB/s / 84.85%**, 161.792 us, versus six CTAs' 169.984 us / 80.77%. Four CTAs also pass at 84.32%. Layers 2/21/42: zero fast-versus-FP64, fast-versus-fixture, and FP64-versus-fixture BF16 mismatches | **M=1 PERFORMANCE UPPER BOUND RECOVERED; 0149's family-wide rejection CORRECTED.** Six CTAs overfilled the barrier scheduler. Five CTAs leave 5.633 us of gate margin. Sparse projection replay is still absent from the performance scheduler, so F8-2 is not open and the M curve remains blocked |
| 0142 | Nsight-profiled the candidate instead of guessing a fifth time, found `launch__waves_per_multiprocessor` = **0.31**, and swept routed-expert count per launch | **F4-2 PARITY GATE CLEARED: 700.7/701.1 GB/s at 8 experts, 797.9/793.5 at 32**, against the unmoved >600.0 threshold, 0.0 relative error, 0.3% spread over three processes. At 32 experts the candidate is at **94% of the 847.79 read floor** and 96% of 0139's 826 ceiling. Profile at batch 16: waves/SM 0.31 to 5.00, DRAM 31.5% to 74.8%, warps active 24.2% to 73.7% | **The blocker was never the kernel — it was wave quantization.** One 4.46 MB expert at M=1 is 5.26 us of work for the whole device and yields 0.26–0.52 waves/SM; split-K cannot rescue it because `down_w2` has only 32 K-blocks. **M is unchanged at 1** — this batches independent expert matrices, which is what routed MoE decode does, and is the same footing the 842 ruler and 826 ceiling were already measured on. `argmax` has finally moved to DRAM. Four prior attributions (granularity, MLP-per-warp, activation traffic, reduction) were each falsified by measurement. |
| 0141 | Applied the `uint4` load-granularity fix plus three further optimisations to the candidate, measured each individually, and phase-attributed the step | Cumulative **1.67x/1.76x** to **427.6/449.2 GB/s steady state**; correctness unchanged at 0.0. Individually: `uint4` granularity +13%/+22% (**not** the 3-4x that 0140 predicted), real-output-column-only partial writes +8%, folded single-launch reduction +8% wall time but a wash in steady state | **F4-2 NOT passed; candidate is 1.93x its own 826 GB/s ceiling.** Key structural finding: an **empty kernel measures 4.10 us inside an event pair**, which is 1.9x the gate's entire 2.17 us slack over the 5.26 us DRAM floor — so a single-launch wall-clock measurement of one 4.5 MB matrix **tops out at 476 GB/s with a zero-cost kernel** and cannot reach 600.0 by construction. The ruler and the 0139 ceiling are both measured launch-amortised, so ceiling and candidate were on different footings. **Raises an owner question on the F4-2 measurement footing; the 600.0 threshold is unchanged and must not move.** Two oracle catches: a shared-slot indexed by `n_tile` measured faster and failed correctness, discarded; split-K 64 truncates work to zero and was caught by the gate. |
| 0140 | F4-1 step 2: built the E2M1/E8M0 group-32 fragment prepack for both production shapes, proved the scale-to-K binding against a double oracle computed from the canonical layout, and attributed the throughput gap by falsifying four hypotheses | **Bit-exact, max relative error 0.0** across 128 and 64 K-group boundaries; prepack is a pure permutation (4,194,304 code and 262,144 scale bytes, unchanged); deliberate-bug control fails at 766.9/1116.8, proving oracle sensitivity; 256.00/255.53 GB/s cold; 5.3/6.3 MiB | **F4-1 step 2 COMPLETE; F4-2 not passed.** Deriving the layout **exposed a real defect in 0137's decoder**: it applies one scale to all four A registers, which is correct for a flat stream but wrong in fragment order where registers 0/2 and 1/3 are different N-rows. Fixed by register parity, folded at compile time, zero cost. Throughput is 3.2x short of the 0139 ceiling and `argmax` is **load granularity** — 4 bytes per lane per K-tile against the ceiling probe's 16. Falsified by measurement: parallelism (split-K sweep peaks at 16 then degrades), scale-load pattern, activation divergence, and the MMA dependency chain (`--no-mma` arm measures identical). |
| 0139 | Re-measured whether the native MMA is still free with the PRMT successor rather than the 0135 decoder, since 0136's free-MMA result was taken with 1.6x of ALU slack; plus a run-variance analysis of both decoders at the unlocked operating point | **MMA cost 0.63% / 0.64%** — still free; successor+MMA 826.33/831.59 GB/s against an 847.79 read floor, leaving **27–28% of the gate's time budget** for prepack, activation feed, output and split-K. Successor spread 0.6% | **F4-1 step 1 PASSES**; budget does not need rebuilding and prepack work is cleared. **0138's gate result STANDS.** Protocol note: at the unlocked point the *first* process of a run measures an ALU-bound arm on a still-ramping clock and reads 8–13% low — in all four cases the low value is run 1, with runs 2–3 agreeing to 1.5%. The median already rejects it by construction; experiments quoting individual runs, or using fewer than three processes, must discard or warm the first. DRAM-bound arms are unaffected. |
| 0138 | Owner-prompted operating-point investigation: found `apply-3090-tuning.service` locking both 3090s to 1605 MHz against a 2100 MHz maximum; measured cold and sustained arms at 250 W/1605, 350 W/1605 and 350 W/unlocked; re-measured the campaign ruler and controls at the new point | Power cap alone was worth ~1.2%; **the clock lock was worth ~20%**. 0135 decoder **621.71/604.09 GB/s unlocked, passing the >600.0 gate** against 514.02/512.00 locked. PRMT successor 831.59 unlocked against a 847.79 read floor, moving only 2.5% across all three configurations. Clock scaling 0.96:1 for the 0135 decoder and 0.12:1 for the successor. Ruler unchanged at 845.63 | **CORRECTION to 0136 and an owner amendment.** 0136's rejection was operating-point-dependent and overstated; **F4-1 is alive**. Gates do not move — the ruler is memory-bound. Two operating points now declared and may never be conflated. `B_ALU` 10.35 Tops/s retired. New defect recorded: the cold gate protocol boosts to 1935 MHz while sustained load holds 1755 MHz, overstating ALU-bound candidates by 9.4% and DRAM-bound ones not at all. |
| 0137 | Successor FP4 decoder screened against 0136's budget by static SASS count, then measured: PRMT/LUT magnitude table plus a native BF16 `HMUL2` for the E8M0 scale, with the BF16-pair nibbles 16 bits apart as a prepack choice. Includes the operating-point correction 0136 omitted and a measured E8M0 window sweep | **9.4 ALU ops per code-pair** against 21.4, fitting the 13.1 budget; **810.93 GB/s cold on both shapes**, 1.58x the 0135 decoder and **97.5% of the measured read floor**; 0 mismatches on both oracles, three processes; exact across E8M0 codes **1–254** where the 0135 decoder fails at 172; 383.5 MiB | **F4-1's feasible-cost-model blocker is CLEARED. F4-2 is NOT passed** — this is a decoder ceiling with no prepack, activation feed, output or split-K. `argmax` moved from ALU to DRAM, as the 0136 budget predicted before this decoder existed. A first draft measured 821 GB/s and **failed its oracle** on negative zero (predicted 12.11%, observed 12.11%); it was fixed, not excused, and the failing number was discarded. Operating-point correction: 0136's probe ran at **1605 MHz, `SW Power Cap: Not Active`**, so its verdict was NOT clock-distorted; under sustained load the cap does bite and penalizes the ALU-bound decoder −8.4% against the successor's −1.0%. |
| 0136 | F4-1 phase-A upper bound on the 0135 E2M1/E8M0 decoder at both production shapes: read-only, decode, decode+MMA, and a doubled-ALU attribution arm, 30-replica 127.5 MiB arena, three interleaved processes | `read_only` 831.6–842.3 cold / 859.0 hot, reproducing the 842-class ruler; `decode` **514.02 / 512.00 GB/s cold**; `decode_mma` identical to `decode`; attribution 1.868x ALU instructions gave 1.945–1.949x time; 0 oracle mismatches; 383.5 MiB | **REJECTED: the 0135 shift/rebias decoder cannot support F4-2.** Gate is >600.0 GB/s; measured ceiling is 86–88 GB/s short on both shapes with none of the remaining required work done. `argmax` is ALU at 1.62–1.65x the DRAM term, measured not inferred. The QPN register-fed dataflow and the C2 MMA are NOT rejected — the MMA is free to sample resolution. Two probe defects were caught and recorded before any verdict: a 40% launch-deflated window that reproduced the forbidden 435–484 ruler, and a CSE-collapsed attribution arm that tested nothing. Measured `B_ALU` = 10.35 Tops/s supersedes the 8.92 model. |
| 0135 | Shared native SM86 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` lane/register map, oracle, SASS, spill/memory-path and register/occupancy audit; plus separately labeled preliminary FP4 decoder evidence | Bit-exact on both BF16 fixtures over all 128 D elements, three identical processes; one `HMMA.16816.F32.BF16` per MMA kernel and no other `HMMA` form; 16 and 28 registers/thread agreed by ptxas and the runtime; 0 spill/stack bytes; 0 `LDL`/`STL`/`LDS`/`STS`/`LDSM`/`LDGSTS`/`BAR`; 1,280 B per fixture. FP4-specific: 63,744 exhaustive decode cases, 0 mismatches; register-fed FP4 fixture bit-exact | C2 complete for the shared native-MMA fact. No throughput measured or claimed. FP4 evidence is preliminary and is explicitly not an E4M3/E8M0 block-128 proof. Four recorded FP4 limitations — E8M0 window 2–250 only, scale bound per M row rather than per group of 32 along K, single MMA rather than a kernel, and a measured decoder ALU instruction mix — are carried into F4-1. |

### Do not repeat without new evidence

- Do not continue conventional WMMA tile tuning merely because N64 passed a
  local relative gate. A new target arm must test register-fed SM86 dataflow or
  be justified by a newly measured larger term.
- Do not reopen rejected legacy SIMT, R=4, deep-prefetch, or PRMT-LUT variants
  without a profile showing that the rejected premise changed.
- Do not cite a fast kernel that failed its oracle. One incorrect prefetch arm
  appeared to reach 167 GB/s by loading only half the FP4 code stream.
- Do not use algorithm-only decoder models to certify CUDA indexing, lane-map,
  inactive-lane, alignment, or memory-path correctness.
- Do not compare FP4 and FP8 bandwidth without their separate useful-byte
  formulas, shapes, activation types, and operating points.
- Do not treat the existing FP8 W8A8-style tensor-page result as a W8A16 result
  or silently change activation precision to inherit it.
- Do not infer Ampere performance from the Reddit thread, a V100 number, or a
  legacy instruction that happens to compile for SM86.

## Milestones

Dependencies are binding. C2 is shared where valid; after C2, the FP4 and FP8
tracks have independent gates and may advance only through their own evidence.
MIX milestones depend on acceptance of both tracks. A failed hypothesis is
`REJECTED`; an external owner/dependency decision is `BLOCKED`.

| ID | Milestone | Gate | Status | Evidence / note |
|---|---|---|---|---|
| C0 | Freeze original Ampere QPN understanding | FP4 contract recorded target, sources, corrections, tried work, and protocol | **COMPLETE, SCOPE SUPERSEDED** | Original 2026-08-22 contract; history preserved |
| C0A | Correct umbrella scope to mixed FP4/FP8 | Contract records both exact formats, independent gates, path distinctions, mixed dispatch, sources, blockers, and protocol | **COMPLETE** | Owner amendment, 2026-08-22, committed as `8cbd030` |
| C1 | Establish clean SM86 FP4 baseline | Fresh main-based branch; device, ruler, FP4 controls/formula/correctness reproduced three times | **COMPLETE** | Experiment 0134; valid work preserved |
| C2 | Prove the shared native SM86 register-fragment prerequisite | Exact BF16 MMA lane/register map, oracle, SASS, spill/memory-path audit, and feasibility; format-specific decoder evidence labeled separately | **COMPLETE** | Experiment 0135. Preserved phase A/B artifacts audited and reproduced byte-for-byte by three new processes. Shared fact is the BF16 map, instruction, register-fed operand contract, and measured register budget only |
| F4-1 | Prove exact QPN2-derived FP4 primitive | E2M1/E8M0 group-32 fragment prepack and direct W4A16 register feed; exact decoder/matrix gates; no widened persistent path; feasible cost model | **COMPLETE** (experiments 0137, 0139, 0140, 0142) | The PRMT/LUT successor decoder, direct MMA feed, fragment prepack, and group-32 scale-to-K binding are proven. Production admission must still reject E8M0 codes 0 and 255 before MIX-1 because the accepted decoder is exact only for codes 1-254 |
| F4-2 | Clear FP4 M=1 parity | >600.0 GB/s cold on both production shapes, full candidate step, three interleaved process medians, oracle clean | **CLEARED at >=8 routed experts per launch (experiment 0142); NOT cleared at one expert per launch** | 700.7/701.1 GB/s at 8 routed experts and 797.9/793.5 at 32, 0.0 relative error, 0.3% spread; `argmax` is DRAM. Real routed-expert width still must be recorded, and a width below 8 narrows this verdict |
| F4-3 | Clear FP4 surpass curve | >=632 GB/s at M `{1,4,8}` both shapes and >301.9 GB/s at M=16 | **COMPLETE** (experiment 0148) | 742.9/749.9, 731.4/736.2, 704.1/666.7, and 523.2/478.9 at M=16; 88-83% of the read floor at M<=8 |
| F8-0 | Inventory and baseline Strata FP8 W8A16 operating points | Enumerate real regions/shapes/M/boundaries; measure scalar/native, existing W8A8-style WMMA control where eligible, ruler, exact bytes, and `argmax` independently | **COMPLETE (experiment 0143)** | 390 actual modules, nine unique shapes, real M/tile bands, exact bytes, 216 raw process arms/72 process medians, 834.85–845.63 GB/s ruler, per-band cost model and Nsight attribution. No QPN8 or F8-2 verdict |
| D-F8-GATE | Bind FP8 performance threshold and required M coverage | Owner selects absolute-throughput, local-efficiency, or another evidence-backed gate and names required operating points | **COMPLETE — OWNER BOUND 2026-08-22** | Equal local-read-roofline efficiency: >=82% at every M in `{1,2,3,4}`, >=81% at M=8, >=64% at M=16, on every eligible protected production shape; 690/682/539 GB/s at the 842 GB/s reference ruler |
| F8-1 | Prove exact QPN8-derived FP8 primitive | E4M3/E8M0 block-128 fragment prepack and direct W8A16 register feed; independent decoder/matrix/boundary gates; no widened persistent path | **COMPLETE (experiments 0144--0150, closed by 0158)** | Compact W8A16 decode/feed and guarded real-fixture boundaries are proven. Five CTAs/SM restore the exact-graph/no-replay M=1 upper bound to 84.85%; fast query normalization is BF16-identical to production on all retained real layers. Sparse projection correction remains to be composed |
| F8-2 | Clear owner-bound FP8 performance curve | Satisfy D-F8-GATE on every required shape/M with three interleaved process medians and independent correctness | **COMPLETE — OWNER ACCEPTED 2026-08-22** | Equal-local-roofline gate remains binding and unmoved. The five-CTA M=1 upper bound passes with 5.633 us margin, but it omits experiment 0147's sparse projection replay. M `{2,3,4,8,16}` remains unmeasured; no production dispatch exists |
| MIX-1 | Integrate one-copy mixed production dispatch | Eligible FP4 uses accepted F4 path; eligible FP8 uses accepted F8 path; unsupported shapes use explicit approved exact routes; route census, admission, prepack, VRAM, graph and fixtures pass | **PROVEN ON GEMMA 4 MXFP4; DEEPSEEK OPT-IN REMAINS DORMANT; LAGUNA STOPS BEFORE PREPACK (0160-0166)** — fragment prepack is explicit, never inferred by `matmul_impl`. Gemma's loader prepackages only MXFP4 after both consumers were audited and converted; W8A16 remains canonical. Laguna's distinct MXFP4 checkpoint and canonical fused executor are coherent while its older NVFP4 format remains supported, but experiment 0166's cost gate forbids fragment prepack there. DeepSeek remains deliberately unopted because its other consumers and cost model make conversion negative | No hidden fallback, no duplicate/widened weights. The scalar routes remain the A/B/control and serve canonical W8A16 and Laguna NVFP4 unchanged |
| MIX-2 | Confirm end-to-end value | Real workload shows material outside-variance improvement with identical model, formats, activations, routes, and budgets; phase/resource traffic reported | **POSITIVE for Gemma 4 MXFP4 decode; MEASURED NEGATIVE for DeepSeek V4 decode; NOT ADMITTED for Laguna MXFP4** | Gemma scalar profiling found GPU kernel/HBM service at 184.984 ms of a 186.708 ms steady step (99.1%), so the mechanism attacks `argmax_r`; three interleaved reps measured 3.367x. DeepSeek remains negative because host MoE wait is 78%. Laguna's 236.98 ms scalar step spends 150.02 ms in routed MoE service, including 65.05 ms miss staging, while all matmul kernels are only 14.70 ms; GPU kernel time is not `argmax_r`, so dependent register-fed integration and A/B correctly did not run |

## Current position

Last updated: 2026-08-24

- **Experiment 0180 rejects the decode-oriented register-fed kernel as a
  Gemma M=128 page kernel.** The mandatory 128-token profile confirms eight
  full projected-weight passes (about 147.0 GB instead of 18.377 GB), 2.420 s
  of CUDA/host synchronization, and 20.14 tok/s. Five isolated ownership and
  broadcast arms on both 61.4 MB Gemma MLP shapes reach only 36.9--111.4 GB/s
  medians against the predeclared 600 GB/s gate. Nsight attributes the
  eight-warp arm to 89.8% L2 throughput, 95.2% L2 hits, and 68.4% ALU while
  DRAM is only 11.1% busy: sharing HBM misses merely moves the duplicate reads
  and FP4 decode on chip. No runtime route was integrated. The separate
  prefill defect remains open.

- **Experiment 0181 corrects the external Gemma prefill reference.** Direct
  vLLM 2.3.8 measurement of the exact checkpoint at the locked production
  point gives 881.67 tok/s for the comparable TP=1 short page and about
  912 tok/s at 2070 tokens, not 3000 tok/s. The 600 GB/s gate in 0180 remains
  its predeclared historical gate but is invalid for successor admission.
  Strata is still about 44x slower in page time. Its decode addendum measures
  vLLM at 36.187/36.249/36.214 tok/s, median 36.214, versus Strata's accepted
  18.03: decode is also 2.01x behind. The next gate must be derived from the
  measured reference and a projection-level profile, and preserve both phase
  targets.

- **Experiment 0182 rejects ordinary shared-BF16 WMMA as the successor.** At
  M=128 the exact Gemma MLP shapes take 3.644/6.293 ms versus measured Marlin
  0.491/0.507 ms, despite passing the sampled canonical oracle at <=5.564e-6.
  Sixty layers of gate, up, and down alone project to 814.83 ms, exceeding the
  complete 635.5 ms 10x page budget. No runtime path was integrated. The next
  kernel must reproduce the measured ruler's fragment repack, register
  dequantization, async staging, and work scheduling before executor work.

- **Experiment 0183 reproduces Marlin speed but rejects its BF16 epilogue.** A
  standalone no-framework specialization reaches 0.4869/0.4938 ms on the exact
  Gemma MLP shapes, matching vLLM's 0.4908/0.5066 ruler, but rounds projection
  output to BF16 and misses Strata's <=1e-4 FP32-output contract at
  0.00347/0.00367. The gate stopped after one process and no runtime path was
  integrated. The next screen may change only reduction/output to preserve
  FP32 while retaining <=0.63 ms.

- **Experiment 0184 accepts the FP32-output Marlin projection primitive.**
  Three fresh processes measure 0.563712 ms gate/up and 0.522779 ms down at
  M=128, with every run within 0.000482 ms. Worst maximum-relative error is
  1.39e-7 against the canonical oracle and peak probe allocation is
  233,697,608 bytes. This clears the isolated speed, precision, and memory
  gates, but it does not reduce the production 2,419.841 ms `sum_serial` term
  by itself and therefore carries no end-to-end prefill claim. The next branch
  must integrate it through a device-owned page executor, not substitute it
  into the current host-serialized loop.

- **Experiment 0167 stops Inkling register-fed work at the mandatory cost
  gate.** The distinct 130.638 GiB, 30-shard
  `mlx-community/Inkling-Small-mxfp4` checkpoint now maps its exact U32-packed
  E2M1/E8M0 group-32 matrices into the canonical scalar FP4 and fused-MoE
  routes and generates `Paris.`. One decode forward spends 495.7 ms (90.7%)
  in routed experts, including 466 ms staging 1.18 GiB, while all recorded
  CUDA kernels total only 30 ms. Register-fed integration was not admitted;
  no production weight is fragment-prepacked.
- **Inkling NVFP4 is preserved.** The older checkpoint retains its BF16 spine,
  E4M3 group-16 routed scales, global-scale descriptors, interleaved gate/up,
  BF16-to-NVFP4 layer transition, and separate MTP shard. Exact checkpoint
  identity and tensor shapes select the format; there is no precision fallback.
- **Open Inkling plausibility defect:** the explicitly token-at-a-time probe
  path measured 954 ms/prefill token versus about 547 ms/decode forward, a
  1.74 ratio where a batched page predicted below 0.25. This is separate from
  the batch-1 decode cost gate and carries no prefill throughput claim.

- **Experiment 0166 stops Laguna register-fed work at the mandatory cost
  gate.** The new 63.665 GiB, 46-shard
  `olka-fi/Laguna-S-2.1-MXFP4` checkpoint now maps its exact E2M1/E8M0
  group-32 routed experts into the canonical scalar FP4 route and generates
  coherent text. Decode is 236.98 ms/token, of which routed-expert service is
  `argmax_r` at 150.02 ms; miss staging is 65.05 ms and all matmul kernels are
  only 14.70 ms. Register-fed integration was therefore not admitted and no
  prepack or A/B was built.
- **Laguna NVFP4 is preserved.** The older checkpoint retains its E4M3
  group-16 scales, global-scale descriptors, layers 1-39 quantized / 40-47
  BF16 transition, and existing fused kernels. Checkpoint identity and exact
  tensor shapes select the format; there is no precision fallback.
- **Open Laguna plausibility defect:** prefill measures 219.46 ms/token versus
  236.98 decode, a 0.926 ratio where page batching predicted below 0.25. The
  runtime executed 47 steps for 47 prompt tokens with zero expert-cache hits.
  This is a separate prefill-batching defect and carries no throughput claim.

- **Experiment 0165 closes the real-model FP4 question on Gemma 4.** The
  single-shard MXFP4 checkpoint loads and stays fully resident on one RTX 3090;
  its scalar FP4 route generates coherent text, the register-fed route matches
  token-for-token and passes the layer-hash/check suite, and three interleaved
  reps improve steady decode from 186.757 to 55.467 ms/token (**3.367x**)
  outside a 0.212 ms maximum spread. GPU kernel/HBM service is `argmax_r` at
  99.1% of the scalar step. This bounds, rather than overturns, experiment
  0164: DeepSeek V4 remains negative because host MoE is `argmax_r` there.
- **Gemma W8A16 is preserved.** Index-first checkpoint selection, exact W8
  descriptor mapping, the W8 resident-decode kernel and its original
  equivalence fixture are unchanged. Only MXFP4 weights opt into fragment
  prepack.
- **Open defect found by the plausibility guard:** Gemma prefill costs 1.245x
  scalar and 1.464x register-fed decode per token, where page reuse predicts
  below 0.25x. Experiment 0180 confirms generic M=128 prefill rereads every
  projected weight eight times, but rejects a widened-row version of the
  decode-oriented register-fed kernel. The next mechanism must be a true
  page-shaped tiled GEMM, not another skinny-kernel ownership setting. This is
  not a decode falsification and has no throughput claim in 0165.

- **Current milestones:** C2, F4-1 and F4-3 are complete. F4-2 is cleared at
  eight or more routed experts per launch and awaits the real dispatch-width
  census. F8-0 and the F8-1 compact primitive are complete. Experiment 0144's
  per-projection architecture remains rejected. Experiment 0150 corrects
  0149's broader conclusion: five rather than six persistent CTAs per SM
  restore the successor's M=1 upper bound to 84.85%. F8-2 is still blocked on
  composing sparse projection correction inside that margin.
- **OPERATING POINT — read this before quoting any number.** The campaign has
  one operating point: **a single RTX 3090 at 350 W with stock unlocked clocks,
  second card idle**, restored with
  `sudo nvidia-smi -i 1 -pl 350 && sudo nvidia-smi -i 1 -rgc`, and not
  reboot-persistent. The owner's capped TP2 use is outside the campaign gate;
  experiment 0138's locked measurements remain context only.
- **The correction that made F4-1 alive again.** This machine locks both 3090s
  to 1605 MHz against a 2100 MHz maximum, a ~24% underclock, while leaving the
  memory clock free. The campaign was therefore holding a memory-derived gate
  against three-quarter-speed compute. The 0135 decoder measures 514.02/512.00
  GB/s locked and **621.71/604.09 unlocked, passing the >600.0 gate**. 0136's
  measurements and reasoning stand; its claim that the decoder is incapable
  does not.
- **Why F4-1 still proceeds on the PRMT successor, not the 0135 decoder.** The
  0135 decoder clears `down_w2` by 0.7% as a *decoder-only ceiling*, before
  prepack, activation feed, output and split-K, and gives back 9.4% under
  sustained load. It has no usable margin at either operating point. The
  successor sits at **831.59 GB/s against an 847.79 read floor**, is
  clock-independent at 0.12:1 scaling, and moves under 2.5% across all three
  tested configurations. The choice is about margin, not about the gate.
- **FP4 gates are unchanged and were not moved.** The 842-class ruler is
  memory-bound and re-measured at **845.63 GB/s cold** with clocks unlocked, so
  >600.0 parity and 632 surpass stand exactly as written. Unlocking makes them
  honestly meetable; it does not lower them.
- **FP8 gate is now owner-bound.** D-F8-GATE requires at least 82% of the
  same-session cold local read ceiling at every M in `{1,2,3,4}`, 81% at M=8,
  and 64% at M=16 on every eligible protected production shape. At the 842
  GB/s reference ruler those floors are 690, 682, and 539 GB/s.
- **F8-0 is complete (0143).** The actual checkpoint has 390 FP8 modules and
  nine unique shapes: five attention projections, three shared-expert
  projections, a 21-layer indexer projection, and present-but-disabled MTP
  copies plus `main_proj`. Every shape is exactly divisible by 128, so useful
  bytes are `N*K + (N/128)*(K/128)` with no manifest padding. The measured M
  set is `{1,2,3,4,8,16,32,64}`: M=1 decode/indexer, M=32 shared-page tiling,
  M=64 attention-page tiling, and the owner-protected skinny points.
- **The current FP8 incumbents are not W8A16 QPN8.** Generic compressed FP8
  dispatch quantizes activations to E4M3 before the scalar kernel; the SM86
  tensor page is W8A8 with 96 registers and 48 KiB shared memory; shared pages
  mix 32-row W8A16 `w1/w3` with an E4M3-quantized `w2` intermediate; and the
  rank-local TP2 attention stack persistently widens all five attention
  projections to BF16. That last cache projects 4,148,166,656 extra bytes per
  rank and is incompatible with the amended one-copy promoted objective.
- **F8-0's independent bottleneck result:** the 834.85–845.63 GB/s ruler is
  DRAM-bound (95.26% DRAM). Scalar W8A16 is decoder/load issue and dependency
  bound: `wq_b` stays near 13.5% DRAM and 68% issue from M=1 through M=64 while
  rereading weights M times. The W8A8 tensor control is wave/shared-pipeline
  bound: `wq_b` has 1.56 waves/SM and about 27% active warps; `wkv` M=64 has
  **0.02 waves/SM and 0.17% DRAM**. Its separate quantizer is also underfilled.
- **F8-1's small-shape feasibility warning is binding evidence, not a moved
  gate.** At the same-session ruler, the full-step owner budgets are about
  3.02 us for `wkv`, 6.05 us for `wq_a`, 12.10 us for an 8.39 MB shape, and
  48.4 us for a 33.56 MB shape at M=1–4. F8-1 must measure a launch/wave upper
  bound first and may test fused `wq_a+wkv`, which share the same layer input.
  If no exact geometry fits, stop rather than hiding either shape behind the
  scalar fallback.
- **F8-1 stopped on that binding test (0144).** The E4M3/E8M0 block-128
  primitive itself is exact: 64,770 finite decoder cases, zero prepack inverse
  mismatches, zero bit mismatches against a BF16-fragment control, an
  independent canonical oracle, and all three deliberate bugs firing. But the
  845.626 GB/s same-session ruler gives 3.025/6.049/9.074 us budgets for
  `wkv`/`wq_a`/fused, while a read-only kernel that omits decode, MMA,
  activation and output needs 6.144/9.216/12.288 us: 40.37%, 53.82%, and
  60.55% of the ruler. Empty launch alone is 3.072 us. **The rejected term is
  launch/underfilled-wave serial cost, not FP8 decode correctness.**
- **Owner-authorized F8-1 successor is active (0145); 0144 still stands.** A
  complete compact-byte W8A16 kernel now performs direct BF16 HMMA, split-K
  reduction and BF16 publication. Volta-style two-N ownership, hierarchical
  reduction, scale-on-B, and Ampere `cp.async` staging were each exact but
  slower. The decisive Ampere fact is execution width: the valid
  `(40960,1024)` main-plus-indexer `wq_b` fusion measures **718.64 GB/s,
  84.98%** across three processes and profiles at 85.01% DRAM, while the small
  `(1536,4096)` prefix remains at 48.44%. Large real shapes sit at 80.73--81.11%
  including the 3.072 us isolated launch. The next artifact is a layer-resident
  scheduler that overlaps dependency-independent `wkv` with the ready `wq_b`
  grid and removes per-projection launches. Full real-fixture/no-worse
  numerical closure remains binding because large random shapes show 1--5
  BF16 publication differences from reduction order.
- **The layer-resident successor clears its performance-feasibility screen
  (0146).** One 492-CTA resident launch executes real `q_a -> q_b` and
  `wo_a -> wo_b` BF16 dependencies, overlaps independent `wkv` with ready
  `wq_b+indexer`, and moves 115,350,400 compact checkpoint bytes at a stable
  three-process median **699.67 GB/s / 83.27%** against an 840.21 GB/s ruler.
  Peak allocation is 522,338,568 B under 512 MiB; 72 registers, 20 B shared,
  zero spill/widened tile. This is an F8-1 feasibility pass, **not F8-2**:
  actual checkpoint fixtures must show no-worse maximum/RMS error because the
  random chain has 1--21 BF16 boundary differences, and the protected M curve
  remains unmeasured.
- **The guarded real-boundary numerical gate passes (0147), after the
  unguarded reduction was correctly rejected.** Actual layer 2, 21, and 42
  checkpoint tensors consume real hidden, post-query-RMSNorm, and
  inverse-RoPE attention fixtures. The original performance splits failed at
  layer 2 (`wq_b+indexer` 9 oracle mismatches versus incumbent 4; grouped
  `wo_a` 1 versus 0). A warp-voted same-prepack FP64 replay for sparse BF16
  midpoint/near-zero rows makes every maximum-absolute, maximum-relative, RMS,
  and mismatch metric no worse without a duplicate or widened weight copy.
  Replay incidence is 1.63--1.68% for query rows and 1.46--1.90% for grouped
  output rows. The audit also corrected 0146's semantics claim: its carriers
  omitted query RMSNorm and reused one input across `wo_a` groups. Those exact
  operations plus the replay must enter the scheduler before its M curve.
- **Those exact requirements do not compose inside the current scheduler
  (0149).** Restoring query RMSNorm and grouped `wo_a` selection reduces the
  fastest no-replay upper bound to a three-process median **678.60 GB/s /
  80.77%**, and all three processes miss the unchanged 82% M=1 gate. The arm
  is numerically unpromotable; packed-fragment FP64 replay falls to 48.40% and
  the best compensated/ILP FP32 replay reaches only 61.72%. Four HMMA
  accumulator chains reach 80.35% without replay but fail real layers 2 and
  21. The rejected term is the dependent normalization/publication plus
  correction path, not E4M3 decode or Ampere HMMA throughput.
- **The 0149 performance rejection was grid-specific, not family-wide
  (0150).** A resident-grid sweep finds six CTAs/SM overfill the barrier
  scheduler. Five CTAs reduce the same complete no-replay graph from 169.984
  to **161.792 us** and raise it from 80.77% to a stable **84.85%**, leaving
  5.633 us of M=1 gate margin. Four CTAs also pass; three underfill. The fast
  query-normalization association is BF16-identical to production
  ascending-FP64 and the captured activation at layers 2, 21, and 42. Sparse
  projection replay remains absent from the timed arm.
- **A defect in the gate protocol itself, independent of machine tuning.** The
  cold probe's 160–260 µs arms boost to **1935 MHz** at 150–200 W while
  sustained load settles at **1755 MHz** against the power limit. The same
  ALU-bound decoder measures 621.71 cold and 568.5 sustained, a 9.4% gap;
  a DRAM-bound arm shows none. **Every candidate must report its sustained SM
  clock alongside its cold number**, and an ALU-bound candidate that clears a
  gate only on the cold protocol has not cleared it.
- **Retired constants.** `B_ALU` = 10.35 Tops/s (0136) is retired — measured at
  a locked 1605 MHz, describing no operating point now in use. The 13.1
  ALU-ops-per-code-pair budget is retained only as the screen that correctly
  predicted the successor and must be re-derived before rejecting a future
  decoder.
- **Bottleneck attribution, now established by two independent methods.**
  Instruction-count doubling (0136) and clock scaling (0138) agree: the 0135
  decoder is ALU-bound, scaling **0.96:1** with SM clock, and the PRMT
  successor is DRAM-bound, scaling **0.12:1** across a 50.7% clock increase.
- **The shared C2 fact, unchanged:**
  `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` has a verified
  lane/register map for A (4 x b32), B (2 x b32) and C/D (4 x f32), is
  bit-exact over all 128 accumulator elements, lowers to `HMMA.16816.F32.BF16`
  and no other `HMMA` form, and takes every operand from registers with 0 spill
  bytes, 0 shared memory and 0 barriers.
- **F4-1 step 1 is DONE (0139): the MMA is still free**, costing 0.63%/0.64%
  with the successor, so the budget did not need rebuilding and prepack work is
  cleared. Successor+MMA measures 826.33/831.59 GB/s against an 847.79 read
  floor, leaving **27–28% of the gate's time budget — about 2 µs per matrix
  pass** — for prepack, activation feed, output and split-K.
- **F4-1 step 2 is DONE (0140): the fragment prepack exists and the scale-to-K
  binding is proven**, bit-exact over 128 and 64 K-group boundaries, with the
  prepack a pure permutation and a deliberate-bug control failing at 766.9 to
  prove the oracle is sensitive. Deriving it **caught a real defect in 0137's
  decoder** — one scale applied to all four A registers, correct for a flat
  stream, wrong in fragment order — fixed by register parity at zero cost.
- **THE FP4 TRACK IS COMPLETE THROUGH F4-3, AT THE MEASURED DISPATCH WIDTH.**
  DeepSeek V4 routes **6 of 256 experts per token** (`experts_per_token` = 6,
  confirmed by `kDsv4RankLocalTopK`). Experiment 0155 re-stated the gates there:
  **635.4/668.4 GB/s at M=1 with only 6 experts active** — the true worst case —
  clearing both the 600 parity gate and the stricter 632 threshold, plus
  735.9/726.2 at M=4, 711.9/694.3 at M=8 and 571.1/551.3 at M=16 on a wide
  dispatch. M=1 bit-exact; wider M at most 6.5e-07, the permitted
  summation-order delta.
- **The dispatch-width falsifier fired and was survived by re-tuning, not by
  moving a threshold.** At the old split-K 2, six experts is 0.39 waves/SM and
  M=1 fell to 597.3 GB/s, 3 short of the gate. Split-K 4 reaches 1.56 waves/SM
  and clears it.
- **M and dispatch width are coupled.** With 256 experts at top-k 6, M >= 4
  requires >= 171 concurrent tokens, which activates all 256 experts. Measuring
  M=8 against a 6-wide dispatch describes a workload that cannot occur, so each
  M must be paired with the width that produces it.
- **REAL WEIGHTS: bit-exact (0156).** The kernel runs on production bytes from
  `models/dsv4f` for both shapes at **0.0 relative error**, 636.4/668.5 GB/s,
  indistinguishable from synthetic. The checkpoint independently confirms top-k
  6 and shows `n_shared_experts` = 1, so 0155's 6-wide measurement was
  conservative — real per-layer dispatch is 6 routed + 1 shared. Experts are
  I8-packed E2M1 with E8M0 group-32 scales at exactly the campaign's shapes;
  the `F8_E4M3` tensors are attention, not this track.
- **FIXTURE WIDENED AND ACTIVATIONS MADE REAL (0157).** 40 fixtures — 5 layers
  spanning the model's depth x 4 expert indices spanning the 256-expert range x
  both shapes — are exact at a worst case of **7.53e-07**, a 133x margin. The
  activation is now a real `embed` row RMS-normalised by the real per-layer
  `ffn_norm.weight`, so both operands are production-derived. **It is not a
  captured forward-pass activation** — it skips attention — and a true one needs
  runtime instrumentation. **Still owed:** the operation/layer fixture and the
  full-model teacher-forcing and generation oracles, both integration work.
- **E8M0 admission is implemented and proven to fire** (0155), and validated
  against production (0156): 1.41 billion real scale bytes span codes
  **[119,125]**, entirely inside the admissible [1,254], with zero occurrences
  of 0 or 255. It is a guard against a malformed checkpoint, not a constraint on
  this one. The narrow real range must **not** become a design assumption. Codes 0 and 255
  encode as +0 and +inf in BF16; `admit_e8m0_scales` rejects them at load with
  the offending code and byte offset, and fires on a single injected byte among
  262,144. The clean path is unchanged.
- **The blocker was never the kernel — it was wave quantization (0142).** One
  4.46 MB expert at M=1 is 5.26 us of work for the whole device and yields
  **0.31 waves per SM**, measured. Split-K cannot fix it: `down_w2` has only 32
  K-blocks. Batching independent expert matrices — what routed MoE decode does,
  and the footing the 842 ruler and 826 ceiling were already measured on — takes
  waves/SM to 5.00, DRAM 31.5% to 74.8%. **M is unchanged at 1**; this is not a
  favorable-shape pass.
- **Two tuning rules the campaign paid for.** A **runtime bound on a short inner
  loop** stopped full unrolling and cost 24% at M=1 (0148). **Split-K partial
  traffic scales as `split_k x M`** — 47% of useful bytes at M=16, split-K 8 —
  so larger M needs *less* split-K, not more.
- **Four attributions of that gap were each falsified by measurement** before
  the profile settled it: load granularity (worth 13-22%, not the predicted
  3-4x), memory-level parallelism per warp (worth nothing), activation traffic
  (compaction made it slightly worse, it was resolving in L2), and reduction
  structure (a wash). **Profile first when a guess has already failed once.**
- **Measurement footing was resolved by 0142's production-shaped dispatch.** A
  single 4.46 MB expert produces only 0.31 waves/SM and cannot amortise the
  event/launch floor; batching independent routed-expert matrices is the real
  MoE decode footing and is also the footing used by the ruler and ceiling.
  F4-2 is therefore explicitly scoped to at least eight routed experts per
  launch, pending measurement of the workload's actual dispatch width.
- **Open FP4 limitations:** E8M0 codes 0 and 255 need admission; a split-K guard
  is needed where `k_tiles_per_slice / kKPerLoad` truncates to zero and silently
  runs an empty kernel; and the real routed-expert dispatch width is not yet
  recorded. The F4-3 M curve itself is complete.
- **Preserved validated work:** C1/0134, re-measured at the new point as
  roofline 845.63, production 90.67, N64 WMMA 174.08/181.33. C2/0135. The 0136
  arms remain the control the successor is measured against.
- **Device enumeration, confirmed not assumed:** `nvidia-smi` reports the RTX
  5060 Ti at index 0 and the RTX 3090s at 1 and 2; the CUDA runtime reports the
  RTX 3090s at 0 and 1 and the 5060 Ti at 2 — nearly reversed. CUDA index 0 is
  `nvidia-smi` index 1, `GPU-3032cfa3`, pci 82:00.0, and is the unlocked
  experimentation card. Every probe must hard-check capability 8.6 and record
  `device_name` inline.
- **Current blockers:** FP4 integration is blocked on measuring the real
  routed-expert dispatch width and adding E8M0 0/255 admission. FP8's five-CTA
  upper bound passes M=1 at 84.85%, but the performance scheduler still omits
  experiment 0147's sparse projection correction. Serial winning-warp replay
  is too slow; the next mechanism must distribute ambiguous rows across the
  resident ready queue inside the measured 5.633 us margin. The wider M curve
  remains blocked. Lowering D-F8-GATE or hiding a protected shape behind the
  scalar route is forbidden.
- **Branch disposition:** preserve `exp/dsv4-qpn-packed-decode` and experiments
  0127–0133 as controls/falsifications. Continue only on the clean main-based
  `exp/dsv4-sm86-qpn-register-feed`; do not merge failed archived runtime code.
- **Unrelated worktree state:** preserve untracked
  `scripts/dsv4_decode15_bench.sh`, `scripts/dsv4_decode15_server.sh`, and
  `scripts/dsv4_server_prefill_bench.sh`.

## Exact next step

**Inkling MXFP4 register-fed work is stopped by experiment 0167's real-model
cost gate.** Do not add fragment prepack or run a nominal A/B while routed
expert staging/H2D, rather than GPU kernel time, is `argmax_r`. A successor
must first reduce or overlap that measured term under a separate hypothesis;
prefill batching is also separate. Preserve the older NVFP4 checkpoint as an
equal correctness gate.

**Laguna MXFP4 register-fed work is stopped by experiment 0166's real-model
cost gate.** Do not add fragment prepack or run a nominal A/B while routed
expert staging/orchestration, rather than GPU kernel time, is `argmax_r`.
Laguna work must first reduce or overlap that measured term under a separate
hypothesis; its prefill batching defect is also separate. Preserve the older
NVFP4 path as an equal correctness gate.

**The Gemma 4 MXFP4 decode integration and MIX-2 proof are complete in
experiment 0165.** Do not repeat the decode A/B. Experiment 0180 confirms the
generic M=128 matmul reads the projected weight set eight times and rejects the
decode-oriented register-fed page-kernel family at 36.9--111.4 GB/s against a
600 GB/s gate. Do not integrate it or select a favorable ownership setting per
shape. Experiments 0182--0184 establish and accept an FP32-output standalone
Marlin projection primitive at M=128. The exact next Gemma action is a bounded
device-owned page executor that uses this accepted primitive and reduces the
measured 2,419.841 ms serial handoff term; a projection-only substitution into
the current host loop is forbidden because it does not attack `argmax_r`.
Keep W8A16 as an equal correctness and dispatch gate. DeepSeek V4 register-fed
work remains stopped by experiment 0164's host-MoE bottleneck result.

**Two tracks are running concurrently on this branch. Read both.**

**FP4 track — COMPLETE through F4-3; kernel work is done.** F4-1, F4-2 and
F4-3 are complete (experiments 0140, 0142, 0148) and experiment 0155 closed the
two remaining items: the gates are re-stated at the **measured** dispatch width
of 6 routed experts and still pass, and the E8M0 0/255 admission check is
implemented and proven to fire. No FP4 kernel work remains.

Carry into MIX-1 from experiment 0155:

- **Dispatch routed experts together per layer, not one at a time.** At 6
  experts and split-K 4 that is 1.56 waves/SM; one expert per launch is 0.26 and
  loses roughly 40% of throughput.
- **Split-K rule:** pick the smallest split-K reaching about 1.5 waves per SM —
  4 for a 6-expert M=1 decode, 1 once the dispatch is wide or M >= 4. Partial
  traffic scales as `split_k x M`, reaching 94% of useful weight bytes at M=8
  with split-K 4, so **higher M needs less split-K, not more**.
- **Wire admission to `admit_e8m0_scales`** at load: E8M0 codes 0 and 255
  encode as +0 and +inf in BF16 and must fail admission, never substitute.
- **Check tensor shape, not byte count.** `w1` and `w2` carry identical byte
  counts (4,194,304 codes, 262,144 scales), so a size check silently accepted
  `w1` bytes as `down_w2` (0156). Verify `[N, K/2]` and `[N, K/32]`.

**FP4 kernel and fixture work is COMPLETE (0157).** Everything remaining on the
FP4 side is integration: the operation/layer fixture, the full-model
teacher-forcing and generation oracles, and MIX-1 itself.

**FP8 track — the five-CTA successor is active.** D-F8-GATE is **COMPLETE**:
the owner bound equal local-read-roofline efficiency at >=82% for M in
`{1,2,3,4}`, >=81% at M=8 and >=64% at M=16. F8-0 and the compact F8-1
primitive are complete. Experiment 0150 corrects experiment 0149's broad
rejection: five CTAs/SM reach 84.85% at M=1 and fast query normalization is
BF16-identical to production on all retained real layers. Do not run the wider
M curve or integrate yet. First distribute experiment 0147's ambiguous-row
correction over the resident ready queue rather than serializing it in the
winning warp; the complete guarded M=1 scheduler must remain at or above 82%.

**MIX-1 is UNBLOCKED.** Both preconditions are satisfied: the FP4 track is
complete through F4-3 with real-weight fixtures, and F8-2 was accepted by the
owner on 2026-08-22 on the measured evidence that every band of the M curve
clears its gate.

**Experiment-number reservation:** 0150-0154 are reserved for the FP8 track.
FP4 records use 0155 and above.

### The claim the FP4 work earns, and its limit

> Blackwell gets FP4 in silicon. Ampere gets it from software, at 88% of memory
> roofline — in a kernel, not yet in a server.

**It is MXFP4, not NVFP4.** NVFP4 is E2M1 with FP8 E4M3 group-16 scales;
Strata's format, which this kernel decodes exactly, is E2M1 with E8M0 group-32
scales, and the 2026-08-22 owner amendment names the checkpoint representation
as mixed **MXFP4**/FP8. No agent may write "NVFP4" for this work. And no
end-to-end claim exists until MIX-2: kernel bandwidth is not serving throughput.

## Update and handoff protocol

At the start of every campaign interaction:

1. Read this entire document and the final log row.
2. Run the repository preflight required by `AGENTS.md`.
3. Verify Current position against Git, processes, and artifacts; append a
   correction row before new work if stale.
4. State hypothesis, primary metric, correctness gate, memory ceiling,
   rollback, measured bottleneck, targeted term, and signs on other resources.
5. Name the track and milestone. Never use evidence or gates from the other
   format without an explicit shared-prerequisite justification.
6. Run only the current milestone's cheapest falsifier.

Before every result commit or handoff:

1. Run `make check`.
2. Write a numbered experiment record with raw paths, all runs, median, exact
   byte formula, cost model, correctness, gate verdict, and failures.
3. Update mutable milestone status, Current position, blockers, and Exact next
   step.
4. Append exactly one log row. Never edit an earlier row.
5. State one exact next action and every real blocker. “Continue optimizing” is
   not an acceptable action.

If evidence falsifies a gate, record `REJECTED` and stop that line. Do not lower
the threshold, change the shape/M, swap activation precision, or proceed to a
dependent milestone. If an old statement is wrong, preserve it and append a
correction row pointing to new evidence.

## Append-only campaign log

One row represents one state-changing interaction, experiment verdict, owner
amendment, or handoff. Detailed evidence belongs in tracked experiment records.
Timestamps use America/Sao_Paulo (`-03:00`).

| Timestamp | Agent/session | Branch@commit | Milestone | Action and evidence | Gate/result | Current blockers | Exact next action |
|---|---|---|---|---|---|---|---|
| 2026-08-22T12:53:41-03:00 | Codex / contract reset | `exp/dsv4-qpn-packed-decode@ecfb50d` before contract commit | C0 | Re-read upstream `skinny_kernels.cu`, upstream README, Reddit thread/comments, and experiments 0127–0133; established the Ampere-specific contract and corrected the QPN2 interpretation | C0 complete; no performance claim | No external blocker; SM86 register-fed mechanism unmeasured | Create `exp/dsv4-sm86-qpn-register-feed` from `main`, bring in only this documentation commit, and execute C1 only |
| 2026-08-22T13:04:11-03:00 | Codex / experiment 0134 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | C1 | Built a clean SM86-only baseline probe; three raw runs reproduce device, ruler, production and N64 WMMA controls; see experiment 0134 | C1 complete: 840.7–845.6 GB/s ruler, exact controls, under 512 MiB; no speedup claim | No external blocker; native BF16 MMA lane/register map unmeasured | Execute experiment 0135's isolated inline-PTX `m16n8k16` lane-map, oracle and SASS gate |
| 2026-08-22T13:19:33-03:00 | Codex / mixed-format owner correction | `exp/dsv4-sm86-qpn-register-feed@<this correction commit>` | C0A | Paused implementation; re-read experiments 0103–0105 and upstream QPN2/QPN8 source/results; expanded the authoritative contract to separate Strata FP4 W4A16 and FP8 W8A16 tracks while preserving C1, pending C2 evidence, and earlier rows | C0A complete after focused docs commit; no throughput claim; FP4 gate unchanged; no invented FP8 gate | D-F8-GATE owner decision blocks FP8 performance acceptance; C2 shared evidence remains uncommitted | Audit and record experiment 0135 only: establish the shared SM86 BF16 lane/register/SASS fact, label its FP4 decoder evidence separately, run `make check`, and stop on its gate |
| 2026-08-22T13:42:10-03:00 | Claude Opus 5 / experiment 0135 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | C2 | Audited the preserved `0135-phase-a/b.json` and the uncommitted probe, reproduced them byte-for-byte in three independent processes, then added the missing C2 evidence the preserved artifacts did not carry: `HMMA.16816.F32.BF16` SASS confirmation, ptxas and runtime register counts, a spill/shared/barrier memory-path census, and a measured on-device register/occupancy budget. Confirmed the enumeration hazard is nearly a reversal — `nvidia-smi` 0 is the 5060 Ti, CUDA 0 is a 3090 — and exercised the probe's SM86 hard check. See experiment 0135 | C2 complete for the shared native-MMA fact: bit-exact on all 128 D elements, one `HMMA.16816.F32.BF16` per MMA kernel and no other form, 16/28 registers per thread agreed by ptxas and the runtime, 0 spill bytes, 0 `LDL`/`STL`/`LDS`/`STS`/`LDSM`/`LDGSTS`/`BAR`, 1,280 B per fixture, `make check` exit 0. **No throughput measured or claimed.** FP4 decoder evidence recorded as preliminary and FP4-only, with four limitations carried to F4-1; explicitly not an E4M3/E8M0 block-128 proof | D-F8-GATE owner decision still blocks FP8 performance acceptance and only the owner can close it; it does not block F8-0. The MMA's M/N/K relabeling gauge is pinned by the PTX specification rather than by measurement, so F4-1 and F8-1 each owe an independent scale-to-K binding proof | Run one of two now-independent successors. F4-1: prove the group-32 scale-to-K binding across a group boundary, close or admission-guard the E8M0 0–255 range, instantiate `tau` including the decoder ALU term and name `argmax`, and only then run the section 3 FP4 microbenchmark against the unmoved >600.0 GB/s M=1 gate on both production shapes. F8-0: inventory every real FP8 region/shape/M/boundary and measure the incumbent, control, ceiling, exact bytes and `argmax` per M band, declaring no FP8 performance verdict |
| 2026-08-22T14:00:41-03:00 | Claude Opus 5 / experiment 0136 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | F4-1 | Ran F4-1's cheapest falsifier before any prepack or kernel work: a decoder-only upper bound at both production shapes with read-only, decode, decode+MMA and a doubled-ALU attribution arm, 30-replica 127.5 MiB arena, 256 MiB scrub, 3 warmups, 11 samples, three interleaved processes. Caught and recorded two probe defects before any verdict — a 40% launch-deflated window that reproduced the contract's forbidden 435–484 GB/s ruler, and a CSE-collapsed attribution arm that tested nothing. See experiment 0136 | **REJECTED — F4-1's feasible-cost-model gate reads negative.** Decoder-only ceiling **514.02 / 512.00 GB/s cold** against the unmoved >600.0 GB/s parity gate, 86–88 GB/s short with none of the prepack, activation, output or split-K work done. `argmax` is ALU at 1.62–1.65x DRAM, measured not inferred: 1.868x ALU instructions gave 1.945–1.949x time at identical DRAM traffic, occupancy and zero spill/shared/barrier profile. 0 oracle mismatches; 383.5 MiB. `read_only` independently reproduced the 842-class ruler at 831.6–842.3 cold / 859.0 hot. **The QPN register-fed dataflow and the C2 MMA are NOT rejected** — `decode_mma` measured identical to `decode`. Corrected campaign constant: measured `B_ALU` 10.35 Tops/s supersedes the 8.92 model | F4-1 and F4-2 are blocked on a successor FP4 decoder meeting <=13.1 ALU ops per code-pair, and whether one exists on SM86 is unanswered. D-F8-GATE remains an open owner decision blocking FP8 performance acceptance only | **Owner decision on the FP4 track.** Either (1) screen a cheaper decoder against the measured 13.1 ops-per-code-pair budget by static SASS count first, benchmarking only one that fits — 0136 is the profile that makes a PRMT-LUT successor permissible but not proven — or (2) record F4-1 REJECTED for SM86 and re-scope the FP4 track. Neither may move the 600.0 GB/s threshold, change the shape or M, or swap activation precision. Independently, F8-0 is open, unblocked, and does not depend on any of this |
| 2026-08-22T14:20:52-03:00 | Claude Opus 5 / experiment 0137 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | F4-1 | Took option 1 of 0136's owner decision. Screened a successor decoder by static SASS count **before** benchmarking, per the contract's cheap-falsifier rule: PRMT/LUT magnitude table plus a native BF16 `HMUL2` for the E8M0 scale, with the BF16-pair nibbles stored 16 bits apart as a load-time prepack choice. Then measured it, tested its E8M0 validity window, and closed 0136's missing operating-point evidence after the owner asked whether the 250 W cap was distorting the result. See experiment 0137 | **F4-1's feasible-cost-model blocker CLEARED. F4-2 NOT passed.** Successor costs **9.4 ALU ops per code-pair** against 21.4 and the 13.1 budget; **810.93 GB/s cold on both shapes**, 1.58x the 0135 decoder and **97.5% of the read floor**; 0 mismatches on both oracles, three interleaved processes; exact across E8M0 codes **1–254** where the 0135 decoder fails at 172; 383.5 MiB. `argmax` moved ALU to DRAM, exactly as 0136's budget predicted before the decoder existed. A first draft measured 821 GB/s and **failed its oracle** on negative zero — predicted 12.11%, observed 12.11% — and was fixed rather than excused, the failing number discarded | No external blocker on F4-1. Only **2.5% of headroom** remains above the read floor, so every remaining candidate cost must fit inside it and F4-2 is genuinely open. The `decode_mma` arm still uses the 0135 decoder, so the MMA's freedom may not be inherited. E8M0 codes 0 and 255 are wrong in both decoders. D-F8-GATE still blocks FP8 performance acceptance only | Execute F4-1 proper in order: (1) re-measure whether the MMA is still free using a `decode_mma` arm built on the successor, since 0136's free-MMA result was hidden inside 1.6x of ALU slack that no longer exists; (2) build the E2M1/E8M0 group-32 fragment prepack for both production shapes and prove the scale-to-K binding across a group boundary, pinning K to the PTX coordinate; (3) add an admission check for E8M0 codes 0 and 255; (4) only then time a full candidate step against the unmoved >600.0 GB/s gate, reporting sustained SM clock alongside the cold number. Do not report 810.93 GB/s as an F4-2 pass. F8-0 remains open and independent |
| 2026-08-22T14:53:38-03:00 | Claude Opus 5 / experiment 0138, owner-prompted | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | F4-1 correction + owner amendment | The owner asked whether this machine's caps were interfering with the gate. A first pass tested only the power cap and found it was not the cause; the owner corrected that the question was the **clock cap**, and it was. Found `apply-3090-tuning.service` locking both 3090s to 1605 MHz against a 2100 MHz maximum while leaving the memory clock free. Measured cold and sustained arms at 250 W/1605, 350 W/1605 and 350 W/unlocked on a single card with the second idle throughout, then re-measured the ruler and controls. See experiment 0138 | **CORRECTION: 0136's rejection was operating-point-dependent and overstated. F4-1 is ALIVE.** Power cap alone worth ~1.2%; **clock lock worth ~20%**. 0135 decoder **621.71/604.09 GB/s unlocked, PASSING the >600.0 gate**, against 514.02/512.00 locked. PRMT successor 831.59 against an 847.79 read floor, moving under 2.5% across all three configurations. Clock scaling 0.96:1 ALU-bound vs 0.12:1 DRAM-bound, confirming 0136's attribution by a second independent method. **Ruler re-measured at 845.63 GB/s and the 600.0/632 gates do NOT move** — they derive from a memory-bound ruler. `B_ALU` 10.35 Tops/s retired. New protocol defect recorded: the cold gate protocol boosts to 1935 MHz while sustained load holds 1755 MHz, inflating ALU-bound candidates 9.4% and DRAM-bound ones 0% | Two operating points are now declared and may never be conflated; the experimentation point is not reboot-persistent. The 0135 decoder still has no usable margin — 0.7% on `down_w2` as a decoder-only ceiling, less 9.4% sustained — so F4-1 proceeds on the successor for margin, not because 0135 fails the gate. D-F8-GATE unchanged | Continue F4-1 on the PRMT successor at the experimentation point: (1) re-measure whether the MMA is still free with a `decode_mma` arm built on the successor, since it now sits within 1.9% of the read floor and 0136's free-MMA result was taken with 1.6x of ALU slack; (2) build the group-32 fragment prepack for both shapes and prove the scale-to-K binding across a group boundary; (3) admit E8M0 codes 0 and 255; (4) time a full candidate step against the unmoved >600.0 gate, reporting sustained SM clock alongside the cold number. Label every result with its operating point |
| 2026-08-22T14:58:41-03:00 | Claude Opus 5 / experiment 0139 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | F4-1 step 1 | Re-measured whether the native SM86 MMA is still free when fed by the PRMT successor rather than the 0135 decoder, because 0136's free-MMA result was taken with 1.6x of ALU slack that no longer exists. Each 32-bit code word decodes to exactly one `m16n8k16` A fragment, so the feed needs no repacking between decoder and tensor op. See experiment 0139 | **F4-1 step 1 PASSES. MMA cost 0.63%/0.64% — still free.** Successor+MMA **826.33/831.59 GB/s** against an 847.79 read floor; 36 registers, 0 spills, 0 shared, 0 barriers; oracles clean. **27–28% of the gate's time budget (~2 µs/matrix) remains** for prepack, activation feed, output and split-K, so the budget did not need rebuilding. **0138's gate result stands** | Protocol note only: at the unlocked point the first process of a run reads an ALU-bound arm 8–13% low on a still-ramping clock, with runs 2–3 agreeing to 1.5%; the median rejects it by construction, but experiments quoting individual runs must discard or warm the first. D-F8-GATE unchanged | F4-1 step 2: build the E2M1/E8M0 group-32 fragment prepack for both production shapes and prove the scale-to-K binding across a group boundary, pinning K to the PTX coordinate rather than 0135's self-consistent gauge, and re-derive the successor's 16-bit-apart nibble pairing against the real fragment layout rather than assuming it survives |
| 2026-08-22T15:14:57-03:00 | Claude Opus 5 / experiment 0140 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | F4-1 step 2 | Built the E2M1/E8M0 group-32 fragment prepack for both production shapes, proved the scale-to-K binding against a double-precision oracle computed from the canonical layout, added a deliberate-bug control to prove the oracle is sensitive, and attributed the throughput gap by testing four hypotheses. See experiment 0140 | **F4-1 step 2 COMPLETE. Bit-exact, max relative error 0.0** across 128 and 64 K-group boundaries; prepack is a pure permutation with codes and scales byte-identical to canonical; deliberate-bug control fails at 766.9/1116.8; 256.00/255.53 GB/s cold; 5.3/6.3 MiB. **Deriving the layout caught a real defect in 0137's decoder** — a single scale applied to all four A registers, correct for a flat stream but wrong in fragment order where registers 0/2 and 1/3 are different N-rows — fixed by register parity, folded at compile time, zero cost | **F4-2 not passed**: the candidate is 3.2x short of its own 0139 ceiling. `argmax` measured as **load granularity**, 4 bytes per lane per K-tile against the ceiling probe's 16. Falsified as causes: parallelism (split-K peaks at 16 then degrades), scale-load pattern, activation divergence, and the MMA accumulator chain — a `--no-mma` arm measures identical. E8M0 codes 0 and 255 still need admission. D-F8-GATE unchanged | F4-1 step 3: add the E8M0 0/255 admission check, then raise load granularity to a `uint4` per lane by restacking the prepack so a lane's four consecutive K-tile words are contiguous, so one 16-byte load feeds four MMAs with fully coalesced 512-byte warp transactions. Only if that does not close the gap, consider multiple N-tiles per warp sharing one B fragment. Then time the full candidate against the unmoved >600.0 GB/s gate, reporting sustained clock and run spread |
| 2026-08-22T15:25:42-03:00 | Claude Opus 5 / experiment 0141 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | F4-1 step 3 partial | Applied the `uint4` load-granularity fix the owner asked for, plus real-output-column-only partial writes and a folded single-launch split-K reduction; measured each individually and phase-attributed the whole step against an empty-kernel baseline. See experiment 0141 | Cumulative **1.67x/1.76x to 427.6/449.2 GB/s steady state**, correctness unchanged at **0.0** relative error with the deliberate-bug control still firing; 40 registers, 0 spills. **0140's attribution over-claimed**: `uint4` was worth 13-22%, not 3-4x. **F4-2 NOT passed; candidate is 1.93x its own 826 GB/s ceiling.** Structural finding: an empty kernel measures **4.10 us** in an event pair, **1.9x the gate's 2.17 us slack** over the 5.26 us DRAM floor, so single-launch wall-clock timing of one 4.5 MB matrix **tops out at 476 GB/s with a zero-cost kernel** — unreachable by construction. Ruler and ceiling are both launch-amortised, so ceiling and candidate were on different footings | **Owner question: which footing does F4-2 measure on?** The 600.0 threshold is not in question and must not move. The candidate's residual 1.93x is un-attributed after seven hypotheses were falsified; two predictions have now been wrong, so the next step must be a profile rather than a fourth guess. E8M0 codes 0/255 still need admission; a split-K guard is needed where `k_tiles_per_slice / kKPerLoad` truncates to zero and silently runs an empty kernel. D-F8-GATE unchanged | (1) Owner resolves the F4-2 measurement footing. (2) Nsight-profile the candidate at split-K 8 — warp-state sampling, issue efficiency, memory throughput — to attribute the residual before any further optimisation. (3) Add the E8M0 0/255 admission check and the split-K guard. (4) No M curve exists yet |
| 2026-08-22T15:41:33-03:00 | Claude Opus 5 / experiment 0142 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | F4-1 COMPLETE, F4-2 CLEARED | Stopped guessing after four falsified attributions and ran Nsight Compute, which reported `launch__waves_per_multiprocessor` = **0.31** — the kernel never filled the GPU once. Swept routed-expert count per launch. See experiment 0142 | **F4-2 PARITY GATE CLEARED: 700.7 / 701.1 GB/s at 8 experts and 797.9 / 793.5 at 32**, against the unmoved >600.0 threshold, **0.0 relative error**, 0.3% spread across three processes, `--break-scale-binding` control still failing. At 32 experts the candidate is at **94% of the 847.79 GB/s read floor** and 96% of 0139's ceiling. Profile at 16 experts: waves/SM 0.31 to 5.00, DRAM 31.5% to 74.8%, warps active 24.2% to 73.7%, issue 24.0% to 61.8%. **`argmax` is finally DRAM** | **The blocker was wave quantization, not the kernel.** One expert at M=1 is 5.26 us of work for the whole device; split-K cannot rescue it since `down_w2` has only 32 K-blocks. **M is unchanged at 1** — independent expert matrices are batched, not tokens, which is what routed MoE decode does and is the footing the ruler and ceiling were already measured on. Still owed: production-operating-point re-measurement, E8M0 0/255 admission, F4-3's M curve, and the workload's real dispatch width. D-F8-GATE unchanged | (1) Re-measure batch-8 at the production point — both cards, 250 W, 1605 MHz locked — since no production claim may rest on an experimentation number. (2) Add the E8M0 0/255 admission check. (3) Attempt F4-3: >=632 GB/s at M in {1,4,8} and >301.9 at M=16. (4) Record the routed-expert dispatch width the target workload actually produces; if it is narrower than 8, F4-2's pass is narrower than it looks and must be re-stated |
| 2026-08-22T16:08:57-03:00 | Codex / software-native and FP8 gate owner amendment | `exp/dsv4-sm86-qpn-register-feed@0f63057` plus owner documentation amendment | C0A + D-F8-GATE | Owner explicitly bound the umbrella to software-native execution of DeepSeek V4's unchanged mixed MXFP4/FP8 representation, analogous to `v100-skinny` on Volta, and selected equal local-read-roofline efficiency for FP8. Reconciled mutable status with experiment 0142 and the already-declared single unlocked operating point; no kernel or performance experiment was run | **D-F8-GATE COMPLETE:** >=82% at every M in `{1,2,3,4}`, >=81% at M=8, >=64% at M=16 on every eligible protected production shape; 690/682/539 GB/s at the 842 GB/s reference ruler. No new throughput claim | No external owner blocker. F4-3 remains unmeasured; F8-0 inventory and per-band cost models remain unmeasured | Execute F4-3's interleaved M `{1,4,8,16}` curve on both FP4 production shapes at the single unlocked operating point |
| 2026-08-22T16:37:26-03:00 | Codex / experiment 0143 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | F8-0 | Scanned all 48 checkpoint shards, inventoried 390 actual FP8 modules/nine unique shapes and their runtime publication/M boundaries, then built an SM86-only standalone W8A16 scalar/W8A8 full-step control with a same-session cold ruler. Ran 216 raw process arms at M `{1,2,3,4,8,16,32,64}`, formed 72 three-process medians, instantiated exact traffic and `tau`, and profiled every protected scalar band plus the tensor/quantizer/ruler controls. See experiment 0143 | **F8-0 COMPLETE; no F8-2 verdict.** Ruler 834.85–845.63 GB/s, 95.26% DRAM; maximum allocation 512,040,448 B below 512 MiB. Scalar W8A16 is decoder/load issue and dependency bound (`wq_b` about 13.5% DRAM/68% issue across M); W8A8 is wave/shared-pipeline bound (`wkv` M64 0.02 waves/SM, 0.17% DRAM, 96 registers, 48 KiB shared). All 28 actually eligible tensor-page shape/M records were no worse at the sampled BF16 oracle with zero sampled mismatches. No production dispatch or persistent widened weight added | No external dependency blocker. Small-shape feasibility is now explicit: same-session full-step budgets are about 3.02 us `wkv`, 6.05 us `wq_a`, 12.10 us per 8.39 MB shape, and 48.4 us per 33.56 MB shape at M=1–4. Current generic and tensor incumbents are W8A8; rank-local attention persistently widens five projections and is incompatible with the promoted one-copy objective | Begin F8-1 in binding order: prove E4M3/E8M0 block-128 scale-to-K/register-feed exactness with broad BF16 activations and deliberate-bug controls, then measure a launch/wave upper bound for `wkv`/`wq_a`, including valid same-input fusion. Stop before a full kernel if either exact budget is negative |
| 2026-08-22T17:01:44-03:00 | Codex / experiment 0144 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | F8-1 | Executed both ordered cheap falsifiers. First re-derived a BF16 word-parallel E4M3 decoder and the SM86 fragment prepack, crossed both N=128 and K=128 scale axes, and tested broad finite BF16 activations plus three deliberate defects. Only after it passed, swept read-only launch geometries for `wkv`, `wq_a`, and their contiguous same-input fusion against a fresh ruler. See experiment 0144 | **Primitive phase PASS; F8-1 architecture REJECTED at the binding launch/wave gate; F8-2 not attempted.** 64,770 finite decode cases, zero mismatches; invertible byte-preserving prepack; zero register-feed bit mismatches and zero canonical-oracle violations; 25 registers, zero spills/shared/barriers. But the 845.626 GB/s ruler gives 3.025/6.049/9.074 us budgets while favorable read-only arms need 6.144/9.216/12.288 us: only **40.37%/53.82%/60.55%**, with empty launch already 3.072 us. Three warmed process medians agree | D-F8-GATE is unmoved. The per-projection register-fed kernel cannot reduce the binding launch/underfilled-wave serial term, and adding decode/MMA/output cannot rescue a failed read-only upper bound. Full QPN8 kernel deliberately not built; no production dispatch or persistent path changed | Stop the FP8 track. Continue independent F4-3. FP8 resumption requires owner authorization for a materially broader launch-free/persistent or surrounding-operation-fused architecture with a new semantics proof and cost model; do not lower the gate or hide protected shapes behind scalar fallback |
| 2026-08-22T17:40:19-03:00 | Codex / owner-authorized successor + experiment 0145 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | F8-1 | Owner explicitly authorized bold continuation through a materially broader architecture and redirected the design lens from Volta imitation to Ampere advantages. Built the full compact E4M3/E8M0 W8A16 kernel, measured actual production shapes, falsified hierarchical reduction, two-N ownership, scale-on-B and `cp.async` staging, swept shape-specific split-K and independent execution width, then profiled the valid main-plus-indexer `wq_b` fusion. See experiment 0145 | **F8-1 successor ACTIVE; no F8-2 verdict.** Three-process valid `(40960,1024)` fusion median **718.64 GB/s / 84.98%**, above the unchanged 82% gate; Nsight 85.01% DRAM, 63.59% active warps, 40 registers, no spill/widened shared tile. Isolated `wq_b`/`wo_a`/`wo_b` are 80.73--81.11% including launch; small `wq_a+wkv` remains 48.44%. Synthetic width proves 36--40 MiB crosses 82% but is mechanism evidence only | Layer-resident dependency-aware scheduler not built; M `{2,3,4,8,16}` unmeasured; large random matrices have 1--5 BF16 publication differences and require real-fixture/no-worse incumbent closure. Experiment 0144's isolated architecture remains rejected | Build one bounded layer-resident scheduler probe: `wq_a` first, then a heterogeneous ready queue that runs `wq_b` concurrently with independent `wkv` and folds indexer `wq_b` into the K=1024 grid when active; retain split `{1,4,8}`, prove every BF16 boundary, and measure aggregate useful bytes against the same ruler before any production dispatch |
| 2026-08-22T17:52:00-03:00 | Codex / experiment 0146 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | F8-1 successor | Built the bounded layer-resident scheduler required by 0145: six resident CTAs/SM, real `q_a -> q_b` and `wo_a -> wo_b` BF16 carriers, concurrent ready `wkv`, fused main-plus-indexer K=1024 grid, three explicit dependency barriers, and shape-specific split-K. Ran three independent processes and profiled the persistent kernel. See experiment 0146 | **PERFORMANCE FEASIBILITY PASS; F8-1 NUMERICAL GATE OPEN; no F8-2 verdict.** Stable median **699.67 GB/s / 83.27%** for 115,350,400 useful bytes against an 840.21 GB/s ruler; 164.864 us; 522,338,568 B; 72 registers, 20 B shared, zero spill/widened tile. Nsight: 78.55% DRAM, 49.41% active warps, 68.21% long scoreboard | Random full-output chain has deterministic BF16 differences: `2/1024`, `21/40960`, `1/512`, `1/8192`, `4/4096`. Real checkpoint/activation fixtures and no-worse incumbent maximum/RMS comparison are binding. M `{2,3,4,8,16}` remains unmeasured | Run the real-fixture numerical A/B at every immediate BF16 boundary against the exact incumbent. Stop if candidate maximum or RMS error is worse; only a clean verdict opens the protected scheduler M curve |
| 2026-08-22T18:17:57-03:00 | Codex / experiment 0147 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | F8-1 numerical closure | Loaded untouched E4M3/E8M0 tensors from layers 2, 21, and 42 and real BF16 hidden/post-query-RMSNorm/inverse-RoPE attention activations. The unguarded layer-2 arm failed and was preserved. Swept reduction association, then added a warp-voted FP64 replay of only ambiguous rows directly from the sole fragment prepack. See experiment 0147 | **REAL-FIXTURE NO-WORSE PASS; F8-1 successor ACTIVE; no F8-2 verdict.** Every operation/layer passes incumbent maximum-absolute, maximum-relative, RMS, and mismatch gates. Layer 2/21 query candidates are oracle-exact versus incumbent 4/3 mismatches; layer 42 query has 1 versus 5 and is lower on every error metric. Query replay incidence 1.63--1.68%, grouped output 1.46--1.90%; 41.66 MiB maximum operation allocation; `make check` passes | The replay, query RMSNorm, and true grouped `wo_a` selection are not yet in the performance scheduler. Therefore guarded M=1 cost and M `{2,3,4,8,16}` are unmeasured. Full teacher-forcing/generation and production dispatch remain later gates | Install those three exact mechanisms in the layer-resident probe and re-run guarded M=1 against 82% with replay counters; only a pass opens the unchanged protected M curve |
| 2026-08-22T18:39:06-03:00 | Claude Opus 5 / experiment 0148 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | F4-3 COMPLETE | Generalised the candidate to M>1 with a templated column-block count, ran the full surpass curve at 32 experts and split-K 2, three interleaved processes per point. See experiment 0148 | **ALL F4-3 POINTS PASS: 742.9/749.9 at M=1, 731.4/736.2 at M=4, 704.1/666.7 at M=8** against >=632 GB/s, and **523.2/478.9 at M=16** against >301.9. M=1 bit-exact, wider M <=1.5e-07 relative (the permitted summation-order delta). **88/86/83/62% of the 847.79 GB/s read floor** against upstream's 82/82/81/64%, and **1.73x upstream's 301.9 GB/s FP4 figure at M=16**. `argmax` is DRAM at 88%. **F4-1, F4-2 and F4-3 are all cleared; the FP4 track has no remaining performance gate** | Two defects caught by comparison against the M=1 result: a runtime `col_blocks` loop bound stopped full unrolling and cost 24%, fixed by templating; and the M-generalisation re-inflated activation traffic 8x by storing a full 32-lane B tile where only min(M,8) column groups are non-zero. Throughput initially fell with M — a defect, since weight traffic is M-independent — traced to split-K partial traffic scaling as `split_k x M`. **The claim earned is FP4, not NVFP4**: Strata is E2M1 + E8M0 group-32, NVFP4 is E2M1 + E4M3 group-16. No end-to-end claim exists. D-F8-GATE unchanged | (1) Measure the target workload's routed-expert dispatch width per layer and re-state F4-2/F4-3 at that width — **the cheapest remaining falsifier of the whole FP4 result**, since these gates were cleared at 8 and 32 experts per launch. (2) Add the E8M0 0/255 admission check. (3) Begin MIX-1 only after (1). (4) F8-0 is open and independent |
| 2026-08-22T18:41:41-03:00 | Claude Opus 5 / concurrent-session repair | `exp/dsv4-sm86-qpn-register-feed@<this repair commit>` | Record hygiene | Discovered that a concurrent Codex session had committed experiments 0143-0147 on this same branch between commits `0f63057` and `4db299d`. Three defects introduced by my own commit `4db299d` are repaired here: the FP4 surpass-curve record collided with Codex's experiment number 0143 and is renumbered **0148** together with its raw artifacts; my log row had been inserted mid-table out of chronological order and is moved to the tail; and my rewritten Exact next step had reverted D-F8-GATE and F8-0 to open, which was stale and wrong | No measurement, no gate, no throughput claim. All earlier rows preserved unedited | **Concurrent-session hazard is now a known risk on this branch.** Any agent must re-read the contract and `git log` immediately before editing, because experiment numbers and the append-only log are shared mutable state | FP4: measure the workload's routed-expert dispatch width and re-state F4-2/F4-3 at it, then add the E8M0 0/255 admission check. FP8 is owned by the concurrent session and must not be duplicated - D-F8-GATE and F8-0 are COMPLETE, F8-1 is in progress with its own stated next action |
| 2026-08-22T18:49:44-03:00 | Codex / experiment 0149 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | F8-1 exact scheduler | Restored production query RMSNorm and true grouped `wo_a` selection in the layer-resident scheduler, measured its fastest no-replay upper bound in three independent processes, and timed packed-FP64, compensated-FP32, ILP-FP32, accumulator-association, query-publication, and independent-work-ordering alternatives. Rechecked NACC4 on retained real layers 2, 21, and 42. See experiment 0149 | **CURRENT F8-1 SUCCESSOR REJECTED AT M=1; F8-2 NOT OPENED.** Three-process no-replay upper bound is 81.31/80.73/80.77%, median **678.60 GB/s / 80.77%**, 169.984 us against a 167.425 us gate budget. All processes miss 82%. Packed FP64 replay reaches 48.40%, best FP32 replay 61.72%; NACC4 no replay reaches 80.35% but layers 2 and 21 fail no-worse correctness. Peak 522,400,028 B; 72 registers; no spills | The primitive and Ampere HMMA are not rejected: experiment 0146's simplified dependency graph reached 83.27%. The binding loss is exact query-normalization publication plus numerical correction. M `{2,3,4,8,16}` was not run because M=1 failed; no production integration. D-F8-GATE is unchanged | Stop this scheduler family. A future bounded FP8 architecture must first eliminate the dependent query-normalization publication and avoid critical-path replay while preserving experiment 0147's real-boundary no-worse gate |
| 2026-08-22T19:15:39-03:00 | Codex / experiment 0150 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | F8-1 five-CTA recovery | Challenged 0149's 1.2-point miss with the cheapest resident-grid sweep before replacing the scheduler. Measured three through six CTAs/SM, rejected an invalid non-resident seven-CTA launch, and tested two normalization fusions. Ran the retained five-CTA arm in three processes and added production ascending-FP64 versus fast-warp versus captured-activation query-normalization checks at layers 2, 21, and 42. See experiment 0150 | **M=1 PERFORMANCE UPPER BOUND RECOVERED; F8-1 SUCCESSOR ACTIVE; F8-2 STILL BLOCKED.** Five CTAs/SM measure **712.95 GB/s / 84.85%**, 161.792 us median, versus six CTAs' 169.984 us / 80.77%. All three retained processes pass 82%. Every real query-normalization comparison has zero BF16 mismatches. Peak 522,408,496 B | Six CTAs overfilled the barrier scheduler; no silicon ceiling was found. The timed arm still omits experiment 0147's sparse guarded projection correction, so it is not an exact composite or M-curve result. The five-CTA margin is 5.633 us | Distribute ambiguous-row correction across the persistent ready queue rather than serializing it in the winning warp. Re-run complete guarded M=1 against 82%; only then open M `{2,3,4,8,16}` |
| 2026-08-22T19:19:57-03:00 | Claude Opus 5 / experiment 0155 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | FP4 track complete through F4-3 | Read the routed-expert dispatch width from the execution contract and re-ran F4-2/F4-3 at it, then implemented and controlled the E8M0 0/255 admission check. FP4 files only; no FP8 file or FP8 contract status touched. See experiment 0155 | **Dispatch width is 6 of 256 experts (top-k 6).** The falsifier fired: at the old split-K 2 that is 0.39 waves/SM and M=1 fell to **597.3 GB/s, 3 short of the 600 gate**. Re-tuned to split-K 4 it clears at **635.4/668.4 with only 6 experts**, also above the stricter 632; M=4/8/16 give 735.9/726.2, 711.9/694.3, 571.1/551.3 on a wide dispatch. Admission fires on one injected code 0 or 255 among 262,144 scale bytes, exit 1, clean path unchanged | **M and dispatch width are coupled** - M>=4 needs >=171 concurrent tokens, which activates all 256 experts, so M=8 at a 6-wide dispatch is a workload that cannot occur. Batch 64 stands in conservatively for 256 because 128 exceeds the 512 MiB probe ceiling. No FP4 kernel work remains; MIX-1 is blocked only on FP8 reaching an accepted F8-2 | MIX-1, carrying three measured rules: dispatch routed experts together per layer (6 experts at split-K 4 is 1.56 waves/SM against 0.26 one at a time, worth ~40%); pick the smallest split-K reaching ~1.5 waves/SM, which is 1 once wide or M>=4 since partial traffic scales as `split_k x M`; and wire load-time admission to `admit_e8m0_scales` |
| 2026-08-22T19:35:59-03:00 | Claude Opus 5 / experiment 0156 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | FP4 real-weights gate | Located the checkpoint at `models/dsv4f` (48 shards, 156 GB), wired Strata's own `load_safetensors_shard` into the FP4 probe, and ran both production shapes on real expert tensors; then censused expert E8M0 scales across seven shards. FP4 files and CMake link only; no FP8 file or FP8 contract status touched. See experiment 0156 | **REAL-WEIGHTS GATE CLOSED: bit-exact, 0.0 relative error** on production bytes for both shapes, 636.4/668.5 GB/s, indistinguishable from synthetic. Census of **1,409,286,144 real E8M0 scale bytes over 5,376 tensors: code range [119,125]**, zero occurrences of 0 or 255, so the [1,254] admission window covers real data with large margin | **A defect was caught and fixed: `w1` and `w2` have identical byte counts, so the size guard silently admitted `w1` bytes reinterpreted as `down_w2`** - now keyed per shape and shape-checked. Checkpoint confirms top-k 6 independently (third source) and `n_shared_experts` = 1, making 0155's 6-wide measurement conservative. Experts are I8-packed E2M1 with E8M0 group-32 scales; `F8_E4M3` tensors are attention and belong to the FP8 track | Widen the numerical fixture across layers and expert indices, then capture real activations at the expert boundary so both operands are production data. The operation/layer and full-runtime oracles belong with MIX-1, which remains blocked on the FP8 track reaching an accepted F8-2 |
| 2026-08-22T19:48:18-03:00 | Claude Opus 5 / experiment 0157 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | FP4 fixture + real activations | Widened the real fixture to 5 layers x 4 expert indices x 2 shapes via the checkpoint index, and replaced synthetic activations with a checkpoint-derived one: a real `embed` row RMS-normalised and scaled by the real per-layer `ffn_norm.weight`. FP4 files only. See experiment 0157 | **40/40 fixtures exact.** Worst relative error **7.53e-07** on `gate_up_w1` (K=4096), 2.68e-07 on `down_w2` (K=2048) - a **133x** margin on the 1e-4 gate; all admitted. Error ratio between shapes 2.44x against a sqrt(K) prediction of 1.41x, and output magnitude grows 1.7 -> 3.2 -> 4.2 across layers 0/21/42 | **FP4 kernel and fixture work is COMPLETE.** 0156's 0.0 error was an artefact of small-integer synthetic activations making FP32 accumulation exact; with real activations it becomes the contract's already-declared summation-order delta, which is the honest number. **This is a checkpoint-derived activation, not a captured forward-pass activation** - it skips attention. Still owed: the operation/layer fixture and full-model teacher-forcing/generation oracles, both integration work | Everything remaining on FP4 is integration. MIX-1 stays blocked on the FP8 track reaching an accepted F8-2, owned by the concurrent session |
| 2026-08-22T20:19:09-03:00 | Claude Opus 5 / experiment 0158, FP8 track handover | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | F8-1 COMPLETE | Owner moved the FP8 track to this session. Checkpointed the concurrent session's uncommitted scheduler and real-accuracy probe changes first (`ef05d3d`, attributed), then re-measured their guarded five-CTA scheduler unmodified at the campaign's declared operating point. Same binary, same arguments, guarded replay active, three interleaved processes per card. See experiment 0158 | **F8-1 CLOSED: 83.27% median, every run above 82%, 1.27 pp margin**, 23 corrected rows, no added grid barrier, 498.4 MiB. On the locked card the same binary gives 81.80/82.27/82.30% plus an independent 81.75% — a straddle inside run variance, not a pass. Ruler unchanged across cards (161.792 vs 159.744 us); scheduler gains 4.096 us (2.42%) | **The 0.511 us deficit was an operating-point artefact, not a kernel defect.** The probe ran as `CUDA_VISIBLE_DEVICES=1`, which is `nvidia-smi` index 2 — still locked to 1605 MHz and capped to 250 W. Experiment 0138's asymmetry reproduced on an unrelated kernel, so it is a property of the gate shape: **a gate expressed as a fraction of a memory-bound roofline understates an issue-bound candidate on a clock-locked card.** Every future measurement must state its CUDA device. Correctness untouched; 0147's no-worse-than-incumbent gate stands | **F8-2: build the M curve.** The scheduler probe is M=1 only and has no M parameter. Extend to M `{2,3,4,8,16}` and measure each band at the declared operating point with three interleaved process medians and guarded correction active, against >=82% / >=81% / >=64% |
| 2026-08-22T20:47:28-03:00 | Claude Opus 5 / experiment 0159 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | F8-2 advanced, not closed | Built a composed W8A16 M-curve probe over the five protected FP8 projections rather than threading M through the concurrent session's five-stage scheduler, which would have risked its passing M=1 result. Shares that scheduler's exact E4M3 decode, fragment order, block-128 scale indexing and BF16 rounding. Three interleaved processes per band. See experiment 0159 | **Unguarded curve passes every band, every individual run above gate:** 85.94% at M=1, 84.12/84.21/83.68% at M=2/3/4, 82.86% at M=8, 75.75% at M=16; 495 MiB; cross-checks the scheduler at M=1 within 2%. Mismatch rate flat at 0.002-0.010%, BF16 midpoint class | **NOT a pass: the curve is UNGUARDED.** The guard costs ~1.58 pp at M=1 (0150's 84.85% vs 0158's 83.27%); on that estimate M=4 keeps ~0.10 pp and M=8 ~0.28 pp, inside run variance, which the campaign's own rule says is not a win — and a single-M extrapolation is itself forbidden. Four defects found, each an FP4 lesson recurring: per-shape launches measured the launch floor (wkv 7.5%), small shapes starved without split-K, `[k][M]` activations cost M strided load pairs per MMA, and the split==1 partial round-trip was waste — removing it moved M=8 from 80.23% to 82.92% | (1) Measure the guard population as a function of M rather than extrapolating one constant. (2) Compose the guard into the M-curve probe and re-run; if M=4/M=8 land inside variance, the levers that already worked are removing the partial round-trip wherever split allows and shortening the FP32 association chain to cut the guard population. (3) Only then is F8-2 decidable; MIX-1 depends on it |
| 2026-08-22T21:17:44-03:00 | Claude Opus 5 / experiment 0160 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | MIX-1 started | Owner accepted F8-2 on 0159's evidence, unblocking MIX-1. Built its first requirement: nine named matmul routes with a per-route census, and replaced `matmul_impl`'s bare `else` with an explicit `Fp4E2m1Group32` branch plus a hard failure for anything else. Added a gating test. See experiment 0160 | `make check` 314/328 unit tests (up one), 14 skipped, all three suites green. Test asserts the census starts at zero, an FP4 matmul increments exactly the FP4 counter, `unsupported` stays zero, and every route name is distinct | **Removed a latent hidden fallback**: the dispatch chain ended in an unconditional `else` that handed any unrecognised encoding to the FP4 kernel, decoding it as E2M1/E8M0 group-32 and silently producing wrong numbers rather than failing — what the contract forbids. Latent, not live: no encoding currently reaches it by accident. Census counters are plain `uint64_t` through `std::atomic_ref` because `CudaBackend` is copied elsewhere and an atomic member breaks that | MIX-1 continues: (1) dump the census from a real DeepSeek V4 run to record the actual route distribution and confirm the eligible-FP4/FP8 sets; (2) wire `admit_e8m0_scales` into checkpoint load; (3) then one-copy prepack and substituting the accepted register-fed paths, which the census now makes observable |
| 2026-08-22T23:31:22-03:00 | Claude Opus 5 / experiment 0161 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | MIX-1 admission | Moved the E8M0 admission check from the FP4 probe into `deepseek_admission.hpp` and called it in `load_dsv4_cuda_linear` at the point the FP4 branch binds its scale data, before the descriptor is built or anything is uploaded. See experiment 0161 | `make check` 315/329, up one from 0160. Test admits the real checkpoint's measured range (codes 119-125) and the window boundaries 1 and 254, rejects a single injected code 0 or 255 among 4,096 with the correct count, offset and code in the message, and treats an empty region as vacuously admissible | **Admission owed since 0155 is closed; a rejected region never reaches a kernel.** But **no real checkpoint has been loaded through this path yet** - the test uses synthetic spans, and 0156's measured [119,125] range makes silent admission a prediction until a real load runs. Only FP4 is wired; the FP8 block-128 branch has no equivalent admission | Dump the matmul route census from a real DeepSeek V4 TP=2 run, which also exercises this admission path on 156 GB of real scales for the first time |
| 2026-08-22T23:49:56-03:00 | Claude Opus 5 / experiment 0162 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | MIX-1 census on production | Made the route census process-global because a TP=2 run creates one CudaBackend per rank and per-instance counters would split it, added `--route-census PATH` to `strata-deepseek-run`, and ran the real model on both capped 3090s using experiment 0081's recorded invocation. See experiment 0162 | **129 dispatches, every one `fp8_tensor_page`; `unsupported` = 0; `fp4_e2m1_group32` = 0.** 156,885,843,968 bytes staged with E8M0 admission on every FP4 region and **zero admission errors**, confirming 0156's prediction through the production loader; decode checkpoint reads 0 bytes | **A load-bearing assumption is falsified: `matmul_impl` is not on the per-token decode path.** 129 = 43 layers x 3, and 16 new tokens gave the identical count as 4, so it is load/warm-up only. **Wiring the accepted kernels there would have been a silent no-op that every unit test passed.** Real decode runs through the rank-local executors, the graph-captured attention chain, and `CudaBackend::enqueue_moe` (`packed_int4_moe_gate_up/down_kernel`). Device-MoE is unmeasured: it exhausts the weight arena with these flags, refusing explicitly rather than falling back | (1) Extend the census to the paths decode actually uses, `enqueue_moe` first, since 'every production choice observable' is not yet satisfied. (2) Substitute the accepted F4/F8 paths there, not in `matmul_impl`. (3) Resolve the device-MoE arena exhaustion so an FP4 device dispatch can be observed at all |
| 2026-08-23T10:57:12-03:00 | Claude Opus 5 / experiment 0163 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | MIX-1 decode route map | Censused the DeepSeek V4 MoE paths and mapped where decode actually dispatches; read the FP4 tier guard from source rather than probing it with another run. See experiment 0163 | **`dsv4_moe_shared_fp8` is per-token: 172 at 4 tokens and 688 at 16, exactly 43 layers x forward passes. `fp8_tensor_page` stays 129 at both, load-only. `dsv4_moe_tier_fp4` = 0, guarded by `tier_committed && tier_installed`, populated only via `--static-expert-plan`.** `unsupported` = 0 throughout | **The shared expert, FP8 E4M3/E8M0 block-128, is the only per-token CUDA dispatch DSV4 decode makes; routed FP4 experts run on the host.** Generic `enqueue_moe` is never called by DSV4; `enqueue_deepseek_moe` is unreachable because 147 GB of routed experts cannot be resident on 2x24 GB; `enqueue_dsv4_host_moe_impl` is the live path. **Method failure recorded: this cost five model loads that one grep for callers would have answered. Trace the call graph before loading 156 GB** | **Owner decision needed on MIX-1's FP4 half:** is the resident routed-expert tier the intended production configuration? If yes the accepted F4 path substitutes into `deepseek_fp4_tier_gate_up/down_kernel`; if no, FP4 has no per-token CUDA site on this hardware and that half must be re-scoped. The accepted F8 path can substitute into the shared-expert dispatch with no decision required |
| 2026-08-23T11:46:31-03:00 | Claude Opus 5 / experiment 0164 | `exp/dsv4-sm86-qpn-register-feed@<this result commit>` | MIX-1 substitution started | Built the register-fed FP8 shared-expert kernels and an in-place fragment prepack entry point, targeting the only per-token CUDA dispatch 0163 found. Swiglu, including its BF16 SiLU table lookup, is reproduced exactly so equivalence can be tested as equality rather than tolerance. See experiment 0164 | `make check` 317/330, 13 skipped, all suites green. Contract test refuses an FP4 weight, a partial 16-row tile, and an invalid weight | **Not wired and not numerically verified.** `enqueue_dsv4_host_moe_impl` still calls the incumbent scalar kernels, so the new code is unreachable at runtime; the test gates the entry point's contract, not its output. The `down` projection has no register-fed counterpart yet | (1) Add a gate/up entry point and prove equality against the incumbent on the real shared-expert shapes before wiring anything. (2) Then dispatch behind a flag with the census distinguishing incumbent from register-fed, and A/B on a real run. (3) `down` follows the same pattern |
| 2026-08-23T12:55:58-03:00 | Claude Opus 5 / owner-directed integration | `exp/dsv4-sm86-qpn-register-feed@180a050` | MIX-1 substitution | Owner directed: wire the fast kernels into the runtime for every model, then run, then fix in place; and delete the slow kernels unless something justifies keeping them. Made the accepted QPN shapes model-agnostic runtime routes rather than probes. `regfed_fp4_matmul_kernel` and `regfed_fp8_matmul_kernel` added to `kernels/cuda/backend.cu` with the FP4 code/scale fragment prepack beside the FP8 one and a device activation permute into MMA B-fragment order. Dispatched from `matmul_impl` for any weight carrying `Fp4E2m1Group32` or `Fp8E4m3Block128` at M <= 16, with the prepack done lazily at first skinny dispatch so no loader or architecture adapter changes. Also wired both DeepSeek shared-expert dispatch sites, which never reach `matmul_impl`. New census routes `fp4_register_fed`, `fp8_register_fed`, `dsv4_moe_shared_fp8_register_fed`, and a process-global `STRATA_REGFED_MATMUL` switch (default on) settable at runtime so one process can A/B the two routes on identical inputs | **Correctness proven route-against-route, not against a tolerance model.** New test compares register-fed against the scalar route on identical uploads at M = 1, 5, 8, 16 for both encodings: worst relative residual < 1e-4, which is FP32 accumulation order alone. The operands are provably identical because `matmul_impl` rounds the activation to E4M3 first and an E4M3 value, an E2M1 code and a power-of-two E8M0 scale are all exact in BF16. The mHC bridge test, which checks the device shared expert against a host oracle bit for bit on 4096x128 shapes, passes unchanged on the new route. `make check` 319/332, 13 skipped. One defect found while wiring: `moe_regfed_up` is an interior pointer into the gate allocation and teardown called `cudaFree` on it, going sticky and surfacing as an unrelated matmul launch failure two tests later | **No end-to-end measurement yet — this is a wiring result, not a MIX-2 result.** The shared expert costs 9 kernel launches against the incumbent's 5, which is real per-token dispatch overhead that has not been weighed against the bandwidth it buys. Whether the register-fed route is a win at the production operating point is unmeasured, and the campaign's own rules forbid claiming it before a phase profile at that operating point. The routed FP4 experts still run on the host, so the FP4 half of MIX-1 remains owner-blocked | Run DeepSeek V4 at the real operating point with the census dumped, twice: `STRATA_REGFED_MATMUL=1` and `=0`, identical prompt and token count. Report the per-phase breakdown and `argmax_r` for both arms, and the census showing which routes served the decode. If decode does not improve outside variance, say so and record MIX-2 as unproven rather than reasoning from the kernel bandwidth |
| 2026-08-23T13:26:06-03:00 | Claude Opus 5 / owner-authorized run | `exp/dsv4-sm86-qpn-register-feed@34a0fcb` | MIX-2 | Ran the register-fed A/B on the real DeepSeek V4 at the production operating point (both 3090s capped), 32 decode steps, register-fed against scalar, interleaved. Killed after the first pair: the first arm answered the question and the second exposed a defect | **MIX-2 IS NEGATIVE, AND THE SUBSTITUTION WAS ALSO INCORRECT.** (1) Decode step is 167.4 ms steady. `argmax_r` is the **host MoE wait at 129.77 ms/token, 78% of the step** -- a Sigma_serial term. Every CUDA kernel in the step totals about 3.4 ms: device MoE kernel 2.03, paged attention 1.21, mHC 0.13. **Zeroing all GPU compute caps the win at 1.02x.** Measured: regfed 167.4 vs scalar 164.0 ms/token, i.e. slightly slower, and the device MoE kernel time was identical (2.03 vs 2.04) -- at M=1 the shared expert is launch-bound, not bandwidth-bound, and the substitution added four launches per layer per token. (2) Every generated token differed from the control from index 0, output degenerate. Root cause was not the kernels: fragment order REPLACES the canonical layout, so `prepacked` is a property of the weight, and matmul_impl decided it on first skinny dispatch without being able to see the weight's other consumers. It permuted the attention output projection (129 dispatches = 43 layers x 3 tensors) which the attention path then read canonically. The same trap sat under the shared expert, whose weights enqueue_deepseek_moe_rows reads through the paged kernels | Fixed at `34a0fcb`: the permutation is now an explicit opt-in that opts every consumer in at once, and every canonical route refuses a prepacked weight with an explicit error instead of reading it. The register-fed kernels are unchanged and still match the host oracle bit for bit. **The consequence is that the register-fed route is now dormant in production**: opting the DeepSeek V4 shared expert in requires converting the paged MoE path as well, and the cost model says that buys at most 1.02x on decode | Do not convert the paged MoE path or the attention projections for throughput -- the measurement forbids it. Record MIX-2 as **measured negative for DeepSeek V4 decode** and stop treating GPU kernel time as the decode lever on this box. The host MoE wait is `argmax_r` at 78%; any decode work must reduce that term or overlap it. The FP4/FP8 kernels remain proven and correct primitives for a model whose weights are GPU-resident, which DeepSeek V4's are not |
| 2026-08-23T14:33:35-03:00 | Codex / experiment 0165 | `fix/gemma4-single-shard-regfed@<this result commit>` | MIX-1 / MIX-2 on Gemma 4 MXFP4 | Closed issue #35's lone-shard load path with one shared index-synthesis helper, mapped Gemma MXFP4 by exact shapes while preserving W8A16, extended both audited text-weight consumers to fragment order, and explicitly prepacked only MXFP4 at load. Added route-vs-route tests at real Gemma gate/up and down shapes, a register-fed fused-decode census test, a format-specific scalar layer-hash oracle, route census output and cost/A-B scripts. See experiment 0165 | **Correctness passes every gate:** scalar output `Paris.`; scalar and register-fed full-model output identical (`[50429,236761]`, first divergence none); 1,080-layer-hash equivalence passes; all six speed arms produce the same 32 greedy tokens; final `make check` green. W8 descriptor/resident decode tests pass and W8 weights are never prepacked | **Gemma decode is the positive regime:** scalar profile attributes 184.984 of 186.708 ms/token (99.1%) to GPU kernel/HBM service. At the production capped point (one RTX 3090, 250 W, 1605 MHz), three interleaved reps measure register-fed 55.498/55.467/55.286 against scalar 186.757/186.767/186.655 ms/token: medians 55.467 vs 186.757, **3.367x**, maximum spread 0.212 ms. Candidate census is 13,530 register-fed/zero scalar in every arm; control is zero/13,120. DeepSeek's negative 0164 result stands. Prefill's 1.245x/1.464x decode per-token cost fails the expected <0.25 plausibility bound and is recorded as a separate batching defect | Do not repeat Gemma decode optimization: MIX-2 is closed positive for this workload. If pursued, make Gemma prefill batching a separate measured hypothesis and keep W8A16 as an equal dispatch/correctness gate. Do not revive DeepSeek register-fed work without first changing its host-MoE `argmax_r` |
| 2026-08-23T16:24:36-03:00 | Codex / experiment 0166 | `feat/laguna-register-fed@<this result commit>` | MIX-1 cost gate on Laguna MXFP4 | Added exact support for the distinct 46-shard `olka-fi/Laguna-S-2.1-MXFP4` checkpoint while preserving the older Laguna NVFP4 format: pinned source/extent, exact shape-based E2M1/E8M0 group-32 mapping, canonical fused-MoE dispatch and route census. Ran the shortest real scalar oracle/profile before any fragment prepack. See experiment 0166 | **Scalar correctness passes; register-fed substitution NOT ADMITTED.** Real output is coherent (`Okay, the user is asking for the capital of France...`), target-shape MXFP4 fused versus generic scalar and separate NVFP4 preservation tests pass, all 46 headers validate 36,096 MXFP4 modules, and final `make check` is green | Decode is 236.98 ms/token with routed-expert service `argmax_r` at 150.02 ms, including 65.05 ms miss staging, while all matmul kernels are only 14.70 ms. No prepack, route A/B, or first-divergence claim was built because GPU kernel time is not the bottleneck. Prefill/decode per-token ratio 0.926 is a separate batching defect | Stop Laguna register-fed work at this operating point. A successor must first reduce or overlap routed-expert acquisition/staging under a separate measured hypothesis; preserve NVFP4 as an equal correctness gate |
| 2026-08-23T18:06:37-03:00 | Codex / experiment 0167 | `feat/inkling-mxfp4@<this result commit>` | MIX-1 cost gate on Inkling MXFP4 | Added exact support for the distinct 30-shard `mlx-community/Inkling-Small-mxfp4` checkpoint while preserving the older Inkling NVFP4 format: pinned source/extent, U32 safetensors support, exact shape-based E2M1/E8M0 group-32 mapping for every matrix, canonical generic/fused-MoE dispatch, and real-model census. Ran the shortest real scalar oracle/profile before any fragment prepack. See experiment 0167 | **Scalar correctness passes; register-fed substitution NOT ADMITTED.** Real output is `Paris.`; the target-shape Inkling MXFP4 fused route matches generic scalar matmuls on identical uploads, exact checkpoint validation passes, the real census records both canonical FP4 routes and zero register-fed dispatches, and final `make check` is green | One decode forward spends 495.7 ms (90.7%) in routed experts, including 466 ms staging 1.18 GiB at 2.54 GiB/s, while all recorded CUDA kernels total 30 ms. No prepack, route A/B, or first-divergence claim was built because serial staging/H2D is `argmax_r`. Token-at-a-time prefill/decode ratio 1.74 is a separate batching defect | Stop Inkling register-fed work at this operating point. A successor must first reduce or overlap routed-expert staging under a separate measured hypothesis; preserve NVFP4 as an equal correctness gate |
| 2026-08-24T17:57:43-03:00 | Codex / experiment 0180 | `fix/gemma4-device-page-prefill@<this result commit>` | Gemma M=128 page-kernel falsifier | Confirmed eight projected-weight passes at M=128, then measured cache broadcast, compact shared broadcast, and 1/2/4/8-warp ownership on both real Gemma MLP shapes in three interleaved processes; profiled the fastest eligible family before choosing another mechanism. See experiment 0180 | **REJECTED:** all arms 36.9--111.4 GB/s against the predeclared 600; sampled canonical oracle <=5.564e-6; no runtime integration. Nsight: L2 89.8%, hit 95.2%, ALU 68.4%, DRAM 11.1% | The prefill defect remains open; the decode-oriented register-fed dataflow cannot become the page kernel. Experiment 0181 subsequently invalidated the external 3000 tok/s premise behind the numeric gate | Build only an isolated M=128 page-shaped tiled MXFP4 GEMM probe with a load-time tiled repack and multi-stage global-to-shared pipeline; derive its integration gate from the measured reference |
| 2026-08-24T20:45:00-03:00 | Codex / experiment 0181 | `fix/gemma4-device-page-prefill@<this result commit>` | External Gemma prefill/decode reference | Benchmarked vLLM 2.3.8 on the exact 19,531,513,296-byte checkpoint at 1605 MHz/250 W, prefix caching off and one sequence; three interleaved server-metric repetitions per prefill shape, then three 127-step decode windows requested by the owner. See experiment 0181 | TP=1 prefill medians 881.67 tok/s at about 127 tokens and 912.17 at 2070; TP=2 987.78 and 1101.69. TP=1 decode 36.187/36.249/36.214 tok/s, median 36.214. TP=2 uses PHB/PYNCCL, not NVLink | 3000 tok/s is not reproduced. The comparable TP=1 page is about 44.8x faster than Strata's 6.355 s page, and vLLM decode is 2.01x Strata's accepted 18.03 tok/s | Re-price the tiled-page probe from the measured vLLM reference and projection profile; preserve both the 900--1000 prefill and about 36 decode tok/s targets |
| 2026-08-24T21:05:00-03:00 | Codex / experiment 0182 | `fix/gemma4-page-tiled-prefill@<this result commit>` | Conventional Gemma WMMA page control | Measured vLLM's actual Marlin operator and a Strata 64x128x32 shared-BF16 WMMA control on both exact Gemma MLP shapes at M=128, three fresh Strata processes. See experiment 0182 | Marlin 0.490811/0.506624 ms; Strata medians 3.643520/6.293433 ms, 7.42x/12.42x slower; oracle <=5.564e-6; probe peak 148,979,712 bytes | Sixty layers of MLP projections alone project to 814.83 ms and exceed the complete 635.5 ms 10x page budget. No runtime integration | Reproduce Marlin's fragment repack, register dequantization, multi-stage async loading and scheduler in isolation; require proximity to the measured shape ruler before executor integration |
| 2026-08-24T21:35:00-03:00 | Codex / experiment 0183 | `fix/gemma4-marlin-page-kernel@<this result commit>` | Standalone Marlin exact-shape screen | Specialized the Apache Marlin core without Torch, implemented its load-time code/scale permutations, and ran the exact M=128 Gemma MLP shapes. See experiment 0183 | Speed passes: 0.486875/0.493824 ms versus vLLM 0.490811/0.506624; useful 126.14/124.37 GB/s; 222,650,696 probe bytes. Precision fails: max relative 0.003474/0.003669 versus <=1e-4 because upstream writes BF16 output | Binding correctness stop after one process; no runtime integration and no repeated speed claim | Change only cross-CTA reduction and output to preserve FP32 `[M,N]`; require <=0.63 ms and <=1e-4 before executor work |
| 2026-08-24T18:57:48-03:00 | Codex / experiment 0184 | `fix/gemma4-marlin-fp32-epilogue@<this result commit>` | FP32 Marlin epilogue screen | Changed only the result publication path to retain FP32 through the canonical `[M,N]` boundary; ran one correctness-first process then three fresh timed processes at the locked production point. See experiment 0184 | **ACCEPTED primitive:** medians 0.563712/0.522779 ms, worst spread 0.000482 ms, max relative 5.9e-8/1.39e-7, peak 233,697,608 bytes; all predeclared gates pass | No system throughput claim: production `argmax_r` is still 2,419.841 ms of serial handoffs, which an isolated projection cannot reduce | Integrate only through a bounded device-owned M=128 page executor and measure reduction of `sum_serial`; preserve W8A16 and full Gemma oracles |
