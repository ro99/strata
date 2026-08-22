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
| F4-3 | Clear FP4 surpass curve | >=632 GB/s at M `{1,4,8}` both shapes and >301.9 GB/s at M=16 | **PENDING on F4-2** | Per-M cost models required |
| F8-0 | Inventory and baseline Strata FP8 W8A16 operating points | Enumerate real regions/shapes/M/boundaries; measure scalar/native, existing W8A8-style WMMA control where eligible, ruler, exact bytes, and `argmax` independently | **COMPLETE (experiment 0143)** | 390 actual modules, nine unique shapes, real M/tile bands, exact bytes, 216 raw process arms/72 process medians, 834.85–845.63 GB/s ruler, per-band cost model and Nsight attribution. No QPN8 or F8-2 verdict |
| D-F8-GATE | Bind FP8 performance threshold and required M coverage | Owner selects absolute-throughput, local-efficiency, or another evidence-backed gate and names required operating points | **COMPLETE — OWNER BOUND 2026-08-22** | Equal local-read-roofline efficiency: >=82% at every M in `{1,2,3,4}`, >=81% at M=8, >=64% at M=16, on every eligible protected production shape; 690/682/539 GB/s at the 842 GB/s reference ruler |
| F8-1 | Prove exact QPN8-derived FP8 primitive | E4M3/E8M0 block-128 fragment prepack and direct W8A16 register feed; independent decoder/matrix/boundary gates; no widened persistent path | **REJECTED at binding launch/wave gate (experiment 0144); primitive correctness phase PASSED** | 64,770 finite decoder cases, invertible byte-preserving prepack, exact register feed and boundary controls pass. The favorable read-only ceiling is only 40.37% `wkv`, 53.82% `wq_a`, and 60.55% fused versus the required 82%; stop before the full kernel |
| F8-2 | Clear owner-bound FP8 performance curve | Satisfy D-F8-GATE on every required shape/M with three interleaved process medians and independent correctness | **BLOCKED by rejected F8-1 architecture** | Equal-local-roofline gate remains binding and unmoved. Resumption requires a newly authorized launch-free architecture hypothesis, not a lower gate or scalar fallback |
| MIX-1 | Integrate one-copy mixed production dispatch | Eligible FP4 uses accepted F4 path; eligible FP8 uses accepted F8 path; unsupported shapes use explicit approved exact routes; route census, admission, prepack, VRAM, graph and fixtures pass | **PENDING on F4-3 and F8-2** | No hidden fallback, no duplicate/widened weights |
| MIX-2 | Confirm end-to-end value | Real workload shows material outside-variance improvement with identical model, formats, activations, routes, and budgets; phase/resource traffic reported | **PENDING on MIX-1** | Kernel bandwidth alone is not an end-to-end claim |

## Current position

Last updated: 2026-08-22

- **Current milestones:** C2 and F4-1 are complete. F4-2 is cleared at eight or
  more routed experts per launch; F4-3 is next on the FP4 track. F8-0 is
  complete. F8-1's exact primitive phase passes, but its binding read-only
  launch/wave ceiling is negative in experiment 0144, so the per-projection
  QPN8-derived architecture is rejected and F8-2 is blocked.
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
- **F4-1 is COMPLETE and F4-2 is CLEARED (0142): 700.7/701.1 GB/s at 8 routed
  experts per launch, 797.9/793.5 at 32, against the unmoved >600.0 gate, with
  0.0 relative error.** At 32 experts the candidate runs at **94% of the 847.79
  GB/s read floor**, so decode, scales, MMA, split-K and dispatch together cost
  about 6%. **`argmax` is finally DRAM.**
- **The blocker was never the kernel — it was wave quantization.** One 4.46 MB
  expert at M=1 is 5.26 us of work for the whole device and yields **0.31 waves
  per SM**, measured. Split-K cannot fix it: `down_w2` has only 32 K-blocks
  total. Batching independent expert matrices — which is what routed MoE decode
  does, and the same footing the 842 ruler and 0139's 826 ceiling were already
  measured on — takes waves/SM to 5.00, DRAM 31.5% to 74.8%, warps active 24.2%
  to 73.7%. **M is unchanged at 1**; this is not a favorable-shape pass.
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
  runs an empty kernel; no F4-3 M curve exists; and the real routed-expert
  dispatch width is not yet recorded.
- **Preserved validated work:** C1/0134, re-measured at the new point as
  roofline 845.63, production 90.67, N64 WMMA 174.08/181.33. C2/0135. The 0136
  arms remain the control the successor is measured against.
- **Device enumeration, confirmed not assumed:** `nvidia-smi` reports the RTX
  5060 Ti at index 0 and the RTX 3090s at 1 and 2; the CUDA runtime reports the
  RTX 3090s at 0 and 1 and the 5060 Ti at 2 — nearly reversed. CUDA index 0 is
  `nvidia-smi` index 1, `GPU-3032cfa3`, pci 82:00.0, and is the unlocked
  experimentation card. Every probe must hard-check capability 8.6 and record
  `device_name` inline.
- **Current blockers:** F4-3 depends on its own M-curve measurement. F8-2 is
  blocked by F8-1's rejected per-projection launch architecture. Resuming FP8
  requires owner authorization for a materially broader launch-free executor
  or operation fusion and a new cost model; lowering D-F8-GATE or hiding either
  protected shape behind the scalar route is forbidden.
- **Branch disposition:** preserve `exp/dsv4-qpn-packed-decode` and experiments
  0127–0133 as controls/falsifications. Continue only on the clean main-based
  `exp/dsv4-sm86-qpn-register-feed`; do not merge failed archived runtime code.
- **Unrelated worktree state:** preserve untracked
  `scripts/dsv4_decode15_bench.sh`, `scripts/dsv4_decode15_server.sh`, and
  `scripts/dsv4_server_prefill_bench.sh`.

## Exact next step

**F4-1 is COMPLETE and F4-2 is CLEARED at >= 8 routed experts per launch**
(experiment 0142: 700.7/701.1 GB/s, 0.0 error).

**Exact next action on the FP8 track:** stop. Experiment 0144 passed the exact
register-feed primitive but falsified the ordered launch/wave gate even for a
read-only same-input-fused upper bound. Do not build the full QPN8 kernel.
Resumption requires an owner-authorized launch-free architecture hypothesis;
the 82%/81%/64% D-F8-GATE remains binding.

Independently, F4-3 remains open: run the interleaved M `{1,4,8,16}` curve on
both FP4 production shapes against its unchanged gates.

Before MIX-1, the campaign must also add the E8M0 0/255 admission check and
split-K guard, measure the real routed-expert dispatch width, and complete
F8-1 and F8-2. **D-F8-GATE is complete:** F8-2 is bound to 82%/81%/64%
of the same-session local read ceiling at M=1-4/M=8/M=16 respectively.

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
