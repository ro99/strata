# RTX 3090 mixed FP4/FP8 skinny-kernel campaign contract

Status: **ACTIVE — authoritative campaign contract**

Established: 2026-08-22

Owner amendment: 2026-08-22 — expanded from an FP4-only campaign to the
checkpoint's mixed FP4/FP8 execution stack

Target: NVIDIA RTX 3090, Ampere GA102, SM86

Umbrella objective:

> Build Ampere-native, register-fed skinny kernels for Strata's mixed low-bit
> regions: a QPN2-derived W4A16 path for FP4 regions and a QPN8-derived W8A16
> path for FP8 regions, preserving each region's exact checkpoint format,
> numerical semantics, one-copy residency, and measured dispatch contract.

Binding FP4 headline goal: **greater than 600 GB/s cold effective
packed-weight bandwidth**

Binding FP8 headline goal: **BLOCKED on owner decision D-F8-GATE; do not copy
the FP4 threshold**

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

- **FP4 regions:** E2M1 codes with E8M0 group-32 scales through an
  Ampere-adapted, QPN2-derived, register-fed **W4A16** architecture.
- **FP8 regions:** E4M3 weight codes with E8M0 block-128 scales through an
  Ampere-adapted, QPN8-derived, register-fed **W8A16** architecture.

This is an RTX 3090/SM86 campaign analogous to the mixed-format execution
stack in `dnv2003/v100-skinny`. It is not a literal Volta source, format, lane
map, or opcode port. “QPN2-derived” and “QPN8-derived” name the register-fed,
fragment-prepacked dataflow. They do not claim that Strata's formats or
numerical boundaries are identical to upstream.

The final production result is mixed dispatch:

1. eligible FP4 regions use the accepted QPN2-derived W4A16 path;
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

The exact FP8 throughput threshold is not yet owner-bound. Upstream QPN8 on a
V100 read ceiling of 879 GB/s reports:

| M | Upstream QPN8 | V100 roofline efficiency | Same efficiency against the measured 842 GB/s SM86 ruler |
|---:|---:|---:|---:|
| 1–4 | 718.6–720.5 GB/s | 82% | approximately 690 GB/s |
| 8 | 712.4 GB/s | 81% | approximately 682 GB/s |
| 16 | 558.5 GB/s | 64% | approximately 539 GB/s |

Matching absolute upstream throughput and matching its fraction of the local
read ceiling are different gates. The owner has not selected between them or
defined required Strata M coverage. Therefore **D-F8-GATE is a blocking owner
decision for FP8 performance acceptance**. No agent may invent a number, apply
the FP4 >600/632 thresholds, or call an FP8 performance win before that row is
resolved. Measurement, correctness, and resource-model work may proceed, but
performance promotion may not.

Independently of the unresolved numeric threshold, reject any FP8 candidate
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
- The FP4 >600/632 thresholds do not apply to FP8. D-F8-GATE must be resolved
  explicitly before an FP8 performance pass can be declared.
- A native BF16 MMA lane map can be shared evidence; an FP4 decoder or group-32
  scale proof cannot be silently treated as an FP8/block-128 proof.
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
| F4-1 | Prove exact QPN2-derived FP4 primitive | E2M1/E8M0 group-32 fragment prepack and direct W4A16 register feed; exact decoder/matrix gates; no widened persistent path; feasible cost model | **UNBLOCKED — cost-model blocker cleared (experiment 0137); prepack and full candidate still owed** | The PRMT/LUT successor decoder costs 9.4 ALU ops per code-pair against the 13.1 budget and ceilings at 810.93 GB/s, 97.5% of the read floor, oracle clean on both shapes. Still owed before F4-2: re-measure whether the MMA is still free with only 2.5% of headroom left, build the fragment prepack, prove the group-32 scale-to-K binding across a group boundary, and admit E8M0 codes 0 and 255 |
| F4-2 | Clear FP4 M=1 parity | >600.0 GB/s cold on both production shapes, full candidate step, three interleaved process medians, oracle clean | **PENDING on F4-1** | No favorable-shape pass; threshold unchanged at >600.0. Experiment 0136 rejected the 0135 decoder here by a measured 86–88 GB/s. Experiment 0137's successor ceilings at 810.93 GB/s, but that is a decoder ceiling and **not an F4-2 pass**: only 2.5% of headroom remains above the read floor, and prepack, activation feed, output and split-K must all come out of it |
| F4-3 | Clear FP4 surpass curve | >=632 GB/s at M `{1,4,8}` both shapes and >301.9 GB/s at M=16 | **PENDING on F4-2** | Per-M cost models required |
| F8-0 | Inventory and baseline Strata FP8 W8A16 operating points | Enumerate real regions/shapes/M/boundaries; measure scalar/native, existing W8A8-style WMMA control where eligible, ruler, exact bytes, and `argmax` independently | **NEXT (FP8 track)** | Experiments 0103–0105 are prior evidence, not the new W8A16 baseline. Not blocked by D-F8-GATE and not dependent on F4-1. 0135's FP4 decoder evidence carries nothing here |
| D-F8-GATE | Bind FP8 performance threshold and required M coverage | Owner selects absolute-throughput, local-efficiency, or another evidence-backed gate and names required operating points | **BLOCKED — OWNER DECISION** | Derived candidates: about 690/682/539 GB/s for equal V100 efficiency; not binding |
| F8-1 | Prove exact QPN8-derived FP8 primitive | E4M3/E8M0 block-128 fragment prepack and direct W8A16 register feed; independent decoder/matrix/boundary gates; no widened persistent path | **PENDING on C2 and F8-0** | Independent FP8 microbenchmark/resource model |
| F8-2 | Clear owner-bound FP8 performance curve | Satisfy D-F8-GATE on every required shape/M with three interleaved process medians and independent correctness | **BLOCKED on D-F8-GATE and F8-1** | FP4 thresholds forbidden here |
| MIX-1 | Integrate one-copy mixed production dispatch | Eligible FP4 uses accepted F4 path; eligible FP8 uses accepted F8 path; unsupported shapes use explicit approved exact routes; route census, admission, prepack, VRAM, graph and fixtures pass | **PENDING on F4-3 and F8-2** | No hidden fallback, no duplicate/widened weights |
| MIX-2 | Confirm end-to-end value | Real workload shows material outside-variance improvement with identical model, formats, activations, routes, and budgets; phase/resource traffic reported | **PENDING on MIX-1** | Kernel bandwidth alone is not an end-to-end claim |

## Current position

Last updated: 2026-08-22

- **Current milestone:** C2 COMPLETE (0135). **F4-1's feasible-cost-model
  blocker is CLEARED (0137)**; F4-1 proper — prepack, scale-to-K binding, and a
  full candidate step — is unblocked and is the active work. **F4-2 is NOT
  passed.** F8-0 is open, unblocked, and independent.
- **The FP4 decoder question, resolved.** Experiment 0136 rejected the 0135
  shift/rebias decoder: a decoder-only upper bound measured 514.02/512.00 GB/s
  against the >600.0 GB/s parity gate, ALU-bound at 1.62–1.65x the DRAM term.
  Experiment 0137's PRMT/LUT successor costs **9.4 ALU ops per code-pair**
  against 21.4 and ceilings at **810.93 GB/s on both shapes, 97.5% of the
  measured read floor**, oracle clean. `argmax` has moved from ALU to DRAM.
- **The 13.1 ops-per-code-pair budget was derived in 0136 before the successor
  existed, and it held.** A static SASS instruction count is therefore the
  campaign's accepted cheap screen for any future decoder: reject one that
  cannot fit the budget before benchmarking it.
- **What 0137 does NOT establish.** It is a decoder ceiling, not a candidate:
  no fragment prepack, no activation feed, no output publication, no split-K.
  Only **2.5% of headroom** remains above the read floor and every remaining
  cost must come out of it, so F4-2 is genuinely open. The `decode_mma` arm
  still uses the 0135 decoder, so **whether the MMA is still free must be
  re-measured with the successor** and may not be inherited from 0136, where it
  was hidden inside 1.6x of ALU slack.
- **Operating point, and the 250 W cap.** The RTX 3090s run at a 250 W limit
  against a 350 W default, an owner constraint that was not changed. Sampled
  during the probe the device holds **1605 MHz at 101–120 W with `SW Power Cap:
  Not Active`**, because arms are 160–260 µs with a 256 MiB scrub between them.
  **Experiment 0136's verdict was therefore not clock-distorted.** Under
  sustained 10 s load the cap does bite, and the finding is directional: an
  ALU-bound FP4 kernel is penalized twice, in instructions and in clock, while
  a memory-bound one is nearly immune. Sustained, the 0135 decoder drops to
  1440 MHz and loses 8.4% of throughput; the successor drops much harder to
  1065 MHz and loses **1.0%**, because ALU is no longer its `argmax`. Any
  future arm that runs long must report its sustained clock, not only its cold
  number.
- **Measured campaign constants, at M=1 on the two production shapes.** Read
  floor 831.59–842.32 GB/s cold on the FP4 stream, reproducing the 842-class
  ruler. `B_ALU` **10.35 Tops/s**, superseding the 8.92 predicted by 64 INT32
  lanes, because `IMAD` issues on the FMA pipe. Neither may be reused at
  another operating point.
- **E8M0 validity, measured not asserted.** The 0135 additive decoder is exact
  only across codes 2–250 and fails at 172 mismatches on 1–254. The 0137
  successor is exact across **1–254**, the full range representable as normal
  BF16, closing 0135 limitation 1. **Codes 0 and 255 are wrong in both** — code
  0 is subnormal in BF16 and 255 is the E8M0 NaN encoding — and require an
  explicit admission check in any promoted design.
- **The shared C2 fact, unchanged:**
  `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` has a verified
  lane/register map for A (4 x b32), B (2 x b32) and C/D (4 x f32), is
  bit-exact over all 128 accumulator elements, lowers to
  `HMMA.16816.F32.BF16` and no other `HMMA` form, and takes every operand from
  registers with 0 spill bytes, 0 shared memory and 0 barriers.
- **Register budget for both tracks:** 65,536 registers and 48 warps per SM, so
  100% occupancy needs <= 32 registers per thread, 67% <= 64, 44% <= 96,
  33% <= 128. 0135's reported 33.3% occupancy is the 16-blocks-per-SM cap
  meeting a one-warp-per-block probe and is not a kernel occupancy result.
- **Known gauge:** a functional MMA test cannot distinguish a shared M
  relabeling between A and D, a shared N relabeling between B and D, or a
  shared K relabeling between A and B. These are invariants of the instruction,
  pinned by the PTX specification rather than by measurement. The K relabeling
  becomes observable wherever an external index binds to the true K coordinate,
  which is what an FP4 group-32 scale and an FP8 block-128 scale both do.
- **Open FP4 limitations inherited from 0135 and 0136:** the group-32
  scale-to-K binding is unproven and no probe has yet crossed a group boundary
  in fragment order; E8M0 codes 0 and 255 need admission; no M curve exists;
  and no fragment prepack has been built.
- **Preserved validated work:** C1/0134 — ruler 840.7–845.6 GB/s, production
  control 87.04 GB/s, exact N64 conventional WMMA 161.37/174.08 GB/s,
  418.5–419.0 MiB. C2/0135. The 0136 rejection and its arms remain as the
  control that the successor is measured against.
- **What does not exist:** an accepted Ampere-native register-fed production
  kernel for either Strata FP4 W4A16 or Strata FP8 W8A16, any full FP4
  candidate step, an independent new FP8 W8A16 operating-point baseline, or
  accepted mixed production dispatch.
- **Device enumeration, confirmed not assumed:** `nvidia-smi` reports the RTX
  5060 Ti at index 0 and the RTX 3090s at 1 and 2; the CUDA runtime reports the
  RTX 3090s at 0 and 1 and the 5060 Ti at 2 — nearly reversed. Every probe must
  hard-check compute capability 8.6 and record `device_name` inline.
- **Current blockers:** D-F8-GATE is an open owner decision blocking FP8
  performance acceptance only; it does not block F8-0. F4-1 has no external
  blocker.
- **Branch disposition:** preserve `exp/dsv4-qpn-packed-decode` and experiments
  0127–0133 as controls/falsifications. Continue only on the clean main-based
  `exp/dsv4-sm86-qpn-register-feed`; do not merge failed archived runtime code.
- **Unrelated worktree state:** preserve untracked
  `scripts/dsv4_decode15_bench.sh`, `scripts/dsv4_decode15_server.sh`, and
  `scripts/dsv4_server_prefill_bench.sh`.

## Exact next step

Execute F4-1 proper, in this order, because each step gates the next and the
remaining margin is thin:

1. **Re-measure whether the MMA is still free**, by adding a `decode_mma` arm
   built on the 0137 successor rather than the 0135 decoder. In 0136 the MMA
   cost nothing measurable, but it was hidden inside 1.6x of ALU slack that no
   longer exists — only 2.5% of headroom now separates the successor from the
   read floor. This is a few lines in an existing probe and it decides whether
   the candidate's budget survives the tensor op. If it does not, rebuild the
   budget before any prepack work.
2. **Build the E2M1/E8M0 group-32 fragment prepack** for
   `gate_up_w1 [N=2048,K=4096]` and `down_w2 [N=4096,K=2048]`, and **prove the
   scale-to-K binding across a group boundary**, pinning K to the PTX
   coordinate rather than to 0135's self-consistent gauge. Confirm the
   successor's 16-bit-apart nibble pairing composes with the fragment order
   rather than fighting it.
3. **Carry an explicit admission check for E8M0 codes 0 and 255**, which both
   decoders get wrong and which the widened 1–254 window does not cover.
4. **Only then** time a full candidate step — prepack, activation feed, MMA,
   output publication and any split-K included — against the unmoved
   **>600.0 GB/s** parity gate on both production shapes, three interleaved
   process medians, oracle clean. Report the sustained SM clock alongside the
   cold number.

Do not report 0137's 810.93 GB/s as an F4-2 pass. It is a decoder ceiling.

**Independently, F8-0 is open, unblocked, and does not depend on any of the
above.** Inventory every real FP8 region, operation, `(M,N,K)`, activation
carrier, publication boundary, and frequency in the target workload, then
measure the scalar/native incumbent, the existing W8A8-style WMMA control where
eligible, the read ceiling, the exact `W_FP8 = N*K + (N/128)*(K/128)` bytes,
and `argmax` independently per M band. Declare no FP8 performance verdict:
**D-F8-GATE remains open and only the owner can close it.**

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
