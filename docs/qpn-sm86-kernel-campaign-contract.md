# RTX 3090 FP4 QPN kernel campaign contract

Status: **ACTIVE — authoritative campaign contract**

Established: 2026-08-22

Target: NVIDIA RTX 3090, Ampere GA102, SM86

Headline goal: **greater than 600 GB/s cold effective packed-weight bandwidth**

Archived implementation evidence head: `exp/dsv4-qpn-packed-decode@ecfb50d`

Required next implementation base: a fresh `exp/` branch from `main`

This document is the single source of truth for the RTX 3090 FP4 expert-kernel
campaign. Every agent must read it in full, including the last row of the log,
before planning, coding, profiling, interpreting a result, or handing off work.

The **Contract** section is immutable unless the owner explicitly changes it.
Agents may update the milestone status, Current position, blockers, next step,
and append-only log. Older entries must never be edited or deleted; correct an
error with a new log row and, when necessary, an explicitly dated amendment.

## Contract

### 1. What is expected

Build an exact, dependency-light C/C++/CUDA kernel for Strata's production FP4
expert projections that uses the RTX 3090 well enough to exceed 600 GB/s cold
effective packed-weight bandwidth. The campaign is inspired by the execution
architecture in `dnv2003/v100-skinny`, but it is an **Ampere/SM86 campaign**.
It is not a Volta porting exercise and success on V100 is not evidence of
success on RTX 3090.

The binding performance ladder is:

1. **Parity:** exceed 600.0 GB/s cold at M=1 on both production per-rank expert
   shapes, `gate_up_w1 [N=2048,K=4096]` and
   `down_w2 [N=4096,K=2048]`.
2. **Surpass:** reach at least 75% of the measured 842 GB/s cold RTX 3090 read
   ceiling, currently 632 GB/s after rounding, at the real M=1–8 operating
   points. At minimum, report M in `{1,4,8}` on both shapes. Also beat the
   published v100-skinny NVFP4 M=16 result of 301.9 GB/s on both shapes.
3. **Production:** preserve the win after dispatch, prepack, reduction, memory,
   graph, and end-to-end routed-decode costs are included. An isolated probe win
   is necessary but is not a production win.

The absolute numbers are the target. Hardware labels, theoretical bandwidth,
and relative speedups cannot substitute for them. A result within observed
variance is not a pass.

### 2. Exact workload and correctness contract

The target is Strata's actual DeepSeek V4 per-rank expert projection contract:

- packed E2M1 FP4 weight codes, two codes per byte;
- E8M0 group scales, one scale per 32 weights;
- fp32 activation storage whose values are BF16-representable at this boundary;
- fp32 accumulation/reduction behavior under the existing declared numerical
  contract;
- production shapes and real skinny decode/verification M values;
- unchanged precision, scale semantics, routing, expert count, top-k, and model
  behavior.

Every candidate must pass the existing int4 reference oracle. M=1 has so far
been bit-identical to production for accepted probe arms. Wider M may retain
only already-declared summation-order differences; a new delta requires an
explicit numerical-contract review and may not be normalized as harmless.
Exhaustive decoder tests are necessary but insufficient: random matrix tests,
both production shapes, real-weight/real-activation tests, and eventually the
full runtime oracle are mandatory. The V100 project itself found an FP16
overflow only with real activations; synthetic-only correctness is not enough.

### 3. Measurement contract

All headline kernel numbers are cold measurements on an identified RTX 3090.
Use the established L2 scrub and arena rotation, three warmups, eleven
interleaved samples, and report the median of at least three independent,
interleaved process repetitions. State the byte formula used for effective
bandwidth; packed codes and scale traffic must not be silently omitted or
double-counted when comparing arms.

The current achievable ruler is **842 GB/s cold** and approximately 857 GB/s
hot, measured by experiment 0129 with a 128 MiB ILP-4 stream. Re-measure it on
the new branch and whenever clocks, power, driver, CUDA, or device change.
Never use the older 435–484 GB/s launch-deflated ruler.

For every optimization, instantiate
`tau = max_r(W_r / B_r) + Sigma_serial` at the actual operating point. Report
main and reduction phase time, DRAM bytes and bandwidth, executed instructions,
relevant ALU/tensor utilization, registers, shared memory, spills, grid/waves,
occupancy, eligible/no-eligible cycles, and barrier/scoreboard stalls. Name the
resource `argmax` and the serial term before selecting a mechanism. Profiled
replay timings are attribution evidence, not substitutes for clean native event
timings.

CUDA enumeration is not stable across tools on this machine. Standalone CUDA
probe ordinals 0 and 1 were RTX 3090s and ordinal 2 was the RTX 5060 Ti in the
recorded session; `nvidia-smi`/PCI order differed. Verify and record
`device_name` every session. The target result is invalid if collected on the
5060 Ti by mistake.

### 4. Memory and integration contract

- There must be only one persistent packed representation of the weights in
  the production design. A fragment-order prepack may replace the canonical
  device layout after all consumers and fallbacks are accounted for; it may not
  become an unnoticed second full weight copy.
- No persistent FP16/BF16-expanded weights may exist in HBM. Weights must stay
  packed across HBM and widen only at point of consumption, preferably directly
  into registers.
- Prepack time, temporary workspace, scale storage, partial reductions, and
  output buffers must be measured and reported. Probe-only duplicate buffers
  must be declared and removed or admitted before production promotion.
- Exact mode must either execute the exact path or report failure. No silent
  fallback may make a benchmark appear correct or fast.

### 5. The transferable QPN thesis

The promising part of v100-skinny is its **dataflow**, not its GPU-generation
specific instruction:

1. Keep low-bit codes compressed through HBM.
2. Pre-permute codes at weight-load time into the fragment order consumed by
   the tensor-core instruction.
3. Arrange nibble pairs so decoder output already occupies the required
   operand registers; avoid inner-loop shuffles and materialized widened tiles.
4. Decode codes and apply scales at the point of consumption in registers.
5. Share activation operands across as much N work as the SM86 instruction and
   lane map permit.
6. Avoid shared-memory staging and block-wide barriers in the main weight loop
   when the native SM86 fragment path permits it; retain only reductions that
   measured geometry requires.
7. Use split-K and independent accumulator sets as measured latency/eligibility
   controls, not copied constants.

This thesis targets the two measured defects in our recent arms: excessive
ALU/instruction work and the large no-eligible `Sigma_serial` exposed by
shared-memory/barrier-heavy geometry. Expected costs must also be charged:
fragment prepacking, more registers, possible occupancy loss, activation
conversion/load work, scale handling, split-K partial storage, and reduction.

Using tensor cores remains plausible. The rejected object is the current
high-level, shared-memory-heavy WMMA architecture as the primary path—not tensor
cores as a class.

### 6. What does not transfer automatically from Volta

The following are reference evidence, not Ampere design decisions:

- Volta's inline PTX
  `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`;
- its quadpair/lane/fragment mapping and all-four-quadpairs-on-N geometry;
- FP16 activation operands;
- FP8 E4M3 group-16 scales and the reference's scale-folding sequence;
- V100-specific split-K values `{8,16,32}`, NACC values `{1,2}`, register
  count, occupancy, launch geometry, and dispatcher boundaries;
- V100 throughput numbers and the assumption that a legacy instruction which
  compiles on SM86 is efficient there.

Our likely primary candidate is an inline, Ampere-native register-fed MMA path,
with native BF16 MMA a strong candidate because of Strata's activation
contract. The exact SM86 instruction shape, operand types, lane mapping,
fragment prepack, scale application, split-K, and accumulator count are
**unresolved experimental questions**. They must be selected by an exact SM86
mechanism probe and SASS/counter evidence. Neither high-level WMMA nor a literal
Volta `m8n8k4` port is the presumed answer.

### 7. Source-of-truth evidence

Read these sources before changing the interpretation:

1. Upstream implementation:
   <https://raw.githubusercontent.com/dnv2003/v100-skinny/refs/heads/main/kernels/skinny_kernels.cu>
   The audited 2026-08-22 copy is byte-identical to
   `results/qpn-surpass/reference/skinny_kernels.cu`, SHA-256
   `112ae57bdae0a0763d9898ad7b49eaf927be411cda0bde5344a76821d166533e`.
   Relevant code is `skinny_nvfp4_qpn2`, its fragment-order prepack contract,
   `MMA_8N8K4`, split-K, and NACC—not merely the older SIMT/WMMA kernels near
   the start of the file.
2. Upstream README snapshot:
   `results/qpn-surpass/reference/v100-skinny-README.md`, SHA-256
   `433e66b4b538297893223431fe6da56a4df3d2ec33d175809b12e73331a0162b`.
   It reports QPN2 at 679.5/676.6/619.8 GB/s for M=1/4/8 and first-generation
   QPN at 301.9 GB/s for M=16, against an 879 GB/s V100 read ceiling.
3. Reddit discussion:
   <https://www.reddit.com/r/LocalLLaMA/comments/1vsq3zg/nvfp4_on_volta_despite_being_built_for_blackwell/>
   The author's RTX 3090 statement is: “3090 support is next on the list, it
   should work; the question really is if it is better than the int4 path.”
   Therefore the thread contains no measured Ampere validation. The disputed
   comment claiming stock vLLM already widens NVFP4 after HBM supplies no
   benchmark or kernel evidence and is not a campaign result.
4. Local experiment records 0127–0133 under `docs/experiments/`. They are the
   authority for what Strata actually measured, but their old interpretation
   of which upstream mechanism produced the V100 headline is superseded below.

Evidence ranks as: target-hardware measurement with raw artifacts and oracle
gate; inspected source/SASS; upstream measurement on other hardware; author or
commenter claim; inference. Never promote a lower-ranked claim over a higher-
ranked contradictory result.

### 8. Explicit corrections to prior campaign drift

These corrections are binding:

- The published v100-skinny M=1–8 NVFP4 results are the shipping **QPN2** path,
  not the legacy `skinny_nvfp4_simt` path. Experiment 0128 and both 2026-08-21
  handovers state or imply otherwise because they did not inspect the complete
  current upstream source. That interpretation is wrong.
- Experiment 0130's “faithful-port line” was a faithful attempt at selected
  legacy SIMT mechanisms under Strata's format. It was not a faithful port of
  current QPN2. Its rejection remains valid for that SIMT architecture only.
- Experiments 0127, 0128, and 0132–0133 optimize a conventional WMMA pipeline
  with decoded operands staged through shared memory. QPN2's central feature is
  register-fed fragment-order MMA with no shared memory in the main loop. Those
  experiments measure useful controls, not the target architecture.
- Experiment 0133's 10.7–17.9% N64 improvement is real and correctly
  attributed, but approximately 150–161 GB/s at the binding wide-M points is
  nowhere near the greater-than-600 goal. It must not redirect the campaign
  into endless WMMA geometry tuning.
- “Volta versus Ampere” is neither an excuse nor permission for a literal port.
  Ampere is the target; every mechanism must be re-derived and measured for
  SM86 while preserving the transferable QPN dataflow.

## Recorded work and lessons

### Experiment ledger

| Experiment | What was actually tested | RTX 3090 result | Binding verdict and lesson |
|---|---|---|---|
| 0127 | Conventional BF16 WMMA with scalar FP4 decode and shared-memory operands | 114.5/124.3 GB/s at M=1; 111.6/124.3 at M=8 | M=1 gate failed. Flat ceiling was not DRAM. Activation fp32 must be explicitly converted to BF16. |
| 0128 | Packed shift/rebias decode in the same WMMA pipeline; then a legacy-style SIMT arm | WMMA 128.0/140.4 at M=1 and 135.7/145.5 at M=8; SIMT 88.8/87.0 at M=1 | 200 GB/s gate failed. Exact scale application consumed the apparent decode headroom. The record's claim that SIMT produced upstream's headline is superseded. |
| 0129 | Corrected 128 MiB ILP-4 cold-read ruler | 840–846 GB/s cold, campaign denominator 842 | Kept. The old 435–484 ruler was launch-deflated and is forbidden. |
| 0130 | Legacy SIMT staging, R=2/R=4, prefetch, and lane-private multi-code decode | Best 151.0/120.9 GB/s at M=1 | Pre-stated 300 GB/s kill fired. Staging and R=2 helped; R=4 and deep prefetch failed. Reject this SIMT line, not QPN2 dataflow. |
| 0131 | M=1 Nsight attribution of SIMT and packed WMMA | SIMT ALU/issue-bound. WMMA had 62–65% no-eligible cycles, 16–24% occupancy, ~7 us DRAM term, and 19–22 us serial residual | Main kernels were not DRAM-bound. Stop inventing latency stories; reduce instruction work and serial/barrier exposure. |
| 0132 | Independent M=8/M=16 WMMA profile | 124–140 GB/s; same eligibility defect and only 20–22% DRAM use | Batch did not move the bottleneck. A bounded N64 control was justified. |
| 0133 | N64 versus N128 conventional WMMA geometry | N64 150.1–161.2 GB/s at M=8/16, 1.107–1.179x faster | Kept as a probe control only. It reduced serial residual but remains far below the campaign goal and is not production-promoted. |

### Do not repeat without new evidence

- Do not continue conventional WMMA tile tuning merely because N64 passed its
  local relative gate. A new arm must test the register-fed SM86 thesis or be
  justified by a newly measured larger term.
- Do not reopen the rejected legacy SIMT line, R=4 geometry, or D=4/D=8 deep
  prefetch without a new profile showing that its rejected premise changed.
- Do not use the PRMT-LUT `dequant_pair` variant; it was about 28% slower than
  shift/rebias in the recorded comparison.
- Do not infer latency-boundedness from hot approximately equaling cold. A
  streamed no-reuse weight path makes that equality structural.
- Do not cite a fast number from a kernel that failed the oracle. A prior
  incorrect prefetch build reached an apparent 167 GB/s by loading only half
  the code stream.
- Do not trust algorithm-only decoder models to catch CUDA indexing, lane-map,
  inactive-lane, alignment, or shared-memory bugs. Use exhaustive bit tests and
  device-side matrix/oracle probes.
- Do not use word-level XOR swizzles with `float4`; they can break 16-byte
  alignment. Do not allow idle lanes to read unstaged shared memory; NaN times
  zero remains NaN.
- Do not compare different formats, scale group sizes, activation types,
  shapes, batches, or byte-accounting formulas as though they were equal.
- Do not treat the Reddit thread, a V100 result, or successful SM86 compilation
  as an RTX 3090 performance measurement.

## Milestones

Milestone order is binding. A milestone advances only when its gate is recorded
in a committed experiment document and linked from a new log row. `BLOCKED`
means an external dependency prevents the next falsifier; a failed hypothesis
is `REJECTED`, recorded, and replaced only by a newly justified hypothesis.

| ID | Milestone | Gate | Status | Evidence / note |
|---|---|---|---|---|
| C0 | Freeze correct campaign understanding | Contract records Ampere target, exact goal, sources, corrections, tried work, and update protocol | **COMPLETE** | This document, 2026-08-22 |
| C1 | Establish clean SM86 baseline | Fresh branch from `main`; contract commit present; device identity, 842-class ruler, production controls, byte formula, and correctness reproduced in three interleaved repetitions | **NEXT** | Do not inherit experimental runtime code from `exp/dsv4-qpn-packed-decode` |
| C2 | Prove an exact register-fed SM86 primitive | Exact E2M1 + E8M0/group-32 decode feeds a chosen native SM86 MMA operand mapping without persistent widening; intended instructions confirmed in SASS; no inner-loop spill or unintended widened-memory path; cost model shows the design can still exceed 600 after unavoidable terms | **PENDING** | Compare against conventional WMMA and, if useful, legacy `m8n8k4` only as controls |
| C3 | Clear M=1 parity on production shapes | Greater than 600.0 GB/s cold at both `[2048,4096]` and `[4096,2048]`, full candidate step including its reduction, three interleaved process medians, oracle clean | **PENDING** | No threshold shopping or favorable-shape-only pass |
| C4 | Clear the surpass curve | At least 632 GB/s cold at M `{1,4,8}` on both shapes and greater than 301.9 GB/s at M=16, with per-operating-point cost model and correctness | **PENDING** | Re-measure every M; do not reuse M=1 constants |
| C5 | Integrate one-copy production dispatch | Runtime uses the accepted path with one persistent packed representation; prepack/admission/VRAM/graphs/fallback accounting and operation/layer fixtures pass; `make check` green | **PENDING** | Probe success alone cannot advance this milestone |
| C6 | Confirm end-to-end value | Real routed decode shows a material, outside-variance improvement with identical model/precision/routes/budgets; phase times and all resource traffic reported | **PENDING** | Kernel bandwidth is not itself an end-to-end claim |

## Current position

Last updated: 2026-08-22

- **Current milestone:** C1 — establish a clean SM86 baseline.
- **What exists:** a measurement/probe branch containing experiments 0127–0133
  and useful controls, ending at `ecfb50d`. Its best conventional WMMA/SIMT
  arms remain roughly one quarter of the greater-than-600 target or less.
- **What does not exist:** an exact Ampere-native, fragment-prepacked,
  register-fed QPN-style kernel for Strata's E2M1 + E8M0/group-32 contract.
- **Current measured bottlenecks:** legacy SIMT is ALU/instruction-bound;
  conventional packed WMMA has an ALU resource maximum plus a still larger
  no-eligible/barrier scheduling residual. Neither is DRAM-bound.
- **Current blockers:** no external blocker. The technical unknowns are the
  correct native SM86 MMA shape/lane mapping, exact register-side scale/decode
  sequence, and geometry that retains enough eligibility without spills.
- **Branch disposition:** preserve `exp/dsv4-qpn-packed-decode` and experiments
  0127–0133 as an archive of controls and falsifications. Do not delete or merge
  its failed runtime code. Begin implementation from `main`; transfer this
  contract and its AGENTS/CLAUDE pointers as a documentation-only commit.
- **Unrelated worktree state:** preserve the three untracked scripts
  `scripts/dsv4_decode15_bench.sh`, `scripts/dsv4_decode15_server.sh`, and
  `scripts/dsv4_server_prefill_bench.sh`.

## Exact next step

Complete C1 and nothing beyond it in the same experiment:

1. Create a new `exp/dsv4-sm86-qpn-register-feed` branch from `main` and bring
   in only the documentation-only commit containing this contract and its
   AGENTS/CLAUDE pointers—none of the probe/runtime changes from
   `exp/dsv4-qpn-packed-decode`.
2. Reconstruct the smallest production-shape probe needed to reproduce device
   identity, the 842-class cold ruler, the production reference, and the old
   conventional WMMA control. Record exact byte accounting and three
   interleaved process medians.
3. Record experiment 0134 as the clean-baseline gate. Only after C1 passes may
   an agent design C2's SM86 register-fragment mapping.

The first C2 hypothesis must be stated narrowly: a fragment-order prepack and
register-side E2M1/E8M0 decoder can feed one selected **native SM86** MMA
instruction without inner-loop shared-memory materialization while preserving
the oracle. Measure this mechanism before building runtime dispatch.

## Update and handoff protocol

At the start of every campaign interaction:

1. Read this whole document and the final log row.
2. Run the repository preflight required by `AGENTS.md`.
3. Verify that Current position matches Git and artifacts. If stale, correct it
   and append a log row before new work.
4. State hypothesis, primary metric, correctness gate, memory ceiling,
   rollback, measured bottleneck, targeted term, and effects on other resources.
5. Work on only the current milestone's cheapest falsifier.

Before every result commit or handoff:

1. Run `make check`.
2. Write a numbered experiment record containing raw-artifact paths, complete
   runs, median, cost model, correctness, gate verdict, and failures.
3. Update only the mutable milestone status and Current position sections.
4. Append exactly one log row. Never edit an earlier row.
5. State one exact next action and all real blockers. “Continue optimizing” is
   not an acceptable next action.

If evidence falsifies a gate, record `REJECTED` and stop that line. Do not lower
the gate, change the batch/shape, or proceed to a dependent milestone. If an
old statement is wrong, preserve it and add a correction row pointing to the
new evidence.

## Append-only campaign log

One row represents one state-changing interaction, experiment verdict, owner
amendment, or handoff. Keep the row concise; put detailed evidence in a tracked
experiment document. Timestamps use America/Sao_Paulo (`-03:00`).

| Timestamp | Agent/session | Branch@commit | Milestone | Action and evidence | Gate/result | Current blockers | Exact next action |
|---|---|---|---|---|---|---|---|
| 2026-08-22T12:53:41-03:00 | Codex / contract reset | `exp/dsv4-qpn-packed-decode@ecfb50d` before contract commit | C0 | Re-read upstream `skinny_kernels.cu`, upstream README, Reddit thread/comments, and experiments 0127–0133; established the Ampere-specific contract and corrected the QPN2 interpretation | C0 complete; no performance claim | No external blocker; SM86 register-fed mechanism unmeasured | Create `exp/dsv4-sm86-qpn-register-feed` from `main`, bring in only this documentation commit, and execute C1 only |
