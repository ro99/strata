# GLM-5.2 throughput — progressive agent handoffs

Status: **T0 and T1 complete; later performance stages have not started**.

This document turns the topology, residency, and transfer proposal into a
sequence of bounded assignments. It is written for agents that may not share a
chat history. Repository state and committed artifacts, rather than previous
conversation, are the handoff mechanism.

## Controlling rule

One agent session performs **one stage only**.

An agent must stop after completing, rejecting, or documenting a blocker for
its assigned stage. It must not begin the next stage, even when that work looks
small or closely related. The next stage is a fresh assignment based on the
previous stage's committed evidence.

Later stages are conditional. A listed implementation is not authorization to
perform it when an earlier measurement or simulation gate fails.

## Branch lineage and stage promotion

The accepted stage is the source of truth for the next stage. Agents must never
start a later stage from an old `main`, an unrelated branch, or an uncommitted
worktree when the prerequisite stage has accepted commits.

Every stage handoff must name both the stage branch and the exact base revision
used to create it. The next agent must verify that revision before editing:

```bash
git status --short --branch
git rev-parse HEAD
git merge-base --is-ancestor <base-revision> HEAD
```

When a stage passes its gate, its accepted implementation and evidence commits
must be merged into `main` before the next stage branch is created. Keep the
stage branch for auditability, and create the next branch from the updated
`main`:

```bash
git switch main
git merge --ff-only <accepted-stage-branch>
git switch -c <next-stage-branch> main
```

If the merge cannot be performed immediately, the next-stage branch may be
created directly at the accepted stage's commit, but the handoff must record
that commit and the merge must happen before the next stage is treated as a
new baseline. Do not silently rebase, squash, or substitute another revision.

When a stage is rejected, do not merge its experimental runtime behavior into
`main`. Leave the branch and experiment record reachable for reproducibility.
Generic tests or tooling may be extracted in a separate reviewed commit only
when they do not carry the rejected policy. A blocked stage is not merged.

An accepted stage becomes part of the validated baseline; a rejected or blocked
stage remains an isolated experiment. This is what keeps a later LLM session
from running against code that differs from the previous accepted stage.

Use this minimal prompt when assigning work in a fresh chat:

```text
Execute only stage T<n> from docs/glm52-throughput-handoffs.md. The accepted
prerequisite base revision is <base-revision>; verify HEAD contains it before
editing and create the assigned branch from that baseline. Read AGENTS.md and
all prerequisites first. Stop after its gate and handoff record; do not start
or implement any later stage in this chat.
```

## Shared experiment contract

Unless a stage explicitly narrows a metric or resource limit, all agents use
this contract.

- Model: pinned `QuantTrio/GLM-5.2-Int4-Int8Mix` revision already declared by
  the repository.
- Precision: native INT4 group-128 routed experts, INT8 group-128 ordinary
  linears, channelwise INT8 MTP tensors, and declared BF16/FP32 tensors.
- Execution: exact base autoregressive graph, exact top-8 routing, MTP disabled
  until a later plan explicitly authorizes it.
- Baseline prompt: `What is the closer start to sun, and how distant it is from it?`
- Baseline workload: official chat template, 30 prompt tokens, 128 requested
  new tokens, greedy decode, maximum context 256, devices `0,1,2`, and VRAM
  cache fraction `0.85`.
- Primary performance metric: median decode tok/s from at least three
  interleaved repetitions.
- Companion metrics: prefill tok/s and latency, initialization, demand and
  prefetch NVMe bytes, physical block reads, host writes, weight and activation
  H2D/D2H bytes, cache hits/misses/evictions, allocations, synchronization,
  RSS, and per-GPU VRAM.
- Correctness gate: `make check`, fixed-input deterministic greedy agreement,
  and the applicable target-format operation, teacher-forcing, or generation
  oracle. No hidden fallback is permitted.
- Memory ceiling: process RSS at most 216 GiB; runtime VRAM budget remains at
  the declared `0.85` cache fraction unless a stage compares equal explicit
  budgets. Record actual peak RSS and per-device VRAM.
- Rollback condition: reject or revert runtime behavior on any semantic or
  determinism failure, hidden fallback, memory ceiling breach, unaccounted I/O,
  or throughput result within observed variance. A byte-reduction experiment
  also fails if it merely moves cost to another unreported tier.

The current post-fix reference median is 0.283 decode tok/s. Each 128-token run
requests about 910.2 GB of checkpoint reads and weight H2D, with approximately
200,396 cache hits, 140,292 misses, and 134,570 evictions. The reference is
I/O-dependent. It must be refreshed if hardware, driver, power, model, or
baseline workload changes.

Pulsar's published GLM throughput is not a comparison baseline: its roughly
197 GB GGUF uses routed precision below Strata's four-bit floor. Its measured
topology, static expert ownership, and transfer-pipeline ideas are research
inputs only.

## Instructions for every agent

Before editing:

1. Read `AGENTS.md`, this entire document, the previous stage's handoff record,
   and every file named as an input to the assigned stage.
2. Run:

   ```bash
   git status --short --branch
   git log --oneline --decorate -8
   git branch -vv
   ```

3. Preserve unrelated changes. Do not work on `main`; create the stage branch
   named below, or a single-purpose branch with the same prefix when that name
   already exists.
4. Verify the prerequisite's recorded base revision and create the assigned
   branch from the accepted prerequisite baseline. Record the parent revision
   before editing.
5. State the stage hypothesis, primary metric, correctness gate, memory
   ceiling, and rollback condition before changing files.
6. Do not import runtime code, quantization, or model artifacts from Pulsar or
   Colibri. Implement Strata-native C/C++ interfaces and preserve the existing
   checkpoint.

During the stage:

- Keep raw route traces, profiler captures, generated binaries, and large logs
  in ignored deterministic paths under `results/` or `build/`.
- Put reusable long-job commands in `scripts/` and run them in a named `tmux`
  session.
- Keep prefill, decode, initialization, admission, and warm-up measurements
  separate.
- Run `make check` before any result commit.
- Make reversible, single-purpose commits. Do not fold a failed runtime policy
  into a successful infrastructure commit.

At the end, append a handoff record to the stage's experiment document using
this template:

```text
Stage:
Status: complete | rejected | blocked
Base revision:
Branch and commit:
Hypothesis:
Files changed:
Commands and tmux session:
Ignored artifacts:
Correctness results:
Run results and median:
Peak RSS and per-GPU VRAM:
NVMe, H2D/D2H, cache, allocation, and synchronization counters:
Decision:
Unresolved risks:
Next stage ready: yes | no
```

Update only the assigned stage's status in the ledger. Do not mark a later
stage in progress.

## Stage ledger

| Stage | Branch | Assignment | Status |
|---|---|---|---|
| T0 | `infra/glm52-observability` | Separate and complete runtime evidence | complete |
| T1 | `infra/glm52-topology-probe` | Measure the real hardware paths | complete |
| T2 | `exp/glm52-placement-sim` | Replay real routes against topology-aware policies | pending |
| T3 | `exp/glm52-expert-bundles` | Test atomic expert-triple caching | pending |
| T4 | `exp/glm52-static-owners` | Test stable hot-expert ownership | pending |
| T5 | `exp/glm52-host-tier` | Test explicit NUMA-aware RAM residency | pending |
| T6 | `exp/glm52-transfer-pipeline` | Test pinned and overlapped cold transfers | pending |
| T7 | `exp/glm52-gpu-roles` | Test measured cold-stream and attention roles | pending |
| T8 | `exp/glm52-expert-fusion` | Reduce expert kernel round trips | pending |
| T9 | `exp/glm52-prefetch` | Evaluate advisory prefetch on useful bytes | pending |

## T0 — complete runtime observability

**Scope:** instrumentation only. Do not change placement, replacement, kernel
math, prefetch, or I/O policy.

**Hypothesis:** the present aggregate counters hide enough prefill/decode and
per-device behavior to make placement decisions unsafe.

**Primary metric:** accounting completeness. For a run, all checkpoint reads,
weight transfers, activation transfers, cache events, and timed phases must be
attributable to prefill or decode and to a device where applicable.

**Required work:**

- Emit an ignored sequential route trace with request, token position, layer,
  selected experts in router order, and exact coefficients. Aggregate expert
  frequency is not sufficient.
- Split checkpoint, weight H2D, activation H2D/D2H, cache, allocation, and
  synchronization counters into prefill and decode intervals.
- Add per-device weight and activation transfer counters and per-device cache
  occupancy/evictions.
- Report pinned resident-spine bytes separately from evictable expert bytes.
- Time checkpoint read, upload wait, kernel execution, activation copies, and
  host aggregation without double-counting concurrent device time.
- Extend the reusable baseline script and JSON output. Keep raw traces ignored.
- Record a new experiment document under `docs/experiments/`.

**Gate:** counters reconcile on a short run, full baseline runs retain identical
tokens, `make check` passes, and instrumentation overhead is quantified. If
overhead materially changes throughput, make detailed timing opt-in while
retaining low-cost counters.

**Stop:** hand off the trace format, results path, metric definitions, and one
complete trace. Do not build a topology probe.

## T1 — measure the real hardware paths

**Prerequisite:** T0 complete and its metric definitions accepted.

**Scope:** a standalone C++/CUDA characterization tool and reproducible script.
Do not alter runtime device selection.

**Hypothesis:** measured service costs differ materially from the current
VRAM-capacity-only 2:3:3 schedule.

**Primary metrics:** median pinned H2D GB/s and row-1 expert service time per
GPU and NUMA source. Companion metrics include pageable H2D, D2H, allocation
cost, synchronization cost, and staged device-to-device activation transfer.

**Required work:**

- Identify devices by CUDA index, PCI BDF, model, compute capability, free and
  total VRAM, NUMA node, and current PCIe link state when available.
- Measure pinned and pageable H2D from memory first-touched on NUMA nodes 0 and
  1. Measure D2H using the same matrix.
- Confirm peer-access capability; when unavailable, measure the explicit
  pinned-host staging path rather than implying P2P.
- Benchmark actual INT4 and INT8 kernels at decode row counts and representative
  prefill row counts on all three GPUs.
- Measure aligned checkpoint-range reads at controlled queue depths after
  separating page-cache-hot and physical-read cases.
- Emit machine-readable results keyed by hardware and driver identity.

Use checksums or byte comparisons so a bandwidth result cannot come from an
elided or incomplete transfer.

**Gate:** at least three repetitions per path, stable results outside observed
noise, `make check`, and no runtime behavior change. If paths are statistically
indistinguishable, record that result; do not invent GPU roles.

**Stop:** publish the measured cost matrix and limitations. Do not implement a
new scheduler.

## T2 — topology-aware placement simulation

**Prerequisites:** T0 route trace and T1 cost matrix.

**Scope:** simulator only. Do not change the inference runtime.

**Hypothesis:** a topology-aware policy can project a material improvement over
the current modulo assignment and per-linear LRU at equal route, RAM, VRAM, and
I/O budgets.

**Primary metric:** projected end-to-end expert service time per decode token.
Required companion metrics are weight H2D bytes, physical/demand NVMe bytes,
host-tier hits, per-device utilization, evictions, and activation staging.

**Required work:**

- Model each GPU separately, including measured transfer and kernel costs.
- Use real tensor sizes. Treat gate/up/down as one expert placement unit while
  retaining their exact source extents and transfer bytes.
- Model pinned resident-spine bytes separately from expert capacity.
- Reproduce the current runtime's capacity-weighted assignment and per-linear
  LRU as the baseline.
- Compare, at minimum: atomic-triple LRU, stable hot triples, layer/expert
  affinity, a measured fastest-link cold path, and combinations justified by
  the trace.
- Do not use future trace events to make an online decision. An offline warm
  census must be labeled separately and evaluated on a later trace segment.
- Test sensitivity across realistic RAM and VRAM budgets instead of reporting
  one favorable point.

**Gate:** a candidate must project at least 20% lower expert service time or
weight H2D bytes than the best conventional baseline without increasing total
transfer or violating capacity. If no candidate clears the gate, stop and
record the negative result; T3 is not authorized.

**Stop:** nominate exactly one first runtime policy and freeze its simulation
configuration and expected counters. Do not implement it.

## T3 — atomic expert-triple caching

**Prerequisite:** T2 authorizes atomic triples as the first runtime policy.

**Scope:** change cache granularity only. Keep the current GPU assignment and
replacement ranking so the experiment isolates bundling.

**Hypothesis:** admitting and evicting a complete `(layer, expert)` gate/up/down
triple prevents partial-expert churn and reduces weight H2D.

**Primary metric:** decode weight H2D bytes per token. Headline acceptance still
requires median decode tok/s outside baseline variance.

**Required work:**

- Introduce an explicit expert-bundle identity and byte accounting.
- Reserve, admit, and evict all three expert tensors atomically.
- Keep dense, attention, router, and shared-expert weights pinned exactly as in
  the baseline.
- Preserve deterministic expert output aggregation order.
- Run interleaved current-cache versus bundled-cache repetitions on the same
  recorded workload and budgets.

**Gate:** exact output passes; measured byte movement agrees with the T2 model
within a documented tolerance; H2D materially decreases; and median decode
improves beyond variance. Otherwise reject the runtime policy while retaining
generally useful tests only in a separate commit.

**Stop:** record the decision. Do not add static hot placement.

## T4 — stable hot-expert ownership

**Prerequisites:** T3 complete and T2 nominates stable ownership.

**Scope:** replace evictable hot-cache behavior with a bounded static owner map
for selected expert triples. Do not add a host cache or asynchronous I/O.

**Hypothesis:** a trace-selected stable hot set avoids LRU churn and makes spare
GPU VRAM more valuable than uniform modulo placement.

**Primary metric:** resident-owner hit coverage and decode weight H2D bytes per
token; headline metric remains median decode tok/s.

**Required work:**

- Build an auditable `(layer, expert) -> device` map from an earlier warm trace
  segment or admitted sidecar; never use future events from the measured run.
- Respect exact per-device capacity after the resident spine and workspaces.
- Execute an owned expert on its owner and return only its final partial output.
- Preserve router order and deterministic aggregation.
- Compare equal-memory current/bundled replacement against static ownership
  using interleaved runs and a held-out route segment.

**Gate:** observed owner coverage and bytes agree with simulation; no GPU is a
service-time bottleneck hidden by aggregate utilization; exact output passes;
and throughput improves beyond variance without increasing physical NVMe.

**Stop:** freeze the owner-map format and evidence. Do not implement the RAM
tier.

## T5 — explicit NUMA-aware host expert tier

**Prerequisite:** T4 supplies stable ownership evidence and T2 predicts a host
tier win at the declared RSS ceiling.

**Scope:** explicit RAM residency and replacement only. Keep checkpoint reads
and device uploads synchronous for this stage.

**Hypothesis:** a bounded, expert-aware host tier uses the 216 GiB RSS budget
more effectively than passive filesystem page cache while enabling NUMA-local
GPU uploads.

**Primary metric:** physical NVMe bytes per decode token. Companion metrics are
host-tier hits, duplicate page-cache bytes, host writes, local/remote NUMA H2D,
RSS, and decode tok/s.

**Required work:**

- Store complete expert triples or explicitly justified immutable tensor slabs.
- First-touch or bind allocations to the intended NUMA node and report policy
  failures rather than silently accepting remote memory.
- Prevent an unbounded duplicate of the same bytes in the explicit cache and
  filesystem page cache; document the chosen I/O/cache interaction.
- Use a byte-bounded, auditable admission and replacement policy selected by
  T2. Do not add speculative prefetch.
- Test several fixed RAM budgets below 216 GiB and compare equal total resident
  memory.

**Gate:** physical NVMe demand materially decreases, RSS and NUMA placement
reconcile, exact output passes, and total H2D does not rise enough to erase the
storage gain. Retain the tier only if decode improves beyond variance or it is
required for a declared admission contract with no regression.

**Stop:** hand off the host-buffer ownership and lifetime contract. Do not make
transfers asynchronous.

## T6 — pinned and overlapped cold-transfer pipeline

**Prerequisites:** T1 transfer matrix and T5 buffer ownership contract.

**Scope:** transfer mechanics only. Keep placement and replacement fixed.

**Hypothesis:** reusable pinned buffers and bounded overlap of reads, uploads,
and independent expert work reduce miss service time without changing bytes.

**Primary metric:** cold-miss service time and upload-wait time per decode token;
headline metric remains median decode tok/s.

**Required work:**

- Add bounded persistent pinned-host staging arenas with explicit ownership.
- Use CUDA events to prevent slot reuse before consumers finish.
- Batch or pipeline the exact expert extents without repacking or changing
  precision.
- Overlap independent reads/uploads where the exact dependency graph permits;
  do not report summed concurrent times as wall time.
- Separate demand and future-prefetch queues even though prefetch remains
  disabled.
- Reconcile requested bytes, completed bytes, useful bytes, and waits.

**Gate:** byte counts remain equal to the fixed-policy baseline, exact outputs
pass, no slot lifetime race appears under sanitizers/stress, memory stays
bounded, and median decode improves outside variance.

**Stop:** document queue depths, arena sizes, and event ownership. Do not change
which GPU receives a cold expert.

## T7 — measured GPU roles

**Prerequisites:** T1 cost matrix, T4 owner map, and T6 transfer pipeline.

**Scope:** compare role assignments. Do not fuse kernels or add prediction.

**Hypothesis:** directing cold experts through the measured best end-to-end path
while retaining hot experts on their owners beats the capacity-only schedule.

**Primary metric:** median decode tok/s. Companion metrics are per-device cold
bytes, kernel busy time, activation staging, owner hits, and queue wait.

**Required comparisons:**

1. Existing capacity-weighted layer/expert schedule.
2. Static hot owners plus the measured best cold-stream device.
3. If T2 projects a win, a dedicated attention-capacity GPU with cold streaming
   and hot experts on the remaining devices.

Role selection must minimize measured service cost, not rely on GPU names,
advertised PCIe width, or a universal assumption that device 0 is fastest.
Retain an explicit override for reproducibility.

**Gate:** at least three interleaved repetitions; exact deterministic output;
no undeclared P2P or remote-NUMA path; and median decode above the best previous
stage outside variance. Reject dedicated attention placement independently if
it loses, even when cold-stream selection wins.

**Stop:** publish the winning role map and machine identity. Do not change CUDA
expert math.

## T8 — reduce expert kernel round trips

**Prerequisite:** the best placement/transfer baseline through T7 is stable.

**Scope:** one kernel/dataflow optimization at a time. The first assignment
must choose either device-resident expert intermediates or a fused gate/up/SwiGLU
operation, not both unless they are inseparable in the ABI.

**Hypothesis:** reducing activation H2D/D2H and stream synchronizations in the
three-matmul expert path increases throughput after weight movement is bounded.

**Primary metric:** expert execution and synchronization time per decode token;
headline metric remains median decode tok/s.

**Required work:**

- Match the INT4 reference oracle within the declared numerical contract on
  sm_86 and sm_120.
- Keep the final expert reduction order deterministic.
- Record kernel calls, activation bytes, synchronization count/time, occupancy,
  and workspace VRAM.
- Compare against the best T7 configuration with identical placement and
  route sequence.

**Gate:** operation and layer fixtures pass, full greedy output remains within
the exact contract, VRAM stays within budget, and median decode improves beyond
variance.

**Stop:** one accepted kernel/dataflow change per stage iteration. A future
agent may repeat T8 for the next independent optimization.

## T9 — advisory prefetch

**Prerequisite:** demand placement and transfer behavior through T8 is stable.

**Scope:** simulation first, then one separately authorized runtime predictor.
Prediction may affect scheduling or prefetch only.

**Hypothesis:** bounded prediction can hide remaining cold latency with a high
ratio of useful-prefetch bytes and without evicting demand-resident experts.

**Primary metric:** useful-prefetch bytes divided by total prefetch bytes.
Headline acceptance also requires lower demand-wait time and higher median
decode tok/s.

**Required work:**

- Compare no prefetch, the existing past-only transition predictor, and any
  proposed router lookahead in the simulator first.
- Give prediction a separate byte budget, queue-depth budget, and minimum
  lease. Demand always has priority.
- Count useful, late, duplicate, and wasted prefetch bytes and resulting demand
  evictions.
- Preserve exact router results; predictor output never enters the numerical
  graph.

**Gate:** simulation must materially beat no prefetch on held-out sequential
traces before runtime work is authorized. Runtime acceptance requires exact
output, bounded memory/I/O, useful-byte improvement, lower demand wait, and
decode throughput outside variance. Prediction recall alone is not a win.

**Stop:** record a negative result when prefetch does not pay for its bytes. Do
not proceed to MTP or multi-request wavefront work under this plan.

## Out of scope until this plan completes

- Precision below four bits or a smaller comparison checkpoint.
- MTP speculative decoding.
- Multi-request expert-ticket wavefront scheduling.
- Expert Compute Fabric or network peers.
- New model architectures.
- Power-limit or clock changes hidden inside an optimization result.

Those may become later research plans. They must not be used to rescue a failed
stage or blur attribution for a simpler optimization.
