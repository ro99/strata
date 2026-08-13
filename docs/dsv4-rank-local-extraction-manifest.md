# DSV4 rank-local extraction manifest

This manifest is the traceability record for the production landing of the
rank-local TP2 decode topology. It classifies **every file** in
`main...a31ac58` — the complete 42-commit experiment delta — into exactly one
of four dispositions:

- **ported** — the mechanism is production code on this branch;
- **documentation** — preserved as a research record, no runtime code;
- **excluded** — deliberately not landed, with the reason stated; or
- **deferred** — reserved for the separate Stage 10 CPU-parity work.

It is a traceability gate, not an authorization to widen production scope.
Nothing may be added to the landing because it appears here; the manifest only
records where each existing mechanism went.

## Provenance anchors

```text
landing branch    feat/dsv4-rank-local-decode
landing base      61f1f02  (main, validated research baseline)
experiment branch exp/dsv4-a2-ownership-screen   (preserved, unmodified)
experiment head   a31ac58  exp: pass M3 topology timing gate
M3 boundary tag   dsv4-m3-accepted -> a31ac58   (immutable)
delta size        91 files, +31,619 / -119
```

`exp/dsv4-a2-ownership-screen` and `a31ac58` are never rebased, rewritten, or
merged wholesale. The landing branch is cut directly from `main` and reproduces
only the accepted mechanisms.

## Disposition summary

| disposition | files |
|---|---:|
| ported to production | 27 |
| preserved as research documentation | 20 |
| intentionally excluded | 44 |
| deferred to Stage 10 | 0 files (mechanisms only, below) |
| **total** | **91** |

The authoritative per-path listing is the complete classification index at the
end of this document; `scripts/audit_dsv4_extraction_manifest.sh` checks it
against the live delta.

## Ported mechanisms

Each row names the production mechanism, the files carrying it, and the
experiment commit(s) it is extracted from. Landing commit messages cite these.

| mechanism | files | source commits |
|---|---|---|
| NUMA/rank checkpoint sharding — rank-local weight and scale slices validated against the actual manifest, quantized weight and scale treated as one atomic contract | `include/strata/deepseek_rank_shard.hpp`, `src/deepseek_rank_shard.cpp` | `579f70a` |
| Sliced checkpoint reads — `read_slice_into` for rank-local extents without materializing a whole sharded tensor | `include/strata/deepseek_checkpoint.hpp`, `src/deepseek_checkpoint.cpp` | `579f70a` |
| Persistent NUMA-bound CPU rank pools — one node's 24 CPUs per rank over one transformed intermediate shard, same addressed dispatch and same-node stealing policy as the production 48-worker pool | `include/strata/dsv4_host_moe_executor.hpp`, `src/dsv4_host_moe_executor.cpp` | `74dd7cf` |
| CPU-list worker pool construction — pin a pool to an explicit CPU set rather than a width | `include/strata/worker_pool.hpp`, `src/worker_pool.cpp` | `74dd7cf` |
| Allocation-free router and chain-state ops — live routing with no timed-path allocation | `include/strata/deepseek_ops.hpp`, `src/deepseek_ops.cpp` | `60b7fe6`, `a561f77` |
| Rank-local layer executor — two persistent CUDA/NCCL rank contexts, queued 43-layer chain, status and data collectives, BF16 publication, failure state machine, single final completion | `include/strata/dsv4_rank_local_layer_executor.hpp`, `kernels/cuda/dsv4_rank_local_layer_executor.cu` | `e0c4fd2`, `0409bdd`, `0f35333`, `7f4b112`, `c849e2f`, `f565c3b`, `3be4da4`, `ab84b2b`, `a561f77`, `aba772c`, `a31ac58` |
| Rank-local CUDA device primitives — prepared-query copy, mHC device views, FP32 branch reduction/commit/abort, transition router, host-MoE device view, MoE chain completion | `include/strata/cuda_backend.hpp`, `kernels/cuda/backend.cu`, `src/cuda_backend_stub.cpp` | `7e24251`, `74dd7cf`, `e0c4fd2`, `7f4b112`, `c849e2f`, `f565c3b`, `3be4da4`, `ab84b2b`, `aba772c` |
| **Fixed per-command host staging** — every queued submission that returns without synchronizing owns a private pinned upload slot; only the fully synchronous host-visible path keeps the shared buffer. Covers both `dsv4_paged_attention_to_mhc` and `dsv4_prepare_attention`. | `kernels/cuda/backend.cu`, `include/strata/cuda_backend.hpp` | `7da38b7`, `70b41fa` |
| Explicit VRAM byte admission — `applied = min(free * fraction, explicit)` applied before arena reservation, reported per device in the memory plan | `include/strata/runtime_support.hpp`, `src/runtime_support.cpp`, `include/strata/deepseek_admission.hpp`, `include/strata/deepseek_runtime.hpp`, `src/deepseek_runtime.cpp` (budget plumbing only), `apps/strata_deepseek_run.cpp` | `5f9df61` |
| Production NCCL build option — `STRATA_ENABLE_NCCL` compiles the topology into `strata_core` and defines `STRATA_HAS_NCCL`; OFF keeps every normal build supported and unlinked against NCCL | `CMakeLists.txt` | new on this branch; replaces the probe-only `STRATA_ENABLE_NCCL_PROBE` wiring from `579f70a`/`74dd7cf` |
| Focused unit coverage for the above | `tests/test_deepseek_rank_shard.cpp`, `tests/test_deepseek_ops.cpp`, `tests/test_cuda_backend.cpp`, `tests/test_runtime.cpp` | `579f70a`, `60b7fe6`, `a561f77`, `5f9df61` |
| Agent working-directory ignore | `.gitignore` | `579f70a` |

`src/deepseek_runtime.cpp` is ported **partially and deliberately**: only the
explicit-VRAM-budget plumbing from `5f9df61`. Its `.d4c` dependent-state
capture hook is excluded (see below). Stages 2–4 of the landing add the
production topology integration to this file as new code, not as a port.

## Preserved as research documentation

No runtime code. These records are binding for their declared scopes and are
never relabeled, including the negative results.

| file | why it is preserved |
|---|---|
| `docs/dsv4-decode-10toks-mission.md` | Canonical four-milestone state machine, stable operating point, hard invariants, and the append-only handoff log |
| `docs/experiments/0073-…-tp2-ownership.md` | Original TP2 ownership derivation |
| `docs/experiments/0074-…-transport-gate.md` | NCCL transport gate |
| `docs/experiments/0075-…-tp2-attention.md` | Rank-local attention screen |
| `docs/experiments/0076-…-attention-actual-replay.md` | Actual-format attention replay contract |
| `docs/experiments/0077-…-moe-actual-replay.md` | Actual-format MoE replay contract |
| `docs/experiments/0078-…-moe-production-chain.md` | Stage 5 production MoE review |
| `docs/experiments/0079-…-moe-residual-attribution.md` | **Rejection**: residual attribution arm |
| `docs/experiments/0080-…-full-chain-falsifier.md` | Stage 6A full-chain falsifier |
| `docs/experiments/0081-…-canonical-baseline-reconstruction.md` | Centralized baseline 149.099 ms/fwd; **rejected on the VRAM gate**, which motivated `0082` |
| `docs/experiments/0082-…-explicit-vram-byte-admission.md` | The admission contract this landing depends on; 151.156 ms/fwd ceiling-compliant baseline |
| `docs/experiments/0083-…-stage6-nccl-memory-gate.md` | NCCL memory boundary |
| `docs/experiments/0085-…-exact-rank-local-layer-executor.md` | M2 exact one-layer executor closure |
| `docs/experiments/0086-…-adjacent-chain-rejection.md` | **Rejection**: adjacent-chain arm; remains binding history |
| `docs/experiments/0087-…-current-callback-gap-attribution.md` | PASS_LIMITED attribution only; N4 recorded INVALID |
| `docs/experiments/0088-…-43-layer-correctness.md` | Callback-backed 43-layer chain-state correctness |
| `docs/experiments/0089-…-terminal-publication.md` | Terminal head, logits `343d766f3f5c0af3`, token `8806` |
| `docs/experiments/0090-…-device-only-admission-rejection.md` | **Rejection**: M3 implementation rejected before timing; cause recorded as unproven |
| `docs/experiments/0091-…-callback-free-staging.md` | **The asynchronous staging ownership failure and the production-queue-depth lesson.** Layer 1's hash moved from `f233565ecfa96477` at queue depth 2 to `384d71deb36191ee` at depth 43; one- and two-layer prefixes passed *against the defect* |
| `docs/experiments/0092-…-m3-timing-falsifier.md` | M3 PASS at `114.944312 ms` median under the user-amended `<=115 ms` gate; the old `<=30 ms` non-CPU condition retained as a **falsified planning assumption**, not relabeled |

The lessons carried by excluded fixture and probe code are captured in
`docs/dsv4-rank-local-architecture.md` and in the records above. Fixture code is
never ported merely to preserve its lesson.

## Intentionally excluded

### Fixture readers and their tests — 12 files

`include/strata/dsv4_attention_replay.hpp`, `src/dsv4_attention_replay.cpp`,
`include/strata/dsv4_moe_replay.hpp`, `src/dsv4_moe_replay.cpp`,
`include/strata/dsv4_dependent_replay.hpp`, `src/dsv4_dependent_replay.cpp`,
`include/strata/dsv4_oracle_artifact.hpp`, `src/dsv4_oracle_artifact.cpp`,
`tests/test_dsv4_attention_replay.cpp`, `tests/test_dsv4_moe_replay.cpp`,
`tests/test_dsv4_dependent_replay.cpp`, `tests/test_dsv4_oracle_artifact.cpp`

**Reason.** `.d4c`/`.d4r`/`.d4m`/`.d4o` are capture/replay transports built to
make the topology testable before a live path existed. The production runtime
must own live state. This is load-bearing, not cosmetic: the `.d4r` record
carries `candidates`, `indexed_positions`, `compressor_values/scores`,
`index_compressor_values/scores`, `sinks`, and materialized `pages` as fixture
data, so the M3 chain **consumed pre-computed candidate selection and
pre-materialized pages**. Stage 3 of this landing must compute all of it live.
That is the single largest scope difference between the `114.944312 ms` fixture
measurement and a live decode token, and it is why the landing gate is set from
the centralized baseline rather than carried over from `8.70 forward/s`.

### Probe and driver applications — 9 files

`apps/strata_dsv4_rank_local_layer.cu` (5,136 lines),
`apps/strata_dsv4_rank_local_attention_probe.cu` (4,954),
`apps/strata_dsv4_nccl_reduce_probe.cu` (2,423),
`apps/strata_dsv4_rank_local_moe_probe.cu` (1,613),
`apps/strata_dsv4_stage5r_chain.cu` (1,435),
`apps/strata_dsv4_tp_reduce_probe.cu` (793),
`apps/strata_dsv4_rank_shard_probe.cpp` (548),
`apps/strata_dsv4_dependent_replay_check.cpp` (205),
`apps/strata_dsv4_replay_dump.cpp` (31)

**Reason.** Standalone experiment harnesses. They own fixture loading,
independent oracles, hash diagnostics, failure injection, and timing
classifiers — none of which belongs in a production binary. The reusable
mechanism they exercised already lives in the ported executor library, by
design (`e0c4fd2` split the executor out of the driver for exactly this reuse).
Their correctness obligations return in Stage 5 as focused tests at production
queue depth.

### KV replay instrumentation — 3 files

`include/strata/deepseek_kv_cache.hpp`, `src/deepseek_kv_cache.cpp`,
`tests/test_deepseek_kv_cache.cpp`

**Reason.** The entire delta on these files is `Dsv4KvCache::read_physical_block`,
labelled in-source as "Stage-4 replay instrumentation". It exists to snapshot a
physical block for capture. Production has no caller.

### Dependent-state capture hook — partial, in `src/deepseek_runtime.cpp`

**Reason.** The M1 `.d4c` capture hook (`dependent_replay_directory`,
`dependent_replay_limit`, `dependent_replay_enabled`) is diagnostic scaffolding
for producing fixtures. Excluded with the fixture readers it feeds. The
VRAM-budget hunks in the same file are ported.

### Experiment scripts — 19 files

All of `scripts/dsv4-m3-*.sh`, `scripts/run_dsv4_stage*.sh`,
`scripts/run_dsv4_rank_local_*.sh`, `scripts/run_dsv4_dependent_replay_capture.sh`,
`scripts/run_dsv4_baseline_reconstruction.sh`,
`scripts/run_dsv4_vram_byte_admission.sh`,
`scripts/run_dsv4_stage6_nccl_memory_gate.sh`

**Reason.** Wrappers bound to fixture paths, probe binaries, and per-experiment
result directories. Their commands are transcribed verbatim into the experiment
records above, which is where they are reproducible from. Stage 6 adds one
landing-specific interleaved A/B script instead.

### Agent skill — 1 file

`hardware/luna-loop/SKILL.md`

**Reason.** Not runtime code, and `hardware/` is already in `.gitignore` on
`main` — the file was force-added on the experiment branch. Excluding it
restores the repository's own ignore contract.

### Probe-only build wiring — within `CMakeLists.txt`

**Reason.** `STRATA_ENABLE_NCCL_PROBE` and its six probe targets are replaced by
the single production `STRATA_ENABLE_NCCL` option, which compiles the topology
into `strata_core` rather than into standalone probe executables.

## Deferred to Stage 10

Mechanisms, not files. Nothing here is a defect in the landing; each is
explicitly out of scope and reserved for the separate CPU-parity experiment.

| deferred item | current measured state | why deferred |
|---|---|---|
| Routed CPU gate/up, down, and weighted-reduction arithmetic parity | `73.896784 ms` at `46.677143 GB/s` over `3,449,290,752 B/forward` (0092); centralized comparison `84.710043 ms` at `40.718794 GB/s` (0082) | The routed CPU body is `argmax_r`. M3 was a topology milestone and was never expected to establish CPU parity; the roadmap reserves this as the final bridge toward `<=100 ms/forward` |
| Decomposition of the non-CPU envelope | `41.047528 ms` total (0092); attention, HBM, link, and NCCL service were never independently timed | 0092 states the all-resource `argmax` inside this envelope is indeterminate. No optimization mechanism may be selected from it until it is instantiated at the live operating point |
| The residual gap to 10 tok/s | `14.944312 ms/forward` above 100 ms at the M3 fixture scope | Stage 10 work, re-measured at that later operating point rather than extrapolated from here |

## M3 versus this landing

They are different scopes and the manifest exists partly to keep them from being
conflated:

| | M3 (0092) | this landing |
|---|---|---|
| state source | `.d4c`/`.d4r`/`.d4m` fixtures | live checkpoint and live KV |
| candidate selection | replayed from `.d4r` | computed live on both ranks |
| pages | pre-materialized by earlier in-process arms | live transactional two-rank append |
| window | submission → terminal head | full live token incl. embedding and sampling input |
| result | `114.944312 ms` median, `8.70 forward/s` | gate: `<=125.0 ms/token`, `>=8.0 tok/s` |

The `8.70 forward/s` figure is **fixture scope only** and is never reported as an
end-to-end throughput result.

## Capability preservation ledger

Thirteen capabilities were built or established over the course of this
program. Every one is accounted for here and in
`docs/dsv4-rank-local-architecture.md`, regardless of whether it ships in the
production binary.

A capability may be preserved in one of three ways:

- **reusable production code** — general, lands on this branch;
- **DSV4-specific production code** — lands, but its shape is model-bound; or
- **documented reference** — deliberately not in the binary, preserved as
  records and immutable commits.

Exclusion from the binary never permits omission of a capability's invariants,
evidence, failure lessons, or reuse guidance. `reuse` and `model-specific`
columns are the program's own assessment, carried verbatim.

| id | capability | reuse | disposition | model-specific parts |
|---|---|---|---|---|
| CAP-01 | Cost model and bottleneck procedure | very high | documented reference | resource constants must be remeasured |
| CAP-02 | Memory admission and VRAM/RSS accounting | very high | reusable production code | tensor sizes and residency policy |
| CAP-03 | Rank-sharded checkpoint loading | high | reusable production code | tensor names, shapes, ownership map |
| CAP-04 | NUMA-bound CPU pools and resident expert storage | high for MoE | reusable production code | expert encoding and arithmetic |
| CAP-05 | NCCL FP32/status reductions and failure closure | very high | DSV4-specific production code | exact reduction placement |
| CAP-06 | Device-resident state and one-completion scheduling | high | DSV4-specific production code | model's layer state machine |
| CAP-07 | Replay capture and exact-oracle methodology | high as a pattern | documented reference | replay schemas and operation boundaries |
| CAP-08 | Physical KV-page infrastructure | moderate | DSV4-specific production code | page format and attention semantics |
| CAP-09 | Rank-local attention kernels | low-moderate | DSV4-specific production code | head layout, compression, attention algorithm |
| CAP-10 | Routed/shared MoE execution | moderate-high for similar MoE | DSV4-specific production code | router, top-k, scaling, expert shape |
| CAP-11 | mHC transitions | low | DSV4-specific production code | DeepSeek V4-specific |
| CAP-12 | Lightning Index / compressor / DSpark handling | low | DSV4-specific production code | DeepSeek-specific |
| CAP-13 | End-to-end benchmark and resource ledger | very high | documented reference | workload and thresholds must be remeasured |

Full entries follow. `scripts/audit_dsv4_extraction_manifest.sh` verifies that
each carries invariants, evidence, a failure lesson, and reuse guidance, and
that each is also named in the architecture document's capability index.

### CAP-01 — Cost model and bottleneck procedure

*Reuse: very high. Model-specific: resource constants must be remeasured.*

**Preserved as** documented reference: `research/moe-tiered-memory-decode-optimization.md`
and the decision procedure in `CLAUDE.md`.

**Invariants.** `tau = max_r W_r/B_r + Sigma_serial`. Instantiate `B_r` and
`W_r` at the real operating point *before* designing. A mechanism that does not
reduce `argmax_r` cannot improve `tau`. Check the sign on every other resource:
under a `max`, inflating the bottleneck to shrink a non-bottleneck is strictly
negative. Separate volume from overlap — a term large because it is serial is
an overlap defect, not a volume problem. Overlapping spans are never summed.

**Evidence.** Instantiated three times at this operating point: 0081
(`149.099058 = 84.171250 + 64.927808`), 0082 (`151.155686`, routed body
`84.710043` at `40.718794 GB/s`), 0092 (`114.944312 = 73.896784 + 41.047528`).

**Failure lesson.** `docs/experiments/0025-dsv4-shadow-speculative-decode-measurement-2026-07-25.md`
at `exp/shadow-speculative-moe-offload` `b3f0fd4`: ~2,900 lines of runtime code
rejected because the model was available but never instantiated. The mechanism
cut cold-transfer volume 30% while roughly doubling compute, which was already
the larger term.

**Reuse guidance.** Directly reusable as a procedure. Every constant in it is
a function of context length, prompt, cache bound, and batch shape — `tau(L)`,
not `tau`. Never carry one across operating points; 0092's `8.70 forward/s` is
the worked example of why this landing re-derived its gate instead.

### CAP-02 — Memory admission and VRAM/RSS accounting

*Reuse: very high. Model-specific: tensor sizes and residency policy.*

**Preserved as** reusable production code: `include/strata/runtime_support.hpp`,
`src/runtime_support.cpp`, `include/strata/deepseek_admission.hpp` (`5f9df61`),
and `include/strata/dsv4_rank_local_topology.hpp`, `src/dsv4_rank_local_topology.cpp`
(this branch).

**Invariants.** `applied = min(free * fraction, explicit)`, applied **before**
arena reservation and never shaved afterwards. Components are accounted
separately, never as one total, so a rejection names the component that
overran. Sums saturate, so a malformed component cannot wrap into a passing
value. Exactly one component — the centralized prefill expert cache — may be
capped; every other overrun is a hard rejection. A rejection leaves no usable
plan.

**Evidence.** 0081 rejected on this gate: `23,787,077,632` and
`23,789,174,784 B/GPU` against the `21,287,272,448 B` ceiling. 0082 passes at
exactly the ceiling with a `2,500,755,456 B/GPU` reduction.

**Failure lesson.** A planner that admits a fractional budget while the program
declares a lower byte ceiling will silently overrun. Two accounting views
(`nvidia-smi` process peak and runtime `total-free`) are not interchangeable and
must never be added.

**Reuse guidance.** The bound-selection and component-attribution logic is
model-independent. The tensor sizes, the choice of which component is
cap-eligible, and the ceilings themselves are not.

### CAP-03 — Rank-sharded checkpoint loading

*Reuse: high. Model-specific: tensor names, shapes, ownership map.*

**Preserved as** reusable production code: `include/strata/deepseek_rank_shard.hpp`,
`src/deepseek_rank_shard.cpp`, `Dsv4CheckpointReader::read_slice_into`
(`579f70a`), and `src/dsv4_rank_local_weights.cpp` (this branch).

**Invariants.** A quantized weight and its scale are one atomic contract:
either both are described consistently or descriptor construction fails.
Block/group alignment is checked against the **payload-relative** offset, not
the absolute file offset — 128 logical elements for FP8 blocks, 32 for FP4
groups, 1 for plain. No complete sharded tensor is ever materialized. Any
weight or scale read failure clears both payload vectors; there is no partial
load.

**Evidence.** `tests/test_deepseek_rank_shard.cpp` covers malformed rank,
shape, alignment, and extent contracts; validation is deliberately independent
of the reader so those paths are testable without a checkpoint.

**Failure lesson.** Aligning on the absolute offset rather than the
payload-relative one passes on a single-shard checkpoint and corrupts on a
multi-shard one.

**Reuse guidance.** The descriptor/slice machinery and the three ownership
modes (replicated, contiguous rows, strided columns) transfer directly. The
name-to-ownership map is per-model and is stated in CAP-03's architecture
section.

### CAP-04 — NUMA-bound CPU pools and resident expert storage

*Reuse: high for MoE. Model-specific: expert encoding and arithmetic.*

**Preserved as** reusable production code: `include/strata/dsv4_host_moe_executor.hpp`,
`src/dsv4_host_moe_executor.cpp`, the `HostWorkerPool` CPU-set constructor
(`74dd7cf`), and `plan_dsv4_rank_local_cpus` (this branch).

**Invariants.** Each rank owns one NUMA node's CPUs, and the two sets must be
**disjoint** — sharing cores makes the pools contend and silently destroys the
throughput the topology exists to gain. At least 24 CPUs per rank. Both ranks
read the *same* transformed tiled arena; routed-expert host storage is never
duplicated. Arithmetic, route, coefficients, precision, and accumulation order
are identical to the centralized pool — only the ownership partition differs.

**Evidence.** Same `3,449,290,752 B/forward` payload: `46.677143 GB/s`
rank-local (0092) against `40.718794 GB/s` centralized (0082).

**Failure lesson.** An interleaved arena makes roughly `(nodes-1)/nodes` of
every thread's reads remote; binding bytes to a node without also confining
compute to that node's cores captures none of the gain.

**Reuse guidance.** Pool construction, CPU assignment, and the disjointness
requirement are general to any host-executed MoE. The expert encoding, tiling,
and per-expert arithmetic are not.

### CAP-05 — NCCL FP32/status reductions and failure closure

*Reuse: very high. Model-specific: exact reduction placement.*

**Preserved as** DSV4-specific production code:
`kernels/cuda/dsv4_rank_local_layer_executor.cu` (`e0c4fd2` through `a31ac58`).

**Invariants.** Two collectives, never interchangeable — data is
`AllReduce(4096, ncclFloat, ncclSum)`; status is
`AllReduce(1, ncclUint32, ncclMax)`. The failure path **enters both
collectives** rather than skipping them, so the state machine closes
symmetrically instead of deadlocking one rank against a peer that already
returned. A failing rank sets its status by `cudaMemsetAsync(status, 1, 4)`,
giving `0x01010101` = `16843009`; `MAX` propagates it to every rank. All
outputs are withheld and zeroed on failure; the failed command is drained by a
single owner; post-failure reuse is exact.

**Evidence.** 0088 closed all eight inherited logical failure arms with exact
same-executor reuse; 0089 closed the terminal `MoePostRank1` arm with nonzero
status on both ranks and every output zeroed.

**Failure lesson.** Skipping the collective on the failing rank is the obvious
implementation and is wrong: the peer blocks forever.

**Reuse guidance.** The two-collective split, the `MAX` status discipline, and
the enter-on-failure rule are general to any rank-parallel executor. Where in
the layer the reductions sit is model-specific.

### CAP-06 — Device-resident state and one-completion scheduling

*Reuse: high. Model-specific: the model's layer state machine.*

**Preserved as** DSV4-specific production code: the executor's
`enqueue_chain_layer` / `finish_chain` / `abort_chain`, and the mHC device
views in `kernels/cuda/backend.cu`.

**Invariants.** No per-layer D2H and no per-layer host control continuation. A
43-layer chain reaches exactly **one** final completion. Command order is
strict and has no host-arithmetic fallback. `abort_chain()` is fail-closed and
the executor is reusable after it.

**Evidence.** 0088 and 0091 ran the full 43-layer chain with zero page
callbacks, zero decode checkpoint I/O, and zero timed allocations.

**Failure lesson.** See CAP-09 — queue depth is the variable that makes
one-completion scheduling correct or silently wrong, and it is invisible at
small depth.

**Reuse guidance.** The queue/drain/abort lifecycle transfers. The set of
per-layer commands and their ordering is per-model.

### CAP-07 — Replay capture and exact-oracle methodology

*Reuse: high as a pattern. Model-specific: replay schemas and operation boundaries.*

**Preserved as** documented reference. The `.d4c`/`.d4r`/`.d4m`/`.d4o` schemas,
readers, writers, and their tests are **excluded from the binary** and remain
immutable at `dsv4-m3-accepted` (`a31ac58`); see the excluded-fixtures section
above. Records: 0076, 0077, 0085, 0088, 0089, 0091.

**Invariants of the pattern.** Capture at an accepted boundary in the target
format, at the production host-visible precision. Compare against an
*independent* oracle, not against the implementation under test. Compare
exactly — hashes, not tolerances — where the contract declares exactness.
Preserve every failed arm; a rejection is evidence, not waste.

**Landing lesson — read the record before filling a gap.** `a31ac58` is much
narrower than this landing needs, so most gaps in it are real and must be
written rather than moved. That is not licence to write them without reading
it first.

Stage 3's `Dsv4RankLocalKvTransaction` has no counterpart at `a31ac58`: the
experiment patched **one** sliding row at a compile-time position from a
fixture, with no reservation, no compressed or learned-index row, and no
rollback. The transaction genuinely had to be written. But `a31ac58` computes
its page offsets once and hands *both ranks the same ones*
(`configure_live_page_patches`, `for (auto& context : contexts)`) — one logical
row, two device buffers. The first landed version of the transaction instead
reserved per rank, which `Dsv4KvCache::reserve_physical_append` rejects as
non-contiguous because reservation advances the sequence's end row. Reading the
experiment first would have shown the convention and prevented the defect; it
was instead found two commits later, by writing the tests that should have
existed from the start.

The landed offsets do agree with the record where the record reaches:
`reserve_physical_append` produces `physical_row * token_data_bytes` and
`capacity_rows * token_data_bytes + physical_row * token_scale_bytes`, which is
`a31ac58`'s formula under `physical_row`↔`kPosition`,
`capacity_rows`↔`block_rows`.

**Evidence.** M1 produced 15 valid files, 86 ordered records each, with a
complete manifest hash. The methodology caught real defects repeatedly
(0079, 0086, 0090).

**Failure lesson — the important one.** Fixtures made the topology testable
before a live path existed, but the `.d4r` record carries `candidates`,
`indexed_positions`, `index_compressor_*`, `sinks`, and materialized `pages`.
The M3 measurement therefore consumed pre-computed selection and
pre-materialized pages, and `114.944312 ms` is not a live decode token. **Do
not let a fixture boundary silently become the measurement boundary.** State
which terms the fixture supplies, and subtract them from any throughput claim.

**Reuse guidance.** The pattern is directly reusable and recommended. The
schemas are not — they encode DSV4 operation boundaries. Rebuild them per
model rather than generalizing these.

### CAP-08 — Physical KV-page infrastructure

*Reuse: moderate. Model-specific: page format and attention semantics.*

**Preserved as** DSV4-specific production code already on `main`:
`include/strata/deepseek_kv_cache.hpp`, `src/deepseek_kv_cache.cpp`,
`src/dsv4_attention_kv.cpp`. The replay-only `read_physical_block` accessor is
excluded (see above). The live transactional append is Stage 3 of this landing.

**Invariants.** Physical formats are `PhysicalFp8E4m3Group64Bf16Rope` and
`PhysicalFp8E4m3PerTensor`; 256-source-token blocks; block payload is validated
before use. Sliding, compressed, and learned-index state are **replicated** on
both ranks, not sharded, so either rank can serve attention for any position
without a cross-rank fetch. Host-visible KV metadata commits only after both
ranks and the status collective succeed.

**Evidence.** 0088 verified page/KV evolution across all 43 layers against the
sequential control.

**Failure lesson.** The M3 arms read pages already materialized by earlier
in-process arms (0091 states this explicitly). A path that never materializes a
page under test has not tested materialization.

**Reuse guidance.** The block allocator, table, and validation transfer. Page
format and the attention semantics reading it do not.

### CAP-09 — Rank-local attention kernels

*Reuse: low-moderate. Model-specific: head layout, compression, attention algorithm.*

**Preserved as** DSV4-specific production code: the rank-local attention paths
in `kernels/cuda/backend.cu` and `include/strata/cuda_backend.hpp`
(`7e24251` through `aba772c`), with the staging ownership corrections
`7da38b7` and `70b41fa`.

**Invariants.** **The asynchronous host-staging rule is the load-bearing one:**
any command that submits an async H2D and returns without synchronizing must
own a private pinned slot for that copy's lifetime. Only a fully synchronous
host-visible path may share a buffer. Realized as
`fixed_command_staging = host_deferred || device_only` for
`dsv4_prepare_attention` and `defer_host_moe_input || rank_local` for
`dsv4_paged_attention_to_mhc`, over 43 fixed 16 KiB slots per device, allocated
once at first use and never on the queued path.

**Evidence.** 0091: layer 1's observation hash was `f233565ecfa96477` at queue
depth 2 and `384d71deb36191ee` at depth 43 on identical state and fixtures.

**Failure lesson — the production-queue-depth rule.** A one-layer and an
adjacent two-layer callback-free prefix both **passed against the defect**. A
depth-bounded prefix cannot falsify a defect whose trigger is the host
outrunning the stream. Any discriminator for a queued-ownership question must
reach production queue depth or drive the host/stream race directly. The
`70b41fa` fix also shows the corollary: a live page callback was *masking* the
defect, so removing an unrelated feature silently reverted the staging path.

**Reuse guidance.** The staging-ownership rule and the queue-depth testing rule
are general and should be carried to any queued multi-command backend. The
attention kernels themselves are DSV4-shaped and largely will not transfer.

### CAP-10 — Routed/shared MoE execution

*Reuse: moderate-high for similar MoE. Model-specific: router, top-k, scaling, expert shape.*

**Preserved as** DSV4-specific production code: `src/dsv4_host_moe_executor.cpp`
(routed, host), the shared-expert GPU path in `kernels/cuda/backend.cu`, and
the allocation-free router in `src/deepseek_ops.cpp` (`60b7fe6`, `a561f77`).

**Invariants.** 256 experts, top-k 6. The selection bias changes **only** top-k
membership; coefficients are gathered from unbiased `sqrt(softplus(logit))`,
normalized, then routed-scaled. Layers 0-2 route by the checkpoint `tid2eid`
token-to-expert table rather than a learned bias — and that table is made fully
resident at load, because reading one row per token would be checkpoint I/O
inside the decode window. Precision, router semantics, expert count, top-k,
scoring, normalization, and routed scaling never change silently.

**Evidence.** 0088 independently recomputed every route from live router logits
and the checkpoint's routing data across all 43 layers. 0082 recorded 645/645
decode batches and 3,870 routed expert associations per run.

**Failure lesson.** Routing membership and coefficient derivation are separate
contracts; applying the bias to the coefficients as well as to the selection is
a silent semantics change of exactly the kind the invariants forbid.

**Reuse guidance.** The dispatch, stealing policy, and rank partitioning
transfer to a similar MoE. The router contract does not — re-derive it from the
target model's manifest.

### CAP-11 — mHC transitions

*Reuse: low. Model-specific: DeepSeek V4-specific.*

**Preserved as** DSV4-specific production code: `dsv4_mhc_*` device operations
in `kernels/cuda/backend.cu` and the executor's per-rank mHC seeding.

**Invariants.** mHC state is seeded per rank and evolves device-side across the
43 layers with no host round trip. A branch is reduced in FP32 and then either
committed (`dsv4_mhc_commit_reduced_branch`) or aborted
(`dsv4_mhc_abort_branch`); an aborted branch leaves **no partial mutation**.
The transition router and the next layer's attention mHC are queued device
operations, so each layer owns its successor's attention mHC weights.

**Evidence.** 0088/0089 terminal hashes `e1a9a77f0b01a361` (weighted),
`122a716defe84e1b` (input), `5017083817dd2848` (hidden).

**Failure lesson.** The `q_norm`/`kv_norm` staging defect (CAP-09) surfaced
here first as a layer-1 state divergence, not as an obvious transfer bug —
device-side state machines report host-staging faults as arithmetic errors.

**Reuse guidance.** Low. The commit/abort transaction shape is worth copying;
the rest is model-bound.

### CAP-12 — Lightning Index / compressor / DSpark handling

*Reuse: low. Model-specific: DeepSeek-specific.*

**Preserved as** DSV4-specific production code on `main` (indexer, compressor,
DSpark verification), extended by this landing with device-side selection over
physical-format pages (`CudaBackend::dsv4_physical_lightning_index`). In this
landing, live candidate selection is Stage 3; in M3 it was replayed from
`.d4r` (see CAP-07).

**Invariants.** DSpark verification, the declared attention/compression layout,
selection and scoring functions, top-k normalization, and routed scaling are
preserved exactly as declared by the model manifest. Both ranks run selection
independently on replicated state and must agree; correctness builds verify
rank agreement. Compressor and index values are raw FP32 where the target
format declares it, not BF16. Device selection is bit identical to
`dsv4_index_scores_f32` followed by `dsv4_index_topk_f32`, not merely close:
the dot product keeps the reference's sequential FP32 order under explicit
`__fmul_rn`/`__fadd_rn`, so no fma contraction or reassociation occurs.

**Evidence.** 0092 reproduced logits `343d766f3f5c0af3` and token `8806`;
`docs/experiments/0064` and the Lightning-Indexer correctness gate on
`feat/dsv4-gpu-lightning-indexer` hold the component records. Device selection
is verified position for position against the scalar oracle at a partially
filled tail page and at the full 262,144-candidate width, and timed by
`strata-dsv4-index-probe`: 3.158 ms per indexed layer at that width on one
3090, 66.3 ms/token across the 21 ratio-4 layers.

**Failure lesson.** Two, and both are about format and shape rather than
policy.

The FP4 GPU Lightning Indexer is admitted only with the exact compact block KV
cache. That is a *format* constraint, not a preference: block KV stores index
rows as FP4 E2M1 with per-32 E8M0 scales, physical KV stores them as E4M3 with
one F32 scale per row, and one kernel cannot read both. Config validation
rejects the pairing rather than silently producing a different selection. The
consequence went unnoticed for longer than it should have: PhysicalDevice mode
fell through to host scalar scoring, which is 1.85e11 FLOP/token at the
declared 1M context and cannot meet any decode budget. A capability fenced off
for a real reason still leaves a hole where it used to be.

The device replacement then reproduced the very defect it was written to fix.
Its first version resolved top-k with a `<<<1,1>>>` scan of 65,536 histogram
bins -- the FP4 path's serial merge, relocated. It showed up as shape, not
magnitude: 1.497 ms/layer at only 1,024 candidates, and 256x more candidates
costing 13x more time. A cost that barely moves with the work is a constant,
and a constant that large is a defect. Fixing the scan and the block geometry
took the same measurement from 842 to 66.3 ms/token.

**Reuse guidance.** Effectively none outside DeepSeek. Preserved for exactness
auditing, not portability.

### CAP-13 — End-to-end benchmark and resource ledger

*Reuse: very high. Model-specific: workload and thresholds must be remeasured.*

**Preserved as** documented reference: the operating point and gates in
`docs/dsv4-decode-10toks-mission.md`, the wrappers and ledgers recorded in
0081 and 0082, and the Stage 6 interleaved A/B script added by this landing.

**Invariants.** Report every run and the median of at least three interleaved
repetitions, with observed ranges wherever a performance gate applies. Separate
initialization, prefill, transition, and decode. Record NVMe demand/prefetch
bytes, host writes, H2D/D2H, cache hits/evictions, allocation/synchronization,
RSS, and per-GPU VRAM. Measure useful-prefetch bytes, not prediction recall.
**Never call a result a win when it is within observed run variance.** State
the fixed-setup to measured-window ratio before launching a long run.

**Evidence.** 0081's three fresh controls (`149.099058` median,
`148.433299–152.113645` range) and 0082's binding matrix (`151.155686` median,
`146.828976–151.520846`) are the reusable template.

**Failure lesson.** A DeepSeek arm at a 3,565-token prompt is 51 s of
initialization, 38.5 minutes of prefill, and 73 s of decode: if decode
throughput is the hypothesis, 95% of the arm measures nothing. Workload
parameters are part of the experiment design and are never inherited from a
script written for a different hypothesis.

**Reuse guidance.** The ledger structure, interleaving discipline, and variance
rule transfer unchanged. Every threshold and the workload itself must be
remeasured — this landing's own `8.0 tok/s` gate was re-derived from the
centralized baseline rather than carried from M3's fixture figure.

## Audit procedure

Before landing review, re-run the traceability gate:

```bash
git diff --name-status main...a31ac58 | sort -k2
```

Every path in that listing must appear in exactly one section above. A file that
is unclassified blocks the landing; a substantive mechanism inside a classified
file that is not named in the ported table above also blocks it. Passing this
gate does not authorize adding production code — an unclassified mechanism is
resolved by classifying it, and only by porting it if the landing plan already
called for it.

## Complete classification index

Authoritative per-path listing, verified by `scripts/audit_dsv4_extraction_manifest.sh`.

| path | disposition | note |
|---|---|---|
| `.gitignore` | ported |  |
| `CMakeLists.txt` | ported | production STRATA_ENABLE_NCCL only; probe targets excluded |
| `apps/strata_deepseek_run.cpp` | ported |  |
| `apps/strata_dsv4_dependent_replay_check.cpp` | excluded | probe/driver harness |
| `apps/strata_dsv4_nccl_reduce_probe.cu` | excluded | probe/driver harness |
| `apps/strata_dsv4_rank_local_attention_probe.cu` | excluded | probe/driver harness |
| `apps/strata_dsv4_rank_local_layer.cu` | excluded | probe/driver harness |
| `apps/strata_dsv4_rank_local_moe_probe.cu` | excluded | probe/driver harness |
| `apps/strata_dsv4_rank_shard_probe.cpp` | excluded | probe/driver harness |
| `apps/strata_dsv4_replay_dump.cpp` | excluded | probe/driver harness |
| `apps/strata_dsv4_stage5r_chain.cu` | excluded | probe/driver harness |
| `apps/strata_dsv4_tp_reduce_probe.cu` | excluded | probe/driver harness |
| `docs/dsv4-decode-10toks-mission.md` | documentation |  |
| `docs/experiments/0073-dsv4-rank-local-tp2-ownership.md` | documentation |  |
| `docs/experiments/0074-dsv4-rank-local-tp2-transport-gate-2026-08-09.md` | documentation |  |
| `docs/experiments/0075-dsv4-rank-local-tp2-attention-2026-08-10.md` | documentation |  |
| `docs/experiments/0076-dsv4-rank-local-attention-actual-replay-2026-08-10.md` | documentation |  |
| `docs/experiments/0077-dsv4-rank-local-moe-actual-replay-2026-08-11.md` | documentation |  |
| `docs/experiments/0078-dsv4-rank-local-moe-production-chain-2026-08-11.md` | documentation |  |
| `docs/experiments/0079-dsv4-rank-local-moe-residual-attribution-2026-08-11.md` | documentation |  |
| `docs/experiments/0080-dsv4-rank-local-full-chain-falsifier-2026-08-11.md` | documentation |  |
| `docs/experiments/0081-dsv4-canonical-baseline-reconstruction-2026-08-11.md` | documentation |  |
| `docs/experiments/0082-dsv4-explicit-vram-byte-admission-2026-08-11.md` | documentation |  |
| `docs/experiments/0083-dsv4-stage6-nccl-memory-gate-2026-08-11.md` | documentation |  |
| `docs/experiments/0085-dsv4-exact-rank-local-layer-executor-2026-08-12.md` | documentation |  |
| `docs/experiments/0086-dsv4-rank-local-adjacent-chain-rejection-2026-08-12.md` | documentation |  |
| `docs/experiments/0087-dsv4-current-callback-gap-attribution-2026-08-12.md` | documentation |  |
| `docs/experiments/0088-dsv4-rank-local-43-layer-correctness-2026-08-12.md` | documentation |  |
| `docs/experiments/0089-dsv4-rank-local-terminal-publication-2026-08-12.md` | documentation |  |
| `docs/experiments/0090-dsv4-m3-device-only-admission-rejection-2026-08-12.md` | documentation |  |
| `docs/experiments/0091-dsv4-m3-callback-free-staging-2026-08-12.md` | documentation |  |
| `docs/experiments/0092-dsv4-m3-timing-falsifier-2026-08-12.md` | documentation |  |
| `hardware/luna-loop/SKILL.md` | excluded | agent skill; hardware/ is gitignored on main |
| `include/strata/cuda_backend.hpp` | ported |  |
| `include/strata/deepseek_admission.hpp` | ported |  |
| `include/strata/deepseek_checkpoint.hpp` | ported |  |
| `include/strata/deepseek_kv_cache.hpp` | excluded | replay instrumentation (read_physical_block) |
| `include/strata/deepseek_ops.hpp` | ported |  |
| `include/strata/deepseek_rank_shard.hpp` | ported |  |
| `include/strata/deepseek_runtime.hpp` | ported |  |
| `include/strata/dsv4_attention_replay.hpp` | excluded | fixture reader/test |
| `include/strata/dsv4_dependent_replay.hpp` | excluded | fixture reader/test |
| `include/strata/dsv4_host_moe_executor.hpp` | ported |  |
| `include/strata/dsv4_moe_replay.hpp` | excluded | fixture reader/test |
| `include/strata/dsv4_oracle_artifact.hpp` | excluded | fixture reader/test |
| `include/strata/dsv4_rank_local_layer_executor.hpp` | ported |  |
| `include/strata/runtime_support.hpp` | ported |  |
| `include/strata/worker_pool.hpp` | ported |  |
| `kernels/cuda/backend.cu` | ported |  |
| `kernels/cuda/dsv4_rank_local_layer_executor.cu` | ported |  |
| `scripts/dsv4-m3-callback-free-arm.sh` | excluded | experiment wrapper |
| `scripts/dsv4-m3-timing-arm.sh` | excluded | experiment wrapper |
| `scripts/run_dsv4_baseline_reconstruction.sh` | excluded | experiment wrapper |
| `scripts/run_dsv4_dependent_replay_capture.sh` | excluded | experiment wrapper |
| `scripts/run_dsv4_rank_local_attention_stage4.sh` | excluded | experiment wrapper |
| `scripts/run_dsv4_rank_local_layer_m2.sh` | excluded | experiment wrapper |
| `scripts/run_dsv4_rank_local_tp2_nccl.sh` | excluded | experiment wrapper |
| `scripts/run_dsv4_rank_local_tp2_rank_shard_fixtures.sh` | excluded | experiment wrapper |
| `scripts/run_dsv4_rank_local_tp2_transport.sh` | excluded | experiment wrapper |
| `scripts/run_dsv4_stage4r_capture.sh` | excluded | experiment wrapper |
| `scripts/run_dsv4_stage4r_validate.sh` | excluded | experiment wrapper |
| `scripts/run_dsv4_stage5_capture.sh` | excluded | experiment wrapper |
| `scripts/run_dsv4_stage5_probe.sh` | excluded | experiment wrapper |
| `scripts/run_dsv4_stage5_profile.sh` | excluded | experiment wrapper |
| `scripts/run_dsv4_stage5r21_attribution.sh` | excluded | experiment wrapper |
| `scripts/run_dsv4_stage5r_control.sh` | excluded | experiment wrapper |
| `scripts/run_dsv4_stage6_nccl_memory_gate.sh` | excluded | experiment wrapper |
| `scripts/run_dsv4_stage6a_control.sh` | excluded | experiment wrapper |
| `scripts/run_dsv4_vram_byte_admission.sh` | excluded | experiment wrapper |
| `src/cuda_backend_stub.cpp` | ported |  |
| `src/deepseek_checkpoint.cpp` | ported |  |
| `src/deepseek_kv_cache.cpp` | excluded | replay instrumentation (read_physical_block) |
| `src/deepseek_ops.cpp` | ported |  |
| `src/deepseek_rank_shard.cpp` | ported |  |
| `src/deepseek_runtime.cpp` | ported | VRAM-budget plumbing only; .d4c capture hook excluded |
| `src/dsv4_attention_replay.cpp` | excluded | fixture reader/test |
| `src/dsv4_dependent_replay.cpp` | excluded | fixture reader/test |
| `src/dsv4_host_moe_executor.cpp` | ported |  |
| `src/dsv4_moe_replay.cpp` | excluded | fixture reader/test |
| `src/dsv4_oracle_artifact.cpp` | excluded | fixture reader/test |
| `src/runtime_support.cpp` | ported |  |
| `src/worker_pool.cpp` | ported |  |
| `tests/test_cuda_backend.cpp` | ported |  |
| `tests/test_deepseek_kv_cache.cpp` | excluded | replay instrumentation (read_physical_block) |
| `tests/test_deepseek_ops.cpp` | ported |  |
| `tests/test_deepseek_rank_shard.cpp` | ported |  |
| `tests/test_dsv4_attention_replay.cpp` | excluded | fixture reader/test |
| `tests/test_dsv4_dependent_replay.cpp` | excluded | fixture reader/test |
| `tests/test_dsv4_moe_replay.cpp` | excluded | fixture reader/test |
| `tests/test_dsv4_oracle_artifact.cpp` | excluded | fixture reader/test |
| `tests/test_runtime.cpp` | ported |  |
