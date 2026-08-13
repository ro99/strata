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
