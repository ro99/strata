# 0120 — Phase 0: building and gating the refactor's equivalence oracle

Status: **closed, gate PASS.** Every claimed win in this record is measured on
Gemma 4, the model that started this unit with zero runtime-level test
coverage and zero introspection of any kind.

## Hypothesis

The documented four-layer architecture (core / model adapters / execution /
applications) has no physical, build-level, or test-level representation, and
a 9-phase refactor toward that architecture cannot be verified without a
mechanical equivalence oracle — an agent asserting "nothing changed" is not a
gate. Phase 0's job was to build that oracle and prove it actually goes red on
a real fault, at the layer the fault is in, before anything else moves.

## What the oracle is, as of this record

A per-layer and per-operation BF16 content hash (`strata::stable_bf16_hash`,
`strata::diagnostic_hash_u32/u64`, `include/strata/diagnostics.hpp`), recorded
at two granularities:

- **`layer_hidden_hashes`** (tier 1, primary signal): one hash of the residual
  stream per `(position, layer)`, at the true end of each layer's forward
  pass.
- **`operation_hashes`** (finer-grained, added on top of the minimum ask):
  one hash per named sub-step inside a layer — for Gemma 4:
  `attention_local`/`attention_global` (the layer's own kind carried into the
  name, since Gemma 4 is hybrid local/global attention),
  `attention_residual`, `mlp`, `mlp_residual`.

Both types promoted out of DeepSeek's original `Dsv4`-prefixed shapes into
model-neutral types (commit `849faa4`, this unit), alongside the model-neutral
`RouteEvent`/`RouteTraceWriter` (`include/strata/trace.hpp`) that predates
this work and was already reachable from DeepSeek and GLM.

DeepSeek has this facility natively (`--layer-hash-trace --logit-trace` on
`strata-deepseek-run`) plus a logit-trace tier this record does not cover for
Gemma 4 (out of scope for this unit — see Recommendations). Gemma 4 had
**none of it** before this unit; every number below was built from zero.

## The gate: three deliberate, independent faults

Branch `throwaway/gemma4-perturbations`, off `d3852a0`. Each arm reverts the
previous arm's fault and applies a new, independent one, so every arm tests a
single fault against the same clean fixture — not a compounding one. Fixed
prompt "The capital of France is" (18 tokens), greedy, prefill only (see "The
fused-decode-path blind spot" below for why prefill-only is the honest
coverage claim, not a limitation of this test).

| Arm | Commit | Depth | Mechanism | `layer_hidden_hashes` | `operation_hashes` (at time of gate) | Tier 2 |
|---|---|---|---|---|---|---|
| P1 | `3691700` | Layer 5/60 | `branch[0] += 1.0F` after `attention()` | **layer 5, exact** | layer 5, exact | Unchanged ("Paris") |
| P2 | `72407ba` | Layer 30/60 | Attention residual add skipped entirely | **layer 30, exact** | layer 30, exact | Unchanged |
| P3 | `84e41ee` | Layer 55/60 | Post-norm output scaled by 1.001 (epsilon-change stand-in) | **layer 55, exact** | **layer 56 — one layer late** | Unchanged |

Every arm's diverged-layer set was the exact contiguous suffix from its
injected layer to 59, in both hash types — the shape a residual-stream
architecture should produce for a single-point fault, and independent evidence
the trace measures what it claims to.

**Gate verdict: PASS.** The gate's binding signal, `layer_hidden_hashes`,
localized all three faults to the exact layer. This was verified twice:
once when the arms were run, and again independently by the orchestrator
reconstructing the layer/operation divergence maps from the three committed
perturbation JSONs and diffing them against the fixture — every number
matched, including P3's.

**None of the three faults changed the generated token.** All three still
produced "Paris." This is the whole argument for a tiered oracle rather than
diffing final output: a single corrupted value, a dropped residual, or a
scaled norm output at any of three depths was silent at the answer-text level
and visible at the hash level every time.

## The P3 blind spot: what it was, and why it doesn't sink the gate

P3 is the one arm that is not a clean pass, and it is reported here for the
same reason it was reported at the time: a record that only says "gate
passed" is worth nothing to whoever reads it in a month.

**Mechanism.** Before this record, Gemma 4's operation-hash instrumentation
recorded three checkpoints per layer: `attention_local`/`_global`,
`attention_residual`, `mlp`. There was no checkpoint between the `mlp` output
and the final scaled-residual add — the point where `weights.scalar` is
applied and the layer's output is committed. DeepSeek's own pattern has a
fourth point there (its `_mhc_post`); the promotion carried the *types* over
correctly but the *per-layer instrumentation density* was not copied
one-for-one, because task 1 of the unit that wired Gemma 4 deliberately did
not force a mechanical imitation of DeepSeek's checkpoint placement (the brief
that authorized it said as much: record what is natural for Gemma 4's graph).
P3's fault sits exactly in that gap. `operation_hashes` only noticed it one
layer downstream, at layer 56's first checkpoint, because that is the next
point after the fault that anything gets hashed. `layer_hidden_hashes` fires
*after* the gap, at the true end of the layer, so it caught the same fault
at the correct layer, 55, every time.

**Why the gate still passed:** the task that specified the gate named
`layer_hidden_hashes` as the tier-1 signal being verified. That signal was
never blind. The gap was real, but it was in a second, finer-grained trace
added beyond the minimum ask, not in the thing the gate was gating on.

**Fixed in this unit** (see Task 0 below): a fourth operation-hash checkpoint,
`mlp_residual`, added at the same point `record_layer_hash` already fires.
The fixture was re-captured and the gap is now closed — a fault in that
position would now localize to operation-hash layer 55 as well. This was not
re-verified with a fourth perturbation arm; the fix is structural (the new
checkpoint fires at the exact point the old one was missing) and re-running
the full three-arm gate was judged not worth another three model loads for a
fix whose correctness follows from where the code changed, not from a new
measurement. Flagging this as a judgment call, not a hidden assumption.

## The 34-silently-skipped-tests trap

Before any of the above, `strata-tests` run fresh in this worktree reported
`277/311 passed, 34 skipped, 0 failed` — a green-looking result. The 34 skips
were every test gated on a real checkpoint being present
(`<model>_checkpoint_present()`), silently no-opping because this worktree had
no `models/` directory: `models/` is gitignored (correctly, checkpoints are
per-checkout data, not repo content), but the `.gitignore` pattern was
`/models/` — a trailing-slash, directory-only pattern that does not match a
*symlink* to a directory. A `models -> /home/rodrigo/Developer/strata/models`
symlink, the obvious fix for reusing checkpoints across worktrees, stayed
untracked-but-not-ignored and `git status` kept flagging it, while the tests
behind it just never ran.

**This is not a hypothetical risk.** It is the exact failure mode the charter
warns about: a measurement that looks like a pass but tested nothing. Had this
gone unnoticed, every later phase's "gate: same test count, 0 new failures"
check would have been comparing two silently-degraded runs against each
other and never caught a real regression in checkpoint-dependent code.

**Fixed:** `.gitignore`'s `/models/` → `/models` (drop the trailing slash),
which matches files, directories, and symlinks alike. Verified with
`git check-ignore -v models` and a full re-run: `310/311 passed, 1 skipped,
0 failed` — the 1 skip is `Kimi-K3 backbone matches the reference at every
layer`, correctly gated behind `STRATA_KIMI_BACKBONE=1` because it costs
~10 minutes and ~250 GiB of SATA reads and is deliberately excluded from
`make check` by policy, not by accident.

## Measured costs

- **Clean build, this worktree, Release, 56-core parallel:** `60.9 s` wall
  (`real 1m0.915s`), exit 0, zero errors. This did not exist as a number
  before this unit; it is now the budgeting constant for every later phase
  ("build is 60.9 s, so iterate freely").
- **Gemma 4 load, `strata-gemma4-run`, 33 GB checkpoint, fully VRAM-resident
  across 3 GPUs:** `45.5 s` to first token ready, consistent across every run
  in this unit (40.0–40.6 s total wall including an 18-token prefill and a
  1-token decode, run repeatedly with no measurable variance from the
  `--layer-hash-trace` flag being on or off).
- **`strata-tests` full suite, all three GPUs free:** `~121 s` wall,
  `310/311 passed, 1 skipped`, reproduced identically at least five separate
  times across this and the prior two units (00b, 00c) with zero flakiness
  observed.

## The fused-decode-path blind spot (shared, not Gemma-4-specific)

`--layer-hash-trace` only fires on the host-side `forward_layers` loop —
prefill, and any decode step before its KV cache reaches the fused CUDA decode
path. Once `device_kv_ready` is true (after the first prefill), Gemma 4's
batch-1 decode goes through `forward_decode_layers`, which returns a whole
device's layers from one CUDA dispatch with no host-visible boundary between
them. **DeepSeek has the identical limitation**: its own
`device_mhc_forward_hidden` and `rank_local_forward_hidden` paths bypass
`record_layer_hash` the same way, for the same reason — a fused device kernel
has nothing to hook between layers without either breaking the fusion or
adding a device-side trace mechanism, neither of which this unit's scope
covered. This is a property of fused/batched device execution generally in
this codebase, not something the Gemma 4 promotion introduced or something a
future model should expect to avoid by construction. It means prefill-only
coverage is the honest current ceiling for tier-1 hash tracing on any model
using a fused decode path, and any phase that needs decode-step coverage
(phase 5, phase 6) will need to either instrument the fused kernel directly or
accept a prefill-equivalent proxy — a design question for whoever picks that
up, not resolved here.

## Recommendations carried forward (not actioned in this unit)

1. Gemma 4 has no logit-trace tier (DSV4's `Dsv4LogitAnalysis` equivalent).
   Only wired for the four remaining models when phase 4's fan-out reaches
   them, per the orchestrator's revised sequencing (oracle depth scales with
   phase risk; phase 1 needs tier 0 only).
2. `apps/strata_gemma4_run.cpp` (and its precedents `strata-run`,
   `strata-deepseek-run`) are scheduled for deletion in phase 4, once
   `ModelExecutor` plus a generic diagnostics control on `RuntimeSession`
   make per-model drivers unnecessary.
