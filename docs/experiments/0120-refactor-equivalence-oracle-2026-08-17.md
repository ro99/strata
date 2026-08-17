# 0120 — Strata refactor programme record

Single record for the whole programme (protocol amendment, this unit):
append a section per phase here instead of a new numbered doc per phase.
Target length per phase is what a reader needs in a month, not a transcript.

## Phase 0 — the equivalence oracle

**Hypothesis.** The documented four-layer architecture has no build- or
test-level representation, and a multi-phase refactor toward it cannot be
verified by an agent asserting "nothing changed" — it needs a mechanical
oracle, proven to go red on a real fault before anything else moves.

**What it is.** A per-layer and per-operation BF16 content hash
(`strata::stable_bf16_hash`, `include/strata/diagnostics.hpp`), two
granularities: `layer_hidden_hashes` (tier 1, one hash per `(position,
layer)` at the true end of each layer — the binding signal) and
`operation_hashes` (finer-grained, per named sub-step). Built for Gemma 4,
which had zero runtime introspection before this unit; DeepSeek already had
an equivalent facility the types were promoted from.

**The gate.** Three independent injected faults (layers 5/30/55 of 60,
branch `throwaway/gemma4-perturbations`) against a fixed 18-token prompt.
All three detected and localized by `layer_hidden_hashes`; none changed the
generated token, which is the argument for a tiered oracle over diffing
output text. **Correction:** P1 and P2 land directly on an existing
checkpoint, so their "exact localization" is near-definitional. P3 landed in
a then-uninstrumented gap and missed by one layer — the only arm that
actually tested localization as a general property, and the one that
failed. The gap (`mlp_residual`) is now closed structurally but has never
been re-tested with a fourth perturbation arm; that is a judgment call, not
a hidden assumption. **The three arms were never re-run against the code
that actually shipped** (`7385e89` changed instrumentation twice after they
ran) — the real evidence the hash function still computes what it always
did is the fixture re-capture itself: all 1080 `layer_hidden_hashes` and the
aggregate `trace_hash` (`9fdbee6c3f216c01`) identical before/after.

**The 34-silently-skipped-tests trap.** A fresh run reported `277/311
passed, 34 skipped, 0 failed` — every checkpoint-gated test no-opping
because this worktree had no `models/` entry at all. **What is known:** no
`models/` directory or symlink existed when the suite first ran here, which
alone explains all 34 skips; creating the fix symlink surfaced a real,
independent `.gitignore` bug (`/models/` trailing-slash doesn't match a
symlink — fixed to `/models`). **What is not known:** whether an earlier
symlink existed and was removed by something like `git clean -fd`; no
evidence in this session shows that. The original causal claim (gitignore
bug *caused* the skips) does not hold and is retracted; the bug is real and
fixed regardless. Fixed state: `310/311 passed, 1 skipped` (Kimi-K3 backbone,
gated behind `STRATA_KIMI_BACKBONE=1` by policy — ~10 min, ~250 GiB SATA).

**Scoping facts.** Five of six models have no equivalence fixture; only
Gemma 4 does. Decode has no coverage on any model — `--layer-hash-trace`
only fires on the host `forward_layers` loop; once the fused CUDA decode
path takes over (both Gemma 4 and DeepSeek), there is no host-visible
per-layer boundary to hook. Prefill-only is the honest current ceiling for
any model on a fused decode path, not a gap this unit introduced.

**Open question (not decided here).** `stable_bf16_hash` rounds f32→bf16
before hashing, so a difference that never crosses a bf16 rounding boundary
is invisible by construction — exactly the class ordinary code motion
produces (inlining, FMA contraction shifting across a TU boundary). Phase 3
moves code wholesale. One option: an opt-in f32-level hash for phases that
must change *nothing*, alongside the bf16 one for phases where bf16
rounding is the actual contract. **Resolved by the protocol amendment:**
does not block Phase 3, whose gate is a pure `git diff -M` rename check (see
below) that needs no numerical oracle at all. Bites at Phase 5, where code
genuinely moves between functions; deferred to be decided there.

**Measured costs.** Clean build 60.9 s (Release, 56-core). Gemma 4 load
45.5 s to first token, 33 GB checkpoint, fully VRAM-resident. Full suite
~121 s, reproduced flake-free five times across three units.

**Fixed in this unit (brief 05 cold-review remediation).** Nothing had run
the oracle — the fixture was referenced by zero test/script/CMake target.
Added `scripts/check_equivalence.py` + a guarded `strata-equivalence-gemma4`
ctest entry (`SKIP_RETURN_CODE` on missing checkpoint) + `make
check-equivalence`. Verified against the real checkpoint: `PASS gemma4
equivalence oracle: 1080 layer hashes, generated_token_ids and answer all
match`. The diagnostic JSON serializer, hand-copied and already diverged
between `strata_gemma4_run.cpp` and `strata_deepseek_run.cpp`, moved to
`strata_platform` (`diagnostics_json.{hpp,cpp}`) — the genuinely
byte-identical part only (`hex_u64`, the `layer_hidden_hashes` block); each
binary's own divergent guard on emitting `operation_hashes` was left alone,
since unifying it would be a real behavior change this unit didn't make.
Added `operation_hash_trace_hash` (mirrors the existing layer-hash
aggregate) so routine comparison doesn't require serializing 4320+ records
per arm.

**Carried forward, not actioned:** Gemma 4 has no logit-trace tier
(DSV4's equivalent); wire it per-model as Phase 4's fan-out reaches them.
`apps/strata_gemma4_run.cpp` and its precedents are scheduled for deletion
once `ModelExecutor` + a generic diagnostics control on `RuntimeSession`
make per-model drivers unnecessary.

## Phase 1 — splitting `strata_core` into six layered targets

**Hypothesis.** 55–60 translation units across four documented layers
compiled into one archive; layering was unenforceable because nothing could
violate a boundary that wasn't a link edge. Split into `strata_platform <
strata_device < strata_kernels < strata_engine < strata_models < strata_app`
with strictly downward `target_link_libraries`, plus a static include-level
lint (`scripts/check_layers.py`), file moves deferred to phase 3.

**The 4→30 prediction miss.** A prior review predicted 4 violations; the
real six-target split plus checker found 30 (19 upward-include edges + 11
model-identifier hits). Root cause: `model.hpp` and `model_adapter.hpp` are
generic-sounding but are, in content, the closed-enum/bundled-struct
architecture this refactor exists to fix; correcting their bucket to
`strata_models` pulled five otherwise-unrelated "no model" files along with
it. **The rule that decided almost everything else:** a contract consumed by
two tiers pushes down below both; a genuinely model-specific file pulls up
into `strata_models`, however generic its name looks.

**Cause B — a real link failure, not a lint finding.** `CudaBackend::
flash_attention` (device tier) called `validate_flash_attention_request`,
only implemented in the (then) engine tier — four binaries failed to link
under strict per-target linking; four others linked anyway, by luck, because
they pulled every archive through the eventual `strata_core` alias
regardless of declared order. Fixed (A2): split `attention.*` at the
contract/CPU-reference boundary, contract into `strata_platform`, CPU
reference into `strata_kernels`. Confirmed by `strata-topology-probe` (the
binary that failed) now linking clean. Also found in the split:
`laguna_runtime.cpp` and `glm_runtime.cpp` used `flash_attention_reference_
f32` without ever including `attention.hpp` — picked up by accident through
a transitive chain one shared archive quietly manufactured. That accident is
Phase 1's whole thesis, demonstrated rather than argued for.

**The lint's own two defects, found and fixed while building it:** it only
scanned `.cpp` files and missed a violation living entirely in a paired
`.hpp` (fixed — scans both); its model-identifier check flagged
`glm_int4.{hpp,cpp}` (an INT4 codec kernel named after its first user, not a
GLM concept) as a false positive, fixed by exempting exactly the
owner-equals-target case via an explicit allowlist rather than a broader
carve-out.

**B1 — decided and applied.** `checkpoint.{hpp,cpp}` (shared by three
models via `GlmCheckpointReader`, typed against GLM's own tensor-role
taxonomy) reassigned `strata_platform` → `strata_models`, zero code changes.
**B2/B3 — deferred to phase 4** (orchestrator decision): `placement_model.
cpp`'s six inventory builders need a registry each model populates;
`tokenizer.cpp`'s two model-specific pretokenizers need the same shape one
level down. Both recorded as expiring lint exceptions
(`scripts/layer_exceptions.py`) rather than silently tolerated debt — a
stale exception (no longer matching a real violation) fails the run,
verified by deliberately breaking one and watching it fail, then reverting.

**Cold review remediation (brief 05) — Phase 1 did not actually land on the
first pass; here is what was wrong and what fixed it.**

- **F1 (blocking).** `check_layers.py` checks `#include` edges, never
  symbols. `nm` over all six archives found 24 upward symbol references,
  `strata_engine → strata_models`, all from `placement_model.cpp.o` (the
  same six inventory builders B2 already deferred) — invisible because
  every executable links through the `strata_core` INTERFACE alias, the
  identical "luck, not correctness" mechanism Cause B named one tier down.
  Fixed: `scripts/check_symbols.py`, an `nm`-based link-level checker,
  sharing `layer_exceptions.py`'s same exception list and `CURRENT_PHASE` so
  the two checkers cannot disagree about who owns what. Demonstrated failing
  on the unmodified tree first (`1 total violation(s)`, exact match to the
  reviewer's count) before the fix landed. The B2 exception was widened, not
  narrowed, to state honestly that it forgives this link cycle too.
- **F2 (blocking).** Nothing ran the equivalence oracle — see Phase 0's own
  fix above; noted here because it shared this unit's gate.
- **F3 (blocking).** `make check-layers` was never part of `make check`,
  making the charter's mandatory gate opt-in. One line.
- **F4 (blocking).** `strata_device`'s CMake-variable and
  `target_sources()`-appended files were invisible to the checker — it
  scanned 0 of `strata_device`'s files while its own summary claimed a
  complete scan. Fixed: `parse_targets` resolves `set(VAR file)` (both
  branches of the CUDA/no-CUDA conditional) and `target_sources`, and now
  raises instead of silently continuing on an unresolved source list or an
  undeclared `strata_*` target. Also fixed the model-identifier check's
  narrower-than-documented scope (a self-named file with no bad includes
  could sit in the wrong target undetected) and angle-bracket includes,
  which the original regex missed.
- **F11 (blocking).** `expiry_phase` was read only to print, never
  enforced. Fixed: `layer_exceptions.py` declares `CURRENT_PHASE` (1 today,
  bumping it is the orchestrator's call); both checkers now fail any
  exception whose `expiry_phase <= CURRENT_PHASE`, by name.

**Final gate, this unit.** Every binary links (the four that failed solely
on B1 now build). `make check-layers` and `make check-symbols` both exit 0.
`strata-tests`: 310/311 passed, 1 skipped, line-for-line identical
PASS/SKIP set to the pre-Phase-1 baseline. `make check` — build, both lint
layers, equivalence oracle — passes.

**What phase 4 inherits.** Three recorded, expiry-4, self-checking
exceptions: `placement_model.cpp`'s inventory-builder registry, `tokenizer.
cpp`'s pretokenizer split, and the model-neutral manifest contract that
would let `GlmCheckpointReader` (and `compressed_tensors.hpp`'s
quantization vocabulary) stop depending on GLM's own types for
infrastructure three-to-six models actually share — all one underlying
pattern: generic-purpose code shaped by whichever model needed it first,
wanting one registry designed once against all six, not stretched
piecemeal.

## Phase 2 — the facade defects, and the only phase allowed to change behavior

Scope, per the orchestrator's decisions (protocol amendment): B6 gets an
explicit-unsupported schema (`std::optional`), not hardcoded 0/false; Kimi-K3
does not get device fields bolted onto its config to look uniform with the
other five, because its architecture is host/NUMA-shaped by design;
`-Werror=switch` is scoped to `src/runtime.cpp` only, not a general
warnings-as-errors push (would drag in `deepseek_runtime.cpp:4185`'s
pre-existing `-Wconversion` warning as an unrelated side effect).

**B5 — `RuntimeSession::initialize`'s Inkling arm never copied `devices` or
`vram_cache_fraction`** into `InklingRuntimeConfig`, so `--devices`/
`--vram-cache-fraction` were silently discarded and the config's own
defaults (`{0,1,2}`, `0.85`) applied regardless of what the caller asked
for. Kimi's arm does *not* share this shape, corrected from an earlier
survey: `KimiK3RuntimeConfig` has no `devices`/`vram_cache_fraction` fields
at all — nothing exists to forget, device information reaches it only
through the placement plan. Fixed: the same two-line copy the other four
arms already had, added to Inkling's. **Not independently covered by a
red-first automated test** — `InklingRuntime::initialize` opens the
checkpoint before it ever inspects `devices`, so with `model_directory=
"not-present"` (the only checkpoint-free option) the failure is identical
before and after the fix, for the same "checkpoint missing" reason, in both
cases. This matches the Phase 2 survey's own conclusion: this property is
not observable without a real checkpoint or a `ModelExecutor`-style seam
that doesn't exist yet. Verified instead by direct inspection: the added
lines are byte-identical in shape to the same two lines in the four arms
that already worked.

**B6 — the Inkling and Kimi arms of `generate_chat_stream` never set
`reused_prompt_tokens`/`incremental_kv_continuation`**, leaving them at
`GenerationMetrics`'s hardcoded `0`/`false` defaults — indistinguishable
from "measured, zero reused." Corrected from an earlier survey: this is a
schema defect, not an omission — neither `InklingRunMetrics` nor
`KimiK3RunMetrics` has these fields, and neither runtime implements
incremental prefix reuse, so there was nothing to copy. Fixed: both fields
on `GenerationMetrics` are now `std::optional` (`include/strata/
runtime.hpp`); the four arms that do support reuse assign into them exactly
as before (implicit conversion from the concrete `uint64_t`/`bool`), and the
two that don't simply leave them unset — `nullopt` by construction, no code
change needed in either arm. Downstream JSON emitters
(`openai_protocol.cpp`, `strata_chat.cpp`, `strata_server.cpp`) render
`.value_or(0)`/`.value_or(false)`, preserving the existing wire format
exactly rather than inventing a new one this unit wasn't asked to design.
Test: a pure struct-shape check — default-constructed fields have no value,
assigned fields round-trip — no runtime, no checkpoint, exactly the "cheap
because honest" property the schema choice was picked for.

**B7 — `RuntimeSession::initialize` dispatched five models via `if` blocks
then fell through to `DeepSeekV4Runtime` with no `case`, no `default`, no
error** — a seventh model, or any out-of-range `RuntimeModel` value, would
silently load DeepSeek against the new model's directory and report
success. `-Werror` is not set anywhere in this build (only `-Wall -Wextra
-Wpedantic -Wconversion -Wshadow`), so `-Wswitch` (part of `-Wall`) was only
ever a warning here, never a build failure — confirmed by
`deepseek_runtime.cpp:4185`'s own pre-existing `-Wconversion` warning
persisting across every build this session. **A second, previously
unflagged instance of the identical defect** was found in the same file:
`placement_model()` (`runtime.cpp:31–41`) is already a `switch` with no
`default` and a trailing `return PlacementModel::Gemma4;` standing in for a
seventh model.

Fixed: both converted to `switch` statements with no `default` label,
compiled with `-Werror=switch` scoped to `src/runtime.cpp` only
(`CMakeLists.txt`, `set_source_files_properties`) — a forgotten case for a
real seventh `RuntimeModel` enumerator is now a compile error in this file,
not a silent runtime default, in both places. A closed-enum runtime cast
(`static_cast<RuntimeModel>(99)`, the shape a compile-time check can't
catch) is handled separately: `is_known_runtime_model`, itself a
no-default switch under the same `-Werror=switch`, gates the very top of
`initialize()` before placement or the filesystem are touched, returning
`"unhandled runtime model"` — `placement_model()`'s own trailing case is
therefore unreachable in practice (both call sites outside `runtime.cpp`
derive `RuntimeModel` from a closed ternary that can only produce one of
the six real values) and now `std::abort()`s loudly instead of silently
returning Gemma 4's plan for a value that should be structurally
impossible to reach.

Verified red-first: with only `src/runtime.cpp` reverted to its pre-fix
if-chain (`GenerationMetrics`'s new optional fields are backward-compatible
with the old assignment code, so this isolates the dispatch change
cleanly), the new "unhandled runtime model" test failed for exactly the
predicted wrong reason — the old code fell through to the DeepSeek arm and
produced a checkpoint-not-found error with no mention of an unhandled
model — then passed again after restoring the fix. A test that only checked
`!result.ok()` would have passed on the old code too; checking the error
text is what makes the test catch the actual defect instead of a coincidence.

**New, not one of B5/B6/B7 — recorded, not fixed.** Investigating Kimi's
device-field decision surfaced that `KimiK3RuntimeConfig::placement`, "a
pre-solved placement, borrowed for the duration of initialize" per its own
header comment, and populated by `RuntimeSession::initialize` on every
call, has zero reads anywhere in `kimi_k3_runtime.cpp`. `KimiK3Runtime::
initialize` does not honor the placement plan's device restriction because
it does not consult the placement plan at all. Per the orchestrator's
decision, this is a separate bug, recorded rather than fixed and not
addressed by adding a field.

**Two free conformance tests, over all six `RuntimeModel` values, needing
no checkpoint** (Phase 2 survey's own conclusion: these are the only two
assertable today): a failed `initialize()` leaves the session at
`std::monostate`, so `generate_stream` afterward reports the same "not
initialized" error a session that was never touched gets, for every model;
and the unhandled-model rejection above. Both in `tests/test_runtime.cpp`.

**`make check` gate, this unit:** build, `check-layers`, `check-symbols`,
`check-equivalence`, and the full suite all green — see Phase 3 below for
the file-move gate that follows immediately, per the protocol amendment's
authorization to run remediation → Phase 2 → Phase 3 continuously.

## Recommendations carried forward (not actioned)

1. `apps/strata_chat.cpp` and `apps/strata_server.cpp` derive `RuntimeModel`
   from their own closed ternary chains, duplicating (in a different shape)
   the same "forgot to handle a new model" risk B7 fixed in `runtime.cpp`.
   Out of this unit's scope (the orchestrator's Phase 2 decision named only
   `runtime.cpp`'s two switches); worth the same treatment when a seventh
   model is actually added.
2. F8 (bf16-vs-f32 hash sensitivity) — deferred to Phase 5, see Phase 0.
