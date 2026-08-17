# 0120 — Phase 0: building and gating the refactor's equivalence oracle

Status: **closed, gate PASS, with corrections from an independent cold review
(brief 05) applied in place below rather than as a separate erratum.** Every
claimed win in this record is measured on Gemma 4, the model that started
this unit with zero runtime-level test coverage and zero introspection of
any kind. Read the "Correction" callouts throughout before trusting the
sections around them: the causal story for the 34-skipped-tests trap was
wrong and is corrected in place; "localized exactly" is a narrower claim than
it originally read; the three perturbation arms should not be cited as
validating the code that actually shipped, only the fixture re-capture's
identity should be; and the oracle now runs automatically (`make
check-equivalence`, brief 05 F2) rather than existing only as a fixture
nothing referenced.

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

**Correction (brief 05, cold review): "localized exactly" is a weaker claim
for P1 and P2 than the summary above lets it sound, and only P3 tests what
localization actually means.** P1 and P2 inject their fault directly into an
operation that already has its own checkpoint (`attention_local`/`_global`
for P1, `attention_residual` for P2) — a hash recorded at the exact point the
fault happens will, near-definitionally, name that exact point. That is not
nothing (it rules out the hash simply being wrong or not firing at all), but
it does not exercise whether the oracle can find a fault that lands *between*
checkpoints. P3 is the only one of the three that does: it was injected into
what was, at the time, an uninstrumented gap, which is exactly why it missed
by one layer instead of matching exactly. P3 is therefore the one arm that
actually tested localization as a general property, and it is also the one
arm that failed that test. **The gap P3 found is now closed structurally**
(`mlp_residual`, see below) **and that fix has never been tested** — no
fourth perturbation arm was run against it. The claim that stands is narrower
than "the gate proved localization": it is "the gate proved detection on
three arms, and proved exact localization only where a checkpoint already
sat, with the one arm that tested the harder case failing at the time and
its fix unverified since."

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
The fixture was re-captured and the gap is now closed in the sense that a
fault in that position would land on an instrumented checkpoint at layer 55
going forward. This was not re-verified with a fourth perturbation arm; the
fix is structural (the new checkpoint fires at the exact point the old one
was missing) and re-running the full three-arm gate was judged not worth
another three model loads for a fix whose correctness follows from where the
code changed, not from a new measurement. Flagging this as a judgment call,
not a hidden assumption.

**Correction (brief 05, F5 and F6): stop citing the three perturbation arms
as validating the code actually shipped.** `7385e89` changed the
instrumentation twice after the arms were run — the `mlp_residual` checkpoint
just described, *and*, separately, rerouting `analyze_logits`/
`stable_bf16_hash` through the FNV helpers promoted in the same commit. This
record disclosed the first change and not the second, and either way: **the
three arms were never re-run against the code that actually shipped.** The
real evidence that the shipped hash function still computes what it always
did is the fixture re-capture itself, not the arms: the re-captured
`layer_hidden_hashes` has all 1080 entries equal to the pre-`7385e89`
capture, and the aggregate `trace_hash` is bit-for-bit identical
(`9fdbee6c3f216c01`) before and after every instrumentation change since. That
identity is the claim this record can actually stand behind. The P3 gap fix
is untested (previous paragraph); the hash function's continued correctness
after promotion is tested, by the fixture match, not by the arms.

## The 34-silently-skipped-tests trap

Before any of the above, `strata-tests` run fresh in this worktree reported
`277/311 passed, 34 skipped, 0 failed` — a green-looking result. The 34 skips
were every test gated on a real checkpoint being present
(`<model>_checkpoint_present()`), silently no-opping because this worktree had
no `models/` directory at all.

**Correction (brief 05, cold review): the causal story originally recorded
here was wrong, and it was repeated forward into later briefs, so the
correction matters beyond this file.** The original account said a
`.gitignore` pattern mismatch (`/models/`, trailing-slash, directory-only)
caused the skips by leaving an existing symlink untracked. **A `.gitignore`
pattern cannot cause a test to skip by itself** — `std::filesystem::exists`
resolves a path at runtime regardless of what git thinks is tracked; git
never touches whether a symlink target exists on disk. **What is actually
known:** this worktree had no `models/` entry of any kind — no directory, no
symlink — when `strata-tests` was first run in it, which is sufficient on its
own to explain all 34 skips. A `models -> /home/rodrigo/Developer/strata/models`
symlink was then created as the fix, and creating it is what surfaced the
separate, real `.gitignore` bug: the `/models/` pattern does not match a
symlink, so the new symlink stayed untracked-but-not-ignored, flagged by
every `git status`. **What is not known:** whether a symlink existed at some
earlier point and was removed by something (a plausible mechanism is a
destructive git operation such as `clean -fd`, which would sweep up an
untracked, unignored symlink exactly because the same pattern bug left it
unprotected) — no evidence in this session shows that happened, and none of
the work in this worktree ran such a command. The pattern bug is real,
independently verified, and worth having fixed regardless of which of these
is the true history; the causal claim that it *produced* the 34 skips does
not hold, and is retracted.

**This is not a hypothetical risk, independent of the corrected mechanism.**
A green-looking result that measured 34 tests' worth of nothing is the exact
failure mode the charter warns about. Had it gone unnoticed, every later
phase's "gate: same test count, 0 new failures" check would have been
comparing two silently-degraded runs against each other and never caught a
real regression in checkpoint-dependent code.

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

## Scoping facts (brief 05, F7)

Stated plainly rather than left implicit: **five of the six models have no
equivalence fixture at all** — only Gemma 4 does. **Decode has no coverage on
any model**, Gemma 4 included — the fused-decode-path blind spot above means
every fixture and every perturbation arm in this record exercises prefill
only. Both are scoping facts of Phase 0, not gaps introduced by this
remediation unit, and both were already implied by earlier sections of this
record; this section exists so a reader does not have to reconstruct them
from context.

## Open question, not decided here (brief 05, F8)

`stable_bf16_hash` rounds f32 to bf16 before hashing. That means **any
difference that never moves a value across a bf16 rounding boundary is
invisible to this oracle by construction.** That is precisely the class of
difference ordinary code motion produces — inlining, vectorization, FMA
contraction crossing a translation-unit boundary differently than before. The
A2 unit did exactly this kind of motion once already, moving
`flash_attention_reference_f32` into its own translation unit; phase 3 does it
wholesale, to every file in the codebase. So "the trace is byte-identical" and
"the computation is bit-identical" are not the same claim, and the gap between
them is exactly where phase 3 will spend most of its risk. It cuts in both
directions: the same insensitivity means the oracle will not false-alarm on
code motion that is genuinely behavior-preserving at full precision but
happens to round differently at bf16 — which is arguably correct, since bf16
rounding is the declared contract for inference in this codebase, not F32.

Whether to do anything about this is the orchestrator's decision, not
recorded as resolved here. One option raised: an opt-in F32-level hash
alongside the existing bf16 one, so a phase that is supposed to change
*nothing at all* (as opposed to a phase operating inside the declared bf16
contract) can assert the stronger property on demand, while every phase where
bf16 rounding is the actual contract keeps using the weaker, correct-for-that-
contract hash. Whether that is the right shape, or whether there is a cheaper
way to get the same guarantee, is open.

## Cold review remediation (brief 05)

**F2 (blocking) — nothing ran the oracle.** `tests/fixtures/gemma4/
layer-hash-trace.json` was referenced by zero files outside its own README:
no test, no script, no CMake target, no make target. This record's own
"Phase 0's job was to build that oracle" claim was true of building the
*hashing*, not of building anything that *re-runs the comparison*.
Re-verification meant a human typing a `diff` against a 33 GB checkpoint by
hand, and nothing failed if that never happened — the exact thing this
record's own hypothesis names Phase 0 as existing to prevent, unmet by Phase
0 itself.

Fixed: `scripts/check_equivalence.py` runs `strata-gemma4-run
--layer-hash-trace --json` fresh and diffs the result against the committed
fixture (`layer_hidden_hashes` entries, the aggregate `trace_hash`,
`generated_token_ids`, and `answer`), registered as a ctest entry
(`strata-equivalence-gemma4`) with `SKIP_RETURN_CODE` set so an absent
checkpoint reports as skipped, not passed or silently absent — the same
distinction every other checkpoint-gated test in this suite already makes,
extended to the oracle itself. Also runnable standalone as
`make check-equivalence`. Verified by actually running it against the real
checkpoint present on this machine: `PASS gemma4 equivalence oracle: 1080
layer hashes, generated_token_ids and answer all match
tests/fixtures/gemma4/layer-hash-trace.json`.

**F9 (cheap, makes phase 4 easier) — the diagnostic JSON serializer was
hand-copied per binary, and had already diverged.** `strata_gemma4_run.cpp`
and `strata_deepseek_run.cpp` each had their own `hex_u64` and their own copy
of the `layer_hidden_hashes`/`operation_hashes` JSON-printing logic.
Diverged at birth: DeepSeek guards `operation_hashes` emission on
`layer_hash_trace_enabled && !operation_hashes.empty()`; Gemma 4 only checked
the first half, emitting `"operation_hashes":[]` where DeepSeek would omit
the key entirely. Phase 0 promoted the diagnostic *types*
(`include/strata/diagnostics.hpp`) and left the code that renders them
duplicated per binary — phase 4's fan-out would have made that six copies.

Fixed: `hex_u64` and the `layer_hidden_hashes` object-printing logic (which
*was* byte-identical between the two binaries already) moved into
`strata_platform` as `include/strata/diagnostics_json.hpp` +
`src/diagnostics_json.cpp`. The divergent guard condition was deliberately
**not** reconciled — each binary still decides for itself whether to call
the shared `print_operation_hashes_fields`, preserving each one's existing
output exactly. Unifying that guard would be a real behaviour change to at
least one binary's JSON shape, and this unit does not make behaviour
changes.

**F10 (cheap, makes phase 4 easier) — `operation_hashes` had no aggregate
hash.** `layer_hidden_hashes` has had one since promotion
(`layer_hash_trace_hash`); `operation_hashes` did not, so any routine
comparison had to serialize the entire list — 4320 records / 525 KB for an
18-token prompt, and, at the charter's stated 3,565-token operating point,
roughly 100 MB per captured arm if committed to git the way this unit's
fixtures are. Fixed: `operation_hash_trace_hash` added to `DiagnosticTrace`,
folded by `record_operation_hash` in both `gemma4_runtime.cpp` and
`deepseek_runtime.cpp` using the identical fold-position-token-layer-then-hash
pattern `record_layer_hash` already uses, plus the operation name's own bytes
(a distinguishing input the layer-hash aggregate does not need, since it has
no comparable per-record string field). This is a genuine JSON-shape
*addition*, not present for either binary before, so there is no prior
per-caller behaviour to preserve — unlike F9's guard condition.

**Minor, taken:** `Gemma4Runtime::Impl::forward_layers`'s `tokens` parameter
had a `= {}` default with exactly one call site, which already passes real
tokens; a future second call site could have silently recorded `input_token:
0` instead of failing to compile. Default removed — zero behaviour change
today, closes a future footgun for free. `reset_diagnostics()` now also
reserves `operation_hashes` (4x `layer_hashes`' reservation, matching Gemma
4's current four-checkpoints-per-layer wiring, with a comment explaining the
multiplier rather than a bare magic number).

**Minor, not taken, with the one-line reason asked for:** the same reserve
fix was not applied to `DeepSeekV4Runtime`'s `operation_hashes`, because
DeepSeek's operation-hash count is not a fixed multiple of its layer-hash
count the way Gemma 4's is — it varies by prefill/decode path and tiling —
so a safe reserve would need a conservative overestimate rather than an exact
multiplier, which is a bigger judgment call than the one-line fix Gemma 4
got.

## Recommendations carried forward (not actioned in this unit)

1. Gemma 4 has no logit-trace tier (DSV4's `Dsv4LogitAnalysis` equivalent).
   Only wired for the four remaining models when phase 4's fan-out reaches
   them, per the orchestrator's revised sequencing (oracle depth scales with
   phase risk; phase 1 needs tier 0 only).
2. `apps/strata_gemma4_run.cpp` (and its precedents `strata-run`,
   `strata-deepseek-run`) are scheduled for deletion in phase 4, once
   `ModelExecutor` plus a generic diagnostics control on `RuntimeSession`
   make per-model drivers unnecessary.
