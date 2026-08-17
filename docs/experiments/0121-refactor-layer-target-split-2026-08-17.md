# 0121 — Phase 1: splitting `strata_core`, and what a 4 → 30 prediction miss means

Status: **CLOSED. Phase 1 gate green: every binary links, `make check-layers` exits clean, `make check`
(build included) passes for the first time this phase. B1 applied. B2 and B3 deferred to phase 4 with
recorded, self-checking lint exceptions.**

## Hypothesis

`CMakeLists.txt` compiled 55-60 translation units from all four documented layers
(core / model adapters / execution / applications) into one `strata_core` archive.
Layering was unenforceable because nothing could violate a boundary that did not
exist as a link edge. Splitting into targets with declared downward dependencies,
plus a static lint check, makes the boundary real and checkable without moving a
single file — file moves are phase 3.

## The 4 → 30 prediction miss, and its one root cause

A code review predicted exactly four layering violations. All four were confirmed
exactly as described. Building the actual six-target split and writing a static
checker (`scripts/check_layers.py`) found **30 total hits** (19 upward-include
edges + 11 model-identifier hits, with overlap where one bad include trips both
checks) across **11 distinct header-level relationships**.

**Single root cause for most of the gap:** the target-bucket table proposed for
this unit built from filenames, and two files — `include/strata/model.hpp` and
`include/strata/model_adapter.hpp` — have deeply misleading names. Both are
generic-sounding; both are, in content, the closed-enum/bundled-struct
architecture the mission context names as the problem. `model.hpp` (280 lines)
holds `DeepSeekV4Spec`, `Gemma4Spec`, `LagunaSpec`, `InklingSpec`,
`GlmMoeDsaSpec`, the aggregating `ModelSpec`, and all six
`*_spec()`/`validate_*` functions. `model_adapter.hpp` (373 lines) holds all six
models' `ExecutionContract` structs and their per-model layer-classification
helpers. Correcting their bucket to `strata_models` — not filed anywhere in the
original proposal — is the one change that took the violation count from 4 to
30, because five otherwise-unrelated "no model" files (`worker_pool.hpp`,
`compressed_tensors.hpp`, `cuda_backend.hpp`, `glm_int4.hpp`, `attention.hpp`)
each `#include "strata/model.hpp"` and, on inspection, needed nothing from it but
`ValidationResult` — itself only present because `model.hpp` re-exports
`result.hpp`.

**The lesson generalizes:** a file's name is not evidence of its content, and
this applies as much to the *enforcement* as to the target table it enforces —
see "The lint's own two defects" below for a checker that made exactly this
mistake in the opposite direction.

## The rule that decided almost everything else

> If the header is a contract consumed by two tiers, push the contract down
> below both. If the header is genuinely model-specific, pull the *file* up
> into `strata_models`.

Two violations looked identical on the lint's output line — a lower tier
including a model-named header — and needed opposite fixes:

- `attention.hpp` (Cause B) was a **contract**: a model-neutral request struct
  the device backend and the CPU reference both consume. It moved down.
- `deepseek_host_expert.hpp` (violation #1) is **model-specific**: read in
  full, it declares `Dsv4HostExpertWeights`, `dsv4_tiled_expert_*`, FP4 E2M1
  with UE8M0 scales, reproducing `deepseek_fp4_gate_up_kernel` term for term.
  Nothing generic. The *kernel that used it*, `kernels/cpu/dsv4_fp4_expert.cpp`,
  moved up (A3) instead.

The same test, applied to the three items still open (B1-B3), is what the next
brief needs to answer.

## Cause B: a real link failure, not a lint finding

`CudaBackend::flash_attention` (`kernels/cuda/backend.cu`, device tier) calls
`validate_flash_attention_request`, which only `attention.cpp` (then engine
tier) implemented. Building the six-target split with strict downward
`target_link_libraries` **failed to link** four binaries —
`strata-chat`, `strata-deepseek-run`, `strata-server`, `strata-topology-probe`
— with `undefined reference to 'strata::validate_flash_attention_request'`.
`strata-tests`, `strata-run`, `strata-gemma4-run`, and the CPU-only probes still
linked, because they pulled the full transitive closure through the `strata_core`
alias regardless of declared per-target order, and GNU `ld`'s single-pass
archive resolution happened to find every symbol on those specific link lines.
That was luck, not correctness — the dependency was always there, just
invisible while everything shared one archive.

**Fix (A2):** split `attention.*` at exactly the boundary independently
verified before this unit started (`attention.cpp` at line 259; the contract
above, the CPU reference below). Contract
(`FlashAttentionSegment`, `FlashAttentionNumerics`,
`should_dispatch_flash_attention_cuda`, `FlashAttentionRequest`,
`FlashAttentionShape`, `validate_flash_attention_request`, and their private
helpers `multiply`/`add`) stayed in `include/strata/attention.hpp` +
`src/attention.cpp`, reassigned to `strata_platform`. One of those "private"
helpers, `segment_shape`/`SegmentShape`, had to become a real declared function
instead of an anonymous-namespace one, because the CPU reference now calls it
across a translation-unit boundary it didn't previously cross. The CPU
reference (`flash_attention_reference_f32`) moved to a new
`include/strata/attention_reference.hpp` + `kernels/cpu/attention_reference.cpp`,
`strata_kernels`.

**Confirmed fixed, not just plausible:** after the split, `strata-topology-probe`
— the exact binary that failed on this symbol — links clean.

**What this fixes beyond the link error:** a prior review found `attention.hpp`
was the one genuinely good cross-model seam in the codebase, and that only
Gemma 4 sat on it. Putting the contract in `strata_platform` makes it
structurally the shared seam instead of a well-designed file nobody was
obliged to use.

## Corrections made while applying A1-A4, found by the compiler

Two, both disclosed rather than quietly folded into the "applied cleanly"
story:

1. **A stray leftover `}  // namespace`** in `attention.cpp`, orphaned when
   `segment_shape` moved out of the anonymous namespace, was closing
   `namespace strata` early and pushing `validate_flash_attention_request` and
   everything after it outside the namespace. `-Wall` did not catch it;
   `gcc`'s "did you mean `strata::FlashAttentionShape`" diagnostic did.
2. **`compressed_tensors.hpp` was not actually Cause A.** The first-pass
   symbol check (brief-01's report) grepped a partial list of `model.hpp`'s
   declared names and found only `ValidationResult`, so it went in the "safe
   to swap" batch. The compiler disagreed: it also needs
   `QuantizedWeightSpec` and `QuantizationGranularity`, two names the partial
   grep never included. Re-run against the complete symbol list extracted
   from `model.hpp` (27 names, every `struct`/`enum class`/`[[nodiscard]]`
   declaration in the file) confirmed the other four Cause-A files really are
   clean, and reverted `compressed_tensors.hpp` to `#include "strata/model.hpp"`,
   left in `strata_platform` as before. **This reintroduces a violation the
   lint now reports** — new information, not a fix applied. `bits`,
   `granularity`, and `group_size` are pure quantization-format vocabulary
   with no model name attached, exactly the shape the contract-versus-specific
   rule calls "push down" — but that is a `model.hpp`-splitting decision, out
   of scope for this unit, and not decided here.

## Compile-time ripple from the attention split

Three model runtimes (`gemma4_runtime.cpp`, `glm_runtime.cpp`,
`laguna_runtime.cpp`) and two tests (`test_attention.cpp`,
`test_cuda_backend.cpp`) called `flash_attention_reference_f32` through a
transitive include `attention.hpp` no longer carries once the CPU reference
moved out. All five files now include `strata/attention_reference.hpp`
directly instead of depending on the accident described next.

**The finding worth naming plainly:** `laguna_runtime.cpp` and
`glm_runtime.cpp` **never included `attention.hpp` at all.** They picked up
`flash_attention_reference_f32`'s declaration purely by accident, through
`checkpoint.hpp` → `cuda_backend.hpp` → `attention.hpp` — the exact chain B1
is about. Two files that use a function got away with never declaring that
they need it, because one shared archive quietly manufactured the dependency
for them. That is not a hypothetical risk this phase is guarding against; it
is a dependency that already existed, undeclared, in the codebase this record
started from. It is the entire thesis of phase 1, demonstrated by accident
rather than argued for.

## The lint's own two defects

1. **The `.hpp` blind spot.** The first version of `scripts/check_layers.py`
   only scanned each target's `.cpp` translation units, and missed violation
   #4 (`checkpoint.hpp` including `cuda_backend.hpp` and `glm_manifest.hpp`)
   entirely on its first run, because `checkpoint.cpp` only includes its own
   header — the bad includes live inside `checkpoint.hpp`, which the script
   never opened. Fixed by scanning each target's paired public headers
   (same-basename `include/strata/<stem>.hpp`) alongside its `.cpp` files. A
   tool that has silently missed something once is more dangerous than no
   tool; the miss is recorded here alongside the fix, not just the fix.
2. **The identifier check's false-positive mode.** As specified ("no file in
   a no-model target may include a header whose path contains a model
   identifier"), the check flagged `kernels/cpu/glm_int4.cpp includes
   glm_int4.hpp` — a model identifier in a no-model target, by the letter of
   the rule. But `glm_int4.*` is deliberately kept in `strata_kernels`: it is
   an INT4 group-128 *codec* kernel named after its first user, not a GLM
   concept (review finding C6; the rename belongs with the phase-8 codec
   work, noted here, not done now). The direction check never flagged this
   edge, because owner equals target. Fixed (A5) by making the identifier
   check skip exactly that case: a model identifier is only a violation when
   the *owning* target is not the target doing the including, reusing the
   direction check's own ownership resolution rather than re-deriving it, so
   the two checks cannot disagree about who owns what.

## Intermediate gate (A1-A5, brief 02)

`strata-tests`: **310/311 passed, 1 skipped, 0 failed** — line-for-line
identical `PASS`/`SKIP` set to the pre-Phase-1 baseline (`diff` empty).

`make check-layers`: reported exactly three violation groups and nothing
else — B1 (`checkpoint.hpp`, 2 upward includes + 1 identifier hit), B2
(`placement_model.cpp`, 7 upward includes + 6 identifier hits), B3
(`tokenizer.cpp`, 2 upward includes + 2 identifier hits) — **plus one item not
originally on the review's list**: `compressed_tensors.hpp` (1 upward include
+ 1 identifier hit), the corrected-classification finding above. 21 total
lines across these four files, all expected, none silently absorbed.

**Binaries:** linked clean for `strata-tests`, `strata-gemma4-run`,
`strata-deepseek-run`, `strata-topology-probe`, and every CPU-only probe. Did
**not** link for `strata-chat`, `strata-run`, `strata-server`, and
`strata-host-expert-probe` — all four failed solely on B1
(`checkpoint.cpp`'s undefined references to
`build_glm52_w4a16_index_manifest`/`validate_glm52_w4a16_checkpoint`), not
approved for that unit. The brief's stated gate text ("all binaries build")
and its explicit instruction that "the B1-B3 files stay untouched" could not
both be literally satisfied simultaneously; that record resolved the conflict
in favor of the untouched-files instruction and reported the four broken
binaries plainly. The orchestrator confirmed this was the correct resolution
and that the self-contradiction was theirs, not an error to charge against
the executor.

## B1 — decided and applied: `checkpoint.{hpp,cpp}` → `strata_models`

Independently verified consumers, twice, by both sides: `glm_runtime.hpp`,
`laguna_checkpoint.hpp`, `inkling_checkpoint.hpp`, `laguna_runtime.hpp`,
`inkling_runtime.hpp`, `placement_model.cpp`, `checkpoint.cpp`, and one probe.
**Three models share `GlmCheckpointReader`** — shared infrastructure with a
GLM-shaped name, not a GLM-only file; renaming it would have been wrong.

`GlmCheckpointReader`'s public surface is typed against
`GlmIndexManifest`/`GlmManifestTensor`, and those are not a generic
tensor-index shape: `GlmManifestTensor` carries `GlmTensorRole` and
`GlmTensorComponent`, GLM's own tensor-classification taxonomy, baked into the
type the reader is built against by construction. Calling the reader a
platform primitive was a claim the code did not support.

**Applied: reassigned `checkpoint.hpp` + `checkpoint.cpp`, unchanged, from
`strata_platform` to `strata_models`.** Zero code changes. Every current
consumer was already `strata_models` except `placement_model.cpp` (already
covered by B2's own violations, not made worse) and the probe (outside the
layered targets). This accepts, for now, that `GlmCheckpointReader` is
models-tier infrastructure rather than a core platform primitive — honest
given what it actually returns.

A third option considered and rejected in the prior unit — split
`load_glm_cuda_linear` out, leave the reader behind — stays rejected:
`glm_manifest.hpp` alone still forces `strata_models` regardless of what
happens to the CUDA-upload half.

**The generalization alternative is the right end state, and is not this
unit.** Extracting a genuinely model-neutral `ManifestTensor`/`IndexManifest`
contract (name, shard, dtype, shape, offset, bytes) into `strata_platform`,
with `GlmManifestTensor` extending or wrapping it and `GlmCheckpointReader`
parameterized over the generic base, is what would make three models sharing
one manifest type stop being an accident of convenience. It changes the
reader's public API and needs a look at all three current model call sites —
recorded as a lint exception with **expiry: phase 4**, the same shape and the
same reason as B2 and B3: it wants to happen once, with all six models'
actual manifest shapes in view, not designed against three today and
stretched to fit three more later.

## B2 and B3 — deferred to phase 4, with recorded lint exceptions

The deferral decision was made by the orchestrator, not the executor — the
prior unit's report was explicit that choosing to defer was not its call, only
that deferring was the recommendation if someone else made the call. It was
made: **both defer to phase 4.** Extracting `placement_model.cpp`'s six
inventory builders needs `build_inventory`'s dispatch to become a registry
each model populates; `tokenizer.cpp`'s two model-specific pretokenizers need
the same shape of decision one level down. Designing that registry once,
against all six models' actual needs at phase 4's fan-out, beats designing it
now against two and stretching it to fit four more later.

**`scripts/layer_exceptions.py`** records three entries, each naming what it
defers, why, and its expiry phase in the file itself:

- `quantization-contract-in-model-hpp` — `compressed_tensors.hpp`'s need for
  `QuantizedWeightSpec`/`QuantizationGranularity`, folded in as the same
  question as B1's generalization alternative rather than left as an
  unexplained fourth item, since both stem from `model.hpp` bundling generic
  quantization vocabulary alongside genuinely model-specific specs.
- `placement-model-inventory-registry` — B2, 8 matched `(file, header)` pairs
  (7 named in the original review plus `checkpoint.hpp`, newly upward once B1
  moved it — the same file, the same deferred fix, one more line covered by
  the same exception, not a new item).
- `tokenizer-pretokenizer-split` — B3, 2 matched pairs.

All three: **expiry phase 4.**

**The exception mechanism has the two required properties, both verified by
actually breaking one and watching it fail:**

1. A listed `(file, header)` pair that no longer corresponds to a real
   violation fails the run. Verified: appended a fourth exception entry
   naming a header nothing includes, ran `check_layers.py`, got exit 1 and
   `== STALE EXCEPTIONS (1) == [fake-stale-test] no longer matches a real
   violation: src/tokenizer.cpp includes nonexistent_header.hpp`, then
   reverted the test entry. A stale exception cannot silently keep a
   boundary open past whatever fixed it.
2. `check_layers.py` prints every exception it applied, by name, with the
   exact violations each one covered, on every run — not just on request, not
   summarized away. See the "exceptions applied" section of any
   `make check-layers` run.

## Final gate for Phase 1

- **Every binary links.** `strata-chat`, `strata-run`, `strata-server`, and
  `strata-host-expert-probe` — the four that failed solely on B1 — now build
  and link, confirmed by a from-scratch rebuild.
- **`make check-layers` exits 0**, printing all three exceptions with 11/11
  matches consumed and 0 stale.
- **`strata-tests`: 310/311 passed, 1 skipped** — line-for-line identical
  `PASS`/`SKIP` set to the brief-00b baseline, re-verified on this unit's
  final state.
- **`make check` — the whole thing, build included — passes.** First time in
  this phase that command has exited 0.

## What's committed

`c1a1d73` (initial split, non-linking), `3805dc8` (A1-A4, six-target split
links except for B1's four binaries), `46d14b8` (A5, identifier check
target-aware), `9349366` (this record, first draft), `f2c0f8b` (B1
reassignment + the three recorded exceptions + `check_layers.py`'s exception
consumer), plus this record's final update. `placement_model.cpp` and
`tokenizer.cpp` were never edited, per every unit's scope fence in this
phase — both are deferred, not fixed, and the record says so in the code
itself via the exceptions file, not just here.

## Phase 1 is closed. What phase 4 inherits.

Three recorded exceptions, each with expiry phase 4, each naming a registry
or contract that needs designing once against all six models rather than
piecemeal: `placement_model.cpp`'s inventory-builder registry,
`tokenizer.cpp`'s per-model pretokenizer split, and the model-neutral
manifest contract that would let `GlmCheckpointReader` (and
`compressed_tensors.hpp`'s quantization vocabulary) stop depending on GLM's
own types for infrastructure three-to-six models actually share. All three
are the same underlying pattern — generic-purpose code that took on one
model's shape because it was built for that model first — and all three want
the same fix: design the shared shape once, when every model's actual need is
in view, not incrementally against whichever model happened to need it first.

Per this unit's own instruction: **phase 2 does not start here.** A
cold-context review sits between this green phase and any landing to `main`;
that review, and any landing that follows it, is out of scope for the agent
that did the work.
