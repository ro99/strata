# 0121 — Phase 1: splitting `strata_core`, and what a 4 → 30 prediction miss means

Status: **A1-A5 applied and gated green. B1-B3 proposed, not applied — awaiting decision.**

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
moved out. Two of the three runtimes (`laguna_runtime.cpp`, `glm_runtime.cpp`)
never included `attention.hpp` directly at all — they got the declaration only
by accident, through `checkpoint.hpp` → `cuda_backend.hpp` → `attention.hpp`
(the exact B1 chain). All five files now include
`strata/attention_reference.hpp` directly instead of depending on that
accident.

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

## Gate for A1-A5

`strata-tests`: **310/311 passed, 1 skipped, 0 failed** — line-for-line
identical `PASS`/`SKIP` set to the pre-Phase-1 baseline (`diff` empty).

`make check-layers`: reports exactly three violation groups and nothing else —
B1 (`checkpoint.hpp`, 2 upward includes + 1 identifier hit), B2
(`placement_model.cpp`, 7 upward includes + 6 identifier hits), B3
(`tokenizer.cpp`, 2 upward includes + 2 identifier hits) — **plus one item not
originally on this list**: `compressed_tensors.hpp` (1 upward include + 1
identifier hit), the corrected-classification finding above. 21 total lines
across these four files, all expected, none silently absorbed.

**Binaries:** links clean for `strata-tests`, `strata-gemma4-run`,
`strata-deepseek-run`, `strata-topology-probe`, and every CPU-only probe. Does
**not** link for `strata-chat`, `strata-run`, `strata-server`, and
`strata-host-expert-probe` — all four fail solely on B1
(`checkpoint.cpp`'s undefined references to
`build_glm52_w4a16_index_manifest`/`validate_glm52_w4a16_checkpoint`), which is
explicitly not approved for this unit. The brief's stated gate text ("all
binaries build") and its explicit instruction that "the B1-B3 files stay
untouched" cannot both be literally satisfied simultaneously; this record
resolves that in favor of the untouched-files instruction, since it was the
more specific and more recent one, and reports the resulting four broken
binaries plainly rather than silently declaring the gate met.

## B1-B3: proposed, not applied

### B1 — `include/strata/checkpoint.hpp`

Independently verified consumers: `glm_runtime.hpp`, `laguna_checkpoint.hpp`,
`inkling_checkpoint.hpp`, `laguna_runtime.hpp`, `inkling_runtime.hpp`,
`placement_model.cpp`, `checkpoint.cpp`, and one probe. **Three models share
`GlmCheckpointReader`** — it is shared infrastructure with a GLM-shaped name,
not a GLM-only file, and renaming it would be wrong.

Reading the file: `GlmCheckpointReader` is a generic mmap/read-slicing engine
(`open`, `read`, `read_slice`, `view`, `read_f32`, ...) whose public surface is
typed against `GlmIndexManifest`/`GlmManifestTensor`
(`include/strata/glm_manifest.hpp`) — and those two types are not a generic
tensor-index shape; `GlmManifestTensor` carries `GlmTensorRole` and
`GlmTensorComponent`, GLM's own tensor-classification taxonomy, baked into the
type the reader is templated against by construction, not by parameter. The
one function needing `cuda_backend.hpp`, `load_glm_cuda_linear`, is a thin
wrapper that reads through the same reader and uploads to a `CudaBackend` —
inseparable from the reader's own coupling, not an independent problem.

**Cheapest correct option:** reassign the whole file — `checkpoint.hpp` +
`checkpoint.cpp`, unchanged — from `strata_platform` to `strata_models` in
`CMakeLists.txt`. Zero code changes. Every current consumer already lives in
`strata_models` except `placement_model.cpp` (B2, already broken, not made
worse) and the probe (not part of the layered targets). Cost: it accepts, for
now, that `GlmCheckpointReader` is models-tier infrastructure rather than a
core platform primitive — architecturally honest given what it actually
returns, but it means nothing in `strata_platform`/`device`/`kernels`/`engine`
can use it without becoming a models-tier consumer. It does not foreclose a
later generalization; it just stops pretending one already happened.

**Alternative (larger, not proposed for this unit):** extract a genuinely
model-neutral `ManifestTensor`/`IndexManifest` contract (name, shard, dtype,
shape, offset, bytes — the fields that are not GLM-specific) into
`strata_platform`, and have `GlmManifestTensor` extend or wrap it, with
`GlmCheckpointReader` parameterized over the generic base for its storage/read
logic while keeping GLM's role/component classification as a thin layer above.
This is the change that would make three models sharing one manifest type
stop being an accident of convenience — but it changes the reader's public API
and needs at least a look at every one of its three current model call sites,
which is exactly the design commitment flagged as "not today."

A third option — split `load_glm_cuda_linear` out into its own file, leaving
`GlmCheckpointReader` behind — was considered and rejected: it removes only
the `cuda_backend.hpp` half of the coupling. `glm_manifest.hpp` remains
regardless, so the reader still cannot stay below `strata_models`, and the
split adds a file and a decision for no lower final tier. Not listed as the
alternative above because it is strictly dominated by it.

### B2 — `src/placement_model.cpp`

1,174 lines. Six `build_<model>_inventory` functions (lines 100-949, about 72%
of the file) each convert one model's checkpoint reader into the already
model-neutral `PlacementInventory` type; `build_inventory` (line 949)
dispatches on architecture kind to the right one; `plan_model_placement`
(line 1050) is the actual solver entry point. The review's finding matches
what's on the page: the solver is already close to "a pure function of
inventory plus hardware" — `PlacementInventory` is the right shape — but the
six builders are inlined in an engine-tier file instead of living with each
model.

**Recommendation: defer to phase 4, not a phase-1 move.** Moving the six
builders out cleanly needs more than relocation — `build_inventory`'s
dispatch would need to become a registry (each model registers its own
builder callback) so `placement_model.cpp` never names a model directly, and
that registry's shape is a design decision, not a mechanical split. Phase 4 is
explicitly when "each adopter is already inside its own model's code" for the
oracle fan-out; relocating that model's own inventory builder in the same pass
is one visit to the file instead of two, and the registry shape only needs to
be decided once, with all six models' actual needs in view, rather than
designed now for one and stretched to fit five more later. **Expiry: phase 4.**
If the answer is "defer," the lint should carry `placement_model.cpp`'s seven
includes as a recorded exception with that expiry — not implemented in this
unit, since deciding to defer is not this unit's call to make.

### B3 — `src/tokenizer.cpp`

1,960 lines. `inkling_unicode.hpp` and `laguna_unicode.hpp` are Unicode-15.0
category tables generated specifically for each model's own pretokenizer
regex classes, with the model-specific pretokenize functions
(`pretokenize_laguna_chunk`, `laguna_letter`, etc.) inlined directly into the
shared `ModelTokenizer` dispatch. Structurally the same shape as B2: a
generic-purpose engine-tier class with two models' specific logic branched
inside it instead of living with those models.

**Recommendation: defer to phase 4, same reasoning as B2, same expiry.**
Pretokenization is exactly "inside the model's own code" territory once each
model's fan-out pass happens; deciding it now would mean designing the
per-model-pretokenizer dispatch mechanism twice (once ad hoc for two models,
again properly for six) rather than once, with all six in view.

## What's committed

`c1a1d73` (Phase 1 split, non-linking, superseded by the next commit's fixes),
`3805dc8` (A1-A4 applied, six-target split links clean except for B1's four
binaries, gate green), plus this record. `scripts/check_layers.py`'s A5 fix
was committed separately. None of `checkpoint.hpp`, `placement_model.cpp`,
`tokenizer.cpp`, `model.hpp`, or `model_adapter.hpp` have been edited.

## Recommendation for the next unit

1. Decide B1: reassign `checkpoint.hpp`/`checkpoint.cpp` to `strata_models`
   (cheapest), or commit to generalizing the manifest contract now.
2. Decide B2/B3: apply the phase-4-deferred recorded exception, or pull them
   forward.
3. `compressed_tensors.hpp`'s reintroduced violation is a new, small
   corollary of B1's own question (contract vocabulary bundled inside
   `model.hpp` alongside genuinely model-specific specs) — worth folding into
   whichever `model.hpp`-splitting decision B1's alternative eventually
   triggers, not a separate unit on its own.
