# Adding a model

A procedure derived from the six existing adapters. Follow it in order; the
ordering is part of the correctness contract.

## What it actually costs

Plumbing is cheap now; the model mathematics is not, and no amount of
architecture changes that. For calibration, the smallest recent bring-up
(Inkling) is **about 3,900 lines** across its manifest, checkpoint reader, ops,
runtime and executor. Budget accordingly: roughly 95% of the effort is exact
router semantics, attention layout and codec decode. The five shared-file edits
below are an afternoon.

## The five shared files

Everything else lives in `src/models/<name>/`. These five are the whole
cross-cutting cost:

| File | Edit |
|---|---|
| `include/strata/engine/model_executor.hpp` | one `RuntimeModel` enumerator |
| `include/strata/engine/placement.hpp` | one `PlacementModel` enumerator |
| `CMakeLists.txt` | your sources in the `strata_models` list |
| `src/models/common/placement_model.cpp` | three `switch (PlacementModel)` arms |
| `src/models/common/tokenizer.cpp` | a pretokenizer arm, if your tokenizer differs |

`-Werror=switch` is scoped to `src/app/runtime.cpp`, so **a missing
`RuntimeModel` case there is a build failure, not a warning.** The other
switches are not covered; the checklist below is what stands in for that.

## Order of work

### 1. Pin the checkpoint before writing code

Record revision, per-file sizes and hashes, indexed tensor count and byte
total. Kimi-K3 is the only model that reads `config.json` at runtime; the other
five carry a `constexpr` execution contract in `model_adapter.hpp`. **Prefer
Kimi's approach** — a contract parsed from the checkpoint cannot drift from it,
and five models currently cannot load a variant of themselves without a
rebuild.

### 2. Write the manifest and checkpoint reader

Assign every tensor an explicit role. Reject unknown, duplicate, missing,
overlapping or semantically invalid tensors. Do not guess tensor names in
runtime code — the offline importer may, the runtime consumes stable roles.

Do not write a new mmap/read-slicing engine. `GlmCheckpointReader` in
`include/strata/models/common/checkpoint.hpp` is shared by GLM, Laguna and Inkling despite
its name; check whether it fits before adding a sixth reader.

### 3. Write ops with fixtures from real target bytes

Operation- and layer-level fixtures built from the actual checkpoint. Shape-
reduced generated fixtures may exercise error paths; they are not a milestone,
and a smaller pretrained model is not a substitute.

### 4. Write the runtime, host path first

Get a correct scalar/host forward pass before any device path. It becomes the
oracle for the device path — this is what `test_inkling_runtime.cpp`'s "device
logits match the host oracle" case is.

### 5. Capture a layer-hash fixture **before** optimising

Not after. This is the single highest-leverage step and the one most often
skipped.

- Emit `LayerHashTraceRecord` / `OperationHashTraceRecord` from
  `include/strata/platform/diagnostics.hpp` behind a default-off config flag.
- Capture a fixture over a short fixed prompt; commit it under
  `tests/fixtures/<model>/`.
- Add a ctest entry guarded on the checkpoint being present.

Every later change is then gated on bit-identity rather than on judgement.

### 6. Write the executor and register

`src/models/<name>/<name>_executor.cpp`: wrap the runtime, translate config and
result, declare a file-scope `ModelRegistrar`. Copy an existing one — GLM's is
the simplest, DeepSeek's shows a flag bundle.

Set `progress_by_default`, `verbose_by_default` and
`flash_attention_by_default` to what your model actually wants. Leave
`accepts_deepseek_controls` false unless you implement them; a request for
rank-local decode that quietly ran your model would report the accepted path
while executing a different one.

### 7. Placement inventory

`build_<name>_inventory` in `placement_model.cpp`, plus the three switch arms.
Descriptive is fine and is what GLM and DeepSeek do — reproduce the placement
your runtime already performs and report it, so a planning defect cannot
regress a working runtime. Prescriptive (Gemma 4) is a later step.

## Rules that are not obvious

**Zero means "ask the hardware."** A config field defaulting to `0`, or an
empty `devices` list, is resolved from `HardwareProfile` at initialize. Do not
write a measurement of your development machine into a struct default. If you
need a policy, express it as a fraction of a probed value and say in a comment
which measurement it reproduces and where.

**`GenerationMetrics` fields are `std::optional` for a reason.** If your runtime
does not implement incremental prefix reuse, leave `reused_prompt_tokens` and
`incremental_kv_continuation` unset. Do not write `0`/`false` — that is
indistinguishable from a measured zero, which is the defect B6 fixed.

**Never add a model identifier to a no-model target.** `strata_platform`,
`strata_device`, `strata_kernels` and `strata_engine` are checked. If you need
something from one of them that does not exist, the question is whether the
thing is a *contract* consumed by two tiers (push it down) or *model-specific*
(pull your file up into `strata_models`). `attention.hpp` was the first;
`dsv4_fp4_expert.cpp` the second.

**Check whether the capability already exists before building it.** Three
mechanisms that looked DeepSeek-private turned out to be generic with a
degenerate caller: `enqueue_moe` takes a `rows` argument that three models
passed as `1U`; `upload` takes an `UploadCompletion` that one caller ever set
to `Deferred`; `numa_bind_range` and the CPU-pinned `HostWorkerPool` had one
caller between them. Follow the call graph, not the file name — reasoning from
model-named *files* to model-private *capability* is an expensive category
error.

**Do not set a performance default you have not screened.** Inkling's paged
prefill shipped defaulting on, and an arithmetic screen afterwards showed it
raises host stream drains from 64 to ~199 per layer for a 1.93x fetch dedup.
It is opt-in now. Ten minutes of arithmetic before the default, not after.

## Before you claim it works

- [ ] `make check` green — includes both lints, the suite, and the oracle.
- [ ] `check-layers` reports **0 violations, 0 exceptions**. An exception is a
      deferral with an expiry phase, not a way to pass.
- [ ] Your layer-hash fixture exists and a ctest entry consumes it.
- [ ] A deliberate perturbation turns that fixture red, at the right layer.
      **An oracle that has never been red is an assumption wearing a test's
      clothes.**
- [ ] Any test needing a GPU skips *loudly* without one. A test that passes
      while never executing its subject is worse than a missing test — see the
      34 tests that silently skipped for want of a `models` symlink.
- [ ] `--devices 0` actually restricts to device 0. Inkling accepted and
      discarded it for months.
- [ ] `--model-type <yours>` is rejected before the checkpoint is opened if the
      registration is missing, rather than loading a different model.
- [ ] A local record under the Git-ignored `experiments/docs/` workspace stating
      what was verified and, explicitly, what was not measured.
