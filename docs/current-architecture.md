# Current Strata architecture

This document describes the code that exists today. The intended expert-ticket
wavefront is documented separately in [`architecture.md`](architecture.md) and
must not be treated as implemented behavior.

## Layers

Six static libraries, dependencies pointing strictly downward:

```text
strata_platform   result/types, safetensors and shard I/O, json, numerics,
                  worker pool, NUMA topology, hardware profile, trace,
                  diagnostics, the FlashAttention request contract
strata_device     the CUDA backend, or its error-returning stub
strata_kernels    CPU reference kernels (Q4, INT4 group-128, attention)
strata_engine     placement solver and cache, residency, sampling, chat and
                  session support, the model registry
strata_models     six models, one directory each, plus the shared checkpoint
                  reader, tokenizer and placement inventories
strata_app        RuntimeSession and the OpenAI protocol
```

`strata_core` is an interface alias forwarding to `strata_app`, so every
`target_link_libraries(<binary> PRIVATE strata_core)` keeps working. It links
`strata_models` under `--whole-archive`; see "Model registration" below.

`src/` mirrors this: `src/platform/`, `src/engine/`, `src/app/`, and
`src/models/{dsv4,glm52,gemma4,kimi_k3,laguna,inkling,common}/`. Public headers
stay flat under `include/strata/` — the tier is expressed by the build target
and checked by the two lints, not by header path.

### The layering is enforced, not asserted

Two checks, both run by `make check`:

- `scripts/check_layers.py` reads the `add_library` source lists out of
  `CMakeLists.txt` and fails on an include pointing upward, on a model
  identifier appearing in a no-model target, and on any header it cannot
  assign to a target. That last case matters: an unowned header is not scanned
  at all, which is worse than a violation, and a cold review demonstrated the
  hole by hiding a DeepSeek include inside `model_executor.hpp`.
- `scripts/check_symbols.py` reads real `nm` output from the built archives.
  It exists because the include check cannot see a link edge with no
  corresponding `#include`, and there was one: `strata_engine` referenced
  `strata_models` through `placement_model.cpp`, and it linked only because
  every binary goes through one archive list that let `ld`'s single-pass
  resolution absorb it.

Both report zero violations with **zero exceptions**. Read the symbol score as
"zero upward references expressible as undefined symbols" — a function-pointer
registry and inline code in headers are both invisible to `nm`, and the script
says so at the point it prints the number.

## Model registration

`include/strata/model_executor.hpp` is the seam. `ModelExecutor` has three
virtual members — `initialize`, `generate_chat_stream`, `accepts_images` — and
each model registers itself with a file-scope `ModelRegistrar` in its own
translation unit. `RuntimeSession` is a registry lookup and two virtual calls;
it contains no model names.

The registration carries what applications used to hardcode: the
`--model-type` token, the placement model, whether the model accepts the
DeepSeek-only controls, and the per-model presentation defaults. `strata-chat`
and `strata-server` resolve through `find_model_by_cli_name` and have no
string-to-enum chain.

`--whole-archive` on `strata_models` is load-bearing: a static library drops
any member nothing references, and nothing references a self-registering
translation unit by definition. `strata_app` links ahead of it because `ld`
resolves in one pass. A test asserts all six models are registered, because
without the flag `find_model` returns null for every model and the rest of the
suite still passes.

## What a new model costs

Its own directory under `src/models/`, and **five shared files**:

1. `include/strata/model_executor.hpp` — the `RuntimeModel` enumerator.
2. `include/strata/placement.hpp` — the `PlacementModel` enumerator.
3. `CMakeLists.txt` — the `strata_models` source list.
4. `src/models/common/placement_model.cpp` — three `switch (PlacementModel)`.
5. `src/models/common/tokenizer.cpp` — the pretokenizer dispatch.

Items 4 and 5 are the remaining per-model switches; both are inside
`strata_models`, so they are a code-organisation cost rather than a layering
defect. See [`model-bringup-guide.md`](model-bringup-guide.md) for the
procedure.

## Execution model

The executors are architecture-specific and exact. GLM performs a batched
prefill then token-at-a-time decode. Gemma 4 performs bounded prefill with
whole vision blocks, hybrid local/global attention, and a BF16
local-ring/global-full KV cache. DeepSeek performs bounded layer-major prefill
pages with a multi-row router projection and exact row-ordered causal
transitions. Kimi-K3 batches over a token span throughout. Inkling has an
opt-in paged prefill (`prefill_page_tokens`) that runs attention and its four
short convolutions row-serial — both carry row-ordered state — and batches the
routed MoE expert-major between them; it is bit-identical to its
token-at-a-time path and defaults off pending measurement.

Each executor performs its own exact attention, router, shared-expert,
routed-expert, residual and cache-state transitions. They share lifecycle,
device planning, storage I/O, route tracing, numerical primitives, chat-prompt
preparation and the application facade.

There is no cross-request ticket ring, peer expert RPC, or route-affinity
cohort scheduler. Those are target architecture items and cannot be claimed by
current benchmarks.

## Hardware

`include/strata/hardware_profile.hpp` probes the machine once and caches it:
`MemTotal`, the process's CPU affinity mask (not the machine's core count — a
cgroup or taskset makes those differ), NUMA topology, and the smallest node's
CPU count. `resolve_runtime_devices` turns an empty device list into every
visible device.

The convention: **zero in a config field means "ask the hardware"; a non-zero
value is an explicit operator ceiling and is used verbatim.** A measurement
belongs in the profile; a policy — a fraction, a reserve, a floor — belongs in
the code that applies it. The rank-local ceilings are fractions chosen to
reproduce the figures experiment 0082 validated on the machine it validated
them on.

Not yet portable: `CudaBackend` is concrete with no virtual members, the CPU
fallback is a hand-mirrored stub that every new method must be added to twice,
`CUDA_ARCHITECTURES` is `86 120`, and eleven SM86 gates sit on DeepSeek's
device paths. A different NVIDIA architecture is an edit; a different vendor is
not.

## Equivalence oracle

`make check-equivalence` (also a ctest entry, guarded on the checkpoint being
present) runs Gemma 4 against `tests/fixtures/gemma4/layer-hash-trace.json` —
a per-layer hidden-state hash plus per-operation hashes over a fixed prompt.
The types are model-neutral (`include/strata/diagnostics.hpp`); DeepSeek emits
the same records, and the remaining four models do not yet.

Its limits, stated because a gate nobody understands is worse than none: it
covers **prefill only** — once the device KV path engages, a whole device's
worth of layers returns from one CUDA call with no host-visible boundary, and
DeepSeek shares that blind spot — and `stable_bf16_hash` rounds to BF16 before
hashing, so a difference that moves no value across a rounding boundary is
invisible by construction. That is the class of difference code motion
produces, which is exactly why the file-move phase gated on
renames-not-rewrites rather than on this.

## Placement planning

`strata/placement.hpp` sizes a checkpoint against measured hardware before any
weight is read: an **inventory** (model-specific, from checkpoint headers only,
hardware-independent), a **solver** (a pure function of inventory plus a
hardware probe), and a **plan cache** keyed by checkpoint identity, GPU
identity, context size, device list, VRAM fraction and flags.

`plan_model_placement` in `strata_engine` is a thin dispatcher through a
registered `PlacementPlanner`; the implementation that opens six different
checkpoints lives in `strata_models` and installs itself at static-init. That
inversion is why `strata_engine` names no model symbol.

Placement is **prescriptive for Gemma 4** and **descriptive for GLM and
DeepSeek** — the solver reproduces the round-robin those runtimes already
perform and reports it without changing it, so a planning defect cannot
regress a validated runtime.

## Route traces

Both runtimes emit `strata.route_trace` JSONL version 2, and the simulator
parser consumes the same schema. Every event carries request, phase, token
position, layer, ordered experts and exact coefficients.

## Initialization contract

A concrete runtime may be initialized once successfully. A failed attempt
leaves generation disabled, and a retry starts from a fresh implementation
object. `RuntimeSession` commits an executor only after initialization
succeeds, and rejects an unregistered `RuntimeModel` before any placement work
or checkpoint access.
