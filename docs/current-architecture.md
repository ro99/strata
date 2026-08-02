# Current Strata architecture

This document describes the code that exists today. The intended expert-ticket
wavefront is documented separately in [`architecture.md`](architecture.md) and
must not be treated as implemented behavior.

## Dependency layers

Strata has four dependency layers:

1. **Core infrastructure** owns result types, Safetensors/shard I/O, numerical
   primitives, CUDA boundaries, worker pools, sampling, and the versioned route
   trace contract.
2. **Model adapters** own immutable pinned execution contracts, tokenizer and
   chat-template behavior, tensor classification, router semantics, and exact
   GLM-5.2, DeepSeek-V4, or Gemma 4 operations.
3. **Execution** owns device admission and placement, runtime initialization,
   request/session state, generation, cache policy, and metrics. Applications
   use `RuntimeSession`; research tools may use the concrete model runtimes for
   model-specific diagnostics.
4. **Applications** parse CLI options and render output. They do not select
   tensor names, model operations, or cache placement.

Dependencies point downward. DeepSeek code does not depend on GLM code for
generic numerical operations, and applications do not branch over concrete
runtime result types for ordinary generation.

## Current execution model

The current executors are architecture-specific and exact. GLM performs a
batched prefill followed by token-at-a-time decode. Gemma 4 performs bounded
prefill with whole vision blocks, hybrid local/global attention, and a BF16
local-ring/global-full KV cache. DeepSeek performs bounded
layer-major prefill pages with a multi-row router projection, exact row-ordered
causal/cache transitions, and token-at-a-time decode. Each executor performs its
own exact attention, router, shared-expert, routed-expert, residual, and
cache-state transitions.

The concrete runtime translation units are model executors, not the future
cross-request scheduler. They share lifecycle, device planning, storage I/O,
route tracing, numerical primitives, and the application facade, while model
mathematics stays isolated behind pinned adapter contracts.

## Current residency and scheduling

Gemma 4 places its dense text graph and vision tower resident across the
capacity-weighted GPU schedule. GLM uses a per-device LRU weight cache with a pinned dense spine and optional
host execution for cold routed experts. DeepSeek stages canonical routed expert
weights in host RAM, pins its dense/shared spine in VRAM, and leases exact
top-k expert triplets during device execution. Device assignment is a shared,
capacity-weighted schedule; see "Placement planning" for how it is sized,
reported, and cached. Its opt-in past-only predictor can queue bounded
host-to-VRAM expert prefetch without changing exact routes or coefficients;
demand cancels queued duplicates and may evict prefetched entries first.

There is no cross-request ticket ring, peer expert RPC, route-affinity cohort
scheduler yet. Those are target architecture items and cannot be claimed by
current benchmarks.

## Placement planning

`strata/placement.hpp` sizes a checkpoint against measured hardware before any
weight is read. It has three pieces:

- An **inventory** — every placeable module with its device bytes, host bytes,
  and the bytes it contributes to one batch-1 decode step. Model-specific, built
  from checkpoint headers only, hardware-independent, and therefore reproducible
  and testable without a GPU.
- A **solver** — a pure function from inventory plus a hardware probe to a plan.
  It assigns layer blocks, applies the tier order device → host → NVMe to
  spillable classes only, and reports the per-resource `W_r` volume of a decode
  step. It measures no bandwidth and produces no duration.
- A **plan cache** — the solved plan as JSON under `~/.cache/strata/plans`,
  keyed by checkpoint identity (shard names, sizes, mtimes), GPU identity,
  context size, device list, VRAM fraction, and flags. Any mismatch is a miss
  and the plan is recomputed; a plan is never reinterpreted across a schema
  version.

Only sparse classes are marked spillable. A densely read class is read on every
step, so moving it out of VRAM to make room for a sparsely read one is negative
under a `max` over resources, and for a dense model larger than aggregate
resident memory the plan reports I/O dependence rather than manufacturing
sparsity.

Placement is **prescriptive for Gemma 4**: the plan chooses contiguous,
byte-balanced layer blocks sized to each device's admitted budget, and
`Gemma4Runtime::initialize` consumes that assignment and those budgets. It is
**descriptive for GLM and DeepSeek**: the solver reproduces the VRAM-weighted
round-robin those runtimes already use and reports the resulting placement
without changing it, so a planning defect cannot regress a validated runtime.
DeepSeek's KV and compressor-state sizing is delegated to
`plan_dsv4_resident_topology` rather than restated.

`RuntimeSession::initialize` resolves and verifies a plan before constructing a
runtime. Verification re-probes free VRAM, because the cached budget was taken
at plan time and another process may have claimed memory since; a prescriptive
plan that no longer fits is an error, and a descriptive one warns.

## Route traces

Both runtimes emit `strata.route_trace` JSONL version 2. The simulator parser
consumes that same schema and retains backward compatibility with the original
numeric text fixture format. Every event carries request, phase, token position,
layer, ordered experts, and exact coefficients.

## Initialization contract

A concrete runtime may be initialized once successfully. A failed attempt leaves
generation disabled, and a retry starts from a fresh implementation object.
`RuntimeSession` commits a concrete runtime only after initialization succeeds.
