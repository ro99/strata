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
   GLM-5.2 or DeepSeek-V4 operations.
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
batched prefill followed by token-at-a-time decode. DeepSeek currently advances
one token through every layer, including during prefill. Each executor performs
its own exact attention, router, shared-expert, routed-expert, residual, and
cache-state transitions.

The concrete runtime translation units are model executors, not the future
cross-request scheduler. They share lifecycle, device planning, storage I/O,
route tracing, numerical primitives, and the application facade, while model
mathematics stays isolated behind pinned adapter contracts.

## Current residency and scheduling

GLM uses a per-device LRU weight cache with a pinned dense spine and optional
host execution for cold routed experts. DeepSeek stages canonical routed expert
weights in host RAM, pins its dense/shared spine in VRAM, and leases exact
top-k expert triplets during device execution. Device assignment is a shared,
capacity-weighted schedule.

There is no cross-request ticket ring, peer expert RPC, route-affinity cohort
scheduler, or predictor-driven runtime prefetch path yet. Those are target
architecture items and cannot be claimed by current benchmarks.

## Route traces

Both runtimes emit `strata.route_trace` JSONL version 2. The simulator parser
consumes that same schema and retains backward compatibility with the original
numeric text fixture format. Every event carries request, phase, token position,
layer, ordered experts, and exact coefficients.

## Initialization contract

A concrete runtime may be initialized once successfully. A failed attempt leaves
generation disabled, and a retry starts from a fresh implementation object.
`RuntimeSession` commits a concrete runtime only after initialization succeeds.
