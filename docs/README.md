# Documentation map

Strata documentation separates runnable state, target contracts, active plans,
and historical evidence. Read a document according to its class; an old plan or
benchmark must not be used as evidence of current behavior.

## Current state

- `../README.md` — current capabilities, validation limits, commands, and promoted
  machine baselines.
- `current-architecture.md` — implemented dependency and execution boundaries,
  including explicit scheduler gaps.
- `deepseek-v4-runtime.md` — current DeepSeek contract plus clearly labeled
  historical bring-up evidence.
- `../kernels/cuda/README.md` — native CUDA and non-CUDA stub behavior.

These files must be updated when a change alters user-visible behavior or makes a
previous “not implemented” statement false.

## Contracts and plans

- `architecture.md` — target expert-ticket architecture; it is not an
  implementation claim.
- `model-format.md` — native format contract and the implemented foreign-extent
  path; standalone pack tooling is still planned.
- `model-bringup.md` — staged correctness contract with per-stage status notes.
- `research-roadmap.md` — phase scope and current high-level status.
- `glm52-throughput-handoffs.md` — historical T0/T1 handoff plan, superseded as a
  live status page by experiment records and the root README.

## Immutable evidence

`experiments/` contains dated records of accepted, rejected, and screened work.
Preserve their original observations and conditions. If later work supersedes a
result, link the newer record or clarify the status in a living document rather
than rewriting the historical measurement.

## Freshness rule

Before changing a status claim, verify it against the build selection, runtime
code, tests, and accepted experiment record. Use precise labels such as
“implemented,” “implemented with promotion gates open,” “partial,” “planned,” or
“historical.” Never use the presence of an interface, stub, fixture, or roadmap
item by itself as evidence that runtime behavior exists.
