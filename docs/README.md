# Documentation map

Strata documentation separates runnable state, target contracts, active plans,
and historical evidence. Read a document according to its class; an old plan or
benchmark must not be used as evidence of current behavior.

## Current state

- `../README.md` — current capabilities, validation limits, commands, and promoted
  machine baselines.
- `current-architecture.md` — implemented dependency and execution boundaries,
  including explicit scheduler gaps and the placement planner behind
  `--dry-run`.
- `deepseek-v4-runtime.md` — current DeepSeek contract plus clearly labeled
  historical bring-up evidence.
- `kimi-k3-runtime.md` — Kimi-K3's pinned contract, cost model, NVMe write
  constraint, chat format, and a per-gate status table that says which gates
  have been measured and which have not.
- Gemma 4's pinned checkpoint, tokenizer, text/vision graph, and public runtime
  contract are described in the root README and `current-architecture.md`.
- `tui.md` — Ratatui build, operator controls, memory bounds, and the versioned
  `strata-chat` frontend protocol.
- `../kernels/cuda/README.md` — native CUDA and non-CUDA stub behavior.

These files must be updated when a change alters user-visible behavior or makes a
previous “not implemented” statement false.

## How to run the work

- `orchestrated-experiment-protocol.md` — operating manual for running
  performance investigations as an orchestrator driving a coding agent in a
  herdr pane: roles, the executor's standing protocol, gate design, measuring
  the noise floor before setting thresholds, verifying a claim before acting on
  it, budgets, and landing rules. Derived from the August 2026 DSV4 prefill
  session. `../CLAUDE.md` remains the research charter and outranks it.

## Contracts and plans

- `architecture.md` — target expert-ticket architecture; it is not an
  implementation claim.
- `model-format.md` — native format contract and the implemented foreign-extent
  path; standalone pack tooling is still planned.
- `model-bringup-guide.md` — **how to add a seventh model.** The procedure, the
  five shared files, and the rules that are not obvious. Start here for a new
  architecture.
- `model-bringup.md` — the historical GLM-5.2 / DeepSeek-V4 bring-up contract:
  pinned revisions and staged correctness gates. Not a procedure for a new
  model; see the guide above.
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
