# Documentation map

This directory contains Strata product documentation: supported behavior,
architecture, formats, interfaces, and operator runbooks. Research records,
handovers, campaign trackers, and issue notes belong in the Git-ignored
`experiments/docs/` workspace.

## Current state

- `../README.md` — current capabilities, validation limits, commands, and promoted
  machine baselines.
- `models/` — operator runbooks with copy-paste build, chat, server, and
  benchmark commands tied to measured operating points. Start with
  `models/laguna.md`.
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
- `../kernels/cuda/README.md` — native CUDA and non-CUDA stub behavior.

These files must be updated when a change alters user-visible behavior or makes
a previous “not implemented” statement false.

## Contracts and interfaces

- `architecture.md` — target expert-ticket architecture; it is not an
  implementation claim.
- `model-format.md` — native format contract and the implemented foreign-extent
  path; standalone pack tooling is still planned.
- `cli.md` — every command-line flag and the dry-run placement planner.
- `sampling.md` — sampler options, their exact semantics, reproducibility.
- `server.md` — `strata-server` and the OpenAI-compatible API.
- `model-bringup-guide.md` — **how to add a seventh model.** The procedure, the
  five shared files, and the rules that are not obvious. Start here for a new
  architecture.

## Freshness rule

Before changing a status claim, verify it against the build selection, runtime
code, tests, and the applicable measurements. Use precise labels such as
“implemented,” “implemented with promotion gates open,” “partial,” “planned,” or
“historical.” Never use the presence of an interface, stub, or fixture by itself
as evidence that runtime behavior exists.
