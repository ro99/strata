# Experiment 0040 — Engram conditional memory (issue #3)

Status: **rejected before code, no runtime gate run. Held.**

Issue #3 asked for Engram conditional memory
(`https://arxiv.org/html/2601.07372`) integrated into the DeepSeek V4 runtime.
This record states why no manifest field, container format, lookup module, or
graph wiring was written, and what would have to exist before the work
restarts.

## What Engram is (from the paper)

Engram adds K=8 hashed n-gram (bigram/trigram) embedding tables of dimension
`d_mem=1280` per table group, retrieved by a deterministic multiplicative-XOR
hash of a compressed token suffix, concatenated, gated against the current
hidden state (`alpha_t = sigmoid(RMSNorm(h_t)^T RMSNorm(k_t) / sqrt(d))`), and
merged into the residual stream at two specific layers (layers 2 and 15 of a
30-layer, `d=2560` reference model) via a short depthwise causal convolution.
Reported sizes: 5.7B and 18.5B embedding parameters for the 27B/40B reference
models. Critically, the embedding tables are **trained jointly with the
backbone using a 5x learning-rate Adam schedule** — they are canonical learned
weights of a specific checkpoint, not a retrofittable side table.

## Rejection by counting (2026-07-31, before any code)

1. **No Engram tensors exist in the only checkpoint this repo can target.**
   Every tensor name in `models/dsv4f/model.safetensors.index.json` was
   enumerated and pattern-grouped (`layers.N.attn.*`, `layers.N.ffn.*`,
   `layers.N.hc_*`, `mtp.N.*`, embed/head/norm). None match an n-gram hash
   table, a memory embedding, a per-layer gate key/value projection, or a
   depthwise-conv fusion module. `models/dsv4f/config.json` has no `d_mem`,
   no hash-table count, no Engram layer indices, and no vocabulary-compression
   field. DeepSeek-V4-Flash-0731 was never trained with Engram.
2. **The charter's correctness gate cannot be instantiated.** Issue #3
   requires "layer-level and full-model teacher-forcing oracles compare
   enabled and disabled behavior," "greedy generation is token-identical to
   the declared target oracle," and a written target-format contract "before
   implementation." An oracle needs a reference implementation running the
   *same trained weights* Strata would load. No such weights exist for any
   model this repo supports (DeepSeek-V4-Flash-0731, GLM-5.2). Writing a
   lookup module against synthetic or untrained tables would produce a
   result with no target to agree or disagree with — the exact "unrun
   simulation fed by assumed constants" case the charter's research
   discipline section rejects for the same reason.
3. **The measurement plan is not instantiable either.** Issue #3's primary
   metrics are "exact retrieval accuracy" and "end-to-end prefill/decode
   throughput... without increasing steady-state I/O," each against a
   fixed context/RAM budget. Retrieval accuracy needs known static facts the
   trained table actually encodes; throughput needs a real host-DRAM-resident
   table of the reported 5.7-18.5B-parameter scale to measure PCIe prefetch
   overlap against. Absent trained weights, any number reported would be a
   property of a synthetic fixture, not of the model.

A model-format contract can be *written* independent of weights (issue #3
acceptance criterion #1), but the issue asks for the contract, the lookup
module, admission accounting, and graph integration together, gated by an
oracle that does not exist. Per the charter's "do not launder a falsification"
rule, building the module and integration now, then hoping a later stage
supplies the oracle, would be building stage N+1 while stage N (a trained,
Engram-bearing checkpoint) is blocked. The plan is blocked at stage N.

## What would unblock this

- A published Engram-bearing checkpoint (weights + reference inference code)
  that Strata can target, from which teacher-forcing and greedy-generation
  oracles can be derived, the same way DeepSeek-V4-Flash-0731's own bundled
  `models/dsv4f/inference/` reference code grounds every other DeepSeek
  correctness gate in this repo.
- Failing that, a from-scratch Engram training run is out of scope for an
  inference engine project and is not something this repository does.

## Rollback

No runtime code, manifest field, or CLI flag was added. `main` is unchanged
except for this record. If a target checkpoint becomes available, re-open
with the checkpoint's tensor names, `d_mem`, hash-table sizes, and layer
indices substituted for the reference model's, and restart at "write the
target-format contract."
