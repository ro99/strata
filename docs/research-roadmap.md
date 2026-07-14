# Research roadmap

Each phase answers one bounded question. A failed mechanism is recorded and
removed from runtime code; a larger model is not used to compensate for a
missing correctness oracle.

## Phase 0 — foundation (complete)

- C/C++ build with no third-party runtime.
- Precision-floor and DeepSeek semantic validation.
- Signed-int4 reference matvec.
- Sequential trace parser, past-only predictor, residency simulator, and
  bounded-cold-read contract.

Gate: sanitizer tests and deterministic simulator output.

## Phase 1 — immutable model pack (next)

- Incremental Safetensors reader with bounded memory.
- Manifest and fixed tensor directory.
- Q4_K64 reference codec and deterministic quantizer.
- Resumable, hash-verified, atomic `strata-pack` conversion.
- Generated corruption and interruption fixtures.

Gate: byte-exact layout tests, bounded reconstruction error, restart without
rewriting verified extents, and peak RSS independent of checkpoint size.

## Phase 2 — dense CPU oracle

- RMSNorm, RoPE, GQA/MHA, dense matmul, SwiGLU, embeddings, and output head.
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` adapter and immutable source-precision
  references.
- Per-tensor-role Q4_K64 quality evaluation.
- Tokenizer boundary and deterministic generation harness.

Gate: teacher forcing and greedy generation pass declared tolerances, with
source and packed model identity recorded.

## Phase 3 — conventional MoE oracle

- `allenai/OLMoE-1B-7B-0125` adapter.
- Exact router, top-k aggregation, and expert placement units.
- Row-synchronous CPU reference followed by expert-ticket execution.
- Sequential real-route capture and affinity analysis.

Gate: router IDs, weights, logits, and generation pass the oracle; wavefront and
row-synchronous paths perform identical work.

## Phase 4 — DeepSeek oracle

- `deepseek-ai/DeepSeek-V2-Lite` adapter.
- MLA compressed KV, dense prefix, shared experts, fine-grained routed experts,
  group-limited routing, and correction-bias selection semantics.
- Shared/routed overlap and DeepSeek route capture.

Gate: layerwise router and coefficient checks, teacher forcing, greedy
generation, and manifest-derived KV/residency bounds all pass.

## Phase 5 — evidence-driven residency

- Analyze conventional and DeepSeek traces at equal RAM/VRAM budgets.
- Compare LRU, LFU, leases, route-affinity cohorts, and uncertainty-budgeted
  prefetch.
- Measure useful/wasted read bytes, working-set size, H2D amplification, and
  peer fan-out.
- Define admission from measured hot sets rather than model-name heuristics.

Gate: a new policy reduces physical NVMe bytes by at least 2x versus the best
conventional policy at equal capacity, without increasing total transfer cost or
changing model work.

## Phase 6 — persistent CPU execution

- NUMA-local expert arenas and persistent worker queues.
- Expert-major low-row Q4_K64 kernels for AVX2 and AVX-512.
- Shared-expert/routed overlap.
- Bounded wavefront and continuous multi-request batching.

Gate: measurable throughput gain over the row-synchronous reference with
identical work, quality, and memory ceiling.

## Phase 7 — optional CUDA backend

- Dense-spine residency and device-resident activations.
- Persistent low-row expert queues.
- Separate compute, demand H2D, and speculative-prefetch streams.
- Explicit multi-GPU placement and topology-aware activation movement.

Gate: no silent CPU fallback; per-device tests; fixed replay; equal VRAM budget;
weight H2D reported separately from activation transfers.

## Phase 8 — Expert Compute Fabric

- Binary peer protocol and content-addressed model identity.
- Route-affinity partitioning that minimizes peer fan-out.
- Activation shipping to weight owners, replication, deadlines, health, and
  exact local failover.
- Aggregate throughput and per-request latency measurement on 1/10/25 GbE.

Gate: peer execution beats local cold reads without changing output or exceeding
network and memory ceilings.

## Phase 9 — read-aware speculation

- Four-bit-or-higher draft adapter.
- Verification-depth controller using accepted tokens per NVMe/H2D byte.
- Expert-union growth and false-prefetch feedback.

Gate: repeatable single-request latency or aggregate-throughput win. Acceptance
rate alone is not a success metric.

## Phase 10 — frontier-scale acceptance

- GLM-5.2 architecture adapter and scale validation.
- Selected DeepSeek-V4-class checkpoint validation.
- Resumable full conversion, cold/warm load, multi-GPU placement, and endurance
  accounting.

Gate: smaller architecture oracles already pass; full-model correctness passes;
steady-state physical NVMe reads satisfy the declared execution contract; all
throughput and memory claims are reproducible.

The precise download and conversion gates are documented in
[`model-bringup.md`](model-bringup.md).
