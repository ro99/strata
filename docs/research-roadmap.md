# Research roadmap

Each phase is gated by measurements. A failed mechanism is recorded and removed
from runtime code.

## Phase 0 — foundation (current)

- C/C++ build with no third-party runtime.
- Precision-floor and DeepSeek semantic validation.
- Exact int4 reference kernel.
- Sequential trace parser, online predictor, residency simulator, and cold-read
  contract.

Gate: sanitizer tests and deterministic simulator output.

## Phase 1 — real route evidence

- Capture sequential GLM-5.2 and DeepSeek traces with request, token, layer,
  top-k IDs, and coefficients.
- Add offline LRU/LFU/lease and affinity-hypergraph analysis.
- Measure working-set size, transition stability, useful prediction bytes, and
  peer fan-out at several memory budgets.

Gate: a novel policy must reduce NVMe bytes by at least 2x versus the best
conventional policy at equal capacity before runtime implementation.

## Phase 2 — model pack and CPU oracle

- Implement incremental Safetensors ingestion and the v1 immutable pack.
- Implement RMSNorm, RoPE, dense matmul, router variants, SwiGLU, conventional
  attention, and MLA reference operations.
- Validate a tiny dense model, tiny conventional MoE, and tiny DeepSeek model.

Gate: teacher forcing and greedy generation match architecture references.

## Phase 3 — persistent CPU execution

- NUMA-local expert arenas and persistent worker queues.
- Expert-major low-row int4 kernels for AVX2 and AVX-512.
- Shared-expert/routed overlap.
- Bounded wavefront and multi-request scheduling.

Gate: measurable throughput gain over row-synchronous reference execution with
identical work and memory.

## Phase 4 — optional CUDA backend

- Dense layer residency and device-resident activations.
- Persistent low-row int4 expert queues.
- Separate compute, demand H2D, and speculative-prefetch streams.
- Topology-aware activation movement without a mandatory NCCL dependency.

Gate: no silent CPU fallback; per-device tests; fixed replay; equal VRAM budget.

## Phase 5 — Expert Compute Fabric

- Binary peer protocol and content-addressed model identity.
- Route-affinity partitioner minimizing peer fan-out.
- Replication, deadlines, health, and exact local failover.
- Aggregate throughput and per-request latency measurement on 1/10/25 GbE.

Gate: peer execution beats local cold reads without changing output or exceeding
network and memory ceilings.

## Phase 6 — read-aware speculation

- Int4-or-higher draft adapter.
- Verification-depth controller using accepted tokens per NVMe/H2D byte.
- Expert-union and false-prefetch feedback.

Gate: repeatable single-request latency or aggregate-throughput win. Acceptance
rate alone is not a success metric.
