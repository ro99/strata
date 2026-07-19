# Research roadmap

GLM-5.2 is the primary target from the first implementation branch.
DeepSeek-V4-Flash-DSpark is the native mixed-FP4/FP8 resident proof of concept.
Every phase answers one bounded question against one of those real checkpoints.

Status snapshot: this roadmap mixes completed foundations with open promotion
gates and future research. A phase's bullet list is its intended scope, not a
claim that every item is implemented. `README.md` and
`current-architecture.md` describe runnable behavior; dated records under
`docs/experiments/` preserve the evidence for accepted and rejected experiments.

## Phase 0 — foundation (complete)

- C/C++ build with no third-party runtime.
- Precision-floor and MoE semantic validation.
- Signed-int4 reference matvec.
- Sequential trace parser, past-only predictor, residency simulator, and
  bounded-cold-read contract.

Gate: sanitizer tests and deterministic simulator output.

## Phase 1 — GLM-5.2 source lock and one-module vertical slice (implemented; external oracle gate open)

- Pin `QuantTrio/GLM-5.2-Int4-Int8Mix` configuration, index, license, hashes,
  and byte totals.
- Parse and classify all 177,569 tensors across 128 shards.
- Range-read one real expert's packed I32 values, BF16 group scales, and I64
  logical shape.
- Decode the native signed INT4 group-128 representation with bounded memory.
- Hash, reconstruct, and compare its W4A16 output to a frozen target oracle.

Gate: complete deterministic manifest, bounded RSS, deterministic output, and
recorded reconstruction metrics on an actual GLM-5.2 quantized module.

## Phase 2 — zero-rewrite GLM-5.2 import (implemented runtime path)

- Build a content-addressed sidecar over the 124 main and four MTP shards.
- Validate every packed value/scale/shape triplet against the pinned
  `compressed-tensors` policy.
- Preserve INT4 experts, INT8 linears and MTP, and BF16/FP32 tensors exactly.
- Open source shards read-only and keep runtime state outside the model files.
- Resume downloads without creating a second converted weight copy.

Gate: interruption tests pass; final role/byte/hash reconciliation passes;
missing, changed, overlapping, or writable source extents are rejected.

## Phase 3 — exact GLM-5.2 graph (base executor implemented; promotion gates open)

- Dense spine, low-rank attention, compressed KV, RoPE, and residual graph.
- DSA top-2048 selection and IndexShare ownership.
- Exact sigmoid/`noaux_tc` top-8 routing, shared expert, and routed experts.
- MTP, tokenizer, sampling, and generation.
- Operation, layer, multi-layer, and complete-model reference artifacts.

Gate: layerwise router/attention checks, teacher forcing, greedy generation, and
MTP acceptance pass declared numerical contracts on the real checkpoint.

## Phase 4 — GLM-5.2 resident execution (partial)

- NUMA-aware RAM expert arenas.
- VRAM-resident dense spine and explicit expert placement.
- Persistent native INT4 group-128 and INT8 CPU/CUDA kernels.
- Expert-ticket wavefront and continuous multi-request batching.
- Separate compute, demand H2D, and budgeted-prefetch queues.

Gate: fixed replay performs identical work without silent fallback; memory,
cache, transfer, and physical-I/O counters reconcile.

## Phase 5 — GLM-5.2 performance challenge (active research)

- Reproduce the known baseline contract on the exact target hardware.
- Run interleaved A/B measurements at equal model, precision, route sequence,
  RAM, VRAM, and warm/cold state.
- Compare row-synchronous, ticket-wavefront, and residency policies.
- Measure prefill, decode, MTP, H2D, NVMe, RSS, VRAM, and synchronization.

Gate: Strata exceeds median baseline decode tok/s outside run variance without a
quality, workload, memory, or I/O mismatch.

## Phase 6 — native DeepSeek-V4-Flash import (implemented)

- Pin the official 48-shard DSpark checkpoint.
- Build a zero-rewrite sidecar manifest over native Safetensors extents.
- Implement FP4 E2M1 K32 and FP8 E4M3 block-scaled reference paths.
- Stage the model into the combined RAM+VRAM resident budget.

Gate: all source bytes map to validated roles and steady-state admitted decode
performs zero physical NVMe reads.

## Phase 7 — exact DeepSeek-V4 graph (base executor implemented; promotion gates open)

- Hybrid compressed sparse/heavily compressed attention.
- Manifold-constrained hyper-connections and Sinkhorn invariants.
- Exact sqrtsoftplus/`noaux_tc` top-6 routing and shared/routed execution.
- YaRN, tokenizer/response encoding, and complete-model generation.

Gate: layerwise routing, attention, mHC, teacher forcing, and generation pass the
frozen full-model oracle.

## Phase 8 — DSpark wavefront (not implemented)

- Three-stage, five-token semi-autoregressive draft execution.
- Markov-logit correction and confidence-scheduled verification.
- Expert-union batching across provisional rows.
- Exact provisional KV commit/rollback.
- Adaptive block depth from accepted tokens per target-forward and expert row.

Gate: reproducible end-to-end throughput win; acceptance rate alone is not a
success metric.

## Phase 9 — evidence-driven residency (simulation infrastructure implemented; policy gate open)

- Capture sequential GLM-5.2 and DeepSeek-V4 routes.
- Compare LRU, LFU, leases, route-affinity cohorts, and uncertainty-budgeted
  prefetch at equal memory.
- Measure useful/wasted read bytes, H2D amplification, and expert-union growth.
- Define admission from measured working sets.

Gate: a new policy materially reduces physical NVMe bytes versus the best
conventional policy without increasing total transfer cost or changing work.

## Phase 10 — Expert Compute Fabric (not implemented)

- Content-addressed peer ownership and binary protocol.
- Activation shipping to resident expert owners.
- Route-affinity partitioning, replication, deadlines, and exact failover.
- Aggregate throughput and per-request latency on 1/10/25 GbE.

Gate: peer execution beats local cold reads without changing output or exceeding
network and memory ceilings.

Exact source revisions and detailed gates are documented in
[`model-bringup.md`](model-bringup.md).

## DeepSeek long-context follow-up

The runtime now admits arbitrary user-selected DeepSeek logical context ceilings
up to the checkpoint limit (1,048,576 tokens), with lazy host-page commitment
for the large cache and the declared learned top-512 index selection for ratio-4
layers beyond the 2,048-token index window. Short decode requests do not pay for
index maintenance merely because a large logical cache was configured. This is
storage/admission groundwork; execution evidence currently covers the first
selection boundary only.

The initial bounded batched-prefill path uses layer-major pages and multi-row
router projections while retaining exact row-ordered attention, cache, routing,
mHC, and expert semantics. Short and learned-index-boundary full-model oracle
equivalence pass. The 32k/200k/1m execution gates and further batched
attention/expert work remain open, with separate prefill metrics required.
