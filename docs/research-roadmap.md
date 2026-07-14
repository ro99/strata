# Research roadmap

GLM-5.2 is the primary target from the first implementation branch.
DeepSeek-V4-Flash-DSpark is the native mixed-FP4/FP8 resident proof of concept.
Every phase answers one bounded question against one of those real checkpoints.

## Phase 0 — foundation (complete)

- C/C++ build with no third-party runtime.
- Precision-floor and MoE semantic validation.
- Signed-int4 reference matvec.
- Sequential trace parser, past-only predictor, residency simulator, and
  bounded-cold-read contract.

Gate: sanitizer tests and deterministic simulator output.

## Phase 1 — GLM-5.2 source lock and one-tensor vertical slice (next)

- Pin `zai-org/GLM-5.2-FP8` configuration, index, license, hashes, and byte
  totals.
- Parse and classify all 118,629 tensors across 141 shards.
- Read one real FP8 expert tensor and its 128×128 block scales.
- Convert it to Q4_K64 with bounded memory.
- Reopen, hash, dequantize, and compare it numerically to the source.

Gate: complete deterministic manifest, bounded RSS, deterministic output, and
recorded reconstruction metrics on an actual GLM-5.2 tensor.

## Phase 2 — resumable GLM-5.2 pack

- Convert one official shard at a time into immutable target extents.
- Quantize routed expert matrices first; preserve sensitive and MTP tensors.
- Journal and hash progress at extent granularity.
- Resume without rewriting completed output.
- Optionally release source shards only after durable verification.

Gate: interruption tests pass; final role/byte/hash reconciliation passes; the
planner's disk and memory ceilings are respected.

## Phase 3 — exact GLM-5.2 graph

- Dense spine, low-rank attention, compressed KV, RoPE, and residual graph.
- DSA top-2048 selection and IndexShare ownership.
- Exact sigmoid/`noaux_tc` top-8 routing, shared expert, and routed experts.
- MTP, tokenizer, sampling, and generation.
- Operation, layer, multi-layer, and complete-model reference artifacts.

Gate: layerwise router/attention checks, teacher forcing, greedy generation, and
MTP acceptance pass declared numerical contracts on the real checkpoint.

## Phase 4 — GLM-5.2 resident execution

- NUMA-aware RAM expert arenas.
- VRAM-resident dense spine and explicit expert placement.
- Persistent Q4_K64 CPU/CUDA expert kernels.
- Expert-ticket wavefront and continuous multi-request batching.
- Separate compute, demand H2D, and budgeted-prefetch queues.

Gate: fixed replay performs identical work without silent fallback; memory,
cache, transfer, and physical-I/O counters reconcile.

## Phase 5 — GLM-5.2 performance challenge

- Reproduce the known baseline contract on the exact target hardware.
- Run interleaved A/B measurements at equal model, precision, route sequence,
  RAM, VRAM, and warm/cold state.
- Compare row-synchronous, ticket-wavefront, and residency policies.
- Measure prefill, decode, MTP, H2D, NVMe, RSS, VRAM, and synchronization.

Gate: Strata exceeds median baseline decode tok/s outside run variance without a
quality, workload, memory, or I/O mismatch.

## Phase 6 — native DeepSeek-V4-Flash import

- Pin the official 48-shard DSpark checkpoint.
- Build a zero-rewrite sidecar manifest over native Safetensors extents.
- Implement FP4 E2M1 K32 and FP8 E4M3 block-scaled reference paths.
- Stage the model into the combined RAM+VRAM resident budget.

Gate: all source bytes map to validated roles and steady-state admitted decode
performs zero physical NVMe reads.

## Phase 7 — exact DeepSeek-V4 graph

- Hybrid compressed sparse/heavily compressed attention.
- Manifold-constrained hyper-connections and Sinkhorn invariants.
- Exact sqrtsoftplus/`noaux_tc` top-6 routing and shared/routed execution.
- YaRN, tokenizer/response encoding, and complete-model generation.

Gate: layerwise routing, attention, mHC, teacher forcing, and generation pass the
frozen full-model oracle.

## Phase 8 — DSpark wavefront

- Three-stage, five-token semi-autoregressive draft execution.
- Markov-logit correction and confidence-scheduled verification.
- Expert-union batching across provisional rows.
- Exact provisional KV commit/rollback.
- Adaptive block depth from accepted tokens per target-forward and expert row.

Gate: reproducible end-to-end throughput win; acceptance rate alone is not a
success metric.

## Phase 9 — evidence-driven residency

- Capture sequential GLM-5.2 and DeepSeek-V4 routes.
- Compare LRU, LFU, leases, route-affinity cohorts, and uncertainty-budgeted
  prefetch at equal memory.
- Measure useful/wasted read bytes, H2D amplification, and expert-union growth.
- Define admission from measured working sets.

Gate: a new policy materially reduces physical NVMe bytes versus the best
conventional policy without increasing total transfer cost or changing work.

## Phase 10 — Expert Compute Fabric

- Content-addressed peer ownership and binary protocol.
- Activation shipping to resident expert owners.
- Route-affinity partitioning, replication, deadlines, and exact failover.
- Aggregate throughput and per-request latency on 1/10/25 GbE.

Gate: peer execution beats local cold reads without changing output or exceeding
network and memory ceilings.

Exact source revisions and detailed gates are documented in
[`model-bringup.md`](model-bringup.md).
