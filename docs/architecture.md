# Strata architecture

## Decision

Strata is an **expert-centric wavefront runtime**. The engine does not regard
NVMe, RAM, and VRAM as interchangeable caches with different speeds. It gives
each tier a different contract:

- VRAM executes dense layers and the active expert frontier.
- local RAM is the canonical capacity tier;
- peers execute experts beside their resident weights;
- NVMe is immutable recovery and warm-up storage.

The goal is zero local storage reads during steady-state decode for any request
that passed admission under a zero-read contract.

## Why this differs from existing offload engines

Token-synchronous engines discover a route, fault the expert, compute it, and
repeat at the next layer. Cache policy can reduce faults but cannot create reuse
that the scheduler never exposes.

Strata separates the exact transformer dependency graph from physical execution.
Rows advance independently through a bounded wavefront. Exact router results
become tickets keyed by `(model, layer, expert, owner)`. The scheduler may combine
tickets from different requests, prompt positions, or canonical speculative
verification rows without changing their mathematical order.

## Resident model spine

The following components are never demand-paged during an admitted request:

- embeddings and output head;
- normalization and router weights;
- attention projections for the assigned layer range;
- KV state;
- every shared expert;
- scheduler and kernel workspaces.

For GLM-5.2, low-rank attention, DSA/IndexShare, and the shared expert are part
of this spine. For DeepSeek-V4, hybrid-attention compression state, sparse
indexers, mHC parameters, and the shared expert are resident. Treating a shared
expert as cacheable routed data is an architectural error because it executes on
every token.

## Core loop

1. **Admission.** Estimate the request's route cohort, canonical working set,
   peer coverage, context memory, and maximum cold reads. Issue a residency/I/O
   contract or queue/refuse the request.
2. **Dense frontier.** Run normalization, attention, residual update, and the
   exact router while the hidden row remains on its assigned GPU.
3. **Ticket emission.** Publish expert IDs, row IDs, and exact coefficients into
   a preallocated ring. Launch the shared expert concurrently.
4. **Expert aggregation.** Merge compatible tickets across ready rows. A bounded
   latency deadline prevents batching from harming interactive requests.
5. **Owner selection.** Prefer VRAM, then local RAM compute or persistent H2D
   promotion, then peer execution. NVMe is a contract-governed last resort.
6. **Execution.** Persistent low-row kernels consume expert queues. There are no
   per-expert hot-path allocations or global device synchronizations.
7. **Scatter and advance.** Weighted expert results and the shared result join in
   the original row, which advances to the next layer.

## Resident Execution Contract

Admission returns one of three explicit contracts:

- `resident-local`: predicted canonical working set fits local RAM/VRAM;
- `resident-fabric`: missing experts have healthy peer owners;
- `bounded-cold`: at most a declared number of NVMe bytes may be read.

If a strict contract runs out of cold-read budget, the engine stops or queues the
row. It does not continue thrashing storage. Approximate expert output is not an
available fallback.

## Residency epochs and leases

Residency is planned over short epochs rather than one miss at a time. Each page
receives a utility estimate from observed transitions, active route cohorts,
expected reuse, transfer cost, cache pressure, and forecast uncertainty.

A prefetched page receives a minimum lease. Eviction prefers expired, low-value
pages. Prediction traffic has its own byte and queue-depth budget so it cannot
starve exact demand traffic.

The predictor is trained online from past routes only. Its output cannot enter
the numerical graph. The relevant metric is:

```text
useful prefetched bytes / total prefetched bytes
```

Recall alone is insufficient because false-positive expert reads are large.

## Route-affinity cohorts

Per-token top-k sets form hyperedges between experts. Offline and online
partitioners use these hyperedges to:

- co-locate experts commonly selected together;
- minimize the number of peers contacted by one layer;
- group requests with compatible warm sets;
- preserve compute and capacity balance.

This is substantially different from assigning experts by raw frequency or
device throughput alone. The optimization target includes network fan-out and
future reuse.

## Expert Compute Fabric

Remote RAM is not exposed as faultable shared memory. Pulling a 19–22 MB expert
over 1 GbE is normally worse than local NVMe. Strata instead ships a compact
activation batch to the peer that owns the expert.

One layer RPC contains the activation rows, selected expert IDs, and router
coefficients for all experts owned by that peer. The response is one weighted
partial output per row. Persistent binary connections, model hashes, bounded
queues, and request deadlines are sufficient for the first implementation;
RDMA is optional future work.

For a 7,168-wide hidden state, one fp16 row is about 14 KB. A 14 KB request plus
14 KB response per peer is orders of magnitude smaller than moving expert
weights. Canonical activation transport precision is declared by the model
execution contract and may not silently change.

## Multi-request and temporal batching

Multiple requests can provide independent exact rows immediately. One request
cannot batch unknown future tokens without speculation because autoregressive
dependency is real.

Temporal batching therefore uses canonical speculative verification only. Draft
weights must also be four bits or higher. GLM MTP and DeepSeek DSpark rows enter
the same provisional expert-ticket wavefront, but each architecture retains its
exact verification and KV commit rules. Verification depth is controlled by
accepted tokens per target forward, expert row, and storage byte—not merely
draft acceptance.

## Architecture adapters

### Dense

MHA, GQA, MQA, and MLA are graph operations rather than separate engines. If a
dense model exceeds aggregate resident memory, every weight is active and storage
traffic is unavoidable. Admission must say so.

### Conventional MoE

The manifest declares softmax/sigmoid scoring, normalization, capacity behavior,
top-k, and optional shared experts. Expert gate/up/down tensors form one placement
unit unless an explicitly validated sharding plan says otherwise.

### GLM-5.2

The `glm_moe_dsa` adapter validates and executes low-rank query/KV projections,
DSA top-2048 selection, IndexShare ownership, the three-layer dense prefix,
fine-grained top-8 MoE, the always-resident shared expert, sigmoid/`noaux_tc`
routing, routed scaling, and MTP. DSA indexers and MTP are graph components, not
optional approximations hidden behind a generic attention interface.

For the pinned QuantTrio checkpoint, the adapter also binds every linear role to
its declared `compressed-tensors` encoding: INT4 group-128 for routed experts,
INT8 group-128 for ordinary linears, channelwise INT8 for MTP, and BF16/FP32 for
sensitive tensors. Packed I32 storage never overrides those logical semantics.

### DeepSeek-V4

The `deepseek_v4` adapter validates and executes:

- compressed sparse and heavily compressed hybrid attention;
- per-layer compression ratios, sparse indexing, and YaRN scaling;
- manifold-constrained hyper-connections and Sinkhorn normalization;
- native FP4 E2M1 routed experts and the always-resident shared expert;
- `sqrtsoftplus` scoring, `noaux_tc` selection, top-6 normalization, routed
  scaling, and clipped SwiGLU semantics;
- three DSpark stages, Markov-logit correction, confidence scheduling, and exact
  speculative verification.

The shared expert overlaps routed execution. DSpark's five provisional rows are
exposed to the ticket scheduler so resident experts can be reused, but only the
target model may commit tokens or KV state.

## Backend boundary

The scheduler, model graph, and memory ownership are C++20. Kernels use a stable C
ABI so CPU, CUDA, and future backends do not leak allocator or C++ ABI state across
the boundary.

The default CPU build has no external runtime dependencies. CUDA will be optional
and dynamically isolated. Persistent arenas and explicit events replace per-call
allocation and synchronization.
