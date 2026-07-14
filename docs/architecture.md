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

For DeepSeek, MLA and shared experts are part of this spine. Treating a shared
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
weights must also be int4 or higher. Verification depth is controlled by accepted
tokens per storage byte and expert-union growth, not merely draft acceptance.

## Architecture adapters

### Dense

MHA, GQA, MQA, and MLA are graph operations rather than separate engines. If a
dense model exceeds aggregate resident memory, every weight is active and storage
traffic is unavoidable. Admission must say so.

### Conventional MoE

The manifest declares softmax/sigmoid scoring, normalization, capacity behavior,
top-k, and optional shared experts. Expert gate/up/down tensors form one placement
unit unless an explicitly validated sharding plan says otherwise.

### DeepSeek

DeepSeek is a native adapter, not a collection of special cases. It validates and
executes:

- MLA compressed KV;
- the dense-prefix boundary;
- fine-grained routed experts;
- always-resident shared experts;
- group-limited routing where configured;
- `noaux_tc` correction-bias selection semantics;
- raw router weights versus selection scores;
- top-k normalization and routed scaling.

The shared expert overlaps remote/local routed execution. Group selection provides
a lower-entropy signal for conservative prefetch, but only the exact final router
may select computation.

## Backend boundary

The scheduler, model graph, and memory ownership are C++20. Kernels use a stable C
ABI so CPU, CUDA, and future backends do not leak allocator or C++ ABI state across
the boundary.

The default CPU build has no external runtime dependencies. CUDA will be optional
and dynamically isolated. Persistent arenas and explicit events replace per-call
allocation and synchronization.
