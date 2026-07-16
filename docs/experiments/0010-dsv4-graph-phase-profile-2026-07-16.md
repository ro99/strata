# DeepSeek V4 exact decode graph-phase profile

Status: **instrumentation complete; nominate VRAM weight arena**.

Two 128-token `Hello` runs used the exact `main` defaults: device MoE, 28 host
attention workers, devices `0,1,2`, 216 GiB host admission, 0.85 VRAM fraction,
and zero decode checkpoint reads. The instrumentation adds wall timers only.

| Phase | Run 1 seconds | Run 2 seconds |
|---|---:|---:|
| Decode wall | 28.9544 | 31.7929 |
| mHC pre | 4.8284 | 4.8255 |
| Attention | 10.2526 | 10.3815 |
| MoE | 12.7227 | 15.4435 |
| mHC post | 0.5502 | 0.5486 |
| Output head | 0.3043 | 0.2983 |

The second run split attention into 3.4087 seconds of query work, 1.1855
seconds of KV/compression work, 2.5293 seconds of score/value work, and 3.2023
seconds of output projection. No single attention subphase supplies a large
low-risk ceiling.

MoE was different. Router work took 0.3273 seconds and device execution took
6.0748 seconds, while descriptor/cache preparation took **9.0299 seconds**.
Decode performed 3,858 cache misses, 7,716 CUDA weight allocations, and 17.193
GB of weight H2D. There were no evictions; the observed route working set fit
the admitted cache. The 2.72-second run-to-run decode spread appeared entirely
inside MoE preparation, while mHC and attention were stable.

## Decision

Nominate a preallocated, capacity-bounded per-device weight arena as the next
experiment. It must retain the existing LRU/lease semantics, byte ceilings,
atomic expert admission, and exact CUDA weight descriptors while replacing
per-weight `cudaMalloc`/`cudaFree` synchronization with reusable suballocations.
Reject it if preparation time and median decode throughput do not improve
materially or if any correctness, memory, lease, or zero-read gate changes.

Artifacts are ignored under `results/deepseek-v4-graph-phase-v1/` and
`results/deepseek-v4-graph-phase-v2/`.
