# Experiment 0057 — cutting serial terms out of the DeepSeek-V4 decode step

Status: **promoted, partial.** Decode is **1.140x** faster at the 586-token
operating point (253.8 → 222.6 ms/step, 3.941 → 4.493 tok/s) with byte-identical
output and identical cache behaviour. Prefill also improves 1.130x as a side
effect. The goal — matching the external stack's 10.2 tok/s — is **not** met, and
this record states what the remaining gap is made of and what it would cost.

Follows experiment 0056, which fixed prefill and left decode untouched by
construction. Follows experiment 0055 for the external control arm.

## Contract

- Hypothesis: decode's 253.8 ms/step is not a bandwidth cost. Instantiating
  `τ = max_r W_r/B_r + Σ_serial` at the real operating point should show the
  step dominated by `Σ_serial` — host-side bookkeeping, single-thread kernel
  phases, and host↔device round trips — none of which the design requires.
- Bottleneck measured before the change (63 decode steps, 586-token prompt):

  | term | ms/step | share | nature |
  |---|---:|---:|---|
  | MoE expert H2D (host blocked) | 56.5 | 22.3% | PCIe, serial |
  | MoE cache eviction bookkeeping | 14.3 | 5.6% | host, serial |
  | MoE execution | 46.5 | 18.3% | 26.6 GPU kernel + host |
  | attention score | 36.1 | 14.2% | 18.2 GPU kernel + host |
  | attention query | 26.8 | 10.6% | 2 matmul round trips |
  | attention output | 24.6 | 9.7% | 2 matmul round trips |
  | mHC pre + post | 32.3 | 12.7% | host, single core |
  | attention KV, router, head | 16.7 | 6.6% | |

  `argmax_r` is not a resource at all: ~124 ms/step is host serial work and
  ~65 ms is GPU kernel, against 56.5 ms of PCIe.
- Which resources the mechanisms reduce: host instruction count (eviction,
  mHC), GPU serialization inside one kernel (FlashAttention softmax), and
  copy/compute overlap on the PCIe link (upload stream).
- Sign on every other resource: expert H2D bytes, cache hits, misses,
  evictions, top-k, precision and routing all unchanged; VRAM unchanged; host
  DRAM grows by one pinned matmul staging buffer per device (≤ 64 MB each).
- Correctness gate, stated before the work: **generated token ids and answer
  text byte-identical**, and **cache hits/misses/evictions/demand bytes
  identical to the digit** — the second is what proves the recency lists pick
  the same victims the ranking scan did.
- Kill criterion: any output byte moves, or any cache counter moves.
- Rollback: each mechanism is independent; reverting this commit restores 0056.

## The four mechanisms

### 1. Eviction ranked every entry on every miss — 14.3 ms/step

`ensure()` chose its victim with a full scan of `state.entries`, ranking on
`(prefetched, last_use)`. A full device holds ~5,300 entries and decode evicts
~127 times a step: ~670,000 hash-node visits a step, in a container whose
iteration is a pointer chase.

Replaced with two intrusive recency lists per device — one for demand entries,
one for prefetched — ordered by `last_use`, which is exactly the key the scan
ranked on, with prefetched checked first, exactly the preference the scan had.
Pinned entries are kept out of the lists rather than stepped over; leased ones
stay listed and are skipped, because they were touched on the step that holds
them and therefore sit at the back.

This is a replacement of a ranking function by an index, so the victim is
unchanged. The gate is the cache counters, and they are identical.

### 2. The FlashAttention softmax ran on one thread — 18.2 → 9.7 ms/step

`flash_attention_reference_f32_kernel` computed scores across the block, then
did the entire softmax on `threadIdx.x == 0`: a scan for the maximum, a
sequential `__dadd_rn` fold of `exp(score - max)` in **double**, and then a
third pass recomputing the same double `exp` to normalise. Two double
exponentials per key row, serialised on one lane, while 255 lanes idled.

Only the denominator is order-dependent. Split accordingly:

- the maximum reduces — `fmaxf` ignores NaN from either side, so a tree fold and
  a sequential fold give the same float;
- the exponentials are per row and move to the whole block, into a dynamic
  shared-memory scratch of one double per key row;
- the denominator stays a sequential `__dadd_rn` fold over the same summands in
  the same order, now reading values the block already computed;
- the final divide is per row.

Thread 0's work drops from `2 × visible_rows` double exponentials to
`visible_rows` double adds. Beyond a 32 KB shared-memory budget the kernel keeps
the original single-thread fold, so long contexts stay correct.

Prefill uses the same kernel and gains from it too.

### 3. The mHC projection is bound by one core's line fills — 27.6 → 18.3 ms/step

`dsv4_mhc_project_prepacked_avx2` reads the whole packed projection: 1.57 MB a
call, 86 calls a decode step, 135 MB a step. Measured 5.8 GB/s, which is about
what one Broadwell core sustains with ten line-fill buffers — the term is memory
parallelism, not arithmetic.

Two changes. Blocks are now tiled four at a time so four accumulator chains are
in flight instead of one (the chain was 16,384 dependent `vaddpd`); and the
block range is split across host workers. Each lane owns whole output rows and
folds its columns in the same order, so no value moves. Prefill keeps the inline
path: it calls this once per row, where a per-row dispatch would cost more than
the projection it splits.

The tiling alone measured flat — the term was never latency-bound — which is
recorded here because it is the kind of plausible mechanism that does nothing.

### 4. Weight uploads shared the execution stream — host wait 64.8 → 0.07 ms/step

Demand expert transfers were issued on the same stream as the kernels that read
them, so within a layer the copy engine and the SMs ran strictly in series, and
the host then blocked in `cudaStreamSynchronize` for the whole transfer.

Uploads moved to a per-device copy stream. The ordering a consumer actually
needs is a device-side dependency, so it is now expressed as an event the
execution stream waits on, and the host returns immediately.

Also in this commit: matmul activations stage through pinned host memory. A
`cudaMemcpyAsync` whose host side is pageable is not asynchronous — the driver
stages it and blocks — and decode paid that on both legs of every matmul.
Device D2H time halved, 8.4 → 4.7 ms/step. On its own this measured flat end to
end, because the cost moved into the sync rather than disappearing; it is kept
because it is unambiguously the correct call and it makes the sync counter mean
what it says.

## Result

586-token prompt, 64 generated tokens, three GPUs, 216 GiB host ceiling, 0.95
VRAM fraction, `--flash-attention --pin-resident-arena`. Cumulative arms.

| arm | ms/step | tok/s | prefill s |
|---|---:|---:|---:|
| baseline (0056 on main) | 253.77 | 3.941 | 75.24 |
| + eviction index, + FlashAttention softmax | 235.07 | 4.254 | 70.43 |
| + pinned matmul staging, + mHC tiling | 234.78 | 4.259 | 69.53 |
| + mHC across host workers | 229.35 | 4.360 | 68.87 |
| + upload stream | **222.58** | **4.493** | **66.58** |

**1.140x decode, 1.130x prefill.**

| phase | base | final |
|---|---:|---:|
| MoE prepare (host demand wait) | 70.77 | 2.44 |
| MoE total | 119.99 | 111.59 |
| attention | 97.25 | 83.36 |
| mHC pre | 27.62 | 18.26 |
| FlashAttention kernel | 18.21 | 9.70 |

**Correctness: every arm emits identical answer text, and cache hits, misses,
evictions and demand bytes are identical to the digit across all arms —
70,377 / 7,995 / 7,995 / 35.629 GB.** `make check` passes 260/260.

## What the remaining gap is made of

The target was the external stack's 10.2 tok/s, or 98 ms/step. At 222.6 ms/step
the step now decomposes as:

| term | ms/step | what it would take |
|---|---:|---|
| matmul round trips | ~104 | fewer submissions; see below |
| expert PCIe, now overlapped only with itself | ~65 | resident/arriving split, or CPU experts |
| MoE FP4 GEMV kernel | ~27 | the kernel runs at 44.6 GB/s of 936 |
| mHC | ~23 | pool barrier is now most of it |
| FlashAttention | ~14 | |

Decode issues about **407 blocking matmul submissions per token**, each costing
~256 µs end to end while only ~94 µs of device work is measured inside it. A
standalone microbenchmark of the same shape on this box — pinned H2D, kernel,
pinned D2H, synchronize — costs **28.6 µs**, and stays 29 µs whether it runs back
to back or with a 1 ms host gap between iterations, so the residual is neither
submission cost nor a GPU power state.

A pinned/unpinned arena screen accounts for it, and the answer is that it is not
waste:

| arm | ms/step | matmul sync | matmul kernel | MoE execution |
|---|---:|---:|---:|---:|
| `--pin-resident-arena` | 223.65 | 377 µs/call | 69 µs/call | 106.75 |
| unpinned | 324.81 | 233 µs/call | 69 µs/call | 47.90 |

Unpinned is 1.45x **worse** overall, so pinning stays; but its matmul sync is
*shorter*. With uploads on the copy stream, a pinned expert transfer is fast
enough to keep the link busy across the whole layer, and the matmul's own
activation copies now queue against it. The per-call residual is the expert
transfer, showing up in the matmul bucket instead of a separate block — which is
what overlap is supposed to look like. Unpinned, the transfers are synchronous
inside `upload()` and the residual moves back out of the sync and into a much
larger total.

So after this experiment the real `argmax_r` for decode is the expert transfer
itself: 566 MB/step at ~10 GB/s aggregate over the three links (Gen3 x8, x8,
x16). That is a floor near 56-65 ms/step, or roughly 15-17 tok/s, with **every
other term needing to fit underneath it**.

Reaching the external stack's 98 ms/step therefore needs the remaining ~158
ms/step of non-transfer work cut to ~33 ms — a 4.8x reduction that no
term-by-term fix reaches. The structural difference is the submission count: the
external decode step is one `FULL_DECODE_ONLY` CUDA graph replay plus one CPU MoE
call, against Strata's ~700 host↔device round trips per token. That is the
project the next experiment has to take on, and items 1 and 2 below are its first
two stages rather than independent wins.

## Next, in order, with measured sizes

1. **Attention chains.** `wq_a → q_norm → wq_b` and `wo_a → bf16 → wo_b` are
   adjacent matmuls separated only by elementwise work. Fusing each into one
   submission removes 86 round trips a step, ~22 ms.
2. **Resident/arriving split inside the MoE command.** ~79% of a layer's expert
   acquisitions hit the VRAM cache. Launching their gate/quantize/down slice
   before the upload event and the rest after would hide ~21 ms/step of transfer
   behind kernel work already on the critical path, in one command, with no
   extra workspace.
3. **The FP4 GEMV kernel**, 44.6 GB/s against 936. Each thread loads one byte,
   so a warp requests 16 unique bytes per group. Loading cooperatively into
   shared memory and keeping the existing column→thread mapping would preserve
   the accumulation order exactly. ~27 ms/step of wall time is in this kernel.
4. **Capture the decode step as a CUDA graph**, which is what 1 and 2 are
   building toward and the only mechanism sized to the remaining gap. Blocked
   on the attention chain being device-resident, because a graph cannot contain
   a host round trip.

Not on this list, and why: **raising the VRAM hit rate**. The cache already
holds 62.4 GB of a 64 GB physical budget at a 0.95 fraction, and the expert →
device map is already capacity-weighted (measured transfer shares 24.7 / 39.4 /
35.9% against capacity shares 24.6 / 37.7 / 37.7%). There is no headroom there.

## Artifacts

`src/deepseek_runtime.cpp` (recency lists, mHC lanes), `kernels/cuda/backend.cu`
(FlashAttention block softmax, upload stream, pinned matmul staging,
matmul_issue/matmul_finish counters), `src/deepseek_ops.cpp` (mHC block tiling
and lane split), `include/strata/cuda_backend.hpp`,
`include/strata/deepseek_ops.hpp`, `src/cuda_stats_delta.hpp`,
`apps/strata_deepseek_run.cpp`. Run JSON under `results/dsv4-decode/` (ignored).
The round-trip microbenchmark is throwaway and lives in the session scratchpad.
