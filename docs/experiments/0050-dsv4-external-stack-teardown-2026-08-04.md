# Experiment 0050 — teardown of the external DeepSeek-V4-Flash stacks reporting 33 tok/s

Status: **investigation; no runtime code changed.** Reverse-engineers the two
repositories behind a public 33 tok/s single / 68 tok/s aggregate claim for
DeepSeek-V4-Flash-0731 on 2x RTX 3090, and reconciles them against Strata's own
measured decode at 4.0-4.2 tok/s.

## The claim under examination

A `r/LocalLLaMA` post reports DeepSeek-V4-Flash (284B MoE) at 33 tok/s single /
68 tok/s aggregate on 2x RTX 3090 plus a used quad-socket Xeon DDR4 server.
Their stated configuration, which is the operating point everything below is
compared against:

- Dell PowerEdge R940, **4x Xeon Platinum 8268** (96C/192T, Cascade Lake,
  **AVX512-VNNI**, no AMX)
- **768 GB DDR4-2933, 24 DIMMs, 6 channels/socket, 4 NUMA nodes** — 563 GB/s
  theoretical, and they quote 141 GB/s x 4 nodes = 564 GB/s
- 2x RTX 3090, TP=2, **6.6 GB of VRAM used for weights per card**, GPUs at ~25%
  utilisation; power-capping them 350 W -> 250 W changed throughput by zero
- Same checkpoint as ours: 156 GiB / 48 shards, routed experts native MXFP4
  (E2M1 + per-32 E8M0 — this is the encoding our manifest calls FP4 E2M1/E8M0)
- DSpark speculative decoding at depth 5, `max_num_seqs` 4

Their own control arm is the load-bearing number: **the same box on
`ik_llama.cpp` hybrid does 12.2 tok/s single-stream.** The 33 tok/s is a 2.6x
on top of that from the spec-decode + Marlin path.

Two repositories are named:

- `guqiong96/Lvllm` — vLLM fork whose MoE layers call **`lk_moe`**, a
  closed-source CPU-GPU hybrid MoE engine shipped as a wheel. The DeepSeek-V4
  build is `guqiong96/Lvllmds4-x` ("CPU-GPU hybrid inference for DeepSeek-V4 on
  NVIDIA SM80+"). SM86 is the 3090, so this is the arm that produced the number.
- `yhfgyyf/vllm-deepseek-v4-sm89` — an Ada capability port of vLLM's DeepSeek-V4
  path. Not the reported arm: it targets 4x RTX 4090 48 GB, where the model is
  **fully VRAM-resident**. Recorded here only because it prices DSpark.

## Result

**The claim is real, fully explained by arithmetic, and contains no algorithm
Strata is missing.** It is a different placement of the same work: their expert
weights never cross PCIe, because the expert matmuls execute on the CPU where
the weights already live.

**The 6x gap is not one mechanism.** At Strata's operating point PCIe is *not*
`argmax_r` and never was the largest term. The gap decomposes as roughly
2.5-4x of host scalar arithmetic, 1.3x of expert transfer, and a dispatch
overhead term on top of both.

**The headline trick does not transfer to this machine unchanged.** Their memory
subsystem is roughly 5x this one's. Substituting CPU expert compute for PCIe
transfer here, at today's DIMM population, is a *regression* on that term
(≈86 ms/step against 60 ms/step measured). It only pays as a hybrid, and it pays
much better after a hardware change identified below.

## What the external stack actually does

Read from `vllm/model_executor/layers/fused_moe/routed_experts.py` in `Lvllm`.
`lk_moe` itself is a binary wheel; the call boundary is fully legible.

```python
def _cpu_decode(self, hidden_states, topk_weights, topk_ids):
    stream_ptr = torch.cuda.current_stream().cuda_stream
    self.lk_moe.cpu_decode(
        stream_ptr,                        # a CUDA stream
        hidden_states.size(0), self.top_k,
        hidden_states.data_ptr(),          # DEVICE pointer
        topk_ids.data_ptr(),               # DEVICE pointer
        topk_weights.data_ptr(),           # DEVICE pointer
        RoutedExperts.output_gpu.data_ptr())  # static DEVICE buffer
```

Four properties, each load-bearing:

1. **Expert weights stay in host DRAM and are never transferred.** PCIe carries
   only the activation down and the MoE result up: at hidden 4096, that is 8 KB
   down and 16 KB up per layer per token, ≈1 MB/token against the 3.45 GB/token
   of routed-expert weights DeepSeek-V4-Flash activates. Three orders of
   magnitude less PCIe traffic, achieved by moving the *compute*, not the bytes.
2. **The CPU MoE is a stream-ordered operation.** It takes a `cudaStream_t` and
   device pointers, so it composes inside a CUDA graph rather than forcing a
   host round-trip. `RoutedExperts.output_gpu` is a *static class-level* buffer
   — the fixed address a graph capture requires.
3. **CUDA graphs carry the rest of the step.** Their changelog, 2025-10-14:
   "Enabled cuda graph, decode speed doubled!!" The documented run command sets
   `--compilation_config.cudagraph_mode FULL_DECODE_ONLY`.
4. **The split is tunable per layer.** `LVLLM_GPU_RESIDENT_MOE_LAYERS=0-5` keeps
   named MoE layers on the GPU; the rest go to the CPU. MTP layers are forced
   GPU-resident (`is_lk_moe_mtp_layer` ⇒ `is_lk_moe_gpu_resident_layer`).

NUMA is treated as a first-class placement input: `numactl --physcpubind=<cores>
--membind=<node>` per worker, `LK_THREAD_BINDING=CPU_CORE`, and BIOS guidance of
NPS4/EPYC or SNC4/Xeon to *maximise* node count. This is the same defect
experiment 0026 found in `Dsv4ResidentWeightStore::stage`, which still allocates
`MAP_PRIVATE|MAP_ANONYMOUS` with no NUMA policy.

## Why their number is 33 tok/s — instantiated, not assumed

DeepSeek-V4-Flash-0731 from `config.json`: 43 layers, 256 routed experts, top-6,
hidden 4096, `moe_intermediate_size` 2048, FP4 E2M1 + per-32 E8M0.

Per expert triplet: `3 x 4096 x 2048 x 0.53125 B` = 13.37 MB.
Per token: `43 x 6 x 13.37 MB` = **3.449 GB** of routed-expert weights.
(Check: `43 x 256 x 13.37 MB` = 147.2 GB, matching the admitted
147,169,738,752 B exactly.)

Expert term as a pure bandwidth quotient, `W_r/B_r`:

| resource | GB/s | ms/token | tok/s ceiling |
|---|---:|---:|---:|
| PCIe 3.0 x8, one link | 5.5 | 627 | 1.6 |
| Strata measured serial round-robin H2D | 6.7 | 515 | 1.9 |
| Strata measured 3-device overlapped H2D | 10.5 | 328 | 3.0 |
| **this machine, DDR4, measured** | **76** | **45** | **22** |
| their R940, 24 ch DDR4-2933, quoted | 563 | 6.1 | ~163 |
| their R940 at a realistic ~80% of peak | ~450 | 7.7 | ~130 |

Their non-speculative arm is the check: **12.2 tok/s on `ik_llama.cpp`** — i.e.
82 ms/step against a 7.7 ms expert-weight floor, so that engine is ~10x off its
own memory bound and the weight term is not what limits it. Their 33 tok/s is
that engine replaced (Marlin weight-only kernels, AVX512-VNNI expert kernels)
plus DSpark depth 5. The 68 tok/s aggregate is concurrent requests sharing one
weight read — the same DRAM traffic amortised over more tokens. Nothing is
unexplained, and nothing in it is a bandwidth miracle.

## Measured on this machine

### DDR4 read bandwidth

Probe: 8 independent AVX2 accumulators, THP-backed arena, each thread
first-touching and reading its own node-local region, physical cores only.
`scratchpad/membw2.c`.

| placement | threads | GB/s |
|---|---:|---:|
| node 0 only | 1 | 14.2 |
| node 0 only | 4 | 38.9 |
| node 0 only | 14 | 38.3 |
| both nodes, node-local | 8 | 74.5 |
| both nodes, node-local | 28 | **76.3** |

A first version of this probe that first-touched the whole arena on one thread
measured 32 GB/s at 8 threads and **fell to 23 GB/s at 28** — every thread
hammering node 0's controller. Recorded because it is the same class of defect
as the unbound resident arena, and because it is how a NUMA problem looks when
the probe does not control placement.

### The DIMM population is the reason 76 is not 153

From `/sys/devices/system/edac/mc/mc*/dimm*/dimm_label`:

| controller | populated |
|---|---|
| `CPU_SrcID#0_Ha#1` | `Chan#0_DIMM#0`, `Chan#1_DIMM#0` (2 x 64 GB) |
| `CPU_SrcID#1_Ha#0` | `Chan#0_DIMM#0`, `Chan#1_DIMM#0` (2 x 64 GB) |

**Two of four channels per socket.** E5-2680 v4 has two home agents x two
channels. Measured 38.0 GB/s per socket against 38.4 GB/s theoretical for two
channels of DDR4-2400 — **99% of spec for the channels that exist**. The memory
subsystem is not underperforming; half of it is absent.

Populating the remaining four DIMM slots is the single cheapest 2x available on
this machine, and it lands precisely on the resource that a CPU-expert design
makes `argmax_r`.

### Does an AVX2 FP4 kernel keep up with 76 GB/s?

Broadwell: AVX2, **no AVX-512, no AMX**. W4A16 GEMV over exact expert shapes,
E2M1 nibbles + E8M0 per-32 scales, L3-resident so this is the *compute* ceiling.
`scratchpad/fp4gemv.c`, 28 threads:

| shape | GB/s of weights |
|---|---:|
| w1/w3 `[2048,4096]` | 38.3 |
| w2 `[4096,2048]` | 61.5 |

So a first-cut kernel already runs at 38-61 GB/s against a 76 GB/s memory
ceiling — viable, but the kernel and the DRAM are the same order, so the
combined rate will land near 40 GB/s, not 76. A second attempt using a float
LUT and `permutevar8x32_ps` measured *worse* (16 GB/s): it hoisted the per-group
scale into an activation pre-pass placed inside the row loop, making it O(N·K).
Recorded so the mistake is not repeated. These are single-kernel screens, not a
tuned figure; treat 40 GB/s as the planning constant and re-measure before
building.

## Reconciliation: where Strata's step actually goes

Strata is **not** PCIe-bound at the production operating point. Experiment 0034
measured this directly and it is already in the repository:

Production baseline, 18-token chat prompt, 152 generated tokens, three GPUs,
`--flash-attention --pin-resident-arena`: **238-250 ms/step, 4.0-4.2 tok/s**.

| resource | ms/step | share |
|---|---:|---:|
| **host arithmetic (scalar C++)** | **104-187** | **42-75%** |
| PCIe demand H2D | 60.0 | 24% |
| GPU kernels | 41.0 | 16% |

Experiment 0029, independently: "critical GPU kernels account for only 5.013 of
45.452 seconds, so the GPU is idle for about 89% of the step."

Two structural facts behind the host term:

- **Attention scoring and value accumulation are scalar host C++ with
  double-precision `std::exp` per element** — `deepseek_runtime.cpp:2340-2385`,
  64 heads x 512 dims x ~640 positions x 43 layers, per token.
- **1,244 CUDA matmul calls per decode step** (experiment 0037): 903 MoE
  operations issued concurrently, plus **341 serial** direct projections at
  153 µs/call, of which 17.5 µs is activation H2D and 24.3 µs is activation D2H.
  Activations round-trip host↔device on every projection.
- **`grep -c cudaGraph kernels/cuda/backend.cu` = 0.** There is no graph capture
  anywhere in the runtime.

The external stack's decode step is one graph launch over device-resident
activations with the MoE dispatched to SIMD CPU cores. Strata's is ~1,244
host-issued calls around a scalar host graph. That, not the expert transfer, is
the bulk of 6x.

## Ranked transfers, ordered by `argmax_r`

Per the charter, a mechanism that does not reduce `argmax_r` cannot improve `τ`.
`argmax_r` is host arithmetic. The ordering below is therefore binding, and
**item 3 is the reddit trick but it is third, not first.**

1. **Move attention scoring/softmax and mHC onto the device, and keep
   activations device-resident across the graph.** Targets the 104-187 ms term
   directly. No new hardware, no architecture change, no precision change.
   `exp/dsv4-device-activations` already exists as a starting point. Projected
   ceiling if the host term goes to zero: 250 → ~100 ms/step, 4.0 → ~10 tok/s.
2. **Capture the decode step in a CUDA graph.** Targets the 341 serial
   round-trips and the per-call dispatch cost. Strictly depends on 1: graph
   capture requires device-resident activations at fixed addresses, which is why
   `lk_moe` writes into a static class-level output buffer. Their own changelog
   prices this at 2x.
3. **Compute *cold-miss* experts on the CPU instead of transferring them.**
   Targets the 60 ms PCIe term. **Check the sign before building:** the VRAM
   expert cache currently absorbs 83% of expert reads, so only ~581 MB/token
   crosses PCIe, not 3.45 GB. A wholesale CPU MoE costs
   3.45 GB / 40 GB/s ≈ 86 ms/step and is a **regression**. The hybrid —
   VRAM cache for hot experts, CPU compute for misses — costs
   581 MB / 40 GB/s ≈ 14.5 ms of CPU work that *overlaps* GPU attention,
   against 60 ms of serial PCIe stall. That is the version worth building, and
   it is what `LVLLM_GPU_RESIDENT_MOE_LAYERS` is.
4. **Populate the four empty DIMM slots.** 76 → ~150 GB/s, halving item 3's
   term. Only worth buying *after* item 3 is what makes DRAM the bottleneck;
   before that it changes nothing.
5. **Bind the resident arena to a NUMA node.** Already nominated as
   experiment 0026's "Next term" and still not done. Cheap, and it becomes
   substantially more valuable under item 3, where every expert read is a CPU
   read rather than a DMA source.
6. **DSpark / MTP speculative decoding — last, and gated.** The checkpoint
   carries the stages (`dspark_block_size` 5, target layers 40-42, Markov rank
   256) and `feat/dsv4-dspark-verification` exists.

   **Correction to an earlier draft of this document**, which discounted DSpark
   on the grounds that `vllm-deepseek-v4-sm89`'s 3.5-4.2x was measured on a
   compute-bound, fully-VRAM-resident 4x4090 config and would not transfer. The
   OP supplies the missing control: **the same hybrid CPU-GPU box measures
   12.2 tok/s on `ik_llama.cpp` against 33 tok/s on the spec-decode + Marlin
   path, a 2.6x in exactly our regime.** DSpark is a real multiplier here, not
   an artifact, and the earlier caution understated it.

   The ordering constraint stands for a different reason: speculation multiplies
   host arithmetic, and host arithmetic is `argmax_r` today — the shape of the
   experiment 0025 shadow-speculative rejection. Build it after items 1 and 2,
   then re-derive the break-even. Note also that their 2.6x bundles an engine
   change with the spec change, so it is an upper bound on spec alone.

## The structural asymmetry, and why we must not copy their design

| | them | us | |
|---|---:|---:|---|
| memory channels | 24 x DDR4-2933 | 4 x DDR4-2400 | **7.3x them** |
| measured/quoted DRAM | ~450-563 GB/s | 76 GB/s | **~6-7x them** |
| cores / ISA | 96C/192T AVX512-VNNI | 28C/56T AVX2 | **3.4x them, plus ISA** |
| VRAM used for weights | 13.2 GB total | 46.6 GB expert cache + 9.07 GB spine | **4x us** |
| expert bytes/token from host | 3.449 GB (all) | **0.586 GB** (17% miss) | **5.9x us** |
| **cost of that term** | 3.449/450 = **7.7 ms** | 0.586/76 = **7.7 ms** | **parity** |

**Our VRAM expert cache cancels our DRAM deficit almost exactly.** They have
48 GB of VRAM and put 13.2 GB of weights in it; the expert set streams from DRAM
in full, every token. We hold 46.6 GB of the 147 GB expert set in VRAM at a
measured 83% hit rate, so only 17% ever needs to come from the host.

This is the whole strategic point. Adopting their architecture — all experts on
the CPU — costs us 3.449 GB / 40 GB/s = **86 ms/step**, against 60 ms/step
measured on PCIe today. It is a regression, and it would throw away the one
resource where we are ahead of them. The transferable idea is narrower and
better: **use the CPU for cache misses only.**

## Budget to 30 tok/s on this hardware

30 tok/s is 33.3 ms/step against 238-250 ms/step measured today — 7.5x.

Per-token byte volumes, all from the pinned manifest and the admission table:

- VRAM reads: 9.069 GB spine + 0.83 x 3.449 = 2.86 GB cached experts = **11.93 GB**
- host reads: 0.17 x 3.449 = **0.586 GB**
- PCIe activations, if the MoE moves host-side: ~1 MB/token (8 KB down, 16 KB up
  per layer), i.e. negligible against 581 MB/token of weights today

| term | best case | notes |
|---|---:|---|
| VRAM reads, bandwidth-balanced placement | 5.1 ms | 11.93 GB / (936+936+448) GB/s |
| VRAM reads, current capacity split `[0,0,1,1,1,2,2,2]` | 16.2 ms | 25% of the work on the 448 GB/s card |
| host expert misses, DRAM-limited | 7.7 ms | 0.586 GB / 76 GB/s |
| host expert misses, AVX2-kernel-limited | 14.7 ms | 0.586 GB / 40 GB/s measured |
| attention + mHC + router, on device | 3-5 ms | est., not measured |
| graph launch + residual dispatch | 2-4 ms | est., not measured |

Balanced placement with the CPU miss path **overlapped**: ~12-16 ms → 60-80 tok/s.
Balanced placement with it **serial**: ~20-28 ms → 36-50 tok/s.
Current placement, misses serial: ~35-40 ms → 25-29 tok/s.

**30 tok/s has headroom on this machine without DSpark**, and DSpark is a
further 2.6x by their measurement. The gap is not capacity — it is that
104-187 ms/step of scalar host arithmetic and ~1,244 host-issued CUDA calls sit
on top of a ~15 ms machine.

Two of these numbers are estimates and are flagged as such; the VRAM-read term
in particular has **never been measured** — whether our spine and experts are
distributed across the three GPUs by bandwidth or by capacity decides an
11-point swing (5.1 vs 16.2 ms) and is one probe away.

## Staged plan, with gates

Each stage's gate is the previous stage's `argmax_r`. Stage dependencies are
binding: stage N produces the operating point that prices stage N+1.

1. **Device-resident activations; attention scoring, softmax and mHC on the
   GPU.** Targets host arithmetic, `argmax_r` at 42-75%. `exp/dsv4-device-activations`
   exists. Gate: host arithmetic below 15% of the step. Projected 250 → ~100-110
   ms/step, **9-10 tok/s**.
2. **One fused MoE call per layer, routing read from device memory; then CUDA
   graph capture of the decode step.** Targets the 903 concurrent + 341 serial
   calls. Note the ordering constraint: a CUDA graph needs fixed kernel
   arguments, so the 903 per-expert matmuls *cannot* be captured as they stand —
   they must first collapse into a single kernel that loops over experts
   internally, reading `topk_ids` from a device buffer. This is exactly why
   `lk_moe` exposes one `cpu_decode` call writing to a static device buffer.
   Projected → ~50-60 ms/step, **17-20 tok/s**.
3. **Measure, then rebalance, layer/expert placement by device bandwidth.**
   Cheap, and worth up to 3x on the VRAM-read term if the split is currently
   capacity-weighted. Projected → ~35-45 ms/step, **22-28 tok/s**.
4. **AVX2 MXFP4 W4A16 CPU kernel for cache-miss experts, overlapped with GPU
   work.** Replaces the residual serial PCIe stall with CPU work on a resource
   that is otherwise idle. Gate: must beat the measured PCIe miss cost at the
   *then-current* hit rate, not today's. Projected → ~28-33 ms/step,
   **30-36 tok/s. Target met.**
5. **Optional headroom: DSpark depth 5.** Their measured 2.6x on a hybrid box.
   Not before stages 1-2: speculation multiplies host arithmetic, and host
   arithmetic is `argmax_r` today — the shape of the experiment 0025 rejection.
   Re-derive the break-even at the stage-4 operating point.

Cheap adjacent levers, both previously nominated and neither done:

- **Populate the four empty DIMM slots.** 76 → ~150 GB/s. Only matters from
  stage 4 onward. The OP's "+2 DIMMs moved throughput ~5%, within noise" is not
  evidence against this: they went 22 → 24 of 24 slots, 92% → 100%. We would go
  50% → 100%.
- **NUMA-bind the resident arena** (experiment 0026's unexecuted "Next term").
  Worth ~1.02x today; worth much more at stage 4, where every expert read
  becomes a CPU read rather than a DMA source.
- **Frequency-aware expert placement.** The current `expert_device(e) =
  schedule[e % 8]` is arbitrary with respect to route frequency. The miss rate is
  the multiplier on the entire stage-4 term: 83% → 90% hit would cut host bytes
  from 586 MB to 345 MB per token.

## What is *not* worth taking

- **The sm89 fork's kernels.** Sparse MLA via FlashInfer, Marlin FP4 MoE
  dequant, E8M0 Triton fixes — all are capability ports for a VRAM-resident
  model. They solve "Ada lacks FP4 tensor cores", which is not a problem Strata
  has on 8.6/12.0.
- **`lk_moe` itself.** Closed-source wheel, and the charter forbids a Python or
  framework runtime in the runtime path. The *design* is what transfers; the
  binary cannot.
- **Their aggregate-throughput framing.** 68 tok/s aggregate is concurrent-request
  batching. `exp/dsv4-cross-request-scheduling-screen` rejected that here, but at
  a PCIe-bound operating point; the rejection should be re-derived, not reused,
  if item 3 lands.

## Open defects surfaced, not resolved

- The 42-75% host-arithmetic range is still open (experiment 0034's own note).
  Closing it needs the blocking-sync arm with the graph thread pinned, which
  requires per-thread affinity that does not exist yet.
- Device 2 does not respond to NUMA source placement (6.71/6.56/6.64 GB/s)
  despite reporting gen3 x16, where its x8 neighbours reach 5.9 and 11.9
  (experiment 0026). Still unexplained.

## Artifacts

Probes are throwaway and live in the session scratchpad, not in the repository:
`membw.c` (naive, retained as the NUMA-defect demonstration), `membw2.c`
(node-local read bandwidth), `fp4gemv.c` (AVX2 W4A16 expert GEMV),
`fp4gemv2.c` (rejected float-LUT variant). Every number above is reproducible
from the two committed sources of truth — `config.json` and experiments 0026,
0029, 0034 and 0037 — plus `lscpu`, EDAC, and the four probes.
