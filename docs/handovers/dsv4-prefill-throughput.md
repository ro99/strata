# Handover — DeepSeek V4 TP2 prompt throughput

Goal: get rank-local TP2 prompt processing to roughly the external reference's
numbers. Everything below is measured on the reference pair (devices 1,2) with
`models/dsv4f`.

This handover was written at `20de4e9`. Read `docs/experiments/0095` and `0096`
for that baseline, but also read 0100, 0106 and 0107 before acting on its cost
model. Two original diagnoses below were corrected in place after direct
measurement: candidate metadata was not the attention bottleneck (0100), and
the production tiled expert arena did not lack a NUMA policy (0106). Experiment
0107 is the promotion measurement for the mechanisms that followed.

## Where we are

| prompt | Strata | reference, no spec | gap |
| ---: | ---: | ---: | ---: |
| 677 | 10.5 tok/s | 66.6 | 6.3x |
| 2,612 | 11.6 tok/s | 197.9 | 17x |

The gap widens with length. That is the whole diagnosis, and it is not about
throughput at one point.

Fit both stacks as `fixed_per_chunk + marginal_per_token`:

| | fixed | marginal |
| --- | ---: | ---: |
| reference | 10–13 s | **1.14 ms/token** |
| Strata | ~4 s | **85.1 ms/token** |

They have a big fixed cost and almost no marginal cost; we have the opposite.
**The work is not "make prefill faster", it is "move cost from marginal to
fixed"** — turn per-token work into per-page work. Measure that ratio after
every change; it predicts the curve, and tok/s at one prompt length does not.

The expert upload has crossed over: its bytes per token fall to 0.32x from 677
to 2,612 tokens because the distinct experts a page touches saturates toward
256. It is 12% of the cost. Nothing else has crossed over.

## Current decomposition, 677 tokens, 64.02 s median

| term | seconds | note |
| --- | ---: | --- |
| attention | 33.8 | 18.2 scoring, 10.3 query, 4.1 kv |
| MoE upload | 16.1 | 73.8 GB, bimodal — see defect below |
| MoE kernel | 4.0 | coalesced, gated against the oracle |
| MoE other | ~6 | unattributed, worth splitting |
| mHC | 2.7 | fine |

At 2,612 tokens attention is 144.8 s of 226 s and gets *worse* per token
(51.03 → 55.44 ms/token).

## What to do next, in measured order

### 1. Attention scoring — 18.2 s at 677, 73.4 s at 2,612

`physical_paged_attention` runs once per (row, layer): 29,111 calls at 677
tokens, 112,316 at 2,612. **Two thirds of it is host, not device** — at 2,612
tokens `maximum_device_dsv4_paged_attention_seconds` is 24.4 s of the 73.4 s.

**Corrected by experiment 0100:** the original attribution to candidate
resolution and page-set construction was false. At 677 tokens, candidate
resolution measured 0.209 s and page-set construction 0.072 s out of an
approximately 18.5 s scoring bucket. The dominant defect was structural:
29,111 per-(row, layer) paged-attention calls caused 553,109 kernel launches and
6,989,956,736 bytes of page reads for a roughly 30 MB physical-page working
set. Metadata was about 1.5% of the bucket; dispatch, synchronization and
redundant device reads were the work to remove.

The KV-lease obstacle described by the original handover was real: appends must
finish before any shared read lease is held. The landed implementation keeps
the append/attend split, then gives `CudaDsv4PagedAttentionRequest` a row
dimension and issues one physical attention request per page and layer.
Experiment 0107 measured 86 calls, 1,655 launches and 30,564,224 page bytes,
with median scoring falling from 27.618 s on the same-build main-equivalent arm
to 14.320 s. The page-set hoist was a prerequisite, not the performance
mechanism by itself.

### 2. Expert arena locality and variance — policy exists; cause remains open

**Corrected by experiment 0106:** production rank-local TP2 prefill does not
consume the bare unbound resident allocation. It consumes the tiled arena from
`Dsv4ResidentWeightStore::stage`, which binds shard 0 to NUMA node 0 and shard
1 to node 1 before first touch. Live `numa_maps` accounting found exactly
77,913,391,104 bytes on each bound node with no expert page on the wrong node.
The separate bare `MAP_PRIVATE|MAP_ANONYMOUS` allocation still exists, but it
is not the source used by this production path.

Both reference GPUs are NUMA-affine to node 1, so approximately half of every
expert upload is deterministically remote. Each node has only about 129 GB for
a roughly 156 GB tiled arena, making full node-1 locality impossible. Whether a
capacity-aware asymmetric policy can improve mean upload service is a real open
question, but it is distinct from run-to-run variance and must account for what
becomes less local.

The 15.50/16.11/32.38 s figures quoted by the original handover came from
page-64 runs moving approximately 468 GB, not the page-8192 production point.
Do not reuse them as a production-page noise floor. Experiment 0107 measured
three interleaved pairs at 677 tokens and page 8192: the main-equivalent arm
ranged over 13.53 s total prefill, while the candidate ranged over 2.63 s. The
spread was concentrated in expert demand wait; attention itself remained
stable. Its cause is open because 0106 ruled out variable arena page placement.
Use the spread measured in the campaign and operating point under test rather
than carrying either range to another page size or build.

### 3. Activation residency — 10.3 s of attention's query term

The projections round-trip through the host: `linear_rows` downloads
677 × 32,768 floats per layer for the query projection alone. Rounding to BF16
now happens on the device, but the download does not. There is an existing
`exp/dsv4-device-activations` branch ("keep DeepSeek activations device-resident
across the graph") — start there rather than from scratch.

### 4. Page-locking the resident arena — ~11 s, blocked on reserve accounting

The guard refusing it is gone. Registering 156 GB costs ~72 MB of device-side
mapping, and the window between "arena large enough for the resident spine" and
"total under the 21.0 GiB `kDsv4RankLocalPerDeviceVramCeiling`" is under 50 MB.
Five configurations were tried; at the edge it goes non-deterministic. Raising
the ceiling 128 MiB did not resolve it and was reverted. Do not nudge the
constant — work out where the ~3 GiB between the ceiling and the card's 24 GiB
is actually going.

### 5. Prefetch window — the reference's `LVLLM_GPU_PREFETCH_WINDOW=1`

One layer of expert upload overlapped with the previous layer's compute. Worth
about the kernel's 4 s once 2 and 4 have shrunk the upload.

### Not worth doing

**A faster expert GEMM.** The premise was that the kernel is 5x off the
reference's implied efficiency. That was true at 21.7 s; after coalescing it is
4.02 s, so 5x is worth ~3 s. It has fallen out of the top five.

**Anything on the CPU expert path for prefill.** 13.0 GFLOP/token against a
Broadwell AVX2 ceiling of ~1.3 TFLOP/s caps CPU prefill near 100 tok/s however
well it is written. 0095 measured a real 2.18x there and it was still the wrong
placement.

## Traps that cost real time

**The traced arm does not exercise the production attention path.** Enabling
`--layer-hash-trace` disables the fused attention command in both arms. Two
structural blockers were invisible to the exactness gate because of this. Always
run one untraced arm before believing a page-major candidate.

**`mhc_post_seconds` is mostly the MoE wait, not mHC work.** Device mHC kernel
time is ~1e-4 s against ~2 s of reported `mhc_post`: the transition blocks on
the stream and the stream is waiting for the routed-expert callback. Read
`device_moe_runtime.routed_cpu_seconds` for the real term.

**A KV block admits one operation at a time.** It refuses a reservation or a
mutation while any device lease is outstanding. This is what forces a page onto
the unfused attention command and costs ~13 s at 677 tokens.

**Read the reference stack before designing.** `/home/rodrigo/Developer/Lvllmds4-x`
is a local checkout with `bench/launch.sh`, `bench/results.txt`, and
`vllm/model_executor/layers/fused_moe/routed_experts.py`. Four environment
variables in that launch script set the whole design direction. This handover's
author spent a full cycle designing from first principles before reading it, and
`docs/experiments/0055` had already written the headline — *"the largest gap
between the two stacks is prefill, not decode, and it is a re-execution
defect"* — on 5 August.

**Power-of-two row strides alias in L1.** A byte-permutation kernel measured
0.29 GB/s because all 32 rows of a block landed in one cache set. Blocking the
transpose was 4x.

## Reproducing the measurements

```bash
cmake -S . -B build-pagemajor -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_NCCL=ON
cmake --build build-pagemajor --target strata-deepseek-run strata-tests -j4

# throughput at one prompt length, per-phase breakdown printed
PAGES='8192' PROMPT_WORDS=420  scripts/run_dsv4_prefill_sweep.sh   # ~677 tokens
PAGES='8192' PROMPT_WORDS=1630 CONTEXT=8192 scripts/run_dsv4_prefill_sweep.sh

# against what main used to do, with generated-token equality
REPETITIONS=1 PAGE_TOKENS=8192 PROMPT_WORDS=420 \
  scripts/run_dsv4_page_major_prefill_ab.sh

# full-model comparison; page-major itself is bit-exact, the GPU expert
# kernel is not, and the harness reports which gates changed
PAGE_TOKENS=4 scripts/run_dsv4_page_major_prefill_correctness.sh
```

Each arm is about two minutes of model load plus the prefill window. Say the
arm budget before launching anything longer.

## Correctness contract as it now stands

Bit-exact, gated: mHC slot interleaving against sequential rows; page-major
execution against page 1 (at pages 4 and 64, prompts of 8, 52 and 144 tokens);
transformed-shard addressing against the canonical kernel; the transformed
kernel against itself for determinism.

Not bit-exact, gated differently: the prefill routed-expert kernel is a
reassociation and is held to `dsv4_host_expert_fp4` within one BF16 mantissa
step of the row's largest oracle magnitude, at the production shape. Prefill
and decode consequently use different expert kernels. Routes, precision,
top-k, expert count, expert residency and mHC semantics are unchanged, and
decode is untouched throughout.

Any candidate that changes prefill must report: prefill seconds and tok/s at
677 and 2,612 tokens, the fixed/marginal split, per-phase breakdown, expert
H2D bytes, cache misses and evictions, decode checkpoint reads (must stay 0),
**decode tok/s unregressed**, RSS and per-GPU VRAM.
