# Handover — DeepSeek V4 TP2 prompt throughput

Goal: get rank-local TP2 prompt processing to roughly the external reference's
numbers. Everything below is measured on the reference pair (devices 1,2) with
`models/dsv4f`.

`main` is at `20de4e9`. `make check` 100%, CUDA suite 307/308 with one opt-in
skip. Read `docs/experiments/0095` and `0096` before touching anything.

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

The host cost is candidate resolution: 640 entries per row (kWindow=128 sliding
plus kIndexTopK=512 compressed), each doing a table search plus an
`unordered_map` lookup into a page set that is **rebuilt from scratch every
row**, with fresh `leases`/`pages` vectors. 71.9M of those at 2,612 tokens.

Hoist the page set to per (layer, page). The obstacle is real and is why it was
not done: KV read leases held across rows block the next row's append (a block
refuses to be appended to while any lease is outstanding). The fix is to split
`attention_prepared` into "project and append" and "attend", and run the layer
as two loops — append every row's KV first, then attend them all against one
shared page set. That is exactly the shape `attention_page`'s `batch_cuda`
branch already uses for the Block cache, and causality is preserved because each
row's candidate list is bounded by its own position.

After that, the device 24.4 s wants one call per (layer, page) instead of per
(layer, row), which means a row dimension on `CudaDsv4PagedAttentionRequest`.
That is the bigger, later piece.

### 2. Expert upload NUMA placement — worth ~16 s of pure variance

Three runs of identical code moved the identical 73.8 GB in 15.50, 16.11 and
**32.38** s. `Dsv4ResidentWeightStore::stage` allocates
`MAP_PRIVATE|MAP_ANONYMOUS` with no NUMA policy; experiments 0026 and 0050 both
flagged this and it has never been fixed. Which half of the 156 GB arena lands
on the far node relative to each GPU's PCIe root is decided by first-touch and
varies per run. Bind the arena per node and re-measure. Cheap, and it also makes
every other measurement on this path trustworthy.

**Until this is fixed, do not compare total prefill seconds between runs.**
Compare per-phase terms; attention held to 1% across the same three runs.

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
