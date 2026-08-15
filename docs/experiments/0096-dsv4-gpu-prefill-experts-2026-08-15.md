# Experiment 0096 — routed experts on the GPU for prefill

Status: **accepted.** Physical-device prompt processing executes its routed
experts on the GPU, reading the transformed shards the resident host arena
already holds. At a 677-token prompt prefill is 64.02 s against 101.05 s for
the same page path reading canonical weights out of the checkpoint, and
against roughly 75 s for the CPU-expert page path 0095 accepted.

This experiment also corrects 0095's framing. Page-major execution is the
prerequisite and stands; grouping a page's rows by expert *on the CPU* is
superseded and its code is removed.

## Why this was the wrong term to begin with

0095 optimised CPU routed-expert execution because it was `argmax_r`. It was,
and the 2.18x it measured is real. But the reference stack does not execute
prefill's experts on the CPU at all, and the arithmetic says it cannot: 13.0
GFLOP per token of routed work against a Broadwell AVX2 ceiling of roughly 1.3
TFLOP/s with no VNNI and no AMX (0050) caps CPU prefill near 100 tok/s however
well it is written.

`bench/launch.sh` in the user's own checkout of the reference stack says what
it does instead:

```
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=128     # >=128 rows => routed experts on GPU
LVLLM_GPU_PREFETCH_WINDOW=1              # one layer of upload overlapped
--enable-chunked-prefill --max-num-batched-tokens 8192
--enable-prefix-caching
```

`routed_experts.py` gates on exactly that, and `moe_runner.py` dispatches:
capturing a CUDA graph goes to `cpu_decode`, a batch of 128 or more goes to
`gpu_prefill`, anything else to `cpu_prefill`. The CPU MoE is their decode
path only.

## The recipe, and the one thing it settles

`_process_wna16` hands `lk_moe.MOE_WNA16` raw host pointers to the
checkpoint-native weights and a config carrying `stride = 32`; the engine
copies them into its own arena, and `clean_weights_after_loading` then deletes
vLLM's tensors. `gpu_prefill` takes activations, routing and a stream — **no
weight pointers**.

So the engine keeps **one host copy in one layout, and its GPU kernel reads
that layout**. There is no canonical/transformed duality to resolve, because
they never created one.

That decides a question this repository could not answer from first
principles. Strata's single copy is the 156 GB transformed arena the host
expert kernels read; 251 GB of RAM cannot hold the canonical 147 GB beside it.
So the device reads the transformed layout.

## What it cost to find out

Two rejected attempts are recorded so they are not retried:

- **Reading canonical weights from the checkpoint.** This is what the device
  MoE did when first reached, because `Dsv4ResidentWeightStore::find` misses
  for routed experts when the arena is transformed. 69.9 GB and 27.4 s of file
  reads for a 677-token prompt, with the same weights already in host memory.
- **Rebuilding canonical bytes on the host.** Exact — the transform is a
  permutation whose only non-injective step duplicates each group-32 scale
  across two group-16 halves — but 1.21 GB/s single-threaded, or 58 s of
  rebuild per page. Its first version was 0.29 GB/s because walking the
  canonical side reads the transform with a 32-byte stride and the canonical
  row stride is a power of two, so all 32 rows of a block land in one L1 set.
  Blocking the transpose was 4x and still not enough. Removed.

## Mechanism

`Fp4E2m1Tiled32` is one intermediate-dimension TP shard of one expert as the
arena holds it: w13 packed, w13 scales, w2 packed, w2 scales, concatenated,
output rows blocked by 32, each group-32 scale duplicated across two group-16
halves. `execute_moe_page` acquires shards instead of canonical triplets, and
the checkpoint leaves the prefill path — 69.9 GB of reads becomes 0.57 GB.

The kernel had to be reshaped, and this is the part that carries a numerical
consequence. Addressing the transformed layout from the canonical
decomposition is bit-identical and was measured that way, but it is **2.87x
slower**: the transform interleaves 32 output rows, the canonical block owns
one, so a warp uses a thirty-second of every sector it fetches. Giving a warp
the whole transform block and each lane one output row makes every fetch one
contiguous 32-byte run, and takes the kernel from 21.67 s to 4.02 s.

That reassociates the canonical 256-partial tree into a per-lane sequential sum
in increasing column order. Same terms, same per-term
`fma(input * fp4_value, scale, accumulator)`, different order. **It is gated
against the scalar oracle, not against the canonical kernel**: at the
production shape, three rows through the page command against
`dsv4_host_expert_fp4`, every output within one BF16 mantissa step of the row's
largest oracle magnitude — the granularity both sides round to anyway. The
kernel is separately required to be deterministic bit for bit.

The weight row tile also went from 8 to 32. A 677-row page gives about 33 rows
per touched expert, so a tile of 8 read every expert four times over; 64 spills
the accumulators and measured worse.

## Measured

677-token prompt, one page, devices 1,2, three repetitions of the accepted
state and one of each intermediate.

| arm | prefill | tok/s | MoE | upload | kernel | attention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| canonical from checkpoint | 101.05 s | 6.7 | 62.18 | 48.30 | 7.56 | 34.62 |
| transformed shards | 93.46 s | 7.2 | 54.81 | 24.18 | 21.67 | 34.33 |
| + row tile 32 | 86.25 s | 7.8 | 47.49 | 26.81 | 13.82 | 34.43 |
| + coalesced kernel | 61.55 s | 11.0 | 22.69 | 12.98 | 4.02 | 34.55 |
| **accepted, median of 3** | **64.02 s** | **10.6** | 25.94 | 16.11 | ~4.0 | 33.81 |

The three repetitions were 64.02, 79.97 and 63.51 s. **All of the spread is the
expert upload** — 16.11, 32.38, 15.50 s for the identical 73.8 GB — while
attention held to 33.81, 33.46, 33.79. See the open defect below.

Scaling, 677 to 2,612 tokens, on the state before the attention changes:

| term | 677 ms/tok | 2,612 ms/tok | ratio |
| --- | ---: | ---: | ---: |
| expert H2D **bytes**/token | 109.0 MB | 35.3 MB | **0.32** |
| MoE upload wait | 19.18 | 10.35 | 0.54 |
| MoE kernel | 5.94 | 5.61 | 0.94 |
| attention | 51.03 | 55.44 | 1.09 |
| prefill | 90.91 | 86.58 | 0.95 |

The amortisation works where it is plumbed: expert traffic per token falls to
0.32x because the distinct experts a page touches saturates toward 256. It
reaches 12% of the cost. Fitted as fixed-per-chunk plus marginal-per-token,
Strata is about 4 s fixed and **85.1 ms/token marginal**; the reference is
10–13 s fixed and **1.14 ms/token**. That ratio, not throughput at one prompt
length, is what predicts the curve.

## Against the accepted baseline

Page 1 is what `main` executes: token-major, routed experts in the CPU shards.
677-token prompt, devices 1,2:

| arm | prefill | tok/s | attention | routed CPU | MoE |
| --- | ---: | ---: | ---: | ---: | ---: |
| page 1, CPU experts (`main`) | 89.61 s | 7.555 | 22.25 | 54.37 | 2.77 |
| page 8192, GPU experts | 64.55 s | 10.488 | 34.22 | 0.14 | 25.99 |

**1.388x**, and the two arms generate the same token.

Attention is larger in the page arm because a page cannot use the fused
attention command — it holds the layer's KV device leases until the collect,
and a KV block refuses to be appended to while any lease is outstanding, so
the second row of any page fails. That is 0095's finding and it still stands.

The full-model hash comparison no longer passes, by construction: the routed
experts execute on a different kernel. The first divergence is exactly where it
should be — position 0, layer 0, `ffn_output`, the first routed expert, with
all seven preceding operations bit-identical. At an 8-token prompt the top two
logits are 0.17 apart and the arms select differently; at 677 tokens they
agree.

## Open defect: the upload is bimodal

The same 73.8 GB took 15.50, 16.11 and 32.38 s across three runs of identical
code. `Dsv4ResidentWeightStore::stage` allocates `MAP_PRIVATE|MAP_ANONYMOUS`
with no NUMA policy, which experiment 0026 found and 0050 restated. Which half
of the 156 GB arena lands on the far node relative to each GPU's PCIe root is
then decided by first-touch during staging and varies per run. Binding the
arena per node is the obvious test and has not been run.

## Rejected in passing

**Page-locking the arena.** The guard refusing it ("host-routed experts are
CPU-only") is gone, because prefill now DMAs out of that arena. It still does
not fit: registering 156 GB costs about 72 MB of device-side mapping, and the
window between an arena large enough for the resident spine and a total under
the 21.0 GiB rank-local admission ceiling is under 50 MB. Five configurations
were tried; at the edge it is non-deterministic. Raising the ceiling by a
documented 128 MiB did not resolve it either and was reverted. Worth about 11 s
and needs the reserve accounting examined, not a constant moved.

## Reproduce

```bash
cmake -S . -B build-pagemajor -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_NCCL=ON
cmake --build build-pagemajor --target strata-deepseek-run strata-tests -j4

PAGES='8192' PROMPT_WORDS=420 scripts/run_dsv4_prefill_sweep.sh
PAGES='8192' PROMPT_WORDS=1630 CONTEXT=8192 scripts/run_dsv4_prefill_sweep.sh
```

`make check` 100%, CUDA suite 307/308 with one opt-in skip.
