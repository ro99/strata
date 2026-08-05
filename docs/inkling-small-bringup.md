# Inkling-Small-NVFP4 bring-up

Status: text backbone and MTP heads implemented and gated against the pinned
checkpoint, executing on three GPUs with the routed expert set in host RAM.
Vision and audio towers are validated by the reader but not executed.

## Pinned source facts

| Property | Value |
|---|---:|
| Checkpoint | `thinkingmachines/Inkling-Small-NVFP4` |
| Index SHA-256 | `2e3a00b15c5498687538e56f5adcb5784373079074636afb90ecefa1ca9baeee` |
| Indexed tensors | 1,360 |
| Weight shards | 9 main + 1 MTP |
| Indexed tensor payload | 170,733,074,592 bytes |
| Weight shard files | 170,733,233,632 bytes |
| Source precision | NVFP4 routed experts (layers 3–41), BF16 everywhere else |
| Strata handling | Preserve native extents; no additional quantization |

Read on 2026-08-03 from the local copy in `models/inkling-s`.

## Why the semantics had to come from the reference

Three properties of this architecture are load-bearing for correctness and none
of them is inferable from the checkpoint alone. They were taken from the
published reference implementation and pinned in
`include/strata/model_adapter.hpp`.

**There is no rotary embedding anywhere.** Position enters only as a learned
per-head bias added to the pre-softmax logits:

```
scores[q][k] = (q · k) / head_dim + B_rel[head][q_pos - k_pos] + mask
```

`B_rel` comes from a fourth attention projection `wr_du` producing 16 values per
head, projected through a per-layer bank of distance profiles. The bank is
`[16, 512]` on the 35 sliding layers and `[16, 1024]` on the 7 full-attention
layers, so the extent is a property of the layer type. Beyond its extent the
bias is zero and a global layer becomes content-only. A runtime that added RoPE
here, or that used one extent for all layers, would load without complaint.

**Attention divides by `head_dim`, not by its square root.** Q and K are
per-head RMS-normed to unit norm before the dot product, so the reference scale
is `1/128`. Using `1/sqrt(128)` is an 11x error on every logit.

**The two shared experts are routing sinks.** The gate has 258 rows: 256 routed
plus 2 shared. Selection is `sigmoid(logit) + correction_bias` over the routed
range only — the sinks never compete. The selected routed logits and both sink
logits are then renormalized together by log-sigmoid followed by a softmax over
that joint set of 8, and scaled by `route_scale * global_scale`. The sinks
therefore absorb probability mass, which is what lets a confident routed
decision down-weight the shared experts. Note the asymmetry the tests pin: the
correction bias steers selection but never reaches the weights, which are
derived from raw logits.

Two further storage facts were established by measurement rather than assumed:

- **NVFP4 here is the ModelOpt convention, not compressed-tensors.** The
  per-expert FP32 scale is a *multiplier* (`amax / (6 * 448)`), so
  `w = e2m1 * e4m3_block * scale2`. Reconstructing one routed expert row gives
  rms 0.131 against 0.127 for the unquantized BF16 experts of layer 2; the
  reciprocal rule would give 9e5. The direction is pinned by that comparison.
- **`w13` rows interleave gate and up** as `[g0, u0, g1, u1, ...]`, for dense,
  shared and routed projections alike. Reading it as two contiguous halves
  mixes gate rows into up rows and still produces plausible-looking text.

## Layer structure

- 42 layers, hidden 4096, GQA 32 queries over 8 KV heads, head_dim 128.
- Full attention on layers 5, 11, 17, 23, 29, 35, 41; the other 35 slide over a
  512-token window.
- Four depthwise causal convolutions per layer, kernel 4 with a residual, on the
  attention keys, the attention values, the attention output and the MLP output.
  They carry local mixing, so a decode step needs the last three rows of each
  stream as state alongside the KV cache.
- Layers 0–1 are dense (intermediate 16384). Layers 2–41 are MoE: 256 routed
  experts, top-6, 2 shared, intermediate 2048. Layer 2's routed experts ship
  BF16; layers 3–41 are NVFP4 group-16.
- Long-context temperature `tau = 1 + 0.1*log(max(1, (pos+1)/128000))` scales the
  normalized queries and the relative bias, on global layers only.
- Logits are produced over the padded 201,024-row table, divided by the muP
  width multiplier 16, then truncated to the unpadded 200,058.
- 8 MTP depth blocks; depths 1 and 3 are full-attention, the rest slide.

## Correctness gates

`make check` runs these; the checkpoint-backed ones skip when
`models/inkling-s` is absent.

- Contract validation rejects a changed top-k, routed scale, sink rule,
  interleaving, relative extent, attention scale or expert group size.
- The reader validates all 1,360 tensors against the pinned manifest, and
  rejects a local relative extent presented as a global one.
- One routed expert is dequantized and its magnitude compared against the
  unquantized layer, which is what fixes the NVFP4 scale direction.
- The tokenizer round-trips mixed case, contractions, paths, whitespace and
  non-ASCII text, and groups digits in runs of at most three.
- Greedy decoding of `"The capital of France is"` yields `" Paris."`.
- Teacher forcing reproduces its logits bit for bit across runs, which is what
  catches a stale KV row or an unreset convolution history — generation alone
  would still look plausible.

## Measured operating point

All figures decode-only; a cold prefill otherwise sets every share. Median of
three interleaved repetitions on a 5-token prompt.

### Host baseline

```
  moe routed experts   1646 ms/token  79.4%   <- argmax_r
  attention             165 ms/token   7.9%
  moe shared experts    163 ms/token   7.8%
  output head            55 ms/token   2.6%
  dense mlp              29 ms/token   1.4%
```

0.58 tok/s. Three identical repeats of a deterministic prompt took 12.27,
11.94 and 12.17 s wall. Greedy decoding routes to the same experts and reads
the same bytes every time, so a repeat that does not get faster rules out
page-cache misses: the term is host compute at 2.05 GiB/s of scalar
dequantize-and-multiply, not storage.

### Three GPUs plus host RAM

```
  moe routed experts     90 ms/token  67.9%   <- still argmax_r
  attention              23 ms/token  17.0%
  moe shared experts      9 ms/token   6.8%
  short conv              5 ms/token   3.5%
  output head             3 ms/token   2.0%
```

**9.19 tok/s, 15.9x over the host baseline.** RSS 153.6 GiB, page cache
165 GiB, expert cache hit rate 83.0%, H2D 9.3 GiB/s.

Design: the ~12 GiB spine is uploaded once and round-robins over the devices;
the 154 GiB NVFP4 routed set stays in host RAM behind a per-device LRU sized
from what the spine leaves (16.1 + 17.4 + 10.9 = 44.4 GiB, 28% of the set).
Routed experts and sinks each go up as one batched device command. They cannot
share a command because a batch is single-encoding and the sinks are BF16.

Two reuses avoid new kernels. The shared NVFP4 kernel divides by its global
scale where Inkling's ModelOpt scale multiplies, so the descriptor carries the
reciprocal. Gate and up are de-interleaved during staging because the device
MoE wants them separate.

### Two defects the profile named

Neither was predicted by the design; both came out of attributing a number
that did not match the shape the mechanism should have.

- **Staging appeared to run at 2.2 GiB/s** against a link rated far higher,
  which the cost model calls a serialization bug rather than a bandwidth
  limit. Splitting the upload showed only 2.06 s of 6.95 s was CUDA at all
  (alloc 0.36, copy 0.86, wait 0.84). The true H2D rate was 9.3 GiB/s and the
  other 70% was the host faulting cold pages of a 160 GiB mapping. Host memory
  exceeds the model, so the expert set is now faulted in at load and
  steady-state decode does not touch NVMe. That one change is most of the
  15.9x; without it the device waits on storage, not on PCIe.
- **Dense MLP and the output head were uploaded but never called.** They were
  still running on the host at 86 ms and 172 ms per token. Now 5 ms and 9 ms.

### What the remaining bottleneck is

Routed experts are still `argmax_r` at 67.9% of a 108 ms step, and staging
misses are the larger part of that. Raising the hit rate is the next term.

The 83% hit rate is measured on a 5-token prompt repeated three times and is
flattered by that reuse; the cache never evicted once during it. A long,
diverse workload will sit nearer the 28% capacity ratio. **Do not carry this
number to another operating point** — re-measure on a decode trace of the
workload actually being optimized, which is what the placement simulator is
for.

## What is not done

- **Vision and audio towers.** The reader validates their tensors; nothing
  executes them. The vision path is a 4-layer hierarchical patch MLP and the
  audio path is a discrete-mel embedding, both inferable from their shapes.
- **Speculative decode.** The MTP heads load and propose, and the runtime scores
  their acceptance, but a proposal is never consumed. Skipping a backbone step
  requires KV and convolution rollback on rejection, and the acceptance rate
  this path measures is the gate for deciding whether that is worth building.
  Enabling `enable_mtp_speculation` today costs time and changes nothing.
- **Prefill batching.** Prefill dispatches one token at a time, so it pays a
  device round trip per token and per layer instead of amortizing weight reads
  across rows. Prefill at 7.9 tok/s against decode at 9.2 tok/s is the ratio
  the design says should be far apart, which makes it a defect rather than a
  cost; it has its own branch.
- **Expert prefetch.** Routing for layer N+1 is known once layer N's router
  runs, so a miss could be staged while the current layer computes. Today the
  stage is serial with the command that needs it.
- **MTP on device.** The depth blocks still run on the host path.
