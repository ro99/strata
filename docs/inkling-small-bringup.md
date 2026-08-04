# Inkling-Small-NVFP4 bring-up

Status: text backbone and MTP heads implemented and gated against the pinned
checkpoint. Vision and audio towers are validated by the reader but not
executed. This is a correctness runtime; no throughput claim is made from it.

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

Six forward tokens on a five-token prompt, 3 GPUs idle, host execution:

```
  moe routed experts   15901 ms  71.9%   <- argmax_r
  attention             2540 ms  11.5%
  moe shared experts    2531 ms  11.4%
  output head            540 ms   2.4%
  dense mlp              417 ms   1.9%
  short conv              66 ms   0.3%
  embedding                3 ms   0.0%
```

Decode 0.84 tok/s, prefill 0.25 tok/s, RSS 23 GiB, 3.37 GiB of routed-expert
weights touched per token against 154 GiB resident on disk.

Reading this against the cost model in
`research/moe-tiered-memory-decode-optimization.md`:

- `argmax_r` is **host compute**, not I/O. The checkpoint sits in page cache on
  a 251 GiB machine, so `W_nvme` is zero in steady state and the expert term is
  scalar dequantize-and-multiply.
- The top-6 selection means a token touches 2.2% of the routed set. That ratio
  is what makes the set cacheable at all, and it is measured, not assumed.
- The 64 GiB of idle VRAM across three devices is the obvious next term to
  attack, but that is a separate hypothesis with its own gate: it reduces the
  compute term, which is the current bottleneck, so the sign is right, but the
  transfer cost of streaming 3.37 GiB/token over PCIe has to be measured before
  any design is committed to. Do not carry these constants to a different
  context length or batch shape.

## What is not done

- **Vision and audio towers.** The reader validates their tensors; nothing
  executes them. The vision path is a 4-layer hierarchical patch MLP and the
  audio path is a discrete-mel embedding, both inferable from their shapes.
- **Speculative decode.** The MTP heads load and propose, and the runtime scores
  their acceptance, but a proposal is never consumed. Skipping a backbone step
  requires KV and convolution rollback on rejection, and the acceptance rate
  this path measures is the gate for deciding whether that is worth building.
  Enabling `enable_mtp_speculation` today costs time and changes nothing.
- **Device execution.** Everything runs on the host. The CUDA backend, expert
  cache and placement plan that the GLM and Laguna runtimes use are wired for
  inventory purposes but not for execution.
