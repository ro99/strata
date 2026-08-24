# Experiment 0157 — FP4 across layers and experts, with real activations

Status: **BOTH REMAINING FP4 VALIDATION ITEMS CLOSED.** The kernel is exact
across **40 real fixtures** — 5 layers x 4 expert indices x 2 production shapes
— reading production weights *and* a checkpoint-derived activation, with a
worst case of **7.53e-07** relative error against a **133x** margin on the
declared gate.

Operating point: single RTX 3090, 350 W, unlocked clocks. FP4 files only.

## What experiment 0156 left open

0156 closed the real-weights gate on **one expert in one layer**, with
synthetic activations. Two gaps remained: a single fixture cannot expose a
layout assumption that varies with layer or expert index, and only one of the
two operands was production data.

## Fixture 1: layers and expert indices

Layers 0, 10, 21, 32 and 42 span the model's full depth of 43; expert indices
0, 1, 128 and 255 span the 256-expert range including both ends. Tensors are
resolved through `model.safetensors.index.json` with Strata's own
`parse_safetensors_index`, so fixtures that live in different shards are
located the same way the runtime locates them.

| Shape | Fixtures | Worst relative error | Median | Margin on the 1e-4 gate |
|---|---:|---:|---:|---:|
| `gate_up_w1` (K=4096) | 20 | **7.53e-07** | 4.25e-07 | **133x** |
| `down_w2` (K=2048) | 20 | **2.68e-07** | 1.74e-07 | **373x** |

**All 40 admitted, all 40 pass.** No layer or expert index behaves differently.

Two internal consistency checks, neither of which was tuned for:

- **Error tracks accumulation depth.** `gate_up_w1` accumulates over K=4096 and
  `down_w2` over K=2048; the median error ratio is 2.44x against a
  `sqrt(K)` prediction of 1.41x — same direction and order, which is what an
  FP32 summation-order delta should look like. A layout or scale bug would not
  respect K.
- **Output magnitude grows with depth**, roughly 1.7 at layer 0, 3.2 at layer
  21 and 4.2 at layer 42. That is the residual stream growing, and it is
  evidence the fixture is exercising genuinely different real data per layer
  rather than re-reading one tensor.

## Fixture 2: real activations

The activation is now built from checkpoint tensors and the production
normalisation: a real `embed.weight` row, RMS-normalised, scaled by the real
`layers.L.ffn_norm.weight` for the same layer as the expert under test. Only
8 KiB of the 1.06 GB embedding is read, by slicing the tensor descriptor rather
than materialising it.

**This is stated precisely because it matters: it is a checkpoint-derived
activation, not a captured forward-pass activation.** It skips attention, so it
is not the exact vector the expert would see at inference. What it does carry,
and what synthetic stimulus does not, is the model's real per-channel scale
structure from `ffn_norm`, real value distribution, and realistic magnitude.
The output magnitudes above — order 1 to 4, against roughly 60,000 for the
synthetic stimulus of 0155 — are the visible consequence.

The remaining step to a true activation is a captured forward pass, which
requires runtime instrumentation and belongs with the operation/layer fixture in
MIX-1.

**The numerical consequence of real activations:** 0156 measured 0.0 relative
error with small-integer synthetic activations, because every product was a
dyadic rational and FP32 accumulation was exact. Real activations are arbitrary
BF16 values, so summation order now matters and the error becomes the expected
1e-07-class delta. **That is the contract's already-declared summation-order
delta, not a regression** — and it is the honest number, because the 0.0 was an
artefact of the stimulus.

## Gate verdict

| Gate | Required | Result | Verdict |
|---|---|---|---|
| Multi-layer fixture | exact across model depth | 5 layers, worst 7.53e-07 | **PASS** |
| Multi-expert fixture | exact across expert range | indices 0, 1, 128, 255 | **PASS** |
| Real activations | production-derived operand | `embed` + real `ffn_norm`, RMSNorm | **PASS** |
| Numerical contract | only a summation-order delta | <= 7.53e-07, 133x margin | **PASS** |
| E8M0 admission | no inadmissible codes | 40/40 admitted | PASS |

## What this does not establish

- **Not a captured forward-pass activation**, as above.
- **Not an operation- or layer-level fixture.** Routing, the shared expert, the
  residual path and publication are untouched; this is the expert matmul in
  isolation.
- **No full-model teacher-forcing or generation oracle.** Both remain required
  by the contract and belong with integration.
- Nothing about FP8.

## Exact next action

FP4 kernel and fixture work is complete. Everything remaining is integration:
the operation/layer fixture, the full runtime oracle, and MIX-1's one-copy
mixed dispatch with route census, admission wired to `admit_e8m0_scales`,
load-time prepack cost, VRAM accounting and graph integration.

**MIX-1 remains blocked on the FP8 track reaching an accepted F8-2**, owned by
the concurrent session.
