# Experiment 0156 — the FP4 kernel on real DeepSeek V4 checkpoint weights

Status: **REAL-WEIGHTS GATE CLOSED.** The register-fed W4A16 kernel is
**bit-exact, 0.0 relative error**, on production bytes read straight from the
DeepSeek V4 checkpoint for **both** production shapes, at the real 6-expert
dispatch width. Throughput is unchanged from synthetic stimulus. A census of
**1.4 billion real E8M0 scale bytes** shows the admission window is correct and
will never fire on this checkpoint.

Operating point: single RTX 3090, 350 W, unlocked clocks. FP4 files only.

## Why this was owed

Contract section 3 lists "real weights/activations, operation/layer fixtures,
and the full runtime oracle" as **independent** gates. Every number in
experiments 0136-0155 came from xorshift-synthetic codes and scales. A decoder
and prepack that are exact on synthetic data can still be wrong on production
data if the checkpoint's layout, dtype or value distribution differs from what
was assumed.

## What the checkpoint actually contains

`models/dsv4f`, 48 shards, 156 GB. From `config.json`:

- `n_routed_experts` **256**, `num_experts_per_tok` **6**,
  `moe_intermediate_size` **2048**, `hidden_size` **4096**,
  `n_shared_experts` **1**.

This is a **third independent confirmation of top-k 6**, after
`kDeepSeekV4ExecutionContract` and `kDsv4RankLocalTopK`. It also shows
experiment 0155 was **conservative**: real per-layer dispatch is 6 routed plus
1 shared = **7** matrices, not 6.

The checkpoint's top-level `quantization_config` declares FP8 E4M3 with UE8M0
block-128 scales, which initially reads as "there is no FP4 here". The dtype
census resolves it — the checkpoint is genuinely mixed, exactly as the campaign
contract says:

| Tensor | Dtype | Shape | Track |
|---|---|---|---|
| `layers.L.ffn.experts.E.w1.weight` | **I8** | [2048, 2048] | **FP4** — packed E2M1, two codes per byte |
| `layers.L.ffn.experts.E.w1.scale` | **F8_E8M0** | [2048, 128] | **FP4** — one byte per group of 32 along K |
| `layers.L.ffn.experts.E.w2.weight` | **I8** | [4096, 1024] | **FP4** |
| `layers.L.ffn.experts.E.w2.scale` | **F8_E8M0** | [4096, 64] | **FP4** |
| `layers.L.attn.wkv.weight` | F8_E4M3 | [512, 4096] | FP8 — the concurrent session's track |

**The expert weights are exactly the shapes and byte counts this campaign has
been measuring.** `w1` is 2048 x 2048 = 4,194,304 packed bytes with 2048 x 128
= 262,144 scale bytes, which is `N*K/2` and `N*K/32` for `N=2048, K=4096`. `w2`
is `N=4096, K=2048`. Both match `W_FP4` = 4,456,448 exactly.

## A defect the byte-count guard did not catch

The first real-weights run reported `real_weights=true` for **both** shapes
while only `w1` had been requested. `w1` and `w2` carry **identical byte
counts** — 4,194,304 codes and 262,144 scales — so a size check cannot tell
them apart, and `down_w2` was silently fed `w1`'s bytes reinterpreted as
`[N=4096, K=2048]`. The oracle still passed, because it reinterpreted them the
same way.

That is precisely the silent substitution the contract forbids. Fixed two ways:
the tensor name is now derived per shape (`.w1` for `gate_up_w1`, `.w2` for
`down_w2`), and the guard checks the **declared shape**, `[N, K/2]` for codes
and `[N, K/32]` for scales, not only the total size.

## Results

Real tensors `layers.0.ffn.experts.0.w1` and `.w2`, M=1, 6 experts per launch,
split-K 4, median of three independent processes.

| Shape | Source tensor | Real weights | Admitted | Max rel. error | max output | GB/s |
|---|---|---|---|---:|---:|---:|
| `gate_up_w1` | `layers.0.ffn.experts.0.w1` | yes | yes | **0.0** | 13.781 | **636.4** |
| `down_w2` | `layers.0.ffn.experts.0.w2` | yes | yes | **0.0** | 10.875 | **668.5** |

**Bit-exact against a double-precision oracle computed from the same real
bytes**, in every run, on both shapes.

Throughput is indistinguishable from the synthetic result at the same operating
point — 635.4 / 668.4 in experiment 0155 against 636.4 / 668.5 here — which is
expected for a DRAM-bound kernel whose cost does not depend on weight values,
and confirms the synthetic stimulus was representative for timing.

Output magnitudes are small, 13.8 and 10.9, against roughly 60,000 for the
synthetic stimulus. That is the real value distribution: production E8M0 scales
sit far below the synthetic range, as the census below shows. A probe that only
ever sees synthetic magnitudes cannot notice this.

## The E8M0 admission window, validated against production data

Experiment 0155 added admission for E8M0 codes 0 and 255, which encode as `+0`
and `+inf` in BF16. Whether those codes actually occur was unknown. Census over
seven shards:

- **5,376 expert E8M0 scale tensors, 1,409,286,144 scale bytes**
- **Observed code range: [119, 125]**
- **Code 0: 0 occurrences. Code 255: 0 occurrences.**

The admissible window is [1, 254], so real data sits **entirely inside it** with
enormous margin. The admission check is a genuine guard against a malformed or
differently-quantized checkpoint, not a constraint on this one, and it will
never fire here.

Two secondary observations, recorded but **not** built upon:

- The real exponent range spans only seven values, 2^-8 to 2^-2. That is
  checkpoint-specific and must not become a design assumption; a decoder that
  only handled [119,125] would be wrong for any other checkpoint.
- The synthetic stimulus used codes 120-134, which is **wider** than production.
  Synthetic testing was therefore harder than reality on scale range, not
  easier.

## Gate verdict

| Gate | Required | `gate_up_w1` | `down_w2` | Verdict |
|---|---|---:|---:|---|
| Real weights, both production shapes | exact vs double oracle | **0.0** | **0.0** | **PASS** |
| Correct tensor per shape | shape-checked, not size-checked | `.w1` [2048,2048] | `.w2` [4096,1024] | **PASS** |
| E8M0 admission on real data | no inadmissible codes | 0 of 262,144 | 0 of 262,144 | PASS |
| Admission window vs checkpoint census | window covers real range | [119,125] within [1,254] | same | **PASS** |
| Throughput unchanged from synthetic | within variance | 636.4 vs 635.4 | 668.5 vs 668.4 | PASS |

**The real-weights gate is closed for the FP4 expert path.**

## What this does not establish

- **One expert, one layer.** `layers.0.ffn.experts.0` is a single fixture. The
  scale census covers 5,376 tensors, but the numerical gate does not.
- **Not an operation- or layer-level fixture.** The contract also requires
  operation and layer fixtures built from the target format, and a full-model
  teacher-forcing and generation oracle. This exercises the expert matmul in
  isolation, not a layer's routing, activation, shared expert, or residual path.
- **Activations are still synthetic.** Real weights are now used; the activation
  vector is still generated. A real activation trace at the expert boundary is
  a separate fixture.
- Nothing about FP8. `wkv` E4M3 tensors belong to the concurrent session.

## Exact next action

1. **Widen the numerical fixture** across layers and experts — at minimum a
   sample spanning early, middle and late layers and several expert indices,
   since a single expert cannot expose a layout assumption that varies with
   layer.
2. **Capture real activations at the expert boundary** and re-run the oracle, so
   both operands are production data.
3. **Then the operation/layer fixture and the full runtime oracle**, which are
   integration work and belong with MIX-1.

MIX-1 remains blocked on the FP8 track reaching an accepted F8-2, owned by the
concurrent session.
