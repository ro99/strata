# Experiment 0164 — the register-fed FP8 shared-expert kernels, built but not wired

Status: **MIX-1 SUBSTITUTION STARTED, NOT COMPLETE.** The three kernels the
shared-expert substitution needs now exist and build: an in-place fragment
prepack, a fused register-fed gate/up, and a split-K reduce whose swiglu is
reproduced bit-for-bit from the incumbent. A public prepack entry point is
gated by a contract test. **Nothing dispatches them yet and no numerical
equivalence has been demonstrated.**

## Why the shared expert is the target

Experiment 0163 established that the shared expert is the **only** per-token
CUDA dispatch DeepSeek V4 decode makes — 43 per layer per forward pass, through
`deepseek_fp8_gate_up_kernel` and `deepseek_fp8_down_kernel` inside
`enqueue_dsv4_host_moe_impl`. Its shapes are exactly the family the accepted
W8A16 work measured:

| Tensor | Dtype | Shape | Scales |
|---|---|---|---|
| `ffn.shared_experts.w1` | F8_E4M3 | [2048, 4096] | E8M0 [16, 32] |
| `ffn.shared_experts.w3` | F8_E4M3 | [2048, 4096] | E8M0 [16, 32] |
| `ffn.shared_experts.w2` | F8_E4M3 | [4096, 2048] | E8M0 [32, 16] |

The incumbent is a **scalar matvec**: one block per output row, one weight byte
per lane, no tensor cores. Experiment 0143 measured that shape at about 13.5%
of DRAM and 68% issue-bound; the accepted register-fed path reaches 83–86% of
the local read roofline on the same format.

## What was built

**`dsv4_fp8_fragment_prepack_kernel`** — canonical `[N][K]` E4M3 to `m16n8k16`
A-fragment order, one `uint4` per lane per 32-column pair group.

**`dsv4_fp8_regfed_gate_up_kernel<kAccs>`** — gate and up share the activation,
so one pass over the hidden vector feeds both weight streams. `kAccs`
independent accumulators keep the FP32 chain short over K=4096 and are
tree-combined, which is the association fix from the F8 track. Only activation
column 0 is live at M=1; the other seven MMA columns are zero and cost nothing
extra.

**`dsv4_fp8_regfed_swiglu_kernel`** — sums the split-K partials and applies
swiglu. The rounding, the finite check, the clamp order and the **BF16 SiLU
table lookup** are reproduced exactly from the incumbent. This path must be
numerically indistinguishable from it, not merely close, so the table lookup
was copied rather than replaced with a computed SiLU.

**`CudaBackend::dsv4_fp8_prepack_fragment`** — permutes a weight in place. The
fragment order **replaces** the canonical device layout rather than adding a
second buffer, which is what one-copy residency requires; the transient scratch
is released immediately. This matters here because the weight arena is already
tight enough that a 4 GB tier reservation exhausts it.

## Gate

A contract test asserts the entry point accepts a block-128 FP8 weight of valid
shape and **refuses** three things rather than silently doing the wrong thing:
an FP4 weight (wrong element size), a row count that is not a multiple of the
16-row MMA tile (the fragment map is undefined for a partial tile), and an
invalid weight (reported, not dereferenced).

`make check` passes: 317/330 unit tests, 13 skipped, all three suites green.

## What this does not establish

This is deliberately explicit, because the kernels look finished and are not.

- **Nothing dispatches them.** `enqueue_dsv4_host_moe_impl` still calls the
  incumbent scalar kernels. The new code is unreachable at runtime.
- **No numerical equivalence has been shown.** The test gates the entry point's
  contract, not its output. A byte-level readback of a prepacked weight is not
  currently possible from a test — `download_buffer` works on `CudaBuffer`, not
  `CudaWeight` — so the permutation has not been compared against an
  independently computed fragment order.
- **No throughput measurement.** The 83–86% figure is from experiment 0159's
  standalone probe on the attention shapes, not from this kernel on the shared
  expert.
- The `down` projection has no register-fed counterpart yet; only gate/up.

## Exact next action

1. **Prove equivalence before wiring.** Add a gate/up entry point and compare
   its output against the incumbent kernel on the real shared-expert shapes,
   with the same weights and hidden vector. The swiglu was copied to make this
   comparison exact rather than approximate, so the gate should be equality, not
   a tolerance.
2. **Then wire it behind a flag** in `enqueue_dsv4_host_moe_impl`, with the
   route census distinguishing incumbent from register-fed, and A/B both on a
   real run.
3. The `down` projection follows the same pattern once gate/up is proven.
