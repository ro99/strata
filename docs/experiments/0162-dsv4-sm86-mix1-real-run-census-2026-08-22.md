# Experiment 0162 — the route census on a real run, and what it falsified

Status: **MIX-1 CENSUS OBSERVED ON PRODUCTION.** A real DeepSeek V4 TP=2 run
emits the census, and it falsifies a load-bearing assumption: **`matmul_impl`
is not on the per-token decode path.** It is called exactly 129 times — 43
layers x 3 — whether the run generates 4 tokens or 16. Wiring the accepted
register-fed kernels there would not touch production decode at all.

Two secondary results: **E8M0 admission ran on the full 156 GB checkpoint and
rejected nothing**, confirming experiment 0156's prediction on real data; and
`unsupported` is **0**, so no dispatch reached the branch that used to be a
silent FP4 fallback.

Operating point: both RTX 3090s at the production point, 250 W and 1605 MHz
locked. **The cap is irrelevant here** — the census counts route *choices*, not
time.

## Run

`scripts/dsv4_route_census.sh`, in tmux session `mix1-census`, using the
recorded working invocation of experiment 0081:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 \
build-release/strata-deepseek-run --model models/dsv4f \
  --devices 0,1 --host-memory 216G --vram-fraction .95 \
  --max-context 256 --max-new 4 --device-resident-runtime --host-routed-moe \
  --host-attention-threads 28 --route-census RESULTS.json
```

Load: 95 s total, 156,885,843,968 resident bytes staged at 2.04 GiB/s,
admission 2.65 ms. Prefill 9 of 9 tokens in 4.81 s; generation 4 tokens in
0.56 s; **decode checkpoint reads 0 bytes**, satisfying the zero-NVMe-decode
contract.

A first attempt without `--host-routed-moe` and `--host-memory` failed at
`layers.2.ffn.shared_experts.w1` with *"CUDA weight arena is exhausted;
refusing per-weight allocation fallback."* That is the no-hidden-fallback rule
behaving correctly and is recorded rather than worked around.

## The census

| Route | Count |
|---|---:|
| `fp8_tensor_page` | **129** |
| `fp4_e2m1_group32` | 0 |
| `fp8_e4m3_block128` | 0 |
| `plain_bf16_matvec`, `plain_generic`, `packed_int8_group32`, `packed_offset_int`, `nvfp4_group16` | 0 |
| **`unsupported`** | **0** |

## What it falsified

`129 = 43 x 3` exactly, which is one dispatch per layer for the whole run
rather than per token. The cheapest falsifier settled it: **re-running with 16
new tokens instead of 4 produced the identical 129.** A per-token path would
have scaled with 12 forward passes against 24.

So `matmul_impl` is exercised during load and warm-up and then not again.
Production decode runs through other kernels entirely — the rank-local
executors, the graph-captured attention chain, and `CudaBackend::enqueue_moe`,
which launches `packed_int4_moe_gate_up_kernel` and
`packed_int4_moe_down_kernel` for the FP4 routed experts.

**This matters because MIX-1's requirement is that eligible FP4 use the
accepted F4 path.** Experiments 0160 and 0161 wired the census and admission
into `matmul_impl` and the checkpoint loader, which is the correct place for
*admission* — it runs at load, and it demonstrably ran over the whole 156 GB.
But it is the wrong place for the *dispatch substitution*. Had the census not
been built first, the accepted kernel would have been wired into `matmul_impl`,
every unit test would have passed, and production decode would have been
completely unaffected — a silent no-op change.

Two further observations, recorded but not built upon:

- `fp4_e2m1_group32` is 0 partly because `--host-routed-moe` runs routed
  experts on the host. A device-MoE configuration would change this, and the
  arena exhaustion above must be resolved before that can be measured.
- Only 3 dispatches per layer appear, not the 5 attention projections. `wo_a`
  is converted to `Plain` BF16 at load, and the others reach the device through
  `dsv4_prepare_attention` and the tensor-page path rather than `matmul_impl`.

## Admission on real weights

The load staged 156,885,843,968 bytes with `dsv4_admit_e8m0_scales_for` on
every FP4 region and produced **no admission error**. Experiment 0156 measured
the checkpoint's E8M0 codes at [119, 125] over 1.41 billion scale bytes and
predicted silent admission; that prediction is now confirmed against the whole
checkpoint through the production loader, which 0161 explicitly listed as
outstanding.

## Gate verdict

| Gate | Required | Result | Verdict |
|---|---|---|---|
| Census observable on production | dump from a real run | 129 dispatches, all routes named | **PASS** |
| No hidden fallback | `unsupported` = 0 | 0 | **PASS** |
| Admission on real checkpoint | no false rejection | 156 GB staged, zero errors | **PASS** |
| Zero NVMe decode | 0 bytes | 0 bytes | PASS |
| **Accepted F4/F8 paths dispatched** | **MIX-1's core requirement** | **not wired; `matmul_impl` is load-only** | **OPEN** |

## What this does not establish

- No throughput claim. This run exists to observe routes, and its prompt and
  token counts are deliberately tiny.
- The census covers `matmul_impl` only. `enqueue_moe`, the rank-local
  executors and the attention chain are **not** yet censused, so production
  decode's route distribution is still unobserved.
- Device-MoE was not measured, because the weight arena exhausts in that
  configuration with these flags.

## Exact next action

1. **Extend the census to the paths decode actually uses** — `enqueue_moe`
   first, since that is where the FP4 routed experts are dispatched, then the
   rank-local executors and the attention chain. Until those are censused,
   "every production choice is observable" is not satisfied.
2. **Then substitute the accepted paths there**, not in `matmul_impl`.
3. Resolve the device-MoE arena exhaustion so an FP4 device dispatch can be
   observed at all.
