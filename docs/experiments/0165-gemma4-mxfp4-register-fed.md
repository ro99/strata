# Experiment 0165 — Gemma 4 MXFP4 closes the register-fed full-model gate

Status: **POSITIVE FOR GEMMA 4 MXFP4 DECODE.** The single-shard checkpoint now
loads, its exact MXFP4 layout reaches the scalar FP4 oracle, explicit fragment
prepack is safe for every consumer, and the register-fed route is correct and
materially faster on the real model. This does not overturn experiment 0164's
negative DeepSeek V4 result: it bounds it to a host-MoE-dominated workload.

## Branch, model, hardware, and invariants

- Branch: `fix/gemma4-single-shard-regfed`, based directly on
  `main@6668743`.
- Checkpoint: `models/gemma4/model.safetensors`, one 19,531,513,296-byte
  shard, 1,598 tensors and 19,531,296,472 tensor bytes.
- GPU: one RTX 3090, PCI bus `00000000:82:00.0`, exposed as logical device 0
  with `CUDA_VISIBLE_DEVICES=1`; the second 3090 was idle.
- Operating point: the owner's production cap, **250 W and SM clock locked to
  1605 MHz** against a 2100 MHz maximum. No tuning was changed. These are
  production measurements, not unlocked kernel-gate measurements.
- Build: `build-release`, verified `CMAKE_BUILD_TYPE=Release` before the A/B.
- Memory ceiling: one 24 GiB card. The run peaked at about 19.84 GiB. Fragment
  prepack replaces the canonical bytes in place; it does not retain a widened
  or second weight copy.
- Rollback/control: `STRATA_REGFED_MATMUL=0`. W8A16 weights are never
  prepacked and retain their existing `OffsetPackedInt8` path.

## Stage 1 — issue #35

Five loaders assumed `model.safetensors.index.json`. A shared
`load_safetensors_index` helper now preserves indexed-checkpoint precedence and,
only when the index is absent, synthesizes a one-shard index from
`model.safetensors`' own header. Unit tests cover synthesis and precedence.

`strata-gemma4-run` opened the lone shard, loaded all 60 text layers plus the
vision weights, completed prefill and decode, and generated `Paris.`. The
equivalence harness now treats either the index or lone shard as a present
checkpoint; it no longer reports that a present checkpoint is absent.

## Stage 2 — correctness

### Scalar oracle

The loader admits both existing formats by exact tensor shape, never byte
count:

- W8A16: I32 `[N,K/4]`, BF16 `[N,K/32]`, logical-shape I64 `[2]` ->
  `OffsetPackedInt8`.
- MXFP4: U8 `[N,K/2]`, U8 `[N,K/32]`, no logical-shape tensor ->
  `Fp4E2m1Group32`, with `packed_columns=K/2`,
  `scale_columns=K/32`, and `group_size=32`.

The fused Gemma batch-1 decoder previously accepted W8A16 only. Its bounded
extension dispatches canonical MXFP4 through the existing scalar
`native_fp4_matmul_kernel`, including the same E4M3 activation rounding as
generic matmul. With `STRATA_REGFED_MATMUL=0`, the real checkpoint generated
coherent text: `Paris.` (`[50429, 236761]`). This is G1.

### Layout ownership and route comparison

The consumer audit found exactly two consumers of Gemma text projection
weights:

1. generic `CudaBackend::matmul`, used by prefill and the output head;
2. `CudaBackend::gemma4_decode_layers`, the fused batch-1 decoder.

No direct attention kernel reads those weights; attention consumes projection
outputs. Vision weights are plain BF16. Both text consumers now accept fragment
order, and both refuse a prepacked weight when the enabled/admissible
register-fed route is unavailable. The Gemma loader prepackages only MXFP4,
only when the process-global switch is enabled. W8A16 remains canonical.

The route-vs-route unit test uses separate, byte-identical uploads at the real
Gemma shapes `[21504,5376]` and `[5376,21504]`. It compares scalar and
register-fed results under the existing `1e-4` relative accumulation-order
bound. A second resident-decode test prepackages all seven projections and
requires exactly seven `fp4_register_fed` census events.

Full-model G2 used the same prompt, weights, seed, and temperature zero:

| Arm | Text | Token IDs | Scalar FP4 census | Register-fed FP4 census |
|---|---|---|---:|---:|
| Scalar | `Paris.` | `[50429,236761]` | 1,230 | 0 |
| Register-fed | `Paris.` | `[50429,236761]` | 0 | 1,640 |

First divergence index: **none**. The census totals differ because the
18-row prefill becomes two M<=16 chunks in fragment order, whereas the scalar
route records one dispatch per caller.

G3 passed against a new format-specific MXFP4 scalar fixture: 1,080 layer
hashes, generated IDs, and answer all matched. The indexed W8A16 checkpoint
keeps its original fixture and index-first selection. G4 passed:

```text
strata-tests                  Passed  75.91 s
strata-sim-smoke              Passed   0.11 s
strata-equivalence-gemma4     Passed  23.73 s
100% tests passed, 0 failed
```

## Stage 3 — instantiate the cost model first

Hypothesis: register-fed FP4 reduces the GPU/HBM weight-read term. It adds a
one-time load prepack and per-call activation-fragment/split-K work, while
checkpoint I/O and steady activation H2D/D2H volumes remain unchanged.

The scalar profiling arm used a 31-token rendered counting prompt and produced
32 tokens, hence 31 batch-1 decode steps. The first step is reported
separately. Over the 30 steady steps:

| Term | Total | Per token |
|---|---:|---:|
| Wall decode | 5.601255 s | 186.708 ms |
| GPU kernel service | 5.549512 s | **184.984 ms** |
| Activation H2D | 0.000988 s | 0.033 ms |
| Activation D2H | 0.014223 s | 0.474 ms |
| Residual host/serial wall | about 0.0365 s | about 1.22 ms |

Therefore `argmax_r` is **GPU kernel/HBM service**, 99.1% of steady decode
wall time. The projection plus output-head weight volume is 18,377,146,368
bytes per decode step. This is the term the mechanism reduces, so the speed A/B
was admitted. The first scalar decode step was 188.286 ms.

### Measurement budget

The cheaper mechanism-only experiment was rejected because experiments
0142/0148 had already established the kernel result; only a full-model run
could answer MIX-2. Fresh processes are required because fragment layout is
chosen at load. The cached load plus short prefill was about 20 seconds of
fixed setup; the measured steady window was about 1.7 seconds register-fed and
5.6 seconds scalar, roughly 12:1 and 3.6:1 fixed-to-measured respectively.
Keeping the prompt at 31 tokens minimized the known-bad prefill term. Six arms
finished in about 2 minutes 45 seconds, below the projected 4–6 minutes.

## Interleaved A/B

Order was register-fed/scalar, scalar/register-fed, then
register-fed/scalar. Every arm used the identical prompt, 32-token cap,
temperature zero, and seed. Primary metric excludes the first batch-1 decode
step.

| Arm | Steady runs, ms/token | Median | Range/spread | First-step median |
|---|---|---:|---:|---:|
| Register-fed | 55.498, 55.467, 55.286 | **55.467** | 55.286–55.498 / 0.212 | 57.820 ms |
| Scalar | 186.757, 186.767, 186.655 | **186.757** | 186.655–186.767 / 0.111 | 188.649 ms |

Result: **3.3670x faster**, a 131.290 ms/token median reduction. The difference
is more than 600 times the larger observed absolute spread, so it is not a
within-variance result.

Every register-fed arm recorded 13,530 `fp4_register_fed` and zero scalar FP4
routes. Every scalar arm recorded 13,120 scalar FP4 and zero register-fed
routes. All six arms produced the same 32 greedy token IDs and coherent
counting continuation; first divergence index is none.

## Separate defect: prefill is not batching weight reads

Before the run the expected prefill/decode per-token ratio was below 0.25,
because a page should amortize every weight read across its rows. It measured
1.464 register-fed and 1.245 scalar. The generic scalar grid indexes each input
row independently and rereads weights; fragment order merely chunks the 31-row
page through M<=16 kernels. This is an implausible number and therefore an open
prefill batching defect, not evidence for or against batch-1 decode. No prefill
throughput claim is made here.

## Conclusion

Gemma 4 supplies the operating regime DeepSeek V4 did not: dense MXFP4 weights
fully resident on one GPU and decode dominated by their GPU service. The same
register-fed FP4 primitive that was end-to-end irrelevant behind DeepSeek's
host MoE is correct and materially valuable here. MIX-2 is therefore
**positive for Gemma 4 MXFP4 decode and remains negative for DeepSeek V4
decode**.

Raw ignored artifacts are under:

- `results/gemma4-regfed/correctness-r1/`
- `results/gemma4-regfed/profile-scalar/`
- `results/gemma4-regfed/ab-capped-250w-1605mhz/`
