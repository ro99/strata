# Experiment 0166 — Laguna S 2.1 MXFP4 loads, but register-fed integration stops at the cost gate

Status: **MXFP4 SCALAR EXECUTION PROVEN; REGISTER-FED SUBSTITUTION NOT
ADMITTED.** The new Laguna checkpoint is MXFP4, not the older Laguna NVFP4
checkpoint. Its exact format now loads and produces coherent text while the old
NVFP4 format remains supported. A real-model scalar profile found routed-expert
service, staging, and host orchestration above GPU matmul time. GPU matmul
kernels are not `argmax_r`, so no fragment prepack or register-fed A/B was
built.

## Branch, model, hardware, and invariants

- Branch: `feat/laguna-register-fed`, based directly on `main@e5a4138`.
- Checkpoint: `models/laguna`, pinned to
  `olka-fi/Laguna-S-2.1-MXFP4`; 46 indexed shards, 72,865 tensors,
  68,350,039,552 indexed tensor bytes, and 68,359,318,472 shard-file bytes
  (63.665 GiB). The index SHA-256 is
  `9a3ce5a1e798b20f6fba133472c1e25a137ddf18c46354e8d88375d0fb19cbfb`.
- Build: `build-release`, verified `CMAKE_BUILD_TYPE=Release` before the real
  run.
- Devices: logical CUDA 0 and 1 are the two RTX 3090s; logical CUDA 2 is the
  RTX 5060 Ti. The 3090s were at the owner's production point, 250 W and
  clock-locked to 1605 MHz against a 2100 MHz maximum. No cap or clock was
  changed.
- Memory ceiling: 54,890,207,526 arena bytes (51.121 GiB), of which 6.901 GiB
  is pinned spine and 44.219 GiB is expert cache. The 63.665 GiB model exceeds
  aggregate resident weight memory, so it is correctly treated as
  cache/staging dependent rather than fully resident.
- Correctness gate: exact checkpoint identity and tensor shapes, canonical
  MXFP4 fused-MoE versus the existing scalar FP4 route on identical uploads,
  coherent real-model greedy output, and `make check`.
- Rollback/stop: retain canonical layout and stop before prepack if scalar text
  is incoherent or GPU kernel time is not the measured bottleneck. The second
  condition fired.

## Checkpoint and executor support

The reader now distinguishes the two pinned checkpoints by exact identity and
extent:

| Checkpoint | Routed expert layout | Quantized sparse layers |
|---|---|---:|
| Existing `poolside/Laguna-S-2.1-NVFP4` | E2M1 plus E4M3 group-16 scales and global scales | 1-39 |
| New `olka-fi/Laguna-S-2.1-MXFP4` | E2M1 plus E8M0 group-32 scales, no global scale | 1-47 |

The new checkpoint contains 36,096 packed expert projections: 47 sparse layers
x 256 experts x gate/up/down. Every mapping is checked by tensor shape, never
byte count:

- gate/up: packed U8 `[1024,1536]`, scale U8 `[1024,96]`, logical
  `[1024,3072]`;
- down: packed U8 `[3072,512]`, scale U8 `[3072,32]`, logical
  `[3072,1024]`.

They map to `CudaWeightEncoding::Fp4E2m1Group32` with
`packed_columns=columns/2`, `scale_columns=columns/32`, and `group_size=32`.
Attention, router, shared-expert, dense-layer, embedding, and output-head BF16
weights remain plain. The older NVFP4 descriptor, its layer-40 transition to
BF16, and its fused kernels are separate branches and were not replaced.

Laguna decode does not call generic matmul for its routed experts. It batches
the selected experts through `CudaBackend::enqueue_moe`, so the scalar executor
needed canonical MXFP4 gate/up and down kernels plus the same E4M3 activation
rounding as `native_fp4_matmul_kernel`. A target-shape CUDA test runs the fused
batch and the three generic scalar FP4 matmuls on the same uploaded weights and
requires the existing `1e-4` bound; the MXFP4 MoE census must increment exactly
once. No fragment order is involved.

`strata-inspect` validates either exact checkpoint. On the new checkpoint both
index-only and all-46-header scans report `status=ok`, architecture
`laguna_s_2_1_mxfp4`, and 36,096 MXFP4 modules.

## Scalar real-model oracle

The cheapest real run that could test both scalar coherence and the cost gate
used one repetition, a 47-token rendered prompt, and a 16-token generation cap:

```text
build-release/strata-laguna-profile --model models/laguna \
  --devices 0,1,2 --context 128 --max-new 16 --repetitions 1 \
  --prompt 'The capital of France is'
```

Load took 2.57 s, prefill 10.315 s, and decode 3.555 s. Fixed setup was 3.63x
the measured decode window; the complete run took about 16.4 s, so a longer
arm was unnecessary. The mechanism-only unit test was cheaper but could not
establish the real model's bottleneck.

The model generated 15 tokens at 4.220 tok/s:

```text
Okay, the user is asking for the capital of France. Let me start by
```

This is coherent and establishes the scalar MXFP4 oracle. The process exited
zero.

## Cost model and binding verdict

For decode, `tau = max_r W_r/B_r + Sigma_serial` instantiated as:

| Measured term | ms/token | Share of 236.98 ms wall |
|---|---:|---:|
| Routed MoE phase | **150.02** | 63.3% |
| routed experts, slowest device | 146.02 | 61.6% |
| miss staging, slowest slot | 65.05 | 27.4% |
| weight memcpy, slowest device | 63.82 | 26.9% |
| attention phase | 66.27 | 28.0% |
| stream synchronization wait | 19.78 | 8.3% |
| all matmul kernels, slowest device | **14.70** | 6.2% |
| flash-attention kernels | 7.55 | 3.2% |

The cache made 1,410 lookups/token with an 89.8% hit rate, but still missed 144
times/token and staged 229.50 MiB/token. The routed path serially acquires/cache
checks the selected weights, stages misses, synchronizes uploads, launches the
batch, and collects its output. The approximately 146 ms slowest-device expert
service is therefore the dominant phase; its roughly 65 ms staging and a
similar host acquisition/orchestration residual are each much larger than the
14.70 ms GPU matmul service.

Thus `argmax_r` is **routed-expert serial service/cache staging**, not GPU
matmul/HBM service. Register-fed FP4 targets only the latter and would add a
fragment permutation on every newly staged cache entry. Even the impossible
upper bound of deleting every measured matmul-kernel nanosecond is only
236.98 / (236.98 - 14.70) = **1.066x**, before prepack overhead and without
separating non-expert matmuls. Per the campaign gate, the register-fed
substitution is stopped before implementation. There is no register-fed/scalar
token comparison, first-divergence claim, or performance A/B because that
dependent stage was not admitted.

## Plausibility defect

Before the run, prefill was expected to cost below 0.25 of decode per token if
the page amortized expert weight reads across its rows. It measured 219.46
ms/token against decode's 236.98 ms/token, a ratio of **0.926**. The profile
also reports 47 prefill steps for 47 prompt tokens, zero cache hits, and 652.18
MiB of weight upload per step. Laguna prefill is consequently not batching the
prompt page. This is an open prefill defect, not evidence for the batch-1
register-fed decode hypothesis, and no prefill throughput claim is made.

## Final correctness gates

The real MXFP4 checkpoint test validates early and late expert projections and
proves layer 47 remains MXFP4. A separate target-shape CUDA test keeps the old
NVFP4 fused route distinct: it matches its canonical scalar route and records
`moe_nvfp4_group16=1` with `moe_fp4_e2m1_group32=0`. The original NVFP4 host
decoder fixtures and source-contract mutation tests remain green.

Final `make check` after all code and documentation changes:

```text
check-layers: 0 total violation(s)
check-symbols: 0 total violation(s)
strata-tests                  Passed  77.76 s
strata-sim-smoke              Passed   0.09 s
strata-equivalence-gemma4     Passed  24.00 s
100% tests passed, 0 failed
```

## Conclusion

Laguna S 2.1 MXFP4 is now a supported, exact checkpoint format and its scalar
CUDA executor is coherent on the real model. Existing Laguna NVFP4 support is
preserved. Unlike dense, fully resident Gemma 4, this 63.665 GiB MoE checkpoint
is dominated by routed-expert orchestration and staging on the available GPU
memory. The cheapest real profile falsified register-fed integration before an
in-place layout change could be built.

Raw ignored artifacts are under:

- `results/laguna-regfed/scalar-oracle/output.log`
- `results/laguna-regfed/scalar-oracle/exit-code`
