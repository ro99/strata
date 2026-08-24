# Experiment 0167 — Inkling MXFP4 scalar execution passes; register-fed integration stops at the cost gate

Status: **MXFP4 SCALAR EXECUTION PROVEN; REGISTER-FED SUBSTITUTION NOT
ADMITTED.** `models/inkling` is the 30-shard MXFP4 checkpoint, not the older
Inkling NVFP4 checkpoint. Its exact format now loads and produces coherent
text while the existing NVFP4 reader and executor remain distinct. The first
real-model profile found serial routed-expert staging/H2D to be `argmax_r`, not
GPU FP4 kernel service, so no weight was fragment-prepacked and no register-fed
A/B was built.

## Branch, model, hardware, and invariants

- Branch: `feat/inkling-mxfp4`, based directly on `main@a46502d`.
- Checkpoint: `models/inkling`, pinned to
  `mlx-community/Inkling-Small-mxfp4`; 30 indexed shards, 1,508 tensors,
  140,271,331,492 indexed tensor bytes, and 140,271,531,888 shard-file bytes
  (130.638 GiB). The index SHA-256 is
  `a62dd17e5d1bbfa2a5722dfb5ce4e708d8b953ae22bc51af7761dadf7f9ebdde`.
- Build: `build-release`, verified as `CMAKE_BUILD_TYPE=Release` before the
  real run.
- Devices: logical CUDA 0 and 1, the two RTX 3090s. Both stayed at the owner's
  production point, 250 W and clock-locked to 1605 MHz. No cap or clock was
  changed. An earlier diagnostic accidentally selected logical devices 1 and
  2, a 3090 plus the RTX 5060 Ti; it was discarded and the recorded profile
  was rerun on 0 and 1.
- Memory ceiling: the live device allocation was 2.590 GiB of resident spine
  plus 37.320 GiB of routed-expert cache, 39.910 GiB total. The 130.638 GiB
  checkpoint cannot be device-resident on this pair, so it is correctly
  reported as cache/staging dependent.
- Correctness gate: exact checkpoint identity and tensor shapes, canonical
  MXFP4 fused MoE versus the generic scalar FP4 route on identical target-shape
  uploads, coherent greedy generation, preservation of the old NVFP4 format,
  and `make check`.
- Rollback/stop: keep canonical layout and stop before `prepack_fragment` if
  scalar text is incoherent or GPU kernel/HBM service is not `argmax_r`. The
  second condition fired.

## Checkpoint and executor support

The reader selects either pinned format by exact checkpoint extent and source
identity; it does not infer precision from a filename or silently fall back:

| Checkpoint | Matrix representation | Routed-expert organization |
|---|---|---|
| Existing `thinkingmachines/Inkling-Small-NVFP4` | BF16 spine; routed experts transition from BF16 to E2M1/E4M3 group-16 plus global scales | interleaved gate/up w13 plus down; separate MTP shard |
| New `mlx-community/Inkling-Small-mxfp4` | every matrix except routers is E2M1/E8M0 group-32 | separate gate, up, and down; no MTP shard |

The MXFP4 checkpoint stores packed codes as U32 tensors, with eight logical
columns per U32. Their byte view is still canonical low-nibble-first E2M1:
`[N,K/8] U32` therefore supplies `N*K/2` code bytes. Scale tensors are
`[N,K/32] U8` E8M0. Every module is checked by dtype and tensor **shape**, not
byte count, before it becomes a `CudaWeightEncoding::Fp4E2m1Group32`
descriptor with:

```text
packed_columns = columns / 2
scale_columns  = columns / 32
group_size     = 32
```

This validation covers embeddings, the output head, every attention
projection, dense MLPs, two shared experts, and all 256 routed experts in each
of 40 sparse layers. Routers, norms, relative projections, and short
convolutions retain their declared BF16/F32 types. The checkpoint's per-expert
`gate_scale` and `out_scale` arrays are also required to contain exact identity
values; a non-one value fails load rather than changing semantics silently.

Inkling's routed decode uses `CudaBackend::enqueue_moe`, not only generic
`cuda.matmul`. The cache now stages separate canonical MXFP4 gate/up/down
slices through its existing registered scratch and dispatches the established
scalar fused-MoE route. The NVFP4 deinterleave, E4M3 group-16 scales, global
scales, BF16 transition, and MTP behavior remain in separate format branches.
MXFP4 MTP requests fail explicitly because this checkpoint has no MTP weights.

`strata-inspect` now recognizes the exact MXFP4 index. Both index-only and
all-30-header scans report `status=ok`, architecture
`inkling_small_mxfp4`, and 458 group-32 quantized modules.

An Inkling-shape CUDA test uses logical gate/up `[2048,4096]` and down
`[4096,2048]` on identical uploads, comparing the fused canonical MoE result
against the three generic scalar FP4 matmuls. It requires the established
`1e-4` arithmetic bound and exactly one `moe_fp4_e2m1_group32` dispatch. The
existing NVFP4 fixtures remain separate; checkpoint-dependent NVFP4 tests skip
honestly when `models/inkling-s` is absent.

## Scalar real-model oracle

The cheapest real run that could establish both coherence and the resource
gate used five prompt tokens and only one decode forward. The first generated
token comes from the prefill logits, so a two-token generation cap yields one
steady decode forward:

```text
build-release/strata-inkling-probe --model models/inkling \
  --devices 0,1 --no-warm \
  --prompt 'The capital of France is' --tokens 2
```

Load took 2.24 s, prefill took 4.77 s, and the one decode forward took 0.55 s.
Fixed load plus prefill setup was 7.01 s, 12.8 times the measured decode
window; the whole run was about 7.6 s. A synthetic route test was cheaper but
could not identify the real model's bottleneck, while a longer token window
was unnecessary after the first forward separated staging from kernels.

The exact continuation was coherent:

```text
Paris.
```

The route census was:

```text
fp4_e2m1_group32=1298
moe_fp4_e2m1_group32=480
fp4_register_fed=0
```

This establishes the canonical scalar MXFP4 oracle on the real checkpoint.

## Cost model and binding verdict

For the one decode forward, `tau = max_r W_r/B_r + Sigma_serial` instantiated
as:

| Measured term | ms/forward | Share of 546.7 ms phase sum |
|---|---:|---:|
| Routed-expert phase | **495.7** | **90.7%** |
| cache-miss staging/H2D | 466.0 | 85.2% |
| upload stream wait | 108.0 | 19.8% |
| upload allocation | 34.0 | 6.2% |
| all recorded CUDA kernels | **30.0** | **5.5%** |
| attention | 26.1 | 4.8% |
| shared experts | 11.2 | 2.0% |
| output head | 5.3 | 1.0% |

The forward touched 2.99 GiB of routed weights and staged 1.18 GiB of cache
misses in 466 ms, only 2.54 GiB/s. Allocation, copying, upload synchronization,
kernel launch, and collection are serial inside the routed phase; the measured
495.7 ms service is the dominant term. Thus `argmax_r` is **serial
routed-expert staging/H2D**, not GPU kernel/HBM service.

Register-fed FP4 can reduce only the 30 ms kernel term and would additionally
permute each newly staged cache entry. Even the impossible upper bound of
deleting every recorded CUDA kernel is only 546.7 / (546.7 - 30.0) =
**1.058x**, before prepack overhead and while including kernels the substitution
does not target. Per the staged gate, the register-fed path is not admitted.
No production weight is prepacked, so there is no token first-divergence claim
or performance A/B; presenting either would compare the same scalar route
twice.

## Plausibility defect

Before the run, a batched prompt page was expected to cost below 0.25 of
batch-1 decode per token because it should amortize routed-weight reads. The
probe's explicitly selected `prefill_page_tokens=0` path instead took 954
ms/prompt token against about 547 ms/decode forward, a ratio of **1.74**. This
confirms that the default probe path is token-at-a-time rather than a production
prefill batch. It is a separate scheduling/measurement defect and carries no
prefill throughput claim.

## Final correctness gates

The final suite includes:

- U32 safetensors dtype parsing and byte-width coverage;
- exact real-checkpoint identity, extent, early attention, late routed-expert,
  router, embedding, and output-head shapes;
- index-only and all-30-header `strata-inspect` validation;
- scalar E2M1/E8M0 decoding with low-nibble-first packing;
- the real Inkling-shape fused-MoE comparison on identical uploads;
- coherent real-checkpoint generation with scalar and fused-scalar census
  counters nonzero and register-fed zero;
- all existing Inkling NVFP4 source, decoding, and executor fixtures.

Final `make check` after all code and documentation changes is recorded below:

```text
check-layers: 0 total violation(s)
check-symbols: 0 total violation(s)
strata-tests                  Passed  83.88 s
strata-sim-smoke              Passed   0.11 s
strata-equivalence-gemma4     Passed  24.88 s
100% tests passed, 0 failed
```

## Conclusion

Inkling-Small MXFP4 is now an exact supported checkpoint format and its scalar
CUDA executor is coherent on the real model. The previous Inkling NVFP4 format
remains supported rather than being reinterpreted. Unlike dense, resident
Gemma 4, this 130.638 GiB MoE model is dominated by routed-expert staging on
the available device memory. The cheapest real profile therefore falsified
register-fed integration before an in-place layout change was built.

Raw ignored artifacts are under:

- `results/inkling-mxfp4/scalar-cost-3090.log`
- `results/inkling-mxfp4/tests.log`
