# Experiment 0063 — reference SM86 attention/KV insertion contract

Status: **accepted as a component contract; not integrated into the live
runtime and not a throughput win.**

The installed Lvllmds4-x v2.3.8 production operator establishes two different
Ampere cache formats from Strata's current compact host cache:

- main DS-MLA rows use a block-major 256-token page with 576 data bytes and
  eight scale bytes per token, 584 bytes/token and 149,504 bytes/page;
- learned-index rows use 128 E4M3 bytes and one FP32 scale, 132 bytes/token and
  33,792 bytes/page.  The reference launch has
  `use_fp4_indexer_cache=False`; Strata's current 68-byte FP4 index row is not
  the reference-exact device format.

The production fused query RMSNorm/RoPE plus main-KV insert measures
**0.023851 ms median**, and the default FP8 index insert measures
**0.012114 ms median**, across 11 interleaved SM86 samples.  These operations
are prerequisites, not an explanation for Strata's 71.140 ms attention term.
The retained result is the explicit `lvllmds4x-sm86-attention-kv-v1` layout,
admission calculator, production-byte fixture generator, and dependency-light
C++ fixture consumer.  The live runtime still uses the old row-major
FP8/BF16 plus FP4-index cache and has not changed performance.

## Lineage and bounded decision

- base: `d22f3be docs: accept the reference mHC component contract`
- branch: `exp/dsv4-reference-attention-kv-contract`
- external source checkout: `/home/rodrigo/Developer/Lvllmds4-x` at
  `691cc0ae2056dc07ed23e4bc4f1dac25b7582f77`
- installed external package: `lvllmds4-x 2.3.8`
- target model: `DeepSeek-V4-Flash-0731`, 43 layers
- target device for the operation gate: RTX 3090, SM86
- target context: 32,768 tokens

The predeclared operating-point model remained:

```text
tau = max_r W_r / B_r + sum_serial

current total median:          244.812 ms/token
complete MoE:                  114.667 ms
routed CPU median:             106.869 ms
routed CPU best:                89.226 ms
standalone routed CPU:          87.879 ms
maximum median CPU opportunity: 18.990 ms
serialized non-MoE path:       130.145 ms
attention:                      71.140 ms
mHC pre + post:                 53.061 ms
target:                         98--100 ms/token, >=10 tok/s
```

- **Hypothesis.**  The production reference byte/arithmetic contract for the
  smallest query/KV insertion boundary is reproducible on SM86, inexpensive,
  and fits full-context admission.  If it is not, the complete device-resident
  reference route is blocked before runtime construction.
- **Target term.**  This gate selects a prerequisite for removing persistent-KV
  and host-handoff work from the 130.145 ms serialized non-MoE path.  It does
  not claim that isolated insertion reduces that term.
- **Primary metric.**  Production-shape fused main/index insertion time and
  agreement with raw production cache bytes.
- **Correctness gate.**  Exact query output and padding behavior; exact
  main-cache scale and BF16 RoPE bytes; production E4M3 bytes authoritative at
  measured conversion tie points; decoded NoPE/index error within their
  declared FP8 bounds; exact-mode rejection of a non-256-row page.
- **Memory ceiling.**  The existing 0.95 per-GPU admission and 216 GiB host
  ceiling.  The entire 32,768-token reference cache plus compressor/index state
  must be accounted before runtime work.
- **Bottleneck resource.**  The current defect is `sum_serial`: host ownership
  and cross-engine handoffs serialize 130.145 ms.  CPU routed DRAM is not
  reopened; the maximum remaining median CPU opportunity is only 18.990 ms.
- **Effects on other resources.**  A future integration adds 151.228 MB of
  persistent HBM across the three devices and tiny insertion SM work.  It must
  remove decode-time host KV gathering and link traffic, leave routed CPU
  bytes/semantics unchanged, add no NVMe dependency, and preserve BF16/FP8
  contracts.  This component-only probe changes none of those runtime terms.
- **Rollback condition.**  Reject before integration if production bytes
  exceed the declared FP8 error bounds, scale/RoPE bytes differ, exact mode
  falls back to FP4 or another page shape, or full-context admission exceeds
  any device ceiling.  None fired.

## Cheapest falsifier and fixed-cost ratio

The source/layout audit preceded executable work.  It immediately found that
the previous 120,148,480-byte compact-cache statement described Strata's own
583-byte row-major main rows and FP4 index rows, not the physical reference
device allocation.  This was cheaper than loading the 178 GB model or building
a runtime adapter.

The executable gate then loaded no model weights.  It validated the real
checkpoint configuration, used the production TP2 rank shape (32 live query
heads padded to 64), position 32,767, 512/64/128 semantic dimensions, and
256-token pages.  Fixed import/device setup dominated the approximately
11-second process; the measured 2,200 operation calls occupied only tens of
milliseconds.  A full-model arm was rejected because no architecture adapter
had yet passed this prerequisite.

## Source audit: the two physical formats

The SM86 main path is called by
`DeepseekV4Attention._fused_qnorm_rope_kv_insert` and writes:

```text
page data plane:
  256 * (448 E4M3 NoPE bytes + 64 BF16 RoPE values) = 147,456 bytes
page scale plane:
  256 * (7 UE8M0 exponent bytes + 1 zero pad)         =   2,048 bytes
page total:                                               149,504 bytes
```

The indexer compressor is constructed with `use_fp4_cache=False` on the
reference RTX 3090 launch.  Its default page is:

```text
page data plane:  256 * 128 E4M3 bytes = 32,768 bytes
page scale plane: 256 * one FP32 scale =  1,024 bytes
page total:                              33,792 bytes
```

Relevant external sources:

```text
vllm/models/deepseek_v4/attention.py
vllm/models/deepseek_v4/compressor.py
vllm/models/deepseek_v4/sparse_mla.py
vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py
csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu
tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py
```

## Numerical-contract finding

The external test source calls exact equality between the fused production
cache and a decomposed Triton quantize/insert oracle.  The installed v2.3.8
SM86 operator does not satisfy that assertion on the target-context fixture:

| Region | Differing bytes | Maximum code distance |
|---|---:|---:|
| NoPE E4M3 payload | 12 / 448 | 1 |
| seven UE8M0 scales | 0 / 7 | 0 |
| scale padding | 0 / 1 | 0 |
| BF16 RoPE payload | 0 / 128 | 0 |

Every differing production byte was one E4M3 code below the decomposed byte.
The fused kernel uses NVIDIA's saturating E4M3 conversion intrinsic; the
decomposed path uses the Triton integer encoder.  The difference occurs only at
conversion tie boundaries.  Decoded production NoPE error was 0.03125 under a
0.0625 block bound.  Query output was bit exact, every padded query value was
zero, and the RoPE tail was bit exact.

The production fused bytes are therefore the accepted contract.  Silently
substituting the nominal decomposed exact-byte oracle would fail to reproduce
the stack actually used for the 10.2 tok/s control.  This is an explicit narrow
contract decision, not a general relaxation of correctness.

## Interleaved operation measurement

Command:

```bash
CUDA_VISIBLE_DEVICES=1 \
  /home/rodrigo/Developer/Lvllmds4-x/venv/bin/python \
  scripts/probe_dsv4_reference_attention_kv.py \
  --model models/dsv4f \
  --fixture results/deepseek-v4-reference-attention-kv/reference-attention-kv.bin \
  --output results/deepseek-v4-reference-attention-kv/reference-attention-kv.json \
  --repetitions 11 --calls-per-sample 100 --warmup-calls 10
```

The arms alternated `main,index` then `index,main` on every repetition.

| Component | Median | Min--max |
|---|---:|---:|
| fused query RMSNorm/RoPE + main KV insert | **0.023851 ms** | 0.023480--0.026235 ms |
| FP8 index insert | **0.012114 ms** | 0.011182--0.013107 ms |

No baseline speedup is claimed.  The result only shows that these insertion
nodes are not themselves a material fraction of a 98--100 ms target step.

Phase/resource record:

| Item | Result |
|---|---:|
| admission | source/calculator only; no full cache allocated by this probe |
| load | 0.094137 s; 8,422,672 initial H2D bytes |
| prefill | not applicable, 0 s |
| warm-up | 0.007270 s, 10 calls/component |
| measured decode window | 11 interleaved samples, 100 calls/sample |
| decode H2D | 0 bytes |
| decode D2H | 0 bytes during measurement; 283,184-byte fixture afterward |
| synchronization | one CUDA-event synchronization per 100-call sample; no per-call host handoff |
| attention / mHC / complete MoE / routed CPU | not executed by the component probe |
| filesystem/NVMe reads after probe start | 0 bytes |
| filesystem writes | 286,720 bytes, fixture/report only |
| swaps | 0 |
| RSS | 1,339,637,760 bytes |
| Torch peak allocation / reservation | 21,138,432 / 23,068,672 bytes |
| measured-device VRAM delta | 67,108,864 bytes |

## Full-context admission

The Python probe and independent C++ implementation agree exactly:

| Allocation | Bytes |
|---|---:|
| sliding main cache | 12,857,344 |
| compressed main cache | 103,456,768 |
| FP8 learned-index cache | 22,708,224 |
| cache payload | **139,022,336** |
| compressor FP32 state | 11,862,016 |
| index-compressor FP32 state | 344,064 |
| total cache + persistent state | **151,228,416** |

This is 18,873,856 bytes above the previous 132,354,560-byte Strata compact
KV/state total.  It is a required precision/layout correction, not hidden
workspace growth.

At the current repeating `[0,0,1,1,1,2,2,2]` layer schedule:

| Device slot | Payload | Persistent state | Total |
|---:|---:|---:|---:|
| 0 | 33,662,976 | 3,031,040 | **36,694,016** |
| 1 | 70,051,840 | 3,522,560 | **73,574,400** |
| 2 | 35,307,520 | 5,652,480 | **40,960,000** |

The largest reservation is 70.166 MiB on a 24 GiB RTX 3090.  All three are
well below their explicit 0.95 VRAM ceilings and the existing 768 MiB/device
workspace reserve.  No host fallback, precision substitution, or NVMe tier is
required.

## Retained implementation and gates

Retained:

- `Dsv4ReferenceKvLayout` for the main FP8/BF16 and FP8-index block-major
  formats;
- strict 256-row exact-mode admission with no FP4 fallback;
- independent full-context byte accounting;
- dependency-light C++ decoders and malformed-page checks;
- a real-target-format external fixture generator and native fixture consumer;
- unit coverage for byte offsets, scales, padding, decoded values, and the
  complete 32,768-token allocation.

Not retained or claimed:

- no change to the live `Dsv4KvCache` format selection;
- no device allocator, compressor kernel, sparse-attention adapter, hidden
  buffer, CPU-MoE boundary, reduction, or CUDA graph;
- no full-model teacher-forcing/generation claim before a layer adapter exists;
- no inference throughput change.

The native fixture gate reports:

```text
accepted: true
main_max_absolute: 0.03125
index_max_absolute: 0.03125
query_padding_nonzero: 0
target_context_kv_bytes: 151228416
allocated_blocks: 1450
```

## Decision and next gate

Accept the SM86 production insertion/layout contract.  In a future reference
device mode, Strata must use the 132-byte FP8 index format explicitly and fail
if it cannot; it must not silently reuse the current FP4 index cache.

Do **not** wire the full runtime from this operation result alone.  The next
stage dependency is a real target-format layer differential at representative
early, middle, and late layers.  It must capture the compressor reduction,
compressed page bytes, index query/key bytes and scales, selected positions,
attention output, and residual boundary.  Only if that gate accepts the
layer-level contract may the complete fixed-buffer hidden + persistent KV +
attention + accepted mHC + stream-ordered CPU-MoE + GPU-reduction chain be
built.  CPU scheduler optimization and isolated projection migration remain
out of scope.

## Reproduction

Generate and measure the external production fixture with the command above,
then validate it without Lvllmds4-x:

```bash
build/strata-dsv4-attention-kv-contract-probe \
  results/deepseek-v4-reference-attention-kv/reference-attention-kv.bin
```

Ignored raw artifacts are under:

```text
results/deepseek-v4-reference-attention-kv/
```
