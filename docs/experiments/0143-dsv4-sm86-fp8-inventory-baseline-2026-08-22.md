# Experiment 0143 — SM86 FP8 inventory and W8A16 baseline

Date: 2026-08-22

Milestone: **F8-0 COMPLETE.** This experiment inventories the checkpoint and
runtime FP8 boundaries, establishes the independent SM86 ruler and W8A16
scalar baseline, and profiles the existing W8A8-style tensor-page control. It
does not implement QPN8, pass D-F8-GATE, or make an end-to-end throughput
claim.

## Predeclared experiment contract

- **Hypothesis:** the real FP8 census and per-M measurements identify the
  resource term a QPN8-derived path must reduce. If register-fed W8A16 cannot
  plausibly reduce that measured `argmax`, F8-1 must not begin.
- **Primary metrics:** full kernel/control time, exact checkpoint bytes,
  effective GB/s, same-session cold read ceiling, and resource/serial
  attribution at every protected M point and real tile boundary.
- **Correctness gate:** actual E4M3/E8M0 block-128 shapes and bytes; the scalar
  arm is an exact source-level replica of `native_fp8_matmul_kernel` with a
  BF16-rounded activation carrier; the eligible W8A8 control must be no worse
  at the BF16 publication oracle; unsupported SMs must fail.
- **Memory ceiling:** 512 MiB including the cold arena, ruler, scrub,
  activations, outputs, and all temporary/duplicate probe storage. Production
  receives no widened or duplicate weight.
- **Rollback:** no production dispatch is changed. An implausible ruler,
  format/boundary failure, or control regression stops at F8-0.
- **Bottleneck before measurement:** unknown. No mechanism or resource saving
  was claimed before the profile.
- **Cheapness:** static checkpoint/runtime census first, then an isolated
  microbenchmark. Setup is CUDA allocation and at most 112 MiB of replicated
  matrix data; each arm has three warmups and eleven samples. The complete 216
  process-arm matrix took minutes, not an end-to-end model run.

The production operating point was CUDA device 0, verified as the unlocked
350 W RTX 3090 at PCI 82:00.0 (`nvidia-smi` index 1), with the second RTX 3090
idle. The probe hard-checks SM86 and records the device name inline. CUDA
device 2, the RTX 5060 Ti, failed with `F8-0 baseline requires SM86`.

## Actual checkpoint census

`strata-inspect --model models/dsv4f --headers --json` resolved all 72,317
tensors across 48 shards and validated 35,718 quantized layouts: 35,328 MXFP4
and exactly **390 FP8 E4M3/E8M0 block-128 modules**. The index-to-module census
is in `results/dsv4-sm86-fp8-f8-0/fp8-modules.tsv`; its pattern counts are:

| Region | Modules | Shape `(N,K)` | Execution status |
|---|---:|---:|---|
| main `attn.wq_a` | 43 | `(1024,4096)` | active every layer |
| main `attn.wq_b` | 43 | `(32768,1024)` | active every layer |
| main `attn.wkv` | 43 | `(512,4096)` | active every layer |
| main `attn.wo_a` | 43 | `(8192,4096)` | active, grouped output projection |
| main `attn.wo_b` | 43 | `(4096,8192)` | active every layer |
| main shared `w1,w3` | 86 | `(2048,4096)` | active every layer |
| main shared `w2` | 43 | `(4096,2048)` | active every layer |
| ratio-4 `attn.indexer.wq_b` | 21 | `(8192,1024)` | active from the long-context index threshold |
| MTP five attention projections | 15 | same attention shapes | present and verified; execution disabled |
| MTP shared `w1,w2,w3` | 9 | same shared shapes | present and verified; execution disabled |
| MTP `main_proj` | 1 | `(4096,12288)` | present and verified; execution disabled |

The last 25 tensors have runtime frequency zero: initialization explicitly
rejects `enable_dspark` because the base executor verifies the tensors but does
not execute speculative MTP. They remain protected future shapes; they were
measured here but are not relabeled as current verification traffic.

Every representative tensor was read from the real checkpoint header. There
is no manifest padding in these shapes, so the exact useful-byte formula is:

```text
W_FP8 = N*K + (N/128)*(K/128)
```

| Shape | E4M3 bytes | E8M0 bytes | `W_FP8` |
|---|---:|---:|---:|
| `wq_a` | 4,194,304 | 256 | 4,194,560 |
| `wq_b` | 33,554,432 | 2,048 | 33,556,480 |
| `wkv` | 2,097,152 | 128 | 2,097,280 |
| `wo_a` | 33,554,432 | 2,048 | 33,556,480 |
| `wo_b` | 33,554,432 | 2,048 | 33,556,480 |
| shared `w1` or `w3` | 8,388,608 | 512 | 8,389,120 |
| shared `w2` | 8,388,608 | 512 | 8,389,120 |
| indexer `wq_b` | 8,388,608 | 512 | 8,389,120 |
| MTP `main_proj` | 50,331,648 | 3,072 | 50,334,720 |

The source header evidence is
`results/dsv4-sm86-fp8-f8-0/representative-tensors.txt`.

## Runtime boundaries and real M bands

The audit found four materially different current paths. They must not be
collapsed into one “FP8 kernel”:

1. Generic compressed FP8 dispatch quantizes each activation/K128 block to
   E4M3 before `native_fp8_matmul_kernel`. Its output normally publishes BF16.
   This is the current scalar W8A8 behavior, not the desired W8A16 path.
2. The SM86 attention-page control, eligible only for ungrouped FP8 rows > 1,
   quantizes activations to compact E4M3/E8M0, widens both operands through
   48 KiB shared BF16 tiles, then uses BF16 WMMA. It covers `wq_a`, `wq_b`,
   `wkv`, and `wo_b`; grouped `wo_a`, the single-row indexer, and shared-page
   kernels are not eligible.
3. Shared-expert pages use 32-row scalar tiling: `w1/w3` consume the BF16
   boundary directly and publish BF16 before SwiGLU; the `w2` intermediate is
   then quantized to E4M3. This is a separate mixed W8A16/W8A8 incumbent.
4. Rank-local TP2 widens all five attention projections to persistent BF16 at
   load time. The old accepted capacity evidence projects 4,148,166,656 extra
   bytes per rank for 43 layers. It is a valid historical incumbent but
   violates the amended one-copy software-native objective and cannot be the
   promoted result.

The active decode/indexer band is M=1. The default attention/MoE page is M=64
with arbitrary tail rows 2–63; shared scalar tiling changes at M=32 and the
attention WMMA tile changes at M=64. M `{2,3,4,8,16}` is protected by the owner
gate for a future verification executor, but no DeepSeek verification-batch
executor currently exists. The experiment therefore measured M
`{1,2,3,4,8,16,32,64}` and records rather than invents current verification
frequency.

## Probe and protocol

`strata-dsv4-sm86-fp8-baseline-probe` is a standalone CUDA executable, never a
production dispatch. One invocation owns one `(M,N,K)` point and reports:

- a 128 MiB ILP-4 ruler after a 256 MiB L2 scrub;
- the exact scalar E4M3/E8M0 kernel with BF16-rounded activation values and no
  activation requantization — the F8-0 W8A16 baseline primitive;
- the existing W8A8 attention-page control including production-equivalent
  activation quantization; and
- all eleven raw sample times, exact traffic, oracle observations, and total
  allocation.

The fixture uses E4M3-exact values behind the BF16 carrier so the W8A16 and
W8A8 controls see the same mathematical activation. This is sufficient for
baseline timing and a boundary comparison, but is not F8-1's required broad
BF16 activation oracle.

The arena uses 2–8 replicas within a fixed 112 MiB budget. Total allocation
peaked at **512,040,448 B**, below the 536,870,912 B ceiling. All duplicate
codes exist only inside this isolated cold probe. Production code, weights,
dispatch, and residency were unchanged.

## Matrix results

There are 216 raw process arms and 72 three-process medians under
`results/dsv4-sm86-fp8-f8-0/full-step/`. The same-session ruler range over all
process medians was **834.85–845.63 GB/s**; at the binding M points it was
840.21–845.63 GB/s. The ruler profile reached **95.26% DRAM throughput**.

The scalar kernel physically rereads the compressed matrix M times. The first
range below counts those physical rereads; the second divides only one useful
checkpoint read by the full time, which is the footing of D-F8-GATE.

| M | scalar physical GB/s, active shapes | scalar one-read-equivalent GB/s |
|---:|---:|---:|
| 1 | 81.11–163.03 | 81.11–163.03 |
| 2 | 110.71–169.35 | 55.35–84.68 |
| 3 | 131.08–172.17 | 43.69–57.39 |
| 4 | 142.03–180.05 | 35.51–45.01 |
| 8 | 144.04–180.05 | 18.01–22.51 |
| 16 | 159.03–213.49 | 9.94–13.34 |
| 32 | 157.29–217.74 | 4.92–6.80 |
| 64 | 157.55–221.23 | 2.46–3.46 |

The W8A8 tensor-page control reads one matrix for M<=64 but includes compact
activation quantization. Only its four actual attention-page shapes are in
this range:

| M | W8A8 one-read-equivalent GB/s |
|---:|---:|
| 2 | 2.48–51.61 |
| 3 | 2.55–53.20 |
| 4 | 2.65–55.54 |
| 8 | 2.38–50.73 |
| 16 | 2.59–54.80 |
| 32 | 2.34–50.43 |
| 64 | 1.97–42.78 |

These are controls, not F8-2 candidates. Neither approaches the owner gate.
No favorable M=677 result was reused as a skinny baseline.

## Instantiated cost model

`results/dsv4-sm86-fp8-f8-0/full-step/cost-model.json` instantiates

```text
tau = max_r(W_r/B_r) + Sigma_serial
```

for every active shape and measured M. It records exact weight traffic,
W8A16 activation reads, output writes, W8A8 quantizer reads/writes, the
same-session `B_DRAM`, the weight-only DRAM floor, measured time, non-weight
residual, and the owner-bound full-step time budget.

Representative values show why a mechanism is justified:

| Point | measured scalar | scalar DRAM floor | non-DRAM residual | W8A8 full step | W8A8 weight floor | owner budget |
|---|---:|---:|---:|---:|---:|---:|
| `wkv`, M=1 | 25.86 us | 2.48 us | 23.38 us | ineligible | — | **3.02 us** |
| `wq_a`, M=1 | 35.84 us | 4.96 us | 30.88 us | ineligible | — | **6.05 us** |
| `wq_b`, M=1 | 209.92 us | 39.68 us | 170.24 us | ineligible | — | **48.39 us** |
| `wq_b`, M=16 | 3,376.03 us | 634.92 us | 2,741.11 us | 612.35 us | 39.68 us | **62.00 us** |
| `wq_b`, M=64 | 13,631.49 us | 2,539.68 us | 11,091.81 us | 784.38 us | 39.68 us | not gated |
| `wkv`, M=64 | 638.98 us | 158.73 us | 480.25 us | 1,065.98 us | 2.48 us | not gated |

The “non-DRAM residual” is measured residual, not silently classified as pure
launch time; the profile below separates its cause.

## Resource attribution and `argmax`

Nsight Compute profiled the scalar at every binding M point plus the M=64 page
band, the small `wkv` point, the W8A8 tensor control, its quantizer, and the
ruler. Complete CSVs are under `results/dsv4-sm86-fp8-f8-0/ncu/`.

| Profile | waves/SM | DRAM | warps active | issue active | long scoreboard | registers | shared |
|---|---:|---:|---:|---:|---:|---:|---:|
| ruler | 1.33 | 95.26% | 90.87% | 3.38% | 94.47% | 38 | 0 B |
| scalar `wq_b`, M=1 | 66.60 | 14.08% | 92.58% | 67.48% | 38.25% | 22 | 32 B |
| scalar `wq_b`, M=16 | 1065.63 | 13.55% | 93.02% | 67.98% | 38.04% | 22 | 32 B |
| scalar `wq_b`, M=64 | 4262.50 | 13.53% | 93.04% | 68.05% | 38.05% | 22 | 32 B |
| scalar `wkv`, M=1 | 1.04 | 7.71% | 92.78% | 54.80% | 53.01% | 22 | 32 B |
| W8A8 `wq_b`, M=1 | 1.56 | 5.89% | 27.88% | 24.69% | 30.17% | 96 | 48 KiB |
| W8A8 `wq_b`, M=64 | 1.56 | 4.55% | 27.04% | 25.92% | 34.27% | 96 | 48 KiB |
| W8A8 `wkv`, M=64 | **0.02** | **0.17%** | 16.67% | 19.21% | 33.21% | 96 | 48 KiB |
| quantizer `wq_b`, M=16 | 0.13 | 2.24% | 12.76% | 12.88% | 16.55% | 22 | 512 B |

For scalar W8A16, `argmax` is the decoder/load issue and dependency chain,
not DRAM bandwidth: changing M by 64x leaves DRAM near 13.5% and issue near
68%, while each row rereads the matrix. The small `wkv` shape additionally
pays a launch/wave residual.

For the existing W8A8 control, `argmax` is wave quantization plus the
96-register/48-KiB shared-memory pipeline and its separate underfilled
quantizer. `wkv` launches four tensor CTAs — 0.02 waves/SM — so its large time
is a serialization/occupancy defect, not a weight-volume limit.

The static binary agrees: scalar/tensor use 22/96 registers, zero stack/local
spill, and 32 B/49,152 B shared memory. Tensor SASS contains 16 HMMA sites;
the scalar contains none. The raw resource and SASS censuses are preserved in
`resource-usage.txt` and `sass-census.txt`.

## Correctness and limitations

- All 390 actual FP8 layouts and representative source shapes/scales passed
  header validation.
- The probe is a source-exact scalar kernel replica with the activation
  quantization deliberately removed to establish W8A16 timing. On the sampled
  independent BF16 oracle, active scalar records had at most one mismatch in
  512 samples (`shared_w2`, M=1), reflecting FP32 reduction association rather
  than a format or byte-layout change. F8-1 owes its own declared association
  oracle and may not inherit this observation as acceptance.
- All **28 actually eligible** tensor-page shape/M records (four attention
  shapes, M>1) were no worse than scalar at the sampled BF16 oracle and had
  zero sampled oracle mismatches. The known ineligible `shared_w2` M=1 and
  inactive MTP observations are retained, not used to pass the control.
- E4M3-exact activation fixtures do not prove arbitrary BF16 W8A16 operands.
  F8-1 must add broad finite BF16 values, scale-boundary fixtures, deliberate
  permutation/scale controls, and the actual operation publication oracles.
- The rank-local persistent BF16 attention cache remains production debt. F8-1
  must not use or preserve it as the promoted path.

`make check` passed: `strata-tests` and `strata-sim-smoke` passed; the Gemma 4
equivalence test was explicitly skipped because its fixture is unavailable.

## Verdict and next falsifier

**F8-0 is complete. F8-1 is authorized, but no FP8 performance gate is
cleared.** A register-fed QPN8 path directly targets both measured incumbent
bottlenecks: it can remove repeated scalar decoder/load issue work and avoid
the W8A8 control's shared-memory/barrier/activation-quantization path.

The owner budgets are extremely tight for the small shapes: about 3.02 us for
`wkv`, 6.05 us for `wq_a`, 12.10 us for an 8.39 MB matrix, and 48.4 us for a
33.56 MB matrix at M=1–4. Therefore F8-1's cheapest first falsifier is not a
full kernel. It must:

1. prove E4M3/E8M0 block-128 decode and scale-to-K binding directly into the
   C2 BF16 fragment registers, with broad BF16 activation and deliberate-bug
   controls; and
2. measure a launch/wave upper bound for the 2.10/4.19 MiB shapes, including a
   fused `wq_a+wkv` option because both consume the same layer input.

If no exact geometry can fit the 3.02/6.05 us budgets, the gate reads negative
for those shapes and F8-1 must stop rather than hiding them behind the scalar
fallback.

## Reproduction and raw evidence

```bash
cmake --build build-release --target strata-dsv4-sm86-fp8-baseline-probe -j2
results/dsv4-sm86-fp8-f8-0/run_matrix.sh
results/dsv4-sm86-fp8-f8-0/run_ncu.sh
make check
```

Key evidence:

```text
results/dsv4-sm86-fp8-f8-0/manifest-scan.json
results/dsv4-sm86-fp8-f8-0/fp8-modules.tsv
results/dsv4-sm86-fp8-f8-0/representative-tensors.txt
results/dsv4-sm86-fp8-f8-0/full-step/all-runs.json
results/dsv4-sm86-fp8-f8-0/full-step/process-medians.json
results/dsv4-sm86-fp8-f8-0/full-step/cost-model.json
results/dsv4-sm86-fp8-f8-0/ncu/summary.tsv
results/dsv4-sm86-fp8-f8-0/resource-usage.txt
results/dsv4-sm86-fp8-f8-0/sass-census.txt
results/dsv4-sm86-fp8-f8-0/non-sm86.exit
```
