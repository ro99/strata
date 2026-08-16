# Experiment 0101 — DSV4 BF16 page GEMM mechanism screen

Status: **standalone plain-BF16 mechanism screen positive, but superseded as a
DeepSeek V4 production screen by experiment 0102.** The actual `wq_a`, `wq_b`
and `wkv` checkpoint tensors are FP8 E4M3 block-128, and production dispatches
`native_fp8_matmul_kernel`, not the plain-BF16 kernel measured here. Production
work was not started. The current plain-BF16 row kernel is
13.19--20.06x slower than the already-linked tensor-op cuBLAS path at the exact
677-row projection dimensions, but that ratio does not describe the production
projection encoding or kernel. Numerical reassociation is small in absolute
terms but is not covered uniformly by the repository's existing
reassociation-test formula.

## Predeclared contract and budget

Hypothesis: DeepSeek V4 page projections are already batched on the host, but
the CUDA backend executes them with `bf16_matvec_rows_kernel`, a CUDA-core
matvec reduction that does not exploit reuse across the 677 rows. A
`cublasGemmEx` BF16 tensor-op GEMM at the same M/N/K should materially reduce
the 7.159 s query projection device term measured in experiment 0100.

The primary metric is median kernel time, achieved GFLOP/s and cuBLAS/current
ratio at each exact production shape. The falsifier is a small ratio. The
correctness screen reports maximum absolute and relative error on identical
inputs, both before and after the existing BF16 publication boundary, and
checks repeat determinism. It does not silently grant a new numerical
contract. The device-memory ceiling is 256 MiB; the largest measured allocation
is 160,004,096 bytes (152.59 MiB). This screen changes no runtime resource:
weights, activation movement, host work, synchronization, model semantics and
decode are untouched. Only isolated GPU projection compute is compared.

Budget was five minutes of source audit, eight minutes to implement/build, less
than five minutes of GPU measurement and five minutes of analysis, below 25
minutes total. There is no model load, model arm or tmux job. The probe uses
five warmups and 21 interleaved samples per implementation and shape on GPU 1,
an NVIDIA GeForce RTX 3090.

## Exact production shapes

`kDeepSeekV4ExecutionContract` declares hidden size 4096, query LoRA rank 1024,
64 attention heads and head dimension 512. Runtime preload and
`attention_page` therefore establish these row-major products at M=677:

| Projection | Product | M | N | K |
|---|---|---:|---:|---:|
| `wq_a` | `[M,4096] * [1024,4096]^T` | 677 | 1,024 | 4,096 |
| `wq_b` | `[M,1024] * [32768,1024]^T` | 677 | 32,768 | 1,024 |
| `wkv` | `[M,4096] * [512,4096]^T` | 677 | 512 | 4,096 |

The standalone probe duplicates the production
`bf16_matvec_rows_kernel<16>` exactly rather than changing or exporting the
runtime kernel. Both arms consume identical BF16 weights and mathematically
identical BF16 activations; the current arm expands activation values exactly
to FP32 because that is its production ABI. The cuBLAS arm uses BF16 operands,
FP32 accumulation/output, `CUBLAS_TENSOR_OP_MATH` and
`CUBLAS_GEMM_DEFAULT_TENSOR_OP`. A separate paired timing includes the same
BF16-publication rounding kernel in both arms.

The first numerical probe used fixed binary scales. It produced bit-exact sums
for every output because these synthetic dot products reduced to exact integer
sums at the tested K, an implausible result for a reassociation screen. That
probe input was defective. The preserved source now varies finite BF16
exponents and mantissas; no mechanism or timing shape changed.

## Throughput result

The RTX 3090 denominator is 142.0 TFLOP/s dense BF16 tensor throughput; sparse
doubling is not counted.

| Projection | Current ms | cuBLAS ms | Current GFLOP/s | cuBLAS GFLOP/s | Ratio | Current peak | cuBLAS peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| `wq_a` | 2.1369 | 0.1321 | 2,657.7 | 42,992.1 | 16.18x | 1.87% | 30.28% |
| `wq_b` | 16.9083 | 0.8427 | 2,687.0 | 53,914.0 | 20.06x | 1.89% | 37.97% |
| `wkv` | 0.9421 | 0.0714 | 3,014.1 | 39,756.2 | 13.19x | 2.12% | 28.00% |

Including BF16 publication, the current/cuBLAS ratios remain 15.60x, 16.17x
and 12.82x respectively. The mechanism screen therefore supports the stated
resource diagnosis: the hand-written kernel sustains only about 2% of the
card's dense BF16 tensor peak while cuBLAS moves the exact shapes to 28--38%.
This is not an end-to-end speedup claim.

## Numerical result and precedent

Relative error uses `max(abs(current), 1e-7)` as its denominator; the large
`wq_b` maximum is a near-zero cancellation and is reported rather than hidden.

| Projection | Raw max abs | Raw max rel | Raw mean abs | BF16 max abs | BF16 max rel | BF16 mismatches |
|---|---:|---:|---:|---:|---:|---:|
| `wq_a` | 0.000024796 | 0.128571 | 0.000003228 | 0.03125 | 0.128571 | 677 / 693,248 (0.0977%) |
| `wq_b` | 0.000015259 | 10.430813 | 0.000001058 | 0.03125 | 10.430813 | 12,991 / 22,183,936 (0.0586%) |
| `wkv` | 0.000030518 | 0.090909 | 0.000003235 | 0.03125 | 0.094118 | 379 / 346,624 (0.1093%) |

Repeated cuBLAS output had zero FP32 bit mismatches at all three shapes. The
maximum BF16-published absolute difference is one actual BF16 spacing at the
largest `wq_b` magnitude and half a spacing at the largest `wq_a`/`wkv`
magnitudes. Near zero, however, cancellation makes ULP and relative metrics
large even when absolute error is small.

The two existing `cublasGemmEx` sites in `kernels/cuda/backend.cu` use BF16
operands, FP32 compute and direct BF16 output under tensor-op math. Their
batched-attention fixture is bit-exact against the per-row path because both
arms use the same cuBLAS reduction; it does not declare a scalar-versus-cuBLAS
reassociation tolerance. The closest explicit repository precedent is the
transformed routed-expert fixture: deterministic output and absolute error at
most `maximum canonical magnitude / 256`, described as one BF16 mantissa step.

This screen passes that formula for `wq_a` (0.654x its bound) and `wkv`
(0.657x), but `wq_b` is 1.143x the formula despite differing by exactly one
actual BF16 spacing at the maximum magnitude. Therefore the existing formula
does **not** admit all three synthetic projection shapes as-is. A production
candidate would need an explicitly approved, production-shape oracle contract,
at minimum deterministic output plus a declared absolute BF16 reassociation
bound, followed by generated-token/decode gates. Using one actual BF16 spacing
at the canonical maximum would admit this probe, but that is a new declaration,
not a result silently inherited from the existing cuBLAS sites.

## Decision

The plain-BF16 compute hypothesis survives its isolated test by a large margin,
but experiment 0102 found that it was not instantiated on the production weight
encoding. It therefore makes no production-speed claim.
No production kernel, runtime path, model arm, query allocation, attention host
remainder, append/compressor path or MoE work was touched. The next decision is
whether to authorize a production candidate and which numerical oracle contract
must gate it. Raw probe output is preserved at
`results/dsv4-0101-bf16-gemm-screen/probe.json` (ignored by Git), and the
standalone probe source is committed with this record.
