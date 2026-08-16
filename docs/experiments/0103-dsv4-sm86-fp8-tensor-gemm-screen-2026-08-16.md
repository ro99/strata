# Experiment 0103 — DSV4 SM86 FP8 tensor GEMM screen

Status: **performance and traffic mechanism confirmed; numerical no-worse gate
failed, so no production work was started.** A tile-local E4M3-to-BF16
tensor-core tile is 6.50--23.62x faster than the production FP8 kernel at the
three 677-row shapes, but it is materially worse than the incumbent in maximum
absolute and RMS oracle error for `wq_a` and `wkv`.

## Lineage and predeclared contract

Experiment 0101 showed that plain-BF16 cuBLAS reaches 28--38% of the RTX 3090's
dense BF16 tensor peak while a hand-written row kernel reaches about 2%.
Experiment 0102 then corrected the production premise: DeepSeek V4 query/KV
weights are FP8 E4M3 block-128 and execute `native_fp8_matmul_kernel`, so the
0101 kernel is unreachable for this term.

0103 screens the actual production encoding. Hypothesis: the incumbent's one
block per `(output_row,batch_row)` rereads every FP8 weight for every page row,
while the reference stack's SM80 mechanism decodes E4M3 bytes to BF16 inside a
tile, feeds BF16 tensor-core dots, and applies E8M0 block scales after each
K=128 dot. The target resource is device global-memory work and CUDA-core
reduction/dispatch in the 7.159487 s production query matmul term.

Primary metrics were logical weight bytes, median kernel time, GFLOP/s and
fraction of the RTX 3090's 142 TFLOP/s dense BF16 tensor peak. Correctness used
4,096 deterministic output samples spanning each complete production M/N/K
shape, with FP64 scalar accumulation of decoded E4M3 operands and E8M0 scales.
Both implementations were compared to truth; the tensor path had to be no worse
in both maximum absolute and RMS error. The probe device ceiling was 256 MiB.
Any contradiction stopped production work.

Budget was eight minutes of reference/source audit, twelve minutes for the
standalone implementation/build, five minutes of GPU/oracle screening and five
minutes for analysis/record/tests. There was no model load or model arm.

## Reference mechanism

`Lvllmds4-x/SM80_DEEPSEEK_V4_NOTES.md` section B states that Ampere cannot
represent or tensor-dot `fp8e4nv`. Its block-scaled Triton launcher bitcasts
both E4M3 operands to `uint8`; `_w8a8_triton_block_scaled_mm` decodes bytes
in-register, casts exactly representable values to BF16, performs BF16
tensor-core dots, and applies activation/weight block scales after each dot.
The reference reports max-relative error around 0.006 against its BF16
reference and warns that SM80 performance is untuned.

Strata quantizes activations to unscaled E4M3 values, so the standalone probe
uses an implicit activation scale of one and the checkpoint's E8M0 weight scale
per `[128,128]` block. Its CUDA tile is M=64, N=128, K=128, matching the
reference default shape. Each CTA has eight warps and 48 KiB of shared memory;
the CUDA WMMA implementation decodes into that transient shared BF16 tile before
loading tensor fragments, whereas Triton keeps the decoded tile in compiler-
managed registers. No expanded global or persistent weight matrix is
materialized in either shape.

## Verified traffic arithmetic

The production grid is exactly `[N,M]`, with one 256-thread block per
`(output_row,batch_row)`. Each block loops over K and reads K FP8 weight bytes.
At M=677:

| Projection | Current weight reads/layer | One-read floor/layer | M64 tensor reads/layer |
|---|---:|---:|---:|
| `wq_a` | 2,839,543,808 B | 4,194,304 B | 46,137,344 B |
| `wq_b` | 22,716,350,464 B | 33,554,432 B | 369,098,752 B |
| `wkv` | 1,419,771,904 B | 2,097,152 B | 23,068,672 B |
| **sum** | **26,975,666,176 B** | **39,845,888 B** | **438,304,768 B** |

Across 43 layers the incumbent implies 1,159,953,645,568 logical weight bytes
(1.160 TB decimal), versus a 1,713,373,184-byte one-read floor: exactly 677x
amplification. The M64 candidate has eleven row tiles and therefore reads each
weight byte eleven times, 18,847,105,024 bytes across 43 layers. It removes
61.55x of incumbent logical weight reads but does not reach the one-read floor.
This is source-level global-load volume; L2 hits can make physical HBM service
smaller and are not mislabeled as eliminated instructions.

The candidate also reduces logical activation reads. The incumbent reads one
FP32 activation for every output weight, about 107.90 GB/layer; the tiled path
reads encoded one-byte activations once per N=128 tile, about 210.76 MB/layer.
Its launch grid is 2,948 CTAs/layer across the three shapes versus 23,223,808
CTAs/layer for the incumbent.

## Performance result

The declared and executed protocol was three warmups and eleven interleaved
timed samples on GPU 1, an NVIDIA GeForce RTX 3090.

| Projection | Current ms | Tensor ms | Speedup | Current GFLOP/s | Tensor GFLOP/s | Current peak | Tensor peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| `wq_a` | 16.1976 | 1.5083 | 10.74x | 350.6 | 3,765.3 | 0.247% | 2.652% |
| `wq_b` | 151.6943 | 6.4236 | 23.62x | 299.5 | 7,072.8 | 0.211% | 4.981% |
| `wkv` | 7.4752 | 1.1509 | 6.50x | 379.9 | 2,467.3 | 0.268% | 1.738% |

The production-shaped probe closes 0102's discrepancy cleanly: `wq_a+wq_b`
is 167.892 ms/layer, or 7.219 s over 43 layers, within 0.8% of 0100's measured
7.159487 s query device service. The tensor screen is 7.932 ms/layer for those
two projections, or 0.341 s over 43 layers. This is an isolated mechanism
ceiling, not a runtime prediction or value claim.

## FP64-oracle result

Relative error uses `max(abs(oracle),1e-9)`. The near-zero-sensitive relative
metric is reported alongside absolute and RMS error, not substituted for them.

| Projection | Current max abs | Tensor max abs | Current RMS | Tensor RMS | Current max rel | Tensor max rel |
|---|---:|---:|---:|---:|---:|---:|
| `wq_a` | 0.0139618 | 0.0323486 | 0.0027633 | 0.0036280 | 0.004337 | 0.001572 |
| `wq_b` | 0.0064545 | 0.0076904 | 0.0011146 | 0.0009153 | 0.000305 | 0.000024 |
| `wkv` | 0.0132446 | 0.0285339 | 0.0028667 | 0.0038869 | 0.004270 | 0.000256 |

All tensor maximum-relative errors are below the reference's approximately
0.006 figure. That does not clear the predeclared truth gate. Against the same
FP64 samples, tensor maximum absolute error is 2.32x current for `wq_a`, 1.19x
for `wq_b` and 2.15x for `wkv`; RMS is 31.3% and 35.6% worse for `wq_a`/`wkv`,
while `wq_b` RMS is 17.9% better. The candidate is not uniformly no worse than
the incumbent and therefore cannot claim an improved numerical contract.

The routed-expert precedent permits an explicitly declared reassociation bound
and determinism gate. It does not rescue this experiment: 0103 deliberately
asked the stronger question of whether the tensor path improved truth error,
and the measured answer is no for two of three shapes.

## Complete resource signs

- **Reduced:** logical FP8 weight reads 61.55x; logical activation reads about
  512x across these shapes; CTA count about 7,878x; CUDA-core scalar reduction
  is replaced by BF16 tensor-core work.
- **Increased:** every global FP8 byte is decoded to BF16 per tile; each CTA
  uses 48 KiB shared memory and substantially more registers; each K=128 dot
  adds a scaled FP32 partial. These costs hold achieved throughput to 1.7--5.0%
  of dense BF16 peak, below plain-BF16 cuBLAS in 0101.
- **Unchanged residency:** weights remain one FP8 byte each plus existing E8M0
  scales. Expansion is in-register/shared-memory only, so persistent weight
  bytes, weight-cache capacity and eviction pressure do not grow.
- **Activation workspace:** a production path would need encoded-byte activation
  input rather than the incumbent's in-place FP32 representation. Replacing
  that representation reduces device activation storage 4x; retaining both
  would add workspace and is not implied by this probe.
- **Unchanged:** output remains FP32 before the existing BF16 publication
  boundary; checkpoint precision, expert residency, router/top-k and decode are
  untouched because no runtime code exists.

Largest probe device allocation was 129,297,408 bytes (123.31 MiB), below the
256 MiB ceiling.

## Verdict

The structural/performance mechanism is real and the current 677x traffic
amplification is confirmed. The numerical no-worse gate fails, so no production
implementation or model timing is authorized by this result. The standalone
probe and ignored raw result are preserved for review. Any follow-up that
changes accumulation accuracy or declares a bounded reassociation contract is a
new hypothesis and must be approved before code.
