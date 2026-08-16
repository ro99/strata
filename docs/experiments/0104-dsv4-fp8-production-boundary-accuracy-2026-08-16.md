# Experiment 0104 — DSV4 FP8 production-boundary accuracy

Status: **post-boundary no-worse gate passed at every production shape; no
runtime integration was started.** After the exact FP32-carrier-to-BF16
boundary used by query and KV projections, the tensor path has the same oracle
mismatch count and no worse maximum absolute, maximum relative or RMS error
than the incumbent on all 4,096 deterministic samples per shape.

## Lineage and contract

Experiment 0103 established the production mechanism: the incumbent's FP8
kernel reproduces 0100's query device time within 0.8%, reads 1.160 TB of
logical weights across 43 layers, and is 6.50--23.62x slower than a tile-local
E4M3-to-BF16 tensor-core screen. Its stronger FP32 no-worse gate failed because
tensor maximum absolute/RMS error exceeded incumbent for `wq_a` and `wkv`.

0104 asks whether those differing FP32 bits survive the precision at which the
model actually consumes projection output. The primary gate compares each path
to the same FP64 scalar oracle after the exact production boundary. Declared
metrics are maximum absolute error, maximum relative error, RMS error and BF16
code mismatch rate. Tensor must be no worse than incumbent on every metric and
shape. Only if this gate failed was one Kahan/two-sum inter-K-block variant
authorized. Scope remained standalone only: no runtime change or model arm.

Budget was five minutes of boundary audit, eight minutes for instrumentation
and one screen, up to eight minutes for compensation only if gated, and five
minutes for record/tests, below 26 minutes. The probe retained 0103's 256 MiB
device ceiling and compact FP8 weight residency.

## Production boundary and consumers

`attention_page` passes `round_output=true` to all three `linear_rows` calls:

- `wq_a` output is rounded to BF16, downloaded, and consumed by query-rank RMS
  norm; that normalized activation then feeds `wq_b`.
- `wq_b` output is rounded to BF16, downloaded, and consumed by per-head RMS
  normalization, RoPE and paged attention.
- `wkv` output is rounded to BF16, downloaded, and consumed by KV norm, RoPE,
  sliding-cache append and learned-index compression.

In `CudaBackend::matmul_impl`, `dsv4_round_float_bf16` runs on `state.output`
after the projection kernel and before the D2H copy. Therefore no consumer on
this page path observes the projection's unrounded FP32 result. Later query/KV
normalization and RoPE add further BF16 boundaries, but they are downstream of
the boundary screened here.

The oracle computes the decoded E4M3/E8M0 dot in FP64, casts through the same
FP32 carrier that the projection ABI produces, and rounds RNE to BF16. Each GPU
path is independently rounded RNE to BF16 and compared with that oracle BF16
value. Relative error uses `max(abs(oracle_bf16),1e-9)`.

## Result

The same three-warmup, eleven-interleaved-sample RTX 3090 screen was repeated
with post-boundary accounting. Timing remained consistent with 0103:

| Projection | Current ms | Tensor ms | Speedup |
|---|---:|---:|---:|
| `wq_a` | 16.1975 | 1.5065 | 10.75x |
| `wq_b` | 152.4193 | 6.4102 | 23.78x |
| `wkv` | 7.5385 | 1.1398 | 6.61x |

Post-boundary oracle metrics over 4,096 deterministic full-shape samples:

| Projection | Path | Max abs | Max rel | RMS | Mismatches | Rate |
|---|---|---:|---:|---:|---:|---:|
| `wq_a` | incumbent | 0 | 0 | 0 | 0 | 0% |
|  | tensor | 0 | 0 | 0 | 0 | 0% |
| `wq_b` | incumbent | 0 | 0 | 0 | 0 | 0% |
|  | tensor | 0 | 0 | 0 | 0 | 0% |
| `wkv` | incumbent | 0.0078125 | 0.005025126 | 0.000122070 | 1 | 0.0244141% |
|  | tensor | 0.0078125 | 0.005025126 | 0.000122070 | 1 | 0.0244141% |

Both paths select exactly the same BF16 code as the oracle for every sampled
`wq_a` and `wq_b` output. At `wkv`, both miss the oracle on the same aggregate
count and have identical declared errors. The tensor path is therefore no
worse than incumbent on all declared production-boundary metrics and shapes.

The FP32 differences from 0103 remain honestly recorded; this experiment does
not rewrite or relax that result. It establishes that those differences do not
change the measured carried value at the immediate BF16 boundary. The routed-
expert precedent uses a bounded reassociation contract because its changed
values remain visible at its declared boundary. Here no new tolerance is
needed for the sampled projections: the candidate meets the incumbent's oracle
quality after the existing boundary.

## Conditional accumulation stage

The current tensor probe already uses a plain FP32 running accumulator across
K=128 blocks with each block's E8M0 scale applied before accumulation. Kahan or
two-sum was authorized only if the post-boundary gate still failed. Because the
gate passed, that stage was not built or timed; proceeding would violate the
declared dependency and optimize bits the current consumer discards.

## Resource signs and verdict

Resource signs are unchanged from 0103: logical weight reads fall 61.55x,
activation reads about 512x and CTA count about 7,878x; tile decode ALU, 48 KiB
shared memory and register pressure increase. FP8 weights and E8M0 scales remain
compact, with no expanded persistent weight matrix or cache-capacity cost. A
future production design still must choose a byte activation representation
without retaining the incumbent FP32 encoded-value workspace alongside it.

The numerical decision gate passes. This authorizes no runtime code by itself:
the user explicitly reserved production integration for the next decision.
Raw output is preserved at
`results/dsv4-0104-boundary-accuracy/probe.json` (ignored by Git), and the
standalone screen remains the only changed executable path.
