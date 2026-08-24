# Experiment 0145: SM86 QPN8 kernel and Ampere-native execution screen

## Result

A production-shaped W8A16 kernel now exists. It consumes the checkpoint's
unchanged E4M3 bytes and E8M0 block-128 scales, decodes directly into native
SM86 BF16 MMA registers, accumulates in FP32, and publishes BF16 without a
persistent widened weight copy. The valid same-input `wq_b + indexer.wq_b`
fusion reaches a three-process median **718.64 GB/s, 84.98%** of its
same-session local ruler, above the binding 82% M=1 target.

This does **not** complete F8-1 or pass the F8-2 curve. The isolated small
`wq_a + wkv` prefix remains launch/wave bound, the layer-resident scheduler is
not built, M `{2,3,4,8,16}` is unmeasured, and the few full-output BF16
rounding differences on large random matrices still require a no-worse
comparison against the production incumbent and real fixtures.

## Predeclared experiment contract

- **Hypothesis:** after experiment 0144 rejected one launch per skinny
  projection, an actual compact W8A16 inner kernel plus an Ampere-native
  layer-resident/fused execution grid can meet the unchanged equal-roofline
  gate. Native BF16 `m16n8k16`, shape-specific split-K, and enough independent
  N work are the Ampere advantages; Volta's `m8n8k4` ownership is not copied.
- **Primary metric:** useful checkpoint bytes divided by the complete kernel
  time and by the same-session 128 MiB cold ruler; kernel time includes
  activation feed, decode, MMA, split reduction, and BF16 output.
- **Correctness gate:** unchanged E4M3/E8M0 bytes and block-128 binding,
  unchanged BF16 activation boundary, FP64 decoded oracle at BF16 publication,
  and no persistent widened weights. No production acceptance without real
  fixtures and no-worse incumbent comparison.
- **Memory ceiling:** 512 MiB device allocation.
- **Rollback:** preserve any negative mechanism as evidence; do not lower
  D-F8-GATE, hide a protected shape, or dispatch a numerically unaccepted arm.
- **Measured starting bottleneck:** experiment 0144 measured the rejected
  per-projection term as launch plus underfilled waves. The new mechanism must
  reduce that serial/eligibility term rather than decoder ALU alone.

The harness uses a 256 MiB L2 scrub, 128 MiB same-session ruler, at most 88 MiB
of compact rotating weights, three warmups, eleven samples, and a maximum
observed allocation of **497,204,736 bytes**. A typical arm is seconds, so no
long system run was warranted.

## Kernel

`apps/strata_dsv4_sm86_fp8_qpn_probe.cu` implements:

- a byte-preserving fragment prepack into one E4M3 byte per weight;
- one `uint4` load per lane covering two K16 fragments and four loads issued
  together per K128 block;
- the exact fast E4M3/E8M0 decoder admitted by the checkpoint scale scan;
- native `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`;
- alternating FP32 accumulators and a single-launch split-K reduction;
- shape-selectable split-K, exact useful-byte accounting, cold rotation, and a
  full CPU oracle.

The 48 checkpoint shards contain 300,288 inspected attention scale bytes:
minimum 114, maximum 120, and none outside the fast admitted range. The
accepted default path has no local-memory spill and no widened shared weight
tile. Nsight reports 40 registers/thread on the winning split-1 fusion.

## What was tested and rejected

The failed variants are binding evidence about the silicon difference:

| Variant | Prediction | Measured result |
|---|---|---|
| Eight-warp hierarchical reduction | remove most global partial traffic | barriers/registers erased the saving; no shape improved |
| Two N16 tiles per warp | copy Skinny's activation reuse at Ampere tile size | slower on every shape; split-16 `wkv` 11.264 us versus 8.192 us |
| Move E8M0 scale from weights to B | halve BF16 pair multiplies | exact but slower; `wkv` 9.216 us versus 8.192 us |
| `cp.async` compact-byte staging | hide the long-scoreboard weight load | exact but slower; the winning split has too few K128 stages to amortize shared staging |
| K128 accumulator scaling | exploit native BF16 MMA and remove weight-pair multiplies | exact on the random oracle and 40 rather than 48 registers, but no timing change |

The implication is direct: decoder ALU is not the remaining `argmax`. Ampere's
larger native BF16 MMA completes a skinny matrix with too little independent
work. More micro-decoder imitation of Skinny cannot repair that.

## Real-shape results

Three independent processes were interleaved over the selected real-shape
geometries. Each process used eleven samples and its own same-session ruler.

| Operation / valid fusion | `(N,K)` | split-K | median us | median GB/s | ruler efficiency | BF16 differences / outputs |
|---|---:|---:|---:|---:|---:|---:|
| `wq_a + wkv` | `(1536,4096)` | 8 | 15.360 | 409.63 | 48.44% | 0 / 1,536 |
| `wq_b` | `(32768,1024)` | 1 | 49.152 | 682.71 | 81.11% | 4 / 32,768 |
| `wq_b + indexer.wq_b` | `(40960,1024)` | 1 | 58.368 | **718.64** | **84.98%** | 5 / 40,960 |
| `wo_a` | `(8192,4096)` | 4 | 49.152 | 682.71 | 80.73% | 2 / 8,192 |
| `wo_b` | `(4096,8192)` | 8 | 49.152 | 682.71 | 80.73% | 2 / 4,096 |

The large-shape differences are BF16 publication differences from FP32 MMA
reduction order, not format/decode differences. Maximum error divided by the
sum of absolute products was `9.74e-4` for `wq_b`, `2.72e-4` for the valid
fusion, `6.50e-6` for `wo_a`, and `7.08e-6` for `wo_b`. They are reported as
an open correctness gate, not excused as exact.

The valid fusion profile reports **85.01% DRAM throughput**, 0.65 waves/SM,
63.59% active warps, 26.40% issue active, and 85.21% long-scoreboard stalls.
At 718.64 GB/s useful throughput the `argmax` has moved to the compact DRAM
stream. There are 40 registers, no local spill, and no widened shared tile.

## Ampere execution conclusion

The synthetic independent-work sweep is a mechanism probe, not a production
pass. It establishes the grid threshold cleanly: 36--40 MiB of independent
compact projection work crosses 82%, and 64 MiB reaches about 87%. The real
layer has three 33.56 MiB projections. Their isolated measurements are within
one measured 3.072 us launch floor of the gate, while the 41.95 MiB valid
`wq_b + indexer` fusion passes even with that launch.

The production successor is therefore a long-lived, layer-resident projection
scheduler, not a Volta kernel clone. It must:

1. compute `wq_a` first from the layer input;
2. after query-rank publication, schedule `wq_b` concurrently with the
   independent `wkv` work, adding indexer `wq_b` to the same K=1024 grid when
   active;
3. retain shape-specific split-K `{1,4,8}` for `wq_b`, `wo_a`, and `wo_b`;
4. synchronize only at actual attention dependencies, without widened weight
   residency or hidden fallback; and
5. validate every intermediate BF16 boundary before measuring the full M
   curve.

That scheduler is the next experiment. Until it exists, the launch-free body
numbers are a cost model, not a throughput claim.
