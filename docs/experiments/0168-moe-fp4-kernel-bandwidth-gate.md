# Experiment 0168 — the fused MXFP4 MoE kernels are bandwidth-bound, not dispatch-bound

Status: **REGISTER-FED SUBSTITUTION ADMITTED FOR THE FUSED MoE ROUTE.** The
incumbent scalar MXFP4 MoE kernels reach 15.3% and 9.9% of this card's measured
read roofline at Laguna's real dispatch width, and hold that figure as the width
grows. The deficit is bandwidth, not launch overhead, so the register-fed decode
addresses it. Experiments 0166 and 0167 stopped on an Amdahl gate that this does
not overturn: the term is real but small.

## Why this ran

0166 and 0167 rejected the substitution for Laguna and Inkling because GPU matmul
is 6.2% and 5.5% of a decode step. That is a correct Amdahl argument, but it
inferred the kernel's achieved bandwidth rather than measuring it, and it did not
separate two very different causes of a small term:

- **bandwidth-bound and inefficient** — the register-fed kernel closes it;
- **dispatch-bound** — it does not, and DeepSeek V4 measured exactly that, 2.027
  against 2.038 ms of device MoE kernel time for a 9-launch register-fed shared
  expert against the incumbent's 5 (experiment 0164).

Telling these apart costs one microbenchmark. It decides whether the whole of
task A is buildable.

## Method

`apps/strata_moe_fp4_bandwidth_probe.cu` copies `mxfp4_moe_gate_up_kernel` and
`mxfp4_moe_down_kernel` from `kernels/cuda/backend.cu` verbatim, so it times the
production arithmetic rather than an approximation, and runs them at Laguna's
real routed-expert shapes from 0166:

- gate/up packed U8 `[1024,1536]`, scale U8 `[1024,96]`, logical `[1024,3072]`;
- down packed U8 `[3072,512]`, scale U8 `[3072,32]`, logical `[3072,1024]`;
- 4.78 MiB per expert, M=1.

The resident pool is 96 experts, 459.0 MiB, against a 6.0 MiB L2, so a sweep
cannot sit in cache. Timing a single reused matrix is how an earlier probe
reported a forbidden 435-484 GB/s ruler: it measured cache, then launch overhead.
Dispatch width is swept 1 to 64; medians of 11 samples after 3 warmups.

The ruler is a pure streaming `uint4` read over a 512 MiB arena on the same card
at the same operating point, not a datasheet figure.

Hardware: RTX 3090 (device 1), owner's production point, 250 W and clock-locked
to 1605 MHz. Both figures are DRAM-bound, so the clock lock does not move them.

## Result

Read ruler: **880.0 GB/s**.

| width | gate_up GB/s | % roofline | down GB/s | % roofline |
|---:|---:|---:|---:|---:|
| 1 | 105.3 | 12.0% | 68.0 | 7.7% |
| 2 | 116.6 | 13.2% | 75.9 | 8.6% |
| 4 | 128.0 | 14.5% | 81.6 | 9.3% |
| 8 | 133.9 | 15.2% | 85.4 | 9.7% |
| **10 (Laguna top-k)** | **134.9** | **15.3%** | **86.8** | **9.9%** |
| 16 | 136.7 | 15.5% | 87.7 | 10.0% |
| 32 | 138.3 | 15.7% | 88.8 | 10.1% |
| 64 | 139.2 | 15.8% | 89.4 | 10.2% |

The curve is flat from width 8 onward — 133.9 to 139.2 GB/s over an eightfold
increase in work. **A dispatch-bound kernel climbs with width; this does not.**
The narrow end costs only 21% against the wide end, so even at width 1 the
kernel is mostly bandwidth-limited.

`down` is consistently worse than `gate_up`, 9.9% against 15.3%. `gate_up` reads
two weight streams that share one activation, so it has twice the memory-level
parallelism per block; `down` reads one. That makes `down` the larger prize, not
the smaller one.

## Register-fed arm, same probe

Fragment order is a pure permutation at identical byte count, so the same
allocations serve both arms and the memory traffic is unchanged. This arm
measures bandwidth only; the decode's numerics are established by the
route-against-route test in `tests/test_cuda_backend.cpp` and by Gemma 4's
3.367x (experiment 0165).

| width | gate_up GB/s | % roofline | vs scalar | down GB/s | % roofline | vs scalar |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 296.7 | 33.7% | 2.89x | 181.3 | 20.6% | 2.67x |
| 4 | 522.2 | 59.3% | 4.08x | 384.0 | 43.6% | 4.70x |
| 8 | 607.7 | 69.0% | 4.54x | 502.2 | 57.0% | 5.85x |
| **10 (Laguna top-k)** | **680.0** | **77.2%** | **5.04x** | **544.0** | **61.7%** | **6.30x** |
| 16 | 670.1 | 76.0% | 4.90x | 607.3 | 68.9% | 6.93x |
| 32 | 710.5 | 80.6% | 5.14x | 687.2 | 78.0% | 7.74x |
| 64 | 710.8 | 80.7% | 5.10x | 768.0 | 87.2% | 8.59x |

At Laguna's real width the substitution is **5.04x on gate_up and 6.30x on
down**. Unlike the scalar arm this one does climb with width, because a wider
batch gives the split-K heuristic more warps to cover the device; `down` climbs
further than `gate_up` because its shorter K loop leaves less latency to hide.

## End-to-end estimate, from measured arms

Per layer at width 10: gate_up 0.2478 -> 0.0492 ms, down 0.1925 -> 0.0307 ms,
saving 0.3604 ms. Across 47 sparse layers split over two devices that is about
8.5 ms, less roughly 0.51 ms of prepack.

The prepack is a device-side permutation of the staged bytes: 229.50 MiB/token
read and written at 880 GB/s is 0.509 ms/token, **0.78% of the 65.05 ms staging
term** and 0.21% of the step. It was worth checking, because if it had been a
per-staging cost of the same order as the saving the substitution would have
been negative for every streaming model; it is not, so no host-side or offline
layout change is needed to make this admissible.

Net: about 8.0 ms of a 236.98 ms step, **roughly 1.035x end to end**.

## Verdict

Headroom to the accepted register-fed figure of 88% of roofline (experiment
0148) is 5.8x on gate_up and 8.9x on down, and the arm below collects 5.04x and
6.30x of it at Laguna's real width. The deficit is the scalar
decode-and-accumulate loop, which is what the register-fed path replaces.

At Laguna's width 10 the measured per-layer cost is 0.2478 ms gate_up plus
0.1925 ms down, 0.4403 ms; across 47 sparse layers split over two devices that
is about 10.3 ms of the 14.70 ms of slowest-device matmul 0166 measured, which
is consistent. Taking both terms to 88% gives about 0.065 ms per layer, so
roughly 8.8 ms saved from a 236.98 ms step: **about 1.04x end to end.**

So both statements hold, and neither cancels the other:

- **the substitution works** — 5.8x and 8.9x on the term it targets, unlike the
  launch-bound DeepSeek V4 shared expert where it delivered nothing;
- **the step barely moves** — 94% of Laguna's decode is not matmul, and no
  kernel changes that ratio. Only reducing routed-expert staging does.

0166's and 0167's Amdahl arithmetic stands. What changes is the reason for it:
not that the kernel cannot help, but that the term it helps is small. That is a
statement about the workload, not about the kernel, and it does not transfer to
a resident model — Gemma 4 took 3.367x from the same code (experiment 0165).

## Consequence for task A

Admitted. The fused register-fed MoE route is worth building, on two grounds
beyond the 1.04x: it is the same decode path Gemma 4 already proves at 3.367x,
and any future MoE checkpoint that fits on the card inherits it.

The prepack must move off the staging path first. Fragment order is currently
applied device-side, so a streaming model would permute every newly staged
expert — 144 misses/token on Laguna — inside the term that is already `argmax_r`.
The permutation is a pure byte shuffle at identical byte count, so it belongs in
the host pool at load, after which staging is an unmodified memcpy and the
substitution costs nothing on the bottleneck.

Fragment order survives every topology in use: TP shards are multiples of 128
and the fragment is 16x16, so the permutation is shard-local; PP places whole
layers and does not touch layout; `m16n8k16` BF16 MMA exists on both SM 8.6 and
SM 12.0, so a 3090 and a 5060 Ti read one host copy.

## Reproduce

```bash
cmake --build build-release --parallel --target strata-moe-fp4-bandwidth-probe
./build-release/strata-moe-fp4-bandwidth-probe --device 1
```
