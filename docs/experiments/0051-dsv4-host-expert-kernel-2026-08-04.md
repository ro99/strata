# Experiment 0051 — a bit-exact host FP4 expert, and what it is worth

Status: **kernel accepted and gated off; the scheduling change that uses it is
rejected at 0.76x.** The host FP4 expert is bit-identical to the device kernel
at two shapes and runs at 26-29 GB/s standalone against PCIe's 6.9. Diverting
cache-miss experts to it is nonetheless a measured regression, for a reason that
took four policy variants and a correctness bug to isolate: the host path should
stay off until decode has more than one diverted expert per layer.

**What landed on main.** The kernel, its header and its four test cases only —
`kernels/cpu/dsv4_fp4_expert.cpp`, `include/strata/deepseek_host_expert.hpp`,
`tests/test_deepseek_host_expert.cpp`. The `--host-expert-misses` flag, its
runtime wiring in `src/deepseek_runtime.cpp`, the two `Dsv4DeviceMoeStats`
counters and `scripts/run_deepseek_v4_host_expert_ab.sh` are **not** on main:
they are the rejected policy, and the charter does not merge failed runtime
code. They remain on `infra/dsv4-external-stack-teardown` for reference. The
kernel on main therefore has no caller, which is deliberate and is recorded in
the header. Experiment 0054 later rejected the other candidate caller, static
placement, by arithmetic.

## Contract

- Hypothesis: the 100.3 ms/step of demand H2D at the chat operating point can be
  removed by computing cache-miss experts on the CPU, where the weights already
  are, instead of transferring them to the GPU.
- Primary metric: sustained GB/s of routed-expert weights through the host
  kernel at the real expert shape, 28 threads, against the 6.9 GB/s the PCIe
  path achieves for the same bytes.
- Correctness gate, stated before the work: **bit-identical to the device
  kernel, or the work is rejected.** This is not a numerics trade like
  `--fast-mhc`; a routed expert that disagrees with the device changes decode
  output, and the charter forbids that silently.
- Kill criterion: if the host kernel cannot beat 6.9 GB/s, the mechanism does
  not reduce `argmax_r` and is rejected.
- Memory ceiling: unchanged. The kernel reads the existing resident arena and
  allocates one `intermediate`-float scratch per worker.
- Rollback: the kernel is additive and unreferenced by the runtime. Deleting
  `kernels/cpu/dsv4_fp4_expert.cpp` and its two CMake lines restores `main`.

## Baseline, re-measured on this machine

`scripts/run_deepseek_v4_decode_profile.sh`, experiment 0034's chat operating
point reproduced: 18-token prompt, three GPUs, 216 GiB host ceiling, 0.95 VRAM
fraction, 32,768-token context, `--flash-attention --pin-resident-arena`,
`--detailed-timing`. Seven decode steps, 1.71863 s.

| term | ms/step | share |
|---|---:|---:|
| PCIe demand H2D — 693 MB/step at 6.9 GB/s | 100.3 | 41% |
| host arithmetic (residual) | ~104 | 42% |
| GPU kernels (MoE 28.0, FlashAttention 2.4) | 41.2 | 17% |
| **decode step** | **245.5** | **4.07 tok/s** |

1,244 matmul calls/step and 659 synchronizations/step, matching experiment 0037
exactly. Expert cache hit rate over the decode phase is 87.5% by weight, 1,089
weight misses in 7 steps.

## The contract the host kernel had to reproduce

Reading `deepseek_fp4_gate_up_kernel` and `deepseek_fp4_down_kernel`, the device
decomposition turns out to be exactly reproducible rather than merely
approximable. With `column = group * 32 + lane` and `group = warp; group += 8`,
CUDA thread `t = warp*32 + lane` owns columns `t, t+256, t+512, ...` — a plain
stride-256 walk. So a host kernel can hold 256 partial accumulators and visit
the same columns in the same order.

Four things had to match, and each was found by a failing test rather than by
reading:

1. **FMA contraction.** The build passes no `-fmad=false`, so nvcc contracts
   `sum += input * fp4_value(e) * scale` into `fma(input * fp4_value, scale,
   sum)`. The host uses `std::fma` in the same association.
2. **`reduce_block`.** Five shuffle-down rounds per warp, then the eight warp
   sums in lanes 0-7 with the rest zeroed and five more rounds. The rounds that
   add the zeroed lanes are performed, not skipped, because `-0.0 + 0.0` is
   `+0.0`.
3. **E8M0 edge cases.** `fp8_e8m0_scale_bits` maps `0xff` to NaN and a zero
   exponent to `2^-127`, not to float zero. The first draft returned `0.0F` for
   a zero exponent and `+inf` for `0xff`.
4. **The activation quantizer, which is the one that mattered.** Between gate/up
   and the down projection the device launches
   `quantize_activation_e4m3_kernel`: one dynamic power-of-two scale per
   128-column block, then FP8 E4M3 rounding of every activation. This is the
   checkpoint's declared dynamic E4M3 activation scheme. Omitting it put all 512
   outputs wrong by 2-5% — the mantissa width of E4M3, not a rounding step. A
   tolerance-based test would have passed it at `1e-2` and shipped a silently
   wrong expert.

## Result: bit-identical

`tests/test_deepseek_host_expert.cpp`, four cases, all passing under
`make check` (201/201):

- host AVX2 against host scalar: bit-identical;
- host against the device `enqueue_deepseek_moe` on byte-identical uploaded
  weights at the 512/256 shape, compared as `uint32_t` bit patterns;
- the same at the **production 4096/2048 shape**, whose loop trip counts differ
  (16 offset steps against 2);
- mismatched weight extents are rejected rather than read out of bounds.

End to end, the A/B arms emit identical token sequences for 64 generated
tokens, which is the claim that actually matters.

## Corrected per-phase attribution

The graph timers *are* emitted under `phases.decode.graph`; an earlier reading of
this baseline missed them and reconstructed the host share by subtraction. The
measured decomposition of the same 245.5 ms/step run:

| phase | ms/step | share |
|---|---:|---:|
| `moe` total | 150.25 | 61.2% |
| — of which `moe_prepare` (the demand-H2D wait) | 101.15 | 41.2% |
| — of which MoE execution | 49.10 | 20.0% |
| `attention` total | 70.17 | 28.6% |
| — query projection | 25.02 | 10.2% |
| — output projection | 23.30 | 9.5% |
| — score (43 CUDA dispatches, 2.4 ms of device time) | 13.47 | 5.5% |
| — KV | 7.94 | 3.2% |
| `mhc_pre` (prepacked) | 16.36 | 6.7% |
| `mhc_post` | 4.74 | 1.9% |
| router / output head / branch norm | 4.89 | 2.0% |

Two things this corrects. `mhc_pre` is 16.36 ms, not the 37.3 ms of experiment
0034 — `--prepack-mhc` is already default on `main` and has taken most of it,
which is why `--fast-mhc` does not port. And **attention is the term that
becomes `argmax_r` once the demand wait goes**, at 70.17 ms, with only 2.4 ms of
it being device time.

## Screened and rejected: bandwidth-weighted layer placement

`weighted_round_robin` builds the device schedule from `total_bytes`, i.e. VRAM
capacity, giving `[0,0,1,1,1,2,2,2]` — 25% of layers and experts on the 448 GB/s
5060 Ti against two 936 GB/s 3090s. That looked like a defect worth fixing.

It is not, and the reason is worth recording because the first estimate was
wrong by 3x. Layers are **sequential**, so per-token VRAM read time is the *sum*
over devices, not the max. Costing 11.93 GB/token of VRAM reads:

| schedule | ms/step |
|---|---:|
| capacity-weighted `[2,3,3]` (current) | 16.22 |
| bandwidth-weighted `[1,2,2]` | 15.52 |
| equal `[1,1,1]` | 17.37 |
| 3090s only `[0,1,1]` | 12.75 |

Bandwidth weighting is worth **1.04x**. Rejected. An earlier note in experiment
0050 claimed up to 3x here by dividing the aggregate bandwidth, which silently
assumed the three GPUs run concurrently; they do not. Dropping the 5060 Ti
entirely is worth 1.27x on this term but costs 15.4 GB of expert cache, which
is a different experiment.

## Result: throughput, and the three defects found on the way

`scratchpad/bench_host_expert.cpp`, real shape (hidden 4096, intermediate 2048,
13.37 MB per expert triplet), 56 distinct experts so the run streams rather than
replaying one L3-resident expert.

| revision | 1 thread | 28 threads |
|---|---:|---:|
| first working version | 0.58 | 9.35 GB/s |
| + cheap nibble expansion, + trimmed reduction | 1.20 | 15.22 |
| + hoisted E8M0 decode, + broken accumulator chain | 1.34 | 16.82 |
| + int8 LUT via `shuffle_epi8`, x2 folded into the scale | 1.85 | 18.64 |
| + dropped the per-row memsets | 1.96 | 21.77 |
| + eight accumulator chains, 64 columns per step | 2.07 | **26.23** |
| the same, under `numactl --interleave=all` | — | **28.62** |

against **6.9 GB/s** for the PCIe path it replaces: a 4.1x on the mechanism.

Folding the factor of two into the group scale is exact and worth recording:
E2M1 doubled is an exact int8, so one `_mm_shuffle_epi8` decodes sixteen
nibbles *including sign*, replacing a seven-op permute-and-sign sequence. The
scale is a power of two, so halving it and doubling the weight leaves the FMA's
exact product unchanged — the bit-identity tests hold across the change.

The defects, recorded because each cost more than it looked like it would:

- `_mm256_setr_epi32` from eight extracted scalars is roughly fifteen uops per
  eight FMAs. `_mm_unpacklo_epi8` on a duplicated 4-byte load plus
  `_mm256_cvtepu8_epi32` is three.
- `reduce_block` as literally written is ~280 scalar operations and 256 float
  copies per output row, over 8,192 rows per expert. Only lane 0 is ever read,
  and lane 0 never reads above the current offset, so half of every round is
  dead — dropping it is the same arithmetic, not a reassociation.
- Sixteen dependent FMAs into one accumulator is latency-bound at ~4 cycles
  each. Reordering the loop nest so consecutive iterations touch different
  accumulators preserves each accumulator's ascending column order, which is
  what bit-identity actually requires. Four chains was still not enough to
  cover the ~25-cycle load-to-accumulate chain; eight was.
- Zero-initialising a 512-float scale buffer and clearing 256 partials *per
  output row*, over 8,192 rows per expert, is 24 MB of memset per expert. Only
  the first `groups` scales are read and every partial is written
  unconditionally, so both are dead.
- A lambda does not inherit the enclosing function's `target("avx2,fma")`
  attribute. Factoring the inner step into one produced `inlining failed in
  call to always_inline` for every intrinsic — and because the build error was
  filtered out of a grep, the next benchmark silently measured the stale
  binary. Always check the build actually succeeded before reading a number.

At 28 threads the kernel sits at 26-29 GB/s against a 76 GB/s DRAM ceiling, so
it is uop-bound rather than memory-bound; `numactl --interleave=all` is worth a
further 1.09x, which is the same unbound-arena defect experiment 0026 found in
`Dsv4ResidentWeightStore::stage`.

## What it is worth — and why the kernel should not be optimised further

693 MB/step of miss bytes, against 41.2 ms/step of GPU kernel time the CPU work
can hide under:

| path | ms/step for the miss bytes | serial residue |
|---|---:|---:|
| PCIe today, serial with compute | 100.3 | 100.3 |
| host kernel measured, 16.82 GB/s | 41.2 | **0.0** |
| host kernel optimised, ~30 GB/s | 23.1 | 0.0 |
| DRAM ceiling, 76 GB/s | 9.1 | 0.0 |

At 16.82 GB/s the host work already fits **entirely inside** the GPU kernel time
of the same step, so the residue is zero and the PCIe term goes away. Every
further GB/s is invisible. Per the charter's rule that a mechanism must reduce
`argmax_r`, optimising this kernel past 16.82 GB/s is wasted work until
something else makes it the bottleneck again.

Projected step, PCIe term removed: **245.5 → 145.2 ms/step, 4.07 → 6.89 tok/s,
1.69x.** This is a projection from measured parts, not an end-to-end result. It
assumes the miss experts of a layer can be issued to host workers concurrently
with that layer's cached experts on the GPU, which is the scheduling change that
has not been built.

## Wiring it in: three rejected policies before a working one

The kernel was the easy half. Deciding *which* experts to divert took four
measured attempts, and the first three were regressions. All four produced
byte-identical output text, which is the correctness gate and never moved.

**v1 — divert every miss, no admission. 404.5 ms/step against 229.3 (1.76x
regression).** Demand H2D fell to exactly 0.0 MB/step, which is the tell: *all*
258 experts/step ran on the host, not the ~39 that miss. The cache only ever
loads through `acquire()`, so diverting every miss starves it permanently — it
never fills, so everything misses, forever. A self-reinforcing feedback loop
created by the mechanism itself.

**v2 — divert every miss, plus write-behind admission via `request_prefetch`.
951.5 ms/step (4.15x regression).** The reasoning was that an expert the
scheduler diverted is a *known* use, so admitting it afterwards is deferred
caching rather than the prediction experiment 0035 rejected. That part is
sound; the arithmetic is not. The cache is already full — `used_bytes` equals
`capacity_bytes` on all three devices — so every admission evicts a live entry,
which then misses, which requests another admission. Measured 217 GB of
prefetch H2D in a 64-token run, essentially all of it `wasted_prefetch_bytes`,
saturating the link so hard that *attention* went from 70 ms to 591 ms/step.
Write-behind into a full cache is strictly negative.

**v3 — divert decode misses only; let prefill fill the cache. 290.9 ms/step
against 224.2 (1.30x regression).** The diversion now works as designed: decode
misses go to the host, everything else hits the cache, demand H2D is zero. But
it still loses, and the reason is a granularity mismatch that the byte counts
name exactly. The cache misses at *weight* granularity and the device path
transfers only the matrices it lacks — 518.6 MB/step. The host path has to read
the whole triplet. Diverting on "any of three missing" fired on 118
experts/step against 38.8 experts' worth of real misses: **1.58 GB of host
reads to displace 519 MB of transfer.** Three times the bytes to avoid the
transfer, which no bandwidth ratio can rescue.

**v4 — divert only experts whose three matrices are all missing. 283.0 ms/step
against 226.3 (1.25x regression), and byte-for-byte the same counters as v3.**
The change made *no difference at all*, which falsifies the v3 diagnosis above:
misses are already clustered at expert granularity, because the cache admits and
evicts an expert atomically. "Any missing" and "all missing" select the same
experts. The 118-per-step figure was my own arithmetic error — the cache's
hit/miss counters include the 341 attention projections per step, not just the
777 expert-weight lookups, so dividing the lookup difference by three was
meaningless.

## Why it actually loses: there is not enough work per barrier

Measured directly instead of inferred. MoE *execution* — the phase excluding the
demand wait — goes from 48.75 ms/step on the reference to 147.05 ms/step with
the host path. That is 511 MB/step of diverted experts moving at an effective
5.2 GB/s, against 26 GB/s for the same kernel standalone.

The standalone benchmark gives each thread a whole expert. The runtime cannot:
decode routes about **0.9 diverted experts per layer**, so one expert has to be
split across the pool, twice per layer, with a barrier between the phases.
`scratchpad/bench_pool.cpp` prices exactly that shape:

| pool workers | empty `parallel_for` | 43 layers x 1 expert | effective |
|---|---:|---:|---:|
| 28 | 73.6 us | 58.35 ms | **9.85 GB/s** |
| 14 | 24.3 us | 52.62 ms | 10.93 GB/s |
| 8 | 13.7 us | 64.54 ms | 8.91 GB/s |

The empty barrier is only 6.3 ms/step of the loss at 28 workers; the rest is
straggler cost — each phase takes as long as its slowest tile, 86 times per
step, on tiles of ~150 us. **Intra-expert parallelism caps the host path at
about 10 GB/s, which is 1.4x PCIe, not the 4.1x the kernel can do.** At that
rate 511 MB/step is 46-98 ms of work against a ~28 ms GPU MoE window to hide
under, so it is exposed rather than free.

## The correctness bug the throughput A/B found

The v1-v4 arms all reported identical output text, which was wrong: the
comparison only looked at the leading characters. Comparing properly, the arms
**diverge at generated token 13 of 64** — a greedy argmax flip, meaning the host
path was not bit-exact in production even though every unit test passed.

The cause is a second missed kernel, the same class as the first.
`enqueue_deepseek_moe` launches `quantize_activation_e4m3_kernel` **twice**:
once on the MoE *input* before gate/up, and once on the intermediate activation
before the down projection. The first record of this experiment found the
second and missed the first.

It survived four tests because of the test data. The inputs were
`(index % 9 - 4) * 0.125`, and every one of those values is exactly
representable in E4M3, so the quantizer the host path was skipping was a no-op
on exactly the numbers being compared. Regenerating the fixtures from
non-power-of-two steps makes the test fail before the fix and pass after it.

**Test data must not be exact in the format under test.** A bit-identity claim
verified against inputs that cannot express the difference is not verified.

## Decision: the scheduling change is rejected; the kernel is kept

With the input quantizer fixed, the arms emit **identical token sequences**, so
the path is bit-exact end to end through the real model — the correctness claim
holds. The throughput claim does not.

Final A/B, three interleaved repetitions at the same operating point,
`--host-expert-misses` against the default:

| arm | median ms/step | tok/s | all runs |
|---|---:|---:|---|
| reference | 225.6 | 4.43 | 224.4, 240.2, 225.6 |
| host expert | 296.6 | 3.37 | 312.7, 289.8, 296.6 |

Diverted experts were 7,691 per run in all three repetitions, so the policy is
deterministic and the arms are not drifting. Phase detail from the first pair:

| | moe | moe_prepare | attention | host expert work |
|---|---:|---:|---:|---:|
| reference | 127.4 | 78.1 | 71.4 | — |
| host expert | 169.2 | 2.1 | 88.2 | 148.6 ms/step over 122.1 experts |

**0.76x. Rejected.** The demand wait is genuinely eliminated — 78.1 ms/step to
2.1 — and it is replaced by 148.6 ms/step of host expert work over 122.1
diverted experts, an effective 11.0 GB/s.

The 122.1 figure is the finding. The reference misses only 38.8 experts' worth
of bytes per step, so the mechanism is diverting **three times more work than
there is miss to remove**, and it does so because of itself: with no
decode-time admission the resident set is frozen at whatever prefill left, goes
stale, and the miss rate rises from ~15% to ~47%. That is the v1 feedback loop
again, weakened but not removed.

So there is no configuration of this policy that wins here. Either the cache
keeps learning during decode, which means paying the demand stall, or it does
not, which triples the work the host has to absorb. And the host cannot absorb
it, because one expert per layer split across a 28-thread pool runs at 10 GB/s,
not 26.

**The regime that would have been required**, recorded so this is not re-run at
a configuration chosen to make it pass:

- **Four or more diverted experts per layer**, so the pool gets enough work per
  barrier to reach its standalone rate. Speculative decoding is the natural
  source: DSpark at block size 5 verifies five tokens per forward pass, which
  multiplies expert work per layer by roughly five against the *same* two
  barriers. This kernel is a prerequisite for that combination, not a
  competitor to it.
- **Or a decode-time admission path that does not evict live entries** — a
  reserved slice of cache for write-behind, rather than `request_prefetch` into
  a full LRU.
- **Or a longer context**, where the demand term grows against a fixed GPU
  window.

The kernel itself is kept and stays gated off. It is bit-exact against the
device at two shapes, it is 4.1x the PCIe path standalone, and it is the piece
that has to exist before any of the above can be tried.

## Why this does not reach 50 tok/s, stated plainly

50 tok/s is 20.0 ms/step. The PCIe term is 100.3 ms of 245.5. Removing it
completely — the best this mechanism can ever do — leaves 145.2 ms, of which
~104 ms is host scalar arithmetic and 41.2 ms is GPU kernels. **Both remaining
terms have to go as well**, and neither is touched by anything in this
experiment.

## Corrections to the first version of this record

- The projected 1.69x from removing the demand term assumed the host work would
  hide entirely under GPU kernel time. It does not, because the *rate* it
  assumed (16.82 GB/s) is only reachable with a whole expert per thread, and
  decode has 0.9 diverted experts per layer. Measured 11.0 GB/s in situ.
- "Optimising the kernel past 16.82 GB/s is wasted work" was wrong twice over:
  the kernel went on to 26.23 GB/s standalone with straightforward changes, and
  the binding constraint was never the kernel's rate but the parallel
  decomposition around it.
- The claim that bandwidth-weighted layer placement was worth up to 3x was an
  arithmetic error; measured, it is 1.04x. See the screen above.

## Next, in order

Wiring is done and rejected; what follows is ordered by the measured
decomposition of the 225.6 ms/step baseline, not by the host-expert path.

1. **Attention, 71.4 ms/step, of which 2.4 ms is device time.** The largest
   term that is not the demand wait. The score path re-uploads the whole KV
   working set every layer every token — `CudaBackend::flash_attention`
   host-copies each row into pinned staging and H2Ds it, 43 times per step.
   Only one row per layer is new. A device-resident KV with incremental append
   removes almost all of it, and the saving grows with context: 7.8 MB/step at
   this 82-position operating point, but ~56 MB/token at 3,565 positions.
2. **The 341 serial projections, ~48 ms/step of query and output projection.**
   Every `CudaBackend::matmul` is H2D, kernel, D2H, `cudaStreamSynchronize`.
   `exp/dsv4-device-activations` moved exactly one chain of these (`wo_a` to
   `wo_b`) and was rejected for being within variance — which is the expected
   result for 1 of 341. The experiment to run is the whole layer, not one pair.
3. **CUDA graph capture**, which requires item 2 plus collapsing the 903
   per-expert MoE matmuls into one kernel reading routing from a device buffer.
   There is no `cudaGraph` call anywhere in the runtime today, and LvLLM's
   changelog prices graph capture at 2x on the same class of model.
4. **DSpark**, last, and now with a second reason: besides amortising weight
   reads, block size 5 is what would give the host expert path enough work per
   barrier to run at its standalone rate.

## Artifacts

`kernels/cpu/dsv4_fp4_expert.cpp`, `include/strata/deepseek_host_expert.hpp`,
`tests/test_deepseek_host_expert.cpp`,
`scripts/run_deepseek_v4_decode_profile.sh`. Baseline JSON under
`results/deepseek-v4-decode-profile/` (ignored). The throughput harness is
throwaway and lives in the session scratchpad.
