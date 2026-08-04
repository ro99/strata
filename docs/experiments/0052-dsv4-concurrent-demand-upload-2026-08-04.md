# Experiment 0052 — the routed-expert demand upload is serial across three PCIe links

Status: **code kept; measurement invalid and not re-run.** The mechanism is
correct and the defect it removes is real, but every A/B in this record was run
on a **Debug** binary, and the mechanism is worth ~6% against a goal that needs
11-30x. It is not worth more wall time. See "What this record got wrong".

The demand wait that experiments 0034, 0040 and 0051 all recorded as "PCIe
transfer" is not, in the main, transfer. Three devices with three independent
PCIe links are driven one at a time, because `CudaBackend::upload` waits out
each copy where it is issued and the MoE prepare loop runs on one host thread.
Measured on the existing baseline JSON, the per-device demand wait **sums** to
99.1 ms/step against a per-device **maximum** of 40.2 ms/step. That difference
is `Σ_serial`, not `W_r/B_r`, and no reduction in transferred bytes addresses it.

## Contract

- **Hypothesis.** `moe_prepare`'s 101.1 ms/step is dominated by cross-device
  serialization of demand H2D. Deferring each upload's stream synchronize until
  every acquire for a layer has been issued lets the three links run
  concurrently, reducing `Σ_serial` without changing a transferred byte.
- **Primary metric.** Median decode ms/step over three interleaved repetitions
  at the chat operating point. Mechanism-level metric: `moe_prepare_seconds`
  and the per-device `upload_wait_seconds` as both a sum and a maximum.
- **Measured bottleneck, and the sign on every other resource.** The term
  reduced is host wall time blocked in `cudaStreamSynchronize` for weight
  uploads — 99.1 ms/step, the sum over three devices. PCIe *bytes* are
  unchanged at 693 MB/step, GPU kernel work is unchanged, host arithmetic is
  unchanged, VRAM is unchanged, and no allocation is added. Instantaneous PCIe
  demand rises from ~6.6 to ~12.5 GB/s aggregate, which is below the sum of the
  three links' measured solo rates, so no link is oversubscribed. Host DRAM
  read demand rises with it, to 16% of the 76 GB/s node ceiling.
- **Correctness gate.** Byte-identical generated token sequence against the
  serial arm over 64 tokens, compared as full id lists rather than leading
  characters, plus `make check`. This is an ordering change with no numerics
  content: any output difference is a bug, not a trade.
- **Memory ceiling.** Unchanged — 216 GiB host admission, 0.95 VRAM fraction.
- **Rollback.** `--serial-expert-upload` restores the pre-0052 behaviour
  exactly. Revert if the median is not below baseline beyond run variance, or
  if any output byte changes.
- **Kill criterion, declared before building.** The isolated probe prices the
  mechanism at 49.4–50.0 ms/step. If end-to-end `moe_prepare` does not fall by
  at least **35 ms/step** — 70% of the isolated figure — the runtime is not
  realising the overlap the probe shows, and the mechanism is rejected rather
  than tuned.

## Instantiating the cost model on the existing baseline

No new profiling run was needed to find this; it is in the JSON experiment 0051
already produced (`results/deepseek-v4-decode-profile/smoke/run-01`, 7 decode
steps, `--detailed-timing`, 245.5 ms/step).

| device | link | demand bytes | upload wait | implied rate |
|---|---|---:|---:|---:|
| GPU0 RTX 5060 Ti | PCIe 3.0 x8 | 183.4 MB/step | 27.5 ms/step | 6.66 GB/s |
| GPU1 RTX 3090 | PCIe 3.0 x8 | 233.0 MB/step | 40.2 ms/step | 5.80 GB/s |
| GPU2 RTX 3090 | PCIe 3.0 x16 | 276.9 MB/step | 31.4 ms/step | 8.82 GB/s |
| **sum** | | **693.3 MB/step** | **99.1 ms/step** | 7.00 GB/s |

`moe_prepare` is 101.1 ms/step. The upload waits account for 98% of it, and
they add rather than max. Each *link* is running at a plausible fraction of its
rated width — this is not a slow-link problem — but only one is ever running.

`nvidia-smi topo -m` confirms the links are independent enough to matter: GPU0
hangs off NUMA node 0, GPU1 and GPU2 off node 1 sharing a host bridge.

## The cheapest falsifying measurement, run before any runtime code

`scratchpad/bench_h2d_overlap.cu` reproduces the production access pattern
rather than a warm reused buffer: a 4 GiB pinned arena, randomly placed 4.456 MB
slices (one FP4 expert matrix — 4.19 MB packed plus 0.26 MB of E8M0 scales), the
measured per-device byte split, and the transfers shuffled because the rank loop
visits experts in routing order, not device order. Two arms, five interleaved
repetitions, median:

| arm | ms/step | aggregate GB/s |
|---|---:|---:|
| sync-each (what the runtime does today) | 105.37 | 6.56 |
| deferred, one sync per device (proposed) | **55.42** | **12.46** |
| | | |
| GPU0 alone | 31.32 | 5.83 |
| GPU1 alone | 32.32 | 7.17 |
| GPU2 alone | 41.64 | 6.64 |
| sum of per-device alone | 105.28 | |
| max of per-device alone | 41.64 | perfect-overlap floor |

**1.90x, 49.95 ms/step.** The sync-each arm reproduces production to within 6%
(105.4 ms against 99.1 measured, 6.56 GB/s against 7.00), which is what makes
the probe trustworthy: it is measuring the same defect.

Two things the probe settles that would otherwise have been separate
experiments:

- The links do **not** share a bottlenecking uplink. Had they, both arms would
  have measured the same and no runtime code should have been written.
- **NUMA placement is irrelevant to this term.** Under `numactl
  --interleave=all` the deferred arm measures 55.76 ms against 55.42 — noise.
  The residual 13.8 ms above the perfect-overlap floor is PCIe and host-bridge
  contention, not DRAM or QPI. This does not contradict experiment 0026's NUMA
  finding, which is about the *CPU* reading the arena, not about DMA.

## Mechanism

`CudaBackend::upload` gains an `UploadCompletion` mode. Deferred leaves the
copies in flight and sets a per-device flag; `synchronize_uploads(device)` waits
them out and attributes the wait. There is exactly one stream per device and
every consumer of a weight is issued on it, so the device side is already
ordered — what the immediate wait actually protects is the *host source buffer*.
Deferral is therefore gated in `load_dsv4_cuda_linear` on both payload spans
coming from the resident arena, which outlives the batch; a checkpoint read or
the `wo_a` FP8-to-BF16 conversion decodes into a temporary and stays
synchronous.

`Dsv4WeightCache::UploadBatch` is the scope. It holds a `DemandGuard` for its
whole life so the prefetch worker cannot issue on the same stream while copies
are in flight, and releases it only after the wait. The MoE prepare loop opens
one and closes it before the first `enqueue_deepseek_moe`, so the wait stays
inside `moe_prepare` where the serial version paid it.

## Correction to the operating point this is measured at

The first A/B arm was launched with `--detailed-timing` and measured 649.6
ms/step, against the 225.6 ms/step this baseline is quoted at. That is not a
regression; the two numbers are different operating points, and the run was
discarded and relaunched rather than reported.

Worth recording because the handoff and experiment 0051 both invite the
confusion: **0051's per-phase decomposition table is from an 8-token
`--detailed-timing` run at 245.5 ms/step, while the 225.6 ms/step headline is a
64-token run without it.** The phase shares are not directly transferable
between them — attention alone measures 70.2 ms/step in the first and 138.8
ms/step in the second, because the KV gather and score both grow with position.
Per the charter, `τ(L)`, not `τ`. The A/B below is run without detailed timing,
at 64 tokens, which is the configuration the 225.6 ms/step figure came from.

## Result — and why it is not reportable

Three interleaved repetitions per arm, at two operating points. Every arm
emitted **identical token sequences** and moved **identical bytes** (the
correctness gate held throughout), and `moe_prepare` fell by a consistent
~23 ms/step at both points, which is what the mechanism predicts.

| | 64 tokens | 8 tokens |
|---|---:|---:|
| moe_prepare, serial → concurrent | 92.83 → 69.81 | 105.08 → 81.77 |
| upload wait, sum over devices | 75.71 → 53.65 | 99.82 → 76.86 |
| ms/step | 648.74 → 611.53 | 646.21 → 635.19 |

**These absolute numbers are invalid.** `build/` was configured `Debug`;
`cmake --build build` does not re-run configure, so it never picked up the
`-DCMAKE_BUILD_TYPE=Release` the Makefile passes. Host-arithmetic phases are
inflated roughly 11x (`mhc_pre` 186 ms/step against a recorded 16.4), while
GPU-wait terms match the Release record exactly — which is precisely what made
the corruption look like a finding instead of a broken build.

The mechanism-level delta (~23 ms/step of `moe_prepare`) is a GPU-wait term and
is expected to survive a Release build largely unchanged, since it is the same
bytes over the same links. On a ~226 ms Release step that would be ~1.10x. **It
has not been re-measured and should not be quoted until it is.**

## What this record got wrong

Three errors, all mine, recorded so they are not repeated:

1. **The probe measured a batch size the runtime cannot deliver.** It overlapped
   a whole step's 155 transfers at once and reported 1.90x. The runtime can only
   batch one layer, because layer N+1's routing does not exist until layer N's
   output does. At the production miss rate of ~0.9 experts/layer — and misses
   cluster at expert granularity, so a miss is three matrices on *one* device —
   **only 6 of 43 layers touch two or more devices.** Re-priced at layer
   granularity the mechanism is 1.25x / 12.9 ms, not 1.90x / 49.6 ms. This is
   the same granularity error experiment 0051 records for the host-expert
   policy, made again after reading it.
2. **The kill criterion inherited that error.** 35 ms/step was 70% of an
   unachievable figure. Derived correctly it would have been ~16 ms — and the
   right question at that point was whether a 7% mechanism deserved a session
   at all. It did not.
3. **The build was never verified before measuring.** ~45 minutes of A/B matrices
   produced two confident, wrong conclusions that were reported before being
   caught: that mHC scales superlinearly with context (it is flat — 185.9
   ms/step at 8 tokens, 188.6 at 64), and that the realistic-length baseline is
   1.6 tok/s.

## Standing

The code is kept because it is correct, additive, and strictly ordered better
than what it replaces: identical bytes, fewer synchronizations, byte-identical
output, `--serial-expert-upload` restores the old behaviour exactly. It is not
promoted on a throughput claim, and the next session should not spend time
re-measuring it — the ranked list in experiment 0050 has larger terms.

## Artifacts

`scripts/run_deepseek_v4_concurrent_upload_ab.sh`,
`scripts/summarize_concurrent_upload_ab.py`,
`scripts/run_deepseek_v4_decode_profile.sh`. Results under
`results/deepseek-v4-concurrent-upload-ab/` (ignored). The probe is throwaway
and lives in the session scratchpad as `bench_h2d_overlap.cu`.
