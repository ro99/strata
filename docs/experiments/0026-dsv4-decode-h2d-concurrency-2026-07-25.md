# Experiment 0026 — decode H2D concurrency, and the rejection of the "5.9x residual"

## Contract

- Hypothesis under test (inherited from experiment 0024's "Next term"): decode
  H2D reaches 6.74 GB/s effective against 11.9 GB/s for the same transfer in
  isolation, so pinning captured 3.33x of an available ~5.9x. The residual was
  attributed to `CudaBackend::upload` issuing H2D on the compute stream and
  synchronizing it inline, and the proposed fix was a per-device dedicated copy
  stream with `cudaStreamWaitEvent` ordering.
- Primary metric: effective decode demand-H2D rate, and the projected effect on
  median decode steps/s.
- Kill criterion, stated before the work: if the mechanism named in the
  hypothesis — deferring the inline synchronize — does not raise the measured
  rate, the hypothesis is rejected regardless of what other mechanism the probe
  happens to suggest.
- Correctness: no code changed. This experiment is measurement only.

## Result: the hypothesis is rejected, and its premise was wrong

**There is no 5.9x.** The 11.9 GB/s figure in experiment 0024 was measured on
one device and generalized to "the link". The three links differ by 2x, and each
is already running at its own slot's practical rate.

## Cheap measurement first

A standalone probe reproduces the production access pattern exactly: a
registered anonymous mapping sampled at cold, randomly placed, non-repeating
4,456,448-byte offsets (4,194,304 B of FP4 weights plus 262,144 B of scales,
which is what every routed-expert matrix is), issued as the runtime issues them.
Five arms, 512 copies per device, ~40 seconds total:

| arm | GB/s | ms/copy |
|---|---:|---:|
| 1 single device, sync per copy — dev 0 | 5.740 | 0.776 |
| 1 single device, sync per copy — dev 1 | 11.793 | 0.378 |
| 1 single device, sync per copy — dev 2 | 6.288 | 0.709 |
| 2 single device, batch 64 — dev 0 | 5.984 | 0.745 |
| 2 single device, batch 64 — dev 1 | 11.708 | 0.381 |
| 2 single device, batch 64 — dev 2 | 6.617 | 0.673 |
| 3 round-robin 3 devices, sync per copy | 7.250 | 0.615 |
| 4 3 devices in flight, sync per layer | **16.470** | 0.271 |
| 5 3 threads, 3 devices, batch 64 | 16.570 | 0.269 |

Arm 3 is the production pattern: one host thread, `cudaSetDevice` and an inline
`cudaStreamSynchronize` per copy, round-robin across devices. It reproduces
production well — 7.250 GB/s against the runtime's measured 6.742 GB/s
(73,852,256,256 B in 10.954 s), the 7% residual being the cache lookup, arena
extent lookup and lease bookkeeping the probe does not model.

Three things follow, and only the third survives.

**The inline synchronize costs nothing.** Arm 1 against arm 2, on the same
device, is +4.3%, -0.7% and +5.2%. That is inside noise. The mechanism named in
the hypothesis — deferring the sync so a copy is not blocked — recovers nothing,
because a 4.46 MB copy already saturates its link and there is no second copy
queued behind it to benefit from the deeper queue.

**The links are unequal and each is at rate.** Under load `nvidia-smi` reports
device 0 at gen3 x8, device 1 at gen3 x8, device 2 at gen3 x16; the GPUs are a
RTX 5060 Ti on NUMA node 0 and two RTX 3090s sharing a host bridge on NUMA node
1. Whatever the reported width, the measured 5.98 / 11.71 / 6.62 GB/s are the
hardware's figures, not a software defect. Serial round-robin over three links
of those rates cannot exceed 7.250 GB/s, and production is at 6.742 — **93% of
the ceiling the current access pattern can possibly reach**. This is the
falsification: the gap experiment 0024 called a serialization defect is, for the
most part, simply three unequal links used one at a time.

**Cross-device concurrency is the only real term.** Arm 3 to arm 4 is 2.27x,
and arm 5 shows a thread pool adds nothing over it: one host thread issuing to
per-device copy streams and synchronizing once per layer captures the whole
effect, because wall time becomes `max_d` instead of `Σ_d`.

## But the production miss pattern barely offers that concurrency

A 2.27x on the mechanism is not a 2.27x on the term. Replaying the recorded
decode route trace from experiment 0024's pinned arm against the runtime's own
cache model — `expert_device(e) = schedule[e % 8]` with
`schedule = [0,0,1,1,1,2,2,2]`, per-device LRU over
`capacity_bytes - pinned_bytes` — reproduces the measured workload closely
(18,564 simulated weight misses against 16,572 measured; 10.702 s of modelled
serial upload against 10.954 s of measured demand wait) and gives the
distribution that matters:

| devices missed on, per layer-step | layer-steps | share |
|---|---:|---:|
| 0 | 2,086 | 38.2% |
| 1 | 2,118 | 38.8% |
| 2 | 1,001 | 18.3% |
| 3 | 256 | 4.7% |

77% of decode layer-steps have nothing to overlap. Only ~1.13 experts miss per
layer-step, and an expert's three matrices all live on one device, so the
typical miss burst is three copies to a single link with the other two links
idle. Applying the per-device measured rates:

| | seconds |
|---|---:|
| serial upload (modelled) | 10.702 |
| perfectly overlapped across devices | 8.071 |
| **ceiling on this term** | **1.326x** |

Against a 40.35 s decode that is 2.69 s, or **1.071x end-to-end at a ceiling of
zero implementation overhead**. Allowing each device's MoE to be enqueued as
soon as its own uploads land — overlapping the 0.63 ms/layer-step of maximum
device MoE kernel time as well — raises the ceiling to roughly 1.077x.

## Cost model at the operating point

511-token prompt, 127 decode steps, 40.351 s, **317.7 ms/step**
(`results/deepseek-v4-pinned-arena-ab/pinned/run-01`):

| term | ms/step | share |
|---|---:|---:|
| MoE demand wait (H2D, serial) | 86.25 | 27.2% |
| attention score (host scalar) | 51.97 | 16.4% |
| MoE compute | 46.92 | 14.8% |
| mHC pre | 37.45 | 11.8% |
| attention query projection | 26.96 | 8.5% |
| attention output projection | 24.64 | 7.8% |
| MoE prepare, excluding wait | 11.89 | 3.7% |
| attention KV | 8.92 | 2.8% |
| mHC post | 4.66 | 1.5% |
| output head | 4.37 | 1.4% |
| router | 2.68 | 0.8% |
| branch norm | 1.06 | 0.3% |
| unattributed | 9.42 | 3.0% |

`argmax_r` is the H2D link at 86.25 ms/step. A mechanism that perfectly
parallelizes it across the three links removes 21 of those 86 ms. That is the
whole prize, and it is 6.7% of a step.

## Decision

**Do not build the copy stream on this evidence.** The hypothesis as stated is
rejected: the inline synchronize is not a defect, and the "available ~5.9x" does
not exist. The one real mechanism — cross-device concurrency — has a measured
ceiling of 1.326x on a term worth 27% of the step, which is 1.07x end-to-end
before any implementation cost, on a change that must get lease lifetime and
`moe_in_flight` ordering right against a cache that currently guarantees a
weight is resident before it is read.

Recording the regime that would have been required, per the charter's rule
against manufacturing a favourable one: this mechanism needs miss bursts that
span devices. It would pay if the per-layer-step miss count were several times
higher — a smaller VRAM cache, a larger expert set, or a longer context that
evicts more per step — or if the expert-to-device map were changed so one
expert's three matrices split across links. Neither is the operating point that
was asked about, and the second trades a guaranteed 3x increase in per-expert
issue count for a 1.33x ceiling.

## What the probe defect cost, again

Experiment 0024 already recorded that `strata-topology-probe` reported pinned
and pageable H2D as identical because it timed a warm reused buffer. The same
probe also measures one device at a time and never concurrently, which is why
the one property that turned out to matter — that three unequal links are used
serially — was invisible in it, and why a 5.9x that does not exist was carried
forward into a plan. The probe is corrected in this branch: it now samples cold
randomly-placed slices of a large registered mapping, and reports concurrent
aggregate H2D across all devices alongside the per-device figures.

## Where the term actually goes

581 MB/step is moved in 43 bursts of ~13.5 MB, each serial with the compute
that consumes it. Aggregate link capacity is 24.3 GB/s, so those bytes have a
24 ms/step floor against 86 ms measured — and the gap is latency and idle link
time, not bandwidth. Reaching that floor requires taking the transfer off the
critical path rather than widening it, which means prefetch. Experiment 0022's
simulation already cleared its gate for two past-only predictions at 0.75
confidence (7.61% fewer modelled bytes, 84.66% useful) and authorized an opt-in
runtime experiment that has not been run. That, not a copy stream, is the branch
this term deserves next — and per the charter its predictions stay advisory.
