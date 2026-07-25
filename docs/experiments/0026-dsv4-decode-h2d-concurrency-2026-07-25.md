# Experiment 0026 — decode H2D: rejecting the copy stream, finding the NUMA term

## Contract

- Hypothesis under test, inherited from experiment 0024's "Next term": decode
  H2D reaches 6.74 GB/s effective against 11.9 GB/s for the same transfer in
  isolation, so pinning captured 3.33x of an available ~5.9x. The residual was
  attributed to `CudaBackend::upload` issuing H2D on the compute stream and
  synchronizing it inline, and the proposed fix was a per-device dedicated copy
  stream with `cudaStreamWaitEvent` ordering.
- Primary metric: effective decode demand-H2D rate and its projected effect on
  median decode steps/s.
- Kill criterion, stated before the work: if deferring the inline synchronize —
  the mechanism the hypothesis names — does not raise the measured rate, the
  hypothesis is rejected regardless of what else the probe suggests.
- Correctness: no runtime code changed. This experiment is measurement only.

## Result

The hypothesis is **rejected**. The inline synchronize costs nothing, and the
"~5.9x available" was a measurement of a configuration the runtime does not
have. Two smaller real terms were found in its place, and neither is a copy
stream.

## Cheap measurement first

`strata-topology-probe` gained a cold-slice stage that reproduces the
production access pattern: a registered anonymous mapping sampled at randomly
permuted, non-repeating 4,456,448-byte offsets — one FP4 routed-expert matrix,
4,194,304 B of weights plus 262,144 B of scales — issued as the runtime issues
them. Four arms, 512 copies per device, about 40 seconds.

Default placement, three independent runs, reproducible to within 1%:

| arm | GB/s |
|---|---:|
| single device, sync per copy — dev 0 / 1 / 2 | 5.82 / 7.14 / 6.38 |
| single device, batch 64 — dev 0 / 1 / 2 | 5.87 / 7.19 / 6.70 |
| serial round-robin, 3 devices, sync per copy | 6.37 |
| **3 devices in flight, one sync per layer** | **10.52** |
| 3 threads, 3 devices, batch 64 | 10.79 |

**The inline synchronize costs nothing.** Arm 1 against arm 2 on the same
device is +0.9%, +0.7% and +5.0%. A 4.46 MB copy already saturates its link and
there is no second copy queued behind it, so a deeper queue has nothing to
recover. The mechanism the hypothesis named is worth zero, which is the kill
criterion.

**Serial round-robin is the real ceiling of the current pattern.** 6.37 GB/s
against production's measured 6.742 GB/s (73,852,256,256 B in 10.954 s).
Production is *at* the ceiling its access pattern permits, not an eighth of a
link's rated figure. There is no 5.9x.

**Cross-device concurrency is the one real mechanism**, worth 1.66x here, and a
thread pool adds nothing over a single host thread issuing to per-device
streams — wall time becomes `max_d` instead of `Σ_d`.

## Where the 11.9 GB/s came from: the arena has no NUMA policy

The first run of the probe reported device 1 at 11.79 GB/s and could not be
reproduced afterwards. Per the charter that is a defect report, not an outlier
to discard, and the cause is the finding of this experiment. GPU 0 sits on NUMA
node 0; GPUs 1 and 2 share a host bridge on node 1. Re-running the identical
probe under three placement policies:

| | dev 0 | dev 1 | dev 2 | round-robin | overlapped |
|---|---:|---:|---:|---:|---:|
| `numactl --membind=0` | 5.87 | 7.18 | 6.71 | 6.49 | 10.80 |
| `numactl --interleave=all` | 5.91 | 9.68 | 6.56 | 7.00 | 14.71 |
| `numactl --membind=1` | 5.98 | **11.93** | 6.64 | 7.40 | **16.45** |

Device 1's H2D rate is **1.66x** on source placement alone, and the aggregate
overlapped rate is **1.52x**. The 11.9 GB/s recorded in experiment 0024 is
therefore a real figure — it is device 1 reading a node-1-local source — but it
is not what the runtime gets, because `Dsv4ResidentWeightStore::stage` allocates
the arena with `mmap(MAP_PRIVATE | MAP_ANONYMOUS)` and no NUMA policy, and fills
it from a pool of read workers, so each tensor's pages land on whichever node
its staging worker happened to run on. Production's measured 6.742 GB/s sits
between the `membind=0` and `interleave` round-robin figures, which is what an
unbound arena should look like.

Two things follow. First, generalizing one device's isolated figure to "the
link" is what produced a 5.9x that does not exist. Second, **the placement of
the resident arena is an unmanaged input to the step's bottleneck resource**,
which also makes the term non-deterministic run to run.

Device 2 does not respond to placement at all (6.71 / 6.56 / 6.64) despite
reporting gen3 x16 under load, where its two neighbours at x8 reach 5.9 and
11.9. That is unexplained and is recorded here as an open question, not
attributed.

## The production miss pattern barely offers the concurrency

A 1.66x on the mechanism is not a 1.66x on the term. Replaying the recorded
decode route trace from experiment 0024's pinned arm against the runtime's own
cache model — `expert_device(e) = schedule[e % 8]` with
`schedule = [0,0,1,1,1,2,2,2]`, per-device LRU over `capacity_bytes -
pinned_bytes` — reproduces the measured workload closely (18,564 modelled weight
misses against 16,572 measured; scaled modelled serial time 10.97 s against
10.954 s of measured demand wait) and gives the distribution that decides it:

| devices missed on, per layer-step | layer-steps | share |
|---|---:|---:|
| 0 | 2,086 | 38.2% |
| 1 | 2,118 | 38.8% |
| 2 | 1,001 | 18.3% |
| 3 | 256 | 4.7% |

**77% of decode layer-steps have nothing to overlap.** Only ~1.13 experts miss
per layer-step, and an expert's three matrices all live on one device, so the
typical burst is three copies down one link with the other two idle.

| projection (modelled, scaled to measured) | seconds |
|---|---:|
| serial, current placement | 10.95 |
| perfectly overlapped, current placement | 8.20 |
| serial, arena bound to node 1 | 10.25 |
| perfectly overlapped, arena bound to node 1 | 7.74 |

Against a 40.35 s decode: overlap alone is **1.07x**, NUMA binding alone is
**1.02x**, and both together are **1.09x** — all at a ceiling of zero
implementation overhead.

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

`argmax_r` is the H2D link at 86.25 ms/step. Perfectly parallelizing it across
the three links removes 21 of those 86 ms. That is the whole prize.

## Decision

**Do not build the copy stream.** The named mechanism is worth zero, the claimed
5.9x does not exist, and the one real mechanism has a measured ceiling of 1.33x
on a term worth 27% of the step — 1.07x end-to-end before paying for lease
lifetime and `moe_in_flight` ordering against a cache that currently guarantees
a weight is resident before it is read.

Recording the regime that would have been required, per the charter's rule
against manufacturing a favourable one: this mechanism needs miss bursts that
span devices. It would pay if the per-layer-step miss count were several times
higher — a smaller VRAM cache, a larger expert set, or a longer context — or if
one expert's three matrices were split across links, which trades a guaranteed
3x rise in per-expert issue count for a 1.33x ceiling. Neither is the operating
point that was asked about.

## Next term

**Bind the resident arena's pages to the NUMA node hosting the majority of the
GPUs.** It targets `B_r` of the bottleneck resource rather than its overlap, it
is a placement call at `stage()` rather than a change to the read path, and it
removes a currently unmanaged input to the step's largest term. Projected 1.02x
here, which is thin on its own; the stronger argument is determinism, since the
term's rate currently depends on which staging worker touched which tensor.
It needs its own branch, hypothesis and gate — including the question of what
`interleave` costs the majority node, and whether an unbalanced expert-to-device
schedule should follow the arena rather than device VRAM.

Beyond that: 581 MB/step moves in 43 bursts of ~13.5 MB, each serial with the
compute that consumes it, so the links sit idle most of the step. Aggregate
capacity gives those bytes a ~24 ms/step floor against 86 ms measured, and the
gap is idle link time, not bandwidth. Closing it means taking the transfer off
the critical path rather than widening it — prefetch, whose simulation gate
experiment 0022 already cleared (7.61% fewer modelled bytes, 84.66% useful) and
whose authorized opt-in runtime experiment has never been run. Per the charter
its predictions stay advisory.

## What the probe defect cost

Experiment 0024 recorded that `strata-topology-probe` reported pinned and
pageable H2D as identical because it timed a warm reused buffer. The same probe
also measured one device at a time, on an unbound source buffer, and never
concurrently — so the two properties that turned out to matter, serial use of
three links and the NUMA placement of the source, were both invisible in it.
The probe is corrected in this branch: it samples cold, randomly placed slices
of a large registered mapping and reports serial, batched, round-robin and
overlapped arms across all devices. Evidence:
`results/deepseek-v4-cold-slice-probe.json`.
