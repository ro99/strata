# Experiment 0126 — the tier's ordering, and what it makes the 5060 Ti worth

Status: **the cap on the routed-expert tier is an ordering defect, measured
and repaired in the tree. The repair is not yet gated end to end — that needs
the served model stopped.** The handover's question was whether the idle RTX
5060 Ti is worth cross-device choreography for about +1 tok/s. It is not. With
the ordering fixed it is worth about +2.8 tok/s, and the fix is worth more than
the card.

## The question handed over

Experiment 0124 validated the placement — decode routing is 2.03x concentrated,
the concentration belongs to the model rather than the conversation, and a
14.4 GB set chosen from one prompt covers 38.6% of another's decode activations
against a 10.4% null — and then built a dispatch that did not work. After the
repair the tier is exact on the rank devices, and its measured marginal return
decays sharply:

| tier (total) | MoE ms/tok | decode tok/s | marginal |
| ---: | ---: | ---: | ---: |
| 0 GB | 72.81 | 8.625 | — |
| 4 GB | 66.30 | 8.955 | 1.63 ms/GB |
| 10 GB | 60.70 | 9.228 | **0.93 ms/GB** |

Extrapolated, the 5060 Ti's ~14.4 GB is worth about +1 tok/s. The handover was
right to say that does not justify cross-device event choreography, and right
to ask for the decision before building.

**These three rows are inherited, not re-measured here.** They were taken at
the production operating point on the rank devices; everything below that is
labelled measured was measured today.

## The decay is not a property of the placement

Under the cost model the tier should turn a sum into a `max`: the host pools
read DRAM while the device reads VRAM, concurrently. Checking that against the
inherited row rather than against the previous row is what opens this:

- At 10 GB the plan covers about 31% of decode activations (interpolated from
  0124's held-out coverage table), so the host share should fall to
  72.81 x 0.69 = **50.2 ms**.
- The device share is 0.31 x 258 expert-calls = 80 calls at the 3090's measured
  0.128 ms, split across two devices, so **6.0 ms** wall.
- Overlapped, the term should be `max(50.2, 6.0) = 50.2 ms`. Serial, it should
  be `50.2 + 6.0 = 56.2 ms`.
- **Measured: 60.70 ms.** That is the serial prediction, not the overlapped one.

The code says why. `kernels/cuda/backend.cu` enqueues the tier kernels
*behind* the `cudaLaunchHostFunc` callback, and that callback runs the host MoE
inline (`state.cpu->run`, `dsv4_rank_local_layer_executor.cu`). Host functions
are stream-ordered, so the tier cannot begin until the host share has finished.
The two are serial by construction, and the marginal decay is that sum being
paid once per GB.

This is the charter's "separate volume from overlap" a second time in the same
mechanism. 0124 caught the first instance — a blocking worker handoff — and
replaced it with a stream-ordered dispatch that removed the blocking but kept
the serialization.

## The mechanism, priced before the system

`strata-dsv4-tier-overlap-probe` runs the two orderings with a spin kernel
calibrated to the tier's real per-layer duration and no model stage. The open
question was not whether CUDA overlaps streams; it was whether a host function
occupying stream A blocks kernels on stream B, since 0124 found that *blocking*
a driver callback thread stalled the whole context. A 1,040 us host share and a
330 us tier share are the measured production per-layer figures at 38.6%
coverage.

Rank device 1 (RTX 3090), tier device 0 (RTX 5060 Ti), 60–100 iterations,
median:

| host us | tier us | serial | overlap |
| ---: | ---: | ---: | ---: |
| 1040 | 330 | 1.52x of max | **1.00x** |
| 900 | 600 | 2.04x of max | **0.98x** |
| 700 | 900 | 2.72x of max | **1.02x** |
| 1040 | 100 | 1.13x of max | **1.00x** |

Same-device (both on device 0) at 1040/330: serial 1.30x, overlap 1.00x, with
0.7 us of overhead. Cross-device costs about 1 us per layer, needs no P2P, and
lands on the achievable `max` in every arm.

The serial column is also the shape 0124 misread. It attributed the tier's
poor showing to the overlap *window* shrinking as the tier covered more — "the
more the tier succeeds at removing DRAM bytes, the less time it has to hide its
own latency behind". The real relation is the inverse and it is an ordering
cost: 1.13x at 100 us of tier work, 2.72x at 900 us. The better the tier
covers, the more the ordering takes back.

## The other sign 0124 got wrong has since flipped

0124 concluded the placement could not pay because the engine was slower than
the one it replaced: the batch-1 FP4 expert GEMV measured 0.334 ms per expert
on the 5060 Ti against the host pools' 0.282 ms, so "the CPU is currently
faster per expert than the GPU is". Commit `ca87539` made the E2M1 decode
branch-free. Re-measured today with `strata-dsv4-static-tier-probe` on device 0:

| routed experts per call | ms | per expert |
| ---: | ---: | ---: |
| 1 | 0.1608 | 0.161 |
| 2 | 0.2858 | **0.143** |
| 6 | 1.0735 | 0.179 |

**0.143 ms against the host's 0.282 ms.** The card is now 1.97x faster per
expert than the CPU pools it displaces, where 0124 measured it 1.18x slower.
0.143 ms for a 13.37 MB triplet is 93.5 GB/s, still only 21% of the card's
448 GB/s, so the kernel remains far from its own ceiling — but it is no longer
the term that decides the design.

The two-expert rate is the one to use: at 38.6% coverage the tier serves 2.32
experts per layer, not six.

## What the 5060 Ti is worth, and the decision

43 layers x top-6 = 258 expert-calls per token. Host 0.282 ms each (72.81/258),
5060 Ti 0.143 ms each (measured), coverage 38.6% (0124, held out). Non-MoE is
43.09 ms of the 115.9 ms step.

| dispatch | host share | device 0 | MoE term | step | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| today, no tier | 72.81 | — | 72.81 | 115.9 | 8.625 |
| 5060 Ti, serial ordering | 44.7 | 16.4 | 61.1 | 104.2 | 9.60 |
| **5060 Ti, overlapped** | 44.7 | 16.4 | **44.7** | 87.8 | **11.39** |

The serial row reproduces the handover's +1 tok/s almost exactly. That estimate
was not wrong; it was a correct reading of a defective ordering.

So the decision the handover asked for: **the cross-device choreography is not
worth building for the tier as dispatched, and is worth building once the
ordering is fixed.** The fix is the prerequisite, it is ~50 lines of stream
plumbing, and it pays on the rank devices too — 10 GB overlapped projects
50.2 ms against the 60.70 ms measured, or 9.228 -> 10.72 tok/s with no new
hardware involved.

Both tiers together, 10 GB on the ranks and 14.4 GB on the 5060 Ti, is about
48.6% coverage: host 37.4 ms, device 0 14.2 ms, rank devices 1.7 ms, all
concurrent, for a 80.5 ms step and **12.42 tok/s — 1.44x**. That is a
projection from measured per-expert costs and held-out coverage, not a result.

**The ceiling is unchanged and still binds.** Non-MoE is 43.09 ms and remains
unattributed; every `cudaEventRecord` in the rank-local executor is gated on
`!chain_mode`, so production decode records nothing (0125 records this as an
open defect). Even a free MoE leaves 43.09 ms, or 23.2 tok/s. The 5060 Ti
cannot reach 15 tok/s alone and this does not change that.

## What landed

- `apps/strata_dsv4_tier_overlap_probe.cu` — the two orderings, priced in
  seconds against a calibrated spin kernel, on any device pair.
- The split itself. `CudaDsv4DeviceInputHostMoeRouteCallback` is an optional
  first host node that decides the route and publishes the tier selection;
  `tier_route_ready` is recorded between the halves and releases a tier stream;
  `tier_finished` rejoins it before the join.
- The tier accumulates into its own partials, because with the two concurrent
  the rank-partial upload would be writing the buffer the tier's `atomicAdd`
  is writing. The join sums it into the first rank's term — where the serial
  ordering accumulated it — before the same rounding.
- One router. The host share consumes the route the first node left in the
  rank's scratch rather than deriving it a second time.
- The split is taken only when a tier is active, and the route callback is
  optional throughout, so a no-tier build keeps the original ordering. Passing
  null is the rollback.

## Gates

`make check`: 2/3. `strata-tests` passes (211 s), layering and symbol checks
pass. `strata-equivalence-gemma4` fails on VRAM admission — 11.12 GiB needed
against 1.36 GiB free on device 1 — because the served endpoint holds 21 GB of
both 3090s. It is not reached by this change.

## Not done, and it is the next thing

**The tier's exactness under the split is not gated.** The reordering is
argued from the code and from the kernels' own declared reassociation, not
demonstrated: no run with `--static-expert-plan` has confirmed token IDs are
unchanged. That needs the served model stopped, which needs the operator.
Until then this is an implementation with a projection, not a result.

The end-to-end arm to run, in this order:

1. `--static-expert-plan` on the rank devices, tier 10 GB, token IDs against
   the no-tier arm. Exactness first.
2. The same arm's decode: 60.70 ms is the number to beat, 50.2 ms is the
   prediction. If it does not move, the split is not doing what this probe
   says it does and the 5060 Ti work stops here.
3. `scripts/dsv4_server_prefill_bench.sh` at ~2,300 words, because the rank
   tier takes its VRAM from the prefill expert cache.

## Risks

- **The host share may not shrink linearly.** Every projection above holds the
  pools at 47.4 GB/s aggregate while removing up to half their work. Fewer
  experts per layer may mean less bandwidth utilization, in which case the host
  share falls more slowly than 1:1 with coverage and every row overstates.
  Stage 2 above measures it.
- The tier's cross-slot sum is reassociated once more than before. The down
  kernel already declared its slot order unfixed — it accumulates by
  `atomicAdd` across independent blocks — so this is the same reassociation
  class rather than a new one, but it is a change and it is recorded here.
- Coverage is 0124's, from two prompts on one model on one machine. Held-out
  across prompts, not across traffic.
- **The handover's prefill constraint is stale.** It records the tier's VRAM
  costing the prefill expert cache 8.92 -> 3.64 GB and prefill 7.27 -> 1.64
  tok/s. Experiment 0125 falsified that attribution on three independent
  grounds — the flag was never in the served config, the code was not in the
  binary, and the reservation is inert without the flag. The 5060 Ti's "free
  capacity" argument is still true, but the cost on the other side of it has
  never been demonstrated, so it should not be cited as measured.
