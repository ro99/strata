# Experiment 0178 — the overlapped tier split is a regression, the serial tier is the win, and the tier bricked the server until it was fixed

Status: **the ordering "fix" of experiment 0126 is falsified end to end. It is
slower than the ordering it replaced and it is not exact. The serial tier it
replaced is worth 1.070x on short-prompt decode and 1.106x at long context,
costs 0.917x on prefill, and needed a weight-arena defect fixed before it was
safe to run at all.**

## What 0126 handed over, and what it turned out to be

0126 diagnosed the routed-expert tier's poor marginal return as an ordering
defect: the tier kernels were enqueued behind the `cudaLaunchHostFunc` that
runs the host MoE inline, so the two were serial by construction. It built the
split (`483d563`), measured a standalone probe at 1.00x of the achievable
`max`, and projected 9.228 -> 10.72 tok/s on the rank devices alone.

It also said, plainly, what had not been done: *"The tier's exactness under the
split is not gated... this is an implementation with a projection, not a
result."* This experiment runs that gate, and the two it also deferred.

## A defect found before any arm was valid

This box's login shell exports `CUDA_DEVICE_ORDER=FASTEST_FIRST`, and
`apps/strata_server.cpp:863` pins `PCI_BUS_ID` only when the variable is unset.
So `--devices 1,2` selected the **RTX 5060 Ti plus one 3090**, capped both
ranks' symmetric admission at the 16 GiB card (13.4 GiB per rank instead of
20.3), and left a 3090 at 4 MiB. That is verbatim the failure the comment above
that line warns about; the guard loses to an exported value. The tell is the
asymmetric `admitted budget` columns in the placement report, 22.14 GiB against
14.57 GiB. `scripts/dsv4_decode15_server.sh` now exports `PCI_BUS_ID` itself.

## Stage 1 and 2, on the pre-merge tree

Two RTX 3090s, rank-local TP2, 16k context, `--vram-fraction 0.95`,
`--prefill-page-tokens 8192`, greedy, `max_tokens=256`, 27-token prompt.

| arm | tier | ordering | ms/tok | tok/s | vs no tier | distinct outputs |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| A | none | — | 118.7 | 8.424 | — | 1 of 3 |
| B | 10 GB | overlapped split | 115.2 | 8.680 | 1.030x | **2 of 7** |
| C | 10 GB | serial (pre-`483d563`) | **106.1** | **9.425** | **1.119x** | 1 of 5 |
| E | 12 GB | serial | 106.4 | 9.398 | 1.115x | 1 of 5 |
| D | 14 GB | serial | — | — | — | **VRAM failure** |

**Stage 1, exactness: failed.** Seven greedy reps of arm B produced exactly two
completions, 4 and 3, one of which opens on a garbage token (`וניב`) before
recovering into fluent text. Arm A is 3/3 identical and arms C and E are 5/5,
so the harness is sound and the tier itself is deterministic — the
nondeterminism belongs to the split alone.

A bimodal distribution is the signature of a race with two outcomes, not of
float reassociation, which smears rather than clusters. 0126 argued exactness
from the code, on the grounds that the tier's `atomicAdd` was "the same
reassociation class" the down kernel already had. The data falsifies that: the
class that already existed is deterministic in practice and the new one is not.
`483d563` fixed one race — the rank-partial upload against the tier's
accumulation — and left at least one more.

**Stage 2, overlap: failed, and worse than not moving.** The split loses to the
serial ordering by 9.1 ms/tok, giving back 72% of the serial tier's 12.6 ms
gain. 0126's kill criterion was *"if it does not move, the split is not doing
what this probe says it does and the 5060 Ti work stops here."* It moved
backwards.

The `strata-dsv4-tier-overlap-probe` result is not wrong about CUDA. It is
wrong about this system: a spin kernel with no model stage does not reproduce
the production dispatch. That is the charter's "reproduce the production access
pattern, or the probe lies" applied to a mechanism probe.

**The tier saturates at about 10 GB.** Arm E holds 962 pairs against arm C's
802, 20% more residency, and measures 106.4 against 106.1. That matches 0126's
marginal decay (1.63 ms/GB at 4 GB, 0.93 at 10 GB) reaching zero.

## Re-measured after merging main

Main had advanced 57 commits, including `7ca3298`, which wires the register-fed
FP8 route into the DeepSeek shared expert by default. The pre-merge numbers
therefore describe a different binary and were re-taken.

| arm | tier | short decode | long-context decode | prefill (~2,000 words) |
| --- | ---: | ---: | ---: | ---: |
| G | none | 116.7 ms/tok, 8.571 tok/s | 7.302 tok/s | 18.147 tok/s |
| L | 10 GB | **108.9 ms/tok, 9.171 tok/s** | **8.077 tok/s** | 16.649 tok/s |
| | | **1.070x** | **1.106x** | **0.917x** |

Main's fast kernels are worth 118.7 -> 116.7 ms/tok on DeepSeek V4 by
themselves, about 1.7%. That is the size the campaign contract predicts:
*"DeepSeek V4 remains negative because host MoE is `argmax_r` there."* The two
lines of work are complementary — the kernels make the one per-token CUDA
dispatch faster, and the tier removes host-DRAM bytes from the term that
dominates.

**Stage 3, the prefill cost, is now measured rather than assumed: 0.917x.**
0126 recorded the handover's prefill constraint as "stale" because 0125 had
falsified its attribution. The attribution was indeed wrong; the cost is real.
It is the tier and the new reserve below taking their VRAM from the prefill
cache.

## The tier bricked the server, and why

Before the fix below, a tier of any size killed the server after a handful of
requests. Every later request returned a sticky `DeepSeek device mHC slot
reservation is out of order`, so the first failure was the real one:

```
layers.19.ffn.shared_experts.w2: CUDA weight arena is exhausted
(device 1, wanted 8.0 MiB, free 20.0 MiB of 20504.2 MiB in 3 blocks,
 largest 6.8 MiB)
```

20 MiB free of 20.5 GiB: **exhaustion, not fragmentation.** The occupancy
report in that message is new; the old one could not tell the two apart, and
they need opposite fixes.

The rank-local sizing set the routed cache to
`min(arena_expert_bytes, cache_expert_bytes)`, where `arena_expert_bytes` was
every byte left after the store, the pinned bytes and the tier. **Without a
tier, `cache_expert_bytes` is the smaller term**, so the cache stops early and
the leftover slack silently absorbs everything else the arena must hold. A tier
makes the arena term binding, the slack goes to zero, and the next
shared-expert re-stage has nowhere to go.

The shared expert is acquired from that same cache once per layer on every
forward pass, so its whole 43-layer set must stay stageable no matter how many
routed experts have been admitted. It is now reserved out of what the routed
cache may claim: **1.016 GiB per rank**, derived from the declared shapes.

Cliff before the fix, by tier size: 14 GB failed on the first request, 10 GB on
request 6, 6 GB on request 5. A smaller tier did not fix it, it moved the
cliff — which is what identified this as an accounting defect rather than a
budget one. After the fix, 12 consecutive requests mixing 27-token and
~2,000-word prompts complete with zero errors.

This also explains arm D. Its `atomic in-flight expert set exceeds a device
VRAM budget` at 7.0 GiB per rank is the same zero-slack arena, reported by a
different consumer.

## What landed

- The device-order repair in `scripts/dsv4_decode15_server.sh`.
- `[deepseek-tier]` build logging. The tier build was silent, so a partial
  build and a full one looked identical and "the gain is small because the tier
  is small" could not be ruled out from a log.
- **`--static-expert-bytes` is per rank, not total.** It truncates to
  `vram_budget_bytes / kTripletBytes` *after* slicing the ranking by rank, so
  0124's and 0126's "10 GB (total)" row is `5G`. Passing `10G` builds twice the
  intended residency.
- The shared-expert arena reserve, and arena occupancy on exhaustion.
- **The serial ordering is the default.** `483d563`'s split is opt-in behind
  `STRATA_DSV4_TIER_OVERLAP=1`, kept only so the defect can be worked.

## The 5060 Ti stays blocked

`static_expert_tiers` is `std::array<Dsv4StaticExpertTier*, 2U>`, one slot per
rank, and the backend enqueues the MoE only on rank devices, so nothing would
dispatch a third device's kernels. The card needs a third slot *and* genuine
cross-device dispatch — the choreography 0126 gated behind a working overlap.
That gate reads negative, so building it would be stacking stage N+1 on a
falsified stage N.

## Gates

`make check` 3/3. Generated tokens are unchanged across the merge and the
fix: the no-tier arm reproduces its pre-merge hash and the tier arm reproduces
its own, exactly.

## Risks and limits

- One prompt pair, one machine, one model. The tier's coverage is 0124's, held
  out across prompts but not across traffic.
- The tier is **opt-in and should stay that way** until the prefill cost is
  acceptable for the deployment: it is a decode-for-prefill trade, not a free
  win.
- 43.09 ms of non-MoE is still unattributed, and 0125's open defect — every
  `cudaEventRecord` in the rank-local executor gated on `!chain_mode`, so
  production decode records nothing — is why. At 9.17 tok/s the MoE term is
  still the larger one, so that remains the place to work.
