# Experiment 0178 — the overlapped tier split is a regression, and the serial tier is the win

Status: **the ordering "fix" of experiment 0126 is falsified end to end. It is
slower than the ordering it replaced and it is not exact. The serial tier it
replaced is worth 1.119x on decode, deterministic, and is now the default.**

## What 0126 handed over, and what it turned out to be

0126 diagnosed the routed-expert tier's poor marginal return as an ordering
defect: the tier kernels were enqueued behind the `cudaLaunchHostFunc` that
runs the host MoE inline, so the two were serial by construction. It built the
split (`483d563`), measured a standalone probe at 1.00x of the achievable
`max`, and projected 9.228 -> 10.72 tok/s on the rank devices alone.

It also said, plainly, what had not been done: *"The tier's exactness under the
split is not gated... this is an implementation with a projection, not a
result."* This experiment runs that gate. Both stages fail.

## The operating point, and a defect found before any of it

Two RTX 3090s, rank-local TP2, 16k context, `--vram-fraction 0.95`,
`--prefill-page-tokens 8192`. Greedy (`temperature: 0`), `max_tokens=256`,
27-token prompt. Medians of 3-7 reps.

**Before any arm was valid, the device binding had to be repaired.** This box's
login shell exports `CUDA_DEVICE_ORDER=FASTEST_FIRST`, and
`apps/strata_server.cpp:863` pins `PCI_BUS_ID` only when the variable is unset.
So `--devices 1,2` selected the **RTX 5060 Ti plus one 3090**, capped both
ranks' symmetric admission at the 16 GiB card (13.4 GiB per rank instead of
20.3), and left a 3090 at 4 MiB. That is verbatim the failure the comment above
that line warns about; the guard simply loses to an exported value. The tell is
in the placement report — asymmetric `admitted budget` columns, 22.14 GiB
against 14.57 GiB. `scripts/dsv4_decode15_server.sh` now exports `PCI_BUS_ID`
itself.

Every number below was taken after that repair, on the two 3090s.

## Both gates fail

| arm | tier | ordering | ms/tok | tok/s | vs no tier | distinct outputs |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| A | none | — | 118.7 | 8.424 | — | 1 of 3 |
| B | 10 GB | overlapped split | 115.2 | 8.680 | 1.030x | **2 of 7** |
| C | 10 GB | serial (pre-`483d563`) | **106.1** | **9.425** | **1.119x** | 1 of 5 |
| E | 12 GB | serial | 106.4 | 9.398 | 1.115x | 1 of 5 |
| D | 14 GB | serial | — | — | — | **admission failure** |

**Stage 1, exactness: failed.** Seven greedy reps of arm B produced exactly two
completions, 4 and 3, one of which opens on a garbage token (`וניב`) before
recovering into fluent text. Arm A is 3/3 identical and arms C and E are 5/5
identical, so the harness is sound and the tier itself is deterministic — the
nondeterminism belongs to the split alone.

A bimodal distribution is the signature of a race with two outcomes, not of
float reassociation, which smears rather than clusters. 0126 argued exactness
from the code, on the grounds that the tier's `atomicAdd` was "the same
reassociation class" the down kernel already had. The data falsifies that
argument: the class that already existed is deterministic in practice, and the
new one is not. `483d563` fixed one race — the rank-partial upload against the
tier's accumulation — and left at least one more.

**Stage 2, overlap: failed, and worse than not moving.** The split was supposed
to beat the serial ordering. It loses to it by 9.1 ms/tok, giving back 72% of
the serial tier's 12.6 ms gain. 0126's kill criterion was *"if it does not
move, the split is not doing what this probe says it does and the 5060 Ti work
stops here."* It moved backwards.

The `strata-dsv4-tier-overlap-probe` result — 1.00x of the achievable `max`,
cross-device and same-device — is not wrong about CUDA. It is wrong about this
system: a spin kernel with no model stage does not reproduce the production
dispatch, which is the charter's rule 2 on making measurements cheap
("reproduce the production access pattern, or the probe lies") applied to a
mechanism probe rather than to a bandwidth probe.

## What the serial tier is worth, and where it stops

The serial ordering hits its own prediction almost exactly. 0126 predicted
60.70 ms for the MoE term at a 10 GB tier; arm C measures a 106.1 ms step
against arm A's 118.7, which at 43.09 ms of non-MoE puts the MoE term at
63.0 ms — the same row, 2.4% off, matching the 2.4% by which this baseline runs
slower than the inherited one.

**The tier saturates at about 10 GB.** Arm E holds 962 pairs against arm C's
802, 20% more residency, and measures 106.4 against 106.1 — no gain. That is
consistent with 0126's marginal decay (1.63 ms/GB at 4 GB, 0.93 at 10 GB)
reaching zero, and it means the remaining headroom in this design is not in the
rank tier's size.

**Above 12 GB it does not merely stop paying, it breaks.** Arm D at 7.0 GiB per
rank boots, builds both tiers, and then fails every request with `DeepSeek
atomic in-flight expert set exceeds a device VRAM budget` (the mHC
out-of-order error after it is the sticky follow-on, not the cause). Admission
passed and the runtime allocation did not: `deepseek_runtime.cpp:6803`
subtracts the tier reservation from the prefill cache sizing but nothing
subtracts it from what the in-flight expert set checks. That is a separate
defect and it is recorded here rather than fixed.

## The tier size is now observable, which it was not

The tier build was silent. This experiment could compute 802 pairs from the
plan and the truncation math but could not observe them, so a partial build and
a full one looked identical, and "the gain is small because the tier is small"
could not be ruled out. `dsv4_static_expert_tier.cpp` now prints one line per
rank at build:

```
[deepseek-tier] device=1 rank=0/2 pairs=401 bytes=5361106944 (4.99 GiB)
[deepseek-tier] device=2 rank=1/2 pairs=401 bytes=5361106944 (4.99 GiB)
```

That confirmed every arm's residency before its timing was read.

**`--static-expert-bytes` is per rank, not total.**
`dsv4_static_expert_tier.cpp:42` truncates to `vram_budget_bytes / kTripletBytes`
*after* slicing the ranking by rank, so the "10 GB (total)" row of 0124 and
0126 is `5G`, not `10G`. Passing `10G` builds a 20 GB tier and compares it
against a 10 GB row.

## The 5060 Ti is still blocked, and not by a small thing

The branch's goal was the idle card. It cannot be reached from here:
`static_expert_tiers` is `std::array<Dsv4StaticExpertTier*, 2U>`, one slot per
rank, and the backend enqueues the MoE only on rank devices, so nothing would
dispatch a third device's kernels. The card needs a third slot *and* genuine
cross-device dispatch — the choreography 0126 gated behind a working overlap.

That gate now reads negative, so building the cross-device path on top of it
would be building stage N+1 on a falsified stage N. The card stays idle until
the split's race is found, or until the overlap is rebuilt from something other
than the probe that mispredicted it.

## What landed

- The device-order repair in `scripts/dsv4_decode15_server.sh`.
- `[deepseek-tier]` build logging.
- **The serial ordering is the default.** `483d563`'s split is now opt-in
  behind `STRATA_DSV4_TIER_OVERLAP=1`, kept only so the defect can be worked.
  Arm F re-runs arm C's configuration with no environment variable set, to
  confirm the default is the measured-good path.

## Gates

`make check` — recorded with the commit.

## Risks and limits

- One prompt, one machine, one model. The tier's coverage is 0124's, held out
  across prompts but not across traffic.
- 43.09 ms of non-MoE is still unattributed, and 0125's open defect — every
  `cudaEventRecord` in the rank-local executor gated on `!chain_mode`, so
  production decode records nothing — is why. Even a free MoE leaves 23.2 tok/s,
  and at 9.4 tok/s the MoE term is still the larger one, so this remains the
  place to work.
- Arm D's admission defect means the tier's VRAM ceiling is currently found by
  bisection at run time rather than reported at admission.
