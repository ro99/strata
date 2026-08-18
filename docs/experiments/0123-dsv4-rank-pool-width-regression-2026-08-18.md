# Experiment 0123 — the rank-local pool width regression, bisected and fixed

Status: **fixed.** Decode 3.066 -> 8.780 tok/s on the server, **2.86x**, with
identical generated tokens. `main` had been carrying a 2.6x decode regression
since 2026-08-17.

## How it was found

The operator's daily figures for

```
strata-server --model-type deepseek --devices 1,2 --context-size 16384
  --vram-fraction 0.95 --decode-topology rank-local-tp2 --prefill-page-tokens 8192
```

were 6.30 tok/s decode. Measuring the same launch on `main` gave **3.066 tok/s**,
and the standalone runner independently gave 3.0-3.3 across a 37-token and a
2,430-token prompt. Two binaries agreed with each other and disagreed with the
operator by about 2x, which is a defect, not a datapoint.

Their `build-nccl/strata-server` (Release + NCCL, built 2026-08-16) measured
**8.203 / 8.390 / 8.469 tok/s** on the identical launch. So `main` was slower
than a binary from two days earlier, and the operator's own 6.30 was itself
below what that binary could do.

## Bisect

`git bisect` between `04d7932` (the pre-merge tip of `main`, same day as the
operator's build) and `fb53e55` (`main`), probing with one decode arm per
commit and a 6.0 tok/s split:

| commit | decode tok/s | verdict |
| --- | ---: | --- |
| `04d7932` pre-merge `main` | 8.403 | good |
| `ec2d0c8` | 8.601 | good |
| `af14370` | 8.491 | good |
| `079b82a` | 8.524 | good |
| **`6fd2637`** | **3.398** | **first bad** |
| `599269f` | 3.193 | bad |
| `fb53e55` `main` | 3.19 | bad |

`6fd2637` is *feat: Phase 7 -- one hardware profile replaces the box compiled
into the source*, on the `infra/00-equivalence-oracle` branch merged as
`7269e0b`.

## Cause

Phase 7 replaced nine hardcoded machine constants with a probed profile. One of
them was the rank-local CPU pool width. Its commit message records the intent:

> `kDsv4RankLocalMinimumCpusPerRank` was doing two jobs -- the floor below which
> a rank cannot work, and the pool width to assign -- and 24 was only correct
> for both on one box. [...] The width now comes from the smallest online node
> (24 there, unchanged) and the constant is only the floor.

**"24 there" was wrong.** This box's NUMA nodes hold **28 logical CPUs each**
(`node0: 0-13,28-41`, `node1: 14-27,42-55`) over **14 physical cores**. So the
width did not stay at 24; it silently widened to 28, and both rank pools
together took **56 workers on 56 logical CPUs** — leaving no runnable headroom
for the main thread, the CUDA driver's threads, NCCL, or the HTTP server.

The failure is a cliff rather than a slope because these pools are **barrier
synchronized**. `Dsv4HostMoeExecutor::run` forks and joins three times per
layer — gate_up, down, reduce — across 43 layers per token per rank, so 129
barriers per token per rank. One preempted worker stalls its entire cohort at
every one of them.

Measured decode against pool width, everything else identical:

| pool width per rank | decode tok/s | MoE ms/token |
| ---: | ---: | ---: |
| 28 logical (Phase 7) | 3.19 | 222.3 |
| 26 | 7.66 | 78.3 |
| 24 (pre-Phase 7) | 8.43 | 73.1 |
| **14 (one per physical core)** | **8.49** | **72.7** |

Two things fall out of that sweep. The regression is entirely the last two
CPUs of headroom — 26 recovers most of it and 28 falls off a cliff. And the
kernel **saturates at one worker per physical core**: 14 matches 24 within
noise and edges ahead of it, so the 14 SMT siblings were buying nothing and
paying for the cliff.

## Fix

The width is now one worker per physical core on the smallest node, and the
assignment picks those cores rather than the first N logical CPUs.

- `NumaTopology` gains `node_primary_cpus` — per node, the lowest-numbered
  logical CPU of each physical core, read from
  `/sys/devices/system/cpu/cpuN/topology/thread_siblings_list` — and
  `smallest_node_cores()`. Both are left empty and fall back to the logical
  lists when sysfs does not expose siblings, so a machine that cannot report
  topology gets the old behaviour rather than a silently narrower pool.
- `admit_dsv4_rank_local` takes its width from `smallest_node_cores()`.
- `plan_dsv4_rank_local_cpus` assigns each rank its node's primaries, so a pool
  of N never lands two workers on one core's siblings while another core idles.

`kDsv4RankLocalMinimumCpusPerRank` remains the floor only, which is what Phase 7
correctly separated it into.

## Result

Server, the operator's exact launch, three requests, 27-token prompt, 64 tokens:

| binary | decode tok/s | vs `main` |
| --- | ---: | ---: |
| `main` (`fb53e55`) | 3.066 | 1.00x |
| operator's `build-nccl`, 2026-08-16 | 8.203 / 8.390 / 8.469 | 2.72x |
| **this fix** | **8.651 / 8.780 / 8.787** | **2.86x** |

Median **8.780 tok/s**, which also beats the 2026-08-16 binary by 1.04x and the
operator's stated 6.30 tok/s baseline by 1.39x. The standalone runner agrees at
8.625 tok/s, and `moe_seconds` falls 222.3 -> 72.8 ms/token.

Generated token IDs are identical to every earlier arm
(`3054, 2572, 1192, 260, 44213, ...`): this changes how many threads run the
same arithmetic, not the arithmetic. `ctest` 3/3.

## What this says about the standing MoE numbers

Every MoE figure measured on `main` since 2026-08-17 was taken on the regressed
pool and overstates the term. The corrected decode step is 115.9 ms/token with
**72.8 ms of it MoE (63%)**, not 222 ms of 313. MoE is still `argmax_r` and
still the lever, but it is 3x smaller than the pre-fix attribution implied, and
the 15.3 GB/s figure that motivated experiment 0122's hugepage hypothesis was an
artifact of this regression — the corrected rate is 3.449 GB / 72.8 ms =
**47.4 GB/s aggregate**, against 0058's 56.7-60.9 GB/s bound. There is far less
headroom left in the host kernel than 0122 assumed.

## Reproduce

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_NCCL=ON \
  -DSTRATA_NCCL_INCLUDE_DIR=<nccl>/include -DSTRATA_NCCL_LIBRARY=<nccl>/lib/libnccl.so.2
cmake --build build-release --target strata-server -j
./build-release/strata-server --model models/dsv4f --model-type deepseek \
  --model-id strata-deepseek-v4 --devices 1,2 --context-size 16384 \
  --vram-fraction 0.95 --decode-topology rank-local-tp2 \
  --prefill-page-tokens 8192 --host 127.0.0.1 --port 8033 &
bash scripts/dsv4_server_decode_bench.sh 8033 label 3
```
