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

## Addendum — the corrected decode budget, and what 20 tok/s needs

With the fix in place the step is **115.9 ms/token (8.625 tok/s)**:

| term | ms/token | share |
| --- | ---: | ---: |
| `moe_seconds` (host routed experts) | 72.81 | 62.8% |
| non-MoE inside the layer loop | 41.54 | 35.8% |
| KV / output head / candidate / outside | 7.42 | 6.4% |

Every other decode counter reads exactly zero, and the cause is
`kernels/cuda/dsv4_rank_local_layer_executor.cu:1436`:

```cpp
if (output.success && !impl_->chain_mode) {   // event timing, sequential only
```

Production decode runs chain mode, so the CUDA event breakdown -- attention,
attention collective, publication, router, shared, MoE collective, final
transition -- is never computed. The 41.54 ms is therefore unattributed by
construction, not by omission. `cpu_moe_phases` *is* populated (line 550), in
`kernels/`, which an earlier grep over `src/ apps/ include/` missed.

One suspect priced without instrumenting anything. Rank-local decode issues 86
NCCL all-reduces per token (43 attention, 43 MoE) between two 3090s with no
working P2P, each 4,096 floats. Measured standalone on this box, devices 1 and
2, 500 iterations x 3:

```
all-reduce 4096 floats (16 KB): 0.0158 ms each  ->  86/token = 1.36 ms/token
```

**Collectives are 1.36 ms of the 41.54, or 3.3%.** They are not the term. What
remains is ~40 ms of per-layer host submission and device work, 0.97 ms per
layer across 43 layers.

### The arithmetic to 50 ms/token

- The host MoE kernel now runs at 3.449 GB / 72.81 ms = **47.4 GB/s aggregate**
  against 0058's 56.7-60.9 GB/s bound, so kernel tuning alone is worth at most
  1.28x: 72.81 -> ~57 ms. Not sufficient.
- Removing bytes from the CPU is the lever. Serving the VRAM-resident fraction
  of each layer's six experts on the GPU and leaving the CPU only the misses,
  at hit rates from the LRU replay in [[dsv4-expert-cache-lever]]:

| expert cache | decode hit rate | CPU bytes/token | MoE ms | step ms | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| none (today) | 0% | 3.449 GB | 72.8 | 115.9 | 8.6 |
| 17.7 GB (already admitted) | ~65% | 1.21 GB | 25.5 | 68.6 | 14.6 |
| ~47 GB (reclaimed + 5060 Ti) | ~85% | 0.52 GB | 10.9 | 54.0 | 18.5 |

- Even a **free** MoE leaves 41.54 + 7.42 = 48.96 ms, which is 20.4 tok/s. So
  the MoE term alone cannot reach the target with margin; the 41.54 ms must
  come down too.

**20 tok/s = MoE to ~11 ms and the non-MoE layer term to ~31 ms.** That is 6.6x
on the first and 1.34x on the second. Both are required; neither suffices alone.

The next measurement is therefore the 41.54 ms, and the cheapest way in is to
compute the existing event timings in chain mode rather than only in sequential
mode -- the events are already recorded, and the guard above is the only thing
between them and an attribution.
