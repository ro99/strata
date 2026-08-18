# Experiment 0122 — transparent hugepages on the DSV4 tiled expert arena

Status: **rejected.** 0.956x on decode over two interleaved repetitions,
with the mechanism verified as landed. The host routed-expert MoE is not
translation-bound.

## Where this came from

The production baseline the user runs daily is **6.30 tok/s decode** (158.7
ms/token) and 7.30 tok/s prefill, from `strata-server --model-type deepseek
--devices 1,2 --context-size 16384 --vram-fraction 0.95 --decode-topology
rank-local-tp2 --prefill-page-tokens 8192`. The objective is **20 tok/s decode**,
which is 50 ms/token, a 3.17x.

A decode-shaped arm on this branch (37-token prompt, 64 generated tokens,
devices 1,2, rank-local TP2) attributed the step for the first time at this
operating point:

| term | ms/token | share of decode |
| --- | ---: | ---: |
| `rank_local_layer_seconds` | 311.11 | 99.3% |
| **`moe_seconds`** | **225.52** | **72.0%** |
| `rank_local_kv_seconds` | 4.90 | 1.6% |
| `output_head_seconds` | 2.82 | 0.9% |
| everything else measured | < 1 | < 1% |

`argmax_r` is the routed-expert MoE. In rank-local TP2 that term is **entirely
host**: the rank-local weight store holds attention, router, shared-expert and
mHC weights only, and routed experts are read from the tiled NUMA arena by
`Dsv4HostMoeExecutor::run` on each rank's own NUMA node. No VRAM expert cache
participates in rank-local decode at all.

At 3.449 GB of routed-expert weight per token, 225.52 ms is **15.3 GB/s
aggregate**. The relevant comparisons:

| figure | GB/s | source |
| --- | ---: | --- |
| DDR4 peak, 2 of 4 channels per socket | 76.3 | machine |
| host FP4 kernel, `MPOL_BIND` per node | 56.7–60.9 | 0058 |
| host FP4 kernel, standalone, 28 threads | 26.23 | 0051 |
| host FP4 kernel, in situ over the full arena | 11.7–13.1 | 0058 |
| **this arm, in situ** | **15.3** | here |

0058 attributed the standalone-to-in-situ gap to "cold reads over the cold
148 GB arena" but did not identify a mechanism.

## The mechanism

`Dsv4ResidentWeightStore::stage` allocates the tiled expert arena with a plain
anonymous `mmap` and binds each shard to its NUMA node. It never calls
`madvise`. This machine's THP policy is `[madvise]`, so a region that does not
ask gets no hugepages — and measurement confirms it: during a baseline arm the
process reports `AnonHugePages: 0 kB` against 38 GB of RSS, and system-wide
`AnonHugePages` is `0 kB`.

The arena is ~148 GB. On 4 KiB pages that is 36.2M pages, and one 13.37 MB
expert triplet spans 3,264 of them. The E5-2680 v4 (Broadwell) L2 STLB holds
1,536 entries. Decode reads six randomly-selected experts per layer across 43
layers — 258 expert triplets per token — so essentially every expert read is a
page walk, and the arena is far too large for the walk to hit cached page-table
levels.

At 2 MiB the same arena is ~74K pages and one expert triplet is 7 pages, a
~466x reduction in translation work per token.

This predicts precisely the observed shape: a kernel that reaches 26.23 GB/s
standalone over a small resident set, and 11.7–15.3 GB/s in situ over 148 GB.

`src/models/glm52/checkpoint.cpp` already calls `madvise` on its mapping. The
DSV4 arena is the one that does not.

## Change

One `madvise(MADV_HUGEPAGE)` on the arena, issued after `mmap` and **before**
first touch so the fault path can hand back 2 MiB pages rather than promoting
4 KiB ones afterwards, and before the existing `numa_bind_range` calls so the
binding is unchanged.

It is a parameter (`Dsv4RuntimeConfig::hugepage_expert_arena`, CLI
`--arena-hugepages` / `--no-arena-hugepages`, default on) rather than an
unconditional call, so one binary measures both arms. `MADV_HUGEPAGE` is
advisory: the kernel may refuse, and a refusal is recorded, never an error.

## Predeclared contract

- **Hypothesis.** Backing the tiled expert arena with transparent hugepages
  reduces `moe_seconds` per token at the measured operating point, because the
  term is host DRAM reads over a 148 GB region whose translation cost, not
  whose bandwidth, is binding.
- **Primary metric.** Median decode tok/s over interleaved repetitions, every
  run reported.
- **Diagnostic metric.** `moe_seconds` per token, and the implied aggregate
  GB/s against 3.449 GB/token. Process `AnonHugePages` must be large in the
  candidate and zero in the baseline, or the arm did not test the mechanism.
- **Resource signs.** Reduces host TLB/page-walk work. Adds nothing to device
  compute, PCIe, VRAM or H2D volume. The adverse sign is staging time: with
  `defrag=[madvise]` a madvised region may take synchronous compaction on
  fault, so `resident_staging_seconds` is reported for both arms. Host RSS
  should be unchanged to within hugepage rounding.
- **Correctness.** Identical generated token IDs across arms — the change moves
  no bytes and alters no arithmetic, so anything else is a defect, not a
  tolerance. `make check` before the result commit.
- **Memory ceiling.** Unchanged; no new allocation. Arena bytes identical.
- **Rollback.** Any token-ID difference, any decode regression outside the
  measured spread, or a staging-time increase large enough to matter to a
  server that stages once per process, reverts to `--no-arena-hugepages` as
  the default.

## Design

One binary, `--vram-fraction 0.95`, `--max-context 16384`,
`--prefill-page-tokens 8192`, devices 1,2, rank-local TP2, 37-token prompt, 64
generated tokens. Order is baseline/candidate, repeated twice. A short prompt
is deliberate: rank-local decode reads routed experts from the host arena
regardless of prompt length or VRAM cache warmth, so this operating point
isolates the translation cost while keeping each arm near four minutes.

Expected wall time is roughly 4 minutes per arm and about 16 minutes total.

## Results

One binary, `--arena-hugepages` / `--no-arena-hugepages`, two interleaved
repetitions, 37-token prompt, 64 generated tokens, devices 1,2, rank-local TP2,
`--vram-fraction 0.95`.

| arm | decode tok/s | ms/token | MoE ms/token | staging s | prefill s |
|---|---:|---:|---:|---:|---:|
| no-arena-hugepages | 3.272 | 305.6 | 222.3 | 76.3 | 12.2 |
| no-arena-hugepages | 3.228 | 309.8 | 228.3 | 78.0 | 14.5 |
| arena-hugepages | 3.055 | 327.3 | 241.3 | 95.9 | 10.0 |
| arena-hugepages | 3.157 | 316.7 | 232.4 | 88.4 | 20.1 |
| **no-arena-hugepages median** | **3.250** | **307.7** | **225.3** | | |
| **arena-hugepages median** | **3.106** | **322.0** | **236.9** | | |

Decode **0.956x**. The MoE term is 0.951x, i.e. slower. Both repetitions agree
in direction and the arms do not overlap on the MoE term.

The mechanism landed and was verified, so this is a rejection of the hypothesis
and not of the implementation:

- baseline process: `AnonHugePages: 0 kB` against 154 GB RSS, and system-wide
  `AnonHugePages: 0 kB`;
- candidate process: `AnonHugePages: 106,844,160 kB` (106.8 GB) against
  154.6 GB RSS, so about 69% of the resident set moved to 2 MiB pages.

The predicted adverse sign appeared: staging rose from 76.3/78.0 s to 95.9/88.4
s, +12 to +20 s of synchronous compaction under `defrag=[madvise]`.

Generated token IDs are identical across all four arms, as required.

`ctest` against the Release+NCCL build passes 3/3 after the change
(`strata-tests` 259.46 s, `strata-sim-smoke`, `strata-equivalence-gemma4`).

## Verdict

**Rejected.** The default is set to off; the selector is retained so the
rejection stays reproducible from one binary.

The host routed-expert MoE is not bound by address translation. A 466x
reduction in page-table work made it slightly slower, which also rules out
TLB pressure as the explanation 0058 was reaching for when it attributed the
standalone-to-in-situ gap to "cold reads over the cold 148 GB arena". That gap
is still unexplained and is still worth explaining: 0051 measured 26.23 GB/s
standalone and this operating point measures 7.4-7.8 GB/s per rank.

## What this experiment found on the way, which matters more than its result

**1. `cpu_moe_phases` is dead instrumentation.** It is declared in three places
in `dsv4_rank_local_layer_executor.hpp` and populated nowhere in the tree. The
chain path's accumulation therefore always read zero, and `moe_nanoseconds`
silently came from the sequential per-layer accumulation instead -- which
includes `moe_collective_ms`. So the 222-233 ms/token MoE term reported here is
CPU expert arithmetic **plus** the per-layer collective, with no way to separate
them. Splitting that term is the prerequisite for any further MoE work.

**2. `strata-server` did not pin CUDA device order.** `strata_deepseek_run.cpp`
line 1110 sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`; `strata_server.cpp` did not. On
this box the orderings differ:

```
default:     cuda 0,1 = RTX 3090 (sm_86), cuda 2 = RTX 5060 Ti (sm_120)
PCI_BUS_ID:  cuda 0 = RTX 5060 Ti (sm_120), cuda 1,2 = RTX 3090 (sm_86)
```

so `--devices 1,2` selected **one 3090 paired with the 5060 Ti** under the
server and the two 3090s under the runner. Consequences: an sm_120 card in a
rank-local pair whose mHC contract requires sm_86 (the server now fails with
`exact DeepSeek device mHC requires an SM86 device` on the current build),
symmetric VRAM admission capped by the 16 GiB card rather than 24 GiB, and one
3090 left idle. Fixed by setting the same variable in the server, only when the
operator has not chosen an order.

**3. An unexplained 2x against the operator's baseline.** The user's daily
figures for this launch are 6.30 tok/s decode and 7.30 tok/s prefill. On this
branch, with the device order corrected, the same launch measures **3.066 tok/s
decode and 3.07 tok/s prefill** (server, 27-token prompt, 64 tokens, cold
cache), and the standalone runner independently measures 3.0-3.3 tok/s across
37-token and 2,430-token prompts. Two binaries agree with each other and
disagree with the operator by about 2x in both phases.

This is recorded as an open defect, not a datapoint. No further optimization
should be built until it is explained, because the target path may not be the
one being measured.
