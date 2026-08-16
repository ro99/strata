# Experiment 0106 — DSV4 expert-arena NUMA attribution

Status: **stopped before policy work because the unbound-production-arena
premise is false.** Rank-local TP2 device-resident execution stages and consumes
the already NUMA-bound tiled arena. Live page accounting shows exact, stable
MPOL_BIND placement: 77,913,391,104 bytes on node 0 and the same on node 1,
plus a 1,059,061,760-byte embedding tail under default policy. No placement
candidate or three-arm variance matrix was run.

## Predeclared hypothesis and gate

The proposed hypothesis was that production prefill consumed the bare
`MAP_PRIVATE | MAP_ANONYMOUS` resident arena, allowing loader-thread first touch
to place its approximately 156 GB inconsistently relative to GPUs 1 and 2. The
target resource was remote host-memory service during routed-expert H2D. The
eventual primary gate would have been the spread of expert-upload time over
three identical 677-token, page-8192 arms, compared with the recorded
15.50/16.11/32.38 s spread.

The dependency was explicit: first identify the allocation production uses and
measure its live pages. If production used the bound tiled site, that finding
outranked a placement change and stopped the experiment. Correctness and the
216 GiB host-memory admission were unchanged. The diagnostic budget was one
approximately three-minute live-placement run, followed only if justified by
three approximately four-minute variance arms. No 2,612-token arm was planned.

## Source ownership result

Both allocation sites exist, but the second one is not active at this operating
point:

1. The canonical resident path at `deepseek_checkpoint.cpp` around line 515 is
   a bare anonymous mapping populated by shard-task reader threads.
2. The tiled path around line 354 allocates the transformed two-shard expert
   arena and calls `numa_bind_range` on each complete shard before any worker
   touches it: shard 0 to node 0 and shard 1 to node 1.

The CLI settles which branch production takes. Both
`--device-resident-runtime` and `--decode-topology rank-local-tp2` set
`host_routed_moe=true`. Initialization passes that value as the final
`tiled_experts` argument to `Dsv4ResidentWeightStore::stage`, so the production
0105/0106 command selects the first, bound branch. The output JSON independently
reports `host_routed_moe: true`.

Centralized page prefill then checks `resident.tiled_experts()`, acquires both
transformed shards with `acquire_tiled_expert`, and uploads both shards of each
selected expert to the expert's assigned GPU. The bare canonical extents are
therefore not the source of the measured prefill expert traffic. Rank-local
decode has a separate device store, but it does not replace centralized
prefill ownership.

This corrects two records: the handover's statement that the production arena
has no NUMA policy is outdated, and the second unbound allocation cannot be
used to explain this production configuration merely because it exists in the
same class.

## Hardware topology and live placement

`nvidia-smi topo -m` confirms:

- GPU 0: NUMA affinity 0, CPU affinity `0-13,28-41`;
- GPU 1: NUMA affinity 1, CPU affinity `14-27,42-55`;
- GPU 2: NUMA affinity 1, CPU affinity `14-27,42-55`.

Each node has about 129 GB. The complete resident mapping is
156,885,843,968 bytes, so making every page local to node 1 is impossible under
capacity even before accounting for the rest of the process.

The preserved capture sampled `/proc/<pid>/numa_maps` every 250 ms while the
real 0105 binary staged the model. At the end of staging it reported:

| Region | Policy | Node 0 pages/bytes | Node 1 pages/bytes |
|---|---|---:|---:|
| transformed shard 0 | `bind:0` | 19,021,824 / 77,913,391,104 B | 0 |
| transformed shard 1 | `bind:1` | 0 | 19,021,824 / 77,913,391,104 B |
| embedding tail | `default` | 86,663 / 354,971,648 B | 171,897 / 704,090,112 B |

The rows sum exactly to the runtime's recorded resident-stage byte count:
`2 * 77,913,391,104 + 1,059,061,760 = 156,885,843,968` bytes. The two expert
VMAs never contained a page on the wrong node. This is policy placement, not a
first-touch accident, and it cannot alternate between the 15 s and 32 s
prefill-upload regimes across identical runs.

The eight staging workers were also sampled. Across 640 scheduler observations,
293 (45.8%) ran on node 0 CPUs and 347 (54.2%) on node 1 CPUs; individual
threads migrated. That can affect staging write locality and load time, but
MPOL_BIND fixes the destination pages before those writes and the workers have
ended before prefill. Their CPU placement cannot explain the later expert H2D
variance.

## Current resource signs, not a new policy

Because both consuming GPUs are on node 1 while every expert consists of both
transformed shards, each cold expert upload reads approximately half its bytes
locally from node 1 and half remotely from node 0. That is a deterministic
capacity/topology cost. It may leave a separate mean-throughput opportunity,
but it is not the run-to-run placement defect this experiment was authorized
to test. Moving more expert bytes to node 1 would necessarily consume scarce
node-1 capacity and make some other host allocation or expert bytes remote; no
such policy was designed or built after the attribution gate fired.

## Execution and preservation

The diagnostic process reached complete resident staging, which is why the
page totals above are exact. It was then terminated by the agent that launched
it, before a model result was produced, because continuing after the ownership
gate would only spend time on a falsified premise. The empty partial JSON and
the complete `numa-maps.log`, `thread-cpus.log`, topology capture and probe
script remain under `results/dsv4-0106-numa-attribution/` and on this experiment
branch. No runtime source changed.

`make check` passed before this result commit.

## Verdict

0106 falsifies absent/variable NUMA policy as the source of the expert-upload
bimodality. The requested three-run variance gate has no placement candidate to
evaluate and was deliberately not launched. The 15.50/16.11/32.38 s spread
remains a real measurement blocker, but its cause must be re-attributed from
the upload path rather than assigned to arena page placement. Any such
attribution, or any separate capacity-aware mean-locality policy, is a new
decision requiring authorization.
