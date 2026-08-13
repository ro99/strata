# Experiment 0083 — Stage-6 NCCL/runtime memory gate

Date: 2026-08-11
Branch: `exp/dsv4-stage6-nccl-memory-gate`
Base: `5f9df610109d235316ed9ccc94b70f64a7bd2674`
Result: **PASS FOR THE NCCL MEMORY BOUNDARY / ELIGIBLE FOR HUMAN REVIEW ONLY**

This is a read/measure-only prerequisite for the user-reviewed Stage-6
decision. It does not implement rank-local model topology, mHC, attention,
MoE integration, graph capture, CPU arithmetic, or Stage 7. Experiments 0077,
0078, 0079, 0080, 0081, and 0082 retain their original scopes and decisions.

## Question and predeclared gate

The hypothesis was that the actual two-rank NCCL/runtime boundary can be
priced after lazy allocations are materialized, rather than treating NCCL as
free or assuming its memory overhead is zero. The target resource is VRAM
capacity; latency, CPU routed work, GPU service, PCIe, host issue, and
`Sigma_serial` have neutral signs for this experiment. A larger resident
working set, an allocation after warmup, or a physical SHM allocation that
cannot be reserved is a positive capacity sign and rejects the gate.

The primary metric is the maximum per-rank memory used at each boundary:

```text
pre-context
 -> explicit CUDA contexts
 -> fixed device buffers + two streams + final completion event
 -> NCCL 2.28.9 communicator initialization
 -> first warmup 86-reduction FP32/U32 chain (lazy allocations)
 -> second production-shaped 86-reduction chain
```

The chain contains 43 attention-boundary and 43 MoE-boundary reductions. Each
reduction performs an FP32 hidden all-reduce, a U32 MAX status all-reduce, and
BF16 fail-closed publication. The fixed probe arena is 57,356 B/rank. No
centralized model arena is allocated.

Hard gates were:

- both RTX 3090 ranks, `CUDA_VISIBLE_DEVICES=1,2`, runtime devices 0/1,
  P2P capability/use `0/0`, and no unrelated GPU process;
- exact rank-ordered FP32 association and BF16 publication;
- four injected failures: pre- and post-data-collective on ranks 0 and 1,
  with both ranks participating and external output withheld;
- no post-warmup CUDA allocation, callback, fallback, or non-finite result;
- host RSS below `231,928,233,984 B` and candidate memory below
  `21,287,272,448 B/GPU`; and
- conservative Stage-6 total, using the full measured peak as an additive
  reserve, at least 1 GiB below the per-GPU ceiling.

The allocator variance reserve was predeclared as:

```text
reserve = max_peak_across_fresh_arms - min_peak_across_fresh_arms
```

The first arm was a cheap structural falsifier. After it passed, three fresh
processes were run to establish the range. Each process had one warmup and one
measured production chain; process setup was dominated by approximately 0.35 s
NCCL communicator initialization while the post-warmup chain was 4.303–4.656
ms. This intentionally short probe was chosen over a model load or any
Stage-6 candidate launch, both of which were more expensive and unauthorized.

## Operating point and implementation

The reusable runner is:

```text
scripts/run_dsv4_stage6_nccl_memory_gate.sh
```

The measured command was equivalent to:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 \
build-stage6-nccl-memory/strata-dsv4-nccl-reduce-probe \
  --device0 0 --device1 1 --memory-gate --warmups 1 --repetitions 1
```

NCCL was the installed 2.28.9 CUDA 13 C API. The probe uses two CUDA primary
contexts, one nonblocking stream per rank, one final completion event, fixed
BF16/FP32/status buffers, and two grouped rank communicators. NCCL logs select
`SHM/direct/direct` with `isAllCudaP2p 0`; no peer access is enabled.

The opt-in probe adds dynamic NVML process snapshots keyed by PID/GPU UUID and
CUDA `cudaMemGetInfo` snapshots. `/dev/shm` is enumerated at each boundary.
The visible file count was zero at every snapshot because NCCL unlinks or
maps its shareable files; NCCL logs show its requested shareable buffers, but
resident physical SHM transport bytes remain **not measured**, not zero.

## Exactness and failure result

All three binding arms completed the second 86-reduction chain exactly. The
production chain times were:

| arm | production 86-chain ms | exactness |
|---|---:|---|
| arm2 | 4.303117 | pass |
| arm3 | 4.383272 | pass |
| arm4 | 4.278794 | pass |

The first exploratory arm, before a warning-only rebuild, measured 4.655570 ms
and is retained separately as `arm1`; it had the same memory values and all
gates passed. The binding production range is therefore `4.278794–4.383272 ms`
(the arm1 value is not used for the final timing range).

All four failure cases passed in every process:

```text
pre-data collective:  rank 0 pass, rank 1 pass
post-data collective: rank 0 pass, rank 1 pass
```

The failure status is reduced with U32 MAX on both ranks, BF16 output is
withheld/zeroed, and no process exited through a fallback. There were zero
post-warmup allocations; the production chain's memory delta from the warmup
snapshot was exactly zero.

## Per-GPU staged memory

CUDA `cudaMemGetInfo` values were identical on both ranks and in all three
binding arms:

| stage | used B/GPU | delta from context | delta from prior stage |
|---|---:|---:|---:|
| context ready | 278,003,712 | 0 | — |
| fixed buffers/streams/event | 280,100,864 | 2,097,152 | 2,097,152 |
| communicators initialized | 391,249,920 | 113,246,208 | 111,149,056 |
| first warmup chain complete | 428,998,656 | 150,994,944 | 37,748,736 |
| production chain complete | 428,998,656 | 150,994,944 | 0 |
| failure sweep complete | 428,998,656 | 150,994,944 | 0 |

Each device reported `total_bytes=25,298,141,184`. The raw fixed arena is
57,356 B/rank; the 2,097,152 B CUDA delta is allocator granularity, not a
claim that the raw buffers consume only the raw size.

In-process NVML mapped CUDA device 0 to physical GPU UUID
`GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` (NVML index 1) and CUDA device 1
to `GPU-81fe4578-59b2-37c4-421e-287cdac78704` (NVML index 2). Target-process
NVML usage was identical on both ranks:

| stage | NVML target used B/GPU |
|---|---:|
| context ready | 268,435,456 |
| fixed buffers/streams/event | 270,532,608 |
| communicators initialized | 381,681,664 |
| first warmup chain complete | 419,430,400 |
| production chain complete | 419,430,400 |

The process PID had zero unrelated processes on all NVML snapshots. External
`nvidia-smi` before/after files show only the two idle target RTX 3090s; the
probe process had exited by the after snapshot, so the in-process NVML samples
are the binding per-process evidence.

The measured CUDA peak across fresh arms was exactly identical:

```text
max_peak - min_peak = 428,998,656 - 428,998,656 = 0 B
allocator variance reserve = 0 B
```

The stage-only incremental boundary after context was 150,994,944 B. For the
required conservative gate, the full pre-context-to-warmup peak
428,998,656 B is added without subtracting any projected rank-local storage.

## Stage-6 reserve arithmetic

The retained rank-local projection from Stage 4/5 is:

```text
rank-local projection             10,384,097,480 B/GPU
full measured context/NCCL peak      428,998,656 B/GPU
repeat-range allocator reserve               0 B/GPU
conservative candidate total       10,813,096,136 B/GPU
hard ceiling                       21,287,272,448 B/GPU
remaining headroom                 10,474,176,312 B/GPU
```

The remaining headroom is greater than 1 GiB. This is a memory-boundary
eligibility result only. It does not combine centralized control residency
with the rank-local projection, and it does not authorize the Stage-6 model
candidate. NCCL physical SHM resident bytes remain an explicit unresolved
component; the arithmetic is conservative with respect to measured CUDA
runtime usage but cannot claim an unmeasured host-SHM byte bound.

Logical communication accounting per rank over the 86-chain is:

```text
FP32 data all-reduce input/output       1,409,024 / 1,409,024 B
U32 status all-reduce input/output            344 /       344 B
BF16 publication writes                 704,512 B
application H2D/D2H in probe                    0 / 0 B
physical NCCL SHM transport             not_measured
```

The communication probe is not a complete decode and reports no throughput
win. Its capacity target is orthogonal to the accepted 0082 CPU-routed
argmax. No `W_r/B_r` latency constant is substituted for the unknown physical
SHM volume.

## Artifacts and validation

Ignored deterministic results are under:

```text
results/dsv4-stage6-nccl-memory/arm1/  # exploratory pre-final-binary arm
results/dsv4-stage6-nccl-memory/arm2/  # binding
results/dsv4-stage6-nccl-memory/arm3/  # binding
results/dsv4-stage6-nccl-memory/arm4/  # binding
```

Binding raw generation-log hashes:

```text
arm2/generation.log 5fb936d0d4010597059f0ddc8aeb0433a0ac2da679a64ae4cc482749ebe774c3
arm3/generation.log 032607a067240ea564c24b761251e6502a4ad35d748629c747001d4a65282599
arm4/generation.log 15efc8aac4780a1427e22528092a6c899fbeb89c0e260cfb2d3e879ae3af22f7
```

The final probe and runner hashes are:

```text
build-stage6-nccl-memory/strata-dsv4-nccl-reduce-probe
dfb8cfb141d56da3cef44ec07cfa80401225b102286452dc7c943c10023760c2

scripts/run_dsv4_stage6_nccl_memory_gate.sh
41d60c62b2a6c3c58fef8448cc58d7cbd3d366db355119b7962deeaf3709f484
```

The ignored per-arm `sha256sums.txt` files retain the environment, GPU
before/after accounting, status, binary, and raw-log hashes. The NCCL raw logs
also preserve the `SHM/direct/direct` and `isAllCudaP2p 0` selection evidence.

The focused NCCL target build passed, the default non-memory NCCL probe smoke
returned status 0 with exactness and unchanged grouped accounting, root
`make check` passed both CTest targets (`2/2`), and `git diff --check` passed.
The full staged diff was reviewed before the result commit.

## Decision and review boundary

**PASS / REVIEW REQUIRED for the Stage-6 NCCL/runtime memory boundary.** The
measured CUDA/NVML peak and zero post-warmup delta satisfy the declared memory
reserve gate, exactness and failure closure pass, and no unrelated GPU process
was present. Physical NCCL SHM transport bytes remain not measured and must be
reserved or rechecked if a model candidate is later authorized.

Stage 6 remains **BLOCKED / NOT AUTHORIZED** until root/user review accepts
this memory prerequisite together with the prior topology/control evidence.
No Stage-6 candidate, mHC integration, attention integration, graph work, CPU
arithmetic work, or later stage was started here.
