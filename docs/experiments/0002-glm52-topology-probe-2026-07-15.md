# Experiment 0002 — GLM-5.2 hardware topology probe

Status: **complete; measured cost matrix accepted**.

## Hypothesis and contract

The actual GPU, NUMA, peer, kernel, and storage service costs differ materially
from the runtime's current VRAM-capacity-only 2:3:3 schedule. T1 is
characterization-only: it does not change runtime device selection, placement,
replacement, precision, kernel math, prefetch, or I/O policy.

- Primary metrics: median pinned H2D GB/s and cold row-1 INT4 expert service
  time for every CUDA-device and NUMA-source pair.
- Correctness: `make check`; complete byte comparisons for every transfer;
  expected-value checks for the actual INT4/INT8 kernels; physical/logical byte
  reconciliation for direct reads; and valid machine-readable JSON.
- Memory: process RSS at most 216 GiB; bounded probe buffers; no change to the
  runtime's 0.85 VRAM-cache budget.
- Rollback: an incomplete transfer, kernel mismatch, unaccounted direct I/O,
  memory breach, hidden runtime change, or a path distinction within measured
  noise rejects that conclusion.

The accepted prerequisite and branch base is
`b99db8d8e784e2dfc33b385d29382abb3ccbe538`. HEAD contained that revision
before editing. The standalone implementation commit is `d65364c` on
`infra/glm52-topology-probe`.

## Probe and metric definitions

`strata-topology-probe` emits schema `strata.topology_probe`, version 1. The
reusable `scripts/run_glm52_topology_probe.sh` pins CUDA enumeration to PCI bus
order, verifies the checkpoint index hash, builds the CUDA probe, captures
system and block-device evidence, validates every result with `jq`, and writes
the reduced cost matrix to `cost-matrix.json`.

Each complete matrix uses three within-path repetitions. Three complete
matrices were then run in order, and the tables below report the median of their
per-run medians plus cross-run ranges where they affect interpretation. This
reports all runs and prevents one favorable matrix from deciding the result.

Transfer measurements use 64 MiB buffers allocated and first-touched with
`numa_alloc_onnode`; sampled pages must query back on the requested NUMA node.
Pinned paths register those same buffers with CUDA. Timed copies are followed
by an untimed byte-for-byte round trip and checksum. Peer paths use an 8 MiB
activation extent. CUDA allocation measures a 64 MiB `cudaMalloc`/`cudaFree`;
synchronization measures an empty nonblocking stream.

Kernel measurements call Strata's existing CUDA backend with real GLM-5.2
dimensions. INT4 uses a `2048 x 6144` group-128 routed-expert projection and
INT8 uses a `6144 x 6144` group-128 ordinary linear. Decode uses one row and
representative prefill uses 30 rows. Packed values and BF16 scales produce a
known nonzero output checked after every call. Cold expert service allocates,
uploads, and executes the exact gate/up/down dimensions for a 19,464,192-byte
INT4 expert triple at one row; it is a transfer/three-matmul characterization,
not a fused SwiGLU runtime implementation.

Checkpoint I/O reads aligned 4 MiB ranges totaling 256 MiB per repetition at
queue depths 1, 4, and 8. The hot case warms the identical buffered range
before timing. The physical case uses `O_DIRECT`; `/proc/self/io` must report
exactly 256 MiB of physical reads for each timed sample. Checksums are computed
after timing so validation does not cap measured I/O throughput.

## Hardware identity

All three matrices used Linux `6.8.0-100-generic`, CUDA runtime 12.8, CUDA
driver API 13.2 / NVIDIA driver 595.71.05, two NUMA nodes, and the pinned
QuantTrio index hash
`43298345833417b1ad2a8b76d012a83d4f2275d532e5ab38e118566f1ac7b12b`.
The probe captured this live PCIe state before measurement:

| CUDA | Device | PCI BDF | CC | NUMA | Current PCIe | Total / free VRAM MiB |
|---:|---|---|---:|---:|---|---:|
| 0 | GeForce RTX 5060 Ti | `0000:03:00.0` | 12.0 | 0 | 8.0 GT/s x8 | 15,847 / 15,708 |
| 1 | GeForce RTX 3090 | `0000:82:00.0` | 8.6 | 1 | 8.0 GT/s x8 | 24,126 / 23,861 |
| 2 | GeForce RTX 3090 | `0000:83:00.0` | 8.6 | 1 | 8.0 GT/s x16 | 24,126 / 23,861 |

The JSON also records each CUDA UUID. Device numbers in this document are
valid only with `CUDA_DEVICE_ORDER=PCI_BUS_ID` on this machine identity.

## Host transfer cost matrix

Values are GB/s, median of the three complete-run medians. Pinned H2D was the
primary transfer metric and was stable across complete runs; its cross-run
range is included.

| GPU | NUMA source | Pageable H2D | Pinned H2D (run range) | Pageable D2H | Pinned D2H |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 5.985 | 6.715 (6.705–6.725) | 3.314 | 7.111 |
| 0 | 1 | 6.339 | 6.634 (6.571–6.648) | 5.349 | 4.544 |
| 1 | 0 | 5.870 | 5.888 (5.886–5.890) | 3.343 | 4.676 |
| 1 | 1 | 5.918 | 5.999 (5.988–6.017) | 6.278 | 6.579 |
| 2 | 0 | 6.061 | 7.009 (7.005–7.236) | 3.348 | 5.654 |
| 2 | 1 | 7.970 | 12.006 (11.976–12.045) | 10.082 | 13.119 |

GPU 2 from local NUMA node 1 is materially faster than every other pinned H2D
path. GPU 0's two H2D sources and GPU 1's two H2D sources are not separated
enough to justify roles from this matrix. Several pageable and pinned D2H run
medians varied substantially; they remain companion observations, not role
evidence.

## Kernel and cold-expert service matrix

Resident kernel seconds are medians of complete-run medians. All run ranges
were narrow.

| GPU | Encoding | Rows | Kernel median s | Cross-run range s |
|---:|---|---:|---:|---:|
| 0 | INT4 group-128 | 1 | 0.000112448 | 0.000111776–0.000113152 |
| 0 | INT4 group-128 | 30 | 0.003096160 | 0.003093632–0.003097216 |
| 0 | INT8 group-128 | 1 | 0.000388448 | 0.000387680–0.000391488 |
| 0 | INT8 group-128 | 30 | 0.011327744 | 0.011322368–0.011332960 |
| 1 | INT4 group-128 | 1 | 0.000094208 | 0.000092160–0.000095232 |
| 1 | INT4 group-128 | 30 | 0.002381120 | 0.002380416–0.002381376 |
| 1 | INT8 group-128 | 1 | 0.000261120 | 0.000260096–0.000262144 |
| 1 | INT8 group-128 | 30 | 0.007448320 | 0.007445088–0.007450176 |
| 2 | INT4 group-128 | 1 | 0.000093184 | 0.000092160–0.000093184 |
| 2 | INT4 group-128 | 30 | 0.002380928 | 0.002375744–0.002381184 |
| 2 | INT8 group-128 | 1 | 0.000261120 | 0.000260096–0.000261120 |
| 2 | INT8 group-128 | 30 | 0.007446048 | 0.007445856–0.007446272 |

Cold row-1 expert service includes allocation, pinned upload, activation copies,
and three actual INT4 kernels:

| GPU | NUMA source | Median service ms | Cross-run range ms | Median upload-wait ms |
|---:|---:|---:|---:|---:|
| 0 | 0 | 3.668 | 3.643–3.668 | 1.898 |
| 0 | 1 | 3.671 | 3.652–3.677 | 1.904 |
| 1 | 0 | 3.992 | 3.974–4.018 | 2.165 |
| 1 | 1 | 3.936 | 3.935–4.128 | 2.149 |
| 2 | 0 | 2.411 | 2.378–3.787 | 1.080 |
| 2 | 1 | 2.346 | 2.321–2.351 | 1.061 |

GPU 2 with local NUMA 1 is the stable best cold-service path. GPU 2 with NUMA
0 had one complete-run median of 3.787 ms while the other two were 2.378 and
2.411 ms. That path must be modeled with sensitivity bounds; its favorable
median alone is not accepted as a stable distinction.

## Inter-GPU activation staging

`cudaDeviceCanAccessPeer` returned false for all six ordered pairs, so no P2P
result is implied. The probe measured explicit device-to-pinned-host-to-device
staging. Values are median GB/s with cross-run range:

| Source -> destination | Staged GB/s | Cross-run range |
|---|---:|---:|
| 0 -> 1 | 2.861 | 2.845–3.285 |
| 0 -> 2 | 3.709 | 3.374–3.729 |
| 1 -> 0 | 3.250 | 2.661–3.273 |
| 1 -> 2 | 4.260 | 2.965–4.269 |
| 2 -> 0 | 4.489 | 2.686–4.492 |
| 2 -> 1 | 4.162 | 2.543–4.164 |

The staging paths are directional and several were unstable across complete
runs. T2 must use the recorded range or a conservative bound rather than a
single universal activation-transfer cost.

## Checkpoint I/O, allocation, and synchronization

| Read mode | Queue depth | Median GB/s | Cross-run range | Physical bytes per run |
|---|---:|---:|---:|---:|
| page-cache hot | 1 | 1.636 | 1.621–1.776 | 0 |
| page-cache hot | 4 | 5.533 | 5.042–5.643 | 0 |
| page-cache hot | 8 | 9.851 | 8.113–10.066 | 0 |
| `O_DIRECT` physical | 1 | 1.191 | 1.171–1.293 | 805,306,368 |
| `O_DIRECT` physical | 4 | 3.466 | 3.411–3.467 | 805,306,368 |
| `O_DIRECT` physical | 8 | 3.366 | 3.344–3.414 | 805,306,368 |

Every direct sample reconciled 268,435,456 logical bytes to exactly the same
`/proc/self/io` physical bytes. Queue depth 4 was the stable best physical-read
case; queue depth 8 did not improve it. Whole-session block reads were
3,195,838,464 bytes for the first run and 2,415,919,104 bytes for runs two and
three. The first run also physically populated the ranges used by the hot
warm-up. Whole-session block writes were 7,761,920 / 6,803,456 / 5,459,968
bytes and are benchmark logs/telemetry, not checkpoint writes.

Median 64 MiB allocation/free time across runs was 0.402 ms on GPU 0, 1.937 ms
on GPU 1, and 1.892 ms on GPU 2. Median empty-stream synchronization was 0.371,
0.326, and 0.312 microseconds. Initial allocation samples and several D2H and
staging paths had warm-state variance; raw arrays remain in the ignored JSON.

## Correctness and memory results

`make check` passed both tests. All three matrices passed the script's schema,
repetition-count, transfer-byte, kernel-reference, and direct-I/O validation.
All measured transfers and all six staged activation paths passed byte
comparison and checksum verification. All 12 kernel cases and all six cold
expert cases produced the declared nonzero reference output.

Maximum RSS across runs was 541,304 KiB, far below 216 GiB. In-process CUDA
high-water observations were 212,926,464 bytes on GPU 0 and 345,112,576 bytes
on GPUs 1 and 2. These are standalone-probe use, not a change to runtime VRAM
admission. `nvidia-smi` polling is retained as companion evidence but can miss
short allocations; the in-process observations are authoritative here.

## Decision

Accept T1. The hypothesis is supported: transfer and kernel service costs are
not represented by VRAM capacity alone. GPU 2's local x16 path is materially
faster for pinned H2D and stable cold expert service, while GPU 0's kernels are
slower than the RTX 3090 kernels at both decode and prefill rows. This is a cost
matrix, not authorization to change device roles or the scheduler.

T2 may use the frozen schema and ignored `cost-matrix.json` files. It must carry
the GPU 2/NUMA 0 cold-service range and the staged-transfer ranges into
sensitivity analysis. The current PCIe state was sampled before the matrix,
not continuously under load; link retraining and GPU power/clock state remain
possible contributors to the unstable companion paths.

## Handoff record

Stage: T1

Status: complete

Base revision: `b99db8d8e784e2dfc33b385d29382abb3ccbe538`

Branch and commit: `infra/glm52-topology-probe`, implementation `d65364c`

Hypothesis: real transfer and kernel costs differ materially from the current
VRAM-capacity-only schedule; supported for the stable GPU 2/NUMA 1 path and
the per-GPU kernel costs.

Files changed: standalone CUDA/NUMA topology probe, CMake target, reproducible
T1 script, offline-tools index, this experiment record, and the T1 ledger.

Commands and tmux session: `make check`; full matrices in
`strata-glm52-t1` and `strata-glm52-t1-repeats` using
`scripts/run_glm52_topology_probe.sh`.

Ignored artifacts: `results/t1-smoke.json`,
`results/glm52-topology/wrapper-smoke/`, and
`results/glm52-topology/full-v1/` through `full-v3/`, including system,
probe, block-I/O, VRAM, and session logs.

Correctness results: `make check` passed 2/2; all three full JSON matrices and
every transfer, kernel, expert, staging, and direct-I/O validation passed.

Run results and median: three complete matrices, each with three samples per
path. Primary across-run medians are 12.006 GB/s pinned H2D and 2.346 ms cold
row-1 expert service on GPU 2/NUMA 1. Full run medians and ranges are reported
above.

Peak RSS and per-GPU VRAM: 541,304 KiB RSS; in-process high-water used VRAM
212,926,464 / 345,112,576 / 345,112,576 bytes on CUDA 0 / 1 / 2.

NVMe, H2D/D2H, cache, allocation, and synchronization counters: each full
matrix issued 2,415,919,104 direct-read bytes plus hot warm-up/read traffic;
every timed hot read had zero physical bytes and every direct read reconciled.
Timed bulk H2D and D2H each covered 12 paths x 3 x 64 MiB. Each cold expert
sample uploaded 19,464,192 bytes. Runtime cache counters are not applicable
because the runtime was not executed or changed. Allocation and synchronization
medians are reported above.

Decision: accept the measured cost matrix and its explicit instability bounds;
do not implement roles or scheduler changes in T1.

Unresolved risks: GPU 2/NUMA 0 cold-service variance, staged-transfer and D2H
variance, single-snapshot PCIe link state, and short-lived allocations missed
by external VRAM polling.

Next stage ready: yes
