# Experiment 0060 — device-resident DeepSeek hidden-pipeline gate

Status: **the serial ascending-double device realization is rejected; the
external-style fused architecture is unresolved.**  No runtime path was
implemented or retained.  The complete 32,768-token compact KV allocation
fits admission, but Strata's exact sequential device dependency subset alone
takes 207.565 ms/token on either RTX 3090 and 277.167 ms/token on the RTX 5060
Ti.  This rejects that implementation before attention projections, paged
attention, KV work, CPU-MoE callbacks, shared/routed reduction, or
cross-device handoffs are added.  It does not measure or bound the external
stack's parallel TileLang mHC kernels.

This record follows experiment 0059's authoritative updated binding term.  It
does not reopen CPU scheduling: no new CPU mismatch was measured, and no CPU
scheduler or expert-kernel change was made.

## Branch and authority

- base: `e38f4d1 docs: retain the DeepSeek reference CPU-MoE arm`
- branch: `exp/dsv4-device-resident-hidden-pipeline`
- predecessor implementation: `383eeca exp: reproduce reference-shaped
  DeepSeek CPU MoE`
- governing current result: experiment 0059, especially its living delta table
  and re-instantiated whole-step model
- older next-action language that names more CPU callback profiling is
  superseded by 0059 and was not followed

Experiments 0050, 0053, 0055 and 0057 were reread before the gate.  Their
constraints remain binding:

- one fixed-address complete chain is the reference shape;
- removing projection round trips in isolation did not improve the step;
- the fused attention-input arm made the exact reductions slower on the GPU and
  diverged generation at token 33;
- the earlier fused CUDA mHC arm reassociated its reductions, changed routes and
  hashes, and was rejected;
- persistent FP8 DS-MLA KV must be read in place rather than gathered and
  restaged from the host.

## Contract stated before implementation

- **Hypothesis.**  Keeping the smallest complete dependent hidden, attention,
  KV and mHC chain at fixed device addresses, with stream-ordered CPU-MoE
  activation/output boundaries and GPU-side reduction, reduces the serialized
  130.145 ms non-MoE/handoff path.  An isolated projection or isolated kernel
  migration is not the experiment.
- **Target term.**  `Σ_serial` caused by returning the hidden state and nearly
  every projection to host ownership.
- **Primary metric.**  Median one-row decode at the retained operating point,
  with at most 100 ms/token (at least 10 tok/s) as the final gate.
- **Cheapest gate as originally declared.**  The mandatory exact fixed-address
  dependency subset was required to leave room above the 87.879 ms full-store
  CPU floor.  Its declared budget was therefore `100 - 87.879 = 12.121
  ms/token`, or `12.121 / 43 = 0.281884 ms/layer`.  That additive budget is
  valid for exposed serial work, but it is too strict as a bound on total GPU
  work in the intended overlapped graph.  The corrected architecture-level
  model is a measured CPU/GPU/PCIe dependency DAG: GPU work may exceed 12.121
  ms while hidden under the CPU lane, but its critical lane and unavoidable
  barriers must still complete within 100 ms/token.  The measured 207.565 ms
  exact GPU subset exceeds even that corrected whole-step ceiling.
- **Correctness gate if the mechanism gate passes.**  Real-format operation and
  layer fixtures, then full-model teacher-forcing hashes and generation-token
  equality while preserving DeepSeek attention/compression, mHC, router,
  scoring/top-6 normalization, shared experts, routed scaling and BF16/FP8
  contracts.
- **Memory ceiling.**  216 GiB host, the current 0.95 per-GPU VRAM admission,
  full 32,768-token KV/index capacity, fixed workspaces, zero steady-state
  NVMe, and explicit failure rather than fallback.
- **Rollback.**  If the lower bound exceeds 12.121 ms/token, changes an exact
  operation, or exceeds admission, stop before runtime wiring and retain only
  the failed experiment record.

The mechanism gate was expected to take seconds.  Loading the 155.827 GB tiled
arena would spend roughly 75 seconds in staging plus about 27 seconds in a
21-token prefill to measure only seconds of decode.  That full-system cost was
rejected until the dependency floor passed.

## Re-instantiated current cost model

The retained three-run candidate median from experiment 0059 is:

```text
complete graph MoE:  114.667 ms
attention:            71.140 ms
mHC pre + post:        53.061 ms
branch norm:            1.374 ms
remaining:              4.572 ms
total:                244.812 ms
```

The hidden is host-visible between these phases, so the current operating
point is a sum of serial terms, not an overlapped maximum:

```text
τcurrent = 114.667 ms complete MoE + 130.145 ms Σserial
         = 244.812 ms/token
```

The resource instantiation uses the retained candidate's 15-step decode JSON,
not a constant copied from a different context:

| Resource | Work per token `W_r` | Measured/effective `B_r` | Current term |
|---|---:|---:|---:|
| host DRAM, routed CPU | 3.449290752 GB canonical payload | 32.276 GB/s aggregate, 16.138 GB/s/socket | 106.869 ms routed; 114.667 ms complete MoE |
| host mHC projection/transform | 135,266,304 projection bytes plus exact pre/post arithmetic | about 3.04 GB/s over the measured 44.509 ms mHC-pre phase | 53.061 ms pre+post |
| PCIe H2D activations | 11,399,168 B | about 37.10 GB/s across the three links from the 0.307219 ms critical-device service | 0.307 ms link service, embedded in serial owners |
| PCIe D2H activations | 9,533,440 B | about 22.75 GB/s across the three links from the 0.419110 ms critical-device service | 0.419 ms link service, embedded in serial owners |
| GPU resident spine | 9,069,011,072 B of admitted resident weights visited by the step | rated 448/936/936 GB/s; the capacity-weighted serial placement has a roughly 12 ms pure-read floor before kernels and handoffs | current critical-device kernel counter is 2.901 ms, but it is a max across devices and must not be added as though dependent layer devices ran concurrently |
| compact KV/index at 32,768 tokens | 120,148,480 B capacity; the measured 586-row host path stages 18.07 MiB/token | no device in-place service existed to measure before this experiment | 6.33 ms direct staging on the slowest GPU plus host gather/synchronization in attention |
| NVMe | 0 decode B | not on the steady-state path | 0 ms |

Thus the binding term before the experiment is `Σ_serial = 130.145 ms`, not
CPU DRAM.  Perfecting routed CPU from its retained median to the 87.879 ms
probe can save only 18.990 ms.  The best real routed arm at 89.226 ms proves
the probe regime is attainable; it does not justify another CPU mechanism.

The sign check for the proposed chain was:

- host DRAM should lose roughly 135 MB/token of mHC projection reads while the
  3.449 GB routed payload remains unchanged;
- PCIe should approach only the stream-ordered CPU-MoE activation and result
  boundaries, roughly one 16 KiB row in each direction per layer plus route
  metadata;
- HBM gains mHC parameter and compact-KV reads;
- GPU SM work gains exact mHC/query/KV reductions and fused transforms;
- CPU routed work, precision, route order, expert count and top-k are unchanged;
- no host fallback, NVMe read, or speculative/recomputed pass is allowed.

The gate below measures the negative GPU-SM sign before building the rest.

## Full-context KV admission before code

The existing admission-only executable was run against the real checkpoint:

```bash
build/strata-deepseek-run \
  --model models/dsv4f --devices 0,1,2 \
  --host-memory 216G --vram-fraction 0.95 \
  --max-context 32768 --tiled-host-moe --block-kv-cache \
  --admission-only --quiet --json
```

It admitted the exact compact cache with:

| Quantity | Bytes |
|---|---:|
| FP8/BF16 KV plus FP4 learned-index payload | 119,791,040 |
| block metadata | 357,440 |
| physical compact cache | **120,148,480** |
| compressor/index state included in total KV state | 132,354,560 |
| required host bytes | 159,438,359,388 |
| aggregate 0.95 VRAM budget | 63,186,052,709 |
| resident spine | 9,069,011,072 |
| existing per-device workspace reserve | 805,306,368 |
| remaining expert-cache budget before a device-KV reservation | 53,311,735,269 |

At the current `[0,0,1,1,1,2,2,2]` layer schedule the complete physical cache
is 28,839,936 / 61,384,704 / 29,923,840 bytes, or
27.504 / 58.541 / 28.538 MiB by device slot.  FP32 mHC parameters add
135,266,304 bytes aggregate, split approximately by the 12/16/15 layer counts,
and fixed hidden workspaces are sub-megabyte per slot.  All are inside the
existing 5% physical-VRAM reserve.  Capacity is not the falsifying term.

## Cheapest exact dependency-floor probe

The ignored throwaway CUDA probe used fixed process-lifetime addresses and one
stream.  Each measured step queued all 43 dependent layers and synchronized
only once through CUDA events.  Every layer queued:

1. exact mHC hidden RMS fold in ascending order;
2. 24 independent mHC projection rows, each an ascending 16,384-term double
   fold;
3. the reference-order 4x4 Sinkhorn transform, mHC pre reduction, BF16 round,
   and branch RMSNorm;
4. mHC post into the fixed next hidden buffer;
5. exact 1,024-wide query-rank RMSNorm, 64 independent 512-wide query
   RMSNorms, and the 512-wide KV RMSNorm;
6. the second mHC pre/post transition into the next fixed hidden buffer.

The next hidden pointer became the following operation's input, so the queue
preserved the complete transition dependency.  The probe deliberately omitted
all attention projection matmuls, RoPE, paged FP8 insert/read, attention score
and value work, the router, both CPU activation boundaries, CPU MoE, GPU shared
expert, routed/shared reduction and cross-GPU transitions.  It also reused one
zero-valued 1.5 MiB projection for all layers, keeping it warm in cache instead
of reading the production layer-specific 135 MB.  These omissions and warm
reuse can only understate the complete pipeline.

The accepted host oracle accumulates every mHC projection row in ascending
column order in double precision.  A one-lane-per-row CUDA fold preserves that
order.  Parallel reduction was not used: experiment 0009's parallel CUDA mHC
changed routes and hashes, while experiment 0053's fused attention chain
diverged generation.  Those results reject those particular reassociations;
they do not establish the numerical behavior of the external stack's
production TileLang reduction tree.

Build command:

```bash
nvcc -O3 -std=c++20 -lineinfo \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_120,code=sm_120 \
  exact_dependency_floor.cu -o exact_dependency_floor
```

CUDA 12.8 (`V12.8.93`) compiled the probe.  It ran three warm-up steps followed
by eleven measured steps per device.  The CUDA runtime enumerated the two 3090s
as devices 0/1 and the 5060 Ti as device 2; names, not the operator-facing PCI
ordering, identify the rows below.

| GPU | min | median | max | median/layer | gate/layer | ratio to gate |
|---|---:|---:|---:|---:|---:|---:|
| RTX 3090 A | 207.561 ms | **207.565 ms** | 207.582 ms | **4.82709 ms** | 0.281884 ms | **17.12x** |
| RTX 3090 B | 207.561 ms | **207.564 ms** | 207.583 ms | **4.82707 ms** | 0.281884 ms | **17.12x** |
| RTX 5060 Ti | 277.162 ms | **277.167 ms** | 277.186 ms | **6.44575 ms** | 0.281884 ms | **22.87x** |

The eleven-sample 3090 ranges are only 0.0215/0.0225 ms wide.  This is not a
variance classification: the optimistic lower bound misses by 195.443 ms on
the faster GPU, about nine thousand times the observed range.  It also exceeds
the entire current 130.145 ms non-MoE/hand-off target term by 77.420 ms,
**1.595x**, despite omitting most of the proposed chain.  The result therefore
falsifies both the 100 ms final gate and the more basic claim that this exact
device chain reduces the term it targets.

The probe owned 1,874,116 payload bytes; CUDA rounded that to a 2,097,152-byte
free-memory delta.  Reported physical VRAM was 25,298,141,184 bytes on each
3090 and 16,617,177,088 bytes on the 5060 Ti.  Maximum process RSS was
98,548 KiB on the 3090s and 105,460 KiB on the 5060 Ti.  Every arm reported
zero major faults, swaps, filesystem input and filesystem output.  It did not
open model shards, so admission, load, prefill, warm-up and decode checkpoint
reads are not conflated with the measured device window.

## Decision

The gate is negative and binding for the measured ascending-double
implementation.  Its mandatory exact dependency subset is 207.565 ms on the
faster GPU, more than twice the complete 100 ms target even under the corrected
overlap model.  Adding the omitted projections, in-place KV attention,
callbacks, reduction and handoffs cannot make that implementation eligible.
Therefore:

- do not implement or retain the complete runtime pipeline using this serial
  ascending-double device reduction;
- do not present the old isolated projection or mHC arms as substitutes;
- do not run a full-model teacher-forcing, generation or three-arm throughput
  matrix after a failed prerequisite gate;
- do not reopen CPU scheduling: this experiment measured no new CPU mismatch;
- retain no failed runtime code.

The complete device-resident architecture remains eligible for a new branch
only after the external production mHC kernel is timed directly and a
real-format operation fixture plus full-model teacher-forcing and generation
oracle establish whether its FP32/TF32 parallel arithmetic satisfies an
explicitly accepted numerical contract.  That is a different bounded
hypothesis, not a continuation of this exact sequential realization.  Under
the present ascending-double implementation,
the current host AVX2 mHC path remains required even though host ownership is
the whole-step bottleneck.

## Artifacts

The probe source and generated binary were throwaway files under ignored
`results/dsv4-device-hidden-pipeline-gate/` and were removed after the numbers,
compiler identity, allocation accounting and hashes were copied into this
tracked record:

```text
source sha256  6dfee53e4ad92fe55afc26c0d0e73a5e3b923facaeacb875062dce3c6b631ef9
binary sha256  58b68450f07256f2646c6f5ab389fa374a6069bfec3e0bac75a859607e9555c3
```

After removing those failed artifacts, `make check` passed both CTest targets
in 154.79 seconds.  No hardware directory was added or force-added.
