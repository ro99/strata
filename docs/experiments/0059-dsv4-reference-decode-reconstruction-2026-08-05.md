# Experiment 0059 — reconstruct the 10.2 tok/s reference decode

Status: **the reference reconstruction is retained, and the bounded 1 ms
worker-spin plus same-node gate/up and down work-stealing arm is retained on a
three-run interleaved short gate.**  This is not a 98 ms/step target claim.  The
external no-speculative stack reproduced 10.0 tok/s under Nsight Systems,
against its previously measured unprofiled 10.2 tok/s median.  The trace and
live NUMA state identify two architectural differences large enough to explain
the original 122 ms/step gap: the reference keeps the decode activation and KV
caches on the GPUs behind one graph launch, and it splits every expert over two
CPU ranks whose transformed arenas occupy opposite NUMA nodes.

## Contract

- Observation: Strata takes 222.6 ms/step while the same checkpoint takes
  97.6 ms/step in the external stack without speculative decoding.  Source
  inspection showed breakable CUDA graphs, GPU-resident paged KV caches and a
  stream-ordered CPU MoE call, but did not price their critical paths.
- Hypothesis: the reference's CPU-MoE/DRAM work is the graph's longest per-layer
  term, while graph capture, fixed buffers, attention fusion and resident KV
  keep GPU and submission work below it.
- Primary metric: one-row decode wall time and the CPU/GPU launch, graph,
  synchronization and transfer timeline at a 666-token measured prompt.
- Correctness gate: reproduce the no-speculative reference rate without
  changing model code, precision, routing, sampling or checkpoint state.
- Memory ceiling: the reference's two roughly 78 GiB CPU ranks and two 3090s at
  `--gpu-memory-utilization 0.87`; Strata was not resident concurrently.
- Rollback: delete the ignored `results/dsv4-reference-profile` artifacts and
  the launch harness.  No runtime path changed.

The first arm retained the last 15 seconds of a graph-granularity trace.  Two
three-second node-trace attempts were also made.  Model load was not under test;
the short measured windows were cheaper than an end-to-end matrix.

Node attribution was not recoverable with the installed profiler/driver pair.
Nsight 2024.6 preserved 86 MiB of raw node events but its importer failed on an
unrecognized CUDA 13.2 GPU UUID.  A temporary 2025.5 CLI was extracted under
`/tmp` without system installation.  It could not import the old collector's
missing timestamp metadata; a fresh 2025.5 collection reported roughly 400,000
CUPTI events per worker but warned that its CUDA 13.1 libraries do not support
the installed 13.2 driver, and the retained report omitted the CUDA tables.
The repeat stop condition was applied: no third node-trace attempt.  The graph
trace, source order, live NUMA state and disassembly remain independently usable.

## Reproduction

The checked-in harness launches the installed site package, which is the code
actually used by the reference environment:

```bash
nsys start --session-new=dsv4ref \
  --sample=none --cpuctxsw=none \
  --output=$PWD/results/dsv4-reference-profile/decode

nsys launch --session=dsv4ref --trace=cuda,nvtx,osrt \
  --cuda-graph-trace=graph \
  scripts/run_deepseek_v4_reference_nospec.sh

/home/rodrigo/Developer/Lvllmds4-x/venv/bin/python \
  /home/rodrigo/Developer/Lvllmds4-x/bench/bench.py \
  --skip-prefill --tg-length 512 --tg-max-tokens 96 --tg-runs 1

nsys stop --session=dsv4ref --keep=15
```

The benchmark's `--decode-only` option is parsed but not consulted; the working
flag is `--skip-prefill`.  The first attempt was stopped after three unrelated
prefill rows and rerun with the working flag.

## Reference result

```text
prompt tokens:       666
completion tokens:    96
TTFT:              10.58 s
total:             20.16 s
profiled decode:    10.0 tok/s
profiled interval: 100.8 ms/step (95 intervals in 9.58 s)
unprofiled control: 10.2 tok/s median, 97.6 ms/forward
```

The graph trace retained 82 graph executions per rank over 8.21 seconds.  Their
starts are about 100 ms apart and their GPU spans average 93.87 and 93.98 ms.
The tracing overhead accounts for the small difference from the unprofiled
control; this arm validates the execution shape rather than replacing the
three-run result in experiment 0055.

## One reference decode step

One TP rank executes the following ordered path.  Both ranks do it concurrently
and reduce their tensor-parallel partials through NCCL; there is no GPU P2P on
this machine.

| Order | Stage | Processor and overlap | Representation / boundary |
|---:|---|---|---|
| 1 | request metadata and fixed input buffers | vLLM worker CPU, before replay | token, positions and slot maps copied into persistent device buffers |
| 2 | embedding and first attention mHC pre | GPU graph | BF16 hidden; TileLang mHC also fuses attention RMSNorm |
| 3 | later branch/layer transition | GPU graph | one `mhc_fused_post_pre_tilelang` fuses prior post, next pre and RMSNorm |
| 4 | attention input projections | four CUDA streams | fused `wq_a+wkv` on default; compressor score, indexer weight and indexer-compressor score GEMMs on three reusable auxiliary streams |
| 5 | Q/KV normalization | GPU graph | fused Q-rank and 512-wide KV RMSNorm |
| 6 | metadata-dependent attention body | GPU graph in FULL decode mode | default stream runs `wq_b` plus fused Q norm/RoPE and KV quantized paged insert; auxiliary streams overlap the indexer and compressor |
| 7 | sparse MLA + SWA attention | GPU | FlashMLA reads persistent `fp8_ds_mla` paged caches; no historical KV is staged through the host |
| 8 | attention output | GPU graph | inverse RoPE, grouped `wo_a`, then FP8 Marlin `wo_b` |
| 9 | attention post / FFN pre / norm | GPU graph | fused mHC kernel, BF16 result |
| 10 | router | GPU graph | FP32 router logits and exact top-6 |
| 11 | routed and shared experts | persistent CPU workers plus an auxiliary GPU stream | one stream-ordered `cpu_decode(stream, rows, topk, hidden_gpu, ids_gpu, weights_gpu, output_gpu)` call per layer overlaps the GPU shared expert; fixed FP32 routed output buffer |
| 12 | join, TP reduction and next layer | GPU graph / NCCL | shared and routed results join, then the two intermediate-sharded rank partials are reduced before the dependent next layer |
| 13 | final mHC head, norm, logits, sample | GPU plus vLLM scheduler | fixed buffers remain device-resident through logits |

`attention_impl` is decorated with `@eager_break_during_capture`, but that
wrapper explicitly executes without breaking when the runtime mode is `FULL`.
The log says `Capturing CUDA graphs (decode, FULL)`, and both the graph trace
and CUDA API table show exactly one graph launch per rank and step.  Thus decode
attention is captured inside one uninterrupted graph in this configuration;
the breakable fallback applies to other runtime modes.

## One Strata decode step

`forward_token` allocates a host float hidden state, calls `forward_hidden`,
then returns through the host-visible output head.  Each of 43 layers executes
two branches serially:

1. Copy the full mHC residual on the host.
2. Run host FP32 mHC pre, BF16-round its output, and host RMSNorm.
3. For attention, submit `wq_a`, `wq_b` and `wkv` as independent CUDA matmuls.
   Every matmul uploads its host activation, launches, downloads FP32 output and
   synchronizes.  Query normalization, RoPE and KV bookkeeping run on CPU.
4. Gather sliding and compressed KV rows into host vectors.  `flash_attention`
   repacks and uploads the complete selected working set, launches one kernel,
   downloads attended heads and synchronizes.  Host inverse RoPE follows.
5. Submit `wo_a` and `wo_b` as two more upload/launch/download/synchronize
   matmuls, then run mHC post on the host.
6. Repeat mHC pre and norm for FFN.  Route on the host-visible activation,
   acquire/copy selected expert weights, execute the routed and shared experts
   on GPUs, collect FP32 outputs to the host, combine and run mHC post.

Layers are assigned to three GPUs, but layers themselves are dependent and run
sequentially.  The resident expert cache reduces NVMe and some PCIe volume; it
does not create the reference's tensor-parallel execution or keep the hidden
chain on-device.

## Direct measured comparison

| Term, one decode step | Reference | Strata main | Consequence |
|---|---:|---:|---|
| wall time | about 100.0 ms traced; 97.6 ms control | 222.6 ms | 122.6--125.0 ms gap |
| outer graph launches | 1 per TP rank | 0 | reference amortizes the whole fixed-shape chain |
| blocking matmul submissions | none at the Python layer inside replay | about 407 | Strata serializes device/host ownership repeatedly |
| CUDA stream synchronizes | 1 per rank, 79.2 ms average wait | about 536 total synchronizations | reference waits once on the completed step |
| graph GPU span | 93.9 ms per rank | no graph | contains attention, mHC, projections, collectives and CPU-MoE host nodes |
| GPU kernels outside the graph | 54.5/rank, 0.94 ms/rank | n/a | request preparation and sampling, not the model attention body |
| visible H2D outside graph | 0.0507 MiB/rank | 12.67 MiB aggregate activation H2D | reference keeps hidden and KV state resident |
| visible D2H outside graph | effectively bytes/rank | 13.76 MiB aggregate activation D2H | Strata returns almost every projection to CPU |
| selected attention KV staging | persistent paged cache | 18.07 MiB/step, 6.33 ms on the slowest GPU | volume and a synchronization disappear in the reference |
| attention | inside graph | 83.36 ms | includes five projection round trips and host transforms |
| mHC pre/post | fused GPU graph nodes | 23.22 ms | reference fuses post+pre+norm across branch/layer boundaries |
| MoE | TP2 CPU routed + GPU shared | 111.59 ms, including about 65 ms expert PCIe and 27 ms GPU kernels | different placement, packing and parallel decomposition |

Reference CUDA API totals over the retained two-rank window:

```text
cudaGraphLaunch:       162 calls, 5.61 ms average host duration
cudaStreamSynchronize: 162 calls, 79.23 ms average
cudaEventSynchronize:  408 calls; TP1 waits were short, TP0 had one async
                       completion waiter per step
```

One representative step begins its graph launch at 0.00 ms, returns from that
API at 5.61 ms, starts the TP0 completion event wait at 9.85 ms and the worker's
final stream wait at 15.66 ms.  Both waits end near 95.8 ms; the next launch
starts at 100.93 ms.  These are overlapping waits, not additive phase costs.

## CPU-MoE placement and packing

The installed Python integration initializes all 256 experts in both ranks but
sets `intermediate_size_per_partition` and `num_processes=tp_size=2`.  Each rank
therefore owns one intermediate-dimension half of every expert, about 74 GiB,
rather than assigning whole experts to devices.  `lk_moe` transforms that half
into a private NUMA arena and releases the loader tensors.

Live `numastat` after all 43 layers were initialized:

| process | total | private node 0 | private node 1 | placement |
|---|---:|---:|---:|---|
| TP0 | 77,931.7 MiB | 4,187.9 MiB | 72,551.3 MiB | expert half predominantly node 1 |
| TP1 | 77,954.6 MiB | 71,158.2 MiB | 5,603.8 MiB | expert half predominantly node 0 |

This is deliberate exclusive placement: both memory controllers stream one
half of every selected expert concurrently, even though both 3090s are attached
to NUMA node 1.  The GPU pointers cross the inter-socket link only for small
activations and partial outputs, not for expert weights.

Affinity inspection confirms persistent binding rather than incidental first
touch.  TP0 has 24 worker threads pinned one per logical CPU across node 1 and
additional worker/control threads allowed on the full node-1 mask.  TP1 has the
mirrored per-CPU set on node 0.  Unrelated Python/CUDA support threads remain
allowed on all CPUs.

The AVX2 loop near `0xcb538` confirms the transformed layout.  It broadcasts
two BF16 input columns, loads 16 packed bytes spanning 16 output rows, expands
low and high E2M1 nibbles with vector permutes, multiplies contiguous per-row
scales, and accumulates two vectors of output rows.  It does not horizontally
reduce an output dot product.  The consequences are:

- one activation broadcast feeds 16 output accumulators;
- output accumulators visit input columns in order;
- packed weights and scales are K-major / output-tiled in the NUMA arena;
- persistent workers execute coarse K/output tiles with work stealing;
- gate and up are stored together as `w13`;
- the two TP ranks reduce partial down-projection results on the GPUs.

At 3.449 GB of routed-expert bytes per token and 93.9 ms for the entire graph,
the reference sustains an aggregate **lower bound** of 36.7 GB/s over the two
expert arenas, at least 18.35 GB/s per socket/rank.  The true CPU-MoE bandwidth
is higher because attention, mHC, collectives and scheduling occupy part of the
same 93.9 ms.  Strata's rejected static host path measured only 11.7--13.1 GB/s
aggregate because it retained the row-dot/reduction kernel and whole-expert
scheduling; it did not reproduce this reference mechanism.

## Living delta table

| Rank | Reference behavior | Strata behavior | Evidence | Estimated wall impact | Next verification |
|---:|---|---|---|---:|---|
| 1 | TP2 CPU MoE: every expert split across opposite-socket transformed arenas; 24 persistent output-vector workers/rank; 192/768/128 contiguous jobs; workers remain hot and steal unfinished jobs only within their NUMA node | implementation arm now reproduces the layout, initial job shape, 1 ms hot-worker interval and same-node stealing for the measured gate/up and down tails | full-store probe reaches 39.25 GB/s and matches a real `lk_moe` expert; retained runtime median is 106.87 ms routed CPU and 244.81 ms total, with a best routed arm of 89.23 ms | 1 ms removes sleep/wake cost; stealing collapses gate/up and down completion tails and saves 17.47 ms routed CPU and 25.16 ms total at the interleaved median | CPU work is now close enough to the 87.88 ms probe to re-price the full serialized hidden/attention/mHC path; do not add another CPU mechanism without a new mismatch |
| 2 | hidden, fixed buffers and KV remain on GPU across one full decode graph replay | host FP32 hidden; about 407 blocking matmuls and 26.4 MiB activation round trips/step | Nsight API/graph/memcpy tables and Strata phase counters | the largest part of attention's 83 ms and mHC's 23 ms; partial device-chain experiments alone were neutral | after static CPU experts make fixed graph inputs possible, capture the smallest complete dependent chain rather than another partial projection chain |
| 3 | persistent paged FP8 DS-MLA KV; fused insert and sparse FlashMLA reads it in place | gather selected host rows and upload 18.07 MiB/step to one-call reference-numeric attention | source plus 6.33 ms/step FlashAttention operation counter | at least 6.3 ms direct, plus avoided host gather and synchronization | device-resident KV fixture with real block and sparse-index sequence |
| 4 | mHC post+next pre+RMSNorm is one TileLang GPU kernel | separate host pre, norm and post with BF16 round trips | model source; Strata 23.22 ms/step | up to about 20 ms, but earlier isolated device chain regressed | use node timing and layer-hash bisect before revisiting |
| 5 | fused/parallel attention projections on four streams | five serial projection calls and host transforms | attention source; Strata attention 83.36 ms | overlaps projection kernels and removes boundaries; not independently additive | retain as part of the whole device-resident chain, not another isolated projection experiment |

## Gap model at experiment start

```text
Current Strata:                    222.6 ms/step
Reference control:                 97.6 ms/step
Observed gap:                     125.0 ms

Measured Strata terms exposed by reference mechanisms:
  expert PCIe critical wait:       about 65 ms
  MoE GPU wall:                    about 27 ms
  attention host/device chain:      83.4 ms total
  mHC host chain:                   23.2 ms total

These terms overlap and cannot be summed as savings.  The reference places the
complete dependent layer pipeline inside a 93.9 ms graph span, so the binding
target is that span, not a sum of CUDA activity.
```

The unavailable node-only split prevents claiming an exact isolated CPU-MoE
duration, but it does not weaken the mechanism gate: the complete 93.9 ms graph
must read 3.449 GB of routed weights, so 36.7 GB/s aggregate is a conservative
floor.  The cheapest faithful next probe is therefore one production-shape
MXFP4 expert transformed into the reference's K-major/output-16 layout and
split into two NUMA-local intermediate shards.  It must clear 36.7 GB/s
aggregate before any 147 GiB runtime arena is built.  If it does not, the
CPU-MoE runtime plan is falsified at the mechanism level.

## Output-tiled mechanism gate

Disassembly call-site tracing refined the layout to 32-output blocks (the
public `MOEConfigV2.stride`), with a 16-row AVX2 inner tile.  Each checkpoint
group-32 scale is duplicated for the two group-16 input loops.  A two-shard,
100-expert arena reproducing 43 layers x top-6 measured:

```text
command: ./build/strata-dsv4-host-expert-probe \
           --output-tiled --layers 43 --iterations 7 --threads 56
median:                 75.57 ms/token
canonical bytes:         3.449 GB/token
canonical throughput:   45.64 GB/s
layout throughput:      48.33 GB/s
last gate/up:            43.75 ms
last activation:          6.66 ms
last down:               23.40 ms
last combine:             1.19 ms
```

This clears the predeclared 36.7 GB/s gate by 24.4%.  Values do not affect the
matrix loop's access pattern; nevertheless, layout correctness was checked on
the real layer-0/expert-0 checkpoint tensors.  With a BF16 sine input, two
Strata shards and two `MOE_MXFP4.cpu_prefill` shards produced:

```text
                         Strata              lk_moe
output[0]              -1.133799791        -1.133800030
weighted output sum  -599.1094088        -599.1094690
```

The checksum gap is 6.0e-5 on magnitude 599.1.  This check covers w1/w3
concatenation, intermediate TP slicing, w2 column slicing, scale duplication,
the output-block transpose and every output row.

One material semantic difference was also measured rather than hidden.  For
`activation_type=0`, the installed closed decode kernel reads BF16 input and
keeps gate/up/SwiGLU in FP32; controlled large-value inputs show that it ignores
the configured clamp and intermediate E4M3 quantizer.  Reintroducing Strata's
device-exact clamp and scalar E4M3 path raised the isolated CPU block to
114--130 ms/token, below the mechanism gate.  The new runtime arm is therefore
explicitly reported as `reference_tiled_cpu_moe_autoregressive`, not as Strata's
`exact_base_autoregressive` mode.  It must pass full-model output comparison
before it can be retained.

The staged representation is 155,826,782,208 bytes (145.125 GiB): it replaces
the 147,169,738,752-byte canonical routed arena and adds only 8,657,043,456
bytes for duplicated scales. Admission accounts for that delta.

## Full-store implementation and system falsification

The current implementation then matched the wheel's controller-level batch-1
shape: 48 persistent workers, 24 per socket; contiguous 32-output ranges; four
workers initially assigned to each selected expert; 192 fused gate/up+SwiGLU
jobs, 768 down jobs and 128 reduction jobs per shard; and process-level scratch.
The retained commands were:

```bash
make check

FULL_STORE_PROBE=1 ITERATIONS=7 THREADS=56 \
  scripts/run_deepseek_v4_tiled_store_probe.sh

FULL_STORE_PROBE=1 ITERATIONS=7 THREADS=48 \
  scripts/run_deepseek_v4_tiled_store_probe.sh

RESULT_DIR=results/deepseek-v4-tiled-schedule-full1 \
MAX_NEW_TOKENS=64 PROMPT_SENTENCES=39 \
  scripts/run_deepseek_v4_tiled_host_moe.sh
```

The full-store A/B, with one 145.125 GiB stage and seven measured route
sequences per arm, was:

| Term | Before | Reference-shaped schedule |
|---|---:|---:|
| median routed token | 231.510 ms | **87.879 ms** |
| useful aggregate bandwidth | 14.899 GB/s | **39.250 GB/s** |
| useful bandwidth/socket | 7.450 GB/s | **19.625 GB/s** |
| gate/up + activation | 141.589 ms | **53.514 ms** |
| down | 70.037 ms | **27.661 ms** |
| reduction/combine | 1.029 ms | 5.635 ms |
| stage time | 75.996 s | 74.621 s |

Live `numastat` during the candidate stage reported 62,474 MiB private on node
0 and 62,809 MiB on node 1. Maximum RSS was 153,345,140 KiB; major faults and
swaps were zero. The real expert oracle remained `-1.133799791` with weighted
checksum `-599.1094088`. Three separate post-change `make check` executions
passed in 143.36, 151.62 and 116.45 seconds.

The mechanism win did not survive the system gate. At 586 prompt tokens and 63
decode steps, the candidate measured:

```text
total:                       308.340 ms/step
MoE:                         165.532 ms/step
attention:                    99.082 ms/step
mHC pre + post:               35.011 ms/step
branch norm:                   2.523 ms/step
RSS:                     157,787,598,848 bytes
VRAM:               15.55 / 23.80 / 23.81 GB
decode checkpoint reads:               0 bytes
```

The preceding comparable single arm was 288.221 ms/step. These arms were not
interleaved, so they do not establish a regression distribution, but they
falsify the claim that the 2.63x probe win is an end-to-end win.

A 21-prompt-token, 15-step diagnostic then split the MoE join:

| Runtime term | ms/step |
|---|---:|
| routed gate/up + SwiGLU | 110.002 |
| routed down | 56.523 |
| routed reduction | 8.854 |
| **routed CPU total** | **175.399** |
| GPU shared collect/wait | 2.658 |
| final combine | 1.058 |
| graph MoE including route/prepare | 186.027 |

Both matrix phases, whose bytes and loop counts do not depend on activation
values, slow by essentially the same factor relative to the full-store probe:
2.06x for gate/up and 2.04x for down. The GPU shared expert is not the hidden
term, and SwiGLU values cannot alone explain the gap.

Decision: retain the implementation only as an experimental arm, not as an
accepted throughput result. The proven isolated improvement is necessary but
insufficient. The binding next measurement is the in-runtime CPU execution
environment—worker occupancy, effective clock, memory contention and TLB/page
behavior—because routed CPU is still `argmax`. Do not proceed to the next
reference mechanism or quote 87.879 ms as runtime MoE until the 2.05x gap is
explained.

## Expert worker sleep/wake arm

Source inspection found a direct reference mismatch consistent with the equal
matrix slowdown: lk_moe workers spin across ordinary decode gaps, whereas every
Strata worker entered a condition-variable wait as soon as its phase closure
returned. The immediate-sleep diagnostic recorded 478,898 voluntary context
switches in 15 decode steps.

The bounded implementation added a duration-valued idle-spin parameter to the
existing worker pool and enabled 10 ms only for the 48-worker tiled DeepSeek
expert pool. Attention and every other pool retained immediate sleep. A small
worker-pool test repeatedly dispatches while the workers are spinning, covering
the dispatch-generation/wait race. Commands:

```bash
make check

RESULT_DIR=results/deepseek-v4-tiled-idle-spin \
MAX_NEW_TOKENS=16 PROMPT_SENTENCES=1 \
  scripts/run_deepseek_v4_tiled_host_moe.sh
```

`make check` passed in 117.85 seconds. The one-arm production-context result,
compared with the identical phase-split diagnostic, was:

| Term | Immediate sleep | 10 ms expert spin | Change |
|---|---:|---:|---:|
| gate/up + SwiGLU | 110.002 ms | 67.443 ms | -38.7% |
| down | 56.523 ms | 33.419 ms | -40.9% |
| reduction | 8.854 ms | 3.729 ms | -57.9% |
| **routed CPU** | **175.399 ms** | **104.608 ms** | **-40.4%** |
| useful aggregate bandwidth | 19.66 GB/s | 32.97 GB/s | +67.7% |
| useful bandwidth/socket | 9.83 GB/s | 16.49 GB/s | +67.7% |
| shared collect | 2.658 ms | 2.088 ms | -21.4% |
| attention | 81.030 ms | 101.999 ms | +25.9% |
| mHC pre + post | 40.989 ms | 75.912 ms | +85.2% |
| total | 317.595 ms | 296.245 ms | -6.7% |
| voluntary context switches | 478,898 | 321,030 | -33.0% |
| involuntary context switches | 1,207 | 91,881 | +7,513% |

RSS was 157,748,445,184 bytes, VRAM was
15,538,913,280/23,795,466,240/23,791,271,936 bytes, and decode checkpoint
reads and swaps were zero.

Conclusion: immediate sleep/wake is proven to account for 70.8 ms of the
runtime routed-CPU term, and the spin arm closes most of the 2.05x probe gap.
The 10 ms calibration is not retained as a system win yet. It keeps 48 workers
runnable across Strata's CPU-side attention and mHC phases, raising those terms
by 55.9 ms and producing 91,881 involuntary switches. The next arm changes only
the calibration window; the gate is lower total time while preserving the MoE
gain. No new storage, kernel, or scheduling mechanism is justified until that
overlap tradeoff is measured.

### 1 ms calibration result

The 1 ms arm was built and checked before the session pause:

```bash
make check

RESULT_DIR=results/deepseek-v4-tiled-idle-spin-1ms \
MAX_NEW_TOKENS=16 PROMPT_SENTENCES=1 \
  scripts/run_deepseek_v4_tiled_host_moe.sh
```

`make check` passed in 113.39 seconds. The benchmark completed in 1:56.04 with
21 prompt tokens and 15 decode steps:

| Term | Immediate sleep | 10 ms | **1 ms** |
|---|---:|---:|---:|
| gate/up + SwiGLU | 110.002 ms | 67.443 ms | **81.055 ms** |
| down | 56.523 ms | 33.419 ms | **40.759 ms** |
| reduction | 8.854 ms | 3.729 ms | **5.226 ms** |
| **routed CPU** | **175.399 ms** | **104.608 ms** | **127.060 ms** |
| useful aggregate bandwidth | 19.66 GB/s | 32.97 GB/s | **27.15 GB/s** |
| useful bandwidth/socket | 9.83 GB/s | 16.49 GB/s | **13.58 GB/s** |
| complete graph MoE | 186.027 ms | 112.232 ms | **136.571 ms** |
| attention | 81.030 ms | 101.999 ms | **74.795 ms** |
| mHC pre + post | 40.989 ms | 75.912 ms | **58.163 ms** |
| branch norm | 2.683 ms | 1.513 ms | **2.049 ms** |
| **total** | **317.595 ms** | **296.245 ms** | **278.135 ms** |
| voluntary context switches | 478,898 | 321,030 | **377,974** |
| involuntary context switches | 1,207 | 91,881 | **5,026** |

The arm decoded at 3.595 token/s. RSS was 157,748,584,448 bytes, VRAM was
15,538,913,280/23,795,466,240/23,791,271,936 bytes, decode checkpoint reads
were zero, and swaps were zero.

Decision at pause: 1 ms is the best single short arm, improving total by 12.4%
against immediate sleep and 6.1% against 10 ms. It remains unretained until two
more identical arms establish a three-run median. If retained, routed CPU at
127.06 ms is still above both the 87.879 ms full-store result and the 98 ms
whole-step target, so the CPU path remains the first implementation target.

### Three-run 1 ms retention

The two prescribed repeats used the same 21-token prompt and 15 measured decode
steps:

```bash
RESULT_DIR=results/deepseek-v4-tiled-idle-spin-1ms-r2 \
MAX_NEW_TOKENS=16 PROMPT_SENTENCES=1 \
  scripts/run_deepseek_v4_tiled_host_moe.sh

RESULT_DIR=results/deepseek-v4-tiled-idle-spin-1ms-r3 \
MAX_NEW_TOKENS=16 PROMPT_SENTENCES=1 \
  scripts/run_deepseek_v4_tiled_host_moe.sh
```

Each ran in its own named tmux session.  The three-run distribution, including
the original arm, was:

| Term | Median | Min--max |
|---|---:|---:|
| total | **246.612 ms/step** | 242.645--278.135 |
| routed gate/up + SwiGLU | 68.364 ms | 66.656--81.055 |
| routed down | 33.579 ms | 32.609--40.759 |
| routed reduction | 4.466 ms | 3.806--5.226 |
| **routed CPU** | **105.763 ms** | 103.747--127.060 |
| shared collect | 2.393 ms | 1.991--2.707 |
| final combine | 0.906 ms | 0.625--1.134 |
| complete graph MoE | 112.570 ms | 111.881--136.571 |
| attention | 70.460 ms | 66.530--74.795 |
| mHC pre + post | 58.163 ms | 53.495--61.293 |
| branch norm | 1.558 ms | 1.328--2.049 |
| voluntary context switches | 377,974 | 338,073--410,573 |
| involuntary context switches | 4,041 | 2,909--5,026 |
| RSS | 157,748,584,448 bytes | 157,748,441,088--157,750,693,888 |

VRAM was exactly
15,538,913,280/23,795,466,240/23,791,271,936 bytes in all three runs.
Decode checkpoint reads and swaps were zero in every run.  All three produced
the same text.  The median improves total by 70.983 ms, 22.4%, against the
immediate-sleep arm, and even the slowest 1 ms run is 39.460 ms faster.
Decision: retain the 1 ms calibration.

### Per-lane completion measurement

The next arm added fixed, process-lifetime timestamp slots for all 48 expert
lanes.  It recorded lane start, work, finish and controller-return times for
gate/up and down without changing initial ownership.  No per-phase allocation
was added.  `make check` passed before the measurement.  The retained artifact
is `results/deepseek-v4-tiled-lane-tail`.

At 251.629 ms total and 110.568 ms routed CPU, the per-layer averages were:

| Phase/node | start skew | median lane work | slowest lane work | completion tail | return after last lane |
|---|---:|---:|---:|---:|---:|
| gate/up node 0 | 0.129 ms | 1.335 ms | 1.528 ms | **0.202 ms** | 0.033 ms |
| gate/up node 1 | 0.140 ms | 1.319 ms | 1.518 ms | **0.204 ms** | 0.031 ms |
| down node 0 | 0.061 ms | 0.643 ms | 0.735 ms | **0.100 ms** | 0.017 ms |
| down node 1 | 0.063 ms | 0.640 ms | 0.734 ms | **0.100 ms** | 0.016 ms |

The two sockets are symmetric.  Dispatch start and return cost are small, while
median-to-last completion tails account for about 13.1 ms/token across 43
layers.  This selects the reference's same-node stealing; it does not select a
new allocator, a wider spin interval, page changes or another kernel layout.

### Same-node gate/up and down stealing

The retained scheduler preserves each lane's contiguous initial interval.
Each job is claimed with an atomic cursor; after a lane drains its interval it
may claim unfinished jobs only from the other 23 lanes on the same NUMA node.
The final reduction keeps its original static contiguous loop because its tiny
jobs did not exhibit the measured matrix tail.  A post-dispatch check rejects
any unfinished initial interval.

One preliminary arm applied atomic claiming to reduction as well.  It measured
239.317 ms total and 103.779 ms routed CPU, but reduction rose from 3.910 to
5.577 ms.  That scope was rejected before the final A/B; only gate/up and down
retain stealing.

After narrowing the phase scope, `make check` passed in 108.81 seconds.  The
real layer-0/expert-0 oracle remained:

```text
Strata output[0]             -1.133799791
Strata weighted checksum   -599.1094088
lk_moe output[0]             -1.133800030
lk_moe weighted checksum   -599.1094690
```

The final short A/B used the saved pre-steal instrumented binary for baselines
and the final binary for candidates.  Runs were sequential and interleaved:

| Order | Arm | total | routed CPU | gate/up | down | reduction |
|---:|---|---:|---:|---:|---:|---:|
| 1 | candidate C1 | 272.383 | 126.366 | 77.343 | 40.846 | 8.156 |
| 2 | baseline B2 | 318.130 | 151.227 | 95.260 | 49.216 | 6.730 |
| 3 | candidate C2 | 224.523 | 89.226 | 56.869 | 28.015 | 4.319 |
| 4 | baseline B3 | 269.972 | 124.335 | 78.195 | 40.109 | 6.014 |
| 5 | candidate C3 | 244.812 | 106.869 | 68.021 | 34.654 | 4.175 |
| 6 | baseline B4 | 252.714 | 111.707 | 71.297 | 35.828 | 4.568 |

All values are ms/step.  The three adjacent candidate-to-following-baseline
pairs saved 45.747/24.861, 45.449/35.109 and 7.901/4.838 ms in total/routed
CPU respectively; every pair has the expected sign.

| Term | Baseline median (min--max) | Candidate median (min--max) | Median change |
|---|---:|---:|---:|
| total | 269.972 (252.714--318.130) | **244.812 (224.523--272.383)** | **-25.160 ms (-9.3%)** |
| routed gate/up | 78.195 (71.297--95.260) | **68.021 (56.869--77.343)** | -10.174 ms |
| routed down | 40.109 (35.828--49.216) | **34.654 (28.015--40.846)** | -5.455 ms |
| routed reduction | 6.014 (4.568--6.730) | 4.319 (4.175--8.156) | -1.695 ms |
| **routed CPU** | **124.335 (111.707--151.227)** | **106.869 (89.226--126.366)** | **-17.466 ms (-14.0%)** |
| shared collect | 2.862 (2.461--3.383) | 2.401 (2.259--2.725) | -0.461 ms |
| final combine | 0.971 (0.756--1.227) | 0.694 (0.689--0.794) | -0.277 ms |
| complete graph MoE | 133.785 (119.911--162.452) | 114.667 (96.593--134.926) | -19.118 ms |
| attention | 75.132 (71.805--80.358) | 71.140 (69.863--74.564) | -3.992 ms |
| mHC pre + post | 54.151 (53.405--65.545) | 53.061 (52.113--55.851) | -1.090 ms |
| branch norm | 1.869 (1.592--2.522) | 1.374 (1.335--1.792) | -0.495 ms |
| voluntary context switches | 355,922 (346,487--359,047) | 345,331 (340,533--370,380) | -10,591 |
| involuntary context switches | 2,839 (1,614--3,540) | 2,679 (2,571--3,643) | -160 |
| RSS | 157,748,621,312 (157,748,609,024--157,748,621,312) | 157,748,613,120 (157,748,568,064--157,750,919,168) | -8,192 bytes |

VRAM was exactly unchanged in all six arms.  Every arm generated the same text,
with zero decode checkpoint reads and zero swaps.  Candidate gate/up completion
tails were 0.052--0.123 ms/layer versus baseline 0.173--0.735; down tails were
0.009--0.049 versus 0.096--0.385.  These causal distributions do not overlap.
Decision: retain same-node stealing for gate/up and down.

### Re-instantiated whole-step model

At the retained median, the CPU path reads 3.449290752 GB of canonical routed
payload in 106.869 ms, or 32.276 GB/s aggregate and 16.138 GB/s/socket.  It is
18.990 ms, 1.216x, above the 87.879 ms full-store probe; the best arm is 89.226
ms and therefore demonstrates that the runtime can reach the probe regime.

The median whole step is now:

```text
complete graph MoE:  114.667 ms
attention:            71.140 ms
mHC pre + post:        53.061 ms
branch norm:            1.374 ms
remaining:              4.572 ms
total:                244.812 ms
```

These exposed phases are serialized by the current host-visible hidden chain;
they are not additive savings and cannot be replaced with a max until their
handoffs overlap.  The non-MoE serial path is 130.145 ms, already larger than
complete MoE.  Even perfecting routed CPU from its median to the standalone
probe removes only 18.990 ms and cannot approach 98 ms/step.  The next binding
reference mismatch is therefore the documented GPU-resident hidden,
attention/KV and fused mHC graph pipeline.  Another CPU scheduler mechanism is
not justified without a new measured mismatch.
