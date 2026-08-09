# Experiment 0065: DeepSeek reference fixed-buffer device-envelope gate

## Status

**Accepted as the cheapest prerequisite for the complete device chain.** On
both RTX 3090s, one captured 43-layer graph containing every stream-ordered
CPU-MoE activation/route boundary, host callback, routed-output return, GPU
shared/routed join, and next-layer consumption costs 2.318/2.182 ms median.
Added conservatively to the measured 87.879 ms standalone routed-CPU floor,
the resulting 90.197/90.061 ms floors remain below the 100 ms/token target.

This is not an end-to-end throughput result. The callback body deliberately
does no expert arithmetic, and the auxiliary GPU work is a dependency-correct
placeholder. The result accepts fixed addresses, CUDA-graph host nodes, the
explicit transfer envelope, and next-layer device ownership. It authorizes
integration of the already accepted mHC and attention/KV arithmetic; it does
not claim that their non-overlapped GPU work fits the remaining 9.803/9.939 ms.

## Hypothesis and current operating point

The bounded whole experiment remains: fixed hidden/residual buffers,
persistent paged FP8 DS-MLA KV, in-place sparse attention, fused mHC,
stream-ordered CPU-MoE activation/output, GPU shared/routed reduction, and
next-layer device consumption reduce Strata's serialized 130.145 ms
non-MoE/handoff path. An isolated projection or kernel migration is not the
experiment.

The retained operating point is:

```text
tau = max_r W_r / B_r + sum_serial

complete MoE                         114.667 ms
routed CPU median / best       106.869 / 89.226 ms
standalone routed CPU                  87.879 ms
serialized non-MoE/handoff            130.145 ms
total                                 244.812 ms/token
target                              <=100.000 ms/token
```

- **Target term and bottleneck.** The mechanism targets the 130.145 ms serial
  hidden/attention/mHC/ownership term. No new routed-CPU mismatch exists.
- **Primary metric for this cheapest gate.** Median full 43-layer envelope
  graph span, with `87.879 ms + envelope <= 100 ms`.
- **Correctness gate.** Every callback observes the preceding fixed device
  hidden and exact six route IDs/weights; its output is consumed by the next
  layer; all callbacks execute once per graph; SM86 is mandatory and exact
  mode has no fallback.
- **Memory ceiling.** Host <=216 GiB, explicit 0.95 VRAM admission, the accepted
  151,228,416-byte full-context KV/state allocation split by layer placement,
  and zero model/checkpoint dependency in the envelope probe.
- **Resource signs.** Routed CPU payload remains 3.449290752 GB/token. The
  explicit boundary moves 354,320 D2H bytes and 704,512 H2D bytes per step.
  HBM gains the admitted persistent KV/state plus fixed hidden/output buffers;
  GPU gains the join and eventual accepted component work. No extra route,
  recompute, precision change, host KV tier, or steady-state NVMe work is
  allowed.
- **Rollback.** A projected floor above 100 ms, callback/next-consumer failure,
  non-SM86 fallback, or VRAM admission failure rejects the architecture before
  live component wiring.

## Why this was the cheapest faithful gate

The external `cpu_decode` boundary is five ordered stream operations per
layer: D2H BF16 hidden, D2H int32 IDs, D2H FP32 weights, one CUDA host function,
and H2D FP32 routed output. The reference captures those nodes inside its one
approximately 93.9 ms graph. Strata had never established whether 43 native
CUDA host callbacks plus their transfers and GPU joins fit the 12.121 ms margin
above the standalone CPU floor.

Counting CUDA calls could not answer that because host-node scheduling is a
runtime cost. Loading the 155+ GB runtime before proving this envelope would
spend roughly 75 seconds staging and additional prefill time to test a
mechanism measurable in milliseconds. The probe therefore allocates fixed
process-lifetime buffers, reserves the accepted rank-local KV/state bytes,
captures the complete boundary topology once, performs three warm-ups, then
runs eleven interleaved envelope/control repetitions. Expected wall time was a
few seconds per device.

The control graph retains the same 43 auxiliary-stream fan-outs, GPU joins,
and next-layer consumption but synthesizes routed output on the device. The
envelope graph replaces that synthesis with the real D2H/host-node/H2D
boundary. Neither graph contains attention, mHC, projection, shared-expert, or
routed-expert arithmetic, so their timing cannot be presented as a complete
model estimate.

## Results

| Metric | RTX 3090 device 0 / slot 0 | RTX 3090 device 1 / slot 1 |
|---|---:|---:|
| envelope median (min--max) | **2.318336** (2.255872--2.369376) ms | **2.182144** (2.082816--2.248608) ms |
| control median (min--max) | 0.108544 (0.107520--0.118784) ms | 0.108544 (0.106496--0.108544) ms |
| median envelope minus control | 2.209792 ms | 2.073600 ms |
| CPU floor + full envelope | **90.197334 ms** | **90.061142 ms** |
| margin to 100 ms | **9.802666 ms** | **9.938858 ms** |
| callbacks, warm-up plus measured | 602 | 602 |
| callback failures | 0 | 0 |
| KV/state reservation | 36,694,016 B | 73,574,400 B |
| D2H / H2D per graph | 354,320 / 704,512 B | 354,320 / 704,512 B |
| maximum RSS | 102,556 KiB | 103,308 KiB |
| swaps | 0 | 0 |

All eleven samples are on the same side of the gate; this is not a variance
classification. Device 0 reported 16 process-wide major faults and 5,560
filesystem input blocks during CUDA process initialization, while device 1
reported zero. The probe opens no checkpoint, model shard, tensor fixture, or
result file itself; shell `tee` owns the ignored JSON output. These process-wide
counters are therefore initialization accounting, not evidence of a decode
NVMe dependency. Decode has no file-backed input by construction.

Admission took 0.471905/0.420538 seconds, allocation/graph load
0.005245/0.005312 seconds, and warm-up 0.007659/0.007648 seconds. Prefill is not
applicable. The measured window consists only of eleven alternating
envelope/control graph pairs.

## Decision

The fixed-buffer graph/host-function envelope passes and is retained as a
prerequisite. It clears a concrete architecture risk in checklist items 2, 3,
and 7:

- the explicit CPU-MoE boundary itself fits the whole-step ceiling;
- CUDA graph capture preserves the 43 callback dependencies;
- fixed device hidden ownership survives each host callback and is consumed by
  the next layer;
- the accepted full-context FP8 KV/state reservation fits the declared slots.

The next gate is live integration of the accepted fused mHC and real
compressor/paged-KV/selection/attention contracts into this ownership envelope.
The complete GPU lane must either overlap the CPU work or leave its unavoidable
barriers inside the measured 100 ms step. No speedup, full-model correctness,
or green checklist status is claimed yet.

## Reproduction

```bash
build/strata-dsv4-device-envelope-probe \
  --device 0 --slot 0 --repetitions 11 --warmups 3

build/strata-dsv4-device-envelope-probe \
  --device 1 --slot 1 --repetitions 11 --warmups 3
```

Raw JSON is ignored under
`results/dsv4-reference-complete-device-chain/`.
