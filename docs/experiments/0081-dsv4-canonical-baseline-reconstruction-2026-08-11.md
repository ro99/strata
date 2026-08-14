# Experiment 0081: DSV4 canonical baseline reconstruction

Date: 2026-08-11
Branch: `exp/dsv4-baseline-reconstruction`
Base: `d6a93a6618e5cd58e0c575483d632efe246bae2d`
Binary: `build-stage5/strata-deepseek-run`
Binary SHA-256: `30a2a343675e63e3444a1643d6bd2c32eba6d139ed84eef18c90f5dd6b532ba9`

## Question and predeclared decision

The 634.398 ms one-forward Stage-6A control was a large outlier against the
accepted 152.263 ms/forward decode. This control-only experiment asked the
cheapest falsifying question first: is the outlier a cold first step, a trace
arm, or a current-source/configuration mismatch? It did not implement or run a
rank-local candidate.

The primary metric was complete 15-forward decode wall time at the declared
operating point. The resource gate was the exact routed CPU payload of
`3,449,290,752 B/forward` at at least `36.7 GB/s`; the historical accepted
body comparison is `84.915091 ms/forward`, or `40.620468 GB/s`. The hard
correctness gates were the exact generated token sequence, 645 decode MoE
batches, zero callback failures/non-finite values, zero decode checkpoint
reads/KV misses/promotions, and one final process completion. The memory gates
were `21,287,272,448 B/GPU` and `231,928,233,984 B` host RSS. Rollback was to
the clean base commit; any memory, exactness, I/O, or output failure rejects
the reconstruction regardless of timing.

Before the run, the completeness comparison tolerance was fixed at ±5% of the
accepted `152.263169533 ms/forward`, namely `144.650011–159.876328 ms`. The
single-process setup (model admission, 216 GiB host reservation, resident
staging and warmup) took about 100–105 s while the measured decode window was
about 2.23 s. The rejected cheaper alternatives were the already-rejected
one-step 0080 control (which could not establish a steady distribution) and a
full candidate launch before a valid control. Three fresh control processes
were budgeted at roughly 6–8 minutes total; no candidate or long full-model
matrix was authorized.

The cost model was instantiated as
`tau = max_r(W_r/B_r) + Sigma_serial`. At this operating point the CPU routed
body is the largest directly measured resource service term. CUDA synchronization
and shared-collect values are dependency-ordered spans that overlap that body;
they are not independently summed into `tau`.

## Operating point and provenance

The reusable wrapper is
`scripts/run_dsv4_baseline_reconstruction.sh`. Its timed command is:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 \
build-stage5/strata-deepseek-run \
  --model models/dsv4f --devices 0,1 --host-memory 216G \
  --vram-fraction .95 --max-context 256 --max-new 16 \
  --prompt results/dsv4-lk-moe-phase-profile/step2-prompt.txt \
  --device-resident-runtime --host-routed-moe \
  --host-attention-threads 28 --quiet --json
```

There is no speculation flag in this CLI; speculation is disabled by default.
The tracked CLI also makes `--host-routed-moe` true when
`--device-resident-runtime` is supplied; the fresh arms were launched before
the wrapper was made explicit, but their JSON and source path are identical to
the explicit command above. No logit, route, handoff, layer-hash, or profiler
trace was enabled in the performance controls. The effective JSON execution mode is
`host_routed_cpu_moe_autoregressive`, with the production transformed tiled
NUMA arenas and 48-worker/1 ms host executor.

The historical accepted 152.263169533 ms arm is preserved at
`results/dsv4-moe-handoff/`. Its `system-before.txt` records:

```text
commit 61f1f02e7bfe22c4f6bbd81e27aebbf9857c84be
branch exp/dsv4-device-resident-kv-handoff
dirty: apps/strata_deepseek_run.cpp, include/strata/cuda_backend.hpp,
       include/strata/deepseek_runtime.hpp, kernels/cuda/backend.cu,
       src/deepseek_runtime.cpp, plus the 0072/timeline scripts
CUDA_VISIBLE_DEVICES=1,2; runtime_devices=0,1; context=256; max_new=16;
vram_fraction=.95; prompt=step2-prompt.txt
binary SHA-256 4114764b33273c08e9a1bcfb56c0c2489c9674c5fff4a034ec4d940e2c969904
```

The preserved `candidate.diff` SHA-256 is
`a0e1791ea304e1c5b27600d220f43bd6c6403f60297e8e28b786d67fde0ed748`. The
current branch descends from 61f, but its HEAD includes later Stage-4/5 source
commits and is not the historical binary/source state. Reference integration
commits `233ed98` and `3ad6664` are not ancestors and were not merged. The
current executable is therefore a fresh current-source reproduction, not a
claim that the old dirty binary was rebuilt exactly.

## Instrumentation and first-step result

The default-off `STRATA_DSV4_DECODE_TIMELINE_PATH` hook was added to the
runtime. It reserves the 16-row vector, snapshots existing graph/MoE/CUDA and
checkpoint counters around each `forward_token`, and writes one CSV after the
decode loop. It does not write in the timed path or alter arithmetic. The
timeline-on arm was compared with a fresh default-off arm after the rebuild:

| arm | decode seconds | ms/forward | generated IDs | timeline |
|---|---:|---:|---|---|
| timeline on (`control-timeline`) | 2.232544651 | 148.836310 | exact 16/16 | on |
| timeline off (`control-off2`) | 2.231819096 | 148.787940 | exact 16/16 | off |

The paired difference is `0.048370 ms/forward` (`0.0325%`), well below the
predeclared 5% tolerance. The timeline has all positions 104–118. Step 1 is
`148.353396 ms`; the full row range is `147.006223–150.903648 ms`, so there is
no distinct cold first-step penalty. The rows are retained in
`results/dsv4-baseline-reconstruction/control-timeline/decode-timeline.csv`.

Important span interpretation: a typical row reports approximately 82 ms CPU
body, 126 ms shared-collect/synchronization span, and 148 ms wall. The 208 ms
MoE `body + collect` field is an overlapping diagnostic envelope, not an
additive timing result. The CSV is evidence for boundaries and distribution,
not permission to sum overlapping resource spans.

## Fresh control matrix

Three fresh processes were run sequentially after the timeline perturbation
check, with no timeline and no trace flags. Every process generated exactly:

```text
[43, 8806, 440, 5270, 4496, 1205, 9238, 304, 366, 260, 3418, 294,
 6719, 8454, 305, 3345]
```

| repetition | complete decode ms/forward | routed CPU body ms/forward | routed GB/s |
|---:|---:|---:|---:|
| control-r1 | 149.099058 | 84.171250 | 40.979441 |
| control-r2 | 152.113645 | 85.966045 | 40.123874 |
| control-r3 | 148.433299 | 82.051309 | 42.038217 |
| median | **149.099058** | **84.171250** | **40.979441** |
| range | 148.433299–152.113645 | 82.051309–85.966045 | 40.123874–42.038217 |

The complete median is within the predeclared accepted-baseline tolerance and
the CPU body clears the hard throughput floor. It is not called a throughput
win over the historical arm: the source/binary state and trace scope differ,
and the result is rejected below on the unchanged VRAM gate.

The median phase/resource terms from the three fresh controls were:

| term | median ms/forward | observed range | interpretation/sign |
|---|---:|---:|---|
| routed gate/up | 53.452145 | 51.877541–54.664018 | CPU body, positive load |
| routed down | 25.880785 | 25.051920–26.174004 | CPU body, positive load |
| routed reduction | 3.917453 | 3.785166–4.024120 | CPU body, positive load |
| routed CPU body | **84.171250** | 82.051309–85.966045 | current `max_r W/B` term |
| shared-collect span | 126.341356 | 125.972012–129.101491 | overlaps body; not additive |
| CUDA synchronization span | 126.288853 | 125.925466–129.062730 | final/dependency envelope |
| CUDA kernel service | 1.955437 | 1.921197–2.120348 | positive GPU load, overlapped |
| attention graph counter | 15.775734 | 15.507746–15.939848 | positive host/GPU work |
| mHC pre | 0.112226 | 0.106316–0.116431 | positive work |
| mHC post | 3.354179 | 3.322705–3.447867 | positive work |
| complete wall | **149.099058** | 148.433299–152.113645 | measured makespan |

Per forward, application activation transfers were exactly `2,370,221 B H2D`
and `1,693,016 B D2H` (decode totals `35,553,312 B` and `25,395,240 B`).
The observed CUDA copy spans were approximately `0.001632–0.001690 ms` H2D
and `0.002769–0.003488 ms` D2H per forward. Decode checkpoint calls/bytes,
KV misses/promotions, weight allocations, workspace allocations, and callback
failures were all zero. The 645/645 decode batches and 3,870 routed expert
associations were present in every run.

## Cost model at the reconstructed point

Using the three-run median:

```text
W_CPU = 3,449,290,752 B/forward
B_CPU = W_CPU / 0.084171250 s = 40.979441 GB/s
max_r(W_r/B_r) = 84.171250 ms (CPU routed body)
measured wall = 149.099058 ms
Sigma_serial envelope = 149.099058 - 84.171250 = 64.927808 ms
tau = 84.171250 + 64.927808 = 149.099058 ms/forward
```

The residual is a makespan residual, not a sum of all rows. CUDA
synchronization/collect spans are larger than CPU body but overlap it; treating
126.3 ms as an additional resource term would double count. H2D/D2H volume and
kernel service are positive, non-binding loads. Host callback scheduling,
attention, mHC, output/head work, and final completion contribute positive
serial/overlap terms; none is claimed free. The 0080 634 ms body is therefore
falsified as a steady-state CPU operating point, while the complete 10 tok/s
goal is still not met (`149.099058 > 100 ms`).

## Memory audit and gate result

The process monitor recorded only the intended runtime PID on the two RTX
3090 UUIDs (`GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` and
`GPU-81fe4578-59b2-37c4-421e-287cdac78704`); no unrelated compute process was
subtracted. Peak samples were 22,678 MiB on each GPU and returned to 264 MiB
after exit. All three fresh controls reported the same runtime values:

```text
GPU/runtime device 0: 23,787,077,632 B
GPU/runtime device 1: 23,789,174,784 B
declared ceiling:    21,287,272,448 B per GPU
excess:               2,499,805,184 B / 2,501,902,336 B
host RSS:            157,732,253,696–157,732,421,632 B
host ceiling:        231,928,233,984 B
```

The generation memory plan was `47,514,353,664 B` aggregate. Its accounting
was `9,204,991,520 B` resident spine plus `37,765,254,304 B` expert-cache
portion, `536,870,912 B` workspace reserve, and `7,236,928 B` physical KV.
The resident spine and expert cache are portions of the single allocated
weight arena; they must not be added twice. The measured CUDA weight-arena
allocation was `46,970,245,536 B`, plus the physical KV and fixed/dynamic
runtime allocations. The observed aggregate GPU use was `47,576,252,416 B`,
`61,898,752 B` above that plan (driver/runtime fixed buffers and allocator
overhead are not fully represented by the plan). This is a genuine residency
overage, not an unrelated-process or cache-capacity subtraction.

The smallest obvious way to fit the unchanged residency is to lower the
per-device VRAM admission fraction/capacity by about 2.5 GB per GPU (roughly
an `.85`-class fraction on this machine), but that changes the declared `.95`
operating point and may change cache residency. No such correction was
silently applied or measured here. The current planner admits the `.95` arena
even though the program-level 21,287,272,448 B/GPU ceiling is lower; reconciling
that planner contract requires explicit human review and a new bounded control
arm. Thus the timing/CPU gates pass, but the hard memory gate rejects this
experiment as a canonical accepted baseline.

## Decision and handoff

**Timing reproduction: PASS. Memory-gated control: REJECTED / REVIEW REQUIRED.**

This experiment classifies the 634.398 ms Stage-6A control as a cold/trace/path
or source-scope mismatch, not evidence against rank-local topology: the fresh
current control is 149.099 ms median, the first timed row is ordinary, and the
routed CPU rate is 40.98 GB/s. It also does not prove a rank-local win or
authorize Stage 6. The historical 152.263 arm itself reported the same GPU
residency, so its timing remains an accepted historical comparison but not a
solution to the declared ceiling mismatch.

Stage 6 and all later stages remain blocked. The next legitimate action is
human review of whether to (a) revise/repair the admission contract with an
explicit semantics-preserving cache-capacity control arm, or (b) reject the
current `.95` operating point as memory-invalid and stop. A candidate launch,
reference-branch merge, source reconstruction from performance alone, or
Stage-6 continuation is not authorized by this result.

## Artifacts and hashes

All large artifacts remain ignored under
`results/dsv4-baseline-reconstruction/`:

```text
control-one/generation.json       8cae2bd55658e50157d301ddc6d10fee0966dd609e572e8f37dbfaae707ec1fe
control-timeline/generation.json  e93e3365dd7c2515b5af4f370870757cb730594c0a2f572208695e0d5fa6365b
control-timeline/decode-timeline.csv
  a04a1c1d7ec02c32fe25e7c12bed2758cafcd0e0ad9ea63c335a4e743f8f7d07
control-off2/generation.json      b3f87fb3ebaac54c74ca10e305dc1ab1199953499ddd9fee7aa31a2b2f20c4ac
control-r1/generation.json        54463d41d6494a76e0b94aa8e9cc89cdadd4ff44675f79c23083863d665f7cc8
control-r2/generation.json        c7d301f63df5f43d16fb11199ed303fe310da3bfcf2b9d9b0ffc05d3e6a60c8d
control-r3/generation.json        92b2d3c0b08a18eb144da27a0b2177dd112ddbce1b1f25240a3f99327fe9a1f2
```

The three corresponding `summary.json` hashes are respectively
`2b8ee7e1096e6d0f01b033eedb8c5211c9d026cf4057d06026ba0c8aab96a62e`,
`007237f187d88aa8820c5889765bfcad4518faa8560c4fb83c775146864d23d2`, and
`332f4dabe0630755a7eee03b0d36aa681e95e280d052f3a19727fa821c5a91b2`.
The prompt hash is
`082d43114147a15eb345cc7eb2bba99a920299cacacfe5c45823bffb729c15fe`.

No Stage-6 candidate artifacts were created. The result commit is the single
Git commit containing this experiment, the default-off timeline hook, and the
reusable control wrapper; its final hash is reported by the handoff status.
