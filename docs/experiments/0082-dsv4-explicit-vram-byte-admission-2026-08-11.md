# Experiment 0082 — explicit DSV4 VRAM byte admission

Date: 2026-08-11
Branch: `fix/dsv4-vram-byte-admission`
Base: `b019de73d9c30c1f560f6f73869d27a367c358dc` (`exp/dsv4-baseline-reconstruction`)
Result: **PASS FOR THE MEMORY-CONTRACT HYPOTHESIS / REVIEW REQUIRED**

## Question and predeclared decision

0081 reproduced the canonical current-source timing and CPU throughput, but its
`.95` admission plan reserved about 2.5 GB/GPU above the unchanged measured
ceiling. This bounded experiment asked whether an explicit per-device byte
ceiling could cap the total admission plan before arena allocation, without
changing the fractional `.95` command, populated working set, exact decode,
or steady-state performance.

The hypothesis was:

```text
applied_budget = min(cudaMemGetInfo().free_bytes * 0.95,
                     21,256,421,376 B per device)
```

The expected reduction was `2,500,755,456 B/GPU` from the 0081 post-context
fractional plan. The primary metric was the applied/observed per-GPU budget and
the zero-miss exact decode at the same operating point. The correctness gate
was the exact 104-token/15-forward output, 645/645 decode MoE batches, no
callback failure/non-finite value, and fail-closed parsing/admission. The
memory gates were `21,287,272,448 B/GPU` and
`231,928,233,984 B` host RSS. Decode checkpoint reads, KV misses/promotions,
cache misses/evictions, demand H2D, and timed allocations had to remain zero.
Rollback is the parent `b019de7` behavior; any memory, exactness, I/O,
allocation, or material timing regression rejects the correction.

The setup/window ratio was deliberately accepted because this is a resident
model control: each fresh process took 1:42.58–1:48.39 to load, while the
measured window was 2.2024–2.2728 s for 15 forwards. The first cheap arm was
budgeted at about 2 minutes; the three fresh-process matrix was budgeted at
about 6 minutes total. The rejected cheaper alternative was lowering
`--vram-fraction` or silently changing the workload; neither preserves the
declared `.95` operating point or identifies the planner defect.

Resource signs were predeclared. The target is capacity/admission, not a
latency resource. CPU routed `W/B`, GPU kernel/shared service, attention, mHC,
host issue/callback, PCIe activation transfers, and `Sigma_serial` should be
neutral; a cache-pressure increase would be positive (rejecting), while a
smaller reserved arena is beneficial only to the VRAM-capacity feasibility
term. The measured bottleneck remains CPU routed work.

## Implementation

The runtime now accepts optional `--vram-budget-bytes BYTES`, interpreted as a
positive per-selected-device byte ceiling. The absent value is zero internally
and preserves the old fractional-only path. Parsing rejects zero, malformed
values, and decimal/suffix overflow before model admission. The planner keeps
the historical double fractional product, computes both bounds, applies the
stricter bound before `reserve_weight_arena`, then subtracts the fixed
workspace, KV, and device mHC reservations to derive the arena/cache
capacity. It does not shave an unmeasured amount after allocation.

The JSON `memory_plan` reports, per device:

```text
fractional_vram_budget_bytes
explicit_vram_budget_bytes
applied_vram_budget_bytes
vram_budget_bound = fractional | explicit | equal
```

The same fields are present in `--admission-only --json`. A pure runtime
budget-selection test covers the stricter bound, absent explicit cap, and tie;
CLI probes rejected zero and `18446744073709551616` with status 2. The focused
admission-only probe produced valid JSON with the explicit bound on both
devices. No precision, router, expert, context, cache policy, arithmetic,
workload, or lazy-allocation behavior changed.

Tracked implementation files are:

```text
apps/strata_deepseek_run.cpp
include/strata/deepseek_admission.hpp
include/strata/deepseek_runtime.hpp
include/strata/runtime_support.hpp
src/deepseek_runtime.cpp
src/runtime_support.cpp
tests/test_runtime.cpp
scripts/run_dsv4_vram_byte_admission.sh
```

The reusable script preserves `.95` and passes the explicit decimal byte
budget. It records the command, Git state, binary/prompt hashes, process PID,
before/during/after GPU accounting, `/usr/bin/time -v`, JSON, validation, and
SHA-256 files under ignored `results/dsv4-vram-byte-admission/`.

## Operating point and gate result

The timed command was:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 \
build-stage5/strata-deepseek-run \
  --model models/dsv4f --devices 0,1 --host-memory 216G \
  --vram-fraction 0.95 --vram-budget-bytes 21256421376 \
  --max-context 256 --max-new 16 \
  --prompt results/dsv4-lk-moe-phase-profile/step2-prompt.txt \
  --device-resident-runtime --host-routed-moe \
  --host-attention-threads 28 --quiet --json
```

This is the same corrected 104-token prompt and two RTX 3090 operating point
used by 0081: 15 timed decode forwards, batch one, no speculation, no trace,
216 GiB host limit, and zero decode checkpoint/KV I/O. `control-r1` was the
cheap gate arm. After it passed, `control-r2`–`control-r4` were three fresh
process repetitions.

| arm | wall ms/forward | routed CPU body ms/forward | routed GB/s | VRAM bytes/GPU | RSS bytes |
|---|---:|---:|---:|---:|---:|
| control-r1 (cheap) | 150.782265 | 84.789283 | 40.680740 | 21,287,272,448 | 157,732,618,240 |
| control-r2 | 151.520846 | 85.096486 | 40.533880 | 21,287,272,448 | 157,732,532,224 |
| control-r3 | 146.828976 | 81.021438 | 42.572569 | 21,287,272,448 | 157,732,597,760 |
| control-r4 | 151.155686 | 84.710043 | 40.718794 | 21,287,272,448 | 157,732,503,552 |

The binding fresh matrix median/range (`r2`–`r4`) is **151.155686 ms** and
**146.828976–151.520846 ms/forward**. Routed body is **84.710043 ms** with
**81.021438–85.096486 ms** range, and the corresponding median bandwidth is
**40.718794 GB/s** with **40.533880–42.572569 GB/s** range. The result is
within the predeclared 0081 comparison window
`144.650011–159.876328 ms` and above the `36.7 GB/s` floor. It is not called
a speed win: the 0081 timing range overlaps this result and the correction
was selected for capacity, not latency.

All four arms generated exactly:

```text
[43, 8806, 440, 5270, 4496, 1205, 9238, 304, 366, 260,
 3418, 294, 6719, 8454, 305, 3345]
```

Every arm had 645/645 decode batches, 3,870 routed expert associations, zero
callback failures, zero decode checkpoint bytes, zero KV misses/promotions,
zero cache misses/evictions, zero demand-H2D bytes, and zero decode weight or
workspace allocations. The monitored PID was the only target compute process;
both GPU UUIDs peaked at 20,292 MiB (`21,277,704,192 B` by the monitor's MiB
conversion), while the runtime's `total-free` measurement was exactly the
hard ceiling on both devices. These are two accounting views and are not
added. `/usr/bin/time` maximum RSS ranged from
`157,721,206,784` to `157,724,704,768 B`; the runtime JSON end snapshots
ranged from `157,732,503,552` to `157,732,618,240 B`. Both are well below the
host ceiling. The raw `generation.log` files retain the exact KiB values.

## Admission and memory ledger

At runtime after CUDA context initialization, both selected devices reported:

```text
fractional .95 budget       23,757,176,832 B/GPU
explicit byte budget        21,256,421,376 B/GPU
applied bound                explicit on both GPUs
reduction                    2,500,755,456 B/GPU
aggregate applied plan      42,512,842,752 B
```

The corrected aggregate plan is:

```text
resident spine (including device mHC)  9,204,991,520 B
fixed workspace reserve                   536,870,912 B
device physical KV                         7,236,928 B
remaining expert VRAM cache             32,763,743,392 B
total applied budget                    42,512,842,752 B
```

These are disjoint portions of the plan. The actual CUDA weight allocation was
`41,968,734,624 B` in 88 setup allocations; setup workspace allocation was
`12,456,964 B` in 77 calls and decode allocations were zero. The per-device
weight-cache capacities after fixed reservations were
`[20,984,311,392, 20,848,443,072] B`; populated/peak cache use was
`[4,073,858,048, 5,594,845,696] B`, pinned use was
`[4,073,858,048, 4,995,026,432] B`, and unused capacity was
`[16,910,453,344, 15,253,597,376] B`. Decode cache hits were 7,050 and all
miss/eviction/demand-H2D counters were zero.

The runtime's measured used bytes minus the applied plan are
`30,851,072 B/GPU`, exactly the historical context/driver allowance. The
arm therefore does not hide an allocator overrun or shave an unmeasured
amount. The separate `nvidia-smi` process peak is lower, as noted above. This
is a pass of the declared measured-overhead gate, not evidence that the two
accounting systems expose identical allocations.

## Cost model and signs

For the binding fresh median, the exact routed payload is
`3,449,290,752 B/forward`. Using the inclusive runtime routed CPU body:

```text
W_CPU = 3,449,290,752 B/forward
B_CPU = 40.718794 GB/s
W_CPU/B_CPU = 84.710043 ms/forward
argmax_r = routed CPU body
wall = 151.155686 ms/forward
Sigma_serial envelope = wall - W_CPU/B_CPU = 66.445643 ms/forward
tau = max_r(W_r/B_r) + Sigma_serial = 151.155686 ms/forward
```

The gate/up, down, and weighted-reduction phase counters were respectively
`53.964325`, `25.835641`, and `3.799708 ms/forward` (sum `83.599674 ms`).
Their approximately `1.110369 ms` difference from the inclusive CPU body is
the scheduler/callback/body accounting remainder; it is not silently added to
the wall residual. Shared-collect was a median `128.065328 ms/forward` and
CUDA synchronization `128.024237 ms/forward`; both overlap the CPU body and
are not additive terms. GPU kernel service was `1.943991 ms/forward` and
attention graph work `16.088768 ms/forward`; mHC pre/post were `0.106619` and
`3.392663 ms/forward`. Decode activation traffic was unchanged at
`2,370,221 B H2D` and `1,693,016 B D2H` per forward. Weight demand H2D was
zero after setup.

The byte cap reduces reserved VRAM capacity and has a beneficial capacity
sign. It does not reduce the CPU argmax or any measured `W/B` term. CPU,
GPU, attention, mHC, application transfer, host RSS, and serial terms remain
neutral within the observed 0081 variance. A cache miss, eviction, demand
H2D, or timing regression would have been a positive sign and a rejection;
none occurred. No NCCL candidate arm was run, so NCCL logical/physical
candidate traffic remains not applicable/not measured here.

## Stage-6 capacity reserve and authorization

The independently retained rank-local projection remains
`10,371,440,068 B/rank` for Stage 4 plus `12,657,412 B/rank` of Stage-5
buffers, or `10,384,097,480 B/rank`. Adding the newly measured conservative
runtime/context allowance `30,851,072 B` gives
`10,414,948,552 B/rank`, leaving `10,872,323,896 B/rank` below the hard
`21,287,272,448 B` ceiling. Centralized control residency and rank-local
projection are not summed. This is a feasibility check only; Stage-6 NCCL
allocations and physical SHM traffic still require an actual candidate
measurement, and this experiment does not authorize Stage 6.

Stage 5/5R/5R.1/6A and 0081 remain preserved with their original rejection
scopes. 0082 makes the current control memory-valid under the explicit
contract, but the next stage remains **BLOCKED / NOT AUTHORIZED** pending
human review of this memory-contract result and the prior topology evidence.
No rank-local candidate, Stage 6, graph, mHC integration, attention
integration, or CPU arithmetic work was started.

## Artifacts and validation

Ignored deterministic artifacts are under:

```text
results/dsv4-vram-byte-admission/control-one/       # preserved JSON-quoting failure arm
results/dsv4-vram-byte-admission/control-r1/
results/dsv4-vram-byte-admission/control-r2/
results/dsv4-vram-byte-admission/control-r3/
results/dsv4-vram-byte-admission/control-r4/
```

The malformed first arm is preserved: generation JSON SHA-256
`b6a6b2e113cf848f353e6ae7a5b8083b0bdca4cfa42b240145d7eba5c0f257a4`. It
reached exact output and the cap but was rejected because the first serializer
emitted unquoted bound labels; it is not used in the timing matrix.

Binding generation JSON SHA-256 values are:

```text
control-r1 48142b9f96c1d586e86a3fcf29f40a747d022248f66d8c975200103468b732aa
control-r2 81df01d0997767b665e7dc0d54c7ee53a02e16d54487df9f4a612630582ea85f
control-r3 6a71697bfbb44d7a101e4adc7b13ccde9f784f69102f1bcb45afdda1e9b5de01
control-r4 bee97cdfb4462d0284e0643eee430a3ed3db1f22d128f419ae1bf556acbf5789
```

The final Release runner SHA-256 is
`9df4416bf351982d9d89a00728bf9d5b950d0348b65e7465d91144f2166def6b`; the
script SHA-256 is
`fb3cb4c1820f67682176c1f417f94e2c12d5bf01b046ce69d9ac9e723ebb73ed`.
The four-arm deterministic matrix summary is
`results/dsv4-vram-byte-admission/matrix-summary.json`, SHA-256
`93de52f303b76fcfa3d996e61642cf9c0d51457292826faffc8be71aef51531f`.
Focused builds of `strata-tests` and `strata-deepseek-run` passed, the full
test binary passed `274/275` with one existing opt-in skip, CLI parser probes
passed, the admission-only JSON probe passed, and the three fresh runtime
arms passed their structural validation. `make check` and `git diff --check`
are required again immediately before the single result commit.

This experiment stops at the memory review boundary.
