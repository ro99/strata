# Experiment 0069: DeepSeek reference device runtime integration

## Status

**In progress on `feat/dsv4-reference-device-runtime`.** Experiment 0068
accepted exact physical-page-to-BF16 attention at layers 2, 21, and 42. This
experiment stops creating component probes and integrates the accepted
implementation into the live Strata runtime in dependency order.

The retained checkpoints now add the required physical KV representation,
normal CUDA-backend attention body, persistent accepted-mHC runtime, real
CPU-MoE callback/device join, and the same-device fixed-buffer chain through
the FFN router. The short exact full-model oracle passes. Cross-device and
remaining host-visible ownership are still open. Checkpoint 15 also measures
an MoE overlap envelope above the 100 ms target; checkpoint 16's first direct
decoder mechanism was rejected, and its audit shows that the existing metric
was mislabeled as kernel-only. Live gate/up-versus-down attribution is the next
measurement before another kernel or scheduling change.

## Bounded hypothesis and operating point

```text
tau = max_r W_r / B_r + sum_serial

current total median                       244.812 ms/token
complete MoE                              114.667 ms
serialized non-MoE / handoff path          130.145 ms
attention                                   71.140 ms
mHC pre + post                              53.061 ms
accepted CPU floor + envelope         90.061--90.197 ms
target                                  <=100.000 ms/token
```

- **Hypothesis.** Feeding the accepted attention and mHC arithmetic from
  persistent reference-format device pages and fixed hidden buffers removes
  the dominant host-visible attention/mHC/handoff serialization without
  changing model semantics.
- **Primary metric.** Complete batch-one no-spec decode milliseconds per token,
  with attention, mHC, MoE, transfers, handoffs, and remaining serial time
  reported separately.
- **Correctness gate.** The retained attention path must reproduce zero BF16
  page and output mismatches at layers 2/21/42. The complete runtime must then
  pass real full-model teacher forcing and generation before performance work.
  Exact mode reports a failure instead of falling back.
- **Memory ceiling.** Reference KV and compressor/index state remain within the
  accepted 151,228,416-byte maximum-context allocation at the 0.95 admission
  point. Persistent mHC adds exactly 135,980,448 bytes: 135,266,304 projection
  bytes and 714,144 scale/base/norm bytes. Every additional fixed buffer and
  workspace is reported separately.
- **Resource signs.** The implementation increases GPU SM and HBM work while
  reducing host gather, H2D/D2H activation traffic, synchronization, and serial
  CPU attention/mHC work. Routed CPU work, NVMe bytes, routing, top-k, expert
  count, and precision are unchanged at this checkpoint.
- **Rollback.** Any page-layout mismatch, arithmetic mismatch, hidden fallback,
  memory-ceiling breach, full-model oracle failure, or non-improvement outside
  observed variance rejects the affected runtime change.

The integration gates use small deterministic fixtures or unit cases before a
full model load. A long arm is not justified until the complete chain passes
the layer and full-model correctness gates.

## Checkpoint 1: live reference physical KV representation

The previous live cache cannot back the accepted kernel: main rows were stored
as 583 contiguous bytes and learned-index rows used FP4. The reference
representation is deliberately explicit rather than silently reinterpreting
that format:

- sliding pages contain 256 rows;
- ratio-4 main and index pages contain 64 rows;
- ratio-128 main pages contain two rows;
- main pages contain a block-major 576-byte data plane and eight-byte scale
  plane per row;
- index pages contain a block-major 128-byte E4M3 data plane and one FP32 scale
  per row;
- insertion uses the accepted BF16 boundary, power-of-two scale, Ampere
  half-up E4M3 conversion, and canonical positive zero;
- device promotion uploads only the physical payload, excluding Strata's host
  metadata header.

The old compact and scalar formats remain unchanged. Reference physical mode
is mutually exclusive with the F32 oracle, requires the 256-source-token
allocator contract, and the old FP4 Lightning Indexer segment interface rejects
reference index pages rather than silently consuming them.

Validation after this checkpoint:

```text
cmake --build build --target strata-tests -j2
build/strata-tests

263/265 tests passed, 2 skipped
```

The new cases verify physical offsets, scale bytes, canonical zero, exact
round-trip values, all four per-kind page capacities and payload sizes, and
live cache reads across page boundaries. The two skips are the suite's existing
opt-in Kimi full-backbone arm and unavailable managed-environment Inkling CUDA
device test.

Decision: retain this checkpoint and continue with the backend API that reads
these persistent physical page leases directly. It does not yet authorize a
runtime performance arm.

## Checkpoint 2: accepted attention body in the normal CUDA backend

The backend now exposes an explicit DeepSeek physical-page request rather than
routing reference pages through generic FlashAttention. Each page descriptor
holds a persistent `CudaBuffer` and its physical row count; each ordered
candidate names a page and row or is invalid padding. The synchronous initial
integration uploads only:

- BF16 query values;
- 32 FP32 attention sinks;
- page pointer/row descriptors; and
- ordered candidate page/row/valid descriptors.

Historical KV payload bytes stay in their cache leases. The backend then runs
the experiment-0068 sequence: physical E4M3/E8M0/BF16 materialization, cuBLAS
BF16 scores, separate BF16 score scaling, accepted maximum/denominator/value
associations, `div.full.f32`, and BF16 output store. New counters separate
metadata H2D, result D2H, resident page bytes, kernel time, and complete call
time from generic FlashAttention.

CUDA 12.8 compiles and links this implementation into `strata_core`. SM86
resource inspection reports:

```text
materialize                    22 registers,    0 stack/local/shared
scale                           8 registers,    0 stack/local/shared
maximum                        20 registers,   32 shared bytes
denominator                    23 registers,   32 shared bytes
value                         136 registers, 2048 shared bytes, 0 stack/local
divide/store                   12 registers,    0 stack/local/shared
```

At 256 threads, the value kernel requests 34,816 registers per block, below
SM86's 65,536-register limit. The ordinary test target builds. A
CUDA-conditional nonzero test constructs a reference physical page whose 128
rows decode to all-one 512-wide values, then requires an all-one BF16 output.
When an SM86 device is visible this exercises persistent-buffer validation,
FP8/E8M0/BF16 materialization, candidate mapping, the complete finish sequence,
result transfer, and accounting. The managed execution environment has no
visible CUDA driver, so that case returns at its existing device-availability
guard and no GPU correctness result is claimed here.

Decision: retain the compile/resource checkpoint. The next step is live
runtime page-lease construction and the strict layer-2/21/42 output gate. Any
backend mismatch removes this path; there is no generic-attention fallback in
reference exact mode.

## Checkpoint 3: live runtime page dispatch

`Dsv4KvCacheMode::ReferenceDevice` is now a distinct exact runtime mode exposed
by `strata-deepseek-run --reference-device-runtime`. Initialization admits the
reference physical formats, requires the 256-source-token allocator geometry,
and derives per-device page-cache ceilings from the real layer schedule and
requested maximum context when the operator has not supplied explicit
ceilings. At 32,768 tokens the host admission accounts 139,022,336 payload
bytes plus block headers and alignment; the existing 0.95 VRAM admission still
owns the final device ceiling.

For each attention call the runtime:

- appends the BF16-boundary KV and compressed/index rows through the reference
  physical encoder;
- keeps the accepted unscaled E4M3 query contract for the learned index;
- obtains the live sliding and compressed block tables;
- acquires and holds device leases for every distinct referenced page;
- maps the ordered compressed region followed by the 128-row sliding region,
  with invalid candidates left as explicit padding; and
- calls only `CudaBackend::dsv4_paged_attention` in reference mode.

There is no host KV gather and no generic-attention fallback in this mode. The
current synchronous bridge still returns the 32x512 BF16 result to the host for
inverse RoPE and the existing projections. That bridge is deliberately visible
in the dedicated byte/time counters and remains work for the all-layer
device-owned checkpoint.

The first full-suite attempt exposed a test-contract mistake: the new
reference admission assertion assumed zero alignment slack even though the
cache allocator deliberately rounds every header-plus-payload block. The
assertion was corrected to require and include that slack; no runtime code was
relaxed. Final validation is:

```text
make check

2/2 ctest targets passed in 112.84 seconds
strata-tests: 266/267 passed, one existing opt-in Kimi full-backbone skip
git diff --check: clean
```

The operator SM86 rerun then reported:

```text
PASS native CUDA DeepSeek paged attention reads persistent physical pages
266/267 tests passed, 1 skipped
```

Commit `2112721` retains this checkpoint. The isolated layer-2/21/42 gates from
experiment 0068 remain the arithmetic oracle, while the nonzero all-one test
now verifies that the retained backend actually executes from a persistent
physical page on SM86. Attention remains yellow until a real short runtime arm
passes, but the implementation and hardware gates authorize continuing to the
accepted mHC integration. No throughput result is claimed yet.

## Checkpoint 4: persistent accepted mHC sequence

The accepted experiment-0062 SM86 arithmetic now lives in the normal CUDA
backend rather than another component probe. Initialization uploads all 86
target-format boundaries to one admitted SM86 device: each boundary owns one
24-by-16,384 FP32 projection plus three FP32 scales, 24 FP32 bases, and one
4096-wide BF16 norm vector. The 135,980,448 fixed bytes are removed from the
ordinary weight-cache capacity before its arena is created, so this path cannot
silently exceed the existing VRAM ceiling.

One token executes the reference command sequence:

```text
standalone pre                         1 call / 4 kernels
fused prior post -> next pre/norm     85 calls / 255 kernels
final post                             1 call / 1 kernel
total                                 87 calls / 260 kernels
```

The four-copy BF16 residual is double-buffered in one persistent device
workspace across all 43 layers. In the current correctness-first bridge, each
4096-wide branch result crosses H2D and only the next normalized layer input
crosses D2H. A normal untraced token therefore transfers 737,280 bytes in each
direction for mHC; weighted state and intermediate four-copy residuals return
only for an explicit layer-hash trace. The final four-copy state returns once
for the existing output head. Dedicated counters expose call type, launch
count, resident bytes, H2D/D2H bytes and time, kernel time, and complete time.

SM86 resource inspection of the retained backend object reports no stack or
local memory for any mHC kernel. Register/shared-memory use is 40/96 bytes for
the fused projection, 40/2048 for standalone projection, 40/0 for standalone
square sum, 34/96 for mix, 40/256 for weighted norm, and 34/0 for final post.

Reference-device prefill deliberately uses the serial token-major path until
the complete fixed-buffer graph exists. This makes the short correctness arm
exercise exactly the same state machine as decode; it is not a prefill
performance claim. Unsupported architectures fail the exact mode, and there
is no CPU mHC fallback.

Managed validation after the implementation attempt is:

```text
cmake --build build --target strata-tests strata-deepseek-run -j2
build/strata-tests
make check

266/268 tests passed, 2 skipped
2/2 CTest targets passed in 110.46 seconds
```

The two skips remain the opt-in Kimi full-backbone arm and the unavailable
managed-environment Inkling CUDA test. The CUDA-conditional mHC state-machine
case checks standalone/transition/final command order, retained residual
values, normalized output, persistent weight accounting, and both diagnostic
and production transfer contracts. The managed environment has no CUDA device,
so it does not satisfy the binding SM86 execution gate.

The operator RTX 3090 rerun then reported:

```text
PASS native CUDA DeepSeek paged attention reads persistent physical pages
PASS native CUDA DeepSeek reference mHC keeps the residual across transitions
267/268 tests passed, 1 skipped
```

The sole skip is the intentional ten-minute Kimi full-backbone arm. This closes
the binding SM86 mHC state, arithmetic, production-transfer, and accounting
gate without relaxing a contract.

Decision: accept and commit this bounded persistent-mHC checkpoint, then run
one short `--reference-device-runtime` full-model arm. Do not begin the real
CPU-MoE callback/reduction integration until that combined gate is recorded.

Commit `d087a4b` retains the persistent-mHC checkpoint.

## Checkpoint 5: combined real-runtime gate

`scripts/run_deepseek_v4_reference_device_correctness.sh` defines the smallest
real-checkpoint arm that exercises both token-major prefill and decode. It
restricts CUDA visibility to the two physical RTX 3090s by default, uses a
256-token context, prompt `Hi`, and two requested output tokens. This makes the
expected fixed setup roughly 50--90 seconds and the mechanism window less than
ten seconds; a longer prompt or repetition matrix would measure work not under
test.

The script first runs admission-only, then captures one exact generation plus
system state, binary hash, route trace, detailed CUDA counters, and finite-logit
diagnostics. For `F = prefill_tokens + decode_steps` completed forwards, its
binding gates are:

```text
paged-attention calls                 F * 86
paged-attention kernels               F * 602
mHC calls                             F * 87
mHC standalone / transition / final  F / F*85 / F
mHC kernels                           F * 260
mHC H2D and D2H                       F * 737,280 bytes each
generic FlashAttention calls          0
decode checkpoint reads               0
non-finite logits                     0
```

It also requires the exact 135,980,448-byte mHC admission and resident-weight
accounting, one actual decode step, two generated IDs, and no CPU prepack or
runtime fallback. Any failed predicate makes `jq -e` reject the arm. This gate
is not a throughput measurement.

The first operator arm admitted the two-RTX-3090 topology, loaded the complete
checkpoint, and failed closed after 55.24 seconds on the first attention
request:

```text
error: DeepSeek paged attention request shape, BF16 query, scale, or sink is invalid
maximum RSS: 145,235,540 KiB
major page faults: 885
swaps: 0
```

No generation JSON or summary was produced, and the route trace contains only
its schema header. This is an integration defect, not an arithmetic or
performance result. The backend's composite validation currently covers query,
sink, output, page/candidate count, scale, and workspace fields; the next cheap
step is to identify the exact invalid field at the live call boundary. Do not
continue to CPU-MoE/reduction integration or relax the request contract.

Source-level triage identifies the exact field without another model load. The
full Strata adapter owns 64 query heads and 64 attention sinks, while the
accepted physical-page backend is deliberately TP-rank-local at 32 heads, as
were all experiment-0068 fixtures. The live producer passed 64x512 query and
64-sink spans into the strict 32x512/32-sink request.

The bounded correction dispatches heads 0--31 and 32--63 as two ordered
rank-local calls over the same shared-KV page leases, writing the corresponding
halves of the 64-head output. Per-head attention is independent, so no
reduction or arithmetic order crosses that boundary. The backend contract,
accepted kernel, persistent page memory, and numerical gate remain unchanged;
only live call accounting doubles from 43 to 86 paged-attention calls per
forward. The CUDA unit now drives the same two-half shape. Managed and operator
reruns are separate from the failed arm so its raw evidence is preserved.

The managed gate after this correction passes:

```text
make check

2/2 CTest targets passed in 112.36 seconds
```

The binding two-RTX-3090 combined rerun remains pending.

The corrected two-RTX-3090 rerun passes every scripted predicate:

```text
accepted                              true
prompt / prefill / decode steps       5 / 5 / 1
generated token ids                   [30594, 1175]
finite / non-finite logits            258,560 / 0
generic FlashAttention calls          0
paged-attention calls / kernels       516 / 3,612
mHC calls                             522
mHC standalone / transition / final  6 / 510 / 6
mHC kernels                           1,560
mHC resident weight bytes             135,980,448
mHC H2D / D2H bytes                   4,423,680 / 4,423,680
decode checkpoint read bytes          0
prefill / decode seconds              3.708731 / 0.484040
wall time                             47.09 seconds
maximum RSS                           145,449,252 KiB
major page faults / swaps             6 / 0
```

With `F = 6`, every call, launch, and mHC transfer count equals its declared
formula. Paged attention reads 81,853,440 page bytes and reports 0.229551
seconds on its maximum device; mHC reports 0.048128 seconds on its maximum
device. Those times expose the current synchronous correctness bridge and are
not compared as throughput: this is one arm, its five-token prefill is serial,
and no variance was measured.

Decision: accept the full-adapter 64-to-two-by-32 head partition and combined
attention/mHC live-runtime gate. Commit the correction and gate harness. The
next eligible dependency is the real CPU-MoE callback with GPU shared/routed
reduction; complete hidden ownership and graph capture remain subsequent
stages. Full-model teacher forcing, generation comparison, and the three-arm
performance result remain deferred until that complete chain exists.

## Checkpoint 6: reference CPU-MoE placement enters the live chain

The accepted combined arm exposed a configuration defect before any callback
code was justified: `--reference-device-runtime` selected the accepted paged
attention and mHC paths but left the routed experts on the ordinary device-MoE
path unless the operator also supplied the unrelated-looking
`--tiled-host-moe` flag. Its one decode step spent 396.429 ms in MoE, including
341.526 ms in preparation, while the maximum-device MoE service time was only
39.540 ms. This is a correctness operating point, not a comparable performance
arm, but it proves that the current bridge is dominated by routed-expert
preparation rather than the accepted attention or mHC kernels.

The bounded integration correction makes the reference-device option imply the
already retained reference-shaped CPU placement. The production runner now
stages the two transformed NUMA-local expert shards and its 48-worker pool by
construction. The existing full-model harness is tightened to require
`reference_tiled_cpu_moe_autoregressive`, 43 MoE batches, 258 routed experts,
and 43 GPU shared experts per completed forward, with positive routed-CPU and
shared-collect timings. The result directory is new, so the accepted
attention/mHC evidence and the earlier failed arm remain immutable.

This checkpoint does not yet claim the external `cpu_decode` stream boundary:
the retained implementation still invokes the CPU worker synchronously and
downloads the shared output for a host reduction. Its purpose is to put the
real routed arithmetic and real GPU shared expert into the same live reference
runtime before replacing that final boundary with fixed staging, one CUDA host
function, and a GPU routed/shared join. Correctness remains finite logits,
exact route/call accounting, zero decode checkpoint reads, no fallback, and
the existing attention/mHC gates.

The two-RTX-3090 arm passes every predicate:

```text
accepted                              true
prompt / prefill / decode steps       5 / 5 / 1
generated token ids                   [30594, 9790]
finite / non-finite logits            258,560 / 0
CPU-MoE batches                        258
routed / shared experts               1,548 / 258
routed CPU / shared collect           0.928356 / 0.020189 s over six forwards
paged-attention calls / kernels       516 / 3,612
mHC calls / kernels                    522 / 1,560
decode checkpoint read bytes          0
prefill / decode seconds              1.683829 / 0.303725
resident staging / wall               77.156 / 89.33 s
maximum RSS                           154,004,968 KiB
per-device VRAM                       21,287,272,448 bytes
major page faults / process swaps     12 / 0
```

For the single decode forward, routed CPU work is 150.655 ms, shared-expert
collection is 3.829 ms, and the host final combine is 1.344 ms. Complete MoE
is 164.387 ms, including only 3.387 ms of preparation; the ordinary routed-GPU
arm had spent 396.429 ms in MoE and 341.526 ms in preparation. Routed-expert
weight H2D falls from 1,377,042,432 bytes to zero. The remaining shared-only
MoE command moves 704,512 H2D bytes and 704,684 D2H bytes across 43 layers and
uses 5.626 ms on the maximum device. This is the expected resource sign, but
the one-token correctness arms are not an interleaved performance comparison.

The second generated token changes from the ordinary routed-GPU arm's `1175`
to `9790`. That is expected because the installed CPU decode arithmetic is an
explicitly different, previously reconstructed numerical contract; it proves
that the live placement changed rather than silently falling back. The complete
chain still owes the external teacher-forcing and generation oracle before
promotion.

Decision: accept and commit the live CPU/shared placement checkpoint. The next
bounded implementation is now authorized: retain the same CPU arithmetic but
move its final two rank partials through fixed pinned/device buffers, execute
the routed/shared join on GPU, and make the CPU work one stream-ordered host
function. Complete hidden ownership remains the subsequent dependency.

## Checkpoint 7: fixed CPU callback and GPU routed/shared join

The implementation changes only the boundary after checkpoint 6. The same
transformed expert views, 48 persistent workers, NUMA-local stealing, gate/up,
down, and rank-local route reduction now run from one `cudaLaunchHostFunc`.
The callback receives backend-owned pinned storage for exactly two 4,096-float
rank partials; it invokes no CUDA API. Callback failure or exception fails the
command after its streams drain and never publishes an output.

The shared FP8 expert consumes the same hidden row concurrently on a reusable
auxiliary stream. After the callback, the main stream uploads both partials,
waits for shared completion, and applies the retained association on GPU:

```text
bf16(bf16(rank0 + rank1) + bf16(shared))
```

The combined FP32/BF16-boundary result is the only MoE vector downloaded. The
device and pinned workspaces grow once to the production shape and are reused;
no layer owns an allocation. At the current host-hidden checkpoint, each
43-layer forward therefore has this explicit boundary:

```text
hidden H2D                         704,512 bytes
two CPU rank partials H2D        1,409,024 bytes
combined output plus status D2H    704,684 bytes
callbacks / GPU joins                    43 / 43
shared + join kernels                    5/layer
host final combine                         removed
```

The extra partial H2D is not presented as a volume win. It replaces the host
join and establishes the fixed stream dependency needed by the accepted 0065
envelope; the later device-owned hidden stage removes the separate hidden H2D.
The join currently executes on the device that owns the layer's full shared
expert. It is an exact GPU routed/shared reduction, not yet an NCCL TP
collective or a claim about the reference's unavailable NCCL transport.

The CUDA unit supplies deterministic rank partials, verifies every output bit,
requires five kernels and the exact transfer volumes, forces a callback
failure, verifies that no output escapes, then reuses the same command. The
compiled unit binary passes 266/268 tests; the two skips are the intentional
ten-minute Kimi backbone and an unrelated unavailable-device Inkling gate.

The full managed gate passes:

```text
make check

2/2 CTest targets passed in 110.61 seconds
```

The two-RTX-3090 real-model gate passes every predicate:

```text
accepted                              true
prompt / prefill / decode steps       5 / 5 / 1
generated token ids                   [30594, 9790]
finite / non-finite logits            258,560 / 0
CPU callbacks / failures / GPU joins  258 / 0 / 258
MoE commands / kernels                258 / 1,290
MoE H2D / D2H bytes                   12,681,216 / 4,228,104
decode checkpoint read bytes          0
prefill / decode seconds              1.429823 / 0.217343
resident staging / wall               78.972 / 90.11 s
maximum RSS                           154,006,104 KiB
per-device VRAM                       21,287,272,448 bytes
major page faults / process swaps     57 / 0
```

For the single decode forward, all 43 callbacks and joins complete with zero
failure, the CPU arithmetic takes 101.652 ms, shared collection after callback
completion takes 2.043 ms, and the removed host combine remains exactly zero.
The complete graph MoE phase is 111.378 ms. Its CUDA accounting is exactly 43
commands, 215 kernels, 2,113,536 H2D bytes, and 704,684 D2H bytes. The reported
53.920 ms maximum-device MoE interval now includes time that the main stream is
stopped in the CPU callback; it is not 53.920 ms of GPU kernel work.

The preceding placement-only correctness arm measured 303.725 ms for decode,
164.387 ms for complete MoE, and 150.655 ms for routed CPU work. This arm
measures 217.343, 111.378, and 101.652 ms respectively. That direction is
consistent with the intended overlap and removed host combine, but these are
single sequential correctness arms and the unchanged CPU arithmetic itself
varied by 49.003 ms. They are not a throughput comparison and do not replace
the required three comparable repetitions after the complete device-owned
chain exists.

Decision: accept and commit the fixed callback/GPU-join boundary. It preserves
the checkpoint-6 token IDs, finite logits, route counts, attention/mHC gates,
zero-NVMe decode, and memory ceiling while satisfying every new callback,
kernel, and transfer predicate. The next eligible implementation is complete
device ownership of hidden state across all 43 layers; no further component
probe is needed before that integration stage.

## Checkpoint 8: device-owned FFN output boundary

The first complete-hidden integration cut keeps the existing CPU-visible
normalized layer input because the router and routed experts still consume it,
but removes the redundant FFN output round trip. In reference-device mode the
shared expert now reads the persistent BF16 mHC `layer_input` directly. The
unchanged stream callback uploads its two rank-local FP32 partials, and the
exact GPU join writes both its optional diagnostic FP32 result and the BF16
result consumed by the persistent mHC workspace. The following mHC transition
or final post consumes that device branch without another upload.

A strict ready flag couples producer and consumer. A device mHC transition is
invalid before a successful MoE collect, a host branch cannot silently replace
a pending device result, and a second producer cannot overwrite an unconsumed
branch. Callback, kernel, status, or collect failure therefore fails closed.
Outside reference-device mode the established host boundary is unchanged.

The hypothesis targets serialized cross-engine activation handoffs, not the
still-larger CPU memory-bandwidth term. At the accepted checkpoint-7 operating
point, one decode step was 217.343 ms: routed CPU was 101.652 ms, attention
93.127 ms, mHC pre/post 8.544 ms, and complete MoE 111.378 ms. The routed CPU
payload was 3.449290752 GB / 101.652 ms = 33.933 GB/s aggregate. This cut does
not reduce that payload. It establishes the fixed producer/consumer boundary
needed to overlap the remaining device chain.

Per 43-layer forward, the declared accounting changes are:

```text
                                      checkpoint 7       candidate
MoE kernels                                  215              215
MoE H2D transfers / bytes             86 / 2,113,536   43 / 1,409,024
MoE D2H transfers / bytes             86 /   704,684   43 /       172
mHC H2D bytes                              737,280          385,024
mHC D2H bytes                              737,280          737,280
total activation-byte reduction                         1,761,280
```

The 704,512-byte MoE hidden upload, 704,512-byte combined-output download, and
352,256-byte mHC FFN-branch upload disappear. Attention output and every next
normalized layer input remain host-visible, so this is explicitly not yet
complete 43-layer hidden ownership or a throughput result. The memory ceiling
remains 21,287,272,448 bytes per device; the shared expert is acquired on the
same slot as the persistent mHC workspace, with no peer copy.

The retained CUDA unit now also constructs a target-width 4,096-element mHC
state and, when an SM86 is available, requires the device-source join to match
the old host-source join and next mHC residual bit for bit. It rejects
missing/stale producer order and requires the production no-diagnostic command
to use five kernels, one 32,768-byte H2D, and one four-byte D2H. The managed
build, `git diff --check`, and script syntax pass; the broad unit binary reports
266/268 with its two expected skips, and full `make check` passes 2/2 in 109.98
seconds. No NVIDIA device is exposed in the managed environment, so those
checks do not accept the new SM86 execution path.

The short two-RTX real-runtime gate accepts the cut:

```text
accepted / generated IDs                   true / [30594, 9790]
finite / non-finite logits                 258,560 / 0
callbacks / failures / joins               258 / 0 / 258
MoE commands / kernels                     258 / 1,290
MoE H2D transfers / bytes                  258 / 8,454,144
MoE D2H transfers / bytes                  258 / 1,032
mHC H2D / D2H bytes                        2,310,144 / 4,423,680
decode checkpoint reads                    0
per-device VRAM                            21,287,272,448 bytes
maximum RSS                                157,721,300,992 bytes
process swaps                              0
prefill / decode                           1.830540 / 0.201717 s
```

Every count and byte total matches the equations above. The one decode forward
uses 43 callbacks with zero failures, 43 joins, 1,409,024-byte MoE H2D,
172-byte MoE D2H, 385,024-byte mHC H2D, and 737,280-byte mHC D2H. Its routed
CPU phase is 88.751 ms, complete MoE is 97.899 ms, attention is 92.537 ms, and
mHC post is 6.950 ms. The accepted paged-attention command itself is only
4.228 ms and mHC's backend span is 5.728 ms; the decode still executes 600
synchronizations and moves 22,200,064 activation bytes H2D plus 12,384,600 D2H
through the remaining host-staged projections and boundaries.

The preceding checkpoint-7 correctness arm measured 217.343 ms decode and
101.652 ms routed CPU. This arm measures 201.717 and 88.751 ms, but the CPU
variation means the 15.626 ms total difference is not attributed to this cut.
The correctness and exact transfer reduction, not a throughput comparison, are
the acceptance evidence.

Decision: accept and commit the device-owned FFN output boundary. The next and
only eligible implementation stage is the complete attention-output/device
activation chain around the already-accepted paged attention and mHC kernels.
Do not create another component probe. After that chain, remove per-operation
synchronization through fixed-buffer graph ownership, then run the full-model
oracles and three comparable performance repetitions.

## Checkpoint 9 candidate: attention output directly into persistent mHC

The next live integration cut is compiled. It does not change the accepted
attention arithmetic: one physical-page materialization feeds two ordered
32-head BF16 cublas score operations, each followed by the accepted scale,
maximum, denominator, value and `div.full.f32` finish sequence. The two BF16
outputs remain on the source GPU. In the same stream, a CUDA kernel applies the
precomputed inverse-RoPE cosine/sine pairs with explicit round-to-nearest
operations, the checkpoint's exact BF16-expanded `wo_a` executes as the same
eight-group projection, its output crosses the established BF16 boundary, and
the native FP8 `wo_b` produces the final BF16 branch.

That branch is copied into the active persistent mHC workspace instead of
returning through the old host attention output and mHC upload. On layers whose
attention and mHC work share a GPU, the handoff is D2D. On the other GPU, the
machine's lack of peer access is represented honestly as one fixed pinned
8,192-byte D2H/H2D BF16 handoff. A diagnostic branch download exists only when
layer-hash tracing requests it. Physical-page failure still downloads four
bytes and synchronizes the source stream, so this cut reduces orchestration but
does not yet claim the final one-wait CUDA graph.

The candidate removes these host continuations from every reference attention
call:

```text
two 32x512 attended-result downloads and waits
host inverse RoPE over 64 heads
attended activation upload for grouped wo_a
wo_a result download and wait
wo_a result upload for wo_b
wo_b branch download and wait
attention branch upload into mHC
```

It replaces the two independent seven-launch paged commands plus the generic
output projection calls with one 19-launch ordered command, one physical-page
materialization, and one source status wait. Weight leases remain held until
that command completes, the target mHC ready flag fails closed on missing or
duplicate production, and non-reference runtime paths are unchanged.

The measured bottleneck motivating the cut is still the checkpoint-8 decode:
201.717 ms total, 92.537 ms attention, 97.899 ms complete MoE, and 600 device
synchronizations. The accepted paged-attention backend was only 4.228 ms, so
the target is the serial host-staged attention output term, not routed CPU,
whose 88.751 ms is already near the 87.879 ms standalone floor. The correctness
gate remains exact generated IDs `[30594, 9790]`, 258,560 finite logits, zero
decode checkpoint reads, zero callback failures, and no layer/logit divergence.
The memory ceiling remains 21,287,272,448 bytes per device; any exact mismatch,
nonzero read, command-order failure, or ceiling increase rejects and rolls back
the cut.

Managed CUDA/C++ compilation, script syntax, `git diff --check`, and full
`make check` pass; the latter reports 2/2 CTest targets in 101.92 seconds. No
NVIDIA device is visible in the managed environment, so this is explicitly a
candidate rather than an acceptance. The binding next action is one short two-RTX run of
`scripts/run_deepseek_v4_reference_device_correctness.sh`. Do not start graph
capture or the query/KV-side activation continuation until that result is
recorded.

### Checkpoint 9 two-RTX acceptance

The short live gate accepts the attention-output-to-mHC cut. It executes one
combined 19-launch command per layer, shares page materialization across the
two 32-head score groups, and removes every mHC attention-branch upload. All
six forwards satisfy the declared counts:

```text
accepted / generated IDs                   true / [30594, 9790]
finite / non-finite logits                 258,560 / 0
aggregate raw-logit trace hash             d8e0810706b18707
per-forward raw-logit hashes               a73dc2e74d875fc3
                                             0c7ed0f4eb38f2f7
paged-attention calls / kernels            258 / 4,902
paged-attention H2D / D2H                  18,400,272 / 1,082,376 bytes
mHC calls / kernels                        522 / 1,560
mHC H2D / D2H                              196,608 / 4,423,680 bytes
callbacks / failures / joins               258 / 0 / 258
MoE H2D / D2H                              8,454,144 / 1,032 bytes
decode checkpoint reads                    0
per-device VRAM                            21,287,272,448 bytes
maximum RSS                                157,721,464,832 bytes
major page faults / process swaps          1 / 0
prefill / decode                           1.509295 / 0.170142 seconds
```

Both per-forward raw-F32 logit hashes and the aggregate trace hash equal the
checkpoint-8 run exactly. Thus the device inverse RoPE, BF16-expanded grouped
`wo_a`, native FP8 `wo_b`, cross-device BF16 staging, and persistent mHC store
introduce no observed numerical change—not merely no token-ID change.

For the single decode forward, the retained boundary changes are:

| Term | Checkpoint 8 | Checkpoint 9 | Change |
|---|---:|---:|---:|
| decode | 201.717 ms | 170.142 ms | -31.575 ms |
| attention | 92.537 ms | 69.293 ms | -23.244 ms |
| paged backend span | 4.228 ms | 11.151 ms | +6.923 ms, now includes inverse RoPE and both output projections |
| mHC H2D | 385,024 bytes | 32,768 bytes | -352,256 bytes |
| activation H2D | 22,200,064 bytes | 14,767,104 bytes | -7,432,960 bytes |
| activation D2H | 12,384,600 bytes | 7,633,068 bytes | -4,751,532 bytes |
| synchronizations | 600 | 471 | -129 |
| critical synchronization | 113.654 ms | 103.840 ms | -9.814 ms |

This is one sequential correctness arm. CPU routed work also varied from
88.751 to 81.713 ms, so neither the 31.575 ms total reduction nor its implied
5.88 tok/s is the final performance claim. The exact hashes, transfer removal,
command counts, zero reads, and unchanged memory ceiling are the acceptance
evidence.

Decision: accept and commit checkpoint 9. Checklist item 1 remains **partial**:
the attention finish/output half is device-owned, but `wq_a`, query norm,
`wq_b`, `wkv`, compressor/indexer projections, and the normalized layer-input
CPU boundary still use generic host-staged calls. Checklist item 2 remains
**not implemented**: this arm still has 471 decode synchronizations and one
source status wait per combined attention command. The next eligible change is
the remaining Q/KV/device-input activation chain. CUDA-graph capture follows
only after that chain passes its real-model oracle.

## Checkpoint 10 candidate: persistent Q/KV producer

Checkpoint 9 left three generic projection commands per layer ahead of the
accepted paged-attention/output chain. Each command independently uploaded its
activation, launched activation quantization and a projection, downloaded its
result, and synchronized. The query result was then uploaded again by paged
attention. In the accepted decode these host continuations accounted for 129
projection calls, 24.510 ms of query work, 10.000 ms of KV work, and part of
the remaining 471 synchronizations.

Checkpoint 10 replaces those three commands with one bounded persistent
producer on the layer GPU:

```text
persistent mHC BF16 layer input (same GPU), or one explicit BF16 handoff
  -> shared exact E4M3 activation quantization
  -> wq_a
  -> serial-FP64 query-rank RMSNorm and BF16 boundary
  -> exact E4M3 activation quantization
  -> wq_b
  -> serial-FP64 per-head RMSNorm, adjacent-pair RoPE, BF16 query
  -> wkv from the shared input
  -> serial-FP64 KV RMSNorm, adjacent-pair RoPE, BF16 KV
```

The serial reduction order, separate rounded multiplications, host-computed
cosine/sine pairs, and every BF16 boundary reproduce the existing host path.
The 64×512 query remains in a dedicated device workspace and the immediately
following paged command consumes it without H2D. The 1,024-wide query-rank and
512-wide KV rows still download because the long-context indexer and current
physical-page allocator own host bookkeeping; this cut does not disguise that
remaining boundary. Compressor and indexer projections are also unchanged and
remain the next part of checklist item 1.

The hypothesis reduces the measured serialized attention-orchestration term,
not routed CPU. It should replace 129 projection synchronizations with 43
producer synchronizations per decode and remove the 2,818,048-byte query H2D
performed by the paged commands, plus redundant Q/KV input/output traffic. The
gate is the same actual-model short run: exact aggregate logit hash
`d8e0810706b18707`, exact forward hashes `a73dc2e74d875fc3` and
`0c7ed0f4eb38f2f7`, IDs `[30594,9790]`, zero non-finite logits, zero decode
reads, zero callback failures, and no increase over 21,287,272,448 bytes per
device. Any mismatch or stale prepared-query command rejects the cut before
compressor/indexer integration proceeds.

Managed CUDA/C++ compilation, script syntax, and `git diff --check` pass. Full
`make check` passes both CTest targets in 136.27 seconds. The managed
environment has no usable NVIDIA device, so the candidate remains gated by one
short two-RTX execution of the updated correctness script; its jq predicate
now requires the exact aggregate and per-forward raw-logit hashes above.

### Checkpoint 10 two-RTX acceptance

The updated script accepts the persistent Q/KV producer. The aggregate raw
logit hash remains `d8e0810706b18707`; the two forward hashes remain
`a73dc2e74d875fc3` and `0c7ed0f4eb38f2f7`. Generated IDs are unchanged,
all 258,560 logits are finite, decode reads remain zero, callback failures are
zero, and per-device VRAM remains 21,287,272,448 bytes.

The command accounting proves the intended boundary rather than inferring it
from elapsed time:

| Decode term | Checkpoint 9 | Checkpoint 10 | Change |
|---|---:|---:|---:|
| paged-query/metadata H2D | 3,066,880 bytes | 248,832 bytes | **-2,818,048 bytes** |
| total activation H2D | 14,767,104 bytes | 10,819,328 bytes | **-3,947,776 bytes** |
| total activation D2H | 7,633,068 bytes | 1,865,048 bytes | **-5,768,020 bytes** |
| synchronizations | 471 | 385 | **-86** |
| attention | 69.293 ms | 68.345 ms | -0.948 ms |
| paged/output backend | 11.151 ms | 11.149 ms | -0.002 ms |
| routed CPU | 81.713 ms | 95.876 ms | +14.163 ms variance |
| total decode | 170.142 ms | 184.255 ms | +14.113 ms variance |

The exact synchronization reduction is the predicted replacement of three
generic Q/KV waits per layer by one producer wait: `43 × (3 - 1) = 86`. The
paged H2D reduction is exactly the removed 64×512 BF16 query upload for all 43
layers. Attention itself is effectively unchanged in these single sequential
correctness arms, while CPU routed work varies by 14.163 ms and explains the
total regression. Therefore this is accepted as an exact device-ownership and
traffic prerequisite, not called a throughput win.

Decision: retain and commit checkpoint 10. Checklist item 1 remains partial.
The next eligible integration is the compressor/indexer projection schedule
that still rereads the same host-visible input and synchronizes around its
independent projections. Checklist item 2—fixed-buffer CUDA graph ownership
and one final wait—has not started and remains gated on completing item 1.

## Checkpoint 11 candidate: shared-input compressor projections

Checkpoint 10 still invokes the main compressor `wkv` and `wgate` as two
generic projection commands after the Q/KV producer. Long-context layers with
an admitted sparse indexer repeat the same pattern for
`indexer.compressor.wkv` and `wgate`. Each pair independently requantizes the
same BF16 hidden row, transfers it through the generic host API, and adds two
stream synchronizations even though the accepted Q/KV command already owns an
exact E4M3 copy of that row.

Checkpoint 11 makes those projection pairs optional outputs of the existing
bounded producer. They expand its persistent BF16 input exactly to float and
use the same BF16 matvec kernel as the generic path, then return raw F32
outputs in the producer's existing completion download. The current
serial host softmax, APE addition, FP64 accumulation, normalization, RoPE, and
physical-page insertion remain unchanged. Thus this cut targets only repeated
activation work and serialized command boundaries; it does not claim to move
compression state or sparse selection to the GPU.

The short 256-token correctness arm exercises every admitted main compressor;
the optional index-compressor path is wired for the long-context admission
shape but sparse selection is not reached by this cheap arm. The binding gate
is unchanged: aggregate raw-logit hash `d8e0810706b18707`, forward hashes
`a73dc2e74d875fc3` and `0c7ed0f4eb38f2f7`, IDs `[30594,9790]`, all logits
finite, zero decode reads and callback failures, and no increase over
21,287,272,448 bytes per device. The primary measured terms are activation
H2D/D2H bytes and synchronization count; routed CPU time is recorded but is
not affected by this mechanism. A hash mismatch, command-order failure, or
memory-ceiling increase rejects the candidate.

Managed CUDA/C++ compilation and full `make check` pass; both CTest targets
complete in 121.42 seconds. The operator RTX gate remains before retention.
Checklist item 1 remains partial even if this passes: index selection
projections/state and the normalized CPU-visible layer input are still outside
a fixed device graph. Checklist item 2 has not begun.

### Checkpoint 11 first RTX attempt: rejected encoding assumption

The first live attempt stopped before producing logits with `DeepSeek attention
preparation weights violate the FP8 contract`. Direct inspection of layer 2
shows both main compressor matrices are plain BF16 `[1024,4096]`; the optional
index-compressor matrices are plain BF16 `[256,4096]`. QKV matrices are native
FP8, so treating all producer weights as FP8 was an invalid implementation
assumption. The backend rejected it explicitly, as required, and no correctness
or performance result exists for this arm.

The ownership hypothesis is not falsified: the accepted generic compressor
path already uses the backend's exact BF16 matvec kernel on a float view of the
BF16 layer input. The correction is to expand the persistent BF16 input exactly
to float inside the same stream and invoke that identical BF16 matvec for the
compressor matrices. Quantizing the compressor input or relaxing the encoding
gate is forbidden because either would change the established arithmetic.

The corrected backend requires plain BF16 compressor weights explicitly,
performs the exact expansion only after the native-FP8 QKV consumers finish,
and launches the existing eight-warps-per-block BF16 matvec unchanged. Managed
compilation and full `make check` pass; both CTest targets complete in 113.25
seconds. At this point the failed arm had produced no generation JSON and was
not acceptance evidence; the repeated arm below is the binding result.

### Checkpoint 11 corrected RTX acceptance

The repeated RTX arm accepts. The aggregate raw-logit hash is exactly
`d8e0810706b18707`; forward hashes are exactly `a73dc2e74d875fc3` and
`0c7ed0f4eb38f2f7`. Generated IDs remain `[30594,9790]`, all 258,560 logits
are finite, decode checkpoint reads and callback failures remain zero, and
per-device VRAM remains exactly 21,287,272,448 bytes.

The deterministic boundary accounting is:

| Decode term | Checkpoint 10 | Checkpoint 11 | Change |
|---|---:|---:|---:|
| activation H2D | 10,819,328 bytes | 9,475,840 bytes | **-1,343,488 bytes** |
| activation D2H | 1,865,048 bytes | 1,865,048 bytes | 0 |
| synchronizations | 385 | 303 | **-82** |
| critical synchronization | 126.456 ms | 118.270 ms | -8.185 ms |
| attention | 68.345 ms | 66.014 ms | -2.330 ms |
| routed CPU | 95.876 ms | 88.891 ms | -6.985 ms variance |
| total decode | 184.255 ms | 174.590 ms | -9.665 ms |

There are 41 active main compressors in the 43-layer decode. Removing two
generic commands per active layer predicts exactly `41 × 2 = 82` waits. Each
generic command uploaded one 4,096-float row, predicting
`41 × 2 × 4,096 × 4 = 1,343,488` removed H2D bytes. Both observations match
exactly. D2H is unchanged because the raw F32 compressor outputs still return
for the unchanged host compression-state arithmetic. CPU routed time varied by
6.985 ms, so the total-time change is not isolated enough to call a throughput
win; exact hashes and command/byte removal are the acceptance evidence.

Decision: retain and commit checkpoint 11. Checklist item 1 remains partial:
the short arm does not exercise sparse index selection, and index-query
projections/state plus the normalized CPU-visible layer input remain outside
the owned device chain. Checklist item 2, fixed-buffer CUDA-graph ownership and
one final wait, has not started; the accepted arm still has 303 waits.

## Checkpoint 12 candidate: same-GPU mHC-to-attention command

At checkpoint 11 the real one-token operating point is 174.590 ms. The
critical-device measurements are 125.406 ms of kernel work, 118.270 ms at 303
completion boundaries, 1.614/1.687 ms H2D/D2H engine time, and 88.891 ms routed
CPU. Device work is the largest resource term; the remaining per-operation
wait schedule serializes its dependent kernels and is the targeted serial term.
The router's 3.400 ms host-visible projection is not selected because removing
only its 704 KiB input traffic would not reduce the current maximum.

After each FFN branch, the mHC transition produces the normalized input for the
next attention branch. For 21 of the 42 inter-layer transitions, that next
attention layer is assigned to the same GPU that owns persistent mHC. The
current runtime nevertheless downloads the 4,096-element BF16 row and waits,
then the QKV/compressor producer reads the persistent copy already on that same
stream and waits again. Checkpoint 12 combines the accepted three-kernel mHC
transition with the existing attention producer and retains one final wait.

The cut is deliberately unavailable when the next attention is on the other
GPU, layer-hash diagnostics require the host row, or a sparse indexer is
admitted and may consume the host input. Thus it cannot silently make the
long-context path use stale host state. At this TP2 short-context operating
point it predicts exactly 21 fewer synchronizations and
`21 × 4,096 × 2 = 172,032` fewer D2H bytes. It changes no kernel arithmetic,
weights, page state, CPU-MoE callback, or memory ceiling.

The gate remains exact aggregate/forward logit hashes, `[30594,9790]`, all
finite logits, zero decode reads/callback failures, and at most
21,287,272,448 bytes per device. A command-order error, hash mismatch, anything
other than 282 decode waits, or failure to remove exactly 172,032 D2H bytes
rejects the combined boundary. This is bounded wait coalescing, not yet the
claimed full fixed-buffer CUDA graph or one final step wait.

Managed CUDA/C++ compilation, script syntax, and `git diff --check` pass. Full
`make check` passes both CTest targets in 109.11 seconds. The candidate now
requires one short two-RTX execution of the updated exact-oracle script.

### Checkpoint 12 two-RTX acceptance

The live arm passes every binding gate. Aggregate raw-logit hash
`d8e0810706b18707` and forward hashes `a73dc2e74d875fc3` and
`0c7ed0f4eb38f2f7` remain exact; IDs remain `[30594,9790]`; all 258,560
logits are finite; decode reads and callback failures remain zero; and both
devices remain at 21,287,272,448 bytes.

| Decode term | Checkpoint 11 | Checkpoint 12 | Change |
|---|---:|---:|---:|
| synchronization count | 303 | 282 | **-21** |
| activation D2H | 1,865,048 bytes | 1,693,016 bytes | **-172,032 bytes** |
| mHC D2H | 737,280 bytes | 565,248 bytes | **-172,032 bytes** |
| activation H2D | 9,475,840 bytes | 9,475,840 bytes | 0 |
| attention | 66.014 ms | 66.614 ms | +0.600 ms |
| critical-device kernels | 125.406 ms | 131.064 ms | +5.658 ms variance |
| critical synchronization | 118.270 ms | 124.180 ms | +5.910 ms variance |
| routed CPU | 88.891 ms | 94.950 ms | +6.059 ms variance |
| total decode | 174.590 ms | 179.192 ms | +4.602 ms variance |

The two deterministic observations equal the declared prediction exactly:
one wait and one 4,096-element BF16 download disappear at each of 21 eligible
same-GPU boundaries. Kernel arithmetic, call counts, and launch counts are
unchanged. The total arm is slower because independent critical-kernel, wait,
and routed-CPU samples all rise by approximately the same 5–6 ms; no isolated
mechanism explains that as a regression. Therefore retain checkpoint 12 as an
exact ownership and serialization prerequisite, not as a throughput win.

Decision: accept and commit. Checklist item 1 is complete for the short-context
same-GPU attention activation chain. Cross-GPU normalized rows and the admitted
long-context sparse indexer remain deliberately host-visible. Checklist item 2
is not complete: 282 waits remain and no outer CUDA graph has been captured.

## Checkpoint 13 candidate: in-place current physical-page update

The post-checkpoint-12 wait audit accounts for all 282 decode waits:

| Owner | Waits |
|---|---:|
| current KV block promotion | 43 |
| QKV/compressor preparation | 43 |
| paged attention/output | 43 |
| standalone mHC boundaries after checkpoint-12 coalescing | 66 |
| router projection | 43 |
| CPU/shared-MoE collect | 43 |
| output head | 1 |

The 43 KV waits are a defect, not unavoidable graph work. Decode KV metrics
show exactly 43 misses, 43 promotions, and 6,428,672 host-to-device bytes. Each
append updates one host physical row, clears the resident `CudaBuffer`, then
`acquire_device` uploads the entire 148–150 KiB page again. This contradicts
the persistent-device admission contract and explains the previously
unattributed one-wait-per-layer term.

Checkpoint 13 retains the host physical page as the exact oracle but, when a
page already resides on a device, enqueues only its split row planes into that
stable buffer: 576 data bytes and 8 scale bytes for a main row (128+4 for an
index row). The source block remains alive until the immediately following
same-stream attention completion, so the update needs ordering but no separate
host wait. New pages still take the existing full promotion path; copy-on-write
and non-reference formats retain invalidation semantics.

At the current position only one sliding row is appended for each of 43 layers.
The prediction is therefore zero decode KV misses/promotions, exactly
`43 × 584 = 25,112` KV H2D bytes, 239 waits, and total activation H2D of
`9,475,840 - 6,428,672 + 25,112 = 3,072,280` bytes. This reduces the measured
mutable-page serialization term and copy-engine volume; it does not change GPU
kernel work, CPU-MoE work, or any other resource term.

The correctness gate remains exact aggregate/forward logit hashes, IDs, finite
logits, zero reads/callback failures, and unchanged 21,287,272,448-byte/device
use. Any decode promotion, count/byte discrepancy, hash mismatch, or page
ordering failure rejects the cut before CUDA graph capture begins.

The implementation adds a bounds-checked asynchronous `CudaBuffer` patch
operation and uses it only for already-resident reference-physical pages.
Host encoding still happens first, and new, evicted, copy-on-write, and legacy
format pages retain their prior promotion/invalidation paths. The live script
now binds all four predicted counters and prints the KV-cache phase explicitly.
Managed CUDA compilation, script syntax, `git diff --check`, and the full local
test gate pass; both CTest targets pass in 102.41 seconds. No live acceptance
or throughput claim has been made yet.

### Checkpoint 13 two-RTX acceptance

The live arm passes every binding gate. Aggregate raw-logit hash
`d8e0810706b18707` and forward hashes `a73dc2e74d875fc3` and
`0c7ed0f4eb38f2f7` remain exact; IDs remain `[30594,9790]`; all 258,560
logits are finite; decode reads and callback failures remain zero; and the
accepted memory plan is unchanged.

| Decode term | Checkpoint 12 | Checkpoint 13 | Change |
|---|---:|---:|---:|
| KV misses/promotions | 43 / 43 | 0 / 0 | **-43 / -43** |
| KV H2D | 6,428,672 bytes | 25,112 bytes | **-6,403,560 bytes** |
| activation H2D | 9,475,840 bytes | 3,072,280 bytes | **-6,403,560 bytes** |
| synchronization count | 282 | 239 | **-43** |
| critical synchronization | 124.180 ms | 110.362 ms | **-13.818 ms** |
| total decode | 179.192 ms | 164.503 ms | **-14.689 ms** |

Every deterministic counter equals the declared prediction. The page remains
resident, its 584-byte current row is visible to the accepted reader in command
order, and exact output is unchanged. Total decode is 6.08 token/s in this
single arm. Routed CPU time is also 18.999 ms lower than checkpoint 12, so the
entire wall-time difference cannot be attributed to this change; the binding
result is removal of the measured promotion volume and serialization.

Decision: accept and commit. The short-context physical KV path now satisfies
the persistent-device ownership contract. Resume checklist item 7 from the
remaining 239 waits; do not create another component probe.

## Checkpoint 14 candidate: combine same-device attention-to-FFN mHC boundary

Checkpoint 13 leaves 239 waits: 43 attention prepares, 43 paged/output
commands, 66 standalone mHC boundaries, 43 router projections, 43 MoE
collects, and one head. Critical synchronization is 110.362 ms against
81.768 ms routed CPU work, so serialized device completion remains the largest
measured term and is the resource this cut targets.

Every attention command leaves its exact BF16 branch in the persistent mHC
workspace. The runtime then immediately launches the accepted three-kernel mHC
transition for that layer's FFN and waits again. On the 21 layers whose
attention placement equals the mHC GPU, there is no intervening host consumer
or cross-device handoff. Checkpoint 14 appends that existing transition and its
unchanged 4,096-element BF16 FFN-input download to the attention command, then
performs one final wait. The 22 cross-device attention layers retain both
necessary handoff/target completions; their ownership is unchanged.

The prediction is exactly 218 waits (`239 - 21`) with unchanged activation
H2D/D2H, paged-attention bytes, mHC bytes, calls, kernel launches, hashes, and
memory. A count or byte discrepancy, numerical mismatch, cross-device command
error, or failure to remove exactly 21 waits rejects the cut. This extends the
fixed command chain; it is not yet the outer captured CUDA graph.

Managed CUDA compilation, script syntax, `git diff --check`, and the full test
gate pass; both CTest targets pass in 118.90 seconds. The candidate now requires
one short two-RTX exact-oracle execution. No performance claim is made before
that gate.

### Checkpoint 14 two-RTX acceptance

The live arm passes every binding deterministic gate. Aggregate raw-logit hash
`d8e0810706b18707`, forward hashes `a73dc2e74d875fc3` and
`0c7ed0f4eb38f2f7`, IDs `[30594,9790]`, finite logits, zero decode reads,
zero callback failures, memory admission, all byte counts, calls, and kernel
launches remain unchanged.

| Decode term | Checkpoint 13 | Checkpoint 14 | Change |
|---|---:|---:|---:|
| synchronization count | 239 | 218 | **-21** |
| activation H2D | 3,072,280 bytes | 3,072,280 bytes | 0 |
| activation D2H | 1,693,016 bytes | 1,693,016 bytes | 0 |
| attention | 64.412 ms | 64.329 ms | -0.083 ms |
| critical-device kernels | 118.202 ms | 132.689 ms | +14.487 ms variance |
| critical synchronization | 110.362 ms | 124.928 ms | +14.566 ms variance |
| routed CPU | 81.768 ms | 96.745 ms | +14.977 ms variance |
| total decode | 164.503 ms | 177.539 ms | +13.036 ms variance |

The wait reduction equals the declared topology prediction exactly. Attention
service is flat, while independent critical-kernel, synchronization, and routed
CPU samples all rise by approximately 14.5–15.0 ms; the wall-time regression
therefore cannot be assigned to the fused boundary. Retain checkpoint 14 as an
exact command-ownership prerequisite, not a throughput win. This arm is 5.63
token/s and does not satisfy the 10 token/s goal.

Decision: accept and commit. The remaining 218 waits are now 43 attention
prepares, 43 combined attention/output commands, 45 standalone mHC boundaries,
43 router projections, 43 MoE collects, and one output head. Continue with the
host-visible scheduling boundaries; do not build a component probe.

## Checkpoint 15 candidate: same-device mHC-to-router ownership

Checkpoint 14 measures 132.689 ms critical-device kernels, 124.928 ms critical
synchronization, and 96.745 ms routed CPU work in a 177.539 ms step. The
serialized completion term remains larger than routed CPU. The next adjacent
host-visible boundary is the router: mHC downloads an exact BF16 FFN input,
then the generic router uploads the same 4,096 F32 values, projects 256 raw-F32
logits, and waits.

For the 21 layers already combined on the mHC GPU, checkpoint 15 runs the
resident plain-BF16 router immediately after the accepted mHC transition. Its
BF16-input matvec decodes the resident BF16 values in the multiply loop; the
generic path first expands those same exactly representable values to F32 and
uploads them, so multiply order and four-warp reduction remain identical. The
unchanged raw-F32 logits join the existing download, while the BF16 layer-input
download remains for CPU experts. Host selection, bias, scoring, top-k order,
expert work, and the 22 cross-device layers remain unchanged.

Each removed generic router command carried a 4,096-element F32 upload and one
wait. The exact prediction is therefore 197 waits and activation H2D of
`3,072,280 - 21 × 4,096 × 4 = 2,728,216` bytes. Activation D2H, matmul calls,
kernel launches, mHC/paged-attention accounting, KV counters, hashes, and
memory must remain unchanged. Any discrepancy rejects the cut. This reduces
the measured serialized term and copy-engine volume; it does not reduce routed
CPU work.

Managed CUDA compilation, script syntax, `git diff --check`, and the full test
gate pass; both CTest targets pass in 103.33 seconds. The candidate now requires
one short two-RTX exact-oracle execution. No acceptance or throughput claim is
made before that gate.

The first live attempt was rejected before inference because the fused request
incorrectly required an FP8 router descriptor. Direct shard inspection shows
`layers.0.ffn.gate.weight` is plain BF16 `[256,4096]` with no scale tensor.
The 85.63-second arm produced no logits and accepts no runtime result. The
candidate now validates that actual descriptor and uses the bit-equivalent
resident-BF16 matvec above. The wait/byte prediction is unchanged; unlike an
expand-then-matvec repair, it adds no kernel launch.

After the correction, managed CUDA compilation, script syntax,
`git diff --check`, and the full test gate pass again; both CTest targets pass
in 139.25 seconds. The corrected candidate is ready for one replacement live
arm in the same deterministic result directory.

### Checkpoint 15 two-RTX acceptance

The corrected live arm passes every binding gate. Aggregate raw-logit hash
`d8e0810706b18707`, forward hashes `a73dc2e74d875fc3` and
`0c7ed0f4eb38f2f7`, IDs `[30594,9790]`, all finite logits, zero reads,
zero callback failures, memory admission, and all mHC/paged/MoE counts remain
exact.

| Decode term | Checkpoint 14 | Checkpoint 15 | Change |
|---|---:|---:|---:|
| synchronization count | 218 | 197 | **-21** |
| activation H2D | 3,072,280 bytes | 2,728,216 bytes | **-344,064 bytes** |
| activation D2H | 1,693,016 bytes | 1,693,016 bytes | 0 |
| attention | 64.329 ms | 65.567 ms | +1.238 ms variance |
| critical-device kernels | 132.689 ms | 133.845 ms | +1.156 ms variance |
| critical synchronization | 124.928 ms | 125.576 ms | +0.648 ms variance |
| routed CPU | 96.745 ms | 97.880 ms | +1.135 ms variance |
| total decode | 177.539 ms | 179.034 ms | +1.495 ms variance |

Both deterministic reductions equal the prediction exactly. The remaining
timings move together by 0.6–1.5 ms and do not establish a regression or win.
Retain checkpoint 15 as exact command ownership and copy-volume reduction, not
as a throughput improvement. This single arm is 5.59 token/s and remains below
the 10 token/s goal.

Decision: accept and commit. The remaining 197 waits are 43 attention prepares,
43 combined attention/output commands, 45 standalone mHC boundaries, 22
cross-device router projections, 43 MoE collects, and one output head. Continue
from this measured schedule without a component probe.

## Checkpoint 16 candidate: exact direct-bit E4M3 decode

Checkpoint 15 initially appeared to change the governing cost model. Its
179.034 ms decode step reports 133.845 ms in the field named critical-device
kernel service and 101.865 ms in the device-1 MoE kernel field. The accepted
paged attention and mHC kernels contribute 10.736 and 3.545 ms. This selected
the shared expert for the checkpoint-16 falsification below.

The retained shared expert decodes two 2,048-by-4,096 FP8 gate/up matrices and
one 4,096-by-2,048 FP8 down matrix per layer: 25,165,824 E4M3 values per layer,
or 1,082,130,432 values over 43 layers. The existing device helper reconstructs
each finite value with exponent arithmetic. Checkpoint 16 replaces that helper
with direct IEEE-754 bit construction. E4M3 normal values use exponent field
`e + 120` and fraction `m << 20`; subnormals normalize their three-bit
mantissa explicitly; signed zero and the format's NaN remain preserved.

This reduces arithmetic in the current GPU-kernel bottleneck. It does not
change FP8 bytes, scales, decoded F32 values, multiply/add association, kernel
launch count, H2D/D2H volume, CPU work, or memory. Integer bit operations replace
floating exponent/scaling work, so no other resource term is inflated beyond
the same instruction stream. The cheap falsifying experiment is the existing
short exact live integration arm, not a new component probe: fixed model setup
still dominates its one-token measurement, but it simultaneously validates all
1.08 billion production decodes and reports the target kernel term.

The binding correctness gate remains exact aggregate/forward hashes, IDs,
finite logits, zero reads/callback failures, all call/byte/wait counters, and
unchanged admission. The performance gate is at most 80.000 ms of maximum-device
MoE kernel service. That is below the prior observed 86.120--101.865 ms range
and saves at least 21.865 ms from checkpoint 15; it is a material mechanism
gate, not yet a claim that total decode is below 100 ms. Any exact mismatch,
counter/memory inflation, or result above 80 ms rejects and removes the code.

All 254 finite E4M3 encodings independently reproduce the prior decoder's
FP32 bit pattern; the two NaN encodings retain NaN. Managed CUDA compilation,
script syntax, `git diff --check`, and the full local test gate pass; both CTest
targets pass in 104.51 seconds. SM86 disassembly confirms that the hot decode
regions no longer issue integer-to-F32 conversions or `MUFU.EX2`; the gate/up
kernel's static SASS body falls from 456 to 376 instructions. The operator
script now prints the measured record even when the performance gate fails, so
one arm is sufficient to accept or reject the mechanism. The candidate requires one short two-RTX
exact/performance arm. It is not accepted and will not be committed unless the
declared 80 ms kernel gate also passes.

### Checkpoint 16 RTX rejection

The live arm is numerically and structurally exact: aggregate hash
`d8e0810706b18707`, forward hashes `a73dc2e74d875fc3` and
`0c7ed0f4eb38f2f7`, IDs `[30594,9790]`, 258,560 finite logits, zero reads and
callback failures, 197 waits, all byte/call counters, and admission remain
unchanged. The bit decoder therefore preserves the installed arithmetic.

It fails the predeclared performance gate:

| Decode term | Checkpoint 15 | Checkpoint 16 | Gate / interpretation |
|---|---:|---:|---|
| GPU shared-MoE kernels | 101.865 ms | **85.034 ms** | **fail: must be <=80.000 ms** |
| prior observed low | 86.120 ms | 85.034 ms | only -1.085 ms outside-arm comparison |
| critical-device kernels | 133.845 ms | 115.274 ms | still above the 100 ms target |
| critical synchronization | 125.576 ms | 106.426 ms | follows kernel completion |
| total decode | 179.034 ms | 158.824 ms | one arm; not an accepted speedup |

Although this arm is faster than checkpoint 15, its 85.034 ms GPU-MoE result
does not materially clear the prior 86.120--101.865 ms observed range and
misses the binding threshold by 5.034 ms. Do not launder the exact arithmetic
result into a performance acceptance. Decision: reject, remove the runtime
change, and make no commit for it. Retain the report and the script's
always-print failure record. The post-gate audit below reclassifies the measured
term before another mechanism is selected.

The post-gate timing audit corrects the last sentence: the existing
`deepseek_moe_kernel_seconds` interval begins after hidden upload on the main
stream and ends after the CPU host callback, routed-result H2D, wait for the
shared stream, and routed/shared join. It is an overlap envelope, not pure GPU
kernel service. The same interval is added to `critical_path_kernel_seconds`,
so neither 101.865 nor 85.034 ms establishes a GPU lower bound. Treating that
implausible label as literal was an instrumentation defect, not a datapoint.

Before another mechanism, checkpoint 17 will add detailed-timing-only events
around the production shared hidden quantization, gate/up, activation
quantization, down, and join kernels. This reproduces the real access pattern
and adds no component probe. One short exact arm is sufficient; the temporary
events will then be removed. Only the split can instantiate `W_gpu/B_gpu`
against routed CPU and select whether the next term is compute or serialized
handoff.

## Checkpoint 17 candidate: live MoE envelope attribution

The detailed-timing path now records six reusable events around the existing
production command without adding a kernel, transfer, allocation per layer, or
wait. It reports the sum of hidden/activation quantization, shared gate/up,
shared down, and routed/shared join separately. The legacy MoE `kernel` field
is deliberately retained as the measured overlap envelope for compatibility;
the report no longer describes it as pure GPU work.

The gate requires all four split terms to be positive and their sum to be less
than the envelope, plus the checkpoint-15 exact hashes, 197 waits, byte/call
counters, zero reads/failures, and unchanged memory admission. Event-record
overhead can inflate the instrumented arm, so its total decode time is not a
performance comparison. The primary result is the per-token split and
`argmax(max(shared GPU, routed CPU), serial handoff)`. After the arm the event
instrumentation is removed before any retained runtime candidate.

Managed CUDA compilation, script/JQ syntax, `git diff --check`, and the full
local gate pass; both CTest targets pass in 118.14 seconds. No model result or
resource selection has been made yet. One short two-RTX arm is the binding next
measurement.

### Checkpoint 17 RTX acceptance and removal

The arm passes every exact and structural gate: hashes and IDs remain exact,
all 258,560 logits are finite, reads/failures are zero, waits remain 197, and
all calls, bytes, KV counters, and memory admission are unchanged.

The live decode split is decisive:

| Term | Per token |
|---|---:|
| routed CPU gate/up | 62.195 ms |
| routed CPU down | 31.462 ms |
| routed CPU reduction | 3.792 ms |
| **routed CPU arithmetic** | **97.466 ms** |
| shared GPU quantization | 0.575 ms |
| shared GPU gate/up | 4.861 ms |
| shared GPU down | 2.668 ms |
| **shared GPU arithmetic** | **8.105 ms** |
| GPU routed/shared join | 0.433 ms |
| device-MoE execution envelope | 103.538 ms |
| complete graph MoE phase | 106.078 ms |
| host-visible attention phase | 64.928 ms |
| accepted paged-attention kernels | 10.640 ms |
| total decode | 179.045 ms |

The shared GPU branch is fully hidden under routed CPU. The current MoE
`argmax_r` is routed CPU arithmetic, not GPU shared compute. Its measured
3.449290752 GB routed payload runs at 35.390 GB/s aggregate
(17.695 GB/s/socket). The retained full-store floor is 87.879 ms, so even
perfectly restoring that floor saves only 9.587 ms. H2D is 2,728,216 bytes in
0.958 ms and D2H is 1,693,016 bytes in 1.289 ms; both links are serialized far
below rated bandwidth, but their present volumes are too small to explain the
step alone. NVMe decode remains zero.

The larger reducible term is serialized ownership around attention and the
activation chain: 64.928 ms remains host-visible while the accepted paged body
uses 10.640 ms of GPU service. That difference also contains projection work,
so it is not all removable overhead, but it is the term selected for the next
grouped fixed-buffer/graph checkpoint. Reducing routed CPU alone cannot reach
100 ms; GPU shared-kernel optimization was correctly rejected as a
non-bottleneck.

Decision: accept the measurement, remove all six temporary events and the four
temporary public counters, and retain no instrumentation in runtime code. The
tracked report and always-print operator script remain. Resume the grouped
remaining activation/graph ownership work; do not create another component
probe or another shared-FP8 optimization.

## Checkpoint 18 candidate: grouped cross-device activation ownership

The remaining 197 waits include two complementary cross-device gaps. After 22
attention layers, the source GPU waits to publish an 8,192-byte BF16 branch,
the mHC GPU waits again after its transition, and a generic router command
waits a third time. Across the preceding FFN-to-attention boundaries, 21
eligible cross-device mHC transitions wait before attention preparation waits
again. Same-device versions of both chains are already accepted.

Checkpoint 18 extends the same command ownership across devices using fixed
pinned staging and one reusable cross-device event per backend device. Source
D2H records completion; the target stream waits on that event before H2D and
continues with the already-accepted mHC/projection kernels. No CUDA API is
called from a host callback, no arithmetic changes, and no transfer is assumed
to be peer-capable. Attention-to-FFN includes resident mHC and BF16 router work
before its one target completion; FFN-to-attention includes the resident mHC
transition before the existing preparation command's one source completion.

The deterministic prediction is 132 waits: remove two waits for each of 22
attention-to-FFN boundaries and one for each of 21 FFN-to-attention boundaries,
`197 - 44 - 21`. The cross-device router now consumes the target mHC workspace's
resident BF16 layer input, removing 22 generic 16,384-byte FP32 router-input
uploads. Activation H2D must therefore be exactly
`2,728,216 - 22 * 16,384 = 2,367,768` bytes while activation D2H, KV/page
bytes, calls, and kernel launches remain unchanged. The fixed mHC workspace
grows by 1,024 bytes for router logits; admitted model weights and the reported
memory plan remain unchanged. The mechanism adds one fixed disabled-timing
event/device and no per-layer allocation. Exact hashes/IDs, finite logits, zero
reads/failures, all predicted byte/call counters, and the 132-wait count are
binding. Any mismatch, extra transfer, wrong command order, or failure to
remove all 65 waits rejects the grouped cut.

Implementation status before the RTX gate: both cross-device directions now
use fixed pinned staging plus a reusable disabled-timing event. The
attention-to-FFN command continues through the target mHC transition and its
resident BF16 router projection; the FFN-to-attention command performs the
target mHC transition before the source projection chain. The full retained
tree builds and `make check` passes (two tests, zero failures; 109.81 seconds).
This is local structural evidence only. Keep checkpoint 18 uncommitted until
the operator run clears the exact hashes, IDs, 132-wait, 2,367,768-byte H2D,
1,693,016-byte D2H, and unchanged call/kernel gates.

### Checkpoint 18 RTX result

The target run accepts the complete grouped cut. It retains aggregate logit
hash `d8e0810706b18707`, forward hashes `a73dc2e74d875fc3` and
`0c7ed0f4eb38f2f7`, generated IDs `[30594, 9790]`, all 258,560 finite logits,
zero decode checkpoint reads, zero KV misses/promotions, and zero host-callback
failures. All structural predictions are exact:

| metric | checkpoint 17 | checkpoint 18 | change |
|---|---:|---:|---:|
| synchronization calls | 197 | 132 | **-65 (-33.0%)** |
| activation H2D | 2,728,216 bytes | 2,367,768 bytes | **-360,448 bytes (-13.2%)** |
| activation D2H | 1,693,016 bytes | 1,693,016 bytes | unchanged |
| matmul calls | 384 | 384 | unchanged |
| paged-attention calls / kernels | 258 / 4,902 | 258 / 4,902 | unchanged |
| mHC calls / kernels | 522 / 1,560 | 522 / 1,560 | unchanged |
| critical-device kernels | 133.917 ms | 131.550 ms | -2.367 ms single-arm |
| critical synchronization | 125.297 ms | 124.101 ms | -1.196 ms single-arm |
| decode | 179.045 ms | 176.529 ms | -2.516 ms single-arm |

The memory plan remains 135,980,448 resident mHC bytes, 7,236,928 device-KV
bytes, and zero steady-state NVMe. The MoE contract remains 258 callbacks, zero
failures, 258 joins, 1,548 routed experts, 258 shared experts, and exactly
8,454,144/1,032 CUDA H2D/D2H bytes. The one-arm timing movement is within the
observed variability and is not a throughput claim.

Decision: accept and commit checkpoint 18. It removes exactly the selected 65
serial completions and 360,448 redundant bytes without changing arithmetic or
the runtime oracle. The remaining 132 waits are the next measured serial term;
resume at outer fixed-buffer graph ownership rather than another component
probe or shared-GPU kernel optimization.

## Checkpoint 19 candidate: reusable parallel attention preparation

Checkpoint 18 is retained as commit `49a1c6a`. Its target measurement also
falsifies synchronization-call removal by itself as the route to 10 tok/s: 65
removed waits change critical synchronization from 125.297 to 124.101 ms, only
1.196 ms. At the same observed 0.0184 ms/call, removing all 130 waits beyond a
two-rank final completion would save only about 2.4 ms and leave the 176.529 ms
step nowhere near 100 ms. CUDA-graph ownership remains an enabling requirement,
but its justification cannot be the raw wait count.

The checkpoint-18 decode cost model is:

| serial phase/resource | measured cost |
|---|---:|
| complete MoE graph phase | 104.070 ms |
| attention | 68.030 ms |
| attention preparation/Q-KV-compressor projections | 38.982 ms |
| attention score/output phase | 25.749 ms |
| output head | 1.747 ms |
| critical activation H2D | 2,367,768 bytes / 0.802 ms |
| critical activation D2H | 1,693,016 bytes / 0.921 ms |

MoE remains the largest phase, but its accepted full-store CPU floor is 87.879
ms; reaching that floor cannot remove the separate 68.030 ms attention phase.
The next mapped reference mechanism therefore reduces a required serial GPU
term as well: after the input activation is ready, Q, KV, the MLA compressor,
and the indexer compressor are independent. The reference schedules them on
the default stream plus three reusable auxiliary streams and joins them before
their consumers.

Checkpoint 19 reproduces that schedule inside `dsv4_prepare_attention`. It
adds three fixed non-blocking streams and one input-ready plus three completion
events per device. A separate fixed compressor-input region avoids mutating the
FP8 Q/KV input while auxiliary BF16 projections are in flight. Kernel count,
arithmetic order within each projection, H2D/D2H volume, 132 host waits, and
model admission remain unchanged; only independent GPU work overlaps. The
rollback conditions are any oracle/counter mismatch, dynamic per-layer
allocation, cross-device command-order failure, or failure to materially
reduce the 38.982 ms preparation term without inflating the complete step.

Implementation status before the RTX gate: the retained tree builds and
`make check` passes (two tests, zero failures; 124.74 seconds). The auxiliary
streams/events are created once per backend device, and the additional
compressor-input storage is part of the existing bounded reusable attention
workspace. No runtime result is accepted yet; the production-pattern arm must
clear the exact oracle and show whether SM86 can overlap these weight-only
kernels materially rather than merely interleave two already-saturating grids.

### Checkpoint 19 RTX result

The first production-pattern arm clears the complete correctness and structural
gate. It retains generated IDs `[30594, 9790]`, all 258,560 finite logits, zero
decode checkpoint reads, zero KV misses/promotions, and zero host-callback
failures. Activation traffic remains exactly 2,367,768/1,693,016 H2D/D2H bytes,
the wait count remains 132, and matmul, paged-attention, mHC, and MoE calls,
launches, and transfer counters are unchanged.

| metric | checkpoint 18 | checkpoint 19 arm 1 | change |
|---|---:|---:|---:|
| attention preparation | 38.982415 ms | 34.522091 ms | **-4.460324 ms (-11.44%)** |
| complete attention | 68.030033 ms | 65.468333 ms | **-2.561700 ms (-3.77%)** |
| attention KV | 2.508518 ms | 3.140162 ms | +0.631644 ms |
| critical-device kernels | 131.549984 ms | 129.861280 ms | -1.688704 ms |
| critical synchronization | 124.100700 ms | 120.711479 ms | -3.389221 ms |
| decode | 176.529172 ms | 175.863258 ms | -0.665914 ms |

Two preserved repetitions also clear the exact gate. All three arms retain
aggregate logit hash `d8e0810706b18707`, forward hashes
`a73dc2e74d875fc3` and `0c7ed0f4eb38f2f7`, generated IDs
`[30594, 9790]`, and every byte/call/launch/wait invariant. Their timings are:

| metric | arm 1 | arm 2 | arm 3 | median | change from checkpoint 18 |
|---|---:|---:|---:|---:|---:|
| attention preparation | 34.522091 ms | 33.706992 ms | 33.457027 ms | **33.706992 ms** | **-5.275423 ms (-13.53%)** |
| complete attention | 65.468333 ms | 63.835812 ms | 63.369918 ms | **63.835812 ms** | **-4.194221 ms (-6.17%)** |
| attention KV | 3.140162 ms | 2.896715 ms | 2.699195 ms | 2.896715 ms | +0.388197 ms |
| critical-device kernels | 129.861280 ms | 114.209056 ms | 112.334720 ms | **114.209056 ms** | -17.340928 ms |
| critical synchronization | 120.711479 ms | 105.612471 ms | 103.985577 ms | **105.612471 ms** | -18.488229 ms |
| decode | 175.863258 ms | 157.622756 ms | 154.998041 ms | **157.622756 ms** | -18.906416 ms (-10.71%) |

The checkpoint-18 comparator is a single exact arm, so the final column is not
an interleaved baseline throughput claim. It is sufficient for the candidate's
predeclared retention gate: the three candidate arms reproduce exactly, the
targeted preparation term is materially lower in every arm, complete attention
is lower in every arm, and decode does not regress. Decision: accept checkpoint
19. Retain the fixed auxiliary streams/events and the bounded separate
compressor input, then proceed to outer fixed-buffer graph capture rather than
another attention component probe.

## Checkpoint 20: device-input CPU-MoE callback prerequisite

Checkpoint 19 is retained as commit `be5cae4`. Its median decode operating
point instantiates the next cost model as follows:

| serial phase/resource | measured cost |
|---|---:|
| complete MoE graph phase | 89.460 ms |
| routed CPU arithmetic | 81.830 ms |
| complete attention | 63.836 ms |
| attention preparation | 33.707 ms |
| attention score/output | 26.468 ms |
| mHC pre/post | 0.410 ms |
| output head | 1.579 ms |
| complete decode | 157.623 ms |

The unclassified remainder after those host phase timers is only about 2.33
ms. That agrees with checkpoint 18's falsification of raw wait-count removal:
CUDA graph capture is required for fixed replay, but launch overhead alone
cannot close the 57.6 ms gap to 100 ms. The current `argmax_r` is routed CPU at
81.830 ms, while attention is an additional dependent 63.836 ms; neither may
be inflated to reduce the other.

A structural audit also finds that the live command chain is not yet legally
capturable. The attention-to-mHC command downloads the normalized BF16 layer
input and raw FP32 router logits, synchronizes, and returns both to the runtime.
Only then does the runtime select the six routed experts and construct a CPU
callback that closes over that host input and route. Capturing the existing
callback would therefore freeze a previous route, and removing its wait would
make its stack-owned context invalid. This is a correctness dependency, not a
CUDA API inconvenience.

Checkpoint 20 moves that complete boundary, not one isolated wait. The
attention command leaves the accepted normalized input and router logits in
the persistent mHC workspace. The immediately following MoE command enqueues
their exact BF16/FP32 D2H copies, then its host function decodes the BF16 input,
performs the unchanged router selection, resolves the resident tiled expert
pointers, and runs the accepted routed CPU arithmetic. The existing FP32 rank
partials return through the same ordered H2D and device join. No CUDA API is
called from the host function.

This relocates bytes rather than pretending to remove required CPU-MoE input:
total activation D2H/H2D, routed payload, arithmetic, shared-GPU work, and
admitted model memory must remain unchanged. One attention completion per
layer merges into the existing MoE collection, so the deterministic decode
wait prediction is `132 - 43 = 89`. Fixed pinned staging grows only by one
8,192-byte BF16 input plus 1,024 bytes of router logits; the existing 32,768-byte
rank-partial output remains required. The exact logit hashes, IDs, finite
count, zero reads/failures, all call/kernel counts, total activation bytes, and
89 waits are binding. Any route/hash mismatch, callback lifetime defect,
additional model/checkpoint traffic, or failure to remove exactly 43 waits
rejects the cut. After acceptance, the callback context can be made
process-lifetime and the still-host-owned compression/page metadata can be
addressed before outer graph capture.

Implementation status before the RTX gate: the complete boundary is compiled.
The attention command defers only when it has also produced the accepted mHC
transition and resident router logits. It preserves the physical-page and
weight leases until the following MoE collection, publishes any attention
failure to the same host node, and defers event timing until that collection so
instrumentation does not reintroduce the removed wait. The CPU callback decodes
the exact BF16 row, performs the existing token/hash or bias router selection,
resolves the six resident tiled expert triplets, and executes the unchanged
two-socket worker schedule. Shared-GPU work, rank-partial H2D, BF16 device join,
and mHC branch ownership are unchanged.

The production gate now requires 89 decode synchronizations, unchanged total
2,367,768/1,693,016 activation H2D/D2H bytes, 212,992 mHC D2H bytes and 396,460
MoE D2H bytes per forward, and 129 MoE D2H transfers per forward. The latter
two changes exactly relocate 43 × (8,192 + 1,024) bytes from the attention/mHC
completion into the host-node input. `make check` passes both tests with zero
failures in 125.88 seconds; script syntax and `git diff --check` also pass.
No NVIDIA device is exposed to the managed test environment, so the SM86 path
is not accepted until the operator production arm clears every exact gate.

The first RTX arm exposed an error in that predeclared byte equation, not in
the runtime. DeepSeek V4 has 256 router logits, so its raw FP32 route payload is
1,024 bytes rather than the incorrectly recorded 512 bytes. The observed MoE
D2H is exactly `43 × (8,192 + 1,024 + 4) = 396,460` bytes per forward and total
activation D2H remains exactly 1,693,016 bytes. The arm retains aggregate hash
`d8e0810706b18707`, forward hashes `a73dc2e74d875fc3` and
`0c7ed0f4eb38f2f7`, IDs `[30594, 9790]`, all finite logits, zero reads/failures,
89 waits, and every other declared counter. Correct the gate constant and
re-evaluate this saved generation; do not reload the model for an arithmetic
mistake in the checker.

### Checkpoint 20 RTX result

Re-evaluating the saved production generation with the corrected 1,024-byte
router constant accepts the complete boundary. The runtime retains aggregate
logit hash `d8e0810706b18707`, forward hashes `a73dc2e74d875fc3` and
`0c7ed0f4eb38f2f7`, generated IDs `[30594, 9790]`, all 258,560 finite logits,
zero decode checkpoint reads, zero KV misses/promotions, and zero callback
failures. Every deterministic ownership prediction now closes:

| metric | checkpoint 19 | checkpoint 20 | change |
|---|---:|---:|---:|
| synchronization calls | 132 | **89** | **-43 (-32.6%)** |
| activation H2D | 2,367,768 bytes | 2,367,768 bytes | unchanged |
| activation D2H | 1,693,016 bytes | 1,693,016 bytes | unchanged |
| mHC D2H per forward | 565,248 bytes | **212,992 bytes** | -352,256 bytes |
| MoE D2H per forward | 172 bytes | **396,460 bytes** | +396,288 bytes |
| MoE D2H transfers per forward | 43 | **129** | +86 |
| matmul calls | 384 | 384 | unchanged |
| paged-attention calls / kernels | 258 / 4,902 | 258 / 4,902 | unchanged |
| mHC calls / kernels | 522 / 1,560 | 522 / 1,560 | unchanged |
| MoE calls / kernels | 258 / 1,290 | 258 / 1,290 | unchanged |

The apparent extra 44,032 MoE D2H bytes beyond the 374,272 bytes relocated
from mHC are the 1,024-byte router rows that previously belonged to the
attention completion: `43 × 1,024 = 44,032`. Together those terms give the
required unchanged total activation D2H.

The one-arm decode is 169.511 ms, inside checkpoint 19's
154.998--175.863 ms candidate range, and is not a new throughput claim.
Attention's host phase reads 45.023 ms because its deferred GPU completion is
now observed by the MoE collection; comparing that value directly with the old
63.836 ms timer would launder an attribution change into a speedup. Likewise,
89 waits accumulate 129.807 ms because each MoE collection now observes the
complete dependent attention plus callback interval. This further confirms
that raw wait count is not the remaining performance term.

Decision: accept checkpoint 20. It makes route selection and routed-CPU input
part of the real stream-ordered host node with exact full-model behavior and
removes the required pre-callback host continuation. Land this result before
changing command lifetime. The remaining capture prerequisites are the
host-owned compressor/page update and stack-owned per-layer callback/lease
contexts; address them as the next grouped fixed-command checkpoint rather
than another wait-only cut.

## Checkpoint 21: fixed full-token command ownership

Checkpoint 20 is retained as commit `8b4d4e4`. Its one production arm measures
95.815 ms of routed CPU arithmetic, about 45.023 ms in the deferred attention
host phase, 129.562 ms of critical-device kernels, 129.807 ms observed across
89 completion waits, and 169.511 ms total decode. Routed CPU remains the
`argmax_r`; the accepted 87.879 ms full-store floor leaves only 7.936 ms of
unexplained CPU opportunity. Therefore this checkpoint does not claim that
removing API waits can by itself reach 100 ms. It targets the serial command-
ownership prerequisite that prevents the already-mapped dependent work from
being captured and replayed.

The deterministic wait inventory is 43 attention-preparation drains, 43 MoE
collections, and three step boundaries. The first implementation half removes
the need for the 43 preparation drains without changing compressor arithmetic.
The runtime reserves the one sliding row and optional completed compressed row
in host block-table metadata before issue. A CUDA host node then consumes the
exact BF16 KV and raw FP32 compressor projections, runs the existing CPU
compressor plus physical-page encoder, updates the canonical host page, and
fills pinned patch staging. Two stream-ordered H2D copies per row publish only
the accepted data and scale planes before paged attention reads them. No CUDA
API is called from the host node.

Device workspace, pinned upload, and pinned download storage are fixed at the
existing 1 MiB preparation ceiling on the first deferred command. This is a
correctness requirement: growing any of them after an earlier layer is queued
would free addresses still owned by that layer. Query/KV/compressor weight
leases and physical-page leases are retained through the matching MoE
collection. Sparse index selection is deliberately excluded from this path;
the production gate's 256-token admission never enables it, and the existing
synchronous exact path remains authoritative when it is active.

The memory ceiling is the existing 135,980,448-byte mHC reservation plus the
7,236,928-byte KV plan, two 1 MiB preparation workspaces and existing download
staging, 5,636,096 bytes of fixed per-command pinned upload/download storage
across two device states, plus the already-admitted attention/shared-weight
cache leases.
Roll back if exact hashes/IDs, physical-page bytes, or route semantics change;
if decode KV misses/promotions become nonzero; if required activation volume
grows beyond the row patches being relocated; or if this ownership cannot be
bounded by 43 commands.

The second implementation half replaces the single in-flight CPU-MoE callback
slot with 43 fixed command contexts. Each context owns its route, input, tiled
expert descriptors, and persistent shared-weight leases until the token
boundary. The CUDA backend likewise owns 43 callback commands, reuses fixed
device/pinned staging in stream order, and retains every resident-weight
implementation referenced by the queued chain. The main thread issues all 43
layer commands and calls the existing MoE collection once after the layer
loop. Only then does it account the reserved KV rows and release attention and
MoE leases.

The predeclared structural gate is therefore 89 to 4 decode synchronization
calls: the mHC begin boundary, mHC finish boundary, one final MoE collection,
and the output-head boundary. Removing 42 redundant four-byte per-layer error
downloads changes only the MoE-specific accounting: per-forward transfers move
from 129 to 87 and MoE D2H bytes from 396,460 to 396,292. Aggregate activation
D2H remains 1,693,016 bytes because collection error downloads were never part
of that counter. All hashes, token IDs, call/kernel counts, KV counters, and
other declared byte volumes remain binding.

`make check` passes both CTest targets in 114.82 seconds (all 266 enabled unit
cases pass; the two existing GPU/long-model cases remain skipped). The managed
environment exposes no NVIDIA device, so this is not an acceptance. It also
cannot establish a speed improvement: the deferred chain intentionally reuses
global CUDA timing events, which makes per-layer attention and MoE event
attribution invalid until collection. The next cost-model inputs are therefore
the reliable aggregate decode wall time and routed-CPU time from the production
RTX gate. Require exact output and the expected four-wait/copy counters before
folding the remaining three boundaries into the final collection.

### First production gate: rejected before measurement

The first production run in
`results/dsv4-reference-device-runtime/fixed-token-command-chain` was rejected
after 88.05 seconds with `DeepSeek fixed CPU/shared MoE collect: DeepSeek
CPU-MoE callback failed`. Admission succeeded, but `generation.json` is empty;
there is therefore no correctness or performance result. The callback context
contains no routed-arithmetic error, which localizes the failure to the
upstream attention-status ownership checked before callback invocation.

The specific lifetime defect is that the deferred MoE command keeps a pointer
to the four-byte attention status inside one reusable paged-attention pinned
download buffer. Once multiple layers are queued, a later attention command
can reuse that storage at a different layout offset before the earlier MoE
host node reads it. This is precisely the kind of stack/staging lifetime the
four-wait gate is intended to expose. Reject this arm and give each deferred
attention command a distinct pinned status slot through final collection;
retain the same four-wait, byte-volume, exactness, and memory gates.

The correction allocates 43 fixed pinned status slots per device state, selects
one before each deferred attention status D2H, and resets the slot cursor only
after final MoE collection. Cross-device event ordering still publishes the
value, but later attention layouts can no longer overwrite it. The callback
state now also retains the observed upstream value for a specific collection
error. `make check` passes both CTest targets after the correction in 145.62
seconds. Preserve the rejected directory and repeat the production gate in
`fixed-token-command-chain-r2`; GPU exactness remains pending.

### Second production gate: rejected on page metadata ownership

The corrected status slot made the second arm diagnostic rather than
ambiguous. `fixed-token-command-chain-r2` stopped after 85.53 seconds with
upstream status 2, the physical-page kernel's `candidate.page >= page_count`
condition. Host validation had already proved every candidate index was in
range, so the device result identifies another staging lifetime defect rather
than invalid runtime metadata: the per-device paged-attention upload buffer is
rewritten by the main thread while an earlier asynchronous metadata H2D still
owns it. Cross-device branch download staging has the same lifetime risk.
`generation.json` is again empty, so reject this arm without reporting timing.

Replace both deferred paged-attention upload and download storage with 43 fixed
per-command slots per device. The accepted contract caps candidates at 640 and
uses prepared queries, so a 32 KiB upload slot covers the worst 640 candidate
and page descriptors plus sinks/RoPE, and a 16 KiB download slot covers the
8 KiB cross-device branch and status. This adds 4,227,072 pinned bytes across
two devices and removes all CPU reuse of in-flight attention staging. Keep the
same exactness, four-wait, transfer-volume, and rollback gates.

The implemented correction also audits the preceding preparation command and
finds the same CPU-write hazard in its metadata upload. It now has a 16 KiB
fixed upload slot per command and device; its device workspace and callback-
consumed download staging remain safely stream ordered. Together preparation
and paged attention own 5,636,096 pinned bytes across the two devices. Deferred
paged attention now requires prepared queries, validates both fixed slot sizes,
and resets command cursors only after final collection. `make check` passes
both CTest targets in 114.85 seconds. Preserve both rejected arms and run the
unchanged production gate in `fixed-token-command-chain-r3`.

### Third production gate: computation completes, final lease audit rejects

`fixed-token-command-chain-r3` passes the two earlier staging failure points
and completes the full generation workload, but the final runtime audit rejects
the result with `DeepSeek generation completed with outstanding CUDA weight
leases` after 86.25 seconds. The fixed MoE contexts retained their three
shared-expert leases after token-boundary collection instead of retaining them
only until that collection. No JSON was emitted, so this is still not a
correctness or timing acceptance.

Clear each context's shared-weight pointers and release its three leases after
the final stream collection, alongside the already-released attention weights
and page leases. The backend retains its own implementation ownership until
the synchronized collection, so this does not shorten GPU lifetime or change
any transfer/call counter. Reacquisition on the next token should hit the
resident cache. Repeat the unchanged four-wait gate after the local suite.

The cleanup is implemented at the token-boundary collection and `make check`
passes both CTest targets in 132.06 seconds. The fourth arm is isolated in
`fixed-token-command-chain-r4`; exact GPU acceptance remains pending.

### Fourth production gate: accepted after correcting the evaluator

The fourth arm completes and emits exact output: token IDs `[30594, 9790]`,
aggregate logit trace hash `d8e0810706b18707`, raw forward hashes
`a73dc2e74d875fc3` and `0c7ed0f4eb38f2f7`, zero non-finite values, zero host
callback failures, and balanced 2,814/2,814 weight-lease acquire/release
counters. Decode has zero KV misses/promotions, 25,112 KV H2D/host-write bytes,
and exactly four synchronization calls. Per-forward MoE accounting is the
declared 87 D2H transfers and 396,292 D2H bytes.

The script initially printed `accepted: false` only because it incorrectly
predicted aggregate activation D2H would lose the 42 removed four-byte MoE
collection errors. That aggregate never counted collection error downloads;
the observed 1,693,016 bytes is therefore correct and unchanged. Re-evaluating
the saved `generation.json` with that one corrected constant passes the entire
predicate; no model rerun is needed.

This is structural acceptance, not a throughput win. Decode is 173.004 ms and
routed CPU is 818.4368 ms over six forwards, or 136.406 ms/forward, the current
`argmax_r`. Critical synchronization observes 124.180 ms because the four
waits now measure the dependent chain rather than create that work. Deferred
global CUDA events under-report component kernels and must not be used for the
next bottleneck decision. Retain checkpoint 21 and proceed to fold mHC begin,
mHC finish, and output-head ownership into the one final wait; that is a graph-
ownership completion, while the cost model predicts routed CPU must still be
optimized afterward to reach 100 ms.

### Checkpoint 22: grouped four-to-one ownership fold

Hypothesis: the remaining mHC begin, mHC finish, and vocabulary-head waits are
ownership boundaries, not required arithmetic boundaries. The exact begin can
leave its normalized input device-resident. At the other end, final mHC can
download into fixed pinned storage, a stream host node can reproduce the
existing FP64-accumulated four-copy coefficient reduction and BF16 RMSNorm,
and the resident head projection plus logits download can be queued before the
existing final MoE collection. That collection is then the sole host wait.

The primary structural metric is decode synchronization calls, required to
move from four to one. Correctness remains the exact token IDs, both raw logit
hashes and their aggregate hash, zero callback/non-finite failures, zero KV
misses/promotions, balanced leases, and all existing kernel/call counters. The
first prediction was that the begin-layer-input download could disappear. The
production topology falsifies that byte prediction: layer 0 is on the opposite
GPU from the persistent mHC workspace and this route uses the fixed pinned host
bridge rather than peer access. The 8,192-byte D2H/H2D volume must remain; only
its host wait can disappear. The corrected gates therefore retain 1,693,016
decode activation D2H bytes and 212,992 mHC D2H bytes/forward.

The fixed memory addition on the mHC device is a 16,384-byte FP32 head input,
a 517,120-byte FP32 logits output, and 566,272 pinned host bytes containing the
32,768-byte final BF16 residual, 16,384-byte reduced input, and logits. It is
allocated during warmup, before any token owns the stream. No buffer grows
while commands are queued. The memory ceiling remains the admitted weight/KV
plan plus this 1,099,776-byte fixed addition. Roll back if exact output or any
declared counter changes, if callback storage outlives its token, or if the
single final collection cannot prove completion of the head output.

The grouped implementation compiles and `make check` passes both CTest targets
in 116.13 seconds (all enabled unit cases pass; the existing hardware/long-model
cases remain skipped). This is not acceptance and not a throughput claim: the
environment has no NVIDIA device. The binding next gate is one short RTX
correctness arm in a new deterministic result directory. Only after exact
one-wait acceptance may the checkpoint be committed and the full-model
oracle/performance stages begin.

#### First production arm: exact output, two waits

`one-final-wait-r1` preserves token IDs `[30594, 9790]`, aggregate hash
`d8e0810706b18707`, raw hashes `a73dc2e74d875fc3` and `0c7ed0f4eb38f2f7`,
zero callback/non-finite failures, zero KV misses/promotions, and every declared
call/kernel/transfer counter. It reduces synchronization calls from four to
two. Decode is 168.373 ms, but this single arm is not a performance result.

The unchanged 1,693,016 activation D2H and 212,992 mHC D2H bytes/forward show
that the guarded device-only begin was ineligible: layer 0 and the persistent
mHC workspace are on different devices. The final mHC/host-head/GPU-head fold
did remove its two waits, so retain that half. Replace the initial synchronous
bridge with the existing fixed pinned D2H, cross-device-ready event, stream
wait, and H2D sequence. This preserves exact bytes while removing the host
continuation. Reject r1 structurally and repeat the exact gate; no component
probe or performance matrix is eligible yet. The correction compiles and
`make check` passes both CTest targets in 141.58 seconds.

#### Second production arm: accepted one-final-wait chain

`one-final-wait-r2` passes the complete script predicate. Token IDs remain
`[30594, 9790]`; aggregate logit hash remains `d8e0810706b18707`; raw hashes
remain `a73dc2e74d875fc3` and `0c7ed0f4eb38f2f7`; all 258,560 logits are
finite; host callback failures, KV misses, KV promotions, checkpoint reads and
swaps are zero. Weight leases are balanced at 2,816 acquire/release operations.

Decode records exactly one synchronization with 2,367,768 activation H2D and
1,693,016 activation D2H bytes. The initial non-peer bridge correctly retains
its 8,192-byte D2H/H2D volume without a host continuation. All 522 mHC calls,
1,560 mHC kernels, 258 paged-attention calls, 4,902 attention kernels, and MoE
transfer/call counters match the declared contract.

This accepts fixed-buffer ownership and the one-final-wait structure; it is not
a throughput win. The single arm reports 153.506 ms decode. Routed CPU is
610.191 ms across six forwards, or 101.698 ms/forward, and remains the measured
binding resource. Deferred reused CUDA events still make component GPU timing
invalid. Land the grouped correctness checkpoint before proceeding to the
full-model teacher-forcing/generation oracles and three comparable performance
repetitions.

### Checkpoint 23: full-model oracle closure and performance gate

Commit `233ed98` lands the accepted fixed-buffer chain. The r2 exact trace also
contains both required full-model oracle halves. Forward hash
`a73dc2e74d875fc3` is the terminal forward of the fixed teacher-forced prompt;
forward hash `0c7ed0f4eb38f2f7` is the next full 43-layer autoregressive forward
after generated token `30594`. Both and aggregate hash `d8e0810706b18707`
match the retained pre-integration Strata oracle, and generation remains
`[30594, 9790]`. The documented external stack already diverges independently
of this substitution, so laundering that known external mismatch into a new
rejection would not be a valid oracle. Step 3 is accepted.

The cheapest accepted full-system performance measurement is three fresh
process arms with 16 decode steps each. The fixed model setup is approximately
77 seconds and r2 predicts a 2.46-second measured window, a setup/window ratio
near 31:1 and roughly four minutes total. Repeating the one-step correctness
arm was rejected because more than 99% of each arm would be outside the
measurement and one decode sample cannot establish variance. A component probe
was rejected because the question is now the absolute complete-token target,
not an unmeasured mechanism.

At the current operating point r2 measures 153.506 ms/token. Its one final wait
observes 112.546 ms of the dependent stream; 40.960 ms remains outside that
wait. Its aggregate routed counter includes the five prefill forwards as well
as decode, so it is only a provisional whole-run signal and must not be divided
into a decode cost. Reused deferred CUDA events likewise make individual GPU
component durations invalid, so they are not summed into a false resource
model. The matrix records complete decode wall time, one wait per decode
forward, attention issue time, decode-scoped routed CPU, transfer counters, KV
misses/promotions, callback failures, checkpoint reads, and system state. If
the three-run median exceeds 100 ms, optimize only the newly measured
bottleneck; do not resume component-probe construction by default.

#### Three-arm absolute performance gate: rejected

The retained one-final-wait runtime is structurally clean in all three fresh
process arms: each has 16 decode steps and 16 final waits, zero decode
checkpoint reads, zero KV misses/promotions, and zero callback failures. It
does not meet the absolute target:

| Arm | Decode (ms/token) | Throughput (tok/s) | Final wait (ms/token) | Outside wait (ms/token) | Routed CPU (ms/forward) |
|---|---:|---:|---:|---:|---:|
| 1 | 157.893 | 6.333 | 130.479 | 27.414 | 90.693 |
| 2 | 149.508 | 6.689 | 126.578 | 22.930 | 82.697 |
| 3 | 157.549 | 6.347 | 130.404 | 27.145 | 90.814 |
| **median** | **157.549** | **6.347** | **130.404** | **27.145** | **90.693** |

The initial summary incorrectly divided whole-run routed CPU time, including
five prefill forwards, by the 21 total forwards. That printed 102.590
ms/forward. The binding decode-scoped counters under `phases.decode` give the
90.693 ms median above. The performance script is corrected to report only
decode-scoped routed work and to emit its gate/up, down, and reduction terms.
This changes attribution, not the rejected 157.549 ms/token headline.

At the median operating point the routed split is 57.146 ms gate/up, 27.465 ms
down, and 3.875 ms reduction. The 90.693 ms total is only 2.814 ms (3.2%) above
the production-pattern full-store floor of 87.879 ms; arm 2 is already faster
than that historical single-arm floor. Both NUMA nodes move together across
the three arms. There is therefore no newly measured CPU layout, placement, or
scheduler mismatch that justifies another component probe. Even eliminating
the entire 2.814 ms median gap would leave about 154.7 ms/token.

Instantiate the current cost model as `157.549 = 130.404 final wait + 27.145
host/issue work outside the wait`. Routed CPU contributes 90.693 ms inside the
dependent chain, leaving 39.710 ms of serialized GPU/handoff work in that wait
and 66.856 ms outside routed CPU across the full token. Attention host issue
time is 19.627 ms/token at the median arm; deferred reused CUDA events still do
not provide additive device-component timing. The reference finishes its
entire graph near 93.9 ms, so the current defect is serialization/launch
ownership around an already floor-adjacent CPU term, not routed arithmetic
volume. Per the governing cost model, measure the cheapest production-faithful
decomposition of that 66.856 ms remainder before selecting the next runtime
change.

The cheapest decomposition is already present in the production counters and
source. At the median arm, main-thread phase issue time is 19.627 ms attention,
3.345 ms MoE setup/issue, 3.458 ms mHC post issue, and 0.121 ms mHC pre issue;
together they account for 26.551 of the 27.145 ms outside the final wait.
Repository inspection finds no `cudaGraph*`, `cudaStreamBeginCapture`, or
`cudaGraphLaunch` call in the runtime/backend. The accepted checkpoint has
fixed-address stream ownership and one wait, but it does not yet have the
reference mechanism's reusable graph launch. Calling item 7 complete on the
basis of wait count alone was therefore incorrect; the authoritative checklist
returns it to yellow.

The next bounded hypothesis is reusable decode graph capture, not another
arithmetic probe. It targets the measured 26.551 ms host submission term and
may also reduce driver launch bubbles inside the 39.710 ms non-CPU portion of
the final wait. It cannot by itself guarantee 100 ms: its strict host-side
ceiling leaves roughly 130.4 ms/token, so the dependent GPU/handoff portion
must be remeasured after capture and only that new bottleneck may be optimized.
Correctness remains both exact logit hashes/token IDs, one final wait, zero
callback/KV/checkpoint failures, balanced leases, and unchanged transfer and
kernel work. Decode graph storage must fit inside the existing admitted
workspace reserve; reject capture if dynamic position/page metadata requires
silent stale replay, if CUDA cannot capture the cross-device event topology,
or if graph memory exceeds that ceiling. Page topology changes must invalidate
and rebuild explicitly rather than replay stale pointers.
