# DeepSeek V4 rank-local TP2 transport gate — 2026-08-09

Status: Stage 2 is accepted. The NCCL layer-reduction transport passes the
binding 86-reduction gate, and the corrected masked VocabParallelEmbedding
ownership arm passes exactness. Stage 3 rank-sharded loader and actual-format
fixture gates are now complete; stages 4–10 remain stopped. The stream-ordered
pinned-host transport is rejected as an implementation.

## Hypothesis and cheapest falsifier

Hypothesis: two rank-local DeepSeek V4 chains can replace the current
`mhc_slot`-centralized dependency chain if the required rank reductions fit
the measured non-callback budget.  The target term is the 62.986 ms measured
inter-callback dependent gap; the mechanism reduces centralized cross-device
bridges and serial ownership, while adding TP communication.

The cheapest falsifier was an isolated production-shaped reduction probe, not a
full-model arm.  It reproduces 43 dependent reductions at hidden width 4,096,
BF16 activation storage, FP32 rank-ordered association, two CUDA streams, D2H
copies, stream-ordered host callbacks, a persistent host coordinator, H2D
publication to both ranks, and per-rank event waits.  It makes no CUDA
allocation in the timed chain.  Fixed setup in the final arm was 367.828 ms
against an 11.275 ms measured 43-reduction window (32.549×); the rejected
cheaper experiment was any full-model implementation or long decode run.

## Operating point

The final arm used the requested runtime mapping:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2
runtime device 0: NVIDIA GeForce RTX 3090
runtime device 1: NVIDIA GeForce RTX 3090
batch: 1-equivalent reduction row
hidden width: 4096
reductions per chain: 43
dtype: BF16 inputs and published outputs
association: FP32, rank 0 then rank 1
P2P: capability 0 in both directions; no peer access used
```

## Measured transport

Three repetitions were run in the named `dsv4-tp2-transport-target2` tmux
session using `scripts/run_dsv4_rank_local_tp2_transport.sh`:

| repetition | 43-reduction wall time |
|---:|---:|
| 0 | 11.265 ms |
| 1 | 11.275 ms |
| 2 | 11.361 ms |
| median | **11.275 ms** |
| range | 11.265–11.361 ms |

The reduction output matched the exact host reference on both ranks.  A
separate target-device injection at rank 1, reduction 21 reported
`fail_closed_injection=rank1/reduction21 result=pass`; no reduced H2D
publication was issued for the failed reduction.

The measured-path accounting was:

| item | count/bytes |
|---|---:|
| fixed device buffers | 6 (partial, reduced, status × 2 ranks) |
| fixed pinned host slots | 129 (rank 0, rank 1, reduced × 43) |
| pre-created events | 172 |
| dependent stream waits | 84 |
| host callbacks | 86 |
| D2H | 704,856 bytes (8,192 payload + 4 status × 86) |
| H2D | 704,512 bytes (8,192 reduced payload × 86) |
| CUDA allocations in timed path | 0 |

The fixed raw storage footprint was 16,388 bytes per rank for the three device
buffers (8,192-byte partial + 8,192-byte reduced + 4-byte status), 32,776
bytes aggregate device data, and 1,073,280 bytes of pinned slot storage
(`sizeof(HostSlot)=8,320` × 129).  CUDA event and stream implementation
overhead is not included in those raw-byte totals.  This is a transport probe,
not a model admission: no full-model per-GPU VRAM or weight-lease result is
claimed.

The implementation is explicitly pinned-host staging.  It does not call
`cudaMemcpyPeerAsync`, enable P2P, use NCCL, or label host staging as peer
access.

## Historical serialized-arm calculation

This section preserves the original 11.275 ms result and its calculation.  It
is no longer a binding rejection because its 4.600 ms external term did not
cover the complete attention-plus-MoE transport scope.

The matched external reference reports approximately 20.6 ms of non-body
remainder, including 4.600 ms of TP reduction events.  Replacing only that
transport term with the measured staged transport gives:

```text
projected non-callback remainder
  = 20.6 - 4.600 + 11.275
  = 27.275 ms/forward (approximately)
```

The binding limit is at most 23 ms/forward.  This serialized implementation
therefore fails the gate by approximately 4.275 ms/forward.  That conclusion
is scoped to the serialized host-continuation transport; it does not reject
the later installed NCCL arm.  The full runtime stages remain unrun:

- rank-sharded checkpoint loading and fixtures: unrun;
- rank-local attention: unrun;
- rank-local shared/routed MoE: unrun;
- rank-local mHC chain: unrun;
- full correctness closure: unrun;
- eager candidate/control gate: unrun;
- CUDA graph reconsideration: ineligible and unrun;
- CPU parity work: unrun and remains a separate hypothesis.

The existing centralized topology, final-handoff, device-resident-KV, and graph
results remain unchanged and were not reopened.

## Revised Stage-2 transport arms

The serialized arm above used one host coordinator wait per reduction.  The
revised probe adds a fully stream-ordered pinned-host arm.  For each reduction
it queues:

```text
rank 0 D2H of one contiguous payload/status slot + rank-0 D2H event
rank 1 D2H of one contiguous payload/status slot
rank 1 waits on the rank-0 D2H event
rank 1 ordered host callback performs rank-0-then-rank-1 FP32 reduction
rank 1 records sum-ready event and queues its H2D
rank 0 waits on sum-ready and queues its H2D
```

There is no per-reduction host wait, no CUDA call from the callback, no
redundant same-stream output wait, and one final rank-0 synchronization boundary
that waits for rank 1 completion.  The callback writes only a fixed reduced
slot; injected failure sanitizes that slot, marks the run failed, and withholds
the final output validation.

The 43-reduction and 86-reduction arms were run in three interleaved
producer/control repetitions on the two RTX 3090s:

| arm | producer median/range | stream-ordered median/range | callback CPU median/range | stream minus producer median |
|---|---:|---:|---:|---:|
| MoE-only, 43 | 0.340 ms / 0.336–0.372 | 11.282 ms / 9.559–14.640 | 8.481 ms / 6.879–11.193 | **10.942 ms** |
| full layer, 86 | 0.656 ms / 0.656–0.658 | 26.653 ms / 19.893–30.156 | 19.224 ms / 14.512–24.174 | **25.997 ms** |

The 43-reduction result is effectively unchanged from the preserved serialized
11.275 ms median.  The full-layer scope is the required 86-reduction result;
its candidate transport term is 25.997 ms after removing the synthetic
producer chain.  The wide range is retained as evidence rather than treated
as a win.

Per-reduction stream-arm accounting is exact:

| scope | reductions | D2H copies/bytes | H2D copies/bytes | events | waits | callbacks |
|---|---:|---:|---:|---:|---:|---:|
| MoE-only | 43 | 86 / 704,856 | 86 / 704,512 | 87 | 87 | 43 |
| full layer | 86 | 172 / 1,409,712 | 172 / 1,409,024 | 173 | 173 | 86 |

Each D2H is one 8,196-byte contiguous payload/status copy per rank.  The
producer-only control has no payload copies or callbacks and reports one final
completion event/wait.  Fixed device data storage remains 16,388 bytes per
rank; the three maximum-size pinned arenas hold 2,146,560 raw bytes.

Exact output validation passed for both 43 and 86 arms.  The injected rank-1
reduction-21 failure passed for both scopes with callbacks completed, first
failure index 21, and external output withheld.

## Installed NCCL arm

The previous availability conclusion was incorrect because it did not inspect
the external reference venv.  The installed reference package is usable by a
standalone C/CUDA probe:

The interim status before measuring this arm was: “The stream-ordered
pinned-host transport is rejected. The Stage-2 transport hypothesis remains
open pending the installed NCCL arm.”

```text
package: nvidia-nccl-cu13 2.28.9
header:  /home/rodrigo/Developer/Lvllmds4-x/venv/lib/python3.12/site-packages/nvidia/nccl/include/nccl.h
library: /home/rodrigo/Developer/Lvllmds4-x/venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
header version: 2.28.9
ncclGetVersion(): 2.28.9 (encoded 22809)
loaded library: /home/rodrigo/Developer/Lvllmds4-x/venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
```

The probe is `apps/strata_dsv4_nccl_reduce_probe.cu`.  It uses only the NCCL C
API and CUDA runtime APIs; it does not use Python, PyTorch, vLLM, or a framework
runtime.  `STRATA_NCCL_INCLUDE_DIR` and `STRATA_NCCL_LIBRARY` are CMake cache
variables, and the launch script accepts `STRATA_NCCL_LIBRARY`; the reference
venv path is not production source.  The build was configured with:

```text
cmake -S . -B build-tp2-nccl -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_ENABLE_NCCL_PROBE=ON \
  -DSTRATA_NCCL_INCLUDE_DIR=/home/rodrigo/Developer/Lvllmds4-x/venv/lib/python3.12/site-packages/nvidia/nccl/include \
  -DSTRATA_NCCL_LIBRARY=/home/rodrigo/Developer/Lvllmds4-x/venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
```

NCCL initialization used `ncclGroupStart`/`ncclGroupEnd` for both rank
communicators.  Each reduction grouped both ranks' data all-reduces and both
ranks' fixed four-byte status all-reduces.  The status uses `ncclUint32`/max;
the reduced status is latched on-device across the dependent chain.  A local
injected error therefore still participates in every collective, poisons both
rank-local outputs, and withholds external output.

NCCL debug output selected the installed no-CUDA-P2P path:

```text
Check P2P Type ... isAllDirectP2p 0 directMode 1 isAllCudaP2p 0
MMAP allocated shareable host buffer /dev/shm/nccl-...
Channel ... via SHM/direct/direct
```

The log also reports `NCCL version 2.28.9+cuda13.0`; the selected rank-to-rank
transport is host shared memory with NCCL proxy/direct staging, not CUDA peer
access and not the pinned-host callback implementation.  The exact raw log is
`results/dsv4-rank-local-tp2/nccl-final/transport-probe.txt`; the preserved
earlier failed-status smoke output is in
`results/dsv4-rank-local-tp2/nccl/failure-smoke.txt`, and the corrected smoke
output is in `results/dsv4-rank-local-tp2/nccl/failure-smoke-latched.txt`.

### NCCL timings

The final run used one warmup and three interleaved repetitions for each dtype
and scope on only the two RTX 3090s.  `complete` includes the fixed producer,
NCCL data/status collectives, status latch/publication, and the single final
completion boundary.  `net` is the per-repetition complete-minus-producer
value, not a difference of independent extrema.

| dtype | scope | producer median/range | complete median/range | net transport median/range | exactness |
|---|---:|---:|---:|---:|---:|
| BF16 NCCL sum | 43 | 0.531 / 0.531–0.590 ms | 1.949 / 1.948–1.959 ms | **1.417 / 1.370–1.418 ms** | pass |
| BF16 NCCL sum | 86 | 1.049 / 1.030–1.051 ms | 3.838 / 3.835–3.865 ms | **2.808 / 2.785–2.814 ms** | pass |
| FP32 NCCL sum → BF16 | 43 | 0.721 / 0.720–0.733 ms | 2.147 / 2.140–2.181 ms | **1.427 / 1.419–1.448 ms** | pass |
| FP32 NCCL sum → BF16 | 86 | 1.425 / 1.419–1.447 ms | 4.264 / 4.242–4.331 ms | **2.845 / 2.817–2.884 ms** | pass |

All 12 measured candidate samples matched the exact rank-0-then-rank-1
FP32-oracle followed by BF16 publication, for both NCCL dtypes.  The gate uses
the FP32 86-reduction result, 2.845 ms/forward, because it directly preserves
the declared FP32 association contract; the BF16 result is reported but is not
chosen merely for speed.  The four injected-error cases (BF16/FP32 × 43/86)
all passed fail-closed validation with rank 1, reduction 21 injected.

NCCL has no host callbacks.  The probe reports zero application-issued D2H/H2D
copies and zero timed-path application allocations.  NCCL initialization and
the fixed device arenas are outside the measured window.  NCCL selected
`SHM/direct/direct`, with host shared-memory/proxy activity in its own
transport; physical NCCL transport bytes were not instrumented and are
therefore unknown, not zero.  Per layer scope and dtype, the exact logical
accounting is:

| dtype/scope | device arena/rank | NCCL data collectives | NCCL status collectives | NCCL group calls | NCCL data input/output bytes, all ranks | status input/output bytes, all ranks | final event/wait/sync | callbacks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16/43 | 57,356 B | 86 / 86 | 86 / 86 | 43 | 704,512 / 704,512 | 344 / 344 | 1 / 1 / 1 | 0 |
| BF16/86 | 57,356 B | 172 / 172 | 172 / 172 | 86 | 1,409,024 / 1,409,024 | 688 / 688 | 1 / 1 / 1 | 0 |
| FP32/43 | 57,356 B | 86 / 86 | 86 / 86 | 43 | 1,409,024 / 1,409,024 | 344 / 344 | 1 / 1 / 1 | 0 |
| FP32/86 | 57,356 B | 172 / 172 | 172 / 172 | 86 | 2,818,048 / 2,818,048 | 688 / 688 | 1 / 1 / 1 | 0 |

The arena contains three BF16 arrays, two FP32 arrays, and three four-byte
status words.  The table reports application API payloads submitted to NCCL,
not physical link/shared-memory traffic or peer-access bandwidth.

## NCCL ownership-extension arm

After the layer-only arm, the probe was extended without changing or replacing
the preserved 43- and 86-reduction measurements. The first endpoint extension
used two synthetic nonzero embedding inputs; its raw values and exactness
result are retained below as a non-production probe result. It was corrected
to model the actual `VocabParallelEmbedding` lookup, where exactly one rank
owns the selected token and the other rank emits zeros.

The corrected endpoint extension measures the two endpoint associations and
one combined communication chain:

```text
embedding:  local BF16 vocab partial -> NCCL BF16 all-reduce -> [4096] on both ranks
layers:     86 dependent hidden reductions (FP32 NCCL sum -> BF16 publication)
output:     local BF16 [64640] logits -> NCCL BF16 all-gather -> [129280] on both ranks
```

The output all-gather uses rank order, with rank 0 publishing token rows
`[0,64640)` and rank 1 publishing `[64640,129280)`. This is the matched
CUDA/reference contract: `ParallelLMHead` owns `[64640,4096]` BF16 rows per
rank and the selected logits path uses `use_all_gather=True`. The embedding
uses the same rank intervals for `embed.weight` and publishes one complete
BF16 `[4096]` row before the separate mHC expansion.

The corrected arm uses token 12,345 in rank 0's interval and token 76,985 in
rank 1's interval. For each case, the owning rank emits the BF16 row and the
non-owning rank emits exact zeros; validation checks both raw partials and the
all-reduced result. The extension uses fixed endpoint arenas, fixed rank-local
streams and communicators, grouped collectives for both ranks, zero timed-path
allocations, and one final completion boundary per arm. It keeps a fixed
four-byte status collective alongside each data collective. Endpoint and
combined injected failures participated in all required collectives, poisoned
the internal output, and passed fail-closed validation. The earlier
two-nonzero raw output remains at
`results/dsv4-rank-local-tp2/nccl-ownership-extension/transport-probe.txt`.
The corrected masked-lookup raw output, including NCCL `SHM/direct/direct`
selection, is preserved at
`results/dsv4-rank-local-tp2/nccl-ownership-masked/transport-probe.txt`.

The following are three interleaved repetitions on only the two RTX 3090s.
`producer` is the producer-only chain; `complete` includes the collective(s),
publication/status handling, and the single final completion boundary; `net`
is the paired complete-minus-producer value. All samples were exact.

The extension run also repeated the preserved layer arms. These values are
reported separately from, and do not replace, the accepted isolated layer-arm
result above:

| dtype | scope | producer median/range | complete median/range | net median/range | exactness |
|---|---:|---:|---:|---:|---:|
| BF16 NCCL sum | 43 | 0.648 / 0.645–0.679 ms | 2.032 / 2.000–2.043 ms | 1.385 / 1.322–1.397 ms | pass |
| BF16 NCCL sum | 86 | 1.220 / 1.210–1.222 ms | 4.042 / 4.027–4.048 ms | 2.826 / 2.807–2.832 ms | pass |
| FP32 NCCL sum → BF16 | 43 | 0.788 / 0.785–0.824 ms | 2.291 / 2.268–2.343 ms | 1.483 / 1.468–1.555 ms | pass |
| FP32 NCCL sum → BF16 | 86 | 1.636 / 1.607–1.656 ms | 4.541 / 4.462–4.614 ms | 2.906 / 2.856–2.958 ms | pass |

| earlier synthetic arm | producer median/range | complete median/range | net transport median/range | data/status collectives | data input/output, all ranks | exactness |
|---|---:|---:|---:|---:|---:|---:|
| embedding BF16 all-reduce | 0.030 / 0.030–0.032 ms | 0.066 / 0.065–0.066 ms | **0.035 / 0.034–0.036 ms** | 1 / 1 | 16,384 / 16,384 B | pass |
| output BF16 all-gather | 0.026 / 0.023–0.029 ms | 0.113 / 0.110–0.114 ms | **0.088 / 0.081–0.091 ms** | 1 / 1 | 258,560 / 517,120 B | pass |
| combined embedding + 86 FP32 layer + output | 1.629 / 1.591–1.674 ms | 4.644 / 4.640–4.760 ms | **3.053 / 3.011–3.085 ms** | 88 / 88 | 3,092,992 / 3,351,552 B | pass |

Those three rows are deliberately labeled as the earlier synthetic arm; they
are not the production ownership oracle. The corrected masked-lookup arm is:

| corrected masked arm | producer median/range | complete median/range | net transport median/range | data/status collectives | data input/output, all ranks | exactness |
|---|---:|---:|---:|---:|---:|---:|
| embedding, owner rank 0, token 12,345 | 0.034 / 0.030–0.034 ms | 0.074 / 0.070–0.075 ms | **0.040 / 0.040–0.041 ms** | 1 / 1 | 16,384 / 16,384 B | pass |
| embedding, owner rank 1, token 76,985 | 0.034 / 0.028–0.037 ms | 0.075 / 0.063–0.082 ms | **0.042 / 0.036–0.045 ms** | 1 / 1 | 16,384 / 16,384 B | pass |
| combined, owner rank 0, token 12,345 | 1.862 / 1.420–1.897 ms | 5.152 / 4.223–5.240 ms | **3.290 / 2.803–3.343 ms** | 88 / 88 | 3,092,992 / 3,351,552 B | pass |
| combined, owner rank 1, token 76,985 | 1.852 / 1.418–1.881 ms | 5.135 / 4.206–5.193 ms | **3.283 / 2.788–3.312 ms** | 88 / 88 | 3,092,992 / 3,351,552 B | pass |

The combined arm contains 86 FP32 hidden data collectives, one BF16 embedding
data collective, one BF16 output all-gather, and the corresponding 88 status
collectives. It therefore has 88 grouped collective calls and reports 704
status bytes in and out across both ranks. The fixed endpoint arena is
412,428 B/rank for either standalone endpoint arm and 469,784 B/rank for the
combined arm. The three-repetition per-sample records are in the raw output;
the corrected normal endpoint samples were:

```text
embedding owner0 net: 0.041, 0.040, 0.040 ms
embedding owner1 net: 0.045, 0.042, 0.036 ms
output net:    0.088, 0.081, 0.091 ms
combined owner0 net: 3.290, 3.343, 2.803 ms
combined owner1 net: 3.283, 3.312, 2.788 ms
```

The hidden 43/86 failure smoke remained passing after the extension for BF16
and FP32. The corrected embedding arm injected failures on both the owning
and non-owning rank for both token-owner cases; all four cases reported
`result=pass`. The existing output failure arm also remains passing. An
externally visible reduced embedding or logits vector was not accepted. The
exactness contract remains rank 0 then rank 1 FP32 association followed by
BF16 publication for hidden reductions, with BF16 NCCL association for
embedding and output. No callback or application-issued D2H/H2D copy is
present in these NCCL arms, and no timed-path application allocation occurred.
The corrected raw failure logs are
`results/dsv4-rank-local-tp2/nccl-ownership-masked/failure-embedding-owner0-rank0.txt`,
`failure-embedding-owner0-rank1.txt`, `failure-embedding-owner1-rank0.txt`,
and `failure-embedding-owner1-rank1.txt`. The earlier endpoint failure logs
remain under `results/dsv4-rank-local-tp2/nccl-ownership-extension/`.

NCCL selected `SHM/direct/direct` and may move data internally through host
shared memory/proxy buffers because CUDA P2P is unavailable. Application
issued D2H/H2D bytes are therefore zero, but physical NCCL transport bytes
were not instrumented and are recorded as unknown, not zero.

### Extension accounting by scope

The endpoint associations are outside the measured callback-chain scope. They
are reported separately and are not added to the 23 ms layer-chain gate.

| scope | dtype | fixed arena/rank | data collectives | status collectives | groups | logical data input/output, all ranks | status input/output, all ranks | callbacks | timed allocations |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| embedding association | BF16 | 412,428 B | 1 / 1 | 1 / 1 | 1 | 16,384 / 16,384 B | 8 / 8 B | 0 | 0 |
| output all-gather | BF16 | 412,428 B | 1 / 1 | 1 / 1 | 1 | 258,560 / 517,120 B | 8 / 8 B | 0 | 0 |
| combined decode communication | BF16 endpoints + FP32 layer | 469,784 B | 88 / 88 | 88 / 88 | 88 | 3,092,992 / 3,351,552 B | 704 / 704 B | 0 | 0 |

The layer-only 86-reduction accounting and the earlier pinned-host D2H/H2D
accounting remain unchanged above. The combined value is a communication
probe wall time, not a complete model decode result: it cannot be substituted
for the external callback-chain envelope without an external trace containing
the embedding and output associations at the same scope.

## Correct equal-scope gate

The external 4.600 ms measurement is retained as the MoE TP term only.  The
external attention `wo_b` TP term is not isolated in the available trace.  Let
`A` be that non-negative attention term.  Because the complete external TP
transport is within the approximately 20.6 ms non-body envelope:

```text
0 <= A <= 20.6 - 4.600 = 16.0 ms
external attention+MoE transport = A + 4.600

projected non-body
  = 20.6 - (A + 4.600) + 2.845
  = 18.845 - A
  = 2.845 .. 18.845 ms/forward
```

The NCCL 86-reduction candidate is 2.845 ms median, with a 2.817–2.884 ms
range, and is below the 23 ms limit even at the conservative upper projection
bound.  No external attention isolation is required to decide this bound.
The NCCL transport arm therefore passes the Stage-2 communication gate on
equal scope; this is not an acceptance of the full rank-local runtime.

The stream-ordered pinned-host transport remains rejected as an implementation
because its 86-reduction net median was 25.997 ms with a 19.893–30.156 ms
complete range and a 14.512–24.174 ms callback range.  That wide result is
preserved and is not promoted to a hardware constant.  Stages 3–10 remain
stopped for review; no rank-sharded runtime implementation, graph capture, or
CPU-kernel optimization was started.

## Updated decision

The stream-ordered pinned-host transport is rejected for this callback-heavy
implementation. The installed NCCL C-API arm passes the 86 layer-local
hidden-reduction gate provisionally: FP32 net median 2.845 ms, range
2.817–2.884 ms, exactness pass, no application D2H/H2D copies, and physical
NCCL bytes unmeasured. The equal-scope projected non-body bound remains
`2.845..18.845 ms/forward`, below the 23 ms limit even without isolating the
external attention term. The appended combined run independently measured the
same FP32 86-reduction arm at 2.906 ms net (2.856–2.958 ms), also below the
layer-chain gate; it does not replace the accepted isolated-arm value used for
the projection.

The endpoint and combined measurements close the Stage-2 communication
accounting for the corrected Stage-1 map, but they do not claim an end-to-end
decode improvement. Stage 2 is recorded as:

> The NCCL layer-reduction transport passes provisionally; full TP ownership
> and end-to-end communication accounting remain under review.

Stage 2 is accepted by review. Stage 3 explicit rank-sharded loading,
actual-format fixtures, and projected memory closure are complete and passing;
Stage 4 rank-local attention and stages 5–10 remain stopped. The Stage-3
descriptor/fixture evidence is recorded in experiment 0073 and raw output is
preserved at `results/dsv4-rank-local-tp2/rank-shard-fixtures/`.
