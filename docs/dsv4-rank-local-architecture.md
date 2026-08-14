# DSV4 rank-local TP2 decode architecture

Canonical description of the rank-local tensor-parallel decode topology for
DeepSeek V4 on two GPUs. This is the contract document: it states what each
rank owns, when asynchronous host memory may be reused, how state is committed
or rolled back, and what admission must prove before the topology runs.

It describes the **production** topology on `feat/dsv4-rank-local-decode`.
Where a contract was established by an experiment, that experiment is cited.
File-level provenance is in `docs/dsv4-rank-local-extraction-manifest.md`.

## Scope boundary: M3 versus Stage 10

Two different things are easy to conflate, and most reporting errors in this
program came from conflating them.

**Milestone 3** (Experiment 0092) validated the *topology* against replayed
state. It measured `114.944312 ms/forward` median (`112.322810–116.790787`),
approximately `8.70 forward/s`, and passed the user-amended `<=115 ms` gate.
That window ran from chain submission through the terminal output head, with
per-layer attention state, candidate selection, and physical pages supplied by
`.d4c`/`.d4r`/`.d4m` fixtures.

**This landing** runs the same topology on live state: live candidate
selection, live transactional KV append, live page materialization, embedding,
and sampling. Those terms are additional to the M3 window, so the M3 constant
does not transfer. The strict landing target is `<=125.0 ms/token` (`>=8.0
tok/s`), derived from the ceiling-compliant centralized baseline of
`151.155686 ms/forward` (Experiment 0082), not carried across from `8.70
forward/s`. The user separately approved `125 +/- 2 ms` as a review boundary:
a median through `127.0 ms/token` may advance but is never relabelled as 8.0
tok/s when it misses the strict target.

**Stage 10** is the separate CPU gate/up, down, and weighted-reduction parity
work. It is out of scope here. In accepted production measurement the routed
CPU body remains `argmax_r` at `75.895 ms/token`. The external `72.889 ms`
callback body leaves only `3.006 ms` of navigation gap at this operating point,
and is not equal-scope proof. The historical `11--12.4 ms` estimate came from
the old `84.710 ms` centralized body and does not transfer after rank-local
CPU bandwidth improved. Stage 10 cannot by itself close either the current
`11.177 ms` short-context distance to 10 tok/s or the 1M index term.

`8.70 forward/s` is a fixture-scope figure. It is never reported as end-to-end
throughput.

## Ownership

### GPU and rank

Two CUDA devices, two runtime ranks, no P2P. At the stable operating point
`CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=1,2` map physical
devices 1 and 2 to runtime ranks 0 and 1.

Each rank owns one persistent CUDA context, one persistent NCCL communicator,
and its own stream ordering. Ranks are symmetric: neither is a coordinator.
Every layer is queued on both ranks and joined by collectives; there is no
per-layer host control continuation and no per-layer device-to-host readback.
A chain of 43 layers reaches exactly one final completion.

### Tensor sharding

Attention projections, the router, shared experts, routed experts, mHC weights,
and the output head are sharded across the two ranks by
`Dsv4RankShardDescriptor`. The sharded axis is described per tensor as
replicated (`-1`), contiguous rows (`0`), or strided columns (`1`).

A quantized weight and its scale are one atomic contract: either both are
described consistently or descriptor construction fails. Slices must respect
the block/group granularity of their encoding — 128 logical elements for FP8
blocks, 32 for FP4 groups, 1 for plain tensors — checked against the
payload-relative offset, not the absolute file offset. Rank-local extents are
read with `Dsv4CheckpointReader::read_slice_into`, so no complete sharded
tensor is ever materialized. Any weight or scale read failure clears both
payload vectors and returns errors; there is no partial load.

The output head is sharded `64640 x 2` locally and gathered to `129280 x 2`.

### NUMA and CPU

Routed experts execute on the host. Each rank owns one NUMA node's 24 CPUs as a
persistent `Dsv4HostMoeExecutor` pool over **one** transformed intermediate
shard, using the same addressed dispatch and same-node stealing policy as the
centralized 48-worker pool. The arithmetic, route, coefficients, precision, and
accumulation order are identical to the centralized path — only the ownership
partition differs.

Both ranks read the *same* transformed tiled NUMA arena. The rank-local
topology does not duplicate host routed-expert storage; host RSS is unchanged
in kind by enabling it.

Admission requires two usable NUMA nodes with at least 24 assigned CPUs per
rank.

## Asynchronous host-staging lifetimes

This is the contract most easily violated, and the one that produced the
program's hardest defect (Experiment 0091).

**Rule.** Any command that submits an asynchronous H2D and returns *without
synchronizing* must own a private pinned staging slot for the lifetime of that
copy. Only a fully synchronous host-visible path may share one buffer.

In `CudaBackend`, the predicate is `fixed_command_staging`:

- `dsv4_prepare_attention`: `host_deferred || device_only`
- `dsv4_paged_attention_to_mhc`: `defer_host_moe_input || rank_local`

Both are queued submissions. Each takes a slot indexed by the per-device
command counter, from a slab of `kDsv4FixedCommandCount = 43` slots:

```text
attention preparation upload   16 KiB/slot   43 slots =  688 KiB/device
attention upload               32 KiB/slot
attention download             16 KiB/slot
```

The slab is allocated once at first use — never on the queued path — and the
command counter is reset when the chain drains. A queued command whose slot
index would reach `kDsv4FixedCommandCount` is rejected as a command-order
error rather than wrapping.

### Why this rule exists

`dsv4_prepare_attention` uploads the layer's `q_norm[1024]`, `kv_norm[512]`,
and two RoPE vectors. Both norms are **per-layer checkpoint weights**. Each
queued layer wrote them with `memcpy`, issued an async H2D, and returned. With a
shared buffer, once the host outran the stream, the queued copies of earlier
layers read the norms of *later* layers.

The signature was decisive and is worth recognizing again: layer 1's
observation hash was `f233565ecfa96477` at queue depth 2 and
`384d71deb36191ee` at queue depth 43, on identical state and fixtures. An
already-enqueued layer changed because more layers were enqueued *after* it.
Device state is stream-ordered, so nothing device-side can do that; only a host
write to memory a queued copy has not yet read can.

The original fix (`7da38b7`) covered `dsv4_paged_attention_to_mhc` only. The
preparation command was left on shared staging, and a live page callback
happened to mask it — removing the callback silently reverted that path to the
shared buffer. `70b41fa` extended the rule to the preparation command.

### Testing lesson — production queue depth is mandatory

A one-layer and an adjacent two-layer callback-free prefix both **passed
against this defect**. A depth-bounded prefix cannot falsify a defect whose
trigger is the host outrunning the stream.

Any discriminator for a queued-ownership question must reach production queue
depth or drive the host/stream race directly. Stage 5 fixture tests therefore
run at full 43-layer depth, not at one- or two-command ownership granularity.

## Transactional state ownership

A decode token is a transaction across both ranks. Either every rank-local
mutation for that position commits, or none becomes visible.

### Replicated, not sharded

Sliding KV, compressed KV, learned-index state, the compressor, and mHC state
are **replicated on both GPUs**, not split. Both ranks therefore run the exact
live candidate selection independently and must agree; correctness builds verify
rank agreement explicitly. Replication is what allows either rank to serve
attention for any position without a cross-rank KV fetch on the decode path.

### Token transaction

1. Reserve one logical token position in both device slots. Nothing
   host-visible is published yet.
2. Compute and publish query, KV, and compressor rows in stream order, using
   fixed per-command staging or device-resident buffers.
3. Run exact live candidate selection with the production algorithm — the
   Lightning-Indexer scoring, compressor state, and top-k selection that the M3
   fixture chain replayed from `.d4r`.
4. Run each layer, carrying the reduced attention input forward; queue the
   terminal layer and reach the single final completion through the head.
5. Commit host-visible KV metadata **only after** both ranks' data and status
   collectives have succeeded.

Step 2 is one host-visible preparation per layer, not per rank. The rank-local
weight set carries no compressor weights and the executor's own preparation
computes no compressor projection, so the query rank, the key/value row and
both compressor projections come from a single `dsv4_prepare_attention` on the
slot that owns the centralized compressor weights. The result is replicated by
construction; computing it on each rank could only agree or diverge, and the
second outcome is a silent replica fault. The row is then encoded once and
written to each rank's device page with `update_buffer`, which keeps the
executor's own preparation device-only and costs no host synchronization.

Step 3 sits between the preparation and the layer call rather than inside it.
`Dsv4RankLocalLayerCall` takes candidates as an input, but sparse selection
needs this layer's query rank, which only exists once preparation has run. This
is the same constraint that makes the queued 43-layer chain a short-context
mode: a chain cannot select candidates for a layer whose query it has not yet
computed.

Replication is asserted, not assumed. The ranks' reduced attention input must
be identical between layers and their terminal hidden state identical before
publication. Both are hard errors: a divergence is a replica fault, and without
the first check one rank's state would silently drive both ranks' next
selection.

Before the first rank-local token of a sequence, centralized prefill state is
replicated to both ranks. Continuation and later chat turns transfer only newly
added or modified state.

### mHC

mHC state is seeded per rank at the start of a chain and evolves device-side
across the 43 layers with no host round trip. The transition router and the
next layer's attention mHC are queued device operations
(`dsv4_mhc_transition_router_device`, `dsv4_mhc_transition_next_device`). A
branch is reduced in FP32 and then either committed
(`dsv4_mhc_commit_reduced_branch`) or aborted (`dsv4_mhc_abort_branch`); an
aborted branch leaves no partial mHC mutation.

## Collectives

Two distinct collectives with different types and different reduction
semantics. They are not interchangeable.

| collective | payload | type | op | purpose |
|---|---|---|---|---|
| data | `kHidden = 4096` | `ncclFloat` | `ncclSum` | Combine rank partial hidden-state contributions |
| status | `1` | `ncclUint32` | `ncclMax` | Propagate failure to every rank |
| head | vocabulary shard | `ncclBfloat16` | `AllGather` | Publish `64640 x 2` local into `129280 x 2` gathered logits |

**Status semantics.** Zero is success. A failing rank sets its status word by
`cudaMemsetAsync(status, 1, 4)`, giving `0x01010101` = `16843009`. Because the
reduction is `MAX`, any nonzero status reaches every rank. The failure path
still *enters both collectives* — it does not skip them — so the state machine
closes symmetrically instead of deadlocking one rank against a peer that
already returned.

**Data semantics.** FP32 sum. The reduction is exact-order for the declared
contract; precision, router semantics, expert count, top-k, scoring,
normalization, routed scaling, and mHC association are unchanged from the
centralized path.

## BF16 publication contract

Rank publications cross the rank boundary as BF16 at the production
host-visible boundary. Raw FP32 is retained only where the target format
declares it: projection outputs, sinks, and compressor/indexer values.

The output head publishes `64640 x 2` local BF16 logits per rank, all-gathered
to `129280 x 2`. The gathered logits are byte-identical between the sequential
rank-local control and the queued chain — logits hash `343d766f3f5c0af3`,
next token `8806` at the M3 reference position (Experiments 0089, 0091, 0092).

Production copies one canonical logits vector for the existing sampler. The
duplicate local-logit readback, terminal residual, full-hidden, and hash
diagnostics that the fixture driver carried are **not** in the production hot
path; detailed phase instrumentation is available only behind an explicit
diagnostic setting.

## Failure and rollback

Exact mode either completes exact work or reports failure. There is no hidden
fallback, and — once rank-local admission has succeeded — no fallback to
centralized execution. Any rank, NCCL, KV, or correctness failure aborts the
token and returns an explicit error.

`Dsv4RankLocalFailure` enumerates the injectable failure points: attention
pre/post and MoE pre/post on either rank, plus the diagnostic
`MoeBeforeEnqueueRank1` case where rank 0 is enqueued while rank 1's
device-input metadata is still pending and must be cancelled.

On failure:

- both rank chains abort;
- every token-local KV, compressor, and mHC mutation is truncated;
- all outputs are withheld and zeroed — no partial token is ever published;
- the failed command is drained by a single owner; and
- the executor may be reseeded and reused. Post-failure reuse is exact, and is
  tested, not assumed.

`abort_chain()` provides fail-closed cleanup for a partially queued chain.

## Admission

Rank-local decode is explicit opt-in via `--rank-local-decode`. Centralized
decode remains the default and is unchanged.

The topology is **fail-closed before model loading**. Requesting it in a build
without NCCL is rejected at that point, not after weights are resident:
`STRATA_ENABLE_NCCL=OFF` keeps every normal build supported and unlinked
against NCCL, and leaves `STRATA_HAS_NCCL` undefined.

Admission must prove all of:

- exactly two CUDA devices;
- physical-device DSV4 KV mode;
- a supported DSV4 checkpoint with FP4 routed experts;
- two usable NUMA nodes, at least 24 assigned CPUs per rank;
- static weights, KV, NCCL buffers, and workspace below
  **`21,287,272,448 B` per GPU**; and
- host RSS below **`231,928,233,984 B`** (216 GiB).

Static weights, routed CPU storage, CUDA workspaces, KV capacity, NCCL buffers,
and head buffers are accounted **separately**. Admission is rejected if measured
or projected bounds fail.

### Coexistence with centralized prefill

Prefill runs centralized, so both resident sets must fit under the same per-GPU
ceiling. The centralized plan at the operating point is `9,204,991,520 B`
resident spine, `536,870,912 B` fixed workspace reserve, `7,236,928 B` physical
KV, with the remainder as expert VRAM cache (Experiment 0082). The centralized
prefill cache is capped to free room for the rank-local resident set — measured
at `8,190,558,208 B/GPU` in 0092 — rather than the two being summed at full
capacity. Weights are never reloaded during decode.

**Projected coexistence at the operating point.** Combining those measured
constants:

```text
rank-local sharded weights      8,190,558,208 B   (0092, measured)
centralized resident spine      9,204,991,520 B   (0082, planned)
fixed workspace reserve           536,870,912 B   (0082)
physical KV                         7,236,928 B   (0082)
                              -----------------
fixed subtotal                 17,939,657,568 B
per-GPU ceiling                21,287,272,448 B
remaining for cache + NCCL + head buffers
                                3,347,614,880 B
```

Measured centralized prefill cache use was `4,073,858,048 B` on device 0 and
`5,594,845,696 B` on device 1 (0082), both above what remains. Prefill will
therefore demand-upload the shortfall rather than serving it from cache.

This is accepted, not a defect: the expert cache is a prefill term, prefill is
outside the measured decode window, and admission caps the cache rather than
rejecting. It is also a **projection**, not a measurement — the 0092 figure was
taken by a harness that did not hold the centralized spine, so the two constants
come from different configurations. Stage 6 must report actual per-component
residency at the live operating point rather than relying on this sum. If the
fixed subtotal alone exceeds the ceiling in practice, admission rejects and
names the component responsible.

The `21,287,272,448 B/GPU` ceiling is enforceable because of the explicit VRAM
byte admission from Experiment 0082:

```text
applied_budget = min(free_bytes * vram_fraction, explicit_budget_bytes)
```

applied **before** arena reservation, not shaved afterwards. Experiment 0081
recorded the failure this prevents: a `.95` fractional plan alone reserved
`23,787,077,632` and `23,789,174,784 B/GPU`, about `2.5 GB/GPU` over the
ceiling, and was rejected on the memory gate.

## Steady-state invariants

Inside the measured decode window, all of the following are zero:

- checkpoint reads and decode I/O;
- workspace and weight allocations;
- page callbacks;
- KV misses, demand promotions, and cache evictions; and
- hidden fallbacks of any kind.

Per-layer host continuation is absent by construction: the chain reaches one
final completion.

## Cost model at this operating point

The governing model is `research/moe-tiered-memory-decode-optimization.md`:

```text
tau = max_r (W_r / B_r) + Sigma_serial
```

It is instantiated, not assumed. At the declared operating point:

| arm | `tau` | `argmax_r` (routed CPU) | `B_CPU` | `Sigma_serial` |
|---|---:|---:|---:|---:|
| centralized, ceiling-compliant (0082) | `151.155686 ms` | `84.710043 ms` | `40.718794 GB/s` | `66.445643 ms` |
| centralized, pre-cap (0081) | `149.099058 ms` | `84.171250 ms` | `40.979441 GB/s` | `64.927808 ms` |
| rank-local, M3 fixture scope (0092) | `114.944312 ms` | `73.896784 ms` | `46.677143 GB/s` | `41.047528 ms` |
| rank-local production, pre-correction | `126.675822 ms` | `75.895000 ms` | `45.448000 GB/s` | `51.921000 ms` |
| rank-local production, corrected | `111.177092 ms` | `75.895000 ms` | `45.448000 GB/s` | about `35.3 ms` |

The corrected row is the current binding short-context result and it meets the
strict `<=125.0 ms` / `>=8.0 tok/s` target at `8.994659 tok/s`, so it is not a
tolerance acceptance.

### The dependent envelope is attributed, not indeterminate

Experiment 0092 recorded the `argmax` *inside* the non-CPU envelope as
indeterminate. It is no longer. One Nsight Systems capture of the accepted
production arm, with an unprofiled control at the same shape to price the
`+2.6%` profiler perturbation, decomposes it over ten steady tokens without any
hot-path instrumentation:

```text
GPU kernel busy union, rank 0                   36.97 ms/token
GPU kernel busy union, rank 1                   42.07 ms/token
routed CPU body, as the 43 gaps before
  dsv4_host_moe_join_mhc_kernel            81.72 / 77.13 ms/token
page-materialization wait                   8.41 / 10.57 ms/token
terminal head and standalone projection      2.05 /  1.95 ms/token
remaining launch gaps                             about 6 ms/token
```

The CPU term recovered from the trace reproduces the independently measured
`75.895 ms` body, which is what validates the gap attribution. The envelope is
therefore **dependent GPU work serialized behind the CPU body by the layer
chain**, not an overlap defect: within a layer, attention precedes the router,
which precedes the routed CPU body, which precedes the join, and layer `L+1`
depends on layer `L`.

**The single-threaded-kernel defect.** The attribution located most of the
removable time inside the GPU busy term. Three attention-preparation kernels
were launched effectively single-threaded:

```text
dsv4_query_rank_norm      <<<1, 1>>>     240.6 us   43/token   10.345 ms/token
dsv4_key_value_norm_rope  <<<1, 1>>>     124.8 us   43/token    5.368 ms/token
dsv4_query_norm_rope      <<<64, 1>>>     88.2 us   43/token    3.791 ms/token
```

Together they cost more than every projection matmul on the layer
(`dsv4_rank_bf16_matmul`, `9.27 ms/token`). This is the same shape as CAP-12's
recorded `<<<1,1>>>` radix-merge defect, in a different path: a cost that barely
moves with the work is a constant, and a constant that large is a defect.

The declared contract pins the **order** of the FP64 accumulation, not where its
operands are read. The corrected kernels stage the dequantize/square and the
scale/store phases across the block and leave the ascending `__dadd_rn` chain on
one thread reading shared memory, which is bit-identical by construction. The
mechanism was screened standalone before any production edit — `5.92x`, zero bit
mismatches, on two GPU architectures — and integration is exact against the
32-token rank-local sequential oracle. Because these kernels live in
`dsv4_prepare_attention`, which both topologies share, centralized generated IDs
were checked and are bit-identical across all 32 tokens.

**Not every narrow launch is the same defect.** Enumerating the remaining
kernels by launch width rather than by cost gives two more candidates, and they
resolve differently:

- `dsv4_mhc_weighted_norm` (`<<<1,64>>>`, `1.442 ms/token`) is the same defect,
  but its FP32 reduction is *already* a tree — 64 accumulators, each summing
  four blocks in order, combined by a fixed xor pattern — so that shape is
  itself the contract. The correction preserves the accumulator count and every
  combination order, staging FP32 values in shared so the pinned reduction sees
  identical operands, and widens only the elementwise phases. Screened at
  `2.53x`, zero bit mismatches on both outputs.
- `dsv4_mhc_mix` (`<<<1,32>>>`, `0.547 ms/token`) is **not** this defect and was
  deliberately left alone. Its cost is a 20-iteration dependent shuffle chain
  over 16 lanes: inherently serial normalization whose iteration order is
  contract-bearing, and whose `6.4 us` is mostly launch and dependent-shuffle
  latency rather than parallelizable work. Widening it cannot help.

The distinction matters for reuse: the diagnosis is "a cost that does not scale
with the work", not "a small launch geometry". A serial algorithm with a small
launch is correctly sized; an elementwise phase with a small launch is not.

The `dsv4_mhc_weighted_norm` correction's `0.8 ms/token` predicted gain is below
the observed inter-repetition spread, and the confirming matrix measured
`111.721523 ms/token` against the prior `111.177092 ms`. **No end-to-end change
is claimed for it in either direction**; the standalone screen is its evidence,
and the matrix's role is exactness and gate confirmation.

The routed CPU body is `argmax_r` in every arm, over an exact
`3,449,290,752 B/forward` payload. The rank-local topology improves it by
partitioning the pool across two NUMA nodes, and improves `Sigma_serial` by
removing per-layer host continuation.

Two standing rules apply to every number above. Overlapping spans are never
summed into `tau` — the `126.3 ms` shared-collect and CUDA-synchronization
spans in 0081 overlap the CPU body and adding them would double count. And no
constant here transfers to another operating point: costs are `tau(L)`, functions
of context length, prompt, cache bound, and batch shape.

The `argmax` *inside* the rank-local `41.047528 ms` non-CPU envelope remains
indeterminate — attention, HBM, link, and NCCL service were never independently
timed (0092). No optimization mechanism may be selected from that envelope until
it is instantiated at the live operating point.

## Routing and selection contracts

These are declared semantics. They never change silently, and a change to any of
them is a different model, not an optimization.

**Router.** 256 experts, top-k 6. The learned selection bias changes **only**
top-k membership; it is not applied to the coefficients. Coefficients are
gathered from unbiased `sqrt(softplus(logit))`, normalized, then routed-scaled.

**Hash-routed prefix.** Layers 0-2 do not use the learned bias. They route by
the checkpoint's `layers.<n>.ffn.gate.tid2eid` token-to-expert table. That table
is made **fully resident at load**: reading one row per token would be checkpoint
I/O inside the decode window, which the steady-state gate forbids. Exactly one
routing membership is populated per layer view — bias or token-expert row, never
both.

**Selection.** Candidate selection runs live on both ranks over replicated
sliding, compressed, and learned-index state, using the production
Lightning-Indexer scoring and compressor. Both ranks must agree; correctness
builds verify rank agreement explicitly. Compressor and index values are raw
FP32 where the target format declares it — the BF16 boundary applies to
publications, not to these.

**DSpark.** Verification, the declared attention/compression layout,
shared-expert execution, mHC state, scoring functions, top-k normalization, and
routed scaling are preserved exactly as declared by the model manifest.

**Admission coupling.** The GPU Lightning Indexer is admitted only with the
exact compact block KV cache and is rejected at config validation otherwise,
rather than silently producing a different selection.

## Fixture methodology and its boundary

The `.d4c`/`.d4r`/`.d4m`/`.d4o` capture-and-replay transports are **not** in the
production binary. They remain immutable at `dsv4-m3-accepted` (`a31ac58`), and
the extraction manifest records why they were excluded.

The methodology they embody is worth keeping and reusing: capture at an accepted
boundary in the target format at production precision; compare against an
*independent* oracle rather than the implementation under test; compare exactly
by hash where the contract declares exactness; and preserve every failed arm as
evidence.

The boundary that matters for reading any M3 number: `.d4r` carries
`candidates`, `indexed_positions`, `index_compressor_values/scores`, `sinks`,
and materialized `pages` as fixture data. A measurement taken against it has not
measured live selection or live page materialization. **A fixture boundary must
never silently become the measurement boundary** — state which terms the fixture
supplies, and subtract them from any throughput claim.

## Measurement and evidence discipline

Binding for any performance claim about this topology.

- Three interleaved repetitions minimum; report the median **and** the observed
  range wherever a gate applies. Report every run.
- Separate initialization, prefill, transition, and decode.
- Record NVMe demand/prefetch bytes, host writes, H2D/D2H, cache
  hits/evictions, allocation and synchronization counts, RSS, and per-GPU VRAM.
- Measure useful-prefetch bytes, not prediction recall alone.
- **Never call a result a win when it is within observed run variance.**
- State the fixed-setup to measured-window ratio before launching a long run,
  and say which cheaper experiment was rejected.
- Check every headline number against the shape the design predicts, not only
  against the previous run. A number that contradicts the mechanism is a bug
  report.

The reusable template is 0081's three fresh controls and 0082's binding matrix.
This landing's strict target — `<=125.0 ms/token`, `>=8.0 tok/s` — was
re-derived from the ceiling-compliant centralized baseline rather than carried
across from M3's `8.70 forward/s` fixture figure. The separate `<=127.0 ms`
review boundary is reported as tolerance acceptance, never as strict 8 tok/s.


## Long context: 1M tokens with the same decode budget

Long context is a fundamental requirement, not an extension: a topology that
reaches its throughput gate only at a short context has not produced a usable
model. The relevant question is therefore whether `tau` holds at the model's
declared 1,048,576-token context and whether its exact residency fits. The
residency now fits under enforced admission; end-to-end `tau(1M)` remains
unmeasured because the original Step 4 sharding mechanism was falsified by the
component lower bound below. Short-context acceptance is not full-context
support.

### Measured admission plan

From the admission planner on `models/dsv4f`, not estimated:

| term | 256 context | 1,048,576 context |
|---|---:|---:|
| `kv_state_bytes` (includes index) | `19,105,280` | `4,082,533,760` |
| `index_state_bytes` | `0` | `732,512,256` |
| `required_host_bytes` | `159,189,843,804` | `163,253,272,284` |
| `zero_nvme_decode` | true | true |

Host residency at 1M is `163.25 GB` against the `231,928,233,984 B` ceiling, so
the host side is not the constraint. `index_state_bytes` is zero at 256 because
the sparse indexer only engages above `index_topk * ratio = 2048` active tokens.

The 1M `kv_state_bytes` value is the complete host admission account, not the
CUDA payload promoted to each rank. Its exact decomposition is:

| physical KV term | bytes |
|---|---:|
| promoted CUDA payload (sliding + compressed + learned index) | `4,050,137,088` |
| host block metadata | `16,258,432` |
| host block alignment | `3,932,160` |
| host physical block cache | `4,070,327,680` |
| compressor and index-compressor state | `12,206,080` |
| complete host `kv_state_bytes` | `4,082,533,760` |

Rank-local physical pages are one logical stream replicated on both GPUs, so
each device is admitted for the full `4,050,137,088`-byte promoted payload.
Neither the host metadata/alignment nor the compressor's host state is charged
again as CUDA page payload.

### Why `tau` is nearly context-independent

The bottleneck term does not grow at all. The routed CPU MoE body moves
`3,449,290,752 B/forward` of expert weights, which is a function of the model,
not of `L`. It is `argmax_r` at short context.

Two terms *do* grow, and an earlier revision of this document was wrong about
both. It claimed attention was "bounded by construction" at `640` rows per
layer and bracketed the indexer at `0.8-5.3 ms/token`. The corrected
production-page measurement puts the indexer at `68.482 ms/token`, thirteen
to eighty-six times the bracket, and the
boundedness claim holds for only half the layers.

**Attention is bounded on the ratio-4 layers only.** `compression_ratios`
alternates `4` and `128` from layer 2, so of 43 layers, 21 run at ratio 4, 20
at ratio 128, and 2 have no compressor. Only the ratio-4 layers engage the
sparse indexer and attend the fixed `512 + 128 = 640` rows. The ratio-128
layers attend their compressed history *densely*: `L/128` rows, which is `2` at
256 context and `8,192` at 1M. Attended rows per token therefore rise from
about `13 MB` to about `105 MB` — still small against HBM bandwidth, but the
per-candidate host work in `physical_paged_attention`'s `locate()` is a linear
block-table search per candidate and does not have that excuse.

**Indexer scoring, measured.** It must score all `L/ratio` compressed
candidates to select its top `512`: at 1M, `262,144` candidates per layer over
the 21 ratio-4 layers. `strata-dsv4-index-probe` on one 3090:

```text
                          baseline                  corrected
context    candidates   ms/layer   ms/token     ms/layer   ms/token
    4,096       1,024     0.268      5.619        0.165      3.463
   65,536      16,384     0.433      9.083        0.250      5.241
  262,144      65,536     0.996     20.910        0.519     10.892
1,048,576     262,144     3.261     68.482        1.623     34.092
```

The corrected column is `2.01x` at the 1M operating point and comes from two
exact kernel corrections, each screened before integration and each verified
against the scalar oracle at the full 262,144-candidate width:

- **Scoring, `2.15x`.** The kernel was not bound by what it looked bound by. A
  deliberately incorrect query-address-collapse variant priced query re-read
  traffic at only `32%` of the kernel, falsifying the L2-bound reading; the
  residual was a latency-hiding defect, issuing at about `0.28` instructions per
  cycle at full occupancy because each thread carried a single dependent
  `__fadd_rn` chain fed by one load. Scoring four rows per thread gives
  independent chains and amortizes the query load. Exactness holds because every
  accumulator still walks columns in ascending order; only which thread computes
  a row changed.
- **Radix pivot, `3.01x`.** Bin counts are integers, so the sums are
  order-independent and only access patterns were in question. The group scan
  read one uint32 per thread at 256 B stride, costing `8x` read amplification,
  and now reads 16 `uint4` per group. The final walk ran up to 64 dependent
  *global* loads on one thread and now stages its 64 bins coalesced into shared.

A caution worth carrying: the `uint4` read requires 16 B alignment, and the
histogram workspace region was reserved at `alignof(std::uint32_t)`. The
resulting misaligned vector loads left a sticky CUDA error that failed 21 tests
from one root cause, including a *valid* request being rejected — a symptom far
from its cause. A vector load added to a kernel silently inherits whatever
alignment its allocator happened to give it.

Substituting the measurement into the accepted landing runtime gives the
navigation lower bound:

```text
corrected short-context production      111.177092 ms/token
corrected replicated 1M indexer          34.092000 ms/token
unchanged-chain navigation total        145.269092 ms/token
```

The permitted complete index budget, derived from the corrected short path
rather than assumed, is `127.000000 - 111.177092 = 15.822908 ms/token` at the
review ceiling, or `13.822908 ms/token` against the strict target. A single
replicated indexer therefore still does not fit, and candidate sharding is
required rather than optional. With scoring halved across the two ranks and the
fixed-cost terms unchanged:

```text
scoring / 2                                  13.940000 ms/token
pivot, fixed at 65,536 bins per rank          0.990000 ms/token
histogram and remaining select, halved        1.250000 ms/token
outside (host launch and synchronization)     2.730000 ms/token
                                            ---------------------
sharded index projection                     18.910000 ms/token
corrected short path                        111.177092 ms/token
projected tau(1M)                           130.087092 ms/token
review ceiling                              127.000000 ms/token
remaining gap                                 3.087092 ms/token
```

The gap stood at `29.66 ms/token` when this landing was handed over. The largest
named remaining term is the `2.73 ms/token` of host launch and synchronization,
which the full-context design must remove in any case by moving selection inside
the queued chain.

This remains a **projection, not a measurement**. The exchange, the
deterministic merge, the ratio-128 dense attention growth and the linear host
`locate()` are unmeasured and are charged nowhere in it. No 1M throughput claim
follows from these component figures, and none is made.

Two further 1M terms remain **unmeasured and are not charged above**: the
ratio-128 dense attention growth to `8,192` rows across 20 layers, and
`physical_paged_attention`'s host `locate()`, which is linear per candidate.

**This misses both gates.** `125.0 ms/token` is the strict 8 tok/s budget and
`127.0 ms/token` is the explicit review ceiling. The index query is produced
by the preceding dependent state and must select positions before that layer's
attention, so this work cannot be moved under the same layer's later routed
CPU body without speculation.

Candidate sharding is exact in principle: a global top-512 candidate must be
inside its owning rank's local top 512, so exchanging the two local sets and
merging their 1,024 entries reproduces the scalar result. It is not sufficient
in this runtime. Nsight measures the 1M exact score kernel at `2.855265
ms/indexed layer`, or `59.960565 ms/token`; even perfect 2x score sharding with
free local selection, free exchange, and free merge leaves `29.980283
ms/token`. Added to the then-accepted `126.675822 ms/token` short path, that was
a strict lower bound of `156.656105 ms/token`, which falsified the original
sharding design before integration. That rejection stands on its own terms and
is not relabelled: sharding the *current* scorer is still not sufficient. On the
corrected `111.177092 ms/token` short path the same ideal shard gives
`141.157375 ms/token`, still above both gates.

The decisive bound is independent of the scorer entirely. The index's
non-scoring residue is about `8.52 ms/token`, so **even a free scorer** leaves
`111.177092 + 8.52 = 119.70 ms/token` — inside the review ceiling but only
because the short path was corrected first; against the pre-correction path the
same arithmetic gave `135.20 ms/token` and no scorer work of any size could have
passed. This is why the short-path correction had to precede any index work, and
why the two savings are reported as a joint requirement rather than stacked.

### Device residency at 1M

The production admission arm measures rank-specific weights and the initial
CUDA baseline rather than applying an aggregate spine value to both devices.
The controlling 1M component ledger is:

| per-device term | rank 0 bytes | rank 1 bytes |
|---|---:|---:|
| initial CUDA/device usage | `290,586,624` | `290,586,624` |
| measured rank-local weights | `8,283,797,760` | `7,675,623,680` |
| centralized pinned spine plus resident mHC | `4,209,838,496` | `4,995,026,432` |
| fixed workspace reserve | `268,435,456` | `268,435,456` |
| replicated KV/index CUDA payload capacity | `4,050,137,088` | `4,050,137,088` |
| NCCL and terminal-head reserve | `83,886,080` | `83,886,080` |
| admitted centralized expert cache | `5,361,896,800` | `5,184,882,944` |
| admitted total | `22,548,578,304` | `22,548,578,304` |

The cache cap is applied to the live centralized prefill cache and is also
bounded by the CUDA weight-arena suballocation left after the rank-local store.
The one-step 1M-capacity arm uses only `7,213,568` bytes of KV payload per rank
because pages are lazily committed; its final observed VRAM is
`18,397,396,992` bytes/device. That lower instantaneous number is not treated
as full-context physical residency. The admitted total is the binding capacity
contract, and Step 4 must fill and execute the long-context access pattern.

That gate was the **observed peak** of the centralized baseline, recorded as a
regression tripwire. It was never a hardware bound, and rank-local decode is
expected to exceed it, because it holds a second weight set the baseline never
had. The ceiling is therefore raised to `22,548,578,304 B` (21 GiB). The card
is 24 GiB and measures `23.561 GiB` free with nothing resident, leaving about
`2.6 GiB` for the CUDA context, cuBLAS workspaces, NCCL internals and
allocator fragmentation.

What 0082 validated at its budget was **zero decode weight and workspace
allocations** — that property, not the byte count, is what a higher ceiling
puts at risk, and Stage 6 must re-assert it rather than assume it survives.

This is headroom for integration, not the resolution. Releasing the spine
after prefill returns `9.2 GB` rather than the `0.76 GiB` the raise buys, and
remains the correct fix; it costs a reload before the next request's prefill,
roughly `0.8 s` at `12 GB/s`. The other alternative — sharding KV across ranks
instead of replicating it — would reintroduce a cross-rank fetch on the decode
path and is rejected for that reason.

Because the attended `512` compressed rows are data-dependent and scattered
across the full `4.05 GB` device payload, that whole set must be device-resident;
demand-paging it over PCIe would cost far more than the entire step budget.

### Selection is device work, and why it has to be

The learned index lives in physical KV as E4M3 rows with one F32 scale each.
The FP4 Lightning Indexer cannot read that layout — it expects FP4 E2M1 with
per-32 E8M0 scales — which is why config validation admits it only with block
KV. PhysicalDevice mode consequently fell through to *host scalar* scoring.

That path cannot serve the declared context, and arithmetic settles it without
a run: `262,144` candidates over `64` heads of `128` dimensions across 21
layers is `1.85e11 FLOP/token`. Forty-eight cores at a generous `20 GFLOP/s`
each is a `192 ms/token` floor — past the whole budget before any attention or
MoE work exists.

`CudaBackend::dsv4_physical_lightning_index` supplies the device path. Its
output is bit identical to `dsv4_index_scores_f32` followed by
`dsv4_index_topk_f32`: the dot product keeps the reference's sequential FP32
order under explicit `__fmul_rn`/`__fadd_rn`, so the compiler cannot contract
it into an fma and reassociate. That choice costs the tensor-core path, which
would reassociate the accumulation and could change which rows are selected
near the score threshold — a semantic change, not a rounding one. The cost of
exactness is reported above rather than traded away silently.

Selection remains **outside the queued chain**. `index_select` needs a
host-visible `query_rank` to project index queries, and in the queued path that
value exists only inside the CUDA host callback, where calling a CUDA API is
forbidden. The 21 indexed layers therefore take the synchronous
`attention_prepared` path and the other 22 keep the deferred page update. This
is a real overlap cost, recorded rather than hidden.

### The 43-layer chain is a short-context mode, not the general one

`Dsv4RankLocalLayerExecutor` offers two shapes: `run()` executes one layer
against live mHC state, and `enqueue_chain_layer`/`finish_chain` queue many
layers and drain once. M3's `114.944 ms` was the chain.

**The chain cannot serve a context where the indexer engages.**
`Dsv4RankLocalLayerCall` takes `candidates` as an *input* span, so every
layer's candidates must be known before the drain. For a ratio-4 layer the
candidates come from `index_select`, which projects index queries from
`query_rank = wq_a x input`, and that input is the previous layer's output —
which does not exist until the drain has already run. All 43 must be known
before the drain; 21 cannot be known until after it.

This is why M3 replayed `indexed_positions` from `.d4r`. That was not a fixture
convenience, it was the only way a full chain could be formed at all, and it is
a further reason `8.70 forward/s` must never be quoted as end-to-end
throughput.

The consequence is that the two operating points this landing must serve run
**different execution shapes**:

```text
active tokens <= 2048   indexer disengaged   one 43-layer chain, one drain
active tokens >  2048   indexer engaged      layer-by-layer run(), ~22 drains
```

`2048` is `index_topk * ratio`; below it the runtime attends the compressed set
densely and never selects. The 256-context performance gate therefore measures
the chain, and 1M does not measure the same mechanism at all. Stage 6 must
report them as two measurements, and must not present a chain figure as
evidence for the long-context path.

Closing the gap requires index query projection, scoring and top-k to run as
device work *inside* the queued chain, so no host round trip interrupts it.
`CudaBackend::dsv4_physical_lightning_index` is device work but is dispatched
from the host and synchronizes its stream, so it does not yet satisfy that.

### What long context does not fix

Prefill at 1M is a separate and much larger cost, outside the decode window and
outside this landing's gate. Nothing here claims a 1M prefill budget.

## Capability index

The thirteen program capabilities, and where each is specified here. Full
disposition, provenance, evidence, and reuse guidance for every one is in
`docs/dsv4-rank-local-extraction-manifest.md`.

| id | capability | specified in this document |
|---|---|---|
| CAP-01 | Cost model and bottleneck procedure | Cost model at this operating point |
| CAP-02 | Memory admission and VRAM/RSS accounting | Admission; Coexistence with centralized prefill |
| CAP-03 | Rank-sharded checkpoint loading | Ownership → Tensor sharding |
| CAP-04 | NUMA-bound CPU pools and resident expert storage | Ownership → NUMA and CPU |
| CAP-05 | NCCL FP32/status reductions and failure closure | Collectives; Failure and rollback |
| CAP-06 | Device-resident state and one-completion scheduling | Ownership → GPU and rank; Transactional state ownership |
| CAP-07 | Replay capture and exact-oracle methodology | Fixture methodology and its boundary |
| CAP-08 | Physical KV-page infrastructure | Transactional state ownership → Replicated, not sharded; Token transaction |
| CAP-09 | Rank-local attention kernels | Asynchronous host-staging lifetimes (incl. the queue-depth rule) |
| CAP-10 | Routed/shared MoE execution | Routing and selection contracts |
| CAP-11 | mHC transitions | Transactional state ownership → mHC |
| CAP-12 | Lightning Index / compressor / DSpark handling | Routing and selection contracts |
| CAP-13 | End-to-end benchmark and resource ledger | Measurement and evidence discipline |
