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
does not transfer. The landing gate is `<=125.0 ms/token` (`>=8.0 tok/s`),
derived from the ceiling-compliant centralized baseline of `151.155686 ms/forward`
(Experiment 0082), not carried across from `8.70 forward/s`.

**Stage 10** is the separate CPU gate/up, down, and weighted-reduction parity
work. It is out of scope here. The routed CPU body remains `argmax_r` at
`73.896784 ms` and `46.677143 GB/s`; closing the remaining distance to
`<=100 ms/forward` is its job, re-measured at that operating point.

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
3. Run exact live candidate selection on both ranks with the production
   algorithm — the Lightning-Indexer scoring, compressor state, and top-k
   selection that the M3 fixture chain replayed from `.d4r`.
4. Queue all 43 layers, then reach the single final completion.
5. Commit host-visible KV metadata **only after** both ranks' data and status
   collectives have succeeded.

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
KV, with the remainder as expert VRAM cache (Experiment 0082). Measured expert
cache use was only `4,073,858,048` and `5,594,845,696 B` against roughly
`20.9 GB` of capacity, so the centralized prefill cache is capped to free room
for the rank-local resident set — measured at `8,190,558,208 B/GPU` in 0092 —
rather than the two being summed at full capacity. Weights are never reloaded
during decode.

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
