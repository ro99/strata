# Experiment 0091: M3 callback-free queued attention-preparation staging

Date: 2026-08-12
Branch: `exp/dsv4-a2-ownership-screen`
Base: `c54697e`
Disposition: **0090 CAUSE IDENTIFIED AND CLOSED; M3 CALLBACK-FREE 43-LAYER
CORRECTNESS PASS; performance gates remain unmeasured and M4 remains blocked**

## Question and kill rule

Experiment 0090 rejected the M3 implementation because the first callback-free
43-layer warm-up mismatched the sequential rank-local control, and explicitly
recorded that the cause was unproven: neither prepared-query reuse nor physical
page publication had been demonstrated. This experiment asked one bounded
question — what actually differs when the live attention page callback is
removed — and required that the answer be exhibited by measurement before any
correction was accepted.

The predeclared ladder was binding and ordered, per the 0090 next-action rule:

1. one callback-free layer must be exact;
2. an adjacent two-layer callback-free prefix must be exact; and
3. only then may the 43-layer callback-free chain run.

Any mismatch stops the arm. No latency, bandwidth, or `tau` value may be
reported from any arm in this experiment, passing or failing.

## Attribution

`CudaBackend::dsv4_prepare_attention` selected its pinned host upload buffer on
`request.host_callback != nullptr`, not on whether the command was queued:

- with a page callback (`host_deferred`), the upload used the fixed
  per-command slab `dsv4_attention_prepare_fixed_host_upload`, indexed by
  `dsv4_attention_prepare_host_command_count`, 43 slots of 16 KiB per device;
- without one (`device_only`), it fell back to the single shared
  `dsv4_attention_prepare_host_upload`.

The upload carries this layer's `q_norm[1024]`, `kv_norm[512]`, and the two
RoPE vectors, and both `q_norm` and `kv_norm` are per-layer checkpoint weights.
Each queued layer writes them with `std::memcpy`, issues an async H2D from that
pinned buffer, and returns without synchronizing. Once the host outruns the
stream, the queued copies of earlier layers read the norms of later layers.

This is the same defect commit `7da38b7` corrected for
`dsv4_paged_attention_to_mhc`, where `fixed_command_staging =
defer_host_moe_input || rank_local` moved rank-local commands onto fixed
per-command slots and removed the post-layer21 barrier. The preparation command
was never included in that correction, so the callback was masking it: removing
the callback silently reverted the queued path to shared staging.

The two mechanisms 0090 named are not on the differential path. In
`host_deferred` mode `dsv4_prepare_attention` returns before the synchronous
download block, so it never writes the caller's `query_rank` / `key_value`
spans at all. The executor's reusable `prepared_query` / `prepared_key_value`
storage is therefore never populated in the callback-backed chain, which is why
0090's event-only guard over that storage could not change the result. That
arm's negative finding stands and is now explained rather than reclassified.

## Discriminator and results

Three preserved arms, all at the fixed position-104 two-RTX-3090 operating
point with `CUDA_VISIBLE_DEVICES=1,2`. Every arm first reproduces the accepted
callback-backed 0088/0089 result in the same process, then materializes nothing
new: the physical row 104 is already resident in each arm's page buffers from
those callback-backed arms, so the callback-free arms read a materialized page.

```text
callback-free-r1 combined.log SHA-256 77eed78e1405ebbeb806b52a576ba19d1594c1392f83eabf4f1f6a443eece708
callback-free-r1 source.diff  SHA-256 fcb2b8b561ccfa2059517d73a6e5211f005becf85f84d207cd6d1944d449cf2a
callback-free-r1 exit status  1   wall 87 s   staging: shared (unfixed)
callback-free-r2 combined.log SHA-256 5e3ad86cdef92dfa102412d66343ce51c4ca86f06da1616d2cf25eccd4c89b79
callback-free-r2 source.diff  SHA-256 5102a64527be2dc0191048d7b4e8526301bbf61489e71acdd4c2c6a6da7eb6e2
callback-free-r2 exit status  1   wall 95 s   staging: shared (unfixed)
callback-free-r3 combined.log SHA-256 e2a0a2aedd487e5dc2f807d22b6ea2db5bf698a510666baca5c3fc16bcfce709
callback-free-r3 source.diff  SHA-256 381bc3567e459582dd1e07a36fa2965602c75d8fc1956cd4d4ad45c75cd73f6c
callback-free-r3 exit status  0   wall 103 s  staging: fixed per command
```

`callback-free-r1` is the first negative arm. `callback-free-r2` reruns the
same unfixed binary after the harness was extended to report every layer's
observation hash instead of stopping at the first mismatch; it is the binding
negative. `callback-free-r3` is the binding positive and differs from r2 only
by the backend staging correction.

### The queue-depth signature

Per-layer observation hashes, callback-free, against the sequential rank-local
control:

| layer | prefix 1 | prefix 2 | 43 unfixed (r2) | 43 fixed (r3) | control |
|---|---|---|---|---|---|
| 0 | `8863ffd31b978def` | `8863ffd31b978def` | `8863ffd31b978def` | `8863ffd31b978def` | `8863ffd31b978def` |
| 1 | — | `f233565ecfa96477` | **`384d71deb36191ee`** | `f233565ecfa96477` | `f233565ecfa96477` |
| 2 | — | — | **`ec9a78a397e039de`** | `c25ee19900c97337` | `c25ee19900c97337` |

Layer 0 is exact in every arm, fixed or not. Layer 1 is exact at queue depth 2
and wrong at queue depth 43, on identical state and identical fixtures. The
first failing layer in both unfixed 43-layer arms is layer 1, rank 0, index 0
of the encoded FFN input.

That is the decisive result. A layer that was already enqueued produced a
different value solely because more layers were enqueued after it. Device state
is stream-ordered, so no device-side workspace reuse can change an earlier
command retroactively; only a host write to memory that a queued asynchronous
copy has not yet read can. The only such host memory on the differential path
is the shared preparation upload buffer.

It also disposes of both 0090 hypotheses directly. Physical-page publication
and prepared-query ownership are per-layer mechanisms that layer 0 exercises in
full; layer 0 is exact at every queue depth, and the one-layer and two-layer
prefixes pass unfixed.

### Correction

`fixed_command_staging = host_deferred || device_only` in
`dsv4_prepare_attention`. Both are queued submissions that return without
synchronizing, so both take a fixed per-command upload slot; only the fully
synchronous host-visible path keeps the shared buffer. The device-only return
path now retires its slot, and its command record stays empty, so the existing
drain reports no host-node failure for it. No arithmetic, precision, router
semantics, expert count, top-k, page contract, or failure behaviour changed.

### Callback-free 43-layer result

`callback-free-r3` passes all 43 callback-free layers against the sequential
rank-local control, with zero page callbacks invoked, and reproduces the
accepted 0089 terminal object exactly:

```text
weighted hash e1a9a77f0b01a361
input hash    122a716defe84e1b
hidden hash   5017083817dd2848
logits hash   343d766f3f5c0af3
next token    8806
```

Every route was independently recomputed from live router logits and the
checkpoint's routing data, layer 0 against its actual `.d4m` fixture and layers
1--42 against the dynamic control. The rank-local output head published
`64640x2` local and `129280x2` gathered BF16 logits identical to the sequential
control's on both ranks.

The callback-backed arms in the same process still pass — sequential,
candidate1, candidate2, the terminal `MoePostRank1` queued failure with both
ranks' status `16843009` and all outputs withheld, and the exact 43-layer reuse
— so the correction does not regress the accepted P0 path.

### Resource gates

Measured across the callback-free 43-layer arm alone:

```text
checkpoint calls/B            0 / 0
workspace allocation calls/B  0 / 0
weight allocation calls/B     0 / 0
activation H2D/D2H bytes      3,953,408 / 1,375,576
paged H2D/D2H bytes             486,656 / 0
```

The whole-process window remained at `158,618,931,200 B` RSS and
`8,190,558,208 B/GPU`, below the `231,928,233,984 B` host and
`21,287,272,448 B/GPU` ceilings. The fixed preparation slab is
`43 x 16 KiB = 688 KiB` of pinned host memory per device and is allocated once
at first use, not on the queued path.

## What this experiment does not establish

This is a correctness result only. No arm was timed, no repetition was
measured, and `tau` is not instantiated. The `120 ms`, `36.7 GB/s`, and `30 ms`
M3 kill gates remain **unevaluated**, exactly as after 0090. The reported
transfer counters are correctness-harness accounting, not a production timing
result, and the arm still carries the terminal local-logit diagnostic that a
timing arm must measure or remove.

The prefix ladder is preserved as evidence with an important caveat for future
use: **a one-layer and an adjacent two-layer callback-free prefix both passed
against the defect.** The 0090 next-action rule, followed literally and stopped
at its authorized boundary, would have reported PASS and handed the 43-layer
chain the same failure. Depth-bounded prefixes cannot falsify a defect whose
trigger is the host outrunning the stream; a discriminator for queued-ownership
questions must reach production queue depth or drive the host/stream race
directly.

## Preserved negative evidence

Experiments 0086, 0090, and every raw arm under
`results/dsv4-rank-local-executor/` remain binding for their declared scopes and
are not relabeled. 0090 remains a correct rejection of the implementation as it
stood; this experiment supplies the cause it recorded as unproven, and the
prepared-query event guard it tested remains falsified.

## Decision and next authorization

M3's callback-free correctness admission is closed. The next bounded action is
the M3 production-shaped timing falsifier that 0090 never reached: apply the
`120 ms` / `36.7 GB/s` / `30 ms` gates to the callback-free chain, after
accounting for or removing the terminal local-logit diagnostic. Graph capture,
CPU arithmetic optimization, teacher forcing, Stage 7, and M4 remain blocked,
and the `94.282 ms` external figure remains feasibility context.
