# Experiment 0088: exact rank-local 43-layer dependent chain

Date: 2026-08-12
Branch: `exp/dsv4-a2-ownership-screen`
Base: `7da38b75f68f1675a11ada2e8410cb676a198561`
Disposition: **M3 43-LAYER CHAIN-STATE CORRECTNESS PASS / REVIEW REQUIRED;
terminal head/logits/token, queued-chain failure injection, timing, and M4
remain blocked**

## Question and scope

The bounded M3 question was whether the accepted one-layer rank-local executor
and fixed per-command attention staging could execute one actual 43-layer
dependent forward with one final completion, exactly matching a sequential
rank-local control. This is a correctness experiment. It does not instantiate
`tau`, measure a production forward, or make a throughput claim.

The fixed operating point was position 104 on two RTX 3090s selected with
`CUDA_VISIBLE_DEVICES=1,2`. The run loaded all 43 actual `.d4r` attention and
`.d4m` MoE fixtures, the 86-record `.d4c` dependent replay, actual checkpoint
weights, and the checkpoint's two routing forms: token-to-expert hash rows for
layers 0--2 and learned selection biases for layers 3--42.

The correctness oracle is association-aware:

- layer 0 is exact against the captured `.d4r`/`.d4m` starting-state fixture;
- layers 1--42 are exact against the independently executed sequential
  rank-local chain;
- every layer's route is independently recomputed from its live router logits
  and checkpoint hash row or learned bias;
- both ranks must agree on live input, logits, route, query, key/value state,
  terminal BF16 state, and final hidden state; and
- centralized downstream and terminal state are diagnostics only because the
  accepted rank-local BF16 publication association intentionally differs from
  the centralized association and compounds across layers.

Any mismatch, callback failure, non-finite value, fallback, checkpoint read,
timed workspace/weight allocation, or memory-ceiling breach was a rejection.
The host ceiling was `231,928,233,984 B`; the per-rank GPU ceiling was
`21,287,272,448 B`.

## Implementation correction

The M3 implementation adds fixed caller-owned observations for all 43 callback
slots, a terminal-layer result, and an exact 43-enqueue/one-finish harness. It
also adds an allocation-free hash-routing entry point so layers 0--2 preserve
the checkpoint's production token-to-expert membership instead of incorrectly
requiring a learned bias.

Two harness defects were corrected during review:

1. The first downstream comparison used centralized layer-1 query/state as a
   hard oracle. That is invalid after the rank-local layer-0 publication. The
   corrected oracle uses the sequential rank-local chain for downstream state
   while still recomputing every route independently.
2. The sequential terminal control finishes a one-command terminal queue after
   42 synchronous layers, so its expected `chain_count` is 1. The queued
   candidates correctly report 43. Terminal state comparison intentionally
   ignores that bookkeeping difference.

No production arithmetic, precision, router semantics, expert count, top-k,
or failure behavior changed to make the result pass.

## Binding result

Binding directory:

```text
results/dsv4-rank-local-executor/m3-43-chain/correctness-r8/
```

Hashes:

```text
combined.log  3eef2dde3a1cc59fbd7aceede9a9e50c95c6302e98f7f9152c0969eda2f53bb1
source.diff   957e96378ff080a97ce9bc13319405e172bf2be010dc4874e3283d6bb4a77e9e
exit.status   0
```

The sequential control and both queued repeats passed all 43 layer
observations. Candidate observations were bit-exact against the sequential
rank-local observations for encoded FFN input, router logits, routes,
coefficients, and rank-associated CPU partials. Query, key/value, row patch,
historical page bytes, and rank association also passed on every layer.

All three terminal results were identical:

```text
weighted hash  e1a9a77f0b01a361
input hash     122a716defe84e1b
hidden hash    5017083817dd2848
```

The sequential terminal queue reported `chain_count=1`; candidate 1 and
candidate 2 each reported `chain_count=43`. Both candidates used exactly 43
enqueues followed by one `finish_chain` and performed no intermediate
top-level result collection.

The centralized terminal diagnostic was retained rather than hidden:

| state | mismatches | maximum absolute delta |
|---|---:|---:|
| weighted | 4,016 / 4,096 | 2.125 |
| normalized input | 4,017 / 4,096 | 0.197265625 |
| final hidden | 15,952 / 16,384 | 4.0 |

This diagnostic is not an exactness failure: after layer 0, centralized and
rank-local publication associations are different execution contracts. The
binding exact comparison is queued rank-local against sequential rank-local.

## Failure, memory, I/O, and traffic gates

The same final invocation reran all eight inherited logical failure cases:
attention and MoE, pre- and post-collective, on both ranks. Every case produced
matching nonzero phase status on both ranks, zeroed/withheld payloads, closed
collectives, and an exact same-executor success reuse. The partial-enqueue
exceptional cleanup remains accepted M2 evidence and its implementation path
was not changed by M3.

This does not inject a failure into the queued 43-layer chain itself. The
current chain API deliberately rejects diagnostic injection in chain mode, so
queued-chain status persistence/failure closure remains an explicit M3 item.

Setup memory passed at:

```text
RSS              158,305,931,264 B
GPU rank 0         7,611,744,256 B
GPU rank 1         7,611,744,256 B
```

Across the two queued candidates, the measured window reported:

```text
RSS before/after              158,610,980,864 / 158,611,144,704 B
GPU rank 0 before/after         7,657,881,600 /   7,657,881,600 B
GPU rank 1 before/after         7,657,881,600 /   7,657,881,600 B
checkpoint calls/bytes          0 / 0
workspace allocation calls/B    0 / 0
weight allocation calls/B       0 / 0
activation H2D/D2H bytes         7,941,728 / 2,245,984
MoE H2D/D2H bytes                5,636,096 / 1,585,856
paged H2D/D2H bytes                973,312 / 0
physical page bytes             28,854,272
synchronization calls                    4
```

Those transfer totals include both queued repetitions. They expose the
correctness harness's live host page and CPU-MoE callback boundaries and must
not be presented as a device-resident production timing result.

## Preserved intermediate and negative evidence

All earlier arms remain under `results/dsv4-rank-local-executor/m3-43-chain/`:

- `correctness-r1`: invalid command spelling; no experiment result;
- `correctness-r2`: setup rejected the incorrect all-layer learned-bias
  assumption;
- `correctness-r3`: layer 0 passed and the downstream failure lacked adequate
  localization;
- `correctness-r4`: proved the layer-1 centralized query was the incompatible
  oracle (`0xbbb1` rank-local versus `0xbb6a` centralized);
- `correctness-r5`: all 43 sequential observations passed, then the harness
  incorrectly required `chain_count=43` from the one-command sequential
  terminal queue; and
- `correctness-r6`: first complete correctness pass, superseded by r7 because
  it did not rerun the eight-arm failure matrix and retained a stale preamble;
  and
- `correctness-r7`: complete pass superseded by r8 so the binding source
  snapshot includes the final hash-router non-finite parity correction.

The historical 0086 adjacent-chain rejection also remains binding history. It
is not renamed or erased by this corrected implementation.

## Decision

M3's bounded 43-layer **correctness slice passes**. This proves exact queued
multi-layer ownership at one actual decode position and closes the defect that
blocked entry to M3. It does not complete M3's performance kill gates:
`>120 ms`, routed CPU `<36.7 GB/s`, and non-CPU dependency `>30 ms` remain
unmeasured. `tau` is therefore intentionally not instantiated here.

It also does not execute the rank-local output head or validate terminal logits
and next-token association. The measured terminal object is the exact final
`4x4096` mHC hidden state. Therefore this is not full M3 completion.

Stop at this review boundary. A production-shaped timing slice requires a new
explicit authorization and must first remove or account for the correctness
harness's host page diagnostic boundary. M4, teacher forcing, generation,
graph capture, CPU arithmetic optimization, and any 10 tok/s claim remain
blocked.

Final validation passed `286/287` direct tests with one documented opt-in
skip, root `make check` (`2/2`), and `git diff --check`.
