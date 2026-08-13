# Experiment 0089: rank-local terminal publication and queued failure closure

Date: 2026-08-12
Branch: `exp/dsv4-a2-ownership-screen`
Base: `a561f77`
Disposition: **M3 CORRECTNESS COMPLETE / REVIEW REQUIRED; performance gates remain open**

## Question and scope

Experiment 0088 proved exact 43-layer state chaining but stopped before the
rank-local output head and before failure injection inside the queued chain.
This bounded continuation asked whether the same executor could:

- execute the replicated final mHC reduction/norm and the two contiguous
  `head.weight` row shards;
- publish two `64640`-row FP32 projections as rank-ordered BF16 logits through
  a real NCCL all-gather;
- reproduce terminal hidden, all `129280` logits, and the next-token
  association across a sequential control and queued candidates; and
- retain a terminal queued failure on both ranks, withhold every output, close
  the command exactly once, and then run an exact 43-layer reuse.

This remains a correctness experiment. It does not time the candidate or
instantiate `tau`. The operating point, fixtures, precision, routing, page
state, memory ceilings, and rollback rules are unchanged from Experiment 0088.

## Implementation

The accepted output ownership contract is now represented directly in the
rank-local executor. Each device owns a fixed BF16 shard and full-logit buffer.
The existing final mHC/output-head primitive exposes a borrowed device result;
the executor rounds the local projection to BF16 and performs NCCL all-gather
in rank order. The final API publishes only after both ranks complete and
validates rank equality, local-shard association, finiteness, and greedy token
association.

The queued failure API permits only a logical failure on the terminal layer.
The binding arm injects `MoePostRank1` at layer 42. A defect found by the first
arm was fixed at the ownership boundary: chain mode had consumed the failed
MoE command during enqueue and `finish_chain` attempted to drain it again.
The corrected path leaves the terminal command in flight and lets
`finish_chain` drain it once, copy the reduced status, clear payloads, and
reset for reuse.

The harness also resets caller-owned observation/page-callback state before a
reuse. This is harness lifecycle state, not model arithmetic.

## Binding result

Binding directory:

```text
results/dsv4-rank-local-executor/m3-43-chain/correctness-r13/
```

```text
combined.log SHA-256 b78ec2058773a3f24ab2a10c9d99257201b1d75db9e3d84633bb402889c23887
source.diff  SHA-256 1d49ed2fb93d2a4f0059a5cbf5afbf8405a9bf240432a62b8de26fb219d0a162
exit status  0
wall interval 93 seconds
```

The sequential control, two queued candidates, and post-failure reuse were
identical:

```text
weighted hash  e1a9a77f0b01a361
input hash     122a716defe84e1b
hidden hash    5017083817dd2848
logits hash    343d766f3f5c0af3
next token     8806
```

For every successful arm, both ranks produced exact local FP32 shards, exact
rank-ordered BF16 full logits, finite values, and the same next token. Token
`8806` also matches the captured position-104 centralized generation token;
the binding numerical oracle remains queued rank-local versus sequential
rank-local because their accumulated publication association differs from the
centralized state after layer 0.

The queued terminal failure produced reduced MoE status `16843009` on both
ranks, four expected status errors, zero terminal weighted/input/hidden
payloads, zero logits hash/token publication, and closed collectives. The
immediately following 43-layer reuse matched the sequential hashes above.
All eight inherited one-layer attention/MoE pre/post failure arms also passed.

## Resource gates

Setup with both output-head shards passed at:

```text
RSS                 158,309,720,064 B
GPU rank 0            8,144,420,864 B
GPU rank 1            8,144,420,864 B
```

Across candidate 1, candidate 2, the failed chain, and exact reuse:

```text
RSS before/after      158,618,910,720 / 158,619,095,040 B
GPU before/after        8,190,558,208 /   8,190,558,208 B per rank
checkpoint calls/B    0 / 0
workspace allocs/B    0 / 0
weight allocs/B       0 / 0
```

All values remain below the `231,928,233,984 B` host and
`21,287,272,448 B/GPU` ceilings. The logged activation/page/CPU-MoE traffic is
correctness-harness accounting across four arms, not a production timing
result. The output primitive also retains its accepted host final-mHC callback
and diagnostic local-logit staging; a timing arm must measure or remove those
serial terms rather than calling this correctness result device-only.

## Preserved failed arms

- `correctness-r9`: terminal head passed, but queued failure status was lost
  because enqueue and finish both attempted to consume the same failed command.
- `correctness-r10`: diagnostic rerun confirmed the lost status was `0,0`.
- `correctness-r11`: single-drain status closure passed, but reuse exposed stale
  caller-owned callback `invoked` flags; this was not a valid reuse arm.
- `correctness-r12`: first complete pass after both lifecycle corrections,
  superseded only by r13's final device-view validation and source review.
- `correctness-r13`: binding result from the final reviewed source.

These arms remain in place and must not be renamed as passing evidence.

## Decision

M3 correctness is complete at this fixed position: all 43 dependent layers,
routes, pages, terminal hidden, rank-local logits/token, queued-chain failure
closure, same-executor reuse, memory, I/O, and allocation gates pass. This does
not pass M3 performance. The next bounded action is the production-shaped M3
timing falsifier: remove or separately account for the correctness harness's
live page callbacks and local-logit diagnostics, then apply the declared
`120 ms`, `36.7 GB/s`, and `30 ms` gates before any M4 work.
