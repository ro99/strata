# DeepSeek V4 tiered block KV cache manager

Status: **correctness accepted; performance not classified**.

## Contract

- Hypothesis: fixed typed KV blocks with bounded host residency and an explicit
  device-promotion lease can replace the scalar KV oracle without changing
  DeepSeek V4 attention, compression, learned-index, or generation results.
- Primary metric: exact generated-token, logit, layer-hash, operation-hash, and
  route agreement. Cache allocation, occupancy, gather, and promotion counters
  are the implementation metrics; throughput is descriptive only.
- Correctness gate: `make check`, typed-table and ownership fixtures, fork/COW,
  reset/truncate/release, host-capacity failure, CUDA promotion/eviction leases,
  and a full-model first learned-index-boundary A/B run.
- Memory ceiling: the admitted host KV budget and separate per-device KV
  budgets, while preserving the existing workspace and weight-cache reserves.
- Rollback: scalar KV mode remains the default; reject block mode on any exact
  mismatch, failed admission, stale-row/UAF, or device-lease failure.

The experiment branch `feat/dsv4-tiered-kv-cache` was created directly from
`main` at `099a1aa`; it does not inherit the unpromoted batched-expert branch.

## Implementation

`Dsv4KvCache` now owns separate `(kind, layer)` tables for sliding, CSA, HCA,
and learned-index rows. Blocks carry owner, logical position, format, layer,
compression ratio, used/capacity rows, reference count, in-flight leases, and
per-device residency. Appends are contiguous and bounded by a fixed host
allocator; forked sequences share blocks and append performs copy-on-write.
Reset, release, and truncation mask stale rows and release references without
double-freeing shared or in-flight blocks. Device promotion is explicit and
lease-protected, with LRU eviction of non-in-flight blocks and CUDA H2D,
allocation, synchronization, gather, hit/miss, and promotion telemetry.

Runtime admission keeps KV budgets separate from weight-cache budgets. The
block manager is opt-in through `--block-kv-cache`; `--scalar-kv-cache` keeps
the existing paged scalar oracle. The runner reports host/device capacity and
occupancy, host writes, gathers, promotions, RSS, and per-GPU VRAM.

## Validation

`make check` passed both CTest targets (100%). The sanitizer test binary also
passed all 95 tests with `ASAN_OPTIONS=detect_leaks=0`; LeakSanitizer itself is
not usable in this sandbox because its ptrace restriction aborts before tests.

The final full-model run is under
`results/deepseek-v4-kv-cache-index-boundary/`. It used the same frozen runner,
model, devices, budgets, and deterministic decode for both variants:

- 2,054 prompt tokens and two generated tokens (`43, 8806`) in both runs;
- 63 prefill and 21 decode learned-index queries in both runs;
- identical logits, layer hashes, operation hashes, and sequential route trace;
- zero generation and decode checkpoint-read bytes;
- block mode stayed below its 79,839,232-byte host cap (50,495,488-byte peak),
  with 1,817 allocations, 527 live blocks, 22,388,628 hits, and 45,785,720,832
  gathered bytes;
- scalar/block RSS was approximately 148.94 GB, with unchanged per-GPU VRAM
  snapshots.

The one-pair timings are diagnostic only and do not establish a performance
win: scalar prefill/decode was 457.389/1.037 s and block mode was 456.509/1.024
s. The charter's three-interleaved-repetition median gate remains open.

An earlier short 198-token pair also matched all exact diagnostics. A first
boundary attempt was discarded after the runner was rebuilt during the A/B
script; the final script copies and hashes one frozen runner before either
variant starts.
