# DeepSeek V4 long-context cache and learned index boundary

Status: **correctness accepted at the first learned-index boundary; performance
not classified from one pair**.

## Contract

- Hypothesis: lazily committed host KV/index pages and exact learned top-512
  selection extend the DeepSeek cache past 2,048 tokens without changing the
  exact decode contract or short-request behavior.
- Primary metric: exact generated-token and diagnostic agreement; throughput is
  reported only descriptively because the requested A/B sample has one pair.
- Correctness gate: independent scalar learned-index score/top-k oracle,
  full-model boundary generation, finite logits and layer hashes, zero
  checkpoint reads during prefill/decode, and `make check`.
- Memory ceiling: 216 GiB host admission, 0.85 VRAM fraction, and the existing
  256 MiB per-device workspace reserve.
- Rollback: reject on oracle mismatch, post-boundary checkpoint reads, failed
  admission, non-finite diagnostics, or any measured regression confirmed by a
  three-repetition matrix.

Candidate revision: `b173742` (`feat: add exact DeepSeek long-context cache`).
The main reference executable was built from `1e9eebe`. Both used the same
DeepSeek-V4-Flash-DSpark checkpoint, devices `0,1,2`, 216 GiB host ceiling,
0.85 VRAM fraction, exact device MoE, and deterministic greedy decoding.

## Correctness evidence

The C++ operation oracle covers the target indexer's weighted-ReLU score
reduction and deterministic 513-candidate/top-512 boundary. `make check` passed
2/2 tests.

The full-model boundary run used a 2,054-token prompt, `--max-context 32768`,
and two requested new tokens. It completed in 474.79 s prefill plus one decode
step with:

- 63 learned-index queries and 32,319 candidates during prefill;
- 21 learned-index queries and 10,773 candidates during post-boundary decode;
- zero generation and decode checkpoint-read bytes;
- 258,560 finite traced logits and 88,365 layer-hash records;
- peak RSS 145,357,932 KiB (138.62 GiB), below the 216 GiB ceiling;
- exit status 0 and the required diagnostics gate.

The artifact is under `results/deepseek-v4-long-context-boundary-b173742/`.

## Single short-context pair

The requested one interleaved pair used the existing `Hello`/128-token matrix
configuration at logical context 2,048:

| Variant | Decode seconds | Decode tok/s |
|---|---:|---:|
| main reference | 25.9572 | 4.8927 |
| candidate | 26.6432 | 4.7667 |

Generated token IDs were identical. The candidate is 2.58% slower in this one
pair; no performance conclusion is drawn because the charter requires a median
of at least three interleaved repetitions and rejects results within observed
variance. Artifacts are under
`results/deepseek-v4-long-context-single-ab-b173742/`.

## Decision and remaining work

The exact first-boundary correctness gate passes, and short-request behavior is
token-identical in the single pair. This does not establish a throughput win or
validate practical 32k/200k/1m ingestion. Production-scale batched prefill and
a three-repetition performance matrix remain follow-up work documented in
[`issues/production-scale-batched-prefill.md`](../issues/production-scale-batched-prefill.md).
