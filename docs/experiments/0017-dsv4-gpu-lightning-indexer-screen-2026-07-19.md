# GPU Lightning Indexer scoring screen

Status: **rejected; exact but slower in the isolated causal metric**.

## Contract

- Hypothesis: persistent per-layer CUDA key caches and exact CUDA candidate
  scoring reduce DeepSeek learned-index time at the 4K chat boundary.
- Primary metric: end-to-end decode tokens/s, with attention-index seconds as
  the causal submetric.
- Correctness gate: scalar score oracle, generated tokens, sequential routes,
  full diagnostic hashes at the first learned-index boundary, zero decode
  checkpoint reads, balanced cache leases, and `make check`.
- Memory ceiling: existing 216 GiB host ceiling, 0.95 VRAM fraction, and the
  declared 256 MiB per-device CUDA workspace.
- Rollback: any exactness failure, workspace failure at an admitted context, or
  isolated attention-index regression.

## Candidate

The candidate retained compressed index keys in per-layer F32 device caches,
uploaded only newly produced rows, computed target-order F32/BF16 scores on
CUDA, downloaded scores, and retained deterministic host top-512 selection.
`--flash-attention` enabled the candidate for chat. The concrete runner exposed
separate host/CUDA index controls so the performance screen could hold CUDA
FlashAttention constant.

The operation fixture matched all 600 scalar scores bit-for-bit, including an
incremental append and request reset. `make check` passed 2/2 test targets.

## Full-model correctness

The one-pair 2,054-token gate is under
`results/deepseek-v4-lightning-indexer-correctness/`. Both runs completed with
identical generated tokens, routes, logit traces, operation/layer hashes, zero
decode checkpoint reads, balanced leases, and host/VRAM ceilings respected.
The candidate executed 84 CUDA index calls over 43,092 candidates and uploaded
8,289,792 bytes.

## Isolated 4K screen

The one-pair screening artifact is under
`results/deepseek-v4-lightning-indexer-4k/`. Both variants used the same binary,
4,096-token prompt, nine decode steps, device MoE, CUDA FlashAttention, model,
route sequence, and budgets. Only host versus CUDA learned-index scoring
changed.

| Metric | Host index | CUDA index | Change |
|---|---:|---:|---:|
| Prefill seconds | 861.727 | 861.719 | -0.0% |
| Prefill index seconds | 65.746 | 80.355 | +22.2% |
| Decode seconds | 8.233 | 6.477 | -21.3% |
| Decode tokens/s | 1.093 | 1.390 | +27.1% |
| Decode index seconds | 0.330 | 0.350 | +6.1% |

Generated tokens and routes matched and decode checkpoint reads were zero. The
headline tok/s difference is not an indexer win: the isolated index metric
regressed while unrelated MoE/system variance moved in the opposite direction.
This is one screening pair, not the required three-repetition classification.

## Decision

Reject and remove this runtime path. It adds a synchronization boundary per
index call, downloads every score for host top-k, and uses an O(context) F32
device cache that cannot satisfy the declared million-token range inside the
fixed workspace. At 4K, host indexing was only about 36.6 ms per decode token,
so it also cannot explain a large progressive chat slowdown.

The production GPU DSA follow-up remains
[`docs/issues/lightning-indexer-dsa.md`](../issues/lightning-indexer-dsa.md):
compact FP4 keys, fused scoring/top-k, no per-layer host round trip, batched
prefill, exact long-range validation, and explicit observability. The chat
slowdown investigation should next capture per-token expert-route, cache,
H2D, MoE preparation, attention, and thermal timing rather than attributing the
effect to DSA without causal evidence.
