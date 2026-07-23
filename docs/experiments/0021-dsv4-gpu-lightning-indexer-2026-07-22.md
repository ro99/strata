# GPU Lightning Indexer candidate — 2026-07-22

Status: **correctness accepted; performance not promoted**.

The issue-5 candidate keeps compact FP4 learned-index blocks in the existing
KV cache, applies query Hadamard/FP4 simulation on CUDA, scores candidates with
the target weighted-ReLU/BF16 boundaries, and performs deterministic lower-row
tie-stable top-512 selection in a bounded workspace. The scalar path remains
the independent exact oracle. Device-resident blocks are used when explicit
per-device KV budgets are supplied; otherwise compact host blocks are staged.

## Correctness gate

The reproducible runner is
`scripts/run_deepseek_v4_lightning_indexer_correctness.sh`. The one-pair gate
used the frozen `b247829` runner, 2,054 prompt tokens, 2 decode steps, 64-token
prefill pages, 216 GiB host ceiling, and 256 MiB/device KV budgets. Artifacts
are under `results/deepseek-v4-lightning-indexer-correctness-v2/`.

Reference and CUDA candidate both passed:

- generated-token, route, logits, layer-hidden, and operation hashes;
- sequential selected-position trace hash and equal query/candidate/selected
  counts (63 prefill index queries; 21 decode queries);
- finite logits and zero decode checkpoint reads;
- scalar and CUDA paths were both exercised explicitly.

The candidate reported 84 CUDA index calls over 43,092 candidates and selected
43,008 positions. Its aggregate index transfer totals were 2,786,112 H2D bytes,
172,368 D2H bytes, and 2,924,544 useful-selection bytes; the maximum device
kernel time was 0.677 s and the maximum end-to-end index time was 0.679 s.

## Performance classification

This is not a win claim. In the single correctness pair, scalar learned-index
time was 4.145 s and CUDA learned-index time was 4.524 s; prefill was 612.798 s
and 605.861 s respectively, so unrelated variance dominates the latter. The
required interleaved median-of-three 32k/200k/1m matrix remains open.
