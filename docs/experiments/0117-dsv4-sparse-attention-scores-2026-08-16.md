# Experiment 0117 — score only the attended candidates

Status: **accepted.** Attention at 2,612 tokens falls 78.14 s to 60.44 s and the
score bucket 51.36 s to 32.28 s. Generated tokens unchanged at both lengths.

## Premise

The paged attention path computed scores with a dense `cublasGemmEx` of shape
`(flat_rows) x (rows * kDsv4PagedHeads) x kDsv4PagedHeadDim`: every gathered KV
row against every query row. The three finish passes then read only the 640
candidates each row actually attends, indexing the dense matrix by
`pages[candidate.page].flat_begin + candidate.row`.

That makes attention `O(N_kv * N_q)` where the reference's indexed kernel is
`O(N_q * 640)`. At 2,612 tokens `flat_rows` is about 2,818 against 640
candidates, so roughly 4.4x of the score work was discarded, and the score
region was `rows * 64 * flat_rows * 2` bytes.

The reference's `_accumulate_indexed_attention_chunk_multihead_kernel` in
`vllm/v1/attention/backends/mla/sparse_mla_kernels.py` takes an explicit
`indices` tensor and touches only those rows. That file is open; the MoE
kernels it sits beside are not.

## Change

`dsv4_sparse_scores_kernel` scores only each row's candidates. One block owns
one row and eight heads; the KV row is staged once in shared memory and each
warp dots its own head against it, 16 elements per lane with a shuffle
reduction. Invalid candidates store zero.

The dense GEMM is removed from both the standalone `dsv4_paged_attention` and
the fused `dsv4_paged_attention_to_mhc` paths. The three finish passes index by
candidate rather than by flat KV row, so `score_width` becomes the candidate
count. The score region becomes `rows * 64 * candidates * 2` bytes.

The two-pass softmax, its per-head sink, the `ex2.approx` exponential that
matches Triton, and the candidate ordering are all untouched.

## Result

| metric | before | after |
| --- | ---: | ---: |
| attention @2,612 | 78.139 s | 60.437 s |
| score @2,612 | 51.363 s | 32.277 s |
| attention @677 | 16.826 s | 13.240 s |
| score @677 | 11.625 s | 6.570 s |
| paged-attention calls @2,612 | 236 | 172 |
| prefill @2,612 | 136.12 s | 132.38 s (19.73 tok/s) |

Generated IDs `2107, 8777, 1277, 440` at 677 and 2,612 tokens. `make check` 2/2.

## Numerical contract

Not bit-exact against the dense path. The dot is now a per-lane sequential FP32
accumulation over 16 elements followed by a shuffle tree, where cuBLAS used its
own order; both accumulate in FP32 and round to BF16 on store, and the
downstream `dsv4_scale_scores` rounding is unchanged. Generated tokens are
identical at both measured lengths. This is the same class of reassociation
accepted in experiments 0104 and 0105.

## What did not happen

The workspace did not shrink: `prefill_max_workspace_bytes` is 171,114,496 at
2,612 tokens before and after. The score region was not the binding term.
Modelling the layout at the observed ~475-row sub-chunk gives 175 MB against
171 MB measured, split as:

| region | MB | share |
| --- | ---: | ---: |
| value accumulator, `rows*64*512*4` | 62.3 | 35.6% |
| score, `rows*64*640*2` | 38.9 | 22.3% |
| attended, `rows*H*512*2` | 31.1 | 17.8% |
| decoded, `rows*H*512*2` | 31.1 | 17.8% |
| branch | 8.4 | 4.8% |
| gathered KV | 2.9 | 1.7% |

So sub-chunking is now bounded by the FP32 value accumulator, which is linear
in `rows * heads * head_dim` and cannot be removed by making scores sparse. At
the full 2,611-row page it would be 342 MB for `value` alone plus as much again
for `attended` and `decoded`, so whole-page dispatch remains out of reach under
the 384 MiB cap. Calls fell 236 to 172, not to 43.

This matters for the expert path: rows per expert per dispatch rise only
modestly, so the reference's dequantise-then-GEMM MoE design is still working
against a small row count here. Raising it further requires shrinking the
accumulator regions, not the scores.
