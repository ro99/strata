# DeepSeek batched expert dispatch

Status: **correctness accepted; performance screened as neutral**.

## Contract

- Hypothesis: grouping page rows by routed expert leases each expert once and
  reduces expert kernel launches without changing exact routed/shared output.
- Primary metric: median page-64 prefill tokens/s over three interleaved
  baseline/candidate repetitions.
- Correctness gate: page-size-one and page-size-64 generated tokens, logits,
  operation/layer hashes, ordered routes, expert-row counts, finite values,
  zero prefill/decode checkpoint reads, and balanced leases.
- Memory ceiling: existing 216 GiB admission, 0.85 VRAM fraction, and the
  bounded 256 MiB per-device workspace contract.
- Rollback: revert on oracle mismatch, dropped/duplicated rows, lease leak,
  extra I/O, workspace overrun, or a three-repetition regression.

## Implementation

`CudaBackend` now accepts grouped routed rows. The runtime groups each page's
selected rows by expert and device, leases one W1/W3/W2 triplet per unique
expert, uploads hidden rows and row/coefficient descriptors once per device,
executes batched FP4 gate/SwiGLU/down kernels, and scatters outputs in original
row/rank order before adding the shared expert. Page size one remains the
row-at-a-time runtime oracle. Decode remains row-at-a-time.

## Correctness pair

The reusable `scripts/run_deepseek_v4_bounded_prefill_correctness.sh` pair used
the 213-token prompt, page 1 versus page 64, and two decode tokens. The pair
passed every gate, including byte-identical ordered routes and balanced leases.
Page 64 reduced unique routed experts from 54,954 to 12,706 and MoE kernel
launches from 127,009 to 2,580. Artifacts are under
`results/deepseek-v4-batched-expert-correctness/`.

## Fair matrix

`scripts/run_deepseek_v4_bounded_prefill_matrix.sh` compared the unmodified
`4e93545` binary and this candidate with page 64 for both, 260 prompt tokens,
and three interleaved repetitions. Median results:

| metric | baseline | candidate | change |
| --- | ---: | ---: | ---: |
| prefill seconds | 53.2977 | 53.1111 | -0.35% |
| prefill tokens/s | 4.8783 | 4.8954 | +0.35% |
| routed expert rows | 67,080 | 67,080 | equal |
| unique routed experts | 67,080 | 8,326 | -87.6% |
| MoE kernel launches | 155,568 | 3,209 | -97.9% |
| MoE execution seconds | 12.1742 | 9.5768 | -21.3% |
| MoE preparation seconds | 10.4245 | 12.5152 | +20.0% |
| prefill synchronization calls | 85,710 | 55,220 | -35.5% |
| routed MoE H2D bytes | 510,050,304 | 549,081,536 | +7.7% |
| routed MoE D2H bytes | 1,282,336,364 | 1,282,214,404 | -0.01% |
| cache hits / misses | 269,283 / 10,038 | 60,126 / 10,038 | misses equal |
| maximum RSS (KiB) | 145,356,012 | 145,364,928 | +0.006% |

The prefill difference is within observed run variance, so this is not called
an end-to-end throughput win. The expert dispatch reduction is real and exact,
but preparation and descriptor-transfer costs offset it on this matrix.

## Decode

The same matrix measured decode separately: median decode time was 0.7382 s
baseline versus 0.6042 s candidate, with 258 routed expert rows, 620 MoE
kernel launches, and 760 synchronization calls in both variants. Cross-request
decode scheduling remains out of scope.
