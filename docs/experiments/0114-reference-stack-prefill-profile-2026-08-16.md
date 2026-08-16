# Experiment 0114 — direct profile of the reference stack's prefill

Status: **measurement, accepted.** First direct profile of Lvllmds4-x rather
than inference from its published table. It relocates the DSV4 prefill gap from
transfer and dispatch to two kernels, and corrects three of our standing
assumptions.

Run by the orchestrator; the codex executor was unavailable (usage limit).

## Method

Reference server launched from `bench/launch.sh` with
`--profiler-config.profiler=torch --profiler-config.torch_profiler_dir`, its
own unique-random prompt generator, and its own TTFT definition (first SSE
chunk). Two warmups, then `/start_profile`, one ~3,035-token prefill,
`/stop_profile`. Traces preserved outside Git at `/tmp/refprof/`.

## Part 1 — the reference's curve, measured

Five prompt lengths, three interleaved repetitions each, warm server.

| N | median TTFT | spread | tok/s |
| ---: | ---: | ---: | ---: |
| 774 | 12.640 s | 0.036 s | 61.2 |
| 1,545 | 14.171 s | 0.264 s | 109.0 |
| 3,035 | 15.682 s | 0.044 s | 193.5 |
| 4,544 | 17.265 s | 0.084 s | 263.2 |
| 6,326 | 18.966 s | 0.159 s | 333.5 |

Fit `12.17 s + 1.101 ms/token`, residuals under 0.4 s. Their stack is highly
reproducible warm: run-to-run spread 36-264 ms, against Strata's 13.5 s at
2,612 tokens.

Two corrections to prior beliefs:

- Their published 669-token point (20.3 tok/s) is a cold-start artifact. Warm
  at 774 tokens they measure 61.2 tok/s, so the short-prompt gap is worse than
  their own table implies.
- A first, under-warmed five-point sweep produced non-monotonic results (4,522
  tokens faster than 1,550) and a spread of +/-15 s. That was insufficient
  warmup, not intrinsic variance. It was briefly and wrongly reported as
  falsifying the cost model; the warmed data reproduces the earlier two-point
  estimate (13.31 s + 1.14 ms/token) to within 9% on both terms.

## Part 2 — GPU time inside one ~3,035-token prefill

Rank 0. GPU busy 7.314 s across an 8.091 s span, so about 90% utilised.

| term | seconds | share of compute |
| --- | ---: | ---: |
| expert H2D (pinned) | 5.032 | — |
| MXFP4 dequantisation | 1.572 | 68.9% |
| expert GEMM (`moe_v2_gpu_prefill` + Marlin) | 0.218 | 9.6% |
| **attention, all kernels** | **0.052** | **2.3%** |
| everything else | 0.440 | 19.3% |

H2D moved 28.85 GB on this rank at 5.73 GB/s, in copies of 2.10 MB and 0.13 MB
paired, about 11,850 pairs — one per expert per layer, so the routed set is
streamed in full each request.

## Part 3 — against Strata at 2,612 tokens

| term | reference | Strata | ratio |
| --- | ---: | ---: | ---: |
| attention kernel | 0.052 s | 14.916 s | **286x** |
| expert GEMM | 0.218 s | 14.334 s | **66x** |
| expert H2D rate | 5.73 GB/s | ~4.3 GB/s | 1.3x |
| attention share of compute | 2.3% | ~51% | — |

## What this overturns

**Transfer rate is not the gap.** Our H2D runs at a comparable rate to theirs.
We move more bytes (92 GB against their 57.7 GB across both ranks) and overlap
worse, but this is a 3x term. Pinning the arena and enabling prefetch were
being treated as major items; they are worth roughly 10 s of 145 s.

**Porting Marlin was the wrong frame.** Marlin accounts for 0.015 s of their
0.218 s expert GEMM. The expert matmul is not where anyone's time goes.

**Their dominant compute is dequantisation, not arithmetic.** MXFP4 unpacking
to BF16 is 1.572 s, seven times their GEMM. They pay an explicit unpack pass so
the GEMM can be a dense tensor-core operation. Strata reads FP4 directly inside
its matmul and pays 66x in the matmul instead. That is an architectural choice,
not a tuning gap.

**Attention is the dominant defect and was repeatedly deprioritised.** At 286x
it is larger than everything else combined. Earlier sessions ranked MoE first
because MoE dominates the 677-token operating point; at length, attention is
51% of Strata device time against 2.3% of theirs.

## Source availability

| component | location | status |
| --- | --- | --- |
| `_accumulate_indexed_attention_chunk_multihead_kernel` | `vllm/v1/attention/backends/mla/sparse_mla_kernels.py` | open Triton |
| `_finish_attention_state_with_sink_kernel` | same file | open Triton |
| `_fp8_mqa_logits_kernel` | `vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py` | open Triton |
| `mxfp4_dequant_reorder_kernel` | lk_moe wheel only | closed binary |
| `moe_v2_gpu_prefill` | lk_moe wheel only | closed binary |
| `fill_padding_rows_kernel`, `moe_reorder_tokens`, `moe_weighted_sum_kernel` | lk_moe wheel only | closed binary |

The 286x item is readable. The 66x item is binary, but its design — separate
dequantisation pass, then dense GEMM — is established by the trace and is the
part worth copying.

## Ranked consequences

1. Attention kernel: gather KV to contiguous, then a tensor-core dot, following
   the open Triton implementation. 286x, and 51% of our device time.
2. Expert path: adopt dequantise-then-GEMM instead of decode-inside-matmul.
   66x, design known, code not available.
3. Expert transfer volume and overlap: real but roughly 3x. Tracked separately.

## Reproduce

```bash
# profiling launcher (adds --profiler-config to bench/launch.sh)
tmux new-session -d -s ref-prof "bash <scratch>/launch-prof.sh"
# warm twice, /start_profile, one prefill, /stop_profile
```

Traces are large (65 MB gzipped per rank) and are not tracked in Git.
