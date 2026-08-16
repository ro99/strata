# Experiment 0108 — landed DSV4 prefill curve measurement

Status: **incomplete measurement; 2,612-token arm blocked by an existing
workspace contract.** No runtime code, instrumentation, or design was changed.

## Scope and build

This is a measurement-only record of the landed stack, not a hypothesis test or
A/B comparison. The requested operating point was untraced TP2 rank-local
prefill on devices 1 and 2 with `--prefill-page-tokens 8192` and
`--max-context 8192`, using one arm at approximately 677 tokens and one at
approximately 2,612 tokens.

The runner was built from detached commit
`54505bacb7310d3f8dd29f6bf2a3216f3c563e52` (`54505ba`) with
`STRATA_ENABLE_NCCL=ON`. The final runner SHA-256 was
`d1d1b4861cf856e8c04d025b832a771685b4468924b844c4d70d46e48fa2b11b`.

Two startup-only failures are preserved separately. The first runner was built
without NCCL and rejected rank-local decode; the second used the detached
checkout's nonexistent default model path. Neither reached model execution.
The successful arm uses the actual checkpoint at
`/home/rodrigo/Developer/strata/models/dsv4f`.

## 677-token arm

The arm completed in approximately 166.5 seconds wall time, from 11:03:34 to
11:06:21 local time. Raw files are under
`results/dsv4-0108-landed-prefill-curve/677-real/`.

| Metric | Measurement |
|---|---:|
| Prompt tokens | 677 |
| Total prefill | 53.793264 s |
| Prefill throughput | 12.5852 tok/s |
| Attention | 22.377361 s |
| Query | 3.764747 s |
| KV | 3.915576 s |
| Score | 14.280580 s |
| MoE | 27.145546 s |
| mHC | 2.741972 s |
| Query matmul device kernel | 0.335592 s |
| Paged-attention calls | 86 |
| Paged-attention kernel launches | 1,655 |
| Paged-attention page bytes | 30,564,224 B |
| Expert demand H2D bytes | 74,380,770,816 B (74.381 GB) |
| Expert demand wait | 17.217182 s |
| Demand bytes / wait | 4.320 GB/s |
| Demand bytes / MoE seconds | 2.740 GB/s |
| Cache hits / misses / evictions | 85,230 / 10,512 / 7,888 |
| Decode tok/s | 15,070.45 tok/s* |
| Decode checkpoint reads | 0 B |
| RSS | 158,772,056,064 B |
| GPU 1 VRAM | 22,941,925,376 B |
| GPU 2 VRAM | 22,864,330,752 B |

\* The sweep arm generated one token; its decode interval was only
66.355 microseconds, so this decode rate is not a meaningful steady-state
throughput measurement. It is reported as emitted by the existing counters and
was not investigated or replaced with another arm.

The arm generated token ID `2107` and reported zero decode checkpoint reads.

## 2,612-token arm

The intended `PROMPT_WORDS=1630` arm started at 11:07:11 and stopped after
approximately 113 seconds, before producing model JSON. It failed with the
existing runtime error:

`DeepSeek attention-to-mHC workspace exceeds its bounded contract`

Raw system, log and empty JSON files are preserved under
`results/dsv4-0108-landed-prefill-curve/2612-real/`. No alternative context,
workspace bound, page size, or prompt length was tried, because this task is a
measurement-only curve and the requested operating point could not complete.

## Curve and expert-byte conclusions

A two-point `fixed + marginal_per_token` fit cannot be computed: the 2,612-token
point has no total-prefill measurement. The only valid landed point is
53.793264 s at 677 tokens. Consequently there is no defensible Strata-versus-
reference comparison for fixed cost or marginal cost from this run.

The 677-token arm moved 74.381 GB of expert demand H2D traffic. Whether bytes
rise toward 156.9 GB or rise beyond it at 2,612 tokens is **undetermined**;
the blocked arm emitted no demand-byte counter. No extrapolation is made.

## Verdict and next step

The 677-token measurement completed and is preserved. The requested curve is
blocked at 2,612 tokens by the bounded attention-to-mHC workspace contract.
This record proposes no mechanism and makes no claim about the missing curve.
Further work requires an explicitly authorized decision about that existing
workspace admission failure.
