# GLM W4A16 integrated parity screen — 2026-08-01

## Decision

The integrated runtime passed the exact full-checkpoint correctness screen but
did not improve short-context decode throughput. Generated token IDs remained
`[16, 13]`, all phase/device accounting reconciled, and no fallback ran.
Decode fell from 0.27019 to 0.21480 tok/s (-20.5%). Per the experiment contract,
this arm was not repeated.

## Hypothesis and gate

The measured baseline bottleneck was MoE: 3.178 seconds of a 3.701-second
decode step, including 3.122 seconds in routed execution. The hypothesis was
that persistent dispatch workers and a common device-resident gate/up/SwiGLU/
down command would reduce its serial host/device term. The primary metric was
exact decode tok/s. Correctness required the pinned checkpoint, exact greedy
IDs `[16, 13]`, reconciled counters, `make check`, at most 0.85 VRAM admission,
and at most 216 GiB RSS.

The target-shape CUDA fixture and `make check` passed before this screen. The
single end-to-end arm was required to validate real checkpoint execution; the
requested one-arm rule superseded the usual three-repetition median.

## Arm

- Session: `strata-glm-w4a16-integrated`
- Results: `results/glm-w4a16-integrated-2026-08-01`
- Checkpoint index: `74d73bfaa26425beaf618342f4a0851b21d9198138b76bfb678f88164d987beb`
- Devices: CUDA 0,1,2; VRAM fraction 0.85
- Prompt tokens: 30; generated IDs: `[16, 13]`
- Maximum context: 256; FlashAttention and detailed timing enabled
- Repetitions: exactly one

## Result

| Metric | Prefill | Decode |
|---|---:|---:|
| Time | 37.5025 s | 4.65545 s |
| Throughput | 0.79995 tok/s | 0.21480 tok/s |
| Checkpoint bytes | 144,989,110,272 | 9,790,500,864 |
| Weight H2D | 144,988,766,208 | 9,790,488,576 |
| Activation H2D | 890,290,176 | 23,261,184 |
| Activation D2H | 1,026,164,480 | 303,174,272 |
| Weight allocations | 44,694 | 3,018 |
| Workspace allocations | 72 | 3 |
| Synchronizations | 30,487 | 2,343 |
| MoE commands / kernels | 7,524 / 15,048 | 218 / 436 |
| Attention | 4.32009 s | 1.00861 s |
| MoE | 32.8203 s | 3.61949 s |

Whole-arm logical checkpoint reads were 154,779,611,136 bytes and weight H2D
was 154,779,254,784 bytes. Block-device accounting observed 392,380,416 bytes
read and 14,553,088 bytes written; the unusually low physical read count shows
that this one arm ran with a warm OS page cache, so it is reported but not used
as a comparative storage claim.

Maximum RSS was 2,329,747,456 bytes (2.170 GiB) at this short context. End-of-run
per-device VRAM attribution was 13,418,692,608, 20,859,453,440, and
20,851,064,832 bytes. The retained one-million-token compact cache contract is
199,716,831,232 bytes (186.0 GiB), excluding transient per-step workspaces.

## Cost-model interpretation

The command path reduced decode synchronization calls from 4,081 to 2,343
(-42.6%), and prefill improved from 43.2858 to 37.5025 seconds (-13.4%). It did
not reduce the decode `argmax`: MoE rose from 3.1784 to 3.6195 seconds. Logical
decode checkpoint reads also rose by 38,928,384 bytes because the exact
indexer adds resident-spine tensors.

Compact KV reconstruction changed another resource in the wrong direction at
the short operating point. Reconstructing the 31 selected rows through
`kv_b_proj` before FlashAttention raised decode activation D2H from 45,506,048
to 303,174,272 bytes (6.66x), and attention rose from 0.49287 to 1.00861
seconds. The synchronization reduction therefore did not translate into a
decode win under `tau = max_r W_r/B_r + sum_serial`.

This is a correctness pass and a throughput falsification, not a performance
win. No second or third arm was run.
