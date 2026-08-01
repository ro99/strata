# GLM W4A16 bounded host expert cache

Status: **rejected after one binding pair**.

## Contract

- Hypothesis: retaining routed-expert triplets in a bounded 128 GiB host LRU
  reduces the measured SSD bottleneck enough to improve exact decode.
- Primary metric: decode tokens/s; checkpoint and physical-read bytes are the
  mechanism metrics.
- Correctness gate: identical greedy token IDs and `make check`.
- Memory ceiling: 216 GiB RSS and the unchanged 0.85 VRAM fraction.
- Rollback: remove the runtime policy if one screening pair is negative. The
  user explicitly shortened this early gate from three pairs to one.

The decode-only route replay projected a 62% checkpoint-byte reduction at a
128 GiB host budget. The candidate retained each routed expert's complete
gate/up/down packed values and scales as one atomic LRU entry. The simulator's
decode-only measurement support is retained; the negative runtime policy is
not.

## Result

Both arms used the same checkpoint, prompt, devices, FlashAttention path,
0.85 VRAM fraction, and eight measured decode steps.

| Metric | Baseline | 128 GiB cache | Change |
|---|---:|---:|---:|
| Decode tokens/s | 0.26931 | 0.22220 | -17.5% |
| Decode seconds | 29.705 | 36.004 | +21.2% |
| Decode checkpoint bytes | 49.634 GB | 22.072 GB | -55.5% |
| Decode checkpoint wall time | 16.016 s | 13.157 s | -17.9% |
| Physical reads, whole arm | 93.718 GB | 53.629 GB | -42.8% |
| Prefill seconds | 58.771 | 85.516 | +45.5% |
| Peak RSS | 2.17 GiB | 130.0 GiB | +127.8 GiB |

Both generated
`[16,13,220,3070,2082,55481,279,6145,25]`. `make check` passed all 131
unit cases plus the simulator smoke.

A second candidate arm had already completed when the matrix was stopped. It
also generated identical tokens and remained negative at 0.23866 tokens/s,
33.521 seconds, and the same 22.072 GB of decode checkpoint reads. No second
baseline arm completed, and it is not used for the decision.

Artifacts are under `results/glm-w4a16-host-cache/steady-ab/`.

## Decision

The mechanism reduced the intended resource but increased a larger unmeasured
serial term. The runtime cache, its default, and its CLI surface were removed.
Graph-phase and RSS attribution is the next measurement; no smaller cache or
different operating point is used to rescue this rejected hypothesis.
