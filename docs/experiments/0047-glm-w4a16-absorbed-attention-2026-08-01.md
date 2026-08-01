# GLM W4A16 absorbed-attention A/B — 2026-08-01

## Decision

Accepted. GLM dense causal attention now applies `kv_b_proj` by MLA weight
absorption instead of reconstructing every cached latent into expanded per-head
K/V. The indexed path beyond 2,048 tokens is unchanged.

The implementation adapts the attention-absorption algebra used by the
Apache-2.0 Colibri GLM CUDA backend to Strata's OffsetPackedInt4 weights,
bounded workspaces, validation, and accounting.

## Cost model and gate

The parity screen decoded in 4.655449 s: routed MoE was 3.606202 s and attention
was 1.008611 s. Compact-KV reconstruction made the attention phase serially
expand all cached 512-value latents through `kv_b_proj`, download expanded K/V,
and upload it again for FlashAttention. The mechanism removes that serial host
round trip; weight precision, attention scores, causality, router semantics,
and sparse selection do not change.

The target-shape CUDA fixture uses the real 28,672 by 512 group-128
OffsetPackedInt4 projection and matches expanded K/V attention within a 0.2%
relative/absolute-floor reassociation contract. `make check` passed 134/134
unit tests plus the simulator smoke test before the full checkpoint ran.

The single full-model gate produced exact IDs `[16,13]` and reduced decode from
4.655449 s to 3.461937 s. That cleared the binding 3.701097-second rollback
threshold, so the result advanced to three interleaved A/B pairs.

## Interleaved full-checkpoint A/B

Named session: `strata-glm-absorbed-attention-ab`

Baseline revision: `fac2f20`

Results: `results/glm-w4a16-absorbed-attention-ab-2026-08-01`

| Pair | Parity baseline seconds | Absorbed seconds | Parity tok/s | Absorbed tok/s |
|---:|---:|---:|---:|---:|
| 1 | 5.260811 | 3.483140 | 0.190085 | 0.287097 |
| 2 | 5.106904 | 3.505198 | 0.195813 | 0.285291 |
| 3 | 4.699166 | 3.476442 | 0.212804 | 0.287650 |
| median | 5.106904 | 3.483140 | 0.195813 | 0.287097 |

Median decode time improved 31.8%; median throughput improved 46.6%. Every arm
produced `[16,13]` and reconciled phase, device, checkpoint, and cache totals.
The absorbed result also beats the original pre-regression baseline of 3.701097
s / 0.270190 tok/s by 5.9% in time and 6.3% in throughput.

| Median decode metric | Parity baseline | Absorbed |
|---|---:|---:|
| attention seconds | 0.886305 | 0.159610 |
| MoE seconds | 4.201456 | 3.296057 |
| routed MoE seconds | 4.187541 | 3.283374 |
| checkpoint bytes | 9,790,500,864 | 9,790,500,864 |
| weight H2D bytes | 9,790,488,576 | 9,790,488,576 |
| combined activation H2D bytes | 345,305,400 | 28,992,312 |
| combined activation D2H bytes | 308,286,392 | 30,970,808 |
| Flash staging H2D bytes | 322,044,216 | 10,683,192 |
| Flash staging D2H bytes | 5,112,120 | 5,112,120 |
| weight allocations | 3,018 | 3,018 |
| workspace allocations | 3 | 0 |
| synchronizations | 2,343 | 2,265 |

Combined activation totals count specialized Flash staging once. The mechanism
does not change checkpoint or weight-transfer volume. It reduces Flash H2D by
96.7%, combined activation D2H by 90.0%, and attention time by 82.0%. Removing
the large host reconstruction also reduced the following MoE phase consistently
in all three pairs; routed MoE remains the next bottleneck.

Median maximum RSS was 2,275,120 KiB. Absorbed per-device VRAM usage was
13,406,109,696, 20,846,870,528, and 20,838,481,920 bytes, within the 0.85
admission budget. Median physical reads were 294,912 bytes from the warm page
cache and median physical writes were 9,768,960 bytes. No weight arena, pinned
arena, prefetch policy, host expert cache, or precision change was enabled.
