# Experiment 0110 — DSV4 curve with explicit KV host budget

Status: **capability gate passed; measurement complete.** This is a
configuration-only follow-up to 0108/0109. No runtime code or instrumentation
changed.

## Scope and configuration

The 0109 2,612-token arm exposed `DeepSeek KV host cache capacity is
exhausted`. Its 677-token JSON measured a 42.5 MiB computed host capacity and
about 32 KB of KV host peak per prompt token. This run tests that admission
diagnosis with `--kv-host-cache 4G` on both arms.

Both arms used the same runner built from commit
`24cdea98cd4cfcbd116541a03fe4338d4b13d1d2` (`24cdea9`), page size 8192,
context 8192, devices 1,2, untraced TP2 rank-local execution, and `--max-new
4`. The raw captures are under
`results/dsv4-0110-kv-host-4g/`.

The measured resource is KV host-cache capacity admission. The configuration
increases that capacity from the computed 44,561,792 B to 4,294,967,296 B;
the expected sign is higher permitted host residency and potentially higher
RSS, with no change to weights, precision, router semantics, expert count,
top-k, attention workspace, or device VRAM policy. This is not a throughput
optimization experiment.

## Budget and wall time

The planned model time was approximately 3 minutes for 677 tokens and 5–15
minutes for 2,612 tokens. Actual `/usr/bin/time` elapsed wall time was 2:56.49
for the short arm and 4:50.42 for the long arm, 7:46.91 total. The long arm
did not exceed 25 minutes. No 16G escalation or other retry was needed.

## Results

| Metric | 677 tokens | 2,612 tokens |
|---|---:|---:|
| Total prefill | 63.904203 s | 178.441332 s |
| Prefill tok/s | 10.59398 | 14.63786 |
| Attention | 22.803820 s | 96.499184 s |
| Query | 3.931629 s | 14.449289 s |
| KV | 3.832638 s | 14.911487 s |
| Score | 14.624810 s | 61.008242 s |
| MoE | 36.934874 s | 64.631798 s |
| mHC | 2.718212 s | 11.294376 s |
| Query matmul device kernel | 0.336787 s | 1.126964 s |
| Prefill paged-attention calls | 86 | 236 |
| Prefill paged-attention launches | 1,655 | 4,579 |
| Prefill paged-attention page bytes | 30,564,224 B | 383,225,472 B |
| Expert demand H2D bytes | 74,380,770,816 B (74.381 GB) | 92,121,114,624 B (92.121 GB) |
| Demand wait | 26.992911 s | 28.135119 s |
| Demand GB/s: bytes / wait | 2.7556 | 3.2742 |
| Demand GB/s: bytes / MoE | 2.0138 | 1.4253 |
| Cache hits / misses / evictions | 85,230 / 10,512 / 7,888 | 460,413 / 13,034 / 10,395 |
| Decode tok/s | 8.9918 | 8.0643 |
| Decode checkpoint reads | 0 B | 0 B |
| RSS | 158,858,788,864 B | 159,315,767,296 B |
| GPU 1 / GPU 2 VRAM | 22,971,285,504 / 22,895,788,032 B | 23,808,049,152 / 23,524,933,632 B |
| Generated IDs | `2107, 8777, 1277, 440` | `2107, 8777, 1277, 440` |

Both arms completed. The 2,612-token arm therefore passes the primary
capability gate. It also confirms that 0109's row-subchunking remains active:
the prefill attention call count rises from 86 to 236, with 4,579 launches and
383 MB of page reads at the long prompt. These are phase-scoped counters;
decode counters are excluded.

## KV host-cache accounting

| KV block | 677 tokens | 2,612 tokens |
|---|---:|---:|
| Host capacity | 4,294,967,296 B | 4,294,967,296 B |
| Host peak | 21,729,792 B | 81,616,576 B |
| Host used at report | 8,866,944 B | 23,733,760 B |
| Allocated blocks | 252 | 1,135 |
| Promotions | 252 | 1,135 |

The long-run peak is 31,246.8 B/token; the short-run peak is 32,097.2
B/token. At the observed long-run rate, `--max-context 8192` should budget
approximately 255,973,580 B (243.9 MiB) for KV host peak, before any explicit
implementation overhead. The prior computed 44,561,792 B (42.5 MiB) budget was
about 5.7x below that extrapolated context requirement. The admission
under-budgeting is recorded as an open defect; it is not fixed here.

## Curve fit and expert-byte conclusion

Using the requested two-point model `T = fixed + marginal * prompt_tokens`:

- Strata fixed term: **23.831006 s**
- Strata marginal term: **59.192315 ms/token**
- Reference: **13.31 s + 1.14 ms/token**

Strata is therefore +10.521006 s in the fitted fixed term and +58.052315
ms/token in the marginal term. This is a two-point, single-repetition curve
measurement, not a promotion claim.

Expert demand H2D bytes rise by **17,740,343,808 B (17.740 GB, +23.85%)** from
677 to 2,612 tokens: 74.381 GB to 92.121 GB. The long run is well below the
reference's 156.9 GB one-time expert stream and does not show re-reading past
that volume. It does not yet demonstrate the reference's fixed-cost behavior;
the bytes still rise with prompt length, but the increase is not a full expert
set reread.

## Verdict

The explicit 4G KV host budget restores the 2,612-token capability. Both
requested arms completed with correct generated IDs and zero decode checkpoint
reads. The measurement establishes the landed curve and the KV admission
under-budget defect, while making no mechanism or throughput promotion claim.
