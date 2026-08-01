# GLM W4A16 short-context indexer no-op screen — 2026-08-01

## Decision

Rejected as a decode-throughput optimization. Keep the result, remove the runtime change.

GLM DSA selects every key while context is at most 2,048 tokens. The screened change
therefore skipped indexer weights, projections, and cache storage when the admitted
maximum context was 256. This removed work exactly, but did not reduce the measured
decode bottleneck.

## Single falsification arm

Named session: `strata-glm-index-noop-screen`

Result: `results/glm-w4a16-index-noop-2026-08-01/run-01.json`

| Metric | Integrated parity | Indexer no-op |
|---|---:|---:|
| generated IDs | `[16,13]` | `[16,13]` |
| decode seconds | 4.655449 | 4.665267 |
| decode tok/s | 0.214802 | 0.214350 |
| decode checkpoint bytes | 9,790,500,864 | 9,751,572,480 |
| decode weight H2D bytes | 9,790,488,576 | 9,751,560,192 |
| decode activation H2D bytes | 23,261,184 | 22,056,960 |
| decode activation D2H bytes | 303,174,272 | 302,816,768 |
| decode synchronizations | 2,343 | 2,274 |
| decode attention seconds | 1.008611 | 1.009521 |
| decode MoE seconds | 3.619490 | 3.626455 |

The arm removed 38,928,384 checkpoint/H2D bytes and 69 synchronizations, but decode
was unchanged within noise and remained 26.0% slower in time than the 3.701097-second
pre-parity baseline. Maximum RSS was 2,274,020 KiB. Physical reads were 65,536 bytes
from a warm page cache; physical writes were 12,926,976 bytes.

The next independent hypothesis is MLA weight absorption, targeting the compact-KV
reconstruction traffic rather than the non-bottleneck indexer work.
