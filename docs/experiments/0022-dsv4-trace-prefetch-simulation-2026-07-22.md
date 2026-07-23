# Experiment 0022 — trace-driven DeepSeek expert prefetch simulation

## Contract

- Hypothesis: two past-only transition predictions at 0.75 confidence reduce
  demand reads by more bytes than incorrect predictions add.
- Primary metric: total modeled read bytes and useful-prefetch ratio on an
  archived 5,676-event sequential DeepSeek route trace.
- Correctness: prediction remains advisory; exact routes, coefficients, top-k,
  and precision are unchanged.
- Memory: identical 1 GB RAM, 200 MB VRAM, 19 MB expert pages, and lease 16.
- Rollback: do not promote if total bytes rise or usefulness is at most 50%.

## Interleaved simulation

The command below was run baseline/candidate for three interleaved repetitions;
the simulator is deterministic, so each repetition matched its peers.

```bash
build/strata-sim \
  --trace results/deepseek-v4-device-moe-attention-workers-screen/runs/reference/run-01/routes.jsonl \
  --ram 1G --vram 200M --expert-bytes 19M \
  --prefetch N --lease 16 --confidence 0.75 --json
```

`N` was 0 for each baseline and 2 for each candidate.

| Median | No prefetch | Two predictions |
|---|---:|---:|
| Total read bytes | 647,064,000,000 | 597,797,000,000 |
| Demand misses | 34,056 | 24,533 |
| Prefetch bytes | 0 | 131,670,000,000 |
| Useful / wasted prefetches | 0 / 0 | 5,867 / 1,063 |
| Useful-prefetch ratio | n/a | 84.66% |

The candidate reduces total modeled bytes by 49,267,000,000 (7.61%) and clears
the simulation gate. This result authorizes an opt-in runtime experiment; it is
not an end-to-end throughput claim.
