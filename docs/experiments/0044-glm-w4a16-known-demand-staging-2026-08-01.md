# GLM W4A16 known-demand expert staging

Status: **rejected after one binding run**.

## Contract

- Hypothesis: reading the next already-routed expert while the current expert
  executes reduces the measured routed-MoE serial term.
- Primary metric: exact decode tokens/s.
- Correctness gate: identical greedy token IDs and unchanged checkpoint/H2D
  bytes.
- Memory ceiling: one transient expert triplet per active device.
- Rollback: remove staging if the single screening run is negative. The user
  explicitly required one run rather than a three-pair gate at this phase.

The candidate did not predict routes or skip work. Each device staged only its
next known gate/up/down triplet and passed those bytes into the ordinary W4A16
upload path.

## Result

Both measurements used the same checkpoint, prompt, three devices, 0.85 VRAM
fraction, FlashAttention path, and one measured decode step.

| Metric | Existing path | Staged path | Change |
|---|---:|---:|---:|
| Decode tokens/s | 0.27019 | 0.24246 | -10.3% |
| Decode seconds | 3.701 | 4.124 | +11.4% |
| Routed MoE seconds | 3.122 | 3.515 | +12.6% |
| Checkpoint bytes | 9.7516 GB | 9.7516 GB | 0% |
| Checkpoint wall time | 1.639 s | 1.663 s | +1.5% |
| W4A16 H2D bytes | 9.7516 GB | 9.7516 GB | 0% |
| End-of-run RSS | 798 MB | 1,002 MB | +25.5% |

Both generated `[16,13]`. The candidate issued 600 staging requests, all
9.7516 GB were consumed, and no staged bytes were duplicates. Its critical
stage waits still totalled 1.498 seconds, so the mechanism moved reads to many
short-lived tasks without hiding the storage service time.

Artifacts are under `results/glm-w4a16-known-demand-overlap/`.

## Decision

The exact path and I/O volume were unchanged, but routed-MoE service became
slower. No repetitions were run. The staging implementation and its metrics
were removed; this operating point does not justify another expert prefetch
variant.
