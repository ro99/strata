# Experiment 0191: DeepSeek parallel page-MoE reduction rejected

Date: 2026-08-25  
Branch: `fix/dsv4-page-moe-reduction`  
Depends on: experiment 0190 cost model

## Decision

Reject parallel host reduction of independent prefill rows. It shortened the
page-MoE execution tail, but waking the existing 48 pinned expert workers just
before the next layer's weight acquisition contended with demand H2D staging.
The larger resource moved in the wrong direction: expert wait nearly doubled,
and prefill regressed by 23.5%.

The runtime change was removed. No reduced thread count or alternate page size
was searched after this production-point gate failed.

## Predeclared hypothesis and gates

- Hypothesis: page expert outputs are reduced serially on one host thread even
  though rows are independent; preserve rank 0..5 then shared ordering within
  each row while distributing rows across the existing expert worker pool.
- Target term: the serial host portion of MoE execution/join, measured after
  the two device collections. MoE was the 44.34-second phase `argmax` in 0190.
- Primary metric: 619-token detailed prefill and its MoE phase.
- Correctness: identical per-row accumulation order, then `make check` and the
  full DeepSeek oracle only if the performance screen passed.
- Memory ceiling: unchanged 0.95 explicit VRAM budget, 22,135,873,536 bytes per
  device. The candidate allocated no new persistent memory.
- Rollback: any correctness drift, no material throughput improvement, or an
  increase in another bottleneck resource.

The faithful arm required about 106 seconds setup for an expected 41-second
measured window (2.6:1). A synthetic array-loop probe was rejected because it
would not reproduce NUMA placement or GPU DMA-written output buffers. Arm
budget was 2.5 minutes; it completed in 2.9 minutes.

## Mechanism

The candidate:

1. preserved the exact rank 0, 1, 2, 3, 4, 5, shared accumulation order inside
   every row;
2. assigned independent rows to `expert_workers->parallel_for()`;
3. fused the existing idempotent BF16 round of downloaded expert outputs into
   their only reduction read, removing one serial memory pass;
4. changed no kernel, transfer volume, route, coefficient, precision, or VRAM
   allocation.

## Result

Same Release build, checkpoint, devices, 1605 MHz clock, 250 W cap, rank-local
TP2, 0.95 VRAM fraction, 16,384 context, and 8,192-token page as 0190.

| metric | main baseline | candidate | candidate/base |
|---|---:|---:|---:|
| prompt tokens | 619 | 619 | 1.000 |
| prefill seconds | 40.777 | 53.289 | 1.307 |
| prefill tok/s | 15.18 | 11.62 | **0.765** |
| attention seconds | 11.802 | 10.912 | 0.925 |
| MoE seconds | 24.663 | 38.236 | **1.550** |
| MoE prepare seconds | 16.872 | 32.036 | **1.899** |
| demand H2D bytes | 72.484 GB | 72.484 GB | 1.000 |
| demand wait seconds | 16.880 | 32.043 | **1.898** |
| maximum device MoE command seconds | 7.863 | 2.735 | 0.348 |

Raw candidate result, ignored by Git:
`results/0191-dsv4-parallel-page-moe/screen/page8192-w384.json`.

The large negative is outside any plausible run variance, so one screen is
sufficient to invoke the predeclared rollback. It is not reported as an A/B
win and no repetition matrix was run after the negative gate.

## Cost-model interpretation

The mechanism did reduce its intended serial execution term: maximum device
MoE command time plus the host join fell substantially. But the sign check on
the controlling preparation resource failed. The pool uses 48 pinned workers
with a one-millisecond idle spin. After each reduction those workers remain
active while the next layer stages cold expert slices from the resident arena.
Transfer volume stayed byte-identical while achieved service time nearly
doubled, proving contention/serialization rather than extra work.

Under `tau = max_r W_r/B_r + sum(serial)`, shrinking the host join while
inflating the larger demand-staging term is negative. A smaller worker count or
different idle policy would be an attempt to manufacture a favorable regime
after the production gate failed, so it was not pursued.

## Outcome

Rejected and rolled back. The next mechanism must avoid competing host cores
and memory bandwidth during the next layer's weight upload—preferably by
eliminating the host page join or moving it behind an already-existing device
boundary rather than adding CPU concurrency.
