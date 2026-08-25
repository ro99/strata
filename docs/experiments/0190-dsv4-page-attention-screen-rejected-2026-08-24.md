# Experiment 0190: DeepSeek page-attention screen rejected

Date: 2026-08-24  
Branch: `fix/dsv4-page-attention`  
Origin: production-throughput investigation requested after a two-turn chat
reported 2.32 prefill tok/s and 6.53 decode tok/s.

## Decision

Do not change the sparse-attention CUDA kernel yet. At the production
operating point, MoE is the largest prefill phase and the proposed score-kernel
change would reduce only 4.13 seconds of a 100.42-second forward. The larger
attention cost is outside that kernel, while the MoE page contains a directly
observed serial host reduction.

This is a measured mechanism rejection, not a claim that attention is fast.
Attention remains almost tied with MoE and needs its own follow-up after the MoE
serial term is removed.

## Predeclared screen

- Hypothesis: the sparse physical-page score kernel is the long-prompt prefill
  bottleneck because each block loops over all candidates with two block-wide
  barriers per candidate.
- Primary metric: prefill phase time and the maximum-device paged-attention
  kernel time at roughly 2,000 prompt tokens.
- Correctness gate for any later kernel: the physical-page operation/layer
  fixtures and the full DeepSeek numerical oracle, with no precision, route,
  layout, or arithmetic-contract change.
- Memory ceiling: the existing explicit 0.95 VRAM admission, 22,135,873,536
  bytes per device; no reduction in the expert-cache budget.
- Rollback: do not build the kernel if paged-attention CUDA is not the measured
  `argmax` term.

The arm used 105.80 seconds of fixed setup and 100.42 seconds of measured
prefill, a 1.05:1 ratio. The cheaper 619-token detailed profile was rejected as
the sole decision input because it left attention and MoE close and does not
reproduce the long page shape under investigation. Budget was 3.2 minutes; the
arm completed in 3.6 minutes because detailed timing perturbs the CPU path.

## Operating point

- Release build from `main` plus documentation-only experiments 0188-0189
- checkpoint `models/dsv4f`
- physical devices 1 and 2 under `CUDA_DEVICE_ORDER=PCI_BUS_ID`
- 2 x RTX 3090, SM clock 1605 MHz, 250 W cap
- PCIe links: Gen3 x8 and Gen3 x16, same NUMA node
- rank-local TP2, NCCL enabled
- `--vram-fraction 0.95 --max-context 16384`
- `--prefill-page-tokens 8192`, no static expert tier
- 1,925 actual prompt tokens, one generated token
- ignored raw result:
  `results/0190-dsv4-page-attention/profile-main/page8192-w1200.json`

## Cost model

The governing model is

```text
tau = max_r(W_r / B_r) + sum(serial boundaries)
```

For this prefill forward, phases are sequential and each phase has its own
resource maximum. The measured wall-time decomposition is:

| phase | seconds | ms/token | controlling observation |
|---|---:|---:|---|
| embedding + mHC pre | 0.22 | 0.11 | negligible |
| attention | 42.25 | 21.95 | CPU/query and synchronization dominate the 4.13 s sparse CUDA kernel |
| MoE router | 4.35 | 2.26 | serial exact per-row route construction |
| MoE prepare | 20.00 | 10.39 | 87.30 GB demand H2D, 19.97 s wait |
| MoE execution/join | 24.34 | 12.64 | device command 8.57 s maximum; remaining host collection/reduction is serial |
| mHC post | 9.09 | 4.72 | device/host transition work |
| **total** | **100.42** | **52.17** | **MoE is phase argmax: 44.34 s** |

Resource volumes and achieved rates at the real point:

| resource | work | measured time/rate | rated/structural check |
|---|---:|---:|---|
| routed-weight H2D | 87.30 GB | 19.97 s, 4.37 GB/s aggregate | 23.63 GB/s aggregate link line rate; 18.5%, not an order-of-magnitude proof by itself |
| all CUDA kernels | n/a | 17.85 s | useful compute, split across attention, MoE, projections, and mHC |
| activation H2D | 16.71 GB | 2.65 s critical path | 6.31 GB/s |
| activation D2H | 23.51 GB | 6.78 s critical path | 3.47 GB/s |
| paged attention | 5.88 GB H2D, 0.68 GB D2H | 4.13 s kernel, 4.69 s device command | kernel is only 9.8% of its 42.25 s phase |
| page-MoE outputs | 9.49 GB D2H | 1.35 s transfer | followed by a serial host rank-order reduction |

The full graph reports 44.34 seconds in MoE and 42.25 seconds in attention.
Therefore `argmax_r` for the next bounded prefill mechanism is the MoE serial
path, specifically the host rank-order reduction after page expert outputs are
downloaded. The 87.30 GB weight transfer is also large, but experiments 0188
and 0189 already falsified the two cheap pinning mechanisms without changing
the operating point.

## Sign check

Removing barriers inside the sparse score kernel would reduce GPU compute by
at most the observed 4.13-second kernel term. It would not reduce the 20.00 s
MoE preparation, 24.34 s MoE execution/join, query projection, CPU RMS/RoPE,
or mHC terms. A direct-KV version would also increase global-memory reads by up
to eightfold to remove synchronization. Under the measured cost model that is
not the first mechanism to build.

The next hypothesis instead parallelizes the existing exact host page-MoE
rank-order reduction across independent rows. It reduces a measured serial
term, preserves the accumulation order within every row, adds only bounded
worker-pool CPU load after device collection, and changes neither transfer
volume nor VRAM.

## Prefill/decode shape check

Before measurement the expected prefill/decode per-token ratio was below 0.25
once a page amortized weights. The production server baselines were:

| shape | prefill | prefill ms/token | decode ms/token | ratio |
|---|---:|---:|---:|---:|
| about 500 tokens | 17.43 tok/s | 57.37 | 115.92 | 0.495 |
| about 1,950 tokens | 22.76 tok/s | 43.94 | 115.92 | 0.379 |

Both fail the predicted ratio and reproduce the open throughput defect. The
detailed arm is slower because observability perturbs it; it is used only for
attribution, not as the production throughput headline.

## Outcome

Gate negative. No runtime code was changed. The sparse-attention candidate is
rejected at this stage because it does not reduce the current phase `argmax`.
The raw profile stays ignored; this record is the durable evidence.
