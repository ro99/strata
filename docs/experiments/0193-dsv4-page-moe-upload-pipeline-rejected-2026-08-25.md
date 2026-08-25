# Experiment 0193: DeepSeek page-MoE upload pipeline rejected

Date: 2026-08-25  
Branch: `fix/dsv4-page-moe-pipeline`  
Depends on: experiments 0190 and 0191

## Decision

Reject fixed-wave overlap of cold expert upload with page-MoE execution. At an
identical 133-token production shape, the candidate moved exactly the same
52.6 GB and produced the same token, but increased demand wait from 16.89 to
23.68 seconds and regressed total prefill by 26.9%.

The runtime and backend changes were removed. The predeclared gate forbade
searching another wave size after this negative result.

## Predeclared hypothesis and gates

- Hypothesis: whole-page preparation serializes all cold expert H2D before any
  page GEMM. Divide each rank's distinct routed experts into fixed groups of 32
  and upload wave N+1 on the existing upload stream while wave N executes.
- Target term: the serial boundary between MoE preparation and execution. At
  1,925 tokens in experiment 0190, MoE was the 44.34-second phase `argmax`,
  comprising 20.00 seconds preparation and 24.34 seconds execution/join.
- Primary metric: detailed prefill wall time and MoE time, screened cheaply at
  133 tokens before the planned 619- and 1,925-token arms.
- Correctness: preserve expert groups, row order, coefficients, kernels, and
  final rank order; require the page-MoE fixture, `make check`, and the full
  DeepSeek oracle only if the performance screen passed.
- Memory ceiling: unchanged 0.95 VRAM admission. Two waves held separate cache
  leases but introduced no persistent allocation.
- Rollback: any output drift, H2D-volume increase, admission failure, or no
  greater-than-5% improvement in both MoE and total prefill.

The faithful screen required about 106 seconds setup for a 20--30 second
measured window, approximately 4.5:1. A synthetic copy/GEMM microbenchmark was
rejected because it would not reproduce the runtime's weight-cache eviction,
lease ownership, or two unequal PCIe links. The two-arm budget was five
minutes; both arms completed inside it.

## Mechanism

The candidate retained the existing routed group construction and final host
reduction. For each fixed 32-expert wave it:

1. acquired distinct weight-cache leases for the current and next waves;
2. enqueued the current wave on the CUDA execution stream;
3. copied the next wave through the already-existing CUDA upload stream;
4. inserted an upload event dependency after current execution and before the
   next wave could consume its weights; and
5. collected each wave directly into its original group-major output span.

The backend's synchronous-upload exclusion remained intact. Only deferred
weights in distinct leased arena allocations were allowed to overlap an
in-flight MoE command.

## Matched screen

Both arms used Release+NCCL builds, checkpoint `models/dsv4f`, physical devices
1 and 2 under `CUDA_DEVICE_ORDER=PCI_BUS_ID`, 1,605 MHz SM clocks, 250 W caps,
rank-local TP2, 0.95 VRAM fraction, 16,384 context, an 8,192-token page upper
bound, no static expert tier, 133 actual prompt tokens, and one generated
token. The baseline was rebuilt from commit `cbb6ffe` in a detached worktree.

| metric | baseline | pipeline | pipeline/base |
|---|---:|---:|---:|
| prefill seconds | 22.33 | 28.33 | **1.269** |
| prefill tok/s | 5.96 | 4.69 | **0.787** |
| attention seconds | 2.92 | 2.87 | 0.983 |
| MoE seconds | 18.52 | 24.58 | **1.327** |
| demand H2D bytes | 52.6 GB | 52.6 GB | 1.000 |
| demand wait seconds | 16.89 | 23.68 | **1.402** |
| cache misses | 7,432 | 7,432 | 1.000 |
| cache evictions | 5,219 | 5,219 | 1.000 |
| generated token | 2107 (`It`) | 2107 (`It`) | identical |

Ignored raw results:

- `results/0193-dsv4-page-moe-pipeline/baseline-smoke/page8192-w80.json`
- `results/0193-dsv4-page-moe-pipeline/smoke2/page8192-w80.json`

The regression is far larger than observed run variance, so the negative
screen invokes rollback. It is not reported as a win and did not justify the
long repetition matrix or full-model oracle.

## Cost-model interpretation

The experiment separated volume from overlap: work volume, cache misses, and
evictions were byte-for-byte/count-for-count unchanged. Only service time grew.
Holding the current wave's leases while acquiring the next reduced eviction
freedom and made DMA staging contend with the active MoE path. Demand wait rose
40.2%, more than erasing any possible compute/upload overlap.

Under `tau = max_r(W_r / B_r) + sum(serial)`, the intended serial boundary did
not shrink. The candidate inflated the controlling H2D service term while
leaving attention, routing, mHC, arithmetic work, precision, and output volume
unchanged. The sign is therefore negative.

## Outcome

Rejected and fully rolled back. A future overlap mechanism needs residency or
workspace ownership that lets copy and compute proceed without constraining
the production weight cache. Wave-size tuning is not evidence for that
mechanism and was not attempted.
