# DeepSeek V4 capacity-bounded CUDA weight arena

Status: **accepted on the experiment branch; median exact decode +10.1%**.

## Contract

- Hypothesis: replacing per-weight `cudaMalloc`/`cudaFree` with one reusable,
  capacity-bounded arena per GPU materially reduces the measured DeepSeek MoE
  preparation bottleneck.
- Primary metric: median exact decode tokens/second across three interleaved
  baseline/candidate pairs.
- Correctness gate: identical generated token IDs, exact device-MoE path,
  balanced leases, unchanged routing/cache behavior, and zero decode checkpoint
  reads.
- Memory ceiling: 216 GiB host admission and 0.85 VRAM fraction, with the
  existing 256 MiB per-device workspace reserve.
- Rollback: reject on correctness or admission change, arena fallback, memory
  overrun, or a decode result within observed variance.

The baseline is commit `069b01d`. Both executables used the same model, prompt
(`Hello`), 128 requested new tokens, devices `0,1,2`, 28 attention workers,
detailed timing, and exact device MoE. Power and hardware telemetry capture was
disabled. Runs were ordered baseline/candidate for each repetition.

## Results

| Pair | Baseline tok/s | Arena tok/s | Baseline decode s | Arena decode s |
|---|---:|---:|---:|---:|
| 1 | 4.5661 | 4.8930 | 27.8134 | 25.9556 |
| 2 | 4.2132 | 5.0314 | 30.1432 | 25.2417 |
| 3 | 4.5814 | 5.0288 | 27.7205 | 25.2543 |
| **Median** | **4.5661** | **5.0288** | **27.8134** | **25.2543** |

The ratio of medians is a **10.13% decode throughput improvement**, with median
decode wall time down **9.20%**. All three paired deltas were positive. An
earlier candidate screen produced 4.5402 tok/s and is reported but excluded
from the interleaved median.

The mechanism matched the hypothesis:

| Median phase/metric | Baseline | Arena | Change |
|---|---:|---:|---:|
| MoE preparation | 5.5484 s | 3.4131 s | -38.49% |
| Total MoE | 11.8930 s | 9.6518 s | -18.84% |
| Decode weight allocations | 7,716 | 0 | -100% |
| Prefill wall | 4.5967 s | 3.2035 s | -30.31% |

The candidate reserves three CUDA allocations totaling 55,729,582,592 bytes,
the admitted weight-cache budgets rounded down by at most 255 bytes per device.
Suballocations are aligned, coalesced on release, and have no per-weight
allocation fallback. Candidate peak RSS was 138.61 GiB, below the 216 GiB
ceiling. Cache usage was 38,427,963,904 bytes, with zero evictions.

Every run generated the same 127 token IDs. Every decode recorded zero
checkpoint-read bytes, 114,681 lease acquires and releases, and no active lease
at completion. `make check` passed 2/2, including strict arena exhaustion,
reuse, coalescing, and the existing exact CUDA kernel oracles.

## Decision

Accept the arena as a measured, exact improvement. It removes allocation
variance and makes the candidate substantially more stable (4.8930--5.0314
tok/s versus 4.2132--4.5814 tok/s). The next large boundary is no longer an
allocator tweak: about 19.6 seconds of median decode remain in host mHC and
attention plus device MoE execution. A materially larger step requires keeping
more of the token graph resident on GPU and removing host/device round trips.

Artifacts are ignored under `results/deepseek-v4-weight-arena-screen/` and
`results/deepseek-v4-weight-arena-matrix/`.
