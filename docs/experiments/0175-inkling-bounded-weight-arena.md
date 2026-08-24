# Experiment 0175 — Inkling removes per-miss CUDA allocation

Status: **ACCEPTED.** Inkling now reserves one bounded weight arena per device
before uploading the resident spine. Routed-expert cache entries suballocate
from the remainder instead of calling `cudaMalloc`/`cudaFree` for each of their
three projections.

## Hypothesis and resource accounting

The direct-mapped 128-token profile spent 10.31 seconds in expert staging, while
the CUDA allocation/copy/wait and routed kernel counters explained only about
4.23 seconds. Unlike Laguna, Inkling had never enabled the backend's coalescing
weight arena. Cache misses and evictions therefore repeatedly entered the CUDA
allocator and driver synchronization path.

- Primary metric: decode tokens/s and expert-stage wall time on identical
  greedy routes.
- Correctness: identical continuation and CUDA route census, followed by
  `make check`.
- Memory: the arena is the existing 85% VRAM budget, not an additional
  allocation. Resident spine bytes are subtracted before the remainder is
  exposed to the expert LRU. Exact K/V and workspaces stay in the existing 15%
  headroom.
- Target resource: serial driver allocation/free time and fragmentation.
- Other resources: expert H2D bytes, model arithmetic, precision, routing,
  prepack work, and aggregate VRAM do not change.
- Rollback: admission failure, numerical/census difference, or a stage median
  within run variance.

## Results

Both matrices used page-cache-resident checkpoint mappings, no expert warmup,
three interleaved fresh processes per arm, and
`CUDA_DEVICE_ORDER=FASTEST_FIRST` over two RTX 3090s plus one RTX 5060 Ti.

| workload | control median | arena median | decode speedup | stage wall |
|---|---:|---:|---:|---:|
| 16 generated tokens | 4.793 tok/s | 5.384 tok/s | 1.123x | 2.760 → 2.400 s (1.150x) |
| 128 generated tokens | 7.911 tok/s | 8.738 tok/s | 1.105x | 9.215 → 7.836 s (1.176x) |

The short arm's per-run rates were `4.793, 4.794, 4.751` for the control and
`5.325, 5.384, 5.396` for the arena. The long arm's were
`8.013, 7.850, 7.911` and `8.748, 8.408, 8.738`. Every arm emitted the identical
continuation and route census (`fp4_register_fed=4336,
moe_fp4_register_fed=1600` short; `28640, 10560` long).

At 128 tokens, measured decode allocation time fell from a 0.489-second median
to effectively zero. The arena's slightly stricter all-in 85% accounting caused
one additional cache miss (`3,357 → 3,358`) and 11 additional evictions, so the
wall-clock win is not a favourable-cache artifact.

## Failed first cut

The first eviction-heavy candidate failed explicitly with:

```
CUDA weight arena is exhausted; refusing per-weight allocation fallback
```

`InklingExpertCache::acquire` uploads the incoming expert before applying its
logical LRU capacity check. Per-weight allocation had hidden that temporary
peak in otherwise free VRAM. The accepted implementation reserves one
worst-case plain-BF16 Inkling expert (48 MiB) inside each already-admitted arena
as transient upload space. It neither expands the budget nor falls back.

## Reproduction

```bash
cmake --build build-release --target strata-inkling-probe -j
scripts/inkling_weight_arena_ab.sh
RESULT_DIR=results/inkling-weight-arena-128b TOKENS=128 \
  scripts/inkling_weight_arena_ab.sh
```

Raw logs are ignored under `results/inkling-weight-arena*`.

## Remaining bottleneck

The arena removes allocation churn but does not change the serial upload
schedule. The 128-token candidate still spends 7.84 seconds staging 31.58 GiB.
Laguna already uploads persistent mapped weights on the copy stream and runs a
token's experts across their owning devices concurrently; Inkling still performs
three synchronous uploads per expert and puts all six routed experts on the
layer's one device. Those are separate measured hypotheses, not part of this
commit.
