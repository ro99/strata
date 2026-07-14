# Strata

Strata is a ground-up C/C++ inference engine for models that are larger than
local VRAM. It targets RAM-rich, VRAM-poor workstations and small local clusters,
with first-class support planned for dense transformers, conventional MoE, and
DeepSeek's MLA + fine-grained/shared-expert architecture.

This is not a Colibri fork. Colibri is an important experimental baseline;
Strata changes the unit of scheduling from a synchronous token-forward to
expert work tickets moving through a bounded execution wavefront.

## Non-negotiable rules

- Model weights are **int4 or higher precision**. Sub-4-bit weights, shadow
  models, predictors, caches, and fallback representations are forbidden.
- A predictor may change prefetch and placement only. It may never change the
  exact router result or emitted model output.
- No silent CPU fallback, expert dropping, reduced top-k, or altered routing.
- NVMe is immutable cold backing. An admitted steady-state request should not
  depend on token-by-token storage reads.
- Runtime code is C++20 with C kernel ABIs. Python and heavyweight frameworks
  are not runtime dependencies.

## Current milestone

The repository currently provides the testable foundation for storage-policy
research:

- model and DeepSeek semantic validation;
- an ABI-stable model header with a four-bit precision floor;
- an exact signed-int4 CPU matvec oracle;
- a sequential route-trace format and parser;
- an online, no-future-information route-transition predictor;
- VRAM/RAM/peer/NVMe residency simulation;
- LRU, LFU, and lease-aware replacement;
- explicit NVMe, H2D, network, useful-prefetch, wasted-prefetch, and cold-budget
  accounting;
- a dependency-free test harness.

It is not yet a language-model inference binary. The next gate is a real
GLM/DeepSeek route trace, followed by an int4 model packer and a tiny-model
forward oracle.

## Build and test

```bash
make check
```

For a sanitizer build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DSTRATA_ENABLE_SANITIZERS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Route simulation

Trace format:

```text
# strata-route-trace-v1
# REQUEST TOKEN LAYER EXPERT [EXPERT ...]
0 0 3 12 97 103 211
0 0 4 8 44 131 190
```

Compare replacement policies at a fixed capacity:

```bash
./build/strata-sim \
  --trace tests/data/route_trace_v1.txt \
  --compare \
  --ram 180G \
  --vram 58G \
  --expert-bytes 19M \
  --prefetch 8 \
  --lease 16
```

`--cold-read-budget` and `--strict` model the planned admission contract. Peer
execution can be explored with `--peer-resident-pct` and
`--peer-activation-bytes`.

## Architecture in one paragraph

Attention, routers, KV state, and shared experts form a resident model spine.
Exact routers emit expert tickets into persistent queues. The scheduler groups
tickets across requests and speculative verification rows, sends work to the
device or peer that already owns the canonical int4 expert, and scatters weighted
partial results back into device-resident hidden states. Requests receive an
explicit residency/I/O contract before admission. See
[`docs/architecture.md`](docs/architecture.md).

## Repository layout

```text
apps/             command-line tools
include/strata/   public C and C++ interfaces
src/              architecture and scheduling core
kernels/cpu/      CPU reference and optimized kernels
kernels/cuda/     future optional CUDA backend
tests/            dependency-free correctness tests and fixtures
tools/            offline conversion and trace tools
docs/             architecture, format, roadmap, experiments
```

## License

Apache-2.0.
