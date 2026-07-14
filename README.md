# Strata

**Run models larger than VRAM—without turning the SSD into the decode loop.**

Strata is a ground-up C/C++ inference engine for dense and mixture-of-experts
models on memory-constrained workstations and small local clusters. It is built
for machines with useful CPU RAM, limited GPU memory, and models whose weights
do not fit in either one alone.

The central idea is simple: **move small activation batches to resident weights,
not giant weight matrices to one token.** Strata keeps a model's dense spine and
shared experts resident, converts exact router decisions into expert work
tickets, and combines tickets across requests into an expert-centric execution
wavefront. NVMe is immutable cold storage and recovery—not a normal per-token
memory tier.

## Why Strata exists

Most offload designs optimize how quickly a cache miss can be served. Strata
tries to make the miss disappear from steady-state decode:

- **Resident Execution Contracts** admit a request only with an explicit local,
  peer-backed, or bounded-cold-read plan.
- **Expert-centric wavefronts** expose reuse across requests instead of forcing
  every row through a synchronous load-compute-evict cycle.
- **DeepSeek is a native architecture**, including MLA, dense prefixes,
  fine-grained routed experts, shared experts, group-limited routing, and exact
  selection semantics.
- **Expert Compute Fabric** sends compact activation rows to LAN peers that own
  the weights. It does not pretend remote RAM is a fast disk.
- **Prediction is advisory only.** It can place and prefetch exact weights but
  can never alter routing or model output.
- **No framework runtime.** The engine is C++20 with stable C kernel ABIs; CUDA
  is an optional backend.

For dense models, Strata is deliberately honest: when every weight is used on
every token and the model exceeds aggregate resident memory, storage bandwidth
remains a hard limit. The largest gains are expected for sparse MoE models.

## Status

Strata is an early performance-research engine, not yet a language-model
inference binary. The repository currently contains:

- model and DeepSeek semantic validation;
- a fixed model-pack header with a four-bit precision floor;
- a signed-int4 CPU matvec correctness oracle;
- a sequential route-trace parser and past-only transition predictor;
- a VRAM/RAM/peer/NVMe residency simulator;
- LRU, LFU, and lease-aware policy experiments with explicit byte accounting;
- dependency-free tests.

The CPU oracle validates packing and arithmetic. It is not the final production
quantization layout or optimized kernel.

## Start here

**Do not download GLM-5.2 or a full DeepSeek-V4-class checkpoint yet.** Those
models are expensive acceptance tests, not productive bring-up targets. Starting
there would mix format, quantization, architecture, kernel, scheduler, and
storage bugs into one multi-hour feedback loop.

The next development branch is:

```text
feat/modelpack-q4k64
```

Its only job is to implement a resumable, shard-streaming `strata-pack` tool and
the first production weight layout:

```text
Q4_K64: signed two's-complement int4 groups of 64 weights + one FP16 scale per group
execution: W4A16, with FP32 accumulation in the reference path
```

The first fixtures are generated tensors measured in kilobytes—not a downloaded
model. The branch is complete when `inspect`, pack, unpack, hashing, atomic
rename, interruption recovery, and numerical-error tests pass. It must never
load an entire checkpoint into RAM.

Then bring up models in this order:

| Stage | Reference target | What it proves | Download now? |
|---|---|---|---|
| 0 | Deterministic generated tensors | Container, Q4_K64 codec, hashes, resumability | **No model download** |
| 1 | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Llama-style dense graph, tokenizer boundary, logits, generation | After Stage 0 |
| 2 | `allenai/OLMoE-1B-7B-0125` | Conventional top-k MoE and expert batching | After dense oracle passes |
| 3 | `deepseek-ai/DeepSeek-V2-Lite` | MLA plus fine-grained and shared-expert execution | After standard MoE passes |
| 4 | GLM-5.2 and the selected DeepSeek-V4-class target | Scale, placement, CUDA, and NVMe-wear acceptance | Only after the smaller gates pass |

This is architecture-first support, not a hard-coded model whitelist. A new
checkpoint is supported when its operator graph, tensor mapping, routing
semantics, tokenizer boundary, and reference tests are covered. A familiar
`config.json` alone is not enough.

The detailed sequence, quality gates, and disk-space rules are in
[`docs/model-bringup.md`](docs/model-bringup.md).

## Quantization contract

Strata never emits or consumes weights, predictors, drafts, shadow models, or
fallback representations below four-bit precision.

Q4_K64 is the first general-purpose candidate, not a requirement that every
tensor become int4. The packer may preserve routers, normalization parameters,
embeddings, output heads, attention projections, and measured outliers in
BF16/FP16 when a quality gate justifies it. Native int4 checkpoints should be
preserved when their layout and semantics are validated; BF16/FP16 source
checkpoints are quantized offline, shard by shard.

No large conversion is accepted without comparison to the source model using
fixed logits, teacher forcing, greedy generation, and a declared evaluation
set. Smaller files are not evidence of a usable model.

## Build the current foundation

Requirements: a C++20 compiler, CMake, and Make.

```bash
make check
```

For a sanitizer build:

```bash
cmake -S . -B build-asan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DSTRATA_ENABLE_SANITIZERS=ON
cmake --build build-asan --parallel
ctest --test-dir build-asan --output-on-failure
```

## Explore storage policy today

The existing `strata-sim` tool replays exact expert routes against fixed memory
budgets. It measures cold reads, demand/prefetch bytes, useful and wasted
prefetches, H2D traffic, peer traffic, evictions, and contract violations.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

./build/strata-sim \
  --trace tests/data/route_trace_v1.txt \
  --compare \
  --ram 180G \
  --vram 58G \
  --expert-bytes 19M \
  --prefetch 8 \
  --lease 16
```

Peer execution can be modeled with `--peer-resident-pct` and
`--peer-activation-bytes`. Use `--cold-read-budget` with `--strict` to test an
admission contract.

## The execution loop

1. Admission estimates the request's resident expert cohort, KV allocation,
   peer coverage, and maximum permitted cold reads.
2. Attention, normalization, routers, KV state, and shared experts run as a
   resident model spine.
3. Exact router results produce expert tickets; they do not immediately trigger
   disk reads.
4. The scheduler groups compatible tickets across ready rows until a bounded
   latency deadline.
5. Work executes where the canonical expert already lives: VRAM, local RAM, or
   a peer worker. Only a declared bounded-cold contract may reach NVMe.
6. Weighted results return to the original row, which advances through the
   exact model graph.

Read [`docs/architecture.md`](docs/architecture.md) for the full design and
[`docs/research-roadmap.md`](docs/research-roadmap.md) for the gated build plan.

## Repository layout

```text
apps/             command-line tools
include/strata/   public C and C++ interfaces
src/              architecture and scheduling core
kernels/cpu/      numerical oracles and optimized CPU kernels
kernels/cuda/     optional CUDA backend
tests/            dependency-free correctness tests and fixtures
tools/            offline packing, conversion, trace, and replay tools
docs/             architecture, formats, roadmap, and experiment records
```

## Engineering rules

- Exact model semantics by default: no expert dropping, reduced top-k, hidden
  CPU fallback, or approximate router substitution.
- Model files are immutable at runtime. Cache metadata and routing history live
  elsewhere.
- Every performance claim needs reproducible A/B measurements and correctness
  gates.
- Complexity earns its place only through evidence.

## License

Apache-2.0.
