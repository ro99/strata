# Strata

**Run frontier-scale MoE models across VRAM, RAM, and read-only NVMe—without
making storage the token loop.**

Strata is a ground-up C/C++ inference engine for enormous sparse models on
consumer workstations and small local clusters. It is designed around the two
models that expose the real problem immediately:

- **GLM-5.2** is the primary engineering and performance target.
- **DeepSeek-V4-Flash-DSpark** is the native four-bit, resident proof of concept.

The central idea is to move small activation batches to resident weights, not
giant expert matrices to one token. Exact router decisions become expert work
tickets; the scheduler groups tickets across requests and speculative rows;
execution happens where the weights already live. NVMe is immutable model
backing and cold-start storage. It is never a place for runtime writes and must
not become an unbounded per-token dependency.

## Why Strata is different

- **Expert-centric execution.** Rows wait briefly for compatible expert work
  instead of forcing a load-compute-evict cycle at every layer.
- **Resident Execution Contracts.** A request is admitted with an explicit
  local, peer-backed, or bounded-cold-read plan.
- **Architecture-native kernels.** GLM-5.2's DSA/IndexShare and MTP, and
  DeepSeek-V4's hybrid attention, mHC, routing, native FP4 experts, and DSpark
  are model graph operations—not opaque framework calls.
- **Expert Compute Fabric.** LAN peers execute beside their resident weights;
  only compact activations and partial results cross the network.
- **Advisory prediction.** Prediction may prefetch or place exact weights. It
  cannot choose experts, change coefficients, or alter the model output.
- **No framework runtime.** The engine is C++20 with stable C kernel ABIs and an
  optional CUDA backend.

## The two starting models

### GLM-5.2 — the primary target

[`zai-org/GLM-5.2-FP8`](https://huggingface.co/zai-org/GLM-5.2-FP8) is the
model Strata is built to beat on equal hardware and memory budgets. It is the
right first target because it combines every tier we need to engineer: a
resident dense spine, 256 routed experts, a shared expert, top-8 routing,
DSA/IndexShare sparse attention, MTP, VRAM placement, a large RAM working set,
and read-only NVMe capacity.

At the pinned source revision documented in
[`docs/model-bringup.md`](docs/model-bringup.md), the official repository has
141 Safetensors weight shards and 755,617,140,416 bytes of tensor payload. Strata
will consume the official FP8 source incrementally and produce its own immutable
mixed-precision pack. Routed expert matrices are the first Q4_K64 candidates;
routers, the dense spine, DSA indexers, and MTP remain at their source or higher
precision until measurements justify changing them.

We have a known external throughput baseline for this model. Every comparison
must use the same checkpoint semantics, prompt or replay, decode length,
RAM/VRAM budgets, warm state, GPU topology, and quality mode. The primary metric
is median decode tok/s; NVMe bytes per emitted token is a co-primary constraint.

### DeepSeek-V4-Flash-DSpark — the resident proof of concept

[`deepseek-ai/DeepSeek-V4-Flash-DSpark`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark)
is a 284B-total/13B-active MoE checkpoint whose official repository occupies
166,886,535,336 bytes. It has 256 routed experts, top-6 activation, one shared
expert, hybrid compressed attention, manifold-constrained hyper-connections,
and a three-stage DSpark speculative module.

No additional quantization is planned. The checkpoint already uses native FP4
E2M1 expert weights with per-32-weight scales, FP8 supporting matrices, and
BF16/FP32 sensitive tensors. Four bits is the allowed floor. Strata will build a
small verified sidecar manifest over the 48 original Safetensors shards and use
their extents read-only instead of rewriting another 167 GB copy.

DSpark is particularly valuable to Strata: it proposes a five-token block from
hidden states captured at target layers 40–42, a Markov head, and a confidence
head. Those provisional rows create expert batches naturally. We will measure
accepted tokens per target forward and the growth of the routed-expert union,
not assume that speculation is automatically faster.

## Start here

The next branch is:

```text
feat/glm52-stream-pack
```

It starts with GLM-5.2—not a smaller substitute—and delivers one bounded vertical
slice:

1. Pin and parse the official `config.json` and Safetensors index.
2. Validate all 118,629 tensor mappings, 141 shards, architecture fields, and
   source byte totals without downloading weight payloads.
3. Read a real GLM-5.2 FP8 expert tensor and its 128×128 block scales.
4. Convert that tensor to Q4_K64 with the C++ reference codec.
5. Write, hash, reopen, dequantize, and numerically verify the Strata extent.
6. Generalize the same path into a resumable one-shard-at-a-time conversion.

Only when step 5 passes do we begin the full weight transfer. The complete
source is downloaded over time, but it never has to coexist on disk: a source
shard may be released only after every derived extent is durable and
hash-verified. Expect roughly 400 GB for the resulting pack plus one source
shard, conversion workspace, and a safety margin—not 761 GB of source plus a
second complete model.

After the GLM pack and graph are running, the DeepSeek-V4-Flash-DSpark adapter is
the second vertical target. It exercises native FP4/FP8 loading and fully
resident execution without adding a quantization experiment.

The detailed implementation gates and pinned checkpoint facts are in
[`docs/model-bringup.md`](docs/model-bringup.md). The gated research sequence is
in [`docs/research-roadmap.md`](docs/research-roadmap.md).

## Precision contract

No Strata model representation may use less than four bits per weight. Four bits
is a floor, not an instruction to force every tensor to the minimum.

- **GLM-5.2:** Q4_K64 begins with routed expert matrices. The dense spine,
  routers, DSA indexers, and MTP stay FP8/BF16/FP32 until role-by-role quality and
  performance evidence supports another representation.
- **DeepSeek-V4-Flash-DSpark:** preserve the official FP4 E2M1 expert layout and
  FP8/BF16/FP32 supporting tensors exactly. Do not requantize it.
- **Drafts and predictors:** never below four bits. Placement predictors do not
  enter the numerical graph.

No conversion is accepted merely because it creates a smaller file. GLM-5.2
requires tensor reconstruction checks, layerwise router checks, fixed-logit
comparisons, teacher forcing, greedy generation, and measured MTP acceptance.

## Current status

Strata is an early performance-research engine, not yet a language-model
inference binary. The repository currently contains:

- model and DeepSeek semantic validation;
- a fixed model-pack header with a four-bit precision floor;
- a signed-int4 CPU matvec correctness oracle;
- a sequential route-trace parser and past-only transition predictor;
- a VRAM/RAM/peer/NVMe residency simulator;
- LRU, LFU, and lease-aware policy experiments with explicit byte accounting;
- dependency-free tests.

The current CPU matvec is an arithmetic oracle. It is not yet the production
Q4_K64 storage layout or an optimized kernel.

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
4. The scheduler groups compatible tickets across ready requests and verified
   speculative blocks until a bounded latency deadline.
5. Work executes where the canonical expert already lives: VRAM, local RAM, or
   a peer worker. Only a declared bounded-cold contract may reach NVMe.
6. Weighted results return to the original row, which advances through the
   exact model graph.

Read [`docs/architecture.md`](docs/architecture.md) for the full design.

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
- Model weights are immutable and read-only during inference.
- Every performance claim needs reproducible A/B measurements and correctness
  gates.
- Complexity earns its place only through evidence.

## License

Apache-2.0.
