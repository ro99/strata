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

[`QuantTrio/GLM-5.2-Int4-Int8Mix`](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix)
is the exact model Strata targets. It combines every tier we need to engineer: a
resident dense spine, 256 routed experts, a shared expert, top-8 routing,
DSA/IndexShare sparse attention, MTP, VRAM placement, a large RAM working set,
and read-only NVMe capacity.

At the pinned source revision documented in
[`docs/model-bringup.md`](docs/model-bringup.md), the repository has 177,569
indexed tensors in 128 Safetensors shards and 405,459,090,304 bytes of tensor
payload. The checkpoint already has the required mixed precision: layer 0 is
BF16; routed expert matrices in layers 3–77 are symmetric W4A16 group-128;
quantized ordinary linears in layers 1–77 are symmetric W8A16 group-128; the
MTP layer is channelwise W8A16; and sensitive tensors remain BF16/FP32. Strata
preserves those native extents behind a verified sidecar manifest instead of
requantizing or creating a second 405 GB copy.

The first external baseline is the model card's verified vLLM 0.23.0
`compressed-tensors` configuration. Every performance comparison must use this
exact revision, prompt or replay, decode length, RAM/VRAM budgets, warm state,
GPU topology, MTP configuration, and reasoning effort. The primary metric is
median decode tok/s; NVMe bytes per emitted token is a co-primary constraint.

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
feat/glm52-compressed-import
```

It starts with GLM-5.2—not a smaller substitute—and delivers one bounded vertical
slice:

1. Pin and parse the target `config.json` and Safetensors index.
2. Validate all 177,569 tensor mappings, 128 shards, architecture fields, and
   source byte totals without downloading weight payloads.
3. Range-read one real expert's `weight_packed`, `weight_scale`, and
   `weight_shape` extents.
4. Decode the native signed INT4 group-128 representation in C++.
5. Verify reconstruction and W4A16 output against a frozen target-format oracle.
6. Generalize the same validation into a content-addressed sidecar over all
   original shards.

Only when step 5 passes do we begin the full weight transfer. The final disk
budget is the 405,481,014,016 bytes of source shard files plus the small sidecar,
download state, and safety margin. There is no converted second model copy.

After the GLM native manifest and graph are running, the
DeepSeek-V4-Flash-DSpark adapter is the second vertical target. It exercises
native FP4/FP8 loading and fully resident execution without adding a
quantization experiment.

The detailed implementation gates and pinned checkpoint facts are in
[`docs/model-bringup.md`](docs/model-bringup.md). The gated research sequence is
in [`docs/research-roadmap.md`](docs/research-roadmap.md).

## Precision contract

No Strata model representation may use less than four bits per weight. Four bits
is a floor, not an instruction to force every tensor to the minimum.

- **GLM-5.2:** preserve the checkpoint's INT4 group-128 experts, INT8 group-128
  linears, INT8 channelwise MTP, and BF16/FP32 sensitive tensors exactly. No
  silent conversion to a different four-bit layout is permitted.
- **DeepSeek-V4-Flash-DSpark:** preserve the official FP4 E2M1 expert layout and
  FP8/BF16/FP32 supporting tensors exactly. Do not requantize it.
- **Drafts and predictors:** never below four bits. Placement predictors do not
  enter the numerical graph.

Native import still requires tensor reconstruction checks, layerwise router
checks, fixed-logit comparisons, teacher forcing, greedy generation, and
measured MTP acceptance.

## Current status

Strata now has a first bounded GLM-5.2 inference path intended to establish the
unoptimized machine baseline. The repository currently contains:

- model, GLM-5.2 target-contract, and DeepSeek semantic validation;
- bounded JSON, Safetensors, and zero-rewrite checkpoint readers;
- native offset-packed INT4/INT8 groupwise CUDA matvecs for compute 8.6 and
  12.0, plus FP32 correctness oracles;
- exact GLM RMSNorm, interleaved RoPE, softmax, sigmoid/`noaux_tc` routing,
  dense/SwiGLU, shared-expert, and routed-expert operations;
- a 78-layer main-model graph with causal MLA state and multi-GPU expert
  dispatch;
- the pinned byte-level BPE tokenizer and chat rendering;
- `strata-run`, which reports prompt-processing and decode throughput
  separately, together with checkpoint, H2D/D2H, and VRAM-cache counters;
- a fixed model-pack header with a four-bit precision floor;
- a sequential route-trace parser and past-only transition predictor;
- a VRAM/RAM/peer/NVMe residency simulator;
- LRU, LFU, and lease-aware policy experiments with explicit byte accounting;
- dependency-free tests.

The first runtime is deliberately bounded to at most 2,048 tokens, where GLM's
DSA selection covers the entire causal history and therefore has exactly the
same result as dense causal attention. MTP proposal acceleration is explicitly
disabled for the first base-model benchmark; it does not change the greedy main
model result. The absorbed MLA decode kernel, active DSA beyond 2,048 tokens,
MTP verification, and optimized resident scheduling remain subsequent measured
hypotheses.

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

## Run the GLM-5.2 baseline

After all 124 main and four MTP shards have finished downloading into
`models/glm52`, run one exact greedy generation directly from the original
Safetensors extents:

```bash
./build/strata-run \
  --model models/glm52 \
  --devices 0,1,2 \
  --prompt 'What is the closer start to sun, and how distant it is from it?' \
  --max-new 128 \
  --json
```

The reproducible benchmark script validates the checkpoint and prompt, records
system/GPU telemetry, and performs three repetitions by default. Because this
is a long load and generation, launch it in a named tmux session:

```bash
tmux new-session -d -s strata-glm52-baseline \
  'cd /home/rodrigo/Developer/strata && scripts/run_glm52_baseline.sh'
tmux attach -t strata-glm52-baseline
```

Ignored outputs are written under `results/glm52-baseline/`. Each run records
`prompt_processing_tok_s`, `generation_tok_s`, requested checkpoint bytes,
block-device physical reads/writes, weight and activation transfers, CUDA
calls, and per-device cache occupancy.

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
