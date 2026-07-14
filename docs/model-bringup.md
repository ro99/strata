# Model bring-up plan

This document answers two questions: what should be built next, and when should
a large checkpoint be downloaded? The short answer is **format first, smallest
useful architecture second, frontier-scale models last**.

## Decision

Do not begin with GLM-5.2 or a DeepSeek-V4-class model. Do not write a one-off
converter for either model. Build one streaming model-pack pipeline and prove it
on generated tensors and successively richer reference architectures.

A giant checkpoint makes a poor development fixture:

- source download, conversion, and load cycles are too slow;
- source and converted weights may temporarily require hundreds of gigabytes;
- a wrong logit does not identify whether packing, quantization, graph mapping,
  routing, kernels, or scheduling failed;
- conversion experiments create large disposable files and unnecessary storage
  writes;
- full-scale debugging encourages accidental dependence on model-specific
  layouts.

Large checkpoints become valuable only after the same path has already produced
correct output on smaller models.

## Stage 0 — model pack and Q4_K64

Create `feat/modelpack-q4k64`. Do not download a model for this stage.

Implement in this order:

1. A bounded-memory Safetensors header and tensor-directory reader.
2. A Strata manifest carrying architecture, tensor roles, exact shapes,
   quantization layout, tokenizer identity, source identity, and content hashes.
3. The `Q4_K64` reference codec: groups of 64 weights along the input dimension,
   signed two's-complement values in `[-8, 7]` and one FP16 scale per group.
4. `strata-pack inspect`, `convert`, and `verify` commands.
5. Temporary output, per-shard checkpoints, `fsync`, hash verification, and
   atomic final rename. A restart must reuse already verified extents.
6. Tiny deterministic fixtures covering odd shapes, partial groups, saturation,
   zero groups, corrupt headers, corrupt payloads, and interrupted conversion.

The initial quantizer should use deterministic round-to-nearest. More advanced
calibration may be added only after the simple reference path is measurable;
container correctness and quantizer research are separate hypotheses.

### Stage 0 gate

- byte-exact pack/unpack layout tests;
- deterministic output across runs;
- bounded reconstruction error for each fixture;
- rejected precision values below four bits;
- source tensor and output extent hashes verified;
- peak RSS independent of total checkpoint size;
- interrupted conversion resumes without rewriting verified extents;
- no published pack appears before complete verification.

The existing per-row int4 matvec remains a numerical oracle. It is not the
Q4_K64 storage specification and should not silently define it.

## Stage 1 — small dense reference

Use `TinyLlama/TinyLlama-1.1B-Chat-v1.0` as the first real download. Its
Llama-compatible graph is small enough for rapid CPU debugging while exercising
real tokenizer, RoPE, GQA, SwiGLU, RMSNorm, embedding, and output-head behavior.

Implement the dense CPU graph before an optimized CUDA path. Preserve sensitive
tensors in BF16/FP16 initially. Quantize one tensor role at a time and compare
against a source-precision oracle so quality loss has an owner.

### Stage 1 gate

- manifest and tensor mapping are complete—no name-based runtime guessing;
- selected tensor dequantization matches the reference codec;
- fixed-prompt source-precision logits are recorded immutably;
- teacher forcing and greedy generation pass declared tolerances;
- Q4_K64 quality is evaluated against the same source checkpoint;
- peak RSS and model-pack bytes are recorded.

Tokenizer work is part of the architecture adapter. For bring-up, a small
offline tokenizer utility or pre-tokenized fixtures are acceptable; the final
runtime must not require a framework or Python process.

## Stage 2 — conventional MoE

Use `allenai/OLMoE-1B-7B-0125`. It is large enough to expose real routing and
expert reuse but small enough to iterate on a workstation.

This stage adds exact top-k routing, expert gate/up/down placement groups,
expert-major ticket batching, route trace capture, and comparison of
row-synchronous versus wavefront execution. First run everything through the CPU
reference graph; storage policy results are invalid until routing is correct.

### Stage 2 gate

- exact router IDs, weights, normalization, and aggregation match the oracle;
- dense-only and MoE-layer logits pass declared tolerances;
- sequential traces contain no future-derived information;
- the wavefront performs identical mathematical work;
- cache and NVMe counters reconcile at byte granularity.

## Stage 3 — native DeepSeek

Use `deepseek-ai/DeepSeek-V2-Lite`. This is the first model that jointly proves
MLA and DeepSeekMoE behavior without frontier-scale iteration costs.

The adapter must explicitly validate:

- MLA compressed KV and decoupled RoPE paths;
- dense-prefix and MoE-layer boundaries;
- fine-grained expert dimensions and top-k;
- always-resident shared experts;
- group-limited routing where configured;
- selection bias/correction semantics separately from combination weights;
- routed scaling and normalization.

Shared experts belong to the resident spine and should overlap routed work.
Group selection may inform conservative prefetch, but only the final exact router
can choose executed experts.

### Stage 3 gate

- architecture oracle passes teacher forcing and greedy generation;
- exact selected experts and coefficients are validated layer by layer;
- shared experts never enter the demand-paged cache;
- MLA KV memory matches the manifest-derived bound;
- generated route traces drive a reproducible residency-policy comparison.

## Stage 4 — optimized execution

Only now add persistent CPU kernels, CUDA residency, H2D overlap, multi-request
batching, admission contracts, and peer execution. Optimization starts from
fixed replays generated by Stages 2 and 3, so every candidate performs equal
work.

A storage policy is useful only when it reduces physical bytes read at equal
capacity without causing larger H2D traffic, destructive prefetch, or hidden
latency. Page-cache hits are not zero I/O unless the measurement proves that the
bytes were resident.

## Stage 5 — frontier-scale acceptance

GLM-5.2 and the selected DeepSeek-V4-class checkpoint enter here. They validate
scale; they do not define the basic engine.

Before downloading either one, record:

- exact checkpoint revision and license;
- source precision and sharding layout;
- expected source, temporary, final-pack, and safety-margin bytes;
- storage filesystem and device endurance/health baseline;
- RAM, VRAM, NUMA, PCIe/NVLink, and peer topology;
- the already-passing smaller architecture oracle;
- conversion and load commands plus restart behavior.

Conversion is always shard-streaming. The whole source model must never be
resident in RAM. Source shards are read sequentially, output extents are written
once, and verified progress is reused after interruption. Deleting source shards
is a separate, explicit operator decision after the final pack is complete and
verified.

## How other models are added

Support is split into reusable operator families and exact architecture
manifests:

| Family | Reused machinery | Model-specific proof |
|---|---|---|
| Llama-like dense | RMSNorm, RoPE, GQA/MHA, SwiGLU | tensor mapping, RoPE variant, tokenizer, logits |
| Conventional MoE | exact router, top-k aggregation, expert tickets | scoring, normalization, capacity, expert layout |
| DeepSeek | MLA, shared/routed overlap, fine-grained experts | config-era routing semantics, dense prefix, tensor mapping |
| New family | existing kernels where mathematically identical | missing ops, manifest schema, tiny oracle, generation gate |

Recognizing a model name is not support. Strata adds an architecture only when
unsupported semantics fail loudly and the supported path has a reproducible
oracle.

## Precision and quality policy

Four bits is the absolute floor, not the universal target. The starting policy
is:

- routed expert matrices: evaluate Q4_K64 first;
- dense FFN and attention matrices: Q4_K64 only after per-role quality gates;
- routers, norms, embeddings, output heads, small tensors, and measured
  outliers: BF16/FP16 by default;
- activations and KV: declared per architecture and never silently changed;
- drafts, predictors, caches, and fallbacks: never below four bits.

Every mixed-precision choice is stored in the manifest. Runtime code executes
the declared representation or fails; it does not silently choose a cheaper
one.
