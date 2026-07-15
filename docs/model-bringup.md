# GLM-5.2 and DeepSeek-V4 bring-up plan

Strata starts with frontier-scale MoE models. GLM-5.2 is the primary target from
the first branch; DeepSeek-V4-Flash-DSpark is the first fully resident proof of
concept. There is no smaller pretrained model in the critical path.

## Pinned source facts

The first implementation must pin revisions rather than follow moving model
repository heads.

| Property | GLM-5.2 | DeepSeek-V4-Flash-DSpark |
|---|---:|---:|
| Repository | `QuantTrio/GLM-5.2-Int4-Int8Mix` | `deepseek-ai/DeepSeek-V4-Flash-DSpark` |
| Revision inspected | `1d3bcfe5ec549ecd000fd80b37f191183842e983` | `62af8fffb2f7030cac4de2f0169f5b8d1101b646` |
| Index SHA-256 | `43298345833417b1ad2a8b76d012a83d4f2275d532e5ab38e118566f1ac7b12b` | `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b` |
| Indexed tensors | 177,569 | 72,317 |
| Weight shards | 128 (124 main + 4 MTP) | 48 |
| Indexed tensor payload | 405,459,090,304 bytes | 166,878,536,440 bytes |
| Weight shard files | 405,481,014,016 bytes | 166,886,535,336 bytes repository storage |
| License reported by repository | MIT | MIT |
| Source precision | INT4/INT8 packed weights plus BF16/F32 | FP4 experts, FP8 spine, BF16/F32 sensitive tensors |
| Strata handling | Preserve native extents; no additional quantization | Preserve native extents; no additional quantization |

These values were read from the pinned target repositories on 2026-07-14. The
downloader must verify the pinned revision, per-file size/hash, index totals, and
license files again before transferring weights.

## Target A — GLM-5.2

GLM-5.2 defines Strata's primary architecture and performance contract:

- 78 main layers plus one MTP layer;
- three dense-prefix layers followed by sparse MoE layers;
- 256 routed experts, top-8 selection, and one shared expert;
- sigmoid scores, `noaux_tc` selection, normalized top-k probabilities, and a
  routed scaling factor of 2.5;
- MLA-style low-rank query/KV projections;
- DSA top-2048 sparse attention with IndexShare reuse every four layers;
- a one-million-token maximum context;
- symmetric INT4 group-128 routed experts, symmetric INT8 group-128 ordinary
  linears, channelwise INT8 MTP, and BF16/FP32 sensitive tensors.

The source uses the `compressed-tensors` `pack-quantized` convention. Quantized
modules are represented by an I32 `weight_packed` tensor, a BF16 `weight_scale`
tensor, and an I64 two-element `weight_shape` tensor. Logical precision and
granularity come from the pinned `quantization_config`; physical I32 storage is
not itself a precision declaration.

| Layer scope | Declared weight representation |
|---|---|
| Layer 0 | BF16 |
| Layers 1–2 | ordinary linears: symmetric INT8 group-128 |
| Layers 3–77 | routed experts: symmetric INT4 group-128; ordinary linears: symmetric INT8 group-128 |
| Layer 78 MTP | channelwise symmetric INT8 |
| Routers, indexers, norms, embeddings, special heads | pinned BF16 or FP32 source dtype |

### Stage A0 — metadata-only target lock

Create `feat/glm52-compressed-import`. Fetch only the pinned configuration,
generation configuration, tokenizer assets, license, Safetensors index, and
file metadata. Do not begin the bulk transfer yet.

Implement a C++ inspector that:

1. parses the 177,569-entry tensor index with bounded memory;
2. recognizes all 124 main shards, four MTP shards, and the
   405,459,090,304-byte indexed payload total;
3. assigns every tensor an explicit role—dense spine, router, shared expert,
   routed expert, DSA/IndexShare, MTP, embedding, head, norm, or metadata;
4. joins every quantized module's packed values, scales, and logical shape and
   derives its exact precision, granularity, placement group, and source extent;
5. rejects unknown, duplicate, missing, overlapping, or semantically invalid
   tensors;
6. emits a deterministic zero-rewrite sidecar plan without touching weight
   payloads.

Gate: the manifest covers every indexed tensor and byte. Tensor-name guessing is
allowed in the offline architecture importer only; runtime code consumes stable
roles and IDs.

### Stage A1 — one real GLM quantized module

Select one routed expert module's `weight_packed`, `weight_scale`, and
`weight_shape` tensors from the pinned index. Range-read only those extents from
the containing source shard after verifying the shard size and SHA-256.

Implement the reference path:

1. bounded Safetensors range reader and header validation;
2. little-endian I32 unpack into signed INT4 values in the exact
   `compressed-tensors` order;
3. BF16 scale decode with one symmetric scale per 128 logical input elements;
4. logical-shape and byte-count reconciliation for gate, up, and down matrices;
5. deterministic W4A16 reference matvec with FP32 accumulation;
6. reconstructed-weight and output-vector comparison against a frozen oracle
   produced by the pinned checkpoint and compatible `compressed-tensors`
   implementation.

Gate: peak RSS is bounded by the tile budget, output is deterministic, native
extents survive hash verification, and reconstruction/output metrics are
recorded. This uses a real target-format GLM-5.2 tensor, not a substitute.

### Stage A2 — zero-rewrite native import

Generalize A1 across the full checkpoint. The sidecar plan is fixed before
weight transfer begins.

- Reference each official Safetensors extent by pinned shard hash, offset,
  length, dtype, shape, role, and placement group.
- Validate that INT4 group-128 tensors pack eight logical weights per I32 and
  have one BF16 scale per output-row/group pair.
- Validate that INT8 group-128 tensors pack four logical weights per I32 and
  have one BF16 scale per output-row/group pair.
- Validate that MTP INT8 tensors pack four logical weights per I32 and have one
  BF16 scale per output channel.
- Preserve BF16 embeddings, norms, indexers, special heads, and FP32 router
  tensors without conversion.
- Open model shards read-only at runtime; placement state and route history
  never modify them.
- Record full source hashes and manifest completion, but do not write duplicate
  weight payloads.

The base disk budget is 405,481,014,016 bytes for the source shard files plus
metadata, resumable download state, the small sidecar, and a safety margin. The
exact planner output—not the model card's rounded 378 GiB figure—authorizes the
download. A second converted pack is outside the plan.

Gate: a final scan verifies all expected tensor roles, native quantization
triplets, extents, and hashes; runtime rejects a missing, changed, overlapping,
or writable source shard.

### Stage A3 — GLM graph from actual weights

Implement the graph in dependency order using tensors from the verified native
manifest:

1. embedding, RMSNorm, RoPE, and dense linear primitives;
2. low-rank attention projections and compressed KV state;
3. DSA indexer and IndexShare ownership;
4. exact sigmoid/`noaux_tc` router selection and combination weights;
5. shared expert and routed expert SwiGLU;
6. residual path and layer advancement;
7. MTP head and verification;
8. tokenizer and generation boundary.

Correctness progresses through actual target slices: individual target tensors,
one dense layer, the first MoE layer, DSA selected positions, a multi-layer
replay, and finally the complete model. Shape-reduced generated fixtures may
exercise error paths, but they are not a model milestone.

Gate: layerwise logits, router IDs/coefficients, attention selections, teacher
forcing, greedy generation, and MTP acceptance match frozen reference artifacts
within declared numerical contracts. No operation may silently fall back or be
skipped.

### Stage A4 — memory hierarchy and performance

Bring up execution tiers in this order:

1. memory-mapped immutable source shards and CPU reference execution;
2. RAM-resident expert arenas with NUMA placement;
3. VRAM-resident dense spine and explicit expert cache;
4. persistent expert-ticket wavefront across requests;
5. overlapped demand H2D and strictly budgeted prefetch;
6. bounded-cold admission contracts;
7. optional peer execution beside remote resident experts.

The first performance comparison reproduces the pinned model card's vLLM 0.23.0
`compressed-tensors` execution contract on the same hardware before comparing
Strata. Record every run and report median prefill time, decode tok/s, physical
NVMe bytes per emitted token, H2D bytes, RSS, per-device VRAM, expert rows,
cache events, and MTP acceptance.

Gate: correctness passes, median tok/s exceeds the baseline outside run
variance, and no hidden quality, routing, precision, memory, or I/O difference
explains the result.

## Target B — DeepSeek-V4-Flash-DSpark

This target is not a generic older-DeepSeek stand-in. It introduces architecture
and storage formats that Strata must support natively:

- 43 main layers and three DSpark stages;
- 256 routed experts, top-6 activation, and one shared expert;
- `sqrtsoftplus` routing, `noaux_tc` selection, routed scale 1.5, and clipped
  SwiGLU behavior;
- hybrid compressed sparse attention and heavily compressed attention with
  per-layer compression ratios;
- manifold-constrained hyper-connections using Sinkhorn normalization;
- one-million-token context with YaRN scaling;
- native FP4 E2M1 expert values with UE8M0 per-32-weight scales;
- FP8 E4M3 supporting matrices with 128×128 scales;
- a five-token DSpark block driven from hidden states at layers 40, 41, and 42,
  plus rank-256 Markov and confidence heads.

### Stage B0 — zero-rewrite native import

Status: implemented for the pinned local checkpoint. Header/layout validation,
native FP4/FP8 CUDA encodings, real target-byte fixtures, and resident-topology
admission are automated.

Do not convert or repack 167 GB of already-valid weights. Build a small,
content-addressed sidecar manifest that references extents in the 48 official
Safetensors shards. The original files remain immutable and are opened read-only.
The pinned index contains 72,317 tensors totaling 166,878,536,440 bytes.

Implement native reference codecs for:

- FP4 E2M1 packed nibbles with per-32-element UE8M0 scales;
- FP8 E4M3 matrices with their declared block scales;
- BF16, FP32, I64, and scale-only tensor roles.

Gate: every byte and tensor is covered, native extents reconstruct correctly,
and startup can stage the entire checkpoint into the declared RAM+VRAM budget
without runtime model writes.

### Stage B1 — exact DeepSeek-V4 graph

Status: base-model executor implemented with an exact 2,048-token admission
ceiling; frozen layer, teacher-forcing, and full-generation oracle promotion is
still pending. See `docs/deepseek-v4-runtime.md`.

Implement and verify hybrid attention compression, sparse indexing, mHC,
sqrtsoftplus routing, shared/routed experts, and tokenizer/response encoding.
Base-model decode must pass before speculative decoding is enabled.

Gate: full-model teacher forcing and greedy/sampled generation match frozen
reference artifacts; routing and mHC invariants pass layerwise; steady-state
decode performs zero physical NVMe reads after resident admission.

### Stage B2 — DSpark as a scheduler primitive

Status: manifest semantics and all tensors are validated; execution remains
disabled pending exact verification and provisional-state rollback fixtures.

DSpark is a confidence-scheduled semi-autoregressive draft, not merely an MTP
toggle. Strata should expose its five provisional rows directly to the
expert-centric scheduler:

1. capture target hidden states from layers 40–42;
2. execute the three attached DSpark stages;
3. combine block logits with the Markov head;
4. use the confidence head to schedule a verification depth;
5. form an expert union across provisional rows;
6. batch resident expert work without allowing drafts to alter exact target
   verification;
7. commit only accepted tokens and roll back provisional KV state exactly.

Measure accepted tokens per target forward, draft time, verification time,
expert-union size, expert rows, cache pressure, and physical I/O. If speculation
raises expert work more than it saves target forwards, disable or shorten it;
acceptance rate alone is not a win.

Gate: output distribution/greedy correctness satisfies the declared speculative
contract and median end-to-end throughput improves outside variance.

## Other models

No third checkpoint is on the critical path. Reusable dense, MoE, attention,
quantization, and scheduler operations should remain architecture-neutral, but
new model adapters wait until GLM-5.2 and DeepSeek-V4-Flash-DSpark are working.
The project is judged by these two targets, not by the length of a compatibility
list.
