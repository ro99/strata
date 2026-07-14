# GLM-5.2 and DeepSeek-V4 bring-up plan

Strata starts with frontier-scale MoE models. GLM-5.2 is the primary target from
the first branch; DeepSeek-V4-Flash-DSpark is the first fully resident proof of
concept. There is no smaller pretrained model in the critical path.

## Pinned source facts

The first implementation must pin revisions rather than follow moving model
repository heads.

| Property | GLM-5.2 | DeepSeek-V4-Flash-DSpark |
|---|---:|---:|
| Repository | `zai-org/GLM-5.2-FP8` | `deepseek-ai/DeepSeek-V4-Flash-DSpark` |
| Revision inspected | `ba978f7d347eaf65d22f1a86833408afdb953541` | `62af8fffb2f7030cac4de2f0169f5b8d1101b646` |
| Weight shards | 141 | 48 |
| Indexed tensor payload | 755,617,140,416 bytes | 166,878,536,440 bytes |
| Repository storage | 761,025,363,709 bytes | 166,886,535,336 bytes |
| License reported by repository | MIT | MIT |
| Source precision | FP8 E4M3 blocks plus BF16/F32 | FP4 experts, FP8 spine, BF16/F32 sensitive tensors |
| Strata handling | Stream-convert routed experts to Q4_K64 | Preserve native extents; no additional quantization |

These values were read from the official repositories on 2026-07-14. The
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
- an official FP8 source layout using 128×128 block scales.

### Stage A0 — metadata-only target lock

Create `feat/glm52-stream-pack`. Fetch only the pinned configuration, generation
configuration, tokenizer assets, license, Safetensors index, and file metadata.
Do not begin the bulk transfer yet.

Implement a C++ inspector that:

1. parses the 118,629-entry tensor index with bounded memory;
2. recognizes all 141 expected shards and the 755,617,140,416-byte total;
3. assigns every tensor an explicit role—dense spine, router, shared expert,
   routed expert, DSA/IndexShare, MTP, embedding, head, norm, or metadata;
4. derives shape, source precision, target precision, placement group, and
   expected target extent;
5. rejects unknown, duplicate, missing, overlapping, or semantically invalid
   tensors;
6. emits a deterministic conversion plan without touching weight payloads.

Gate: the manifest covers every indexed tensor and byte. Tensor-name guessing is
allowed in the offline architecture importer only; runtime code consumes stable
roles and IDs.

### Stage A1 — one real GLM tensor

Select one routed expert matrix and its FP8 inverse-scale tensor from the pinned
index. Download only the containing source shard, with HTTP resume and source
hash verification.

Implement the reference path:

1. bounded Safetensors range reader;
2. FP8 E4M3 plus 128×128 block-scale decode to FP32 tiles;
3. deterministic Q4_K64 encoding, one output-row group at a time;
4. Strata extent header, payload, scales, and content hash;
5. reopen and dequantize the written extent;
6. absolute, relative, and output-vector error comparison against the FP8
   source tensor.

Gate: peak RSS is bounded by the tile budget, output is deterministic, the
extent survives reopen/hash verification, and reconstruction metrics are
recorded. This uses a real GLM-5.2 tensor, not a substitute checkpoint.

### Stage A2 — resumable shard-streaming conversion

Generalize A1 across the full checkpoint. The conversion plan is fixed before
weight transfer begins.

- Process one source shard at a time and each source extent sequentially.
- Write each target extent exactly once whenever possible.
- Preserve routers, normalization, embeddings, output head, DSA indexers, dense
  spine, and MTP at FP8/BF16/F32 initially.
- Convert routed expert matrices to Q4_K64 first.
- Do not quantize MTP to int4 until draft acceptance has an FP8 baseline.
- Record source hash, target hash, quantization metrics, bytes read/written, and
  completion state per extent.
- `fsync` data and journal state before marking an extent complete.
- On restart, reread metadata but never rewrite a verified target extent.
- Releasing a verified source shard is an explicit operator option, never an
  implicit cleanup side effect.

The operational disk budget is approximately 400 GB for the target pack plus
the largest source shard, conversion workspace, journal, and safety margin. The
exact planner output—not this estimate—authorizes conversion.

Gate: a forced interruption at every state boundary resumes correctly; a final
scan verifies all expected tensor roles and hashes; no partial pack is published
under its final name.

### Stage A3 — GLM graph from actual weights

Implement the graph in dependency order using tensors from the real pack:

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

1. memory-mapped immutable pack and CPU reference execution;
2. RAM-resident expert arenas with NUMA placement;
3. VRAM-resident dense spine and explicit expert cache;
4. persistent expert-ticket wavefront across requests;
5. overlapped demand H2D and strictly budgeted prefetch;
6. bounded-cold admission contracts;
7. optional peer execution beside remote resident experts.

The first performance comparison uses the known GLM-5.2 baseline contract on
the same hardware. Record every run and report median prefill time, decode
tok/s, physical NVMe bytes per emitted token, H2D bytes, RSS, per-device VRAM,
expert rows, cache events, and MTP acceptance.

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

Implement and verify hybrid attention compression, sparse indexing, mHC,
sqrtsoftplus routing, shared/routed experts, and tokenizer/response encoding.
Base-model decode must pass before speculative decoding is enabled.

Gate: full-model teacher forcing and greedy/sampled generation match frozen
reference artifacts; routing and mHC invariants pass layerwise; steady-state
decode performs zero physical NVMe reads after resident admission.

### Stage B2 — DSpark as a scheduler primitive

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
