# GLM-5.3-Flash: text runbook

GLM-5.3-Flash is a 45-layer, 288-routed-expert model with four mHC streams,
three Kimi Delta Attention layers for every sparse MLA layer, eight selected
experts plus one shared expert, and block-128 FP8 E4M3 weights with F32 inverse
scales. Strata consumes the checkpoint as published and streams modules from
storage; it does not requantize the model or reduce its expert count or top-k.

This adapter currently supports text only. Image or video content is rejected
before generation. The admitted context is capped at 2,048 tokens. At that
length the model's sparse index `top_k` is 2,048, so every causally visible key
is selected and dense causal MLA is exactly equivalent to running the sparse
indexer. Longer contexts fail admission until the k-pool indexer is implemented;
there is no silent dense or truncated fallback.

## Build and preflight

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel --target strata-chat strata-server

./build-release/strata-chat \
  --model models/glm53f --model-type glm53 \
  --devices 0,1,2 --context-size 2048 --max-new 256 \
  --vram-fraction 0.85 --dry-run
```

The pinned release has 62 safetensors shards, 76,108 indexed tensors, and
328,326,771,576 indexed payload bytes. Admission validates those extents, the
hybrid layer schedule, text tensor roles, representative shapes and dtypes,
and the FP8 block geometry before generation.

## Chat and server

```bash
./build-release/strata-chat \
  --model models/glm53f --model-type glm53 \
  --devices 0,1,2 --context-size 2048 --max-new 256

./build-release/strata-server \
  --model models/glm53f --model-type glm53 --model-id glm53f \
  --devices 0,1,2 --context-size 2048 --max-new 256 --port 8080
```

The server exposes the same text runtime through its OpenAI-compatible chat
completion endpoint. Do not send image content: multimodal support is outside
this adapter's current contract.

## Current operating point

The runtime discovers CPU width, free VRAM, peer topology and storage at
startup. On hosts where the routed checkpoint is much larger than the usable
CUDA cache, it maps canonical FP8 experts once and executes them directly from
host memory while keeping the non-expert spine, fused KDA state and mHC
transitions on CUDA. Prompt pages group rows by expert so an expert is traversed
once for every row that selected it; decode does not reread routed experts from
the checkpoint through Strata's explicit I/O path per token (the OS still owns
page-cache residency for mapped payloads). Systems with a high-speed two-GPU
peer fabric additionally admit the full TP2 route; PCIe systems use a
contiguous pipeline schedule.

MTP drafting and verification are implemented but opt in because acceptance is
workload-dependent. Set `STRATA_GLM53_MTP=1` for an acceptance campaign. The
exact production latency path leaves it disabled. Resident absorbed MLA remains
an explicit experimental route (`STRATA_GLM53_RESIDENT_MLA=1`) until its
BF16-boundary equivalence gate is closed; it is never selected silently.
