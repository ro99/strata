# GLM-5.3-Flash: text runbook

GLM-5.3-Flash is a 45-layer, 288-routed-expert model with four mHC streams,
three Kimi Delta Attention layers for every sparse MLA layer, eight selected
experts plus one shared expert. Strata supports both published representations:
block-128 FP8 E4M3 with F32 inverse scales, and the Quark MXFP4 release whose
routed experts use E2M1 group-32 weights while its mixed-precision corrections
and shared experts remain BF16. It consumes either checkpoint as published; it
does not requantize the model or reduce its expert count or top-k.

The release is resolved once from the checkpoint index and tensor contract.
There is no precision flag: point `--model` at `models/glm53f` or
`models/glm53f-mxfp4`, and the runtime selects the corresponding validators,
host expert decoder, activation boundary, and device shared-expert kernel. A
checkpoint matching neither pinned release is refused rather than guessed at.

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
  --model models/glm53f-mxfp4 --model-type glm53 \
  --devices 1,2 --context-size 2048 --max-new 256 \
  --vram-fraction 0.85 --dry-run
```

| release | shards | indexed tensors | indexed payload bytes |
|---|---:|---:|---:|
| FP8 E4M3 block-128 | 62 | 76,108 | 328,326,771,576 |
| MXFP4 group-32 | 120 | 72,466 | 227,486,055,288 |

Admission validates the selected release's exact extents, hybrid layer
schedule, text tensor roles, representative shapes and dtypes, and quantized
block geometry before generation.

## Chat and server

```bash
./build-release/strata-chat \
  --model models/glm53f-mxfp4 --model-type glm53 \
  --devices 1,2 --context-size 2048 --max-new 256

./build-release/strata-server \
  --model models/glm53f-mxfp4 --model-type glm53 --model-id glm53f-mxfp4 \
  --devices 1,2 --context-size 2048 --max-new 256 --port 8080
```

The server exposes the same text runtime through its OpenAI-compatible chat
completion endpoint. Do not send image content: multimodal support is outside
this adapter's current contract.

## Current operating point

The runtime discovers CPU width, free VRAM, peer topology and storage at
startup. On hosts where the routed checkpoint is much larger than the usable
CUDA cache, it maps checkpoint-native routed experts once and executes them
directly from host memory while keeping the non-expert spine, fused KDA/MLA
state and mHC transitions on CUDA. The one shared expert in each MoE layer is
resident on that layer's GPU and overlaps the eight routed host experts. FP8
uses its E4M3/F32-scale dot; MXFP4's BF16 shared weights use an exact BF16 dot.
Both return raw linear results so every BF16 rounding, clamp and SwiGLU remains
on the host. Set `STRATA_GLM53_SHARED_EXPERT_DEVICE=0` only to force the slower
host control.

The remaining admitted expert-arena capacity is filled once at startup with a
static routed-expert tier. Its compact built-in ranking comes from the
representative code, prose, multilingual, reasoning, and concurrent-server
workloads; any route that misses the tier continues through the unchanged host
path. The tier never replaces an expert during inference, so it has none of the
replacement latency that rejected the earlier dynamic-cache experiment. The
runtime discovers the available arena capacity and device placement rather
than assuming a particular GPU count or memory size. It stores FP8 experts in
their canonical E4M3 layout and MXFP4 experts in their checkpoint-native E2M1
group-32 layout, and preserves the host dot-product association exactly. Set
`STRATA_GLM53_STATIC_EXPERT=0` only for the same-binary diagnostic control.

Prompt pages group rows by expert so an expert is traversed once for every row
that selected it; decode does not reread routed experts from the checkpoint
through Strata's explicit I/O path per token (the OS still owns page-cache
residency for mapped payloads). Systems with a high-speed two-GPU peer fabric
additionally admit the full TP2 route; PCIe systems use a contiguous pipeline
schedule.

On the reference two-RTX-3090 host, the protected static-tier 3+3 alternating
A/B used a bounded 16-token decode. MXFP4 improved from 3.84 to 3.07 seconds
(-20.1%, 4.17 to 5.21 tok/s) while serving 35% of routed bytes from the tier;
FP8 improved from 5.76 to 4.66 seconds (-19.1%, 2.78 to 3.43 tok/s) at 29%
coverage. The control and candidate ranges did not overlap, every timed output
matched its same-checkpoint control byte for byte, and the timed path performed
no allocations. These are bounded short-context measurements, not projections
of long-context throughput, and the two checkpoint outputs must not be used to
infer relative quality.

MTP drafting and verification are implemented but opt in because acceptance is
workload-dependent. Set `STRATA_GLM53_MTP=1` for an acceptance campaign. The
exact production latency path leaves it disabled. Resident absorbed MLA is on
by default after its isolated exactness gate closed; all 45 layers now use the
same device mHC path. `STRATA_GLM53_RESIDENT_MLA=0` exists as a diagnostic
control, not as the recommended production route.
