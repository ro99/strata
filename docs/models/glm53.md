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
before generation. It implements the checkpoint's k-pool sparse indexer, so the
admitted context is 262,144 tokens. At or below the model's sparse index
`top_k` of 2,048 the selection is the identity -- every causally visible key is
chosen -- and the runtime keeps the dense causal MLA path there, bit for bit.
Above 2,048 the indexer chooses which history each query reads.

## Build and preflight

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel --target strata-chat strata-server

./build-release/strata-chat \
  --model models/glm53f-mxfp4 --model-type glm53 \
  --devices 1,2 --context-size 2048 --max-new 256 \
  --vram-fraction 0.85 --dry-run
```

`--context-size` above 2,048 selects the sparse indexer path; see "Context and
the k-pool sparse indexer" below for what that changes.

| release | shards | indexed tensors | indexed payload bytes |
|---|---:|---:|---:|
| FP8 E4M3 block-128 | 62 | 76,108 | 328,326,771,576 |
| MXFP4 group-32 | 120 | 72,466 | 227,486,055,288 |

Admission validates the selected release's exact extents, hybrid layer
schedule, text tensor roles, representative shapes and dtypes, and quantized
block geometry before generation.

## Chat and server

Two prefixes are **required** to reach the measured rates, not optional tuning:

- `CUDA_DEVICE_ORDER=PCI_BUS_ID` — many shells export `FASTEST_FIRST`. Without
  this, `--devices 1,2` can silently select a different card than intended; on
  the reference host it picks the 16 GiB RTX 5060 Ti plus one 3090 instead of
  the two 3090s, and the run still succeeds while producing different output.
- `numactl --interleave=all` — worth about 1.13x, and it removes a run-to-run
  placement lottery. Under the default policy the checkpoint lands 55.9/44.1
  across the two NUMA nodes in one run and 44.4/55.6 in the next, random in
  direction, which is a 6.7% spread on identical code (record 0218).

```bash
env CUDA_DEVICE_ORDER=PCI_BUS_ID numactl --interleave=all \
  ./build-release/strata-chat \
  --model models/glm53f-mxfp4 --model-type glm53 \
  --devices 1,2 --context-size 2048 --max-new 256 \
  --vram-fraction 0.85

env CUDA_DEVICE_ORDER=PCI_BUS_ID numactl --interleave=all \
  ./build-release/strata-server \
  --model models/glm53f-mxfp4 --model-type glm53 --model-id glm53f-mxfp4 \
  --devices 1,2 --context-size 2048 --max-new 256 \
  --vram-fraction 0.85 --port 8080
```

### Reproducing the published decode rate

The 5.490 tok/s MXFP4 / 3.550 tok/s FP8 figures are this exact command. Run it
twice and read the second run: the static expert tier uploads about 9.5 GiB at
startup, so a cold first process understates by up to 1.8x.

```bash
env CUDA_DEVICE_ORDER=PCI_BUS_ID numactl --interleave=all \
  ./build-release/strata-chat \
  --model models/glm53f-mxfp4 --model-type glm53 \
  --prompt 'Write the natural numbers in order, one per line, starting at 1. Continue until you reach 1000.' \
  --context-size 2048 --max-new 129 --devices 1,2 \
  --vram-fraction 0.85 --temperature 0 --seed 33377335 --no-color
```

## Context and the k-pool sparse indexer

GLM-5.3 chooses which history a sparse-attention layer reads through a k-pool
indexer (`index_topk` 2,048, `index_kpool` 4, `index_kpool_always_select_tail`).
Each query scores compressed four-token pools, keeps the best 512, expands them
back to 2,048 raw positions, and appends the current incomplete pool as a tail,
for a selection at most 2,051 wide. The adapter runs that selection and attends
only to the positions it names.

Two details are taken from the reference rather than inferred, and both are
silent at or below 2,048 and wrong above it: the softmax scale is applied
*inside* the ReLU, where DeepSeek-V4's otherwise similar indexer applies it
outside; and the indexer's key norm is `nn.LayerNorm(head_dim, eps=1e-6)` --
mean subtracting, with a bias -- not the RMSNorm this model uses everywhere
else, including the attention `k_norm`.

This is what bounds cost at long context. The MLA workspace was dense in
history -- `history x 32,768 x 4 B` per sparse layer, so 2.95 GiB across the
eleven layers at 2,048 tokens but 47.2 GiB at 32,768 -- and with the selection
it is constant at any context. Decode is bounded the same way: MLA expands a
fixed selection instead of the whole history.

| context | dense MLA workspace | with selection |
|---:|---:|---:|
| 2,048 | 2.95 GiB | 2.95 GiB |
| 32,768 | 47.24 GiB | 2.95 GiB |
| 131,072 | 188.98 GiB | 2.95 GiB |
| 262,144 | 377.96 GiB | 2.95 GiB |

262,144 is the admitted ceiling. The remaining bound is host sequence state --
about 33.5 KB per token across the eleven sparse layers, the 512-wide latent
plus the 256-wide indexer row -- which is roughly 8.8 GiB at that length. The
checkpoint declares 1,048,576; that is not offered until it has been measured.

### Where the attention runs

The indexer lives on the host path, so a sequence admitted with a context above
2,048 runs its MLA attention on the host for its whole life, and device prefill
is disabled for it. That is a property of the admitted context, not of the
current position: the resident device chain keeps its own latent cache and
computes no indexer state, so switching at the crossing would leave the host
MLA and indexer caches empty for every earlier token. A sequence admitted at or
below 2,048 cannot reach a non-identity selection, keeps the resident device
path, and pays nothing for the indexer.

The consequence is that long context and the resident device MLA are currently
exclusive. Porting the indexer into the resident CUDA chain is open work.

### Measured long-context behaviour

On the reference two-RTX-3090 host with the MXFP4 release, a 2,591-token prompt
at `--context-size 4096` completes and answers correctly from a fact placed
around token 550 -- inside the pooled region, far from the always-selected tail:
`prefill 2,591 tok in 1816.31 s (1.43 tok/s)`, `decode 127 tok in 556.74 s
(0.23 tok/s)`.

**Long-context decode is much slower than the 2,048-token rate and this is a
known open item, not a tuning gap.** At 2,048 tokens decode is 0.182 s/token and
MLA is nearly free because the resident device chain attends against a
pre-expanded KV cache. Above the threshold the indexer forces attention onto the
host, which moves `selection x 32,768 x 4 B` -- about 2.96 GB per token at a
saturated selection -- across PCIe every step. Phase profiling puts 88% of a
long-context decode step in MLA, with no MLA device kernels at all.

The measured feed-forward floor is 0.206 s/token and does not depend on history,
so the 2,048-token operating point is MoE-bound. Because the selection makes MLA
work constant in context, long-context decode should ultimately equal
short-context decode; reaching that requires the indexer to run in the resident
CUDA chain so nothing proportional to context crosses the link. That is the
open performance item.

Decode cost against history, seconds per token, cold process:

| history | seconds/token |
|---:|---:|
| 89 | 0.405 |
| 534 | 0.525 |
| 1,046 | 0.778 |

`--context-size` above 2,048 selects the sparse path for *every* sequence the
runtime admits, whatever its prompt length, so these short-prompt arms exercise
exactly the long-context decode path and make it measurable in minutes rather
than in a 30-minute prefill.

### Exactness

At or below 2,048 the selection is the identity, so the sparse and dense paths
must agree byte for byte; that is the adapter's regression gate and it is free.
It is also blind to everything above 2,048, which is why the indexer went
unimplemented through a whole performance campaign while every gate passed. The
gate that is not blind is `tests/fixtures/glm53/indexer-oracle.json`, frozen by
`scripts/glm53_indexer_oracle.py` from `Glm5NextTextIndexer` in
`huggingface/transformers`, which pins the selected positions on both sides of
the threshold against the reference implementation.

The server exposes the same text runtime through its OpenAI-compatible chat
completion endpoint. Do not send image content: multimodal support is outside
this adapter's current contract.

The server admits requests from the memory that remains after model warm-up,
not from a fixed request-count limit. It charges the complete persistent host
state and the actual KDA/MLA CUDA allocation on every selected device, keeps a
five-percent device safety reserve, and reserves host capacity independently
for exact immutable prefix snapshots. Admission, live/free/fragmented state,
prefix occupancy and eviction, scheduler batch width, TTFT, and inter-token
latency are reported by the GLM runtime.

Active MXFP4 decode requests are grouped at iteration boundaries. Their
independent recurrent states stay disjoint while routed experts are executed
expert-major across the cohort; static and shared device-resident experts use
the same per-request inputs without changing reduction order. FP8 remains
fully supported, but its multi-row device-expert composition did not pass the
full-model exactness gate, so concurrent FP8 requests deliberately use the
accepted batch-1 device path within each scheduler iteration. This is an exact
checkpoint-specific fallback, not a precision or expert-policy change.

Prefix reuse is runtime-local and keyed by the exact token prefix; checkpoint,
tokenizer, configuration, state layout, and numerical mode are immutable
properties of that runtime. Cached state is copy-on-write, mutable tails are
never shared, and completion or cancellation releases the request's persistent
CUDA state before another request is admitted.

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
