# GLM-5.3-Flash: text runbook

GLM-5.3-Flash is a 45-layer, 288-routed-expert model with four mHC streams,
three Kimi Delta Attention layers for every sparse MLA layer, eight selected
experts plus one shared expert. Strata supports all three published
representations:

- **FP8**: block-128 E4M3 with F32 inverse scales, throughout.
- **MXFP4** (Quark): routed experts in E2M1 group-32 with E8M0 scale bytes,
  while the mixed-precision corrections -- layers 3, 5 and 6 -- and the shared
  experts remain BF16.
- **NVFP4** (compressed-tensors `nvfp4-pack-quantized`): routed experts in the
  same E2M1 nibble packing, but with an E4M3 scale byte per 16 columns and one
  F32 per-tensor global scale. Its correction leaves no routed expert behind,
  so every MoE layer from 3 to 44 is packed; the shared experts stay BF16.

It consumes any of the three as published; it does not requantize the model or
reduce its expert count or top-k.

The release is resolved once from the checkpoint index and tensor contract.
There is no precision flag: point `--model` at `models/glm53f`,
`models/glm53f-mxfp4` or `models/glm53f-nvfp4`, and the runtime selects the
corresponding validators, host expert decoder, activation boundary, and device
shared-expert kernel. A checkpoint matching no pinned release is refused rather
than guessed at.

An NVFP4 weight is `e2m1(nibble) * (e4m3(scale) / global_scale)`. The division
is applied to the group scale before the weight is formed, not to the finished
sum, because that is the order the exporter's own dequantization uses; the host
scalar dot, the host AVX2 dot and the device expert kernel all follow it, which
is what lets a host result and a device result be compared bit for bit. Note
that this is the compressed-tensors direction: the ModelOpt NVFP4 exports
Strata reads for other models *multiply* by their per-tensor scale, and the two
must never be interchanged.

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

| release | shards | indexed tensors | declared `total_size` | tensor payload bytes |
|---|---:|---:|---:|---:|
| FP8 E4M3 block-128 | 62 | 76,108 | 328,326,771,576 | 328,326,771,576 |
| MXFP4 group-32 | 120 | 72,466 | 227,486,055,288 | 227,486,055,288 |
| NVFP4 group-16 | 62 | 110,457 | 190,213,869,288 | 190,198,265,848 |

The last two columns coincide for the FP8 and MXFP4 exports, which follow the
Hugging Face convention where `metadata.total_size` is the sum of the tensor
extents. The NVFP4 export declares the shard file total there instead, so the
index's claim and what the shards actually hold are pinned separately rather
than one number being checked twice.

Admission validates the selected release's exact extents, hybrid layer
schedule, text tensor roles, representative shapes and dtypes, and quantized
block geometry before generation.

## Chat and server

Two prefixes and one flag are **required** to reach the measured rates, not optional tuning:

- `CUDA_DEVICE_ORDER=PCI_BUS_ID` — many shells export `FASTEST_FIRST`. Without
  this, `--devices 1,2` can silently select a different card than intended; on
  the reference host it picks the 16 GiB RTX 5060 Ti plus one 3090 instead of
  the two 3090s, and the run still succeeds while producing different output.
- `numactl --interleave=all` — worth about 1.13x, and it removes a run-to-run
  placement lottery. Under the default policy the checkpoint lands 55.9/44.1
  across the two NUMA nodes in one run and 44.4/55.6 in the next, random in
  direction, which is a 6.7% spread on identical code (record 0218).
- `--device-prefill` — worth **1.38x** on prompt processing and by far the larger of the
  two prefill levers, almost entirely by moving KDA off the host: 15.69 s to 0.14 s at
  619 tokens, 112x (record 0240). It defaults **off**, and it is unavailable above
  `--context-size 2048`, because the device chain computes no indexer k-pool state and
  `device_prefill_for_context()` refuses it for any context that can cross `kIndexTopK`.
  `STRATA_GLM53_DEVICE_PREFILL=1` still sets it; the flag and the variable are OR-ed.
- `--prefill-page-tokens 128` — worth a further **1.02x**, byte-identical. The compiled
  default is 64 and the curve is flat from 128 upward, so 128 is the knee. Do not expect
  more from it: weight bytes fall 4.7x between width 64 and 619 and prefill barely moves,
  because the host expert path is **compute-bound** at these shapes, saturating near
  77 GMAC/s — roughly 11% of this host's AVX2 FMA peak (records 0241, 0242).

Together the two are **1.411x** at 619 tokens, byte-identical: 129.40 s to 91.71 s,
medians of interleaved repetitions. Above `--context-size 2048` only the width half
applies, and it is worth about 1.06x.

A third improvement needs no flag at all. The routed page-MoE reduction used to
dispatch one worker task per (row, column) -- about nine multiply-adds behind one
contended atomic increment -- which cost a flat **16.4 s** of every prefill regardless
of page width or prefill placement. Blocking that dispatch at 1024 takes the term to
**0.19 s**, worth **1.194x** on total prefill on its own, byte-identical (record 0243).
It is the default since `2d454be`; `STRATA_GLM53_EXPERT_REDUCTION_BLOCK=1` restores the
old behaviour for A/B work.

A prefill page also runs its routed experts on the device now, using the pinned
expert tier that previously served decode only -- `feedforward_page`'s
`allow_device_tiers` defaulted to false at both prefill call sites, so a page ran
every routed expert on the host. Worth a further **1.09x**, byte-identical, on VRAM
that was already allocated and idle during prefill (record 0244). Demand-staging
the *non*-tier experts is implemented and exact but off by default
(`STRATA_GLM53_DEVICE_PAGE_STAGING=1`): it moves 100% of routed experts to the GPU
and still loses 2.3 s, because the 110.6 GB it uploads runs serially at 3.80 GB/s.

At long prompts the routed experts are demand-staged to the device and every one
of them executes there, on **both** cards rather than only the one that owns the
layer. At 1,925 tokens prefill falls **248.60 s to 148.95 s, 7.743 to 12.924
tok/s (1.67x)**, byte-identical, on defaults with no environment overrides:
staging (1.27x), an O(n) counting sort replacing a quadratic one (1.09x), a
grouped kernel launch (1.03x), expert parallelism across devices (1.13x), and a
staging reserve without which none of it engages (1.66x on its own, because the
static tier otherwise fills the device cache at load and leaves demand staging
no room). Records 0244 and 0245.

**Current best at 619 tokens: 129.40 s to 70.07 s, 4.784 to 8.834 tok/s, 1.847x**, with
no kernel rewrite and no precision change. In the long-context
regime (`--context-size 4096`, where device prefill is unavailable) the same three
changes give **147.04 s to 125.89 s, 1.168x**; the reduction fix carries there in full
absolute terms and attention, at 28.6% of the phase, becomes the next target.

```bash
env CUDA_DEVICE_ORDER=PCI_BUS_ID numactl --interleave=all \
  ./build-release/strata-chat \
  --model models/glm53f-mxfp4 --model-type glm53 \
  --devices 1,2 --context-size 2048 --max-new 256 \
  --device-prefill --prefill-page-tokens 128 --vram-fraction 0.85

env CUDA_DEVICE_ORDER=PCI_BUS_ID numactl --interleave=all \
  ./build-release/strata-server \
  --model models/glm53f-mxfp4 --model-type glm53 --model-id glm53f-mxfp4 \
  --devices 1,2 --context-size 2048 --max-new 256 \
  --device-prefill --prefill-page-tokens 128 --vram-fraction 0.85 --port 8080
```

### Reasoning

GLM-5.3-Flash always reasons. Its chat template opens a `<think>` block
unconditionally and ships no `enable_thinking` toggle, so unlike GLM-5.2 --
whose renderer emits `<think></think>` when thinking is disabled -- there is no
value that turns it off. What the template does accept is a budget:

```bash
--reasoning-effort low | high | max      # default max
```

The flag is rejected for a model whose registration declares no budget, and an
unrecognized value is rejected rather than accepted, because the template maps
anything it does not recognize to `Max` -- silently running the most verbose
setting for a request that asked for the least.

Budget interacts with `--max-new`: generation has to reach `</think>` before
any answer appears, so a `--max-new` that expires inside the scratchpad returns
reasoning and no answer. At `max` on a 26-token prompt, 48 tokens were not
enough to finish thinking; at `low` the same prompt answered in 10.

`strata-server` takes the same flag as the server-wide default, and a request
may override it per call. Both the OpenAI spelling and the `chat_template_kwargs`
form vLLM clients send are accepted:

```jsonc
{"reasoning_effort": "low"}                          // OpenAI
{"chat_template_kwargs": {"reasoning_effort": "low"}} // vLLM
```

Other template kwargs are skipped rather than rejected, so a client that also
talks to vLLM and sends `enable_thinking` keeps working -- but note that
`enable_thinking` does nothing here, exactly as it does nothing on vLLM against
this checkpoint, because the template never reads it. A budget the model does
not accept is a 400, not a 500.

Chat responses separate the two halves, matching the shape vLLM produces with
`--reasoning-parser`:

```jsonc
"message": {"role": "assistant",
            "reasoning_content": "15*37 = 555",
            "content": "555"}
```

Streaming emits `reasoning_content` deltas and then `content` deltas; the piece
that completes `</think>` may carry both. `include_reasoning: false` withholds
the reasoning from the response -- it is still generated, because the model
cannot be told not to. The completions endpoint has no field for reasoning, so
its text is passed through whole. JSON mode validates the answer rather than the
scratchpad.

`strata-chat` applies the budget but does not split the block, so `</think>`
appears inline in its output.

### Reproducing the published decode rate

The 5.490 tok/s MXFP4 / 3.550 tok/s FP8 figures are this exact command. Run it
twice and read the second run: the static expert tier uploads about 9.5 GiB at
startup, so a cold first process understates by up to 1.8x.

```bash
env CUDA_DEVICE_ORDER=PCI_BUS_ID STRATA_GLM53_DEVICE_PREFILL=1 \
  numactl --interleave=all \
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

### Choosing `--context-size` and `--max-new`

Any context in [1, 262,144] loads, and any `--max-new` below it generates.
There is no combination to discover by trial: the only rule is that a turn's
prompt and its generation share the one window, so `--max-new` must leave room
for a prompt. Both `strata-chat` and `strata-server` reject `--max-new >=
--context-size` at argument parsing rather than after the checkpoint has
loaded, and a chat whose history outgrows the window drops its oldest turn and
re-renders, as the other runtimes do, instead of failing the session.

What a larger context costs is VRAM, and the runtime now charges it rather
than hoping it fits. Two allocations live outside the weight arena and scale
with the admitted context: the MLA activation workspace, and one complete
sequence state -- the per-layer latent cache plus either the dense BF16
expansion or the bounded sparse arena. Above the indexer threshold the
workspace is flat, because a call expands a bounded selection instead of the
history; the sequence state still grows as the latent cache does. Whatever
those two need beyond the headroom the vram fraction already leaves is taken
out of the static routed-expert tier, so a long context serves at a lower tier
coverage instead of running the device out of memory. The short-context
operating point above is unchanged: at 2,048 the demand fits the existing
headroom and the tier is not touched.

Until this was charged, the dense workspace was reserved for sparse contexts
too -- 4 GiB per device at 32,768 and 32 GiB at the ceiling -- so admission
turned on a margin of tens of megabytes. On the reference host `--context-size
32000` loaded and `32768` did not, with nothing to tell the two apart.

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
`prefill 2,591 tok in 1139.72 s (2.27 tok/s)`, `decode 127 tok in 35.16 s
(3.61 tok/s)`.

The indexer runs in the resident CUDA chain. It keeps the compressed latent
history on device, gathers the selection into a bounded BF16 arena and expands
only the pools that entered it since the previous token -- measured pool overlap
between consecutive tokens is 94%, and in practice fewer than two pools per
token are expanded, with 45% of tokens expanding none. Scores are computed on
device; the softmax stays on the host in glibc `exp` with BF16-rounded
coefficients, because that is what keeps the path byte-exact.

**The MoE term is flat in context; what still grows is MLA-side.** Of a 277.96 ms decode step at 2,591 tokens, feed-forward is 184.41 ms -- 66% -- and MLA-side cost is 91.79 ms, of which graph MLA is 80.00 and KDA 4.07. With attention at exactly zero this prompt would reach about 5.42 tok/s, and the measured 3.59 tok/s is 66% of that ceiling.

Feed-forward per decode token does not depend on history:

| history | decode window | feed-forward ms/token |
|---:|---:|---:|
| 89 | 15 | 179.4 |
| 534 | 15 | 181.2 |
| 1,046 | 15 | 183.1 |
| 1,046 | 127 | 181.5 |
| 2,591 | 127 | 184.4 |

That is a 2.8% spread over a 29x range of history. Routed weight volume is flat as well at 3.69-3.89 GB/token, and at a matched 127-token decode window history 2,591 reads slightly *fewer* routed bytes per token than history 1,046, 3.838 against 3.888, and misses the static tier slightly less often, 228.8 against 231.7 per token. The static-tier coverage difference that looks like a context effect is decode-position aliasing: widening the window from 15 to 127 tokens at constant history moves misses from 218.1 to 231.7 per token, because later generated tokens route to less popular experts.

Two figures that are easy to misread, and one that was wrong:

- **An earlier revision of this section reported 221.3 ms/token of feed-forward at 2,591, 77% of the step, about 51 ms of MLA-side cost left over, and a 4.43 tok/s ceiling that the run was 81% of.** That subtraction crossed two commits. The 276.9 ms step was measured at `ec1a044`; the 221.3 ms feed-forward came from `3ddee8a`, whose own profile records 284.4 ms/token of MLA-side cost in a 505.7 ms step. Feed-forward at a fixed operating point varies by up to 2.4x across commits according to how much concurrent host work the MLA path is doing, so it has to be read from the same profile as the step it is subtracted from. Record 0239 has the measurements.
- **The MoE floor is prompt-dependent.** 184.4 ms/token belongs to this prompt's expert routing, not to the model, and the published 5.49 tok/s short-context rate comes from a different prompt, so it is not a like-for-like target. But the ceiling this prompt's MoE floor allows is 5.42 tok/s, so a long-context rate near the short-context one is not arithmetically excluded the way the earlier 4.43 figure made it look.
- **The context-dependent term is the indexer, not the MoE.** At 2,591 the largest host-side sparse-MLA phases are pool scoring at 24.5 ms/token and indexer projection at 22.96, against 7.96 for the projection at history 1,046, because `wq_b` runs only above `index_topk`. Pools grow as `history/4`, so pool scoring is the term that sets the long-context bound.

At short context the sparse path is at parity with the dense one it replaces:
on the same prompt, dense 5.27 tok/s against sparse 5.14.

Decode cost against history, seconds per token, cold process:

| history | seconds/token |
|---:|---:|
| 89 | ~0.20 |
| 534 | ~0.21 |
| 1,046 | 0.224 |
| 2,591 (saturated) | 0.277 |

`--context-size` above 2,048 selects the sparse path for *every* sequence the runtime admits, whatever its prompt length, so short-prompt arms exercise the same decode path and make it measurable in minutes rather than in a 19-minute prefill. One term escapes them: `wq_b` runs only above `index_topk`, so an arm whose *history* is below 2,048 never executes it, and indexer projection reads 7.96 ms/token there against 22.96 at history 2,591. The short harness is blind to the largest context-dependent term in the indexer, which is exactly the kind of thing it was built to find.

Cold and warm runs agree here: the same saturated arm measured 3.50 tok/s cold
and 3.52 warm, 0.6% apart. The runbook's warm-up caveat describes short bounded
runs, where the 9.5 GiB static-tier upload is a large share of the arm; a long
prefill amortizes it away.

**Cost of the device path.** Delta reuse needs one persistent expanded arena per
sparse layer, since expanded rows are layer-specific -- eleven arenas totalling
about 1.85 GB. That reduces concurrent sequence admission to four on this host.

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
uses its E4M3/F32-scale dot; the BF16 shared weights of both FP4 releases use an
exact BF16 dot. All return raw linear results so every BF16 rounding, clamp and
SwiGLU remains on the host. Set `STRATA_GLM53_SHARED_EXPERT_DEVICE=0` only to force the slower
host control.

The remaining admitted expert-arena capacity is filled once at startup with a
static routed-expert tier. Its compact built-in ranking comes from the
representative code, prose, multilingual, reasoning, and concurrent-server
workloads; any route that misses the tier continues through the unchanged host
path. The tier never replaces an expert during inference, so it has none of the
replacement latency that rejected the earlier dynamic-cache experiment. The
runtime discovers the available arena capacity and device placement rather
than assuming a particular GPU count or memory size. It stores FP8 experts in
their canonical E4M3 layout, MXFP4 experts in their checkpoint-native E2M1
group-32 layout and NVFP4 experts in their E2M1 group-16 layout with the F32
divisor carried on the descriptor rather than in device memory, and preserves
the host dot-product association exactly. Set
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
