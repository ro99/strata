# DeepSeek-V4-Flash-DSpark runtime

## Design answer

Strata's scheduling and residency design is reusable across GLM-5.2,
DeepSeek-V4, and future dense models, but the original executable graph was not
architecture-neutral. Before this adapter, the model specification, manifest,
checkpoint reader, tokenizer boundary, graph, and CUDA encodings in the runnable
path were GLM-specific.

The safe extension point is an additive architecture adapter over shared
backend and memory-policy interfaces:

| Layer | Shared policy | Architecture-owned semantics |
|---|---|---|
| Source import | bounded Safetensors parsing and immutable extents | tensor roles, shapes, source revision, quantization layouts |
| Numerical graph | CUDA upload/matmul and host state ownership | attention, router, experts, residual/mHC, tokenizer, draft verification |
| Placement | explicit RAM/VRAM budgets and cache accounting | resident-spine membership and atomic expert placement |
| Admission | exact capacity and I/O contract | active tensor set, KV layout, dense-vs-sparse declaration |
| Scheduling | layer/expert device assignment and advisory prefetch | exact route IDs and coefficients; commit/rollback rules |

This keeps the measured GLM executor intact. A Gemma-class dense model that fits
in VRAM should select a `vram-resident` topology and bypass expert/NVMe policy;
DeepSeek selects `resident RAM canonical tier + VRAM spine/cache`; GLM may retain
its bounded-cold NVMe topology. Cache policy cannot turn a dense model that does
not fit aggregate memory into a sparse model.

## Pinned target

The adapter accepts only the inspected `DeepSeek-V4-Flash-DSpark` checkpoint:

- repository revision `62af8fffb2f7030cac4de2f0169f5b8d1101b646`;
- index SHA-256
  `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`;
- 48 shards, 72,317 tensors, and 166,878,536,440 indexed bytes;
- 35,328 native FP4 E2M1/per-32 E8M0 modules and 390 native FP8
  E4M3/128x128 E8M0 modules.

Every shard header, tensor extent, dtype, shape, role, and native quantization
pair is validated before admission. The loader does not create a converted
checkpoint. The output projection `wo_a` follows the supplied target executor:
its FP8 source extent is dequantized once into a BF16 VRAM representation, while
other declared FP8 linears retain native execution.

## Implemented base-model contract

`strata-deepseek-run` implements the main 43-layer autoregressive graph:

- BF16 embedding and response-head boundary with the target tokenizer/template;
- low-rank 64-head attention, 128-token sliding state, attention sinks, YaRN,
  and compression ratios 4 and 128;
- logical cache admission through the declared 1,048,576-token model limit,
  with lazily committed host pages for compressed KV and learned-index state;
- exact compressed-position membership through 2,048 tokens, followed by the
  target's learned top-512 selection in ratio-4 layers; ratio-128 layers retain
  their full heavily-compressed history;
- full-model execution evidence through the first learned-index boundary;
  production-scale 32k/200k/1m ingestion is not yet claimed;
- four-lane mHC pre/post mixing with 20 Sinkhorn iterations;
- hash routes from `tid2eid` in layers 0–2 and bias-selected
  sqrt-softplus/noaux top-6 routes in later layers;
- native FP4 routed experts, FP8 shared experts, asymmetric SwiGLU limit 10,
  routed scale 1.5, and the target's once-applied routed coefficient;
- greedy base-model decode with checkpoint-read accounting.

Before initialization, admission reserves the non-DSpark routed experts and
embedding in anonymous RAM, the main-model spine and workspaces in VRAM, and KV
and index state in RAM. Cache capacity is admitted logically, but 256 KiB host
pages are committed only as rows are produced, so a large configured ceiling
does not zero or touch the complete cache up front. The staged weight arena
becomes read-only. After warm-up, any checkpoint read during decode violates
the zero-NVMe contract and fails the request.

The checkpoint's three DSpark stages, target-layer IDs, block size, noise token,
Markov head, confidence head, and tensor layouts are validated. Optional DSpark
execution is deliberately disabled until provisional KV rollback and target
verification have an oracle. Requesting it is an error; it never becomes an
unverified shortcut. This does not disable any operation in the exact base
model, matching the target's ordinary generation path.

## Current hardware admission

With devices `0,1,2`, an 0.85 fraction of currently free VRAM, a 216 GiB host
ceiling, and context 2,048, the inspected machine admitted this plan on
2026-07-15:

| Resource | Bytes |
|---|---:|
| Required host memory | 150,559,504,732 |
| Main routed-expert source extents | 147,169,738,752 |
| Host parameters | 2,270,214,492 |
| KV/compressor state | 45,809,664 |
| Host workspace reserve | 1,073,741,824 |
| Aggregate VRAM budget | 56,534,889,266 |
| Resident main-model spine | 9,069,011,072 |
| VRAM workspace reserve | 805,306,368 |
| Remaining expert VRAM cache | 46,660,571,826 |
| Steady-state NVMe | 0 |

Admission uses currently free VRAM, so the exact budget changes when other GPU
processes are present.

On the same topology, admission at the model's 1,048,576-token ceiling requires
164,965,579,100 host bytes. Of that total, 14,451,884,032 bytes are logical
KV/index capacity, including 2,818,916,352 bytes for learned-index state. The
VRAM weight plan and zero-NVMe decode contract are unchanged because these
caches are host-resident and lazily committed.

## Resident smoke evidence

The routed expert coefficient is applied once before the down projection, as in
the bundled target inference code. The host executor, CUDA MoE kernel, and CUDA
reference fixture use that same contract.

The initial mHC post-mix treated the Sinkhorn combination matrix's rows as
destinations and columns as sources. The target expression uses source rows and
destination columns. That transpose affected both residual branches in every
layer and caused repetitive or truncated multilingual words despite valid
tokenization and UTF-8 decoding. A non-symmetric operation fixture now pins the
target orientation.

The interactive chat now defaults to greedy decoding (temperature 0) for
deterministic output. Pass `--temperature 1` to enable seeded Gumbel-max
sampling. Exact runtime benchmarks remain greedy unless sampling is
explicitly configured.

The reusable smoke completed in tmux session `strata-deepseek-v4-smoke` on
2026-07-15 with prompt `Hi`, five prompt tokens, two emitted tokens, and one
decode forward. It produced `You are` and a complete 258-row route trace (six
positions by 43 layers):

| Measurement | Result |
|---|---:|
| Initialization | 222.603 s |
| RAM staging | 181.498 s |
| Staged immutable bytes | 148,228,800,512 |
| Prefill | 6.967 s |
| One decode forward | 0.947 s |
| Generation checkpoint reads | 0 bytes |
| Decode checkpoint reads | 0 bytes |
| Peak RSS | 146,298,548 KiB |
| VRAM cache hits / misses / evictions | 4,442 / 3,572 / 0 |

This is a functional smoke, not a throughput claim: it is one run, includes a
very short prompt, and has not yet been compared with the frozen target-model
generation oracle.

## Commands

Validate all source headers and native layouts:

```bash
./build/strata-inspect \
  --model models/DeepSeek-V4-Flash-DSpark --headers --json
```

Check the real machine topology without staging 147 GB:

```bash
./build/strata-deepseek-run \
  --model models/DeepSeek-V4-Flash-DSpark \
  --devices 0,1,2 --host-memory 216G --max-context 2048 \
  --admission-only --json
```

Run the exact base graph:

```bash
./build/strata-deepseek-run \
  --model models/DeepSeek-V4-Flash-DSpark \
  --devices 0,1,2 --host-memory 216G --max-context 2048 \
  --prompt 'Hello' --max-new 16 --json
```

Resident loading uses eight shard readers by default. Spine warm-up uses up to
three device workers and overlaps resident staging, so the usual command takes
the validated fast-loading path without additional flags. For controlled
comparisons, `--serial-resident-warmup` restores sequential staging and warm-up;
`--resident-read-workers` and `--spine-warmup-workers` expose the bounded worker
counts.

The reusable long smoke command is:

```bash
tmux new-session -d -s strata-deepseek-v4-smoke \
  'scripts/run_deepseek_v4_smoke.sh'
```

It writes ignored artifacts under `results/deepseek-v4-smoke/`.

## Remaining correctness gates

Native target-byte operation fixtures, manifest coverage, tokenizer behavior,
CUDA format kernels, and admission are automated. Promotion to a validated
DeepSeek baseline still requires frozen layer outputs from the supplied target
executor, full-model teacher forcing, greedy generation agreement, and an
independent physical-I/O observation. DSpark additionally requires draft,
confidence, target-verification, and rollback fixtures. Until those gates pass,
performance numbers are diagnostic rather than a research win.
