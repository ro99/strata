# Inkling Small: fast-path runbook

This is the copy-paste path for Inkling Small (sometimes called “Inklink” in
conversation) on Strata's fastest accepted configuration. The commands use the
MXFP4 checkpoint at `models/inkling`; the native NVFP4 checkpoint is also
supported. Strata reads Safetensors directly, so there is no conversion step.

## Build once

Use a Release build for chat, serving, and measurements:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel \
  --target strata-chat strata-server strata-inkling-probe
```

Rebuild after pulling runtime changes. `make check` is the correctness suite;
it is not the optimized binary used for the rates below.

## Copy-paste: 16K chat with up to 2K output

The reference machine uses two RTX 3090 24 GiB cards and one RTX 5060 Ti
16 GiB card. Preserve the measured CUDA order and check that the cards are
free:

```bash
export CUDA_DEVICE_ORDER=FASTEST_FIRST
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free \
  --format=csv

./build-release/strata-chat \
  --model models/inkling --model-type inkling \
  --devices 0,1,2 --context-size 16384 --max-new 2048 \
  --vram-fraction 0.85 --dry-run
```

On the reference workstation, `FASTEST_FIRST` makes CUDA devices 0 and 1 the
RTX 3090s and device 2 the RTX 5060 Ti. `nvidia-smi` displays PCI-bus order, so
its printed indices need not match the CUDA indices above.

If preflight says the placement fits, start interactive chat:

```bash
./build-release/strata-chat \
  --model models/inkling --model-type inkling \
  --devices 0,1,2 --context-size 16384 --max-new 2048 \
  --vram-fraction 0.85
```

Type one user turn per line. Keep the process alive: the checkpoint's host
pages and routed-expert VRAM cache are reused across turns. The final decode
rate appears on stderr as `[done] ... tok/s`.

For one prompt:

```bash
./build-release/strata-chat \
  --model models/inkling --model-type inkling \
  --devices 0,1,2 --context-size 16384 --max-new 2048 \
  --vram-fraction 0.85 \
  --prompt "Explain how an LRU cache works, then implement one in C++."
```

The fast paths are automatic: direct MXFP4 mapped staging, the bounded CUDA
weight arena, deferred copy-stream uploads, and persistent exact BF16 device KV
attention at long context. There is no Inkling TP or expert-parallel flag to
add. DeepSeek's `--device-resident-runtime` and
`--decode-topology rank-local-tp2` do not apply.

## Context sizing

`--context-size` covers the rendered chat prompt, retained history, and output.
The runtime fails explicitly if prompt plus `--max-new` exceeds it.

| Desired use | Flags | Maximum rendered input |
|---|---|---:|
| 16K total, reserve 2K output | `--context-size 16384 --max-new 2048` | 14,336 tokens |
| 16K input plus 2K output | `--context-size 18432 --max-new 2048` | 16,384 tokens |

Chat-template tokens count as input. If the raw user content itself must reach
16K, leave margin beyond 18,432:

```bash
./build-release/strata-chat \
  --model models/inkling --model-type inkling \
  --devices 0,1,2 --context-size 20000 --max-new 2048 \
  --vram-fraction 0.85 --dry-run
```

At a 16,384-token bound, the exact device KV rings occupy about 518 MiB total:
seven global layers retain the whole window and 35 local layers retain 512
rows. This is outside the 85% weight arena but inside the deliberately reserved
15% headroom. Attention automatically moves to the exact CUDA path after 512
live rows; the short-context scalar path remains the conservative default below
that crossover.

## Copy-paste: OpenAI-compatible server

```bash
export CUDA_DEVICE_ORDER=FASTEST_FIRST

./build-release/strata-server \
  --model models/inkling --model-type inkling --model-id inkling \
  --devices 0,1,2 --context-size 16384 --max-new 2048 \
  --vram-fraction 0.85 \
  --host 127.0.0.1 --port 8080
```

From another terminal:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "inkling",
    "messages": [{"role": "user", "content": "Write a lock-free queue."}],
    "temperature": 0,
    "max_tokens": 2048,
    "stream": true
  }'
```

The startup `--max-new` is the request default. An explicit `max_tokens` in
JSON overrides it within the configured context ceiling. Use a context of at
least 18,432 for a 16K rendered input plus a 2K completion.

## What has actually been measured

These are greedy, batch-one decode rates, not aggregate serving throughput:

| State | Decode speed | Meaning |
|---|---:|---|
| Original reported path | about 3.75 tok/s | pinned bounce staging and allocator/ordering defects |
| Fresh 16-token median | **5.673 tok/s** | compulsory VRAM-cache admission dominates |
| Fresh 128-token median | **9.072 tok/s** | 89.4% expert-cache hits amortize admission |
| 2,048-token validation | **12.187 tok/s** | 16K admission; live history grows from 5 to 2,052 |
| Repeated identical 16-token route | **28.010 tok/s** | selected expert working set already in VRAM |

Reference hardware: two RTX 3090 24 GiB cards, one RTX 5060 Ti 16 GiB card,
251 GiB RAM, `CUDA_DEVICE_ORDER=FASTEST_FIRST`, devices `0,1,2`, and VRAM
fraction 0.85. The 128-token median is from three interleaved fresh processes:
`9.037, 9.072, 9.107 tok/s`, a five-token prompt, context bound 512, and a
page-cache-resident 130.638 GiB MXFP4 checkpoint. It staged 31.58 GiB during
127 decode forwards. The repeated-route number is the third generation in one
process and is a cache ceiling, not a promise for unrelated prompts.

Relative to 3.75 tok/s, the accepted fresh 128-token point is 2.42x and the
warm route is 7.47x. The requested 10x / 30+ tok/s fresh-route target has not
been reached. This distinction matters: quoting 28 tok/s without saying that
the route was already resident would misrepresent first-response performance.

The 2,048-token functional run completed all requested tokens in 168.05
seconds, staged 42.92 GiB, and allocated KV as 0.03 / 0.03 / 0.45 GiB across
the three devices. It validates `--context-size 16384 --max-new 2048` without
fallback. It does not fill a 16K history, so it is not a 16K-live throughput
measurement.

The vLLM gap is therefore not explained by TP=2 alone. The measured Strata
first response is dominated by compulsory routed-expert transfers from a model
larger than aggregate VRAM; once those bytes are resident, Strata is already
near vLLM's reported 30 tok/s. An exact expert-parallel screen was rejected:
splitting Inkling's existing six-expert fused batch added command/barrier cost,
slowed the zero-miss path to 18.142 tok/s, and heterogeneous GPU numerics changed
later routing. That failed runtime remains isolated on its experiment branch.

## Reproduce the rates

First fault the checkpoint into RAM and show the same-route cache ceiling:

```bash
export CUDA_DEVICE_ORDER=FASTEST_FIRST

./build-release/strata-inkling-probe \
  --model models/inkling --devices 0,1,2 \
  --context 512 --tokens 16 --repeat 3
```

Look at repetitions 2 and 3. The first repetition admits its selected experts;
later repetitions reuse that exact route. To reproduce the fresh 128-token
point after the host mapping is resident:

```bash
RESULT_DIR=results/inkling-deferred-upload-128 TOKENS=128 \
  scripts/inkling_deferred_upload_ab.sh
```

The accepted arm is `deferred-*`. The script runs three interleaved controls
and candidates and takes only a few minutes on the reference machine.

## Keep the fast path fast

- Use a Release build and keep `CUDA_DEVICE_ORDER=FASTEST_FIRST` when comparing
  with the numbers above.
- Keep chat/server alive. Restarting discards the VRAM expert cache.
- Let startup warm the checkpoint's host pages. The model exceeds aggregate
  VRAM, but it fits the reference machine's RAM; steady decode should not wait
  on NVMe.
- Use all three reference GPUs and VRAM fraction 0.85 for like-for-like rates.
  A different device set changes spine placement and cache capacity.
- Do not force pinned bounce staging: it measured slower than direct mapped
  MXFP4 uploads.
- Do not add TP=2 or expert parallelism by analogy with DeepSeek. Inkling's
  measured bottleneck and fused expert command shape are different.
- Measure the live context you care about. The 9.072 and 28.010 tok/s points
  use short histories and are not 16K-live-context claims.

## Evidence

- [Direct mapped MXFP4 staging](../experiments/0173-inkling-direct-mapped-mxfp4-staging.md)
- [Persistent long-context KV attention](../experiments/0174-inkling-device-resident-kv-attention.md)
- [Bounded CUDA weight arena](../experiments/0175-inkling-bounded-weight-arena.md)
- [Deferred mapped expert uploads](../experiments/0176-inkling-deferred-mapped-expert-upload.md)
- [CLI and placement flags](../cli.md)
- [OpenAI-compatible server](../server.md)
