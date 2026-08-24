# Gemma 4 31B-IT: fast-path runbook

This is the copy-paste path for the fastest accepted Gemma 4 configuration in
Strata. It uses the single-shard MXFP4 checkpoint at `models/gemma4` directly;
there is no conversion step. The model, vision tower, workspace, and a 16K KV
cache fit on one RTX 3090 at a VRAM fraction of 0.95, avoiding cross-device
activation hops.

## Build once

Use a Release build for chat, serving, and measurements:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel \
  --target strata-chat strata-server strata-gemma4-run
```

Rebuild after pulling runtime changes. `make check` is the correctness gate;
it is not the optimized binary used for the rates below.

## Copy-paste: chat

The reference measurements use PCI-bus ordering and the second physical RTX
3090. Preserve that mapping and confirm the card is free:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free \
  --format=csv

./build-release/strata-chat \
  --model models/gemma4 --model-type gemma4 \
  --devices 1 --vram-fraction 0.95 \
  --context-size 16384 --max-new 2048 --dry-run
```

If the preflight reports zero cross-device hops and a `fits` verdict, start
interactive chat with the same placement:

```bash
./build-release/strata-chat \
  --model models/gemma4 --model-type gemma4 \
  --devices 1 --vram-fraction 0.95 \
  --context-size 16384 --max-new 2048
```

For a one-shot prompt:

```bash
./build-release/strata-chat \
  --model models/gemma4 --model-type gemma4 \
  --devices 1 --vram-fraction 0.95 \
  --context-size 16384 --max-new 2048 \
  --prompt "Explain the invariants of a lock-free queue."
```

Gemma's MXFP4 Marlin layout, device-resident decode, persistent device KV, and
bounded first-page executor are automatic. Do not add DeepSeek's
`--device-resident-runtime` or `--decode-topology` flags; they are different
runtime contracts.

## Context sizing

`--context-size` is the total rendered prompt, retained history, and output
budget. The runtime fails explicitly if prompt plus `--max-new` exceeds it.

| Desired use | Flags | Maximum rendered input |
|---|---|---:|
| 16K total, reserve 2K output | `--context-size 16384 --max-new 2048` | 14,336 tokens |
| 16K input plus 2K output | `--context-size 18432 --max-new 2048` | 16,384 tokens |

Chat-template tokens count as input. Leave additional margin if the raw user
content itself must reach 16K:

```bash
./build-release/strata-chat \
  --model models/gemma4 --model-type gemma4 \
  --devices 1 --vram-fraction 0.95 \
  --context-size 20000 --max-new 2048 --dry-run
```

At the measured 16,384-token bound, preflight admits 20.93 GiB on the one
3090: 17.15 GiB of weights and vision data, 2.03 GiB of exact BF16 KV, and a
768 MiB workspace. The placement has zero host/storage weight traffic and zero
cross-device hops.

The headline prefill fast path is currently narrower than the context window:
it applies only to a text-only first rendered prompt of at most 128 tokens on
one GPU. Longer prompts remain correct but use the older page path and must not
be expected to reproduce the 128-token headline. Decode remains on the Marlin
device path after either prefill path.

## Copy-paste: OpenAI-compatible server

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID

./build-release/strata-server \
  --model models/gemma4 --model-type gemma4 --model-id gemma4 \
  --devices 1 --vram-fraction 0.95 \
  --context-size 16384 --max-new 2048 \
  --host 127.0.0.1 --port 8080
```

From another terminal:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma4",
    "messages": [{"role": "user", "content": "Write a C++ arena allocator."}],
    "temperature": 0,
    "max_tokens": 2048,
    "stream": true
  }'
```

The startup `--max-new` is the request default. An explicit `max_tokens` in
JSON overrides it within the configured context ceiling.

## What has actually been measured

These are single-stream rates, not aggregate serving throughput:

| Engine/path | 128-token prefill | Batch-one decode |
|---|---:|---:|
| Original Strata defect | **20.14 tok/s** | **18.03 tok/s** |
| Current Strata fast path | **668.99 tok/s** | **29.747 tok/s** |
| vLLM 2.3.8 TP=1 reference | **881.67 tok/s** | **36.214 tok/s** |

Operating point: the exact 19,531,513,296-byte MXFP4 checkpoint, one RTX 3090
at PCI bus `82:00.0`, SM clock locked to 1,605 MHz, 250 W, VRAM fraction 0.95,
context bound 512, temperature zero, and no cross-device hops. Final accepted
medians and every run are recorded in experiment 0186.

The page executor removes the defect opened by experiment 0165: a 128-token
page now costs 1.495 ms/token, versus 33.617 ms for steady batch-one
decode. Prefill is therefore over 20 times cheaper per token, as the
architecture predicts. Strata has not reached vLLM parity: the remaining gap
is tracked in [issue #36](https://github.com/ro99/strata/issues/36), and the
first-page executor still needs a wider prompt admission contract.

## Reproduce the controlled page A/B

The script runs three counterbalanced fresh processes per arm and verifies the
Release build before loading weights:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
scripts/gemma4_device_page_ab.sh results/gemma4-device-page-ab
```

The candidate is the default. `STRATA_GEMMA4_DEVICE_PAGE=0` exists only as the
experiment control; do not set it for normal chat or serving. The script's
prompt renders to exactly 128 tokens and requests one output token so decode
does not contaminate the page measurement.

Reproduce the controlled 126-step steady decode window with:

```bash
scripts/gemma4_decode_bench.sh results/gemma4-decode
```

## Keep the fast path fast

- Use a Release build, PCI-bus CUDA order, device 1, and VRAM fraction 0.95 for
  comparisons with the reference numbers.
- Keep all 60 text layers on one 3090. The default 0.85 fraction can split the
  model and adds cross-device activation hops.
- Confirm the 250 W / 1,605 MHz production operating point before comparing
  runs; do not mix clock states.
- Keep the process alive to avoid paying the roughly 19-second checkpoint load
  for every request.
- Do not quote the 128-token rate for a longer prompt. Measure the live prompt
  and context shape that matters to the deployment.
- Avoid future-entropy sampling when measuring raw decode: `N` candidates cost
  `N+1` forward passes per emitted token by design.

## Evidence

- [Origin of the prefill defect](../experiments/0165-gemma4-mxfp4-register-fed.md)
- [Direct vLLM reference](../experiments/0181-gemma4-vllm-prefill-reference-2026-08-24.md)
- [Accepted FP32 Marlin primitive](../experiments/0184-gemma4-marlin-fp32-epilogue-accepted-2026-08-24.md)
- [Unified M=1 layout](../experiments/0185-gemma4-marlin-unified-m1-layout-accepted-2026-08-24.md)
- [Device-page integration and KV fix](../experiments/0186-gemma4-device-page-runtime-accepted-2026-08-24.md)
- [Remaining vLLM parity work](https://github.com/ro99/strata/issues/36)
- [CLI and placement flags](../cli.md)
- [OpenAI-compatible server](../server.md)
