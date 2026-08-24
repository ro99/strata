# Laguna S 2.1: fast-path runbook

This is the short path from a Laguna checkpoint to the fastest accepted Strata
configuration. The commands use the MXFP4 checkpoint at `models/laguna`; Strata
also admits the model's native NVFP4 representation. It reads the original
Safetensors directly, so there is no conversion step.

## Copy-paste: build and chat

Build an optimized binary once, and rebuild it after pulling runtime changes.
`make check` is the correctness suite; it is not the binary used for the speed
measurement.

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel \
  --target strata-chat strata-server strata-laguna-profile
```

Reproduce the measured GPU order, check that all three cards are free, then
preflight the placement without loading weights:

```bash
export CUDA_DEVICE_ORDER=FASTEST_FIRST
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free \
  --format=csv

./build-release/strata-chat \
  --model models/laguna --model-type laguna \
  --devices 0,1,2 --context-size 16384 --max-new 2048 \
  --vram-fraction 0.85 --dry-run
```

On the reference mixed-GPU workstation, `FASTEST_FIRST` makes CUDA devices 0
and 1 the RTX 3090s and device 2 the RTX 5060 Ti. This ordering is intentional:
Laguna's weighted schedule then puts all twelve global-attention layers on the
3090s. `nvidia-smi` numbers cards in PCI-bus order instead, but selecting
`0,1,2` still selects the same three physical cards. Do not replace this export
with `PCI_BUS_ID` when reproducing the headline.

If the dry run says `fits with a host tier`, start interactive chat with the
same placement flags:

```bash
./build-release/strata-chat \
  --model models/laguna --model-type laguna \
  --devices 0,1,2 --context-size 16384 --max-new 2048 \
  --vram-fraction 0.85
```

Keep the process open: Laguna's routed-expert cache warms during the first
response and is reused by later turns. Each answer is followed by that turn's
prefill and decode rates, and `/exit` (or Ctrl+D) prints the session averages.

For a one-shot prompt:

```bash
./build-release/strata-chat \
  --model models/laguna --model-type laguna \
  --devices 0,1,2 --context-size 16384 --max-new 2048 \
  --vram-fraction 0.85 \
  --prompt "Write a small C++ HTTP parser and explain its invariants."
```

Laguna enables CUDA attention and device-resident BF16 decode KV automatically.
There is no Laguna TP flag to add. `--device-resident-runtime` and
`--decode-topology rank-local-tp2` are DeepSeek-only and are rejected for this
model.

## Context sizing

`--context-size` is the total budget for the rendered chat prompt, retained
history, and requested output. Strata fails explicitly when the prompt plus
`--max-new` exceeds it.

| Desired use | Flags | Maximum rendered input |
|---|---|---:|
| 16K total window, reserve 2K output | `--context-size 16384 --max-new 2048` | 14,336 tokens |
| 16K input plus 2K output | `--context-size 18432 --max-new 2048` | 16,384 tokens |

Chat-template tokens count as input, so use a little more than 18,432 if the
raw user content itself must reach 16K. A conservative command is:

```bash
./build-release/strata-chat \
  --model models/laguna --model-type laguna \
  --devices 0,1,2 --context-size 20000 --max-new 2048 \
  --vram-fraction 0.85
```

Always repeat `--dry-run` after changing the context bound. At 16,384 tokens,
Laguna's persistent BF16 KV rings occupy 840 MiB in total: twelve global layers
retain the full context and 36 sliding layers retain 512 rows. The measured
18.626 tok/s point used a short live context; global attention work grows with
the live history, so 18.626 tok/s is not a 16K-context claim.

## Copy-paste: OpenAI-compatible server

The server uses the same Laguna runtime and keeps its expert cache alive across
requests:

```bash
export CUDA_DEVICE_ORDER=FASTEST_FIRST

./build-release/strata-server \
  --model models/laguna --model-type laguna --model-id laguna \
  --devices 0,1,2 --context-size 16384 --max-new 2048 \
  --vram-fraction 0.85 \
  --host 127.0.0.1 --port 8080
```

In another terminal, send a streaming chat request:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "laguna",
    "messages": [{"role": "user", "content": "Write a lock-free queue."}],
    "temperature": 0,
    "max_tokens": 2048,
    "stream": true
  }'
```

The startup `--max-new` is the default when a request omits `max_tokens`.
Passing `max_tokens` in the JSON makes the request explicit. Use
`--context-size 18432` or more on the server for a 16K rendered input plus a 2K
completion, for the same reason as chat.

## What speed has actually been measured

The accepted headline is a controlled single-stream decode result, not
aggregate batch throughput:

| State | Decode speed | Step time |
|---|---:|---:|
| Fresh first generation | 9.109--9.183 tok/s | expert cache still admitting |
| Warm steady-state median | **18.626 tok/s** | **53.69 ms/token** |
| Fastest controlled warm arm | 18.701 tok/s | 53.47 ms/token |

Operating point: commit `8ede56b`, a 63.665 GiB Laguna S 2.1 checkpoint, two RTX
3090 24 GiB cards plus one RTX 5060 Ti 16 GiB card, 251 GiB RAM,
`--vram-fraction 0.85`, context bound 256, 57 prompt tokens, 79 measured decode
steps, greedy sampling, and the second generation in the same process. The six
interleaved arms took less than five minutes in total and emitted identical
text.

The accepted change removed an accidental KV-history restage and serial score
path. It reduced FlashAttention H2D traffic from 37.93 to 1.73 MiB/token and
the flash phase from 56.59 to 4.20 ms/token. The controlled old path was 9.446
tok/s, so the measured improvement is 1.971x. Relative to the originally
reported 3.75 tok/s, the current warm point is 4.97x; it is not yet the requested
10x or 30 tok/s.

## Reproduce the 18.6 tok/s measurement

The profiling binary repeats generation without reloading the checkpoint, which
makes the cold-versus-warm distinction visible:

```bash
export CUDA_DEVICE_ORDER=FASTEST_FIRST

./build-release/strata-laguna-profile \
  --model models/laguna --devices 0,1,2 \
  --context 256 --max-new 80 --repetitions 2 \
  --vram-fraction 0.85 --no-detailed-cuda-timing
```

Look for `decode_tok_s` under repetition 2. Use the normal serving setting
`--no-detailed-cuda-timing`; CUDA event instrumentation changes the operating
point. For the full three-pair old/new experiment, run:

```bash
scripts/laguna_device_kv_ab.sh models/laguna results/laguna-device-kv-ab
```

That matrix intentionally includes `--host-kv` control arms and takes under
five minutes on the reference workstation. `--host-kv` exists only in the
profiler to reproduce the rejected old path; do not use it for chat or serving.

## Rules for keeping the fast path fast

- Use a Release build. An unoptimized build is not a throughput result.
- Keep one chat or server process alive so the routed-expert cache remains warm.
- Use all three reference GPUs and `--vram-fraction 0.85` when comparing with
  the number above. A different GPU set changes layer placement and cache size.
- Leave Laguna's automatic CUDA attention path enabled. No extra flash or KV
  switch is required.
- Measure the live context you care about. A 256-token admission does not
  characterize a 16K history.
- Avoid `--future-entropy` when measuring raw throughput: with `N` candidates it
  deliberately costs `N+1` decode passes per emitted token.
- Do not add TP=2 expecting DeepSeek's result. Laguna's exact two-3090
  projection probe produced only 1.115x on Q, 1.011x on O, and a 0.597x
  regression on the shared projection, so no TP runtime was promoted.

## Evidence and deeper reference

- [Device-resident KV decode experiment](../experiments/0171-laguna-device-resident-kv-decode.md)
- [Rejected Laguna TP=2 gate](../experiments/0172-laguna-spine-tp2-gate.md)
- [All CLI and placement flags](../cli.md)
- [OpenAI-compatible server](../server.md)
- [Sampling semantics](../sampling.md)
