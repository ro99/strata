# DeepSeek V4: fast-path runbook

This is the copy-paste path for DeepSeek V4 on Strata's fastest accepted
configuration. The commands use the checkpoint at `models/dsv4f`. Strata reads
Safetensors directly, so there is no conversion step.

DeepSeek V4 is not like the other models in this folder. Its routed-expert set
is **147 GB against 48 GB of aggregate VRAM**, so it can never be resident:
decode reads routed experts from host DRAM every token, and that read — not any
GPU kernel — is what sets the rate. Every instruction below follows from that.

## Build once

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_NCCL=ON
cmake --build build-release --parallel \
  --target strata-chat strata-server strata-deepseek-run
```

Rank-local TP2 requires NCCL and fails closed when support is absent, so do not
omit `-DSTRATA_ENABLE_NCCL=ON`. CMake searches the active Python environment,
`NCCL_HOME`, standard system paths, and the CUDA toolkit. Rebuild after pulling
runtime changes. `make check` is the correctness suite; it is not the optimized
binary used for the rates below.

## Pin the device order first

**This is not optional and it is not the same as the other runbooks.** Laguna
and Inkling ask for `FASTEST_FIRST`. DeepSeek's rank-local TP2 route needs the
two RTX 3090s as a matched pair, and `FASTEST_FIRST` does not give you that on
a mixed box:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free \
  --format=csv
```

Under `PCI_BUS_ID` on the reference workstation, device 0 is the RTX 5060 Ti
and devices **1,2** are the two RTX 3090s.

`strata-server` pins this order itself, but only when you have not set the
variable — so a shell that exports `FASTEST_FIRST` silently wins, and
`--devices 1,2` then selects the 5060 Ti plus one 3090. That caps both ranks'
symmetric VRAM admission at the 16 GiB card and leaves a 3090 idle, while the
server still boots and still produces correct tokens.

**Check the placement report before trusting any measurement.** The two
`admitted budget` columns must be equal:

```
device total       44.27 GiB            22.14 GiB   22.14 GiB
admitted budget                         22.14 GiB   22.14 GiB
```

If they read `22.14 GiB   14.57 GiB`, the order is wrong.

## Copy-paste: chat

First run the placement preflight. It reads no checkpoint weights, and catches
the wrong device order, missing NCCL support, unsupported GPUs, and insufficient
host or device memory before the roughly 100--115-second warm-cache model load:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID

./build-release/strata-chat \
  --model models/dsv4f --model-type deepseek \
  --devices 1,2 --vram-fraction 0.95 \
  --context-size 16384 --max-new 2048 \
  --decode-topology rank-local-tp2 \
  --prefill-page-tokens 8192 --dry-run
```

Verify that the placement reports the two RTX 3090s, equal per-rank admitted
budgets, and a `fits` verdict. Then start interactive chat with the same
settings:

```bash
./build-release/strata-chat \
  --model models/dsv4f --model-type deepseek \
  --devices 1,2 --vram-fraction 0.95 \
  --context-size 16384 --max-new 2048 \
  --decode-topology rank-local-tp2 \
  --prefill-page-tokens 8192
```

For a one-shot prompt:

```bash
./build-release/strata-chat \
  --model models/dsv4f --model-type deepseek \
  --devices 1,2 --vram-fraction 0.95 \
  --context-size 16384 --max-new 2048 \
  --decode-topology rank-local-tp2 --prefill-page-tokens 8192 \
  --prompt "Explain how tensor parallel inference works."
```

`--context-size` covers the rendered prompt, retained chat history, and output.
With `--context-size 16384 --max-new 2048`, at most 14,336 rendered input
tokens remain when the full output allowance is reserved. Chat-template tokens
count toward that input budget. Incremental KV continuation is automatic, so
later turns prefill only the uncached suffix unless `--full-reprefill` is set.

`--decode-topology rank-local-tp2` automatically enables DeepSeek's complete
device-resident runtime contract, including physical KV pages, CUDA attention,
device mHC, and NUMA-local routed experts. Do not redundantly add
`--device-resident-runtime`.

`--prefill-page-tokens 8192` is the accepted high-throughput prompt shape for
both chat and server. It does not allocate an 8,192-row page for a shorter
prompt; it is an upper bound, so keep it in the command for short and long
conversations alike. Omitting it restores the conservative 64-token DeepSeek
default and can make long prompt ingestion substantially slower.

## Copy-paste: OpenAI-compatible server

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID

./build-release/strata-server \
  --model models/dsv4f --model-type deepseek \
  --model-id strata-deepseek-v4 --devices 1,2 \
  --context-size 16384 --vram-fraction 0.95 \
  --decode-topology rank-local-tp2 --prefill-page-tokens 8192 \
  --host 127.0.0.1 --port 8133
```

Loading is not instant: the checkpoint's 156.9 GB host tier is faulted into RAM
before the server is ready. Wait for `[ready] http://127.0.0.1:8133`. Recent
reference runs loaded in roughly 100--115 seconds with filesystem pages warm;
a genuinely cold load can take several minutes.

From another terminal:

```bash
curl -N http://127.0.0.1:8133/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "strata-deepseek-v4",
    "messages": [{"role": "user", "content": "Write a lock-free queue."}],
    "temperature": 0,
    "max_tokens": 256,
    "stream": true
  }'
```

Keep the process alive. Restarting discards the host tier and both ranks' VRAM
caches, and pays the load again.

## The optional routed-expert tier: faster decode, slower prefill

Decode's dominant cost is 3.449 GB of routed-expert weight read from host DRAM
per token. The tier keeps the hottest experts resident in the ranks' VRAM so
the host pools never read them.

It is **opt-in, and it is a trade, not a free win.** Turn it on when decode
latency matters more than prompt throughput:

```bash
./build-release/strata-server \
  --model models/dsv4f --model-type deepseek \
  --model-id strata-deepseek-v4 --devices 1,2 \
  --context-size 16384 --vram-fraction 0.95 \
  --decode-topology rank-local-tp2 --prefill-page-tokens 8192 \
  --static-expert-plan models/dsv4f/expert-residency-hot.plan \
  --static-expert-bytes 5G \
  --host 127.0.0.1 --port 8133
```

**`--static-expert-bytes` is a per-rank budget, not a total.** Each rank slices
the ranking and then truncates to what fits, so `5G` builds a 10 GB tier across
the pair. Passing `10G` builds 20 GB and is not the configuration measured
below. Confirm what was actually built from the startup log:

```
[deepseek-tier] device=1 rank=0/2 pairs=401 bytes=5361106944 (4.99 GiB)
[deepseek-tier] device=2 rank=1/2 pairs=401 bytes=5361106944 (4.99 GiB)
```

**Do not raise it past `5G`.** The tier's return saturates there — a 12 GB tier
measures the same as a 10 GB one — and 14 GB exhausts the weight arena.

## What has actually been measured

Reference hardware is two RTX 3090 24 GiB cards at 1,605 MHz / 250 W, 251 GiB
RAM, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, devices `1,2`, VRAM fraction 0.95,
rank-local TP2, 16K context, and no static expert tier unless stated otherwise.

The current accepted production result is **26.231 prefill tok/s** at exactly
1,925 prompt tokens: 21.592, 27.828, and 26.231 tok/s, median of three
interleaved candidate arms. The disabled-path control ran 21.581, 20.882, and
19.142 tok/s, median 20.882, so keeping query projection through attention on
the GPU improved the median by 25.6%. A 619-token screen reached 17.10 tok/s
and a 133-token smoke reached 7.84 tok/s; these two are single screens, not
median claims.

Fresh batch-one server baselines, taken before the page-query change whose
dispatch excludes single-row decode, measured **8.627 decode tok/s** median for
a 128-token response after a 19-token prompt. Decode becomes slower as retained
context grows because attention has more KV to scan. For example, the earlier
long-context no-tier experiment measured 7.302 tok/s around 2,000 words.

Prefill throughput is strongly shape-dependent. More rows amortize each weight
read, so do not expect the 1,925-token rate from a 30-token prompt and do not
average both into one "session speed." The fresh like-for-like baseline matrix
that motivated the current fix was:

| Engine | ~36-token prefill | ~500-token prefill | ~1,950-token prefill | short-context decode |
|---|---:|---:|---:|---:|
| vLLM | 8.439 tok/s | 41.901 tok/s | 127.490 tok/s | 10.346 tok/s |
| Strata before device query | 4.332 tok/s | 17.431 tok/s | 22.757 tok/s | 8.627 tok/s |
| Strata current | not re-run | 17.10 tok/s screen at 619 | **26.231 tok/s at 1,925** | unchanged path |

The short vLLM prefill result itself ranged from 7.896 to 19.027 tok/s because
fixed request cost dominates such a small prompt. The 26.231 tok/s Strata
result is good enough to land, but it is not vLLM-equivalent; the remaining
long-page gap is recorded, not hidden.

The optional 10 GB routed-expert tier was measured before the device-query
change. It bought 7% decode on a short prompt (8.571 to 9.171 tok/s) and 11% at
long context (7.302 to 8.077 tok/s), while reducing then-current long prefill by
8% (18.147 to 16.649 tok/s). Use it only when that decode trade is preferable;
the current prefill-plus-tier combination has not been re-benchmarked.

Two numbers that are **not** claims about this configuration:

- The register-fed FP8 shared-expert kernels are worth about **1.7%** here
  (118.7 -> 116.7 ms/tok). DeepSeek V4 is the model where GPU kernel work does
  not pay, because the host DRAM read is `argmax_r` and the shared expert is
  the only per-token CUDA dispatch decode makes.
- **The RTX 5060 Ti is not used and cannot currently be used.** Every attempt
  is recorded as a rejection; see the evidence list.

## Reproduce the rates

For an immediate user-facing check, use either copy-paste command above. Chat
prints per-turn prefill and decode rates after each answer. The server returns
the same measurements in its timing fields. Keep `--context-size 16384`,
`--max-new 2048` in chat, and `--prefill-page-tokens 8192`; the page value is
the important prefill setting and remains only an upper bound.

For a controlled decode matrix:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Start a server on 8133 with whatever flags you are testing.
./scripts/dsv4_decode15_server.sh /tmp/dsv4.log \
  --static-expert-plan models/dsv4f/expert-residency-hot.plan \
  --static-expert-bytes 5G

# From another terminal, once [ready] appears:
./scripts/dsv4_decode15_bench.sh 8133 tier 5 results/dsv4 "" 256
```

The bench script saves every response as JSON so greedy output can be compared
byte for byte between arms. Do that: it is what caught a tier dispatch that was
3% faster and intermittently produced a corrupted first token.

A 27-token prompt makes `prompt_per_second` meaningless — it is dominated by
fixed per-request cost. Pass a real prompt file as the fifth argument when
prefill is the question.

## Keep the fast path fast

- Use a Release build, and `PCI_BUS_ID` — not `FASTEST_FIRST`.
- Verify the two `admitted budget` columns are equal before believing a number.
- Keep the server alive; the host tier and VRAM caches are per process.
- Let the host tier fault in fully. The model exceeds aggregate VRAM but fits
  the reference machine's RAM, so steady decode should never wait on NVMe.
- Do not raise `--static-expert-bytes` past `5G`, and remember it is per rank.
- Do not set `STRATA_DSV4_TIER_OVERLAP=1`. It selects a dispatch that is both
  slower and non-deterministic, and exists only so the defect can be worked.
- Measure decode on decode. The tier's coverage was validated on held-out
  decode traces; a prefill-heavy trace flatters every locality claim.

## Related documentation

- [CLI and placement flags](../cli.md)
- [OpenAI-compatible server](../server.md)
