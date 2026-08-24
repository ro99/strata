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
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel \
  --target strata-chat strata-server strata-deepseek-run
```

Rebuild after pulling runtime changes. `make check` is the correctness suite;
it is not the optimized binary used for the rates below.

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
before the server is ready. Wait for `[ready] http://127.0.0.1:8133` — roughly
three minutes cold on the reference machine, less when the pages are already
resident.

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

Greedy, batch-one, `temperature: 0`, medians of three to seven interleaved
reps. Reference hardware: two RTX 3090 24 GiB cards, 251 GiB RAM,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, devices `1,2`, VRAM fraction 0.95, rank-local
TP2, 16K context.

| Configuration | Decode, 27-token prompt | Decode, ~2,000-word prompt | Prefill, ~2,000 words |
|---|---:|---:|---:|
| No tier | 8.571 tok/s (116.7 ms/tok) | 7.302 tok/s | 18.147 tok/s |
| **10 GB tier** | **9.171 tok/s (108.9 ms/tok)** | **8.077 tok/s** | 16.649 tok/s |
| Relative | **1.070x** | **1.106x** | **0.917x** |

So the tier buys 7% of decode on a short prompt and 11% at long context, and
costs 8% of prefill. Both are real and both were measured on the same binary.

Two numbers that are **not** claims about this configuration:

- The register-fed FP8 shared-expert kernels are worth about **1.7%** here
  (118.7 -> 116.7 ms/tok). DeepSeek V4 is the model where GPU kernel work does
  not pay, because the host DRAM read is `argmax_r` and the shared expert is
  the only per-token CUDA dispatch decode makes.
- **The RTX 5060 Ti is not used and cannot currently be used.** Every attempt
  is recorded as a rejection; see the evidence list.

## Reproduce the rates

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

## Evidence

- [The tier split falsified; the serial tier measured](../experiments/0178-dsv4-tier-overlap-falsified-2026-08-24.md)
- [Where DeepSeek V4 decode actually dispatches](../experiments/0163-dsv4-sm86-mix1-decode-route-map-2026-08-23.md)
- [The static hot-expert tier and its held-out coverage](../experiments/0124-dsv4-static-hot-expert-tier-2026-08-18.md)
- [The rank-pool width regression](../experiments/0123-dsv4-rank-pool-width-regression-2026-08-18.md)
- [CLI and placement flags](../cli.md)
- [OpenAI-compatible server](../server.md)
