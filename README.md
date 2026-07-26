# Strata

Strata is a C++20/CUDA inference engine for dense and mixture-of-experts models whose weights exceed local VRAM, and often exceed local RAM as well. It splits a single model checkpoint across GPU VRAM, host RAM, and read-only NVMe storage, and executes it with the precision, router semantics, expert count, and top-k the checkpoint actually declares. There is no Python or ML framework in the runtime; it builds as a static library plus a set of CLI binaries.

The motivation is ordinary: most people who want to run a large model do not have a rack of data-center GPUs. They have one or a few consumer GPUs and a reasonable amount of system RAM. Strata's job is to make that combination usable for models that would otherwise only run on much larger machines, without quietly making the model worse to get there.

## Development machine

This is the machine the project is built and measured on. It's an ordinary workstation, not a server:

| Memory Type | Hardware |
|---|---|
| GPUs | RTX 5060 Ti 16 GiB, RTX 3090 24 GiB, RTX 3090 24 GiB (64 GiB total) |
| RAM | 251 GiB DDR4 |
| CPU | Xeon E5-2680 v4 |

On this hardware Strata currently runs two checkpoints: DeepSeek-V4-Flash-DSpark (167 GB, 43 layers, 256 experts, top-6), staged into host RAM so decode does not touch storage after warm-up, and GLM-5.2 (405 GB, 78 layers, 256 experts, top-8), which is larger than the machine's combined VRAM and RAM and therefore runs directly against the checkpoint on disk.

## What "exact" means here

Fitting a large model into limited memory is easy if you're willing to degrade it — drop below four bits, cut experts, reduce top-k, or fall back to a cheaper path when the fast one isn't available. Strata's execution model is built against doing that:

- Weight precision does not go below four bits anywhere in the system: not in
  canonical weights, not in caches, not in KV storage, not in speculative
  drafts, not in prefetch predictors.
- Precision, router semantics, expert count, and top-k are read from the
  checkpoint and never changed silently.
- A predictor may only influence scheduling or prefetch order. It never
  changes what gets computed.
- Exact mode either completes the exact computation or reports a failure. It
  does not fall back to something cheaper without saying so.
- If a dense model is larger than the machine's combined resident memory,
  Strata reports it as I/O-dependent rather than hiding the cost behind a
  cache.

These are enforced in code, not just documented, and they're part of why some numbers below (GLM's decode speed, for instance) are unglamorous: the honest number for a checkpoint that doesn't fit in memory is a slow one.

## Quick start

You need a C++20 compiler, CMake 3.20+, Make, and CUDA 12.8 (or a compatible toolchain) for the GPU backend.

```bash
git clone https://github.com/ro99/strata
cd strata
make check
```

Place a checkpoint under `models/` in its original Safetensors form. Strata reads the shards as published — there's no conversion step and no second copy of the weights on disk.

```bash
./build/strata-chat \
  --model models/DeepSeek-V4-Flash-DSpark --model-type deepseek \
  --context-size 8192 --max-new 256 --devices 0,1,2
```

Startup prints the selected devices, the VRAM budget per device, load progress, and elapsed load time. Decoding defaults to greedy (`--temperature 0`) for reproducible output; `--temperature 1` enables seeded sampling. Multi-turn chat reuses the cached prefix and only prefills new tokens.

Flags worth knowing:

| Flag | Purpose |
|---|---|
| `--devices 0,1,2` | CUDA devices to use |
| `--context-size N` | Context ceiling enforced by the runtime |
| `--vram-fraction F` | Fraction of free VRAM budgeted for weight caching (default `0.85`) |
| `--host-memory 216G` | Host RAM ceiling for the resident weight arena |

### Terminal UI

`strata-tui` is a Ratatui frontend over the same runtime process: a launch form, streamed output, a throughput graph, prefill/decode telemetry, context usage, and exact-versus-sampled status.

```bash
make tui
./target/release/strata-tui
```

`F1` shows the keyboard map, `Ctrl+L` shows runtime diagnostics. Rust is only a dependency for this frontend, not for the runtime itself. Details in [docs/tui.md](docs/tui.md).

### HTTP server

`strata-server` exposes the same runtime over an OpenAI-compatible HTTP API. It is a thin layer on top of the C++ runtime, not a second inference path:

```bash
./build/strata-server --model models/glm52 --model-type glm \
  --model-id glm52 --context-size 2048 --devices 0,1,2

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm52","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

It serves `/v1/models`, `/v1/health`, `/v1/chat/completions`, `/v1/completions`, and `/v1/tokenize`, with streaming over SSE. An endpoint the loaded model has no exact implementation for returns an explicit error. Measured serving overhead is about 0.04% of a decode step.

## Models

| Model | Layout | Native precision | Status on this hardware |
|---|---|---|---|
| DeepSeek-V4-Flash-DSpark | 43 layers, 256 experts, top-6 | FP4 E2M1 experts, FP8 E4M3 spine, BF16/F32 | Resident in RAM; zero checkpoint reads during decode |
| GLM-5.2 | 78 layers, 256 experts, top-8 | INT4 group-128 experts, INT8 group-128 linears, BF16/F32 | Larger than combined memory; I/O-dependent |

Both run their declared attention and routing mechanisms as-is: hybrid compressed attention, manifold-constrained hyper-connections, and `sqrtsoftplus`/`noaux_tc` routing for DeepSeek; MLA-style projections, compressed KV, and sigmoid/`noaux_tc` top-8 routing for GLM.

Measured on the machine above:

| | DeepSeek-V4 | GLM-5.2 |
|---|---:|---:|
| Checkpoint size | 167 GB | 405 GB |
| Decode | ~4.0 tok/s | 0.283 tok/s |
| Checkpoint reads during decode | 0 | 910 GB/run |
| Load time | ~22 s | ~23 s |

DeepSeek's number is from an 18-token prompt, 152 generated tokens, three GPUs, 216 GiB host ceiling, `--flash-attention --pin-resident-arena`. GLM's is a median of three runs, 30-token prompt, 128 generated tokens. Neither number transfers to a different context length or prompt — see [docs/experiments/](docs/experiments/) for the full records and their operating points.

The difference between the two rows is mostly explained by whether the checkpoint fits in RAM. DeepSeek does, so decode after warm-up doesn't touch storage. GLM doesn't, so every decode step pays for storage traffic. Reducing that cost for checkpoints in GLM's position, without changing precision or routing, is the current research direction.

## How it's structured

- **Loading.** A content-addressed sidecar manifest references byte ranges
  inside the original Safetensors shards. The shards are opened read-only and
  are never duplicated or repacked.
- **Admission.** Before generation starts, the runtime computes a placement for
  every tensor — VRAM spine, host arena, or storage — and either commits to
  that plan or refuses to start. `--admission-only --json` prints the plan
  without loading the model.
- **Residency.** The dense/shared spine is pinned in VRAM. Routed experts live
  in a host RAM arena and are leased into VRAM per decode step through an LRU
  cache with a capacity-weighted schedule across GPUs.
- **Kernels.** Native INT4 group-128, INT8 group-128, FP4 E2M1, and FP8 E4M3
  CUDA kernels for compute capabilities 8.6 and 12.0, checked against a CPU
  reference implementation.
- **Instrumentation.** Every run reports checkpoint reads, H2D/D2H bytes, cache
  hits/misses/evictions, per-phase timings, RSS, and per-GPU VRAM as JSON.

For more detail: [docs/current-architecture.md](docs/current-architecture.md) describes what's implemented, [docs/architecture.md](docs/architecture.md) describes the target scheduler design, and [docs/deepseek-v4-runtime.md](docs/deepseek-v4-runtime.md) covers the DeepSeek contract specifically.

## Roadmap

Model adapters are kept narrow by design — an adapter owns its tokenizer, tensor roles, router semantics, and operations, and nothing outside that. Gemma, Qwen, and diffusion models are the intended next architectures once GLM-5.2 and DeepSeek-V4 clear their remaining correctness gates.

The main open research problem is making decode independent of storage for checkpoints like GLM's, where the model doesn't fit in the machine's combined memory. Work on custom quantization or pruning to shrink checkpoints further is also of interest, but it will be held to the same rule as everything else here: it has to clear a measured quality gate before it ships, and the four-bit floor is not something that moves to get there.

## Project discipline

Strata is run as a research project, not just an engine. Changes are expected to start from a measured bottleneck against the cost model in [research/moe-tiered-memory-decode-optimization.md](research/moe-tiered-memory-decode-optimization.md), not from a guess about what's slow. Results are reported as medians over interleaved repetitions, with the full operating point stated alongside the number. Rejected experiments are recorded with the same care as accepted ones in [docs/experiments/](docs/experiments/).

If you're contributing, [CLAUDE.md](CLAUDE.md) has the full rules. In short: state a hypothesis and a kill criterion before building, measure before optimizing, run `make check` before claiming a result, and don't call something a win if it's within run-to-run variance.

## Repository layout

```text
apps/             command-line tools and the Rust TUI
include/strata/   public C and C++ interfaces
src/              runtime, model adapters, checkpoint, scheduling
kernels/cpu/      numerical reference implementations
kernels/cuda/     CUDA backend
tests/            dependency-free correctness tests and fixtures
scripts/          reproducible benchmarks and determinism checks
docs/             architecture, contracts, and dated experiment records
```

Start with the [docs index](docs/README.md) — it distinguishes current behavior from target design, active plans, and historical evidence.

## License

Apache-2.0.
