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

On this hardware Strata supports two checkpoints: DeepSeek-V4-Flash-0731 (167 GB, 43 layers, 256 experts, top-6), staged into host RAM so decode does not touch storage after warm-up, and GLM-5.2 (405 GB, 78 layers, 256 experts, top-8), which is larger than the machine's combined VRAM and RAM and therefore runs directly against the checkpoint on disk.

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
  --model models/dsv4f --model-type deepseek \
  --context-size 8192 --max-new 256 --devices 0,1,2 \
  --vram-fraction 0.95 --pin-resident-arena --flash-attention
```

Startup prints the selected devices, the VRAM budget per device, load progress, elapsed load time, and the active sampler. Decoding defaults to greedy (`--temperature 0`) for reproducible output. Multi-turn chat reuses the cached prefix and only prefills new tokens.

The DeepSeek command above favors decode throughput: pinning adds about 17
seconds to startup on the development machine, then avoids pageable host-staging
stalls while loading expert weights into VRAM.

Flags worth knowing:

| Flag | Purpose |
|---|---|
| `--devices 0,1,2` | CUDA devices to use |
| `--context-size N` | Context ceiling enforced by the runtime |
| `--vram-fraction F` | Fraction of free VRAM budgeted for weight caching (default `0.85`) |
| `--host-memory 216G` | Host RAM ceiling for the resident weight arena |
| `--pin-resident-arena` | Page-lock DeepSeek's resident weights for faster host-to-device demand loads |
| `--flash-attention` | Use the exact CUDA attention fast path where it is faster |
| `--no-prepack-mhc` | Disable the default exact AVX2-packed mHC projection path |

### Samplers

The sampler runs as a fixed pipeline. Penalties rewrite the logits, every
truncation stage then reads the model's own distribution, and temperature
rescales only the surviving candidates before a seeded Gumbel-max draw:

```
presence/frequency/repetition → DRY → n-gram ban → logit bias
  → top_k → top_p → min_p → typical_p → XTC → future entropy
  → temperature → draw
```

Truncating on the natural distribution is what makes the thresholds mean what
they say: `--min-p 0.05` is "at least 5% as likely as the best token according
to the model" at any temperature.

**Just want temperature?** Pass `--temperature` with no `--preset` and nothing
else runs — no truncation, no penalties, no XTC:

```bash
./build/strata-chat --model models/dsv4f --model-type deepseek --temperature 0.9
```

The startup banner echoes exactly what's active, e.g. `[sampler]
temperature=0.9`; if other stages were silently on, they'd show there too. In
the TUI, this means leaving the `SAMPLER` field on `PRECISE` — it writes no
stages of its own — and editing `TEMPERATURE` directly; an edited temperature
always overrides whatever the preset last wrote there.

Four presets bundle the stages that are otherwise on separate flags:

| Preset | Stages | What to expect |
|---|---|---|
| `--preset precise` | none (temperature only, default `0`) | Deterministic at temperature 0; with temperature raised, plain softmax sampling over the full vocabulary — the most expensive path per token, ~4.6 ms/token at DeepSeek's vocabulary, and the one most prone to picking an implausible tail token at high temperature since nothing truncates it |
| `--preset balanced` | `min_p 0.05`, `repetition_penalty 1.05` over the last 256 tokens | Cuts tokens under 5% as likely as the best one, so the tail can't get picked; mild pushback on restating the same tokens. Closest to "temperature sampling, but safe" |
| `--preset creative` | `min_p 0.02`, XTC 50% at threshold 0.1, DRY, `repetition_penalty 1.03` over the last 512 tokens | Half the time, removes whichever safe/obvious token was in reach and forces a still-plausible alternative instead — the mechanism aimed at "goes to the mean" prose. DRY leans against the loops that removing the safe choice tends to invite. Expect more variance run to run; verify on your own material before trusting it unsupervised |
| `--preset future-entropy` | `min_p 0.05`, 20-candidate lookahead over the top-30 future, `alpha 0`, DRY, `repetition_penalty 1.03` over the last 512 tokens | Scores each candidate by how much future choice it unlocks. **Costs 21 forward passes per token**, so it decodes roughly 21× slower than the same settings without it — this is a quality knob, not a throughput one |

A preset writes defaults; any flag after it overrides them — including
`--temperature`, which is how you'd run e.g. `--preset creative --temperature
0.7` to keep XTC/DRY but sample less aggressively.

| Flag | Purpose |
|---|---|
| `--temperature F` | Rescales survivors before the draw; `0` is greedy |
| `--top-k N` | Keep the `N` most likely tokens |
| `--top-p F` | Keep the smallest set carrying mass `F` |
| `--min-p F` | Keep tokens at least `F` times as likely as the best one |
| `--typical-p F` | Keep the tokens whose surprisal is nearest the distribution's entropy |
| `--xtc-probability F` `--xtc-threshold F` | With probability `F`, drop every candidate above the threshold except the least likely of them |
| `--presence-penalty F` `--frequency-penalty F` | Subtractive repetition penalties |
| `--repetition-penalty F` `--penalty-window N` | Multiplicative penalty, optionally bounded to the last `N` tokens |
| `--dry-multiplier F` `--dry-base F` `--dry-allowed-length N` `--dry-window N` | Penalize the token extending the longest repeated suffix |
| `--no-repeat-ngram N` | Hard ban on completing an `N`-gram already in the output |
| `--future-entropy N` `--future-entropy-top-n N` | Look ahead one step past the `N` likeliest survivors and score them by the entropy of the top-`n` future |
| `--alpha F` `--future-entropy-curve NAME` | Crossfade between probability and future entropy; which exponent mapping to use |
| `--alpha-wave-amplitude F` `--alpha-wave-period F` | Oscillate alpha over the generated tokens |

XTC is the stage aimed squarely at flat prose: it removes the safe continuation
and leaves a plausible one in its place. It draws from the generator even at
temperature zero, so a run using it is seeded-reproducible but not greedy, and
the startup banner reports it as sampled rather than exact.

#### Future entropy

Prefers tokens that keep the next step's options open, after
[Count Bayesie, *Making LLMs better at creative writing using
entropy*](https://www.countbayesie.com/blog/2026/7/1/making-llms-better-at-creative-writing-using-entropy).
The model is run one step past each surviving candidate `w` to get
`q_w = p(V | c + w)`; the normalized entropy of that distribution's top-`n`,
`H(w) = [-Σ q̃ ln q̃] / ln n ∈ [0, 1]`, reweights the candidate:

```
s(w) = p(w | c)^a · H(w)^b
```

and the draw is made from `s` renormalized over the candidates, so
`--temperature` rescales the blended score rather than the raw probability.
`--alpha` sets both exponents at once: `-1` is ordinary sampling and the stage
becomes a no-op, `+1` scores on the future alone. Two mappings are available
because the article and the reference implementation disagree between the
endpoints — `--future-entropy-curve article` (default) uses
`a = 1 − max(0, α)`, `b = 1 − max(0, −α)`, so `α = 0` is exactly the article's
headline `s = p · H`; `crossfade` uses `a = 1 − t`, `b = t` for `t = (α+1)/2`,
so `α = 0` is `√(p · H)`. They agree at `α = ±1` and nowhere else.

Two things are load-bearing:

- **It costs forward passes.** One per candidate, sequentially, because the KV
  cache holds a single sequence — `--future-entropy 20` makes a token 21 decode
  steps instead of one. Every other stage in the pipeline is arithmetic on
  logits that are already in hand; this one is not. The batched form the
  reference implementation uses needs `k` forked sequences decoded at one
  position, which the GLM cache cannot express today.
- **A relative-plausibility cut in front of it is not optional.** Broken
  word-fragments have maximally uncertain futures, so entropy selects them
  unless something has already removed them. Keep `--min-p` at 0.05 or above;
  the preset does. The lookahead runs last, on the survivors only, which is
  also what keeps the cost proportional to what the cheaper stages accepted.

The lookahead is exact: each speculative pass is rolled back to a
bit-identical cache before the next candidate runs, so the emitted token is
decoded from the same state it would have been without the stage. It draws
nothing from the generator, so `--temperature 0` with future entropy is still
reported as exact greedy decoding — the argmax of `s` rather than of `p`.

Every knob is also accepted by the OpenAI-compatible server, under the same
names, on both `/v1/chat/completions` and `/v1/completions`.

Reported `logprobs` are the model's natural log probabilities, computed from the
unmodified logits before penalties, truncation, and temperature. They describe
the model rather than the sampler settings, so they stay comparable across
requests that used different knobs.

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
| DeepSeek-V4-Flash-0731 | 43 layers, 256 experts, top-6 | FP4 E2M1 experts, FP8 E4M3 spine, BF16/F32 | Resident in RAM; zero checkpoint reads during decode |
| GLM-5.2 | 78 layers, 256 experts, top-8 | INT4 group-128 experts, INT8 group-128 linears, BF16/F32 | Larger than combined memory; I/O-dependent |

Both run their declared attention and routing mechanisms as-is: hybrid compressed attention, manifold-constrained hyper-connections, and `sqrtsoftplus`/`noaux_tc` routing for DeepSeek; MLA-style projections, compressed KV, and sigmoid/`noaux_tc` top-8 routing for GLM.

The current 0731 checkpoint has not yet been benchmarked. For context, the last
validated DeepSeek measurement used the now-unsupported preview checkpoint:

| | DeepSeek V4 preview (historical) | GLM-5.2 |
|---|---:|---:|
| Checkpoint size | 167 GB | 405 GB |
| Decode | ~4.0 tok/s | 0.283 tok/s |
| Checkpoint reads during decode | 0 | 910 GB/run |
| Load time | ~22 s | ~23 s |

The historical DeepSeek number is from an 18-token prompt, 152 generated tokens,
three GPUs, 216 GiB host ceiling, `--flash-attention --pin-resident-arena`; it
must not be attributed to 0731. GLM's is a median of three runs, 30-token prompt,
128 generated tokens. Neither number transfers to a different context length or
prompt — see [docs/experiments/](docs/experiments/) for the full records and
their operating points.

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
