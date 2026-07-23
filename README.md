# Strata

Strata is a dependency-light C++20/CUDA inference engine for dense and sparse
mixture-of-experts models whose weights exceed local VRAM. It keeps the model
semantics exact while distributing work across GPU VRAM, host RAM, and
read-only NVMe-backed weight storage.

GLM-5.2 remains the current performance target. A separate executable adapter
now brings up the pinned `DeepSeek-V4-Flash-DSpark` base model without changing
the GLM execution path.

## Current status

The current GLM-5.2 path provides:

- a zero-rewrite Safetensors loader for the pinned QuantTrio checkpoint;
- native INT4 group-128, INT8 group-128, channelwise INT8, BF16, and FP32
  handling, preserving the checkpoint precision and tensor extents;
- exact tokenizer/chat rendering, RMSNorm, RoPE, compressed KV attention,
  causal attention, shared-expert execution, and sigmoid/`noaux_tc` top-8
  routing;
- a 78-layer autoregressive graph with multi-GPU expert dispatch;
- CUDA kernels for compute capabilities 8.6 and 12.0;
- separate prefill and decode throughput counters, checkpoint-read counters,
  H2D/D2H counters, and VRAM-cache statistics;
- a deterministic greedy-decode diagnostic for repeated fixed-input runs.

The current GLM-5.2 runtime is intentionally bounded to a maximum context of
2,048 tokens. MTP proposal acceleration is disabled in the current baseline,
and active DSA beyond that context, MTP verification, and optimized residency
scheduling remain future measured work.

The DeepSeek adapter validates all 48 native shards and executes the 43-layer
base graph with native FP4/FP8 weights, hybrid compressed attention, mHC,
hash/sqrt-softplus top-6 routing, and the target tokenizer. Its admission plan
stages the main-model routed experts and embedding into read-only RAM, keeps the
spine and an expert cache in VRAM, and enforces zero checkpoint reads during
decode. DeepSeek admission accepts logical context ceilings through the model's
declared 1,048,576-token limit. Compressed KV and learned-index rows are
committed in host pages as the sequence grows; ratio-4 layers switch to the
declared learned top-512 selection beyond 2,048 tokens. Prefill uses bounded
64-token layer-major pages by default, with page size one retaining the exact
token-at-a-time oracle. Short full-model page equivalence passes; production-
scale 32k/200k/1m ingestion is not yet claimed. DSpark tensors are validated but
speculative execution is rejected until target verification and provisional-
state rollback have frozen oracles.

The latest correctness and determinism gates pass:

```text
make check                         2/2 tests passed
full 128-token greedy repetition  3/3 token sequences identical
```

The determinism fix synchronizes newly uploaded weights on the same CUDA
execution stream used by matmul. Before the fix, a newly loaded expert could be
read before its default-stream transfer had completed.

## GLM-5.2 baseline

This is the current post-fix machine baseline. It uses the exact prompt below,
30 prompt tokens, 128 requested new tokens, greedy autoregressive decoding,
MTP disabled, all three GPUs, and a 0.85 VRAM-cache fraction.

| Run | Prefill tok/s | Decode tok/s |
|---:|---:|---:|
| 1 | 0.716 | 0.267 |
| 2 | 0.619 | 0.283 |
| 3 | 0.657 | 0.305 |
| **Median** | **0.657** | **0.283** |

Additional measurements from the same three runs:

| Measurement | Value |
|---|---:|
| Initialization/load median | 22.52 s |
| Requested checkpoint reads | 910.2 GB per run |
| Weight H2D traffic | 910.2 GB per run |
| VRAM cache hits / misses / evictions | 200,396 / 140,292 / 134,570 |
| Peak weight cache | 13.0 + 19.8 + 19.8 GiB |
| Generated sequence agreement | 3/3 identical |

This is an I/O-dependent baseline: the checkpoint is larger than the machine’s
combined resident memory budget, so caching does not remove the storage path.
The validation runs above did not collect block-device physical-read counters.
The earlier instrumented three-GPU baseline measured a median of 163.1 GB of
physical NVMe reads; rerun the physical-I/O benchmark after any storage or
residency change before comparing that metric.

### Baseline contract

Future performance results must keep these variables equal unless the result is
explicitly labeled as a different experiment:

- checkpoint revision, tensor precision, router semantics, expert count, and
  top-k;
- prompt, chat template, generation length, greedy policy, and MTP setting;
- GPU devices, VRAM fraction, host-memory limit, topology, and power settings;
- warm-up/admission treatment and measurement definitions.

The primary metric is median decode tok/s. Report prefill tok/s, initialization,
requested and physical NVMe bytes, H2D/D2H traffic, cache hits/misses/evictions,
RSS, and per-GPU VRAM with every performance result. Report at least three
interleaved repetitions and do not call a result a win when it is within the
observed run variance.

## DeepSeek-V4 screened baseline

The DeepSeek-V4 base executor is the `main` baseline for subsequent DeepSeek
experiments. On the admitted three-GPU topology (216 GiB host-memory ceiling,
0.85 VRAM fraction, exact device MoE, FP4 switch, and 28 exact host-attention
workers), one matched 128-token `Hello` screen completed 127 decode steps in
29.00231959 seconds, or **4.3789600899 decode steps/s**. Its matched serial
attention reference reached 3.0227143404 steps/s, a 1.4486847240x speedup.

Tokens, sequential routes, logits, and every recorded layer BF16 hash matched;
decode checkpoint reads were zero, cache leases were balanced, and the measured
RSS/VRAM stayed within admission. This is the working comparison baseline, not
a three-repetition median or an external target-model validation claim. Future
DeepSeek experiments may land only when they preserve this exact contract and
demonstrate a material, properly replicated improvement against it. The frozen
target-layer, teacher-forcing, greedy-generation, and independent physical-I/O
promotion gates remain open.

The default DeepSeek loader now stages tensors directly into the resident arena
with eight shard readers, uses block-wise FP8 lookup conversion, warms the GPU
spine with up to three device workers, and overlaps that warm-up with resident
staging. On the same 216 GiB topology, the cumulative single-pair loading
screens reduced initialization from approximately 191 seconds to 22.177 seconds.
Use `--serial-resident-warmup` only to diagnose or compare the non-overlapped
path. See
[`docs/experiments/0008-dsv4-fast-loading-promotion-2026-07-16.md`](docs/experiments/0008-dsv4-fast-loading-promotion-2026-07-16.md)
for the experiment lineage and matched accounting gates.

## Build and test

Requirements are a C++20 compiler, CMake, Make, and optionally CUDA 12.8 or a
compatible CUDA toolchain for the native backend.

```bash
make check
```

## Interactive terminal chat

The recommended interactive experience is `strata-tui`, a responsive
[Ratatui](https://github.com/ratatui/ratatui) cockpit for launching and chatting
with either runtime. It provides a guided launch form, streamed output,
prefill/decode telemetry, a live throughput graph, context usage, prompt
history, runtime diagnostics, and explicit exact-versus-sampled contract
status.

Build and open the launch form:

```bash
make tui
./target/release/strata-tui
```

Or launch a configured session immediately:

```bash
./target/release/strata-tui \
  --model models/DeepSeek-V4-Flash-DSpark \
  --model-type deepseek --context-size 8192 --max-new 256 \
  --devices 0,1,2
```

The TUI keeps one long-lived `strata-chat` process and communicates through a
versioned JSON-lines protocol. It does not implement model behavior: precision,
routing, top-k, sampling, admission, and exact-mode failures remain owned by the
C++ runtime. Press `F1` for the keyboard map, `Ctrl+L` for runtime diagnostics,
and `Ctrl+Q` to quit. See [the TUI operator guide](docs/tui.md) for the complete
interface and protocol contract.

`strata-chat` remains available as the minimal streaming terminal interface. It
continuously reports decoded tok/s in the terminal title without moving the
answer cursor. Select the runtime and context ceiling at launch; CUDA devices
default to `0,1,2`:

```bash
./build/strata-chat --model models/glm52 --model-type glm \
  --context-size 2048 --max-new 512 --devices 0,1,2

./build/strata-chat --model models/DeepSeek-V4-Flash-DSpark \
  --model-type deepseek --context-size 8192 --max-new 256 --devices 0,1,2
```

The interactive form reads one prompt per line. For a single streamed request,
add `--prompt 'Explain sparse attention.'`. `--context-size` is enforced by
the selected runtime; the prompt plus `--max-new` must fit within it. Startup
prints the selected devices, VRAM budgets, model-loading progress, and elapsed
load time. Chat defaults to greedy decoding (`--temperature 0`) for deterministic
output. Pass `--temperature 1` to enable seeded Gumbel-max sampling. Runtime and
benchmark APIs remain greedy by default, so sampling is never enabled silently.

Multi-turn chat reuses an exact cached token prefix and prefills only newly
appended tokens. A changed or truncated history resets the cache and takes the
full-prefill fallback. `--full-reprefill` forces that baseline for validation;
DeepSeek chat also accepts `--block-kv-cache`. JSON `turn_done` events report
full, processed, and reused prompt-token counts.

`--vram-fraction` controls the fraction of each GPU's free memory budgeted for
weight caching (default 0.85, valid range `(0, 0.95]`). Raise it (e.g., `0.95`)
to keep more experts resident when free VRAM allows.

For a sanitizer build:

```bash
cmake -S . -B build-asan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DSTRATA_ENABLE_SANITIZERS=ON
cmake --build build-asan --parallel
ctest --test-dir build-asan --output-on-failure
```

## Run GLM-5.2

Place the pinned checkpoint at `models/glm52`. The loader reads the original
Safetensors shards and does not create a converted model copy.

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON
cmake --build build --parallel

./build/strata-run \
  --model models/glm52 \
  --devices 0,1,2 \
  --prompt 'What is the closer start to sun, and how distant it is from it?' \
  --max-context 256 \
  --max-new 128 \
  --json
```

For the complete physical-I/O benchmark, use the reusable script in a named
tmux session:

```bash
tmux new-session -d -s strata-glm52-baseline \
  './scripts/run_glm52_baseline.sh'
```

The script defaults to three repetitions and writes ignored artifacts under
`results/glm52-baseline/`. It records system telemetry, block-device reads and
writes, checkpoint reads, transfers, CUDA calls, and cache occupancy.

To check greedy determinism without collecting block-device telemetry:

```bash
tmux new-session -d -s strata-glm52-determinism \
  './scripts/check_glm52_determinism.sh'
```

The script reports the first divergent token. For opt-in layer and expert state
hashes during diagnosis:

```bash
DIAGNOSTIC_TRACE=1 scripts/check_glm52_determinism.sh
```

## Run DeepSeek-V4-Flash-DSpark

The exact resident topology can be checked without staging the model:

```bash
./build/strata-deepseek-run \
  --model models/DeepSeek-V4-Flash-DSpark \
  --devices 0,1,2 --host-memory 216G --max-context 2048 \
  --admission-only --json
```

To execute the base model:

```bash
./build/strata-deepseek-run \
  --model models/DeepSeek-V4-Flash-DSpark \
  --devices 0,1,2 --host-memory 216G --max-context 2048 \
  --prompt 'Hello' --max-new 16 --json
```

`--max-context` is the per-runtime logical cache ceiling and accepts any value
from 1 through 1,048,576. This is an admission/storage limit, not a claim that
prefill is practical or fully validated at every ceiling. `--prefill-page-tokens`
selects a bounded page in `[1,512]` (default 64); one selects the original
token-at-a-time oracle. The rendered prompt plus `--max-new` must fit that
ceiling. Use `--admission-only --json` first to inspect `kv_state_bytes`,
`index_state_bytes`, and total host admission for large contexts such as 32,768
or 200,000 tokens.

Exact device MoE and 28 host-attention workers are the defaults. Use
`--serial-device-moe` and `--serial-host-attention` only for controlled
reference or diagnostic runs.

The generic tiled online-softmax CUDA backend is available as an opt-in
correctness/performance candidate. Add `--flash-attention` to either model
runner; `--scalar-attention` pins the existing oracle. DeepSeek uses its
parallel scalar kernel below the measured 256-row crossover and CUDA for larger
KV histories; `--flash-attention-minimum-rows 0` forces CUDA for diagnostics.
The production kernel passes full-model trace gates and improves measured
640-row attention and long prefill. The replicated gate shows no end-to-end
decode win within observed variance, so the scalar path remains the default;
`--flash-attention` enables the shape-aware hybrid policy.

Opt-in `--block-kv-cache` stores KV and learned-index rows in their native FP8 and
FP4 encodings with versioned block headers and corruption checks, reducing per-token
cache bytes without numerical change. `--scalar-kv-cache` (default) retains the F32
oracle. `--gpu-lightning-indexer` enables the independently gated CUDA learned-index
path and implies `--block-kv-cache`; it applies query Hadamard/FP4 simulation,
weighted-ReLU scoring, and deterministic top-512 selection in a bounded workspace.
Compact blocks remain device-resident when `--kv-device-cache` budgets permit, and
only selected positions return to the host. The scalar indexer remains the default.

See [`docs/deepseek-v4-runtime.md`](docs/deepseek-v4-runtime.md) for the pinned
checkpoint contract, measured admission plan, design boundary, and remaining
oracle gates.

## Model and precision contract

The pinned GLM-5.2 checkpoint contains 177,569 indexed tensors in 128
Safetensors shards. Strata preserves its native representation:

- routed experts: symmetric INT4, group size 128;
- ordinary quantized linears: symmetric INT8, group size 128;
- MTP modules: channelwise INT8;
- sensitive tensors: BF16 or FP32.

No weight representation, cache, predictor, draft, or storage codec may use
less than four bits. Precision, routing semantics, expert count, and top-k may
not change silently. Prediction is advisory and may affect prefetch or
scheduling only.

Exact mode either completes exact work or reports a failure. It must not hide a
CPU fallback, drop experts, reduce top-k, or manufacture sparsity through
caching. Dense models larger than aggregate resident memory must be reported as
I/O-dependent.

## Repository layout

```text
apps/             command-line tools
include/strata/   public C and C++ interfaces
src/              runtime, model, checkpoint, and scheduling code
kernels/cpu/      numerical references and CPU kernels
kernels/cuda/     optional CUDA backend
tests/            dependency-free correctness tests and fixtures
scripts/          reproducible benchmark and determinism checks
docs/             architecture, formats, roadmap, and experiment records
```

The residency simulator can replay sequential expert routes against fixed RAM,
VRAM, peer, and NVMe budgets:

```bash
./build/strata-sim \
  --trace tests/data/route_trace_v1.txt \
  --compare \
  --ram 180G \
  --vram 58G \
  --expert-bytes 19M \
  --prefetch 8 \
  --lease 16
```

See the [`docs` index](docs/README.md) for the distinction between current-state,
target-design, planning, and historical evidence documents. Detailed entry points
include [`docs/model-bringup.md`](docs/model-bringup.md),
[`docs/current-architecture.md`](docs/current-architecture.md),
[`docs/architecture.md`](docs/architecture.md), and
[`docs/research-roadmap.md`](docs/research-roadmap.md) for detailed contracts
and future experiments. The original staged, one-agent-at-a-time GLM throughput
plan is preserved in
[`docs/glm52-throughput-handoffs.md`](docs/glm52-throughput-handoffs.md).

## Engineering rules

- Runtime code is C/C++; do not add Python or a framework runtime.
- Keep model files, raw traces, profiler captures, generated binaries, and
  large logs out of Git.
- Preserve unrelated worktree changes.
- Run `make check` before result commits.
- Use reversible, single-purpose branches and commits.

## License

Apache-2.0.
