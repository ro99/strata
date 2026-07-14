# Strata

Strata is a dependency-light C++20/CUDA inference engine for dense and sparse
mixture-of-experts models whose weights exceed local VRAM. It keeps the model
semantics exact while distributing work across GPU VRAM, host RAM, and
read-only NVMe-backed weight storage.

GLM-5.2 is the current runtime and performance target. The repository also
contains the model-format, routing, residency, and simulation foundations for
future architectures; those paths are not part of the current GLM throughput
baseline.

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

The current runtime is intentionally bounded to a maximum context of 2,048
tokens. MTP proposal acceleration is disabled in the current baseline, and
active DSA beyond that context, MTP verification, and optimized residency
scheduling remain future measured work.

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

The baseline artifacts are retained under
[`results/glm52-fix-full`](results/glm52-fix-full/), with the aggregate result
in [`summary.json`](results/glm52-fix-full/summary.json). The source of the
numbers above is the three JSON run files in that directory.

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

## Build and test

Requirements are a C++20 compiler, CMake, Make, and optionally CUDA 12.8 or a
compatible CUDA toolchain for the native backend.

```bash
make check
```

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
  'cd /home/rodrigo/Developer/strata && scripts/run_glm52_baseline.sh'
```

The script defaults to three repetitions and writes ignored artifacts under
`results/glm52-baseline/`. It records system telemetry, block-device reads and
writes, checkpoint reads, transfers, CUDA calls, and cache occupancy.

To check greedy determinism without collecting block-device telemetry:

```bash
tmux new-session -d -s strata-glm52-determinism \
  'cd /home/rodrigo/Developer/strata && scripts/check_glm52_determinism.sh'
```

The script reports the first divergent token. For opt-in layer and expert state
hashes during diagnosis:

```bash
DIAGNOSTIC_TRACE=1 scripts/check_glm52_determinism.sh
```

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

See [`docs/model-bringup.md`](docs/model-bringup.md),
[`docs/architecture.md`](docs/architecture.md), and
[`docs/research-roadmap.md`](docs/research-roadmap.md) for detailed contracts
and future experiments.

## Engineering rules

- Runtime code is C/C++; do not add Python or a framework runtime.
- Keep model files, raw traces, profiler captures, generated binaries, and
  large logs out of Git.
- Preserve unrelated worktree changes.
- Run `make check` before result commits.
- Use reversible, single-purpose branches and commits.

## License

Apache-2.0.
