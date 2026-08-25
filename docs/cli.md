# Command-line usage

Every CLI flag, the dry-run planner, samplers, the terminal UI and the HTTP
server. Extracted from the README, which is now a landing page.

For what Strata is and which models it runs, see the [README](../README.md).
For how the code is organised, see
[`current-architecture.md`](current-architecture.md).

## Quick start

You need a C++20 compiler, CMake 3.20+, Make, and CUDA 12.8 (or a compatible
toolchain) for the GPU backend. Gemma 4 image requests additionally use the
installed ImageMagick decoder for bounded PNG, JPEG, and WebP inputs.

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

#### Fastest DeepSeek V4 route: rank-local TP2

`--decode-topology rank-local-tp2` gives each of two GPUs rank-local ownership of
the decode chain and joins the layers with collectives instead of routing every
layer through the host. On the reference pair it is the fastest supported path.

It is opt-in and admitted fail-closed, because it needs all of:

- a build with NCCL (the default build has it **off**);
- exactly **two** CUDA devices;
- both devices at compute capability **8.6** — the DeepSeek device-resident
  kernels are validated only there, so a 4090, A100, H100, or 50-series device
  is refused rather than approximated ([#23](https://github.com/ro99/strata/issues/23));
- at least **two NUMA nodes**, one per rank;
- a context of **65,536 tokens or fewer**
  ([#22](https://github.com/ro99/strata/issues/22)).

The first four are checked before the checkpoint is opened, so an unusable
configuration costs a second rather than a model load, and Strata names the
condition that failed. The context bound is the exception: it is not checked at
admission today, so an over-large `--context-size` is accepted, loads, and then
fails at the first affected layer. Either way nothing silently falls back to a
slower or less exact path.

`-DSTRATA_ENABLE_NCCL=ON` locates NCCL itself. It looks in the active Python
environment (`nvidia-nccl-cu12`), `NCCL_HOME`, a tar install under
`/usr/local/nccl*` or `/opt/nccl*`, the distribution packages under `/usr`, and
the CUDA toolkit — printing the header and library it chose, both taken from the
same installation. If you have none of these, install NCCL by
[NVIDIA's guide](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html)
or `pip install nvidia-nccl-cu12`; to point at a specific copy, pass
`-DSTRATA_NCCL_INCLUDE_DIR=` and `-DSTRATA_NCCL_LIBRARY=` together.

```bash
cmake -S . -B build-nccl -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_NCCL=ON
cmake --build build-nccl --target strata-server -j

# Pin the device order: CUDA does not enumerate GPUs the way nvidia-smi does,
# so without this --devices can select different physical cards on the same box.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
./build-nccl/strata-server \
  --model models/dsv4f --model-type deepseek --model-id dsv4 \
  --devices 1,2 --context-size 4096 --vram-fraction 0.95 \
  --decode-topology rank-local-tp2 --port 8080

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

`--decode-topology rank-local-tp2` implies `--device-resident-runtime`, so the
physical KV pages, device-resident mHC, CUDA attention, and NUMA-local routed
experts come with it. Pass `--device-resident-runtime` alone for the same
contract on the centralized topology. Both flags work identically on
`strata-chat` and are rejected by every non-DeepSeek `--model-type`.

##### Prompt processing

The device-resident path processes the prompt a page at a time. A page runs
layer-major with one fused device mHC state per row, and its routed experts
execute **on the GPU**, reading the same transformed shards the host expert
kernels read — the routed experts are 156 GB and exist in host memory in one
layout, so the device reads that layout rather than requiring a second copy.
Decode is unchanged and keeps the experts in the two NUMA-local CPU shards,
which is where a one-row step wants them.

Physical attention is batched across all rows in a page: appends complete first
to satisfy the KV lease contract, then one page request covers every row for a
layer. On SM86, multi-row E4M3 block-128 query and KV projections decode FP8 in
registers and use BF16 tensor-core matrix multiplication without expanding the
persistent weights. Unsupported capabilities and single-row decode retain the
native kernel.

On the reference 2× RTX 3090 pair, three interleaved 677-token/page-8192 pairs
improved median total prefill from **81.24 s / 8.33 tok/s** to
**52.96 s / 12.78 tok/s**, a **1.53×** speedup. Physical-attention dispatches
fell from 29,111 to 86, kernel launches from 553,109 to 1,655, and page reads
from 6.99 GB to 30.56 MB. The query projection's device term fell from 7.21 s
to 0.334 s. The measured total-prefill ranges were 13.53 s for the
main-equivalent arm and 2.63 s for the candidate, so the median improvement and
every paired improvement were outside the observed spread.

Prefill and decode therefore use different expert kernels, as they do in every
CPU-GPU hybrid stack. The prefill kernel is held to the scalar oracle within
one BF16 mantissa step per output rather than to bit equality with the CPU
one; page-major execution itself remains bit-exact, and routes, precision,
top-k, expert count and mHC semantics are unchanged. See experiments
[0095](experiments/0095-dsv4-page-major-tp2-prefill-2026-08-15.md) and
[0096](experiments/0096-dsv4-gpu-prefill-experts-2026-08-15.md) for the
page-major and GPU-expert foundation,
[0105](experiments/0105-dsv4-sm86-fp8-tensor-page-projections-2026-08-16.md)
for the FP8 projection contract, and
[0107](experiments/0107-dsv4-promotion-campaign-2026-08-16.md) for the
promotion campaign.

Measured on 2× RTX 3090, an 18-token prompt and 31 decode steps including
warm-up:

| topology | ms/token | tok/s |
| --- | --- | --- |
| centralized | 133.080 | 7.51 |
| rank-local-tp2 | 112.494 | 8.89 |

Those are single-turn figures at a 4,096-token bound. Decode cost is a function
of context length, so a long multi-turn session is dominated by re-prefill
rather than by the step measured here.

Each topology is deterministic and exact against its own oracle, but the two are
**not token-identical to each other**: rank-local reduces in a different order,
so greedy decode can select a different token a dozen steps in. Switching
topology changes the text a request returns, not only its speed.

Gemma 4 uses its native W8A16 checkpoint directly and enables CUDA attention:

```bash
./build/strata-chat \
  --model models/gemma4 --model-type gemma4 \
  --context-size 2048 --max-new 256 --devices 0,1,2
```

Laguna S 2.1 runs its NVFP4 experts as shipped; the resident spine is pinned
first and the routed experts fill whatever VRAM is left:

```bash
./build/strata-chat \
  --model models/laguna-s-21 --model-type laguna \
  --context-size 2048 --max-new 256 --devices 0,1,2
```

Startup prints a banner with the checkpoint, the selected devices, the VRAM
budget, the attention path and the active sampler, then load progress and the
elapsed load time. Decoding defaults to greedy (`--temperature 0`) for
reproducible output. Multi-turn chat reuses the cached prefix and only prefills
new tokens.

### The interactive session

Once loaded, `strata-chat` prompts with `›`. The prompt is a real line editor:
history on `↑`/`↓`, `←`/`→` and `Home`/`End` to move, and the usual
`Ctrl+A`/`Ctrl+E`/`Ctrl+U`/`Ctrl+K`/`Ctrl+W` edits. It moves over whole
characters, so a prompt containing non-ASCII text edits correctly.

| Command | |
|---|---|
| `/help` | list the commands |
| `/clear` | forget the conversation so far (a `--system` message is kept) |
| `/regen` | generate the last answer again |
| `/stats` | throughput so far this session |
| `/save FILE` | write the transcript to `FILE` as Markdown |
| `/exit` | quit; `Ctrl+D` does the same |

`Ctrl+C` stops a generation in progress and keeps the partial answer, marking
the turn `[interrupted]`; on an empty prompt it quits. A second `Ctrl+C` during
the same generation exits immediately, which is the way out if a model's decode
loop does not honour cancellation.

#### Where the numbers are reported

Throughput is reported twice, and never as a live counter — a running figure
only ever describes the last few tokens.

After each answer, that turn:

```
  prefill 3,565 tok in 5.57 s (640.21 tok/s)   decode 128 tok in 20.29 s (6.31 tok/s)
```

and on exit, the session:

```
  session
    turns     3
    prefill   10,695 tok in 16.71 s   640.02 tok/s average
    decode    312 tok in 49.70 s   6.28 tok/s average
    load      51.32 s
```

Each average is that phase's total tokens over its total seconds, so a
two-token turn does not weigh as much as a two-hundred-token one. Prefill and
decode are never combined: a prompt token and a generated token do not cost the
same thing. When a turn reuses a cached prefix, the prompt tokens that were not
re-prefilled are reported separately rather than counted as prefill work.

#### Streams

**Only generated text goes to stdout.** The banner, load progress, the prompt,
slash-command output, per-turn and session figures, and every error go to
stderr. So this writes exactly the answer, with the chrome still on the
terminal:

```bash
./build/strata-chat --model models/gemma4 --model-type gemma4 \
  --prompt "Name three primary colors." > answer.txt
```

Colour is used only when stderr is a terminal, and is disabled by `--no-color`
or by setting `NO_COLOR`. With stdin redirected, `strata-chat` reads one prompt
per line and prints one answer per line, with no editor and no colour.

### Planning a load before making it (`--dry-run`)

Whether a model is fast usually comes down to one question that is invisible
until the weights are already loaded: does everything fit on the GPUs, or is
some class of weight being streamed on every decode step? `--dry-run` answers it
in about a second, without reading a single weight:

```bash
./build/strata-chat \
  --model models/gemma4 --model-type gemma4 \
  --context-size 8192 --devices 0,1,2 --dry-run
```

```
component               size     tier      cuda:0      cuda:1      cuda:2    per step
-------------------------------------------------------------------------------------
attention           8.39 GiB   device    3.36 GiB    3.49 GiB    1.55 GiB    8.39 GiB
feed-forward       20.59 GiB   device    8.24 GiB    8.58 GiB    3.78 GiB   20.59 GiB
norm                5.06 MiB   device    2.02 MiB    2.11 MiB  950.00 KiB    5.06 MiB
output-head         2.62 GiB   device           -           -    2.62 GiB    2.62 GiB
vision              1.03 GiB   device    1.03 GiB           -           -           -
kv-cache            1.41 GiB   device  576.00 MiB  592.00 MiB  272.00 MiB    1.41 GiB
workspace           2.25 GiB   device  768.00 MiB  768.00 MiB  768.00 MiB           -
-------------------------------------------------------------------------------------
device total       36.29 GiB            13.93 GiB   13.40 GiB    8.96 GiB
admitted budget                         19.81 GiB   19.81 GiB   13.04 GiB

  layer blocks  cuda:0=0..23, cuda:1=24..48, cuda:2=49..59
  hops          2 cross-device activation transfers per decode step
  decode reads  33.02 GiB device, 0 B host-to-device, 0 B nvme  (bytes per step, not a duration)
  max context   200599 tokens at this placement
  verdict       fits: every weight, cache, and workspace is device resident
```

The plan is then cached under `~/.cache/strata/plans` and the next real load
reuses it, so what runs is what the dry run printed. A plan is keyed by
checkpoint contents, GPU set, context size, and device list; change any of them
and it is recomputed automatically. `--replan` forces a fresh one,
`--no-plan-cache` neither reads nor writes one, and `--plan-cache DIR` (or
`$STRATA_PLAN_CACHE`) moves the directory.

Read the table as an instance of the cost model in
`research/moe-tiered-memory-decode-optimization.md`: the `per step` column is the
`W_r` volume each component contributes to one batch-1 decode step, split by the
resource that serves it. It is a byte count, not a duration — the planner
measures no bandwidth and converts nothing to milliseconds.

What each verdict means:

| Verdict | Meaning |
|---|---|
| `fits` | Every weight, cache, and workspace is device resident. Decode reads no PCIe and no NVMe. |
| `fits with a host tier` | Sparse classes are cached in VRAM over a host-resident copy. Decode streams the misses over PCIe. |
| `I/O dependent` | The resident set exceeds VRAM plus host RAM, so steady-state decode reads NVMe. Caching cannot manufacture sparsity in a dense model. |

Placement is prescriptive for Gemma 4: the plan chooses contiguous, byte-balanced
layer blocks sized to each GPU's admitted budget, and the load performs exactly
that assignment. For GLM and DeepSeek the plan is descriptive — it sizes and
admits the placement those runtimes already perform and reports it without
changing it, so a planning defect cannot regress a validated runtime.

`--dry-run` exits `0` when the configuration fits and `1` when it does not, so it
works as a preflight check in a script. `strata-server` takes the same flags.

The DeepSeek command above favors decode throughput: pinning adds about 17
seconds to startup on the development machine, then avoids pageable host-staging
stalls while loading expert weights into VRAM.

Flags worth knowing:

| Flag | Purpose |
|---|---|
| `--devices 0,1,2` | CUDA devices to use |
| `--context-size N` | Context ceiling enforced by the runtime |
| `--max-new N` | Tokens generated per turn (default `256`) |
| `--prompt TEXT` | Answer `TEXT` and exit instead of prompting |
| `--system TEXT` | Prepend a system message to the conversation |
| `--no-color` | Plain output even when stderr is a terminal (`NO_COLOR` does the same) |
| `--vram-fraction F` | Fraction of free VRAM budgeted for weight caching (default `0.85`) |
| `--host-memory 216G` | Host RAM ceiling for the resident weight arena |
| `--pin-resident-arena` | Page-lock DeepSeek's resident weights for faster host-to-device demand loads |
| `--flash-attention` | Use the exact CUDA attention fast path where it is faster |
| `--no-prepack-mhc` | Disable the default exact AVX2-packed mHC projection path |
| `--dry-run` | Size and place every component against this machine, print the plan, cache it, and exit without reading weights |
| `--replan` | Recompute and overwrite a cached placement plan |
| `--plan-cache DIR` | Directory for cached plans (default `~/.cache/strata/plans`) |
| `--no-plan-cache` | Neither read nor write a cached plan |
| `--temperature F` | Default sampling temperature when a request does not set one (default `1.0`) |
| `--top-p F` | Default nucleus cutoff when a request does not set one (default `1.0`) |
| `--seed N` | Default seed when a request does not set one |
| `--models-preset FILE` | Run as a model router over a catalog instead of loading one model — see [`server.md`](server.md) |
| `--models-max N` | Router: maximum concurrently loaded children (default `1`) |
| `--no-models-autoload` | Router: reject requests for unloaded models instead of autoloading |


## Elsewhere

Three subjects that used to live here have their own documents:

| | |
|---|---|
| [`sampling.md`](sampling.md) | sampler options, their exact semantics, reproducibility, future-entropy lookahead |
| [`server.md`](server.md) | `strata-server` and the OpenAI-compatible API |
| [`tui.md`](tui.md) | `strata-tui`, the Ratatui operator frontend |
