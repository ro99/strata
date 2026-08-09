# Experiment 0064: DeepSeek reference compressor/selection/attention layer gate

## Status

**Accepted: the real-context reference self-check and dependency-light C++
layer gate both pass.**  The first launcher paired a 1,024-block KV
allocation with a 4,096-token ceiling and failed admission.  Reducing both the
sequence and batch ceilings to 2,304 passed admission but made the installed
`lk_moe` GPU-prefill warm-up issue an illegal memory access.  The gate now uses
the reference's already proven 32,768 / 6,144-block / 8,192-batched-token
geometry.  That configuration started successfully and reached the target
decode position, where the hook failed because one installed helper was not
re-exported from `common.ops`.  The hook now imports all helpers from their
defining modules, verified against the installed environment.  The corrected
capture produced all three fixtures.  Post-quantization compressor replay is
exact, both learned top-512 sets are exact, and reference plus C++ sparse
attention replay remain within the declared contract.  This accepts the layer
arithmetic prerequisite; it makes no inference-throughput claim.

## Hypothesis and operating-point model

The bounded hypothesis is that a reference-compatible realization of the
smallest dependent compressor → paged FP8 KV → learned selection → sparse MLA
attention boundary can reproduce Lvllmds4-x at real early, middle, and late
layers.  Passing that correctness gate is a prerequisite for replacing
Strata's serialized host-owned attention/KV path with the complete dependent
device chain.  An isolated projection or insertion kernel is not the
experiment.

The current measured model remains:

```text
tau = max_r W_r / B_r + sum_serial

current total median                       244.812 ms/token
complete MoE                              114.667 ms
routed CPU median / best            106.869 / 89.226 ms
standalone routed CPU                       87.879 ms
maximum remaining median CPU opportunity    18.990 ms
serialized non-MoE / handoff path          130.145 ms
attention                                   71.140 ms
mHC                                         53.061 ms
target                                   about 98--100 ms/token
```

- **Target term and bottleneck.**  The target is the 130.145 ms serial
  non-MoE/handoff term.  No new CPU mismatch exists, so routed CPU scheduling is
  not reopened.
- **Primary metric.**  The first differing stage among main compressor row,
  index compressor row, exact selected-position set, and attention output at
  layers 2, 21, and 42.
- **Correctness gate.**  Main decoded cache error at most 0.0625; index decoded
  cache error at most 0.03125; exact learned top-512 set; attention maximum and
  mean absolute error at most 0.0625 and 0.002; every value finite.  Exact mode
  has no FP4 or host fallback.
- **Memory ceiling.**  Host RSS remains under 216 GiB and each GPU under its
  explicit 0.87 server admission for this correctness capture and 0.95 for the
  eventual Strata arm.  The complete 32,768-token production KV/state contract
  remains 151,228,416 bytes; no precision change, host KV tier, or NVMe
  dependency is admitted.
- **Resource signs.**  A successful future integration must reduce host KV
  gathering, H2D/D2H, blocking submissions, and ownership handoffs.  It adds
  persistent HBM and SM work, leaves CPU routed bytes and semantics unchanged,
  and adds no steady-state checkpoint reads.  This trace tooling changes no
  runtime resource term.
- **Rollback condition.**  Stop at the first cache-byte, selection, attention,
  finite-value, memory-admission, or exact-mode failure.  Do not build the next
  stage to rescue a negative prerequisite.

## Cheapest meaningful fixture

The gate uses one teacher-forced 2,175-token prompt and traces the first decode
forward at position 2,175 on tensor-parallel rank 0:

| Layer | Role | Compression | Why selected |
|---:|---|---:|---|
| 2 | early | 4 | first learned-index layer |
| 21 | middle | 128 | deterministic compressed-attention family |
| 42 | late | 4 | late learned-index layer |

`2,176` is divisible by both 4 and 128, so every selected compressor emits a
row in the same forward.  Ratio-4 has `2,176 / 4 = 544` compressed candidates,
which forces an actual top-512 choice.  A 20-token or 512-token prompt was
rejected because it would test only the trivial all-candidates selection path.

The first real launch measured 135.793 seconds of fixed model load.  After
admission, budget roughly 15--60 seconds for prefill depending on eager-path
overhead and one decode forward.  The setup-to-measured-window ratio is
intentionally high
because real hidden history and production cache metadata are the mechanism
under test; the source-only audit was performed first and cannot supply those
tensors.

## Source audit, including vllm.cpp

The installed Lvllmds4-x source establishes:

- ratio-4 uses an eight-row overlapping FP32 compressor window; ratio-128 uses
  128 rows without overlap;
- compressor scores receive APE before a per-dimension softmax, weighted KV
  sum, FP32 RMSNorm, GPT-J RoPE, and production page quantization;
- the Ampere index path is FP8.  Its query scale is folded into the 64 head
  weights, while each 128-wide key retains one FP32 scale;
- index logits sum the weighted positive part of each head's query/key dot and
  select 512 compressed positions;
- sparse attention combines those compressed positions with the recent
  128-token SWA region and includes a per-head sink in the softmax denominator.

The newly available `/home/rodrigo/Developer/vllm.cpp` independently agrees
with the visible compressor, weighted-ReLU index logits, top-k, sink-softmax,
and output-reduction formulas.  It is a useful readable source witness, not a
production oracle: its own `deepseek_v4.cpp` says real-GGUF resident decode uses
dense MLA for short contexts and that full real-geometry DSA is still a named
residual.  Its tiny-config host composition therefore cannot replace the real
Lvllmds4-x fixture.

## Retained tooling

- `scripts/dsv4_reference_layer_trace/sitecustomize.py` injects an opt-in eager
  rank-0 trace without editing the installed external package.
- `scripts/run_dsv4_reference_layer_gate_server.sh` launches TP2 with the
  reference's proven 32,768-context, 6,144-block, 8,192-batched-token geometry
  under the explicit 0.87 admission.  It writes `server.failed` if vLLM exits
  before serving.
- `scripts/query_dsv4_reference_layer_gate.py` submits exactly 2,175 token IDs,
  forces two generated tokens, and verifies all three trace files exist.
- `scripts/check_dsv4_reference_layer_trace.py` independently decodes the raw
  physical page planes and runs the declared stage gates.

The tracer saves allocator-independent state windows and only the physical
cache blocks needed by the selected request.  It does not save an allocator-
sized cache.  In particular, it never treats `cache[block,row]` as a physical
584- or 132-byte row; production pages place all token data before the page's
scale plane.

Python compilation, shell parsing, import-hook injection, and synthetic
584/132-byte physical page decodes pass in the external environment.

## First real-GPU admission result

The two GPU workers loaded the model and reported:

```text
Model loading took 6.06 GiB memory and 135.793081 seconds
Available KV cache memory: 10.12 GiB
Overriding num_gpu_blocks=10455 with num_gpu_blocks_override=1024
ValueError: max seq len 4096 needs 1.65 GiB KV cache;
            1024 blocks provide 0.99 GiB, estimated max length 2452
```

The query appeared to hang because its health loop waited while the failed
server could never become healthy.  A second launch with both ceilings set to
2,304 passed KV admission, but startup sparse-MLA warm-up ran a 2,304-token
dummy forward through `lk_moe`'s GPU-prefill kernel and failed at
`moe_v2_gpu_prefill.cu:81` with an illegal memory access.  The reference's own
successful launcher instead uses 32,768 model tokens, 6,144 blocks and 8,192
batched tokens.  The trace launcher now preserves that proven geometry and the
query watches a server-failure sentinel in addition to its 300-second
readiness and 900-second request bounds.  No real request, prefill, decode,
tensor capture, or correctness comparison ran.  This is neither a pass nor a
falsification, and no live runtime code was added.

The next reference-geometry run completed admission and warm-up, opened the API
and began the traced request.  At decode position 2,175 the hook raised:

```text
ImportError: cannot import name 'build_combined_sparse_mla_decode_valid_mask'
            from 'vllm.models.deepseek_v4.common.ops'
```

The function is defined in
`vllm.v1.attention.backends.mla.sparse_mla_kernels`; the other two helpers are
defined in `common.ops.cache_utils`.  Direct imports from those exact installed
modules now pass.  No partial trace file was written, so the arithmetic gate is
still unrun at that point.

## Accepted reference result

The corrected reference-geometry run loaded in 132.653 seconds, admitted 6,144
GPU KV blocks (65,006 tokens, 1.98 target-length requests), and served the
2,175-token prompt plus two generated tokens successfully.  The request took
14.999 seconds including prefill, first-shape JIT work, tracing, and two decode
steps; it is not a decode-throughput measurement.  The server reported 217.5
prompt tok/s and 0.5% KV-cache use for the request.

The first checker invocation incorrectly compared the stored FP8/BF16 cache
against the unquantized FP32 compressor output.  That measured normal
quantization error and produced an invalid negative.  Replaying the actual
production boundary -- BF16 input rounding, power-of-two UE8M0 scale, Ampere
half-up E4M3 encoding, and BF16 RoPE store -- gives zero code/scale/RoPE
differences at all captured layers.  The corrected declared gate reports:

| Layer | Ratio | Main max | Index max | Top-512 set difference | Attention max | Attention mean |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 0 | 0 | 0 | 0.0177016 | 0.000694218 |
| 21 | 128 | 0 | n/a | n/a | 0.0296428 | 0.000777281 |
| 42 | 4 | 0 | 0 | 0 | 0.0290079 | 0.00100806 |

All values are finite.  Main/index maximum limits remain 0.0625/0.03125;
attention maximum/mean limits remain 0.0625/0.002.

The real tensors also refine the physical page contract.  The allocator block
still represents 256 source tokens, but a compressed cache page contains
`256 / compress_ratio` physical rows: 64 rows (37,376 bytes main, 8,448 bytes
index) for ratio 4 and 2 rows (1,168 bytes main) for ratio 128.  The prior
operation fixture used a standalone 256-row page.  Full-context payload bytes
remain 151,228,416 because the smaller page has proportionally more allocator
blocks; no precision or memory-ceiling decision changes.

## Accepted dependency-light C++ gate

`scripts/export_dsv4_reference_layer_fixture.py` writes compact, versioned
binary fixtures (0.81--0.90 MiB each).  The consumer
`strata-dsv4-layer-contract-probe` links only `strata_core`: it has no LibTorch,
Python, Lvllmds4-x, host fallback, or NVMe dependency.  It independently replays
compressor softmax/RMSNorm/quantization, the FP8 weighted-ReLU learned index,
exact top-512 membership, and sink-aware sparse attention.

| Layer | Main max | Index max | Top-512 set difference | Attention max | Attention mean |
|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 0 | 0 | 0.0177019 | 0.000694218 |
| 21 | 0 | n/a | 0 | 0.0296421 | 0.000777289 |
| 42 | 0 | 0 | 0 | 0.0290003 | 0.00100806 |

The C++ gate therefore accepts the reference layer contract.  It does not yet
wire the live Strata cache, attention, fixed hidden buffers, accepted mHC,
CPU-MoE handoff, GPU reduction, or graph ownership.

The retained C++ changes and probe were rebuilt with the normal CUDA-enabled
configuration. `make check` passed both registered test targets: `strata-tests`
and `strata-sim-smoke` (2/2, 128.37 seconds). Python byte compilation, shell
syntax checking, and `git diff --check` also pass.

## Execution and next decision

On a shell with the two RTX 3090s visible, launch the server in a named tmux
session, issue the request, then run the checker:

```bash
tmux new-session -d -s dsv4-layer-gate \
  'cd /home/rodrigo/Developer/strata && scripts/run_dsv4_reference_layer_gate_server.sh'

/home/rodrigo/Developer/Lvllmds4-x/venv/bin/python \
  scripts/query_dsv4_reference_layer_gate.py \
  --trace-dir results/dsv4-reference-layer-gate/trace \
  --output results/dsv4-reference-layer-gate/request.json

/home/rodrigo/Developer/Lvllmds4-x/venv/bin/python \
  scripts/check_dsv4_reference_layer_trace.py \
  results/dsv4-reference-layer-gate/trace/rank00-layer02-position02175.pt \
  results/dsv4-reference-layer-gate/trace/rank00-layer21-position02175.pt \
  results/dsv4-reference-layer-gate/trace/rank00-layer42-position02175.pt \
  --output results/dsv4-reference-layer-gate/check.json

/home/rodrigo/Developer/Lvllmds4-x/venv/bin/python \
  scripts/export_dsv4_reference_layer_fixture.py \
  results/dsv4-reference-layer-gate/trace/rank00-layer02-position02175.pt \
  results/dsv4-reference-layer-gate/trace/rank00-layer21-position02175.pt \
  results/dsv4-reference-layer-gate/trace/rank00-layer42-position02175.pt \
  --output-dir results/dsv4-reference-layer-gate/fixtures

for fixture in results/dsv4-reference-layer-gate/fixtures/*.bin; do
  build/strata-dsv4-layer-contract-probe "$fixture" || exit $?
done
```

Both checker stages are positive.  The next authorized implementation is the
smallest complete live dependent chain: fixed device hidden/residual buffers,
these persistent physical KV pages and in-place attention, accepted fused mHC,
stream-ordered CPU-MoE activation/output, GPU-side shared/routed reduction, and
next-layer consumption.  Full-model teacher forcing/generation gates precede
performance measurement, followed by at least three interleaved repetitions.
No speedup is claimed here.
