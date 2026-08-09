# Experiment 0067: DeepSeek reference attention-finish contract gate

## Status

**Score and every finish-intermediate gate accepted; isolated layer output next.**
The opt-in external trace retains the actual
layer-42 BF16 `torch.bmm` output before scaling, the same production buffer
after in-place scaling, and maximum/denominator/value states after every
128-candidate boundary of a trace-only finish replay. The installed production
finish still produces the model output unchanged. The boundary tensors are
accepted because the trace replay is BF16-bit-exact with that untouched output.

The real layer-42 trace has a `[1, 32, 640]` BF16 score buffer and five complete
finish boundaries. Raw-to-scaled scores, boundary maxima, final partial output,
and replay-to-production output all have zero mismatches; all partials are
finite and every denominator is positive and nondecreasing. No native CUDA
attention candidate, runtime wiring, numerical relaxation, throughput claim,
or installed-package edit is present. The next action is to apply the accepted
finish association to the isolated physical-page layer-42 candidate, not
runtime or full-model integration.

## Bounded hypothesis and operating point

The installed reference's production BF16 score buffer and its visible
128-candidate finish association can localize the remaining layer-42 mismatch.
If the captured scores differ from a future native diagnostic, the BMM contract
is first. If scores agree, the first differing maximum, denominator, or value
partial selects the finish operation that must be reproduced.

```text
tau = max_r W_r / B_r + sum_serial

current total median                       244.812 ms/token
complete MoE                              114.667 ms
routed CPU median / best            106.869 / 89.226 ms
standalone routed CPU                       87.879 ms
serialized non-MoE / handoff path          130.145 ms
attention                                   71.140 ms
accepted CPU floor + envelope         90.061--90.197 ms
remaining margin to 100 ms             9.939--9.803 ms
target                                  <=100.000 ms/token
```

- **Target term and bottleneck.** The eventual mechanism targets the 71.140 ms
  attention share inside the current 130.145 ms serialized non-MoE path. This
  measurement changes no routed-CPU work; CPU scheduling remains out of scope.
- **Primary metric.** The first mismatch among the BF16 score buffer and the
  five cumulative maximum, denominator, and 512-wide value states.
- **Correctness gate.** Before the intermediates are trusted, the trace replay
  must have zero BF16 mismatches against the untouched production output, zero
  non-finite partials, exact score scaling, and exact boundary maxima. A future
  native result remains bounded by 0.0625 maximum output error, 0.002 mean
  error, zero non-finites, and zero physical-page materialization mismatches.
- **Memory ceiling.** Host RSS stays below 216 GiB; the reference server uses
  explicit 0.87 VRAM admission; eventual Strata admission remains 0.95; and the
  accepted full-context KV/state allocation remains 151,228,416 bytes.
- **Resource signs.** The opt-in trace adds one diagnostic kernel, about 465 KiB
  of retained finish tensors, and post-operation D2H/file output. These costs
  are outside any performance measurement. Model precision, cache layout,
  routing, CPU payload, steady-state storage traffic, and production output are
  unchanged. A future candidate may add GPU SM/HBM work only after localization.
- **Rollback.** Stop if the trace replay is not output-exact, if a required
  tensor is absent, or if the captured states cannot distinguish BMM from
  finish. Do not build another CUDA body or wire the runtime to rescue a
  negative gate.

## Installed source contract

The audited installed files are:

```text
/home/rodrigo/Developer/Lvllmds4-x/venv/lib/python3.12/site-packages/
  vllm/models/deepseek_v4/nvidia/flashmla.py
  vllm/v1/attention/backends/mla/sparse_mla_kernels.py
```

At batch-one compressed decode, `flashmla.py` materializes 512 compressed plus
128 SWA candidates, allocates a `[1, 32, 640]` BF16 score buffer, and calls the
matmul finish with `value_block_size=512` and `candidate_block_size=128`.
`matmul_sparse_mla_attention_with_sink` executes:

```text
torch.bmm(BF16 q, BF16 kv^T, out=BF16 score_buffer)
score_buffer.mul_(1 / sqrt(512))
candidate-block sink-aware finish
```

The installed finish makes one 128-candidate maximum pass, then a second pass
that accumulates denominator and 512-wide values relative to the final maximum.
Its launch uses eight warps. The trace kernel preserves that visible operation
order and launch shape while storing each boundary. It runs only after the
installed production finish; it neither replaces production output nor enters
Strata's C++ runtime.

The raw and scaled scores are each 40,960 bytes. Five FP32 boundaries contain
640-byte maximums, 640-byte denominators, and 327,680 bytes of value partials.
A padded BF16 replay output adds 65,536 bytes, for about 476,416 added payload
bytes before container metadata.

## Retained tooling

- `scripts/dsv4_reference_layer_trace/sitecustomize.py` remains opt-in through
  `PYTHONPATH` and `STRATA_DSV4_LAYER_TRACE_DIR`. It patches only the worker
  process and never edits the installed package.
- `scripts/check_dsv4_reference_attention_finish_trace.py` validates schema,
  BF16 score scaling, five exact maximum boundaries, finite/positive cumulative
  states, final-partial reconstruction, and BF16-bit-exact replay output.
- Existing layer-trace checking and fixture export accept both the retained v2
  traces and the new v3 container.

The capture's fixed cost is expected to be about 133 seconds of model load plus
15--25 seconds for the real 2,175-token request and trace. Budget 2.5--3.5
minutes total. The useful measurement is one layer-42 decode call, so setup is
far larger than the measured window. A short prompt was rejected because it
would not reproduce the non-trivial 544-candidate top-512 selection; a synthetic
kernel would not establish the installed production BMM buffer.

## Accepted GPU result

The reference loaded the model in 149.051 seconds, initialized and warmed the
engine in another 30.75 seconds, and served the traced 2,175-token request plus
two generated tokens in 15.306 seconds. The server admitted 6,144 KV blocks,
65,006 tokens of capacity, and 1.98 target-length requests at explicit 0.87
admission. It shut down cleanly after the checkers completed.

The pre-existing real layer gate remained positive:

| Layer | Main max | Index max | Top-512 difference | Attention max | Attention mean | Page mismatches |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 0 | 0 | 0.0177017 | 0.000694219 | 0 |
| 21 | 0 | n/a | n/a | 0.0135112 | 0.000691958 | 0 |
| 42 | 0 | 0 | 0 | 0.0534005 | 0.000895742 | 0 |

All three remain below the declared 0.0625 maximum / 0.002 mean attention
bounds. Layer 42's finish capture reports:

```text
score dtype / shape                    BF16 / [1, 32, 640]
score scale                            0.04419417382415922
raw-to-scaled BF16 mismatches          0
128-candidate boundaries               5
boundary-maximum max / mean error      0 / 0
non-finite partial values              0
non-positive denominators              0
denominator decreases                  0
final-partial-to-replay mismatches      0
replay-to-production BF16 mismatches   0
replay-to-production max / mean error  0 / 0
```

The layer-42 v3 trace is 1,928,735 bytes, compared with 1,450,291 bytes for the
same physical-page trace before finish capture. This matches the predicted
roughly 476 KiB addition. The request timing includes prefill and first-shape
JIT compilation, including the trace kernel, and is not an inference-throughput
measurement.

Decision: accept the captured score and boundary tensors as the reference
contract. This gate does not yet choose BMM versus finish because the rejected
experiment-0066 candidate retained only final outputs and timings. The next
cheapest action is a diagnostic that emits native BF16 scores and the same five
states from the existing real layer-42 fixture. Do not implement or time a
complete CUDA attention candidate before that comparison identifies the first
differing stage.

## Native BF16 score diagnostic accepted

The first dependent localization stage is accepted.
`scripts/export_dsv4_reference_attention_finish_fixture.py` exports only the
captured layer-42 query, 640 materialized BF16 candidates, and expected raw and
scaled BF16 scores. The ignored fixture is 770,084 bytes. It contains no model
weights, physical pages, Python object encoding, finish states, or output.

`strata-dsv4-attention-score-probe` is an SM86-only standalone CUDA/cuBLAS
diagnostic. It performs exactly one `[32,512] x [512,640]` BF16 GEMM with FP32
accumulation and BF16 output, then applies the score scale in a separate BF16
boundary. It reports raw and scaled BF16 mismatch counts and exits nonzero on
the first gate failure. It has no finish kernel, attention output, page decoder,
runtime hook, fallback, or performance claim.

The probe owns 688,128 H2D bytes for query/candidates, 81,920 D2H verification
bytes for raw/scaled scores, and 770,048 device bytes. The measured wall time is
0.697516 seconds including process/CUDA initialization. This is cheaper and
more specific than rebuilding the rejected physical-page attention body.

Build and non-GPU validation are complete. The CUDA 12.8 release target builds
for SM86, Python fixture export compiles, `git diff --check` passes, and the
required full `make check` passes both registered targets in 153.27 seconds.
Only pre-existing warnings were emitted.

The exact score gate ran on the first RTX 3090:

```bash
cd /home/rodrigo/Developer/strata
set -o pipefail
build/strata-dsv4-attention-score-probe \
  results/dsv4-reference-attention-finish-contract/layer42-attention-score.bin \
  --device 0 | tee \
  results/dsv4-reference-attention-finish-contract/score-probe-device0.json
```

Decision tree:

1. Any raw-score mismatch selects the BMM contract and stops finish work.
2. Exact raw scores plus any scaled-score mismatch selects the BF16 scaling
   boundary and stops finish work.
3. Only exact raw and scaled scores authorize the next diagnostic for the five
   finish boundaries. They do not authorize a complete attention candidate.

The measured result is exact:

```text
device                                NVIDIA GeForce RTX 3090 (device 0)
shape                                 [32, 640, 512]
raw BF16 mismatches                   0 / 20,480
raw maximum / mean absolute           0 / 0
scaled BF16 mismatches                0 / 20,480
scaled maximum / mean absolute        0 / 0
non-finite raw / scaled               0 / 0
fixture / owned device bytes          770,084 / 770,048
H2D / D2H bytes                       688,128 / 81,920
elapsed                               0.697515841 s
```

Decision: BMM and the separate BF16 scaling boundary are accepted and no longer
eligible explanations for the experiment-0066 layer-42 error. The first
divergence lies inside the finish. The next binding gate compares boundary
maximums first, then denominators, then value partials. Stop at the first
nonzero mismatch; do not emit a final attention output to rescue it.

## Native finish-maximum diagnostic accepted

The next dependent stage is deliberately maximum-only because maximum is the
first finish state and does not depend on the exponential, denominator, or
512-wide value reduction association. The exporter retains the scaled BF16
scores, the 640-entry validity mask, 32 FP32 attention sinks, and the five
reference FP32 running maxima. The ignored fixture is 42,404 bytes with SHA-256
`be3ee650bf6fd0e109521b9051e404c6336331558ca2402d29df08cfee9e2ecf`.

`strata-dsv4-attention-finish-max-probe` launches one 128-thread SM86 block per
head. Each block consumes the candidates in the reference's five 128-candidate
boundaries, performs a four-warp maximum reduction, combines it with the
per-head sink, and emits only the five cumulative maxima. Acceptance requires
all 160 FP32 states to be bit exact and finite. The probe does not evaluate an
exponential, denominator, value partial, final attention output, physical page,
or runtime path, so a later stage cannot conceal a maximum mismatch.

The probe owns 41,728 H2D bytes, 640 D2H bytes, and 42,368 device bytes. CUDA
startup dominates; budget less than one second for the arm. A combined finish
kernel was rejected because it would make the first divergent operation
ambiguous. The initial build saw a stale CMake configure, and the first
post-configure compile found an unavailable `CUDART_INF_F` spelling. An
explicit device negative-infinity bit value replaced that spelling without
changing the algorithm; the CUDA 12.8 SM86 target then built successfully.

Run the gate on the first RTX 3090:

```bash
cd /home/rodrigo/Developer/strata
set -o pipefail
build/strata-dsv4-attention-finish-max-probe \
  results/dsv4-reference-attention-finish-contract/layer42-attention-finish-max.bin \
  --device 0 | tee \
  results/dsv4-reference-attention-finish-contract/finish-max-probe-device0.json
```

The RTX 3090 result is exact at every boundary:

```text
candidate end                          128   256   384   512   640
FP32 mismatches                          0     0     0     0     0
maximum absolute error                   0     0     0     0     0
non-finite values                        0     0     0     0     0
fixture / owned device bytes        42,404 / 42,368
H2D / D2H bytes                     41,728 / 640
elapsed                              0.495953022 s
```

Decision: accept score masking, sink handling, and the running-maximum
association. All 160 FP32 maximum states are bit exact. This authorizes only a
denominator-only diagnostic. It does not authorize value implementation, a
final attention body, runtime wiring, or a throughput claim.

## Native finish-denominator diagnostic accepted

The accepted final maximum makes denominator evaluation the next dependent
operation. Cached PTX for the captured SM86 trace shows the compiled contract:
FP32 subtraction, multiplication by FP32 `log2(e)` (`0x3fb8aa3b`),
`ex2.approx.f32`, four 32-lane XOR sum trees over each 128-candidate block, and
a four-warp XOR reduction with offsets 2 and 1. The cumulative denominator
starts at the sink weight and adds one block sum at each boundary.

The denominator fixture contains only scaled BF16 scores, validity, sinks, the
already-accepted final maxima, and five expected FP32 denominator boundaries.
The CUDA probe emits no maximum, value partial, or attention output. Acceptance
requires all 160 denominators to be FP32-bit-exact, finite, positive, and
nondecreasing. Any mismatch stops value work. Export validation passes and
produces a 42,532-byte ignored fixture with SHA-256
`5734dca76c21064bc1384d5483c3d260b4e98c95eded08a560d14c896b274946`.
The probe requires 41,856 H2D bytes, 640 D2H bytes, and 42,496 owned device
bytes. CUDA initialization again dominates, so the arm budget is less than one
second. The CUDA 12.8 SM86 target builds successfully.
`cuobjdump` confirms that its cubin retains the reference immediate
`0x3fb8aa3b`, `MUFU.EX2`, the five 32-lane `SHFL.BFLY` stages, and the final
two `SHFL.BFLY` stages over four warp sums. The required full repository check
passes both registered tests in 99.42 seconds, and `git diff --check` passes.

Run the denominator gate on the first RTX 3090:

```bash
cd /home/rodrigo/Developer/strata
set -o pipefail
build/strata-dsv4-attention-finish-denom-probe \
  results/dsv4-reference-attention-finish-contract/layer42-attention-finish-denom.bin \
  --device 0 | tee \
  results/dsv4-reference-attention-finish-contract/finish-denom-probe-device0.json
```

The RTX 3090 result is exact at every boundary:

```text
candidate end                          128   256   384   512   640
FP32 mismatches                          0     0     0     0     0
maximum absolute error                   0     0     0     0     0
non-finite values                        0     0     0     0     0
non-positive / decreases                       0 / 0
fixture / owned device bytes        42,532 / 42,496
H2D / D2H bytes                     41,856 / 640
elapsed                              0.461056481 s
```

Decision: accept the compiled fast exponential and scalar sum association.
All 160 FP32 denominator states are bit exact, finite, positive, and
nondecreasing. This authorizes only a value-partial diagnostic. It does not
authorize final-output implementation, runtime wiring, or a throughput claim.

## Native finish-value diagnostic accepted

The cached trace PTX maps each output dimension across four candidate-residue
groups. Each group accumulates 32 candidates in local order `1, 0, 2..31` with
stride four using an initial FP32 multiply followed by FP32 FMAs. The four
group partials combine as `(group0 + group2) + (group1 + group3)`, and that
block result is added to the running value state. This is the final retained
finish operation before division and BF16 output.

The prepared diagnostic consumes the scaled scores, validity, accepted final
maxima, and materialized BF16 values. It emits only the five `[32,512]` FP32
value boundaries. Explicit round-to-nearest multiply/FMA/add intrinsics pin the
observed association. It does not consume a denominator, divide, produce an
attention output, read physical pages, or enter the runtime. Any mismatch among
the 81,920 FP32 states stops final-body work. Export validation passes and
produces a 1,024,808-byte ignored fixture with SHA-256
`61ae837e1d8966bfea67cc97a77ff852003aa8212f209ff7ab8fac2c619c0361`.
The probe requires 697,088 H2D bytes, 327,680 D2H bytes, and 1,024,768 owned
device bytes. CUDA initialization dominates, so the arm budget remains less
than one second. The CUDA 12.8 SM86 target builds successfully. Its cubin uses
142 registers, 640 shared bytes, and zero stack/local bytes. `cuobjdump`
confirms `MUFU.EX2`, the explicit FP32 multiply/FMA chains, the two pair sums,
their final group sum, and a separate running-state add. The full repository
check passes both registered tests in 100.10 seconds, and `git diff --check`
passes.

Run the value-partial gate on the first RTX 3090:

```bash
cd /home/rodrigo/Developer/strata
set -o pipefail
build/strata-dsv4-attention-finish-value-probe \
  results/dsv4-reference-attention-finish-contract/layer42-attention-finish-value.bin \
  --device 0 | tee \
  results/dsv4-reference-attention-finish-contract/finish-value-probe-device0.json
```

Decision for a repetition: any FP32 mismatch or non-finite state supersedes the
accepted result and stops at the value association.

The first RTX 3090 arm produced no value data. CUDA rejected the launch with
`too many resources requested for launch`: 142 registers across 512 threads
requires 72,704 registers for one block, above the SM86 65,536-register limit.
This is an instrumentation defect, not a correctness result. Split the 512
dimensions across two 256-thread blocks per head, reducing the requirement to
36,352 registers per block while preserving each dimension's arithmetic. The
split duplicates only the diagnostic's 128 weight preparations and 640 shared
bytes; fixture traffic, device allocation, and the acceptance gate are
unchanged. The source now launches a `[32,2]` grid with 256 threads per block;
the rebuilt cubin still uses 142 registers per thread with zero stack/local
bytes, for 36,352 registers per block. Its audited arithmetic remains unchanged.
The repeated full `make check` passes both registered tests in 99.98 seconds,
and `git diff --check` passes. Retry the same command; `tee` will overwrite the
zero-byte ignored result from the rejected launch.

The corrected RTX 3090 result is exact across the complete retained state:

```text
shape                                 [32, 5, 512]
candidate ends                        128, 256, 384, 512, 640
FP32 mismatches per boundary          0, 0, 0, 0, 0
maximum / mean absolute error         0 / 0
non-finite values                     0
first mismatch                        none
dimension blocks / threads            2 / 256
fixture / owned device bytes          1,024,808 / 1,024,768
H2D / D2H bytes                       697,088 / 327,680
elapsed                               0.534666886 s
```

Decision: accept the weighted-value multiply/FMA and cross-group reduction
association. All 81,920 FP32 value states are bit exact. Together with the
accepted scores, maxima, and denominators, this identifies the complete
reference finish arithmetic. The next authorized gate is the isolated
physical-page layer-42 output under the existing 0.0625 maximum / 0.002 mean
oracle. Layers 2 and 21 must then pass before runtime/device-chain integration.

## Reproduction

Run the server in a named session, issue the one request, validate the existing
three-layer contract, then validate the new layer-42 finish trace:

```bash
tmux new-session -d -s dsv4-finish-contract \
  'cd /home/rodrigo/Developer/strata && \
   RESULT_DIR=/home/rodrigo/Developer/strata/results/dsv4-reference-attention-finish-contract \
   scripts/run_dsv4_reference_layer_gate_server.sh'

/home/rodrigo/Developer/Lvllmds4-x/venv/bin/python \
  scripts/query_dsv4_reference_layer_gate.py \
  --trace-dir results/dsv4-reference-attention-finish-contract/trace \
  --output results/dsv4-reference-attention-finish-contract/request.json

/home/rodrigo/Developer/Lvllmds4-x/venv/bin/python \
  scripts/check_dsv4_reference_layer_trace.py \
  results/dsv4-reference-attention-finish-contract/trace/rank00-layer02-position02175.pt \
  results/dsv4-reference-attention-finish-contract/trace/rank00-layer21-position02175.pt \
  results/dsv4-reference-attention-finish-contract/trace/rank00-layer42-position02175.pt \
  --output results/dsv4-reference-attention-finish-contract/layer-check.json

/home/rodrigo/Developer/Lvllmds4-x/venv/bin/python \
  scripts/check_dsv4_reference_attention_finish_trace.py \
  results/dsv4-reference-attention-finish-contract/trace/rank00-layer42-position02175.pt \
  --output results/dsv4-reference-attention-finish-contract/finish-check.json
```

Expected outputs are the three ignored `.pt` traces, `request.json`,
`layer-check.json`, `finish-check.json`, and `serve.log` under
`results/dsv4-reference-attention-finish-contract/`. The server remains in
`tmux` for inspection; stop it after both checkers complete:

```bash
tmux send-keys -t dsv4-finish-contract C-c
```

The recorded run is accepted. For a fresh repetition, a rejected
`finish-check.json` supersedes this capture and stops the branch. An accepted
repeat still does not authorize a complete CUDA candidate by itself.

## Targeted validation before the GPU arm

The Python files compile in the installed external environment, the launcher
passes `bash -n`, the import hook patches the installed FlashMLA alias without
editing it, and the pre-existing v2 physical-page traces still pass the layer
checker and exporter. The new maximum exporter compiles and emits the declared
42,404-byte fixture. Its SM86 CUDA target builds. Because this retained stage
adds compiled CUDA and CMake content, the full `make check` suite is required
before the result commit. That suite passes both registered tests in 112.27
seconds, and `git diff --check` passes.
