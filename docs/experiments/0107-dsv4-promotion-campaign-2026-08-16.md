# Experiment 0107 — DSV4 accumulated prefill promotion campaign

Status: **in progress.** This campaign compares main-equivalent behaviour with
the accumulated row-batched physical attention and SM86 FP8 tensor projection
stack in one binary at the production 677-token, page-8192 operating point.

## Corrections carried forward

Experiment 0106 proved that production rank-local TP2 prefill consumes the
explicitly bound tiled expert arena: shard 0 is bound to node 0 and shard 1 to
node 1. The separate bare anonymous resident allocation still exists around
`deepseek_checkpoint.cpp` line 515, but this operating point does not consume
it. Therefore the handover's claim that the production arena has no NUMA policy
is outdated; variable first-touch placement cannot explain these runs.

The previously quoted approximately 16-second expert-upload noise floor was
also imported from page-64 runs moving 468 GB and does not describe page 8192.
Preserved page-8192 runs move about 74 GB and cluster much more tightly. This
campaign measures the operating-point spread directly instead of reusing a
constant across page shapes.

## Predeclared hypothesis, resource model, and gates

At 677 tokens, the preserved stack has already measured two deterministic
resource reductions:

- physical attention dispatches fell from 29,111 calls / 553,109 launches /
  6.99 GB of KV page reads to about 86 / 1,677 / 30.56 MB;
- query projection device service fell from 7.159 s to about 0.339 s by
  replacing the multi-row native FP8 CUDA-core reduction with an in-register
  E4M3 decode feeding BF16 WMMA on SM86.

The hypothesis is that these reductions lower the serial attention term enough
to reduce total prefill outside the run-to-run spread measured in this same
campaign. The primary metric is median total prefill seconds and tokens/s over
three interleaved repetitions, with every run and each arm's spread reported.

Correctness requires identical generated IDs between every arm, zero decode
checkpoint reads, unregressed decode throughput, and `make check`. Precision,
router semantics, expert count, top-k, residency and mHC semantics remain
unchanged. Host admission remains 216 GiB; the tensor path retains byte FP8
activations and the physical-attention workspace remains within its declared
384 MiB/device ceiling. Rollback is not automatic: any gate result is preserved
and reported for a landing decision.

Resource signs are already bounded by the constituent experiments. The stack
reduces attention launch/synchronization volume, redundant page reads, native
FP8 weight-read amplification, and query device compute. It adds per-page
candidate/workspace storage and SM86 WMMA conversion work, but no expanded
persistent weights and no second activation representation. Weight cache,
expert upload volume, host RSS, router work, and decode's single-row path should
remain unchanged; all are measured here.

## Campaign design and budget

One preserved binary selects both mechanisms explicitly:

- baseline: `--no-dsv4-batched-page-attention`
  `--no-dsv4-fp8-tensor-page`;
- candidate: `--dsv4-batched-page-attention`
  `--dsv4-fp8-tensor-page`.

The order is baseline/candidate repeated three times. Every arm is untraced,
uses devices 1 and 2, 677 prompt tokens, page 8192, and four generated tokens.
Expected model time is roughly four minutes per arm and approximately 25
minutes total, plus one build. No 2,612-token arm or fixed/marginal fit is in
scope.

## Results

Pending.
