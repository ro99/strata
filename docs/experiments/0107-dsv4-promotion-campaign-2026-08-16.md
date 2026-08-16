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

All six arms completed with the same preserved binary, SHA-256
`b0c125f9168ccb82db676df45dbcf16b860902f79868f010cf990b9f6e276a5f`.
The baseline selector reproduced main's deterministic structure exactly in all
three arms: 29,111 paged-attention calls, 553,109 kernel launches and
6,989,956,736 page bytes. The candidate reproduced the accumulated stack:
86 calls, 1,655 launches and 30,564,224 page bytes, with the query device term
at 0.332--0.338 s.

### Individual performance results

All rates are computed from the recorded 677 prefill tokens or four generated
tokens. `Expert wait` is the prefill weight cache's `demand_wait_seconds`
counter, the serial expert-demand upload term.

| Arm | Prefill s | tok/s | Attention s | Query s | KV s | Score s | Query device s | MoE s | Expert wait s | Decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline 1 | 81.242948 | 8.3330 | 43.435496 | 10.675473 | 4.615248 | 27.753098 | 7.282198 | 33.484284 | 23.447275 | 9.1056 |
| candidate 1 | 52.958281 | 12.7836 | 21.788353 | 3.534635 | 3.666934 | 14.206908 | 0.337863 | 27.056734 | 17.803755 | 9.1406 |
| baseline 2 | 86.668919 | 7.8113 | 42.569493 | 10.284354 | 4.335886 | 27.569016 | 7.144656 | 39.904993 | 30.356269 | 8.9692 |
| candidate 2 | 54.182870 | 12.4947 | 22.359804 | 3.759287 | 3.910688 | 14.320013 | 0.333580 | 27.518866 | 17.651567 | 8.9059 |
| baseline 3 | 73.142670 | 9.2559 | 43.174087 | 10.647901 | 4.495700 | 27.618395 | 7.210668 | 25.758613 | 15.895113 | 8.4132 |
| candidate 3 | 51.554424 | 13.1318 | 22.831761 | 3.859001 | 3.952410 | 14.628241 | 0.332141 | 24.360771 | 14.378979 | 8.8974 |
| **baseline median** | **81.242948** | **8.3330** | **43.174087** | **10.647901** | **4.495700** | **27.618395** | **7.210668** | **33.484284** | **23.447275** | **8.9692** |
| **candidate median** | **52.958281** | **12.7836** | **22.359804** | **3.759287** | **3.910688** | **14.320013** | **0.333580** | **27.056734** | **17.651567** | **8.9059** |

The median total-prefill improvement is 28.284667 s, or **1.5341x**;
throughput rises 53.4%. Pairwise savings are 28.284667, 32.486049 and
21.588246 s (1.5341x, 1.5996x and 1.4187x). The attention reduction itself is
stable and mechanistically local: median attention falls 20.814284 s, composed
of 6.888614 s in query, 0.585012 s in KV and 13.298382 s in scoring. The query
device counter falls 21.62x at the medians.

### Measured spread and promotion verdict

The imported approximately 1.2-second production-page spread was too small for
this campaign. Baseline total prefill spans 73.142670--86.668919 s, a
**13.526249 s** range. Candidate total prefill spans 51.554424--54.182870 s, a
**2.628446 s** range. The baseline's attention bucket spans only 0.866003 s;
its MoE bucket spans 14.146380 s and expert demand wait spans 14.461156 s. Thus
the new large baseline spread is again owned by expert service, but it occurs
at page 8192 despite the arena's explicit, stable NUMA placement. Placement was
correctly falsified by 0106; the cause of service-time variance remains open.

Using the larger 13.526249-second within-arm range as the conservative measured
noise band, the promotion result remains outside noise. The least favourable
cross-arm comparison, fastest baseline against slowest candidate, still saves
18.959800 s. All three paired savings also exceed the conservative band. This
is therefore a **promotable total-prefill result**, subject to the user's
curated commit-by-commit landing decision; this branch is not itself suitable
for wholesale merge because its ancestry deliberately preserves rejected
experiments.

### Structural and resource results

The deterministic counters are identical within each arm, so each row below
applies to all three repetitions of that arm. Decimal GB is used for byte
traffic; byte counts are included to avoid ambiguity.

| Metric | Baseline | Candidate | Sign |
|---|---:|---:|---:|
| Paged-attention calls | 29,111 | 86 | -338.5x |
| Paged-attention kernel launches | 553,109 | 1,655 | -334.2x |
| Paged-attention page bytes | 6,989,956,736 (6.990 GB) | 30,564,224 (0.0306 GB) | -228.7x |
| Query matmul device median | 7.210668 s | 0.333580 s | -21.62x |
| Expert demand H2D bytes | 73,984,409,088 (73.984 GB) | 74,380,770,816 (74.381 GB) | +0.54% |
| Prefill cache hits | 143,280 | 85,230 | -58,050 |
| Prefill cache misses | 10,456 | 10,512 | +56 |
| Prefill cache evictions | 7,828 | 7,884 | +56 |
| RSS median | 158,745,137,152 B | 158,858,199,040 B | +113,061,888 B |
| GPU 1 VRAM | 22,683,975,680 B | 22,988,062,720 B | +304,087,040 B |
| GPU 2 VRAM | 22,606,381,056 B | 22,910,468,096 B | +304,087,040 B |

The approximately 290 MiB/device VRAM increase is the row-batched attention
workspace and remains below the declared 384 MiB/device ceiling. The tensor
projection retains compact byte FP8 activation storage and does not add an
expanded persistent weight copy. Expert residency is unchanged; the 0.54%
demand-byte and 56-entry cache differences are deterministic consequences of
the different attention-side lease/acquisition schedule and are reported as a
small adverse sign rather than hidden.

### Correctness

Every arm generated exactly `2107, 8777, 1277, 440`, and every arm reported
zero decode checkpoint-read bytes. Decode is the unchanged single-row path.
Its paired candidate/baseline throughput ratios are 1.00384, 0.99294 and
1.05755, for a paired median of **1.00384** (+0.38%); two of three pairs favour
the candidate. The separate raw arm medians are 8.9692 tok/s baseline and
8.9059 tok/s candidate (-0.71%), well inside the overlapping run ranges. The
paired campaign therefore shows no decode regression while stating the raw
median honestly.

`make check` passed before the campaign binary was preserved and again before
this result commit.

## Verdict

0107 passes the correctness, structural and promotion gates. At the actual
677-token/page-8192 operating point, the accumulated row-batched attention and
SM86 FP8 tensor projection stack improves median total prefill from 81.243 s
(8.333 tok/s) to 52.958 s (12.784 tok/s), a 1.534x speedup outside the measured
spread. The full branch remains preserved for research traceability. No merge,
push or ref movement is authorized or performed.
