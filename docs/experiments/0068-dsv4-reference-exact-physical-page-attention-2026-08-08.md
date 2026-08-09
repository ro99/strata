# Experiment 0068: exact DeepSeek physical-page attention output gate

## Status

**Accepted at layers 2, 21, and 42; component-probe work stops here.** The
retained layer-2/21/42 v3 traces now export a dependency-light
binary fixture whose timed inputs stop at the captured physical main/SWA pages,
slot maps, BF16 query, validity mask, and attention sink. The pre-materialized
BF16 candidates and production BF16 attention output are verification-only.

The standalone SM86 CUDA/cuBLAS graph decodes the physical pages, performs the
accepted BF16 BMM and separate BF16 score scale, replays the exact maximum,
denominator, and value associations from experiment 0067, performs
`div.full.f32`, and rounds the result to BF16. Acceptance requires zero valid
page-materialization mismatches and zero BF16 output mismatches. The former
0.0625 maximum / 0.002 mean tolerance is not an acceptance path.

The first operator RTX 3090 run produced exact output but localized four
materialization mismatches to signed-zero encoding. After canonicalizing E4M3
zero magnitude to production's BF16 positive zero, the layer-42 rerun has zero
page and output BF16 mismatches. Its complete page-to-output median is 0.057344
ms. The subsequent layer-2 cross-check is also page- and output-exact with a
0.058368 ms median. The final ratio-128 layer-21 cross-check is page- and
output-exact with a 0.031744 ms median. All three required gates are accepted;
the next action is actual Strata runtime integration on a new local branch.

## Bounded hypothesis and operating point

Experiment 0067 changed the attention defect from an unknown end-output error
into an exact operation contract: raw and scaled scores, all maxima and
denominators, and all 81,920 FP32 value partials reproduce the installed
reference bit-for-bit. The bounded hypothesis is that applying that contract to
the already exact physical-page materializer also reproduces the production
BF16 layer output exactly.

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

- **Target term and bottleneck.** Eventual runtime integration replaces work
  inside the 71.140 ms attention share of the current 130.145 ms serialized
  non-MoE/handoff path. The isolated gate does not claim a throughput win.
- **Primary metrics.** BF16 output mismatch count and median captured
  physical-page-to-BF16-output device time across 11 repetitions after three
  graph warmups.
- **Correctness gate.** Layer 42 must have zero materialization failure codes,
  zero valid-page BF16 mismatches, zero output BF16 mismatches, and zero
  non-finites. The same gate then applies independently to layers 2 and 21.
- **Memory ceiling.** The probe reserves the same 36,694,016-byte comparable
  workspace used by the rejected complete-chain candidate and reports every
  explicit input/scratch/output allocation. This is below the accepted
  151,228,416-byte full-context KV/state envelope; host RSS remains far below
  216 GiB. Eventual Strata admission remains 0.95.
- **Resource signs.** The candidate spends GPU SM work on physical E4M3/E8M0
  decode, BF16 tensor-core BMM, exponentials, and reductions, and reads/writes
  the compact physical pages and scratch buffers through HBM. It does not add
  CPU-MoE work, NVMe traffic, host writes, routing changes, prediction,
  precision changes, or fallback behavior.
- **Rollback.** Any mismatch or non-finite rejects this body. Stop without
  layers 2/21, a favorable tolerance, runtime wiring, or a new branch.

The fixed process/CUDA setup is expected to dominate the eleven sub-millisecond
device samples, but the complete arm remains under a second on the operator
GPU. A component probe is no longer cheaper for the question under test:
experiment 0067 already accepted every component, while only the complete path
can test the final division/store and report page-to-output cost.

## Fixture contract

`scripts/export_dsv4_reference_attention_page_output_fixture.py` requires the
three real position-2,175 v3 traces and emits ignored `D4PAG01` fixtures:

| Layer | Ratio | Compressed valid/width | SWA valid/width | Main/SWA pages | Candidates | Bytes | SHA-256 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 4 | 512/512 | 128/128 | 9/2 | 640 | 1,135,520 | `f590c0ff6cb004d8720db55d00ca36045192eb32c1e2ef7d6b9a280a4b8399cb` |
| 21 | 128 | 17/128 | 128/128 | 9/2 | 256 | 414,512 | `8f100ab626a606e6bc61942845c26b37ffa5958309e9b1e87d3482f2d6bba7db` |
| 42 | 4 | 512/512 | 128/128 | 9/2 | 640 | 1,135,520 | `bb5945f72c839e50432080d04dce51287e19cc9cfa9394bfcf401aa6e940885e` |

Each physical main/SWA row has 576 data bytes followed by eight E8M0 scale
bytes in its page-scale region. The first 448 data bytes are E4M3 values in
seven 64-value groups; the remaining 128 bytes are 64 BF16 RoPE values copied
bit-for-bit. Invalid candidates materialize as zero but are excluded from the
page oracle and from every finish reduction.

## Candidate arithmetic and timing scope

The graph contains, in order:

1. physical page lookup and E4M3 × E8M0-to-BF16 materialization;
2. cuBLAS `[32,512] x [512,candidates]` BF16 GEMM with FP32 accumulation and
   BF16 output;
3. a distinct multiply by `1/sqrt(512)` followed by BF16 rounding;
4. four-warp sink-aware maximum reductions in 128-candidate boundaries;
5. `value * log2(e)` plus `ex2.approx.f32` denominator reduction with the
   accepted four-warp XOR association;
6. the accepted value order `1,0,2..31-stride4`, four FMA residue groups, and
   `(group0+group2)+(group1+group3)` combination;
7. `div.full.f32` and round-to-nearest BF16 store.

CUDA events surround one graph launch. Three complete graph launches warm the
captured path, followed by eleven measured launches. Page/output D2H copies are
outside the timed interval and are reported separately as verification bytes.

## Implementation and non-GPU validation

The first target request found only a stale build graph. Reconfiguring the
existing Release tree regenerated the target, after which CUDA 12.8 compiled
and linked it for SM86. Cubin resource inspection reports:

```text
materialize                 24 registers,   0 local,   0 shared
maximum                     20 registers,   0 local,  20 shared
denominator                 23 registers,   0 local,  20 shared
value                      132 registers,   0 local, 640 shared
divide/store                12 registers,   0 local,   0 shared
```

The value launch is `[32,2]` with 256 threads, so it consumes 33,792 registers
per block and stays below SM86's 65,536-register limit. The retained PTX
contains both `ex2.approx.f32` and `div.full.f32`. Fixture export compiles, the
manifest round-trips against all three file sizes, `git diff --check` passes,
and the required full `make check` passes both registered tests in 112.19
seconds.

The attempted managed-environment invocation stopped at:

```text
cudaSetDevice: no CUDA-capable device is detected
```

This is an environment limitation, not a gate result. The operator then ran
layer 42 on the first RTX 3090:

```bash
cd /home/rodrigo/Developer/strata
set -o pipefail

build/strata-dsv4-attention-page-output-probe \
  results/dsv4-reference-exact-page-output/fixtures/layer42-position02175.bin \
  --device 0 | tee \
  results/dsv4-reference-exact-page-output/layer42-device0.json
```

The result is:

```text
accepted                                      false
valid materialized BF16 mismatches          4 / 327,680
materialization maximum / mean error         0 / 0
materialization non-finite / failure code    0 / 0
output BF16 mismatches                       0 / 16,384
output maximum / mean error                  0 / 0
complete minimum / median / maximum          0.056320 / 0.058368 / 0.060416 ms
explicit / reserved / owned device bytes     1,242,204 / 36,694,016 / 37,936,220
H2D / verification D2H                       447,320 / 688,132 bytes
maximum RSS / major faults / swaps            499,996 KiB / 4 / 0
```

Decision: reject the overall layer-42 gate while retaining the exact output and
complete cost as diagnostic evidence. The only authorized continuation is to
identify the four bit patterns and make physical materialization bit-exact.
Do not run layers 2/21 or create the integration branch yet.

## Signed-zero localization

The four bit differences are exactly the four E4M3 negative-zero codes `0x80`
selected from the layer-42 main pages:

```text
candidate 26,  dimension 425
candidate 272, dimension 41
candidate 294, dimension 139
candidate 457, dimension 431
```

The production materialized oracle stores BF16 positive zero at every one of
these positions and contains no BF16 negative zero in any valid candidate. The
probe decoder preserved the E4M3 sign when its decoded magnitude was zero, so
its four actual values were BF16 negative zero. This explains both the four bit
mismatches and exactly zero maximum/mean numerical error.

The correction canonicalizes only E4M3 zero magnitude to positive zero before
scaling and BF16 conversion. It changes no nonzero value, page mapping, BMM,
finish arithmetic, tolerance, or timing scope. Layer 42 must be rerun and still
requires zero page and output BF16 mismatches before layers 2/21.

The corrected CUDA 12.8 SM86 target rebuilds successfully. Resource usage is
unchanged at 24 registers for materialization and 132 registers for the
256-thread value kernel. `git diff --check` passes, and the required full
`make check` passes both registered tests in 99.41 seconds. The next action is
only the same layer-42 RTX 3090 command against this rebuilt binary.

## Corrected layer-42 acceptance

The corrected operator rerun is fully exact:

```text
accepted                                      true
valid materialized BF16 mismatches          0 / 327,680
materialization maximum / mean error         0 / 0
materialization non-finite / failure code    0 / 0
output BF16 mismatches                       0 / 16,384
output maximum / mean error                  0 / 0
complete minimum / median / maximum          0.057344 / 0.057344 / 0.059392 ms
explicit / reserved / owned device bytes     1,242,204 / 36,694,016 / 37,936,220
H2D / verification D2H                       447,320 / 688,132 bytes
maximum RSS / major faults / swaps            500,000 KiB / 0 / 0
```

Decision: accept layer 42 under the strict zero-mismatch page and output gate.
The 0.057344 ms median is the requested complete physical-page-to-BF16-output
cost at this 640-candidate operating point; it is not yet a runtime throughput
measurement. Run layer 2 next and record its result before layer 21. Runtime
integration remains blocked until both cross-layer fixtures are also exact.

## Layer-2 acceptance

The next sequential cross-layer gate also passes exactly:

```text
accepted                                      true
valid materialized BF16 mismatches          0 / 327,680
materialization maximum / mean error         0 / 0
materialization non-finite / failure code    0 / 0
output BF16 mismatches                       0 / 16,384
output maximum / mean error                  0 / 0
complete minimum / median / maximum          0.057344 / 0.058368 / 0.060416 ms
explicit / reserved / owned device bytes     1,242,204 / 36,694,016 / 37,936,220
H2D / verification D2H                       447,320 / 688,132 bytes
maximum RSS / major faults / swaps            474,888 KiB / 0 / 0
```

Decision: accept layer 2 under the same strict page and output bit gates. Its
shape and transfer volumes match layer 42 because both use ratio 4 and 640
candidates. Run and record layer 21 next; no runtime wiring or integration
branch is authorized until that final ratio-128 fixture is exact.

## Layer-21 acceptance and final decision

The final ratio-128 cross-layer gate passes exactly:

```text
accepted                                      true
valid materialized BF16 mismatches          0 / 74,240
materialization maximum / mean error         0 / 0
materialization non-finite / failure code    0 / 0
output BF16 mismatches                       0 / 16,384
output maximum / mean error                  0 / 0
complete minimum / median / maximum          0.031744 / 0.031744 / 0.033760 ms
explicit / reserved / owned device bytes     496,620 / 36,694,016 / 37,190,636
H2D / verification D2H                       119,528 / 294,916 bytes
maximum RSS / major faults / swaps            457,044 KiB / 0 / 0
```

The accepted complete-cost summary is:

| Layer | Ratio | Valid candidates | Page BF16 mismatches | Output BF16 mismatches | Complete median (ms) |
|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 640 | 0 | 0 | 0.058368 |
| 21 | 128 | 145 | 0 | 0 | 0.031744 |
| 42 | 4 | 640 | 0 | 0 | 0.057344 |

Decision: accept the isolated physical-page attention implementation under the
strict zero-mismatch contract. Stop creating component probes. Commit this
single-purpose experiment on the local-only experiment branch, then create a
new local runtime-integration branch intentionally based on that accepted
commit. The next bounded implementation is to put this attention body into
Strata's persistent device KV path, integrate accepted mHC, connect the real
CPU-MoE callback and GPU shared/routed reduction, and keep hidden state device
owned across all 43 layers. Only then run full-model teacher forcing,
generation, and three comparable performance repetitions.
