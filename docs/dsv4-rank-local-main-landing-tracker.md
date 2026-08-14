# DSV4 Rank-Local TP2 Main-Landing Tracker

Last updated: 2026-08-13

Branch: `feat/dsv4-rank-local-decode`

Starting HEAD: `e289f46fb053f73032a0dcc5aa8e5792acfb5620`

Main baseline: `61f1f02` (merge base at tracker creation)

## Mission

Land rank-local TP2 decode on `main` as an explicit, fail-closed opt-in while
preserving centralized decode as the bit-unchanged default. The supported
rank-local path targets the model's declared 1,048,576-token context at
8.0 token/s (125.0 ms per decoded token). On 2026-08-13 the user explicitly
accepted a review variance of `±2.0 ms` around that target, so a three-run
median at or below 127.0 ms may advance. The strict 125.0 ms result is always
reported separately and is never relabelled as 8.0 token/s when it misses.

This file is the canonical recovery ledger. A step advances only when its exit
criteria are supported by committed code and reproducible artifacts. Negative
results remain recorded here; a later arm does not replace them.

## Governing evidence

- `docs/dsv4-rank-local-rescue-brief.md`
- `docs/dsv4-rank-local-architecture.md`
- `docs/dsv4-rank-local-extraction-manifest.md`
- `docs/experiments/0079-dsv4-rank-local-moe-residual-attribution-2026-08-11.md`
- `docs/experiments/0091-dsv4-m3-callback-free-staging-2026-08-12.md`
- `docs/experiments/0092-dsv4-m3-timing-falsifier-2026-08-12.md`
- accepted fixture implementation at `a31ac58`
- `research/moe-tiered-memory-decode-optimization.md`

## Immutable acceptance contract

Correctness:

- Centralized decode is the default and matches `main` bit for bit under the
  full-model teacher-forcing and generation oracles.
- Rank-local mode is explicit and fail-closed. No rejected or failed rank-local
  request silently falls back to centralized execution.
- Rank-local preserves exact routing, attention/compression layout, mHC state,
  sparse selection, routed/shared association, FP32 reductions, BF16
  publication, DSpark verification, and failure closure.

Performance:

- Three interleaved repetitions at the declared operating point.
- Strict target: median rank-local decode is at most 125.0 ms/token and at
  least 8.0 token/s.
- User-approved review boundary: the three-run median may be at most 127.0
  ms/token (`125 ± 2 ms`). A result admitted by this boundary is reported as a
  tolerance acceptance, not as meeting the strict 8.0 token/s target.
- The 114.944312 ms M3 result is fixture scope only and is never reported as
  end-to-end production throughput.

Memory and I/O:

- Every admitted byte account controls the allocation it describes.
- Per-device VRAM stays within the declared ceiling using actual runtime usage,
  not planner arithmetic alone.
- Host RSS stays within 231,928,233,984 bytes.
- Steady-state decode performs zero checkpoint reads and zero timed-path weight
  or workspace allocations.
- KV/index capacity reflects the physical ownership actually used by both
  ranks at 1,048,576 tokens.

## Baseline at tracker creation

Latest saved short-context production results, 18-token prompt, 4,096-token
configured context, seven decode steps:

| arm | ms/token | token/s | exact generated token IDs | decode checkpoint bytes |
|---|---:|---:|---|---:|
| centralized | 153.503050 | 6.514529 | yes | 0 |
| rank-local TP2 | 418.462005 | 2.389703 | yes | 0 |

Latest rank-local runtime device usage:

- rank 0: 23,933,878,272 bytes
- rank 1: 23,931,781,120 bytes
- declared rank-local ceiling: 22,548,578,304 bytes/device
- overrun: 1,385,299,968 and 1,383,202,816 bytes

Accepted M3 fixture result:

- 114.944312 ms/forward median, 8.70 forwards/s
- routed CPU term: 73.896784 ms at 46.677143 GB/s
- non-CPU envelope: 41.047528 ms
- execution shape: 43 queued layers and one completion

Initial review verdict: **NOT MERGEABLE**. The production runtime executes the
sequential M3 control (`run()` for layers 0-41, one queued terminal layer), not
the accepted 43-layer queued candidate. The 1M sparse-selection path and
allocation-enforced admission are not complete.

## Step 1 — Restore the accepted short-context execution mechanism

State: **IMPLEMENTED; FINAL DEFAULT-BUILD CHECK PENDING**

Target term: remove per-layer completion from `Sigma_serial` and restore the
CPU/GPU/layer overlap demonstrated by M3.

Required work:

1. Audit the live layer call against `run_m3_candidate()` at `a31ac58`.
2. Move compressor/index-compressor preparation into rank-local executor-owned
   command state; do not perform a separate host-visible preparation per layer.
3. Publish each layer's live KV rows through fixed per-command staging and the
   executor page-patch boundary, preserving the ownership fix from experiment
   0091.
4. Queue all 43 layers and call `finish_chain()` once for the no-selection
   operating point.
5. Preserve fail-closed status publication and terminal head behavior.

Exit criteria:

- Exact adjacent-layer and full 43-layer results.
- Exactly 43 queued commands and one completion per decoded token.
- Zero per-layer host collection, zero host-only duplicate preparation, and
  zero measured-path host `update_buffer` page transport.
- Focused tests, default tests, NCCL tests, and `git diff --check` pass.

Rollback condition: if live dependencies cannot be represented by executor-
owned command state without a broad semantic redesign, record the precise
dependency and stop this implementation arm before performance claims.

## Step 2 — Prove short-context production exactness and throughput

State: **ACCEPTED AT USER-APPROVED VARIANCE BOUNDARY; STRICT TARGET MISSED**

Target term: verify that the restored overlap reduces observed production
`tau`, not only fixture time.

Required work:

1. Add a deterministic production acceptance runner with centralized and
   rank-local arms.
2. Compare queued rank-local generated IDs and hashes against an independently
   synchronized rank-local execution of the same accepted TP2 publication
   association. Compare centralized execution separately against `main`; a
   centralized/rank-local bit comparison is not an oracle because their BF16
   arithmetic association is intentionally different.
3. Run at least three interleaved repetitions after warm-up.
4. Record the full per-phase `tau` decomposition and identify `argmax_r`.
5. Record checkpoint I/O, allocations, synchronization, RSS, and per-GPU VRAM.

Exit criteria:

- All exactness and failure-injection gates pass.
- Median production rank-local decode is at most the explicit 127.0 ms review
  ceiling; the strict 125.0 ms evaluation is recorded independently.
- No rank-local repetition is a material regression hidden by variance.

Rollback condition: if the queued production mechanism is exact but remains
above the centralized control, retain the measurement and re-instantiate the
cost model before changing another mechanism.

## Step 3 — Make admission and KV ownership describe reality

State: **COMPLETED**

Target term: no throughput target; this is a mandatory safety and residency
gate.

Required work:

1. Apply the admitted expert-cache capacity before CUDA arena/cache allocation,
   or release the centralized prefill-only residency before rank-local decode.
2. Re-derive the per-device ceiling from per-device components; never charge an
   aggregate spine value as though it were per-device.
3. Define physical KV/index ownership at 1M and make runtime capacity planning,
   page acquisition, tests, and documentation use the same ownership model.
4. Compare planner totals with actual CUDA usage and explain residual context,
   allocator, cuBLAS, and NCCL bytes.

Exit criteria:

- Actual per-device usage is within the declared ceiling.
- Admission rejection occurs before an over-budget session becomes active.
- The admitted cache cap changes the real allocation.
- Full-context KV/index pages fit without eviction-driven I/O or missing-rank
  promotion churn.

Rollback condition: reject rank-local admission when the exact residency cannot
fit; do not raise a program ceiling without measured physical headroom and a
complete component ledger.

## Step 4 — Build and measure the full-context sparse-index mechanism

State: **OPEN — SHORT PATH CORRECTED AND STRICT-GATE PASSING; INDEX MECHANISM
STILL UNBUILT AND GATED ON A MEASURED SCORER SCREEN**

The original candidate-sharding mechanism remains falsified. The short-path
correction required before any index work is now done and measured: the
production median is `111.177092 ms/token` (`8.994659 token/s`), which passes
the strict `125.0 ms` target rather than the tolerance boundary. The permitted
complete index budget is therefore `15.822908 ms/token` at the review ceiling.
The next action is a standalone exact-scorer screen against that budget; no
sharding, exchange, or merge work is justified until it passes.

Target term: the corrected production-page measurement of 68.482 ms/token for
the 1M Lightning Indexer, of which the exact score kernel accounts for at least
59.961 ms/token.

Required work:

1. Execute exact index query projection, score, and top-k within the dependent
   device command sequence rather than through a host synchronization.
2. Shard candidate scoring across the two ranks, exchange each rank's local top
   512, and deterministically merge the 1,024 candidates into the exact global
   top 512.
3. Prove equality with the scalar reference, including ties and boundary scores.
4. Measure ratio-4 sparse attention, ratio-128 dense compressed attention, page
   lookup, communication, and the full 43-layer token at 1,048,576 context.

Exit criteria:

- Exact full-context teacher-forcing and generation gates pass.
- `tau(1M)` is measured end to end, not assembled only from component probes.
- Three interleaved repetitions achieve the strict 125.0 ms/token target, or
  at most the explicit 127.0 ms review ceiling with the strict result reported
  separately.
- Zero steady-state NVMe reads and all Step 3 residency gates remain satisfied.

Rollback condition: if exact rank-sharded selection plus the remaining
dependent envelope cannot fit the explicit 127 ms review ceiling, record the
binding negative result and do not describe short-context performance as
full-context support. That condition has fired for the original mechanism:
even perfect 2x score sharding with zero merge and zero local-selection cost
leaves a 29.980 ms/token dependent score term on top of the accepted 126.676
ms/token short-context path.

### Step 4 successor handoff

The next agent must not implement the original candidate-sharding design as a
standalone mechanism: the component lower bound has already falsified it. The
next decision must begin from the current production cost model:

```text
accepted short-context wall                 126.675822 ms/token
routed CPU argmax                            75.895000 ms/token
dependent non-CPU envelope                   51.921000 ms/token
1M exact score kernel, one rank              59.960565 ms/token
ideal 2x sharded score lower bound            29.980283 ms/token
review ceiling                               127.000000 ms/token
```

Therefore even an otherwise free index shard needs at least `29.656105
ms/token` removed from the accepted short path before it can fit, and real
local selection, exchange, merge, ratio-128 attention growth, and integration
make the required removal larger. The historical Stage-10 CPU estimate cannot
supply it: the current production CPU body is only `3.006 ms` above the
unequal-scope external navigation value.

The first successor action is a bounded attribution of the current accepted
production non-CPU envelope, not another index implementation. Use the
existing temporary phase instrumentation or one production-shaped trace to
identify a concrete term that can remove roughly 30 ms while preserving the
exact queued topology. State fixed setup versus measured-window cost before
running it; the saved acceptance arm is about 126 seconds of initialization,
3.4 seconds of prefill, and 4.0 seconds of measured decode. The pre-topology
0087 trace is navigation only and cannot substitute for the current chain.

Only after a measured short-path correction creates enough headroom may the
agent reopen an exact dual-rank index microbenchmark. Its gate must be derived
as `127 ms - measured corrected short tau` and must include local selection,
communication, deterministic merge, and the ratio-128 attention term. If no
bounded exact short-path correction can create that headroom, close the 1M
landing mission as rejected or ask the user to authorize a narrower supported
context; do not advance to Step 5 under the current full-context claim.

## Step 5 — Centralized regression, landing hygiene, and merge boundary

State: **BLOCKED BY STEP 4**

Required work:

1. Compare centralized behavior on this branch with `main` using full-model
   teacher-forcing and generation oracles; require bit identity.
2. Run `make check` in default and NCCL builds, focused rank-local tests,
   failure injection, the production acceptance matrix, extraction audit, and
   `git diff --check`.
3. Remove diagnostic-only production surface that is not required for the
   accepted mechanism; keep negative experiment evidence and reusable records.
4. Update architecture, extraction manifest, experiment record, and this
   tracker to match what actually runs.
5. Review the complete `main...HEAD` diff for unrelated changes and construct
   reversible, single-purpose landing commits.

Exit criteria:

- Every mission gate is green with artifact paths and hashes recorded below.
- Centralized remains the default and bit-unchanged.
- Rank-local remains explicit and fail-closed.
- Review boundary reached; no unrelated optimization begins.

## Artifact ledger

| date | step | artifact | hash/result | binding interpretation |
|---|---|---|---|---|
| 2026-08-13 | baseline | `results/dsv4-rank-local-phase/centralized.json.log` | 153.503050 ms/token | short-context centralized navigation point |
| 2026-08-13 | baseline | `results/dsv4-rank-local-phase/rl-attr3.json.log` | 418.462005 ms/token | binding negative production rank-local result |
| 2026-08-13 | review | default `ctest` | 2/2 pass | build health only; not production acceptance |
| 2026-08-13 | review | NCCL `ctest` | 2/2 pass | build health only; not production acceptance |
| 2026-08-13 | review | `scripts/audit_dsv4_extraction_manifest.sh` | PASS, 13/13 | traceability only; not runtime correctness |
| 2026-08-13 | Step 1 negative | `results/dsv4-rank-local-main-landing/step1-falsifier/` | SHA256 `d66199300397874c092db248d8718ff64469a63840538f61ec3dac20bf76da91`, exit 1 | first live queued arm exposed checkpoint tensor-name mismatch |
| 2026-08-13 | Step 1 intermediate | `results/dsv4-rank-local-main-landing/step1-falsifier-r2/` | SHA256 `b502cd20313cea52f64bd3f3313445d93727989a29162ce25942379512a47b8d`, 403.980 ms/token | 43-layer chain exact, but production used 28 CPUs per rank rather than the calibrated 24 |
| 2026-08-13 | Step 1 intermediate | `results/dsv4-rank-local-main-landing/step1-falsifier-r3-cpu24/` | SHA256 `d946b38eeca689d19a037ad5335fb00aa4ed08c59cff65e6205cc43427eeb515`, 232.445 ms/token | exact 24-thread pools removed the SMT-contention regression |
| 2026-08-13 | Step 1 negative | `results/dsv4-rank-local-main-landing/step1-falsifier-r4-attribution/` | SHA256 `1787f0b7f209dd4d7a403daf278a477fa17aaa3f665dc39d2fe30d739e82f6cc`, exit 1 | broad event timing perturbed the callback/accounting path; no performance claim |
| 2026-08-13 | Step 1 intermediate | `results/dsv4-rank-local-main-landing/step1-falsifier-r5-attribution-noevents/` | SHA256 `aedc6710d8b839ab11b9866800f3eb84618651d4d616e5d4b35e5cce052920d`, 247.153 ms/token | exact no-event attribution arm; cold workspace and page setup still included |
| 2026-08-13 | Step 1 negative | `results/dsv4-rank-local-main-landing/step1-falsifier-r6-late-replica-wait/` | SHA256 `1787f0b7f209dd4d7a403daf278a477fa17aaa3f665dc39d2fe30d739e82f6cc`, exit 1 | first late replica-wait arm failed before callback diagnostics were sufficient |
| 2026-08-13 | Step 1 intermediate | `results/dsv4-rank-local-main-landing/step1-falsifier-r7-account-diagnostic/` | SHA256 `ecedfb63bfb2d1cfeaab2f32599dbb6360eb2608e771c14e44a60ce324d9a18a`, 219.828 ms/token | exact; moving rank 1's wait to the page callback restored QKV overlap |
| 2026-08-13 | Step 1 negative | `results/dsv4-rank-local-main-landing/step1-seven-step-r8/` | SHA256 `4ae43085edf0e484b1cc0efd01bee0631d30a3b3efdb9e8f4b98a0c5802baa99`, exit 1 | second token exposed previous-token KV leases retained in reusable scratch |
| 2026-08-13 | Step 1 negative | `results/dsv4-rank-local-main-landing/step1-seven-step-r9-lease-fix/` | SHA256 `955c37d7640e7f8d4ecb9c239fae5628c4bd3db0842e69a507affd35cad8b2e7`, exit 1 | lease fix passed reservation, then an existing prefill physical callback reported an unlocalized failure |
| 2026-08-13 | Step 1 confirmation | `results/dsv4-rank-local-main-landing/step1-seven-step-r10-callback-diagnostic/` | SHA256 `575170f6405bcbfb9f12e5e40375a6357015c7c205ef81b11b3e59422c387270`, 7 steps, 145.869 ms/token | same arm passed after layer-qualified callback diagnostics; first eight generated IDs match the saved exact baseline |
| 2026-08-13 | Step 1/2 confirmation | `results/dsv4-rank-local-main-landing/step1-long-r11/` | SHA256 `2daf28accf749f2353437937765874f6bd94b7cbb0b04933e3dee71e27bae79c`, 31 steps, 131.646 ms/token | repeated queue reuse passed; 7.596 token/s is below the gate and detailed-event counters contain an invalid interval, so this is not the binding clean timing arm |
| 2026-08-13 | Step 2 falsifier | `results/dsv4-rank-local-main-landing/step2-clean-r12/` | SHA256 `50ecf70a5f9ee3b3dc92b21284b9f317b845a8a8ff9ee8e790cdf1d0375c0dd0`, 31 steps, 130.303 ms/token | clean 7.674 token/s arm is exact against r11 but misses the 8.0 token/s gate by 5.303 ms/token; queued layer wall is the binding term |
| 2026-08-13 | Step 2 discriminator | `results/dsv4-rank-local-main-landing/step2-cpu-overlap-r13/` | SHA256 `f1f0a4462c21c168b8643db7b1276caad3900084a73bdc3af27533358f3eefcf`, 31 steps, 129.439 ms/token | rank CPU callbacks overlap for 74.485 of a 77.650 ms/token union; CPU serialization is falsified, leaving about 50.2 ms/token of dependent non-CPU envelope |
| 2026-08-13 | Step 2 correction falsifier | `results/dsv4-rank-local-main-landing/step2-replica-device-r14/` | SHA256 `96b313286108a1dc9a8baacb83f848796956948882a36af03ee1836e32828002`, exact token `[22510]`, 207.881 ms cold wall | replacing rank 1's host callback/QKV download with device-only publication removed exactly 132,268 D2H bytes/token and about 12 ms from the comparable cold arm; long gate justified |
| 2026-08-13 | Step 2 correction | `results/dsv4-rank-local-main-landing/step2-replica-device-r15/` | SHA256 `5c99d80b97ba8747ec388267e18765de3cac9ef6d7a6ed8c32768fa907391588`, 31 steps, 128.309 ms/token | exact and lower than r12/r13, but aggregate still includes first-step setup and remains 3.309 ms above the steady-state gate |
| 2026-08-13 | Step 2 instrumentation negative | `results/dsv4-rank-local-main-landing/step2-steady-r16/` | SHA256 `41782552e77ff01bfa99748622637b988498871bc524e276881b6394e8af9579`, 31 steps, 126.916 ms/token | execution passed, but the per-step vector was cleared by the existing metrics-snapshot restore; unusable for warm-up separation, preserved as a measurement defect |
| 2026-08-13 | Step 2 correctness negative | `results/dsv4-rank-local-main-landing/step2-steady-r17/` | exit 1, layer 32 physical append commit failure | localized intermittent centralized-prefill lifetime bug: collection inspected/reused the secondary device's callback context without first draining that device's stream; no decode timing result |
| 2026-08-13 | Step 2 correctness negative | `results/dsv4-rank-local-main-landing/step2-steady-r18/` | exit 1, layer 32 callback saw `valid=0` while collection saw `valid=1` | proved concurrent context replacement also occurs on the primary attention stream; the primary MoE completion event does not dominate every deferred attention host node |
| 2026-08-13 | Step 2 timing navigation | `results/dsv4-rank-local-main-landing/step2-steady-r19/generation.json` | SHA256 `3148df3e63f367c3871d502c3aadd08d16145e97d68de821cc890eae64303e68`, steady median 124.575 ms/token | first clean arm below 125 ms, but it used only the saved short prefix as its oracle; retain as navigation, not acceptance evidence |
| 2026-08-13 | Step 2 invalid cross-topology oracle | `results/dsv4-rank-local-main-landing/step2-acceptance/rep-0-{centralized,rank-local}.json` | SHA256 `40142098aeb4da8a66057efc5919498087f6e9c0a37f2b5ec455c88a80e2f73b` / `467896f2e1dcfd370f33767b8898857ff124b72723513333652fa29068f6d471`; first mismatch generated index 13: `9544` / `2107` | the run was correctly stopped under its declared gate, but 0073 and 0088 show that this gate was invalid: centralized and accepted rank-local BF16 publication have different arithmetic association and are not bit oracles for one another; retain the timing as navigation only |
| 2026-08-13 | Step 2 state discriminator | `results/dsv4-rank-local-main-landing/step2-state-discriminator/` | centralized/rank-local SHA256 `3fbf8c8f0952c74f32099f7a03141dfd81ad86bb27278d0d78401ceb8e870fe7` / `62bbf1a0cb3a3c5bd0fb7b2a33e7fd82252f3fc35e4af4bf0589690ba12911ff`; comparison `61438628c8f5c5052f3770bd1bb1cf7c54a6138754aaa8c722b4e1c559ac9d12` | first internal mismatch is already position 18, layer 1 FFN input (`51e60d0bb8c9fc32` / `a9be9aa2e86a9ab5`); layer 0 FFN input is exact and both rank replicas agree throughout, so accumulated-token corruption and replica disagreement are falsified |
| 2026-08-13 | Step 2 association boundary | `results/dsv4-rank-local-main-landing/step2-boundary-discriminator/` | centralized/rank-local SHA256 `9c7b0d2a7efe60eaf14e695b17f0a86b2f4bb1f48f40a726b3c13fa31146b3d2` / `282f678836e3c82be2ec414aec91275755595f523045a77f8477aaaae06e6aba` | position-18 layer-0 query `82de...`, KV `79a0...`, FFN input `4f7c...`, and route coefficients `e319...` all match; layer-1 query/KV differ only after layer-0 TP2 MoE publication, locating the cross-topology difference at the exact rank-local association declared by 0073 and explicitly excluded as a centralized oracle by 0088 |
| 2026-08-13 | Step 2 live rank-local oracle | `results/dsv4-rank-local-main-landing/step2-sequential-oracle/` | sequential SHA256 `c425db0e7e4f1c1e05f0834a96a097c76455f8022d2b22c1fb1b8a9f093a1ee2`; queued SHA256 `62bbf1a0cb3a3c5bd0fb7b2a33e7fd82252f3fc35e4af4bf0589690ba12911ff` | all 14 generated IDs, 13 decode terminal hidden hashes, 559 rank-0 FFN inputs, 559 rank-1 FFN inputs, and 559 canonical KV page patches are bit-exact; queued production preserves the accepted sequential TP2 association at this depth |
| 2026-08-13 | Step 2 full rank-local oracle | `results/dsv4-rank-local-main-landing/step2-sequential-oracle-32/sequential.json` | SHA256 `614e4a0b4095f826ea9eac67ec082618a44e533adc5289d8181ae7225345a7cb`, exit 0, 32 generated IDs, 31 decode steps, zero decode checkpoint bytes | independently synchronized accepted TP2 sequence for the complete timing workload; every clean queued acceptance repetition must match all 32 IDs exactly |
| 2026-08-13 | Step 2 clean acceptance | `results/dsv4-rank-local-main-landing/step2-acceptance-r2/` | summary SHA256 `31abf28f720ac7e9d09b15d56b23eaf4d2260e737e5f6b7722b459e3b87e0868`; rank medians `125.477063`, `126.675822`, `127.453849` ms; median `126.675822` ms = `7.894166` token/s; central medians `156.385173`, `151.825566`, `149.786787` ms | all three rank arms exactly match all 32 sequential-oracle IDs and every arm has zero decode checkpoint I/O; strict 125 ms gate **FAILS**, explicit user-approved 127 ms review gate **PASSES**; first summary attempt exited 5 due a runner jq context bug, then the corrected runner reproduced the summary from the untouched six raw arms without rerunning them |
| 2026-08-13 | Step 3 validation negative | NCCL `ctest` before correcting the test constant | 298/300 pass, 1 opt-in skip, 1 failure: expected historical `4,082,533,760` B but canonical physical device payload is `4,050,137,088` B | useful test failure: the old architecture number mixed a broader KV-state account with the payload that the device cache actually promotes; the runtime capacity gate must use the latter and expose the terms separately |
| 2026-08-13 | Step 3 intermediate | `results/dsv4-rank-local-main-landing/step3-vram-falsifier-pre-arena-cap/` | generation SHA256 `9215d78fc487c0f706f71f33fee240716bcb732875194668ebbe473f0f7f61a1`; summary SHA256 `5b5f95dddacd1097554ce3659b1740cccf89bba634c11fa468ca30eefb37af53`; exact `[671,22510]`, zero decode checkpoint I/O, 22,419,734,528 / 22,417,637,376 B final VRAM | physical ceiling passed, but review rejected the admission ledger: its expert-cache cap filled the abstract ceiling without subtracting the rank-local store from the shared CUDA weight arena; preserved as intermediate evidence, not the binding Step 3 result |
| 2026-08-13 | Step 3 short-context allocation pass | `results/dsv4-rank-local-main-landing/step3-vram-falsifier/` | generation SHA256 `c1b4e6b998b2d7e3c8da5278a8f3e25886a4482fbcd96319ea6ec33ffb835eeb`; summary SHA256 `96604135054e38b925f462a674426ae16a25cadd41dfffd052565f0d66895086`; exact `[671,22510]`, zero decode checkpoint I/O | final VRAM 22,419,734,528 / 22,417,637,376 B leaves 128,843,776 / 130,940,928 B; live cache capacity equals pinned spine plus admitted expert cap on both ranks; admitted and general required host totals both equal 159,211,656,924 B |
| 2026-08-13 | Step 3 1M admission pass | `results/dsv4-rank-local-main-landing/step3-1m-admission/` | generation SHA256 `6656da119f49a0caaf99f535478d5350aaa48baa11bb54212c8d8a3ff32ea0a4`; summary SHA256 `ac5e4990473cf2d58ed0058d0a25ffe442f6cc1e5cab6a4ce515499a02d26fc2`; exact `[671,22510]`, zero decode checkpoint I/O | both ranks receive the full `4,050,137,088` B promoted payload capacity; host admission is `163,253,272,284` B; live expert caps are `5,361,896,800` / `5,184,882,944` B; only `7,213,568` B/rank is committed by the short active sequence, so this proves full-context capacity, not Step 4 full-context execution |
| 2026-08-13 | Step 3 admission-only negative | `results/dsv4-rank-local-main-landing/step3-1m-admission-only-bypass-negative/` | exit 0, JSON SHA256 `19a0771aae3b31c8d359af7765f0f4b64d6369f6c671942b2fdb37cf2f92b4f4` | implausible success exposed that the CLI's cheap `--admission-only` branch ignored rank-local topology; no rank-local admission claim is taken from this arm |
| 2026-08-13 | Step 3 under-cap rejection | `results/dsv4-rank-local-main-landing/step3-1m-under-cap-rejection/` | exit 1, log SHA256 `65e2e369d4b22003215398741d7cc65e98818a2a6638116c0be1a105292057f6` | normal initialization rejects rank 0 at `4,050,137,087` B, exactly one byte below the replicated 1M payload, before resident staging or session activation |
| 2026-08-13 | Step 3 admission-only fail-closed correction | `results/dsv4-rank-local-main-landing/step3-rank-local-admission-only-rejection/` | exit 1, log SHA256 `f3152eb18c8340f3353a47a159275d8b7bf00a077ca0cdcac20197539684dace` | rank-local `--admission-only` now refuses to claim what it cannot measure and directs the request to full initialization |
| 2026-08-13 | Step 3 final NCCL test | `ctest --test-dir build-landing-nccl --output-on-failure` | 2/2 pass in 146.46 s | final Step 3 source, admission tests, CUDA tests, and simulator smoke all pass; Step 4 unblocked |
| 2026-08-13 | Step 1 test | NCCL `strata-tests` | 299/300 pass, one declared opt-in skip | all ordinary and focused rank-local tests pass after queued integration |
| 2026-08-13 | Step 4 corrected production-page baseline | `results/dsv4-rank-local-main-landing/step4-index-baseline-production-pages/probe.txt` | SHA256 `69a9238d10bab6e02d9c3f5e408f71f39b58b84aa710bd6a3b1899d526c53642`; 1M median `3.261 ms/layer`, `68.482 ms/token`, five repetitions | binding single-rank 1M index baseline with the production 64-row ratio-4 page geometry |
| 2026-08-13 | Step 4 Nsight discriminator | `results/dsv4-rank-local-main-landing/step4-index-profile/` | report SHA256 `3f9ec7d695723f54a253afc366d6c9190e104c129930042cbd5e5e835b844422`; kernel CSV SHA256 `8107fe14a83543bf51ce4cc7b0440402e653d797dd4629db91d862123c1fa8f8` | exact score kernel is 82.6% of aggregate GPU index time; its 1M call is 2.855265 ms, proving that ideal two-rank score sharding still leaves about 29.980 ms/token before selection or merge |
| 2026-08-13 | Step 4 rejected scorer ILP arm | `results/dsv4-rank-local-main-landing/step4-index-score-ilp-r1/probe.txt` | SHA256 `672bacab5ae95564e364ce50fdfaad35cc1719f203b75440038d5e2c71696364`; `72.572 ms/token` | four interleaved exact accumulators and 16 rows/block regress the binding baseline; source reverted |
| 2026-08-13 | Step 4 rejected scorer metadata arm | `results/dsv4-rank-local-main-landing/step4-index-score-metadata-r1/probe.txt` | SHA256 `5b97eb06db292e949cc67c3fe080d8ef18b50868f045ded11815bf542aecad91`; `67.742 ms/token` | 0.740 ms/token reduction is non-material and cannot clear the mechanism gate; source reverted |
| 2026-08-13 | Step 4 rejected scorer specialization arm | `results/dsv4-rank-local-main-landing/step4-index-score-specialized-r1/probe.txt` | SHA256 `6715dcb0f61d87a1eed57520228a4730da1d3a983539d9d0d39f1d91384bf923`; `68.854 ms/token` | compile-time 64x128 exact inner-loop specialization is negative; source reverted |
| 2026-08-13 | Step 4 build-freshness defect | `build-landing-nccl/strata-deepseek-run` rebuilt to SHA256 `d8ce228bbf37ec3ee1bc1245e8bc80f65a175f551045805fec04b83dbef4bf0a` | binary relinked on first full build after the reverted scorer screens | the post-revert "clean probe rebuild" rebuilt only the probe target, so the runner predated the revert; no prior measurement is retracted because the reverted hunks are index-only and the short path does not run them, but every Step 4 arm below uses the verified binary |
| 2026-08-13 | Step 4 envelope attribution | `results/dsv4-rank-local-main-landing/step4-envelope-profile/` | control SHA256 `3a6db41aab2ca8dd06f30b55015b409c6f09574527e809831e35800acbf5e71a`; profiled SHA256 `7fa90ddf40ee578ccc02bc00f3370e32381b9107f57370ee85ceea7fe8a7347a`; trace SHA256 `27ddda2f24381f9fe67fc737469d3e5d587f595f366aa5e4bd491aabed82e77c` | binding attribution of the dependent non-CPU envelope over 10 steady tokens; both arms exact against the sequential-oracle prefix with zero decode checkpoint I/O; nsys perturbation measured at `+2.6%` (136.43 vs 132.97 ms), and **no hot-path instrumentation was added** |
| 2026-08-13 | Step 4 mechanism screen | `results/dsv4-rank-local-main-landing/step4-norm-correction-r1/norm_screen.cu` and `mechanism-screen-device0.txt` | device 1 `204.53` to `34.52 us` (`5.92x`); device 0 `204.48` to `34.62 us` (`5.91x`); **0 bit mismatches** on both | standalone screen before any production edit: staging the query-rank norm's loads/stores off the single thread while preserving the ascending FP64 `__dadd_rn` order is bit-identical and about 5.9x faster, reproducibly on two different GPU architectures |
| 2026-08-13 | Step 4 norm correction | `results/dsv4-rank-local-main-landing/step4-norm-correction-r1/run.json` | 12-token arm, steady median `111.90 ms/token` against the identical pre-correction arm's `132.97 ms/token` | exact against all 12 sequential-oracle IDs, zero decode checkpoint I/O; `-21.07 ms/token` at equal arm shape; per-step spread also tightened from `124.9-142.8` to `109.9-115.0 ms` |
| 2026-08-13 | Step 4 default-build defect | `results/dsv4-rank-local-main-landing/step4-make-check-default.log` | `src/deepseek_runtime.cpp:6360`, `'rank_local_executor' was not declared in this scope`, exit 2 | **pre-existing worktree defect, not introduced by the kernel correction**: the queued short-context path called the executor outside `STRATA_HAS_NCCL`, so any default `STRATA_ENABLE_NCCL=OFF` build failed to compile. Confirmed by reproducing the identical error with the kernel change stashed. Fixed by guarding the call and keeping it fail-closed |
| 2026-08-13 | Step 4 default `make check` | `results/dsv4-rank-local-main-landing/step4-make-check-default-r2.log` | 2/2 pass, exit 0 | default non-NCCL build compiles and passes after the guard fix |
| 2026-08-13 | Step 4 NCCL `make check` | `results/dsv4-rank-local-main-landing/step4-make-check-nccl.log` | 2/2 pass, exit 0 | NCCL build compiles and passes with the corrected kernels; `git diff --check` clean |
| 2026-08-13 | Step 4 scorer diagnostic (query-collapse) | `results/dsv4-rank-local-main-landing/step4-scorer-diagnosis/query-collapsed-DIAGNOSTIC-ONLY.txt` | `3.130` to `2.134 ms/layer` kernel at 1M | **intentionally incorrect, never committed, fully reverted**: collapsing the query address isolates query memory traffic at about 32% of the kernel, **falsifying** the L2-bound hypothesis as the primary cause; the residual identifies a latency-hiding limit at 0.28 instructions/cycle under full occupancy |
| 2026-08-13 | Step 4 scorer row sweep | `results/dsv4-rank-local-main-landing/step4-scorer-diagnosis/ilp-r{2,4,8}-probe.txt` | 1M ms/layer: base `3.261`, R2 `1.999`, R4 `1.717`, R8 `1.933` | R=4 is the measured optimum; R=8 loses more to occupancy than it gains in ILP. The earlier rejected ILP screen paired 4 rows/thread with 1024 threads/block (one block per SM), so that arm is a configuration result, not a refutation |
| 2026-08-13 | Step 4 scorer correction | `results/dsv4-rank-local-main-landing/step4-scorer-diagnosis/ilp-r4-binding.txt` | SHA256 `abd18287dfe61fb9e3109e6ca33ae30dbfb3c4255d3624b582af6616af0b4f1e`; complete index `68.482` to `36.062 ms/token` (`1.90x`); score kernel `2.855` to `1.326 ms/layer` (`2.15x`) | exact: **"physical Lightning Indexer matches the scalar oracle at the 1M candidate count" PASSES** at the full 262,144-candidate width; suite 299/300 with one declared opt-in skip |
| 2026-08-13 | Step 4 `locate()` fix | `results/dsv4-rank-local-main-landing/step4-locate-fix-r1/run.json` | exact against all 12 sequential-oracle IDs; steady median `112.53 ms/token` against the pre-fix arm's `111.90` | `locate_physical_kv_block()` replaces both linear scans with direct index, ordered fallback, exhaustive fallback, validating the same predicate on every path; neutral at 4,096 context as expected since tables are tiny there, and removes `2,890 ms/token` at 1M; suite 299/300, one declared skip |
| 2026-08-13 | Step 4 `block_table()` screen | `results/dsv4-rank-local-main-landing/step4-locate-screen/blocktable_screen.cpp`, `blocktable-screen.txt` | by value `11.0--13.6 ms/token`; caller buffer without the dead field `2.9--3.2 ms/token`; `3.8--4.2x`, saving `8.1--10.4 ms/token` | 4,096-block table materialized once per kind per layer per token at 1M; `Dsv4KvBlockInfo::device_resident` was populated on every call and **read by nothing** anywhere in the tree, costing a heap allocation per block |
| 2026-08-13 | Step 4 `block_table()` fix | `results/dsv4-rank-local-main-landing/step4-blocktable-fix-r1/run.json` | exact against all 12 sequential-oracle IDs; steady median `113.68 ms/token`, inside the `111.90`/`112.53` arm spread | dead `device_resident` field removed; `block_table_into()` added and used with process-lifetime buffers at all four hot call sites, including the two learned-index tables; `block_table()` retained as a thin wrapper for non-hot callers; suite 299/300, one declared skip |
| 2026-08-13 | Step 4 gates after `locate()` fix | `results/dsv4-rank-local-main-landing/step4-make-check-default-r5.log` | default 2/2 pass exit 0; NCCL suite 299/300 with one declared skip; `git diff --check` clean | both builds green with all five corrected mechanisms: three attention norms, `dsv4_mhc_weighted_norm`, the index scorer, the radix pivot, and `locate()` |
| 2026-08-13 | **Step 4 binding final acceptance** | `results/dsv4-rank-local-main-landing/step4-final-acceptance/summary.json` | SHA256 `9dc391c245e93d49f536088d62dcaf146471f0abd28c1b67e040ca0a3c1e8285`; rank medians `111.496762`, `114.787396`, `110.820323` ms; median **`111.496762` ms = `8.968870` token/s**; central medians `133.919461`, `128.710203`, `130.545682` ms | binding short-context result on the **complete corrected code**, run after every Step 4 correction rather than assembled from earlier arms. Strict `125.0` ms / `8.0` token/s gate **PASSES**. All three rank arms bit-exact against the 32-token sequential oracle; centralized IDs bit-identical to the pre-correction `step2-acceptance-r2` baseline; zero decode checkpoint I/O in all six arms |
| 2026-08-13 | Step 4 memory gates at final acceptance | same summary | VRAM `22,419,734,528` / `22,417,637,376` B/device against the `22,548,578,304` ceiling; peak RSS `158,350,274,560` B against `231,928,233,984` | Step 3 residency gates re-verified on the corrected code: headroom `128,843,776` / `130,940,928` B/device and `73,577,959,424` B host |
| 2026-08-13 | Step 4 extraction audit | `scripts/audit_dsv4_extraction_manifest.sh` | PASS, 91 paths, 13/13 capabilities | traceability gate still green after the Step 4 corrections |
| 2026-08-13 | Step 4 gates after `block_table()` fix | `results/dsv4-rank-local-main-landing/step4-make-check-default-r6.log` | default 2/2 pass exit 0; NCCL suite 299/300 with one declared skip; `git diff --check` clean | final gate state for the Step 4 correction set |
| 2026-08-13 | Step 4 scorer counters (ncu now permitted) | `results/dsv4-rank-local-main-landing/step4-scorer-diagnosis/ncu-score-counters.txt` | `sm__throughput` and `gpu__compute_memory_throughput` both **91.29% of peak sustained**; achieved occupancy `93.38%`; IPC `0.62`/scheduler; L1 sector hit rate `78.47%` | **corrects this document**: the corrected scorer is near-saturated, so the earlier "`34.7%` of ALU ceiling, `4.1x` therefore plausible" framing was wrong — the binding resource is the combined compute/memory pipe, not the FP32 ALU, and little headroom remains without an algorithm change the exactness contract forbids |
| 2026-08-13 | Step 4 host `locate()` measurement | `results/dsv4-rank-local-main-landing/step4-locate-screen/` | linear scan `2889.637 ms/token` at 1M; binary search `22.206 ms/token`, `130x`, **0 index mismatches** | **binding negative**: the term every 1M projection charged at zero is 22x the entire decode budget. No 1M throughput claim survives it. Binary search is verified equivalent but still exceeds the index budget; an `O(1)` direct index is available and unmeasured |
| 2026-08-13 | Step 4 Nsight Compute availability | `results/dsv4-rank-local-main-landing/step4-scorer-diagnosis/ncu-score-PERMISSION-DENIED.log` | `ERR_NVGPUCTRPERM` on device 1 | correction to the record: Nsight Compute 2025.1.1 **is** installed and recognizes every device; counter *collection* is refused by the driver permission setting, which is a different thing. The scorer diagnosis' hand-derived rates are inferences, not counter measurements, and are labelled as such |
| 2026-08-13 | Step 4 pivot alignment defect | full `strata-tests` run before the alignment fix | 278/300, 21 failures incl. `physical Lightning Indexer rejects malformed pages and shapes` at `tests/test_cuda_backend.cpp:504` | **defect I introduced and caught**: the pivot's new `uint4` group read needs 16 B alignment, but the histogram workspace region was reserved with `alignof(std::uint32_t)`. Misaligned vector loads left a sticky CUDA error, so 21 failures traced to one root cause. Fixed by reserving the region at `alignof(uint4)` |
| 2026-08-13 | Step 4 pivot correction | `results/dsv4-rank-local-main-landing/step4-scorer-diagnosis/pivot-fixed-binding.txt` | SHA256 `0eceffc332f340ce2dfe12c87f35c03673927ffccabca825107b9dfd438f834a`; complete index `36.062` to `34.092 ms/token`; pivot kernel `47.288` to `15.728 us` (`3.01x`) | pivot drops from 22.2% to 8.6% of index GPU time; all four indexer exactness tests pass and the suite returns to 299/300 with one declared opt-in skip |
| 2026-08-13 | Step 4 gates after mHC correction | `results/dsv4-rank-local-main-landing/step4-make-check-default-r3.log`, `step4-make-check-nccl-r2.log` | default 2/2 pass exit 0; NCCL 2/2 pass exit 0; `git diff --check` clean | both builds re-verified with all four corrected kernels |
| 2026-08-13 | Step 4 mHC acceptance | `results/dsv4-rank-local-main-landing/step4-mhc-acceptance/summary.json` | rank medians `111.895850`, `111.721523`, `111.528517` ms; median `111.721523` ms = `8.950827` token/s; central median `133.867678` ms | strict gate still **PASSES**; oracle-exact; centralized IDs still bit-identical; the `+0.544 ms` against the previous matrix is **inside** that matrix's own `2.932 ms` repetition spread, so **no end-to-end change is claimed in either direction** |
| 2026-08-13 | Step 4 mHC mechanism screen | `results/dsv4-rank-local-main-landing/step4-norm-correction-r1/mhc_screen.cu` and `mhc-screen-device0.txt` | device 1 `15.35` to `6.07 us` (`2.53x`); device 0 `15.47` to `6.09 us` (`2.54x`); **0 bit mismatches on both outputs** | `dsv4_mhc_weighted_norm`'s xor reduction shape is itself the contract, so the 64 accumulators and their combination order are preserved and only the elementwise phases widen; `1.320` to `0.522 ms/token` |
| 2026-08-13 | Step 4 corrected acceptance | `results/dsv4-rank-local-main-landing/step4-norm-acceptance/summary.json` | rank medians `109.335778`, `112.268811`, `111.177092` ms; median `111.177092` ms = `8.994659` token/s; central medians `135.012134`, `134.005181`, `137.349264` ms | **strict `125.0` ms / `8.0` token/s gate PASSES**; all three rank arms match all 32 sequential-oracle IDs; centralized IDs bit-identical to the pre-correction arm across all 32 tokens; zero decode checkpoint I/O in all six arms |

## Temporary instrumentation ledger

This ledger is binding for Step 5. Measurement code is not retained merely
because it was useful during recovery. Every diagnostic-only field, clock read,
hash, event, or JSON surface added during Steps 1--4 must appear here before it
is introduced. No row marked `REMOVE` or `ACTIVE TEMPORARY` may be present in
the promotion diff. A `KEEP` decision requires an execution/correctness need or
a deliberately supported production telemetry contract, plus a measured claim
that it does not compromise the acceptance result.

| instrumentation/change | landing disposition | reason |
|---|---|---|
| Fixed 43-slot CPU callback start/finish timestamps and JSON `rank_local_cpu_*` interval fields | **REMOVED FROM THE CURRENT WORKTREE; DO NOT LAND** | temporary discriminator; 86 clock reads per token and a new public metrics surface are not required by execution |
| Per-step decode wall vector and JSON `decode_step_seconds` | **ACTIVE TEMPORARY; REMOVE after the final performance acceptance run** | runner-facing measurement separated first-use setup from steady-state decode in Step 2 and will support the final full-context timing decision; storage is reserved before timing, but this diagnostic surface must not land |
| Aggregate rank-local phase clocks and public `rank_local_*_nanoseconds` / `Dsv4HostMoePhaseTimings` surfaces | **STEP 5 LANDING REVIEW; DEFAULT TO REMOVE unless explicitly justified** | cost-model telemetry is useful during recovery, but clocks and diagnostic API surface on the hot path are not automatically part of the supported production feature |
| Rank-local memory/admission JSON ledger (`rank_local_memory`) | **KEEP** | supported setup-time fail-closed admission evidence: it records initial VRAM, measured resident rank-local weights, the applied centralized expert-cache cap, and admitted device/host totals without adding a hot-path clock or hash |
| Per-layer KV-patch, attention query/KV, FFN-input, and route hashes added to the existing layer trace | **REMOVED FROM THE CURRENT WORKTREE; DO NOT LAND** | correctness discriminator only; the saved raw artifacts retain the evidence without extending production output or hot-path work |
| Temporary sequential rank-local execution selected by `--layer-hash-trace` | **REMOVED FROM THE CURRENT WORKTREE; DO NOT LAND** | the 14-token queued/sequential comparison passed and the complete 32-token sequential oracle was captured; production diagnostics now use the normal queued schedule |
| Read `chain_cpu_moe_phases` after both rank streams finish | **KEEP ONLY IF the aggregate phase telemetry survives Step 5; otherwise remove with it** | if the counters remain, they must be sampled only after their callback owners finish; this fixes telemetry correctness, not model execution |
| Layer-qualified physical/rank-local callback and KV-account errors | **KEEP** | fail-closed diagnostics run only on an error path and localize the owning layer/rank |
| CUDA `host_callback_wait_event` API at the old page-publication boundary | **CURRENTLY HAS NO CALLER; REMOVE IN STEP 5 unless a supported path is proven to require it** | the corrected replica path uses its page-ready ownership event instead; an unused synchronization API must not land as residue |
| Detailed-event arm with an invalid mHC interval | **DO NOT USE OR EXTEND** | rejected measurement evidence; no counter derived from the invalid interval may support acceptance |
| All-device deferred-attention stream drain at centralized collection | **KEEP** | correctness ownership edge; r18 proved the MoE event does not dominate every attention host node even on the primary device, so all streams with live commands must drain before context reuse |

## Decision log

### 2026-08-13 — Recovery opened

- Preserve the experiment branch and accepted primitive extraction.
- Do not merge the present production integration.
- Work in the five dependency-ordered steps above.
- The first binding defect is the sequential production driver plus its
  host-visible preparation/page-transport dependencies.
- The 1M gate remains mandatory; short-context success alone cannot close the
  mission.

### 2026-08-13 — Short-context queued production mechanism restored

- Production now reserves every layer's command-owned storage before enqueue,
  queues all 43 layers, and performs one `finish_chain()` per decoded token.
- Compressor projection is part of each device command. Rank 0 commits the
  canonical physical rows; rank 1 waits only at the page-publication boundary,
  verifies exact Q/KV equality, and publishes the same encoded patch.
- The production topology now uses the calibrated 24 CPUs per NUMA rank. The
  previous 28-thread choice included sibling SMT threads and inflated the live
  one-token arm from 232.445 to 403.980 ms.
- Reusable per-layer KV scratch releases previous-token leases before the next
  transaction reserves the same physical blocks.
- A 31-step reuse arm completed without callback, allocation-lifetime, or KV
  accounting failure. Its first eight token IDs equal the saved centralized
  baseline. A same-depth centralized oracle remains part of Step 2.
- The clean, no-detailed-event Step 2 timing arm is the next binding result;
  the clean result is 130.303 ms/token. The chain accounts for 128.674
  ms/token, versus 3.179 ms/token of rank-0 page callback work and 2.290
  ms/token of terminal head work (both nested in or adjacent to that wall).
- The first CPU phase counter was sampled before the asynchronous callbacks
  drained, so its 37.063 ms/token value is not admissible evidence. The fixed
  interval discriminator measured rank 0 at 74.787, rank 1 at 77.348, overlap
  at 74.485, and their union at 77.650 ms/token. CPU serialization is therefore
  falsified; the remaining delta is in the dependent non-CPU chain.
- The accepted M3 fixture had no page callback. Production r13 added two host
  callbacks plus two Q/KV diagnostic downloads per layer. Rank 1 does not own
  encoding, so its callback was removed: it now waits on the existing rank-0
  page-ready event and uploads the one canonical encoded patch directly from a
  fixed pinned slot. The one-token falsifier is exact and removes 132,268 D2H
  bytes/token.
- The first valid per-step arm measured a 124.575 ms steady median. The first
  interleaved matrix was stopped when centralized and rank-local generation
  diverged at index 13 (`9544` / `2107`). Subsequent review of experiments 0073
  and 0088 showed that the matrix had imposed the wrong oracle: accepted
  rank-local execution deliberately performs a same-rank BF16 routed/shared
  join before its FP32 TP reduction, while centralized execution uses a
  different association. Their downstream bits are therefore not required to
  match. The stopped arm remains recorded and is not relabelled as acceptance.
- That discriminator localized the first semantic difference to the first
  decoded token, position 18, between layer 0's exact FFN input and layer 1's
  differing FFN input. Rank 0 and rank 1 agree at every observed layer. The
  later generated-token mismatch is amplification, not corruption.
- The one-step boundary trace proved that layer 0 query, KV, FFN input, and
  routes match across topologies and that the first difference is the accepted
  layer-0 TP2 publication association. The correct live oracle is therefore an
  independently synchronized rank-local chain. Against that oracle, the queued
  path is bit-exact for all 14 generated IDs, every decode terminal hidden
  state, both ranks' FFN inputs, and every canonical KV patch. A single
  32-token sequential oracle completed with 32 IDs, 31 steps, zero decode
  checkpoint reads, and an empty error log. The discriminator hashes and
  trace-triggered sequential execution were then removed before rebuilding the
  acceptance binary. Each clean queued repetition is gated against all 32
  saved IDs, not merely a prefix.

### 2026-08-13 — Step 2 accepted at an explicit tolerance boundary

- The clean interleaved matrix completed all six arms. Rank-local repetition
  medians were `125.477063`, `126.675822`, and `127.453849 ms/token`; their
  median is `126.675822 ms/token`, or `7.894166 token/s`. The strict 125.0 ms / 8
  token/s target therefore fails by `1.675822 ms/token`.
- After seeing the complete range, the user explicitly accepted `125 ± 2 ms`
  as the Step 2 review boundary. The 126.675822 ms median passes that 127.0 ms
  ceiling. This is a tolerance acceptance, not a retrospective claim that the
  strict target passed.
- All three rank-local outputs match the independently synchronized 32-token
  rank-local oracle exactly. Both topologies are deterministic within their
  own arithmetic association, and all six arms report zero decode checkpoint
  bytes.
- At the median repetition, the instantiated production chain is
  `tau = max(CPU routed 75.895 ms at 45.448 GB/s, other overlapped resources)
  + 51.921 ms dependent residual = 127.815 ms` by aggregate wall accounting.
  The CPU payload is `3,449,290,752 B/token` and remains `argmax_r`; the
  dependent residual contains GPU work, rank joins, page publication, and the
  terminal head and must not be added again. Observable subterms are 3.378 ms
  canonical KV callback, 0.169 ms candidate preparation, and 2.384 ms terminal
  head. Per token, CUDA reports 4,242,658 H2D bytes, 1,761,796 D2H bytes, and
  14,427,136 physical page-read bytes, with zero weight H2D bytes.
- The decode interval reports 73 workspace allocations totalling 7,776,268
  bytes, all associated with first-use setup in the excluded cold step; Step 3
  must make the allocation/admission boundary explicit before the final
  steady-state zero-allocation claim.
- Actual rank-local usage is 23,933,878,272 and 23,931,781,120 bytes/device,
  above the declared 22,548,578,304-byte ceiling. Host RSS peaks at
  158,347,603,968 bytes and fits. Step 2 closes only correctness and the revised
  short-context timing boundary; Step 3 is now the binding safety gate.

### 2026-08-13 — Step 3 admission correction opened

- This step targets residency correctness, not throughput. The binding defect
  is actual rank-local usage above the 22,548,578,304-byte per-device program
  ceiling; CPU routed work remains the measured throughput `argmax_r`.
- The runtime previously derived a new-allocation plan from current free VRAM
  without subtracting the CUDA/context usage that already existed at planning
  time. Rank-local planning now limits future allocations to the smaller of
  the caller's explicit budget and the program ceiling minus measured initial
  device usage, then rejects the session if actual post-setup usage still
  exceeds the ceiling.
- Physical rank-local KV rows are replicated on both ranks. Default planning
  now assigns the full declared-context physical payload to each device;
  caller-supplied capacities below that replicated requirement fail closed.
- At 1,048,576 tokens the canonical promoted device payload decomposes as
  `12,857,344` B sliding + `3,310,616,576` B compressed + `726,663,168` B
  learned index = `4,050,137,088` B per rank. The older `4,082,533,760` B
  architecture figure is a broader KV-state account and is not reused as the
  CUDA device-cache capacity. Step 5 must update that document to distinguish
  device payload, host block metadata/alignment, and compressor state.
- Admission's expert-cache result is applied to the live centralized prefill
  cache before the session becomes active. The supported `rank_local_memory`
  ledger exposes the controlling values. The first test is a short one-step
  allocation falsifier; the 1M-context admission arm remains gated on it.
- The first short arm fit physically, but its advertised expert-cache capacity
  exceeded the suballocation space left in the shared CUDA weight arena after
  the rank-local store. Review therefore withheld acceptance. The corrected
  cap is the minimum of the arena remainder and the total program-ceiling
  remainder; that latter account now includes measured initial device usage
  and the centralized resident mHC allocation.
- The corrected short arm is exact and ceiling-compliant, and the live cache
  capacities equal pinned bytes plus the admitted expert caps on both ranks.
  At 1M, device payload capacity is `4,050,137,088` B/rank while the full host
  KV-state account is `4,082,533,760` B; admitted host residency is
  `163,253,272,284` B. A one-byte under-cap device request fails before model
  residency is staged.
- The cheap CLI `--admission-only` path cannot measure the rank-local resident
  store and initially returned a false success for that under-cap request. It
  now rejects rank-local admission-only requests explicitly. Centralized
  admission-only behavior is unchanged.

### 2026-08-13 — Step 4 production-shaped index audit opened

- At the 1M operating point, routed CPU remains `75.895 ms/token`; the corrected
  production-page index probe reports `68.482 ms/token` across 21 ratio-4
  layers. The remaining short-context dependent envelope is `51.921 ms/token`;
  terms are not added twice when the final full-chain wall is measured.
- The live long-context rank-local path is still sequential: it performs a
  host-visible preparation, synchronous index selection, and `run()` completion
  for layers 0--41, then queues only the terminal layer. Even below top-512 at
  a 1M configured capacity, Step 3's one active decode step was about 210 ms;
  that path cannot satisfy the accepted short-context schedule.
- The inherited index probe used 256 learned-index rows per descriptor. The
  production ratio-4 cache uses 64 rows, so the probe scanned the right payload
  volume with one quarter of the real descriptor count. The corrected 64-row
  probe is the binding baseline.

### 2026-08-13 — Step 4 original mechanism falsified before integration

- The corrected five-repetition probe measures `3.261 ms/indexed layer`, or
  `68.482 ms/token` across the 21 ratio-4 layers. Nsight Systems attributes
  `82.6%` of aggregate index GPU time to exact scoring. The 1M scoring call is
  `2.855265 ms`, so scoring alone is `59.960565 ms/token`; radix selection,
  uploads, launch, output, and synchronization make up the rest.
- The accepted short-context production median is `126.675822 ms/token`, which
  leaves only `0.324178 ms` beneath the user-approved 127 ms review ceiling.
  Even an ideal 2x candidate shard with a free local top-k, free exchange, and
  free exact merge leaves `59.960565 / 2 = 29.980283 ms/token` of dependent
  scoring, a lower-bound total of `156.656105 ms/token`. Real execution is
  strictly slower. Candidate sharding alone therefore cannot pass Step 4.
- Three source-local scorer screens were preserved and reverted: interleaving
  four rows per thread regressed to `72.572 ms/token`; resolving page metadata
  once per row reached `67.742 ms/token`, only `0.740 ms` below baseline and far
  short of the gate; specializing the exact 64x128 inner loop reached `68.854
  ms/token`. None changes the decision.
- The current five-step landing plan is blocked here. Full 1M support cannot be
  claimed from the accepted short-context result, and Step 5 cannot begin as a
  merge exercise. A successor plan must reduce the exact scorer by a further
  material factor and reduce the production non-CPU envelope. The old Stage-10
  `11--12.4 ms` opportunity was derived from an `84.710 ms` centralized CPU
  body; the accepted rank-local body is already `75.895 ms`, only `3.006 ms`
  above the unequal-scope external `72.889 ms` navigation value. CPU parity is
  not the missing 30 ms and assumed savings may not be stacked.

### 2026-08-13 — Step 4 envelope attributed; the missing term was not where the handoff assumed

- The handoff's "remove roughly 30 ms from the short path" was derived against
  the *current* scorer. The decisive bound is different and tighter: the index's
  non-scoring residue is `8.52 ms/token`, so **even a free scorer leaves
  `126.676 + 8.52 = 135.20 ms/token`**, above the `127.0 ms` ceiling. No scorer
  work of any size can pass Step 4 by itself; the short path must fall
  regardless. The joint requirement, with ideal 2x sharding, is
  `ΔE + ΔS/2 >= 39.3 ms`.
- Attribution used one Nsight Systems capture of the accepted production arm
  plus an unprofiled control at the same shape, so **no hot-path instrumentation
  was added** and the temporary-instrumentation ledger is unchanged. Setup was
  about 126 s initialization and 3.4 s prefill for a 1.4 s measured decode
  window per arm, roughly 5 minutes for both; decode was shortened from 32 to 12
  tokens because attribution needs steady-state proportions, not a new gate. The
  rejected cheaper alternatives were the existing aggregate `rank_local_*`
  counters, which cannot separate dependent GPU work from handoff gaps, and the
  0087 trace, which predates the accepted topology.
- Over 10 steady tokens, GPU busy union is only `36.97 ms/token` on rank 0 and
  `42.07 ms/token` on rank 1 — about 30% of the window. The idle remainder is
  dominated by the routed CPU body, correctly identified in the trace as the
  43 per-token gaps preceding `dsv4_host_moe_join_mhc_kernel`
  (`81.72` / `77.13 ms/token`), which reproduces the independently measured
  `75.895 ms` CPU term and validates the gap-attribution method.
- The envelope is therefore **not** an overlap defect: it is dependent GPU work
  serialized behind the CPU body by the layer chain, plus `8.41` / `10.57
  ms/token` of page-materialization wait and about `6 ms/token` of launch gaps.
- The binding finding is inside the GPU busy term, and it is the defect shape
  the charter names. Three attention-preparation kernels are launched
  effectively single-threaded:

```text
dsv4_query_rank_norm      <<<1, 1>>>     240.6 us   43/token   10.345 ms/token
dsv4_key_value_norm_rope  <<<1, 1>>>     124.8 us   43/token    5.368 ms/token
dsv4_query_norm_rope      <<<64, 1>>>     88.2 us   43/token    3.791 ms/token
dsv4_mhc_weighted_norm    <<<1, 64>>>     16.8 us   86/token    1.442 ms/token
```

  Together they cost more than every real matmul on the layer
  (`dsv4_rank_bf16_matmul`, `9.27 ms/token`). This is CAP-12's recorded
  `<<<1,1>>>` failure lesson repeating in the attention path: a cost that barely
  moves with the work is a constant, and a constant that large is a defect.
- The declared contract pins the *order* of the FP64 accumulation, not where its
  operands are read. Staging the dequantize/square and the scale/store phases
  across the block while leaving the ascending `__dadd_rn` chain on one thread
  reading shared memory is bit-identical by construction, and was screened
  standalone before any production edit: `5.92x`, zero bit mismatches.

### 2026-08-13 — Corrected short path passes the strict target; the 1M budget is now derivable

- The clean interleaved matrix over three repetitions gives rank-local medians
  `109.335778`, `112.268811`, `111.177092 ms/token`; their median is
  **`111.177092 ms/token` = `8.994659 token/s`**. The **strict `125.0 ms` /
  `8.0 token/s` target passes**, so this is no longer a tolerance acceptance.
  All three arms match all 32 sequential-oracle IDs exactly and all six arms
  report zero decode checkpoint I/O.
- The three corrected kernels sit in `dsv4_prepare_attention`, which both
  topologies share, so centralized was the invariant most at risk. Centralized
  generated IDs are **bit-identical** to the pre-correction `step2-acceptance-r2`
  arm across all 32 tokens, and the centralized median also improves to
  `135.012134 ms/token`.
- The permitted complete index budget, derived as the handoff requires:

```text
review ceiling                     127.000000 ms/token
strict target                      125.000000 ms/token
measured corrected short tau       111.177092 ms/token
permitted index budget (review)     15.822908 ms/token
permitted index budget (strict)     13.822908 ms/token
```

- Against that budget the measured 1M index term is `68.482 ms/token`, of which
  `59.960565` is exact scoring and about `8.52` is the non-scoring residue
  (radix select, uploads, launch, output, synchronization). With ideal 2x
  candidate sharding and a free exchange/merge, the index still costs
  `29.980283 + 8.52 = 38.50 ms/token`, which **exceeds the review budget by
  `22.68 ms/token`**. Step 4 therefore remains open.
- To fit the review budget, exact scoring must reach
  `15.822908 - 8.52 = 7.30 ms/token` sharded, so `14.61 ms/token` unsharded — a
  **`4.1x` exact scorer improvement**. The scorer's own arithmetic ceiling makes
  this a bounded engineering target rather than a wish: `9.02e10 FLOP/token`
  against the `17.79 T op/s` non-FMA FP32 issue rate of one 3090 is a
  `5.07 ms/token` floor, so the measured `59.96 ms` is `8.5%` of ceiling and the
  target `14.61 ms` is `34.7%`. The exactness contract forbids FMA contraction
  and tensor cores, so `17.79 T op/s` is the correct ceiling, not the
  `35.6 TFLOP/s` headline figure.
- Two terms in that budget remain **unmeasured** and must not be assumed zero:
  the ratio-128 dense attention growth at 1M (`8,192` rows per layer across 20
  layers) and `physical_paged_attention`'s host `locate()`, which is linear per
  candidate. Neither is charged above.
- Step 4's remaining gate is therefore a single question with a measured target:
  can exact scoring reach about `35%` of its ALU ceiling? Until that is screened
  standalone, no sharding, exchange, or merge work is justified, and the 1M
  claim stays open.

> **RETRACTED by the counter measurement later the same day.** The ALU-ceiling
> reasoning in the two entries above is wrong: with Nsight Compute counters
> enabled the corrected scorer measures `91.29%` of peak sustained throughput,
> so the binding resource was never the FP32 ALU and the implied headroom does
> not exist. The entries are retained because the arithmetic they contain is
> still the correct way to bound *work*; only the conclusion drawn about
> available headroom is withdrawn. See the entry on `locate()` and the counter
> measurement below.
- Correctness state after the correction: default `make check` 2/2, NCCL
  `make check` 2/2, `git diff --check` clean, centralized generated IDs
  bit-identical across all 32 tokens, and all three rank-local arms exact
  against the 32-token sequential oracle. **No result commit exists yet**; the
  worktree remains dirty by instruction.
- The `<<<1,1>>>` correction was deliberately scoped to the three kernels the
  trace named, so the measured delta attributes to one coherent change.

### 2026-08-13 — Remaining under-parallelized kernels: one corrected, one rejected

Re-reading the same trace by launch width rather than by cost enumerates every
remaining candidate rather than relying on the ones already noticed:

```text
kernel                    launch       threads   per token   ms/token
dsv4_mhc_weighted_norm    <<<1,64>>>        64      86          1.442
dsv4_mhc_mix              <<<1,32>>>        32      86          0.547
dsv4_mhc_fused_post_...   <<<12,256>>>    3072      85          0.441
```

- **`dsv4_mhc_mix` is rejected, not deferred.** Its cost is a 20-iteration
  dependent shuffle chain over 16 lanes — inherently serial normalization, not
  under-parallelization. At `6.4 us` per call the arithmetic is a small fraction
  of launch and dependent-shuffle latency, so widening it cannot help, and its
  iteration order is contract-bearing. Changing it would risk exactness for no
  measurable gain.
- **`dsv4_mhc_weighted_norm` was corrected.** Its FP32 reduction is *already* a
  tree — 64 accumulators each summing four blocks in order, combined by a fixed
  xor pattern — so unlike the attention norms, that shape is itself the
  contract. The correction preserves the accumulator count, the block order, the
  even-then-odd lane order and the xor tree exactly, staging the FP32 values in
  shared memory so the pinned reduction consumes identical operands, and widens
  only the two elementwise phases. Screened standalone at `2.53x` / `2.54x` with
  **zero bit mismatches on both outputs**, `1.320` to `0.522 ms/token`.
- **The end-to-end effect is not resolvable, and is not claimed as a win.** The
  corrected matrix medians are `111.895850`, `111.721523`, `111.528517 ms`,
  median `111.721523 ms` = `8.950827 token/s`. That is `+0.544431 ms` against
  the previous matrix's `111.177092 ms`, while the previous matrix's own
  repetition spread was `2.932 ms` (`109.335778`--`112.268811`). A `0.8 ms`
  predicted mechanism gain is below that variance in both directions, so this
  arm establishes **no end-to-end change**: not an improvement, and not a
  regression. The standalone screen is the evidence for the mechanism; the
  matrix's role here is to confirm exactness and that the strict gate still
  holds.
- Both matrices pass the strict `125.0 ms` / `8.0 token/s` gate, all rank-local
  arms match the 32-token sequential oracle exactly, centralized generated IDs
  remain bit-identical to the pre-correction baseline, and all arms report zero
  decode checkpoint I/O.
### 2026-08-13 — Scorer diagnosed and corrected 1.90x; gate still not met, but the bound moved 25 ms

- The declared kill criterion was `0.70 ms/indexed layer`. Scoring stands at
  `1.326 ms/layer`, so **that criterion is not met** and no 1M claim is made.
  It is not treated as closing the mission, because the criterion assumed the
  rest of the index was fixed, and the post-correction attribution has since
  named two further terms of the class already corrected twice today.
- The diagnosis falsified its own hypothesis first, which is why the correction
  worked. Query re-read traffic is `8.59 GB/layer` and implies `3.01 TB/s` at
  the measured time, which looks exactly like an L2 bound. A deliberately
  incorrect query-address-collapse variant priced it at only **32%** of the
  kernel: even eliminating it entirely leaves `44.6 ms/token`, three times the
  target. A fix aimed at query traffic alone would have missed.
- The residual is a latency-hiding defect: at full occupancy (48 warps/SM) the
  loop issues at about `0.28` instructions per cycle because each thread carries
  a single dependent `__fadd_rn` chain fed by one load per iteration. Scoring
  four rows per thread gives independent chains *and* amortizes the query load,
  adding rows to the block rather than threads so occupancy is unchanged.
- Exactness is preserved by construction: every accumulator still walks columns
  in ascending order under explicit `__fmul_rn`/`__fadd_rn`, and only which
  thread computes a row changed. The 1M scalar-oracle gate passes at the full
  candidate width.
- Post-correction attribution names the remaining terms:

```text
score kernel      1.326 ms/layer   69.1% of index GPU   27.85 ms/token
pivot kernel      0.142 ms/layer   22.2%                 2.98 ms/token
other select      0.083 ms/layer                         1.75 ms/token
outside (launch)  0.129 ms/layer                         2.71 ms/token
```

  `dsv4_physical_index_pivot_kernel` is launched `<<<1, 1024>>>` — one block on
  an 82-SM GPU — and is now the second largest index term. Its own source
  comment records that a `<<<1,1>>>` scan of 65,536 bins was the original
  defect; the fix widened the block but never the grid.
- Standing projection, with ideal 2x sharding and a still-free exchange/merge:

```text
corrected short path                        111.177092 ms/token
sharded index projection                     20.520000 ms/token
projected tau(1M)                           131.700000 ms/token
review ceiling                              127.000000 ms/token
remaining gap                                 4.700000 ms/token
```

  Against the handoff's `156.656105 ms/token` bound this is `25 ms` better, but
  it remains above the ceiling and is a **projection, not a measurement**. The
  ratio-128 attention growth, the linear host `locate()`, and the exchange and
  deterministic merge are still unmeasured and are charged nowhere above.

### 2026-08-13 — Pivot corrected 3.01x; index now 2.01x total, projection 3.1 ms over the ceiling

- `dsv4_physical_index_pivot_kernel` had two defects, and because bin counts are
  integers neither carried an exactness constraint — histogram sums are
  order-independent, so only access patterns and parallelism were in question.
  - The group scan read `histogram[thread * 64 + index]`, so a warp touched 32
    separate 256 B regions and each 4 B request still pulled a 32 B sector:
    `8x` read amplification, turning 256 KiB into about 2 MiB of traffic. A
    group is 64 contiguous uint32, so it is now read as 16 `uint4`, cutting
    amplification to `2x`.
  - The final walk ran up to 64 **dependent global loads** on one thread. It is
    inherently sequential — it stops at the first bin reaching the quota — but
    the group's 64 bins are now staged coalesced into shared first, so the walk
    chases shared memory instead of global.
- Result: pivot `47.288` to `15.728 us` (`3.01x`), from `22.2%` to `8.6%` of
  index GPU time, `2.98` to `0.99 ms/token`. The probe delta of `1.95 ms/token`
  matches the kernel delta of `1.99 ms/token`, which confirms the mechanism
  rather than assuming it.
- **A defect I introduced, and the gate caught it.** The `uint4` read requires
  16 B alignment; the histogram workspace region was reserved at
  `alignof(std::uint32_t)`. The misaligned vector loads produced a sticky CUDA
  error and the suite fell to 278/300 — 21 failures from one root cause,
  including a *valid* request being rejected. Reserving the region at
  `alignof(uint4)` restored 299/300 with the one declared skip and preserved the
  full timing win. Recorded because the failure mode is instructive: a vector
  load added to a kernel silently inherits whatever alignment its allocator
  happened to give, and the symptom appeared far from the cause.
- Cumulative index result and the standing projection:

```text
complete index, baseline                     68.482000 ms/token
complete index, scorer + pivot corrected     34.092000 ms/token   2.01x

sharded projection (scoring halved, pivot fixed-cost, exchange/merge free)
  scoring / 2                                13.940000 ms/token
  pivot, unchanged at 65,536 bins per rank     0.990000 ms/token
  histogram and remaining select, halved       1.250000 ms/token
  outside (host launch and synchronization)    2.730000 ms/token
                                             ---------------------
  sharded index projection                    18.910000 ms/token
  corrected short path                       111.177092 ms/token
  projected tau(1M)                          130.087092 ms/token
  review ceiling                             127.000000 ms/token
  remaining gap                                3.087092 ms/token
```

- The gap is now `3.09 ms/token`, from `29.66` at the handoff. The largest named
  remaining term is the `2.73 ms/token` of host launch and synchronization,
  which Step 4 must remove anyway by moving selection inside the queued chain —
  so the remaining arithmetic is no longer obviously short. It is still a
  **projection**: exchange, deterministic merge, the ratio-128 attention growth
  and the linear host `locate()` remain unmeasured and are charged nowhere.

### 2026-08-13 — The unmeasured term dominates everything: host `locate()` is 2,890 ms/token at 1M

This is the binding Step 4 result, and it invalidates the projection above.

- `physical_paged_attention`'s host `locate()` runs `std::find_if` over the
  whole per-layer block table for **every attended candidate**. Rescue-brief
  defect 8 recorded it as `O(candidates x blocks)` and never measured. Measured
  now, with the production geometry (`256/ratio` rows per block,
  `block_table(sequence, kind, layer)` so the table is per layer):

```text
                     blocks   candidates    ms/layer
ratio-128 layer       4,096        8,192     135.369
ratio-4 layer         4,096          512       8.679

per decoded token, single threaded
  20 ratio-128 layers                       2707.371 ms
  21 ratio-4 layers                          182.266 ms
  total                                     2889.637 ms/token
```

- **`2,890 ms/token` is 22x the entire decode budget.** Every 1M projection in
  this document charged this term at zero. The `130.087092 ms/token` figure is
  therefore not a near miss on the ceiling; it omits a term twenty-two times
  larger than the whole budget, and no 1M claim of any kind survives it.
- The order of work today is vindicated by this and not by the kernel wins: the
  short-path corrections and the `2.01x` index improvement were real and exact,
  but continuing to optimize kernels toward a `3.09 ms` projected gap would have
  been tuning against a number whose omitted term was three orders of magnitude
  larger. The stopping rule — *stop when the remaining gap is smaller than the
  uncertainty of the estimate it is measured against* — fired correctly.
- A fix is available and is verified equivalent. The table is ordered by
  `logical_begin`, so binary search reaches the same block with the same
  predicate: **zero index mismatches** across every dense and sparse candidate,
  at `130x`:

```text
linear scan (current)                       2889.637 ms/token
binary search                                 22.206 ms/token   130x
```

- Even corrected, `22.206 ms/token` still exceeds the entire permitted index
  budget of `15.8--18.9 ms/token`, so binary search is necessary but not
  sufficient. Because blocks are uniform (`256/ratio` rows each, all full except
  the last), the owning block is directly computable as
  `logical_row / rows_per_block`, which is `O(1)` and should reduce this to well
  under a millisecond. That variant is **not yet measured** and its uniformity
  assumption is not yet verified against a table that has seen eviction or
  reuse.
- A second defect is visible in the same code and is **not** included in any
  figure above: `kv_cache->block_table(...)` returns a `std::vector` **by
  value**, so a 4,096-element table of `Dsv4KvBlockInfo` — each carrying its own
  `std::vector<bool> device_resident` — is constructed per layer per token. At
  1M that is about `176,000` block copies and their heap allocations per decoded
  token, on the timed path, which also contradicts the steady-state
  zero-allocation invariant. Unmeasured; recorded as an open defect.

### 2026-08-13 — `locate()` fixed, and counters correct the scorer headroom claim

- `locate_physical_kv_block()` replaces the linear scan at both call sites. It
  tries a direct index from the first block's geometry, then an ordered binary
  search, then the original exhaustive scan, validating the same predicate on
  every path — so the block returned is the one the scan would have found, and a
  table reordered by eviction or reuse still resolves correctly. The 12-token
  arm is exact against all 12 sequential-oracle IDs; the suite is 299/300 with
  the one declared skip; the steady median is `112.53 ms/token` against the
  pre-fix `111.90`, which is within the arm-to-arm spread and expected, since at
  4,096 context the tables are a few blocks and there was nothing to save.
- **Nsight Compute counter access has been enabled on this host**, so the
  scorer's real behaviour is now measured rather than inferred:

```text
sm__throughput (pct of peak sustained)                  91.29%
gpu__compute_memory_throughput (pct of peak sustained)  91.29%
achieved occupancy                                      93.38%
instructions per cycle per scheduler                     0.62
L1 sector hit rate                                      78.47%
```

- This **corrects an earlier claim in this document**. The scorer was described
  as sitting at `34.7%` of an exactness-constrained ALU ceiling, with a `4.1x`
  further improvement therefore plausible. Measured, the corrected kernel runs
  at `91.29%` of peak sustained throughput on both the SM and the combined
  compute/memory pipe: the binding resource was never the FP32 ALU, so that
  ceiling was the wrong comparison and the headroom it implied does not exist.
  Further scoring gains would require reducing the work, not the inefficiency,
  and the exactness contract fixes the work.
- Consequence for the 1M question: exact scoring is effectively at its floor for
  this algorithm at about `27.9 ms/token` single-rank, or about `13.9` sharded.
  With the pivot corrected and the remaining select and launch terms, the
  sharded index lands near `18.9 ms/token` against a permitted budget of
  `15.8 ms`. The gap is real, small, and no longer attackable by tuning this
  kernel.

### 2026-08-13 — `block_table()` copy removed; the timed-path allocation is gone

- `Dsv4KvCache::block_table()` returned `std::vector<Dsv4KvBlockInfo>` **by
  value** and is called once per kind per layer per decoded token. At 1M a
  compressed table holds 4,096 blocks, so a token materialized roughly
  `176,000` block infos. Worse, each copy carried a
  `std::vector<bool> device_resident` that the call populated per block and that
  **nothing in the tree ever read** — a heap allocation per block for dead data.
- Measured on the real structure (hash map of block state, per-table id vector,
  the same copy loop): `11.0--13.6 ms/token` by value against `2.9--3.2` with a
  caller-owned buffer and the dead field gone, a `3.8--4.2x` reduction worth
  `8.1--10.4 ms/token`. Against a permitted index budget of `15.8 ms/token`
  that is not a rounding term.
- Fixed by removing the dead field, adding
  `Dsv4KvCache::block_table_into(..., std::vector<Dsv4KvBlockInfo>& output)`
  which clears and refills caller storage, and giving the runtime
  process-lifetime buffers for the sliding, compressed and learned-index
  tables. All four hot call sites use it; `block_table()` remains as a thin
  wrapper for setup-time callers. Only one layer's tables are live at a time,
  so three buffers suffice.
- This also repairs a **steady-state contract violation** independent of speed:
  the decode window is required to perform zero timed-path allocations, and the
  by-value table allocated on every call.
- Exactness: the 12-token arm matches all 12 sequential-oracle IDs, the suite is
  299/300 with the one declared skip. The steady median of `113.68 ms/token`
  sits inside the `111.90`/`112.53` spread of the preceding arms — neutral at
  4,096 context, as expected, since the tables there are a handful of blocks
  rather than 4,096. The gain is a long-context gain and is claimed only there.

### 2026-08-13 — Binding final acceptance on the complete corrected code

Re-run from scratch after every Step 4 correction, rather than carried over from
an earlier arm, because the preceding matrix predated the `locate()` and
`block_table()` changes and those had only been spot-checked at 12 tokens.

```text
rank-local repetition medians   111.496762  114.787396  110.820323 ms
rank-local median               111.496762 ms = 8.968870 token/s
centralized repetition medians  133.919461  128.710203  130.545682 ms
centralized median              130.545682 ms
strict 125.0 ms / 8.0 token/s gate                        PASS
```

Correctness and residency at the same arms:

- all three rank-local arms bit-exact against the 32-token sequential oracle;
- centralized generated IDs bit-identical to the pre-correction
  `step2-acceptance-r2` baseline, so the shared `dsv4_prepare_attention`
  kernels remain neutral for the default topology;
- zero decode checkpoint I/O in all six arms;
- VRAM `22,419,734,528` / `22,417,637,376` B/device against the
  `22,548,578,304` ceiling, headroom `128,843,776` / `130,940,928` B;
- peak RSS `158,350,274,560` B against `231,928,233,984`, headroom
  `73,577,959,424` B;
- `scripts/audit_dsv4_extraction_manifest.sh` PASS, 91 paths, 13/13
  capabilities.

**Step 4's short-context half is complete.** The strict target is met on merit
at `8.968870 token/s`, not admitted by the `127 ms` tolerance boundary, and the
result rests on one matrix over the final code rather than on a chain of arms.

The full-context half remains open and unclaimed. What is left there is
integration — dual-rank candidate sharding, the exchange, a deterministic merge,
and selection inside the queued chain — plus three terms that are still
unmeasured and charged nowhere: the exchange, the merge, and the ratio-128 dense
attention growth. Every defect discoverable by reading and measuring the
existing path has now been corrected.

### Step 5 prerequisites now due

- `decode_step_seconds` and its JSON surface are marked **ACTIVE TEMPORARY;
  remove after the final performance acceptance run** in the instrumentation
  ledger. That run is the matrix above, so the trigger has fired: this surface
  must not appear in the promotion diff.
- The extraction manifest classifies `include/strata/deepseek_kv_cache.hpp` and
  `src/deepseek_kv_cache.cpp` as *excluded* on the grounds that the `a31ac58`
  delta on them is replay instrumentation. That classification still holds --
  it describes the experiment delta, not this branch -- but the landing now
  modifies those files for its own reasons (`block_table_into`, removal of the
  dead `device_resident` field) and the manifest should say so.
- **No result commit exists.** The entire landing, Steps 1--4, is uncommitted
  worktree state.

- The binding short-path figure for budget derivation stays the better-supported
  `111.177092 ms/token` from the larger-spread matrix only if a single number is
  needed; taking the two matrices together the corrected short path is about
  `111.2--111.7 ms/token`, and the permitted index budget is about
  `15.3--15.8 ms/token`. The `4.1x` exact-scorer requirement is unchanged by
  this arm.
