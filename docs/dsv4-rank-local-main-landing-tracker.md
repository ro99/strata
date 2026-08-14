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

State: **OPEN — INDEX MECHANISM BUILT, EXACT AND IN-CHAIN; FULL-CONTEXT CLAIM
BLOCKED BY A CAPABILITY CEILING AT 65,536 TOKENS IN DENSE COMPRESSED ATTENTION**

The index work required by items 1 and 3 below is done, exact and measured. The
blocker is elsewhere and was found on 2026-08-14: both device attention entry
points reject more than 640 candidates, and the 20 ratio-128 layers need
`roundup(context/128, 128) + 128` — `8,320` at the declared context. The
supported context ceiling of the device attention path is therefore `65,536`
tokens, and no `tau(1M)` can be measured until that is lifted. Item 4 below is
blocked on a kernel design task, not on tuning. The candidate-sharding
mechanism in item 2 is separately obviated: in-chain selection removed the host
boundary that made it look attractive, and both ranks now select concurrently.

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

Rollback condition — **FIRED, and honoured.** Full-context support is not
claimed anywhere in this landing. The supported context ceiling is 65,536
tokens; above it the ratio-128 layers exceed the attention kernel's
640-candidate bound and the step fails closed (experiment 0093, issue #22).
Short-context performance is reported as short-context performance.

The original rollback text follows, and its condition also fired for the
original mechanism:
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

State: **IN PROGRESS — unblocked by explicit user decision on 2026-08-14**

Step 4's remaining exit criterion is an end-to-end `tau(1M)`, which is not
reachable on this machine: prefill at that context is prohibitively slow and is
being addressed as separate work, and independently of that the ratio-128
attention path cannot execute above 65,536 tokens at all (experiment 0093,
issue #22). The user's decision is that this does not block Step 5, and that the
landing proceeds against a **supported context ceiling of 65,536 tokens** rather
than the declared 1,048,576. Nothing in this landing claims full-context
throughput, and the architecture document now opens its long-context section
with that bound.

Progress:

```text
1  centralized bit-identity against main    PASSED   five oracles, both builds
2  gates                                    partial  suite green in the default
                                                     tree; NCCL tree and the
                                                     remaining gates pending
3  diagnostic surface                       RESOLVED three ledger rows closed
4  documents match what runs                partial  architecture, manifest and
                                                     experiment 0093 done
5  promotion diff review                    partial
```

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
| 2026-08-13 | Step 4 sparse-path below-threshold negative | `results/dsv4-rank-local-main-landing/step4-sparse-path/centralized.json` | `prompt_tokens` `1,593`, active `1,605` | preserved: a `9,155`-character prompt tokenized below the `2,048` threshold, so the arm exercised nothing new. Caught by checking `prompt_tokens`, the check 0087's INVALID N4 arm failed |
| 2026-08-13 | **Step 4 first real sparse decode** | `results/dsv4-rank-local-main-landing/step4-sparse-path/` | prompt SHA256 `0c7b142e7c24f9d1990ac98ebf11839abd141a342fd87b999bca0bbf1d9bb88f`; centralized `014aa827326144ab8c89d87409cc219773caabc93199e19d5ef675f695b035e2`; rank-local `5e64228e0dc04cba52592d5acba51b3892ecda4dba0ab462f2787239f00caa04` | closes rescue-brief defect 9 at this context: `231` CUDA index dispatches, **`0` scalar**, exactly `118,272 = 231 x 512` selected, identical answer text, zero decode checkpoint I/O on both topologies |
| 2026-08-13 | Step 4 sequential-arm penalty | same arms | centralized `157.603`, rank-local `174.285 ms/token` at `2,685` active tokens | rank-local is `16.682 ms/token` **slower** than centralized above the threshold, and `62.788 ms` above its own queued short-context median: the measured value of moving selection inside the queued chain |
| 2026-08-13 | Step 4 below-threshold queued control | `results/dsv4-rank-local-main-landing/step4-sparse-path/rank-local-below.json` | `2,014` active tokens, `0` index dispatches, steady median `109.612 ms/token` | removes the context-growth confound: the queued path is context-insensitive from ~50 to 2,014 active tokens, so about `61` of the `64.673 ms` sequential-arm gap is topology, not indexing |
| 2026-08-14 | Step 4 Stage 2a positional page indexing | `results/dsv4-rank-local-main-landing/step4-sparse-path/rank-local-stage2a2.json` | token IDs and answer identical; `161.338` against Stage 1's `158.353 ms/token` | page index now equals block-table index, the precondition for device-side candidate resolution. `+2.985 ms/token` here, but the same code extrapolates to `1.11 s/token` at the declared context, so the page array must be cached across tokens before Stage 2 goes to 1M |
| 2026-08-14 | **Step 4 Stage 1 device index-query preparation** | `results/dsv4-rank-local-main-landing/step4-sparse-path/rank-local-stage1.json`, `centralized-stage1.json` | rank-local `174.285` to `158.353`; centralized `157.603` to `142.777 ms/token`; **selection trace hashes identical on both** (`f14e5cde177258bd`, `d7e791756a7355d2`) | RoPE, bf16 rounding and half-up E4M3 moved onto the device in one kernel. Token IDs and answer text identical on both topologies; `231` CUDA and `0` scalar dispatches; zero decode checkpoint I/O. Removes about 172,000 host `log2`/`exp2` calls per token. **Both topologies gain about 15 ms, so the relative gap is unchanged**: rank-local is still `15.576 ms/token` slower above the threshold |
| 2026-08-14 | Step 4 B3 E4M3 query-quantization screen | `results/dsv4-rank-local-main-landing/step4-inchain-selection/e4m3_screen.cu`, `e4m3-screen.txt` | **0 mismatches in 4,000,663 probes** on both architectures, after a first form disagreed on 32 | the backend's existing `quantize_e4m3_value` rounds ties to even and cannot serve the half-up index-query contract; and the exponent must come from `log2f`, not `frexpf`, to reproduce the reference at power-of-two boundaries |
| 2026-08-14 | Suspected reference defect: E4M3 half-up at power-of-two boundaries | same screen | host encodes `0.0312499963` as `2^-6`, a factor-of-two error | `log2f` rounding to exactly `-5.0` sends the host into its sub-1 mantissa branch spuriously. Reproduced deliberately for exactness; **not fixed here**, since changing it silently would alter selection. Reachable at roughly `1e-5` of values. Needs its own branch and review |
| 2026-08-14 | Step 4 A3 end-to-end exactness arm | `results/dsv4-rank-local-main-landing/step4-sparse-path/rank-local-a3.json` | token IDs and answer text **identical** to the pre-A3 baseline; `174.422` against `174.285 ms/token`; `231` CUDA and `0` scalar index dispatches; zero decode checkpoint I/O | the optional device-selection entry point does not disturb the live path: with the parameter null the behaviour is unchanged end to end, not merely in the unit gates |
| 2026-08-13 | Step 4 in-chain selection A1+A2 screen | `results/dsv4-rank-local-main-landing/step4-inchain-selection/` | 3 geometries, 516 probes each, **0 block and 0 row mismatches**, both architectures | device candidate resolution reproduces the host `locate_physical_kv_block` exactly, including short final blocks and block-boundary probes; the subtle half of in-chain selection is de-risked before any runtime change |
| 2026-08-13 | **Step 4 binding final acceptance** | `results/dsv4-rank-local-main-landing/step4-final-acceptance/summary.json` | SHA256 `9dc391c245e93d49f536088d62dcaf146471f0abd28c1b67e040ca0a3c1e8285`; rank medians `111.496762`, `114.787396`, `110.820323` ms; median **`111.496762` ms = `8.968870` token/s**; central medians `133.919461`, `128.710203`, `130.545682` ms | binding short-context result on the **complete corrected code**, run after every Step 4 correction rather than assembled from earlier arms. Strict `125.0` ms / `8.0` token/s gate **PASSES**. All three rank arms bit-exact against the 32-token sequential oracle; centralized IDs bit-identical to the pre-correction `step2-acceptance-r2` baseline; zero decode checkpoint I/O in all six arms |
| 2026-08-13 | Step 4 memory gates at final acceptance | same summary | VRAM `22,419,734,528` / `22,417,637,376` B/device against the `22,548,578,304` ceiling; peak RSS `158,350,274,560` B against `231,928,233,984` | Step 3 residency gates re-verified on the corrected code: headroom `128,843,776` / `130,940,928` B/device and `73,577,959,424` B host |
| 2026-08-13 | Step 4 extraction audit | `scripts/audit_dsv4_extraction_manifest.sh` | PASS, 91 paths, 13/13 capabilities | traceability gate still green after the Step 4 corrections |
| 2026-08-13 | Step 4 gates after `block_table()` fix | `results/dsv4-rank-local-main-landing/step4-make-check-default-r6.log` | default 2/2 pass exit 0; NCCL suite 299/300 with one declared skip; `git diff --check` clean | final gate state for the Step 4 correction set |
| 2026-08-13 | Step 4 scorer counters (ncu now permitted) | `results/dsv4-rank-local-main-landing/step4-scorer-diagnosis/ncu-score-counters.txt` | `sm__throughput` and `gpu__compute_memory_throughput` both **91.29% of peak sustained**; achieved occupancy `93.38%`; IPC `0.62`/scheduler; L1 sector hit rate `78.47%` | **corrects this document**: the corrected scorer is near-saturated, so the earlier "`34.7%` of ALU ceiling, `4.1x` therefore plausible" framing was wrong — the binding resource is the combined compute/memory pipe, not the FP32 ALU, and little headroom remains without an algorithm change the exactness contract forbids |
| 2026-08-13 | Step 4 host `locate()` measurement | `results/dsv4-rank-local-main-landing/step4-locate-screen/` | linear scan `2889.637 ms/token` at 1M; binary search `22.206 ms/token`, `130x`, **0 index mismatches** | **binding negative**: the term every 1M projection charged at zero is 22x the entire decode budget. No 1M throughput claim survives it. Binary search is verified equivalent but still exceeds the index budget; an `O(1)` direct index is available and unmeasured |
| 2026-08-13 | Step 4 Nsight Compute availability | `results/dsv4-rank-local-main-landing/step4-scorer-diagnosis/ncu-score-PERMISSION-DENIED.log` | `ERR_NVGPUCTRPERM` on device 1 | correction to the record: Nsight Compute 2025.1.1 **is** installed and recognizes every device; counter *collection* is refused by the driver permission setting, which is a different thing. The scorer diagnosis' hand-derived rates are inferences, not counter measurements, and are labelled as such |
| 2026-08-13 | Step 4 pivot alignment defect | full `strata-tests` run before the alignment fix | 278/300, 21 failures incl. `physical Lightning Indexer rejects malformed pages and shapes` at `tests/test_cuda_backend.cpp:504` | **defect I introduced and caught**: the pivot's new `uint4` group read needs 16 B alignment, but the histogram workspace region was reserved with `alignof(std::uint32_t)`. Misaligned vector loads left a sticky CUDA error, so 21 failures traced to one root cause. Fixed by reserving the region at `alignof(uint4)` |
| 2026-08-13 | Step 4 pivot correction | `results/dsv4-rank-local-main-landing/step4-scorer-diagnosis/pivot-fixed-binding.txt` | SHA256 `0eceffc332f340ce2dfe12c87f35c03673927ffccabca825107b9dfd438f834a`; complete index `36.062` to `34.092 ms/token`; pivot kernel `47.288` to `15.728 us` (`3.01x`) | pivot drops from 22.2% to 8.6% of index GPU time; all four indexer exactness tests pass and the suite returns to 299/300 with one declared opt-in skip |
| 2026-08-14 | Step 4 Stage 2 gate A: device index projections | `results/dsv4-rank-local-main-landing/step4-sparse-path/rank-local-gateA.json`, `centralized-gateA.json` | SHA256 `7b42280b537fa5bb28abbb81ad14f05748ca6fa1176cd8fd53939af5796ad4a8` / `1f04d79202dc21eb3a363c7554604e6e2064e4a3b155b0b6ecf56ae557587e31`; rank-local `159.673` to `156.080`, centralized `142.777` to `140.610 ms/token` | both index projections moved onto the device against the preparation command's own activations. Selection trace hashes **identical** on both topologies (`f14e5cde177258bd`, `d7e791756a7355d2`), token IDs and answer text identical, `231` CUDA and `0` scalar dispatches, zero decode checkpoint I/O. Exact by construction: the same kernel `matmul` would dispatch, over the activation its own upload would have produced |
| 2026-08-14 | Step 4 Stage 2 gate B: device candidate resolution | `results/dsv4-rank-local-main-landing/step4-sparse-path/centralized-gateB.json` | SHA256 `4d03784b3683f9422e8a6025e8fa234aa0adc7b27420c3d423fa161878db0c61`; `142.339 ms/token` | the resolve kernel, the positional block table and the device-candidate attention input, gated on the centralized arm where the selection is still host-visible and everything else is held constant. Identical IDs, answer and selection trace. **No timing claim**: four single centralized arms of this workload span `140.610`--`143.908 ms/token`, so the `+1.729` against gate A is inside the spread |
| 2026-08-14 | **Step 4 Stage 2 gate C: in-chain selection** | `results/dsv4-rank-local-main-landing/step4-sparse-path/rank-local-gateC5.json` | SHA256 `17df56adb0b8def079c5a505428d564527f140be820ba6101c3f55571436a0f8`; steady median **`115.795 ms/token`** against Stage 1's `159.673` (`-43.878`) | selection runs inside the queued chain: projection, score, top-k and candidate resolution are enqueued in each layer's own command sequence and no indexed layer leaves the chain. **Generated token IDs and answer text identical** to the sequential baseline; `462` CUDA and `0` scalar index dispatches (both ranks select on their own device); zero decode checkpoint I/O; decode synchronization calls `1,462` to `748`; D2H `27.96` to `19.85 MB`. VRAM `22,419,734,528` B/device against the `22,548,578,304` ceiling and RSS `158,365,499,392` B |
| 2026-08-14 | Step 4 Stage 2 final state after cleanup | `results/dsv4-rank-local-main-landing/step4-sparse-path/rank-local-final.json`, `centralized-final.json` | rank-local SHA256 `3a783c8adac28202593e4ee4b31f621ec643c21bcbea885f01fb24ae08602fba`; rank-local `118.168`, centralized `143.908 ms/token` | run after removing the centralized device-resolution wiring and restricting the queued path to positions where every indexed layer actually selects. Both arms exact: identical token IDs and answer text, `0` scalar dispatches, zero decode checkpoint I/O. The rank-local spread across the two in-chain arms is `115.795`--`118.168`, so **the binding claim is the range, not either endpoint**; a three-repetition acceptance matrix is still owed |
| 2026-08-13 | Step 4 gates after mHC correction | `results/dsv4-rank-local-main-landing/step4-make-check-default-r3.log`, `step4-make-check-nccl-r2.log` | default 2/2 pass exit 0; NCCL 2/2 pass exit 0; `git diff --check` clean | both builds re-verified with all four corrected kernels |
| 2026-08-13 | Step 4 mHC acceptance | `results/dsv4-rank-local-main-landing/step4-mhc-acceptance/summary.json` | rank medians `111.895850`, `111.721523`, `111.528517` ms; median `111.721523` ms = `8.950827` token/s; central median `133.867678` ms | strict gate still **PASSES**; oracle-exact; centralized IDs still bit-identical; the `+0.544 ms` against the previous matrix is **inside** that matrix's own `2.932 ms` repetition spread, so **no end-to-end change is claimed in either direction** |
| 2026-08-13 | Step 4 mHC mechanism screen | `results/dsv4-rank-local-main-landing/step4-norm-correction-r1/mhc_screen.cu` and `mhc-screen-device0.txt` | device 1 `15.35` to `6.07 us` (`2.53x`); device 0 `15.47` to `6.09 us` (`2.54x`); **0 bit mismatches on both outputs** | `dsv4_mhc_weighted_norm`'s xor reduction shape is itself the contract, so the 64 accumulators and their combination order are preserved and only the elementwise phases widen; `1.320` to `0.522 ms/token` |
| 2026-08-13 | Step 4 corrected acceptance | `results/dsv4-rank-local-main-landing/step4-norm-acceptance/summary.json` | rank medians `109.335778`, `112.268811`, `111.177092` ms; median `111.177092` ms = `8.994659` token/s; central medians `135.012134`, `134.005181`, `137.349264` ms | **strict `125.0` ms / `8.0` token/s gate PASSES**; all three rank arms match all 32 sequential-oracle IDs; centralized IDs bit-identical to the pre-correction arm across all 32 tokens; zero decode checkpoint I/O in all six arms |
| 2026-08-14 | **Step 4 ratio-128 attention growth** | `results/dsv4-rank-local-main-landing/step4-ratio128-attention-growth/probe-r1.txt`, `-r2`, `-r3` | SHA256 `8ac542e7a96ad85a113f56fff00f75039e9a6cc01efc882f4986e84e4b7504e6` / `64171af7208db0888323d071434f58c5ba52d00af9ce5c5ff9ec6534f709a60d` / `9bee399706a5e452d71d4696b0fbc4a25a80b488ce7a17dc3e3975d8dcaf396d`; three interleaved runs of nine repetitions, device 1, Release `build-landing`, idle GPU | the context-dependent attention term measured across every candidate width the kernel accepts, and **rejected at every width above it**. Kernel medians `0.043`/`0.049`/`0.056`/`0.064 ms` per layer at `256`/`384`/`512`/`640` candidates; `1,152`, `2,176` and `8,320` candidates are refused. The 1M requirement is `8,320`. A fourth run taken while `ctest` held the same device was **discarded**, not averaged: it broke monotonicity at 512 candidates |

| 2026-08-14 | **Step 5 centralized parity against `main`** | `results/dsv4-rank-local-main-landing/step5-centralized-main-parity/main.json`, `branch.json`, `nccl.json` | SHA256 `6943a15bdd766ae3465c3ed0be1a0ef18e4ef1bee5d71d0f8b3ce8675507a6c7` / `9477b7e3e9bd2281fca12add5a997183ffb465a6143163e1aa4e89259bdb1faf` / `6c1b9798e50ed8b4b89d244571b05c24598075b375b6f9440f315529429d6a54` | **GATE PASSED.** Centralized decode is bit-identical to `main` (`61f1f02`, built Release from a clean worktree) on all five oracles: logit trace hash `3efb25b705d23937` across all three arms, all 32 per-forward top-k records identical, 32 generated token IDs identical, 18 prompt token IDs identical, 129-character answer identical. Run on the NCCL-off **and** NCCL-on branch builds, so the NCCL build does not perturb the default path either. Only flags present on both revisions were used |

| 2026-08-14 | **Step 5 final production acceptance** | `results/dsv4-rank-local-main-landing/step5-final-acceptance/summary.json` | SHA256 `88f0084e2998fdde69b1ef0d1a15dab14561985d64018f9a8b2f4b3f320d1d9e`; rank medians `111.941600`, `111.564517`, `113.387087` ms; median `111.941600` ms = **`8.933229 token/s`**; centralized medians `128.346237`, `128.938956`, `129.426872` ms | **strict `125.0` ms / `8.0` token/s gate PASSES** on the complete landed code, three interleaved pairs, all six arms exit 0. Rank-local exact against the 32-token sequential oracle; zero decode checkpoint I/O in all six arms; VRAM `22,419,734,528` and `22,417,637,376` B against the `22,548,578,304` ceiling; RSS `158,351,544,320` B against the `231,928,233,984` ceiling. Rank-local repetition spread is `1.823 ms`, so the `17.0 ms` advantage over centralized is well outside it |

| 2026-08-14 | Step 5 promotion diff review | `git diff main...HEAD` | 72 files, `+23,201 / -548`; runtime surface 34 files, `+11,584 / -547` | **no unrelated change found.** Every peripheral edit is additive and preserves its default path: the `HostWorkerPool` CPU-list constructor is inert when the affinity list is empty; `read_slice_into` is new; the explicit VRAM budget reduces to the historical `free * fraction` product when unset; `block_table()` is retained as a wrapper over `block_table_into()`. The two non-`dsv4`-named additions in `backend.cu` are a new `download_buffer` (absent on `main`) and one kernel template. `flash_attention` is **byte-identical** between revisions — its apparent diff is displacement by inserted code, confirmed by hashing the extracted function on both. The one behavioural correction outside the rank-local mechanism is `dsv4_physical_kv_admission`, which sized compressed blocks with the sliding layout's block bytes and count; that is the Step 3 admission fix, not an unrelated change |

| 2026-08-14 | **Step 5 binding acceptance, on the shipped code** | `results/dsv4-rank-local-main-landing/step5-final-acceptance-r2/summary.json` | SHA256 `90246cca7dbc06dd6f5bab74b45ddb3c221bae6d92bbc540542a457266f57b06`; rank medians `112.099790`, `110.286939`, `110.610722` ms; median `110.610722` ms = **`9.040715 token/s`**; centralized medians `128.916082`, `135.364031`, `129.416257` ms | re-run **after** the instrumentation resolution, so the accepting artifact certifies the code that lands rather than its predecessor. Strict `125.0` ms / `8.0` token/s gate **PASSES**; oracle-exact; deterministic within each topology; zero decode checkpoint I/O in all six arms; VRAM `22,419,734,528` / `22,417,637,376` B and RSS `158,346,399,744` B, all unchanged from the pre-removal matrix. Against that matrix's `111.941600` the difference is `-1.331 ms`, **inside** both matrices' own repetition spreads (`1.823` and `1.813 ms`), so **no performance change is claimed for the instrumentation resolution in either direction**. Every arm satisfied the runner's `decode_step_seconds == 31` validation, which is what proves the `--detailed-timing` gating is wired correctly |

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
| Per-step decode wall vector and JSON `decode_step_seconds` | **RESOLVED 2026-08-14: emission gated behind `--detailed-timing`; not in the default output** | the final acceptance run is complete, so the removal condition has fired. Removing it outright would delete the branch's own regression gate: `run_dsv4_rank_local_acceptance.sh`, `run_dsv4_rank_local_sparse_gate.sh`, `run_dsv4_rank_local_step4_envelope_profile.sh` and `compare_dsv4_sparse_gate.py` all derive the steady median from it, and a landing that deletes the gate proving it is worse than one that keeps the gate behind a flag. `--detailed-timing` is the pre-existing diagnostic flag eight other scripts already use for exactly this, so the default production output no longer carries the surface, which is what this row required. Capture cost is one `steady_clock` read per decoded token into storage reserved before timing |
| Aggregate rank-local phase clocks and public `rank_local_*_nanoseconds` / `Dsv4HostMoePhaseTimings` surfaces | **RESOLVED 2026-08-14: KEEP, under this row's own "unless explicitly justified" clause** | the justification is parity with an existing supported contract, not utility. `main` already exposes **22** `*_nanoseconds` phase counters in this same metrics struct — including `routed_gate_up_nanoseconds`, `routed_down_nanoseconds` and `routed_reduce_nanoseconds`, which are precisely the CPU-MoE phases `Dsv4HostMoePhaseTimings` reports for the rank-local path. Removing the rank-local counters would leave the rank-local topology reporting zeros where the centralized topology reports real numbers, degrading a surface `main` already ships rather than declining to add one. Measured cost: about 500 `steady_clock` reads per token across 43 layers, roughly `12 us` against a `111.9 ms` token, or `0.01%` — inside the `1.823 ms` repetition spread of the accepting matrix by two orders of magnitude |
| `CudaDsv4AttentionPrepareRequest::host_callback_wait_event` and its two backend handling sites | **REMOVED 2026-08-14** | an optional API capability that nothing ever set. This branch has already paid for an untested API path once: the physical indexer sized its bounded workspace from its request spans, so a device-projected call reserved zero bytes for the query and its staging overwrote the page descriptors — a defect that existed only because that path had no production caller. Unused surface is removed rather than carried |
| Rank-local memory/admission JSON ledger (`rank_local_memory`) | **KEEP** | supported setup-time fail-closed admission evidence: it records initial VRAM, measured resident rank-local weights, the applied centralized expert-cache cap, and admitted device/host totals without adding a hot-path clock or hash |
| Per-layer KV-patch, attention query/KV, FFN-input, and route hashes added to the existing layer trace | **REMOVED FROM THE CURRENT WORKTREE; DO NOT LAND** | correctness discriminator only; the saved raw artifacts retain the evidence without extending production output or hot-path work |
| Temporary sequential rank-local execution selected by `--layer-hash-trace` | **REMOVED FROM THE CURRENT WORKTREE; DO NOT LAND** | the 14-token queued/sequential comparison passed and the complete 32-token sequential oracle was captured; production diagnostics now use the normal queued schedule |
| Read `chain_cpu_moe_phases` after both rank streams finish | **KEEP ONLY IF the aggregate phase telemetry survives Step 5; otherwise remove with it** | if the counters remain, they must be sampled only after their callback owners finish; this fixes telemetry correctness, not model execution |
| Layer-qualified physical/rank-local callback and KV-account errors | **KEEP** | fail-closed diagnostics run only on an error path and localize the owning layer/rank |
| CUDA `host_callback_wait_event` API at the old page-publication boundary | **CURRENTLY HAS NO CALLER; REMOVE IN STEP 5 unless a supported path is proven to require it** | the corrected replica path uses its page-ready ownership event instead; an unused synchronization API must not land as residue |
| Detailed-event arm with an invalid mHC interval | **DO NOT USE OR EXTEND** | rejected measurement evidence; no counter derived from the invalid interval may support acceptance |
| In-chain resolution status bits and the numeric attention status in the rank-local chain error | **KEEP** | fail-closed diagnostics on an error path only. The three causes an in-chain resolution can raise are otherwise indistinguishable from a downstream decode failure, and the command has no host boundary of its own to report at |
| `scripts/run_dsv4_rank_local_sparse_gate.sh` and `scripts/compare_dsv4_sparse_gate.py` | **KEEP** | the Stage 2 gate itself, reusable and deterministic; no production surface and no hot-path work |
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

### 2026-08-13 — The sparse indexer path has now run inside a real decode

Rescue-brief defect 9 recorded that the sparse indexer had **never run inside a
real decode**: every end-to-end arm in the program's history sat below the
`2,048`-token engagement threshold (`index_topk * ratio`), so the entire
selection path was unexercised on both topologies. It has now run on both.

Reaching it does not require the 1M context. `active_context_tokens` is
`prompt_tokens + max_new`, so a prompt above about `2,036` tokens engages the
indexer, which costs minutes rather than the roughly two days that prefilling
1,048,576 tokens at the measured `128 ms/token` would take.

A first arm at `9,155` characters tokenized to only `1,593` tokens, leaving
active context at `1,605` — **below** the threshold, so it exercised nothing new.
It was caught by checking `prompt_tokens` in the output, which is the same check
experiment 0087's INVALID N4 arm failed. The binding arms use a `15,413`
character prompt giving `2,673` prompt tokens and `2,685` active.

```text
                                centralized      rank-local
steady median                   157.603 ms       174.285 ms
index CUDA dispatches                  231             231     = 21 layers x 11 steps
index scalar dispatches                  0               0     no host fallback
index selected                     118,272         118,272     = 231 x 512 exactly
index candidates                   154,623         154,623     = 231 x ~669 = L/4
index selection entries             56,364          56,364
selection trace hash        d7e791756a7355d2  f14e5cde177258bd
decode checkpoint bytes                  0               0
generated answer text                        identical
```

- **The mechanism works.** Selection runs on `dsv4_physical_lightning_index`
  with **zero scalar dispatches**, selects exactly `512` of `~669` candidates
  per indexed layer, and both topologies produce identical answer text with zero
  decode checkpoint I/O.
- The selection trace hashes differ between topologies and are **expected** to.
  Selection depends on each layer's query, and the accepted rank-local BF16
  publication association differs from centralized from layer 0 onward
  (0073, 0088). The entry counts match exactly, which is the comparable
  invariant.
- **Rank-local is currently `16.682 ms/token` slower than centralized above the
  threshold.** This is the known structural gap, now measured rather than
  asserted: above `2,048` active tokens the runtime leaves the queued 43-layer
  chain and drives layers sequentially, because `Dsv4RankLocalLayerCall` takes
  candidates as an input and a chain cannot select candidates for a layer whose
  query it has not yet computed.
- **The sequential arm costs `62.788 ms/token`** against the queued short-context
  median of `111.497`. That is the value of moving selection inside the queued
  chain: it is not an optimization but the precondition for rank-local being
  worth selecting at all at long context, and it is now a measured quantity
  rather than an estimate.

### 2026-08-13 — The sequential-arm penalty attributed: about 61 ms is topology, not indexing

The first `62.788 ms/token` figure was confounded: it compared the queued arm at
about 50 active tokens against the sequential arm at 2,685, mixing topology,
index work and context growth. A third arm just **below** the engagement
threshold separates them.

```text
arm                                        active   indexer   ms/token
rank-local queued                           2,014       off    109.612
rank-local sequential                       2,685        on    174.285
centralized                                 2,685        on    157.603
rank-local queued, short context               50       off    111.497
```

- The queued arm at `2,014` active tokens is indistinguishable from the same arm
  at about 50 tokens (`109.612` against `111.497`). **The queued path is
  context-insensitive across that range**, which is what the cost model predicts:
  the routed CPU body is a function of the model, not of `L`.
- Index compute at this width is about `3.5 ms/token` by the probe
  (`0.165 ms/indexed layer` at 4,096 context across 21 layers).
- Therefore of the `64.673 ms` gap between the queued and sequential arms,
  roughly **`61 ms/token` is the sequential topology itself** and only about
  `3.5 ms` is the indexing it was introduced to serve.
- Target for in-chain selection, stated before the work: rank-local at `2,685`
  active tokens should land near `113 ms/token`, against `174.285` today and
  `157.603` for centralized. Anything that does not materially beat the
  centralized arm at this operating point has not earned the topology.

### 2026-08-13 — In-chain selection: the design, from reading the data flow

Recorded before implementation so it does not have to be re-derived. Three facts
from the current code decide the shape:

1. **The attention kernel already reads device-resident candidates.**
   `dsv4_paged_attention_to_mhc` stages `CudaDsv4AttentionCandidate` entries into
   a pinned buffer and uploads them to `workspace + candidate_offset` as
   `Dsv4DeviceAttentionCandidate`. Candidates merely *arrive* there by host
   upload. **The attention kernel itself does not need to change** -- something
   else must write that array device-side.
2. **The index call is host-synchronizing by construction.**
   `dsv4_physical_lightning_index` ends with `cudaStreamSynchronize` plus a D2H
   of the winners into a host span. That is the host round trip to remove.
3. **The page mapping is host work.** `locate()` resolves a selected logical row
   to `{page, row}`, where `page` indexes a *compacted* list built lazily in
   first-touch order over the selected candidates.

Fact 3 is the real constraint. Device-side selection cannot build a compacted
list, because the selection is not known when the list would be built. The
resolution is to index the **full** per-layer block list instead: at 1M that is
4,096 page descriptors of 16 B, about 64 KiB per layer and 1.3 MB per token
across the 21 indexed layers, roughly 0.1 ms at link speed and cacheable while
the block table is stable. The attention kernel indexes a larger array and is
otherwise unchanged.

The work therefore decomposes into four pieces, in dependency order:

```text
A1  device page-descriptor table: {logical_begin, used_rows, buffer, capacity}
    for every block of a layer, uploaded once per layer and reused
A2  resolve kernel: selected logical row -> Dsv4DeviceAttentionCandidate,
    reproducing locate_physical_kv_block's predicate on device
A3  non-synchronizing index entry point leaving winners on device
A4  an optional device-candidate input on CudaDsv4PagedAttentionRequest that
    skips host staging when the candidates are already resident
```

A1+A2 carry the subtle correctness risk -- the resolved `{page, row}` must equal
what the host produces -- and are verifiable in isolation against the host path
without touching the chain. A3+A4 are plumbing that only pays off once A1+A2 are
exact. Only after all four does the selection move inside the queued chain,
which is where the measured `61 ms/token` is recovered.

Note that the `index_selections` trace hash is computed host-side from the
downloaded positions; once selection stays on device that diagnostic requires an
explicit download and must become opt-in rather than always-on.

**A1+A2 are screened and exact.** Before any runtime change, the device
resolution was verified against the host `locate_physical_kv_block` plus the
host candidate arithmetic, over three production geometries and including the
boundary probes an off-by-one would hit -- row 0, the last row, and both sides
of a block boundary -- with a deliberately short final block:

```text
ratio-4   @ 2,685        11 blocks   516 probes   0 block, 0 row mismatches
ratio-4   @ 1,048,576  4,096 blocks  516 probes   0 block, 0 row mismatches
ratio-128 @ 1,048,576  4,096 blocks  516 probes   0 block, 0 row mismatches
```

Reproduced on both GPU architectures present. The screen is retained at
`results/dsv4-rank-local-main-landing/step4-inchain-selection/`.

**A3 is implemented and gated.** `dsv4_physical_lightning_index` takes an
optional `CudaDsv4DeviceIndexSelection*`. With it null the behaviour is exactly
as before -- download the positions, synchronize -- so every existing caller and
the scalar-oracle gate are untouched. With it non-null the kernels are left
enqueued, `output` may be empty, and the caller receives device pointers to the
winners and the error flag, with the selection counters still recorded. The
host-visible tail that a queued chain cannot afford per indexed layer is
precisely what that branch skips.

The device pointers reference the backend's persistent per-device Lightning
workspace, so they are valid only until the next selection on that device. That
is sufficient for an in-chain consumer, which reads them in stream order before
the following layer's selection overwrites them, and the contract is stated on
the struct.

Gate: NCCL suite 299/300 with the one declared skip, zero failures. The index
probe measures `1.544--1.549 ms/indexed layer` in this session state against the
`1.623` recorded earlier; the default path is behaviour-unchanged and
oracle-verified, so this is **not** a performance effect of the edit and is not
claimed as one. The earlier figure was taken immediately after sustained
acceptance-matrix load and this one after an idle period, which is the likely
difference. The binding pivot-fix comparison stays with its own
same-session baseline.

### The A1--A4 decomposition was incomplete; corrected scope

Reading `index_select` before implementing A4 shows the decomposition above
understated the work, and the correction is worth recording precisely.

A4 assumed the index *queries* were already device-resident, so that only the
selection and its candidate resolution had to move. They are not. Everything the
selection consumes is computed on the host from `query_rank`, and `linear()` is
`weights->matmul()` taking **host spans in and out**:

```text
index_select, per indexed layer, all host-visible today
  wq_b projection          query_rank[1024] -> queries[64 x 128]
  per-head RoPE on the trailing 64 dims, then round_bf16
  dsv4_physical_quantize_query_e4m3_f32 on each head
  weights_proj projection  input[4096] -> index_weights[64]
  scale by 1/sqrt(head_dim * heads), then round_bf16
  -> CudaDsv4PhysicalIndexRequest{queries, weights, pages}
```

A queued chain cannot perform any of that: `query_rank` and `input` exist only
on the device mid-chain, and the CUDA host callback where they would otherwise
be visible is forbidden from calling CUDA APIs. There is also **no partial
win** available -- keeping the query projection on the host requires the very
synchronization the change exists to remove, so the pipeline moves device-side
in full or not at all.

Corrected decomposition, with status:

```text
B1  device wq_b index-query projection                        NOT NEEDED for stage 1: host matmul retained
B2  device RoPE + round_bf16 on the query tail                DONE AND GATED, trace-hash identical
B3  device E4M3 query quantization                            DONE AND GATED, trace-hash identical
B4  device weights_proj projection, scale, round_bf16         NOT NEEDED for stage 1: host matmul retained
B5  device block-descriptor table                             SCREENED (A1)
B6  resolve kernel, selected row -> attention candidate       SCREENED (A2)
B7  non-synchronizing index entry point                       DONE AND GATED (A3)
B8  attention integration and runtime wiring                  TO DO
```

### Stage 1 opened: every B1--B4 exactness question is now resolved

Stage 1's risk was never the plumbing, it was whether four host computations can
be reproduced bit-exactly on the device. Each is now either screened or exact by
construction, and the reasoning is recorded because it is not re-derivable from
the code alone.

- **B1 and B4 carry no exactness risk at all.** `linear()` is
  `weights->matmul()` is `backend_.matmul()`, and `matmul_impl` already runs the
  projection as a **device kernel**, uploading the input and downloading the
  output. The device-resident form runs the *same* kernel and simply does not
  download, so the arithmetic is identical by construction rather than by
  comparison. B4 adds a scale multiply and a `round_bf16`.
- **B2's application arithmetic is safe, and this had to be checked.**
  `apply_rope` computes `first * cosine - second * sine` with plain operators.
  Had the host contracted that into an FMA, a non-contracted device form would
  differ. A discriminating probe -- values where a fused and an unfused
  evaluation genuinely disagree -- shows the host does **not** contract at
  `-O3`:

```text
separate   0                 <- what the host actually produces
contracted 1.42108547e-14    <- what an FMA would produce
host contracts into FMA: NO
```

  So the existing `dsv4_rope_first`/`dsv4_rope_second`, which use explicit
  `__fmul_rn`/`__fsub_rn`/`__fadd_rn`, already match.
- **B2 needs no device trigonometry**, per the note above: the cosines and sines
  are computed host-side from `position` and the layer frequencies and uploaded,
  exactly as `dsv4_prepare_attention` already does for the attention query.
- **B3 is screened** at 4,000,663 probes, and **`round_bf16`**, shared by B1, B2
  and B4, at 15,728,728.

The consequence is that Stage 1 has no remaining unknown arithmetic. What is
left is mechanical: port the screened E4M3 kernel, add the RoPE-plus-round
kernel over uploaded trig, add device-resident forms of the existing matmul that
skip the download, and let `index_select` consume them while still
synchronizing. Its gate is unchanged -- the 2,685-token arm must reproduce the
recorded baseline exactly.

### 2026-08-14 — Stage 1 built and passed, and it is not performance-neutral

The device index-query preparation is implemented and live for PhysicalDevice
KV: `CudaBackend::dsv4_index_query_rope_quantize` applies the RoPE, the bf16
rounding of the rotated region, and the half-up E4M3 quantization in one kernel,
one block per index head. `index_select` uses it in place of its per-head host
loop; the selection still synchronizes, so the topology is untouched.

The gate is the strongest available, and it passes:

```text
generated token IDs            identical to the pre-Stage-1 baseline
answer text                    identical
index selection trace hash     f14e5cde177258bd  ==  f14e5cde177258bd
index selection entries        56,364
index CUDA / scalar dispatches 231 / 0
decode checkpoint bytes        0
NCCL suite                     299/300, one declared skip
```

The trace hash is the decisive one: it folds every selected position across all
56,364 selection entries, so an identical hash means the device pipeline chose
bit-identical candidates to the host it replaced. That is what the screens
predicted -- B3 exact by measurement, B2 exact by construction with uploaded
trigonometry and non-contracted rotation -- confirmed in production.

**Stage 1 was expected to be performance-neutral and is not:**

```text
rank-local, host index-query preparation     174.285 ms/token
rank-local, device index-query preparation   158.353 ms/token
centralized, same operating point            157.603 ms/token
```

`-15.932 ms/token`. The removed work is larger than it looks: the host loop ran
64 heads x 128 dimensions of E4M3 encoding per indexed layer, and that encoding
calls `log2` and `exp2` per element -- about **172,000 transcendental calls per
decoded token** across the 21 indexed layers, on the critical path, plus the
rotation and rounding. None of it was visible as a kernel or a transfer, which
is why the earlier Nsight attribution of the *short-context* envelope never
showed it: at 4,096 context the indexer does not engage at all.

This closes most of the gap to centralized at this operating point without any
topology change: rank-local moves from `16.682 ms/token` slower to `0.750 ms`
slower. Stage 2, which keeps the queued chain, is still where the remaining
`~45 ms` sits.

`index_select` is shared by both topologies, so centralized above the threshold
should benefit identically. **Measured, and it does** -- which corrects the
comparison above:

```text
                      before Stage 1   after Stage 1     delta
centralized             157.603          142.777       -14.825
rank-local              174.285          158.353       -15.932
rank-local minus centralized                   +15.576
```

Centralized generated the identical tokens and the identical selection trace
hash `d7e791756a7355d2`, so its gain is the same removal of host transcendental
work, not a behaviour change.

**The earlier reading of "rank-local is now within 0.750 ms of centralized" was
wrong**, because it compared a Stage-1 rank-local arm against a pre-Stage-1
centralized baseline. Measured on the same build, rank-local remains
`15.576 ms/token` slower above the threshold -- essentially the `16.682 ms` gap
it had before, since both topologies gained about the same `15 ms`. Stage 1 made
both faster; it did not change their relative standing, and the sequential
topology is still the whole of the difference.

The Stage 2 target is correspondingly restated: rank-local should reach about
`113 ms/token` -- the `109.612` queued arm plus roughly `3.5 ms` of index work --
which would be about `30 ms/token` **faster** than centralized at this operating
point rather than `15.6` slower. That is the number Stage 2 has to earn.

### 2026-08-14 — Stage 2a: positional page indexing works, and exposes a scaling trap

Stage 2's first obstacle is not a kernel. The host assigns a candidate's page
index lazily, in **first-touch order** over the selected rows. Device-side
selection cannot reproduce that: it does not know which blocks a candidate set
will touch, or in what order. The mapping has to become **positional** -- page
index equals block-table index -- which means every attendable block must have a
page before selection is known.

That change is now in and gated. Both arms produce identical token IDs and
identical answer text at 2,685 active tokens, so the numbering scheme is
behaviour-preserving:

```text
lazy first-touch page indexing (Stage 1)      158.353 ms/token
positional, leasing the whole table           167.013 ms/token   +8.660
positional, leasing only in-use blocks        161.338 ms/token   +2.985
```

The first attempt leased every block the table held, which is sized for the
configured context rather than the live one -- 32 blocks where 11 hold
attendable rows. Bounding it to blocks below `compressed_count` recovers most of
the cost.

**The residual `+2.985 ms/token` does not transfer, and that is the finding.**
It is the price of leasing all blocks rather than only selected ones, and it
scales with the block count:

```text
implied cost per lease                              6.46 us
blocks per indexed layer at 2,685 active              11
blocks per indexed layer at 1,048,576 active       4,096
naive pre-lease cost at the declared context    52.9 ms/layer, 1.11 s/token
```

A second of per-token leasing would dwarf every other term in this program. So
positional indexing as implemented is correct but **must not be carried to the
declared context in this form**. The page array for a layer changes only when a
new block is appended -- once per 256 source tokens at ratio 4 -- so it has to be
built once and reused across tokens, rebuilt only when the block count changes,
rather than reconstructed per token per layer.

That caching is now a prerequisite of Stage 2 rather than an optimization of it,
and it was invisible at the short-context operating point where the same code
costs three milliseconds.

> **RETRACTED the same day by the experiment it motivated.** The prefix cache
> was built, gated -- token IDs and answer text identical -- and measured at
> `161.399 ms/token` against the uncached `161.338`. It removed `-0.061 ms`,
> which is nothing. If the `+2.985 ms` had been per-lease cost, caching the
> leases across tokens would have removed nearly all of it. **The per-lease
> attribution is therefore false, and the `1.11 s/token` extrapolation built on
> it is withdrawn.**
>
> The `+2.985 ms` is real and reproducible but **not yet attributed**. It is not
> repeated leasing. A plausible remaining cause is that positional numbering
> changes the order in which the attention kernel walks its pages, where the
> lazy scheme happened to number them in selection order; that is a hypothesis
> and is recorded as one, not as a finding.
>
> **Why the cache did nothing, and the completed attribution.**
> `rank_local_prepare_layer` clears the layer's leases before
> `rank_local_candidates` runs, and for a real reason: a block that still holds
> a lease cannot receive the next token's appended row. The cache was therefore
> defeated upstream and never hit, which is why it removed nothing. It has been
> removed.
>
> The discriminating experiment is positional numbering with *lazy* leasing,
> which is also the better design -- an unselected block is never leased:
>
> ```text
> stage 1, lazy numbering and lazy leasing     158.353 ms/token
> positional numbering, pre-leasing            161.338 ms/token   +2.985
> positional numbering, lazy leasing           159.673 ms/token   +1.320
> ```
>
> So `1.665 ms` was leasing blocks the selection never touched, now eliminated,
> and `1.320 ms` is the positional numbering itself. The residual is small and
> **still unattributed**; the leading hypothesis is that the page-descriptor
> array's shape and traversal order change what the attention kernel walks, and
> it is recorded as a hypothesis.
>
> Lazy leasing also disposes of the 1M scaling worry on its own terms: a layer
> leases the blocks its 512 selected rows touch, not all 4,096.

### How B8 should be sequenced so it stays gateable

B8 cannot be screened in isolation -- it changes what the attention kernel
reads -- and it also cannot be reached without B1, B2 and B4, because the
selection consumes index queries that are host-computed today. Taken together
that is roughly 700 lines across `kernels/cuda/backend.cu`,
`include/strata/cuda_backend.hpp`, `src/deepseek_runtime.cpp` and the executor,
behind a **single** end-to-end gate of about nine minutes per attempt. Written
in one step it has no intermediate checkpoint, which is the shape that produced
the `uint4` alignment defect earlier in this step, where the symptom surfaced 21
tests away from its cause.

There is a decomposition that keeps a gate at every stage, and it should be
preferred even though it runs the work twice:

```text
stage 1  build the device pipeline -- B1 B2 B3 B4 B5 B6 -- and use it INSIDE
         the existing host index_select, still synchronizing at the end.
         No performance win; the topology is untouched.
         GATE: the 2,685-token arm must reproduce the recorded baseline
         exactly -- identical token IDs and answer text, 231 CUDA and 0 scalar
         dispatches. This validates every new kernel in production against a
         known-good result, with the sequential topology holding everything
         else constant.

stage 2  remove the host synchronization and let the queued chain keep all 43
         layers above the threshold, using B7's device-selection entry point.
         GATE: the same arm, now expected near 113 ms/token rather than
         174.285, still bit-exact.
```

Stage 1 converts "did I write six kernels correctly" into a question the arm can
answer, while stage 2 isolates the topology change to a diff that touches
scheduling alone. Attempting both at once conflates a correctness failure with a
scheduling failure, and the arm cannot distinguish them.

B1--B4 each need a device kernel that is **bit-identical** to its host
reference, not merely close: the selection is a hard top-k, so a single
differing low bit in a query can change which rows are attended. Each is
screenable in isolation against the host function the same way B5 and B6 were,
which is the cheap way to build them.

### B3 screened, and two findings that would each have been silent bugs

**The existing device quantizer cannot be reused.** `quantize_e4m3_value()` in
the backend uses `frexpf` plus `rintf`, which rounds ties to **even**. The
index-query contract is `encode_e4m3_half_up`, `floor(x * 8 + 0.5)`, which
rounds ties **up**. Reusing the existing kernel would have changed which
candidates a hard top-k selects, with no error raised anywhere.

**The obvious exponent extraction is also wrong here.** A first device form took
the exponent from `frexpf`, which is the mathematically exact binade. It
disagreed with the host on **32 of 4,000,663 probes**, all immediately below a
power of two:

```text
value 0.0312499963  (just below 2^-5)     host 0x08     device 0x10
```

The cause is in the reference, not the port. Host `log2f` rounds that value to
exactly `-5.0`, so `floor` yields `-5`, the mantissa becomes `0.99999988` which
is `< 1`, and the host falls into its **sub-1 mantissa branch spuriously**,
encoding a value of `0.03125` as `2^-6 = 0.015625` -- a factor-of-two error.
Taking the exponent from `log2f` on the device as well reproduces it exactly:

```text
4,000,663 probes, boundaries + half-up ties + saturation + random
encode mismatches vs host reference : 0     on both GPU architectures
```

### B1, B2 and B4 are routing, not new exact arithmetic

Reading the host references changes what these three cost. None of them needs a
newly invented exact kernel; each reduces to reusing arithmetic that already
exists, which makes exactness true **by construction** rather than by
comparison. That is a materially smaller and safer job than the earlier note
implied.

- **B2 needs no device trigonometry.** `apply_rope` calls `std::cos`/`std::sin`,
  and device `cosf`/`sinf` would differ in the last ulp. But the existing design
  already avoids this: `dsv4_prepare_attention` takes `rope_cosines` and
  `rope_sines` as *inputs*, computed host-side into `scratch.cosines` /
  `scratch.inverse_sines`. The RoPE angles depend only on `position` and the
  layer frequencies, both known before the chain is enqueued, so the index
  query's cosines and sines can be precomputed and uploaded the same way. What
  remains is applying them, and `dsv4_rope_first`/`dsv4_rope_second` already do
  that with explicit `__fmul_rn`/`__fadd_rn`/`__fsub_rn`, so no fma contraction
  can occur.
- **B1 and B4 are the same matmul, routed device-to-device.** `linear()` is
  `weights->matmul()`, which is `backend_.matmul()` -- already a device kernel --
  followed by a **host-side** `round_bf16`. The device-resident form must run
  the *same* kernel rather than a differently tiled one, since a different
  accumulation order would change low bits; exposing a device-to-device variant
  of that exact kernel makes the projection bit-identical by construction. B4
  adds only a scale multiply and a rounding step.

**The shared rounding step is screened.** All three end in `round_bf16`, host
bit-manipulation RNE against device `__float2bfloat16_rn`:

```text
15,728,728 probes: exhaustive low-bit sweeps across 60 exponents, exact
half-ulp ties, zero, denormal, infinity and NaN
round_bf16 mismatches : 0     on both GPU architectures
```

So of the four pieces called "exact-critical", only B3 required genuinely new
exact arithmetic, and it is done. B1, B2 and B4 now carry a proof obligation --
route the same kernel, upload the same trig -- rather than an invention.

**This is recorded as a suspected latent defect in the reference**, not fixed
here. Exactness against the declared scalar reference is the binding contract,
so the device path must reproduce the quirk; changing it would be a silent
semantics change of exactly the kind the invariants forbid. It is reachable in
production -- roughly 1 in 10^5 values by the random sweep, so on the order of
one or two of the ~172,000 index-query elements per token -- and the current
host path already exhibits it, so reproducing it preserves today's behaviour
rather than introducing anything. It deserves its own review on a separate
branch.

This does not change the value of the work -- the measured prize is still about
`61 ms/token` -- but it does change its size, and the estimate that in-chain
selection was "four pieces, two of them screened" was wrong.

### 2026-08-14 — Stage 2 built in three gated steps

The handoff said items 3 to 6 have no intermediate gate and that the only
signal is an end-to-end arm reporting "the answer text differs" without
localizing the cause. Two of the three risks turned out to be separable, and
separating them cost two arms and repaid both.

```text
gate A  both index projections onto the device, selection still host-visible
        rank-local  159.673 -> 156.080   trace f14e5cde177258bd unchanged
        centralized 142.777 -> 140.610   trace d7e791756a7355d2 unchanged

gate B  device candidate resolution, positional pages, device-candidate
        attention input -- gated on the centralized arm, where the selection
        is still host-visible and only the consumer changes
        centralized 140.610 -> 142.339   same IDs, answer and trace
```

Gate A is exact **by construction rather than by comparison**, and that is the
reason to prefer it over a hand-written kernel: `linear()` is
`CudaBackend::matmul` plus a host `round_bf16`, so the device form routes the
same kernel `matmul` would dispatch for that weight, over an activation already
in the state `matmul`'s own upload would have produced. Two encodings appear
here and each pairs with exactly one activation state -- FP8 with the
E4M3-quantized query rank, plain BF16 with the raw expanded layer input --
so the backend requires the encoding rather than dispatching over it. Feeding
one to the other's kernel would be silently wrong rather than rejected.

Gate B established exactness, not a cost. It read `+1.729 ms/token` against
gate A, and **that number is not attributable**: four single centralized arms
of the identical workload span `140.610` to `143.908 ms/token`, so any
per-arm difference of this size is inside the spread. It is recorded as
unresolved at n=1 rather than as the price of anything.

What gate B did establish is structural. **First-touch leasing and device
selection are mutually exclusive**, which the lazy-leasing commit earlier in
this step did not anticipate: the host leases a page when a selected row first
names it, and a selection the host never sees has no first touch. An unleased
slot in a positional page array is an empty page, and the command would name
it. So an in-chain layer leases every attendable block -- 11 per indexed layer
here, 4,096 at the declared context, which is the open question recorded
below.

### 2026-08-14 — Rank 1 owns no compressor, and that broke the first in-chain arm

The first end-to-end in-chain arm failed closed with "CUDA index projections
have no prepared attention source". The cause is a deliberate asymmetry: the
compressor is **resident on rank 0 alone**, because it is one logical
replicated KV stream whose encoded row is copied to both ranks, and loading a
second identical set would add about 0.6 GiB for nothing. The expanded BF16
layer input the index weight projection reads was computed only as a side
effect of a compressor projection, so on rank 1 it did not exist.

Rank 1 still has to select. The preparation now expands its layer input on
request, which is one 4,096-element widening, and the request names why.

This is the shape of defect the gate decomposition was meant to expose: it is a
residency asymmetry, not a kernel error, and an undecomposed arm would have
reported it as the same "the answer differs" that a wrong candidate set does.

### 2026-08-14 — A workspace sized from its request spans, and an enqueue-only path that had no caller

The second in-chain arm failed differently: the chain ran, and both ranks
reported a non-zero attention status. The status word is the attention
command's own, carried verbatim into the mHC workspace, so the fix was to make
the in-chain resolution raise **distinct bits** for its three causes -- the
selection rejected its own input, a selected row is owned by no block, a
resolved candidate falls outside its page -- and to report the numeric status
rather than the fact of failure. That diagnostic is fail-closed-path only and
stays.

The cause was found by inspection before the instrumented arm returned. The
physical Lightning Indexer sizes its bounded workspace from the *request
spans*:

```cpp
!checked_bytes(request.queries.size(), 1U, sizeof(float), query_bytes) ||
!checked_bytes(request.weights.size(), 1U, sizeof(float), weight_bytes) ||
```

A device-projected call carries no host spans by construction, so both regions
were **zero bytes**, and the transposed query staging wrote 8,192 floats over
the page descriptors, scores and keys that follow it. Sizing from the shape
rather than from the spans is the fix.

This is a defect in a path that until now had **no production caller**: the
enqueue-only selection form was committed with the device-selection entry point
and gated only by a unit test that never exercised device projection. An
unused API is not a tested one.

**Recorded, not yet acted on:** the same enqueue-only path uploads its page
descriptors from a `std::vector` local to the call. For pageable host memory
CUDA stages the copy before returning, so this is correct, but the driver may
also synchronize the stream to do it -- which in a queued chain is exactly the
serialization the chain exists to remove. If the in-chain timing disappoints,
measure this before designing anything else.

### 2026-08-14 — Stage 2 closes: the topology is finally worth its cost above the threshold

```text
must not exceed                          158.353 ms/token   (Stage 1 bar)
target                                   about 113
measured, in-chain selection             115.795 and 118.168
centralized, four single arms            140.610 to 143.908
below-threshold queued arm               109.612
```

The two in-chain arms differ by `2.373 ms` and the four centralized arms of an
unchanged workload span `3.298 ms`, so single arms resolve about `3 ms` here and
no smaller difference in this document is a result. The `-43.9` is four times
larger than that; the `+1.7` gate B read is not.

Token IDs and answer text are **identical** to the sequential baseline, so the
gain is entirely topology and not a different candidate set. Rank-local
above the threshold moves from `15.576 ms/token` **slower** than centralized to
about `25 ms` **faster**, which is the standing the queued chain always had
below the threshold and could not carry across it.

The mechanism shows in two counters rather than one: decode synchronization
calls fall from `1,462` to `748`, and decode D2H from `27.96` to `19.85 MB`.
Both ranks now select on their own device -- `462` CUDA dispatches where the
sequential arm issued `231` -- and that duplication is deliberate. The two
ranks read identical inputs with deterministic kernels, so they agree by
construction; the work is concurrent on two devices, and no exchange or merge
enters the critical path. **The candidate-sharding mechanism this step
falsified in April is not needed at all**: what the chain wanted was not a
cheaper score but no host boundary.

The residual against the `109.612` below-threshold arm is about `6`--`9
ms/token` for 21 indexed layers, which is the index work itself plus the
pre-leasing the device selection requires. It is quoted as a range because a
single arm does not resolve better than that here.

**What this does not establish.** Every number here is at 2,685 active tokens.
The leasing term scales with the block count and the score term with the
candidate count, and one extrapolation in this document has already been
withdrawn for exactly that reason. Two things must be measured before any 1M
claim: what pre-leasing every attendable block costs at 4,096 blocks per
indexed layer, and whether the enqueue-only page-descriptor upload synchronizes
the stream.

### 2026-08-14 — The ratio-128 term is not slow at 1M, it is unimplemented at 1M

The remaining unmeasured attention term was the 20 ratio-128 layers, the only
part of attention whose width depends on context. Reading the runtime's
candidate assembly first (`attention_physical`, `deepseek_runtime.cpp:3552`)
bounds the question before any measurement:

```text
ratio 0    layers   0 compressed + 128 sliding                     = 128    fixed
ratio 4    layers   512 index top-k + 128 sliding                  = 640    fixed
ratio 128  layers   roundup(context/128, 128) + 128 sliding        = grows
```

`sliding_window` is **128**, not 2,048 — an early note in this session read the
contract's positional field order wrongly and is corrected here. So ratio-4 and
ratio-0 attention are already at their 1,048,576-token width in every
short-context arm ever run on this branch, and the entire context-dependent
attention term is the 20 ratio-128 layers.

`apps/strata_dsv4_attention_probe.cpp` walks that term across every candidate
width the kernel accepts, at the production page geometry (`Hca`, 2 rows per
block, 584 B per row) and the production candidate layout:

Medians of three interleaved runs of nine repetitions on an idle device:

```text
context    comp rows   cands   pages   ms/layer   ms/token   kernel ms   status
    4,096         32     256      17      0.159      3.188       0.043   ok
   16,384        128     256      65      0.158      3.167       0.043   ok
   32,768        256     384     129      0.170      3.400       0.049   ok
   49,152        384     512     193      0.177      3.537       0.056   ok
   65,536        512     640     257      0.186      3.729       0.064   ok
  131,072      1,024   1,152     513          -          -           -   REJECTED
  262,144      2,048   2,176   1,025          -          -           -   REJECTED
1,048,576      8,192   8,320   4,097          -          -           -   REJECTED
```

The three runs agree to `0.001`--`0.002 ms` on every kernel cell. A fourth run
was taken while `ctest` held the same device and is **discarded rather than
averaged in**: it reported `0.083 ms` at 512 candidates against `0.067` at 640,
which is not a monotone function of candidate count and so is contention, not
signal. It is recorded here because the mistake was mine and the shape of the
error — a non-monotone cell in a term that must be monotone — is the cheapest
way to catch it.

Tracked as [issue #22](https://github.com/ro99/strata/issues/22).

**The finding is the last three rows, not the first five.** `dsv4_paged_attention`
and `dsv4_paged_attention_to_mhc` both validate `candidates > 640U` as invalid
(`backend.cu:7545` and `backend.cu:9278`). A ratio-128 layer exceeds 640
candidates as soon as `compressed_width > 512`, that is as soon as context
exceeds `512 x 128 = 65,536` tokens. The first five rows are the control that
identifies the cause: the same queries, sinks, scale and entry point pass at
640 and fail at 1,152, and 1,152 is a multiple of the required 128, so the
candidate cap is the only clause that can fire.

A second ceiling sits behind the first. The kernel's KV staging is
`candidates * 512 * sizeof(uint16)`, and production sets
`maximum_workspace_bytes = 4 MiB` (`deepseek_runtime.cpp:3606`). At 640
candidates that staging is `655 KB`; at 8,320 it is `8.52 MB`. Lifting the
candidate cap alone would move the rejection, not remove it.

**Therefore the declared 1,048,576-token context is not currently executable at
all**, on either topology, for reasons that have nothing to do with the index,
the chain, or rank-local decode. The supported context ceiling of the device
attention path is `65,536` tokens. This is a capability bound, and it fails
closed, which is the contract behaving correctly — but no throughput number for
1M can exist until it is lifted.

What the growth would cost if it were lifted, stated as an extrapolation and
not a measurement: the kernel term is linear in candidates to within the
resolution of the sweep. Fitting the endpoints gives `5.469e-5 ms/candidate`
with a `0.029 ms` intercept, which predicts `0.0500` and `0.0570` at the two
interior widths against `0.049` and `0.056` measured. At 8,320 candidates that
is `0.484 ms/layer` of kernel, about `9.7 ms/token` over 20 layers against the
`0.86 ms/token` at the current gate arm's 256 — a growth of about
`+8.8 ms/token`.

**This is a 13x extrapolation beyond the measured range and is not a result.**
Two documented extrapolations in this file have already been withdrawn, and a
linear fit is exactly what would fail here: at 8,320 candidates the kernel's KV
staging no longer fits the workspace it is written against, so whatever runs at
that width will not be this kernel unmodified.

Two cautions on reading the table:

- The `ms/layer` column is a **standalone** entry-point call and includes a
  host round trip the production fused and queued paths do not pay. Roughly
  `0.12`--`0.13 ms` of every row is outside the kernel. The `kernel ms` column
  is the part that transfers.
- Page count grows with context too (`4,097` descriptors at 1M against `257` at
  65,536), and the descriptor upload is inside the untimed part of that
  overhead. It is charged nowhere in the extrapolation above.

**A reporting defect rides along with it.** Nothing between the configured
`maximum_context_tokens` and the attention call bounds the context against 640
candidates, so a run at, say, 100,000 tokens does not fail admission — it
reaches the first ratio-128 layer and returns *"DeepSeek paged attention
request shape, BF16 query, scale, or sink is invalid"*. That fails closed,
which is the contract, but it names the query and the scale for what is
actually an unsupported context length. Whatever lifts the ceiling should also
make the bound explicit where the context is accepted.

Consequence for the standing 1M arithmetic, with each term's provenance:

```text
below-threshold queued base, measured at ~2,000 tokens   109.612   does not describe 1M
1M in-chain index, measured at 1M page geometry           +31.362   34.092 less the 2.730 host term
ratio-128 growth, extrapolated 13x beyond measurement      +8.8     not a result
                                                        ---------
                                                          149.8
review ceiling                                            127.000
```

Pre-leasing at 4,096 blocks per layer, the descriptor upload, and the growth of
the base terms themselves are all still charged nowhere.

## Stage 3 handoff

Stage 2 is closed: in-chain selection is committed, exact and measured at
`115.795 ms/token` on the 2,685-active-token arm. Everything below is what a
successor needs next.

### What is done and must not be rebuilt

```text
device index-query preparation   committed, gated, -15.9 ms both topologies
device index projections         committed, gated, exact by construction
device candidate resolution      committed, gated on the centralized arm
positional page numbering        committed, gated, page index = block index
in-chain selection               committed, gated, -43.9 ms rank-local
E4M3 half-up encoder             committed and live in the backend
```

### The blocking question, which is not about performance

**The device attention path cannot execute above 65,536 tokens of context.**
Both attention entry points reject more than 640 candidates, and a ratio-128
layer needs `roundup(context/128, 128) + 128`, which passes 640 at 65,537
tokens and reaches 8,320 at the declared context. The 4 MiB workspace bound is
a second ceiling behind it. See the 2026-08-14 entry above for the measured
sweep and the control that identifies the cause.

Nothing else in this handoff can produce a 1M number until this is resolved,
and resolving it is a kernel design task — tiled or multi-pass dense attention
over an unbounded compressed history with an online softmax — not a tuning
task. Until then the honest description of this branch is a supported context
ceiling of 65,536 tokens, not 1,048,576.

Tracked as [issue #22](https://github.com/ro99/strata/issues/22), which carries
the measured sweep, the reproduction command, the proposed kernel shape and its
exactness bar, and the reason prefill work does not unblock it. **This does not
block Step 5**, by explicit user decision on 2026-08-14: end-to-end `tau(1M)`
is not reachable on this box until prefill throughput is addressed separately,
so Step 5 proceeds against the 65,536-token supported ceiling.

### The two open questions, both about 1M and both measurable

Both are downstream of the blocker above; neither can be answered end to end
until it lifts. Both are measurements nobody has taken, and both were created
by decisions Stage 2 made.

1. **Pre-leasing every attendable block.** Device selection has no first touch
   to lease on, so an in-chain layer leases every attendable compressed block
   and every learned-index block for both ranks. At 2,685 active tokens that is
   11 blocks per indexed layer and the whole term is inside the `6.183 ms`
   residual against the below-threshold arm. At the declared context it is
   **4,096 blocks per indexed layer per rank**. Measure `acquire_device` at
   that block count before assuming anything: an earlier `6.46 us`-per-lease
   attribution in this document was measured, extrapolated, and then withdrawn.
   If it does not scale, the shape of the fix is known -- a full block never
   receives another append, so only the tail block's lease has to move.
2. **The enqueue-only page-descriptor upload.** The physical indexer uploads
   its segment descriptors from a `std::vector` local to the call. For pageable
   memory CUDA stages the copy before returning, which is correct, but the
   driver may synchronize the stream to do it, and a per-indexed-layer stream
   synchronization is exactly what the chain exists to remove. At 4,096
   descriptors it is also 96 KB per layer per token.

### What Step 5 still owes

- `decode_step_seconds` and its JSON surface are **ACTIVE TEMPORARY** in the
  instrumentation ledger, to be removed after the final acceptance run.
- The aggregate `rank_local_*_nanoseconds` phase clocks default to REMOVE
  unless a supported telemetry contract is written for them.
- `host_callback_wait_event` still has no caller.
- The extraction manifest classifies `include/strata/deepseek_kv_cache.hpp` and
  `src/deepseek_kv_cache.cpp` as *excluded* because the `a31ac58` delta on them
  is replay instrumentation. That still describes the experiment delta, but the
  landing modifies those files for its own reasons and the manifest should say
  so.
- A centralized full-model teacher-forcing and generation comparison against
  `main` is required for bit identity, and has not been run since Stage 2.

### The gate, unchanged in shape

```bash
scripts/run_dsv4_rank_local_sparse_gate.sh <tag>          # both arms
ARMS=rank-local scripts/run_dsv4_rank_local_sparse_gate.sh <tag>
python3 scripts/compare_dsv4_sparse_gate.py <baseline>.json <arm>.json
```

Required, all of them:

```text
prompt_tokens                     2,673   (active 2,685, above the 2,048 gate)
generated token IDs               identical to rank-local-lazypos.json
answer text                       identical
attention_index_scalar_dispatches 0
decode_checkpoint_read_bytes      0
steady median                     <= 115.795 ms/token
```

An arm whose `prompt_tokens` falls below about 2,036 has not engaged the
indexer and proves nothing; check it before reading any other number. The index
selection trace hash is **no longer comparable across the topology change**:
the host performs 231 fewer selections per arm because they moved onto the
device, so the running hash necessarily differs. Generated tokens are the
binding gate; a differing hash at an equal entry count is still a real failure,
and the comparison script distinguishes the two.

### Traps found the hard way

- **First-touch leasing and device selection are mutually exclusive.** The host
  numbers and leases a page when a selected row first names it; a selection the
  host never sees has no first touch. An unleased slot in a positional page
  array is an empty page the command would name.
- **Leases block appends.** `rank_local_prepare_layer` clears a layer's leases
  before candidate resolution because a leased block cannot receive the next
  token's row. Any scheme that holds compressed leases across tokens is
  silently defeated there, which is how the prefix cache came to measure zero.
- **Rank 1 owns no compressor.** It is one logical replicated KV stream and
  rank 0 owns it. Anything rank 1 needs that was a side effect of a compressor
  projection has to be asked for explicitly.
- **A bounded workspace sized from its request spans.** The physical indexer
  reserved zero bytes for a query that arrives on the device rather than in a
  span. An unused API is not a tested one.
- **A vector load inherits its allocator's alignment.** The pivot's `uint4`
  read needed 16-byte alignment the histogram region did not have; the symptom
  was 21 failing tests, including a *valid* request being rejected.
- **The reference has an E4M3 quirk that must be reproduced.** Just below a
  power of two the host encodes a factor of two low. It is matched
  deliberately; see the suspected-defect entry.
- **Do not reuse `quantize_e4m3_value`.** It rounds ties to even; the
  index-query contract is half-up, and the difference changes selection.
- **Costs measured at 2,685 tokens do not transfer to 1M**, and one
  extrapolation in this document was already withdrawn for exactly that reason.
