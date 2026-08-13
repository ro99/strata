# DSV4 decode 10 tok/s mission

Status date: 2026-08-12

Mission status: **M1 DONE; M2 DONE; 0091 closed callback-free correctness; 0092 M3 PASS / REVIEW REQUIRED at the user-amended <=115 ms topology gate; the original <=30 ms non-CPU assumption is falsified; Stage 10 CPU parity remains the planned final bridge to 10 tok/s**

Active milestone: **M3 PASS / REVIEW REQUIRED after 0092; successor-stage authorization remains a review decision**

Branch: `exp/dsv4-a2-ownership-screen`
Implementation base (accepted P0 checkpoint): `f565c3b94e7b419a04f0e1c0c3b84d4aeb34d7a6`
Validated capability base: `3dbd1dc9cd97636dba2e26c9912c0ddc6888f036`
Historical A1 queue checkpoint: `3be4da4ec0b15f34b495726ef94262128c8753d8`,
reverted after the binding A2 correctness rejection; it is not accepted
multi-layer readiness. The final documentation and revert commits are reported
by the Git handoff and are intentionally not embedded here.
The A2 recovery source checkpoint is `ab84b2b8c5aecea2f6f5fa196920b22178320fc1`; the
recovery commit is intentionally not self-embedded.

This is the canonical tracked orchestrator/executor state machine for the
current DSV4 rank-local mission. It supersedes the ignored
`hardware/dsv4-lk-moe-optimization-roadmap.md` only for current
authorization and status. Historical experiment documents and ignored raw
artifacts remain authoritative evidence for the scopes they declare. This
document is the durable governance state plus the concrete M1 and M2
checkpoint results. The ignored roadmap remains a compatibility/historical
mirror; this tracked document is the current authorization source.

## Objective and current state

The final objective is batch-one, no-speculation DSV4 decode at **<=100
ms/forward (>=10 tok/s)** on the fixed two-RTX-3090 operating point below.

The current state is deliberately narrow:

- **Implementation prerequisite: M1 DONE.** The exact
  dependent state required to build a faithful rank-local executor was captured
  and validated on the accepted centralized physical-device path.
- **Milestone 1: DONE.** Its schema, opt-in hook, checker, tests, and one
  diagnostic capture arm passed the declared correctness/resource gates.
- **Milestone 2: DONE for exact reusable one-layer readiness.** Root accepted
  M2-A, M2-B, M2-C1, M2-C2a, M2-C2b, M2-C3A, M2-C3B, and the D1-D5 closure:
  eight logical failure arms, partial-enqueue drain/reuse, exact same-executor
  success reuse, actual memory/I/O/allocation/traffic accounting, stable
  portable `.d4o` artifacts, collision-safe publication, and the 78-line
  M2-E Ponytail cleanup. The D4D arm is diagnostic only: transfer timing is
  unavailable, all-resource argmax is indeterminate, and `tau_full` is not
  instantiated. It is not a performance win and must not be extrapolated as
  `4.336894 ms * 43`. The earlier CPU mHC comparison was an invalid oracle
  requirement, centralized attention and Stage-5 input were wrong
  rank-association comparisons, the old executor was not M3-callable, and
  failure/resource evidence was incomplete. Those findings invalidated the
  old validator/measurement, not the rank-local mechanism. Preserve `e0c4fd2`,
  D4A, D4B, D4C, and every rejected raw diagnostic; never relabel them as
  passing.
- **Governance evidence only.** G0 commit `002b61c` tightened Luna work into
  bounded, root-reviewed slices. It is not M2 implementation progress.
- **M3 callback-backed correctness complete, current production implementation
  rejected before timing.** Historical A1 and
  0086/A2 no-barrier rejection remain binding evidence and are not relabeled.
  The fresh ownership discriminator restored exactness with one post-layer21
  main-stream barrier, then the narrow fixed per-command attention staging
  correction restored exactness without that barrier. This classifies the
  adjacent corruption as asynchronous staging ownership/reuse. No timing or
  performance claim was made by the ownership discriminator. Experiment 0088
  then passed one sequential control and two exact 43-layer queued repeats at
  position 104, including live route/page evolution, identical terminal state,
  memory/I/O/allocation gates, and the inherited eight-arm failure/reuse
  matrix. Experiment 0089 then passed exact rank-local terminal head/logits and
  token `8806`, one terminal queued failure with both-rank status and complete
  output withholding, and an exact 43-layer same-executor reuse. Experiment
  0090 then removed the live page callbacks only after materializing the
  fixture row. Both the original arm and one narrow prepared-workspace event
  correction failed the first callback-free warm-up against the sequential
  terminal result. No timing repetition ran. Experiment 0091 then identified
  the cause: `dsv4_prepare_attention` selected its pinned upload buffer on
  `host_callback != nullptr` rather than on whether the command was queued, so
  removing the callback moved 43 queued per-layer `q_norm`/`kv_norm` uploads
  back onto one shared staging buffer that the host rewrites before the queued
  H2D reads it. Extending the accepted `7da38b7` fixed-per-command staging to
  the preparation command restored exactness; the callback-free 43-layer chain
  now reproduces the 0089 terminal object and token `8806`. Experiment 0092
  then measured the exact callback-free chain at `114.944312 ms` median and
  `46.677143 GB/s` routed CPU bandwidth. Under the user-amended `<=115 ms`
  topology gate, M3 passes. The old `<=30 ms` non-CPU assumption is retained as
  falsified planning evidence, not as the reviewed M3 acceptance criterion.
- **Experiment 0087: PASS_LIMITED/REVIEWED for aggregate/source-family
  attribution only, not throughput.** N2/N3 reconciled exactly `1305 =
  15*(43 attention/page + 43 MoE + 1 output head)` callbacks. The combined
  attention GPU union was `669.499507 ms` total (`44.637142 ms/forward`,
  `67.648%` of the profiled gap envelope); all-GPU plus attention-callback
  union was `56.033489 ms/forward`, with `9.475883 ms/forward` uncovered.
  These are profiler-perturbed diagnostics, not production timing, savings,
  or mechanism selection. The binding unprofiled control remains
  `151.155686/84.710043/66.445643/51.155686 ms` for wall/CPU/residual/goal
  gap, and 0076's only equal-scope attention saving is `1.513952 ms`; the old
  synthetic `58.721 ms` is not a promise. N1's profiler hook was removed and
  raw evidence is preserved. N4 is **INVALID**, not a mechanism rejection:
  the literal prompt pathname produced 31 tokens/wrong IDs and existing
  scalar event storage had device-0 zeros versus device-1 nonzeros.
- **Experiment 0092 closes M3 as PASS / REVIEW REQUIRED.** The exact
  callback-free chain measured `114.944312 ms` median wall and `46.677143 GB/s`
  routed CPU bandwidth, satisfying the user-amended `<=115 ms` topology gate.
  Its `41.047528 ms` non-CPU envelope disproves the old `<=30 ms` planning
  assumption; that miss remains visible but no longer defines M3 acceptance.
  This intermediate result is about `8.70` measured forwards/s, not a final
  end-to-end decode claim. The remaining `14.944312 ms` to 10 tok/s is the
  expected scope of later full-runtime closure and the hardware roadmap's
  separate Stage 10 CPU gate/up, down, and weighted-reduction parity work.
- **0091 records a binding lesson about depth-bounded discriminators.** The
  one-layer and adjacent two-layer callback-free prefixes both passed against
  the live defect, because the trigger is the host outrunning the stream rather
  than any per-layer mechanism. The discriminator 0090 authorized would have
  reported PASS and handed the 43-layer chain the same failure. A queued
  ownership question must be discriminated at production queue depth, or by
  driving the host/stream race directly.
- **0084 historical disposition: PENDING IMPLEMENTATION, not a mechanism
  rejection.** Its fail-closed selector proved only that the implementation
  was absent at that checkpoint: the runtime then owned the dependent path
  through `mhc_slot`. Its raw evidence and commit `aaae42e` remain preserved;
  0085 later supplied the reusable one-layer prerequisite. 0084 supplied no
  candidate timing, correctness result, or topology regression claim.
- **0077, 0078, and 0079 remain binding negatives for their declared isolated
  scopes.** They are not a program-level rejection of a faithful full
  rank-local topology.
- **0082 is an accepted explicit byte-admission prerequisite** for the
  canonical centralized control, and **0083 is an accepted NCCL/runtime
  memory-boundary prerequisite** for later review. Neither is a topology win
  or permission to skip M2's exact layer gate.

The reset was completed on the validated capability base. The accepted
pre-C3B source checkpoint is recorded above; the final M2 result commit is
reported by the Git handoff and is not embedded here because this document
must not contain a self-referential hash.

## Validated base, branch, and baseline

The executor branch was created directly from the validated 0083 capability
base `3dbd1dc9cd97636dba2e26c9912c0ddc6888f036`. The standalone Luna skill
source commit `196bb70715eb3e614d312d2d0b67ebce1db47c71` was cherry-picked as
`66426a4b1ec628c4993c64d8fec830d685f52087`; that cherry-pick contains only
`hardware/luna-loop/SKILL.md`. The failed 0084 implementation commit `aaae42e`
is excluded from this branch's ancestry except for this historical reference.
No runtime, source, script, test, or 0084 scaffold is carried here.

The canonical accepted operating-point control is experiment 0082:

| quantity | canonical value |
|---|---:|
| complete decode | **151.155686 ms/forward** |
| routed CPU body | **84.710043 ms/forward** |
| routed payload | **3,449,290,752 B/forward** |
| routed CPU bandwidth | **40.718794 GB/s** |
| residual envelope | **66.445643 ms/forward** |
| decode checkpoint/KV I/O | **exactly zero** |
| runtime VRAM | **21,287,272,448 B/GPU** |

The residual envelope is `151.155686 - 84.710043 = 66.445643 ms`; it is a
makespan remainder, not a license to add overlapping phase counters. The
canonical control also passed the exact output, zero-miss, zero-promotion,
zero decode-allocation, and host-memory gates recorded by 0082.

The branch intentionally does not inherit the unpromoted 0084 source. The
0084 raw selector result remains inspectable through its historical commit and
ignored artifact directory, while the implementation prerequisite is reopened
as a real milestone rather than hidden behind a fail-only flag.

## Cost model and required budget

Every later measurement must instantiate the governing model at its actual
operating point:

```text
tau = max_r(W_r / B_r) + Sigma_serial
```

At the canonical 0082 point, the goal gap is:

```text
151.155686 - 100.000000 = 51.155686 ms/forward
```

The external stack's CPU callback body of 72.889 ms suggests at most

```text
84.710043 - 72.889000 = 11.821043 ms/forward
```

of CPU parity opportunity at this operating point. That number is navigation
evidence, not a guaranteed saving; it must be freshly remeasured after a
faithful topology exists. If later CPU parity is needed to close the goal, the
topology must therefore demonstrate at least

```text
51.155686 - 11.821043 = 39.334643 ms/forward
```

of actual complete-forward removal. These are not independent best cases to
stack. The external **94.282 ms full graph** result is feasibility context
only; it is not an equal-scope Strata pass.

The obsolete **2.125 ms Stage-5R residual allowance is not an additive
full-program gate**. It remains a historical gate for the declared Stage-5R
MoE-only scope and must not be subtracted from, or stacked with, this complete
decode budget.

The current target term is the **66.445643 ms non-CPU/residual envelope and
centralized dependency ownership**. The routed CPU body remains the measured
`argmax_r` resource and must not regress below **36.7 GB/s**. A future
candidate must show which portion of the residual is truly serial and which
work overlaps; it may not assume that the 0082 residual is all topology
opportunity.

Resource signs are fixed for the implementation sequence:

| resource or term | required candidate sign |
|---|---|
| routed CPU bytes/body | non-positive; preserve the argmax and at least 36.7 GB/s |
| attention, mHC, shared GPU, and HBM service | may overlap, but added launches/state are positive load unless measured away |
| NCCL data/status and physical SHM transport | positive communication/load unless a measured dependency reduction pays for it; physical bytes must be measured or marked `not_measured` |
| H2D/D2H and host callbacks | zero on the dependent path where the contract requires device residency; any added transfer/wait is positive |
| `Sigma_serial` | target is a measured reduction in centralized cross-layer handoffs, not a summed phase subtraction |
| VRAM/RSS and allocations | remain within hard ceilings; capacity or timed allocation growth is positive |
| checkpoint/KV/NVMe I/O | exactly zero in steady-state decode; any nonzero demand is a failure |

## Experiment 0087 — current callback-gap attribution

Experiment 0087 is **PASS_LIMITED / REVIEWED for aggregate/source-family
attribution only**, not a throughput or production-memory pass. Its complete
evidence record is
`docs/experiments/0087-dsv4-current-callback-gap-attribution-2026-08-12.md`;
the final Git handoff carries the closure hash and is not self-embedded here.
The default-off callback trace preserved the existing exact output and gates.
P3 produced 645 valid rows (15 forwards x 43 layers), with median forward
`151.070785 ms`, body `85.153019 ms`, inter-callback chain `62.069549 ms`,
finish-to-drain `1.554183 ms`, and post remainder `0.376029 ms`; every row
reconciled to <=1 ns. P4 found `630/630` issue starts before the previous
callback finished and zero positive late-issue lower bound; its one
`168.813209 ms` step-6 outlier makes performance timing non-binding while the
ordering result remains valid.

The upgraded Nsight result reconciled `1305 = 15*(43 attention/page + 43 MoE +
1 output head)` callbacks. Its combined attention GPU union was
`669.499507 ms` total (`44.637142 ms/forward`, `67.648%`), all-GPU plus
attention-callback union was `56.033489 ms/forward`, and uncovered median was
`9.475883 ms/forward`. These are profiler-perturbed diagnostic coverage, not
production timing, savings, or mechanism selection. N4 is **INVALID**, not a
rejection: its literal prompt pathname produced `prompt_tokens=31` and wrong
IDs, while scalar event storage yielded device-0 zeros versus device-1
nonzeros. N1's temporary profiler hook was removed; all raw evidence remains
preserved.

The binding unprofiled 0082 control remains `151.155686 ms` wall,
`84.710043 ms` CPU body, `66.445643 ms` residual, and `51.155686 ms` goal gap.
Experiment 0076's only equal-scope attention saving is `1.513952 ms`; the old
synthetic `58.721 ms` is not a promise. The exact Strata/external difference
`151.155686 - 94.282 = 56.873686 ms` remains unequal-scope feasibility
evidence, not a causal decomposition. Transfer service and physical NCCL
timing remain `NOT_MEASURED`, so full-resource `tau` is not instantiated.

The fresh A2 ownership discriminator is complete at the correctness boundary.
The historical no-barrier arm remains binding rejection evidence: its raw
localization log is
`results/dsv4-rank-local-executor/m3-a2-adjacent/a2-scratch-diagnostic.log`
(SHA-256
`03694fbfc9f2ece47e45f2445ad8ab530c6910993c693454d2396cb8e4c8b0aa`). One
post-layer21 main-stream barrier restored exactness in
`results/dsv4-rank-local-executor/m3-a2-ownership-screen/barrier-r1/combined.log`
(SHA-256
`5785a80fe186f1b1d746a83fd87ac3de14b0f2a1d0fe5d6a18f7530dc2d8c9ba`). The
narrow fixed per-command staging correction passed without a barrier in
`results/dsv4-rank-local-executor/m3-a2-ownership-screen/fixed-upload-r1/combined.log`
(SHA-256
`2724b7576948e452c69af762a33350efec5d211d3a842f564808412b860b1793`), with
source diff SHA `5a2ade67347565a900cca97f68fc60c7e3bc41ff2d010d63e1e1e36144c21d3e`.
Sequential, queued1, and queued2 terminal hashes were respectively
`0741b15a80dc788d`, `0137ac8e33fadf9f`, and `12deb4effdbd4133` for residual,
weighted, and input; all three matched, and layer22 query index 0 was
`0xbb6a` on both ranks. This proves asynchronous staging ownership/reuse,
does not relabel the 0086 rejection, and makes the next action one bounded
exact 43-layer dependent-chain correctness slice after root acceptance.
Timing/performance, graph capture, CPU optimization, Stage 7, and M4 remain
blocked.

## Architecture blueprint

The candidate architecture removes global `mhc_slot` ownership. It does not
change precision, router semantics, expert count, top-k, scoring,
normalization, routed scaling, sparse attention, or DSpark verification.

There are two rank contexts, each owning its device/rank and its own fixed
state. Each context owns:

- replicated immutable mHC, router, compressor, indexer, and norm state;
- rank-sharded attention, shared-expert, routed-expert, head, and embedding
  weights using the checked `Dsv4RankShard` descriptors;
- mutable `4x4096` residual and transition state;
- physical KV/page descriptors and page state;
- main and auxiliary streams, fixed status/data/publication buffers, and fixed
  CPU-partial destinations; and
- its communicator, failure state, and process-lifetime workspaces.

The exact per-layer dependency is:

```text
replicated mHC pre/norm
  -> rank-local 32-head physical attention
  -> local wo_a/wo_b
  -> FP32 data + U32 MAX status NCCL reduction
  -> BF16 publication
  -> replicated mHC transition/router
  -> concurrent rank-local CPU routed shard + GPU shared shard
  -> exact local BF16 join
  -> FP32 data + U32 MAX status NCCL reduction
  -> BF16 publication
  -> rank-local mHC post/next-layer state
```

Embedding is vocabulary-row sharded, followed by the required BF16 reduction
before four-copy expansion. After all 43 layers, final mHC/norm and the
rank-local output head are followed by BF16 all-gather/sampling as required by
the target contract.

The measured path has no hidden host-visible per-layer continuation, no
central compatibility fallback, no per-layer output collection, and one final
logical completion boundary. Both ranks enter every collective after a local
failure; failed output is withheld. There are no timed-path allocations or
decode I/O operations.

The implementation should reuse or adapt the existing `CudaBackend` per-device
mHC primitives, `Dsv4RankShard` descriptors, accepted Stage-4 physical-page
attention arithmetic, `Dsv4HostMoeExecutor`, and Stage-5R device-view/NCCL
chain. Probe `main` loops are evidence and test scaffolding, not production
runtime architecture; they must not be copied wholesale into the runtime.

## Stable four-milestone state machine

Exactly four milestones exist in this mission. Milestone 1 is complete and
Milestone 2 is complete for exact one-layer readiness. Milestone 3 is
reopened/in progress at the correctness boundary after the A2 ownership
discriminator; 0086 remains a binding historical rejection. Experiment 0087
is PASS_LIMITED/reviewed and N4 is invalid. Timing/performance and M4 remain
blocked.
Root owns strategy, architecture, gate changes, and review authorization;
the executor owns the bounded implementation and evidence for the active
milestone.

### Milestone 1 — exact dependent-state capture (DONE)

Implement a versioned C++ replay schema and default-off diagnostic capture over
the accepted centralized physical-device path at positions **104–118**. The
capture is an implementation prerequisite for the later topology; its timing
is invalid by design.

Each of the **15 files** must contain:

- the initial `4x4096` BF16 residual and terminal state/token association;
- **86 ordered records** (43 layers × attention/FFN branch) with layer and
  branch identity;
- the mHC weighted/pre vector `[4096]`;
- normalized branch input `[4096]`;
- branch output `[4096]`; and
- the resulting `4x4096` residual.

The capture may use the existing synchronous `--layer-hash-trace` diagnostic
path. That path changes timing and is **INVALID as performance evidence**; no
latency, throughput, or topology claim may use it.

Milestone-1 gates are:

- all 15 files have exactly 86 ordered records and the declared version;
- every vector has the exact shape, expected BF16/FP32 association, and finite
  values;
- adjacent residuals are continuous and the terminal state/token association
  is present;
- hashes equal the existing operation/layer diagnostic hashes and the exact
  accepted token sequence is preserved;
- reader/writer tests fail closed on version mismatch, truncation, and trailing
  bytes;
- the default runtime path is unchanged and the capture is opt-in;
- decode checkpoint/KV I/O remains zero and both VRAM/RSS ceilings hold; and
- the result is one reversible commit with raw deterministic artifacts kept
  outside Git.

#### M1 concrete result

The single fixed capture arm ran on 2026-08-11 from source HEAD
`f8123be80780110f896fcd6850887e4e92c6e602` using the Release CUDA runner
`build-stage5/strata-deepseek-run` and the checker
`build-stage5/strata-dsv4-dependent-replay-check`. The wrapper recorded the
exact fixed point: `CUDA_VISIBLE_DEVICES=1,2`, runtime devices `0,1`, model
`models/dsv4f`, the 104-token corrected prompt, `--max-new 16`, physical KV,
host-routed MoE, 28 host-attention threads, `.95` admission plus the explicit
`21,256,421,376 B` bound, and `--layer-hash-trace`. The diagnostic path's
`/usr/bin/time` result is intentionally invalid and is not a throughput
measurement.

The ignored raw result directory is
`results/dsv4-rank-local-executor/m1-dependent-capture/`. It contains 15
collision-refusing `.d4c` files for positions 104–118, each exactly
`15,634,250 B` (total `234,513,750 B`), with the accepted input/generated
token association:

```text
[43, 8806, 440, 5270, 4496, 1205, 9238, 304, 366, 260, 3418, 294,
 6719, 8454, 305, 3345]
```

The checker reported `15` files, `86` records per file, `5,160` operation
hashes, and `645` FFN/layer hashes. Its output was compared with the existing
diagnostic JSON using canonical `jq -cS` output and `cmp`: both comparisons
passed. The operation comparison hash is
`ace6df31de8a7bf8ca0d9929a9fe76c45587a40f35a2472361b9257a5e7e1604`; the
layer comparison hash is
`f308c2be8b2ac9e5f25d271cd70a3af1107ed58dbd2791c1102a71ad485de7a8`.
The generation JSON hash is
`d3a95bc05760b3b458521c13692c228228da2f8e2de2ff334b3e294a5a314487`, the
checker JSON hash is
`3b5174c0eaed42f1bc4b9caf6ae925fe5a76fe250ee309277b7f72d2c314445c`, and
the complete raw SHA-256 manifest (including every `.d4c`) is
`06e440e4d7914fe3eeb8496ad5a17ed12779a2a164b58de5248ed8c5eecaa796`.

The resource gates passed: RSS was `157,740,879,872 B` against the
`231,928,233,984 B` ceiling; both runtime GPU values were exactly
`21,287,272,448 B` (at the declared hard ceiling); decode checkpoint bytes
were zero; decode cache demand H2D, misses, evictions, KV misses, KV
evictions, promotions, and dependent KV D2H were zero; timed weight/workspace
allocation calls were zero; and all `645` decode MoE callbacks completed
without failure. The nonzero `553,681,920 B` checkpoint value is setup/load
I/O, not decode I/O. The physical KV path's expected host-to-device page writes
remain reported separately and are not demand reads or misses.

The four new schema tests passed: exact round trip, shape/order/continuity/
finite rejection, unknown-version/truncation/trailing-byte rejection, and
invalid-write/collision closure. The full `strata-tests` harness passed
`278/279` with one pre-existing opt-in skip. The two invalid environment probes
also failed closed before checkpoint work (missing `--layer-hash-trace` and
limit `16`), and `git diff --check` passed. All raw captures, JSON, logs,
memory snapshots, and generated binaries remain ignored.

M1 success authorizes **only Milestone 2 after root review**. A missing record,
shape mismatch, continuity break, altered default path, or diagnostic output
that cannot be tied to the accepted hashes leaves the implementation
prerequisite OPEN.

M1 scope is limited to the versioned C++ schema/reader/writer and tests, the
default-off hook around the accepted centralized diagnostic path, and a
reusable capture/validation command. It does not include rank-local runtime
code, a candidate selector, performance timing, graph capture, CPU arithmetic,
or an end-to-end run.

### Milestone 2 — exact rank-local layer (DONE; exact reusable one-layer readiness)

At middle layer **21** and position **104**, execute one exact dependent
attention+FFN layer from captured incoming residual to outgoing residual. Use
two rank contexts and both real reductions; no `mhc_slot` and no host layer
continuation are allowed.

Reuse the actual Stage-4 attention and Stage-5R MoE fixtures/weights plus the
M1 dependent state. The fixed fixtures are: d4c
`1712565387b3565c983da3860c17d6e0648b25792b57da172fd638738adff24a`, d4r
`da8f59de692b2c63aa8e83a09aabbf39c0aa0f1c829cf2934d0193f66cff5d00`, and d4m
`f754e8ebd54a8ae57d16fb77e626434a46e163534a1e2f4380d28fe7a7ea6bdf`.
Root review is required between every slice; Ponytail Full governs reuse and
minimal coding, and Ponytail Review precedes handoff, but neither weakens the
AGENTS numerical/resource gates nor replaces root correctness review.

M2 executes exactly these root-authorized slices; each has one question, one
artifact, and no implicit follow-up. C1 through C3B are accepted correctness
evidence, not performance claims. Their exact checkpoint is:

- **M2-A DONE** — commit `5bd660489c58b27ffd47d823ffa880c3e98d3fb9` replayed
  the correct fused prefix from the initial residual through records 0..44
  (44 transitions per rank), with zero mismatches on both ranks. The exported
  Stage-4 publication is
  `results/dsv4-rank-local-executor/m2-rank-local-layer/m2-a/stage4-rank-local-l21-p104.d4o`,
  SHA-256 `f8b5eb041e242094f6a47dda6306a28405bac7c0175464c7cdeee3969a94ff3a`;
  final log SHA-256 is
  `009ad74d05ea0166174812563a768dec54f27130ace5ee8a3b7df9085041d6b9`.
  The rejected fresh-record42 arm remains at the preserved M2-A raw path; it
  skipped the fused prior mHC state and is not a fixture discontinuity.
- **M2-B DONE** — commit `8c8b564d147c49a23bec8174480b2192d2404841`
  proved host-visible CUDA versus device-only mHC transition/router exactness
  on both ranks: 4096 input values, 256 logits, expert IDs, and six
  coefficients all had zero mismatches. The target route is
  `80,49,28,75,202,34` with coefficients
  `0.394806057,0.38957417,0.229772747,0.176799551,0.17297478,0.136072874`.
  The target publication differs from centralized Stage-4 by one BF16 value
  (maximum `9.53674316e-7`), while resulting FFN input, logits, routes, and
  coefficients match the central association exactly. Raw log:
  `results/dsv4-rank-local-executor/m2-rank-local-layer/m2-b-transition-oracle/transition.log`,
  SHA-256 `0a334129d13ec58ba99fc4e428918761eb1786848b51fe828afd0b3a43ea35ec`.
  Timing and resources were not run; no performance claim is made.

- **M2-C1 DONE** — commit `87b15f9f35380f3b58c5b298970ad217ae818f9c`
  independently exported the Stage-5 rank-local BF16 publication for
  layer 21/position 104. Both rank publications were equal and the three
  oracle repetitions were byte-stable. The artifact is
  `results/dsv4-rank-local-executor/m2-rank-local-layer/m2-c1-outgoing-oracle/stage5-rank-local-l21-p104.d4o`,
  SHA-256 `3dca011bb401d748ee0fdd501c65e150781ef6cba721e34fb4766eff1fe07a88`.
  Accepted host-visible CUDA prefix transitions consumed both independent
  publications and produced exact target record-44 residual/input on both
  ranks. Stable target hashes are Stage-5 `21b4285cf0c36f8b`, residual
  `5b42d728a3030bdd`, weighted `18034115ca2830db`, and normalized input
  `cba74f5aeeb1503e`.
- **M2-C2a DONE** — commit `60b7fe657847728edf74caef38a2a4b303f4524e`
  factored the exact `dsv4_route_sqrtsoftplus_f32_into` operation over
  caller-owned scratch/output spans. Existing and into APIs share one
  arithmetic path; parity, tie ordering, invalid-input closure, and 100-call
  no-allocation tests passed.
- **M2-C2b DONE** — commit `0409bddb64c0df1fb02fb827cc929698c24d51ea`
  removed replay ownership from the executor callback. It derives routes and
  coefficients from live device logits plus the setup-owned bias using fixed
  scratch/output storage; `.d4m` remains an independent oracle only.
- **M2-C3A DONE** — accepted source checkpoint
  `0f35333a3eb3f24b853752eba222cfd0ac77ce3d` owns initialization once and
  exposes a dynamic live-state call: `initialize once -> consume already-live
  mHC state -> run layer -> leave next state live`. Replay/fixed-layer
  ownership and success-path per-run result allocation were removed. Uniform
  failure drain/reset and clean reuse were closed by D1-D3.
- **M2-C3B DONE** — the prepared query was exact on both ranks, hash
  `eb419d5524d8fb63`. The initial bad-seed arm and corrected localization arms
  remain preserved. R1 independently proved the association defect: accepted
  publication A was `BF16(FP32(partial0 + partial1))`, hash
  `66b3e1c0ef45c065`; premature local-BF16 publication B was
  `BF16(BF16(partial0) + BF16(partial1))`, hash `69162ed5dca96abb`, with
  1514 mismatches, first index 0 (`bfda` versus `bfdb`), and maximum delta
  `0.015625`. R2 changed only the rank-local collective input to raw FP32
  partials before BF16 publication. The final one-arm live-layer proof passed
  both ranks exactly: attention hash `66b3e1c0ef45c065`, MoE hash
  `21b4285cf0c36f8b`, exact routes/coefficients/CPU partials/rank equality,
  and outgoing hashes `5b42d728a3030bdd`, `18034115ca2830db`, and
  `cba74f5aeeb1503e`. Central record-44 diagnostics were unchanged and are
  association diagnostics only: post 3606/max `0.03125`, weighted
  1308/max `0.0078125`, input 1270/max `0.00390625`. No timing or performance
  claim was made.

The C3B raw evidence is retained under the ignored M2 directory. The final
close log is
`results/dsv4-rank-local-executor/m2-rank-local-layer/m2-c3b-close-1/c3b-close-1.log`,
SHA-256 `07e538436516cf313b6dbd33c0423d08fe380db7e2a8999f04d090aa3a023f10`.
The exact-query log SHA is `61ddf7aac367e6be9c9bee15a1e5b505a50fcf6e3c64ac579be898171614f940`,
R1 association log SHA is
`5707af74f68fa247fc9f760d46a918861e77445b97cf3489b669757819d24193`,
R2 raw-FP32 log SHA is
`20c6766aa2b8ddf578b454245af1363029c2374e53ff765a722de298e8547ed2`,
and publication-localization log SHA is
`34431b53f0e06811ac695ff087cec42e498bc5f2eaf1baf9b668fb037c1281c5`.
The first bad-seed arm, corrected-seed arm, and negative association arm are
rejected evidence, not passing results, and remain at their original ignored
paths. The actual corrected-seed log hash, when cited, is preserved as
`0acc057e4f6a32491973d08f21556c0e41a7b577a5b674dd0242dca1e38f09a5`; no
malformed hash is used here. Validation after the close was `280/281` direct
tests with one documented skip, `make check` 2/2, and clean `git diff --check`.
Ponytail Review found the raw-FP32 borrowed destination to be the minimum
reuse of the persistent reduction buffer; duplicate pointer validation was
consolidated, while independent diagnostic complexity was intentionally
retained for oracle provenance.

M2-D closure was root-reviewed as complete. D1/D3 ran all eight logical
failure arms with matching nonzero phase status, zeroed payloads, and clean
same-executor success reuse; D2 proved the rank-0-enqueued/rank-1-pending
exceptional cleanup and exact reuse. D4D ran one warmup plus three exact
diagnostic repetitions with checkpoint/workspace/weight allocation deltas at
zero, actual memory samples below ceiling, and honest unmeasured transfer
accounting. D5 replaced native `.d4o` struct transport with fixed little-endian
bytes, O_EXCL publication, and inode-safe collision cleanup. M2-E removed 78
lines of dead measurement plumbing without changing formulas or gates. The
final documentation/Git closure is the current handoff boundary; its result
hash is reported by Git rather than embedded here.

The remaining slices are:

| slice/status | primary artifact and scope (hard window) | prerequisite and pass condition | stop condition and next authorization |
|---|---|---|---|
| **M2-A DONE** | Accepted-CUDA prefix calibration and independent Stage-4 publication artifact; two app files, no executor/backend redesign or timing; **30 min**. | M1 fixtures and root review. Pass: records 0..44 and rank-order BF16 publication exact on both ranks. | Accepted; no successor work was implicit. |
| **M2-B DONE** | Host-visible versus device-only mHC transition/router oracle; layer app only, no MoE or timing; **30 min**. | M2-A artifact. Pass: target FFN input, logits, routes, and coefficients exact on both ranks. | Accepted; C1 was separately authorized after review. |
| **M2-C1 DONE** | Independent stable Stage-5 publication plus accepted-CUDA target record-44 oracle; two app files, **<=200 net lines**, **30 min**. | M2-A/B artifacts and root review. Pass: three stable bytes, rank equality, exact target residual/input. | Accepted; C2a was separately authorized after review. |
| **M2-C2a DONE** | Allocation-free `route_into` primitive and parity/failure tests; three library/test files, **<=140 net lines**, **20 min**. | C1 artifact and root authorization. Pass: one arithmetic path, exact parity/ties, no success-path allocation. | Accepted; C2b was separately authorized after review. |
| **M2-C2b DONE** | Live callback logits+bias routing with no `.d4m` executor ownership; three source files, **<=160 net lines**, **30 min**. | C2a API/tests and root review. Pass: fixed scratch/output lifetime and no callback allocation/replay comparison. | Accepted; C3A was separately authorized after review. |
| **M2-C3A DONE** | Initialize-once/live-state/dynamic-call lifecycle and fixed success-path storage; three source files, **45 min**. | C2b and root blueprint review. Pass: no replay ownership/begin/residual upload, next mHC state remains live. | Accepted; D1-D3 later closed failure-drain/reuse. |
| **M2-C3B DONE** | Raw-FP32 rank-local attention collective correction and one exact live-layer proof; three source files, no timing; **45 min**. | C3A and root correction review. Pass: all publication, route, partial, outgoing-state, rank-equality, and live-state gates. | Accepted; D1-D5 closure followed. |
| **M2-D DONE** | Eight failure arms; matching data/status collective closure; zero/withheld failed publications; clean success after each failure; allocation-free measured window; actual I/O/allocation/RSS/VRAM/traffic evidence; one warmup plus three diagnostic repetitions; stable artifact transport and closure evidence. | C3B pass, root authorization, fixed fixtures/artifacts, and the D1-D5 evidence above. Full-cost-model transfer timing remains unavailable; this is readiness evidence, not a performance claim. | Accepted by root at the M2 boundary; M3 was then authorized, but its A2 correctness gate later rejected multi-layer readiness. Experiment 0087/N2/N3 were subsequently reviewed as PASS_LIMITED attribution only and N4 was invalid. The current A2 ownership discriminator is authorized separately. |
| **M2-E DONE** | Ponytail cleanup of dead layer-app measurement plumbing; one app file, **15 min**. | D4D evidence and root review. Pass: 78-line net reduction, unchanged formulas/modes, app build, full test harness, and diff check. | Accepted as cleanup only; no timing or correctness claim changed. |

M2 is complete for exact one-layer reusable readiness. A slice never starts its
successor without root review, and completed slices are correctness/state-
machine evidence, not a performance claim or a license to redesign the
topology. D4D's transfer timing remains unmeasured, so the full-resource
`tau` is intentionally not instantiated. The accepted P0 one-layer live-page
capability remains the production boundary; A1's queued checkpoint is
historical/reverted evidence only.

### M3-A1/A2 rejection and ownership-recovery closure

M3-A1 (`3be4da4`) passed a one-layer queued smoke, but that result did not
prove safe multi-command residency. The binding M3-A2 attempt used distinct
layer21/layer22 position-104 fixtures and exact production candidate contracts.
The sequential control passed, and queued layer21 callbacks were exact, but
same-stream post-layer21 snapshots diverged on both ranks before layer22
preparation: residual `16302/16384`, weighted `4085/4096`, and input
`4091/4096` BF16 mismatches. Layer22 query index 0 was `0xbbb1` versus control
`0xbb6a` on both ranks. The callback diagnostic log is
`results/dsv4-rank-local-executor/m3-a2-adjacent/a2-diagnostic.log` (SHA-256
`24ea843f5c709766494516a84be557fe21cf0da2d8ec8fb0a0e83e92f0c92796`); the
corrected same-stream localization log is SHA-256
`03694fbfc9f2ece47e45f2445ad8ab530c6910993c693454d2396cb8e4c8b0aa`.
The full ignored evidence manifest is
`results/dsv4-rank-local-executor/m3-a2-adjacent/a2-rejection-manifest.md`.

This is a correctness rejection before timing, not a performance datapoint.
The 0086 rejection remains binding historical evidence: preserve A1 and every
0086 raw arm as rejected history, and do not relabel the one-layer P0
capability. The fresh ownership discriminator then supplied two exact controls:
the one-barrier positive arm and the no-barrier fixed-slot correction. Its
combined logs, terminal hashes, and query `0xbb6a` are recorded above. The
result classifies asynchronous attention staging ownership/reuse; it does not
erase 0086, add a throughput claim, or close the 43-layer milestone.

M2 remains DONE for exact one-layer readiness. Experiment 0088 passed the
bounded 43-layer chain-state slice, and Experiment 0089 closed terminal
head/logits/token plus queued-chain failure/reuse. Neither erases the historical
0086 rejection. Experiment 0090 showed that this correctness depended on the
live page-callback boundary: callback-free execution mismatched before timing,
including after the sole bounded event-ownership correction. Experiment 0091
then attributed that dependence to the preparation command's host staging
selector and corrected it, so callback-free execution is now exact. M3's
production-shaped timing was then measured by Experiment 0092. M3 is **PASS /
REVIEW REQUIRED** under the user-amended `<=115 ms` topology gate. The old
non-CPU condition is retained as a falsified planning assumption; successor
work remains subject to review.

### Milestone 3 — exact 43-layer dependent forward (PASS / REVIEW REQUIRED)

The first M3 action would have been one bounded production-shaped exact chain:
chain all **43 layers** for one actual decode position using fixed buffers and one final
completion. There is no per-layer D2H or host control continuation. Add
rank-sharded production loader/admission integration only as required by the
exact chain; do not begin with graph capture or CPU arithmetic tuning.

Require exact adjacent state, terminal hidden/logits/token against the target
rank-local contract, exact route evolution and KV/page evolution, all failure,
I/O, allocation, lease, and memory gates, and then one cheap
production-shaped candidate forward. The original falsifier used `>120 ms`,
`<36.7 GB/s`, and `>30 ms` non-CPU conditions. Review showed that the last
condition incorrectly charged later CPU-parity scope to M3. The user-amended
M3 acceptance gate is median wall **<=115 ms**, with the exactness,
CPU-bandwidth, memory, I/O, and allocation gates unchanged. The historical
`30 ms` miss remains recorded rather than laundered.

Experiment 0088 executed one sequential rank-local control and two queued
43-layer repeats at position 104. Layer 0 matched the actual starting fixtures;
layers 1--42 matched the dynamic sequential rank-local control; every route was
independently recomputed from live logits and checkpoint routing data. The
three terminal hashes were identical: weighted `e1a9a77f0b01a361`, input
`122a716defe84e1b`, and hidden `5017083817dd2848`. All eight inherited logical
failure arms passed with closed collectives, withheld payloads, and exact
same-executor reuse. Candidate checkpoint/workspace/weight-allocation deltas
were zero. Peak observed correctness-harness values were
`158,611,144,704 B` RSS and `7,657,881,600 B/GPU`, below both ceilings.

Experiment 0089 extends that exact state with terminal output publication. The
sequential control, two queued candidates, and post-failure reuse share logits
hash `343d766f3f5c0af3` and next token `8806`. The terminal layer-42
`MoePostRank1` arm reduced nonzero status on both ranks, withheld all outputs,
closed once, and reused exactly. The binding log is
`results/dsv4-rank-local-executor/m3-43-chain/correctness-r13/combined.log`
(SHA-256
`b78ec2058773a3f24ab2a10c9d99257201b1d75db9e3d84633bb402889c23887`);
the run-time source diff SHA-256 is
`1d49ed2fb93d2a4f0059a5cbf5afbf8405a9bf240432a62b8de26fb219d0a162`.
The detailed records are Experiments 0088 and 0089.

Experiment 0090 attempted that production admission. The callback-backed
control remained exact, but the first callback-free warm-up mismatched the
sequential terminal result in two independent arms. Guarding reusable prepared
query storage until attention consumption did not change the failure and was
removed. Consequently no measured repetitions ran and none of the `120 ms`,
`36.7 GB/s`, or `30 ms` gates was evaluated.

Experiment 0091 closed that admission. The differential was
`dsv4_prepare_attention`'s host staging selector: with a page callback the
per-layer `q_norm`/`kv_norm`/RoPE upload used 43 fixed per-command slots, and
without one it used a single shared pinned buffer that the host rewrites before
the queued asynchronous H2D reads it. Layer 0 was exact at every queue depth
and a two-layer callback-free prefix was exact, but layer 1's observation hash
moved from `f233565ecfa96477` at queue depth 2 to `384d71deb36191ee` at queue
depth 43 — an already-enqueued layer changed because later layers were enqueued
after it, which only host-side staging reuse can cause. Extending the accepted
`7da38b7` fixed-per-command staging to the preparation command made the
callback-free 43-layer chain exact against the sequential control, with the
0089 terminal hashes and token `8806`, zero page callbacks, zero decode
checkpoint I/O, and zero timed allocations. The binding arms are
`results/dsv4-rank-local-executor/m3-43-chain/callback-free-r2` (negative,
combined log SHA-256
`5e3ad86cdef92dfa102412d66343ce51c4ca86f06da1616d2cf25eccd4c89b79`) and
`callback-free-r3` (positive, combined log SHA-256
`e2a0a2aedd487e5dc2f807d22b6ea2db5bf698a510666baca5c3fc16bcfce709`).

Experiment 0092 measured one warm-up and three exact callback-free repetitions.
The binding `timing-r3` medians are `114.944312 ms` wall, `73.896784 ms` routed
CPU body, `46.677143 GB/s` routed CPU bandwidth, and `41.047528 ms` non-CPU
dependency. Wall and CPU bandwidth pass their `120 ms` and `36.7 GB/s` gates;
the non-CPU envelope misses the old `30 ms` assumption by `11.047528 ms`. The entire
terminal head measured only `2.288864 ms`; even subtracting all of it leaves a
`38.758664 ms` lower-bound residual, so the retained diagnostic does not cause
the rejection. All repetitions matched the accepted terminal hashes and token,
used zero page callbacks, checkpoint I/O, or timed allocations, and stayed
within memory ceilings. The binding log is
`results/dsv4-rank-local-executor/m3-43-chain/timing-r3/combined.log`
(SHA-256
`fec60357ee1adc3fb4522a7f08e742a39be15869d4cd58d1161bb96479e46540`).

This measured scope is approximately `8.70 forwards/s`, still `14.944312 ms`
above 100 ms, but it is not a complete end-to-end throughput claim. M3 is
**PASS / REVIEW REQUIRED** because `114.944312 <=115 ms` with exactness and all
resource gates closed. The hardware roadmap's Stage 10 CPU gate/up, down, and
weighted-reduction parity remains the separate planned final bridge after
topology validation; M3 was not expected to establish that parity. Successor
work requires review, and the 94.282 ms external number remains feasibility
context, never an equal-scope pass.

### Milestone 4 — end-to-end 10 tok/s closure (BLOCKED)

At the fixed operating point, run one smoke followed by three interleaved
control/candidate repetitions, after teacher-forcing and generation
correctness are established.

Pass requires all of the following:

- candidate median **<=100 ms/forward** and improvement outside observed
  variance;
- exact operation, layer, route, output, generation, and failure closure;
- routed CPU throughput **>=36.7 GB/s**;
- non-CPU dependency **<=23 ms/forward**;
- no hidden fallback, no per-layer host continuation, and one final completion;
- zero decode I/O, KV misses/promotions, and timed-path allocations;
- per-GPU VRAM **<=21,287,272,448 B** and host RSS
  **<=231,928,233,984 B**; and
- no unexplained resource inflation or non-finite output.

If the correct topology is above 100 ms, stop and instantiate a fresh cost
model at that new operating point before any graph or CPU-kernel work. Do not
automatically start a fifth milestone.

## Stable operating point and hard invariants

Unless a milestone explicitly says it is a fixture-only test, use:

- model `models/dsv4f`;
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`;
- `CUDA_VISIBLE_DEVICES=1,2`, runtime devices/ranks `0,1`, and no P2P;
- batch one, no speculation, corrected 104-token prompt;
- 16 requested tokens and 15 timed forwards;
- 28 attention threads;
- `.95` admission plus explicit `21,256,421,376 B` per-device budget;
- zero decode checkpoint/KV I/O, misses, and promotions; and
- separated setup/prefill/decode accounting, with fresh repetitions reported
  as medians and observed ranges where a performance gate applies.

The AGENTS hard invariants are binding:

- no weight, cache, predictor, draft, storage, or fallback precision below
  four bits;
- precision, router semantics, expert count, top-k, scoring, normalization,
  routed scaling, mHC association, sparse attention, and DSpark verification
  do not change silently;
- predictors, if any, are advisory only;
- exact mode either completes exact work or reports failure, with no hidden
  fallback;
- dense models larger than aggregate resident memory are reported as I/O
  dependent rather than made artificially sparse;
- runtime implementation remains dependency-light C/C++, not Python or a
  framework runtime; and
- model files, raw traces, profiler captures, generated binaries, and large
  logs remain outside Git.

Out of scope until Milestone 4 review are graph capture, Stage-10 CPU
arithmetic optimization, speculation, precision/router/top-k changes, a
friendlier workload, proprietary reverse engineering, and documentation-only
substages. Documentation records a concrete result; it is not implementation
progress by itself.

## Milestone status and handoff

| milestone | status | capability required | definition of done | evidence commit + artifacts | next authorized action |
|---|---|---|---|---|---|
| 1. Exact dependent-state capture | **DONE** | Versioned C++ state schema and opt-in capture of the accepted centralized physical-device path | 15 valid files, 86 ordered records/file, exact continuity/hashes/token, failure tests, unchanged default path, zero decode I/O, memory gates, one reversible commit | `results/dsv4-rank-local-executor/m1-dependent-capture/`; 15 files/234,513,750 B; generation `d3a95bc...`; manifest `06e440e4d7914fe3eeb8496ad5a17ed12779a2a164b58de5248ed8c5eecaa796`; implementation/result commit is the single Git handoff commit (not self-embedded) | Root reviews M1; only then M2 may start |
| 2. Exact rank-local layer | **DONE** | M1 state plus bounded slices M2-A through M2-E; two rank contexts, exact attention/FFN dependency, real reductions, no `mhc_slot` | Layer-21/position-104 exact incoming-to-outgoing state, rank ownership/failure gates, reusable lifecycle, resource evidence, stable artifact transport | M2-A/B/C1/C2a/C2b/C3A/C3B/D1-D5 accepted checkpoints; final close evidence above; preserved `e0c4fd2`, A1, A2, and rejected arms remain classified; P0 remains accepted | Complete; no further M2 action. M3 correctness recovery is the current boundary |
| 3. Exact 43-layer dependent forward | **PASS / REVIEW REQUIRED** | M2 layer contract plus callback-free rank-local page/state ownership and fixed one-completion chain | Exact chain and user-amended median `<=115 ms`, CPU `>=36.7 GB/s`, resource gates | 0091 callback-free ownership fix; 0092 exact median `114.944312 ms`, `46.677143 GB/s`; old non-CPU `<=30 ms` assumption falsified | Review successor-stage authorization |
| 4. End-to-end 10 tok/s closure | **ELIGIBLE AFTER REVIEW; NOT STARTED** | M3 complete dependent chain plus teacher-forcing/generation closure | Three interleaved fixed-point repetitions, candidate median <=100 ms, exact/resource gates, variance-clearing improvement | M3 passes its amended topology gate; `14.944312 ms` remains; hardware-roadmap Stage 10 CPU parity is the planned final bridge | Root/user review before execution |

## Append-only handoff log

Updates to this document occur only after concrete milestone results. Root
reviews actual diffs and raw evidence, not a callback summary alone.

| date | orchestrator decision | executor result | evidence | classification | next authorization |
|---|---|---|---|---|---|
| 2026-08-11 | Reset the mission onto validated capability base `3dbd1dc`; retain only the standalone Luna skill cherry-pick; exclude 0084 implementation ancestry | Created the canonical mission state machine; no runtime, experiment, measurement, or script work performed | Branch `exp/dsv4-rank-local-executor`; skill-only commit `66426a4`; this document's single result commit is reported by Git handoff | **COMPLETED governance reset; implementation prerequisite OPEN** | M1 exact dependent-state capture is NEXT; no later milestone is authorized |
| 2026-08-11 | Reclassify historical 0084 selector result | The selector failed closed before model/CUDA admission because no exact two-rank dependent executor existed; no candidate timing was produced | Historical commit `aaae42e`; ignored `results/dsv4-rank-local-full-chain-r2/candidate-structural-gate/`; experiment 0084 raw record | **PENDING IMPLEMENTATION / invalid as mechanism rejection** | Reopen with M1 capture; preserve 0084 and do not call it a topology result |
| 2026-08-11 | Execute only M1 on the clean executor branch; preserve 0077–0084 classifications and stop before M2 | Implemented the versioned `.d4c` schema, fail-closed reader/writer/tests, default-off centralized diagnostic hook, checker, and fixed-point capture wrapper. One named-tmux arm produced 15 exact files; all continuity, hash, token, memory, I/O, and failure gates passed. | `results/dsv4-rank-local-executor/m1-dependent-capture/`; generation `d3a95bc...`, checker `3b5174c...`, operation comparison `ace6df3...`, layer comparison `f308c2b...`, complete manifest `06e440e4d7914fe3eeb8496ad5a17ed12779a2a164b58de5248ed8c5eecaa796`; full harness `278/279`, one skip | **COMPLETED M1 correctness prerequisite; capture timing invalid; no throughput claim** | Root review is required; M2 exact rank-local layer is NEXT, M3/M4 remain BLOCKED |
| 2026-08-11 | Root authorized the bounded M2 rank-local layer; stop at the first exact-gate defect and preserve raw evidence | Added reusable `Dsv4RankLocalLayerExecutor` library composition (two persistent CUDA/NCCL contexts, two NUMA-bound 24-thread CPU pools, rank-local attention, existing device-input host MoE, BF16 publications, and eight failure enums), narrow borrowed mHC/device-input bridges, thin d4c/d4r/d4m fixture driver, launcher, and stub/order test. Required fixture hashes matched; fixture boundaries and routes passed. The independent host mHC oracle failed before hardware timing: 453/4096 attention-input values, max abs `0.0009765625`; a pre-existing corrected-device diagnostic failed attention publication at 3834/4096, max abs `0.13671875`; strict callback validation failed 3311/4096 inputs per rank. The rejected diagnostic had 257.164/27.747/26.395 ms (median 27.747 ms) but used a temporary callback bypass and failed exact publication, so it is not a performance result. | `results/dsv4-rank-local-executor/m2-rank-local-layer/raw.review-strict.log` SHA-256 `484aff6133e860aed8be326d9721716ef96fde06d75b2fd41aef65daa045a813`; rejected hardware diagnostic `raw.fixture-branch-compare.log` SHA-256 `28a0bbd0c2de18d42b4437f6e3b2f2bcb71c15adf50b0f6210abd6aa86651e1e`; strict callback `raw.strict-fixture-3.log` SHA-256 `1ef7d82f7258a5731c3ea2749c49ca85d8707d6f2adf4b812ce7c2b32b599fb9`; metadata/fixture hashes `raw.hardware-and-fixtures.log` SHA-256 `4f17da7a42d195ea9fbe316a286d1e3e5bd3420913f09a73ac6915888c3a5a11`; `make check` passed 2/2 | **REVIEW REQUIRED; no exact M2 result, no throughput claim, no valid failure matrix** | Root review of mHC/attention state mismatch; M3 remains BLOCKED and was not started |
| 2026-08-11 | Root reviewed `e0c4fd2`; record G0 governance commit `002b61c` as governance evidence only | The four review findings recorded above invalidate the validator/measurement, not the rank-local mechanism; preserve `e0c4fd2` and all rejected raw diagnostics, with no M2 progress claimed. | Existing M2 raw paths/hashes above; governance-only commit `002b61c` | **REVIEW REQUIRED; prior timing/validator rejected** | M2-A central calibration and independent Stage-4 rank-local oracle is the sole next action; root review required |
| 2026-08-11 | Root accepted M2-A and M2-B; keep M2 open and authorize only C1 | M2-A corrected the fused prefix and exact Stage-4 publication; M2-B proved host-visible/device-only transition and router parity on both ranks. No MoE body, timing, or resource claim was made. | Commits `5bd6604` and `8c8b564`; M2-A/B raw evidence and hashes above; all `e0c4fd2` and failed-arm classifications preserved | **M2 IN PROGRESS/REVIEW OPEN; A/B DONE; no M2 completion claim** | M2-C1 Stage-5 publication/record44 oracle is the sole next action; root review required; M3/M4 remain blocked |
| 2026-08-12 | Root accepted C1, C2a, C2b, C3A, and C3B; keep M2 open and authorize only M2-D | C1 exported stable Stage-5 bytes and exact target record-44 state; C2a/b established allocation-free live routing; C3A made the executor initialize-once and live-state callable; C3B proved the corrected raw-FP32 attention association and one exact live layer on both ranks. No full-forward timing or performance claim was made. | Commits/checkpoints `87b15f9`, `60b7fe6`, `0409bdd`, `0f35333`; final close log and query/R1/R2/localization hashes above; all initial bad-seed, corrected-seed, negative-association, `e0c4fd2`, and rejected raw classifications preserved; validation `280/281` with one skip, `make check` 2/2 | **M2 IN PROGRESS/REVIEW OPEN; A/B/C1/C2a/C2b/C3A/C3B DONE; M2-D remains open** | M2-D eight failure arms, failure-drain/reuse, actual resource accounting, and one warmup plus three diagnostic repetitions are the sole next action; M3 is not authorized |
| 2026-08-12 | Root accepted M2-D and the M2-E cleanup; mark the exact reusable one-layer milestone DONE and authorize M3 only | D1-D3 closed all eight logical failures and the partial-enqueue error with same-executor reuse; D4D supplied one warmup plus three exact diagnostic repetitions, memory/I/O/allocation evidence, and explicit unmeasured transfer timing; D5 supplied portable collision-safe `.d4o` transport; M2-E removed 78 dead measurement lines. No full-forward timing or performance claim was made. | Eight-arm matrix log SHA `6a577181b4fe511b2012423ac38a6bd736ee775a4c609f3b9ffb00555fed9cc8`; D4D log SHA `500625ea17a78ae1577521bfb0d58088ae7805194bbf572743268a32bf5a1e4c`; final error smoke SHA `c065c90276e723a320acd98d55b616719b998659ce8785042522e26505dc76a6`; D1-D5 and final Git handoff are the closure record | **M2 DONE; exact reusable one-layer readiness only; D4D full cost model incomplete/non-gating** | M3 NEXT/AUTHORIZED, NOT STARTED: one bounded production-shaped exact 43-layer chain; root review required; M4 remains blocked |
| 2026-08-12 | Root reviewed Experiment 0087 and closed host issue/queue depth as the principal cause | P3's 645 callback rows reconciled exactly (median forward `151.070785 ms`, body `85.153019 ms`, inter-callback `62.069549 ms`); P4 found `630/630` issue starts before the preceding callback finish and zero positive late-issue lower bound. P4 step-6 `168.813209 ms` is a non-binding timing outlier; ordering remains valid. No throughput win is claimed. N2/N3 later supplied limited aggregate/source-family profiler attribution; N4 was invalid and not a mechanism rejection. | P3/P4/N2/N3/N4 raw binaries and CSVs remain at their recorded paths/hashes; full provenance and the final source correction are in Experiment 0087. | **0087 PASS_LIMITED / REVIEWED for attribution only; M3 remains REJECTED/BLOCKED; M4 BLOCKED** | The earlier Nsight subphase authorization is completed as limited evidence. The sole current authorization is a fresh-branch A2 ownership-localization discriminator, correctness only |
| 2026-08-12 | Root authorized the fresh-branch A2 ownership discriminator; preserve the 0086 rejection and stop before timing | The one-barrier arm restored exact adjacent execution; the narrow fixed per-command attention staging correction then restored exactness without a barrier. Sequential/queued1/queued2 residual hashes were `0741b15a80dc788d`, weighted hashes `0137ac8e33fadf9f`, and input hashes `12deb4effdbd4133` (all equal across arms); layer22 query index 0 was `0xbb6a` on both ranks. No timing/performance claim. | Historical no-barrier `results/dsv4-rank-local-executor/m3-a2-adjacent/a2-scratch-diagnostic.log` SHA `03694fbfc9f2ece47e45f2445ad8ab530c6910993c693454d2396cb8e4c8b0aa`; barrier `results/dsv4-rank-local-executor/m3-a2-ownership-screen/barrier-r1/combined.log` SHA `5785a80fe186f1b1d746a83fd87ac3de14b0f2a1d0fe5d6a18f7530dc2d8c9ba`; fixed-upload `results/dsv4-rank-local-executor/m3-a2-ownership-screen/fixed-upload-r1/combined.log` SHA `2724b7576948e452c69af762a33350efec5d211d3a842f564808412b860b1793`; source diff SHA `5a2ade67347565a900cca97f68fc60c7e3bc41ff2d010d63e1e1e36144c21d3e` | **A2 OWNERSHIP RECOVERY PASS at correctness boundary; 0086 remains binding historical rejection; M3 REOPENED/IN PROGRESS; no throughput claim** | After root acceptance, one bounded exact 43-layer dependent-chain correctness slice; timing/performance, graph capture, CPU optimization, Stage 7, and M4 remain blocked |
| 2026-08-12 | Execute the bounded 43-layer M3 chain-state correctness slice directly and stop before timing | Added exact hash-layer routing for layers 0--2, fixed 43-slot observations and terminal collection, then ran one sequential control and two queued chains. All 43 live observations, routes, page/KV evolution, rank associations, and terminal hidden hashes matched; eight inherited one-layer failure arms closed with exact reuse. Terminal head/logits/token and queued-chain injection were not run. | Experiment 0088; `results/dsv4-rank-local-executor/m3-43-chain/correctness-r8/`; log SHA `3eef2dde3a1cc59fbd7aceede9a9e50c95c6302e98f7f9152c0969eda2f53bb1`; source diff SHA `957e96378ff080a97ce9bc13319405e172bf2be010dc4874e3283d6bb4a77e9e`; terminal hashes `e1a9a77f0b01a361`, `122a716defe84e1b`, `5017083817dd2848`; zero checkpoint/workspace/weight-allocation deltas; memory below ceilings | **M3 43-LAYER CHAIN-STATE CORRECTNESS PASS / REVIEW REQUIRED; M3 OPEN; no throughput claim** | Stop at review boundary. No successor slice, performance, M4, graph, teacher forcing, generation, or CPU optimization is authorized |
| 2026-08-12 | Close the two remaining M3 correctness surfaces directly and preserve every failed arm | Added rank-local output-head shards and BF16 NCCL all-gather, exact terminal logits/token validation, terminal-only queued failure injection, single-owner failed-command drain, output withholding, and clean 43-layer reuse. Sequential, two candidates, and reuse matched logits hash `343d766f3f5c0af3` and token `8806`; terminal failure status reached both ranks and all outputs were zero. | Experiment 0089; `correctness-r13`; log SHA `b78ec2058773a3f24ab2a10c9d99257201b1d75db9e3d84633bb402889c23887`; source diff SHA `1d49ed2fb93d2a4f0059a5cbf5afbf8405a9bf240432a62b8de26fb219d0a162`; zero checkpoint/workspace/weight-allocation deltas; memory below ceilings; r9-r12 preserved | **M3 CORRECTNESS COMPLETE / REVIEW REQUIRED; no throughput claim** | One bounded production-shaped M3 performance falsifier is NEXT; remove/account diagnostic page and local-logit boundaries, apply `120 ms` / `36.7 GB/s` / `30 ms` gates, and keep M4 blocked |
| 2026-08-12 | Execute the sole bounded M3 production-admission falsifier; stop before timing on any mismatch | Both arms reproduced the callback-backed terminal hashes, then failed the first callback-free warm-up at `timing_vs_sequential`. A narrow prepared-query ownership event correction did not change the result and was removed. Zero timing repetitions ran. | Experiment 0090; r1 log SHA `6602535bf692da07faeb80564445b3c30febf1e10b44b4231e5007384cd9e709`; r2 log SHA `d9cd96bdf5e58a6bfcab4603b0b2af07a0f0ada02e89413d068a77339afabc49`; raw source diffs preserved | **CURRENT M3 IMPLEMENTATION REJECTED BEFORE TIMING; M4 BLOCKED** | Review, then only a one-layer callback-free first-difference discriminator; no 43-layer rerun, profiling, graph, CPU optimization, or M4 |
| 2026-08-12 | Identify the 0090 cause by measurement before accepting any correction; report no timing from any arm | Attributed the callback-free mismatch to `dsv4_prepare_attention` selecting shared pinned upload staging whenever no page callback was present, and corrected it by extending the accepted `7da38b7` fixed-per-command staging to that command. The one-layer and adjacent two-layer callback-free prefixes passed even unfixed; the 43-layer arm failed at layer 1, and layer 1's hash moved with queue depth (`f233565ecfa96477` at depth 2, `384d71deb36191ee` at depth 43), proving host-side staging reuse rather than page publication or prepared-query ownership. After the correction all 43 callback-free layers, the terminal head, and the callback-backed arms passed in one process. | Experiment 0091; negative `callback-free-r2` log SHA `5e3ad86cdef92dfa102412d66343ce51c4ca86f06da1616d2cf25eccd4c89b79`; positive `callback-free-r3` log SHA `e2a0a2aedd487e5dc2f807d22b6ea2db5bf698a510666baca5c3fc16bcfce709`; first negative `callback-free-r1` log SHA `77eed78e1405ebbeb806b52a576ba19d1594c1392f83eabf4f1f6a443eece708`; terminal `e1a9a77f0b01a361`/`122a716defe84e1b`/`5017083817dd2848`/`343d766f3f5c0af3`, token `8806`; zero checkpoint/workspace/weight allocation deltas | **M3 CALLBACK-FREE CORRECTNESS PASS / REVIEW REQUIRED; performance gates UNMEASURED; no throughput claim; M4 BLOCKED** | The production-shaped M3 timing falsifier is NEXT: account for or remove the terminal local-logit diagnostic, then apply `120 ms` / `36.7 GB/s` / `30 ms`. No graph capture, CPU optimization, teacher forcing, Stage 7, or M4 |
| 2026-08-12 | Run and review the production-shaped M3 timing falsifier after 0091 correctness closure | One warm-up and three exact callback-free 43-layer repetitions measured `114.944312 ms` wall, `73.896784 ms` routed CPU body, `46.677143 GB/s`, and `41.047528 ms` non-CPU dependency medians. The user amended M3 acceptance to median `<=115 ms`, recognizing that the old `<=30 ms` non-CPU condition incorrectly included work reserved for later CPU parity. Every repetition was exact with zero page callbacks, checkpoint I/O, or timed allocations and memory below ceilings. | Experiment 0092; `timing-r3/combined.log` SHA `fec60357ee1adc3fb4522a7f08e742a39be15869d4cd58d1161bb96479e46540`; source diff SHA `ba345132f0aeacc9f70da078bfec59a9195630f38663a32ce3682f67810c574f`; measured scope about `8.70 forwards/s` | **M3 PASS / REVIEW REQUIRED under amended gate; old non-CPU assumption FALSIFIED; no end-to-end tok/s claim** | Review successor-stage authorization; Stage 10 CPU parity remains the planned bridge for the remaining `14.944312 ms` |

The M1 result commit contains the implementation, focused tests, checker, and
wrapper. `e0c4fd2` contains the bounded M2 executor/bridges/driver and focused
order test; its original validator and measurements remain rejected historical
evidence, while the later accepted slices and their raw diagnostics are listed
above. M2 is DONE only for exact reusable one-layer readiness; D4D does not
instantiate full-resource `tau` and makes no throughput claim. No generated
artifact is tracked. The 0086 no-barrier A2 rejection remains binding history;
the barrier-positive and fixed-upload no-barrier arms classify asynchronous
staging ownership/reuse. Experiment 0087 remains PASS_LIMITED for attribution
only and N4 is INVALID. Experiments 0088 and 0089 complete M3 callback-backed
correctness; 0090 remains a binding rejection of the implementation as it then
stood, and 0091 supplies the cause it left unproven and closes callback-free
correctness. Experiment 0092 passes M3 under the user-amended `<=115 ms`
topology gate at `114.944312 ms`; the old `<=30 ms` non-CPU condition remains
a falsified planning assumption. Successor work requires review, and Stage 10
remains the planned CPU-parity bridge for the remaining `14.944312 ms`.
