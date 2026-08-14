# Experiment 0087 — current DSV4 callback-gap attribution

Date: 2026-08-12
Branch: `exp/dsv4-current-callback-gap`
Disposition: **PASS_LIMITED / REVIEWED for aggregate/source-family attribution; not a throughput win**

## Question and contract

The accepted 0082 centralized control is 151.155686 ms/forward, of which
84.710043 ms is routed CPU body and 66.445643 ms is the makespan residual.
This experiment asked whether the residual was principally a late host issue
or an already-issued cross-callback dependency chain. The primary metric was
the non-overlapping callback identity and its issue-start ordering, not a
candidate throughput number.

Correctness required the existing exact decode output, 15 forwards, 43
callbacks per forward, monotonic timestamps, and a zero/rounding-only
reconciliation residual. The memory gates were 21,287,272,448 B per GPU and
231,928,233,984 B RSS; steady-state checkpoint/KV I/O, decode allocations,
callback failures, misses, promotions, and demand H2D had to remain zero.
Rollback was to preserve the raw arm and leave the source uncommitted if the
trace changed default behavior, failed identity validation, or could not be
reconciled. The historical source change was a default-off `steady_clock`
trace hook only; it did not alter inference, CUDA scheduling, synchronization,
or arithmetic, and it is removed in this closure.

The governing model remains:

```text
tau = max_r(W_r / B_r) + Sigma_serial
```

No full-resource `tau` is claimed here because transfer service and physical
NCCL timing were not measured. Unknowns are reported as `NOT_MEASURED`, never
as zero service.

## Operating point and workload comparability

Both arms used the current Release runner, `models/dsv4f`, CUDA visible
devices 1,2 with runtime ranks 0,1, batch one, no speculation, the corrected
104-token prompt (SHA
`082d43114147a15eb345cc7eb2bba99a920299cacacfe5c45823bffb729c15fe`), 16
requested tokens/15 timed forwards, 28 host attention threads, explicit
21,256,421,376 B admission, and zero decode I/O. These fields match the
accepted 0082 operating point.

| field | Strata trace arm | external reference | comparability |
|---|---|---|---|
| model/precision/TP | dsv4f, BF16, TP2 | dsv4f, BF16, TP2 | exact match |
| prompt/batch/speculation | 104-token corrected prompt, batch1, none | same prompt, batch1/no speculation in client record | exact match in declared fields |
| timed work | 15 Strata forwards from 16 requests | external client has 16 completion tokens; 94.282 ms is an internal graph phase | material boundary difference |
| execution mode | current Strata eager/decode path | external CUDA graph/full decode path | material difference |
| devices/P2P | visible 1,2, runtime 0,1, no P2P | TP2; external custom P2P allreduce disabled | mostly matched; physical placement detail is not identical |
| CPU/NUMA | 28 host attention threads, current Strata admission | server/client CPU placement in external records | exact placement equivalence NOT_MEASURED |
| residency/I/O | explicit VRAM cap, zero decode checkpoint/KV demand | GPU memory util .87, KV FP8, external cache policy | material runtime-contract difference |
| output | exact Strata token sequence and zero callback failures | external request output record | not an equal-scope teacher-forcing proof |

Therefore the 94.282 ms number remains feasibility/navigation context, not an
equal-scope pass or a causal decomposition target.

## Provenance and commands

P1 read-only reconstruction inputs (SHA-256): 0081
`bae963f8ca6e195a2fe20fca98773165af495be2067fd602a7e9f6273fad12c6`, 0082
`a93f15adf794a4191ac697294f9e9d9667a3a7a9ab0a893802d2df51de6bf12a`, 0055
`39eefe11963396bec60f1fb58120d2864b137dc4fdbfb07d595c5f055e52ca8a`,
`hardware/temp.txt`
`319c7fe7a3df38cc4d689eb034831a0c30efdf529bd3eb367c8d6ad894b11534`, and
the canonical mission before this closure
`180c7a0044721f7d2d3cffef4e7b2b9218b0dd1d06875746cee5a5ff3f4d3480`.
The Strata step-2 inventory is
`dee01c2c198771559a2014475d501db44bc45c31d63e1508b532d217ed711fc7`; its
external client and serve hashes are
`ae467f355d0c1eb35b68df92a88cc2673a2047200fc5f808a46a36d6d1052338` and
`7f564f862f35a8be1b3f4bec37d515926db694211fc9d6a40e94a1b6e73fd531`.
The step-3 inventory is
`87a475f095645e46646114003632728eeccad759b315ba0a2ec01e823ff365`; its
client/serve hashes are
`09d159e829ffcb769adb3e2fb1f2315cc674a482ac2b4cf11ae3b801e85437e7` and
`9ed30c035f72ad0089f927fa2a57f4a6af9a7098de3d7ffc07851e07c3d592a2`.

The P3 command was the existing wrapper with explicit overrides, not a cloned
runner:

```text
env CUDA_VISIBLE_DEVICES_SET=1,2 STRATA_DSV4_RUNNER=/home/rodrigo/Developer/strata/build-release/strata-deepseek-run STRATA_DSV4_BYTE_ADMISSION_RESULT_DIR=/home/rodrigo/Developer/strata/results/dsv4-current-callback-gap/trace-r1 STRATA_DSV4_RUNTIME_DEVICES=0,1 STRATA_DSV4_MAX_CONTEXT=256 STRATA_DSV4_MAX_NEW=16 STRATA_DSV4_HOST_ATTENTION_THREADS=28 STRATA_DSV4_VRAM_BUDGET_BYTES=21256421376 STRATA_DSV4_DECODE_CALLBACK_TRACE_PATH=/home/rodrigo/Developer/strata/results/dsv4-current-callback-gap/trace-r1/callback-trace.csv /home/rodrigo/Developer/strata/scripts/run_dsv4_vram_byte_admission.sh
```

P3 raw directory: `results/dsv4-current-callback-gap/trace-r1`, named session
`dsv4-current-callback-gap-p3`. The pre-correction binary SHA is
`271b477eed13f1ffb5740c017e37092fd10a3537147c26280f1fac7d6d74cf0a`, source
diff SHA `3dd2ef16200cce00a2af6fc3abcd9df3df49548c2185d54876e924dcd4b50d94`,
CSV SHA `a67d7ad65d6bf952c19b2e361b9bbde87f72edb1ab9d75874aa0e10ebc692855`,
generation JSON SHA `36c114a15de3dfc683a064ff2732e4b02850c98b63868cebffe72f5d7d43addf`,
generation log SHA `4c1aea2ae743b22874aa83dbf9d4fffd050ad33aa04aa6bbefbf4d9f40bf89dc`,
and provenance SHA `2f7447c43768718e6f88e87948b59d6f04d5105aaa6c8346331e8a10b8c53319`.

P4 used the same wrapper and one fresh directory,
`results/dsv4-current-callback-gap/issue-r1`, named session
`dsv4-current-callback-gap-p4b`; its pre-correction binary SHA is
`7e416765da5cf068e5a54ed87c59e061b9cb2d26c4e8afbdbcdd6a00bc584012`, source
diff SHA `8b36e4273962e9b9bbd6211bf6864f3240fa11d899a48612d4df1e2c2ef47948`,
CSV SHA `cdfcf7027d1e6b640c3362c5ae5643e2f1620f4ecfecf2cb87671a2994e6b097`,
generation JSON SHA `109a071d2e4a38a386f143f71093178ab01fcccb9efd03c15494cea65e342052`,
generation log SHA `fa38bcbd5720e5b21d043868a9523afc277dc276f26c743a302dc179c5815fbd`,
and provenance SHA `8637b2a2669f9f08219b784b10d11051bc7cbd9349745351f738ec1b263b78f5`.
These binaries are intentionally retained as the provenance of P3/P4; the
final source correction is validated by the closure build and is not used to
rewrite their hashes. The final Git handoff carries the result commit hash;
this document does not self-reference it.

## P3 callback identity

There were 645 rows (15 forwards x 43 layers), all `trace_valid=1`, zero
errors, monotonic timestamps, and residual <=1 ns. The wrapper passed exact
tokens, zero callback failures, zero checkpoint/KV/miss/promotion/demand-H2D
bytes, zero decode allocations, and the RSS/GPU ceilings.

| step | forward | initial | callback body | inter-callback | finish->drain | post remainder |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 150.875236 | 1.727677 | 85.392132 | 61.790994 | 1.548939 | 0.415494 |
| 2 | 151.070785 | 1.648754 | 85.735671 | 61.773328 | 1.547161 | 0.365871 |
| 3 | 151.545060 | 1.610249 | 85.733518 | 62.290165 | 1.563535 | 0.347593 |
| 4 | 153.217270 | 1.692454 | 84.676805 | 64.945454 | 1.521606 | 0.380951 |
| 5 | 152.913046 | 1.586959 | 87.019388 | 62.373094 | 1.557576 | 0.376029 |
| 6 | 153.983165 | 1.596404 | 88.861197 | 61.605835 | 1.554183 | 0.365546 |
| 7 | 149.310359 | 1.587233 | 83.931866 | 61.870613 | 1.564990 | 0.355657 |
| 8 | 153.348417 | 1.605962 | 84.956955 | 64.900646 | 1.545799 | 0.339055 |
| 9 | 149.188638 | 1.576869 | 83.744250 | 61.917214 | 1.559825 | 0.390480 |
| 10 | 149.065317 | 1.599863 | 83.484658 | 62.043547 | 1.558494 | 0.378755 |
| 11 | 153.093714 | 1.667495 | 86.050646 | 63.025819 | 1.951486 | 0.398268 |
| 12 | 155.660179 | 1.779302 | 85.153019 | 66.166979 | 1.556867 | 1.004012 |
| 13 | 149.322748 | 1.673539 | 83.340847 | 62.385820 | 1.542559 | 0.379983 |
| 14 | 150.908772 | 1.588274 | 85.346715 | 62.069549 | 1.541803 | 0.362431 |
| 15 | 148.541318 | 1.619862 | 82.976764 | 62.052396 | 1.553498 | 0.338798 |

Median/range in ms: forward `151.070785 / 148.541318–155.660179`, initial
`1.610249 / 1.576869–1.779302`, body `85.153019 / 82.976764–88.861197`,
inter-callback `62.069549 / 61.605835–66.166979`, finish-to-drain
`1.554183 / 1.521606–1.951486`, and post `0.376029 / 0.338798–1.004012`.
The mean forward was 151.469601600 ms and the mean residual envelope was
66.375972867 ms. Median body share was about 56.21%; the remaining envelope
was about 43.79%. The raw callback-body sum was 1,276.404431 ms versus the
JSON routed CPU body 1,276.346988 ms, a 0.057443 ms total boundary difference
(3.83 us/forward).

The inter-callback term is distributed: layers 1..42 have per-layer median
gaps approximately 1.430–1.558 ms. The ten largest observed gaps were
1.759841 ms (step12/layer32), 1.759518 (12/28), 1.720725 (12/36), 1.712049
(12/40), 1.700275 (8/20), 1.697563 (12/8), 1.695057 (12/24), 1.693487
(4/8), 1.690448 (4/36), and 1.690415 (8/28). These are spread across the
chain; the list does not identify a causal layer type.

## P4 issue-start attribution

P4 retained the accepted P3 identity fields and added the existing
`execution_started` boundary. Across all 630 transitions, before/equal/after
previous callback finish was **630/0/0**. The positive late-issue lower bound
was 0 ms per forward; every issue start preceded the prior callback finish.
The median queue age was 65.682474 ms (range 0.849035–138.249886 ms), and the
median issue-start span was 18.201143 ms (range 17.846350–35.710359 ms).
The median per-layer issue lead was negative at layer1 (-2.713290 ms), layer21
(-63.445300 ms), and layer42 (-125.462000 ms). Thus host issue/queue depth is
closed as the principal cause: the approximately 62.07 ms term remains after
commands were issue-started, in stream/device/transfer/dependency execution.

One P4 forward had a 168.813209 ms body/forward outlier (step 6). P4 timing is
therefore non-binding for performance, but its ordering and lead result remain
valid because the issue timestamps and monotonicity gates passed. The P4 JSON
host-side graph envelopes were attention 16.495527 ms, MoE graph 3.417252 ms,
MoE prepare 0.163204 ms, mHC post 3.367882 ms, and mHC pre 0.117947 ms per
forward. They overlap and are not added to wall time.

## Cost-model reconciliation and decision

Directly measured per-forward terms are callback body (CPU service), initial
dependency, inter-callback chain, chain-drain, and post remainder. The
non-overlapping P3 identity is:

```text
forward = initial + sum(callback bodies) + sum(inter-callback gaps)
        + callback_finish_to_chain_drain + post_chain_drain_remainder
```

It reconciled every row to <=1 ns. The Strata side can therefore instantiate
the measured makespan as CPU body plus a residual envelope, but not as a sum of
overlapping CUDA/host graph counters. Routed CPU payload is
3,449,290,752 B/forward at 84.710043 ms in 0082 (40.718794 GB/s). The P3
callback body is the same boundary at 85.153019 ms median. Attention, MoE graph,
mHC, H2D/D2H transfer throughput, physical NCCL bytes, and per-subphase CUDA
service are overlapping or **NOT_MEASURED** for this arm. Checkpoint/KV demand,
allocations, and failures were zero under the declared gates; they are not
negative resource terms.

The exact wall difference is `151.155686 - 94.282000 = 56.873686 ms`, but it
cannot be reconciled into equal-scope causal components. The only honest
Strata decomposition is 84.710043 ms CPU plus 66.445643 ms residual; the
external 94.282 ms graph is a different execution mode and its 72.889 ms body
is navigation context. Any subtraction between those numbers leaves a mixed-
scope/unreconciled remainder, not a measured saving.

The 0082 goal gap is 51.155686 ms. CPU parity can offer at most 11.821043 ms
from the external 72.889 ms navigation body, leaving a topology/non-body
opportunity of at least 39.334643 ms; those opportunities cannot be stacked as
guaranteed savings. Without CPU improvement, the current body alone leaves at
most 15.289957 ms for all non-body work at 100 ms. With the 72.889 ms body
context, all non-body work would need to be <=27.111 ms. The distributed
62.07 ms inter-callback chain is the only directly measured non-body term large
enough to carry the required >=39.33 ms topology removal, but this is a target
gate, not architecture acceptance.

The upgraded N2 Nsight capture and N3 reclassification are **PASS_LIMITED** for
aggregate temporal and source-family attribution only. They are not a
production performance or memory pass and do not select a mechanism. The
corrected callback schedule is exactly `1305 = 15 * (43 attention/page + 43
MoE + 1 output head)`; the prior 1290 expectation was wrong. Over the 630
inter-MoE gaps, the profiler observed a `989.680661 ms` envelope; combined
attention GPU union was `669.499507 ms` total, median `44.637142 ms/forward`
(`67.648%`), all GPU plus attention-callback union was median `56.033489
ms/forward`, and uncovered median was `9.475883 ms/forward`. These values are
profiler-perturbed coverage, not savings or production timing. Non-additive
service sums were q_a `26.304575 ms`, q_b `171.051807 ms`, wkv `18.472909 ms`,
wo_a `80.291723 ms`, wo_b `145.583322 ms`, query-rank norm `152.532777 ms`,
query norm/RoPE `56.813280 ms`, and key/value norm/RoPE `79.569172 ms`.

The raw N2 report, SQLite, generation, and manifest hashes are respectively
`a8f8bc7c0c9d2e8e9d2f1360a1b2349bd431161313b874a155dea66a862936a2`,
`7735fd8f39e8418092b85e532a692a8b684ce6779aedbdad830bbd2379e7008a`,
`97e4aa16eb52d2477afbf6a0b9d2afa71190f65743d621bab79b878cce70d1d5`, and
`729e2d723dbac1453a14ed1da341cc3cf64ddde297fc6b4442b3abe689028e1d`.
The N1 default-off profiler source hook was diagnostic-only and is removed in
this closure; its raw profile remains preserved. No production source change
or throughput claim remains.

The N4 fixed-event arm is **INVALID**, not a mechanism rejection. Its literal
prompt pathname produced `prompt_tokens=31` and wrong token IDs, and the
existing scalar event storage produced device-0 zeros versus device-1
nonzeros, so it cannot produce the requested two-device envelope; no retry was
made. Its summary, generation, and manifest hashes are
`c5cfdd058fc9d676865a28b9a0c52c67f998fa8cfc2712af2788cebbb63a8232`,
`c905939d72180989a20adec78fe0c6dea503167de206830728163c343a3c8d32`, and
`3aa4b1eb59c8c0e59b5a76a9e9fb4f363d5514760d45969a4c1001ff03d4361a`.

The binding unprofiled control remains `151.155686 ms/forward`, `84.710043
ms` CPU body, `66.445643 ms` residual, and `51.155686 ms` goal gap. Experiment
0076's equal-scope attention saving remains only `1.513952 ms` (`22.388704 →
20.874752 ms`); the old synthetic `58.721 ms` is not a promise.

## Rejected evidence and handoff

The 0081 memory-gated rejection, 0086 A2 multi-layer correctness rejection,
P3/P4 pre-correction binaries, P4 timing outlier, and N2/N3 profiler artifacts
remain preserved at their original paths and hashes. P0 one-layer capability
remains accepted; A1 is historical/reverted and A2 is rejected before timing.
N2/N3 are limited attribution evidence, N4 is invalid with no retry, and no
full-forward or 10 tok/s claim is made by 0087.

The next authorized action is one fresh-branch, correctness-only A2
ownership-localization discriminator. Reuse historical A1 commit `3be4da4`
and the preserved A2 source; compare the identical queued layer21→layer22
harness with exactly one main-stream completion barrier after layer21. If it
restores exactness, classify async ownership/reuse and return to root; if not,
stop and pivot. This has no timing claim and authorizes no backend redesign,
full 43-layer integration, graph capture, CPU optimization, or Stage 7. The
0086 record remains the binding original A2 rejection; this closure does not
relabel that negative arm. M4 remains blocked.
