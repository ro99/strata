# Experiment 0080 — rank-local full-dependent-chain falsifier — 2026-08-11

Status: **REJECTED / BLOCKED / REVIEW REQUIRED**. User review authorized this
bounded Stage-6A falsifier despite the Stage-5R residual rejection because the
isolated MoE probes cannot measure the cross-layer overlap that the topology is
intended to create. That amendment does not relabel 0077, 0078, or 0079 as
passes. This experiment stopped after the cheapest faithful control gate; no
rank-local candidate, Stage 7 work, graph work, or CPU arithmetic work was
started.

Branch: `exp/dsv4-rank-local-full-chain`
Base and current pre-result HEAD: `78f49bed6d7fdbad4f4dee2e61f9bbe28b321599`
Target: exact no-speculation batch-one decode on two RTX 3090s at
`<=100 ms/forward` (10 tok/s).

## Decision

The hypothesis was that removing the central `mhc_slot` owner and running the
full dependent layer sequence as two rank-local chains could shrink the
approximately 62.986 ms dependent-gap opportunity. The primary metric was a
fresh complete dependent-chain wall time, with CPU routed bytes/body and the
serial remainder priced by

```text
tau = max_r(W_r / B_r) + Sigma_serial
```

The cheapest actual-runtime control completed one 104-token prompt and one
decode forward. It generated the expected short-arm token IDs `[43, 8806]`,
had 43 attention calls, 43 MoE calls, 87 mHC calls, zero callback failures and
zero non-finite logits, but measured:

| quantity | fresh control arm |
|---|---:|
| complete decode wall | **634.398071 ms/forward** |
| routed CPU body | **542.039955 ms/forward** |
| routed CPU body per layer | 12.605580 ms |
| decode GPU critical synchronization | 528.680659 ms |
| decode GPU critical kernels | 12.650720 ms |
| attention graph counter | 68.492104 ms |
| mHC post/transition graph counter | 14.062838 ms |
| device shared-collect envelope | 528.743323 ms |
| decode checkpoint reads | 0 calls / 0 B |

The routed payload is the accepted 43-layer value
`3,449,290,752 B`. The measured CPU rate is
`3,449,290,752 / 0.542039955 = 6.363536 GB/s` (5.926505 GiB/s), below the
predeclared `>=36.7 GB/s` control-validity gate and the roadmap's accepted
runtime comparison of about `40.6 GB/s` (`3,449,290,752 B / 84.915 ms =
40.620512 GB/s`). The CPU argmax therefore fails before a candidate can be
judged. Its wall time is also 524.398071 ms above the 110 ms cheap-kill
ceiling. This is a control-validity/performance rejection, not evidence that a
rank-local topology would have a measured speedup.

The first diagnostic arm used `--layer-hash-trace` to exercise operation
boundaries and produced the same exact token IDs, but that option selects the
synchronous diagnostic branch and adds host-visible mHC state transfers. Its
769.046441 ms decode is retained for correctness context only and is explicitly
not a performance datapoint.

Because the admitted control does not reach the required production CPU
operating point, the candidate was not implemented or launched. There is no
candidate/control comparison, no Stage-6A acceptance, and no authorization for
Stage 7. Human review must decide whether to repair the current runtime control
to the declared production CPU gate as a new bounded prerequisite. Continuing
from this failed control by concatenating independent replay fixtures would be
an invalid experiment.

## Predeclared hypothesis, resources, gates, and budget

The target resource was the dependent serial term between attention, mHC,
router, CPU/shared MoE, join, TP reduction and the next layer. The expected
binding resource was the routed CPU body; a successful topology candidate had
to preserve that body while reducing cross-layer `Sigma_serial`. The signs
were declared before execution:

| resource/term | expected sign of a rank-local candidate | measured here |
|---|---|---|
| routed CPU expert bytes/body | must not increase; reducing the argmax is required | **positive load: 542.039955 ms; 6.363536 GB/s** |
| GPU attention/mHC/shared service | may overlap, but extra rank-local state/launches add load unless hidden | kernels and graph counters measured; candidate delta not measured |
| H2D/D2H | must remain fixed/device-resident; extra transfers are negative | control decode activation H2D/D2H 2,367,768/1,693,016 B; candidate not measured |
| NCCL data/status and SHM transport | serial/physical traffic must be measured, not free | no Stage-6A candidate collective; physical SHM not measured |
| callback/host issue and fill/drain | target only if it is removed by the full chain | current control has one decode synchronization; candidate delta not measured |
| memory/I/O/allocations | must stay within ceilings and zero steady-state decode I/O | host/RSS and runtime counters recorded; VRAM plan exceeds the declared per-rank ceiling |

The hard gates were exact actual-format arithmetic and routes, complete
dependent state, both-rank failure closure before and after every collective,
zero non-finite values, no hidden fallback or central owner, fixed buffers and
one final logical completion, zero decode checkpoint reads/KV misses/promotions,
zero timed allocations, and the memory ceilings of
`21,287,272,448 B/GPU` and `231,928,233,984 B host`. Rollback was mandatory on
any gate failure, a central continuation, or a complete-chain result that could
not plausibly reach 110 ms.

The single arm's fixed setup was approximately 110.858742 s initialization,
107.784710 s resident staging and 26.346146 s resident warmup for a
0.634398071 s decode window: setup/decode was about 175:1. A three-arm
full-model matrix was rejected as the more expensive next experiment. The
cheaper 0078/0079 replay-chain probes were also rejected for this question:
they do not carry dependent mHC/attention state and their production-shape
control failed the later 6.7-vs-40.6 GB/s audit. Their raw negative evidence is
preserved and is not reused as Stage-6A timing.

## Faithfulness audit

The mandatory audit was performed before considering implementation. The
existing replay formats are sufficient for their own component contracts but
not for a true 43-layer dependent rank-local chain:

| required state | existing evidence | Stage-6A sufficiency |
|---|---|---|
| attention input, query/KV/compressor/index values, physical pages/candidates, centralized branch output | `Dsv4AttentionReplay` fields in `include/strata/dsv4_attention_replay.hpp:25-50`; 645 actual-format files from 0076 | component boundary only; no rank-local mHC residual/next-layer state |
| MoE hidden input, router logits, six routes/coefficients, two FP32 CPU partials, centralized accepted output | `Dsv4MoeReplay` fields in `include/strata/dsv4_moe_replay.hpp:11-27`; 5,117 captures from 0077 | component boundary only; no preceding attention/mHC dependency |
| rank-local mHC pre/post mix, four-copy residual, normalized next-layer hidden, page state and failure status | current runtime and target-format mHC contract | absent from both replay schemas |
| attention TP result -> mHC transition -> router -> MoE -> next-layer state | current runtime's live dependency chain | not reproducible by joining files keyed only by layer/position |

The accepted runtime has a real dependent sequence, but its implementation
still stores one global `mhc_slot` (`src/deepseek_runtime.cpp:1406-1413`) and
uses `devices[mhc_slot]` for the deferred shared/MoE path and mHC ownership
(`src/deepseek_runtime.cpp:3762-3835`, `:4883-4912`, `:5540-5584`). It is a
centralized control, not the requested candidate. Therefore no independent
layer inputs were concatenated and no synthetic delay or fabricated overlap was
introduced.

## Control arms and exact evidence

Both arms used `models/dsv4f`, `CUDA_VISIBLE_DEVICES=1,2`, runtime devices
`0,1`, batch one, no speculation, the corrected prompt at
`results/dsv4-lk-moe-phase-profile/step2-prompt.txt`, `--max-context 256`,
`--device-resident-runtime`, 28 host attention threads and `--max-new 2`.

### Diagnostic correctness arm (not a timing arm)

The named tmux session `dsv4-stage6a-control` ran the exact runtime with
`--layer-hash-trace --logit-trace`. It exited 0 after 3:16.19 wall, with
154,023,936 KiB maximum RSS. The arm reported 43 decode MoE calls,
`host_callback_failures=0`, 258,560 finite logits and generated IDs
`[43,8806]`. It measured 769.046441 ms decode and 535.197882 ms routed CPU;
the layer trace changes branch behavior and is not used for the control gate.

Artifacts (ignored):

```text
results/dsv4-rank-local-full-chain/stage6a-control-one/generation.json
  bad7a036a2a8be0db506d6d673c4b3a9b7178039eb8a77f4d636aaa15bb41141
results/dsv4-rank-local-full-chain/stage6a-control-one/generation.log
  22ae8ecc451d72deab15aab451da51f60b2c30929e9f101258676559eaececed
```

### Binding no-trace production control

The second named session `dsv4-stage6a-control-prod` ran the same command
without layer tracing and exited 0 after 3:05.08 wall. Its exact oracle and
structural observations are:

```text
prompt_tokens=104, generated_tokens=2, decode_steps=1
generated_token_ids=[43, 8806], answer="I notice"
attention calls=43, paged-attention kernels=817
MoE calls=43, routed experts=258, shared experts=43
mHC calls=87, mHC kernels=260
host_callback_failures=0, non_finite_logits=0
decode checkpoint reads=0 calls / 0 B
decode KV misses=0, promotions=0, CUDA workspace allocations=0
```

The measured phase counters are envelopes/cumulative spans and must not be
added as if independent:

| decode term | ms/forward | interpretation |
|---|---:|---|
| complete wall | **634.398071** | measured one-forward control |
| routed gate/up | 354.790392 | CPU body subphase |
| routed down | 167.099457 | CPU body subphase |
| routed reduction | 8.291372 | CPU body subphase |
| routed CPU total | **542.039955** | `argmax` work term; includes route/conversion overhead |
| shared-collect envelope | 528.743323 | completion wait envelope overlapping CPU callback work |
| CUDA critical synchronization | 528.680659 | not an additional wall term |
| CUDA critical kernel | 12.650720 | GPU service envelope |
| attention graph counter | 68.492104 | graph phase counter, overlaps the dependent path |
| mHC post/transition | 14.062838 | graph phase counter, overlaps the dependent path |
| activation H2D / D2H | 2,367,768 B / 1,693,016 B | control bridge volume; not candidate evidence |

The directly measured control decomposition is therefore
`634.398071 = 542.039955 CPU body + 92.358116 residual envelope`. The latter
is not a claimed additive `Sigma_serial`: the CUDA synchronization, shared
collect, attention and mHC counters overlap and their clock boundaries do not
identify one serial interval. For the only measured resource term,

```text
W_CPU = 3,449,290,752 B
B_CPU = 6.363535972 GB/s
W_CPU / B_CPU = 542.039955 ms
argmax_r = routed CPU body
```

No candidate quantities exist, so candidate `Sigma_serial`, NCCL data/status,
physical SHM traffic, rank imbalance, candidate H2D/D2H, local join and
publication cannot be honestly filled in; they remain `not_measured`.

The control process reported `rss_bytes=157,733,453,824 B`, below the host
ceiling by 74,194,780,160 B. Its runtime memory plan was
`total_vram_budget_bytes=47,514,353,664 B`, or 23,757,176,832 B/rank, and
`nvidia-smi` reported 23,787,077,632 B and 23,789,174,784 B. Both exceed the
declared 21,287,272,448 B/GPU ceiling by approximately 2.50 GB. This is another
unresolved control gate, not a candidate pass; no Stage-6A memory claim is
made from it.

## Why no candidate was built

The faithful candidate would have required new capture or a source-level
rank-local runtime implementation carrying every mHC residual/page/workspace
and attention/MoE collective dependency. The existing runtime is centralized,
and the no-trace control itself misses the CPU production gate by 30.34 GB/s
and 432.039955 ms over the 110 ms kill ceiling. Implementing a rank-local
candidate on top of that invalid control would conflate a scheduler/layout
repair with topology overlap and could not produce an equal operating-point
comparison. The predeclared stop condition therefore fired.

The least expensive corrective prerequisite is one new control-only arm that
reconciles the live `HostWorkerPool` callback path with the accepted 84.915 ms
CPU-body reference: instrument worker/node placement, callback enqueue-to-start,
per-phase body, and completion without changing arithmetic or workload. It must
first pass `>=36.7 GB/s`, the host/RAM and VRAM ceilings, zero decode I/O and one
final completion. Only then could a separately reviewed capture/implementation
question be considered. This recommendation does not authorize Stage 7 or
retroactively pass Stage 5R.

## Preserved rejection and handoff

Experiments 0077, 0078 and 0079 remain binding negative results for their
declared scopes. Stage 5R's 4.873485 ms residual remains above its 2.125 ms
allowance; Stage 5R.1 selected no correction. Stage 6A is **REJECTED / BLOCKED /
REVIEW REQUIRED** at its corrected-control gate, and Stages 7, 8 and 9 remain
blocked. Stage 10 remains separate/deferred. No Stage-6A candidate, mHC
integration, graph capture, CPU arithmetic change, speculation or workload
change was started.

The tracked reusable control wrapper is
`scripts/run_dsv4_stage6a_control.sh`. Its default leaves layer-hash tracing
off, refuses to overwrite result directories, validates the 43-layer/zero
decode-I/O/callback gates with `jq`, and records environment, summary and
SHA-256 files under ignored results. It was syntax-checked; generated result
artifacts remain outside Git.

Binding no-trace artifacts:

```text
results/dsv4-rank-local-full-chain/stage6a-control-production-one/generation.json
  52339d9368b66ded58f8c54f7bfd1a7a71f70eb6287ce683a9d4fb60a4741775
results/dsv4-rank-local-full-chain/stage6a-control-production-one/generation.log
  d75f0642863455ccee6777f6ed1f4eff8da0c9fda769a4fac69aeecaa251f4a9
results/dsv4-lk-moe-phase-profile/step2-prompt.txt
  082d43114147a15eb345cc7eb2bba99a920299cacacfe5c45823bffb729c15fe
```

The roadmap is ignored by repository policy and is updated locally with this
record; it is not force-tracked. The single result commit contains only this
tracked experiment record and reusable control script. Stop at the Stage-6A
review boundary.
