# Experiment 0070: reusable DeepSeek decode graph rejection

Date: 2026-08-09  
Branch: `feat/dsv4-reference-device-runtime`  
Candidate commit: `7d4c5fd`  
Target: DeepSeek V4 Flash, exact no-speculation batch-one TP2 decode on two
RTX 3090s

## Decision

Reject the reusable outer CUDA graph. It passes the exact full-model oracle and
all structural replay gates, but it does not improve complete decode and still
misses the 100 ms/token target by 57.937 ms. Remove the runtime candidate and
retain this report. Do not optimize on top of the captured path.

## Predeclared hypothesis and gates

The retained one-final-wait baseline measured 157.549 ms/token as 130.404 ms
inside the final wait plus 27.145 ms of host issue work outside it. Routed CPU
was 90.693 ms/forward, only 2.814 ms above the accepted 87.879 ms full-store
floor. The candidate therefore targeted serial host/driver issue, not CPU
expert arithmetic: capture the fixed-address dependent chain once and replay
it with one graph launch and one final wait.

Correctness required the retained token IDs and raw-logit hashes, finite
logits, zero callback/KV/checkpoint failures, balanced leases, exactly one
synchronization, the declared 2,395,072 activation H2D bytes, and a nonempty
graph. Performance required exactly three fresh 16-step arms, each with one
capture, 16 launches and 15 replays. The target was a median at or below
100 ms/token; no improvement or a regression required rollback. Graph storage
was bounded by the existing 2 GiB workspace reserve and node count by 8,192.

## Correctness gate

The first arm rejected an internally recorded completion event used as an
external wait. The second passed that correction and rejected unsupported
elapsed-time queries between captured event nodes. Both failures occurred
after all 258 expected route records and before the exact oracle. The final
`reusable-decode-graph-r3` arm passed:

- generated IDs `[30594, 9790]`, expected raw-logit hashes, and 258,560 finite
  logits;
- one capture, one launch, zero replays and 2,772 nodes;
- one synchronization, zero checkpoint reads, KV misses/promotions and
  callback failures, and 470/470 balanced decode leases;
- 43 MoE callbacks, 258 routed experts, 43 shared experts, 43 paged-attention
  calls and 87 mHC calls;
- exactly 2,395,072 activation H2D bytes, including the declared fixed page
  patch descriptors/dummy slots.

The compiled candidate passed both CTest targets. Captured internal component
event pairs are deliberately unavailable; whole-token wall time and the sole
external completion wait remain measured.

## Three-arm replay result

All arms passed the structural gates and had zero checkpoint reads, KV
misses/promotions and callback failures.

| Arm | ms/token | tok/s | final wait ms/token | outside wait ms/token | routed CPU ms/forward |
|---:|---:|---:|---:|---:|---:|
| 1 | 157.937 | 6.332 | 148.697 | 9.241 | 89.717 |
| 2 | 154.023 | 6.493 | 145.006 | 9.017 | 85.549 |
| 3 | 158.235 | 6.320 | 149.142 | 9.093 | 89.947 |
| **median** | **157.937** | **6.332** | **148.697** | **9.093** | **89.717** |

The prior median was 157.549 ms/token and 6.347 tok/s. The candidate is 0.388
ms/token (0.25%) slower, well within normal arm variance but clearly not a win.
It removes about 18.052 ms/token from outside-final-wait host issue while the
wait grows about 18.293 ms/token. The grouped submission moves work into the
dependent wait instead of shortening the token.

Instantiate the candidate cost model as `157.937 = 148.697 final wait + 9.093
outside wait`. Routed CPU contributes 89.717 ms, leaving 58.980 ms in the wait
that is not routed CPU. Relative to the retained baseline, routed CPU improves
0.976 ms while that non-CPU dependent remainder grows 19.270 ms. The extra
27,304 H2D bytes/token are only about 0.002 ms at measured PCIe bandwidth and
cannot explain the increase. The failed mechanism targets a non-binding issue
term and alters scheduling/overlap adversely; it is rejected rather than
laundered as progress toward 10 tok/s.

## Rollback and next measurement

Revert candidate commit `7d4c5fd`; keep the accepted fixed-buffer one-wait
baseline at `f09bd5e` plus this rejection record. The active baseline model
remains `157.549 = 130.404 final wait + 27.145 outside wait`, with 90.693 ms
routed CPU and a 39.711 ms non-CPU remainder inside the wait. The next target
term is that serialized GPU/cross-engine remainder, not CPU scheduling and not
another outer graph.

The cheapest production-faithful next measurement is a decode-only CUDA
timeline of the reverted one-wait path, captured around the measured decode
window rather than the roughly 80-second model setup. It must attribute the
39.711 ms remainder among device kernels, transfers, cross-device handoffs and
idle dependency gaps before another runtime mechanism is selected. Existing
reused CUDA event counters are not additive and must not be used to invent that
decomposition.

### Decode-window timeline checkpoint

Rollback commit `2059fe6` restores the retained one-wait runtime. The installed
Nsight Systems 2024.6.2 profiler supports `cudaProfilerApi` range capture. The
measurement checkpoint adds an opt-in `STRATA_PROFILE_DECODE=1` range around
only the autoregressive loop and a reusable script that captures CUDA activity,
exports SQLite, and emits CUDA kernel, memory and API summaries. Normal runtime
execution does not call the profiler API.

The arm uses 16 decode steps so its fixed setup/measured-window ratio remains
about 31:1, but it performs only one model initialization and records only the
roughly 2.5-second decode range. Capturing the full process was rejected because
roughly 80 seconds of initialization and prefill do not test the 39.711 ms
term and would enlarge profiler overhead and output. A shorter one-token trace
was rejected because it cannot show stable repeated-token gaps or distinguish
first-step effects. The arm budget is about 85 seconds total and one process.

Correctness requires 16 final waits, zero decode checkpoint reads, zero KV
misses/promotions and zero callback failures. This is measurement only: no
throughput win is claimed and no mechanism is authorized until the exported
timeline identifies the largest non-CPU dependent term.

The opt-in profiler range compiles, the script passes `bash -n`, and the
required full `make check` passes both CTest targets in 127.38 seconds. The
single RTX timeline arm is ready.

The first profiler arm produces no valid runtime or timeline result. Nsight's
`stop-shutdown` policy terminates the target before buffered JSON/route output
is flushed: `generation.json` contains profiler progress rather than the
application object and the final route line is truncated. Nsight then reports
`Cannot find bucket for a bucket index` while processing CUDA string metadata.
The 106 KiB report contains only 23 runtime calls, 12 memcopies, two
synchronizations and no kernel table, so its successfully exported fragment is
not a decode timeline and must not be analyzed as one.

The bounded correction uses `capture-range-end=stop`, lets the application exit
normally, disables the premature profiler-stop buffer flush, and redirects the
target JSON/log inside a child shell so Nsight status output cannot contaminate
them. No runtime operation, range boundary, workload or gate changes. Repeat
the single arm only after shell validation; do not run a performance matrix.
`bash -n` and a local child-redirection round trip both pass.

The corrected second arm completes the model workload with 16 decode steps,
16 waits, zero decode checkpoint/KV/callback failures and valid JSON, but the
profiler processing defect repeats. Its SQLite has only 20 runtime calls, 12
memcopies, two synchronization records and no kernel table; Nsight again emits
`Cannot find bucket for a bucket index`. The printed 301.886 ms/token is
profiler overhead and is not a throughput result. The script's `accepted` flag
was incomplete because it gated the application but not report completeness;
this arm is rejected and no timeline attribution is made.

Do not retry late-start range capture. Both lifecycle variants lose CUDA string
metadata in this Nsight build. The full-process fallback also fails with the
same `Cannot find bucket for a bucket index` error: it produces valid
application JSON and all 16 decode waits, but its SQLite has five runtime calls
and no kernel table. The application-side decode counters from that run are
valid for workload accounting (16 waits, 31.957 ms total critical CUDA kernel
time, 47.84 us H2D and 84.928 us D2H critical transfer time, and 1.458241 s
routed CPU), but the Nsight artifact is not a timeline and the profiler-run
2.590 s decode wall time is not a throughput measurement. Nsight Systems is
closed for this question after three failed capture variants. The profiler
hook and script are removed; no Nsight result is accepted.

The required full `make check` after removing the hook passes both CTest
targets in 133.84 seconds. The remaining 39.711 ms/token baseline remainder
cannot be decomposed by this tool and needs an in-runtime measurement path.

### In-runtime per-step timeline checkpoint

The failed Nsight exporter and profiler hook are removed. The replacement
`STRATA_DECODE_STEP_TIMELINE_PATH` path records one CSV row around each of the
16 `forward_token` calls in the retained one-final-wait run. It snapshots the
existing cumulative CUDA, graph-phase and device-MoE counters before and after
the call, and records total wall time, synchronization/upload waits, critical
H2D/kernel/D2H counters, attention/mHC/MoE phase counters, and routed CPU
subphases. It creates no CUDA events, graph nodes, waits, or extra device
transfers; the CSV is buffered and is not flushed per step.

The bounded arm remains one model load with a 16-step decode window (about
85 seconds total). Its gate is valid JSON with 16 waits, zero checkpoint/KV/
callback failures, and exactly 16 data rows. This is measurement only: do not
select a mechanism until the per-step rows identify the largest serial term.

The source and script pass `bash -n` and `git diff --check`. The required full
`make check` after adding this measurement path passes both CTest targets in
119.39 seconds (119.38 seconds for `strata-tests`).

The RTX step-timeline gate passes with 16 waits, zero checkpoint/KV/callback
failures, and 154.833 ms/token for the single accounting arm. Across the 16
rows, medians are 155.004 ms total, 127.947 ms final synchronization,
87.805 ms routed CPU, and 39.847 ms synchronization-minus-routed-CPU
(38.319--42.129 ms). Critical device kernel time is only 1.885 ms; critical
activation H2D/D2H are 0.001632/0.002752 ms and upload wait is zero. The phase
spans overlap the final wait and must not be added. This rejects PCIe volume
and GPU arithmetic as the binding term and selects the serialized
CUDA-host-callback/dependency handoff for the next bounded measurement; it is
not a throughput win or a CPU scheduler re-open.

The timeline arm's `system-before.txt` records commit `6467ad2`. The retained
three-arm performance artifact under `one-final-wait-performance` records the
rejected graph commit `7d4c5fd`, so its 157.549/157.937 ms/token figures are
not a same-commit performance comparison for this arm. A fresh three-arm
baseline is required before any later candidate is judged.

That fresh baseline is now complete at commit `3ad6664`: all three arms pass
the structural zero-NVMe/zero-KV-miss/zero-callback gate, but the target rejects
the median 150.572 ms/token (6.641 tok/s). The median final wait is 127.556
ms/token, outside-final-wait work is 23.015 ms/token, routed CPU is 83.645
ms/token, and the synchronization-minus-CPU remainder is 43.911 ms/token.
This is the binding operating point for the next handoff candidate.

The next source-only measurement adds two cumulative runtime counters and two
CSV columns: the wall span from the first fixed MoE enqueue to the last host
callback completion, and the sum of callback-body durations. It uses timestamps
already taken by the callback path and adds no CUDA API call, event, wait,
transfer, or graph node. `make check` passes both CTest targets in 154.60 s.
The required RTX arm is the existing 16-step step-timeline script; no component
probe or three-arm throughput matrix is justified until this attribution is
read.

The attribution arm passes with 156.330 ms/token, 129.490 ms/token final wait,
153.744 ms/token callback-chain span, and 89.509 ms/token callback body. The
chain-minus-body remainder is 64.235 ms/token; subtracting the 26.839 ms/token
outside-final-wait issue leaves about 37.396 ms/token inside the final wait.
Callback body and routed CPU agree within 0.003 ms/token, so the remaining
target is the CUDA host-callback/dependency handoff, not another CPU arithmetic
or expert-layout change. The arm initially used the script's default result
directory because `RESULT_DIR` was not accepted; the script now accepts both
spellings so a later measurement cannot silently overwrite that directory.
The next cheap falsifier is one 16-step runtime arm with `CUDA_DEVICES=0`: if
removing the 21 cross-device attention-to-mHC transitions materially shrinks
the roughly 37--40 ms in-wait remainder, that boundary is the target; if not,
do not redesign cross-device staging and measure the same-device command
handoff instead.

That falsifier is rejected as an invalid operating point, not as a timing
result. With one visible RTX 3090 the runtime reaches the callback chain but
fails `DeepSeek attention preparation host callback failed`, followed by
`DeepSeek reference KV append accounting is invalid`, and emits no JSON. The
admitted production schedule is the two-GPU path; no one-device speed claim or
cross-device mechanism is inferred from this failed arm.

Because the admitted chain also contains one deferred attention/page-update
host callback per layer, the source-only attribution now records its chain and
body spans alongside the MoE spans. This adds no runtime operation; it closes
the risk of misclassifying attention-page callback work as CUDA launch time.
The compiled check passes both CTest targets in 140.51 s. The next arm must use
the admitted two-GPU schedule and the same 16-step timeline; no mechanism is
selected from the invalid one-GPU topology.

### Two-GPU callback attribution checkpoint

The admitted two-GPU 16-step arm at commit `ae14fc2` passes the structural
gate: 16 waits, zero checkpoint reads, KV misses/promotions and callback
failures, and the expected token IDs. It measures 2.519818817 s decode,
157.489 ms/token, and 130.572 ms/token in the final synchronization wait;
the remaining host issue work is 26.917 ms/token.

The cumulative callback spans are wall envelopes, not additive service terms:

| Span | Mean ms/forward |
|---|---:|
| MoE callback chain | 154.804 |
| MoE callback body | 90.570 |
| attention/page callback chain | 152.741 |
| attention/page callback body | 3.405 |
| routed CPU body | 90.567 |

The MoE body agrees with the routed-CPU counter within 0.003 ms/token. The
attention body is small and is nested in the same dependent callback envelope;
the two chain spans overlap and must not be summed. Using the larger body as a
conservative bound leaves about 40.001 ms/token of final-wait time outside
callback bodies (130.572 - 90.570), consistent with the earlier
37--44 ms/token remainder. This closes attention-page arithmetic and CPU
expert layout as targets. The live bottleneck is the serialized CUDA
host-callback/stream-dependency handoff and its outside-wait issue work.

This is an attribution gate, not a throughput win. The next experiment must
reduce one measured handoff term in the admitted two-GPU schedule, preserve
the one-final-wait and exact output gates, and report a fresh three-arm median.
Do not retry the rejected reusable outer graph, add another callback-body
probe, or infer a cross-device redesign from the invalid one-GPU arm.

## Branch closure audit

The `feat/dsv4-reference-device-runtime` research scope is complete. It
retains exact physical-page attention, persistent mHC, the reference-shaped
TP2 CPU routed path, GPU shared/routed joining, persistent device KV and hidden
ownership, fixed-buffer command ownership, one final wait, and the required
teacher-forced and autoregressive full-model hashes. It also ran the required
three-arm absolute performance gate. Missing the 100 ms/token target is the
measured result, not an incomplete gate.

The reusable outer graph was the branch's final declared throughput mechanism.
It passed correctness, failed its performance gate, and was removed. Subsequent
measurement bounded the retained path's remaining defect to callback/stream
handoff and host-issue time around an already floor-adjacent routed CPU body.
Continuing to implement another handoff mechanism here would start a new
hypothesis after this experiment's rejection.

The authoritative hardware checklist retains one yellow performance knowledge
gap: the live Strata mHC contribution has not been measured with a comparable
complete-chain target-point method against the reference's 0.956 ms sequence.
Reference-only knowledge gaps also remain for direct `lk_moe` callback and
phase timings, CPU counters, per-layer overlap events, and the selected NCCL
transport. These are explicit follow-up measurements, not silently skipped
integration work.

The strongest next falsifier belongs on a new experiment branch: instrument the
installed Lvllmds4-x no-spec TP2 reference at the real batch-one operating
point and separate callback enqueue-to-start, callback body, callback-finish-
to-consumer, routed gate/up, down, reduction, GPU shared expert, and TP join.
Without those reference phase measurements, the existing 93.9 ms graph span
cannot assign an individual causal budget to `lk_moe` internals.

Ignored raw data:

- `results/dsv4-reference-device-runtime/reusable-decode-graph-r3/`
- `results/dsv4-reference-device-runtime/one-final-wait-performance/run-1/`
- `results/dsv4-reference-device-runtime/one-final-wait-performance/run-2/`
- `results/dsv4-reference-device-runtime/one-final-wait-performance/run-3/`
- `results/dsv4-reference-device-runtime/one-final-wait-performance/summary.json`
