# Experiment 0062 — authorized external mHC arithmetic acceptance

Status: **the native operation gate passes under the explicitly authorized
`lvllmds4x-sm86-v1` contract.  The mHC component contract is accepted and
remains a prerequisite for the 10 tok/s device pipeline.  A full-model
diagnostic confirms that mHC alone does not repair Strata's pre-existing
external generation divergence.**

The retained result is an SM86 CUDA operation probe which reproduces every
tested fused Lvllmds4-x post-to-pre-to-RMSNorm output bit exactly on four real
target-format fixtures.  The standalone graph-entry pre path differs only at
the reference TF32 reduction association: at most `2.9802322e-8` in the post
mix, `7.4505806e-9` in the combination mix, and one BF16 code / `0.001953125`
in the normalized layer input.

A deliberately host-staged version of that accepted arithmetic was then used
for one full-model correctness arm.  It was active -- all 1,505 layer hashes
and all 16 raw-logit hashes changed relative to current Strata -- but it did
not move the generated token sequence toward the external control.  The prompt
IDs were exact, the first five generated tokens matched, and only 6 of 16
top-1 tokens matched across the complete response.  The candidate generated
the same 16 token IDs as current Strata.  This falsifies adopting mHC arithmetic
alone as a sufficient full-model parity step.  It does not falsify the
reference's measured 0.957 ms fused kernel or the complete device-resident
pipeline.

The disposable host-staged runtime harness and its CLI switch were removed.
That harness was not the external persistent fixed-buffer architecture.  Its
removal does not reject the accepted mHC arithmetic or the retained native
probe.  No throughput repetitions were run and no speedup is claimed.

## Lineage and bounded decision

- base: `38bebf3 docs: record the external mHC contract and timing gate`
- branch: `exp/dsv4-reference-mhc-contract-acceptance`
- external source: `/home/rodrigo/Developer/Lvllmds4-x` at
  `691cc0ae2056dc07ed23e4bc4f1dac25b7582f77`
- target model: `DeepSeek-V4-Flash-0731`, full 43-layer target checkpoint
- target hardware: two RTX 3090 SM86 GPUs plus the admitted RTX 5060 Ti

The numerical-contract decision was explicit: reproduce the visible
Lvllmds4-x arithmetic and test it before authorizing the device pipeline.  It
did not authorize a silent precision change, fallback, router change, or a
different model-level correctness standard.

The predeclared model remained:

```text
tau = max_r W_r / B_r + sum_serial

current total median:          244.812 ms/token
complete MoE:                  114.667 ms
routed CPU median:             106.869 ms
routed CPU best:                89.226 ms
standalone routed CPU:          87.879 ms
maximum median CPU opportunity: 18.990 ms
serialized non-MoE path:       130.145 ms
target:                         98--100 ms/token, >=10 tok/s
```

- **Hypothesis.**  Accepting and reproducing the external mHC association is a
  prerequisite for keeping the smallest complete hidden/attention/KV/mHC chain
  device-resident and reducing the serialized 130.145 ms term.
- **Target term.**  Serialized host mHC and hidden-state handoff inside the
  130.145 ms non-MoE path; CPU scheduling was not reopened.
- **Primary metric.**  Real-format operation agreement for the isolated mHC
  component.  Exact external full-model agreement is a final complete-pipeline
  integration metric because current Strata already diverges outside mHC.
- **Correctness gate.**  Finite real-format layer fixtures under the declared
  component contract, with no fallback.  Full-model prompt/generation was also
  recorded as a diagnostic, but cannot reject one component when the unchanged
  baseline already fails that whole-model oracle.
- **Memory ceiling.**  Existing 0.95 per-GPU admission and 216 GiB host limit;
  the full 32,768-token compact KV allocation remains 120,148,480 bytes.  The
  candidate changes no weight, activation, or KV precision.
- **Bottleneck resource.**  Current `sum_serial`, not CPU routed DRAM.  A
  retained persistent implementation would trade host folds, handoffs, and
  synchronization for GPU SM/HBM work while leaving CPU routed bytes fixed.
- **Other resources.**  The operation probe adds CUDA projection reads and
  fixed activation buffers, no NVMe dependency, and no routed-CPU work.  The
  disposable full-model checker intentionally adds per-boundary allocation,
  H2D, D2H, and synchronization and therefore cannot measure performance.
- **Rollback.**  If any operation fixture failed the declared contract, reject
  the mHC candidate.  The full-model mismatch still required removal of the
  disposable harness and stopped a timing matrix, but it rejects only the claim
  that mHC alone establishes whole-model parity.

## Cheapest gate: real-format operation fixtures

Fixture schema v3 contains the real external fused outputs, standalone outputs,
the 64 standalone projection partials, and the 64 squared-sum partials.  Four
fixtures cover graph entry and separated layer/branch positions:

| Fixture | Purpose |
|---|---|
| layer 0 attention pre | actual graph-entry standalone path |
| layer 0 FFN | early fused transition |
| layer 21 attention | middle fused transition |
| layer 42 FFN | final-layer fused transition |

The native candidate reproduces the visible external implementation:

- fused `tile_n=2`, eight-split FP32 projection and square reduction;
- projection and square sum consume unrounded FP32 `new_r` while the carried
  residual is stored as BF16;
- the exact warp-XOR and cross-warp reduction order;
- the exact 20-iteration Sinkhorn butterfly;
- the exact 64-thread weighted BF16 RMSNorm association;
- the standalone SM86 TF32 MMA path with 64 splits and the separately observed
  squared-sum fold.

The explicit `lvllmds4x-sm86-v1` limits are:

```text
fused residual/post/combination/layer input: bit exact
standalone post mix max absolute:             3e-8
standalone combination max absolute:          1e-8
standalone layer input max absolute:           0.002
standalone layer input BF16 code distance:     <=1
all compared values:                           finite
```

All four fixtures pass:

| Fixture | fused mismatches | standalone post max abs | standalone combination max abs | standalone layer input |
|---|---:|---:|---:|---:|
| layer 0 attention pre | **0** | 0 | 7.4505806e-9 | 2 / 4,096, max 0.0001220703, one code |
| layer 0 FFN | **0** | 2.9802322e-8 | 2.7939677e-9 | exact |
| layer 21 attention | **0** | 0 | 2.7939677e-9 | exact |
| layer 42 FFN | **0** | 0 | 1.8626451e-9 | 1 / 4,096, max 0.001953125, one code |

The standalone projection partials differ in roughly 1,200 of 1,536 FP32
values, with maxima in the `1.86e-8` to `2.24e-8` range, because CUDA WMMA
exposes an `m16n16k8` accumulator while the captured Triton PTX issues direct
`m16n8k8` MMA instructions.  The standalone squared-sum partials are bit exact.
The accepted final-output limits are intentionally much tighter than the
external test's historical `atol=rtol=1e-2`.

This positive gate retains the candidate as a probe only.  It does not by
itself authorize a runtime or a throughput claim.

## Full-model sufficiency diagnostic

The correctness-only runtime arm replaced initial mHC pre and every fused
post-to-next-pre boundary while leaving attention, KV, router, scoring, top-k,
shared experts, routed scaling, CPU routed execution, output head, and BF16/FP8
storage unchanged.  It required one-token traversal and an admitted SM86
device and had no host-mHC fallback.

Workload:

```text
prompt: The harbour master kept a ledger of every vessel that crossed the bar at dawn.
thinking chat template: enabled
prompt tokens: 20
generated tokens: 16
temperature: 0
external controls: 3 identical repetitions
```

The prompt IDs matched all three external controls exactly, including the final
`128821` thinking token.  External generation was deterministic:

```text
671 3967 10059 678 304 5448 260 16145 2951 377 270 2910 10083 10175 16 455
```

The candidate produced:

```text
671 3967 10059 678 304 28295 411 3947 10175 1055 7891 16 455 10175 344 28
```

Result:

| Gate | Result |
|---|---:|
| prompt token IDs | exact |
| all values finite | pass |
| generated common prefix | 5 / 16 |
| generated IDs exact | fail |
| external top-1 matches | 6 / 16 |
| external repetitions deterministic | 3 / 3 |
| candidate vs current Strata generated IDs | 16 / 16 equal |
| candidate vs current layer hashes | 0 / 1,505 equal |
| candidate vs current raw-logit hashes | 0 / 16 equal |
| candidate/current top-20 overlap | min 15, median 19, max 20 |

The zero equal layer/logit hashes proves the new arithmetic executed throughout
the full model; unchanged generated IDs are not evidence that the runtime arm
silently used the old path.  Instead, the mHC reassociation changes internal
values but does not explain the pre-existing external generation divergence.

The external responses do not expose numeric token-ID-indexed top-20 logits in
a form the local comparator can align, so the relative-logit and top-20 external
subgates remain unavailable.  Exact generated-ID/top-1 equality remains a
binding final complete-pipeline gate, but it is not an attributable rejection
of mHC because the baseline already fails it.  A separate teacher-forcing
expansion was not launched after this diagnostic established that other
component differences remain.

## Resource accounting for the disposable arm

The full-model checker used a host-staged candidate interface so correctness
could be tested before building persistent buffers.  Each of 35 forwards made
87 candidate calls.  Each call uploaded 1,622,204 bytes and downloaded 55,712
bytes, for exactly 4,939,611,180 candidate H2D bytes and 169,643,040 candidate
D2H bytes over the run.  It also allocates/frees every buffer and synchronizes
every call.  These costs are deliberately disqualifying for performance use.

| Resource/phase | Observed value |
|---|---:|
| admission + load | 73.763 s initialization; 72.427 s resident staging |
| prefill | 9.752 s / 20 tokens |
| decode | 7.070 s / 15 timed steps = 471.300 ms/step |
| checker initial standalone mHC-pre counter | 39.186 ms total = 2.612 ms/step |
| checker fused-boundary counter | 3.693 s total = 246.216 ms/step |
| complete MoE counter | 2.042 s total = 136.138 ms/step |
| routed CPU counter | 1.921 s total = 128.034 ms/step |
| attention | 1.043 s total = 69.545 ms/step |
| candidate H2D / D2H | 4,939,611,180 / 169,643,040 bytes |
| core CUDA activation H2D / D2H | 398,659,584 / 323,845,120 bytes |
| RSS | 157,734,313,984 bytes, below 216 GiB ceiling |
| per-device VRAM used | 15,530,524,672 / 23,784,980,480 / 23,784,980,480 bytes |
| target-context compact KV admission | 120,148,480 bytes; not allocated by this scalar correctness arm |
| decode checkpoint reads | 0 bytes |
| process swaps | 0; system swap changed by about 12.85 MB during the arm |
| process filesystem input | 104,918,832 KiB during model load; no decode checkpoint reads |

The 471.300 ms result is **not** compared with 244.812 ms as a performance arm.
It measures the intentionally disposable host-staged checker, not fixed buffers,
stream ordering, persistent KV, a CUDA graph, or the bounded device-resident
hypothesis.  No variance or speedup claim is made from one repetition.

## Decision and next gate

Retain:

- the v3 real-format fixture generator;
- the native SM86 fused/standalone arithmetic candidate in the operation probe;
- the explicit finite/narrow contract checker;
- the thinking-template switch and external full-model oracle tooling;
- this full-model diagnostic showing that mHC alone is insufficient.

Do not retain:

- the full-model reference-mHC runtime path or its CLI switch;
- per-boundary host staging/allocation/synchronization;
- a claim that mHC adoption alone reproduces Lvllmds4-x;
- a throughput matrix after the failed correctness gate.

The mHC component contract is accepted; persistent integration is deferred
until it can be placed in the complete dependent chain rather than another
isolated host-staged path.

The next independent experiment should locate the first non-mHC full-model
divergence before device-pipeline implementation.  The cheapest useful gate is
a teacher-forced real-context attention/compressor/KV fixture at representative
early, middle, and late layers, comparing external and Strata query, compressed
KV, sparse selection, attention output, and residual-boundary bytes.  If that
gate identifies an accepted attention/KV contract, then the complete
device-resident hidden + persistent KV + attention + mHC + stream-ordered MoE
chain becomes eligible.  Isolated projection migration and CPU scheduler work
remain out of scope.

## Reproduction

Generate the real-format fixtures with the external environment:

```bash
CUDA_VISIBLE_DEVICES=0 \
  /home/rodrigo/Developer/Lvllmds4-x/venv/bin/python \
  scripts/probe_dsv4_reference_mhc.py \
  --external-repo /home/rodrigo/Developer/Lvllmds4-x \
  --model models/dsv4f
```

Check a fixture with the native SM86 contract:

```bash
build/strata-dsv4-mhc-contract-probe \
  --contract lvllmds4x-sm86-v1 \
  results/dsv4-reference-mhc-contract/fixtures/layer00_attn_pre.bin
```

The ignored full-model artifacts and comparison are under:

```text
results/dsv4-mhc-contract-acceptance/strata-reference-mhc/
results/dsv4-mhc-contract-acceptance/reference-control/
results/dsv4-mhc-contract-acceptance/reference-mhc-comparison.json
```
