# Experiment 0061 — external DeepSeek mHC timing and contract gate

Status: **the external production mHC performance gate passes, but its current
Strata numerical-contract gate fails.  Stop before runtime implementation.**
The complete batch-one production sequence -- one standalone pre, 85 fused
post-to-pre transitions with RMSNorm, and one standalone post -- takes a
0.957 ms median as a CUDA graph on either RTX 3090 while reading all 135.266 MB
of layer-distinct projection weights.  It is not the source of the external
stack's roughly 94 ms graph span.

The fast path is not bit-exact with Strata's current oracle.  Three target-shape
fixtures using real layer 0, 21 and 42 checkpoint tensors changed every
post/combination mix and changed 117--397 of 4,096 BF16 fused RMSNorm outputs.
No numerical-contract change was authorized.  Following the predeclared gate,
no native kernel, hidden-state pipeline, teacher-forcing arm, generation arm,
or throughput claim was built from this result.

## Lineage and scope

- original clean experiment base: `e38f4d1 docs: retain the DeepSeek reference
  CPU-MoE arm`
- corrected predecessor: `9db1325 docs: scope the DeepSeek device pipeline
  rejection`
- branch: `exp/dsv4-reference-mhc-contract-timing`
- external source: `/home/rodrigo/Developer/Lvllmds4-x` at
  `691cc0ae2056dc07ed23e4bc4f1dac25b7582f77`
- external package version: vLLM 2.3.8, Torch 2.11.0+cu130, TileLang 0.1.9

The installed mHC wrapper and kernel sources have the same SHA-256 hashes as
the checkout:

```text
tilelang.py:          2b152d112ef69cff869bc50bba0f51d514c598688fc4f719a3e88527efc9d2d0
tilelang_kernels.py:  96495938fcd188d514871136543ed58c246186406028b7472412a6a2e8bee470
```

The external checkout's existing untracked `bench/` directory was not touched.

## Contract and cost model stated before implementation

- **Hypothesis.**  The external production mHC call sequence replaces Strata's
  53.061 ms exposed host mHC/handoff term with a device lane that can
  participate in a measured at-most-100 ms overlapped graph.
- **Target term.**  Exposed `Sigma_serial` from host-owned hidden/mHC
  boundaries, not the already-bounded CPU expert scheduler.
- **Primary metric.**  Complete real-weight mHC CUDA-graph time and its
  contribution to a final at-most-100 ms/token dependency critical path.
- **Correctness gate.**  Real-format operation/layer fixtures must satisfy the
  current Strata oracle before full-model teacher forcing and generation.
- **Memory ceiling.**  Existing 0.95 per-GPU admission, the accounted
  120,148,480-byte target-context compact KV allocation, fixed buffers, zero
  steady-state NVMe, and explicit failure rather than fallback.
- **Bottleneck resource.**  The current 130.145 ms serialized non-MoE path.
  The change should remove host mHC reads and hidden PCIe/synchronization while
  adding GPU SM/HBM service for mHC.  CPU routed work is unchanged.
- **Rollback.**  Stop if the production mHC sequence cannot support a measured
  at-most-100 ms graph or if its reassociation fails the accepted numerical
  contract.

The retained current operating point remains:

```text
complete graph MoE:          114.667 ms
  routed CPU median:         106.869 ms
  routed CPU best:            89.226 ms
  standalone routed probe:    87.879 ms
serialized non-MoE path:     130.145 ms
total:                       244.812 ms/token
target:                      <=100.000 ms/token
```

The candidate model is a dependency graph, not `87.879 + all GPU work`:

```text
tau_candidate = critical_path(CPU DRAM lane, GPU lanes, PCIe lanes)
                + unavoidable serial barriers
```

Only exposed work outside overlap is constrained by `100 - 87.879 = 12.121
ms`.  The total GPU work may be larger while hidden under the CPU lane, but no
critical lane may exceed 100 ms.  This corrects the over-broad additive reading
recorded by the first version of experiment 0060.

## Cheapest production-shape timing gate

The tracked probe loads only the target checkpoint's 43 pairs of mHC
projection, scale, base and branch-norm tensors.  It does not load attention,
MoE, embedding, head, KV or routed-expert tensors.  Fixed CUDA-graph addresses
are used after capture.  The measured sequence is the one implemented by the
external model:

```text
1 x mhc_pre_tilelang(first attention pre + attention RMSNorm)
43 x attention-post -> FFN-pre -> FFN-RMSNorm transition
42 x FFN-post -> next-attention-pre -> attention-RMSNorm transition
1 x mhc_post_tilelang(final reconstruction)
```

That is 86 projection calls, including 85 fused post-to-pre calls.  Each
sequence reads:

| Payload | Bytes |
|---|---:|
| layer-distinct FP32 projection weights | 135,266,304 |
| BF16 branch RMSNorm weights | 704,512 |
| all admitted mHC/norm tensors | 135,980,104 |

The full 135 MB projection stream is larger than GPU cache and avoids the
invalid warm-single-weight access pattern.  Inputs are deterministic BF16
target-shape tensors generated on the CPU and uploaded once; no input or output
transfer occurs inside graph replay.

The fixed setup is about 0.14 seconds of cached tensor loading and 8.9--9.0
seconds of warm-up/cache loading on SM86.  Initial SM120 TileLang compilation
took 51.405 seconds.  The 11-sample measured window was about 0.34 seconds per
3090 arm, so fixed setup was roughly 26 times the measured window.  This was
accepted because it answered the mechanism in under a minute per architecture;
the rejected alternative was a roughly 100-second full checkpoint load plus
prefill and decode.

### Interleaved results

Each repetition alternated eager then graph / graph then eager.  Graph samples
average 20 replays; the table reports per-replay times.

| Runtime GPU | Repetitions | eager median | graph min | graph median | graph max |
|---|---:|---:|---:|---:|---:|
| RTX 3090 A, SM86 | 11 | 12.895 ms | 0.956160 ms | **0.956365 ms** | 0.956518 ms |
| RTX 3090 B, SM86 | 11 | 12.442 ms | 0.956314 ms | **0.956570 ms** | 0.957133 ms |
| RTX 5060 Ti, SM120 | 11 | 13.338 ms | 0.728107 ms | **0.728902 ms** | 0.729846 ms |

The two 3090 graph medians differ by 0.000205 ms (0.021%).  Their complete
ranges overlap.  The eager spread is host launch/topology overhead and is not
used as the device mechanism metric.

The primary 3090 graph moves at least 135,980,104 parameter bytes in 0.956 ms,
an effective 142 GB/s parameter-service rate before counting intermediate
traffic.  It is about 1.02% of the external 93.9 ms graph span and 55 times
smaller than Strata's exposed 53.061 ms host mHC term.  These comparisons make
the mechanism worth a correctness gate; they are not an end-to-end speedup
claim.

The complete-sequence output hashes were identical on SM86 and SM120 once the
input bytes were generated on the CPU:

```text
922bdaec5f8bea482ce340b34f10f48daed587dabb9881f134436e1c8d6b09cf
2196dfd788e918ae66260d8366fac094e8a639b6b80884b19958aa09e33ecd51
cac62f5203ae10e56f429d73a4286284eb2b855c6910cfa2c02b5447c421048b
c476bc3c2c2401b23007de6a320d9a68c37d7ac4f137f1531889bce341c2e12d
```

An earlier apparent cross-architecture hash difference was traced to generating
the random fixture on each GPU.  It is not retained as an arithmetic result.

## Arithmetic established by source

The external implementation does not use Strata's ascending double fold:

- standalone mHC pre selects the TF32 HC pre-norm GEMM path on SM8x and SM12x;
- batch-one fused post-to-pre selects `tile_n=2` and `n_splits=8`;
- `mhc_fused_tilelang` uses FP32 per-thread accumulators followed by warp and
  cross-warp reductions;
- the fused post residual is stored as BF16, while the projection and squared
  sum consume the unrounded FP32 `new_r` value inside the same kernel;
- the next weighted layer input is stored as BF16 before the fused RMSNorm
  scale is applied;
- the external unit test accepts `atol=1e-2, rtol=1e-2` for fused outputs.

Strata's current runtime instead rounds mHC post to BF16, feeds that rounded
state to ascending double projection and squared-sum folds, rounds the reduced
activation to BF16, performs an ascending double RMSNorm fold, and rounds its
output to BF16.  The two implementations therefore have distinct arithmetic
contracts even when their semantic formulas and fusion boundaries agree.

## Real-format operation gate

Because timing passed, three ignored binary fixtures were generated.  They use
the real target checkpoint's layer 0 FFN, layer 21 attention and layer 42 FFN
mHC/norm tensors at the full `[1, 4, 4096]` / `[24, 16384]` shapes.  Inputs are
deterministic finite BF16 tensors.  Each input post/combination mix is produced
by the real preceding external mHC-pre operator.  The external and Strata
implementations then receive identical input bytes.

The tracked C++ checker calls Strata's production `dsv4_mhc_post_f32`,
`dsv4_mhc_pre_f32`, BF16 boundary and `rms_norm_f32` functions.  It does not
reimplement the current oracle in Python.

| Fixture | residual BF16 mismatches | post FP32 mismatches | combination FP32 mismatches | layer-input BF16 mismatches | layer-input max abs |
|---|---:|---:|---:|---:|---:|
| layer 0 FFN | 0 / 16,384 | 4 / 4 | 16 / 16 | **390 / 4,096 (9.52%)** | 0.00390625 |
| layer 21 attention | 0 / 16,384 | 4 / 4 | 16 / 16 | **397 / 4,096 (9.69%)** | 0.0009765625 |
| layer 42 FFN | 1 / 16,384 | 4 / 4 | 16 / 16 | **117 / 4,096 (2.86%)** | 0.0078125 |

The residual post mapping happens to reach the same BF16 value almost
everywhere.  The projection/squared-sum association changes every next mix, and
those changes survive the BF16 fused RMSNorm boundary in all three layers.  The
absolute differences fit under the external test's loose 1e-2 tolerance, but
they fail Strata's current bit-exact operation/hash contract.

This is the negative prerequisite gate.  Per the declared stage dependency, a
full target model load, teacher-forcing run, generation run and three-arm
throughput matrix were not launched to rescue it.

## Resource accounting

| Resource/phase | Measurement |
|---|---|
| admission | 135,980,104 B real mHC/norm tensors; 139,457,024 B peak Torch allocation; 155,189,248 B peak reservation |
| target-context KV | 120,148,480 B already accounted; not allocated by this mHC-only probe |
| H2D | 136,012,872 B setup on deterministic runs; 0 B inside graph replay |
| D2H | 65,616 B after timing for hashes; 0 B inside graph replay |
| load | 0.139--0.146 s with cached model pages |
| warm-up | 8.9--9.0 s SM86; 51.405 s first SM120 compilation |
| decode mHC graph | 0.957 ms median on each reference 3090 |
| attention / complete MoE / routed CPU | not executed; retained values remain 71.140 / 114.667 / 106.869 ms in current Strata |
| synchronization | CUDA events bracket each sample; one fixed-address graph replay, no per-node host synchronization |
| process RSS | about 1.52--1.69 GB for the external Python/JIT environment |
| swaps | 0 B in every arm |
| block/NVMe reads | 0 B in final cached 3090 arms; 633,786,368 B on the first coldish smoke; not a steady graph dependency |
| filesystem writes | 0 B final 3090 timing; 63,930,368 B during first SM120 JIT; 5,001,216 B fixture generation |

Worst-case replication of the 136 MB mHC set plus each rank's compact KV share
remains far inside the explicit 5% physical-VRAM reserve.  The probe makes no
precision change, host fallback or NVMe-dependent decode path.

## Decision

The performance half of the bounded hypothesis is accepted: the visible
external fusion boundary is backed by a genuinely fast implementation, and
the 207.565 ms experiment-0060 probe says nothing about this parallel kernel.

The current-contract half is rejected at the first real-format operation gate.
Therefore:

- retain the measurement and reusable fixture checker;
- retain no DeepSeek runtime path or kernel from this branch;
- do not claim 10 tok/s from the isolated 0.957 ms result;
- do not reopen CPU scheduler work, because no new CPU mismatch was measured;
- do not silently replace the current exact oracle with the external 1e-2
  tolerance or FP32/TF32 reduction tree;
- require an explicit numerical-contract decision before full-model or runtime
  work resumes.

If a new contract is explicitly authorized, the next experiment must define
that contract first, run these operation fixtures under it, then run full-model
teacher forcing and generation before a native C++/CUDA reproduction and the
complete device-resident hidden/KV graph become eligible.

## Reproduction

Timing and fixture generation:

```bash
CUDA_VISIBLE_DEVICES=0 \
  /home/rodrigo/Developer/Lvllmds4-x/venv/bin/python \
  scripts/probe_dsv4_reference_mhc.py \
  --external-repo /home/rodrigo/Developer/Lvllmds4-x \
  --model models/dsv4f \
  --repetitions 11 --replays-per-sample 20 --warmup-replays 3

CUDA_VISIBLE_DEVICES=0 \
  /home/rodrigo/Developer/Lvllmds4-x/venv/bin/python \
  scripts/probe_dsv4_reference_mhc.py \
  --external-repo /home/rodrigo/Developer/Lvllmds4-x \
  --model models/dsv4f \
  --repetitions 3 --replays-per-sample 5 --warmup-replays 1 \
  --fixture-dir results/dsv4-reference-mhc-contract/fixtures
```

Current-oracle comparison:

```bash
build/strata-dsv4-mhc-contract-probe \
  results/dsv4-reference-mhc-contract/fixtures/layer00_ffn.bin
build/strata-dsv4-mhc-contract-probe \
  results/dsv4-reference-mhc-contract/fixtures/layer21_attn.bin
build/strata-dsv4-mhc-contract-probe \
  results/dsv4-reference-mhc-contract/fixtures/layer42_ffn.bin
```

Generated fixture binaries remain ignored under
`results/dsv4-reference-mhc-contract/`.  The tracked artifacts are the external
probe, the C++ current-oracle checker, and this report.
