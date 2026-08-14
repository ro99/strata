# Experiment 0085 — exact reusable DSV4 rank-local layer executor

Date: 2026-08-12
Branch: `exp/dsv4-rank-local-executor`
Disposition: **M2 PASS/DONE for exact one-layer reusable readiness; no
throughput claim; M3 authorized but not started**

The final result commit is reported by the Git handoff and is intentionally
not embedded in this record or the canonical mission document. This avoids a
self-referential evidence hash. The current tracked mission document remains
the authorization source; the ignored roadmap is only a compatibility mirror.

## Hypothesis and scope

One actual layer (layer 21, position 104) can execute exactly from a live
incoming mHC state through attention, routed/shared MoE, and outgoing mHC
state with two persistent rank-local CUDA contexts. Attention and MoE
boundaries remain device-resident except for the declared CPU routed-expert
callback staging. The target term is centralized cross-layer continuation and
`Sigma_serial`; the sign adds two FP32 data and two U32 status collective
pairs while preserving CPU routed work. No throughput improvement is inferred
until a production-shaped 43-layer point instantiates every resource in
`tau = max_r(W_r/B_r) + Sigma_serial`.

The fixed gates were actual d4c/d4r/d4m fixtures, exact FP32/BF16 association,
real NCCL reductions, rank equality, fail-closed status closure, no fallback,
zero checkpoint/KV demand in the measured executor interval, fixed persistent
storage, RSS/VRAM ceilings, and no precision/router/top-k/scale changes.
The scope was one reusable library executor plus independent actual-format
oracles and bounded diagnostics; no Stage 6/graph/full-forward implementation
was included.

## Accepted implementation slices

The capability was built and reviewed in bounded slices:

| slice | accepted capability |
|---|---|
| M2-A `5bd6604` | Correct fused mHC prefix calibration from the initial residual; independent Stage-4 publication oracle. |
| M2-B `8c8b564` | Host-visible versus device-only mHC transition/router exactness. |
| M2-C1 `87b15f9` | Independent stable Stage-5 rank-local publication and target record-44 host-visible oracle. |
| M2-C2a `60b7fe6` | Allocation-free `dsv4_route_sqrtsoftplus_f32_into` with one arithmetic path and parity/tie/failure tests. |
| M2-C2b `0409bdd` | Live device logits+bias routing; replay `.d4m` removed from executor ownership. |
| M2-C3A `0f35333` | Initialize-once, consume-live-state, dynamic layer call, and live next-state lifecycle. |
| M2-C3B/R1/R2 | Prepared-query exactness; independent proof of premature local BF16 association; raw FP32 collective correction and exact live-layer pass. |
| D1–D5 / M2-E | Eight logical failures, partial-enqueue cleanup/reuse, resource accounting, portable artifact transport, and a 78-line dead-plumbing cleanup. These closure edits are included in this final Git handoff. |

The old validator and timing were not silently repaired: CPU mHC exactness was
an invalid oracle, centralized attention and centralized Stage-5 input were
wrong association targets, the original executor was not M3-callable, and
failure/resource evidence was incomplete. Those findings invalidate the old
validator/measurement, not the rank-local mechanism.

## Exact oracle evidence

Canonical fixture hashes:

```text
d4c 1712565387b3565c983da3860c17d6e0648b25792b57da172fd638738adff24
d4r da8f59de692b2c63aa8e83a09aabbf39c0aa0f1c829cf2934d0193f66cff5d00
d4m f754e8ebd54a8ae57d16fb77e626434a46e163534a1e2f4380d28fe7a7ea6bdf
Stage4 artifact f8b5eb041e242094f6a47dda6306a28405bac7c0175464c7cdeee3969a94ff3a
Stage5 artifact 3dca011bb401d748ee0fdd501c65e150781ef6cba721e34fb4766eff1fe07a88
```

The independent Stage-4 publication is hash
`66b3e1c0ef45c065`. The prepared query matched on both ranks with hash
`eb419d5524d8fb63`. R1 independently reproduced the rejected association:
accepted `BF16(FP32(partial0 + partial1))` hash `66b3e1c0ef45c065` versus
premature `BF16(BF16(partial0) + BF16(partial1))` hash
`69162ed5dca96abb`, 1514 mismatches, first index 0 (`bfda`/`bfdb`), maximum
decoded delta `0.015625`. R2 changed only the collective input to raw FP32
partials before BF16 publication.

The independent Stage-5 publication is hash `21b4285cf0c36f8b`. Target
outgoing hashes on both ranks are:

```text
residual  5b42d728a3030bdd
weighted  18034115ca2830db
input     cba74f5aeeb1503e
```

Routes and coefficients were exact on both ranks, with IDs
`80,49,28,75,202,34` and coefficients
`0.394806057,0.38957417,0.229772747,0.176799551,0.17297478,0.136072874`.
The central record-44 diagnostics remained association diagnostics only:
post 3606/max `0.03125`, weighted 1308/max `0.0078125`, input 1270/max
`0.00390625`.

## Failure closure

The eight logical arms ran in binding order: AttentionPreRank0,
AttentionPreRank1, AttentionPostRank0, AttentionPostRank1, MoePreRank0,
MoePreRank1, MoePostRank0, MoePostRank1. Each had matching nonzero expected
phase status on both ranks, zero other-phase status, zero/withheld payloads,
closed data/status collectives, and an exact successful reuse on the same
executor. Matrix log SHA-256:

```text
results/dsv4-rank-local-executor/m2-rank-local-layer/m2-d3-failure-matrix-final/matrix.log
6a577181b4fe511b2012423ac38a6bd736ee775a4c609f3b9ffb00555fed9cc8
```

The exceptional partial-enqueue arm injected `MoeBeforeEnqueueRank1` after
rank 0 enqueue. It drained rank 0, cancelled rank 1 pending metadata, zeroed
payloads, preserved the primary error, and then passed an exact same-executor
reuse run. Log SHA-256 is
`c065c90276e723a320acd98d55b616719b998659ce8785042522e26505dc76a6`;
the command record SHA-256 is
`309910481c7e78e5102ee75b84eec5826268d674e238a520007ee0d4c82044ce`.

## D4D diagnostic resource arm

One warmup and exactly three measured executor repetitions ran after setup.
The timing is an engineering envelope only; transfer service was unavailable,
so `argmax_all_resources=indeterminate`, `tau_full=not_instantiated`, and the
full cost-model gate is incomplete/non-gating.

| repetition | total ms | CPU critical ms | GPU/link measured ms | NCCL envelope ms | unassigned wall residual ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.651396 | 2.988363 | 1.114016 | 0.637952 | 1.663033 |
| 2 | 4.309595 | 2.680717 | 1.055456 | 0.338944 | 1.628878 |
| 3 | 4.336894 | 2.727904 | 1.089888 | 0.309248 | 1.608990 |

Median total is `4.336894 ms`, range `4.309595–4.651396 ms`. Median CPU
bandwidth is `29.405750 GB/s`; the isolated `36.7 GB/s` floor therefore
**FAILs as non-gating navigation evidence**. Full-forward performance is not
evaluated. Total and CPU stability ratios were below `1.20`.

The final D4D log and command hashes are:

```text
log     500625ea17a78ae1577521bfb0d58088ae7805194bbf572743268a32bf5a1e4c
command e1ff8c00929f0cc5832a6af2a291e123371432363abd3fa1df784cf6460e6e6e
```

Per-layer accounting recorded 80,216,064 CPU payload bytes aggregate,
checkpoint calls/bytes `0/0`, workspace allocations `0/0`, weight allocations
`0/0`, direct diagnostic D2H `180,240 B`, logical NCCL bytes `65,552 B`, and
physical NCCL SHM `not_measured`. RSS after the window was
`158,755,717,120 B` against `231,928,233,984 B`; global CUDA used memory was
`699,531,264 B` on each device against `21,287,272,448 B`; resident store
bytes were `156,885,843,968`. Diagnostic D2H is explicitly outside the future
one-completion production claim; positive callback/metadata transfers are
reported, not treated as falsifications here.

Earlier measurement arms remain preserved and classified: D4A preliminary
log `55b247abacd1142bbf78cdb4f35fe0d50c20c685e1005f1abd47cfd664f925d6`, D4B
invalid zero-as-service claim `1b82efd92b82adb8d9e1265d3b03960bfa6409feed8e201b0127306bab605a09`,
and D4C noisy resource arm `7106ccc7ae3865b1b10bf061a4dd2760ffa2b2784de14a3c529898a97e65c4cd`.
The rejected `e0c4fd2` validator and its raw arms are likewise retained and
never relabeled.

## Artifact transport and validation state

The `.d4o` utility uses an exact 24-byte little-endian header, little-endian
u16 payload values, strict shape/overflow/trailing-byte checks, POSIX
`O_CREAT|O_EXCL` publication, and `st_dev`/`st_ino` identity checks before
partial cleanup. Canonical bytes and collision/replacement preservation are
covered by focused C++ tests.

M2-G closure validation reports `285/286` tests passed with one documented
skip, `make check` 2/2 passed, and final `git diff --check` passed. The result
commit hash is reported by the Git handoff rather than embedded here. Ponytail
Review retained correctness/evidence machinery, removed 78 dead app lines in
M2-E, and deferred public test-hook isolation and backend timing-helper
consolidation.

The disposition is **M2 PASS/DONE for exact reusable one-layer readiness only**.
No 43-layer chain, tok/s result, full-forward cost model, or M3 implementation
was started. The next authorized action is a bounded production-shaped M3
43-layer exact chain. Its kill gates remain candidate `>120 ms`, routed CPU
`<36.7 GB/s`, or non-CPU dependency `>30 ms`; if any fails, planning pivots to
a same-operating-point Strata-versus-`94.282 ms` reference phase-gap
reconstruction. The `94.282 ms` value is feasibility context, not an
equal-scope pass.
