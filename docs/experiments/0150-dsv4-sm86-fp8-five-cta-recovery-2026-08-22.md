# Experiment 0150: SM86 FP8 five-CTA scheduler recovery

## Result

**The exact-graph M=1 performance upper bound is recovered without replacing
the kernel.** Experiment 0149 used six persistent CTAs per SM and measured a
80.77% median. A direct resident-grid sweep shows that six CTAs overfill the
global-barrier scheduler: five CTAs per SM reduce complete scheduler time from
169.984 to **161.792 us** and raise the three-process median to **712.95 GB/s /
84.85%** of the same-process local-read ruler. Four CTAs also pass at 84.32%.

This corrects experiment 0149's overly broad architectural rejection. Its
six-CTA implementation remains rejected, but the persistent/fused family is
alive. The recovered arm still omits experiment 0147's sparse projection
replay, so it is a performance upper bound rather than an F8-2 result.

The query-normalization arithmetic was also closed on the actual layer 2, 21,
and 42 fixtures. The fast warp association produces exactly the same 1,024
BF16 outputs as both the production ascending-FP64 association and the captured
real post-normalization activation at every layer: zero mismatches in all nine
pairwise checks. It therefore needs no FP64 correction on these retained real
boundaries.

## Contract

- **Hypothesis:** the 1.2-point miss is resident-grid overfill and barrier
  contention, not a silicon or decoder ceiling; reducing the grid from six to
  five CTAs per SM recovers at least 2.56 us without changing arithmetic.
- **Primary metric:** complete scheduler useful-byte throughput divided by the
  same-process cold ruler, with the unchanged 82% M=1 threshold.
- **Correctness gate:** zero query-normalization BF16 differences against both
  the ascending-FP64 production association and the captured real activation
  at layers 2, 21, and 42; existing projection no-worse metrics remain binding.
- **Memory ceiling:** 512 MiB. Peak allocation is 522,408,496 bytes
  (498.21 MiB).
- **Rollback:** retain six CTAs if five does not recover 2.56 us, or reject a
  faster normalization association on any real-layer mismatch.

The measured bottleneck is the synchronization/issue residual around the
persistent grid barriers, not additional checkpoint bytes. Reducing the grid
removes 82 CTAs and their barrier atomics while retaining enough warps to cover
the DRAM term. It does not change weight, activation, partial, or output volume.

## Resident-grid sweep

One independent process per cheap screen point:

| CTAs / SM | Total CTAs | Scheduler us | Efficiency | Verdict |
|---:|---:|---:|---:|---|
| 3 | 246 | 174.080 | 78.87% | underfilled |
| 4 | 328 | 162.816 | **84.32%** | pass |
| 5 | 410 | **161.792** | **84.85%** | best |
| 6 | 492 | 169.984 | 80.77% | barrier-overfilled |
| 7 | 574 | not valid | not valid | exceeds simultaneous residency; persistent barrier cannot complete |

The seven-CTA arm was stopped once it demonstrated non-residency. It is an
invalid launch for a software global barrier, not a performance datapoint.

## Three-process retained result

| Process | Ruler GB/s | Scheduler us | Effective GB/s | Efficiency | Query-norm mismatches |
|---:|---:|---:|---:|---:|---:|
| 1 | 840.21 | 161.792 | 712.95 | 84.85% | 0 |
| 2 | 840.21 | 160.768 | 717.50 | 85.40% | 0 |
| 3 | 840.21 | 161.792 | 712.95 | 84.85% | 0 |
| **median** | **840.21** | **161.792** | **712.95** | **84.85%** | **0** |

The 82% budget is 167.425 us, so the retained arm now has **5.633 us** of
measured margin. The multi-arm probe compiles to 79 registers, 44 bytes of
static shared memory, and zero stack or local memory. SHA-256 hashes:

- process 1: `a203a5248e3dc6d9d68364f92e3aaa057f6ab9c2630f3232df82f7550f2354b5`;
- process 2: `f334f1fd7dbfed9a20a815db5c5862f37f0a5439db1c6e80baf117b4b52b361a`;
- process 3: `a203a5248e3dc6d9d68364f92e3aaa057f6ab9c2630f3232df82f7550f2354b5`.

## Normalization falsifiers

Two more invasive mechanisms were rejected before the grid sweep:

- publishing 1,024 FP64 squares and preserving the exact sequential FP64 fold
  takes 206.848 us / 67.22%; and
- eliminating the normalized buffer but recomputing scale/rounding inside
  every `wq_b` fragment load repeats work across 40,960 output rows and takes
  213.088 us / 64.43%.

The cheap grid sweep found the win those rewrites did not. Raw ignored results
are under `results/dsv4-0150-sm86-fp8-fused-qnorm/`.

Both CUDA probes compile cleanly. `make check` passed all registered tests;
the Gemma equivalence target skipped as expected because its external fixture
was unavailable.

## Verdict and next action

The five-CTA performance upper bound passes M=1 and reopens the successor. Do
not run the wider M curve yet: the scheduler still lacks the sparse guarded
projection correction required by experiment 0147. The next bounded mechanism
is to distribute ambiguous-row correction over the persistent ready queue
instead of serializing all replay in the winning warp. It has 5.633 us of
measured M=1 margin and must preserve the same single compact prepack.
