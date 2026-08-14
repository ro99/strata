# DeepSeek V4 rank-local TP2 attention, actual replay — 2026-08-10

Status: Stage 4 is closed and accepted as a program milestone by user review.
Exactness, failure closure, the 23 ms absolute ceiling, projected memory, and
directional improvement all pass. The narrower predeclared claim of a
reproducible improvement greater than 10% did not pass and remains recorded as
unproven. Stage 5 is the next stage; no Stage-5 implementation was started in
this experiment.

This is an explicit plan amendment, not a relabeling of either raw run. The
probe's original materiality predicate and its pass/fail output remain intact.
Review accepted Stage 4 because its role in the larger plan is to establish an
exact, feasible, non-regressing TP2 attention boundary below 23 ms, while the
large topology saving is assigned to Stages 5 and 6.

## Why this is a new result

Experiment 0075 remains a valid rejection of its synthetic-position probe. It
reported 58.721 ms for 43 attention layers. That arm did not reproduce the
production decode access pattern or accepted physical-page arithmetic, so it
could not decide whether rank-local attention was intrinsically too slow.

This follow-up captures the accepted runtime at all 43 layers and decode
positions 104 through 118, producing 645 ignored replay files. The binding
validator uses layers 2, 21, and 42 at all 15 positions: 45 actual fixtures,
three interleaved repetitions per fixture, and one warmup. The capture contains
the real hidden/query/KV boundary, physical page payloads and descriptors,
candidate validity, sinks, compressed-family ratio, and accepted centralized
branch output. The sparse selection is replayed from the production page and
candidate sequence; this probe does not recompute Lightning Index selection.

## Hypothesis, metric, and gates

The measured system bottleneck remains the 62.986 ms dependent gap between
MoE callback bodies in the 152.263 ms Strata decode. The Stage-4 hypothesis is
that concurrent 32-head rank-local attention, including an exact FP32 TP
reduction and BF16 publication, materially lowers the attention portion of
that serial dependency.

The primary metric is the equal-scope device chain from input projection
through accepted physical-page attention, `wo_a`, rank-local `wo_b`, two NCCL
collectives (FP32 data sum and four-byte status maximum), and BF16 publication.
The predeclared performance gate requires both:

- projected 43-layer candidate dependency at or below 23 ms; and
- candidate strictly below 90% of the equal-scope centralized control.

The correctness gate is zero mismatch at the captured BF16/FP32 contracts,
including the query, compressor and KV values that are present in the replay,
accepted backend page body, attended/inverse-RoPE value, `wo_a` BF16 and FP8
intermediate, an independent 256-lane `wo_b` association oracle, rank-ordered
FP32 partial sum, and BF16 publication. Either rank's injected pre- or
post-data-collective failure must close both ranks. No timed allocation,
application H2D/D2H, host callback, checkpoint read, P2P fallback, precision
change, or non-finite value is permitted.

The per-GPU ceiling is 21,287,272,448 bytes. The Stage-3 full-rank projection
is 6,223,273,412 bytes before the BF16 execution cache evaluated here.

## Correctness defects found before measurement

The user's stopped run reported 4,749/16,384 BF16 page mismatches with maximum
absolute error 0.01953125. The runtime was not wrong: the probe issued eight
independent four-head GEMMs, while the accepted backend issues one 32-head
GEMM per rank. Four candidate reduction groups had been confused with query
head grouping. Matching the accepted 32-head association reduces the page
mismatch to zero.

The first parallel RMSNorm candidate passed position 104 but changed one BF16
query at position 116. It was rejected. The retained kernel parallelizes loads
and squares but performs double accumulation in original column order with
explicit round-to-nearest operations. It is exact over the complete window.

The strict `wo_b` oracle initially appeared allocation-dependent. That was a
validator race: it launched on a nonblocking CUDA stream and immediately used
a synchronous host copy without synchronizing that stream. Diagnostic copies
only delayed the copy enough to hide the race. The final validator explicitly
synchronizes the untimed oracle and leaves the measured path copy-free.

Other rejected candidates remain rejected: a full cuBLAS Q path changed one
BF16 value, a 64-thread exact matvec and multi-row CTA were slower, and the
tree-reduction norm changed association at the actual operating point.

## Retained mechanism

The bounded probe retains:

- explicit Stage-3 rank shards, never a full tensor with ignored halves;
- two independent device contexts, main/auxiliary streams, page tables,
  workspaces, status, and outputs;
- 32 local heads and accepted 32-head physical-page GEMM grouping;
- exact 128-physical-thread emulation of the required 256 logical matvec lanes;
- an exact BF16 expansion cache for FP8 attention projections, while retaining
  the canonical FP8 weights and never reducing precision below four bits;
- one grouped NCCL issue containing the FP32 data and status collectives; and
- an untimed independent 256-lane `wo_b` association check.

The BF16 cache changes storage, not model precision or arithmetic input. Its
full-model incremental projection conservatively charges every attention layer
the largest measured expansion.

## Authoritative commands and evidence

Capture:

```bash
STRATA_DSV4_MODEL_DIR=/home/rodrigo/Developer/strata/models/dsv4f \
STRATA_DSV4_STAGE4R_CAPTURE_RESULT_DIR=results/dsv4-rank-local-attention-parity/stage4r-capture-v4 \
scripts/run_dsv4_stage4r_capture.sh
```

Each complete validation used the reusable script in a named tmux session:

```bash
STRATA_DSV4_MODEL_DIR=/home/rodrigo/Developer/strata/models/dsv4f \
STRATA_DSV4_STAGE4R_REPLAY_DIR=results/dsv4-rank-local-attention-parity/stage4r-capture-v4/replays \
STRATA_DSV4_STAGE4R_RESULT_DIR=results/dsv4-rank-local-attention-parity/stage4r-final-v2 \
STRATA_DSV4_WARMUPS=1 STRATA_DSV4_REPETITIONS=3 \
scripts/run_dsv4_stage4r_validate.sh
```

The two final result directories are:

```text
results/dsv4-rank-local-attention-parity/stage4r-final-v1
results/dsv4-rank-local-attention-parity/stage4r-final-v2
```

Both used only the two RTX 3090s as runtime devices 0 and 1 under
`CUDA_VISIBLE_DEVICES=1,2`. P2P capability and use are zero. NCCL 2.28.9
selected `SHM/direct/direct`; physical NCCL transport bytes are uninstrumented
and reported as `not_measured`, never as zero.

## Branch and worktree provenance

The Stage-4 work was initially performed in:

```text
/home/rodrigo/Developer/strata-dsv4-rank-local-attention-parity
branch: exp/dsv4-rank-local-attention-parity
base commit: 1392a56b9cac2aae7e181dd06eff279cac0bd434
```

That path was a Git worktree of the same repository, not a copied or independent
repository. It shared Git object storage and history with the primary checkout
but had its own branch, index, and working files.

The primary checkout was already on
`exp/dsv4-device-resident-kv-handoff` and contained unrelated uncommitted
changes in `.gitignore`, the DeepSeek runner/runtime, CUDA backend, and earlier
handoff scripts. The original execution instruction explicitly required a
clean separate worktree from validated `main` so those user-owned changes
would not be stashed, discarded, overwritten, or inherited implicitly.

The temporary split was removed after operator review. The pre-existing primary
checkout changes were first preserved on
`exp/dsv4-device-resident-kv-handoff` at commit `97234ed`. Stage 4 was preserved
on `exp/dsv4-rank-local-attention-parity` at commit `88e1ff4`. The redundant
worktree was then removed and the primary checkout at
`/home/rodrigo/Developer/strata` was switched to the Stage-4 branch. No merge or
cherry-pick was needed, and neither branch's work was mixed with the other.

The ignored Stage-4 evidence was moved into the primary checkout before the
worktree was removed. The destination contains 365 MiB of results, including
645 version-1 and 645 version-2 replay fixtures. The final-run logs retained
their pre-move SHA-256 hashes:

```text
stage4r-final-v1/raw.log  4398f635a4f88c44b116ada91d2ab60b58a2e3edd2a86e663167a91fa5f845ed
stage4r-final-v2/raw.log  8539dbe79278ce38d81dff987bf0aba3acd6c848b569dda6d82c544f7f0046c2
```

The same consolidation removed the older clean
`/home/rodrigo/Developer/strata-dsv4-rank-local-tp2` worktree. Its branch
`feat/dsv4-rank-local-tp2` remains intact at commit `9d31ba5`, while its 51
ignored evidence files were moved to `results/dsv4-rank-local-tp2`. Their
aggregate content-manifest hash remained
`79fb316a2b5ea3ba96fe9a83387c127ef2cce908995148ad607a4cdb5caeed47`.
Only its generated build trees were discarded.

## Correctness and memory result

Each final run has:

- 45/45 fixture records with exactness, failure closure, and local memory pass;
- 270/270 exact rank-query checks with zero BF16 mismatch;
- 270/270 accepted-backend physical-page checks with zero BF16 mismatch;
- 135 interleaved control/candidate pairs; and
- 180 injected failure arms (pre and post on each of two ranks), all closed.

The final-source run projects:

```text
Stage-3 rank VRAM:                      6,223,273,412 B
maximum BF16 expansion per layer/rank:    96,468,992 B
43-layer incremental expansion:        4,148,166,656 B
projected rank VRAM:                   10,371,440,068 B
ceiling:                               21,287,272,448 B
```

This is a conservative capacity projection; CUDA driver allocations remain
opaque and are not silently relabeled as zero.

## Timing and decision

The final-source run's pooled candidate phase medians are:

| phase | median | observed range |
|---|---:|---:|
| projection | 0.253952 ms | 0.176128–0.464704 ms |
| accepted page attention | 0.064512 ms | 0.035840–0.209920 ms |
| output projection | 0.096256 ms | 0.094336–0.140288 ms |
| FP32 data + status NCCL interval | 0.092320 ms | 0.029568–0.332800 ms |
| BF16 publication | 0.004096 ms | 0.004064–0.018560 ms |
| auxiliary work, overlapped | 0.156448 ms | 0.071488–0.244736 ms |

Projection is the candidate's largest measured phase. NCCL is a serial
handoff, not a byte-volume limit; Stage 2's lower per-reduction result queued
many reductions ahead and is not substituted for this eager dependent-layer
measurement.

The two complete runs disagree at the materiality boundary:

| run | projected control | projected candidate | improvement | <=23 ms | <90% control |
|---|---:|---:|---:|---:|---:|
| final-v1 | 22.560256 ms | 20.106720 ms | 10.875% | pass | pass |
| final-v2, final source | 22.388704 ms | 20.874752 ms | 6.762% | pass | fail |

The paired fixture medians tell the same story. In final-v1, 43/45 fixtures
favor the candidate and the median candidate/control ratio is 0.887. In
final-v2, 39/45 favor it but the median ratio is 0.908. A third run cannot
erase the already-observed pass/fail variance. Calling the first run a win
would violate the rule against accepting a result within observed variance.

The raw final-source probe therefore reports
`not_materially_faster_than_control`; that narrow greater-than-10% performance
claim is not accepted. The stronger facts needed by the program milestone are
reproducible: both runs are exact, below 23 ms, within memory, fail closed, and
faster than their equal-scope centralized controls.

User review closes Stage 4 as accepted under the amended program gate:

```text
required for Stage-4 closure
  exact rank-local ownership and arithmetic       pass
  real no-P2P transport and failure closure       pass
  projected attention dependency <= 23 ms         pass
  projected per-rank memory <= ceiling            pass
  candidate faster than equal-scope control       pass in both full runs

recorded but not required by the amended gate
  reproducible candidate < 90% of control         unproven
```

No result is deleted or renamed. The original materiality experiment remains
a negative result inside an accepted engineering stage.

## Relation to 10 tok/s and next stage

Stage 4 was never expected to remove the complete gap to 10 tok/s. Its
final-source run saves 1.514 ms against the corrected equal-scope centralized
attention control. The old 58.721 ms synthetic number is not an equal-scope
baseline and is not used to claim a 58.7-to-20.9 ms win. Likewise, the
22.389 ms control is Strata's corrected centralized replay control, not an
external-vLLM attention measurement.

Added only as directional arithmetic to the 152.263 ms baseline, the measured
Stage-4 saving gives about 150.749 ms/forward or 6.63 tok/s. The remaining
50.749 ms is the budget for the rest of the roadmap, principally the roughly
42 ms topology/dependency opportunity in Stages 5 and 6 and the separately
measured 11--12 ms CPU-kernel parity opportunity after the eager topology is
validated. Stages 7 and 8 close correctness and end-to-end eager performance;
Stage 9 reconsiders graph capture only after that eager path exists.

The next authorized work is Stage 5: actual-replay rank-local shared and
routed MoE. It must measure the current per-layer CPU gate/up, down, weighted
reduction, GPU shared expert, overlap, upload/join, real TP reduction, rank
imbalance, and dependent critical path before choosing or integrating a
mechanism. It must preserve actual route/top-k semantics, precision, separate
rank-local destinations/status, and the accepted Stage-4 boundary. Stage 5
must not optimize CPU kernels or begin graph capture.

## Verification and handoff state

- The NCCL-enabled Stage-4 target and replay-dump utility build successfully.
- The replay reader remains compatible with the 645 captured version-1 files;
  new captures use an explicit version-2 candidate encoding.
- `make check` passes both CTest targets with zero failures.
- `git diff --check` passes.
- Raw captures, profiles, and validation logs remain ignored and outside Git.
- Stage-4 source and documentation are committed on
  `exp/dsv4-rank-local-attention-parity`; the earlier primary-checkout work is
  preserved separately on `exp/dsv4-device-resident-kv-handoff`.
- The primary checkout now hosts the Stage-4 branch and ignored evidence. The
  redundant Stage-4 worktree and its rebuildable generated build trees were
  removed; no raw experiment evidence was deleted.
