# Experiment 0158 — the FP8 guarded scheduler was being measured on the wrong card

Status: **F8-1 COMPLETE.** The guarded five-CTA scheduler, with the sparse
projection correction composed, clears the 82% gate at the campaign's declared
operating point: **83.27% median, every run above 82%, 1.27 pp margin**, with
23 corrected rows and no added grid barrier. The 0.511 µs deficit that had
blocked it was an operating-point artefact, not a kernel defect.

**Track handoff:** the owner moved the FP8 track to this session. Experiments
0143-0150 and the guarded-correction design are the concurrent session's work
and are preserved unmodified; their in-progress probe changes were checkpointed
first in commit `ef05d3d` with attribution.

## The defect

The FP8 scheduler probe was being run as `CUDA_VISIBLE_DEVICES=1`. On this
machine CUDA device 1 is `nvidia-smi` index 2 — the RTX 3090 that
`apply-3090-tuning.service` still locks to **1605 MHz** and caps to **250 W**.
The campaign's declared operating point, set by owner amendment on 2026-08-22
and recorded in Contract section 8, is the **other** card: CUDA device 0,
unlocked, 350 W.

This matters here for a specific reason established in experiment 0138 and now
confirmed on a second, independent kernel:

> The gate is a **ratio to the local read roofline**. The roofline is
> memory-bound and therefore clock-insensitive. An issue-bound kernel is
> clock-sensitive. Measuring both on a clock-locked card depresses the
> numerator while leaving the denominator alone, so the efficiency ratio is
> understated.

The FP8 scheduler is decoder- and issue-bound by the concurrent session's own
attribution, so it is exactly the kernel shape that this penalises.

## Measurement

Same binary, same arguments, guarded replay active, three interleaved processes
per operating point.

| Operating point | Ruler | Scheduler | Individual efficiencies | Median | Gate 82% |
|---|---:|---:|---|---:|---|
| locked 1605 MHz / 250 W | 161.792 µs | 168.960 µs | 81.80%, 82.27%, 82.30% | 82.27% | **straddles** |
| unlocked 350 W (declared) | 159.744 µs | 164.864 µs | 82.76%, 83.27%, 83.27% | **83.27%** | **PASS** |

An earlier independent run on the locked card measured **81.75%**, also below
the gate.

**The ruler is unchanged across cards** — 161.792 against 159.744 µs, within
one 1.024 µs event tick — confirming memory bandwidth does not care about the
SM clock lock. **The scheduler gains 4.096 µs, 2.42%**, because it does. The
efficiency ratio moves **1.00 pp**.

### Why this is a change in the result's quality, not just its value

On the locked card the median technically clears 82% by 0.27 pp, but
**individual runs fall below the gate** — 81.75% and 81.80% of four observed.
The charter is explicit that a result inside its own run variance is not a win,
so the locked-card result was never a pass; it was a straddle. At the declared
operating point every observed run clears the gate and the margin is 1.27 pp.

This also explains the 0.511 µs deficit the concurrent session was chasing. It
was 0.3% of the step, entirely inside the 2.42% the operating point accounts
for. **No further correction machinery was required to close it.**

## Correctness

Unchanged and not re-litigated. Experiment 0147 closed the real-boundary gate
positively on a **no-worse-than-incumbent** criterion, not a zero-mismatch one,
and reported the candidate strictly better than the incumbent — one mismatch
against five, maximum absolute 5.960e-8 against 1.207e-6.

The guarded configuration measured here reproduces stable mismatch counts
across all three processes: `q_a` 2, `q_norm` 0, `q_b_indexer` 1, `wkv` 1,
`wo_a` 2, `wo_b` 0, with 23 replayed rows and 23,736 replay weight bytes. Peak
device allocation 522,627,712 B, inside the 512 MiB ceiling.

## Gate verdict

| Gate | Required | Result | Verdict |
|---|---|---|---|
| D-F8-GATE at M=1 | >= 82% of local read roofline | **83.27%**, all runs above | **PASS** |
| Guarded correction composed | sparse projection replay included | `guarded_replay: true`, 23 rows | **PASS** |
| No added grid barrier | — | none | PASS |
| Real-boundary correctness | no worse than incumbent | closed by 0147 | PASS |
| Device allocation | < 512 MiB | 498.4 MiB | PASS |

**F8-1 is complete: the exact QPN8-derived primitive, the five-CTA scheduler,
and the sparse projection correction now compose and clear the M=1 gate.**

## What this does not establish

- **F8-2 is not passed.** Its gate requires >= 82% at M in `{1,2,3,4}`, >= 81%
  at M=8 and >= 64% at M=16, on every eligible protected shape. **Only M=1 is
  measured**; the scheduler probe has no M parameter and the M curve does not
  exist yet.
- No production dispatch, admission, route census or graph integration.
- This does not revisit experiments 0143-0150's design decisions; it re-measures
  their result at the correct operating point.

## A rule this makes general

Experiment 0138 found the clock lock distorting an FP4 gate. This is the same
distortion on an unrelated FP8 kernel, so it is a property of the campaign's
gate shape rather than of one kernel: **any gate expressed as a fraction of a
memory-bound roofline will understate an issue-bound candidate on a
clock-locked card.** Every FP8 and FP4 measurement must state its CUDA device
and confirm it is the unlocked one. `CUDA_VISIBLE_DEVICES=1` is the locked card
on this machine; the declared operating point is device 0.

## Exact next action

**F8-2: build the M curve.** The scheduler probe is M=1 only. Extend it to
M in `{2,3,4,8,16}`, then measure each band at the declared operating point with
three interleaved process medians and the guarded correction active, against
>= 82% / >= 81% / >= 64%.
