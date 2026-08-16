# Experiment 0109 — bounded DSV4 page-attention chunks

Status: **implementation correctness gate passed; model arms pending.**

## Predecessor and defect

Experiment 0108 measured landed commit `54505ba`. Its 677-token arm completed,
but the 2,612-token arm stopped safely after approximately 113 seconds with
`DeepSeek attention-to-mHC workspace exceeds its bounded contract`. The landed
page-attention path submitted every row in the scheduling page as one CUDA
command while fixing that command's workspace ceiling at 384 MiB. Workspace
grows with query rows, so the longer page could not be admitted.

The pre-merge per-row path had completed this operating point. This is therefore
a functional regression in the landed row-batched path, not a throughput
hypothesis.

## Pre-change contract

- Hypothesis: splitting only the query-row dimension at the exact 384 MiB
  workspace boundary restores long-prompt admission without changing any row's
  gather, score, output projection, or mHC slot arithmetic.
- Primary metric: the 2,612-token, page-8192, untraced arm completes and emits
  its model JSON.
- Correctness gate: forced split versus unsplit output is bit-exact; the
  677-token generated IDs remain `2107, 8777, 1277, 440`; decode checkpoint
  reads remain zero; decode is unregressed; `make check` passes.
- Memory ceiling: the existing 384 MiB per-device attention workspace cap is
  binding and unchanged. No second workspace or persistent allocation is
  added.
- Rollback/stop condition: preserve and report if the forced boundary differs
  by one bit, if 2,612 tokens still fails admission, or if the 677-token path
  splits or regresses outside observed variance.

The reduced resource is peak source-device attention workspace per command.
The expected negative signs are additional host submissions, kernel launches,
synchronizations, and repeated reads/materialization of the shared KV pages for
each sub-chunk. Weight precision/residency, router semantics, expert count,
top-k, KV contents, and mHC slots are unchanged.

## Reference shape

The implementation follows `split_indexer_prefill_chunks` in the local
reference stack's `vllm/v1/attention/backends/mla/indexer.py`: host admission is
against a bounded workspace and an oversized request is sub-chunked on the
query dimension. Strata keeps its scheduling page intact and applies the
workspace bound only to contiguous row slices.

## Implementation and fixture

The CUDA backend now exposes the exact multi-row physical-page
attention-to-mHC workspace calculation and uses that same layout object during
execution. Runtime admission binary-searches the largest row extent under
384 MiB. Every command shares the page descriptors and output weights while
queries, candidates, RoPE vectors, mHC slots, and diagnostic output are sliced
by row. A possible singleton tail is rebalanced into the preceding chunk
because the backend's one-row command intentionally has a different mHC state
contract.

The existing CUDA bit-exact page fixture now sets a deliberately small
attention-to-mHC workspace bound, verifies that the full row extent exceeds
it, obtains the admitted row count from the production planner, and compares
all split output bits against unsplit per-row truth. The focused fixture passed.
`make check` passed with 277 tests passed and 34 fixture-dependent skips.

## Arm budget

After the implementation checkpoint, run exactly two untraced page-8192 arms
from the same binary and devices 1,2: one 677-token arm (estimated four minutes)
and one 2,612-token arm. The landed implementation has no successful long-arm
timing; estimate 8–15 minutes from the reduced dispatch path, with a total
planned model window of approximately 12–19 minutes. Preserve actual wall
times. No repetitions, baseline variant, or additional prompt length are in
scope.

## Results

Pending.
