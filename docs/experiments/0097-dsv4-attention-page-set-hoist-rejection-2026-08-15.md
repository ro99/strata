# Experiment 0097 — attention page-set hoist rejection

Status: **rejected.** Splitting physical prefill attention into append and
attend loops and sharing its page-set container across a layer/page does not
reduce the measured scoring term. The first short pair falsified transfer to
production: scoring was 18.448 s before and 18.806 s after. A counter-only
diagnosis then showed why.

**One-sentence cause:** the isolated screen's apparent win did not transfer
because it removed repeated construction of a roughly 72 ms page-set subterm
while leaving all 4,605,666 row-local block-table searches and page-map lookups,
and all 29,111 attention calls, unchanged inside an 18.5 s scoring phase.

The candidate runtime is not promotable and must be rolled back before any
result commit. No replacement mechanism is proposed here.

## Pre-change contract

Hypothesis: a physical prefill page could append all rows, then attend all rows
against one shared page-set/map, reducing the 18.2 s scoring term measured at
677 tokens and the 73.4 s term at 2,612 tokens.

Primary metric: attention scoring seconds at 677 and 2,612 tokens, followed by
the fixed/marginal prefill fit only if the scoring term improved materially.

Correctness gate: preserve routes, precision, expert residency and page-major
results; keep decode checkpoint reads at zero and decode unregressed. Memory
ceiling: the accepted 216 GiB host / 21 GiB-per-device admission contract.
Rollback: stop if scoring did not improve outside observed variance or any
correctness/memory gate failed.

At the handover operating point, attention was the 677-token bottleneck:
33.8 s total, comprising about 18.2 s scoring, 10.3 s query work and 4.1 s KV
work. The mechanism targeted host metadata work inside scoring. It did not
change expert H2D, GPU attention work, router work or numerical semantics.

## Cheapest screen, and the mistake in interpreting it

`strata-dsv4-candidate-resolution-probe` reproduced the table lookup, map and
page-container operations without loading the model:

| tokens | baseline | shared set | ratio | set lifetimes |
| ---: | ---: | ---: | ---: | ---: |
| 677 | 303.049 ms | 278.139 ms | 1.090x | 29,111 -> 473 |
| 2,612 | 2,570.117 ms | 2,454.268 ms | 1.047x | 112,316 -> 1,763 |

The build-count reduction looked large, but the absolute opportunity was only
24.9 ms and 115.8 ms in the synthetic mechanism. The probe never established
that this metadata accounted for the production scoring phase. Treating its
within-mechanism ratio as if it applied to 18.2 s was the experimental defect.

## Candidate shape

The candidate split `attention_prepared` into project/append and attend. A
physical page appended every row first, then attended every row while retaining
one `PhysicalAttentionPageSet`. A page-scoped sliding-retention floor was also
required: appending a complete page before attending its first row otherwise
evicted that row's earliest causal key.

Reading the integrated path after the negative first pair exposed what had not
moved:

- `attention_attend_prepared` still calls `physical_paged_attention` once per
  row.
- Every call still snapshots both block tables and allocates its candidate
  vector.
- Every compressed and sliding candidate still calls
  `locate_physical_kv_block` and then `page_indices.find`.
- The shared object only preserves already-acquired leases, page descriptors
  and map entries after those row-local lookups.

Thus the integration shared a container but did not hoist candidate resolution.

## Production falsification and stop

The planned interleaved matrix should have stopped after the first short pair.
It was stopped with seven completed arms and the second long baseline killed.
No median or throughput win is claimed from this incomplete matrix.

| length / repetition | baseline scoring | candidate scoring | result |
| --- | ---: | ---: | --- |
| 677 / 1 | 18.448 s | 18.806 s | candidate +1.9% |
| 677 / 2 | 18.315 s | 18.711 s | candidate +2.2% |
| 2,612 / 1 | 71.401 s | 73.483 s | candidate +2.9% |
| 2,612 / 2 | killed | 73.059 s | no pair |

The MoE term varied independently: for example, the two completed 677-token
baseline arms took 89.53 s and 198.11 s in MoE. That variance explains total
prefill spread, but cannot explain the scoring regression. Continuing the
matrix after the first scoring pair was contrary to the predeclared rollback
gate.

Raw partial data: `results/dsv4-step1-attention-scoring-ab/` (ignored).

## Counter diagnosis

The follow-up added phase counters to the actual `f1ef47e` source and to the
candidate. It measured one 677-token arm of each, no repetition and no long
prompt. Planned budget was 4--7 minutes per arm and under 14 minutes total;
actual setup plus prefill was 5.63 minutes baseline and 4.67 minutes candidate,
10.30 minutes total.

`page-set build` below is container setup plus map-miss lease/page insertion.
`candidate resolution` is the block-table search and map lookup, excluding the
measured map-miss interval.

| counter | baseline | candidate | delta |
| --- | ---: | ---: | ---: |
| page-set lifetimes | 29,111 | 516 | -98.2% |
| page entries inserted | 84,007 | 1,612 | -98.1% |
| page-set build time | 72.236 ms | 28.942 ms | -43.294 ms |
| candidate resolutions | 4,605,666 | 4,605,666 | unchanged |
| candidate-resolution time | 209.621 ms | 224.742 ms | +15.121 ms |
| scoring phase | 18.502 s | 18.971 s | +0.469 s |
| device paged-attention time | 6.457 s | 6.519 s | +0.062 s |
| paged-attention calls | 29,111 | 29,111 | unchanged |
| paged-attention kernel launches | 553,109 | 553,109 | unchanged |

The metadata measured by the probe is only 0.282 s of the 18.502 s baseline
score phase. The candidate reduced it to 0.254 s, a 28 ms net reduction. That
matches the probe's 25 ms absolute prediction and proves that the probe itself
was not lying; its result was applied to the wrong parent term.

Device calls, kernel launches, D2H bytes and candidate count were unchanged.
The candidate's total scoring increase is not attributed to NUMA or promoted as
a separate finding: the kill criterion had already fired, and explaining a
sub-percent negative residual is unnecessary to reject a mechanism whose
maximum measured benefit is tens of milliseconds.

Raw counter data:
`results/dsv4-step1-attention-resolution-diagnosis/` (ignored).

## Correctness work completed before rejection

The split passed the local cache/runtime gates used during development:

- the new page-scoped sliding-retention fixture passed;
- the CTest CPU suites passed 2/2;
- current page-4 output, operation/layer diagnostics and routes matched the
  pre-change page-4 binary exactly;
- current page-1 matched the pre-split page-1 output exactly;
- untraced page-64 physical prefill completed with zero decode checkpoint
  reads.

The known page-1 versus GPU-expert page mismatch still begins at layer 0
`ffn_output`, the documented routed-expert reassociation from experiment 0096,
not in attention. These gates establish that the candidate was numerically
coherent; they do not rescue its negative performance result.

## Scope of the work that was not built

Actually removing candidate resolution would require page-level resolved
metadata, not merely a shared lease map: stable page numbering for the retained
block tables plus a direct logical-row-to-physical-page/row mapping from which
each row's causal sliding interval and selected compressed positions can be
materialized without `locate_physical_kv_block` or a hash lookup. At longer
contexts the compressed selections are row-specific Lightning Indexer outputs,
so a single candidate list cannot be shared blindly.

That changes data ownership and the per-row CUDA request construction and is a
materially larger hypothesis. It requires a new cost-model screen, correctness
gate and rollback statement before implementation; experiment 0097 provides no
authorization or positive evidence for it.

## Reproduce the diagnosis

The counter pair was run in the named tmux session
`strata-dsv4-step1-resolution-diagnosis` with:

```bash
BASELINE_RUNNER=/tmp/strata-step1-diagnosis.kNVYaC/baseline/build-diagnosis/strata-deepseek-run \
CANDIDATE_RUNNER=build-step1/strata-deepseek-run \
RESULT_DIR=results/dsv4-step1-attention-resolution-diagnosis \
scripts/run_dsv4_attention_resolution_diagnosis.sh
```

The baseline was a detached `f1ef47e` worktree carrying only the same diagnostic
counters. No further A/B arms were launched after the stop instruction.
