# DeepSeek V4 rank-local TP2 attention falsifier — 2026-08-10

Program disposition update: this synthetic-position arm remains rejected on
its own evidence, but it has been superseded for Stage-4 closure by the actual
decode replay and explicit review decision in experiment 0076. Do not use the
58.721 ms result below as the production attention cost.

Status: Stage 4 is rejected by its necessary eager attention-chain gate.
Stages 5–10 remain stopped. Stage 2 and Stage 3 remain accepted; this arm did
not modify the production DeepSeek runtime or begin MoE integration.

## Hypothesis and binding gate

The measured bottleneck is the 62.986 ms inter-callback dependent-gap term.
The Stage 4 hypothesis was that two concurrent rank-local attention chains,
with 32 query heads per rank and a real FP32 TP reduction after `wo_b`, would
remove the centralized attention dependency. The primary metric is the
equal-scope device critical path from projection through page attention,
output projection, FP32 NCCL reduction, and BF16 publication.

The necessary gate is a material improvement over the centralized control and
a projected 43-layer attention dependency no greater than 23 ms. The
correctness gate is the actual-format early/middle/late fixture oracle,
including page bytes/descriptors, sparse selection, query-head ownership,
compressor/indexer outputs, each rank's `wo_b` partial, rank-ordered FP32
reduction, BF16 publication, failure closure, and finite values. The per-GPU
VRAM ceiling is 21,287,272,448 B. Rollback/rejection conditions are any
oracle mismatch, NCCL fallback or host application transport, failed closure,
allocation in the timed chain, non-finite result, memory ceiling breach, or
latency-gate failure.

## Scope and implementation

The bounded probe is `apps/strata_dsv4_rank_local_attention_probe.cu`, built
as `strata-dsv4-rank-local-attention-probe`. It uses only the explicit
`Dsv4RankShardDescriptor` loader. Replicated Q/KV projections, compressors,
and indexer weights remain replicated; `wq_b` and `wo_a` are row-sharded;
`wo_b` is strided-column-sharded. The candidate launches two independent
main/auxiliary stream chains with separate pages, descriptors, workspaces,
events, status, and output buffers. It uses 32 local query heads and a
`ncclFloat32` sum followed by on-device BF16 publication. No application
H2D/D2H copy, pinned-host transport, P2P access, host callback, timed
allocation, or fallback is present in the measured chain.

The centralized control uses all 64 heads and full attention tensors on one
3090. The candidate uses the rank-local shards on both 3090s. The probe
validates actual checkpoint layers 2 (ratio 4), 21 (ratio 128), and 42
(ratio 4), with 1 warmup and 3 interleaved control/candidate repetitions per
layer. It injects a pre- and post-collective failure on each rank; all four
cases per layer pass fail-closed validation.

## Exact commands and environment

The authoritative run was the named `strata-stage4-attention-final-v3` tmux
session. The reusable script was launched with:

```bash
STRATA_NCCL_INCLUDE_DIR=/home/rodrigo/Developer/Lvllmds4-x/venv/lib/python3.12/site-packages/nvidia/nccl/include \
STRATA_NCCL_LIBRARY=/home/rodrigo/Developer/Lvllmds4-x/venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2 \
STRATA_DSV4_MODEL_DIR=/home/rodrigo/Developer/strata/models/dsv4f \
STRATA_DSV4_DEVICE0=0 STRATA_DSV4_DEVICE1=1 \
STRATA_DSV4_WARMUPS=1 STRATA_DSV4_REPETITIONS=3 \
STRATA_DSV4_STAGE4_RESULT_DIR=/home/rodrigo/Developer/strata-dsv4-rank-local-tp2/results/dsv4-rank-local-tp2/stage4-attention-final-v3 \
scripts/run_dsv4_rank_local_attention_stage4.sh
```

The script configures and builds with the same include/library values, then
sets `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,GRAPH,SHM`. Runtime devices
0 and 1 are both `NVIDIA GeForce RTX 3090`; peer capability is zero in both
directions and `used=0`. `CUDA_VISIBLE_DEVICES` was unset because this host's
runtime enumeration already maps devices 0 and 1 to the two requested 3090s.

NCCL provenance from the log:

```text
ncclGetVersion=22.8.9 encoded=22809 result=success
nccl_loaded_library=/home/rodrigo/Developer/Lvllmds4-x/venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
nccl_configured_library=/home/rodrigo/Developer/Lvllmds4-x/venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
NCCL version 2.28.9+cuda13.0
Channel ... via SHM/direct/direct
```

The raw log and environment record are retained outside Git at:

```text
/home/rodrigo/Developer/strata-dsv4-rank-local-tp2/results/dsv4-rank-local-tp2/stage4-attention-final-v3/raw.log
/home/rodrigo/Developer/strata-dsv4-rank-local-tp2/results/dsv4-rank-local-tp2/stage4-attention-final-v3/environment.txt
```

NCCL's internal SHM/direct/direct movement is not application D2H/H2D and its
physical transport bytes were not measured; they are therefore reported as
`not_measured`, not zero.

## Correctness and memory results

| layer | family | control runs | candidate runs | exactness | failure closed | rank-0 / rank-1 accounted VRAM | ceiling |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | ratio 4 | 3 | 3 | pass | pass | 104,716,344 / 104,716,344 B | pass |
| 21 | ratio 128 | 3 | 3 | pass | pass | 83,184,824 / 83,184,824 B | pass |
| 42 | ratio 4 | 3 | 3 | pass | pass | 104,716,344 / 104,716,344 B | pass |

The exact validator compares each rank's 32-head query slice, compressed and
indexer values, physical page bytes and descriptors, sparse positions, sinks,
RoPE and attention outputs, local `wo_a` output, each rank's `wo_b` partial
against a host calculation using that rank's actual shard, and both ranks'
reduced FP32 result against the rank-0-then-rank-1 oracle. Final outputs are
validated after BF16 publication. All results are finite. The injected status
is all-reduced so either rank's pre- or post-collective failure produces
sanitized output and nonzero status on both ranks.

## Timing results

All times below are CUDA event measurements unless identified as `wall` or
`synchronization`. `device_chain` is the maximum rank-local start-to-finish
event interval and is the critical-path metric used by the projection.

| layer | control device chain median (ms) | TP2 device chain median (ms) | FP32 NCCL median (ms) | BF16 publication median (ms) | rank imbalance median (ms) |
|---:|---:|---:|---:|---:|
| 2 | 1.657632 | 1.513472 | 0.216064 | 0.004096 | 0.189664 |
| 21 | 1.544000 | 1.320960 | 0.120832 | 0.004096 | 0.103296 |
| 42 | 1.616896 | 1.386496 | 0.131072 | 0.004096 | 0.096256 |

Across the three representative layers and three repetitions:

```text
control_device_chain:       median 1.616896 ms, range 1.543168–1.678336
candidate_device_chain:     median 1.386496 ms, range 1.287168–1.607680
candidate_projection:       median 0.612512 ms, range 0.566272–0.708608
candidate_attention_page:   median 0.399360 ms, range 0.362336–0.401408
candidate_output_projection:median 0.248832 ms, range 0.245760–0.251904
candidate_auxiliary:         median 0.372736 ms, range 0.228352–0.420864
candidate_nccl_fp32:         median 0.137216 ms, range 0.104448–0.246784
candidate_bf16_publication:  median 0.004096 ms, range 0.003072–0.004096
control_wall:                median 1.624867 ms, range 1.551749–1.701282
candidate_wall:              median 1.397189 ms, range 1.296181–1.656826
candidate_synchronization:  median 0.017002 ms, range 0.009013–0.049146
candidate_rank_imbalance:    median 0.103296 ms, range 0.081920–0.263168
```

The rank-local per-rank stage medians and ranges are in the raw log. The
auxiliary and main event intervals are stream-activity proxies; no hardware
occupancy/CUPTI utilization counter was collected by this standalone probe.

## Transport and accounting

The timed path reports:

```text
timed_path_allocations=0 host_callbacks=0 checkpoint_reads=0
application_timed_h2d_copies=0 application_timed_d2h_copies=0
application_timed_h2d_bytes=0 application_timed_d2h_bytes=0
```

The control records 9 events, uses 2 stream waits, and has one final event
synchronization. The TP2 candidate records 20 events (including the separate
NCCL-completion and BF16-publication events), uses 4 stream waits, and has two
final event synchronizations. The candidate issues two grouped NCCL
collective operations per forward: FP32 hidden data and four-byte status; both
ranks participate in both.

Post-boundary exactness reads are deliberately outside the timed chain. Their
application D2H bytes per successful run are:

| layer | control verification D2H | TP2 verification D2H | each rank's `wo_b` partial D2H |
|---:|---:|---:|---:|
| 2 / 42 | 1,058,824 B | 1,708,048 B aggregate | 16,384 B |
| 21 | 1,021,960 B | 1,634,320 B aggregate | 16,384 B |

These verification copies are not production transport and are excluded from
the latency gate. NCCL internal physical bytes remain unmeasured.

## Cost-model projection and decision

Using the actual contract's 43 attention layers (21 ratio-4, 20 ratio-128,
and two unmeasured layers conservatively charged the slower measured family),
the representative device-chain medians project to:

```text
representative centralized control:  68.666609 ms / 43 layers
representative rank-local candidate:  58.720769 ms / 43 layers
candidate improvement:               14.484245% (median flag: material)
projected candidate attention chain: 58.720769 ms / forward
23 ms necessary gate:                FAIL
```

The exact probe output is authoritative: `material_improvement=1`,
`layer_chain_gate_le_23ms=0`, and
`stage4_eager_gate=rejected reason=projected_attention_dependency_above_23ms`.
The measured candidate is lower than control, but the projected attention
dependency is about 35.7 ms above the allowed non-callback ceiling. Stage 4 is
therefore rejected; this arm does not justify progressing to MoE/full-chain
implementation or attempting graph capture.

## Stage disposition

Accepted within this falsifier: actual-format rank-local attention fixtures,
32-head ownership, page/descriptors, rank-sharded weight use, NCCL
SHM/direct/direct selection, FP32 reduction/publication exactness, failure
closure, zero timed allocations/host transport, and memory.

Rejected: the eager rank-local attention mechanism as a sufficient Stage 4
candidate at the 23 ms projected 43-layer gate.

Unrun and still stopped: Stage 5 rank-local shared/routed MoE; Stage 6 full
rank-local mHC chain; Stage 7 full correctness closure; Stage 8 eager complete
decode gate; Stage 9 graph reconsideration; Stage 10 CPU parity work.
