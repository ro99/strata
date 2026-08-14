# Experiment 0092: M3 callback-free timing falsifier

Date: 2026-08-12
Branch: `exp/dsv4-a2-ownership-screen`
Base: `70b41fa`
Disposition: **M3 PASS / REVIEW REQUIRED under the user-amended topology gate;
the original non-CPU planning assumption is falsified and retained as evidence**

## Question and gates

After Experiment 0091 made the 43-layer queued callback-free chain exact, this
experiment measured that same position-104 chain from submission through the
terminal output head. Setup and replay reset were outside the measured window.
One warm-up preceded three repetitions in the same initialized process.

The original falsifier recorded these engineering gates:

- wall median `<=120 ms/forward`;
- routed CPU bandwidth median `>=36.7 GB/s`;
- non-CPU dependency median `<=30 ms/forward`;
- exact terminal output in every repetition;
- zero page callbacks, decode checkpoint I/O, and timed allocations; and
- RSS and per-device VRAM below the established ceilings.

The run showed that the standalone `non-CPU <=30 ms` condition was not an
appropriate M3 acceptance gate: it incorrectly required the topology milestone
to absorb work reserved for later CPU parity. After reviewing the result, the
user amended M3 acceptance to **median wall <=115 ms/forward**, with exactness,
CPU-bandwidth, I/O, allocation, and memory gates unchanged. This does not erase
the original `30 ms` miss; it records that assumption as falsified.

M3 is an intermediate topology milestone, not the final `<=100 ms/forward`
(10 tok/s) program target. The roadmap reserves CPU gate/up, down, and
weighted-reduction parity for the separate Stage 10 experiment after topology
validation.

## Workload and accounting

The measured arm queues all 43 exact rank-local layers with no page callback
and one terminal completion. It retains the local-logit diagnostic and counts
it inside wall time, then separately measures the entire terminal-head interval.
That diagnostic accounts for `1,165,312 B` D2H, `32,768 B` H2D, and `258,560 B`
of logical head all-gather traffic.

The routed CPU term is the slower rank's summed exact CPU-MoE body time over
`3,449,290,752 B` of gate/up, down, and reduction payload. The non-CPU envelope
is `wall - CPU critical`. It contains attention, mHC, device transfers, shared
expert work, NCCL/join/publication, orchestration, and terminal service; this
experiment does not isolate all of those terms. The reported shared-expert CUDA
event belongs only to the last queued command and is not a full-chain GPU term.

The complete arm, including setup and correctness controls, took about 85 s;
the three measured forwards themselves took less than one second. A shorter
probe could not test the queue-depth ownership behavior that caused 0090.

## Binding result

Artifacts:

```text
results/dsv4-rank-local-executor/m3-43-chain/timing-r3/
combined.log SHA-256 fec60357ee1adc3fb4522a7f08e742a39be15869d4cd58d1161bb96479e46540
source.diff  SHA-256 ba345132f0aeacc9f70da078bfec59a9195630f38663a32ce3682f67810c574f
exit status 3 (intentional gate rejection)
```

| rep | wall ms | CPU critical ms | CPU GB/s | non-CPU ms | terminal head ms | non-CPU less entire head ms |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 116.790787 | 75.386430 | 45.754796 | 41.404357 | 2.331919 | 39.072438 |
| 1 | 114.944312 | 73.896784 | 46.677143 | 41.047528 | 2.288864 | 38.758664 |
| 2 | 112.322810 | 71.550569 | 48.207733 | 40.772241 | 2.287216 | 38.485025 |
| median | **114.944312** | **73.896784** | **46.677143** | **41.047528** | **2.288864** | **38.758664** |

The wall range is `112.322810–116.790787 ms`; the terminal-head range is
`2.287216–2.331919 ms`. The measured scope therefore executes at approximately
`8.70 forwards/s` and remains `14.944312 ms` above the final 100 ms target.
This is not a complete end-to-end decode throughput claim.

Gate disposition after review:

- amended M3 topology gate: **PASS**, `114.944312 <= 115 ms`;
- routed CPU bandwidth: **PASS**, `46.677143 >= 36.7 GB/s`;
- original non-CPU planning assumption: **FALSIFIED**,
  `41.047528 > 30 ms` by `11.047528 ms`; it is diagnostic, not the reviewed
  M3 acceptance gate;
- even subtracting the entire terminal-head diagnostic: **FAIL**,
  `38.758664 > 30 ms` by `8.758664 ms`.

The terminal diagnostic therefore cannot explain the old assumption's miss.

## Correctness and resources

Every measured repetition reproduced the accepted terminal object exactly:

```text
weighted e1a9a77f0b01a361
input    122a716defe84e1b
hidden   5017083817dd2848
logits   343d766f3f5c0af3
token    8806
```

Every repetition invoked zero page callbacks and recorded zero checkpoint I/O,
workspace allocations, and weight allocations in the measured path. RSS was
`158,623,404,032–158,623,412,224 B`; each GPU used `8,190,558,208 B`, below
the `231,928,233,984 B` host and `21,287,272,448 B/GPU` ceilings.

The two earlier arms, `timing-r1` and `timing-r2`, are preserved as preliminary
non-binding evidence. They reached the same rejection, but their reporting did
not yet distinguish the last-command shared-expert event from a full-chain GPU
term. `timing-r3` is binding.

## Cost model and disposition

At this operating point the directly instantiated envelope is:

```text
tau_observed = 114.944312 ms
CPU routed term = 73.896784 ms at 46.677143 GB/s
Sigma_non_CPU envelope = 41.047528 ms
```

The all-resource `argmax` inside the non-CPU envelope remains indeterminate
because attention/HBM/link/NCCL service was not independently timed. No new
optimization mechanism is selected from this result.

M3 is **PASS / REVIEW REQUIRED** under the user-amended `<=115 ms` topology
gate. Its exact callback-free implementation, `114.944312 ms` median, and
CPU-bandwidth/resource gates pass. The remaining distance to the final target
is `14.944312 ms/forward`. That gap is intentionally left for subsequent
full-runtime validation and the roadmap's separate Stage 10 CPU gate/up, down,
and weighted-reduction parity work, re-measured at that later operating point.
The original `non-CPU <=30 ms` condition remains recorded as a falsified
planning assumption and is not relabeled as having passed.
