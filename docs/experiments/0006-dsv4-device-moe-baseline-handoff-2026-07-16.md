# DeepSeek-V4 device-MoE baseline handoff — 2026-07-16

## Outcome

The best credible DeepSeek result currently recorded on
`exp/dsv4-device-moe` is **2.9757502295 exact decode steps/s**. Treat this as
the working performance baseline for the next experiment.

This is a single 128-token performance screen, not the research-charter median
of three interleaved repetitions. It is therefore an operational baseline, not
yet a fully validated result outside observed run variance.

The deterministic result artifact is:

`results/deepseek-v4-device-moe-fp4-switch-screen/summary.json`

This supersedes the initial 2.6087406356 steps/s device-MoE screen described
below.

## FP4 lookup continuation

The unfinished worktree replaced a dynamically indexed local FP4 value table
with an exhaustive switch and added a bit-exact oracle for all 16 E2M1 codes.
`make check` passed both CTest targets after the change.

A one-repetition, interleaved 128-token screen ran in tmux session
`dsv4-fp4-switch-screen` using the same model, prompt, devices, memory limits,
and exact contract as the initial screen. Its ignored artifact is:

`results/deepseek-v4-device-moe-fp4-switch-screen/summary.json`

| Metric | Reference | Device MoE |
|---|---:|---:|
| Decode seconds | 58.20821232 | 42.67831310 |
| Timed decode steps | 127 | 127 |
| Decode steps/s | 2.1818227178 | **2.9757502295** |
| Matched speedup | 1.000x | **1.3638826864x** |
| Prefill seconds | 8.907590903 | 4.656941916 |
| Decode activation H2D bytes | 2,923,999,232 | 1,605,947,392 |
| Decode activation D2H bytes | 2,373,896,192 | 1,747,585,024 |
| Decode synchronization calls | 161,846 | 62,285 |
| Maximum resident set, KiB | 146,298,740 | 146,297,760 |

Relative to the prior device-MoE screen, the candidate improved from
2.6087406356 to 2.9757502295 steps/s, a 14.1% single-run increase. On the
reference path, summed per-device CUDA kernel service fell from approximately
49.26 to 16.38 seconds, proving that the indexed device table was a material
kernel-path penalty.

The FP4 screen passed completed-run, exact execution, token, sequential route,
zero-decode-read, host-memory, and balanced-lease gates. It failed only the
5.0 steps/s target. Diagnostic traces were intentionally disabled for the
128-token performance screen; the exhaustive operation oracle and earlier
post-SiLU full-model trace remain the correctness evidence.

## Initial matched screen

The screen ran at revision `06a49a4eee5aa6dd6831bc65ebe6c4ace2d919a2`
with the following fixed conditions:

- model: `models/DeepSeek-V4-Flash-DSpark`;
- execution: exact base autoregressive, with DSpark disabled;
- prompt: `Hello`;
- 128 generated tokens and 127 timed decode steps;
- devices: GPU 0 RTX 5060 Ti, GPU 1 RTX 3090, GPU 2 RTX 3090;
- host-memory limit: 216 GiB;
- per-device VRAM fraction: 0.85;
- maximum context: 2048 tokens;
- identical model, binary, prompt, route sequence, RAM, VRAM, and I/O contract
  for the reference and candidate.

| Metric | Reference | Device MoE |
|---|---:|---:|
| Decode seconds | 87.71924102 | 48.68249387 |
| Timed decode steps | 127 | 127 |
| Decode steps/s | 1.4478009445 | **2.6087406356** |
| Speedup | 1.000x | **1.8018641620x** |
| Prefill seconds | 7.168148312 | 4.884771343 |
| Decode activation H2D bytes | 2,923,999,232 | 1,605,947,392 |
| Decode activation D2H bytes | 2,373,896,192 | 1,747,585,024 |
| Decode synchronization calls | 161,846 | 62,285 |
| Maximum resident set, KiB | 146,298,920 | 146,296,860 |

The candidate's peak framebuffer use was 9,398 MiB, 14,238 MiB, and
13,638 MiB on devices 0, 1, and 2 respectively.

The correct throughput calculation uses the 127 timed decode transitions:

```text
127 / 48.68249387 = 2.6087406356 decode steps/s
```

Dividing all 128 generated tokens by decode time produces 2.6292819 and
incorrectly charges the first token, which is produced by prefill, to decode.

## Correctness and admission evidence

The 128-token screen completed exact execution successfully. Reference and
candidate token IDs and sequential route traces were identical. Decode
checkpoint reads were zero, cache leases were balanced, no leases remained
active, and no cache eviction occurred. The physical-read counters in the
artifact include initialization/model staging and must not be interpreted as
decode NVMe demand.

The preceding post-SiLU diagnostic smoke is recorded in:

`results/deepseek-v4-device-moe-correctness-silu-smoke/summary.json`

That smoke passed identical tokens, routes, logits, and per-layer BF16 hashes.
It also recorded zero decode checkpoint bytes, balanced leases, no evictions,
and exact-mode completion. The longer performance screen intentionally omitted
the expensive diagnostic traces, so its diagnostic equality field is null,
while tokens and routes remained equal.

The performance harness reports `acceptance_pass: false` only because the
candidate did not meet the experiment's aspirational 5.0 steps/s target. Its
completion, token, route, zero-decode-read, host-memory, and lease gates passed.

## Repository state at handoff

The branch head contains two device-MoE commits:

- `72eed08` — grouped DeepSeek device MoE;
- `06a49a4` — preserve host SiLU semantics in device MoE.

The current `build/strata-deepseek-run` SHA-256 after the FP4 lookup rewrite is
`7b66c49419d99b629e2e42a1105f744cde33042de463ffc562ce3ac9ae6066a1`,
which matches both binary hashes recorded by the 2.9757502295 steps/s screen.

One older worktree item remains unfinished and must be preserved:

- `scripts/bench_deepseek_tok.sh` is untracked and predates the device-MoE
  screen.

The FP4 switch and exhaustive test are now built, tested, and screened as part
of the 2.9757502295 baseline. The untracked benchmark script omits
`--device-moe` and computes decode throughput using all generated tokens, so it
must not be used unchanged to reproduce either baseline.

No named DeepSeek benchmark tmux session was active at handoff; only a generic
attached session named `0` existed.

## Required next validation

Before calling the device-MoE result validated, execute at least three
interleaved reference/candidate 128-token repetitions under the same contract.
Report all runs and their medians; do not call a change a win if it is within
measured variance.
