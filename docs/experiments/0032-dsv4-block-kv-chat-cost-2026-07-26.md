# Experiment 0032 — `--block-kv-cache` costs 11.8% of decode at chat scale

Status: **one-pair, descriptive only. Not a promotion or rejection gate.**
Recorded so a future agent does not have to re-discover this before proposing
compact KV storage as a chat default, and does not misattribute the phase-
profile artifact this run also surfaces.

## Origin

Found while instantiating the cost model at a chat operating point, per the
charter's first mandatory step, prompted by a report of 3.44 -> 3.56 tok/s
after enabling `--pin-resident-arena` and `--flash-attention` in
`strata-chat` (3.5%, against experiment 0024's 1.626x for pinning alone at a
different operating point). The phase profile below was the falsifying check
on whether that gap was a broken pin (it was not) or a smaller-than-benchmark
demand-wait term (it was, and something else was also present).

## Operating point

`strata-deepseek-run --json`, one prompt ("what is the closes star to us, and
how far is it?", 18 prompt tokens), 152 generated tokens, greedy, three GPUs,
216 GiB host ceiling, 0.95 VRAM fraction, `--flash-attention
--pin-resident-arena`, 32,768-token context ceiling. This is far from
experiment 0026/0027/0030's 511-token, 0.85-VRAM, 128-token benchmark
contract; no number here should be transplanted back onto that baseline or
forward onto a different chat prompt.

## What was measured

Two runs, same command, only `--block-kv-cache` present or absent. Not
interleaved, not repeated, no hash/route/logit equality check beyond visual
inspection of the answer text (both produced the same greedy continuation).
The workload was confirmed identical by the runtime's own counters:
`demand_h2d_bytes` 91,954,348,032 in both, 21,188 weight-cache misses and
8,672 evictions in both, decode `matmul_calls` 189,088 in both, generated
token count 152 in both.

| | `--block-kv-cache` | `--scalar-kv-cache` (default) | delta |
|---|---:|---:|---:|
| decode seconds / 152 steps | 42.366 | 37.887 | -4.479 |
| decode steps/s | 3.588 | 4.012 | **1.118x** |
| ms/step | 278.7 | 249.3 | -29.5 |
| `attention_seconds` (graph total) | 16.977 | 12.228 | -4.749 |
| sum of attention sub-timers (query+kv+score+output) | 12.125 | 12.152 | +0.027 |
| **attention time outside the sub-timers** | **4.852** | **0.076** | **-4.776** |
| MoE total (s) | 18.276 | 18.440 | +0.164 |
| — demand wait | 9.618 | 9.799 | +0.181 (see caveat) |
| mHC pre (s) | 5.629 | 5.693 | +0.064 |
| `kv_state_bytes` (admission plan, 32,768-token ceiling) | 132,354,560 | 474,365,952 | block KV is smaller, as designed |
| `rss_bytes` | 148,841,140,224 | 148,827,324,416 | +0.01%, noise |

Full JSON: block-KV arm and scalar arm are both reproduced verbatim in this
session's transcript; neither was saved to `results/` because
`strata-deepseek-run` does not currently emit to a file and this was an
interactive, unscripted pair. Re-running requires only the command above with
`--block-kv-cache` toggled; no fixed inputs beyond the prompt text need
preserving.

## What this actually shows

**A phase-attribution defect, not a new insight about the mechanism.** The
`attention_seconds` graph timer includes work that none of its four published
sub-timers (`attention_query_seconds`, `attention_kv_seconds`,
`attention_score_seconds`, `attention_output_seconds`) account for whenever
compact block storage is active. That residual is 4.852 s here (28.6% of
`attention_seconds`) and vanishes to 0.076 s (0.6%) on the scalar oracle.
Compact rows must be decoded from FP8/FP4 block storage before the existing
scalar attention oracle can consume them (`docs/experiments/0019`); that
decode cost is real work, and it is currently uncounted by any of the four
sub-timers a reader would otherwise sum to reconcile against the total. This
is worth its own small fix — attribute compact-block gather to its own timer
— before anyone reasons about attention cost from the sub-timers alone on a
`--block-kv-cache` run.

**Everything outside attention moved the other way, by less.** MoE total
+0.164 s, mHC pre +0.064 s — a combined +0.27 s against -4.749 s on attention,
consistent with 0019's one-pair finding that block KV is a net decode cost
(0019 measured turn-1 prefill/decode both slower under compact storage; this
is the first decode-only, chat-scale number). `demand_wait_seconds` moving
+0.181 s (1.9%) between two byte-identical-workload arms is within this
machine's noise floor per experiment 0026's control measurements and should
not be read as block KV affecting the weight cache; it does not touch weight
storage at all.

**The capacity case for block KV is untouched by this result.** `kv_state_bytes`
at this 32,768-token ceiling is 474 MB scalar against 132 MB compact — compact
is smaller, as designed, and 0019 projects the gap widens sharply at long
context (3.7 GB vs 14.5 GB at 1,048,576 tokens). Against 148 GB RSS neither
figure matters at this context length. The decode-cost finding here says
nothing about whether compact storage is still worth its cost once KV state
itself becomes the resource under pressure.

## Why this is not a rejection

No three-repetition interleaved measurement, no hash/route/logit equality
gate, no `make check` re-run tied to this pair, and the operating point is a
single 152-token chat generation rather than the repository's frozen 511-token
contract. `strata-chat` already defaults `--block-kv-cache` off
(`apps/strata_chat.cpp`), so no default is being challenged — this record
exists so a future proposal to flip that default, or to promote
`--block-kv-cache` for chat capacity reasons, starts from a real decode-cost
number instead of re-discovering it.

## Follow-ups this opens, not yet started

1. Attribute compact-block gather to its own graph timer so
   `attention_seconds` reconciles against its sub-timers on every KV mode.
2. A proper interleaved three-repetition A/B at a representative chat context
   length, with the standard exactness gates, before any doc or default
   changes based on the 1.118x figure.
3. Find the KV-state-bytes crossover context length at which block KV's
   capacity saving is worth its measured decode cost, rather than assuming
   one end or the other dominates at chat scale.

## Also recorded from the same session: the pin and the profile it enabled

- `--pin-resident-arena` is reachable from `strata-chat` as of `b9877a3` and
  was confirmed pinning correctly in this session by three independent
  signals: `"resident_arena_pinned": true` in `strata-deepseek-run --json`,
  `resident_pin_seconds` ~17.1-17.7 s matching experiment 0024's ~16.9 s
  registration cost, and an effective decode weight-H2D rate of 6.6-6.7 GB/s
  matching 0024's pinned figure (0024's unpinned reference was 1.8-3.0 GB/s).
  `strata-chat` does not currently surface either field, so this confirmation
  required `strata-deepseek-run --json`; chat-side surfacing remains open.
- At this chat operating point (18-prompt-token, 152-decode-token, 0.95 VRAM
  fraction, `--flash-attention --pin-resident-arena`, no block KV), the decode
  phase profile is: MoE demand wait 64.5 ms/step (25.9%, still `argmax_r`),
  MoE compute 48.0 ms/step (19.3%), mHC pre 37.5 ms/step (15.0%), attention
  query projection 26.2 ms/step (10.5%), attention output projection
  24.3 ms/step (9.7%), attention score 21.9 ms/step (8.8%), attention KV
  7.6 ms/step (3.0%), MoE prepare excl. wait 6.8 ms/step (2.7%), remainder
  10.5 ms/step (4.2%), unattributed 1.5 ms/step (0.6%). Demand wait remains
  the largest single term with no attribution gap hiding a larger one, which
  is the basis for treating experiment 0031's copy-stream mechanism as
  targeting a real bottleneck at this operating point, not only at the 511-
  token benchmark contract. This profile is one pair at one prompt and must be
  re-measured, not assumed, before it is used to size any mechanism's ceiling.
