# Experiment 0125 — DSV4 multi-turn KV lease leak, and the prefill regression that was not one

Status: **fix landed; the reported prefill regression is falsified as a
measurement artifact, and the routed-expert tier is exonerated.**

## What was handed over

An operator-reported prefill regression of 7.27 -> 1.64 tok/s (4.4x) on the
served DeepSeek-V4-Flash endpoint, decode flat. The handover attributed it with
"high confidence" to the routed-expert tier of 0124 reserving its VRAM out of
the centralized prefill expert cache, and recorded the mitigation as having
removed `--static-expert-plan` from the served llama-swap config.

## The attribution was wrong, on three independent grounds

1. **The flags were never in the served config.** `diff` against the recorded
   backup `config.yaml.bak-20260818-232828` shows exactly one changed line:
   `strata_server_nccl` moved from `build-nccl/` to `build-release/`. Neither
   version passes `--static-expert-plan`; `grep -rn static-expert` over the
   whole llama-swap directory finds nothing.
2. **The tier code is not in the binary the operator was running.**
   `build-nccl/strata-server` is dated 2026-08-16 21:57 — before the tier
   commits, before the pool-width regression `6fd2637` (08-17 21:25), and
   before the equivalence-oracle refactor. `strings` finds 0 occurrences of
   `static-expert` in it, against 4 in `build-release`.
3. **The reservation is inert without the flag.** `deepseek_runtime.cpp`
   guards it as `config.static_expert_plan_path.empty() ? 0U :
   config.static_expert_tier_bytes`.

The binary and the config were both unchanged across the reported regression.
A cause was assigned to a mechanism that was not running.

## Measured prefill, current main, served endpoint

Two repetitions at a 3.2k-token prompt, then a length sweep on the same warm
server. NVMe read bytes were sampled around every request.

| rep | prefill tok/s | decode tok/s | nvme read | wall |
|-----|---------------|--------------|-----------|------|
| 1   | 20.003        | 6.880        | 164.9 GB (model load) | 288.0 s |
| 2   | 21.460        | 6.993        | **0.0 GB**            | 161.2 s |

Steady-state prefill reads zero NVMe bytes and the page cache is flat, so the
cold-cache hypothesis is dead as well. The length sweep:

| prompt tokens | prefill tok/s | ms/token |
|---------------|---------------|----------|
| 6      | 2.24  | 445.5 |
| 16     | 3.26  | 306.8 |
| 219    | 10.61 | 94.3  |
| 867    | 18.50 | 54.0  |
| 2,173  | 21.98 | 45.5  |
| 3,363  | 21.76 | 46.0  |
| 6,721  | 19.98 | 50.0  |

`prompt_per_second` rises about 10x from a short prompt to a realistic one,
because per-token cost amortizes (223 -> 77 -> 36 ms/token). Both operator
numbers sit on the low-amortization end of this curve; the operator confirmed
the failing turn was "what is a banana?", 8 prompt tokens. Current main is
about 3x faster at a realistic prompt than the 7.27 tok/s that was being
treated as the healthy baseline. **There is no prefill regression.**

A three-term least-squares fit over the 219..6,721 points gives
`t(L) = 13.34 s + 35.75 ms*L + 1.833 us*L^2`. That fit predicts 1.42 tok/s at
20 tokens and 6.80 at 120, which appears to reproduce the operator's pair —
but direct measurement at 6 and 16 tokens refutes its constant term: measured
prefill there is 2.673 s and 4.908 s, not the ~13.5 s the fit implies. The fit
was extrapolated below its own data range and its `F` is an artifact. Recorded
here because it is exactly the failure the charter warns about, and it was
briefly believed.

`prompt_per_second` is computed as `prefill_tokens / prefill_seconds` where
`prefill_tokens` is only the non-reused tail (`subspan(prefill_offset)`), so in
a multi-turn chat the figure covers only the new message. llama-swap's
`/api/metrics/activity`, which the operator reads, reports exactly this.

## The real defect

The second turn of every conversation failed:

```
DeepSeek KV cannot mutate an in-flight device block
```

Rank-local attention page leases are dropped at the next token's same-layer
prepare, which an ended generation never runs, so a finished turn leaves every
layer holding leases on the blocks it last read. `reset_sequence` drops them,
but runs only when `prefill_offset == 0`. A continuation keeps the sequence,
skips `reset_sequence`, and prefills the new turn into the last block of that
same sequence; `Dsv4KvCache::append` refuses to mutate a leased block.

The operator's log splits exactly on this: `chatcmpl-2` (`messages=1`)
completed, `chatcmpl-3` (`messages=3`) failed. The cancelled `chatcmpl-1` was
incidental.

## Fix and gates

`release_retained_kv_leases()` backs both paths. `reset_sequence` keeps its
behaviour and additionally clears `index_leases`, `index_pages` and
`pending_attention_leases`, none of which it had ever cleared; the continuation
path calls it before prefilling.

- `make check`: 3/3, twice.
- New regression test `DeepSeek KV append refuses a block a retained lease
  still holds` pins the guard: an append into a still-leased block is refused,
  and dropping that lease is what admits it.

## Open defects, not addressed here

- **The mHC wedge.** One aborted request bricks the endpoint until restart:
  `dsv4_mhc_stage` is left at 1, and every later slot reservation fails with
  `slot reservation is out of order (stage=1 moe_in_flight=false)`. The
  recovery path exists but `abort_chain()` returns early on
  `!chain_active || chain_count == 0` before reaching
  `dsv4_mhc_abort_branch`, and all three call sites discard its result.
- **The dark 41.54 ms decode term.** All nine `cudaEventRecord` sites in the
  rank-local executor are gated on `!impl_->chain_mode`, so production decode
  records nothing; the 46 sub-counters read zero by construction.

## Rules this experiment re-earned

- A stated cause is not a measured one. Three cheap checks — a config diff, a
  binary timestamp, and `strings` — falsified the handover's attribution before
  any run.
- Do not extrapolate a fit below its data range. The 13.34 s constant term was
  produced by exactly that and was wrong by 5x.
- A throughput figure whose denominator is a per-request phase is meaningless
  at small numerators. An 8-token turn cannot measure prefill.
