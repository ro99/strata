# Experiment 0008 — DeepSeek fast-loading promotion

Status: **promoted as the default DeepSeek loading path on `main`**.

## Scope and contract

This promotion combines five bounded loading experiments without changing
model precision, routing, expert selection, top-k normalization, scoring,
shared-expert execution, mHC state, or decode arithmetic. The admitted resident
bytes and the zero-checkpoint-read generation contract remain unchanged.

- Primary metric: DeepSeek runtime initialization seconds.
- Correctness: `make check`; identical generated tokens, resident staged bytes,
  generation/decode checkpoint reads, and cache hit/miss/eviction accounting.
- Memory: 216 GiB host-memory ceiling and 0.85 VRAM fraction on devices 0,1,2.
- Rollback: any correctness/accounting mismatch, admission regression, or
  failure to reduce the matched initialization time materially.

The screens used the pinned `DeepSeek-V4-Flash-DSpark` checkpoint, context 2048,
one generated token from prompt `Hi`, and one candidate/reference pair per
bounded experiment. These are deliberately large-effect engineering screens,
not three-repetition variance measurements.

## Promoted changes

1. `ea7de33` reads checkpoint tensors directly into their final anonymous
   resident arena, removing the temporary tensor buffer and full-size copy.
2. `5bf081f` stages independent checkpoint shards concurrently with a bounded
   default of eight readers and per-shard exact read accounting.
3. `81698cd` converts FP8 `wo_a` in 128x128 blocks with a 256-entry E4M3 lookup,
   applying the block scale once and writing BF16 directly. Its scalar-oracle
   fixture preserves the declared numerical result.
4. `5f5a0fb` warms independent GPU spine partitions concurrently with up to
   three workers, while checkpoint tensor ownership and cache accounting remain
   isolated.
5. `c447aff` overlaps immutable spine warm-up reads with resident staging. The
   warm-up path bypasses the not-yet-complete resident map, and generation does
   not begin until staging has joined successfully.

The library and CLI now enable overlap by default. The explicit
`--serial-resident-warmup` option retains the reference path, while
`--resident-read-workers` and `--spine-warmup-workers` retain bounded tuning and
diagnostic control.

## Matched screens

| Experiment | Reference init | Candidate init | Speedup | Candidate staging |
|---|---:|---:|---:|---:|
| Parallel resident reads | 149.0748 s | 64.2800 s | 2.319x | 22.2686 s |
| Block-wise `wo_a` lookup | 60.7659 s | 36.6405 s | 1.658x | 21.9671 s |
| Parallel spine warm-up | 35.5170 s | 28.0393 s | 1.267x | 20.3450 s |
| Overlapped stage/warm-up | 31.4793 s | **22.1771 s** | **1.419x** | 20.9074 s |

The earlier direct-resident-read screen reduced initialization from roughly
190.97 seconds to 144.12 seconds. Across the full lineage, initialization fell
from approximately 191 seconds to 22.177 seconds, about 8.6x.

Every recorded comparison matched generated tokens and staged bytes. The two
later fully instrumented comparisons also matched generation/decode checkpoint
reads and cache hits, misses, and evictions; generation and decode checkpoint
reads remained zero. The final candidate staged 148,228,800,512 resident bytes,
reported 145,345,388 KiB maximum RSS, and completed `make check`.

Ignored deterministic artifacts are stored under:

- `results/deepseek-v4-parallel-read-ab/`
- `results/deepseek-v4-woa-lookup-ab/`
- `results/deepseek-v4-parallel-warmup-ab/`
- `results/deepseek-v4-overlap-stage-warmup-ab/`

## Remaining opportunity

Resident staging is now the critical path: 20.907 of 22.177 initialization
seconds in the final screen. The next loading experiment should target arena
page-fault and storage overhead rather than add more warm-up concurrency. The
final process recorded roughly 38.3 million minor faults, making a huge-page
arena hint and matched first-touch screen the next bounded hypothesis.
