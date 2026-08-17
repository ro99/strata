# 0119 — Host KV admission ignores the prefill page width

Status: **fixed.** `--prefill-page-tokens 8192` is worth 2.5x on a 236-token
prompt and about 4.8x on a 3,845-token one, but before this fix any prompt over
roughly 2,700 tokens aborted with
`DeepSeek KV host cache capacity is exhausted`.

## Why the flag matters

`strata-server` never set `--prefill-page-tokens`, so it ran the 64-token
default while every DSV4 prefill benchmark passed 8192. Measured at the served
operating point (`--decode-topology rank-local-tp2`, `--max-context 16384`,
236-token prompt, identical generated ids):

| `--prefill-page-tokens` | prefill | PP |
| ---: | ---: | ---: |
| 64 | 58.68 s | 4.02 tok/s |
| 8192 | 23.52 s | **10.04 tok/s** |

2.50x, bit-identical output. This is the whole gap between the served
configuration and the benchmark configuration; nothing else in the served
command line was wrong.

## The defect the flag exposed

`plan_dsv4_resident_topology` budgeted the sliding stream at

```cpp
maximum_sliding_rows = min(maximum_context_tokens, window + sliding_capacity - 1)
```

= 128 + 256 - 1 = **383 rows**, two 256-row blocks per layer. That is the
steady-state decode figure, and it is wrong for the batched prefill page path.

A page appends all of its rows before attending any of them, so it holds a
retention floor at `position_base + 1 - kWindow` for the life of the page. Row
`r` of the page attends `[base + r + 1 - window, base + r]`, so the union over
the page is `[base + 1 - window, base + N - 1]` — **`N + window - 1` live rows**,
not `window`. At `N` = 8192 that is 8,319 rows against a 383-row budget, and
`allocate_block` fails the first time the page crosses the budget.

The failure is therefore a function of *page width*, not context length, which
is why it never appeared at 64 (retains ~192) and why our 2,612-token benchmark
at page 8192 passed by luck: 2,611 rows of sliding state came to roughly 67 MB
against a 71.23 MiB budget. A few hundred tokens more and it would have failed
during the promotion campaign.

## Fix

Admission now takes the page width and budgets the span the page actually pins:

```cpp
const auto page_rows = std::max<std::uint64_t>(1U, config.prefill_page_tokens);
const auto maximum_sliding_rows = std::min<std::uint64_t>(
    config.maximum_context_tokens, window + page_rows + sliding_capacity - 2U);
```

`Dsv4AdmissionConfig::prefill_page_tokens` defaults to 1, so every caller that
does not page keeps the previous budget exactly. The cap against
`maximum_context_tokens` is unchanged, so the budget never exceeds the declared
context: at 16,384 tokens and page 8192 it is 8,576 rows.

## Gates

- 3,845-token prompt, page 8192, `rank-local-tp2`, 16,384-token context: failed
  with `KV host cache capacity is exhausted` before, prefill 197.5 s / **19.47
  tok/s** after — a single 3,844-row page, beyond the 2,611 rows any previous
  run had built.
- 236-token A/B above: identical generated ids at both page widths.
- `make check` 2/2.

## Known gap, not fixed here

`placement_model.cpp` builds its own `Dsv4AdmissionConfig` for the descriptive
placement report and has no page width to give it — `PlacementRequest` carries
none. The printed `kv-cache` row therefore still reports the 383-row figure when
the runtime is paging. The report is descriptive only; the runtime's KV budget
comes from its own admission, which this change fixes. Threading the page width
through `PlacementRequest` changes the plan-cache key and is left separate.
