# 0118 — DSV4 paged prefill leases an evicted sliding row

Status: **fixed.** Any served prompt longer than about 129 tokens failed at
`forward_prefill`. Not a regression from the 2026-08-16 evening merge; the
defect entered `main` at 10:16 that day and was masked in every benchmark by a
flag the server does not set.

## Symptom

Served requests returned
`DeepSeek KV device row is not retained`, and every subsequent request on the
same process then returned
`DeepSeek device mHC slot reservation is out of order` until restart. An
8-token prompt completed 231 tokens normally; a 404-token prompt failed
mid-prefill, about 27 s in.

## Mechanism

`physical_paged_attention_page` leases every block of the sliding table by the
block's **first** row:

```cpp
const auto logical_row = block.logical_begin / block.compression_ratio;  // 0
auto lease = kv_cache->acquire_device(active_sequence, kind, layer,
                                      logical_row, slot);
```

The lease names a block, not a row — every row in the block resolves to the same
device buffer — but `acquire_device` validates the row against the table's
retained window `[minimum_row, end_row)`.

Those two granularities do not match. Blocks are retired whole, at
`kDsv4PhysicalKvBlockRows` = 256 rows. The sliding window retires rows one at a
time: after appending a page based at `position_base`,

```
minimum_row = min(end_row - kWindow, position_base + 1 - kWindow)
```

So a block can be live — most of its rows are legitimate candidates — while its
first row has already fallen out of the window. The first prefill page based at
or beyond `kWindow` triggers it. With the default `prefill_page_tokens` = 64:

| page | base | end_row at lease time | minimum_row | block 0 first row | result |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 64 | 0 | 0 | ok |
| 1 | 64 | 128 | 0 | 0 | ok |
| 2 | 128 | 192 | **1** | 0 | **rejected** |

Measured exactly: `outside retained window [1,192); kind=0 layer=0 row=0`.

## Why no benchmark caught it

Every recent DSV4 prefill script passes `--prefill-page-tokens 8192`, which puts
the whole prompt in **one** page based at 0 — the single configuration in which
`position_base + 1 - kWindow` can never exceed 0. `strata-server` never sets the
flag and takes the default of 64. The benchmark did not reproduce the production
access pattern, so it reported no problem where a deterministic one existed.

## Fix

Lease the block by its most recent row, which is retained for exactly as long as
the block is:

```cpp
const auto first_row = block.logical_begin / block.compression_ratio;
const auto logical_row = block.used_rows == 0U
    ? first_row : first_row + block.used_rows - 1U;
```

Page indices, candidate rows and the numerical contract are unchanged: `locate`
derives `candidate.row` from the block's `logical_begin`, not from the row used
to take the lease.

## Provenance

`lease_table` was introduced by `f89b1c6` and reached `main` in `54505ba`
(10:16). `git diff e56d75a..ab34b84` — the evening merge — touches
`deepseek_runtime.cpp` in three hunks only: a `getenv`-guarded trace print and
two `linear_rows` tensor-path flags. It does not touch `lease_table`,
`acquire_device`, the append path, or the retention floor, and
`deepseek_kv_cache.cpp` is untouched.

## Gates

- The failing arm (404-token prompt, `--decode-topology rank-local-tp2`,
  `--max-context 16384`, default page tokens) reproduced deterministically in
  both `strata-deepseek-run` and `strata-server`, with and without
  `--detailed-timing`, with and without the trace env var.
- After the fix the same arm generates; `strata-server` served a 404-token
  prompt (295 completion tokens) and a following 9-token prompt on the same
  process.
- `make check` 2/2.

## Open defect

A failed generation leaves the device mHC stage non-zero, so every later request
on that process fails `dsv4_mhc_reserve_slots` until restart. One bad request
bricks the server. Tracked separately; the error now reports the stage it
observed.
