# Experiment 0169 — the routed-expert staging term is at link rate, not serialized

Status: **HYPOTHESIS FALSIFIED, TASK B STOPPED.** The premise was that Laguna's
65 ms staging term ran far below the PCIe link and was therefore a serialization
or pinning defect with a simple fix. Measured on this box, a single expert-sized
H2D lands at roughly the same rate the runtime achieves. There is no simple
defect to correct.

## The claim being tested

Experiment 0166 measured Laguna decode at 236.98 ms/token with routed-expert
staging as `argmax_r`: 65.05 ms of miss staging, of which 63.82 ms is weight
memcpy, moving 229.50 MiB/token. That is **3.51 GB/s**.

That figure was compared against a PCIe 4.0 x16 rate of about 25 GB/s, then
against Gen 3 x16 at about 12.5, and the resulting 3.5-7x gap was read as the
signature CLAUDE.md names: "compare measured link throughput against the
hardware's rated figure; an order-of-magnitude gap is a serialization bug, not a
bandwidth limit."

Two things were wrong with that reading, and both are the same mistake — a rated
figure substituted for a measured one.

## What the hardware actually is

```
index  name                 gen.max  width.max  width.current
0      RTX 5060 Ti          3        16         8
1      RTX 3090             3        16         8
2      RTX 3090             3        16         16
```

**PCIe Gen 3, and two of three cards are at x8.** Gen 3 x8 is about 8 GB/s
theoretical and 6 GB/s practical, not 12.5 and not 25. The gap being chased was
mostly an arithmetic artifact of quoting the wrong link.

## Measured H2D at expert granularity

A standalone probe copies one Laguna expert projection (1,572,864 bytes) from
the mmap'd checkpoint, sweeping cold and warm page-cache state, pageable against
pinned staging, on each device. Medians of 60 reps.

| source | page cache | host-blocking | device | effective |
|---|---|---:|---:|---:|
| pageable | cold, first touch | 3.007 ms | 3.048 ms | 0.52 GB/s |
| pageable | warm | 0.196 ms | 0.258 ms | 6.09 GB/s |
| pinned | warm | 0.171 ms | 0.301 ms | 5.23 GB/s |

Repeated later under load the same copies ranged 2.1 to 3.7 GB/s on all three
devices, with high run-to-run variance.

Two results, both against the hypothesis:

- **Pinning is not the fix.** Pageable reaches 6.09 GB/s against pinned's 5.23,
  because pinning adds a host memcpy the pageable path does not pay. Every
  device showed pinned at or below pageable. The DeepSeek arena result that
  motivated this — pinned 12.1 GB/s against a much slower cold pageable read —
  was a different access pattern on a different mapping, and does not transfer.
- **3.51 GB/s is inside the measured range.** 229.50 MiB/token costs 37.4 ms at
  6 GB/s, 56.0 ms at 4, and 74.7 ms at 3. The runtime's 63.82 ms sits squarely
  between. It is moving its bytes at approximately the rate this link moves
  bytes.

The 3.007 ms cold figure is page-cache misses, not the DMA path; the checkpoint
is 63.665 GiB against a 51.121 GiB arena, so some of that is unavoidable and is
exactly what the charter means by reporting a model as I/O dependent rather than
hiding it behind a cache.

## Verdict

There is no simple defect here. The staging term is large because 229.50 MiB
crosses a Gen 3 x8 link every token, and that link is running at about its
practical rate. Making it smaller means moving fewer bytes or overlapping the
transfer with compute, and neither is a small correction:

- **fewer bytes** — the VRAM expert cache already hits 89.8%; raising it means
  capacity or policy work, and the model does not fit the arena;
- **overlap** — prefetching the next layer's experts behind the current layer's
  compute is a real scheduling change, not a fix.

Both are legitimate projects. Neither is the "simple defect, simple correction"
this task was scoped as, so it stops here rather than expanding into one.

## Lesson

The rule that produced this was applied with a rated figure instead of a
measured one, which is the same error the charter warns about two rules earlier:
"State every cost constant, where it was measured, and at what operating point."
An order-of-magnitude gap is evidence of a serialization bug only when the
denominator was measured on the link in front of you. Here the denominator was
wrong by 2-4x, and the entire gap was in it.

## Reproduce

The probe is not committed; it is 60 lines and reproducing it is cheaper than
maintaining it. Copy one expert-sized slice of an mmap'd shard to each device,
pageable and via `cudaMallocHost` staging, with `posix_madvise(..., DONTNEED)`
between reps to control page-cache state, and compare against
`nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.width.current`.
