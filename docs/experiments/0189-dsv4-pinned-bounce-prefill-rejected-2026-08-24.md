# Experiment 0189 — DeepSeek pinned bounce staging is a 1.07x mechanism

Status: **rejected by the predeclared 2x mechanism gate.** No runtime code was
built. Explicitly copying pageable expert shards into pinned buffers raises the
two-link transfer rate by only 7.3%, which projects to about 1.03x end-to-end
prefill while requiring roughly 4 GiB of pinned host workspace.

## Hypothesis and resource signs

Experiment 0188 measured a 619-token page moving 72.484 GB of routed-expert
weights in a 16.880 s demand term. Full-arena registration failed rank-local
VRAM admission. This independent hypothesis was that a bounded pinned bounce
arena could remove the same pageable staging cost without registering the
156.9 GB resident mapping.

The target was the current 16.880 s `argmax_r`. H2D volume, HBM traffic, CUDA
kernels, routes, precision, expert count, top-k, and device-resident weights
would remain unchanged. The negative signs were an extra host read/write pass,
about 4 GiB of pinned host memory at maximum page width, CPU copy work, and
small mapped-host bookkeeping on both GPUs. The host ceiling remained 231.9 GB
and the per-rank VRAM ceiling remained 22,135,873,536 B.

Rollback was binding if the isolated mechanism did not reach 2x, if its
end-to-end projection was not material, if any copied byte changed, or if
either memory ceiling failed.

## Cheapest falsifying measurement

The probe reproduces the current transformed format rather than inheriting a
matrix-sized transfer from another experiment:

- exact transformed shard: 7,077,888 bytes;
- 126 distinct shuffled shards per rank, close to the measured 619-token
  layer's routed working set;
- CUDA PCI-bus-order devices 1 and 2 concurrently;
- one pageable source arena per rank, NUMA-interleaved across both nodes;
- one distinct device destination and pinned staging slot per slice, so no
  source is overwritten while DMA owns it;
- three interleaved pageable/bounce repetitions in order A/B/B/A/A/B; and
- first, middle, and last device slices downloaded and byte-compared after
  every arm.

The fixed allocation and population cost dominated the six-arm measured window
by more than 10x. A full model load was rejected because this 5.36-second probe
could falsify the transfer mechanism before any runtime implementation.

## Result

Every arm copied 1,783,627,776 bytes across the pair and passed byte identity.

| Repetition | Pageable | Pinned bounce |
|---:|---:|---:|
| 1 | 5.9116 GB/s | 6.5564 GB/s |
| 2 | 6.1207 GB/s | 6.5649 GB/s |
| 3 | 6.1270 GB/s | 6.5703 GB/s |
| **median** | **6.1207 GB/s** | **6.5649 GB/s** |

The mechanism is **1.0726x**, far below the 2x gate. Scaling the measured
16.880 s demand term by that ratio predicts about 15.74 s, a 1.14 s saving from
a 40.777 s page: approximately **1.029x** end to end before accounting for
runtime integration or pinned-workspace management.

Peak probe RSS was 3,683,756 KiB. A production buffer sized for the maximum
per-rank layer working set would be larger, so accepting 4 GiB of pinned memory
for a 3% projected result is negative even before variance.

## Attribution correction

Experiment 0188 compared 4.294 GB/s in-system against rated PCIe payload and
described the gap as host staging/serialization. This exact-granularity probe
narrows that statement: the current term reaches about 70% of the pageable
probe's 6.121 GB/s and bounce staging recovers only 7.3%. It is not an
order-of-magnitude bandwidth gap. Most of the 16.880 s is the page's routed
working-set volume, not a removable pageable-copy defect.

## Verdict

Rejected. Do not add a pinned bounce arena to the runtime and do not build a
long model A/B. The next independent mechanism must reduce routed-expert volume
or change its overlap/ownership substantially; repackaging the same bytes does
not explain the vLLM gap.

Ignored artifacts are under `results/0189-dsv4-staged-page-upload/`.
