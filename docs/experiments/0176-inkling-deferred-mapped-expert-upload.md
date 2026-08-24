# Experiment 0176 — Inkling orders mapped expert uploads once per batch

Status: **ACCEPTED.** MXFP4 routed-expert bytes now transfer on the CUDA copy
stream directly from their persistent checkpoint mapping. One device event
orders the complete selected-expert batch before execution; the host no longer
waits after gate, up, and down for every cache miss.

## Cost model and contract

With the bounded arena enabled, the 128-token control spent 9.03 seconds in
routed MoE and 7.84 seconds staging 31.58 GiB. Routed work was the per-forward
`argmax` at roughly 71 ms against attention's 34 ms. The mechanism targets the
serial cross-engine handoff, not transfer volume.

- H2D bytes, prepack kernels, expert count/top-k, precision, routing, cache
  capacity, and HBM bytes are unchanged.
- Only MXFP4's direct mapped path may defer: its source spans live for the
  checkpoint reader's lifetime. NVFP4 and the pinned MXFP4 control reuse
  scratch and remain synchronous.
- `synchronize_uploads` records an event after all pending copies/prepacks and
  makes the execution stream wait on-device. `collect_moe` remains the host
  completion boundary.
- Correctness requires identical generated text and CUDA route census, then
  `make check`.
- Rollback is a whole-model regression outside variance or any lifetime,
  numerical, or asynchronous error.

The change also closes an existing ordering hole for resident spine uploads.
Inkling already requested deferred spine copies during initialization but never
ordered execution behind them. They normally finished during load by accident;
initialization now installs the same explicit per-device event dependency.

## Results

Three interleaved fresh processes per arm, resident OS page cache, no expert
warmup, and two RTX 3090s plus one RTX 5060 Ti:

| workload | synchronous median | deferred median | decode speedup | stage wall |
|---|---:|---:|---:|---:|
| 16 generated tokens | 5.385 tok/s | 5.673 tok/s | 1.054x | 2.370 → 2.150 s (1.102x) |
| 128 generated tokens | 8.774 tok/s | 9.072 tok/s | 1.034x | 7.828 → 7.142 s (1.096x) |

The short rates were `5.350, 5.385, 5.396` against
`5.675, 5.670, 5.673`. The long rates were `8.774, 8.798, 8.732` against
`9.037, 9.072, 9.107`. All twelve arms emitted identical continuations and
route censuses. At 128 tokens the measured upload wait fell from a 444 ms
median to 1 ms; the copy-call counter rose slightly (`2.206 → 2.347 s` median),
so the 686 ms stage reduction is the net result rather than a cherry-picked
subcounter.

## Reproduction

```bash
cmake --build build-release --target strata-inkling-probe -j
scripts/inkling_deferred_upload_ab.sh
RESULT_DIR=results/inkling-deferred-upload-128 TOKENS=128 \
  scripts/inkling_deferred_upload_ab.sh
```

Raw logs are ignored under `results/inkling-deferred-upload*`.

## Decision

Keep deferred mapped uploads. The gain is real but deliberately modest because
all six routed experts still use only the layer's assigned GPU. The next gate
distributes exact expert batches over all admitted devices, which targets the
remaining 7.14-second stage term and the otherwise idle copy/compute engines.
