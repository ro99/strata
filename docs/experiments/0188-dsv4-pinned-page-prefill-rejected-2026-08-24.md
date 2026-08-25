# Experiment 0188 — pinned DeepSeek page prefill is outside the TP2 envelope

Status: **rejected at admission; no throughput arm ran.** The existing
`--pin-resident-arena` mechanism cannot be part of the rank-local TP2 production
recipe at the declared 0.95 VRAM fraction because CUDA host registration raises
both ranks above the program VRAM ceiling. The runtime failed closed before
generation, as required.

## Hypothesis and gates

The 156.9 GB transformed routed-expert arena is pageable. A cold 619-token page
issued 72.484 GB of demand H2D in 16.880 s, only 4.294 GB/s aggregate. The
hypothesis was that registering that arena would remove the driver's pageable
staging copy and reduce the current `argmax_r` without changing transfer volume.

- Primary metric: prefill tok/s for one 619-token page, followed by three
  interleaved process repetitions if the mechanism arm passed admission.
- Correctness: identical generated IDs, the DeepSeek operation/layer oracle,
  and `make check`.
- Memory ceiling: the existing rank-local program ceiling of 22,135,873,536 B
  per RTX 3090; no change to the 0.95 production VRAM fraction.
- Rollback: registration absent, no material H2D-rate gain, any output change,
  or either rank exceeding its declared ceiling.

Page-locking changes no model arithmetic, H2D bytes, HBM bytes, CUDA kernels,
router semantics, expert count, top-k, or VRAM-resident weights. Its negative
resource signs are a one-time registration cost, an unswappable 156.9 GB host
allocation, and CUDA's mapped-host bookkeeping.

## Cost model before the arm

Release + NCCL, two RTX 3090s at 250 W and 1605 MHz, PCI bus-order devices
`1,2`, 16K context, VRAM fraction 0.95, rank-local TP2, physical KV, and an
8,192-token scheduling page:

| Serial phase, 619-token page | Time |
|---|---:|
| expert preparation / demand H2D | **16.872 s** |
| device-MoE remainder | 7.791 s |
| attention | 11.802 s |
| router | 1.342 s |
| mHC post | 2.829 s |
| complete prefill | 40.777 s |

Demand traffic was 72,483,896,832 B in 16.880 s, or 4.294 GB/s aggregate.
GPU 1 is PCIe Gen3 x8 and GPU 2 Gen3 x16 at this operating point, about
23.63 GB/s of aggregate rated payload. The current term is host staging and
serial submission, not link capacity. It is `argmax_r`.

The unprofiled production medians over three request repetitions were 4.332,
17.431, and 22.757 tok/s at approximately 36, 500, and 1,950 tokens. Decode was
8.627 tok/s, or 115.92 ms/token. Prefill cost was 57.37 ms/token at about 500
tokens and 43.94 ms/token at about 1,950, ratios 0.495 and 0.379 against decode.
Both fail the predeclared below-0.25 plausibility bound.

## Cheap arm and result

One current-runtime profile was budgeted at about 106 s fixed setup plus a
29--41 s measured window. A longer A/B matrix was contingent on admission. The
older cold-random-slice microbenchmark predicted the mechanism, but its constant
was not reused as a result at this operating point.

The candidate command added only `--pin-resident-arena`. It failed after the
rank-local executor was installed:

```text
rank-local CUDA device 1 uses 22210019328 B after setup,
above the 22135873536 B program ceiling
rank-local CUDA device 2 uses 22203727872 B after setup,
above the 22135873536 B program ceiling
```

The overruns are 74,145,792 B (70.71 MiB) and 67,854,336 B (64.71 MiB).
Registration therefore has a real device-memory sign that the older centralized
experiment did not expose. No prefill number or correctness result exists for
this arm because continuing after this gate would violate the admission
contract.

## Verdict

Rejected. Do not add `--pin-resident-arena` to the DeepSeek rank-local TP2
runbook, do not raise the 0.95 envelope to manufacture a passing regime, and do
not quote the older centralized pinning gain for this configuration. The next
independent hypothesis must attack a serial term that fits inside the existing
memory contract.

Raw logs are ignored under
`results/0188-dsv4-production-throughput/profile-{main,pinned}/`.
