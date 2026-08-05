# Experiment 0058 — host-experts decode path: static CPU placement of all routed experts

Status: **rejected at the operating point.** Running every layer's six routed
experts on the host (`--host-experts`) measured **449.2 ms/step (2.22 tok/s)
against the 222.58 ms/step (4.49 tok/s) GPU baseline** on the same 586-token
prompt, same three GPUs, same memory budget — a **0.49x** decode. The host MoE
term alone is 294.6 ms/step. The hypothesis fails by construction and by
measurement, and this record states why, what the probes established, and the
limit of this investigation.

Follows experiment 0057 (serial-term cuts, which set the 222.58 baseline) and
0054 (the same static-placement idea, rejected by arithmetic without code).
This is the first time the static path has actually run end to end.

## Contract

- Hypothesis: the routed-expert bytes already sit in a resident host arena
  (148 GB, 3.449 GB read per token), so computing each expert on the cores of
  the node its bytes were bound to makes every read node-local, uses both
  memory controllers on disjoint data, and beats the GPU path's 111.59 ms/step
  MoE term. Break-even for host vs GPU is ~32.2 GB/s in situ (0054).
- Mechanism built on the branch: `Dsv4ResidentWeightStore::stage()` binds each
  expert's bytes to `expert % nodes` with MPOL_BIND before first touch;
  `host_moe()` partitions each layer's routed experts by node and runs the
  FP4 gate/up/quantize/down phases on the addressed 28-worker pool pinned to
  physical cores 0-27, with lane i on the node of CPU i.
- Bottleneck measurement plan, before the run: instantiate the host path's
  read rate at the real operating point; the 26.23 GB/s standalone kernel
  constant from prior experiments was an operating point of its own and was not
  assumed to transfer.
- Correctness gate: bit-identity with the CUDA path per the declared numerical
  contract (this run did not gate on bytes because the throughput gate below
  already closed it; the host kernel itself remains bit-exact and tested).
- Kill criterion (0054, binding at the operating point that matters): host MoE
  must beat 32.2 GB/s in situ to be competitive with the device path. It
  measured **11.7 GB/s** in production and **13.1 GB/s** in a faithful kernel
  probe. Negative; the work stops here.

## Measurement, cheap first

Two standalone probes plus one production run. Both probes are new apps in this
session, kept in the tree for reproducibility.

**Stream probe (`strata-numa-probe`)** — the placement lever in isolation,
13.4 MB slabs, unbound staging first-touch, addressed 28-lane stream:

| arm | 1.28 GB | 64 GB |
|---|---:|---:|
| bind (`expert % nodes`, MPOL_BIND) | 56.7 GB/s | 60.9 GB/s |
| no-bind (default first-touch) | 22.9 GB/s | — |
| interleave | 29.4 GB/s | — |

Pages land even on both nodes under bind (N0=164112 / N1=163568 pages). The
placement mechanism works; scale to 64 GB does not collapse it; madvise huge
pages do not materialize on this host (AnonHugePages stays 0, THP broken).

**Kernel probe (`strata-dsv4-host-expert-probe`)** — the *exact* `host_moe()`
dispatch (per-layer node split, `run_rows` row partition, addressed pool, the
real FP4 kernels, the E4M3 input/scratch quantize, output combine) over a
synthetic 100-expert arena, 43 layers, 6 routed experts per layer:

| arm | ms/token | GB/s |
|---|---:|---:|
| bind | 262.5 | 13.1 |
| no-bind | 319.0 | 10.8 |
| interleave | 294.3 | 11.7 |
| bind, 1 layer (warm) | 4.62 | 17.4 |
| bind, 8 layers | 40.8 | 15.7 |

The kernel probe reproduces production almost exactly (273 vs 294.6 ms/step).
**Placement is not the limiter**: bind→no-bind moves the kernel only 1.21x
(13.1→10.8), though it moves a pure stream 2.47x. The kernel is latency/uop
bound, not memory bound, so it does not ride the placement lever. Even warm
(6 experts resident), it reaches only 17.4 GB/s — never the 26.23 GB/s
standalone constant, which was measured at a more favourable operating point.

**Production run** (`--host-experts`, 586-token prompt, 63 decode steps):

| metric | value |
|---|---:|
| decode | 28.30 s / 63 = 449.2 ms/step |
| host MoE (decode phase) | 18.56 s / 63 = 294.6 ms/step |
| in-situ host MoE rate | 16254 experts × 13.37 MB × 63 / 18.56 s ≈ 11.7 GB/s |
| attention | 5.95 s / 63 = 94.4 ms/step |

## Why the gate reads negative

- The host path's in-situ rate is **11.7-13.1 GB/s** against a **32.2 GB/s**
  break-even. The kernel itself is the wall; no placement, caching, or
  dispatch change on this branch reaches the break-even.
- The 26.23 GB/s standalone constant does not transfer to the cold 148 GB
  arena: it was a warm/harness operating point. Reusing it is exactly the
  "reuse a measured constant across operating points" error the charter
  forbids; the kernel probe is the honest constant, and it halves the 0054
  arithmetic (which already predicted a best case of 0.939x).
- `τ` for the host path is `max_r`: at 13.1 GB/s the DRAM read term is
  ~265 ms/step — alone larger than the entire device-path MoE term (111.59 ms)
  it was meant to replace. The mechanism reduces neither the PCIe term it
  targets (it removes it) nor the new argmax it creates (host DRAM read at a
  kernel-limited rate), and it inflates compute and host serial work while
  doing so.

## Rejection, not rescue

The static host-expert placement hypothesis is recorded as rejected. It is
binding at the operating point that matters: 450.5 vs 222.58 ms/step. The NUMA
placement code itself is harmless and is worth keeping for the stream-rate
behaviour it gives the arena (56-61 GB/s), but it does not justify the
`host_experts` wiring, which stays off and unmerged.

The only path that could make host experts competitive is lk_moe's design —
groupN/groupK row batching to ~35 GB/s plus decode-time overlap so the weight
read hides under other work — which is a different kernel and scheduler, gated
by 0054's arithmetic and out of scope for this branch.

## Skill boundary

The branch's remaining objective — matching the external stack's 98 ms/step —
is above the current investigation's reach. The probes here closed the
host-experts question cheaply and decisively, but pursuing the gap further
(the FP4 GEMV kernel at 44.6 GB/s of 936, the ~407 blocking matmul
submissions per token, the CUDA-graph decode step named in 0057, or a host
row-batched kernel) requires work beyond this agent's current competence to
drive reliably. **A better agent with deeper GPU-kernel and CUDA-graph
experience will be necessary to pursue it.** The branch state, probes, and
this record are the handoff: the mechanism, its measured constants, and the
reason it fails are all reproducible from this document.

## Artifacts

- `apps/strata_numa_probe.cpp` — placement/stream probe (new, uncommitted).
- `apps/strata_dsv4_host_expert_probe.cpp` — exact-host_moe kernel probe (new,
  uncommitted).
- `scripts/run_deepseek_v4_host_experts_numa.sh` — production run driver with
  numa_maps sampling (new, uncommitted).
- `results/dsv4-host-experts-numa/run.json` — production run (ignored).
- Uncommitted branch work: `--host-experts` wiring in
  `src/deepseek_runtime.cpp` / `apps/strata_deepseek_run.cpp`,
  `src/numa_topology.cpp` + `include/strata/numa_topology.hpp`,
  `src/deepseek_checkpoint.cpp` stage() placement, `src/worker_pool.cpp`
  dual-queue dispatch. Failed runtime code must not be merged.
