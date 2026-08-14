# DeepSeek V4 Stage 5R production-chain rank-local MoE — 2026-08-11

Status: **REJECTED / REVIEW REQUIRED** at the Stage 5R boundary. Stage 6 is blocked and was not started.

Branch: `exp/dsv4-rank-local-moe-r2`
Parent/result baseline: `7e242519d937c4d2e6d655b819989d6d139f5ecc`
Operating point: `CUDA_VISIBLE_DEVICES=1,2`, runtime devices 0/1, two RTX 3090s, no CUDA P2P, batch-one actual replay, position 104, all 43 layers.

## Decision and hypothesis

Experiment 0077 remains an exact-scope rejection. Its isolated probe measured about 6.7 GB/s routed CPU work, used per-fixture condition-variable collection, and bridged shared output through the host; it did not reproduce the accepted production scheduler's roughly 40.6 GB/s operating point. This is a new, predeclared Stage 5R measurement; it does not rename, overwrite, or rescue 0077.

The Stage 5R hypothesis was:

> A 43-layer actual-replay chain using the accepted production CPU scheduler and stream-ordered device-resident rank-local outputs can preserve production routed-CPU behavior and exact rank-local ownership with no material complete-chain regression. Candidate incremental non-body/serial cost must be at most 2.125 ms per 43-layer forward.

The target term was the centralized dependent MoE handoff, not isolated GPU throughput. The governing model was instantiated at this scope as `tau = max_r(W_r/B_r) + Sigma_serial`.

Predeclared gates were: exact routes, coefficients, captured CPU partials, control outputs, rank-local BF16 join association, FP32 NCCL data and U32 MAX status reduction, BF16 publication, and failure closure; aggregate control routed CPU at least 36.7 GB/s; 43 reductions with one final completion and no per-layer host collection; zero measured checkpoint reads, timed CUDA workspace allocations, and measured-path host allocations; memory below 21,287,272,448 B/GPU and 231,928,233,984 B host; three interleaved complete repetitions after one warmup; no material candidate regression; and candidate incremental residual at most 2.125 ms.

The cheaper rejected experiment was the previous isolated 0077 probe. A full model load was not used: one replay position supplies all 43 real MoE layers, with fixed setup and one short measured chain per arm. The final arm used one warmup and three measured control/candidate chains.

## Corrected control prerequisite

The control uses `Dsv4ResidentWeightStore::stage(..., 48, false, true)`, the production `Dsv4HostMoeExecutor` extraction, its 1 ms hot-worker policy, same-node addressed dispatch/stealing, transformed tiled arenas, and setup-time fixed callback arenas. It queues all 43 host-MoE commands and calls one final status completion; it does not collect output per layer. Control exactness passed for all 43 layers, including routes and coefficients validated against the actual checkpoint router and captured accepted outputs/partials.

The first control attempt was deliberately preserved as a harness/build arm: an unoptimized `build-stage5` measured 6.603181 GB/s and failed the 36.7 GB/s control gate. Rebuilding the same source as Release produced:

```
wall_ms=79.411842
cpu_ms=77.045958
routed_gbps=44.769263
exact_outputs=43 exact_partials=43 exact_routes=43
one_final_completion=yes per_layer_collection=no
```

This passed the corrected-control prerequisite. The final three-repetition control arm below is binding.

## Candidate implementation

The candidate uses the smallest device-resident path permitted by Stage 5R:

- two independent 24-worker rank-local executors using the production addressed/stealing policy and the NUMA node owning each resident logical tiled shard;
- actual FP4 routed weights from the resident tiled store;
- rank-sharded shared FP8 triplets (`w1/w3` contiguous rows and `w2` strided columns) on the two GPUs;
- backend-owned stream-ordered hidden and rank-partial H2D staging;
- backend local exact BF16 routed/shared join;
- FP32 NCCL data all-reduce and U32 MAX status all-reduce on the backend stream;
- device-resident publication to fixed BF16 buffers; and
- one final status completion per rank forming one logical chain boundary.

The candidate does not change precision, arithmetic, routes, expert count, top-k, coefficients, worker math, or workload. The candidate rank-local oracle is intentionally distinct from centralized full-shared rounding: each rank rounds its local CPU/shared sum to BF16, NCCL sums the two FP32 rank outputs, and publication rounds to BF16. Control remains the oracle for the captured centralized output. A preserved early candidate arm differed from that centralized output by 1–2 BF16 ULPs; it was rejected as the wrong oracle scope, while the rank-local association oracle then passed all fixtures.

## Binding complete-chain result

Binding artifact:

```
results/dsv4-rank-local-moe-r2/stage5r-chain-binding2/control.log
sha256 e7ca73545b69881fad71fb3448bffaece3ed8cbfa43f607b9ea8e36d370b5ead
```

It uses one warmup followed by three interleaved complete 43-layer chains:

| arm | median ms/43 layers | range ms/43 layers |
|---|---:|---:|
| corrected control | **81.212542** | 79.027202–82.281586 |
| rank-local candidate | **77.172562** | 76.362031–78.061712 |

The candidate is faster by 4.039980 ms (4.97458%) at this MoE-only scope and does not regress outside observed control variance. That is not sufficient for acceptance: the candidate serial/non-body residual is too large.

Exact and failure gates:

```
control:   exact_outputs=43 exact_partials=43 exact_routes=43
candidate: exact_outputs=43 exact_partials=43 exact_routes=43 failure_cases=4
failure_closure=pass
```

The four failure cases are pre- and post-collective injection on both ranks. Both ranks participate in every NCCL collective; publication is withheld and zeros are returned after failure. No non-finite output escaped.

## Cost model and resource signs

Values are per complete 43-layer MoE forward, not per-rank summed wall time. The CPU body is a concurrent aggregate resource; candidate CPU body uses the maximum rank body for the critical path.

| resource/term | control | candidate | sign and interpretation |
|---|---:|---:|---|
| routed payload `W_CPU` | 3,449,290,752 B | same | workload and arithmetic held constant |
| CPU routed body `W_CPU/B_CPU` | 77.899226 ms, 44.278883 GB/s | 72.299077 ms max rank body, 47.7086 GB/s aggregate equivalent | candidate reduces body, but this remains `argmax` |
| isolated shared GPU envelope | 5.735872 ms max rank | 5.735872 ms max rank | non-binding; overlapped GPU work is not the target |
| `Sigma_serial` residual | 3.313316 ms | **4.873485 ms** | candidate adds 1.560169 ms at this scope |
| complete wall | 81.212542 ms | 77.172562 ms | no material regression, but residual gate fails |
| candidate FP32 NCCL data | — | 1,409,024 logical B/chain | added communication; physical SHM traffic unmeasured |
| candidate U32 status | — | 344 logical B/chain | required fail-closed collective; added control work |
| application H2D summary | 704,512 B/chain | 2,818,048 B/chain | candidate adds rank-local hidden/partial staging |
| final status D2H | 4 B/chain | 8 B/chain | one status boundary per active rank; no output D2H |

Thus:

```
control tau = max(77.899226 CPU, 5.735872 GPU, transfer terms)
              + 3.313316 serial = 81.212542 ms

candidate tau = max(72.299077 CPU, 5.735872 GPU, transfer terms)
                + 4.873485 serial = 77.172562 ms
```

The CPU routed body is the argmax in both arms. The candidate reduces that body by using half-intermediate rank shards, but adds stream/NCCL/join/status serial work. The candidate residual is **4.873485 ms > 2.125 ms**, so the amended Stage 5R gate is binding-negative. The accepted Stage-2 43-reduction FP32 NCCL term of about 1.427 ms is inside this allowance; it is not stacked as an additional independent saving. No physical NCCL SHM byte count is claimed.

## Memory, I/O, and allocation evidence

The binding output reports:

```
rss_bytes=158621954048
resident_bytes=156885843968
host_ceiling_bytes=231928233984
gpu0_used_bytes=2062680064
gpu1_used_bytes=837877760
vram_ceiling_bytes=21287272448
measured_checkpoint_calls=0 measured_checkpoint_bytes=0
measured_cuda_workspace_alloc_calls=0 measured_cuda_workspace_alloc_bytes=0
measured_host_allocations=0
```

All memory and I/O gates pass. Setup staging is not charged to the measured chain; its checkpoint accounting is retained in the log. Candidate logical NCCL physical transport remains uninstrumented rather than being reported as zero.

## Preserved failed and intermediate arms

No raw arm was deleted or overwritten:

| artifact | SHA256 | disposition |
|---|---|---|
| `stage5r-control-release-r1/control.log` | `05726cd9d28a357b21a2342dde80e08925e7da02127c0211e3aa63a0f6bf45ee` | corrected-control prerequisite passed |
| `stage5r-chain-r1/control.log` | `5c39ad92d22138af19332223e57ee63d9e143d26aedd36ef32337b7303e60de0` | unoptimized/old candidate harness failure |
| `stage5r-chain-debug-r2/control.log` | `a69b5b11305b7ed06cb9f7c8aee617f1781e585582c9493022b8b43aef57481c` | centralized-output oracle failure, 1–2 BF16 ULP scope mismatch |
| `stage5r-chain-oracle-r3/control.log` | `df624ed821fb620448bb8375ac0e98b54b4128b29398045165f82c2a5991dd4d` | remote-NUMA candidate arm, rejected placement |
| `stage5r-chain-r4/control.log` | `51c068e4e4bf9cc0bb6f1f3a6c7d1cecfea456765c7940b9b43d911d4d693100` | NUMA-local arm before fixed arenas |
| `stage5r-chain-final/control.log` | `89380ceed39cfc936afca42336593361ca673458cf5981a862ffb95a9a40f0d5` | `.back()` callback-pointer defect |
| `stage5r-chain-fixed-debug/control.log` | `296882bb70cb0eb26d68a50825343f2f70cb9c698891951cb812efe679030198` | fixed-array pointer defect |
| `stage5r-chain-fixed-debug2/control.log` | `6adf90bacb884465ff76c4a663296eb8137b625ad02a0394929c9e5b71a271cb` | same preserved pointer-defect diagnosis |
| `stage5r-chain-binding/control.log` | `0470cd6aacbb553a7edada9c3fed6dffeadfe80b42b9f41f3fb39def9e5223f3` | valid fixed-arena arm, superseded by binding2 |

Experiment 0077 and all of its negative/intermediate capture/probe arms remain under `results/dsv4-rank-local-moe/` unchanged. The binding Stage 5R result is the final fixed-arena arm above; the earlier positive wall result does not erase its failed residual gate.

## Implementation and validation state

Tracked Stage 5R sources are:

```
apps/strata_dsv4_stage5r_chain.cu
include/strata/dsv4_host_moe_executor.hpp
src/dsv4_host_moe_executor.cpp
scripts/run_dsv4_stage5r_control.sh
include/strata/cuda_backend.hpp
kernels/cuda/backend.cu
src/cuda_backend_stub.cpp
include/strata/worker_pool.hpp
src/worker_pool.cpp
CMakeLists.txt
```

The CUDA target is `strata-dsv4-stage5r-chain`, built in Release `build-stage5` with the installed NCCL 2.28.9 CUDA 13 library. Existing upgraded Nsight Systems 2026.1.3 statistics were decoded read-only from `results/dsv4-rank-local-moe/stage5-profile-nsys20260811/trace.nsys-rep`; those reports describe the old 0077 trace and are not used to convert this Stage 5R result into a win. No Nsight Compute counters are claimed (`ERR_NVGPUCTRPERM` remains the prior result).

Final repository validation passed before the result commit:

```
make check
git diff --check
full diff review
```

## Disposition and review boundary

Stage 5R is **REJECTED / REVIEW REQUIRED** solely because its predeclared candidate incremental serial/non-body residual is 4.873485 ms, above 2.125 ms. Exactness, memory, I/O, allocation, control throughput, and no-material-wall-regression gates pass. This is technical completion of the Stage 5R experiment, not authorization to continue implementation.

No device-resident correction, CPU arithmetic optimization, graph capture, friendlier workload, mHC integration, attention integration, or Stage 6 work is justified after this negative gate. Stage 6 remains **BLOCKED / NOT AUTHORIZED** pending explicit human review. The roadmap is updated locally but remains intentionally ignored; raw results and generated binaries remain outside Git.
