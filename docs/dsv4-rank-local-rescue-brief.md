# Handover: rescue the DSV4 rank-local TP2 decode landing

> **Path note (2026-08-18).** This is a historical record and its file paths
> predate the layering refactor. `src/<name>.cpp` moved to `src/platform/`,
> `src/engine/`, `src/app/` or `src/models/<model>/`; see
> [`current-architecture.md`](current-architecture.md). Some paths here never
> existed on `main` at all — they are from experiment branches that were never
> merged. The record is left as written; only this note is added.


You are taking over a landing that produced a **2.8x performance regression where
none should exist**, on work that was supposed to be a transplantation of an
already-measured, already-accepted implementation. Read this whole brief before
touching code. The previous agent did not, and that is the root cause of
everything below.

---

## 1. The repository and its rules

`/home/rodrigo/Developer/strata` — a ground-up, dependency-light C/C++ inference
engine for dense and MoE models larger than local VRAM. Read `AGENTS.md` first;
it is a binding research charter, not style guidance. The rules that matter most
here:

- **Measure the bottleneck before choosing a mechanism.**
  `research/moe-tiered-memory-decode-optimization.md` is a decision procedure:
  `tau = max_r W_r/B_r + Sigma_serial`. Instantiate it before designing.
- **Separate volume from overlap.** A term that is large because it is serial is
  an overlap defect, not a volume problem.
- **Treat an implausible measurement as a defect, not a datapoint.**
- **Do not launder a falsification.**
- `docs/experiments/0025` is the worked example: ~2,900 lines of runtime code
  built and rejected because the cost model was never instantiated. Every rule
  in the charter derives from one of its errors.

Hard invariants: no weight precision below four bits anywhere; predictors are
advisory only; exact mode either completes exact work or reports failure, never
falls back silently; runtime code is C/C++ with no Python or framework runtime.

Hardware: 3 GPUs — device 0 is a 16 GiB RTX 5060 Ti, devices **1 and 2 are 24
GiB RTX 3090s** and are the only symmetric pair, so rank-local always runs
`--devices 1,2`. No CUDA P2P between them. Host has two NUMA nodes, 24 CPUs
each. Model is `models/dsv4f`, 156 GB on disk, DeepSeek-V4-Flash.

---

## 2. What the user asked for

The user's **Production Landing Plan: DSV4 Rank-Local TP2 Decode**, verbatim in
intent:

> Create `feat/dsv4-rank-local-decode` directly from `main` (currently
> `61f1f02`). Keep `exp/dsv4-a2-ownership-screen` and commit `a31ac58`
> unchanged as the reproducible experiment record.
>
> The objective is to extract only the accepted rank-local mechanisms,
> integrate them into real generation, and prove at least 8.5 tok/s on the
> established workload. **Do not merge the 42-commit experiment branch
> wholesale.**
>
> The feature is explicit opt-in and fail-closed. Centralized decode remains
> the default.

Six stages: (1) extract primitives, (2) weight ownership + admission, (3) live
replicated transactional KV/state, (4) wire real generation, (5) correctness and
failure tests, (6) performance gate + docs.

**Three binding addenda the user added later:**

1. **Provenance.** A tracked extraction manifest classifying every file and
   mechanism in `main...a31ac58` as ported / documentation / excluded-with-reason
   / deferred. Source experiment commits referenced in landing commit messages.
   One canonical architecture document. A final traceability audit — explicitly
   "a traceability gate, not permission to expand the production code scope".
2. **Thirteen capabilities.** Every capability in the reuse table must appear in
   the manifest and architecture doc with invariants, evidence, failure lesson
   and reuse guidance. The audit must account for all thirteen.
3. **1M context is not negotiable.** The user's words: *"1M context is not a
   'nice to have'. It is a fundamental requirement... if we reach 8 tok/s, but
   with a 256 of context, this is equivalent to garbage... So, 1M context AND 8
   tok/s are FUNDAMENTAL."*

**The gate was lowered by the user from 8.5 to 8.0 tok/s** (<= 125.0 ms/token)
after the 1M work showed the original target was not reachable at the declared
context without further optimization. That lowering is the user's decision and
is not to be revisited downward again.

The user's repeated, explicit framing, which the previous agent failed to honour:

> *"are you correctly using the correct reference from commit a31ac58? The code
> is already at commit a31ac58. This is more of a transplantation rather than
> inventing new code. Do you understand that? ... My only concern is that I am
> worried that you may be inventing code that we already created, and may
> introduce errors that we do not have at commit a31ac58."*

and later:

> *"we are over with the experiments. When I drew this plan, its purpose was not
> a new round of infinite experiments. We are done with experiments. I need to
> have rank-local at main. This is the most important. The performance tuning
> and new experiments will come after that."*

and finally, on seeing the regression:

> *"a regression of this magnitude is clearly an issue in the way that you are
> wiring all this. The proof is simple: in the reference commit a31ac58 the
> timing is MUCH BETTER than this... All the lessons are FULLY documented, you
> just have to check the last experiment documents to see the issues and how
> they were one by one fixed chronologically. docs/experiments"*

---

## 3. Where the 1M requirement came from, and what was built for it

Mid-plan the user raised an external reference implementation ("Lvllmds4-x")
that sustains **1M context at ~10 tok/s** on comparable hardware, and made 1M a
hard requirement alongside the throughput gate.

The DSV4 execution contract (`kDeepSeekV4ExecutionContract` in
`include/strata/model_adapter.hpp`): hidden 4096, 43 layers, 64 heads, head_dim
512, rope 64, sliding_window 128, index_heads 64, index_head_dim 128,
**index_topk 512**, 256 routed experts, top-k 6, vocab 129280, mhc_multiplier 4,
max_context 1,048,576.

`compression_ratios` = `{0,0,4,128,4,128,...,4,0,0,0}` (46 entries, 43 used):
**21 layers at ratio 4** (these run the sparse Lightning Indexer), **20 layers at
ratio 128** (dense compressed attention over `L/128` rows), **2 layers at ratio
0**. The sparse indexer engages only when
`compressor.ratio == 4 && active_context_tokens > index_topk * ratio` (= 2048).

Work done for 1M, and this part is probably sound:

- Established by **arithmetic, not a run**, that the host scalar indexer cannot
  serve 1M: 262,144 candidates x 64 heads x 128 dims across 21 ratio-4 layers is
  1.85e11 FLOP/token, a >=192 ms/token floor on any CPU budget here.
- Built `CudaBackend::dsv4_physical_lightning_index` in `kernels/cuda/backend.cu`
  — a device scoring + radix-select top-k over the physical E4M3 KV page format.
  It is **bit-exact against the scalar oracle**, verified position-for-position
  at the full 262,144-candidate width (`tests/test_cuda_backend.cpp`).
- Optimized 842 -> 66.3 ms/token in three measured steps. The first version was
  a defect: a `<<<1,1>>>` pivot kernel had replaced one serial top-k with
  another.
- Corrected an arithmetic error along the way: per-token indexer cost had been
  computed over 43 layers when only 21 are ratio-4, so every figure was 2x too
  high.

**Open at 1M:** the indexer alone is 66.3 ms of the 125 ms budget, and a
component-sum projection of `tau(1M)` came to 181.3 ms/token (5.5 tok/s), which
**misses the 8 tok/s gate**. That was recorded as a miss, not laundered. The
identified mitigation (deferred, not built) is sharding index scoring across the
two ranks — exact, because a global top-512 member is necessarily in its own
rank's local top-512, at the cost of an 8 KB exchange, projected ~33 ms.

`tau(1M)` has **never been measured end to end**, only summed from components.

---

## 4. The reference you must follow: `a31ac58`

Branch `exp/dsv4-a2-ownership-screen`, commit `a31ac58`, file
`apps/strata_dsv4_rank_local_layer.cu` (5,136 lines). Extract it with
`git show a31ac58:apps/strata_dsv4_rank_local_layer.cu`.

Its measured result, from `docs/experiments/0092`: **114.944312 ms/forward
median**, CPU critical path 73.896784 ms at 46.677143 GB/s, non-CPU 41.047528
ms, terminal head 2.288864 ms. This is *fixture scope* (submission through
terminal head, replaying `.d4c`/`.d4r`/`.d4m` fixtures) and must never be quoted
as end-to-end throughput — but it is the correct yardstick for the executor's
internal cost, and the current landing is **3.6x worse than it**.

The chronological experiment record — **read these, in order, before writing
code**:

| doc | what it settles |
|---|---|
| `0079-dsv4-rank-local-moe-residual-attribution` | The accepted operating point is **"one final completion boundary"**. Resource gates that must read **zero** include **"per-layer host collection"**. Stage 5R: candidate 77.17 ms, max rank CPU body 72.30 ms, `Sigma_serial = 4.873485 ms per 43 layers`. Also: CUDA events at every layer boundary **perturb** the measurement (83.08 event-off vs 87.89 event-on) and were rejected as causal evidence. |
| `0080-dsv4-rank-local-full-chain-falsifier` | full-chain falsifier |
| `0082-dsv4-explicit-vram-byte-admission` | Source of the **21,287,272,448 B/GPU** figure. This is the *observed peak of the centralized baseline*, recorded as a regression tripwire — **not** a hardware bound. Also the source of the 151.155686 ms/forward centralized control. |
| `0083-dsv4-stage6-nccl-memory-gate` | NCCL memory gate |
| `0085-dsv4-exact-rank-local-layer-executor` | the executor's exactness contract |
| `0086-dsv4-rank-local-adjacent-chain-rejection` | a rejected chain variant — read before proposing chain changes |
| `0087-dsv4-current-callback-gap-attribution` | Centralized control **151.155686 ms/forward** = 84.710043 ms routed CPU body + 66.445643 ms makespan residual. |
| `0088-dsv4-rank-local-43-layer-correctness` | 43-layer correctness across all layers |
| `0089-dsv4-rank-local-terminal-publication` | terminal publication contract |
| `0090-dsv4-m3-device-only-admission-rejection` | a rejection of a device-only variant |
| `0091-dsv4-m3-callback-free-staging` | **The accepted timed configuration is the callback-free queued chain.** Also documents a real defect: `dsv4_prepare_attention` chose its pinned upload buffer on `host_callback != nullptr` rather than on whether the command was queued, so once the host outran the stream, queued layers read later layers' norms. Fixed by fixed per-command staging, 43 slots x 16 KiB per device. |
| `0092-dsv4-m3-timing-falsifier` | The 114.944312 ms result and its gates. |

Key functions in `a31ac58:apps/strata_dsv4_rank_local_layer.cu`:

- `prepare_a2_arm():1839` — assembles one `Dsv4RankLocalLayerCall`. Field order,
  weight order, and the cosine/inverse-sine convention are all here.
- `seed_m3_layer0():2875` — seeds mHC per rank.
- `m3_head_request():2421` — the terminal head request.
- `m3_head_callback():2334` — the head's host reduction.
- `run_m3_sequential():3004` — **the per-layer verification control.** `run()`
  for layers 0..41, then `enqueue_chain_layer` + `finish_chain` for the terminal.
- `run_m3_candidate():3062` — **the timed configuration.** `enqueue_chain_layer`
  for all 43 layers, then one `finish_chain`.
- `configure_live_page_patches():876` and `upload_pages():948` — the page-patch
  callback path.
- `load_all_weights():861`, `load_mhc():689`.

**The previous agent transplanted `run_m3_sequential` and believed it was the
production shape. It is not. It is the control arm.** That single mistake is
most of the regression.

---

## 5. What was built, and what is now suspect

Branch `feat/dsv4-rank-local-decode`, 22 commits ahead of `main`, ~16,900
insertions across 60 files. `make check` is green on both the default `build/`
and `build-landing-nccl/`. The traceability audit passes (91 paths, 13/13
capabilities). **None of that means the design is right.**

| stage | what exists | confidence |
|---|---|---|
| 1 — primitives | `dsv4_rank_local_topology`, `deepseek_rank_shard`, `dsv4_host_moe_executor`, `dsv4_rank_local_layer_executor.cu` lifted from the experiment | probably sound; largely file-level copies |
| 2 — ownership + admission | `Dsv4RankLocalWeightStore`, `admit_dsv4_rank_local` | **suspect** — see defects below |
| 3 — transactional KV | `Dsv4RankLocalKvTransaction` (reserve once, peer device leases, `commit_layer` encodes once and copies, `page_writes` per rank), driven by host-side `update_buffer` | **architecturally suspect** — `a31ac58` commits KV rows *inside* the chain through the executor's page-patch callbacks with fixed per-command staging. The landing invented a host-side transport instead. |
| 4 — real generation | `rank_local_forward_hidden` + `rank_local_prepare_layer` in `src/deepseek_runtime.cpp` | **known wrong** — per-layer `run()`; must become the queued chain |
| 5 — tests | `tests/test_dsv4_rank_local_kv.cpp` (8 cases incl. a device-page transport test), topology tests | tests the invented design, so they may be validating the wrong thing |
| 6 — perf gate + docs | not started | — |
| 1M indexer | `dsv4_physical_lightning_index`, `apps/strata_dsv4_index_probe.cpp` | probably sound, bit-exact, independently measured |

### The current measured state

18-token prompt, 4,096-token context, devices 1,2, `--device-resident-runtime`:

```
centralized   7 decode steps in 1.035 s   148 ms/step   coherent output
rank-local    7 decode steps in 2.930 s   418 ms/step   identical output text
a31ac58 (fixture scope)                   114.94 ms/forward
```

Rank-local **produces correct output** — token-for-token identical to
centralized — so the arithmetic, the collectives, the replicated KV, the two
replica-divergence gates and the row-sharded head all work. It is purely a
performance catastrophe.

Attribution per step (from `--detailed-timing --json`, `phases.decode.graph`):

```
executor total                    363.9
  router->moe span                287.1   <-- 79% of the step
  routed CPU body + collective     98.9
  attention                        35.5
  attention/moe collectives        18.9
  per-layer diagnostic boundary    11.4
  transition                        3.6
host-visible preparation           31.7
KV transaction + page patches       2.7
candidate/page construction         0.5
```

The landing's own glue is ~35 ms of a ~270 ms gap. The `router->moe` span bounds
the shared-expert kernels **plus** the host routed-MoE callback and its device
join; the CPU body inside it self-reports 99 ms, so **~190 ms/step is waiting**.

**Diagnosis:** per-layer driving does not merely cost the boundary's own 11.4 ms
— it destroys the overlap the design depends on. `0079` records
`Sigma_serial = 4.873485 ms across all 43 layers` because the routed CPU body is
concurrent with GPU work and with other layers' scheduling. Driven one layer at
a time those terms leave the `max` and add, 43 times over.

---

## 6. Open defects, including ones not yet investigated

1. **Stage 4 is the wrong arm.** Must become `enqueue_chain_layer` x 43 +
   one `finish_chain`, per `0079`/`0091`/`0092`.
2. **VRAM accounting disagrees with reality by ~1.4 GB.** Admission approved a
   22,548,578,304 B/device ceiling; the run reports
   `device_vram_used_bytes: [23933878272, 23931781120]` — **23.93 GB/device**.
   Either the byte model is incomplete or the ceiling is meaningless. No memory
   claim in the landing docs is trustworthy until this is resolved.
3. **The ceiling raise may rest on an aggregate.** It was raised from `0082`'s
   21,287,272,448 B to 22,548,578,304 B, reasoned from a 9,204,991,520 B
   "centralized prefill spine" figure. `Dsv4MemoryPlan::resident_spine_vram_bytes`
   is an **aggregate across devices** (admission sums it against the aggregate
   VRAM budget), so that reasoning may be wrong by the device count. A related
   bug was already fixed — `admit_rank_local` was charging each rank the
   aggregate spine and expert-cache figures, double-counting on a two-device
   topology — but the *ceiling justification itself* was never re-derived.
4. **`Dsv4PhaseMetrics::graph` is a positional aggregate initializer.** ~30
   unnamed `after.x - before.x` expressions. Any field inserted mid-struct in
   `Dsv4GraphStats` silently shifts every later member into the wrong slot — no
   compiler error, all `uint64_t`. This already produced one wrong measurement
   during this work. Consequence found and left as-is: `future_entropy_*` has
   been silently zero in every phase report because the initializer is shorter
   than the struct.
5. **`timing.total_ms` includes the per-layer diagnostic boundary**, so it
   cannot be differenced against a caller's wall clock to isolate it. This also
   produced a wrong intermediate conclusion.
6. **`0079` warns that per-layer CUDA events perturb** (83.08 event-off vs 87.89
   event-on). The instrumentation added during this work has not been checked
   against that gate.
7. `tau(1M)` never measured end to end; only component-summed to 181.3 ms.
8. `physical_paged_attention`'s host `locate()` is `O(candidates x blocks)`;
   never measured. At 1M the 20 ratio-128 layers attend 8,192 rows each.
9. The sparse indexer path has **never run inside a real decode** — every
   end-to-end run so far was below the 2,048-token threshold.
10. Releasing the centralized prefill spine after prefill would return ~9.2 GB
    versus the 0.76 GiB the ceiling raise bought. Not done.

---

## 7. New mechanisms the previous agent invented, which you should re-derive from `a31ac58` rather than trust

The experiment replayed `.d4r` fixtures whose pages already contained compressed
and learned-index rows, and whose live page patch rewrote **only the sliding
row**. So three things genuinely do not exist in `a31ac58`, and the previous
agent built them:

- **Compressor weights.** Neither `RankWeights` in the experiment nor
  `Dsv4RankLocalRankLayerWeights` in the landing carries `compressor.wkv/wgate`
  or `indexer.compressor.wkv/wgate`, and the executor's preparation sets no
  compressor pointers. The landing worked around this with a **separate
  host-visible `dsv4_prepare_attention` per layer** on the slot owning the
  centralized compressor weights, which costs 31.7 ms/step and required a new
  `request.host_only` backend flag (the complement of `device_only`) so the
  first preparation would not publish a prepared query the executor's own
  preparation would then reject as out of order.
  **The correct fix is almost certainly to add the compressor weights to the
  rank-local weight set and to the executor's preparation request, so the
  page-patch callback receives `view.compressor_values` and commits the row
  inside the chain — which is exactly what the centralized queued path already
  does via `complete_physical_attention_prepare`.**
- **`compress_state` pooled-row output.** An optional out-parameter added to
  return the pooled row instead of publishing it.
- **Host-side page transport.** `Dsv4RankLocalKvTransaction::commit_layer` +
  `page_writes` + `CudaBackend::update_buffer` per rank, plus a new
  `CudaBackend::download_buffer` added purely to make the transport testable.
  `a31ac58` instead patches pages through the executor's page-patch callbacks
  with fixed per-command staging.

Also added and worth reviewing: the `--decode-topology centralized|rank-local-tp2`
CLI opt-in, and a fix making the per-device output-head reservation a *capacity*
rather than a *shape* (centralized prefill projects the full 129,280-row
vocabulary while rank-local decode projects a 64,640-row shard, and both are
resident on one device under the opt-in).

---

## 8. The previous agent's modus operandi, and why it produced this

State this plainly so you do not repeat it:

- **It treated a transplantation as a design exercise.** The user said three
  times that this was transplantation. The agent read `a31ac58` for *shapes* —
  function signatures, struct field order, calling conventions — and then
  designed its own integration around them, instead of reproducing the accepted
  configuration.
- **It read the experiment source but not the experiment record.** It opened
  `apps/strata_dsv4_rank_local_layer.cu` many times and never opened
  `docs/experiments/0079`–`0092` until after the regression appeared. Those
  documents state the accepted operating point, the zero-gates, and the measured
  numbers, in order, chronologically. The single sentence that would have
  prevented the regression — *"one final completion boundary"*, and *"per-layer
  host collection"* among the gates that must read zero — was in the first one.
- **It copied the control arm instead of the timed arm.** `run_m3_sequential`
  and `run_m3_candidate` sit 60 lines apart in the same file.
- **It reported hypotheses with the confidence of measurements.** It asserted
  the per-layer diagnostic boundary was the cause, from a plausible mechanism
  (counting 84 syncs and ~1,100 blocking copies per token), before measuring it.
  The boundary is 11.4 ms.
- **It misread its own instruments twice** and reported both wrong intermediate
  conclusions to the user before checking — the positional aggregate initializer,
  and `timing.total_ms` already containing the boundary.
- **It optimized before the base path ran.** The 1M device indexer was designed,
  built, and optimized through three measured iterations before a single
  rank-local token had ever been produced.

The charter names this failure mode directly, and `docs/experiments/0025` is a
2,900-line worked example of it. The agent had a memory note titled *"Read the
record before deriving"* and did not.

---

## 9. Your task

**Goal:** land rank-local TP2 decode on `main`, opt-in and fail-closed, serving
the model's full 1,048,576-token declared context, at **>= 8.0 tok/s
(<= 125.0 ms/token)**. Centralized decode stays the default and must be
bit-unchanged.

**Method: transplant, do not design.** For every mechanism, first establish what
`a31ac58` does and what `docs/experiments/0079`–`0092` accepted or rejected.
Only build new code where the experiment genuinely has none — and where it does
not, say so explicitly and justify the new mechanism against the cost model.

**Order of work:**

1. **Read `docs/experiments/0079` through `0092` in order**, plus `CLAUDE.md`
   and `research/moe-tiered-memory-decode-optimization.md`. Then read
   `run_m3_candidate()` at `a31ac58:apps/strata_dsv4_rank_local_layer.cu:3062`
   and diff its structure against
   `DeepSeekV4Runtime::Impl::rank_local_forward_hidden` in
   `src/deepseek_runtime.cpp`.
2. **Audit stages 1–3 against the reference before extending stage 4.** The user
   believes they may be compromised. In particular decide whether
   `Dsv4RankLocalKvTransaction`'s host-side `update_buffer` transport should be
   replaced by the executor's page-patch callbacks, and whether the compressor
   weights belong in the rank-local weight set. If a stage is sound, say why; if
   not, say what the reference does instead.
3. **Rebuild stage 4 as the queued chain.** Two known blockers, both real:
   candidates must be known before the first `enqueue_chain_layer` (trivially
   true below 2,048 active tokens, where selection is `iota` over <= 512
   compressed rows; genuinely hard above it, and 1M is above it), and KV rows
   must be committed inside the chain because layer L's row depends on layer
   L-1's output. Honour `0091`'s fixed per-command staging warning.
4. **Re-measure.** The centralized control is 148–151 ms/step at this operating
   point; `a31ac58` is 114.94 ms/forward at fixture scope. Anything above ~150
   ms/step means the chain is still not overlapping.
5. **Then, and only then**, address 1M: the sparse-selection-inside-the-chain
   problem, the rank-sharded index scoring (~66.3 -> ~33 ms), and a measured
   `tau(1M)`.
6. Resolve the VRAM accounting discrepancy (defect 2) before any memory claim
   ships.

**Reproduce the current state:**

```bash
cmake --build build-landing-nccl -j 8
./build-landing-nccl/strata-deepseek-run \
  --model models/dsv4f --prompt "what is the closes star to us, and how far is it?" \
  --max-new 8 --devices 1,2 --max-context 4096 --host-memory 216G \
  --vram-fraction 0.95 --device-resident-runtime \
  --decode-topology rank-local-tp2 --detailed-timing --json
```

Swap `--decode-topology centralized` for the control. `scripts/run_dsv4_rank_local_smoke.sh`
runs both. Long runs belong in named tmux sessions with results in ignored
deterministic paths.

**Gates before any result commit:** `make check` green on both build
directories; `scripts/audit_dsv4_extraction_manifest.sh` PASS with 13/13
capabilities; `docs/dsv4-rank-local-extraction-manifest.md` and
`docs/dsv4-rank-local-architecture.md` updated to describe what actually runs.

**What not to do:** do not merge the 42-commit experiment branch wholesale; do
not rewrite `exp/dsv4-a2-ownership-screen` or `a31ac58`; do not let rank-local
fall back to centralized silently — a refused opt-in must report; do not quote
the 114.944312 ms fixture figure as end-to-end throughput; do not quote
ms/step figures measured at an 18-token prompt as if they described the 1M
operating point.
