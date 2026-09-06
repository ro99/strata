# GLM-5.3 long-context prefill — handover brief

Written 2026-09-06 for an engineer or agent picking this up cold. It assumes no
prior context. Everything needed to resume is here or is named by path.

Repo state at handover: `main` = `e97d387`, pushed. Working tree clean.
Companion reference: `docs/models/glm53.md`, section
"Long-context prefill: measured state and the open lever".

---

## 1. The system, in one page

**Strata** is a dependency-light C/C++ inference engine for MoE models larger
than VRAM. It keeps the non-expert spine on GPU and streams routed experts from
host memory.

**GLM-5.3-Flash** here: 45 layers — 34 KDA (linear-attention) + 11 sparse **MLA**
layers — 42 MoE layers, 288 routed experts, top-8 plus one shared expert.
Attention shape constants (`src/models/glm53/glm53_runtime.cpp`):

| constant | value | meaning |
| --- | --- | --- |
| `kHeads` | 64 | attention heads |
| `kMlaHead` | 256 | per-head width |
| `kMlaWidth` | 16384 | `kHeads * kMlaHead` |
| `kKvRank` | 512 | latent KV rank (what the cache stores) |
| `kHidden` | 4096 | residual width |
| `kQueryRank` | 1536 | q_a rank |
| `kIndexTopK` (:1077) | 2048 | **the regime divider** |
| `kSparseExpansionRows` (:1095) | 4096 | per-group expansion cap |
| `kPageExpansionBudgetRows` (:1103) | 8192 | per-page expansion cap (new) |

**Hardware on this box:** 2x RTX 3090 (SM86, 24 GiB) + 1x RTX 5060 Ti (SM120,
16 GiB), no peer link. The two 3090s share a root complex: ~11.98 GB/s to one
card, 13.10 GB/s aggregate — **host-to-device transfers do not scale across
cards.** GPUs are clock-locked at 1605 MHz.

> **Trap:** the shell exports `CUDA_DEVICE_ORDER=FASTEST_FIRST`. Always run with
> `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `--devices 1,2` for the two 3090s.
> Without it `--devices 1,2` silently becomes 5060 Ti + one 3090.

---

## 2. The objective and its status

**Goal:** fast prefill at context 32,768 with genuinely long prompts. The
premise, from the user, is that on DeepSeek in this same machine longer context
makes prefill *faster*; GLM getting slower means the design is wrong somewhere.

**Status: not achieved.** Fitted marginal cost is ~171 ms/token against a
reference stack (`~/Developer/Lvllmds4-x`) at **1.14 ms/token**. What the
campaign did produce is a correct diagnosis of *why the curve bends upward*, the
fix for it half-built, and an instrument that makes the next attempt cheap.

**The campaign metric is the fitted marginal (ms/token), not tok/s.**

---

## 3. The single most important fact: the regime split

Prefill has two regimes divided by `kIndexTopK` (2,048).

- **Below 2,048** the k-pool sparse indexer's selection is the identity and the
  dense page path runs.
- **Above 2,048** the sparse path runs and the cost structure is completely
  different.

An 8,192-token arm is ~20 minutes; a 32,768-token arm is hours. So most of this
campaign's arms were run at 619 and 2,591 tokens — **the regime users do not
run** — and two accepted conclusions turned out to be artifacts of that and were
withdrawn. The shares invert with length:

| term | at 2,591 tokens | at 8,192 tokens |
| --- | --- | --- |
| MLA (attention) | 26% | **64%**, accelerating (67.2 -> 124.9 ms/token) |
| feed-forward (experts) | 58% | 26%, saturating (83.8 -> 30.8 ms/token) |

**Do not accept any prefill result measured below 2,048 tokens.**

---

## 4. Where the time actually goes

The sparse attend core lives in `attention_mla_page_sparse`
(`src/models/glm53/glm53_runtime.cpp:7274`). Per layer-page it:

1. projects q_b and the two indexer projections (`linear_batch`);
2. selects, per row, ~2,051 history positions via the k-pool indexer;
3. **groups** rows until their combined selection would exceed
   `kSparseExpansionRows`, then for each group expands that group's union of
   latent rows through `kv_b_proj` (`linear_batch`);
4. runs QK, softmax (host, for exactness), and AV per row;
5. `o_proj` back to `kHidden`.

**The defect is step 3.** Below ~4,000 tokens of history one group covers a
whole page. Above it the page shatters, and every group re-runs `kv_b_proj` over
close to the 4,096-row cap to serve a couple of dozen query rows. Measured at
8,192 tokens (11 MLA layers x 3 sparse pages):

| page (`history_begin`) | groups | row-expansions |
| --- | --- | --- |
| 2048 | 11 | 44,701 |
| 4096 | 194 | 780,201 |
| 6144 | 699 | 2,771,835 |
| **total** | **904** | **3,596,737** |

Against 28,501 at 2,591 tokens: **126x the expansion work for 3.16x the
tokens.** Fragmentation is layer-dependent — layers 3/7/11 never split, 19/23
are worst — tracking how concentrated each layer's indexer is.

One expanded row is `kKvRank * kHeads*2*kMlaHead` = 512 x 32,768 = **16.78
MMAC**, i.e. 128 KiB of output. 3.60M rows = 60.4 TMAC against a measured 135.56 s
of expansion, so the effective rate is **~446 GMAC/s**. Use that number for
arithmetic screening.

**Budget of the 1,229 s at 8,192 tokens:** MLA 785.6 s, FF 317.7 s. Inside the
MLA group loop (715.8 s wall): expansion 135.6 s serial, QK+AV ~6,522 s of CPU
over ~21 effective threads = ~311 s wall, leaving **~270 s of unexplained
per-group serial overhead** (one `linear_batch` round trip per group).

---

## 5. The exactness contract — read before touching anything

This branch's whole value is that every optimization is **bit-identical**. The
rules that make that true:

- The host translation unit is built **without FMA**, so `score += a*b` is
  `mulss` + `addss` with double rounding. Device kernels must use
  `__fmul_rn`/`__fadd_rn` and **never** fused `fmaf`.
- Softmax stays on the host: host libm and device trig differ in the last ulp.
  Coefficients are rounded to BF16 *before* they multiply anything.
- **Row-parallelism is safe; reordering within a row is not.** Parallelize
  independent rows (`parallel_page_rows`, :6432) while keeping each row's
  internal operation order — that is byte-identical and already paid 3.06x once.
- **Tile ROWS, never the union.** A row's softmax reduction must stay whole.
- Grouping only decides *which rows share an expansion*, never any within-row
  order — so regrouping is exactness-neutral by construction.
- The gate is a per-layer FNV hash of the MLA latent cache,
  `print_mla_layer_hashes` (:10447), emitted as `[glm53-prefill-mla-layer]`.
  Two arms must produce identical hashes.
- `kv_b_proj` expansion is **batch-invariant** — verified, see §7.

---

## 6. Build, run, measure

```bash
cd /home/rodrigo/Developer/strata
cmake -S . -B build && cmake --build build -j 16      # Release; verify this!
./build/strata-tests                                  # gate: 378 pass, 38 skip
```

> **Trap:** a Debug `build/` once cost 45 minutes of A/B runs and two wrong
> conclusions. Confirm `CMAKE_BUILD_TYPE=Release` in `build/CMakeCache.txt`.

**The fast instrument** — one layer-page of the real sparse path at any history,
in seconds instead of the hours a full arm costs:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID ./build/strata-glm53-attnbench \
  --model models/glm53f-nvfp4 --devices 1,2 --context 32768 \
  --layer 19 --rows 2048 --history 4096 [--smoothing 0.9] \
  [--sweep 4096,8192,16384,30720] [--repeats N]
```

Source: `apps/strata_glm53_attnbench.cpp`; implementation
`Impl::attention_bench` (:10994). It loads real weights and runs the production
control flow over a **fabricated** cache, so timing and group structure are real
but activations are LCG noise — the attention output is meaningless and
`checksum` is only a same-input/same-output change detector (which is exactly
what you want for A/B-ing an exactness-preserving change).

> **Calibrate before quoting it.** Noise makes neighbouring queries select
> disjoint history, so it over-fragments: 755 groups at layer 19 / history 4,096
> where a real run gives **86**. `--smoothing` walks the row activations to
> restore that overlap. **Raise it until the group count matches ~86, then read
> the sweep.** This has not been done yet and is work item 4.

**Full arms** (20 min at 8,192): `experiments/scripts/glm53-prefill-arm.sh` and
`glm53-prefill-ab.sh`. Note `experiments/` is **gitignored** — the scripts, the
raw arm outputs and records 0248–0269 exist only on this box, not in the repo.

---

## 7. Flags landed, all OFF by default

| env var | what it does | why off |
| --- | --- | --- |
| `STRATA_GLM53_DEVICE_ATTEND` | stages 1–4: indexer projections, k-pool selection, sparse QK, sparse AV on device | exact, but a **wash** at 8,192 (1229.10 -> 1239.04 s) |
| `STRATA_GLM53_PAGE_EXPAND` | fix (1): expand a page's union once, slice groups out of it | exact, 1.27x, but incomplete (see §8.1) |
| `STRATA_GLM53_EXPAND_INVARIANCE_CHECK` | probe: expand a union as 1 batch vs 2 halves, compare bitwise | diagnostic only |
| `STRATA_GLM53_REGFED_EXPERTS` | register-fed NVFP4 expert kernel | 1.8–2.3x on a 2% term = 0.999x wall; **also changes generated text** |
| `STRATA_GLM53_STAGING_RESERVE_GIB` | demand-staging reserve, default 2 | never swept |

**Why stages 1–4 are a wash:** they remove ~6,522 s of host CPU, but that was
spread over ~21 threads so it was only ~311 s of *wall*, and it is replaced by
320.5 s of **serial** device time issued per group on one stream. Parallel host
work trades evenly against serial device work at this thread count. They are
kept because they are exact and become worthwhile once the group loop stops
fragmenting — not because they pay today.

**The invariance answer (measured, this session):**
`[glm53-expand-invariance] layer=19 union_rows=3071 exact=1`.
A 3,071-row union expanded as one call and as two half-batches is
**bit-identical**. So a split-K GEMM is *not* tiling K as a function of M, and
**the per-token expanded-K/V cache is legal.** This unblocks work item 2.

---

## 8. The work, ranked

### 8.1 Finish fix (1) — index in place instead of materializing  *(~half a day)*

`STRATA_GLM53_PAGE_EXPAND` is landed, exact and worth **1.27x** on a fragmented
layer-page (194.8 s -> 153.6 s, identical checksums at 755 groups). But its
expansion term only falls 99.6 s -> 82.9 s where the arithmetic says one
6,144-row GEMM should cost **~0.2 s**.

The gap is the slicing itself: each group still materializes a contiguous
`expanded` buffer, copying `expanded_rows x 32,768` floats — ~484 MB per group,
~365 GB over the page. **It trades a GEMM for a memcpy of the same order.**

Fix: make the QK and AV loops read through `mla_page_pos` into `page_expanded`
directly instead of gathering each group's rows. See the group loop from
:7462 (page union) and :7554 (`use_page_expansion` slice). Then run the hash
gate and flip the default on if green.

### 8.2 Fix (2) — per-token expanded-K/V cache  *(the main lever, now unblocked)*

`kv_b_proj(latent[i])` is a **pure function of token i**. It is identical on
every page and in every group, and the current design recomputes it every time.
Expanding each latent row once when it enters the cache turns the quadratic
expansion term linear: ~3.06M expansions at 32,768 tokens becomes **~360k**.

The open design question is *where the cache lives*. An expanded row is 128 KiB
(F32) or 64 KiB (BF16), so one layer at 32,768 tokens is 4.3 GB / 2.1 GB and all
eleven sparse layers cannot co-reside. Two viable shapes:

- **layer-outer / page-inner prefill loop** — only one layer's cache is live
  (4.3 GB). Requires restructuring the prefill loop, which is currently page-outer
  (`advance_prefill`, :10577 area) with layers inside `forward_prompt`. All pages'
  hidden activations resident is only 32,768 x 4,096 x 4 B = 671 MB, so this fits.
- **host-resident cache** — gathering a union over PCIe costs ~65 ms per
  layer-page against ~930 ms to recompute it, so **the transfer is not the
  obstacle**; and the data must reach the device anyway for QK/AV.

Precondition already satisfied (§7). Keep the hash gate green throughout.

### 8.3 Re-evaluate stages 1–4 after 8.1 and 8.2

They are a wash only because serial device time ~= parallel host wall. Collapse
904 groups to 44 and the device-call count collapses with it, changing the
trade. They may flip default-on for free. Re-measure; do not assume.

### 8.4 Calibrate the bench, then sweep to 30,720  *(cheap, and blocking)*

See §6. Nothing the bench reports about 32,768 is quotable until its group count
matches a real arm at a shape both can reach.

### 8.5 Staging-reserve sweep — free, never run

`STRATA_GLM53_STAGING_RESERVE_GIB` at 2/4/6. Low priority: FF is 26% of prefill
at length and saturating.

### 8.6 Carried-forward defect

Softmax CPU rose 46.84 -> 62.79 s in the 8,192 device-attend arm with **no stage
touching softmax**. One interleaved run says whether it is contention or thread
spread.

### 8.7 Housekeeping

The 8,192 / 16,384 arm shapes live only in the gitignored arm script, so the
long-shape harness is not reproducible from a fresh clone.

---

## 9. Expected ceiling — be honest about this

Items 8.1 and 8.2 attack expansion (135.6 s) plus most of the ~270 s of
per-group serial overhead — roughly **400 s of the 1,229 s** at 8,192 tokens, so
about **1.4–1.5x**. That leaves ~311 s of genuine QK/AV attention arithmetic,
which is what the device stages exist for.

**None of this closes the gap to the reference stack** (~171 ms/token fitted
marginal against 1.14). Fixing the expansion quadratic stops the curve bending
upward; it does not make Strata competitive on prefill. That two-orders-of-
magnitude gap is a different problem from the one this campaign worked on, and
whoever resumes should know it going in.

---

## 10. Closed — do not re-investigate

- **Expert transfer is at its floor.** 106 GB must move; the cache is 4.3 GB;
  there is zero cross-layer reuse. Closed arithmetic: 22,533/3 ~= 7,500
  expert-instances x 14 MB = 105 GB against 106 GB measured.
- **Transfers do not scale across the two 3090s** (shared root complex).
- **The device expert kernel is not the lever** — 2% of prefill, overlapped in
  decode. Recorded three separate times on this repo.
- **Routing changes are forbidden.** Model-quality degradation is against
  Strata's core concept; do not propose approximation.
- **Cache sizing cannot help** (arithmetic above).

## 11. Measurement traps that have already cost days

1. **`STRATA_GLM53_PHASE_PROFILE` is not neutral between arms.** It drains a
   CUDA event pair *per device command*, so it penalizes whichever arm issues
   more commands — it manufactured a fake 2% decode regression. Any arm that
   changes the *number* of device commands must be re-read with
   `PHASE_PROFILE=0` before its wall time is believed.
2. **Read a decomposition from one profile.** Subtracting a phase across
   commits once invented a 20% effect that did not exist.
3. **Do not divide parallel CPU totals by wall time.** Doing that mis-sized the
   attend core by ~3x and produced projections that had to be withdrawn.
   `graph_mla_nanoseconds` wraps the *whole* attention block (mHC, q_a/kv_a,
   norms, o_proj), not `attention_mla_page_sparse`.
4. **Screen by arithmetic before running anything.** Use 16.78 MMAC/expanded-row
   and ~446 GMAC/s. A 15-minute A/B to confirm a 6% win is a bad trade.
5. **`pgrep NAME` truncates at 15 characters.** `strata-glm53-attnbench` is 22,
   so it silently never matches and reports a live job as dead. Use `pgrep -af`,
   and check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`
   before believing a GPU is free. A stale job holding 38 GB produced
   `pinned spine leaves insufficient CUDA cache` — an error that points at
   configuration, not contention.
6. **Restarting the served model needs the owner.** `pkill` and llama-swap
   unload are both blocked; ask before a long run, not after.
