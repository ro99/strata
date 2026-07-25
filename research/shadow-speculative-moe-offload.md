# Shadow-Speculative MoE Offload — An Engineering Design

**How to make an MoE model bigger than VRAM decode fast on a VRAM + RAM + SSD box, losslessly.**

Companion to the theory doc ([moe-tiered-memory-decode-optimization.md](moe-tiered-memory-decode-optimization.md)). That one asks *what is optimal*; this one asks *what do we build*. Every claim here was adversarially vetted for physical soundness, losslessness, and novelty — the honest speedup accounting and the rejected overclaims are kept in, not hidden.

---

## The aha

> **In an MoE, the part that decides *which* weights you need and the part that *is* the expensive weights are two different computations — and only one of them is expensive.**

- The **router** is tiny (a `[d_model × N]` matmul), always resident, runs at VRAM speed.
- The **expert FFNs** are the whole multi-hundred-GB bulk, and at batch-1 decode you touch only top-$k$ of $N$ per token — unpredictably.

Batch-1 decode stalls because you discover *which* cold experts you need only when the router fires, then you wait on a PCIe/SSD fetch, every layer, every token. The router (cheap) is welded to the fetch (expensive) by a data dependency.

**Break the weld.** Run the model as **its own draft**, at full router fidelity, but compute every *cold* (non-resident) expert with a tiny **resident shadow** — a rank-$r$ ($r\approx8\text{–}16$) or shared-basis surrogate that lives in VRAM for all $N$ experts at ~1–2% of full weight size. The draft **never stalls** (everything it touches is resident), yet because its **real routers run bit-exact at every layer**, it emits — $\gamma$ tokens ahead — the **exact set of cold experts** the real forward pass will need.

That set is a **lossless prefetch oracle**: no separate predictor to train, no biased proposal distribution, no guessing. Then you spend the lead time fetching those experts *once*, coalesced, hidden behind compute, and verify the whole speculative window as a single **expert-major "mini-prefill."**

The name for the shape: **self-speculation where the degraded axis is "cold experts → shadows," and the draft's own router trace is the fetch schedule.**

---

## Why this beats the obvious idea

The tempting first move is *residency-biased speculative decoding*: bias the draft to propose tokens whose experts are already resident, and lean on rejection sampling to stay lossless. It **is** output-lossless — but it's **self-defeating**, and we reject it:

> Biasing the draft's proposal $q$ away from the target $p$ **lowers the acceptance rate** $\alpha$. Worse, the tokens that get rejected are exactly the ones whose true continuations need the non-resident experts you steered around — so you burn fetches on discarded branches and advance fewer tokens per pass. You trade cheaper fetches for fewer accepted tokens, and the trade is usually a loss.

The shadow oracle sidesteps this entirely: **it never touches $q$.** The draft proposes from (near) the true distribution; the shadow only changes *how cold experts are approximated during drafting*, which affects *which experts we prefetch*, never *which tokens we propose*. Zero acceptance-rate tax.

---

## The two invariants (losslessness is conditional — enforce in code)

The output is **bit-exact to the full-precision target** if and only if both hold. These are not footnotes; they are coded assertions.

1. **Exact tree rejection sampling.** Use SpecInfer/EAGLE-style exact multi-candidate verification with correct residual resampling. **Medusa-style "typical acceptance" is forbidden** — it is an explicitly lossy relaxation that changes the output distribution.

2. **Verify never substitutes a shadow.** The verify pass computes the **true** expert for every routed expert. On a prefetch miss it **fetches-and-stalls** — it must *never* fall back to a shadow to dodge the stall. The shadow lives only in the draft's proposal $q$; the moment it leaks into verify's target logits $p$, output is silently biased. This is the single most dangerous line in the system, because it is one tempting latency optimization away from wrong.

Why this is clean: standard speculative sampling is output-exact for **any** proposal $q$, provided verify computes true $p$ and resampling is correct. The shadow's approximation is confined to $q$. So losslessness is **independent of shadow fidelity** — a bad shadow costs *speed* (more prefetch misses), never *correctness*.

---

## The pipeline (per decode step)

```
┌─ 1. DRAFT (self-speculation, resident-only, zero stalls) ─────────────┐
│  Run target over a token TREE (width W, depth γ):                     │
│    • attention + shared experts + REAL routers  → full fidelity        │
│    • routed experts: if resident → true expert                         │
│                      else        → resident shadow  Ê_i = shared + UᵢVᵢ │
│  Byproduct: the real routers log the union of (layer, cold-expert-id)  │
│             the verify pass will need  →  the MANIFEST                  │
└───────────────────────────────────────────────────────────────────────┘
                              │  manifest (high-recall, NOT exact)
                              ▼
┌─ 2. MANIFEST → PREFETCH ──────────────────────────────────────────────┐
│  U = manifest \ resident                                               │
│  if bytes(U) > VRAM − hotpath − KV:  prune tree branches by            │
│        (draft_prob × expert_reuse) until U saturates residency         │
│  issue ONE coalesced, consumption-ordered GPUDirect gather             │
│        (early-layer experts first), double-buffered on a copy stream   │
│  per expert e:  b = #tree-positions routing to e                       │
│        b large  → stream weights to VRAM (transfer amortized ×b)       │
│        b == 1   → compute CPU-in-place, ship the ~14KB activation      │
│                   (not the ~88MB weights)                              │
│  over-fetch margin M > k  (manifest is high-recall, not exact)         │
└───────────────────────────────────────────────────────────────────────┘
                              │  experts resident (or CPU-staged)
                              ▼
┌─ 3. VERIFY (batch-γ mini-prefill, expert-major grouped-GEMM) ─────────┐
│  for each layer:                                                       │
│    gather window tokens by TRUE routed expert id                       │
│    for each unique expert: read ONCE, apply to all its tokens          │
│  TRUE routing + TRUE experts throughout.                               │
│  prefetch miss → demand-fetch-and-stall. NEVER substitute a shadow.    │
└───────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ 4. ACCEPT ───────────────────────────────────────────────────────────┐
│  exact tree rejection sampling + residual resample → bit-exact to p    │
└───────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ 5. REGRESSION GUARD (mandatory) ─────────────────────────────────────┐
│  measure α and window dedup each step; EWMA.                           │
│  if  α × dedup < 1  →  fall back to plain autoregressive decode         │
└───────────────────────────────────────────────────────────────────────┘
```

**The dataflow collapse in step 3 is the second half of the trick.** A speculative verify pass over a $\gamma$-token tree already has many tokens in flight — it *is* a batch-$\gamma$ mini-prefill. So route it through the **same expert-major grouped-GEMM kernel prefill uses**: load each needed expert once, apply to every window token that routed to it. One code path serves both prefill and decode.

---

## Unifying prefill and decode

Prefill is the native case for the kernel above: compute-bound, touches ~all experts, stream them in static layer order (FlexGen-style). The design **adds nothing to prefill throughput** beyond code unification — and we say so plainly. What it *does* add at the prefill→decode seam:

- **Free oracle warming:** prefill touches all experts anyway, so it populates the expert-tag store (below) at zero extra cost — the oracle is warm the instant decode starts.
- **Cold-start kill:** rank prompt tokens by prefill attention, pre-warm that working set into VRAM before token 1, so the first decode token doesn't hit the cold cliff.

---

## Supporting mechanisms (each vetted, each scoped)

| Mechanism | What it adds | Status |
|---|---|---|
| **Coalesced runahead prefetch + overlap** | One ordered GPUDirect scatter-gather, early-layer-first, double-buffered behind compute; removes per-layer fetch-latency **bubbles**. | Verified **~1.2–2×** when per-fetch latency is high (SSD) and locality good; decays to ~1× (possibly net-negative after shadow cost) on high-entropy text. This is the honest incremental win — it removes bubbles, it does **not** create the amortization. |
| **`b`-aware placement break-even** (Fiddler-style) | Per cold expert, stream-to-VRAM vs. compute-CPU-in-place by how many window positions `b` hit it. Singletons compute on CPU, shipping the ~14KB activation not ~88MB weights. | Sound and cheap. Mild FP-nondeterminism at the accept/reject margin (output still distributed as $p$). |
| **Attention-tagged-KV oracle** (ASTER Part 2 only) | Tag each cached token with the experts it routed to (~1–2% KV overhead, warmed free in prefill); use the attention histogram over tagged tokens to predict *deep-layer* experts with more lead time than a shadow forward gives. | **Lossless** (pure cache hint, router runs exactly, demand-fetch on miss). **Unproven** vs. a trained one-layer gate (Pre-gated MoE) or activation trace (MoE-Infinity) — must win a correlation study first. Niche: the SSD tier, where one-layer lead is insufficient. |
| **Opt-in LOSSY expert compression** (QMoE / BitDelta / low-rank) | Shrink cold-expert payload ~3–5× to push the transfer term under the RAM-tier roofline. Shares the shared-basis infra with the shadows. | The **only** lever that actually crosses the roofline at the RAM tier — but **not lossless**: rejection sampling then makes output exact to the *lossy* model, not full precision. Clearly-labeled opt-in, with a quality-vs-rank/bit curve. |

---

## The honest speedup accounting

This is where most "10× decode!" pitches cheat. The breakdown, baseline-first:

**Baseline (not novel — this is the comparison point, not the win):**
Tree speculation + batched expert-major verify already gives **~2–4×** over naive batch-1 demand-fetch, purely from $\gamma$-/$\tau$-amortization — each cold expert read *once per window* instead of once per token. **This amortization is standard speculative decoding (Leviathan/Chen/SpecInfer), not a property of shadows.** Crediting it to this design is double-counting; we don't.

**Marginal win from the structural tricks here, on top of that baseline:**
- Runahead coalescing + overlap + bubble removal: **~1.2–2×** (high-latency tier + good locality), decaying to ~1× otherwise.
- Shadow oracle: near-exact cold-set discovery at **zero acceptance-rate tax** (vs. a $q$-biasing scheme that depresses $\alpha$).

**Net honest magnitude:**
- **~1.5–3×** over naive batch-1 offload.
- **~1.2–1.8× *incremental*** over an already-strong tree-spec + demand-fetch baseline.
- Regime-confined: RAM/PCIe-tier offload, high expert temporal locality, decent $\alpha$.

**Hard ceilings, stated up front:**
1. **Hot-path wall (~50 tok/s).** The resident hot path (~40 GB read/token at ~2 TB/s ≈ 20 ms) is untouched. No amount of expert-transfer reduction goes past it.
2. **SSD floor.** Unique-cold-bytes / link-BW at 2–14 GB/s is *hundreds of ms* — it cannot hide behind batch-1 compute ($\gamma\approx5\text{–}8$ lifts arithmetic intensity nowhere near the ~300 needed to be compute-bound). Only the **opt-in lossy** mode crosses it, and only to the lossy model.
3. **Fits-in-VRAM → ~1×.** No cold fetch to hide, nothing to do.
4. **Prefill → ~1×** on throughput (compute-bound already).

---

## The failure mode you must design against

There is a **hard anti-correlation** at the heart of this, and it's the thing to watch:

> Acceptance rate $\alpha$ is **worst exactly in the high-offload regime the trick is for.** More offload ⇒ most experts cold ⇒ the draft runs mostly on the shadow manifold ⇒ per-layer approximation error compounds ⇒ $\alpha$ tanks. And rare-but-decisive experts (domain/format-specific) are precisely the ones that are **cold *and* high-gate** — where the shadow is most wrong.

Consequences:
- The amortization factor is $\dfrac{\gamma \cdot k \cdot L}{|\text{unique cold experts over window}|}$ — large only for high-locality experts **you'd have cached anyway**, and it collapses toward 1× on the one-off cold experts that dominate the offload bill. **It amortizes best where you need it least.**
- **Throughput can regress below plain decode** when $\alpha \times \text{dedup} < 1$: verify fetches the full cold union to advance at most $\alpha\gamma$ accepted tokens.

Hence the **mandatory regression guard** in step 5 — measure $\alpha$ and dedup, fall back when the product drops below break-even. Without it, this design is *slower* than a dumb loop on adversarial inputs.

Also: the shadows themselves ($r \cdot N \cdot L$ factors, a few GB) compete for the *exact* VRAM (KV + hot experts) whose scarcity causes the cold misses. Budget it; measure it; don't assume it's free.

---

## What we explicitly rejected (and why)

Kept visible so the design isn't quietly re-inventing them:

- **$q$-distorting residency/coherence bias** — lossless but self-defeating ($\alpha$ drop; fetches on rejected branches). Replaced by shadow-as-pure-oracle.
- **"Exact oracle for free."** False. Layer-$L$ shadows perturb the hidden state feeding layer-$(L{+}1)$'s router, so the manifest is **high-recall, not exact**. Needs over-fetch margin $M>k$ + demand-fetch fallback.
- **Coactivation-cartridge sequential-DMA as a bandwidth multiplier.** Evaporates: the oracle hands you the whole window set up front, so you issue it as high-queue-depth **parallel** reads of already-multi-MB residuals that hit ~peak BW with no re-layout and no over-fetch. Contiguous bundling only adds over-fetch risk. Demoted to a scheduling hint gated on measured co-fire probability.
- **Medusa typical-acceptance** — lossy, forbidden.
- **Shadow-in-verify to dodge a stall** — forbidden invariant violation.
- **ASTER's attention-driven KV quantization (Part 3)** — lossy (perturbs attention → logits → tokens; it's H2O/SnapKV, not novel). Split into an explicitly lossy opt-in section; does not ride the lossless headline.
- **Overclaim narratives** — "transfer → 0", "hidden entirely behind draft compute", "migrate to the compute roofline", "near-peak from coalescing an 88 MB read". Physically unsound: hideable bytes ≈ draft-compute-time × link-BW ≈ *about one expert* at the SSD tier.

---

## Prior art this composes (novelty is the *combination*)

None of the primitives are new; the specific composition — *cold-experts-only shadow degradation as the self-speculation axis, whose router trace is the MoE fetch schedule, verified as an expert-major mini-prefill* — is the contribution.

- **Self-speculation:** Draft&Verify (Zhang et al., ACL 2024), LayerSkip (Elhoushi et al. 2024) — same model, degraded along one axis, lossless via rejection sampling. Here the axis is *cold experts → shadows*.
- **Speculative decoding:** Leviathan/Chen 2023, SpecInfer (tree + exact rejection), Medusa, EAGLE/EAGLE-2. **SpecExec** (Svirschevski et al. 2024) — spec decoding under weight offload.
- **Router-ahead expert prefetch:** Pre-gated MoE (Hwang et al., ISCA 2024), MoE-Infinity, ProMoE, Lina. Mixtral-offloading (Eliseev & Mazur 2023) already does speculative expert prefetch + LRU.
- **CPU/GPU expert orchestration:** Fiddler, KTransformers, PowerInfer/PowerInfer-2.
- **Expert compression:** QMoE (Frantar & Alistarh), BitDelta, low-rank/MPO factorization.
- **KV/oracle bits:** H2O, SnapKV (the ASTER Part-3 lineage); PagedAttention (blockwise KV).

**Benchmark discipline:** report marginal speedup over **both** SpecInfer+demand-fetch **and** Pre-gated-MoE single-token — not just naive decode — or the gains are indistinguishable from plain tree speculation. Report $\alpha$ as a function of offload ratio **and** gate-weight-on-cold-experts (worst case, not the friendly case).

---

## Load-bearing assumptions (kill-criteria for the whole design)

1. **Expert temporal locality:** unique-cold $\ll \gamma k L$ over a window. If routing is near-uniform per token, the amortization is gone.
2. **Draft-router ≈ target-router agreement** at depth on the shadow manifold — i.e. the shadow's perturbed hidden states still predict the *true* cold set with high recall.
3. **RAM/PCIe tier, not SSD.** SSD needs the lossy mode to cross the roofline at all.

If a workload violates 1 or 2, the regression guard should already be bouncing you to plain decode — which is the honest, safe floor.
