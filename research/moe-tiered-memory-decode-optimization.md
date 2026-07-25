# Maximizing MoE Decode Throughput across VRAM + RAM + SSD

**A mathematical formulation.**

> **Problem.** Given a memory hierarchy (VRAM, RAM, SSD) whose total fast capacity is smaller than the model, and a Mixture-of-Experts LLM, choose where every weight block and KV block lives, where each is computed, and what to prefetch — so as to **maximize decode throughput (tok/s)**.

This document was adversarially reviewed for correctness; a change log of the corrections applied is at the end.

---

## 0. The governing intuition

Decode is single-token, batch-1, so arithmetic intensity is ≈ 1 op/byte — it is **bandwidth-bound, not compute-bound**. Throughput is therefore set almost entirely by:

> **bytes that must be read/moved per token ÷ the bandwidth of the tier they come from.**

Every mechanism below (pinning, caching, prediction, overlap, CPU-in-place compute) exists to minimize that quantity on the *bottleneck* resource.

---

## 1. Sets and indices

| Symbol | Meaning |
|---|---|
| $t \in \mathcal{T} = \{g, c, d\}$ | memory **tiers**: VRAM ($g$), RAM ($c$), SSD ($d$) |
| $r \in \mathcal{R}$ | **resources that operate in parallel** (each with its own clock) |
| $b \in \mathcal{B}$ | **weight blocks** (attention, router, layernorms, shared experts, and each routed expert) |
| $\ell \in \{1..L\}$ | **KV blocks** (one per past token, or per page à la PagedAttention) |
| $L$ | current context length (grows during decode) |

The resource set is:

$$
\mathcal{R} = \{\underbrace{r_{\text{gpu}}, r_{\text{cpu}}}_{\text{compute engines (FLOP/s)}},\ \underbrace{r_{\text{hbm}}}_{\text{VRAM bandwidth (B/s)}},\ \underbrace{r_{\text{pcie}}, r_{\text{membus}}, r_{\text{ssd}}}_{\text{data links (B/s)}}\}
$$

> **Correction (critical):** $r_{\text{hbm}}$ — VRAM/HBM read bandwidth (~TB/s) — is an explicit resource. A VRAM-resident weight is not "free"; its dominant cost in a bandwidth-bound decode *is* its HBM read. Omitting this makes the all-in-VRAM case falsely look compute-bound.

---

## 2. Given parameters

**Hardware**
- $C_t$ — capacity of tier $t$ (bytes). *e.g. VRAM 64 GB, RAM 256 GB, SSD large.*
- $B_r$ — throughput of resource $r$ (bytes/s for links & HBM; FLOP/s for compute engines). Order of magnitude: $B_{\text{hbm}}\!\sim\!1\text{–}3\,\mathrm{TB/s}$, $B_{\text{membus}}\!\sim\!50\text{–}100\,\mathrm{GB/s}$, $B_{\text{pcie}}\!\sim\!16\text{–}64\,\mathrm{GB/s}$, $B_{\text{ssd}}\!\sim\!2\text{–}14\,\mathrm{GB/s}$.

**Model**
- $s_b$ — size of weight block $b$ (bytes); $f_b$ — its FLOPs per token.
- $p_b \in [0,1]$ — **access probability** of block $b$ per token. Hot path (attention, router, shared experts): $p_b = 1$. Routed expert $e$: $p_e = \Pr[e \in \text{top-}k]$ — the **routing skew**.
- $\kappa$ — aggregate KV bytes **per token** (folds in $2 \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{bytes}$). Total KV at length $L$ is $\kappa L$. Shrunk by GQA / MLA / KV-quant.
- $a_\ell \in [0,1]$ — probability KV block $\ell$ is read. **For causal full attention during decode, $a_\ell = 1$** for every existing block; $a_\ell < 1$ only under sliding-window / sparse attention.

---

## 3. Decision variables

$$
x_{b,t} \in \{0,1\},\quad \textstyle\sum_{t} x_{b,t} = 1 \qquad\text{(placement of each weight block)}
$$
$$
y_{\ell,t} \in \{0,1\},\quad \textstyle\sum_{t} y_{\ell,t} = 1 \qquad\text{(placement of each KV block)}
$$
$$
z_{b}^{\text{cpu}} \in \{0,1\} \qquad\text{(compute block }b\text{ CPU-in-place vs. ship to GPU)}
$$
$$
h_b \in [0,1] \qquad\text{(prefetch success: prob. a needed non-resident block is staged into VRAM before it is needed)}
$$

---

## 4. Cost model — a makespan (max), not a sum

Transfers overlap compute on separate engines/copy-streams, so per-token latency is the **makespan**: the load on the *busiest* resource under ideal pipelining — **not** the sum of all work.

Define the random 0/1 access indicators $\chi_b$ (block $b$ fired this token, $\mathbb{E}[\chi_b]=p_b$) and $\xi_\ell$ (KV block $\ell$ read, $\mathbb{E}[\xi_\ell]=a_\ell$). The per-token **load** on resource $r$ is:

$$
W_r \;=\; \underbrace{\sum_{b} f_b\,\chi_b\,\mathbf 1[b \text{ computes on } r]}_{\text{compute (gpu/cpu)}}
\;+\; \underbrace{\sum_{b} s_b\,\chi_b\,\big[(1-h_b)\,\mathbf 1[\text{transfer of }b \text{ crosses } r] + \mathbf 1[x_{b,g}{=}1]\,\mathbf 1[r{=}r_{\text{hbm}}]\big]}_{\text{weight bytes: link transfer if non-resident, else HBM read}}
$$
$$
\;+\; \underbrace{\sum_{\ell} \kappa\,\xi_\ell\,\big[\mathbf 1[\text{transfer of }\ell \text{ crosses } r] + \mathbf 1[y_{\ell,g}{=}1]\,\mathbf 1[r{=}r_{\text{hbm}}]\big]}_{\text{KV bytes: link transfer if non-resident, else HBM read}}
$$

Token latency under ideal overlap:

$$
\boxed{\;\tau(L) \;=\; \max_{r \in \mathcal{R}} \frac{W_r(L)}{B_r} \;+\; \Sigma_{\text{serial}}(L)\;}
$$

**Reading of the terms:**

- **A resident block is billed to $r_{\text{hbm}}$** at ~TB/s (its bytes still must be read from VRAM), *not* to the GPU FLOP clock. This is what keeps the model bandwidth-bound.
- **A non-resident block** is billed to whichever link carries it ($r_{\text{pcie}}$ / $r_{\text{membus}}$ / $r_{\text{ssd}}$), discounted by $(1-h_b)$: a reliably prefetched block contributes to link *load* but its latency is hidden behind compute.
- **Placement chooses which clock the bytes are billed to** — the whole game.

**The weight-vs-KV asymmetry (corrected).** Both weight and KV transfer terms sit *inside* $\max_r$, so both are equally **hideable** by prefetch — KV blocks $\ell = 1..L{-}1$ already exist from prior tokens and can be staged during earlier compute. The genuine asymmetry is about **volume, not hideability**:

> Weights get a $(1-h_b)$ **volume discount** — un-accessed or resident experts move no bytes. KV at $a_\ell = 1$ gets **no such discount**: no predictor can skip history the attention actually needs. KV volume is *irreducible*; only its *latency* is hideable. Only the freshly produced block $\ell = L$ (~$\kappa$, negligible vs. $\kappa L$) is truly serial.

**$\Sigma_{\text{serial}}$** captures only latency that $\max_r$ *cannot*: (1) pipeline fill/drain bubbles at sequence start; (2) demand-miss stalls where a cold SSD/PCIe fetch sits on the critical path with no independent work behind it; (3) non-overlapping cross-engine handoffs (e.g. GPU attention → CPU expert). It does **not** include same-engine ordering like attention-before-FFN — those FLOPs already both sum into $W_{\text{gpu}}$ and are serialized by dividing by $B_{\text{gpu}}$.

---

## 5. Constraints

**Capacity (per tier):**
$$
\sum_{b} s_b\, x_{b,t} \;+\; \sum_{\ell} \kappa\, y_{\ell,t} \;\le\; C_t \qquad \forall\, t, L
$$

This is the crux: **weights and KV compete for the same $C_g$.** As $L$ grows, $\sum_\ell \kappa\, y_{\ell,g}$ swells and squeezes experts out of VRAM — the VRAM budget for experts is a *shrinking* function of context length.

**Compute–placement coupling:**
$$
z_b^{\text{cpu}} \le x_{b,c}, \qquad h_b \le x_{b,g} + \big(\text{prefetchable from } c/d \text{ within the lookahead window}\big)
$$

**Prefetch causality** — you can hide $b$ only if its transfer fits in the compute window $\Delta_b$ available before it is needed (router-lookahead / cross-layer predictor sets $\Delta_b$; $\Pi_b$ is predictor accuracy):
$$
h_b \;\le\; \Pi_b \cdot \mathbf 1\!\left[\frac{s_b}{B_{\text{path}(b)}} \le \Delta_b\right]
$$

---

## 6. Stochastic, over a trajectory

The access indicators $\chi_b, \xi_\ell$ are random per token, and $L$ marches $1 \to N$. The objective is expected total decode time; tok/s is its reciprocal:

$$
\boxed{\;\max_{x,\,y,\,z,\,h}\quad \text{tok/s} \;=\; \frac{N}{\displaystyle\sum_{L=1}^{N} \mathbb{E}_{\text{routing}}\!\big[\tau(L)\big]}\;}
$$

subject to all constraints holding for every $L$ ($x, y$ are *policies*, re-optimizable online as $L$ grows).

> **Jensen caveat.** $\max_r(\cdot)$ is convex, so
> $$\mathbb{E}\big[\max_r W_r/B_r\big] \;\ge\; \max_r \mathbb{E}[W_r]/B_r.$$
> Plugging the mean loads ($p_b, a_\ell$) into $W_r$ and then taking $\max_r$ gives the **max-of-means**, a *lower bound* on the true **mean-of-max**. At batch-1 decode, routing is a single hard top-$k$ draw per token, so per-token load variance is large and this gap is material. Treat the mean-plugged $\tau$ as an optimistic bound and, for accuracy, model the fired-set distribution or add a variance/tail correction on the bottleneck load.

---

## 7. Problem class and the value-density relaxation

- With integrality and per-tier capacity and $\sum_t x = 1$, placement is a **generalized assignment problem** fused with a **caching / prefetch** problem ($h_b$ and eviction) — **NP-hard**.
- The $\max_r$ objective makes it a **min-makespan (bottleneck)** program, not a linear-cost one.
- The online version (re-placing as $L$ grows) is a **Markov Decision Process**.

**Relaxation for intuition (single-bottleneck).** *Assume one dominant slow resource*, so $\tau$ reduces to a single $W_{\text{slow}}/B_{\text{slow}}$ term. Then placement becomes a fractional knapsack: fill VRAM by descending **value density** = bottleneck-bytes removed per VRAM byte spent. Accounting for prefetch (residency only helps the fraction prefetch can't hide):

$$
\rho_b \;\propto\; \underbrace{p_b}_{\text{access freq}} \cdot \underbrace{(1-h_b)}_{\substack{\text{residency beats prefetch}\\\text{only on the unhidden part}}} \cdot \underbrace{\left(\tfrac{1}{B_{\text{slow}(b)}} - \tfrac{1}{B_{g}}\right)}_{\approx\, 1/B_{\text{slow}}\ \text{since } B_g \gg B_{\text{slow}}}
$$

> **Corrections applied here:**
> - The $(1-h_b)$ factor is **retained** so it matches $W_r$: a reliably *prefetchable* expert has low pinning value (prefetch already hides it), while **unhideable state** — KV ($a_\ell{=}1, h{=}0$) and unpredictable experts — correctly outranks it for scarce VRAM. (State your regime: this assumes prefetched bytes are *not* the bottleneck; in a strictly link-bandwidth-bound regime, prefetched bytes still consume the link, so drop $(1-h_b)$ from *both* $W_r$ and $\rho_b$.)
> - This is exact **only** under the single-bottleneck relaxation. For the general multi-resource makespan, greedy-by-$\rho$ is a **heuristic**, not the optimum: moving a block to VRAM helps only if it offloads the *current* bottleneck.

Greedy admission by descending $\rho_b$ yields a **waterline**: blocks above it (that fit in $C_g$) go to VRAM, below it to RAM/SSD.

> **Sign correction.** As $L$ grows and KV eats $C_g$, the effective expert budget *shrinks*, so the greedy stops earlier and the density cutoff **RISES** — fewer, denser blocks survive. (The *number* of resident experts drops; the $\rho$ threshold rises.)

This relaxation *derives* every heuristic:
- Hot path ($p_b{=}1$, $h$ irrelevant) → highest $\rho$ → **pinned first**.
- KV ($a_\ell{=}1$, unhideable *volume*) → high $\rho$ → **second claim on VRAM**.
- Routed experts ranked by $p_e \cdot (1-h_e)$ → hottest / least-predictable fill the remainder.
- Cold, predictable experts ($p_e$ small, $h_e$ high) → below the line → **RAM/SSD, prefetched**.

---

## 8. Mapping to concrete techniques

| Lever | Effect in the model |
|---|---|
| Weight quantization (GPTQ/AWQ) | shrinks $s_b$ |
| GQA / MLA / KV-quant | shrinks $\kappa$ |
| Expert predictor / router-lookahead (Pre-gated MoE, PowerInfer-style) | raises $\Pi_b$, hence achievable $h_b$ |
| PagedAttention (vLLM) | makes $y_{\ell,t}$ a feasible per-block variable (blockwise KV → tierable) |
| StreamingLLM (sinks + window) | drives $a_\ell \to 0$ for old middle context |
| CUDA copy-streams / overlap | realizes the $\max_r$ (vs. a serial sum) |
| CPU-in-place expert compute | $z_b^{\text{cpu}}$: pay $B_{\text{membus}}$, skip $B_{\text{pcie}}$ |

**Aggregate metrics (corrected labels):**
- **Expected resident-hits per token** $= \sum_b p_b h_b$ (a *count*, can exceed 1).
- **Hit rate** $\in [0,1]$ $= \dfrac{\sum_b p_b h_b}{\sum_b p_b}$ (normalized by expected accesses).
- **The one number to minimize:** expected **bottleneck-resource bytes per token**, $\max_r \mathbb{E}[W_r]/B_r$.

---

## 9. Compact statement

> **Given** tiers $(C_t, B_r)$ and a model $(s_b, f_b, p_b, \kappa, a_\ell)$, **choose** placement $x, y$, compute-site $z$, and prefetch policy $h$ to
> $$\max\ \frac{N}{\sum_{L=1}^{N} \mathbb{E}\big[\max_r W_r(L)/B_r + \Sigma_{\text{serial}}(L)\big]}$$
> **s.t.** $\sum_b s_b x_{b,t} + \sum_\ell \kappa\, y_{\ell,t} \le C_t\ \ \forall t, L$;
> prefetch causality $h_b \le \Pi_b \mathbf 1[s_b/B_{\text{path}(b)} \le \Delta_b]$;
> compute–placement coupling $z_b^{\text{cpu}} \le x_{b,c}$, $h_b \le x_{b,g} + (\text{prefetchable})$.

---

## 10. Prior art to raid

- **FlexGen** — throughput-optimal offload scheduling across the hierarchy as a cost-minimization.
- **PowerInfer / PowerInfer-2** — hot/cold activation placement + prediction (the $\Pi_b$ idea).
- **Pre-gated MoE, expert-prefetch work** — the canonical MoE $h_b$ predictor.
- **KTransformers, llama.cpp MoE offload** — GPU-hot-path + CPU-cold-expert split ($z_b^{\text{cpu}}$).
- **PagedAttention / vLLM, SGLang RadixAttention** — blockwise KV, prefix reuse ($y_{\ell,t}$).
- **StreamingLLM, H2O, SnapKV** — attention sinks / KV eviction ($a_\ell$).
- **MLA (DeepSeek-V2/V3), KVQuant, KIVI** — architectural / low-bit KV shrink ($\kappa$).
- **HAWQ / AWQ / SpQR / SqueezeLLM** — sensitivity-driven mixed precision ($s_b$).

---

## Appendix — change log vs. the first draft (from adversarial review)

| # | Severity | Fix applied |
|---|---|---|
| 1 | **critical** | Added $r_{\text{hbm}}$ (VRAM bandwidth) to $\mathcal{R}$ and a resident-read term to $W_r$, so a VRAM-resident block is billed to ~TB/s HBM — restoring the bandwidth-bound premise (was billed to the GPU FLOP clock, i.e. effectively free). |
| 2 | **major** | Re-cast the KV claim: transfer *latency* is hideable (blocks $1..L{-}1$ pre-exist and prefetch like weights); the real asymmetry is *volume*-irreducibility at $a_\ell{=}1$. "Unhideable" was a non-sequitur. |
| 3 | **major** | Restored the $(1-h_b)$ factor in $\rho_b$ to match $W_r$; noted the regime dependence. Was over-valuing prefetchable experts for pinning. |
| 4 | minor | Waterline **rises** (not drops) as KV eats capacity. |
| 5 | minor | "hit rate" normalized to $(\sum p_b h_b)/(\sum p_b)$; unnormalized form relabeled "expected resident-hits per token." |
| 6 | minor | Removed attention→FFN from $\Sigma_{\text{serial}}$ (same-engine compute already in $W_{\text{gpu}}$); redefined it as fill/drain + demand-miss + cross-engine handoff only. |
| 7 | clarify | Fractional-knapsack/waterline flagged as exact only under a single-bottleneck relaxation; heuristic for the general makespan. |
| 8 | clarify | Added the Jensen caveat (mean-of-max ≥ max-of-mean) — the mean-plugged $\tau$ is an optimistic bound, material at batch-1. |

*Confirmed correct in review (no change needed): the $\max_r$ makespan abstraction as an ideal-overlap bound; the GAP + caching NP-hardness framing; $\kappa L$ KV sizing; $p_e = \Pr[e \in \text{top-}k]$; $a_\ell = 1$ for causal full attention; the $s_b$/$\kappa$ quantization mapping; the prior-art attributions.*
