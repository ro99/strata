# Strata Research Charter

These rules apply to the entire repository.

## Mission

Build a ground-up, dependency-light C/C++ inference engine for dense and MoE
models that exceed local VRAM. Optimize measured throughput on RAM-rich consumer
machines while making steady-state decode independent of NVMe whenever the
admission contract says it should be.

## Hard invariants

- Weight precision below four bits is forbidden everywhere: canonical weights,
  caches, predictors, drafts, storage codecs, and fallback modes.
- Precision, router semantics, expert count, and top-k may not change silently.
- A predictor is advisory. Prediction may affect scheduling or prefetch only.
- Exact mode either completes exact work or reports a failure. No hidden fallback.
- Dense models larger than aggregate resident memory must be reported as I/O
  dependent; caching cannot manufacture sparsity.
- Runtime code is C/C++. Do not add Python or a framework runtime.
- Keep model files, raw traces, profiler captures, generated binaries, and large
  logs out of Git.

## Before changes

Run:

```bash
git status --short --branch
git log --oneline --decorate -8
git branch -vv
```

State the hypothesis, primary metric, correctness gate, memory ceiling, and
rollback condition. Preserve unrelated changes.

For anything that claims a throughput improvement, also state the measured
bottleneck resource, which resource the change reduces, and its effect on every
other resource. See "Measure the bottleneck before choosing a mechanism".

## Measure the bottleneck before choosing a mechanism

`research/moe-tiered-memory-decode-optimization.md` is the governing cost model:
`τ = max_r W_r/B_r + Σ_serial`. It is a **decision procedure**, not background
reading, and its first step is mandatory.

Before designing, planning, or building any optimization:

1. **Instantiate the model.** Measure `B_r` and `W_r` for every resource at the
   real operating point. Emit the per-phase breakdown of a step in ms and state
   which resource is `argmax_r`.
2. **Name the target term.** Say which resource the mechanism reduces and whether
   that resource is the current bottleneck. A mechanism that does not reduce
   `argmax_r` cannot improve `τ`, however elegant it is.
3. **Check the sign on every other resource.** Under a `max`, inflating the
   bottleneck to shrink a non-bottleneck is strictly negative. Speculation,
   recompute, and draft passes all add load to compute; account for it before
   building.
4. **Separate volume from overlap.** A term that is large because it is serial
   (`Σ_serial`: demand-miss stalls, cross-engine handoffs) is an overlap defect,
   not a volume problem. Fix overlap first — it is usually cheaper and strictly
   larger. Compare measured link throughput against the hardware's rated
   figure; an order-of-magnitude gap is a serialization bug, not a bandwidth
   limit.

Do the cheapest measurement that can falsify the idea, first. One profiling run
beats any amount of design. If a mechanism cannot be justified from measured
`W_r` on the target hardware, it is not ready to plan.

## Make the measurement cheap before making it long

A long run is a cost, not a virtue. Before launching one, state the ratio of
fixed setup to measured window, and the cheaper experiment you rejected.

1. **Measure the mechanism before the system.** A microbenchmark that
   reproduces the mechanism in isolation usually decides the question in
   minutes. Pinning the DeepSeek arena was settled by a 5-minute standalone
   H2D benchmark; the 4.4-hour end-to-end A/B only confirmed it.
2. **Reproduce the production access pattern, or the probe lies.** The
   topology probe reported pinned and pageable H2D as identical, 12.1 GB/s
   both, because it timed a warm reused buffer. The runtime reads a cold,
   randomly placed slice of a 147 GB mapping, where the same comparison is
   1.32 ms against 0.37 ms. A probe that does not reproduce the real access
   pattern can report no problem where a 3.5x problem exists.
3. **Find which part of the run is not under test, and shrink that.** Measure
   it; do not guess which part dominates. A DeepSeek arm at a 3,565-token
   prompt is 51 s of initialization, 38.5 min of prefill, and 73 s of decode.
   If decode throughput is the hypothesis, 95% of the arm is measuring nothing,
   and six arms come to six hours. Shortening the prompt to about 512 tokens
   makes the same arm roughly two minutes.
4. **Never inherit workload parameters from another script.** Prompt length,
   token count, and repetition count are part of the experiment design. Sizes
   copied from a script written for a different hypothesis are how a two-minute
   question becomes a six-hour one.
5. **State the arm budget before launching.** Give expected wall time per arm
   and in total. If the total exceeds roughly half an hour, say what was
   shortened and why the rest must stay.

## Do not launder a falsification

- A kill criterion derived before the work is binding **at the operating point
  that matters**, not at whatever configuration lets the hypothesis survive.
  Finding or manufacturing a regime where a rejected idea wins is not a rescue;
  record it as a rejection and say which regime would have been required.
- When a gate reads negative, stop and report. Do not continue building on the
  argument that a later stage will vindicate it.
- Stage dependencies in a plan are binding. If stage N produces the data that
  gates stage N+1, do not build N+1 first. If N is blocked, the plan is blocked.
- Never reuse a measured constant across operating points. Costs are functions of
  context length, prompt, cache bound, and batch shape — `τ(L)`, not `τ`. A
  number from a 128-token run does not describe a 4k-token run. Re-measure or
  state the extrapolation as an assumption in the plan's risk list.
- Prefer the trace and the workload you are actually optimizing. A prefill-heavy
  or low-entropy trace flatters every locality claim; decode locality must be
  measured on decode.
- Derive guard and fallback thresholds from the measured cost model. A threshold
  that assumes a cost the hardware does not have will let a regression through.

## Research discipline

- Work on one bounded hypothesis per experiment branch.
- Capture sequential route traces; aggregate frequency alone is insufficient for
  cache-policy evaluation.
- Simulate placement and replacement policies before implementing them in the
  runtime. Require a material projected improvement over the best baseline.
- A simulation is only as good as its inputs. State every cost constant, where it
  was measured, and at what operating point; treat a simulation fed by assumed
  constants as unrun. Re-run it against measured values before acting on it.
- Compare equal model, precision, route sequence, RAM, VRAM, peer, and I/O
  budgets.
- Report every run and the median of at least three interleaved repetitions.
- Separate prefill, decode, admission, load, and warm-up time.
- Record NVMe demand/prefetch bytes, host writes, H2D/D2H, network activation
  bytes, cache hits/evictions, allocation/synchronization, RSS, and per-GPU VRAM.
- Measure useful-prefetch bytes, not prediction recall alone.
- Never call a result a win when it is within observed run variance.

## Correctness

Run `make check` before every result commit. Kernel optimizations must match the
int4 reference oracle within the declared numerical contract. Architecture
adapters require operation- and layer-level fixtures built from the actual
target format, followed by full-model teacher-forcing and generation oracles.
Shape-reduced generated fixtures may test error paths, but smaller pretrained
models are not target substitutes.

DeepSeek support must preserve the declared attention/compression layout,
shared-expert execution, mHC state, selection and scoring functions, top-k
normalization, routed scaling, and DSpark verification exactly as declared by
the model manifest.

## Long jobs

Run long model loads, profiles, and benchmark matrices in named tmux sessions.
Write commands into reusable scripts and results into ignored deterministic
paths. Hand back the session, log, expected summary, and current stage.

## Worked example of the failure these rules prevent

`docs/experiments/0025` rejected shadow-speculative MoE offload after roughly
2,900 lines of runtime code. Every rule above is derived from one of its errors.

The governing cost model was written, adversarially reviewed, and available. Its
parameters were never instantiated on the target hardware, so the bottleneck was
never identified. At the production operating point a decode step was 576 ms:
255 ms of **serial** cold H2D at 2.46 GB/s — an eighth of the link's rated
bandwidth, so a serialization bug — plus 198 ms attention, 43 ms mHC, and 66 ms
MoE compute. The mechanism built reduced cold-transfer *volume* by 30% while
roughly doubling compute, which was already the larger term. Under a `max` that
is negative by construction, and the measurement agreed: 0.47–0.68×, and
0.66–0.94× even at perfect acceptance.

Three compounding errors, each now a rule:

- The plan's own break-even required `T_cold ≳ 260 ms/step`; production measured
  255 ms/step. That was a falsification, and the response was to build a knob
  that manufactured a more favourable regime instead of stopping.
- Two of three cost constants were taken from a 128-token profile and applied to
  a 3,565-token prompt. Base step was 203 ms assumed against 321 ms measured, and
  the draft cost 130 ms assumed against 321 ms measured — a draft that skips only
  the stall still pays a full forward pass.
- The simulation gate ran on a prefill-heavy, low-entropy trace. Decode dedup was
  1.44, not the 2.92 that trace implied.

The regression guard would not have caught any of it: its threshold assumed a
free draft, so a measured 1.5× regression passed. The cheapest falsifying
measurement — one phase profile at the real operating point — would have closed
the question before any code was written, and it pointed at a 1.79× win from
overlapping the transfer the design was trying to deduplicate.

## Git hygiene

- `main` is a validated research baseline, not a scratch branch.
- Create task branches from `main` unless the user explicitly names another
  base; never inherit unpromoted branch work implicitly.
- Use `exp/`, `feat/`, `fix/`, and `infra/` branches.
- Make reversible, single-purpose commits.
- Record failed experiments; do not merge failed runtime code.
- Never rewrite shared history or use destructive recovery commands.
