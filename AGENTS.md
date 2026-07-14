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

## Research discipline

- Work on one bounded hypothesis per experiment branch.
- Capture sequential route traces; aggregate frequency alone is insufficient for
  cache-policy evaluation.
- Simulate placement and replacement policies before implementing them in the
  runtime. Require a material projected improvement over the best baseline.
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

## Git hygiene

- `main` is a validated research baseline, not a scratch branch.
- Use `exp/`, `feat/`, `fix/`, and `infra/` branches.
- Make reversible, single-purpose commits.
- Record failed experiments; do not merge failed runtime code.
- Never rewrite shared history or use destructive recovery commands.
