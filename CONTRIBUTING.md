# Strata engineering guide

This is the operating policy for humans and coding agents changing Strata. It
exists to keep the repository a product codebase rather than a record of every
experiment used to develop it. Read this file together with
[`docs/current-architecture.md`](docs/current-architecture.md) before making a
structural change.

Strata is a C++20/CUDA **application**, not an installable C++ SDK. Its shipped
surfaces are the command-line programs and HTTP server. The headers and static
libraries in this repository are internal implementation interfaces.

## Before changing anything

1. Read the root README, this guide, and the current architecture document.
   For model work, also read the applicable file under `docs/models/` and the
   [model bring-up guide](docs/model-bringup-guide.md).
2. Inspect `git status` before editing. Preserve unrelated work and never
   discard a user's changes to obtain a clean tree.
3. Classify the work before choosing a path:
   - product code, tests, operator documentation, and reusable build checks
     belong in Git;
   - probes, hypotheses, benchmark scratch code, handovers, issue notes, and
     raw measurements belong under the ignored `experiments/` workspace;
   - generated files belong in an ignored build or result directory;
   - Strata UI work belongs in the separate `strata-ui` repository.
4. Find the owning CMake target and layer before introducing a source or
   header. Do not place a file first and repair the architecture afterward.

## Repository layout

| Path | What belongs there | What does not belong there |
|---|---|---|
| `apps/` | Thin production entry points for shipped executables | Business logic, headers, probes, benchmarks, vendored code, a TUI, or web assets |
| `include/strata/` | Internal interfaces shared across translation units, arranged by owning layer | Private helpers, implementation fragments, generated headers, or third-party headers |
| `src/platform/` | Model-neutral types, parsing, storage I/O, numerics, hardware discovery, tracing, and diagnostics | CUDA implementation or model policy |
| `src/engine/` | Model-neutral placement, residency, routing, sampling, sessions, and registries | Concrete model implementations or application presentation logic |
| `src/models/<model>/` | One model's manifest, checkpoint reader, operations, executor, and runtime | Generic facilities that multiple models should share |
| `src/models/common/` | Facilities genuinely shared by model implementations | A dumping ground for code used by only one model |
| `src/app/` | Application facade, model catalog, and protocol handling | Model math, device kernels, or CLI `main` functions |
| `kernels/cpu/` | CPU reference and production kernels | Standalone experiments or model orchestration |
| `kernels/cuda/` | CUDA owning translation units and Strata-side adapters | Raw third-party source trees |
| `kernels/cuda/detail/` | Private CUDA implementation fragments grouped by responsibility | Public interfaces or unrelated collections of kernels |
| `tests/` | Deterministic product tests, fixtures, and smoke-test helpers | One-off performance probes and raw benchmark output |
| `scripts/` | Reusable repository checks and test oracles | Personal automation or one-off experiment drivers |
| `docs/` | Current product behavior, architecture, formats, and operator runbooks | Research diaries, handovers, campaign trackers, or issue scratchpads |
| `vendor/<project>/` | An exceptional, reviewed third-party source snapshot with its license and provenance | First-party code or dependencies copied for convenience |
| `experiments/` | Local probes, experiment CMake, research notes, scripts, and raw results | Anything required to build, test, document, or operate the product |

`build/`, `build-*`, `models/`, `results/`, and `experiments/` are ignored local
state. They may be deleted when their contents are no longer needed. Never
commit them or bypass the ignore rules with `git add -f`.

`AGENTS.md`, `CLAUDE.md`, `.codex/`, and `.claude/` are also ignored because
they contain developer-specific agent configuration. Repository-wide rules
belong in this file so every contributor receives the same policy.

## Dependency architecture

The six production libraries form one downward-only chain:

```text
strata_app
    ↓
strata_models
    ↓
strata_engine
    ↓
strata_kernels
    ↓
strata_device
    ↓
strata_platform
```

A layer may depend on itself or anything below it, never anything above it.
The lower four layers are model-neutral: they must not acquire DeepSeek,
Gemma, GLM, Inkling, Kimi, or Laguna policy merely because one model is the
first consumer of a feature.

When adding or moving a file:

- add every production source to its owning target in `CMakeLists.txt`;
- give every new `include/strata/**/*.hpp` an owner that
  `scripts/check_layers.py` can derive, adding a reviewed ownership override
  only when normal same-basename ownership is impossible;
- update both the native CUDA implementation and
  `src/platform/cuda_backend_stub.cpp` when changing `CudaBackend`;
- avoid a new target unless its position in the dependency order is explicit
  and both architecture checks understand it;
- use the existing registry seams instead of adding model-name switches to
  application or engine code.

The include-graph and archive-symbol checks are architecture gates, not
advisory lint. Do not suppress them casually. Any temporary exception must be
narrow, documented, and carry an expiry; the preferred exception count is
zero.

## Headers and includes

`include/strata/` mirrors source ownership:

```text
include/strata/{platform,device,kernels,engine,app}/
include/strata/models/{common,<model>}/
```

The outer `include/` directory is a conventional namespace container. Today
it contains only `strata/`. A second directory would make sense only for a
deliberate second first-party namespace or separately owned component. It is
not a home for vendored headers; those remain under `vendor/` and behind a
Strata adapter.

Put a header under `include/strata/` only when more than one translation unit
needs its declaration or it forms a real cross-component seam. Keep private
headers beside their implementation under `src/` or `kernels/*/detail/`.
`apps/` should normally contain no `.hpp` files.

Every source must directly include the standard and project headers that
provide the names it uses. Do not rely on transitive includes. For example,
code using `std::none_of` includes `<algorithm>` itself even if another header
happens to make the local build pass. Include paths and filenames must match
their architectural owner and use the repository's canonical model name.

## Applications stay thin

An `apps/*.cpp` file should parse command-line input, call an application
interface, render the result, and select an exit code. If logic needs unit
tests, reuse by another executable, or knowledge of a model/runtime, move it
to the appropriate `src/` layer and expose the smallest internal interface.

Only maintained user-facing executables belong in `apps/`. A profiler or
probe is not promoted merely because it compiles. Put it in
`experiments/probes/` with an ignored `experiments/CMakeLists.txt` until its
functionality becomes a supported application. Strata has no TUI, and TUI
code or Rust TUI build files must not be reintroduced. The browser application
is independently owned by `strata-ui`; do not add a `web/` directory, frontend
toolchain, or generated web bundle here.

## Cohesive files, not monoblocks

Each production file should have one nameable responsibility. File size is a
warning signal, but cohesion and dependency ownership decide the boundary.
Split an implementation when any of these becomes true:

- it mixes lifecycle, loading, execution, protocol, diagnostics, and
  hardware-specific concerns;
- a change in one concern routinely requires navigating unrelated sections;
- helpers need broad shared state only because several subsystems occupy one
  file;
- tests cannot exercise a component without constructing an unrelated
  runtime;
- the filename can no longer describe most of its contents.

Prefer real `.cpp`/`.cu` translation units with small private headers when the
pieces have stable interfaces. Use `.inc.cpp` or `.inc.cuh` fragments only
when code must share private implementation state or CUDA compilation context;
name each fragment after a responsibility and include it from exactly one
owning translation unit. An include fragment is an implementation boundary,
not permission to create a new miscellaneous bucket.

`kernels/cuda/backend.cu` is intentionally a small owner that assembles
responsibility-specific private fragments. `kernels/cuda/marlin_adapter.cu` is
separate because it owns the third-party boundary. New CUDA functionality must
join the relevant fragment or create a focused fragment/translation unit; it
must not rebuild `backend.cu` into a monolithic implementation. The same rule
applies to model runtimes: do not grow a large runtime file when a coherent
loader, attention path, MoE executor, or generation component can be owned and
tested separately.

Do not split mechanically into numbered files or fragments such as
`part1.inc.cpp`. A useful split has a domain name, a narrow interface, and a
clear owner.

## Third-party code

The default policy is no new runtime dependencies. Strata should prefer the
standard library, CUDA/runtime facilities already in use, and small first-party
implementations. An exception requires explicit maintainer agreement and a
concrete capability or maintenance benefit; convenience alone is insufficient.

If source vendoring is genuinely necessary:

1. Place the unmodified snapshot at `vendor/<project>/`, never under `apps/`,
   `include/`, or a Strata namespace.
2. Preserve its license and add provenance: upstream URL, pinned version or
   commit, local modifications, and update procedure.
3. Isolate third-party types, macros, and includes behind one narrow
   first-party adapter. Production code outside that adapter must speak Strata
   types.
4. Keep local patches explicit and minimal. Do not casually edit the snapshot
   until it is impossible to distinguish upstream code from ours.
5. Add the license/build implications to the review and verify both CUDA and
   non-CUDA configurations.

Marlin is the current exception. Its source lives in `vendor/marlin/`, and
`kernels/cuda/marlin_adapter.cu` is the only production source allowed to
include it directly.

## Experiments and promotion

The ignored workspace has a stable local layout:

```text
experiments/
├── docs/       hypotheses, experiment records, handovers, and issue notes
├── probes/     one-off C++/CUDA measurement programs
├── scripts/    experiment-only orchestration
├── results/    raw logs, profiles, traces, and tables
└── CMakeLists.txt  optional local targets, excluded from the default build
```

An experiment is promoted only when its behavior is intentionally supported.
Promotion means extracting cohesive first-party code into the correct product
layer, adding deterministic tests, registering production build sources, and
updating current product documentation. It does **not** mean moving the probe
wholesale into `apps/` or copying a research diary into `docs/`.

Product documentation may summarize measured evidence, but it must state the
hardware, operating point, and limitations. Keep rejected approaches and raw
evidence local. Never claim that a stub, interface, fixture, or planned design
is implemented behavior.

## Tests, builds, and documentation

Every behavior change needs a test at the lowest useful layer. Every bug fix
needs a regression test when practical. Tests must be deterministic and must
not depend on a developer's ignored models or experiment results unless they
skip explicitly and report why.

Before considering a product change complete, run:

```bash
make check
```

This runs the layer check, builds the project, checks archive dependencies, and
runs CTest. For changes to headers, build configuration, CUDA interfaces, or
portable C++ assumptions, also reproduce the GitHub CPU job from a fresh build
directory:

```bash
cmake -S . -B build-ci -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=OFF
python3 scripts/check_layers.py
cmake --build build-ci --parallel 2
python3 scripts/check_symbols.py build-ci
ctest --test-dir build-ci --output-on-failure
```

Run relevant native-CUDA and model equivalence gates when the required hardware
and checkpoints are available. A skipped optional gate is not a pass: report
the skip and its reason.

Update documentation in the same change when behavior, flags, file ownership,
supported models, or architecture boundaries change. Keep
`docs/current-architecture.md` factual and present-tense. Keep proposed designs
in explicitly labeled target documents. Operator commands belong in
`docs/models/` or the appropriate interface runbook.

## Change hygiene

- Keep changes focused; do not combine a feature with unrelated repository
  cleanup.
- Preserve user work and avoid destructive Git operations.
- Use names that describe the product concept consistently across source,
  header, test, target, and documentation paths.
- Do not commit generated output, checkpoints, local hardware notes, agent
  state, or experiment artifacts.
- Do not weaken an architecture or correctness gate merely to make a change
  pass.
- Review `git diff --check`, the final diff, and `git status` before commit.
- Report exactly which gates ran, which skipped, and which could not run.

## Completion checklist

Before handing off a change, answer all of these:

- Is every file in the directory that owns its responsibility?
- Are `apps/` thin, `docs/` product-only, and experiments still ignored?
- Are public-to-the-build headers under the correct `include/strata/` subtree
  and private headers near their implementation?
- Are all direct dependencies included explicitly?
- Does the six-layer dependency direction still hold with zero unexplained
  exceptions?
- Did a new CUDA API receive a matching non-CUDA stub?
- Is third-party code confined to `vendor/` behind an adapter?
- Did the change avoid creating or enlarging a monoblock?
- Do tests and documentation describe the behavior being shipped?
- Did the appropriate clean build and test gates pass?

If any answer is “no,” the change is not ready to merge.
