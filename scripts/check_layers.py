#!/usr/bin/env python3
"""Phase 1 layering enforcement (brief 01, task 1; exceptions brief 03;
device-tier resolution, unknown-target and expiry enforcement brief 05).

Two checks, both static (no build required -- see check_symbols.py for the
link-level companion, which does need a build):

1. Include-graph direction: for every `#include "strata/X.hpp"` or
   `#include <strata/X.hpp>` (or a same-directory local include such as
   `"inkling_unicode.hpp"`) found in a file assigned to target A, the
   included header's owning target B must not rank above A in LAYER_ORDER.
   A lower target including a higher one is a layering violation.
2. Model-identifier leakage, two forms:
   a. No file assigned to strata_platform, strata_device, strata_kernels, or
      strata_engine may `#include` a header whose path contains one of
      MODEL_IDENTIFIERS as a substring, *unless* the included header is
      owned by the same target doing the including (owner == target, the
      direction check's own ownership resolution, reused rather than
      re-derived).
   b. No file *assigned to* one of those four targets may itself be named
      after a model identifier unless it is explicitly allowlisted in
      MODEL_NAMED_FILES_ALLOWED (glm_int4.*: an INT4 codec kernel named
      after its first user, not GLM-specific content -- review finding C6,
      rename deferred to phase 8). A self-contained
      kernels/cpu/deepseek_moe_kernel.{hpp,cpp} that never includes anything
      upward would pass check (a) trivially (owner == target for its own
      header), so without check (b) a genuinely model-specific file could
      sit in a no-model target undetected forever. This is the fix for a
      cold review's finding that the identifier check's docstring claimed
      more than its code did; now both say the same thing.

Both checks operate on *direct* includes only, parsed by regex, not a real
preprocessor. That is a deliberate scope choice: every violation found so
far is a direct include, so a one-hop check is sufficient and avoids
building a real transitive resolver before anyone has decided one is needed.

TARGETS mirrors the `add_library` source lists in CMakeLists.txt, resolving
`${VAR}` source-list variables (both branches of a `set()` are kept, not
just whichever CUDA availability picks on this machine -- see
resolve_variable_sources) and `target_sources(target PRIVATE ...)` calls
(the NCCL-conditional executor). CMakeLists.txt is parsed, not duplicated,
so the two cannot drift silently. Every declared `strata_*` STATIC target
must both appear in LAYER_ORDER and resolve to a non-empty file list, or the
run fails loudly instead of silently scanning nothing for that tier -- a
cold review found strata_device contributed 0 of the 61 files a prior
version scanned, entirely unremarked in that version's own summary line.

Recorded exceptions (scripts/layer_exceptions.py) are matched against real
violations, not exempted from being found in the first place -- see that
file's own docstring for the properties this enforces: a listed (file,
header) pair that stops being a real violation fails the run instead of
being silently accepted, every applied exception is printed by name with
what it covered, and an exception whose expiry_phase has arrived fails the
run rather than continuing to pass.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer_exceptions import CURRENT_PHASE, EXCEPTIONS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Declared downward order: index i may depend on any index < i, never on a
# higher index. This is the brief's proposed ordering, taken as given for
# this unit -- see docs/experiments/0121 for why nothing found contradicts it.
LAYER_ORDER = [
    "strata_platform",
    "strata_device",
    "strata_kernels",
    "strata_engine",
    "strata_models",
    "strata_app",
]

MODEL_IDENTIFIERS = (
    "dsv4", "deepseek", "glm", "gemma", "kimi", "laguna", "inkling",
)
NO_MODEL_TARGETS = {
    "strata_platform", "strata_device", "strata_kernels", "strata_engine",
}

# Files deliberately named after a model despite living in a no-model
# target, checked by hand against what they actually declare -- not GLM- or
# DeepSeek-specific content, a codec/format named after its first user.
MODEL_NAMED_FILES_ALLOWED = {
    "glm_int4.hpp": "INT4 group-128 codec kernel, not GLM-specific (C6, "
                    "rename deferred to phase 8)",
    "glm_int4.cpp": "same as glm_int4.hpp",
}

# Headers with no same-basename .cpp in any target's source list need an
# explicit owner. Every one of these is here because it was checked by hand
# against what it actually declares -- see docs/experiments/0121 for
# cuda_backend.hpp, model.hpp and model_adapter.hpp specifically, which are
# the surprising ones (generic-sounding names, model-specific or
# device-specific content).
HEADER_OVERRIDES = {
    "result.hpp": "strata_platform",
    "types.hpp": "strata_platform",
    "kernel_abi.h": "strata_platform",
    "cuda_backend.hpp": "strata_device",
    "glm_int4.hpp": "strata_kernels",
    "model_adapter.hpp": "strata_models",
    "deepseek_host_expert.hpp": "strata_models",
    "dsv4_rank_local_layer_executor.hpp": "strata_models",
    # The Phase 4 seam and its two companions. Without these three the
    # ownership map cannot classify them, and an unowned header is not
    # scanned at all -- see the unowned-header check below, which now makes
    # that a hard failure rather than a silent hole.
    "model_executor.hpp": "strata_engine",
    "lru_residency.hpp": "strata_engine",
    "quantization.hpp": "strata_platform",
    # model.hpp and checkpoint.hpp each have a same-basename .cpp assigned to
    # strata_models below, so neither needs an override; listed here only as
    # a note that both were checked, not assumed, given the misleading names.
}

# Local (non "strata/"-prefixed) headers under src/ with no public counterpart.
LOCAL_HEADER_OVERRIDES = {
    "json_cursor.hpp": "strata_platform",
    "checkpoint_common.hpp": "strata_platform",
    "cuda_stats_delta.hpp": "strata_device",
    "inkling_unicode.hpp": "strata_models",
    "laguna_unicode.hpp": "strata_models",
    "executor_support.hpp": "strata_models",
}

INCLUDE_RE = re.compile(
    r'#include\s*[<"](?:strata/)?([A-Za-z0-9_./]+\.(?:hpp|h))[>"]')
ADD_LIBRARY_RE = re.compile(
    r'add_library\(\s*(strata_\w+)\s+STATIC(.*?)\n\)', re.DOTALL)
SET_VAR_RE = re.compile(r'set\(\s*(\w+)\s+([\w./-]+\.(?:cpp|cu))\s*\)')
TARGET_SOURCES_RE = re.compile(
    r'target_sources\(\s*(\w+)\s+PRIVATE\s*\n?\s*([\w./-]+\.(?:cpp|cu))')
SOURCE_FILE_RE = re.compile(r'([\w./-]+\.(?:cpp|cu))')
VAR_REF_RE = re.compile(r'\$\{(\w+)\}')


def resolve_variable_sources(cmake_text: str) -> dict[str, list[str]]:
    """VAR -> every file any set(VAR file) assignment gives it.

    Both branches of an if/else are kept (e.g. STRATA_CUDA_SOURCE resolves
    to both kernels/cuda/backend.cu and src/cuda_backend_stub.cpp) rather
    than guessing which one this machine's CUDA availability would pick --
    a lint that only sees the branch active on the machine that happens to
    run it is exactly the kind of gap this unit exists to close.
    """
    variables: dict[str, list[str]] = {}
    for match in SET_VAR_RE.finditer(cmake_text):
        variables.setdefault(match.group(1), []).append(match.group(2))
    return variables


def resolve_target_sources(cmake_text: str) -> dict[str, list[str]]:
    """target -> files added via target_sources(target PRIVATE ...) calls,
    which add_library's own body never sees (the NCCL-conditional
    dsv4_rank_local_layer_executor.cu, added to strata_models)."""
    extra: dict[str, list[str]] = {}
    for match in TARGET_SOURCES_RE.finditer(cmake_text):
        extra.setdefault(match.group(1), []).append(match.group(2))
    return extra


def parse_targets(cmake_text: str) -> dict[str, list[str]]:
    variables = resolve_variable_sources(cmake_text)
    extra_sources = resolve_target_sources(cmake_text)

    targets: dict[str, list[str]] = {}
    unrecognized: list[str] = []
    for match in ADD_LIBRARY_RE.finditer(cmake_text):
        name = match.group(1)
        body = match.group(2)
        files: list[str] = []
        for var_match in VAR_REF_RE.finditer(body):
            files.extend(variables.get(var_match.group(1), []))
        files.extend(SOURCE_FILE_RE.findall(VAR_REF_RE.sub('', body)))
        files.extend(extra_sources.get(name, []))
        if name not in LAYER_ORDER:
            # A 7th strata_* target with no declared rank. Silently skipping
            # it (the previous behaviour) is exactly the "does not appear at
            # all, unremarked" failure mode this unit exists to close, so
            # this is collected and reported as a hard error, not dropped.
            unrecognized.append(name)
            continue
        targets[name] = files

    if unrecognized:
        raise SystemExit(
            "check-layers: add_library target(s) not in LAYER_ORDER: "
            + ", ".join(sorted(unrecognized)) +
            " -- add to LAYER_ORDER (with a declared rank) or this check "
            "silently scans nothing for them.")

    empty = [name for name in LAYER_ORDER if name in targets and not targets[name]]
    if empty:
        raise SystemExit(
            "check-layers: target(s) with an unresolved or empty source "
            "list: " + ", ".join(empty) +
            " -- a ${VAR} source list variable or target_sources() call "
            "could not be resolved. Fix resolve_variable_sources/"
            "resolve_target_sources rather than let this pass silently.")

    missing = [name for name in LAYER_ORDER if name not in targets]
    if missing:
        raise SystemExit(
            "check-layers: LAYER_ORDER names target(s) with no add_library "
            "in CMakeLists.txt: " + ", ".join(missing))

    return targets


def owning_target(header_name: str, cpp_to_target: dict[str, str]) -> str | None:
    base = Path(header_name).name
    stem = Path(base).stem
    if base in HEADER_OVERRIDES:
        return HEADER_OVERRIDES[base]
    if base in LOCAL_HEADER_OVERRIDES:
        return LOCAL_HEADER_OVERRIDES[base]
    return cpp_to_target.get(stem)


def build_exception_index() -> dict[tuple[str, str], list[dict]]:
    """Maps (file, included_header) -> the exception entries that cover it.

    A pair could in principle be listed by more than one entry; kept as a
    list so that is visible rather than silently overwritten.
    """
    index: dict[tuple[str, str], list[dict]] = {}
    for exception in EXCEPTIONS:
        for file_rel, included in exception["matches"]:
            index.setdefault((file_rel, included), []).append(exception)
    return index


def unowned_headers(cpp_to_target: dict[str, str]) -> list[str]:
    """Every header the ownership map cannot classify.

    An unowned header is worse than a violation: `owning_target` returns None,
    so no include of it can be a direction hit, AND the header itself is never
    added to any target's scan set, so its own includes are never read. A cold
    review demonstrated the hole by adding a DeepSeek include to
    model_executor.hpp and an include of that from strata_engine -- the run
    reported a clean tree.

    This is the same failure the docstring above records for `strata_device`
    contributing 0 files to an earlier version of this script. Treat it the
    same way: fail loudly, never scan nothing.
    """
    problems: list[str] = []
    for header in sorted(ROOT.glob("include/strata/**/*.hpp")):
        if owning_target(header.name, cpp_to_target) is None:
            problems.append(
                f"include/strata/{header.name} is owned by no target -- add it "
                f"to HEADER_OVERRIDES; until then its includes are unscanned")
    for header in sorted(ROOT.glob("src/**/*.hpp")):
        if owning_target(header.name, cpp_to_target) is None:
            problems.append(
                f"{header.relative_to(ROOT)} is owned by no target -- add it to "
                f"LOCAL_HEADER_OVERRIDES; until then its includes are unscanned")
    return problems


def check_expiry(current_phase: int = CURRENT_PHASE) -> list[str]:
    """Exceptions whose expiry_phase has arrived or passed. Shared with
    check_symbols.py so both checks fail identically on an expired entry
    regardless of which one happens to run first."""
    return [
        f"[{exception['name']}] expired: recorded expiry_phase "
        f"{exception['expiry_phase']}, CURRENT_PHASE is {current_phase} -- "
        f"retire this exception (fix the underlying code) or renew it with "
        f"a new expiry_phase and a stated reason for the renewal"
        for exception in EXCEPTIONS
        if exception["expiry_phase"] <= current_phase
    ]


def main() -> int:
    cmake_text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    targets = parse_targets(cmake_text)

    cpp_to_target: dict[str, str] = {}
    header_files: dict[str, list[str]] = {t: [] for t in targets}
    for target, files in targets.items():
        for rel in files:
            stem = Path(rel).stem
            cpp_to_target[stem] = target
            # Each .cpp's paired public header (include/strata/<stem>.hpp) is
            # part of the same target's surface and must be scanned too --
            # a header can carry an upward include the .cpp itself never
            # repeats (that is exactly how checkpoint.hpp's violation used to
            # hide: checkpoint.cpp only includes its own header, which is
            # where glm_manifest.hpp and cuda_backend.hpp actually were).
            header = ROOT / "include" / "strata" / f"{stem}.hpp"
            if header.exists():
                header_files[target].append(str(header.relative_to(ROOT)))
    for header_name, target in HEADER_OVERRIDES.items():
        rel = f"include/strata/{header_name}"
        if (ROOT / rel).exists() and rel not in header_files.get(target, []):
            header_files.setdefault(target, []).append(rel)

    exception_index = build_exception_index()
    consumed: set[tuple[str, str]] = set()

    violations: list[str] = []
    identifier_hits: list[str] = []
    self_named_hits: list[str] = []
    excepted: list[str] = []

    for target, files in targets.items():
        rank = LAYER_ORDER.index(target)
        all_files = files + header_files.get(target, [])

        if target in NO_MODEL_TARGETS:
            for rel in all_files:
                base = Path(rel).name
                if base in MODEL_NAMED_FILES_ALLOWED:
                    continue
                if any(ident in base.lower() for ident in MODEL_IDENTIFIERS):
                    self_named_hits.append(
                        f"{rel} ({target}) is itself named after a model "
                        f"and is not in MODEL_NAMED_FILES_ALLOWED")

        for rel in all_files:
            path = ROOT / rel
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for inc_match in INCLUDE_RE.finditer(text):
                included = inc_match.group(1)
                key = (rel, included)
                is_exception = key in exception_index

                owner = owning_target(included, cpp_to_target)
                direction_hit = (
                    owner is not None and owner in LAYER_ORDER and
                    LAYER_ORDER.index(owner) > rank)
                identifier_hit = (
                    target in NO_MODEL_TARGETS and owner != target and
                    any(ident in included.lower() for ident in MODEL_IDENTIFIERS))

                if not (direction_hit or identifier_hit):
                    continue

                if is_exception:
                    consumed.add(key)
                    names = ", ".join(sorted({e["name"] for e in exception_index[key]}))
                    excepted.append(f"{rel} ({target}) includes {included} "
                                    f"-- excepted by [{names}]")
                    continue

                if direction_hit:
                    violations.append(
                        f"{rel} ({target}) includes {included} "
                        f"(owned by {owner}) -- upward dependency")
                if identifier_hit:
                    identifier_hits.append(
                        f"{rel} ({target}) includes {included} -- "
                        f"model identifier in a no-model target")

    stale: list[str] = []
    for exception in EXCEPTIONS:
        for file_rel, included in exception["matches"]:
            if (file_rel, included) not in consumed:
                stale.append(
                    f"[{exception['name']}] no longer matches a real "
                    f"violation: {file_rel} includes {included}")

    expired = check_expiry()
    unowned = unowned_headers(cpp_to_target)

    print(f"check-layers: {sum(len(v) for v in targets.values())} files "
          f"across {len(targets)} targets "
          f"({', '.join(f'{t}={len(f)}' for t, f in targets.items())})")
    print()
    print(f"== exceptions applied ({len(excepted)} entries, "
          f"{len(EXCEPTIONS)} recorded, CURRENT_PHASE={CURRENT_PHASE}) ==")
    if not EXCEPTIONS:
        print("  (none recorded)")
    for exception in EXCEPTIONS:
        covered = [line for line in excepted if f"[{exception['name']}]" in line
                   or f", {exception['name']}]" in line
                   or f"[{exception['name']}," in line]
        print(f"  {exception['name']} (expires phase {exception['expiry_phase']}, "
              f"{len(covered)}/{len(exception['matches'])} matches consumed)")
        for line in covered:
            print(f"    {line}")
    print()
    print(f"== include-graph direction ({len(violations)} violation(s)) ==")
    for line in violations:
        print(f"  {line}")
    print()
    print(f"== model-identifier leakage, included headers ({len(identifier_hits)} violation(s)) ==")
    for line in identifier_hits:
        print(f"  {line}")
    print()
    print(f"== model-identifier leakage, self-named files ({len(self_named_hits)} violation(s)) ==")
    for line in self_named_hits:
        print(f"  {line}")
    if stale:
        print()
        print(f"== STALE EXCEPTIONS ({len(stale)}) -- fix the exception or the "
              f"code, do not ignore ==")
        for line in stale:
            print(f"  {line}")
    if expired:
        print()
        print(f"== EXPIRED EXCEPTIONS ({len(expired)}) ==")
        for line in expired:
            print(f"  {line}")
    if unowned:
        print()
        print(f"== UNOWNED HEADERS ({len(unowned)}) -- these are not scanned "
              f"at all, which is worse than a violation ==")
        for line in unowned:
            print(f"  {line}")

    total = (len(violations) + len(identifier_hits) + len(self_named_hits) +
             len(stale) + len(expired) + len(unowned))
    print()
    print(f"check-layers: {total} total violation(s) "
          f"({len(excepted)} excepted, {len(stale)} stale, "
          f"{len(expired)} expired)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
