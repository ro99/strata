#!/usr/bin/env python3
"""Phase 1 layering enforcement (brief 01, task 1).

Two checks, both static (no build required):

1. Include-graph direction: for every `#include "strata/X.hpp"` (or a
   same-directory local include such as `"inkling_unicode.hpp"`) found in a
   file assigned to target A, the included header's owning target B must not
   rank above A in LAYER_ORDER. A lower target including a higher one is a
   layering violation.
2. Model-identifier leakage: no file assigned to strata_platform,
   strata_device, strata_kernels, or strata_engine may `#include` a header
   whose path contains one of MODEL_IDENTIFIERS as a substring.

Both checks operate on *direct* includes only, parsed by regex, not a real
preprocessor. That is a deliberate scope choice for this unit: every
violation found so far -- the four the review predicted and the ones this
script adds -- is a direct include, so a one-hop check is sufficient and
avoids building a real transitive resolver before anyone has decided whether
one is needed.

TARGETS below mirrors the `add_library` source lists in CMakeLists.txt. It is
the single source of truth this script reads from; CMakeLists.txt is parsed,
not duplicated, so the two cannot drift silently.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Declared downward order: index i may depend on any index < i, never on a
# higher index. This is the brief's proposed ordering, taken as given for
# this unit -- see the report for why nothing found here contradicts it.
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

# Headers with no same-basename .cpp in any target's source list need an
# explicit owner. Every one of these is here because it was checked by hand
# against what it actually declares -- see the report for cuda_backend.hpp,
# model.hpp and model_adapter.hpp specifically, which are the surprising
# ones (generic-sounding names, model-specific or device-specific content).
HEADER_OVERRIDES = {
    "result.hpp": "strata_platform",
    "types.hpp": "strata_platform",
    "kernel_abi.h": "strata_platform",
    "cuda_backend.hpp": "strata_device",
    "glm_int4.hpp": "strata_kernels",
    "model_adapter.hpp": "strata_models",
    "deepseek_host_expert.hpp": "strata_models",
    "dsv4_rank_local_layer_executor.hpp": "strata_models",
    # model.hpp has a same-basename model.cpp (assigned to strata_models
    # below), so it does not need an override; listed here only as a note
    # that it was checked, not assumed, given the misleading name.
}

# Local (non "strata/"-prefixed) headers under src/ with no public counterpart.
LOCAL_HEADER_OVERRIDES = {
    "json_cursor.hpp": "strata_platform",
    "checkpoint_common.hpp": "strata_platform",
    "cuda_stats_delta.hpp": "strata_device",
    "inkling_unicode.hpp": "strata_models",
    "laguna_unicode.hpp": "strata_models",
}

INCLUDE_RE = re.compile(r'#include\s*"(?:strata/)?([A-Za-z0-9_./]+\.(?:hpp|h))"')
ADD_LIBRARY_RE = re.compile(
    r'add_library\(\s*(strata_\w+)\s+STATIC(.*?)\n\)', re.DOTALL)


def parse_targets(cmake_text: str) -> dict[str, list[str]]:
    targets: dict[str, list[str]] = {}
    for match in ADD_LIBRARY_RE.finditer(cmake_text):
        name = match.group(1)
        if name not in LAYER_ORDER:
            continue
        body = match.group(2)
        files = re.findall(r'([\w./-]+\.(?:cpp|cu))', body)
        targets[name] = files
    return targets


def owning_target(header_name: str, cpp_to_target: dict[str, str]) -> str | None:
    base = Path(header_name).name
    stem = Path(base).stem
    if base in HEADER_OVERRIDES:
        return HEADER_OVERRIDES[base]
    if base in LOCAL_HEADER_OVERRIDES:
        return LOCAL_HEADER_OVERRIDES[base]
    return cpp_to_target.get(stem)


def main() -> int:
    cmake_text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    targets = parse_targets(cmake_text)
    if not targets:
        print("check-layers: no strata_* add_library targets found in "
              "CMakeLists.txt -- has the target split landed?", file=sys.stderr)
        return 2

    cpp_to_target: dict[str, str] = {}
    file_to_target: dict[str, str] = {}
    header_files: dict[str, list[str]] = {t: [] for t in targets}
    for target, files in targets.items():
        for rel in files:
            stem = Path(rel).stem
            cpp_to_target[stem] = target
            file_to_target[rel] = target
            # Each .cpp's paired public header (include/strata/<stem>.hpp) is
            # part of the same target's surface and must be scanned too --
            # a header can carry an upward include the .cpp itself never
            # repeats (that is exactly how checkpoint.hpp's violation hides:
            # checkpoint.cpp only includes its own header, which is where
            # glm_manifest.hpp and cuda_backend.hpp actually are).
            header = ROOT / "include" / "strata" / f"{stem}.hpp"
            if header.exists():
                header_files[target].append(str(header.relative_to(ROOT)))
    for header_name, target in HEADER_OVERRIDES.items():
        rel = f"include/strata/{header_name}"
        if (ROOT / rel).exists() and rel not in header_files.get(target, []):
            header_files.setdefault(target, []).append(rel)

    violations: list[str] = []
    identifier_hits: list[str] = []

    for target, files in targets.items():
        rank = LAYER_ORDER.index(target)
        for rel in files + header_files.get(target, []):
            path = ROOT / rel
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for inc_match in INCLUDE_RE.finditer(text):
                included = inc_match.group(1)
                owner = owning_target(included, cpp_to_target)
                if owner is not None and owner in LAYER_ORDER:
                    owner_rank = LAYER_ORDER.index(owner)
                    if owner_rank > rank:
                        violations.append(
                            f"{rel} ({target}) includes {included} "
                            f"(owned by {owner}) -- upward dependency")
                if target in NO_MODEL_TARGETS:
                    lowered = included.lower()
                    if any(ident in lowered for ident in MODEL_IDENTIFIERS):
                        identifier_hits.append(
                            f"{rel} ({target}) includes {included} -- "
                            f"model identifier in a no-model target")

    print(f"check-layers: {sum(len(v) for v in targets.values())} files "
          f"across {len(targets)} targets")
    print()
    print(f"== include-graph direction ({len(violations)} violation(s)) ==")
    for line in violations:
        print(f"  {line}")
    print()
    print(f"== model-identifier leakage ({len(identifier_hits)} violation(s)) ==")
    for line in identifier_hits:
        print(f"  {line}")

    total = len(violations) + len(identifier_hits)
    print()
    print(f"check-layers: {total} total violation(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
