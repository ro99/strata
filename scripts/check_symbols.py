#!/usr/bin/env python3
"""Phase 1 layering enforcement, symbol level (brief 05, F1).

`check_layers.py` checks `#include` edges. It cannot see a link edge that
exists with no corresponding include -- exactly the shape an independent
cold review found: `src/placement_model.cpp` (`strata_engine`) calls
`GlmCheckpointReader::open`, `Dsv4CheckpointReader::open`,
`Gemma4CheckpointReader::find`, `InklingCheckpointReader::*`,
`KimiCheckpointReader::open`, and the six `*_spec()` functions -- all defined
in `strata_models` -- through headers whose *declarations* the include-graph
check already flags (that's B2), but the *link* dependency is a separate,
stronger claim than "this file includes a header it should not": it is
"this object file's undefined symbols are only satisfiable by an archive
above it in the declared order", which is exactly what the six-target split
was supposed to make impossible to hide. It didn't, because every executable
links through the `strata_core` INTERFACE alias, which puts every archive on
one link line regardless of declared per-target order, and GNU `ld`'s
single-pass resolution silently absorbs the upward reference. That is the
identical mechanism `docs/experiments/0121` names for Cause B ("luck, not
correctness"), still present one tier up, undetected until now because
nothing tested any *single* target's link closure in isolation.

This script does: for each of the six `strata_*` static archives (already
built -- this check needs `build/`, unlike `check_layers.py`, which is
purely static), extract the defined and undefined global symbols with `nm`.
For every ordered pair (lower tier, higher tier), any symbol undefined in the
lower archive but defined in the higher one is an upward link dependency --
a real one, proven by the linker's own symbol table, not inferred from an
include.

Recorded exceptions are shared with `check_layers.py`: entries in
`layer_exceptions.py` may carry a `symbol_objects` list of `.o` basenames.
An upward symbol is forgiven only if the `.o` file that references it (found
via `nm`'s per-object listing) is named in some exception's `symbol_objects`
-- same two properties as the include-level exceptions: a listed object that
turns out not to reference anything upward fails loudly as stale, and every
applied exception is printed with exactly what it covered.

Both checks (`check_layers.py`, this one) plus the expiry check use the same
`CURRENT_PHASE`/`expiry_phase` mechanism from `layer_exceptions.py`.

LIMITS -- read the score as "0 upward references expressible as undefined
symbols", which is weaker than "0 upward dependencies". Two shapes are
invisible here by construction:

  1. A function-pointer registry. strata_engine holds a null pointer that
     strata_models fills at static-init time (register_model / find_model, and
     register_placement_planner / plan_model_placement). The runtime
     dependency is unchanged -- link strata_engine without strata_models and
     every plan fails with "no placement planner is registered" -- but there
     is no undefined symbol for nm to report. This is deliberate dependency
     inversion and it is how the last recorded symbol exception was retired;
     it is recorded here so nobody reads the zero as stronger than it is.
     It also converts a link failure into a runtime one.

  2. Inline and template code in headers. An upward dependency consumed
     entirely through inline members leaves no undefined symbol in the lower
     archive. Combined with an unowned header, which check_layers.py does not
     scan, such a dependency is invisible to both checks at once -- which is
     why check_layers.py now fails on any header it cannot classify.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_layers import LAYER_ORDER, check_expiry, parse_targets  # noqa: E402
from layer_exceptions import CURRENT_PHASE, EXCEPTIONS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

NM_LINE_RE = re.compile(
    r'^(?P<archive>.+?):(?P<object>[^:/]+\.o):\s*'
    r'(?:[0-9a-fA-F]*\s+)?(?P<type>[A-Za-z])\s+(?P<symbol>\S+)\s*$')


def run_nm(archive: Path) -> list[tuple[str, str, str]]:
    """Returns (object_basename, symbol_type, symbol_name) for every symbol
    nm reports in the archive, using --print-file-name so each line is
    traceable back to the exact .o that defines or references it."""
    proc = subprocess.run(
        ["nm", "--print-file-name", str(archive)],
        capture_output=True, text=True, check=True)
    entries = []
    for line in proc.stdout.splitlines():
        match = NM_LINE_RE.match(line)
        if not match:
            continue
        entries.append((match.group("object"), match.group("type"),
                        match.group("symbol")))
    return entries


def build_exception_index() -> dict[str, list[dict]]:
    """object basename -> exceptions whose symbol_objects list it."""
    index: dict[str, list[dict]] = {}
    for exception in EXCEPTIONS:
        for obj in exception.get("symbol_objects", []):
            index.setdefault(obj, []).append(exception)
    return index


def main() -> int:
    build_dir = ROOT / "build"
    cmake_text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    targets = parse_targets(cmake_text)
    if not targets:
        print("check-symbols: no strata_* add_library targets found in "
              "CMakeLists.txt", file=sys.stderr)
        return 2

    archives: dict[str, Path] = {}
    for target in targets:
        archive = build_dir / f"lib{target}.a"
        if not archive.exists():
            print(f"check-symbols: {archive} does not exist -- build first "
                  f"(this check reads real archives, it cannot run "
                  f"statically)", file=sys.stderr)
            return 2
        archives[target] = archive

    # Per target: defined-symbol set (global, any section), and a map of
    # undefined symbol -> set of .o basenames that reference it undefined.
    defined: dict[str, set[str]] = {}
    undefined_refs: dict[str, dict[str, set[str]]] = {}
    for target, archive in archives.items():
        entries = run_nm(archive)
        defined[target] = {sym for obj, typ, sym in entries if typ != "U"}
        refs: dict[str, set[str]] = {}
        for obj, typ, sym in entries:
            if typ == "U":
                refs.setdefault(sym, set()).add(obj)
        undefined_refs[target] = refs

    exception_index = build_exception_index()
    consumed_objects: set[str] = set()

    violations: list[str] = []
    excepted: list[str] = []

    for lower in LAYER_ORDER:
        lower_rank = LAYER_ORDER.index(lower)
        for higher in LAYER_ORDER:
            higher_rank = LAYER_ORDER.index(higher)
            if higher_rank <= lower_rank:
                continue
            upward_symbols = set(undefined_refs[lower]) & defined[higher]
            if not upward_symbols:
                continue
            # Group by the referencing .o so the report -- and the exception
            # matching -- is actionable, not a wall of mangled names.
            objects_hit: dict[str, set[str]] = {}
            for sym in upward_symbols:
                for obj in undefined_refs[lower][sym]:
                    objects_hit.setdefault(obj, set()).add(sym)
            for obj, syms in sorted(objects_hit.items()):
                exceptions_for_obj = exception_index.get(obj, [])
                if exceptions_for_obj:
                    consumed_objects.add(obj)
                    names = ", ".join(sorted({e["name"] for e in exceptions_for_obj}))
                    excepted.append(
                        f"{lower}/{obj} -> {higher} ({len(syms)} symbol(s)) "
                        f"-- excepted by [{names}]")
                else:
                    violations.append(
                        f"{lower}/{obj} -> {higher} ({len(syms)} symbol(s), "
                        f"e.g. {sorted(syms)[0]})")

    stale: list[str] = []
    for exception in EXCEPTIONS:
        for obj in exception.get("symbol_objects", []):
            if obj not in consumed_objects:
                stale.append(
                    f"[{exception['name']}] symbol_objects entry "
                    f"'{obj}' matches no real upward symbol reference")

    expired = check_expiry()

    print(f"check-symbols: {len(archives)} archives inspected via nm "
          f"(CURRENT_PHASE={CURRENT_PHASE})")
    print()
    print(f"== exceptions applied ({len(excepted)}) ==")
    for line in excepted:
        print(f"  {line}")
    print()
    print(f"== upward symbol references ({len(violations)}) ==")
    for line in violations:
        print(f"  {line}")
    if stale:
        print()
        print(f"== STALE SYMBOL EXCEPTIONS ({len(stale)}) ==")
        for line in stale:
            print(f"  {line}")
    if expired:
        print()
        print(f"== EXPIRED EXCEPTIONS ({len(expired)}) ==")
        for line in expired:
            print(f"  {line}")

    total = len(violations) + len(stale) + len(expired)
    print()
    print(f"check-symbols: {total} total violation(s) "
          f"({len(excepted)} excepted, {len(stale)} stale, "
          f"{len(expired)} expired)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
