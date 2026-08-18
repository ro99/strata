"""Recorded exceptions to Phase 1's layering enforcement (docs/experiments/0121).

Every entry here is a *deferred* violation, not an accepted one: the checks
that consume this file (`check_layers.py` for includes, `check_symbols.py`
for the link-level symbol graph) still find the underlying defect, still
classify it as real, and only then treat it as forgiven because it appears
in one of the `matches` (include-level) or `symbol_objects` (link-level)
lists below. Properties both consumers enforce, not just document:

- If a listed `(file, included_header)` pair, or a listed `symbol_objects`
  entry, stops corresponding to a real violation -- the include is removed,
  the header moves, the object file no longer references anything upward --
  the check treats that as a hard error, not a silent pass. A stale
  exception is how a boundary quietly reopens; the check must notice the
  boundary was already crossed by something else, not assume the exception
  is still needed.
- Both checks always print every exception they applied, by name, with the
  violations it covered. Nothing here is invisible to a normal run.
- An exception whose `expiry_phase` has arrived or passed fails the run,
  loudly, rather than continuing to pass silently. See CURRENT_PHASE below.

Three entries, all with the same expiry: phase 4, when each model's own
fan-out work is already touching the file in question, and the underlying fix
in every case needs a registry or a generalized contract designed once with
all six models in view -- not designed now against one or two and stretched
to fit the rest later. See docs/experiments/0121 for the full reasoning.
"""

from __future__ import annotations

# Bump this when the programme actually enters the next phase -- not before,
# and not as a way to make an inconvenient exception pass. Any EXCEPTIONS
# entry whose expiry_phase <= CURRENT_PHASE fails the run (see
# check_layers.py / check_symbols.py's expiry check) until it is retired or
# explicitly renewed with a new expiry and a stated reason for the renewal.
# Phase 1 remediation (brief 05) does not advance this: brief 05 is still
# phase 1, phase 2's start is the orchestrator's decision, not a side effect
# of a lint fix landing.
CURRENT_PHASE = 1

EXCEPTIONS = [
]
