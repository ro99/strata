#!/usr/bin/env python3
"""Runs Phase 0's equivalence oracle against the committed Gemma 4 fixture
(brief 05, F2).

Before this script existed, `tests/fixtures/gemma4/layer-hash-trace.json`
was referenced by exactly zero files outside its own README -- no test, no
script, no CMake target, no make target. Re-verifying it meant a human
typing a diff against a 33 GB checkpoint by hand, and nothing failed if that
never happened. Every later phase reduces to an agent asserting nothing
changed unless something actually re-runs the comparison; this is that
something.

Registered as a ctest entry (see CMakeLists.txt) with SKIP_RETURN_CODE=77,
so `ctest`/`make check` reports it as skipped, not passed, when the pinned
checkpoint is absent -- the same distinction `strata-tests` already makes
for every checkpoint-gated unit test, extended to this oracle run. Also
runnable directly as `make check-equivalence` for a human who wants the
comparison without running the rest of the suite.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_EXIT_CODE = 77

MODEL_DIR = ROOT / "models" / "gemma4"
CHECKPOINT_INDEX = MODEL_DIR / "model.safetensors.index.json"
SINGLE_SHARD = MODEL_DIR / "model.safetensors"
W8A16_FIXTURE = ROOT / "tests" / "fixtures" / "gemma4" / "layer-hash-trace.json"
MXFP4_FIXTURE = (ROOT / "tests" / "fixtures" / "gemma4" /
                 "layer-hash-trace-mxfp4.json")
BINARY = Path(os.environ.get(
    "STRATA_GEMMA4_EQUIVALENCE_BINARY",
    ROOT / "build" / "strata-gemma4-run"))

PROMPT = "The capital of France is"
MAX_NEW = "8"


def skip(reason: str) -> int:
    print(f"SKIP gemma4 equivalence oracle: {reason}")
    return SKIP_EXIT_CODE


def fail(reason: str) -> int:
    print(f"FAIL gemma4 equivalence oracle: {reason}", file=sys.stderr)
    return 1


def first_divergence(reference: dict, actual: dict) -> str | None:
    ref_layers = reference["diagnostics"]["layer_hidden_hashes"]["entries"]
    act_layers = actual["diagnostics"]["layer_hidden_hashes"]["entries"]
    if len(ref_layers) != len(act_layers):
        return (f"layer_hidden_hashes entry count differs: "
                f"{len(ref_layers)} vs {len(act_layers)}")
    for index, (ref, act) in enumerate(zip(ref_layers, act_layers)):
        if ref != act:
            return (f"layer_hidden_hashes[{index}] differs: "
                    f"position={ref.get('position')} layer={ref.get('layer')} "
                    f"reference_hash={ref.get('bf16_hash')} "
                    f"actual_hash={act.get('bf16_hash')}")
    ref_trace = reference["diagnostics"]["layer_hidden_hashes"]["aggregate"]["trace_hash"]
    act_trace = actual["diagnostics"]["layer_hidden_hashes"]["aggregate"]["trace_hash"]
    if ref_trace != act_trace:
        return f"aggregate trace_hash differs: {ref_trace} vs {act_trace}"
    if reference.get("generated_token_ids") != actual.get("generated_token_ids"):
        return (f"generated_token_ids differs: "
                f"{reference.get('generated_token_ids')} vs "
                f"{actual.get('generated_token_ids')}")
    if reference.get("answer") != actual.get("answer"):
        return f"answer differs: {reference.get('answer')!r} vs {actual.get('answer')!r}"
    return None


def main() -> int:
    if not CHECKPOINT_INDEX.exists() and not SINGLE_SHARD.exists():
        return skip(f"pinned Gemma 4 checkpoint is absent "
                    f"(neither {CHECKPOINT_INDEX} nor {SINGLE_SHARD} exists)")
    if not BINARY.exists():
        return fail(f"{BINARY} does not exist -- build first "
                    f"(cmake --build build --parallel)")
    # The indexed checkpoint is the pinned W8A16 model; the lone-shard format
    # is the separately pinned MXFP4 model. Their quantized weights and thus
    # their layer hashes differ, so each needs its own oracle. Index precedence
    # mirrors load_safetensors_index when both files happen to exist.
    fixture = W8A16_FIXTURE if CHECKPOINT_INDEX.exists() else MXFP4_FIXTURE
    if not fixture.exists():
        return fail(f"{fixture} does not exist -- nothing to compare against")

    environment = os.environ.copy()
    # Equivalence is an architecture oracle, not a register-fed oracle. Keep
    # the canonical scalar layout so this gate remains stable across the A/B.
    environment["STRATA_REGFED_MATMUL"] = "0"
    command = [str(BINARY), "--model", str(MODEL_DIR), "--prompt", PROMPT,
               "--max-new", MAX_NEW, "--temperature", "0",
               "--layer-hash-trace", "--json"]
    if devices := os.environ.get("STRATA_GEMMA4_EQUIVALENCE_DEVICES"):
        command.extend(["--devices", devices])
    proc = subprocess.run(
        command,
        capture_output=True, text=True, env=environment)
    if proc.returncode != 0:
        return fail(f"{BINARY} exited {proc.returncode}\nstderr:\n{proc.stderr}")

    try:
        actual = json.loads(proc.stdout)
    except json.JSONDecodeError as error:
        return fail(f"could not parse {BINARY} output as JSON: {error}")

    reference = json.loads(fixture.read_text(encoding="utf-8"))
    divergence = first_divergence(reference, actual)
    if divergence is not None:
        return fail(f"oracle disagrees with the committed fixture: {divergence}")

    entry_count = len(reference["diagnostics"]["layer_hidden_hashes"]["entries"])
    print(f"PASS gemma4 equivalence oracle: {entry_count} layer hashes, "
          f"generated_token_ids and answer all match "
          f"{fixture.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
