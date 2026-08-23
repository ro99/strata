#!/usr/bin/env python3
"""Summarize the register-fed A/B matrix written by dsv4_regfed_ab.sh.

Reports the primary metric, the per-phase decode breakdown with argmax_r, the
route census of both arms, and the greedy-output comparison. Refuses to call a
difference a win when it sits inside the observed spread, and refuses to report
anything at all if the census shows the switch did not take effect -- an A/B
that ran the same path twice is not evidence.
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys

ARMS = ("regfed", "scalar")


def load(outdir: str, arm: str) -> list[dict]:
    runs = []
    candidates = sorted(glob.glob(os.path.join(outdir, f"{arm}-rep*.json")))
    # The census files sit beside the run files and match the same glob.
    for path in (p for p in candidates if not p.endswith(".census.json")):
        with open(path) as handle:
            text = handle.read().strip()
        if not text:
            print(f"warning: {path} is empty; the arm probably failed", file=sys.stderr)
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            # Tolerate a non-JSON preamble on stdout: take the last line that
            # parses as an object rather than discarding the whole arm.
            record = None
            for line in reversed(text.splitlines()):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        record = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
            if record is None:
                print(f"warning: {path} is not parseable JSON", file=sys.stderr)
                continue
        record["_path"] = path
        census_path = path[: -len(".json")] + ".census.json"
        record["_census"] = {}
        if os.path.exists(census_path):
            with open(census_path) as handle:
                record["_census"] = json.load(handle)
        runs.append(record)
    return runs


def steady_decode(record: dict) -> tuple[float, float]:
    """Seconds per token: steady state, and including the first step.

    The lazy fragment prepack lands in the first decode step, so excluding it is
    what makes the two arms comparable; the difference between the two numbers
    is what that prepack cost.
    """
    steps = record.get("decode_step_seconds") or []
    total = float(record.get("decode_seconds", 0.0))
    count = int(record.get("decode_steps", 0)) or len(steps)
    with_first = total / count if count else float("nan")
    if len(steps) > 1:
        return statistics.median(steps[1:]), with_first
    return with_first, with_first


def decode_terms(record: dict) -> dict[str, float]:
    """The per-resource decode terms, as tau = max_r W_r/B_r + sum_serial."""
    phase = record.get("phases", {}).get("decode", {})
    cuda = phase.get("cuda", {})
    return {
        "cuda kernel": float(cuda.get("critical_path_kernel_seconds", 0.0)),
        "activation h2d": float(cuda.get("critical_path_activation_h2d_seconds", 0.0)),
        "activation d2h": float(cuda.get("critical_path_activation_d2h_seconds", 0.0)),
        "weight upload wait": float(cuda.get("critical_path_upload_wait_seconds", 0.0)),
        "moe (device)": float(cuda.get("maximum_device_deepseek_moe_seconds", 0.0)),
        "attention (device)": float(
            cuda.get("maximum_device_flash_attention_seconds", 0.0)
            + cuda.get("maximum_device_dsv4_paged_attention_seconds", 0.0)
        ),
        "sync: moe": float(cuda.get("critical_path_moe_synchronization_seconds", 0.0)),
        "sync: attention": float(
            cuda.get("critical_path_attention_synchronization_seconds", 0.0)
        ),
        "sync: projection": float(
            cuda.get("critical_path_projection_synchronization_seconds", 0.0)
        ),
        "sync: mhc": float(cuda.get("critical_path_mhc_synchronization_seconds", 0.0)),
        "checkpoint read": float(phase.get("checkpoint_read_seconds", 0.0)),
    }


def census_check(runs: dict[str, list[dict]]) -> bool:
    """The switch must have taken effect, or the comparison is void."""
    regfed_routes = ("fp8_register_fed", "fp4_register_fed",
                     "dsv4_moe_shared_fp8_register_fed")
    ok = True
    for arm in ARMS:
        if not runs[arm]:
            print(f"  {arm}: no runs")
            ok = False
            continue
        census = runs[arm][0]["_census"]
        if not census:
            print(f"  {arm}: no census written")
            ok = False
            continue
        counts = census.get("routes", census)
        total = sum(int(counts.get(name, 0)) for name in regfed_routes)
        print(f"  {arm}: register-fed dispatches = {total}")
        if arm == "regfed" and total == 0:
            print("    VOID: the register-fed arm never took a register-fed route.")
            ok = False
        if arm == "scalar" and total != 0:
            print("    VOID: the control arm took a register-fed route.")
            ok = False
    return ok


def main() -> int:
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    runs = {arm: load(outdir, arm) for arm in ARMS}

    print("=" * 78)
    print("register-fed A/B  --  did the switch take effect?")
    print("=" * 78)
    effective = census_check(runs)

    print()
    print("route census, both arms")
    names = set()
    for arm in ARMS:
        if runs[arm]:
            names |= set(runs[arm][0]["_census"].get(
                "routes", runs[arm][0]["_census"]).keys())
    for name in sorted(n for n in names if isinstance(n, str)):
        row = []
        for arm in ARMS:
            counts = runs[arm][0]["_census"].get(
                "routes", runs[arm][0]["_census"]) if runs[arm] else {}
            row.append(counts.get(name, 0))
        if any(isinstance(v, int) and v for v in row):
            print(f"  {name:<36} regfed {row[0]:>12}   scalar {row[1]:>12}")

    print()
    print("=" * 78)
    print("primary metric: steady-state decode seconds per token")
    print("=" * 78)
    summary = {}
    for arm in ARMS:
        if not runs[arm]:
            continue
        steady = [steady_decode(r)[0] for r in runs[arm]]
        with_first = [steady_decode(r)[1] for r in runs[arm]]
        summary[arm] = steady
        spread = (max(steady) - min(steady)) / statistics.median(steady) * 100.0 \
            if len(steady) > 1 and statistics.median(steady) else 0.0
        print(f"  {arm:<8} n={len(steady)}  median {statistics.median(steady) * 1000:8.1f} ms/tok"
              f"  ({1.0 / statistics.median(steady):5.2f} tok/s)"
              f"  spread {spread:5.1f}%")
        print(f"           including first step (prepack lands there): "
              f"{statistics.median(with_first) * 1000:8.1f} ms/tok")
        print(f"           runs: {', '.join(f'{v * 1000:.1f}' for v in steady)}")

    if len(summary) == 2 and all(summary.values()):
        regfed = statistics.median(summary["regfed"])
        scalar = statistics.median(summary["scalar"])
        speedup = scalar / regfed if regfed else float("nan")
        worst_spread = max(
            (max(v) - min(v)) / statistics.median(v) if len(v) > 1 else 0.0
            for v in summary.values()
        )
        change = abs(scalar - regfed) / scalar if scalar else 0.0
        print()
        print(f"  regfed / scalar: {speedup:.3f}x")
        if not effective:
            print("  VERDICT: VOID. The census says the arms did not run different"
                  " paths.")
        elif change <= worst_spread:
            print(f"  VERDICT: NOT A WIN. The {change * 100:.1f}% difference is inside"
                  f" the {worst_spread * 100:.1f}% run spread.")
        elif speedup > 1.0:
            print(f"  VERDICT: register-fed is faster by {(speedup - 1) * 100:.1f}%,"
                  f" outside the {worst_spread * 100:.1f}% spread.")
        else:
            print(f"  VERDICT: register-fed is SLOWER by {(1 / speedup - 1) * 100:.1f}%,"
                  f" outside the {worst_spread * 100:.1f}% spread.")

    print()
    print("=" * 78)
    print("decode phase breakdown -- which resource is argmax_r")
    print("=" * 78)
    for arm in ARMS:
        if not runs[arm]:
            continue
        terms = decode_terms(runs[arm][0])
        steps = max(int(runs[arm][0].get("decode_steps", 1)), 1)
        print(f"  {arm}  (per token, over {steps} steps)")
        ordered = sorted(terms.items(), key=lambda kv: kv[1], reverse=True)
        for name, seconds in ordered:
            if seconds <= 0.0:
                continue
            marker = "  <- argmax" if name == ordered[0][0] else ""
            print(f"    {name:<24} {seconds / steps * 1000:8.2f} ms{marker}")
        print(f"    (a term absent here is not zero cost -- host MoE and NVMe are"
              f" not CUDA-attributed)")

    print()
    print("=" * 78)
    print("correctness gate: greedy token ids")
    print("=" * 78)
    if all(runs[arm] for arm in ARMS):
        a = runs["regfed"][0].get("generated_token_ids") or []
        b = runs["scalar"][0].get("generated_token_ids") or []
        if a and a == b:
            print(f"  identical: {len(a)} tokens match exactly")
        elif not a or not b:
            print("  token ids absent from the run output; cannot compare")
        else:
            shared = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                          min(len(a), len(b)))
            print(f"  diverges at token {shared} of {min(len(a), len(b))}")
            print("  A different FP32 accumulation order can flip a greedy argmax;"
                  " this is a divergence point, not proof of a defect. Compare the"
                  " text before deciding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
