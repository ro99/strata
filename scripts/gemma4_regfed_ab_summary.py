#!/usr/bin/env python3
"""Summarize the interleaved Gemma 4 MXFP4 register-fed A/B."""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys

ARMS = ("register-fed", "scalar")


def load(outdir: str, arm: str) -> list[dict]:
    records = []
    for path in sorted(glob.glob(os.path.join(outdir, f"{arm}-rep*.json"))):
        if path.endswith(".census.json"):
            continue
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        census_path = path.removesuffix(".json") + ".census.json"
        with open(census_path, encoding="utf-8") as handle:
            record["_census"] = json.load(handle)
        record["_path"] = path
        records.append(record)
    return records


def steady_seconds_per_token(record: dict) -> float:
    metrics = record["metrics"]
    return float(metrics["steady_decode_seconds"]) / int(
        metrics["steady_decode_tokens"])


def first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def main() -> int:
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    runs = {arm: load(outdir, arm) for arm in ARMS}
    if any(len(runs[arm]) < 3 for arm in ARMS):
        print("VOID: fewer than three completed repetitions in an arm")
        return 2

    print("route census")
    effective = True
    for arm in ARMS:
        counts = [int(run["_census"]["routes"]["fp4_register_fed"])
                  for run in runs[arm]]
        scalar = [int(run["_census"]["routes"]["fp4_e2m1_group32"])
                  for run in runs[arm]]
        print(f"  {arm}: fp4_register_fed={counts}; fp4_scalar={scalar}")
        if arm == "register-fed" and any(value == 0 for value in counts):
            effective = False
        if arm == "scalar" and any(value != 0 for value in counts):
            effective = False
    if not effective:
        print("VOID: the route census says the switch did not produce distinct arms")
        return 2

    print("\nprimary: steady decode, first batch-1 step excluded")
    samples: dict[str, list[float]] = {}
    for arm in ARMS:
        samples[arm] = [steady_seconds_per_token(run) for run in runs[arm]]
        values = samples[arm]
        first = [float(run["metrics"]["first_decode_seconds"])
                 for run in runs[arm]]
        print(f"  {arm}: median={statistics.median(values) * 1000:.3f} ms/token; "
              f"range={min(values) * 1000:.3f}-{max(values) * 1000:.3f}; "
              f"spread={(max(values)-min(values)) * 1000:.3f} ms; "
              f"runs={[round(value * 1000, 3) for value in values]}")
        print(f"    first-step median={statistics.median(first) * 1000:.3f} ms")

    candidate = statistics.median(samples["register-fed"])
    control = statistics.median(samples["scalar"])
    speedup = control / candidate
    median_delta = control - candidate
    worst_spread = max(max(values) - min(values) for values in samples.values())
    print(f"  speedup={speedup:.4f}x; median delta={median_delta * 1000:.3f} ms/token")
    if abs(median_delta) <= worst_spread:
        verdict = "NOT A WIN: median difference is inside observed spread"
    elif speedup > 1.0:
        verdict = "WIN: register-fed is faster outside observed spread"
    else:
        verdict = "NEGATIVE: register-fed is slower outside observed spread"
    print(f"  verdict={verdict}")

    print("\ncorrectness")
    reference = runs["scalar"][0]["generated_token_ids"]
    divergence = None
    divergent_path = None
    for arm in ARMS:
        for run in runs[arm]:
            index = first_divergence(reference, run["generated_token_ids"])
            if index is not None:
                divergence = index
                divergent_path = run["_path"]
                break
        if divergence is not None:
            break
    if divergence is None:
        print(f"  identical: all {sum(map(len, runs.values()))} runs match "
              f"the same {len(reference)} greedy tokens")
    else:
        print(f"  DIVERGED: first index {divergence} in {divergent_path}")
        return 1

    print("\nprefill plausibility (diagnostic, not the primary metric)")
    for arm in ARMS:
        ratios = []
        for run in runs[arm]:
            metrics = run["metrics"]
            prefill = float(metrics["prefill_seconds"]) / int(metrics["prefill_tokens"])
            ratios.append(prefill / steady_seconds_per_token(run))
        print(f"  {arm}: median prefill/decode per-token ratio "
              f"{statistics.median(ratios):.3f}")
    print("  Expected <0.25; scalar's failure is the separately recorded prefill "
          "batching defect, not decode evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
