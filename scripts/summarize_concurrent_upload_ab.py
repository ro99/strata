#!/usr/bin/env python3
"""Summarize the concurrent-upload A/B.

Reports the median decode ms/step per arm, the mechanism-level counters the
contract names (moe_prepare, and the per-device upload wait as both a sum and a
maximum), and checks that the arms emitted identical token sequences. The token
check compares the full id list, not the leading characters of the text: a
greedy argmax flip late in a 64-token generation is exactly what a prefix
comparison misses.
"""

import json
import pathlib
import statistics
import sys


def load(result_dir: pathlib.Path, arm: str) -> list[dict]:
    runs = []
    for run in sorted((result_dir / arm).glob("run-*")):
        path = run / "generation.json"
        if path.exists():
            runs.append(json.loads(path.read_text()))
    return runs


def per_step(run: dict, seconds: float) -> float:
    return 1000.0 * seconds / run["decode_steps"]


def summarize(run: dict) -> dict:
    decode = run["phases"]["decode"]
    devices = decode["cuda"]["devices"]
    waits = [1000.0 * d.get("upload_wait_seconds", 0.0) / run["decode_steps"]
             for d in devices]
    return {
        "ms_step": 1000.0 * run["decode_seconds"] / run["decode_steps"],
        "tok_s": run["decode_steps"] / run["decode_seconds"],
        "moe_prepare": per_step(run, decode["graph"]["moe_prepare_seconds"]),
        "moe": per_step(run, decode["graph"]["moe_seconds"]),
        "attention": per_step(run, decode["graph"]["attention_seconds"]),
        "wait_sum": sum(waits),
        "wait_max": max(waits),
        "weight_mb": decode["cuda"]["weight_h2d_bytes"] / run["decode_steps"] / 1e6,
    }


def main() -> int:
    result_dir = pathlib.Path(sys.argv[1])
    arms = {arm: load(result_dir, arm) for arm in ("serial", "concurrent")}
    for arm, runs in arms.items():
        if not runs:
            print(f"no runs for arm {arm}", file=sys.stderr)
            return 1

    tokens = {arm: [tuple(r["generated_token_ids"]) for r in runs]
              for arm, runs in arms.items()}
    identical = len({t for values in tokens.values() for t in values}) == 1
    print(f"{'metric':<22}{'serial':>12}{'concurrent':>14}{'delta':>12}")
    keys = [("ms_step", "ms/step"), ("tok_s", "tok/s"),
            ("moe_prepare", "moe_prepare ms"), ("moe", "moe total ms"),
            ("attention", "attention ms"), ("wait_sum", "upload wait sum ms"),
            ("wait_max", "upload wait max ms"), ("weight_mb", "weight H2D MB")]
    medians = {}
    for arm, runs in arms.items():
        values = [summarize(r) for r in runs]
        medians[arm] = {k: statistics.median(v[k] for v in values)
                        for k, _ in keys}
    for key, label in keys:
        a, b = medians["serial"][key], medians["concurrent"][key]
        print(f"{label:<22}{a:>12.2f}{b:>14.2f}{b - a:>+12.2f}")

    print()
    for arm, runs in arms.items():
        allruns = ", ".join(
            f"{1000.0 * r['decode_seconds'] / r['decode_steps']:.1f}"
            for r in runs)
        print(f"{arm:<12} all runs ms/step: {allruns}")
    speedup = medians["serial"]["ms_step"] / medians["concurrent"]["ms_step"]
    saving = medians["serial"]["ms_step"] - medians["concurrent"]["ms_step"]
    print(f"\nspeedup {speedup:.3f}x   saving {saving:.2f} ms/step")
    print(f"token sequences identical across all arms and runs: {identical}")
    if not identical:
        for arm, values in tokens.items():
            for index, value in enumerate(values):
                print(f"  {arm} run-{index + 1}: {value[:24]}")
    return 0 if identical else 2


if __name__ == "__main__":
    sys.exit(main())
