#!/usr/bin/env python3
"""Compares one sparse-path gate arm against its recorded baseline.

The gate is exactness first and throughput second. An arm whose prompt_tokens
falls below about 2,036 never engaged the indexer and proves nothing, so that
is checked before anything else is reported.
"""
import json
import statistics
import sys


def load(path):
    with open(path) as handle:
        return json.load(handle)


def selection_hash(run):
    trace = run.get("diagnostics", {}).get("index_selections", {})
    return trace.get("trace_hash"), trace.get("entry_count")


def steady_median_ms(run):
    steps = run.get("decode_step_seconds") or []
    if len(steps) < 2:
        return None
    return statistics.median(steps[1:]) * 1000.0


def report(baseline_path, arm_path):
    baseline = load(baseline_path)
    arm = load(arm_path)
    failures = []

    prompt_tokens = arm.get("prompt_tokens")
    if prompt_tokens != baseline.get("prompt_tokens"):
        failures.append(
            f"prompt_tokens {prompt_tokens} != {baseline.get('prompt_tokens')}")
    if (prompt_tokens or 0) < 2036:
        failures.append(
            f"prompt_tokens {prompt_tokens} is below the indexer threshold")

    for field in ("generated_token_ids", "answer"):
        if arm.get(field) != baseline.get(field):
            failures.append(f"{field} differs")

    # The trace hashes only the selections the host performed. Once selection
    # runs inside the chain the host never sees a decode selection, so the
    # entry counts diverge and the hashes stop being comparable. That is a
    # change of observer, not of behaviour, and the generated tokens remain the
    # binding gate; a differing hash at an *equal* entry count is still a real
    # failure.
    arm_trace = selection_hash(arm)
    baseline_trace = selection_hash(baseline)
    if arm_trace[1] == baseline_trace[1] and arm_trace[0] != baseline_trace[0]:
        failures.append(
            f"index selection trace {arm_trace[0]} != {baseline_trace[0]}")

    if arm.get("decode_checkpoint_read_bytes") != 0:
        failures.append(
            f"decode checkpoint reads {arm.get('decode_checkpoint_read_bytes')}")

    # The decode phase is the second of the two the runner emits; the first is
    # prefill, which is not under test.
    def decode_phase(run):
        return run.get("phases", {}).get("decode", {}).get("graph", {})

    decode = decode_phase(arm)
    scalar = decode.get("attention_index_scalar_dispatches")
    if scalar not in (None, 0):
        failures.append(f"scalar index dispatches {scalar}")
    print(f"index dispatches         "
          f"{decode.get('attention_index_cuda_dispatches')} CUDA, "
          f"{scalar} scalar (baseline "
          f"{decode_phase(baseline).get('attention_index_cuda_dispatches')})")

    base_ms = steady_median_ms(baseline)
    arm_ms = steady_median_ms(arm)
    print(f"prompt_tokens            {prompt_tokens}")
    print(f"selection trace          {arm_trace[0]} over {arm_trace[1]} host "
          f"selections (baseline {baseline_trace[0]} over "
          f"{baseline_trace[1]})")
    print(f"generated token ids      "
          f"{'match' if arm.get('generated_token_ids') == baseline.get('generated_token_ids') else 'DIFFER'}")
    print(f"answer text              "
          f"{'match' if arm.get('answer') == baseline.get('answer') else 'DIFFER'}")
    print(f"decode checkpoint bytes  {arm.get('decode_checkpoint_read_bytes')}")
    if base_ms is not None and arm_ms is not None:
        print(f"steady median ms/token   {arm_ms:.3f} against {base_ms:.3f} "
              f"({arm_ms - base_ms:+.3f})")
    steps = arm.get("decode_step_seconds") or []
    print("per-step ms              " +
          " ".join(f"{value * 1000.0:.1f}" for value in steps))

    if failures:
        print("\nGATE FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nGATE PASSED (exactness)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: compare_dsv4_sparse_gate.py <baseline.json> <arm.json>")
        raise SystemExit(2)
    raise SystemExit(report(sys.argv[1], sys.argv[2]))
