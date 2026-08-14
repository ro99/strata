#!/usr/bin/env python3
"""Step 5 gate: centralized decode must be bit-identical to main.

Takes the reference arm first, then one or more comparison arms. Reports a
per-oracle verdict and exits non-zero if any binding oracle differs.

The binding oracles are, in order of what they would catch:

  logit trace hash    every logit value of every forward, folded into one
                      fnv1a64. Catches a numerical divergence anywhere in the
                      forward pass, including one that sampling would hide.
  per-forward top-k   localizes such a divergence to a position, so a failure
                      is diagnosable rather than merely known.
  generated token IDs the generation oracle.
  answer text         the same claim after detokenization; kept because a
                      tokenizer change would pass the ID oracle.

Timing fields are deliberately not compared. They are expected to differ.
"""

from __future__ import annotations

import json
import sys


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def trace_hash(run):
    return (run.get("diagnostics") or {}).get("logits", {}).get(
        "aggregate", {}).get("trace_hash")


def forwards(run):
    return (run.get("diagnostics") or {}).get("logits", {}).get("forwards", [])


def first_top_k_divergence(reference, arm):
    left, right = forwards(reference), forwards(arm)
    if len(left) != len(right):
        return f"forward count {len(left)} against {len(right)}"
    for index, (a, b) in enumerate(zip(left, right)):
        if a.get("position") != b.get("position"):
            return f"forward {index}: position {a.get('position')} against {b.get('position')}"
        if a.get("selected_token") != b.get("selected_token"):
            return (f"forward {index} at position {a.get('position')}: "
                    f"selected {a.get('selected_token')} against {b.get('selected_token')}")
        if a.get("top") != b.get("top"):
            return (f"forward {index} at position {a.get('position')}: "
                    "top-k logits differ")
    return None


def compare(reference, arm, name):
    checks = []
    ref_hash, arm_hash = trace_hash(reference), trace_hash(arm)
    if ref_hash is None or arm_hash is None:
        checks.append(("logit trace hash", False,
                       "absent -- rerun with --logit-trace"))
    else:
        checks.append(("logit trace hash", ref_hash == arm_hash,
                       f"{ref_hash} against {arm_hash}"))

    divergence = first_top_k_divergence(reference, arm)
    count = len(forwards(reference))
    checks.append(("per-forward top-k", divergence is None,
                   divergence or f"{count} forwards identical"))

    ref_ids = reference.get("generated_token_ids")
    arm_ids = arm.get("generated_token_ids")
    checks.append(("generated token IDs", ref_ids == arm_ids,
                   f"{len(arm_ids or [])} tokens"))

    checks.append(("prompt token IDs",
                   reference.get("prompt_token_ids") == arm.get("prompt_token_ids"),
                   f"{len(arm.get('prompt_token_ids') or [])} tokens"))

    checks.append(("answer text",
                   reference.get("answer") == arm.get("answer"),
                   f"{len(arm.get('answer') or '')} chars"))

    print(f"\n== {name} against reference ==")
    width = max(len(label) for label, _, _ in checks)
    for label, passed, detail in checks:
        print(f"  {label.ljust(width)}  {'MATCH ' if passed else 'DIFFER'}  {detail}")
    return all(passed for _, passed, _ in checks)


def main(argv):
    if len(argv) < 3:
        print("usage: compare_dsv4_centralized_parity.py <reference.json> <arm.json>...",
              file=sys.stderr)
        return 2
    reference = load(argv[1])
    print(f"reference: {argv[1]}")
    if not forwards(reference):
        print("  WARNING: reference carries no logit trace; "
              "the teacher-forcing oracle is not being exercised")
    ok = True
    for path in argv[2:]:
        name = path.rsplit("/", 1)[-1].removesuffix(".json")
        ok &= compare(reference, load(path), name)
    print("\nGATE PASSED (centralized parity)" if ok
          else "\nGATE FAILED (centralized parity)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
