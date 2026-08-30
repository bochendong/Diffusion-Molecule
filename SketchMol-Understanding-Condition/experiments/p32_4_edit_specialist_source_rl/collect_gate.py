#!/usr/bin/env python3
"""Apply the preregistered P32.4 small editing gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STEPS = (5, 10, 20)


def load_summary(eval_root: Path, step: int) -> dict[str, object]:
    return json.loads((eval_root / f"step-{step:03d}" / "summary.json").read_text())


def metrics(summary: dict[str, object]) -> dict[str, float]:
    aggregate = summary["aggregate"]
    return {
        "strict": float(aggregate["edit_strict_065_macro"]),
        "relaxed": float(aggregate["edit_relaxed_015_macro"]),
        "valid": float(aggregate["edit_valid_macro"]),
    }


def assess(baseline_summary: dict[str, object], candidate_summary: dict[str, object]):
    baseline = metrics(baseline_summary)
    candidate = metrics(candidate_summary)
    bucket_deltas = {
        bucket: float(candidate_summary["buckets"][bucket]["strict_rate"])
        - float(baseline_summary["buckets"][bucket]["strict_rate"])
        for bucket in baseline_summary["buckets"]
    }
    deltas = {key: candidate[key] - baseline[key] for key in baseline}
    nonnegative_buckets = sum(delta >= -1e-12 for delta in bucket_deltas.values())
    passed = (
        deltas["strict"] >= 0.02 - 1e-12
        and deltas["relaxed"] >= -0.01 - 1e-12
        and deltas["valid"] >= -0.01 - 1e-12
        and nonnegative_buckets >= 7
    )
    return {
        "metrics": candidate,
        "deltas": deltas,
        "bucket_strict_deltas": bucket_deltas,
        "nonnegative_strict_buckets": nonnegative_buckets,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    baseline_summary = load_summary(args.eval_root, 0)
    candidates = {
        str(step): assess(baseline_summary, load_summary(args.eval_root, step))
        for step in STEPS
    }
    passing = [step for step in STEPS if candidates[str(step)]["passed"]]
    selected = max(
        passing,
        key=lambda step: (
            candidates[str(step)]["metrics"]["strict"],
            candidates[str(step)]["metrics"]["relaxed"],
            -step,
        ),
        default=None,
    )
    result = {
        "protocol": "p32_4_edit_specialist_source_constrained_online_rloo_v1",
        "baseline": metrics(baseline_summary),
        "candidates": candidates,
        "selected_checkpoint": selected,
        "decision": "PROMOTE_TO_FULL_EDIT_EVAL" if selected is not None else "STOP_P32_4",
        "construction_policy": "frozen and unchanged",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "COLLECT_COMPLETE").touch()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
