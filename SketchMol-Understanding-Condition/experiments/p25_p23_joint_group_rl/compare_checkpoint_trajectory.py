#!/usr/bin/env python3
"""Summarize P25 baseline and saved-checkpoint gate trajectories without selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    names = ("baseline", "checkpoint-013", "checkpoint-026", "rl")
    summaries = {
        name: json.loads((args.gate_dir / name / "summary.json").read_text())["aggregate"]
        for name in names
    }
    baseline = summaries["baseline"]
    result = {
        "protocol": "p25_checkpoint_trajectory_diagnostic_v1",
        "posthoc_diagnostic_only": True,
        "checkpoint_selection_authorized": False,
        "trajectory": {
            name: {
                "aggregate": values,
                "delta_from_baseline": {
                    key: float(values[key]) - float(baseline[key]) for key in sorted(baseline)
                },
            }
            for name, values in summaries.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
