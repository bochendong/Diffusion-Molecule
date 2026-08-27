#!/usr/bin/env python3
"""Apply the preregistered P25 promotion rule to paired frozen-gate summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--rl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())["aggregate"]
    rl = json.loads(args.rl.read_text())["aggregate"]
    keys = sorted(set(baseline) | set(rl))
    deltas = {key: float(rl[key]) - float(baseline[key]) for key in keys}
    gates = {
        "de_novo_valid_drop_le_002": deltas["de_novo_valid_macro"] >= -0.02,
        "edit_valid_drop_le_002": deltas["edit_valid_macro"] >= -0.02,
        "de_novo_strict_non_regression": deltas["de_novo_strict_macro"] >= 0.0,
        "edit_strict_065_non_regression": deltas["edit_strict_065_macro"] >= 0.0,
        "edit_relaxed_015_drop_le_001": deltas["edit_relaxed_015_macro"] >= -0.01,
        "joint_primary_gain_ge_001": (
            deltas["de_novo_strict_macro"] + deltas["edit_strict_065_macro"] >= 0.01
        ),
    }
    promote = all(gates.values())
    result = {
        "protocol": "p25_joint_gate_comparison_v1",
        "baseline": baseline,
        "rl": rl,
        "deltas": deltas,
        "gates": gates,
        "decision": "PROMOTE_FULL_EVAL" if promote else "STOP_AFTER_GATE",
        "full_table_evaluation_authorized": promote,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
