#!/usr/bin/env python3
"""Combine P30.3 de-novo and editing Raw@1 gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--denovo", required=True, type=Path)
    parser.add_argument("--edit-baseline", required=True, type=Path)
    parser.add_argument("--edit-refined", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    denovo = json.loads(args.denovo.read_text())
    baseline = json.loads(args.edit_baseline.read_text())["aggregate"]
    refined = json.loads(args.edit_refined.read_text())["aggregate"]
    edit_deltas = {key: float(refined[key]) - float(baseline[key]) for key in baseline}
    gates = {
        "denovo_promoted": denovo["decision"] == "RUN_FULL_BUDGET_CURVE",
        "edit_strict_non_regression": edit_deltas["edit_strict_065_macro"] >= 0.0,
        "edit_relaxed_drop_le_001": edit_deltas["edit_relaxed_015_macro"] >= -0.01,
        "edit_valid_drop_le_001": edit_deltas["edit_valid_macro"] >= -0.01,
    }
    result = {
        "protocol": "p30_3_joint_raw1_gate_v1",
        "denovo": denovo,
        "editing_baseline": baseline,
        "editing_refined": refined,
        "editing_deltas": edit_deltas,
        "gates": gates,
        "decision": "RUN_FULL_BUDGET_CURVE" if all(gates.values()) else "STOP_AFTER_JOINT_GATE",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# P30.3 joint Raw@1 gate",
        "",
        f"- de-novo Raw@1 delta: {100 * float(denovo['deltas']['strict_macro']):+.1f} pp",
        f"- de-novo validity delta: {100 * float(denovo['deltas']['valid_macro']):+.1f} pp",
        f"- editing strict delta: {100 * edit_deltas['edit_strict_065_macro']:+.2f} pp",
        f"- editing relaxed delta: {100 * edit_deltas['edit_relaxed_015_macro']:+.2f} pp",
        f"- editing validity delta: {100 * edit_deltas['edit_valid_macro']:+.2f} pp",
        f"- decision: {result['decision']}",
        "",
    ]
    (args.output_dir / "RESULT.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
