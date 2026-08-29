#!/usr/bin/env python3
"""Collect every P31.1 checkpoint's frozen de-novo and editing gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--edit-baseline", required=True, type=Path)
    parser.add_argument("--steps", default="25,50,100")
    args = parser.parse_args()
    baseline = json.loads(args.edit_baseline.read_text())["aggregate"]
    results = []
    for step in [int(value) for value in args.steps.split(",") if value]:
        tag = f"step-{step:03d}"
        denovo = json.loads((args.output_root / "gate" / tag / "denovo" / "result.json").read_text())
        edit = json.loads((args.output_root / "gate" / tag / "edit" / "summary.json").read_text())
        edit_aggregate = edit["aggregate"]
        edit_deltas = {
            key: float(edit_aggregate[key]) - float(baseline[key])
            for key in (
                "edit_valid_macro", "edit_strict_065_macro", "edit_relaxed_015_macro"
            )
        }
        gates = {
            "de_novo_raw1_gain_ge_002": float(denovo["deltas"]["strict_macro"]) >= 0.02,
            "de_novo_validity_drop_le_001": float(denovo["deltas"]["valid_macro"]) >= -0.01,
            "edit_strict_non_regression": edit_deltas["edit_strict_065_macro"] >= 0.0,
            "edit_relaxed_drop_le_001": edit_deltas["edit_relaxed_015_macro"] >= -0.01,
            "edit_validity_drop_le_001": edit_deltas["edit_valid_macro"] >= -0.01,
        }
        results.append({
            "step": step,
            "de_novo": denovo,
            "editing": edit,
            "editing_baseline": baseline,
            "editing_deltas": edit_deltas,
            "gates": gates,
            "promoted": all(gates.values()),
        })
    promoted = [item for item in results if item["promoted"]]
    best = max(
        promoted,
        key=lambda item: (
            float(item["de_novo"]["deltas"]["strict_macro"]),
            float(item["editing_deltas"]["edit_strict_065_macro"]),
        ),
        default=None,
    )
    result = {
        "protocol": "p31_1_joint_checkpoint_gate_v1",
        "checkpoints": results,
        "selected_step": None if best is None else best["step"],
        "decision": "EXTEND_RLOO_TO_300" if best is not None else "STOP_TOKEN_LEVEL_RL",
    }
    output_dir = args.output_root / "joint_gate"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = ["# P31.1 joint checkpoint gate", ""]
    for item in results:
        lines.extend([
            f"## {item['step']} informative updates per mode",
            "",
            f"- de-novo Raw@1 delta: {100 * float(item['de_novo']['deltas']['strict_macro']):+.2f} pp",
            f"- de-novo validity delta: {100 * float(item['de_novo']['deltas']['valid_macro']):+.2f} pp",
            f"- editing strict delta: {100 * item['editing_deltas']['edit_strict_065_macro']:+.2f} pp",
            f"- editing relaxed delta: {100 * item['editing_deltas']['edit_relaxed_015_macro']:+.2f} pp",
            f"- editing validity delta: {100 * item['editing_deltas']['edit_valid_macro']:+.2f} pp",
            f"- promoted: {item['promoted']}",
            "",
        ])
    lines.append(f"Decision: **{result['decision']}**")
    (output_dir / "RESULT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
