#!/usr/bin/env python3
"""Collect the held-out P30 decision and paper-facing strict deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--final-comparison", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    final = json.loads(args.final_comparison.read_text())
    confirmed = bool(selection["dev_promoted"] and final["decision"] == "PROMOTE_FULL_EVAL")
    result = {
        "protocol": "p30_balanced_shared_policy_rl_result_v1",
        "selected_step": selection["selected_step"],
        "dev_promoted": selection["dev_promoted"],
        "final_promoted": final["decision"] == "PROMOTE_FULL_EVAL",
        "confirmed_joint_improvement": confirmed,
        "final_de_novo_strict_delta_pp": 100.0 * float(final["deltas"]["de_novo_strict_macro"]),
        "final_edit_strict_delta_pp": 100.0 * float(final["deltas"]["edit_strict_065_macro"]),
        "native_table_evaluation_authorized": confirmed,
        "final_comparison": final,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# P30 balanced shared-policy RL result",
        "",
        f"- selected dev checkpoint: step {result['selected_step']}",
        f"- de novo strict delta: {result['final_de_novo_strict_delta_pp']:+.3f} pp",
        f"- editing strict delta: {result['final_edit_strict_delta_pp']:+.3f} pp",
        f"- confirmed joint improvement: {'yes' if confirmed else 'no'}",
        f"- native Table 1/2 authorized: {'yes' if confirmed else 'no'}",
        "",
    ]
    (args.output_dir / "RESULT.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
