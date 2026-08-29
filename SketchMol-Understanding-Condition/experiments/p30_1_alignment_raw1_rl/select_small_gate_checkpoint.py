#!/usr/bin/env python3
"""Select an eligible P30.1 checkpoint using only the frozen small gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def score(record: dict[str, object]) -> tuple[float, float, float]:
    aggregate = record["aggregate"]
    return (
        float(record["decision"] == "RUN_FULL_BUDGET_CURVE"),
        float(aggregate["strict_macro"]),
        float(aggregate["valid_macro"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    tags = {10: "step010", 20: "step020", 30: "rl"}
    records = []
    for step, tag in tags.items():
        path = args.output_root / "small_gate" / tag / "result.json"
        record = json.loads(path.read_text())
        record["step"] = step
        record["tag"] = tag
        record["result_path"] = str(path)
        records.append(record)
    eligible = [record for record in records if record["decision"] == "RUN_FULL_BUDGET_CURVE"]
    selected = max(eligible, key=score) if eligible else None
    result = {
        "protocol": "p30_1_small_gate_checkpoint_selection_v1",
        "uses_full_budget_results": False,
        "candidates": records,
        "eligible_steps": [int(record["step"]) for record in eligible],
        "selected_step": int(selected["step"]) if selected else None,
        "selected_adapter": (
            str(args.output_root / "model" / "balanced_shared_rl" / f"checkpoint-{int(selected['step']):03d}" / "adapter")
            if selected else None
        ),
        "decision": "RUN_FULL_BUDGET_CURVE" if selected else "STOP_AFTER_CHECKPOINT_SCREENS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

