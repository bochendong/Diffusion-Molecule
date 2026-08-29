#!/usr/bin/env python3
"""Select one P30 checkpoint from dev comparisons without reading final results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def score(record: dict[str, object]) -> tuple[float, float, float, float]:
    deltas = record["deltas"]
    de_novo = float(deltas["de_novo_strict_macro"])
    edit = float(deltas["edit_strict_065_macro"])
    promoted = float(record["decision"] == "PROMOTE_FULL_EVAL")
    step = int(record["step"])
    return promoted, min(de_novo, edit), de_novo + edit, -float(step)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", nargs="+", type=int, default=[10, 20, 30, 40, 50, 60])
    args = parser.parse_args()
    records = []
    for step in args.steps:
        path = args.comparison_dir / f"comparison_step{step:03d}.json"
        record = json.loads(path.read_text())
        record["step"] = step
        record["comparison_path"] = str(path)
        records.append(record)
    selected = max(records, key=score)
    step = int(selected["step"])
    adapter = args.model_dir / f"checkpoint-{step:03d}" / "adapter"
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(adapter / "adapter_model.safetensors")
    result = {
        "protocol": "p30_dev_checkpoint_selection_v1",
        "selection_uses_final_gate": False,
        "selected_step": step,
        "selected_adapter": str(adapter),
        "dev_promoted": selected["decision"] == "PROMOTE_FULL_EVAL",
        "selection_score": score(selected),
        "selected_comparison": selected,
        "all_candidates": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
