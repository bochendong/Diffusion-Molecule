#!/usr/bin/env python3
"""Prepare the frozen 12-target SketchMol OOD reference and P23 prompts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import p23_protocol as protocol


TARGETS = {
    "LogP": (8.0, 9.0, 10.0),
    "TPSA": (160.0, 170.0, 180.0),
    "HBA": (11.0, 12.0, 13.0),
    "RB": (11.0, 12.0, 13.0),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    references: list[dict[str, object]] = []
    prompts: list[dict[str, object]] = []
    for prop, values in TARGETS.items():
        for target in values:
            target_label = str(int(target)) if target.is_integer() else str(target)
            condition_id = f"sketchmol_ood_{prop.lower()}_{target_label}"
            row: dict[str, object] = {
                "condition_id": condition_id,
                "sample_id": condition_id,
                "task_type": "de_novo_design",
                "condition_properties": prop,
                "property_count": 1,
                f"target_{prop}": target,
                f"{prop}_active": True,
            }
            references.append(row)
            messages, source, mode = protocol.build_prompt(row)
            program = protocol.condition_program(row, mode)
            prompts.append(
                {
                    "condition_id": condition_id,
                    "sample_id": condition_id,
                    "task_mode": mode,
                    "source_smiles": source,
                    "messages": messages,
                    "condition_hash": protocol.condition_hash_from_program(program),
                    "task_key": protocol.task_key(program),
                }
            )

    reference_path = args.output_dir / "sketchmol_ood_12.reference.csv"
    fields = [
        "condition_id", "sample_id", "task_type", "condition_properties", "property_count",
        *[f"target_{prop}" for prop in TARGETS],
        *[f"{prop}_active" for prop in TARGETS],
    ]
    with reference_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(references)

    prompt_path = args.output_dir / "sketchmol_ood_12.prompts.jsonl"
    with prompt_path.open("w", encoding="utf-8") as handle:
        for row in prompts:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "protocol": "p23_sketchmol_supp_table3_ood_12_v1",
        "conditions": len(references),
        "candidates_per_condition": 40,
        "targets": {prop: list(values) for prop, values in TARGETS.items()},
        "generation_target_molecule_access": False,
        "property_aware_selection": False,
    }
    (args.output_dir / "sketchmol_ood_12.manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
