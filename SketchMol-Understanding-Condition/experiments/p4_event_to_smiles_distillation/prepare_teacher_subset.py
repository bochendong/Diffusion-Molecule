#!/usr/bin/env python3
"""Build a balanced, train-only Table1 subset for frozen D3 teacher sampling."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
if str(PROJECT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from train_direct_smiles_generator_rl import table1_edit_specs  # noqa: E402
from sketchmol_understanding_condition.chem import canonical_smiles  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--excluded-eval-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--per-task", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def task_key(row: dict[str, str]) -> str:
    specs = table1_edit_specs(row)
    return "+".join(f"{prop}:{direction}" for prop, direction in specs)


def main() -> int:
    args = parse_args()
    excluded = {
        canonical_smiles(row.get("source_smiles", ""))
        for row in read_rows(args.excluded_eval_csv)
    }
    excluded.discard("")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    dropped_overlap = 0
    for row in read_rows(args.input_csv):
        source = canonical_smiles(row.get("source_smiles", ""))
        if not source or not str(row.get("target_smiles", "") or "").strip():
            continue
        if source in excluded:
            dropped_overlap += 1
            continue
        key = task_key(row)
        if key:
            grouped[key].append(dict(row))
    selected: list[dict[str, str]] = []
    for offset, key in enumerate(sorted(grouped)):
        values = list(grouped[key])
        random.Random(int(args.seed) + offset).shuffle(values)
        for row in values[: int(args.per_task)]:
            row["p4_teacher_task_key"] = key
            row["task_mode"] = "edit"
            selected.append(row)
    random.Random(int(args.seed)).shuffle(selected)
    if not selected:
        raise ValueError("No train-only Table1 rows selected")
    fields = list(dict.fromkeys(key for row in selected for key in row))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    manifest = {
        "protocol": "p4_train_only_teacher_subset_v1",
        "input_csv": str(args.input_csv),
        "excluded_eval_csv": str(args.excluded_eval_csv),
        "seed": int(args.seed),
        "per_task": int(args.per_task),
        "rows": len(selected),
        "tasks": {key: sum(1 for row in selected if row["p4_teacher_task_key"] == key) for key in sorted(grouped)},
        "excluded_source_count": len(excluded),
        "dropped_source_overlap": dropped_overlap,
    }
    args.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
