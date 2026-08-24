#!/usr/bin/env python3
"""Freeze balanced, bounded P6 de-novo and editing gate subsets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--mode", choices=("denovo_hard", "edit_table1"), required=True)
    parser.add_argument("--rows-per-group", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def identity(row: dict[str, str]) -> str:
    for key in ("condition_id", "example_id", "sample_id", "pair_id"):
        if row.get(key):
            return row[key]
    return json.dumps(row, sort_keys=True)


def group(row: dict[str, str], mode: str) -> str:
    if mode == "denovo_hard":
        return f"{int(float(row.get('property_count') or 0))}p"
    return str(row.get("moledit_task_key") or row.get("benchmark_task") or row.get("task") or "unknown")


def main() -> int:
    parsed = args()
    with parsed.input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = group(row, parsed.mode)
        if parsed.mode == "denovo_hard" and key not in {"6p", "7p"}:
            continue
        grouped[key].append(row)
    selected = []
    for key in sorted(grouped):
        values = sorted(
            grouped[key],
            key=lambda row: hashlib.sha256(f"{parsed.seed}:{identity(row)}".encode()).hexdigest(),
        )
        selected.extend(values[: int(parsed.rows_per_group)])
    parsed.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with parsed.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    manifest = {
        "protocol": "p6_balanced_single_seed_gate_subset",
        "mode": parsed.mode,
        "seed": parsed.seed,
        "rows_per_group": parsed.rows_per_group,
        "input_rows": len(rows),
        "selected_rows": len(selected),
        "groups": {key: min(len(values), int(parsed.rows_per_group)) for key, values in sorted(grouped.items())},
        "ids": [identity(row) for row in selected],
    }
    parsed.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected_rows": len(selected), "groups": manifest["groups"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
