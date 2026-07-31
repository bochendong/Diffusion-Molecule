#!/usr/bin/env python3
"""Build a deterministic, task-balanced train-only pool for verifier search."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import unified_smiles_generator as core  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--rows-per-group", type=int, default=100)
    parser.add_argument("--seed", type=int, default=71)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def group_key(row: Mapping[str, str]) -> str:
    mode = core.task_mode_for_row(row)
    if mode == core.DE_NOVO_MODE:
        parsed = core.parse_float(row.get("property_count", ""))
        count = int(parsed) if math.isfinite(parsed) else len(core.selected_properties(row))
        return f"de_novo:{count}p"
    specs = sorted((prop, direction) for prop, direction in core.instruction_task_specs(row) if prop)
    if not specs:
        specs = sorted((prop, core.property_direction(row, prop)) for prop in core.selected_properties(row))
    return "edit:" + ("+".join(f"{prop}:{direction:+d}" for prop, direction in specs) or "unknown")


def write_rows(path: Path, rows: list[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_rows(args.input_csv):
        grouped[group_key(row)].append(row)
    rng = random.Random(int(args.seed))
    selected = []
    counts = {}
    for key, rows in sorted(grouped.items()):
        pool = list(rows)
        rng.shuffle(pool)
        chosen = pool[: max(1, int(args.rows_per_group))]
        selected.extend(chosen)
        counts[key] = len(chosen)
    rng.shuffle(selected)
    if not selected:
        raise SystemExit("No source rows available for the search pool")
    write_rows(args.output_csv, selected)
    manifest = {
        "protocol": "unified_molecular_transformation_policy_search_pool_v1",
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "rows_per_group": int(args.rows_per_group),
        "seed": int(args.seed),
        "group_counts": counts,
        "output_rows": len(selected),
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
