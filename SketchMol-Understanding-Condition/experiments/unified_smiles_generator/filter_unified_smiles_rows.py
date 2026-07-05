#!/usr/bin/env python3
"""Filter unified SMILES rows for focused training or evaluation subsets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--task-mode", choices=("all", "de_novo", "edit"), default="all")
    parser.add_argument(
        "--benchmark-task-contains",
        default="",
        help="Optional comma-separated substrings that must appear in benchmark_task.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    filters = [item.strip().lower() for item in str(args.benchmark_task_contains).split(",") if item.strip()]
    source_rows = read_rows(args.input_csv)
    rows = []
    for row in source_rows:
        if args.task_mode != "all" and task_mode_for_row(row) != args.task_mode:
            continue
        if filters:
            benchmark_task = str(row.get("benchmark_task", "") or "").strip().lower()
            if not any(item in benchmark_task for item in filters):
                continue
        rows.append(row)
    if not rows:
        raise SystemExit(
            f"No rows matched task_mode={args.task_mode!r} "
            f"benchmark_task_contains={filters!r} in {args.input_csv}"
        )
    write_rows(args.output_csv, rows)
    print(
        json.dumps(
            {
                "input_csv": str(args.input_csv),
                "output_csv": str(args.output_csv),
                "input_rows": len(source_rows),
                "output_rows": len(rows),
                "task_mode": args.task_mode,
                "benchmark_task_contains": filters,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def task_mode_for_row(row: Mapping[str, object]) -> str:
    raw = str(row.get("task_mode", "") or row.get("unified_task_mode", "") or "").strip().lower()
    normalized = raw.replace("-", "_").replace(" ", "_")
    if normalized in {"de_novo", "denovo", "generate", "generation"}:
        return "de_novo"
    if normalized in {"edit", "conditional_edit", "source_edit", "edit_generation"}:
        return "edit"
    source = str(row.get("source_smiles", "") or row.get("molecule_smiles", "") or "").strip()
    return "edit" if source else "de_novo"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = infer_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def infer_fieldnames(rows: Sequence[Mapping[str, object]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            text = str(key)
            if text not in seen:
                out.append(text)
                seen.add(text)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
