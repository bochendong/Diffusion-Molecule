#!/usr/bin/env python3
"""Merge existing benchmark row CSVs into unified train/eval sets."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Mapping, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--input-csv", action="append", default=[], type=Path)
    parser.add_argument("--limit", type=int, default=0, help="0 keeps all merged rows.")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: "" if value is None else str(value) for key, value in row.items()} for row in reader]


def write_rows(path: Path, rows: Sequence[Mapping[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: str(row.get(key, "") or "") for key in fieldnames})


def merge_rows(inputs: Sequence[Path]) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    seen_fields: set[str] = set()
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        part = read_rows(path)
        rows.extend(part)
        for key in (part[0].keys() if part else []):
            if key not in seen_fields:
                fieldnames.append(key)
                seen_fields.add(key)
    return rows, fieldnames


def main() -> int:
    args = parse_args()
    if not args.input_csv:
        raise SystemExit("Provide at least one --input-csv")
    rows, fieldnames = merge_rows(args.input_csv)
    if args.limit and len(rows) > int(args.limit):
        rng = random.Random(int(args.seed))
        rows = rng.sample(rows, int(args.limit))
    write_rows(args.output_csv, rows, fieldnames)
    print(
        {
            "output_csv": str(args.output_csv),
            "inputs": [str(path) for path in args.input_csv],
            "rows": len(rows),
            "columns": len(fieldnames),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
