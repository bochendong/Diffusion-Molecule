#!/usr/bin/env python3
"""Create source-copy predictions for external source-conditioned sanity checks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-csv", required=True, type=Path)
    parser.add_argument("--prediction-csv", required=True, type=Path)
    parser.add_argument("--source-column", default="source_smiles")
    parser.add_argument("--generated-column", default="generated_smiles")
    parser.add_argument("--method", default="source_copy_sanity")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_rows(args.rows_csv)
    output_rows = []
    for row in rows:
        source = str(row.get(args.source_column, "") or "").strip()
        out = dict(row)
        out[args.generated_column] = source
        out["method"] = str(args.method)
        out["direct_candidate_count"] = "1"
        out["direct_unique_candidate_count"] = "1" if source else "0"
        out["direct_valid_candidate_count"] = "1" if source else "0"
        out["direct_unique_valid_candidate_count"] = "1" if source else "0"
        out["direct_best_candidate_rank"] = "0"
        output_rows.append(out)
    write_rows(args.prediction_csv, output_rows)
    print(f"wrote source-copy predictions: {args.prediction_csv} ({len(output_rows)} rows)")
    return 0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
