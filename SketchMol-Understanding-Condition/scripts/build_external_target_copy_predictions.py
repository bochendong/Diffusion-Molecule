#!/usr/bin/env python3
"""Build target-copy sanity predictions for external multi-property rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-csv", required=True, type=Path)
    parser.add_argument("--prediction-csv", required=True, type=Path)
    parser.add_argument("--method", default="target_copy_sanity")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_rows(args.rows_csv)
    output_rows = []
    for row in rows:
        target = str(row.get("target_smiles") or "").strip()
        source = str(row.get("source_smiles") or "").strip()
        out = dict(row)
        out["generated_smiles"] = target or source
        out["method"] = str(args.method)
        out["direct_candidate_count"] = 1
        out["target_copy_used_source_fallback"] = "True" if not target else "False"
        output_rows.append(out)
    write_rows(args.prediction_csv, output_rows)
    print(f"wrote {len(output_rows)} target-copy rows to {args.prediction_csv}")
    return 0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
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
