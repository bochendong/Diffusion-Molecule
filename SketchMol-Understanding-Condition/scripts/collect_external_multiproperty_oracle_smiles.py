#!/usr/bin/env python3
"""Collect unique SMILES for external multiproperty oracle scoring."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_understanding_condition.chem import canonical_smiles  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", action="append", default=[], type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument(
        "--smiles-columns",
        default="generated_smiles,source_smiles,target_smiles,smiles,SMILES,canonical_smiles",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input_csv:
        raise ValueError("At least one --input-csv is required")
    columns = [item.strip() for item in str(args.smiles_columns).split(",") if item.strip()]
    smiles = collect_smiles(args.input_csv, columns=columns)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["smiles"])
        writer.writeheader()
        writer.writerows({"smiles": smi} for smi in smiles)
    print(f"wrote {args.output_csv} unique_smiles={len(smiles)}")
    return 0


def collect_smiles(paths: Sequence[Path], *, columns: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for column in columns:
                    smi = canonical_or_blank(row.get(column))
                    if smi and smi not in seen:
                        seen.add(smi)
                        ordered.append(smi)
                        break
    return ordered


def canonical_or_blank(value: object) -> str:
    try:
        return canonical_smiles(str(value or "").strip()) or ""
    except RuntimeError:
        return str(value or "").strip()


if __name__ == "__main__":
    raise SystemExit(main())
