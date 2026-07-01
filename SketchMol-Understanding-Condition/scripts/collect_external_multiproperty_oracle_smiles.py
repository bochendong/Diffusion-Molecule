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
        "--exclude-properties-csv",
        action="append",
        default=[],
        type=Path,
        help="Property CSVs whose fully covered SMILES should be skipped.",
    )
    parser.add_argument(
        "--required-properties",
        default="",
        help="Comma-separated properties required in every exclude CSV hit.",
    )
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
    required_properties = [canonical_prop(item) for item in str(args.required_properties).split(",") if item.strip()]
    exclude_lookup = load_property_coverage(args.exclude_properties_csv, required_properties=required_properties)
    smiles = collect_smiles(args.input_csv, columns=columns, exclude_lookup=exclude_lookup)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["smiles"])
        writer.writeheader()
        writer.writerows({"smiles": smi} for smi in smiles)
    print(f"wrote {args.output_csv} unique_smiles={len(smiles)}")
    return 0


def collect_smiles(
    paths: Sequence[Path],
    *,
    columns: Sequence[str],
    exclude_lookup: set[str] | None = None,
) -> list[str]:
    exclude_lookup = exclude_lookup or set()
    seen: set[str] = set()
    ordered: list[str] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for column in columns:
                    smi = canonical_or_blank(row.get(column))
                    if smi in exclude_lookup:
                        continue
                    if smi and smi not in seen:
                        seen.add(smi)
                        ordered.append(smi)
    return ordered


def load_property_coverage(paths: Sequence[Path], *, required_properties: Sequence[str]) -> set[str]:
    covered = set()
    if not paths or not required_properties:
        return covered
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                smi = canonical_or_blank(row.get("smiles") or row.get("SMILES") or row.get("canonical_smiles"))
                if not smi:
                    continue
                if all(str(row.get(prop, "")).strip() for prop in required_properties):
                    covered.add(smi)
    return covered


def canonical_or_blank(value: object) -> str:
    try:
        return canonical_smiles(str(value or "").strip()) or ""
    except RuntimeError:
        return str(value or "").strip()


def canonical_prop(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


if __name__ == "__main__":
    raise SystemExit(main())
