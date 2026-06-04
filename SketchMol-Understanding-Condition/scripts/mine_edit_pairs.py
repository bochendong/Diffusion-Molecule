#!/usr/bin/env python
"""Mine scaffold-preserving edit pairs from a CSV file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

from sketchmol_understanding_condition import MoleculeRecord, mine_scaffold_edit_pairs


def _float_or_skip(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_records(
    path: Path,
    *,
    smiles_column: str,
    id_column: str | None,
    property_columns: Iterable[str],
) -> list[MoleculeRecord]:
    records: list[MoleculeRecord] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            smiles = row.get(smiles_column)
            if not smiles:
                continue
            props = {}
            for name in property_columns:
                value = _float_or_skip(row.get(name, ""))
                if value is not None:
                    props[name] = value
            records.append(
                MoleculeRecord(
                    smiles=smiles,
                    mol_id=row.get(id_column) if id_column else None,
                    properties=props,
                )
            )
    return records


def write_pairs(path: Path, pairs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_id",
        "target_id",
        "source_smiles",
        "target_smiles",
        "instruction",
        "scaffold",
        "similarity",
        "property_name",
        "property_delta",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            writer.writerow({name: getattr(pair, name) for name in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--property-name", default=None)
    parser.add_argument("--direction", choices=["increase", "decrease"], default="increase")
    parser.add_argument("--min-abs-delta", type=float, default=0.1)
    parser.add_argument("--min-similarity", type=float, default=0.25)
    parser.add_argument("--max-similarity", type=float, default=0.9)
    parser.add_argument("--max-pairs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    property_columns = [args.property_name] if args.property_name else []
    records = read_records(
        args.input_csv,
        smiles_column=args.smiles_column,
        id_column=args.id_column,
        property_columns=property_columns,
    )
    pairs = mine_scaffold_edit_pairs(
        records,
        property_name=args.property_name,
        direction=args.direction,
        min_abs_delta=args.min_abs_delta,
        min_similarity=args.min_similarity,
        max_similarity=args.max_similarity,
        max_pairs=args.max_pairs,
    )
    write_pairs(args.output_csv, pairs)
    print(f"Wrote {len(pairs)} edit pairs to {args.output_csv}")


if __name__ == "__main__":
    main()
