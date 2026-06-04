#!/usr/bin/env python
"""Build a canonical molecule/property database for multi-property edits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_DIR / "SketchMol-MultiProperty-EditDataset"
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from sketchmol_multiproperty_dataset.common import PROPERTY_COLUMNS, normalize_properties
from sketchmol_understanding_condition.chem import canonical_smiles, molecular_properties, render_molecule_image, scaffold_smiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--id-column", default="row_id")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--render-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.image_dir is not None:
        args.image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    seen_smiles: set[str] = set()
    read_rows = 0
    invalid_rows = 0
    duplicate_rows = 0
    no_scaffold_rows = 0

    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if args.smiles_column not in (reader.fieldnames or []):
            raise ValueError(f"Missing SMILES column {args.smiles_column!r} in {args.input_csv}")
        for raw_idx, raw in enumerate(reader):
            if args.limit is not None and read_rows >= args.limit:
                break
            read_rows += 1
            smiles = (raw.get(args.smiles_column) or "").strip()
            canonical = canonical_smiles(smiles) if smiles else None
            if canonical is None:
                invalid_rows += 1
                continue
            if canonical in seen_smiles:
                duplicate_rows += 1
                continue
            scaffold = scaffold_smiles(canonical)
            if not scaffold:
                no_scaffold_rows += 1
                continue
            props = normalize_properties(raw)
            computed = molecular_properties(canonical) or {}
            props.update(normalize_properties(computed))
            if any(prop not in props for prop in PROPERTY_COLUMNS):
                invalid_rows += 1
                continue
            mol_id = (raw.get(args.id_column) if args.id_column else None) or f"mol_{raw_idx:08d}"
            image_path = ""
            if args.render_images and args.image_dir is not None:
                image_path = (
                    render_molecule_image(canonical, args.image_dir / f"{len(rows):08d}.png", args.image_size)
                    or ""
                )
            row = {
                "mol_index": len(rows),
                "mol_id": mol_id,
                "canonical_smiles": canonical,
                "scaffold": scaffold,
                "image_path": image_path,
            }
            for prop in PROPERTY_COLUMNS:
                row[prop] = props[prop]
            rows.append(row)
            seen_smiles.add(canonical)

    fieldnames = ["mol_index", "mol_id", "canonical_smiles", "scaffold", *PROPERTY_COLUMNS, "image_path"]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "read_rows": read_rows,
        "molecules": len(rows),
        "invalid_rows": invalid_rows,
        "duplicate_rows": duplicate_rows,
        "no_scaffold_rows": no_scaffold_rows,
        "unique_scaffolds": len({row["scaffold"] for row in rows}),
        "render_images": bool(args.render_images),
        "image_dir": str(args.image_dir) if args.image_dir else None,
    }
    args.output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
