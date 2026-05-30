"""Build paired SMILES/image manifests for SketchSMILES experiments."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    input_smiles: str
    canonical_smiles: str
    valid: bool
    image_path: str
    error: str = ""


def build_pair_manifest(
    input_csv: str | Path,
    output_dir: str | Path,
    smiles_column: str = "smiles",
    image_size: int = 256,
    limit: int | None = None,
) -> list[PairRecord]:
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    records: list[PairRecord] = []
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if smiles_column not in (reader.fieldnames or []):
            raise ValueError(f"Missing SMILES column {smiles_column!r} in {input_csv}.")
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            smiles = (row.get(smiles_column) or "").strip()
            pair_id = f"mol_{idx:06d}"
            image_path = image_dir / f"{pair_id}.png"
            records.append(_render_pair(pair_id, smiles, image_path, image_size=image_size))

    manifest_path = output_dir / "pairs.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(PairRecord.__dataclass_fields__.keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    summary = summarize_pairs(records)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return records


def summarize_pairs(records: list[PairRecord]) -> dict[str, float]:
    total = len(records)
    valid = sum(1 for record in records if record.valid)
    rendered = sum(1 for record in records if record.image_path and Path(record.image_path).exists())
    return {
        "molecules": float(total),
        "valid_smiles": float(valid),
        "rendered_images": float(rendered),
        "valid_fraction": float(valid / total) if total else 0.0,
        "rendered_fraction": float(rendered / total) if total else 0.0,
    }


def _render_pair(pair_id: str, smiles: str, image_path: Path, image_size: int) -> PairRecord:
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except Exception as exc:
        return PairRecord(pair_id=pair_id, input_smiles=smiles, canonical_smiles="", valid=False, image_path="", error=f"rdkit_unavailable:{exc}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return PairRecord(pair_id=pair_id, input_smiles=smiles, canonical_smiles="", valid=False, image_path="", error="invalid_smiles")
    canonical = Chem.MolToSmiles(mol, canonical=True)
    try:
        image = Draw.MolToImage(mol, size=(image_size, image_size))
        image.save(image_path)
    except Exception as exc:
        return PairRecord(pair_id=pair_id, input_smiles=smiles, canonical_smiles=canonical, valid=True, image_path="", error=f"render_failed:{exc}")
    return PairRecord(pair_id=pair_id, input_smiles=smiles, canonical_smiles=canonical, valid=True, image_path=str(image_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a paired SMILES/rendered-image manifest.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    records = build_pair_manifest(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        smiles_column=args.smiles_column,
        image_size=args.image_size,
        limit=args.limit,
    )
    print(json.dumps(summarize_pairs(records), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
