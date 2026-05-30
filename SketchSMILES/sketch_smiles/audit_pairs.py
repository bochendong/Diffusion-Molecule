"""Audit paired SMILES/image manifests for SketchSMILES."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any


def audit_pair_manifest(
    pair_dir: str | Path,
    sample_count: int = 64,
    seed: int = 7,
    expected_image_size: int = 256,
    contact_sheet_cols: int = 8,
    contact_thumb_size: int = 160,
) -> dict[str, Any]:
    pair_dir = Path(pair_dir)
    pairs_path = pair_dir / "pairs.csv"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Missing paired manifest: {pairs_path}")

    rows = _read_rows(pairs_path)
    rdkit = _load_rdkit()
    pillow = _load_pillow()
    audit_rows: list[dict[str, Any]] = []
    property_rows: list[dict[str, float]] = []

    for row in rows:
        image_path = _resolve_image_path(row.get("image_path", ""), pair_dir)
        image_exists = bool(image_path and image_path.exists())
        width, height, image_error = _image_dimensions(image_path, pillow)
        canonical_check = _canonical_check(row, rdkit)
        props = _molecule_properties(row, rdkit)
        if props:
            property_rows.append(props)
        audit_rows.append(
            {
                **row,
                "resolved_image_path": str(image_path) if image_path else "",
                "image_exists": image_exists,
                "image_width": width,
                "image_height": height,
                "image_error": image_error,
                "image_size_ok": bool(width == expected_image_size and height == expected_image_size),
                **canonical_check,
                **props,
            }
        )

    sample_rows = _sample_rows(audit_rows, sample_count=sample_count, seed=seed)
    sample_pairs_path = pair_dir / "sample_pairs.csv"
    _write_rows(sample_pairs_path, sample_rows)
    contact_sheet_path = _write_contact_sheet(
        sample_rows=sample_rows,
        pair_dir=pair_dir,
        pillow=pillow,
        cols=contact_sheet_cols,
        thumb_size=contact_thumb_size,
    )
    audit_rows_path = pair_dir / "audit_rows.csv"
    _write_rows(audit_rows_path, audit_rows)

    summary = _summarize_audit(
        audit_rows=audit_rows,
        property_rows=property_rows,
        rdkit_available=bool(rdkit),
        pillow_available=bool(pillow),
        expected_image_size=expected_image_size,
        sample_pairs_path=sample_pairs_path,
        contact_sheet_path=contact_sheet_path,
        audit_rows_path=audit_rows_path,
    )
    (pair_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_rdkit() -> dict[str, Any] | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

        return {"Chem": Chem, "Crippen": Crippen, "Descriptors": Descriptors, "QED": QED, "rdMolDescriptors": rdMolDescriptors}
    except Exception:
        return None


def _load_pillow() -> dict[str, Any] | None:
    try:
        from PIL import Image, ImageDraw

        return {"Image": Image, "ImageDraw": ImageDraw}
    except Exception:
        return None


def _resolve_image_path(value: str, pair_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    candidate = pair_dir / path
    if candidate.exists():
        return candidate
    image_name_candidate = pair_dir / "images" / path.name
    if image_name_candidate.exists():
        return image_name_candidate
    return path


def _image_dimensions(image_path: Path | None, pillow: dict[str, Any] | None) -> tuple[int, int, str]:
    if image_path is None:
        return 0, 0, "missing_image_path"
    if not image_path.exists():
        return 0, 0, "missing_image"
    if not pillow:
        return 0, 0, "pillow_unavailable"
    try:
        with pillow["Image"].open(image_path) as image:
            width, height = image.size
        return int(width), int(height), ""
    except Exception as exc:
        return 0, 0, f"image_open_failed:{exc}"


def _canonical_check(row: dict[str, str], rdkit: dict[str, Any] | None) -> dict[str, Any]:
    if not rdkit:
        return {"rdkit_valid": "", "canonical_match": "", "canonical_error": "rdkit_unavailable"}
    smiles = row.get("input_smiles", "")
    stored = row.get("canonical_smiles", "")
    mol = rdkit["Chem"].MolFromSmiles(smiles)
    if mol is None:
        return {"rdkit_valid": False, "canonical_match": False, "canonical_error": "invalid_input_smiles"}
    canonical = rdkit["Chem"].MolToSmiles(mol, canonical=True)
    return {"rdkit_valid": True, "canonical_match": canonical == stored, "canonical_error": "" if canonical == stored else canonical}


def _molecule_properties(row: dict[str, str], rdkit: dict[str, Any] | None) -> dict[str, float]:
    if not rdkit:
        return {}
    smiles = row.get("canonical_smiles") or row.get("input_smiles", "")
    mol = rdkit["Chem"].MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "mw": float(rdkit["Descriptors"].MolWt(mol)),
        "logp": float(rdkit["Crippen"].MolLogP(mol)),
        "qed": float(rdkit["QED"].qed(mol)),
        "tpsa": float(rdkit["rdMolDescriptors"].CalcTPSA(mol)),
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
    }


def _sample_rows(rows: list[dict[str, Any]], sample_count: int, seed: int) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row.get("image_exists") and not row.get("image_error")]
    rng = random.Random(seed)
    if len(candidates) <= sample_count:
        return list(candidates)
    return rng.sample(candidates, sample_count)


def _write_contact_sheet(
    sample_rows: list[dict[str, Any]],
    pair_dir: Path,
    pillow: dict[str, Any] | None,
    cols: int,
    thumb_size: int,
) -> str:
    if not pillow or not sample_rows:
        return ""
    image_cls = pillow["Image"]
    draw_cls = pillow["ImageDraw"]
    cols = max(1, int(cols))
    rows = int(math.ceil(len(sample_rows) / cols))
    label_height = 34
    cell_w = int(thumb_size)
    cell_h = int(thumb_size + label_height)
    sheet = image_cls.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = draw_cls.Draw(sheet)

    for idx, row in enumerate(sample_rows):
        image_path = Path(str(row["resolved_image_path"]))
        try:
            with image_cls.open(image_path) as image:
                image = image.convert("RGB")
                image.thumbnail((thumb_size, thumb_size))
                x = (idx % cols) * cell_w + (cell_w - image.width) // 2
                y = (idx // cols) * cell_h + (thumb_size - image.height) // 2
                sheet.paste(image, (x, y))
        except Exception:
            continue
        label = f"{row.get('pair_id', '')} {row.get('canonical_smiles', '')}"[:32]
        draw.text(((idx % cols) * cell_w + 4, (idx // cols) * cell_h + thumb_size + 4), label, fill=(0, 0, 0))

    path = pair_dir / "sample_contact_sheet.png"
    sheet.save(path)
    return str(path)


def _summarize_audit(
    audit_rows: list[dict[str, Any]],
    property_rows: list[dict[str, float]],
    rdkit_available: bool,
    pillow_available: bool,
    expected_image_size: int,
    sample_pairs_path: Path,
    contact_sheet_path: str,
    audit_rows_path: Path,
) -> dict[str, Any]:
    total = len(audit_rows)
    image_exists = sum(1 for row in audit_rows if row["image_exists"])
    image_size_ok = sum(1 for row in audit_rows if row["image_size_ok"])
    canonical_checked = [row for row in audit_rows if row["canonical_match"] != ""]
    canonical_matches = sum(1 for row in canonical_checked if row["canonical_match"])
    summary: dict[str, Any] = {
        "pairs": float(total),
        "rdkit_available": rdkit_available,
        "pillow_available": pillow_available,
        "expected_image_size": float(expected_image_size),
        "image_exists": float(image_exists),
        "image_exists_fraction": float(image_exists / total) if total else 0.0,
        "image_size_ok": float(image_size_ok),
        "image_size_ok_fraction": float(image_size_ok / total) if total else 0.0,
        "canonical_checked": float(len(canonical_checked)),
        "canonical_matches": float(canonical_matches),
        "canonical_match_fraction": float(canonical_matches / len(canonical_checked)) if canonical_checked else 0.0,
        "sample_pairs": str(sample_pairs_path),
        "sample_contact_sheet": contact_sheet_path,
        "audit_rows": str(audit_rows_path),
    }
    for key in ("mw", "logp", "qed", "tpsa", "heavy_atoms"):
        values = [row[key] for row in property_rows if key in row]
        if values:
            summary[f"{key}_mean"] = float(sum(values) / len(values))
            summary[f"{key}_min"] = float(min(values))
            summary[f"{key}_max"] = float(max(values))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a SketchSMILES paired SMILES/image manifest.")
    parser.add_argument("--pair-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--expected-image-size", type=int, default=256)
    parser.add_argument("--contact-sheet-cols", type=int, default=8)
    parser.add_argument("--contact-thumb-size", type=int, default=160)
    args = parser.parse_args()

    summary = audit_pair_manifest(
        pair_dir=args.pair_dir,
        sample_count=args.sample_count,
        seed=args.seed,
        expected_image_size=args.expected_image_size,
        contact_sheet_cols=args.contact_sheet_cols,
        contact_thumb_size=args.contact_thumb_size,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
