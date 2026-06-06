#!/usr/bin/env python3
"""Prepare generated UniVideo molecule images for MolScribe OCR."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Mapping


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_understanding_condition.chem import scaffold_smiles  # noqa: E402
from sketchmol_understanding_condition.unified_condition_dataset import PROPERTY_COLUMNS, read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", required=True, type=Path)
    parser.add_argument("--eval-jsonl", required=True, type=Path)
    parser.add_argument("--generated-images-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--method", default="univideo_image_diffusion")
    parser.add_argument("--require-all-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = _read_rows(args.predictions_csv)
    samples = read_jsonl(args.eval_jsonl)
    sample_lookup = _sample_lookup(samples)
    out_rows = []
    missing_images = []

    for index, row in enumerate(predictions):
        image_path = args.generated_images_dir / f"generated_{index:05d}.png"
        if not image_path.exists():
            missing_images.append(str(image_path))
            if args.require_all_images:
                continue
            continue
        sample = _lookup_sample(row, sample_lookup, samples, index)
        out_rows.append(_benchmark_row(index, row, sample, image_path=image_path, method=args.method))

    if args.require_all_images and missing_images:
        preview = "\n".join(missing_images[:5])
        raise FileNotFoundError(f"Missing {len(missing_images)} generated images. First missing:\n{preview}")
    if not out_rows:
        raise ValueError(f"No generated images found in {args.generated_images_dir}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(args.output_csv, out_rows)
    summary = {
        "predictions_csv": str(args.predictions_csv),
        "eval_jsonl": str(args.eval_jsonl),
        "generated_images_dir": str(args.generated_images_dir),
        "output_csv": str(args.output_csv),
        "rows": len(out_rows),
        "missing_images": len(missing_images),
        "method": args.method,
    }
    args.output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _sample_lookup(samples) -> dict[str, object]:
    lookup: dict[str, object] = {}
    for sample in samples:
        condition_id = sample.metadata.get("condition_id", "")
        for key in (sample.sample_id, condition_id):
            if key:
                lookup.setdefault(key, sample)
    return lookup


def _lookup_sample(row: Mapping[str, str], lookup: Mapping[str, object], samples, index: int):
    for key in (row.get("sample_id", ""), row.get("condition_id", "")):
        if key and key in lookup:
            return lookup[key]
    if index < len(samples):
        return samples[index]
    raise KeyError(f"Cannot match prediction row {index} to eval JSONL sample")


def _benchmark_row(index: int, row: Mapping[str, str], sample, *, image_path: Path, method: str) -> dict[str, object]:
    condition_id = sample.metadata.get("condition_id", "") or row.get("condition_id", "")
    source_scaffold = _safe_scaffold(sample.source_smiles)
    target_scaffold = _safe_scaffold(sample.target_smiles)
    out: dict[str, object] = {
        "method": method,
        "row_index": index,
        "sample_id": sample.sample_id,
        "condition_id": condition_id,
        "pair_id": sample.metadata.get("pair_id", ""),
        "split": sample.split,
        "image_path": str(image_path.resolve()),
        "source_smiles": sample.source_smiles or row.get("source_smiles", ""),
        "target_smiles": sample.target_smiles or row.get("target_smiles", ""),
        "source_scaffold": source_scaffold,
        "target_scaffold": target_scaffold,
        "condition_properties": sample.condition_properties,
        "property_count": sample.property_count or row.get("property_count", ""),
        "instruction": sample.instruction or sample.prompt,
        "source_tanimoto_target": sample.source_tanimoto,
        "source_similarity_bin": sample.source_similarity_bin,
        "latent_mse": row.get("latent_mse", ""),
        "latent_mae": row.get("latent_mae", ""),
        "target_latent_cosine": row.get("target_latent_cosine", ""),
        "source_latent_cosine": row.get("source_latent_cosine", ""),
    }
    for prop in PROPERTY_COLUMNS:
        out[f"source_{prop}"] = sample.source_properties.get(prop, "")
        out[f"target_{prop}"] = sample.target_properties.get(prop, "")
        out[f"delta_{prop}"] = sample.property_deltas.get(prop, "")
        out[f"{prop}_active"] = sample.active_properties.get(prop, prop in _selected_props(sample.condition_properties))
        out[f"{prop}_direction"] = sample.directions.get(prop, "")
    return out


def _selected_props(text: str) -> set[str]:
    return {item.strip() for item in str(text or "").split(",") if item.strip()}


def _safe_scaffold(smiles: str) -> str:
    try:
        return scaffold_smiles(smiles) or ""
    except RuntimeError:
        return ""


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
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
    main()
