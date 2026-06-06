#!/usr/bin/env python3
"""Run MolScribe OCR on a CSV containing an image_path column."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from molscribe import MolScribe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--image-csv", required=True, type=Path)
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.image_csv)
    if not rows:
        raise ValueError(f"No rows found in {args.image_csv}")
    if args.image_column not in rows[0]:
        raise ValueError(f"{args.image_csv} must contain an {args.image_column!r} column")

    paths = [str(row.get(args.image_column, "")) for row in rows]
    missing = [path for path in paths if not path or not Path(path).exists()]
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} image files. First missing:\n{preview}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = MolScribe(str(args.model_path), device=device)
    smiles, scores = _predict(model, paths, batch_size=args.batch_size)
    if len(smiles) != len(rows):
        raise ValueError(f"MolScribe returned {len(smiles)} predictions for {len(rows)} images")

    for row, pred, score in zip(rows, smiles, scores):
        row["SMILES"] = pred or ""
        row["molscribe_score"] = "" if score is None else score
    _write_rows(args.image_csv, rows)
    print(f"save to {args.image_csv}")
    print("done")


def _predict(model: MolScribe, paths: list[str], *, batch_size: int) -> tuple[list[str], list[float | None]]:
    try:
        outputs = model.predict_image_files(paths, return_atoms_bonds=False, return_confidence=True)
    except TypeError:
        outputs = None

    if outputs is not None:
        if isinstance(outputs, tuple):
            smiles = list(outputs[0])
            scores = [None] * len(smiles)
            return smiles, scores
        smiles = [str(output.get("smiles", "") or "") for output in outputs]
        scores = [output.get("confidence") for output in outputs]
        return smiles, scores

    if hasattr(model, "predict_images_from_csv"):
        result = model.predict_images_from_csv(paths, batch_size=batch_size)
        if isinstance(result, tuple) and len(result) >= 3:
            return list(result[0]), [float(score) for score in result[2]]
        if isinstance(result, tuple):
            smiles = list(result[0])
            return smiles, [None] * len(smiles)

    result = model.predict_image_files(paths)
    if isinstance(result, tuple):
        smiles = list(result[0])
        return smiles, [None] * len(smiles)
    return [str(item) for item in result], [None] * len(result)


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
