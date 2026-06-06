#!/usr/bin/env python3
"""Run MolScribe OCR on a CSV containing an image_path column."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--image-csv", required=True, type=Path)
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--raw-smiles-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For vendored MolScribe, use decoder token SMILES when graph conversion returns empty/invalid.",
    )
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

    import torch
    from molscribe import MolScribe

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = MolScribe(str(args.model_path), device=device)
    smiles, scores, diagnostics = _predict(
        model,
        paths,
        batch_size=args.batch_size,
        raw_smiles_fallback=args.raw_smiles_fallback,
    )
    if len(smiles) != len(rows):
        raise ValueError(f"MolScribe returned {len(smiles)} predictions for {len(rows)} images")

    for row, pred, score, diagnostic in zip(rows, smiles, scores, diagnostics):
        row.update(diagnostic)
        row["SMILES"] = pred or ""
        row["molscribe_score"] = "" if score is None else score
    _write_rows(args.image_csv, rows)
    print(f"save to {args.image_csv}")
    print("done")


def _predict(
    model: Any,
    paths: list[str],
    *,
    batch_size: int,
    raw_smiles_fallback: bool = True,
) -> tuple[list[str], list[float | None], list[dict[str, object]]]:
    if raw_smiles_fallback and _looks_like_vendored_molscribe(model):
        try:
            return _predict_vendored_with_raw_fallback(model, paths, batch_size=batch_size)
        except Exception as exc:
            print(f"WARNING: raw-token MolScribe fallback failed; using public API path: {exc}")

    try:
        outputs = model.predict_image_files(paths, return_atoms_bonds=False, return_confidence=True)
    except TypeError:
        outputs = None

    if outputs is not None:
        if isinstance(outputs, tuple):
            smiles = list(outputs[0])
            scores = [None] * len(smiles)
            return smiles, scores, _empty_diagnostics(len(smiles), source="predict_image_files_tuple")
        smiles = [str(output.get("smiles", "") or "") for output in outputs]
        scores = [output.get("confidence") for output in outputs]
        return smiles, scores, _empty_diagnostics(len(smiles), source="predict_image_files_dict")

    if hasattr(model, "predict_images_from_csv"):
        result = model.predict_images_from_csv(paths, batch_size=batch_size)
        if isinstance(result, tuple) and len(result) >= 3:
            smiles = list(result[0])
            return smiles, [float(score) for score in result[2]], _empty_diagnostics(
                len(smiles),
                source="predict_images_from_csv",
            )
        if isinstance(result, tuple):
            smiles = list(result[0])
            return smiles, [None] * len(smiles), _empty_diagnostics(len(smiles), source="predict_images_from_csv")

    result = model.predict_image_files(paths)
    if isinstance(result, tuple):
        smiles = list(result[0])
        return smiles, [None] * len(smiles), _empty_diagnostics(len(smiles), source="predict_image_files_plain")
    smiles = [str(item) for item in result]
    return smiles, [None] * len(smiles), _empty_diagnostics(len(smiles), source="predict_image_files_plain")


def _looks_like_vendored_molscribe(model: Any) -> bool:
    return all(hasattr(model, name) for name in ("encoder", "decoder", "transform"))


def _predict_vendored_with_raw_fallback(
    model: Any,
    paths: list[str],
    *,
    batch_size: int,
) -> tuple[list[str], list[float | None], list[dict[str, object]]]:
    import cv2
    import torch
    from molscribe.chemistry import convert_graph_to_smiles

    input_images = []
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Could not read image for MolScribe OCR: {path}")
        input_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    predictions = []
    device = model.device
    for start in range(0, len(input_images), batch_size):
        batch_images = input_images[start : start + batch_size]
        images = [model.transform(image=image, keypoints=[])["image"] for image in batch_images]
        images = torch.stack(images, dim=0).to(device)
        with torch.no_grad():
            features, hiddens = model.encoder(images)
            predictions.extend(model.decoder.decode(features, hiddens))

    raw_smiles = [_clean_smiles(pred.get("chartok_coords", {}).get("smiles", "")) for pred in predictions]
    node_coords = [pred.get("chartok_coords", {}).get("coords", []) for pred in predictions]
    node_symbols = [pred.get("chartok_coords", {}).get("symbols", []) for pred in predictions]
    edges = [pred.get("edges", []) for pred in predictions]
    token_scores = [_mean_score(pred.get("chartok_coords", {}).get("scores")) for pred in predictions]
    graph_smiles, _molblocks, _success = convert_graph_to_smiles(
        node_coords,
        node_symbols,
        edges,
        images=input_images,
    )
    graph_smiles = [_clean_smiles(value) for value in graph_smiles]

    final_smiles: list[str] = []
    diagnostics: list[dict[str, object]] = []
    for graph_value, raw_value in zip(graph_smiles, raw_smiles):
        final, source = _select_smiles(graph_value, raw_value)
        final_smiles.append(final)
        diagnostics.append(
            {
                "molscribe_graph_smiles": graph_value,
                "molscribe_raw_smiles": raw_value,
                "molscribe_decode_source": source,
            }
        )
    return final_smiles, token_scores, diagnostics


def _select_smiles(graph_value: object, raw_value: object) -> tuple[str, str]:
    graph_smiles = _clean_smiles(graph_value)
    raw_smiles = _clean_smiles(raw_value)
    if _usable_smiles(graph_smiles):
        return graph_smiles, "graph"
    if _usable_smiles(raw_smiles):
        return raw_smiles, "raw_token_fallback"
    return "", "empty"


def _clean_smiles(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return re.sub(r"\[S@?@?SP[123]\]", "S", text)


def _usable_smiles(value: object) -> bool:
    text = _clean_smiles(value)
    return bool(text and "invalid" not in text.lower() and text != "<invalid>")


def _mean_score(value: object) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return None
    return float(arr.mean())


def _empty_diagnostics(count: int, *, source: str) -> list[dict[str, object]]:
    return [{"molscribe_decode_source": source} for _ in range(count)]


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
