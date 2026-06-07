#!/usr/bin/env python3
"""Run MolScribe OCR on a CSV containing an image_path column."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_understanding_condition.molscribe_images import (  # noqa: E402
    load_preprocessed_rgb_image,
)


def _prepend_sys_path_ordered(entries: list[Path]) -> None:
    """Put runtime overlays on sys.path in an exact left-to-right order."""

    ordered: list[str] = []
    for entry in entries:
        text = str(entry)
        if text and text not in ordered:
            ordered.append(text)
    for text in reversed(ordered):
        while text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


def _ensure_vendored_molscribe_path() -> Path | None:
    """Prefer SketchMol evaluate/molscribe and the onmt220 overlay over pip copies."""

    path_entries: list[Path] = []
    onmt_overlay = os.environ.get("SUCC_ONMT_OVERLAY") or os.environ.get(
        "SKETCHMOL_ONMT_OVERLAY", "/scratch/bdong/python_overlays/onmt220"
    )
    if onmt_overlay:
        overlay_dir = Path(onmt_overlay)
        if overlay_dir.is_dir():
            path_entries.append(overlay_dir)

    candidates = [
        os.environ.get("SUCC_MOLSCRIBE_WORKDIR"),
        os.environ.get("SKETCHMOL_MOLSCRIBE_WORKDIR"),
        str(REPO_DIR / "Research/Molecule Generation/SketchMol/SketchMol-v1-main/evaluate"),
        str(REPO_DIR / "Research/Molecule Generation/SketchMol/SketchMol-v1-main"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate)
        if (root / "molscribe").is_dir():
            evaluate_dir = root
        elif (root / "evaluate" / "molscribe").is_dir():
            evaluate_dir = root / "evaluate"
        else:
            continue
        path_entries.append(evaluate_dir)
        path_entries.append(evaluate_dir.parent)
        _prepend_sys_path_ordered(path_entries)
        return evaluate_dir
    _prepend_sys_path_ordered(path_entries)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--image-csv", required=True, type=Path)
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backend",
        choices=("sketchmol", "custom"),
        default="sketchmol",
        help=(
            "sketchmol: SketchMol evaluate/predict_csv.py graph decode + postprocess; "
            "custom: local wrapper with optional binarization/raw-token fallback."
        ),
    )
    parser.add_argument(
        "--preprocess-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optional UC binarization before MolScribe. SketchMol predict_csv.py "
            "does not use this; default is off to match SketchMol."
        ),
    )
    parser.add_argument(
        "--raw-smiles-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Custom backend only: use decoder token SMILES when graph conversion fails.",
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

    evaluate_dir = _ensure_vendored_molscribe_path()
    from molscribe import MolScribe

    if evaluate_dir is not None:
        print(f"Using SketchMol MolScribe from {evaluate_dir}", flush=True)
    if args.device != "cpu" and not torch.cuda.is_available():
        print("WARNING: CUDA unavailable; MolScribe OCR requires GPU for reliable decoding.", flush=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = MolScribe(str(args.model_path), device=device)
    if args.backend == "sketchmol":
        smiles, scores, diagnostics = _predict_sketchmol(
            model,
            paths,
            batch_size=args.batch_size,
            preprocess_images=args.preprocess_images,
        )
    else:
        smiles, scores, diagnostics = _predict(
            model,
            paths,
            batch_size=args.batch_size,
            preprocess_images=args.preprocess_images,
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


def _postprocess_smiles_sketchmol(
    input_smiles: list[object],
    scores: list[object],
) -> tuple[list[str], float, float]:
    """Match SketchMol evaluate/predict_csv.py postprocessing."""

    cleaned: list[str | None] = []
    broken_num = 0
    low_score = 0
    total = len(input_smiles)
    for index in range(total):
        text = _clean_smiles(input_smiles[index])
        score = float(scores[index])
        if "." in text:
            cleaned.append(None)
            broken_num += 1
        elif "invalid" in text.lower():
            cleaned.append(None)
        elif score < 0.85:
            low_score += 1
            cleaned.append(text)
        else:
            cleaned.append(text)
    broken_rate = broken_num / total if total else 0.0
    low_score_rate = low_score / total if total else 0.0
    return ["" if value is None else value for value in cleaned], broken_rate, low_score_rate


def _predict_sketchmol(
    model: Any,
    paths: list[str],
    *,
    batch_size: int,
    preprocess_images: bool = False,
) -> tuple[list[str], list[float | None], list[dict[str, object]]]:
    if preprocess_images:
        input_images = [
            load_preprocessed_rgb_image(path, preprocess=True) for path in paths
        ]
        if hasattr(model, "predict_imagespredict_images_from_csv_helper"):
            result = model.predict_imagespredict_images_from_csv_helper(input_images, batch_size)
        else:
            raise ValueError("MolScribe model missing predict_imagespredict_images_from_csv_helper")
    elif hasattr(model, "predict_images_from_csv"):
        # Match SketchMol evaluate/predict_csv.py: cv2.imread paths, internal CropWhite/ToGray.
        result = model.predict_images_from_csv(paths, batch_size)
    else:
        input_images = [load_preprocessed_rgb_image(path, preprocess=False) for path in paths]
        result = model.predict_imagespredict_images_from_csv_helper(input_images, batch_size)
    if isinstance(result, tuple) and len(result) >= 3:
        graph_smiles, _molblock, token_scores = result[:3]
    else:
        raise ValueError(f"Unexpected MolScribe output shape: {type(result)!r}")

    smiles, broken_rate, low_score_rate = _postprocess_smiles_sketchmol(graph_smiles, token_scores)
    diagnostics = [
        {
            "molscribe_decode_source": "sketchmol_graph",
            "molscribe_graph_smiles": _clean_smiles(raw),
            "molscribe_raw_smiles": "",
        }
        for raw in graph_smiles
    ]
    print(
        f"SketchMol OCR postprocess: broken_rate={broken_rate:.3f} low_score_rate={low_score_rate:.3f}",
        flush=True,
    )
    return smiles, [float(score) for score in token_scores], diagnostics


def _predict(
    model: Any,
    paths: list[str],
    *,
    batch_size: int,
    preprocess_images: bool = True,
    raw_smiles_fallback: bool = False,
) -> tuple[list[str], list[float | None], list[dict[str, object]]]:
    if _looks_like_vendored_molscribe(model):
        return _predict_vendored(
            model,
            paths,
            batch_size=batch_size,
            preprocess_images=preprocess_images,
            raw_smiles_fallback=raw_smiles_fallback,
        )

    try:
        outputs = model.predict_image_files(paths, return_atoms_bonds=False, return_confidence=True)
    except TypeError:
        outputs = None

    if outputs is not None:
        if isinstance(outputs, tuple):
            smiles = [_clean_smiles(value) for value in outputs[0]]
            scores = [None] * len(smiles)
            return smiles, scores, _empty_diagnostics(len(smiles), source="predict_image_files_tuple")
        smiles = [_clean_smiles(output.get("smiles", "") or "") for output in outputs]
        scores = [output.get("confidence") for output in outputs]
        return smiles, scores, _empty_diagnostics(len(smiles), source="predict_image_files_dict")

    if hasattr(model, "predict_images_from_csv"):
        result = model.predict_images_from_csv(paths, batch_size=batch_size)
        if isinstance(result, tuple) and len(result) >= 3:
            smiles = [_clean_smiles(value) for value in result[0]]
            return smiles, [float(score) for score in result[2]], _empty_diagnostics(
                len(smiles),
                source="predict_images_from_csv",
            )
        if isinstance(result, tuple):
            smiles = [_clean_smiles(value) for value in result[0]]
            return smiles, [None] * len(smiles), _empty_diagnostics(len(smiles), source="predict_images_from_csv")

    result = model.predict_image_files(paths)
    if isinstance(result, tuple):
        smiles = [_clean_smiles(value) for value in result[0]]
        return smiles, [None] * len(smiles), _empty_diagnostics(len(smiles), source="predict_image_files_plain")
    smiles = [_clean_smiles(str(item)) for item in result]
    return smiles, [None] * len(smiles), _empty_diagnostics(len(smiles), source="predict_image_files_plain")


def _looks_like_vendored_molscribe(model: Any) -> bool:
    return all(hasattr(model, name) for name in ("encoder", "decoder", "transform"))


def _predict_vendored(
    model: Any,
    paths: list[str],
    *,
    batch_size: int,
    preprocess_images: bool,
    raw_smiles_fallback: bool,
) -> tuple[list[str], list[float | None], list[dict[str, object]]]:
    import torch
    from molscribe.chemistry import convert_graph_to_smiles

    input_images = []
    for path in paths:
        image = load_preprocessed_rgb_image(path, preprocess=preprocess_images)
        input_images.append(image)

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
        num_workers=min(16, max(1, len(input_images))),
    )
    graph_smiles = [_clean_smiles(value) for value in graph_smiles]

    final_smiles: list[str] = []
    diagnostics: list[dict[str, object]] = []
    for graph_value, raw_value in zip(graph_smiles, raw_smiles):
        final, source = _select_smiles(graph_value, raw_value, allow_raw_fallback=raw_smiles_fallback)
        final_smiles.append(final)
        diagnostics.append(
            {
                "molscribe_graph_smiles": graph_value,
                "molscribe_raw_smiles": raw_value,
                "molscribe_decode_source": source,
            }
        )
    return final_smiles, token_scores, diagnostics


def _select_smiles(
    graph_value: object,
    raw_value: object,
    *,
    allow_raw_fallback: bool,
) -> tuple[str, str]:
    graph_smiles = _clean_smiles(graph_value)
    raw_smiles = _clean_smiles(raw_value)
    if _usable_smiles(graph_smiles):
        return graph_smiles, "graph"
    if allow_raw_fallback and _usable_smiles(raw_smiles):
        return raw_smiles, "raw_token_fallback"
    return "", "empty"


def _clean_smiles(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return re.sub(r"\[S@?@?SP[123]\]", "S", text)


def _usable_smiles(value: object) -> bool:
    text = _clean_smiles(value)
    if not text or "invalid" in text.lower() or text == "<invalid>":
        return False
    if len(text) > 512:
        return False
    if text.count("(") != text.count(")"):
        return False
    if text.count("[") != text.count("]"):
        return False
    try:
        from sketchmol_understanding_condition.chem import canonical_smiles

        return canonical_smiles(text) is not None
    except Exception:
        return False


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
