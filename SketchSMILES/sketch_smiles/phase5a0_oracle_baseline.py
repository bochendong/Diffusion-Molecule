"""Phase 5A-0 oracle paired-output baseline.

This establishes the paired SMILES/image evaluation contract before training a
learned synchronized decoder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

from .audit_pairs import _load_pillow, _load_rdkit, _resolve_image_path


def run_oracle_paired_baseline(
    pair_dir: str | Path,
    output_dir: str | Path = "outputs/runs/phase5a0_oracle_baseline",
    train_fraction: float = 0.8,
    seed: int = 7,
    limit: int | None = None,
    image_size: int = 256,
    sample_count: int = 64,
    contact_sheet_cols: int = 8,
    contact_thumb_size: int = 144,
) -> dict[str, Any]:
    pair_dir = Path(pair_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = pair_dir / "pairs.csv"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Missing paired manifest: {pairs_path}")

    rdkit = _load_rdkit()
    pillow = _load_pillow()
    if not rdkit:
        raise RuntimeError("RDKit is required for Phase 5A-0 oracle rendering.")
    if not pillow:
        raise RuntimeError("Pillow is required for Phase 5A-0 image consistency metrics.")

    rows = _read_rows(pairs_path)
    if limit is not None:
        rows = rows[: int(limit)]
    train_rows, eval_rows = _split_rows(rows, train_fraction=train_fraction, seed=seed)
    _write_rows(output_dir / "train_pairs.csv", train_rows)
    _write_rows(output_dir / "eval_pairs.csv", eval_rows)

    oracle_image_dir = output_dir / "oracle_images"
    oracle_image_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[dict[str, Any]] = []
    for row in eval_rows:
        prediction_rows.append(
            _oracle_prediction_row(
                row=row,
                pair_dir=pair_dir,
                oracle_image_dir=oracle_image_dir,
                image_size=image_size,
                rdkit=rdkit,
                pillow=pillow,
            )
        )

    predictions_path = output_dir / "oracle_predictions.csv"
    _write_rows(predictions_path, prediction_rows)
    sample_rows = _sample_rows(prediction_rows, sample_count=sample_count, seed=seed)
    sample_predictions_path = output_dir / "sample_predictions.csv"
    _write_rows(sample_predictions_path, sample_rows)
    contact_sheet_path = _write_oracle_contact_sheet(
        sample_rows=sample_rows,
        pillow=pillow,
        cols=contact_sheet_cols,
        thumb_size=contact_thumb_size,
        output_path=output_dir / "sample_contact_sheet.png",
    )

    metrics = _summarize_predictions(
        prediction_rows=prediction_rows,
        train_rows=train_rows,
        eval_rows=eval_rows,
        pair_dir=pair_dir,
        output_dir=output_dir,
        predictions_path=predictions_path,
        sample_predictions_path=sample_predictions_path,
        contact_sheet_path=contact_sheet_path,
        train_fraction=train_fraction,
        seed=seed,
        image_size=image_size,
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "phase": "phase5a0_oracle_paired_output_baseline",
                "research_question": "What is the upper-bound paired-output contract when SMILES and molecular sketch are both oracle-rendered from the same molecule?",
                "pair_dir": str(pair_dir),
                "pairs_csv": str(pairs_path),
                "output_dir": str(output_dir),
                "train_fraction": train_fraction,
                "seed": seed,
                "limit": limit,
                "image_size": image_size,
                "sample_count": sample_count,
                "contact_sheet_cols": contact_sheet_cols,
                "contact_thumb_size": contact_thumb_size,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics


def _oracle_prediction_row(
    row: dict[str, str],
    pair_dir: Path,
    oracle_image_dir: Path,
    image_size: int,
    rdkit: dict[str, Any],
    pillow: dict[str, Any],
) -> dict[str, Any]:
    pair_id = row.get("pair_id", "")
    target_smiles = row.get("canonical_smiles", "") or row.get("input_smiles", "")
    generated_smiles, smiles_error = _canonicalize(target_smiles, rdkit)
    generated_image_path = oracle_image_dir / f"{pair_id}.png"
    render_error = _render_smiles(generated_smiles, generated_image_path, image_size=image_size, rdkit=rdkit)
    target_image_path = _resolve_image_path(row.get("image_path", ""), pair_dir)
    image_metrics = _image_pair_metrics(target_image_path, generated_image_path, pillow)
    exact_smiles_match = bool(generated_smiles and generated_smiles == target_smiles)
    generated_image_exists = generated_image_path.exists()
    target_image_exists = bool(target_image_path and target_image_path.exists())
    return {
        "pair_id": pair_id,
        "target_smiles": target_smiles,
        "generated_smiles": generated_smiles,
        "smiles_valid": bool(generated_smiles),
        "smiles_error": smiles_error,
        "exact_smiles_match": exact_smiles_match,
        "target_image_path": str(target_image_path) if target_image_path else "",
        "generated_image_path": str(generated_image_path),
        "target_image_exists": target_image_exists,
        "generated_image_exists": generated_image_exists,
        "render_error": render_error,
        "paired_output_success": bool(generated_smiles and exact_smiles_match and generated_image_exists),
        **image_metrics,
    }


def _canonicalize(smiles: str, rdkit: dict[str, Any]) -> tuple[str, str]:
    mol = rdkit["Chem"].MolFromSmiles(smiles)
    if mol is None:
        return "", "invalid_smiles"
    return rdkit["Chem"].MolToSmiles(mol, canonical=True), ""


def _render_smiles(smiles: str, image_path: Path, image_size: int, rdkit: dict[str, Any]) -> str:
    if not smiles:
        return "missing_smiles"
    try:
        from rdkit.Chem import Draw

        mol = rdkit["Chem"].MolFromSmiles(smiles)
        if mol is None:
            return "invalid_smiles"
        image = Draw.MolToImage(mol, size=(image_size, image_size))
        image.save(image_path)
        return ""
    except Exception as exc:
        return f"render_failed:{exc}"


def _image_pair_metrics(target_image_path: Path | None, generated_image_path: Path, pillow: dict[str, Any]) -> dict[str, Any]:
    if target_image_path is None or not target_image_path.exists():
        return {"image_compared": False, "image_exact_match": False, "image_mse": "", "image_rmse": "", "image_compare_error": "missing_target_image"}
    if not generated_image_path.exists():
        return {"image_compared": False, "image_exact_match": False, "image_mse": "", "image_rmse": "", "image_compare_error": "missing_generated_image"}
    try:
        with pillow["Image"].open(target_image_path) as target_image:
            target = target_image.convert("RGB")
        with pillow["Image"].open(generated_image_path) as generated_image:
            generated = generated_image.convert("RGB")
        if target.size != generated.size:
            return {
                "image_compared": False,
                "image_exact_match": False,
                "image_mse": "",
                "image_rmse": "",
                "image_compare_error": f"size_mismatch:{target.size}!={generated.size}",
            }
        diff = pillow["ImageChops"].difference(target, generated)
        stat = pillow["ImageStat"].Stat(diff)
        extrema = diff.getextrema()
        max_abs_diff = max((channel_extrema[1] for channel_extrema in extrema), default=0)
        denom = max(1, target.width * target.height * len(stat.sum2))
        mse = float(sum(stat.sum2) / denom)
        return {
            "image_compared": True,
            "image_exact_match": bool(max_abs_diff == 0),
            "image_mse": float(mse),
            "image_rmse": float(math.sqrt(mse)),
            "image_max_abs_diff": float(max_abs_diff),
            "image_compare_error": "",
        }
    except Exception as exc:
        return {"image_compared": False, "image_exact_match": False, "image_mse": "", "image_rmse": "", "image_compare_error": f"compare_failed:{exc}"}


def _summarize_predictions(
    prediction_rows: list[dict[str, Any]],
    train_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    pair_dir: Path,
    output_dir: Path,
    predictions_path: Path,
    sample_predictions_path: Path,
    contact_sheet_path: str,
    train_fraction: float,
    seed: int,
    image_size: int,
) -> dict[str, Any]:
    total = len(prediction_rows)
    compared = [row for row in prediction_rows if row["image_compared"]]
    image_mse_values = [float(row["image_mse"]) for row in compared if row["image_mse"] != ""]
    return {
        "phase": "phase5a0_oracle_paired_output_baseline",
        "pair_dir": str(pair_dir),
        "output_dir": str(output_dir),
        "train_fraction": float(train_fraction),
        "seed": float(seed),
        "image_size": float(image_size),
        "pairs": float(len(train_rows) + len(eval_rows)),
        "train_pairs": float(len(train_rows)),
        "eval_pairs": float(len(eval_rows)),
        "smiles_valid": float(_count(prediction_rows, "smiles_valid")),
        "smiles_valid_fraction": _fraction(_count(prediction_rows, "smiles_valid"), total),
        "exact_smiles_matches": float(_count(prediction_rows, "exact_smiles_match")),
        "exact_smiles_match_fraction": _fraction(_count(prediction_rows, "exact_smiles_match"), total),
        "target_images": float(_count(prediction_rows, "target_image_exists")),
        "target_image_fraction": _fraction(_count(prediction_rows, "target_image_exists"), total),
        "generated_images": float(_count(prediction_rows, "generated_image_exists")),
        "generated_image_fraction": _fraction(_count(prediction_rows, "generated_image_exists"), total),
        "image_compared": float(len(compared)),
        "image_compared_fraction": _fraction(len(compared), total),
        "image_exact_matches": float(_count(prediction_rows, "image_exact_match")),
        "image_exact_match_fraction": _fraction(_count(prediction_rows, "image_exact_match"), len(compared)),
        "image_mse_mean": float(sum(image_mse_values) / len(image_mse_values)) if image_mse_values else 0.0,
        "image_mse_max": float(max(image_mse_values)) if image_mse_values else 0.0,
        "paired_output_success": float(_count(prediction_rows, "paired_output_success")),
        "paired_output_success_fraction": _fraction(_count(prediction_rows, "paired_output_success"), total),
        "predictions": str(predictions_path),
        "sample_predictions": str(sample_predictions_path),
        "sample_contact_sheet": contact_sheet_path,
    }


def _write_oracle_contact_sheet(
    sample_rows: list[dict[str, Any]],
    pillow: dict[str, Any],
    cols: int,
    thumb_size: int,
    output_path: Path,
) -> str:
    if not sample_rows:
        return ""
    image_cls = pillow["Image"]
    draw_cls = pillow["ImageDraw"]
    cols = max(1, int(cols))
    rows = int(math.ceil(len(sample_rows) / cols))
    label_height = 42
    cell_w = int(thumb_size * 2)
    cell_h = int(thumb_size + label_height)
    sheet = image_cls.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = draw_cls.Draw(sheet)
    for idx, row in enumerate(sample_rows):
        x0 = (idx % cols) * cell_w
        y0 = (idx // cols) * cell_h
        _paste_thumb(image_cls, sheet, row.get("target_image_path", ""), x0, y0, thumb_size)
        _paste_thumb(image_cls, sheet, row.get("generated_image_path", ""), x0 + thumb_size, y0, thumb_size)
        label = f"{row.get('pair_id', '')} mse={row.get('image_mse', '')}"[:44]
        draw.text((x0 + 4, y0 + thumb_size + 4), label, fill=(0, 0, 0))
    sheet.save(output_path)
    return str(output_path)


def _paste_thumb(image_cls: Any, sheet: Any, image_path: str, x0: int, y0: int, thumb_size: int) -> None:
    try:
        with image_cls.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_size, thumb_size))
            x = x0 + (thumb_size - image.width) // 2
            y = y0 + (thumb_size - image.height) // 2
            sheet.paste(image, (x, y))
    except Exception:
        return


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


def _split_rows(rows: list[dict[str, str]], train_fraction: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    split_idx = int(round(len(shuffled) * float(train_fraction)))
    split_idx = min(max(split_idx, 0), len(shuffled))
    return shuffled[:split_idx], shuffled[split_idx:]


def _sample_rows(rows: list[dict[str, Any]], sample_count: int, seed: int) -> list[dict[str, Any]]:
    candidates = list(rows)
    rng = random.Random(seed)
    if len(candidates) <= sample_count:
        return candidates
    return rng.sample(candidates, sample_count)


def _count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if bool(row.get(key)))


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5A-0 oracle paired-output baseline.")
    parser.add_argument("--pair-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/runs/phase5a0_oracle_baseline")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--contact-sheet-cols", type=int, default=8)
    parser.add_argument("--contact-thumb-size", type=int, default=144)
    args = parser.parse_args()

    metrics = run_oracle_paired_baseline(
        pair_dir=args.pair_dir,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        seed=args.seed,
        limit=args.limit,
        image_size=args.image_size,
        sample_count=args.sample_count,
        contact_sheet_cols=args.contact_sheet_cols,
        contact_thumb_size=args.contact_thumb_size,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
