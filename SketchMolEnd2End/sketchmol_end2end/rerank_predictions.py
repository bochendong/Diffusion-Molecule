"""Offline candidate reranking diagnostics for SketchMolEnd2End runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path
from typing import Any

from sketch_smiles.audit_pairs import _load_pillow, _load_rdkit
from sketch_smiles.phase5a0_oracle_baseline import _fraction, _image_pair_metrics, _render_smiles, _write_rows
from sketch_smiles.phase5a1_learned_smiles_decoder import _scaffold_match, _set_rdkit_error_logging, _tanimoto


RERANK_MODES = {"beam", "predicted_fingerprint", "render_mse", "oracle_tanimoto"}


def run_rerank_diagnostic(
    predictions_csv: str | Path,
    output_dir: str | Path,
    rerank_modes: list[str] | None = None,
    image_size: int = 128,
    limit: int | None = None,
) -> dict[str, Any]:
    """Rerank saved beam candidates without retraining the image-to-structure model."""

    predictions_path = Path(predictions_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    modes = [_normalize_mode(mode) for mode in (rerank_modes or ["beam", "predicted_fingerprint", "render_mse", "oracle_tanimoto"])]

    rdkit = _load_rdkit()
    pillow = _load_pillow()
    if not rdkit:
        raise RuntimeError("RDKit is required for rerank diagnostics.")
    if "render_mse" in modes and not pillow:
        raise RuntimeError("Pillow is required for render_mse reranking.")
    _set_rdkit_error_logging(enabled=False)

    rows = _read_rows(predictions_path)
    if limit is not None:
        rows = rows[: int(limit)]

    all_result_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for mode in modes:
        mode_dir = output_path / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        render_dir = mode_dir / "rendered_candidates"
        result_rows = [
            _rerank_row(
                row=row,
                mode=mode,
                rdkit=rdkit,
                pillow=pillow,
                render_dir=render_dir,
                image_size=image_size,
            )
            for row in rows
        ]
        mode_predictions_path = mode_dir / "reranked_predictions.csv"
        _write_rows(mode_predictions_path, result_rows)
        summary = _summarize_rows(
            rows=result_rows,
            mode=mode,
            predictions_csv=predictions_path,
            output_dir=mode_dir,
            image_size=image_size,
        )
        summary["reranked_predictions"] = str(mode_predictions_path)
        (mode_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summaries.append(summary)
        all_result_rows.extend(result_rows)

    summary_path = output_path / "rerank_summary.json"
    summary_csv_path = output_path / "rerank_summary.csv"
    all_rows_path = output_path / "reranked_predictions.csv"
    payload = {
        "phase": "sketchmol_end2end_offline_rerank_diagnostic",
        "predictions_csv": str(predictions_path),
        "output_dir": str(output_path),
        "image_size": float(image_size),
        "limit": float(limit) if limit is not None else None,
        "modes": summaries,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_rows(summary_csv_path, summaries)
    _write_rows(all_rows_path, all_result_rows)
    payload["summary_json"] = str(summary_path)
    payload["summary_csv"] = str(summary_csv_path)
    payload["reranked_predictions"] = str(all_rows_path)
    return payload


def _rerank_row(
    row: dict[str, str],
    mode: str,
    rdkit: dict[str, Any],
    pillow: dict[str, Any],
    render_dir: Path,
    image_size: int,
) -> dict[str, Any]:
    candidates = _candidate_list(row)
    target_smiles = row.get("target_smiles", "")
    target_mol = rdkit["Chem"].MolFromSmiles(target_smiles) if target_smiles else None
    candidate_scores = _candidate_score_map(row)
    render_scores: dict[str, float] = {}
    render_errors: dict[str, str] = {}

    if mode == "beam":
        ranked = candidates
    elif mode == "predicted_fingerprint":
        ranked = _rank_descending(candidates, candidate_scores)
    elif mode == "oracle_tanimoto":
        target_scores = {smiles: _tanimoto(target_mol, rdkit["Chem"].MolFromSmiles(smiles), rdkit) for smiles in candidates}
        ranked = _rank_oracle(candidates, target_smiles=target_smiles, scores=target_scores)
        candidate_scores = target_scores
    elif mode == "render_mse":
        render_scores, render_errors = _render_mse_scores(
            row=row,
            candidates=candidates,
            render_dir=render_dir,
            image_size=image_size,
            rdkit=rdkit,
            pillow=pillow,
        )
        ranked = _rank_ascending(candidates, render_scores)
        candidate_scores = render_scores
    else:
        raise ValueError(f"Unsupported rerank mode: {mode}")

    top1 = ranked[0] if ranked else ""
    top1_mol = rdkit["Chem"].MolFromSmiles(top1) if top1 else None
    top1_tanimoto = _tanimoto(target_mol, top1_mol, rdkit)
    best_tanimoto = max((_tanimoto(target_mol, rdkit["Chem"].MolFromSmiles(smiles), rdkit) for smiles in candidates), default=0.0)
    scaffold_match = _scaffold_match(target_smiles, top1, rdkit)
    top1_score = candidate_scores.get(top1, 0.0) if top1 else 0.0

    result = {
        "pair_id": row.get("pair_id", ""),
        "rerank_mode": mode,
        "target_smiles": target_smiles,
        "generated_smiles": top1,
        "candidate_count": float(len(candidates)),
        "canonical_candidates": "|".join(ranked),
        "candidate_scores": "|".join(_format_score(candidate_scores.get(smiles, 0.0)) for smiles in ranked),
        "top1_valid": bool(top1),
        "top1_exact_match": bool(top1 and top1 == target_smiles),
        "topk_exact_match": bool(target_smiles in candidates),
        "top1_scaffold_match": bool(scaffold_match),
        "top1_target_tanimoto": float(top1_tanimoto),
        "mean_best_tanimoto": float(best_tanimoto),
        "top1_rerank_score": float(top1_score),
        "oracle_gap_tanimoto": float(best_tanimoto - top1_tanimoto),
        "source_image_path": row.get("source_image_path", ""),
        "target_image_path": row.get("target_image_path", ""),
    }
    if mode == "render_mse":
        result["render_mse_scores"] = "|".join(_format_score(render_scores.get(smiles, math.inf)) for smiles in ranked)
        result["render_errors"] = "|".join(render_errors.get(smiles, "") for smiles in ranked)
    return result


def _render_mse_scores(
    row: dict[str, str],
    candidates: list[str],
    render_dir: Path,
    image_size: int,
    rdkit: dict[str, Any],
    pillow: dict[str, Any],
) -> tuple[dict[str, float], dict[str, str]]:
    render_dir.mkdir(parents=True, exist_ok=True)
    target_image_path = _best_target_image_path(row)
    scores: dict[str, float] = {}
    errors: dict[str, str] = {}
    with tempfile.TemporaryDirectory(dir=render_dir) as tmp_dir:
        tmp_path = Path(tmp_dir)
        for idx, smiles in enumerate(candidates):
            generated_path = tmp_path / f"candidate_{idx:02d}.png"
            render_error = _render_smiles(smiles, generated_path, image_size=image_size, rdkit=rdkit)
            metrics = _image_pair_metrics(target_image_path, generated_path, pillow)
            if render_error:
                errors[smiles] = render_error
                scores[smiles] = math.inf
            elif metrics.get("image_compared") and metrics.get("image_mse") != "":
                scores[smiles] = float(metrics["image_mse"])
                errors[smiles] = ""
            else:
                scores[smiles] = math.inf
                errors[smiles] = str(metrics.get("image_compare_error", "image_compare_failed"))
    return scores, errors


def _candidate_list(row: dict[str, str]) -> list[str]:
    raw = row.get("beam_canonical_candidates") or row.get("canonical_candidates") or row.get("raw_samples") or ""
    candidates: list[str] = []
    seen: set[str] = set()
    for value in raw.split("|"):
        smiles = value.strip()
        if smiles and smiles not in seen:
            seen.add(smiles)
            candidates.append(smiles)
    return candidates


def _candidate_score_map(row: dict[str, str]) -> dict[str, float]:
    candidates = _split_pipe(row.get("canonical_candidates", ""))
    scores = [_safe_float(value) for value in _split_pipe(row.get("candidate_condition_tanimotos", ""))]
    return {smiles: score for smiles, score in zip(candidates, scores)}


def _best_target_image_path(row: dict[str, str]) -> Path | None:
    for key in ("target_image_path", "source_image_path"):
        value = row.get(key, "")
        if value:
            path = Path(value)
            if path.exists():
                return path
    return None


def _rank_descending(candidates: list[str], scores: dict[str, float]) -> list[str]:
    return [smiles for _idx, smiles in sorted(enumerate(candidates), key=lambda item: (scores.get(item[1], 0.0), -item[0]), reverse=True)]


def _rank_ascending(candidates: list[str], scores: dict[str, float]) -> list[str]:
    return [smiles for _idx, smiles in sorted(enumerate(candidates), key=lambda item: (scores.get(item[1], math.inf), item[0]))]


def _rank_oracle(candidates: list[str], target_smiles: str, scores: dict[str, float]) -> list[str]:
    return [
        smiles
        for _idx, smiles in sorted(
            enumerate(candidates),
            key=lambda item: (item[1] == target_smiles, scores.get(item[1], 0.0), -item[0]),
            reverse=True,
        )
    ]


def _summarize_rows(
    rows: list[dict[str, Any]],
    mode: str,
    predictions_csv: Path,
    output_dir: Path,
    image_size: int,
) -> dict[str, Any]:
    total = len(rows)
    tanimoto_values = [float(row["top1_target_tanimoto"]) for row in rows if row.get("top1_target_tanimoto") != ""]
    best_values = [float(row["mean_best_tanimoto"]) for row in rows if row.get("mean_best_tanimoto") != ""]
    gap_values = [float(row["oracle_gap_tanimoto"]) for row in rows if row.get("oracle_gap_tanimoto") != ""]
    return {
        "rerank_mode": mode,
        "predictions_csv": str(predictions_csv),
        "output_dir": str(output_dir),
        "image_size": float(image_size),
        "eval_examples": float(total),
        "top1_valid": float(_count(rows, "top1_valid")),
        "top1_valid_fraction": _fraction(_count(rows, "top1_valid"), total),
        "top1_exact_matches": float(_count(rows, "top1_exact_match")),
        "top1_exact_match_fraction": _fraction(_count(rows, "top1_exact_match"), total),
        "topk_exact_matches": float(_count(rows, "topk_exact_match")),
        "topk_exact_match_fraction": _fraction(_count(rows, "topk_exact_match"), total),
        "top1_scaffold_matches": float(_count(rows, "top1_scaffold_match")),
        "top1_scaffold_match_fraction": _fraction(_count(rows, "top1_scaffold_match"), total),
        "top1_target_tanimoto": _mean(tanimoto_values),
        "mean_best_tanimoto": _mean(best_values),
        "mean_oracle_gap_tanimoto": _mean(gap_values),
        "mean_candidate_count": _mean([float(row["candidate_count"]) for row in rows if row.get("candidate_count") != ""]),
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _split_pipe(value: str) -> list[str]:
    return [item.strip() for item in value.split("|") if item.strip()]


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_score(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{float(value):.6f}"


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if bool(row.get(key)))


def _normalize_mode(mode: str) -> str:
    value = mode.strip().lower().replace("-", "_")
    aliases = {
        "sequence": "beam",
        "none": "beam",
        "fingerprint": "predicted_fingerprint",
        "predicted_fp": "predicted_fingerprint",
        "condition_fingerprint": "predicted_fingerprint",
        "image_mse": "render_mse",
        "render": "render_mse",
        "oracle": "oracle_tanimoto",
        "target_tanimoto": "oracle_tanimoto",
    }
    value = aliases.get(value, value)
    if value not in RERANK_MODES:
        raise ValueError(f"Unsupported rerank mode {mode!r}; expected one of {sorted(RERANK_MODES)}.")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline rerank diagnostics for SketchMolEnd2End predictions.")
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rerank-modes", default="beam,predicted_fingerprint,render_mse,oracle_tanimoto")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    modes = [mode.strip() for mode in args.rerank_modes.split(",") if mode.strip()]
    summary = run_rerank_diagnostic(
        predictions_csv=args.predictions_csv,
        output_dir=args.output_dir,
        rerank_modes=modes,
        image_size=args.image_size,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
