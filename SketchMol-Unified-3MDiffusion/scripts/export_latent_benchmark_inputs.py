#!/usr/bin/env python3
"""Export Unified 3M eval latents for benchmark materialization."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_unified_3m_diffusion.benchmark_export import write_edit_latent_benchmark_inputs  # noqa: E402
from sketchmol_unified_3m_diffusion.unified_condition_dataset import (  # noqa: E402
    EDIT_GENERATION,
    UnifiedConditionSample,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-jsonl", required=True, type=Path)
    parser.add_argument("--latents-npy", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fingerprint-dim", type=int, default=None)
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variant", default="full")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = [sample for sample in read_jsonl(args.eval_jsonl) if sample.task_type == EDIT_GENERATION]
    if not samples:
        raise ValueError(f"No edit_generation rows found in {args.eval_jsonl}")
    latents = np.load(args.latents_npy)
    samples, latents, alignment = _align_samples_and_latents(
        samples,
        latents,
        limit=args.limit,
        metrics_json=args.metrics_json,
        predictions_csv=args.predictions_csv,
    )
    fingerprint_dim = args.fingerprint_dim or _fingerprint_dim_from_metrics(args.metrics_json) or 512
    summary = write_edit_latent_benchmark_inputs(
        samples,
        latents,
        args.output_dir,
        fingerprint_dim=fingerprint_dim,
        variant=args.variant,
    )
    summary.update(
        {
            "eval_jsonl": str(args.eval_jsonl),
            "latents_npy": str(args.latents_npy),
            "metrics_json": str(args.metrics_json) if args.metrics_json else None,
            "predictions_csv": str(args.predictions_csv) if args.predictions_csv else None,
            "alignment": alignment,
        }
    )
    (args.output_dir / "benchmark_export_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _align_samples_and_latents(
    samples: list[UnifiedConditionSample],
    latents: np.ndarray,
    *,
    limit: int | None,
    metrics_json: Path | None,
    predictions_csv: Path | None,
) -> tuple[list[UnifiedConditionSample], np.ndarray, dict[str, object]]:
    latents = np.asarray(latents, dtype=np.float32)
    if latents.ndim != 2:
        raise ValueError(f"Expected a 2D latent array, got shape {latents.shape}")

    total_edit_samples = len(samples)

    if limit is not None and limit > 0:
        samples = samples[:limit]
        latents = latents[:limit]

    if predictions_csv is not None and predictions_csv.exists():
        samples = _samples_from_predictions_csv(predictions_csv, samples)
        latents = latents[: len(samples)]
    elif limit is None:
        metrics_rows = _rows_from_metrics(metrics_json)
        target_rows = metrics_rows if metrics_rows is not None else int(latents.shape[0])
        if len(samples) != target_rows or latents.shape[0] != target_rows:
            if latents.shape[0] > len(samples):
                raise ValueError(
                    f"Latent rows ({latents.shape[0]}) exceed available edit samples ({len(samples)})"
                )
            samples = samples[:target_rows]
            latents = latents[:target_rows]

    if latents.shape[0] != len(samples):
        raise ValueError(f"Latent rows ({latents.shape[0]}) do not match samples ({len(samples)})")

    alignment = {
        "edit_samples_in_eval_jsonl": total_edit_samples,
        "latent_rows": int(latents.shape[0]),
        "matched_rows": len(samples),
    }
    if metrics_json is not None and metrics_json.exists():
        alignment["metrics_rows"] = _rows_from_metrics(metrics_json)
    if predictions_csv is not None and predictions_csv.exists():
        alignment["predictions_csv"] = str(predictions_csv)
    return samples, latents, alignment


def _samples_from_predictions_csv(
    predictions_csv: Path,
    samples: list[UnifiedConditionSample],
) -> list[UnifiedConditionSample]:
    by_id = {sample.sample_id: sample for sample in samples}
    ordered: list[UnifiedConditionSample] = []
    missing: list[str] = []
    with predictions_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                continue
            sample = by_id.get(sample_id)
            if sample is None:
                missing.append(sample_id)
                continue
            ordered.append(sample)
    if not ordered:
        raise ValueError(f"No sample_id rows found in {predictions_csv}")
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} prediction sample_id values were not found in eval jsonl. First missing: {preview}"
        )
    return ordered


def _rows_from_metrics(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("rows")
    try:
        rows = int(value)
    except (TypeError, ValueError):
        return None
    return rows if rows > 0 else None


def _fingerprint_dim_from_metrics(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("fingerprint_dim")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
