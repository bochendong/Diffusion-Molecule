"""Diagnose decoder sensitivity to oracle, noisy, planner, and calibrated latents."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import write_examples_csv
from .experiment import _candidate_surface_metrics, _canonical_pool, _write_predictions
from .features import MOLECULE_LATENT_VERSION, matrix_from_examples
from .oracle_latent_diffusion import OracleLatentSmilesDiffusion
from .phase2_planned_decoder import (
    _append_decoder_pool_membership,
    _decoder_pool_metrics,
    _load_decoder_train_pool,
    _load_splits,
    _mean_cosine,
    _resolve_oracle_model_dir,
    _retag_candidates,
)
from .report import summarize_predictions_csv
from .verifier import score_candidates, summarize_scores


@dataclass(frozen=True)
class LatentSource:
    name: str
    kind: str
    latents: np.ndarray


def run_latent_sensitivity_diagnostic(
    decoder_dir: str | Path,
    output_dir: str | Path = "outputs/runs/phase2_latent_sensitivity",
    decoder_pool_dir: str | Path | None = None,
    planner_run_dir: str | Path | None = None,
    calibrated_run_dir: str | Path | None = None,
    dataset_csv: str | Path | None = None,
    train_csv: str | Path | None = None,
    eval_csv: str | Path | None = None,
    feature_dim: int = 256,
    latent_dim: int | None = None,
    top_k: int = 8,
    noisy_cosines: tuple[float, ...] = (0.32, 0.38, 0.63, 0.78),
    interpolation_alphas: tuple[float, ...] = (0.25, 0.50, 0.75),
    train_fraction: float = 0.8,
    max_eval_tasks: int | None = None,
    seed: int = 7,
    decoder_device: str = "auto",
) -> list[dict[str, float | str]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decoder_model_dir = _resolve_oracle_model_dir(decoder_dir)
    decoder = OracleLatentSmilesDiffusion.load(decoder_model_dir, device=decoder_device)
    decoder_dim = int(decoder.config.condition_dim)
    latent_dim = int(latent_dim or decoder_dim)
    if latent_dim != decoder_dim:
        raise ValueError(f"latent_dim={latent_dim} must match decoder condition_dim={decoder_dim}.")

    train_examples, eval_examples = _load_splits(dataset_csv, train_csv, eval_csv, train_fraction=train_fraction, seed=seed)
    if max_eval_tasks is not None:
        eval_examples = eval_examples[: max(1, int(max_eval_tasks))]
    _, eval_targets, _ = matrix_from_examples(eval_examples, feature_dim=feature_dim, latent_dim=latent_dim)
    planner_train_pool = _canonical_pool(example.target_smiles for example in train_examples)
    pool_root = Path(decoder_pool_dir) if decoder_pool_dir else decoder_model_dir.parent
    decoder_train_pool = _load_decoder_train_pool(pool_root)

    sources = _build_latent_sources(
        eval_targets=eval_targets,
        planner_run_dir=planner_run_dir,
        calibrated_run_dir=calibrated_run_dir,
        noisy_cosines=noisy_cosines,
        interpolation_alphas=interpolation_alphas,
        seed=seed,
    )

    summaries: list[dict[str, float | str]] = []
    for source in sources:
        source_dir = output_dir / source.name
        source_dir.mkdir(parents=True, exist_ok=True)
        decoded = _retag_candidates(decoder.decode(source.latents, top_k=top_k), origin=f"latent_sensitivity_{source.name}")
        scores_by_task = [score_candidates(example, candidates) for example, candidates in zip(eval_examples, decoded)]
        metrics = summarize_scores(scores_by_task)
        metrics.update(
            {
                "latent_source": source.name,
                "latent_source_kind": source.kind,
                "latent_cosine_to_oracle": _mean_cosine(source.latents, eval_targets),
                "latent_mse_to_oracle": float(np.mean((source.latents - eval_targets) ** 2)),
                "eval_tasks": float(len(eval_examples)),
            }
        )
        metrics.update(_candidate_surface_metrics(scores_by_task, planner_train_pool, eval_examples))
        if decoder_train_pool:
            metrics.update(_decoder_pool_metrics(scores_by_task, decoder_train_pool))

        (source_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        predictions_path = source_dir / "predictions.csv"
        _write_predictions(predictions_path, eval_examples, scores_by_task, train_pool=planner_train_pool)
        _append_decoder_pool_membership(predictions_path, decoder_train_pool)
        summarize_predictions_csv(predictions_path, out_dir=source_dir)
        summaries.append(metrics)

    summary_rows = [_summary_row(item) for item in summaries]
    _write_summary_csv(output_dir / "source_summary.csv", summary_rows)
    (output_dir / "source_summary.json").write_text(json.dumps(summary_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_examples_csv(output_dir / "eval_examples.csv", eval_examples)
    run_config = {
        "phase": "phase2_latent_sensitivity_diagnostic",
        "research_question": "Does the decoder fail because planner latents are inaccurate, off-manifold, or because the decoder is insensitive to latent changes?",
        "decoder_dir": str(decoder_model_dir),
        "decoder_pool_dir": str(pool_root),
        "planner_run_dir": str(planner_run_dir) if planner_run_dir else None,
        "calibrated_run_dir": str(calibrated_run_dir) if calibrated_run_dir else None,
        "dataset_csv": str(dataset_csv) if dataset_csv else None,
        "train_csv": str(train_csv) if train_csv else None,
        "eval_csv": str(eval_csv) if eval_csv else None,
        "feature_dim": feature_dim,
        "latent_dim": latent_dim,
        "molecule_latent_version": MOLECULE_LATENT_VERSION,
        "top_k": top_k,
        "noisy_cosines": list(noisy_cosines),
        "interpolation_alphas": list(interpolation_alphas),
        "max_eval_tasks": max_eval_tasks,
        "train_fraction": train_fraction,
        "seed": seed,
        "decoder_device": getattr(decoder, "device_name", decoder_device),
        "sources": [source.name for source in sources],
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_rows


def _build_latent_sources(
    eval_targets: np.ndarray,
    planner_run_dir: str | Path | None,
    calibrated_run_dir: str | Path | None,
    noisy_cosines: tuple[float, ...],
    interpolation_alphas: tuple[float, ...],
    seed: int,
) -> list[LatentSource]:
    sources = [LatentSource("oracle_target", "oracle", eval_targets.astype(np.float32))]
    rng = np.random.default_rng(seed)
    for cosine in noisy_cosines:
        safe_cosine = min(0.999, max(-0.999, float(cosine)))
        sources.append(
            LatentSource(
                f"noisy_oracle_c{_format_float_for_name(safe_cosine)}",
                "noisy_oracle",
                _noisy_latents_at_cosine(eval_targets, safe_cosine, rng),
            )
        )

    planner_latents: np.ndarray | None = None
    if planner_run_dir:
        planner_latents = _load_eval_latents(Path(planner_run_dir) / "planner_eval_latents.npy", eval_targets.shape)
        sources.append(LatentSource("planner_predicted", "planner", planner_latents))
        for alpha in interpolation_alphas:
            safe_alpha = min(1.0, max(0.0, float(alpha)))
            sources.append(
                LatentSource(
                    f"interp_planner{_format_float_for_name(safe_alpha)}",
                    "target_planner_interpolation",
                    _normalize_rows((1.0 - safe_alpha) * eval_targets + safe_alpha * planner_latents),
                )
            )

    if calibrated_run_dir:
        calibrated_path = Path(calibrated_run_dir) / "calibrated_eval_latents.npy"
        calibrated_latents = _load_eval_latents(calibrated_path, eval_targets.shape)
        sources.append(LatentSource("calibrated_predicted", "calibrated_planner", calibrated_latents))
        if planner_latents is not None:
            sources.append(
                LatentSource(
                    "interp_calibrated050",
                    "target_calibrated_interpolation",
                    _normalize_rows(0.5 * eval_targets + 0.5 * calibrated_latents),
                )
            )
    return sources


def _load_eval_latents(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing latent artifact: {path}")
    values = np.load(path).astype(np.float32)
    if values.shape != expected_shape:
        raise ValueError(f"{path} shape={values.shape} does not match expected shape={expected_shape}.")
    return values


def _noisy_latents_at_cosine(targets: np.ndarray, cosine: float, rng: np.random.Generator) -> np.ndarray:
    targets = _normalize_rows(targets)
    noise = rng.normal(size=targets.shape).astype(np.float32)
    projection = np.sum(noise * targets, axis=1, keepdims=True) * targets
    orthogonal = _normalize_rows(noise - projection)
    out = cosine * targets + np.sqrt(max(0.0, 1.0 - cosine**2)) * orthogonal
    return _normalize_rows(out)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-8, None)


def _format_float_for_name(value: float) -> str:
    return f"{value:.2f}".replace("-", "m").replace(".", "_")


def _summary_row(metrics: dict[str, float | str]) -> dict[str, float | str]:
    keys = [
        "latent_source",
        "latent_source_kind",
        "latent_cosine_to_oracle",
        "latent_mse_to_oracle",
        "top1_validity",
        "top1_target_tanimoto",
        "mean_best_tanimoto",
        "topk_target_hit",
        "top1_scaffold_match",
        "top1_property_success",
        "topk_property_success",
        "top1_property_delta_success",
        "topk_property_delta_success",
        "top1_decoder_train_pool_member",
        "candidate_decoder_train_pool_member_fraction",
    ]
    return {key: metrics.get(key, "") for key in keys}


def _write_summary_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_float_tuple(value: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None or value.strip() == "":
        return default
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose decoder sensitivity to different latent sources.")
    parser.add_argument("--decoder-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/runs/phase2_latent_sensitivity")
    parser.add_argument("--decoder-pool-dir", default=None)
    parser.add_argument("--planner-run-dir", default=None)
    parser.add_argument("--calibrated-run-dir", default=None)
    parser.add_argument("--dataset-csv", default=None)
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--eval-csv", default=None)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--noisy-cosines", default="0.32,0.38,0.63,0.78")
    parser.add_argument("--interpolation-alphas", default="0.25,0.50,0.75")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--max-eval-tasks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--decoder-device", default="auto")
    args = parser.parse_args()
    summary = run_latent_sensitivity_diagnostic(
        decoder_dir=args.decoder_dir,
        output_dir=args.output_dir,
        decoder_pool_dir=args.decoder_pool_dir,
        planner_run_dir=args.planner_run_dir,
        calibrated_run_dir=args.calibrated_run_dir,
        dataset_csv=args.dataset_csv,
        train_csv=args.train_csv,
        eval_csv=args.eval_csv,
        feature_dim=args.feature_dim,
        latent_dim=args.latent_dim,
        top_k=args.top_k,
        noisy_cosines=_parse_float_tuple(args.noisy_cosines, (0.32, 0.38, 0.63, 0.78)),
        interpolation_alphas=_parse_float_tuple(args.interpolation_alphas, (0.25, 0.50, 0.75)),
        train_fraction=args.train_fraction,
        max_eval_tasks=args.max_eval_tasks,
        seed=args.seed,
        decoder_device=args.decoder_device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
