"""Phase 2D: fine-tune decoder with strong oracle anchors and light planner exposure."""

from __future__ import annotations

import argparse
import csv
import json
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
from .schema import BenchmarkExample
from .verifier import score_candidates, summarize_scores


def run_phase2_oracle_anchored_decoder(
    oracle_decoder_dir: str | Path,
    planner_run_dir: str | Path,
    output_dir: str | Path = "outputs/runs/phase2_oracle_anchored_decoder",
    decoder_pool_dir: str | Path | None = None,
    calibrated_run_dir: str | Path | None = None,
    dataset_csv: str | Path | None = None,
    train_csv: str | Path | None = None,
    eval_csv: str | Path | None = None,
    feature_dim: int = 256,
    latent_dim: int | None = None,
    top_k: int = 8,
    train_fraction: float = 0.8,
    seed: int = 7,
    decoder_device: str = "auto",
    decoder_finetune_epochs: int = 4,
    decoder_finetune_batch_size: int = 128,
    decoder_finetune_lr: float = 2e-5,
    decoder_finetune_weight_decay: float = 1e-4,
    decoder_oracle_repeats: int = 8,
    decoder_noisy_repeats: int = 1,
    decoder_noisy_cosines: tuple[float, ...] = (0.78, 0.90),
    decoder_planner_repeats: int = 1,
    decoder_calibrated_repeats: int = 1,
    decoder_interpolation_repeats: int = 1,
    decoder_interpolation_alphas: tuple[float, ...] = (0.10, 0.25),
) -> dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    oracle_model_dir = _resolve_oracle_model_dir(oracle_decoder_dir)
    decoder = OracleLatentSmilesDiffusion.load(oracle_model_dir, device=decoder_device)
    base_decoder_config = json.loads((oracle_model_dir / "config.json").read_text(encoding="utf-8"))
    decoder_dim = int(decoder.config.condition_dim)
    latent_dim = int(latent_dim or decoder_dim)
    if latent_dim != decoder_dim:
        raise ValueError(f"latent_dim={latent_dim} must match decoder condition_dim={decoder_dim}.")

    train_examples, eval_examples = _load_splits(dataset_csv, train_csv, eval_csv, train_fraction=train_fraction, seed=seed)
    _, train_targets, _ = matrix_from_examples(train_examples, feature_dim=feature_dim, latent_dim=latent_dim)
    _, eval_targets, _ = matrix_from_examples(eval_examples, feature_dim=feature_dim, latent_dim=latent_dim)
    planner_run_dir = Path(planner_run_dir)
    train_planner_latents = _load_latents(planner_run_dir / "planner_train_latents.npy", train_targets.shape)
    eval_planner_latents = _load_latents(planner_run_dir / "planner_eval_latents.npy", eval_targets.shape)
    train_calibrated_latents: np.ndarray | None = None
    eval_calibrated_latents: np.ndarray | None = None
    if calibrated_run_dir:
        calibrated_root = Path(calibrated_run_dir)
        train_calibrated_latents = _load_latents(calibrated_root / "calibrated_train_latents.npy", train_targets.shape)
        eval_calibrated_latents = _load_latents(calibrated_root / "calibrated_eval_latents.npy", eval_targets.shape)

    finetune_smiles, finetune_conditions, aug_summary = _build_oracle_anchored_rows(
        train_examples=train_examples,
        target_latents=train_targets,
        planner_latents=train_planner_latents,
        calibrated_latents=train_calibrated_latents,
        seed=seed,
        oracle_repeats=decoder_oracle_repeats,
        noisy_repeats=decoder_noisy_repeats,
        noisy_cosines=decoder_noisy_cosines,
        planner_repeats=decoder_planner_repeats,
        calibrated_repeats=decoder_calibrated_repeats,
        interpolation_repeats=decoder_interpolation_repeats,
        interpolation_alphas=decoder_interpolation_alphas,
    )
    decoder.config.epochs = int(decoder_finetune_epochs)
    decoder.config.batch_size = int(decoder_finetune_batch_size)
    decoder.config.lr = float(decoder_finetune_lr)
    decoder.config.weight_decay = float(decoder_finetune_weight_decay)
    decoder.config.seed = int(seed)
    decoder.fit_conditioned(
        finetune_smiles,
        condition_latents=finetune_conditions,
        reset_model=False,
        history_backend="phase2d_oracle_anchored_decoder_finetune",
    )

    planner_train_pool = _canonical_pool(example.target_smiles for example in train_examples)
    pool_root = Path(decoder_pool_dir) if decoder_pool_dir else oracle_model_dir.parent
    decoder_train_pool = _load_decoder_train_pool(pool_root)
    source_metrics = _evaluate_sources(
        decoder=decoder,
        eval_examples=eval_examples,
        eval_targets=eval_targets,
        eval_planner_latents=eval_planner_latents,
        eval_calibrated_latents=eval_calibrated_latents,
        output_dir=output_dir,
        top_k=top_k,
        planner_train_pool=planner_train_pool,
        decoder_train_pool=decoder_train_pool,
    )
    metrics = dict(source_metrics["planner_predicted"])
    metrics.update(
        {
            "train_tasks": float(len(train_examples)),
            "eval_tasks": float(len(eval_examples)),
            "planner_train_latent_mse": float(np.mean((train_planner_latents - train_targets) ** 2)),
            "planner_train_latent_cosine": _mean_cosine(train_planner_latents, train_targets),
            "planner_latent_mse": float(np.mean((eval_planner_latents - eval_targets) ** 2)),
            "planner_latent_cosine": _mean_cosine(eval_planner_latents, eval_targets),
            **{key: float(value) for key, value in aug_summary.items()},
        }
    )
    if train_calibrated_latents is not None and eval_calibrated_latents is not None:
        metrics.update(
            {
                "calibrated_train_latent_mse": float(np.mean((train_calibrated_latents - train_targets) ** 2)),
                "calibrated_train_latent_cosine": _mean_cosine(train_calibrated_latents, train_targets),
                "calibrated_latent_mse": float(np.mean((eval_calibrated_latents - eval_targets) ** 2)),
                "calibrated_latent_cosine": _mean_cosine(eval_calibrated_latents, eval_targets),
            }
        )
    for source_name, source_metric in source_metrics.items():
        for key in ("top1_validity", "top1_target_tanimoto", "mean_best_tanimoto", "topk_target_hit", "top1_scaffold_match"):
            metrics[f"{source_name}_{key}"] = float(source_metric.get(key, 0.0))

    run_config = {
        "phase": "phase2d_oracle_anchored_decoder",
        "research_question": "Can decoder fine-tuning preserve oracle target control while adding light robustness to planner and calibrated latents?",
        "oracle_decoder_dir": str(oracle_model_dir),
        "decoder_pool_dir": str(pool_root),
        "planner_run_dir": str(planner_run_dir),
        "calibrated_run_dir": str(calibrated_run_dir) if calibrated_run_dir else None,
        "base_decoder_config": base_decoder_config,
        "dataset_csv": str(dataset_csv) if dataset_csv else None,
        "train_csv": str(train_csv) if train_csv else None,
        "eval_csv": str(eval_csv) if eval_csv else None,
        "feature_dim": feature_dim,
        "latent_dim": latent_dim,
        "molecule_latent_version": MOLECULE_LATENT_VERSION,
        "top_k": top_k,
        "train_fraction": train_fraction,
        "seed": seed,
        "decoder_device": getattr(decoder, "device_name", decoder_device),
        "decoder_finetune_epochs": decoder_finetune_epochs,
        "decoder_finetune_batch_size": decoder_finetune_batch_size,
        "decoder_finetune_lr": decoder_finetune_lr,
        "decoder_finetune_weight_decay": decoder_finetune_weight_decay,
        "decoder_oracle_repeats": decoder_oracle_repeats,
        "decoder_noisy_repeats": decoder_noisy_repeats,
        "decoder_noisy_cosines": list(decoder_noisy_cosines),
        "decoder_planner_repeats": decoder_planner_repeats,
        "decoder_calibrated_repeats": decoder_calibrated_repeats,
        "decoder_interpolation_repeats": decoder_interpolation_repeats,
        "decoder_interpolation_alphas": list(decoder_interpolation_alphas),
        "decoder_augmentation": aug_summary,
        "decoder_history": decoder.history,
        "decoder": {
            "mode": "phase1_initialized_oracle_anchored_planner_robust_decoder",
            "ranking": "decoder_likelihood_plus_rdkit_validity_no_target_oracle",
            "can_leave_training_pool": True,
            "fine_tune_goal": "preserve oracle_target control while lightly exposing planner and calibrated latent shifts",
        },
    }

    decoder.save(output_dir / "decoder")
    np.save(output_dir / "planner_train_latents.npy", train_planner_latents.astype(np.float32))
    np.save(output_dir / "planner_eval_latents.npy", eval_planner_latents.astype(np.float32))
    if train_calibrated_latents is not None and eval_calibrated_latents is not None:
        np.save(output_dir / "calibrated_train_latents.npy", train_calibrated_latents.astype(np.float32))
        np.save(output_dir / "calibrated_eval_latents.npy", eval_calibrated_latents.astype(np.float32))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_examples_csv(output_dir / "train_examples.csv", train_examples)
    write_examples_csv(output_dir / "eval_examples.csv", eval_examples)
    return metrics


def _build_oracle_anchored_rows(
    train_examples: list[BenchmarkExample],
    target_latents: np.ndarray,
    planner_latents: np.ndarray,
    calibrated_latents: np.ndarray | None,
    seed: int,
    oracle_repeats: int,
    noisy_repeats: int,
    noisy_cosines: tuple[float, ...],
    planner_repeats: int,
    calibrated_repeats: int,
    interpolation_repeats: int,
    interpolation_alphas: tuple[float, ...],
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    target_latents = np.asarray(target_latents, dtype=np.float32)
    planner_latents = np.asarray(planner_latents, dtype=np.float32)
    if target_latents.shape != planner_latents.shape:
        raise ValueError("target_latents and planner_latents must have the same shape.")
    if calibrated_latents is not None:
        calibrated_latents = np.asarray(calibrated_latents, dtype=np.float32)
        if target_latents.shape != calibrated_latents.shape:
            raise ValueError("target_latents and calibrated_latents must have the same shape.")

    rng = np.random.default_rng(seed)
    smiles_rows: list[str] = []
    condition_rows: list[np.ndarray] = []
    counts = {
        "decoder_oracle_rows": 0,
        "decoder_noisy_rows": 0,
        "decoder_planner_rows": 0,
        "decoder_calibrated_rows": 0,
        "decoder_interpolation_rows": 0,
    }
    target_smiles = [example.target_smiles for example in train_examples]

    def add_rows(name: str, conditions: np.ndarray, repeats: int) -> None:
        repeats = max(0, int(repeats))
        if repeats == 0:
            return
        condition_values = np.asarray(conditions, dtype=np.float32)
        for _ in range(repeats):
            smiles_rows.extend(target_smiles)
            condition_rows.extend(condition_values)
            counts[name] += len(target_smiles)

    add_rows("decoder_oracle_rows", target_latents, oracle_repeats)
    for cosine in noisy_cosines:
        noisy = _noisy_latents_at_cosine(target_latents, cosine, rng)
        add_rows("decoder_noisy_rows", noisy, noisy_repeats)
    for alpha in interpolation_alphas:
        safe_alpha = min(1.0, max(0.0, float(alpha)))
        add_rows("decoder_interpolation_rows", (1.0 - safe_alpha) * target_latents + safe_alpha * planner_latents, interpolation_repeats)
        if calibrated_latents is not None:
            add_rows("decoder_interpolation_rows", (1.0 - safe_alpha) * target_latents + safe_alpha * calibrated_latents, interpolation_repeats)
    add_rows("decoder_planner_rows", planner_latents, planner_repeats)
    if calibrated_latents is not None:
        add_rows("decoder_calibrated_rows", calibrated_latents, calibrated_repeats)
    if not condition_rows:
        raise ValueError("No decoder fine-tuning rows were generated.")
    conditions = np.stack(condition_rows).astype(np.float32)
    counts["decoder_finetune_rows"] = len(smiles_rows)
    return smiles_rows, conditions, counts


def _evaluate_sources(
    decoder: OracleLatentSmilesDiffusion,
    eval_examples: list[BenchmarkExample],
    eval_targets: np.ndarray,
    eval_planner_latents: np.ndarray,
    eval_calibrated_latents: np.ndarray | None,
    output_dir: Path,
    top_k: int,
    planner_train_pool: set[str],
    decoder_train_pool: set[str],
) -> dict[str, dict[str, float]]:
    sources: list[tuple[str, np.ndarray]] = [
        ("oracle_target", eval_targets),
        ("planner_predicted", eval_planner_latents),
    ]
    if eval_calibrated_latents is not None:
        sources.append(("calibrated_predicted", eval_calibrated_latents))
    source_metrics: dict[str, dict[str, float]] = {}
    summary_rows: list[dict[str, float | str]] = []
    for source_name, latents in sources:
        source_dir = output_dir / source_name
        source_dir.mkdir(parents=True, exist_ok=True)
        origin = "phase2_jepa_oracle_anchored_decoder" if source_name == "planner_predicted" else f"phase2d_{source_name}"
        decoded = _retag_candidates(decoder.decode(latents, top_k=top_k), origin=origin)
        scores_by_task = [score_candidates(example, candidates) for example, candidates in zip(eval_examples, decoded)]
        metrics = summarize_scores(scores_by_task)
        metrics.update(
            {
                "latent_source": source_name,
                "latent_cosine_to_oracle": _mean_cosine(latents, eval_targets),
                "latent_mse_to_oracle": float(np.mean((latents - eval_targets) ** 2)),
            }
        )
        metrics.update(_candidate_surface_metrics(scores_by_task, planner_train_pool, eval_examples))
        if decoder_train_pool:
            metrics.update(_decoder_pool_metrics(scores_by_task, decoder_train_pool))
        predictions_path = source_dir / "predictions.csv"
        _write_predictions(predictions_path, eval_examples, scores_by_task, train_pool=planner_train_pool)
        _append_decoder_pool_membership(predictions_path, decoder_train_pool)
        summarize_predictions_csv(predictions_path, out_dir=source_dir)
        (source_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        source_metrics[source_name] = metrics
        summary_rows.append(_summary_row(metrics))
        if source_name == "planner_predicted":
            _write_predictions(output_dir / "predictions.csv", eval_examples, scores_by_task, train_pool=planner_train_pool)
            _append_decoder_pool_membership(output_dir / "predictions.csv", decoder_train_pool)
            summarize_predictions_csv(output_dir / "predictions.csv", out_dir=output_dir)
    _write_summary_csv(output_dir / "source_summary.csv", summary_rows)
    (output_dir / "source_summary.json").write_text(json.dumps(summary_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return source_metrics


def _load_latents(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing latent artifact: {path}")
    values = np.load(path).astype(np.float32)
    if values.shape != expected_shape:
        raise ValueError(f"{path} shape={values.shape} does not match expected shape={expected_shape}.")
    return values


def _noisy_latents_at_cosine(targets: np.ndarray, cosine: float, rng: np.random.Generator) -> np.ndarray:
    targets = _normalize_rows(targets)
    safe_cosine = min(0.999, max(-0.999, float(cosine)))
    noise = rng.normal(size=targets.shape).astype(np.float32)
    projection = np.sum(noise * targets, axis=1, keepdims=True) * targets
    orthogonal = _normalize_rows(noise - projection)
    out = safe_cosine * targets + np.sqrt(max(0.0, 1.0 - safe_cosine**2)) * orthogonal
    return _normalize_rows(out)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-8, None)


def _summary_row(metrics: dict[str, float | str]) -> dict[str, float | str]:
    keys = [
        "latent_source",
        "latent_cosine_to_oracle",
        "latent_mse_to_oracle",
        "top1_validity",
        "top1_target_tanimoto",
        "mean_best_tanimoto",
        "topk_target_hit",
        "top1_scaffold_match",
        "top1_property_success",
        "topk_property_success",
        "top1_decoder_train_pool_member",
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
    parser = argparse.ArgumentParser(description="Run Phase 2D oracle-anchored robust decoder fine-tuning.")
    parser.add_argument("--oracle-decoder-dir", required=True)
    parser.add_argument("--planner-run-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/runs/phase2_oracle_anchored_decoder")
    parser.add_argument("--decoder-pool-dir", default=None)
    parser.add_argument("--calibrated-run-dir", default=None)
    parser.add_argument("--dataset-csv", default=None)
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--eval-csv", default=None)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--decoder-device", default="auto")
    parser.add_argument("--decoder-finetune-epochs", type=int, default=4)
    parser.add_argument("--decoder-finetune-batch-size", type=int, default=128)
    parser.add_argument("--decoder-finetune-lr", type=float, default=2e-5)
    parser.add_argument("--decoder-finetune-weight-decay", type=float, default=1e-4)
    parser.add_argument("--decoder-oracle-repeats", type=int, default=8)
    parser.add_argument("--decoder-noisy-repeats", type=int, default=1)
    parser.add_argument("--decoder-noisy-cosines", default="0.78,0.90")
    parser.add_argument("--decoder-planner-repeats", type=int, default=1)
    parser.add_argument("--decoder-calibrated-repeats", type=int, default=1)
    parser.add_argument("--decoder-interpolation-repeats", type=int, default=1)
    parser.add_argument("--decoder-interpolation-alphas", default="0.10,0.25")
    args = parser.parse_args()
    metrics = run_phase2_oracle_anchored_decoder(
        oracle_decoder_dir=args.oracle_decoder_dir,
        planner_run_dir=args.planner_run_dir,
        output_dir=args.output_dir,
        decoder_pool_dir=args.decoder_pool_dir,
        calibrated_run_dir=args.calibrated_run_dir,
        dataset_csv=args.dataset_csv,
        train_csv=args.train_csv,
        eval_csv=args.eval_csv,
        feature_dim=args.feature_dim,
        latent_dim=args.latent_dim,
        top_k=args.top_k,
        train_fraction=args.train_fraction,
        seed=args.seed,
        decoder_device=args.decoder_device,
        decoder_finetune_epochs=args.decoder_finetune_epochs,
        decoder_finetune_batch_size=args.decoder_finetune_batch_size,
        decoder_finetune_lr=args.decoder_finetune_lr,
        decoder_finetune_weight_decay=args.decoder_finetune_weight_decay,
        decoder_oracle_repeats=args.decoder_oracle_repeats,
        decoder_noisy_repeats=args.decoder_noisy_repeats,
        decoder_noisy_cosines=_parse_float_tuple(args.decoder_noisy_cosines, (0.78, 0.90)),
        decoder_planner_repeats=args.decoder_planner_repeats,
        decoder_calibrated_repeats=args.decoder_calibrated_repeats,
        decoder_interpolation_repeats=args.decoder_interpolation_repeats,
        decoder_interpolation_alphas=_parse_float_tuple(args.decoder_interpolation_alphas, (0.10, 0.25)),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
