"""Phase 3A: train the JEPA planner to produce decoder-compatible latents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dataset import write_examples_csv
from .experiment import _build_model, _candidate_surface_metrics, _canonical_pool, _write_predictions
from .features import MOLECULE_LATENT_VERSION, matrix_from_examples
from .image_context import attach_rendered_image_context
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


def run_phase3_decoder_compatible_planner(
    oracle_decoder_dir: str | Path,
    output_dir: str | Path = "outputs/runs/phase3_decoder_compatible_planner",
    dataset_csv: str | Path | None = None,
    train_csv: str | Path | None = None,
    eval_csv: str | Path | None = None,
    feature_dim: int = 256,
    latent_dim: int | None = None,
    top_k: int = 8,
    ridge: float = 1e-3,
    train_fraction: float = 0.8,
    seed: int = 7,
    render_image_context: bool = True,
    backend: str = "torch_denoiser",
    torch_hidden_dim: int = 1024,
    torch_epochs: int = 35,
    torch_batch_size: int = 128,
    torch_lr: float = 1e-3,
    torch_weight_decay: float = 1e-4,
    torch_diffusion_steps: int = 16,
    torch_train_noise: float = 0.35,
    torch_direct_loss_weight: float = 1.0,
    torch_delta_loss_weight: float = 0.2,
    torch_cosine_loss_weight: float = 2.0,
    torch_positive_loss_weight: float = 12.0,
    torch_contrastive_loss_weight: float = 0.75,
    torch_contrastive_temperature: float = 0.04,
    torch_hard_negative_loss_weight: float = 0.0,
    torch_hard_negative_margin: float = 0.10,
    torch_norm_loss_weight: float = 2.0,
    torch_decoder_compat_loss_weight: float = 4.0,
    torch_decoder_compat_cosine_margin: float = 0.78,
    torch_normalize_predictions: bool = True,
    torch_device: str = "auto",
    decoder_device: str = "auto",
) -> dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    oracle_model_dir = _resolve_oracle_model_dir(oracle_decoder_dir)
    decoder = OracleLatentSmilesDiffusion.load(oracle_model_dir, device=decoder_device)
    decoder_config = json.loads((oracle_model_dir / "config.json").read_text(encoding="utf-8"))
    decoder_dim = int(decoder.config.condition_dim)
    latent_dim = int(latent_dim or decoder_dim)
    if latent_dim != decoder_dim:
        raise ValueError(f"Planner latent_dim={latent_dim} must match oracle decoder condition_dim={decoder_dim}.")

    train_examples, eval_examples = _load_splits(dataset_csv, train_csv, eval_csv, train_fraction=train_fraction, seed=seed)
    if render_image_context:
        train_examples, train_image_meta = attach_rendered_image_context(train_examples, output_dir / "rendered_context" / "train")
        eval_examples, eval_image_meta = attach_rendered_image_context(eval_examples, output_dir / "rendered_context" / "eval")
    else:
        train_image_meta = {"rdkit_available": None, "rendered_images": 0, "masked_images": 0, "skipped_images": len(train_examples)}
        eval_image_meta = {"rdkit_available": None, "rendered_images": 0, "masked_images": 0, "skipped_images": len(eval_examples)}

    train_conditions, train_targets, train_sources = matrix_from_examples(train_examples, feature_dim=feature_dim, latent_dim=latent_dim)
    eval_conditions, eval_targets, eval_sources = matrix_from_examples(eval_examples, feature_dim=feature_dim, latent_dim=latent_dim)
    planner = _build_model(
        backend=backend,
        feature_dim=feature_dim,
        latent_dim=latent_dim,
        ridge=ridge,
        torch_hidden_dim=torch_hidden_dim,
        torch_epochs=torch_epochs,
        torch_batch_size=torch_batch_size,
        torch_lr=torch_lr,
        torch_weight_decay=torch_weight_decay,
        torch_diffusion_steps=torch_diffusion_steps,
        torch_train_noise=torch_train_noise,
        torch_direct_loss_weight=torch_direct_loss_weight,
        torch_delta_loss_weight=torch_delta_loss_weight,
        torch_cosine_loss_weight=torch_cosine_loss_weight,
        torch_positive_loss_weight=torch_positive_loss_weight,
        torch_contrastive_loss_weight=torch_contrastive_loss_weight,
        torch_contrastive_temperature=torch_contrastive_temperature,
        torch_hard_negative_loss_weight=torch_hard_negative_loss_weight,
        torch_hard_negative_margin=torch_hard_negative_margin,
        torch_norm_loss_weight=torch_norm_loss_weight,
        torch_decoder_compat_loss_weight=torch_decoder_compat_loss_weight,
        torch_decoder_compat_cosine_margin=torch_decoder_compat_cosine_margin,
        torch_normalize_predictions=torch_normalize_predictions,
        torch_device=torch_device,
        seed=seed,
    ).fit(train_conditions, train_targets, train_sources)
    train_pred_latents = planner.predict(train_conditions, train_sources)
    eval_pred_latents = planner.predict(eval_conditions, eval_sources)

    decoded = _retag_candidates(decoder.decode(eval_pred_latents, top_k=top_k), origin="phase3_decoder_compatible_planner")
    scores_by_task = [score_candidates(example, candidates) for example, candidates in zip(eval_examples, decoded)]
    metrics = summarize_scores(scores_by_task)
    metrics.update(
        {
            "train_tasks": float(len(train_examples)),
            "eval_tasks": float(len(eval_examples)),
            "planner_train_latent_mse": float(np.mean((train_pred_latents - train_targets) ** 2)),
            "planner_train_latent_cosine": _mean_cosine(train_pred_latents, train_targets),
            "planner_train_cosine_ge_margin": _fraction_cosine_at_least(train_pred_latents, train_targets, torch_decoder_compat_cosine_margin),
            "planner_latent_mse": float(np.mean((eval_pred_latents - eval_targets) ** 2)),
            "planner_latent_cosine": _mean_cosine(eval_pred_latents, eval_targets),
            "planner_cosine_ge_margin": _fraction_cosine_at_least(eval_pred_latents, eval_targets, torch_decoder_compat_cosine_margin),
        }
    )
    metrics.update(_latent_norm_metrics("planner_train", train_pred_latents, train_targets))
    metrics.update(_latent_norm_metrics("planner_eval", eval_pred_latents, eval_targets))
    planner_train_pool = _canonical_pool(example.target_smiles for example in train_examples)
    metrics.update(_candidate_surface_metrics(scores_by_task, planner_train_pool, eval_examples))
    decoder_train_pool = _load_decoder_train_pool(oracle_model_dir.parent)
    if decoder_train_pool:
        metrics.update(_decoder_pool_metrics(scores_by_task, decoder_train_pool))

    run_config = {
        "phase": "phase3a_decoder_compatible_planner",
        "research_question": "Can a JEPA planner be trained to output latents that stay inside the Phase 1 decoder-readable manifold?",
        "oracle_decoder_dir": str(oracle_model_dir),
        "oracle_decoder_config": decoder_config,
        "dataset_csv": str(dataset_csv) if dataset_csv else None,
        "train_csv": str(train_csv) if train_csv else None,
        "eval_csv": str(eval_csv) if eval_csv else None,
        "feature_dim": feature_dim,
        "latent_dim": latent_dim,
        "molecule_latent_version": MOLECULE_LATENT_VERSION,
        "top_k": top_k,
        "ridge": ridge,
        "train_fraction": train_fraction,
        "seed": seed,
        "render_image_context": render_image_context,
        "backend": backend,
        "torch_hidden_dim": torch_hidden_dim if backend == "torch_denoiser" else None,
        "torch_epochs": torch_epochs if backend == "torch_denoiser" else None,
        "torch_batch_size": torch_batch_size if backend == "torch_denoiser" else None,
        "torch_lr": torch_lr if backend == "torch_denoiser" else None,
        "torch_weight_decay": torch_weight_decay if backend == "torch_denoiser" else None,
        "torch_diffusion_steps": torch_diffusion_steps if backend == "torch_denoiser" else None,
        "torch_train_noise": torch_train_noise if backend == "torch_denoiser" else None,
        "torch_direct_loss_weight": torch_direct_loss_weight if backend == "torch_denoiser" else None,
        "torch_delta_loss_weight": torch_delta_loss_weight if backend == "torch_denoiser" else None,
        "torch_cosine_loss_weight": torch_cosine_loss_weight if backend == "torch_denoiser" else None,
        "torch_positive_loss_weight": torch_positive_loss_weight if backend == "torch_denoiser" else None,
        "torch_contrastive_loss_weight": torch_contrastive_loss_weight if backend == "torch_denoiser" else None,
        "torch_contrastive_temperature": torch_contrastive_temperature if backend == "torch_denoiser" else None,
        "torch_hard_negative_loss_weight": torch_hard_negative_loss_weight if backend == "torch_denoiser" else None,
        "torch_hard_negative_margin": torch_hard_negative_margin if backend == "torch_denoiser" else None,
        "torch_norm_loss_weight": torch_norm_loss_weight if backend == "torch_denoiser" else None,
        "torch_decoder_compat_loss_weight": torch_decoder_compat_loss_weight if backend == "torch_denoiser" else None,
        "torch_decoder_compat_cosine_margin": torch_decoder_compat_cosine_margin if backend == "torch_denoiser" else None,
        "torch_normalize_predictions": torch_normalize_predictions if backend == "torch_denoiser" else None,
        "torch_device": getattr(planner, "device_name", torch_device) if backend == "torch_denoiser" else None,
        "decoder_device": getattr(decoder, "device_name", decoder_device),
        "train_image_context": train_image_meta,
        "eval_image_context": eval_image_meta,
        "planner_history": planner.history,
        "decoder": {
            "mode": "frozen_phase1_oracle_latent_decoder",
            "ranking": "decoder_likelihood_plus_rdkit_validity_no_target_oracle",
            "can_leave_training_pool": True,
        },
    }

    planner.save(output_dir / "planner")
    np.save(output_dir / "planner_train_latents.npy", train_pred_latents.astype(np.float32))
    np.save(output_dir / "planner_eval_latents.npy", eval_pred_latents.astype(np.float32))
    np.save(output_dir / "train_target_latents.npy", train_targets.astype(np.float32))
    np.save(output_dir / "eval_target_latents.npy", eval_targets.astype(np.float32))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_examples_csv(output_dir / "train_examples.csv", train_examples)
    write_examples_csv(output_dir / "eval_examples.csv", eval_examples)
    predictions_path = output_dir / "predictions.csv"
    _write_predictions(predictions_path, eval_examples, scores_by_task, train_pool=planner_train_pool)
    _append_decoder_pool_membership(predictions_path, decoder_train_pool)
    summarize_predictions_csv(predictions_path, out_dir=output_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3A decoder-compatible JEPA planner with a frozen Phase 1 decoder.")
    parser.add_argument("--oracle-decoder-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/runs/phase3_decoder_compatible_planner")
    parser.add_argument("--dataset-csv", default=None)
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--eval-csv", default=None)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--render-image-context", action="store_true")
    parser.add_argument("--backend", choices=["ridge", "torch_denoiser"], default="torch_denoiser")
    parser.add_argument("--torch-hidden-dim", type=int, default=1024)
    parser.add_argument("--torch-epochs", type=int, default=35)
    parser.add_argument("--torch-batch-size", type=int, default=128)
    parser.add_argument("--torch-lr", type=float, default=1e-3)
    parser.add_argument("--torch-weight-decay", type=float, default=1e-4)
    parser.add_argument("--torch-diffusion-steps", type=int, default=16)
    parser.add_argument("--torch-train-noise", type=float, default=0.35)
    parser.add_argument("--torch-direct-loss-weight", type=float, default=1.0)
    parser.add_argument("--torch-delta-loss-weight", type=float, default=0.2)
    parser.add_argument("--torch-cosine-loss-weight", type=float, default=2.0)
    parser.add_argument("--torch-positive-loss-weight", type=float, default=12.0)
    parser.add_argument("--torch-contrastive-loss-weight", type=float, default=0.75)
    parser.add_argument("--torch-contrastive-temperature", type=float, default=0.04)
    parser.add_argument("--torch-hard-negative-loss-weight", type=float, default=0.0)
    parser.add_argument("--torch-hard-negative-margin", type=float, default=0.10)
    parser.add_argument("--torch-norm-loss-weight", type=float, default=2.0)
    parser.add_argument("--torch-decoder-compat-loss-weight", type=float, default=4.0)
    parser.add_argument("--torch-decoder-compat-cosine-margin", type=float, default=0.78)
    parser.add_argument("--torch-normalize-predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-device", default="auto")
    parser.add_argument("--decoder-device", default="auto")
    args = parser.parse_args()
    metrics = run_phase3_decoder_compatible_planner(
        oracle_decoder_dir=args.oracle_decoder_dir,
        output_dir=args.output_dir,
        dataset_csv=args.dataset_csv,
        train_csv=args.train_csv,
        eval_csv=args.eval_csv,
        feature_dim=args.feature_dim,
        latent_dim=args.latent_dim,
        top_k=args.top_k,
        ridge=args.ridge,
        train_fraction=args.train_fraction,
        seed=args.seed,
        render_image_context=args.render_image_context,
        backend=args.backend,
        torch_hidden_dim=args.torch_hidden_dim,
        torch_epochs=args.torch_epochs,
        torch_batch_size=args.torch_batch_size,
        torch_lr=args.torch_lr,
        torch_weight_decay=args.torch_weight_decay,
        torch_diffusion_steps=args.torch_diffusion_steps,
        torch_train_noise=args.torch_train_noise,
        torch_direct_loss_weight=args.torch_direct_loss_weight,
        torch_delta_loss_weight=args.torch_delta_loss_weight,
        torch_cosine_loss_weight=args.torch_cosine_loss_weight,
        torch_positive_loss_weight=args.torch_positive_loss_weight,
        torch_contrastive_loss_weight=args.torch_contrastive_loss_weight,
        torch_contrastive_temperature=args.torch_contrastive_temperature,
        torch_hard_negative_loss_weight=args.torch_hard_negative_loss_weight,
        torch_hard_negative_margin=args.torch_hard_negative_margin,
        torch_norm_loss_weight=args.torch_norm_loss_weight,
        torch_decoder_compat_loss_weight=args.torch_decoder_compat_loss_weight,
        torch_decoder_compat_cosine_margin=args.torch_decoder_compat_cosine_margin,
        torch_normalize_predictions=args.torch_normalize_predictions,
        torch_device=args.torch_device,
        decoder_device=args.decoder_device,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _fraction_cosine_at_least(left: np.ndarray, right: np.ndarray, margin: float) -> float:
    left_norm = left / np.clip(np.linalg.norm(left, axis=1, keepdims=True), 1e-8, None)
    right_norm = right / np.clip(np.linalg.norm(right, axis=1, keepdims=True), 1e-8, None)
    cosine = np.sum(left_norm * right_norm, axis=1)
    return float(np.mean(cosine >= float(margin)))


def _latent_norm_metrics(prefix: str, pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred_norm = np.linalg.norm(pred, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    return {
        f"{prefix}_pred_norm_mean": float(np.mean(pred_norm)),
        f"{prefix}_pred_norm_std": float(np.std(pred_norm)),
        f"{prefix}_target_norm_mean": float(np.mean(target_norm)),
        f"{prefix}_norm_mae": float(np.mean(np.abs(pred_norm - target_norm))),
    }


if __name__ == "__main__":
    main()
