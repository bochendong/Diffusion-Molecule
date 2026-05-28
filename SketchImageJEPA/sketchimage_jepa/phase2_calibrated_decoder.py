"""Phase 2C: calibrate JEPA planner latents before decoder sampling."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
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


@dataclass
class LatentCalibrationConfig:
    mode: str = "residual_ridge"
    ridge: float = 1e-2
    blend: float = 1.0
    normalize: bool = True


class LatentCalibrator:
    def __init__(self, config: LatentCalibrationConfig | None = None):
        self.config = config or LatentCalibrationConfig()
        self.weights_: np.ndarray | None = None
        self.offset_: np.ndarray | None = None
        self.history: list[dict[str, float | str]] = []

    def fit(self, planner_latents: np.ndarray, target_latents: np.ndarray) -> "LatentCalibrator":
        planner_latents = np.asarray(planner_latents, dtype=np.float32)
        target_latents = np.asarray(target_latents, dtype=np.float32)
        if planner_latents.shape != target_latents.shape:
            raise ValueError("planner_latents and target_latents must have the same shape.")
        mode = self.config.mode
        if mode == "none":
            self.history = [_calibration_record("none", planner_latents, target_latents, planner_latents)]
            return self
        if mode == "mean_shift":
            self.offset_ = np.mean(target_latents - planner_latents, axis=0, keepdims=True).astype(np.float32)
            calibrated = self._finish(planner_latents, planner_latents + self.offset_)
            self.history = [_calibration_record("mean_shift", planner_latents, target_latents, calibrated)]
            return self
        if mode == "ridge":
            self.weights_ = _ridge_solve(_design_matrix(planner_latents), target_latents, self.config.ridge)
            calibrated = self._finish(planner_latents, _design_matrix(planner_latents) @ self.weights_)
            self.history = [_calibration_record("ridge", planner_latents, target_latents, calibrated)]
            return self
        if mode == "residual_ridge":
            residual = target_latents - planner_latents
            self.weights_ = _ridge_solve(_design_matrix(planner_latents), residual, self.config.ridge)
            calibrated = self._finish(planner_latents, planner_latents + _design_matrix(planner_latents) @ self.weights_)
            self.history = [_calibration_record("residual_ridge", planner_latents, target_latents, calibrated)]
            return self
        raise ValueError(f"Unknown calibration mode: {mode}")

    def predict(self, planner_latents: np.ndarray) -> np.ndarray:
        planner_latents = np.asarray(planner_latents, dtype=np.float32)
        mode = self.config.mode
        if mode == "none":
            raw = planner_latents
        elif mode == "mean_shift":
            if self.offset_ is None:
                raise RuntimeError("LatentCalibrator must be fit before predict().")
            raw = planner_latents + self.offset_
        elif mode == "ridge":
            if self.weights_ is None:
                raise RuntimeError("LatentCalibrator must be fit before predict().")
            raw = _design_matrix(planner_latents) @ self.weights_
        elif mode == "residual_ridge":
            if self.weights_ is None:
                raise RuntimeError("LatentCalibrator must be fit before predict().")
            raw = planner_latents + _design_matrix(planner_latents) @ self.weights_
        else:
            raise ValueError(f"Unknown calibration mode: {mode}")
        return self._finish(planner_latents, raw)

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(asdict(self.config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out_dir / "metadata.json").write_text(json.dumps({"history": self.history}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if self.weights_ is not None:
            np.save(out_dir / "weights.npy", self.weights_.astype(np.float32))
        if self.offset_ is not None:
            np.save(out_dir / "offset.npy", self.offset_.astype(np.float32))

    def _finish(self, planner_latents: np.ndarray, raw: np.ndarray) -> np.ndarray:
        blend = min(1.0, max(0.0, float(self.config.blend)))
        out = (1.0 - blend) * planner_latents + blend * np.asarray(raw, dtype=np.float32)
        if self.config.normalize:
            out = _normalize_rows(out)
        return out.astype(np.float32)


def run_phase2_calibrated_decoder(
    decoder_dir: str | Path,
    output_dir: str | Path = "outputs/runs/phase2_calibrated_decoder",
    decoder_pool_dir: str | Path | None = None,
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
    torch_epochs: int = 25,
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
    torch_device: str = "auto",
    decoder_device: str = "auto",
    calibration_mode: str = "residual_ridge",
    calibration_ridge: float = 1e-2,
    calibration_blend: float = 1.0,
    calibration_normalize: bool = True,
) -> dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decoder_model_dir = _resolve_oracle_model_dir(decoder_dir)
    decoder = OracleLatentSmilesDiffusion.load(decoder_model_dir, device=decoder_device)
    decoder_config = json.loads((decoder_model_dir / "config.json").read_text(encoding="utf-8"))
    decoder_dim = int(decoder.config.condition_dim)
    latent_dim = int(latent_dim or decoder_dim)
    if latent_dim != decoder_dim:
        raise ValueError(f"Planner latent_dim={latent_dim} must match decoder condition_dim={decoder_dim}.")

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
        torch_device=torch_device,
        seed=seed,
    ).fit(train_conditions, train_targets, train_sources)
    train_pred_latents = planner.predict(train_conditions, train_sources)
    eval_pred_latents = planner.predict(eval_conditions, eval_sources)

    calibrator = LatentCalibrator(
        LatentCalibrationConfig(
            mode=calibration_mode,
            ridge=calibration_ridge,
            blend=calibration_blend,
            normalize=calibration_normalize,
        )
    ).fit(train_pred_latents, train_targets)
    train_calibrated_latents = calibrator.predict(train_pred_latents)
    eval_calibrated_latents = calibrator.predict(eval_pred_latents)

    decoded = _retag_candidates(decoder.decode(eval_calibrated_latents, top_k=top_k), origin="phase2_jepa_calibrated_decoder")
    scores_by_task = [score_candidates(example, candidates) for example, candidates in zip(eval_examples, decoded)]
    metrics = summarize_scores(scores_by_task)
    metrics.update(
        {
            "train_tasks": float(len(train_examples)),
            "eval_tasks": float(len(eval_examples)),
            "planner_train_latent_mse": float(np.mean((train_pred_latents - train_targets) ** 2)),
            "planner_train_latent_cosine": _mean_cosine(train_pred_latents, train_targets),
            "planner_latent_mse": float(np.mean((eval_pred_latents - eval_targets) ** 2)),
            "planner_latent_cosine": _mean_cosine(eval_pred_latents, eval_targets),
            "calibrated_train_latent_mse": float(np.mean((train_calibrated_latents - train_targets) ** 2)),
            "calibrated_train_latent_cosine": _mean_cosine(train_calibrated_latents, train_targets),
            "calibrated_latent_mse": float(np.mean((eval_calibrated_latents - eval_targets) ** 2)),
            "calibrated_latent_cosine": _mean_cosine(eval_calibrated_latents, eval_targets),
        }
    )
    planner_train_pool = _canonical_pool(example.target_smiles for example in train_examples)
    metrics.update(_candidate_surface_metrics(scores_by_task, planner_train_pool, eval_examples))
    pool_root = Path(decoder_pool_dir) if decoder_pool_dir else decoder_model_dir.parent
    decoder_train_pool = _load_decoder_train_pool(pool_root)
    if decoder_train_pool:
        metrics.update(_decoder_pool_metrics(scores_by_task, decoder_train_pool))

    run_config = {
        "phase": "phase2c_latent_calibrated_decoder",
        "research_question": "Can a latent calibration adapter map JEPA-predicted latents back onto the decoder's oracle latent manifold?",
        "decoder_dir": str(decoder_model_dir),
        "decoder_pool_dir": str(pool_root),
        "decoder_config": decoder_config,
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
        "torch_device": getattr(planner, "device_name", torch_device) if backend == "torch_denoiser" else None,
        "decoder_device": getattr(decoder, "device_name", decoder_device),
        "calibration": asdict(calibrator.config),
        "train_image_context": train_image_meta,
        "eval_image_context": eval_image_meta,
        "planner_history": planner.history,
        "calibration_history": calibrator.history,
        "decoder": {
            "mode": "latent_calibrated_decoder",
            "ranking": "decoder_likelihood_plus_rdkit_validity_no_target_oracle",
            "can_leave_training_pool": True,
        },
    }

    planner.save(output_dir / "planner")
    calibrator.save(output_dir / "calibrator")
    np.save(output_dir / "planner_train_latents.npy", train_pred_latents.astype(np.float32))
    np.save(output_dir / "planner_eval_latents.npy", eval_pred_latents.astype(np.float32))
    np.save(output_dir / "calibrated_train_latents.npy", train_calibrated_latents.astype(np.float32))
    np.save(output_dir / "calibrated_eval_latents.npy", eval_calibrated_latents.astype(np.float32))
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
    parser = argparse.ArgumentParser(description="Run Phase 2C latent calibration before decoder sampling.")
    parser.add_argument("--decoder-dir", required=True)
    parser.add_argument("--decoder-pool-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/runs/phase2_calibrated_decoder")
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
    parser.add_argument("--torch-epochs", type=int, default=25)
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
    parser.add_argument("--torch-device", default="auto")
    parser.add_argument("--decoder-device", default="auto")
    parser.add_argument("--calibration-mode", choices=["none", "mean_shift", "ridge", "residual_ridge"], default="residual_ridge")
    parser.add_argument("--calibration-ridge", type=float, default=1e-2)
    parser.add_argument("--calibration-blend", type=float, default=1.0)
    parser.add_argument("--calibration-normalize", action="store_true")
    parser.add_argument("--no-calibration-normalize", dest="calibration_normalize", action="store_false")
    parser.set_defaults(calibration_normalize=True)
    args = parser.parse_args()
    metrics = run_phase2_calibrated_decoder(
        decoder_dir=args.decoder_dir,
        output_dir=args.output_dir,
        decoder_pool_dir=args.decoder_pool_dir,
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
        torch_device=args.torch_device,
        decoder_device=args.decoder_device,
        calibration_mode=args.calibration_mode,
        calibration_ridge=args.calibration_ridge,
        calibration_blend=args.calibration_blend,
        calibration_normalize=args.calibration_normalize,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _design_matrix(latents: np.ndarray) -> np.ndarray:
    latents = np.asarray(latents, dtype=np.float32)
    return np.concatenate([latents, np.ones((len(latents), 1), dtype=np.float32)], axis=1)


def _ridge_solve(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    ridge = max(0.0, float(ridge))
    rows, cols = x64.shape
    if cols <= rows:
        reg = ridge * np.eye(cols, dtype=np.float64)
        reg[-1, -1] = 0.0
        return np.linalg.solve(x64.T @ x64 + reg, x64.T @ y64).astype(np.float32)
    reg = ridge * np.eye(rows, dtype=np.float64)
    alpha = np.linalg.solve(x64 @ x64.T + reg, y64)
    return (x64.T @ alpha).astype(np.float32)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-8, None)


def _calibration_record(mode: str, planner: np.ndarray, target: np.ndarray, calibrated: np.ndarray) -> dict[str, float | str]:
    return {
        "mode": mode,
        "train_planner_mse": float(np.mean((planner - target) ** 2)),
        "train_planner_cosine": _mean_cosine(planner, target),
        "train_calibrated_mse": float(np.mean((calibrated - target) ** 2)),
        "train_calibrated_cosine": _mean_cosine(calibrated, target),
    }


if __name__ == "__main__":
    main()
