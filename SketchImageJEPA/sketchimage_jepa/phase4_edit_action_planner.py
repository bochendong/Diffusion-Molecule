"""Phase 4A: source-conditioned edit/action latent planner."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .chem import canonicalize_smiles
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
from .schema import BenchmarkExample, Candidate, TaskType
from .verifier import score_candidates, summarize_scores


SOURCE_CONDITIONED_TASKS = {TaskType.EDIT, TaskType.INPAINT, TaskType.FRAGMENT_GROW}


def run_phase4_edit_action_planner(
    oracle_decoder_dir: str | Path,
    output_dir: str | Path = "outputs/runs/phase4_edit_action_planner",
    dataset_csv: str | Path | None = None,
    train_csv: str | Path | None = None,
    eval_csv: str | Path | None = None,
    feature_dim: int = 256,
    latent_dim: int | None = None,
    top_k: int = 8,
    samples_per_alpha: int = 2,
    action_alphas: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0, 1.25),
    alpha_score_penalty: float = 0.02,
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
    torch_delta_loss_weight: float = 0.0,
    torch_cosine_loss_weight: float = 2.0,
    torch_positive_loss_weight: float = 12.0,
    torch_contrastive_loss_weight: float = 0.75,
    torch_contrastive_temperature: float = 0.04,
    torch_hard_negative_loss_weight: float = 0.0,
    torch_hard_negative_margin: float = 0.10,
    torch_device: str = "auto",
    decoder_device: str = "auto",
    oracle_action_control: bool = True,
    action_target_mode: str = "raw_delta",
    action_step_mode: str = "implicit",
    action_step_ridge: float = 1e-2,
    action_step_clip_quantile: float = 0.98,
    action_correction_mode: str = "none",
    action_neighbor_key: str = "condition_source",
    action_neighbor_count: int = 16,
    action_neighbor_temperature: float = 0.07,
    action_neighbor_blend: float = 0.5,
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
    if action_correction_mode != "none" and action_target_mode != "unit_direction":
        raise ValueError("Action correction requires action_target_mode='unit_direction'.")

    raw_train_examples, raw_eval_examples = _load_splits(dataset_csv, train_csv, eval_csv, train_fraction=train_fraction, seed=seed)
    train_examples, train_excluded = _source_conditioned_examples(raw_train_examples)
    eval_examples, eval_excluded = _source_conditioned_examples(raw_eval_examples)
    if not train_examples:
        raise ValueError("Phase 4A needs at least one source-conditioned train example.")
    if not eval_examples:
        raise ValueError("Phase 4A needs at least one source-conditioned eval example.")

    if render_image_context:
        train_examples, train_image_meta = attach_rendered_image_context(train_examples, output_dir / "rendered_context" / "train")
        eval_examples, eval_image_meta = attach_rendered_image_context(eval_examples, output_dir / "rendered_context" / "eval")
    else:
        train_image_meta = {"rdkit_available": None, "rendered_images": 0, "masked_images": 0, "skipped_images": len(train_examples)}
        eval_image_meta = {"rdkit_available": None, "rendered_images": 0, "masked_images": 0, "skipped_images": len(eval_examples)}

    train_base_conditions, train_targets, train_sources = matrix_from_examples(train_examples, feature_dim=feature_dim, latent_dim=latent_dim)
    eval_base_conditions, eval_targets, eval_sources = matrix_from_examples(eval_examples, feature_dim=feature_dim, latent_dim=latent_dim)
    train_conditions = _action_conditions(train_base_conditions, train_sources)
    eval_conditions = _action_conditions(eval_base_conditions, eval_sources)
    train_actions = train_targets - train_sources
    eval_actions = eval_targets - eval_sources
    train_action_targets = _action_training_targets(train_actions, action_target_mode)
    train_action_steps = _row_norms(train_actions)
    eval_action_steps = _row_norms(eval_actions)
    step_calibrator = _fit_action_step_calibrator(
        action_step_mode=action_step_mode,
        train_conditions=train_conditions,
        train_steps=train_action_steps,
        ridge=action_step_ridge,
        clip_quantile=action_step_clip_quantile,
    )
    zero_train_sources = np.zeros_like(train_sources, dtype=np.float32)
    zero_eval_sources = np.zeros_like(eval_sources, dtype=np.float32)

    planner = _build_model(
        backend=backend,
        feature_dim=int(train_conditions.shape[1]),
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
    ).fit(train_conditions, train_action_targets, zero_train_sources)

    train_pred_action_outputs = planner.predict(train_conditions, zero_train_sources)
    eval_pred_action_outputs = planner.predict(eval_conditions, zero_eval_sources)
    train_uncorrected_actions, train_uncorrected_steps = _planned_action_vectors(
        action_outputs=train_pred_action_outputs,
        conditions=train_conditions,
        action_target_mode=action_target_mode,
        step_calibrator=step_calibrator,
    )
    eval_uncorrected_actions, eval_uncorrected_steps = _planned_action_vectors(
        action_outputs=eval_pred_action_outputs,
        conditions=eval_conditions,
        action_target_mode=action_target_mode,
        step_calibrator=step_calibrator,
    )
    train_neighbor_keys = _action_neighbor_keys(action_neighbor_key, train_base_conditions, train_sources, train_conditions)
    eval_neighbor_keys = _action_neighbor_keys(action_neighbor_key, eval_base_conditions, eval_sources, eval_conditions)
    train_corrected_outputs, train_correction_metrics = _correct_action_outputs(
        action_outputs=train_pred_action_outputs,
        query_keys=train_neighbor_keys,
        train_keys=train_neighbor_keys,
        train_target_actions=train_actions,
        train_pred_action_outputs=train_pred_action_outputs,
        correction_mode=action_correction_mode,
        neighbor_count=action_neighbor_count,
        temperature=action_neighbor_temperature,
        blend=action_neighbor_blend,
        exclude_self=True,
        prefix="action_train_correction",
    )
    eval_corrected_outputs, eval_correction_metrics = _correct_action_outputs(
        action_outputs=eval_pred_action_outputs,
        query_keys=eval_neighbor_keys,
        train_keys=train_neighbor_keys,
        train_target_actions=train_actions,
        train_pred_action_outputs=train_pred_action_outputs,
        correction_mode=action_correction_mode,
        neighbor_count=action_neighbor_count,
        temperature=action_neighbor_temperature,
        blend=action_neighbor_blend,
        exclude_self=False,
        prefix="action_eval_correction",
    )
    train_pred_actions, train_pred_steps = _planned_action_vectors(
        action_outputs=train_corrected_outputs,
        conditions=train_conditions,
        action_target_mode=action_target_mode,
        step_calibrator=step_calibrator,
    )
    eval_pred_actions, eval_pred_steps = _planned_action_vectors(
        action_outputs=eval_corrected_outputs,
        conditions=eval_conditions,
        action_target_mode=action_target_mode,
        step_calibrator=step_calibrator,
    )
    train_composed = _normalize_rows(train_sources + train_pred_actions)
    eval_composed = _normalize_rows(eval_sources + eval_pred_actions)

    alpha_summary, decoded = _decode_action_beam(
        decoder=decoder,
        examples=eval_examples,
        source_latents=eval_sources,
        action_latents=eval_pred_actions,
        target_latents=eval_targets,
        action_alphas=action_alphas,
        top_k=top_k,
        samples_per_alpha=samples_per_alpha,
        alpha_score_penalty=alpha_score_penalty,
    )
    scores_by_task = [score_candidates(example, candidates) for example, candidates in zip(eval_examples, decoded)]
    metrics = summarize_scores(scores_by_task)
    metrics.update(
        {
            "train_tasks": float(len(train_examples)),
            "eval_tasks": float(len(eval_examples)),
            "excluded_train_tasks": float(train_excluded),
            "excluded_eval_tasks": float(eval_excluded),
            "uncorrected_action_latent_mse": float(np.mean((eval_uncorrected_actions - eval_actions) ** 2)),
            "uncorrected_action_train_latent_mse": float(np.mean((train_uncorrected_actions - train_actions) ** 2)),
            "uncorrected_action_latent_cosine": _mean_cosine(eval_uncorrected_actions, eval_actions),
            "uncorrected_action_train_latent_cosine": _mean_cosine(train_uncorrected_actions, train_actions),
            "uncorrected_composed_latent_mse": float(np.mean((_normalize_rows(eval_sources + eval_uncorrected_actions) - eval_targets) ** 2)),
            "uncorrected_composed_train_latent_mse": float(np.mean((_normalize_rows(train_sources + train_uncorrected_actions) - train_targets) ** 2)),
            "uncorrected_composed_latent_cosine": _mean_cosine(_normalize_rows(eval_sources + eval_uncorrected_actions), eval_targets),
            "uncorrected_composed_train_latent_cosine": _mean_cosine(_normalize_rows(train_sources + train_uncorrected_actions), train_targets),
            "action_latent_mse": float(np.mean((eval_pred_actions - eval_actions) ** 2)),
            "action_train_latent_mse": float(np.mean((train_pred_actions - train_actions) ** 2)),
            "action_latent_cosine": _mean_cosine(eval_pred_actions, eval_actions),
            "action_train_latent_cosine": _mean_cosine(train_pred_actions, train_actions),
            "composed_latent_mse": float(np.mean((eval_composed - eval_targets) ** 2)),
            "composed_train_latent_mse": float(np.mean((train_composed - train_targets) ** 2)),
            "composed_latent_cosine": _mean_cosine(eval_composed, eval_targets),
            "composed_train_latent_cosine": _mean_cosine(train_composed, train_targets),
            "mean_candidate_count": float(np.mean([len(candidates) for candidates in decoded])) if decoded else 0.0,
        }
    )
    metrics.update(_action_norm_metrics("action_eval", eval_pred_actions, eval_actions))
    metrics.update(_action_norm_metrics("action_train", train_pred_actions, train_actions))
    metrics.update(_action_step_metrics("action_eval_step", eval_pred_steps, eval_action_steps))
    metrics.update(_action_step_metrics("action_train_step", train_pred_steps, train_action_steps))
    metrics.update(_action_step_metrics("uncorrected_action_eval_step", eval_uncorrected_steps, eval_action_steps))
    metrics.update(_action_step_metrics("uncorrected_action_train_step", train_uncorrected_steps, train_action_steps))
    metrics.update(train_correction_metrics)
    metrics.update(eval_correction_metrics)
    planner_train_pool = _canonical_pool(example.target_smiles for example in train_examples)
    metrics.update(_candidate_surface_metrics(scores_by_task, planner_train_pool, eval_examples))
    decoder_train_pool = _load_decoder_train_pool(oracle_model_dir.parent)
    if decoder_train_pool:
        metrics.update(_decoder_pool_metrics(scores_by_task, decoder_train_pool))

    if oracle_action_control:
        oracle_decoded = _retag_candidates(decoder.decode(eval_targets, top_k=top_k), origin="phase4_oracle_action_control")
        oracle_scores_by_task = [score_candidates(example, candidates) for example, candidates in zip(eval_examples, oracle_decoded)]
        oracle_metrics = summarize_scores(oracle_scores_by_task)
        metrics.update({f"oracle_action_{key}": value for key, value in oracle_metrics.items()})
        oracle_predictions_path = output_dir / "oracle_action_predictions.csv"
        _write_predictions(oracle_predictions_path, eval_examples, oracle_scores_by_task, train_pool=planner_train_pool)
        _append_decoder_pool_membership(oracle_predictions_path, decoder_train_pool)

    if action_correction_mode != "none":
        phase_name = "phase4c_retrieval_guided_source_conditioned_edit_action_planner"
    elif action_target_mode == "raw_delta" and action_step_mode == "implicit":
        phase_name = "phase4a_source_conditioned_edit_action_planner"
    else:
        phase_name = "phase4b_normalized_source_conditioned_edit_action_planner"
    if action_correction_mode != "none":
        latent_composition = "normalize(source_latent + alpha * predicted_step * corrected_action_direction)"
    elif action_target_mode == "raw_delta" and action_step_mode == "implicit":
        latent_composition = "normalize(source_latent + alpha * predicted_action_latent)"
    else:
        latent_composition = "normalize(source_latent + alpha * predicted_step * normalized_predicted_action_direction)"
    run_config = {
        "phase": phase_name,
        "research_question": "Can source-conditioned molecular editing be learned as an action latent instead of direct target latent prediction?",
        "oracle_decoder_dir": str(oracle_model_dir),
        "oracle_decoder_config": decoder_config,
        "dataset_csv": str(dataset_csv) if dataset_csv else None,
        "train_csv": str(train_csv) if train_csv else None,
        "eval_csv": str(eval_csv) if eval_csv else None,
        "raw_train_tasks": len(raw_train_examples),
        "raw_eval_tasks": len(raw_eval_examples),
        "excluded_train_tasks": train_excluded,
        "excluded_eval_tasks": eval_excluded,
        "included_task_types": sorted(task.value for task in SOURCE_CONDITIONED_TASKS),
        "feature_dim": feature_dim,
        "action_condition_dim": int(train_conditions.shape[1]),
        "latent_dim": latent_dim,
        "molecule_latent_version": MOLECULE_LATENT_VERSION,
        "top_k": top_k,
        "samples_per_alpha": samples_per_alpha,
        "action_alphas": list(action_alphas),
        "alpha_score_penalty": alpha_score_penalty,
        "action_target_mode": action_target_mode,
        "action_step_mode": action_step_mode,
        "action_step_ridge": action_step_ridge,
        "action_step_clip_quantile": action_step_clip_quantile,
        "action_step_calibrator": step_calibrator,
        "action_correction_mode": action_correction_mode,
        "action_neighbor_key": action_neighbor_key,
        "action_neighbor_count": action_neighbor_count,
        "action_neighbor_temperature": action_neighbor_temperature,
        "action_neighbor_blend": action_neighbor_blend,
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
        "train_image_context": train_image_meta,
        "eval_image_context": eval_image_meta,
        "planner_history": planner.history,
        "decoder": {
            "mode": "frozen_phase1_oracle_latent_decoder",
            "latent_composition": latent_composition,
            "ranking": "decoder_likelihood_plus_validity_minus_alpha_distance_penalty_no_target_oracle",
            "can_leave_training_pool": True,
        },
    }

    planner.save(output_dir / "planner")
    np.save(output_dir / "planner_train_actions.npy", train_pred_actions.astype(np.float32))
    np.save(output_dir / "planner_eval_actions.npy", eval_pred_actions.astype(np.float32))
    np.save(output_dir / "planner_train_uncorrected_actions.npy", train_uncorrected_actions.astype(np.float32))
    np.save(output_dir / "planner_eval_uncorrected_actions.npy", eval_uncorrected_actions.astype(np.float32))
    np.save(output_dir / "planner_train_corrected_action_outputs.npy", train_corrected_outputs.astype(np.float32))
    np.save(output_dir / "planner_eval_corrected_action_outputs.npy", eval_corrected_outputs.astype(np.float32))
    np.save(output_dir / "planner_train_action_outputs.npy", train_pred_action_outputs.astype(np.float32))
    np.save(output_dir / "planner_eval_action_outputs.npy", eval_pred_action_outputs.astype(np.float32))
    np.save(output_dir / "train_target_actions.npy", train_actions.astype(np.float32))
    np.save(output_dir / "eval_target_actions.npy", eval_actions.astype(np.float32))
    np.save(output_dir / "planner_eval_composed_latents.npy", eval_composed.astype(np.float32))
    np.save(output_dir / "eval_target_latents.npy", eval_targets.astype(np.float32))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_alpha_summary(output_dir / "alpha_summary.csv", alpha_summary)
    (output_dir / "alpha_summary.json").write_text(json.dumps(alpha_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_examples_csv(output_dir / "train_examples.csv", train_examples)
    write_examples_csv(output_dir / "eval_examples.csv", eval_examples)
    predictions_path = output_dir / "predictions.csv"
    _write_predictions(predictions_path, eval_examples, scores_by_task, train_pool=planner_train_pool)
    _append_decoder_pool_membership(predictions_path, decoder_train_pool)
    summarize_predictions_csv(predictions_path, out_dir=output_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4A source-conditioned edit/action latent planner.")
    parser.add_argument("--oracle-decoder-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/runs/phase4_edit_action_planner")
    parser.add_argument("--dataset-csv", default=None)
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--eval-csv", default=None)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--samples-per-alpha", type=int, default=2)
    parser.add_argument("--action-alphas", default="0.25,0.50,0.75,1.00,1.25")
    parser.add_argument("--alpha-score-penalty", type=float, default=0.02)
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
    parser.add_argument("--torch-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--torch-cosine-loss-weight", type=float, default=2.0)
    parser.add_argument("--torch-positive-loss-weight", type=float, default=12.0)
    parser.add_argument("--torch-contrastive-loss-weight", type=float, default=0.75)
    parser.add_argument("--torch-contrastive-temperature", type=float, default=0.04)
    parser.add_argument("--torch-hard-negative-loss-weight", type=float, default=0.0)
    parser.add_argument("--torch-hard-negative-margin", type=float, default=0.10)
    parser.add_argument("--torch-device", default="auto")
    parser.add_argument("--decoder-device", default="auto")
    parser.add_argument("--oracle-action-control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-target-mode", choices=["raw_delta", "unit_direction"], default="raw_delta")
    parser.add_argument("--action-step-mode", choices=["implicit", "target_norm_median", "condition_ridge"], default="implicit")
    parser.add_argument("--action-step-ridge", type=float, default=1e-2)
    parser.add_argument("--action-step-clip-quantile", type=float, default=0.98)
    parser.add_argument("--action-correction-mode", choices=["none", "neighbor_direction", "neighbor_residual"], default="none")
    parser.add_argument("--action-neighbor-key", choices=["condition", "source", "condition_source"], default="condition_source")
    parser.add_argument("--action-neighbor-count", type=int, default=16)
    parser.add_argument("--action-neighbor-temperature", type=float, default=0.07)
    parser.add_argument("--action-neighbor-blend", type=float, default=0.5)
    args = parser.parse_args()
    metrics = run_phase4_edit_action_planner(
        oracle_decoder_dir=args.oracle_decoder_dir,
        output_dir=args.output_dir,
        dataset_csv=args.dataset_csv,
        train_csv=args.train_csv,
        eval_csv=args.eval_csv,
        feature_dim=args.feature_dim,
        latent_dim=args.latent_dim,
        top_k=args.top_k,
        samples_per_alpha=args.samples_per_alpha,
        action_alphas=_parse_float_tuple(args.action_alphas, (0.25, 0.50, 0.75, 1.0, 1.25)),
        alpha_score_penalty=args.alpha_score_penalty,
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
        oracle_action_control=args.oracle_action_control,
        action_target_mode=args.action_target_mode,
        action_step_mode=args.action_step_mode,
        action_step_ridge=args.action_step_ridge,
        action_step_clip_quantile=args.action_step_clip_quantile,
        action_correction_mode=args.action_correction_mode,
        action_neighbor_key=args.action_neighbor_key,
        action_neighbor_count=args.action_neighbor_count,
        action_neighbor_temperature=args.action_neighbor_temperature,
        action_neighbor_blend=args.action_neighbor_blend,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _source_conditioned_examples(examples: list[BenchmarkExample]) -> tuple[list[BenchmarkExample], int]:
    kept = [example for example in examples if example.source_smiles and example.task_type in SOURCE_CONDITIONED_TASKS]
    return kept, len(examples) - len(kept)


def _action_conditions(base_conditions: np.ndarray, source_latents: np.ndarray) -> np.ndarray:
    return np.concatenate([base_conditions, source_latents], axis=1).astype(np.float32)


def _action_neighbor_keys(
    action_neighbor_key: str,
    base_conditions: np.ndarray,
    source_latents: np.ndarray,
    action_conditions: np.ndarray,
) -> np.ndarray:
    if action_neighbor_key == "condition":
        return np.asarray(base_conditions, dtype=np.float32)
    if action_neighbor_key == "source":
        return np.asarray(source_latents, dtype=np.float32)
    if action_neighbor_key == "condition_source":
        return np.asarray(action_conditions, dtype=np.float32)
    raise ValueError(f"Unsupported action_neighbor_key={action_neighbor_key!r}.")


def _action_training_targets(raw_actions: np.ndarray, action_target_mode: str) -> np.ndarray:
    if action_target_mode == "raw_delta":
        return np.asarray(raw_actions, dtype=np.float32)
    if action_target_mode == "unit_direction":
        return _normalize_rows(raw_actions)
    raise ValueError(f"Unsupported action_target_mode={action_target_mode!r}.")


def _planned_action_vectors(
    action_outputs: np.ndarray,
    conditions: np.ndarray,
    action_target_mode: str,
    step_calibrator: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    action_outputs = np.asarray(action_outputs, dtype=np.float32)
    mode = str(step_calibrator.get("mode", "implicit"))
    if action_target_mode == "raw_delta" and mode == "implicit":
        return action_outputs, _row_norms(action_outputs)

    directions = _normalize_rows(action_outputs)
    if mode == "implicit":
        steps = np.ones((action_outputs.shape[0],), dtype=np.float32)
    else:
        steps = _predict_action_steps(step_calibrator, conditions)
    return (directions * steps[:, None]).astype(np.float32), steps.astype(np.float32)


def _correct_action_outputs(
    action_outputs: np.ndarray,
    query_keys: np.ndarray,
    train_keys: np.ndarray,
    train_target_actions: np.ndarray,
    train_pred_action_outputs: np.ndarray,
    correction_mode: str,
    neighbor_count: int,
    temperature: float,
    blend: float,
    exclude_self: bool,
    prefix: str,
) -> tuple[np.ndarray, dict[str, float]]:
    action_outputs = np.asarray(action_outputs, dtype=np.float32)
    if correction_mode == "none":
        return action_outputs, {
            f"{prefix}_neighbor_similarity_mean": 0.0,
            f"{prefix}_direction_cosine_to_base": 1.0,
            f"{prefix}_blend": 0.0,
        }
    base_dirs = _normalize_rows(action_outputs)
    train_target_dirs = _normalize_rows(train_target_actions)
    train_pred_dirs = _normalize_rows(train_pred_action_outputs)
    top_indices, top_similarities, weights = _weighted_neighbors(
        query_keys=query_keys,
        train_keys=train_keys,
        neighbor_count=neighbor_count,
        temperature=temperature,
        exclude_self=exclude_self,
    )
    neighbor_target_dirs = train_target_dirs[top_indices]
    weighted_target_dirs = np.einsum("nk,nkd->nd", weights, neighbor_target_dirs).astype(np.float32)
    if correction_mode == "neighbor_direction":
        corrected_dirs = _normalize_rows((1.0 - float(blend)) * base_dirs + float(blend) * weighted_target_dirs)
    elif correction_mode == "neighbor_residual":
        neighbor_residuals = train_target_dirs[top_indices] - train_pred_dirs[top_indices]
        weighted_residuals = np.einsum("nk,nkd->nd", weights, neighbor_residuals).astype(np.float32)
        corrected_dirs = _normalize_rows(base_dirs + float(blend) * weighted_residuals)
    else:
        raise ValueError(f"Unsupported action_correction_mode={correction_mode!r}.")
    return corrected_dirs.astype(np.float32), {
        f"{prefix}_neighbor_similarity_mean": float(np.mean(top_similarities[:, 0])) if top_similarities.size else 0.0,
        f"{prefix}_neighbor_similarity_weighted_mean": float(np.mean(np.sum(weights * top_similarities, axis=1))) if top_similarities.size else 0.0,
        f"{prefix}_direction_cosine_to_base": _mean_cosine(corrected_dirs, base_dirs),
        f"{prefix}_blend": float(blend),
    }


def _weighted_neighbors(
    query_keys: np.ndarray,
    train_keys: np.ndarray,
    neighbor_count: int,
    temperature: float,
    exclude_self: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = _normalize_rows(query_keys)
    train = _normalize_rows(train_keys)
    similarities = query @ train.T
    if exclude_self and similarities.shape[0] == similarities.shape[1] and similarities.shape[0] > 1:
        np.fill_diagonal(similarities, -np.inf)
    count = max(1, min(int(neighbor_count), similarities.shape[1]))
    if count == similarities.shape[1]:
        top_indices = np.argsort(-similarities, axis=1)[:, :count]
    else:
        top_indices = np.argpartition(-similarities, kth=count - 1, axis=1)[:, :count]
        top_values = np.take_along_axis(similarities, top_indices, axis=1)
        order = np.argsort(-top_values, axis=1)
        top_indices = np.take_along_axis(top_indices, order, axis=1)
    top_similarities = np.take_along_axis(similarities, top_indices, axis=1)
    weights = _softmax(top_similarities / max(float(temperature), 1e-6), axis=1)
    return top_indices, top_similarities.astype(np.float32), weights.astype(np.float32)


def _softmax(values: np.ndarray, axis: int = 1) -> np.ndarray:
    finite_values = np.where(np.isfinite(values), values, -1e9)
    shifted = finite_values - np.max(finite_values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(np.sum(exp, axis=axis, keepdims=True), 1e-12, None)


def _fit_action_step_calibrator(
    action_step_mode: str,
    train_conditions: np.ndarray,
    train_steps: np.ndarray,
    ridge: float,
    clip_quantile: float,
) -> dict[str, object]:
    train_steps = np.asarray(train_steps, dtype=np.float32)
    low, high = _step_clip_bounds(train_steps, clip_quantile)
    if action_step_mode == "implicit":
        return {"mode": "implicit", "clip_low": low, "clip_high": high}
    if action_step_mode == "target_norm_median":
        return {
            "mode": "target_norm_median",
            "value": float(np.median(train_steps)),
            "clip_low": low,
            "clip_high": high,
        }
    if action_step_mode == "condition_ridge":
        x = np.asarray(train_conditions, dtype=np.float64)
        y = np.asarray(train_steps, dtype=np.float64)
        design = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
        xtx = design.T @ design
        reg = np.eye(xtx.shape[0], dtype=np.float64) * max(float(ridge), 0.0)
        reg[0, 0] = 0.0
        try:
            weights = np.linalg.solve(xtx + reg, design.T @ y)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(xtx + reg) @ design.T @ y
        train_pred = np.clip(design @ weights, low, high).astype(np.float32)
        return {
            "mode": "condition_ridge",
            "weights": weights.astype(np.float32).tolist(),
            "clip_low": low,
            "clip_high": high,
            "train_step_mae": float(np.mean(np.abs(train_pred - train_steps))),
            "train_step_pred_mean": float(np.mean(train_pred)),
        }
    raise ValueError(f"Unsupported action_step_mode={action_step_mode!r}.")


def _predict_action_steps(step_calibrator: dict[str, object], conditions: np.ndarray) -> np.ndarray:
    mode = str(step_calibrator.get("mode", "implicit"))
    low = float(step_calibrator.get("clip_low", 0.0))
    high = float(step_calibrator.get("clip_high", np.inf))
    if mode == "target_norm_median":
        steps = np.full((conditions.shape[0],), float(step_calibrator["value"]), dtype=np.float32)
    elif mode == "condition_ridge":
        weights = np.asarray(step_calibrator["weights"], dtype=np.float32)
        x = np.asarray(conditions, dtype=np.float32)
        design = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float32), x], axis=1)
        steps = design @ weights
    else:
        steps = np.ones((conditions.shape[0],), dtype=np.float32)
    return np.clip(steps, low, high).astype(np.float32)


def _step_clip_bounds(train_steps: np.ndarray, clip_quantile: float) -> tuple[float, float]:
    quantile = min(max(float(clip_quantile), 0.5), 1.0)
    low_quantile = 1.0 - quantile
    low = float(np.quantile(train_steps, low_quantile))
    high = float(np.quantile(train_steps, quantile))
    if high < low:
        high = low
    return low, high


def _decode_action_beam(
    decoder: OracleLatentSmilesDiffusion,
    examples: list[BenchmarkExample],
    source_latents: np.ndarray,
    action_latents: np.ndarray,
    target_latents: np.ndarray,
    action_alphas: tuple[float, ...],
    top_k: int,
    samples_per_alpha: int,
    alpha_score_penalty: float,
) -> tuple[list[dict[str, float | str]], list[list[Candidate]]]:
    per_alpha_scores: list[dict[str, float | str]] = []
    decoded_by_alpha: list[tuple[float, list[list[Candidate]]]] = []
    for alpha in action_alphas:
        composed = _normalize_rows(source_latents + float(alpha) * action_latents)
        decoded = _retag_candidates(decoder.decode(composed, top_k=max(1, int(samples_per_alpha))), origin=f"phase4_edit_action_alpha{_format_float(alpha)}")
        scores_by_task = [score_candidates(example, candidates) for example, candidates in zip(examples, decoded)]
        summary = summarize_scores(scores_by_task)
        summary.update(
            {
                "alpha": float(alpha),
                "latent_cosine": _mean_cosine(composed, target_latents),
                "latent_mse": float(np.mean((composed - target_latents) ** 2)),
            }
        )
        per_alpha_scores.append(summary)
        decoded_by_alpha.append((float(alpha), decoded))

    combined: list[list[Candidate]] = []
    task_count = len(examples)
    for task_idx in range(task_count):
        best_by_key: dict[str, Candidate] = {}
        for alpha, decoded in decoded_by_alpha:
            for candidate in decoded[task_idx]:
                key = canonicalize_smiles(candidate.smiles) or f"invalid:{candidate.smiles}"
                adjusted = float(candidate.score) - float(alpha_score_penalty) * abs(float(alpha) - 1.0)
                updated = Candidate(smiles=candidate.smiles, origin=candidate.origin, score=adjusted, rank=0)
                if key not in best_by_key or updated.score > best_by_key[key].score:
                    best_by_key[key] = updated
        ranked = sorted(best_by_key.values(), key=lambda item: item.score, reverse=True)[:top_k]
        combined.append([Candidate(smiles=item.smiles, origin=item.origin, score=item.score, rank=idx) for idx, item in enumerate(ranked, start=1)])
    return per_alpha_scores, combined


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-8, None)


def _row_norms(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(values, dtype=np.float32), axis=1).astype(np.float32)


def _action_norm_metrics(prefix: str, pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred_norm = np.linalg.norm(pred, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    return {
        f"{prefix}_pred_norm_mean": float(np.mean(pred_norm)),
        f"{prefix}_pred_norm_std": float(np.std(pred_norm)),
        f"{prefix}_target_norm_mean": float(np.mean(target_norm)),
        f"{prefix}_norm_mae": float(np.mean(np.abs(pred_norm - target_norm))),
    }


def _action_step_metrics(prefix: str, pred_steps: np.ndarray, target_steps: np.ndarray) -> dict[str, float]:
    pred_steps = np.asarray(pred_steps, dtype=np.float32)
    target_steps = np.asarray(target_steps, dtype=np.float32)
    return {
        f"{prefix}_pred_mean": float(np.mean(pred_steps)),
        f"{prefix}_pred_std": float(np.std(pred_steps)),
        f"{prefix}_target_mean": float(np.mean(target_steps)),
        f"{prefix}_mae": float(np.mean(np.abs(pred_steps - target_steps))),
    }


def _write_alpha_summary(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_float_tuple(value: str, default: tuple[float, ...]) -> tuple[float, ...]:
    if not value:
        return default
    parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    return parsed or default


def _format_float(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "_").replace("-", "m")


if __name__ == "__main__":
    main()
