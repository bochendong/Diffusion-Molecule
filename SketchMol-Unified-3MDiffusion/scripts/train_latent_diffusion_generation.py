#!/usr/bin/env python3
"""Train latent diffusion generation stream from edit condition tokens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_unified_3m_diffusion.edit_condition_tokens import (  # noqa: E402
    EditConditionTokenConnector,
    edit_condition_loss,
)
from sketchmol_unified_3m_diffusion.latent_diffusion_generation import (  # noqa: E402
    EditLatentDenoiser,
    GaussianLatentDiffusion,
)
from sketchmol_unified_3m_diffusion.runtime import (  # noqa: E402
    checkpoint_dir,
    device_report,
    latest_checkpoint_path,
    move_batch_to_device,
    resolve_device,
)
from sketchmol_unified_3m_diffusion.unified_condition_dataset import EDIT_GENERATION, PROPERTY_COLUMNS, read_jsonl  # noqa: E402
from sketchmol_unified_3m_diffusion.unified_featurization import (  # noqa: E402
    hidden_sequence_for_sample,
    source_latent_vector,
    target_latent_vector,
)


class LatentDiffusionDataset(Dataset):
    def __init__(self, jsonl: Path, *, token_dim: int, fingerprint_dim: int, limit: int | None = None):
        samples = [sample for sample in read_jsonl(jsonl) if sample.task_type == EDIT_GENERATION]
        if limit is not None:
            samples = samples[:limit]
        if not samples:
            raise ValueError(f"No edit_generation rows found in {jsonl}")
        self.hidden = np.stack([hidden_sequence_for_sample(sample, token_dim=token_dim) for sample in samples])
        self.latents = np.stack([target_latent_vector(sample, fingerprint_dim=fingerprint_dim) for sample in samples])
        self.source_latents = np.stack([source_latent_vector(sample, fingerprint_dim=fingerprint_dim) for sample in samples])
        self.source_tanimoto = np.asarray([_float_or_nan(sample.source_tanimoto) for sample in samples], dtype=np.float32)
        self.latent_mean = self.latents.mean(axis=0, keepdims=True)
        self.latent_std = self.latents.std(axis=0, keepdims=True)
        self.latent_std = np.where(self.latent_std < 1e-6, 1.0, self.latent_std)
        self.latents = (self.latents - self.latent_mean) / self.latent_std
        self.source_latents = (self.source_latents - self.latent_mean) / self.latent_std

    def __len__(self) -> int:
        return int(self.hidden.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "hidden": torch.from_numpy(self.hidden[idx]).float(),
            "latent": torch.from_numpy(self.latents[idx]).float(),
            "source_latent": torch.from_numpy(self.source_latents[idx]).float(),
            "source_tanimoto": torch.tensor(self.source_tanimoto[idx], dtype=torch.float32),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--condition-connector", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fingerprint-dim", type=int, default=512)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--diffusion-objective", choices=["pred_noise", "pred_x0"], default="pred_x0")
    parser.add_argument("--diffusion-target", choices=["target", "residual", "source_residual"], default="residual")
    parser.add_argument("--prior-loss-weight", type=float, default=0.0)
    parser.add_argument("--source-regret-loss-weight", type=float, default=0.0)
    parser.add_argument("--source-regret-margin", type=float, default=0.0)
    parser.add_argument("--source-radius-loss-weight", type=float, default=0.0)
    parser.add_argument("--source-radius-margin", type=float, default=0.05)
    parser.add_argument("--source-similarity-weight-floor", type=float, default=0.25)
    parser.add_argument("--source-fingerprint-prior-blend", type=float, default=0.0)
    parser.add_argument("--fingerprint-guard-loss-weight", type=float, default=0.0)
    parser.add_argument("--fingerprint-guard-margin", type=float, default=0.02)
    parser.add_argument(
        "--source-residual-radius-multiplier",
        type=float,
        default=0.0,
        help=(
            "When positive, cap predicted source residuals to this multiple of the connector prior's "
            "source radius before source-aware guard losses are computed."
        ),
    )
    parser.add_argument(
        "--source-residual-radius-margin",
        type=float,
        default=0.0,
        help="Extra normalized-radius allowance added to the source residual cap.",
    )
    parser.add_argument(
        "--source-residual-min-radius",
        type=float,
        default=0.0,
        help="Minimum normalized source residual radius allowed by the cap.",
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train-connector", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--allow-incompatible-resume-weights", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(json.dumps({"event": "device", **device_report(device)}, sort_keys=True))
    connector_payload = torch.load(args.condition_connector, map_location=device)
    connector_config = connector_payload["config"]
    dataset = LatentDiffusionDataset(
        args.train_jsonl,
        token_dim=int(connector_config["token_dim"]),
        fingerprint_dim=args.fingerprint_dim,
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    connector = EditConditionTokenConnector(
        input_hidden_dim=int(connector_config["token_dim"]),
        context_dim=int(connector_config["context_dim"]),
        num_queries=int(connector_config["num_queries"]),
        hidden_dim=int(connector_config["hidden_dim"]),
        fingerprint_dim=int(connector_config["fingerprint_dim"]),
    ).to(device)
    connector.load_state_dict(connector_payload["model_state"])
    if not args.train_connector:
        connector.eval()
        for param in connector.parameters():
            param.requires_grad = False

    latent_dim = int(dataset.latents.shape[1])
    denoiser = EditLatentDenoiser(
        latent_dim=latent_dim,
        context_dim=int(connector_config["context_dim"]),
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    )
    diffusion = GaussianLatentDiffusion(denoiser, timesteps=args.timesteps, objective=args.diffusion_objective).to(device)
    params = list(diffusion.parameters()) + [p for p in connector.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    latent_mean = torch.from_numpy(dataset.latent_mean.astype(np.float32)).to(device)
    latent_std = torch.from_numpy(dataset.latent_std.astype(np.float32)).to(device)

    history = []
    config = _config(args, dataset, connector_config, latent_dim)
    start_epoch = 0
    if args.resume_checkpoint is not None:
        payload = torch.load(args.resume_checkpoint, map_location=device)
        resume_config = payload.get("config", {})
        if not _resume_config_compatible(resume_config, config):
            if args.allow_incompatible_resume_weights:
                try:
                    diffusion.load_state_dict(payload["diffusion_state"])
                    connector.load_state_dict(payload["connector_state"])
                except RuntimeError as exc:
                    print(
                        json.dumps(
                            {
                                "event": "resume_checkpoint_skipped",
                                "reason": f"incompatible weights: {exc}",
                                "checkpoint": str(args.resume_checkpoint),
                            },
                            sort_keys=True,
                        )
                    )
                else:
                    print(
                        json.dumps(
                            {
                                "event": "warm_started_incompatible_resume_weights",
                                "reason": "training objective changed",
                                "checkpoint": str(args.resume_checkpoint),
                            },
                            sort_keys=True,
                        )
                    )
            else:
                print(
                    json.dumps(
                        {
                            "event": "resume_checkpoint_skipped",
                            "reason": "training objective changed",
                            "checkpoint": str(args.resume_checkpoint),
                        },
                        sort_keys=True,
                    )
                )
        else:
            diffusion.load_state_dict(payload["diffusion_state"])
            connector.load_state_dict(payload["connector_state"])
            resume_train_connector = bool(resume_config.get("train_connector", False)) if isinstance(resume_config, dict) else False
            if resume_train_connector == bool(args.train_connector):
                try:
                    optimizer.load_state_dict(payload["optimizer_state"])
                except ValueError as exc:
                    print(
                        json.dumps(
                            {
                                "event": "optimizer_state_skipped",
                                "reason": str(exc),
                                "checkpoint": str(args.resume_checkpoint),
                            },
                            sort_keys=True,
                        )
                    )
            else:
                print(
                    json.dumps(
                        {
                            "event": "optimizer_state_skipped",
                            "reason": "train_connector setting changed",
                            "checkpoint_train_connector": resume_train_connector,
                            "current_train_connector": bool(args.train_connector),
                            "checkpoint": str(args.resume_checkpoint),
                        },
                        sort_keys=True,
                    )
                )
            history = list(payload.get("history", []))
            start_epoch = int(payload.get("epoch", 0))
            print(json.dumps({"event": "resumed", "checkpoint": str(args.resume_checkpoint), "start_epoch": start_epoch}, sort_keys=True))

    for epoch in range(start_epoch, args.epochs):
        diffusion.train()
        losses = []
        diffusion_losses = []
        prior_losses = []
        source_regret_losses = []
        source_radius_losses = []
        source_fingerprint_losses = []
        source_residual_clamped_rates = []
        source_residual_scale_means = []
        source_worse_rates = []
        source_fingerprint_worse_rates = []
        source_fingerprint_gains = []
        train_target_mae = []
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(args.train_connector):
                condition = connector(batch["hidden"].float())
            tokens = condition.tokens if args.train_connector else condition.tokens.detach()
            mask = condition.attention_mask if args.train_connector else condition.attention_mask.detach()
            target_latent = batch["latent"].float()
            prior_latent = condition_prior_latent(
                condition,
                connector_config=connector_config,
                latent_mean=latent_mean,
                latent_std=latent_std,
                fingerprint_dim=args.fingerprint_dim,
                latent_dim=latent_dim,
                source_latent=batch["source_latent"].float(),
                source_tanimoto=batch["source_tanimoto"].float(),
                source_fingerprint_prior_blend=args.source_fingerprint_prior_blend,
                source_similarity_weight_floor=args.source_similarity_weight_floor,
            )
            if not args.train_connector:
                prior_latent = prior_latent.detach()
            source_latent = batch["source_latent"].float()
            diffusion_target = diffusion_target_latent(
                args.diffusion_target,
                target_latent=target_latent,
                prior_latent=prior_latent,
                source_latent=source_latent,
            )
            diffusion_loss, pred_diffusion_target = diffusion.loss_and_pred_x0(diffusion_target, tokens, mask)
            prior_loss = F.mse_loss(prior_latent, target_latent)
            pred_final_latent = final_latent_from_diffusion_prediction(
                args.diffusion_target,
                pred_diffusion_target=pred_diffusion_target,
                prior_latent=prior_latent,
                source_latent=source_latent,
            )
            pred_guard_latent, clamp_logs = clamp_source_residual_latent(
                pred_final_latent,
                source_latent,
                reference_latent=prior_latent,
                radius_multiplier=args.source_residual_radius_multiplier,
                radius_margin=args.source_residual_radius_margin,
                min_radius=args.source_residual_min_radius,
            )
            source_losses = source_guard_losses(
                pred_guard_latent,
                target_latent,
                source_latent,
                batch["source_tanimoto"].float(),
                fingerprint_dim=args.fingerprint_dim,
                source_regret_margin=args.source_regret_margin,
                source_radius_margin=args.source_radius_margin,
                source_similarity_weight_floor=args.source_similarity_weight_floor,
                fingerprint_guard_margin=args.fingerprint_guard_margin,
                latent_mean=latent_mean,
                latent_std=latent_std,
            )
            loss = (
                diffusion_loss
                + float(args.prior_loss_weight) * prior_loss
                + float(args.source_regret_loss_weight) * source_losses["source_property_regret"]
                + float(args.source_radius_loss_weight) * source_losses["source_radius_regret"]
                + float(args.fingerprint_guard_loss_weight) * source_losses["source_fingerprint_regret"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            diffusion_losses.append(float(diffusion_loss.item()))
            prior_losses.append(float(prior_loss.item()))
            source_regret_losses.append(float(source_losses["source_property_regret"].item()))
            source_radius_losses.append(float(source_losses["source_radius_regret"].item()))
            source_fingerprint_losses.append(float(source_losses["source_fingerprint_regret"].item()))
            source_residual_clamped_rates.append(float(clamp_logs["source_residual_clamped_rate"].item()))
            source_residual_scale_means.append(float(clamp_logs["source_residual_scale_mean"].item()))
            source_worse_rates.append(float(source_losses["source_property_worse_rate"].item()))
            source_fingerprint_worse_rates.append(float(source_losses["source_fingerprint_worse_rate"].item()))
            source_fingerprint_gains.append(float(source_losses["source_fingerprint_cosine_gain"].item()))
            train_target_mae.append(float(torch.mean(torch.abs(diffusion_target)).item()))
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "diffusion_loss": float(np.mean(diffusion_losses)),
            "prior_mse": float(np.mean(prior_losses)),
            "source_property_regret": float(np.mean(source_regret_losses)),
            "source_radius_regret": float(np.mean(source_radius_losses)),
            "source_fingerprint_regret": float(np.mean(source_fingerprint_losses)),
            "source_residual_clamped_rate": float(np.mean(source_residual_clamped_rates)),
            "source_residual_scale_mean": float(np.mean(source_residual_scale_means)),
            "source_property_worse_rate": float(np.mean(source_worse_rates)),
            "source_fingerprint_worse_rate": float(np.mean(source_fingerprint_worse_rates)),
            "source_fingerprint_cosine_gain": float(np.mean(source_fingerprint_gains)),
            "diffusion_target_mae": float(np.mean(train_target_mae)),
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True))
        if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
            _save_checkpoint(
                args.output_dir,
                epoch=epoch + 1,
                diffusion=diffusion,
                connector=connector,
                optimizer=optimizer,
                config=config,
                history=history,
            )

    torch.save(
        {
            "diffusion_state": diffusion.state_dict(),
            "connector_state": connector.state_dict(),
            "config": config,
            "history": history,
        },
        args.output_dir / "latent_diffusion_generation.pt",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"history": history, "config": config, "device": device_report(device)}, indent=2) + "\n",
        encoding="utf-8",
    )


def _config(
    args: argparse.Namespace,
    dataset: LatentDiffusionDataset,
    connector_config: dict[str, object],
    latent_dim: int,
) -> dict[str, object]:
    return {
        "latent_dim": latent_dim,
        "fingerprint_dim": args.fingerprint_dim,
        "timesteps": args.timesteps,
        "diffusion_objective": args.diffusion_objective,
        "diffusion_target": args.diffusion_target,
        "prior_loss_weight": args.prior_loss_weight,
        "source_regret_loss_weight": args.source_regret_loss_weight,
        "source_regret_margin": args.source_regret_margin,
        "source_radius_loss_weight": args.source_radius_loss_weight,
        "source_radius_margin": args.source_radius_margin,
        "source_similarity_weight_floor": args.source_similarity_weight_floor,
        "source_fingerprint_prior_blend": args.source_fingerprint_prior_blend,
        "fingerprint_guard_loss_weight": args.fingerprint_guard_loss_weight,
        "fingerprint_guard_margin": args.fingerprint_guard_margin,
        "source_residual_radius_multiplier": args.source_residual_radius_multiplier,
        "source_residual_radius_margin": args.source_residual_radius_margin,
        "source_residual_min_radius": args.source_residual_min_radius,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "connector_config": connector_config,
        "latent_mean": dataset.latent_mean.tolist(),
        "latent_std": dataset.latent_std.tolist(),
        "train_connector": bool(args.train_connector),
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory),
        "seed": args.seed,
    }


def source_guard_losses(
    pred_latent: torch.Tensor,
    target_latent: torch.Tensor,
    source_latent: torch.Tensor,
    source_tanimoto: torch.Tensor,
    *,
    fingerprint_dim: int,
    source_regret_margin: float,
    source_radius_margin: float,
    source_similarity_weight_floor: float,
    fingerprint_guard_margin: float,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    num_props = len(PROPERTY_COLUMNS)
    prop_slice = slice(fingerprint_dim, fingerprint_dim + num_props)
    pred_prop_error = torch.mean(torch.abs(pred_latent[:, prop_slice] - target_latent[:, prop_slice]), dim=1)
    source_prop_error = torch.mean(torch.abs(source_latent[:, prop_slice] - target_latent[:, prop_slice]), dim=1)
    weights = _source_similarity_weights(
        source_tanimoto,
        floor=source_similarity_weight_floor,
        device=pred_latent.device,
        dtype=pred_latent.dtype,
    )

    regret = torch.relu(pred_prop_error - source_prop_error - float(source_regret_margin))
    pred_source_radius = torch.mean(torch.abs(pred_latent - source_latent), dim=1)
    target_source_radius = torch.mean(torch.abs(target_latent - source_latent), dim=1)
    radius_regret = torch.relu(pred_source_radius - target_source_radius - float(source_radius_margin))

    fp_slice = slice(0, fingerprint_dim)
    pred_fp = _unnormalized_block(pred_latent, latent_mean, latent_std, fp_slice)
    target_fp = _unnormalized_block(target_latent, latent_mean, latent_std, fp_slice)
    source_fp = _unnormalized_block(source_latent, latent_mean, latent_std, fp_slice)
    pred_target_cosine = F.cosine_similarity(pred_fp, target_fp, dim=1, eps=1e-8)
    source_target_cosine = F.cosine_similarity(source_fp, target_fp, dim=1, eps=1e-8)
    fingerprint_regret = torch.relu(source_target_cosine - pred_target_cosine - float(fingerprint_guard_margin))
    return {
        "source_property_regret": torch.mean(weights * regret.pow(2)),
        "source_radius_regret": torch.mean(weights * radius_regret.pow(2)),
        "source_property_worse_rate": torch.mean((pred_prop_error > source_prop_error).to(dtype=pred_latent.dtype)),
        "source_fingerprint_regret": torch.mean(weights * fingerprint_regret.pow(2)),
        "source_fingerprint_worse_rate": torch.mean((pred_target_cosine < source_target_cosine).to(dtype=pred_latent.dtype)),
        "source_fingerprint_cosine_gain": torch.mean(pred_target_cosine - source_target_cosine),
    }


def _source_similarity_weights(
    source_tanimoto: torch.Tensor,
    *,
    floor: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = source_tanimoto.to(device=device, dtype=dtype).reshape(-1)
    finite = torch.isfinite(values)
    scaled = ((values - 0.4) / 0.55).clamp(0.0, 1.0)
    scaled = torch.where(finite, scaled, torch.full_like(scaled, 0.5))
    floor_value = min(max(float(floor), 0.0), 1.0)
    return floor_value + (1.0 - floor_value) * scaled


def _float_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _resume_config_compatible(resume_config: object, current_config: dict[str, object]) -> bool:
    if not isinstance(resume_config, dict):
        return False
    keys = [
        "diffusion_objective",
        "diffusion_target",
        "prior_loss_weight",
        "source_regret_loss_weight",
        "source_regret_margin",
        "source_radius_loss_weight",
        "source_radius_margin",
        "source_similarity_weight_floor",
        "source_fingerprint_prior_blend",
        "fingerprint_guard_loss_weight",
        "fingerprint_guard_margin",
        "source_residual_radius_multiplier",
        "source_residual_radius_margin",
        "source_residual_min_radius",
        "train_connector",
    ]
    for key in keys:
        if resume_config.get(key) != current_config.get(key):
            return False
    return True


def diffusion_target_latent(
    mode: str,
    *,
    target_latent: torch.Tensor,
    prior_latent: torch.Tensor,
    source_latent: torch.Tensor,
) -> torch.Tensor:
    if mode == "target":
        return target_latent
    if mode == "residual":
        return target_latent - prior_latent
    if mode == "source_residual":
        return target_latent - source_latent
    raise ValueError(f"Unsupported diffusion target: {mode}")


def final_latent_from_diffusion_prediction(
    mode: str,
    *,
    pred_diffusion_target: torch.Tensor,
    prior_latent: torch.Tensor,
    source_latent: torch.Tensor,
) -> torch.Tensor:
    if mode == "target":
        return pred_diffusion_target
    if mode == "residual":
        return pred_diffusion_target + prior_latent
    if mode == "source_residual":
        return pred_diffusion_target + source_latent
    raise ValueError(f"Unsupported diffusion target: {mode}")


def clamp_source_residual_latent(
    pred_latent: torch.Tensor,
    source_latent: torch.Tensor,
    *,
    reference_latent: torch.Tensor,
    radius_multiplier: float,
    radius_margin: float,
    min_radius: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    multiplier = float(radius_multiplier)
    if multiplier <= 0.0:
        zero = torch.zeros((), device=pred_latent.device, dtype=pred_latent.dtype)
        one = torch.ones((), device=pred_latent.device, dtype=pred_latent.dtype)
        return pred_latent, {"source_residual_clamped_rate": zero, "source_residual_scale_mean": one}

    residual = pred_latent - source_latent
    pred_radius = torch.mean(torch.abs(residual), dim=1, keepdim=True)
    reference_radius = torch.mean(torch.abs(reference_latent - source_latent), dim=1, keepdim=True)
    budget = reference_radius * multiplier + float(radius_margin)
    budget = budget.clamp_min(float(min_radius))
    scale = torch.minimum(torch.ones_like(pred_radius), budget / pred_radius.clamp_min(1e-8))
    clamped = source_latent + residual * scale
    logs = {
        "source_residual_clamped_rate": torch.mean((scale < 0.999).to(dtype=pred_latent.dtype)),
        "source_residual_scale_mean": torch.mean(scale),
    }
    return clamped, logs


def condition_prior_latent(
    condition,
    *,
    connector_config: dict[str, object],
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    fingerprint_dim: int,
    latent_dim: int,
    source_latent: torch.Tensor | None = None,
    source_tanimoto: torch.Tensor | None = None,
    source_fingerprint_prior_blend: float = 0.0,
    source_similarity_weight_floor: float = 0.25,
) -> torch.Tensor:
    """Build a normalized latent prior from edit connector prediction heads."""

    batch = condition.target_fingerprint_logits.shape[0]
    device = condition.target_fingerprint_logits.device
    dtype = condition.target_fingerprint_logits.dtype
    num_props = len(PROPERTY_COLUMNS)

    fingerprint = torch.sigmoid(condition.target_fingerprint_logits[:, :fingerprint_dim])
    if source_latent is not None and float(source_fingerprint_prior_blend) > 0.0:
        source_fingerprint = _unnormalized_block(
            source_latent.to(device=device, dtype=dtype),
            latent_mean.to(dtype=dtype),
            latent_std.to(dtype=dtype),
            slice(0, fingerprint_dim),
        )
        blend = _source_fingerprint_blend(
            source_tanimoto,
            batch=batch,
            max_blend=source_fingerprint_prior_blend,
            source_similarity_weight_floor=source_similarity_weight_floor,
            device=device,
            dtype=dtype,
        )
        fingerprint = blend * source_fingerprint + (1.0 - blend) * fingerprint
    prop_mean = _config_tensor(connector_config, "property_mean", device=device, dtype=dtype)
    prop_std = _config_tensor(connector_config, "property_std", device=device, dtype=dtype)
    delta_mean = _config_tensor(connector_config, "delta_mean", device=device, dtype=dtype)
    delta_std = _config_tensor(connector_config, "delta_std", device=device, dtype=dtype)
    props = condition.target_properties[:, :num_props] * prop_std[:, :num_props] + prop_mean[:, :num_props]
    deltas = condition.property_deltas[:, :num_props] * delta_std[:, :num_props] + delta_mean[:, :num_props]
    active = torch.sigmoid(condition.active_logits[:, :num_props])

    raw = torch.zeros(batch, latent_dim, device=device, dtype=dtype)
    raw[:, :fingerprint_dim] = fingerprint
    prop_start = fingerprint_dim
    delta_start = fingerprint_dim + 32
    active_start = fingerprint_dim + 64
    raw[:, prop_start : prop_start + num_props] = props
    raw[:, delta_start : delta_start + num_props] = deltas
    raw[:, active_start : active_start + num_props] = active
    return ((raw - latent_mean) / latent_std.clamp_min(1e-6)).to(dtype=torch.float32)


def _config_tensor(config: dict[str, object], key: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(config[key], dtype=dtype, device=device)


def _unnormalized_block(
    latent: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    block: slice,
) -> torch.Tensor:
    return latent[:, block] * latent_std[:, block].clamp_min(1e-6) + latent_mean[:, block]


def _source_fingerprint_blend(
    source_tanimoto: torch.Tensor | None,
    *,
    batch: int,
    max_blend: float,
    source_similarity_weight_floor: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    max_value = min(max(float(max_blend), 0.0), 1.0)
    if source_tanimoto is None:
        return torch.full((batch, 1), max_value, device=device, dtype=dtype)
    weights = _source_similarity_weights(
        source_tanimoto,
        floor=source_similarity_weight_floor,
        device=device,
        dtype=dtype,
    )
    return (max_value * weights).reshape(batch, 1)


def _save_checkpoint(
    output_dir: Path,
    *,
    epoch: int,
    diffusion: GaussianLatentDiffusion,
    connector: EditConditionTokenConnector,
    optimizer: torch.optim.Optimizer,
    config: dict[str, object],
    history: list[dict[str, float]],
) -> None:
    payload = {
        "epoch": epoch,
        "diffusion_state": diffusion.state_dict(),
        "connector_state": connector.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "history": history,
    }
    ckpt_dir = checkpoint_dir(output_dir)
    torch.save(payload, ckpt_dir / f"epoch_{epoch:04d}.pt")
    torch.save(payload, latest_checkpoint_path(output_dir))


if __name__ == "__main__":
    main()
