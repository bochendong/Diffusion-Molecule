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

from sketchmol_unified_3m_diffusion.edit_condition_tokens import EditConditionTokenConnector  # noqa: E402
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
from sketchmol_unified_3m_diffusion.unified_featurization import hidden_sequence_for_sample, target_latent_vector  # noqa: E402


class LatentDiffusionDataset(Dataset):
    def __init__(self, jsonl: Path, *, token_dim: int, fingerprint_dim: int, limit: int | None = None):
        samples = [sample for sample in read_jsonl(jsonl) if sample.task_type == EDIT_GENERATION]
        if limit is not None:
            samples = samples[:limit]
        if not samples:
            raise ValueError(f"No edit_generation rows found in {jsonl}")
        self.hidden = np.stack([hidden_sequence_for_sample(sample, token_dim=token_dim) for sample in samples])
        self.latents = np.stack([target_latent_vector(sample, fingerprint_dim=fingerprint_dim) for sample in samples])
        self.latent_mean = self.latents.mean(axis=0, keepdims=True)
        self.latent_std = self.latents.std(axis=0, keepdims=True)
        self.latent_std = np.where(self.latent_std < 1e-6, 1.0, self.latent_std)
        self.latents = (self.latents - self.latent_mean) / self.latent_std

    def __len__(self) -> int:
        return int(self.hidden.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "hidden": torch.from_numpy(self.hidden[idx]).float(),
            "latent": torch.from_numpy(self.latents[idx]).float(),
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
    parser.add_argument("--diffusion-target", choices=["target", "residual"], default="residual")
    parser.add_argument("--prior-loss-weight", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train-connector", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
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
        diffusion.load_state_dict(payload["diffusion_state"])
        connector.load_state_dict(payload["connector_state"])
        resume_config = payload.get("config", {})
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
            )
            if not args.train_connector:
                prior_latent = prior_latent.detach()
            diffusion_target = target_latent - prior_latent if args.diffusion_target == "residual" else target_latent
            diffusion_loss = diffusion.loss(diffusion_target, tokens, mask)
            prior_loss = F.mse_loss(prior_latent, target_latent)
            loss = diffusion_loss + float(args.prior_loss_weight) * prior_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            diffusion_losses.append(float(diffusion_loss.item()))
            prior_losses.append(float(prior_loss.item()))
            train_target_mae.append(float(torch.mean(torch.abs(diffusion_target)).item()))
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "diffusion_loss": float(np.mean(diffusion_losses)),
            "prior_mse": float(np.mean(prior_losses)),
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
    }


def condition_prior_latent(
    condition,
    *,
    connector_config: dict[str, object],
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    fingerprint_dim: int,
    latent_dim: int,
) -> torch.Tensor:
    """Build a normalized latent prior from edit connector prediction heads."""

    batch = condition.target_fingerprint_logits.shape[0]
    device = condition.target_fingerprint_logits.device
    dtype = condition.target_fingerprint_logits.dtype
    num_props = len(PROPERTY_COLUMNS)

    fingerprint = torch.sigmoid(condition.target_fingerprint_logits[:, :fingerprint_dim])
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
