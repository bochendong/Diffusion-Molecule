#!/usr/bin/env python3
"""Train latent diffusion generation stream from edit condition tokens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_understanding_condition.edit_condition_tokens import EditConditionTokenConnector  # noqa: E402
from sketchmol_understanding_condition.latent_diffusion_generation import (  # noqa: E402
    EditLatentDenoiser,
    GaussianLatentDiffusion,
)
from sketchmol_understanding_condition.unified_condition_dataset import EDIT_GENERATION, read_jsonl  # noqa: E402
from sketchmol_understanding_condition.unified_featurization import hidden_sequence_for_sample, target_latent_vector  # noqa: E402


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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fingerprint-dim", type=int, default=512)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train-connector", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    connector_payload = torch.load(args.condition_connector, map_location="cpu")
    connector_config = connector_payload["config"]
    dataset = LatentDiffusionDataset(
        args.train_jsonl,
        token_dim=int(connector_config["token_dim"]),
        fingerprint_dim=args.fingerprint_dim,
        limit=args.limit,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    connector = EditConditionTokenConnector(
        input_hidden_dim=int(connector_config["token_dim"]),
        context_dim=int(connector_config["context_dim"]),
        num_queries=int(connector_config["num_queries"]),
        hidden_dim=int(connector_config["hidden_dim"]),
        fingerprint_dim=int(connector_config["fingerprint_dim"]),
    )
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
    diffusion = GaussianLatentDiffusion(denoiser, timesteps=args.timesteps, objective="pred_noise")
    params = list(diffusion.parameters()) + [p for p in connector.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    history = []
    for epoch in range(args.epochs):
        diffusion.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad()
            with torch.set_grad_enabled(args.train_connector):
                condition = connector(batch["hidden"].float())
            tokens = condition.tokens if args.train_connector else condition.tokens.detach()
            mask = condition.attention_mask if args.train_connector else condition.attention_mask.detach()
            loss = diffusion.loss(batch["latent"].float(), tokens, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses))}
        history.append(record)
        print(json.dumps(record, sort_keys=True))

    config = {
        "latent_dim": latent_dim,
        "fingerprint_dim": args.fingerprint_dim,
        "timesteps": args.timesteps,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "connector_config": connector_config,
        "latent_mean": dataset.latent_mean.tolist(),
        "latent_std": dataset.latent_std.tolist(),
    }
    torch.save(
        {
            "diffusion_state": diffusion.state_dict(),
            "connector_state": connector.state_dict(),
            "config": config,
            "history": history,
        },
        args.output_dir / "latent_diffusion_generation.pt",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps({"history": history, "config": config}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

