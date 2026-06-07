#!/usr/bin/env python3
"""Train molecule-language/image-language alignment on unified JSONL rows."""

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

from sketchmol_unified_3m_diffusion.edit_condition_tokens import MoleculeLanguageAlignmentModel  # noqa: E402
from sketchmol_unified_3m_diffusion.runtime import (  # noqa: E402
    checkpoint_dir,
    device_report,
    latest_checkpoint_path,
    move_batch_to_device,
    resolve_device,
)
from sketchmol_unified_3m_diffusion.unified_condition_dataset import read_jsonl  # noqa: E402
from sketchmol_unified_3m_diffusion.unified_featurization import (  # noqa: E402
    image_feature,
    molecule_feature,
    text_feature,
)


class AlignmentDataset(Dataset):
    def __init__(self, jsonl: Path, *, molecule_dim: int, text_dim: int, image_dim: int, limit: int | None = None):
        samples = read_jsonl(jsonl)
        if limit is not None:
            samples = samples[:limit]
        self.rows = []
        for sample in samples:
            smiles = sample.molecule_smiles or sample.source_smiles or sample.target_smiles
            text = sample.description or sample.instruction or sample.prompt
            if not smiles or not text:
                continue
            self.rows.append(
                {
                    "molecule": molecule_feature(smiles, molecule_dim),
                    "text": text_feature(text, text_dim),
                    "image": image_feature(sample.source_image, image_dim),
                    "image_mask": bool(sample.source_image),
                }
            )
        if not self.rows:
            raise ValueError(f"No alignment rows found in {jsonl}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        return {
            "molecule": torch.from_numpy(row["molecule"]),
            "text": torch.from_numpy(row["text"]),
            "image": torch.from_numpy(row["image"]),
            "image_mask": torch.tensor(row["image_mask"], dtype=torch.bool),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--eval-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--molecule-dim", type=int, default=512)
    parser.add_argument("--text-dim", type=int, default=256)
    parser.add_argument("--image-dim", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(json.dumps({"event": "device", **device_report(device)}, sort_keys=True))

    train_data = AlignmentDataset(
        args.train_jsonl,
        molecule_dim=args.molecule_dim,
        text_dim=args.text_dim,
        image_dim=args.image_dim,
        limit=args.limit,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    model = MoleculeLanguageAlignmentModel(
        molecule_dim=args.molecule_dim,
        text_dim=args.text_dim,
        image_dim=args.image_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    start_epoch = 0
    if args.resume_checkpoint is not None:
        payload = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        history = list(payload.get("history", []))
        start_epoch = int(payload.get("epoch", 0))
        print(json.dumps({"event": "resumed", "checkpoint": str(args.resume_checkpoint), "start_epoch": start_epoch}, sort_keys=True))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            loss, _ = model.contrastive_loss(
                batch["molecule"].float(),
                batch["text"].float(),
                image_features=batch["image"].float(),
                image_mask=batch["image_mask"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses))})
        print(json.dumps(history[-1], sort_keys=True))
        if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
            _save_checkpoint(
                args.output_dir,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                config=_config(args),
                history=history,
            )

    checkpoint = {
        "model_state": model.state_dict(),
        "config": _config(args),
        "history": history,
    }
    torch.save(checkpoint, args.output_dir / "alignment_model.pt")
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"history": history, "device": device_report(device)}, indent=2) + "\n",
        encoding="utf-8",
    )


def _config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "molecule_dim": args.molecule_dim,
        "text_dim": args.text_dim,
        "image_dim": args.image_dim,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory),
        "seed": args.seed,
    }


def _save_checkpoint(
    output_dir: Path,
    *,
    epoch: int,
    model: MoleculeLanguageAlignmentModel,
    optimizer: torch.optim.Optimizer,
    config: dict[str, object],
    history: list[dict[str, float]],
) -> None:
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "history": history,
    }
    ckpt_dir = checkpoint_dir(output_dir)
    torch.save(payload, ckpt_dir / f"epoch_{epoch:04d}.pt")
    torch.save(payload, latest_checkpoint_path(output_dir))


if __name__ == "__main__":
    main()
