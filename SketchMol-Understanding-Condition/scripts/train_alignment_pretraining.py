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

from sketchmol_understanding_condition.edit_condition_tokens import MoleculeLanguageAlignmentModel  # noqa: E402
from sketchmol_understanding_condition.unified_condition_dataset import read_jsonl  # noqa: E402
from sketchmol_understanding_condition.unified_featurization import (  # noqa: E402
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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_data = AlignmentDataset(
        args.train_jsonl,
        molecule_dim=args.molecule_dim,
        text_dim=args.text_dim,
        image_dim=args.image_dim,
        limit=args.limit,
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    model = MoleculeLanguageAlignmentModel(
        molecule_dim=args.molecule_dim,
        text_dim=args.text_dim,
        image_dim=args.image_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
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

    checkpoint = {
        "model_state": model.state_dict(),
        "config": {
            "molecule_dim": args.molecule_dim,
            "text_dim": args.text_dim,
            "image_dim": args.image_dim,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
        },
        "history": history,
    }
    torch.save(checkpoint, args.output_dir / "alignment_model.pt")
    (args.output_dir / "metrics.json").write_text(json.dumps({"history": history}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

