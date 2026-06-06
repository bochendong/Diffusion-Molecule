#!/usr/bin/env python3
"""Train a SUCC molecule-image VAE with SketchMol-style 4x32x32 latents."""

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

from sketchmol_understanding_condition.molecule_image_vae import (  # noqa: E402
    MoleculeImageVAE,
    image_to_tensor,
    save_molecule_image_vae,
    tensor_to_uint8_image,
    vae_loss,
)
from sketchmol_understanding_condition.unified_condition_dataset import EDIT_GENERATION, read_jsonl  # noqa: E402


class MoleculeImageDataset(Dataset):
    """Source and target molecule images from unified edit rows."""

    def __init__(self, jsonl: Path, *, image_size: int, limit: int | None = None) -> None:
        samples = [sample for sample in read_jsonl(jsonl) if sample.task_type == EDIT_GENERATION]
        rows = []
        for sample in samples:
            if sample.source_image or sample.source_smiles:
                rows.append({"image_path": sample.source_image, "smiles": sample.source_smiles, "role": "source"})
            if sample.target_image or sample.target_smiles:
                rows.append({"image_path": sample.target_image, "smiles": sample.target_smiles, "role": "target"})
            if limit is not None and len(rows) >= limit:
                rows = rows[:limit]
                break
        if not rows:
            raise ValueError(f"No source/target images or SMILES found in {jsonl}")
        self.rows = rows
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[idx]
        image = image_to_tensor(
            image_path=str(row["image_path"]),
            smiles=str(row["smiles"]),
            image_size=self.image_size,
        )
        return {"image": image.float(), "role": str(row["role"])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--eval-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--foreground-weight", type=float, default=8.0)
    parser.add_argument("--foreground-gamma", type=float, default=1.0)
    parser.add_argument("--ink-loss-weight", type=float, default=4.0)
    parser.add_argument("--ink-fraction-weight", type=float, default=2.0)
    parser.add_argument("--sample-latent", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=29)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    train_data = MoleculeImageDataset(args.train_jsonl, image_size=args.image_size, limit=args.limit)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    eval_loader = None
    if args.eval_jsonl is not None and args.eval_jsonl.exists() and args.eval_limit != 0:
        eval_data = MoleculeImageDataset(args.eval_jsonl, image_size=args.image_size, limit=args.eval_limit)
        eval_loader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MoleculeImageVAE(
        image_size=args.image_size,
        latent_channels=args.latent_channels,
        latent_size=args.latent_size,
        base_channels=args.base_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        train_recon = []
        train_kl = []
        train_ink = []
        train_ink_fraction = []
        train_foreground = []
        train_background = []
        train_blank = []
        train_target_ink = []
        train_recon_ink = []
        for batch in train_loader:
            images = batch["image"].to(device)
            optimizer.zero_grad()
            output = model(images, sample=args.sample_latent)
            loss, logs = vae_loss(
                output,
                images,
                kl_weight=args.kl_weight,
                foreground_weight=args.foreground_weight,
                foreground_gamma=args.foreground_gamma,
                ink_loss_weight=args.ink_loss_weight,
                ink_fraction_weight=args.ink_fraction_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_recon.append(float(logs["reconstruction_l1"].item()))
            train_kl.append(float(logs["kl"].item()))
            train_ink.append(float(logs["ink_l1"].item()))
            train_ink_fraction.append(float(logs["ink_fraction_l1"].item()))
            train_foreground.append(float(logs["foreground_l1"].item()))
            train_background.append(float(logs["background_l1"].item()))
            train_blank.append(float(logs["blank_canvas_l1"].item()))
            train_target_ink.append(float(logs["target_ink_fraction"].item()))
            train_recon_ink.append(float(logs["reconstruction_ink_fraction"].item()))
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)),
            "train_reconstruction_l1": float(np.mean(train_recon)),
            "train_ink_l1": float(np.mean(train_ink)),
            "train_ink_fraction_l1": float(np.mean(train_ink_fraction)),
            "train_foreground_l1": float(np.mean(train_foreground)),
            "train_background_l1": float(np.mean(train_background)),
            "train_blank_canvas_l1": float(np.mean(train_blank)),
            "train_target_ink_fraction": float(np.mean(train_target_ink)),
            "train_reconstruction_ink_fraction": float(np.mean(train_recon_ink)),
            "train_kl": float(np.mean(train_kl)),
        }
        if eval_loader is not None:
            record.update(
                _evaluate(
                    model,
                    eval_loader,
                    device=device,
                    kl_weight=args.kl_weight,
                    foreground_weight=args.foreground_weight,
                    foreground_gamma=args.foreground_gamma,
                    ink_loss_weight=args.ink_loss_weight,
                    ink_fraction_weight=args.ink_fraction_weight,
                )
            )
        history.append(record)
        print(json.dumps(record, sort_keys=True))

    config = {
        "image_size": args.image_size,
        "latent_channels": args.latent_channels,
        "latent_size": args.latent_size,
        "base_channels": args.base_channels,
        "latent_dim": args.latent_channels * args.latent_size * args.latent_size,
        "foreground_weight": args.foreground_weight,
        "foreground_gamma": args.foreground_gamma,
        "ink_loss_weight": args.ink_loss_weight,
        "ink_fraction_weight": args.ink_fraction_weight,
        "sample_latent": args.sample_latent,
        "train_jsonl": str(args.train_jsonl),
        "eval_jsonl": str(args.eval_jsonl) if args.eval_jsonl else None,
        "history": history,
    }
    save_molecule_image_vae(args.output_dir / "molecule_image_vae.pt", model, config=config, metrics={"history": history})
    (args.output_dir / "metrics.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _save_reconstructions(model, train_data, args.output_dir / "reconstructions", device=device, max_images=8)


@torch.no_grad()
def _evaluate(
    model: MoleculeImageVAE,
    loader: DataLoader,
    *,
    device: torch.device,
    kl_weight: float,
    foreground_weight: float,
    foreground_gamma: float,
    ink_loss_weight: float,
    ink_fraction_weight: float,
) -> dict[str, float]:
    model.eval()
    losses = []
    recon = []
    kls = []
    ink_l1s = []
    ink_fraction_l1s = []
    foreground_l1s = []
    background_l1s = []
    blank_l1s = []
    target_ink_fractions = []
    reconstruction_ink_fractions = []
    for batch in loader:
        images = batch["image"].to(device)
        output = model(images, sample=False)
        loss, logs = vae_loss(
            output,
            images,
            kl_weight=kl_weight,
            foreground_weight=foreground_weight,
            foreground_gamma=foreground_gamma,
            ink_loss_weight=ink_loss_weight,
            ink_fraction_weight=ink_fraction_weight,
        )
        losses.append(float(loss.item()))
        recon.append(float(logs["reconstruction_l1"].item()))
        kls.append(float(logs["kl"].item()))
        ink_l1s.append(float(logs["ink_l1"].item()))
        ink_fraction_l1s.append(float(logs["ink_fraction_l1"].item()))
        foreground_l1s.append(float(logs["foreground_l1"].item()))
        background_l1s.append(float(logs["background_l1"].item()))
        blank_l1s.append(float(logs["blank_canvas_l1"].item()))
        target_ink_fractions.append(float(logs["target_ink_fraction"].item()))
        reconstruction_ink_fractions.append(float(logs["reconstruction_ink_fraction"].item()))
    return {
        "eval_loss": float(np.mean(losses)),
        "eval_reconstruction_l1": float(np.mean(recon)),
        "eval_ink_l1": float(np.mean(ink_l1s)),
        "eval_ink_fraction_l1": float(np.mean(ink_fraction_l1s)),
        "eval_kl": float(np.mean(kls)),
        "eval_foreground_l1": float(np.mean(foreground_l1s)),
        "eval_background_l1": float(np.mean(background_l1s)),
        "eval_blank_canvas_l1": float(np.mean(blank_l1s)),
        "eval_target_ink_fraction": float(np.mean(target_ink_fractions)),
        "eval_reconstruction_ink_fraction": float(np.mean(reconstruction_ink_fractions)),
    }


@torch.no_grad()
def _save_reconstructions(
    model: MoleculeImageVAE,
    dataset: MoleculeImageDataset,
    output_dir: Path,
    *,
    device: torch.device,
    max_images: int,
) -> None:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for idx in range(min(max_images, len(dataset))):
        item = dataset[idx]
        image = item["image"].unsqueeze(0).to(device)
        output = model(image, sample=False)
        original = tensor_to_uint8_image(image[0])
        recon = tensor_to_uint8_image(output.reconstruction[0])
        combined = np.concatenate([original, recon], axis=1)
        Image.fromarray(combined).save(output_dir / f"reconstruction_{idx:03d}.png")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


if __name__ == "__main__":
    main()
