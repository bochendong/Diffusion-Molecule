#!/usr/bin/env python3
"""Export target latents for UniVideo materialized candidate libraries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_understanding_condition.unified_condition_dataset import EDIT_GENERATION, read_jsonl  # noqa: E402
from sketchmol_understanding_condition.unified_featurization import target_latent_vector  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--output-npy", required=True, type=Path)
    parser.add_argument(
        "--latent-backend",
        choices=["fingerprint_property_vector", "image_vae", "sketchmol_vae"],
        default="fingerprint_property_vector",
    )
    parser.add_argument("--fingerprint-dim", type=int, default=512)
    parser.add_argument("--image-vae-checkpoint", type=Path, default=None)
    parser.add_argument("--sketchmol-root", type=Path, default=None)
    parser.add_argument("--sketchmol-vae-config", type=Path, default=None)
    parser.add_argument("--sketchmol-vae-checkpoint", type=Path, default=None)
    parser.add_argument("--sketchmol-scale-factor", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    samples = [sample for sample in read_jsonl(args.jsonl) if sample.task_type == EDIT_GENERATION]
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError(f"No edit_generation rows found in {args.jsonl}")

    if args.latent_backend == "fingerprint_property_vector":
        latents = np.stack(
            [target_latent_vector(sample, fingerprint_dim=args.fingerprint_dim) for sample in samples]
        ).astype(np.float32)
    else:
        device = resolve_device(args.device)
        vae = load_image_vae(args, device=device)
        latents = encode_target_image_latents(
            samples,
            vae,
            image_size=args.image_size,
            batch_size=args.batch_size,
            device=device,
        )

    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, latents.astype(np.float32))
    summary = {
        "jsonl": str(args.jsonl),
        "output_npy": str(args.output_npy),
        "rows": int(latents.shape[0]),
        "latent_shape": list(latents.shape),
        "latent_backend": args.latent_backend,
    }
    args.output_npy.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def resolve_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_image_vae(args: argparse.Namespace, *, device):
    if args.latent_backend == "image_vae":
        from sketchmol_understanding_condition.molecule_image_vae import load_molecule_image_vae

        if args.image_vae_checkpoint is None:
            raise ValueError("--image-vae-checkpoint is required for --latent-backend image_vae")
        model = load_molecule_image_vae(args.image_vae_checkpoint, map_location=device).to(device)
    elif args.latent_backend == "sketchmol_vae":
        from sketchmol_understanding_condition.sketchmol_vae_adapter import load_sketchmol_vae_adapter

        missing = [
            name
            for name, value in {
                "--sketchmol-root": args.sketchmol_root,
                "--sketchmol-vae-config": args.sketchmol_vae_config,
                "--sketchmol-vae-checkpoint": args.sketchmol_vae_checkpoint,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(f"{', '.join(missing)} are required for --latent-backend sketchmol_vae")
        model = load_sketchmol_vae_adapter(
            sketchmol_root=args.sketchmol_root,
            config_path=args.sketchmol_vae_config,
            checkpoint_path=args.sketchmol_vae_checkpoint,
            map_location=device,
            scale_factor=args.sketchmol_scale_factor,
        )
    else:
        raise ValueError(f"Unsupported image latent backend: {args.latent_backend}")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def encode_target_image_latents(
    samples,
    image_vae,
    *,
    image_size: int,
    batch_size: int,
    device,
) -> np.ndarray:
    import torch
    from sketchmol_understanding_condition.molecule_image_vae import image_to_tensor

    out = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            batch = torch.stack(
                [
                    image_to_tensor(
                        image_path=sample.target_image,
                        smiles=sample.target_smiles,
                        image_size=image_size,
                    )
                    for sample in chunk
                ]
            ).to(device)
            latents = image_vae.encode(batch, sample=False).detach().cpu().numpy().astype(np.float32)
            out.extend(latent.reshape(-1).astype(np.float32) for latent in latents)
    return np.stack(out).astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
