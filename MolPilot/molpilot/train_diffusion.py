"""Stage 3 trainer: conditional molecular latent diffusion."""

from __future__ import annotations

import argparse

import numpy as np

from .artifacts import ensure_dir, save_json, write_csv
from .autoencoder import load_autoencoder
from .condition_model import load_condition_model, predict_condition_latents
from .diffusion import DiffusionConfig, MolecularLatentDiffusion
from .stage_data import build_condition_table, load_smiles_and_pairs


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    autoencoder = load_autoencoder(args.autoencoder_dir)
    condition_model = load_condition_model(args.alignment_dir)
    _, pairs = load_smiles_and_pairs(args.data, limit=args.limit)
    raw_conditions, target_smiles, _, rows = build_condition_table(
        pairs,
        condition_dim=args.condition_dim,
        render_missing_images=args.render_missing_images,
        render_dir=str(out_dir / "rendered_inputs"),
    )
    conditions = predict_condition_latents(condition_model, raw_conditions, pairs, autoencoder)
    target_latents = autoencoder.encode_many(target_smiles)
    diffusion = MolecularLatentDiffusion(
        DiffusionConfig(
            latent_dim=target_latents.shape[1],
            condition_dim=conditions.shape[1],
            timesteps=args.timesteps,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            lr=args.lr,
            seed=args.seed,
        )
    )
    diffusion.fit_latents(target_latents, conditions, train_smiles=target_smiles, codec=autoencoder)
    diffusion.save(out_dir)
    np.save(out_dir / "aligned_conditions.npy", conditions)
    np.save(out_dir / "target_latents.npy", target_latents)
    write_csv(rows, out_dir / "tables" / "requests.csv")
    metrics = {
        "stage": "stage3_diffusion",
        "requests": float(len(rows)),
        "condition_dim": float(conditions.shape[1]),
        "latent_dim": float(target_latents.shape[1]),
        "backend": diffusion.backend,
        "final_loss": float(diffusion.history[-1].get("loss", 0.0)) if diffusion.history else 0.0,
    }
    save_json(metrics, out_dir / "metrics.json")
    print("Stage 3 conditional molecular diffusion complete")
    print(f"requests={len(rows)} backend={diffusion.backend} final_loss={metrics['final_loss']:.6f}")
    print(f"diffusion_dir={out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MolPilot conditional latent diffusion.")
    parser.add_argument("--data", default=None)
    parser.add_argument("--autoencoder-dir", default="outputs/stages/default/stage1_autoencoder")
    parser.add_argument("--alignment-dir", default="outputs/stages/default/stage2_understanding")
    parser.add_argument("--output-dir", default="outputs/stages/default/stage3_diffusion")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--condition-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--render-missing-images", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
