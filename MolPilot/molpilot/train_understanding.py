"""Stage 2 trainer: align multimodal understanding to molecular latents."""

from __future__ import annotations

import argparse

import numpy as np

from .alignment import AlignmentConfig, UnderstandingAlignment
from .artifacts import ensure_dir, save_json, write_csv
from .autoencoder import load_autoencoder
from .condition_model import build_source_latents
from .jepa import JEPAConfig, MolecularJEPAPredictor
from .stage_data import build_condition_table, load_smiles_and_pairs


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    autoencoder = load_autoencoder(args.autoencoder_dir)
    _, pairs = load_smiles_and_pairs(args.data, limit=args.limit)
    conditions, target_smiles, _, rows = build_condition_table(
        pairs,
        condition_dim=args.condition_dim,
        render_missing_images=args.render_missing_images,
        render_dir=str(out_dir / "rendered_inputs"),
    )
    target_latents = autoencoder.encode_many(target_smiles)
    source_latents = build_source_latents(autoencoder, pairs, latent_dim=target_latents.shape[1])
    if args.model_kind == "jepa":
        model = MolecularJEPAPredictor(
            JEPAConfig(
                input_dim=conditions.shape[1],
                latent_dim=target_latents.shape[1],
                hidden_dim=args.hidden_dim,
                layers=args.layers,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                contrastive_weight=args.contrastive_weight,
                delta_weight=args.delta_weight,
                sigreg_weight=args.sigreg_weight,
                seed=args.seed,
            )
        )
        model.fit(conditions, target_latents, source_latents)
    else:
        model = UnderstandingAlignment(
            AlignmentConfig(
                input_dim=conditions.shape[1],
                latent_dim=target_latents.shape[1],
                hidden_dim=args.hidden_dim,
                layers=args.layers,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                contrastive_weight=args.contrastive_weight,
                seed=args.seed,
            )
        )
        model.fit(conditions, target_latents)
    model.save(out_dir)
    np.save(out_dir / "raw_conditions.npy", conditions)
    np.save(out_dir / "target_latents.npy", target_latents)
    np.save(out_dir / "source_latents.npy", source_latents)
    write_csv(rows, out_dir / "tables" / "requests.csv")
    metrics = {
        "stage": "stage2_understanding",
        "model_kind": args.model_kind,
        "requests": float(len(rows)),
        "input_dim": float(conditions.shape[1]),
        "latent_dim": float(target_latents.shape[1]),
        "backend": model.backend,
        "final_loss": float(model.history[-1].get("loss", 0.0)) if model.history else 0.0,
    }
    save_json(metrics, out_dir / "metrics.json")
    print("Stage 2 understanding alignment complete")
    print(f"requests={len(rows)} backend={model.backend} final_loss={metrics['final_loss']:.6f}")
    print(f"alignment_dir={out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MolPilot understanding alignment.")
    parser.add_argument("--data", default=None)
    parser.add_argument("--autoencoder-dir", default="outputs/stages/default/stage1_autoencoder")
    parser.add_argument("--output-dir", default="outputs/stages/default/stage2_understanding")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--condition-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--delta-weight", type=float, default=0.25)
    parser.add_argument("--sigreg-weight", type=float, default=0.01)
    parser.add_argument("--model-kind", choices=["jepa", "alignment"], default="jepa")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--render-missing-images", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
