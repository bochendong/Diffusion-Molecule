"""Stage 1 trainer: molecular latent autoencoder."""

from __future__ import annotations

import argparse

from .artifacts import ensure_dir, save_json, write_csv
from .autoencoder import AutoencoderConfig, MolecularAutoencoder
from .data import load_smiles_csv
from .sequence_autoencoder import MolecularSequenceAutoencoder, SequenceAutoencoderConfig


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    smiles = load_smiles_csv(args.data, limit=args.limit)
    if args.codec == "sequence":
        model = MolecularSequenceAutoencoder(
            SequenceAutoencoderConfig(
                representation=args.representation,
                feature_dim=args.feature_dim,
                latent_dim=args.latent_dim,
                embedding_dim=args.embedding_dim,
                hidden_dim=args.hidden_dim,
                layers=args.layers,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                max_length=args.max_length,
                seed=args.seed,
            )
        )
    else:
        model = MolecularAutoencoder(
            AutoencoderConfig(
                feature_dim=args.feature_dim,
                latent_dim=args.latent_dim,
                hidden_dim=args.hidden_dim,
                layers=args.layers,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
            )
        )
    model.fit(smiles)
    model.save(out_dir)
    rows = [{"idx": idx, "smiles": smi} for idx, smi in enumerate(smiles)]
    write_csv(rows, out_dir / "tables" / "molecules.csv")
    metrics = {
        "stage": "stage1_autoencoder",
        "molecules": float(len(smiles)),
        "latent_dim": float(args.latent_dim),
        "feature_dim": float(args.feature_dim),
        "codec": args.codec,
        "backend": model.backend,
        "final_loss": float(model.history[-1].get("loss", 0.0)) if model.history else 0.0,
    }
    save_json(metrics, out_dir / "metrics.json")
    print("Stage 1 molecular autoencoder complete")
    print(f"molecules={len(smiles)} backend={model.backend} final_loss={metrics['final_loss']:.6f}")
    print(f"autoencoder_dir={out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MolPilot molecular autoencoder.")
    parser.add_argument("--data", default=None)
    parser.add_argument("--output-dir", default="outputs/stages/default/stage1_autoencoder")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--codec", choices=["feature", "sequence"], default="sequence")
    parser.add_argument("--representation", choices=["auto", "selfies", "smiles"], default="auto")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    main()
