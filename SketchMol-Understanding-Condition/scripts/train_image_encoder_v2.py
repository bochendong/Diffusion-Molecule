#!/usr/bin/env python
"""Train image encoder v2 from exported RDKit supervision targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sketchmol_understanding_condition.image_encoder_v2 import train_image_encoder_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = train_image_encoder_v2(
        targets_npz=args.targets_npz,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        image_size=args.image_size,
        seed=args.seed,
    )
    summary["targets_npz"] = str(args.targets_npz)
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
