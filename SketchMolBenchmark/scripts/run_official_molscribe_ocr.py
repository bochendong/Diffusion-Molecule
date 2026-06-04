#!/usr/bin/env python
"""Run official MolScribe OCR on a SketchMol image_path.csv file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from molscribe import MolScribe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to the MolScribe checkpoint.")
    parser.add_argument("--image-csv", required=True, help="SketchMol image_path.csv to update.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_csv = Path(args.image_csv)
    frame = pd.read_csv(image_csv)
    if "image_path" not in frame.columns:
        raise ValueError(f"{image_csv} must contain an image_path column")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = MolScribe(args.model_path, device=device)
    outputs = model.predict_image_files(
        [str(path) for path in frame["image_path"]],
        return_atoms_bonds=False,
        return_confidence=True,
    )

    frame["SMILES"] = [output.get("smiles", "") for output in outputs]
    frame["molscribe_score"] = [output.get("confidence") for output in outputs]
    frame.to_csv(image_csv, index=False)
    print(f"save to {image_csv}")
    print("done")


if __name__ == "__main__":
    main()
