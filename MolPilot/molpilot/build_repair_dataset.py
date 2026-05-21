"""CLI for exporting automatic MolPilot-R repair datasets."""

from __future__ import annotations

import argparse
from collections import Counter

from .artifacts import ensure_dir, save_json, write_csv
from .data import load_smiles_csv
from .repair_dataset import build_repair_examples, examples_to_rows


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    smiles = load_smiles_csv(args.data, limit=args.limit)
    examples = build_repair_examples(
        smiles,
        corruption_types=args.repair_corruptions,
        corruptions_per_molecule=args.repair_corruptions_per_molecule,
        seed=args.seed,
    )
    rows = examples_to_rows(examples)
    write_csv(rows, out_dir / "repair_dataset.csv")
    counts = Counter(example.corruption_type for example in examples)
    split_counts = Counter(example.split for example in examples)
    metrics = {
        "molecules": float(len(smiles)),
        "examples": float(len(examples)),
        "corruptions_per_molecule": float(args.repair_corruptions_per_molecule),
        **{f"corruption_{key}": float(value) for key, value in sorted(counts.items())},
        **{f"split_{key}": float(value) for key, value in sorted(split_counts.items())},
    }
    save_json(metrics, out_dir / "repair_dataset_metrics.json")
    print(f"Wrote {len(rows)} repair examples -> {out_dir / 'repair_dataset.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build automatic corrupted-to-valid repair examples.")
    parser.add_argument("--data", default=None)
    parser.add_argument("--output-dir", default="outputs/repair_dataset")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--repair-corruptions", default=None)
    parser.add_argument("--repair-corruptions-per-molecule", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    main()
